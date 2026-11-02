"""Cooperative incremental Newton stages, ported to MuJoCo Warp 3.16.

Derived from mujoco_warp solver.py, Copyright 2025 The Newton Developers,
licensed under the Apache License, Version 2.0.
Local changes fuse accepted constraint state, gradient and Hessian work.
The separate multiply/add in rounded_step preserves the pinned solver's
atomic-add rounding; a contracted FMA can change short-budget convergence.
"""

from functools import cache

import warp as wp

from instantmimic.utils.mjw_kernels.private import block_cholesky, math, solver, types
from instantmimic.utils.mjw_kernels.tree_fusion import sync


@wp.func_native(snippet="return __fadd_rn(a, __fmul_rn(b, c));")
def rounded_step(a: float, b: float, c: float) -> float: ...


@wp.func_native(snippet="return (int)__ballot_sync(0xFFFFFFFFu, pred);")
def warp_ballot(pred: int) -> int: ...


@wp.func_native(snippet="return __popc((unsigned)mask & ((1u << lane) - 1u));")
def bits_below(mask: int, lane: int) -> int: ...


@wp.func_native(snippet="return __popc((unsigned)mask);")
def bit_count(mask: int) -> int: ...


# Warp's whole-block tile_reduce of one value per thread: a shuffle-down tree
# within each warp, then thread 0 adds the later warps' partials in order. The
# line search reduces over the first warp only, where every later warp's
# partial would be +0; called from that warp, these reproduce the block sum
# bitwise and broadcast it to its lanes with no shared memory or barrier.
_WARP_SUM_TAIL = """
for (int offset = 16; offset > 0; offset /= 2) {{
    auto other = s;
    {shuffle}
    s = wp::add(s, other);
}}
{broadcast}
const auto zero = decltype(s)();
for (int w = 1; w < (WP_TILE_BLOCK_DIM + 31) / 32; ++w)
    s = wp::add(s, zero);
return s;
"""


def _warp_sum_snippet(elements: str) -> str:
    shuffle = f"for (int i = 0; i < {elements}; ++i) other.c[i] = __shfl_down_sync(0xFFFFFFFFu, s.c[i], offset, 32);"
    broadcast = f"for (int i = 0; i < {elements}; ++i) s.c[i] = __shfl_sync(0xFFFFFFFFu, s.c[i], 0, 32);"
    return "auto s = v;" + _WARP_SUM_TAIL.format(shuffle=shuffle, broadcast=broadcast)


_MAT_SNIPPET = "auto s = v;" + _WARP_SUM_TAIL.format(
    shuffle="for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) "
    "other.data[i][j] = __shfl_down_sync(0xFFFFFFFFu, s.data[i][j], offset, 32);",
    broadcast="for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) "
    "s.data[i][j] = __shfl_sync(0xFFFFFFFFu, s.data[i][j], 0, 32);",
)


@wp.func_native(snippet=_warp_sum_snippet("2"))
def block_sum_vec2(v: wp.vec2) -> wp.vec2: ...


@wp.func_native(snippet=_warp_sum_snippet("3"))
def block_sum_vec3(v: wp.vec3) -> wp.vec3: ...


@wp.func_native(snippet=_MAT_SNIPPET)
def block_sum_mat33(v: wp.mat33) -> wp.mat33: ...


@cache
def ls_factory(ls_iterations: int, warn_overflow: int, reduction_width: int):
    """Dense pyramidal line search with a fixed logical reduction width."""
    if reduction_width != 32:
        raise ValueError("the line search reduces over exactly one warp of rows")
    LS_ITERATIONS = ls_iterations
    REDUCTION_WIDTH = reduction_width

    @wp.func_native(snippet="WP_TILE_SYNC();")
    def _syncthreads():
        pass

    _compute_efc_eval_pt = solver._compute_efc_eval_pt_pyramidal
    _compute_efc_eval_pt_alpha_zero = solver._compute_efc_eval_pt_alpha_zero_pyramidal
    _compute_efc_eval_pt_3alphas = solver._compute_efc_eval_pt_3alphas_pyramidal

    @wp.func
    def kernel(
        nv: int,
        opt_tolerance: wp.array[float],
        opt_ls_tolerance: wp.array[float],
        opt_impratio_invsqrt: wp.array[float],
        stat_meaninertia: wp.array[float],
        ne_in: wp.array[int],
        nf_in: wp.array[int],
        nefc_in: wp.array[int],
        qfrc_smooth_in: wp.array2d[float],
        contact_friction_in: wp.array[types.vec5],
        contact_dim_in: wp.array[int],
        contact_efc_address_in: wp.array2d[int],
        efc_type_in: wp.array2d[int],
        efc_id_in: wp.array2d[int],
        efc_J_rownnz_in: wp.array2d[int],
        efc_J_rowadr_in: wp.array2d[int],
        efc_J_colind_in: wp.array3d[int],
        efc_J_in: wp.array3d[float],
        efc_D_in: wp.array2d[float],
        efc_frictionloss_in: wp.array2d[float],
        njmax_in: int,
        nacon_in: wp.array[int],
        ctx_search_unchanged_in: wp.array[bool],
        ctx_Jaref_in: wp.array2d[float],
        ctx_search_in: wp.array2d[float],
        ctx_search_dot_in: wp.array[float],
        ctx_mv_in: wp.array2d[float],
        ctx_jv_in: wp.array2d[float],
        ctx_quad_in: wp.array2d[wp.vec3],
        ctx_done_in: wp.array[bool],
        qacc_out: wp.array2d[float],
        efc_force_out: wp.array2d[float],
        efc_state_out: wp.array2d[int],
        efc_Ma_out: wp.array2d[float],
        ctx_Jaref_out: wp.array2d[float],
        ctx_jv_out: wp.array2d[float],
        ctx_quad_out: wp.array2d[wp.vec3],
        ctx_improvement_out: wp.array[float],
        ctx_alpha_out: wp.array[float],
        ctx_ls_exhausted_out: wp.array[bool],
        quad_changed_ids_out: wp.array2d[int],
        quad_changed_count_out: wp.array[int],
        state_changed_count_out: wp.array[int],
        overflow_out: wp.array[int],
        world_in: int,
        lane_in: int,
    ):
        worldid, tid = (world_in, lane_in)
        reduction_lane = tid
        if tid == 0:
            quad_changed_count_out[worldid] = 0
            state_changed_count_out[worldid] = 0
        _syncthreads()
        if ctx_done_in[worldid]:
            return
        ne = ne_in[worldid]
        nf = nf_in[worldid]
        nefc = wp.min(njmax_in, nefc_in[worldid])
        recompute_jv = True
        recompute_jv = not ctx_search_unchanged_in[worldid]
        if recompute_jv:
            # Each row's Jv is its own sum: every lane of the block takes rows.
            for efcid in range(tid, nefc, wp.block_dim()):
                jv = float(0.0)
                if nv > 50:
                    # Match upstream's 20-DOF partial sums above its fused-Jv
                    # threshold. Preserve separate rounding at chunk boundaries.
                    for start in range(0, nv, 20):
                        partial = float(0.0)
                        for i in range(start, wp.min(start + 20, nv)):
                            partial += (
                                efc_J_in[worldid, efcid, i] * ctx_search_in[worldid, i]
                            )
                        jv = rounded_step(jv, 1.0, partial)
                else:
                    for i in range(nv):
                        jv += efc_J_in[worldid, efcid, i] * ctx_search_in[worldid, i]
                ctx_jv_out[worldid, efcid] = jv
        _syncthreads()
        # The first warp alone runs the scalar search (its rows are the reduction
        # rows), then shares the step with the block through one barrier.
        alpha = float(0.0)
        improvement = float(0.0)
        noise_floor = float(0.0)
        ls_converged = bool(False)
        if tid < 32:
            tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
            ls_tolerance = opt_ls_tolerance[worldid % opt_ls_tolerance.shape[0]]
            snorm = wp.sqrt(ctx_search_dot_in[worldid])
            meaninertia = stat_meaninertia[worldid % stat_meaninertia.shape[0]]
            scale = meaninertia * wp.float(nv)
            gtol = wp.max(tolerance * ls_tolerance * snorm * scale, 1e-06)
            # Upstream 3.16 scalar search; retain one-warp reductions and fused updates.
            local_gauss = wp.vec2(0.0)
            for dofid in range(reduction_lane, nv, REDUCTION_WIDTH):
                search = ctx_search_in[worldid, dofid]
                local_gauss += wp.vec2(
                    search
                    * (efc_Ma_out[worldid, dofid] - qfrc_smooth_in[worldid, dofid]),
                    0.5 * search * ctx_mv_in[worldid, dofid],
                )
            for efcid in range(reduction_lane, ne, REDUCTION_WIDTH):
                jv = ctx_jv_in[worldid, efcid]
                jvD = jv * efc_D_in[worldid, efcid]
                local_gauss += wp.vec2(
                    jvD * ctx_Jaref_in[worldid, efcid], 0.5 * jv * jvD
                )
            gauss_sum = block_sum_vec2(local_gauss)
            gauss_reduced = gauss_sum
            ctx_quad_gauss = wp.vec3(0.0, gauss_reduced[0], gauss_reduced[1])
            local_p0 = wp.vec3(0.0)
            for efcid in range(ne + reduction_lane, nefc, REDUCTION_WIDTH):
                local_p0 += _compute_efc_eval_pt_alpha_zero(
                    efcid,
                    ne,
                    nf,
                    efc_D_in[worldid, efcid],
                    efc_frictionloss_in[worldid],
                    ctx_Jaref_in[worldid, efcid],
                    ctx_jv_in[worldid, efcid],
                )
            p0_sum = block_sum_vec3(local_p0)
            p0 = (
                wp.vec3(ctx_quad_gauss[0], ctx_quad_gauss[1], 2.0 * ctx_quad_gauss[2])
                + p0_sum
            )
            p0_delta = wp.vec3(0.0, p0[1], p0[2])
            rows = p0_sum
            q1_abs = wp.sqrt(
                2.0 * wp.max(rows[0], 0.0) * wp.max(rows[2], 0.0)
            ) + wp.abs(ctx_quad_gauss[1])
            gtol_accept = wp.max(gtol, solver._ALPHA_NOISE_EPS * q1_abs)
            noise_floor = float(0.0)
            noise_floor = solver._ALPHA_NOISE_EPS * wp.max(
                1.0, math.safe_div(q1_abs, p0[2])
            )
            alpha = float(0.0)
            improvement = float(0.0)
            ls_converged = wp.abs(p0[1]) < solver._ALPHA_NOISE_EPS * q1_abs
            lo_alpha_in = -math.safe_div(p0[1], p0[2])
            lo_in = wp.vec3(0.0)
            if not ls_converged:
                local_lo_in = wp.vec3(0.0)
                for efcid in range(ne + reduction_lane, nefc, REDUCTION_WIDTH):
                    local_lo_in += _compute_efc_eval_pt(
                        efcid,
                        lo_alpha_in,
                        ne,
                        nf,
                        efc_D_in[worldid, efcid],
                        efc_frictionloss_in[worldid],
                        ctx_Jaref_in[worldid, efcid],
                        ctx_jv_in[worldid, efcid],
                    )
                lo_in_sum = block_sum_vec3(local_lo_in)
                lo_in = solver._eval_pt(ctx_quad_gauss, lo_alpha_in) + lo_in_sum
                if wp.abs(lo_in[1]) < gtol_accept and lo_in[0] < 0.0:
                    ls_converged = True
                    alpha = lo_alpha_in
                    improvement = -lo_in[0]
            if not ls_converged:
                lo_less = lo_in[1] < p0[1]
                lo = wp.where(lo_less, lo_in, p0_delta)
                lo_alpha = wp.where(lo_less, lo_alpha_in, 0.0)
                hi = wp.where(lo_less, p0_delta, lo_in)
                hi_alpha = wp.where(lo_less, 0.0, lo_alpha_in)
                for _ in range(LS_ITERATIONS):
                    lo_next_alpha = lo_alpha - math.safe_div(lo[1], lo[2])
                    hi_next_alpha = hi_alpha - math.safe_div(hi[1], hi[2])
                    mid_alpha = 0.5 * (lo_alpha + hi_alpha)
                    local_lo = wp.vec3(0.0)
                    local_hi = wp.vec3(0.0)
                    local_mid = wp.vec3(0.0)
                    for efcid in range(ne + reduction_lane, nefc, REDUCTION_WIDTH):
                        r_lo, r_hi, r_mid = _compute_efc_eval_pt_3alphas(
                            efcid,
                            lo_next_alpha,
                            hi_next_alpha,
                            mid_alpha,
                            ne,
                            nf,
                            efc_D_in[worldid, efcid],
                            efc_frictionloss_in[worldid],
                            ctx_Jaref_in[worldid, efcid],
                            ctx_jv_in[worldid, efcid],
                        )
                        local_lo += r_lo
                        local_hi += r_hi
                        local_mid += r_mid
                    local_combined = wp.mat33(
                        local_lo[0],
                        local_hi[0],
                        local_mid[0],
                        local_lo[1],
                        local_hi[1],
                        local_mid[1],
                        local_lo[2],
                        local_hi[2],
                        local_mid[2],
                    )
                    result = block_sum_mat33(local_combined)
                    gauss_lo, gauss_hi, gauss_mid = solver._eval_pt_3alphas(
                        ctx_quad_gauss, lo_next_alpha, hi_next_alpha, mid_alpha
                    )
                    lo_next = gauss_lo + wp.vec3(
                        result[0, 0], result[1, 0], result[2, 0]
                    )
                    hi_next = gauss_hi + wp.vec3(
                        result[0, 1], result[1, 1], result[2, 1]
                    )
                    mid = gauss_mid + wp.vec3(result[0, 2], result[1, 2], result[2, 2])
                    conv_lo = wp.abs(lo_next[1]) < gtol_accept and lo_next[0] < 0.0
                    conv_hi = wp.abs(hi_next[1]) < gtol_accept and hi_next[0] < 0.0
                    conv_mid = wp.abs(mid[1]) < gtol_accept and mid[0] < 0.0
                    converged = conv_lo or conv_hi or conv_mid
                    swap_lo = False
                    swap_hi = False
                    if converged:
                        conv_pt = wp.vec3(types.MJ_MAXVAL, 0.0, 0.0)
                        conv_alpha = float(0.0)
                        if conv_lo and lo_next[0] < conv_pt[0]:
                            conv_pt = lo_next
                            conv_alpha = lo_next_alpha
                        if conv_hi and hi_next[0] < conv_pt[0]:
                            conv_pt = hi_next
                            conv_alpha = hi_next_alpha
                        if conv_mid and mid[0] < conv_pt[0]:
                            conv_pt = mid
                            conv_alpha = mid_alpha
                        lo = conv_pt
                        lo_alpha = conv_alpha
                        hi = conv_pt
                        hi_alpha = conv_alpha
                    else:
                        swap_lo_lo_next = solver._in_bracket(lo, lo_next)
                        lo = wp.where(swap_lo_lo_next, lo_next, lo)
                        lo_alpha = wp.where(swap_lo_lo_next, lo_next_alpha, lo_alpha)
                        swap_lo_mid = solver._in_bracket(lo, mid)
                        lo = wp.where(swap_lo_mid, mid, lo)
                        lo_alpha = wp.where(swap_lo_mid, mid_alpha, lo_alpha)
                        swap_lo_hi_next = solver._in_bracket(lo, hi_next)
                        lo = wp.where(swap_lo_hi_next, hi_next, lo)
                        lo_alpha = wp.where(swap_lo_hi_next, hi_next_alpha, lo_alpha)
                        swap_lo = swap_lo_lo_next or swap_lo_mid or swap_lo_hi_next
                        swap_hi_hi_next = solver._in_bracket(hi, hi_next) or (
                            hi[1] < 0.0 and hi_next[1] > 0.0
                        )
                        hi = wp.where(swap_hi_hi_next, hi_next, hi)
                        hi_alpha = wp.where(swap_hi_hi_next, hi_next_alpha, hi_alpha)
                        swap_hi_mid = solver._in_bracket(hi, mid)
                        hi = wp.where(swap_hi_mid, mid, hi)
                        hi_alpha = wp.where(swap_hi_mid, mid_alpha, hi_alpha)
                        swap_hi_lo_next = solver._in_bracket(hi, lo_next)
                        hi = wp.where(swap_hi_lo_next, lo_next, hi)
                        hi_alpha = wp.where(swap_hi_lo_next, lo_next_alpha, hi_alpha)
                        swap_hi = swap_hi_hi_next or swap_hi_mid or swap_hi_lo_next
                    ls_done = (
                        converged
                        or (not swap_lo and (not swap_hi))
                        or (lo[0] < 0.0 and lo[1] < 0.0 and (lo[1] > -gtol))
                        or (hi[0] < 0.0 and hi[1] > 0.0 and (hi[1] < gtol))
                    )
                    improved = lo[0] < 0.0 or hi[0] < 0.0
                    lo_better = lo[0] < hi[0]
                    best_alpha = wp.where(lo_better, lo_alpha, hi_alpha)
                    best_delta = wp.where(lo_better, lo[0], hi[0])
                    alpha = wp.where(improved, best_alpha, alpha)
                    improvement = wp.where(improved, -best_delta, improvement)
                    if ls_done:
                        ls_converged = True
                        break
            if tid == 0:
                ctx_alpha_out[worldid] = alpha
        _syncthreads()
        alpha = ctx_alpha_out[worldid]
        # Per-dof and per-row updates: every lane of the block.
        for dofid in range(tid, nv, wp.block_dim()):
            qacc_out[worldid, dofid] += alpha * ctx_search_in[worldid, dofid]
            efc_Ma_out[worldid, dofid] += alpha * ctx_mv_in[worldid, dofid]
        for efcid in range(tid, nefc, wp.block_dim()):
            new_jaref = rounded_step(
                ctx_Jaref_in[worldid, efcid], alpha, ctx_jv_in[worldid, efcid]
            )
            ctx_Jaref_out[worldid, efcid] = new_jaref
            old_state = efc_state_out[worldid, efcid]
            is_equality = efcid < ne
            is_friction = not is_equality and efcid < ne + nf
            frictionloss = efc_frictionloss_in[worldid, efcid] if is_friction else 0.0
            res = solver._eval_constraint(
                is_equality,
                is_friction,
                False,
                new_jaref,
                efc_D_in[worldid, efcid],
                frictionloss,
                efcid,
                -1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            new_state = int(res[1])
            efc_force_out[worldid, efcid] = res[0]
            efc_state_out[worldid, efcid] = new_state
            old_quad = old_state == types.ConstraintState.QUADRATIC.value
            new_quad = new_state == types.ConstraintState.QUADRATIC.value
            # Flag the row; the first warp lists flagged rows below.
            quad_changed_ids_out[worldid, efcid] = int(old_quad != new_quad)
            if old_state != new_state:
                wp.atomic_add(state_changed_count_out, worldid, 1)
        _syncthreads()
        # List the rows whose quadratic state changed in ascending order, so the
        # Hessian update adds them in a fixed order whatever the lane schedule.
        # In place: a row's slot never lies past the row itself, and every row
        # of a 32-row chunk is read before the chunk's ballot.
        if tid < 32:
            count = int(0)
            for base in range(0, nefc, 32):
                efcid = base + tid
                changed = int(0)
                if efcid < nefc:
                    changed = quad_changed_ids_out[worldid, efcid]
                mask = warp_ballot(changed)
                if changed != 0:
                    quad_changed_ids_out[worldid, count + bits_below(mask, tid)] = efcid
                count += bit_count(mask)
            if tid == 0:
                quad_changed_count_out[worldid] = count
        if tid == 0:
            ctx_improvement_out[worldid] = improvement
            ctx_alpha_out[worldid] = alpha
            ls_exhausted = wp.abs(alpha) < noise_floor
            ctx_ls_exhausted_out[worldid] = ls_exhausted
            if ls_exhausted:
                wp.atomic_add(state_changed_count_out, worldid, 1)
            if not ls_converged:
                if wp.static(bool(warn_overflow & solver.OverflowType.LS_ITERATIONS)):
                    wp.printf(
                        "linesearch iterations limit reached - please increase ls_iterations to %u\n",
                        wp.static(LS_ITERATIONS),
                    )
                wp.atomic_or(
                    overflow_out, worldid, wp.static(solver.OverflowType.LS_ITERATIONS)
                )

    return kernel


@wp.func_native(snippet="return fmaf(a, b, c);")
def fma(a: float, b: float, c: float) -> float: ...


@wp.func
def hessian_step(
    J: wp.array3d[float],
    D: wp.array2d[float],
    state: wp.array2d[int],
    w: int,
    r: int,
    row: int,
    col: int,
    delta: float,
) -> float:
    """delta plus row r's change of H[row, col], +-D a b by the row's new state, as
    the contracted ``delta += D * a * b`` (``-=``) of a full sweep; unchanged when
    the row does not reach both dofs."""
    a = J[w, r, row]
    b = J[w, r, col]
    da = D[w, r] * a
    if state[w, r] != types.ConstraintState.QUADRATIC.value:
        da = -da
    return wp.where(a != 0.0 and b != 0.0, fma(da, b, delta), delta)


@cache
def augmented_factor_solve_factory(block_size: int, pad: int, nv: int):
    """``create_blocked_cholesky_augmented_factorize_solve_newton_func`` with the
    same tile operations in the same order, its block loads moved into helper
    functions. Warp sums every shared tile a function owns but takes only the
    largest helper's, so the factor needs 4.4 KB of shared memory instead of
    7.5 KB, and more Newton blocks fit on an SM."""
    B = block_size
    border = B - 1
    decrement = block_cholesky._create_newton_decrement_func(pad, nv)

    @wp.func
    def gram_update(U: wp.array2d[float], j: int, k: int, A_kk: wp.tile[float, B, B]):
        U_block = wp.tile_load(
            U,
            shape=(B, B),
            offset=(j, k),
            storage="shared",
            bounds_check=False,
            aligned=True,
        )
        wp.tile_matmul(wp.tile_transpose(U_block), U_block, A_kk, alpha=-1.0)

    @wp.func
    def panel_update(
        U: wp.array2d[float], j: int, k: int, i: int, A_ki: wp.tile[float, B, B]
    ):
        U_jk = wp.tile_load(
            U,
            shape=(B, B),
            offset=(j, k),
            storage="shared",
            bounds_check=False,
            aligned=True,
        )
        U_ji = wp.tile_load(
            U,
            shape=(B, B),
            offset=(j, i),
            storage="shared",
            bounds_check=False,
            aligned=True,
        )
        wp.tile_matmul(wp.tile_transpose(U_jk), U_ji, A_ki, alpha=-1.0)

    @wp.func
    def back_update(U: wp.array2d[float], i: int, j: int, rhs: wp.tile[float, pad, 1]):
        U_tile = wp.tile_load(
            U,
            shape=(B, B),
            offset=(i, j),
            storage="shared",
            bounds_check=False,
            aligned=True,
        )
        x = wp.tile_view(rhs, shape=(B, 1), offset=(j, 0))
        tmp = wp.tile_view(rhs, shape=(B, 1), offset=(i, 0))
        wp.tile_matmul(U_tile, x, tmp, alpha=-1.0)

    @wp.func
    def back_solve(U: wp.array2d[float], i: int, rhs: wp.tile[float, pad, 1]):
        U_tile = wp.tile_load(
            U,
            shape=(B, B),
            offset=(i, i),
            storage="shared",
            bounds_check=False,
            aligned=True,
        )
        tmp = wp.tile_view(rhs, shape=(B, 1), offset=(i, 0))
        wp.tile_upper_solve_inplace(U_tile, tmp)

    @wp.func
    def factor_solve(
        A: wp.array2d[float],
        b: wp.array2d[float],
        matrix_size: int,
        U_out: wp.array2d[float],
        result_out: wp.array2d[float],
    ):
        rhs_tile = wp.tile_zeros(shape=(pad, 1), dtype=float, storage="shared")
        for k in range(0, matrix_size, B):
            end = k + B
            input_rhs = wp.tile_load(
                b, shape=(B, 1), offset=(k, 0), storage="shared", bounds_check=False
            )
            A_kk_tile = wp.tile_load(
                A,
                shape=(B, B),
                offset=(k, k),
                storage="shared",
                bounds_check=False,
                aligned=True,
            )
            input_diagonal_rhs = wp.tile_view(
                input_rhs, shape=(border, 1), offset=(0, 0)
            )
            if end == matrix_size:
                wp.tile_assign(A_kk_tile, input_diagonal_rhs, offset=(0, border))
                # Keep the augmented rank-one correction below float32 precision.
                A_kk_tile[border, border] = 1.0e30
            for j in range(0, k, B):
                gram_update(U_out, j, k, A_kk_tile)
            wp.tile_cholesky_inplace(A_kk_tile, fill_mode="upper")
            diagonal_border = wp.tile_view(
                A_kk_tile, shape=(border, 1), offset=(0, border)
            )
            if end == matrix_size:
                wp.tile_assign(rhs_tile, diagonal_border, offset=(k, 0))
            wp.tile_store(
                U_out, A_kk_tile, offset=(k, k), bounds_check=False, aligned=True
            )
            for i in range(end, matrix_size, B):
                A_ki_tile = wp.tile_load(
                    A,
                    shape=(B, B),
                    offset=(k, i),
                    storage="shared",
                    bounds_check=False,
                    aligned=True,
                )
                if i + B == matrix_size:
                    wp.tile_assign(A_ki_tile, input_rhs, offset=(0, border))
                for j in range(0, k, B):
                    panel_update(U_out, j, k, i, A_ki_tile)
                wp.tile_lower_solve_inplace(wp.tile_transpose(A_kk_tile), A_ki_tile)
                panel_border = wp.tile_view(A_ki_tile, shape=(B, 1), offset=(0, border))
                if i + B == matrix_size:
                    wp.tile_assign(rhs_tile, panel_border, offset=(k, 0))
                wp.tile_store(
                    U_out, A_ki_tile, offset=(k, i), bounds_check=False, aligned=True
                )
        for i in range(matrix_size - B, -1, -B):
            for j in range(i + B, matrix_size, B):
                back_update(U_out, i, j, rhs_tile)
            back_solve(U_out, i, rhs_tile)
        return decrement(rhs_tile, b, result_out)

    return factor_solve


@cache
def factor_factory(nv: int, pad: int, sparse_plan=None):
    """Incremental H update plus factor/solve; ``sparse_plan`` swaps the dense
    blocked Cholesky for the static-pattern sparse L'DL (sparse_hessian.py)."""
    SPARSE = sparse_plan is not None
    if SPARSE:
        from instantmimic.utils.mjw_kernels import sparse_hessian

        native = sparse_hessian.native_func(sparse_plan)
        factor = solve = None
    else:
        native = None
        factor = augmented_factor_solve_factory(16, pad, nv)
        solve = solver.create_blocked_cholesky_solve_newton_func(16, pad, nv)

    @wp.func
    def kernel(
        J: wp.array3d[float],
        D: wp.array2d[float],
        state: wp.array2d[int],
        ids: wp.array2d[int],
        changes: wp.array[int],
        state_changes: wp.array[int],
        done: wp.array[bool],
        grad: wp.array3d[float],
        H: wp.array3d[float],
        U: wp.array3d[float],
        search: wp.array3d[float],
        search_dot: wp.array[float],
        decrement: wp.array[float],
        dof_union: wp.array2d[int],
        world_in: int,
        lane_in: int,
    ):
        w, lane = (world_in, lane_in)
        if done[w] or state_changes[w] == 0:
            return
        count = changes[w]
        if count > 0:
            # Only H entries whose row and column dofs some changed row touches can
            # change. Collect those dofs in ascending order (dof_union[w, :n], n in
            # dof_union[w, nv]) and visit their pairs only; each entry sums the
            # changed rows in the same order as a full sweep, so H is identical.
            for dof in range(lane, nv, wp.block_dim()):
                hit = int(0)
                for t in range(count):
                    if J[w, ids[w, t], dof] != 0.0:
                        hit = 1
                        break
                dof_union[w, dof] = hit
            sync()
            if lane < 32:
                # Compact in place with ballots, as the changed-row list.
                n = int(0)
                for base in range(0, nv, 32):
                    dof = base + lane
                    hit = int(0)
                    if dof < nv:
                        hit = dof_union[w, dof]
                    mask = warp_ballot(hit)
                    if hit != 0:
                        dof_union[w, n + bits_below(mask, lane)] = dof
                    n += bit_count(mask)
                if lane == 0:
                    dof_union[w, nv] = n
            sync()
            n = dof_union[w, nv]
            if wp.static(SPARSE):
                n = 0  # the sparse path updates its packed H in the native call
            for e in range(lane, n * (n + 1) // 2, wp.block_dim()):
                cj = (int(wp.sqrt(float(1 + 8 * e))) - 1) // 2
                row = dof_union[w, e - cj * (cj + 1) // 2]
                col = dof_union[w, cj]
                # Four rows per step so their loads issue together; the sum keeps
                # the row order (and rounding) of a full sweep.
                delta = float(0.0)
                t = int(0)
                while t + 4 <= count:
                    delta = hessian_step(J, D, state, w, ids[w, t], row, col, delta)
                    delta = hessian_step(J, D, state, w, ids[w, t + 1], row, col, delta)
                    delta = hessian_step(J, D, state, w, ids[w, t + 2], row, col, delta)
                    delta = hessian_step(J, D, state, w, ids[w, t + 3], row, col, delta)
                    t += 4
                while t < count:
                    delta = hessian_step(J, D, state, w, ids[w, t], row, col, delta)
                    t += 1
                if delta != 0.0:
                    H[w, row, col] = H[w, row, col] + delta
        sync()
        sums = wp.vec2(0.0)
        if wp.static(SPARSE):
            mode = int(1)
            if count > 0:
                mode = 0
            sums = native(
                J,
                D,
                state,
                ids,
                count,
                dof_union,
                w,
                H[w],
                grad[w],
                U[w],
                search[w],
                lane,
                mode,
            )
        elif wp.static(nv <= 32):
            # Small native contexts have no persistent hfactor. Keep their
            # exact tile Cholesky solve instead of the augmented blocked path.
            mat = wp.tile_load(H[w], shape=(nv, nv))
            wp.tile_cholesky_inplace(mat, fill_mode="upper")
            inputs = wp.tile_load(grad[w], shape=(nv, 1))
            outputs = wp.tile_cholesky_solve(mat, inputs, fill_mode="upper")
            sums = wp.tile_reduce(
                wp.add, wp.tile_map(solver.solve_search_sums, inputs, outputs)
            )[0]
            wp.tile_store(search[w], wp.tile_map(wp.mul, outputs, -1.0))
        elif count > 0:
            sums = factor(H[w], grad[w], pad, U[w], search[w])
        else:
            sums = solve(U[w], grad[w], pad, search[w])
        if lane == 0:
            search_dot[w] = sums[0]
            decrement[w] = sums[1]

    return kernel


@wp.func
def gradient_function(
    nv: int,
    nefc_in: wp.array[int],
    efc_J_in: wp.array3d[float],
    efc_force_in: wp.array2d[float],
    njmax: int,
    qfrc_smooth_in: wp.array2d[float],
    efc_Ma_in: wp.array2d[float],
    state_changed_count_in: wp.array[int],
    ctx_alpha_in: wp.array[float],
    ctx_done_in: wp.array[bool],
    qfrc_constraint_out: wp.array2d[float],
    ctx_grad_out: wp.array2d[float],
    ctx_grad_dot_out: wp.array[float],
    ctx_newton_decrement_out: wp.array[float],
    ctx_grad_scale_out: wp.array[float],
    ctx_search_unchanged_out: wp.array[bool],
    world_in: int,
    lane_in: int,
):
    worldid, rank = (world_in, lane_in)
    done = ctx_done_in[worldid]
    changed = state_changed_count_in[worldid] != 0
    if rank == 0:
        ctx_search_unchanged_out[worldid] = done or not changed
    if done:
        return
    if not changed:
        if rank == 0:
            sigma = ctx_grad_scale_out[worldid]
            new_sigma = sigma - ctx_alpha_in[worldid]
            ratio = float(0.0)
            if sigma != 0.0:
                ratio = new_sigma / sigma
            ratio_sq = ratio * ratio
            ctx_grad_dot_out[worldid] *= ratio_sq
            ctx_newton_decrement_out[worldid] *= ratio_sq
            ctx_grad_scale_out[worldid] = new_sigma
        return
    grad_sq = float(0.0)
    for dof in range(rank, nv, wp.block_dim()):
        sum_qfrc = float(0.0)
        for efcid in range(min(njmax, nefc_in[worldid])):
            sum_qfrc += efc_J_in[worldid, efcid, dof] * efc_force_in[worldid, efcid]
        qfrc_constraint_out[worldid, dof] = sum_qfrc
        grad = efc_Ma_in[worldid, dof] - qfrc_smooth_in[worldid, dof] - sum_qfrc
        ctx_grad_out[worldid, dof] = grad
        grad_sq += grad * grad
    grad_dot = wp.tile_sum(wp.tile(grad_sq))
    wp.tile_store(ctx_grad_dot_out, grad_dot, offset=worldid)
    if rank == 0:
        ctx_newton_decrement_out[worldid] = 0.0
        ctx_grad_scale_out[worldid] = 1.0
