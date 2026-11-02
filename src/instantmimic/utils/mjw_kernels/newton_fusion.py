"""Per-world Newton iteration with persistent setup and explicit eligibility.

Derived from mujoco_warp solver.py, Copyright 2025 The Newton Developers,
licensed under the Apache License, Version 2.0.
Local changes hoist the upstream per-iteration setup into a persistent per-model
allocation and gate the fused path on structural eligibility.

The stock initialization and final force recovery remain in mjw_step_optimized.
Models padded to 64 or more DOFs use a 64-thread block while line search retains
the model's reduction order.
"""

from functools import cache
from math import ceil

import warp as wp

from instantmimic.utils.mjw_kernels.newton_stages import (
    factor_factory,
    gradient_function,
    ls_factory,
)
from instantmimic.utils.mjw_kernels.private import solver, types
from instantmimic.utils.mjw_kernels.tree_fusion import sync
from instantmimic.utils.warp_kernels import expose_kernel_name

# Upper bound on the CTAs-per-SM the kernel asks the compiler to fit. Measured
# on the SNU 56-DOF model at 1,024/2,048/4,096 worlds: 12 is the optimum at both
# world counts that exceed one wave, 16 is within half a point, and 24 is a
# regression because the resulting 42-register budget spills. The optimum is flat
# between 12 and 16, so this stays a constant rather than a config surface that
# would invite unmeasured values.
_NEWTON_MAX_BLOCKS_PER_SM = 12


def newton_min_blocks_per_sm(nworld: int, sm_count: int) -> int:
    """CTAs per SM to demand through CUDA ``__launch_bounds__``, or 0 for none.

    ``__launch_bounds__(width, n)`` caps registers per thread at
    ``65536 / (width * n)``, which raises how many CTAs can stay resident. That
    only buys anything when the grid has more CTAs than one register-limited
    wave holds, so the request tracks what the grid actually supplies: at
    1,024 worlds on a 170-SM device this returns 6, whose 170-register budget
    does not constrain the compiler's natural 168, and the cap self-disables
    exactly where it measured useless.
    """
    if nworld <= 0 or sm_count <= 0:
        return 0
    return max(1, min(nworld // sm_count, _NEWTON_MAX_BLOCKS_PER_SM))


def _device_sm_count(m) -> int:
    device = getattr(getattr(m, "body_parentid", None), "device", None)
    return int(getattr(device, "sm_count", 0) or 0)


@cache
def make_incremental_newton_kernel(
    ls_iterations: int,
    nv_static: int,
    pad_static: int,
    reduction_width: int,
    warn_overflow: int,
    min_blocks_per_sm: int = 0,
    sparse_plan=None,
):
    ls = ls_factory(ls_iterations, warn_overflow, reduction_width)
    gradient = gradient_function
    factor = factor_factory(nv_static, pad_static, sparse_plan)
    SPARSE = sparse_plan is not None
    sparse_native = None
    if SPARSE:
        from instantmimic.utils.mjw_kernels import sparse_hessian

        sparse_native = sparse_hessian.native_func(sparse_plan)

    kernel_options = {
        "module": "unique",
        "enable_backward": False,
        "module_options": {"enable_mathdx_gemm": False},
    }
    if min_blocks_per_sm > 0:
        block_width = 64 if pad_static >= 64 else reduction_width
        kernel_options["launch_bounds"] = (block_width, min_blocks_per_sm)

    @wp.kernel(**kernel_options)
    @expose_kernel_name
    def solve_incremental_newton_kernel(
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
        qfrc_dummy: wp.array2d[float],
        M_rowadr: wp.array[int],
        M_col: wp.array[int],
        M_adr: wp.array[int],
        M: wp.array2d[float],
        grad: wp.array2d[float],
        grad_dot: wp.array[float],
        grad_scale: wp.array[float],
        decrement: wp.array[float],
        H: wp.array3d[float],
        U: wp.array3d[float],
        grad3: wp.array3d[float],
        search3: wp.array3d[float],
        dof_union: wp.array2d[int],
        niter: wp.array[int],
        overflow: wp.array[int],
        nsolving: wp.array[int],
        maxiter: int,
    ):
        worldid, lane = wp.tid()
        if wp.static(SPARSE):
            if not ctx_done_in[worldid]:
                # the init kernel's packed H and factor become block-resident
                sparse_native(
                    efc_J_in,
                    efc_D_in,
                    efc_state_out,
                    quad_changed_ids_out,
                    0,
                    dof_union,
                    worldid,
                    H[worldid],
                    grad3[worldid],
                    U[worldid],
                    search3[worldid],
                    lane,
                    2,
                )
        while not ctx_done_in[worldid]:
            if not ctx_search_unchanged_in[worldid]:
                for dof in range(lane, nv, wp.block_dim()):
                    value = float(0.0)
                    for k in range(M_rowadr[dof], M_rowadr[dof + 1]):
                        value += M[worldid, M_adr[k]] * ctx_search_in[worldid, M_col[k]]
                    ctx_mv_in[worldid, dof] = value
            sync()
            ls(
                nv,
                opt_tolerance,
                opt_ls_tolerance,
                opt_impratio_invsqrt,
                stat_meaninertia,
                ne_in,
                nf_in,
                nefc_in,
                qfrc_smooth_in,
                contact_friction_in,
                contact_dim_in,
                contact_efc_address_in,
                efc_type_in,
                efc_id_in,
                efc_J_rownnz_in,
                efc_J_rowadr_in,
                efc_J_colind_in,
                efc_J_in,
                efc_D_in,
                efc_frictionloss_in,
                njmax_in,
                nacon_in,
                ctx_search_unchanged_in,
                ctx_Jaref_in,
                ctx_search_in,
                ctx_search_dot_in,
                ctx_mv_in,
                ctx_jv_in,
                ctx_quad_in,
                ctx_done_in,
                qacc_out,
                efc_force_out,
                efc_state_out,
                efc_Ma_out,
                ctx_Jaref_out,
                ctx_jv_out,
                ctx_quad_out,
                ctx_improvement_out,
                ctx_alpha_out,
                ctx_ls_exhausted_out,
                quad_changed_ids_out,
                quad_changed_count_out,
                state_changed_count_out,
                overflow,
                worldid,
                lane,
            )
            sync()
            gradient(
                nv,
                nefc_in,
                efc_J_in,
                efc_force_out,
                njmax_in,
                qfrc_smooth_in,
                efc_Ma_out,
                state_changed_count_out,
                ctx_alpha_out,
                ctx_done_in,
                qfrc_dummy,
                grad,
                grad_dot,
                decrement,
                grad_scale,
                ctx_search_unchanged_in,
                worldid,
                lane,
            )
            sync()
            factor(
                efc_J_in,
                efc_D_in,
                efc_state_out,
                quad_changed_ids_out,
                quad_changed_count_out,
                state_changed_count_out,
                ctx_done_in,
                grad3,
                H,
                U,
                search3,
                ctx_search_dot_in,
                decrement,
                dof_union,
                worldid,
                lane,
            )
            sync()
            if lane == 0:
                niter[worldid] += 1
                tolerance = opt_tolerance[worldid % opt_tolerance.shape[0]]
                scale = stat_meaninertia[worldid % stat_meaninertia.shape[0]] * float(
                    nv
                )
                converged = (
                    ctx_alpha_out[worldid] == 0.0
                    or (
                        ctx_improvement_out[worldid] > 0.0
                        and ctx_improvement_out[worldid] / scale < tolerance
                    )
                    or wp.sqrt(grad_dot[worldid]) / scale < tolerance
                    or 0.5 * decrement[worldid] / scale < tolerance
                )
                if converged or niter[worldid] == maxiter:
                    if not converged:
                        if wp.static(
                            bool(warn_overflow & types.OverflowType.ITERATIONS)
                        ):
                            wp.printf(
                                "solver iterations limit reached - please increase iterations to %u\n",
                                maxiter,
                            )
                        overflow[worldid] = (
                            overflow[worldid] | types.OverflowType.ITERATIONS.value
                        )
                    ctx_done_in[worldid] = True
                    wp.atomic_add(nsolving, 0, -1)
            sync()

    return solve_incremental_newton_kernel


def attach_scratch(m, d, ctx, device) -> None:
    """Persistent per-world scratch of the fused iteration on ``ctx``: the dofs the
    changed constraint rows touch (``nv`` slots and their count)."""
    ctx.newton_dof_union = wp.zeros((d.nworld, m.nv + 1), dtype=int, device=device)


def eligible(m, ctx, *, compact=False):
    # rounded_step relies on CUDA rounding intrinsics; keep the stock loop elsewhere.
    return (
        not compact
        and m.body_parentid.device.is_cuda
        and 0 < m.nv <= 79
        and m.nv_pad % 16 == 0
        and m.nv_pad <= 80
        and solver._use_incremental(m)
        and m.opt.integrator != types.IntegratorType.DISCRETE
        and m.opt.cone == types.ConeType.PYRAMIDAL
        and not m.is_sparse
        and not solver._sparse_compact(ctx)
        and m.opt.iterations > 0
        and m.block_dim.linesearch_iterative == 32
    )


def newton_block_dim(m) -> int:
    """Select the physical Newton block width from the padded DOF layout."""
    return 64 if m.nv_pad >= 64 else m.block_dim.linesearch_iterative


def run(m, d, ctx, nsolving):
    args = [
        m.nv,
        m.opt.tolerance,
        m.opt.ls_tolerance,
        m.opt.impratio_invsqrt,
        m.stat.meaninertia,
        d.ne,
        d.nf,
        d.nefc,
        d.qfrc_smooth,
        d.contact.friction,
        d.contact.dim,
        d.contact.efc_address,
        d.efc.type,
        d.efc.id,
        d.efc.J_rownnz,
        d.efc.J_rowadr,
        d.efc.J_colind,
        d.efc.J,
        d.efc.D,
        d.efc.frictionloss,
        d.njmax,
        d.nacon,
        ctx.search_unchanged,
        ctx.Jaref,
        ctx.search,
        ctx.search_dot,
        ctx.mv,
        ctx.jv,
        ctx.quad,
        ctx.done,
        d.qacc,
        d.efc.force,
        d.efc.state,
        d.efc.Ma,
        ctx.Jaref,
        ctx.jv,
        ctx.quad,
        ctx.improvement,
        ctx.alpha,
        ctx.ls_exhausted,
        ctx.quad_changed_ids,
        ctx.quad_changed_count,
        ctx.state_changed_count,
    ]
    args += [
        d.qfrc_constraint,
        m.M_mulm_rowadr,
        m.M_mulm_col,
        m.M_mulm_madr,
        d.M,
        ctx.grad,
        ctx.grad_dot,
        ctx.grad_scale,
        ctx.newton_decrement,
        ctx.h,
        ctx.hfactor,
        ctx.grad.reshape((d.nworld, ctx.grad.shape[1], 1)),
        ctx.search.reshape((d.nworld, m.nv, 1)),
        ctx.newton_dof_union,
        d.solver_niter,
        d.overflow,
        nsolving,
        m.opt.iterations,
    ]
    wp.launch_tiled(
        make_incremental_newton_kernel(
            m.opt.ls_iterations,
            m.nv,
            m.nv_pad,
            m.block_dim.linesearch_iterative,
            int(m.opt.warn_overflow),
            newton_min_blocks_per_sm(int(d.nworld), _device_sm_count(m)),
            _fused_plan(ctx),
        ),
        dim=d.nworld,
        block_dim=newton_block_dim(m),
        inputs=args,
    )


@wp.kernel(enable_backward=False)
def _init_efc_settling_empty_worlds(
    nefc_in: wp.array[int],
    qacc_smooth_in: wp.array2d[float],
    solver_niter_out: wp.array[int],
    ctx_search_dot_out: wp.array[float],
    ctx_done_out: wp.array[bool],
    qacc_out: wp.array2d[float],
):
    worldid, dofid = wp.tid()
    empty = nefc_in[worldid] == 0
    if dofid == 0:
        solver_niter_out[worldid] = 0
        ctx_search_dot_out[worldid] = 0.0
        ctx_done_out[worldid] = empty
    if empty:
        qacc_out[worldid, dofid] = qacc_smooth_in[worldid, dofid]


def init_context(m, d, ctx, mul_m_call) -> None:
    """``solver.init_context`` for the fused path, finishing rowless worlds.

    A world with no constraint row has MuJoCo's ``nefc == 0`` result,
    ``qacc = qacc_smooth``, so it starts done: the Hessian assembly,
    factorization and Newton iterations, which all skip done worlds, only
    run where rows exist. ``Ma = M qacc`` is still formed for every world
    because implicit integration reads it. The remaining steps are the
    upstream dense, non-compact ones.
    """
    wp.launch(
        _init_efc_settling_empty_worlds,
        dim=(d.nworld, m.nv),
        inputs=[d.nefc, d.qacc_smooth],
        outputs=[d.solver_niter, ctx.search_dot, ctx.done, d.qacc],
    )
    dofs_per_thread = 20 if m.nv > 50 else 50
    threads_per_efc = ceil(m.nv / dofs_per_thread)
    if threads_per_efc > 1:
        ctx.Jaref.zero_()
    wp.launch(
        solver._solve_init_jaref_kernel(False, m.nv, dofs_per_thread, False),
        dim=(d.nworld, d.njmax, threads_per_efc),
        inputs=[
            d.nefc,
            d.qacc,
            d.efc.J_rownnz,
            d.efc.J_rowadr,
            d.efc.J_colind,
            d.efc.J,
            d.efc.aref,
            d.dof_cdof,
        ],
        outputs=[ctx.Jaref],
    )
    mul_m_call(m, d, d.efc.Ma, d.qacc)
    solver._update_constraint(m, d, ctx)
    plan = _fused_plan(ctx)
    if plan is None:
        solver._update_gradient(m, d, ctx)
    else:
        _update_gradient_sparse(m, d, ctx, plan)


def _fused_plan(ctx):
    plan = getattr(ctx, "sparse_hessian_plan", None)
    return plan if plan is not None and plan.path == "fused" else None


def _update_gradient_sparse(m, d, ctx, plan) -> None:
    """``solver._update_gradient`` for dense pyramidal Newton with H assembled on the
    sparse pattern (no dense H) and factored by the sparse L'DL."""
    from instantmimic.utils.mjw_kernels import sparse_hessian

    wp.launch(
        solver._update_gradient_zero_grad_dot(False),
        dim=d.nworld,
        inputs=[d.nefc, ctx.alpha, ctx.done],
        outputs=[
            ctx.grad_dot,
            ctx.newton_decrement,
            ctx.grad_scale,
            ctx.search_unchanged,
        ],
    )
    wp.launch(
        solver._update_gradient_grad(False),
        dim=(d.nworld, m.nv),
        inputs=[d.qfrc_smooth, d.qfrc_constraint, d.efc.Ma, d.nefc, ctx.done],
        outputs=[ctx.grad, ctx.grad_dot],
    )
    wp.launch_tiled(
        sparse_hessian.init_factor_kernel(plan),
        dim=d.nworld,
        inputs=[
            ctx.done,
            d.nefc,
            d.efc.J,
            d.efc.D,
            d.efc.state,
            d.M,
            ctx.grad.reshape((d.nworld, ctx.grad.shape[1], 1)),
            ctx.hfactor,
            ctx.search.reshape((d.nworld, m.nv, 1)),
            ctx.search_dot,
            ctx.newton_decrement,
        ],
        block_dim=plan.block,
    )


@wp.kernel(enable_backward=False)
def qfrc_constraint_from_grad(
    nefc_in: wp.array[int],
    qfrc_smooth_in: wp.array2d[float],
    efc_Ma_in: wp.array2d[float],
    ctx_grad_in: wp.array2d[float],
    ctx_grad_scale_in: wp.array[float],
    qfrc_constraint_out: wp.array2d[float],
):
    """Upstream ``_qfrc_constraint_from_grad``; zero for worlds without rows."""
    worldid, dofid = wp.tid()
    if nefc_in[worldid] == 0:
        qfrc_constraint_out[worldid, dofid] = 0.0
        return
    grad = ctx_grad_scale_in[worldid] * ctx_grad_in[worldid, dofid]
    qfrc_constraint_out[worldid, dofid] = (
        efc_Ma_in[worldid, dofid] - qfrc_smooth_in[worldid, dofid] - grad
    )
