"""Capability boundaries for the optional dense cooperative Newton loop."""

from types import SimpleNamespace

import pytest
from mujoco_warp._src import types

from instantmimic.utils.mjw_kernels import newton_fusion as fusion


def _model(**updates):
    model = SimpleNamespace(
        nv=34,
        nv_pad=48,
        is_sparse=False,
        opt=SimpleNamespace(
            solver=types.SolverType.NEWTON,
            cone=types.ConeType.PYRAMIDAL,
            iterations=4,
            integrator=types.IntegratorType.IMPLICITFAST,
        ),
        block_dim=SimpleNamespace(linesearch_iterative=32),
        body_parentid=SimpleNamespace(device=SimpleNamespace(is_cuda=True)),
    )
    for key, value in updates.items():
        setattr(model, key, value)
    return model


@pytest.mark.parametrize(
    "nv,pad,expected",
    [
        (0, 0, False),
        (12, 12, False),
        (13, 16, True),
        (16, 16, True),
        (17, 32, True),
        (31, 32, True),
        (32, 32, True),
        (32, 48, True),
        (33, 48, True),
        (35, 48, True),
        (56, 64, True),
        (60, 64, True),
        (61, 64, True),
        (64, 80, True),
        (70, 80, True),
        (79, 80, True),
        (80, 96, False),
        (34, 40, False),
    ],
)
def test_eligibility_uses_dof_layout_not_model_name(monkeypatch, nv, pad, expected):
    monkeypatch.setattr(fusion.solver, "_sparse_compact", lambda ctx: False)
    assert fusion.eligible(_model(nv=nv, nv_pad=pad), object()) is expected


@pytest.mark.parametrize("pad,expected", [(32, 32), (48, 32), (64, 64), (80, 64)])
def test_newton_block_width_uses_padded_dof_layout(pad, expected):
    assert fusion.newton_block_dim(_model(nv_pad=pad)) == expected


@pytest.mark.parametrize(
    "reason",
    [
        "compact",
        "sparse",
        "sparse_compact",
        "cg",
        "discrete",
        "elliptic",
        "zero_iterations",
        "reduction_layout",
        "cpu_device",
    ],
)
def test_unsupported_solver_modes_keep_original_path(monkeypatch, reason):
    model = _model()
    monkeypatch.setattr(
        fusion.solver, "_sparse_compact", lambda ctx: reason == "sparse_compact"
    )
    if reason == "sparse":
        model.is_sparse = True
    if reason == "discrete":
        model.opt.integrator = types.IntegratorType.DISCRETE
    if reason == "cg":
        model.opt.solver = types.SolverType.CG
    if reason == "elliptic":
        model.opt.cone = types.ConeType.ELLIPTIC
    if reason == "zero_iterations":
        model.opt.iterations = 0
    if reason == "reduction_layout":
        model.block_dim.linesearch_iterative = 64
    if reason == "cpu_device":
        model.body_parentid.device.is_cuda = False
    assert not fusion.eligible(model, object(), compact=reason == "compact")


def _chain_xml(nv: int) -> str:
    balls, hinges = divmod(nv - 6, 3)
    links = [
        '<body pos="0 0 0.08"><joint type="ball"/><geom type="sphere" size="0.06" mass="0.3"/>'
        for _ in range(balls)
    ]
    links += [
        '<body pos="0 0 0.08"><joint type="hinge" axis="0 1 0"/><geom type="sphere" size="0.06" mass="0.3"/>'
        for _ in range(hinges)
    ]
    xml = (
        '<mujoco><default><joint armature="0.02"/></default><option jacobian="dense" cone="pyramidal"/><worldbody><geom type="plane" size="10 10 0.1"/><body pos="0 0 0.055"><freejoint/><geom type="sphere" size="0.06" mass="1"/>'
        + "".join(links)
        + "</body>" * len(links)
        + "</body></worldbody></mujoco>"
    )
    return xml


def _chains_xml(chains: int, nv: int) -> str:
    """``chains`` separate floating chains (6 + hinges each) totalling ``nv`` dofs:
    each contact row touches only its own chain's dofs."""
    per, extra = divmod(nv, chains)
    bodies = []
    for c in range(chains):
        hinges = per + (c < extra) - 6
        links = "".join(
            '<body pos="0 0 0.08"><joint type="hinge" axis="0 1 0"/>'
            '<geom type="sphere" size="0.06" mass="0.3"/>'
            for _ in range(hinges)
        )
        bodies.append(
            f'<body pos="{0.4 * c} 0 0.055"><freejoint/>'
            '<geom type="sphere" size="0.06" mass="1"/>'
            + links
            + "</body>" * hinges
            + "</body>"
        )
    return (
        '<mujoco><default><joint armature="0.02"/></default><option jacobian="dense" '
        'cone="pyramidal"/><worldbody><geom type="plane" size="10 10 0.1"/>'
        + "".join(bodies)
        + "</worldbody></mujoco>"
    )


def _assert_fused_matches_stock(
    xml, nv, iterations, ls_iterations, warmstart_scale, *, warn_mask=0
):
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp

    from instantmimic.utils import mjw_step_optimized as runtime
    from instantmimic.utils.mjw_model import put_model

    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    cpu = mujoco.MjModel.from_xml_string(xml)
    cpu.opt.iterations = iterations
    cpu.opt.ls_iterations = ls_iterations
    with wp.ScopedDevice("cuda"):
        m = put_model(cpu)
        m.opt.warn_overflow = warn_mask
        d = mjw.put_data(cpu, mujoco.MjData(cpu), nworld=8, nconmax=64, njmax=256)
        rng = np.random.default_rng(29)
        d.qvel.assign(rng.normal(0, 0.02, (d.nworld, nv)).astype(np.float32))
        mjw.forward(m, d)
        d.qacc_warmstart.assign(
            (warmstart_scale * rng.normal(size=(d.nworld, nv))).astype(np.float32)
        )
        scratch = runtime.make_optimized_fwd_position_scratch(
            m, d, "cuda", include_optimized_kernel_scratch=True
        )
        ctx = scratch.solver_context
        assert fusion.eligible(m, ctx)
        arrays = [
            d.qacc,
            d.qfrc_constraint,
            d.efc.Ma,
            d.efc.force,
            d.efc.state,
            d.solver_niter,
            d.overflow,
            scratch.solver_nsolving,
        ]
        arrays += [v for v in vars(ctx).values() if isinstance(v, wp.array) and v.size]
        saved = [(a, wp.clone(a)) for a in {a.ptr: a for a in arrays}.values()]

        def restore():
            for dst, src in saved:
                wp.copy(dst, src)

        results = []
        for enabled in (False, False, True):
            restore()
            runtime.solve_with_scratch(m, d, scratch, use_optimized_kernel=enabled)
            restore()
            with wp.ScopedCapture() as capture:
                runtime.solve_with_scratch(m, d, scratch, use_optimized_kernel=enabled)
            restore()
            wp.capture_launch(capture.graph)
            results.append(
                [
                    a.numpy()
                    for a in (
                        d.qacc,
                        d.qfrc_constraint,
                        d.efc.force,
                        d.efc.state,
                        d.solver_niter,
                        d.overflow,
                    )
                ]
            )
        # Check reference repeatability using the same scaled norm as candidate
        # parity; elementwise relative error overweights near-zero components.
        for original, repeated in zip(results[0][:3], results[1][:3], strict=False):
            ref = original.astype(float).reshape(d.nworld, -1)
            err = (repeated - original).astype(float).reshape(d.nworld, -1)
            assert (
                np.max(
                    np.linalg.norm(err, axis=1)
                    / np.maximum(1, np.linalg.norm(ref, axis=1))
                )
                < 1e-3
            )
        for original, repeated in zip(results[0][3:], results[1][3:], strict=False):
            np.testing.assert_array_equal(original, repeated)
        for original, optimized in zip(results[0][:3], results[2][:3], strict=False):
            assert np.isfinite(optimized).all()
            ref = original.astype(float).reshape(d.nworld, -1)
            err = (optimized - original).astype(float).reshape(d.nworld, -1)
            assert (
                np.max(
                    np.linalg.norm(err, axis=1)
                    / np.maximum(1, np.linalg.norm(ref, axis=1))
                )
                < 1e-3
            )
        for original, optimized in zip(results[0][3:], results[2][3:], strict=False):
            np.testing.assert_array_equal(original, optimized)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "mask",
    [
        0,
        types.OverflowType.ITERATIONS,
        types.OverflowType.LS_ITERATIONS,
        types.OverflowType.BROADPHASE,
    ],
)
def test_fused_solver_respects_individual_warning_bits(capfd, mask):
    _assert_fused_matches_stock(_chain_xml(35), 35, 1, 1, 50.0, warn_mask=mask)
    import ctypes

    import warp as wp

    wp.synchronize()
    ctypes.CDLL(None).fflush(None)
    output = capfd.readouterr().out
    if not mask & types.OverflowType.ITERATIONS:
        assert "solver iterations limit reached" not in output
    if not mask & types.OverflowType.LS_ITERATIONS:
        assert "linesearch iterations limit reached" not in output


@pytest.mark.cuda
@pytest.mark.parametrize("nv", [13, 16, 29, 31, 32, 33, 35, 56, 60, 61, 62, 70, 79])
@pytest.mark.parametrize("iterations,ls_iterations", [(4, 4), (100, 50)])
@pytest.mark.parametrize("warmstart_scale", [0.0, 50.0])
def test_captured_newton_matches_stock_at_layout_boundaries(
    nv, iterations, ls_iterations, warmstart_scale
):
    _assert_fused_matches_stock(
        _chain_xml(nv), nv, iterations, ls_iterations, warmstart_scale
    )


@pytest.mark.cuda
@pytest.mark.parametrize("chains,nv", [(2, 40), (3, 45), (4, 68), (5, 79)])
@pytest.mark.parametrize("warmstart_scale", [0.0, 50.0])
def test_captured_newton_matches_stock_when_rows_touch_few_dofs(
    chains, nv, warmstart_scale
):
    _assert_fused_matches_stock(_chains_xml(chains, nv), nv, 100, 50, warmstart_scale)


@pytest.mark.cuda
def test_fused_newton_lists_changed_rows_in_order_and_repeats_bitwise():
    """Rows spread over both warps of a 64-lane block: the changed-row list the
    Hessian update sums in must be ascending, so repeated solves agree bitwise."""
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp

    from instantmimic.utils import mjw_step_optimized as runtime
    from instantmimic.utils.mjw_model import put_model

    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    boxes = "".join(
        f'<body pos="{0.3 * i} 0 0.049"><freejoint/>'
        '<geom type="box" size="0.1 0.1 0.05" mass="1"/></body>'
        for i in range(12)
    )
    cpu = mujoco.MjModel.from_xml_string(
        '<mujoco><option jacobian="dense" cone="pyramidal"/><worldbody>'
        '<geom type="plane" size="10 10 0.1"/>' + boxes + "</worldbody></mujoco>"
    )
    cpu.opt.iterations = 3
    with wp.ScopedDevice("cuda"):
        m = put_model(cpu)
        m.opt.warn_overflow = False
        d = mjw.put_data(cpu, mujoco.MjData(cpu), nworld=64, nconmax=64, njmax=256)
        rng = np.random.default_rng(29)
        d.qvel.assign(rng.normal(0, 0.5, (d.nworld, m.nv)).astype(np.float32))
        mjw.forward(m, d)
        d.qacc_warmstart.assign(
            (50.0 * rng.normal(size=(d.nworld, m.nv))).astype(np.float32)
        )
        scratch = runtime.make_optimized_fwd_position_scratch(
            m, d, "cuda", include_optimized_kernel_scratch=True
        )
        ctx = scratch.solver_context
        assert fusion.newton_block_dim(m) == 64 and fusion.eligible(m, ctx)
        assert d.nefc.numpy().min() > 64
        arrays = [d.qacc, d.efc.Ma, d.efc.force, d.efc.state, d.solver_niter]
        arrays += [d.overflow, scratch.solver_nsolving]
        arrays += [v for v in vars(ctx).values() if isinstance(v, wp.array) and v.size]
        saved = [(a, wp.clone(a)) for a in {a.ptr: a for a in arrays}.values()]
        solutions = []
        for _ in range(8):
            for dst, src in saved:
                wp.copy(dst, src)
            runtime.solve_with_scratch(m, d, scratch, use_optimized_kernel=True)
            count = ctx.quad_changed_count.numpy()
            ids = ctx.quad_changed_ids.numpy()
            assert count.max() > 1
            for w in range(d.nworld):
                assert np.all(np.diff(ids[w, : count[w]]) > 0)
            solutions.append(d.qacc.numpy())
        for repeated in solutions[1:]:
            np.testing.assert_array_equal(solutions[0], repeated)


@pytest.mark.cuda
def test_fused_newton_finishes_rowless_worlds_at_the_smooth_acceleration():
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp

    from instantmimic.utils import mjw_step_optimized as runtime

    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    links = "".join(
        '<body pos="0 0 0.08"><joint type="hinge" axis="0 1 0"/>'
        '<geom type="sphere" size="0.06" mass="0.3"/>'
        for _ in range(10)
    )
    xml = (
        '<mujoco><default><joint armature="0.02"/></default>'
        '<option jacobian="dense" cone="pyramidal"/><worldbody>'
        '<geom type="plane" size="10 10 0.1"/><body pos="0 0 0.055"><freejoint/>'
        f'<geom type="sphere" size="0.06" mass="1"/>{links}'
        + "</body>" * 10
        + "</body></worldbody></mujoco>"
    )
    cpu = mujoco.MjModel.from_xml_string(xml)
    nworld = 8
    with wp.ScopedDevice("cuda"):
        m = mjw.put_model(cpu)
        m.opt.warn_overflow = False
        d = mjw.put_data(cpu, mujoco.MjData(cpu), nworld=nworld, nconmax=64, njmax=256)
        rng = np.random.default_rng(3)
        qpos = d.qpos.numpy()
        qpos[::2, 2] = 3.0  # even worlds fly: no contact rows
        d.qpos.assign(qpos)
        d.qvel.assign(rng.normal(0, 0.05, (nworld, m.nv)).astype(np.float32))
        mjw.forward(m, d)
        d.qacc_warmstart.assign(rng.normal(size=(nworld, m.nv)).astype(np.float32))
        nefc = d.nefc.numpy()
        assert np.all(nefc[::2] == 0) and np.all(nefc[1::2] > 0)
        scratch = runtime.make_optimized_fwd_position_scratch(
            m, d, "cuda", include_optimized_kernel_scratch=True
        )
        assert runtime.newton_fusion.eligible(m, scratch.solver_context)
        ctx = scratch.solver_context
        arrays = [d.qacc, d.qfrc_constraint, d.efc.Ma, d.efc.force, d.efc.state]
        arrays += [d.solver_niter, d.overflow, scratch.solver_nsolving]
        arrays += [v for v in vars(ctx).values() if isinstance(v, wp.array) and v.size]
        saved = [(a, wp.clone(a)) for a in {a.ptr: a for a in arrays}.values()]
        results = []
        for enabled in (False, True):
            for dst, src in saved:
                wp.copy(dst, src)
            with wp.ScopedCapture() as capture:
                runtime.solve_with_scratch(m, d, scratch, use_optimized_kernel=enabled)
            wp.capture_launch(capture.graph)
            results.append(
                [a.numpy().copy() for a in (d.qacc, d.qfrc_constraint, d.efc.Ma)]
                + [d.solver_niter.numpy().copy()]
            )
        stock, fused = results
        smooth = d.qacc_smooth.numpy()
        np.testing.assert_array_equal(fused[0][::2], smooth[::2])
        np.testing.assert_array_equal(fused[1][::2], 0.0)
        np.testing.assert_array_equal(fused[3][::2], 0)
        for reference, candidate in zip(stock[:3], fused[:3], strict=True):
            scale = np.maximum(1.0, np.linalg.norm(reference, axis=1))
            assert np.max(np.linalg.norm(candidate - reference, axis=1) / scale) < 1e-3


@pytest.mark.parametrize(
    "nworld,sm_count,expected",
    [
        # The request tracks what the grid supplies, so it self-disables where a
        # cap cannot add residency and clamps where a tighter cap measured worse.
        (1024, 170, 6),
        (2048, 170, 12),
        (4096, 170, 12),
        (8192, 170, 12),
        (512, 170, 3),
        (100, 170, 1),
        # Device portability: the same world count crosses the wave boundary at
        # different points on smaller devices.
        (1024, 108, 9),
        (1024, 132, 7),
        # Degenerate inputs disable the cap instead of guessing.
        (0, 170, 0),
        (1024, 0, 0),
        (-1, 170, 0),
    ],
)
def test_newton_min_blocks_tracks_grid_and_device(nworld, sm_count, expected):
    assert fusion.newton_min_blocks_per_sm(nworld, sm_count) == expected


def test_newton_min_blocks_never_exceeds_measured_maximum():
    for sm_count in (1, 8, 108, 170):
        for nworld in (1, 64, 1024, 65536):
            value = fusion.newton_min_blocks_per_sm(nworld, sm_count)
            assert 0 <= value <= fusion._NEWTON_MAX_BLOCKS_PER_SM


def test_newton_kernel_launch_bounds_follow_min_blocks(monkeypatch):
    built = {}

    def fake_kernel(**options):
        built.update(options)

        def decorate(func):
            return func

        return decorate

    monkeypatch.setattr(fusion.wp, "kernel", fake_kernel)
    fusion.make_incremental_newton_kernel.cache_clear()
    try:
        fusion.make_incremental_newton_kernel(50, 56, 64, 32, False, 0)
        assert "launch_bounds" not in built
        built.clear()
        fusion.make_incremental_newton_kernel(50, 56, 64, 32, False, 12)
        assert built["launch_bounds"] == (64, 12)
        built.clear()
        # A pad-48 model launches the line-search width, not 64.
        fusion.make_incremental_newton_kernel(50, 35, 48, 32, False, 9)
        assert built["launch_bounds"] == (32, 9)
    finally:
        fusion.make_incremental_newton_kernel.cache_clear()


@pytest.mark.cuda
@pytest.mark.parametrize("iterations,ls_iterations", [(1, 1), (4, 4), (100, 50)])
@pytest.mark.parametrize("warmstart_scale", [0.0, 50.0])
def test_fused_line_search_includes_equality_in_gaussian(
    iterations, ls_iterations, warmstart_scale
):
    xml = (
        _chain_xml(35)
        .replace('<body pos="0 0 0.055">', '<body name="base" pos="0 0 0.055">')
        .replace("</mujoco>", '<equality><weld body1="base"/></equality></mujoco>')
    )
    _assert_fused_matches_stock(xml, 35, iterations, ls_iterations, warmstart_scale)


def _factor_kernel(factor, pad):
    import warp as wp

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        H: wp.array3d[float],
        grad: wp.array3d[float],
        U: wp.array3d[float],
        search: wp.array3d[float],
        sums: wp.array[wp.vec2],
    ):
        w, _lane = wp.tid()
        s = factor(H[w], grad[w], pad, U[w], search[w])
        sums[w] = s

    return kernel


@pytest.mark.cuda
@pytest.mark.parametrize("nv,pad", [(40, 48), (56, 64), (70, 80), (79, 80)])
def test_lean_augmented_factor_matches_upstream_bitwise(nv, pad):
    import numpy as np
    import warp as wp
    from mujoco_warp._src import block_cholesky

    from instantmimic.utils.mjw_kernels import newton_stages

    nworld = 8
    rng = np.random.default_rng(nv)
    a = rng.standard_normal((nworld, pad, pad)).astype(np.float32)
    h = a @ a.transpose(0, 2, 1) / pad + 4.0 * np.eye(pad, dtype=np.float32)
    h[:, nv:, :] = 0.0
    h[:, :, nv:] = 0.0
    h[:, np.arange(nv, pad), np.arange(nv, pad)] = 1.0
    grad = np.zeros((nworld, pad, 1), np.float32)
    grad[:, :nv, 0] = rng.standard_normal((nworld, nv))
    results = []
    for factor in (
        block_cholesky.create_blocked_cholesky_augmented_factorize_solve_newton_func(
            16, pad, nv
        ),
        newton_stages.augmented_factor_solve_factory(16, pad, nv),
    ):
        args = [wp.array(h, device="cuda:0"), wp.array(grad, device="cuda:0")]
        outs = [
            wp.zeros((nworld, pad, pad), dtype=float, device="cuda:0"),
            wp.zeros((nworld, nv, 1), dtype=float, device="cuda:0"),
            wp.zeros(nworld, dtype=wp.vec2, device="cuda:0"),
        ]
        wp.launch_tiled(
            _factor_kernel(factor, pad),
            dim=nworld,
            inputs=args,
            outputs=outs,
            block_dim=64,
            device="cuda:0",
        )
        results.append([o.numpy().view(np.uint32) for o in outs])
    assert np.isfinite(results[0][1].view(np.float32)).all()
    for upstream, lean in zip(*results, strict=True):
        np.testing.assert_array_equal(lean, upstream)
