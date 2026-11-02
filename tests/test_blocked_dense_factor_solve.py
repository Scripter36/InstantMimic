"""Blocked dense inertia factor+solve: gating, wiring, and numerical parity."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import instantmimic.utils.mjw_step_optimized as optimized


def _dense_model(nv: int, *, tiles=None, **overrides):
    if tiles is None:
        tiles = (
            SimpleNamespace(
                size=nv,
                adr=SimpleNamespace(size=1),
                elemid=SimpleNamespace(
                    size=nv * nv,
                    shape=(nv * nv,),
                    numpy=lambda: np.arange(nv * nv, dtype=np.int32),
                ),
            ),
        )
    fields = {
        "nv": nv,
        "is_sparse": True,
        "qLD_block_total": nv * nv,
        "M_tiles": tiles,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "model,qld_width",
    [
        (_dense_model(8), 8 * 8),
        (_dense_model(56), 56 * 56 + 1),
        (
            _dense_model(
                56,
                tiles=(
                    SimpleNamespace(
                        size=28,
                        adr=SimpleNamespace(size=2),
                        elemid=SimpleNamespace(size=0, shape=(0,)),
                    ),
                ),
            ),
            56 * 56,
        ),
        (SimpleNamespace(nv=object()), 0),
    ],
)
def test_blocked_scratch_is_skipped_for_unsupported_models(model, qld_width):
    data = SimpleNamespace(
        nworld=4,
        M=object(),
        qLD=SimpleNamespace(shape=(4, qld_width)),
    )
    assert optimized.create_blocked_dense_factor_solve_scratch(model, data) is None


@pytest.mark.parametrize("constraint_is_sparse", [False, True])
def test_blocked_scratch_uses_m_layout_not_constraint_jacobian(
    monkeypatch, constraint_is_sparse
):
    arrays = iter(["U"])
    monkeypatch.setattr(optimized.wp, "zeros", lambda *_args, **_kwargs: next(arrays))
    monkeypatch.setattr(optimized.wp, "launch", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        optimized,
        "make_indexed_cholesky_factor_solve_kernel",
        lambda *args: "indexed-kernel",
    )
    monkeypatch.setattr(optimized.wp, "array", lambda *args, **kwargs: args[0])
    model = _dense_model(56, is_sparse=constraint_is_sparse)
    data = SimpleNamespace(
        nworld=4,
        M=SimpleNamespace(shape=(4, 56 * 56)),
        qLD=SimpleNamespace(shape=(4, 56 * 56)),
    )

    scratch = optimized.create_blocked_dense_factor_solve_scratch(model, data)

    assert scratch is not None
    assert scratch.U == "U"
    assert scratch.indices.shape == (64 * 64,)
    assert np.count_nonzero(scratch.identity) == 8


def test_blocked_scratch_allows_explicit_small_dense_model(monkeypatch):
    arrays = iter(["U"])
    monkeypatch.setattr(optimized.wp, "zeros", lambda *_args, **_kwargs: next(arrays))
    monkeypatch.setattr(
        optimized,
        "make_indexed_cholesky_factor_solve_kernel",
        lambda *args: "indexed-kernel",
    )
    monkeypatch.setattr(optimized.wp, "array", lambda *args, **kwargs: args[0])
    model = _dense_model(31)
    data = SimpleNamespace(
        nworld=4,
        M=SimpleNamespace(shape=(4, 31 * 31)),
        qLD=SimpleNamespace(shape=(4, 31 * 31)),
    )

    default = optimized.create_blocked_dense_factor_solve_scratch(model, data)
    required = optimized.create_blocked_dense_factor_solve_scratch(
        model,
        data,
        minimum_nv=1,
    )

    assert default is None
    assert required is not None
    assert required.nv == 31
    assert required.pad == 32


def test_factor_solve_i_dense_delegates_without_scratch(monkeypatch):
    calls = []

    def fake_no_store(model, data, matrix, x, y):
        calls.append((model, data, matrix, x, y))

    monkeypatch.setattr(optimized, "_factor_solve_i_dense_no_l_store", fake_no_store)
    optimized._factor_solve_i_dense("m", "d", "M", "x", "y", None)
    assert calls == [("m", "d", "M", "x", "y")]


def test_pre_advance_modifier_can_supply_acceleration_and_drift_velocity() -> None:
    calls = []
    corrected_qacc = object()
    drift_qvel = object()

    def modifier(model, data, qacc, qvel):
        calls.append(("modifier", model, data, qacc, qvel))
        return corrected_qacc, drift_qvel

    def advance(model, data, qacc, qvel):
        calls.append(("advance", model, data, qacc, qvel))

    scratch = SimpleNamespace(
        advance_call=advance,
        pre_advance_modifier=SimpleNamespace(name="contact", apply=modifier),
    )

    optimized._advance_with_scratch("m", "d", "free-qacc", scratch)

    assert calls == [
        ("modifier", "m", "d", "free-qacc", None),
        ("advance", "m", "d", corrected_qacc, drift_qvel),
    ]


def test_pre_advance_without_modifier_preserves_original_arguments() -> None:
    calls = []

    def advance(model, data, qacc, qvel):
        calls.append((model, data, qacc, qvel))

    scratch = SimpleNamespace(advance_call=advance, pre_advance_modifier=None)

    optimized._advance_with_scratch("m", "d", "qacc", scratch, "qvel")

    assert calls == [("m", "d", "qacc", "qvel")]


@pytest.mark.parametrize("use_optimized_kernel", [False, True])
def test_implicitfast_pre_advance_modifier_cannot_be_bypassed(
    monkeypatch, use_optimized_kernel
) -> None:
    calls = []

    def modifier(model, data, qacc, qvel):
        calls.append(("modifier", qacc, qvel))
        return "corrected-qacc", "drift-qvel"

    def advance(model, data, qacc, qvel):
        calls.append(("advance", qacc, qvel))

    def fail_implicit(*_args):
        raise AssertionError("the stock implicit adapter bypasses the modifier")

    monkeypatch.setattr(optimized, "_implicitfast_needs_deriv", lambda *_args: False)
    monkeypatch.setattr(
        optimized, "_has_implicit_dense_deriv_scratch", lambda *_args: False
    )
    monkeypatch.setattr(optimized, "_has_implicit_factor_scratch", lambda *_args: False)
    model = SimpleNamespace(
        opt=SimpleNamespace(integrator=optimized.IntegratorType.IMPLICITFAST)
    )
    data = SimpleNamespace(qacc="free-qacc")
    scratch = SimpleNamespace(
        advance_call=advance,
        pre_advance_modifier=SimpleNamespace(name="contact", apply=modifier),
        implicit_call=fail_implicit,
        implicit_qh_contributors=(),
    )

    optimized._integrate(
        model,
        data,
        scratch,
        use_optimized_kernel=use_optimized_kernel,
    )

    assert calls == [
        ("modifier", "free-qacc", None),
        ("advance", "corrected-qacc", "drift-qvel"),
    ]


def test_implicitfast_native_derivative_fallback_dispatches_modifier(monkeypatch):
    calls = []

    def deriv(model, data, qderiv):
        calls.append(("deriv", qderiv))

    def factor(model, data, qderiv, qld, diag, qacc, source, tree_factor_solve=None):
        calls.append(("factor", qderiv, qld, diag, qacc, source))

    def modifier(model, data, qacc, qvel):
        calls.append(("modifier", qacc, qvel))
        return "corrected-qacc", None

    def advance(model, data, qacc):
        calls.append(("advance", qacc))

    monkeypatch.setattr(optimized, "_implicitfast_needs_deriv", lambda *_args: True)
    monkeypatch.setattr(
        optimized, "_has_implicit_dense_deriv_scratch", lambda *_args: False
    )
    monkeypatch.setattr(optimized, "_has_implicit_factor_scratch", lambda *_args: False)
    monkeypatch.setattr(optimized, "factor_solve_i_no_l_store", factor)
    model = SimpleNamespace(
        opt=SimpleNamespace(integrator=optimized.IntegratorType.IMPLICITFAST)
    )
    data = SimpleNamespace(efc=SimpleNamespace(Ma="ma"))
    scratch = SimpleNamespace(
        advance_call=advance,
        pre_advance_modifier=SimpleNamespace(name="contact", apply=modifier),
        implicit_call=lambda *_args: pytest.fail("stock implicit bypassed modifier"),
        implicit_deriv_call=deriv,
        implicit_qderiv="qderiv",
        implicit_qld="qld",
        implicit_ql_diag_inv="diag",
        implicit_qacc="qacc",
        implicit_qh_contributors=(),
    )

    optimized._integrate(model, data, scratch, use_optimized_kernel=False)

    assert calls == [
        ("deriv", "qderiv"),
        ("factor", "qderiv", "qld", "diag", "qacc", "ma"),
        ("modifier", "qacc", None),
        ("advance", "corrected-qacc"),
    ]


def test_factor_solve_i_dense_launches_blocked_kernels(monkeypatch):
    launches = []
    tiled = []

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        launches.append((kernel, dim, inputs, outputs))

    def fake_launch_tiled(kernel, *, dim, inputs, outputs, block_dim, **_kwargs):
        tiled.append((kernel, dim, inputs, outputs, block_dim))

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)
    monkeypatch.setattr(optimized.wp, "launch_tiled", fake_launch_tiled)

    scratch = optimized.BlockedDenseFactorSolveScratch(
        nv=56,
        pad=64,
        block_size=16,
        block_dim=32,
        U="U",
        kernel="blocked-kernel",
        indices="indices",
        identity="identity",
    )
    data = SimpleNamespace(nworld=3)
    x = SimpleNamespace(reshape=lambda shape: ("x", shape))
    y = SimpleNamespace(reshape=lambda shape: ("y", shape))
    optimized._factor_solve_i_dense("m", data, "M", x, y, scratch)

    assert launches == []
    assert tiled == [
        (
            "blocked-kernel",
            3,
            ["M", "indices", "identity", ("y", (3, 56, 1)), 64, 56],
            ["U", ("x", (3, 56, 1))],
            32,
        )
    ]


def test_factor_solve_i_dense_can_target_external_factor_workspace(monkeypatch):
    tiled = []

    monkeypatch.setattr(
        optimized.wp,
        "launch_tiled",
        lambda kernel, *, dim, inputs, outputs, block_dim, **_kwargs: tiled.append(
            (kernel, dim, inputs, outputs, block_dim)
        ),
    )
    scratch = optimized.BlockedDenseFactorSolveScratch(
        nv=56,
        pad=64,
        block_size=16,
        block_dim=32,
        U="default-U",
        kernel="blocked-kernel",
        indices="indices",
        identity="identity",
    )
    data = SimpleNamespace(nworld=3)
    x = SimpleNamespace(reshape=lambda shape: ("x", shape))
    y = SimpleNamespace(reshape=lambda shape: ("y", shape))

    optimized._factor_solve_i_dense(
        "m",
        data,
        "M",
        x,
        y,
        scratch,
        factor_workspace="physical-U",
    )

    assert tiled[0][3] == ["physical-U", ("x", (3, 56, 1))]


def test_factor_solve_i_dense_can_persist_blocked_factor_for_cg(monkeypatch):
    launches = []

    monkeypatch.setattr(
        optimized.wp,
        "launch",
        lambda kernel, *, dim, inputs, outputs, **_kwargs: launches.append(
            (kernel, dim, inputs, outputs)
        ),
    )
    monkeypatch.setattr(
        optimized.wp,
        "launch_tiled",
        lambda *_args, **_kwargs: None,
    )
    scratch = optimized.BlockedDenseFactorSolveScratch(
        nv=48,
        pad=64,
        block_size=16,
        block_dim=32,
        U="U",
        kernel="blocked-kernel",
        indices="indices",
        identity="identity",
    )
    model = SimpleNamespace(qLD_block_adr="qLD-block-adr")
    data = SimpleNamespace(nworld=2, qLD="qLD")

    optimized._factor_solve_i_dense(
        model,
        data,
        "M",
        SimpleNamespace(reshape=lambda shape: ("x", shape)),
        SimpleNamespace(reshape=lambda shape: ("y", shape)),
        scratch,
        retain_factor=True,
    )

    assert launches[-1] == (
        optimized._copy_dense_factor_kernel,
        (2, 48 * 48),
        ["U", "qLD-block-adr", 48],
        ["qLD"],
    )


def _chain_mjcf(num_links: int) -> str:
    body = ""
    for index in range(num_links):
        body += (
            f'<body name="link{index}" pos="0 0 -0.3">'
            f'<joint type="ball" damping="{1.0 + 0.1 * index}" armature="0.02"/>'
            f'<geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" mass="{1.0 + index % 3}"/>'
        )
    body += "</body>" * num_links
    return f"""
    <mujoco>
      <option timestep="0.005" integrator="implicitfast" jacobian="dense"/>
      <worldbody>
        <body name="root" pos="0 0 3">
          <freejoint/>
          <geom type="box" size="0.1 0.1 0.1" mass="5"/>
          {body}
        </body>
      </worldbody>
    </mujoco>
    """


def _snu_sized_chain_mjcf() -> str:
    body = ""
    for index in range(16):
        body += (
            f'<body name="ball{index}" pos="0 0 -0.2">'
            '<joint type="ball" damping="1" armature="0.02"/>'
            '<geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2" mass="1"/>'
        )
    for index in range(2):
        body += (
            f'<body name="hinge{index}" pos="0 0 -0.2">'
            '<joint type="hinge" axis="0 1 0" damping="1" armature="0.02"/>'
            '<geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2" mass="1"/>'
        )
    body += "</body>" * 18
    return f"""
    <mujoco>
      <option timestep="0.005" integrator="implicitfast" jacobian="dense"/>
      <worldbody>
        <body name="root" pos="0 0 5">
          <freejoint/>
          <geom type="box" size="0.1 0.1 0.1" mass="5"/>
          {body}
        </body>
      </worldbody>
    </mujoco>
    """


def _mixed_tree_mjcf(small_links: int, large_links: int) -> str:
    def tree(name: str, count: int, x: float) -> str:
        bodies = ""
        for index in range(count):
            bodies += (
                f'<body name="{name}{index}" pos="0 0 0.1">'
                '<joint type="hinge" axis="0 1 0" armature="0.01"/>'
                '<geom type="capsule" size="0.01" fromto="0 0 0 0 0 0.1" '
                'mass="0.1" contype="0" conaffinity="0"/>'
            )
        return f'<body pos="{x} 0 0">{bodies}{"</body>" * (count + 1)}'

    return f"""
    <mujoco>
      <option jacobian="sparse"/>
      <worldbody>
        {tree("small", small_links, -1.0)}
        {tree("large", large_links, 1.0)}
      </worldbody>
    </mujoco>
    """


@pytest.mark.cuda
@pytest.mark.parametrize("num_links", [10, 14, 17])
def test_blocked_factor_solve_matches_single_tile_on_cuda(num_links):
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("CUDA device required")
    import mujoco
    import mujoco_warp as mjw

    m_mj = mujoco.MjModel.from_xml_string(_chain_mjcf(num_links))
    # The additional long-chain case includes rotor inertia to keep a direct
    # single-vs-blocked FP32 comparison well conditioned. Original cases remain.
    if num_links == 17:
        m_mj.dof_armature[6:] = 0.2
    d_mj = mujoco.MjData(m_mj)
    nworld = 6
    with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(m_mj)
        d = mjw.put_data(m_mj, d_mj, nworld=nworld, nconmax=4, njmax=16)
        rng = np.random.default_rng(3)
        qpos = np.tile(m_mj.qpos0, (nworld, 1))
        qpos += rng.normal(scale=0.4, size=qpos.shape)
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
        d.qpos.assign(qpos.astype(np.float32))
        mjw.forward(m, d)

        scratch = optimized.create_blocked_dense_factor_solve_scratch(m, d)
        assert scratch is not None, "chain model should expose one dense M tile"
        assert scratch.nv == m.nv and scratch.pad == ((m.nv + 15) // 16) * 16

        y = wp.array(
            rng.normal(scale=30.0, size=(nworld, m.nv)).astype(np.float32),
            dtype=float,
        )
        x_ref = wp.zeros((nworld, m.nv), dtype=float)
        x_blk = wp.zeros((nworld, m.nv), dtype=float)
        optimized._factor_solve_i_dense_no_l_store(m, d, d.M, x_ref, y)
        optimized._factor_solve_i_dense(m, d, d.M, x_blk, y, scratch)
        # Retained factors must match the stock layout needed by later CG work.
        optimized._factor_solve_i_dense(
            m, d, d.M, x_blk, y, scratch, retain_factor=True
        )
        factor = d.qLD.numpy().copy()
        from mujoco_warp._src import smooth

        smooth.factor_m(m, d)
        stock_factor = d.qLD.numpy()
        np.testing.assert_allclose(factor, stock_factor, rtol=2e-4, atol=2e-4)
        wp.synchronize()

        ref = x_ref.numpy()
        blk = x_blk.numpy()
        scale = max(float(np.abs(ref).max()), 1e-6)
        assert np.abs(blk - ref).max() / scale < 1e-4

        # Residual check against the CPU dense inertia for world 0.
        d_mj.qpos[:] = qpos[0]
        mujoco.mj_forward(m_mj, d_mj)
        dense = np.zeros((m_mj.nv, m_mj.nv))
        mujoco.mj_fullM(m_mj, d_mj, dense)
        residual = np.abs(dense @ blk[0] - y.numpy()[0]).max()
        assert residual / max(float(np.abs(y.numpy()[0]).max()), 1e-6) < 1e-3


@pytest.mark.cuda
def test_blocked_factor_multi_solve_matches_dense_inertia_on_cuda():
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    from instantmimic.utils.mjw_kernels.indexed_factor import (
        make_indexed_cholesky_factor_multi_solve_kernel,
        make_indexed_cholesky_multi_solve_kernel,
    )

    if not wp.is_cuda_available():
        pytest.skip("CUDA device required")

    m_mj = mujoco.MjModel.from_xml_string(_snu_sized_chain_mjcf())
    assert m_mj.nv == 56
    d_mj = mujoco.MjData(m_mj)
    nworld = 3
    rhs_count = 5
    with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(m_mj)
        d = mjw.put_data(m_mj, d_mj, nworld=nworld, nconmax=4, njmax=16)
        rng = np.random.default_rng(31)
        qpos = np.tile(m_mj.qpos0, (nworld, 1))
        qpos += rng.normal(scale=0.2, size=qpos.shape)
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
        d.qpos.assign(qpos.astype(np.float32))
        mjw.forward(m, d)

        scratch = optimized.create_blocked_dense_factor_solve_scratch(m, d)
        assert scratch is not None
        rhs = rng.normal(size=(nworld, m.nv, rhs_count)).astype(np.float32)
        rhs_wp = wp.array(rhs, dtype=wp.float32)
        active = wp.array([1, 0, 1], dtype=wp.int32)
        factor = wp.zeros(
            (nworld, scratch.pad, scratch.pad),
            dtype=wp.float32,
        )
        reused_factor = wp.zeros_like(factor)
        solution = wp.zeros_like(rhs_wp)
        reused_solution = wp.zeros_like(rhs_wp)
        kernel = make_indexed_cholesky_factor_multi_solve_kernel(
            scratch.block_size,
            scratch.pad,
            scratch.nv,
            rhs_count,
        )
        wp.launch_tiled(
            kernel,
            dim=nworld,
            block_dim=scratch.block_dim,
            inputs=[
                d.M,
                scratch.indices,
                scratch.identity,
                rhs_wp,
                active,
                scratch.pad,
                scratch.nv,
            ],
            outputs=[factor, solution],
        )
        single_rhs = wp.array(rhs[:, :, 0], dtype=wp.float32)
        single_solution = wp.zeros_like(single_rhs)
        optimized._factor_solve_i_dense(
            m,
            d,
            d.M,
            single_solution,
            single_rhs,
            scratch,
            factor_workspace=reused_factor,
        )
        reuse_kernel = make_indexed_cholesky_multi_solve_kernel(
            scratch.block_size,
            scratch.pad,
            scratch.nv,
            rhs_count,
        )
        wp.launch_tiled(
            reuse_kernel,
            dim=nworld,
            block_dim=scratch.block_dim,
            inputs=[reused_factor, rhs_wp, active],
            outputs=[reused_solution],
        )
        wp.synchronize()

        actual = solution.numpy()
        np.testing.assert_array_equal(reused_solution.numpy(), actual)
        np.testing.assert_array_equal(actual[1], 0.0)
        for world in (0, 2):
            d_mj.qpos[:] = qpos[world]
            mujoco.mj_forward(m_mj, d_mj)
            dense = np.zeros((m_mj.nv, m_mj.nv))
            mujoco.mj_fullM(m_mj, d_mj, dense)
            expected = np.linalg.solve(dense, rhs[world])
            scale = max(float(np.abs(expected).max()), 1.0e-6)
            assert np.abs(actual[world] - expected).max() / scale < 2.0e-3


@pytest.mark.cuda
def test_no_store_factor_solve_matches_mixed_dense_and_sparse_m_on_cuda():
    import warp as wp

    if not wp.is_cuda_available():
        pytest.skip("CUDA device required")
    import mujoco
    import mujoco_warp as mjw
    from mujoco_warp._src import smooth

    m_mj = mujoco.MjModel.from_xml_string(_mixed_tree_mjcf(8, 65))
    d_mj = mujoco.MjData(m_mj)
    d_mj.qpos[:] = np.linspace(-0.2, 0.2, m_mj.nq)
    mujoco.mj_forward(m_mj, d_mj)
    nworld = 3
    with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(m_mj)
        d_ref = mjw.put_data(m_mj, d_mj, nworld=nworld, nconmax=0, njmax=0)
        d = mjw.put_data(m_mj, d_mj, nworld=nworld, nconmax=0, njmax=0)
        assert any(tile.size == 8 for tile in m.M_tiles)
        assert d.qLD.shape[1] > m.qLD_block_total

        rng = np.random.default_rng(19)
        rhs = rng.normal(size=(nworld, m.nv)).astype(np.float32)
        y = wp.array(rhs, dtype=float)
        x_ref = wp.zeros((nworld, m.nv), dtype=float)
        x = wp.zeros((nworld, m.nv), dtype=float)
        smooth.factor_solve_i(
            m,
            d_ref,
            d_ref.M,
            d_ref.qLD,
            d_ref.qLDiagInv,
            x_ref,
            y,
        )
        optimized._factor_solve_i_dense_no_l_store(m, d, d.M, x, y)
        wp.synchronize()

        actual = x.numpy()
        expected = x_ref.numpy()
        np.testing.assert_allclose(actual[:, :8], expected[:, :8], rtol=1e-6, atol=1e-6)
        scale = max(float(np.abs(expected).max()), 1e-6)
        assert np.abs(actual - expected).max() / scale < 5e-3


@pytest.mark.parametrize(
    ("dense", "tree"), [("blocked-scratch", None), (None, "tree-scratch")]
)
def test_integrate_routes_actuator_free_implicitfast_through_factor_path(
    monkeypatch, dense, tree
):
    """The blocked dense kernel or, for inertia split into several trees, the
    generated tree solve keeps implicit-fast off the stock path."""
    events = []

    def fake_deriv(model, data, out):
        events.append(("deriv", out))

    def fake_factor(
        model,
        data,
        matrix,
        x,
        y,
        dense_factor_solve=None,
        *,
        factor_workspace=None,
        tree_factor_solve=None,
    ):
        assert factor_workspace is None
        events.append(("factor", matrix, x, y, dense_factor_solve, tree_factor_solve))

    def fake_advance(model, data, qacc):
        events.append(("advance", qacc))

    def fail_stock_implicit(*_args, **_kwargs):
        raise AssertionError("stock implicit must not run when blocked scratch exists")

    monkeypatch.setattr(optimized, "deriv_smooth_vel", fake_deriv)
    monkeypatch.setattr(optimized, "_factor_solve_i_dense", fake_factor)
    monkeypatch.setattr(optimized.mjw_forward, "_advance", fake_advance)
    monkeypatch.setattr(optimized.mjw_forward, "implicit", fail_stock_implicit)

    model = SimpleNamespace(
        opt=SimpleNamespace(
            integrator=optimized.IntegratorType.IMPLICITFAST, disableflags=0
        ),
        is_sparse=True,
        nu=0,
        nv=56,
    )
    data = SimpleNamespace(nworld=2, efc=SimpleNamespace(Ma="ma"))
    scratch = SimpleNamespace(
        implicit_qderiv="qderiv",
        implicit_qacc="qacc",
        implicit_deriv_call=None,
        implicit_call=fail_stock_implicit,
        implicit_actuator_vel=None,
        implicit_qderiv_pair_mask=None,
        implicit_max_moment_rownnz=0,
        implicit_qh_contributors=(),
        implicit_qld=None,
        implicit_ql_diag_inv=None,
        dense_factor_solve=dense,
        tree_factor_solve=tree,
    )

    optimized._integrate(model, data, scratch, use_optimized_kernel=True)

    assert events == [
        ("deriv", "qderiv"),
        ("factor", "qderiv", "qacc", "ma", dense, tree),
        ("advance", "qacc"),
    ]
    assert optimized.supports_allocation_free_implicit_capture(
        model, scratch, use_optimized_kernel=True
    )


def test_integrate_keeps_stock_implicit_without_blocked_scratch(monkeypatch):
    calls = []

    monkeypatch.setattr(
        optimized.mjw_forward, "implicit", lambda m, d: calls.append("stock")
    )
    model = SimpleNamespace(
        opt=SimpleNamespace(
            integrator=optimized.IntegratorType.IMPLICITFAST, disableflags=0
        ),
        is_sparse=False,
        nu=0,
        nv=56,
    )
    data = SimpleNamespace(nworld=2, efc=SimpleNamespace(Ma="ma"))
    scratch = SimpleNamespace(
        implicit_qderiv="qderiv",
        implicit_qacc="qacc",
        implicit_deriv_call=None,
        implicit_call=None,
        implicit_actuator_vel=None,
        implicit_qderiv_pair_mask=None,
        implicit_max_moment_rownnz=0,
        implicit_qh_contributors=(),
        implicit_qld=None,
        implicit_ql_diag_inv=None,
        dense_factor_solve=None,
    )
    optimized._integrate(model, data, scratch, use_optimized_kernel=True)
    assert calls == ["stock"]


def test_tree_solve_does_not_stand_in_for_a_retained_factor():
    """A modifier that keeps the qH factor needs the blocked dense layout."""
    model = SimpleNamespace(nactuator=0)
    scratch = SimpleNamespace(
        implicit_qderiv="qderiv",
        implicit_qacc="qacc",
        dense_factor_solve=None,
        tree_factor_solve="tree-scratch",
        pre_advance_modifier=None,
    )
    assert optimized._has_implicit_factor_scratch(model, scratch)
    scratch.pre_advance_modifier = SimpleNamespace(implicit_factor="implicit-U")
    assert not optimized._has_implicit_factor_scratch(model, scratch)


_MULTI_TREE_MJCF = """
<mujoco>
  <option integrator="implicitfast" timestep="0.004"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="leg" pos="0 0 1">
      <freejoint/>
      <geom type="capsule" size="0.05 0.2"/>
      <body pos="0 0 -0.3">
        <joint type="ball" damping="2"/>
        <geom type="capsule" size="0.04 0.2"/>
        <body pos="0 0 -0.3">
          <joint type="hinge" axis="0 1 0" damping="3"/>
          <geom type="box" size="0.1 0.05 0.03"/>
        </body>
      </body>
    </body>
    <body name="block_r" pos="0.3 0 0.05">
      <freejoint/>
      <geom type="box" size="0.12 0.05 0.02"/>
      <body pos="0.1 0 0">
        <joint type="hinge" axis="0 1 0" damping="1" stiffness="4"/>
        <geom type="box" size="0.03 0.05 0.01"/>
      </body>
    </body>
    <body name="block_l" pos="-0.3 0 0.05">
      <freejoint/>
      <geom type="box" size="0.12 0.05 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.cuda
def test_multi_tree_implicitfast_tree_solve_matches_stock_on_cuda(monkeypatch):
    """Inertia split into several trees has no single dense tile: implicit-fast
    takes the generated tree solve and integrates like upstream's implicit."""
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp

    mj_model = mujoco.MjModel.from_xml_string(_MULTI_TREE_MJCF)
    model = mjw.put_model(mj_model)
    nworld = 16
    rng = np.random.default_rng(0)
    datas = []
    for _ in range(2):
        data = mjw.put_data(
            mj_model, mujoco.MjData(mj_model), nworld=nworld, nconmax=32, njmax=128
        )
        datas.append(data)
    qvel = rng.standard_normal((nworld, mj_model.nv)).astype(np.float32)
    for data in datas:
        wp.copy(data.qvel, wp.array(qvel, device=data.qvel.device))
    scratch = optimized.make_optimized_fwd_position_scratch(
        model, datas[1], "cuda:0", include_optimized_kernel_scratch=True
    )
    assert len(model.M_tiles) > 1
    assert optimized._dense_factor_solve_scratch(scratch) is None
    assert optimized._has_implicit_factor_scratch(model, scratch)
    tree_solves = []
    tree_factor_solve = optimized.tree_ldl.tree_factor_solve

    def spy(*args, **kwargs):
        tree_solves.append(args[0])
        return tree_factor_solve(*args, **kwargs)

    monkeypatch.setattr(optimized.tree_ldl, "tree_factor_solve", spy)
    for _ in range(5):
        mjw.step(model, datas[0])
        optimized.step_optimized_fwd_position(
            model, datas[1], (), scratch=scratch, use_optimized_kernel=True
        )
    wp.synchronize()
    assert tree_solves and all(arg is scratch.tree_factor_solve for arg in tree_solves)
    for name in ("qpos", "qvel"):
        np.testing.assert_allclose(
            getattr(datas[1], name).numpy(),
            getattr(datas[0], name).numpy(),
            rtol=1.0e-4,
            atol=1.0e-5,
            err_msg=name,
        )


def test_implicitfast_routes_qh_factor_to_modifier_workspace(monkeypatch):
    calls = []

    def factor(*args, **kwargs):
        calls.append(("factor", args[2], kwargs.get("factor_workspace")))

    def modifier(model, data, qacc, qvel):
        calls.append(("modifier", qacc))
        return qacc, qvel

    monkeypatch.setattr(optimized, "_factor_solve_i_dense", factor)
    model = SimpleNamespace(
        opt=SimpleNamespace(integrator=optimized.IntegratorType.IMPLICITFAST)
    )
    data = SimpleNamespace(efc=SimpleNamespace(Ma="ma"))
    scratch = SimpleNamespace(
        advance_call=lambda *_args: calls.append(("advance",)),
        pre_advance_modifier=SimpleNamespace(
            name="contact", apply=modifier, implicit_factor="implicit-U"
        ),
        dense_factor_solve="blocked",
    )
    pending = optimized.OptimizedImplicitDerivResult(
        started=True, stream=None, qderiv="qH", qacc="qacc"
    )

    optimized._integrate(
        model,
        data,
        scratch,
        use_optimized_kernel=True,
        pending_implicit_deriv=pending,
    )

    assert calls == [
        ("factor", "qH", "implicit-U"),
        ("modifier", "qacc"),
        ("advance",),
    ]


def test_unblocked_implicitfast_rejects_requested_qh_factor(monkeypatch):
    monkeypatch.setattr(optimized, "_implicitfast_needs_deriv", lambda *_args: True)
    monkeypatch.setattr(
        optimized, "_has_implicit_dense_deriv_scratch", lambda *_args: False
    )
    monkeypatch.setattr(optimized, "_has_implicit_factor_scratch", lambda *_args: False)
    monkeypatch.setattr(
        optimized,
        "factor_solve_i_no_l_store",
        lambda *_args: pytest.fail("unretained qH factor must not be used"),
    )
    model = SimpleNamespace(
        opt=SimpleNamespace(integrator=optimized.IntegratorType.IMPLICITFAST)
    )
    scratch = SimpleNamespace(
        advance_call=lambda *_args: None,
        pre_advance_modifier=SimpleNamespace(
            name="contact",
            apply=lambda _m, _d, qacc, qvel: (qacc, qvel),
            implicit_factor="implicit-U",
        ),
        implicit_deriv_call=lambda *_args: None,
        implicit_qderiv="qderiv",
        implicit_qld="qld",
        implicit_ql_diag_inv="diag",
        implicit_qacc="qacc",
        implicit_qh_contributors=(),
    )

    with pytest.raises(RuntimeError, match="implicit qH factor"):
        optimized._integrate(
            model,
            SimpleNamespace(efc=SimpleNamespace(Ma="ma")),
            scratch,
            use_optimized_kernel=False,
        )


def test_factor_workspace_requires_blocked_dense_path():
    with pytest.raises(RuntimeError, match="blocked dense path"):
        optimized._factor_solve_i_dense(
            "m", "d", "M", "x", "y", None, factor_workspace="workspace"
        )
