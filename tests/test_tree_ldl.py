"""The generated tree L'DL factor+solve against a dense solve of the same M."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import warp as wp

from instantmimic.utils.mjw_kernels.tree_ldl import (
    create_tree_factor_solve_scratch,
    plan_tree_ldl,
    scratch_for_plan,
    tree_factor_solve,
    tree_ldl_structure,
)

_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]

# A free root with two branches (one with a ball joint) and a second free body:
# nested ancestor rows, a 3-dof joint and two separate trees.
_BRANCHED = """
<mujoco><worldbody>
  <body><freejoint/><geom size=".1" mass="2"/>
    <body pos=".2 0 0"><joint type="hinge" axis="0 1 0"/><geom size=".05" mass=".5"/>
      <body pos=".2 0 0"><joint type="hinge" axis="1 0 0"/><geom size=".04" mass=".3"/></body>
    </body>
    <body pos="-.2 0 0"><joint type="ball"/><geom size=".05" mass=".4"/>
      <body pos="-.2 0 0"><joint type="slide" axis="0 0 1"/><geom size=".03" mass=".2"/></body>
    </body>
  </body>
  <body pos="1 0 0"><freejoint/><geom size=".1" mass="1"/></body>
</worldbody></mujoco>
"""


def _chain(links: int, axis: str) -> str:
    body = ""
    for _ in range(links):
        body = (
            f'<body pos=".1 0 0"><joint type="hinge" axis="{axis}"/>'
            f'<geom size=".03" mass=".2"/>{body}</body>'
        )
    return body


def _limbs(limbs: int, links: int) -> str:
    """A free torso with ``limbs`` hinge chains: the humanoid shape."""
    axes = ["0 1 0", "1 0 0", "0 0 1", "1 1 0"]
    chains = "".join(_chain(links, axes[n % len(axes)]) for n in range(limbs))
    return (
        '<mujoco><worldbody><body><freejoint/><geom size=".1" mass="2"/>'
        f"{chains}</body></worldbody></mujoco>"
    )


_LIMBS = _limbs(4, 6)
_CHAIN = f'<mujoco><worldbody><body><freejoint/><geom size=".1"/>{_chain(30, "0 1 0")}</body></worldbody></mujoco>'


def _model_and_matrices(xml: str, worlds: int):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    rows = np.repeat(np.arange(model.nv), model.M_rownnz)
    sparse, dense = [], []
    for _ in range(worlds):
        data.qpos[:] = model.qpos0
        data.qpos[7:] += rng.normal(0.0, 0.3, model.nq - 7)
        mujoco.mj_normalizeQuat(model, data.qpos)
        mujoco.mj_forward(model, data)
        full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, full)
        # An implicit qH keeps M's sparsity; add a damping-like diagonal.
        full += np.diag(rng.uniform(0.0, 0.2, model.nv))
        sparse.append(full[rows, model.M_colind])
        dense.append(full)
    return model, np.array(sparse, np.float32), np.array(dense)


def _model_arrays(model, device: str = "cpu"):
    return SimpleNamespace(
        nv=model.nv,
        **{
            name: wp.array(
                getattr(model, name).astype(np.int32), dtype=int, device=device
            )
            for name in ("M_rowadr", "M_rownnz", "M_colind")
        },
    )


def _structure(model):
    return tree_ldl_structure(model.M_rowadr, model.M_rownnz, model.M_colind)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("xml", [_BRANCHED, _LIMBS], ids=["branched", "limbs"])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_tree_factor_solve_matches_a_dense_solve(
    device: str, xml: str, groups: int
) -> None:
    # Not a multiple of the 32 worlds a CUDA block serves.
    worlds = 37
    model, sparse, dense = _model_and_matrices(xml, worlds)
    structure = _structure(model)
    plan = plan_tree_ldl(*structure, groups)
    scratch = scratch_for_plan(structure, plan, device)
    y = np.random.default_rng(1).normal(size=(worlds, model.nv)).astype(np.float32)
    x = wp.zeros((worlds, model.nv), dtype=float, device=device)
    tree_factor_solve(
        scratch,
        wp.array(sparse, dtype=float, device=device),
        x,
        wp.array(y, dtype=float, device=device),
    )
    expected = np.stack([np.linalg.solve(a, b) for a, b in zip(dense, y, strict=True)])
    np.testing.assert_allclose(x.numpy(), expected, rtol=2e-4, atol=2e-5)


def test_plans_cover_every_dof_once_and_split_limbs_off_a_trunk() -> None:
    model, _, _ = _model_and_matrices(_LIMBS, 1)
    structure = _structure(model)
    for groups in (1, 2, 4):
        plan = plan_tree_ldl(*structure, groups)
        dofs = sorted([*plan.trunk, *(k for nodes in plan.groups for k in nodes)])
        assert dofs == list(range(model.nv))
    single = plan_tree_ldl(*structure, 1)
    split = plan_tree_ldl(*structure, 4)
    assert not single.trunk
    # The free joint goes to the trunk and one limb runs per group.
    assert split.trunk == tuple(range(6))
    assert sorted(len(nodes) for nodes in split.groups) == [6, 6, 6, 6]
    assert split.span < single.span


def test_tree_solve_is_used_only_where_it_beats_the_dense_cost() -> None:
    limbs, _, _ = _model_and_matrices(_LIMBS, 1)
    chain, _, _ = _model_and_matrices(_CHAIN, 1)
    assert create_tree_factor_solve_scratch(_model_arrays(limbs)) is not None
    # A long chain has nothing to split, would spill, and stays on the dense
    # factorization.
    assert create_tree_factor_solve_scratch(_model_arrays(chain)) is None


def test_tree_structure_requires_nested_ancestor_rows() -> None:
    model, _, _ = _model_and_matrices(_BRANCHED, 1)
    rowadr, rownnz, colind = model.M_rowadr, model.M_rownnz, model.M_colind
    assert tree_ldl_structure(rowadr, rownnz, colind) is not None
    # A row whose ancestor list is not its ancestor's row plus itself.
    broken = colind.copy()
    k = int(np.flatnonzero(rownnz > 2)[-1])
    start = rowadr[k]
    broken[start], broken[start + 1] = broken[start + 1], broken[start]
    assert tree_ldl_structure(rowadr, rownnz, broken) is None
    # The diagonal must close each row.
    shifted = colind.copy()
    shifted[rowadr[k] + rownnz[k] - 1] = 0
    assert tree_ldl_structure(rowadr, rownnz, shifted) is None


def test_tree_factor_solve_rejects_strided_rows() -> None:
    model, sparse, _ = _model_and_matrices(_LIMBS, 2)
    structure = _structure(model)
    scratch = scratch_for_plan(structure, plan_tree_ldl(*structure, 1), "cpu")
    wide = wp.zeros((2, 2 * model.nv), dtype=float, device="cpu")
    with pytest.raises(ValueError, match="contiguous"):
        tree_factor_solve(
            scratch,
            wp.array(sparse, dtype=float, device="cpu"),
            wide[:, ::2],
            wide[:, ::2],
        )


@pytest.mark.cuda
def test_runtime_tree_solve_covers_a_sparse_m_tail() -> None:
    """A tree over 64 dofs (a sparse tail in mujoco_warp) solves like upstream."""
    import mujoco_warp as mjw
    from mujoco_warp._src import smooth

    import instantmimic.utils.mjw_step_optimized as optimized

    # Four 16-hinge limbs on a free torso (70 dofs) beside an 8-hinge tree.
    xml = (
        _limbs(4, 16)
        .replace("<mujoco>", '<mujoco><option><flag contact="disable"/></option>')
        .replace(
            "</worldbody>",
            f'<body pos="2 0 0">{_chain(8, "0 1 0")}</body></worldbody>',
        )
    )
    m_mj = mujoco.MjModel.from_xml_string(xml)
    m_mj.dof_armature[:] = 0.01
    d_mj = mujoco.MjData(m_mj)
    d_mj.qpos[7:] = np.linspace(-0.3, 0.3, m_mj.nq - 7)
    mujoco.mj_forward(m_mj, d_mj)
    dense = np.zeros((m_mj.nv, m_mj.nv))
    mujoco.mj_fullM(m_mj, d_mj, dense)
    nworld = 37
    with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(m_mj)
        d = mjw.put_data(m_mj, d_mj, nworld=nworld, nconmax=0, njmax=0)
        assert optimized._has_sparse_m_tail(m, d)
        scratch = create_tree_factor_solve_scratch(m)
        assert scratch is not None
        y = wp.array(
            np.random.default_rng(3).normal(size=(nworld, m.nv)).astype(np.float32),
            dtype=float,
        )
        upstream = wp.zeros((nworld, m.nv), dtype=float)
        smooth.factor_solve_i(m, d, d.M, d.qLD, d.qLDiagInv, upstream, y)
        smooth_solution = wp.zeros((nworld, m.nv), dtype=float)
        optimized._factor_solve_i_dense(
            m, d, d.M, smooth_solution, y, tree_factor_solve=scratch
        )
        implicit_solution = wp.zeros((nworld, m.nv), dtype=float)
        optimized.factor_solve_i_no_l_store(
            m,
            d,
            d.M,
            wp.empty_like(d.qLD),
            wp.empty((nworld, m.nv), dtype=float),
            implicit_solution,
            y,
            tree_factor_solve=scratch,
        )
        exact = np.linalg.solve(dense, y.numpy().astype(np.float64).T).T
        scale = float(np.abs(exact).max())
        upstream_error = float(np.abs(upstream.numpy() - exact).max()) / scale
        for solution in (smooth_solution, implicit_solution):
            error = float(np.abs(solution.numpy() - exact).max()) / scale
            # No worse than the upstream sparse factorization in float32.
            assert error < max(2.0 * upstream_error, 1e-5)
