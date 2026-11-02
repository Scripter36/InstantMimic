"""Static-pattern sparse L'DL of the fused Newton's Hessian (mjw_kernels/sparse_hessian.py)."""

from __future__ import annotations

import re

import mujoco
import numpy as np
import pytest
import warp as wp

from instantmimic.utils.mjw_kernels import sparse_hessian

LEG = """
<body name="thigh{s}" pos="0 {y} -0.2"><joint type="ball"/>
  <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.05" contype="2" conaffinity="1"/>
  <body name="shin{s}" pos="0 0 -0.35"><joint type="hinge" axis="0 1 0" range="-2 0"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.04" contype="2" conaffinity="1"/>
    <body name="foot{s}" pos="0 0 -0.35"><joint type="hinge" axis="0 1 0"/><joint type="hinge" axis="1 0 0"/>
      <geom type="box" pos="0.05 0 -0.02" size="0.1 0.05 0.02" contype="{foot}" conaffinity="1"/>
    </body>
  </body>
</body>"""
ARM = """
<body name="arm{s}" pos="0 {y} 0.15"><joint type="hinge" axis="0 1 0"/>
  <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03" contype="0" conaffinity="0"/>
  <body name="fore{s}" pos="0 0 -0.25"><joint type="hinge" axis="0 1 0"/><joint type="hinge" axis="1 0 0"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03" contype="0" conaffinity="0"/>
  </body>
</body>"""
# torso (6) + legs 2 x 6 + arms 2 x 3 + two free boxes (12): nv 36, nv_pad 48. Legs
# touch the floor; box A also touches the left foot (a cross-tree coupling), box B
# only the floor.
XML = f"""
<mujoco>
  <option timestep="0.005" cone="pyramidal" jacobian="dense"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1" contype="1" conaffinity="6"/>
    <body name="torso" pos="0 0 0.97"><freejoint/>
      <geom type="box" size="0.15 0.1 0.2" contype="0" conaffinity="0"/>
      {LEG.format(s="L", y=0.1, foot=10)}{LEG.format(s="R", y=-0.1, foot=2)}
      {ARM.format(s="L", y=0.2)}{ARM.format(s="R", y=-0.2)}
    </body>
    <body name="boxA" pos="0.15 0.1 0.12"><freejoint/>
      <geom type="box" size="0.08 0.08 0.08" contype="4" conaffinity="9"/></body>
    <body name="boxB" pos="-0.3 -0.1 0.12"><freejoint/>
      <geom type="box" size="0.08 0.08 0.08" contype="4" conaffinity="1"/></body>
  </worldbody>
</mujoco>
"""


def _crates_xml(n: int) -> str:
    """XML plus n free crates that touch only the floor: 6 DOFs each, so 4 crates
    give nv 60 (nv_pad 64) and 6 give nv 72 (nv_pad 80), both on the fused Newton's
    two-warp block with a sparse pattern."""
    crates = "".join(
        f'<body name="crate{i}" pos="{0.6 + 0.25 * i} 0.6 0.1"><freejoint/>'
        '<geom type="box" size="0.05 0.05 0.05" contype="4" conaffinity="1"/></body>'
        for i in range(n)
    )
    return XML.replace("  </worldbody>", crates + "\n  </worldbody>")


def _cpu_model(xml: str = XML):
    from instantmimic.utils.mjw_model import put_model

    mjm = mujoco.MjModel.from_xml_string(xml)
    with wp.ScopedDevice("cpu"):
        return mjm, put_model(mjm)


def test_symbolic_fill_contains_the_numeric_factor():
    """Every nonzero of a numeric L'DL lies in the symbolic pattern (fill included)."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        n = 30
        adj = [set() for _ in range(n)]
        for _ in range(60):
            a, b = rng.integers(0, n, 2)
            if a != b:
                adj[a].add(int(b))
                adj[b].add(int(a))
        rowstruct, colstruct, levels = sparse_hessian._symbolic(adj)
        H = np.eye(n) * n
        for a in range(n):
            for b in adj[a]:
                H[a, b] = rng.normal()
        H = (H + H.T) / 2 + n * np.eye(n)
        L = np.linalg.cholesky(H)
        for j in range(n):
            for i in range(j + 1, n):
                if abs(L[i, j]) > 1e-12:
                    assert i in colstruct[j], (i, j)
        # a column is eliminated after every column it depends on
        level = {k: lv for lv, cols in enumerate(levels) for k in cols}
        for i in range(n):
            assert all(level[k] < level[i] for k in rowstruct[i])


def test_pattern_couples_tree_groups_equalities_and_collision_pairs():
    xml = XML.replace(
        "</worldbody>",
        '</worldbody><equality><connect body1="boxA" body2="boxB" anchor="0 0 0"/></equality>',
    )
    mjm, m = _cpu_model(xml)
    adj = sparse_hessian.hessian_pattern(
        m, [[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "footL")]]
    )

    def dofs(body):
        b = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, body)
        return list(range(mjm.body_dofadr[b], mjm.body_dofadr[b] + mjm.body_dofnum[b]))

    torso, foot_l, foot_r, arm_l = (
        dofs("torso"),
        dofs("footL"),
        dofs("footR"),
        dofs("armL"),
    )
    box_a, box_b = dofs("boxA"), dofs("boxB")
    # tree ancestry
    assert set(torso) <= adj[foot_l[0]]
    # sibling branches stay uncoupled: no row touches both arms' or arm and foot dofs
    assert arm_l[0] not in adj[foot_l[0]]
    # collision candidates: box A against the left foot couples it with that foot's
    # chain, not with the right foot
    assert set(foot_l) <= adj[box_a[0]]
    assert foot_r[0] not in adj[box_a[0]]
    # equality connect couples the two boxes
    assert set(box_b) <= adj[box_a[0]]


def test_site_welds_couple_the_sites_bodies():
    xml = (
        XML.replace(
            '<geom type="box" size="0.08 0.08 0.08" contype="4" conaffinity="9"/></body>',
            '<geom type="box" size="0.08 0.08 0.08" contype="4" conaffinity="9"/><site name="sa"/></body>',
        )
        .replace(
            '<geom type="box" size="0.08 0.08 0.08" contype="4" conaffinity="1"/></body>',
            '<geom type="box" size="0.08 0.08 0.08" contype="4" conaffinity="1"/><site name="sb"/></body>',
        )
        .replace(
            "</worldbody>",
            '</worldbody><equality><weld site1="sa" site2="sb"/></equality>',
        )
    )
    mjm, m = _cpu_model(xml)
    adj = sparse_hessian.hessian_pattern(m, [])
    a = mjm.body_dofadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "boxA")]
    b = mjm.body_dofadr[mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "boxB")]
    assert set(range(b, b + 6)) <= adj[a]


PAD_XML = """
<mujoco>
  <option timestep="0.00833333333" cone="pyramidal" jacobian="{jacobian}"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1" contype="2" conaffinity="1"/>
    <body name="heel" pos="0 0 0.04"><freejoint/>
      <geom type="box" size="0.045 0.038 0.025" mass="5" contype="0" conaffinity="0"/>
      <flexcomp name="pad" type="grid" dim="3" dof="full" count="3 3 2" spacing="0.045 0.038 0.012"
        pos="0 0 -0.031" mass="0.05" radius="0.001">
        <edge equality="true"/><contact contype="1" conaffinity="2" condim="3" selfcollide="none"/>
        <pin gridrange="0 0 1 2 2 1"/></flexcomp>
    </body>
  </worldbody>
</mujoco>
"""


def test_flex_elements_couple_their_vertex_bodies():
    mjm, m = _cpu_model(PAD_XML.format(jacobian="sparse"))
    adj = sparse_hessian.hessian_pattern(m, [])
    for f in range(mjm.nflex):
        dim = mjm.flex_dim[f]
        base = mjm.flex_vertadr[f]
        for e in range(mjm.flex_elemnum[f]):
            verts = mjm.flex_elem[mjm.flex_elemdataadr[f] + e * (dim + 1) :][: dim + 1]
            dofs = set()
            for v in verts:
                b = mjm.flex_vertbodyid[base + v]
                dofs |= set(
                    range(mjm.body_dofadr[b], mjm.body_dofadr[b] + mjm.body_dofnum[b])
                )
            for x in dofs:
                assert dofs - {x} <= adj[x]


def _tables(snippet: str) -> dict[str, np.ndarray]:
    found = re.findall(r"static const [a-z ]+ (SH_\w+)\[\] = \{([^}]*)\};", snippet)
    return {
        name: np.array([int(v.rstrip("u")) for v in vals.split(",")], dtype=np.int64)
        for name, vals in found
    }


def _run_generated_tables(m, groups, H, g):
    """Execute a plan's generated factor and solve tables in NumPy, each level as
    one parallel sweep (every lane reads the level's input snapshot), and assert
    that no level writes what another lane of it reads or writes."""
    plan = sparse_hessian.build_plan(m, groups, 64, "fused")
    T = _tables(plan.factor_snippet) | _tables(plan.init_snippet)
    nv, nfac = plan.nv, plan.nslots - plan.nv
    N, iD = np.zeros(plan.nslots), np.zeros(nv)
    pair = T["SH_PAIR"].reshape(nv, nv)
    for s in range(nfac):
        a, b = T["SH_SA"][s], T["SH_SB"][s]
        assert pair[a, b] == s
        N[s] = H[a, b]
    # H's nonzeros, the update's pair table and the row check's mask agree
    words = (nv + 31) // 32
    for a in range(nv):
        for b in range(a, nv):
            in_mask = (T["SH_PM"][a * words + b // 32] >> (b % 32)) & 1
            assert bool(in_mask) == (pair[a, b] >= 0 or a == b)
            assert H[a, b] == 0.0 or pair[a, b] >= 0
    orig = T["SH_ORIG"]
    N[nfac:] = g[orig]
    LP, ES, EC, RP, RAB, RKL = (
        T[k] for k in ("SH_LP", "SH_ES", "SH_EC", "SH_RP", "SH_RAB", "SH_RKL")
    )
    for lvl in range(len(LP) - 1):
        n0, d0 = N.copy(), iD.copy()
        written, read = set(), set()
        for e in range(LP[lvl], LP[lvl + 1]):
            v = 0.0
            for r in range(RP[e], RP[e + 1]):
                a, b, k, ln = (
                    RAB[r] & 0xFFFF,
                    RAB[r] >> 16,
                    RKL[r] & 0xFFFF,
                    RKL[r] >> 16,
                )
                v += float(np.sum(n0[a : a + ln] * n0[b : b + ln] * d0[k : k + ln]))
                read |= {
                    *range(a, a + ln),
                    *range(b, b + ln),
                    *range(-k - ln, -k),
                }
            assert ES[e] not in written
            written.add(ES[e])
            N[ES[e]] = n0[ES[e]] - v
            if EC[e] >= 0:
                iD[EC[e]] = 1.0 / N[ES[e]]
                written.add(-1 - EC[e])
        assert not written & read
    BL, BS, BI, BJ = (T[k] for k in ("SH_BL", "SH_BS", "SH_BI", "SH_BJ"))
    for lvl in range(len(BL) - 1):
        z = N[nfac:].copy()
        src, dst = set(), set()
        for p in range(BL[lvl], BL[lvl + 1]):
            i, j = BI[p], BJ[p]
            assert j not in dst, "two pushes to one unknown in a level"
            src.add(i)
            dst.add(j)
            N[nfac + j] = z[j] - N[BS[p]] * (z[i] * iD[i])
        assert not src & dst
    x = np.zeros(nv)
    x[orig] = N[nfac:] * iD
    return plan, x


def test_generated_factor_and_solve_tables_solve_h():
    """The generated level tables, run as parallel sweeps, solve H x = g on patterns
    with cross-tree contact groups (fill), without a race inside any level."""
    mjm, m = _cpu_model()

    def body(name):
        return mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, name)

    cases = [
        [],
        [[body("footL"), body("boxB")]],
        [[body("footL"), body("footR"), body("boxB")], [body("foreR"), body("boxA")]],
    ]
    rng = np.random.default_rng(5)
    for groups in cases:
        adj = sparse_hessian.hessian_pattern(m, groups)
        nv = int(m.nv)
        H = np.zeros((nv, nv))
        for a in range(nv):
            for b in adj[a]:
                if a < b:
                    H[a, b] = H[b, a] = rng.normal()
        H += np.diag(np.abs(H).sum(1) + 1.0 + rng.random(nv))
        g = rng.normal(size=nv)
        plan, x = _run_generated_tables(m, groups, H, g)
        assert plan.levels > 1
        np.testing.assert_allclose(x, np.linalg.solve(H, g), rtol=1e-10, atol=1e-12)


def test_plan_requires_declared_groups_and_the_fused_newton():
    _, m = _cpu_model()
    plan, reason = sparse_hessian.plan_for_model(m, None, None, 64)
    assert plan is None and "declare" in reason
    # the fused Newton is CUDA-only, so a CPU model never qualifies
    plan, reason = sparse_hessian.plan_for_model(m, object(), [], 64)
    assert plan is None and "fused" in reason


# ---------------------------------------------------------------------------- CUDA


def _cuda_setup(
    nworld: int, plan_groups, drop_pairs: bool = False, jacobian="dense", xml=None
):
    import mujoco_warp as mjw

    from instantmimic.utils import mjw_step_optimized as runtime
    from instantmimic.utils.mjw_kernels import newton_fusion
    from instantmimic.utils.mjw_model import put_model

    if xml is None:
        xml = XML.replace('jacobian="dense"', f'jacobian="{jacobian}"')
    mjm = mujoco.MjModel.from_xml_string(xml)
    with wp.ScopedDevice("cuda"):
        m = put_model(mjm)
        m.opt.warn_overflow = False
        data = mujoco.MjData(mjm)
        d = mjw.put_data(mjm, data, nworld=nworld, nconmax=64, njmax=256)
        rng = np.random.default_rng(11)
        d.qvel.assign((rng.normal(0, 0.3, (nworld, mjm.nv))).astype(np.float32))
        scratch = runtime.make_optimized_fwd_position_scratch(
            m, d, "cuda", include_optimized_kernel_scratch=True
        )
        ctx = scratch.solver_context
        assert newton_fusion.eligible(m, ctx) == (
            jacobian == "dense" and mjm.nflex == 0
        )
        plan = None
        if plan_groups is not None:
            if drop_pairs:
                real = m.nxn_geom_pair_filtered
                m.nxn_geom_pair_filtered = wp.zeros((0, 2), dtype=int, device="cuda")
            plan, reason = sparse_hessian.plan_for_model(
                m, ctx, plan_groups, newton_fusion.newton_block_dim(m)
            )
            if drop_pairs:
                m.nxn_geom_pair_filtered = real
            assert plan is not None, reason
            if plan.path == "fused":
                sparse_hessian.violation_counter(ctx.hfactor, plan).zero_()
        ctx.sparse_hessian_plan = plan
    return runtime, m, d, scratch, plan


@pytest.mark.cuda
@pytest.mark.parametrize("crates", [0, 4, 6])
def test_sparse_hessian_steps_match_the_dense_factor(crates):
    nworld = 8
    xml = _crates_xml(crates)
    runtime, m, dd, sd, _ = _cuda_setup(nworld, None, xml=xml)
    _, ms, ds, ss, plan = _cuda_setup(nworld, [], xml=xml)
    assert plan.density < sparse_hessian.MAX_DENSITY
    assert plan.block == (64 if crates else 32)
    synced = ("qpos", "qvel", "qacc_warmstart", "act", "time")
    contacts = 0
    with wp.ScopedDevice("cuda"):
        for _ in range(40):
            runtime.step_optimized_fwd_position(
                m, dd, (), sd, use_optimized_kernel=True
            )
            runtime.step_optimized_fwd_position(
                ms, ds, (), ss, use_optimized_kernel=True
            )
            wp.synchronize()
            qa, qb = dd.qacc.numpy(), ds.qacc.numpy()
            scale = np.abs(qa).max(axis=1, keepdims=True) + 1.0
            assert np.all(np.abs(qa - qb) <= 2e-4 * scale)
            assert np.array_equal(dd.solver_niter.numpy(), ds.solver_niter.numpy())
            contacts += int(ds.nacon.numpy()[0])
            for name in synced:
                wp.copy(getattr(ds, name), getattr(dd, name))
    assert contacts > 0
    assert not sparse_hessian.violations(ss.solver_context.hfactor, plan).any()


@pytest.mark.cuda
@pytest.mark.parametrize("crates", [0, 6])
def test_sparse_hessian_solve_is_deterministic(crates):
    """Repeated solves of one state give bitwise identical results, eager and
    replayed from a CUDA graph (contact order from collision can differ between
    runs, so whole steps are not compared)."""
    nworld = 8
    runtime, m, d, scratch, _ = _cuda_setup(nworld, [], xml=_crates_xml(crates))
    ctx = scratch.solver_context
    with wp.ScopedDevice("cuda"):
        for _ in range(10):
            runtime.step_optimized_fwd_position(
                m, d, (), scratch, use_optimized_kernel=True
            )
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
        results = []
        for _ in range(3):
            for dst, src in saved:
                wp.copy(dst, src)
            runtime.solve_with_scratch(m, d, scratch, use_optimized_kernel=True)
            wp.synchronize()
            results.append((d.qacc.numpy().copy(), d.efc.force.numpy().copy()))
        with wp.ScopedCapture() as capture:
            runtime.solve_with_scratch(m, d, scratch, use_optimized_kernel=True)
        for _ in range(2):
            for dst, src in saved:
                wp.copy(dst, src)
            wp.capture_launch(capture.graph)
            wp.synchronize()
            results.append((d.qacc.numpy().copy(), d.efc.force.numpy().copy()))
    for qacc, force in results[1:]:
        assert np.array_equal(qacc, results[0][0])
        assert np.array_equal(force, results[0][1])


@pytest.mark.cuda
@pytest.mark.parametrize("crates", [0, 6])
def test_sparse_hessian_counts_rows_outside_the_pattern(crates):
    """A plan that misses the box-foot collision couplings reports the violation, and
    the counter accumulates across solves until it is zeroed."""
    nworld = 8
    runtime, m, d, scratch, plan = _cuda_setup(
        nworld, [], drop_pairs=True, xml=_crates_xml(crates)
    )
    hfactor = scratch.solver_context.hfactor
    totals = []
    with wp.ScopedDevice("cuda"):
        for _ in range(40):
            runtime.step_optimized_fwd_position(
                m, d, (), scratch, use_optimized_kernel=True
            )
            totals.append(int(sparse_hessian.violations(hfactor, plan).sum()))
        # cumulative: the total keeps growing past any single solve's count
        steps = np.diff([0, *totals])
        assert totals == sorted(totals) and totals[-1] > steps.max() > 0
        sparse_hessian.violation_counter(hfactor, plan).zero_()
        assert not sparse_hessian.violations(hfactor, plan).any()
        for _ in range(40):
            runtime.step_optimized_fwd_position(
                m, d, (), scratch, use_optimized_kernel=True
            )
    assert sparse_hessian.violations(hfactor, plan).sum() > 0


@pytest.mark.cuda
@pytest.mark.parametrize("model", ["tree", "flex_pad"])
def test_upstream_iteration_with_sparse_factor_matches_dense(model, monkeypatch):
    """Sparse Jacobians leave the fused Newton; the upstream iteration then uses the
    sparse factor on its dense H, matching the dense factor step by step (the small
    test models stand in for the nv >= 80 models the path is enabled for)."""
    monkeypatch.setattr(sparse_hessian, "NONFUSED_MIN_NV", 0)
    monkeypatch.setattr(sparse_hessian, "MAX_DENSITY", 1.0)
    nworld = 8
    xml = PAD_XML.format(jacobian="sparse") if model == "flex_pad" else None
    runtime, m, dd, sd, _ = _cuda_setup(nworld, None, jacobian="sparse", xml=xml)
    _, ms, ds, ss, plan = _cuda_setup(nworld, [], jacobian="sparse", xml=xml)
    assert plan.path == "nonfused"
    adj = sparse_hessian.hessian_pattern(ms, [])
    synced = ("qpos", "qvel", "qacc_warmstart", "act", "time")
    contacts = 0
    with wp.ScopedDevice("cuda"):
        for _ in range(40):
            runtime.step_optimized_fwd_position(
                m, dd, (), sd, use_optimized_kernel=True
            )
            runtime.step_optimized_fwd_position(
                ms, ds, (), ss, use_optimized_kernel=True
            )
            wp.synchronize()
            qa, qb = dd.qacc.numpy(), ds.qacc.numpy()
            scale = np.abs(qa).max(axis=1, keepdims=True) + 1.0
            assert np.all(np.abs(qa - qb) <= 2e-4 * scale)
            assert np.array_equal(dd.solver_niter.numpy(), ds.solver_niter.numpy())
            # the dense H the upstream assembly built stays inside the pattern
            assert sparse_hessian.pattern_misses(ss.solver_context.h.numpy(), adj) == 0
            contacts += int(ds.nacon.numpy()[0])
            for name in synced:
                wp.copy(getattr(ds, name), getattr(dd, name))
    assert contacts > 0
