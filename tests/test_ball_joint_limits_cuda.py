"""Real solver and graph checks; run on an accessible GPU."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from instantmimic.utils.ball_joint_limits import (
    BallJointLimitCfg,
    configure_ball_joint_limits,
    limit_residuals,
)
from instantmimic.utils.mjw_step_optimized import (
    forward_optimized_fwd_position,
    make_optimized_fwd_position_scratch,
    step_optimized_fwd_position,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def build(sparse=False, count=8, cfg=None):
    host = mujoco.MjModel.from_xml_string(
        '<mujoco><option gravity="0 0 0" timestep=".004166666666666667" integrator="implicitfast"/><worldbody><body><joint name="hip" type="ball" damping=".5"/><inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/></body></worldbody></mujoco>'
    )
    host.opt.jacobian = (
        mujoco.mjtJacobian.mjJAC_SPARSE if sparse else mujoco.mjtJacobian.mjJAC_DENSE
    )
    cfg = cfg or BallJointLimitCfg(joint="hip", swing=(1.2, 1.0), twist=(-0.3, 0.7))
    resolved = configure_ball_joint_limits(host, [cfg])
    m = mjw.put_model(host)
    d = mjw.put_data(
        host, mujoco.MjData(host), nworld=count, njmax=16, nconmax=4, njmax_nnz=48
    )
    scratch = make_optimized_fwd_position_scratch(m, d, ball_joint_limits=resolved)
    return host, m, d, scratch, cfg


@pytest.mark.parametrize("sparse", [False, True])
def test_rows_match_exact_angular_jacobian(sparse):
    cfg = BallJointLimitCfg(
        joint="hip",
        swing=(1.2, 1.0),
        twist=(-0.3, 0.7),
        frame=(0.9238795325, 0.0, 0.3826834324, 0.0),
        neutral=(0.9950041653, 0.0998334166, 0.0, 0.0),
        safety_angle=2.7,
        swing_lower=(-0.8, -0.6),
    )
    host, m, d, scratch, cfg = build(sparse, cfg=cfg)
    q = Rotation.from_euler("XZ", [1.8, 1.1]).as_quat()[[3, 0, 1, 2]]
    state = np.tile(q, (d.nworld, 1)).astype(np.float32)
    vel = np.tile([0.4, -0.3, 0.2], (d.nworld, 1)).astype(np.float32)
    wp.copy(d.qpos, wp.array(state))
    wp.copy(d.qvel, wp.array(vel))
    forward_optimized_fwd_position(m, d, (), scratch)
    r, j = limit_residuals(q, cfg)
    expected = np.flatnonzero(r < 0)
    assert len(expected) > 0
    n = d.nefc.numpy()
    positions = d.efc.pos.numpy()
    jac = d.efc.J.numpy()
    adr = d.efc.J_rowadr.numpy()
    columns = d.efc.J_colind.numpy()
    for world in range(d.nworld):
        assert n[world] == len(expected)
        for k in range(n[world]):
            match = expected[np.argmin(abs(r[expected] - positions[world, k]))]
            actual = jac[world, k, : host.nv] if not sparse else np.zeros(host.nv)
            if sparse:
                a = adr[world, k]
                actual[columns[world, 0, a : a + 3]] = jac[world, 0, a : a + 3]
            np.testing.assert_allclose(positions[world, k], r[match], atol=2e-6)
            np.testing.assert_allclose(actual, j[match], atol=3e-6)
    assert not scratch.ball_limit_buffers.overflow.numpy().any()


@pytest.mark.parametrize("sparse", [False, True])
def test_solver_restores_limits_with_graph_and_quaternion_sign(sparse):
    host, m, d, scratch, cfg = build(sparse)
    q = Rotation.from_euler("XZ", [1.8, 1.1]).as_quat()[[3, 0, 1, 2]].astype(np.float32)
    state = np.tile(q, (d.nworld, 1))
    state[1::2] *= -1
    initial = wp.array(state)
    wp.copy(d.qpos, initial)
    for _ in range(3):
        step_optimized_fwd_position(m, d, (), scratch)
    wp.copy(d.qpos, initial)
    d.qvel.zero_()
    with wp.ScopedCapture() as capture:
        step_optimized_fwd_position(m, d, (), scratch)
    for _ in range(480):
        wp.capture_launch(capture.graph)
    final = d.qpos.numpy()
    velocity = d.qvel.numpy()
    assert np.isfinite(final).all() and np.isfinite(velocity).all()
    for qf in final:
        residual, _ = limit_residuals(qf, cfg)
        assert np.min(residual) > -1e-4
    np.testing.assert_allclose(velocity[::2], velocity[1::2], atol=2e-5)
    assert not scratch.ball_limit_buffers.overflow.numpy().any()
    assert not scratch.ball_limit_buffers.singular.numpy().any()


@pytest.mark.parametrize("sparse", [False, True])
def test_disabled_limits(sparse):
    _, m, d, scratch, _ = build(sparse)
    q = Rotation.from_euler("XZ", [1.8, 1.1]).as_quat()[[3, 0, 1, 2]].astype(np.float32)
    wp.copy(d.qpos, wp.array(np.tile(q, (d.nworld, 1))))
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    forward_optimized_fwd_position(m, d, (), scratch)
    assert not d.nefc.numpy().any()
    assert not scratch.ball_limit_buffers.overflow.numpy().any()


@pytest.mark.parametrize("sparse", [False, True])
def test_joint_rows_precede_contacts_with_equality_and_friction(sparse):
    host = mujoco.MjModel.from_xml_string(
        '<mujoco><option gravity="0 0 0" timestep=".004166666666666667" integrator="implicitfast"/>'
        '<worldbody><geom name="floor" type="plane" size="2 2 .1" contype="2" conaffinity="1"/>'
        '<body name="base" pos="0 0 .05"><freejoint/><geom type="sphere" size=".1" mass="1" contype="1" conaffinity="2"/>'
        '<body><joint name="hip" type="ball" damping=".5" frictionloss=".1"/>'
        '<inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/></body></body></worldbody>'
        '<equality><weld body1="base"/></equality></mujoco>'
    )
    host.opt.jacobian = (
        mujoco.mjtJacobian.mjJAC_SPARSE if sparse else mujoco.mjtJacobian.mjJAC_DENSE
    )
    cfg = BallJointLimitCfg(joint="hip", swing=(1.2, 1.0), twist=(-0.3, 0.7))
    entries = configure_ball_joint_limits(host, [cfg])
    m = mjw.put_model(host)
    d = mjw.put_data(
        host, mujoco.MjData(host), nworld=8, njmax=64, nconmax=8, njmax_nnz=64 * host.nv
    )
    scratch = make_optimized_fwd_position_scratch(m, d, ball_joint_limits=entries)
    state = np.tile(host.qpos0, (d.nworld, 1)).astype(np.float32)
    state[:, 7:] = Rotation.from_euler("XZ", [1.8, 1.1]).as_quat()[[3, 0, 1, 2]]
    wp.copy(d.qpos, wp.array(state))
    forward_optimized_fwd_position(m, d, (), scratch)
    ne, nf, nl, n = d.ne.numpy(), d.nf.numpy(), d.nl.numpy(), d.nefc.numpy()
    rowtypes = d.efc.type.numpy()
    for world in range(d.nworld):
        assert ne[world] == 6 and nf[world] == 3 and nl[world] == 2
        start = ne[world] + nf[world]
        assert np.all(
            rowtypes[world, start : start + nl[world]]
            == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
        )
        assert n[world] > start + nl[world]
        assert np.all(
            rowtypes[world, start + nl[world] : n[world]]
            >= int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
        )
    for _ in range(3):
        step_optimized_fwd_position(m, d, (), scratch, use_optimized_kernel=False)
    with wp.ScopedCapture() as capture:
        step_optimized_fwd_position(m, d, (), scratch, use_optimized_kernel=False)
    for _ in range(240):
        wp.capture_launch(capture.graph)
    assert np.isfinite(d.qpos.numpy()).all() and np.isfinite(d.qvel.numpy()).all()
    assert not scratch.ball_limit_buffers.overflow.numpy().any()
    for qf in d.qpos.numpy()[:, 7:]:
        assert limit_residuals(qf, cfg)[0].min() > -1e-3


@pytest.mark.parametrize("sparse", [False, True])
def test_native_safety_cap_recovers_swing_antipode(sparse):
    _, m, d, scratch, cfg = build(sparse)
    state = np.tile([0.0, 1.0, 0.0, 0.0], (d.nworld, 1)).astype(np.float32)
    wp.copy(d.qpos, wp.array(state))
    forward_optimized_fwd_position(m, d, (), scratch)
    # Off-domain directional rows defer to the well-defined native cap.
    np.testing.assert_array_equal(d.nefc.numpy(), np.ones(d.nworld))
    np.testing.assert_array_equal(scratch.ball_limit_buffers.singular.numpy(), 3)
    for _ in range(480):
        step_optimized_fwd_position(m, d, (), scratch)
    assert np.isfinite(d.qvel.numpy()).all()
    for qf in d.qpos.numpy():
        assert limit_residuals(qf, cfg)[0].min() > -1e-4
    assert not scratch.ball_limit_buffers.overflow.numpy().any()
    assert not scratch.ball_limit_buffers.singular.numpy().any()
