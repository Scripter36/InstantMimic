import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from instantmimic.utils.ball_joint_limits import (
    BallJointLimitCfg,
    configure_ball_joint_limits,
    limit_residuals,
)


def model():
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="hip" type="ball"/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )


def test_configuration_opt_in_and_safety_cap():
    m = model()
    before = m.jnt_limited.copy()
    assert configure_ball_joint_limits(m, []) == ()
    np.testing.assert_array_equal(m.jnt_limited, before)
    resolved = configure_ball_joint_limits(
        m, [BallJointLimitCfg(joint="hip", swing=(1.8, 1.2), twist=(-0.5, 0.8))]
    )
    assert len(resolved) == 1
    assert m.jnt_limited[0]
    assert 0 < m.jnt_range[0, 1] < np.pi


@pytest.mark.parametrize(
    "kwargs",
    [
        {"swing": (0.0, 1.0)},
        {"twist": (1.0, -0.5)},
        {"twist": (-np.pi, np.pi)},
        {"frame": (0.0, 0.0, 0.0, 0.0)},
        {"safety_angle": np.pi},
        {"margin": -0.1},
        {"swing_lower": (0.1, -0.4)},
    ],
)
def test_invalid_limit_rejected(kwargs):
    values = dict(joint="hip", swing=(1.0, 1.0), twist=(-0.5, 0.5))
    values.update(kwargs)
    with pytest.raises(ValueError):
        configure_ball_joint_limits(model(), [BallJointLimitCfg(**values)])


def test_residual_jacobian_sign_and_frame():
    cfg = BallJointLimitCfg(
        joint="hip",
        swing=(1.2, 0.8),
        twist=(-0.3, 0.7),
        frame=(0.9238795325, 0.0, 0.3826834324, 0.0),
        neutral=(0.9800665778, 0.1986693308, 0.0, 0.0),
        swing_lower=(-0.6, -0.4),
    )
    rng = np.random.default_rng(5)
    for _ in range(100):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        r, j = limit_residuals(q, cfg)
        rm, jm = limit_residuals(-q, cfg)
        np.testing.assert_allclose(r, rm, atol=1e-12)
        np.testing.assert_allclose(j, jm, atol=1e-12)
        rot = Rotation.from_quat(q[[1, 2, 3, 0]])
        for k in range(3):
            v = np.eye(3)[k] * 1e-6
            qp = (rot * Rotation.from_rotvec(v)).as_quat()[[3, 0, 1, 2]]
            qm = (rot * Rotation.from_rotvec(-v)).as_quat()[[3, 0, 1, 2]]
            fd = (limit_residuals(qp, cfg)[0] - limit_residuals(qm, cfg)[0]) / 2e-6
            np.testing.assert_allclose(j[:, k], fd, atol=2e-8)


def test_asymmetric_twist_and_swing_boundary():
    cfg = BallJointLimitCfg(joint="hip", swing=(1.2, 0.8), twist=(-0.3, 0.7))
    for angle, valid in [(-0.4, False), (-0.2, True), (0.6, True), (0.8, False)]:
        q = Rotation.from_rotvec([0.0, 0.0, angle]).as_quat()[[3, 0, 1, 2]]
        assert (limit_residuals(q, cfg)[0][1:].min() >= 0) == valid
    for axis, cap in [(0, 1.2), (1, 0.8)]:
        for offset, valid in [(-0.01, True), (0.01, False)]:
            v = np.eye(3)[axis] * (cap + offset)
            q = Rotation.from_rotvec(v).as_quat()[[3, 0, 1, 2]]
            assert (limit_residuals(q, cfg)[0][0] >= 0) == valid


def test_asymmetric_swing_preserves_signed_bounds():
    cfg = BallJointLimitCfg(
        joint="hip", swing=(0.6, 0.6), swing_lower=(-0.4, -0.4), twist=(-0.4, 0.6)
    )
    for axis in [0, 1]:
        for angle, valid in [
            (-0.41, False),
            (-0.39, True),
            (0.59, True),
            (0.61, False),
        ]:
            q = Rotation.from_rotvec(np.eye(3)[axis] * angle).as_quat()[[3, 0, 1, 2]]
            assert (limit_residuals(q, cfg)[0][0] >= 0) == valid


def test_constraint_stage_insertion_is_local_and_ordered(monkeypatch):
    from types import SimpleNamespace

    import warp as wp

    from instantmimic.utils import mjw_step_optimized as optimized

    calls = []
    marker = object()
    monkeypatch.setattr(optimized.constraint, "_limit_ball", lambda *_: marker)
    monkeypatch.setattr(wp, "launch", lambda kernel, **_: calls.append(kernel))

    namespace = {"wp": wp, "optimized": optimized}
    exec(
        "def upstream(m, d):\n"
        "    wp.empty((d.nworld,), dtype=int)\n"
        "    wp.launch('equality', dim=1)\n"
        "    if not m.opt.disableflags:\n"
        "        wp.launch(optimized.constraint._limit_ball(False, True), dim=1)\n"
        "    wp.launch('contact', dim=1)\n",
        namespace,
    )
    upstream = namespace["upstream"]

    monkeypatch.setattr(optimized.constraint, "make_constraint", upstream)
    buffers = SimpleNamespace(
        overflow=SimpleNamespace(zero_=lambda: None),
        singular=SimpleNamespace(zero_=lambda: None),
        launch=lambda *_: calls.append("directional"),
    )
    m = SimpleNamespace(is_sparse=False, opt=SimpleNamespace(solver=2, disableflags=0))
    d = SimpleNamespace(nworld=1)
    wrapper = optimized._BallLimitConstraintCall(m, d, object(), buffers)
    wrapper(m, d)
    assert calls == ["equality", marker, "directional", "contact"]
    assert optimized.constraint.make_constraint is upstream
    calls.clear()
    m.opt.disableflags = int(optimized.DisableBit.LIMIT)
    wrapper(m, d)
    assert calls == ["equality", "contact"]


def test_large_violations_have_restorative_jacobians():
    cfg = BallJointLimitCfg(joint="hip", swing=(1.2, 1.0), twist=(-0.3, 0.7))
    # A polynomial inequality alone can have an extremum in its forbidden
    # region. Solver rows must still push toward the permitted interval.
    for angle in [1.3, 2.4, 2.8]:
        q = Rotation.from_rotvec([angle, 0.0, 0.0]).as_quat()[[3, 0, 1, 2]]
        r, j = limit_residuals(q, cfg)
        assert r[0] < 0 and j[0, 0] < 0
    for angle in [-2.9, -0.4]:
        q = Rotation.from_rotvec([0.0, 0.0, angle]).as_quat()[[3, 0, 1, 2]]
        r, j = limit_residuals(q, cfg)
        assert r[1] < 0 and j[1, 2] > 0
    for angle in [0.8, 2.9]:
        q = Rotation.from_rotvec([0.0, 0.0, angle]).as_quat()[[3, 0, 1, 2]]
        r, j = limit_residuals(q, cfg)
        assert r[2] < 0 and j[2, 2] < 0


def test_configuration_failure_is_atomic_and_preserves_existing_cap():
    m = model()
    m.jnt_limited[0] = True
    m.jnt_range[0] = [0.0, 2.0]
    good = BallJointLimitCfg(joint="hip", swing=(1.0, 1.0), twist=(-0.5, 0.5))
    bad = BallJointLimitCfg(joint="missing", swing=(1.0, 1.0), twist=(-0.5, 0.5))
    with pytest.raises(ValueError):
        configure_ball_joint_limits(m, [good, bad])
    np.testing.assert_array_equal(m.jnt_range[0], [0.0, 2.0])
    configure_ball_joint_limits(m, [good])
    assert m.jnt_range[0, 1] == 2.0
