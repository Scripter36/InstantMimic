"""Host configuration and SO(3) residuals for directional ball constraints."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class BallJointLimitCfg:
    """Angles are radians; frame's local Z is twist, X/Y are swing axes.

    ``neutral`` is the joint quaternion at zero anatomical rotation. ``frame``
    rotates anatomical coordinates into the joint's local coordinates. Existing
    scalar ball caps remain active as safety constraints. An uncapped ball gets
    a 170 degree cap; ``safety_angle`` explicitly replaces that scalar cap.
    """

    joint: str
    swing: tuple[float, float]
    twist: tuple[float, float]
    neutral: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    frame: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    safety_angle: float | None = None
    margin: float = 0.0
    solref: tuple[float, float] = (0.02, 1.0)
    solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
    swing_lower: tuple[float, float] | None = None


@dataclass(frozen=True)
class ResolvedBallJointLimit:
    cfg: BallJointLimitCfg
    joint_id: int
    qposadr: int
    dofadr: int
    minimum_h: float


def _quaternion(values: Sequence[float], name: str) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError(f"{name} must be a finite nonzero wxyz quaternion")
    return q / np.linalg.norm(q)


def _validate(cfg: BallJointLimitCfg) -> None:
    if len(cfg.swing) != 2 or not all(
        math.isfinite(v) and 0 < v < math.pi for v in cfg.swing
    ):
        raise ValueError("swing requires two angles strictly between 0 and pi")
    if len(cfg.twist) != 2 or not all(math.isfinite(v) for v in cfg.twist):
        raise ValueError("twist requires two finite angles")
    low, high = cfg.twist
    if not -math.pi < low < high < math.pi:
        raise ValueError("twist requires -pi < lower < upper < pi")
    if cfg.swing_lower is not None and (
        len(cfg.swing_lower) != 2
        or not all(math.isfinite(v) and -math.pi < v < 0 for v in cfg.swing_lower)
    ):
        raise ValueError("swing_lower requires two angles strictly between -pi and 0")
    if cfg.safety_angle is not None and not (
        math.isfinite(cfg.safety_angle) and 0 < cfg.safety_angle < math.pi
    ):
        raise ValueError("safety_angle must be strictly between 0 and pi")
    if not math.isfinite(cfg.margin) or cfg.margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if len(cfg.solref) != 2 or not all(math.isfinite(v) for v in cfg.solref):
        raise ValueError("solref requires two finite values")
    if not (
        (cfg.solref[0] > 0 and cfg.solref[1] > 0)
        or (cfg.solref[0] <= 0 and cfg.solref[1] <= 0)
    ):
        raise ValueError(
            "solref must use positive time/damping or nonpositive direct format"
        )
    if len(cfg.solimp) != 5 or not all(math.isfinite(v) for v in cfg.solimp):
        raise ValueError("solimp requires five finite values")
    _quaternion(cfg.frame, "frame")
    _quaternion(cfg.neutral, "neutral")


def configure_ball_joint_limits(
    model: mujoco.MjModel,
    entries: Sequence[BallJointLimitCfg | Mapping[str, object]],
    *,
    enforcement: str = "solver",
) -> tuple[ResolvedBallJointLimit, ...]:
    """Resolve directional limits and configure mutually exclusive enforcement.

    ``solver`` retains the historical directional constraint rows and a native
    scalar safety cap. ``force`` disables the native scalar cap; a task can then
    consume the returned metadata to apply explicit generalized forces without
    double enforcement.
    """
    if enforcement not in {"solver", "force"}:
        raise ValueError(
            "ball joint limit enforcement must be 'solver' or 'force', got "
            f"{enforcement!r}"
        )
    resolved = []
    caps = []
    seen = set()
    for entry in entries:
        cfg = (
            entry
            if isinstance(entry, BallJointLimitCfg)
            else BallJointLimitCfg(**dict(entry))
        )
        _validate(cfg)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg.joint)
        if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_BALL:
            raise ValueError(f"{cfg.joint!r} must name an existing ball joint")
        if jid in seen:
            raise ValueError(f"duplicate ball limit for {cfg.joint!r}")
        seen.add(jid)
        cap = cfg.safety_angle
        if cap is None:
            current = float(max(model.jnt_range[jid]))
            cap = (
                current
                if model.jnt_limited[jid] and 0 < current < math.pi
                else math.radians(170)
            )
        # A total rotation cap around identity must also keep neutral away from
        # the swing antipode. Triangle inequality bounds relative rotation.
        neutral_angle = 2 * math.acos(
            min(1.0, abs(float(_quaternion(cfg.neutral, "neutral")[0])))
        )
        if cap + neutral_angle >= math.radians(175):
            raise ValueError(
                "safety_angle + neutral rotation must remain below 175 degrees"
            )
        resolved.append(
            ResolvedBallJointLimit(
                cfg,
                jid,
                int(model.jnt_qposadr[jid]),
                int(model.jnt_dofadr[jid]),
                math.cos((cap + neutral_angle) / 2) ** 2,
            )
        )
        caps.append(cap)
    for item, cap in zip(resolved, caps, strict=True):
        if enforcement == "solver":
            model.jnt_limited[item.joint_id] = True
            model.jnt_range[item.joint_id] = (0.0, cap)
        else:
            model.jnt_limited[item.joint_id] = False
    return tuple(resolved)


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.r_[
        a[0] * b[0] - np.dot(a[1:], b[1:]),
        a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:]),
    ]


def limit_residuals(
    q: np.ndarray, cfg: BallJointLimitCfg
) -> tuple[np.ndarray, np.ndarray]:
    """Return allowed-positive residuals and exact local angular Jacobians.

    Uses a tangent-half-angle ellipse and two angular twist inequalities. The
    scalar safety cap bounds the denominators and excludes the chart cuts.
    """
    frame = _quaternion(cfg.frame, "frame")
    neutral = _quaternion(cfg.neutral, "neutral")
    conj = np.array([1.0, -1.0, -1.0, -1.0])
    qt = _mul(_mul(frame * conj, _mul(neutral * conj, _quaternion(q, "q"))), frame)
    # Canonical relative quaternion: the native safety cap keeps this chart
    # away from w=0 in its admissible region. At the cut choose a deterministic
    # sign for diagnostics; exact singular initial states remain off-domain.
    pivot = 0 if abs(qt[0]) > 1e-15 else int(np.argmax(abs(qt)))
    if qt[pivot] < 0:
        qt = -qt
    w, x, y, z = qt
    h = w * w + z * z
    if h < 1e-12:
        raise ValueError("swing antipode lies outside the safety constraint domain")
    sx, sy = w * x - z * y, w * y + z * x
    lower = cfg.swing_lower or tuple(-v for v in cfg.swing)
    ax, ay = (
        1 / math.tan((upper if value >= 0 else low) / 2) ** 2
        for value, upper, low in zip((sx, sy), cfg.swing, lower, strict=True)
    )
    e = ax * sx * sx + ay * sy * sy
    dh = np.array([2 * w, 0.0, 0.0, 2 * z])
    de = np.array(
        [
            2 * ax * sx * x + 2 * ay * sy * y,
            2 * ax * sx * w + 2 * ay * sy * z,
            -2 * ax * sx * z + 2 * ay * sy * w,
            -2 * ax * sx * y + 2 * ay * sy * x,
        ]
    )
    theta = 2 * math.atan2(z, w)
    dtheta = np.array([-2 * z / h, 0.0, 0.0, 2 * w / h])
    residual = np.array([1 - e / (h * h), theta - cfg.twist[0], cfg.twist[1] - theta])
    gradient = np.stack([-de / (h * h) + 2 * e * dh / (h * h * h), dtheta, -dtheta])
    jac = 0.5 * (
        -gradient[:, :1] * qt[None, 1:]
        + w * gradient[:, 1:]
        - np.cross(qt[None, 1:], gradient[:, 1:])
    )
    # Rotate anatomical tangent vectors into original joint coordinates.
    fw, fv = frame[0], frame[1:]
    jac = jac + 2 * np.cross(fv[None, :], fw * jac + np.cross(fv[None, :], jac))
    return residual, jac
