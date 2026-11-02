from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import mujoco

from instantmimic.envs.config_access import get_block, get_value
from instantmimic.envs.tracking.observation_modes import (
    resolve_joint_pose_mode,
    resolve_joint_velocity_mode,
)
from instantmimic.envs.tracking.reward_kernels import (
    resolve_body_state_reward_frame_mode,
    resolve_reward_combination_mode,
)


class _MotionSettingsLike(Protocol):
    loop: Any


@dataclass(frozen=True)
class RewardSettings:
    body_state_reward: bool
    joint_pose_mode: int
    joint_velocity_mode: int
    combination_mode: int
    pose_w: float
    vel_w: float
    com_w: float
    ee_w: float
    ee_rot_w: float
    alive_reward: float
    body_state_frame_mode: int
    body_state_pos_w: float
    body_state_rot_w: float
    body_state_linvel_w: float
    body_state_angvel_w: float
    mix_w_pose: float
    mix_w_vel: float
    mix_w_body_pos: float
    mix_w_body_rot: float
    mix_w_body_linvel: float
    mix_w_body_angvel: float
    mix_w_com: float
    mix_w_ee_pos: float
    mix_w_ee_rot: float
    pose_include_root: bool
    velocity_include_root: bool
    ee_pos_root_relative: bool
    velocity_seam_fade_frames: int
    velocity_smoothing_radius: int
    velocity_smoothing_mode: str
    vel_valid_all: bool
    power_q: float
    power_beta: float
    power_epsilon: float
    frame_sigma_reference_delta_scale: float | None
    frame_sigma_min_weight_multiplier: float | None
    substep_mean: bool
    # Body-state tracking weight per body name; unnamed reward bodies weigh 1.
    # Empty for joint tracking, whose weights resolve per term below.
    body_weights: tuple[tuple[str, float], ...] = ()
    # Factor on the combined tracking reward (every combination mode).
    combination_scale: float = 1.0
    # Joint-tracking weights on the squared differences by child-body name
    # (tracking.body_weights merged under the term's own body_weights); None
    # keeps the built-in per-DOF weights (see resolve_joint_dof_scales).
    pose_body_weights: tuple[tuple[str, float], ...] | None = None
    velocity_body_weights: tuple[tuple[str, float], ...] | None = None
    # End-effector bodies the rotation term scores; None scores every one.
    ee_rotation_bodies: tuple[str, ...] | None = None
    # "component": mean over the 3 rotation-vector components of every scored
    # body; "body": mean of the squared rotation angle over the scored bodies.
    ee_rotation_mean: str = "component"
    # Bodies the end-effector position term scores; None scores the end
    # effectors (ee_body_name_groups).
    ee_position_bodies: tuple[str, ...] | None = None
    # Body whose position the position term subtracts on both sides (axes stay
    # world-aligned); None compares world positions (or root-relative ones).
    ee_position_anchor: str | None = None
    # "origin": body frame origins (xpos); "com": body centers of mass, for
    # the scored bodies and the anchor alike.
    ee_position_source: str = "origin"
    # Action-smoothness term exp(-weight * mean_d delta_d**2); None disables it.
    action_smoothness_weight: float | None = None
    # "auto" | "torque" | "pd_target" (see resolve_action_smoothness_source).
    action_smoothness_source: str = "auto"
    # PD stiffness that converts a joint-torque change to a PD-target change.
    action_smoothness_torque_gain: float = 450.0
    # weighted_sum weight of the action-smoothness term.
    mix_w_action_smoothness: float = 1.0

    @property
    def action_smoothness_enabled(self) -> bool:
        return self.action_smoothness_weight is not None

    @property
    def scalar_weights(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.pose_w,
            self.vel_w,
            self.com_w,
            self.ee_w,
            self.ee_rot_w,
            self.alive_reward,
        )

    @property
    def body_state_component_weights(self) -> tuple[float, float, float, float]:
        return (
            self.body_state_pos_w,
            self.body_state_rot_w,
            self.body_state_linvel_w,
            self.body_state_angvel_w,
        )

    @property
    def mix_weights(self) -> tuple[float, float, float, float, float]:
        return (
            self.mix_w_pose,
            self.mix_w_vel,
            self.mix_w_com,
            self.mix_w_ee_pos,
            self.mix_w_ee_rot,
        )

    @property
    def power(self) -> tuple[float, float, float]:
        return (self.power_q, self.power_beta, self.power_epsilon)

    @property
    def frame_sigma(self) -> tuple[float, float] | None:
        if self.frame_sigma_reference_delta_scale is None:
            return None
        assert self.frame_sigma_min_weight_multiplier is not None
        return (
            self.frame_sigma_reference_delta_scale,
            self.frame_sigma_min_weight_multiplier,
        )

    @property
    def reference_velocity(self) -> tuple[int, int, str]:
        return (
            self.velocity_seam_fade_frames,
            self.velocity_smoothing_radius,
            self.velocity_smoothing_mode,
        )


def _resolve_combination_scale(reward: Any) -> float:
    combination = get_block(reward, "combination", required=False)
    if combination is None:
        return 1.0
    scale = float(get_value(combination, "scale", 1.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("env.reward.combination.scale must be finite and > 0")
    return scale


def _resolve_combination(
    reward: Any,
    *,
    body_state_reward: bool,
) -> tuple[int, tuple[float, ...], tuple[float, float, float]]:
    combination = get_block(reward, "combination", required=False)
    default_mix = (1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0)
    if combination is None:
        return 0, default_mix, (1.0, 1.0, 1.0e-6)

    mode = resolve_reward_combination_mode(str(get_value(combination, "type")))
    power = (
        float(get_value(combination, "q", 1.0)),
        float(get_value(combination, "beta", 1.0)),
        float(get_value(combination, "epsilon", 1.0e-6)),
    )
    if mode != 1:
        return mode, default_mix, power

    weights = get_block(combination, "weights")
    common = (
        float(get_value(weights, "com", 1.0)),
        float(get_value(weights, "end_effector_position", 1.0)),
        float(get_value(weights, "end_effector_rotation", 1.0)),
    )
    if body_state_reward:
        mix = (
            0.0,
            0.0,
            float(get_value(weights, "position")),
            float(get_value(weights, "rotation")),
            float(get_value(weights, "linear_velocity")),
            float(get_value(weights, "angular_velocity")),
            *common,
        )
    else:
        mix = (
            float(get_value(weights, "pose")),
            float(get_value(weights, "velocity")),
            0.0,
            0.0,
            0.0,
            0.0,
            *common,
        )
    return mode, mix, power


def _resolve_body_weights(
    value: Any, *, key: str = "env.reward.tracking.body_weights"
) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    items = []
    for name, weight in dict(value).items():
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{key}[{name!r}] must be finite and >= 0")
        items.append((str(name), weight))
    return tuple(sorted(items))


# Joint-tracking keys replaced by body_weights. The environment mapping is not
# schema-checked, so a leftover key would otherwise be ignored without notice.
_REMOVED_JOINT_TERM_KEYS = {
    "body_dof_scales": (
        "multiplied each difference before squaring; body_weights multiplies the "
        "squared difference, so a scale s becomes the weight s**2"
    ),
    "dof_weights": "set per-DOF scales by position; name the child bodies in body_weights",
}


def _resolve_joint_term_body_weights(
    tracking: Any, term_block: Any, term: str
) -> tuple[tuple[str, float], ...] | None:
    """The term's weights by child body: its own body_weights over the shared ones."""

    for removed, why in _REMOVED_JOINT_TERM_KEYS.items():
        if get_value(term_block, removed, None) is not None:
            raise ValueError(
                f"env.reward.tracking.{term}.{removed} was replaced by "
                f"env.reward.tracking.{term}.body_weights: it {why}"
            )
    shared = get_value(tracking, "body_weights", None)
    own = get_value(term_block, "body_weights", None)
    if shared is None and own is None:
        return None
    merged = dict(_resolve_body_weights(shared))
    merged.update(
        _resolve_body_weights(own, key=f"env.reward.tracking.{term}.body_weights")
    )
    return tuple(sorted(merged.items()))


def _resolve_ee_rotation_bodies(end_effector: Any) -> tuple[str, ...] | None:
    if end_effector is None:
        return None
    value = get_value(end_effector, "rotation_bodies", None)
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError("env.reward.end_effector.rotation_bodies must be a list")
    names = tuple(str(name) for name in value)
    if not names:
        raise ValueError(
            "env.reward.end_effector.rotation_bodies must name at least one body"
        )
    if len(set(names)) != len(names):
        raise ValueError("env.reward.end_effector.rotation_bodies repeats a body")
    return names


def _resolve_ee_rotation_mean(end_effector: Any) -> str:
    if end_effector is None:
        return "component"
    mean = str(get_value(end_effector, "rotation_mean", "component")).lower()
    if mean not in {"component", "body"}:
        raise ValueError(
            "env.reward.end_effector.rotation_mean must be 'component' or 'body'"
        )
    return mean


def _resolve_ee_position_bodies(end_effector: Any) -> tuple[str, ...] | None:
    if end_effector is None:
        return None
    value = get_value(end_effector, "position_bodies", None)
    if value is None:
        return None
    if isinstance(value, str):
        raise TypeError("env.reward.end_effector.position_bodies must be a list")
    names = tuple(str(name) for name in value)
    if not names:
        raise ValueError(
            "env.reward.end_effector.position_bodies must name at least one body"
        )
    if len(set(names)) != len(names):
        raise ValueError("env.reward.end_effector.position_bodies repeats a body")
    return names


def _resolve_ee_position_anchor(end_effector: Any) -> str | None:
    if end_effector is None:
        return None
    value = get_value(end_effector, "position_anchor", None)
    if value is None:
        return None
    name = str(value).strip()
    if not name:
        raise ValueError(
            "env.reward.end_effector.position_anchor must name a body or be null"
        )
    if bool(get_value(end_effector, "position_root_relative", False)):
        raise ValueError(
            "env.reward.end_effector sets both position_anchor and "
            "position_root_relative; choose one"
        )
    return name


def _resolve_ee_position_source(end_effector: Any) -> str:
    if end_effector is None:
        return "origin"
    source = str(get_value(end_effector, "position_source", "origin")).lower()
    if source not in {"origin", "com"}:
        raise ValueError(
            "env.reward.end_effector.position_source must be 'origin' or 'com'"
        )
    if source == "com" and bool(
        get_value(end_effector, "position_root_relative", False)
    ):
        raise ValueError(
            "env.reward.end_effector.position_source: com does not combine with "
            "position_root_relative (the root position is the free-joint "
            "origin); use position_anchor: <root body> instead"
        )
    return source


ACTION_SMOOTHNESS_SOURCES = frozenset({"auto", "torque", "pd_target"})


def _resolve_action_smoothness(
    reward: Any,
) -> tuple[float | None, str, float]:
    block = get_block(reward, "action_smoothness", required=False)
    if block is None:
        return None, "auto", 450.0
    weight = float(get_value(block, "weight"))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("env.reward.action_smoothness.weight must be finite and >= 0")
    source = str(get_value(block, "source", "auto")).lower()
    if source not in ACTION_SMOOTHNESS_SOURCES:
        raise ValueError(
            "env.reward.action_smoothness.source must be one of "
            + ", ".join(sorted(ACTION_SMOOTHNESS_SOURCES))
        )
    gain = float(get_value(block, "torque_gain", 450.0))
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError(
            "env.reward.action_smoothness.torque_gain must be finite and > 0"
        )
    return weight, source, gain


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""


def resolve_ee_rotation_body_mask(
    settings: RewardSettings,
    model: mujoco.MjModel,
    ee_body_idx: tuple[int, ...] | list[int],
) -> tuple[int, ...] | None:
    """Per-end-effector rotation mask aligned with ee_body_idx, or None if unset.

    Every named body must be an end-effector body, so a typo fails loudly.
    """

    if settings.ee_rotation_bodies is None:
        return None
    names = [_body_name(model, int(i)) for i in ee_body_idx]
    for name in settings.ee_rotation_bodies:
        if name not in names:
            raise ValueError(
                "env.reward.end_effector.rotation_bodies names "
                f"{name!r}, which is not an end-effector body; end-effector "
                f"bodies: {', '.join(n or '?' for n in names)}"
            )
    return tuple(int(name in settings.ee_rotation_bodies) for name in names)


@dataclass(frozen=True)
class EePositionLayout:
    """Kernel layout of the end-effector position term."""

    # Scored bodies; None scores the end effectors at their body origins.
    body_idx: tuple[int, ...] | None
    # Per scored body, the body-frame point compared (body_ipos for "com").
    body_offset: tuple[tuple[float, float, float], ...] | None
    # Anchor body (-1: none) and its body-frame point.
    anchor_body_idx: int
    anchor_offset: tuple[float, float, float]


def _body_id(model: mujoco.MjModel, name: str, *, key: str) -> int:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if body_id <= 0:
        raise ValueError(f"{key} names {name!r}, which is not a model body")
    return body_id


def resolve_ee_position_layout(
    settings: RewardSettings,
    model: mujoco.MjModel,
    ee_body_idx: tuple[int, ...] | list[int],
) -> EePositionLayout:
    """Resolve position_bodies, position_anchor and position_source by name.

    Without any of them the layout is the historical one: the end effectors at
    their body origins and no anchor. Every name must be a model body other
    than the world.
    """

    use_com = settings.ee_position_source == "com"
    if settings.ee_position_bodies is None:
        body_idx = tuple(int(i) for i in ee_body_idx)
        explicit = use_com
    else:
        body_idx = tuple(
            _body_id(model, name, key="env.reward.end_effector.position_bodies")
            for name in settings.ee_position_bodies
        )
        explicit = True

    def offset(body_id: int) -> tuple[float, float, float]:
        if not use_com:
            return (0.0, 0.0, 0.0)
        return tuple(float(v) for v in model.body_ipos[int(body_id)])  # type: ignore[return-value]

    anchor = -1
    anchor_offset = (0.0, 0.0, 0.0)
    if settings.ee_position_anchor is not None:
        anchor = _body_id(
            model,
            settings.ee_position_anchor,
            key="env.reward.end_effector.position_anchor",
        )
        anchor_offset = offset(anchor)
    return EePositionLayout(
        body_idx=body_idx if explicit else None,
        body_offset=tuple(offset(b) for b in body_idx) if explicit else None,
        anchor_body_idx=anchor,
        anchor_offset=anchor_offset,
    )


def joint_root_error_weight(
    body_weights: tuple[tuple[str, float], ...] | None,
    root_body_name: str,
) -> float:
    """Weight on the squared root error of a joint-tracking term.

    Without a body_weights mapping the root error counts twice (the built-in
    weight 2). With one, the root body's weight applies (1 when unnamed).
    """

    if body_weights is None:
        return 2.0
    return float(dict(body_weights).get(root_body_name, 1.0))


_JOINT_DOF_COUNT = {
    int(mujoco.mjtJoint.mjJNT_FREE): 6,
    int(mujoco.mjtJoint.mjJNT_BALL): 3,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}


def resolve_joint_dof_scales(
    body_weights: tuple[tuple[str, float], ...],
    model: mujoco.MjModel,
    *,
    joint_ids: tuple[int, ...] | list[int],
    root_body_idx: int,
    root_dof_count: int,
    tracked_dof_mask: tuple[bool, ...] | list[bool],
    key: str,
) -> tuple[float, ...]:
    """Per-DOF joint-tracking difference scales over the non-root DOFs [root, nv).

    The reward kernel squares each scaled difference, so a DOF scale is the
    square root of its weight. The DOFs of the character joints (``joint_ids``)
    whose child body is named take that body's weight. Every other tracked DOF
    (``tracked_dof_mask``) weighs 1, and an untracked or passive DOF 0. The
    mapping replaces the built-in talus and toe weights; the root body's entry
    weights the root error instead (joint_root_error_weight). Every named body
    must be the root or own a character joint.
    """

    joint_dof_count = int(model.nv) - int(root_dof_count)
    if len(tracked_dof_mask) != joint_dof_count:
        raise ValueError(
            f"tracked_dof_mask has {len(tracked_dof_mask)} entries for "
            f"{joint_dof_count} joint DOFs"
        )
    values = [1.0 if bool(tracked) else 0.0 for tracked in tracked_dof_mask]
    body_joints: dict[int, list[int]] = {}
    for jnt_id in joint_ids:
        body_id = int(model.jnt_bodyid[int(jnt_id)])
        body_joints.setdefault(body_id, []).append(int(jnt_id))
    for name, weight in body_weights:
        body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        if body_id < 0:
            raise ValueError(f"{key} names {name!r}, which is not a model body")
        if body_id == int(root_body_idx):
            continue
        if body_id not in body_joints:
            raise ValueError(
                f"{key} names {name!r}, which owns no character joint; name the "
                "child body of the joints to weight"
            )
        for jnt_id in body_joints[body_id]:
            dof_adr = int(model.jnt_dofadr[jnt_id]) - int(root_dof_count)
            for offset in range(_JOINT_DOF_COUNT[int(model.jnt_type[jnt_id])]):
                values[dof_adr + offset] = math.sqrt(float(weight))
    return tuple(values)


def resolve_reward_body_weight(
    settings: RewardSettings,
    model: mujoco.MjModel,
    reward_body_idx: tuple[int, ...] | list[int],
) -> tuple[float, ...] | None:
    """Per-reward-body weights aligned with reward_body_idx, or None if unset.

    Every named body must be a reward (tracked) body, so a typo fails loudly.
    """

    if not settings.body_weights:
        return None
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i))
        for i in reward_body_idx
    ]
    weights = [1.0] * len(names)
    for name, weight in settings.body_weights:
        if name not in names:
            raise ValueError(
                f"env.reward.tracking.body_weights names {name!r}, which is not a "
                f"tracked body; tracked bodies: {', '.join(n or '?' for n in names)}"
            )
        weights[names.index(name)] = weight
    if sum(weights) <= 0.0:
        raise ValueError("env.reward.tracking.body_weights leave no positive weight")
    return tuple(weights)


def resolve_reward_settings(
    cfg: Any,
    *,
    motion_settings: _MotionSettingsLike | None = None,
) -> RewardSettings:
    reward = get_block(cfg, "reward")
    tracking = get_block(reward, "tracking")
    tracking_type = str(get_value(tracking, "type")).strip().lower()
    if tracking_type not in {"joint", "body_state"}:
        raise ValueError("env.reward.tracking.type must be 'joint' or 'body_state'")
    body_state_reward = tracking_type == "body_state"

    pose_w = 0.0
    vel_w = 0.0
    joint_pose_mode = resolve_joint_pose_mode("default")
    joint_velocity_mode = resolve_joint_velocity_mode("qvel")
    pose_include_root = True
    velocity_include_root = False
    body_state_frame_mode = resolve_body_state_reward_frame_mode("world")
    body_weights = (0.0, 0.0, 0.0, 0.0)
    pose_body_weights = None
    velocity_body_weights = None
    frame_sigma_reference_delta_scale = None
    frame_sigma_min_weight_multiplier = None
    per_body_weights = ()

    if body_state_reward:
        per_body_weights = _resolve_body_weights(
            get_value(tracking, "body_weights", None)
        )
        for term in ("pose", "velocity"):
            block = get_block(tracking, term, required=False)
            if block is not None and get_value(block, "body_weights", None) is not None:
                raise ValueError(
                    f"env.reward.tracking.{term}.body_weights requires tracking.type: "
                    "joint; body-state tracking takes tracking.body_weights"
                )
        weights = get_block(tracking, "weights")
        body_state_frame_mode = resolve_body_state_reward_frame_mode(
            get_value(tracking, "frame", "world")
        )
        body_weights = (
            float(get_value(weights, "position")),
            float(get_value(weights, "rotation")),
            float(get_value(weights, "linear_velocity")),
            float(get_value(weights, "angular_velocity")),
        )
        frame_sigma = get_block(tracking, "frame_sigma", required=False)
        if frame_sigma is not None:
            frame_sigma_reference_delta_scale = float(
                get_value(frame_sigma, "reference_delta_scale")
            )
            frame_sigma_min_weight_multiplier = float(
                get_value(frame_sigma, "min_weight_multiplier")
            )
    else:
        pose = get_block(tracking, "pose")
        velocity = get_block(tracking, "velocity")
        pose_w = float(get_value(pose, "weight"))
        vel_w = float(get_value(velocity, "weight"))
        joint_pose_mode = resolve_joint_pose_mode(
            str(get_value(pose, "mode", "default")).lower()
        )
        joint_velocity_mode = resolve_joint_velocity_mode(
            str(get_value(velocity, "mode", "qvel")).lower()
        )
        pose_include_root = bool(get_value(pose, "include_root", True))
        velocity_include_root = bool(get_value(velocity, "include_root", False))
        pose_body_weights = _resolve_joint_term_body_weights(tracking, pose, "pose")
        velocity_body_weights = _resolve_joint_term_body_weights(
            tracking, velocity, "velocity"
        )

    combination_mode, mix, power = _resolve_combination(
        reward, body_state_reward=body_state_reward
    )
    (
        action_smoothness_weight,
        action_smoothness_source,
        action_smoothness_torque_gain,
    ) = _resolve_action_smoothness(reward)
    mix_w_action_smoothness = 1.0
    if combination_mode == 1:
        weights = get_block(get_block(reward, "combination"), "weights")
        mix_w_action_smoothness = float(get_value(weights, "action_smoothness", 1.0))
        if not math.isfinite(mix_w_action_smoothness) or mix_w_action_smoothness < 0.0:
            raise ValueError(
                "env.reward.combination.weights.action_smoothness must be finite "
                "and >= 0"
            )
    end_effector = get_block(reward, "end_effector", required=False)
    reference_velocity = get_block(reward, "reference_velocity", required=False)

    return RewardSettings(
        body_state_reward=body_state_reward,
        joint_pose_mode=joint_pose_mode,
        joint_velocity_mode=joint_velocity_mode,
        combination_mode=combination_mode,
        pose_w=pose_w,
        vel_w=vel_w,
        com_w=float(get_value(reward, "com_weight", 0.0)),
        ee_w=(
            0.0
            if end_effector is None
            else float(get_value(end_effector, "position_weight", 0.0))
        ),
        ee_rot_w=(
            0.0
            if end_effector is None
            else float(get_value(end_effector, "rotation_weight", 0.0))
        ),
        alive_reward=float(get_value(reward, "alive", 0.0)),
        body_state_frame_mode=body_state_frame_mode,
        body_state_pos_w=body_weights[0],
        body_state_rot_w=body_weights[1],
        body_state_linvel_w=body_weights[2],
        body_state_angvel_w=body_weights[3],
        mix_w_pose=mix[0],
        mix_w_vel=mix[1],
        mix_w_body_pos=mix[2],
        mix_w_body_rot=mix[3],
        mix_w_body_linvel=mix[4],
        mix_w_body_angvel=mix[5],
        mix_w_com=mix[6],
        mix_w_ee_pos=mix[7],
        mix_w_ee_rot=mix[8],
        pose_include_root=pose_include_root,
        velocity_include_root=velocity_include_root,
        ee_pos_root_relative=(
            False
            if end_effector is None
            else bool(get_value(end_effector, "position_root_relative", False))
        ),
        velocity_seam_fade_frames=(
            0
            if reference_velocity is None
            else int(get_value(reference_velocity, "seam_fade_frames", 0))
        ),
        velocity_smoothing_radius=(
            0
            if reference_velocity is None
            else int(get_value(reference_velocity, "smoothing_radius", 0))
        ),
        velocity_smoothing_mode=(
            "all"
            if reference_velocity is None
            else str(get_value(reference_velocity, "smoothing_mode", "all")).lower()
        ),
        vel_valid_all=bool(
            motion_settings is not None
            and (
                motion_settings.loop.velocity_from_first_frame
                or motion_settings.loop.velocity_extrapolate
                or motion_settings.loop.smooth
            )
        ),
        power_q=power[0],
        power_beta=power[1],
        power_epsilon=power[2],
        frame_sigma_reference_delta_scale=frame_sigma_reference_delta_scale,
        frame_sigma_min_weight_multiplier=frame_sigma_min_weight_multiplier,
        substep_mean=bool(get_value(reward, "substep_mean", False)),
        body_weights=per_body_weights,
        combination_scale=_resolve_combination_scale(reward),
        pose_body_weights=pose_body_weights,
        velocity_body_weights=velocity_body_weights,
        ee_rotation_bodies=_resolve_ee_rotation_bodies(end_effector),
        ee_rotation_mean=_resolve_ee_rotation_mean(end_effector),
        ee_position_bodies=_resolve_ee_position_bodies(end_effector),
        ee_position_anchor=_resolve_ee_position_anchor(end_effector),
        ee_position_source=_resolve_ee_position_source(end_effector),
        action_smoothness_weight=action_smoothness_weight,
        action_smoothness_source=action_smoothness_source,
        action_smoothness_torque_gain=action_smoothness_torque_gain,
        mix_w_action_smoothness=mix_w_action_smoothness,
    )


__all__ = (
    "ACTION_SMOOTHNESS_SOURCES",
    "EePositionLayout",
    "RewardSettings",
    "resolve_ee_position_layout",
    "joint_root_error_weight",
    "resolve_joint_dof_scales",
    "resolve_ee_rotation_body_mask",
    "resolve_reward_body_weight",
    "resolve_reward_settings",
)
