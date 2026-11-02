from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import torch
import warp as wp

from instantmimic.envs.config_access import get_block, get_value
from instantmimic.envs.snu.tasks.snu_imitation import termination_kernels

__all__ = (
    "DeviationBoundsSettings",
    "TerminationSettings",
    "can_parallelize_dones_and_rewards",
    "get_dones",
    "finalize_reward_termination",
    "initialize_termination_state",
    "resolve_deviation_joint_groups",
    "resolve_termination_settings",
)

_DEVIATION_BOUND_KEYS = frozenset({"bodies", "threshold"})


@dataclass(frozen=True)
class DeviationBoundsSettings:
    """Bounds of ``termination.early.type: deviation_bounds``; None disables one."""

    com_distance: float | None
    body_distance: float | None
    body_distance_bodies: tuple[str, ...]
    joint_angle: float | None
    joint_angle_bodies: tuple[str, ...]


@dataclass(frozen=True)
class TerminationSettings:
    min_torso_height: float
    motion_end: bool
    nonfinite: bool
    early_enabled: bool
    early_mode: int
    reward_threshold: float
    consecutive_steps: int
    phc_distance: float
    deviation: DeviationBoundsSettings | None = None


def _positive_bound(value: Any, key: str) -> float:
    bound = float(value)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError(f"{key} must be finite and > 0")
    return bound


def _resolve_body_bound(early: Any, name: str) -> tuple[float | None, tuple[str, ...]]:
    key = f"env.termination.early.{name}"
    block = get_block(early, name, required=False)
    if block is None:
        return None, ()
    unknown = sorted(set(dict(block)) - _DEVIATION_BOUND_KEYS)
    if unknown:
        raise ValueError(f"{key} has unknown keys: {', '.join(unknown)}")
    bodies = get_value(block, "bodies")
    if isinstance(bodies, str) or not bodies:
        raise ValueError(f"{key}.bodies must be a non-empty list of body names")
    names = tuple(str(body) for body in bodies)
    if len(set(names)) != len(names):
        raise ValueError(f"{key}.bodies repeats a body")
    return _positive_bound(get_value(block, "threshold"), f"{key}.threshold"), names


def _resolve_deviation_bounds(early: Any) -> DeviationBoundsSettings:
    com = get_value(early, "com_distance", None)
    com_distance = (
        None
        if com is None
        else _positive_bound(com, "env.termination.early.com_distance")
    )
    body_distance, body_names = _resolve_body_bound(early, "body_distance")
    joint_angle, joint_names = _resolve_body_bound(early, "joint_angle")
    if com_distance is None and body_distance is None and joint_angle is None:
        raise ValueError(
            "termination.early.type: deviation_bounds needs at least one of "
            "com_distance, body_distance or joint_angle"
        )
    return DeviationBoundsSettings(
        com_distance=com_distance,
        body_distance=body_distance,
        body_distance_bodies=body_names,
        joint_angle=joint_angle,
        joint_angle_bodies=joint_names,
    )


def resolve_termination_settings(cfg) -> TerminationSettings:
    termination = get_block(cfg, "termination")
    early = get_block(termination, "early", required=False)
    early_enabled = early is not None and bool(get_value(early, "enabled", True))
    early_type = "reward" if early is None else str(get_value(early, "type", "reward"))
    early_mode = termination_kernels.resolve_early_termination_mode(early_type)
    return TerminationSettings(
        min_torso_height=float(get_value(termination, "min_torso_height")),
        motion_end=bool(get_value(termination, "motion_end", False)),
        nonfinite=bool(get_value(termination, "nonfinite", False)),
        early_enabled=early_enabled,
        early_mode=early_mode,
        reward_threshold=(
            0.1 if early is None else float(get_value(early, "reward_threshold", 0.1))
        ),
        consecutive_steps=(
            4 if early is None else int(get_value(early, "consecutive_steps", 4))
        ),
        phc_distance=(
            0.5 if early is None else float(get_value(early, "distance", 0.5))
        ),
        deviation=(
            _resolve_deviation_bounds(early)
            if early_mode == termination_kernels.EARLY_TERMINATION_MODE_DEVIATION_BOUNDS
            else None
        ),
    )


def _body_id(model: mujoco.MjModel, name: str, key: str) -> int:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
    if body_id <= 0:
        raise ValueError(f"{key} names {name!r}, which is not a model body")
    return body_id


def resolve_deviation_joint_groups(
    model: mujoco.MjModel,
    body_names: tuple[str, ...],
    *,
    joint_ids: tuple[int, ...],
    tracked_bodies: tuple[int, ...],
    root_body_idx: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """The joint coordinates whose error each named body's bound limits.

    A body's group holds the hinge and ball character joints (``joint_ids``)
    between it and its nearest tracked ancestor: its own joints and those of
    any untracked intermediate bodies. Returns CSR group starts, joint qpos
    addresses and ball flags.
    """

    key = "env.termination.early.joint_angle.bodies"
    character_joints = frozenset(int(j) for j in joint_ids)
    tracked = frozenset(int(b) for b in tracked_bodies)
    group_start = [0]
    qpos_adr: list[int] = []
    is_ball: list[int] = []
    for name in body_names:
        body_id = _body_id(model, name, key)
        if body_id == int(root_body_idx):
            raise ValueError(f"{key} names the root body {name!r}")
        chain = [body_id]
        parent = int(model.body_parentid[body_id])
        while parent not in tracked and parent not in (0, int(root_body_idx)):
            chain.append(parent)
            parent = int(model.body_parentid[parent])
        entries = []
        for jnt_id in range(int(model.njnt)):
            if jnt_id not in character_joints:
                continue
            if int(model.jnt_bodyid[jnt_id]) not in chain:
                continue
            jnt_type = int(model.jnt_type[jnt_id])
            if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(f"{key} names {name!r}, which has a free joint")
            if jnt_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            entries.append(
                (
                    int(model.jnt_qposadr[jnt_id]),
                    int(jnt_type == int(mujoco.mjtJoint.mjJNT_BALL)),
                )
            )
        if not entries:
            raise ValueError(
                f"{key} names {name!r}, which has no hinge or ball character joint"
            )
        for adr, ball in entries:
            qpos_adr.append(adr)
            is_ball.append(ball)
        group_start.append(len(qpos_adr))
    return tuple(group_start), tuple(qpos_adr), tuple(is_ball)


def initialize_termination_state(env) -> None:
    """Resolve deviation-bound bodies and joints once, before the first step."""
    deviation = env.termination_settings.deviation
    body_ids: tuple[int, ...] = ()
    groups: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = ((0,), (), ())
    if deviation is not None:
        character = frozenset(int(b) for b in env.dof_roles.character_bodies)
        key = "env.termination.early.body_distance.bodies"
        body_ids = tuple(
            _body_id(env.model, name, key) for name in deviation.body_distance_bodies
        )
        for name, body_id in zip(deviation.body_distance_bodies, body_ids, strict=True):
            if body_id not in character:
                raise ValueError(f"{key} names {name!r}, which is a passive body")
        groups = resolve_deviation_joint_groups(
            env.model,
            deviation.joint_angle_bodies,
            joint_ids=tuple(int(j) for j in env.dof_roles.joint_ids),
            tracked_bodies=tuple(int(b) for b in env.reward_body_idx.tolist()),
            root_body_idx=int(env.root_body_idx),
        )

    def int_tensor(values: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(values, device=env.device, dtype=torch.int32)

    env.deviation_body_idx = int_tensor(body_ids)
    env.deviation_joint_group_start = int_tensor(groups[0])
    env.deviation_joint_qpos_adr = int_tensor(groups[1])
    env.deviation_joint_is_ball = int_tensor(groups[2])


def can_parallelize_dones_and_rewards(env) -> bool:
    return not env.termination_settings.early_enabled


def _bound(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _apply_deviation_bounds(env, deviation: DeviationBoundsSettings) -> None:
    wp.launch(
        termination_kernels.apply_deviation_bounds_termination_kernel,
        dim=(env.num_envs,),
        inputs=[
            env.mjw_data.xpos,
            env.mjw_data.subtree_com,
            env.mjw_data.qpos,
            env.reference_body_pos,
            env.ref_com,
            env.motion_qpos,
            env.motion_clip_idx,
            env.ref_frame_idx,
            env.motion_clip_start,
            env.motion_clip_length,
            env.motion_clip_loop_root_quat,
            env.motion_clip_loop_root_pos,
            int(env.root_body_idx),
            _bound(deviation.com_distance),
            env.deviation_body_idx,
            _bound(deviation.body_distance),
            env.deviation_joint_group_start,
            env.deviation_joint_qpos_adr,
            env.deviation_joint_is_ball,
            _bound(deviation.joint_angle),
            int(env._track_done_reasons),
        ],
        outputs=[env.reset_terminated, env._done_reason_mask_buf],
    )


def get_dones(env) -> tuple[torch.Tensor, torch.Tensor]:
    settings = env.termination_settings
    common_inputs = [
        env.mjw_data.xpos,
        env.reference_body_pos,
        env.episode_length_buf,
        env.early_termination_timer,
        env.motion_clip_idx,
        env.ref_frame_idx,
        env.motion_clip_start,
        env.motion_clip_length,
        env.motion_clip_loop_root_quat,
        env.motion_clip_loop_root_pos,
        env.phc_termination_body_idx,
        int(settings.early_enabled),
        int(settings.early_mode),
        int(settings.motion_end),
        env.torso_body_idx,
        settings.min_torso_height,
        env.max_episode_length,
        settings.consecutive_steps,
        settings.phc_distance,
    ]
    common_outputs = [
        env.reset_terminated,
        env.reset_time_outs,
        env.bootstrap_time_outs,
        env.motion_end_time_outs,
    ]
    if env._track_done_reasons:
        wp.launch(
            termination_kernels.compute_dones_with_reason_masks_kernel,
            dim=(env.num_envs,),
            inputs=common_inputs,
            outputs=[env._done_reason_mask_buf, *common_outputs],
        )
    else:
        wp.launch(
            termination_kernels.compute_dones_kernel,
            dim=(env.num_envs,),
            inputs=common_inputs,
            outputs=common_outputs,
        )
    # Deviation bounds exist only for the deviation_bounds early mode.
    if settings.early_enabled and settings.deviation is not None:
        _apply_deviation_bounds(env, settings.deviation)
    return env.reset_terminated, env.reset_time_outs


def finalize_reward_termination(env) -> bool:
    """Apply current reward-based failure; report whether dones need combining."""
    settings = env.termination_settings
    if (
        not settings.early_enabled
        or settings.early_mode != termination_kernels.EARLY_TERMINATION_MODE_REWARD
    ):
        return False
    wp.launch(
        termination_kernels.finalize_reward_termination_kernel,
        dim=(env.num_envs,),
        inputs=[
            env.total_reward,
            settings.reward_threshold,
            settings.consecutive_steps,
            int(env._track_done_reasons),
        ],
        outputs=[
            env.early_termination_timer,
            env.reset_terminated,
            env._done_reason_mask_buf,
        ],
    )
    return True
