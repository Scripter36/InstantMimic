from __future__ import annotations

import numpy as np
import warp as wp

from instantmimic.envs.snu.tasks.snu_imitation.warp_utils import (
    read_quat_from_2d,
    read_vec3_from_2d,
    safe_quat_sub_wp,
)
from instantmimic.motion.warp_utils import (
    apply_loop_transform_vec,
    clip_cycle_count,
    clip_global_frame,
)

EARLY_TERMINATION_MODE_REWARD = 0
EARLY_TERMINATION_MODE_PHC_TRACKING = 1
EARLY_TERMINATION_MODE_DEVIATION_BOUNDS = 2
_EARLY_TERMINATION_MODES = {
    "reward": EARLY_TERMINATION_MODE_REWARD,
    "phc_tracking": EARLY_TERMINATION_MODE_PHC_TRACKING,
    "deviation_bounds": EARLY_TERMINATION_MODE_DEVIATION_BOUNDS,
}
_PHC_TERMINATION_EXCLUDED_BODY_NAMES = frozenset(
    {
        "talusr",
        "talusl",
        "r_ankle",
        "l_ankle",
        "rightfoot",
        "leftfoot",
        "right_foot",
        "left_foot",
        "footthumbr",
        "footthumbl",
        "footpinkyr",
        "footpinkyl",
        "r_toe",
        "l_toe",
        "righttoe",
        "lefttoe",
        "right_toe",
        "left_toe",
    }
)


def resolve_early_termination_mode(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        mode = int(value)
        if mode in _EARLY_TERMINATION_MODES.values():
            return mode
        raise ValueError(f"Unsupported termination.early.type: {value!r}.")

    mode_key = str(value).strip().lower()
    if mode_key in _EARLY_TERMINATION_MODES:
        return _EARLY_TERMINATION_MODES[mode_key]
    raise ValueError(
        "termination.early.type must be one of: "
        f"{', '.join(_EARLY_TERMINATION_MODES)}. Got {value!r}."
    )


def is_phc_termination_body_name(body_name: str | None) -> bool:
    if body_name is None:
        return False
    return body_name.strip().lower() not in _PHC_TERMINATION_EXCLUDED_BODY_NAMES


@wp.kernel(enable_backward=False)
def compute_dones_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    reference_body_pos: wp.array2d(dtype=wp.vec3),
    episode_length_buf: wp.array(dtype=wp.int64),
    early_termination_timer: wp.array(dtype=int),
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    motion_clip_start: wp.array(dtype=int),
    motion_clip_length: wp.array(dtype=int),
    motion_clip_loop_root_quat: wp.array2d(dtype=float),
    motion_clip_loop_root_pos: wp.array2d(dtype=float),
    phc_termination_body_idx: wp.array(dtype=wp.int64),
    enable_early_termination: int,
    early_termination_mode: int,
    terminate_at_motion_end: int,
    torso_body_idx: int,
    termination_height: float,
    max_episode_length: int,
    early_term_time_threshold: int,
    phc_termination_distance: float,
    reset_terminated: wp.array(dtype=wp.bool),
    reset_time_outs: wp.array(dtype=wp.bool),
    bootstrap_time_outs: wp.array(dtype=wp.bool),
    motion_end_time_outs: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()

    ep_len = int(episode_length_buf[env_idx])
    episode_timed_out = ep_len >= max_episode_length
    motion_ended = False
    if terminate_at_motion_end == 1:
        clip_idx = motion_clip_idx[env_idx]
        clip_last_frame = motion_clip_length[clip_idx] - 1
        if ref_frame_idx[env_idx] >= clip_last_frame:
            motion_ended = True
    reset_time_outs[env_idx] = episode_timed_out or motion_ended
    bootstrap_time_outs[env_idx] = episode_timed_out and not motion_ended
    motion_end_time_outs[env_idx] = motion_ended

    torso_z = xpos[env_idx, torso_body_idx][2]
    died = torso_z < termination_height

    if enable_early_termination == 1:
        if early_termination_mode == EARLY_TERMINATION_MODE_PHC_TRACKING:
            clip_idx = motion_clip_idx[env_idx]
            frame_idx = clip_global_frame(
                ref_frame_idx[env_idx], clip_idx, motion_clip_start, motion_clip_length
            )
            cycles = clip_cycle_count(
                ref_frame_idx[env_idx], clip_idx, motion_clip_length
            )
            loop_root_q = read_quat_from_2d(motion_clip_loop_root_quat, clip_idx, 0)
            loop_root_t = read_vec3_from_2d(motion_clip_loop_root_pos, clip_idx, 0)
            body_i = int(0)
            dist_sum = float(0.0)
            while body_i < int(phc_termination_body_idx.shape[0]):
                body_idx = int(phc_termination_body_idx[body_i])
                sim_pos = xpos[env_idx, body_idx]
                ref_pos = apply_loop_transform_vec(
                    reference_body_pos[frame_idx, body_idx],
                    loop_root_q,
                    loop_root_t,
                    cycles,
                )
                diff = sim_pos - ref_pos
                dist_sum = dist_sum + wp.length(diff)
                body_i = body_i + 1
            if int(phc_termination_body_idx.shape[0]) > 0:
                if (
                    dist_sum / float(phc_termination_body_idx.shape[0])
                    > phc_termination_distance
                ):
                    early_termination_timer[env_idx] = (
                        early_termination_timer[env_idx] + 1
                    )
                else:
                    early_termination_timer[env_idx] = 0

                if early_termination_timer[env_idx] >= early_term_time_threshold:
                    died = True
            else:
                early_termination_timer[env_idx] = 0
    else:
        early_termination_timer[env_idx] = 0

    reset_terminated[env_idx] = died


@wp.kernel(enable_backward=False)
def compute_dones_with_reason_masks_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    reference_body_pos: wp.array2d(dtype=wp.vec3),
    episode_length_buf: wp.array(dtype=wp.int64),
    early_termination_timer: wp.array(dtype=int),
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    motion_clip_start: wp.array(dtype=int),
    motion_clip_length: wp.array(dtype=int),
    motion_clip_loop_root_quat: wp.array2d(dtype=float),
    motion_clip_loop_root_pos: wp.array2d(dtype=float),
    phc_termination_body_idx: wp.array(dtype=wp.int64),
    enable_early_termination: int,
    early_termination_mode: int,
    terminate_at_motion_end: int,
    torso_body_idx: int,
    termination_height: float,
    max_episode_length: int,
    early_term_time_threshold: int,
    phc_termination_distance: float,
    reason_masks: wp.array2d(dtype=wp.bool),
    reset_terminated: wp.array(dtype=wp.bool),
    reset_time_outs: wp.array(dtype=wp.bool),
    bootstrap_time_outs: wp.array(dtype=wp.bool),
    motion_end_time_outs: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()

    ep_len = int(episode_length_buf[env_idx])
    episode_timed_out = ep_len >= max_episode_length
    motion_ended = False
    if terminate_at_motion_end == 1:
        clip_idx = motion_clip_idx[env_idx]
        clip_last_frame = motion_clip_length[clip_idx] - 1
        if ref_frame_idx[env_idx] >= clip_last_frame:
            motion_ended = True
    reset_time_outs[env_idx] = episode_timed_out or motion_ended
    bootstrap_time_outs[env_idx] = episode_timed_out and not motion_ended
    motion_end_time_outs[env_idx] = motion_ended

    torso_z = xpos[env_idx, torso_body_idx][2]
    height_terminated = torso_z < termination_height
    early_terminated = False
    if enable_early_termination == 1:
        if early_termination_mode == EARLY_TERMINATION_MODE_PHC_TRACKING:
            clip_idx = motion_clip_idx[env_idx]
            frame_idx = clip_global_frame(
                ref_frame_idx[env_idx], clip_idx, motion_clip_start, motion_clip_length
            )
            cycles = clip_cycle_count(
                ref_frame_idx[env_idx], clip_idx, motion_clip_length
            )
            loop_root_q = read_quat_from_2d(motion_clip_loop_root_quat, clip_idx, 0)
            loop_root_t = read_vec3_from_2d(motion_clip_loop_root_pos, clip_idx, 0)
            body_i = int(0)
            dist_sum = float(0.0)
            while body_i < int(phc_termination_body_idx.shape[0]):
                body_idx = int(phc_termination_body_idx[body_i])
                sim_pos = xpos[env_idx, body_idx]
                ref_pos = apply_loop_transform_vec(
                    reference_body_pos[frame_idx, body_idx],
                    loop_root_q,
                    loop_root_t,
                    cycles,
                )
                diff = sim_pos - ref_pos
                dist_sum = dist_sum + wp.length(diff)
                body_i = body_i + 1
            if int(phc_termination_body_idx.shape[0]) > 0:
                if (
                    dist_sum / float(phc_termination_body_idx.shape[0])
                    > phc_termination_distance
                ):
                    early_termination_timer[env_idx] = (
                        early_termination_timer[env_idx] + 1
                    )
                else:
                    early_termination_timer[env_idx] = 0

                if early_termination_timer[env_idx] >= early_term_time_threshold:
                    early_terminated = True
            else:
                early_termination_timer[env_idx] = 0
    else:
        early_termination_timer[env_idx] = 0

    reason_masks[0, env_idx] = height_terminated
    reason_masks[1, env_idx] = early_terminated
    reset_terminated[env_idx] = height_terminated or early_terminated


@wp.kernel(enable_backward=False)
def finalize_reward_termination_kernel(
    reward: wp.array(dtype=float),
    reward_threshold: float,
    consecutive_steps: int,
    track_reasons: int,
    early_termination_timer: wp.array(dtype=int),
    reset_terminated: wp.array(dtype=wp.bool),
    reason_masks: wp.array2d(dtype=wp.bool),
):
    world = wp.tid()
    if reward[world] < reward_threshold:
        early_termination_timer[world] = early_termination_timer[world] + 1
    else:
        early_termination_timer[world] = 0
    early_terminated = early_termination_timer[world] >= consecutive_steps
    reset_terminated[world] = reset_terminated[world] or early_terminated
    if track_reasons == 1:
        reason_masks[1, world] = early_terminated


@wp.kernel(enable_backward=False)
def apply_deviation_bounds_termination_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    qpos: wp.array2d(dtype=float),
    reference_body_pos: wp.array2d(dtype=wp.vec3),
    reference_com: wp.array(dtype=wp.vec3),
    motion_qpos: wp.array2d(dtype=float),
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    motion_clip_start: wp.array(dtype=int),
    motion_clip_length: wp.array(dtype=int),
    motion_clip_loop_root_quat: wp.array2d(dtype=float),
    motion_clip_loop_root_pos: wp.array2d(dtype=float),
    root_body_idx: int,
    com_distance: float,
    distance_body_idx: wp.array(dtype=int),
    body_distance: float,
    joint_group_start: wp.array(dtype=int),
    joint_qpos_adr: wp.array(dtype=int),
    joint_is_ball: wp.array(dtype=int),
    joint_angle: float,
    track_reasons: int,
    reset_terminated: wp.array(dtype=wp.bool),
    reason_masks: wp.array2d(dtype=wp.bool),
):
    """End an episode on the first step any deviation exceeds its bound.

    A bound <= 0 disables its criterion. Positions compare in the world frame
    against the loop-transformed reference; joint coordinates are relative to
    their parent body and need no loop transform. Each joint group's error is
    the norm over its hinge differences and ball rotation vectors.
    """
    env_idx = wp.tid()
    clip_idx = motion_clip_idx[env_idx]
    frame_idx = clip_global_frame(
        ref_frame_idx[env_idx], clip_idx, motion_clip_start, motion_clip_length
    )
    cycles = clip_cycle_count(ref_frame_idx[env_idx], clip_idx, motion_clip_length)
    loop_root_q = read_quat_from_2d(motion_clip_loop_root_quat, clip_idx, 0)
    loop_root_t = read_vec3_from_2d(motion_clip_loop_root_pos, clip_idx, 0)

    deviated = False
    if com_distance > 0.0:
        ref_com = apply_loop_transform_vec(
            reference_com[frame_idx], loop_root_q, loop_root_t, cycles
        )
        if wp.length(subtree_com[env_idx, root_body_idx] - ref_com) > com_distance:
            deviated = True

    if body_distance > 0.0:
        body_i = int(0)
        while body_i < distance_body_idx.shape[0]:
            body_id = distance_body_idx[body_i]
            ref_pos = apply_loop_transform_vec(
                reference_body_pos[frame_idx, body_id],
                loop_root_q,
                loop_root_t,
                cycles,
            )
            if wp.length(xpos[env_idx, body_id] - ref_pos) > body_distance:
                deviated = True
            body_i = body_i + 1

    if joint_angle > 0.0:
        limit_sq = joint_angle * joint_angle
        group_i = int(0)
        while group_i < joint_group_start.shape[0] - 1:
            error_sq = float(0.0)
            entry = joint_group_start[group_i]
            while entry < joint_group_start[group_i + 1]:
                adr = joint_qpos_adr[entry]
                if joint_is_ball[entry] == 1:
                    rotvec = safe_quat_sub_wp(
                        read_quat_from_2d(qpos, env_idx, adr),
                        read_quat_from_2d(motion_qpos, frame_idx, adr),
                    )
                    error_sq = error_sq + wp.dot(rotvec, rotvec)
                else:
                    diff = qpos[env_idx, adr] - motion_qpos[frame_idx, adr]
                    error_sq = error_sq + diff * diff
                entry = entry + 1
            if error_sq > limit_sq:
                deviated = True
            group_i = group_i + 1

    if deviated:
        reset_terminated[env_idx] = True
    if track_reasons == 1:
        reason_masks[1, env_idx] = deviated
