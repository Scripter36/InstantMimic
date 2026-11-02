from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

from instantmimic.envs.freemusco.tasks.profile import FreemuscoRunLayout
from instantmimic.envs.tracking.warp_features import (
    safe_quat_sub_mjw as safe_quat_sub_wp,
)
from instantmimic.envs.tracking.warp_features import (
    write_ortho6d,
    write_vec3,
)
from instantmimic.motion.warp_math import heading_yaw_mjw_quat as heading_yaw_quat
from instantmimic.motion.warp_math import normalize_mjw_quat as normalize_quat_wp
from instantmimic.utils.mjw_kernels.private import mul_quat, quat_inv, rot_vec_quat

POSE_REG_MODE_JOINT_QPOS = 0
POSE_REG_MODE_HEADING_FRAME_BODY_POSE = 1
REWARD_MODE_TRACK = 0
REWARD_MODE_MAXIMIZE = 1
REWARD_MODE_TRACK_DYNAMIC = 2
REWARD_MODE_RAW_FORWARD_VELOCITY = 3
TORSO_TILT_SOURCE_BODY = 0
TORSO_TILT_SOURCE_ROOT = 1
TORSO_TILT_MODE_XY_NORM = 0
TORSO_TILT_MODE_ROOT_ROT_Y_ABS = 1

# Slots of the packed reward-weight array. The twelve weights are configuration:
# constant for a run, and formerly rebuilt as twelve Python floats and passed as
# twelve launch scalars on every reward evaluation.
REWARD_WEIGHT_TRACK_SIGMA = 0
REWARD_WEIGHT_VELOCITY_SCALE = 1
REWARD_WEIGHT_UPRIGHT_SCALE = 2
REWARD_WEIGHT_HEAD_HEIGHT_TARGET = 3
REWARD_WEIGHT_HEAD_HEIGHT_SIGMA = 4
REWARD_WEIGHT_HEAD_HEIGHT_SCALE = 5
REWARD_WEIGHT_DIRECTION_SCALE = 6
REWARD_WEIGHT_PELVIS_HEIGHT_THRESHOLD = 7
REWARD_WEIGHT_PELVIS_HEIGHT_PENALTY_SCALE = 8
REWARD_WEIGHT_POSE_REGULARIZATION_SCALE = 9
REWARD_WEIGHT_POSE_BODY_POS_WEIGHT = 10
REWARD_WEIGHT_POSE_BODY_ROT_WEIGHT = 11
REWARD_WEIGHT_COUNT = 12


@wp.func
def generic_torso_tilt_angle(
    torso_quat: wp.quat,
    torso_upright_local: wp.vec3,
) -> float:
    torso_q = normalize_quat_wp(torso_quat)
    torso_upright_world = rot_vec_quat(torso_upright_local, torso_q)
    return wp.acos(wp.clamp(torso_upright_world[2], -1.0, 1.0))


@wp.kernel(enable_backward=False)
def generic_apply_action_kernel(
    actions: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
):
    env_idx, act_idx = wp.tid()
    ctrl[env_idx, act_idx] = wp.clamp(actions[env_idx, act_idx], 0.0, 1.0)


@wp.kernel(enable_backward=False)
def generic_native_observation_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    act: wp.array2d(dtype=float),
    actuator_force: wp.array2d(dtype=float),
    actuator_length: wp.array2d(dtype=float),
    actuator_velocity: wp.array2d(dtype=float),
    head_body_id: int,
    pelvis_geom_id: int,
    foot_body_id_0: int,
    foot_body_id_1: int,
    nq: int,
    nv: int,
    na: int,
    nu: int,
    qpos_observation_skip_idx: int,
    forward_velocity_obs: wp.array2d(dtype=float),
    target_forward_velocity_obs: wp.array2d(dtype=float),
    include_target_forward_velocity_obs: int,
    include_velocity_error_obs: int,
    obs: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    ofs = 0
    obs[env_idx, ofs] = xpos[env_idx, head_body_id][2]
    ofs = ofs + 1
    obs[env_idx, ofs] = geom_xpos[env_idx, pelvis_geom_id][2]
    ofs = ofs + 1
    obs[env_idx, ofs] = xpos[env_idx, foot_body_id_0][2]
    ofs = ofs + 1
    obs[env_idx, ofs] = xpos[env_idx, foot_body_id_1][2]
    ofs = ofs + 1

    for j in range(nq):
        if j != qpos_observation_skip_idx:
            obs[env_idx, ofs] = qpos[env_idx, j]
            ofs = ofs + 1
    for j in range(nv):
        obs[env_idx, ofs] = wp.clamp(qvel[env_idx, j], -100.0, 100.0)
        ofs = ofs + 1
    for j in range(na):
        obs[env_idx, ofs] = wp.clamp(act[env_idx, j], -100.0, 100.0)
        ofs = ofs + 1
    for j in range(nu):
        obs[env_idx, ofs] = wp.clamp(actuator_force[env_idx, j] / 1000.0, -100.0, 100.0)
        ofs = ofs + 1
    for j in range(nu):
        obs[env_idx, ofs] = actuator_length[env_idx, j]
        ofs = ofs + 1
    for j in range(nu):
        obs[env_idx, ofs] = wp.clamp(actuator_velocity[env_idx, j], -100.0, 100.0)
        ofs = ofs + 1

    fwd_vel = forward_velocity_obs[env_idx, 0]
    target_velocity = target_forward_velocity_obs[env_idx, 0]
    obs[env_idx, ofs] = fwd_vel
    ofs = ofs + 1
    if include_target_forward_velocity_obs != 0:
        obs[env_idx, ofs] = target_velocity
        ofs = ofs + 1
    if include_velocity_error_obs != 0:
        obs[env_idx, ofs] = target_velocity - fwd_vel


@wp.kernel(enable_backward=False)
def generic_fast_muscle_observation_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    muscle_activation: wp.array2d(dtype=wp.float32),
    muscle_force: wp.array2d(dtype=wp.float32),
    muscle_length: wp.array2d(dtype=wp.float32),
    muscle_velocity: wp.array2d(dtype=wp.float32),
    head_body_id: int,
    pelvis_geom_id: int,
    foot_body_id_0: int,
    foot_body_id_1: int,
    nq: int,
    nv: int,
    muscle_count: int,
    qpos_observation_skip_idx: int,
    forward_velocity_obs: wp.array2d(dtype=float),
    target_forward_velocity_obs: wp.array2d(dtype=float),
    include_target_forward_velocity_obs: int,
    include_velocity_error_obs: int,
    obs: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    ofs = 0
    obs[env_idx, ofs] = xpos[env_idx, head_body_id][2]
    ofs = ofs + 1
    obs[env_idx, ofs] = geom_xpos[env_idx, pelvis_geom_id][2]
    ofs = ofs + 1
    obs[env_idx, ofs] = xpos[env_idx, foot_body_id_0][2]
    ofs = ofs + 1
    obs[env_idx, ofs] = xpos[env_idx, foot_body_id_1][2]
    ofs = ofs + 1
    for j in range(nq):
        if j != qpos_observation_skip_idx:
            obs[env_idx, ofs] = qpos[env_idx, j]
            ofs = ofs + 1
    for j in range(nv):
        obs[env_idx, ofs] = wp.clamp(qvel[env_idx, j], -100.0, 100.0)
        ofs = ofs + 1
    for j in range(muscle_count):
        obs[env_idx, ofs] = muscle_activation[env_idx, j]
        ofs = ofs + 1
    for j in range(muscle_count):
        obs[env_idx, ofs] = wp.clamp(muscle_force[env_idx, j] / 1000.0, -100.0, 100.0)
        ofs = ofs + 1
    for j in range(muscle_count):
        obs[env_idx, ofs] = muscle_length[env_idx, j]
        ofs = ofs + 1
    for j in range(muscle_count):
        obs[env_idx, ofs] = wp.clamp(muscle_velocity[env_idx, j], -100.0, 100.0)
        ofs = ofs + 1

    fwd_vel = forward_velocity_obs[env_idx, 0]
    target_velocity = target_forward_velocity_obs[env_idx, 0]
    obs[env_idx, ofs] = fwd_vel
    ofs = ofs + 1
    if include_target_forward_velocity_obs != 0:
        obs[env_idx, ofs] = target_velocity
        ofs = ofs + 1
    if include_velocity_error_obs != 0:
        obs[env_idx, ofs] = target_velocity - fwd_vel


@wp.kernel(enable_backward=False)
def generic_fast_observation_no_muscle_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    head_body_id: int,
    pelvis_geom_id: int,
    foot_body_id_0: int,
    foot_body_id_1: int,
    nq: int,
    nv: int,
    qpos_observation_skip_idx: int,
    forward_velocity_obs: wp.array2d(dtype=float),
    target_forward_velocity_obs: wp.array2d(dtype=float),
    include_target_forward_velocity_obs: int,
    include_velocity_error_obs: int,
    obs: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    ofs = 0
    obs[env_idx, ofs] = xpos[env_idx, head_body_id][2]
    ofs = ofs + 1
    obs[env_idx, ofs] = geom_xpos[env_idx, pelvis_geom_id][2]
    ofs = ofs + 1
    obs[env_idx, ofs] = xpos[env_idx, foot_body_id_0][2]
    ofs = ofs + 1
    obs[env_idx, ofs] = xpos[env_idx, foot_body_id_1][2]
    ofs = ofs + 1
    for j in range(nq):
        if j != qpos_observation_skip_idx:
            obs[env_idx, ofs] = qpos[env_idx, j]
            ofs = ofs + 1
    for j in range(nv):
        obs[env_idx, ofs] = wp.clamp(qvel[env_idx, j], -100.0, 100.0)
        ofs = ofs + 1

    fwd_vel = forward_velocity_obs[env_idx, 0]
    target_velocity = target_forward_velocity_obs[env_idx, 0]
    obs[env_idx, ofs] = fwd_vel
    ofs = ofs + 1
    if include_target_forward_velocity_obs != 0:
        obs[env_idx, ofs] = target_velocity
        ofs = ofs + 1
    if include_velocity_error_obs != 0:
        obs[env_idx, ofs] = target_velocity - fwd_vel


@wp.kernel(enable_backward=False)
def generic_heading_frame_body_state_observation_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    root_body_id: int,
    body_state_ofs: int,
    obs: wp.array2d(dtype=float),
):
    env_idx, body_idx = wp.tid()
    root_q = normalize_quat_wp(xquat[env_idx, root_body_id])
    yaw_inv = quat_inv(heading_yaw_quat(root_q))
    root_pos = xpos[env_idx, root_body_id]
    root_pos_xz = wp.vec3(root_pos[0], root_pos[1], 0.0)
    body_pos = xpos[env_idx, body_idx]
    body_quat = normalize_quat_wp(xquat[env_idx, body_idx])
    body_cvel = cvel[env_idx, body_idx]
    body_linvel = wp.vec3(body_cvel[3], body_cvel[4], body_cvel[5])
    body_angvel = wp.vec3(body_cvel[0], body_cvel[1], body_cvel[2])
    pos_rel = rot_vec_quat(body_pos - root_pos_xz, yaw_inv)
    quat_rel = normalize_quat_wp(mul_quat(yaw_inv, body_quat))
    ofs = body_state_ofs + body_idx * 15
    write_vec3(obs, env_idx, ofs, pos_rel, 1.0)
    write_ortho6d(obs, env_idx, ofs + 3, quat_rel)
    write_vec3(obs, env_idx, ofs + 9, rot_vec_quat(body_linvel, yaw_inv), 0.1)
    write_vec3(obs, env_idx, ofs + 12, rot_vec_quat(body_angvel, yaw_inv), 0.1)


@wp.kernel(enable_backward=False)
def generic_reward_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    qpos: wp.array2d(dtype=float),
    rest_qpos: wp.array(dtype=float),
    non_root_joint_qpos_idx: wp.array(dtype=int),
    num_non_root_joints: int,
    rest_body_pos_heading: wp.array(dtype=wp.vec3),
    rest_body_quat_heading: wp.array(dtype=wp.quat),
    nbody: int,
    root_body_id: int,
    head_body_id: int,
    torso_body_id: int,
    pelvis_geom_id: int,
    torso_tilt_source: int,
    root_tilt_qpos_idx_0: int,
    root_tilt_qpos_idx_1: int,
    torso_upright_local: wp.vec3,
    root_yaw_qpos_idx: int,
    reward_mode: int,
    reward_pose_regularization_mode: int,
    forward_velocity_obs: wp.array2d(dtype=float),
    target_forward_velocity_obs: wp.array2d(dtype=float),
    reward_weights: wp.array(dtype=float),  # (REWARD_WEIGHT_COUNT,)
    reward_buf: wp.array(dtype=float),
):
    env_idx = wp.tid()
    reward_track_sigma = reward_weights[REWARD_WEIGHT_TRACK_SIGMA]
    reward_velocity_scale = reward_weights[REWARD_WEIGHT_VELOCITY_SCALE]
    reward_upright_scale = reward_weights[REWARD_WEIGHT_UPRIGHT_SCALE]
    reward_head_height_target = reward_weights[REWARD_WEIGHT_HEAD_HEIGHT_TARGET]
    reward_head_height_sigma = reward_weights[REWARD_WEIGHT_HEAD_HEIGHT_SIGMA]
    reward_head_height_scale = reward_weights[REWARD_WEIGHT_HEAD_HEIGHT_SCALE]
    reward_direction_scale = reward_weights[REWARD_WEIGHT_DIRECTION_SCALE]
    reward_pelvis_height_threshold = reward_weights[
        REWARD_WEIGHT_PELVIS_HEIGHT_THRESHOLD
    ]
    reward_pelvis_height_penalty_scale = reward_weights[
        REWARD_WEIGHT_PELVIS_HEIGHT_PENALTY_SCALE
    ]
    reward_pose_regularization_scale = reward_weights[
        REWARD_WEIGHT_POSE_REGULARIZATION_SCALE
    ]
    reward_pose_regularization_body_pos_weight = reward_weights[
        REWARD_WEIGHT_POSE_BODY_POS_WEIGHT
    ]
    reward_pose_regularization_body_rot_weight = reward_weights[
        REWARD_WEIGHT_POSE_BODY_ROT_WEIGHT
    ]
    fwd_vel = forward_velocity_obs[env_idx, 0]
    target_velocity = target_forward_velocity_obs[env_idx, 0]
    if reward_mode == REWARD_MODE_TRACK or reward_mode == REWARD_MODE_TRACK_DYNAMIC:
        vel_err = fwd_vel - target_velocity
        vel_reward = reward_velocity_scale * wp.exp(
            -vel_err * vel_err / (reward_track_sigma * reward_track_sigma)
        )
    elif reward_mode == REWARD_MODE_RAW_FORWARD_VELOCITY:
        vel_reward = reward_velocity_scale * fwd_vel
    else:
        vel_reward = reward_velocity_scale * wp.clamp(
            fwd_vel, -target_velocity, target_velocity
        )

    tilt_angle = 0.0
    if torso_tilt_source == TORSO_TILT_SOURCE_BODY:
        tilt_angle = generic_torso_tilt_angle(
            xquat[env_idx, torso_body_id],
            torso_upright_local,
        )
    else:
        tilt_x = qpos[env_idx, root_tilt_qpos_idx_0]
        tilt_y = qpos[env_idx, root_tilt_qpos_idx_1]
        tilt_angle = wp.sqrt(tilt_x * tilt_x + tilt_y * tilt_y)
    upright_reward = reward_upright_scale * wp.exp(-(tilt_angle * tilt_angle))

    head_z = xpos[env_idx, head_body_id][2]
    head_height_err = head_z - reward_head_height_target
    head_height_reward = reward_head_height_scale * wp.exp(
        -head_height_err
        * head_height_err
        / (reward_head_height_sigma * reward_head_height_sigma)
    )
    root_yaw = qpos[env_idx, root_yaw_qpos_idx]
    direction_reward = reward_direction_scale * (0.5 * (wp.cos(root_yaw) + 1.0))

    pelvis_z = geom_xpos[env_idx, pelvis_geom_id][2]
    height_deficit = reward_pelvis_height_threshold - pelvis_z
    pelvis_height_penalty = 0.0
    if height_deficit > 0.0:
        pelvis_height_penalty = reward_pelvis_height_penalty_scale * height_deficit

    pose_penalty = float(0.0)
    if reward_pose_regularization_scale > 0.0:
        pose_err_sum = float(0.0)
        if reward_pose_regularization_mode == POSE_REG_MODE_JOINT_QPOS:
            for j in range(num_non_root_joints):
                idx = non_root_joint_qpos_idx[j]
                diff = qpos[env_idx, idx] - rest_qpos[idx]
                pose_err_sum = pose_err_sum + diff * diff
        else:
            root_pos = xpos[env_idx, root_body_id]
            root_q = normalize_quat_wp(xquat[env_idx, root_body_id])
            yaw_inv = quat_inv(heading_yaw_quat(root_q))
            for body_idx in range(1, nbody):
                sim_local_pos = rot_vec_quat(
                    xpos[env_idx, body_idx] - root_pos, yaw_inv
                )
                sim_local_quat = normalize_quat_wp(
                    mul_quat(yaw_inv, normalize_quat_wp(xquat[env_idx, body_idx]))
                )
                pos_diff = sim_local_pos - rest_body_pos_heading[body_idx]
                pos_err = (
                    pos_diff[0] * pos_diff[0]
                    + pos_diff[1] * pos_diff[1]
                    + pos_diff[2] * pos_diff[2]
                )
                rot_diff = safe_quat_sub_wp(
                    sim_local_quat, rest_body_quat_heading[body_idx]
                )
                rot_err = (
                    rot_diff[0] * rot_diff[0]
                    + rot_diff[1] * rot_diff[1]
                    + rot_diff[2] * rot_diff[2]
                )
                pose_err_sum = pose_err_sum + (
                    reward_pose_regularization_body_pos_weight * pos_err
                    + reward_pose_regularization_body_rot_weight * rot_err
                )
        pose_penalty = reward_pose_regularization_scale * wp.sqrt(pose_err_sum)

    reward_buf[env_idx] = (
        vel_reward
        + upright_reward
        + head_height_reward
        + direction_reward
        - pelvis_height_penalty
        - pose_penalty
    )


@wp.kernel(enable_backward=False)
def generic_dones_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    episode_length_buf: wp.array(dtype=wp.int64),
    head_body_id: int,
    torso_body_id: int,
    pelvis_geom_id: int,
    torso_tilt_source: int,
    root_tilt_qpos_idx_0: int,
    root_tilt_qpos_idx_1: int,
    torso_tilt_mode: int,
    torso_upright_local: wp.vec3,
    min_head_height: float,
    min_pelvis_height: float,
    max_torso_tilt_rad: float,
    max_joint_velocity: float,
    nv: int,
    max_episode_length: int,
    terminated: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()
    head_z = xpos[env_idx, head_body_id][2]
    pelvis_z = geom_xpos[env_idx, pelvis_geom_id][2]
    if torso_tilt_source == TORSO_TILT_SOURCE_BODY:
        tilt_angle = generic_torso_tilt_angle(
            xquat[env_idx, torso_body_id],
            torso_upright_local,
        )
        torso_tilt_violation = tilt_angle > max_torso_tilt_rad
    else:
        tilt_x = qpos[env_idx, root_tilt_qpos_idx_0]
        tilt_y = qpos[env_idx, root_tilt_qpos_idx_1]
        tilt_angle = wp.sqrt(tilt_x * tilt_x + tilt_y * tilt_y)
        torso_tilt_violation = tilt_angle > max_torso_tilt_rad
        if torso_tilt_mode == TORSO_TILT_MODE_ROOT_ROT_Y_ABS:
            torso_tilt_violation = wp.abs(tilt_y) > max_torso_tilt_rad

    died = (
        head_z < min_head_height or pelvis_z < min_pelvis_height or torso_tilt_violation
    )
    if max_joint_velocity > 0.0:
        for i in range(nv):
            if wp.abs(qvel[env_idx, i]) > max_joint_velocity:
                died = True
                break
    terminated[env_idx] = died
    ep_len = int(episode_length_buf[env_idx])
    time_out[env_idx] = ep_len >= max_episode_length


@wp.kernel(enable_backward=False)
def generic_dones_with_reason_masks_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    episode_length_buf: wp.array(dtype=wp.int64),
    head_body_id: int,
    torso_body_id: int,
    pelvis_geom_id: int,
    torso_tilt_source: int,
    root_tilt_qpos_idx_0: int,
    root_tilt_qpos_idx_1: int,
    torso_tilt_mode: int,
    torso_upright_local: wp.vec3,
    min_head_height: float,
    min_pelvis_height: float,
    max_torso_tilt_rad: float,
    max_joint_velocity: float,
    nv: int,
    max_episode_length: int,
    reason_masks: wp.array2d(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()
    head_height_violation = xpos[env_idx, head_body_id][2] < min_head_height
    pelvis_height_violation = geom_xpos[env_idx, pelvis_geom_id][2] < min_pelvis_height
    if torso_tilt_source == TORSO_TILT_SOURCE_BODY:
        tilt_angle = generic_torso_tilt_angle(
            xquat[env_idx, torso_body_id],
            torso_upright_local,
        )
        torso_tilt_violation = tilt_angle > max_torso_tilt_rad
    else:
        tilt_x = qpos[env_idx, root_tilt_qpos_idx_0]
        tilt_y = qpos[env_idx, root_tilt_qpos_idx_1]
        torso_tilt_violation = (
            wp.sqrt(tilt_x * tilt_x + tilt_y * tilt_y) > max_torso_tilt_rad
        )
        if torso_tilt_mode == TORSO_TILT_MODE_ROOT_ROT_Y_ABS:
            torso_tilt_violation = wp.abs(tilt_y) > max_torso_tilt_rad
    joint_velocity_violation = int(0)
    if max_joint_velocity > 0.0:
        for i in range(nv):
            if wp.abs(qvel[env_idx, i]) > max_joint_velocity:
                joint_velocity_violation = int(1)
                break
    reason_masks[0, env_idx] = head_height_violation
    reason_masks[1, env_idx] = pelvis_height_violation
    reason_masks[2, env_idx] = torso_tilt_violation
    reason_masks[3, env_idx] = joint_velocity_violation != 0
    terminated[env_idx] = (
        head_height_violation
        or pelvis_height_violation
        or torso_tilt_violation
        or joint_velocity_violation != 0
    )
    ep_len = int(episode_length_buf[env_idx])
    time_out[env_idx] = ep_len >= max_episode_length


@wp.kernel(enable_backward=False)
def generic_reset_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    random_seed: wp.array(dtype=int),
    randomize_reset_joints: int,
    reset_joint_range_scale: float,
    dynamic_target_forward_velocity: int,
    target_forward_velocity: float,
    target_forward_velocity_min: float,
    target_forward_velocity_max: float,
    has_target_forward_velocity_override: int,
    target_forward_velocity_override: float,
    reset_qpos: wp.array(dtype=float),
    non_root_joint_qpos_idx: wp.array(dtype=int),
    non_root_joint_low: wp.array(dtype=float),
    non_root_joint_high: wp.array(dtype=float),
    num_non_root_joints: int,
    nq: int,
    nv: int,
    nu: int,
    na: int,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
    act: wp.array2d(dtype=float),
    actions: wp.array2d(dtype=float),
    target_forward_velocity_obs: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    if not reset_mask[env_idx]:
        return
    state = wp.rand_init(random_seed[0], env_idx)
    for j in range(nq):
        qpos[env_idx, j] = reset_qpos[j]
    if randomize_reset_joints != 0 and reset_joint_range_scale > 0.0:
        for j in range(num_non_root_joints):
            idx = non_root_joint_qpos_idx[j]
            lo = non_root_joint_low[j] * reset_joint_range_scale
            hi = non_root_joint_high[j] * reset_joint_range_scale
            qpos[env_idx, idx] = lo + (hi - lo) * wp.randf(state)
    for j in range(nv):
        qvel[env_idx, j] = 0.0
    for j in range(nu):
        ctrl[env_idx, j] = 0.0
        actions[env_idx, j] = 0.0
    for j in range(na):
        act[env_idx, j] = 0.0

    resolved_target_velocity = target_forward_velocity
    if dynamic_target_forward_velocity != 0:
        if has_target_forward_velocity_override != 0:
            resolved_target_velocity = target_forward_velocity_override
        elif target_forward_velocity_max > target_forward_velocity_min:
            resolved_target_velocity = target_forward_velocity_min + (
                target_forward_velocity_max - target_forward_velocity_min
            ) * wp.randf(state)
        else:
            resolved_target_velocity = target_forward_velocity_min
    target_forward_velocity_obs[env_idx, 0] = resolved_target_velocity


@dataclass(frozen=True)
class FreemuscoRunKernelBundle:
    apply_action: Any
    observation: Any
    fast_observation: Any
    fast_observation_no_muscle: Any
    body_state_observation: Any
    reward: Any
    dones: Any
    dones_with_reason_masks: Any
    reset: Any


def build_freemusco_run_kernel_bundle(
    layout: FreemuscoRunLayout,
    *,
    uses_fast_muscle: bool,
    include_muscle_observation_blocks: bool,
) -> FreemuscoRunKernelBundle:
    """Build the single generic kernel bundle from resolved layout data."""
    layout.validate()
    fast_observation = (
        generic_fast_muscle_observation_kernel
        if include_muscle_observation_blocks
        else generic_fast_observation_no_muscle_kernel
    )
    return FreemuscoRunKernelBundle(
        apply_action=generic_apply_action_kernel,
        observation=generic_native_observation_kernel,
        fast_observation=fast_observation,
        fast_observation_no_muscle=generic_fast_observation_no_muscle_kernel,
        body_state_observation=generic_heading_frame_body_state_observation_kernel,
        reward=generic_reward_kernel,
        dones=generic_dones_kernel,
        dones_with_reason_masks=generic_dones_with_reason_masks_kernel,
        reset=generic_reset_kernel,
    )


class FreemuscoRunRuntime:
    """Model-independent Run execution for native and fast-muscle control."""

    def _get_common_kernel_inputs(self, env: Any) -> list[Any]:
        return [
            env.mjw_data.xpos,
            env.mjw_data.xquat,
            env.mjw_data.geom_xpos,
            env.mjw_data.qpos,
            env._reset_qpos_wp,
            env._non_root_joint_qpos_idx_wp,
            env._num_non_root_joints,
            env._rest_body_pos_heading_wp,
            env._rest_body_quat_heading_wp,
            int(env.model.nbody),
            env.root_body_id,
            env.head_body_id,
            env.torso_body_id,
            env.pelvis_geom_id,
            env._torso_tilt_source_int,
            env._root_tilt_qpos_idx_0,
            env._root_tilt_qpos_idx_1,
            env._torso_upright_local,
            env._root_yaw_qpos_idx,
            env._reward_mode_int,
            env._reward_pose_regularization_mode_int,
            env._forward_velocity_obs_wp,
            env._target_forward_velocity_obs_wp,
            env._reward_weights_wp,
        ]

    def _apply_action(self, env: Any) -> None:
        wp.launch(
            env._kernel_bundle.apply_action,
            dim=(env.num_envs, env._nu),
            inputs=[env.actions],
            outputs=[env.mjw_data.ctrl],
        )

    def _get_observations(self, env: Any, out_obs: torch.Tensor | None = None) -> dict:
        target, observations = env._resolve_observation_output(out_obs)
        env._update_forward_velocity_observation()
        if not env._uses_fast_muscle:
            # State-prefix stepping leaves action-dependent forces for their
            # consumer. Refresh them at the integrated activation state.
            env._run_mjw_actuation()
            wp.launch(
                env._kernel_bundle.observation,
                dim=(env.num_envs,),
                inputs=[
                    env.mjw_data.xpos,
                    env.mjw_data.geom_xpos,
                    env.mjw_data.qpos,
                    env.mjw_data.qvel,
                    env.mjw_data.act,
                    env.mjw_data.actuator_force,
                    env.mjw_data.actuator_length,
                    env.mjw_data.actuator_velocity,
                    env.head_body_id,
                    env.pelvis_geom_id,
                    env.foot_body_id_0,
                    env.foot_body_id_1,
                    env._nq,
                    env._nv,
                    env._na,
                    env._nu,
                    env._qpos_observation_skip_idx,
                    env._forward_velocity_obs_wp,
                    env._target_forward_velocity_obs_wp,
                    int(env._include_target_forward_velocity_obs),
                    int(env._include_velocity_error_obs),
                ],
                outputs=[target],
            )
            return observations

        if env._include_muscle_observation_blocks:
            muscle_length, muscle_velocity, muscle_force, muscle_activation = (
                env.fast_muscle.update_muscle_observables()
            )
            wp.launch(
                env._kernel_bundle.fast_observation,
                dim=(env.num_envs,),
                inputs=[
                    env.mjw_data.xpos,
                    env.mjw_data.geom_xpos,
                    env.mjw_data.qpos,
                    env.mjw_data.qvel,
                    muscle_activation,
                    muscle_force,
                    muscle_length,
                    muscle_velocity,
                    env.head_body_id,
                    env.pelvis_geom_id,
                    env.foot_body_id_0,
                    env.foot_body_id_1,
                    env._nq,
                    env._nv,
                    int(env.fast_muscle.muscle_count),
                    env._qpos_observation_skip_idx,
                    env._forward_velocity_obs_wp,
                    env._target_forward_velocity_obs_wp,
                    int(env._include_target_forward_velocity_obs),
                    int(env._include_velocity_error_obs),
                ],
                outputs=[target[:, : env._body_state_obs_ofs]],
            )
        else:
            wp.launch(
                env._kernel_bundle.fast_observation_no_muscle,
                dim=(env.num_envs,),
                inputs=[
                    env.mjw_data.xpos,
                    env.mjw_data.geom_xpos,
                    env.mjw_data.qpos,
                    env.mjw_data.qvel,
                    env.head_body_id,
                    env.pelvis_geom_id,
                    env.foot_body_id_0,
                    env.foot_body_id_1,
                    env._nq,
                    env._nv,
                    env._qpos_observation_skip_idx,
                    env._forward_velocity_obs_wp,
                    env._target_forward_velocity_obs_wp,
                    int(env._include_target_forward_velocity_obs),
                    int(env._include_velocity_error_obs),
                ],
                outputs=[target[:, : env._body_state_obs_ofs]],
            )

        if env._body_state_obs_dim > 0:
            wp.launch(
                env._kernel_bundle.body_state_observation,
                dim=(env.num_envs, int(env.model.nbody)),
                inputs=[
                    env.mjw_data.xpos,
                    env.mjw_data.xquat,
                    env.mjw_data.cvel,
                    env.root_body_id,
                    env._body_state_obs_ofs,
                ],
                outputs=[target],
            )
        offset = env._obs_dim
        if env.fast_muscle.prev_action_obs_dim > 0:
            target[:, offset : offset + env.fast_muscle.prev_action_obs_dim].copy_(
                env.fast_muscle.get_prev_action_observation_tensor()
            )
            offset += env.fast_muscle.prev_action_obs_dim
        if env.fast_muscle.applied_torque_obs_dim > 0:
            env.fast_muscle.update_applied_torque_observation()
            offset = env.fast_muscle.write_applied_torque_block(target, offset)
        if env.fast_muscle.muscle_geometry_observation_dim > 0:
            env.fast_muscle.write_muscle_geometry_observation(target, offset)
        return observations

    def _get_rewards(self, env: Any) -> torch.Tensor:
        env._update_forward_velocity_observation()
        inputs = self._get_common_kernel_inputs(env)
        wp.launch(
            env._kernel_bundle.reward,
            dim=(env.num_envs,),
            inputs=inputs,
            outputs=[env.reward_buf],
        )
        return env.reward_buf

    def _get_dones(self, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
        common = [
            env.mjw_data.xpos,
            env.mjw_data.xquat,
            env.mjw_data.geom_xpos,
            env.mjw_data.qpos,
            env.mjw_data.qvel,
            env.episode_length_buf,
            env.head_body_id,
            env.torso_body_id,
            env.pelvis_geom_id,
            env._torso_tilt_source_int,
            env._root_tilt_qpos_idx_0,
            env._root_tilt_qpos_idx_1,
            env._termination_torso_tilt_mode_int,
            env._torso_upright_local,
            float(env.cfg.min_head_height),
            float(env.cfg.min_pelvis_height),
            float(env.cfg.max_torso_tilt_rad),
            float(env.cfg.get("max_joint_velocity", 0.0)),
            env._nv,
            env.max_episode_length,
        ]
        if env._track_done_reasons:
            wp.launch(
                env._kernel_bundle.dones_with_reason_masks,
                dim=(env.num_envs,),
                inputs=common,
                outputs=[
                    env._done_reason_mask_buf,
                    env.reset_terminated,
                    env.reset_time_outs,
                ],
            )
        else:
            wp.launch(
                env._kernel_bundle.dones,
                dim=(env.num_envs,),
                inputs=common,
                outputs=[env.reset_terminated, env.reset_time_outs],
            )
        return env.reset_terminated, env.reset_time_outs

    def _launch_reset_kernel(self, env: Any, reset_mask: torch.Tensor) -> None:
        wp.launch(
            env._kernel_bundle.reset,
            dim=(env.num_envs,),
            inputs=[
                reset_mask,
                env._sample_reset_seed(),
                int(env._reset_randomize_joints_enabled),
                float(env._reset_joint_range_scale),
                int(env._dynamic_target_forward_velocity_enabled),
                env._fixed_target_forward_velocity,
                env._target_forward_velocity_min,
                env._target_forward_velocity_max,
                int(env._has_target_forward_velocity_override),
                env._target_forward_velocity_override,
                env._reset_qpos_wp,
                env._non_root_joint_qpos_idx_wp,
                env._non_root_joint_low_wp,
                env._non_root_joint_high_wp,
                env._num_non_root_joints,
                env._nq,
                env._nv,
                env._nu,
                env._na,
            ],
            outputs=[
                env.mjw_data.qpos,
                env.mjw_data.qvel,
                env.mjw_data.ctrl,
                env.mjw_data.act,
                env.actions,
                env._target_forward_velocity_obs_wp,
            ],
        )
