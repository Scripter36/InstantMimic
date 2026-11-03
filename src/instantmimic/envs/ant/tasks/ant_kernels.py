from __future__ import annotations

import warp as wp

from instantmimic.motion.warp_math import (
    rotate_inverse_unit_mjw_quat as _quat_rotate_inverse,
)
from instantmimic.motion.warp_math import rotate_mjw_quat as _quat_rotate


@wp.func
def _wrap_to_two_pi(value: float) -> float:
    two_pi = wp.static(2.0 * wp.pi)
    return value - wp.floor(value / two_pi) * two_pi


@wp.func
def _normalize_angle(value: float) -> float:
    return wp.atan2(wp.sin(value), wp.cos(value))


@wp.func
def _euler_xyz(quaternion: wp.quat) -> wp.vec3:
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = qw * qw - qx * qx - qy * qy + qz * qz
    roll = wp.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if wp.abs(sinp) >= 1.0:
        pitch = wp.static(wp.pi / 2.0) if sinp >= 0.0 else wp.static(-wp.pi / 2.0)
    else:
        pitch = wp.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz
    yaw = wp.atan2(siny_cosp, cosy_cosp)
    return wp.vec3(
        _wrap_to_two_pi(roll),
        _wrap_to_two_pi(pitch),
        _wrap_to_two_pi(yaw),
    )


@wp.kernel(enable_backward=False)
def stage_ant_actions_kernel(
    external_actions: wp.array2d(dtype=float),
    staged_actions: wp.array2d(dtype=float),
):
    env_idx, action_idx = wp.tid()
    staged_actions[env_idx, action_idx] = external_actions[env_idx, action_idx]


@wp.kernel(enable_backward=False)
def apply_ant_action_kernel(
    actions: wp.array2d(dtype=float),
    source_to_actuator_idx: wp.array(dtype=wp.int32),
    action_scale: float,
    ctrl: wp.array2d(dtype=float),
):
    env_idx, source_idx = wp.tid()
    actuator_idx = source_to_actuator_idx[source_idx]
    ctrl[env_idx, actuator_idx] = actions[env_idx, source_idx] * action_scale


@wp.kernel(enable_backward=False)
def compute_ant_task_kernel(
    # Writes the reward as well as the termination flags, so the environment
    # launches this from `_get_dones` and `_get_rewards` only returns the buffer.
    # That ordering is why AntEnv refuses to parallelize the two hooks.
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    targets: wp.array2d(dtype=float),
    joint_qpos_idx: wp.array(dtype=wp.int32),
    joint_qvel_idx: wp.array(dtype=wp.int32),
    joint_lower_limits: wp.array(dtype=float),
    joint_upper_limits: wp.array(dtype=float),
    actions: wp.array2d(dtype=float),
    motor_effort_ratio: wp.array(dtype=float),
    episode_length_buf: wp.array(dtype=wp.int64),
    max_episode_length: int,
    physics_dt: float,
    termination_height: float,
    heading_weight: float,
    up_weight: float,
    actions_cost_scale: float,
    energy_cost_scale: float,
    joint_velocity_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    reward: wp.array(dtype=float),
    terminated: wp.array(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
    previous_potentials: wp.array(dtype=float),
    potentials: wp.array(dtype=float),
):
    env_idx = wp.tid()

    torso_position = wp.vec3(qpos[env_idx, 0], qpos[env_idx, 1], qpos[env_idx, 2])
    torso_quaternion = wp.quat(
        qpos[env_idx, 3],
        qpos[env_idx, 4],
        qpos[env_idx, 5],
        qpos[env_idx, 6],
    )
    to_target = wp.vec3(
        targets[env_idx, 0] - torso_position[0],
        targets[env_idx, 1] - torso_position[1],
        0.0,
    )
    target_norm = wp.length(to_target)
    if target_norm < 1.0e-9:
        target_norm = 1.0e-9
    target_direction = to_target / target_norm

    up_vector = _quat_rotate(torso_quaternion, wp.vec3(0.0, 0.0, 1.0))
    heading_vector = _quat_rotate(torso_quaternion, wp.vec3(1.0, 0.0, 0.0))
    up_projection = up_vector[2]
    heading_projection = wp.dot(heading_vector, target_direction)

    actions_cost = 0.0
    electricity_cost = 0.0
    joint_limit_cost = 0.0
    for joint_idx in range(8):
        qpos_idx = joint_qpos_idx[joint_idx]
        qvel_idx = joint_qvel_idx[joint_idx]
        joint_position = qpos[env_idx, qpos_idx]
        joint_velocity = qvel[env_idx, qvel_idx]
        joint_position_scaled = (
            2.0
            * (
                (joint_position - joint_lower_limits[joint_idx])
                / (joint_upper_limits[joint_idx] - joint_lower_limits[joint_idx])
            )
            - 1.0
        )
        if joint_position_scaled > 0.98:
            joint_limit_cost = joint_limit_cost + 1.0
        action = actions[env_idx, joint_idx]
        actions_cost = actions_cost + action * action
        electricity_cost = electricity_cost + wp.abs(
            action
            * joint_velocity
            * joint_velocity_scale
            * motor_effort_ratio[joint_idx]
        )

    old_potential = potentials[env_idx]
    new_potential = -target_norm / physics_dt
    previous_potentials[env_idx] = old_potential
    potentials[env_idx] = new_potential

    if heading_projection > 0.8:
        heading_reward = heading_weight
    else:
        heading_reward = heading_weight * heading_projection / 0.8
    up_reward = up_weight if up_projection > 0.93 else 0.0

    died = torso_position[2] < termination_height
    timed_out = episode_length_buf[env_idx] >= wp.int64(max_episode_length)
    terminated[env_idx] = died
    truncated[env_idx] = timed_out
    total_reward = (
        new_potential
        - old_potential
        + alive_reward_scale
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - joint_limit_cost
    )
    if died:
        total_reward = death_cost
    reward[env_idx] = total_reward


@wp.kernel(enable_backward=False)
def compute_ant_observation_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    targets: wp.array2d(dtype=float),
    joint_qpos_idx: wp.array(dtype=wp.int32),
    joint_qvel_idx: wp.array(dtype=wp.int32),
    joint_lower_limits: wp.array(dtype=float),
    joint_upper_limits: wp.array(dtype=float),
    actions: wp.array2d(dtype=float),
    angular_velocity_scale: float,
    joint_velocity_scale: float,
    observations: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    torso_position = wp.vec3(qpos[env_idx, 0], qpos[env_idx, 1], qpos[env_idx, 2])
    torso_quaternion = wp.quat(
        qpos[env_idx, 3],
        qpos[env_idx, 4],
        qpos[env_idx, 5],
        qpos[env_idx, 6],
    )
    to_target = wp.vec3(
        targets[env_idx, 0] - torso_position[0],
        targets[env_idx, 1] - torso_position[1],
        0.0,
    )
    target_norm = wp.length(to_target)
    if target_norm < 1.0e-9:
        target_norm = 1.0e-9
    target_direction = to_target / target_norm
    up_vector = _quat_rotate(torso_quaternion, wp.vec3(0.0, 0.0, 1.0))
    heading_vector = _quat_rotate(torso_quaternion, wp.vec3(1.0, 0.0, 0.0))
    up_projection = up_vector[2]
    heading_projection = wp.dot(heading_vector, target_direction)

    velocity_world = wp.vec3(qvel[env_idx, 0], qvel[env_idx, 1], qvel[env_idx, 2])
    velocity_local = _quat_rotate_inverse(torso_quaternion, velocity_world)
    angular_velocity_local = wp.vec3(
        qvel[env_idx, 3], qvel[env_idx, 4], qvel[env_idx, 5]
    )
    euler = _euler_xyz(torso_quaternion)
    angle_to_target = (
        wp.atan2(
            targets[env_idx, 2] - torso_position[2],
            targets[env_idx, 0] - torso_position[0],
        )
        - euler[2]
    )

    observations[env_idx, 0] = torso_position[2]
    observations[env_idx, 1] = velocity_local[0]
    observations[env_idx, 2] = velocity_local[1]
    observations[env_idx, 3] = velocity_local[2]
    observations[env_idx, 4] = angular_velocity_local[0] * angular_velocity_scale
    observations[env_idx, 5] = angular_velocity_local[1] * angular_velocity_scale
    observations[env_idx, 6] = angular_velocity_local[2] * angular_velocity_scale
    observations[env_idx, 7] = _normalize_angle(euler[2])
    observations[env_idx, 8] = _normalize_angle(euler[0])
    observations[env_idx, 9] = _normalize_angle(angle_to_target)
    observations[env_idx, 10] = up_projection
    observations[env_idx, 11] = heading_projection

    for joint_idx in range(8):
        qpos_idx = joint_qpos_idx[joint_idx]
        qvel_idx = joint_qvel_idx[joint_idx]
        observations[env_idx, 12 + joint_idx] = (
            2.0
            * (
                (qpos[env_idx, qpos_idx] - joint_lower_limits[joint_idx])
                / (joint_upper_limits[joint_idx] - joint_lower_limits[joint_idx])
            )
            - 1.0
        )
        observations[env_idx, 20 + joint_idx] = (
            qvel[env_idx, qvel_idx] * joint_velocity_scale
        )
        observations[env_idx, 28 + joint_idx] = actions[env_idx, joint_idx]


@wp.kernel(enable_backward=False)
def reset_ant_state_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    default_qpos: wp.array(dtype=float),
    default_qvel: wp.array(dtype=float),
    default_ctrl: wp.array(dtype=float),
    targets: wp.array2d(dtype=float),
    physics_dt: float,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
    previous_potentials: wp.array(dtype=float),
    potentials: wp.array(dtype=float),
):
    env_idx = wp.tid()
    old_potential = potentials[env_idx]
    is_reset = reset_mask[env_idx]
    if is_reset:
        for qpos_idx in range(15):
            qpos[env_idx, qpos_idx] = default_qpos[qpos_idx]
        for qvel_idx in range(14):
            qvel[env_idx, qvel_idx] = default_qvel[qvel_idx]
        for ctrl_idx in range(8):
            ctrl[env_idx, ctrl_idx] = default_ctrl[ctrl_idx]

    dx = targets[env_idx, 0] - qpos[env_idx, 0]
    dy = targets[env_idx, 1] - qpos[env_idx, 1]
    current_potential = -wp.sqrt(dx * dx + dy * dy) / physics_dt
    if is_reset:
        previous_potentials[env_idx] = current_potential
    else:
        previous_potentials[env_idx] = old_potential
    potentials[env_idx] = current_potential


__all__ = (
    "apply_ant_action_kernel",
    "compute_ant_observation_kernel",
    "compute_ant_task_kernel",
    "reset_ant_state_kernel",
    "stage_ant_actions_kernel",
)
