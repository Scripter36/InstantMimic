"""Allocation-free Warp kernels for the optimized BeyondMimic tracking path."""

from __future__ import annotations

import warp as wp

from instantmimic.motion.warp_math import conjugate_wxyz_vec4 as _quat_conjugate
from instantmimic.motion.warp_math import inverse_wxyz_vec4 as _quat_inverse
from instantmimic.motion.warp_math import multiply_wxyz_vec4 as _quat_multiply
from instantmimic.motion.warp_math import (
    rotate_inverse_wxyz_vec4 as _quat_apply_inverse,
)
from instantmimic.motion.warp_math import rotate_wxyz_vec4 as _quat_apply


@wp.func
def _read_vec3(value: wp.array3d(dtype=float), i: int, j: int) -> wp.vec3:
    return wp.vec3(value[i, j, 0], value[i, j, 1], value[i, j, 2])


@wp.func
def _read_vec4(value: wp.array3d(dtype=float), i: int, j: int) -> wp.vec4:
    return wp.vec4(
        value[i, j, 0],
        value[i, j, 1],
        value[i, j, 2],
        value[i, j, 3],
    )


@wp.func
def _write_vec3(
    value: wp.array3d(dtype=float),
    i: int,
    j: int,
    vector: wp.vec3,
):
    value[i, j, 0] = vector[0]
    value[i, j, 1] = vector[1]
    value[i, j, 2] = vector[2]


@wp.func
def _write_vec4(
    value: wp.array3d(dtype=float),
    i: int,
    j: int,
    vector: wp.vec4,
):
    value[i, j, 0] = vector[0]
    value[i, j, 1] = vector[1]
    value[i, j, 2] = vector[2]
    value[i, j, 3] = vector[3]


@wp.func
def _uniform(state: wp.uint32, lower: float, upper: float) -> float:
    return lower + (upper - lower) * wp.randf(state)


@wp.func
def _normalized_rotation_columns(q: wp.vec4) -> wp.spatial_vector:
    norm = wp.max(wp.sqrt(wp.dot(q, q)), 1.1920928955078125e-7)
    w = q[0] / norm
    x = q[1] / norm
    y = q[2] / norm
    z = q[3] / norm
    return wp.spatial_vector(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
    )


@wp.func
def _quaternion_error_magnitude(first: wp.vec4, second: wp.vec4) -> float:
    error = _quat_multiply(first, _quat_conjugate(second))
    if error[0] < 0.0:
        error = -error
    vector_norm = wp.sqrt(
        error[1] * error[1] + error[2] * error[2] + error[3] * error[3]
    )
    return 2.0 * wp.atan2(vector_norm, wp.max(error[0], 1.0e-12))


@wp.kernel(enable_backward=False)
def action_kernel(
    incoming_action: wp.array2d(dtype=float),
    action_scale: wp.array(dtype=float),
    default_joint_position: wp.array(dtype=float),
    encoder_bias: wp.array2d(dtype=float),
    actuator_to_joint_order: wp.array(dtype=int),
    previous_action: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    target_joint_position: wp.array2d(dtype=float),
    target_ctrl: wp.array2d(dtype=float),
    ctrl: wp.array2d(dtype=float),
    action_dim: int,
):
    env_idx = wp.tid()
    for joint_idx in range(action_dim):
        previous_action[env_idx, joint_idx] = action[env_idx, joint_idx]
        action_value = incoming_action[env_idx, joint_idx]
        action[env_idx, joint_idx] = action_value
        target_joint_position[env_idx, joint_idx] = (
            default_joint_position[joint_idx]
            + action_value * action_scale[joint_idx]
            - encoder_bias[env_idx, joint_idx]
        )
    for actuator_idx in range(action_dim):
        joint_idx = actuator_to_joint_order[actuator_idx]
        target = target_joint_position[env_idx, joint_idx]
        target_ctrl[env_idx, actuator_idx] = target
        ctrl[env_idx, actuator_idx] = target


@wp.kernel(enable_backward=False)
def collision_history_kernel(
    sensordata: wp.array2d(dtype=float),
    sensor_address: int,
    substep_idx: int,
    collision_force_history: wp.array4d(dtype=float),
):
    env_idx = wp.tid()
    collision_force_history[env_idx, 0, substep_idx, 0] = sensordata[
        env_idx, sensor_address
    ]
    collision_force_history[env_idx, 0, substep_idx, 1] = sensordata[
        env_idx, sensor_address + 1
    ]
    collision_force_history[env_idx, 0, substep_idx, 2] = sensordata[
        env_idx, sensor_address + 2
    ]


@wp.kernel(enable_backward=False)
def observation_kernel(
    seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    time_steps: wp.array(dtype=wp.int64),
    motion_joint_position: wp.array2d(dtype=float),
    motion_joint_velocity: wp.array2d(dtype=float),
    motion_body_position: wp.array3d(dtype=float),
    motion_body_quaternion: wp.array3d(dtype=float),
    motion_anchor_body_index: int,
    body_ids: wp.array(dtype=int),
    joint_qpos_indices: wp.array(dtype=int),
    joint_qvel_indices: wp.array(dtype=int),
    anchor_body_id: int,
    imu_linear_address: int,
    imu_angular_address: int,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    xpos: wp.array3d(dtype=float),
    xquat: wp.array3d(dtype=float),
    sensordata: wp.array2d(dtype=float),
    default_joint_position: wp.array(dtype=float),
    encoder_bias: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    corruption_enabled: int,
    actor_observation: wp.array2d(dtype=float),
    critic_observation: wp.array2d(dtype=float),
    action_dim: int,
    tracked_body_count: int,
):
    env_idx = wp.tid()
    frame_idx = int(time_steps[env_idx])
    state = wp.rand_init(
        seed[0] + common_step_counter[0] * 17 + 1009,
        env_idx,
    )

    for joint_idx in range(action_dim):
        command_position = motion_joint_position[frame_idx, joint_idx]
        command_velocity = motion_joint_velocity[frame_idx, joint_idx]
        actor_observation[env_idx, joint_idx] = command_position
        actor_observation[env_idx, action_dim + joint_idx] = command_velocity
        critic_observation[env_idx, joint_idx] = command_position
        critic_observation[env_idx, action_dim + joint_idx] = command_velocity

    robot_anchor_position = _read_vec3(xpos, env_idx, anchor_body_id)
    robot_anchor_quaternion = _read_vec4(xquat, env_idx, anchor_body_id)
    reference_anchor_position = _read_vec3(
        motion_body_position, frame_idx, motion_anchor_body_index
    )
    reference_anchor_quaternion = _read_vec4(
        motion_body_quaternion, frame_idx, motion_anchor_body_index
    )
    anchor_position = _quat_apply_inverse(
        robot_anchor_quaternion,
        reference_anchor_position - robot_anchor_position,
    )
    anchor_orientation = _normalized_rotation_columns(
        _quat_multiply(
            _quat_inverse(robot_anchor_quaternion),
            reference_anchor_quaternion,
        )
    )
    for axis in range(3):
        actor_value = anchor_position[axis]
        if corruption_enabled == 1:
            actor_value += _uniform(state, -0.25, 0.25)
        actor_observation[env_idx, 58 + axis] = actor_value
        critic_observation[env_idx, 58 + axis] = anchor_position[axis]
    for component in range(6):
        actor_value = anchor_orientation[component]
        if corruption_enabled == 1:
            actor_value += _uniform(state, -0.05, 0.05)
        actor_observation[env_idx, 61 + component] = actor_value
        critic_observation[env_idx, 61 + component] = anchor_orientation[component]

    for body_order in range(tracked_body_count):
        body_id = body_ids[body_order]
        body_position = _quat_apply_inverse(
            robot_anchor_quaternion,
            _read_vec3(xpos, env_idx, body_id) - robot_anchor_position,
        )
        body_orientation = _normalized_rotation_columns(
            _quat_multiply(
                _quat_inverse(robot_anchor_quaternion),
                _read_vec4(xquat, env_idx, body_id),
            )
        )
        for axis in range(3):
            critic_observation[env_idx, 67 + body_order * 3 + axis] = body_position[
                axis
            ]
        for component in range(6):
            critic_observation[env_idx, 109 + body_order * 6 + component] = (
                body_orientation[component]
            )

    for axis in range(3):
        linear_velocity = sensordata[env_idx, imu_linear_address + axis]
        angular_velocity = sensordata[env_idx, imu_angular_address + axis]
        actor_linear = linear_velocity
        actor_angular = angular_velocity
        if corruption_enabled == 1:
            actor_linear += _uniform(state, -0.5, 0.5)
            actor_angular += _uniform(state, -0.2, 0.2)
        actor_observation[env_idx, 67 + axis] = actor_linear
        actor_observation[env_idx, 70 + axis] = actor_angular
        critic_observation[env_idx, 193 + axis] = linear_velocity
        critic_observation[env_idx, 196 + axis] = angular_velocity

    for joint_idx in range(action_dim):
        joint_position = (
            qpos[env_idx, joint_qpos_indices[joint_idx]]
            - default_joint_position[joint_idx]
        )
        joint_velocity = qvel[env_idx, joint_qvel_indices[joint_idx]]
        actor_joint_position = joint_position + encoder_bias[env_idx, joint_idx]
        actor_joint_velocity = joint_velocity
        if corruption_enabled == 1:
            actor_joint_position += _uniform(state, -0.01, 0.01)
            actor_joint_velocity += _uniform(state, -0.5, 0.5)
        actor_observation[env_idx, 73 + joint_idx] = actor_joint_position
        actor_observation[env_idx, 102 + joint_idx] = actor_joint_velocity
        actor_observation[env_idx, 131 + joint_idx] = action[env_idx, joint_idx]
        critic_observation[env_idx, 199 + joint_idx] = joint_position
        critic_observation[env_idx, 228 + joint_idx] = joint_velocity
        critic_observation[env_idx, 257 + joint_idx] = action[env_idx, joint_idx]


@wp.kernel(enable_backward=False)
def reward_kernel(
    time_steps: wp.array(dtype=wp.int64),
    motion_body_position: wp.array3d(dtype=float),
    motion_body_quaternion: wp.array3d(dtype=float),
    motion_body_linear_velocity: wp.array3d(dtype=float),
    motion_body_angular_velocity: wp.array3d(dtype=float),
    motion_body_indices: wp.array(dtype=int),
    motion_anchor_body_index: int,
    body_ids: wp.array(dtype=int),
    root_body_id: int,
    anchor_body_id: int,
    joint_qpos_indices: wp.array(dtype=int),
    qpos: wp.array2d(dtype=float),
    xpos: wp.array3d(dtype=float),
    xquat: wp.array3d(dtype=float),
    cvel: wp.array3d(dtype=float),
    subtree_com: wp.array3d(dtype=float),
    target_body_position: wp.array3d(dtype=float),
    target_body_quaternion: wp.array3d(dtype=float),
    action: wp.array2d(dtype=float),
    previous_action: wp.array2d(dtype=float),
    soft_joint_lower: wp.array(dtype=float),
    soft_joint_upper: wp.array(dtype=float),
    collision_force_history: wp.array4d(dtype=float),
    reward_terms: wp.array2d(dtype=float),
    reward: wp.array(dtype=float),
    action_dim: int,
    tracked_body_count: int,
    decimation: int,
    step_duration: float,
):
    env_idx = wp.tid()
    frame_idx = int(time_steps[env_idx])

    anchor_position_error = _read_vec3(xpos, env_idx, anchor_body_id) - _read_vec3(
        motion_body_position, frame_idx, motion_anchor_body_index
    )
    raw_anchor_position = wp.exp(
        -wp.dot(anchor_position_error, anchor_position_error) / (0.3 * 0.3)
    )
    anchor_orientation_error = _quaternion_error_magnitude(
        _read_vec4(xquat, env_idx, anchor_body_id),
        _read_vec4(
            motion_body_quaternion,
            frame_idx,
            motion_anchor_body_index,
        ),
    )
    raw_anchor_orientation = wp.exp(
        -(anchor_orientation_error * anchor_orientation_error) / (0.4 * 0.4)
    )

    body_position_squared = float(0.0)
    body_orientation_squared = float(0.0)
    body_linear_velocity_squared = float(0.0)
    body_angular_velocity_squared = float(0.0)
    root_subtree_com = _read_vec3(subtree_com, env_idx, root_body_id)
    for body_order in range(tracked_body_count):
        body_id = body_ids[body_order]
        motion_body_idx = motion_body_indices[body_order]
        position_error = _read_vec3(xpos, env_idx, body_id) - _read_vec3(
            target_body_position, env_idx, body_order
        )
        body_position_squared += wp.dot(position_error, position_error)
        orientation_error = _quaternion_error_magnitude(
            _read_vec4(xquat, env_idx, body_id),
            _read_vec4(target_body_quaternion, env_idx, body_order),
        )
        body_orientation_squared += orientation_error * orientation_error

        body_position = _read_vec3(xpos, env_idx, body_id)
        angular_velocity = wp.vec3(
            cvel[env_idx, body_id, 0],
            cvel[env_idx, body_id, 1],
            cvel[env_idx, body_id, 2],
        )
        linear_velocity = wp.vec3(
            cvel[env_idx, body_id, 3],
            cvel[env_idx, body_id, 4],
            cvel[env_idx, body_id, 5],
        ) - wp.cross(angular_velocity, root_subtree_com - body_position)
        linear_error = linear_velocity - _read_vec3(
            motion_body_linear_velocity, frame_idx, motion_body_idx
        )
        angular_error = angular_velocity - _read_vec3(
            motion_body_angular_velocity, frame_idx, motion_body_idx
        )
        body_linear_velocity_squared += wp.dot(linear_error, linear_error)
        body_angular_velocity_squared += wp.dot(angular_error, angular_error)

    body_count = float(tracked_body_count)
    raw_body_position = wp.exp(-(body_position_squared / body_count) / (0.3 * 0.3))
    raw_body_orientation = wp.exp(
        -(body_orientation_squared / body_count) / (0.4 * 0.4)
    )
    raw_body_linear_velocity = wp.exp(-(body_linear_velocity_squared / body_count))
    raw_body_angular_velocity = wp.exp(
        -(body_angular_velocity_squared / body_count) / (3.14 * 3.14)
    )

    raw_action_rate = float(0.0)
    raw_joint_limit = float(0.0)
    for joint_idx in range(action_dim):
        delta = action[env_idx, joint_idx] - previous_action[env_idx, joint_idx]
        raw_action_rate += delta * delta
        position = qpos[env_idx, joint_qpos_indices[joint_idx]]
        raw_joint_limit += wp.max(soft_joint_lower[joint_idx] - position, 0.0)
        raw_joint_limit += wp.max(position - soft_joint_upper[joint_idx], 0.0)

    raw_self_collision = float(0.0)
    for substep_idx in range(decimation):
        force = wp.vec3(
            collision_force_history[env_idx, 0, substep_idx, 0],
            collision_force_history[env_idx, 0, substep_idx, 1],
            collision_force_history[env_idx, 0, substep_idx, 2],
        )
        if wp.length(force) > 10.0:
            raw_self_collision += 1.0

    contribution_0 = raw_anchor_position * 0.5 * step_duration
    contribution_1 = raw_anchor_orientation * 0.5 * step_duration
    contribution_2 = raw_body_position * step_duration
    contribution_3 = raw_body_orientation * step_duration
    contribution_4 = raw_body_linear_velocity * step_duration
    contribution_5 = raw_body_angular_velocity * step_duration
    contribution_6 = raw_action_rate * -0.1 * step_duration
    contribution_7 = raw_joint_limit * -10.0 * step_duration
    contribution_8 = raw_self_collision * -10.0 * step_duration
    reward_terms[env_idx, 0] = contribution_0
    reward_terms[env_idx, 1] = contribution_1
    reward_terms[env_idx, 2] = contribution_2
    reward_terms[env_idx, 3] = contribution_3
    reward_terms[env_idx, 4] = contribution_4
    reward_terms[env_idx, 5] = contribution_5
    reward_terms[env_idx, 6] = contribution_6
    reward_terms[env_idx, 7] = contribution_7
    reward_terms[env_idx, 8] = contribution_8
    reward[env_idx] = (
        contribution_0
        + contribution_1
        + contribution_2
        + contribution_3
        + contribution_4
        + contribution_5
        + contribution_6
        + contribution_7
        + contribution_8
    )


@wp.kernel(enable_backward=False)
def termination_kernel(
    episode_length: wp.array(dtype=wp.int64),
    max_episode_length: int,
    time_steps: wp.array(dtype=wp.int64),
    motion_body_position: wp.array3d(dtype=float),
    motion_body_quaternion: wp.array3d(dtype=float),
    motion_anchor_body_index: int,
    anchor_body_id: int,
    end_effector_body_ids: wp.array(dtype=int),
    end_effector_tracked_indices: wp.array(dtype=int),
    xpos: wp.array3d(dtype=float),
    xquat: wp.array3d(dtype=float),
    target_body_position: wp.array3d(dtype=float),
    terminated: wp.array(dtype=wp.bool),
    truncated: wp.array(dtype=wp.bool),
    done_reason_mask: wp.array2d(dtype=wp.bool),
    track_done_reasons: int,
    end_effector_count: int,
):
    env_idx = wp.tid()
    frame_idx = int(time_steps[env_idx])
    timeout = episode_length[env_idx] >= wp.int64(max_episode_length)
    anchor_height = (
        wp.abs(
            xpos[env_idx, anchor_body_id, 2]
            - motion_body_position[frame_idx, motion_anchor_body_index, 2]
        )
        > 0.25
    )
    robot_gravity = _quat_apply_inverse(
        _read_vec4(xquat, env_idx, anchor_body_id),
        wp.vec3(0.0, 0.0, -1.0),
    )
    target_gravity = _quat_apply_inverse(
        _read_vec4(
            motion_body_quaternion,
            frame_idx,
            motion_anchor_body_index,
        ),
        wp.vec3(0.0, 0.0, -1.0),
    )
    anchor_orientation = wp.abs(robot_gravity[2] - target_gravity[2]) > 0.8
    end_effector_height = bool(False)
    for ee_idx in range(end_effector_count):
        body_id = end_effector_body_ids[ee_idx]
        tracked_idx = end_effector_tracked_indices[ee_idx]
        if (
            wp.abs(
                xpos[env_idx, body_id, 2]
                - target_body_position[env_idx, tracked_idx, 2]
            )
            > 0.25
        ):
            end_effector_height = True
    terminated_value = anchor_height or anchor_orientation or end_effector_height
    terminated[env_idx] = terminated_value
    truncated[env_idx] = timeout
    if track_done_reasons == 1:
        done_reason_mask[0, env_idx] = anchor_height
        done_reason_mask[1, env_idx] = anchor_orientation
        done_reason_mask[2, env_idx] = end_effector_height


@wp.kernel(enable_backward=False)
def update_current_failures_kernel(
    sample_mask: wp.array(dtype=wp.bool),
    terminated: wp.array(dtype=wp.bool),
    time_steps: wp.array(dtype=wp.int64),
    num_envs: int,
    motion_frame_count: int,
    bin_count: int,
    current_bin_failed: wp.array(dtype=float),
):
    output_bin = wp.tid()
    any_failed = bool(False)
    count = float(0.0)
    for env_idx in range(num_envs):
        if sample_mask[env_idx] and terminated[env_idx]:
            any_failed = True
            frame_idx = int(time_steps[env_idx])
            failed_bin = wp.min(
                (frame_idx * bin_count) // motion_frame_count,
                bin_count - 1,
            )
            if failed_bin == output_bin:
                count += 1.0
    # mjlab uses copy_(bincount(...)) only when at least one selected world
    # failed. A second same-step sampling call therefore overwrites, rather
    # than accumulates, the first call only when it also contains failures.
    if any_failed:
        current_bin_failed[output_bin] = count


@wp.kernel(enable_backward=False)
def build_sampling_cdf_kernel(
    bin_failed_count: wp.array(dtype=float),
    uniform_ratio: float,
    bin_count: int,
    sampling_cdf: wp.array(dtype=float),
):
    if wp.tid() != 0:
        return
    total = float(0.0)
    uniform_mass = uniform_ratio / float(bin_count)
    for bin_idx in range(bin_count):
        total += bin_failed_count[bin_idx] + uniform_mass
    cumulative = float(0.0)
    for bin_idx in range(bin_count):
        cumulative += (bin_failed_count[bin_idx] + uniform_mass) / total
        sampling_cdf[bin_idx] = cumulative
    sampling_cdf[bin_count - 1] = 1.0


@wp.kernel(enable_backward=False)
def sample_motion_frames_kernel(
    sample_mask: wp.array(dtype=wp.bool),
    seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    salt: int,
    sampling_mode: int,
    sampling_cdf: wp.array(dtype=float),
    motion_frame_count: int,
    bin_count: int,
    time_steps: wp.array(dtype=wp.int64),
):
    env_idx = wp.tid()
    if not sample_mask[env_idx]:
        return
    if sampling_mode == 0:
        time_steps[env_idx] = wp.int64(0)
        return
    state = wp.rand_init(
        seed[0] + common_step_counter[0] * 31 + salt,
        env_idx,
    )
    if sampling_mode == 1:
        frame = wp.min(
            int(wp.randf(state) * float(motion_frame_count)),
            motion_frame_count - 1,
        )
        time_steps[env_idx] = wp.int64(frame)
        return
    draw = wp.randf(state)
    sampled_bin = int(bin_count - 1)
    for bin_idx in range(bin_count):
        if draw <= sampling_cdf[bin_idx]:
            sampled_bin = bin_idx
            break
    within_bin = wp.randf(state)
    frame = int(
        (float(sampled_bin) + within_bin)
        / float(bin_count)
        * float(motion_frame_count - 1)
    )
    time_steps[env_idx] = wp.int64(frame)


@wp.kernel(enable_backward=False)
def increment_motion_and_mark_end_kernel(
    time_steps: wp.array(dtype=wp.int64),
    motion_frame_count: int,
    motion_end_mask: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()
    next_step = time_steps[env_idx] + wp.int64(1)
    time_steps[env_idx] = next_step
    motion_end_mask[env_idx] = next_step >= wp.int64(motion_frame_count)


@wp.kernel(enable_backward=False)
def reset_reference_state_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    salt: int,
    time_steps: wp.array(dtype=wp.int64),
    motion_joint_position: wp.array2d(dtype=float),
    motion_joint_velocity: wp.array2d(dtype=float),
    motion_body_position: wp.array3d(dtype=float),
    motion_body_quaternion: wp.array3d(dtype=float),
    motion_body_linear_velocity: wp.array3d(dtype=float),
    motion_body_angular_velocity: wp.array3d(dtype=float),
    motion_root_body_index: int,
    joint_qpos_indices: wp.array(dtype=int),
    joint_qvel_indices: wp.array(dtype=int),
    reset_position_bounds: wp.array2d(dtype=float),
    reset_euler_bounds: wp.array2d(dtype=float),
    reset_velocity_bounds: wp.array2d(dtype=float),
    reset_joint_bounds: wp.array(dtype=float),
    soft_joint_lower: wp.array(dtype=float),
    soft_joint_upper: wp.array(dtype=float),
    reset_episode_state: int,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    previous_action: wp.array2d(dtype=float),
    target_joint_position: wp.array2d(dtype=float),
    target_ctrl: wp.array2d(dtype=float),
    collision_force_history: wp.array4d(dtype=float),
    action_dim: int,
    decimation: int,
):
    env_idx = wp.tid()
    if not reset_mask[env_idx]:
        return
    frame_idx = int(time_steps[env_idx])
    state = wp.rand_init(
        seed[0] + common_step_counter[0] * 43 + salt,
        env_idx,
    )
    position_delta = wp.vec3(
        _uniform(state, reset_position_bounds[0, 0], reset_position_bounds[0, 1]),
        _uniform(state, reset_position_bounds[1, 0], reset_position_bounds[1, 1]),
        _uniform(state, reset_position_bounds[2, 0], reset_position_bounds[2, 1]),
    )
    euler_delta = wp.vec3(
        _uniform(state, reset_euler_bounds[0, 0], reset_euler_bounds[0, 1]),
        _uniform(state, reset_euler_bounds[1, 0], reset_euler_bounds[1, 1]),
        _uniform(state, reset_euler_bounds[2, 0], reset_euler_bounds[2, 1]),
    )
    root_velocity_delta = wp.vec3(
        _uniform(state, reset_velocity_bounds[0, 0], reset_velocity_bounds[0, 1]),
        _uniform(state, reset_velocity_bounds[1, 0], reset_velocity_bounds[1, 1]),
        _uniform(state, reset_velocity_bounds[2, 0], reset_velocity_bounds[2, 1]),
    )
    angular_velocity_delta = wp.vec3(
        _uniform(state, reset_velocity_bounds[3, 0], reset_velocity_bounds[3, 1]),
        _uniform(state, reset_velocity_bounds[4, 0], reset_velocity_bounds[4, 1]),
        _uniform(state, reset_velocity_bounds[5, 0], reset_velocity_bounds[5, 1]),
    )
    half_roll = 0.5 * euler_delta[0]
    half_pitch = 0.5 * euler_delta[1]
    half_yaw = 0.5 * euler_delta[2]
    cr = wp.cos(half_roll)
    sr = wp.sin(half_roll)
    cp = wp.cos(half_pitch)
    sp = wp.sin(half_pitch)
    cy = wp.cos(half_yaw)
    sy = wp.sin(half_yaw)
    euler_quaternion = wp.vec4(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    root_position = (
        _read_vec3(motion_body_position, frame_idx, motion_root_body_index)
        + position_delta
    )
    root_quaternion = _quat_multiply(
        euler_quaternion,
        _read_vec4(
            motion_body_quaternion,
            frame_idx,
            motion_root_body_index,
        ),
    )
    root_linear_velocity = (
        _read_vec3(
            motion_body_linear_velocity,
            frame_idx,
            motion_root_body_index,
        )
        + root_velocity_delta
    )
    root_angular_velocity_world = (
        _read_vec3(
            motion_body_angular_velocity,
            frame_idx,
            motion_root_body_index,
        )
        + angular_velocity_delta
    )
    root_angular_velocity_body = _quat_apply_inverse(
        root_quaternion,
        root_angular_velocity_world,
    )
    for axis in range(3):
        qpos[env_idx, axis] = root_position[axis]
        qvel[env_idx, axis] = root_linear_velocity[axis]
        qvel[env_idx, 3 + axis] = root_angular_velocity_body[axis]
    for component in range(4):
        qpos[env_idx, 3 + component] = root_quaternion[component]
    for joint_idx in range(action_dim):
        joint_position = motion_joint_position[frame_idx, joint_idx] + _uniform(
            state,
            reset_joint_bounds[0],
            reset_joint_bounds[1],
        )
        joint_position = wp.clamp(
            joint_position,
            soft_joint_lower[joint_idx],
            soft_joint_upper[joint_idx],
        )
        qpos[env_idx, joint_qpos_indices[joint_idx]] = joint_position
        qvel[env_idx, joint_qvel_indices[joint_idx]] = motion_joint_velocity[
            frame_idx, joint_idx
        ]
        if reset_episode_state == 1:
            action[env_idx, joint_idx] = 0.0
            previous_action[env_idx, joint_idx] = 0.0
            target_joint_position[env_idx, joint_idx] = 0.0
            target_ctrl[env_idx, joint_idx] = 0.0
    if reset_episode_state == 1:
        for substep_idx in range(decimation):
            collision_force_history[env_idx, 0, substep_idx, 0] = 0.0
            collision_force_history[env_idx, 0, substep_idx, 1] = 0.0
            collision_force_history[env_idx, 0, substep_idx, 2] = 0.0


@wp.kernel(enable_backward=False)
def command_target_kernel(
    time_steps: wp.array(dtype=wp.int64),
    motion_body_position: wp.array3d(dtype=float),
    motion_body_quaternion: wp.array3d(dtype=float),
    motion_body_indices: wp.array(dtype=int),
    motion_anchor_body_index: int,
    anchor_body_id: int,
    xpos: wp.array3d(dtype=float),
    xquat: wp.array3d(dtype=float),
    target_body_position: wp.array3d(dtype=float),
    target_body_quaternion: wp.array3d(dtype=float),
    tracked_body_count: int,
):
    env_idx = wp.tid()
    frame_idx = int(time_steps[env_idx])
    robot_anchor_position = _read_vec3(xpos, env_idx, anchor_body_id)
    robot_anchor_quaternion = _read_vec4(xquat, env_idx, anchor_body_id)
    reference_anchor_position = _read_vec3(
        motion_body_position, frame_idx, motion_anchor_body_index
    )
    reference_anchor_quaternion = _read_vec4(
        motion_body_quaternion, frame_idx, motion_anchor_body_index
    )
    delta = _quat_multiply(
        robot_anchor_quaternion,
        _quat_inverse(reference_anchor_quaternion),
    )
    yaw = wp.atan2(
        2.0 * (delta[0] * delta[3] + delta[1] * delta[2]),
        1.0 - 2.0 * (delta[2] * delta[2] + delta[3] * delta[3]),
    )
    half_yaw = 0.5 * yaw
    delta_yaw = wp.vec4(wp.cos(half_yaw), 0.0, 0.0, wp.sin(half_yaw))
    aligned_origin = wp.vec3(
        robot_anchor_position[0],
        robot_anchor_position[1],
        reference_anchor_position[2],
    )
    for body_order in range(tracked_body_count):
        motion_body_idx = motion_body_indices[body_order]
        relative = (
            _read_vec3(motion_body_position, frame_idx, motion_body_idx)
            - reference_anchor_position
        )
        target_position = aligned_origin + _quat_apply(delta_yaw, relative)
        target_quaternion = _quat_multiply(
            delta_yaw,
            _read_vec4(motion_body_quaternion, frame_idx, motion_body_idx),
        )
        _write_vec3(
            target_body_position,
            env_idx,
            body_order,
            target_position,
        )
        _write_vec4(
            target_body_quaternion,
            env_idx,
            body_order,
            target_quaternion,
        )


@wp.kernel(enable_backward=False)
def update_failure_ema_kernel(
    bin_failed_count: wp.array(dtype=float),
    current_bin_failed: wp.array(dtype=float),
    alpha: float,
    bin_count: int,
):
    bin_idx = wp.tid()
    if bin_idx < bin_count:
        bin_failed_count[bin_idx] = (
            alpha * current_bin_failed[bin_idx]
            + (1.0 - alpha) * bin_failed_count[bin_idx]
        )
        current_bin_failed[bin_idx] = 0.0


@wp.kernel(enable_backward=False)
def interval_push_kernel(
    seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    push_interval_bounds: wp.array(dtype=float),
    push_velocity_bounds: wp.array2d(dtype=float),
    root_body_id: int,
    step_duration: float,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    xpos: wp.array3d(dtype=float),
    cvel: wp.array3d(dtype=float),
    subtree_com: wp.array3d(dtype=float),
    push_time_left: wp.array(dtype=float),
):
    env_idx = wp.tid()
    remaining = push_time_left[env_idx] - step_duration
    push_time_left[env_idx] = remaining
    if remaining >= 1.0e-6:
        return
    state = wp.rand_init(
        seed[0] + common_step_counter[0] * 59 + 4001,
        env_idx,
    )
    push_time_left[env_idx] = _uniform(
        state,
        push_interval_bounds[0],
        push_interval_bounds[1],
    )
    root_position = _read_vec3(xpos, env_idx, root_body_id)
    root_subtree_com = _read_vec3(subtree_com, env_idx, root_body_id)
    angular_world = wp.vec3(
        cvel[env_idx, root_body_id, 0],
        cvel[env_idx, root_body_id, 1],
        cvel[env_idx, root_body_id, 2],
    )
    linear_world = wp.vec3(
        cvel[env_idx, root_body_id, 3],
        cvel[env_idx, root_body_id, 4],
        cvel[env_idx, root_body_id, 5],
    ) - wp.cross(angular_world, root_subtree_com - root_position)
    linear_world += wp.vec3(
        _uniform(state, push_velocity_bounds[0, 0], push_velocity_bounds[0, 1]),
        _uniform(state, push_velocity_bounds[1, 0], push_velocity_bounds[1, 1]),
        _uniform(state, push_velocity_bounds[2, 0], push_velocity_bounds[2, 1]),
    )
    angular_world += wp.vec3(
        _uniform(state, push_velocity_bounds[3, 0], push_velocity_bounds[3, 1]),
        _uniform(state, push_velocity_bounds[4, 0], push_velocity_bounds[4, 1]),
        _uniform(state, push_velocity_bounds[5, 0], push_velocity_bounds[5, 1]),
    )
    root_quaternion = wp.vec4(
        qpos[env_idx, 3],
        qpos[env_idx, 4],
        qpos[env_idx, 5],
        qpos[env_idx, 6],
    )
    angular_body = _quat_apply_inverse(
        root_quaternion,
        angular_world,
    )
    for axis in range(3):
        qvel[env_idx, axis] = linear_world[axis]
        qvel[env_idx, 3 + axis] = angular_body[axis]
