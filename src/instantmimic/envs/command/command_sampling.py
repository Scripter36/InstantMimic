from __future__ import annotations

import math

import torch
import warp as wp

COMMAND_MODE_VELOCITY_HEADING = 0
COMMAND_MODE_POINT_GOAL = 1


def sample_point_goals_with_world_radius_limit_torch(
    *,
    root_pos_xy: torch.Tensor,
    distance_u: torch.Tensor,
    angle_u: torch.Tensor,
    distance_min: float,
    distance_max: float,
    world_radius_max: float,
    out_goal_pos_xy: torch.Tensor,
    out_goal_distance: torch.Tensor,
) -> None:
    radius = max(float(world_radius_max), 0.0)
    desired_min = min(float(distance_min), float(distance_max))
    desired_max = max(float(distance_min), float(distance_max))

    root_radius = torch.linalg.vector_norm(root_pos_xy, dim=-1, keepdim=True)
    base_min = torch.clamp(root_radius - radius, min=0.0)
    base_max = root_radius + radius
    desired_min_t = torch.full_like(root_radius, desired_min)
    desired_max_t = torch.full_like(root_radius, desired_max)
    feasible_min = torch.maximum(desired_min_t, base_min)
    feasible_max = torch.minimum(desired_max_t, base_max)
    has_feasible_distance = feasible_min <= feasible_max
    fallback_distance = torch.where(
        desired_max_t < base_min,
        base_min,
        torch.where(desired_min_t > base_max, base_max, feasible_min),
    )
    distance = torch.where(
        has_feasible_distance,
        feasible_min + (feasible_max - feasible_min) * distance_u,
        fallback_distance,
    )

    eps = torch.full_like(root_radius, 1.0e-6)
    all_angles = (root_radius <= eps) | (distance <= eps)
    denominator = torch.clamp(2.0 * root_radius * distance, min=1.0e-6)
    cosine_limit = (radius * radius - root_radius.square() - distance.square()) / (
        denominator
    )
    cosine_limit = torch.clamp(cosine_limit, -1.0, 1.0)
    root_angle = torch.atan2(root_pos_xy[:, 1:2], root_pos_xy[:, 0:1])
    allowed_start = torch.acos(cosine_limit)
    allowed_span = 2.0 * math.pi - 2.0 * allowed_start
    constrained_angle = root_angle + allowed_start + allowed_span * angle_u
    unconstrained_angle = -math.pi + 2.0 * math.pi * angle_u
    goal_angle = torch.where(all_angles, unconstrained_angle, constrained_angle)

    out_goal_pos_xy[:, 0:1].copy_(torch.cos(goal_angle))
    out_goal_pos_xy[:, 1:2].copy_(torch.sin(goal_angle))
    out_goal_pos_xy.mul_(distance)
    out_goal_pos_xy.add_(root_pos_xy)

    world_distance = torch.linalg.vector_norm(out_goal_pos_xy, dim=-1, keepdim=True)
    world_scale = torch.clamp(
        torch.full_like(world_distance, radius)
        / torch.clamp(world_distance, min=1.0e-6),
        max=1.0,
    )
    out_goal_pos_xy.mul_(world_scale)
    out_goal_distance.copy_(
        torch.linalg.vector_norm(out_goal_pos_xy - root_pos_xy, dim=-1, keepdim=True)
    )


@wp.kernel(enable_backward=False)
def sample_commands_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    random_seed: wp.array(dtype=int),
    command_mode: int,
    target_speed_min: float,
    target_speed_max: float,
    fixed_target_speed_enabled: int,
    fixed_target_speed: float,
    fixed_target_heading_enabled: int,
    fixed_target_heading: float,
    velocity_resample_min_steps: int,
    velocity_resample_max_steps: int,
    point_goal_distance_min: float,
    point_goal_distance_max: float,
    point_goal_world_radius_enabled: int,
    point_goal_world_radius_max: float,
    qpos: wp.array2d(dtype=float),
    target_speed: wp.array2d(dtype=float),
    target_heading: wp.array2d(dtype=float),
    goal_pos_xy: wp.array2d(dtype=float),
    prev_goal_distance: wp.array2d(dtype=float),
    velocity_command_steps_remaining: wp.array(dtype=wp.int32),
):
    env_idx = wp.tid()
    if not reset_mask[env_idx]:
        return

    state = wp.rand_init(random_seed[0] + 7919, env_idx)
    speed = target_speed_min + (target_speed_max - target_speed_min) * wp.randf(state)
    heading = -math.pi + (2.0 * math.pi) * wp.randf(state)
    if fixed_target_speed_enabled == 1:
        speed = fixed_target_speed
    if fixed_target_heading_enabled == 1:
        heading = fixed_target_heading

    target_speed[env_idx, 0] = speed
    target_heading[env_idx, 0] = heading

    if (
        command_mode == COMMAND_MODE_VELOCITY_HEADING
        and velocity_resample_max_steps > 0
    ):
        steps = velocity_resample_min_steps
        if velocity_resample_max_steps > velocity_resample_min_steps:
            span = velocity_resample_max_steps - velocity_resample_min_steps + 1
            steps = steps + wp.int32(wp.floor(wp.randf(state) * float(span)))
        velocity_command_steps_remaining[env_idx] = wp.int32(steps)
    else:
        velocity_command_steps_remaining[env_idx] = wp.int32(0)

    if command_mode == COMMAND_MODE_POINT_GOAL:
        distance_u = wp.randf(state)
        angle_u = wp.randf(state)
        distance_min = wp.min(point_goal_distance_min, point_goal_distance_max)
        distance_max = wp.max(point_goal_distance_min, point_goal_distance_max)
        distance = distance_min + (distance_max - distance_min) * distance_u
        goal_angle = -math.pi + (2.0 * math.pi) * angle_u
        root_x = qpos[env_idx, 0]
        root_y = qpos[env_idx, 1]

        if point_goal_world_radius_enabled == 1:
            radius = wp.max(point_goal_world_radius_max, 0.0)
            root_radius = wp.sqrt(root_x * root_x + root_y * root_y)
            base_min = wp.max(root_radius - radius, 0.0)
            base_max = root_radius + radius
            feasible_min = wp.max(distance_min, base_min)
            feasible_max = wp.min(distance_max, base_max)
            if feasible_min <= feasible_max:
                distance = feasible_min + (feasible_max - feasible_min) * distance_u
            elif distance_max < base_min:
                distance = base_min
            elif distance_min > base_max:
                distance = base_max
            else:
                distance = wp.clamp(distance_min, base_min, base_max)

            if root_radius <= 1.0e-6 or distance <= 1.0e-6:
                goal_angle = -math.pi + (2.0 * math.pi) * angle_u
            else:
                denominator = wp.max(2.0 * root_radius * distance, 1.0e-6)
                cosine_limit = (
                    radius * radius - root_radius * root_radius - distance * distance
                ) / denominator
                cosine_limit = wp.clamp(cosine_limit, -1.0, 1.0)
                allowed_start = wp.acos(cosine_limit)
                allowed_span = 2.0 * math.pi - 2.0 * allowed_start
                root_angle = wp.atan2(root_y, root_x)
                goal_angle = root_angle + allowed_start + allowed_span * angle_u

        goal_x = root_x + wp.cos(goal_angle) * distance
        goal_y = root_y + wp.sin(goal_angle) * distance
        if point_goal_world_radius_enabled == 1:
            radius = wp.max(point_goal_world_radius_max, 0.0)
            world_distance = wp.sqrt(goal_x * goal_x + goal_y * goal_y)
            if world_distance > radius and world_distance > 1.0e-6:
                scale = radius / world_distance
                goal_x = goal_x * scale
                goal_y = goal_y * scale

        goal_pos_xy[env_idx, 0] = goal_x
        goal_pos_xy[env_idx, 1] = goal_y
        delta_x = goal_x - root_x
        delta_y = goal_y - root_y
        prev_goal_distance[env_idx, 0] = wp.sqrt(delta_x * delta_x + delta_y * delta_y)


@wp.kernel(enable_backward=False)
def update_velocity_command_resample_mask_kernel(
    steps_remaining: wp.array(dtype=wp.int32),
    resample_mask: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()
    steps = steps_remaining[env_idx]
    if steps > 0:
        steps = steps - wp.int32(1)
        steps_remaining[env_idx] = steps
        resample_mask[env_idx] = steps <= wp.int32(0)
    else:
        resample_mask[env_idx] = False
