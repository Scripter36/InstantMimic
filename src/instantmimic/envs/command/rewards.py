from __future__ import annotations

import torch

VELOCITY_REWARD_MODE_FORWARD_SPEED_HEADING = "forward_speed_heading"
VELOCITY_REWARD_MODE_VELOCITY_VECTOR_HEADING = "velocity_vector_heading"


def resolve_velocity_heading_reward_mode(raw_mode: str) -> str:
    normalized = str(raw_mode).strip().lower()
    valid_modes = (
        VELOCITY_REWARD_MODE_FORWARD_SPEED_HEADING,
        VELOCITY_REWARD_MODE_VELOCITY_VECTOR_HEADING,
    )
    if normalized not in valid_modes:
        valid = ", ".join(valid_modes)
        raise ValueError(
            f"Unsupported velocity_heading_reward_mode={raw_mode!r}. Valid values: {valid}."
        )
    return normalized


def compute_velocity_heading_reward(
    *,
    planar_velocity: torch.Tensor,
    heading_vec: torch.Tensor,
    target_dir: torch.Tensor,
    target_speed: torch.Tensor,
    speed_sigma: float,
    heading_weight: float,
    reward_mode: str = VELOCITY_REWARD_MODE_FORWARD_SPEED_HEADING,
) -> torch.Tensor:
    speed_reward, heading_reward = compute_velocity_heading_reward_terms(
        planar_velocity=planar_velocity,
        heading_vec=heading_vec,
        target_dir=target_dir,
        target_speed=target_speed,
        speed_sigma=speed_sigma,
        heading_weight=heading_weight,
        reward_mode=reward_mode,
    )
    return speed_reward + heading_reward


def compute_velocity_heading_reward_terms(
    *,
    planar_velocity: torch.Tensor,
    heading_vec: torch.Tensor,
    target_dir: torch.Tensor,
    target_speed: torch.Tensor,
    speed_sigma: float,
    heading_weight: float,
    reward_mode: str = VELOCITY_REWARD_MODE_FORWARD_SPEED_HEADING,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_mode = resolve_velocity_heading_reward_mode(reward_mode)
    target_speed = target_speed.reshape(-1, 1)
    sigma_sq = max(float(speed_sigma) * float(speed_sigma), 1.0e-6)
    if resolved_mode == VELOCITY_REWARD_MODE_VELOCITY_VECTOR_HEADING:
        target_velocity = target_speed * target_dir
        velocity_err = planar_velocity - target_velocity
        speed_reward = torch.exp(
            -torch.sum(velocity_err.square(), dim=-1, keepdim=True) / sigma_sq
        )
    else:
        forward_speed = torch.sum(planar_velocity * target_dir, dim=-1, keepdim=True)
        speed_err = forward_speed - target_speed
        speed_reward = torch.exp(-speed_err.square() / sigma_sq)
    heading_align = torch.clamp(
        torch.sum(heading_vec * target_dir, dim=-1, keepdim=True),
        min=0.0,
    )
    heading_reward = float(heading_weight) * heading_align
    return speed_reward, heading_reward


def compute_point_goal_reward(
    *,
    root_pos_xy: torch.Tensor,
    heading_vec: torch.Tensor,
    goal_pos_xy: torch.Tensor,
    prev_goal_distance: torch.Tensor,
    goal_radius: float,
    progress_weight: float,
    facing_weight: float,
    goal_bonus_value: float,
    stall_penalty_weight: float,
    min_progress_for_stall: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    progress_reward, facing_reward, goal_bonus, stall_penalty = (
        compute_point_goal_reward_terms(
            root_pos_xy=root_pos_xy,
            heading_vec=heading_vec,
            goal_pos_xy=goal_pos_xy,
            prev_goal_distance=prev_goal_distance,
            goal_radius=goal_radius,
            progress_weight=progress_weight,
            facing_weight=facing_weight,
            goal_bonus_value=goal_bonus_value,
            stall_penalty_weight=stall_penalty_weight,
            min_progress_for_stall=min_progress_for_stall,
        )
    )
    reward = progress_reward + facing_reward + goal_bonus + stall_penalty
    return reward, goal_bonus


def compute_point_goal_reward_terms(
    *,
    root_pos_xy: torch.Tensor,
    heading_vec: torch.Tensor,
    goal_pos_xy: torch.Tensor,
    prev_goal_distance: torch.Tensor,
    goal_radius: float,
    progress_weight: float,
    facing_weight: float,
    goal_bonus_value: float,
    stall_penalty_weight: float,
    min_progress_for_stall: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    goal_delta = goal_pos_xy - root_pos_xy
    goal_distance = torch.linalg.vector_norm(goal_delta, dim=-1, keepdim=True)
    goal_dir = goal_delta / torch.clamp(goal_distance, min=1.0e-6)
    progress = prev_goal_distance - goal_distance
    facing = torch.clamp(
        torch.sum(heading_vec * goal_dir, dim=-1, keepdim=True),
        min=0.0,
    )
    stall_penalty = torch.where(
        progress < float(min_progress_for_stall),
        torch.full_like(progress, float(stall_penalty_weight)),
        torch.zeros_like(progress),
    )
    goal_bonus = torch.where(
        goal_distance <= float(goal_radius),
        torch.full_like(progress, float(goal_bonus_value)),
        torch.zeros_like(progress),
    )
    progress_reward = float(progress_weight) * progress
    facing_reward = float(facing_weight) * facing
    stall_penalty_reward = -stall_penalty
    return progress_reward, facing_reward, goal_bonus, stall_penalty_reward


def compute_residual_regularization_terms(
    *,
    residual: torch.Tensor,
    prev_residual: torch.Tensor,
    l2_weight: float,
    rate_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_l2_penalty = -float(l2_weight) * residual.square().mean(
        dim=-1,
        keepdim=True,
    )
    residual_rate_penalty = -float(rate_weight) * (
        residual - prev_residual
    ).square().mean(
        dim=-1,
        keepdim=True,
    )
    return (
        residual_l2_penalty + residual_rate_penalty,
        residual_l2_penalty,
        residual_rate_penalty,
    )


def compute_command_fall_penalty(
    *,
    torso_height: torch.Tensor,
    termination_height: float,
    penalty_weight: float,
) -> torch.Tensor:
    fall_mask = torso_height < float(termination_height)
    return -float(penalty_weight) * fall_mask.to(dtype=torso_height.dtype)
