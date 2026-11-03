from __future__ import annotations

import math

import torch

from instantmimic.motion.quaternion import (
    quat_conj as quat_conjugate,
)
from instantmimic.motion.quaternion import (
    quat_mul,
    quat_rotate,
)
from instantmimic.motion.quaternion import (
    quat_rotate_inverse_unit as quat_rotate_inverse,
)


def normalize_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def get_euler_xyz(quaternion: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Match Isaac Sim's scalar-first extrinsic XYZ helper."""

    qw, qx, qy, qz = quaternion.unbind(dim=-1)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = qw * qw - qx * qx - qy * qy + qz * qz
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = torch.where(
        torch.abs(sinp) >= 1.0,
        torch.sign(sinp) * (math.pi / 2.0),
        torch.asin(sinp),
    )

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return (
        torch.remainder(roll, 2.0 * math.pi),
        torch.remainder(pitch, 2.0 * math.pi),
        torch.remainder(yaw, 2.0 * math.pi),
    )


def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    joint_positions: torch.Tensor,
    joint_lower_limits: torch.Tensor,
    joint_upper_limits: torch.Tensor,
    inverse_start_rotation: torch.Tensor,
    heading_basis: torch.Tensor,
    up_basis: torch.Tensor,
    potentials: torch.Tensor,
    previous_potentials: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, ...]:
    """Compute the Isaac Lab LocomotionEnv intermediate state on-device."""

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    target_directions = to_target / torch.linalg.vector_norm(
        to_target, dim=-1, keepdim=True
    )

    torso_quaternion = quat_mul(torso_rotation, inverse_start_rotation)
    up_vector = quat_rotate(torso_quaternion, up_basis)
    heading_vector = quat_rotate(torso_quaternion, heading_basis)
    up_projection = up_vector[:, 2]
    heading_projection = (heading_vector * target_directions).sum(dim=-1)

    velocity_local = quat_rotate_inverse(torso_quaternion, velocity)
    angular_velocity_local = quat_rotate_inverse(torso_quaternion, angular_velocity)
    roll, pitch, yaw = get_euler_xyz(torso_quaternion)

    # This is deliberately not derived from the flattened target vector. It is
    # the exact source helper behavior used for angle_to_target.
    walk_target_angle = torch.atan2(
        targets[:, 2] - torso_position[:, 2],
        targets[:, 0] - torso_position[:, 0],
    )
    angle_to_target = walk_target_angle - yaw

    joint_positions_scaled = (
        2.0
        * (
            (joint_positions - joint_lower_limits)
            / (joint_upper_limits - joint_lower_limits)
        )
        - 1.0
    )

    previous_potentials.copy_(potentials)
    potentials.copy_(-torch.linalg.vector_norm(to_target, dim=-1) / dt)

    return (
        up_projection,
        heading_projection,
        up_vector,
        heading_vector,
        velocity_local,
        angular_velocity_local,
        roll,
        pitch,
        yaw,
        angle_to_target,
        joint_positions_scaled,
        previous_potentials,
        potentials,
    )


def build_observation(
    *,
    torso_z: torch.Tensor,
    velocity_local: torch.Tensor,
    angular_velocity_local: torch.Tensor,
    yaw: torch.Tensor,
    roll: torch.Tensor,
    angle_to_target: torch.Tensor,
    up_projection: torch.Tensor,
    heading_projection: torch.Tensor,
    joint_positions_scaled: torch.Tensor,
    joint_velocities: torch.Tensor,
    actions: torch.Tensor,
    angular_velocity_scale: float = 1.0,
    joint_velocity_scale: float = 0.2,
) -> torch.Tensor:
    return torch.cat(
        (
            torso_z.unsqueeze(-1),
            velocity_local,
            angular_velocity_local * angular_velocity_scale,
            normalize_angle(yaw).unsqueeze(-1),
            normalize_angle(roll).unsqueeze(-1),
            normalize_angle(angle_to_target).unsqueeze(-1),
            up_projection.unsqueeze(-1),
            heading_projection.unsqueeze(-1),
            joint_positions_scaled,
            joint_velocities * joint_velocity_scale,
            actions,
        ),
        dim=-1,
    )


def compute_rewards(
    *,
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_projection: torch.Tensor,
    heading_projection: torch.Tensor,
    joint_velocities: torch.Tensor,
    joint_positions_scaled: torch.Tensor,
    potentials: torch.Tensor,
    previous_potentials: torch.Tensor,
    heading_weight: float = 0.5,
    up_weight: float = 0.1,
    actions_cost_scale: float = 0.005,
    energy_cost_scale: float = 0.05,
    joint_velocity_scale: float = 0.2,
    death_cost: float = -2.0,
    alive_reward_scale: float = 0.5,
    motor_effort_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    heading_reward = torch.where(
        heading_projection > 0.8,
        torch.full_like(heading_projection, heading_weight),
        heading_weight * heading_projection / 0.8,
    )
    up_reward = torch.where(
        up_projection > 0.93,
        torch.full_like(up_projection, up_weight),
        torch.zeros_like(up_projection),
    )
    actions_cost = torch.sum(actions.square(), dim=-1)
    electricity = torch.abs(actions * joint_velocities * joint_velocity_scale)
    if motor_effort_ratio is not None:
        electricity = electricity * motor_effort_ratio.unsqueeze(0)
    electricity_cost = electricity.sum(dim=-1)
    joint_limit_cost = torch.sum(joint_positions_scaled > 0.98, dim=-1)

    total_reward = (
        potentials
        - previous_potentials
        + alive_reward_scale
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - joint_limit_cost
    )
    return torch.where(
        reset_terminated,
        torch.full_like(total_reward, death_cost),
        total_reward,
    )


__all__ = [
    "build_observation",
    "compute_intermediate_values",
    "compute_rewards",
    "get_euler_xyz",
    "normalize_angle",
    "quat_conjugate",
    "quat_mul",
    "quat_rotate",
    "quat_rotate_inverse",
]
