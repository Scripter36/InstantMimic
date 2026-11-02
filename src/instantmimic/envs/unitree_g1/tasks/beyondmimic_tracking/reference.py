"""Allocation-readable Stage-1 reference semantics.

This module intentionally expresses the public mjlab task as ordinary PyTorch.
The optimized task keeps this implementation as the differential oracle for its
Warp kernels and graph-captured runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from instantmimic.motion.quaternion import (
    quat_conj as _quat_conjugate,
)
from instantmimic.motion.quaternion import (
    quat_inverse,
)
from instantmimic.motion.quaternion import (
    quat_mul as quat_multiply,
)
from instantmimic.motion.quaternion import (
    quat_rotate as quat_apply,
)
from instantmimic.motion.quaternion import (
    quat_rotate_inverse as quat_apply_inverse,
)
from instantmimic.motion.quaternion import (
    quat_to_matrix as quaternion_to_matrix,  # noqa: F401
)

ACTION_DIM = 29
MOTION_COMMAND_DIM = 58
TRACKED_BODY_COUNT = 14
END_EFFECTOR_COUNT = 4
ACTOR_OBSERVATION_DIM = 160
CRITIC_OBSERVATION_DIM = 286

REWARD_WEIGHTS: Mapping[str, float] = {
    "anchor_position": 0.5,
    "anchor_orientation": 0.5,
    "body_position": 1.0,
    "body_orientation": 1.0,
    "body_linear_velocity": 1.0,
    "body_angular_velocity": 1.0,
    "action_rate": -0.1,
    "joint_limit": -10.0,
    "self_collision": -10.0,
}


def _validate_width(name: str, value: torch.Tensor, width: int) -> None:
    if value.ndim != 2 or value.shape[-1] != width:
        raise ValueError(
            f"{name} must have shape (batch, {width}); got {tuple(value.shape)}"
        )


def yaw_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    half_yaw = 0.5 * yaw
    zeros = torch.zeros_like(half_yaw)
    return torch.stack((torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)), dim=-1)


def quaternion_error_magnitude(
    first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    error = quat_multiply(first, _quat_conjugate(second))
    error = torch.where(error[..., :1] < 0.0, -error, error)
    vector_norm = torch.linalg.vector_norm(error[..., 1:], dim=-1)
    angle = 2.0 * torch.atan2(vector_norm, error[..., 0].clamp_min(1e-12))
    return angle


def euler_xyz_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = (0.5 * euler).unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def motion_relative_to_robot_anchor(
    *,
    robot_anchor_position: torch.Tensor,
    robot_anchor_quaternion: torch.Tensor,
    reference_anchor_position: torch.Tensor,
    reference_anchor_quaternion: torch.Tensor,
    reference_body_position: torch.Tensor,
    reference_body_quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align a reference frame's yaw and horizontal origin to the robot."""

    delta_position = robot_anchor_position.clone()
    delta_position[..., 2] = reference_anchor_position[..., 2]
    delta_orientation = yaw_quaternion(
        quat_multiply(
            robot_anchor_quaternion,
            quat_inverse(reference_anchor_quaternion),
        )
    )
    relative_offset = reference_body_position - reference_anchor_position[:, None, :]
    target_position = delta_position[:, None, :] + quat_apply(
        delta_orientation[:, None, :], relative_offset
    )
    target_quaternion = quat_multiply(
        delta_orientation[:, None, :], reference_body_quaternion
    )
    return target_position, target_quaternion


@dataclass(frozen=True)
class ObservationNoise:
    anchor_position: torch.Tensor
    anchor_orientation: torch.Tensor
    base_linear_velocity: torch.Tensor
    base_angular_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor


def assemble_actor_observation(
    *,
    command: torch.Tensor,
    anchor_position: torch.Tensor,
    anchor_orientation: torch.Tensor,
    base_linear_velocity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    joint_position: torch.Tensor,
    default_joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    default_joint_velocity: torch.Tensor,
    action: torch.Tensor,
    noise: ObservationNoise | None = None,
) -> torch.Tensor:
    terms = {
        "command": command,
        "anchor_position": anchor_position,
        "anchor_orientation": anchor_orientation,
        "base_linear_velocity": base_linear_velocity,
        "base_angular_velocity": base_angular_velocity,
        "joint_position": joint_position,
        "default_joint_position": default_joint_position,
        "joint_velocity": joint_velocity,
        "default_joint_velocity": default_joint_velocity,
        "action": action,
    }
    for name, width in (
        ("command", MOTION_COMMAND_DIM),
        ("anchor_position", 3),
        ("anchor_orientation", 6),
        ("base_linear_velocity", 3),
        ("base_angular_velocity", 3),
        ("joint_position", ACTION_DIM),
        ("default_joint_position", ACTION_DIM),
        ("joint_velocity", ACTION_DIM),
        ("default_joint_velocity", ACTION_DIM),
        ("action", ACTION_DIM),
    ):
        _validate_width(name, terms[name], width)

    anchor_position_term = anchor_position
    anchor_orientation_term = anchor_orientation
    base_linear_velocity_term = base_linear_velocity
    base_angular_velocity_term = base_angular_velocity
    joint_position_term = joint_position - default_joint_position
    joint_velocity_term = joint_velocity - default_joint_velocity
    if noise is not None:
        anchor_position_term = anchor_position_term + noise.anchor_position
        anchor_orientation_term = anchor_orientation_term + noise.anchor_orientation
        base_linear_velocity_term = (
            base_linear_velocity_term + noise.base_linear_velocity
        )
        base_angular_velocity_term = (
            base_angular_velocity_term + noise.base_angular_velocity
        )
        joint_position_term = joint_position_term + noise.joint_position
        joint_velocity_term = joint_velocity_term + noise.joint_velocity

    observation = torch.cat(
        (
            command,
            anchor_position_term,
            anchor_orientation_term,
            base_linear_velocity_term,
            base_angular_velocity_term,
            joint_position_term,
            joint_velocity_term,
            action,
        ),
        dim=-1,
    )
    _validate_width("actor observation", observation, ACTOR_OBSERVATION_DIM)
    return observation


def assemble_critic_observation(
    *,
    command: torch.Tensor,
    anchor_position: torch.Tensor,
    anchor_orientation: torch.Tensor,
    body_position: torch.Tensor,
    body_orientation: torch.Tensor,
    base_linear_velocity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    joint_position: torch.Tensor,
    default_joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    default_joint_velocity: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    for name, value, width in (
        ("command", command, MOTION_COMMAND_DIM),
        ("anchor_position", anchor_position, 3),
        ("anchor_orientation", anchor_orientation, 6),
        ("body_position", body_position, TRACKED_BODY_COUNT * 3),
        ("body_orientation", body_orientation, TRACKED_BODY_COUNT * 6),
        ("base_linear_velocity", base_linear_velocity, 3),
        ("base_angular_velocity", base_angular_velocity, 3),
        ("joint_position", joint_position, ACTION_DIM),
        ("default_joint_position", default_joint_position, ACTION_DIM),
        ("joint_velocity", joint_velocity, ACTION_DIM),
        ("default_joint_velocity", default_joint_velocity, ACTION_DIM),
        ("action", action, ACTION_DIM),
    ):
        _validate_width(name, value, width)
    observation = torch.cat(
        (
            command,
            anchor_position,
            anchor_orientation,
            body_position,
            body_orientation,
            base_linear_velocity,
            base_angular_velocity,
            joint_position - default_joint_position,
            joint_velocity - default_joint_velocity,
            action,
        ),
        dim=-1,
    )
    _validate_width("critic observation", observation, CRITIC_OBSERVATION_DIM)
    return observation


def default_relative_action_targets(
    action: torch.Tensor,
    action_scale: torch.Tensor,
    default_joint_position: torch.Tensor,
    encoder_bias: torch.Tensor,
) -> torch.Tensor:
    return default_joint_position + action * action_scale - encoder_bias


@dataclass(frozen=True)
class RewardResult:
    raw: Mapping[str, torch.Tensor]
    contributions: Mapping[str, torch.Tensor]
    total: torch.Tensor


def _tracking_exponential(
    error: torch.Tensor, standard_deviation: float, *, mean_dim: int | None = None
) -> torch.Tensor:
    squared = error.square().sum(dim=-1)
    if mean_dim is not None:
        squared = squared.mean(dim=mean_dim)
    return torch.exp(-squared / standard_deviation**2)


def compute_reward(
    *,
    robot_anchor_position: torch.Tensor,
    target_anchor_position: torch.Tensor,
    robot_anchor_quaternion: torch.Tensor,
    target_anchor_quaternion: torch.Tensor,
    robot_body_position: torch.Tensor,
    target_body_position: torch.Tensor,
    robot_body_quaternion: torch.Tensor,
    target_body_quaternion: torch.Tensor,
    robot_body_linear_velocity: torch.Tensor,
    target_body_linear_velocity: torch.Tensor,
    robot_body_angular_velocity: torch.Tensor,
    target_body_angular_velocity: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    joint_position: torch.Tensor,
    soft_joint_lower: torch.Tensor,
    soft_joint_upper: torch.Tensor,
    self_collision_force_history: torch.Tensor,
    step_duration: float,
) -> RewardResult:
    orientation_error = quaternion_error_magnitude(
        robot_anchor_quaternion, target_anchor_quaternion
    )
    body_orientation_error = quaternion_error_magnitude(
        robot_body_quaternion, target_body_quaternion
    )
    if self_collision_force_history.ndim == 4:
        collision = (
            torch.linalg.vector_norm(self_collision_force_history, dim=-1)
            .gt(10.0)
            .any(dim=1)
            .sum(dim=-1)
        )
    elif self_collision_force_history.ndim == 3:
        collision = (
            torch.linalg.vector_norm(self_collision_force_history, dim=-1)
            .gt(10.0)
            .sum(dim=-1)
        )
    else:
        raise ValueError(
            "self_collision_force_history must have shape (batch, history, 3) "
            "or (batch, contacts, history, 3)"
        )
    lower_violation = (soft_joint_lower - joint_position).clamp_min(0.0)
    upper_violation = (joint_position - soft_joint_upper).clamp_min(0.0)
    raw = {
        "anchor_position": _tracking_exponential(
            robot_anchor_position - target_anchor_position, 0.3
        ),
        "anchor_orientation": torch.exp(-orientation_error.square() / 0.4**2),
        "body_position": _tracking_exponential(
            robot_body_position - target_body_position, 0.3, mean_dim=-1
        ),
        "body_orientation": torch.exp(
            -body_orientation_error.square().mean(dim=-1) / 0.4**2
        ),
        "body_linear_velocity": _tracking_exponential(
            robot_body_linear_velocity - target_body_linear_velocity,
            1.0,
            mean_dim=-1,
        ),
        "body_angular_velocity": _tracking_exponential(
            robot_body_angular_velocity - target_body_angular_velocity,
            3.14,
            mean_dim=-1,
        ),
        "action_rate": (action - previous_action).square().sum(dim=-1),
        "joint_limit": (lower_violation + upper_violation).sum(dim=-1),
        "self_collision": collision.to(dtype=robot_anchor_position.dtype),
    }
    contributions = {
        name: value * REWARD_WEIGHTS[name] * step_duration
        for name, value in raw.items()
    }
    total = torch.stack(tuple(contributions.values()), dim=0).sum(dim=0)
    return RewardResult(raw=raw, contributions=contributions, total=total)


@dataclass(frozen=True)
class TerminationResult:
    timeout: torch.Tensor
    anchor_height: torch.Tensor
    anchor_orientation: torch.Tensor
    end_effector_height: torch.Tensor
    terminated: torch.Tensor


def compute_terminations(
    *,
    elapsed_steps: torch.Tensor,
    max_episode_steps: int,
    robot_anchor_height: torch.Tensor,
    target_anchor_height: torch.Tensor,
    robot_projected_gravity_z: torch.Tensor,
    target_projected_gravity_z: torch.Tensor | float,
    robot_end_effector_height: torch.Tensor,
    target_end_effector_height: torch.Tensor,
) -> TerminationResult:
    timeout = elapsed_steps >= max_episode_steps
    anchor_height = (robot_anchor_height - target_anchor_height).abs() > 0.25
    anchor_orientation = (
        robot_projected_gravity_z - target_projected_gravity_z
    ).abs() > 0.8
    end_effector_height = (
        (robot_end_effector_height - target_end_effector_height)
        .abs()
        .gt(0.25)
        .any(dim=-1)
    )
    terminated = anchor_height | anchor_orientation | end_effector_height
    return TerminationResult(
        timeout=timeout,
        anchor_height=anchor_height,
        anchor_orientation=anchor_orientation,
        end_effector_height=end_effector_height,
        terminated=terminated,
    )


@dataclass(frozen=True)
class AdaptiveSamplerState:
    bin_failed_count: torch.Tensor
    current_bin_failed: torch.Tensor
    motion_frame_count: int
    bin_count: int

    @classmethod
    def create(
        cls,
        *,
        motion_frame_count: int,
        sampling_fps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> AdaptiveSamplerState:
        bin_count = int(motion_frame_count // sampling_fps) + 1
        return cls(
            bin_failed_count=torch.zeros(bin_count, device=device, dtype=dtype),
            current_bin_failed=torch.zeros(bin_count, device=device, dtype=dtype),
            motion_frame_count=motion_frame_count,
            bin_count=bin_count,
        )


def failure_bins(
    frame_index: torch.Tensor, motion_frame_count: int, bin_count: int
) -> torch.Tensor:
    return torch.div(
        frame_index * bin_count, motion_frame_count, rounding_mode="floor"
    ).clamp_max(bin_count - 1)


def update_failure_ema(
    previous: torch.Tensor, current: torch.Tensor, *, alpha: float
) -> torch.Tensor:
    return alpha * current + (1.0 - alpha) * previous


def sampling_probabilities(
    bin_failed_count: torch.Tensor,
    *,
    uniform_ratio: float,
    kernel_size: int,
    adaptive_lambda: float = 0.8,
) -> torch.Tensor:
    probabilities = bin_failed_count + uniform_ratio / float(bin_failed_count.numel())
    kernel = probabilities.new_tensor(
        [adaptive_lambda**index for index in range(kernel_size)]
    )
    probabilities = functional.pad(
        probabilities[None, None, :],
        (0, kernel_size - 1),
        mode="replicate",
    )
    probabilities = functional.conv1d(probabilities, kernel.view(1, 1, -1)).flatten()
    return probabilities / probabilities.sum()


@dataclass(frozen=True)
class ResetState:
    root_position: torch.Tensor
    root_quaternion: torch.Tensor
    root_linear_velocity_world: torch.Tensor
    root_angular_velocity_body: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor


def perturb_reference_state(
    *,
    reference_root_position: torch.Tensor,
    reference_root_quaternion: torch.Tensor,
    reference_root_linear_velocity: torch.Tensor,
    reference_root_angular_velocity: torch.Tensor,
    reference_joint_position: torch.Tensor,
    reference_joint_velocity: torch.Tensor,
    position_delta: torch.Tensor,
    euler_delta: torch.Tensor,
    root_velocity_delta: torch.Tensor,
    joint_position_delta: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
) -> ResetState:
    root_position = reference_root_position + position_delta
    root_quaternion = quat_multiply(
        euler_xyz_to_quaternion(euler_delta), reference_root_quaternion
    )
    root_linear_velocity_world = (
        reference_root_linear_velocity + root_velocity_delta[..., :3]
    )
    root_angular_velocity_world = (
        reference_root_angular_velocity + root_velocity_delta[..., 3:]
    )
    root_angular_velocity_body = quat_apply_inverse(
        root_quaternion, root_angular_velocity_world
    )
    joint_position = (reference_joint_position + joint_position_delta).clamp(
        min=joint_lower, max=joint_upper
    )
    return ResetState(
        root_position=root_position,
        root_quaternion=root_quaternion,
        root_linear_velocity_world=root_linear_velocity_world,
        root_angular_velocity_body=root_angular_velocity_body,
        joint_position=joint_position,
        joint_velocity=reference_joint_velocity,
    )
