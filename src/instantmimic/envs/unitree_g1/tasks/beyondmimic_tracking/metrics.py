"""Pinned mjlab Stage-1 benchmark metrics and episode aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from .reference import quaternion_error_magnitude

METRIC_NAMES: Final = (
    "mpkpe",
    "r_mpkpe",
    "joint_vel_error",
    "ee_pos_error",
    "ee_ori_error",
)


@dataclass(frozen=True)
class BenchmarkReference:
    """Reference tensors snapshotted before the environment step being scored."""

    body_position_world: torch.Tensor
    body_position_relative_world: torch.Tensor
    body_quaternion_relative_world: torch.Tensor
    joint_velocity: torch.Tensor


@dataclass(frozen=True)
class BenchmarkRobotState:
    """Post-step robot tensors paired with a pre-step reference snapshot."""

    body_position_world: torch.Tensor
    body_quaternion_world: torch.Tensor
    joint_velocity: torch.Tensor


def compute_mpkpe(
    reference_body_position_world: torch.Tensor,
    robot_body_position_world: torch.Tensor,
) -> torch.Tensor:
    """Mean global-frame Euclidean position error across tracked bodies."""
    return torch.linalg.vector_norm(
        reference_body_position_world - robot_body_position_world,
        dim=-1,
    ).mean(dim=-1)


def compute_root_relative_mpkpe(
    reference_body_position_relative_world: torch.Tensor,
    robot_body_position_world: torch.Tensor,
) -> torch.Tensor:
    """Mean drift-cancelled Euclidean position error across tracked bodies."""
    return torch.linalg.vector_norm(
        reference_body_position_relative_world - robot_body_position_world,
        dim=-1,
    ).mean(dim=-1)


def compute_joint_velocity_error(
    reference_joint_velocity: torch.Tensor,
    robot_joint_velocity: torch.Tensor,
) -> torch.Tensor:
    """Per-environment RMS joint-velocity error."""
    return torch.sqrt(
        torch.mean(
            (reference_joint_velocity - robot_joint_velocity).square(),
            dim=-1,
        )
    )


def compute_ee_position_error(
    reference_body_position_relative_world: torch.Tensor,
    robot_body_position_world: torch.Tensor,
    *,
    end_effector_indices: tuple[int, ...],
) -> torch.Tensor:
    """Mean drift-cancelled position error over the configured end effectors."""
    if not end_effector_indices:
        return torch.zeros(
            reference_body_position_relative_world.shape[0],
            device=reference_body_position_relative_world.device,
            dtype=reference_body_position_relative_world.dtype,
        )
    return torch.linalg.vector_norm(
        reference_body_position_relative_world[:, end_effector_indices]
        - robot_body_position_world[:, end_effector_indices],
        dim=-1,
    ).mean(dim=-1)


def compute_ee_orientation_error(
    reference_body_quaternion_relative_world: torch.Tensor,
    robot_body_quaternion_world: torch.Tensor,
    *,
    end_effector_indices: tuple[int, ...],
) -> torch.Tensor:
    """Mean shortest-path quaternion error in radians over end effectors."""
    if not end_effector_indices:
        return torch.zeros(
            reference_body_quaternion_relative_world.shape[0],
            device=reference_body_quaternion_relative_world.device,
            dtype=reference_body_quaternion_relative_world.dtype,
        )
    return quaternion_error_magnitude(
        reference_body_quaternion_relative_world[:, end_effector_indices],
        robot_body_quaternion_world[:, end_effector_indices],
    ).mean(dim=-1)


class BeyondMimicBenchmarkAccumulator:
    """Accumulate the pinned mjlab one-episode-per-environment protocol."""

    def __init__(
        self,
        *,
        num_envs: int,
        end_effector_indices: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.num_envs = int(num_envs)
        self.end_effector_indices = tuple(int(index) for index in end_effector_indices)
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.success = torch.zeros_like(self.done)
        self.active_steps = torch.zeros(self.num_envs, dtype=dtype, device=device)
        self.sums = {
            name: torch.zeros(self.num_envs, dtype=dtype, device=device)
            for name in METRIC_NAMES
        }

    def update(
        self,
        *,
        reference: BenchmarkReference,
        robot_body_position_world: torch.Tensor,
        robot_body_quaternion_world: torch.Tensor,
        robot_joint_velocity: torch.Tensor,
    ) -> None:
        """Score one step for environments that had not finished before it."""
        active = ~self.done
        active_float = active.to(dtype=self.active_steps.dtype)
        values = {
            "mpkpe": compute_mpkpe(
                reference.body_position_world,
                robot_body_position_world,
            ),
            "r_mpkpe": compute_root_relative_mpkpe(
                reference.body_position_relative_world,
                robot_body_position_world,
            ),
            "joint_vel_error": compute_joint_velocity_error(
                reference.joint_velocity,
                robot_joint_velocity,
            ),
            "ee_pos_error": compute_ee_position_error(
                reference.body_position_relative_world,
                robot_body_position_world,
                end_effector_indices=self.end_effector_indices,
            ),
            "ee_ori_error": compute_ee_orientation_error(
                reference.body_quaternion_relative_world,
                robot_body_quaternion_world,
                end_effector_indices=self.end_effector_indices,
            ),
        }
        self.active_steps.add_(active_float)
        for name, value in values.items():
            self.sums[name].add_(value * active_float)

    def mark_dones(
        self,
        *,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        """Record first completion and timeout-only success decisions."""
        if terminated.shape != self.done.shape or truncated.shape != self.done.shape:
            raise ValueError(
                "terminated/truncated shape must match accumulator state: "
                f"{tuple(terminated.shape)}, {tuple(truncated.shape)} != "
                f"{tuple(self.done.shape)}"
            )
        newly_done = torch.logical_or(terminated, truncated) & ~self.done
        self.success.logical_or_(newly_done & truncated & ~terminated)
        self.done.logical_or_(newly_done)
        return newly_done

    def finalize(self) -> dict[str, float]:
        """Return per-episode temporal means averaged over environments."""
        if not bool(torch.all(self.done).item()):
            unfinished = int(torch.count_nonzero(~self.done).item())
            raise RuntimeError(
                f"Cannot finalize with {unfinished} unfinished evaluation episodes"
            )
        if bool(torch.any(self.active_steps <= 0).item()):
            raise RuntimeError(
                "Every evaluation episode must contain at least one step"
            )
        metrics = {
            "success_rate": float(self.success.float().mean().item()),
        }
        metrics.update(
            {
                name: float((total / self.active_steps).mean().item())
                for name, total in self.sums.items()
            }
        )
        return metrics
