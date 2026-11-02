"""Pinned BeyondMimic motion tensor loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MotionTensors:
    fps: int
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    body_position_world: torch.Tensor
    body_quaternion_world: torch.Tensor
    body_linear_velocity_world: torch.Tensor
    body_angular_velocity_world: torch.Tensor

    @property
    def frame_count(self) -> int:
        return int(self.joint_position.shape[0])


def load_motion_tensors(
    path: str | Path,
    *,
    device: torch.device,
) -> MotionTensors:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"BeyondMimic motion does not exist: {resolved}. "
            "Prepare the pinned data with instantmimic.tools.prepare_beyondmimic; see docs/guides/beyondmimic.md."
        )
    with np.load(resolved, allow_pickle=False) as archive:
        expected = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        if set(archive.files) != expected:
            raise ValueError(
                f"Unexpected BeyondMimic motion arrays: {sorted(archive.files)!r}"
            )
        fps_array = np.asarray(archive["fps"])
        if fps_array.shape != (1,) or int(fps_array[0]) != 50:
            raise ValueError(f"Expected a single 50 Hz motion, got {fps_array!r}")
        arrays = {
            name: torch.as_tensor(np.asarray(archive[key]), device=device).contiguous()
            for name, key in (
                ("joint_position", "joint_pos"),
                ("joint_velocity", "joint_vel"),
                ("body_position_world", "body_pos_w"),
                ("body_quaternion_world", "body_quat_w"),
                ("body_linear_velocity_world", "body_lin_vel_w"),
                ("body_angular_velocity_world", "body_ang_vel_w"),
            )
        }
    frame_count = arrays["joint_position"].shape[0]
    expected_shapes = {
        "joint_position": (frame_count, 29),
        "joint_velocity": (frame_count, 29),
        "body_position_world": (frame_count, 30, 3),
        "body_quaternion_world": (frame_count, 30, 4),
        "body_linear_velocity_world": (frame_count, 30, 3),
        "body_angular_velocity_world": (frame_count, 30, 3),
    }
    for name, expected_shape in expected_shapes.items():
        value = arrays[name]
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    return MotionTensors(fps=50, **arrays)
