from __future__ import annotations

import mujoco
import numpy as np
import torch

from .mocap_clip import (
    MocapClip,
    MocapLibrary,
    _recompute_qvel_from_qpos,
    build_loop_next_qpos_np,
    compute_loop_root_transform_np,
    mirror_mocap_clip,
    recompute_mocap_body_state_arrays,
    smooth_mocap_clip,
    smooth_mocap_library,
)
from .mocap_clip import (
    _root_body_world_pose_from_qpos_row_np as _common_root_body_world_pose_from_qpos_row_np,
)
from .mocap_clip import (
    _root_euler_xyz_from_rotmat_np as _common_root_euler_xyz_from_rotmat_np,
)
from .mocap_clip import (
    _rot_z_np as _common_rot_z_np,
)


def _recompute_ostrich_body_state_arrays(
    *,
    model: mujoco.MjModel,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return recompute_mocap_body_state_arrays(
        model=model,
        qpos=qpos,
        qvel=qvel,
        device=device,
        reference_body_start=1,
        reference_body_count_offset=2,
    )


def _recompute_ostrich_qvel_from_qpos(
    qpos: torch.Tensor,
    *,
    model: mujoco.MjModel,
    control_dt: float,
    root_body_pos: np.ndarray,
    loop_boundary_vel_from_first_frame: bool,
    loop_boundary_vel_extrapolate: bool,
) -> torch.Tensor:
    return _recompute_qvel_from_qpos(
        qpos,
        model=model,
        control_dt=control_dt,
        root_body_pos=root_body_pos,
        root_qpos_position_indices=(0, 1, 2),
        root_qpos_rotation_indices=(3, 4, 5),
        vertical_axis=2,
        loop_boundary_vel_from_first_frame=loop_boundary_vel_from_first_frame,
        loop_boundary_vel_extrapolate=loop_boundary_vel_extrapolate,
    )


def _root_body_world_pose_from_qpos_row_np(
    qpos_row: np.ndarray,
    *,
    root_body_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _common_root_body_world_pose_from_qpos_row_np(
        qpos_row,
        root_body_pos=root_body_pos,
        root_qpos_position_indices=(0, 1, 2),
        root_qpos_rotation_indices=(3, 4, 5),
    )


_root_euler_xyz_from_rotmat_np = _common_root_euler_xyz_from_rotmat_np
_rot_z_np = _common_rot_z_np


def mirror_ostrich_mocap_clip(
    clip: MocapClip,
    *,
    model: mujoco.MjModel,
    device: torch.device,
) -> MocapClip:
    return mirror_mocap_clip(
        clip,
        model=model,
        device=device,
        reference_body_start=1,
        reference_body_count_offset=2,
    )


def compute_ostrich_loop_root_transform_np(
    qpos_np: np.ndarray,
    *,
    root_body_pos: np.ndarray,
    vertical_axis: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    return compute_loop_root_transform_np(
        qpos_np,
        root_body_pos=root_body_pos,
        vertical_axis=vertical_axis,
    )


def _build_ostrich_loop_next_qpos_np(
    qpos_np: np.ndarray,
    *,
    root_body_pos: np.ndarray,
    loop_root_rot: np.ndarray,
    loop_root_pos: np.ndarray,
) -> np.ndarray:
    return build_loop_next_qpos_np(
        qpos_np,
        root_body_pos=root_body_pos,
        loop_root_rot=loop_root_rot,
        loop_root_pos=loop_root_pos,
    )


def smooth_ostrich_mocap_clip(
    clip: MocapClip,
    *,
    model: mujoco.MjModel,
    control_dt: float,
    smooth_cycle_blend_frames: int,
    device: torch.device,
) -> MocapClip:
    return smooth_mocap_clip(
        clip,
        model=model,
        root_body_name="ostrich",
        control_dt=control_dt,
        smooth_cycle_blend_frames=smooth_cycle_blend_frames,
        device=device,
        reference_body_start=1,
        reference_body_count_offset=2,
    )


def smooth_ostrich_mocap_library(
    library: MocapLibrary,
    *,
    model: mujoco.MjModel,
    control_dt: float,
    smooth_cycle_blend_frames: int,
    device: torch.device,
) -> MocapLibrary:
    return smooth_mocap_library(
        library,
        model=model,
        root_body_name="ostrich",
        control_dt=control_dt,
        smooth_cycle_blend_frames=smooth_cycle_blend_frames,
        device=device,
        reference_body_start=1,
        reference_body_count_offset=2,
    )


__all__ = [
    "_build_ostrich_loop_next_qpos_np",
    "compute_ostrich_loop_root_transform_np",
    "mirror_ostrich_mocap_clip",
    "smooth_ostrich_mocap_clip",
    "smooth_ostrich_mocap_library",
]
