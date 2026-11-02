from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Self, TypeVar

import mujoco
import numpy as np
import torch

from instantmimic.motion import build_clip_offsets
from instantmimic.utils.project_paths import resolve_project_path

_MOCAP_ARRAY_KEYS = ("qpos", "qvel", "xipos", "ximat")
_MocapClipT = TypeVar("_MocapClipT", bound="MocapClip")
_MocapLibraryT = TypeVar("_MocapLibraryT", bound="MocapLibrary")


def _resolve_motion_data_dir(path: str | Path) -> Path:
    return resolve_project_path(path)


def rotmat_array_to_quat_array(ximat: np.ndarray) -> np.ndarray:
    """Convert [frames, bodies, 3, 3] MuJoCo rotation matrices to quaternions."""
    ximat = np.asarray(ximat)
    if ximat.ndim != 4 or tuple(ximat.shape[-2:]) != (3, 3):
        raise ValueError(
            "Expected rotation matrices with shape [frames, bodies, 3, 3], got "
            f"{tuple(ximat.shape)}."
        )

    frame_count, body_count = ximat.shape[:2]
    xquat = np.zeros((frame_count, body_count, 4), dtype=np.float32)
    quat = np.zeros(4, dtype=np.float64)
    for frame_idx in range(frame_count):
        for body_idx in range(body_count):
            mujoco.mju_mat2Quat(
                quat,
                np.asarray(ximat[frame_idx, body_idx], dtype=np.float64).reshape(9),
            )
            xquat[frame_idx, body_idx] = quat.astype(np.float32, copy=False)
    return xquat


def _load_mocap_arrays(
    *,
    motion_data_dir: str | Path,
    clip_name: str,
) -> dict[str, np.ndarray]:
    base_dir = _resolve_motion_data_dir(motion_data_dir)
    arrays: dict[str, np.ndarray] = {}
    for key in _MOCAP_ARRAY_KEYS:
        path = base_dir / key / f"{clip_name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing mocap clip file: {path}")
        arrays[key] = np.load(path)
    return arrays


def _validate_and_convert_mocap_arrays(
    *,
    arrays: dict[str, np.ndarray],
    clip_name: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    qpos = arrays["qpos"]
    qvel = arrays["qvel"]
    xipos = arrays["xipos"]
    ximat = arrays["ximat"]

    if qpos.ndim != 2 or qvel.ndim != 2:
        raise ValueError(
            "Expected qpos/qvel mocap arrays with shape [frames, dof], got "
            f"{qpos.shape} and {qvel.shape}."
        )
    if xipos.ndim != 3 or xipos.shape[-1] != 3:
        raise ValueError(
            "Expected xipos mocap array with shape [frames, bodies, 3], got "
            f"{xipos.shape}."
        )
    if ximat.ndim != 3 or ximat.shape[-1] != 9:
        raise ValueError(
            "Expected ximat mocap array with shape [frames, bodies, 9], got "
            f"{ximat.shape}."
        )

    frame_count = int(qpos.shape[0])
    if frame_count <= 0:
        raise ValueError(f"Mocap clip {clip_name!r} must contain at least one frame.")
    if any(int(arrays[key].shape[0]) != frame_count for key in _MOCAP_ARRAY_KEYS[1:]):
        raise ValueError(
            f"Mocap clip {clip_name!r} has inconsistent frame counts across "
            "qpos/qvel/xipos/ximat."
        )
    if int(xipos.shape[1]) != int(ximat.shape[1]):
        raise ValueError(
            f"Mocap clip {clip_name!r} has inconsistent body counts between "
            "xipos and ximat."
        )

    ximat_reshaped = ximat.reshape(frame_count, -1, 3, 3)
    arrays_with_quat = {
        **arrays,
        "xquat": rotmat_array_to_quat_array(ximat_reshaped),
        "ximat": ximat_reshaped,
    }
    return {
        key: torch.from_numpy(array).to(device=device, dtype=torch.float32)
        for key, array in arrays_with_quat.items()
    }


@dataclass(slots=True)
class MocapClip:
    clip_name: str
    qpos: torch.Tensor
    qvel: torch.Tensor
    xipos: torch.Tensor
    ximat: torch.Tensor
    xquat: torch.Tensor

    @property
    def frame_count(self) -> int:
        return int(self.qpos.shape[0])

    @classmethod
    def load(
        cls: type[_MocapClipT],
        *,
        motion_data_dir: str | Path,
        clip_name: str,
        device: torch.device,
    ) -> _MocapClipT:
        return load_mocap_clip(
            motion_data_dir=motion_data_dir,
            clip_name=clip_name,
            device=device,
            clip_type=cls,
        )

    def phase(self, frame_idx: torch.Tensor, *, loop: bool) -> torch.Tensor:
        if loop:
            frame_idx = torch.remainder(frame_idx, self.frame_count)
        if loop:
            denom = float(max(self.frame_count, 1))
        else:
            denom = float(max(self.frame_count - 1, 1))
        return frame_idx.to(dtype=torch.float32) / denom


@dataclass(slots=True)
class MocapLibrary:
    clip_names: tuple[str, ...]
    qpos: torch.Tensor
    qvel: torch.Tensor
    xipos: torch.Tensor
    ximat: torch.Tensor
    xquat: torch.Tensor
    clip_start: torch.Tensor
    clip_length: torch.Tensor

    clip_type: ClassVar[type[MocapClip]] = MocapClip

    @property
    def clip_count(self) -> int:
        return len(self.clip_names)

    @classmethod
    def from_clips(
        cls: type[Self],
        *,
        clips: list[MocapClip],
        device: torch.device,
    ) -> Self:
        if len(clips) <= 0:
            raise ValueError("clips must contain at least one clip.")

        first = clips[0]
        for clip in clips[1:]:
            for field_name in ("qpos", "qvel", "xipos", "ximat", "xquat"):
                first_shape = tuple(getattr(first, field_name).shape[1:])
                clip_shape = tuple(getattr(clip, field_name).shape[1:])
                if clip_shape != first_shape:
                    raise ValueError(
                        f"All mocap clips must share the same {field_name} shape."
                    )

        clip_start, clip_length = build_clip_offsets(
            clip_frame_counts=[clip.frame_count for clip in clips],
            device=device,
        )
        return cls(
            clip_names=tuple(str(clip.clip_name) for clip in clips),
            qpos=torch.cat([clip.qpos for clip in clips], dim=0).contiguous(),
            qvel=torch.cat([clip.qvel for clip in clips], dim=0).contiguous(),
            xipos=torch.cat([clip.xipos for clip in clips], dim=0).contiguous(),
            ximat=torch.cat([clip.ximat for clip in clips], dim=0).contiguous(),
            xquat=torch.cat([clip.xquat for clip in clips], dim=0).contiguous(),
            clip_start=clip_start,
            clip_length=clip_length,
        )

    @classmethod
    def load_many(
        cls: type[Self],
        *,
        motion_data_dir: str | Path,
        clip_names: list[str] | tuple[str, ...],
        device: torch.device,
    ) -> Self:
        if len(clip_names) <= 0:
            raise ValueError("clip_names must contain at least one clip.")

        clips = [
            cls.clip_type.load(
                motion_data_dir=motion_data_dir,
                clip_name=str(clip_name),
                device=device,
            )
            for clip_name in clip_names
        ]
        return cls.from_clips(clips=clips, device=device)

    def iter_clips(self) -> Iterator[MocapClip]:
        for clip_idx, clip_name in enumerate(self.clip_names):
            clip_start = int(self.clip_start[clip_idx].item())
            clip_length = int(self.clip_length[clip_idx].item())
            yield self.clip_type(
                clip_name=str(clip_name),
                qpos=self.qpos[clip_start : clip_start + clip_length].contiguous(),
                qvel=self.qvel[clip_start : clip_start + clip_length].contiguous(),
                xipos=self.xipos[clip_start : clip_start + clip_length].contiguous(),
                ximat=self.ximat[clip_start : clip_start + clip_length].contiguous(),
                xquat=self.xquat[clip_start : clip_start + clip_length].contiguous(),
            )

    def map_clips(
        self,
        transform: Callable[[MocapClip], MocapClip],
        *,
        device: torch.device,
    ) -> Self:
        return type(self).from_clips(
            clips=[transform(clip) for clip in self.iter_clips()],
            device=device,
        )


def load_mocap_clip(
    *,
    motion_data_dir: str | Path,
    clip_name: str,
    device: torch.device,
    clip_type: type[_MocapClipT] = MocapClip,
) -> _MocapClipT:
    arrays = _load_mocap_arrays(
        motion_data_dir=motion_data_dir,
        clip_name=clip_name,
    )
    tensors = _validate_and_convert_mocap_arrays(
        arrays=arrays,
        clip_name=clip_name,
        device=device,
    )
    frame_count = int(tensors["qpos"].shape[0])
    return clip_type(
        clip_name=clip_name,
        qpos=tensors["qpos"].contiguous(),
        qvel=tensors["qvel"].contiguous(),
        xipos=tensors["xipos"].contiguous(),
        ximat=tensors["ximat"].reshape(frame_count, -1, 3, 3).contiguous(),
        xquat=tensors["xquat"].reshape(frame_count, -1, 4).contiguous(),
    )


_MIRROR_Y_MAT = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def _rot_x_np(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _rot_y_np(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def _rot_z_np(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _root_rotmat_from_qpos_row_np(
    qpos_row: np.ndarray,
    *,
    root_qpos_rotation_indices: tuple[int, int, int],
) -> np.ndarray:
    rx, ry, rz = (float(qpos_row[index]) for index in root_qpos_rotation_indices)
    return _rot_x_np(rx) @ _rot_y_np(ry) @ _rot_z_np(rz)


def _root_euler_xyz_from_rotmat_np(rotmat: np.ndarray) -> np.ndarray:
    sy = float(np.clip(rotmat[0, 2], -1.0, 1.0))
    ry = float(np.arcsin(sy))
    cy = float(np.cos(ry))
    if abs(cy) > 1.0e-8:
        rx = float(np.arctan2(-rotmat[1, 2], rotmat[2, 2]))
        rz = float(np.arctan2(-rotmat[0, 1], rotmat[0, 0]))
    else:
        rx = float(np.arctan2(rotmat[2, 1], rotmat[1, 1]))
        rz = 0.0
    return np.array([rx, ry, rz], dtype=np.float64)


def _root_body_world_pose_from_qpos_row_np(
    qpos_row: np.ndarray,
    *,
    root_body_pos: np.ndarray,
    root_qpos_position_indices: tuple[int, int, int],
    root_qpos_rotation_indices: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    root_pos = np.asarray(root_body_pos, dtype=np.float64).copy()
    root_pos += np.asarray(
        [qpos_row[index] for index in root_qpos_position_indices],
        dtype=np.float64,
    )
    root_rot = _root_rotmat_from_qpos_row_np(
        qpos_row,
        root_qpos_rotation_indices=root_qpos_rotation_indices,
    )
    return root_pos, root_rot


def compute_loop_root_transform_np(
    qpos_np: np.ndarray,
    *,
    root_body_pos: np.ndarray,
    vertical_axis: int = 2,
    root_qpos_position_indices: tuple[int, int, int] = (0, 1, 2),
    root_qpos_rotation_indices: tuple[int, int, int] = (3, 4, 5),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a planar root transform for extrapolated cyclic mocap."""
    if int(qpos_np.shape[0]) <= 0:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    frame_count = int(qpos_np.shape[0])
    last_i = frame_count - 1
    prev_i = max(0, frame_count - 2)
    pose_kwargs = {
        "root_body_pos": root_body_pos,
        "root_qpos_position_indices": root_qpos_position_indices,
        "root_qpos_rotation_indices": root_qpos_rotation_indices,
    }
    p_first, r_first = _root_body_world_pose_from_qpos_row_np(qpos_np[0], **pose_kwargs)
    p_last, r_last = _root_body_world_pose_from_qpos_row_np(
        qpos_np[last_i], **pose_kwargs
    )
    p_prev, r_prev = _root_body_world_pose_from_qpos_row_np(
        qpos_np[prev_i], **pose_kwargs
    )

    r_delta = r_last @ r_prev.T
    r_ext = r_delta @ r_last
    r_rel = r_ext @ r_first.T
    loop_yaw = float(np.arctan2(r_rel[1, 0], r_rel[0, 0]))
    loop_root_rot = _rot_z_np(loop_yaw)

    p_ext = p_last + (p_last - p_prev)
    loop_root_pos = p_ext - loop_root_rot @ p_first
    loop_root_pos = loop_root_pos.astype(np.float64, copy=False)
    loop_root_pos[int(vertical_axis)] = 0.0
    return loop_root_rot, loop_root_pos


def build_loop_next_qpos_np(
    qpos_np: np.ndarray,
    *,
    root_body_pos: np.ndarray,
    loop_root_rot: np.ndarray,
    loop_root_pos: np.ndarray,
    root_qpos_position_indices: tuple[int, int, int] = (0, 1, 2),
    root_qpos_rotation_indices: tuple[int, int, int] = (3, 4, 5),
) -> np.ndarray:
    """Build the first qpos row after applying one loop root transform."""
    next_qpos = np.asarray(qpos_np[0], dtype=np.float64).copy()
    pose_kwargs = {
        "root_body_pos": root_body_pos,
        "root_qpos_position_indices": root_qpos_position_indices,
        "root_qpos_rotation_indices": root_qpos_rotation_indices,
    }
    root_pos_0, root_rot_0 = _root_body_world_pose_from_qpos_row_np(
        qpos_np[0], **pose_kwargs
    )
    next_root_pos = loop_root_rot @ root_pos_0 + loop_root_pos
    next_root_rot = loop_root_rot @ root_rot_0
    for index, value in zip(
        root_qpos_position_indices, next_root_pos - root_body_pos, strict=False
    ):
        next_qpos[index] = value
    next_euler = _root_euler_xyz_from_rotmat_np(next_root_rot)
    for index, value in zip(root_qpos_rotation_indices, next_euler, strict=False):
        next_qpos[index] = value
    return next_qpos


def _normalize_vec_np(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-8:
        return np.asarray(vec, dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / norm


def _paired_joint_name(
    name: str,
    *,
    left_prefix: str,
    right_prefix: str,
) -> str:
    if right_prefix and name.startswith(right_prefix):
        return f"{left_prefix}{name[len(right_prefix) :]}"
    if left_prefix and name.startswith(left_prefix):
        return f"{right_prefix}{name[len(left_prefix) :]}"
    return name


def _infer_mirrored_dof_sign(
    *,
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    rotational: bool,
) -> float:
    reflected_axis = _normalize_vec_np(_MIRROR_Y_MAT @ np.asarray(source_axis))
    resolved_target_axis = _normalize_vec_np(np.asarray(target_axis, dtype=np.float64))
    pos_score = float(np.dot(reflected_axis, resolved_target_axis))
    neg_score = float(np.dot(-reflected_axis, resolved_target_axis))
    if rotational:
        return -1.0 if pos_score >= neg_score else 1.0
    return 1.0 if pos_score >= neg_score else -1.0


def _reference_body_slice(
    model: mujoco.MjModel,
    *,
    reference_body_start: int,
    reference_body_count_offset: int,
) -> slice:
    body_start = int(reference_body_start)
    body_count = int(model.nbody) - int(reference_body_count_offset)
    if body_start < 0 or body_count <= 0:
        raise ValueError(
            "Mocap reference body layout must have a non-negative start and "
            "at least one body."
        )
    body_stop = body_start + body_count
    if body_stop > int(model.nbody):
        raise ValueError(
            "Mocap reference body layout exceeds the MuJoCo model body count."
        )
    return slice(body_start, body_stop)


def recompute_mocap_body_state_arrays(
    *,
    model: mujoco.MjModel,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    device: torch.device,
    reference_body_start: int = 1,
    reference_body_count_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute reference body positions, rotations, and quaternions."""
    body_slice = _reference_body_slice(
        model,
        reference_body_start=reference_body_start,
        reference_body_count_offset=reference_body_count_offset,
    )
    body_count = body_slice.stop - body_slice.start
    frame_count = int(qpos.shape[0])
    xipos = np.zeros((frame_count, body_count, 3), dtype=np.float32)
    ximat = np.zeros((frame_count, body_count, 3, 3), dtype=np.float32)
    xquat = np.zeros((frame_count, body_count, 4), dtype=np.float32)
    helper_data = mujoco.MjData(model)

    for frame_idx in range(frame_count):
        helper_data.qpos[:] = (
            qpos[frame_idx].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        helper_data.qvel[:] = (
            qvel[frame_idx].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        mujoco.mj_forward(model, helper_data)
        xipos[frame_idx] = helper_data.xpos[body_slice].astype(np.float32, copy=False)
        ximat[frame_idx] = (
            helper_data.xmat[body_slice]
            .reshape(-1, 3, 3)
            .astype(np.float32, copy=False)
        )
        xquat[frame_idx] = helper_data.xquat[body_slice].astype(np.float32, copy=False)

    return (
        torch.from_numpy(xipos).to(device=device, dtype=torch.float32).contiguous(),
        torch.from_numpy(ximat).to(device=device, dtype=torch.float32).contiguous(),
        torch.from_numpy(xquat).to(device=device, dtype=torch.float32).contiguous(),
    )


def mirror_mocap_clip(
    clip: _MocapClipT,
    *,
    model: mujoco.MjModel,
    device: torch.device,
    left_prefix: str = "l_",
    right_prefix: str = "r_",
    clip_suffix: str = "__mirror",
    reference_body_start: int = 1,
    reference_body_count_offset: int = 0,
) -> _MocapClipT:
    """Reflect a paired-joint mocap clip across the model's sagittal plane."""
    joint_name_to_id = {
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)): int(
            joint_id
        )
        for joint_id in range(model.njnt)
    }
    qpos_np = clip.qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    qvel_np = clip.qvel.detach().cpu().numpy().astype(np.float64, copy=False)
    mirrored_qpos = np.zeros_like(qpos_np, dtype=np.float64)
    mirrored_qvel = np.zeros_like(qvel_np, dtype=np.float64)

    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            raise ValueError(f"Joint id {joint_id} is unnamed in the model.")
        target_name = _paired_joint_name(
            str(joint_name),
            left_prefix=left_prefix,
            right_prefix=right_prefix,
        )
        target_joint_id = joint_name_to_id.get(target_name)
        if target_joint_id is None:
            raise ValueError(f"Missing mirrored joint for {joint_name!r}.")

        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (
            int(mujoco.mjtJoint.mjJNT_SLIDE),
            int(mujoco.mjtJoint.mjJNT_HINGE),
        ):
            raise ValueError(
                "Paired-joint reflection only supports slide/hinge joints, got "
                f"type={joint_type} for {joint_name!r}."
            )

        sign = _infer_mirrored_dof_sign(
            source_axis=model.jnt_axis[joint_id],
            target_axis=model.jnt_axis[target_joint_id],
            rotational=joint_type == int(mujoco.mjtJoint.mjJNT_HINGE),
        )
        src_qpos_adr = int(model.jnt_qposadr[joint_id])
        dst_qpos_adr = int(model.jnt_qposadr[target_joint_id])
        mirrored_qpos[:, dst_qpos_adr] = sign * qpos_np[:, src_qpos_adr]
        src_dof_adr = int(model.jnt_dofadr[joint_id])
        dst_dof_adr = int(model.jnt_dofadr[target_joint_id])
        mirrored_qvel[:, dst_dof_adr] = sign * qvel_np[:, src_dof_adr]

    mirrored_qpos_t = torch.from_numpy(mirrored_qpos).to(
        device=device, dtype=clip.qpos.dtype
    )
    mirrored_qvel_t = torch.from_numpy(mirrored_qvel).to(
        device=device, dtype=clip.qvel.dtype
    )
    xipos, ximat, xquat = recompute_mocap_body_state_arrays(
        model=model,
        qpos=mirrored_qpos_t,
        qvel=mirrored_qvel_t,
        device=device,
        reference_body_start=reference_body_start,
        reference_body_count_offset=reference_body_count_offset,
    )
    mirrored_name = (
        clip.clip_name[: -len(clip_suffix)]
        if clip_suffix and clip.clip_name.endswith(clip_suffix)
        else f"{clip.clip_name}{clip_suffix}"
    )
    return type(clip)(
        clip_name=mirrored_name,
        qpos=mirrored_qpos_t.contiguous(),
        qvel=mirrored_qvel_t.contiguous(),
        xipos=xipos,
        ximat=ximat,
        xquat=xquat,
    )


def _smooth_cycle_qpos(
    qpos: torch.Tensor,
    *,
    model: mujoco.MjModel,
    blend_frames: int,
    root_body_pos: np.ndarray,
    root_qpos_position_indices: tuple[int, int, int],
    root_qpos_rotation_indices: tuple[int, int, int],
    vertical_axis: int,
) -> torch.Tensor:
    frame_count = int(qpos.shape[0])
    if frame_count < 2:
        return qpos
    blend_count = max(0, min(int(blend_frames), frame_count))
    if blend_count == 0:
        return qpos

    qpos_np = qpos.detach().cpu().numpy().astype(np.float64, copy=True)
    loop_root_rot, loop_root_pos = compute_loop_root_transform_np(
        qpos_np,
        root_body_pos=root_body_pos,
        vertical_axis=vertical_axis,
        root_qpos_position_indices=root_qpos_position_indices,
        root_qpos_rotation_indices=root_qpos_rotation_indices,
    )
    next_qpos = build_loop_next_qpos_np(
        qpos_np,
        root_body_pos=root_body_pos,
        loop_root_rot=loop_root_rot,
        loop_root_pos=loop_root_pos,
        root_qpos_position_indices=root_qpos_position_indices,
        root_qpos_rotation_indices=root_qpos_rotation_indices,
    )
    seam_delta = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, seam_delta, 1.0, qpos_np[-1], next_qpos)

    for idx in range(blend_count):
        frame_idx = frame_count - blend_count + idx
        t = float(idx) / float(max(blend_count - 1, 1))
        s = 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))
        if s <= 0.0:
            continue
        q_frame = qpos_np[frame_idx].copy()
        mujoco.mj_integratePos(model, q_frame, seam_delta * s, 1.0)
        qpos_np[frame_idx] = q_frame
    return torch.as_tensor(qpos_np, device=qpos.device, dtype=qpos.dtype)


def _recompute_qvel_from_qpos(
    qpos: torch.Tensor,
    *,
    model: mujoco.MjModel,
    control_dt: float,
    root_body_pos: np.ndarray,
    root_qpos_position_indices: tuple[int, int, int],
    root_qpos_rotation_indices: tuple[int, int, int],
    vertical_axis: int,
    loop_boundary_vel_from_first_frame: bool,
    loop_boundary_vel_extrapolate: bool,
) -> torch.Tensor:
    frame_count = int(qpos.shape[0])
    qvel_np = np.zeros((frame_count, model.nv), dtype=np.float64)
    if frame_count < 2:
        return torch.as_tensor(qvel_np, device=qpos.device, dtype=qpos.dtype)

    qpos_np = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    tmp = np.zeros(model.nv, dtype=np.float64)
    for frame_idx in range(frame_count - 1):
        mujoco.mj_differentiatePos(
            model,
            tmp,
            float(control_dt),
            qpos_np[frame_idx],
            qpos_np[frame_idx + 1],
        )
        qvel_np[frame_idx] = tmp

    if loop_boundary_vel_from_first_frame:
        loop_root_rot, loop_root_pos = compute_loop_root_transform_np(
            qpos_np,
            root_body_pos=root_body_pos,
            vertical_axis=vertical_axis,
            root_qpos_position_indices=root_qpos_position_indices,
            root_qpos_rotation_indices=root_qpos_rotation_indices,
        )
        next_qpos = build_loop_next_qpos_np(
            qpos_np,
            root_body_pos=root_body_pos,
            loop_root_rot=loop_root_rot,
            loop_root_pos=loop_root_pos,
            root_qpos_position_indices=root_qpos_position_indices,
            root_qpos_rotation_indices=root_qpos_rotation_indices,
        )
        mujoco.mj_differentiatePos(
            model, tmp, float(control_dt), qpos_np[-1], next_qpos
        )
        qvel_np[-1] = tmp
    elif loop_boundary_vel_extrapolate:
        qvel_np[-1] = qvel_np[-2]
    return torch.as_tensor(qvel_np, device=qpos.device, dtype=qpos.dtype)


def smooth_mocap_clip(
    clip: _MocapClipT,
    *,
    model: mujoco.MjModel,
    root_body_name: str,
    control_dt: float,
    smooth_cycle_blend_frames: int,
    device: torch.device,
    reference_body_start: int = 1,
    reference_body_count_offset: int = 0,
    root_qpos_position_indices: tuple[int, int, int] = (0, 1, 2),
    root_qpos_rotation_indices: tuple[int, int, int] = (3, 4, 5),
    vertical_axis: int = 2,
) -> _MocapClipT:
    root_body_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    )
    if root_body_id < 0:
        raise ValueError(f"Missing mocap root body {root_body_name!r}.")
    root_body_pos = np.asarray(model.body_pos[root_body_id], dtype=np.float64)
    qpos = _smooth_cycle_qpos(
        clip.qpos,
        model=model,
        blend_frames=int(smooth_cycle_blend_frames),
        root_body_pos=root_body_pos,
        root_qpos_position_indices=root_qpos_position_indices,
        root_qpos_rotation_indices=root_qpos_rotation_indices,
        vertical_axis=vertical_axis,
    )
    if int(qpos.shape[0]) > 2:
        qpos = qpos[:-1].contiguous()
    qvel = _recompute_qvel_from_qpos(
        qpos,
        model=model,
        control_dt=float(control_dt),
        root_body_pos=root_body_pos,
        root_qpos_position_indices=root_qpos_position_indices,
        root_qpos_rotation_indices=root_qpos_rotation_indices,
        vertical_axis=vertical_axis,
        loop_boundary_vel_from_first_frame=True,
        loop_boundary_vel_extrapolate=False,
    )
    xipos, ximat, xquat = recompute_mocap_body_state_arrays(
        model=model,
        qpos=qpos,
        qvel=qvel,
        device=device,
        reference_body_start=reference_body_start,
        reference_body_count_offset=reference_body_count_offset,
    )
    return type(clip)(
        clip_name=clip.clip_name,
        qpos=qpos.contiguous(),
        qvel=qvel.contiguous(),
        xipos=xipos,
        ximat=ximat,
        xquat=xquat,
    )


def smooth_mocap_library(
    library: _MocapLibraryT,
    *,
    model: mujoco.MjModel,
    root_body_name: str,
    control_dt: float,
    smooth_cycle_blend_frames: int,
    device: torch.device,
    reference_body_start: int = 1,
    reference_body_count_offset: int = 0,
    root_qpos_position_indices: tuple[int, int, int] = (0, 1, 2),
    root_qpos_rotation_indices: tuple[int, int, int] = (3, 4, 5),
    vertical_axis: int = 2,
) -> _MocapLibraryT:
    return library.map_clips(
        lambda clip: smooth_mocap_clip(
            clip,
            model=model,
            root_body_name=root_body_name,
            control_dt=float(control_dt),
            smooth_cycle_blend_frames=int(smooth_cycle_blend_frames),
            device=device,
            reference_body_start=reference_body_start,
            reference_body_count_offset=reference_body_count_offset,
            root_qpos_position_indices=root_qpos_position_indices,
            root_qpos_rotation_indices=root_qpos_rotation_indices,
            vertical_axis=vertical_axis,
        ),
        device=device,
    )


__all__ = [
    "MocapClip",
    "MocapLibrary",
    "build_loop_next_qpos_np",
    "compute_loop_root_transform_np",
    "load_mocap_clip",
    "mirror_mocap_clip",
    "recompute_mocap_body_state_arrays",
    "rotmat_array_to_quat_array",
    "smooth_mocap_clip",
    "smooth_mocap_library",
]
