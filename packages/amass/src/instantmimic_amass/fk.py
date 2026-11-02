from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .motion import quat_wxyz_to_rotvec as _mimickit_quat_wxyz_to_rotvec
from .smpl_kinematics import (
    LOCAL_TRANSLATION,
    PARENT_INDICES,
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
    compute_global_rotations_xyzw,
    compute_global_translations,
    exp_map_to_quat_xyzw,
    quat_conjugate_xyzw,
    quat_mul_xyzw,
)

FK_JOINT_SOURCE_BODIES = {
    "abdomen": ("Pelvis", "Chest"),
    "neck": ("Chest", "Head"),
    "right_shoulder": ("Chest", "R_Shoulder"),
    "right_elbow": ("R_Shoulder", "R_Elbow"),
    "left_shoulder": ("Chest", "L_Shoulder"),
    "left_elbow": ("L_Shoulder", "L_Elbow"),
    "right_hip": ("Pelvis", "R_Hip"),
    "right_knee": ("R_Hip", "R_Knee"),
    "right_ankle": ("R_Knee", "R_Ankle"),
    "left_hip": ("Pelvis", "L_Hip"),
    "left_knee": ("L_Hip", "L_Knee"),
    "left_ankle": ("L_Knee", "L_Ankle"),
}


def _quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for the shared quaternion conversion helper."""

    return _mimickit_quat_wxyz_to_rotvec(quat_wxyz)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.clip(norm, 1.0e-12, None)


def _relative_quat_wxyz(
    parent_global: np.ndarray, child_global: np.ndarray
) -> np.ndarray:
    return _quat_mul_wxyz(_quat_conj_wxyz(parent_global), child_global)


def _hinge_angle_from_quat_wxyz(local_quat: np.ndarray, axis: np.ndarray) -> np.ndarray:
    q = _normalize_quat_wxyz(local_quat)
    axis = np.asarray(axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-8:
        return np.zeros((q.shape[0],), dtype=np.float64)
    axis = axis / axis_norm

    proj_mag = np.sum(q[:, 1:4] * axis[None, :], axis=-1)
    twist = np.zeros_like(q)
    twist[:, 0] = q[:, 0]
    twist[:, 1:4] = proj_mag[:, None] * axis[None, :]
    twist = _normalize_quat_wxyz(twist)

    twist_w = twist[:, 0]
    twist_s = np.sum(twist[:, 1:4] * axis[None, :], axis=-1)
    angle = 2.0 * np.arctan2(twist_s, twist_w)
    return np.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


def _select_continuous_angle_sequence_np(
    angle_wrapped: np.ndarray,
    *,
    raw_lower: float | None = None,
    raw_upper: float | None = None,
) -> np.ndarray:
    angle = np.asarray(angle_wrapped, dtype=np.float64)
    if angle.ndim != 1:
        raise ValueError(f"Expected wrapped angle shape [F], got {angle.shape}")
    if angle.size == 0:
        return angle.copy()

    two_pi = 2.0 * np.pi
    lower = -np.inf if raw_lower is None else float(raw_lower)
    upper = np.inf if raw_upper is None else float(raw_upper)
    out = np.zeros_like(angle)

    def limit_penalty(value: float) -> float:
        below = max(lower - value, 0.0)
        above = max(value - upper, 0.0)
        return float((below + above) ** 2)

    def candidates(base: float, center: float | None = None) -> list[float]:
        if center is None:
            k0 = int(np.round((-base) / two_pi))
        else:
            k0 = int(np.round((center - base) / two_pi))
        return [base + two_pi * float(k0 + dk) for dk in (-1, 0, 1)]

    out[0] = min(
        candidates(float(angle[0])),
        key=lambda value: 50.0 * limit_penalty(value) + 0.01 * float(value * value),
    )
    for frame_idx in range(1, angle.shape[0]):
        prev = float(out[frame_idx - 1])
        out[frame_idx] = min(
            candidates(float(angle[frame_idx]), prev),
            key=lambda value: (
                float((value - prev) ** 2)
                + 20.0 * limit_penalty(value)
                + 0.001 * float(value * value)
            ),
        )
    return out


def _load_amass_npz_metadata(path: Path) -> tuple[int, float]:
    with np.load(path, allow_pickle=False) as data:
        poses = np.asarray(data["poses"])
        trans = np.asarray(data["trans"])
        fps_value = data.get("mocap_framerate", data.get("fps", 0.0))
        if hasattr(fps_value, "item"):
            fps_value = fps_value.item()
    if poses.ndim != 2 or trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(
            f"Invalid AMASS arrays: poses={poses.shape}, trans={trans.shape}"
        )
    if poses.shape[0] != trans.shape[0]:
        raise ValueError(
            f"AMASS frame mismatch: poses={poses.shape[0]}, trans={trans.shape[0]}"
        )
    return int(poses.shape[0]), float(fps_value)


def _target_indices(
    frame_count: int, source_fps: float, target_fps: float
) -> np.ndarray:
    if target_fps <= 0.0 or target_fps >= source_fps:
        return np.arange(frame_count, dtype=np.int64)
    duration = (frame_count - 1) / source_fps
    times = np.arange(0.0, duration + 0.5 / target_fps, 1.0 / target_fps)
    indices = np.rint(times * source_fps).astype(np.int64)
    return np.unique(np.clip(indices, 0, frame_count - 1))


def load_amass_fk_qpos(
    input_path: Path,
    *,
    robot_xml: Path,
    target_fps: float,
) -> tuple[np.ndarray, float]:
    """Load raw AMASS data as direct FK qpos for the reduced humanoid model."""
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    with np.load(input_path, allow_pickle=False) as data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        trans = np.asarray(data["trans"], dtype=np.float32)
        fps_value = data.get("mocap_framerate", data.get("fps", 30.0))
        if hasattr(fps_value, "item"):
            fps_value = fps_value.item()
        source_fps = float(fps_value)

    if poses.ndim != 2 or trans.shape != (poses.shape[0], 3):
        raise ValueError("AMASS poses/trans shapes do not match")
    if poses.shape[0] < 2 or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("AMASS requires at least two frames and positive finite FPS")
    if not np.isfinite(poses).all() or not np.isfinite(trans).all():
        raise ValueError("AMASS poses/trans must be finite")
    if poses.shape[1] < 66:
        raise ValueError(f"Expected at least 66 SMPL pose values, got {poses.shape[1]}")

    source_frame_count = int(poses.shape[0])
    indices = _target_indices(source_frame_count, source_fps, target_fps)
    aligned_fps = (
        source_fps if target_fps <= 0.0 or target_fps >= source_fps else target_fps
    )

    pose_aa = np.concatenate(
        [poses[:, :66], np.zeros((source_frame_count, 6), dtype=np.float32)],
        axis=-1,
    )
    smpl_to_mujoco = [
        SMPL_BONE_ORDER_NAMES.index(name)
        for name in SMPL_MUJOCO_NAMES
        if name in SMPL_BONE_ORDER_NAMES
    ]
    pose_aa_mj = pose_aa.reshape(source_frame_count, 24, 3)[:, smpl_to_mujoco]
    pose_quat = (
        exp_map_to_quat_xyzw(pose_aa_mj.reshape(-1, 3))
        .reshape(source_frame_count, 24, 4)
        .astype(np.float32)
    )

    zup_to_yup = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    yup_to_zup = quat_conjugate_xyzw(zup_to_yup)
    global_rot = compute_global_rotations_xyzw(
        pose_quat.astype(np.float32, copy=False),
        PARENT_INDICES,
    )
    rotated_global_rot = quat_mul_xyzw(
        global_rot.reshape(-1, 4),
        np.broadcast_to(yup_to_zup, global_rot.reshape(-1, 4).shape),
    ).reshape(source_frame_count, -1, 4)
    global_translation = compute_global_translations(
        rotated_global_rot,
        LOCAL_TRANSLATION.astype(np.float32, copy=False),
        PARENT_INDICES,
    )
    global_translation += trans[:, None, :]

    global_rot_wxyz = _normalize_quat_wxyz(
        rotated_global_rot[..., [3, 0, 1, 2]].astype(np.float64)
    )
    global_translation = global_translation.astype(np.float64)
    name_to_index = {name: idx for idx, name in enumerate(SMPL_MUJOCO_NAMES)}

    qpos = np.tile(model.qpos0, (indices.shape[0], 1)).astype(np.float64)
    pelvis_idx = name_to_index["Pelvis"]
    qpos[:, 0:3] = global_translation[indices, pelvis_idx]
    qpos[:, 3:7] = global_rot_wxyz[indices, pelvis_idx]

    for joint_name, (source_parent, source_child) in FK_JOINT_SOURCE_BODIES.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        parent_idx = name_to_index[source_parent]
        child_idx = name_to_index[source_child]
        local_quat = _relative_quat_wxyz(
            global_rot_wxyz[indices, parent_idx],
            global_rot_wxyz[indices, child_idx],
        )
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            qpos[:, qadr : qadr + 4] = local_quat
        elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            angle = _hinge_angle_from_quat_wxyz(
                local_quat,
                np.asarray(model.jnt_axis[joint_id], dtype=np.float64),
            )
            raw_lower = None
            raw_upper = None
            if model.jnt_limited[joint_id]:
                raw_lower = float(model.jnt_range[joint_id, 0])
                raw_upper = float(model.jnt_range[joint_id, 1])
            qpos[:, qadr] = _select_continuous_angle_sequence_np(
                angle,
                raw_lower=raw_lower,
                raw_upper=raw_upper,
            )
            if model.jnt_limited[joint_id]:
                qpos[:, qadr] = np.clip(
                    qpos[:, qadr],
                    float(model.jnt_range[joint_id, 0]),
                    float(model.jnt_range[joint_id, 1]),
                )
        else:
            raise ValueError(
                f"Unsupported FK target joint type {joint_type}: {joint_name}"
            )

    return qpos, float(aligned_fps)


def _normalize_motion_height(
    qpos: np.ndarray,
    model: mujoco.MjModel,
    *,
    height_offset: float = 0.0,
) -> np.ndarray:
    data = mujoco.MjData(model)
    min_z = np.inf
    for row in qpos:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        for geom_id in range(model.ngeom):
            geom_type = int(model.geom_type[geom_id])
            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            size = model.geom_size[geom_id]
            vertical = data.geom_xmat[geom_id].reshape(3, 3)[2]
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = float(size[0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                extent = float(np.abs(vertical) @ size)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = float(size[0] + size[1] * abs(vertical[2]))
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent = float(
                    size[0] * np.linalg.norm(vertical[:2]) + size[1] * abs(vertical[2])
                )
            elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                extent = float(np.linalg.norm(vertical * size))
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = int(model.geom_dataid[geom_id])
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                extent = -float(
                    np.min(model.mesh_vert[start : start + count] @ vertical)
                )
            else:
                raise ValueError(f"Unsupported grounding geometry type: {geom_type}")
            min_z = min(min_z, float(data.geom_xpos[geom_id, 2]) - extent)
    if not np.isfinite(min_z):
        return qpos
    out = qpos.copy()
    out[:, 2] += float(height_offset) - min_z
    return out


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    w0, x0, y0, z0 = np.moveaxis(lhs, -1, 0)
    w1, x1, y1, z1 = np.moveaxis(rhs, -1, 0)
    out = np.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        axis=-1,
    )
    return _normalize_quat_wxyz(out)


def _quat_conj_wxyz(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float64).copy()
    out[..., 1:4] *= -1.0
    return out
