from __future__ import annotations

import numpy as np

SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

SMPL_MUJOCO_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]

PARENT_INDICES = [
    -1,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    12,
    11,
    14,
    15,
    16,
    17,
    11,
    19,
    20,
    21,
    22,
]

LOCAL_TRANSLATION = np.array(
    [
        [0.0000, 0.0000, 0.0000],
        [-0.0068, 0.0695, -0.0914],
        [-0.0045, 0.0343, -0.3752],
        [-0.0437, -0.0136, -0.3980],
        [0.1193, 0.0124, -0.0258],
        [-0.0043, -0.0677, -0.0905],
        [-0.0089, -0.0383, -0.3826],
        [-0.0423, 0.0158, -0.3984],
        [0.1193, -0.0124, -0.0258],
        [-0.0267, -0.0025, 0.1090],
        [0.0011, 0.0055, 0.1352],
        [0.0254, 0.0015, 0.0529],
        [-0.0429, -0.0028, 0.2139],
        [0.0513, 0.0052, 0.0650],
        [-0.0341, 0.0788, 0.1217],
        [-0.0089, 0.0910, 0.0305],
        [-0.0275, 0.2596, -0.0128],
        [-0.0012, 0.2492, 0.0090],
        [-0.0149, 0.0840, -0.0082],
        [-0.0386, -0.0818, 0.1188],
        [-0.0091, -0.0960, 0.0326],
        [-0.0214, -0.2537, -0.0133],
        [-0.0056, -0.2553, 0.0078],
        [-0.0103, -0.0846, -0.0061],
    ],
    dtype=np.float32,
)


def normalize_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def normalize(value: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.clip(norm, eps, None)


def quat_conjugate_xyzw(quat: np.ndarray) -> np.ndarray:
    return np.concatenate((-quat[..., :3], quat[..., 3:]), axis=-1)


def quat_mul_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = lhs[..., 0], lhs[..., 1], lhs[..., 2], lhs[..., 3]
    x2, y2, z2, w2 = rhs[..., 0], rhs[..., 1], rhs[..., 2], rhs[..., 3]
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )


def quat_rotate_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat_vec = quat[..., :3]
    quat_w = quat[..., 3:]
    cross = 2.0 * np.cross(quat_vec, vec, axis=-1)
    return vec + quat_w * cross + np.cross(quat_vec, cross, axis=-1)


def axis_angle_to_quat_xyzw(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    half = np.expand_dims(angle * 0.5, axis=-1)
    xyz = normalize(axis) * np.sin(half)
    w = np.cos(half)
    return normalize(np.concatenate((xyz, w), axis=-1))


def exp_map_to_quat_xyzw(exp_map: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(exp_map, axis=-1)
    axis = exp_map / np.clip(np.expand_dims(angle, axis=-1), 1.0e-9, None)
    angle = normalize_angle(angle)

    default_axis = np.zeros_like(exp_map)
    default_axis[..., -1] = 1.0
    mask = np.abs(angle) > 1.0e-5
    axis = np.where(np.expand_dims(mask, axis=-1), axis, default_axis)
    angle = np.where(mask, angle, np.zeros_like(angle))
    return axis_angle_to_quat_xyzw(axis, angle)


def compute_global_rotations_xyzw(
    local_quats: np.ndarray,
    parents: list[int],
) -> np.ndarray:
    globals_per_joint: list[np.ndarray] = []
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx == -1:
            globals_per_joint.append(local_quats[:, joint_idx, :])
        else:
            globals_per_joint.append(
                quat_mul_xyzw(
                    globals_per_joint[parent_idx], local_quats[:, joint_idx, :]
                )
            )
    return np.stack(globals_per_joint, axis=1)


def compute_global_translations(
    global_rotations: np.ndarray,
    local_offsets: np.ndarray,
    parents: list[int],
) -> np.ndarray:
    frame_count, joint_count, _ = global_rotations.shape
    global_translations = np.zeros(
        (frame_count, joint_count, 3),
        dtype=global_rotations.dtype,
    )
    for joint_idx, parent_idx in enumerate(parents):
        if parent_idx == -1:
            global_translations[:, joint_idx, :] = local_offsets[joint_idx]
            continue
        parent_pos = global_translations[:, parent_idx, :]
        parent_rot = global_rotations[:, parent_idx, :]
        offset = np.broadcast_to(local_offsets[joint_idx], (frame_count, 3))
        global_translations[:, joint_idx, :] = parent_pos + quat_rotate_xyzw(
            parent_rot,
            offset,
        )
    return global_translations
