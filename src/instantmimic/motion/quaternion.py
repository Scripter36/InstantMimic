"""Quaternion helpers shared by the motion pipeline.

``copysign`` and ``get_euler_xyz`` are derived from NVIDIA's IsaacGymEnvs
``torch_jit_utils.py``, Copyright (c) 2018-2023, NVIDIA Corporation, licensed
under BSD-3-Clause; see THIRD_PARTY_NOTICES.md. The rest is repository-original
and follows the ``mujoco_warp`` scalar-first ``(w, x, y, z)`` convention rather
than Isaac's ``(x, y, z, w)``.
"""

import numpy as np
import torch
from numpy.typing import DTypeLike


def quat_normalize_np(q: np.ndarray) -> np.ndarray:
    """Normalize and canonicalize sign, preserving the nonzero input dtype."""
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    out = q / norm
    return -out if out[0] < 0.0 else out


def quat_mul_np(
    q0: np.ndarray,
    q1: np.ndarray,
    *,
    dtype: DTypeLike = np.float64,
) -> np.ndarray:
    """Hamilton product for scalar-first NumPy quaternions."""
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.asarray(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + w1 * x0 + y0 * z1 - z0 * y1,
            w0 * y1 + w1 * y0 + z0 * x1 - x0 * z1,
            w0 * z1 + w1 * z0 + x0 * y1 - y0 * x1,
        ],
        dtype=dtype,
    )


def quat_conj_np(q: np.ndarray) -> np.ndarray:
    """Conjugate one scalar-first quaternion as float64."""
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate_vec_np(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector after scalar-first quaternion normalization."""
    q = quat_normalize_np(q)
    xyz = np.asarray(q[1:4], dtype=np.float64)
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + q[0] * twice_cross + np.cross(xyz, twice_cross)


@torch.jit.script
def quat_mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = q0[..., 0:1], q0[..., 1:2], q0[..., 2:3], q0[..., 3:4]
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    return torch.cat(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + w1 * x0 + y0 * z1 - z0 * y1,
            w0 * y1 + w1 * y0 + z0 * x1 - x0 * z1,
            w0 * z1 + w1 * z0 + x0 * y1 - y0 * x1,
        ),
        dim=-1,
    )


@torch.jit.script
def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Hamilton inverse without assuming a unit-length quaternion."""

    return quat_conj(q) / torch.sum(q.square(), dim=-1, keepdim=True)


def quat_rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate by a scalar-first unit quaternion."""

    xyz = q[..., 1:]
    twice_cross = torch.cross(xyz, vector, dim=-1) * 2.0
    return vector + q[..., :1] * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def quat_rotate_inverse_unit(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Inverse-rotate using the unit-quaternion expanded formula."""

    xyz = q[..., 1:]
    a = vector * (2.0 * q[..., :1].square() - 1.0)
    b = torch.cross(xyz, vector, dim=-1) * q[..., :1] * 2.0
    c = xyz * (torch.sum(xyz * vector, dim=-1, keepdim=True) * 2.0)
    return a - b + c


def quat_rotate_inverse(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply the conjugate rotation formula without normalizing the input."""

    return quat_rotate(quat_conj(q), vector)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first quaternions to matrices after normalization."""

    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(
        torch.finfo(q.dtype).eps
    )
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


@torch.jit.script
def quat_to_scaled_axis_angle(q: torch.Tensor) -> torch.Tensor:
    # stable log-map without data-dependent python branching
    q = q / (torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8))
    # Ensure shortest path: flip quaternion if w < 0
    q = torch.where(q[..., 0:1] < 0.0, -q, q)
    w = torch.clamp(q[..., 0:1], -1.0, 1.0)
    v = q[..., 1:4]
    v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(v_norm, w)
    axis = v / v_norm.clamp_min(1.0e-8)
    aa = axis * angle
    aa_small = 2.0 * v
    return torch.where(v_norm < 1.0e-6, aa_small, aa)


@torch.jit.script
def scaled_axis_angle_to_quat(aa: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True)
    axis = aa / angle.clamp_min(1.0e-8)
    half = 0.5 * angle
    return torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q, extrinsic: bool = True):
    if extrinsic:
        qw, qx, qy, qz = 0, 1, 2, 3
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
        cosr_cosp = (
            q[:, qw] * q[:, qw]
            - q[:, qx] * q[:, qx]
            - q[:, qy] * q[:, qy]
            + q[:, qz] * q[:, qz]
        )
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
        pitch = torch.where(
            torch.abs(sinp) >= 1, copysign(torch.pi / 2.0, sinp), torch.asin(sinp)
        )

        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
        cosy_cosp = (
            q[:, qw] * q[:, qw]
            + q[:, qx] * q[:, qx]
            - q[:, qy] * q[:, qy]
            - q[:, qz] * q[:, qz]
        )
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)
    else:
        raise NotImplementedError("Intrinsic euler angles are not implemented yet")
