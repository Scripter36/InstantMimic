"""Quaternion helpers for the Torch reference geometry.

`utils/mjw_kernels/private.py` exposes the Warp equivalents; these operate on
batched Torch tensors and carry gradients.
"""

from __future__ import annotations

import torch


def quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    """Normalize `(..., 4)` wxyz quaternions, leaving a near-zero one alone."""

    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)


def quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate `vec` by the wxyz quaternion `quat`, normalizing it first."""

    quat = quat_normalize(quat)
    quat_xyz = quat[..., 1:]
    uv = torch.cross(quat_xyz, vec, dim=-1)
    uuv = torch.cross(quat_xyz, uv, dim=-1)
    return vec + 2.0 * (quat[..., :1] * uv + uuv)
