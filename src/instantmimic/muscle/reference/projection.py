"""Torch statements of the two projections the Warp qfrc kernels perform.

`muscle_qfrc_projector.py` computes the same two things on CUDA:
`segment_force_magnitude_to_body_wrench` corresponds to the segment walk that
accumulates a per-body wrench, and `project_body_wrench_to_torque` to the cdof
contraction that turns that wrench into generalized force.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def segment_force_magnitude_to_body_wrench(
    force_magnitude: torch.Tensor,
    *,
    seg_start: torch.Tensor | None,
    seg_end: torch.Tensor | None,
    seg_dir: torch.Tensor | None,
    xipos: torch.Tensor,
    body_count: int,
    apply_segment_muscle_idx: torch.Tensor,
    apply_segment_start_idx: torch.Tensor,
    apply_segment_body_start_idx: torch.Tensor,
    apply_segment_body_end_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spread each segment's force onto the two bodies its endpoints sit on."""

    batch_size = force_magnitude.shape[0]
    body_force = torch.zeros(
        batch_size,
        body_count,
        3,
        device=force_magnitude.device,
        dtype=force_magnitude.dtype,
    )
    body_torque = torch.zeros(
        batch_size,
        body_count,
        3,
        device=force_magnitude.device,
        dtype=force_magnitude.dtype,
    )
    if (
        seg_start is None
        or seg_end is None
        or seg_dir is None
        or apply_segment_start_idx.numel() == 0
    ):
        return body_force, body_torque

    seg_force = (
        force_magnitude.index_select(1, apply_segment_muscle_idx).unsqueeze(-1)
        * seg_dir
    )
    com_start = xipos.index_select(1, apply_segment_body_start_idx)
    com_end = xipos.index_select(1, apply_segment_body_end_idx)
    torque_start = torch.cross(seg_start - com_start, seg_force, dim=-1)
    torque_end = torch.cross(seg_end - com_end, -seg_force, dim=-1)
    body_force.scatter_add_(
        1,
        apply_segment_body_start_idx.view(1, -1, 1).expand(batch_size, -1, 3),
        seg_force,
    )
    body_force.scatter_add_(
        1,
        apply_segment_body_end_idx.view(1, -1, 1).expand(batch_size, -1, 3),
        -seg_force,
    )
    body_torque.scatter_add_(
        1,
        apply_segment_body_start_idx.view(1, -1, 1).expand(batch_size, -1, 3),
        torque_start,
    )
    body_torque.scatter_add_(
        1,
        apply_segment_body_end_idx.view(1, -1, 1).expand(batch_size, -1, 3),
        torque_end,
    )
    return body_force, body_torque


def project_body_wrench_to_torque(
    *,
    body_force: torch.Tensor,
    body_torque: torch.Tensor,
    xipos: torch.Tensor,
    subtree_com: torch.Tensor,
    cdof: torch.Tensor,
    body_count: int,
    full_torque_dim: int,
    root_dof_count: int,
    body_rootid: Sequence[int],
    dof_subtree_mask: torch.Tensor,
) -> torch.Tensor:
    """Contract a per-body wrench with cdof into generalized force.

    The root DOFs are dropped on the way out: the actor only ever writes the
    actuated tail.
    """

    batch_size = body_force.shape[0]
    linear_cdof = cdof[..., 3:]
    angular_cdof = cdof[..., :3]
    qfrc = torch.zeros(
        batch_size,
        full_torque_dim,
        device=body_force.device,
        dtype=body_force.dtype,
    )
    for body_idx in range(body_count):
        force = body_force[:, body_idx].unsqueeze(1)
        torque = body_torque[:, body_idx].unsqueeze(1)
        root_body_idx = body_rootid[body_idx]
        offset = xipos[:, body_idx] - subtree_com[:, root_body_idx]
        offset_expanded = offset.unsqueeze(1).expand(-1, full_torque_dim, -1)
        force_term = torch.sum(linear_cdof * force, dim=-1)
        torque_term = torch.sum(angular_cdof * torque, dim=-1)
        offset_force_term = torch.sum(
            torch.cross(angular_cdof, offset_expanded, dim=-1)
            * force.expand(-1, full_torque_dim, -1),
            dim=-1,
        )
        qfrc.add_(
            (force_term + torque_term + offset_force_term)
            * dof_subtree_mask[:, body_idx].unsqueeze(0)
        )
    return qfrc[:, root_dof_count:]
