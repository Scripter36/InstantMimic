from __future__ import annotations

from typing import Literal

import torch
from pymotion.rotations import quat_torch

from instantmimic.motion import quaternion as torch_utils

ReferenceFrame = Literal["world", "translation", "heading", "root"]


def quat_diff_to_axis_angle(q_ref: torch.Tensor, q_sim: torch.Tensor) -> torch.Tensor:
    """Compute the shortest-path quaternion difference as a scaled axis-angle."""
    q_rel = torch_utils.quat_mul(torch_utils.quat_conj(q_ref), q_sim)
    q_rel = torch.where(q_rel[..., 0:1] < 0.0, -q_rel, q_rel)
    return torch_utils.quat_to_scaled_axis_angle(q_rel)


def get_heading_frame(
    root_pos: torch.Tensor, root_quat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the root x-y translation frame and yaw-only quaternion."""
    root_pos_xz = root_pos.clone()
    root_pos_xz[..., 2] = 0.0

    forward = torch.zeros_like(root_pos)
    forward[..., 0] = 1.0
    root_forward = quat_torch.mul_vec(root_quat, forward)
    root_yaw = torch.atan2(root_forward[..., 1], root_forward[..., 0])

    half_yaw = root_yaw * 0.5
    root_yaw_quat = torch.zeros_like(root_quat)
    root_yaw_quat[..., 0] = torch.cos(half_yaw)
    root_yaw_quat[..., 3] = torch.sin(half_yaw)
    return root_pos_xz, root_yaw_quat


def gather_reference_body_state(
    reference_body_state: torch.Tensor,
    ref_frame_idx: torch.Tensor,
    body_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather reference body pose and velocity for the current frame index."""
    frame_count = int(reference_body_state.shape[0])
    if frame_count <= 0:
        raise ValueError("reference_body_state must contain at least one frame.")
    ref_idx = torch.remainder(ref_frame_idx.long(), frame_count)
    ref_state = reference_body_state[ref_idx]
    if body_indices is not None:
        ref_state = ref_state.index_select(1, body_indices)
    return (
        ref_state[..., :3],
        ref_state[..., 3:7],
        ref_state[..., 7:10],
        ref_state[..., 10:13],
    )


def _express_positions_in_frame(
    body_pos: torch.Tensor,
    root_pos: torch.Tensor | None,
    root_quat: torch.Tensor | None,
    frame: ReferenceFrame,
) -> torch.Tensor:
    if frame == "world":
        return body_pos

    if root_pos is None:
        raise ValueError(f"root_pos is required when frame='{frame}'.")

    local_pos = body_pos - root_pos[:, None, :]
    if frame == "translation":
        return local_pos

    if root_quat is None:
        raise ValueError(f"root_quat is required when frame='{frame}'.")

    if frame == "heading":
        _, root_quat = get_heading_frame(root_pos, root_quat)
    elif frame != "root":
        raise ValueError(
            f"Unsupported reference frame '{frame}'. "
            "Use one of: world, translation, heading, root."
        )

    root_inv = quat_torch.inverse(root_quat)
    root_inv_expanded = root_inv[:, None, :].expand(-1, local_pos.shape[1], -1)
    return quat_torch.mul_vec(root_inv_expanded, local_pos)


def express_vectors_in_frame(
    vectors: torch.Tensor,
    *,
    root_quat: torch.Tensor | None = None,
    frame: ReferenceFrame = "world",
) -> torch.Tensor:
    """Express vectors in the requested frame.

    For velocity-like vectors, translation-only frames are identical to world.
    """
    if frame in ("world", "translation"):
        return vectors

    if root_quat is None:
        raise ValueError(f"root_quat is required when frame='{frame}'.")

    if frame == "heading":
        zeros = torch.zeros(
            root_quat.shape[:-1] + (3,),
            device=root_quat.device,
            dtype=root_quat.dtype,
        )
        _, root_quat = get_heading_frame(zeros, root_quat)
    elif frame != "root":
        raise ValueError(
            f"Unsupported reference frame '{frame}'. "
            "Use one of: world, translation, heading, root."
        )

    root_inv = quat_torch.inverse(root_quat)
    root_inv_expanded = root_inv[:, None, :].expand(-1, vectors.shape[1], -1)
    return quat_torch.mul_vec(root_inv_expanded, vectors)


def compute_mean_position_error(
    sim_body_pos: torch.Tensor,
    ref_body_pos: torch.Tensor,
    *,
    sim_root_pos: torch.Tensor | None = None,
    ref_root_pos: torch.Tensor | None = None,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    frame: ReferenceFrame = "world",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean Euclidean position error per env and the per-body distances."""
    sim_cmp = _express_positions_in_frame(
        sim_body_pos, sim_root_pos, sim_root_quat, frame
    )
    ref_cmp = _express_positions_in_frame(
        ref_body_pos, ref_root_pos, ref_root_quat, frame
    )
    per_body_error = torch.linalg.norm(sim_cmp - ref_cmp, dim=-1)
    if per_body_error.shape[1] == 0:
        mean_error = torch.zeros(
            per_body_error.shape[0],
            device=per_body_error.device,
            dtype=per_body_error.dtype,
        )
    else:
        mean_error = per_body_error.mean(dim=-1)
    return mean_error, per_body_error


def compute_average_pose_error(
    sim_body_pos: torch.Tensor,
    ref_body_pos: torch.Tensor,
    *,
    sim_root_pos: torch.Tensor | None = None,
    ref_root_pos: torch.Tensor | None = None,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    frame: ReferenceFrame = "root",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute average pose error as mean body-position error in a chosen frame."""
    return compute_mean_position_error(
        sim_body_pos,
        ref_body_pos,
        sim_root_pos=sim_root_pos,
        ref_root_pos=ref_root_pos,
        sim_root_quat=sim_root_quat,
        ref_root_quat=ref_root_quat,
        frame=frame,
    )


def compute_end_effector_error(
    sim_body_pos: torch.Tensor,
    ref_body_pos: torch.Tensor,
    *,
    sim_root_pos: torch.Tensor | None = None,
    ref_root_pos: torch.Tensor | None = None,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    frame: ReferenceFrame = "root",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean end-effector position error in a chosen frame."""
    return compute_mean_position_error(
        sim_body_pos,
        ref_body_pos,
        sim_root_pos=sim_root_pos,
        ref_root_pos=ref_root_pos,
        sim_root_quat=sim_root_quat,
        ref_root_quat=ref_root_quat,
        frame=frame,
    )


def compute_mean_velocity_error(
    sim_body_vel: torch.Tensor,
    ref_body_vel: torch.Tensor,
    *,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    frame: ReferenceFrame = "world",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean Euclidean velocity error per env and per body."""
    sim_cmp = express_vectors_in_frame(
        sim_body_vel,
        root_quat=sim_root_quat,
        frame=frame,
    )
    ref_cmp = express_vectors_in_frame(
        ref_body_vel,
        root_quat=ref_root_quat,
        frame=frame,
    )
    per_body_error = torch.linalg.norm(sim_cmp - ref_cmp, dim=-1)
    if per_body_error.shape[1] == 0:
        mean_error = torch.zeros(
            per_body_error.shape[0],
            device=per_body_error.device,
            dtype=per_body_error.dtype,
        )
    else:
        mean_error = per_body_error.mean(dim=-1)
    return mean_error, per_body_error


def compute_mean_acceleration_error(
    sim_body_vel: torch.Tensor,
    ref_body_vel: torch.Tensor,
    prev_sim_body_vel: torch.Tensor,
    prev_ref_body_vel: torch.Tensor,
    *,
    dt: float,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    prev_sim_root_quat: torch.Tensor | None = None,
    prev_ref_root_quat: torch.Tensor | None = None,
    frame: ReferenceFrame = "world",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean acceleration error via finite-difference velocities."""
    dt_safe = max(float(dt), 1.0e-8)
    sim_cmp = express_vectors_in_frame(
        sim_body_vel,
        root_quat=sim_root_quat,
        frame=frame,
    )
    ref_cmp = express_vectors_in_frame(
        ref_body_vel,
        root_quat=ref_root_quat,
        frame=frame,
    )
    prev_sim_cmp = express_vectors_in_frame(
        prev_sim_body_vel,
        root_quat=prev_sim_root_quat,
        frame=frame,
    )
    prev_ref_cmp = express_vectors_in_frame(
        prev_ref_body_vel,
        root_quat=prev_ref_root_quat,
        frame=frame,
    )
    sim_acc = (sim_cmp - prev_sim_cmp) / dt_safe
    ref_acc = (ref_cmp - prev_ref_cmp) / dt_safe
    per_body_error = torch.linalg.norm(sim_acc - ref_acc, dim=-1)
    if per_body_error.shape[1] == 0:
        mean_error = torch.zeros(
            per_body_error.shape[0],
            device=per_body_error.device,
            dtype=per_body_error.dtype,
        )
    else:
        mean_error = per_body_error.mean(dim=-1)
    return mean_error, per_body_error
