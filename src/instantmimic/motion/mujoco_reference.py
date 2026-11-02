from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class MujocoReferencePayload:
    """Startup-materialized MuJoCo reference channels for a motion sequence."""

    xpos: torch.Tensor
    xquat: torch.Tensor
    cvel: torch.Tensor | None
    subtree_com: torch.Tensor | None
    reference_body_state: torch.Tensor
    ref_com: torch.Tensor | None


def _to_target_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(array).to(device=device, dtype=torch.float32).contiguous()


def materialize_mujoco_reference_payload(
    *,
    model: mujoco.MjModel,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    device: torch.device,
    root_body_idx: int | None = None,
    include_cvel: bool = False,
    include_subtree_com: bool = True,
) -> MujocoReferencePayload:
    """Materialize MuJoCo FK channels with one forward pass per motion frame.

    MuJoCo is evaluated on CPU into completed NumPy arrays. Each completed channel
    is then transferred once to ``device``; in particular, no frame-sized tensor
    is copied to the target device inside the FK loop.
    """
    if qpos.ndim != 2 or qvel.ndim != 2:
        raise ValueError(
            "qpos and qvel must have shape [frames, generalized coordinates]."
        )
    if int(qpos.shape[0]) != int(qvel.shape[0]):
        raise ValueError("qpos and qvel must contain the same number of motion frames.")
    if int(qpos.shape[1]) != int(model.nq):
        raise ValueError(
            f"qpos has dimension {int(qpos.shape[1])}, model expects nq={model.nq}."
        )
    if int(qvel.shape[1]) != int(model.nv):
        raise ValueError(
            f"qvel has dimension {int(qvel.shape[1])}, model expects nv={model.nv}."
        )

    if root_body_idx is not None:
        root_body_idx = int(root_body_idx)
        if not 0 <= root_body_idx < int(model.nbody):
            raise ValueError(
                f"root_body_idx={root_body_idx} is outside [0, {model.nbody})."
            )

    frame_count = int(qpos.shape[0])
    xpos_np = np.empty((frame_count, model.nbody, 3), dtype=np.float32)
    xquat_np = np.empty((frame_count, model.nbody, 4), dtype=np.float32)
    cvel_np = (
        np.empty((frame_count, model.nbody, 6), dtype=np.float32)
        if include_cvel
        else None
    )
    subtree_com_np = (
        np.empty((frame_count, model.nbody, 3), dtype=np.float32)
        if include_subtree_com
        else None
    )
    ref_com_np = (
        np.empty((frame_count, 3), dtype=np.float32)
        if root_body_idx is not None
        else None
    )
    reference_body_state_np = np.empty((frame_count, model.nbody, 13), dtype=np.float32)

    qpos_np = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    qvel_np = qvel.detach().cpu().numpy().astype(np.float64, copy=False)
    data = mujoco.MjData(model)
    for frame_idx in range(frame_count):
        data.qpos[:] = qpos_np[frame_idx]
        data.qvel[:] = qvel_np[frame_idx]
        mujoco.mj_forward(model, data)

        xpos_np[frame_idx] = data.xpos
        xquat_np[frame_idx] = data.xquat
        if cvel_np is not None:
            cvel_np[frame_idx] = data.cvel
        if subtree_com_np is not None:
            subtree_com_np[frame_idx] = data.subtree_com
        if ref_com_np is not None:
            ref_com_np[frame_idx] = data.subtree_com[root_body_idx]
        reference_body_state_np[frame_idx, :, :3] = xpos_np[frame_idx]
        reference_body_state_np[frame_idx, :, 3:7] = xquat_np[frame_idx]
        if cvel_np is None:
            reference_body_state_np[frame_idx, :, 7:10] = data.cvel[:, 3:6]
            reference_body_state_np[frame_idx, :, 10:13] = data.cvel[:, :3]
        else:
            reference_body_state_np[frame_idx, :, 7:10] = cvel_np[frame_idx, :, 3:6]
            reference_body_state_np[frame_idx, :, 10:13] = cvel_np[frame_idx, :, :3]

    target_device = torch.device(device)
    return MujocoReferencePayload(
        xpos=_to_target_tensor(xpos_np, target_device),
        xquat=_to_target_tensor(xquat_np, target_device),
        cvel=(None if cvel_np is None else _to_target_tensor(cvel_np, target_device)),
        subtree_com=(
            None
            if subtree_com_np is None
            else _to_target_tensor(subtree_com_np, target_device)
        ),
        reference_body_state=_to_target_tensor(reference_body_state_np, target_device),
        ref_com=(
            None if ref_com_np is None else _to_target_tensor(ref_com_np, target_device)
        ),
    )


__all__ = (
    "MujocoReferencePayload",
    "materialize_mujoco_reference_payload",
)
