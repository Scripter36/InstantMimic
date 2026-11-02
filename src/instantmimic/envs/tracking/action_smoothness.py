"""Per-step command history for the action-smoothness reward term.

The term scores ``mean_d delta_d**2``, where ``delta`` is the change of a
per-control-step command since the previous step, divided by a gain:
``delta_d = (c_d(t) - c_d(t - 1)) / gain`` with ``c_d = values[:, d] *
column_scale[column_index[d]]``. The previous command is kept in a persistent
buffer updated by the same kernel, so the update is allocation-free and runs
inside a captured step. A reset clears a per-environment flag; the first step
after it records the command and scores ``delta = 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import warp as wp


@wp.kernel(enable_backward=False)
def action_smoothness_error_kernel(
    values: wp.array2d(dtype=wp.float32),
    column_scale: wp.array(dtype=wp.float32),
    column_index: wp.array(dtype=wp.int32),
    inv_gain: float,
    previous: wp.array2d(dtype=wp.float32),
    has_previous: wp.array(dtype=wp.int32),
    error: wp.array(dtype=wp.float32),
):
    env_idx = wp.tid()
    width = values.shape[1]
    valid = has_previous[env_idx] != 0
    total = float(0.0)
    for d in range(width):
        command = values[env_idx, d] * column_scale[column_index[d]] * inv_gain
        if valid:
            delta = command - previous[env_idx, d]
            total = total + delta * delta
        previous[env_idx, d] = command
    # Nested branches rather than a short-circuit boolean chain, which older
    # NVRTC releases miscompile on some GPUs.
    result = float(0.0)
    if valid:
        if width > 0:
            result = total / float(width)
    error[env_idx] = result
    has_previous[env_idx] = 1


@wp.kernel(enable_backward=False)
def clear_action_smoothness_history_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    has_previous: wp.array(dtype=wp.int32),
    error: wp.array(dtype=wp.float32),
):
    env_idx = wp.tid()
    if reset_mask[env_idx]:
        has_previous[env_idx] = 0
        error[env_idx] = 0.0


@dataclass
class ActionSmoothnessState:
    """Persistent buffers of one environment batch's smoothness term."""

    source: str
    values: torch.Tensor
    column_scale: torch.Tensor
    column_index: torch.Tensor
    inv_gain: float
    previous: torch.Tensor
    has_previous: torch.Tensor
    error: torch.Tensor


def create_action_smoothness_state(
    *,
    source: str,
    values: torch.Tensor,
    column_scale: torch.Tensor,
    column_index: torch.Tensor,
    gain: float,
) -> ActionSmoothnessState:
    """Allocate the history for a borrowed ``(num_envs, width)`` command buffer."""

    if values.ndim != 2:
        raise ValueError("the per-step command must be a (num_envs, width) tensor")
    if values.dtype != torch.float32 or column_scale.dtype != torch.float32:
        raise TypeError("the per-step command and its scale must be float32")
    width = int(values.shape[1])
    if column_index.ndim != 1 or int(column_index.numel()) != width:
        raise ValueError(
            f"column_index has {int(column_index.numel())} entries for {width} "
            "command columns"
        )
    if width > 0:
        index_min = int(column_index.min().item())
        index_max = int(column_index.max().item())
        if index_min < 0 or index_max >= int(column_scale.numel()):
            raise ValueError("column_index is out of range of column_scale")
    gain = float(gain)
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("the action-smoothness gain must be finite and > 0")
    num_envs = int(values.shape[0])
    device = values.device
    return ActionSmoothnessState(
        source=str(source),
        values=values,
        column_scale=column_scale,
        column_index=column_index.to(dtype=torch.int32).contiguous(),
        inv_gain=1.0 / gain,
        previous=torch.zeros((num_envs, width), device=device, dtype=torch.float32),
        has_previous=torch.zeros(num_envs, device=device, dtype=torch.int32),
        error=torch.zeros(num_envs, device=device, dtype=torch.float32),
    )


def update_action_smoothness_error(state: ActionSmoothnessState) -> torch.Tensor:
    """Score the current command against the previous one and remember it.

    Call once per control step, after the step's command is in ``values``.
    """

    num_envs = int(state.values.shape[0])
    wp.launch(
        action_smoothness_error_kernel,
        dim=(num_envs,),
        inputs=[
            state.values,
            state.column_scale,
            state.column_index,
            float(state.inv_gain),
        ],
        outputs=[state.previous, state.has_previous, state.error],
        device=wp.device_from_torch(state.values.device),
    )
    return state.error


def reset_action_smoothness_history(
    state: ActionSmoothnessState, reset_mask: torch.Tensor
) -> None:
    """Forget the previous command of the masked environments."""

    wp.launch(
        clear_action_smoothness_history_kernel,
        dim=(int(state.has_previous.shape[0]),),
        inputs=[reset_mask],
        outputs=[state.has_previous, state.error],
        device=wp.device_from_torch(state.has_previous.device),
    )


__all__ = (
    "ActionSmoothnessState",
    "action_smoothness_error_kernel",
    "clear_action_smoothness_history_kernel",
    "create_action_smoothness_state",
    "reset_action_smoothness_history",
    "update_action_smoothness_error",
)
