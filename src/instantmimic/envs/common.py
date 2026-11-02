"""Types shared by every environment, plus Warp-to-torch view helpers."""

from __future__ import annotations

from typing import Any, TypeVar

import gymnasium as gym
import torch
import warp as wp

SpaceType = TypeVar("SpaceType", gym.spaces.Space, int, set, tuple, list, dict)
"""Anything a config may write for an observation, state or action space.

`instantmimic.utils.spaces.spec_to_gym_space` resolves each of these forms.
"""

VecEnvObs = dict[str, torch.Tensor | dict[str, torch.Tensor]]
"""One step of observations for every sub-environment, keyed by group.

`"policy"` is required and feeds the actor. `"critic"` is optional and, when
present, feeds an asymmetric critic instead of the policy group. A group holds
either a single concatenated tensor or a mapping of named terms; this repository
uses the concatenated form throughout. The step pipeline binds pre-allocated
tensors through `DirectRLEnv._bind_step_outputs` for the `_get_observations` hook
to write into, so that a captured step stays allocation-free.
"""

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]
"""What `DirectRLEnv.step` hands back, batched over sub-environments.

In order: the observation groups, the per-environment reward, the terminated
flag, the time-out flag, and an extras mapping. Terminated and timed-out are
reported separately because bootstrapping treats them differently -- a time-out
is not an absorbing state, so its value estimate must still be carried.
"""

_WARP_SCALAR_TO_TORCH_DTYPE = {
    wp.float16: torch.float16,
    wp.float32: torch.float32,
    wp.float64: torch.float64,
    wp.int8: torch.int8,
    wp.int16: torch.int16,
    wp.int32: torch.int32,
    wp.int64: torch.int64,
    wp.uint8: torch.uint8,
    wp.bool: torch.bool,
}


def _torch_device_from_warp_device(device: Any) -> torch.device:
    if device.is_cpu:
        return torch.device("cpu")
    return torch.device(f"cuda:{int(device.ordinal)}")


def warp_to_torch(array: Any, requires_grad: bool | None = None) -> torch.Tensor:
    if int(array.size) != 0:
        return wp.to_torch(array, requires_grad=requires_grad)

    if requires_grad is None:
        requires_grad = bool(array.requires_grad)
    scalar_type = getattr(array.dtype, "_wp_scalar_type_", None) or array.dtype
    dtype = _WARP_SCALAR_TO_TORCH_DTYPE.get(scalar_type)
    if dtype is None:
        raise TypeError(
            f"Unsupported empty Warp array dtype for torch view: {array.dtype}"
        )

    dtype_shape = getattr(array.dtype, "_shape_", None) or ()
    tensor = torch.empty(
        tuple(array.shape) + tuple(dtype_shape),
        dtype=dtype,
        device=_torch_device_from_warp_device(array.device),
    )
    if requires_grad and tensor.is_floating_point():
        tensor.requires_grad_(True)
    return tensor
