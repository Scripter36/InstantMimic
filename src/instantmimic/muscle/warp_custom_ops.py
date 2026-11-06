"""Opaque torch.library operators for the Warp-backed muscle autograd path.

Compiled RL losses evaluate a musculoskeletal policy inside a
``torch.compile(fullgraph=True)`` region. Warp launches cannot be traced:
``wp.from_torch`` reads ``data_ptr``/``ctypes.sizeof`` and the hand-written
gradient is a custom autograd Function. Every Warp entry point reachable from
such a region therefore crosses one dispatcher custom op whose body calls the
existing Warp implementation, whose fake kernel only returns metadata, and
whose registered backward calls the existing Warp gradient implementation.

The bodies stay outside Dynamo/AOTAutograd and keep the layer's persistent
geometry workspaces, so they behave like the eager launches they replace: they
run on a ``wp.Stream`` matching the current torch stream and are captured into
the external update CUDA graph unchanged. The shared registry in
utils.torch_warp_capture keeps one wp.Stream per CUDA stream handle; a
temporary wrapper being garbage collected unregisters a handle another stream
still uses, which makes the next capture fail with "Warp error: unknown
stream".

Operators are generic over layer instances: the tensors are the whole data
interface and ``layer_key`` identifies the owning ``MuscleDynamicsLayer``
through a weak registry, so no operator closes over a single layer.
"""

from __future__ import annotations

import contextlib
import itertools
import weakref
from collections.abc import Iterator
from typing import Any

import torch
import warp as wp

from instantmimic.utils.torch_warp_capture import (
    warp_stream_for_handle,
    warp_stream_from_torch,
)

_LAYERS: dict[int, weakref.ReferenceType[Any]] = {}
_LAYER_OWNERS: dict[int, int] = {}
_LAYER_KEYS = itertools.count(1)


def register_warp_op_layer(layer: Any) -> int:
    """Give layer a fresh dispatcher handle.

    Allocating a handle calls next() on the key counter, which Dynamo cannot
    trace inside a compiled loss. A layer copies itself at a Python boundary
    (__copy__/__deepcopy__), so it registers there and its first compiled call
    only reads a plain int.
    """
    key = next(_LAYER_KEYS)
    layer._warp_op_key = key
    _LAYERS[key] = weakref.ref(layer)
    _LAYER_OWNERS[key] = id(layer)
    return int(key)


def warp_op_key(layer: Any) -> int:
    key = getattr(layer, "_warp_op_key", None)
    # A shallow or deep copy carries the original's key. Validating the owner
    # by identity gives every live instance its own registration rather than
    # silently resolving the copy to the layer it was copied from.
    if key is not None and _LAYER_OWNERS.get(int(key)) == id(layer):
        return int(key)
    return register_warp_op_layer(layer)


def _layer_for(key: int) -> Any:
    reference = _LAYERS.get(int(key))
    layer = None if reference is None else reference()
    if layer is None:
        raise RuntimeError(
            f"Warp custom-op layer key {key} is not registered; the muscle layer "
            "was released while a compiled loss still referenced it."
        )
    return layer


@contextlib.contextmanager
def warp_stream(device: Any) -> Iterator[None]:
    """Run Warp launches on the torch stream that is current for the device."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        yield
        return
    torch_stream = torch.cuda.current_stream(torch_device)
    # torch hands back a fresh Python object for the same CUDA stream on every
    # call; the shared registry keys on the integer handle instead.
    selected = warp_stream_from_torch(torch_stream)
    with wp.ScopedStream(selected, sync_enter=False, sync_exit=False):
        yield


@torch.library.custom_op("instantmimic_muscle::torque_forward", mutates_args=())
def muscle_torque_forward(
    activation: torch.Tensor,
    raw_context: torch.Tensor,
    layer_key: int,
    include_passive_force: bool,
) -> torch.Tensor:
    """Differentiable activation-to-torque projection of one layer."""
    layer = _layer_for(layer_key)
    with warp_stream(activation.device):
        mean, _, _ = layer._warp_muscle_dynamics_from_activation_forward(
            activation=activation,
            raw_context=raw_context,
            include_passive_force=include_passive_force,
        )
    return mean


@muscle_torque_forward.register_fake
def _muscle_torque_forward_fake(
    activation: torch.Tensor,
    raw_context: torch.Tensor,
    layer_key: int,
    include_passive_force: bool,
) -> torch.Tensor:
    layer = _layer_for(layer_key)
    return torch.empty(
        (activation.shape[0], layer.torque_dim),
        dtype=torch.float32,
        device=activation.device,
    )


@torch.library.custom_op("instantmimic_muscle::torque_backward", mutates_args=())
def muscle_torque_backward(
    raw_context: torch.Tensor,
    grad_output: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    """Activation gradient of muscle_torque_forward.

    Geometry is rebuilt from the forward's own context, so the operator holds no
    stale views of persistent scratch that another forward may have rewritten.
    """
    layer = _layer_for(layer_key)
    with warp_stream(raw_context.device):
        return layer._warp_muscle_dynamics_from_activation_backward(
            raw_context=raw_context,
            grad_output=grad_output,
        )


@muscle_torque_backward.register_fake
def _muscle_torque_backward_fake(
    raw_context: torch.Tensor,
    grad_output: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    layer = _layer_for(layer_key)
    return torch.empty(
        (raw_context.shape[0], layer.muscle_count),
        dtype=torch.float32,
        device=raw_context.device,
    )


def _torque_forward_setup_context(
    ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor
) -> None:
    ctx.save_for_backward(inputs[1])
    ctx.layer_key = int(inputs[2])
    # Hold the layer for the node's lifetime: the registry is weak, and the
    # backward below must run even if the caller dropped its reference.
    ctx.layer = _layer_for(ctx.layer_key)


def _torque_forward_backward(
    ctx: Any, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, None, None]:
    (raw_context,) = ctx.saved_tensors
    grad_activation = muscle_torque_backward(raw_context, grad_output, ctx.layer_key)
    return grad_activation, None, None, None


muscle_torque_forward.register_autograd(
    _torque_forward_backward,
    setup_context=_torque_forward_setup_context,
)


@torch.library.custom_op(
    "instantmimic_muscle::torque_cached_lbs_forward", mutates_args=()
)
def muscle_torque_cached_lbs_forward(
    activation: torch.Tensor,
    raw_context: torch.Tensor,
    unit_force_magnitude: torch.Tensor,
    passive_force_magnitude: torch.Tensor,
    layer_key: int,
    include_passive_force: bool,
) -> torch.Tensor:
    """Projection whose per-muscle forces were precomputed outside this call."""
    layer = _layer_for(layer_key)
    with warp_stream(activation.device):
        return layer._warp_muscle_dynamics_from_activation_forward_cached_lbs_force(
            activation=activation,
            raw_context=raw_context,
            unit_force_magnitude=unit_force_magnitude,
            passive_force_magnitude=passive_force_magnitude,
            include_passive_force=include_passive_force,
        )


@muscle_torque_cached_lbs_forward.register_fake
def _muscle_torque_cached_lbs_forward_fake(
    activation: torch.Tensor,
    raw_context: torch.Tensor,
    unit_force_magnitude: torch.Tensor,
    passive_force_magnitude: torch.Tensor,
    layer_key: int,
    include_passive_force: bool,
) -> torch.Tensor:
    layer = _layer_for(layer_key)
    return torch.empty(
        (activation.shape[0], layer.torque_dim),
        dtype=torch.float32,
        device=activation.device,
    )


@torch.library.custom_op(
    "instantmimic_muscle::torque_cached_lbs_backward", mutates_args=()
)
def muscle_torque_cached_lbs_backward(
    raw_context: torch.Tensor,
    unit_force_magnitude: torch.Tensor,
    grad_output: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    """Activation gradient of muscle_torque_cached_lbs_forward."""
    layer = _layer_for(layer_key)
    with warp_stream(raw_context.device):
        return layer._warp_muscle_dynamics_from_activation_backward(
            raw_context=raw_context,
            grad_output=grad_output,
            unit_force_magnitude=unit_force_magnitude,
        )


@muscle_torque_cached_lbs_backward.register_fake
def _muscle_torque_cached_lbs_backward_fake(
    raw_context: torch.Tensor,
    unit_force_magnitude: torch.Tensor,
    grad_output: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    layer = _layer_for(layer_key)
    return torch.empty(
        (raw_context.shape[0], layer.muscle_count),
        dtype=torch.float32,
        device=raw_context.device,
    )


def _cached_lbs_forward_setup_context(
    ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor
) -> None:
    ctx.save_for_backward(inputs[1], inputs[2])
    ctx.layer_key = int(inputs[4])
    ctx.layer = _layer_for(ctx.layer_key)


def _cached_lbs_forward_backward(
    ctx: Any, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, None, None, None, None]:
    raw_context, unit_force_magnitude = ctx.saved_tensors
    grad_activation = muscle_torque_cached_lbs_backward(
        raw_context, unit_force_magnitude, grad_output, ctx.layer_key
    )
    return grad_activation, None, None, None, None, None


muscle_torque_cached_lbs_forward.register_autograd(
    _cached_lbs_forward_backward,
    setup_context=_cached_lbs_forward_setup_context,
)


@torch.library.custom_op("instantmimic_muscle::torque_bounds", mutates_args=())
def muscle_torque_bounds(
    raw_context: torch.Tensor,
    layer_key: int,
    select_active: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feasible torque box at the current geometry.

    The bounds are data, not differentiable outputs; the operator returns owned
    copies so a later forward cannot rewrite a live result through the shared
    bounds scratch.
    """
    layer = _layer_for(layer_key)
    with warp_stream(raw_context.device):
        lower, upper = layer._compute_activation_torque_bounds_warp(
            raw_context,
            select_active=select_active,
        )
    return lower.clone(), upper.clone()


@muscle_torque_bounds.register_fake
def _muscle_torque_bounds_fake(
    raw_context: torch.Tensor,
    layer_key: int,
    select_active: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = _layer_for(layer_key)
    width = layer.active_output_dim if select_active else layer.torque_dim
    lower = torch.empty(
        (raw_context.shape[0], width),
        dtype=torch.float32,
        device=raw_context.device,
    )
    return lower, torch.empty_like(lower)


__all__ = [
    "muscle_torque_backward",
    "muscle_torque_bounds",
    "muscle_torque_cached_lbs_backward",
    "muscle_torque_cached_lbs_forward",
    "muscle_torque_forward",
    "register_warp_op_layer",
    "warp_op_key",
    "warp_stream",
    "warp_stream_for_handle",
    "warp_stream_from_torch",
]
