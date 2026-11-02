"""Translate configured space specifications into Gymnasium spaces and tensors.

Environment configs declare `observation_space`, `state_space` and `action_space`
as plain Python values so that most tasks never import Gymnasium. This module is
the single place that turns those declarations into `gymnasium.spaces` objects
and that allocates matching torch tensors on the simulation device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch

if TYPE_CHECKING:
    # Importing this at runtime would close a cycle: instantmimic.envs.common
    # pulls in instantmimic.envs, whose __init__ imports DirectRLEnv, which
    # imports this module. SpaceType is only ever an annotation here.
    from instantmimic.envs.common import SpaceType


def spec_to_gym_space(spec: SpaceType) -> gym.spaces.Space:
    """Return the Gymnasium space a config-level space specification denotes.

    Accepted forms, in the order they are tested:

    - an already-built `gymnasium.spaces.Space`, returned unchanged
    - `int` -> unbounded `Box` of that width
    - `list[int]` -> unbounded `Box` of that shape
    - one-element `set` -> `Discrete` over that many values
    - `list` of one-element `set` -> `MultiDiscrete`
    - `tuple` -> `Tuple`, each entry resolved recursively
    - `dict` -> `Dict`, each value resolved recursively

    Raises:
        ValueError: The specification matches none of the accepted forms.
    """

    if isinstance(spec, gym.spaces.Space):
        return spec
    if isinstance(spec, int):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(spec,))
    if isinstance(spec, set):
        if len(spec) == 1:
            return gym.spaces.Discrete(n=next(iter(spec)))
        raise ValueError(
            f"A set space specification must hold exactly one value, got {spec!r}."
        )
    if isinstance(spec, list):
        if all(isinstance(entry, int) for entry in spec):
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=spec)
        if all(isinstance(entry, set) and len(entry) == 1 for entry in spec):
            return gym.spaces.MultiDiscrete(nvec=[next(iter(entry)) for entry in spec])
        raise ValueError(
            "A list space specification must hold either all integers or all "
            f"one-element sets, got {spec!r}."
        )
    if isinstance(spec, tuple):
        return gym.spaces.Tuple([spec_to_gym_space(entry) for entry in spec])
    if isinstance(spec, dict):
        return gym.spaces.Dict(
            {key: spec_to_gym_space(value) for key, value in spec.items()}
        )
    raise ValueError(f"Unsupported space specification: {spec!r}")


def sample_space(
    space: gym.spaces.Space,
    device: str,
    batch_size: int = -1,
    fill_value: float | None = None,
) -> Any:
    """Allocate torch tensors shaped like `space`, on `device`.

    Composite spaces keep their container: `Dict` yields a dict and `Tuple`
    yields a tuple, each leaf being a tensor whose first dimension is the batch.

    Args:
        space: The space to mirror. Leaves must be `Box`, `Discrete` or
            `MultiDiscrete`.
        device: Torch device the tensors are created on.
        batch_size: Leading batch dimension. Values above zero draw from the
            batched space; anything else leaves the dimension inferred.
        fill_value: When given, every leaf is overwritten with this value
            instead of keeping the drawn sample.

    Raises:
        ValueError: A leaf of `space` is not a supported space type.
    """

    source = (
        gym.vector.utils.batch_space(space, batch_size) if batch_size > 0 else space
    )
    return _tensorize(space, source.sample(), device, batch_size, fill_value)


def _leaf_dtype_and_shape(
    space: gym.spaces.Space,
) -> tuple[torch.dtype, tuple[int, ...]]:
    """Return the torch dtype and per-entry shape a leaf space maps onto.

    Subclasses are matched, mirroring how the composite branches in `_tensorize`
    test their containers.
    """

    if isinstance(space, gym.spaces.Box):
        return torch.float32, tuple(space.shape)
    # Discrete carries no shape of its own; it occupies one column per entry.
    if isinstance(space, gym.spaces.Discrete):
        return torch.int64, (1,)
    if isinstance(space, gym.spaces.MultiDiscrete):
        return torch.int64, tuple(space.shape)
    raise ValueError(f"Unsupported space type for tensor allocation: {space!r}")


def _tensorize(
    space: gym.spaces.Space,
    sample: Any,
    device: str,
    batch_size: int,
    fill_value: float | None,
) -> Any:
    if isinstance(space, gym.spaces.Dict):
        return {
            key: _tensorize(subspace, sample[key], device, batch_size, fill_value)
            for key, subspace in space.items()
        }
    if isinstance(space, gym.spaces.Tuple):
        return tuple(
            _tensorize(subspace, value, device, batch_size, fill_value)
            for subspace, value in zip(space, sample, strict=False)
        )

    dtype, leaf_shape = _leaf_dtype_and_shape(space)
    # torch.tensor rather than as_tensor: the returned tensor owns its storage
    # instead of aliasing the numpy buffer gymnasium handed back.
    tensor = torch.tensor(sample, dtype=dtype, device=device).reshape(
        batch_size, *leaf_shape
    )
    if fill_value is not None:
        tensor.fill_(int(fill_value) if not dtype.is_floating_point else fill_value)
    return tensor
