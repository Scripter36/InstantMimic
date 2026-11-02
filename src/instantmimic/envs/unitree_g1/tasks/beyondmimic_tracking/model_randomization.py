"""Per-world MuJoCo-Warp model expansion for official startup randomization."""

from __future__ import annotations

from typing import Any

import mujoco_warp as mjw
import warp as wp

# The field-expansion structure follows mjlab commit
# 15ebce8840ee2205f4c62d0c20df65dd1794cab4 (Apache-2.0).

_SET_CONST_DERIVED_FIELDS = (
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
)


# `module="unique"` because the kernel is generic over `dtype`: each concrete
# instantiation would otherwise be added to this module and invalidate the
# other kernels compiled into it.
@wp.kernel(module="unique", enable_backward=False)
def _repeat_array_kernel(
    source: wp.array(dtype=Any),
    elements_per_world: int,
    destination: wp.array(dtype=Any),
) -> None:
    index = wp.tid()
    destination[index] = source[index % elements_per_world]


def expand_model_fields(
    model: mjw.Model,
    *,
    num_worlds: int,
    fields: tuple[str, ...],
) -> None:
    """Tile selected first-axis singleton fields to one row per world."""

    if num_worlds == 1:
        return
    for field in fields:
        if not hasattr(model, field):
            raise ValueError(f"MuJoCo-Warp model has no field {field!r}")
        source = getattr(model, field)
        if source.shape[0] != 1:
            if source.shape[0] != num_worlds:
                raise ValueError(
                    f"{field} has incompatible leading dimension {source.shape[0]}"
                )
            continue
        shape = list(source.shape)
        shape[0] = num_worlds
        array_type = {
            1: wp.array,
            2: wp.array2d,
            3: wp.array3d,
            4: wp.array4d,
        }.get(len(shape))
        if array_type is None:
            raise ValueError(f"Cannot expand {field} with shape {source.shape}")
        destination = array_type(shape=shape, dtype=source.dtype, device=source.device)
        source_flat = source.flatten()
        destination_flat = destination.flatten()
        elements_per_world = destination_flat.shape[0] // num_worlds
        wp.launch(
            _repeat_array_kernel,
            dim=destination_flat.shape[0],
            inputs=[source_flat, elements_per_world],
            outputs=[destination_flat],
            device=source.device,
        )
        setattr(model, field, destination)


def expand_startup_randomization_fields(model: mjw.Model, *, num_worlds: int) -> None:
    expand_model_fields(
        model,
        num_worlds=num_worlds,
        fields=("geom_friction", "body_ipos", *_SET_CONST_DERIVED_FIELDS),
    )
