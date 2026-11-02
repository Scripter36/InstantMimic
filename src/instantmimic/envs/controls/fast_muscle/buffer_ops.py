"""Native device operations for persistent controller buffers."""

from typing import Any

import torch
import warp as wp


@wp.kernel
def copy_2d_kernel(source: wp.array2d(dtype=Any), target: wp.array2d(dtype=Any)):
    i, j = wp.tid()
    target[i, j] = type(target[i, j])(source[i, j])


@wp.kernel
def masked_zero_2d_kernel(values: wp.array2d(dtype=Any), mask: wp.array(dtype=wp.bool)):
    i, j = wp.tid()
    if mask[i]:
        values[i, j] = type(values[i, j])(0)


def copy_2d_(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.shape != source.shape or target.ndim != 2:
        raise ValueError("Controller copy requires matching two-dimensional buffers")
    if not target.is_cuda:
        target.copy_(source)
        return
    if target.numel():
        wp.launch(
            copy_2d_kernel,
            dim=tuple(target.shape),
            inputs=[wp.from_torch(source), wp.from_torch(target)],
            device=str(target.device),
        )


def masked_zero_2d_(values: torch.Tensor, mask: torch.Tensor) -> None:
    if not values.is_cuda:
        values.masked_fill_(mask.unsqueeze(-1), 0.0)
        return
    if values.numel():
        wp.launch(
            masked_zero_2d_kernel,
            dim=tuple(values.shape),
            inputs=[wp.from_torch(values), wp.from_torch(mask)],
            device=str(values.device),
        )


def zero_(values: torch.Tensor) -> None:
    if not values.is_cuda:
        values.zero_()
        return
    if values.numel():
        wp.from_torch(values).zero_()
