from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import warp as wp

_KernelCallable = TypeVar("_KernelCallable", bound=Callable[..., object])


def expose_kernel_name(func: _KernelCallable) -> _KernelCallable:
    """Expose a nested kernel's function name without its factory qualname."""
    func.__qualname__ = func.__name__
    return func


@wp.kernel(enable_backward=False)
def clear_1d_float32_kernel(values: wp.array(dtype=wp.float32)):
    i = wp.tid()
    values[i] = 0.0


@wp.kernel(enable_backward=False)
def copy_1d_int64_kernel(
    src: wp.array(dtype=wp.int64),
    dst: wp.array(dtype=wp.int64),
):
    i = wp.tid()
    dst[i] = src[i]
