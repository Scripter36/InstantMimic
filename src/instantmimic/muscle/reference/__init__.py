"""Readable Torch statements of the muscle mechanics the Warp kernels compute.

The layer runs these on CPU and the Warp kernels on CUDA, so they are not a
fallback that may drift: `tests/test_passive_force_models.py` holds the two
implementations of each Hill curve against one another.
"""

from .force_curves import active_force_length_torch, passive_force_length_torch
from .projection import (
    project_body_wrench_to_torque,
    segment_force_magnitude_to_body_wrench,
)
from .quaternion import quat_normalize, quat_rotate

__all__ = (
    "active_force_length_torch",
    "passive_force_length_torch",
    "project_body_wrench_to_torque",
    "quat_normalize",
    "quat_rotate",
    "segment_force_magnitude_to_body_wrench",
)
