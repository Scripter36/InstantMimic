"""Shared startup and runtime utilities for motion imitation environments."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "MotionClipMetadata": ".timeline",
    "MujocoReferencePayload": ".mujoco_reference",
    "build_clip_offsets": ".timeline",
    "build_motion_clip_metadata": ".timeline",
    "materialize_mujoco_reference_payload": ".mujoco_reference",
    "resolve_clip_reference_frames": ".timeline",
}

__all__ = (
    "MotionClipMetadata",
    "MujocoReferencePayload",
    "build_clip_offsets",
    "build_motion_clip_metadata",
    "materialize_mujoco_reference_payload",
    "resolve_clip_reference_frames",
)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
