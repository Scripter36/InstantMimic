"""Shared joint-state observation and reward mode resolution."""

from __future__ import annotations


def resolve_joint_velocity_mode(mode: str) -> int:
    mode_key = str(mode).lower()
    if mode_key == "qvel":
        return 0
    if mode_key == "hinge_local_angvel":
        return 1
    if mode_key == "multi_hinge_local_angvel":
        return 2
    raise ValueError(
        "Unsupported joint velocity mode "
        f"{mode!r}. Valid values: hinge_local_angvel, multi_hinge_local_angvel, qvel."
    )


def resolve_joint_pose_mode(mode: str, *, allow_6d: bool = False) -> int:
    mode_key = str(mode).lower()
    if mode_key in ("default", "qpos"):
        return 0
    if allow_6d and mode_key == "6d":
        return 2
    if mode_key == "multi_hinge_local_quat":
        return 1
    valid = "default, multi_hinge_local_quat, qpos"
    if allow_6d:
        valid += ", 6d"
    raise ValueError(f"Unsupported joint pose mode {mode!r}. Valid values: {valid}.")


__all__ = ("resolve_joint_pose_mode", "resolve_joint_velocity_mode")
