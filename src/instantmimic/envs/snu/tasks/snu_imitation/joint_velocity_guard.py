from __future__ import annotations

import math
from typing import Any

import warp as wp

__all__ = ("apply_substep", "initialize")


@wp.kernel(enable_backward=False)
def _clamp_joint_velocity_kernel(
    joint_dof_adr: wp.array(dtype=wp.int32),
    max_joint_velocity: float,
    qvel: wp.array2d(dtype=float),
):
    env_idx, joint_dof_idx = wp.tid()
    qvel_idx = joint_dof_adr[joint_dof_idx]
    velocity = qvel[env_idx, qvel_idx]
    if velocity > max_joint_velocity:
        qvel[env_idx, qvel_idx] = max_joint_velocity
    elif velocity < -max_joint_velocity:
        qvel[env_idx, qvel_idx] = -max_joint_velocity


def initialize(env: Any) -> None:
    max_joint_velocity = float(getattr(env.cfg, "max_joint_velocity", -1.0))
    if not math.isfinite(max_joint_velocity) or (
        max_joint_velocity != -1.0 and max_joint_velocity <= 0.0
    ):
        raise ValueError(
            "env.max_joint_velocity must be -1 (disabled) or a finite positive value"
        )

    env._max_joint_velocity = max_joint_velocity
    env._joint_velocity_guard_enabled = max_joint_velocity > 0.0
    if not env._joint_velocity_guard_enabled:
        return

    # The character's joint DOFs (instantmimic.envs.dof_roles): the root and
    # passive DOFs keep their free velocity.
    joint_dof_adr = env.joint_dof_adr_i32
    if int(joint_dof_adr.numel()) <= 0:
        raise ValueError(
            "max_joint_velocity requires at least one non-root joint velocity"
        )
    env._joint_velocity_guard_dof_adr = joint_dof_adr
    env._joint_velocity_guard_joint_dof_count = int(joint_dof_adr.numel())


def apply_substep(env: Any) -> None:
    if not bool(getattr(env, "_joint_velocity_guard_enabled", False)):
        return
    wp.launch(
        _clamp_joint_velocity_kernel,
        dim=(env.num_envs, env._joint_velocity_guard_joint_dof_count),
        inputs=[
            env._joint_velocity_guard_dof_adr,
            env._max_joint_velocity,
        ],
        outputs=[env.mjw_data.qvel],
    )
