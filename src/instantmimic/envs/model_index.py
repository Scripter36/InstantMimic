"""MuJoCo model-name and joint-coordinate index resolution."""

from __future__ import annotations

import mujoco


def _object_type_name(objtype: mujoco.mjtObj | int) -> str:
    try:
        enum_name = mujoco.mjtObj(int(objtype)).name
    except ValueError:
        return f"object type {int(objtype)}"
    return enum_name.removeprefix("mjOBJ_").lower()


def resolve_id(
    model: mujoco.MjModel,
    objtype: mujoco.mjtObj | int,
    name: str,
) -> int:
    """Resolve a named MuJoCo object or raise a consistent missing-name error."""
    object_id = int(mujoco.mj_name2id(model, objtype, name))
    if object_id < 0:
        raise ValueError(
            f"MuJoCo {_object_type_name(objtype)} {name!r} was not found in the model."
        )
    return object_id


def qpos_index(model: mujoco.MjModel, joint_name: str) -> int:
    """Return the first qpos coordinate index for a named joint."""
    joint_id = resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[joint_id])


def dof_index(model: mujoco.MjModel, joint_name: str) -> int:
    """Return the first velocity/DoF coordinate index for a named joint."""
    joint_id = resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_dofadr[joint_id])
