"""Warp feature writers shared by tracking-style environments."""

from __future__ import annotations

import warp as wp

from instantmimic.motion.warp_math import normalize_mjw_quat
from instantmimic.utils.mjw_kernels.private import (
    mul_quat,
    quat_inv,
    quat_sub,
    quat_to_mat,
    rot_vec_quat,
)


@wp.func
def local_mjw_quat(parent_q: wp.quat, child_q: wp.quat) -> wp.quat:
    return normalize_mjw_quat(
        mul_quat(
            quat_inv(normalize_mjw_quat(parent_q)),
            normalize_mjw_quat(child_q),
        )
    )


@wp.func
def safe_quat_sub_mjw(q0: wp.quat, q1: wp.quat) -> wp.vec3:
    return quat_sub(normalize_mjw_quat(q0), normalize_mjw_quat(q1))


@wp.func
def write_ortho6d(
    obs: wp.array2d(dtype=float),
    env_idx: int,
    offset: int,
    q: wp.quat,
):
    """Write ortho6d (first two rotation-matrix columns) to a flat buffer."""
    q = normalize_mjw_quat(q)
    mat = quat_to_mat(q)
    obs[env_idx, offset + 0] = mat[0, 0]
    obs[env_idx, offset + 1] = mat[0, 1]
    obs[env_idx, offset + 2] = mat[1, 0]
    obs[env_idx, offset + 3] = mat[1, 1]
    obs[env_idx, offset + 4] = mat[2, 0]
    obs[env_idx, offset + 5] = mat[2, 1]


@wp.func
def ortho6d_component(q: wp.quat, component: int) -> float:
    """Return one ortho6d component from a scalar-first MuJoCo quaternion."""
    q = normalize_mjw_quat(q)
    mat = quat_to_mat(q)
    if component == 0:
        return mat[0, 0]
    if component == 1:
        return mat[0, 1]
    if component == 2:
        return mat[1, 0]
    if component == 3:
        return mat[1, 1]
    if component == 4:
        return mat[2, 0]
    return mat[2, 1]


@wp.func
def write_tan_norm(
    obs: wp.array2d(dtype=float),
    env_idx: int,
    offset: int,
    q: wp.quat,
):
    """Write PHC tangent/normal rotation features to a flat buffer."""
    q = normalize_mjw_quat(q)
    tangent = rot_vec_quat(wp.vec3(1.0, 0.0, 0.0), q)
    normal = rot_vec_quat(wp.vec3(0.0, 0.0, 1.0), q)
    obs[env_idx, offset + 0] = tangent[0]
    obs[env_idx, offset + 1] = tangent[1]
    obs[env_idx, offset + 2] = tangent[2]
    obs[env_idx, offset + 3] = normal[0]
    obs[env_idx, offset + 4] = normal[1]
    obs[env_idx, offset + 5] = normal[2]


@wp.func
def write_vec3(
    obs: wp.array2d(dtype=float),
    env_idx: int,
    offset: int,
    value: wp.vec3,
    scale: float,
):
    """Write a scaled vec3 to a flat feature buffer."""
    obs[env_idx, offset + 0] = value[0] * scale
    obs[env_idx, offset + 1] = value[1] * scale
    obs[env_idx, offset + 2] = value[2] * scale


@wp.func
def write_quat4(
    array: wp.array2d(dtype=float),
    env_idx: int,
    offset: int,
    quat: wp.quat,
):
    """Write a quaternion to a flat feature buffer component by component."""
    array[env_idx, offset + 0] = quat[0]
    array[env_idx, offset + 1] = quat[1]
    array[env_idx, offset + 2] = quat[2]
    array[env_idx, offset + 3] = quat[3]


@wp.func
def read_quat_from_2d(array: wp.array2d(dtype=float), i: int, j: int) -> wp.quat:
    """Read four consecutive scalars as a scalar-first MuJoCo quaternion."""
    return wp.quat(array[i, j], array[i, j + 1], array[i, j + 2], array[i, j + 3])


@wp.func
def read_vec3_from_2d(array: wp.array2d(dtype=float), i: int, j: int) -> wp.vec3:
    return wp.vec3(array[i, j], array[i, j + 1], array[i, j + 2])


@wp.func
def read_body_state_pos(
    body_state: wp.array3d(dtype=float), frame_idx: int, body_idx: int
) -> wp.vec3:
    return wp.vec3(
        body_state[frame_idx, body_idx, 0],
        body_state[frame_idx, body_idx, 1],
        body_state[frame_idx, body_idx, 2],
    )


@wp.func
def read_body_state_quat(
    body_state: wp.array3d(dtype=float), frame_idx: int, body_idx: int
) -> wp.quat:
    return wp.quat(
        body_state[frame_idx, body_idx, 3],
        body_state[frame_idx, body_idx, 4],
        body_state[frame_idx, body_idx, 5],
        body_state[frame_idx, body_idx, 6],
    )


@wp.func
def read_body_state_linvel(
    body_state: wp.array3d(dtype=float), frame_idx: int, body_idx: int
) -> wp.vec3:
    return wp.vec3(
        body_state[frame_idx, body_idx, 7],
        body_state[frame_idx, body_idx, 8],
        body_state[frame_idx, body_idx, 9],
    )


@wp.func
def read_body_state_angvel(
    body_state: wp.array3d(dtype=float), frame_idx: int, body_idx: int
) -> wp.vec3:
    return wp.vec3(
        body_state[frame_idx, body_idx, 10],
        body_state[frame_idx, body_idx, 11],
        body_state[frame_idx, body_idx, 12],
    )


__all__ = (
    "local_mjw_quat",
    "ortho6d_component",
    "read_body_state_angvel",
    "read_body_state_linvel",
    "read_body_state_pos",
    "read_body_state_quat",
    "read_quat_from_2d",
    "read_vec3_from_2d",
    "safe_quat_sub_mjw",
    "write_ortho6d",
    "write_quat4",
    "write_tan_norm",
    "write_vec3",
)
