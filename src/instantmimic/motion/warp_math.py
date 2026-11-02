"""Warp quaternion helpers for MuJoCo scalar-first data.

Warp's native quaternion convention is ``xyzw``. MuJoCo and the pinned
``mujoco_warp`` math layer store ``wxyz`` values in the same ``wp.quat``
container and operate on index 0 as the scalar component. The ``mjw`` helpers
below follow that scalar-first contract and deliberately avoid Warp's native
quaternion operations. The ``vec4`` helpers serve scalar-first buffers that do
not use a quaternion container.
"""

from __future__ import annotations

import warp as wp

from instantmimic.utils.mjw_kernels.private import axis_angle_to_quat, rot_vec_quat


@wp.func
def normalize_mjw_quat(q: wp.quat) -> wp.quat:
    """Normalize scalar-first MuJoCo data with identity/sign canonicalization."""
    norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    if norm_sq <= 1.0e-12:
        return wp.quat(1.0, 0.0, 0.0, 0.0)
    inv_norm = 1.0 / wp.sqrt(norm_sq)
    out = wp.quat(
        q[0] * inv_norm,
        q[1] * inv_norm,
        q[2] * inv_norm,
        q[3] * inv_norm,
    )
    if out[0] < 0.0:
        return wp.quat(-out[0], -out[1], -out[2], -out[3])
    return out


@wp.func
def rotate_mjw_quat(q: wp.quat, value: wp.vec3) -> wp.vec3:
    """Apply a scalar-first MuJoCo quaternion without normalizing it."""
    xyz = wp.vec3(q[1], q[2], q[3])
    twice_cross = 2.0 * wp.cross(xyz, value)
    return value + q[0] * twice_cross + wp.cross(xyz, twice_cross)


@wp.func
def rotate_inverse_unit_mjw_quat(q: wp.quat, value: wp.vec3) -> wp.vec3:
    """Inverse-rotate with the expanded unit-quaternion formula."""
    xyz = wp.vec3(q[1], q[2], q[3])
    a = value * (2.0 * q[0] * q[0] - 1.0)
    b = wp.cross(xyz, value) * q[0] * 2.0
    c = xyz * (wp.dot(xyz, value) * 2.0)
    return a - b + c


@wp.func
def heading_yaw_mjw_quat(root_quat: wp.quat) -> wp.quat:
    """Extract a scalar-first yaw quaternion from MuJoCo orientation data."""
    root_quat = normalize_mjw_quat(root_quat)
    root_forward = rot_vec_quat(wp.vec3(1.0, 0.0, 0.0), root_quat)
    yaw = wp.atan2(root_forward[1], root_forward[0])
    return axis_angle_to_quat(wp.vec3(0.0, 0.0, 1.0), yaw)


@wp.func
def conjugate_wxyz_vec4(q: wp.vec4) -> wp.vec4:
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def multiply_wxyz_vec4(left: wp.vec4, right: wp.vec4) -> wp.vec4:
    w1 = left[0]
    x1 = left[1]
    y1 = left[2]
    z1 = left[3]
    w2 = right[0]
    x2 = right[1]
    y2 = right[2]
    z2 = right[3]
    return wp.vec4(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


@wp.func
def inverse_wxyz_vec4(q: wp.vec4) -> wp.vec4:
    return conjugate_wxyz_vec4(q) / wp.dot(q, q)


@wp.func
def rotate_wxyz_vec4(q: wp.vec4, value: wp.vec3) -> wp.vec3:
    xyz = wp.vec3(q[1], q[2], q[3])
    twice_cross = 2.0 * wp.cross(xyz, value)
    return value + q[0] * twice_cross + wp.cross(xyz, twice_cross)


@wp.func
def rotate_inverse_wxyz_vec4(q: wp.vec4, value: wp.vec3) -> wp.vec3:
    return rotate_wxyz_vec4(conjugate_wxyz_vec4(q), value)


__all__ = (
    "conjugate_wxyz_vec4",
    "heading_yaw_mjw_quat",
    "inverse_wxyz_vec4",
    "multiply_wxyz_vec4",
    "normalize_mjw_quat",
    "rotate_inverse_unit_mjw_quat",
    "rotate_inverse_wxyz_vec4",
    "rotate_mjw_quat",
    "rotate_wxyz_vec4",
)
