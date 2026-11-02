from __future__ import annotations

import warp as wp

from instantmimic.motion.warp_math import normalize_mjw_quat as normalize_quat
from instantmimic.utils.mjw_kernels.private import mul_quat, rot_vec_quat


@wp.func
def loop_local_frame(frame_idx: int, motion_frame_count: int) -> int:
    return frame_idx % motion_frame_count


@wp.func
def loop_cycle_count(frame_idx: int, motion_frame_count: int) -> int:
    return frame_idx // motion_frame_count


@wp.func
def clip_local_frame(
    frame_idx: int,
    clip_idx: int,
    clip_length: wp.array(dtype=int),
) -> int:
    return loop_local_frame(frame_idx, clip_length[clip_idx])


@wp.func
def clip_cycle_count(
    frame_idx: int,
    clip_idx: int,
    clip_length: wp.array(dtype=int),
) -> int:
    return loop_cycle_count(frame_idx, clip_length[clip_idx])


@wp.func
def clip_global_frame(
    frame_idx: int,
    clip_idx: int,
    clip_start: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
) -> int:
    return clip_start[clip_idx] + clip_local_frame(frame_idx, clip_idx, clip_length)


@wp.func
def apply_loop_transform_vec(
    v: wp.vec3,
    loop_root_q: wp.quat,
    loop_root_t: wp.vec3,
    cycles: int,
) -> wp.vec3:
    out = v
    c = int(0)
    while c < cycles:
        out = rot_vec_quat(out, loop_root_q) + loop_root_t
        c = c + 1
    return out


@wp.func
def apply_loop_rotation_vec(v: wp.vec3, loop_root_q: wp.quat, cycles: int) -> wp.vec3:
    out = v
    c = int(0)
    while c < cycles:
        out = rot_vec_quat(out, loop_root_q)
        c = c + 1
    return out


@wp.func
def apply_loop_transform_quat(q: wp.quat, loop_root_q: wp.quat, cycles: int) -> wp.quat:
    """Compose the loop rotation `cycles` times, renormalizing as it goes.

    There were two of these, and they disagreed: the tracking-metrics copy
    normalized the input and every product, this one normalized nothing. Chained
    `mul_quat` accumulates error, so after 4096 cycles the un-normalized result
    had |q| off by 1.2e-4 and a rotation differing by about 0.04 degrees --
    small, but `rot_vec_quat` assumes a unit quaternion, so the drift is a
    latent wrongness rather than a rounding preference. The tracking definition
    wins and lives here, including its deterministic non-negative scalar sign.
    """

    out = normalize_quat(q)
    c = int(0)
    while c < cycles:
        out = normalize_quat(mul_quat(loop_root_q, out))
        c = c + 1
    return out


__all__ = (
    "apply_loop_rotation_vec",
    "apply_loop_transform_quat",
    "apply_loop_transform_vec",
    "clip_cycle_count",
    "clip_global_frame",
    "clip_local_frame",
    "loop_cycle_count",
    "loop_local_frame",
)
