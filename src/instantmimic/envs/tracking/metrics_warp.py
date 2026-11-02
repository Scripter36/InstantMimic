from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from instantmimic.envs.tracking import metrics as tracking_metrics
from instantmimic.motion.warp_math import heading_yaw_mjw_quat as _heading_yaw_quat
from instantmimic.motion.warp_math import normalize_mjw_quat as _normalize_quat_wp
from instantmimic.motion.warp_utils import (
    apply_loop_rotation_vec,
    apply_loop_transform_quat,
    apply_loop_transform_vec,
)
from instantmimic.utils.mjw_kernels.private import (
    quat_inv,
    rot_vec_quat,
)

FRAME_WORLD = 0
FRAME_TRANSLATION = 1
FRAME_HEADING = 2
FRAME_ROOT = 3


@dataclass
class MetricBuffers:
    mean_error: torch.Tensor
    per_body_error: torch.Tensor


@dataclass
class ReferenceTrackingMetricWorkspace:
    sim_ape_body_pos: torch.Tensor
    sim_ee_body_pos: torch.Tensor
    ref_ape_body_pos: torch.Tensor
    ref_ee_body_pos: torch.Tensor
    sim_tracking_body_linvel: torch.Tensor
    ref_tracking_body_linvel: torch.Tensor
    sim_root_pos: torch.Tensor
    ref_root_pos: torch.Tensor
    sim_root_quat: torch.Tensor
    ref_root_quat: torch.Tensor
    ape: MetricBuffers
    global_mpjpe: MetricBuffers
    ee: MetricBuffers
    ref_frame_idx_i32: torch.Tensor | None = None
    motion_clip_idx_i32: torch.Tensor | None = None
    clip_start_i32: torch.Tensor | None = None
    clip_length_i32: torch.Tensor | None = None


@wp.func
def _vec3_length(v: wp.vec3) -> float:
    return wp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@wp.func
def _position_in_frame(
    body_pos: wp.vec3,
    root_pos: wp.vec3,
    root_quat: wp.quat,
    frame_mode: int,
) -> wp.vec3:
    if frame_mode == FRAME_WORLD:
        return body_pos

    local_pos = body_pos - root_pos
    if frame_mode == FRAME_TRANSLATION:
        return local_pos

    frame_quat = _normalize_quat_wp(root_quat)
    if frame_mode == FRAME_HEADING:
        frame_quat = _heading_yaw_quat(frame_quat)
    root_inv = quat_inv(frame_quat)
    return rot_vec_quat(local_pos, root_inv)


@wp.func
def _vector_in_frame(
    vec: wp.vec3,
    root_quat: wp.quat,
    frame_mode: int,
) -> wp.vec3:
    if frame_mode == FRAME_WORLD or frame_mode == FRAME_TRANSLATION:
        return vec

    frame_quat = _normalize_quat_wp(root_quat)
    if frame_mode == FRAME_HEADING:
        frame_quat = _heading_yaw_quat(frame_quat)
    root_inv = quat_inv(frame_quat)
    return rot_vec_quat(vec, root_inv)


# One definition, in `motion/warp_utils.py`. These were duplicated here and the
# quaternion pair disagreed on normalization; see that module for which won and
# why.
_apply_loop_rotation_vec = apply_loop_rotation_vec
_apply_loop_transform_vec = apply_loop_transform_vec
_apply_loop_transform_quat = apply_loop_transform_quat


@wp.kernel(enable_backward=False)
def _position_metric_kernel(
    sim_body_pos: wp.array2d(dtype=wp.vec3),
    ref_body_pos: wp.array2d(dtype=wp.vec3),
    sim_root_pos: wp.array(dtype=wp.vec3),
    ref_root_pos: wp.array(dtype=wp.vec3),
    sim_root_quat: wp.array(dtype=wp.quat),
    ref_root_quat: wp.array(dtype=wp.quat),
    frame_mode: int,
    mean_error: wp.array(dtype=float),
    per_body_error: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    body_count = int(per_body_error.shape[1])
    error_sum = float(0.0)
    body_idx = int(0)
    while body_idx < body_count:
        sim_cmp = _position_in_frame(
            sim_body_pos[env_idx, body_idx],
            sim_root_pos[env_idx],
            sim_root_quat[env_idx],
            frame_mode,
        )
        ref_cmp = _position_in_frame(
            ref_body_pos[env_idx, body_idx],
            ref_root_pos[env_idx],
            ref_root_quat[env_idx],
            frame_mode,
        )
        err = _vec3_length(sim_cmp - ref_cmp)
        per_body_error[env_idx, body_idx] = err
        error_sum = error_sum + err
        body_idx = body_idx + 1

    if body_count == 0:
        mean_error[env_idx] = 0.0
    else:
        mean_error[env_idx] = error_sum / float(body_count)


@wp.kernel(enable_backward=False)
def _velocity_metric_kernel(
    sim_body_vel: wp.array2d(dtype=wp.vec3),
    ref_body_vel: wp.array2d(dtype=wp.vec3),
    sim_root_quat: wp.array(dtype=wp.quat),
    ref_root_quat: wp.array(dtype=wp.quat),
    frame_mode: int,
    mean_error: wp.array(dtype=float),
    per_body_error: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    body_count = int(per_body_error.shape[1])
    error_sum = float(0.0)
    body_idx = int(0)
    while body_idx < body_count:
        sim_cmp = _vector_in_frame(
            sim_body_vel[env_idx, body_idx],
            sim_root_quat[env_idx],
            frame_mode,
        )
        ref_cmp = _vector_in_frame(
            ref_body_vel[env_idx, body_idx],
            ref_root_quat[env_idx],
            frame_mode,
        )
        err = _vec3_length(sim_cmp - ref_cmp)
        per_body_error[env_idx, body_idx] = err
        error_sum = error_sum + err
        body_idx = body_idx + 1

    if body_count == 0:
        mean_error[env_idx] = 0.0
    else:
        mean_error[env_idx] = error_sum / float(body_count)


@wp.kernel(enable_backward=False)
def _acceleration_metric_kernel(
    sim_body_vel: wp.array2d(dtype=wp.vec3),
    ref_body_vel: wp.array2d(dtype=wp.vec3),
    prev_sim_body_vel: wp.array2d(dtype=wp.vec3),
    prev_ref_body_vel: wp.array2d(dtype=wp.vec3),
    dt: float,
    sim_root_quat: wp.array(dtype=wp.quat),
    ref_root_quat: wp.array(dtype=wp.quat),
    prev_sim_root_quat: wp.array(dtype=wp.quat),
    prev_ref_root_quat: wp.array(dtype=wp.quat),
    frame_mode: int,
    mean_error: wp.array(dtype=float),
    per_body_error: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    body_count = int(per_body_error.shape[1])
    dt_safe = wp.max(dt, 1.0e-8)
    error_sum = float(0.0)
    body_idx = int(0)
    while body_idx < body_count:
        sim_cmp = _vector_in_frame(
            sim_body_vel[env_idx, body_idx],
            sim_root_quat[env_idx],
            frame_mode,
        )
        ref_cmp = _vector_in_frame(
            ref_body_vel[env_idx, body_idx],
            ref_root_quat[env_idx],
            frame_mode,
        )
        prev_sim_cmp = _vector_in_frame(
            prev_sim_body_vel[env_idx, body_idx],
            prev_sim_root_quat[env_idx],
            frame_mode,
        )
        prev_ref_cmp = _vector_in_frame(
            prev_ref_body_vel[env_idx, body_idx],
            prev_ref_root_quat[env_idx],
            frame_mode,
        )
        sim_acc = (sim_cmp - prev_sim_cmp) / dt_safe
        ref_acc = (ref_cmp - prev_ref_cmp) / dt_safe
        err = _vec3_length(sim_acc - ref_acc)
        per_body_error[env_idx, body_idx] = err
        error_sum = error_sum + err
        body_idx = body_idx + 1

    if body_count == 0:
        mean_error[env_idx] = 0.0
    else:
        mean_error[env_idx] = error_sum / float(body_count)


@wp.kernel(enable_backward=False)
def _gather_body_subset_vec3_kernel(
    src: wp.array2d(dtype=wp.vec3),
    body_indices: wp.array(dtype=wp.int64),
    out: wp.array2d(dtype=wp.vec3),
):
    env_idx, subset_idx = wp.tid()
    out[env_idx, subset_idx] = src[env_idx, body_indices[subset_idx]]


@wp.kernel(enable_backward=False)
def _gather_body_root_vec3_kernel(
    src: wp.array2d(dtype=wp.vec3),
    body_idx: int,
    out: wp.array(dtype=wp.vec3),
):
    env_idx = wp.tid()
    out[env_idx] = src[env_idx, body_idx]


@wp.kernel(enable_backward=False)
def _gather_body_root_quat_kernel(
    src: wp.array2d(dtype=wp.quat),
    body_idx: int,
    out: wp.array(dtype=wp.quat),
):
    env_idx = wp.tid()
    out[env_idx] = src[env_idx, body_idx]


@wp.kernel(enable_backward=False)
def _gather_reference_subset_vec3_kernel(
    src: wp.array2d(dtype=wp.vec3),
    ref_frame_idx: wp.array(dtype=int),
    motion_frame_count: int,
    body_indices: wp.array(dtype=wp.int64),
    out: wp.array2d(dtype=wp.vec3),
):
    env_idx, subset_idx = wp.tid()
    frame_idx = ref_frame_idx[env_idx] % motion_frame_count
    out[env_idx, subset_idx] = src[frame_idx, body_indices[subset_idx]]


@wp.kernel(enable_backward=False)
def _gather_reference_root_vec3_kernel(
    src: wp.array2d(dtype=wp.vec3),
    ref_frame_idx: wp.array(dtype=int),
    motion_frame_count: int,
    body_idx: int,
    out: wp.array(dtype=wp.vec3),
):
    env_idx = wp.tid()
    frame_idx = ref_frame_idx[env_idx] % motion_frame_count
    out[env_idx] = src[frame_idx, body_idx]


@wp.kernel(enable_backward=False)
def _gather_reference_root_quat_kernel(
    src: wp.array2d(dtype=wp.quat),
    ref_frame_idx: wp.array(dtype=int),
    motion_frame_count: int,
    body_idx: int,
    out: wp.array(dtype=wp.quat),
):
    env_idx = wp.tid()
    frame_idx = ref_frame_idx[env_idx] % motion_frame_count
    out[env_idx] = src[frame_idx, body_idx]


@wp.kernel(enable_backward=False)
def _gather_reference_subset_pos_loop_kernel(
    src: wp.array2d(dtype=wp.vec3),
    ref_frame_idx: wp.array(dtype=int),
    motion_frame_count: int,
    loop_root_quat: wp.array(dtype=float),
    loop_root_pos: wp.array(dtype=float),
    body_indices: wp.array(dtype=wp.int64),
    out: wp.array2d(dtype=wp.vec3),
):
    env_idx, subset_idx = wp.tid()
    local_frame_idx = ref_frame_idx[env_idx] % motion_frame_count
    cycle_count = ref_frame_idx[env_idx] // motion_frame_count
    loop_quat = _normalize_quat_wp(
        wp.quat(
            loop_root_quat[0],
            loop_root_quat[1],
            loop_root_quat[2],
            loop_root_quat[3],
        )
    )
    loop_translation = wp.vec3(loop_root_pos[0], loop_root_pos[1], loop_root_pos[2])
    out[env_idx, subset_idx] = _apply_loop_transform_vec(
        src[local_frame_idx, body_indices[subset_idx]],
        loop_quat,
        loop_translation,
        cycle_count,
    )


@wp.kernel(enable_backward=False)
def _gather_reference_subset_linvel_loop_kernel(
    src: wp.array2d(dtype=wp.vec3),
    ref_frame_idx: wp.array(dtype=int),
    motion_frame_count: int,
    loop_root_quat: wp.array(dtype=float),
    body_indices: wp.array(dtype=wp.int64),
    out: wp.array2d(dtype=wp.vec3),
):
    env_idx, subset_idx = wp.tid()
    local_frame_idx = ref_frame_idx[env_idx] % motion_frame_count
    cycle_count = ref_frame_idx[env_idx] // motion_frame_count
    loop_quat = _normalize_quat_wp(
        wp.quat(
            loop_root_quat[0],
            loop_root_quat[1],
            loop_root_quat[2],
            loop_root_quat[3],
        )
    )
    out[env_idx, subset_idx] = _apply_loop_rotation_vec(
        src[local_frame_idx, body_indices[subset_idx]],
        loop_quat,
        cycle_count,
    )


@wp.kernel(enable_backward=False)
def _gather_reference_root_pos_loop_kernel(
    src: wp.array2d(dtype=wp.vec3),
    ref_frame_idx: wp.array(dtype=int),
    motion_frame_count: int,
    loop_root_quat: wp.array(dtype=float),
    loop_root_pos: wp.array(dtype=float),
    body_idx: int,
    out: wp.array(dtype=wp.vec3),
):
    env_idx = wp.tid()
    local_frame_idx = ref_frame_idx[env_idx] % motion_frame_count
    cycle_count = ref_frame_idx[env_idx] // motion_frame_count
    loop_quat = _normalize_quat_wp(
        wp.quat(
            loop_root_quat[0],
            loop_root_quat[1],
            loop_root_quat[2],
            loop_root_quat[3],
        )
    )
    loop_translation = wp.vec3(loop_root_pos[0], loop_root_pos[1], loop_root_pos[2])
    out[env_idx] = _apply_loop_transform_vec(
        src[local_frame_idx, body_idx],
        loop_quat,
        loop_translation,
        cycle_count,
    )


@wp.kernel(enable_backward=False)
def _gather_reference_root_quat_loop_kernel(
    src: wp.array2d(dtype=wp.quat),
    ref_frame_idx: wp.array(dtype=int),
    motion_frame_count: int,
    loop_root_quat: wp.array(dtype=float),
    body_idx: int,
    out: wp.array(dtype=wp.quat),
):
    env_idx = wp.tid()
    local_frame_idx = ref_frame_idx[env_idx] % motion_frame_count
    cycle_count = ref_frame_idx[env_idx] // motion_frame_count
    loop_quat = _normalize_quat_wp(
        wp.quat(
            loop_root_quat[0],
            loop_root_quat[1],
            loop_root_quat[2],
            loop_root_quat[3],
        )
    )
    out[env_idx] = _apply_loop_transform_quat(
        src[local_frame_idx, body_idx],
        loop_quat,
        cycle_count,
    )


@wp.kernel(enable_backward=False)
def _gather_reference_subset_pos_multi_kernel(
    src: wp.array2d(dtype=wp.vec3),
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    clip_start: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
    clip_loop_root_quat: wp.array(dtype=wp.quat),
    clip_loop_root_pos: wp.array(dtype=wp.vec3),
    body_indices: wp.array(dtype=wp.int64),
    out: wp.array2d(dtype=wp.vec3),
):
    env_idx, subset_idx = wp.tid()
    clip_idx = motion_clip_idx[env_idx]
    local_frame_idx = ref_frame_idx[env_idx] % clip_length[clip_idx]
    cycle_count = ref_frame_idx[env_idx] // clip_length[clip_idx]
    frame_idx = clip_start[clip_idx] + local_frame_idx
    loop_quat = _normalize_quat_wp(clip_loop_root_quat[clip_idx])
    loop_translation = clip_loop_root_pos[clip_idx]
    out[env_idx, subset_idx] = _apply_loop_transform_vec(
        src[frame_idx, body_indices[subset_idx]],
        loop_quat,
        loop_translation,
        cycle_count,
    )


@wp.kernel(enable_backward=False)
def _gather_reference_subset_linvel_multi_kernel(
    src: wp.array2d(dtype=wp.vec3),
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    clip_start: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
    clip_loop_root_quat: wp.array(dtype=wp.quat),
    body_indices: wp.array(dtype=wp.int64),
    out: wp.array2d(dtype=wp.vec3),
):
    env_idx, subset_idx = wp.tid()
    clip_idx = motion_clip_idx[env_idx]
    local_frame_idx = ref_frame_idx[env_idx] % clip_length[clip_idx]
    cycle_count = ref_frame_idx[env_idx] // clip_length[clip_idx]
    frame_idx = clip_start[clip_idx] + local_frame_idx
    loop_quat = _normalize_quat_wp(clip_loop_root_quat[clip_idx])
    out[env_idx, subset_idx] = _apply_loop_rotation_vec(
        src[frame_idx, body_indices[subset_idx]],
        loop_quat,
        cycle_count,
    )


@wp.kernel(enable_backward=False)
def _gather_reference_root_pos_multi_kernel(
    src: wp.array2d(dtype=wp.vec3),
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    clip_start: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
    clip_loop_root_quat: wp.array(dtype=wp.quat),
    clip_loop_root_pos: wp.array(dtype=wp.vec3),
    body_idx: int,
    out: wp.array(dtype=wp.vec3),
):
    env_idx = wp.tid()
    clip_idx = motion_clip_idx[env_idx]
    local_frame_idx = ref_frame_idx[env_idx] % clip_length[clip_idx]
    cycle_count = ref_frame_idx[env_idx] // clip_length[clip_idx]
    frame_idx = clip_start[clip_idx] + local_frame_idx
    loop_quat = _normalize_quat_wp(clip_loop_root_quat[clip_idx])
    loop_translation = clip_loop_root_pos[clip_idx]
    out[env_idx] = _apply_loop_transform_vec(
        src[frame_idx, body_idx],
        loop_quat,
        loop_translation,
        cycle_count,
    )


@wp.kernel(enable_backward=False)
def _gather_reference_root_quat_multi_kernel(
    src: wp.array2d(dtype=wp.quat),
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    clip_start: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
    clip_loop_root_quat: wp.array(dtype=wp.quat),
    body_idx: int,
    out: wp.array(dtype=wp.quat),
):
    env_idx = wp.tid()
    clip_idx = motion_clip_idx[env_idx]
    local_frame_idx = ref_frame_idx[env_idx] % clip_length[clip_idx]
    cycle_count = ref_frame_idx[env_idx] // clip_length[clip_idx]
    frame_idx = clip_start[clip_idx] + local_frame_idx
    loop_quat = _normalize_quat_wp(clip_loop_root_quat[clip_idx])
    out[env_idx] = _apply_loop_transform_quat(
        src[frame_idx, body_idx],
        loop_quat,
        cycle_count,
    )


def _frame_mode(frame: tracking_metrics.ReferenceFrame | str) -> int:
    frame_key = str(frame).strip().lower()
    if frame_key == "world":
        return FRAME_WORLD
    if frame_key == "translation":
        return FRAME_TRANSLATION
    if frame_key == "heading":
        return FRAME_HEADING
    if frame_key == "root":
        return FRAME_ROOT
    raise ValueError(
        f"Unsupported reference frame '{frame}'. Use one of: world, translation, heading, root."
    )


def _default_root_pos(
    body_tensor: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros(
        (int(body_tensor.shape[0]), 3),
        device=body_tensor.device,
        dtype=body_tensor.dtype,
    )


def _default_root_quat(
    body_tensor: torch.Tensor,
) -> torch.Tensor:
    quat = torch.zeros(
        (int(body_tensor.shape[0]), 4),
        device=body_tensor.device,
        dtype=body_tensor.dtype,
    )
    quat[:, 0] = 1.0
    return quat


def create_metric_buffers(
    *,
    env_count: int,
    body_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> MetricBuffers:
    return MetricBuffers(
        mean_error=torch.empty(env_count, device=device, dtype=dtype),
        per_body_error=torch.empty(env_count, body_count, device=device, dtype=dtype),
    )


def create_reference_tracking_metric_workspace(
    *,
    env_count: int,
    ape_body_count: int,
    ee_body_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ReferenceTrackingMetricWorkspace:
    return ReferenceTrackingMetricWorkspace(
        sim_ape_body_pos=torch.empty(
            env_count, ape_body_count, 3, device=device, dtype=dtype
        ),
        sim_ee_body_pos=torch.empty(
            env_count, ee_body_count, 3, device=device, dtype=dtype
        ),
        ref_ape_body_pos=torch.empty(
            env_count, ape_body_count, 3, device=device, dtype=dtype
        ),
        ref_ee_body_pos=torch.empty(
            env_count, ee_body_count, 3, device=device, dtype=dtype
        ),
        sim_tracking_body_linvel=torch.empty(
            env_count, ape_body_count, 3, device=device, dtype=dtype
        ),
        ref_tracking_body_linvel=torch.empty(
            env_count, ape_body_count, 3, device=device, dtype=dtype
        ),
        sim_root_pos=torch.empty(env_count, 3, device=device, dtype=dtype),
        ref_root_pos=torch.empty(env_count, 3, device=device, dtype=dtype),
        sim_root_quat=torch.empty(env_count, 4, device=device, dtype=dtype),
        ref_root_quat=torch.empty(env_count, 4, device=device, dtype=dtype),
        ape=create_metric_buffers(
            env_count=env_count,
            body_count=ape_body_count,
            device=device,
            dtype=dtype,
        ),
        global_mpjpe=create_metric_buffers(
            env_count=env_count,
            body_count=ape_body_count,
            device=device,
            dtype=dtype,
        ),
        ee=create_metric_buffers(
            env_count=env_count,
            body_count=ee_body_count,
            device=device,
            dtype=dtype,
        ),
    )


def _int32_index_tensor(
    src: torch.Tensor,
    current: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if src.dtype == torch.int32:
        return src, current
    if (
        current is None
        or current.shape != src.shape
        or current.device != src.device
        or current.dtype != torch.int32
    ):
        current = torch.empty(src.shape, device=src.device, dtype=torch.int32)
    current.copy_(src)
    return current, current


def compute_average_pose_error_into_warp(
    sim_body_pos: torch.Tensor,
    ref_body_pos: torch.Tensor,
    *,
    sim_root_pos: torch.Tensor | None = None,
    ref_root_pos: torch.Tensor | None = None,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    frame: tracking_metrics.ReferenceFrame = "root",
    out: MetricBuffers,
) -> MetricBuffers:
    launch_device = wp.device_from_torch(sim_body_pos.device)
    sim_root_pos_t = (
        _default_root_pos(sim_body_pos) if sim_root_pos is None else sim_root_pos
    )
    ref_root_pos_t = (
        _default_root_pos(ref_body_pos) if ref_root_pos is None else ref_root_pos
    )
    sim_root_quat_t = (
        _default_root_quat(sim_body_pos) if sim_root_quat is None else sim_root_quat
    )
    ref_root_quat_t = (
        _default_root_quat(ref_body_pos) if ref_root_quat is None else ref_root_quat
    )
    wp.launch(
        _position_metric_kernel,
        dim=int(sim_body_pos.shape[0]),
        inputs=[
            sim_body_pos,
            ref_body_pos,
            sim_root_pos_t,
            ref_root_pos_t,
            sim_root_quat_t,
            ref_root_quat_t,
            int(_frame_mode(frame)),
        ],
        outputs=[
            out.mean_error,
            out.per_body_error,
        ],
        device=launch_device,
    )
    return out


def compute_end_effector_error_into_warp(
    sim_body_pos: torch.Tensor,
    ref_body_pos: torch.Tensor,
    *,
    sim_root_pos: torch.Tensor | None = None,
    ref_root_pos: torch.Tensor | None = None,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    frame: tracking_metrics.ReferenceFrame = "root",
    out: MetricBuffers,
) -> MetricBuffers:
    return compute_average_pose_error_into_warp(
        sim_body_pos,
        ref_body_pos,
        sim_root_pos=sim_root_pos,
        ref_root_pos=ref_root_pos,
        sim_root_quat=sim_root_quat,
        ref_root_quat=ref_root_quat,
        frame=frame,
        out=out,
    )


def compute_mean_velocity_error_into_warp(
    sim_body_vel: torch.Tensor,
    ref_body_vel: torch.Tensor,
    *,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    frame: tracking_metrics.ReferenceFrame = "world",
    out: MetricBuffers,
) -> MetricBuffers:
    launch_device = wp.device_from_torch(sim_body_vel.device)
    sim_root_quat_t = (
        _default_root_quat(sim_body_vel) if sim_root_quat is None else sim_root_quat
    )
    ref_root_quat_t = (
        _default_root_quat(ref_body_vel) if ref_root_quat is None else ref_root_quat
    )
    wp.launch(
        _velocity_metric_kernel,
        dim=int(sim_body_vel.shape[0]),
        inputs=[
            sim_body_vel,
            ref_body_vel,
            sim_root_quat_t,
            ref_root_quat_t,
            int(_frame_mode(frame)),
        ],
        outputs=[
            out.mean_error,
            out.per_body_error,
        ],
        device=launch_device,
    )
    return out


def compute_mean_acceleration_error_into_warp(
    sim_body_vel: torch.Tensor,
    ref_body_vel: torch.Tensor,
    prev_sim_body_vel: torch.Tensor,
    prev_ref_body_vel: torch.Tensor,
    *,
    dt: float,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    prev_sim_root_quat: torch.Tensor | None = None,
    prev_ref_root_quat: torch.Tensor | None = None,
    frame: tracking_metrics.ReferenceFrame = "world",
    out: MetricBuffers,
) -> MetricBuffers:
    launch_device = wp.device_from_torch(sim_body_vel.device)
    sim_root_quat_t = (
        _default_root_quat(sim_body_vel) if sim_root_quat is None else sim_root_quat
    )
    ref_root_quat_t = (
        _default_root_quat(ref_body_vel) if ref_root_quat is None else ref_root_quat
    )
    prev_sim_root_quat_t = (
        _default_root_quat(prev_sim_body_vel)
        if prev_sim_root_quat is None
        else prev_sim_root_quat
    )
    prev_ref_root_quat_t = (
        _default_root_quat(prev_ref_body_vel)
        if prev_ref_root_quat is None
        else prev_ref_root_quat
    )
    wp.launch(
        _acceleration_metric_kernel,
        dim=int(sim_body_vel.shape[0]),
        inputs=[
            sim_body_vel,
            ref_body_vel,
            prev_sim_body_vel,
            prev_ref_body_vel,
            float(dt),
            sim_root_quat_t,
            ref_root_quat_t,
            prev_sim_root_quat_t,
            prev_ref_root_quat_t,
            int(_frame_mode(frame)),
        ],
        outputs=[
            out.mean_error,
            out.per_body_error,
        ],
        device=launch_device,
    )
    return out


def _launch_position_metric(
    sim_body_pos: torch.Tensor,
    ref_body_pos: torch.Tensor,
    *,
    sim_root_pos: torch.Tensor | None,
    ref_root_pos: torch.Tensor | None,
    sim_root_quat: torch.Tensor | None,
    ref_root_quat: torch.Tensor | None,
    frame: tracking_metrics.ReferenceFrame | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = sim_body_pos.device
    dtype = sim_body_pos.dtype
    env_count = int(sim_body_pos.shape[0])
    body_count = int(sim_body_pos.shape[1])
    buffers = create_metric_buffers(
        env_count=env_count,
        body_count=body_count,
        device=device,
        dtype=dtype,
    )
    compute_average_pose_error_into_warp(
        sim_body_pos,
        ref_body_pos,
        sim_root_pos=sim_root_pos,
        ref_root_pos=ref_root_pos,
        sim_root_quat=sim_root_quat,
        ref_root_quat=ref_root_quat,
        frame=frame,
        out=buffers,
    )
    return buffers.mean_error, buffers.per_body_error


def _launch_velocity_metric(
    sim_body_vel: torch.Tensor,
    ref_body_vel: torch.Tensor,
    *,
    sim_root_quat: torch.Tensor | None,
    ref_root_quat: torch.Tensor | None,
    frame: tracking_metrics.ReferenceFrame | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = sim_body_vel.device
    dtype = sim_body_vel.dtype
    env_count = int(sim_body_vel.shape[0])
    body_count = int(sim_body_vel.shape[1])
    buffers = create_metric_buffers(
        env_count=env_count,
        body_count=body_count,
        device=device,
        dtype=dtype,
    )
    compute_mean_velocity_error_into_warp(
        sim_body_vel,
        ref_body_vel,
        sim_root_quat=sim_root_quat,
        ref_root_quat=ref_root_quat,
        frame=frame,
        out=buffers,
    )
    return buffers.mean_error, buffers.per_body_error


def compute_average_pose_error_warp(
    sim_body_pos: torch.Tensor,
    ref_body_pos: torch.Tensor,
    *,
    sim_root_pos: torch.Tensor | None = None,
    ref_root_pos: torch.Tensor | None = None,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    frame: tracking_metrics.ReferenceFrame = "root",
) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_position_metric(
        sim_body_pos,
        ref_body_pos,
        sim_root_pos=sim_root_pos,
        ref_root_pos=ref_root_pos,
        sim_root_quat=sim_root_quat,
        ref_root_quat=ref_root_quat,
        frame=frame,
    )


def compute_end_effector_error_warp(
    sim_body_pos: torch.Tensor,
    ref_body_pos: torch.Tensor,
    *,
    sim_root_pos: torch.Tensor | None = None,
    ref_root_pos: torch.Tensor | None = None,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    frame: tracking_metrics.ReferenceFrame = "root",
) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_position_metric(
        sim_body_pos,
        ref_body_pos,
        sim_root_pos=sim_root_pos,
        ref_root_pos=ref_root_pos,
        sim_root_quat=sim_root_quat,
        ref_root_quat=ref_root_quat,
        frame=frame,
    )


def compute_mean_velocity_error_warp(
    sim_body_vel: torch.Tensor,
    ref_body_vel: torch.Tensor,
    *,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    frame: tracking_metrics.ReferenceFrame = "world",
) -> tuple[torch.Tensor, torch.Tensor]:
    return _launch_velocity_metric(
        sim_body_vel,
        ref_body_vel,
        sim_root_quat=sim_root_quat,
        ref_root_quat=ref_root_quat,
        frame=frame,
    )


def compute_mean_acceleration_error_warp(
    sim_body_vel: torch.Tensor,
    ref_body_vel: torch.Tensor,
    prev_sim_body_vel: torch.Tensor,
    prev_ref_body_vel: torch.Tensor,
    *,
    dt: float,
    sim_root_quat: torch.Tensor | None = None,
    ref_root_quat: torch.Tensor | None = None,
    prev_sim_root_quat: torch.Tensor | None = None,
    prev_ref_root_quat: torch.Tensor | None = None,
    frame: tracking_metrics.ReferenceFrame = "world",
) -> tuple[torch.Tensor, torch.Tensor]:
    device = sim_body_vel.device
    dtype = sim_body_vel.dtype
    env_count = int(sim_body_vel.shape[0])
    body_count = int(sim_body_vel.shape[1])
    buffers = create_metric_buffers(
        env_count=env_count,
        body_count=body_count,
        device=device,
        dtype=dtype,
    )
    compute_mean_acceleration_error_into_warp(
        sim_body_vel,
        ref_body_vel,
        prev_sim_body_vel,
        prev_ref_body_vel,
        dt=dt,
        sim_root_quat=sim_root_quat,
        ref_root_quat=ref_root_quat,
        prev_sim_root_quat=prev_sim_root_quat,
        prev_ref_root_quat=prev_ref_root_quat,
        frame=frame,
        out=buffers,
    )
    return buffers.mean_error, buffers.per_body_error


def compute_reference_tracking_metrics_into_warp(
    *,
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    body_linvel: torch.Tensor,
    reference_body_pos: torch.Tensor,
    reference_body_quat: torch.Tensor,
    reference_body_linvel: torch.Tensor,
    ref_frame_idx: torch.Tensor,
    root_body_idx: int,
    ape_body_idx: torch.Tensor,
    ee_body_idx: torch.Tensor,
    loop_root_quat: torch.Tensor | None = None,
    loop_root_pos: torch.Tensor | None = None,
    motion_clip_idx: torch.Tensor | None = None,
    clip_start: torch.Tensor | None = None,
    clip_length: torch.Tensor | None = None,
    clip_loop_root_quat: torch.Tensor | None = None,
    clip_loop_root_pos: torch.Tensor | None = None,
    frame: tracking_metrics.ReferenceFrame = "root",
    workspace: ReferenceTrackingMetricWorkspace,
) -> ReferenceTrackingMetricWorkspace:
    env_count = int(body_pos.shape[0])
    launch_device = wp.device_from_torch(body_pos.device)
    motion_frame_count = int(reference_body_pos.shape[0])
    ref_frame_idx_t, workspace.ref_frame_idx_i32 = _int32_index_tensor(
        ref_frame_idx,
        workspace.ref_frame_idx_i32,
    )
    multi_motion = (
        motion_clip_idx is not None
        and clip_start is not None
        and clip_length is not None
        and clip_loop_root_quat is not None
        and clip_loop_root_pos is not None
    )
    single_motion_loop = (
        not multi_motion and loop_root_quat is not None and loop_root_pos is not None
    )
    if multi_motion:
        assert motion_clip_idx is not None
        assert clip_start is not None
        assert clip_length is not None
        motion_clip_idx_t, workspace.motion_clip_idx_i32 = _int32_index_tensor(
            motion_clip_idx,
            workspace.motion_clip_idx_i32,
        )
        clip_start_t, workspace.clip_start_i32 = _int32_index_tensor(
            clip_start,
            workspace.clip_start_i32,
        )
        clip_length_t, workspace.clip_length_i32 = _int32_index_tensor(
            clip_length,
            workspace.clip_length_i32,
        )
    else:
        motion_clip_idx_t = None
        clip_start_t = None
        clip_length_t = None

    wp.launch(
        _gather_body_subset_vec3_kernel,
        dim=(env_count, int(ape_body_idx.shape[0])),
        inputs=[
            body_pos,
            ape_body_idx,
            workspace.sim_ape_body_pos,
        ],
        device=launch_device,
    )
    wp.launch(
        _gather_body_subset_vec3_kernel,
        dim=(env_count, int(ee_body_idx.shape[0])),
        inputs=[
            body_pos,
            ee_body_idx,
            workspace.sim_ee_body_pos,
        ],
        device=launch_device,
    )
    wp.launch(
        _gather_body_subset_vec3_kernel,
        dim=(env_count, int(ape_body_idx.shape[0])),
        inputs=[
            body_linvel,
            ape_body_idx,
            workspace.sim_tracking_body_linvel,
        ],
        device=launch_device,
    )
    wp.launch(
        _gather_body_root_vec3_kernel,
        dim=env_count,
        inputs=[
            body_pos,
            int(root_body_idx),
            workspace.sim_root_pos,
        ],
        device=launch_device,
    )
    wp.launch(
        _gather_body_root_quat_kernel,
        dim=env_count,
        inputs=[
            body_quat,
            int(root_body_idx),
            workspace.sim_root_quat,
        ],
        device=launch_device,
    )
    if multi_motion:
        assert motion_clip_idx_t is not None
        assert clip_start_t is not None
        assert clip_length_t is not None
        assert clip_loop_root_quat is not None
        assert clip_loop_root_pos is not None
        wp.launch(
            _gather_reference_subset_pos_multi_kernel,
            dim=(env_count, int(ape_body_idx.shape[0])),
            inputs=[
                reference_body_pos,
                motion_clip_idx_t,
                ref_frame_idx_t,
                clip_start_t,
                clip_length_t,
                clip_loop_root_quat,
                clip_loop_root_pos,
                ape_body_idx,
                workspace.ref_ape_body_pos,
            ],
            device=launch_device,
        )
        wp.launch(
            _gather_reference_subset_pos_multi_kernel,
            dim=(env_count, int(ee_body_idx.shape[0])),
            inputs=[
                reference_body_pos,
                motion_clip_idx_t,
                ref_frame_idx_t,
                clip_start_t,
                clip_length_t,
                clip_loop_root_quat,
                clip_loop_root_pos,
                ee_body_idx,
                workspace.ref_ee_body_pos,
            ],
            device=launch_device,
        )
        wp.launch(
            _gather_reference_subset_linvel_multi_kernel,
            dim=(env_count, int(ape_body_idx.shape[0])),
            inputs=[
                reference_body_linvel,
                motion_clip_idx_t,
                ref_frame_idx_t,
                clip_start_t,
                clip_length_t,
                clip_loop_root_quat,
                ape_body_idx,
                workspace.ref_tracking_body_linvel,
            ],
            device=launch_device,
        )
        wp.launch(
            _gather_reference_root_pos_multi_kernel,
            dim=env_count,
            inputs=[
                reference_body_pos,
                motion_clip_idx_t,
                ref_frame_idx_t,
                clip_start_t,
                clip_length_t,
                clip_loop_root_quat,
                clip_loop_root_pos,
                int(root_body_idx),
                workspace.ref_root_pos,
            ],
            device=launch_device,
        )
        wp.launch(
            _gather_reference_root_quat_multi_kernel,
            dim=env_count,
            inputs=[
                reference_body_quat,
                motion_clip_idx_t,
                ref_frame_idx_t,
                clip_start_t,
                clip_length_t,
                clip_loop_root_quat,
                int(root_body_idx),
                workspace.ref_root_quat,
            ],
            device=launch_device,
        )
    else:
        if single_motion_loop:
            assert loop_root_quat is not None
            assert loop_root_pos is not None
            wp.launch(
                _gather_reference_subset_pos_loop_kernel,
                dim=(env_count, int(ape_body_idx.shape[0])),
                inputs=[
                    reference_body_pos,
                    ref_frame_idx_t,
                    motion_frame_count,
                    loop_root_quat,
                    loop_root_pos,
                    ape_body_idx,
                    workspace.ref_ape_body_pos,
                ],
                device=launch_device,
            )
            wp.launch(
                _gather_reference_subset_pos_loop_kernel,
                dim=(env_count, int(ee_body_idx.shape[0])),
                inputs=[
                    reference_body_pos,
                    ref_frame_idx_t,
                    motion_frame_count,
                    loop_root_quat,
                    loop_root_pos,
                    ee_body_idx,
                    workspace.ref_ee_body_pos,
                ],
                device=launch_device,
            )
            wp.launch(
                _gather_reference_subset_linvel_loop_kernel,
                dim=(env_count, int(ape_body_idx.shape[0])),
                inputs=[
                    reference_body_linvel,
                    ref_frame_idx_t,
                    motion_frame_count,
                    loop_root_quat,
                    ape_body_idx,
                    workspace.ref_tracking_body_linvel,
                ],
                device=launch_device,
            )
            wp.launch(
                _gather_reference_root_pos_loop_kernel,
                dim=env_count,
                inputs=[
                    reference_body_pos,
                    ref_frame_idx_t,
                    motion_frame_count,
                    loop_root_quat,
                    loop_root_pos,
                    int(root_body_idx),
                    workspace.ref_root_pos,
                ],
                device=launch_device,
            )
            wp.launch(
                _gather_reference_root_quat_loop_kernel,
                dim=env_count,
                inputs=[
                    reference_body_quat,
                    ref_frame_idx_t,
                    motion_frame_count,
                    loop_root_quat,
                    int(root_body_idx),
                    workspace.ref_root_quat,
                ],
                device=launch_device,
            )
        else:
            wp.launch(
                _gather_reference_subset_vec3_kernel,
                dim=(env_count, int(ape_body_idx.shape[0])),
                inputs=[
                    reference_body_pos,
                    ref_frame_idx_t,
                    motion_frame_count,
                    ape_body_idx,
                    workspace.ref_ape_body_pos,
                ],
                device=launch_device,
            )
            wp.launch(
                _gather_reference_subset_vec3_kernel,
                dim=(env_count, int(ee_body_idx.shape[0])),
                inputs=[
                    reference_body_pos,
                    ref_frame_idx_t,
                    motion_frame_count,
                    ee_body_idx,
                    workspace.ref_ee_body_pos,
                ],
                device=launch_device,
            )
            wp.launch(
                _gather_reference_subset_vec3_kernel,
                dim=(env_count, int(ape_body_idx.shape[0])),
                inputs=[
                    reference_body_linvel,
                    ref_frame_idx_t,
                    motion_frame_count,
                    ape_body_idx,
                    workspace.ref_tracking_body_linvel,
                ],
                device=launch_device,
            )
            wp.launch(
                _gather_reference_root_vec3_kernel,
                dim=env_count,
                inputs=[
                    reference_body_pos,
                    ref_frame_idx_t,
                    motion_frame_count,
                    int(root_body_idx),
                    workspace.ref_root_pos,
                ],
                device=launch_device,
            )
            wp.launch(
                _gather_reference_root_quat_kernel,
                dim=env_count,
                inputs=[
                    reference_body_quat,
                    ref_frame_idx_t,
                    motion_frame_count,
                    int(root_body_idx),
                    workspace.ref_root_quat,
                ],
                device=launch_device,
            )
    compute_average_pose_error_into_warp(
        workspace.sim_ape_body_pos,
        workspace.ref_ape_body_pos,
        sim_root_pos=workspace.sim_root_pos,
        ref_root_pos=workspace.ref_root_pos,
        sim_root_quat=workspace.sim_root_quat,
        ref_root_quat=workspace.ref_root_quat,
        frame=frame,
        out=workspace.ape,
    )
    compute_average_pose_error_into_warp(
        workspace.sim_ape_body_pos,
        workspace.ref_ape_body_pos,
        frame="world",
        out=workspace.global_mpjpe,
    )
    compute_end_effector_error_into_warp(
        workspace.sim_ee_body_pos,
        workspace.ref_ee_body_pos,
        sim_root_pos=workspace.sim_root_pos,
        ref_root_pos=workspace.ref_root_pos,
        sim_root_quat=workspace.sim_root_quat,
        ref_root_quat=workspace.ref_root_quat,
        frame=frame,
        out=workspace.ee,
    )
    return workspace


def compute_reference_tracking_metrics_warp(
    *,
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    body_linvel: torch.Tensor,
    reference_body_state: torch.Tensor,
    ref_frame_idx: torch.Tensor,
    root_body_idx: int,
    ape_body_idx: torch.Tensor,
    ee_body_idx: torch.Tensor,
    loop_root_quat: torch.Tensor | None = None,
    loop_root_pos: torch.Tensor | None = None,
    frame: tracking_metrics.ReferenceFrame = "root",
) -> dict[str, torch.Tensor]:
    workspace = create_reference_tracking_metric_workspace(
        env_count=int(body_pos.shape[0]),
        ape_body_count=int(ape_body_idx.shape[0]),
        ee_body_count=int(ee_body_idx.shape[0]),
        device=body_pos.device,
        dtype=body_pos.dtype,
    )
    compute_reference_tracking_metrics_into_warp(
        body_pos=body_pos,
        body_quat=body_quat,
        body_linvel=body_linvel,
        reference_body_pos=reference_body_state[..., :3],
        reference_body_quat=reference_body_state[..., 3:7],
        reference_body_linvel=reference_body_state[..., 7:10],
        ref_frame_idx=ref_frame_idx,
        root_body_idx=root_body_idx,
        ape_body_idx=ape_body_idx,
        ee_body_idx=ee_body_idx,
        loop_root_quat=loop_root_quat,
        loop_root_pos=loop_root_pos,
        frame=frame,
        workspace=workspace,
    )
    return {
        "ape": workspace.ape.mean_error,
        "ape_per_body": workspace.ape.per_body_error,
        "global_mpjpe": workspace.global_mpjpe.mean_error,
        "global_mpjpe_per_body": workspace.global_mpjpe.per_body_error,
        "ee_error": workspace.ee.mean_error,
        "ee_error_per_body": workspace.ee.per_body_error,
        "tracking_body_pos": workspace.sim_ape_body_pos,
        "reference_body_pos": workspace.ref_ape_body_pos,
        "tracking_root_pos": workspace.sim_root_pos,
        "reference_root_pos": workspace.ref_root_pos,
        "tracking_body_linvel": workspace.sim_tracking_body_linvel,
        "reference_body_linvel": workspace.ref_tracking_body_linvel,
        "sim_root_quat": workspace.sim_root_quat,
        "ref_root_quat": workspace.ref_root_quat,
    }
