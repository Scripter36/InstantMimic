from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import warp as wp
from pymotion.rotations import quat_torch

from instantmimic.envs.snu.tasks.snu_imitation.warp_utils import (
    normalize_quat_wp,
    read_body_state_angvel,
    read_body_state_linvel,
    read_body_state_pos,
    read_body_state_quat,
    read_quat_from_2d,
    read_vec3_from_2d,
    write_tan_norm,
    write_vec3,
)
from instantmimic.motion.warp_utils import (
    apply_loop_rotation_vec,
    apply_loop_transform_quat,
    apply_loop_transform_vec,
    clip_cycle_count,
    clip_local_frame,
    loop_cycle_count,
    loop_local_frame,
)
from instantmimic.utils.mjw_kernels.private import mul_quat, quat_inv, rot_vec_quat
from instantmimic.utils.warp_kernels import expose_kernel_name

_FUTURE_OBSERVATION_MODES = frozenset({"none", "phc_v6"})
# Per future body: position, rotation (tangent/normal), linear and angular
# velocity differences, then the reference position and rotation.
PHC_V6_BODY_FEATURES = 24


@dataclass(frozen=True)
class FutureObservationKernelSpec:
    motion_frame_count: int
    future_obs_ofs: int
    future_obs_dim: int
    future_observation_mode: str
    num_future_frames: int
    root_body_idx: int
    future_pose_body_idx: tuple[int, ...]
    future_frame_step: int
    loop_root_q: tuple[float, float, float, float]
    loop_root_t: tuple[float, float, float]
    loop_cycle_table_size: int
    multi_motion: bool = False


@dataclass(frozen=True)
class _FutureObservationKernelContext:
    motion_frame_count: int
    root_body_idx: int
    future_pose_body_idx: tuple[int, ...]
    num_future_frames: int
    future_frame_step: int
    loop_cycle_table_size: int
    num_future_pose_bodies: int
    pose_base_ofs: int
    phc_step_stride: int
    loop_root_q: Any
    loop_root_t: Any


def resolve_future_observation_mode(mode: str) -> str:
    mode_key = str(mode).strip().lower()
    if mode_key not in _FUTURE_OBSERVATION_MODES:
        valid = ", ".join(sorted(_FUTURE_OBSERVATION_MODES))
        raise ValueError(
            f"Unsupported future_observation_mode={mode!r}. Valid values: {valid}."
        )
    return mode_key


def future_observation_uses_phc_v6(mode: str) -> bool:
    return resolve_future_observation_mode(mode) == "phc_v6"


def future_observation_feature_dim(
    *,
    mode: str,
    num_future_pose_bodies: int,
    num_future_frames: int,
) -> int:
    if not future_observation_uses_phc_v6(mode):
        return 0
    return int(num_future_pose_bodies) * int(num_future_frames) * PHC_V6_BODY_FEATURES


def build_loop_cycle_transform_tables(
    *,
    loop_root_quat: torch.Tensor,
    loop_root_pos: torch.Tensor,
    max_cycles: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute repeated loop root transforms for fast per-step lookup."""
    max_cycles = max(0, int(max_cycles))
    device = loop_root_quat.device
    loop_q = loop_root_quat.reshape(4).detach().cpu().to(dtype=torch.float32)
    loop_t = loop_root_pos.reshape(3).detach().cpu().to(dtype=torch.float32)
    cycle_q = torch.zeros(
        (max_cycles + 1, 4), device=loop_q.device, dtype=torch.float32
    )
    cycle_t = torch.zeros(
        (max_cycles + 1, 3), device=loop_q.device, dtype=torch.float32
    )
    cycle_q[0, 0] = 1.0
    for cycle_idx in range(1, max_cycles + 1):
        cycle_q[cycle_idx] = quat_torch.mul(loop_q, cycle_q[cycle_idx - 1])
        cycle_t[cycle_idx] = quat_torch.mul_vec(loop_q, cycle_t[cycle_idx - 1]) + loop_t
    return (
        cycle_q.to(device=device, non_blocking=True).contiguous(),
        cycle_t.to(device=device, non_blocking=True).contiguous(),
    )


def _build_future_observation_kernel_context(
    spec: FutureObservationKernelSpec,
) -> _FutureObservationKernelContext:
    future_pose_body_idx = tuple(int(v) for v in spec.future_pose_body_idx)
    num_future_pose_bodies = len(future_pose_body_idx)
    loop_root_q = wp.quat(
        float(spec.loop_root_q[0]),
        float(spec.loop_root_q[1]),
        float(spec.loop_root_q[2]),
        float(spec.loop_root_q[3]),
    )
    loop_root_t = wp.vec3(
        float(spec.loop_root_t[0]),
        float(spec.loop_root_t[1]),
        float(spec.loop_root_t[2]),
    )
    return _FutureObservationKernelContext(
        motion_frame_count=int(spec.motion_frame_count),
        root_body_idx=int(spec.root_body_idx),
        future_pose_body_idx=future_pose_body_idx,
        num_future_frames=int(spec.num_future_frames),
        future_frame_step=int(spec.future_frame_step),
        loop_cycle_table_size=int(spec.loop_cycle_table_size),
        num_future_pose_bodies=num_future_pose_bodies,
        pose_base_ofs=int(spec.future_obs_ofs),
        phc_step_stride=num_future_pose_bodies * PHC_V6_BODY_FEATURES,
        loop_root_q=loop_root_q,
        loop_root_t=loop_root_t,
    )


def make_future_observation_kernel(spec: FutureObservationKernelSpec):
    if int(spec.future_obs_dim) <= 0 or not future_observation_uses_phc_v6(
        spec.future_observation_mode
    ):
        return None
    ctx = _build_future_observation_kernel_context(spec)
    if spec.multi_motion:
        return _make_multi_phc_v6_future_observation_kernel(ctx)
    return _make_single_phc_v6_future_observation_kernel(ctx)


def _make_single_phc_v6_future_observation_kernel(ctx: _FutureObservationKernelContext):
    motion_frame_count = ctx.motion_frame_count
    root_body_idx = ctx.root_body_idx
    future_frame_step = ctx.future_frame_step
    loop_cycle_table_size = ctx.loop_cycle_table_size
    pose_base_ofs = ctx.pose_base_ofs
    phc_step_stride = ctx.phc_step_stride
    loop_root_q = ctx.loop_root_q
    loop_root_t = ctx.loop_root_t

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def write_future_pose_observation_single_phc_v6_kernel(
        ref_body_state: wp.array3d(dtype=float),
        ref_frame_idx: wp.array(dtype=int),
        future_pose_body_idx_arr: wp.array(dtype=int),
        loop_cycle_quat: wp.array2d(dtype=float),
        loop_cycle_pos: wp.array2d(dtype=float),
        xpos: wp.array2d(dtype=wp.vec3),
        xquat: wp.array2d(dtype=wp.quat),
        cvel: wp.array2d(dtype=wp.spatial_vector),
        root_yaw_inv_buf: wp.array2d(dtype=float),
        obs: wp.array2d(dtype=float),
    ):
        env_idx, step_idx, pose_idx = wp.tid()
        frame_idx = ref_frame_idx[env_idx] + 1 + step_idx * future_frame_step
        local_idx = loop_local_frame(frame_idx, motion_frame_count)
        cycles = loop_cycle_count(frame_idx, motion_frame_count)

        use_cycle_table = cycles < wp.static(loop_cycle_table_size)
        cycle_q = wp.quat(1.0, 0.0, 0.0, 0.0)
        cycle_t = wp.vec3(0.0, 0.0, 0.0)
        if use_cycle_table:
            cycle_q = read_quat_from_2d(loop_cycle_quat, cycles, 0)
            cycle_t = read_vec3_from_2d(loop_cycle_pos, cycles, 0)

        yaw_inv = normalize_quat_wp(read_quat_from_2d(root_yaw_inv_buf, env_idx, 0))
        heading = quat_inv(yaw_inv)
        root_pos = xpos[env_idx, root_body_idx]
        body_idx = future_pose_body_idx_arr[pose_idx]

        ref_body_pos = read_body_state_pos(ref_body_state, local_idx, body_idx)
        ref_body_q = read_body_state_quat(ref_body_state, local_idx, body_idx)
        ref_body_lv = read_body_state_linvel(ref_body_state, local_idx, body_idx)
        ref_body_av = read_body_state_angvel(ref_body_state, local_idx, body_idx)

        if use_cycle_table:
            ref_body_pos = rot_vec_quat(ref_body_pos, cycle_q) + cycle_t
            ref_body_q = normalize_quat_wp(mul_quat(cycle_q, ref_body_q))
            ref_body_lv = rot_vec_quat(ref_body_lv, cycle_q)
            ref_body_av = rot_vec_quat(ref_body_av, cycle_q)
        else:
            ref_body_pos = apply_loop_transform_vec(
                ref_body_pos, loop_root_q, loop_root_t, cycles
            )
            ref_body_q = normalize_quat_wp(
                apply_loop_transform_quat(ref_body_q, loop_root_q, cycles)
            )
            ref_body_lv = apply_loop_rotation_vec(ref_body_lv, loop_root_q, cycles)
            ref_body_av = apply_loop_rotation_vec(ref_body_av, loop_root_q, cycles)

        sim_body_pos = xpos[env_idx, body_idx]
        sim_body_q = normalize_quat_wp(xquat[env_idx, body_idx])
        sim_body_cv = cvel[env_idx, body_idx]
        sim_body_lv = wp.vec3(sim_body_cv[3], sim_body_cv[4], sim_body_cv[5])
        sim_body_av = wp.vec3(sim_body_cv[0], sim_body_cv[1], sim_body_cv[2])

        diff_pos = rot_vec_quat(ref_body_pos - sim_body_pos, yaw_inv)
        diff_rot = normalize_quat_wp(mul_quat(ref_body_q, quat_inv(sim_body_q)))
        diff_rot = normalize_quat_wp(mul_quat(mul_quat(yaw_inv, diff_rot), heading))
        diff_lv = rot_vec_quat(ref_body_lv - sim_body_lv, yaw_inv)
        diff_av = rot_vec_quat(ref_body_av - sim_body_av, yaw_inv)
        local_ref_pos = rot_vec_quat(ref_body_pos - root_pos, yaw_inv)
        local_ref_rot = normalize_quat_wp(mul_quat(yaw_inv, ref_body_q))
        body_ofs = pose_base_ofs + step_idx * phc_step_stride + pose_idx * 24

        write_vec3(obs, env_idx, body_ofs + 0, diff_pos, 1.0)
        write_tan_norm(obs, env_idx, body_ofs + 3, diff_rot)
        write_vec3(obs, env_idx, body_ofs + 9, diff_lv, 0.1)
        write_vec3(obs, env_idx, body_ofs + 12, diff_av, 0.1)
        write_vec3(obs, env_idx, body_ofs + 15, local_ref_pos, 1.0)
        write_tan_norm(obs, env_idx, body_ofs + 18, local_ref_rot)

    return write_future_pose_observation_single_phc_v6_kernel


def _make_multi_phc_v6_future_observation_kernel(ctx: _FutureObservationKernelContext):
    root_body_idx = ctx.root_body_idx
    future_pose_body_idx = ctx.future_pose_body_idx
    num_future_frames = ctx.num_future_frames
    future_frame_step = ctx.future_frame_step
    num_future_pose_bodies = ctx.num_future_pose_bodies
    pose_base_ofs = ctx.pose_base_ofs
    phc_step_stride = ctx.phc_step_stride

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def write_future_pose_observation_multi_phc_v6_kernel(
        ref_body_state: wp.array3d(dtype=float),
        motion_clip_idx: wp.array(dtype=int),
        ref_frame_idx: wp.array(dtype=int),
        clip_start: wp.array(dtype=int),
        clip_length: wp.array(dtype=int),
        clip_loop_root_quat: wp.array2d(dtype=float),
        clip_loop_root_pos: wp.array2d(dtype=float),
        xpos: wp.array2d(dtype=wp.vec3),
        xquat: wp.array2d(dtype=wp.quat),
        cvel: wp.array2d(dtype=wp.spatial_vector),
        root_yaw_inv_buf: wp.array2d(dtype=float),
        obs: wp.array2d(dtype=float),
    ):
        env_idx = wp.tid()
        clip_idx = motion_clip_idx[env_idx]
        fi = ref_frame_idx[env_idx]
        yaw_inv = normalize_quat_wp(read_quat_from_2d(root_yaw_inv_buf, env_idx, 0))
        heading = quat_inv(yaw_inv)
        root_pos = xpos[env_idx, root_body_idx]
        loop_root_q_multi = normalize_quat_wp(
            read_quat_from_2d(clip_loop_root_quat, clip_idx, 0)
        )
        loop_root_t_multi = read_vec3_from_2d(clip_loop_root_pos, clip_idx, 0)

        for step_idx in range(wp.static(num_future_frames)):
            frame_idx = fi + 1 + step_idx * future_frame_step
            local_idx = clip_local_frame(frame_idx, clip_idx, clip_length)
            cycles = clip_cycle_count(frame_idx, clip_idx, clip_length)
            global_idx = clip_start[clip_idx] + local_idx

            for pose_idx in range(wp.static(num_future_pose_bodies)):
                body_idx = wp.static(future_pose_body_idx[pose_idx])

                ref_body_pos = read_body_state_pos(ref_body_state, global_idx, body_idx)
                ref_body_q = read_body_state_quat(ref_body_state, global_idx, body_idx)
                ref_body_lv = read_body_state_linvel(
                    ref_body_state, global_idx, body_idx
                )
                ref_body_av = read_body_state_angvel(
                    ref_body_state, global_idx, body_idx
                )

                ref_body_pos = apply_loop_transform_vec(
                    ref_body_pos, loop_root_q_multi, loop_root_t_multi, cycles
                )
                ref_body_q = normalize_quat_wp(
                    apply_loop_transform_quat(ref_body_q, loop_root_q_multi, cycles)
                )
                ref_body_lv = apply_loop_rotation_vec(
                    ref_body_lv, loop_root_q_multi, cycles
                )
                ref_body_av = apply_loop_rotation_vec(
                    ref_body_av, loop_root_q_multi, cycles
                )

                sim_body_pos = xpos[env_idx, body_idx]
                sim_body_q = normalize_quat_wp(xquat[env_idx, body_idx])
                sim_body_cv = cvel[env_idx, body_idx]
                sim_body_lv = wp.vec3(sim_body_cv[3], sim_body_cv[4], sim_body_cv[5])
                sim_body_av = wp.vec3(sim_body_cv[0], sim_body_cv[1], sim_body_cv[2])

                diff_pos = rot_vec_quat(ref_body_pos - sim_body_pos, yaw_inv)
                diff_rot = normalize_quat_wp(mul_quat(ref_body_q, quat_inv(sim_body_q)))
                diff_rot = normalize_quat_wp(
                    mul_quat(mul_quat(yaw_inv, diff_rot), heading)
                )
                diff_lv = rot_vec_quat(ref_body_lv - sim_body_lv, yaw_inv)
                diff_av = rot_vec_quat(ref_body_av - sim_body_av, yaw_inv)
                local_ref_pos = rot_vec_quat(ref_body_pos - root_pos, yaw_inv)
                local_ref_rot = normalize_quat_wp(mul_quat(yaw_inv, ref_body_q))
                body_ofs = pose_base_ofs + step_idx * phc_step_stride + pose_idx * 24

                write_vec3(obs, env_idx, body_ofs + 0, diff_pos, 1.0)
                write_tan_norm(obs, env_idx, body_ofs + 3, diff_rot)
                write_vec3(obs, env_idx, body_ofs + 9, diff_lv, 0.1)
                write_vec3(obs, env_idx, body_ofs + 12, diff_av, 0.1)
                write_vec3(obs, env_idx, body_ofs + 15, local_ref_pos, 1.0)
                write_tan_norm(obs, env_idx, body_ofs + 18, local_ref_rot)

    return write_future_pose_observation_multi_phc_v6_kernel
