from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import warp as wp
from pymotion.rotations import ortho6d_torch, quat_torch

from instantmimic.envs.snu.tasks.snu_imitation.warp_utils import (
    heading_yaw_quat,
    normalize_quat_wp,
    read_body_state_angvel,
    read_body_state_linvel,
    read_body_state_pos,
    read_body_state_quat,
    read_quat_from_2d,
    read_vec3_from_2d,
    write_ortho6d,
    write_tan_norm,
    write_vec3,
)
from instantmimic.envs.tracking import metrics as tracking_metrics
from instantmimic.motion.warp_utils import (
    apply_loop_rotation_vec,
    apply_loop_transform_quat,
    apply_loop_transform_vec,
    clip_cycle_count,
    clip_global_frame,
    clip_local_frame,
    loop_cycle_count,
    loop_local_frame,
)
from instantmimic.utils.mjw_kernels.private import mul_quat, quat_inv, rot_vec_quat
from instantmimic.utils.warp_kernels import expose_kernel_name

_FUTURE_OBSERVATION_MODES = frozenset(
    {
        "none",
        "ee_future_root",
        "ee_current_root",
        "root_ee_current_root",
        "root_pose_current_root",
        "phc_v6",
        "phc_v6_delayed",
    }
)


@dataclass(frozen=True)
class FutureObservationKernelSpec:
    motion_frame_count: int
    future_obs_ofs: int
    future_obs_dim: int
    future_observation_mode: str
    num_future_frames: int
    root_body_idx: int
    ee_body_idx: tuple[int, ...]
    future_pose_body_idx: tuple[int, ...]
    future_frame_step: int
    loop_root_q: tuple[float, float, float, float]
    loop_root_t: tuple[float, float, float]
    loop_cycle_table_size: int
    multi_motion: bool = False


@dataclass(frozen=True)
class _FutureObservationKernelContext:
    motion_frame_count: int
    future_obs_ofs: int
    future_obs_feature_dim: int
    root_body_idx: int
    ee_body_idx: tuple[int, ...]
    future_pose_body_idx: tuple[int, ...]
    num_future_frames: int
    future_frame_step: int
    loop_cycle_table_size: int
    future_obs_uses_cache: bool
    future_obs_live: bool
    future_obs_includes_root: bool
    future_obs_includes_ee: bool
    future_obs_includes_pose: bool
    future_obs_phc_v6: bool
    future_obs_phc_v6_features: bool
    num_ee: int
    num_future_pose_bodies: int
    root_future_dim: int
    ee_block_width: int
    ee_future_dim: int
    pose_block_width: int
    ee_pos_ofs: int
    ee_rot_ofs: int
    ee_lv_ofs: int
    ee_av_ofs: int
    pose_base_ofs: int
    pose_pos_ofs: int
    pose_rot_ofs: int
    phc_step_stride: int
    root_pos_ofs: int
    root_rot_ofs: int
    loop_root_q: Any
    loop_root_t: Any
    multi_motion: bool


def resolve_future_observation_mode(mode: str) -> str:
    mode_key = str(mode).strip().lower()
    if mode_key not in _FUTURE_OBSERVATION_MODES:
        valid = ", ".join(sorted(_FUTURE_OBSERVATION_MODES))
        raise ValueError(
            f"Unsupported future_observation_mode={mode!r}. Valid values: {valid}."
        )
    return mode_key


def _future_observation_includes_root(mode: str) -> bool:
    return resolve_future_observation_mode(mode) in (
        "root_ee_current_root",
        "root_pose_current_root",
    )


def _future_observation_includes_ee(mode: str) -> bool:
    return resolve_future_observation_mode(mode) in (
        "ee_future_root",
        "ee_current_root",
        "root_ee_current_root",
    )


def future_observation_includes_pose(mode: str) -> bool:
    return resolve_future_observation_mode(mode) in (
        "root_pose_current_root",
        "phc_v6",
        "phc_v6_delayed",
    )


def _future_observation_is_phc_v6(mode: str) -> bool:
    return resolve_future_observation_mode(mode) == "phc_v6"


def future_observation_uses_phc_v6_features(mode: str) -> bool:
    return resolve_future_observation_mode(mode) in ("phc_v6", "phc_v6_delayed")


def future_observation_uses_cache(mode: str) -> bool:
    return resolve_future_observation_mode(mode) == "ee_future_root"


def future_observation_uses_immediate_next_frame(mode: str) -> bool:
    resolved_mode = resolve_future_observation_mode(mode)
    return resolved_mode == "ee_future_root" or resolved_mode == "phc_v6"


def future_observation_is_live(mode: str) -> bool:
    return resolve_future_observation_mode(mode) in (
        "ee_current_root",
        "root_ee_current_root",
        "root_pose_current_root",
        "phc_v6",
        "phc_v6_delayed",
    )


def _future_observation_pose_feature_stride(mode: str) -> int:
    resolved_mode = resolve_future_observation_mode(mode)
    if resolved_mode == "root_pose_current_root":
        return 9
    if resolved_mode in ("phc_v6", "phc_v6_delayed"):
        return 24
    return 0


def future_observation_feature_dim(
    *,
    mode: str,
    num_ee: int,
    num_future_pose_bodies: int,
    num_future_frames: int,
) -> int:
    resolved_mode = resolve_future_observation_mode(mode)
    root_dim = (
        int(num_future_frames) * 9 if _future_observation_includes_root(mode) else 0
    )
    ee_dim = (
        int(num_ee) * int(num_future_frames) * 15
        if resolved_mode
        in ("ee_future_root", "ee_current_root", "root_ee_current_root")
        else 0
    )
    pose_dim = (
        int(num_future_pose_bodies)
        * int(num_future_frames)
        * _future_observation_pose_feature_stride(resolved_mode)
    )
    return root_dim + ee_dim + pose_dim


def get_heading_frame(root_pos: torch.Tensor, root_quat: torch.Tensor):
    return tracking_metrics.get_heading_frame(root_pos, root_quat)


def precompute_future_observation_block(
    motion_qpos: torch.Tensor,
    reference_body_state: torch.Tensor,
    ee_body_idx: torch.Tensor,
    future_pose_body_idx: torch.Tensor,
    num_future_frames: int,
    future_frame_step: int,
    future_observation_mode: str,
    loop_root_quat: torch.Tensor | None = None,
    loop_root_pos: torch.Tensor | None = None,
    root_body_idx: int | None = None,
) -> torch.Tensor:
    """Build a one-cycle future-observation block in final obs layout order."""
    frame_count = int(motion_qpos.shape[0])
    if frame_count == 0 or num_future_frames <= 0:
        return torch.empty(
            frame_count,
            0,
            device=motion_qpos.device,
            dtype=motion_qpos.dtype,
        )

    future_offsets = _future_frame_offsets(
        num_future_frames=num_future_frames,
        future_frame_step=future_frame_step,
        device=motion_qpos.device,
        include_immediate_next_frame=True,
    ).unsqueeze(0)
    base_frame_idx = torch.arange(
        frame_count, device=motion_qpos.device, dtype=torch.long
    )
    future_global_idx = base_frame_idx.unsqueeze(1) + future_offsets

    return _precompute_cached_future_observation_block(
        motion_qpos=motion_qpos,
        reference_body_state=reference_body_state,
        base_indices=base_frame_idx,
        future_indices=future_global_idx,
        ee_body_idx=ee_body_idx,
        future_pose_body_idx=future_pose_body_idx,
        future_observation_mode=future_observation_mode,
        loop_root_quat=loop_root_quat,
        loop_root_pos=loop_root_pos,
        root_body_idx=root_body_idx,
    )


def _future_frame_offsets(
    num_future_frames: int,
    future_frame_step: int,
    device: torch.device,
    *,
    include_immediate_next_frame: bool,
) -> torch.Tensor:
    offsets = torch.arange(num_future_frames, device=device, dtype=torch.long)
    if include_immediate_next_frame:
        return 1 + offsets * int(future_frame_step)
    return (offsets + 1) * int(future_frame_step)


def _build_future_reference_indices(
    ref_frame_idx: torch.Tensor,
    num_future_frames: int,
    future_frame_step: int,
    future_observation_mode: str = "ee_future_root",
) -> torch.Tensor:
    future_offsets = _future_frame_offsets(
        num_future_frames=num_future_frames,
        future_frame_step=future_frame_step,
        device=ref_frame_idx.device,
        include_immediate_next_frame=future_observation_uses_immediate_next_frame(
            future_observation_mode
        ),
    )
    return ref_frame_idx.long()[:, None] + future_offsets[None, :]


def precompute_future_reference_indices(
    ref_frame_idx: torch.Tensor,
    num_future_frames: int,
    future_frame_step: int,
    future_observation_mode: str = "ee_future_root",
) -> torch.Tensor:
    return _build_future_reference_indices(
        ref_frame_idx=ref_frame_idx,
        num_future_frames=num_future_frames,
        future_frame_step=future_frame_step,
        future_observation_mode=future_observation_mode,
    )


def _apply_loop_rotation_vec_torch(
    values: torch.Tensor,
    loop_root_quat: torch.Tensor,
    cycles: torch.Tensor,
) -> torch.Tensor:
    if values.numel() == 0 or cycles.numel() == 0:
        return values
    max_cycles = int(cycles.max().item())
    if max_cycles <= 0:
        return values

    out = values.clone()
    loop_q = loop_root_quat.to(device=values.device, dtype=values.dtype)
    view_shape = (1,) * (values.ndim - 1) + (4,)
    loop_q_expanded = loop_q.view(view_shape).expand(*values.shape[:-1], 4)
    for cycle_step in range(max_cycles):
        mask = cycles > cycle_step
        if not mask.any().item():
            break
        out[mask] = quat_torch.mul_vec(loop_q_expanded[mask], out[mask])
    return out


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


def _apply_loop_transform_vec_torch(
    values: torch.Tensor,
    loop_root_quat: torch.Tensor,
    loop_root_pos: torch.Tensor,
    cycles: torch.Tensor,
) -> torch.Tensor:
    if values.numel() == 0 or cycles.numel() == 0:
        return values
    max_cycles = int(cycles.max().item())
    if max_cycles <= 0:
        return values

    out = values.clone()
    loop_q = loop_root_quat.to(device=values.device, dtype=values.dtype)
    loop_t = loop_root_pos.to(device=values.device, dtype=values.dtype)
    loop_q_expanded = loop_q.view((1,) * (values.ndim - 1) + (4,)).expand(
        *values.shape[:-1], 4
    )
    loop_t_expanded = loop_t.view((1,) * (values.ndim - 1) + (3,)).expand_as(values)
    for cycle_step in range(max_cycles):
        mask = cycles > cycle_step
        if not mask.any().item():
            break
        out[mask] = (
            quat_torch.mul_vec(loop_q_expanded[mask], out[mask]) + loop_t_expanded[mask]
        )
    return out


def _apply_loop_transform_quat_torch(
    values: torch.Tensor,
    loop_root_quat: torch.Tensor,
    cycles: torch.Tensor,
) -> torch.Tensor:
    if values.numel() == 0 or cycles.numel() == 0:
        return values
    max_cycles = int(cycles.max().item())
    if max_cycles <= 0:
        return values

    out = values.clone()
    loop_q = loop_root_quat.to(device=values.device, dtype=values.dtype)
    loop_q_expanded = loop_q.view((1,) * (values.ndim - 1) + (4,)).expand_as(values)
    for cycle_step in range(max_cycles):
        mask = cycles > cycle_step
        if not mask.any().item():
            break
        out[mask] = quat_torch.mul(loop_q_expanded[mask], out[mask])
    return out


def _precompute_cached_future_observation_block(
    motion_qpos: torch.Tensor,
    reference_body_state: torch.Tensor,
    base_indices: torch.Tensor,
    future_indices: torch.Tensor,
    ee_body_idx: torch.Tensor,
    future_pose_body_idx: torch.Tensor,
    future_observation_mode: str,
    loop_root_quat: torch.Tensor | None = None,
    loop_root_pos: torch.Tensor | None = None,
    root_body_idx: int | None = None,
) -> torch.Tensor:
    mode = resolve_future_observation_mode(future_observation_mode)
    if mode != "ee_future_root":
        raise ValueError(
            "cached future observation precompute only supports "
            "future_observation_mode='ee_future_root'. Live future modes use Warp kernels."
        )

    batch_size = int(base_indices.shape[0])
    motion_frame_count = int(motion_qpos.shape[0])
    if batch_size == 0:
        return torch.empty(
            batch_size,
            0,
            device=motion_qpos.device,
            dtype=motion_qpos.dtype,
        )
    if ee_body_idx.numel() == 0:
        raise ValueError(
            "cached future observation mode requires end-effectors, "
            "but ee_body_idx is empty."
        )

    base_idx_global = base_indices.long()
    future_idx_global = future_indices.long()
    future_idx = future_idx_global % motion_frame_count
    base_cycle_idx = torch.div(
        base_idx_global,
        motion_frame_count,
        rounding_mode="floor",
    )
    future_relative_cycles = (
        torch.div(
            future_idx_global,
            motion_frame_count,
            rounding_mode="floor",
        )
        - base_cycle_idx[:, None]
    )

    if root_body_idx is None:
        future_root_pos = motion_qpos[future_idx, :3]
        future_root_quat = motion_qpos[future_idx, 3:7]
    else:
        future_root_state = reference_body_state[future_idx][..., int(root_body_idx), :]
        future_root_pos = future_root_state[..., :3]
        future_root_quat = future_root_state[..., 3:7]
    needs_loop_transform = (
        loop_root_quat is not None
        and loop_root_pos is not None
        and future_relative_cycles.max().item() > 0
    )
    if needs_loop_transform:
        future_root_pos = _apply_loop_transform_vec_torch(
            future_root_pos,
            loop_root_quat,
            loop_root_pos,
            future_relative_cycles,
        )
        future_root_quat = _apply_loop_transform_quat_torch(
            future_root_quat,
            loop_root_quat,
            future_relative_cycles,
        )
    future_root_pos_xz, future_root_yaw_quat = get_heading_frame(
        future_root_pos, future_root_quat
    )

    future_ee_state = reference_body_state[future_idx][:, :, ee_body_idx, :]
    future_ee_pos = future_ee_state[:, :, :, :3]
    future_ee_quat = future_ee_state[:, :, :, 3:7]
    future_ee_linvel = future_ee_state[:, :, :, 7:10]
    future_ee_angvel = future_ee_state[:, :, :, 10:13]
    if needs_loop_transform:
        future_ee_cycles = future_relative_cycles[:, :, None].expand(
            -1, -1, int(ee_body_idx.shape[0])
        )
        future_ee_pos = _apply_loop_transform_vec_torch(
            future_ee_pos,
            loop_root_quat,
            loop_root_pos,
            future_ee_cycles,
        )
        future_ee_quat = _apply_loop_transform_quat_torch(
            future_ee_quat,
            loop_root_quat,
            future_ee_cycles,
        )
        future_ee_linvel = _apply_loop_rotation_vec_torch(
            future_ee_linvel,
            loop_root_quat,
            future_ee_cycles,
        )
        future_ee_angvel = _apply_loop_rotation_vec_torch(
            future_ee_angvel,
            loop_root_quat,
            future_ee_cycles,
        )

    reference_yaw_inv = quat_torch.inverse(future_root_yaw_quat)
    reference_yaw_inv_exp = reference_yaw_inv[:, :, None, :].expand(
        -1, -1, int(ee_body_idx.shape[0]), -1
    )
    future_ee_pos_rel = quat_torch.mul_vec(
        reference_yaw_inv_exp,
        future_ee_pos - future_root_pos_xz[:, :, None, :],
    )
    future_ee_quat_rel = quat_torch.mul(reference_yaw_inv_exp, future_ee_quat)
    future_ee_rot_6d = ortho6d_torch.from_quat(future_ee_quat_rel)
    future_ee_linvel_rel = (
        quat_torch.mul_vec(reference_yaw_inv_exp, future_ee_linvel) * 0.1
    )
    future_ee_angvel_rel = (
        quat_torch.mul_vec(reference_yaw_inv_exp, future_ee_angvel) * 0.1
    )
    return torch.cat(
        (
            future_ee_pos_rel.reshape(batch_size, -1),
            future_ee_rot_6d.reshape(batch_size, -1),
            future_ee_linvel_rel.reshape(batch_size, -1),
            future_ee_angvel_rel.reshape(batch_size, -1),
        ),
        dim=-1,
    ).contiguous()


def _build_future_observation_kernel_context(
    spec: FutureObservationKernelSpec,
) -> _FutureObservationKernelContext:
    mode = resolve_future_observation_mode(spec.future_observation_mode)
    future_obs_feature_dim = int(spec.future_obs_dim)
    motion_frame_count = int(spec.motion_frame_count)
    root_body_idx = int(spec.root_body_idx)
    ee_body_idx = tuple(int(v) for v in spec.ee_body_idx)
    future_pose_body_idx = tuple(int(v) for v in spec.future_pose_body_idx)
    num_future_frames = int(spec.num_future_frames)
    future_frame_step = int(spec.future_frame_step)
    loop_cycle_table_size = int(spec.loop_cycle_table_size)
    future_obs_ofs = int(spec.future_obs_ofs)
    future_obs_uses_cache = future_observation_uses_cache(mode)
    future_obs_live = future_observation_is_live(mode)
    future_obs_includes_root = _future_observation_includes_root(mode)
    future_obs_includes_ee = _future_observation_includes_ee(mode)
    future_obs_includes_pose = future_observation_includes_pose(mode)
    future_obs_phc_v6 = _future_observation_is_phc_v6(mode)
    future_obs_phc_v6_features = future_observation_uses_phc_v6_features(mode)
    num_ee = len(ee_body_idx)
    num_future_pose_bodies = len(future_pose_body_idx)
    root_future_dim = num_future_frames * 9 if future_obs_includes_root else 0
    ee_block_width = num_future_frames * num_ee
    ee_future_dim = ee_block_width * 15 if future_obs_includes_ee else 0
    pose_block_width = num_future_frames * num_future_pose_bodies
    ee_pos_ofs = future_obs_ofs + root_future_dim
    ee_rot_ofs = ee_pos_ofs + ee_block_width * 3
    ee_lv_ofs = ee_rot_ofs + ee_block_width * 6
    ee_av_ofs = ee_lv_ofs + ee_block_width * 3
    pose_base_ofs = future_obs_ofs + root_future_dim + ee_future_dim
    pose_pos_ofs = pose_base_ofs
    pose_rot_ofs = pose_pos_ofs + pose_block_width * 3
    phc_step_stride = num_future_pose_bodies * 24
    root_pos_ofs = future_obs_ofs
    root_rot_ofs = root_pos_ofs + num_future_frames * 3
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
        motion_frame_count=motion_frame_count,
        future_obs_ofs=future_obs_ofs,
        future_obs_feature_dim=future_obs_feature_dim,
        root_body_idx=root_body_idx,
        ee_body_idx=ee_body_idx,
        future_pose_body_idx=future_pose_body_idx,
        num_future_frames=num_future_frames,
        future_frame_step=future_frame_step,
        loop_cycle_table_size=loop_cycle_table_size,
        future_obs_uses_cache=future_obs_uses_cache,
        future_obs_live=future_obs_live,
        future_obs_includes_root=future_obs_includes_root,
        future_obs_includes_ee=future_obs_includes_ee,
        future_obs_includes_pose=future_obs_includes_pose,
        future_obs_phc_v6=future_obs_phc_v6,
        future_obs_phc_v6_features=future_obs_phc_v6_features,
        num_ee=num_ee,
        num_future_pose_bodies=num_future_pose_bodies,
        root_future_dim=root_future_dim,
        ee_block_width=ee_block_width,
        ee_future_dim=ee_future_dim,
        pose_block_width=pose_block_width,
        ee_pos_ofs=ee_pos_ofs,
        ee_rot_ofs=ee_rot_ofs,
        ee_lv_ofs=ee_lv_ofs,
        ee_av_ofs=ee_av_ofs,
        pose_base_ofs=pose_base_ofs,
        pose_pos_ofs=pose_pos_ofs,
        pose_rot_ofs=pose_rot_ofs,
        phc_step_stride=phc_step_stride,
        root_pos_ofs=root_pos_ofs,
        root_rot_ofs=root_rot_ofs,
        loop_root_q=loop_root_q,
        loop_root_t=loop_root_t,
        multi_motion=bool(spec.multi_motion),
    )


def make_future_observation_kernel(spec: FutureObservationKernelSpec):
    ctx = _build_future_observation_kernel_context(spec)
    if ctx.future_obs_feature_dim <= 0:
        return None
    if ctx.future_obs_uses_cache:
        return _make_cached_future_observation_kernel(ctx)
    if not ctx.future_obs_live:
        return None
    if (not ctx.multi_motion) and ctx.future_obs_phc_v6_features:
        return _make_single_phc_v6_future_observation_kernel(ctx)
    if not ctx.multi_motion:
        return _make_single_live_future_observation_kernel(ctx)
    return _make_multi_live_future_observation_kernel(ctx)


def _make_cached_future_observation_kernel(ctx: _FutureObservationKernelContext):
    motion_frame_count = ctx.motion_frame_count
    future_obs_ofs = ctx.future_obs_ofs
    multi_motion = ctx.multi_motion

    if not multi_motion:

        @wp.kernel(enable_backward=False)
        @expose_kernel_name
        def copy_future_observation_single_motion_kernel(
            future_obs_block: wp.array2d(dtype=float),
            ref_frame_idx: wp.array(dtype=int),
            obs: wp.array2d(dtype=float),
        ):
            env_idx, feature_idx = wp.tid()
            fi = ref_frame_idx[env_idx]
            fi_local = loop_local_frame(fi, motion_frame_count)
            obs[env_idx, future_obs_ofs + feature_idx] = future_obs_block[
                fi_local, feature_idx
            ]

        copy_kernel = copy_future_observation_single_motion_kernel

    else:

        @wp.kernel(enable_backward=False)
        @expose_kernel_name
        def copy_future_observation_multi_motion_kernel(
            future_obs_block: wp.array2d(dtype=float),
            motion_clip_idx: wp.array(dtype=int),
            ref_frame_idx: wp.array(dtype=int),
            clip_start: wp.array(dtype=int),
            clip_length: wp.array(dtype=int),
            obs: wp.array2d(dtype=float),
        ):
            env_idx, feature_idx = wp.tid()
            clip_idx = motion_clip_idx[env_idx]
            fi = clip_global_frame(
                ref_frame_idx[env_idx],
                clip_idx,
                clip_start,
                clip_length,
            )
            obs[env_idx, future_obs_ofs + feature_idx] = future_obs_block[
                fi, feature_idx
            ]

        copy_kernel = copy_future_observation_multi_motion_kernel

    return copy_kernel


def _make_single_phc_v6_future_observation_kernel(ctx: _FutureObservationKernelContext):
    motion_frame_count = ctx.motion_frame_count
    root_body_idx = ctx.root_body_idx
    future_frame_step = ctx.future_frame_step
    loop_cycle_table_size = ctx.loop_cycle_table_size
    future_obs_phc_v6 = ctx.future_obs_phc_v6
    pose_base_ofs = ctx.pose_base_ofs
    phc_step_stride = ctx.phc_step_stride
    loop_root_q = ctx.loop_root_q
    loop_root_t = ctx.loop_root_t

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def write_future_pose_observation_single_phc_v6_kernel(
        motion_qpos: wp.array2d(dtype=float),
        ref_body_state: wp.array3d(dtype=float),
        ref_frame_idx: wp.array(dtype=int),
        future_pose_body_idx_arr: wp.array(dtype=int),
        loop_cycle_quat: wp.array2d(dtype=float),
        loop_cycle_pos: wp.array2d(dtype=float),
        xpos: wp.array2d(dtype=wp.vec3),
        xquat: wp.array2d(dtype=wp.quat),
        cvel: wp.array2d(dtype=wp.spatial_vector),
        root_pos_xz_buf: wp.array2d(dtype=float),
        root_yaw_inv_buf: wp.array2d(dtype=float),
        obs: wp.array2d(dtype=float),
    ):
        env_idx, step_idx, pose_idx = wp.tid()
        fi = ref_frame_idx[env_idx]
        frame_idx = fi + (step_idx + 1) * future_frame_step
        if wp.static(future_obs_phc_v6):
            frame_idx = fi + 1 + step_idx * future_frame_step
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


def _make_single_live_future_observation_kernel(ctx: _FutureObservationKernelContext):
    motion_frame_count = ctx.motion_frame_count
    ee_body_idx = ctx.ee_body_idx
    future_pose_body_idx = ctx.future_pose_body_idx
    num_future_frames = ctx.num_future_frames
    future_frame_step = ctx.future_frame_step
    loop_cycle_table_size = ctx.loop_cycle_table_size
    future_obs_includes_root = ctx.future_obs_includes_root
    future_obs_includes_ee = ctx.future_obs_includes_ee
    future_obs_includes_pose = ctx.future_obs_includes_pose
    num_ee = ctx.num_ee
    num_future_pose_bodies = ctx.num_future_pose_bodies
    root_pos_ofs = ctx.root_pos_ofs
    root_rot_ofs = ctx.root_rot_ofs
    ee_pos_ofs = ctx.ee_pos_ofs
    ee_rot_ofs = ctx.ee_rot_ofs
    ee_lv_ofs = ctx.ee_lv_ofs
    ee_av_ofs = ctx.ee_av_ofs
    pose_pos_ofs = ctx.pose_pos_ofs
    pose_rot_ofs = ctx.pose_rot_ofs
    loop_root_q = ctx.loop_root_q
    loop_root_t = ctx.loop_root_t

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def write_future_pose_observation_single_live_kernel(
        motion_qpos: wp.array2d(dtype=float),
        ref_body_state: wp.array3d(dtype=float),
        ref_frame_idx: wp.array(dtype=int),
        loop_cycle_quat: wp.array2d(dtype=float),
        loop_cycle_pos: wp.array2d(dtype=float),
        xpos: wp.array2d(dtype=wp.vec3),
        xquat: wp.array2d(dtype=wp.quat),
        cvel: wp.array2d(dtype=wp.spatial_vector),
        root_pos_xz_buf: wp.array2d(dtype=float),
        root_yaw_inv_buf: wp.array2d(dtype=float),
        obs: wp.array2d(dtype=float),
    ):
        env_idx = wp.tid()
        fi = ref_frame_idx[env_idx]
        yaw_inv = normalize_quat_wp(read_quat_from_2d(root_yaw_inv_buf, env_idx, 0))
        root_pos_xz = wp.vec3(
            root_pos_xz_buf[env_idx, 0],
            root_pos_xz_buf[env_idx, 1],
            root_pos_xz_buf[env_idx, 2],
        )

        for step_idx in range(wp.static(num_future_frames)):
            frame_idx = fi + (step_idx + 1) * future_frame_step
            local_idx = loop_local_frame(frame_idx, motion_frame_count)
            cycles = loop_cycle_count(frame_idx, motion_frame_count)

            use_cycle_table = cycles < wp.static(loop_cycle_table_size)
            cycle_q = wp.quat(1.0, 0.0, 0.0, 0.0)
            cycle_t = wp.vec3(0.0, 0.0, 0.0)
            if use_cycle_table:
                cycle_q = read_quat_from_2d(loop_cycle_quat, cycles, 0)
                cycle_t = read_vec3_from_2d(loop_cycle_pos, cycles, 0)

            if wp.static(future_obs_includes_root):
                ref_root_pos = wp.vec3(
                    motion_qpos[local_idx, 0],
                    motion_qpos[local_idx, 1],
                    motion_qpos[local_idx, 2],
                )
                ref_root_q = normalize_quat_wp(
                    read_quat_from_2d(motion_qpos, local_idx, 3)
                )
                if use_cycle_table:
                    ref_root_pos = rot_vec_quat(ref_root_pos, cycle_q) + cycle_t
                    ref_root_q = normalize_quat_wp(mul_quat(cycle_q, ref_root_q))
                else:
                    ref_root_pos = apply_loop_transform_vec(
                        ref_root_pos, loop_root_q, loop_root_t, cycles
                    )
                    ref_root_q = normalize_quat_wp(
                        apply_loop_transform_quat(ref_root_q, loop_root_q, cycles)
                    )
                ref_root_yaw_q = heading_yaw_quat(ref_root_q)
                ref_root_pos_rel = rot_vec_quat(ref_root_pos - root_pos_xz, yaw_inv)
                ref_root_rot_rel = normalize_quat_wp(mul_quat(yaw_inv, ref_root_yaw_q))
                write_vec3(
                    obs,
                    env_idx,
                    root_pos_ofs + step_idx * 3,
                    ref_root_pos_rel,
                    1.0,
                )
                write_ortho6d(
                    obs, env_idx, root_rot_ofs + step_idx * 6, ref_root_rot_rel
                )

            if wp.static(future_obs_includes_ee):
                for ee_idx in range(wp.static(num_ee)):
                    body_idx = wp.static(ee_body_idx[ee_idx])
                    flat_idx = step_idx * wp.static(num_ee) + ee_idx

                    ref_ee_pos = read_body_state_pos(
                        ref_body_state, local_idx, body_idx
                    )
                    ref_ee_q = read_body_state_quat(ref_body_state, local_idx, body_idx)
                    ref_ee_lv = read_body_state_linvel(
                        ref_body_state, local_idx, body_idx
                    )
                    ref_ee_av = read_body_state_angvel(
                        ref_body_state, local_idx, body_idx
                    )

                    if use_cycle_table:
                        ref_ee_pos = rot_vec_quat(ref_ee_pos, cycle_q) + cycle_t
                        ref_ee_q = normalize_quat_wp(mul_quat(cycle_q, ref_ee_q))
                        ref_ee_lv = rot_vec_quat(ref_ee_lv, cycle_q)
                        ref_ee_av = rot_vec_quat(ref_ee_av, cycle_q)
                    else:
                        ref_ee_pos = apply_loop_transform_vec(
                            ref_ee_pos, loop_root_q, loop_root_t, cycles
                        )
                        ref_ee_q = normalize_quat_wp(
                            apply_loop_transform_quat(ref_ee_q, loop_root_q, cycles)
                        )
                        ref_ee_lv = apply_loop_rotation_vec(
                            ref_ee_lv, loop_root_q, cycles
                        )
                        ref_ee_av = apply_loop_rotation_vec(
                            ref_ee_av, loop_root_q, cycles
                        )

                    ref_ee_pos_rel = rot_vec_quat(ref_ee_pos - root_pos_xz, yaw_inv)
                    ref_ee_rot_rel = normalize_quat_wp(mul_quat(yaw_inv, ref_ee_q))
                    ref_ee_lv_rel = rot_vec_quat(ref_ee_lv, yaw_inv)
                    ref_ee_av_rel = rot_vec_quat(ref_ee_av, yaw_inv)

                    write_vec3(
                        obs,
                        env_idx,
                        ee_pos_ofs + flat_idx * 3,
                        ref_ee_pos_rel,
                        1.0,
                    )
                    write_ortho6d(
                        obs, env_idx, ee_rot_ofs + flat_idx * 6, ref_ee_rot_rel
                    )
                    write_vec3(
                        obs,
                        env_idx,
                        ee_lv_ofs + flat_idx * 3,
                        ref_ee_lv_rel,
                        0.1,
                    )
                    write_vec3(
                        obs,
                        env_idx,
                        ee_av_ofs + flat_idx * 3,
                        ref_ee_av_rel,
                        0.1,
                    )

            if wp.static(future_obs_includes_pose):
                for pose_idx in range(wp.static(num_future_pose_bodies)):
                    body_idx = wp.static(future_pose_body_idx[pose_idx])
                    flat_idx = step_idx * wp.static(num_future_pose_bodies) + pose_idx

                    ref_body_pos = read_body_state_pos(
                        ref_body_state, local_idx, body_idx
                    )
                    ref_body_q = read_body_state_quat(
                        ref_body_state, local_idx, body_idx
                    )

                    if use_cycle_table:
                        ref_body_pos = rot_vec_quat(ref_body_pos, cycle_q) + cycle_t
                        ref_body_q = normalize_quat_wp(mul_quat(cycle_q, ref_body_q))
                    else:
                        ref_body_pos = apply_loop_transform_vec(
                            ref_body_pos, loop_root_q, loop_root_t, cycles
                        )
                        ref_body_q = normalize_quat_wp(
                            apply_loop_transform_quat(ref_body_q, loop_root_q, cycles)
                        )

                    ref_body_pos_rel = rot_vec_quat(ref_body_pos - root_pos_xz, yaw_inv)
                    ref_body_rot_rel = normalize_quat_wp(mul_quat(yaw_inv, ref_body_q))

                    write_vec3(
                        obs,
                        env_idx,
                        pose_pos_ofs + flat_idx * 3,
                        ref_body_pos_rel,
                        1.0,
                    )
                    write_ortho6d(
                        obs,
                        env_idx,
                        pose_rot_ofs + flat_idx * 6,
                        ref_body_rot_rel,
                    )

    return write_future_pose_observation_single_live_kernel


def _make_multi_live_future_observation_kernel(ctx: _FutureObservationKernelContext):
    root_body_idx = ctx.root_body_idx
    ee_body_idx = ctx.ee_body_idx
    future_pose_body_idx = ctx.future_pose_body_idx
    num_future_frames = ctx.num_future_frames
    future_frame_step = ctx.future_frame_step
    future_obs_includes_root = ctx.future_obs_includes_root
    future_obs_includes_ee = ctx.future_obs_includes_ee
    future_obs_includes_pose = ctx.future_obs_includes_pose
    future_obs_phc_v6 = ctx.future_obs_phc_v6
    future_obs_phc_v6_features = ctx.future_obs_phc_v6_features
    num_ee = ctx.num_ee
    num_future_pose_bodies = ctx.num_future_pose_bodies
    root_pos_ofs = ctx.root_pos_ofs
    root_rot_ofs = ctx.root_rot_ofs
    ee_pos_ofs = ctx.ee_pos_ofs
    ee_rot_ofs = ctx.ee_rot_ofs
    ee_lv_ofs = ctx.ee_lv_ofs
    ee_av_ofs = ctx.ee_av_ofs
    pose_base_ofs = ctx.pose_base_ofs
    pose_pos_ofs = ctx.pose_pos_ofs
    pose_rot_ofs = ctx.pose_rot_ofs
    phc_step_stride = ctx.phc_step_stride

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def write_future_pose_observation_multi_live_kernel(
        motion_qpos: wp.array2d(dtype=float),
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
        root_pos_xz_buf: wp.array2d(dtype=float),
        root_yaw_inv_buf: wp.array2d(dtype=float),
        obs: wp.array2d(dtype=float),
    ):
        env_idx = wp.tid()
        clip_idx = motion_clip_idx[env_idx]
        fi = ref_frame_idx[env_idx]
        yaw_inv = normalize_quat_wp(read_quat_from_2d(root_yaw_inv_buf, env_idx, 0))
        heading = quat_inv(yaw_inv)
        root_pos_xz = wp.vec3(
            root_pos_xz_buf[env_idx, 0],
            root_pos_xz_buf[env_idx, 1],
            root_pos_xz_buf[env_idx, 2],
        )
        root_pos = xpos[env_idx, root_body_idx]
        loop_root_q_multi = normalize_quat_wp(
            read_quat_from_2d(clip_loop_root_quat, clip_idx, 0)
        )
        loop_root_t_multi = read_vec3_from_2d(clip_loop_root_pos, clip_idx, 0)

        for step_idx in range(wp.static(num_future_frames)):
            frame_idx = fi + (step_idx + 1) * future_frame_step
            if wp.static(future_obs_phc_v6):
                frame_idx = fi + 1 + step_idx * future_frame_step
            local_idx = clip_local_frame(frame_idx, clip_idx, clip_length)
            cycles = clip_cycle_count(frame_idx, clip_idx, clip_length)
            global_idx = clip_start[clip_idx] + local_idx

            ref_root_pos = wp.vec3(
                motion_qpos[global_idx, 0],
                motion_qpos[global_idx, 1],
                motion_qpos[global_idx, 2],
            )
            ref_root_q = normalize_quat_wp(
                read_quat_from_2d(motion_qpos, global_idx, 3)
            )
            ref_root_pos = apply_loop_transform_vec(
                ref_root_pos, loop_root_q_multi, loop_root_t_multi, cycles
            )
            ref_root_q = normalize_quat_wp(
                apply_loop_transform_quat(ref_root_q, loop_root_q_multi, cycles)
            )
            ref_root_yaw_q = heading_yaw_quat(ref_root_q)

            if wp.static(future_obs_includes_root):
                ref_root_pos_rel = rot_vec_quat(ref_root_pos - root_pos_xz, yaw_inv)
                ref_root_rot_rel = normalize_quat_wp(mul_quat(yaw_inv, ref_root_yaw_q))
                write_vec3(
                    obs,
                    env_idx,
                    root_pos_ofs + step_idx * 3,
                    ref_root_pos_rel,
                    1.0,
                )
                write_ortho6d(
                    obs, env_idx, root_rot_ofs + step_idx * 6, ref_root_rot_rel
                )

            if wp.static(future_obs_includes_ee):
                for ee_idx in range(wp.static(num_ee)):
                    body_idx = wp.static(ee_body_idx[ee_idx])
                    flat_idx = step_idx * wp.static(num_ee) + ee_idx

                    ref_ee_pos = read_body_state_pos(
                        ref_body_state, global_idx, body_idx
                    )
                    ref_ee_q = read_body_state_quat(
                        ref_body_state, global_idx, body_idx
                    )
                    ref_ee_lv = read_body_state_linvel(
                        ref_body_state, global_idx, body_idx
                    )
                    ref_ee_av = read_body_state_angvel(
                        ref_body_state, global_idx, body_idx
                    )

                    ref_ee_pos = apply_loop_transform_vec(
                        ref_ee_pos, loop_root_q_multi, loop_root_t_multi, cycles
                    )
                    ref_ee_q = normalize_quat_wp(
                        apply_loop_transform_quat(ref_ee_q, loop_root_q_multi, cycles)
                    )
                    ref_ee_lv = apply_loop_rotation_vec(
                        ref_ee_lv, loop_root_q_multi, cycles
                    )
                    ref_ee_av = apply_loop_rotation_vec(
                        ref_ee_av, loop_root_q_multi, cycles
                    )

                    ref_ee_pos_rel = rot_vec_quat(ref_ee_pos - root_pos_xz, yaw_inv)
                    ref_ee_rot_rel = normalize_quat_wp(mul_quat(yaw_inv, ref_ee_q))
                    ref_ee_lv_rel = rot_vec_quat(ref_ee_lv, yaw_inv)
                    ref_ee_av_rel = rot_vec_quat(ref_ee_av, yaw_inv)

                    write_vec3(
                        obs, env_idx, ee_pos_ofs + flat_idx * 3, ref_ee_pos_rel, 1.0
                    )
                    write_ortho6d(
                        obs, env_idx, ee_rot_ofs + flat_idx * 6, ref_ee_rot_rel
                    )
                    write_vec3(
                        obs, env_idx, ee_lv_ofs + flat_idx * 3, ref_ee_lv_rel, 0.1
                    )
                    write_vec3(
                        obs, env_idx, ee_av_ofs + flat_idx * 3, ref_ee_av_rel, 0.1
                    )

            if wp.static(future_obs_includes_pose):
                for pose_idx in range(wp.static(num_future_pose_bodies)):
                    body_idx = wp.static(future_pose_body_idx[pose_idx])
                    flat_idx = step_idx * wp.static(num_future_pose_bodies) + pose_idx

                    ref_body_pos = read_body_state_pos(
                        ref_body_state, global_idx, body_idx
                    )
                    ref_body_q = read_body_state_quat(
                        ref_body_state, global_idx, body_idx
                    )
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

                    if wp.static(future_obs_phc_v6_features):
                        sim_body_pos = xpos[env_idx, body_idx]
                        sim_body_q = normalize_quat_wp(xquat[env_idx, body_idx])
                        sim_body_cv = cvel[env_idx, body_idx]
                        sim_body_lv = wp.vec3(
                            sim_body_cv[3], sim_body_cv[4], sim_body_cv[5]
                        )
                        sim_body_av = wp.vec3(
                            sim_body_cv[0], sim_body_cv[1], sim_body_cv[2]
                        )

                        diff_pos = rot_vec_quat(ref_body_pos - sim_body_pos, yaw_inv)
                        diff_rot = normalize_quat_wp(
                            mul_quat(ref_body_q, quat_inv(sim_body_q))
                        )
                        diff_rot = normalize_quat_wp(
                            mul_quat(mul_quat(yaw_inv, diff_rot), heading)
                        )
                        diff_lv = rot_vec_quat(ref_body_lv - sim_body_lv, yaw_inv)
                        diff_av = rot_vec_quat(ref_body_av - sim_body_av, yaw_inv)
                        local_ref_pos = rot_vec_quat(ref_body_pos - root_pos, yaw_inv)
                        local_ref_rot = normalize_quat_wp(mul_quat(yaw_inv, ref_body_q))
                        body_ofs = (
                            pose_base_ofs + step_idx * phc_step_stride + pose_idx * 24
                        )

                        write_vec3(obs, env_idx, body_ofs + 0, diff_pos, 1.0)
                        write_tan_norm(obs, env_idx, body_ofs + 3, diff_rot)
                        write_vec3(obs, env_idx, body_ofs + 9, diff_lv, 0.1)
                        write_vec3(obs, env_idx, body_ofs + 12, diff_av, 0.1)
                        write_vec3(obs, env_idx, body_ofs + 15, local_ref_pos, 1.0)
                        write_tan_norm(obs, env_idx, body_ofs + 18, local_ref_rot)
                    else:
                        ref_body_pos_rel = rot_vec_quat(
                            ref_body_pos - root_pos_xz, yaw_inv
                        )
                        ref_body_rot_rel = normalize_quat_wp(
                            mul_quat(yaw_inv, ref_body_q)
                        )

                        write_vec3(
                            obs,
                            env_idx,
                            pose_pos_ofs + flat_idx * 3,
                            ref_body_pos_rel,
                            1.0,
                        )
                        write_ortho6d(
                            obs,
                            env_idx,
                            pose_rot_ofs + flat_idx * 6,
                            ref_body_rot_rel,
                        )

    return write_future_pose_observation_multi_live_kernel
