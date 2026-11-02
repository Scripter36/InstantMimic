from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp
from tqdm.auto import tqdm

from instantmimic.envs.config_access import get_block, get_value
from instantmimic.envs.snu.tasks.snu_imitation import model_metadata
from instantmimic.envs.snu.tasks.snu_imitation.adaptive_sampling import (
    resolve_adaptive_sampling_chunk_size,
    resolve_adaptive_sampling_method,
)
from instantmimic.envs.snu.tasks.snu_imitation.termination_kernels import (
    is_phc_termination_body_name,
)
from instantmimic.motion import build_motion_clip_metadata
from instantmimic.motion.loader import (
    estimate_loaded_motion_frame_count,
    load_bvh_file,
)

__all__ = (
    "load_motion_reference_data",
    "load_single_motion_reference_data",
    "resolve_motion_sources",
    "set_motion_clip_metadata",
    "compute_reset_start_horizon_steps",
    "clip_frame_slice",
    "validate_shared_motion_reference_source",
    "maybe_smooth_reward_reference_velocity",
)


def _split_reference_body_state_views(
    reference_body_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos, quat = _split_reference_motion_pose_views(reference_body_state)
    return (
        pos,
        quat,
        reference_body_state[..., 7:10],
    )


def _split_reference_motion_pose_views(
    reference_body_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        reference_body_state[..., :3],
        reference_body_state[..., 3:7],
    )


def _copy_shared_motion_reference_data(
    target_env: Any,
    source_env: Any,
    *,
    clone_tensors: bool = False,
) -> None:
    required_fields = (
        "_multi_motion_enabled",
        "motion_qpos",
        "motion_qvel",
        "reference_body_state",
        "ref_com",
        "motion_qpos_mask",
        "motion_qvel_mask",
        "motion_body_mask",
        "motion_frame_count",
        "loop_root_quat",
        "loop_root_pos",
        "motion_clip_start",
        "motion_clip_length",
        "motion_clip_frame_counts",
        "motion_clip_reset_start_count",
        "motion_frame_to_clip_idx",
        "motion_clip_unit_start",
        "motion_clip_unit_count",
        "motion_clip_count",
        "motion_clip_loop_root_quat",
        "motion_clip_loop_root_pos",
        "motion_xpos",
        "motion_xquat",
        "reference_body_pos",
        "reference_body_quat",
        "reference_body_linvel",
        "loop_root_quat_tuple",
        "loop_root_pos_tuple",
        "loop_root_quat_wp",
        "loop_root_pos_wp",
        "reward_body_idx",
        "phc_termination_body_idx",
        "reward_motion_qpos",
        "reward_motion_qvel",
        "reward_reference_body_state",
        "reward_ref_com",
        "reward_motion_frame_count",
        "reward_motion_clip_start",
        "reward_motion_clip_length",
        "reward_motion_clip_loop_root_quat",
        "reward_motion_clip_loop_root_pos",
    )
    for field_name in required_fields:
        if not hasattr(source_env, field_name):
            raise AttributeError(
                f"Shared motion source is missing required field {field_name!r}."
            )
    if not clone_tensors:
        for field_name in required_fields:
            setattr(target_env, field_name, getattr(source_env, field_name))

        if hasattr(source_env, "motion_ctrl"):
            target_env.motion_ctrl = source_env.motion_ctrl
        return

    def _clone_field(field_name: str) -> torch.Tensor:
        value = getattr(source_env, field_name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Shared motion source field {field_name!r} must be a torch.Tensor."
            )
        return value.detach().clone()

    target_env._multi_motion_enabled = bool(source_env._multi_motion_enabled)
    target_env.motion_qpos = _clone_field("motion_qpos")
    target_env.motion_qvel = _clone_field("motion_qvel")
    target_env.reference_body_state = _clone_field("reference_body_state")
    target_env.ref_com = _clone_field("ref_com")
    target_env.motion_qpos_mask = _clone_field("motion_qpos_mask")
    target_env.motion_qvel_mask = _clone_field("motion_qvel_mask")
    target_env.motion_body_mask = _clone_field("motion_body_mask")
    target_env.motion_frame_count = int(source_env.motion_frame_count)
    target_env.loop_root_quat = _clone_field("loop_root_quat")
    target_env.loop_root_pos = _clone_field("loop_root_pos")
    target_env.motion_clip_start = _clone_field("motion_clip_start")
    target_env.motion_clip_length = _clone_field("motion_clip_length")
    target_env.motion_clip_frame_counts = tuple(source_env.motion_clip_frame_counts)
    target_env.motion_clip_reset_start_count = _clone_field(
        "motion_clip_reset_start_count"
    )
    target_env.motion_frame_to_clip_idx = _clone_field("motion_frame_to_clip_idx")
    target_env.motion_clip_unit_start = _clone_field("motion_clip_unit_start")
    target_env.motion_clip_unit_count = _clone_field("motion_clip_unit_count")
    target_env.motion_clip_count = int(source_env.motion_clip_count)
    target_env.motion_clip_loop_root_quat = _clone_field("motion_clip_loop_root_quat")
    target_env.motion_clip_loop_root_pos = _clone_field("motion_clip_loop_root_pos")
    target_env.loop_root_quat_tuple = tuple(
        float(value) for value in source_env.loop_root_quat_tuple
    )
    target_env.loop_root_pos_tuple = tuple(
        float(value) for value in source_env.loop_root_pos_tuple
    )

    target_env.motion_xpos, target_env.motion_xquat = (
        _split_reference_motion_pose_views(target_env.reference_body_state)
    )
    (
        target_env.reference_body_pos,
        target_env.reference_body_quat,
        target_env.reference_body_linvel,
    ) = _split_reference_body_state_views(target_env.reference_body_state)
    # These Warp values are immutable reference transforms in the baseline
    # runtime. Preserve their identity while cloning the mutable tensor graph.
    target_env.loop_root_quat_wp = source_env.loop_root_quat_wp
    target_env.loop_root_pos_wp = source_env.loop_root_pos_wp
    target_env.reward_body_idx = _clone_field("reward_body_idx")
    target_env.phc_termination_body_idx = _clone_field("phc_termination_body_idx")
    target_env.reward_motion_qpos = (
        target_env.motion_qpos
        if source_env.reward_motion_qpos is source_env.motion_qpos
        else _clone_field("reward_motion_qpos")
    )
    target_env.reward_motion_qvel = (
        target_env.motion_qvel
        if source_env.reward_motion_qvel is source_env.motion_qvel
        else _clone_field("reward_motion_qvel")
    )
    target_env.reward_reference_body_state = (
        target_env.reference_body_state
        if source_env.reward_reference_body_state is source_env.reference_body_state
        else _clone_field("reward_reference_body_state")
    )
    target_env.reward_ref_com = (
        target_env.ref_com
        if source_env.reward_ref_com is source_env.ref_com
        else _clone_field("reward_ref_com")
    )
    target_env.reward_motion_frame_count = int(source_env.reward_motion_frame_count)
    target_env.reward_motion_clip_start = (
        target_env.motion_clip_start
        if source_env.reward_motion_clip_start is source_env.motion_clip_start
        else _clone_field("reward_motion_clip_start")
    )
    target_env.reward_motion_clip_length = (
        target_env.motion_clip_length
        if source_env.reward_motion_clip_length is source_env.motion_clip_length
        else _clone_field("reward_motion_clip_length")
    )
    target_env.reward_motion_clip_loop_root_quat = (
        target_env.motion_clip_loop_root_quat
        if source_env.reward_motion_clip_loop_root_quat
        is source_env.motion_clip_loop_root_quat
        else _clone_field("reward_motion_clip_loop_root_quat")
    )
    target_env.reward_motion_clip_loop_root_pos = (
        target_env.motion_clip_loop_root_pos
        if source_env.reward_motion_clip_loop_root_pos
        is source_env.motion_clip_loop_root_pos
        else _clone_field("reward_motion_clip_loop_root_pos")
    )
    if hasattr(source_env, "motion_ctrl"):
        target_env.motion_ctrl = _clone_field("motion_ctrl")


def _torch_devices_compatible(
    source_device: torch.device, target_device: torch.device
) -> bool:
    source = torch.device(source_device)
    target = torch.device(target_device)
    if source.type != target.type:
        return False
    if source.type != "cuda":
        return source.index == target.index
    if source.index is None or target.index is None:
        return True
    return source.index == target.index


def _allocate_motion_reference_buffers(
    *,
    total_frame_count: int,
    nq: int,
    nv: int,
    nbody: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    total_frame_count = int(total_frame_count)
    if total_frame_count <= 0:
        raise ValueError("total_frame_count must be positive.")
    return {
        "motion_qpos": torch.empty(
            total_frame_count, int(nq), device=device, dtype=torch.float32
        ),
        "motion_qvel": torch.empty(
            total_frame_count, int(nv), device=device, dtype=torch.float32
        ),
        "reference_body_state": torch.empty(
            total_frame_count, int(nbody), 13, device=device, dtype=torch.float32
        ),
        "ref_com": torch.empty(
            total_frame_count, 3, device=device, dtype=torch.float32
        ),
    }


def _copy_motion_reference_clip_into_buffers(
    buffers: dict[str, torch.Tensor],
    motion_data: dict[str, Any],
    *,
    frame_slice: slice,
    root_body_idx: int,
) -> None:
    start = int(frame_slice.start or 0)
    stop = int(frame_slice.stop or start)
    expected_frame_count = stop - start
    frame_count = int(motion_data["frame_count"])
    if frame_count != expected_frame_count:
        raise ValueError(
            "Motion frame count mismatch while copying into preallocated buffers: "
            f"expected {expected_frame_count}, got {frame_count}."
        )

    buffers["motion_qpos"][frame_slice].copy_(motion_data["qpos"])
    buffers["motion_qvel"][frame_slice].copy_(motion_data["qvel"])
    buffers["reference_body_state"][frame_slice].copy_(
        motion_data["reference_body_state"]
    )
    buffers["ref_com"][frame_slice].copy_(
        motion_data["subtree_com"][:, int(root_body_idx)]
    )


def _iter_motion_files_with_progress(motion_files: list[str]):
    if len(motion_files) <= 1:
        return motion_files
    return tqdm(
        motion_files,
        desc="Load motions",
        unit="clip",
        dynamic_ncols=True,
    )


def _make_binomial_kernel(
    radius: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    kernel = torch.tensor([1.0], device=device, dtype=dtype)
    for _ in range(radius):
        padded = torch.nn.functional.pad(kernel, (1, 1))
        kernel = padded[:-1] + padded[1:]
    kernel = kernel / kernel.sum().clamp_min(1.0e-8)
    return kernel


def _cyclic_smooth_first_dim(data: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    if data.shape[0] <= 1 or kernel.numel() <= 1:
        return data
    radius = kernel.numel() // 2
    out = torch.zeros_like(data)
    for ki in range(int(kernel.numel())):
        shift = ki - radius
        out = out + torch.roll(data, shifts=shift, dims=0) * kernel[ki]
    return out


def resolve_motion_sources(self) -> list[str]:
    resolved = list(self.motion_settings.resolve_sources())
    self._resolved_motion_sources = tuple(resolved)
    return resolved


def set_motion_clip_metadata(
    self,
    *,
    clip_frame_counts: list[int],
    clip_loop_root_quat: list[torch.Tensor],
    clip_loop_root_pos: list[torch.Tensor],
) -> None:
    self._reset_start_horizon_steps = compute_reset_start_horizon_steps(self)
    adaptive_sampling = get_block(self.cfg, "adaptive_sampling")
    adaptive_method = resolve_adaptive_sampling_method(adaptive_sampling)
    metadata = build_motion_clip_metadata(
        clip_frame_counts=clip_frame_counts,
        chunk_size=resolve_adaptive_sampling_chunk_size(
            self.cfg,
            method=adaptive_method,
        ),
        device=self.device,
        reset_start_horizon_steps=int(self._reset_start_horizon_steps),
    )
    self.motion_clip_start = metadata.clip_start
    self.motion_clip_length = metadata.clip_length
    self.motion_clip_frame_counts = tuple(int(count) for count in clip_frame_counts)
    self.motion_clip_reset_start_count = metadata.clip_reset_start_count
    self.motion_frame_to_clip_idx = metadata.frame_to_clip_idx
    self.motion_clip_unit_start = metadata.clip_unit_start
    self.motion_clip_unit_count = metadata.clip_unit_count
    self.motion_clip_total_unit_count = int(metadata.total_unit_count)
    self.motion_clip_count = int(len(clip_frame_counts))
    self.motion_clip_loop_root_quat = torch.stack(
        [
            tensor.to(device=self.device, dtype=torch.float32)
            for tensor in clip_loop_root_quat
        ],
        dim=0,
    ).contiguous()
    self.motion_clip_loop_root_pos = torch.stack(
        [
            tensor.to(device=self.device, dtype=torch.float32)
            for tensor in clip_loop_root_pos
        ],
        dim=0,
    ).contiguous()


def _set_reward_motion_clip_metadata(
    self,
    *,
    clip_frame_counts: list[int],
    clip_loop_root_quat: list[torch.Tensor],
    clip_loop_root_pos: list[torch.Tensor],
) -> None:
    metadata = build_motion_clip_metadata(
        clip_frame_counts=clip_frame_counts,
        chunk_size=1,
        device=self.device,
        reset_start_horizon_steps=1,
    )
    self.reward_motion_clip_start = metadata.clip_start
    self.reward_motion_clip_length = metadata.clip_length
    self.reward_motion_clip_loop_root_quat = torch.stack(
        [
            tensor.to(device=self.device, dtype=torch.float32)
            for tensor in clip_loop_root_quat
        ],
        dim=0,
    ).contiguous()
    self.reward_motion_clip_loop_root_pos = torch.stack(
        [
            tensor.to(device=self.device, dtype=torch.float32)
            for tensor in clip_loop_root_pos
        ],
        dim=0,
    ).contiguous()


def compute_reset_start_horizon_steps(self) -> int:
    future_horizon = (
        self.observation_settings.future_frames
        * self.observation_settings.future_frame_step
    )
    reset_cfg = get_block(self.cfg, "reset")
    reset_start_horizon_s = get_value(reset_cfg, "start_horizon_s", None)
    loop = self.motion_settings.loop
    loop_reset_range = not self.termination_settings.motion_end and (
        loop.smooth or loop.velocity_from_first_frame or loop.velocity_extrapolate
    )
    if reset_start_horizon_s is None and loop_reset_range:
        return 1
    if reset_start_horizon_s is None:
        base_horizon_steps = int(self.max_episode_length)
    else:
        horizon_dt = max(float(self.step_dt), 1.0e-9)
        base_horizon_steps = int(np.ceil(float(reset_start_horizon_s) / horizon_dt))
    return max(1, base_horizon_steps + max(0, future_horizon))


def clip_frame_slice(self, clip_idx: int) -> slice:
    start = int(self.motion_clip_start[clip_idx].item())
    frame_count = int(self.motion_clip_length[clip_idx].item())
    return slice(start, start + frame_count)


def load_single_motion_reference_data(
    self,
    motion_file: str,
    *,
    target_dt: float | None = None,
    motion_loader_kwargs: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    resolved_target_dt = self.step_dt if target_dt is None else float(target_dt)
    settings = self.motion_settings
    loop = settings.loop
    return load_bvh_file(
        motion_file=motion_file,
        motion_scale=settings.scale,
        motion_offset=list(settings.offset),
        bvh_map=settings.bvh_map,
        device=self.device,
        model=self.model,
        target_dt=resolved_target_dt,
        loop_boundary_vel_from_first_frame=loop.velocity_from_first_frame,
        loop_boundary_vel_extrapolate=loop.velocity_extrapolate,
        smooth_cycle=loop.smooth,
        smooth_cycle_blend_frames=loop.blend_frames,
        smooth_cycle_mode=loop.mode,
        root_auto_calibration=settings.root_auto_calibration,
        rest_bodies=settings.rest_bodies,
        cache_dir=settings.cache_directory,
        cache_mode=settings.cache_mode,
        **(
            model_metadata.get_motion_loader_kwargs(self)
            if motion_loader_kwargs is None
            else motion_loader_kwargs
        ),
    )


def validate_shared_motion_reference_source(
    self,
    source_env: Any,
    *,
    motion_files: list[str],
) -> None:
    source_multi_motion = bool(getattr(source_env, "_multi_motion_enabled", False))
    if source_multi_motion != self._multi_motion_enabled:
        raise ValueError(
            "Shared motion source multi-motion mode does not match target env: "
            f"source={source_multi_motion}, target={self._multi_motion_enabled}."
        )
    if self._multi_motion_enabled:
        source_motion_files = list(
            getattr(source_env, "_resolved_motion_sources", ())
            or source_env.motion_settings.resolve_sources()
        )
        if source_motion_files and source_motion_files != list(motion_files):
            raise ValueError(
                "Shared motion source motion sources do not match target env."
            )
    else:
        target_motion_file = str(motion_files[0])
        source_motion_files = list(
            getattr(source_env, "_resolved_motion_sources", ())
            or source_env.motion_settings.resolve_sources()
        )
        source_motion_file = str(source_motion_files[0])
        if source_motion_file != target_motion_file:
            raise ValueError(
                "Shared motion source motion source does not match target env."
            )

    source_qpos = getattr(source_env, "motion_qpos", None)
    source_qvel = getattr(source_env, "motion_qvel", None)
    source_body_state = getattr(source_env, "reference_body_state", None)
    if not isinstance(source_qpos, torch.Tensor):
        raise TypeError("Shared motion source motion_qpos must be a torch.Tensor.")
    if not isinstance(source_qvel, torch.Tensor):
        raise TypeError("Shared motion source motion_qvel must be a torch.Tensor.")
    if not isinstance(source_body_state, torch.Tensor):
        raise TypeError(
            "Shared motion source reference_body_state must be a torch.Tensor."
        )
    if int(source_qpos.shape[1]) != int(self.model.nq):
        raise ValueError(
            "Shared motion source qpos width does not match target model: "
            f"source={int(source_qpos.shape[1])}, target={int(self.model.nq)}."
        )
    if int(source_qvel.shape[1]) != int(self.model.nv):
        raise ValueError(
            "Shared motion source qvel width does not match target model: "
            f"source={int(source_qvel.shape[1])}, target={int(self.model.nv)}."
        )
    if int(source_body_state.shape[1]) != int(self.model.nbody):
        raise ValueError(
            "Shared motion source body count does not match target model: "
            f"source={int(source_body_state.shape[1])}, target={int(self.model.nbody)}."
        )
    if not _torch_devices_compatible(
        torch.device(source_qpos.device), torch.device(self.device)
    ):
        raise ValueError(
            "Shared motion source tensor device does not match target env device: "
            f"source={source_qpos.device}, target={self.device}."
        )
    source_step_dt = getattr(source_env, "step_dt", None)
    if (
        source_step_dt is not None
        and abs(float(source_step_dt) - float(self.step_dt)) > 1.0e-9
    ):
        raise ValueError(
            "Shared motion source step_dt does not match target env: "
            f"source={float(source_step_dt)}, target={float(self.step_dt)}."
        )


def _alias_policy_rate_reward_reference(self) -> None:
    self.reward_motion_qpos = self.motion_qpos
    self.reward_motion_qvel = self.motion_qvel
    self.reward_reference_body_state = self.reference_body_state
    self.reward_ref_com = self.ref_com
    self.reward_motion_frame_count = int(self.motion_frame_count)
    self.reward_motion_clip_start = self.motion_clip_start
    self.reward_motion_clip_length = self.motion_clip_length
    self.reward_motion_clip_loop_root_quat = self.motion_clip_loop_root_quat
    self.reward_motion_clip_loop_root_pos = self.motion_clip_loop_root_pos


def _load_substep_reward_reference_data(
    self,
    *,
    motion_files: list[str],
) -> None:
    target_dt = float(self.physics_dt)
    if not self._multi_motion_enabled:
        motion_file = motion_files[0]
        motion_data = self._load_single_motion_reference_data(
            motion_file,
            target_dt=target_dt,
        )
        self.reward_motion_qpos = motion_data["qpos"]
        self.reward_motion_qvel = motion_data["qvel"]
        self.reward_reference_body_state = motion_data["reference_body_state"]
        self.reward_ref_com = motion_data["subtree_com"][:, self.root_body_idx]
        self.reward_motion_frame_count = int(motion_data["frame_count"])
        _set_reward_motion_clip_metadata(
            self,
            clip_frame_counts=[int(motion_data["frame_count"])],
            clip_loop_root_quat=[motion_data["loop_root_quat"]],
            clip_loop_root_pos=[motion_data["loop_root_pos"]],
        )
        return

    clip_frame_counts = [
        estimate_loaded_motion_frame_count(
            motion_file,
            target_dt=target_dt,
            smooth_cycle=self.motion_settings.loop.smooth,
        )
        for motion_file in motion_files
    ]
    buffers = _allocate_motion_reference_buffers(
        total_frame_count=sum(clip_frame_counts),
        nq=self.model.nq,
        nv=self.model.nv,
        nbody=self.model.nbody,
        device=self.device,
    )
    clip_loop_root_quat = []
    clip_loop_root_pos = []
    actual_clip_frame_counts = []
    cursor = 0

    for motion_file, expected_frame_count in zip(
        _iter_motion_files_with_progress(motion_files),
        clip_frame_counts,
        strict=False,
    ):
        motion_data = self._load_single_motion_reference_data(
            motion_file,
            target_dt=target_dt,
        )
        frame_count = int(motion_data["frame_count"])
        if frame_count != int(expected_frame_count):
            raise ValueError(
                "Estimated substep reward motion frame count mismatch for "
                f"{motion_file}: expected {expected_frame_count}, got {frame_count}."
            )
        frame_slice = slice(cursor, cursor + frame_count)
        _copy_motion_reference_clip_into_buffers(
            buffers,
            motion_data,
            frame_slice=frame_slice,
            root_body_idx=int(self.root_body_idx),
        )
        clip_loop_root_quat.append(motion_data["loop_root_quat"])
        clip_loop_root_pos.append(motion_data["loop_root_pos"])
        actual_clip_frame_counts.append(frame_count)
        cursor += frame_count
        del motion_data

    if cursor != int(buffers["motion_qpos"].shape[0]):
        raise ValueError(
            "Loaded substep reward motion frame total does not match "
            f"preallocated buffers: loaded={cursor}, "
            f"allocated={int(buffers['motion_qpos'].shape[0])}."
        )

    self.reward_motion_qpos = buffers["motion_qpos"]
    self.reward_motion_qvel = buffers["motion_qvel"]
    self.reward_reference_body_state = buffers["reference_body_state"]
    self.reward_ref_com = buffers["ref_com"]
    self.reward_motion_frame_count = int(self.reward_motion_qpos.shape[0])
    _set_reward_motion_clip_metadata(
        self,
        clip_frame_counts=actual_clip_frame_counts,
        clip_loop_root_quat=clip_loop_root_quat,
        clip_loop_root_pos=clip_loop_root_pos,
    )


def load_motion_reference_data(
    self,
    *,
    shared_motion_reference: Any | None = None,
    clone_shared_motion_reference: bool = False,
):
    motion_files = resolve_motion_sources(self)
    self._multi_motion_enabled = len(motion_files) > 1

    if shared_motion_reference is not None:
        validate_shared_motion_reference_source(
            self,
            shared_motion_reference,
            motion_files=motion_files,
        )
        _copy_shared_motion_reference_data(
            self,
            shared_motion_reference,
            clone_tensors=bool(clone_shared_motion_reference),
        )
        self._reset_start_horizon_steps = compute_reset_start_horizon_steps(self)
        return

    if not self._multi_motion_enabled:
        motion_file = motion_files[0]
        motion_data = self._load_single_motion_reference_data(motion_file)
        self.motion_qpos = motion_data["qpos"]
        self.motion_qvel = motion_data["qvel"]
        self.reference_body_state = motion_data["reference_body_state"]
        self.motion_qpos_mask = motion_data["qpos_mask"]
        self.motion_qvel_mask = motion_data["qvel_mask"]
        self.motion_body_mask = motion_data["body_mask"]
        self.motion_frame_count = motion_data["frame_count"]
        self.loop_root_quat = motion_data["loop_root_quat"]
        self.loop_root_pos = motion_data["loop_root_pos"]
        self.ref_com = motion_data["subtree_com"][:, self.root_body_idx]
        set_motion_clip_metadata(
            self,
            clip_frame_counts=[int(motion_data["frame_count"])],
            clip_loop_root_quat=[motion_data["loop_root_quat"]],
            clip_loop_root_pos=[motion_data["loop_root_pos"]],
        )
    else:
        clip_frame_counts = [
            estimate_loaded_motion_frame_count(
                motion_file,
                target_dt=float(self.step_dt),
                smooth_cycle=self.motion_settings.loop.smooth,
            )
            for motion_file in motion_files
        ]
        buffers = _allocate_motion_reference_buffers(
            total_frame_count=sum(clip_frame_counts),
            nq=self.model.nq,
            nv=self.model.nv,
            nbody=self.model.nbody,
            device=self.device,
        )
        first_qpos_mask = None
        first_qvel_mask = None
        first_body_mask = None
        clip_loop_root_quat = []
        clip_loop_root_pos = []
        actual_clip_frame_counts = []
        cursor = 0

        for motion_file, expected_frame_count in zip(
            _iter_motion_files_with_progress(motion_files),
            clip_frame_counts,
            strict=False,
        ):
            motion_data = self._load_single_motion_reference_data(motion_file)
            frame_count = int(motion_data["frame_count"])
            if frame_count != int(expected_frame_count):
                raise ValueError(
                    "Estimated motion frame count mismatch for "
                    f"{motion_file}: expected {expected_frame_count}, got {frame_count}."
                )
            if first_qpos_mask is None:
                first_qpos_mask = motion_data["qpos_mask"]
                first_qvel_mask = motion_data["qvel_mask"]
                first_body_mask = motion_data["body_mask"]
            elif not torch.equal(motion_data["qpos_mask"], first_qpos_mask):
                raise ValueError(
                    "Multi-motion requires identical qpos masks across clips."
                )
            elif not torch.equal(motion_data["qvel_mask"], first_qvel_mask):
                raise ValueError(
                    "Multi-motion requires identical qvel masks across clips."
                )
            elif not torch.equal(motion_data["body_mask"], first_body_mask):
                raise ValueError(
                    "Multi-motion requires identical body masks across clips."
                )

            frame_slice = slice(cursor, cursor + frame_count)
            _copy_motion_reference_clip_into_buffers(
                buffers,
                motion_data,
                frame_slice=frame_slice,
                root_body_idx=int(self.root_body_idx),
            )
            clip_loop_root_quat.append(motion_data["loop_root_quat"])
            clip_loop_root_pos.append(motion_data["loop_root_pos"])
            actual_clip_frame_counts.append(frame_count)
            cursor += frame_count
            del motion_data

        if (
            first_qpos_mask is None
            or first_qvel_mask is None
            or first_body_mask is None
        ):
            raise ValueError("Multi-motion requires at least one motion clip.")
        if cursor != int(buffers["motion_qpos"].shape[0]):
            raise ValueError(
                "Loaded motion frame total does not match preallocated buffers: "
                f"loaded={cursor}, allocated={int(buffers['motion_qpos'].shape[0])}."
            )

        self.motion_qpos = buffers["motion_qpos"]
        self.motion_qvel = buffers["motion_qvel"]
        self.reference_body_state = buffers["reference_body_state"]
        self.ref_com = buffers["ref_com"]
        self.motion_qpos_mask = first_qpos_mask
        self.motion_qvel_mask = first_qvel_mask
        self.motion_body_mask = first_body_mask
        self.motion_frame_count = int(self.motion_qpos.shape[0])
        self.loop_root_quat = clip_loop_root_quat[0]
        self.loop_root_pos = clip_loop_root_pos[0]
        set_motion_clip_metadata(
            self,
            clip_frame_counts=actual_clip_frame_counts,
            clip_loop_root_quat=clip_loop_root_quat,
            clip_loop_root_pos=clip_loop_root_pos,
        )

    tracking_body_names = tuple(self.motion_settings.tracking_body_names)
    if tracking_body_names:
        tracking_mask = torch.zeros_like(self.motion_body_mask, dtype=torch.bool)
        for body_name in tracking_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(
                    f"motion.tracking_body_names contains unknown body {body_name!r}."
                )
            if not bool(self.motion_body_mask[body_id]):
                raise ValueError(
                    "motion.tracking_body_names requests unavailable reference body "
                    f"{body_name!r}."
                )
            tracking_mask[body_id] = True
        self.motion_body_mask = tracking_mask

    self.motion_xpos, self.motion_xquat = _split_reference_motion_pose_views(
        self.reference_body_state
    )
    (
        self.reference_body_pos,
        self.reference_body_quat,
        self.reference_body_linvel,
    ) = _split_reference_body_state_views(self.reference_body_state)
    self.loop_root_quat_tuple = (
        float(self.loop_root_quat[0]),
        float(self.loop_root_quat[1]),
        float(self.loop_root_quat[2]),
        float(self.loop_root_quat[3]),
    )
    self.loop_root_pos_tuple = (
        float(self.loop_root_pos[0]),
        float(self.loop_root_pos[1]),
        float(self.loop_root_pos[2]),
    )
    self.loop_root_quat_wp = wp.quat(*self.loop_root_quat_tuple)
    self.loop_root_pos_wp = wp.vec3(*self.loop_root_pos_tuple)

    reward_body_mask = model_metadata.reward_body_mask(self, self.motion_body_mask)
    self.reward_body_idx = torch.nonzero(reward_body_mask, as_tuple=False).flatten()
    reward_body_idx_list = [
        int(body_id) for body_id in self.reward_body_idx.detach().cpu().tolist()
    ]
    self.reward_body_idx_tuple = tuple(reward_body_idx_list)
    phc_termination_body_mask = reward_body_mask.clone()
    for body_id in reward_body_idx_list:
        body_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)
        )
        if not is_phc_termination_body_name(body_name):
            phc_termination_body_mask[int(body_id)] = False
    self.phc_termination_body_idx = torch.nonzero(
        phc_termination_body_mask, as_tuple=False
    ).flatten()

    if self._use_reference_action:
        self.motion_ctrl = self._compute_motion_ctrl()

    if self.reward_settings.substep_mean:
        _load_substep_reward_reference_data(self, motion_files=motion_files)
    else:
        _alias_policy_rate_reward_reference(self)
    maybe_smooth_reward_reference_velocity(self)


def maybe_smooth_reward_reference_velocity(self):
    reward_vel_smooth_radius = int(self.reward_settings.velocity_smoothing_radius)
    if reward_vel_smooth_radius <= 0 or self.reward_motion_frame_count <= 1:
        return

    reward_vel_smooth_mode = self.reward_settings.velocity_smoothing_mode
    reward_vel_kernel = _make_binomial_kernel(
        reward_vel_smooth_radius,
        device=self.reward_motion_qvel.device,
        dtype=self.reward_motion_qvel.dtype,
    )
    source_reward_qvel = self.reward_motion_qvel
    self.reward_reference_body_state = self.reward_reference_body_state.clone()

    def _smooth_per_clip_in_place(
        target_tensor: torch.Tensor,
        source_slice: slice,
        tensor_slice: tuple[slice | int, ...],
    ) -> None:
        index = (source_slice,) + tensor_slice
        target_tensor[index] = _cyclic_smooth_first_dim(
            target_tensor[index], reward_vel_kernel
        )

    if reward_vel_smooth_mode in ("angular_only", "angvel", "angular"):
        self.reward_motion_qvel = source_reward_qvel.clone()
        if self._multi_motion_enabled:
            for clip_idx in range(self.motion_clip_count):
                start = int(self.reward_motion_clip_start[clip_idx].item())
                frame_count = int(self.reward_motion_clip_length[clip_idx].item())
                clip_slice = slice(start, start + frame_count)
                _smooth_per_clip_in_place(
                    self.reward_motion_qvel,
                    clip_slice,
                    (slice(3, None),),
                )
                _smooth_per_clip_in_place(
                    self.reward_reference_body_state,
                    clip_slice,
                    (slice(None), slice(10, 13)),
                )
        else:
            self.reward_motion_qvel[:, 3:] = _cyclic_smooth_first_dim(
                self.reward_motion_qvel[:, 3:], reward_vel_kernel
            )
            self.reward_reference_body_state[:, :, 10:13] = _cyclic_smooth_first_dim(
                self.reward_reference_body_state[:, :, 10:13], reward_vel_kernel
            )
        return

    if self._multi_motion_enabled:
        self.reward_motion_qvel = source_reward_qvel.clone()
        for clip_idx in range(self.motion_clip_count):
            start = int(self.reward_motion_clip_start[clip_idx].item())
            frame_count = int(self.reward_motion_clip_length[clip_idx].item())
            clip_slice = slice(start, start + frame_count)
            self.reward_motion_qvel[clip_slice] = _cyclic_smooth_first_dim(
                self.reward_motion_qvel[clip_slice], reward_vel_kernel
            )
            self.reward_reference_body_state[clip_slice, :, 7:13] = (
                _cyclic_smooth_first_dim(
                    self.reward_reference_body_state[clip_slice, :, 7:13],
                    reward_vel_kernel,
                )
            )
        self.reward_motion_qvel = self.reward_motion_qvel.contiguous()
        return

    self.reward_motion_qvel = _cyclic_smooth_first_dim(
        source_reward_qvel, reward_vel_kernel
    ).contiguous()
    self.reward_reference_body_state[:, :, 7:13] = _cyclic_smooth_first_dim(
        self.reward_reference_body_state[:, :, 7:13], reward_vel_kernel
    )
