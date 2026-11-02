from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp

from instantmimic.envs.snu.tasks.snu_imitation import render_runtime

_LOGGER = logging.getLogger(__name__)
__all__ = (
    "NaNDebugStepLifecycle",
    "initialize_nan_debug",
    "nan_debug_prepare_window",
    "nan_debug_capture_state",
    "nan_debug_nonfinite_mask",
    "nan_debug_nonfinite_breakdown",
    "nan_debug_record_nonfinite",
    "nan_debug_record_raw_reward_nonfinite",
    "nan_debug_record_action_nonfinite",
    "sanitize_action_input",
    "nan_debug_is_current_stream_capturing",
    "nan_debug_check_and_fail",
    "nan_debug_dump_videos",
    "nan_debug_motion_clip_path",
    "nan_debug_write_ref_trace",
    "nan_debug_write_state_npz",
    "ensure_nan_debug_renderer",
    "render_nan_debug_frame",
    "nan_debug_write_video_for_env",
)


class NaNDebugStepLifecycle:
    """NaN diagnostics attached to the framework step lifecycle."""

    def __init__(self, env: Any) -> None:
        self._env = env

    def before_step(self, step_io: Any) -> None:
        del step_io
        if not nan_debug_is_current_stream_capturing(self._env):
            nan_debug_prepare_window(self._env)
        nan_debug_capture_state(self._env)

    def after_step(self, step_io: Any) -> None:
        del step_io
        nan_debug_record_nonfinite(self._env)
        if not nan_debug_is_current_stream_capturing(self._env):
            nan_debug_check_and_fail(self._env)

    def prepare_capture_window(self) -> None:
        nan_debug_prepare_window(self._env)

    def finalize_capture_window(self) -> None:
        nan_debug_check_and_fail(self._env)


def initialize_nan_debug(self):
    settings = self.nan_debug_settings
    self._nan_debug_enabled = settings.enabled
    self._nan_debug_buffer_size = settings.buffer_size
    self._nan_debug_max_env_videos = settings.max_env_videos
    self._nan_debug_raise_on_nan = settings.raise_on_nan
    output_dir_name = settings.output_directory
    output_root = (
        Path(self.cfg.log_dir)
        if self.cfg.log_dir is not None and str(self.cfg.log_dir) != ""
        else Path.cwd()
    )
    self._nan_debug_output_dir = output_root / output_dir_name

    self._nan_debug_triggered = False
    self._nan_debug_history_head = 0
    self._nan_debug_history_count = 0

    self._nan_debug_renderer = None
    self._nan_debug_camera = None
    self._nan_debug_render_width = 0
    self._nan_debug_render_height = 0
    self._nan_debug_render_model = None
    self._nan_debug_render_data = None
    self._nan_debug_ghost_mocap_ids = None
    self._nan_debug_ghost_sim_body_ids = None
    self._nan_debug_ghost_offset_np = None
    self._nan_debug_last_video_dump_errors: list[str] = []
    self._nan_debug_last_state_paths: list[str] = []

    self._nan_debug_qpos_view = None
    self._nan_debug_qvel_view = None
    self._nan_debug_qpos_hist = None
    self._nan_debug_qvel_hist = None
    self._nan_debug_ref_idx_hist = None
    self._nan_debug_motion_clip_idx_hist = None
    self._nan_debug_step_bad_mask = None
    self._nan_debug_seen_bad_mask = None
    self._nan_debug_seen_bad_breakdown = None

    self._register_step_lifecycle_component(NaNDebugStepLifecycle(self))

    if not self._nan_debug_enabled:
        return

    self._nan_debug_qpos_view = wp.to_torch(self.mjw_data.qpos)
    self._nan_debug_qvel_view = wp.to_torch(self.mjw_data.qvel)
    self._nan_debug_qpos_hist = torch.zeros(
        self._nan_debug_buffer_size,
        self.num_envs,
        self.model.nq,
        device=self.device,
        dtype=self._nan_debug_qpos_view.dtype,
    )
    self._nan_debug_qvel_hist = torch.zeros(
        self._nan_debug_buffer_size,
        self.num_envs,
        self.model.nv,
        device=self.device,
        dtype=self._nan_debug_qvel_view.dtype,
    )
    self._nan_debug_ref_idx_hist = torch.zeros(
        self._nan_debug_buffer_size,
        self.num_envs,
        device=self.device,
        dtype=torch.int32,
    )
    self._nan_debug_motion_clip_idx_hist = torch.zeros(
        self._nan_debug_buffer_size,
        self.num_envs,
        device=self.device,
        dtype=torch.int32,
    )
    self._nan_debug_step_bad_mask = torch.zeros(
        self.num_envs,
        device=self.device,
        dtype=torch.bool,
    )
    self._nan_debug_seen_bad_mask = torch.zeros(
        self.num_envs,
        device=self.device,
        dtype=torch.bool,
    )
    self._nan_debug_seen_bad_breakdown = {
        key: torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        for key in ("reward", "qpos", "qvel", "obs", "action")
    }


def nan_debug_prepare_window(self):
    if not self._nan_debug_enabled or self._nan_debug_triggered:
        return
    self._nan_debug_seen_bad_mask.zero_()
    if self._nan_debug_seen_bad_breakdown is not None:
        for mask in self._nan_debug_seen_bad_breakdown.values():
            mask.zero_()


def nan_debug_capture_state(self):
    if not self._nan_debug_enabled or self._nan_debug_triggered:
        return
    slot = self._nan_debug_history_head
    self._nan_debug_qpos_hist[slot].copy_(self._nan_debug_qpos_view)
    self._nan_debug_qvel_hist[slot].copy_(self._nan_debug_qvel_view)
    self._nan_debug_ref_idx_hist[slot].copy_(self.ref_frame_idx)
    self._nan_debug_motion_clip_idx_hist[slot].copy_(self.motion_clip_idx)
    self._nan_debug_history_head = (slot + 1) % self._nan_debug_buffer_size
    if self._nan_debug_history_count < self._nan_debug_buffer_size:
        self._nan_debug_history_count += 1


def nan_debug_nonfinite_mask(self) -> torch.Tensor:
    reward_bad = ~torch.isfinite(self.reward_buf)
    qpos_bad = ~torch.isfinite(self._nan_debug_qpos_view).all(dim=1)
    qvel_bad = ~torch.isfinite(self._nan_debug_qvel_view).all(dim=1)
    obs_tensor = (
        self.obs_buf["policy"]
        if isinstance(getattr(self, "obs_buf", None), dict) and "policy" in self.obs_buf
        else self._obs_buf
    )
    obs_bad = ~torch.isfinite(obs_tensor).all(dim=1)
    action_bad = ~torch.isfinite(self.actions).all(dim=1)
    return reward_bad | qpos_bad | qvel_bad | obs_bad | action_bad


def nan_debug_nonfinite_breakdown(self) -> dict[str, torch.Tensor]:
    obs_tensor = (
        self.obs_buf["policy"]
        if isinstance(getattr(self, "obs_buf", None), dict) and "policy" in self.obs_buf
        else self._obs_buf
    )
    return {
        "reward": ~torch.isfinite(self.reward_buf),
        "qpos": ~torch.isfinite(self._nan_debug_qpos_view).all(dim=1),
        "qvel": ~torch.isfinite(self._nan_debug_qvel_view).all(dim=1),
        "obs": ~torch.isfinite(obs_tensor).all(dim=1),
        "action": ~torch.isfinite(self.actions).all(dim=1),
    }


def nan_debug_record_nonfinite(self):
    if not getattr(self, "_nan_debug_enabled", False) or getattr(
        self, "_nan_debug_triggered", False
    ):
        return

    breakdown = nan_debug_nonfinite_breakdown(self)
    self._nan_debug_step_bad_mask.copy_(breakdown["reward"])
    self._nan_debug_step_bad_mask.logical_or_(breakdown["qpos"])
    self._nan_debug_step_bad_mask.logical_or_(breakdown["qvel"])
    self._nan_debug_step_bad_mask.logical_or_(breakdown["obs"])
    self._nan_debug_seen_bad_mask.logical_or_(self._nan_debug_step_bad_mask)
    if self._nan_debug_seen_bad_breakdown is not None:
        for key, mask in breakdown.items():
            self._nan_debug_seen_bad_breakdown[key].logical_or_(mask)


def nan_debug_record_raw_reward_nonfinite(self):
    if not getattr(self, "_nan_debug_enabled", False) or getattr(
        self, "_nan_debug_triggered", False
    ):
        return
    if (
        self._nan_debug_step_bad_mask is None
        or self._nan_debug_seen_bad_mask is None
        or self._nan_debug_seen_bad_breakdown is None
    ):
        return

    self._nan_debug_step_bad_mask.copy_(torch.isfinite(self.reward_buf))
    self._nan_debug_step_bad_mask.logical_not_()
    self._nan_debug_seen_bad_mask.logical_or_(self._nan_debug_step_bad_mask)
    self._nan_debug_seen_bad_breakdown["reward"].logical_or_(
        self._nan_debug_step_bad_mask
    )


def nan_debug_record_action_nonfinite(self, actions: torch.Tensor):
    if not bool(getattr(self, "_nan_debug_enabled", False)) or bool(
        getattr(self, "_nan_debug_triggered", False)
    ):
        return
    if (
        self._nan_debug_step_bad_mask is None
        or self._nan_debug_seen_bad_mask is None
        or self._nan_debug_seen_bad_breakdown is None
        or "action" not in self._nan_debug_seen_bad_breakdown
    ):
        return

    self._nan_debug_step_bad_mask.copy_(torch.isfinite(actions).all(dim=1))
    self._nan_debug_step_bad_mask.logical_not_()
    self._nan_debug_seen_bad_mask.logical_or_(self._nan_debug_step_bad_mask)
    self._nan_debug_seen_bad_breakdown["action"].logical_or_(
        self._nan_debug_step_bad_mask
    )


def sanitize_action_input(self, actions: torch.Tensor) -> torch.Tensor:
    nan_debug_record_action_nonfinite(self, actions)
    terminate_on_nonfinite = self._terminate_on_nonfinite_enabled()
    if not (terminate_on_nonfinite or bool(getattr(self, "_nan_debug_enabled", False))):
        return actions
    return torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)


def nan_debug_is_current_stream_capturing(self) -> bool:
    return bool(self.device.type == "cuda" and torch.cuda.is_current_stream_capturing())


def nan_debug_check_and_fail(self):
    if not self._nan_debug_enabled or self._nan_debug_triggered:
        return

    bad_mask = self._nan_debug_seen_bad_mask
    breakdown = self._nan_debug_seen_bad_breakdown
    if bad_mask is None or breakdown is None:
        bad_mask = nan_debug_nonfinite_mask(self)
        breakdown = nan_debug_nonfinite_breakdown(self)
    if not bool(bad_mask.any().item()):
        return

    self._nan_debug_triggered = True
    bad_env_ids = torch.nonzero(bad_mask, as_tuple=False).squeeze(-1)
    env_ids = [int(v) for v in bad_env_ids.detach().cpu().tolist()]
    breakdown_counts = {key: int(mask.sum().item()) for key, mask in breakdown.items()}
    env_ref_idx = {env_id: int(self.ref_frame_idx[env_id].item()) for env_id in env_ids}
    env_motion_clip_idx = {
        env_id: int(self.motion_clip_idx[env_id].item()) for env_id in env_ids
    }
    env_motion_clip_path = {
        env_id: nan_debug_motion_clip_path(self, env_motion_clip_idx[env_id])
        for env_id in env_ids
    }
    saved_paths: list[str] = []
    ref_trace_paths: list[str] = []
    dump_error = ""
    try:
        saved_paths, ref_trace_paths = nan_debug_dump_videos(self, env_ids)
        video_dump_errors = getattr(self, "_nan_debug_last_video_dump_errors", [])
        if video_dump_errors:
            dump_error = f" (video dump failed: {'; '.join(video_dump_errors)})"
    except Exception as exc:  # noqa: BLE001 - debug artifact failure must not hide non-finite report.
        dump_error = f" (nan debug dump failed: {exc})"

    step_idx = int(self.common_step_counter[0].item())
    self.extras["nan_debug"] = {
        "step": step_idx,
        "env_ids": env_ids,
        "ref_frame_idx": env_ref_idx,
        "motion_clip_idx": env_motion_clip_idx,
        "motion_clip_path": env_motion_clip_path,
        "video_paths": saved_paths,
        "ref_trace_paths": ref_trace_paths,
        "state_paths": list(getattr(self, "_nan_debug_last_state_paths", [])),
        "video_dump_error": dump_error,
    }

    msg = (
        "Non-finite values detected in reward/state buffers. "
        f"step={step_idx}, counts={breakdown_counts}, env_ids={env_ids}, "
        f"motion_clip_idx={env_motion_clip_idx}, ref_frame_idx={env_ref_idx}, "
        f"videos={saved_paths}, traces={ref_trace_paths}, "
        f"states={getattr(self, '_nan_debug_last_state_paths', [])}{dump_error}"
    )
    if self._nan_debug_raise_on_nan:
        raise RuntimeError(msg)
    _LOGGER.warning("nan_debug: %s", msg)


def nan_debug_dump_videos(self, env_ids: list[int]) -> tuple[list[str], list[str]]:
    if self._nan_debug_history_count <= 0:
        return [], []

    self._nan_debug_output_dir.mkdir(parents=True, exist_ok=True)
    max_videos = min(len(env_ids), self._nan_debug_max_env_videos)
    selected_envs = env_ids[:max_videos]
    fps = max(1, int(round(1.0 / self.step_dt)))
    step_idx = int(self.common_step_counter[0].item())

    start_slot = (
        self._nan_debug_history_head - self._nan_debug_history_count
    ) % self._nan_debug_buffer_size
    slots = [
        (start_slot + i) % self._nan_debug_buffer_size
        for i in range(self._nan_debug_history_count)
    ]

    saved_paths: list[str] = []
    ref_trace_paths: list[str] = []
    self._nan_debug_last_video_dump_errors = []
    self._nan_debug_last_state_paths = []
    for env_id in selected_envs:
        ref_idx_now = int(self.ref_frame_idx[env_id].item())
        clip_idx_now = int(self.motion_clip_idx[env_id].item())
        artifact_stem = (
            self._nan_debug_output_dir
            / f"nan_step{step_idx:08d}_env{env_id}_clip{clip_idx_now}_ref{ref_idx_now}"
        )
        ref_trace_path = artifact_stem.with_suffix(".ref_idx.txt")
        nan_debug_write_ref_trace(
            self,
            ref_trace_path,
            env_id,
            slots,
            ref_idx_now,
            clip_idx_now,
        )
        ref_trace_paths.append(ref_trace_path.as_posix())
        state_path = artifact_stem.with_suffix(".state.npz")
        nan_debug_write_state_npz(
            self,
            state_path,
            env_id,
            slots,
            ref_idx_now,
            clip_idx_now,
        )
        self._nan_debug_last_state_paths.append(state_path.as_posix())

        video_path = artifact_stem.with_suffix(".mp4")
        try:
            nan_debug_write_video_for_env(self, video_path, env_id, slots, fps)
        except Exception as exc:  # noqa: BLE001 - keep collecting other debug artifacts.
            self._nan_debug_last_video_dump_errors.append(f"env{env_id}: {exc}")
        else:
            saved_paths.append(video_path.as_posix())

    return saved_paths, ref_trace_paths


def nan_debug_motion_clip_path(self, clip_idx: int) -> str:
    motion_sources = tuple(getattr(self, "_resolved_motion_sources", ()))
    if 0 <= clip_idx < len(motion_sources):
        return motion_sources[clip_idx]
    return motion_sources[0] if motion_sources else ""


def nan_debug_write_ref_trace(
    self,
    trace_path: Path,
    env_id: int,
    slots: list[int],
    ref_idx_now: int,
    clip_idx_now: int,
):
    with trace_path.open("w", encoding="utf-8") as f:
        f.write(f"env_id={env_id}\n")
        f.write(f"motion_clip_idx={clip_idx_now}\n")
        f.write(f"motion_clip_path={nan_debug_motion_clip_path(self, clip_idx_now)}\n")
        f.write(f"nan_ref_frame_idx={ref_idx_now}\n")
        f.write("history_index\tmotion_clip_idx\tref_frame_idx\tmotion_clip_path\n")
        for i, slot in enumerate(slots):
            hist_ref = int(self._nan_debug_ref_idx_hist[slot, env_id].item())
            hist_clip = int(self._nan_debug_motion_clip_idx_hist[slot, env_id].item())
            f.write(
                f"{i}\t{hist_clip}\t{hist_ref}\t"
                f"{nan_debug_motion_clip_path(self, hist_clip)}\n"
            )


def nan_debug_write_state_npz(
    self,
    state_path: Path,
    env_id: int,
    slots: list[int],
    ref_idx_now: int,
    clip_idx_now: int,
):
    qpos_hist = self._nan_debug_qpos_hist[slots, env_id].detach().cpu().numpy()
    qvel_hist = self._nan_debug_qvel_hist[slots, env_id].detach().cpu().numpy()
    ref_idx_hist = self._nan_debug_ref_idx_hist[slots, env_id].detach().cpu().numpy()
    clip_idx_hist = (
        self._nan_debug_motion_clip_idx_hist[slots, env_id].detach().cpu().numpy()
    )
    np.savez_compressed(
        state_path,
        env_id=np.asarray(env_id, dtype=np.int32),
        step=np.asarray(int(self.common_step_counter[0].item()), dtype=np.int32),
        nan_ref_frame_idx=np.asarray(ref_idx_now, dtype=np.int32),
        nan_motion_clip_idx=np.asarray(clip_idx_now, dtype=np.int32),
        nan_motion_clip_path=np.asarray(nan_debug_motion_clip_path(self, clip_idx_now)),
        qpos=qpos_hist,
        qvel=qvel_hist,
        ref_frame_idx=ref_idx_hist,
        motion_clip_idx=clip_idx_hist,
    )


def ensure_nan_debug_renderer(self):
    if self._nan_debug_renderer is not None:
        return
    gl_backend = os.environ.get("MUJOCO_GL", "").lower()
    if not os.environ.get("DISPLAY") and gl_backend not in {"egl", "osmesa"}:
        raise RuntimeError(
            "Headless nan_debug video rendering requires MUJOCO_GL=egl "
            "or MUJOCO_GL=osmesa to be set before starting Python."
        )
    ghost_bundle = render_runtime.build_reference_ghost_render_bundle(self)
    self._nan_debug_render_model = ghost_bundle.model
    self._nan_debug_render_data = ghost_bundle.data
    self._nan_debug_ghost_mocap_ids = ghost_bundle.ghost_mocap_ids
    self._nan_debug_ghost_sim_body_ids = ghost_bundle.ghost_sim_body_ids
    self._nan_debug_ghost_offset_np = ghost_bundle.ghost_offset

    offscreen_w = int(self._nan_debug_render_model.vis.global_.offwidth)
    offscreen_h = int(self._nan_debug_render_model.vis.global_.offheight)
    self._nan_debug_render_width = max(
        1, min(int(self.cfg.viewer.resolution[0]), offscreen_w)
    )
    self._nan_debug_render_height = max(
        1, min(int(self.cfg.viewer.resolution[1]), offscreen_h)
    )
    self._nan_debug_renderer = mujoco.Renderer(
        self._nan_debug_render_model,
        height=self._nan_debug_render_height,
        width=self._nan_debug_render_width,
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = int(self.torso_body_idx)
    camera.distance = 3.0
    camera.elevation = -20.0
    camera.azimuth = 135.0
    self._nan_debug_camera = camera


def render_nan_debug_frame(
    self,
    qpos_np: np.ndarray,
    qvel_np: np.ndarray,
    *,
    ref_idx: int,
    motion_clip_idx: int,
) -> np.ndarray:
    ensure_nan_debug_renderer(self)
    render_data = self._nan_debug_render_data
    render_data.qpos[:] = qpos_np[: self._nan_debug_render_model.nq]
    render_data.qvel[:] = qvel_np[: self._nan_debug_render_model.nv]
    render_runtime.set_reference_ghost_pose_on_render_data(
        self,
        render_data=render_data,
        ghost_mocap_ids=self._nan_debug_ghost_mocap_ids,
        ghost_sim_body_ids=self._nan_debug_ghost_sim_body_ids,
        ghost_offset_np=self._nan_debug_ghost_offset_np,
        ref_idx=ref_idx,
        motion_clip_idx=motion_clip_idx,
    )
    mujoco.mj_forward(self._nan_debug_render_model, render_data)
    self._nan_debug_renderer.update_scene(render_data, camera=self._nan_debug_camera)
    frame = self._nan_debug_renderer.render()
    if frame.size == 0:
        return np.zeros(
            (self._nan_debug_render_height, self._nan_debug_render_width, 3),
            dtype=np.uint8,
        )
    return frame[:, :, :3]


def nan_debug_write_video_for_env(
    self, video_path: Path, env_id: int, slots: list[int], fps: int
):
    import imageio.v2 as imageio

    ensure_nan_debug_renderer(self)
    wrote_frame = False
    with imageio.get_writer(video_path.as_posix(), fps=fps) as writer:
        for slot in slots:
            qpos = self._nan_debug_qpos_hist[slot, env_id].detach().cpu().numpy()
            qvel = self._nan_debug_qvel_hist[slot, env_id].detach().cpu().numpy()
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                continue
            ref_idx = int(self._nan_debug_ref_idx_hist[slot, env_id].item())
            motion_clip_idx = int(
                self._nan_debug_motion_clip_idx_hist[slot, env_id].item()
            )
            frame = render_nan_debug_frame(
                self,
                qpos.astype(np.float64, copy=False),
                qvel.astype(np.float64, copy=False),
                ref_idx=ref_idx,
                motion_clip_idx=motion_clip_idx,
            )
            writer.append_data(frame)
            wrote_frame = True

        if not wrote_frame:
            writer.append_data(
                np.zeros(
                    (
                        self._nan_debug_render_height,
                        self._nan_debug_render_width,
                        3,
                    ),
                    dtype=np.uint8,
                )
            )
