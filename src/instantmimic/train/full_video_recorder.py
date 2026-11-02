from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from instantmimic.play.render import (
    build_video_camera,
    build_video_tile_layout,
    open_streaming_video_writer,
    select_video_encoder_backend,
)
from instantmimic.rendering.mujoco_warp_playback import (
    MujocoWarpPlaybackRenderer,
    MujocoWarpRenderBatch,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainFullVideoFrameBatch:
    qpos: Any
    qvel: Any
    activation: Any | None = None
    mocap_pos: Any | None = None
    mocap_quat: Any | None = None


@dataclass(frozen=True, slots=True)
class TrainFullVideoConfig:
    encoder_backend: str = "auto"
    codec: str = "h264"
    tile_envs: int = 1
    tile_cols: int | None = None
    camera_azimuth_deg: float | None = None
    camera_elevation_deg: float | None = None
    camera_distance: float | None = None
    benchmark_sync: bool = False
    benchmark_warmup_frames: int = 5
    use_textures: bool = True
    floor_shader_mode: str = "texture"
    render_block_dim: int = 64
    lbs_mesh_mode: str = "off"
    lbs_mesh_bundle: str | None = None
    lbs_mesh_shading_mode: str = "smooth"
    lbs_activation_mapping: str | None = None
    activation_log_stats: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainFullVideoConfig:
        camera = value.get("camera", {})
        benchmark = value.get("benchmark")
        warp = value.get("warp", {})
        if not isinstance(camera, Mapping):
            raise TypeError("trainer.recording.full_run.camera must be a mapping")
        if benchmark is not None and not isinstance(benchmark, Mapping):
            raise TypeError("trainer.recording.full_run.benchmark must be a mapping")
        if not isinstance(warp, Mapping):
            raise TypeError("trainer.recording.full_run.warp must be a mapping")
        return cls(
            encoder_backend=str(value.get("encoder_backend", "auto")),
            codec=str(value.get("codec", "h264")),
            tile_envs=int(value.get("tile_envs", 1)),
            tile_cols=(int(value["tile_cols"]) if "tile_cols" in value else None),
            camera_azimuth_deg=(
                float(camera["azimuth_deg"]) if "azimuth_deg" in camera else None
            ),
            camera_elevation_deg=(
                float(camera["elevation_deg"]) if "elevation_deg" in camera else None
            ),
            camera_distance=(
                float(camera["distance"]) if "distance" in camera else None
            ),
            benchmark_sync=benchmark is not None,
            benchmark_warmup_frames=(
                int(benchmark.get("warmup_frames", 5)) if benchmark is not None else 5
            ),
            use_textures=bool(warp.get("use_textures", True)),
            floor_shader_mode=str(warp.get("floor_shader_mode", "texture")),
            render_block_dim=int(warp.get("render_block_dim", 64)),
            lbs_mesh_mode=str(warp.get("lbs_mesh_mode", "off")),
            lbs_mesh_bundle=(
                str(warp["lbs_mesh_bundle"])
                if warp.get("lbs_mesh_bundle") is not None
                else None
            ),
            lbs_mesh_shading_mode=str(warp.get("lbs_mesh_shading_mode", "smooth")),
            lbs_activation_mapping=(
                str(warp["lbs_activation_mapping"])
                if warp.get("lbs_activation_mapping") is not None
                else None
            ),
            activation_log_stats=bool(warp.get("activation_log_stats", False)),
        )


def _percentile_nearest(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return math.nan
    rank = int(math.ceil((float(percentile) / 100.0) * len(sorted_values)))
    return sorted_values[min(len(sorted_values) - 1, max(0, rank - 1))]


def _summarize_ms(values: list[float]) -> dict[str, float | int] | None:
    finite = sorted(v for v in values if math.isfinite(float(v)) and float(v) >= 0.0)
    if not finite:
        return None
    return {
        "count": len(finite),
        "mean": float(sum(finite) / len(finite)),
        "p50": float(_percentile_nearest(finite, 50.0)),
        "p95": float(_percentile_nearest(finite, 95.0)),
    }


def estimate_train_full_video_frame_count(
    *,
    train_timesteps: int,
    decimation: int,
    render_interval: int,
) -> int:
    timestep_count = max(0, int(train_timesteps))
    decimation_count = max(1, int(decimation))
    interval = max(1, int(render_interval))
    if interval >= decimation_count:
        return timestep_count
    return (timestep_count * decimation_count) // interval


def resolve_train_full_video_fps(env_raw: Any) -> float:
    cfg = env_raw.cfg
    decimation = max(1, int(cfg.decimation))
    render_interval = max(1, int(cfg.sim.render_interval))
    if render_interval < decimation:
        physics_dt = float(env_raw.physics_dt)
        return 1.0 / max(physics_dt * float(render_interval), 1.0e-6)
    step_dt = float(env_raw.step_dt)
    return 1.0 / max(step_dt, 1.0e-6)


def _frame_substep_indices_for_step(
    *,
    step_index: int,
    decimation: int,
    render_interval: int,
) -> tuple[int, ...]:
    decimation_count = max(1, int(decimation))
    interval = max(1, int(render_interval))
    if interval >= decimation_count:
        return (0,)
    first_physics_step = int(step_index) * decimation_count
    return tuple(
        local_idx
        for local_idx in range(decimation_count)
        if ((first_physics_step + local_idx + 1) % interval) == 0
    )


def select_train_full_video_frame_batch(
    runtime: Any,
    *,
    decimation: int,
    render_interval: int,
) -> TrainFullVideoFrameBatch:
    qpos_buf = runtime.train_full_video_qpos_buf
    qvel_buf = runtime.train_full_video_qvel_buf
    mocap_pos_buf = getattr(runtime, "train_full_video_mocap_pos_buf", None)
    mocap_quat_buf = getattr(runtime, "train_full_video_mocap_quat_buf", None)
    if mocap_pos_buf is None or mocap_quat_buf is None:
        mocap_pos_buf = None
        mocap_quat_buf = None
    activation_buf = getattr(runtime, "train_full_video_activation_buf", None)
    if (
        activation_buf is not None
        and hasattr(runtime, "train_full_video_activation_sample_count")
        and int(getattr(runtime, "train_full_video_activation_sample_count", 0)) <= 0
    ):
        activation_buf = None
    if qpos_buf is None or qvel_buf is None:
        raise RuntimeError("Train full-video capture buffers are not configured.")

    step_count = max(0, int(getattr(runtime, "train_full_video_step_count", 0)))
    substep_count = max(1, int(getattr(runtime, "train_full_video_substep_count", 1)))
    if step_count <= 0:
        return TrainFullVideoFrameBatch(
            qpos=qpos_buf[:0, 0],
            qvel=qvel_buf[:0, 0],
            activation=None if activation_buf is None else activation_buf[:0, 0],
            mocap_pos=None if mocap_pos_buf is None else mocap_pos_buf[:0, 0],
            mocap_quat=None if mocap_quat_buf is None else mocap_quat_buf[:0, 0],
        )

    if max(1, int(render_interval)) >= max(1, int(decimation)) or substep_count == 1:
        return TrainFullVideoFrameBatch(
            qpos=qpos_buf[:step_count, 0],
            qvel=qvel_buf[:step_count, 0],
            activation=None
            if activation_buf is None
            else activation_buf[:step_count, 0],
            mocap_pos=None if mocap_pos_buf is None else mocap_pos_buf[:step_count, 0],
            mocap_quat=None
            if mocap_quat_buf is None
            else mocap_quat_buf[:step_count, 0],
        )

    qpos_frames: list[torch.Tensor] = []
    qvel_frames: list[torch.Tensor] = []
    activation_frames: list[torch.Tensor] = []
    mocap_pos_frames: list[torch.Tensor] = []
    mocap_quat_frames: list[torch.Tensor] = []
    for step_index in range(step_count):
        indices = _frame_substep_indices_for_step(
            step_index=step_index,
            decimation=int(decimation),
            render_interval=int(render_interval),
        )
        for substep_idx in indices:
            if int(substep_idx) < substep_count:
                qpos_frames.append(qpos_buf[step_index, int(substep_idx)])
                qvel_frames.append(qvel_buf[step_index, int(substep_idx)])
                if activation_buf is not None:
                    activation_frames.append(
                        activation_buf[step_index, int(substep_idx)]
                    )
                if mocap_pos_buf is not None:
                    mocap_pos_frames.append(mocap_pos_buf[step_index, int(substep_idx)])
                if mocap_quat_buf is not None:
                    mocap_quat_frames.append(
                        mocap_quat_buf[step_index, int(substep_idx)]
                    )

    if not qpos_frames:
        return TrainFullVideoFrameBatch(
            qpos=qpos_buf[:0, 0],
            qvel=qvel_buf[:0, 0],
            activation=None if activation_buf is None else activation_buf[:0, 0],
            mocap_pos=None if mocap_pos_buf is None else mocap_pos_buf[:0, 0],
            mocap_quat=None if mocap_quat_buf is None else mocap_quat_buf[:0, 0],
        )
    return TrainFullVideoFrameBatch(
        qpos=torch.stack(qpos_frames, dim=0),
        qvel=torch.stack(qvel_frames, dim=0),
        activation=(
            None if activation_buf is None else torch.stack(activation_frames, dim=0)
        ),
        mocap_pos=(
            None if mocap_pos_buf is None else torch.stack(mocap_pos_frames, dim=0)
        ),
        mocap_quat=(
            None if mocap_quat_buf is None else torch.stack(mocap_quat_frames, dim=0)
        ),
    )


class TrainFullVideoRecorder:
    """Stream one tiled video from actual train rollout states."""

    def __init__(
        self,
        *,
        config: TrainFullVideoConfig,
        env: Any,
        env_raw: Any,
        log_dir: str | Path,
        rollout_steps: int,
        train_timesteps: int,
        activation_getter: Any | None = None,
    ) -> None:
        self.config = config
        self.env = env
        self.env_raw = env_raw
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.rollout_steps = max(1, int(rollout_steps))
        self.train_timesteps = max(0, int(train_timesteps))
        self.activation_getter = activation_getter
        self.frames_written = 0
        self.output_path: Path | None = None
        self._closed = False
        self._activation_stats_logged = False
        self._benchmark_sync_device = (
            torch.device("cuda")
            if config.benchmark_sync and torch.cuda.is_available()
            else None
        )
        self._benchmark_warmup_frames = max(
            0,
            int(config.benchmark_warmup_frames),
        )
        self._render_stage_timing_samples: dict[str, list[float]] = {}
        self._render_stage_timing_seen_frames = 0

        viewer_width, viewer_height = env_raw.cfg.viewer.resolution
        self.tile_layout = build_video_tile_layout(
            requested_envs=config.tile_envs,
            available_envs=int(getattr(env_raw, "num_envs", config.tile_envs)),
            requested_cols=config.tile_cols,
            tile_width=int(viewer_width),
            tile_height=int(viewer_height),
        )
        configure_capture = getattr(env_raw, "configure_train_full_video_capture", None)
        if not callable(configure_capture):
            raise RuntimeError(
                "trainer.recording.full_run requires an env that can configure "
                "train full-video capture buffers."
            )
        configure_capture(
            rollout_steps=self.rollout_steps,
            env_count=int(self.tile_layout.env_count),
        )

        self.expected_frame_count = estimate_train_full_video_frame_count(
            train_timesteps=self.train_timesteps,
            decimation=int(env_raw.cfg.decimation),
            render_interval=int(env_raw.cfg.sim.render_interval),
        )
        self.fps = resolve_train_full_video_fps(env_raw)
        self.encoder_backend = select_video_encoder_backend(
            backend=config.encoder_backend,
            codec=config.codec,
            take_video=True,
            video_render_backend="warp",
            output_width=int(self.tile_layout.output_width),
            output_height=int(self.tile_layout.output_height),
        )
        self.codec = config.codec.strip().lower()
        self.renderer = MujocoWarpPlaybackRenderer(
            env_raw=env_raw,
            model=getattr(env_raw, "render_model", env_raw.model),
            data=getattr(env_raw, "render_data", getattr(env_raw, "data", None)),
            width=int(viewer_width),
            height=int(viewer_height),
            camera=build_video_camera(
                getattr(env_raw, "camera", None),
                azimuth=config.camera_azimuth_deg,
                elevation=config.camera_elevation_deg,
                distance=config.camera_distance,
            ),
            overlay_renderer=None,
            tile_env_count=int(self.tile_layout.env_count),
            tile_cols=int(self.tile_layout.cols),
            use_textures=config.use_textures,
            floor_shader_mode=config.floor_shader_mode,
            render_block_dim=config.render_block_dim,
            lbs_mesh_mode=config.lbs_mesh_mode,
            lbs_mesh_bundle_path=config.lbs_mesh_bundle,
            lbs_mesh_shading_mode=config.lbs_mesh_shading_mode,
            lbs_activation_mapping_path=config.lbs_activation_mapping,
        )
        self.writer, self.output_path = open_streaming_video_writer(
            env=env,
            log_dir=self.log_dir,
            video_length=self.expected_frame_count,
            step_dt=float(env_raw.step_dt),
            backend=self.encoder_backend,
            codec=self.codec,
            output_dir=self.log_dir / "videos" / "train_full",
            output_width=int(self.tile_layout.output_width),
            output_height=int(self.tile_layout.output_height),
            fps=self.fps,
            filename_prefix="train_full",
        )
        _LOGGER.info(
            "Recording full train video with Warp tiles: path=%s fps=%.3f "
            "expected_frames=%d tiles=%d layout=%dx%d.",
            self.output_path.as_posix(),
            self.fps,
            self.expected_frame_count,
            int(self.tile_layout.env_count),
            int(self.tile_layout.cols),
            int(self.tile_layout.rows),
        )
        if self._benchmark_sync_device is not None:
            _LOGGER.info(
                "Train full-video render benchmark sync enabled: warmup_frames=%d.",
                self._benchmark_warmup_frames,
            )

    def prepare_step(self, step: int) -> None:
        prepare = getattr(self.env_raw, "prepare_train_full_video_step_capture", None)
        if callable(prepare):
            activation = (
                self.activation_getter() if callable(self.activation_getter) else None
            )
            prepare(int(step), activation=activation)

    def capture_step(self, step: int) -> None:
        capture = getattr(self.env_raw, "capture_train_full_video_step_state", None)
        if callable(capture):
            activation = (
                self.activation_getter() if callable(self.activation_getter) else None
            )
            capture(int(step), activation=activation)

    def record_rollout(self) -> int:
        getter = getattr(
            self.env_raw,
            "get_train_full_video_render_tile_state_batch",
            None,
        )
        if not callable(getter):
            raise RuntimeError(
                "trainer.recording.full_run requires an env that can expose captured "
                "train full-video tiled states."
            )
        state_batch = getter()
        mocap_pos_batch = None
        mocap_quat_batch = None
        if isinstance(state_batch, TrainFullVideoFrameBatch):
            qpos_batch = state_batch.qpos
            qvel_batch = state_batch.qvel
            activation_batch = state_batch.activation
            mocap_pos_batch = state_batch.mocap_pos
            mocap_quat_batch = state_batch.mocap_quat
        else:
            if len(state_batch) == 5:
                (
                    qpos_batch,
                    qvel_batch,
                    mocap_pos_batch,
                    mocap_quat_batch,
                    activation_batch,
                ) = state_batch
            elif len(state_batch) == 4:
                qpos_batch, qvel_batch, mocap_pos_batch, mocap_quat_batch = state_batch
                activation_batch = None
            elif len(state_batch) == 3:
                qpos_batch, qvel_batch, activation_batch = state_batch
            else:
                qpos_batch, qvel_batch = state_batch
                activation_batch = None
        if (
            self.config.lbs_activation_mapping not in (None, "")
            and activation_batch is None
        ):
            raise RuntimeError(
                "trainer.recording.full_run.warp.lbs_activation_mapping was requested, but no train "
                "activation samples were captured for the rollout."
            )
        if (
            activation_batch is not None
            and not self._activation_stats_logged
            and self.config.activation_log_stats
        ):
            activation_stats = activation_batch.detach()
            _LOGGER.info(
                "Train full-video activation batch: shape=%s min=%.6f max=%.6f "
                "mean=%.6f nonzero_ratio=%.6f.",
                tuple(int(v) for v in activation_stats.shape),
                float(activation_stats.min().item()),
                float(activation_stats.max().item()),
                float(activation_stats.mean().item()),
                float((activation_stats.abs() > 1.0e-5).float().mean().item()),
            )
            self._activation_stats_logged = True
        if int(getattr(qpos_batch, "shape", (0,))[0]) <= 0:
            return 0

        if self.encoder_backend == "pynvcodec":
            frames = self.renderer.render_substep_batch_cuda(
                MujocoWarpRenderBatch(
                    qpos=qpos_batch,
                    qvel=qvel_batch,
                    activation=activation_batch,
                    mocap_pos=mocap_pos_batch,
                    mocap_quat=mocap_quat_batch,
                ),
                benchmark_sync_device=self._benchmark_sync_device,
            )
            self._collect_render_stage_timing_history()
            for frame in frames:
                self.writer.append_frame(frame)
            frame_count = len(frames)
        else:
            qpos_cpu = (
                qpos_batch.detach().cpu().numpy()
                if torch.is_tensor(qpos_batch)
                else qpos_batch
            )
            qvel_cpu = (
                qvel_batch.detach().cpu().numpy()
                if torch.is_tensor(qvel_batch)
                else qvel_batch
            )
            mocap_pos_cpu = (
                mocap_pos_batch.detach().cpu().numpy()
                if torch.is_tensor(mocap_pos_batch)
                else mocap_pos_batch
            )
            mocap_quat_cpu = (
                mocap_quat_batch.detach().cpu().numpy()
                if torch.is_tensor(mocap_quat_batch)
                else mocap_quat_batch
            )
            frames = self.renderer.render_substep_batch(
                MujocoWarpRenderBatch(
                    qpos=qpos_cpu,
                    qvel=qvel_cpu,
                    mocap_pos=mocap_pos_cpu,
                    mocap_quat=mocap_quat_cpu,
                )
            )
            for frame in frames:
                self.writer.append_data(frame)
            frame_count = int(len(frames))

        self.frames_written += int(frame_count)
        return int(frame_count)

    def _collect_render_stage_timing_history(self) -> None:
        if self._benchmark_sync_device is None:
            return
        history = getattr(self.renderer, "_cuda_stage_timing_history_ms", None)
        if not isinstance(history, list):
            return
        for stage_timings in history:
            frame_index = int(self._render_stage_timing_seen_frames)
            self._render_stage_timing_seen_frames += 1
            if frame_index < int(self._benchmark_warmup_frames):
                continue
            if not isinstance(stage_timings, dict):
                continue
            renderer_total = 0.0
            for stage_name, value in stage_timings.items():
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(scalar) or scalar < 0.0:
                    continue
                self._render_stage_timing_samples.setdefault(
                    str(stage_name),
                    [],
                ).append(scalar)
                renderer_total += scalar
            self._render_stage_timing_samples.setdefault(
                "renderer_total",
                [],
            ).append(renderer_total)

    def _log_render_stage_timing_summary(self) -> None:
        if self._benchmark_sync_device is None:
            return
        if not self._render_stage_timing_samples:
            return
        stage_order = [
            "renderer_total",
            "stage_state",
            "refit_bvh",
            "render",
            "muscle_overlay",
            "mesh_overlay",
            "pack",
            "clear_unused",
        ]
        ordered_stages = [
            stage for stage in stage_order if stage in self._render_stage_timing_samples
        ]
        ordered_stages.extend(
            sorted(set(self._render_stage_timing_samples) - set(ordered_stages))
        )
        for stage_name in ordered_stages:
            summary = _summarize_ms(self._render_stage_timing_samples[stage_name])
            if summary is None:
                continue
            _LOGGER.info(
                "TRAIN_FULL_VIDEO_RENDER_TIMING stage=%s count=%d mean_ms=%.6f "
                "p50_ms=%.6f p95_ms=%.6f warmup_frames=%d",
                stage_name,
                int(summary["count"]),
                float(summary["mean"]),
                float(summary["p50"]),
                float(summary["p95"]),
                int(self._benchmark_warmup_frames),
            )

    def close(self) -> Path | None:
        if self._closed:
            return self.output_path
        self._closed = True
        try:
            close_renderer = getattr(self.renderer, "close", None)
            if callable(close_renderer):
                close_renderer()
        finally:
            close_writer = getattr(self.writer, "close", None)
            if callable(close_writer):
                result = close_writer()
                if result is not None:
                    self.output_path = Path(result)
        self._log_render_stage_timing_summary()
        if self.output_path is not None:
            _LOGGER.info(
                "Saved full train video: path=%s frames=%d.",
                self.output_path.as_posix(),
                self.frames_written,
            )
        return self.output_path
