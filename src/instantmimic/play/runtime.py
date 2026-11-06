"""Shared PPO playback runtime used by play, train, and render tooling."""

# Shared playback setup

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm.auto import tqdm

from instantmimic.bootstrap import (
    configure_process,
    prepare_environment_runtime,
)
from instantmimic.config.artifacts import (
    load_artifact_semantic_config,
    load_artifact_semantic_record,
)
from instantmimic.config.config import load_semantic_config
from instantmimic.config.runtime_specs import resolve_seed
from instantmimic.config.schema import SemanticConfig
from instantmimic.play import render as playback_render
from instantmimic.play.config import (
    apply_play_free_run_overrides,
    bundle_config_path,
    cli_dotlist_overrides,
    cli_override_value,
    lower_playback_execution_config,
    resolve_artifact_paths,
    resolve_checkpoint_path,
    task_dotlist_overrides,
    validate_playback_dotlist_overrides,
)
from instantmimic.play.console import (
    PlayModeSummary,
    install_compact_play_logging,
    print_play_header,
    print_play_mode,
    print_play_result,
    print_play_startup_summary,
    should_emit_playback_result_payload,
)
from instantmimic.play.policy import policy_execution_mean_into, policy_mean_into
from instantmimic.play.results import parse_playback_result_payload
from instantmimic.play.setup import build_playback_agent_runtime
from instantmimic.play.stepper import (
    PlaybackGraphRunnerOptions,
    _accumulate_goal_reaches,
    _can_use_fast_headless_playback,
    _configure_playback_env_cuda_graph_mode,
    _episode_done,
    _episode_done_naturally,
    _finalize_tracking_metric_summary,
    _HeadlessPlaybackGraphRunner,
    _initialize_goal_reach_counts,
    _maybe_build_compiled_playback_policy,
    _maybe_collect_tracking_metrics,
    _maybe_extend_playback_episode_length,
    _playback_done_policy_stops,
    _playback_nvtx_range,
    _PlaybackTrackingMetricsGraphRunner,
    _PlaybackVideoStepGraphRunner,
    _policy_sample_into,
    _resolve_play_done_policy,
    _resolve_play_time_limit_policy,
    _resolve_play_use_cuda_graph,
    _resolve_playback_graph_mode,
    _resolve_playback_length,
    _run_fast_headless_playback,
    _step_playback_env_with_io,
    _summarize_goal_reaches,
    _tracking_metric_scalar,
    _update_playback_done_counters,
    _update_tracking_metric_summary,
)
from instantmimic.play.viser_viewer import ViserPlaybackSession, resolve_viewer_backend
from instantmimic.rendering.character import resolve_character_render_mode
from instantmimic.rendering.mujoco_gl_overlay import MujocoGLOverlayRenderer
from instantmimic.rendering.mujoco_gl_playback import (
    CudaGLInteropError,
    MujocoGLPlaybackRenderer,
    MujocoGLRenderBatch,
)
from instantmimic.rendering.mujoco_warp_playback import (
    MujocoWarpPlaybackRenderer,
    MujocoWarpRenderBatch,
)
from instantmimic.rendering.video_capture_setup import (
    enable_substep_render_capture_for_video,
)
from instantmimic.rl.eager_fallback import EagerFallbackPolicy
from instantmimic.rl.runtime import (
    agent_observation_output_views,
    copy_agent_observation_into,
    set_seed,
)
from instantmimic.utils.experiment_logging import resolve_hydra_config_name
from instantmimic.utils.project_paths import (
    resolve_project_log_root,
)
from instantmimic.utils.startup_profiler import StartupProfiler, startup_nvtx_range
from instantmimic.utils.torch_warp_capture import warp_stream_from_torch

_LOGGER = logging.getLogger(__name__)


def _synchronize_video_benchmark_cuda(
    *,
    enabled: bool,
    device: torch.device,
) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start_ns: int | None, end_ns: int | None) -> float | None:
    if start_ns is None or end_ns is None:
        return None
    return max(0.0, float(end_ns - start_ns) / 1_000_000.0)


def _write_video_benchmark_summary(
    *,
    recorder: Any,
    output_path: Path,
    checkpoint_path: Path,
    video_output_path: Path | None,
    video_render_backend: str,
    video_encoder_backend: str,
    video_codec: str,
    tile_layout: Any | None,
) -> None:
    payload = recorder.to_summary()
    payload.update(
        {
            "checkpoint_path": checkpoint_path.as_posix(),
            "video_output_path": (
                video_output_path.as_posix() if video_output_path is not None else None
            ),
            "video_render_backend": str(video_render_backend),
            "video_encoder_backend": str(video_encoder_backend),
            "video_codec": str(video_codec),
        }
    )
    if tile_layout is not None:
        payload["tile_layout"] = {
            "env_count": int(tile_layout.env_count),
            "cols": int(tile_layout.cols),
            "rows": int(tile_layout.rows),
            "tile_width": int(tile_layout.tile_width),
            "tile_height": int(tile_layout.tile_height),
            "output_width": int(tile_layout.output_width),
            "output_height": int(tile_layout.output_height),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _environment_activation(env_raw: Any) -> torch.Tensor | None:
    """Read the current muscle state from its environment owner for rendering."""
    getter = getattr(env_raw, "get_full_video_activation", None)
    return getter() if callable(getter) else None


def _build_playback_renderer(
    *,
    video_render_backend: str,
    renderer_kwargs: Mapping[str, Any],
    video_tile_layout: Any,
    cfg: DictConfig,
    mujoco_gl_overlay_renderer: Any,
    character_render_mode: str,
    activation: torch.Tensor | None,
) -> Any:
    if video_render_backend == "warp":
        return MujocoWarpPlaybackRenderer(
            **renderer_kwargs,
            overlay_renderer=None,
            tile_env_count=video_tile_layout.env_count,
            tile_cols=video_tile_layout.cols,
            use_textures=bool(cfg.get("video_warp_use_textures", True)),
            floor_shader_mode=str(cfg.get("video_warp_floor_shader_mode", "texture")),
            render_block_dim=int(cfg.get("video_warp_render_block_dim", 64)),
            lbs_mesh_mode=cfg.get("video_warp_lbs_mesh_mode", "off"),
            lbs_mesh_bundle_path=cfg.get("video_warp_lbs_mesh_bundle", None),
            lbs_mesh_shading_mode=cfg.get("video_warp_lbs_mesh_shading_mode", "smooth"),
            lbs_activation_mapping_path=cfg.get(
                "video_warp_lbs_activation_mapping",
                None,
            ),
            lbs_activation_tensor=activation,
            character_render_mode=character_render_mode,
        )
    return MujocoGLPlaybackRenderer(
        **renderer_kwargs,
        overlay_renderer=mujoco_gl_overlay_renderer,
        character_render_mode=character_render_mode,
    )


def _initialize_playback_renderer_cuda_interop(
    *,
    playback_renderer: Any,
    video_render_backend: str,
    video_encoder_backend: str,
    requested_video_encoder_backend: str,
) -> str:
    if video_render_backend != "mujoco_gl" or video_encoder_backend != "pynvcodec":
        return video_encoder_backend
    try:
        playback_renderer.enable_cuda_interop()
    except CudaGLInteropError as exc:
        if requested_video_encoder_backend != "auto":
            raise
        _LOGGER.warning(
            "MuJoCo GL CUDA interop is unavailable (%s); "
            "video_encoder_backend=auto is falling back to imageio CPU readback. "
            "Check the NVIDIA driver/runtime installation to restore GPU encoding.",
            exc,
        )
        return "imageio"
    return video_encoder_backend


def _render_playback_video_frames(
    *,
    playback_renderer: Any,
    render_batch: Any,
    video_encoder_backend: str,
    video_render_backend: str,
    overlay_state: Any,
    episode_length: int,
    step_reward: float,
    benchmark_sync_device: torch.device | None,
) -> Any:
    if video_encoder_backend == "pynvcodec":
        cuda_render_kwargs = {
            "overlay_state": overlay_state,
            "episode_length": episode_length,
            "step_reward": step_reward,
        }
        if video_render_backend == "warp":
            cuda_render_kwargs["benchmark_sync_device"] = benchmark_sync_device
        return playback_renderer.render_substep_batch_cuda(
            render_batch,
            **cuda_render_kwargs,
        )
    return playback_renderer.render_substep_batch(
        render_batch,
        overlay_state=overlay_state,
        episode_length=episode_length,
        step_reward=step_reward,
    )


def _optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _optional_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def _accumulate_play_env_diagnostic_means(
    env_raw: Any,
    sums: dict[str, float],
    sample_count: int,
) -> int:
    get_metrics = getattr(env_raw, "get_training_diagnostic_means", None)
    if not callable(get_metrics):
        return sample_count
    metrics = get_metrics()
    if not metrics:
        return sample_count
    sampled = False
    for key, value in metrics.items():
        try:
            metric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(metric_value):
            continue
        sums[str(key)] = sums.get(str(key), 0.0) + metric_value
        sampled = True
    return sample_count + 1 if sampled else sample_count


@dataclass(frozen=True, init=False)
class PlaybackRequest:
    cfg: SemanticConfig
    checkpoint_path: Path
    log_dir: Path | None
    dotlist_overrides: tuple[str, ...]

    def __init__(
        self,
        cfg: SemanticConfig,
        *,
        checkpoint_path: str | Path,
        log_dir: str | Path | None = None,
        dotlist_overrides: Iterable[str] | None = None,
    ) -> None:
        object.__setattr__(self, "cfg", cfg)
        object.__setattr__(self, "checkpoint_path", Path(checkpoint_path).expanduser())
        object.__setattr__(self, "log_dir", _optional_path(log_dir))
        object.__setattr__(
            self,
            "dotlist_overrides",
            tuple(str(override) for override in (dotlist_overrides or ())),
        )


@dataclass(frozen=True)
class PlaybackResult(Mapping[str, str]):
    checkpoint_path: Path | None = None
    log_dir: Path | None = None
    episode_length: int | None = None
    episode_reward: float | None = None
    episode_finish_count: int | None = None
    terminated_count: int | None = None
    truncated_count: int | None = None
    tracking_valid: bool | None = None
    video_output_path: Path | None = None
    metrics: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_cli_payload(
        cls,
        payload: Mapping[str, str],
        *,
        checkpoint_path: str | Path | None = None,
        log_dir: str | Path | None = None,
    ) -> PlaybackResult:
        checkpoint_value = (
            payload.get("checkpoint_path")
            or payload.get("checkpoint")
            or (str(checkpoint_path) if checkpoint_path is not None else None)
        )
        log_dir_value = payload.get("log_dir") or (
            str(log_dir) if log_dir is not None else None
        )
        known_keys = {
            "checkpoint",
            "checkpoint_path",
            "log_dir",
            "episode_length",
            "episode_reward",
            "episode_finish_count",
            "terminated_count",
            "truncated_count",
            "tracking_valid",
            "video_output_path",
        }
        metrics = {
            str(key): str(value)
            for key, value in payload.items()
            if key not in known_keys and value not in (None, "")
        }
        return cls(
            checkpoint_path=_optional_path(checkpoint_value),
            log_dir=_optional_path(log_dir_value),
            episode_length=_optional_int(payload.get("episode_length")),
            episode_reward=_optional_float(payload.get("episode_reward")),
            episode_finish_count=_optional_int(payload.get("episode_finish_count")),
            terminated_count=_optional_int(payload.get("terminated_count")),
            truncated_count=_optional_int(payload.get("truncated_count")),
            tracking_valid=_optional_bool(payload.get("tracking_valid")),
            video_output_path=_optional_path(payload.get("video_output_path")),
            metrics=metrics,
        )

    def to_cli_payload(self) -> dict[str, str]:
        payload: dict[str, str] = {}
        if self.checkpoint_path is not None:
            payload["checkpoint"] = self.checkpoint_path.as_posix()
            payload["checkpoint_path"] = self.checkpoint_path.as_posix()
        if self.log_dir is not None:
            payload["log_dir"] = self.log_dir.as_posix()
        if self.episode_length is not None:
            payload["episode_length"] = str(int(self.episode_length))
        if self.episode_reward is not None:
            payload["episode_reward"] = f"{float(self.episode_reward):.6f}"
        if self.episode_finish_count is not None:
            payload["episode_finish_count"] = str(int(self.episode_finish_count))
        if self.terminated_count is not None:
            payload["terminated_count"] = str(int(self.terminated_count))
        if self.truncated_count is not None:
            payload["truncated_count"] = str(int(self.truncated_count))
        if self.tracking_valid is not None:
            payload["tracking_valid"] = "true" if self.tracking_valid else "false"
        if self.video_output_path is not None:
            payload["video_output_path"] = self.video_output_path.as_posix()
        payload.update(self.metrics)
        return payload

    def to_log_line(self) -> str:
        payload = self.to_cli_payload()
        ordered_keys = [
            "checkpoint",
            "episode_length",
            "episode_reward",
            "episode_finish_count",
            "terminated_count",
            "truncated_count",
            "tracking_valid",
            "video_output_path",
        ]
        emitted: set[str] = set()
        tokens: list[str] = []
        for key in ordered_keys:
            if key in payload:
                tokens.append(f"{key}={payload[key]}")
                emitted.add(key)
        for key in sorted(payload):
            if key not in emitted and key != "checkpoint_path":
                tokens.append(f"{key}={payload[key]}")
        return "PLAYBACK_RESULT " + " ".join(tokens)

    def to_json_line(self) -> str:
        """Lossless process result; retain to_log_line for legacy log consumers."""
        return "PLAYBACK_RESULT_JSON " + json.dumps(self.to_cli_payload())

    def __getitem__(self, key: str) -> str:
        return self.to_cli_payload()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_cli_payload())

    def __len__(self) -> int:
        return len(self.to_cli_payload())


def parse_playback_result_stdout(stdout: str) -> PlaybackResult | None:
    """Read the result line a child `play` process wrote to its stdout.

    This is the inter-process contract behind `emit_playback_result_payload`:
    the child prints `PlaybackResult.to_log_line()` to **stdout** (not through
    logging, which the parent does not capture) and `train/playback.py` reads it
    back from the completed process. Moving that write to a logger would break
    the parent silently -- it would simply find no result.
    """

    payload = parse_playback_result_payload(stdout)
    return PlaybackResult.from_cli_payload(payload) if payload is not None else None


def checkpoint_first_bootstrap_payload(
    argv: list[str],
) -> tuple[SemanticConfig, Path, Path, list[str]] | None:
    if any(
        arg in {"-h", "--help", "--hydra-help", "--cfg", "--info", "-m", "--multirun"}
        for arg in argv
    ):
        return None

    checkpoint_raw = cli_override_value(argv, "play.checkpoint")
    if checkpoint_raw is None:
        return None

    dotlist_overrides = cli_dotlist_overrides(argv)
    resume_path = resolve_checkpoint_path(
        checkpoint_raw,
        resolve_project_log_root("ppo", ""),
    )
    run_log_dir = resume_path.parent.parent.resolve()
    artifact = load_artifact_semantic_config(resume_path)
    cfg = OmegaConf.create(artifact.to_dict())
    if dotlist_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist_overrides))
    return load_semantic_config(cfg), resume_path, run_log_dir, dotlist_overrides


def maybe_run_checkpoint_first_cli(argv: list[str]) -> bool:
    payload = checkpoint_first_bootstrap_payload(argv)
    if payload is None:
        return False

    cfg, resume_path, run_log_dir, dotlist_overrides = payload
    run_playback_request(
        PlaybackRequest(
            cfg,
            checkpoint_path=resume_path,
            log_dir=run_log_dir,
            dotlist_overrides=dotlist_overrides,
        )
    )
    return True


def run_playback_request(request: PlaybackRequest) -> PlaybackResult:
    return play_checkpoint(
        request.cfg,
        checkpoint_path=request.checkpoint_path,
        log_dir=request.log_dir,
        dotlist_overrides=list(request.dotlist_overrides),
    )


# Overlay configuration
def _parse_overlay_color(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 7 and text.startswith("#"):
            try:
                return tuple(int(text[idx : idx + 2], 16) for idx in (1, 3, 5))
            except ValueError:
                return None
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            rgb = tuple(int(v) for v in value)
        except (TypeError, ValueError):
            return None
        if all(0 <= c <= 255 for c in rgb):
            return rgb
    return None


def _build_reward_overlay_state(max_history_steps: int) -> dict[str, Any]:
    return {
        "max_history_steps": max_history_steps,
        "title": "",
        "plots": [],
    }


def _clear_reward_overlay_state(overlay_state: dict[str, Any]) -> None:
    for plot in overlay_state["plots"]:
        for series in plot["series"].values():
            series.clear()


def _as_overlay_scalar(value: Any) -> float | None:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.reshape(-1)[0].detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _as_overlay_scalar(value[0])
    return None


def _append_reward_overlay_sample(
    overlay_state: dict[str, Any],
    overlay_data: dict[str, Any],
) -> bool:
    plots = overlay_data.get("plots")
    if not isinstance(plots, list) or len(plots) == 0:
        return False

    overlay_state["title"] = str(overlay_data.get("title", "Debug Overlay"))
    plot_states = overlay_state["plots"]

    while len(plot_states) < len(plots):
        plot_states.append({"title": "", "series": {}, "colors": {}})

    any_series = False
    for plot_state, plot_data in zip(plot_states, plots, strict=False):
        if not isinstance(plot_data, dict):
            continue
        series_data = plot_data.get("series")
        if not isinstance(series_data, dict) or len(series_data) == 0:
            continue
        plot_state["title"] = str(plot_data.get("title", "plot"))
        plot_colors: dict[str, tuple[int, int, int]] = {}
        raw_colors = plot_data.get("colors", {})
        if isinstance(raw_colors, dict):
            for color_name, color_value in raw_colors.items():
                parsed = _parse_overlay_color(color_value)
                if parsed is not None:
                    plot_colors[str(color_name)] = parsed
        plot_state["colors"] = plot_colors
        for name, raw_value in series_data.items():
            value = _as_overlay_scalar(raw_value)
            if value is None:
                continue
            series_map = plot_state["series"]
            if name not in series_map:
                series_map[name] = deque(maxlen=overlay_state["max_history_steps"])
            series_map[name].append(float(value))
            any_series = True

    if len(plot_states) > len(plots):
        del plot_states[len(plots) :]

    return any_series


# Playback lifecycle


class _PlaybackCloseable(Protocol):
    def close(self) -> object: ...


class _PlaybackVideoWriter(Protocol):
    def close(self) -> str | os.PathLike[str] | None: ...


class _PlaybackSessionRange(Protocol):
    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> object: ...


@dataclass(slots=True)
class _PlaybackLifecycleResources:
    env: _PlaybackCloseable
    session_range: _PlaybackSessionRange
    playback_nvtx_enabled: bool
    video_writer: _PlaybackVideoWriter | None = None
    playback_renderer: _PlaybackCloseable | None = None
    viser_session: _PlaybackCloseable | None = None
    video_output_path: Path | None = None


def _finalize_playback_lifecycle(
    resources: _PlaybackLifecycleResources,
    *,
    close_environment: bool = True,
) -> Path | None:
    video_output_path = resources.video_output_path
    video_error: Exception | None = None
    try:
        if resources.video_writer is not None:
            try:
                with _playback_nvtx_range(
                    "playback_video_finalize",
                    enabled=resources.playback_nvtx_enabled,
                ):
                    closed_path = resources.video_writer.close()
                if closed_path is not None:
                    video_output_path = Path(closed_path).resolve()
                if video_output_path is not None:
                    _LOGGER.info("Saved video to: %s", video_output_path.as_posix())
            except Exception as exc:
                video_error = exc
                _LOGGER.exception("Failed to finalize streaming video writer.")
        if resources.playback_renderer is not None:
            try:
                resources.playback_renderer.close()
            except Exception:
                _LOGGER.exception("Failed to close playback renderer.")
        if resources.viser_session is not None:
            try:
                resources.viser_session.close()
            except Exception:
                _LOGGER.exception("Failed to close viser playback session.")
        if close_environment:
            resources.env.close()
    finally:
        resources.session_range.__exit__(None, None, None)
    if video_error is not None:
        raise RuntimeError("Failed to finalize playback video.") from video_error
    return video_output_path


def _resolve_playback_semantic_config(
    cfg: SemanticConfig,
    *,
    checkpoint_path: Path,
    dotlist_overrides: list[str] | None,
) -> SemanticConfig:
    overrides = list(dotlist_overrides or [])
    validate_playback_dotlist_overrides(overrides)
    requested = cfg
    artifact_record = load_artifact_semantic_record(checkpoint_path)
    values = artifact_record.config.to_dict()
    if artifact_record.resolved_seed is not None:
        values["seed"] = artifact_record.resolved_seed
    values["play"] = requested.to_dict()["play"]
    if overrides:
        merged = OmegaConf.merge(
            OmegaConf.create(values),
            OmegaConf.from_dotlist(overrides),
        )
        resolved = OmegaConf.to_container(merged, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError("resolved playback configuration must be a mapping")
        values = resolved
    return SemanticConfig.from_mapping(values)


def play_checkpoint(
    cfg: SemanticConfig,
    *,
    checkpoint_path: str | Path,
    log_dir: str | Path | None = None,
    dotlist_overrides: list[str] | None = None,
) -> PlaybackResult:
    with ExitStack() as environment_cleanup:
        return _play_checkpoint(
            cfg,
            checkpoint_path=checkpoint_path,
            log_dir=log_dir,
            dotlist_overrides=dotlist_overrides,
            environment_cleanup=environment_cleanup,
        )


def _play_checkpoint(
    cfg: SemanticConfig,
    *,
    checkpoint_path: str | Path,
    log_dir: str | Path | None,
    dotlist_overrides: list[str] | None,
    environment_cleanup: ExitStack,
) -> PlaybackResult:
    configure_process()
    resume_path = Path(checkpoint_path).expanduser().resolve()
    run_log_dir = (
        Path(log_dir).expanduser().resolve()
        if log_dir is not None
        else resume_path.parent.parent.resolve()
    )
    console_enabled = True
    install_compact_play_logging(log_dir=run_log_dir, enabled=console_enabled)
    config_name = resolve_hydra_config_name(default=run_log_dir.name)
    _LOGGER.info("Loading PPO checkpoint from: %s", resume_path)

    semantic_cfg = _resolve_playback_semantic_config(
        cfg,
        checkpoint_path=resume_path,
        dotlist_overrides=dotlist_overrides,
    )
    play_seed = semantic_cfg.play.get("seed")
    requested_seed = semantic_cfg.seed if play_seed is None else int(play_seed)
    # Shared with training. This used `random.randint(0, 10000)`, drawing from
    # the global `random` state over 10001 values while training drew from
    # `secrets` over the full 31-bit range.
    resolved_seed = resolve_seed(requested_seed)
    _LOGGER.info(
        "Playback seed: requested=%s resolved=%s", requested_seed, resolved_seed
    )
    cfg = lower_playback_execution_config(
        semantic_cfg,
        seed=resolved_seed,
        log_dir=run_log_dir,
    )
    cfg = resolve_artifact_paths(
        cfg,
        run_log_dir,
        bundle_cfg_path=bundle_config_path(run_log_dir),
    )
    apply_play_free_run_overrides(cfg)

    with open_dict(cfg):
        cfg.env.scene.num_envs = int(semantic_cfg.play["num_envs"])
        cfg.checkpoint = str(resume_path)

    set_seed(int(cfg.seed))

    take_video = bool(cfg.get("video", False))
    character_render_mode = resolve_character_render_mode(cfg.env)
    headless_play_steps = max(0, int(cfg.get("play_steps", 0)))
    viewer_backend = resolve_viewer_backend(cfg)
    print_play_header(
        checkpoint_path=resume_path,
        config_name=config_name,
        log_dir=run_log_dir,
        enabled=console_enabled,
    )
    tracking_require_full_horizon = bool(
        cfg.get("tracking_require_full_horizon", headless_play_steps > 0)
    )
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if (
        not take_video
        and headless_play_steps <= 0
        and viewer_backend == "mujoco"
        and not has_display
    ):
        msg = (
            "No DISPLAY/WAYLAND_DISPLAY found, but playback is configured for interactive "
            "viewer mode (play.video.enabled=false). For headless servers, set "
            "`play.video.enabled=true` or provide `play.steps=<N>` for headless "
            "fixed-step playback, "
            "or set `++play.viewer.backend=viser` for browser-based playback. "
            "Note: checkpoint bundled config may override your selected config and keep video=false."
        )
        _LOGGER.error(msg)
        raise RuntimeError(msg)

    startup_profiler = StartupProfiler(range_factory=startup_nvtx_range)
    with startup_profiler.phase("gym_make_env"):
        prepare_environment_runtime()
        env = gym.make(
            cfg.task,
            cfg=cfg.env,
            render_mode="rgb_array" if take_video else None,
        )

    environment_cleanup.callback(env.close)
    with startup_profiler.phase("build_agent_runtime"):
        setup = build_playback_agent_runtime(
            cfg=semantic_cfg,
            env=env,
            checkpoint_path=resume_path,
            environment_cfg=cfg.env,
        )
    device = setup.device
    num_envs = setup.num_envs
    agent_action_dim = setup.agent_action_dim
    env_raw = setup.env_raw
    if take_video or viewer_backend == "viser":
        enable_substep_render_capture_for_video(env_raw, cfg_env=cfg.env)
    agent = setup.agent
    if setup.env_state_restored:
        _LOGGER.info("Restored env checkpoint state from: %s", resume_path)
        if setup.action_scales_synced:
            _LOGGER.info(
                "Resynced agent action scales from restored env checkpoint state."
            )

    with startup_profiler.phase("env_reset"):
        obs, _ = env.reset()
    obs_width = (
        int(agent.obs_dim)
        if bool(getattr(agent, "asymmetric_observations", False))
        else int(obs["policy"].shape[1])
    )
    obs_policy = torch.empty(num_envs, obs_width, device=device, dtype=torch.float32)
    copy_agent_observation_into(agent=agent, observations=obs, destination=obs_policy)
    agent_actions = torch.empty(num_envs, agent_action_dim, device=device)
    env_actions = agent_actions
    next_obs_policy = torch.empty_like(obs_policy)
    next_observation_views = agent_observation_output_views(
        agent=agent, destination=next_obs_policy
    )
    next_rewards = torch.empty(num_envs, device=device, dtype=torch.float32)
    next_terminated = torch.empty(num_envs, device=device, dtype=torch.bool)
    next_truncated = torch.empty_like(next_terminated)
    next_dones = torch.empty_like(next_terminated)
    play_torch_stream = (
        torch.cuda.Stream(device=device) if device.type == "cuda" else None
    )
    play_wp_stream = (
        warp_stream_from_torch(play_torch_stream)
        if play_torch_stream is not None
        else None
    )
    playback_nvtx_enabled = device.type == "cuda"
    video_benchmark_sync = playback_render.resolve_video_benchmark_sync(
        cfg,
        take_video=take_video,
    )
    video_benchmark_warmup_frames = (
        playback_render.resolve_video_benchmark_warmup_frames(
            cfg,
            take_video=take_video,
        )
    )
    video_benchmark_stage_sync = playback_render.resolve_video_benchmark_stage_sync(
        cfg,
        take_video=take_video,
    )
    video_benchmark_recorder = (
        playback_render.PlaybackVideoBenchmarkRecorder(
            warmup_frames=video_benchmark_warmup_frames
        )
        if video_benchmark_sync
        else None
    )
    dt = env_raw.step_dt
    auto_resets = hasattr(env_raw, "_reset_idx")
    tracking_metric_frame = str(cfg.get("tracking_metric_frame", "root"))
    tracking_metric_include_root = bool(
        cfg.get("tracking_metric_include_root_in_ape", False)
    )
    tracking_metric_summary: dict[str, Any] = {
        "steps": 0,
        "ape_sum": 0.0,
        "ee_sum": 0.0,
        "ape_last": None,
        "ee_last": None,
        "vel_sum": 0.0,
        "vel_last": None,
        "accel_sum": 0.0,
        "accel_last": None,
        "temporal_steps": 0,
        "accel_steps": 0,
        "frame": tracking_metric_frame,
        "dt": float(dt),
        "prev_sim_body_linvel": None,
        "prev_ref_body_linvel": None,
        "prev_sim_root_quat": None,
        "prev_ref_root_quat": None,
        "vel_buffers": None,
        "accel_buffers": None,
    }
    tracking_failure_info: dict[str, float | int | bool] | None = None
    tracking_metrics_available = callable(
        getattr(env_raw, "get_reference_tracking_metrics", None)
    )
    episode_reward = 0.0
    episode_length = 0
    best_episode_reward = 0.0
    best_episode_length = 0
    episode_index = 1
    playback_done_counters = {
        "episode_finish_count": 0,
        "terminated_count": 0,
        "truncated_count": 0,
    }
    play_env_diagnostic_sums: dict[str, float] = {}
    play_env_diagnostic_steps = 0
    goal_reach_counts = _initialize_goal_reach_counts(
        env_raw,
        num_envs=num_envs,
        device=device,
    )
    goal_reach_eval_steps = 0
    goal_reach_summary: dict[str, int | float | bool] | None = None
    viewer_status_interval_steps = 30
    video_writer: Any | None = None
    video_output_path: Path | None = None
    video_tile_layout: Any | None = None
    render_warning_logged = False
    viewer_overlay_default = (
        viewer_backend == "viser" and not take_video and headless_play_steps <= 0
    )
    overlay_reward_terms = bool(
        cfg.get(
            "video_overlay_plots",
            cfg.get("video_overlay_rewards", viewer_overlay_default),
        )
    )
    overlay_history_seconds = max(
        float(dt), float(cfg.get("video_overlay_history_seconds", 3.0))
    )
    overlay_panel_width = max(220, int(cfg.get("video_overlay_panel_width", 360)))
    overlay_max_history_steps = max(2, int(round(overlay_history_seconds / float(dt))))
    reward_overlay_state = _build_reward_overlay_state(overlay_max_history_steps)
    overlay_warning_logged = False
    video_render_backend = playback_render.resolve_video_render_backend(
        cfg,
        take_video=take_video,
        overlay_reward_terms=overlay_reward_terms,
    )
    video_codec = str(cfg.get("video_codec", "h264")).strip().lower()
    video_encoder_backend = playback_render.resolve_video_encoder_backend(
        cfg,
        take_video=take_video,
        video_render_backend=video_render_backend,
    )
    requested_video_encoder_backend = (
        str(cfg.get("video_encoder_backend", "auto")).strip().lower()
    )
    if take_video:
        _LOGGER.info("Playback video render backend: %s", video_render_backend)
        _LOGGER.info(
            "Playback video character render mode: %s",
            character_render_mode,
        )
        if video_benchmark_sync:
            _LOGGER.info(
                "Playback video benchmark sync enabled: render and encode NVTX ranges "
                "will include explicit CUDA synchronization boundaries; warmup_frames=%d.",
                video_benchmark_warmup_frames,
            )
            if video_benchmark_stage_sync:
                _LOGGER.info(
                    "Playback video benchmark stage sync enabled: tiled Warp render "
                    "will record synchronized stage timings."
                )
    mujoco_gl_overlay_renderer = (
        MujocoGLOverlayRenderer(
            panel_width=overlay_panel_width,
            history_seconds=overlay_history_seconds,
        )
        if overlay_reward_terms
        else None
    )
    playback_renderer: MujocoGLPlaybackRenderer | MujocoWarpPlaybackRenderer | None = (
        None
    )
    viser_session: ViserPlaybackSession | None = None
    _configure_playback_env_cuda_graph_mode(
        env_raw,
        take_video=take_video,
    )
    stochastic_play = bool(cfg.get("stochastic", False))
    if stochastic_play:
        _LOGGER.info(
            "Stochastic playback enabled — actions will be sampled from the policy distribution."
        )

    play_length = _resolve_playback_length(
        cfg=cfg,
        take_video=take_video,
        headless_play_steps=headless_play_steps,
        env_raw=env_raw,
    )
    play_time_limit_policy = _resolve_play_time_limit_policy(cfg, env_raw=env_raw)
    play_done_policy = _resolve_play_done_policy(cfg)
    prev_episode_length_s = (
        float(env_raw.cfg.episode_length_s)
        if hasattr(env_raw, "cfg") and hasattr(env_raw.cfg, "episode_length_s")
        else None
    )
    prev_max_episode_length, max_episode_length = _maybe_extend_playback_episode_length(
        env_raw,
        play_length=play_length,
        time_limit_policy=play_time_limit_policy,
    )
    if prev_max_episode_length is not None and play_length > prev_max_episode_length:
        if play_time_limit_policy == "respect":
            _LOGGER.info(
                "play length (%d) exceeds max_episode_length (%d); "
                "respecting the environment episode limit "
                "(play_done_policy=%s).",
                play_length,
                prev_max_episode_length,
                play_done_policy,
            )
        elif prev_episode_length_s is not None:
            _LOGGER.info(
                "play length (%d) exceeds max_episode_length (%d); "
                "extended episode_length_s from %.3f to %.3f "
                "(new max_episode_length=%d).",
                play_length,
                prev_max_episode_length,
                prev_episode_length_s,
                float(env_raw.cfg.episode_length_s),
                int(max_episode_length),
            )
        else:
            _LOGGER.info(
                "play length (%d) exceeds max_episode_length (%d); "
                "continuing playback across episode boundaries.",
                play_length,
                prev_max_episode_length,
            )
    play_use_cuda_graph = _resolve_play_use_cuda_graph(cfg)
    eager_fallback = EagerFallbackPolicy(
        allowed=bool(cfg.get("play_allow_eager_fallback", False))
    )
    fast_headless_playback = _can_use_fast_headless_playback(
        env_raw=env_raw,
        take_video=take_video,
        play_length=play_length,
        stochastic_play=stochastic_play,
    )
    playback_graph_mode = _resolve_playback_graph_mode(
        requested=play_use_cuda_graph,
        device=device,
        env_raw=env_raw,
        take_video=take_video,
        play_length=play_length,
        stochastic_play=stochastic_play,
    )
    playback_cuda_graph_enabled = playback_graph_mode == "headless"
    playback_graph_runner: _HeadlessPlaybackGraphRunner | None = None
    playback_mode = (
        "video" if take_video else "headless" if headless_play_steps > 0 else "viewer"
    )
    viewer_reset_on_done = bool(cfg.get("viewer_reset_on_done", False))
    print_play_mode(
        PlayModeSummary(
            mode=playback_mode,
            steps=play_length if playback_mode != "viewer" else None,
            graph_mode=playback_graph_mode,
            video_encoder_backend=video_encoder_backend if take_video else None,
            video_render_backend=video_render_backend if take_video else None,
            overlay_enabled=(
                overlay_reward_terms
                if take_video
                or (playback_mode == "viewer" and viewer_backend == "viser")
                else False
            ),
        ),
        enabled=console_enabled,
    )
    if play_use_cuda_graph and playback_graph_mode == "none":
        _LOGGER.info(
            "play_use_cuda_graph=true requested, but this playback mode is not graph-capturable; "
            "falling back to the non-graph playback path."
        )
    if playback_mode == "viewer" and not viewer_reset_on_done:
        _LOGGER.info(
            "Interactive viewer playback will continue across episode done signals. "
            "Set +viewer_reset_on_done=true to restore reset-on-done behavior."
        )
    with startup_profiler.phase("policy_compile_warmup"):
        compiled_playback_policy = _maybe_build_compiled_playback_policy(
            agent=agent,
            device=device,
            obs_policy=obs_policy,
            agent_actions=agent_actions,
            play_torch_stream=play_torch_stream,
            stochastic_play=stochastic_play,
            eager_fallback=eager_fallback,
            use_execution_mean=playback_mode != "viewer",
        )
    tracking_metrics_graph_runner: _PlaybackTrackingMetricsGraphRunner | None = None
    playback_video_step_graph_runner: _PlaybackVideoStepGraphRunner | None = None
    playback_video_step_graph_enabled = playback_graph_mode == "video_device_step"
    if (
        playback_nvtx_enabled
        and tracking_metrics_available
        and play_torch_stream is not None
        and play_wp_stream is not None
    ):
        try:
            tracking_metrics_graph_runner = _PlaybackTrackingMetricsGraphRunner(
                env_raw=env_raw,
                device=device,
                play_torch_stream=play_torch_stream,
                play_wp_stream=play_wp_stream,
                frame=tracking_metric_frame,
                include_root_in_ape=tracking_metric_include_root,
                dt=float(dt),
            )
            if not playback_video_step_graph_enabled:
                with startup_profiler.phase("tracking_graph_capture"):
                    tracking_metrics_graph_runner.capture()
                _LOGGER.info("Captured tracking metrics CUDA graph.")
        except Exception as exc:  # noqa: BLE001 - a capture can fail in any
            # backend layer; the policy decides whether that ends the run.
            eager_fallback.degrade("tracking metrics CUDA graph", exc)
            tracking_metrics_graph_runner = None

    if playback_video_step_graph_enabled:
        if play_torch_stream is None or play_wp_stream is None:
            _LOGGER.info(
                "play_use_cuda_graph=true requested for video playback, but auxiliary CUDA streams are unavailable; "
                "falling back to per-function playback path."
            )
            playback_video_step_graph_enabled = False
        else:
            try:
                playback_video_step_graph_runner = _PlaybackVideoStepGraphRunner(
                    PlaybackGraphRunnerOptions(
                        env=env,
                        env_raw=env_raw,
                        agent=agent,
                        device=device,
                        obs_policy=obs_policy,
                        agent_actions=agent_actions,
                        env_actions=env_actions,
                        play_torch_stream=play_torch_stream,
                        play_wp_stream=play_wp_stream,
                        stochastic_play=stochastic_play,
                    ),
                    compiled_playback_policy=compiled_playback_policy,
                    tracking_metrics_runner=tracking_metrics_graph_runner,
                )
                with startup_profiler.phase("video_step_graph_capture"):
                    playback_video_step_graph_runner.capture()
                _LOGGER.info("Captured playback video device-step CUDA graph.")
            except Exception as exc:  # noqa: BLE001 - as above, any backend
                # layer may raise; the policy decides whether to continue.
                eager_fallback.degrade("playback video device-step CUDA graph", exc)
                playback_video_step_graph_runner = None
                playback_video_step_graph_enabled = False
                if (
                    tracking_metrics_graph_runner is not None
                    and tracking_metrics_graph_runner._graph is None
                ):
                    try:
                        with startup_profiler.phase("tracking_graph_capture"):
                            tracking_metrics_graph_runner.capture()
                        _LOGGER.info(
                            "Captured tracking metrics CUDA graph after video device-step graph fallback."
                        )
                    except Exception as exc:  # noqa: BLE001 - any backend layer
                        # may raise; the policy decides whether to continue.
                        # The runner has to go: a failed recapture leaves it
                        # non-None with no graph, and playback would then call
                        # `step()` on it and raise "has not been captured yet"
                        # much later. Routed through the policy so the stage is
                        # reported like the other three.
                        tracking_metrics_graph_runner = None
                        eager_fallback.degrade(
                            "tracking metrics CUDA graph after video device-step "
                            "graph fallback",
                            exc,
                        )

    # `tracking_metrics_graph_runner` is deliberately kept here. It used to be
    # cleared unconditionally after this block, at the indentation of the
    # `try`/`except` above rather than inside the success branch, which also
    # discarded the runner the fallback path had just re-captured -- so a failed
    # video-step capture silently lost tracking metrics to the eager path as
    # well. The consumer already skips it when the video-step graph owns it
    # (`if playback_video_step_graph_runner is not None: tracking_metrics = None`),
    # so nothing needs clearing on the success path either.

    def _policy_and_step(
        *,
        use_execution_mean: bool,
        reset_done: bool = True,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        torch_stream_ctx = (
            torch.cuda.stream(play_torch_stream)
            if play_torch_stream is not None
            else nullcontext()
        )
        wp_stream_ctx = (
            wp.ScopedStream(play_wp_stream, sync_enter=False, sync_exit=False)
            if play_wp_stream is not None
            else nullcontext()
        )
        with wp_stream_ctx, torch_stream_ctx:
            with _playback_nvtx_range(
                "playback_policy",
                enabled=playback_nvtx_enabled,
            ):
                if stochastic_play:
                    _policy_sample_into(agent, obs_policy, agent_actions)
                else:
                    if use_execution_mean:
                        compiled_execution_mean = compiled_playback_policy.get(
                            "execution_mean"
                        )
                        if callable(compiled_execution_mean):
                            compiled_execution_mean(obs_policy, agent_actions)
                        else:
                            policy_execution_mean_into(
                                agent,
                                obs_policy,
                                agent_actions,
                            )
                    else:
                        compiled_mean = compiled_playback_policy.get("mean")
                        if callable(compiled_mean):
                            compiled_mean(obs_policy, agent_actions)
                        else:
                            policy_mean_into(agent, obs_policy, agent_actions)
            with _playback_nvtx_range(
                "playback_env_step",
                enabled=playback_nvtx_enabled,
            ):
                step_result = _step_playback_env_with_io(
                    env_raw=env_raw,
                    env_actions=env_actions,
                    next_obs_policy=next_obs_policy,
                    next_rewards=next_rewards,
                    next_dones=next_dones,
                    next_terminated=next_terminated,
                    next_truncated=next_truncated,
                    observation_views=next_observation_views,
                    reset_done=reset_done,
                )

        if play_torch_stream is not None:
            with _playback_nvtx_range(
                "playback_sync_stream",
                enabled=playback_nvtx_enabled,
            ):
                play_torch_stream.synchronize()
        return step_result

    playback_session_range = _playback_nvtx_range(
        "playback_session",
        enabled=playback_nvtx_enabled,
    )
    playback_session_range.__enter__()
    try:
        if take_video or headless_play_steps > 0:
            if playback_cuda_graph_enabled and playback_graph_runner is None:
                if play_torch_stream is None or play_wp_stream is None:
                    raise RuntimeError(
                        "play_use_cuda_graph=true requires CUDA playback streams, but they were not created."
                    )
                with _playback_nvtx_range(
                    "playback_graph_setup",
                    enabled=playback_nvtx_enabled,
                ):
                    playback_graph_runner = _HeadlessPlaybackGraphRunner(
                        PlaybackGraphRunnerOptions(
                            env=env,
                            env_raw=env_raw,
                            agent=agent,
                            device=device,
                            obs_policy=obs_policy,
                            agent_actions=agent_actions,
                            env_actions=env_actions,
                            play_torch_stream=play_torch_stream,
                            play_wp_stream=play_wp_stream,
                            stochastic_play=stochastic_play,
                        ),
                    )
                    with startup_profiler.phase("playback_graph_capture"):
                        playback_graph_runner.capture()
                _LOGGER.info(
                    "Captured 1-step headless playback CUDA graph (policy + step_with_io)."
                )
            print_play_startup_summary(
                startup_profiler.records,
                enabled=console_enabled,
            )
            if (
                fast_headless_playback or playback_graph_runner is not None
            ) and not take_video:
                with _playback_nvtx_range(
                    "playback_fast_headless",
                    enabled=playback_nvtx_enabled,
                ):
                    fast_result = _run_fast_headless_playback(
                        env=env,
                        env_raw=env_raw,
                        agent=agent,
                        obs_policy=obs_policy,
                        agent_actions=agent_actions,
                        env_actions=env_actions,
                        play_length=play_length,
                        auto_resets=auto_resets,
                        tracking_metric_frame=tracking_metric_frame,
                        tracking_metric_include_root=tracking_metric_include_root,
                        tracking_metric_summary=tracking_metric_summary,
                        tracking_require_full_horizon=tracking_require_full_horizon,
                        playback_done_counters=playback_done_counters,
                        graph_runner=playback_graph_runner,
                        play_done_policy=play_done_policy,
                    )
                episode_reward = float(fast_result["episode_reward"])
                episode_length = int(fast_result["episode_length"])
                best_episode_reward = float(fast_result["best_episode_reward"])
                best_episode_length = int(fast_result["best_episode_length"])
                episode_index = int(fast_result["episode_index"])
                tracking_failure_info = fast_result["tracking_failure_info"]
                goal_reach_summary = {
                    key: value
                    for key, value in fast_result.items()
                    if key.startswith("goal_reach")
                }
            else:
                if take_video:
                    with _playback_nvtx_range(
                        "playback_video_open",
                        enabled=playback_nvtx_enabled,
                    ):
                        viewer_width, viewer_height = getattr(
                            env_raw, "cfg", None
                        ).viewer.resolution
                        video_tile_layout = playback_render.resolve_video_tile_layout(
                            cfg,
                            env_raw,
                            tile_width=int(viewer_width),
                            tile_height=int(viewer_height),
                        )
                        camera_view_layout = (
                            playback_render.resolve_playback_camera_view_layout(
                                getattr(env_raw, "camera", None),
                                cfg,
                                default_width=int(viewer_width),
                                default_height=int(viewer_height),
                            )
                        )
                        _cam_ema_raw = cfg.get("cam_lookat_ema_alpha", None)
                        _cam_ema_alpha = (
                            float(_cam_ema_raw) if _cam_ema_raw is not None else None
                        )
                        renderer_kwargs = {
                            "env_raw": env_raw,
                            "model": getattr(env_raw, "render_model", env_raw.model),
                            "data": getattr(env_raw, "render_data", env_raw.data),
                            "width": int(camera_view_layout.view_width),
                            "height": int(camera_view_layout.view_height),
                            "camera": camera_view_layout.cameras[0],
                            "cam_lookat_ema_alpha": _cam_ema_alpha,
                        }
                        if video_render_backend == "mujoco_gl":
                            renderer_kwargs.update(
                                {
                                    "cameras": camera_view_layout.cameras,
                                    "camera_layout": camera_view_layout.layout,
                                    "camera_track_bodies": camera_view_layout.track_bodies,
                                    "camera_lookat_offsets": camera_view_layout.lookat_offsets,
                                }
                            )
                        playback_renderer = _build_playback_renderer(
                            video_render_backend=video_render_backend,
                            renderer_kwargs=renderer_kwargs,
                            video_tile_layout=video_tile_layout,
                            cfg=cfg,
                            mujoco_gl_overlay_renderer=mujoco_gl_overlay_renderer,
                            character_render_mode=character_render_mode,
                            activation=_environment_activation(env_raw),
                        )
                        video_encoder_backend = (
                            playback_render.select_video_encoder_backend(
                                backend=requested_video_encoder_backend,
                                codec=video_codec,
                                take_video=True,
                                video_render_backend=video_render_backend,
                                output_width=(
                                    video_tile_layout.output_width
                                    if video_render_backend == "warp"
                                    else camera_view_layout.output_width
                                )
                                + (overlay_panel_width if overlay_reward_terms else 0),
                                output_height=(
                                    video_tile_layout.output_height
                                    if video_render_backend == "warp"
                                    else camera_view_layout.output_height
                                ),
                            )
                        )
                        video_encoder_backend = (
                            _initialize_playback_renderer_cuda_interop(
                                playback_renderer=playback_renderer,
                                video_render_backend=video_render_backend,
                                video_encoder_backend=video_encoder_backend,
                                requested_video_encoder_backend=(
                                    requested_video_encoder_backend
                                ),
                            )
                        )
                        video_writer, video_output_path = (
                            playback_render.open_streaming_video_writer(
                                env=env,
                                log_dir=run_log_dir,
                                video_length=play_length,
                                step_dt=float(dt),
                                backend=video_encoder_backend,
                                codec=video_codec,
                                extra_width=(
                                    overlay_panel_width if overlay_reward_terms else 0
                                ),
                                output_dir=playback_render.resolve_video_output_dir(
                                    cfg,
                                    log_dir=run_log_dir,
                                ),
                                output_width=(
                                    video_tile_layout.output_width
                                    if video_render_backend == "warp"
                                    else camera_view_layout.output_width
                                ),
                                output_height=(
                                    video_tile_layout.output_height
                                    if video_render_backend == "warp"
                                    else camera_view_layout.output_height
                                ),
                                remux_timeout_seconds=cfg.get(
                                    "video_remux_timeout_seconds", None
                                ),
                            )
                        )
                    _LOGGER.info(
                        "Starting PPO playback for %d steps (video_backend=%s). "
                        "Check logs for the output video path.",
                        play_length,
                        video_encoder_backend,
                    )
                else:
                    _LOGGER.info(
                        "Starting headless PPO playback for %d steps (no rendering).",
                        play_length,
                    )
                play_bar = tqdm(range(play_length), desc="Play PPO", dynamic_ncols=True)
                for play_step_idx in play_bar:
                    with _playback_nvtx_range(
                        "playback_step",
                        enabled=playback_nvtx_enabled,
                    ):
                        with _playback_nvtx_range(
                            "playback_tracking_metrics",
                            enabled=playback_nvtx_enabled,
                        ):
                            if playback_video_step_graph_runner is not None:
                                tracking_metrics = None
                            else:
                                tracking_metrics = (
                                    tracking_metrics_graph_runner.step()
                                    if tracking_metrics_graph_runner is not None
                                    else _maybe_collect_tracking_metrics(
                                        env_raw,
                                        frame=tracking_metric_frame,
                                        include_root_in_ape=tracking_metric_include_root,
                                    )
                                )
                                _update_tracking_metric_summary(
                                    tracking_metric_summary, tracking_metrics
                                )
                        if playback_video_step_graph_runner is not None:
                            reward, _done_buf, tracking_metrics = (
                                playback_video_step_graph_runner.step()
                            )
                            terminated = playback_video_step_graph_runner.terminated
                            truncated = playback_video_step_graph_runner.truncated
                            _update_tracking_metric_summary(
                                tracking_metric_summary, tracking_metrics
                            )
                            obs = {"policy": obs_policy}
                        else:
                            obs, reward, terminated, truncated, _ = _policy_and_step(
                                use_execution_mean=True
                            )
                            copy_agent_observation_into(
                                agent=agent, observations=obs, destination=obs_policy
                            )
                        with _playback_nvtx_range(
                            "playback_episode_stats",
                            enabled=playback_nvtx_enabled,
                        ):
                            step_reward = float(reward.mean().item())
                            _accumulate_goal_reaches(env_raw, goal_reach_counts)
                            goal_reach_eval_steps += 1
                            episode_reward += step_reward
                            episode_length += 1
                            play_env_diagnostic_steps = (
                                _accumulate_play_env_diagnostic_means(
                                    env_raw,
                                    play_env_diagnostic_sums,
                                    play_env_diagnostic_steps,
                                )
                            )
                        if overlay_reward_terms:
                            with _playback_nvtx_range(
                                "playback_overlay_collect",
                                enabled=playback_nvtx_enabled,
                            ):
                                overlay_data = env_raw.get_video_overlay_plots(
                                    env_idx=0
                                )
                                appended = _append_reward_overlay_sample(
                                    reward_overlay_state, overlay_data
                                )
                            if not appended and not overlay_warning_logged:
                                _LOGGER.warning(
                                    "video overlay requested, but env.get_video_overlay_plots() returned no plot data."
                                )
                                overlay_warning_logged = True
                        if video_writer is not None:
                            assert playback_renderer is not None
                            benchmark_frame = None
                            render_start_ns = None
                            render_submit_end_ns = None
                            render_sync_start_ns = None
                            render_end_ns = None
                            if video_benchmark_sync:
                                _synchronize_video_benchmark_cuda(
                                    enabled=True,
                                    device=device,
                                )
                                render_start_ns = time.perf_counter_ns()
                            with _playback_nvtx_range(
                                f"playback_render_{video_render_backend}",
                                enabled=playback_nvtx_enabled,
                            ):
                                with _playback_nvtx_range(
                                    "playback_render_state_stage",
                                    enabled=playback_nvtx_enabled,
                                ):
                                    batch_qpos, batch_qvel = (
                                        playback_render.collect_playback_render_state_batch(
                                            env_raw,
                                            video_render_backend=video_render_backend,
                                            video_encoder_backend=video_encoder_backend,
                                            tile_layout=video_tile_layout,
                                        )
                                    )
                                render_mocap_pos = None
                                render_mocap_quat = None
                                mocap_getter_name = (
                                    "get_playback_render_mocap_batch"
                                    if video_render_backend == "warp"
                                    and video_encoder_backend == "pynvcodec"
                                    else "stage_playback_render_mocap_batch_cpu"
                                )
                                mocap_getter = getattr(
                                    env_raw,
                                    mocap_getter_name,
                                    None,
                                )
                                if callable(mocap_getter):
                                    mocap_batch = mocap_getter(batch_qpos)
                                    if mocap_batch is not None:
                                        render_mocap_pos, render_mocap_quat = (
                                            mocap_batch
                                        )
                                if video_render_backend == "warp":
                                    activation = _environment_activation(env_raw)
                                    render_activation_batch = None
                                    if activation is not None:
                                        frame_count = int(batch_qpos.shape[0])
                                        if batch_qpos.ndim == 3:
                                            tile_count = int(batch_qpos.shape[1])
                                            render_activation_batch = activation[
                                                :tile_count
                                            ][None].expand(frame_count, -1, -1)
                                        else:
                                            render_activation_batch = activation[
                                                :1
                                            ].expand(frame_count, -1)
                                    render_batch = MujocoWarpRenderBatch(
                                        qpos=batch_qpos,
                                        qvel=batch_qvel,
                                        activation=render_activation_batch,
                                        mocap_pos=render_mocap_pos,
                                        mocap_quat=render_mocap_quat,
                                    )
                                else:
                                    render_batch = MujocoGLRenderBatch(
                                        qpos=batch_qpos,
                                        qvel=batch_qvel,
                                        mocap_pos=render_mocap_pos,
                                        mocap_quat=render_mocap_quat,
                                    )
                                overlay_state = (
                                    reward_overlay_state
                                    if overlay_reward_terms
                                    else None
                                )
                                video_frames = _render_playback_video_frames(
                                    playback_renderer=playback_renderer,
                                    render_batch=render_batch,
                                    video_encoder_backend=video_encoder_backend,
                                    video_render_backend=video_render_backend,
                                    overlay_state=overlay_state,
                                    episode_length=episode_length,
                                    step_reward=step_reward,
                                    benchmark_sync_device=(
                                        device if video_benchmark_stage_sync else None
                                    ),
                                )
                                if video_benchmark_sync:
                                    render_submit_end_ns = time.perf_counter_ns()
                                with _playback_nvtx_range(
                                    f"playback_render_{video_render_backend}_sync",
                                    enabled=(
                                        playback_nvtx_enabled and video_benchmark_sync
                                    ),
                                ):
                                    if video_benchmark_sync:
                                        render_sync_start_ns = time.perf_counter_ns()
                                    _synchronize_video_benchmark_cuda(
                                        enabled=video_benchmark_sync,
                                        device=device,
                                    )
                                if video_benchmark_sync:
                                    render_end_ns = time.perf_counter_ns()
                            if video_benchmark_recorder is not None:
                                benchmark_frame = playback_render.PlaybackVideoBenchmarkFrame(
                                    render_total_ms=_elapsed_ms(
                                        render_start_ns,
                                        render_end_ns,
                                    ),
                                    render_submit_ms=_elapsed_ms(
                                        render_start_ns,
                                        render_submit_end_ns,
                                    ),
                                    render_sync_ms=_elapsed_ms(
                                        render_sync_start_ns,
                                        render_end_ns,
                                    ),
                                    render_stage_ms=(
                                        playback_render.collect_cuda_render_stage_timings(
                                            playback_renderer
                                        )
                                        if video_encoder_backend == "pynvcodec"
                                        else {}
                                    ),
                                )
                            if len(video_frames) > 0:
                                encode_start_ns = None
                                encode_submit_end_ns = None
                                encode_sync_start_ns = None
                                encode_end_ns = None
                                if video_benchmark_sync:
                                    encode_start_ns = time.perf_counter_ns()
                                with _playback_nvtx_range(
                                    "playback_video_encode",
                                    enabled=playback_nvtx_enabled,
                                ):
                                    with _playback_nvtx_range(
                                        "playback_video_encode_submit",
                                        enabled=(
                                            playback_nvtx_enabled
                                            and video_benchmark_sync
                                        ),
                                    ):
                                        if video_encoder_backend == "pynvcodec":
                                            for video_frame in video_frames:
                                                video_writer.append_frame(video_frame)
                                        else:
                                            for video_frame in video_frames:
                                                video_writer.append_data(video_frame)
                                    if video_benchmark_sync:
                                        encode_submit_end_ns = time.perf_counter_ns()
                                    with _playback_nvtx_range(
                                        "playback_video_encode_post_sync",
                                        enabled=(
                                            playback_nvtx_enabled
                                            and video_benchmark_sync
                                        ),
                                    ):
                                        if video_benchmark_sync:
                                            encode_sync_start_ns = (
                                                time.perf_counter_ns()
                                            )
                                        _synchronize_video_benchmark_cuda(
                                            enabled=video_benchmark_sync,
                                            device=device,
                                        )
                                    if video_benchmark_sync:
                                        encode_end_ns = time.perf_counter_ns()
                                if benchmark_frame is not None:
                                    benchmark_frame.encode_total_ms = _elapsed_ms(
                                        encode_start_ns,
                                        encode_end_ns,
                                    )
                                    benchmark_frame.encode_submit_ms = _elapsed_ms(
                                        encode_start_ns,
                                        encode_submit_end_ns,
                                    )
                                    benchmark_frame.encode_sync_ms = _elapsed_ms(
                                        encode_sync_start_ns,
                                        encode_end_ns,
                                    )
                            elif not render_warning_logged:
                                _LOGGER.warning(
                                    "Skipping video frame: mujoco_gl renderer returned no frames",
                                )
                                render_warning_logged = True
                            if (
                                video_benchmark_recorder is not None
                                and benchmark_frame is not None
                            ):
                                video_benchmark_recorder.record(benchmark_frame)
                        play_bar.set_postfix(
                            reward=f"{step_reward:.3f}",
                            ep_len=episode_length,
                            ep_rew=f"{episode_reward:.2f}",
                            ape=(
                                f"{_tracking_metric_scalar(tracking_metrics, 'ape'):.4f}"
                                if _tracking_metric_scalar(tracking_metrics, "ape")
                                is not None
                                else "-"
                            ),
                            ee=(
                                f"{_tracking_metric_scalar(tracking_metrics, 'ee_error'):.4f}"
                                if _tracking_metric_scalar(tracking_metrics, "ee_error")
                                is not None
                                else "-"
                            ),
                        )
                        # DirectRLEnv auto-resets terminated worlds inside step().
                        if _episode_done(terminated, truncated):
                            done_step = play_step_idx + 1
                            done_before_horizon = done_step < play_length
                            terminated_done, truncated_done = (
                                _update_playback_done_counters(
                                    terminated,
                                    truncated,
                                    playback_done_counters,
                                )
                            )
                            if episode_length > best_episode_length or (
                                episode_length == best_episode_length
                                and episode_reward > best_episode_reward
                            ):
                                best_episode_length = int(episode_length)
                                best_episode_reward = float(episode_reward)
                            if tracking_metrics is not None:
                                ape_scalar = _tracking_metric_scalar(
                                    tracking_metrics, "ape"
                                )
                                ee_scalar = _tracking_metric_scalar(
                                    tracking_metrics, "ee_error"
                                )
                                _LOGGER.info(
                                    "Episode %d finished: length=%d total_reward=%.3f ape=%.6f ee_error=%.6f",
                                    episode_index,
                                    episode_length,
                                    episode_reward,
                                    ape_scalar,
                                    ee_scalar,
                                )
                            else:
                                _LOGGER.info(
                                    "Episode %d finished: length=%d total_reward=%.3f",
                                    episode_index,
                                    episode_length,
                                    episode_reward,
                                )
                            if (
                                done_before_horizon
                                and tracking_require_full_horizon
                                and not _episode_done_naturally(
                                    env_raw, terminated, truncated
                                )
                            ):
                                tracking_failure_info = {
                                    "step": int(done_step),
                                    "terminated": terminated_done,
                                    "truncated": truncated_done,
                                    "play_length": int(play_length),
                                }
                                _LOGGER.warning(
                                    "Tracking evaluation invalidated: episode ended at step %d/%d before the requested horizon.",
                                    done_step,
                                    play_length,
                                )
                                break
                            if _playback_done_policy_stops(play_done_policy):
                                break
                            if overlay_reward_terms:
                                _clear_reward_overlay_state(reward_overlay_state)
                            episode_index += 1
                            episode_reward = 0.0
                            episode_length = 0
                            if playback_renderer is not None:
                                reset_smooth_lookat = getattr(
                                    playback_renderer,
                                    "reset_smooth_lookat",
                                    None,
                                )
                                if callable(reset_smooth_lookat):
                                    reset_smooth_lookat()
                            if not auto_resets:
                                with _playback_nvtx_range(
                                    "playback_reset_env",
                                    enabled=playback_nvtx_enabled,
                                ):
                                    obs, _ = env.reset()
                                copy_agent_observation_into(
                                    agent=agent,
                                    observations=obs,
                                    destination=obs_policy,
                                )
        else:
            print_play_startup_summary(
                startup_profiler.records,
                enabled=console_enabled,
            )
            if viewer_backend == "viser":
                viewer_port = int(cfg.get("viewer_port", 8080))
                viser_session = ViserPlaybackSession(
                    cfg=cfg,
                    env_raw=env_raw,
                    step_dt=float(dt),
                    overlay_enabled=overlay_reward_terms,
                    overlay_history_seconds=overlay_history_seconds,
                )
                _LOGGER.info(
                    "Starting PPO playback with viser at http://127.0.0.1:%d. "
                    "Use the web Stop button or interrupt the process to exit.",
                    viewer_port,
                )
                viewer_step = 0
                step_reward = 0.0
                tracking_metrics = None
                viewer_done_active = False
                viser_session.update_scene()
                viser_session.update_status(
                    step=viewer_step,
                    step_reward=step_reward,
                    episode_length=episode_length,
                    episode_reward=episode_reward,
                )
                while viser_session.is_running():
                    start_time = time.time()

                    if viser_session.consume_reset_request():
                        with _playback_nvtx_range(
                            "playback_reset_env",
                            enabled=playback_nvtx_enabled,
                        ):
                            obs, _ = env.reset()
                        copy_agent_observation_into(
                            agent=agent, observations=obs, destination=obs_policy
                        )
                        episode_reward = 0.0
                        episode_length = 0
                        viewer_step = 0
                        tracking_metrics = None
                        viewer_done_active = False
                        if overlay_reward_terms:
                            _clear_reward_overlay_state(reward_overlay_state)
                            viser_session.clear_overlay()
                        viser_session.update_scene()
                        viser_session.update_status(
                            step=viewer_step,
                            step_reward=0.0,
                            episode_length=episode_length,
                            episode_reward=episode_reward,
                        )
                        continue

                    if not viser_session.should_step():
                        viser_session.update_status(
                            step=viewer_step,
                            step_reward=step_reward,
                            episode_length=episode_length,
                            episode_reward=episode_reward,
                        )
                        time.sleep(0.02)
                        continue

                    with _playback_nvtx_range(
                        "playback_viser_step",
                        enabled=playback_nvtx_enabled,
                    ):
                        obs, reward, terminated, truncated, _ = _policy_and_step(
                            use_execution_mean=False,
                            # Preserve terminal simulator state until metrics and
                            # overlays have consumed it. Viewer reset is explicit
                            # below after the done record is finalized.
                            reset_done=False,
                        )
                        copy_agent_observation_into(
                            agent=agent, observations=obs, destination=obs_policy
                        )
                        with _playback_nvtx_range(
                            "playback_episode_stats",
                            enabled=playback_nvtx_enabled,
                        ):
                            step_reward = float(reward.mean().item())
                            episode_reward += step_reward
                            episode_length += 1
                            viewer_step += 1
                            play_env_diagnostic_steps = (
                                _accumulate_play_env_diagnostic_means(
                                    env_raw,
                                    play_env_diagnostic_sums,
                                    play_env_diagnostic_steps,
                                )
                            )
                        with _playback_nvtx_range(
                            "playback_tracking_metrics",
                            enabled=playback_nvtx_enabled,
                        ):
                            tracking_metrics = (
                                tracking_metrics_graph_runner.step()
                                if tracking_metrics_graph_runner is not None
                                else _maybe_collect_tracking_metrics(
                                    env_raw,
                                    frame=tracking_metric_frame,
                                    include_root_in_ape=tracking_metric_include_root,
                                )
                            )
                            _update_tracking_metric_summary(
                                tracking_metric_summary, tracking_metrics
                            )
                        if overlay_reward_terms:
                            with _playback_nvtx_range(
                                "playback_overlay_collect",
                                enabled=playback_nvtx_enabled,
                            ):
                                overlay_data = env_raw.get_video_overlay_plots(
                                    env_idx=0
                                )
                                appended = _append_reward_overlay_sample(
                                    reward_overlay_state, overlay_data
                                )
                            if not appended and not overlay_warning_logged:
                                _LOGGER.warning(
                                    "viser reward overlay requested, but "
                                    "env.get_video_overlay_plots() returned no plot data."
                                )
                                overlay_warning_logged = True
                            if appended:
                                viser_session.update_overlay(reward_overlay_state)
                        with _playback_nvtx_range(
                            "playback_viser_sync",
                            enabled=playback_nvtx_enabled,
                        ):
                            viser_session.update_scene(animate_substeps=True)
                        viser_tracking_metrics = (
                            {
                                "ape": _tracking_metric_scalar(tracking_metrics, "ape"),
                                "ee_error": _tracking_metric_scalar(
                                    tracking_metrics, "ee_error"
                                ),
                            }
                            if tracking_metrics is not None
                            else None
                        )
                        viser_session.update_status(
                            step=viewer_step,
                            step_reward=step_reward,
                            episode_length=episode_length,
                            episode_reward=episode_reward,
                            tracking_metrics=viser_tracking_metrics,
                        )
                    viser_session.mark_step_consumed()

                    done_now = _episode_done(terminated, truncated)
                    if done_now:
                        if not viewer_done_active:
                            _update_playback_done_counters(
                                terminated,
                                truncated,
                                playback_done_counters,
                            )
                            if episode_length > best_episode_length or (
                                episode_length == best_episode_length
                                and episode_reward > best_episode_reward
                            ):
                                best_episode_length = int(episode_length)
                                best_episode_reward = float(episode_reward)
                            if tracking_metrics is not None:
                                ape_scalar = _tracking_metric_scalar(
                                    tracking_metrics, "ape"
                                )
                                ee_scalar = _tracking_metric_scalar(
                                    tracking_metrics, "ee_error"
                                )
                                _LOGGER.info(
                                    "Episode %d done signal: length=%d total_reward=%.3f ape=%.6f ee_error=%.6f",
                                    episode_index,
                                    episode_length,
                                    episode_reward,
                                    ape_scalar,
                                    ee_scalar,
                                )
                            else:
                                _LOGGER.info(
                                    "Episode %d done signal: length=%d total_reward=%.3f",
                                    episode_index,
                                    episode_length,
                                    episode_reward,
                                )
                        viewer_done_active = True
                        if viewer_reset_on_done:
                            episode_index += 1
                            episode_reward = 0.0
                            episode_length = 0
                            viewer_done_active = False
                            if overlay_reward_terms:
                                _clear_reward_overlay_state(reward_overlay_state)
                                viser_session.clear_overlay()
                            obs, _ = env.reset()
                            copy_agent_observation_into(
                                agent=agent, observations=obs, destination=obs_policy
                            )
                            viser_session.update_scene()
                    else:
                        viewer_done_active = False
                    if (
                        not done_now
                    ) and viewer_step % viewer_status_interval_steps == 0:
                        if tracking_metrics is not None:
                            ape_scalar = _tracking_metric_scalar(
                                tracking_metrics, "ape"
                            )
                            ee_scalar = _tracking_metric_scalar(
                                tracking_metrics, "ee_error"
                            )
                            _LOGGER.info(
                                "Playback step=%d reward=%.3f episode_length=%d episode_reward=%.3f ape=%.6f ee_error=%.6f",
                                viewer_step,
                                step_reward,
                                episode_length,
                                episode_reward,
                                ape_scalar,
                                ee_scalar,
                            )
                        else:
                            _LOGGER.info(
                                "Playback step=%d reward=%.3f episode_length=%d episode_reward=%.3f",
                                viewer_step,
                                step_reward,
                                episode_length,
                                episode_reward,
                            )

                    elapsed = time.time() - start_time
                    sleep_time = (
                        float(dt) / max(float(viser_session.speed), 1.0e-6)
                    ) - elapsed
                    if sleep_time > 0:
                        with _playback_nvtx_range(
                            "playback_viser_sleep",
                            enabled=playback_nvtx_enabled,
                        ):
                            time.sleep(sleep_time)
            else:
                import mujoco.viewer

                model = env_raw.model
                data = env_raw.data
                with mujoco.viewer.launch_passive(model, data) as viewer:
                    _LOGGER.info(
                        "Starting PPO playback. Close the viewer window to exit."
                    )
                    viewer_step = 0
                    viewer_done_active = False
                    while viewer.is_running():
                        start_time = time.time()

                        with _playback_nvtx_range(
                            "playback_viewer_step",
                            enabled=playback_nvtx_enabled,
                        ):
                            obs, reward, terminated, truncated, _ = _policy_and_step(
                                use_execution_mean=False,
                                reset_done=False,
                            )
                            copy_agent_observation_into(
                                agent=agent, observations=obs, destination=obs_policy
                            )
                            with _playback_nvtx_range(
                                "playback_episode_stats",
                                enabled=playback_nvtx_enabled,
                            ):
                                step_reward = float(reward.mean().item())
                                episode_reward += step_reward
                                episode_length += 1
                                viewer_step += 1
                                play_env_diagnostic_steps = (
                                    _accumulate_play_env_diagnostic_means(
                                        env_raw,
                                        play_env_diagnostic_sums,
                                        play_env_diagnostic_steps,
                                    )
                                )
                            with _playback_nvtx_range(
                                "playback_tracking_metrics",
                                enabled=playback_nvtx_enabled,
                            ):
                                tracking_metrics = (
                                    tracking_metrics_graph_runner.step()
                                    if tracking_metrics_graph_runner is not None
                                    else _maybe_collect_tracking_metrics(
                                        env_raw,
                                        frame=tracking_metric_frame,
                                        include_root_in_ape=tracking_metric_include_root,
                                    )
                                )
                                _update_tracking_metric_summary(
                                    tracking_metric_summary, tracking_metrics
                                )

                            with _playback_nvtx_range(
                                "playback_viewer_copy",
                                enabled=playback_nvtx_enabled,
                            ):
                                mjw.get_data_into(data, model, env_raw.mjw_data)
                            with _playback_nvtx_range(
                                "playback_viewer_sync",
                                enabled=playback_nvtx_enabled,
                            ):
                                viewer.sync()

                        done_now = _episode_done(terminated, truncated)
                        if done_now:
                            if not viewer_done_active:
                                _update_playback_done_counters(
                                    terminated,
                                    truncated,
                                    playback_done_counters,
                                )
                                if episode_length > best_episode_length or (
                                    episode_length == best_episode_length
                                    and episode_reward > best_episode_reward
                                ):
                                    best_episode_length = int(episode_length)
                                    best_episode_reward = float(episode_reward)
                                if tracking_metrics is not None:
                                    ape_scalar = _tracking_metric_scalar(
                                        tracking_metrics, "ape"
                                    )
                                    ee_scalar = _tracking_metric_scalar(
                                        tracking_metrics, "ee_error"
                                    )
                                    _LOGGER.info(
                                        "Episode %d done signal: length=%d total_reward=%.3f ape=%.6f ee_error=%.6f",
                                        episode_index,
                                        episode_length,
                                        episode_reward,
                                        ape_scalar,
                                        ee_scalar,
                                    )
                                else:
                                    _LOGGER.info(
                                        "Episode %d done signal: length=%d total_reward=%.3f",
                                        episode_index,
                                        episode_length,
                                        episode_reward,
                                    )
                            viewer_done_active = True
                            if viewer_reset_on_done:
                                episode_index += 1
                                episode_reward = 0.0
                                episode_length = 0
                                viewer_done_active = False
                                obs, _ = env.reset()
                                copy_agent_observation_into(
                                    agent=agent,
                                    observations=obs,
                                    destination=obs_policy,
                                )
                        else:
                            viewer_done_active = False

                        if (
                            not done_now
                        ) and viewer_step % viewer_status_interval_steps == 0:
                            if tracking_metrics is not None:
                                ape_scalar = _tracking_metric_scalar(
                                    tracking_metrics, "ape"
                                )
                                ee_scalar = _tracking_metric_scalar(
                                    tracking_metrics, "ee_error"
                                )
                                _LOGGER.info(
                                    "Playback step=%d reward=%.3f episode_length=%d episode_reward=%.3f ape=%.6f ee_error=%.6f",
                                    viewer_step,
                                    step_reward,
                                    episode_length,
                                    episode_reward,
                                    ape_scalar,
                                    ee_scalar,
                                )
                            else:
                                _LOGGER.info(
                                    "Playback step=%d reward=%.3f episode_length=%d episode_reward=%.3f",
                                    viewer_step,
                                    step_reward,
                                    episode_length,
                                    episode_reward,
                                )

                        elapsed = time.time() - start_time
                        sleep_time = dt - elapsed
                        if sleep_time > 0:
                            with _playback_nvtx_range(
                                "playback_viewer_sleep",
                                enabled=playback_nvtx_enabled,
                            ):
                                time.sleep(sleep_time)
    finally:
        video_output_path = _finalize_playback_lifecycle(
            _PlaybackLifecycleResources(
                env=env,
                session_range=playback_session_range,
                playback_nvtx_enabled=playback_nvtx_enabled,
                video_writer=video_writer,
                playback_renderer=playback_renderer,
                viser_session=viser_session,
                video_output_path=video_output_path,
            ),
            close_environment=False,
        )

    result: dict[str, str] = {}
    if episode_length > best_episode_length or (
        episode_length == best_episode_length and episode_reward > best_episode_reward
    ):
        best_episode_length = int(episode_length)
        best_episode_reward = float(episode_reward)
    reported_episode_length = int(best_episode_length)
    reported_episode_reward = float(best_episode_reward)
    result["episode_length"] = str(reported_episode_length)
    result["episode_reward"] = f"{reported_episode_reward:.6f}"
    result["episode_finish_count"] = str(playback_done_counters["episode_finish_count"])
    result["terminated_count"] = str(playback_done_counters["terminated_count"])
    result["truncated_count"] = str(playback_done_counters["truncated_count"])
    if goal_reach_summary is None:
        goal_reach_summary = _summarize_goal_reaches(
            goal_reach_counts,
            eval_steps=goal_reach_eval_steps,
        )
    if bool(goal_reach_summary.get("goal_reach_metric_available", False)):
        goal_reach_total = int(goal_reach_summary["goal_reach_total"])
        goal_reaches_per_env = float(goal_reach_summary["goal_reaches_per_env"])
        goal_reach_steps = int(goal_reach_summary["goal_reach_eval_steps"])
        goal_reach_num_envs = int(goal_reach_summary["goal_reach_num_envs"])
        _LOGGER.info(
            "GOAL_REACH_SUMMARY checkpoint=%s steps=%d num_envs=%d total=%d mean_per_env=%.6f",
            resume_path.as_posix(),
            goal_reach_steps,
            goal_reach_num_envs,
            goal_reach_total,
            goal_reaches_per_env,
        )
        result["goal_reach_steps"] = str(goal_reach_steps)
        result["goal_reach_num_envs"] = str(goal_reach_num_envs)
        result["goal_reach_total"] = str(goal_reach_total)
        result["goal_reaches_per_env"] = f"{goal_reaches_per_env:.6f}"
    if play_env_diagnostic_steps > 0:
        result["play_env_diagnostic_steps"] = str(play_env_diagnostic_steps)
        for key, total in sorted(play_env_diagnostic_sums.items()):
            result[f"play_env_{key}"] = (
                f"{total / float(play_env_diagnostic_steps):.6f}"
            )
    tracking_summary = _finalize_tracking_metric_summary(tracking_metric_summary)
    tracking_valid = tracking_failure_info is None
    result["tracking_valid"] = "true" if tracking_valid else "false"
    if tracking_failure_info is not None:
        result["tracking_failure_step"] = str(tracking_failure_info["step"])
        result["tracking_failure_play_length"] = str(
            tracking_failure_info["play_length"]
        )
        result["tracking_failure_terminated"] = (
            "true" if bool(tracking_failure_info["terminated"]) else "false"
        )
        result["tracking_failure_truncated"] = (
            "true" if bool(tracking_failure_info["truncated"]) else "false"
        )
    if tracking_summary is not None and tracking_valid:
        _LOGGER.info(
            "TRACKING_METRICS_SUMMARY checkpoint=%s deterministic_reset=%s frame=%s steps=%d ape_mean=%.6f ape_last=%.6f ee_mean=%.6f ee_last=%.6f vel_mean=%.6f vel_last=%.6f accel_mean=%.6f accel_last=%.6f",
            resume_path.as_posix(),
            bool(cfg.env.get("reset", {}).get("deterministic", False)),
            tracking_metric_frame,
            int(tracking_summary["steps"]),
            tracking_summary["ape_mean"],
            tracking_summary["ape_last"],
            tracking_summary["ee_mean"],
            tracking_summary["ee_last"],
            tracking_summary["vel_mean"],
            tracking_summary["vel_last"],
            tracking_summary["accel_mean"],
            tracking_summary["accel_last"],
        )
        result["tracking_frame"] = tracking_metric_frame
        result["tracking_ape_mean"] = f"{tracking_summary['ape_mean']:.6f}"
        result["tracking_ape_last"] = f"{tracking_summary['ape_last']:.6f}"
        result["tracking_ee_mean"] = f"{tracking_summary['ee_mean']:.6f}"
        result["tracking_ee_last"] = f"{tracking_summary['ee_last']:.6f}"
        result["tracking_vel_mean"] = f"{tracking_summary['vel_mean']:.6f}"
        result["tracking_vel_last"] = f"{tracking_summary['vel_last']:.6f}"
        result["tracking_accel_mean"] = f"{tracking_summary['accel_mean']:.6f}"
        result["tracking_accel_last"] = f"{tracking_summary['accel_last']:.6f}"
    elif tracking_summary is not None and not tracking_valid:
        _LOGGER.info(
            "TRACKING_METRICS_INVALID checkpoint=%s deterministic_reset=%s frame=%s steps=%d failure_step=%d ape_mean_pre_failure=%.6f ape_last_pre_failure=%.6f ee_mean_pre_failure=%.6f ee_last_pre_failure=%.6f vel_mean_pre_failure=%.6f vel_last_pre_failure=%.6f accel_mean_pre_failure=%.6f accel_last_pre_failure=%.6f",
            resume_path.as_posix(),
            bool(cfg.env.get("reset", {}).get("deterministic", False)),
            tracking_metric_frame,
            int(tracking_summary["steps"]),
            int(tracking_failure_info["step"]) if tracking_failure_info else -1,
            tracking_summary["ape_mean"],
            tracking_summary["ape_last"],
            tracking_summary["ee_mean"],
            tracking_summary["ee_last"],
            tracking_summary["vel_mean"],
            tracking_summary["vel_last"],
            tracking_summary["accel_mean"],
            tracking_summary["accel_last"],
        )
        result["tracking_ape_mean_pre_failure"] = f"{tracking_summary['ape_mean']:.6f}"
        result["tracking_ape_last_pre_failure"] = f"{tracking_summary['ape_last']:.6f}"
        result["tracking_ee_mean_pre_failure"] = f"{tracking_summary['ee_mean']:.6f}"
        result["tracking_ee_last_pre_failure"] = f"{tracking_summary['ee_last']:.6f}"
        result["tracking_vel_mean_pre_failure"] = f"{tracking_summary['vel_mean']:.6f}"
        result["tracking_vel_last_pre_failure"] = f"{tracking_summary['vel_last']:.6f}"
        result["tracking_accel_mean_pre_failure"] = (
            f"{tracking_summary['accel_mean']:.6f}"
        )
        result["tracking_accel_last_pre_failure"] = (
            f"{tracking_summary['accel_last']:.6f}"
        )
    elif tracking_metrics_available:
        _LOGGER.warning(
            "Tracking metrics were available but no playback steps were recorded."
        )
    else:
        _LOGGER.info(
            "Env does not expose get_reference_tracking_metrics(); skipping tracking metric summary."
        )
    if video_output_path is not None:
        if video_output_path.exists() and reported_episode_length > 0:
            renamed_path = video_output_path.with_name(
                f"{video_output_path.stem}_ep{reported_episode_length}_rew{reported_episode_reward:.2f}{video_output_path.suffix}"
            )
            if renamed_path != video_output_path:
                video_output_path.rename(renamed_path)
                video_output_path = renamed_path
        result["video_output_path"] = video_output_path.as_posix()
    if video_benchmark_recorder is not None:
        benchmark_metrics = video_benchmark_recorder.to_result_metrics()
        result.update(benchmark_metrics)
        render_avg = benchmark_metrics.get("video_benchmark_render_total_avg_ms")
        render_med = benchmark_metrics.get("video_benchmark_render_total_median_ms")
        sync_avg = benchmark_metrics.get("video_benchmark_render_sync_avg_ms")
        encode_avg = benchmark_metrics.get("video_benchmark_encode_total_avg_ms")
        _LOGGER.info(
            "VIDEO_BENCHMARK_SUMMARY frames=%s warmup=%s render_total_avg_ms=%s render_total_median_ms=%s render_sync_avg_ms=%s encode_total_avg_ms=%s",
            benchmark_metrics.get("video_benchmark_frames", "0"),
            benchmark_metrics.get("video_benchmark_warmup_frames", "0"),
            render_avg,
            render_med,
            sync_avg,
            encode_avg,
        )
        if video_output_path is not None:
            benchmark_output_path = video_output_path.with_suffix(".benchmark.json")
            try:
                _write_video_benchmark_summary(
                    recorder=video_benchmark_recorder,
                    output_path=benchmark_output_path,
                    checkpoint_path=resume_path,
                    video_output_path=video_output_path,
                    video_render_backend=video_render_backend,
                    video_encoder_backend=video_encoder_backend,
                    video_codec=video_codec,
                    tile_layout=video_tile_layout,
                )
                result["video_benchmark_output_path"] = benchmark_output_path.as_posix()
                _LOGGER.info(
                    "Saved video benchmark summary to: %s",
                    benchmark_output_path.as_posix(),
                )
            except Exception:
                _LOGGER.exception("Failed to write video benchmark summary.")
    result["resolved_seed"] = str(resolved_seed)
    if eager_fallback.recorded:
        # Only present when something degraded, so a clean run's payload is
        # unchanged and a degraded one cannot be mistaken for it.
        result["eager_fallbacks"] = eager_fallback.as_payload_value()
    playback_result = PlaybackResult.from_cli_payload(
        result,
        checkpoint_path=resume_path,
        log_dir=run_log_dir,
    )
    print_play_result(
        playback_result,
        max_episode_length=max_episode_length,
        enabled=console_enabled,
    )
    if should_emit_playback_result_payload(
        emit_playback_result_payload=bool(
            cfg.get("emit_playback_result_payload", False)
        ),
    ):
        # The parent process parses this off stdout; see
        # `parse_playback_result_stdout`.
        print(playback_result.to_log_line())
        print(playback_result.to_json_line())
    _LOGGER.info(playback_result.to_log_line())
    _LOGGER.info(playback_result.to_json_line())
    return playback_result


def run_playback(cfg: SemanticConfig) -> None:
    log_root_path = resolve_project_log_root("ppo", str(cfg.experiment["directory"]))
    resume_path = resolve_checkpoint_path(cfg.play.get("checkpoint"), log_root_path)
    run_playback_request(
        PlaybackRequest(
            cfg,
            checkpoint_path=resume_path,
            log_dir=resume_path.parent.parent,
            dotlist_overrides=task_dotlist_overrides(),
        )
    )
