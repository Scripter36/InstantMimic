from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from omegaconf import DictConfig

from instantmimic.rendering.video_recording import resolve_render_output_fps
from instantmimic.utils.project_paths import resolve_project_path

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaybackVideoTileLayout:
    env_count: int
    cols: int
    rows: int
    tile_width: int
    tile_height: int

    @property
    def output_width(self) -> int:
        return self.cols * self.tile_width

    @property
    def output_height(self) -> int:
        return self.rows * self.tile_height


@dataclass(frozen=True)
class PlaybackCameraViewLayout:
    """Camera views and per-view dimensions for one composed playback frame."""

    cameras: tuple[Any, ...]
    layout: str
    view_width: int
    view_height: int
    track_bodies: tuple[tuple[str, ...], ...] = ()
    lookat_offsets: tuple[tuple[float, float, float], ...] = ()

    @property
    def cols(self) -> int:
        return len(self.cameras) if self.layout == "horizontal" else 1

    @property
    def rows(self) -> int:
        return len(self.cameras) if self.layout == "vertical" else 1

    @property
    def output_width(self) -> int:
        return self.cols * self.view_width

    @property
    def output_height(self) -> int:
        return self.rows * self.view_height


@dataclass
class PlaybackVideoBenchmarkFrame:
    render_total_ms: float | None = None
    render_submit_ms: float | None = None
    render_sync_ms: float | None = None
    render_stage_ms: dict[str, float | None] = field(default_factory=dict)
    encode_total_ms: float | None = None
    encode_submit_ms: float | None = None
    encode_sync_ms: float | None = None


def _append_finite_ms(values: list[float], value: float | None) -> None:
    if value is None:
        return
    scalar = float(value)
    if math.isfinite(scalar) and scalar >= 0.0:
        values.append(scalar)


def collect_cuda_render_stage_timings(renderer: Any) -> dict[str, float]:
    raw_timings = getattr(renderer, "_last_cuda_stage_timings_ms", None)
    if not isinstance(raw_timings, dict):
        return {}
    timings: dict[str, float] = {}
    for stage_name, value in raw_timings.items():
        if value is None:
            continue
        scalar = float(value)
        if math.isfinite(scalar) and scalar >= 0.0:
            timings[str(stage_name)] = scalar
    return timings


def _percentile_nearest(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return math.nan
    rank = int(math.ceil((float(percentile) / 100.0) * len(sorted_values)))
    index = min(len(sorted_values) - 1, max(0, rank - 1))
    return sorted_values[index]


def _median(sorted_values: list[float]) -> float:
    if not sorted_values:
        return math.nan
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def _summarize_ms(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    sorted_values = sorted(values)
    return {
        "count": len(sorted_values),
        "avg_ms": sum(sorted_values) / float(len(sorted_values)),
        "median_ms": _median(sorted_values),
        "p95_ms": _percentile_nearest(sorted_values, 95.0),
        "min_ms": sorted_values[0],
        "max_ms": sorted_values[-1],
    }


@dataclass
class PlaybackVideoBenchmarkRecorder:
    warmup_frames: int = 1
    frames_seen: int = 0
    frames_recorded: int = 0
    render_total_ms: list[float] = field(default_factory=list)
    render_submit_ms: list[float] = field(default_factory=list)
    render_sync_ms: list[float] = field(default_factory=list)
    render_stage_ms: dict[str, list[float]] = field(default_factory=dict)
    encode_total_ms: list[float] = field(default_factory=list)
    encode_submit_ms: list[float] = field(default_factory=list)
    encode_sync_ms: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.warmup_frames = max(0, int(self.warmup_frames))

    def record(self, frame: PlaybackVideoBenchmarkFrame) -> bool:
        self.frames_seen += 1
        if self.frames_seen <= self.warmup_frames:
            return False

        self.frames_recorded += 1
        _append_finite_ms(self.render_total_ms, frame.render_total_ms)
        _append_finite_ms(self.render_submit_ms, frame.render_submit_ms)
        _append_finite_ms(self.render_sync_ms, frame.render_sync_ms)
        for stage_name, stage_ms in frame.render_stage_ms.items():
            values = self.render_stage_ms.setdefault(str(stage_name), [])
            _append_finite_ms(values, stage_ms)
        _append_finite_ms(self.encode_total_ms, frame.encode_total_ms)
        _append_finite_ms(self.encode_submit_ms, frame.encode_submit_ms)
        _append_finite_ms(self.encode_sync_ms, frame.encode_sync_ms)
        return True

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "warmup_frames": int(self.warmup_frames),
            "frames_seen": int(self.frames_seen),
            "frames_recorded": int(self.frames_recorded),
        }
        for key, values in (
            ("render_total_ms", self.render_total_ms),
            ("render_submit_ms", self.render_submit_ms),
            ("render_sync_ms", self.render_sync_ms),
            ("encode_total_ms", self.encode_total_ms),
            ("encode_submit_ms", self.encode_submit_ms),
            ("encode_sync_ms", self.encode_sync_ms),
        ):
            stats = _summarize_ms(values)
            if stats is not None:
                summary[key] = stats
        stage_summary: dict[str, Any] = {}
        for stage_name, values in sorted(self.render_stage_ms.items()):
            stats = _summarize_ms(values)
            if stats is not None:
                stage_summary[stage_name] = stats
        if stage_summary:
            summary["render_stage_ms"] = stage_summary
        return summary

    def to_result_metrics(self) -> dict[str, str]:
        summary = self.to_summary()
        metrics = {
            "video_benchmark_frames": str(int(summary["frames_recorded"])),
            "video_benchmark_warmup_frames": str(int(summary["warmup_frames"])),
        }
        for source_key, metric_key in (
            ("render_total_ms", "render_total"),
            ("render_submit_ms", "render_submit"),
            ("render_sync_ms", "render_sync"),
            ("encode_total_ms", "encode_total"),
            ("encode_submit_ms", "encode_submit"),
            ("encode_sync_ms", "encode_sync"),
        ):
            stats = summary.get(source_key)
            if not isinstance(stats, dict):
                continue
            for stat_name in ("avg_ms", "median_ms", "p95_ms", "min_ms", "max_ms"):
                value = stats.get(stat_name)
                if value is not None:
                    metrics[f"video_benchmark_{metric_key}_{stat_name}"] = (
                        f"{float(value):.3f}"
                    )
        stage_summary = summary.get("render_stage_ms")
        if isinstance(stage_summary, dict):
            for stage_name, stats in sorted(stage_summary.items()):
                if not isinstance(stats, dict):
                    continue
                safe_stage_name = "".join(
                    ch if ch.isalnum() else "_"
                    for ch in str(stage_name).strip().lower()
                ).strip("_")
                if not safe_stage_name:
                    continue
                for stat_name in ("avg_ms", "median_ms", "p95_ms", "min_ms", "max_ms"):
                    value = stats.get(stat_name)
                    if value is not None:
                        metrics[
                            f"video_benchmark_render_stage_{safe_stage_name}_{stat_name}"
                        ] = f"{float(value):.3f}"
        return metrics


def resolve_video_tile_layout(
    cfg: DictConfig,
    env_raw: Any,
    *,
    tile_width: int,
    tile_height: int,
) -> PlaybackVideoTileLayout:
    requested = max(1, int(cfg.get("video_tile_envs", 1) or 1))
    return build_video_tile_layout(
        requested_envs=requested,
        available_envs=int(getattr(env_raw, "num_envs", requested)),
        requested_cols=cfg.get("video_tile_cols", None),
        tile_width=tile_width,
        tile_height=tile_height,
    )


def build_video_tile_layout(
    *,
    requested_envs: int,
    available_envs: int,
    requested_cols: int | None,
    tile_width: int,
    tile_height: int,
) -> PlaybackVideoTileLayout:
    requested = max(1, int(requested_envs))
    available_envs = max(1, int(available_envs))
    env_count = min(requested, available_envs)
    if requested > available_envs:
        _LOGGER.warning(
            "video_tile_envs=%d exceeds env.num_envs=%d; recording %d tile(s).",
            requested,
            available_envs,
            env_count,
        )
    cols = int(requested_cols) if requested_cols is not None else 0
    if cols <= 0:
        cols = int(math.ceil(math.sqrt(float(env_count))))
    cols = max(1, min(cols, env_count))
    rows = int(math.ceil(float(env_count) / float(cols)))
    return PlaybackVideoTileLayout(
        env_count=env_count,
        cols=cols,
        rows=rows,
        tile_width=int(tile_width),
        tile_height=int(tile_height),
    )


def collect_playback_render_state_batch(
    env_raw: Any,
    *,
    video_render_backend: str,
    video_encoder_backend: str,
    tile_layout: PlaybackVideoTileLayout,
) -> tuple[Any, Any]:
    if video_render_backend == "warp":
        if video_encoder_backend == "pynvcodec":
            state_getter = getattr(
                env_raw,
                "get_playback_render_tile_state_batch",
                None,
            )
            if not callable(state_getter):
                raise RuntimeError(
                    "video_render_backend=warp with video_encoder_backend=pynvcodec "
                    "requires an env that can expose tiled playback render state on device."
                )
            return state_getter(tile_layout.env_count)

        state_getter = getattr(
            env_raw,
            "stage_playback_render_tile_state_batch_cpu",
            None,
        )
        if not callable(state_getter):
            raise RuntimeError(
                "video_render_backend=warp requires an env that can stage tiled "
                "playback render state."
            )
        return state_getter(tile_layout.env_count)

    if video_render_backend == "mujoco_gl":
        if tile_layout.env_count > 1:
            raise RuntimeError(
                "video_tile_envs > 1 is supported only by video_render_backend=warp."
            )
        state_getter = getattr(env_raw, "stage_playback_render_state_batch_cpu", None)
        if not callable(state_getter):
            raise RuntimeError(
                "video_render_backend=mujoco_gl requires an env that can stage "
                "playback render state."
            )
        return state_getter()

    raise ValueError("video_render_backend must be one of: warp, mujoco_gl.")


def resolve_video_benchmark_sync(cfg: DictConfig, *, take_video: bool) -> bool:
    return bool(take_video and cfg.get("video_benchmark_sync", False))


def resolve_video_benchmark_stage_sync(cfg: DictConfig, *, take_video: bool) -> bool:
    return bool(
        resolve_video_benchmark_sync(cfg, take_video=take_video)
        and cfg.get("video_benchmark_stage_sync", False)
    )


def resolve_video_benchmark_warmup_frames(
    cfg: DictConfig,
    *,
    take_video: bool,
) -> int:
    if not resolve_video_benchmark_sync(cfg, take_video=take_video):
        return 0
    return max(0, int(cfg.get("video_benchmark_warmup_frames", 5) or 0))


def open_streaming_video_writer(
    env: Any,
    log_dir: Path,
    video_length: int,
    step_dt: float,
    *,
    backend: str = "imageio",
    codec: str = "h264",
    extra_width: int = 0,
    output_dir: Path | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    fps: float | None = None,
    filename_prefix: str = "play",
    remux_timeout_seconds: float | None = None,
) -> tuple[Any, Path]:
    env_raw = getattr(env, "unwrapped", env)
    try:
        if backend == "imageio":
            import imageio.v2 as imageio
        else:
            imageio = None
    except ImportError as exc:
        raise RuntimeError(
            "Streaming video capture requires imageio. "
            "Install dependencies (e.g. `gymnasium[other]`)."
        ) from exc

    video_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else log_dir / "videos" / "play"
    )
    video_dir.mkdir(parents=True, exist_ok=True)
    safe_prefix = "".join(
        ch if ch.isalnum() or ch in {"_", "-"} else "_"
        for ch in str(filename_prefix or "play")
    ).strip("_")
    if not safe_prefix:
        safe_prefix = "play"
    video_path = video_dir / f"{safe_prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

    fps = (
        float(fps)
        if fps is not None
        else resolve_render_output_fps(env, env_raw=env_raw, step_dt=step_dt)
    )
    frame_width = int(output_width) if output_width is not None else None
    frame_height = int(output_height) if output_height is not None else None

    if backend == "pynvcodec":
        from instantmimic.rendering.video_cuda import PyNvCodecStreamingWriter

        resolution = getattr(getattr(env_raw, "cfg", None), "viewer", None)
        if resolution is None:
            raise RuntimeError(
                "PyNvVideoCodec backend requires env.cfg.viewer.resolution."
            )
        width, height = resolution.resolution
        frame_width = int(width) if frame_width is None else frame_width
        frame_height = int(height) if frame_height is None else frame_height
        writer_kwargs = {
            "output_path": video_path,
            "width": frame_width + max(0, int(extra_width)),
            "height": frame_height,
            "fps": fps,
            "codec": codec,
        }
        if remux_timeout_seconds is not None:
            writer_kwargs["remux_timeout_seconds"] = float(remux_timeout_seconds)
        writer = PyNvCodecStreamingWriter(**writer_kwargs)
        _LOGGER.info(
            "Recording video with PyNvVideoCodec: path=%s fps=%.3f length=%d codec=%s resolution=%dx%d",
            video_path.as_posix(),
            fps,
            video_length,
            codec,
            frame_width + max(0, int(extra_width)),
            frame_height,
        )
    else:
        assert imageio is not None
        writer = imageio.get_writer(
            video_path.as_posix(),
            fps=fps,
        )
        _LOGGER.info(
            "Recording video with streaming encoder: path=%s fps=%.3f length=%d macro_block_size=1",
            video_path.as_posix(),
            fps,
            video_length,
        )
    return writer, video_path


def _pynvcodec_is_available() -> bool:
    try:
        import PyNvVideoCodec  # noqa: F401
    except ImportError:
        return False
    return True


def _mujoco_gl_playback_is_available() -> bool:
    try:
        from instantmimic.rendering.mujoco_gl_playback import (
            mujoco_gl_playback_is_available,
        )
    except ImportError:
        return False
    return bool(mujoco_gl_playback_is_available())


def _mujoco_warp_playback_is_available() -> bool:
    try:
        from instantmimic.rendering.mujoco_warp_playback import (
            mujoco_warp_playback_is_available,
        )

    except ImportError:
        return False
    return bool(mujoco_warp_playback_is_available())


def resolve_video_encoder_backend(
    cfg: DictConfig,
    *,
    take_video: bool,
    video_render_backend: str = "mujoco_gl",
) -> str:
    return select_video_encoder_backend(
        backend=str(cfg.get("video_encoder_backend", "auto")),
        codec=str(cfg.get("video_codec", "h264")),
        take_video=take_video,
        video_render_backend=video_render_backend,
    )


def select_video_encoder_backend(
    *,
    backend: str,
    codec: str,
    take_video: bool,
    video_render_backend: str,
    output_width: int | None = None,
    output_height: int | None = None,
) -> str:
    backend = str(backend).strip().lower()
    if backend not in {"auto", "imageio", "pynvcodec"}:
        raise ValueError(
            "video_encoder_backend must be one of: auto, imageio, pynvcodec."
        )
    if not take_video or backend == "imageio":
        return "imageio"

    codec = str(codec).strip().lower()
    reasons: list[str] = []
    if codec != "h264":
        reasons.append("video_codec must be h264")
    if not _pynvcodec_is_available():
        reasons.append("PyNvVideoCodec is not installed")
    if video_render_backend == "warp":
        if not _mujoco_warp_playback_is_available():
            reasons.append("warp playback backend is unavailable")
    elif video_render_backend == "mujoco_gl":
        if not _mujoco_gl_playback_is_available():
            reasons.append("mujoco_gl playback backend is unavailable")
    else:
        reasons.append("mujoco_gl playback backend is unavailable")

    if not reasons and output_width is not None and output_height is not None:
        from instantmimic.rendering.video_cuda import validate_nvenc_resolution

        try:
            validate_nvenc_resolution(
                width=output_width, height=output_height, codec=codec
            )
        except ValueError as exc:
            if backend == "pynvcodec":
                raise RuntimeError(str(exc)) from exc
            reasons.append(str(exc))

    if backend == "auto":
        if reasons and output_width is not None:
            _LOGGER.warning(
                "NVENC auto selection is using imageio: %s", "; ".join(reasons)
            )
        return "pynvcodec" if len(reasons) == 0 else "imageio"

    if len(reasons) > 0:
        detail = "; ".join(reasons)
        raise RuntimeError(
            "PyNvVideoCodec backend is unavailable: "
            f"{detail}. Disable overlays or use video_encoder_backend=imageio. "
            "Forced pynvcodec requires CUDA video frames."
        )
    return "pynvcodec"


def resolve_video_output_dir(cfg: DictConfig, *, log_dir: Path) -> Path:
    raw_output_dir = str(cfg.get("video_output_dir", "") or "").strip()
    if not raw_output_dir:
        return (Path(log_dir) / "videos" / "play").resolve()

    output_dir = Path(raw_output_dir).expanduser()
    if output_dir.is_absolute():
        return output_dir.resolve()
    return resolve_project_path(output_dir)


def resolve_video_render_backend(
    cfg: DictConfig,
    *,
    take_video: bool,
    overlay_reward_terms: bool = False,
) -> str:
    backend = str(cfg.get("video_render_backend", "auto")).strip().lower()
    if backend not in {"auto", "warp", "mujoco_gl"}:
        raise ValueError("video_render_backend must be one of: auto, warp, mujoco_gl.")
    tile_requested = bool(take_video and int(cfg.get("video_tile_envs", 1) or 1) > 1)
    camera_views_requested = bool(
        take_video and cfg.get("playback_camera_views", None) is not None
    )
    if tile_requested and camera_views_requested:
        raise RuntimeError(
            "playback_camera_views cannot be combined with video_tile_envs > 1."
        )
    if not take_video:
        return "warp" if backend in {"auto", "warp"} else "mujoco_gl"

    warp_available = _mujoco_warp_playback_is_available()
    gl_available = _mujoco_gl_playback_is_available()

    if backend == "warp":
        if camera_views_requested:
            raise RuntimeError(
                "playback_camera_views requires video_render_backend=mujoco_gl."
            )
        if overlay_reward_terms:
            raise RuntimeError(
                "video_render_backend=warp does not support reward overlays yet. "
                "Use video_render_backend=mujoco_gl or disable video overlays."
            )
        if not warp_available:
            raise RuntimeError(
                "warp render backend is unavailable: playback helper is not installed "
                "or could not be imported."
            )
        return "warp"

    if backend == "mujoco_gl":
        if tile_requested:
            raise RuntimeError(
                "video_tile_envs > 1 is supported only by video_render_backend=warp."
            )
        if not gl_available:
            raise RuntimeError(
                "mujoco_gl render backend is unavailable: playback helper is not installed "
                "or could not be imported."
            )
        return "mujoco_gl"

    if tile_requested:
        if overlay_reward_terms:
            raise RuntimeError(
                "video_tile_envs > 1 is supported only by the warp renderer and "
                "cannot be combined with reward overlays yet."
            )
        if not warp_available:
            raise RuntimeError(
                "video_tile_envs > 1 requires the warp render backend, but warp "
                "playback is unavailable."
            )
        return "warp"

    if camera_views_requested:
        if not gl_available:
            raise RuntimeError(
                "playback_camera_views requires the mujoco_gl render backend, but "
                "mujoco_gl playback is unavailable."
            )
        return "mujoco_gl"

    if overlay_reward_terms and gl_available:
        return "mujoco_gl"
    if gl_available:
        return "mujoco_gl"
    if warp_available:
        return "warp"
    raise RuntimeError(
        "No playback render backend is available: neither warp nor mujoco_gl "
        "could be imported."
    )


def clone_mjv_camera(camera: Any) -> mujoco.MjvCamera | None:
    if camera is None:
        return None
    cloned = mujoco.MjvCamera()
    cloned.type = int(getattr(camera, "type", cloned.type))
    cloned.fixedcamid = int(getattr(camera, "fixedcamid", cloned.fixedcamid))
    cloned.trackbodyid = int(getattr(camera, "trackbodyid", cloned.trackbodyid))
    cloned.lookat[:] = np.asarray(
        getattr(camera, "lookat", cloned.lookat),
        dtype=np.float64,
    )
    cloned.distance = float(getattr(camera, "distance", cloned.distance))
    cloned.azimuth = float(getattr(camera, "azimuth", cloned.azimuth))
    cloned.elevation = float(getattr(camera, "elevation", cloned.elevation))
    return cloned


def build_video_playback_camera(base_camera: Any, cfg: DictConfig) -> Any:
    return build_video_camera(
        base_camera,
        azimuth=cfg.get("playback_camera_azimuth_deg", None),
        elevation=cfg.get("playback_camera_elevation_deg", None),
        distance=cfg.get("playback_camera_distance", None),
    )


def build_video_camera(
    base_camera: Any,
    *,
    azimuth: float | None,
    elevation: float | None,
    distance: float | None,
) -> Any:
    if azimuth is None and elevation is None and distance is None:
        return base_camera

    camera = clone_mjv_camera(base_camera)
    if camera is None:
        camera = mujoco.MjvCamera()
    if azimuth is not None:
        camera.azimuth = float(azimuth)
    if elevation is not None:
        camera.elevation = float(elevation)
    if distance is not None:
        camera.distance = float(distance)
    return camera


def _positive_camera_view_resolution(
    raw_resolution: Any,
    *,
    default_width: int,
    default_height: int,
) -> tuple[int, int]:
    if raw_resolution is None:
        return int(default_width), int(default_height)
    try:
        values = list(raw_resolution)
    except TypeError as exc:
        raise ValueError(
            "playback_camera_views.resolution must be [width, height]."
        ) from exc
    if len(values) != 2:
        raise ValueError("playback_camera_views.resolution must be [width, height].")
    width, height = (int(values[0]), int(values[1]))
    if width <= 0 or height <= 0:
        raise ValueError("playback_camera_views.resolution values must be positive.")
    return width, height


def _build_configured_playback_camera(
    base_camera: Any,
    raw_camera: Any,
    *,
    index: int,
) -> mujoco.MjvCamera:
    if not isinstance(raw_camera, Mapping) and not isinstance(raw_camera, DictConfig):
        raise ValueError(f"playback_camera_views.cameras[{index}] must be a mapping.")

    camera = clone_mjv_camera(base_camera)
    if camera is None:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)

    lookat = raw_camera.get("lookat", None)
    if lookat is not None:
        try:
            lookat_values = np.asarray(list(lookat), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"playback_camera_views.cameras[{index}].lookat must contain "
                "three finite numbers."
            ) from exc
        if lookat_values.shape != (3,) or not np.all(np.isfinite(lookat_values)):
            raise ValueError(
                f"playback_camera_views.cameras[{index}].lookat must contain "
                "three finite numbers."
            )
        camera.lookat[:] = lookat_values

    scalar_fields = (
        ("distance", "distance"),
        ("azimuth_deg", "azimuth"),
        ("elevation_deg", "elevation"),
    )
    for config_name, camera_name in scalar_fields:
        raw_value = raw_camera.get(config_name, None)
        if raw_value is None:
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                f"playback_camera_views.cameras[{index}].{config_name} must be finite."
            )
        if config_name == "distance" and value <= 0.0:
            raise ValueError(
                f"playback_camera_views.cameras[{index}].distance must be positive."
            )
        setattr(camera, camera_name, value)

    # Explicit view parameters describe an absolute free camera, independent of
    # an environment's default tracking camera.
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    return camera


def resolve_playback_camera_view_layout(
    base_camera: Any,
    cfg: DictConfig,
    *,
    default_width: int,
    default_height: int,
) -> PlaybackCameraViewLayout:
    """Resolve optional config-driven camera composition for playback video."""

    raw_views = cfg.get("playback_camera_views", None)
    if raw_views is None:
        return PlaybackCameraViewLayout(
            cameras=(build_video_playback_camera(base_camera, cfg),),
            layout="horizontal",
            view_width=int(default_width),
            view_height=int(default_height),
        )
    if not isinstance(raw_views, Mapping) and not isinstance(raw_views, DictConfig):
        raise ValueError("playback_camera_views must be a mapping.")

    layout = str(raw_views.get("layout", "horizontal")).strip().lower()
    if layout not in {"horizontal", "vertical"}:
        raise ValueError(
            "playback_camera_views.layout must be 'horizontal' or 'vertical'."
        )
    view_width, view_height = _positive_camera_view_resolution(
        raw_views.get("resolution", None),
        default_width=default_width,
        default_height=default_height,
    )
    raw_cameras = raw_views.get("cameras", None)
    if raw_cameras is None:
        raise ValueError("playback_camera_views.cameras must be a non-empty list.")
    try:
        camera_configs = list(raw_cameras)
    except TypeError as exc:
        raise ValueError(
            "playback_camera_views.cameras must be a non-empty list."
        ) from exc
    if not camera_configs:
        raise ValueError("playback_camera_views.cameras must be a non-empty list.")

    cameras = tuple(
        _build_configured_playback_camera(base_camera, item, index=index)
        for index, item in enumerate(camera_configs)
    )
    return PlaybackCameraViewLayout(
        cameras=cameras,
        layout=layout,
        view_width=view_width,
        view_height=view_height,
        track_bodies=tuple(
            tuple(item.get("track_bodies") or ()) for item in camera_configs
        ),
        lookat_offsets=tuple(
            tuple(item.get("lookat_offset") or (0.0, 0.0, 0.0))
            for item in camera_configs
        ),
    )
