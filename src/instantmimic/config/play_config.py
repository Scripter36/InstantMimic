"""Playback controls and reusable recording settings."""

from typing import Literal

from instantmimic.config.policy_config import Number
from instantmimic.config.settings import Settings


class PlayActionConfig(Settings):
    mode: Literal["mean", "sample"]


class PlayTrackingConfig(Settings):
    mode: Literal["reference", "free_run"] = "reference"
    metric_frame: Literal["root", "world"] = "root"
    include_root_in_ape: bool = False
    require_full_horizon: bool | None = None
    deterministic_reset: bool = False
    motion_index: int | None = None
    start_frame: int | None = None
    termination: Literal["inherit", "disabled"] | None = None


class PlayViewerConfig(Settings):
    backend: Literal["mujoco", "viser"] = "mujoco"
    resolution: tuple[int, int] | None = None
    port: int | None = None
    reset_on_done: bool = False


class WarpVideoConfig(Settings):
    use_textures: bool | None = None
    floor_shader_mode: Literal["texture"] | None = None
    render_block_dim: int | None = None
    lbs_mesh_mode: Literal["off", "skin", "muscle"] | None = None
    lbs_mesh_bundle: str | None = None
    lbs_mesh_shading_mode: Literal["flat", "smooth"] | None = None
    lbs_activation_mapping: str | None = None


class TrainingWarpVideoConfig(WarpVideoConfig):
    activation_log_stats: bool | None = None


class PlayVideoConfig(Settings):
    enabled: bool = False
    length: int | Literal["episode"] | None = None
    render_backend: Literal["auto", "warp", "mujoco_gl"] | None = None
    encoder_backend: Literal["auto", "imageio", "pynvcodec"] | None = None
    codec: Literal["h264"] | None = None
    tile_envs: int | None = None
    tile_cols: int | None = None
    overlay_rewards: bool | None = None
    overlay_history_seconds: Number | None = None
    overlay_panel_width: int | None = None
    remux_timeout_seconds: Number | None = None
    warp: WarpVideoConfig | None = None


class CameraConfig(Settings):
    azimuth_deg: Number | None = None
    elevation_deg: Number | None = None
    distance: Number | None = None


class CameraViewConfig(CameraConfig):
    lookat: tuple[Number, Number, Number] | None = None
    track_bodies: tuple[str, ...] | None = None
    lookat_offset: tuple[Number, Number, Number] | None = None


class CameraViewsConfig(Settings):
    layout: Literal["horizontal", "vertical"] | None = None
    resolution: tuple[int, int] | None = None
    cameras: tuple[CameraViewConfig, ...]


class PlayCameraConfig(CameraConfig):
    lookat_ema_alpha: Number | None = None
    views: CameraViewsConfig | None = None


class RecordingBenchmarkConfig(Settings):
    warmup_frames: int = 5


class PlayBenchmarkConfig(RecordingBenchmarkConfig):
    stage_sync: bool = False


class PlayOutputConfig(Settings):
    directory: str | None = None
    emit_result_payload: bool = False


class PlayConfig(Settings):
    checkpoint: str | None = None
    seed: int | None = None
    num_envs: int | None = None
    steps: int | None = None
    time_limit_policy: Literal["extend", "respect"] | None = None
    done_policy: Literal["reset_continue", "stop"] = "reset_continue"
    use_cuda_graph: bool = True
    allow_eager_fallback: bool = False
    action: PlayActionConfig | None = None
    viewer: PlayViewerConfig | None = None
    video: PlayVideoConfig | None = None
    tracking: PlayTrackingConfig | None = None
    camera: PlayCameraConfig | None = None
    benchmark: PlayBenchmarkConfig | None = None
    output: PlayOutputConfig | None = None
