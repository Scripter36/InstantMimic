"""Training schedules, evaluation protocols, and recording declarations."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BeforeValidator

from instantmimic.config.play_config import (
    CameraConfig,
    RecordingBenchmarkConfig,
    TrainingWarpVideoConfig,
)
from instantmimic.config.policy_config import Number
from instantmimic.config.settings import Settings
from instantmimic.config.values import FrozenMapping, freeze_value


def _freeze_extension(value: object) -> FrozenMapping:
    if not isinstance(value, Mapping):
        raise ValueError("environment overrides must be a mapping")
    frozen = freeze_value(dict(value))
    if not isinstance(frozen, FrozenMapping):
        raise TypeError("environment overrides must produce a frozen mapping")
    return frozen


type EnvironmentOverrides = Annotated[FrozenMapping, BeforeValidator(_freeze_extension)]


class EvaluationLoggingConfig(Settings):
    statistics: (
        tuple[Literal["mean", "rate", "last", "count", "flag", "step"], ...] | None
    ) = None
    exclude: tuple[str, ...] | None = None


class MotionGroupConfig(Settings):
    name: str
    clip_index: int


class EvaluationObjectiveConfig(Settings):
    metric: str
    direction: Literal["maximize", "minimize"]


class EvaluationTerminationConfig(Settings):
    type: Literal["inherit", "disabled", "phc_tracking"]
    distance: Number | None = None


class EvaluationThresholdsConfig(Settings):
    ape_mean: Number | None = None
    global_mpjpe_mean: Number | None = None
    ee_mean: Number | None = None
    success_rate: Number | None = None
    consecutive_passes: int | None = None


class BenchmarkConfig(Settings):
    interval: int | None = None
    interval_seconds: Number | None = None
    evaluate_at_start: bool | None = None
    steps: int | None = None
    num_envs: int | None = None
    metric_frame: Literal["world", "translation", "heading", "root"] | None = None
    include_root_in_ape: bool | None = None
    deterministic_reset: bool | None = None
    require_full_horizon: bool | None = None
    episode_semantics: Literal["full_horizon", "terminal_success"] | None = None
    termination: EvaluationTerminationConfig | None = None
    best_metric: (
        Literal["score", "success_rate", "reward_mean", "episode_length_mean"] | None
    ) = None
    best_metric_tie_breakers: (
        tuple[
            Literal["score", "success_rate", "reward_mean", "episode_length_mean"], ...
        ]
        | None
    ) = None
    use_best_candidate: bool | None = None
    thresholds: EvaluationThresholdsConfig | None = None
    execution: Literal["inline", "overlap"] | None = None
    save_history: bool | None = None
    history_filename: str | None = None
    metrics: tuple[str, ...] | None = None
    logging: EvaluationLoggingConfig | None = None
    motion_groups: tuple[MotionGroupConfig, ...] | None = None
    env_overrides: EnvironmentOverrides | None = None
    objective: EvaluationObjectiveConfig | None = None
    cuda_graph: bool | None = None


class ClipFailureBenchmarkConfig(Settings):
    interval: int | None = None
    mode: Literal["anchor", "full_motion"] | None = None
    anchor_stride: int | None = None
    window_steps: int | None = None
    batch_size: int | None = None
    execution: Literal["auto", "inline", "subprocess", "shared_motion"] | None = None
    preserve_training_graphs: bool | None = None
    clone_shared_motion_tensors: bool | None = None
    clip_indices: tuple[int, ...] | None = None
    save_history: bool | None = None
    history_filename: str | None = None
    timeout_seconds: Number | None = None


class InitializationConfig(Settings):
    type: Literal["resume", "policy_init"]
    checkpoint: str


class StoppingConfig(Settings):
    type: Literal["reward", "episode_length", "benchmark", "time_budget"]
    warmup_iterations: int = 1
    timeout_seconds: Number | None = None
    fail_on_miss: bool = False
    threshold: Number | None = None


class PeriodicRecordingConfig(Settings):
    interval: int
    length: int


class FullRunRecordingConfig(Settings):
    encoder_backend: Literal["auto", "imageio", "pynvcodec"] | None = None
    codec: Literal["h264"] | None = None
    tile_envs: int | None = None
    tile_cols: int | None = None
    camera: CameraConfig | None = None
    benchmark: RecordingBenchmarkConfig | None = None
    warp: TrainingWarpVideoConfig | None = None


class RecordingConfig(Settings):
    periodic: PeriodicRecordingConfig | None = None
    full_run: FullRunRecordingConfig | None = None


class PostTrainingPlaybackConfig(Settings):
    video: bool | None = None
    video_length: int | None = None
    use_cuda_graph: bool | None = None
    subprocess_timeout_seconds: Number | None = None


class TrainerConfig(Settings):
    max_iterations: int | None = None
    training_stage: Literal["full", "musclenn_only"] | None = None
    benchmark: BenchmarkConfig | None = None
    clip_failure_benchmark: ClipFailureBenchmarkConfig | None = None
    initialization: InitializationConfig | None = None
    stopping: StoppingConfig | None = None
    recording: RecordingConfig | None = None
    post_training_playback: PostTrainingPlaybackConfig | None = None
