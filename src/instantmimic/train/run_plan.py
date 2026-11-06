"""Setup-time training run decisions used by the PPO runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from instantmimic.config.experiment_config import BestCheckpointConfig, CheckpointConfig
from instantmimic.config.runtime_specs import ExecutionContext, RuntimeExecutionSpec
from instantmimic.config.schema import SemanticConfig
from instantmimic.train.benchmark import TrainingBenchmarkPlan
from instantmimic.utils.project_paths import resolve_project_log_root


@dataclass(frozen=True)
class ExperimentContext:
    config_name: str
    setting_name: str
    timestamp_name: str
    base_experiment_name: str
    experiment_name: str
    log_root_path: Path
    log_dir: Path
    checkpoints_dir: Path
    group: str
    wandb_name: str
    logging_backend: str


@dataclass(frozen=True)
class TrainVideoPlan:
    requested: bool
    interval_iterations: int
    length: int

    @classmethod
    def from_semantic_config(cls, cfg: SemanticConfig) -> TrainVideoPlan:
        recording = cfg.trainer.recording
        periodic = recording.periodic if recording is not None else None
        if periodic is None:
            return cls(requested=False, interval_iterations=0, length=0)
        return cls(
            requested=True,
            interval_iterations=periodic.interval,
            length=periodic.length,
        )


_BEST_CHECKPOINT_METRIC_DIRECTIONS: dict[str, Literal["maximize", "minimize"]] = {
    "episode_reward_mean": "maximize",
    "reward_mean": "maximize",
    "success_rate": "maximize",
    "episode_length_mean": "maximize",
    "score": "minimize",
    "fixed_horizon_reward_mean": "maximize",
}


@dataclass(frozen=True)
class BestCheckpointPlan:
    configured_source: Literal["auto", "training", "benchmark"]
    source: Literal["training", "benchmark"]
    benchmark_protocol: Literal["rollout", "clip_failure"] | None
    metric: str
    direction: Literal["maximize", "minimize"]

    @classmethod
    def from_semantic_config(
        cls,
        cfg: SemanticConfig,
        *,
        benchmark: TrainingBenchmarkPlan,
    ) -> BestCheckpointPlan:
        checkpoint = cfg.experiment.checkpoint or CheckpointConfig()
        best = checkpoint.best or BestCheckpointConfig()
        configured_source = best.source

        if configured_source == "auto":
            if benchmark.rollout is not None:
                source = "benchmark"
                benchmark_protocol = "rollout"
                eval_plan = benchmark.rollout_or_disabled
                metric = eval_plan.best_metric or "reward_mean"
            elif benchmark.clip_failure is not None:
                source = "benchmark"
                benchmark_protocol = "clip_failure"
                metric = "fixed_horizon_reward_mean"
            else:
                source = "training"
                benchmark_protocol = None
                metric = "episode_reward_mean"
        elif configured_source == "training":
            source = "training"
            benchmark_protocol = None
            if best.metric is None:
                raise ValueError("experiment.checkpoint.best.metric is required")
            metric = best.metric
        else:
            if benchmark.rollout is None and benchmark.clip_failure is None:
                raise ValueError(
                    "experiment.checkpoint.best.source=benchmark requires "
                    "trainer.benchmark or trainer.clip_failure_benchmark"
                )
            source = "benchmark"
            benchmark_protocol = (
                "rollout" if benchmark.rollout is not None else "clip_failure"
            )
            if best.metric is None:
                raise ValueError("experiment.checkpoint.best.metric is required")
            metric = best.metric

        eval_plan = benchmark.rollout_or_disabled
        if (
            benchmark_protocol == "rollout"
            and metric == eval_plan.best_metric
            and eval_plan.best_metric_direction is not None
        ):
            direction = eval_plan.best_metric_direction
        else:
            try:
                direction = _BEST_CHECKPOINT_METRIC_DIRECTIONS[metric]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported best checkpoint metric {metric!r}."
                ) from exc
        return cls(
            configured_source=configured_source,
            source=source,
            benchmark_protocol=benchmark_protocol,
            metric=metric,
            direction=direction,
        )

    @property
    def uses_training(self) -> bool:
        return self.source == "training"

    @property
    def uses_rollout_benchmark(self) -> bool:
        return self.source == "benchmark" and self.benchmark_protocol == "rollout"

    @property
    def uses_clip_failure_benchmark(self) -> bool:
        return self.source == "benchmark" and self.benchmark_protocol == "clip_failure"


@dataclass(frozen=True)
class TrainingSchedule:
    max_iterations: int
    write_interval: int
    checkpoint_disabled: bool
    checkpoint_interval: int
    episode_metrics_interval: int
    best_checkpoint_interval: int

    @classmethod
    def from_semantic_config(
        cls,
        cfg: SemanticConfig,
        *,
        episode_threshold_enabled: bool,
    ) -> TrainingSchedule:
        checkpoint = cfg.experiment.checkpoint or CheckpointConfig()
        if cfg.trainer.max_iterations is None:
            raise ValueError("trainer.max_iterations is required")
        periodic = checkpoint.periodic
        best = checkpoint.best
        return cls(
            max_iterations=cfg.trainer.max_iterations,
            write_interval=cfg.experiment.write_interval,
            checkpoint_disabled=checkpoint.disabled,
            checkpoint_interval=(
                0 if checkpoint.disabled or periodic is None else periodic.interval or 0
            ),
            episode_metrics_interval=(
                1
                if episode_threshold_enabled
                else cfg.experiment.episode_metrics_interval
            ),
            best_checkpoint_interval=(
                0 if checkpoint.disabled or best is None else best.interval or 0
            ),
        )


@dataclass(frozen=True)
class TrainingStoppingPlan:
    type: Literal["reward", "episode_length", "benchmark", "time_budget"] | None
    threshold: float | None
    warmup_iterations: int
    timeout_seconds: float | None
    fail_on_miss: bool

    @classmethod
    def from_semantic_config(
        cls,
        cfg: SemanticConfig,
        benchmark: TrainingBenchmarkPlan,
    ) -> TrainingStoppingPlan:
        stopping = cfg.trainer.stopping
        if stopping is None:
            return cls(
                type=None,
                threshold=None,
                warmup_iterations=1,
                timeout_seconds=None,
                fail_on_miss=False,
            )
        stopping_type = stopping.type
        if stopping_type == "benchmark" and benchmark.rollout is None:
            raise ValueError(
                "trainer.stopping.type=benchmark requires trainer.benchmark"
            )
        if stopping_type in {"reward", "episode_length"} and stopping.threshold is None:
            raise ValueError("trainer.stopping.threshold is required")
        threshold = (
            float(stopping.threshold)
            if stopping_type in {"reward", "episode_length"}
            else None
        )
        return cls(
            type=stopping_type,
            threshold=threshold,
            warmup_iterations=stopping.warmup_iterations,
            timeout_seconds=(
                float(stopping.timeout_seconds)
                if stopping.timeout_seconds is not None
                else None
            ),
            fail_on_miss=stopping.fail_on_miss,
        )

    @property
    def episode_metrics_required(self) -> bool:
        return self.type in {"reward", "episode_length"}


@dataclass(frozen=True)
class TrainingOrchestrationPlan:
    schedule: TrainingSchedule
    benchmark: TrainingBenchmarkPlan
    best_checkpoint: BestCheckpointPlan
    stopping: TrainingStoppingPlan
    runtime: RuntimeExecutionSpec

    @classmethod
    def from_semantic_config(
        cls,
        cfg: SemanticConfig,
    ) -> TrainingOrchestrationPlan:
        benchmark = TrainingBenchmarkPlan.from_semantic_trainer(
            cfg.trainer,
            env_use_cuda_graph=bool(cfg.env.get("use_cuda_graph", False)),
            compile=bool(cfg.runtime.get("compile", False)),
        )
        stopping = TrainingStoppingPlan.from_semantic_config(cfg, benchmark)
        return cls(
            schedule=TrainingSchedule.from_semantic_config(
                cfg,
                episode_threshold_enabled=stopping.episode_metrics_required,
            ),
            benchmark=benchmark,
            best_checkpoint=BestCheckpointPlan.from_semantic_config(
                cfg,
                benchmark=benchmark,
            ),
            stopping=stopping,
            runtime=RuntimeExecutionSpec.from_semantic_config(cfg),
        )


def create_semantic_experiment_context(
    cfg: SemanticConfig,
    *,
    device: str,
    config_name: str,
    num_envs: int,
    now: datetime | None = None,
    log_root_path: Path | None = None,
    seed_factory: Callable[[], int] | None = None,
) -> tuple[ExperimentContext, ExecutionContext]:
    """Create run metadata for v1 without mutating the semantic config."""

    current_time = now or datetime.now()
    timestamp_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    explicit_name = (cfg.experiment.name or "").strip()
    setting_name = explicit_name or Path(config_name).stem
    base_experiment_name = f"{timestamp_name}_ppo_custom"
    if setting_name:
        base_experiment_name += f"_{setting_name}"
    if cfg.experiment.directory is None:
        raise ValueError("experiment.directory is required")
    resolved_log_root = log_root_path or resolve_project_log_root(
        "ppo", cfg.experiment.directory
    )
    experiment_name, log_dir = create_unique_experiment_log_dir(
        resolved_log_root,
        base_experiment_name,
    )
    checkpoints_dir = log_dir / "checkpoints"
    checkpoint = cfg.experiment.checkpoint
    checkpoint_disabled = checkpoint.disabled if checkpoint is not None else False
    if not checkpoint_disabled:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
    execution = ExecutionContext.from_semantic_config(
        cfg,
        device=device,
        run_name=experiment_name,
        log_dir=log_dir,
        timestamp=current_time.isoformat(),
        num_envs=num_envs,
        seed_factory=seed_factory,
    )
    group = (
        cfg.experiment.group
        if cfg.experiment.group is not None
        else cfg.experiment.directory
    )
    return (
        ExperimentContext(
            config_name=config_name,
            setting_name=setting_name,
            timestamp_name=timestamp_name,
            base_experiment_name=base_experiment_name,
            experiment_name=experiment_name,
            log_root_path=resolved_log_root,
            log_dir=log_dir,
            checkpoints_dir=checkpoints_dir,
            group=group,
            wandb_name=resolve_wandb_run_name(
                timestamp_name=timestamp_name,
                base_experiment_name=base_experiment_name,
                experiment_name=experiment_name,
            ),
            logging_backend=execution.logger_type,
        ),
        execution,
    )


def create_unique_experiment_log_dir(
    log_root_path: Path,
    base_experiment_name: str,
    *,
    max_attempts: int = 10000,
) -> tuple[str, Path]:
    if not base_experiment_name:
        raise ValueError("base_experiment_name must be non-empty")

    for suffix in range(max_attempts):
        experiment_name = (
            base_experiment_name
            if suffix == 0
            else f"{base_experiment_name}_{suffix:04d}"
        )
        log_dir = log_root_path / experiment_name
        try:
            log_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return experiment_name, log_dir

    raise FileExistsError(
        f"Could not create a unique experiment log directory under {log_root_path} "
        f"for base name {base_experiment_name!r} after {max_attempts} attempts."
    )


def resolve_wandb_run_name(
    *,
    timestamp_name: str,
    base_experiment_name: str,
    experiment_name: str,
) -> str:
    if experiment_name == base_experiment_name:
        return timestamp_name
    if experiment_name.startswith(f"{base_experiment_name}_"):
        return f"{timestamp_name}{experiment_name[len(base_experiment_name) :]}"
    return timestamp_name
