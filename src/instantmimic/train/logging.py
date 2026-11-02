from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard import SummaryWriter

from instantmimic.config.schema import SemanticConfig
from instantmimic.train.metrics import (
    _ACTIVE_TRAIN_FPS_METRIC,
    _ACTIVE_TRAIN_TIME_METRIC,
    _CHECKPOINT_TIME_METRIC,
    _TRAIN_FRAME_METRIC,
)
from instantmimic.utils.experiment_logging import (
    build_mlflow_init_kwargs,
    build_wandb_init_kwargs,
)

_LOGGER = logging.getLogger(__name__)


def _consume_optional_done_reason_stats(env_raw: Any) -> dict[str, float]:
    consume_stats = getattr(env_raw, "consume_done_reason_stats", None)
    if not callable(consume_stats):
        return {}
    raw_stats = consume_stats()
    if not isinstance(raw_stats, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in raw_stats.items():
        if isinstance(value, (bool, int | float)):
            result[str(key)] = float(value)
    return result


def _log_optional_done_reason_stats(
    metrics_logger: Any,
    env_raw: Any,
    iteration_index: int,
) -> dict[str, float]:
    stats = _consume_optional_done_reason_stats(env_raw)
    if stats:
        metrics_logger.log_scalars(stats, iteration_index, axis="iteration")
    return stats


def _log_optional_diagnostic_stats(
    metrics_logger: Any,
    env_raw: Any,
    iteration_index: int,
) -> dict[str, float]:
    consume_stats = getattr(env_raw, "consume_diagnostic_stats", None)
    if not callable(consume_stats):
        return {}
    stats = {str(key): float(value) for key, value in consume_stats().items()}
    if stats:
        metrics_logger.log_scalars(stats, iteration_index, axis="iteration")
    return stats


_MLFLOW_METRIC_NAME_INVALID_RE = re.compile(r"[^A-Za-z0-9_. \-:/]+")
_MLFLOW_METRIC_NAME_WHITESPACE_RE = re.compile(r"\s+")


def _sanitize_mlflow_metric_name(tag: str) -> str:
    sanitized = _MLFLOW_METRIC_NAME_INVALID_RE.sub(" ", tag)
    sanitized = _MLFLOW_METRIC_NAME_WHITESPACE_RE.sub(" ", sanitized).strip()
    return sanitized if sanitized else "metric"


def _build_mlflow_metric_payload(scalars: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for tag, value in scalars.items():
        base_name = _sanitize_mlflow_metric_name(tag)
        name = base_name
        suffix = 1
        while name in payload:
            suffix += 1
            name = f"{base_name} {suffix}"
        payload[name] = float(value)
    return payload


def _build_wandb_eval_workspace(
    *,
    run: Any,
    grouped_metrics: Mapping[str, Mapping[str, str]],
) -> Any:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws
    from wandb_workspaces import expr

    panels = []
    for title, metrics_by_group in grouped_metrics.items():
        metric_names = list(metrics_by_group.values())
        panels.append(
            wr.LinePlot(
                title=str(title),
                x=ExperimentMetricsLogger._WANDB_TRAIN_ITERATION,
                y=metric_names,
                line_titles={
                    metric_name: group_name
                    for group_name, metric_name in metrics_by_group.items()
                },
            )
        )
    return ws.Workspace(
        name=f"{run.name} Eval",
        entity=str(run.entity),
        project=str(run.project),
        sections=[ws.Section(name="Eval", panels=panels, is_open=True)],
        runset_settings=ws.RunsetSettings(
            filters=expr.And(expr.Metric("ID") == str(run.id)),
            pinned_runs=[str(run.id)],
        ),
        auto_generate_panels=False,
    )


_MLFLOW_QUEUE_STOP = object()


class _AsyncMlflowMetricsLogger:
    def __init__(self, mlflow_module: Any, *, run_id: str | None) -> None:
        self._mlflow_module = mlflow_module
        self._run_id = run_id
        self._queue: queue.Queue[object] = queue.Queue()
        self._closed = False
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="mlflow-metrics-logger",
            daemon=True,
        )
        self._thread.start()

    def log_metrics(self, payload: dict[str, float], *, step: int) -> None:
        self._raise_if_failed()
        if self._closed:
            _LOGGER.warning("Dropping MLflow metrics after logger close.")
            return
        self._queue.put((dict(payload), int(step)))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_MLFLOW_QUEUE_STOP)
        self._queue.join()
        self._thread.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("MLflow metric logging failed") from self._error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _MLFLOW_QUEUE_STOP:
                    return
                payload, step = item
                try:
                    if self._run_id is not None:
                        self._mlflow_module.log_metrics(
                            payload,
                            step=step,
                            run_id=self._run_id,
                        )
                    else:
                        self._mlflow_module.log_metrics(payload, step=step)
                except Exception as exc:  # noqa: BLE001 - worker boundary logs remote logger failures.
                    if self._error is None:
                        self._error = exc
            finally:
                self._queue.task_done()


def _add_scalars_fast(
    writer: SummaryWriter, scalars: dict[str, float], global_step: int
) -> None:
    """Write multiple scalar tags in one event to reduce writer call overhead."""
    if not scalars:
        return
    summary = Summary(
        value=[
            Summary.Value(tag=tag, simple_value=float(value))
            for tag, value in scalars.items()
        ]
    )
    writer._get_file_writer().add_summary(summary, global_step)


class ExperimentMetricsLogger:
    _WANDB_TRAIN_ITERATION = "Trainer / Iteration"
    _WANDB_TRAIN_FRAME = _TRAIN_FRAME_METRIC
    _WANDB_ITERATION_METRIC_TAGS = (
        "Loss / Policy loss",
        "Loss / Value loss",
        "Loss / Entropy loss",
        "Loss / Log std bound loss",
        "Loss / Policy auxiliary loss",
        "Loss / MuscleNN loss",
        "Loss / MuscleNN target loss",
        "Loss / MuscleNN regularization loss",
        "Learning / Learning rate",
        "Policy / Standard deviation",
        "Policy / Actual standard deviation mean",
        "Policy / Actual standard deviation min",
        "Policy / Actual standard deviation max",
        "Policy / Raw log std clamp lower fraction",
        "Policy / Raw log std clamp upper fraction",
        "Policy / Raw log std clamp overshoot RMS",
        "Policy / Raw log std clamp overshoot max",
        "Timing / Physics (ms)",
        "Timing / PPO update (ms)",
        "Timing / MuscleNN update (ms)",
        "Reward / Total reward (mean)",
        "Episode / Total timesteps (mean)",
        "Eval / Pass streak",
        "Eval / Score",
        "Eval / Best metric value",
        "Timing / Eval runtime (ms)",
        _ACTIVE_TRAIN_FPS_METRIC,
        _ACTIVE_TRAIN_TIME_METRIC,
        _CHECKPOINT_TIME_METRIC,
        "Objective / Time to reward threshold (s)",
        "Objective / Time to episode length threshold (s)",
        "Objective / Time to benchmark threshold (s)",
        "Info / Frames",
        "Info / Updates",
        "Info / MuscleNN buffer size",
    )
    _WANDB_FRAME_METRIC_TAGS = (
        "Loss / Policy loss (frames)",
        "Loss / Value loss (frames)",
        "Loss / Entropy loss (frames)",
        "Loss / Log std bound loss (frames)",
        "Loss / Policy auxiliary loss (frames)",
        "Loss / MuscleNN loss (frames)",
        "Loss / MuscleNN target loss (frames)",
        "Loss / MuscleNN regularization loss (frames)",
        "Learning / Learning rate (frames)",
        "Timing / Physics (ms) (frames)",
        "Timing / PPO update (ms) (frames)",
        "Reward / Total reward (mean) (frames)",
        "Episode / Total timesteps (mean) (frames)",
    )

    def __init__(
        self,
        *,
        backend: str,
        cfg: SemanticConfig,
        experiment_name: str,
        log_dir: Path,
        writer_max_queue: int,
        writer_flush_secs: int,
        group: str | None = None,
        wandb_name: str | None = None,
        config_name: str | None = None,
        setting_name: str | None = None,
    ) -> None:
        self.backend = backend
        self._writer: SummaryWriter | None = None
        self._wandb_run: Any | None = None
        self._wandb_grouped_metrics: dict[str, dict[str, str]] = {}
        self._mlflow_module: Any | None = None
        self._mlflow_metrics_logger: _AsyncMlflowMetricsLogger | None = None
        self._mlflow_run_id: str | None = None
        self._wandb_defined_iteration_metrics = {self._WANDB_TRAIN_ITERATION}
        self._wandb_defined_frame_metrics = {self._WANDB_TRAIN_FRAME}

        if backend == "tensorboard":
            self._writer = SummaryWriter(
                log_dir=log_dir.as_posix(),
                max_queue=writer_max_queue,
                flush_secs=writer_flush_secs,
            )
            return

        if backend == "mlflow":
            try:
                import mlflow

                cfg_payload = cfg.to_dict()
                experiment_cfg = cfg.experiment["logger"]
                mlflow_kwargs = build_mlflow_init_kwargs(
                    experiment_cfg=experiment_cfg,
                    run_name=wandb_name if wandb_name else experiment_name,
                    group=group,
                    job_type="train",
                    config_name=config_name,
                    setting_name=setting_name,
                )
                experiment = mlflow_kwargs.pop("experiment_name", None)
                run_name = mlflow_kwargs.pop("run_name")
                tags = mlflow_kwargs.pop("tags", {})
                tracking_uri = mlflow_kwargs.pop("tracking_uri", None)
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                if experiment:
                    mlflow.set_experiment(experiment)
                active_run = mlflow.start_run(
                    run_name=run_name,
                    tags=tags,
                    **mlflow_kwargs,
                )
                self._mlflow_module = mlflow
                self._mlflow_run_id = active_run.info.run_id
                self._mlflow_metrics_logger = _AsyncMlflowMetricsLogger(
                    mlflow,
                    run_id=self._mlflow_run_id,
                )
                if isinstance(cfg_payload, dict):
                    try:
                        mlflow.log_dict(
                            cfg_payload,
                            "config.json",
                            run_id=self._mlflow_run_id,
                        )
                    except Exception as exc:  # noqa: BLE001 - config artifact logging is best-effort.
                        _LOGGER.warning(
                            "Failed to log MLflow config artifact: %s",
                            exc,
                            exc_info=True,
                        )
                return
            except Exception as exc:
                if self._mlflow_module is not None:
                    self._mlflow_module.end_run()
                    self._mlflow_module = None
                raise RuntimeError(
                    "Failed to initialize requested MLflow experiment logger"
                ) from exc

        try:
            import wandb

            cfg_payload = cfg.to_dict()
            experiment_cfg = cfg.experiment["logger"]
            wandb_kwargs = build_wandb_init_kwargs(
                experiment_cfg=experiment_cfg,
                run_name=wandb_name if wandb_name else experiment_name,
                run_dir=log_dir,
                default_project="InstantMimic",
                sync_tensorboard=False,
                config_payload=(cfg_payload if isinstance(cfg_payload, dict) else None),
                group=group,
                job_type="train",
            )
            self._wandb_run = wandb.init(**wandb_kwargs)
            # Keep distinct x-axes for iteration-based and frame-based metrics.
            self._wandb_run.define_metric(self._WANDB_TRAIN_ITERATION)
            self._wandb_run.define_metric(self._WANDB_TRAIN_FRAME)
            for tag in self._WANDB_ITERATION_METRIC_TAGS:
                self._wandb_run.define_metric(
                    tag, step_metric=self._WANDB_TRAIN_ITERATION
                )
                self._wandb_defined_iteration_metrics.add(tag)
            for tag in self._WANDB_FRAME_METRIC_TAGS:
                self._wandb_run.define_metric(tag, step_metric=self._WANDB_TRAIN_FRAME)
                self._wandb_defined_frame_metrics.add(tag)
        except Exception as exc:
            if self._wandb_run is not None:
                self._wandb_run.finish()
                self._wandb_run = None
            raise RuntimeError(
                "Failed to initialize requested W&B experiment logger"
            ) from exc

    def _define_wandb_metrics_for_payload(
        self,
        scalars: dict[str, float],
        *,
        axis: str,
    ) -> None:
        if self._wandb_run is None:
            return
        if axis == "frame":
            step_metric = self._WANDB_TRAIN_FRAME
            defined_metrics = self._wandb_defined_frame_metrics
        else:
            step_metric = self._WANDB_TRAIN_ITERATION
            defined_metrics = self._wandb_defined_iteration_metrics
        axis_metric_tags = {self._WANDB_TRAIN_ITERATION, self._WANDB_TRAIN_FRAME}
        for tag in scalars:
            if tag in axis_metric_tags:
                continue
            if tag not in defined_metrics:
                self._wandb_run.define_metric(tag, step_metric=step_metric)
                defined_metrics.add(tag)

    def log_eval_scalars(
        self,
        scalars: dict[str, float],
        x: int,
        *,
        grouped_metrics: Mapping[str, Mapping[str, str]],
        axis: str = "iteration",
    ) -> None:
        self.log_scalars(scalars, x, axis=axis)
        if self._wandb_run is None:
            return
        for title, metrics_by_group in grouped_metrics.items():
            self._wandb_grouped_metrics.setdefault(str(title), {}).update(
                {
                    str(group_name): str(metric_name)
                    for group_name, metric_name in metrics_by_group.items()
                }
            )

    def log_scalars(
        self, scalars: dict[str, float], x: int, *, axis: str = "iteration"
    ) -> None:
        if not scalars:
            return
        if self._writer is not None:
            _add_scalars_fast(self._writer, scalars, x)
            return
        if self._wandb_run is not None:
            self._define_wandb_metrics_for_payload(scalars, axis=axis)
            payload = {tag: float(value) for tag, value in scalars.items()}
            if axis == "frame":
                payload[self._WANDB_TRAIN_FRAME] = int(x)
            else:
                payload[self._WANDB_TRAIN_ITERATION] = int(x)
            self._wandb_run.log(payload)
            return
        mlflow_module = getattr(self, "_mlflow_module", None)
        if mlflow_module is not None:
            payload = _build_mlflow_metric_payload(scalars)
            mlflow_metrics_logger = getattr(self, "_mlflow_metrics_logger", None)
            if mlflow_metrics_logger is None:
                mlflow_metrics_logger = _AsyncMlflowMetricsLogger(
                    mlflow_module,
                    run_id=getattr(self, "_mlflow_run_id", None),
                )
                self._mlflow_metrics_logger = mlflow_metrics_logger
            mlflow_metrics_logger.log_metrics(payload, step=int(x))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._wandb_run is not None:
            if self._wandb_grouped_metrics:
                try:
                    workspace = _build_wandb_eval_workspace(
                        run=self._wandb_run,
                        grouped_metrics=self._wandb_grouped_metrics,
                    ).save()
                    _LOGGER.info("W&B native grouped Eval workspace: %s", workspace.url)
                except Exception as exc:  # noqa: BLE001 - remote UI setup is best-effort.
                    _LOGGER.warning(
                        "Failed to create W&B native grouped Eval workspace: %s",
                        exc,
                        exc_info=True,
                    )
            self._wandb_run.finish()
        mlflow_metrics_logger = getattr(self, "_mlflow_metrics_logger", None)
        mlflow_module = getattr(self, "_mlflow_module", None)
        try:
            if mlflow_metrics_logger is not None:
                self._mlflow_metrics_logger = None
                mlflow_metrics_logger.close()
        finally:
            if mlflow_module is not None:
                mlflow_module.end_run()
