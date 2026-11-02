from __future__ import annotations

import json
import logging
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch

from instantmimic.envs.eval_metrics import EvalOutcomeSummary
from instantmimic.train.metrics import (
    EvalMetricLoggingPolicy,
    _append_eval_result_scalars,
    _build_rollout_benchmark_scalars,
)

_LOGGER = logging.getLogger(__name__)
_DEFAULT_CHECKPOINT_METRIC = object()
_DISABLED_EVAL_BEST_METRICS = {"off", "none"}
_MINIMIZE_EVAL_METRICS = {
    "score",
}
_MAXIMIZE_EVAL_METRICS = {"success_rate", "reward_mean", "episode_length_mean"}
_EVAL_EPISODE_SEMANTICS = {"full_horizon", "terminal_success"}


def _format_benchmark_log_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _append_rollout_result_log_fields(
    fields: list[str],
    *,
    prefix: str,
    result: Mapping[str, object],
    exclude: Collection[str] = (),
) -> None:
    for name, value in result.items():
        if name in exclude:
            continue
        if isinstance(value, bool | int | float):
            field_name = f"{prefix}_{name}" if prefix else name
            fields.append(f"{field_name}={_format_benchmark_log_value(value)}")


def _metric_is_finite(result: dict[str, float | int | bool], key: str) -> bool:
    raw_value = result.get(key)
    if raw_value is None:
        return True
    try:
        return math.isfinite(float(raw_value))
    except (TypeError, ValueError):
        return False


def _valid_eval_has_nonfinite_metrics(
    result: dict[str, float | int | bool],
    *,
    finite_metric_names: Collection[str] = (),
) -> bool:
    if not bool(result.get("valid", False)):
        return False
    if not _metric_is_finite(result, "reward_mean"):
        return True
    return any(not _metric_is_finite(result, key) for key in finite_metric_names)


def _invalidate_nonfinite_eval_result(
    result: dict[str, float | int | bool],
    *,
    finite_metric_names: Collection[str] = (),
) -> dict[str, float | int | bool]:
    if not _valid_eval_has_nonfinite_metrics(
        result,
        finite_metric_names=finite_metric_names,
    ):
        return result
    sanitized = dict(result)
    total_envs = int(sanitized.get("total_envs", 0))
    invalid_step = sanitized.get("steps", sanitized.get("episode_length_mean", 0))
    try:
        first_invalid_step = int(float(invalid_step))
    except (TypeError, ValueError):
        first_invalid_step = int(sanitized.get("first_invalid_step", 0))
    success_defined = bool(
        sanitized.get("success_defined", "success_rate" in sanitized)
    )
    success_count = None
    first_failure_step = 0
    if success_defined:
        success_count = int(
            sanitized.get(
                "success_count",
                round(float(sanitized.get("success_rate", 0.0)) * total_envs),
            )
        )
        first_failure_step = int(sanitized.get("failure_step", 0))
    sanitized.pop("invalid_count", None)
    sanitized.update(
        EvalOutcomeSummary(
            total_envs=total_envs,
            valid_count=0,
            first_invalid_step=first_invalid_step,
            success_count=success_count,
            first_failure_step=first_failure_step,
        ).as_result()
    )
    sanitized["nonfinite_metrics"] = True
    return sanitized


def _normalize_eval_best_metric(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in _DISABLED_EVAL_BEST_METRICS:
        return None
    supported_metrics = _MINIMIZE_EVAL_METRICS | _MAXIMIZE_EVAL_METRICS
    if text not in supported_metrics:
        supported = ", ".join(sorted(supported_metrics | _DISABLED_EVAL_BEST_METRICS))
        raise ValueError(
            "Unsupported trainer.benchmark.best_metric="
            f"{value!r}. Supported values: {supported}."
        )
    return text


def _normalize_eval_best_metric_tie_breakers(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in _DISABLED_EVAL_BEST_METRICS:
            return ()
        raw_values = [item.strip() for item in text.split(",")]
    else:
        try:
            raw_values = list(value)
        except TypeError:
            raw_values = [value]

    metrics: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        metric = _normalize_eval_best_metric(raw_value)
        if metric is None or metric in seen:
            continue
        metrics.append(metric)
        seen.add(metric)
    return tuple(metrics)


def _normalize_eval_episode_semantics(value: Any) -> str:
    if value is None:
        return "full_horizon"
    text = str(value).strip().lower()
    if text not in _EVAL_EPISODE_SEMANTICS:
        supported = ", ".join(sorted(_EVAL_EPISODE_SEMANTICS))
        raise ValueError(
            f"Unsupported trainer.benchmark.episode_semantics={value!r}. "
            f"Supported values: {supported}."
        )
    return text


def _tracking_benchmark_passed(
    result: dict[str, float | int | bool],
    *,
    ape_mean_threshold: float | None,
    global_mpjpe_mean_threshold: float | None = None,
    ee_mean_threshold: float | None = None,
    success_rate_threshold: float | None = None,
) -> bool:
    if not bool(result.get("valid", False)):
        return False
    if (
        ape_mean_threshold is not None
        and float(result["ape_mean"]) > ape_mean_threshold
    ):
        return False
    if (
        global_mpjpe_mean_threshold is not None
        and float(result["global_mpjpe_mean"]) > global_mpjpe_mean_threshold
    ):
        return False
    if ee_mean_threshold is not None and float(result["ee_mean"]) > ee_mean_threshold:
        return False
    return not (
        success_rate_threshold is not None
        and float(result.get("success_rate", 0.0)) < success_rate_threshold
    )


def _tracking_benchmark_score(
    result: dict[str, float | int | bool],
    *,
    ape_mean_threshold: float | None,
    global_mpjpe_mean_threshold: float | None = None,
    ee_mean_threshold: float | None = None,
    success_rate_threshold: float | None = None,
) -> float | None:
    if not bool(result.get("valid", False)):
        return None
    if (
        not bool(result.get("tracking_metrics_available", False))
        and ape_mean_threshold is None
        and global_mpjpe_mean_threshold is None
        and ee_mean_threshold is None
        and success_rate_threshold is None
    ):
        return None
    score_terms: list[float] = []
    if ape_mean_threshold is not None:
        score_terms.append(float(result["ape_mean"]) / max(ape_mean_threshold, 1e-12))
    if global_mpjpe_mean_threshold is not None:
        score_terms.append(
            float(result["global_mpjpe_mean"]) / max(global_mpjpe_mean_threshold, 1e-12)
        )
    if ee_mean_threshold is not None:
        score_terms.append(float(result["ee_mean"]) / max(ee_mean_threshold, 1e-12))
    if success_rate_threshold is not None:
        success_rate = float(result.get("success_rate", 0.0))
        score_terms.append(success_rate_threshold / max(success_rate, 1e-12))
    if not score_terms:
        success_rate = float(result.get("success_rate", 0.0))
        ape_mean = float(result.get("ape_mean", 0.0))
        global_mpjpe_mean = float(result.get("global_mpjpe_mean", 0.0))
        ee_mean = float(result.get("ee_mean", 0.0))
        score_terms.append(1.0 / max(success_rate, 1e-12))
        score_terms.append(ape_mean)
        score_terms.append(global_mpjpe_mean)
        score_terms.append(ee_mean)
    return max(score_terms) if score_terms else 0.0


def _eval_metric_value(
    result: dict[str, float | int | bool],
    *,
    metric_key: str | None,
    ape_mean_threshold: float | None,
    global_mpjpe_mean_threshold: float | None = None,
    ee_mean_threshold: float | None = None,
    success_rate_threshold: float | None = None,
) -> float | None:
    if metric_key is None:
        return None
    if not bool(result.get("valid", False)):
        return None
    if metric_key == "score":
        return _tracking_benchmark_score(
            result,
            ape_mean_threshold=ape_mean_threshold,
            global_mpjpe_mean_threshold=global_mpjpe_mean_threshold,
            ee_mean_threshold=ee_mean_threshold,
            success_rate_threshold=success_rate_threshold,
        )
    raw_value = result.get(metric_key)
    if raw_value is None:
        return None
    value = float(raw_value)
    if not math.isfinite(value):
        return None
    return value


def _eval_success_rate_value(
    result: dict[str, float | int | bool],
) -> float | None:
    success_defined = bool(result.get("success_defined", "success_rate" in result))
    if not success_defined:
        return None
    raw_value = result.get("success_rate")
    if raw_value is None:
        return None
    value = float(raw_value)
    if not math.isfinite(value):
        return None
    return value


def _eval_metric_is_better(
    metric_key: str,
    *,
    current_value: float,
    best_value: float | None,
    direction: str | None = None,
) -> bool:
    if best_value is None:
        return True
    if direction is not None:
        if direction == "minimize":
            return current_value < best_value
        if direction == "maximize":
            return current_value > best_value
        raise ValueError(f"Unknown eval metric direction {direction!r}.")
    if metric_key in _MINIMIZE_EVAL_METRICS:
        return current_value < best_value
    if metric_key in _MAXIMIZE_EVAL_METRICS:
        return current_value > best_value
    raise ValueError(f"Unknown eval metric direction for {metric_key!r}.")


def _eval_result_is_better(
    result: dict[str, float | int | bool],
    *,
    best_result: dict[str, float | int | bool] | None,
    metric_key: str,
    metric_direction: str | None = None,
    tie_breakers: tuple[str, ...] = (),
    ape_mean_threshold: float | None,
    global_mpjpe_mean_threshold: float | None = None,
    ee_mean_threshold: float | None = None,
    success_rate_threshold: float | None = None,
) -> bool:
    if not bool(result.get("valid", False)):
        return False
    current_value = _eval_metric_value(
        result,
        metric_key=metric_key,
        ape_mean_threshold=ape_mean_threshold,
        global_mpjpe_mean_threshold=global_mpjpe_mean_threshold,
        ee_mean_threshold=ee_mean_threshold,
        success_rate_threshold=success_rate_threshold,
    )
    if current_value is None:
        return False
    if best_result is None:
        return True

    best_value = _eval_metric_value(
        best_result,
        metric_key=metric_key,
        ape_mean_threshold=ape_mean_threshold,
        global_mpjpe_mean_threshold=global_mpjpe_mean_threshold,
        ee_mean_threshold=ee_mean_threshold,
        success_rate_threshold=success_rate_threshold,
    )
    if best_value is None:
        return True

    # Valid-only metrics are not comparable across different success coverage.
    # Overall rollout metrics already include failed trajectories and must be
    # compared directly.
    if metric_key not in {
        "reward_mean",
        "episode_length_mean",
        "valid_rate",
        "success_rate",
    }:
        current_success_rate = _eval_success_rate_value(result)
        best_success_rate = _eval_success_rate_value(best_result)
        if current_success_rate is not None and best_success_rate is not None:
            if current_success_rate > best_success_rate:
                return True
            if current_success_rate < best_success_rate:
                return False

    if _eval_metric_is_better(
        metric_key,
        current_value=current_value,
        best_value=best_value,
        direction=metric_direction,
    ):
        return True
    if _eval_metric_is_better(
        metric_key,
        current_value=best_value,
        best_value=current_value,
        direction=metric_direction,
    ):
        return False

    for tie_breaker in tie_breakers:
        current_tie_value = _eval_metric_value(
            result,
            metric_key=tie_breaker,
            ape_mean_threshold=ape_mean_threshold,
            global_mpjpe_mean_threshold=global_mpjpe_mean_threshold,
            ee_mean_threshold=ee_mean_threshold,
            success_rate_threshold=success_rate_threshold,
        )
        best_tie_value = _eval_metric_value(
            best_result,
            metric_key=tie_breaker,
            ape_mean_threshold=ape_mean_threshold,
            global_mpjpe_mean_threshold=global_mpjpe_mean_threshold,
            ee_mean_threshold=ee_mean_threshold,
            success_rate_threshold=success_rate_threshold,
        )
        if current_tie_value is None or best_tie_value is None:
            continue
        if _eval_metric_is_better(
            tie_breaker,
            current_value=current_tie_value,
            best_value=best_tie_value,
        ):
            return True
        if _eval_metric_is_better(
            tie_breaker,
            current_value=best_tie_value,
            best_value=current_tie_value,
        ):
            return False
    return False


def _append_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")


@dataclass(frozen=True)
class TrainingEvalPlan:
    interval: int
    interval_seconds: float | None
    evaluate_at_start: bool
    play_steps: int
    metric_frame: str
    include_root_in_ape: bool
    deterministic_reset: bool
    enable_early_termination: bool | None
    early_termination_mode: str | None
    phc_termination_distance: float | None
    require_full_horizon: bool
    episode_semantics: str
    use_best_candidate: bool
    num_envs: int
    ape_mean_threshold: float | None
    global_mpjpe_mean_threshold: float | None
    ee_mean_threshold: float | None
    success_rate_threshold: float | None
    consecutive_passes_required: int
    best_metric: str | None
    best_metric_direction: str | None
    best_metric_tie_breakers: tuple[str, ...]
    metric_names: tuple[str, ...] | None
    logging_statistics: tuple[str, ...]
    logging_exclude: tuple[str, ...]
    save_history: bool
    cuda_graph: bool
    execution: str
    history_filename: str

    @classmethod
    def disabled(cls) -> TrainingEvalPlan:
        return cls(
            interval=0,
            interval_seconds=None,
            evaluate_at_start=False,
            play_steps=0,
            metric_frame="root",
            include_root_in_ape=False,
            deterministic_reset=True,
            enable_early_termination=None,
            early_termination_mode=None,
            phc_termination_distance=None,
            require_full_horizon=True,
            episode_semantics="full_horizon",
            use_best_candidate=False,
            num_envs=1,
            ape_mean_threshold=None,
            global_mpjpe_mean_threshold=None,
            ee_mean_threshold=None,
            success_rate_threshold=None,
            consecutive_passes_required=1,
            best_metric=None,
            best_metric_direction=None,
            best_metric_tie_breakers=(),
            metric_names=None,
            logging_statistics=("mean", "rate"),
            logging_exclude=(),
            save_history=True,
            cuda_graph=True,
            execution="inline",
            history_filename="tracking_history.jsonl",
        )

    @classmethod
    def from_semantic_benchmark(
        cls,
        benchmark: Mapping[str, Any],
    ) -> TrainingEvalPlan:
        termination = benchmark.get("termination", {"type": "inherit"})
        if not isinstance(termination, Mapping):
            raise TypeError("trainer.benchmark.termination must be a mapping")
        termination_type = termination.get("type", "inherit")
        enable_early_termination: bool | None = None
        early_termination_mode: str | None = None
        phc_termination_distance: float | None = None
        if termination_type == "disabled":
            enable_early_termination = False
        elif termination_type == "phc_tracking":
            enable_early_termination = True
            early_termination_mode = "phc_tracking"
            phc_termination_distance = float(termination["distance"])
        elif termination_type != "inherit":
            raise ValueError(
                "trainer.benchmark.termination.type must be inherit, disabled, or phc_tracking"
            )

        thresholds = benchmark.get("thresholds", {})
        if not isinstance(thresholds, Mapping):
            raise TypeError("trainer.benchmark.thresholds must be a mapping")
        cuda_graph = bool(benchmark.get("cuda_graph", True))
        execution_mode = str(benchmark.get("execution", "inline"))
        if execution_mode not in {"inline", "overlap"}:
            raise ValueError(
                f"unsupported trainer.benchmark.execution={execution_mode!r}"
            )
        if execution_mode == "overlap" and not cuda_graph:
            raise ValueError(
                "trainer.benchmark.execution=overlap requires cuda_graph=true"
            )
        raw_metrics = benchmark.get("metrics")
        metric_names = (
            None
            if raw_metrics is None
            else tuple(dict.fromkeys(str(metric) for metric in raw_metrics))
        )
        raw_logging = benchmark.get("logging", {})
        if not isinstance(raw_logging, Mapping):
            raise TypeError("trainer.benchmark.logging must be a mapping")
        logging_statistics = tuple(
            dict.fromkeys(
                str(statistic)
                for statistic in raw_logging.get("statistics", ("mean", "rate"))
            )
        )
        logging_exclude = tuple(
            dict.fromkeys(str(metric) for metric in raw_logging.get("exclude", ()))
        )
        raw_objective = benchmark.get("objective")
        if raw_objective is not None:
            if not isinstance(raw_objective, Mapping):
                raise TypeError("trainer.benchmark.objective must be a mapping")
            best_metric = str(raw_objective["metric"])
            best_metric_direction = str(raw_objective["direction"])
        else:
            best_metric = _normalize_eval_best_metric(benchmark.get("best_metric"))
            best_metric_direction = None
        return cls(
            interval=int(benchmark.get("interval", 0)),
            interval_seconds=(
                float(benchmark["interval_seconds"])
                if "interval_seconds" in benchmark
                else None
            ),
            evaluate_at_start=bool(benchmark.get("evaluate_at_start", False)),
            play_steps=int(benchmark["steps"]),
            metric_frame=str(benchmark.get("metric_frame", "root")),
            include_root_in_ape=bool(benchmark.get("include_root_in_ape", False)),
            deterministic_reset=bool(benchmark.get("deterministic_reset", True)),
            enable_early_termination=enable_early_termination,
            early_termination_mode=early_termination_mode,
            phc_termination_distance=phc_termination_distance,
            require_full_horizon=bool(benchmark.get("require_full_horizon", True)),
            episode_semantics=_normalize_eval_episode_semantics(
                benchmark.get("episode_semantics")
            ),
            use_best_candidate=bool(benchmark.get("use_best_candidate", False)),
            num_envs=int(benchmark["num_envs"]),
            ape_mean_threshold=(
                float(thresholds["ape_mean"]) if "ape_mean" in thresholds else None
            ),
            global_mpjpe_mean_threshold=(
                float(thresholds["global_mpjpe_mean"])
                if "global_mpjpe_mean" in thresholds
                else None
            ),
            ee_mean_threshold=(
                float(thresholds["ee_mean"]) if "ee_mean" in thresholds else None
            ),
            success_rate_threshold=(
                float(thresholds["success_rate"])
                if "success_rate" in thresholds
                else None
            ),
            consecutive_passes_required=int(thresholds.get("consecutive_passes", 1)),
            best_metric=best_metric,
            best_metric_direction=best_metric_direction,
            best_metric_tie_breakers=_normalize_eval_best_metric_tie_breakers(
                benchmark.get("best_metric_tie_breakers")
            ),
            metric_names=metric_names,
            logging_statistics=logging_statistics,
            logging_exclude=logging_exclude,
            save_history=bool(benchmark.get("save_history", True)),
            cuda_graph=cuda_graph,
            execution=execution_mode,
            history_filename=str(
                benchmark.get("history_filename", "tracking_history.jsonl")
            ),
        )

    @property
    def enabled(self) -> bool:
        return (
            self.interval > 0 or self.interval_seconds is not None
        ) and self.play_steps > 0

    @property
    def threshold_enabled(self) -> bool:
        return self.enabled and (
            self.ape_mean_threshold is not None
            or self.global_mpjpe_mean_threshold is not None
            or self.ee_mean_threshold is not None
            or self.success_rate_threshold is not None
        )

    @property
    def best_metric_enabled(self) -> bool:
        return self.enabled and self.best_metric is not None

    @property
    def required_metric_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.best_metric is not None:
            names.append(self.best_metric)
        names.extend(self.best_metric_tie_breakers)
        for name, threshold in (
            ("ape_mean", self.ape_mean_threshold),
            ("global_mpjpe_mean", self.global_mpjpe_mean_threshold),
            ("ee_mean", self.ee_mean_threshold),
            ("success_rate", self.success_rate_threshold),
        ):
            if threshold is not None:
                names.append(name)
        return tuple(dict.fromkeys(names))

    @property
    def interval_ns(self) -> int | None:
        if self.interval_seconds is None:
            return None
        return int(self.interval_seconds * 1e9)

    def history_path(self, log_dir: Path) -> Path | None:
        if not self.enabled or not self.save_history:
            return None
        return (Path(log_dir) / "eval" / self.history_filename).resolve()


@dataclass(frozen=True)
class TrainingEvalOutcome:
    ran: bool
    capture_best_candidate: bool = False
    best_updated: bool = False
    threshold_reached: bool = False
    eval_elapsed_ns: int = 0
    active_elapsed_ns: int = 0
    gate_source: str = "current"
    gate_result: dict[str, float | int | bool] | None = None
    snapshot_iteration: int | None = None
    snapshot_step: int | None = None
    snapshot_frames: int | None = None
    completion_iteration: int | None = None


@dataclass
class _PendingTrainingEval:
    job: Any
    metadata: dict[str, int]
    checkpoint_handle: object | None
    best_candidate_enabled: bool


class TrainingEvalController:
    def __init__(
        self,
        *,
        plan: TrainingEvalPlan,
        runner: Any | None,
        metrics_logger: Any,
        history_path: Path | None,
        checkpoint_controller: Any | None = None,
        checkpoint_metric: str | None | object = _DEFAULT_CHECKPOINT_METRIC,
        checkpoint_metric_direction: str | None = None,
    ) -> None:
        self.plan = plan
        self.runner = runner
        self.metrics_logger = metrics_logger
        self.history_path = history_path
        self.checkpoint_controller = checkpoint_controller
        self.checkpoint_metric = (
            plan.best_metric
            if checkpoint_metric is _DEFAULT_CHECKPOINT_METRIC
            else checkpoint_metric
        )
        self.checkpoint_metric_direction = checkpoint_metric_direction
        self._eval_foreground_time_ns_acc = 0
        self.eval_count = 0
        self.pass_streak = 0
        self.best_score: float | None = None
        self.best_metric_value: float | None = None
        self.best_eval: dict[str, float | int | bool] | None = None
        self.best_checkpoint_metric_value: float | None = None
        self.best_checkpoint_eval: dict[str, float | int | bool] | None = None
        self.last_eval: dict[str, float | int | bool] | None = None
        self.last_eval_snapshot_iteration: int | None = None
        self.next_eval_deadline_ns = plan.interval_ns
        self._pending_eval: _PendingTrainingEval | None = None

    @property
    def eval_time_ns_acc(self) -> int:
        overlap_ns = int(
            getattr(self.checkpoint_controller, "eval_checkpoint_overlap_time_ns", 0)
        )
        return max(0, self._eval_foreground_time_ns_acc - overlap_ns)

    def add_foreground_time_ns(self, elapsed_ns: int) -> None:
        """Include externally measured benchmark work, excluding checkpoint work."""

        self._eval_foreground_time_ns_acc += max(0, int(elapsed_ns))

    def _begin_eval_foreground_time(self) -> int:
        begin = getattr(self.checkpoint_controller, "begin_eval_foreground", None)
        if callable(begin):
            begin()
        return perf_counter_ns()

    def _add_eval_foreground_time(
        self,
        *,
        start_ns: int,
    ) -> None:
        end_ns = perf_counter_ns()
        self._eval_foreground_time_ns_acc += max(0, end_ns - int(start_ns))
        record = getattr(self.checkpoint_controller, "end_eval_foreground", None)
        if callable(record):
            record(int(start_ns), end_ns)

    def due_by_iteration(self, iteration_index: int, total_rollouts: int) -> bool:
        return (
            self.plan.enabled
            and self.plan.interval_seconds is None
            and self.plan.interval > 0
            and int(iteration_index) % int(self.plan.interval) == 0
        ) or (self.plan.enabled and int(iteration_index) == int(total_rollouts))

    def due_by_time(self, active_elapsed_ns: int) -> bool:
        return (
            self.plan.enabled
            and self.plan.interval_ns is not None
            and self.next_eval_deadline_ns is not None
            and int(active_elapsed_ns) >= int(self.next_eval_deadline_ns)
        )

    def should_run(
        self,
        *,
        iteration_index: int,
        total_rollouts: int,
        active_elapsed_ns: int,
    ) -> bool:
        return self.due_by_iteration(
            iteration_index, total_rollouts
        ) or self.due_by_time(active_elapsed_ns)

    def run_if_due(
        self,
        *,
        iteration_index: int,
        total_rollouts: int,
        step_index: int,
        frames: int,
        active_elapsed_ns: int,
        checkpoint_time_ns_acc: int,
        best_candidate_enabled: bool = True,
        force: bool = False,
    ) -> TrainingEvalOutcome:
        due = bool(force) or self.should_run(
            iteration_index=iteration_index,
            total_rollouts=total_rollouts,
            active_elapsed_ns=active_elapsed_ns,
        )
        if self.plan.execution == "overlap" and not force:
            return self._run_overlap(
                due=due,
                iteration_index=iteration_index,
                total_rollouts=total_rollouts,
                step_index=step_index,
                frames=frames,
                active_elapsed_ns=active_elapsed_ns,
                checkpoint_time_ns_acc=checkpoint_time_ns_acc,
                best_candidate_enabled=best_candidate_enabled,
            )
        if not due:
            return TrainingEvalOutcome(ran=False, active_elapsed_ns=active_elapsed_ns)
        if self.runner is None:
            raise RuntimeError(
                "Training eval is due but no benchmark runner is configured."
            )

        foreground_start_ns = self._begin_eval_foreground_time()
        try:
            benchmark_start_ns = perf_counter_ns()
            with torch.cuda.nvtx.range("benchmark_eval"):
                benchmark_result = self.runner.run()
            eval_elapsed_ns = perf_counter_ns() - benchmark_start_ns
            self._advance_time_deadline(active_elapsed_ns)
            return self._process_result(
                benchmark_result=benchmark_result,
                eval_elapsed_ns=eval_elapsed_ns,
                active_elapsed_ns=active_elapsed_ns,
                checkpoint_time_ns_acc=checkpoint_time_ns_acc,
                iteration_index=iteration_index,
                completion_iteration=iteration_index,
                step_index=step_index,
                frames=frames,
                best_candidate_enabled=best_candidate_enabled,
                checkpoint_handle=None,
            )
        finally:
            self._add_eval_foreground_time(
                start_ns=foreground_start_ns,
            )

    def _run_overlap(
        self,
        *,
        due: bool,
        iteration_index: int,
        total_rollouts: int,
        step_index: int,
        frames: int,
        active_elapsed_ns: int,
        checkpoint_time_ns_acc: int,
        best_candidate_enabled: bool,
    ) -> TrainingEvalOutcome:
        completed_outcome: TrainingEvalOutcome | None = None
        pending = self._pending_eval
        if pending is not None and pending.job.done():
            completed_outcome = self._consume_pending(
                completion_iteration=iteration_index,
                checkpoint_time_ns_acc=checkpoint_time_ns_acc,
            )
        elif pending is not None and due:
            wait_start_ns = self._begin_eval_foreground_time()
            try:
                with torch.cuda.nvtx.range("benchmark_eval_wait"):
                    pending.job.wait()
            finally:
                self._add_eval_foreground_time(start_ns=wait_start_ns)
            completed_outcome = self._consume_pending(
                completion_iteration=iteration_index,
                checkpoint_time_ns_acc=checkpoint_time_ns_acc,
            )

        if completed_outcome is not None and completed_outcome.threshold_reached:
            return completed_outcome
        if not due:
            return completed_outcome or TrainingEvalOutcome(
                ran=False,
                active_elapsed_ns=active_elapsed_ns,
            )
        if self.runner is None:
            raise RuntimeError(
                "Training eval is due but no benchmark runner is configured."
            )
        if best_candidate_enabled and self.checkpoint_controller is None:
            raise RuntimeError(
                "Overlapped eval best tracking requires a checkpoint controller "
                "so the evaluated snapshot can be promoted exactly."
            )

        metadata = {
            "iteration": int(iteration_index),
            "step": int(step_index),
            "frames": int(frames),
            "active_elapsed_ns": int(active_elapsed_ns),
        }
        checkpoint_handle: object | None = None
        launch_start_ns = self._begin_eval_foreground_time()
        try:
            if best_candidate_enabled and self.checkpoint_controller is not None:
                checkpoint_handle = self.checkpoint_controller.capture_eval_candidate(
                    metadata=metadata
                )
            with torch.cuda.nvtx.range("benchmark_eval_launch"):
                job = self.runner.launch(snapshot_metadata=metadata)
        except BaseException:
            if checkpoint_handle is not None:
                self.checkpoint_controller.discard_eval_candidate(checkpoint_handle)
            raise
        finally:
            self._add_eval_foreground_time(
                start_ns=launch_start_ns,
            )
        self._pending_eval = _PendingTrainingEval(
            job=job,
            metadata=metadata,
            checkpoint_handle=checkpoint_handle,
            best_candidate_enabled=bool(best_candidate_enabled),
        )
        self._advance_time_deadline(active_elapsed_ns)

        if int(iteration_index) == int(total_rollouts):
            wait_start_ns = self._begin_eval_foreground_time()
            try:
                with torch.cuda.nvtx.range("benchmark_eval_final_wait"):
                    job.wait()
            finally:
                self._add_eval_foreground_time(start_ns=wait_start_ns)
            return self._consume_pending(
                completion_iteration=iteration_index,
                checkpoint_time_ns_acc=checkpoint_time_ns_acc,
            )
        return completed_outcome or TrainingEvalOutcome(
            ran=False,
            active_elapsed_ns=active_elapsed_ns,
        )

    def _consume_pending(
        self,
        *,
        completion_iteration: int,
        checkpoint_time_ns_acc: int,
    ) -> TrainingEvalOutcome:
        pending = self._pending_eval
        if pending is None:
            return TrainingEvalOutcome(ran=False)
        self._pending_eval = None
        consume_start_ns = self._begin_eval_foreground_time()
        try:
            try:
                benchmark_result = pending.job.result()
                eval_elapsed_ns = int(getattr(pending.job, "elapsed_ns", 0))
                metadata = pending.metadata
                return self._process_result(
                    benchmark_result=benchmark_result,
                    eval_elapsed_ns=eval_elapsed_ns,
                    active_elapsed_ns=int(metadata["active_elapsed_ns"]),
                    checkpoint_time_ns_acc=checkpoint_time_ns_acc,
                    iteration_index=int(metadata["iteration"]),
                    completion_iteration=int(completion_iteration),
                    step_index=int(metadata["step"]),
                    frames=int(metadata["frames"]),
                    best_candidate_enabled=pending.best_candidate_enabled,
                    checkpoint_handle=pending.checkpoint_handle,
                )
            except BaseException:
                if pending.checkpoint_handle is not None:
                    assert self.checkpoint_controller is not None
                    self.checkpoint_controller.discard_eval_candidate(
                        pending.checkpoint_handle
                    )
                raise
        finally:
            self._add_eval_foreground_time(
                start_ns=consume_start_ns,
            )

    def drain_pending(
        self,
        *,
        completion_iteration: int,
        checkpoint_time_ns_acc: int,
    ) -> TrainingEvalOutcome:
        pending = self._pending_eval
        if pending is None:
            return TrainingEvalOutcome(ran=False)
        wait_start_ns = self._begin_eval_foreground_time()
        try:
            with torch.cuda.nvtx.range("benchmark_eval_drain"):
                pending.job.wait()
        finally:
            self._add_eval_foreground_time(start_ns=wait_start_ns)
        return self._consume_pending(
            completion_iteration=completion_iteration,
            checkpoint_time_ns_acc=checkpoint_time_ns_acc,
        )

    def _process_result(
        self,
        *,
        benchmark_result: dict[str, float | int | bool],
        eval_elapsed_ns: int,
        active_elapsed_ns: int,
        checkpoint_time_ns_acc: int,
        iteration_index: int,
        completion_iteration: int,
        step_index: int,
        frames: int,
        best_candidate_enabled: bool,
        checkpoint_handle: object | None,
    ) -> TrainingEvalOutcome:
        benchmark_result = _invalidate_nonfinite_eval_result(
            benchmark_result,
            finite_metric_names=getattr(self.runner, "finite_metric_names", ()),
        )

        capture_best_candidate = False
        self.last_eval = benchmark_result
        self.last_eval_snapshot_iteration = int(iteration_index)
        current_score = _tracking_benchmark_score(
            benchmark_result,
            ape_mean_threshold=self.plan.ape_mean_threshold,
            global_mpjpe_mean_threshold=self.plan.global_mpjpe_mean_threshold,
            ee_mean_threshold=self.plan.ee_mean_threshold,
            success_rate_threshold=self.plan.success_rate_threshold,
        )
        if current_score is not None and (
            self.best_score is None or current_score < self.best_score
        ):
            self.best_score = current_score
            if self.plan.threshold_enabled and self.plan.best_metric is None:
                self.best_eval = dict(benchmark_result)

        current_metric_value = _eval_metric_value(
            benchmark_result,
            metric_key=self.plan.best_metric,
            ape_mean_threshold=self.plan.ape_mean_threshold,
            global_mpjpe_mean_threshold=self.plan.global_mpjpe_mean_threshold,
            ee_mean_threshold=self.plan.ee_mean_threshold,
            success_rate_threshold=self.plan.success_rate_threshold,
        )
        benchmark_metric_improved = (
            self.plan.best_metric is not None
            and current_metric_value is not None
            and _eval_result_is_better(
                benchmark_result,
                best_result=self.best_eval,
                metric_key=self.plan.best_metric,
                metric_direction=self.plan.best_metric_direction,
                tie_breakers=self.plan.best_metric_tie_breakers,
                ape_mean_threshold=self.plan.ape_mean_threshold,
                global_mpjpe_mean_threshold=self.plan.global_mpjpe_mean_threshold,
                ee_mean_threshold=self.plan.ee_mean_threshold,
                success_rate_threshold=self.plan.success_rate_threshold,
            )
        )
        if benchmark_metric_improved:
            self.best_metric_value = current_metric_value
            self.best_eval = dict(benchmark_result)

        checkpoint_metric_value = _eval_metric_value(
            benchmark_result,
            metric_key=self.checkpoint_metric,
            ape_mean_threshold=self.plan.ape_mean_threshold,
            global_mpjpe_mean_threshold=self.plan.global_mpjpe_mean_threshold,
            ee_mean_threshold=self.plan.ee_mean_threshold,
            success_rate_threshold=self.plan.success_rate_threshold,
        )
        checkpoint_metric_improved = (
            best_candidate_enabled
            and self.checkpoint_metric is not None
            and checkpoint_metric_value is not None
            and _eval_result_is_better(
                benchmark_result,
                best_result=self.best_checkpoint_eval,
                metric_key=self.checkpoint_metric,
                metric_direction=self.checkpoint_metric_direction,
                tie_breakers=(),
                ape_mean_threshold=self.plan.ape_mean_threshold,
                global_mpjpe_mean_threshold=self.plan.global_mpjpe_mean_threshold,
                ee_mean_threshold=self.plan.ee_mean_threshold,
                success_rate_threshold=self.plan.success_rate_threshold,
            )
        )
        if checkpoint_metric_improved:
            self.best_checkpoint_metric_value = checkpoint_metric_value
            self.best_checkpoint_eval = dict(benchmark_result)
            capture_best_candidate = True

        gate_result = benchmark_result
        gate_score = current_score
        gate_source = "current"
        if self.plan.use_best_candidate and self.best_eval is not None:
            gate_result = self.best_eval
            gate_score = _tracking_benchmark_score(
                gate_result,
                ape_mean_threshold=self.plan.ape_mean_threshold,
                global_mpjpe_mean_threshold=self.plan.global_mpjpe_mean_threshold,
                ee_mean_threshold=self.plan.ee_mean_threshold,
                success_rate_threshold=self.plan.success_rate_threshold,
            )
            gate_source = "best"
        gate_metric_value = _eval_metric_value(
            gate_result,
            metric_key=self.plan.best_metric,
            ape_mean_threshold=self.plan.ape_mean_threshold,
            global_mpjpe_mean_threshold=self.plan.global_mpjpe_mean_threshold,
            ee_mean_threshold=self.plan.ee_mean_threshold,
            success_rate_threshold=self.plan.success_rate_threshold,
        )

        benchmark_passed = self.plan.threshold_enabled and _tracking_benchmark_passed(
            gate_result,
            ape_mean_threshold=self.plan.ape_mean_threshold,
            global_mpjpe_mean_threshold=self.plan.global_mpjpe_mean_threshold,
            ee_mean_threshold=self.plan.ee_mean_threshold,
            success_rate_threshold=self.plan.success_rate_threshold,
        )
        if benchmark_passed:
            self.pass_streak += 1
        else:
            self.pass_streak = 0
        self.eval_count += 1

        self._log_eval(
            result=benchmark_result,
            gate_result=gate_result,
            gate_source=gate_source,
            current_score=current_score,
            current_metric_value=current_metric_value,
            gate_score=gate_score,
            gate_metric_value=gate_metric_value,
            eval_elapsed_ns=eval_elapsed_ns,
            active_elapsed_ns=active_elapsed_ns,
            iteration_index=iteration_index,
            step_index=step_index,
            frames=frames,
        )
        benchmark_scalars = _build_rollout_benchmark_scalars(
            benchmark_pass_streak=self.pass_streak,
            benchmark_current_score=current_score,
            best_benchmark_metric_value=self.best_metric_value,
            benchmark_threshold_enabled=self.plan.threshold_enabled,
            benchmark_eval_elapsed_ns=eval_elapsed_ns,
            benchmark_active_elapsed_ns=active_elapsed_ns,
            checkpoint_time_ns_acc=checkpoint_time_ns_acc,
            frames=frames,
        )
        metric_labels = getattr(self.runner, "metric_labels", {})
        metric_statistics = getattr(self.runner, "metric_statistics", {})
        metric_groups = getattr(self.runner, "metric_groups", ())
        if not isinstance(metric_labels, Mapping):
            metric_labels = {}
        if not isinstance(metric_statistics, Mapping):
            metric_statistics = {}
        logging_policy = EvalMetricLoggingPolicy(
            statistics=frozenset(self.plan.logging_statistics),
            exclude=frozenset(self.plan.logging_exclude),
        )
        grouped_metrics = _append_eval_result_scalars(
            benchmark_scalars,
            benchmark_result,
            labels=metric_labels,
            statistics=metric_statistics,
            groups=metric_groups,
            policy=logging_policy,
        )
        if self.plan.use_best_candidate:
            gate_grouped_metrics = _append_eval_result_scalars(
                benchmark_scalars,
                gate_result,
                labels=metric_labels,
                statistics=metric_statistics,
                groups=metric_groups,
                policy=logging_policy,
                prefix="Eval / Gate / ",
            )
            grouped_metrics.update(gate_grouped_metrics)
        if checkpoint_metric_value is not None:
            benchmark_scalars["Checkpoint / Candidate metric value"] = float(
                checkpoint_metric_value
            )
        if self.best_checkpoint_metric_value is not None:
            benchmark_scalars["Checkpoint / Best metric value"] = float(
                self.best_checkpoint_metric_value
            )
        log_eval_scalars = getattr(self.metrics_logger, "log_eval_scalars", None)
        if callable(log_eval_scalars):
            log_eval_scalars(
                benchmark_scalars,
                iteration_index,
                grouped_metrics=grouped_metrics,
                axis="iteration",
            )
        else:
            self.metrics_logger.log_scalars(
                benchmark_scalars,
                iteration_index,
                axis="iteration",
            )
        if self.history_path is not None:
            _append_jsonl_record(
                self.history_path,
                {
                    "eval_index": int(self.eval_count),
                    "iteration": int(iteration_index),
                    "completion_iteration": int(completion_iteration),
                    "step": int(step_index),
                    "frames": int(frames),
                    "active_train_time_s": float(active_elapsed_ns * 1e-9),
                    "checkpoint_time_s": float(checkpoint_time_ns_acc * 1e-9),
                    "eval_ms": float(eval_elapsed_ns * 1e-6),
                    "best_metric": self.plan.best_metric,
                    "best_metric_tie_breakers": list(
                        self.plan.best_metric_tie_breakers
                    ),
                    "current_metric_value": current_metric_value,
                    "best_metric_value": self.best_metric_value,
                    "metric_improved": bool(benchmark_metric_improved),
                    "checkpoint_metric": self.checkpoint_metric,
                    "checkpoint_metric_value": checkpoint_metric_value,
                    "best_checkpoint_metric_value": (self.best_checkpoint_metric_value),
                    "checkpoint_metric_improved": bool(checkpoint_metric_improved),
                    "gate_source": gate_source,
                    "pass_streak": int(self.pass_streak),
                    "current": dict(benchmark_result),
                    "gate": dict(gate_result),
                },
            )

        threshold_reached = (
            benchmark_passed
            and self.pass_streak >= self.plan.consecutive_passes_required
        )
        if threshold_reached:
            self._log_threshold_reached(
                gate_source=gate_source,
                gate_result=gate_result,
                iteration_index=iteration_index,
                step_index=step_index,
                frames=frames,
            )
        if checkpoint_handle is not None:
            assert self.checkpoint_controller is not None
            if capture_best_candidate:
                self.checkpoint_controller.promote_eval_candidate(checkpoint_handle)
            else:
                self.checkpoint_controller.discard_eval_candidate(checkpoint_handle)
            capture_best_candidate = False
        return TrainingEvalOutcome(
            ran=True,
            capture_best_candidate=capture_best_candidate,
            best_updated=bool(checkpoint_metric_improved),
            threshold_reached=threshold_reached,
            eval_elapsed_ns=eval_elapsed_ns,
            active_elapsed_ns=active_elapsed_ns,
            gate_source=gate_source,
            gate_result=dict(gate_result),
            snapshot_iteration=int(iteration_index),
            snapshot_step=int(step_index),
            snapshot_frames=int(frames),
            completion_iteration=int(completion_iteration),
        )

    def _advance_time_deadline(self, active_elapsed_ns: int) -> None:
        if self.plan.interval_ns is None or self.next_eval_deadline_ns is None:
            return
        while self.next_eval_deadline_ns <= int(active_elapsed_ns):
            self.next_eval_deadline_ns += self.plan.interval_ns

    def _log_eval(
        self,
        *,
        result: dict[str, float | int | bool],
        gate_result: dict[str, float | int | bool],
        gate_source: str,
        current_score: float | None,
        current_metric_value: float | None,
        gate_score: float | None,
        gate_metric_value: float | None,
        eval_elapsed_ns: int,
        active_elapsed_ns: int,
        iteration_index: int,
        step_index: int,
        frames: int,
    ) -> None:
        fields = [
            f"eval_index={self.eval_count}",
            f"iteration={iteration_index}",
            f"step={step_index}",
            f"frames={frames}",
            f"active_train_time={active_elapsed_ns * 1e-9:.6f}",
        ]
        _append_rollout_result_log_fields(
            fields,
            prefix="current",
            result=result,
            exclude=("total_envs",),
        )
        fields.extend(
            (
                f"total_envs={int(result.get('total_envs', 0))}",
                f"pass_streak={self.pass_streak}",
                f"current_score={_format_benchmark_log_value(current_score)}",
                f"best_metric={self.plan.best_metric or 'off'}",
                "current_metric=" + _format_benchmark_log_value(current_metric_value),
                f"gate_source={gate_source}",
            )
        )
        _append_rollout_result_log_fields(
            fields,
            prefix="gate",
            result=gate_result,
            exclude=("total_envs",),
        )
        fields.extend(
            (
                f"gate_score={_format_benchmark_log_value(gate_score)}",
                f"gate_metric={_format_benchmark_log_value(gate_metric_value)}",
                f"eval_ms={eval_elapsed_ns * 1e-6:.3f}",
            )
        )
        _LOGGER.info("BENCHMARK_EVAL %s", " ".join(fields))

    def _log_threshold_reached(
        self,
        *,
        gate_source: str,
        gate_result: dict[str, float | int | bool],
        iteration_index: int,
        step_index: int,
        frames: int,
    ) -> None:
        fields = [f"source={gate_source}"]
        thresholds = {
            "ape_mean": self.plan.ape_mean_threshold,
            "global_mpjpe_mean": self.plan.global_mpjpe_mean_threshold,
            "ee_mean": self.plan.ee_mean_threshold,
            "success_rate": self.plan.success_rate_threshold,
        }
        for name, value in thresholds.items():
            if value is not None:
                fields.append(f"{name}_threshold={value:.6f}")
        fields.append(f"consecutive_passes={self.pass_streak}")
        _append_rollout_result_log_fields(fields, prefix="", result=gate_result)
        fields.extend(
            (
                f"iteration={iteration_index}",
                f"step={step_index}",
                f"frames={frames}",
            )
        )
        _LOGGER.info("TRACKING_BENCHMARK_REACHED %s", " ".join(fields))
