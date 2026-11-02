"""The frontier comparator family: a set of points, not a winner.

A Pareto objective refuses to collapse its axes into one number. What it keeps
is the non-dominated set; the named representatives -- fastest, best-quality,
balanced, the knee -- are reporting views over that set, and the *anchor* is the
one the planner is pointed at this cycle, rotating so the search does not fixate
on a single corner.

Three shapes share this implementation, selected by ``objective.strategy``:

* ``pareto`` -- active train time against a fixed-evaluator quality.
* ``axes_pareto`` -- N named axes, each with its own goal and epsilon.
* ``learning_curve_pareto`` -- final quality against anytime (AUC) quality,
  with an optional floor that constrains the final-quality pick.

The archive accumulates across cycles, so a frontier point found in cycle 3
still competes in cycle 40. Resuming with a changed objective is an error: the
points already in the archive were not measured against it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from instantmimic.hpo.common import ensure_list, resolve_repo_path
from instantmimic.hpo.controller_state import objective_artifact_dir
from instantmimic.hpo.evidence import aggregate_values
from instantmimic.hpo.experiment import NodeId, load_allowed_paths
from instantmimic.hpo.objective.base import (
    Evidence,
    IncompatibleObjectiveError,
    Verdict,
)
from instantmimic.hpo.objective.pareto import (
    ParetoAxis,
    pareto_frontier,
    pareto_recommendations,
    pareto_representatives,
)
from instantmimic.hpo.objective.selection import (
    ObjectiveAxis,
    ObjectiveSpec,
    select_objective,
)

FRONTIER_STRATEGIES = ("pareto", "learning_curve_pareto", "axes_pareto")


@dataclass(frozen=True)
class ParetoObjectiveConfig:
    time_key: str
    quality_key: str
    quality_goal: str
    quality_aggregate: str
    time_epsilon: float
    quality_epsilon: float
    selection: str
    guards: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LearningCurveParetoConfig:
    quality_key: str
    quality_goal: str
    quality_aggregate: str
    quality_statistic: str
    failure_censor_success_rate_key: str | None
    failure_censor_failure_value: float | None
    budget_key: str
    budget_value: float
    terminal_budget_tolerance: float
    terminal_budget_policy: str
    final_epsilon: float
    anytime_epsilon: float
    selection: str
    anytime_quality_floor: float | None
    guards: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AxesParetoAxisConfig:
    key: str
    goal: str
    epsilon: float
    value_type: str
    metric: str
    window: int
    seed_reduce: str
    failure_censor_success_rate_metric: str | None
    failure_censor_failure_value: float | None

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "goal": self.goal,
            "epsilon": self.epsilon,
            "value": {
                "type": self.value_type,
                "metric": self.metric,
                "window": self.window,
                "seed_reduce": self.seed_reduce,
            },
        }
        if self.failure_censor_success_rate_metric is not None:
            payload["failure_censor"] = {
                "success_rate_metric": self.failure_censor_success_rate_metric,
                "failure_value": self.failure_censor_failure_value,
            }
        return payload


@dataclass(frozen=True)
class AxesParetoConfig:
    axes: tuple[AxesParetoAxisConfig, ...]
    budget_key: str
    budget_value: float
    terminal_budget_tolerance: float
    selection: str
    guards: tuple[dict[str, Any], ...]


def _objective_strategy(task_cfg: Any) -> str:
    objective_cfg = task_cfg.get("objective", None)
    if objective_cfg is None:
        raise ValueError("Task is missing required 'objective' section.")
    raw = str(objective_cfg.get("strategy", "weighted_sum") or "").strip().lower()
    return raw or "weighted_sum"


def _motion_configs_locked(task_cfg: Any) -> bool:
    batch_cfg = task_cfg.get("batch", None)
    if batch_cfg is None:
        return True
    return bool(batch_cfg.get("lock_motion_configs", True))


def _eval_source(task_cfg: Any) -> str:
    batch_cfg = task_cfg.get("batch", None)
    if batch_cfg is None:
        return "playback"
    return str(batch_cfg.get("eval_source", "playback")).strip().lower()


def _validate_pareto_batch_contract(task_cfg: DictConfig) -> None:
    if _eval_source(task_cfg) != "train_benchmark":
        raise ValueError("Pareto objective requires batch.eval_source=train_benchmark.")
    seeds = [
        int(item) for item in ensure_list(task_cfg.batch.get("seeds"), "batch.seeds")
    ]
    if not seeds:
        raise ValueError("Pareto objective requires at least one seed.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Pareto objective requires distinct seeds.")
    if not _motion_configs_locked(task_cfg):
        raise ValueError("Pareto objective requires batch.lock_motion_configs=true.")
    if not load_allowed_paths(task_cfg):
        raise ValueError("Pareto objective requires a non-empty parameter_space list.")


def _pareto_objective_config(task_cfg: DictConfig) -> ParetoObjectiveConfig:
    if _objective_strategy(task_cfg) != "pareto":
        raise ValueError("Pareto configuration requires objective.strategy=pareto.")
    _validate_pareto_batch_contract(task_cfg)
    objective_cfg = task_cfg.objective
    time_cfg = objective_cfg.get("time", None)
    quality_cfg = objective_cfg.get("quality", None)
    if not isinstance(time_cfg, (dict, DictConfig)):
        raise ValueError("objective.time must be a mapping for Pareto objectives.")
    if not isinstance(quality_cfg, (dict, DictConfig)):
        raise ValueError("objective.quality must be a mapping for Pareto objectives.")

    time_key = str(time_cfg.get("key", "active_train_time")).strip()
    time_aggregate = str(time_cfg.get("aggregate", "median")).strip().lower()
    quality_key = str(quality_cfg.get("key", "")).strip()
    quality_goal = str(quality_cfg.get("goal", "maximize")).strip().lower()
    quality_aggregate = str(quality_cfg.get("aggregate", "mean")).strip().lower()
    selection = str(objective_cfg.get("selection", "knee")).strip().lower()
    if not time_key or not quality_key:
        raise ValueError("Pareto time.key and quality.key must be non-empty.")
    if time_aggregate != "median":
        raise ValueError("Pareto objective.time.aggregate must be 'median'.")
    if quality_goal not in {"maximize", "minimize"}:
        raise ValueError("Pareto objective.quality.goal must be maximize or minimize.")
    if quality_aggregate not in {"mean", "median"}:
        raise ValueError("Pareto objective.quality.aggregate must be mean or median.")
    if selection not in {"fastest", "quality", "knee"}:
        raise ValueError(
            "Pareto objective.selection must be fastest, quality, or knee."
        )

    time_epsilon = float(time_cfg.get("epsilon", 0.0))
    quality_epsilon = float(quality_cfg.get("epsilon", 0.0))
    if not math.isfinite(time_epsilon) or time_epsilon < 0.0:
        raise ValueError(
            "Pareto objective.time.epsilon must be finite and non-negative."
        )
    if not math.isfinite(quality_epsilon) or quality_epsilon < 0.0:
        raise ValueError(
            "Pareto objective.quality.epsilon must be finite and non-negative."
        )

    guards: list[dict[str, Any]] = []
    for item in ensure_list(objective_cfg.get("guards"), "objective.guards"):
        if not isinstance(item, (dict, DictConfig)):
            raise ValueError("Each objective.guards entry must be a mapping.")
        key = str(item.get("key", "")).strip()
        aggregate = str(item.get("aggregate", "mean")).strip().lower()
        minimum = item.get("min", None)
        maximum = item.get("max", None)
        if not key or aggregate not in {"mean", "median"}:
            raise ValueError("Each Pareto guard needs key and mean/median aggregate.")
        if minimum is None and maximum is None:
            raise ValueError("Each Pareto guard needs min and/or max.")
        guards.append(
            {
                "key": key,
                "aggregate": aggregate,
                "min": None if minimum is None else float(minimum),
                "max": None if maximum is None else float(maximum),
            }
        )
    return ParetoObjectiveConfig(
        time_key=time_key,
        quality_key=quality_key,
        quality_goal=quality_goal,
        quality_aggregate=quality_aggregate,
        time_epsilon=time_epsilon,
        quality_epsilon=quality_epsilon,
        selection=selection,
        guards=tuple(guards),
    )


def _learning_curve_pareto_config(
    task_cfg: DictConfig,
) -> LearningCurveParetoConfig:
    if _objective_strategy(task_cfg) != "learning_curve_pareto":
        raise ValueError(
            "Learning-curve Pareto configuration requires "
            "objective.strategy=learning_curve_pareto."
        )
    _validate_pareto_batch_contract(task_cfg)
    objective_cfg = task_cfg.objective
    final_cfg = objective_cfg.get("final_quality", None)
    anytime_cfg = objective_cfg.get("anytime_quality", None)
    if not isinstance(final_cfg, (dict, DictConfig)):
        raise ValueError("objective.final_quality must be a mapping.")
    if not isinstance(anytime_cfg, (dict, DictConfig)):
        raise ValueError("objective.anytime_quality must be a mapping.")
    quality_key = str(final_cfg.get("key", "")).strip()
    quality_goal = str(final_cfg.get("goal", "maximize")).strip().lower()
    quality_aggregate = str(final_cfg.get("aggregate", "mean")).strip().lower()
    quality_statistic = str(final_cfg.get("statistic", "terminal")).strip().lower()
    selection = str(objective_cfg.get("selection", "balanced")).strip().lower()
    terminal_budget_policy = (
        str(objective_cfg.get("terminal_budget_policy", "strict")).strip().lower()
    )
    if terminal_budget_policy not in {"strict", "hold_last_valid"}:
        raise ValueError(
            "objective.terminal_budget_policy must be strict or hold_last_valid."
        )
    active_time_budget_raw = objective_cfg.get("active_time_budget_seconds", None)
    frame_budget_raw = objective_cfg.get("frame_budget", None)
    if active_time_budget_raw not in (None, "") and frame_budget_raw not in (
        None,
        "",
    ):
        raise ValueError(
            "Learning-curve Pareto objectives must define exactly one of "
            "objective.active_time_budget_seconds or objective.frame_budget."
        )
    if active_time_budget_raw not in (None, ""):
        budget_key = "active_train_time"
        budget_value = float(active_time_budget_raw)
        terminal_budget_tolerance = float(
            objective_cfg.get("terminal_time_tolerance_seconds", 0.0)
        )
        budget_field = "active_time_budget_seconds"
        tolerance_field = "terminal_time_tolerance_seconds"
    elif frame_budget_raw not in (None, ""):
        budget_key = "frames"
        budget_value = float(frame_budget_raw)
        terminal_budget_tolerance = float(
            objective_cfg.get("terminal_frame_tolerance", 0.0)
        )
        budget_field = "frame_budget"
        tolerance_field = "terminal_frame_tolerance"
    else:
        raise ValueError(
            "Learning-curve Pareto objectives require one of "
            "objective.active_time_budget_seconds or objective.frame_budget."
        )
    if not math.isfinite(budget_value) or budget_value <= 0:
        raise ValueError(f"objective.{budget_field} must be finite and positive.")
    if not math.isfinite(terminal_budget_tolerance) or terminal_budget_tolerance < 0:
        raise ValueError(
            f"objective.{tolerance_field} must be finite and non-negative."
        )
    if not quality_key:
        raise ValueError("objective.final_quality.key must be non-empty.")
    if quality_goal not in {"maximize", "minimize"}:
        raise ValueError("objective.final_quality.goal must be maximize or minimize.")
    if quality_aggregate not in {"mean", "median"}:
        raise ValueError("objective.final_quality.aggregate must be mean or median.")
    if quality_statistic not in {"terminal", "best_per_seed"}:
        raise ValueError(
            "objective.final_quality.statistic must be terminal or best_per_seed."
        )
    if selection not in {
        "balanced",
        "final_quality",
        "anytime_quality",
        "final_quality_constrained",
    }:
        raise ValueError(
            "Learning-curve Pareto selection must be balanced, final_quality, "
            "anytime_quality, or final_quality_constrained."
        )
    floor_raw = objective_cfg.get("anytime_quality_floor", None)
    floor = None if floor_raw in (None, "") else float(floor_raw)
    if floor is not None and quality_goal != "maximize":
        raise ValueError(
            "objective.anytime_quality_floor is only supported for maximization "
            "objectives."
        )
    if selection == "final_quality_constrained" and floor is None:
        raise ValueError(
            "final_quality_constrained selection requires anytime_quality_floor."
        )
    final_epsilon = float(final_cfg.get("epsilon", 0.0))
    anytime_epsilon = float(anytime_cfg.get("epsilon", 0.0))
    for label, value in (
        ("final_quality", final_epsilon),
        ("anytime_quality", anytime_epsilon),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"objective.{label}.epsilon must be finite and non-negative."
            )

    failure_censor_cfg = final_cfg.get("failure_censor", None)
    failure_censor_success_rate_key: str | None = None
    failure_censor_failure_value: float | None = None
    if failure_censor_cfg is not None:
        if not isinstance(failure_censor_cfg, (dict, DictConfig)):
            raise ValueError(
                "objective.final_quality.failure_censor must be a mapping."
            )
        failure_censor_success_rate_key = str(
            failure_censor_cfg.get("success_rate_key", "")
        ).strip()
        failure_censor_failure_value = float(failure_censor_cfg.get("failure_value"))
        if not failure_censor_success_rate_key:
            raise ValueError(
                "objective.final_quality.failure_censor.success_rate_key must "
                "be non-empty."
            )
        if not math.isfinite(failure_censor_failure_value):
            raise ValueError(
                "objective.final_quality.failure_censor.failure_value must be finite."
            )

    guards: list[dict[str, Any]] = []
    for item in ensure_list(objective_cfg.get("guards"), "objective.guards"):
        if not isinstance(item, (dict, DictConfig)):
            raise ValueError("Each objective.guards entry must be a mapping.")
        key = str(item.get("key", "")).strip()
        aggregate = str(item.get("aggregate", "mean")).strip().lower()
        minimum = item.get("min", None)
        maximum = item.get("max", None)
        if not key or aggregate not in {"mean", "median"}:
            raise ValueError("Each Pareto guard needs key and mean/median aggregate.")
        if minimum is None and maximum is None:
            raise ValueError("Each Pareto guard needs min and/or max.")
        guards.append(
            {
                "key": key,
                "aggregate": aggregate,
                "min": None if minimum is None else float(minimum),
                "max": None if maximum is None else float(maximum),
            }
        )
    return LearningCurveParetoConfig(
        quality_key=quality_key,
        quality_goal=quality_goal,
        quality_aggregate=quality_aggregate,
        quality_statistic=quality_statistic,
        failure_censor_success_rate_key=failure_censor_success_rate_key,
        failure_censor_failure_value=failure_censor_failure_value,
        budget_key=budget_key,
        budget_value=budget_value,
        terminal_budget_tolerance=terminal_budget_tolerance,
        terminal_budget_policy=terminal_budget_policy,
        final_epsilon=final_epsilon,
        anytime_epsilon=anytime_epsilon,
        selection=selection,
        anytime_quality_floor=floor,
        guards=tuple(guards),
    )


def _axes_pareto_config(task_cfg: DictConfig) -> AxesParetoConfig:
    if _objective_strategy(task_cfg) != "axes_pareto":
        raise ValueError(
            "Axes Pareto configuration requires objective.strategy=axes_pareto."
        )
    _validate_pareto_batch_contract(task_cfg)
    objective_cfg = task_cfg.objective
    raw_axes = ensure_list(objective_cfg.get("axes"), "objective.axes")
    if len(raw_axes) < 2:
        raise ValueError("objective.axes must contain at least two Pareto axes.")
    axes: list[AxesParetoAxisConfig] = []
    for index, raw_axis in enumerate(raw_axes):
        if not isinstance(raw_axis, (dict, DictConfig)):
            raise ValueError(f"objective.axes.{index} must be a mapping.")
        key = str(raw_axis.get("key", "")).strip()
        goal = str(raw_axis.get("goal", "")).strip().lower()
        epsilon = float(raw_axis.get("epsilon", 0.0))
        value_cfg = raw_axis.get("value", None)
        if not key:
            raise ValueError(f"objective.axes.{index}.key must be non-empty.")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
            raise ValueError(
                f"objective.axes.{index}.key must start with a letter and contain "
                "only letters, digits, and underscores."
            )
        if goal not in {"minimize", "maximize"}:
            raise ValueError(
                f"objective.axes.{index}.goal must be minimize or maximize."
            )
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError(
                f"objective.axes.{index}.epsilon must be finite and non-negative."
            )
        if not isinstance(value_cfg, (dict, DictConfig)):
            raise ValueError(f"objective.axes.{index}.value must be a mapping.")
        value_type = str(value_cfg.get("type", "terminal")).strip().lower()
        metric = str(value_cfg.get("metric", "")).strip()
        window = int(value_cfg.get("window", 1))
        seed_reduce = str(value_cfg.get("seed_reduce", "mean")).strip().lower()
        if value_type not in {"terminal", "normalized_auc", "best_per_seed"}:
            raise ValueError(
                f"objective.axes.{index}.value.type must be terminal, "
                "normalized_auc, or best_per_seed."
            )
        if not metric:
            raise ValueError(f"objective.axes.{index}.value.metric must be non-empty.")
        if window < 1:
            raise ValueError(f"objective.axes.{index}.value.window must be >= 1.")
        if seed_reduce not in {"mean", "median"}:
            raise ValueError(
                f"objective.axes.{index}.value.seed_reduce must be mean or median."
            )
        censor_cfg = raw_axis.get("failure_censor", None)
        censor_metric: str | None = None
        censor_value: float | None = None
        if censor_cfg is not None:
            if not isinstance(censor_cfg, (dict, DictConfig)):
                raise ValueError(
                    f"objective.axes.{index}.failure_censor must be a mapping."
                )
            censor_metric = str(censor_cfg.get("success_rate_metric", "")).strip()
            censor_value = float(censor_cfg.get("failure_value"))
            if not censor_metric:
                raise ValueError(
                    f"objective.axes.{index}.failure_censor."
                    "success_rate_metric must be non-empty."
                )
            if not math.isfinite(censor_value):
                raise ValueError(
                    f"objective.axes.{index}.failure_censor.failure_value must "
                    "be finite."
                )
        axes.append(
            AxesParetoAxisConfig(
                key=key,
                goal=goal,
                epsilon=epsilon,
                value_type=value_type,
                metric=metric,
                window=window,
                seed_reduce=seed_reduce,
                failure_censor_success_rate_metric=censor_metric,
                failure_censor_failure_value=censor_value,
            )
        )
    axis_keys = [axis.key for axis in axes]
    if len(set(axis_keys)) != len(axis_keys):
        raise ValueError("objective.axes keys must be unique.")

    active_time_budget = objective_cfg.get("active_time_budget_seconds", None)
    frame_budget = objective_cfg.get("frame_budget", None)
    if (active_time_budget in (None, "")) == (frame_budget in (None, "")):
        raise ValueError(
            "Axes Pareto objectives must define exactly one of "
            "objective.active_time_budget_seconds or objective.frame_budget."
        )
    if frame_budget not in (None, ""):
        budget_key = "frames"
        budget_value = float(frame_budget)
        tolerance = float(objective_cfg.get("terminal_frame_tolerance", 0.0))
    else:
        budget_key = "active_train_time"
        budget_value = float(active_time_budget)
        tolerance = float(objective_cfg.get("terminal_time_tolerance_seconds", 0.0))
    if not math.isfinite(budget_value) or budget_value <= 0.0:
        raise ValueError("Axes Pareto budget must be finite and positive.")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "Axes Pareto terminal budget tolerance must be finite and non-negative."
        )
    selection = str(objective_cfg.get("selection", "balanced")).strip()
    if selection not in {"balanced", *axis_keys}:
        choices = ", ".join(["balanced", *axis_keys])
        raise ValueError(f"objective.selection must be one of: {choices}.")

    guards: list[dict[str, Any]] = []
    for index, item in enumerate(
        ensure_list(objective_cfg.get("guards"), "objective.guards")
    ):
        if not isinstance(item, (dict, DictConfig)):
            raise ValueError(f"objective.guards.{index} must be a mapping.")
        key = str(item.get("key", "")).strip()
        aggregate = str(item.get("aggregate", "mean")).strip().lower()
        minimum = item.get("min", None)
        maximum = item.get("max", None)
        if not key or aggregate not in {"mean", "median"}:
            raise ValueError("Each Pareto guard needs key and mean/median aggregate.")
        if minimum is None and maximum is None:
            raise ValueError("Each Pareto guard needs min and/or max.")
        guards.append(
            {
                "key": key,
                "aggregate": aggregate,
                "min": None if minimum is None else float(minimum),
                "max": None if maximum is None else float(maximum),
            }
        )
    return AxesParetoConfig(
        axes=tuple(axes),
        budget_key=budget_key,
        budget_value=budget_value,
        terminal_budget_tolerance=tolerance,
        selection=selection,
        guards=tuple(guards),
    )


def _reward_threshold_ladder_config(task_cfg: DictConfig) -> tuple[list[float], str]:
    objective_cfg = task_cfg.get("objective", None)
    if objective_cfg is None:
        raise ValueError("Task is missing required 'objective' section.")
    thresholds_raw = ensure_list(
        objective_cfg.get("thresholds"), "objective.thresholds"
    )
    if not thresholds_raw:
        raise ValueError(
            "objective.thresholds must contain at least one value for reward_threshold_ladder."
        )
    thresholds = [float(item) for item in thresholds_raw]
    final_reward_key = str(
        objective_cfg.get("final_reward_key", "mean_benchmark_reward_mean_last")
    ).strip()
    if not final_reward_key:
        raise ValueError("objective.final_reward_key must be non-empty.")
    return thresholds, final_reward_key


def _aggregate_pareto_curve(
    rows: list[dict[str, Any]], task_cfg: DictConfig
) -> tuple[list[dict[str, Any]], list[str]]:
    """Aggregate matching fixed-evaluator points across all required runs."""

    strategy = _objective_strategy(task_cfg)
    if strategy == "learning_curve_pareto":
        learning_config = _learning_curve_pareto_config(task_cfg)
        if learning_config.terminal_budget_policy == "hold_last_valid":
            return _aggregate_budget_limited_curve(rows, task_cfg, learning_config)
        time_key = learning_config.budget_key
        quality_key = learning_config.quality_key
        quality_aggregate = learning_config.quality_aggregate
        guards = learning_config.guards
        failure_censor_success_rate_key = (
            learning_config.failure_censor_success_rate_key
        )
        failure_censor_failure_value = learning_config.failure_censor_failure_value
    else:
        legacy_config = _pareto_objective_config(task_cfg)
        time_key = legacy_config.time_key
        quality_key = legacy_config.quality_key
        quality_aggregate = legacy_config.quality_aggregate
        guards = legacy_config.guards
        failure_censor_success_rate_key = None
        failure_censor_failure_value = None
    if not rows:
        return [], ["no_runs"]
    failed = [row for row in rows if row.get("train_returncode") != 0]
    if failed:
        failed_ids = ",".join(str(row.get("seed", "?")) for row in failed)
        return [], [f"train_failed_for_seed:{failed_ids}"]

    indexed_curves: list[dict[int, dict[str, float]]] = []
    for row in rows:
        curve = row.get("benchmark_curve")
        if not isinstance(curve, list) or not curve:
            return [], [f"missing_benchmark_curve_for_seed:{row.get('seed', '?')}"]
        indexed: dict[int, dict[str, float]] = {}
        for raw_point in curve:
            if not isinstance(raw_point, dict) or raw_point.get("eval_index") is None:
                continue
            raw_index = float(raw_point["eval_index"])
            if not raw_index.is_integer() or raw_index < 1:
                continue
            eval_index = int(raw_index)
            if eval_index in indexed:
                return [], [
                    f"duplicate_eval_index_{eval_index}_for_seed:{row.get('seed', '?')}"
                ]
            indexed[eval_index] = {
                str(key): float(value)
                for key, value in raw_point.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
        if not indexed:
            return [], [f"missing_eval_index_for_seed:{row.get('seed', '?')}"]
        indexed_curves.append(indexed)

    eval_indices = set(indexed_curves[0])
    if any(set(curve) != eval_indices for curve in indexed_curves[1:]):
        return [], ["mismatched_eval_indices"]
    if not eval_indices:
        return [], ["no_common_eval_index"]

    points: list[dict[str, Any]] = []
    rejected: list[str] = []
    required_keys = [time_key]
    if failure_censor_success_rate_key is None:
        required_keys.append(quality_key)
    else:
        required_keys.append(failure_censor_success_rate_key)
    for guard in guards:
        required_keys.append(str(guard["key"]))
    for eval_index in sorted(eval_indices):
        per_run = [curve[eval_index] for curve in indexed_curves]
        missing_keys = sorted(
            {key for point in per_run for key in required_keys if key not in point}
        )
        if missing_keys:
            return [], [
                f"eval_index_{eval_index}_missing_metrics:{','.join(missing_keys)}"
            ]
        times = [float(point[time_key]) for point in per_run]
        raw_qualities: list[float | None] = [
            (float(point[quality_key]) if quality_key in point else None)
            for point in per_run
        ]
        success_rates: list[float] | None = None
        if failure_censor_success_rate_key is None:
            qualities = [float(value) for value in raw_qualities if value is not None]
        else:
            assert failure_censor_failure_value is not None
            success_rates = [
                float(point[failure_censor_success_rate_key]) for point in per_run
            ]
            if any(rate < 0.0 or rate > 1.0 for rate in success_rates):
                return [], [
                    f"eval_index_{eval_index}_invalid_{failure_censor_success_rate_key}"
                ]
            qualities = [
                rate
                * (
                    raw_quality
                    if raw_quality is not None
                    else failure_censor_failure_value
                )
                + (1.0 - rate) * failure_censor_failure_value
                for raw_quality, rate in zip(raw_qualities, success_rates, strict=True)
            ]
        if any(value < 0.0 for value in times):
            return [], [f"eval_index_{eval_index}_negative_{time_key}"]
        guard_values: dict[str, float] = {}
        guard_failures: list[str] = []
        for guard in guards:
            key = str(guard["key"])
            value = aggregate_values(
                [float(point[key]) for point in per_run], str(guard["aggregate"])
            )
            guard_values[key] = value
            if guard["min"] is not None and value < float(guard["min"]):
                guard_failures.append(f"{key}<{guard['min']}")
            if guard["max"] is not None and value > float(guard["max"]):
                guard_failures.append(f"{key}>{guard['max']}")
        if guard_failures:
            rejected.append(
                f"eval_index_{eval_index}_guard_failed:{','.join(guard_failures)}"
            )
            continue

        median_time = aggregate_values(times, "median")
        quality_mean = aggregate_values(qualities, "mean")
        quality_median = aggregate_values(qualities, "median")
        point = {
            "eval_index": eval_index,
            time_key: median_time,
            "quality": aggregate_values(qualities, quality_aggregate),
            "run_count": len(per_run),
            "time_values": times,
            "time_min": min(times),
            "time_max": max(times),
            "time_max_over_median": (
                max(times) / median_time if median_time > 0.0 else None
            ),
            "quality_values": qualities,
            "quality_raw_values": raw_qualities,
            "quality_mean": quality_mean,
            "quality_median": quality_median,
            "quality_min": min(qualities),
            "quality_max": max(qualities),
            "quality_spread": max(qualities) - min(qualities),
            "guards": guard_values,
        }
        if success_rates is not None:
            point["quality_success_rate_values"] = success_rates
        if time_key != "active_train_time" and all(
            "active_train_time" in item for item in per_run
        ):
            active_times = [float(item["active_train_time"]) for item in per_run]
            point["active_train_time"] = aggregate_values(active_times, "median")
            point["active_train_time_values"] = active_times
        points.append(point)
    for row_index in range(len(indexed_curves)):
        run_times = [
            float(indexed_curves[row_index][index][time_key])
            for index in sorted(eval_indices)
        ]
        if any(
            next_time < time
            for time, next_time in zip(run_times, run_times[1:], strict=False)
        ):
            return [], [f"nonmonotonic_{time_key}_for_run:{row_index}"]
    if not points:
        return [], rejected or ["no_valid_pareto_points"]
    return points, []


def _aggregate_budget_limited_curve(
    rows: list[dict[str, Any]],
    task_cfg: DictConfig,
    config: LearningCurveParetoConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resample each seed using only observations made within the fixed budget.

    Iterations may finish beyond the stopping threshold. Those measurements
    establish coverage only: interpolating from them would credit training that
    occurred after the budget. Hold the last eligible observation instead. The
    first observation is also held back to zero, matching the existing AUC rule.
    """
    strict_cfg = OmegaConf.create(OmegaConf.to_container(task_cfg, resolve=True))
    strict_cfg.objective.terminal_budget_policy = "strict"
    # Validate/censor each seed independently before aligning different schedules.
    unguarded_cfg = OmegaConf.create(OmegaConf.to_container(strict_cfg, resolve=True))
    unguarded_cfg.objective.guards = [
        {**guard, "min": -math.inf, "max": math.inf} for guard in config.guards
    ]
    if not rows:
        return [], ["no_runs"]
    failed = [row for row in rows if row.get("train_returncode") != 0]
    if failed:
        failed_ids = ",".join(str(row.get("seed", "?")) for row in failed)
        return [], [f"train_failed_for_seed:{failed_ids}"]
    seed_curves: list[list[dict[str, Any]]] = []
    terminals: list[float] = []
    for row in rows:
        raw_curve = row.get("benchmark_curve")
        if not isinstance(raw_curve, list) or not raw_curve:
            return [], [f"missing_benchmark_curve_for_seed:{row.get('seed', '?')}"]
        indexed_samples: dict[int, tuple[float, dict[str, Any]]] = {}
        for raw_point in raw_curve:
            if not isinstance(raw_point, dict) or raw_point.get("eval_index") is None:
                continue
            raw_index = float(raw_point["eval_index"])
            if not raw_index.is_integer() or raw_index < 1:
                continue
            eval_index = int(raw_index)
            if eval_index in indexed_samples:
                return [], [
                    f"duplicate_eval_index_{eval_index}_for_seed:{row.get('seed', '?')}"
                ]
            raw_time = raw_point.get(config.budget_key)
            if not isinstance(raw_time, (int, float)) or not math.isfinite(
                float(raw_time)
            ):
                return [], [
                    f"eval_index_{eval_index}_missing_metrics:{config.budget_key}"
                ]
            indexed_samples[eval_index] = (float(raw_time), raw_point)
        if not indexed_samples:
            return [], [f"missing_eval_index_for_seed:{row.get('seed', '?')}"]
        ordered_samples = [indexed_samples[index] for index in sorted(indexed_samples)]
        times = [time_value for time_value, _ in ordered_samples]
        if any(time_value < 0.0 for time_value in times):
            return [], [f"negative_{config.budget_key}_for_seed:{row.get('seed', '?')}"]
        if any(
            next_time < time_value
            for time_value, next_time in zip(times, times[1:], strict=False)
        ):
            return [], [f"nonmonotonic_{config.budget_key}_for_run:0"]
        terminal = times[-1]
        if terminal < config.budget_value - config.terminal_budget_tolerance:
            return [], [f"incomplete_budget_for_seed:{row.get('seed', '?')}"]
        eligible_samples = [
            raw_point
            for time_value, raw_point in ordered_samples
            if time_value <= config.budget_value
        ]
        if not eligible_samples:
            return [], [f"no_in_budget_evaluation_for_seed:{row.get('seed', '?')}"]
        validation_row = {**row, "benchmark_curve": eligible_samples}
        curve, reasons = _aggregate_pareto_curve([validation_row], unguarded_cfg)
        if reasons:
            return [], reasons
        terminals.append(terminal)
        for point in curve:
            point.update(point["guards"])
        seed_curves.append(curve)
    times = sorted(
        {config.budget_value}
        | {float(point[config.budget_key]) for curve in seed_curves for point in curve}
    )
    synthetic_rows = []
    source_indices: list[list[int]] = []
    for row, curve in zip(rows, seed_curves, strict=True):
        samples = []
        indices = []
        position = 0
        for index, time_value in enumerate(times, 1):
            while position + 1 < len(curve) and (
                curve[position + 1][config.budget_key] <= time_value
            ):
                position += 1
            left = curve[position]
            right = curve[min(position + 1, len(curve) - 1)]
            width = float(right[config.budget_key] - left[config.budget_key])
            fraction = (
                max(0.0, (time_value - float(left[config.budget_key])) / width)
                if width > 0.0
                else 0.0
            )
            sample = {"eval_index": index, config.budget_key: time_value}
            for key in ["quality", *(str(guard["key"]) for guard in config.guards)]:
                sample[key] = float(left[key]) + fraction * (
                    float(right[key]) - float(left[key])
                )
            samples.append(sample)
            indices.append(int(left["eval_index"]))
        synthetic_rows.append({**row, "benchmark_curve": samples})
        source_indices.append(indices)
    # Quality is already failure-censored; apply cross-seed guards at shared times.
    strict_cfg.objective.final_quality.key = "quality"
    strict_cfg.objective.final_quality.pop("failure_censor", None)
    points, reasons = _aggregate_pareto_curve(synthetic_rows, strict_cfg)
    if reasons:
        return [], reasons
    for point in points:
        position = int(point["eval_index"]) - 1
        point["source_eval_indices"] = [indices[position] for indices in source_indices]
    points[-1]["observed_terminal_budget_values"] = terminals
    points[-1]["seed_anytime_qualities"] = [
        _normalized_learning_curve_auc(
            [
                {**point, "quality": point["quality_values"][seed_index]}
                for point in points
            ],
            budget_key=config.budget_key,
        )
        for seed_index in range(len(seed_curves))
    ]
    return points, []


def _aggregate_axes_pareto_curve(
    rows: list[dict[str, Any]], task_cfg: DictConfig
) -> tuple[list[dict[str, Any]], list[str]]:
    config = _axes_pareto_config(task_cfg)
    if not rows:
        return [], ["no_runs"]
    failed = [row for row in rows if row.get("train_returncode") != 0]
    if failed:
        failed_ids = ",".join(str(row.get("seed", "?")) for row in failed)
        return [], [f"train_failed_for_seed:{failed_ids}"]

    indexed_curves: list[dict[int, dict[str, float]]] = []
    for row in rows:
        curve = row.get("benchmark_curve")
        if not isinstance(curve, list) or not curve:
            return [], [f"missing_benchmark_curve_for_seed:{row.get('seed', '?')}"]
        indexed: dict[int, dict[str, float]] = {}
        for raw_point in curve:
            if not isinstance(raw_point, dict) or raw_point.get("eval_index") is None:
                continue
            raw_index = float(raw_point["eval_index"])
            if not raw_index.is_integer() or raw_index < 1:
                continue
            eval_index = int(raw_index)
            if eval_index in indexed:
                return [], [
                    f"duplicate_eval_index_{eval_index}_for_seed:{row.get('seed', '?')}"
                ]
            indexed[eval_index] = {
                str(key): float(value)
                for key, value in raw_point.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            }
        if not indexed:
            return [], [f"missing_eval_index_for_seed:{row.get('seed', '?')}"]
        indexed_curves.append(indexed)

    eval_indices = set(indexed_curves[0])
    if any(set(curve) != eval_indices for curve in indexed_curves[1:]):
        return [], ["mismatched_eval_indices"]

    points: list[dict[str, Any]] = []
    rejected: list[str] = []
    for eval_index in sorted(eval_indices):
        per_run = [curve[eval_index] for curve in indexed_curves]
        missing: set[str] = set()
        for run_point in per_run:
            if config.budget_key not in run_point:
                missing.add(config.budget_key)
            for axis in config.axes:
                required_key = axis.failure_censor_success_rate_metric or axis.metric
                if required_key not in run_point:
                    missing.add(required_key)
            for guard in config.guards:
                if str(guard["key"]) not in run_point:
                    missing.add(str(guard["key"]))
        if missing:
            return [], [
                f"eval_index_{eval_index}_missing_metrics:{','.join(sorted(missing))}"
            ]

        budgets = [float(point[config.budget_key]) for point in per_run]
        if any(value < 0.0 for value in budgets):
            return [], [f"eval_index_{eval_index}_negative_{config.budget_key}"]
        point: dict[str, Any] = {
            "eval_index": eval_index,
            config.budget_key: aggregate_values(budgets, "median"),
            f"{config.budget_key}_values": budgets,
            "run_count": len(per_run),
        }
        if config.budget_key != "active_train_time" and all(
            "active_train_time" in item for item in per_run
        ):
            active_times = [float(item["active_train_time"]) for item in per_run]
            point["active_train_time"] = aggregate_values(active_times, "median")
            point["active_train_time_values"] = active_times

        for axis in config.axes:
            values: list[float] = []
            raw_values: list[float | None] = []
            success_rates: list[float] | None = None
            if axis.failure_censor_success_rate_metric is not None:
                success_rates = []
            for run_point in per_run:
                raw_value = (
                    float(run_point[axis.metric]) if axis.metric in run_point else None
                )
                raw_values.append(raw_value)
                if axis.failure_censor_success_rate_metric is None:
                    assert raw_value is not None
                    values.append(raw_value)
                    continue
                rate = float(run_point[axis.failure_censor_success_rate_metric])
                if rate < 0.0 or rate > 1.0:
                    return [], [
                        f"eval_index_{eval_index}_invalid_"
                        f"{axis.failure_censor_success_rate_metric}"
                    ]
                assert success_rates is not None
                success_rates.append(rate)
                assert axis.failure_censor_failure_value is not None
                if raw_value is None and rate > 0.0:
                    return [], [
                        f"eval_index_{eval_index}_missing_axis_metric:{axis.key}"
                    ]
                observed = (
                    raw_value
                    if raw_value is not None
                    else axis.failure_censor_failure_value
                )
                values.append(
                    rate * observed + (1.0 - rate) * axis.failure_censor_failure_value
                )
            point[axis.key] = aggregate_values(values, axis.seed_reduce)
            point[f"{axis.key}_values"] = values
            point[f"{axis.key}_raw_values"] = raw_values
            if success_rates is not None:
                point[f"{axis.key}_success_rate_values"] = success_rates

        guard_failures: list[str] = []
        guard_values: dict[str, float] = {}
        for guard in config.guards:
            key = str(guard["key"])
            value = aggregate_values(
                [float(run_point[key]) for run_point in per_run],
                str(guard["aggregate"]),
            )
            guard_values[key] = value
            if guard["min"] is not None and value < float(guard["min"]):
                guard_failures.append(f"{key}<{guard['min']}")
            if guard["max"] is not None and value > float(guard["max"]):
                guard_failures.append(f"{key}>{guard['max']}")
        if guard_failures:
            rejected.append(
                f"eval_index_{eval_index}_guard_failed:{','.join(guard_failures)}"
            )
            continue
        point["guards"] = guard_values
        points.append(point)

    for row_index, curve in enumerate(indexed_curves):
        run_budgets = [
            float(curve[index][config.budget_key]) for index in sorted(eval_indices)
        ]
        if any(
            next_value < value
            for value, next_value in zip(run_budgets, run_budgets[1:], strict=False)
        ):
            return [], [f"nonmonotonic_{config.budget_key}_for_run:{row_index}"]
    if not points:
        return [], rejected or ["no_valid_pareto_points"]
    return points, []


def _axis_seed_values(
    curve: list[dict[str, Any]],
    axis: AxesParetoAxisConfig,
    *,
    budget_key: str,
) -> list[float]:
    ordered = sorted(curve, key=lambda point: int(point["eval_index"]))
    per_checkpoint = [list(point[f"{axis.key}_values"]) for point in ordered]
    run_count = len(per_checkpoint[0])
    if run_count == 0 or any(len(values) != run_count for values in per_checkpoint):
        raise ValueError(f"Axis {axis.key!r} has inconsistent per-seed values.")
    if axis.value_type == "terminal":
        selected = per_checkpoint[-min(axis.window, len(per_checkpoint)) :]
        return [
            float(sum(values[run_index] for values in selected) / len(selected))
            for run_index in range(run_count)
        ]
    if axis.value_type == "best_per_seed":
        chooser = min if axis.goal == "minimize" else max
        return [
            float(chooser(values[run_index] for values in per_checkpoint))
            for run_index in range(run_count)
        ]

    budgets_per_checkpoint = [list(point[f"{budget_key}_values"]) for point in ordered]
    auc_values: list[float] = []
    for run_index in range(run_count):
        times = [float(values[run_index]) for values in budgets_per_checkpoint]
        qualities = [float(values[run_index]) for values in per_checkpoint]
        terminal_time = times[-1]
        if terminal_time <= 0.0:
            raise ValueError(f"Axis {axis.key!r} has a non-positive terminal budget.")
        area = qualities[0] * times[0]
        for left_index in range(len(times) - 1):
            area += (
                0.5
                * (qualities[left_index] + qualities[left_index + 1])
                * (times[left_index + 1] - times[left_index])
            )
        auc_values.append(float(area / terminal_time))
    return auc_values


def _normalized_learning_curve_auc(
    curve: list[dict[str, Any]], *, budget_key: str = "active_train_time"
) -> float:
    if not curve:
        raise ValueError("Cannot calculate anytime quality from an empty curve.")
    ordered = sorted(curve, key=lambda point: int(point["eval_index"]))
    terminal_time = float(ordered[-1][budget_key])
    if terminal_time <= 0.0:
        raise ValueError(f"Terminal {budget_key} must be positive for AUC.")
    first_time = float(ordered[0][budget_key])
    first_quality = float(ordered[0]["quality"])
    area = first_quality * first_time
    for left, right in zip(ordered, ordered[1:], strict=False):
        left_time = float(left[budget_key])
        right_time = float(right[budget_key])
        left_quality = float(left["quality"])
        right_quality = float(right["quality"])
        area += 0.5 * (left_quality + right_quality) * (right_time - left_time)
    return float(area / terminal_time)


def _validate_terminal_budget(
    experiment_name: str,
    budget_label: str,
    values: list[float],
    budget_value: float,
    tolerance: float,
    *,
    unit: str = "",
) -> None:
    for value in values:
        if not math.isfinite(value) or abs(value - budget_value) > tolerance:
            raise ValueError(
                f"Candidate {experiment_name!r} terminal {budget_label} "
                f"{value:.6f}{unit} is outside the fixed budget "
                f"{budget_value:.6f}{unit} +/- {tolerance:.6f}{unit}. "
                "Every seed must satisfy the budget."
            )


def _build_axes_pareto_summary(
    experiments: list[dict[str, Any]], task_cfg: DictConfig
) -> dict[str, Any]:
    config = _axes_pareto_config(task_cfg)
    points: list[dict[str, Any]] = []
    for experiment in experiments:
        curve = [dict(point) for point in experiment.get("pareto_curve", [])]
        if not curve:
            continue
        ordered = sorted(curve, key=lambda point: int(point["eval_index"]))
        terminal_budget = float(ordered[-1][config.budget_key])
        _validate_terminal_budget(
            str(experiment["name"]),
            config.budget_key,
            [terminal_budget, *ordered[-1][f"{config.budget_key}_values"]],
            config.budget_value,
            config.terminal_budget_tolerance,
        )
        point: dict[str, Any] = {
            "experiment": str(experiment["name"]),
            f"terminal_{config.budget_key}": terminal_budget,
            "eval_count": len(ordered),
            "final_eval_index": int(ordered[-1]["eval_index"]),
            "axis_values": {},
        }
        for axis in config.axes:
            seed_values = _axis_seed_values(
                ordered,
                axis,
                budget_key=config.budget_key,
            )
            point[axis.key] = aggregate_values(seed_values, axis.seed_reduce)
            point["axis_values"][axis.key] = seed_values
        points.append(point)
    objective = select_objective(
        points,
        ObjectiveSpec(
            mode="pareto",
            axes=tuple(
                ObjectiveAxis(
                    key=axis.key,
                    goal=axis.goal,
                    epsilon=axis.epsilon,
                )
                for axis in config.axes
            ),
            selection=config.selection,
        ),
    )
    return {
        "mode": "axes",
        "axes": [axis.metadata() for axis in config.axes],
        "budget": {
            "key": config.budget_key,
            "value": config.budget_value,
            "terminal_tolerance": config.terminal_budget_tolerance,
        },
        "selection": config.selection,
        "points": points,
        "frontier": list(objective["frontier"]),
        "recommendations": dict(objective["representatives"]),
        "selected": objective["selected"],
    }


def _build_pareto_summary(
    experiments: list[dict[str, Any]], task_cfg: DictConfig
) -> dict[str, Any]:
    config = _pareto_objective_config(task_cfg)
    points: list[dict[str, Any]] = []
    for experiment in experiments:
        for point in experiment.get("pareto_curve", []):
            points.append({**dict(point), "experiment": str(experiment["name"])})
    axes = (
        ParetoAxis(
            key="active_train_time", goal="minimize", epsilon=config.time_epsilon
        ),
        ParetoAxis(
            key="quality", goal=config.quality_goal, epsilon=config.quality_epsilon
        ),
    )
    frontier = pareto_frontier(points, axes) if points else []
    representatives = pareto_representatives(
        frontier,
        time_key="active_train_time",
        quality_key="quality",
        quality_goal=config.quality_goal,
    )
    selected = representatives.get(config.selection)
    return {
        "mode": "checkpoint_time_quality",
        "time": {
            "source_key": config.time_key,
            "aggregate": "median",
            "epsilon": config.time_epsilon,
        },
        "quality": {
            "source_key": config.quality_key,
            "goal": config.quality_goal,
            "aggregate": config.quality_aggregate,
            "epsilon": config.quality_epsilon,
        },
        "selection": config.selection,
        "points": points,
        "frontier": frontier,
        "representatives": representatives,
        "selected": selected,
    }


def _build_learning_curve_pareto_summary(
    experiments: list[dict[str, Any]], task_cfg: DictConfig
) -> dict[str, Any]:
    config = _learning_curve_pareto_config(task_cfg)
    points: list[dict[str, Any]] = []
    for experiment in experiments:
        curve = [dict(point) for point in experiment.get("pareto_curve", [])]
        if not curve:
            continue
        ordered = sorted(curve, key=lambda point: int(point["eval_index"]))
        terminal_time = float(ordered[-1][config.budget_key])
        _validate_terminal_budget(
            str(experiment["name"]),
            "active train time"
            if config.budget_key == "active_train_time"
            else config.budget_key,
            [terminal_time, *ordered[-1].get("time_values", [])],
            config.budget_value,
            config.terminal_budget_tolerance,
            unit="s" if config.budget_key == "active_train_time" else "",
        )
        terminal_quality = float(ordered[-1]["quality"])
        quality = terminal_quality
        best_quality_values: list[float] | None = None
        best_eval_indices: list[int] | None = None
        if config.quality_statistic == "best_per_seed":
            raw_quality_values = [point.get("quality_values") for point in ordered]
            if any(not isinstance(values, list) for values in raw_quality_values):
                raise ValueError(
                    f"Candidate {experiment['name']!r} requires per-seed quality "
                    "values for final_quality.statistic=best_per_seed."
                )
            quality_values = [list(values) for values in raw_quality_values]
            run_count = len(quality_values[0])
            if run_count == 0 or any(
                len(values) != run_count for values in quality_values
            ):
                raise ValueError(
                    f"Candidate {experiment['name']!r} has inconsistent per-seed "
                    "quality values."
                )
            best_quality_values = []
            best_eval_indices = []
            for run_index in range(run_count):
                chooser = max if config.quality_goal == "maximize" else min
                best_point = chooser(
                    ordered,
                    key=lambda point: float(point["quality_values"][run_index]),
                )
                best_quality_values.append(
                    float(best_point["quality_values"][run_index])
                )
                best_eval_indices.append(
                    int(best_point["source_eval_indices"][run_index])
                    if "source_eval_indices" in best_point
                    else int(best_point["eval_index"])
                )
            quality = aggregate_values(best_quality_values, config.quality_aggregate)
        point = {
            "experiment": str(experiment["name"]),
            # Keep the established key for archive/consumer compatibility. Its
            # statistic is explicit in the Pareto metadata and on each point.
            "final_quality": quality,
            "quality_statistic": config.quality_statistic,
            "terminal_quality": terminal_quality,
            "anytime_quality": _normalized_learning_curve_auc(
                ordered, budget_key=config.budget_key
            ),
            f"terminal_{config.budget_key}": terminal_time,
            "eval_count": len(ordered),
            "final_eval_index": int(ordered[-1]["eval_index"]),
        }
        if best_quality_values is not None:
            point["best_quality_values"] = best_quality_values
            point["best_eval_indices"] = best_eval_indices
        if config.terminal_budget_policy == "hold_last_valid":
            # Integrate before reducing seeds: median and integration do not commute.
            point["anytime_quality"] = aggregate_values(
                ordered[-1]["seed_anytime_qualities"], config.quality_aggregate
            )
            point["observed_terminal_budget_values"] = ordered[-1][
                "observed_terminal_budget_values"
            ]
        points.append(point)
    objective = select_objective(
        points,
        ObjectiveSpec(
            mode="pareto",
            axes=(
                ObjectiveAxis(
                    key="final_quality",
                    goal=config.quality_goal,
                    epsilon=config.final_epsilon,
                ),
                ObjectiveAxis(
                    key="anytime_quality",
                    goal=config.quality_goal,
                    epsilon=config.anytime_epsilon,
                ),
            ),
            selection=(
                "final_quality"
                if config.selection == "final_quality_constrained"
                else config.selection
            ),
        ),
    )
    frontier = list(objective["frontier"])
    recommendations = dict(objective["representatives"])
    if config.anytime_quality_floor is not None:
        constrained = pareto_recommendations(
            frontier,
            first_key="final_quality",
            second_key="anytime_quality",
            first_goal=config.quality_goal,
            second_goal=config.quality_goal,
            second_constraint=config.anytime_quality_floor,
        ).get("first_constrained")
        if constrained is not None:
            recommendations["final_quality_constrained"] = constrained
    selected = recommendations.get(config.selection)
    summary = {
        "mode": "learning_curve",
        "final_quality": {
            "source_key": config.quality_key,
            "goal": config.quality_goal,
            "aggregate": config.quality_aggregate,
            "statistic": config.quality_statistic,
            "epsilon": config.final_epsilon,
        },
        "anytime_quality": {
            "definition": (
                f"{config.quality_aggregate}_seed_normalized_trapezoidal_auc_over_"
                f"{config.budget_key}"
                if config.terminal_budget_policy == "hold_last_valid"
                else f"normalized_trapezoidal_auc_over_{config.budget_key}"
            ),
            "goal": config.quality_goal,
            "epsilon": config.anytime_epsilon,
            "floor": config.anytime_quality_floor,
        },
        "budget": {
            "key": config.budget_key,
            "value": config.budget_value,
            "terminal_tolerance": config.terminal_budget_tolerance,
            "terminal_policy": config.terminal_budget_policy,
        },
        "selection": config.selection,
        "points": points,
        "frontier": frontier,
        "recommendations": recommendations,
        "selected": selected,
    }
    if config.failure_censor_success_rate_key is not None:
        summary["final_quality"]["failure_censor"] = {
            "success_rate_key": config.failure_censor_success_rate_key,
            "failure_value": config.failure_censor_failure_value,
        }
    if config.budget_key == "active_train_time":
        summary["active_time_budget"] = {
            "seconds": config.budget_value,
            "terminal_tolerance_seconds": config.terminal_budget_tolerance,
        }
    return summary


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _json_load(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_repo_path(path: str | Path) -> Path:
    return resolve_repo_path(path, REPO_ROOT)


def _summary_json_path(evaluation_root: Path) -> Path:
    return evaluation_root / "summary.json"


REPO_ROOT = Path(__file__).resolve().parents[4]


def _selection_excerpt(selection: dict[str, Any], aggregate: dict[str, Any]) -> str:
    representative = selection.get("representative")
    if isinstance(representative, dict):
        axis_values = representative.get("axis_values")
        if isinstance(axis_values, dict):
            return " ".join(f"{key}={representative.get(key)}" for key in axis_values)
        if "final_quality" in representative:
            return (
                f"final_quality={representative.get('final_quality')} "
                f"anytime_quality={representative.get('anytime_quality')}"
            )
        return (
            f"pareto_time={representative.get('active_train_time')} "
            f"pareto_quality={representative.get('quality')}"
        )
    active_threshold = selection.get("active_threshold")
    active_count = selection.get("active_threshold_reached_count")
    active_frames = selection.get("active_threshold_frames_median")
    if active_threshold is not None:
        return (
            f"threshold>={active_threshold} count={active_count} "
            "median_frames="
            f"{active_frames} final_reward="
            f"{aggregate.get('mean_benchmark_reward_mean_last')}"
        )
    return (
        "final_reward="
        f"{aggregate.get('mean_benchmark_reward_mean_last', aggregate.get('mean_reward'))}"
    )


def _update_best_result(
    state_dir: Path,
    status: dict[str, Any],
    cycle_index: int,
    summary_payload: dict[str, Any],
) -> None:
    if _update_pareto_archive(state_dir, status, cycle_index, summary_payload):
        return
    experiments = summary_payload.get("experiments", [])
    valid = [item for item in experiments if item.get("selection", {}).get("valid")]
    if not valid:
        return
    best = max(
        valid,
        key=lambda item: (
            float(item["selection"]["score"]),
            *item["selection"].get("tiebreak_values", []),
        ),
    )
    current_best_score = status.get("best_score")
    next_score = best["selection"]["score"]
    if current_best_score is None or float(next_score) > float(current_best_score):
        status["best_cycle"] = cycle_index
        status["best_experiment"] = str(best["name"])
        status["best_score"] = float(next_score)
        status["best_summary_path"] = str(
            _summary_json_path(_resolve_repo_path(summary_payload["output_root"]))
        )
        aggregate = best.get("aggregate", {})
        selection = best.get("selection", {})
        lines = [
            "# Best Result",
            "",
            f"- Cycle: `{cycle_index}`",
            f"- Experiment: `{best['name']}`",
            f"- Score: `{next_score}`",
            f"- Selection: `{_selection_excerpt(selection, aggregate)}`",
            f"- Mean reward: `{aggregate.get('mean_reward')}`",
            f"- Mean APE: `{aggregate.get('mean_ape')}`",
            f"- Mean EE: `{aggregate.get('mean_ee')}`",
            f"- Total falls: `{aggregate.get('total_falls')}`",
            f"- Summary: `{summary_payload.get('output_root')}`",
        ]
        status["best_result_text"] = "\n".join(lines) + "\n"


def _update_pareto_archive(
    state_dir: Path,
    status: dict[str, Any],
    cycle_index: int,
    summary_payload: dict[str, Any],
) -> bool:
    pareto = summary_payload.get("pareto")
    if not isinstance(pareto, dict):
        return False
    if pareto.get("mode") == "axes":
        return _update_axes_pareto_archive(
            state_dir, status, cycle_index, summary_payload, pareto
        )
    if pareto.get("mode") == "learning_curve":
        return _update_learning_curve_pareto_archive(
            state_dir, status, cycle_index, summary_payload, pareto
        )

    time_config = dict(pareto.get("time") or {})
    quality_config = dict(pareto.get("quality") or {})
    selection = str(pareto.get("selection", "knee"))
    if selection not in {"fastest", "quality", "knee"}:
        raise ValueError(f"Unsupported Pareto selection policy: {selection}")
    quality_goal = str(quality_config.get("goal", "maximize"))
    axes = (
        ParetoAxis(
            key="active_train_time",
            goal="minimize",
            epsilon=float(time_config.get("epsilon", 0.0)),
        ),
        ParetoAxis(
            key="quality",
            goal=quality_goal,
            epsilon=float(quality_config.get("epsilon", 0.0)),
        ),
    )

    archive_path = objective_artifact_dir(state_dir) / "pareto_frontier.json"
    existing = _json_load(archive_path)
    prior_points: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        if (
            existing.get("time") != time_config
            or existing.get("quality") != quality_config
            or existing.get("selection") != selection
        ):
            raise IncompatibleObjectiveError(
                "Pareto objective changed while resuming; use a fresh hpo state."
            )
        prior_points = [dict(point) for point in existing.get("frontier", [])]

    summary_path = str(
        _summary_json_path(_resolve_repo_path(summary_payload["output_root"]))
    )
    current_points = [
        {
            **dict(point),
            "cycle": cycle_index,
            "summary_path": summary_path,
        }
        for point in pareto.get("frontier", [])
    ]
    frontier = pareto_frontier([*prior_points, *current_points], axes)
    representatives = pareto_representatives(
        frontier,
        time_key="active_train_time",
        quality_key="quality",
        quality_goal=quality_goal,
    )
    selected = representatives.get(selection)
    archive = {
        "time": time_config,
        "quality": quality_config,
        "selection": selection,
        "frontier": frontier,
        "representatives": representatives,
        "selected": selected,
    }
    _json_dump(archive_path, archive)
    if selected is None:
        return True

    status["best_cycle"] = selected.get("cycle")
    status["best_experiment"] = str(selected.get("experiment", ""))
    status["best_score"] = None
    status["best_summary_path"] = str(selected.get("summary_path", ""))
    status["pareto_archive_path"] = str(archive_path)
    lines = [
        "# Pareto Result",
        "",
        "The frontier is authoritative; the named points are reporting views.",
        "",
        f"- Selection view: `{selection}`",
        f"- Selected experiment: `{selected.get('experiment', '')}`",
        f"- Active train time: `{selected.get('active_train_time')}`",
        f"- Fixed-evaluator quality: `{selected.get('quality')}`",
        f"- Archive: `{archive_path}`",
        "",
        "## Pareto frontier",
        "",
    ]
    for point in frontier:
        lines.append(
            f"- cycle={point.get('cycle')} experiment={point.get('experiment')} "
            f"eval_index={point.get('eval_index')} "
            f"time={point.get('active_train_time')} quality={point.get('quality')}"
        )
    status["best_result_text"] = "\n".join(lines) + "\n"
    return True


def _update_axes_pareto_archive(
    state_dir: Path,
    status: dict[str, Any],
    cycle_index: int,
    summary_payload: dict[str, Any],
    pareto: dict[str, Any],
) -> bool:
    axes_metadata = [dict(axis) for axis in pareto.get("axes", [])]
    if len(axes_metadata) < 2:
        raise ValueError("Axes Pareto summary must contain at least two axes.")
    selection = str(pareto.get("selection", "balanced"))
    objective_spec = ObjectiveSpec(
        mode="pareto",
        axes=tuple(
            ObjectiveAxis(
                key=str(axis["key"]),
                goal=str(axis["goal"]),
                epsilon=float(axis.get("epsilon", 0.0)),
            )
            for axis in axes_metadata
        ),
        selection=selection,
    )
    budget = dict(pareto.get("budget") or {})
    objective_dir = objective_artifact_dir(state_dir)
    archive_path = objective_dir / "pareto_frontier.json"
    existing = _json_load(archive_path)
    prior_points: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        if existing.get("mode") != "axes":
            raise IncompatibleObjectiveError(
                "Pareto mode changed while resuming; use a fresh hpo state."
            )
        if (
            list(existing.get("axes") or []) != axes_metadata
            or dict(existing.get("budget") or {}) != budget
            or existing.get("selection") != selection
        ):
            raise IncompatibleObjectiveError(
                "Pareto objective changed while resuming; use a fresh hpo state."
            )
        prior_points = [
            dict(point)
            for point in existing.get("points", existing.get("frontier", []))
        ]

    summary_path = str(
        _summary_json_path(_resolve_repo_path(summary_payload["output_root"]))
    )
    current_points = [
        {
            **dict(point),
            "cycle": cycle_index,
            "summary_path": summary_path,
        }
        for point in pareto.get("points", pareto.get("frontier", []))
    ]
    all_points = [*prior_points, *current_points]
    objective = select_objective(all_points, objective_spec)
    frontier = list(objective["frontier"])
    recommendations = dict(objective["representatives"])
    selected = objective["selected"]
    anchor_roles = ("balanced", *(axis.key for axis in objective_spec.axes))
    anchor_role = anchor_roles[(max(1, cycle_index) - 1) % len(anchor_roles)]
    anchor_point = recommendations.get(anchor_role) or selected
    active_anchor = (
        None
        if anchor_point is None
        else {"role": anchor_role, "point": dict(anchor_point)}
    )
    archive = {
        "mode": "axes",
        "axes": axes_metadata,
        "budget": budget,
        "selection": selection,
        "points": all_points,
        "frontier": frontier,
        "recommendations": recommendations,
        "selected": selected,
        "active_anchor": active_anchor,
    }
    _json_dump(archive_path, archive)
    status["pareto_archive_path"] = str(archive_path)
    status["recommendations"] = recommendations
    status["active_anchor"] = active_anchor

    axis_keys = [axis.key for axis in objective_spec.axes]
    lines = [
        "# Pareto Frontier",
        "",
        "The frontier is authoritative. Recommendations and the active anchor are views.",
        "",
    ]
    for point in frontier:
        metrics = " ".join(f"{key}={point.get(key)}" for key in axis_keys)
        lines.append(
            f"- cycle={point.get('cycle')} experiment={point.get('experiment')} "
            f"{metrics}"
        )
    lines.extend(["", "## Recommendations", ""])
    for role, point in recommendations.items():
        if point is not None:
            lines.append(f"- {role}: `{point.get('experiment', '')}`")
    if active_anchor is not None:
        lines.extend(
            [
                "",
                "## Active anchor",
                "",
                f"- Role: `{active_anchor['role']}`",
                f"- Experiment: `{active_anchor['point'].get('experiment', '')}`",
            ]
        )
    (objective_dir / "pareto_frontier.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    if selected is not None:
        status["best_cycle"] = selected.get("cycle")
        status["best_experiment"] = str(selected.get("experiment", ""))
        status["best_score"] = None
        status["best_summary_path"] = str(selected.get("summary_path", ""))
        selected_metrics = [f"- {key}: `{selected.get(key)}`" for key in axis_keys]
        best_lines = [
            "# Best Result",
            "",
            "Compatibility projection of the selected Pareto recommendation.",
            "The authoritative result is `objective/pareto_frontier.md`.",
            "",
            f"- Selection policy: `{selection}`",
            f"- Experiment: `{selected.get('experiment', '')}`",
            *selected_metrics,
            f"- Summary: `{selected.get('summary_path', '')}`",
        ]
        status["best_result_text"] = "\n".join(best_lines) + "\n"
    return True


def _update_learning_curve_pareto_archive(
    state_dir: Path,
    status: dict[str, Any],
    cycle_index: int,
    summary_payload: dict[str, Any],
    pareto: dict[str, Any],
) -> bool:
    final_config = dict(pareto.get("final_quality") or {})
    anytime_config = dict(pareto.get("anytime_quality") or {})
    budget = dict(pareto.get("budget") or {})
    active_time_budget = dict(pareto.get("active_time_budget") or {})
    selection = str(pareto.get("selection", "balanced"))
    quality_goal = str(final_config.get("goal", "maximize"))
    floor_raw = anytime_config.get("floor")
    floor = None if floor_raw in (None, "") else float(floor_raw)
    objective_spec = ObjectiveSpec(
        mode="pareto",
        axes=(
            ObjectiveAxis(
                key="final_quality",
                goal=quality_goal,
                epsilon=float(final_config.get("epsilon", 0.0)),
            ),
            ObjectiveAxis(
                key="anytime_quality",
                goal=quality_goal,
                epsilon=float(anytime_config.get("epsilon", 0.0)),
            ),
        ),
        selection=(
            "final_quality" if selection == "final_quality_constrained" else selection
        ),
    )

    objective_dir = objective_artifact_dir(state_dir)
    archive_path = objective_dir / "pareto_frontier.json"
    existing = _json_load(archive_path)
    prior_points: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        if existing.get("mode") != "learning_curve":
            raise IncompatibleObjectiveError(
                "Pareto mode changed while resuming; use a fresh hpo state."
            )
        if (
            existing.get("final_quality") != final_config
            or existing.get("anytime_quality") != anytime_config
            or dict(existing.get("budget") or {}) != budget
            or dict(existing.get("active_time_budget") or {}) != active_time_budget
            or existing.get("selection") != selection
        ):
            raise IncompatibleObjectiveError(
                "Pareto objective changed while resuming; use a fresh hpo state."
            )
        prior_points = [dict(point) for point in existing.get("frontier", [])]

    summary_path = str(
        _summary_json_path(_resolve_repo_path(summary_payload["output_root"]))
    )
    current_points = [
        {
            **dict(point),
            "cycle": cycle_index,
            "summary_path": summary_path,
        }
        for point in pareto.get("frontier", [])
    ]
    objective = select_objective([*prior_points, *current_points], objective_spec)
    frontier = list(objective["frontier"])
    recommendations = dict(objective["representatives"])
    if floor is not None:
        constrained = pareto_recommendations(
            frontier,
            first_key="final_quality",
            second_key="anytime_quality",
            first_goal=quality_goal,
            second_goal=quality_goal,
            second_constraint=floor,
        ).get("first_constrained")
        if constrained is not None:
            recommendations["final_quality_constrained"] = constrained
    selected = recommendations.get(selection)
    anchor_roles = ("balanced", "final_quality", "anytime_quality")
    anchor_role = anchor_roles[(max(1, cycle_index) - 1) % len(anchor_roles)]
    anchor_point = recommendations.get(anchor_role) or selected
    active_anchor = (
        None
        if anchor_point is None
        else {"role": anchor_role, "point": dict(anchor_point)}
    )
    archive = {
        "mode": "learning_curve",
        "final_quality": final_config,
        "anytime_quality": anytime_config,
        "budget": budget,
        "active_time_budget": active_time_budget,
        "selection": selection,
        "frontier": frontier,
        "recommendations": recommendations,
        "selected": selected,
        "active_anchor": active_anchor,
    }
    _json_dump(archive_path, archive)
    status["pareto_archive_path"] = str(archive_path)
    status["recommendations"] = recommendations
    status["active_anchor"] = active_anchor

    lines = [
        "# Pareto Frontier",
        "",
        "The frontier is authoritative. Recommendations and the active anchor are views.",
        "",
    ]
    for point in frontier:
        lines.append(
            f"- cycle={point.get('cycle')} experiment={point.get('experiment')} "
            f"final_quality={point.get('final_quality')} "
            f"anytime_quality={point.get('anytime_quality')}"
        )
    lines.extend(["", "## Recommendations", ""])
    for role, point in recommendations.items():
        if point is None:
            continue
        lines.append(f"- {role}: `{point.get('experiment', '')}`")
    if active_anchor is not None:
        lines.extend(
            [
                "",
                "## Active anchor",
                "",
                f"- Role: `{active_anchor['role']}`",
                f"- Experiment: `{active_anchor['point'].get('experiment', '')}`",
            ]
        )
    frontier_text = "\n".join(lines) + "\n"
    (objective_dir / "pareto_frontier.md").write_text(frontier_text, encoding="utf-8")

    if selected is not None:
        status["best_cycle"] = selected.get("cycle")
        status["best_experiment"] = str(selected.get("experiment", ""))
        status["best_score"] = None
        status["best_summary_path"] = str(selected.get("summary_path", ""))
        compatibility_lines = [
            "# Best Result",
            "",
            "Compatibility projection of the selected Pareto recommendation.",
            "The authoritative result is `objective/pareto_frontier.md`.",
            "",
            f"- Selection policy: `{selection}`",
            f"- Experiment: `{selected.get('experiment', '')}`",
            f"- Final quality: `{selected.get('final_quality')}`",
            f"- Anytime quality: `{selected.get('anytime_quality')}`",
            f"- Summary: `{selected.get('summary_path', '')}`",
        ]
        status["best_result_text"] = "\n".join(compatibility_lines) + "\n"
    return True


def representative_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    pareto = payload.get("pareto")
    if not isinstance(pareto, dict):
        return {}
    selected = pareto.get("selected")
    if isinstance(selected, dict):
        return selected
    points = pareto.get("points")
    if isinstance(points, list) and points and isinstance(points[0], dict):
        return points[0]
    return {}


@dataclass(frozen=True)
class FrontierObjective:
    """Which frontier shape this task compares with."""

    strategy: str = "pareto"

    @classmethod
    def from_config(cls, objective_cfg: Mapping[str, Any] | None) -> FrontierObjective:
        cfg = objective_cfg or {}
        strategy = str(cfg.get("strategy", "pareto") or "pareto")
        if strategy not in FRONTIER_STRATEGIES:
            raise ValueError(
                f"objective.strategy '{strategy}' is not a frontier strategy; "
                f"expected one of: {', '.join(FRONTIER_STRATEGIES)}."
            )
        return cls(strategy=strategy)


class FrontierComparator:
    """Keep the non-dominated set, and say which point to branch from next."""

    unit_label = "axis units"

    def __init__(
        self,
        objective: FrontierObjective,
        *,
        state_dir: Path | None = None,
        task_cfg: DictConfig | None = None,
    ) -> None:
        self.objective = objective
        self.state_dir = state_dir
        # The axis definitions live under `objective`, but the aggregations
        # that build points from rows also read `batch`, so the comparator
        # keeps the task it was built from.
        self.task_cfg = task_cfg
        self.status: dict[str, Any] = {}

    @property
    def objective_id(self) -> str:
        return self.objective.strategy

    # -- state ---------------------------------------------------------------

    def load_state(
        self, objective_dir: Path, status_objective_state: Mapping[str, Any]
    ) -> None:
        self.state_dir = objective_dir.parent
        self.status = dict(status_objective_state)

    # The archive's shape is its strategy's fingerprint. `pareto` stores no
    # mode key; the other two name themselves.
    _ARCHIVE_MODES = {"axes_pareto": "axes", "learning_curve_pareto": "learning_curve"}

    def check_compatible(self) -> None:
        """Refuse a resume whose archive was built by a different strategy.

        The full comparison -- axes, epsilons, budgets, selection policy --
        needs values derived from a cycle's evidence, so it stays on insertion.
        The strategy itself needs no evidence, and it is the change a resume is
        most likely to make, so catching it here costs a run nothing instead of
        one full cycle of training.
        """

        if self.state_dir is None:
            return
        archive = _json_load(
            objective_artifact_dir(self.state_dir) / "pareto_frontier.json"
        )
        if not isinstance(archive, dict) or not archive:
            return
        stored = archive.get("mode", None)
        expected = self._ARCHIVE_MODES.get(self.objective.strategy, None)
        if stored != expected:
            raise IncompatibleObjectiveError(
                f"this run's frontier was built by a different objective "
                f"(archive mode {stored!r}, task strategy "
                f"'{self.objective.strategy}'); use a fresh hpo state."
            )

    def dump_state(self, objective_dir: Path) -> Mapping[str, Any]:
        del objective_dir
        return dict(self.status)

    def recommend_bases(self) -> Sequence[NodeId]:
        """The anchor first, then the other representatives."""

        bases: list[NodeId] = []
        anchor = self.status.get("active_anchor")
        if isinstance(anchor, dict):
            point = anchor.get("point")
            if isinstance(point, dict):
                bases.append(_node_id(point))
        for point in (self.status.get("recommendations") or {}).values():
            if isinstance(point, dict):
                node = _node_id(point)
                if node not in bases:
                    bases.append(node)
        return tuple(bases)

    # -- comparison ----------------------------------------------------------

    def _points_from(self, evidence: Evidence) -> dict[str, Any]:
        """Turn per-seed rows into this objective's frontier points.

        The evaluator records what every trial measured and stops there. Which
        axes those numbers form, how seeds are reduced onto them, and which
        runs count as complete are all comparison decisions, so they happen
        here.
        """

        summary = dict(evidence.summary or {})
        if not self.task_cfg:
            return summary
        experiments = [dict(item) for item in summary.get("experiments", [])]
        if not experiments:
            return summary
        strategy = self.objective.strategy
        for experiment in experiments:
            rows = list(experiment.get("runs", []))
            curve, invalid = (
                _aggregate_axes_pareto_curve(rows, self.task_cfg)
                if strategy == "axes_pareto"
                else _aggregate_pareto_curve(rows, self.task_cfg)
            )
            experiment["pareto_curve"] = curve
            experiment["pareto_invalid_reasons"] = invalid
        summary["experiments"] = experiments
        if strategy == "learning_curve_pareto":
            summary["pareto"] = _build_learning_curve_pareto_summary(
                experiments, self.task_cfg
            )
        elif strategy == "axes_pareto":
            summary["pareto"] = _build_axes_pareto_summary(experiments, self.task_cfg)
        else:
            summary["pareto"] = _build_pareto_summary(experiments, self.task_cfg)
        return summary

    def compare(self, evidence: Evidence) -> Verdict:
        """Insert this cycle's points and report what the frontier became."""

        if self.state_dir is None:
            raise ValueError("FrontierComparator needs a state directory.")
        before = _frontier_identities(self.status.get("frontier") or ())
        status: dict[str, Any] = dict(self.status)
        summary = self._points_from(evidence)
        _update_best_result(self.state_dir, status, evidence.cycle, summary)
        self.status = status
        archive = _json_load(
            objective_artifact_dir(self.state_dir) / "pareto_frontier.json"
        )
        frontier = list((archive or {}).get("frontier", []))
        self.status["frontier"] = frontier
        after = _frontier_identities(frontier)
        entered = sorted(after - before)
        selected = (archive or {}).get("selected") or {}
        current = summary.get("pareto") or {}
        per_candidate: dict[str, dict[str, Any]] = {}
        for experiment in summary.get("experiments", []):
            name = str(experiment["name"])
            per_candidate[name] = {
                "invalid_reasons": experiment.get("pareto_invalid_reasons", [])
            }
        for point in current.get("points", current.get("frontier", [])):
            per_candidate[str(point.get("experiment", ""))] = {
                **dict(point),
                "objective_value": _selection_axis_value(
                    {"selected": point, "selection": current.get("selection")}
                ),
            }
        return Verdict(
            decision="frontier_changed" if entered else "dominated",
            improved=bool(entered),
            # A frontier has no single value; section 9 Q3 resolves this as the
            # selection axis when one is named, and None otherwise.
            objective_value=_selection_axis_value(archive),
            # `candidate` is the controller's word for a node's name; the
            # frontier stores it as `experiment`.
            representative=(
                {**dict(selected), "candidate": str(selected.get("experiment", ""))}
                if isinstance(selected, dict)
                else {}
            ),
            per_candidate=per_candidate,
            recommended_bases=self.recommend_bases(),
            description=(
                f"{len(frontier)} point(s) on the frontier; "
                f"{len(entered)} entered this cycle"
            ),
            delta_text=(
                "entered the frontier: "
                + ", ".join(f"cycle {cycle} {name}" for cycle, name in entered)
                if entered
                else "no new frontier point"
            ),
            decision_text=(
                "frontier_changed - the frontier gained at least one point"
                if entered
                else "dominated - every candidate was dominated by the frontier"
            ),
            observed_metric_text=_selection_excerpt({"representative": selected}, {}),
            last_result_text=(
                f"selected {selected.get('experiment', '')}"
                if isinstance(selected, dict)
                else ""
            ),
            state={"archive": archive or {}},
        )

    # -- rendering -----------------------------------------------------------

    def render_best(self) -> str:
        """The frontier itself, or the archive's own rendering of it.

        `controller_state.write_best_result` is the only writer of
        `best_result.md`; this returns its body. The archive updaters used to
        write that file themselves, which is how the run ended up with several
        writers racing to describe the same frontier.
        """

        path = (
            None
            if self.state_dir is None
            else objective_artifact_dir(self.state_dir) / "pareto_frontier.md"
        )
        if path is not None and path.exists():
            return path.read_text(encoding="utf-8")
        rendered = str(self.status.get("best_result_text", "") or "").strip()
        if rendered:
            return rendered
        return "- No frontier recorded yet."

    def render_progress_lines(self, verdict: Verdict) -> list[str]:
        return [
            f"- Frontier: {verdict.description}",
            f"- Selected: {verdict.last_result_text or 'none'}",
            f"- Anchor: {_anchor_text(self.status)}",
        ]

    def render_recent_result_line(self, row: Mapping[str, Any]) -> str:
        return (
            f"- cycle {row.get('cycle', '?')}: "
            f"{row.get('decision', row.get('status', ''))}"
        )

    def render_final_lines(self, verdict: Verdict) -> list[str]:
        return [
            f"- Frontier: {verdict.description}",
            f"- Decision: {verdict.decision_text}",
        ]

    def tracking_metrics(self, verdict: Verdict) -> Mapping[str, float]:
        metrics: dict[str, float] = {
            "frontier_size": float(
                len((verdict.state.get("archive") or {}).get("frontier", []))
            ),
        }
        if verdict.objective_value is not None:
            metrics["objective_value"] = float(verdict.objective_value)
        for key, value in (verdict.representative or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"selected_{key}"] = float(value)
        return metrics


def _node_id(point: Mapping[str, Any]) -> NodeId:
    return (int(point.get("cycle", 0) or 0), str(point.get("experiment", "")))


def _frontier_identities(points: Sequence[Mapping[str, Any]]) -> set[NodeId]:
    return {_node_id(point) for point in points if isinstance(point, dict)}


def _selection_axis_value(archive: Mapping[str, Any] | None) -> float | None:
    """The selected point's value on the named selection axis, if there is one."""

    if not isinstance(archive, dict):
        return None
    selected = archive.get("selected")
    selection = archive.get("selection")
    if not isinstance(selected, dict) or not isinstance(selection, str):
        return None
    value = selected.get(selection)
    return float(value) if isinstance(value, (int, float)) else None


def _anchor_text(status: Mapping[str, Any]) -> str:
    anchor = status.get("active_anchor")
    if not isinstance(anchor, dict):
        return "none"
    point = anchor.get("point") or {}
    return f"{anchor.get('role', '')}={point.get('experiment', '')}"
