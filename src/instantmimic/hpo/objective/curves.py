"""Scalar objective derivations over a recorded benchmark curve.

The evaluator records what every trial measured; this module turns that record
into the single number a scalar comparator ranks by. Nothing here reads a log:
its input is ``evidence.TrainEvidence``, so the same derivations work for a
live run, a replay, and a test fixture.

Five derivations, selected by ``objective_name_for``:

* ``wall_clock`` -- the trial's own wall time.
* ``time_to_reward_threshold_excluding_warmup`` / ``..._tracking_benchmark...``
  -- the marker the trainer emitted when it first met the threshold.
* ``frames_to_tracking_benchmark`` -- the same event counted in frames.
* ``tracking_benchmark_auc`` -- the time-average of the best-so-far benchmark
  error, each interval valued at its right endpoint and clamped to the error
  cap, divided by the horizon.

``threshold_mode: capped`` keeps a trial that never met the threshold inside
the comparison, ranked by its best observed benchmark score, instead of
discarding it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from instantmimic.hpo.evidence import TrainEvidence

# Without an explicit cap, a run that never approaches its threshold is
# scored at 1.2x the threshold rather than at its raw error.
DEFAULT_AUC_ERROR_CAP_MULTIPLIER = 1.2


@dataclass(frozen=True)
class BenchmarkEvalMetrics:
    success_rate: float | None = None
    ape_mean: float | None = None
    global_mpjpe_mean: float | None = None
    ee_mean: float | None = None
    score: float | None = None


@dataclass(frozen=True)
class BenchmarkAUCResult:
    auc_score: float
    best_metrics: BenchmarkEvalMetrics


@dataclass(frozen=True)
class Thresholds:
    """The benchmark gate a scalar objective measures time to."""

    success_rate: float | None = None
    ape_mean: float | None = None
    global_mpjpe_mean: float | None = None
    ee_mean: float | None = None
    auc_error_cap: float | None = None

    @classmethod
    def from_objective(cls, objective_cfg: object) -> Thresholds:
        raw = {}
        if objective_cfg is not None:
            thresholds = (
                objective_cfg.get("thresholds", None)
                if hasattr(objective_cfg, "get")
                else None
            )
            raw = dict(thresholds) if thresholds else {}
            cap = objective_cfg.get("auc_error_cap", None)
        else:
            cap = None
        return cls(
            success_rate=_optional_float(raw.get("success_rate")),
            ape_mean=_optional_float(raw.get("ape_mean")),
            global_mpjpe_mean=_optional_float(raw.get("global_mpjpe_mean")),
            ee_mean=_optional_float(raw.get("ee_mean")),
            auc_error_cap=_optional_float(cap),
        )

    def enabled(self) -> bool:
        return benchmark_thresholds_enabled(
            self.ape_mean,
            self.ee_mean,
            self.success_rate,
            global_mpjpe_threshold=self.global_mpjpe_mean,
        )


@dataclass(frozen=True)
class TrialObjective:
    """One trial reduced to the number its comparator ranks by."""

    value: float | None
    tiebreak: float | None = None
    interval_bucket: int | None = None
    metrics: BenchmarkEvalMetrics = BenchmarkEvalMetrics()
    complete: bool = True
    reason: str = ""


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)  # type: ignore[arg-type]


def benchmark_thresholds_enabled(
    ape_threshold: float | None,
    ee_threshold: float | None,
    success_rate_threshold: float | None,
    global_mpjpe_threshold: float | None = None,
) -> bool:
    return (
        ape_threshold is not None
        or ee_threshold is not None
        or success_rate_threshold is not None
        or global_mpjpe_threshold is not None
    )


def normalize_benchmark_objective_unit(value: str | None = None) -> str:
    unit = str(value or "seconds").strip().lower()
    if unit in {"frames", "tuples"}:
        return "frames"
    if unit in {"seconds", "time"}:
        return "seconds"
    raise ValueError(
        "'benchmark_objective_unit' must be 'seconds', 'frames', or 'tuples'."
    )


def objective_name_for(
    *,
    reward_threshold: float | None,
    benchmark_ape_threshold: float | None,
    benchmark_ee_threshold: float | None,
    benchmark_success_rate_threshold: float | None,
    benchmark_global_mpjpe_threshold: float | None = None,
    benchmark_objective_mode: str = "threshold",
    benchmark_objective_unit: str = "seconds",
) -> str:
    if benchmark_thresholds_enabled(
        benchmark_ape_threshold,
        benchmark_ee_threshold,
        benchmark_success_rate_threshold,
        global_mpjpe_threshold=benchmark_global_mpjpe_threshold,
    ):
        if str(benchmark_objective_mode).strip().lower() == "auc":
            return "tracking_benchmark_auc"
        if normalize_benchmark_objective_unit(benchmark_objective_unit) == "frames":
            return "frames_to_tracking_benchmark"
        return "time_to_tracking_benchmark_excluding_warmup"
    if reward_threshold is not None:
        return "time_to_reward_threshold_excluding_warmup"
    return "wall_clock"


def objective_label(objective_name: str) -> str:
    if objective_name == "tracking_benchmark_auc":
        return "TRACKING_BENCHMARK_ERROR_AUC"
    if objective_name == "frames_to_tracking_benchmark":
        return "FRAMES_TO_TRACKING_BENCHMARK"
    if objective_name == "time_to_tracking_benchmark_excluding_warmup":
        return "TIME_TO_TRACKING_BENCHMARK_SECONDS"
    if objective_name == "time_to_reward_threshold_excluding_warmup":
        return "TIME_TO_REWARD_THRESHOLD_SECONDS"
    return "WALL_SECONDS"


def benchmark_score(
    *,
    success_rate: float | None,
    ape_mean: float | None,
    global_mpjpe_mean: float | None = None,
    ee_mean: float | None = None,
    benchmark_success_rate_threshold: float | None = None,
    benchmark_ape_threshold: float | None = None,
    benchmark_global_mpjpe_threshold: float | None = None,
    benchmark_ee_threshold: float | None = None,
) -> float | None:
    success_rate = finite_float_or_none(success_rate)
    ape_mean = finite_float_or_none(ape_mean)
    global_mpjpe_mean = finite_float_or_none(global_mpjpe_mean)
    ee_mean = finite_float_or_none(ee_mean)

    score_terms: list[float] = []
    if success_rate is not None and benchmark_success_rate_threshold is not None:
        score_terms.append(
            float(benchmark_success_rate_threshold) / max(float(success_rate), 1e-12)
        )
    if ape_mean is not None and benchmark_ape_threshold is not None:
        score_terms.append(ape_mean / max(float(benchmark_ape_threshold), 1e-12))
    if global_mpjpe_mean is not None and benchmark_global_mpjpe_threshold is not None:
        score_terms.append(
            global_mpjpe_mean / max(float(benchmark_global_mpjpe_threshold), 1e-12)
        )
    if ee_mean is not None and benchmark_ee_threshold is not None:
        score_terms.append(ee_mean / max(float(benchmark_ee_threshold), 1e-12))
    if not score_terms:
        if success_rate is not None:
            score_terms.append(1.0 / max(float(success_rate), 1e-12))
        if ape_mean is not None:
            score_terms.append(ape_mean)
        if global_mpjpe_mean is not None:
            score_terms.append(global_mpjpe_mean)
        if ee_mean is not None:
            score_terms.append(ee_mean)
    return max(score_terms) if score_terms else None


def finite_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def resolve_auc_error_cap(
    *,
    benchmark_auc_error_cap: float | None,
    benchmark_ape_threshold: float | None,
    benchmark_global_mpjpe_threshold: float | None = None,
    benchmark_ee_threshold: float | None = None,
) -> float:
    if benchmark_auc_error_cap is not None and benchmark_auc_error_cap > 0.0:
        return float(benchmark_auc_error_cap)
    if benchmark_ape_threshold is not None and benchmark_ape_threshold > 0.0:
        return float(benchmark_ape_threshold) * DEFAULT_AUC_ERROR_CAP_MULTIPLIER
    if (
        benchmark_global_mpjpe_threshold is not None
        and benchmark_global_mpjpe_threshold > 0.0
    ):
        return (
            float(benchmark_global_mpjpe_threshold) * DEFAULT_AUC_ERROR_CAP_MULTIPLIER
        )
    if benchmark_ee_threshold is not None and benchmark_ee_threshold > 0.0:
        return float(benchmark_ee_threshold) * DEFAULT_AUC_ERROR_CAP_MULTIPLIER
    return 1.0


def auc_reference_error(
    *,
    benchmark_ape_threshold: float | None,
    benchmark_ee_threshold: float | None,
    benchmark_auc_error_cap: float = 1.0,
    benchmark_global_mpjpe_threshold: float | None = None,
) -> float:
    if benchmark_ape_threshold is not None and benchmark_ape_threshold > 0.0:
        return float(benchmark_ape_threshold)
    if (
        benchmark_global_mpjpe_threshold is not None
        and benchmark_global_mpjpe_threshold > 0.0
    ):
        return float(benchmark_global_mpjpe_threshold)
    if benchmark_ee_threshold is not None and benchmark_ee_threshold > 0.0:
        return float(benchmark_ee_threshold)
    return float(benchmark_auc_error_cap)


def auc_point_score(
    *,
    success_rate: float | None,
    ape_mean: float | None,
    global_mpjpe_mean: float | None = None,
    ee_mean: float | None = None,
    benchmark_success_rate_threshold: float | None = None,
    benchmark_ape_threshold: float | None = None,
    benchmark_global_mpjpe_threshold: float | None = None,
    benchmark_ee_threshold: float | None = None,
    benchmark_auc_error_cap: float,
) -> float | None:
    score_terms: list[float] = []
    success_rate = finite_float_or_none(success_rate)
    ape_mean = finite_float_or_none(ape_mean)
    global_mpjpe_mean = finite_float_or_none(global_mpjpe_mean)
    ee_mean = finite_float_or_none(ee_mean)
    if success_rate is not None and benchmark_success_rate_threshold is not None:
        reference_error = auc_reference_error(
            benchmark_ape_threshold=benchmark_ape_threshold,
            benchmark_global_mpjpe_threshold=benchmark_global_mpjpe_threshold,
            benchmark_ee_threshold=benchmark_ee_threshold,
            benchmark_auc_error_cap=benchmark_auc_error_cap,
        )
        score_terms.append(
            reference_error
            * float(benchmark_success_rate_threshold)
            / max(float(success_rate), 1e-12)
        )
    if ape_mean is not None and benchmark_ape_threshold is not None:
        score_terms.append(ape_mean)
    if global_mpjpe_mean is not None and benchmark_global_mpjpe_threshold is not None:
        if benchmark_ape_threshold is not None and benchmark_ape_threshold > 0.0:
            score_terms.append(
                global_mpjpe_mean
                / max(float(benchmark_global_mpjpe_threshold), 1e-12)
                * float(benchmark_ape_threshold)
            )
        else:
            score_terms.append(global_mpjpe_mean)
    if ee_mean is not None and benchmark_ee_threshold is not None:
        if benchmark_ape_threshold is not None and benchmark_ape_threshold > 0.0:
            score_terms.append(
                ee_mean
                / max(float(benchmark_ee_threshold), 1e-12)
                * float(benchmark_ape_threshold)
            )
        else:
            score_terms.append(ee_mean)
    if not score_terms:
        if success_rate is not None:
            score_terms.append(
                benchmark_auc_error_cap / max(float(success_rate), 1e-12)
            )
        if ape_mean is not None:
            score_terms.append(ape_mean)
        if global_mpjpe_mean is not None:
            score_terms.append(global_mpjpe_mean)
        if ee_mean is not None:
            score_terms.append(ee_mean)
    if not score_terms:
        return None
    return min(max(score_terms), benchmark_auc_error_cap)


def objective_interval_bucket(
    objective_seconds: float | None,
    interval_seconds: float | None,
) -> int | None:
    if objective_seconds is None or interval_seconds is None or interval_seconds <= 0.0:
        return None
    if objective_seconds <= 0.0:
        return 0
    return max(1, int(math.ceil((objective_seconds - 1e-12) / interval_seconds)))


def best_benchmark_metrics(
    train: TrainEvidence, thresholds: Thresholds
) -> BenchmarkEvalMetrics | None:
    """The lowest-scoring evaluation the run recorded.

    Gate metrics describe the checkpoint the trainer actually evaluated, so a
    curve point that carries them is read through the gate prefix; otherwise the
    current series is used. ``EVAL_BEST_CHECKPOINT`` participates as one more
    candidate point.
    """

    points: list[BenchmarkEvalMetrics] = []
    for values in train.benchmark_curve:
        prefix = "gate" if any(key.startswith("gate_") for key in values) else "current"
        points.append(
            _scored(
                BenchmarkEvalMetrics(
                    success_rate=values.get(f"{prefix}_success_rate"),
                    ape_mean=values.get(f"{prefix}_ape_mean"),
                    global_mpjpe_mean=values.get(f"{prefix}_global_mpjpe_mean"),
                    ee_mean=values.get(f"{prefix}_ee_mean"),
                ),
                thresholds,
            )
        )
    if train.best_checkpoint_metrics:
        values = train.best_checkpoint_metrics
        points.append(
            _scored(
                BenchmarkEvalMetrics(
                    success_rate=values.get("success_rate"),
                    ape_mean=values.get("ape_mean"),
                    global_mpjpe_mean=values.get("global_mpjpe_mean"),
                    ee_mean=values.get("ee_mean"),
                ),
                thresholds,
            )
        )

    best: BenchmarkEvalMetrics | None = None
    for candidate in points:
        if _all_none(candidate):
            continue
        if best is None:
            best = candidate
            continue
        if candidate.score is None:
            continue
        if best.score is None or candidate.score < best.score:
            best = candidate
    return best


def _all_none(metrics: BenchmarkEvalMetrics) -> bool:
    return (
        metrics.success_rate is None
        and metrics.ape_mean is None
        and metrics.global_mpjpe_mean is None
        and metrics.ee_mean is None
    )


def _scored(
    metrics: BenchmarkEvalMetrics, thresholds: Thresholds
) -> BenchmarkEvalMetrics:
    return BenchmarkEvalMetrics(
        success_rate=metrics.success_rate,
        ape_mean=metrics.ape_mean,
        global_mpjpe_mean=metrics.global_mpjpe_mean,
        ee_mean=metrics.ee_mean,
        score=benchmark_score(
            success_rate=metrics.success_rate,
            ape_mean=metrics.ape_mean,
            global_mpjpe_mean=metrics.global_mpjpe_mean,
            ee_mean=metrics.ee_mean,
            benchmark_success_rate_threshold=thresholds.success_rate,
            benchmark_ape_threshold=thresholds.ape_mean,
            benchmark_global_mpjpe_threshold=thresholds.global_mpjpe_mean,
            benchmark_ee_threshold=thresholds.ee_mean,
        ),
    )


def benchmark_auc(
    train: TrainEvidence,
    thresholds: Thresholds,
    *,
    horizon_seconds: float | None,
) -> BenchmarkAUCResult | None:
    """Time-average the best-so-far benchmark error over the horizon."""

    error_cap = resolve_auc_error_cap(
        benchmark_auc_error_cap=thresholds.auc_error_cap,
        benchmark_ape_threshold=thresholds.ape_mean,
        benchmark_global_mpjpe_threshold=thresholds.global_mpjpe_mean,
        benchmark_ee_threshold=thresholds.ee_mean,
    )
    points: list[tuple[float, float, BenchmarkEvalMetrics]] = []
    best_score: float | None = None
    best_metrics: BenchmarkEvalMetrics | None = None
    for values in train.benchmark_curve:
        active_time = values.get("active_train_time")
        if active_time is None:
            continue
        prefix = "gate" if any(key.startswith("gate_") for key in values) else "current"
        metrics = BenchmarkEvalMetrics(
            success_rate=values.get(f"{prefix}_success_rate"),
            ape_mean=values.get(f"{prefix}_ape_mean"),
            global_mpjpe_mean=values.get(f"{prefix}_global_mpjpe_mean"),
            ee_mean=values.get(f"{prefix}_ee_mean"),
        )
        score = auc_point_score(
            success_rate=metrics.success_rate,
            ape_mean=metrics.ape_mean,
            global_mpjpe_mean=metrics.global_mpjpe_mean,
            ee_mean=metrics.ee_mean,
            benchmark_success_rate_threshold=thresholds.success_rate,
            benchmark_ape_threshold=thresholds.ape_mean,
            benchmark_global_mpjpe_threshold=thresholds.global_mpjpe_mean,
            benchmark_ee_threshold=thresholds.ee_mean,
            benchmark_auc_error_cap=error_cap,
        )
        if score is None:
            continue
        scored = BenchmarkEvalMetrics(
            success_rate=metrics.success_rate,
            ape_mean=metrics.ape_mean,
            global_mpjpe_mean=metrics.global_mpjpe_mean,
            ee_mean=metrics.ee_mean,
            score=score,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_metrics = scored
        points.append((max(0.0, active_time), best_score, best_metrics))

    if not points or best_metrics is None:
        return None

    points.sort(key=lambda item: item[0])
    horizon = (
        float(horizon_seconds)
        if horizon_seconds is not None and horizon_seconds > 0.0
        else points[-1][0]
    )
    horizon = max(horizon, points[-1][0], 1.0e-12)

    previous_time = 0.0
    previous_score = points[0][1]
    area = 0.0
    for raw_time, score, _metrics in points:
        current_time = min(max(raw_time, previous_time), horizon)
        if current_time > previous_time:
            area += (current_time - previous_time) * score
            previous_time = current_time
        previous_score = score
        if previous_time >= horizon:
            break
    if previous_time < horizon:
        area += (horizon - previous_time) * previous_score
    return BenchmarkAUCResult(auc_score=area / horizon, best_metrics=best_metrics)


def _reached_metrics(train: TrainEvidence) -> BenchmarkEvalMetrics:
    reached = train.reached_tracking_benchmark
    return BenchmarkEvalMetrics(
        success_rate=reached.get("success_rate"),
        ape_mean=reached.get("ape_mean"),
        global_mpjpe_mean=reached.get("global_mpjpe_mean"),
        ee_mean=reached.get("ee_mean"),
    )


def thresholds_met(metrics: BenchmarkEvalMetrics, thresholds: Thresholds) -> bool:
    if thresholds.success_rate is not None and (
        metrics.success_rate is None or metrics.success_rate < thresholds.success_rate
    ):
        return False
    for value, limit in (
        (metrics.ape_mean, thresholds.ape_mean),
        (metrics.global_mpjpe_mean, thresholds.global_mpjpe_mean),
        (metrics.ee_mean, thresholds.ee_mean),
    ):
        if limit is not None and (value is None or value > limit):
            return False
    return True


def trial_objective(
    train: TrainEvidence,
    *,
    objective_name: str,
    thresholds: Thresholds,
    threshold_mode: str = "threshold",
    horizon_seconds: float | None = None,
    wall_seconds: float | None = None,
    interval_seconds: float | None = None,
) -> TrialObjective:
    """Reduce one trial's evidence to its scalar objective.

    A trial that ran cleanly but never met the benchmark is *incomplete*, not
    failed: the evaluator's return code is an execution fact and this decision
    never touches it. ``threshold_mode: capped`` admits such a trial with a
    capped value so it can still be ranked by its best observed score.
    """

    if objective_name == "wall_clock":
        return TrialObjective(
            value=wall_seconds,
            interval_bucket=objective_interval_bucket(wall_seconds, interval_seconds),
            complete=wall_seconds is not None,
            reason="" if wall_seconds is not None else "no wall time recorded",
        )

    if objective_name == "tracking_benchmark_auc":
        result = benchmark_auc(train, thresholds, horizon_seconds=horizon_seconds)
        if result is None:
            return TrialObjective(
                value=None, complete=False, reason="no benchmark curve recorded"
            )
        return TrialObjective(
            value=result.auc_score,
            tiebreak=result.best_metrics.score,
            metrics=result.best_metrics,
        )

    if objective_name == "frames_to_tracking_benchmark":
        frames = train.reached_tracking_benchmark.get("frames")
        return _threshold_objective(
            train,
            thresholds,
            threshold_mode=threshold_mode,
            reached_value=frames,
            capped_value=float("inf"),
            interval_seconds=None,
        )

    # Both time-to-threshold objectives read the marker the trainer emitted.
    marker = (
        "time_to_tracking_benchmark_seconds"
        if objective_name == "time_to_tracking_benchmark_excluding_warmup"
        else "time_to_reward_threshold_seconds"
    )
    return _threshold_objective(
        train,
        thresholds,
        threshold_mode=threshold_mode,
        reached_value=train.threshold_events.get(marker),
        capped_value=horizon_seconds if horizon_seconds is not None else wall_seconds,
        interval_seconds=interval_seconds,
    )


def _threshold_objective(
    train: TrainEvidence,
    thresholds: Thresholds,
    *,
    threshold_mode: str,
    reached_value: float | None,
    capped_value: float | None,
    interval_seconds: float | None,
) -> TrialObjective:
    if reached_value is None:
        if threshold_mode != "capped":
            return TrialObjective(
                value=None, complete=False, reason="benchmark threshold not reached"
            )
        best = best_benchmark_metrics(train, thresholds)
        if best is None:
            return TrialObjective(
                value=None, complete=False, reason="no benchmark evaluation recorded"
            )
        return TrialObjective(
            value=capped_value,
            tiebreak=best.score,
            metrics=best,
            reason="capped: benchmark threshold not reached",
        )

    metrics = _scored(_reached_metrics(train), thresholds)
    if not thresholds_met(metrics, thresholds):
        return TrialObjective(
            value=None,
            complete=False,
            reason="recorded benchmark metrics do not meet the thresholds",
        )
    return TrialObjective(
        value=float(reached_value),
        tiebreak=metrics.score,
        interval_bucket=objective_interval_bucket(
            float(reached_value), interval_seconds
        ),
        metrics=metrics,
    )
