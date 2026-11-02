"""Frozen public benchmark protocol for BeyondMimic Stage-1 tracking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

TRAINING_SEEDS: Final = (42, 43, 44)
TRAINING_ENVIRONMENTS: Final = 4096
TRAINING_ITERATIONS: Final = 6000
CHECKPOINT_INTERVAL: Final = 500
EVALUATION_EPISODES: Final = 1024
EVALUATION_HORIZON_SECONDS: Final = 10.0
EVALUATION_SAMPLING_MODE: Final = "start"

QUALITY_THRESHOLDS: Final = {
    "success_rate": 0.96,
    "mpkpe": 0.120,
    "r_mpkpe": 0.033,
    "ee_pos_error": 0.052,
    "ee_ori_error": 0.180,
}

METRIC_ORDER: Final = (
    "success_rate",
    "mpkpe",
    "r_mpkpe",
    "ee_pos_error",
    "ee_ori_error",
    "joint_vel_error",
)

BENCHMARK_RUN_ORDER: Final = tuple(
    (implementation, seed)
    for seed in TRAINING_SEEDS
    for implementation in ("mjlab", "instantmimic")
)


@dataclass(frozen=True, slots=True)
class EvaluatedCheckpoint:
    """Metrics and identity needed for deterministic checkpoint selection."""

    iteration: int
    path: str
    metrics: Mapping[str, float]


def meets_quality_threshold(metrics: Mapping[str, float]) -> bool:
    """Return whether all frozen public-nightly sanity thresholds pass."""
    missing = set(QUALITY_THRESHOLDS) - set(metrics)
    if missing:
        raise ValueError(f"Missing benchmark metrics: {sorted(missing)}")
    return bool(
        float(metrics["success_rate"]) >= QUALITY_THRESHOLDS["success_rate"]
        and all(
            float(metrics[name]) <= threshold
            for name, threshold in QUALITY_THRESHOLDS.items()
            if name != "success_rate"
        )
    )


def checkpoint_quality_key(checkpoint: EvaluatedCheckpoint) -> tuple[float, ...]:
    """Rank checkpoints by the frozen, public-metric quality ordering.

    Threshold-passing checkpoints outrank non-passing checkpoints. Remaining
    ties are broken by success rate, then lower public tracking errors in
    dashboard order, and finally by the later iteration.
    """
    metrics = checkpoint.metrics
    missing = set(METRIC_ORDER) - set(metrics)
    if missing:
        raise ValueError(f"Missing benchmark metrics: {sorted(missing)}")
    return (
        float(meets_quality_threshold(metrics)),
        float(metrics["success_rate"]),
        -float(metrics["mpkpe"]),
        -float(metrics["r_mpkpe"]),
        -float(metrics["ee_pos_error"]),
        -float(metrics["ee_ori_error"]),
        -float(metrics["joint_vel_error"]),
        float(checkpoint.iteration),
    )


def select_best_checkpoint(
    checkpoints: Sequence[EvaluatedCheckpoint],
) -> EvaluatedCheckpoint:
    """Select one checkpoint using the frozen quality ordering."""
    if not checkpoints:
        raise ValueError("At least one evaluated checkpoint is required")
    return max(checkpoints, key=checkpoint_quality_key)


def convergence_iteration(
    checkpoints: Sequence[EvaluatedCheckpoint],
) -> int | None:
    """Return the earliest saved iteration meeting the frozen quality gate."""
    passing = [
        checkpoint.iteration
        for checkpoint in checkpoints
        if meets_quality_threshold(checkpoint.metrics)
    ]
    return min(passing) if passing else None
