"""Deterministic Pareto-frontier utilities for hpo evaluators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParetoAxis:
    """One independently optimized metric."""

    key: str
    goal: str
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Pareto axis key must be non-empty.")
        if self.goal not in {"minimize", "maximize"}:
            raise ValueError("Pareto axis goal must be 'minimize' or 'maximize'.")
        if not math.isfinite(self.epsilon) or self.epsilon < 0.0:
            raise ValueError("Pareto axis epsilon must be finite and non-negative.")


def _axis_value(point: Mapping[str, Any], axis: ParetoAxis) -> float:
    if axis.key not in point:
        raise ValueError(f"Pareto point is missing metric '{axis.key}'.")
    value = float(point[axis.key])
    if not math.isfinite(value):
        raise ValueError(f"Pareto metric '{axis.key}' must be finite.")
    return value


def pareto_dominates(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    axes: Sequence[ParetoAxis],
) -> bool:
    """Return whether ``left`` epsilon-dominates ``right``."""

    if not axes:
        raise ValueError("Pareto dominance requires at least one axis.")
    strictly_better = False
    for axis in axes:
        left_value = _axis_value(left, axis)
        right_value = _axis_value(right, axis)
        if axis.goal == "minimize":
            if left_value > right_value + axis.epsilon:
                return False
            strictly_better |= left_value < right_value - axis.epsilon
        else:
            if left_value < right_value - axis.epsilon:
                return False
            strictly_better |= left_value > right_value + axis.epsilon
    return strictly_better


def _frontier_sort_key(
    point: Mapping[str, Any], axes: Sequence[ParetoAxis]
) -> tuple[Any, ...]:
    values = []
    for axis in axes:
        value = _axis_value(point, axis)
        values.append(value if axis.goal == "minimize" else -value)
    values.extend(
        [
            str(point.get("experiment", point.get("name", ""))),
            point.get("eval_index", 0),
        ]
    )
    return tuple(values)


def pareto_frontier(
    points: Sequence[Mapping[str, Any]], axes: Sequence[ParetoAxis]
) -> list[dict[str, Any]]:
    """Return all non-dominated points in deterministic axis order."""

    frontier = [
        dict(point)
        for index, point in enumerate(points)
        if not any(
            index != other_index and pareto_dominates(other, point, axes)
            for other_index, other in enumerate(points)
        )
    ]
    return sorted(frontier, key=lambda point: _frontier_sort_key(point, axes))


def pareto_representatives(
    frontier: Sequence[Mapping[str, Any]],
    *,
    time_key: str,
    quality_key: str,
    quality_goal: str,
) -> dict[str, dict[str, Any]]:
    """Choose named views of a frontier without replacing the frontier itself."""

    if not frontier:
        return {}
    if quality_goal not in {"minimize", "maximize"}:
        raise ValueError("quality_goal must be 'minimize' or 'maximize'.")

    points = [dict(point) for point in frontier]

    def quality_rank(point: Mapping[str, Any]) -> float:
        value = float(point[quality_key])
        return -value if quality_goal == "maximize" else value

    fastest = min(
        points,
        key=lambda point: (
            float(point[time_key]),
            quality_rank(point),
            str(point.get("experiment", point.get("name", ""))),
            point.get("eval_index", 0),
        ),
    )
    quality = min(
        points,
        key=lambda point: (
            quality_rank(point),
            float(point[time_key]),
            str(point.get("experiment", point.get("name", ""))),
            point.get("eval_index", 0),
        ),
    )

    times = [float(point[time_key]) for point in points]
    qualities = [float(point[quality_key]) for point in points]
    time_min, time_max = min(times), max(times)
    quality_min, quality_max = min(qualities), max(qualities)

    def knee_rank(point: Mapping[str, Any]) -> tuple[Any, ...]:
        time_value = float(point[time_key])
        quality_value = float(point[quality_key])
        time_desirability = (
            1.0
            if time_max == time_min
            else (time_max - time_value) / (time_max - time_min)
        )
        if quality_max == quality_min:
            quality_desirability = 1.0
        elif quality_goal == "maximize":
            quality_desirability = (quality_value - quality_min) / (
                quality_max - quality_min
            )
        else:
            quality_desirability = (quality_max - quality_value) / (
                quality_max - quality_min
            )
        distance = math.hypot(1.0 - time_desirability, 1.0 - quality_desirability)
        return (
            distance,
            time_value,
            quality_rank(point),
            str(point.get("experiment", point.get("name", ""))),
            point.get("eval_index", 0),
        )

    knee = min(points, key=knee_rank)
    return {"fastest": fastest, "quality": quality, "knee": knee}


def pareto_recommendations(
    frontier: Sequence[Mapping[str, Any]],
    *,
    first_key: str,
    second_key: str,
    first_goal: str,
    second_goal: str,
    second_constraint: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Return champions and a balanced view for two directed objectives.

    ``second_constraint`` is interpreted in the desirable direction: a floor
    for maximization and a ceiling for minimization.
    """

    if not frontier:
        return {}
    if first_goal not in {"minimize", "maximize"}:
        raise ValueError("first_goal must be 'minimize' or 'maximize'.")
    if second_goal not in {"minimize", "maximize"}:
        raise ValueError("second_goal must be 'minimize' or 'maximize'.")
    points = [dict(point) for point in frontier]

    def stable_identity(point: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(point.get("experiment", point.get("name", ""))),
            point.get("eval_index", 0),
        )

    def rank_value(value: float, goal: str) -> float:
        return value if goal == "minimize" else -value

    first = min(
        points,
        key=lambda point: (
            rank_value(float(point[first_key]), first_goal),
            rank_value(float(point[second_key]), second_goal),
            *stable_identity(point),
        ),
    )
    second = min(
        points,
        key=lambda point: (
            rank_value(float(point[second_key]), second_goal),
            rank_value(float(point[first_key]), first_goal),
            *stable_identity(point),
        ),
    )
    first_values = [float(point[first_key]) for point in points]
    second_values = [float(point[second_key]) for point in points]
    first_min, first_max = min(first_values), max(first_values)
    second_min, second_max = min(second_values), max(second_values)

    def balanced_rank(point: Mapping[str, Any]) -> tuple[Any, ...]:
        first_value = float(point[first_key])
        second_value = float(point[second_key])
        if first_max == first_min:
            first_desirability = 1.0
        elif first_goal == "maximize":
            first_desirability = (first_value - first_min) / (first_max - first_min)
        else:
            first_desirability = (first_max - first_value) / (first_max - first_min)
        if second_max == second_min:
            second_desirability = 1.0
        elif second_goal == "maximize":
            second_desirability = (second_value - second_min) / (
                second_max - second_min
            )
        else:
            second_desirability = (second_max - second_value) / (
                second_max - second_min
            )
        return (
            math.hypot(1.0 - first_desirability, 1.0 - second_desirability),
            rank_value(first_value, first_goal),
            rank_value(second_value, second_goal),
            *stable_identity(point),
        )

    recommendations = {
        "first": first,
        "second": second,
        "balanced": min(points, key=balanced_rank),
    }
    if second_constraint is not None:
        eligible = [
            point
            for point in points
            if (
                float(point[second_key]) >= second_constraint
                if second_goal == "maximize"
                else float(point[second_key]) <= second_constraint
            )
        ]
        if eligible:
            recommendations["first_constrained"] = min(
                eligible,
                key=lambda point: (
                    rank_value(float(point[first_key]), first_goal),
                    rank_value(float(point[second_key]), second_goal),
                    *stable_identity(point),
                ),
            )
    return recommendations
