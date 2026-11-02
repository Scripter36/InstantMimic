"""Shared scalar and Pareto objective selection for hpo."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from instantmimic.hpo.objective.pareto import ParetoAxis, pareto_frontier


@dataclass(frozen=True)
class ObjectiveAxis:
    key: str
    goal: str
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        ParetoAxis(key=self.key, goal=self.goal, epsilon=self.epsilon)

    def as_pareto_axis(self) -> ParetoAxis:
        return ParetoAxis(key=self.key, goal=self.goal, epsilon=self.epsilon)


@dataclass(frozen=True)
class ObjectiveSpec:
    mode: str
    axes: tuple[ObjectiveAxis, ...]
    selection: str = "balanced"

    def __post_init__(self) -> None:
        if self.mode not in {"scalar", "pareto"}:
            raise ValueError("Objective mode must be 'scalar' or 'pareto'.")
        if not self.axes:
            raise ValueError("An objective requires at least one axis.")
        if self.mode == "scalar" and len(self.axes) != 1:
            raise ValueError("Scalar objectives require exactly one axis.")
        if self.mode == "pareto" and len(self.axes) < 2:
            raise ValueError("Pareto objectives require at least two axes.")
        valid_selections = {"balanced", *(axis.key for axis in self.axes)}
        if self.selection not in valid_selections:
            choices = ", ".join(sorted(valid_selections))
            raise ValueError(f"Objective selection must be one of: {choices}.")


def _identity(point: Mapping[str, Any]) -> tuple[str, int]:
    return (
        str(point.get("experiment", point.get("name", ""))),
        int(point.get("eval_index", 0) or 0),
    )


def _rank_value(point: Mapping[str, Any], axis: ObjectiveAxis) -> float:
    value = float(point[axis.key])
    if not math.isfinite(value):
        raise ValueError(f"Objective metric '{axis.key}' must be finite.")
    return value if axis.goal == "minimize" else -value


def _axis_champion(
    points: Sequence[Mapping[str, Any]],
    axis: ObjectiveAxis,
    axes: Sequence[ObjectiveAxis],
) -> dict[str, Any]:
    tiebreak_axes = [item for item in axes if item.key != axis.key]
    return dict(
        min(
            points,
            key=lambda point: (
                _rank_value(point, axis),
                *(_rank_value(point, item) for item in tiebreak_axes),
                *_identity(point),
            ),
        )
    )


def _balanced_point(
    points: Sequence[Mapping[str, Any]], axes: Sequence[ObjectiveAxis]
) -> dict[str, Any]:
    bounds = {
        axis.key: (
            min(float(point[axis.key]) for point in points),
            max(float(point[axis.key]) for point in points),
        )
        for axis in axes
    }

    def rank(point: Mapping[str, Any]) -> tuple[Any, ...]:
        squared_distance = 0.0
        directed_values: list[float] = []
        for axis in axes:
            value = float(point[axis.key])
            lower, upper = bounds[axis.key]
            if lower == upper:
                desirability = 1.0
            elif axis.goal == "maximize":
                desirability = (value - lower) / (upper - lower)
            else:
                desirability = (upper - value) / (upper - lower)
            squared_distance += (1.0 - desirability) ** 2
            directed_values.append(_rank_value(point, axis))
        return (math.sqrt(squared_distance), *directed_values, *_identity(point))

    return dict(min(points, key=rank))


def select_objective(
    points: Sequence[Mapping[str, Any]], spec: ObjectiveSpec
) -> dict[str, Any]:
    """Return the authoritative frontier and configured recommendation."""

    if not points:
        return {
            "mode": spec.mode,
            "axes": [axis.__dict__ for axis in spec.axes],
            "frontier": [],
            "representatives": {},
            "selected": None,
        }
    for point in points:
        for axis in spec.axes:
            if axis.key not in point:
                raise ValueError(f"Objective point is missing metric '{axis.key}'.")
            _rank_value(point, axis)

    frontier = pareto_frontier(
        points, tuple(axis.as_pareto_axis() for axis in spec.axes)
    )
    representatives = {
        axis.key: _axis_champion(frontier, axis, spec.axes) for axis in spec.axes
    }
    representatives["balanced"] = _balanced_point(frontier, spec.axes)
    selected = representatives.get(spec.selection, representatives["balanced"])
    return {
        "mode": spec.mode,
        "axes": [axis.__dict__ for axis in spec.axes],
        "selection": spec.selection,
        "frontier": frontier,
        "representatives": representatives,
        "selected": selected,
    }
