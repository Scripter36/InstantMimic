"""Comparators: the only place ``objective.mode`` means anything.

Everything else in ``instantmimic.hpo`` -- schema, planner, experiment
model, evaluator, evidence, state, tracking -- is one path for both modes. A
comparator decides how evaluated evidence is scored, ranked, kept, and
rendered, and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from instantmimic.hpo.objective.base import (
    Evidence,
    ObjectiveComparator,
    Verdict,
)
from instantmimic.hpo.objective.frontier import (
    FrontierComparator,
    FrontierObjective,
)
from instantmimic.hpo.objective.scalar import (
    ScalarComparator,
    ScalarContext,
    ScalarObjective,
    ScalarState,
)

__all__ = [
    "Evidence",
    "FrontierComparator",
    "FrontierObjective",
    "ObjectiveComparator",
    "ScalarComparator",
    "ScalarContext",
    "ScalarObjective",
    "ScalarState",
    "Verdict",
    "build_comparator",
]


def build_comparator(
    objective_cfg: Mapping[str, Any] | None, **kwargs: Any
) -> ObjectiveComparator:
    """Select the comparator family. This is the only read of ``objective.mode``.

    Every other module in the package is written once and runs unchanged under
    either mode; if a second reader of ``mode`` ever appears outside this
    function, the unification has regressed.
    """

    cfg = objective_cfg or {}
    mode = str(cfg.get("mode", "") or "").strip().lower()
    if mode == "scalar":
        kwargs.pop("task_cfg", None)
        return ScalarComparator(ScalarObjective.from_config(cfg), **kwargs)
    if mode == "pareto":
        return FrontierComparator(FrontierObjective.from_config(cfg), **kwargs)
    raise ValueError("objective.mode must be 'scalar' or 'pareto'.")
