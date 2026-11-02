"""What a comparator is, and what it is handed.

A comparator answers one question -- given the evidence one cycle produced, what
is the verdict? -- and owns everything downstream of that: which candidate is
best, whether the run improved, which nodes are worth branching from next, and
how all of that reads to a human and to tracking.

It is handed ``Evidence``: the execution record, never a decision. Nothing in
``Evidence`` is objective-specific, which is why the same evaluator serves both
comparator families.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from instantmimic.hpo.experiment import Candidate, NodeId


class IncompatibleObjectiveError(ValueError):
    """The run's stored objective is not the one the task now configures.

    A changed objective means a new run: the archive and the incoming cycle are
    not measuring the same thing, so merging them would silently mix two
    experiments. This is fatal to the run, not to one cycle -- retrying only
    spends GPU time to reach the same conclusion.
    """


@dataclass(frozen=True)
class Evidence:
    """One cycle's execution record, as the evaluator wrote it."""

    cycle: int
    evidence_path: Path | None = None
    evidence: Mapping[str, Any] | None = None
    trials: Sequence[Mapping[str, Any]] = ()
    candidates: Sequence[Candidate] = ()
    # The evaluator's aggregate summary. It is an execution artifact, not a
    # decision: frontier comparators read their per-candidate points from it.
    summary: Mapping[str, Any] | None = None
    lifecycle: str = "completed"
    error: str = ""

    def candidate_names(self) -> tuple[str, ...]:
        return tuple(candidate.name for candidate in self.candidates)


@dataclass(frozen=True)
class Verdict:
    """What the comparator decided, and how it reads.

    ``decision`` is comparator vocabulary -- the scalar family says
    keep/advance/discard, the frontier family says frontier_changed/dominated --
    while ``improved`` is the one common signal the controller acts on.
    """

    decision: str
    improved: bool = False
    objective_value: float | None = None
    representative: Mapping[str, Any] = field(default_factory=dict)
    per_candidate: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    recommended_bases: Sequence[NodeId] = ()
    description: str = ""
    delta_text: str = ""
    decision_text: str = ""
    observed_metric_text: str = ""
    last_result_text: str = ""
    # Comparator-private carry-through: incumbent overrides, config summaries,
    # frontier archives. The controller stores it and never reads into it.
    state: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObjectiveComparator(Protocol):
    """The one interface ``objective.mode`` selects between."""

    objective_id: str
    unit_label: str

    def load_state(
        self, objective_dir: Path, status_objective_state: Mapping[str, Any]
    ) -> None:
        """Restore what previous cycles decided."""

    def check_compatible(self) -> None:
        """Raise if the restored state was produced by a different objective."""

    def dump_state(self, objective_dir: Path) -> Mapping[str, Any]:
        """Write ``objective/*`` and return what belongs in the status file."""

    def compare(self, evidence: Evidence) -> Verdict:
        """Score this cycle's evidence against everything kept so far."""

    def recommend_bases(self) -> Sequence[NodeId]:
        """Nodes worth branching from, best first."""

    def render_best(self) -> str:
        """The body of ``best_result.md``."""

    def render_progress_lines(self, verdict: Verdict) -> list[str]:
        """The comparator's share of one ``progress.md`` entry."""

    def render_recent_result_line(self, row: Mapping[str, Any]) -> str:
        """One recent-results line for the planner prompt."""

    def render_final_lines(self, verdict: Verdict) -> list[str]:
        """The comparator's share of one cycle's ``final.md``."""

    def tracking_metrics(self, verdict: Verdict) -> Mapping[str, float]:
        """Numbers this comparator wants logged, by name."""
