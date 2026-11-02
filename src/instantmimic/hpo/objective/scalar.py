"""The scalar comparator family: one number, ranked against the best so far.

Every scalar strategy reduces a cycle to one value and compares it to the best
value kept so far, with a margin so noise does not count as progress and a
secondary benchmark score to break exact ties. Three outcomes:

* ``keep`` -- a new global best.
* ``advance`` -- not a global best, but better than the current incumbent
  within the same sweep, so the incumbent moves and the best result stands.
* ``discard`` -- neither.

The comparator owns the incumbent, because "what do we branch from next" is a
comparison question. It does not own Hydra: composing a candidate's effective
config is the controller's job, handed in as ``ScalarContext``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from omegaconf import DictConfig, ListConfig

from instantmimic.hpo.evidence import TrainEvidence, median
from instantmimic.hpo.experiment import NodeId
from instantmimic.hpo.objective import curves
from instantmimic.hpo.objective.base import Evidence, Verdict

# Two objective values within this distance are the same time; the secondary
# benchmark score decides between them.
METRIC_TIE_EPSILON_SECONDS = 1e-6
# An improvement smaller than this is noise, not progress.
DEFAULT_IMPROVEMENT_MARGIN_SECONDS = 0.03

# Strategy -> the objective name `curves.trial_objective` knows how to score.
# Every entry must have a branch there: an unlisted name falls through to the
# time-to-reward-threshold marker and ranks by the wrong number silently.
SCALAR_STRATEGY_OBJECTIVES = {
    "wall_clock": "wall_clock",
    "time_to_reward_threshold": "time_to_reward_threshold_excluding_warmup",
    "time_to_tracking_benchmark": "time_to_tracking_benchmark_excluding_warmup",
    "frames_to_tracking_benchmark": "frames_to_tracking_benchmark",
    "tracking_benchmark_auc": "tracking_benchmark_auc",
}


def _parse_override_value(raw_value: object) -> object:
    if isinstance(raw_value, (bool, int, float)) or raw_value is None:
        return raw_value
    if isinstance(raw_value, str):
        try:
            return yaml.safe_load(raw_value)
        except yaml.YAMLError:
            return raw_value
    return raw_value


def _extract_search_values(search_summary: object) -> list[str]:
    if not isinstance(search_summary, list) or not search_summary:
        return []
    entry = search_summary[0]
    if not isinstance(entry, dict):
        return []
    candidates = entry.get("candidates", None)
    if not isinstance(candidates, list):
        return []
    values: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        raw_value = candidate.get("value", None)
        if raw_value is None:
            continue
        values.append(str(raw_value))
    return values


def _metric_objective_label(metric_objective: str) -> str:
    if metric_objective == "tracking_benchmark_auc":
        return "TRACKING_BENCHMARK_ERROR_AUC"
    if metric_objective == "frames_to_tracking_benchmark":
        return "FRAMES_TO_TRACKING_BENCHMARK"
    if metric_objective == "time_to_tracking_benchmark_excluding_warmup":
        return "TIME_TO_TRACKING_BENCHMARK_SECONDS"
    if metric_objective == "time_to_reward_threshold_excluding_warmup":
        return "TIME_TO_REWARD_THRESHOLD_SECONDS"
    return "WALL_SECONDS"


def _metric_objective_unit_label(metric_objective: str) -> str:
    if metric_objective == "frames_to_tracking_benchmark":
        return "frames"
    if metric_objective == "tracking_benchmark_auc":
        return "score"
    return "seconds"


def _format_metric_value(metric_objective: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    unit = _metric_objective_unit_label(metric_objective)
    if unit == "seconds":
        return f"{value:.6f}s"
    return f"{value:.6f} {unit}"


def _summary_metric_objective(summary_doc: object | None) -> str:
    if not isinstance(summary_doc, dict):
        return "wall_clock"
    objective = str(summary_doc.get("objective", "") or "").strip()
    if objective:
        return objective
    if isinstance(summary_doc.get("tracking_benchmark_auc", None), (int, float)):
        return "tracking_benchmark_auc"
    if isinstance(summary_doc.get("frames_to_tracking_benchmark", None), (int, float)):
        return "frames_to_tracking_benchmark"
    if isinstance(
        summary_doc.get("time_to_tracking_benchmark_seconds", None), (int, float)
    ):
        return "time_to_tracking_benchmark_excluding_warmup"
    if isinstance(
        summary_doc.get("time_to_reward_threshold_seconds", None), (int, float)
    ):
        return "time_to_reward_threshold_excluding_warmup"
    return "wall_clock"


def _summary_objective_seconds(summary_doc: object | None) -> float | None:
    if not isinstance(summary_doc, dict):
        return None
    if isinstance(summary_doc.get("objective_seconds", None), (int, float)):
        return float(summary_doc["objective_seconds"])
    if isinstance(summary_doc.get("tracking_benchmark_auc", None), (int, float)):
        return float(summary_doc["tracking_benchmark_auc"])
    if isinstance(summary_doc.get("frames_to_tracking_benchmark", None), (int, float)):
        return float(summary_doc["frames_to_tracking_benchmark"])
    if isinstance(
        summary_doc.get("time_to_tracking_benchmark_seconds", None), (int, float)
    ):
        return float(summary_doc["time_to_tracking_benchmark_seconds"])
    if isinstance(
        summary_doc.get("time_to_reward_threshold_seconds", None), (int, float)
    ):
        return float(summary_doc["time_to_reward_threshold_seconds"])
    search_summary = summary_doc.get("search_summary", None)
    if isinstance(search_summary, list) and search_summary:
        entry = search_summary[0]
        if isinstance(entry, dict):
            winner = entry.get("winner", None)
            if isinstance(winner, dict):
                if isinstance(
                    winner.get("risk_adjusted_objective_seconds", None), (int, float)
                ):
                    return float(winner["risk_adjusted_objective_seconds"])
                if isinstance(
                    winner.get("median_objective_seconds", None), (int, float)
                ):
                    return float(winner["median_objective_seconds"])
    return None


def _summary_objective_tiebreak(summary_doc: object | None) -> float | None:
    if not isinstance(summary_doc, dict):
        return None
    if isinstance(summary_doc.get("objective_tiebreak_score", None), (int, float)):
        return float(summary_doc["objective_tiebreak_score"])
    search_summary = summary_doc.get("search_summary", None)
    if isinstance(search_summary, list) and search_summary:
        entry = search_summary[0]
        if isinstance(entry, dict):
            winner = entry.get("winner", None)
            if isinstance(winner, dict) and isinstance(
                winner.get("median_objective_tiebreak_score", None), (int, float)
            ):
                return float(winner["median_objective_tiebreak_score"])
    return None


def _summary_objective_interval_bucket(summary_doc: object | None) -> float | None:
    if not isinstance(summary_doc, dict):
        return None
    if isinstance(summary_doc.get("objective_interval_bucket", None), (int, float)):
        return float(summary_doc["objective_interval_bucket"])
    search_summary = summary_doc.get("search_summary", None)
    if isinstance(search_summary, list) and search_summary:
        entry = search_summary[0]
        if isinstance(entry, dict):
            winner = entry.get("winner", None)
            if isinstance(winner, dict) and isinstance(
                winner.get("median_objective_interval_bucket", None), (int, float)
            ):
                return float(winner["median_objective_interval_bucket"])
    return None


def _summary_benchmark_success_rate(summary_doc: object | None) -> float | None:
    if not isinstance(summary_doc, dict):
        return None
    if isinstance(summary_doc.get("benchmark_success_rate", None), (int, float)):
        return float(summary_doc["benchmark_success_rate"])
    search_summary = summary_doc.get("search_summary", None)
    if isinstance(search_summary, list) and search_summary:
        entry = search_summary[0]
        if isinstance(entry, dict):
            winner = entry.get("winner", None)
            if isinstance(winner, dict) and isinstance(
                winner.get("median_benchmark_success_rate", None), (int, float)
            ):
                return float(winner["median_benchmark_success_rate"])
    return None


def _summary_benchmark_ape_mean(summary_doc: object | None) -> float | None:
    if not isinstance(summary_doc, dict):
        return None
    if isinstance(summary_doc.get("benchmark_ape_mean", None), (int, float)):
        return float(summary_doc["benchmark_ape_mean"])
    search_summary = summary_doc.get("search_summary", None)
    if isinstance(search_summary, list) and search_summary:
        entry = search_summary[0]
        if isinstance(entry, dict):
            winner = entry.get("winner", None)
            if isinstance(winner, dict) and isinstance(
                winner.get("median_benchmark_ape_mean", None), (int, float)
            ):
                return float(winner["median_benchmark_ape_mean"])
    return None


def _summary_benchmark_global_mpjpe_mean(
    summary_doc: object | None,
) -> float | None:
    if not isinstance(summary_doc, dict):
        return None
    if isinstance(summary_doc.get("benchmark_global_mpjpe_mean", None), (int, float)):
        return float(summary_doc["benchmark_global_mpjpe_mean"])
    search_summary = summary_doc.get("search_summary", None)
    if isinstance(search_summary, list) and search_summary:
        entry = search_summary[0]
        if isinstance(entry, dict):
            winner = entry.get("winner", None)
            if isinstance(winner, dict) and isinstance(
                winner.get("median_benchmark_global_mpjpe_mean", None), (int, float)
            ):
                return float(winner["median_benchmark_global_mpjpe_mean"])
    return None


def _summary_benchmark_ee_mean(summary_doc: object | None) -> float | None:
    if not isinstance(summary_doc, dict):
        return None
    if isinstance(summary_doc.get("benchmark_ee_mean", None), (int, float)):
        return float(summary_doc["benchmark_ee_mean"])
    search_summary = summary_doc.get("search_summary", None)
    if isinstance(search_summary, list) and search_summary:
        entry = search_summary[0]
        if isinstance(entry, dict):
            winner = entry.get("winner", None)
            if isinstance(winner, dict) and isinstance(
                winner.get("median_benchmark_ee_mean", None), (int, float)
            ):
                return float(winner["median_benchmark_ee_mean"])
    return None


def _summary_metric_bucket_seconds(summary_doc: object | None) -> float | None:
    if not isinstance(summary_doc, dict):
        return None
    interval_bucket = _summary_objective_interval_bucket(summary_doc)
    if interval_bucket is None:
        return None
    interval_seconds = summary_doc.get("benchmark_eval_interval_seconds", None)
    if not isinstance(interval_seconds, (int, float)):
        return None
    return float(interval_bucket) * float(interval_seconds)


def _candidate_objective_value(candidate: dict[str, object]) -> float | None:
    for key in ("median_objective_seconds", "median_wall_seconds"):
        value = candidate.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _candidate_objective_range(candidate: dict[str, object]) -> float:
    """Spread across seeds, from the stored range or the seed results."""

    stored = candidate.get("objective_seconds_range")
    if isinstance(stored, (int, float)) and math.isfinite(float(stored)):
        return max(0.0, float(stored))
    seed_results = candidate.get("seed_results")
    if not isinstance(seed_results, list):
        return 0.0
    values = [
        float(item["objective_seconds"])
        for item in seed_results
        if isinstance(item, dict)
        and isinstance(item.get("objective_seconds", None), (int, float))
    ]
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def _candidate_risk_adjusted_objective(
    candidate: dict[str, object], *, range_penalty_lambda: float
) -> float | None:
    """Penalize seed spread so an unstable fast candidate does not win."""

    objective_value = _candidate_objective_value(candidate)
    if objective_value is None:
        return None
    if range_penalty_lambda <= 0.0:
        return objective_value
    stored = candidate.get("risk_adjusted_objective_seconds")
    if isinstance(stored, (int, float)) and math.isfinite(float(stored)):
        return float(stored)
    return objective_value + (
        range_penalty_lambda * _candidate_objective_range(candidate)
    )


def _select_scalar_winner(
    valid_candidates: list[dict[str, object]],
    *,
    range_penalty_lambda: float = 0.0,
) -> dict[str, object]:
    """Lowest risk-adjusted objective, then tiebreak score, then value text."""

    if not valid_candidates:
        raise ValueError("valid_candidates must not be empty")

    def sort_key(candidate: dict[str, object]) -> tuple[float, float, str]:
        objective_value = _candidate_risk_adjusted_objective(
            candidate, range_penalty_lambda=range_penalty_lambda
        )
        tiebreak = candidate.get("median_objective_tiebreak_score")
        return (
            round(objective_value if objective_value is not None else float("inf"), 6),
            float(tiebreak) if isinstance(tiebreak, (int, float)) else float("inf"),
            str(candidate.get("value", "")),
        )

    return min(valid_candidates, key=sort_key)


# -- scoring: evidence -> this objective's number ------------------------------
#
# The evaluator measures and never decides, so the number this family ranks by
# does not exist until here. `curves.trial_objective` reduces one trial; this
# reduces the seeds of one candidate, and the candidates of one cycle, into the
# scored document `outcome` compares. Without it the comparator reads a
# document the evaluator does not write and calls every cycle a crash.


def _trial_failed(trial: Mapping[str, Any]) -> bool:
    if trial.get("failure", None):
        return True
    returncode = trial.get("returncode", None)
    return isinstance(returncode, int) and returncode != 0


def _score_one_trial(
    trial: Mapping[str, Any],
    objective: ScalarObjective,
    *,
    horizon_seconds: float | None,
    interval_seconds: float | None,
) -> curves.TrialObjective:
    wall_raw = trial.get("wall_seconds", None)
    wall_seconds = float(wall_raw) if isinstance(wall_raw, (int, float)) else None
    return curves.trial_objective(
        TrainEvidence.from_dict(trial.get("train", None)),
        objective_name=objective.metric_objective,
        thresholds=objective.thresholds,
        threshold_mode=objective.threshold_mode,
        horizon_seconds=horizon_seconds,
        wall_seconds=wall_seconds,
        interval_seconds=interval_seconds,
    )


def _score_candidate(
    name: str,
    trials: Sequence[Mapping[str, Any]],
    objective: ScalarObjective,
    *,
    horizon_seconds: float | None,
    interval_seconds: float | None,
) -> dict[str, Any]:
    """One candidate's seeds, reduced to the shape the ranking helpers read."""

    seed_results: list[dict[str, Any]] = []
    values: list[float] = []
    tiebreaks: list[float] = []
    buckets: list[float] = []
    incomplete: list[str] = []
    for trial in trials:
        if _trial_failed(trial):
            incomplete.append(str(trial.get("failure", "") or "trial failed"))
            continue
        scored = _score_one_trial(
            trial,
            objective,
            horizon_seconds=horizon_seconds,
            interval_seconds=interval_seconds,
        )
        if not scored.complete or scored.value is None:
            incomplete.append(scored.reason or "trial produced no objective value")
            continue
        values.append(float(scored.value))
        if scored.tiebreak is not None:
            tiebreaks.append(float(scored.tiebreak))
        if scored.interval_bucket is not None:
            buckets.append(float(scored.interval_bucket))
        seed_results.append(
            {
                "seed": trial.get("seed", None),
                "motion": trial.get("motion", None),
                "objective_seconds": float(scored.value),
            }
        )

    candidate: dict[str, Any] = {
        "value": name,
        "seed_results": seed_results,
        # A candidate whose seeds did not all produce a number is not ranked:
        # comparing a 2-seed median against a 3-seed one is not a comparison.
        "valid": bool(values) and not incomplete,
    }
    if incomplete:
        candidate["incomplete_reason"] = incomplete[0]
    if values:
        candidate["median_objective_seconds"] = float(median(values))
        candidate["objective_seconds_range"] = float(max(values) - min(values))
    if tiebreaks:
        candidate["median_objective_tiebreak_score"] = float(median(tiebreaks))
    if buckets:
        candidate["objective_interval_bucket"] = float(median(buckets))
    return candidate


def score_evidence(
    evidence: Evidence,
    objective: ScalarObjective,
    *,
    horizon_seconds: float | None = None,
    interval_seconds: float | None = None,
) -> dict[str, Any] | None:
    """Score one cycle's trials, or ``None`` when nothing is rankable.

    The result is this comparator's own scored document -- the candidates it
    may rank and the winner among them -- not anything the evaluator writes.
    """

    by_candidate: dict[str, list[Mapping[str, Any]]] = {}
    for trial in evidence.trials:
        if not isinstance(trial, Mapping):
            continue
        by_candidate.setdefault(str(trial.get("candidate", "") or ""), []).append(trial)
    # A cycle with no trial records has nothing to score; say so rather than
    # inventing a number.
    if not by_candidate:
        return None

    ordered = [name for name in evidence.candidate_names() if name in by_candidate]
    ordered.extend(name for name in by_candidate if name not in ordered)
    candidates = [
        _score_candidate(
            name,
            by_candidate[name],
            objective,
            horizon_seconds=horizon_seconds,
            interval_seconds=interval_seconds,
        )
        for name in ordered
    ]
    valid = [candidate for candidate in candidates if candidate.get("valid")]
    if not valid:
        return None

    winner = _select_scalar_winner(
        valid, range_penalty_lambda=objective.range_penalty_lambda
    )
    winner = dict(winner)
    risk_adjusted = _candidate_risk_adjusted_objective(
        winner, range_penalty_lambda=objective.range_penalty_lambda
    )
    if risk_adjusted is not None:
        winner["risk_adjusted_objective_seconds"] = float(risk_adjusted)

    scored: dict[str, Any] = {
        "objective": objective.metric_objective,
        "objective_seconds": winner.get(
            "risk_adjusted_objective_seconds", winner.get("median_objective_seconds")
        ),
        "search_summary": [
            {
                "param": "experiments" if len(candidates) > 1 else "candidate",
                "candidates": candidates,
                "winner": winner,
            }
        ],
    }
    tiebreak = winner.get("median_objective_tiebreak_score")
    if isinstance(tiebreak, (int, float)):
        scored["objective_tiebreak"] = float(tiebreak)
    bucket = winner.get("objective_interval_bucket")
    if isinstance(bucket, (int, float)):
        scored["objective_interval_bucket"] = float(bucket)
    return scored


# The weighted-sum and reward-threshold-ladder strategies score an
# aggregate directly rather than a benchmark curve; they are scalar
# comparators too, and this is their scoring half.


@dataclass(frozen=True)
class ScalarObjective:
    """What this comparator ranks by, and how strictly."""

    metric_objective: str = "wall_clock"
    improvement_margin: float = DEFAULT_IMPROVEMENT_MARGIN_SECONDS
    range_penalty_lambda: float = 0.0
    # Off, and it cannot be switched on: see `from_config`.
    local_incumbent_advance: bool = False
    thresholds: curves.Thresholds = field(default_factory=curves.Thresholds)
    threshold_mode: str = "threshold"
    strategy: str = "wall_clock"

    def __post_init__(self) -> None:
        # A negative margin would make a worse result count as an improvement.
        _non_negative(self.improvement_margin, "objective.improvement_margin")
        _non_negative(self.range_penalty_lambda, "objective.range_penalty_lambda")

    @classmethod
    def from_config(cls, objective_cfg: Mapping[str, Any] | None) -> ScalarObjective:
        cfg = objective_cfg or {}
        strategy = str(cfg.get("strategy", "wall_clock") or "wall_clock")
        if strategy not in SCALAR_STRATEGY_OBJECTIVES:
            raise ValueError(
                f"objective.strategy '{strategy}' has no scalar scorer; "
                f"expected one of: {', '.join(sorted(SCALAR_STRATEGY_OBJECTIVES))}."
            )
        thresholds = curves.Thresholds.from_objective(cfg)
        margin = cfg.get("improvement_margin", None)
        return cls(
            metric_objective=SCALAR_STRATEGY_OBJECTIVES[strategy],
            improvement_margin=(
                DEFAULT_IMPROVEMENT_MARGIN_SECONDS
                if margin in (None, "")
                else float(margin)
            ),
            range_penalty_lambda=float(cfg.get("range_penalty_lambda", 0.0) or 0.0),
            local_incumbent_advance=_refuse_advance(
                cfg.get("local_incumbent_advance", False)
            ),
            thresholds=thresholds,
            threshold_mode=str(cfg.get("threshold_mode", "threshold") or "threshold"),
            strategy=strategy,
        )


def _refuse_advance(value: object) -> bool:
    """`advance` needs the incumbent re-measured in the same cycle.

    The scalar sweep used to re-run the incumbent's own value as one of its
    candidates, so "the winner beat the incumbent" was a within-cycle fact. The
    unified model inherits the base through per-motion effective overrides and
    never re-runs it, and a proposal may not repeat an override set an earlier
    cycle already ran -- so there is nothing in a cycle to compare the
    incumbent against. Deciding what `advance` should mean instead is a design
    question, not a defect to patch over, so asking for it is an error rather
    than a silently ignored flag.
    """

    if bool(value):
        raise ValueError(
            "objective.local_incumbent_advance is not available: the `advance` "
            "decision compared a sweep winner against the incumbent re-run in "
            "the same cycle, and the unified cycle inherits its base instead of "
            "re-running it. Remove the key or set it to false; a candidate is "
            "compared against the global best."
        )
    return False


def _non_negative(value: float, label: str) -> float:
    if value < 0.0:
        raise ValueError(f"'{label}' must be >= 0.")
    return value


def _node_from_status(raw: object) -> NodeId | None:
    """Read back what `ScalarState.as_status` wrote for `best_node`."""

    if isinstance(raw, (list, tuple, ListConfig)) and len(raw) == 2:
        cycle, name = raw
        try:
            return (int(cycle), str(name))
        except (TypeError, ValueError):
            return None
    return None


@dataclass
class ScalarState:
    """What previous cycles decided, and what this one may change."""

    best_value: float | None = None
    best_tiebreak: float | None = None
    best_config_summary: str = ""
    best_node: NodeId | None = None
    incumbent_overrides: dict[str, str] = field(default_factory=dict)
    incumbent_config_summary: str = ""
    metric_objective: str = ""

    @classmethod
    def from_status(cls, payload: Mapping[str, Any] | None) -> ScalarState:
        data = dict(payload or {})
        best_value = data.get("best_metric_seconds", None)
        tiebreak = data.get("best_metric_tiebreak", None)
        overrides = data.get("incumbent_overrides", None)
        return cls(
            best_value=(None if best_value in (None, "") else float(best_value)),
            best_tiebreak=None if tiebreak in (None, "") else float(tiebreak),
            best_config_summary=str(data.get("best_config_summary", "") or ""),
            incumbent_overrides=(
                {str(key): str(value) for key, value in dict(overrides).items()}
                if isinstance(overrides, (dict, DictConfig))
                else {}
            ),
            incumbent_config_summary=str(
                data.get("incumbent_config_summary", "") or ""
            ),
            metric_objective=str(data.get("metric_objective", "") or ""),
            best_node=_node_from_status(data.get("best_node", None)),
        )

    def as_status(self) -> dict[str, Any]:
        return {
            "best_metric_seconds": self.best_value,
            "best_metric_tiebreak": self.best_tiebreak,
            "best_config_summary": self.best_config_summary,
            "incumbent_overrides": dict(self.incumbent_overrides),
            "incumbent_config_summary": self.incumbent_config_summary,
            "metric_objective": self.metric_objective,
            # The node the planner is offered as a base. A fresh comparator is
            # built every cycle and loaded from this payload, so leaving it out
            # made `recommend_bases()` permanently empty and turned
            # `base: "recommended"` into a silent `base: "root"`.
            "best_node": (
                None
                if self.best_node is None
                else [int(self.best_node[0]), str(self.best_node[1])]
            ),
        }


@dataclass(frozen=True)
class ScalarContext:
    """How an override set reads to a human.

    It used to also compose a candidate's effective Hydra config and resolve one
    path's effective value, for the `advance` decision. Nothing injected those,
    and `advance` is refused now, so what is left is the rendering.
    """

    config_summary: Callable[[dict[str, str]], str] = lambda overrides: ", ".join(
        f"{key}={value}" for key, value in overrides.items()
    )


class ScalarComparator:
    """Rank one cycle's sweep against the best result kept so far."""

    unit_label = "seconds"

    def __init__(
        self,
        objective: ScalarObjective,
        *,
        state_dir: Path | None = None,
        context: ScalarContext | None = None,
        state: ScalarState | None = None,
    ) -> None:
        # Both comparator families take the same arguments so the controller
        # never has to ask which one it is building.
        self.state_dir = state_dir
        self.objective = objective
        self.context = context or ScalarContext()
        self.state = state or ScalarState(metric_objective=objective.metric_objective)

    @property
    def objective_id(self) -> str:
        return self.objective.metric_objective

    # -- state ---------------------------------------------------------------

    def load_state(
        self, objective_dir: Path, status_objective_state: Mapping[str, Any]
    ) -> None:
        del objective_dir
        self.state = ScalarState.from_status(status_objective_state)

    def check_compatible(self) -> None:
        stored = self.state.metric_objective
        if stored and stored != self.objective.metric_objective:
            raise ValueError(
                "Scalar objective changed while resuming "
                f"('{stored}' -> '{self.objective.metric_objective}'); "
                "use a fresh hpo state."
            )

    def dump_state(self, objective_dir: Path) -> Mapping[str, Any]:
        payload = self.state.as_status()
        objective_dir.mkdir(parents=True, exist_ok=True)
        (objective_dir / "incumbent.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return payload

    def recommend_bases(self) -> Sequence[NodeId]:
        return () if self.state.best_node is None else (self.state.best_node,)

    # -- comparison ----------------------------------------------------------

    def outcome(
        self,
        *,
        summary_doc: object | None,
        prepared_incumbent_overrides: dict[str, str],
        evidence_text: str = "",
        compact_error_excerpt: str = "",
        failed: bool = False,
    ) -> dict[str, object]:
        """Compare one sweep's result to the best kept so far.

        Returns the decision plus the text that describes it. The controller
        stores the result; it does not re-derive any of it.
        """
        current_metric_objective = self.objective.metric_objective
        previous_metric_objective = (
            self.state.metric_objective or self.objective.metric_objective
        )
        previous_best_raw = (
            self.state.best_value
            if previous_metric_objective == current_metric_objective
            else None
        )
        previous_best_tiebreak_raw = (
            self.state.best_tiebreak
            if previous_metric_objective == current_metric_objective
            else None
        )
        previous_best = (
            float(previous_best_raw) if previous_best_raw not in (None, "") else None
        )
        previous_best_tiebreak = (
            float(previous_best_tiebreak_raw)
            if previous_best_tiebreak_raw not in (None, "")
            else None
        )
        previous_incumbent_overrides = dict(self.state.incumbent_overrides)
        previous_incumbent_config_summary = (
            self.state.incumbent_config_summary
            or self.context.config_summary(previous_incumbent_overrides)
        )
        outcome: dict[str, object] = {
            "metric_seconds": 0.0,
            "status": "crash",
            "description": "cycle crashed before producing a valid summary",
            "observed_metric_text": f"crash - {compact_error_excerpt}",
            "delta_text": "N/A (crash)",
            "decision_text": "discard - restored the pre-cycle experiment files after the crash",
            "last_result_text": f"crash - {compact_error_excerpt}",
            "evidence_text": evidence_text,
            "best_config_summary": self.state.best_config_summary,
            "incumbent_config_summary": previous_incumbent_config_summary,
            "incumbent_overrides": previous_incumbent_overrides,
            "should_keep": False,
        }

        if failed or not isinstance(summary_doc, dict):
            return outcome

        metric: float | None = None
        description = "baseline run with current config"
        observed_metric_text = ""
        summary_metric_objective = _summary_metric_objective(summary_doc)
        metric_label = _metric_objective_label(summary_metric_objective)

        objective_seconds = _summary_objective_seconds(summary_doc)
        objective_interval_bucket = _summary_objective_interval_bucket(summary_doc)
        objective_bucket_seconds = _summary_metric_bucket_seconds(summary_doc)
        objective_tiebreak = _summary_objective_tiebreak(summary_doc)
        benchmark_success_rate = _summary_benchmark_success_rate(summary_doc)
        benchmark_ape_mean = _summary_benchmark_ape_mean(summary_doc)
        benchmark_global_mpjpe_mean = _summary_benchmark_global_mpjpe_mean(summary_doc)
        benchmark_ee_mean = _summary_benchmark_ee_mean(summary_doc)
        search_summary = summary_doc.get("search_summary", None)
        if isinstance(search_summary, list) and search_summary:
            entry = search_summary[0]
            if isinstance(entry, dict):
                param_name = str(entry.get("param", "parameter"))
                winner = entry.get("winner", None)
                values = _extract_search_values(search_summary)
                values_text = ",".join(values) if values else "(unknown values)"
                winner_metric = (
                    float(winner["risk_adjusted_objective_seconds"])
                    if isinstance(winner, dict)
                    and isinstance(
                        winner.get("risk_adjusted_objective_seconds", None),
                        (int, float),
                    )
                    else (
                        float(winner["median_objective_seconds"])
                        if isinstance(winner, dict)
                        and isinstance(
                            winner.get("median_objective_seconds", None), (int, float)
                        )
                        else (
                            float(winner["median_wall_seconds"])
                            if isinstance(winner, dict)
                            and isinstance(
                                winner.get("median_wall_seconds", None), (int, float)
                            )
                            else None
                        )
                    )
                )
                if winner_metric is not None:
                    metric = winner_metric
                    winner_value = str(winner.get("value", "(unknown)"))
                    description = f"{param_name} line search over {values_text}"
                    median_metric = (
                        float(winner["median_objective_seconds"])
                        if isinstance(winner, dict)
                        and isinstance(
                            winner.get("median_objective_seconds", None), (int, float)
                        )
                        else None
                    )
                    range_metric = (
                        float(winner["objective_seconds_range"])
                        if isinstance(winner, dict)
                        and isinstance(
                            winner.get("objective_seconds_range", None), (int, float)
                        )
                        else None
                    )
                    if median_metric is not None and range_metric is not None:
                        observed_metric_text = (
                            f"winner {param_name}={winner_value} with risk-adjusted "
                            f"{metric_label} = {_format_metric_value(summary_metric_objective, metric)} "
                            f"(median={_format_metric_value(summary_metric_objective, median_metric)}, "
                            f"range={_format_metric_value(summary_metric_objective, range_metric)})."
                        )
                    else:
                        observed_metric_text = (
                            f"winner {param_name}={winner_value} with {metric_label} = "
                            f"{_format_metric_value(summary_metric_objective, metric)}."
                        )

        if metric is None and objective_seconds is not None:
            metric = objective_seconds
            repeats = summary_doc.get("runs", None)
            repeat_values: list[str] = []
            if isinstance(repeats, list):
                for run in repeats:
                    if not isinstance(run, dict):
                        continue
                    if isinstance(run.get("objective_seconds", None), (int, float)):
                        repeat_values.append(
                            _format_metric_value(
                                summary_metric_objective,
                                float(run["objective_seconds"]),
                            )
                        )
                        continue
                    if isinstance(
                        run.get("time_to_tracking_benchmark_seconds", None),
                        (int, float),
                    ):
                        repeat_values.append(
                            _format_metric_value(
                                summary_metric_objective,
                                float(run["time_to_tracking_benchmark_seconds"]),
                            )
                        )
                        continue
                    if isinstance(
                        run.get("time_to_reward_threshold_seconds", None), (int, float)
                    ):
                        repeat_values.append(
                            _format_metric_value(
                                summary_metric_objective,
                                float(run["time_to_reward_threshold_seconds"]),
                            )
                        )
            if repeat_values:
                observed_metric_text = (
                    f"median {metric_label} = {_format_metric_value(summary_metric_objective, metric)} from repeats "
                    + ", ".join(repeat_values)
                    + "."
                )
            else:
                observed_metric_text = (
                    f"{metric_label} = "
                    f"{_format_metric_value(summary_metric_objective, metric)}."
                )

        if metric is None or not math.isfinite(metric) or metric < 0.0:
            return outcome

        metric_improvement_margin = self.objective.improvement_margin
        metric_better = (
            previous_best is None or metric < previous_best - metric_improvement_margin
        )
        metric_tied = (
            previous_best is not None
            and abs(metric - previous_best) <= METRIC_TIE_EPSILON_SECONDS
        )
        tiebreak_better = (
            metric_tied
            and objective_tiebreak is not None
            and (
                previous_best_tiebreak is None
                or objective_tiebreak < previous_best_tiebreak
            )
        )
        should_keep = previous_best is None or metric_better or tiebreak_better
        # `advance` is refused at configuration time, so a cycle is either a new
        # global best or a discard.
        local_advance = None
        should_advance_incumbent = False
        selected_overrides = (
            {
                str(key): str(value)
                for key, value in dict(
                    summary_doc.get("selected_overrides", {})
                ).items()
            }
            if isinstance(summary_doc, dict)
            and isinstance(summary_doc.get("selected_overrides", None), dict)
            else {}
        )
        next_incumbent_overrides = (
            {
                **prepared_incumbent_overrides,
                **selected_overrides,
            }
            if should_keep or should_advance_incumbent
            else previous_incumbent_overrides
        )
        next_incumbent_config_summary = (
            self.context.config_summary(next_incumbent_overrides)
            if should_keep or should_advance_incumbent
            else previous_incumbent_config_summary
        )
        if (
            previous_metric_objective != current_metric_objective
            and self.state.best_value is not None
        ):
            delta_text = (
                "N/A (metric objective changed from "
                f"{previous_metric_objective} to {current_metric_objective})."
            )
        else:
            delta_text = (
                "N/A (first valid recorded cycle)"
                if previous_best is None
                else (
                    "better by "
                    f"{_format_metric_value(current_metric_objective, previous_best - metric)} "
                    "versus the previous best "
                    f"({_format_metric_value(current_metric_objective, previous_best)})."
                    if metric_better
                    else (
                        (
                            "same primary time, but better secondary benchmark error score "
                            f"({objective_tiebreak:.6f} vs {previous_best_tiebreak:.6f})."
                        )
                        if tiebreak_better
                        and objective_tiebreak is not None
                        and previous_best_tiebreak is not None
                        else (
                            (
                                f"better than incumbent {local_advance['param_name']}="
                                f"{local_advance['incumbent_value']} inside the sweep, "
                                "but worse by "
                                f"{_format_metric_value(current_metric_objective, metric - previous_best)} "
                                "versus the previous best "
                                f"({_format_metric_value(current_metric_objective, previous_best)})."
                            )
                            if should_advance_incumbent
                            and local_advance is not None
                            and previous_best is not None
                            else (
                                "worse by "
                                f"{_format_metric_value(current_metric_objective, metric - previous_best)} "
                                "versus the previous best "
                                f"({_format_metric_value(current_metric_objective, previous_best)})."
                            )
                        )
                    )
                )
            )
        decision_text = (
            "keep - the cycle produced a new best result"
            if should_keep
            else (
                (
                    "advance - promoted the local sweep winner into the incumbent because it "
                    "beat the current incumbent inside the same sweep, while preserving the "
                    "previous global best result"
                )
                if should_advance_incumbent
                else "discard - restored the pre-cycle experiment files because the result was not better than the current best"
            )
        )
        last_result_text = (
            (
                f"keep - new best metric {_format_metric_value(current_metric_objective, metric)}"
                if metric_better or previous_best is None
                else (
                    "keep - matched the best primary time and improved the secondary benchmark error score"
                )
            )
            if should_keep
            else (
                (
                    f"advance - promoted local incumbent {local_advance['param_name']}="
                    f"{local_advance['winner_value']} at "
                    f"{_format_metric_value(current_metric_objective, metric)} while global best remains "
                    f"{_format_metric_value(current_metric_objective, previous_best)}"
                )
                if should_advance_incumbent and local_advance is not None
                else (
                    "discard - metric "
                    f"{_format_metric_value(current_metric_objective, metric)} was not better than the best "
                    f"{_format_metric_value(current_metric_objective, previous_best)}"
                )
            )
        )
        if (
            isinstance(summary_doc.get("search_summary", None), list)
            and summary_doc["search_summary"]
        ):
            description = (
                f"{description} kept {summary_doc['search_summary'][0].get('winner', {}).get('value', '(unknown)')}"
                if should_keep
                else (
                    (
                        f"{description} advanced incumbent to {local_advance['winner_value']} while preserving the global best"
                    )
                    if should_advance_incumbent and local_advance is not None
                    else f"{description} discarded as worse than best"
                )
            )
        elif not should_keep:
            description = (
                "baseline run with current config advanced the incumbent while preserving the best result"
                if should_advance_incumbent
                else "baseline run with current config discarded as worse than best"
            )

        outcome.update(
            {
                "metric_seconds": metric,
                "metric_interval_bucket": objective_interval_bucket,
                "metric_bucket_seconds": objective_bucket_seconds,
                "metric_tiebreak": objective_tiebreak,
                "benchmark_success_rate": benchmark_success_rate,
                "benchmark_ape_mean": benchmark_ape_mean,
                "benchmark_global_mpjpe_mean": benchmark_global_mpjpe_mean,
                "benchmark_ee_mean": benchmark_ee_mean,
                "status": "keep"
                if should_keep
                else "advance"
                if should_advance_incumbent
                else "discard",
                "description": description,
                "observed_metric_text": observed_metric_text,
                "delta_text": delta_text,
                "decision_text": decision_text,
                "last_result_text": last_result_text,
                "best_config_summary": self.context.config_summary(
                    next_incumbent_overrides
                )
                if should_keep
                else self.state.best_config_summary,
                "incumbent_config_summary": next_incumbent_config_summary,
                "incumbent_overrides": next_incumbent_overrides,
                "should_keep": should_keep,
            }
        )
        return outcome

    def compare(self, evidence: Evidence) -> Verdict:
        """Decide keep / advance / discard for one cycle's evidence.

        The number being compared is derived here, from the trial records the
        evaluator wrote. The evaluator reports execution facts and never a
        score, so scoring on the way in is the whole job.
        """

        scored = (
            None
            if evidence.lifecycle != "completed"
            else score_evidence(evidence, self.objective)
        )
        outcome = self.outcome(
            summary_doc=scored,
            prepared_incumbent_overrides=dict(self.state.incumbent_overrides),
            evidence_text=(
                "" if evidence.evidence_path is None else str(evidence.evidence_path)
            ),
            compact_error_excerpt=(
                evidence.error
                or (
                    ""
                    if scored is not None
                    else "no candidate produced a rankable objective value"
                )
            ),
            failed=evidence.lifecycle != "completed",
        )
        status = str(outcome["status"])
        if status in {"keep", "advance"}:
            self.state.incumbent_overrides = dict(outcome["incumbent_overrides"])
            self.state.incumbent_config_summary = str(
                outcome["incumbent_config_summary"]
            )
            self.state.best_node = (evidence.cycle, _winner_name(evidence, scored))
        if status == "keep":
            self.state.best_value = float(outcome["metric_seconds"])
            self.state.best_tiebreak = outcome.get("metric_tiebreak")
            self.state.best_config_summary = str(outcome["best_config_summary"])
        self.state.metric_objective = self.objective.metric_objective
        return Verdict(
            decision=status,
            improved=status == "keep",
            objective_value=(
                None if status == "crash" else float(outcome["metric_seconds"])
            ),
            representative={
                "candidate": _winner_name(evidence, scored),
                "value": outcome["metric_seconds"],
                "config_summary": outcome["incumbent_config_summary"],
            },
            per_candidate=_per_candidate_metrics(
                scored, self.objective.range_penalty_lambda
            ),
            recommended_bases=self.recommend_bases(),
            description=str(outcome["description"]),
            delta_text=str(outcome["delta_text"]),
            decision_text=str(outcome["decision_text"]),
            observed_metric_text=str(outcome["observed_metric_text"]),
            last_result_text=str(outcome["last_result_text"]),
            state=outcome,
        )

    # -- rendering -----------------------------------------------------------

    def render_best(self) -> str:
        value = _format_metric_value(
            self.objective.metric_objective, self.state.best_value
        )
        return "\n".join(
            [
                f"- Objective: `{self.objective.metric_objective}`",
                f"- Best {_metric_objective_label(self.objective.metric_objective)}: "
                f"`{value}`",
                f"- Config: `{self.state.best_config_summary or '(none)'}`",
                f"- Incumbent: `{self.state.incumbent_config_summary or '(none)'}`",
            ]
        )

    def render_progress_lines(self, verdict: Verdict) -> list[str]:
        return [
            f"- Observed metric: {verdict.observed_metric_text or 'N/A'}",
            f"- Delta vs previous best: {verdict.delta_text or 'N/A'}",
            f"- Decision: {verdict.decision_text or verdict.decision}",
        ]

    def render_recent_result_line(self, row: Mapping[str, Any]) -> str:
        value = row.get("objective_value", "")
        return (
            f"- cycle {row.get('cycle', '?')}: {row.get('decision', row.get('status', ''))}"
            f" {_format_metric_value(self.objective.metric_objective, _optional(value))}"
        )

    def render_final_lines(self, verdict: Verdict) -> list[str]:
        return [
            f"- Result: {verdict.last_result_text or verdict.decision}",
            f"- Observed metric: {verdict.observed_metric_text or 'N/A'}",
        ]

    def tracking_metrics(self, verdict: Verdict) -> Mapping[str, float]:
        metrics: dict[str, float] = {}
        if verdict.objective_value is not None:
            metrics["objective_value"] = float(verdict.objective_value)
        for key in (
            "metric_tiebreak",
            "benchmark_success_rate",
            "benchmark_ape_mean",
            "benchmark_global_mpjpe_mean",
            "benchmark_ee_mean",
        ):
            value = verdict.state.get(key)
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        return metrics


def _optional(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _winner_name(evidence: Evidence, scored: Mapping[str, Any] | None = None) -> str:
    search_summary = (scored or {}).get("search_summary", None)
    if isinstance(search_summary, list) and search_summary:
        winner = search_summary[0].get("winner", None)
        if isinstance(winner, dict):
            return str(winner.get("value", ""))
    names = evidence.candidate_names()
    return names[0] if names else ""


def _per_candidate_metrics(
    summary: Mapping[str, Any] | None,
    range_penalty_lambda: float = 0.0,
) -> dict[str, dict[str, Any]]:
    search_summary = (summary or {}).get("search_summary", None)
    if not isinstance(search_summary, list) or not search_summary:
        return {}
    candidates = search_summary[0].get("candidates", None)
    if not isinstance(candidates, list):
        return {}
    return {
        str(candidate.get("value", index)): {
            "objective_value": (
                _candidate_risk_adjusted_objective(
                    candidate, range_penalty_lambda=range_penalty_lambda
                )
                if candidate.get("valid")
                else None
            ),
            "tiebreak": candidate.get("median_objective_tiebreak_score"),
            "range": candidate.get("objective_seconds_range"),
        }
        for index, candidate in enumerate(candidates)
        if isinstance(candidate, dict)
    }
