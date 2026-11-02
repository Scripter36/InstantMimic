"""One reader for what a trial produced.

The evaluator runs trials; this module says what they measured. It knows about
log markers and nothing about objectives: it emits the **full** benchmark curve
plus the threshold and completion events, and every comparator derives its own
number from that. Keeping the derivation out of here is what lets one evaluator
serve both objective modes.

Markers read from a training log:

* ``BENCHMARK_EVAL`` -- one curve point per evaluation, every numeric field kept.
* ``TRACKING_BENCHMARK_REACHED`` -- the frame and metrics at which the
  configured tracking benchmark was first met.
* ``TIME_TO_REWARD_THRESHOLD_SECONDS`` / ``TIME_TO_TRACKING_BENCHMARK_SECONDS``.
* ``TRAINING_COMPLETE`` -- the run's best mean reward.
* ``EVAL_BEST_CHECKPOINT`` -- metrics of the checkpoint the trainer kept.

And from a playback log: ``PLAYBACK_RESULT``, ``TRACKING_METRICS_SUMMARY``, and
``Episode N finished`` lines.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from instantmimic.hpo.common import (
    parse_keyval_payload as _common_parse_keyval_payload,
)

EVIDENCE_SCHEMA_VERSION = 1

KEYVAL_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]*)")
BENCHMARK_EVAL_RE = re.compile(r"\bBENCHMARK_EVAL\s+(.*)")
EVAL_BEST_CHECKPOINT_RE = re.compile(r"\bEVAL_BEST_CHECKPOINT\s+(.*)")
TRACKING_BENCHMARK_REACHED_RE = re.compile(r"\bTRACKING_BENCHMARK_REACHED\s+(.*)")
TRAINING_COMPLETE_RE = re.compile(
    r"TRAINING_COMPLETE.*best_mean_reward=([0-9eE+\-.]+|None)"
)
TIME_TO_REWARD_THRESHOLD_RE = re.compile(
    r"TIME_TO_REWARD_THRESHOLD_SECONDS=([0-9]+(?:\.[0-9]+)?)"
)
TIME_TO_TRACKING_BENCHMARK_RE = re.compile(
    r"TIME_TO_TRACKING_BENCHMARK_SECONDS=([0-9]+(?:\.[0-9]+)?)"
)
PLAYBACK_RESULT_RE = re.compile(r"PLAYBACK_RESULT\s+(.*)")
TRACKING_METRICS_SUMMARY_RE = re.compile(r"TRACKING_METRICS_SUMMARY\s+(.*)")
EPISODE_FINISHED_RE = re.compile(
    r"Episode\s+(\d+)\s+finished:\s+length=(\d+)\s+total_reward=([-+0-9.eE]+)"
)
WANDB_RUN_DIR_RE = re.compile(
    r"wandb: Run data is saved locally in (.*?/wandb/run-[^\s]+)"
)


def safe_float(value: str | None) -> float | None:
    if value is None or value == "None":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def safe_int(value: str | None) -> int | None:
    if value is None or value == "None":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def recent_mean(values: list[float], window: int) -> float | None:
    if not values:
        return None
    count = max(1, int(window))
    recent_values = values[-count:]
    return float(sum(recent_values) / len(recent_values))


def aggregate_values(values: list[float], method: str) -> float:
    if not values:
        raise ValueError("Cannot aggregate an empty value list.")
    if method == "median":
        result = median(values)
    elif method == "mean":
        result = mean(values)
    elif method == "min":
        result = float(min(values))
    elif method == "max":
        result = float(max(values))
    else:
        raise ValueError(f"Unsupported aggregate method: {method}")
    assert result is not None
    return float(result)


def threshold_key(threshold: float) -> str:
    numeric = float(threshold)
    if numeric.is_integer():
        return str(int(numeric))
    return ("%g" % numeric).replace(".", "p")


def parse_keyval_payload(path: Path, pattern: re.Pattern[str]) -> dict[str, str]:
    return _common_parse_keyval_payload(path, pattern, KEYVAL_RE)


def parse_keyval_payloads(path: Path, pattern: re.Pattern[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    payloads: list[dict[str, str]] = []
    for line in text.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        payloads.append(dict(KEYVAL_RE.findall(match.group(1))))
    return payloads


@dataclass(frozen=True)
class BenchmarkPoint:
    """One ``BENCHMARK_EVAL`` line, reduced to its finite numeric fields."""

    values: dict[str, float]

    def get(self, key: str) -> float | None:
        return self.values.get(key)


@dataclass
class TrainEvidence:
    """Everything a training log said, before any objective is applied."""

    benchmark_curve: list[dict[str, float]] = field(default_factory=list)
    best_checkpoint_metrics: dict[str, float] = field(default_factory=dict)
    threshold_events: dict[str, float] = field(default_factory=dict)
    reached_tracking_benchmark: dict[str, float] = field(default_factory=dict)
    training_complete: dict[str, Any] = field(default_factory=dict)
    benchmark_threshold_frames: dict[str, int] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_curve": [dict(point) for point in self.benchmark_curve],
            "best_checkpoint_metrics": dict(self.best_checkpoint_metrics),
            "threshold_events": dict(self.threshold_events),
            "reached_tracking_benchmark": dict(self.reached_tracking_benchmark),
            "training_complete": dict(self.training_complete),
            "benchmark_threshold_frames": dict(self.benchmark_threshold_frames),
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> TrainEvidence:
        """Rebuild from what `as_dict` wrote into `trials.jsonl`.

        A comparator reads trials back from disk, so the round trip has to
        exist somewhere; doing it here keeps the field list in one place.
        """

        data = dict(payload or {})
        curve = data.get("benchmark_curve", None)
        return cls(
            benchmark_curve=[
                dict(point) for point in (curve or ()) if isinstance(point, Mapping)
            ],
            best_checkpoint_metrics=dict(
                data.get("best_checkpoint_metrics", None) or {}
            ),
            threshold_events=dict(data.get("threshold_events", None) or {}),
            reached_tracking_benchmark=dict(
                data.get("reached_tracking_benchmark", None) or {}
            ),
            training_complete=dict(data.get("training_complete", None) or {}),
            benchmark_threshold_frames=dict(
                data.get("benchmark_threshold_frames", None) or {}
            ),
            summary=dict(data.get("summary", None) or {}),
        )


@dataclass
class PlaybackEvidence:
    """Everything a playback log said."""

    ape_mean: float | None = None
    ee_mean: float | None = None
    vel_mean: float | None = None
    accel_mean: float | None = None
    episode_finish_count: int = 0
    fall_count: int = 0
    terminated_count: int = 0
    truncated_count: int = 0
    payload: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ape_mean": self.ape_mean,
            "ee_mean": self.ee_mean,
            "vel_mean": self.vel_mean,
            "accel_mean": self.accel_mean,
            "episode_finish_count": self.episode_finish_count,
            "fall_count": self.fall_count,
            "terminated_count": self.terminated_count,
            "truncated_count": self.truncated_count,
            "payload": dict(self.payload),
        }


def _numeric_payload(payload: dict[str, str]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, raw_value in payload.items():
        value = safe_float(raw_value)
        if value is not None:
            numeric[key] = value
    return numeric


# The curve keeps every numeric field a line carries, `gate_*` included, so a
# comparator that prefers gate metrics can still find them. The convenience
# summary below reports the `current_*` series, which is what the running
# aggregations read.
CURVE_PREFIX = "current"


def parse_train_log(
    path: Path,
    *,
    recent_window: int = 3,
    reward_thresholds: list[float] | None = None,
) -> TrainEvidence:
    """Read one training log into objective-independent evidence."""

    evidence = TrainEvidence()
    if not path.exists():
        return evidence

    text = path.read_text(encoding="utf-8", errors="replace")
    reward_values: list[float] = []
    reward_frames: list[int] = []
    episode_length_values: list[float] = []
    success_rate_values: list[float] = []
    ape_values: list[float] = []
    ee_values: list[float] = []
    last_payload: dict[str, str] | None = None

    for line in text.splitlines():
        eval_match = BENCHMARK_EVAL_RE.search(line)
        if eval_match is not None:
            payload = dict(KEYVAL_RE.findall(eval_match.group(1)))
            evidence.benchmark_curve.append(_numeric_payload(payload))
            prefix = CURVE_PREFIX
            reward = safe_float(payload.get(f"{prefix}_reward_mean"))
            if reward is not None:
                reward_values.append(reward)
                frames = safe_int(payload.get("frames"))
                reward_frames.append(int(frames) if frames is not None else -1)
            episode_length = safe_float(payload.get(f"{prefix}_episode_length_mean"))
            if episode_length is not None:
                episode_length_values.append(episode_length)
            success_rate = safe_float(payload.get(f"{prefix}_success_rate"))
            if success_rate is not None:
                success_rate_values.append(success_rate)
            ape_mean = safe_float(payload.get(f"{prefix}_ape_mean"))
            if ape_mean is not None:
                ape_values.append(ape_mean)
            ee_mean = safe_float(payload.get(f"{prefix}_ee_mean"))
            if ee_mean is not None:
                ee_values.append(ee_mean)
            last_payload = payload
            continue

        best_match = EVAL_BEST_CHECKPOINT_RE.search(line)
        if best_match is not None:
            evidence.best_checkpoint_metrics = _numeric_payload(
                dict(KEYVAL_RE.findall(best_match.group(1)))
            )
            continue

        reached_match = TRACKING_BENCHMARK_REACHED_RE.search(line)
        if reached_match is not None:
            reached = _numeric_payload(dict(KEYVAL_RE.findall(reached_match.group(1))))
            evidence.reached_tracking_benchmark.update(reached)
            continue

        threshold_match = TIME_TO_REWARD_THRESHOLD_RE.search(line)
        if threshold_match is not None:
            evidence.threshold_events["time_to_reward_threshold_seconds"] = float(
                threshold_match.group(1)
            )
            continue

        benchmark_match = TIME_TO_TRACKING_BENCHMARK_RE.search(line)
        if benchmark_match is not None:
            evidence.threshold_events["time_to_tracking_benchmark_seconds"] = float(
                benchmark_match.group(1)
            )
            continue

        complete_match = TRAINING_COMPLETE_RE.search(line)
        if complete_match is not None:
            raw = complete_match.group(1)
            evidence.training_complete = {
                "best_mean_reward": None if raw == "None" else float(raw)
            }

    for threshold in reward_thresholds or []:
        key = threshold_key(float(threshold))
        for frame, reward in zip(reward_frames, reward_values, strict=False):
            if frame < 0:
                continue
            if float(reward) >= float(threshold):
                evidence.benchmark_threshold_frames[key] = int(frame)
                break

    total_envs = safe_int(last_payload.get("total_envs")) if last_payload else None
    success_count = (
        safe_int(last_payload.get(f"{CURVE_PREFIX}_success_count"))
        if last_payload
        else None
    )
    failure_count_last = (
        max(0, int(total_envs) - int(success_count))
        if total_envs is not None and success_count is not None
        else None
    )
    evidence.summary = {
        "benchmark_eval_count": len(evidence.benchmark_curve),
        "benchmark_reward_mean_last": reward_values[-1] if reward_values else None,
        "benchmark_reward_mean_recent": recent_mean(reward_values, recent_window),
        "benchmark_episode_length_mean_last": (
            episode_length_values[-1] if episode_length_values else None
        ),
        "benchmark_episode_length_mean_recent": recent_mean(
            episode_length_values, recent_window
        ),
        "benchmark_success_rate_last": (
            success_rate_values[-1] if success_rate_values else None
        ),
        "benchmark_success_rate_recent": recent_mean(
            success_rate_values, recent_window
        ),
        "benchmark_ape_mean_last": ape_values[-1] if ape_values else None,
        "benchmark_ape_mean_recent": recent_mean(ape_values, recent_window),
        "benchmark_ee_mean_last": ee_values[-1] if ee_values else None,
        "benchmark_ee_mean_recent": recent_mean(ee_values, recent_window),
        "benchmark_failure_count_last": failure_count_last,
    }
    return evidence


def parse_play_log(
    path: Path,
    *,
    playback_payload: dict[str, str] | None = None,
    play_steps: int | None = None,
    episode_length: int | None = None,
) -> PlaybackEvidence:
    """Read one playback log into objective-independent evidence."""

    evidence = PlaybackEvidence(payload=dict(playback_payload or {}))
    if not path.exists():
        return evidence
    text = path.read_text(encoding="utf-8", errors="replace")

    tracking_payload: dict[str, str] = {}
    for line in reversed(text.splitlines()):
        match = TRACKING_METRICS_SUMMARY_RE.search(line)
        if match is not None:
            tracking_payload = dict(KEYVAL_RE.findall(match.group(1)))
            break
    evidence.ape_mean = safe_float(tracking_payload.get("ape_mean"))
    evidence.ee_mean = safe_float(tracking_payload.get("ee_mean"))
    evidence.vel_mean = safe_float(tracking_payload.get("vel_mean"))
    evidence.accel_mean = safe_float(tracking_payload.get("accel_mean"))

    payload = evidence.payload
    terminated_count = safe_int(payload.get("terminated_count"))
    truncated_count = safe_int(payload.get("truncated_count"))
    episode_finish_count = safe_int(payload.get("episode_finish_count"))
    if terminated_count is not None or truncated_count is not None:
        terminated_count = 0 if terminated_count is None else int(terminated_count)
        truncated_count = 0 if truncated_count is None else int(truncated_count)
        if episode_finish_count is None:
            episode_finish_count = terminated_count + truncated_count
        fall_count = terminated_count
    else:
        episode_finish_count = len(list(EPISODE_FINISHED_RE.finditer(text)))
        reached_horizon = (
            play_steps is not None
            and episode_length is not None
            and episode_length >= play_steps - 1
        )
        fall_count = (
            max(0, episode_finish_count - 1)
            if reached_horizon
            else episode_finish_count
        )
        terminated_count = fall_count
        truncated_count = max(0, episode_finish_count - fall_count)

    evidence.episode_finish_count = int(episode_finish_count)
    evidence.fall_count = int(fall_count)
    evidence.terminated_count = int(terminated_count)
    evidence.truncated_count = int(truncated_count)
    return evidence


@dataclass
class TrialRecord:
    """One candidate x motion x seed x attempt, as executed."""

    candidate: str
    motion: str
    seed: int
    attempt: int
    returncode: int | None = None
    train_command: list[str] = field(default_factory=list)
    train_log: str = ""
    run_dir: str | None = None
    best_checkpoint: str | None = None
    wall_seconds: float | None = None
    effective_overrides: list[str] = field(default_factory=list)
    train: dict[str, Any] = field(default_factory=dict)
    playback: dict[str, Any] | None = None
    failure: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "motion": self.motion,
            "seed": self.seed,
            "attempt": self.attempt,
            "returncode": self.returncode,
            "train_command": list(self.train_command),
            "train_log": self.train_log,
            "run_dir": self.run_dir,
            "best_checkpoint": self.best_checkpoint,
            "wall_seconds": self.wall_seconds,
            "effective_overrides": list(self.effective_overrides),
            "train": dict(self.train),
            "playback": None if self.playback is None else dict(self.playback),
            "failure": self.failure,
        }


# -- per-motion-group evidence ------------------------------------------------
#
# A multi-motion task trains one policy on several motions and logs each one
# under `current_<group>_<metric>`. Averaging those into a single number hides
# the failure mode these tasks care about most -- improving two motions by
# sacrificing the third -- so the planner prompt renders them separately.
# `planner.motion_group_evidence` names which metric gets which statistic.
# Nothing here depends on the objective mode.

MOTION_GROUP_STATISTICS = frozenset({"terminal", "normalized_auc"})


def _cfg_is_mapping(value: object) -> bool:
    return isinstance(value, (DictConfig, dict))


def _planner_key(planner_cfg: Any, key: str) -> Any:
    """One `planner.*` value, from a `DictConfig` or a plain mapping."""

    if not _cfg_is_mapping(planner_cfg):
        return None
    return planner_cfg.get(key, None)


def parse_motion_group_evidence(planner_cfg: Any) -> dict[str, tuple[str, ...]]:
    """Read `planner.motion_group_evidence` into metric -> statistics."""

    raw = (
        None
        if planner_cfg is None
        else _planner_key(planner_cfg, "motion_group_evidence")
    )
    if raw is None:
        return {}
    if not _cfg_is_mapping(raw):
        raise ValueError("planner.motion_group_evidence must be a mapping.")
    parsed: dict[str, tuple[str, ...]] = {}
    for metric in list(raw):
        name = str(metric)
        if not name.isidentifier():
            raise ValueError(
                "planner.motion_group_evidence metric names must be identifiers."
            )
        statistics = raw.get(str(metric), None)
        if isinstance(statistics, str) or not isinstance(statistics, Iterable):
            raise ValueError(
                f"planner.motion_group_evidence.{name} must be a list of statistics."
            )
        items = tuple(str(item) for item in statistics)
        unknown = sorted(set(items) - MOTION_GROUP_STATISTICS)
        if unknown:
            raise ValueError(
                f"planner.motion_group_evidence.{name} supports only "
                f"{', '.join(sorted(MOTION_GROUP_STATISTICS))}; got: "
                f"{', '.join(unknown)}."
            )
        parsed[name] = items
    return parsed


def parse_motion_group_seed_reduce(planner_cfg: Any) -> str:
    """Read `planner.motion_group_seed_reduce`; `median` when unset."""

    raw = (
        None
        if planner_cfg is None
        else _planner_key(planner_cfg, "motion_group_seed_reduce")
    )
    reduce = str(raw or "median").strip().lower()
    if reduce not in {"mean", "median"}:
        raise ValueError("planner.motion_group_seed_reduce must be 'mean' or 'median'.")
    return reduce


def motion_group_names(
    runs: Sequence[Mapping[str, Any]], metrics: Iterable[str]
) -> list[str]:
    """Which `current_<group>_<metric>` groups the curves actually carry."""

    suffixes = [f"_{metric}" for metric in metrics]
    found: set[str] = set()
    prefix = f"{CURVE_PREFIX}_"
    for run in runs:
        for point in run.get("benchmark_curve", []) or ():
            if not isinstance(point, Mapping):
                continue
            for key in point:
                text = str(key)
                if not text.startswith(prefix):
                    continue
                for suffix in suffixes:
                    if text.endswith(suffix) and len(text) > len(prefix) + len(suffix):
                        found.add(text[len(prefix) : -len(suffix)])
    return sorted(found)


def aggregate_curve_across_seeds(
    runs: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    budget_key: str = "frames",
    seed_reduce: str = "median",
) -> list[dict[str, float]]:
    """Reduce one metric's per-seed curves to one curve, by evaluation index."""

    indexed: list[dict[int, Mapping[str, Any]]] = []
    for run in runs:
        curve: dict[int, Mapping[str, Any]] = {}
        for index, point in enumerate(run.get("benchmark_curve", []) or ()):
            if isinstance(point, Mapping) and metric_key in point:
                curve[index] = point
        if curve:
            indexed.append(curve)
    if not indexed:
        return []
    common = set(indexed[0])
    for curve in indexed[1:]:
        common &= set(curve)

    aggregated: list[dict[str, float]] = []
    for index in sorted(common):
        points = [curve[index] for curve in indexed]
        try:
            budgets = [float(point[budget_key]) for point in points]
            values = [float(point[metric_key]) for point in points]
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (*budgets, *values)):
            continue
        budget_median = median(budgets)
        if budget_median is None:
            continue
        aggregated.append(
            {
                "eval_index": float(index),
                budget_key: float(budget_median),
                "value": aggregate_values(values, seed_reduce),
            }
        )
    return aggregated


def normalized_curve_auc(
    curve: Sequence[Mapping[str, float]], *, budget_key: str = "frames"
) -> float | None:
    """Trapezoidal area under `value` over `budget_key`, divided by the span."""

    if len(curve) < 2:
        return None
    total = 0.0
    span = 0.0
    for previous, current in zip(curve, curve[1:], strict=False):
        width = float(current[budget_key]) - float(previous[budget_key])
        if width <= 0.0:
            continue
        total += 0.5 * (float(previous["value"]) + float(current["value"])) * width
        span += width
    if span <= 0.0:
        return None
    return total / span


def render_motion_group_lines(
    experiment: Mapping[str, Any],
    metric_statistics: Mapping[str, Sequence[str]],
    *,
    budget_key: str = "frames",
    seed_reduce: str = "median",
) -> list[str]:
    """One line per motion group, naming each requested statistic."""

    if not metric_statistics:
        return []
    runs = [run for run in experiment.get("runs", []) or () if isinstance(run, Mapping)]
    lines: list[str] = []
    for group in motion_group_names(runs, metric_statistics):
        fields = [f"motion_group={group}"]
        for metric, statistics in metric_statistics.items():
            curve = aggregate_curve_across_seeds(
                runs,
                metric_key=f"{CURVE_PREFIX}_{group}_{metric}",
                budget_key=budget_key,
                seed_reduce=seed_reduce,
            )
            if not curve:
                continue
            if "terminal" in statistics:
                fields.append(f"terminal_{metric}={curve[-1]['value']:.6g}")
            if "normalized_auc" in statistics:
                auc = normalized_curve_auc(curve, budget_key=budget_key)
                if auc is not None:
                    fields.append(f"{metric}_auc={auc:.6g}")
        if len(fields) > 1:
            lines.append(" ".join(fields))
    return lines
