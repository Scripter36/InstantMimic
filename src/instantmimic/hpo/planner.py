"""The planner protocol: two agent turns, one prompt envelope.

A cycle asks the agent twice. The first turn writes ``strategy.md`` -- what the
search believes right now. The second turn proposes the experiment, as JSON the
controller converts into a ``Proposal``. Splitting them is what gives the
controller a checkpoint between "what do we think" and "what shall we run", and
what lets a failed proposal be re-fed with the validator's own words instead of
starting the cycle over.

The agent never edits repository files. It answers; the controller writes
everything. That is why the sandbox can be read-only.

One prompt envelope serves both objective modes. Every section below is
mode-independent except the three the comparator renders: the best result so
far, the bases worth branching from, and how a past result line reads.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from omegaconf import DictConfig, OmegaConf

from instantmimic.hpo import evidence
from instantmimic.hpo.agent_runtime import (
    AgentSpec,
    agent_label,
    resolve_agent_spec,
    run_agent,
)
from instantmimic.hpo.experiment import Axis, RootSpec
from instantmimic.hpo.objective.base import ObjectiveComparator
from instantmimic.hpo.schema import (
    DEFAULT_PLANNER_RETRY_DELAY_SECONDS,
    DEFAULT_PLANNER_TIMEOUT_SECONDS,
    DEFAULT_REASONING_EFFORT,
)

STRATEGY_HEADINGS = (
    "# Current Strategy",
    "## Best Baseline",
    "## What Seems Promising",
    "## What To Avoid",
    "## Next Experiment",
    "## Wildcards",
)

_AGENT_TIMEOUT_RETURNCODE = 124
_OLLAMA_START_TIMEOUT_SECONDS = 60.0
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n(.*)\n```$", re.DOTALL)


@dataclass(frozen=True)
class PlannerConfig:
    """Everything ``planner.*`` says about how the agent is asked."""

    timeout_seconds: float = DEFAULT_PLANNER_TIMEOUT_SECONDS
    timeout_retries: int = 1
    retry_delay_seconds: float = DEFAULT_PLANNER_RETRY_DELAY_SECONDS
    recent_cycles: int = 8
    instructions: str = ""
    initial_candidates: tuple[Mapping[str, Any], ...] = ()
    max_same_dimension_streak: int = 2
    deep_review_interval_cycles: int = 8
    deep_review_results_rows: int = 30
    creativity_interval_cycles: int = 5
    creativity_plateau_cycles: int = 4
    search_base_selection: bool = True
    search_base_max_options: int = 12
    # metric -> statistics, for a multi-motion task that must not trade one
    # motion off against another. Empty means "one aggregate line per cycle".
    motion_group_evidence: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    motion_group_seed_reduce: str = "median"
    auto_start_ollama: bool = True
    host: str = ""
    context_length: int | None = None

    @classmethod
    def from_task(cls, task_cfg: DictConfig) -> PlannerConfig:
        cfg = task_cfg.get("planner", None) or {}
        deep_review = cfg.get("deep_review", None) or {}
        creativity = cfg.get("creativity", None) or {}
        base_selection = cfg.get("search_base_selection", None) or {}
        codex_config = cfg.get("codex_config", None) or {}
        return cls(
            timeout_seconds=float(
                cfg.get("timeout_seconds", DEFAULT_PLANNER_TIMEOUT_SECONDS)
            ),
            timeout_retries=max(0, int(cfg.get("timeout_retries", 1) or 0)),
            retry_delay_seconds=float(
                cfg.get("retry_delay_seconds", DEFAULT_PLANNER_RETRY_DELAY_SECONDS)
            ),
            recent_cycles=max(1, int(cfg.get("recent_cycles", 8) or 8)),
            instructions=str(cfg.get("instructions", "") or "").strip(),
            initial_candidates=tuple(
                OmegaConf.to_container(item, resolve=True)
                if isinstance(item, DictConfig)
                else dict(item)
                for item in (cfg.get("initial_candidates", None) or [])
            ),
            max_same_dimension_streak=max(
                1, int(cfg.get("max_same_dimension_streak", 2) or 2)
            ),
            deep_review_interval_cycles=int(deep_review.get("interval_cycles", 8) or 8),
            deep_review_results_rows=int(deep_review.get("results_rows", 30) or 30),
            creativity_interval_cycles=int(creativity.get("interval_cycles", 5) or 5),
            creativity_plateau_cycles=int(creativity.get("plateau_cycles", 4) or 4),
            search_base_selection=bool(base_selection.get("enabled", True)),
            search_base_max_options=max(
                0, int(base_selection.get("max_options", 12) or 0)
            ),
            motion_group_evidence=evidence.parse_motion_group_evidence(cfg),
            motion_group_seed_reduce=evidence.parse_motion_group_seed_reduce(cfg),
            auto_start_ollama=bool(cfg.get("auto_start_ollama", True)),
            host=str(cfg.get("host", "") or ""),
            context_length=(
                None
                if codex_config.get("model_context_window", None) in (None, "")
                else int(codex_config["model_context_window"])
            ),
        )

    def initial_candidate_for(self, cycle_index: int) -> Mapping[str, Any] | None:
        """The deterministic candidate for this cycle, if one is configured."""

        if cycle_index < 1 or cycle_index > len(self.initial_candidates):
            return None
        return self.initial_candidates[cycle_index - 1]


def resolve_planner_agent(task_cfg: DictConfig) -> AgentSpec:
    return resolve_agent_spec(
        task_cfg,
        "planner",
        default_model=None,
        default_reasoning_effort=DEFAULT_REASONING_EFFORT,
    )


def planner_label(task_cfg: DictConfig) -> str:
    return agent_label(resolve_planner_agent(task_cfg))


# --- managed Ollama ---------------------------------------------------------


def _is_local_host(host: str) -> bool:
    return (urlparse(host).hostname or "") in {"127.0.0.1", "localhost", "::1"}


def _server_is_reachable(host: str) -> bool:
    try:
        with urlopen(host, timeout=1.0):
            return True
    except OSError:
        return False


def start_managed_ollama(
    root: Path, config: PlannerConfig
) -> subprocess.Popen[bytes] | None:
    """Start a local Ollama server for this run, if the task asked for one."""

    host = config.host
    if not host or not config.auto_start_ollama or not _is_local_host(host):
        return None
    if _server_is_reachable(host):
        print(f"[hpo] reusing existing Ollama server at {host}")
        return None

    run_env = dict(os.environ)
    run_env["OLLAMA_HOST"] = host
    if config.context_length is not None:
        run_env["OLLAMA_CONTEXT_LENGTH"] = str(config.context_length)
    process = subprocess.Popen(
        ["ollama", "serve"],
        cwd=str(root),
        env=run_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + _OLLAMA_START_TIMEOUT_SECONDS
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Managed Ollama server exited early with code {process.returncode}."
            )
        if _server_is_reachable(host):
            print(f"[hpo] managed Ollama server is ready at {host}")
            return process
        time.sleep(0.5)
    stop_managed_ollama(process)
    raise RuntimeError(
        f"Timed out waiting for managed Ollama server at {host} to become reachable."
    )


def stop_managed_ollama(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


# --- agent turns ------------------------------------------------------------


def run_phase(
    *,
    root: Path,
    cycle_index: int,
    phase_name: str,
    prompt: str,
    message_path: Path,
    agent: AgentSpec,
    config: PlannerConfig,
    dry_run: bool,
) -> int:
    """Run one agent turn, retrying only a timeout."""

    print(f"[hpo] cycle={cycle_index} phase={phase_name} ({agent_label(agent)})")
    message_path.parent.mkdir(parents=True, exist_ok=True)
    returncode = 0
    for attempt in range(config.timeout_retries + 1):
        returncode = run_agent(
            root=root,
            prompt=prompt,
            message_path=message_path,
            agent=agent,
            timeout_seconds=config.timeout_seconds,
            dry_run=dry_run,
        )
        if returncode != _AGENT_TIMEOUT_RETURNCODE or attempt >= config.timeout_retries:
            break
        print(
            f"[hpo] cycle={cycle_index} planner timed out; retrying "
            f"{phase_name} ({attempt + 1}/{config.timeout_retries})",
            file=sys.stderr,
        )
    return returncode


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def parse_strategy(text: str) -> str:
    """Validate the planning turn's answer against the strategy headings."""

    strategy = _strip_fence(text)
    present = set(strategy.splitlines())
    missing = [heading for heading in STRATEGY_HEADINGS if heading not in present]
    if missing:
        raise ValueError(
            "planning response did not contain required strategy headings: "
            + ", ".join(missing)
        )
    return strategy.rstrip() + "\n"


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the one JSON object out of an agent response."""

    stripped = _strip_fence(text)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("agent response did not contain a JSON object")
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"agent response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("agent response JSON must be an object")
    return payload


@dataclass(frozen=True)
class ProposalResponse:
    """What the prepare turn answered, before it is resolved against a base."""

    base: Any = "root"
    axis: Axis | None = None
    experiments: tuple[Mapping[str, Any], ...] = ()
    rationale: str = ""


def parse_proposal(text: str) -> ProposalResponse:
    """Read the prepare turn's answer.

    Two shapes are accepted because they say different things: an ``axis`` is
    one dimension the controller enumerates, and ``experiments`` is an explicit
    candidate list. Exactly one must be present.
    """

    payload = extract_json_object(text)
    axis_payload = payload.get("axis", None)
    experiments = payload.get("experiments", None)
    if (axis_payload is None) == (experiments is None):
        raise ValueError(
            "prepare response must contain exactly one of `axis` or `experiments`"
        )

    axis: Axis | None = None
    if axis_payload is not None:
        if not isinstance(axis_payload, dict):
            raise ValueError("prepare response field `axis` must be an object")
        name = str(axis_payload.get("name", "")).strip()
        variants = axis_payload.get("variants", None)
        values = axis_payload.get("values", None)
        if variants is not None:
            axis = Axis(
                name=name,
                variants=tuple(
                    (str(variant["label"]), dict(variant["overrides"]))
                    for variant in variants
                ),
            )
        else:
            if not isinstance(values, list):
                raise ValueError("axis `values` must be a list")
            axis = Axis(name=name, values=tuple(values))

    resolved_experiments: tuple[Mapping[str, Any], ...] = ()
    if experiments is not None:
        if not isinstance(experiments, list) or not experiments:
            raise ValueError(
                "prepare response field `experiments` must be a non-empty list"
            )
        resolved_experiments = tuple(dict(item) for item in experiments)

    return ProposalResponse(
        base=payload.get("base", "root"),
        axis=axis,
        experiments=resolved_experiments,
        rationale=str(payload.get("rationale", "") or "").strip(),
    )


# --- the prompt envelope ----------------------------------------------------


@dataclass
class PromptContext:
    """Everything one prompt is built from, gathered once per cycle."""

    root: Path
    task_path: Path
    program_path: Path
    task_cfg: DictConfig
    root_spec: RootSpec
    state_dir: Path
    cycle_index: int
    comparator: ObjectiveComparator
    config: PlannerConfig
    status: Mapping[str, Any] = field(default_factory=dict)
    recent_results: Sequence[Mapping[str, Any]] = ()
    crash_text: str = ""
    validation_feedback: str = ""


def _relpath(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _excerpt(path: Path, max_chars: int = 1800) -> str:
    if not path.exists():
        return f"({path.name} missing)"
    text = path.read_text(encoding="utf-8").strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _recent_progress(state_dir: Path, cycles: int) -> str:
    path = state_dir / "progress.md"
    if not path.exists():
        return "(no progress recorded yet)"
    sections = path.read_text(encoding="utf-8").split("\n## Cycle ")
    tail = sections[-cycles:] if len(sections) > 1 else []
    return "\n## Cycle ".join(tail).strip() or "(no cycle recorded yet)"


def _outcome_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        for key in ("status", "decision"):
            value = str(row.get(key, "") or "")
            if value:
                counts[f"{key}={value}"] = counts.get(f"{key}={value}", 0) + 1
    if not counts:
        return "(no completed cycle yet)"
    return ", ".join(f"{key} x{count}" for key, count in sorted(counts.items()))


def _repeat_guard(rows: Sequence[Mapping[str, Any]], streak_limit: int) -> str:
    """Say when the search has spent too long on one dimension."""

    dimensions: list[str] = []
    for row in reversed(rows):
        overrides = str(row.get("effective_overrides_json", "") or "")
        if not overrides:
            break
        dimensions.append(overrides)
    streak = 0
    for item in dimensions:
        if dimensions and item == dimensions[0]:
            streak += 1
        else:
            break
    if streak >= streak_limit:
        return (
            f"The last {streak} cycles moved the same override set. Change "
            "dimension before proposing another variation of it."
        )
    return "(no repeated-dimension warning)"


def _schedule_note(context: PromptContext) -> str:
    notes: list[str] = []
    config = context.config
    cycle = context.cycle_index
    if config.deep_review_interval_cycles and (
        cycle % config.deep_review_interval_cycles == 0
    ):
        notes.append(
            f"Deep review cycle: read the last {config.deep_review_results_rows} "
            "result rows before proposing, not only the recent window."
        )
    if config.creativity_interval_cycles and (
        cycle % config.creativity_interval_cycles == 0
    ):
        notes.append(
            "Creativity cycle: prefer a materially different mechanism over "
            "another local refinement."
        )
    plateau = int(context.status.get("cycles_since_improvement", 0) or 0)
    if plateau >= config.creativity_plateau_cycles:
        notes.append(
            f"No improvement for {plateau} cycles. Local refinement is not "
            "working; change the dimension."
        )
    return "\n".join(f"- {note}" for note in notes) or "- (no scheduled emphasis)"


def _objective_block(task_cfg: DictConfig) -> str:
    objective = task_cfg.get("objective", None)
    if objective is None:
        return "(no objective configured)"
    return OmegaConf.to_yaml(objective).rstrip()


def _motion_block(root_spec: RootSpec) -> str:
    lines = [
        f"- `{motion}`: `{spec.config_path}` / `{spec.config_name}`"
        for motion, spec in root_spec.motions.items()
    ]
    lines.append(f"- seeds: {list(root_spec.seeds)}")
    lines.append("- fixed overrides:")
    lines.extend(f"  - `{item}`" for item in root_spec.fixed_overrides.items)
    return "\n".join(lines)


def _search_space_block(root_spec: RootSpec) -> str:
    allowed = root_spec.parameter_space
    lines = ["Searchable paths (copy one verbatim):"]
    lines.extend(f"- `{path}`" for path in (allowed or ()))
    if not allowed:
        lines.append("- (unrestricted)")
    lines.append("")
    lines.append("Forbidden prefixes:")
    lines.extend(f"- `{prefix}`" for prefix in root_spec.forbidden_prefixes)
    return "\n".join(lines)


def _bases_block(context: PromptContext) -> str:
    if not context.config.search_base_selection:
        return "- (base selection disabled; branch from the current incumbent)"
    bases = list(context.comparator.recommend_bases())
    if not bases:
        return "- root (nothing has been kept yet)"
    limit = context.config.search_base_max_options
    if limit > 0:
        bases = bases[:limit]
    return "\n".join(f"- `[{cycle}, {name}]`" for cycle, name in bases)


def _motion_group_lines(context: PromptContext, row: Mapping[str, Any]) -> list[str]:
    """Per-motion evidence for one result row, when the task asked for it.

    A multi-motion task can improve its aggregate by sacrificing one motion.
    The comparator's one-line rendering cannot show that, so the requested
    per-group statistics are read back from the cycle's own evidence.
    """

    statistics = context.config.motion_group_evidence
    if not statistics:
        return []
    raw_path = str(row.get("evidence_path", "") or "").strip()
    if not raw_path:
        return []
    summary_path = Path(raw_path)
    if summary_path.is_dir():
        summary_path = summary_path / "summary.json"
    if summary_path.name != "summary.json":
        summary_path = summary_path.with_name("summary.json")
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Evidence for an old cycle can be pruned; the row still renders.
        return []
    if not isinstance(payload, dict):
        return []
    candidate = str(row.get("candidate", "") or "")
    lines: list[str] = []
    for experiment in payload.get("experiments", []) or ():
        if not isinstance(experiment, dict):
            continue
        if candidate and str(experiment.get("name", "")) != candidate:
            continue
        lines.extend(
            f"  {line}"
            for line in evidence.render_motion_group_lines(
                experiment,
                statistics,
                seed_reduce=context.config.motion_group_seed_reduce,
            )
        )
    return lines


def _recent_results_block(context: PromptContext) -> str:
    rows = list(context.recent_results)[-context.config.recent_cycles :]
    if not rows:
        return "(no results recorded yet)"
    lines: list[str] = []
    for row in rows:
        lines.append(context.comparator.render_recent_result_line(row))
        lines.extend(_motion_group_lines(context, row))
    return "\n".join(lines)


def _common_envelope(context: PromptContext) -> list[str]:
    """The sections every planner prompt carries, in a fixed order."""

    task_cfg = context.task_cfg
    return [
        "## Program",
        "",
        f"Read `{_relpath(context.root, context.program_path)}` first. It is the "
        "contract for this protocol.",
        "",
        "## Task",
        "",
        f"- Task file: `{_relpath(context.root, context.task_path)}`",
        f"- Description: {str(task_cfg.get('description', '') or '(none)')}",
        "",
        "### Instructions",
        "",
        context.config.instructions or "(none)",
        "",
        "## Objective",
        "",
        "```yaml",
        _objective_block(task_cfg),
        "```",
        "",
        "## What trains",
        "",
        _motion_block(context.root_spec),
        "",
        "## Search boundary",
        "",
        _search_space_block(context.root_spec),
        "",
        "## Where the run stands",
        "",
        f"- Cycle: {context.cycle_index}",
        f"- Last cycle: {context.status.get('last_status', '(none)') or '(none)'}"
        f" / {context.status.get('last_decision', '(none)') or '(none)'}",
        f"- Cycles since improvement: "
        f"{context.status.get('cycles_since_improvement', 0)}",
        f"- Outcomes so far: {_outcome_digest(context.recent_results)}",
        "",
        "### Best result",
        "",
        context.comparator.render_best(),
        "",
        "### Bases worth branching from",
        "",
        _bases_block(context),
        "",
        "### Recent results",
        "",
        _recent_results_block(context),
        "",
        "### Recent progress",
        "",
        _recent_progress(context.state_dir, context.config.recent_cycles),
        "",
        "### Current strategy",
        "",
        _excerpt(context.state_dir / "strategy.md"),
        "",
        "## Emphasis this cycle",
        "",
        _schedule_note(context),
        "",
        _repeat_guard(context.recent_results, context.config.max_same_dimension_streak),
        "",
        "## Previous failures",
        "",
        context.crash_text or "(no previous crash)",
        "",
        context.validation_feedback or "(no previous validation feedback)",
    ]


def build_planning_prompt(context: PromptContext) -> str:
    """The first turn: update what the search believes."""

    sections = [
        "# HPO planning phase",
        "",
        "Answer with the updated strategy document and nothing else. The "
        "controller writes `strategy.md` from your answer; do not edit any file.",
        "",
        *_common_envelope(context),
        "",
        "## Answer format",
        "",
        "Markdown with exactly these headings, in this order:",
        "",
        *(f"- `{heading}`" for heading in STRATEGY_HEADINGS),
    ]
    return "\n".join(sections) + "\n"


def build_prepare_prompt(context: PromptContext) -> str:
    """The second turn: name the experiment to run."""

    batch_cfg = context.task_cfg.get("batch", None) or {}
    minimum = int(batch_cfg.get("min_experiments_per_cycle", 1) or 1)
    maximum = int(batch_cfg.get("max_experiments_per_cycle", 1) or 1)
    motions = ", ".join(f"`{motion}`" for motion in context.root_spec.motion_names())
    sections = [
        "# HPO prepare phase",
        "",
        "Answer with one JSON object and nothing else. The controller resolves "
        "it, validates it, and runs it; do not edit any file and do not run the "
        "evaluator.",
        "",
        *_common_envelope(context),
        "",
        "## Answer format",
        "",
        "```json",
        "{",
        '  "base": "root" | "recommended" | [cycle, "candidate"],',
        '  "rationale": "one sentence on what this cycle tests",',
        '  "experiments": [',
        '    {"name": "...", "rationale": "...",',
        '     "motions": {"<motion>": {"overrides": ["path=value", ...]}}}',
        "  ]",
        "}",
        "```",
        "",
        "Or, to sweep one dimension and let the controller enumerate it:",
        "",
        "```json",
        '{"base": "recommended", "rationale": "...",',
        ' "axis": {"name": "agent.ratio_clip", "values": [0.1, 0.2, 0.3]}}',
        "```",
        "",
        "Rules:",
        "",
        f"- Every experiment must cover exactly these motions: {motions}.",
        f"- Propose between {minimum} and {maximum} experiments per cycle.",
        "- An `axis` with N values expands to N experiments; stay within that "
        "same bound.",
        "- Every override path must come verbatim from the searchable list "
        "above, and none may start with a forbidden prefix.",
        "- Do not repeat an override set that a previous cycle already ran.",
    ]
    return "\n".join(sections) + "\n"
