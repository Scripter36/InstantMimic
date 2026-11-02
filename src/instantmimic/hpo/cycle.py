"""One cycle, one code path, both objective modes.

Five phases, in order, every cycle, in either mode:

1. **planning** -- the agent updates ``strategy.md``. Skipped when
   ``planner.initial_candidates`` supplies this cycle deterministically.
2. **prepare** -- the agent answers with a proposal; the controller resolves it
   against a base and writes ``cycles/NNN/proposal.yaml``.
3. **validation** -- the proposal is checked in this process against the task's
   search boundary. A failure is re-fed to the next cycle's prompt verbatim.
4. **evaluate** -- one ``evaluate.py`` invocation runs every candidate.
5. **finalize** -- the comparator reads the evidence and returns a verdict; the
   state layer writes everything.

The comparator is the only thing that differs between modes, and it is built
once here from ``objective.mode``.

A failed cycle still consumes a cycle index: the failure is evidence about the
search, and hiding it would make ``runtime.cycles`` mean different things
depending on how well the planner behaved. Ctrl-C leaves ``active_cycle`` set,
so a resume restarts that cycle from the beginning rather than silently
skipping it.
"""

from __future__ import annotations

import json
import shlex
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from instantmimic.hpo import planner as planner_module
from instantmimic.hpo.controller_state import (
    LIFECYCLE_COMPLETED,
    LIFECYCLE_DRY_RUN,
    append_controller_result,
    append_progress_entry,
    cycle_artifact_dir,
    ensure_controller_state,
    failed,
    load_controller_status,
    objective_artifact_dir,
    prepare_cycle_transaction,
    read_controller_results,
    write_best_result,
    write_controller_status,
    write_cycle_final_report,
)
from instantmimic.hpo.evaluator_launch import (
    evaluator_job_name,
    run_launched_evaluator,
)
from instantmimic.hpo.experiment import (
    Candidate,
    OverrideSet,
    Proposal,
    RootSpec,
    expand,
    load_root,
    validate_candidate_configs,
    validate_overrides,
)
from instantmimic.hpo.objective import Evidence, build_comparator
from instantmimic.hpo.objective.base import (
    IncompatibleObjectiveError,
    ObjectiveComparator,
    Verdict,
)
from instantmimic.hpo.run_context import (
    HPORunContext,
    default_hpo_run_root,
    experiment_directory_for_run,
)
from instantmimic.hpo.schema import RuntimeOptions
from instantmimic.hpo.tracking import HPOTracker

EVALUATE_SCRIPT = "src/instantmimic/hpo/evaluate.py"


@dataclass
class CycleFailure(Exception):
    """A phase could not produce what the next phase needs."""

    phase: str
    message: str

    def __str__(self) -> str:
        return self.message


def _relpath(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _objective_mode(task_cfg: DictConfig) -> str:
    objective_cfg = task_cfg.get("objective", None) or {}
    return str(objective_cfg.get("mode", "") or "")


def _program_path(root: Path, task_cfg: DictConfig) -> Path:
    del task_cfg
    return root / "docs" / "reference" / "hpo-task-schema.md"


def _resolve_output_root(root: Path, task_cfg: DictConfig, task_path: Path) -> Path:
    raw = task_cfg.get("output_root", None)
    if raw in (None, ""):
        return default_hpo_run_root(root, task_cfg, task_path) / "evaluator"
    candidate = Path(str(raw)).expanduser()
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


# --- proposal resolution ----------------------------------------------------


def _base_overrides(
    base: Any,
    root_spec: RootSpec,
    results: Sequence[Mapping[str, Any]],
    comparator: ObjectiveComparator,
) -> dict[str, OverrideSet]:
    """The override set a candidate's delta is applied on top of.

    ``recommended`` asks the comparator which node is worth branching from; a
    node id names one explicitly. Both resolve to that node's recorded
    effective overrides, so a base is always something that actually ran.
    """

    empty = {motion: OverrideSet() for motion in root_spec.motions}
    if base in (None, "", "root"):
        return empty
    node = base
    if base == "recommended":
        recommended = list(comparator.recommend_bases())
        if not recommended:
            return empty
        node = recommended[0]
    if not isinstance(node, (list, tuple)) or len(list(node)) != 2:
        raise CycleFailure("prepare", f"unsupported proposal base: {base!r}")
    cycle, candidate = list(node)
    for row in reversed(list(results)):
        if str(row.get("cycle", "")) == str(cycle) and row.get("candidate") == str(
            candidate
        ):
            stored = json.loads(row.get("effective_overrides_json") or "{}")
            return {
                motion: OverrideSet.of(stored.get(motion, []))
                for motion in root_spec.motions
            }
    raise CycleFailure(
        "prepare", f"proposal base [{cycle}, {candidate}] is not a recorded node"
    )


def _previous_validation_feedback(state_dir: Path, cycle_index: int) -> str:
    """The rejection the previous cycle recorded, verbatim.

    A validation failure consumes its cycle, so the log the planner needs to
    read is the one cycle ``N-1`` wrote -- every error it listed, not just the
    first one that reached `status.last_error`.
    """

    if cycle_index <= 1:
        return ""
    path = state_dir / "cycles" / f"{int(cycle_index) - 1:03d}" / "validation.log"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _candidates_from_response(
    response: planner_module.ProposalResponse,
    root_spec: RootSpec,
    base: dict[str, OverrideSet],
) -> tuple[Candidate, ...]:
    if response.axis is not None:
        return expand(response.axis, base, parent=response.base)
    candidates: list[Candidate] = []
    for item in response.experiments:
        name = str(item.get("name", "")).strip()
        if not name:
            raise CycleFailure("prepare", "each experiment requires a non-empty name")
        motions = item.get("motions", None)
        if not isinstance(motions, Mapping):
            raise CycleFailure(
                "prepare", f"experiment '{name}' must contain a motions mapping"
            )
        unknown = sorted(set(motions) - set(root_spec.motions))
        if unknown:
            raise CycleFailure(
                "prepare", f"experiment '{name}' has unknown motions: {unknown}"
            )
        delta = {
            motion: OverrideSet.of((motions.get(motion) or {}).get("overrides", []))
            for motion in root_spec.motions
        }
        candidates.append(
            Candidate(
                name=name,
                rationale=str(item.get("rationale", "") or ""),
                parent=item.get("parent", response.base),
                delta=delta,
                effective={
                    motion: base[motion].merge(delta[motion])
                    for motion in root_spec.motions
                },
            )
        )
    return tuple(candidates)


def _validate_candidates(
    candidates: Sequence[Candidate],
    root_spec: RootSpec,
    task_cfg: DictConfig,
    previous_effective: Sequence[str],
) -> list[str]:
    """Check the proposal against the task's boundary and its own history."""

    batch_cfg = task_cfg.get("batch", None) or {}
    minimum = int(batch_cfg.get("min_experiments_per_cycle", 1) or 1)
    maximum = int(batch_cfg.get("max_experiments_per_cycle", 1) or 1)
    errors: list[str] = []
    if not minimum <= len(candidates) <= maximum:
        errors.append(
            f"proposal has {len(candidates)} experiments; this task allows "
            f"{minimum}..{maximum}"
        )
    seen: set[str] = set()
    seen_effective = set(_normalized(previous_effective))
    for candidate in candidates:
        if candidate.name in seen:
            errors.append(f"duplicate experiment name '{candidate.name}'")
        seen.add(candidate.name)
        for motion, overrides in candidate.delta.items():
            errors.extend(
                validate_overrides(
                    root_spec,
                    candidate_name=candidate.name,
                    motion_name=motion,
                    overrides=overrides,
                )
            )
        signature = _override_signature(
            {
                motion: overrides.items
                for motion, overrides in candidate.effective.items()
            }
        )
        if signature in seen_effective:
            errors.append(
                f"experiment '{candidate.name}' repeats an override set already proposed or run"
            )
        seen_effective.add(signature)
    return errors


def _initial_candidate_proposal(
    payload: Mapping[str, Any], root_spec: RootSpec
) -> Proposal:
    """Turn one ``planner.initial_candidates`` entry into a proposal."""

    response = planner_module.ProposalResponse(
        base="root",
        experiments=(dict(payload),),
        rationale=str(payload.get("rationale", "") or ""),
    )
    base = {motion: OverrideSet() for motion in root_spec.motions}
    return Proposal(
        base="root", candidates=_candidates_from_response(response, root_spec, base)
    )


# --- evaluation -------------------------------------------------------------


def _write_evaluator_task(
    task_cfg: DictConfig,
    *,
    base: dict[str, OverrideSet],
    experiment_directory: str,
    cycle_root: Path,
) -> Path:
    """Materialize common fixed settings; proposal carries per-motion inheritance."""
    del base
    cfg = OmegaConf.create(OmegaConf.to_container(task_cfg, resolve=True))
    fixed = OverrideSet.of(cfg.batch.get("fixed_train_overrides") or [])
    cfg.batch.fixed_train_overrides = list(
        fixed.merge(
            OverrideSet.of([f"experiment.directory={experiment_directory}"])
        ).items
    )
    path = cycle_root / "evaluator_task.yaml"
    OmegaConf.save(cfg, path)
    return path.resolve()


def _run_evaluator(
    *,
    root: Path,
    evaluator_task_path: Path,
    proposal_path: Path,
    output_root: Path,
    cycle_root: Path,
    state_dir: Path,
    cycle_index: int,
    runtime_options: RuntimeOptions,
) -> tuple[int, Path, Path]:
    """Run every candidate once, through the one evaluator."""

    evaluator_log_path = cycle_root / "evaluator.log"
    run_dir = (output_root / f"cycle_{cycle_index:03d}").resolve()
    command = [
        "uv",
        "run",
        "python",
        EVALUATE_SCRIPT,
        "--task",
        str(evaluator_task_path),
        "--proposal",
        str(proposal_path),
        "--output-root",
        str(run_dir),
    ]
    if runtime_options.dry_run:
        command.append("--dry-run")
    print(
        f"[hpo] evaluator_launcher={runtime_options.evaluator.launcher} "
        f"max_parallel={runtime_options.evaluator.max_parallel}"
    )
    print(f"[hpo] evaluator={shlex.join(command)}")
    if runtime_options.dry_run:
        evaluator_log_path.write_text(
            f"dry-run evaluator command: {shlex.join(command)}\n", encoding="utf-8"
        )
        return 0, evaluator_log_path, run_dir

    returncode, _script_path = run_launched_evaluator(
        command=command,
        launch_config=runtime_options.evaluator,
        root=root,
        log_path=evaluator_log_path,
        job_dir=state_dir / "evaluator_jobs",
        job_name=evaluator_job_name(evaluator_task_path, cycle_index),
        log_prefix="hpo",
        task_path=evaluator_task_path,
        output_dir=run_dir,
    )
    return returncode, evaluator_log_path, run_dir


def _load_evidence(
    run_dir: Path, cycle_index: int, candidates: Sequence[Candidate]
) -> Evidence:
    evidence_path = run_dir / "evidence.json"
    trials_path = run_dir / "trials.jsonl"
    summary_path = run_dir / "summary.json"
    payload = (
        json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence_path.exists()
        else None
    )
    trials = [
        json.loads(line)
        for line in (
            trials_path.read_text(encoding="utf-8").splitlines()
            if trials_path.exists()
            else []
        )
        if line.strip()
    ]
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else None
    )
    return Evidence(
        cycle=cycle_index,
        evidence_path=evidence_path if evidence_path.exists() else None,
        evidence=payload,
        trials=trials,
        candidates=tuple(candidates),
        summary=summary,
    )


# --- the cycle --------------------------------------------------------------


def _mark_phase(
    status_path: Path,
    status: dict[str, Any],
    *,
    mode: str,
    cycle_index: int,
    phase: str,
    message_path: Path | None = None,
) -> None:
    status["active_cycle"] = cycle_index
    status["active_phase"] = phase
    if message_path is not None:
        status["last_message_path"] = str(message_path)
    write_controller_status(status_path, status, mode=mode)


def _record_cycle(
    *,
    state_dir: Path,
    status: dict[str, Any],
    tracker: HPOTracker,
    mode: str,
    cycle_index: int,
    lifecycle: str,
    verdict: Verdict | None,
    comparator: ObjectiveComparator | None,
    candidates: Sequence[Candidate],
    proposal_path: Path | None,
    evidence_path: Path | None,
    error: str,
) -> None:
    """Write every artifact one cycle produces, and log it. The only place.

    Tracking used to be called on the success path only, so a failed cycle left
    a gap in the metric series and `hpo.status_failed` was never 1.0
    for a real failure. Logging from here means a cycle cannot be recorded to
    disk without also being recorded to the tracker.
    """

    decision = verdict.decision if verdict is not None else ""
    results_path = state_dir / "results.tsv"
    rows = candidates or (Candidate(name=""),)
    for candidate in rows:
        metrics = (
            verdict.per_candidate.get(candidate.name, {}) if verdict is not None else {}
        )
        append_controller_result(
            results_path,
            {
                "cycle": cycle_index,
                "candidate": candidate.name,
                # A node id is JSON so it round-trips; "root" and
                # "recommended" stay plain text, because a human reads this file.
                "parent": (
                    json.dumps(list(candidate.parent))
                    if isinstance(candidate.parent, (tuple, list))
                    else str(candidate.parent)
                ),
                "mode": mode,
                "status": lifecycle,
                "decision": decision,
                "objective_value": (
                    ""
                    if metrics.get("objective_value") is None
                    else f"{float(metrics['objective_value']):.6f}"
                ),
                "evidence_path": "" if evidence_path is None else str(evidence_path),
                "proposal_path": "" if proposal_path is None else str(proposal_path),
                "error": error,
                "effective_overrides_json": {
                    motion: list(overrides.items)
                    for motion, overrides in candidate.effective.items()
                },
                "metrics_json": metrics,
            },
        )

    comparator_lines = (
        comparator.render_progress_lines(verdict)
        if comparator is not None and verdict is not None
        else []
    )
    append_progress_entry(
        state_dir / "progress.md",
        cycle_index=cycle_index,
        lifecycle=lifecycle,
        decision=decision,
        proposal_path="" if proposal_path is None else str(proposal_path),
        evidence_path="" if evidence_path is None else str(evidence_path),
        comparator_lines=comparator_lines,
    )
    write_cycle_final_report(
        cycle_artifact_dir(state_dir, cycle_index) / "final.md",
        cycle_index=cycle_index,
        lifecycle=lifecycle,
        decision=decision,
        error=error,
        comparator_lines=(
            comparator.render_final_lines(verdict)
            if comparator is not None and verdict is not None
            else []
        ),
    )
    if comparator is not None and verdict is not None:
        write_best_result(state_dir / "best_result.md", comparator.render_best())

    status["cycle"] = cycle_index
    status["active_cycle"] = None
    status["active_phase"] = ""
    status["last_status"] = lifecycle
    status["last_decision"] = decision
    status["last_error"] = error
    status["last_result"] = verdict.last_result_text if verdict is not None else error
    status["last_proposal_path"] = "" if proposal_path is None else str(proposal_path)
    status["last_summary_path"] = "" if evidence_path is None else str(evidence_path)
    if verdict is not None and verdict.improved:
        status["cycles_since_improvement"] = 0
        status["best"] = {
            "cycle": cycle_index,
            "candidate": str(verdict.representative.get("candidate", "")),
            "value": verdict.objective_value,
            "evidence_path": "" if evidence_path is None else str(evidence_path),
        }
    else:
        status["cycles_since_improvement"] = (
            int(status.get("cycles_since_improvement", 0) or 0) + 1
        )
    if comparator is not None:
        status["objective_state"] = dict(
            comparator.dump_state(objective_artifact_dir(state_dir))
        )
    write_controller_status(state_dir / "status.yaml", status, mode=mode)

    cycle_root = cycle_artifact_dir(state_dir, cycle_index)
    evidence_dir = None if evidence_path is None else evidence_path.parent
    tracker.log_cycle_result(
        cycle_index=cycle_index,
        metric_objective=("" if comparator is None else str(comparator.objective_id)),
        outcome={
            "status": lifecycle,
            "decision": decision,
            **(
                comparator.tracking_metrics(verdict)
                if comparator is not None and verdict is not None
                else {}
            ),
        },
        summary_path=evidence_path,
        # These were all passed as None, so no cycle artifact was ever
        # uploaded even with `log_artifacts: true`.
        trials_path=_existing(
            None if evidence_dir is None else evidence_dir / "trials.jsonl"
        ),
        evaluator_log_path=_existing(cycle_root / "evaluator.log"),
        planning_message_path=_existing(cycle_root / "planning.md"),
        prepare_message_path=_existing(cycle_root / "proposal.md"),
        final_message_path=_existing(cycle_root / "final.md"),
        validation_log_path=_existing(cycle_root / "validation.log"),
        cycles_since_improvement=int(status.get("cycles_since_improvement", 0) or 0),
    )


def _existing(path: Path | None) -> Path | None:
    """The path when there is a file there, else nothing to upload."""

    return path if path is not None and path.is_file() else None


def run_one_cycle(
    *,
    root: Path,
    run_context: HPORunContext,
    runtime_options: RuntimeOptions,
    tracker: HPOTracker,
) -> int:
    """Run exactly one cycle. ``run_hpo`` owns the loop."""

    if runtime_options.cycles != 1:
        raise ValueError(
            "The cycle runtime runs exactly one cycle; run_hpo owns cycle iteration."
        )
    task_path = run_context.task_path
    task_cfg = run_context.task_cfg
    state_dir = run_context.state_dir
    mode = _objective_mode(task_cfg)
    task_name = str(task_cfg.get("name", "") or task_path.stem)

    status = ensure_controller_state(
        state_dir=state_dir, task_path=task_path, task_name=task_name, mode=mode
    )
    status_path = state_dir / "status.yaml"
    root_spec = load_root(task_cfg, repo_root=root)
    config = planner_module.PlannerConfig.from_task(task_cfg)
    comparator = build_comparator(
        task_cfg.get("objective", None), state_dir=state_dir, task_cfg=task_cfg
    )
    cycle_index = (
        int(status.get("active_cycle") or 0) or int(status.get("cycle", 0) or 0) + 1
    )
    if not runtime_options.dry_run:
        prepare_cycle_transaction(state_dir, status, cycle_index)
    comparator.load_state(
        objective_artifact_dir(state_dir), status.get("objective_state", {})
    )
    objective_config = OmegaConf.to_container(task_cfg.objective, resolve=True)
    stored_objective = status.get("objective_config")
    if stored_objective is not None and stored_objective != objective_config:
        raise IncompatibleObjectiveError(
            "This run's objective changed; use a fresh hpo state."
        )
    status["objective_config"] = objective_config
    comparator.check_compatible()
    if not runtime_options.dry_run:
        write_best_result(state_dir / "best_result.md", comparator.render_best())

    cycle_root = cycle_artifact_dir(state_dir, cycle_index)
    proposal_path = cycle_root / "proposal.yaml"
    validation_log_path = cycle_root / "validation.log"
    results = read_controller_results(state_dir / "results.tsv")

    ollama = planner_module.start_managed_ollama(root, config)
    try:
        return _run_phases(
            root=root,
            task_path=task_path,
            task_cfg=task_cfg,
            state_dir=state_dir,
            status=status,
            status_path=status_path,
            mode=mode,
            cycle_index=cycle_index,
            cycle_root=cycle_root,
            proposal_path=proposal_path,
            validation_log_path=validation_log_path,
            results=results,
            root_spec=root_spec,
            config=config,
            comparator=comparator,
            runtime_options=runtime_options,
            tracker=tracker,
            run_context=run_context,
        )
    except KeyboardInterrupt:
        # Leave `active_cycle` set so the next run restarts this cycle -- but
        # only while there is still a cycle to restart. Ctrl-C can land after
        # `_record_cycle` has written the row and advanced the index, during
        # tracker upload; re-marking the cycle active there made the next run
        # redo a finished cycle, append a duplicate row, and then stop with
        # "a cycle must advance the cycle index by exactly one".
        persisted = load_controller_status(status_path, mode=mode)
        if int(persisted.get("cycle", 0) or 0) < cycle_index:
            status = persisted
            _mark_phase(
                status_path,
                status,
                mode=mode,
                cycle_index=cycle_index,
                phase=str(status.get("active_phase", "") or "planning"),
            )
        raise
    finally:
        planner_module.stop_managed_ollama(ollama)


def _run_phases(
    *,
    root: Path,
    task_path: Path,
    task_cfg: DictConfig,
    state_dir: Path,
    status: dict[str, Any],
    status_path: Path,
    mode: str,
    cycle_index: int,
    cycle_root: Path,
    proposal_path: Path,
    validation_log_path: Path,
    results: Sequence[Mapping[str, Any]],
    root_spec: RootSpec,
    config: planner_module.PlannerConfig,
    comparator: ObjectiveComparator,
    runtime_options: RuntimeOptions,
    tracker: HPOTracker,
    run_context: HPORunContext,
) -> int:
    agent = planner_module.resolve_planner_agent(task_cfg)
    prompt_context = planner_module.PromptContext(
        root=root,
        task_path=task_path,
        program_path=_program_path(root, task_cfg),
        task_cfg=task_cfg,
        root_spec=root_spec,
        state_dir=state_dir,
        cycle_index=cycle_index,
        comparator=comparator,
        config=config,
        status=status,
        recent_results=results,
        crash_text=str(status.get("last_error", "") or ""),
        # The previous cycle's log, not this one's -- this cycle has not run
        # yet. Reading `cycles/<this>/validation.log` meant the rejection was
        # never re-fed, and only `errors[0]` survived through `last_error`.
        validation_feedback=_previous_validation_feedback(state_dir, cycle_index),
    )
    candidates: tuple[Candidate, ...] = ()
    initial = config.initial_candidate_for(cycle_index)

    try:
        if initial is None:
            _mark_phase(
                status_path,
                status,
                mode=mode,
                cycle_index=cycle_index,
                phase="planning",
            )
            planning_path = cycle_root / "planning.md"
            with tracker.phase(cycle_index=cycle_index, phase_name="planning"):
                prompt = planner_module.build_planning_prompt(prompt_context)
                (cycle_root / "planning.prompt.md").write_text(prompt, encoding="utf-8")
                returncode = planner_module.run_phase(
                    root=root,
                    cycle_index=cycle_index,
                    phase_name="planning",
                    prompt=prompt,
                    message_path=planning_path,
                    agent=agent,
                    config=config,
                    dry_run=runtime_options.dry_run,
                )
            if returncode != 0:
                raise CycleFailure(
                    "planning", f"planning phase exited with code {returncode}"
                )
            if not runtime_options.dry_run:
                (state_dir / "strategy.md").write_text(
                    planner_module.parse_strategy(
                        planning_path.read_text(encoding="utf-8")
                    ),
                    encoding="utf-8",
                )

            _mark_phase(
                status_path,
                status,
                mode=mode,
                cycle_index=cycle_index,
                phase="prepare",
            )
            proposal_message_path = cycle_root / "proposal.md"
            with tracker.phase(cycle_index=cycle_index, phase_name="prepare"):
                prompt = planner_module.build_prepare_prompt(prompt_context)
                (cycle_root / "proposal.prompt.md").write_text(prompt, encoding="utf-8")
                returncode = planner_module.run_phase(
                    root=root,
                    cycle_index=cycle_index,
                    phase_name="prepare",
                    prompt=prompt,
                    message_path=proposal_message_path,
                    agent=agent,
                    config=config,
                    dry_run=runtime_options.dry_run,
                )
            if returncode != 0:
                raise CycleFailure(
                    "prepare", f"prepare phase exited with code {returncode}"
                )
            if runtime_options.dry_run:
                proposal = Proposal(base="root", candidates=())
            else:
                response = planner_module.parse_proposal(
                    proposal_message_path.read_text(encoding="utf-8")
                )
                if response.base == "recommended":
                    recommended = list(comparator.recommend_bases())
                    response = replace(
                        response, base=recommended[0] if recommended else "root"
                    )
                base = _base_overrides(response.base, root_spec, results, comparator)
                proposal = Proposal(
                    base=response.base,
                    axis=response.axis,
                    candidates=_candidates_from_response(response, root_spec, base),
                )
        else:
            print(
                f"[hpo] cycle={cycle_index} deterministic candidate "
                f"'{initial.get('name', '')}' from planner.initial_candidates"
            )
            proposal = _initial_candidate_proposal(initial, root_spec)

        candidates = proposal.candidates
        _mark_phase(
            status_path,
            status,
            mode=mode,
            cycle_index=cycle_index,
            phase="validation",
        )
        if candidates:
            previous = [row.get("effective_overrides_json", "") for row in results]
            errors = _validate_candidates(
                candidates, root_spec, task_cfg, _normalized(previous)
            )
            if not errors:
                errors.extend(validate_candidate_configs(root_spec, tuple(candidates)))
            if errors:
                validation_log_path.write_text(
                    "Proposal validation failed for cycle "
                    f"{cycle_index}.\n\n" + "\n".join(f"- {e}" for e in errors) + "\n",
                    encoding="utf-8",
                )
                raise CycleFailure("validation", errors[0])
            validation_log_path.unlink(missing_ok=True)
            proposal_path.write_text(
                yaml.safe_dump(proposal.as_dict(), sort_keys=False), encoding="utf-8"
            )

        if runtime_options.dry_run:
            # The entrypoint announces the dry run when it stops the loop; a
            # second copy here just printed the same line twice.
            _mark_phase(
                status_path,
                status,
                mode=mode,
                cycle_index=cycle_index,
                phase="evaluate",
            )
            status["active_cycle"] = None
            status["active_phase"] = ""
            status["last_status"] = LIFECYCLE_DRY_RUN
            write_controller_status(status_path, status, mode=mode)
            return 0

        _mark_phase(
            status_path, status, mode=mode, cycle_index=cycle_index, phase="evaluate"
        )
        base = _base_overrides(proposal.base, root_spec, results, comparator)
        evaluator_task_path = _write_evaluator_task(
            task_cfg,
            base=base,
            experiment_directory=experiment_directory_for_run(
                task_cfg, task_path, run_context.run_dir or state_dir.parent
            ),
            cycle_root=cycle_root,
        )
        with tracker.phase(cycle_index=cycle_index, phase_name="evaluate"):
            returncode, evaluator_log_path, run_dir = _run_evaluator(
                root=root,
                evaluator_task_path=evaluator_task_path,
                proposal_path=proposal_path,
                output_root=_resolve_output_root(root, task_cfg, task_path),
                cycle_root=cycle_root,
                state_dir=state_dir,
                cycle_index=cycle_index,
                runtime_options=runtime_options,
            )
        evidence = _load_evidence(run_dir, cycle_index, candidates)
        if returncode != 0:
            evidence = Evidence(
                cycle=cycle_index,
                evidence_path=evidence.evidence_path or evaluator_log_path,
                evidence=evidence.evidence,
                trials=evidence.trials,
                candidates=tuple(candidates),
                summary=evidence.summary,
                lifecycle="failed",
                error=f"evaluator exited with code {returncode}",
            )

    except CycleFailure as failure:
        print(
            f"[hpo] cycle={cycle_index} {failure.phase} failed: {failure}",
            file=sys.stderr,
        )
        _record_cycle(
            state_dir=state_dir,
            status=status,
            tracker=tracker,
            mode=mode,
            cycle_index=cycle_index,
            lifecycle=failed(failure.phase),
            verdict=None,
            comparator=comparator,
            candidates=candidates,
            proposal_path=proposal_path if proposal_path.exists() else None,
            evidence_path=None,
            error=str(failure),
        )
        _sleep_after_failure(config)
        return 0
    except ValueError as exc:
        _record_cycle(
            state_dir=state_dir,
            status=status,
            tracker=tracker,
            mode=mode,
            cycle_index=cycle_index,
            lifecycle=failed("prepare"),
            verdict=None,
            comparator=comparator,
            candidates=candidates,
            proposal_path=None,
            evidence_path=None,
            error=str(exc),
        )
        _sleep_after_failure(config)
        return 0

    _mark_phase(
        status_path, status, mode=mode, cycle_index=cycle_index, phase="finalize"
    )
    print(f"[hpo] cycle={cycle_index} phase=finalize (deterministic)")
    try:
        with tracker.phase(cycle_index=cycle_index, phase_name="finalize"):
            verdict = comparator.compare(evidence)
    except IncompatibleObjectiveError as exc:
        # Fatal to the run, not to one cycle: every later cycle would train
        # first and then reach the same conclusion, draining the budget into an
        # answer that was already known.
        print(
            f"[hpo] cycle={cycle_index} objective mismatch: {exc}",
            file=sys.stderr,
        )
        _record_cycle(
            state_dir=state_dir,
            status=status,
            tracker=tracker,
            mode=mode,
            cycle_index=cycle_index,
            lifecycle=failed("finalize"),
            verdict=None,
            comparator=None,
            candidates=candidates,
            proposal_path=proposal_path,
            evidence_path=evidence.evidence_path,
            error=str(exc),
        )
        return 2
    except ValueError as exc:
        print(
            f"[hpo] cycle={cycle_index} finalize failed: {exc}",
            file=sys.stderr,
        )
        _record_cycle(
            state_dir=state_dir,
            status=status,
            tracker=tracker,
            mode=mode,
            cycle_index=cycle_index,
            lifecycle=failed("finalize"),
            verdict=None,
            comparator=comparator,
            candidates=candidates,
            proposal_path=proposal_path,
            evidence_path=evidence.evidence_path,
            error=str(exc),
        )
        _sleep_after_failure(config)
        return 0
    lifecycle = (
        LIFECYCLE_COMPLETED if evidence.lifecycle == "completed" else failed("evaluate")
    )
    _record_cycle(
        state_dir=state_dir,
        status=status,
        tracker=tracker,
        mode=mode,
        cycle_index=cycle_index,
        lifecycle=lifecycle,
        verdict=verdict,
        comparator=comparator,
        candidates=candidates,
        proposal_path=proposal_path,
        evidence_path=evidence.evidence_path,
        error=evidence.error,
    )
    return 0


def _override_signature(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        {motion: OverrideSet.of(values).as_map() for motion, values in payload.items()},
        sort_keys=True,
    )


def _normalized(rows: Sequence[str]) -> list[str]:
    """Canonicalize stored override sets so a repeat compares equal."""

    normalized: list[str] = []
    for raw in rows:
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        normalized.append(_override_signature(payload))
    return normalized


def _sleep_after_failure(config: planner_module.PlannerConfig) -> None:
    if config.retry_delay_seconds > 0.0:
        time.sleep(config.retry_delay_seconds)
