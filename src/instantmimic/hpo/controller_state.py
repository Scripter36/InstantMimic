"""The one state layer: every file a run writes, written in one place.

Five artifacts describe a run, and this module is the only writer of all of
them:

* ``status.yaml`` -- where the run is, and what the comparator kept.
* ``results.tsv`` -- one row per candidate, ever.
* ``progress.md`` -- one entry per cycle, in the program's documented shape.
* ``best_result.md`` -- the comparator's rendering of what it kept.
* ``final.md`` -- one cycle's closing report, under ``cycles/NNN/``.

Two vocabularies used to be mixed in one ``status`` field. They are separate
now: **lifecycle** says what happened to the cycle (``completed`` or
``failed:<phase>``) and is common to both objective modes; **decision** is the
comparator's own word for what it did with the result, and only a comparator
may produce it.

``schema_version`` is 2. A state directory written by the two-controller
runtime is not resumable: its status file mixed the vocabularies and its result
rows filled disjoint column subsets per mode. Such a run refuses to resume with
a message naming its own directory; the files stay readable as history.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

STATUS_SCHEMA_VERSION = 2

# One row per candidate. Lifecycle columns plus one scalar projection; every
# per-strategy number lives in `metrics_json`, so a new comparator never needs
# a new column.
RESULTS_TSV_FIELDS = (
    "cycle",
    "candidate",
    "parent",
    "mode",
    "status",
    "decision",
    "objective_value",
    "evidence_path",
    "proposal_path",
    "error",
    "effective_overrides_json",
    "metrics_json",
)

RUN_LEVEL_STATE_FILES = (
    "results.tsv",
    "status.yaml",
    "tracking.yaml",
    "progress.md",
    "best_result.md",
    "strategy.md",
    # Whatever the comparator writes for itself.
    "objective/*",
)

CYCLE_ARTIFACT_FILES = (
    "evaluator.log",
    "planning.md",
    "planning.prompt.md",
    "proposal.md",
    "proposal.prompt.md",
    "proposal.yaml",
    "evaluator_task.yaml",
    "final.md",
    "validation.log",
)

CONTROLLER_LOCK_FILENAME = ".run.lock"

# Cycle lifecycle. A comparator's verdict is recorded beside this, never in it.
LIFECYCLE_COMPLETED = "completed"
LIFECYCLE_DRY_RUN = "dry_run"
LIFECYCLE_FAILED_PREFIX = "failed"
PHASES = ("planning", "prepare", "validation", "evaluate", "finalize")


def failed(phase: str) -> str:
    """The lifecycle status for a cycle that failed in ``phase``."""

    return f"{LIFECYCLE_FAILED_PREFIX}:{phase}"


def is_failed(lifecycle: str) -> bool:
    """Whether a recorded lifecycle status means the cycle did not finish."""

    return str(lifecycle).strip().lower().startswith(LIFECYCLE_FAILED_PREFIX)


_STATUS_FIELDS = (
    "schema_version",
    "mode",
    "task_file",
    "task_name",
    "cycle",
    "active_cycle",
    "active_phase",
    "last_status",
    "last_decision",
    "last_error",
    "last_result",
    "last_proposal_path",
    "last_summary_path",
    "last_message_path",
    "cycles_since_improvement",
    "best",
    "objective_state",
    "objective_config",
)


def cycle_artifact_dir(state_dir: Path, cycle_index: int) -> Path:
    path = state_dir / "cycles" / f"{int(cycle_index):03d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def objective_artifact_dir(state_dir: Path) -> Path:
    """Where the comparator keeps whatever state it needs."""

    path = state_dir / "objective"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def controller_state_lock(state_dir: Path) -> Iterator[bool]:
    """Try to hold the run lock for the complete controller invocation."""

    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / CONTROLLER_LOCK_FILENAME
    lock_file = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        acquired = True
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid: {os.getpid()}\n")
        lock_file.flush()
        yield True
    finally:
        if acquired:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def status_defaults(mode: str = "") -> dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "mode": mode,
        "task_file": "",
        "task_name": "",
        "cycle": 0,
        "active_cycle": None,
        "active_phase": "",
        "last_status": "",
        "last_decision": "",
        "last_error": "",
        "last_result": "",
        "last_proposal_path": "",
        "last_summary_path": "",
        "last_message_path": "",
        "cycles_since_improvement": 0,
        "best": {
            "cycle": None,
            "candidate": "",
            "value": None,
            "evidence_path": "",
        },
        "objective_state": {},
        "objective_config": None,
    }


def normalize_controller_status(
    payload: Mapping[str, Any] | None, *, mode: str = ""
) -> dict[str, Any]:
    """Fill in a status mapping without inventing anything mode-specific."""

    normalized = status_defaults(mode)
    raw = dict(payload or {})
    for key in _STATUS_FIELDS:
        if key in raw:
            normalized[key] = raw[key]
    normalized["schema_version"] = STATUS_SCHEMA_VERSION
    normalized["mode"] = mode or str(raw.get("mode", "") or "")

    best = dict(status_defaults()["best"])
    if isinstance(raw.get("best"), Mapping):
        best.update(dict(raw["best"]))
    normalized["best"] = best
    # The comparator owns this blob whole; the controller stores it and hands
    # it back, and never reads into it.
    objective_state = raw.get("objective_state")
    normalized["objective_state"] = (
        dict(objective_state) if isinstance(objective_state, Mapping) else {}
    )
    return normalized


class IncompatibleStateError(ValueError):
    """A state directory this runtime cannot resume."""


def load_controller_status(path: Path, *, mode: str = "") -> dict[str, Any]:
    if not path.exists():
        return normalize_controller_status(None, mode=mode)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        backup_path = _backup_corrupt_status(path)
        raise ValueError(
            f"Corrupt hpo status file '{path}'. "
            f"The original was preserved at '{backup_path}'."
        ) from exc
    if not isinstance(loaded, Mapping):
        backup_path = _backup_corrupt_status(path)
        raise ValueError(
            f"Corrupt hpo status file '{path}': expected a mapping. "
            f"The original was preserved at '{backup_path}'."
        )
    version = int(loaded.get("schema_version", 1) or 1)
    if version != STATUS_SCHEMA_VERSION:
        raise IncompatibleStateError(
            f"HPO state at '{path.parent}' is schema_version {version}; "
            f"this runtime writes {STATUS_SCHEMA_VERSION} and does not migrate. "
            "Start a fresh run; the existing directory stays readable as history."
        )
    return normalize_controller_status(loaded, mode=mode)


def _backup_corrupt_status(status_path: Path) -> Path:
    """Copy malformed status state to a unique sibling before refusing."""

    status_path = status_path.resolve()
    index = 0
    while True:
        suffix = ".corrupt" if index == 0 else f".corrupt.{index}"
        backup_path = status_path.with_name(status_path.name + suffix)
        try:
            with backup_path.open("xb") as backup, status_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    backup.write(chunk)
        except FileExistsError:
            index += 1
            continue
        return backup_path


def write_controller_status(
    path: Path, payload: Mapping[str, Any], *, mode: str = ""
) -> None:
    normalized = normalize_controller_status(payload, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


DEFAULT_STRATEGY = "\n".join(
    [
        "# Current Strategy",
        "",
        "## Best Baseline",
        "",
        "No periodic strategy has been recorded yet.",
        "",
        "## What Seems Promising",
        "",
        "- Establish a trustworthy baseline first.",
        "",
        "## What To Avoid",
        "",
        "- Avoid broad speculative jumps before the first baseline.",
        "",
        "## Next Experiment",
        "",
        "- Compare high-impact semantic axes.",
        "",
        "## Wildcards",
        "",
        "- No creativity notes recorded yet.",
        "",
    ]
)


def ensure_controller_state(
    *, state_dir: Path, task_path: Path, task_name: str, mode: str = ""
) -> dict[str, Any]:
    """Create the run's state files if they are missing; return the status."""

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "cycles").mkdir(parents=True, exist_ok=True)
    status_path = state_dir / "status.yaml"
    status = load_controller_status(status_path, mode=mode)
    status["task_file"] = str(task_path)
    status["task_name"] = task_name
    write_controller_status(status_path, status, mode=mode)

    for path, default in (
        (state_dir / "strategy.md", DEFAULT_STRATEGY),
        (
            state_dir / "progress.md",
            "# Progress Log\n\nAppend one section per cycle.\n",
        ),
        (state_dir / "best_result.md", "# Best Result\n\nNo completed cycle yet.\n"),
    ):
        if not path.exists():
            path.write_text(default, encoding="utf-8")

    results_path = state_dir / "results.tsv"
    if not results_path.exists() or results_path.stat().st_size == 0:
        results_path.write_text("\t".join(RESULTS_TSV_FIELDS) + "\n", encoding="utf-8")
    return status


def append_controller_result(path: Path, row: Mapping[str, Any]) -> None:
    """Append one candidate row. The only writer of ``results.tsv``."""

    payload: dict[str, Any] = {}
    for key in RESULTS_TSV_FIELDS:
        value = row.get(key, "")
        if key.endswith("_json") and not isinstance(value, str):
            value = json.dumps(value, sort_keys=False) if value else ""
        elif isinstance(value, (Mapping, list, tuple)):
            value = json.dumps(value, sort_keys=False)
        payload[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=RESULTS_TSV_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(payload)


def read_controller_results(path: Path) -> list[dict[str, str]]:
    """Read ``results.tsv``. The only reader, matching the only writer."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return []
        if tuple(reader.fieldnames) != RESULTS_TSV_FIELDS:
            raise IncompatibleStateError(
                f"'{path}' was written with a different results header; "
                "this runtime does not migrate legacy result files."
            )
        return [dict(row) for row in reader]


def append_progress_entry(
    path: Path,
    *,
    cycle_index: int,
    lifecycle: str,
    decision: str,
    proposal_path: str = "",
    evidence_path: str = "",
    comparator_lines: Sequence[str] = (),
) -> None:
    """Append one ``progress.md`` entry in the program's documented shape."""

    lines = [
        "",
        f"## Cycle {int(cycle_index)}",
        "",
        f"- Status: {lifecycle}",
        f"- Decision: {decision or '(none)'}",
        f"- Proposal: {proposal_path or '(none)'}",
        f"- Evidence: {evidence_path or '(none)'}",
    ]
    lines.extend(str(line) for line in comparator_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_best_result(path: Path, body: str) -> None:
    """Write ``best_result.md``: one header, then the comparator's rendering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Best Result\n\n" + body.rstrip("\n") + "\n", encoding="utf-8")


def write_cycle_final_report(
    path: Path,
    *,
    cycle_index: int,
    lifecycle: str,
    decision: str,
    error: str = "",
    comparator_lines: Sequence[str] = (),
) -> None:
    """Write one cycle's ``final.md``. The only writer."""

    lines = [
        f"# Cycle {int(cycle_index)} Final Report",
        "",
        f"- Status: {lifecycle}",
        f"- Decision: {decision or '(none)'}",
    ]
    lines.extend(str(line) for line in comparator_lines)
    if error:
        lines.extend(["", "## Error", "", error])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_cycle_transaction(
    state_dir: Path, status: Mapping[str, Any], cycle_index: int
) -> None:
    """Keep status.yaml authoritative if interruption lands during finalization.

    Comparators may replace their archives before the controller commits status.
    Snapshot the prior objective files once; a restarted cycle restores them and
    removes rows/progress that were written without the status commit.
    """
    cycle_root = cycle_artifact_dir(state_dir, cycle_index)
    snapshot = cycle_root / "objective_before"
    objective_dir = objective_artifact_dir(state_dir)
    if not snapshot.exists():
        temporary = cycle_root / ".objective_before.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(objective_dir, temporary)
        temporary.rename(snapshot)
        return
    # This cycle has no status commit, otherwise the controller would run N+1.
    shutil.rmtree(objective_dir)
    shutil.copytree(snapshot, objective_dir)
    completed = int(status.get("cycle", 0) or 0)
    results_path = state_dir / "results.tsv"
    rows = read_controller_results(results_path)
    committed = [row for row in rows if int(row["cycle"]) <= completed]
    if len(committed) != len(rows):
        temporary_results = results_path.with_suffix(".recovering.tsv")
        with temporary_results.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=RESULTS_TSV_FIELDS,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(committed)
        os.replace(temporary_results, results_path)
    progress_path = state_dir / "progress.md"
    progress = progress_path.read_text(encoding="utf-8")
    for match in re.finditer(r"^## Cycle (\d+)\s*$", progress, flags=re.MULTILINE):
        if int(match.group(1)) > completed:
            progress_path.write_text(
                progress[: match.start()].rstrip() + "\n", encoding="utf-8"
            )
            break
