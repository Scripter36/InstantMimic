"""Bounded lifecycle management for scheduler-backed hpo evaluators.

The launcher is deliberately small and command based.  Scheduler CLIs vary by
installation, so adapters expose their command construction and normalization
through an injectable runner.  The Slurm commands use the stable ``sbatch`` /
``squeue`` / ``sacct`` / ``scancel`` interface.  HyperQueue parsing is kept
strict and any ambiguous or unsupported response fails closed.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class SchedulerLifecycleError(RuntimeError):
    """Base class for scheduler ownership and lifecycle failures."""


class UnresolvedSubmissionError(SchedulerLifecycleError):
    """The submit acknowledgement was lost and ownership is ambiguous."""


class UnresolvedCleanupError(SchedulerLifecycleError):
    """Scheduler termination could not be confirmed within the cleanup bound."""


class UnresolvedOwnershipError(SchedulerLifecycleError):
    """An earlier scheduler invocation still has ambiguous ownership."""


@dataclass(frozen=True)
class SchedulerCommandResult:
    """Small runner result used by adapters and deterministic tests."""

    returncode: int
    stdout: str = ""


CommandRunner = Callable[[Sequence[str], float], SchedulerCommandResult]


@dataclass(frozen=True)
class SchedulerIdentity:
    backend: str
    job_id: str
    invocation_token: str
    scheduler_name: str
    context: str | None = None

    def as_record(self) -> dict[str, str]:
        values = {
            "backend": self.backend,
            "job_id": self.job_id,
            "invocation_token": self.invocation_token,
            "scheduler_name": self.scheduler_name,
        }
        if self.context:
            values["context"] = self.context
        return values


@dataclass(frozen=True)
class SchedulerObservation:
    state: str
    terminal: bool
    authoritative: bool
    returncode: int | None = None
    detail: str = ""


@dataclass
class OwnershipRecord:
    backend: str
    invocation_token: str
    scheduler_name: str
    ownership_path: Path
    created_at: float
    phase: str = "submitting"
    identity: SchedulerIdentity | None = None
    cancellation_attempts: list[dict[str, object]] = field(default_factory=list)
    cancellation_confirmed: bool = False

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "backend": self.backend,
            "invocation_token": self.invocation_token,
            "scheduler_name": self.scheduler_name,
            "created_at": self.created_at,
            "phase": self.phase,
            "cancellation_attempts": list(self.cancellation_attempts),
            "cancellation_confirmed": self.cancellation_confirmed,
        }
        if self.identity is not None:
            payload["identity"] = self.identity.as_record()
        return payload


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def default_command_runner(
    command: Sequence[str], timeout_seconds: float
) -> SchedulerCommandResult:
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=max(0.01, float(timeout_seconds)),
    )
    return SchedulerCommandResult(int(completed.returncode), completed.stdout or "")


def _transport_command(command: Sequence[str], host: str | None) -> list[str]:
    if host:
        return ["ssh", host, shlex.join(list(command))]
    return list(command)


def _remaining(deadline: float, clock: Callable[[], float]) -> float:
    return max(0.0, deadline - clock())


def _command_timeout(
    deadline: float,
    clock: Callable[[], float],
    command_timeout_seconds: float,
) -> float | None:
    remaining = _remaining(deadline, clock)
    if remaining <= 0:
        return None
    return min(max(0.01, float(command_timeout_seconds)), remaining)


def _state_from_text(text: str) -> str:
    normalized = text.upper()
    for state in (
        "BOOT_FAIL",
        "ABORTED",
        "CANCELLED",
        "CANCELED",
        "TIMEOUT",
        "DEADLINE",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "COMPLETED",
        "FINISHED",
        "FAILED",
        "ERROR",
        "RUNNING",
        "SUSPENDED",
        "PENDING",
        "WAITING",
        "QUEUED",
    ):
        if re.search(rf"\b{state}\b", normalized):
            if state == "CANCELED":
                return "CANCELLED"
            if state in {"FINISHED"}:
                return "COMPLETED"
            if state in {"ERROR"}:
                return "FAILED"
            if state in {
                "BOOT_FAIL",
                "ABORTED",
                "DEADLINE",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
                "PREEMPTED",
            }:
                return "FAILED"
            if state in {"WAITING", "QUEUED"}:
                return "PENDING"
            return state
    return "UNKNOWN"


def _exit_code_from_text(text: str) -> int | None:
    match = re.search(r"\b(-?\d+):(\d+)\b", text)
    if match:
        exit_code = int(match.group(1))
        signal_number = int(match.group(2))
        if exit_code != 0 or signal_number == 0:
            return exit_code
        return 128 + signal_number
    match = re.search(
        r"(?:exit(?:\s*code)?|exitcode)\s*[:=]\s*(-?\d+)",
        text,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d+):\d+\b", text)
    return int(match.group(1)) if match else None


def _terminal_observation(state: str, text: str = "") -> SchedulerObservation:
    terminal = state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
    return SchedulerObservation(
        state=state,
        terminal=terminal,
        authoritative=terminal,
        returncode=_exit_code_from_text(text),
        detail=text.strip(),
    )


def _hq_payload_observation(
    payload: Mapping[str, object], raw: str
) -> SchedulerObservation:
    """Normalize the documented HQ task-info JSON shapes conservatively."""

    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        if not tasks or any(not isinstance(item, Mapping) for item in tasks):
            return SchedulerObservation("UNKNOWN", False, False, detail=raw.strip())
        observations = [_hq_payload_observation(item, raw) for item in tasks]
        if any(item.state == "UNKNOWN" for item in observations):
            return SchedulerObservation("UNKNOWN", False, False, detail=raw.strip())
        if observations and all(
            item.terminal and item.authoritative for item in observations
        ):
            if any(item.state in {"FAILED", "TIMEOUT"} for item in observations):
                state = "FAILED"
            elif any(item.state == "CANCELLED" for item in observations):
                state = "CANCELLED"
            else:
                state = "COMPLETED"
            return SchedulerObservation(
                state,
                True,
                True,
                (
                    next(
                        (
                            item.returncode
                            for item in observations
                            if item.state in {"FAILED", "TIMEOUT"}
                            and item.returncode not in (None, 0)
                        ),
                        1,
                    )
                    if state == "FAILED"
                    else next(
                        (
                            item.returncode
                            for item in observations
                            if item.state == "CANCELLED"
                            and item.returncode not in (None, 0)
                        ),
                        1,
                    )
                    if state == "CANCELLED"
                    else next(
                        (
                            item.returncode
                            for item in observations
                            if item.returncode is not None
                        ),
                        0,
                    )
                ),
                raw.strip(),
            )
        if observations:
            return SchedulerObservation("RUNNING", False, False, detail=raw.strip())

    info = payload.get("info")
    task_stats = payload.get("task_stats")
    if task_stats is None and isinstance(info, Mapping):
        task_stats = info.get("task_stats")
    if isinstance(task_stats, Mapping):
        known_keys = {
            "waiting",
            "pending",
            "running",
            "assigned",
            "started",
            "finished",
            "completed",
            "failed",
            "error",
            "aborted",
            "canceled",
            "cancelled",
        }
        counts: dict[str, int] = {}
        for key, value in task_stats.items():
            normalized_key = str(key).lower()
            if (
                normalized_key not in known_keys
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                if value != 0:
                    return SchedulerObservation(
                        "UNKNOWN", False, False, detail=raw.strip()
                    )
                continue
            counts[normalized_key] = value
        total = sum(counts.values())
        active = sum(
            counts.get(key, 0)
            for key in ("waiting", "pending", "running", "assigned", "started")
        )
        if total > 0 and active == 0:
            if (
                counts.get("failed", 0)
                or counts.get("error", 0)
                or counts.get("aborted", 0)
            ):
                return SchedulerObservation("FAILED", True, True, 1, raw.strip())
            if counts.get("canceled", 0) or counts.get("cancelled", 0):
                return SchedulerObservation("CANCELLED", True, True, 1, raw.strip())
            return SchedulerObservation("COMPLETED", True, True, 0, raw.strip())
        if active > 0:
            return SchedulerObservation("RUNNING", False, False, detail=raw.strip())

    state_value = payload.get("state", payload.get("status", ""))
    state = _state_from_text(str(state_value))
    exit_value = payload.get(
        "exit_code", payload.get("exitCode", payload.get("exit_status"))
    )
    try:
        returncode = int(exit_value) if exit_value is not None else None
    except (TypeError, ValueError):
        returncode = None
    observation = _terminal_observation(state, raw)
    return SchedulerObservation(
        observation.state,
        observation.terminal,
        observation.authoritative,
        returncode if returncode is not None else observation.returncode,
        raw.strip(),
    )


class SchedulerAdapter(Protocol):
    backend: str

    def submit_command(self) -> list[str]: ...

    def parse_submission(self, output: str) -> SchedulerIdentity: ...

    def reconcile(
        self, runner: CommandRunner, timeout_seconds: float
    ) -> SchedulerIdentity | None: ...

    def query(
        self,
        identity: SchedulerIdentity,
        runner: CommandRunner,
        timeout_seconds: float,
    ) -> SchedulerObservation: ...

    def cancel(
        self,
        identity: SchedulerIdentity,
        runner: CommandRunner,
        timeout_seconds: float,
    ) -> SchedulerCommandResult: ...


def _reject_owned_arguments(
    args: Sequence[str], *, backend: str, owned_prefixes: tuple[str, ...]
) -> None:
    for arg in args:
        token = str(arg)
        if token == "--":
            raise ValueError(f"{backend} scheduler argument '--' is lifecycle-owned.")
        if any(
            token == prefix
            or token.startswith(prefix + "=")
            or (
                prefix.startswith("-")
                and not prefix.startswith("--")
                and token.startswith(prefix)
            )
            for prefix in owned_prefixes
        ):
            raise ValueError(
                f"{backend} scheduler argument {token!r} is lifecycle-owned."
            )
        if token.startswith("--") and "=" not in token:
            # The value-taking lifecycle flags are rejected separately by their
            # bare spelling; this protects ``--flag value`` forms too.
            continue


def validate_slurm_sbatch_args(args: Sequence[str]) -> None:
    if any(str(arg) == ":" for arg in args):
        raise ValueError("Slurm scheduler argument ':' is lifecycle-owned.")
    _reject_owned_arguments(
        args,
        backend="Slurm",
        owned_prefixes=(
            "--wait",
            "--parsable",
            "--job-name",
            "--name",
            "--cluster",
            "--clusters",
            "-M",
            "--comment",
            "--output",
            "--error",
            "--nodes",
            "--ntasks",
            "--gpus-per-node",
            "--array",
            "--het-group",
            "--wrap",
            "-J",
            "-M",
            "-o",
            "-e",
            "-N",
            "-n",
            "-G",
            "-a",
        ),
    )


def validate_hq_submit_args(args: Sequence[str]) -> None:
    _reject_owned_arguments(
        args,
        backend="HQ",
        owned_prefixes=(
            "--wait",
            "--output-mode",
            "--name",
            "--server",
            "--server-dir",
            "--id",
            "--array",
            "--stdout",
            "--stderr",
            "--output",
            "--error",
        ),
    )


def _parse_slurm_identity(
    output: str, *, token: str, scheduler_name: str, context: str | None
) -> SchedulerIdentity:
    values = [line.strip() for line in output.splitlines() if line.strip()]
    if len(values) != 1:
        raise SchedulerLifecycleError("Slurm submit did not return one job identity.")
    match = re.fullmatch(r"(?P<job>\d+)(?:;(?P<cluster>[A-Za-z0-9_.-]+))?", values[0])
    if match is None:
        raise SchedulerLifecycleError("Invalid Slurm --parsable job identity.")
    cluster = match.group("cluster") or context
    return SchedulerIdentity(
        "slurm", match.group("job"), token, scheduler_name, cluster
    )


@dataclass
class SlurmAdapter:
    script_path: Path
    log_path: Path
    scheduler_name: str
    invocation_token: str
    max_parallel: int
    sbatch_args: tuple[str, ...] = ()
    host: str | None = None
    context: str | None = None
    clock: Callable[[], float] = time.monotonic

    backend: str = "slurm"

    def __post_init__(self) -> None:
        validate_slurm_sbatch_args(self.sbatch_args)

    def submit_command(self) -> list[str]:
        return [
            "sbatch",
            "--parsable",
            f"--job-name={self.scheduler_name}",
            "--nodes=1",
            "--ntasks=1",
            f"--gpus-per-node={int(self.max_parallel)}",
            f"--output={self.log_path}",
            f"--error={self.log_path}",
            *self.sbatch_args,
            str(self.script_path),
        ]

    def parse_submission(self, output: str) -> SchedulerIdentity:
        return _parse_slurm_identity(
            output,
            token=self.invocation_token,
            scheduler_name=self.scheduler_name,
            context=self.context,
        )

    def reconcile(
        self, runner: CommandRunner, timeout_seconds: float
    ) -> SchedulerIdentity | None:
        result = runner(
            _transport_command(
                [
                    "squeue",
                    "-h",
                    f"--name={self.scheduler_name}",
                    "--format=%i|%j",
                ],
                self.host,
            ),
            timeout_seconds,
        )
        if result.returncode != 0:
            raise SchedulerLifecycleError("Slurm submission reconciliation failed.")
        matches: list[SchedulerIdentity] = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split("|", 1)]
            if len(fields) != 2 or fields[1] != self.scheduler_name:
                continue
            if not re.fullmatch(r"\d+", fields[0]):
                continue
            matches.append(
                SchedulerIdentity(
                    "slurm",
                    fields[0],
                    self.invocation_token,
                    self.scheduler_name,
                    self.context,
                )
            )
        if len(matches) > 1:
            raise UnresolvedSubmissionError(
                "Multiple Slurm jobs matched the invocation token."
            )
        return matches[0] if matches else None

    def query(
        self,
        identity: SchedulerIdentity,
        runner: CommandRunner,
        timeout_seconds: float,
    ) -> SchedulerObservation:
        deadline = self.clock() + max(0.0, float(timeout_seconds))
        cluster_arg = ["-M", identity.context] if identity.context else []
        squeue = ["squeue", *cluster_arg, "-h", "-j", identity.job_id, "--format=%T"]
        queue_timeout = _remaining(deadline, self.clock)
        if queue_timeout <= 0:
            return SchedulerObservation("UNKNOWN", False, False)
        queue_result = runner(_transport_command(squeue, self.host), queue_timeout)
        if queue_result.returncode == 0 and queue_result.stdout.strip():
            state = _state_from_text(queue_result.stdout)
            if state not in {"UNKNOWN", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}:
                return SchedulerObservation(
                    state, False, False, detail=queue_result.stdout.strip()
                )
        sacct = [
            "sacct",
            *cluster_arg,
            "-n",
            "-X",
            "-P",
            "-j",
            identity.job_id,
            "--format=JobIDRaw,State,ExitCode",
        ]
        accounting_timeout = _remaining(deadline, self.clock)
        if accounting_timeout <= 0:
            return SchedulerObservation("UNKNOWN", False, False)
        accounting = runner(_transport_command(sacct, self.host), accounting_timeout)
        if accounting.returncode != 0 or not accounting.stdout.strip():
            return SchedulerObservation(
                "UNKNOWN", False, False, detail=accounting.stdout.strip()
            )
        rows = [line.strip() for line in accounting.stdout.splitlines() if line.strip()]
        matching = [
            row
            for row in rows
            if row.split("|", 1)[0].split(".", 1)[0] == identity.job_id
        ]
        if not matching:
            return SchedulerObservation(
                "UNKNOWN", False, False, detail=accounting.stdout.strip()
            )
        state = _state_from_text(matching[-1])
        return _terminal_observation(state, matching[-1])

    def cancel(
        self,
        identity: SchedulerIdentity,
        runner: CommandRunner,
        timeout_seconds: float,
    ) -> SchedulerCommandResult:
        command = [
            "scancel",
            *(["-M", identity.context] if identity.context else []),
            identity.job_id,
        ]
        return runner(_transport_command(command, self.host), timeout_seconds)


def _parse_hq_identity(
    output: str, *, token: str, scheduler_name: str, context: str | None
) -> SchedulerIdentity:
    matches = re.findall(
        r"(?:task|job)\s*(?:id)?\s*[:=]?\s*(\d+)\b",
        output,
        re.IGNORECASE,
    )
    if not matches:
        matches = [
            line.strip()
            for line in output.splitlines()
            if re.fullmatch(r"\d+", line.strip())
        ]
    if len(set(matches)) != 1:
        raise SchedulerLifecycleError(
            "HQ submit did not return one unambiguous task identity."
        )
    return SchedulerIdentity("hq", matches[0], token, scheduler_name, context)


@dataclass
class HQAdapter:
    command: list[str]
    scheduler_name: str
    invocation_token: str
    max_parallel: int
    submit_args: tuple[str, ...] = ()
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    context: str | None = None

    backend: str = "hq"

    def __post_init__(self) -> None:
        validate_hq_submit_args(self.submit_args)

    def _base_command(self) -> list[str]:
        command = [
            "hq",
            "--output-mode=quiet",
            *([f"--server-dir={self.context}"] if self.context else []),
            "submit",
            f"--name={self.scheduler_name}",
            "--resource",
            f"gpus/nvidia={int(self.max_parallel)}",
            *self.submit_args,
        ]
        if self.stdout_path is not None:
            command.extend(["--stdout", str(self.stdout_path)])
        if self.stderr_path is not None:
            command.extend(["--stderr", str(self.stderr_path)])
        command.extend(["--", *self.command])
        return command

    def submit_command(self) -> list[str]:
        return self._base_command()

    def parse_submission(self, output: str) -> SchedulerIdentity:
        return _parse_hq_identity(
            output,
            token=self.invocation_token,
            scheduler_name=self.scheduler_name,
            context=self.context,
        )

    def resolve_context(self, runner: CommandRunner, timeout_seconds: float) -> None:
        """Capture and pin the documented HQ server directory context."""

        if self.context:
            return

        result = runner(["hq", "--output-mode=json", "server", "info"], timeout_seconds)
        if result.returncode != 0:
            raise SchedulerLifecycleError("HQ server context could not be confirmed.")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise SchedulerLifecycleError(
                "HQ server context returned invalid JSON."
            ) from exc
        if not isinstance(payload, Mapping):
            raise SchedulerLifecycleError("HQ server context JSON is not an object.")

        context = payload.get("server_dir")
        if not isinstance(context, str) or not context.strip():
            raise SchedulerLifecycleError("HQ server context JSON has no server_dir.")
        self.context = context.strip()

    def reconcile(
        self, runner: CommandRunner, timeout_seconds: float
    ) -> SchedulerIdentity | None:
        # HyperQueue's stable quiet-submit path does not provide a safe
        # server-side token query across CLI versions.  A lost acknowledgement
        # therefore remains unresolved instead of matching by a human name.
        return None

    def query(
        self,
        identity: SchedulerIdentity,
        runner: CommandRunner,
        timeout_seconds: float,
    ) -> SchedulerObservation:
        result = runner(
            [
                "hq",
                "--output-mode=json",
                *([f"--server-dir={identity.context}"] if identity.context else []),
                "job",
                "info",
                identity.job_id,
            ],
            timeout_seconds,
        )
        if result.returncode != 0:
            return SchedulerObservation(
                "UNKNOWN", False, False, detail=result.stdout.strip()
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return SchedulerObservation(
                "UNKNOWN", False, False, detail=result.stdout.strip()
            )
        if not isinstance(payload, Mapping):
            return SchedulerObservation(
                "UNKNOWN", False, False, detail=result.stdout.strip()
            )
        return _hq_payload_observation(payload, result.stdout)

    def cancel(
        self,
        identity: SchedulerIdentity,
        runner: CommandRunner,
        timeout_seconds: float,
    ) -> SchedulerCommandResult:
        return runner(
            [
                "hq",
                *([f"--server-dir={identity.context}"] if identity.context else []),
                "job",
                "cancel",
                identity.job_id,
            ],
            timeout_seconds,
        )


def _write_phase(record: OwnershipRecord, phase: str) -> None:
    record.phase = phase
    _atomic_write_json(record.ownership_path, record.payload())


def _assert_no_unresolved_ownership(ownership_path: Path) -> None:
    safe_phases = {"terminal", "terminal_after_cancel", "submission_deadline"}
    try:
        records = sorted(ownership_path.parent.glob("*.ownership.json"))
    except OSError as exc:
        raise UnresolvedOwnershipError(
            f"Cannot inspect scheduler ownership records in {ownership_path.parent}."
        ) from exc
    for record_path in records:
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise UnresolvedOwnershipError(
                f"Unreadable scheduler ownership record blocks submission: "
                f"{record_path}."
            ) from exc
        if not isinstance(payload, Mapping):
            raise UnresolvedOwnershipError(
                f"Malformed scheduler ownership record blocks submission: "
                f"{record_path}."
            )
        record_backend = payload.get("backend")
        phase = payload.get("phase")
        if not isinstance(record_backend, str) or not record_backend.strip():
            raise UnresolvedOwnershipError(
                f"Malformed scheduler ownership record blocks submission: "
                f"{record_path}."
            )
        if phase not in safe_phases:
            raise UnresolvedOwnershipError(
                f"Unresolved scheduler ownership blocks new submission: {record_path}."
            )
        core_fields_valid = (
            isinstance(payload.get("invocation_token"), str)
            and bool(str(payload["invocation_token"]).strip())
            and isinstance(payload.get("scheduler_name"), str)
            and bool(str(payload["scheduler_name"]).strip())
            and isinstance(payload.get("created_at"), (int, float))
            and not isinstance(payload.get("created_at"), bool)
        )
        identity = payload.get("identity")
        if phase == "submission_deadline":
            if not core_fields_valid or identity is not None:
                raise UnresolvedOwnershipError(
                    f"Malformed scheduler ownership record blocks submission: "
                    f"{record_path}."
                )
            continue
        if not core_fields_valid or not isinstance(identity, Mapping):
            raise UnresolvedOwnershipError(
                f"Malformed scheduler ownership record blocks submission: "
                f"{record_path}."
            )
        identity_fields_valid = (
            identity.get("backend") == record_backend
            and identity.get("job_id")
            and isinstance(identity.get("job_id"), str)
            and identity.get("invocation_token") == payload.get("invocation_token")
            and identity.get("scheduler_name") == payload.get("scheduler_name")
        )
        if not identity_fields_valid:
            raise UnresolvedOwnershipError(
                f"Malformed scheduler ownership record blocks submission: "
                f"{record_path}."
            )
        if (
            phase == "terminal_after_cancel"
            and payload.get("cancellation_confirmed") is not True
        ):
            raise UnresolvedOwnershipError(
                f"Malformed scheduler ownership record blocks submission: "
                f"{record_path}."
            )


def _acquire_ownership_gate(ownership_path: Path) -> tuple[Path, int]:
    ownership_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path = ownership_path.parent / ".scheduler_lifecycle.gate"
    try:
        descriptor = os.open(
            gate_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise UnresolvedOwnershipError(
            f"Scheduler ownership gate already exists: {gate_path}."
        ) from exc
    except OSError as exc:
        raise UnresolvedOwnershipError(
            f"Cannot create scheduler ownership gate: {gate_path}."
        ) from exc
    return gate_path, descriptor


def _release_ownership_gate(gate_path: Path, descriptor: int) -> None:
    os.close(descriptor)
    try:
        gate_path.unlink()
    except OSError:
        # Leaving the sentinel is fail-closed for a future controller.
        return


def _normal_returncode(observation: SchedulerObservation) -> int:
    if observation.state != "COMPLETED":
        return observation.returncode if observation.returncode not in (None, 0) else 1
    return 0 if observation.returncode is None else int(observation.returncode)


def _cleanup_owned_job(
    *,
    adapter: SchedulerAdapter,
    identity: SchedulerIdentity,
    record: OwnershipRecord,
    runner: CommandRunner,
    cleanup_deadline: float,
    poll_interval_seconds: float,
    command_timeout_seconds: float,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> bool:
    _write_phase(record, "cancelling")
    cancel_timeout = _command_timeout(cleanup_deadline, clock, command_timeout_seconds)
    if cancel_timeout is None:
        _write_phase(record, "cleanup_unresolved")
        return False
    try:
        cancel_result = adapter.cancel(
            identity,
            runner,
            cancel_timeout,
        )
        record.cancellation_attempts.append(
            {"returncode": cancel_result.returncode, "at": time.time()}
        )
        _atomic_write_json(record.ownership_path, record.payload())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        record.cancellation_attempts.append({"error": str(exc), "at": time.time()})
        _atomic_write_json(record.ownership_path, record.payload())

    while _remaining(cleanup_deadline, clock) > 0:
        query_timeout = _command_timeout(
            cleanup_deadline, clock, command_timeout_seconds
        )
        if query_timeout is None:
            break
        try:
            observation = adapter.query(identity, runner, query_timeout)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            observation = SchedulerObservation("UNKNOWN", False, False)
        if observation.terminal and observation.authoritative:
            record.cancellation_confirmed = True
            _write_phase(record, "terminal_after_cancel")
            return True
        sleeper(
            min(max(0.01, poll_interval_seconds), _remaining(cleanup_deadline, clock))
        )
    _write_phase(record, "cleanup_unresolved")
    return False


def run_scheduler_lifecycle(
    *,
    adapter: SchedulerAdapter,
    ownership_path: Path,
    timeout_seconds: float,
    cleanup_timeout_seconds: float = 30.0,
    command_timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 0.25,
    runner: CommandRunner | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    deadline: float | None = None,
) -> int:
    """Submit, monitor, and own one scheduler workload to terminal state."""

    if timeout_seconds <= 0 or cleanup_timeout_seconds <= 0:
        raise ValueError("scheduler lifecycle deadlines must be > 0")
    command_runner = runner or default_command_runner
    if isinstance(adapter, SlurmAdapter):
        adapter.clock = clock
    invocation_token = adapter.invocation_token
    scheduler_name = adapter.scheduler_name
    record = OwnershipRecord(
        backend=adapter.backend,
        invocation_token=invocation_token,
        scheduler_name=scheduler_name,
        ownership_path=ownership_path,
        created_at=time.time(),
    )
    gate_path, gate_descriptor = _acquire_ownership_gate(ownership_path)
    try:
        _assert_no_unresolved_ownership(ownership_path)
        _atomic_write_json(ownership_path, record.payload())
    except BaseException:
        os.close(gate_descriptor)
        raise
    _release_ownership_gate(gate_path, gate_descriptor)
    deadline = clock() + float(timeout_seconds) if deadline is None else float(deadline)
    identity: SchedulerIdentity | None = None
    if _remaining(deadline, clock) <= 0:
        _write_phase(record, "submission_deadline")
        return 124
    try:
        submit_timeout = _command_timeout(deadline, clock, command_timeout_seconds)
        if submit_timeout is None:
            _write_phase(record, "submission_deadline")
            return 124
        submit_result = command_runner(
            adapter.submit_command(),
            submit_timeout,
        )
        if submit_result.returncode == 0:
            identity = adapter.parse_submission(submit_result.stdout)
            record.identity = identity
            _write_phase(record, "monitoring")
        else:
            raise SchedulerLifecycleError(
                f"{adapter.backend} submission failed with return code "
                f"{submit_result.returncode}."
            )
    except BaseException as exc:
        reconcile_timeout = _command_timeout(deadline, clock, command_timeout_seconds)
        try:
            identity = (
                adapter.reconcile(command_runner, reconcile_timeout)
                if reconcile_timeout is not None
                else None
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            subprocess.TimeoutExpired,
            KeyboardInterrupt,
            SystemExit,
        ):
            identity = None
        if identity is None:
            _write_phase(record, "submission_unresolved")
            raise UnresolvedSubmissionError(
                f"{adapter.backend} submission ownership is unresolved; see "
                f"{ownership_path}."
            ) from exc
        record.identity = identity
        _write_phase(record, "monitoring")
        confirmed = _cleanup_owned_job(
            adapter=adapter,
            identity=identity,
            record=record,
            runner=command_runner,
            cleanup_deadline=clock() + float(cleanup_timeout_seconds),
            poll_interval_seconds=poll_interval_seconds,
            command_timeout_seconds=command_timeout_seconds,
            clock=clock,
            sleeper=sleeper,
        )
        if not confirmed:
            raise UnresolvedCleanupError(
                f"{adapter.backend} cleanup could not confirm terminal state; see "
                f"{ownership_path}."
            ) from exc
        raise

    assert identity is not None
    try:
        while _remaining(deadline, clock) > 0:
            query_timeout = _command_timeout(deadline, clock, command_timeout_seconds)
            if query_timeout is None:
                break
            observation = adapter.query(
                identity,
                command_runner,
                query_timeout,
            )
            if observation.terminal and observation.authoritative:
                _write_phase(record, "terminal")
                return _normal_returncode(observation)
            sleeper(min(max(0.01, poll_interval_seconds), _remaining(deadline, clock)))
    except BaseException as exc:
        cleanup_deadline = clock() + float(cleanup_timeout_seconds)
        confirmed = _cleanup_owned_job(
            adapter=adapter,
            identity=identity,
            record=record,
            runner=command_runner,
            cleanup_deadline=cleanup_deadline,
            poll_interval_seconds=poll_interval_seconds,
            command_timeout_seconds=command_timeout_seconds,
            clock=clock,
            sleeper=sleeper,
        )
        if not confirmed:
            raise UnresolvedCleanupError(
                f"{adapter.backend} cleanup could not confirm terminal state; see "
                f"{ownership_path}."
            ) from exc
        raise

    cleanup_deadline = clock() + float(cleanup_timeout_seconds)
    confirmed = _cleanup_owned_job(
        adapter=adapter,
        identity=identity,
        record=record,
        runner=command_runner,
        cleanup_deadline=cleanup_deadline,
        poll_interval_seconds=poll_interval_seconds,
        command_timeout_seconds=command_timeout_seconds,
        clock=clock,
        sleeper=sleeper,
    )
    if not confirmed:
        raise UnresolvedCleanupError(
            f"{adapter.backend} cleanup could not confirm terminal state; see "
            f"{ownership_path}."
        )
    return 124
