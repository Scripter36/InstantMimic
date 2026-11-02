"""Shared evaluator launch policy for hpo cycle modules.

Scalar and Pareto hpo differ in how they choose and score candidates,
not in how an evaluator process reaches a GPU. Both therefore resolve the same
``runtime.evaluator`` settings, build the same wrapper commands, and dispatch
through :func:`run_launched_evaluator` here. Adding a launcher in this module
adds it for both cycle modules.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import signal
import subprocess
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, ListConfig

from instantmimic.hpo.common import (
    cfg_int as _common_cfg_int,
)
from instantmimic.hpo.common import (
    cfg_str as _common_cfg_str,
)
from instantmimic.hpo.common import (
    section_cfg as _common_section_cfg,
)
from instantmimic.hpo.scheduler_lifecycle import (
    HQAdapter,
    SchedulerCommandResult,
    SlurmAdapter,
    run_scheduler_lifecycle,
    validate_hq_submit_args,
    validate_slurm_sbatch_args,
)

SUPPORTED_EVALUATOR_LAUNCHERS = {"local", "hq", "slurm"}
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 24.0 * 60.0 * 60.0
_EVALUATOR_TERMINATION_GRACE_SECONDS = 5.0
EVALUATOR_JOB_GROUPS_ENV = "INSTANTMIMIC_EVALUATOR_JOB_GROUPS"
EVALUATOR_TERMINATION_GRACE_ENV = "INSTANTMIMIC_EVALUATOR_TERMINATION_GRACE_SECONDS"
_OWNED_CHILD_TERMINATION_GRACE_SECONDS = 1.0
_SCHEDULER_CLEANUP_TIMEOUT_SECONDS = 30.0
_SCHEDULER_COMMAND_TIMEOUT_SECONDS = 10.0
_SCHEDULER_POLL_INTERVAL_SECONDS = 0.25
_SCHEDULER_FETCH_TIMEOUT_SECONDS = 30.0
_SCHEDULER_TRANSPORT_LOG_MAX_BYTES = 64 * 1024

_HQ_FORWARDED_ENV_VARS = (
    "UV_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "WARP_CACHE_DIR",
    "WARP_CACHE_PATH",
    "CUDA_CACHE_PATH",
)
_HQ_DEFAULT_CACHE_ENV_VARS = {
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "TRITON_CACHE_DIR": "triton",
    "WARP_CACHE_DIR": "warp",
    "WARP_CACHE_PATH": "warp",
    "CUDA_CACHE_PATH": "cuda",
}


def _owned_scheduler_name(job_name: str, invocation_token: str) -> str:
    """Keep the invocation token visible within scheduler name limits."""

    suffix = f"_i{invocation_token[:12]}"
    return f"{job_name[: max(0, 128 - len(suffix))]}{suffix}"


def _prepare_private_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.chmod(0o600)


def _append_private_transport_log(path: Path, output: str) -> None:
    if not output:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    remaining = _SCHEDULER_TRANSPORT_LOG_MAX_BYTES - path.stat().st_size
    if remaining <= 0:
        return
    encoded = output.encode("utf-8", errors="replace")[:remaining]
    with path.open("ab") as handle:
        handle.write(encoded)
    path.chmod(0o600)


def _remaining_scheduler_time(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _append_scheduler_timeout(
    log_path: Path, log_prefix: str, timeout_seconds: float
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{log_prefix}] evaluator timed out after {timeout_seconds:.1f}s\n"
        )


def _section_cfg(cfg: DictConfig, key: str) -> DictConfig | None:
    return _common_section_cfg(cfg, key)


def _cfg_str(section: DictConfig | None, key: str, default: str) -> str:
    return _common_cfg_str(section, key, default)


def _cfg_int(section: DictConfig | None, key: str, default: int) -> int:
    return _common_cfg_int(section, key, default)


@dataclass(frozen=True)
class EvaluatorLaunchConfig:
    launcher: str = "local"
    max_parallel: int = 1
    slurm_host: str | None = None
    slurm_sbatch_args: tuple[str, ...] = ()
    hq_submit_args: tuple[str, ...] = ()


def command_tokens(value: object | None, field_name: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        tokens = tuple(token for token in shlex.split(value) if token.strip())
    elif isinstance(value, (ListConfig, list)):
        tokens = tuple(str(item).strip() for item in list(value) if str(item).strip())
    else:
        raise ValueError(
            f"Task field '{field_name}' must be a string or list of strings."
        )
    if not tokens:
        raise ValueError(f"Task field '{field_name}' must not be empty when present.")
    return tokens


def normalize_evaluator_launcher(value: object | None, default: str = "local") -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    if normalized not in SUPPORTED_EVALUATOR_LAUNCHERS:
        supported = ", ".join(sorted(SUPPORTED_EVALUATOR_LAUNCHERS))
        raise ValueError(
            f"Unsupported evaluator launcher '{normalized}'. "
            f"Supported launchers: {supported}."
        )
    return normalized


def resolve_evaluator_launch_config(
    task_cfg: DictConfig,
    *,
    launcher: str | None,
    max_parallel: int | None,
    slurm_host: str | None = None,
    slurm_sbatch_args: list[str] | tuple[str, ...] | None = None,
    hq_submit_args: list[str] | tuple[str, ...] | None = None,
) -> EvaluatorLaunchConfig:
    evaluator_cfg = _section_cfg(task_cfg, "evaluator")
    resolved_launcher = normalize_evaluator_launcher(
        launcher
        if launcher is not None
        else _cfg_str(evaluator_cfg, "launcher", "local")
    )
    resolved_max_parallel = (
        int(max_parallel)
        if max_parallel is not None
        else _cfg_int(evaluator_cfg, "max_parallel", 1)
    )
    if resolved_max_parallel < 1:
        raise ValueError("evaluator max_parallel must be >= 1.")
    resolved_slurm_host = (
        str(slurm_host).strip()
        if slurm_host is not None
        else _cfg_str(evaluator_cfg, "slurm_host", "")
        or _cfg_str(evaluator_cfg, "host", "")
    )
    if not resolved_slurm_host:
        resolved_slurm_host = None

    if slurm_sbatch_args:
        resolved_slurm_args = tuple(str(item) for item in slurm_sbatch_args)
    else:
        resolved_slurm_args = command_tokens(
            evaluator_cfg.get("slurm_sbatch_args", None)
            if evaluator_cfg is not None
            else None,
            "evaluator.slurm_sbatch_args",
        )
    if hq_submit_args:
        resolved_hq_args = tuple(str(item) for item in hq_submit_args)
    else:
        resolved_hq_args = command_tokens(
            evaluator_cfg.get("hq_submit_args", None)
            if evaluator_cfg is not None
            else None,
            "evaluator.hq_submit_args",
        )

    return EvaluatorLaunchConfig(
        launcher=resolved_launcher,
        max_parallel=resolved_max_parallel,
        slurm_host=resolved_slurm_host,
        slurm_sbatch_args=resolved_slurm_args,
        hq_submit_args=resolved_hq_args,
    )


def evaluator_job_name(task_path: Path, cycle_index: int) -> str:
    raw_name = f"hpo_{task_path.stem}_c{cycle_index:03d}"
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("_.-")
    return normalized[:128] or f"hpo_c{cycle_index:03d}"


def build_slurm_command(
    *,
    script_path: Path,
    slurm_log_path: Path,
    job_name: str,
    max_parallel: int,
    sbatch_args: list[str] | tuple[str, ...],
    slurm_host: str | None = None,
) -> list[str]:
    validate_slurm_sbatch_args(sbatch_args)
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        "--nodes=1",
        "--ntasks=1",
        f"--gpus-per-node={int(max_parallel)}",
        f"--output={slurm_log_path}",
        f"--error={slurm_log_path}",
        *list(sbatch_args),
        str(script_path),
    ]
    if slurm_host:
        return ["ssh", str(slurm_host), shlex.join(command)]
    return command


def _sanitize_hq_cache_segment(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return normalized or "default"


def _hq_compile_cache_segment(job_name: str) -> str:
    job_segment = _sanitize_hq_cache_segment(job_name)
    stable_segment = re.sub(r"_c\d{3,}$", "", job_segment)
    return stable_segment or job_segment


def _default_hq_compile_cache_root(
    job_name: str,
    env: Mapping[str, str] | None,
    *,
    cache_parent: str | Path | None = None,
) -> str:
    source_env = env or {}
    user = _sanitize_hq_cache_segment(
        source_env.get("USER") or os.environ.get("USER") or "user"
    )
    parent = (
        str(cache_parent)
        if cache_parent is not None
        else f"/tmp/instantmimic-hq-cache-{user}/hpo"
    )
    cache_segment = _hq_compile_cache_segment(job_name)
    return f"{parent.rstrip('/')}/{cache_segment}"


def _hq_evaluator_env_args(
    *,
    job_name: str,
    env: Mapping[str, str] | None,
    cache_parent: str | Path | None = None,
) -> list[str]:
    source_env = env or {}
    forwarded = {
        name: value
        for name in _HQ_FORWARDED_ENV_VARS
        if (value := source_env.get(name))
    }
    cache_root = _default_hq_compile_cache_root(
        job_name,
        source_env,
        cache_parent=cache_parent,
    )
    for name, dirname in _HQ_DEFAULT_CACHE_ENV_VARS.items():
        forwarded.setdefault(name, f"{cache_root}/{dirname}")
    return [
        f"--env={name}={forwarded[name]}"
        for name in _HQ_FORWARDED_ENV_VARS
        if name in forwarded
    ]


def build_hq_command(
    *,
    command: list[str],
    job_name: str,
    max_parallel: int,
    submit_args: list[str] | tuple[str, ...],
    env: Mapping[str, str] | None = None,
    cache_parent: str | Path | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> list[str]:
    validate_hq_submit_args(submit_args)
    env_args = _hq_evaluator_env_args(
        job_name=job_name,
        env=env,
        cache_parent=cache_parent,
    )
    evaluator_command = command
    submit_command = [
        "hq",
        "--output-mode=quiet",
        "submit",
        f"--name={job_name}",
        "--resource",
        f"gpus/nvidia={int(max_parallel)}",
        *env_args,
        *list(submit_args),
    ]
    if stdout_path is not None:
        submit_command.extend(["--stdout", str(stdout_path)])
    if stderr_path is not None:
        submit_command.extend(["--stderr", str(stderr_path)])
    submit_command.extend(["--", *evaluator_command])
    return submit_command


def write_slurm_script(
    *,
    root: Path,
    script_path: Path,
    command: list[str],
    max_parallel: int,
) -> None:
    # TODO: this re-implements the compiler-cache environment in bash while
    # every local launcher gets it from `build_candidate_runtime_env`. The two
    # can drift silently, because only the local path is covered by CPU tests.
    # Folding this into that one builder needs a Slurm cluster to verify
    # against, so it is deliberately left alone here rather than changed blind.
    script_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(root))}",
        'export PATH="$HOME/.local/bin:$PATH"',
        f"export HPO_MAX_PARALLEL={int(max_parallel)}",
        'compile_cache_slots="2"',
        'compile_cache_parent="/tmp/instantmimic-slurm-cache/$USER/hpo"',
        'mkdir -p "$compile_cache_parent/locks"',
        'compile_cache_slot=""',
        "for ((slot=0; slot<compile_cache_slots; slot++)); do",
        '  lock_path="$compile_cache_parent/locks/slot_${slot}.lock"',
        '  exec {slot_fd}>"$lock_path"',
        '  if flock -n "$slot_fd"; then',
        '    compile_cache_slot="$slot"',
        '    compile_cache_slot_fd="$slot_fd"',
        "    break",
        "  fi",
        '  eval "exec ${slot_fd}>&-"',
        "done",
        'if [[ -z "$compile_cache_slot" ]]; then',
        '  echo "No free InstantMimic compile cache slot among ${compile_cache_slots} slots." >&2',
        "  exit 72",
        "fi",
        'compile_cache_root="$compile_cache_parent/slot_${compile_cache_slot}"',
        'mkdir -p "$compile_cache_root"',
        "release_compile_cache_slot() {",
        "  status=$?",
        '  if [[ -n "${compile_cache_slot_fd:-}" ]]; then',
        '    eval "exec ${compile_cache_slot_fd}>&-"',
        "  fi",
        '  exit "$status"',
        "}",
        "trap release_compile_cache_slot EXIT",
        'export TORCHINDUCTOR_CACHE_DIR="$compile_cache_root/torchinductor"',
        'export TRITON_CACHE_DIR="$compile_cache_root/triton"',
        'export WARP_CACHE_DIR="$compile_cache_root/warp"',
        'export WARP_CACHE_PATH="$compile_cache_root/warp"',
        'export CUDA_CACHE_PATH="$compile_cache_root/cuda"',
        shlex.join(command),
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)


def sync_file_to_slurm_host(
    path: Path,
    slurm_host: str,
    *,
    timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
) -> int:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    ssh_timeout = max(0.0, deadline - time.monotonic())
    if ssh_timeout <= 0:
        return 124
    try:
        mkdir_completed = subprocess.run(
            ["ssh", slurm_host, "mkdir", "-p", str(path.parent)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=ssh_timeout,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[hpo:evaluator] timed out creating remote directory after "
            f"{timeout_seconds:.1f}s: {slurm_host}"
        )
        return 124
    if mkdir_completed.returncode != 0:
        if mkdir_completed.stdout:
            print(mkdir_completed.stdout, end="")
        return int(mkdir_completed.returncode)
    rsync_timeout = max(0.0, deadline - time.monotonic())
    if rsync_timeout <= 0:
        return 124
    try:
        rsync_completed = subprocess.run(
            ["rsync", "-az", str(path), f"{slurm_host}:{path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=rsync_timeout,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[hpo:evaluator] timed out uploading {path} after {timeout_seconds:.1f}s"
        )
        return 124
    if rsync_completed.stdout:
        print(rsync_completed.stdout, end="")
    return int(rsync_completed.returncode)


def sync_file_from_slurm_host(
    path: Path,
    slurm_host: str,
    *,
    timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rsync_completed = subprocess.run(
            ["rsync", "-az", f"{slurm_host}:{path}", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[hpo:evaluator] timed out downloading {path} after {timeout_seconds:.1f}s"
        )
        return 124
    if rsync_completed.stdout:
        print(rsync_completed.stdout, end="")
    return int(rsync_completed.returncode)


def sync_dir_from_slurm_host(
    path: Path,
    slurm_host: str,
    *,
    timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
) -> int:
    path.mkdir(parents=True, exist_ok=True)
    source = f"{slurm_host}:{str(path).rstrip('/')}/"
    target = f"{str(path).rstrip('/')}/"
    try:
        rsync_completed = subprocess.run(
            ["rsync", "-az", source, target],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[hpo:evaluator] timed out downloading directory {path} after "
            f"{timeout_seconds:.1f}s"
        )
        return 124
    if rsync_completed.stdout:
        print(rsync_completed.stdout, end="")
    return int(rsync_completed.returncode)


def _write_process_group_manifest(path: Path, groups: set[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for group_id in sorted(groups):
                handle.write(f"{int(group_id)}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_process_group_manifest(path: Path) -> set[int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return set()
    groups: set[int] = set()
    for line in lines:
        try:
            group_id = int(line.strip())
        except ValueError:
            continue
        if group_id > 0:
            groups.add(group_id)
    return groups


def register_process_group(
    process_group_id: int | None, manifest_path: Path | None = None
) -> None:
    if process_group_id is None or process_group_id <= 0:
        return
    path = manifest_path or process_group_manifest_path()
    if path is None:
        return
    groups = _read_process_group_manifest(path)
    groups.add(int(process_group_id))
    _write_process_group_manifest(path, groups)


def unregister_process_group(
    process_group_id: int | None, manifest_path: Path | None = None
) -> None:
    if process_group_id is None or process_group_id <= 0:
        return
    path = manifest_path or process_group_manifest_path()
    if path is None or not path.exists():
        return
    groups = _read_process_group_manifest(path)
    groups.discard(int(process_group_id))
    if groups:
        _write_process_group_manifest(path, groups)
    else:
        path.unlink(missing_ok=True)


def process_group_manifest_path() -> Path | None:
    value = os.environ.get(EVALUATOR_JOB_GROUPS_ENV, "").strip()
    return Path(value) if value else None


def owned_process_group_termination_grace_seconds() -> float:
    """Keep child rollback inside the outer evaluator's cleanup budget."""

    raw_outer_grace = os.environ.get(EVALUATOR_TERMINATION_GRACE_ENV, "").strip()
    if not raw_outer_grace:
        return _EVALUATOR_TERMINATION_GRACE_SECONDS
    try:
        outer_grace = float(raw_outer_grace)
    except ValueError:
        return _EVALUATOR_TERMINATION_GRACE_SECONDS
    if not math.isfinite(outer_grace) or outer_grace <= 0.0:
        return _EVALUATOR_TERMINATION_GRACE_SECONDS
    return max(0.01, min(_OWNED_CHILD_TERMINATION_GRACE_SECONDS, outer_grace * 0.5))


def _process_group_exists(process_group_id: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _descendant_process_group_ids(process_id: int) -> set[int]:
    """Discover descendant groups before the evaluator can reparent them."""

    if os.name != "posix":
        return set()
    pending = [int(process_id)]
    descendants: set[int] = set()
    while pending:
        parent_id = pending.pop()
        children_path = Path(f"/proc/{parent_id}/task/{parent_id}/children")
        try:
            children = [int(value) for value in children_path.read_text().split()]
        except (FileNotFoundError, PermissionError, ValueError):
            continue
        for child_id in children:
            if child_id in descendants:
                continue
            descendants.add(child_id)
            pending.append(child_id)

    groups: set[int] = set()
    for descendant_id in descendants:
        try:
            groups.add(os.getpgid(descendant_id))
        except ProcessLookupError:
            continue
    return groups


def _signal_process_group(process_group_id: int, signum: signal.Signals) -> None:
    if os.name != "posix" or process_group_id <= 0:
        return
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        return


def _terminate_timed_out_evaluator(
    process: subprocess.Popen[object],
    process_group_manifest: Path,
    *,
    grace_seconds: float | None = None,
) -> None:
    """Bound evaluator shutdown and all groups owned by this invocation."""

    if grace_seconds is None:
        grace_seconds = _EVALUATOR_TERMINATION_GRACE_SECONDS
    deadline = time.monotonic() + max(0.0, float(grace_seconds))
    evaluator_group = int(process.pid) if os.name == "posix" else None
    groups = _read_process_group_manifest(process_group_manifest)
    if evaluator_group is not None:
        groups.add(evaluator_group)
        groups.update(_descendant_process_group_ids(evaluator_group))
    if evaluator_group is None:
        process.terminate()
    for group_id in groups:
        _signal_process_group(group_id, signal.SIGTERM)

    while time.monotonic() < deadline:
        discovered_groups = _read_process_group_manifest(process_group_manifest)
        if evaluator_group is not None:
            discovered_groups.update(_descendant_process_group_ids(evaluator_group))
        for group_id in discovered_groups - groups:
            groups.add(group_id)
            _signal_process_group(group_id, signal.SIGTERM)
        groups_alive = {
            group_id for group_id in groups if _process_group_exists(group_id)
        }
        if not groups_alive:
            break
        try:
            process.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            continue
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

    groups.update(_read_process_group_manifest(process_group_manifest))
    groups_alive = {group_id for group_id in groups if _process_group_exists(group_id)}
    if evaluator_group is None and process.poll() is None:
        process.kill()
    for group_id in groups_alive:
        _signal_process_group(group_id, signal.SIGKILL)
    try:
        process.wait(timeout=0.1)
    except (ChildProcessError, OSError, subprocess.TimeoutExpired):
        pass


def run_launched_evaluator(
    *,
    command: list[str],
    launch_config: EvaluatorLaunchConfig,
    root: Path,
    log_path: Path,
    job_dir: Path,
    job_name: str,
    log_prefix: str,
    task_path: Path | None = None,
    output_dir: Path | None = None,
    timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
) -> tuple[int, Path | None]:
    """Run ``command`` through the configured launcher.

    Returns the evaluator return code and, for Slurm, the generated script path.
    The caller owns building ``command`` and interpreting whatever the evaluator
    wrote; this function owns only how that command reaches a GPU.
    """

    if timeout_seconds <= 0:
        raise ValueError("evaluator subprocess timeout must be > 0.")

    if launch_config.launcher == "local":
        local_command = command
        job_dir.mkdir(parents=True, exist_ok=True)
        process_group_manifest = (
            job_dir / f"{job_name}.{os.getpid()}.{uuid.uuid4().hex}.groups"
        ).resolve()
        local_env = dict(os.environ)
        local_env[EVALUATOR_JOB_GROUPS_ENV] = str(process_group_manifest)
        local_env[EVALUATOR_TERMINATION_GRACE_ENV] = str(
            _EVALUATOR_TERMINATION_GRACE_SECONDS
        )
        _write_process_group_manifest(process_group_manifest, set())
        process: subprocess.Popen[object] | None = None
        returncode: int | None = None
        timed_out = False
        try:
            with log_path.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    local_command,
                    cwd=str(root),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=local_env,
                    start_new_session=os.name == "posix",
                )
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_timed_out_evaluator(process, process_group_manifest)
                    returncode = 124
                if not timed_out and (
                    returncode != 0
                    or bool(_read_process_group_manifest(process_group_manifest))
                    or (os.name == "posix" and _process_group_exists(int(process.pid)))
                ):
                    _terminate_timed_out_evaluator(process, process_group_manifest)
        except subprocess.TimeoutExpired:
            # Kept for custom Popen implementations that raise at construction.
            returncode = 124
        except BaseException:
            if process is not None:
                _terminate_timed_out_evaluator(process, process_group_manifest)
            raise
        finally:
            process_group_manifest.unlink(missing_ok=True)
        if returncode == 124:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\n[{log_prefix}] evaluator timed out after "
                    f"{timeout_seconds:.1f}s\n"
                )
            return 124, None
        return int(returncode), None

    if launch_config.launcher == "slurm":
        scheduler_deadline = time.monotonic() + timeout_seconds
        script_path = job_dir / f"{job_name}.sbatch.sh"
        write_slurm_script(
            root=root,
            script_path=script_path,
            command=command,
            max_parallel=launch_config.max_parallel,
        )
        if launch_config.slurm_host:
            sync_paths = [script_path]
            if task_path is not None:
                sync_paths.append(task_path)
            for sync_path in sync_paths:
                sync_timeout = _remaining_scheduler_time(scheduler_deadline)
                if sync_timeout <= 0:
                    _append_scheduler_timeout(log_path, log_prefix, timeout_seconds)
                    return 124, script_path
                sync_returncode = sync_file_to_slurm_host(
                    sync_path,
                    launch_config.slurm_host,
                    timeout_seconds=sync_timeout,
                )
                if sync_returncode != 0:
                    return sync_returncode, script_path
        if _remaining_scheduler_time(scheduler_deadline) <= 0:
            _append_scheduler_timeout(log_path, log_prefix, timeout_seconds)
            return 124, script_path
        job_dir.mkdir(parents=True, exist_ok=True)
        invocation_token = uuid.uuid4().hex
        scheduler_name = _owned_scheduler_name(job_name, invocation_token)
        ownership_path = job_dir / f"{job_name}.{invocation_token}.ownership.json"
        transport_log_path = log_path.with_name(f"{log_path.name}.transport")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _prepare_private_log(transport_log_path)
        adapter = SlurmAdapter(
            script_path=script_path,
            log_path=log_path,
            scheduler_name=scheduler_name,
            invocation_token=invocation_token,
            max_parallel=launch_config.max_parallel,
            sbatch_args=launch_config.slurm_sbatch_args,
            host=launch_config.slurm_host,
        )
        print(f"[{log_prefix}] evaluator_submit=slurm owned_invocation")

        def scheduler_runner(
            scheduler_command: list[str] | tuple[str, ...], command_timeout: float
        ):
            try:
                completed = subprocess.run(
                    list(scheduler_command),
                    cwd=str(root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=command_timeout,
                )
                result = SchedulerCommandResult(
                    int(completed.returncode), completed.stdout or ""
                )
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or exc.output or ""
                _append_private_transport_log(transport_log_path, str(output))
                raise
            _append_private_transport_log(transport_log_path, result.stdout)
            return result

        lifecycle_error: BaseException | None = None
        try:
            returncode = run_scheduler_lifecycle(
                adapter=adapter,
                ownership_path=ownership_path,
                timeout_seconds=timeout_seconds,
                cleanup_timeout_seconds=_SCHEDULER_CLEANUP_TIMEOUT_SECONDS,
                command_timeout_seconds=_SCHEDULER_COMMAND_TIMEOUT_SECONDS,
                poll_interval_seconds=_SCHEDULER_POLL_INTERVAL_SECONDS,
                runner=scheduler_runner,
                deadline=scheduler_deadline,
            )
        except BaseException as exc:
            lifecycle_error = exc
            raise
        finally:
            if launch_config.slurm_host:
                try:
                    sync_returncode = sync_file_from_slurm_host(
                        log_path,
                        launch_config.slurm_host,
                        timeout_seconds=_SCHEDULER_FETCH_TIMEOUT_SECONDS,
                    )
                    if sync_returncode != 0:
                        print(
                            f"[{log_prefix}] scheduler log sync returned "
                            f"{sync_returncode}"
                        )
                    if output_dir is not None:
                        sync_returncode = sync_dir_from_slurm_host(
                            output_dir,
                            launch_config.slurm_host,
                            timeout_seconds=_SCHEDULER_FETCH_TIMEOUT_SECONDS,
                        )
                        if sync_returncode != 0:
                            print(
                                f"[{log_prefix}] scheduler output sync returned "
                                f"{sync_returncode}"
                            )
                except OSError as sync_error:
                    print(
                        f"[{log_prefix}] scheduler artifact sync failed: {sync_error}"
                    )
            if lifecycle_error is None and returncode == 124:
                _append_scheduler_timeout(log_path, log_prefix, timeout_seconds)
        return int(returncode), script_path

    if launch_config.launcher == "hq":
        scheduler_deadline = time.monotonic() + timeout_seconds
        job_dir.mkdir(parents=True, exist_ok=True)
        invocation_token = uuid.uuid4().hex
        scheduler_name = _owned_scheduler_name(job_name, invocation_token)
        ownership_path = job_dir / f"{job_name}.{invocation_token}.ownership.json"
        transport_log_path = log_path.with_name(f"{log_path.name}.transport")
        workload_stdout_path = log_path.with_name(f"{log_path.name}.workload.stdout")
        workload_stderr_path = log_path.with_name(f"{log_path.name}.workload.stderr")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        _prepare_private_log(transport_log_path)
        hq_env_args = _hq_evaluator_env_args(
            job_name=job_name,
            env=os.environ,
        )
        adapter = HQAdapter(
            command=command,
            scheduler_name=scheduler_name,
            invocation_token=invocation_token,
            max_parallel=launch_config.max_parallel,
            submit_args=tuple(hq_env_args) + launch_config.hq_submit_args,
            stdout_path=workload_stdout_path,
            stderr_path=workload_stderr_path,
            context=os.environ.get("HQ_SERVER_DIR"),
        )
        print(f"[{log_prefix}] evaluator_submit=hq owned_invocation")

        def scheduler_runner(
            scheduler_command: list[str] | tuple[str, ...], command_timeout: float
        ):
            try:
                completed = subprocess.run(
                    list(scheduler_command),
                    cwd=str(root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=command_timeout,
                )
                result = SchedulerCommandResult(
                    int(completed.returncode), completed.stdout or ""
                )
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or exc.output or ""
                _append_private_transport_log(transport_log_path, str(output))
                raise
            _append_private_transport_log(transport_log_path, result.stdout)
            return result

        lifecycle_error: BaseException | None = None
        try:
            context_timeout = min(
                _SCHEDULER_COMMAND_TIMEOUT_SECONDS,
                _remaining_scheduler_time(scheduler_deadline),
            )
            if context_timeout <= 0:
                returncode = 124
            else:
                try:
                    adapter.resolve_context(scheduler_runner, context_timeout)
                except subprocess.TimeoutExpired:
                    if _remaining_scheduler_time(scheduler_deadline) <= 0:
                        returncode = 124
                    else:
                        raise
                else:
                    lifecycle_timeout = _remaining_scheduler_time(scheduler_deadline)
                    if lifecycle_timeout <= 0:
                        returncode = 124
                    else:
                        returncode = run_scheduler_lifecycle(
                            adapter=adapter,
                            ownership_path=ownership_path,
                            timeout_seconds=lifecycle_timeout,
                            cleanup_timeout_seconds=_SCHEDULER_CLEANUP_TIMEOUT_SECONDS,
                            command_timeout_seconds=_SCHEDULER_COMMAND_TIMEOUT_SECONDS,
                            poll_interval_seconds=_SCHEDULER_POLL_INTERVAL_SECONDS,
                            runner=scheduler_runner,
                            deadline=scheduler_deadline,
                        )
        except BaseException as exc:
            lifecycle_error = exc
            raise
        finally:
            try:
                for workload_path in (workload_stdout_path, workload_stderr_path):
                    if workload_path.exists():
                        with log_path.open("a", encoding="utf-8") as handle:
                            handle.write(f"\n[{log_prefix}] scheduler_workload\n")
                            handle.write(workload_path.read_text(encoding="utf-8"))
            except OSError as log_error:
                print(f"[{log_prefix}] scheduler logs unavailable: {log_error}")
            if lifecycle_error is None and returncode == 124:
                _append_scheduler_timeout(log_path, log_prefix, timeout_seconds)
        return int(returncode), None

    raise ValueError(f"Unsupported evaluator launcher: {launch_config.launcher}")
