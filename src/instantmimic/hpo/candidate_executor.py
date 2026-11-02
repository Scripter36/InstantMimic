"""Persistent per-candidate seed execution shared by hpo evaluators."""

from __future__ import annotations

import fcntl
import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO

from instantmimic.hpo.common import utc_now as _utc_now
from instantmimic.hpo.evaluator_launch import (
    owned_process_group_termination_grace_seconds,
    process_group_manifest_path,
    register_process_group,
    unregister_process_group,
)

_HPO_CACHE_ROOT = Path("/tmp/instantmimic-hpo-cache")
_CACHE_SCOPES = {"isolated", "task"}


def _safe_cache_key(cache_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_key).strip("_") or "job"


def candidate_cache_root(cache_key: str) -> Path:
    return _HPO_CACHE_ROOT / f"{os.getpid()}_{_safe_cache_key(cache_key)}"


def task_cache_root(cache_key: str) -> Path:
    """Return a stable compiler-cache root shared by one hpo task."""

    return _HPO_CACHE_ROOT / "tasks" / _safe_cache_key(cache_key)


def build_candidate_runtime_env(
    *,
    cache_key: str | None = None,
    cache_scope: str = "isolated",
    headless_rendering: bool = False,
) -> dict[str, str]:
    """Build the shared subprocess environment for candidate train/eval runs."""

    normalized_scope = cache_scope.strip().lower()
    if normalized_scope not in _CACHE_SCOPES:
        raise ValueError("cache_scope must be 'isolated' or 'task'.")

    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    if cache_key is not None:
        cache_root = (
            task_cache_root(cache_key)
            if normalized_scope == "task"
            else candidate_cache_root(cache_key)
        )
        if normalized_scope == "task":
            warp_cache_dir = (
                env.get("WARP_CACHE_DIR")
                or env.get("WP_CACHE_DIR")
                or env.get("WARP_CACHE_PATH")
                or str(cache_root / "warp")
            )
            env.setdefault("WARP_CACHE_DIR", warp_cache_dir)
            env.setdefault("WP_CACHE_DIR", warp_cache_dir)
            env.setdefault("WARP_CACHE_PATH", warp_cache_dir)
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "torchinductor"))
            env.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
            env.setdefault("CUDA_CACHE_PATH", str(cache_root / "cuda"))
        else:
            warp_cache_dir = str(cache_root / "warp")
            env["WARP_CACHE_DIR"] = warp_cache_dir
            env["WP_CACHE_DIR"] = warp_cache_dir
            env["WARP_CACHE_PATH"] = warp_cache_dir
            env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
            env["TRITON_CACHE_DIR"] = str(cache_root / "triton")
            env["CUDA_CACHE_PATH"] = str(cache_root / "cuda")
    else:
        warp_cache_dir = (
            env.get("WARP_CACHE_DIR")
            or env.get("WP_CACHE_DIR")
            or env.get("WARP_CACHE_PATH")
            or "/tmp/warp-cache"
        )
        env.setdefault("WARP_CACHE_DIR", warp_cache_dir)
        env.setdefault("WP_CACHE_DIR", warp_cache_dir)
        env.setdefault("WARP_CACHE_PATH", warp_cache_dir)
    if headless_rendering:
        env.setdefault("MUJOCO_GL", "egl")
    return env


def hydra_overrides_from_command(command: Sequence[str]) -> list[str]:
    """Extract Hydra overrides from a resolved ``uv run train`` command."""

    return [
        str(token)
        for token in command
        if ("=" in token or token.startswith("~")) and not token.startswith("--")
    ]


def process_group_exists(process_group_id: int) -> bool:
    if os.name != "posix" or process_group_id <= 0:
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(
    proc: subprocess.Popen[object],
    grace_seconds: float | None = None,
    *,
    process_group_id: int | None = None,
) -> None:
    """Stop one owned process group, escalating SIGTERM to SIGKILL.

    ``process_group_id`` defaults to the child's pid, which is its group when
    the child was started with ``start_new_session``. Callers that recorded the
    id at launch pass it explicitly so termination still works once the process
    object has been reaped.
    """

    if grace_seconds is None:
        grace_seconds = owned_process_group_termination_grace_seconds()
    group_id = proc.pid if process_group_id is None else int(process_group_id)
    if os.name == "posix" and group_id > 0:
        group_exists = False
        try:
            os.killpg(group_id, signal.SIGTERM)
            group_exists = True
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_seconds
        while group_exists and time.monotonic() < deadline:
            try:
                proc.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
            group_exists = process_group_exists(group_id)
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        if group_exists:
            try:
                os.killpg(group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        if proc.poll() is None:
            proc.terminate()
    try:
        proc.wait(timeout=min(grace_seconds, 0.1))
    except subprocess.TimeoutExpired:
        if os.name != "posix":
            proc.kill()
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            return
    except (ChildProcessError, OSError):
        return


def _append_log_message(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


class SeedTrainWorker:
    """Keep one Python/compiler process alive for all seeds of one candidate."""

    def __init__(
        self,
        *,
        config_path: str | None,
        config_name: str,
        cache_key: str = "candidate-seed-worker",
        cache_scope: str = "isolated",
    ) -> None:
        self.config_path = config_path
        self.config_name = config_name
        self.cache_key = cache_key
        self.cache_scope = cache_scope
        self._proc: subprocess.Popen[str] | None = None
        self._process_group_id: int | None = None
        self._cache_locks: list[IO[str]] = []

    def __enter__(self) -> SeedTrainWorker:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        proc = self._proc
        process_group_id = self._process_group_id
        self._proc = None
        self._process_group_id = None
        try:
            if proc is not None:
                terminate_process_group(proc)
        finally:
            try:
                unregister_process_group(
                    process_group_id, process_group_manifest_path()
                )
            except (OSError, ValueError):
                pass
        for lock_handle in reversed(self._cache_locks):
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        self._cache_locks.clear()

    def _acquire_task_cache_locks(self, runtime_env: Mapping[str, str]) -> None:
        if self.cache_scope.strip().lower() != "task" or self._cache_locks:
            return
        cache_dirs = {
            Path(runtime_env[name]).resolve()
            for name in (
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
                "WARP_CACHE_DIR",
                "CUDA_CACHE_PATH",
            )
            if runtime_env.get(name)
        }
        acquired: list[IO[str]] = []
        for cache_dir in sorted(cache_dirs):
            cache_dir.mkdir(parents=True, exist_ok=True)
            lock_handle = (cache_dir / ".instantmimic-cache.lock").open(
                "a+", encoding="utf-8"
            )
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                lock_handle.close()
                for acquired_handle in reversed(acquired):
                    fcntl.flock(acquired_handle.fileno(), fcntl.LOCK_UN)
                    acquired_handle.close()
                raise RuntimeError(
                    f"task-scoped compiler cache is already in use: {cache_dir}"
                ) from exc
            acquired.append(lock_handle)
        self._cache_locks = acquired

    def _start(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        if self._proc is not None:
            self.close()
        worker_script = Path(__file__).with_name("seed_train_worker.py")
        runtime_env = build_candidate_runtime_env(
            cache_key=self.cache_key,
            cache_scope=self.cache_scope,
        )
        process: subprocess.Popen[str] | None = None
        try:
            self._acquire_task_cache_locks(runtime_env)
            process = subprocess.Popen(
                [sys.executable, str(worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                start_new_session=True,
                env=runtime_env,
            )
            self._proc = process
            self._process_group_id = process.pid if os.name == "posix" else None
            register_process_group(
                self._process_group_id, process_group_manifest_path()
            )
        except BaseException:
            if process is not None:
                self._proc = process
                self._process_group_id = process.pid if os.name == "posix" else None
            self.close()
            raise
        assert self._proc is not None
        return self._proc

    def _read_response(
        self,
        proc: subprocess.Popen[str],
        *,
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        if proc.stdout is None:
            raise RuntimeError("seed train worker stdout pipe is unavailable")
        if timeout_seconds is None:
            line = proc.stdout.readline()
        else:
            ready, _, _ = select.select([proc.stdout], [], [], timeout_seconds)
            if not ready:
                raise subprocess.TimeoutExpired(
                    cmd="seed train worker", timeout=timeout_seconds
                )
            line = proc.stdout.readline()
        if line == "":
            returncode = proc.poll()
            raise RuntimeError(
                "seed train worker exited before returning a response"
                + (f" (returncode={returncode})" if returncode is not None else "")
            )
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError("seed train worker response must be a JSON object")
        return payload

    def run_trial(
        self,
        *,
        command: list[str],
        log_file: Path,
        dry_run: bool,
        frames_per_run: int | None,
        trial_timeout_seconds: float | None,
        overrides: Mapping[str, str] | Sequence[str],
    ) -> tuple[float | None, float | None, int, str, str]:
        del command
        if dry_run:
            started_at = _utc_now()
            return 0.0, None, 0, started_at, _utc_now()

        started_at = _utc_now()
        start = time.perf_counter()
        serialized_overrides = (
            [f"{key}={value}" for key, value in overrides.items()]
            if isinstance(overrides, Mapping)
            else [str(item) for item in overrides]
        )
        request = {
            "config_path": self.config_path,
            "config_name": self.config_name,
            "overrides": serialized_overrides,
            "log_file": str(log_file),
            "frames_per_run": frames_per_run,
        }
        proc = self._start()
        try:
            if proc.stdin is None:
                raise RuntimeError("seed train worker stdin pipe is unavailable")
            proc.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
            proc.stdin.flush()
            response = self._read_response(proc, timeout_seconds=trial_timeout_seconds)
        except KeyboardInterrupt:
            self.close()
            raise
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - start
            self.close()
            _append_log_message(
                log_file,
                f"\n[candidate-executor] Trial timed out after "
                f"{trial_timeout_seconds:.3f}s",
            )
            throughput = (
                float(frames_per_run) / wall
                if frames_per_run is not None and wall > 0.0
                else None
            )
            return wall, throughput, 124, started_at, _utc_now()
        except (BrokenPipeError, json.JSONDecodeError, OSError, RuntimeError) as exc:
            wall = time.perf_counter() - start
            self.close()
            _append_log_message(
                log_file, f"\n[candidate-executor] Worker failed: {exc}"
            )
            throughput = (
                float(frames_per_run) / wall
                if frames_per_run is not None and wall > 0.0
                else None
            )
            return wall, throughput, 1, started_at, _utc_now()

        wall = response.get("wall_seconds")
        throughput = response.get("throughput_fps")
        returncode = int(response.get("returncode", 1))
        if returncode != 0:
            self.close()
        return (
            float(wall) if isinstance(wall, int | float) else None,
            float(throughput) if isinstance(throughput, int | float) else None,
            returncode,
            str(response.get("started_at") or started_at),
            str(response.get("finished_at") or _utc_now()),
        )
