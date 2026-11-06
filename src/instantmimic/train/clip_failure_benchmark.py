from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from instantmimic.utils.project_paths import repo_root

_TRACKING_METRIC_FIELDS = {
    "ape_mean": "APE mean (m)",
    "global_mpjpe_mean": "global MPJPE mean (m)",
    "ee_mean": "EE mean (m)",
    "vel_mean": "velocity mean (m/s)",
    "accel_mean": "acceleration mean (m/s^2)",
    "phc_mpjpe_mean": "PHC MPJPE mean (mm)",
    "phc_global_mpjpe_mean": "PHC global MPJPE mean (mm)",
    "phc_vel_mean": "PHC velocity mean (mm/frame)",
    "phc_accel_mean": "PHC acceleration mean (mm/frame^2)",
}

_PHC_METRIC_STEP_FIELDS = {
    "phc_mpjpe_mean": "phc_mpjpe_metric_steps",
    "phc_global_mpjpe_mean": "phc_global_mpjpe_metric_steps",
    "phc_vel_mean": "phc_vel_metric_steps",
    "phc_accel_mean": "phc_accel_metric_steps",
}

DEFAULT_CLIP_FAILURE_WORKER_TIMEOUT_SECONDS = 3600.0
_PROCESS_GROUP_TERM_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class ClipFailureBenchmarkConfig:
    """Resolved clip-failure benchmark settings.

    ``use_cuda_graph`` and ``compile`` are inherited from the run's
    ``env.use_cuda_graph`` and ``runtime.compile``; the benchmark block cannot
    author them.
    """

    enabled: bool
    interval: int
    anchor_stride: int
    window_steps: int
    batch_size: int
    use_cuda_graph: bool
    compile: bool
    clip_indices: tuple[int, ...] | None
    history_filename: str
    save_history: bool = True
    subprocess: str = "auto"
    preserve_training_graphs: bool = False
    shared_motion_clone_tensors: bool | None = None
    mode: str = "anchor"
    timeout_seconds: float = DEFAULT_CLIP_FAILURE_WORKER_TIMEOUT_SECONDS

    @classmethod
    def disabled(cls) -> ClipFailureBenchmarkConfig:
        return cls(
            enabled=False,
            interval=0,
            anchor_stride=1,
            window_steps=1,
            batch_size=1,
            use_cuda_graph=False,
            compile=False,
            clip_indices=None,
            history_filename="clip_failure_history.jsonl",
        )


def _load_analysis_artifacts(
    *,
    output_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    results_path = output_root / "results.csv"
    summary_path = output_root / "summary.csv"
    metadata_path = output_root / "metadata.json"
    if (
        not results_path.exists()
        or not summary_path.exists()
        or not metadata_path.exists()
    ):
        raise RuntimeError(
            f"Clip failure benchmark artifacts missing under {output_root}."
        )
    with results_path.open(newline="", encoding="utf-8") as handle:
        anchor_rows = list(csv.DictReader(handle))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return anchor_rows, summary_rows, metadata


def prewarm_analysis_runtime_graphs(runtime: Any) -> None:
    from instantmimic.train.clip_failure_analysis import (
        prewarm_analysis_runtime_graphs,
    )

    prewarm_analysis_runtime_graphs(runtime)


def _file_tail(path: Path, *, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


class PersistentClipFailureBenchmarkWorker:
    def __init__(
        self,
        *,
        config_name: str,
        output_root: Path,
        batch_size: int,
        use_cuda_graph: bool,
        compile: bool,
        timeout_seconds: float = DEFAULT_CLIP_FAILURE_WORKER_TIMEOUT_SECONDS,
    ) -> None:
        self._output_root = Path(output_root).resolve()
        self._batch_size = int(batch_size)
        self._request_index = 0
        self._closed = False
        self._timeout_seconds = _normalize_timeout_seconds(
            timeout_seconds, owner="clip failure benchmark worker"
        )
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._response_dir = self._output_root / "_worker_responses"
        self._response_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._output_root / "_persistent_worker.log"
        self._log_handle = self._log_path.open("a", encoding="utf-8")
        framework_root = repo_root()
        command = [
            sys.executable,
            str(
                framework_root
                / "scripts"
                / "analysis"
                / "run_clip_failure_benchmark_worker.py"
            ),
            "--persistent",
            "--config-name",
            str(config_name),
            "--batch-size",
            str(self._batch_size),
        ]
        if use_cuda_graph:
            command.append("--use-cuda-graph")
        if compile:
            command.append("--compile")
        process_kwargs = {
            "cwd": str(framework_root),
            "stdin": subprocess.PIPE,
            "stdout": self._log_handle,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if os.name == "posix":
            process_kwargs["start_new_session"] = True
        self._process = subprocess.Popen(command, **process_kwargs)

    @property
    def log_path(self) -> Path:
        return self._log_path

    def _next_response_path(self) -> Path:
        self._request_index += 1
        return self._response_dir / f"request_{self._request_index:06d}.json"

    def _write_request_payload(
        self,
        stdin: Any,
        payload: bytes,
        deadline: float,
    ) -> None:
        try:
            fd = stdin.fileno()
            was_blocking = os.get_blocking(fd)
            os.set_blocking(fd, False)
        except (AttributeError, OSError, ValueError):
            self._write_request_with_thread(stdin, payload, deadline)
            return

        try:
            offset = 0
            while offset < len(payload):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "Clip failure benchmark worker request transmission "
                        f"timed out after {self._timeout_seconds:.1f}s"
                    )
                try:
                    written = os.write(fd, payload[offset:])
                except BlockingIOError:
                    time.sleep(min(0.01, remaining))
                except BrokenPipeError as exc:
                    raise RuntimeError(
                        "Clip failure benchmark worker stdin closed during "
                        "request transmission."
                    ) from exc
                else:
                    if written <= 0:
                        raise RuntimeError(
                            "Clip failure benchmark worker made no progress "
                            "during request transmission."
                        )
                    offset += written
        finally:
            try:
                os.set_blocking(fd, was_blocking)
            except OSError:
                pass

    def _write_request_with_thread(
        self,
        stdin: Any,
        payload: bytes,
        deadline: float,
    ) -> None:
        outcome: list[Exception] = []

        def write_payload() -> None:
            try:
                stdin.write(payload.decode("utf-8"))
                stdin.flush()
            except Exception as exc:  # noqa: BLE001 - capture thread write failures
                outcome.append(exc)

        writer = threading.Thread(target=write_payload, daemon=True)
        writer.start()
        writer.join(timeout=max(0.0, deadline - time.monotonic()))
        if writer.is_alive():
            raise TimeoutError(
                "Clip failure benchmark worker request transmission "
                f"timed out after {self._timeout_seconds:.1f}s"
            )
        if outcome:
            raise outcome[0]

    def _send_request(
        self,
        payload: dict[str, object],
        response_path: Path,
        deadline: float,
    ) -> None:
        if self._closed:
            raise RuntimeError("Clip failure benchmark worker is already closed.")
        if self._process.poll() is not None:
            raise RuntimeError(
                "Clip failure benchmark worker exited unexpectedly "
                f"(exit_code={self._process.returncode}). "
                f"Log tail:\n{_file_tail(self._log_path)}"
            )
        stdin = self._process.stdin
        if stdin is None:
            raise RuntimeError("Clip failure benchmark worker stdin is closed.")
        response_path.unlink(missing_ok=True)
        payload = dict(payload)
        payload["response_path"] = str(response_path)
        self._write_request_payload(
            stdin,
            (json.dumps(payload) + "\n").encode("utf-8"),
            deadline,
        )

    def _wait_for_response(
        self,
        response_path: Path,
        deadline: float,
    ) -> dict[str, object]:
        while True:
            if response_path.exists():
                payload = json.loads(response_path.read_text(encoding="utf-8"))
                if not bool(payload.get("ok", False)):
                    raise RuntimeError(
                        "Clip failure benchmark worker request failed: "
                        f"{payload.get('error', '')}\n"
                        f"{payload.get('traceback', '')}\n"
                        f"Worker log tail:\n{_file_tail(self._log_path)}"
                    )
                return payload
            if self._process.poll() is not None:
                raise RuntimeError(
                    "Clip failure benchmark worker exited before responding "
                    f"(exit_code={self._process.returncode}). "
                    f"Log tail:\n{_file_tail(self._log_path)}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    "Clip failure benchmark worker response timed out after "
                    f"{self._timeout_seconds:.1f}s"
                )
            time.sleep(min(0.05, remaining))

    def run(
        self,
        *,
        checkpoint_path: Path,
        output_root: Path,
        anchor_stride: int,
        window_steps: int,
        mode: str,
        clip_indices: tuple[int, ...] | None,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
        response_path = self._next_response_path()
        deadline = time.monotonic() + self._timeout_seconds
        try:
            self._send_request(
                {
                    "command": "run",
                    "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
                    "mode": str(mode),
                    "anchor_stride": int(anchor_stride),
                    "window_steps": int(window_steps),
                    "output_root": str(Path(output_root).expanduser().resolve()),
                    "clip_indices": (
                        [int(item) for item in clip_indices]
                        if clip_indices is not None
                        else None
                    ),
                },
                response_path,
                deadline,
            )
            self._wait_for_response(response_path, deadline)
        except TimeoutError as exc:
            self._terminate_process_group()
            raise RuntimeError(
                "clip failure benchmark worker timed out after "
                f"{self._timeout_seconds:.1f}s"
            ) from exc
        except BaseException:  # terminate the worker on interruption too.
            self._terminate_process_group()
            raise
        return _load_analysis_artifacts(output_root=Path(output_root))

    def _process_group_exists(self) -> bool:
        self._process.poll()
        try:
            os.killpg(self._process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _terminate_process_group(self) -> None:
        if os.name == "posix":
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except ProcessLookupError:
                self._process.wait(timeout=5)
                return

            grace_deadline = time.monotonic() + _PROCESS_GROUP_TERM_GRACE_SECONDS
            group_alive = self._process_group_exists()
            while group_alive:
                remaining = grace_deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                time.sleep(min(0.05, remaining))
                group_alive = self._process_group_exists()
            if group_alive:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._process.wait(timeout=5)
            return

        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                deadline = time.monotonic() + self._timeout_seconds
                response_path = self._next_response_path()
                self._send_request({"command": "close"}, response_path, deadline)
                self._wait_for_response(response_path, deadline)
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("Clip failure benchmark worker close timed out.")
                self._process.wait(timeout=remaining)
        except Exception:  # noqa: BLE001 - subprocess cleanup must terminate on any protocol failure.
            self._terminate_process_group()
        except BaseException:  # do not orphan the worker on interruption.
            self._terminate_process_group()
            raise
        else:
            self._terminate_process_group()
        finally:
            self._closed = True
            self._log_handle.close()


class ClipFailureBenchmarkRunner:
    def __init__(
        self,
        *,
        config_name: str,
        log_dir: Path,
        benchmark_cfg: ClipFailureBenchmarkConfig,
        source_env_raw: Any | None = None,
    ) -> None:
        self._config_name = str(config_name)
        self._log_dir = Path(log_dir).resolve()
        self._benchmark_cfg = benchmark_cfg
        self._source_env_raw = source_env_raw
        self._output_root = (
            self._log_dir / "eval" / "clip_failure_benchmark"
        ).resolve()
        self._history_path = self._output_root / benchmark_cfg.history_filename
        self._runtime = None
        self._runtime_prewarmed = False
        self._worker: PersistentClipFailureBenchmarkWorker | None = None
        self._last_anchor_rows: tuple[dict[str, object], ...] = ()

    @property
    def output_root(self) -> Path:
        return self._output_root

    @property
    def history_path(self) -> Path:
        return self._history_path

    @property
    def last_anchor_rows(self) -> tuple[dict[str, object], ...]:
        return self._last_anchor_rows

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
            self._runtime_prewarmed = False
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def _ensure_runtime(self, checkpoint_path: str | Path):
        from instantmimic.rl.runtime import (
            load_env_checkpoint_state,
        )
        from instantmimic.train.clip_failure_analysis import (
            load_analysis_runtime,
        )

        checkpoint_path_resolved = Path(checkpoint_path).expanduser().resolve()
        if self._runtime is None:
            self._runtime = load_analysis_runtime(
                checkpoint_path=checkpoint_path_resolved,
                batch_size=int(self._benchmark_cfg.batch_size),
                seed=None,
                use_cuda_graph=bool(self._benchmark_cfg.use_cuda_graph),
                compile=bool(self._benchmark_cfg.compile),
                shared_motion_source_env=(
                    self._source_env_raw
                    if self._benchmark_cfg.subprocess == "shared_motion"
                    else None
                ),
                shared_motion_clone_tensors=(
                    should_clone_shared_motion_tensors_for_clip_failure_benchmark(
                        self._benchmark_cfg
                    )
                ),
            )
            return self._runtime

        self._runtime.agent.load(checkpoint_path_resolved)
        load_env_checkpoint_state(self._runtime.env_raw, checkpoint_path_resolved)
        self._runtime.policy_runner.refresh_preallocated_policy_state()
        self._runtime.checkpoint_path = checkpoint_path_resolved
        self._runtime.run_log_dir = checkpoint_path_resolved.parent.parent.resolve()
        return self._runtime

    def prewarm_graphs(self, checkpoint_path: str | Path) -> None:
        if self._runtime_prewarmed:
            return
        if not (
            self._benchmark_cfg.use_cuda_graph
            and self._benchmark_cfg.subprocess == "shared_motion"
            and self._benchmark_cfg.preserve_training_graphs
        ):
            return

        runtime = self._ensure_runtime(checkpoint_path)
        prewarm_analysis_runtime_graphs(runtime)
        self._runtime_prewarmed = True

    def _load_analysis_artifacts(
        self,
        *,
        output_root: Path,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
        return _load_analysis_artifacts(output_root=output_root)

    def _ensure_worker(self) -> PersistentClipFailureBenchmarkWorker:
        if self._worker is None:
            self._worker = PersistentClipFailureBenchmarkWorker(
                config_name=self._config_name,
                output_root=self._output_root,
                batch_size=int(self._benchmark_cfg.batch_size),
                use_cuda_graph=bool(self._benchmark_cfg.use_cuda_graph),
                compile=bool(self._benchmark_cfg.compile),
                timeout_seconds=float(self._benchmark_cfg.timeout_seconds),
            )
        return self._worker

    def _run_subprocess_analysis(
        self,
        *,
        checkpoint_path: Path,
        output_root: Path,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
        output_root.mkdir(parents=True, exist_ok=True)
        return self._ensure_worker().run(
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            anchor_stride=int(self._benchmark_cfg.anchor_stride),
            window_steps=int(self._benchmark_cfg.window_steps),
            mode=str(self._benchmark_cfg.mode),
            clip_indices=self._benchmark_cfg.clip_indices,
        )

    def run(
        self,
        *,
        checkpoint_path: str | Path,
        iteration_index: int,
        step_index: int,
        frames: int,
        active_train_time_s: float | None = None,
        checkpoint_time_s: float | None = None,
    ) -> dict[str, object]:
        from instantmimic.train.clip_failure_analysis import (
            _clip_name_from_motion_path,
            build_anchor_table_for_clip_indices,
            build_full_motion_table_for_clip_indices,
            prepare_scan_anchors,
            run_anchor_scan,
            run_full_motion_scan,
            summarize_anchor_rows,
            write_analysis_artifacts,
        )

        checkpoint_path_resolved = Path(checkpoint_path).expanduser().resolve()
        benchmark_output_root = self._output_root / f"iter_{iteration_index:06d}"

        if should_run_clip_failure_benchmark_in_subprocess(self._benchmark_cfg):
            start_time = time.perf_counter()
            anchor_rows, summary_rows, metadata = self._run_subprocess_analysis(
                checkpoint_path=checkpoint_path_resolved,
                output_root=benchmark_output_root,
            )
            metadata["runtime_seconds"] = time.perf_counter() - start_time
            self._last_anchor_rows = (
                tuple(anchor_rows) if self._benchmark_cfg.mode == "anchor" else ()
            )
            alignment_metrics = (
                compute_adaptive_alignment_metrics(
                    anchor_rows=anchor_rows,
                    checkpoint_path=checkpoint_path_resolved,
                )
                if self._benchmark_cfg.mode == "anchor"
                else {}
            )
            record = build_clip_failure_benchmark_record(
                iteration_index=iteration_index,
                step_index=step_index,
                frames=frames,
                active_train_time_s=active_train_time_s,
                checkpoint_time_s=checkpoint_time_s,
                checkpoint_path=checkpoint_path_resolved,
                output_root=benchmark_output_root,
                result={
                    "anchor_rows": anchor_rows,
                    "summary_rows": summary_rows,
                    "metadata": metadata,
                },
                alignment_metrics=alignment_metrics,
            )
            if self._benchmark_cfg.save_history:
                append_clip_failure_benchmark_record(self._history_path, record)
                history = load_clip_failure_benchmark_history(self._history_path)
                write_clip_failure_benchmark_history(self._output_root, history)
            return record

        runtime = self._ensure_runtime(checkpoint_path_resolved)

        if not bool(getattr(runtime.env_raw, "_multi_motion_enabled", False)):
            raise RuntimeError(
                "LAFAN clip failure benchmark currently requires a config that "
                "resolves multiple motion.source entries."
            )

        clip_lengths = [
            int(item)
            for item in runtime.env_raw.motion_clip_length.detach().cpu().tolist()
        ]
        clip_names = [
            _clip_name_from_motion_path(path)
            for path in runtime.env_raw._resolved_motion_sources
        ]
        selected_clip_indices = (
            list(self._benchmark_cfg.clip_indices)
            if self._benchmark_cfg.clip_indices is not None
            else list(range(len(clip_lengths)))
        )
        start_time = time.perf_counter()
        if self._benchmark_cfg.mode == "full_motion":
            motions = build_full_motion_table_for_clip_indices(
                clip_lengths=clip_lengths,
                clip_indices=selected_clip_indices,
            )
            anchor_rows = run_full_motion_scan(
                runtime,
                motions=motions,
                clip_names=clip_names,
                clip_lengths=clip_lengths,
            )
        else:
            anchors = build_anchor_table_for_clip_indices(
                clip_lengths=clip_lengths,
                clip_indices=selected_clip_indices,
                anchor_stride=int(self._benchmark_cfg.anchor_stride),
                window_steps=int(self._benchmark_cfg.window_steps),
            )
            scan_anchors = prepare_scan_anchors(anchors, lead_in_steps=0)
            anchor_rows = run_anchor_scan(
                runtime,
                anchors=scan_anchors,
                clip_names=clip_names,
                clip_lengths=clip_lengths,
                window_steps=int(self._benchmark_cfg.window_steps),
            )
        summary_rows = summarize_anchor_rows(anchor_rows)
        metadata = {
            "config_name": self._config_name,
            "checkpoint_path": checkpoint_path_resolved.as_posix(),
            "mode": str(self._benchmark_cfg.mode),
            "anchor_stride": int(self._benchmark_cfg.anchor_stride),
            "window_steps": int(self._benchmark_cfg.window_steps),
            "lead_in_steps": 0,
            "batch_size": int(self._benchmark_cfg.batch_size),
            "clip_count": len(clip_names),
            "anchor_count": len(anchor_rows),
            "motion_count": len(anchor_rows)
            if self._benchmark_cfg.mode == "full_motion"
            else 0,
            "selected_clip_indices": selected_clip_indices,
            "runtime_seconds": time.perf_counter() - start_time,
        }
        write_analysis_artifacts(
            output_root=benchmark_output_root,
            anchor_rows=anchor_rows,
            summary_rows=summary_rows,
            metadata=metadata,
        )
        self._last_anchor_rows = (
            tuple(anchor_rows) if self._benchmark_cfg.mode == "anchor" else ()
        )
        alignment_metrics = (
            compute_adaptive_alignment_metrics(
                anchor_rows=anchor_rows,
                checkpoint_path=checkpoint_path_resolved,
            )
            if self._benchmark_cfg.mode == "anchor"
            else {}
        )
        record = build_clip_failure_benchmark_record(
            iteration_index=iteration_index,
            step_index=step_index,
            frames=frames,
            active_train_time_s=active_train_time_s,
            checkpoint_time_s=checkpoint_time_s,
            checkpoint_path=checkpoint_path_resolved,
            output_root=benchmark_output_root,
            result={
                "anchor_rows": anchor_rows,
                "summary_rows": summary_rows,
                "metadata": metadata,
            },
            alignment_metrics=alignment_metrics,
        )
        if self._benchmark_cfg.save_history:
            append_clip_failure_benchmark_record(self._history_path, record)
            history = load_clip_failure_benchmark_history(self._history_path)
            write_clip_failure_benchmark_history(self._output_root, history)
        return record


def build_benchmark_prior_distribution(
    *,
    anchor_rows: Sequence[dict[str, object]],
    adaptive_unit_count: int,
    adaptive_chunk_size: int,
    clip_unit_start: Sequence[int] | torch.Tensor | None = None,
) -> torch.Tensor | None:
    if adaptive_unit_count <= 0 or adaptive_chunk_size <= 0 or not anchor_rows:
        return None

    if clip_unit_start is not None:
        clip_unit_start_list = [
            int(item)
            for item in torch.as_tensor(clip_unit_start)
            .detach()
            .cpu()
            .reshape(-1)
            .tolist()
        ]
    else:
        clip_unit_start_list = None

    score_sum = torch.zeros(adaptive_unit_count, dtype=torch.float32)
    score_count = torch.zeros(adaptive_unit_count, dtype=torch.float32)

    for row in anchor_rows:
        start_frame_idx = int(row["start_frame_idx"])
        unit_idx = start_frame_idx // adaptive_chunk_size
        if clip_unit_start_list is not None:
            clip_idx = int(row["clip_idx"])
            if clip_idx < 0 or clip_idx >= len(clip_unit_start_list):
                continue
            unit_idx += clip_unit_start_list[clip_idx]
        if unit_idx < 0 or unit_idx >= adaptive_unit_count:
            continue
        score = max(0.0, 1.0 - float(row["survival_ratio"]))
        score_sum[unit_idx] += score
        score_count[unit_idx] += 1.0

    valid_mask = score_count > 0.0
    if not bool(valid_mask.any()):
        return None

    mean_score = torch.zeros_like(score_sum)
    mean_score[valid_mask] = score_sum[valid_mask] / score_count[valid_mask]
    total = float(mean_score.sum().item())
    if total <= 0.0:
        return None
    mean_score.div_(total)
    return mean_score


def apply_clip_failure_benchmark_prior_to_env(
    *,
    anchor_rows: Sequence[dict[str, object]],
    env_raw: Any,
) -> dict[str, float] | None:
    adaptive_state = getattr(env_raw, "adaptive_sampling_state", None)
    if adaptive_state is None or not adaptive_state.sampling_enabled:
        return None

    benchmark_beta = float(adaptive_state.benchmark_beta)
    if benchmark_beta <= 0.0:
        return None

    set_prior = getattr(env_raw, "set_adaptive_sampling_benchmark_prior", None)
    clear_prior = getattr(env_raw, "clear_adaptive_sampling_benchmark_prior", None)
    if not callable(set_prior):
        return None

    clip_unit_start = None
    if bool(getattr(env_raw, "_multi_motion_enabled", False)):
        clip_unit_start = getattr(env_raw, "motion_clip_unit_start", None)

    prior = build_benchmark_prior_distribution(
        anchor_rows=anchor_rows,
        adaptive_unit_count=int(adaptive_state.unit_count),
        adaptive_chunk_size=int(adaptive_state.chunk_size),
        clip_unit_start=clip_unit_start,
    )
    if prior is None:
        if callable(clear_prior):
            clear_prior()
        return {
            "benchmark_beta": benchmark_beta,
            "benchmark_prior_nonzero_units": 0.0,
            "benchmark_prior_peak": 0.0,
        }

    set_prior(prior.to(device=getattr(env_raw, "device", None) or "cpu"))
    return {
        "benchmark_beta": benchmark_beta,
        "benchmark_prior_nonzero_units": float((prior > 0.0).sum().item()),
        "benchmark_prior_peak": float(prior.max().item()),
    }


def resolve_semantic_clip_failure_benchmark_config(
    benchmark: Any,
    *,
    use_cuda_graph: bool,
    compile: bool,
) -> ClipFailureBenchmarkConfig:
    """Lower the benchmark block and inherit graph/compile from the run.

    The benchmark never authors its own execution strategy: ``cuda_graph`` maps
    to ``env.use_cuda_graph`` and ``compile`` to ``runtime.compile`` for the
    checkpoint it evaluates.
    """

    if not hasattr(benchmark, "get"):
        raise TypeError("trainer.clip_failure_benchmark must be a mapping")

    execution = str(benchmark.get("execution", "auto"))
    subprocess_mode = {
        "auto": "auto",
        "inline": "never",
        "subprocess": "always",
        "shared_motion": "shared_motion",
    }.get(execution)
    if subprocess_mode is None:
        raise ValueError(
            "trainer.clip_failure_benchmark.execution must be auto, inline, "
            "subprocess, or shared_motion"
        )
    raw_clip_indices = benchmark.get("clip_indices")
    clip_indices = (
        None
        if raw_clip_indices is None
        else tuple(int(index) for index in raw_clip_indices)
    )
    return ClipFailureBenchmarkConfig(
        enabled=True,
        interval=int(benchmark["interval"]),
        mode=str(benchmark["mode"]),
        anchor_stride=int(benchmark.get("anchor_stride", 32)),
        window_steps=int(benchmark["window_steps"]),
        batch_size=int(benchmark["batch_size"]),
        use_cuda_graph=bool(use_cuda_graph),
        compile=bool(compile),
        clip_indices=clip_indices,
        subprocess=subprocess_mode,
        preserve_training_graphs=bool(benchmark.get("preserve_training_graphs", False)),
        shared_motion_clone_tensors=(
            bool(benchmark["clone_shared_motion_tensors"])
            if "clone_shared_motion_tensors" in benchmark
            else None
        ),
        save_history=bool(benchmark.get("save_history", True)),
        timeout_seconds=_normalize_timeout_seconds(
            benchmark.get(
                "timeout_seconds", DEFAULT_CLIP_FAILURE_WORKER_TIMEOUT_SECONDS
            ),
            owner="clip failure benchmark worker",
        ),
        history_filename=str(
            benchmark.get("history_filename", "clip_failure_history.jsonl")
        ),
    )


def _normalize_timeout_seconds(value: object, *, owner: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner} timeout must be finite and positive") from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(f"{owner} timeout must be finite and positive")
    return timeout


def should_run_clip_failure_benchmark(
    *, iteration_index: int, config: ClipFailureBenchmarkConfig
) -> bool:
    return bool(
        config.enabled
        and config.interval > 0
        and iteration_index > 0
        and iteration_index % config.interval == 0
    )


def should_run_clip_failure_benchmark_in_subprocess(
    config: ClipFailureBenchmarkConfig,
) -> bool:
    if config.subprocess == "shared_motion":
        return False
    if config.subprocess == "never" and config.use_cuda_graph:
        raise ValueError(
            "inline clip failure benchmark execution conflicts with CUDA graph"
        )
    if config.subprocess == "always":
        return True
    if config.subprocess == "never":
        return False
    return bool(config.use_cuda_graph)


def should_rebuild_rollout_cuda_graph_after_clip_failure_benchmark(
    config: ClipFailureBenchmarkConfig,
) -> bool:
    return bool(
        config.use_cuda_graph
        and config.subprocess == "shared_motion"
        and not config.preserve_training_graphs
    )


def should_preserve_training_cuda_graphs_after_clip_failure_benchmark(
    config: ClipFailureBenchmarkConfig,
) -> bool:
    return bool(
        config.use_cuda_graph
        and config.subprocess == "shared_motion"
        and config.preserve_training_graphs
    )


def should_clone_shared_motion_tensors_for_clip_failure_benchmark(
    config: ClipFailureBenchmarkConfig,
) -> bool:
    if config.subprocess != "shared_motion":
        return False
    if config.shared_motion_clone_tensors is not None:
        return bool(config.shared_motion_clone_tensors)
    return bool(config.preserve_training_graphs)


def _clip_metric_stem(clip_name: str) -> str:
    return Path(str(clip_name)).stem


def _overall_clip_metrics(
    summary_rows: Sequence[dict[str, object]],
    *,
    anchor_rows: Sequence[dict[str, object]] = (),
) -> tuple[float, float, float, float, float]:
    total_anchor_count = sum(int(row["anchor_count"]) for row in summary_rows)
    if total_anchor_count <= 0:
        return math.nan, math.nan, math.nan, math.nan, math.nan
    weighted_failure = sum(
        int(row["anchor_count"]) * float(row["failure_rate"]) for row in summary_rows
    )
    weighted_survival = sum(
        int(row["anchor_count"]) * float(row["mean_survival_ratio"])
        for row in summary_rows
    )
    weighted_mean_reward = sum(
        int(row["anchor_count"]) * float(row.get("mean_reward_per_step", math.nan))
        for row in summary_rows
    )
    weighted_mean_episode_reward = sum(
        int(row["anchor_count"]) * float(row.get("mean_episode_reward_sum", math.nan))
        for row in summary_rows
    )
    denom = float(total_anchor_count)
    fixed_horizon_reward_mean = math.nan
    if anchor_rows:
        total_reward = sum(
            float(row.get("episode_reward_sum", 0.0)) for row in anchor_rows
        )
        total_effective_horizon = sum(
            float(row.get("effective_window_steps", row.get("window_steps", 0.0)))
            for row in anchor_rows
        )
        if total_effective_horizon > 0.0:
            fixed_horizon_reward_mean = total_reward / total_effective_horizon
    elif all("fixed_horizon_reward_mean" in row for row in summary_rows):
        fixed_horizon_reward_mean = (
            sum(
                int(row["anchor_count"]) * float(row["fixed_horizon_reward_mean"])
                for row in summary_rows
            )
            / denom
        )
    return (
        weighted_failure / denom,
        weighted_survival / denom,
        weighted_mean_reward / denom,
        weighted_mean_episode_reward / denom,
        fixed_horizon_reward_mean,
    )


def _overall_success_only_tracking_metric(
    summary_rows: Sequence[dict[str, object]],
    key: str,
) -> float:
    weighted_sum = 0.0
    total_weight = 0
    for row in summary_rows:
        value = float(row.get(key, math.nan))
        if not math.isfinite(value):
            continue
        weight = int(row.get("tracking_success_count", 0))
        if weight <= 0:
            continue
        weighted_sum += float(weight) * value
        total_weight += weight
    if total_weight <= 0:
        return math.nan
    return weighted_sum / float(total_weight)


def _row_failed(row: dict[str, object]) -> bool:
    raw_value = row.get("failed", False)
    if isinstance(raw_value, str):
        return raw_value.strip().lower() in {"1", "true", "yes"}
    return bool(raw_value)


def _overall_phc_success_frame_weighted_metric(
    anchor_rows: Sequence[dict[str, object]],
    key: str,
) -> float:
    step_key = _PHC_METRIC_STEP_FIELDS[key]
    weighted_sum = 0.0
    total_weight = 0
    for row in anchor_rows:
        if _row_failed(row):
            continue
        value = float(row.get(key, math.nan))
        if not math.isfinite(value):
            continue
        weight = int(row.get(step_key, 0))
        if weight <= 0:
            continue
        weighted_sum += float(weight) * value
        total_weight += weight
    if total_weight <= 0:
        return math.nan
    return weighted_sum / float(total_weight)


def compute_adaptive_alignment_metrics(
    *,
    anchor_rows: Sequence[dict[str, object]],
    checkpoint_path: str | Path,
) -> dict[str, float]:
    if not anchor_rows:
        return {
            "corr_dist_failure": math.nan,
            "corr_dist_survival": math.nan,
            "top10_mass": math.nan,
            "top20_mass": math.nan,
            "weighted_failure": math.nan,
            "uniform_failure": math.nan,
            "failure_focus_gain": math.nan,
            "weighted_survival": math.nan,
            "uniform_survival": math.nan,
        }

    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    adaptive_state = payload.get("env_state", {}).get("adaptive_sampling", None)
    if not isinstance(adaptive_state, dict):
        return {
            "corr_dist_failure": math.nan,
            "corr_dist_survival": math.nan,
            "top10_mass": math.nan,
            "top20_mass": math.nan,
            "weighted_failure": math.nan,
            "uniform_failure": math.nan,
            "failure_focus_gain": math.nan,
            "weighted_survival": math.nan,
            "uniform_survival": math.nan,
        }

    chunk_size = int(adaptive_state["chunk_size"])
    dist = np.asarray(adaptive_state["dist"], dtype=np.float64)

    clip_meta: list[tuple[int, str, int, int]] = []
    seen_clip_idx = sorted({int(row["clip_idx"]) for row in anchor_rows})
    for clip_idx in seen_clip_idx:
        clip_rows = [row for row in anchor_rows if int(row["clip_idx"]) == clip_idx]
        clip_name = str(clip_rows[0]["clip_name"])
        clip_length = int(clip_rows[0]["clip_length"])
        clip_units = int(math.ceil(float(clip_length) / float(chunk_size)))
        clip_meta.append((clip_idx, clip_name, clip_length, clip_units))

    clip_offset: dict[int, int] = {}
    offset = 0
    for clip_idx, _clip_name, _clip_length, clip_units in clip_meta:
        clip_offset[clip_idx] = offset
        offset += clip_units

    unit_stats: dict[int, dict[str, list[float]]] = {}
    for row in anchor_rows:
        clip_idx = int(row["clip_idx"])
        unit = clip_offset[clip_idx] + int(row["start_frame_idx"]) // chunk_size
        stats = unit_stats.setdefault(unit, {"failed": [], "survival_ratio": []})
        stats["failed"].append(float(row["failed"]))
        stats["survival_ratio"].append(float(row["survival_ratio"]))

    failure_rates: list[float] = []
    survival_ratios: list[float] = []
    probs: list[float] = []
    for unit in sorted(unit_stats.keys()):
        if unit >= len(dist):
            continue
        stats = unit_stats[unit]
        failure_rates.append(float(np.mean(stats["failed"])))
        survival_ratios.append(float(np.mean(stats["survival_ratio"])))
        probs.append(float(dist[unit]))

    fr = np.asarray(failure_rates, dtype=np.float64)
    sr = np.asarray(survival_ratios, dtype=np.float64)
    pp = np.asarray(probs, dtype=np.float64)
    if len(fr) < 2 or float(np.std(pp)) <= 1.0e-12 or float(np.std(fr)) <= 1.0e-12:
        corr_failure = math.nan
    else:
        corr_failure = float(np.corrcoef(pp, fr)[0, 1])
    if len(sr) < 2 or float(np.std(pp)) <= 1.0e-12 or float(np.std(sr)) <= 1.0e-12:
        corr_survival = math.nan
    else:
        corr_survival = float(np.corrcoef(pp, sr)[0, 1])

    order_fail = np.argsort(fr)[::-1]
    cut10 = max(1, int(math.ceil(len(fr) * 0.10)))
    cut20 = max(1, int(math.ceil(len(fr) * 0.20)))
    mask10 = np.zeros(len(fr), dtype=bool)
    mask20 = np.zeros(len(fr), dtype=bool)
    mask10[order_fail[:cut10]] = True
    mask20[order_fail[:cut20]] = True

    weighted_failure = float(np.sum(pp * fr))
    uniform_failure = float(np.mean(fr))
    weighted_survival = float(np.sum(pp * sr))
    uniform_survival = float(np.mean(sr))
    focus_gain = weighted_failure / max(uniform_failure, 1.0e-8)

    return {
        "corr_dist_failure": corr_failure,
        "corr_dist_survival": corr_survival,
        "top10_mass": float(pp[mask10].sum()),
        "top20_mass": float(pp[mask20].sum()),
        "weighted_failure": weighted_failure,
        "uniform_failure": uniform_failure,
        "failure_focus_gain": float(focus_gain),
        "weighted_survival": weighted_survival,
        "uniform_survival": uniform_survival,
    }


def build_clip_failure_benchmark_record(
    *,
    iteration_index: int,
    step_index: int,
    frames: int,
    active_train_time_s: float | None = None,
    checkpoint_time_s: float | None = None,
    checkpoint_path: Path,
    output_root: Path,
    result: dict[str, object],
    alignment_metrics: dict[str, float],
) -> dict[str, object]:
    anchor_rows = list(result.get("anchor_rows", []))
    summary_rows = list(result.get("summary_rows", []))
    metadata = dict(result.get("metadata", {}))
    (
        overall_failure_rate,
        overall_survival_ratio,
        overall_mean_reward_per_step,
        overall_mean_episode_reward_sum,
        overall_fixed_horizon_reward_mean,
    ) = _overall_clip_metrics(summary_rows, anchor_rows=anchor_rows)

    worst_clip_name = ""
    worst_clip_failure_rate = math.nan
    if summary_rows:
        worst_row = max(summary_rows, key=lambda row: float(row["failure_rate"]))
        worst_clip_name = str(worst_row["clip_name"])
        worst_clip_failure_rate = float(worst_row["failure_rate"])

    clip_rows = [
        {
            "clip_name": str(row["clip_name"]),
            "failure_rate": float(row["failure_rate"]),
            "survival_ratio": float(row["mean_survival_ratio"]),
            "mean_reward_per_step": float(row.get("mean_reward_per_step", math.nan)),
            "mean_episode_reward_sum": float(
                row.get("mean_episode_reward_sum", math.nan)
            ),
            "fixed_horizon_reward_mean": float(
                row.get("fixed_horizon_reward_mean", math.nan)
            ),
            "anchor_count": int(row["anchor_count"]),
            **(
                {
                    "tracking_success_count": int(row.get("tracking_success_count", 0)),
                    **{
                        key: float(row.get(key, math.nan))
                        for key in _TRACKING_METRIC_FIELDS
                    },
                }
                if any(key in row for key in _TRACKING_METRIC_FIELDS)
                else {}
            ),
        }
        for row in summary_rows
    ]

    record = {
        "iteration": int(iteration_index),
        "step": int(step_index),
        "frames": int(frames),
        "mode": str(metadata.get("mode", "anchor")),
        "checkpoint_path": str(checkpoint_path),
        "output_root": str(output_root),
        "runtime_seconds": float(metadata.get("runtime_seconds", math.nan)),
        "anchor_count": int(metadata.get("anchor_count", 0)),
        "overall_failure_rate": float(overall_failure_rate),
        "overall_survival_ratio": float(overall_survival_ratio),
        "overall_mean_reward_per_step": float(overall_mean_reward_per_step),
        "overall_mean_episode_reward_sum": float(overall_mean_episode_reward_sum),
        "overall_fixed_horizon_reward_mean": float(overall_fixed_horizon_reward_mean),
        **{
            f"overall_{key}": (
                _overall_phc_success_frame_weighted_metric(anchor_rows, key)
                if anchor_rows and key in _PHC_METRIC_STEP_FIELDS
                else _overall_success_only_tracking_metric(summary_rows, key)
            )
            for key in _TRACKING_METRIC_FIELDS
        },
        "worst_clip_name": worst_clip_name,
        "worst_clip_failure_rate": float(worst_clip_failure_rate),
        "corr_dist_failure": float(
            alignment_metrics.get("corr_dist_failure", math.nan)
        ),
        "corr_dist_survival": float(
            alignment_metrics.get("corr_dist_survival", math.nan)
        ),
        "top10_mass": float(alignment_metrics.get("top10_mass", math.nan)),
        "top20_mass": float(alignment_metrics.get("top20_mass", math.nan)),
        "failure_focus_gain": float(
            alignment_metrics.get("failure_focus_gain", math.nan)
        ),
        "weighted_failure": float(alignment_metrics.get("weighted_failure", math.nan)),
        "uniform_failure": float(alignment_metrics.get("uniform_failure", math.nan)),
        "clip_rows": clip_rows,
    }
    if record["mode"] == "full_motion":
        record["motion_count"] = int(
            metadata.get("motion_count", metadata.get("anchor_count", 0))
        )
        record["motion_success_rate"] = float(1.0 - float(overall_failure_rate))
    if active_train_time_s is not None:
        record["active_train_time_s"] = float(active_train_time_s)
    if checkpoint_time_s is not None:
        record["checkpoint_time_s"] = float(checkpoint_time_s)
    return record


def build_clip_failure_benchmark_scalars(
    record: dict[str, object], *, include_per_clip: bool = True
) -> dict[str, float]:
    scalars = {
        "Clip Failure / Overall failure rate": float(record["overall_failure_rate"]),
        "Clip Failure / Overall survival ratio": float(
            record["overall_survival_ratio"]
        ),
        "Clip Failure / Overall mean reward": float(
            record.get("overall_mean_reward_per_step", math.nan)
        ),
        "Clip Failure / Overall mean episode reward": float(
            record.get("overall_mean_episode_reward_sum", math.nan)
        ),
        "Clip Failure / Overall fixed-horizon reward mean": float(
            record.get("overall_fixed_horizon_reward_mean", math.nan)
        ),
        **{
            f"Clip Failure / Overall {label}": float(
                record.get(f"overall_{key}", math.nan)
            )
            for key, label in _TRACKING_METRIC_FIELDS.items()
        },
        "Clip Failure / Runtime (s)": float(record["runtime_seconds"]),
        "Clip Failure / Adaptive corr(dist,failure)": float(
            record["corr_dist_failure"]
        ),
        "Clip Failure / Adaptive top20 hard mass": float(record["top20_mass"]),
        "Clip Failure / Adaptive focus gain": float(record["failure_focus_gain"]),
    }
    if str(record.get("mode", "anchor")) == "full_motion":
        scalars["Clip Failure / Motion success rate"] = float(
            record.get(
                "motion_success_rate",
                1.0 - float(record["overall_failure_rate"]),
            )
        )
    if "frames" in record:
        scalars["Trainer / Frame"] = float(record["frames"])
    if "active_train_time_s" in record:
        scalars["Trainer / Active train time (s)"] = float(
            record["active_train_time_s"]
        )
    if "checkpoint_time_s" in record:
        scalars["Trainer / Checkpoint time (s)"] = float(record["checkpoint_time_s"])
    if not include_per_clip:
        return scalars
    for row in list(record.get("clip_rows", [])):
        clip_stem = _clip_metric_stem(str(row["clip_name"]))
        scalars[f"Clip Failure / {clip_stem} failure rate"] = float(row["failure_rate"])
        scalars[f"Clip Failure / {clip_stem} survival ratio"] = float(
            row["survival_ratio"]
        )
        scalars[f"Clip Failure / {clip_stem} mean reward"] = float(
            row.get("mean_reward_per_step", math.nan)
        )
        scalars[f"Clip Failure / {clip_stem} mean episode reward"] = float(
            row.get("mean_episode_reward_sum", math.nan)
        )
        scalars[f"Clip Failure / {clip_stem} fixed-horizon reward mean"] = float(
            row.get("fixed_horizon_reward_mean", math.nan)
        )
        for key, label in _TRACKING_METRIC_FIELDS.items():
            scalars[f"Clip Failure / {clip_stem} {label}"] = float(
                row.get(key, math.nan)
            )
    return scalars


def append_clip_failure_benchmark_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


def load_clip_failure_benchmark_history(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    records.sort(key=lambda item: (int(item["iteration"]), int(item["step"])))
    return records


def write_clip_failure_benchmark_history(
    output_dir: Path, records: Sequence[dict[str, object]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        return

    csv_path = output_dir / "history_summary.csv"
    rows = [
        {
            "iteration": int(record["iteration"]),
            "step": int(record["step"]),
            "frames": int(record["frames"]),
            "mode": str(record.get("mode", "anchor")),
            "motion_count": int(record.get("motion_count", 0)),
            "motion_success_rate": float(record.get("motion_success_rate", math.nan)),
            "active_train_time_s": float(record.get("active_train_time_s", math.nan)),
            "checkpoint_time_s": float(record.get("checkpoint_time_s", math.nan)),
            "overall_failure_rate": float(record["overall_failure_rate"]),
            "overall_survival_ratio": float(record["overall_survival_ratio"]),
            "overall_mean_reward_per_step": float(
                record.get("overall_mean_reward_per_step", math.nan)
            ),
            "overall_mean_episode_reward_sum": float(
                record.get("overall_mean_episode_reward_sum", math.nan)
            ),
            **{
                f"overall_{key}": float(record.get(f"overall_{key}", math.nan))
                for key in _TRACKING_METRIC_FIELDS
            },
            "runtime_seconds": float(record["runtime_seconds"]),
            "corr_dist_failure": float(record["corr_dist_failure"]),
            "top20_mass": float(record["top20_mass"]),
            "failure_focus_gain": float(record["failure_focus_gain"]),
            "worst_clip_name": str(record["worst_clip_name"]),
            "worst_clip_failure_rate": float(record["worst_clip_failure_rate"]),
        }
        for record in records
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = output_dir / "history_summary.md"
    lines = [
        "# Clip Failure Benchmark History",
        "",
        "| iter | mode | overall failure | motion success | overall survival | mean reward | mean episode reward | corr(dist,failure) | top20 hard mass | focus gain | worst clip | worst clip failure |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {iteration} | {mode} | {overall_failure_rate:.4f} | "
            "{motion_success_rate:.4f} | {overall_survival_ratio:.4f} | "
            "{overall_mean_reward_per_step:.4f} | "
            "{overall_mean_episode_reward_sum:.2f} | "
            "{corr_dist_failure:.4f} | {top20_mass:.4f} | {failure_focus_gain:.4f} | "
            "{worst_clip_name} | {worst_clip_failure_rate:.4f} |".format(**row)
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    xs = [int(row["iteration"]) for row in rows]
    axes[0].plot(xs, [float(row["overall_failure_rate"]) for row in rows], marker="o")
    axes[0].plot(
        xs,
        [float(row["overall_mean_reward_per_step"]) for row in rows],
        marker="s",
        label="mean reward",
    )
    axes[0].set_ylabel("overall failure rate")
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Clip Failure Benchmark History")
    axes[0].legend(frameon=False)

    axes[1].plot(
        xs,
        [float(row["corr_dist_failure"]) for row in rows],
        marker="o",
        label="corr(dist, failure)",
    )
    axes[1].plot(
        xs,
        [float(row["top20_mass"]) for row in rows],
        marker="s",
        label="top20 hard mass",
    )
    axes[1].plot(
        xs,
        [float(row["failure_focus_gain"]) for row in rows],
        marker="^",
        label="focus gain",
    )
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("adaptive alignment")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "history_overview.png", dpi=180)
    plt.close(fig)
