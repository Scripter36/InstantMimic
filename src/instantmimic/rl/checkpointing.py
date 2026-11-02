from __future__ import annotations

import copy
import logging
import os
import sys
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch

from instantmimic.config.artifacts import (
    CHECKPOINT_METADATA_KEY,
    clone_checkpoint_metadata,
)
from instantmimic.rl.agent_protocol import CheckpointAgent
from instantmimic.rl.checkpoint_io import load_checkpoint_payload
from instantmimic.rl.eager_fallback import training_fallback_policy
from instantmimic.rl.env_access import unwrap_env

_LOGGER = logging.getLogger(__name__)
CHECKPOINT_LOAD_MODES = frozenset({"full", "policy_init"})
REPLAY_SIDECAR_KEY = "replay_sidecar"


def normalize_checkpoint_load_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in CHECKPOINT_LOAD_MODES:
        supported = ", ".join(sorted(CHECKPOINT_LOAD_MODES))
        raise ValueError(
            f"Unsupported checkpoint load mode {value!r}. Supported values: {supported}."
        )
    return mode


def load_full_agent_checkpoint(agent: CheckpointAgent, path: str | Path) -> Path:
    """Restore an agent from a checkpoint in full, and return the path used.

    Training reaches this through its `full` load mode; playback calls it
    directly. Playback has no other way in, which is the point: a partial load
    leaves the value head and optimizer at their initial state, and a run that
    did that while claiming to replay a checkpoint would look right and be
    wrong.
    """

    resolved = Path(path)
    agent.load(resolved)
    return resolved


@dataclass(eq=False)
class CheckpointSnapshotHandle:
    snapshot: dict[str, object]
    ready_event: torch.cuda.Event | None
    env_state: dict[str, object] | None
    serialized_with_producer: bool
    metadata: dict[str, object]
    consumed: bool = False


def _map_tree(value: object, leaf: Callable[[torch.Tensor], torch.Tensor]) -> object:
    """Rebuild a state tree, replacing every tensor with `leaf(tensor)`.

    Four walkers used to carry this recursion, one per leaf transform, and two
    of them dropped an `OrderedDict`'s `_metadata` on the way. That attribute is
    what `load_state_dict` reads for per-module version info, so a checkpoint
    written through the CPU copy came out of a different walker than the one it
    was snapshotted with and lost it.
    """

    if torch.is_tensor(value):
        return leaf(value)
    if isinstance(value, OrderedDict):
        target = OrderedDict(
            (key, _map_tree(item, leaf)) for key, item in value.items()
        )
        metadata = getattr(value, "_metadata", None)
        if metadata is not None:
            target._metadata = copy.deepcopy(metadata)
        return target
    if isinstance(value, dict):
        return {key: _map_tree(item, leaf) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tree(item, leaf) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_tree(item, leaf) for item in value)
    return value


def _clone_tree_to_device(value: object, *, device: torch.device) -> object:
    def leaf(tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach()
        if source.device == device:
            return source.clone()
        return source.to(device=device, copy=True)

    return _map_tree(value, leaf)


def _clone_tree_preserve_device(value: object) -> object:
    return _map_tree(value, lambda tensor: tensor.detach().clone())


def _clone_tree_to_cpu(
    value: object,
    *,
    non_blocking: bool,
    pin_memory: bool,
) -> object:
    def leaf(tensor: torch.Tensor) -> torch.Tensor:
        source = tensor.detach()
        if source.device.type == "cpu":
            return source.clone()
        target = torch.empty_like(source, device="cpu", pin_memory=pin_memory)
        target.copy_(source, non_blocking=non_blocking)
        return target

    return _map_tree(value, leaf)


def _allocate_snapshot_tree(value: object, *, device: torch.device) -> object:
    return _map_tree(
        value, lambda tensor: torch.empty_like(tensor.detach(), device=device)
    )


def _collect_tensor_pairs(
    source: object,
    target: object,
    out_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    if torch.is_tensor(source):
        if not torch.is_tensor(target):
            raise TypeError("Snapshot target must mirror the source tensor tree.")
        out_pairs.append((target, source))
        return
    if isinstance(source, OrderedDict):
        if not isinstance(target, OrderedDict):
            raise TypeError("Snapshot target must mirror the source OrderedDict tree.")
        for key, item in source.items():
            _collect_tensor_pairs(item, target[key], out_pairs)
        return
    if isinstance(source, dict):
        if not isinstance(target, dict):
            raise TypeError("Snapshot target must mirror the source dict tree.")
        for key, item in source.items():
            _collect_tensor_pairs(item, target[key], out_pairs)
        return
    if isinstance(source, list):
        if not isinstance(target, list) or len(source) != len(target):
            raise TypeError("Snapshot target must mirror the source list tree.")
        for src_item, dst_item in zip(source, target, strict=False):
            _collect_tensor_pairs(src_item, dst_item, out_pairs)
        return
    if isinstance(source, tuple):
        if not isinstance(target, tuple) or len(source) != len(target):
            raise TypeError("Snapshot target must mirror the source tuple tree.")
        for src_item, dst_item in zip(source, target, strict=False):
            _collect_tensor_pairs(src_item, dst_item, out_pairs)


def _write_payload_torch_save(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_parent_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _fsync_parent_directory(path: Path) -> None:
    # Windows cannot open directories through os.open for POSIX fsync. The
    # payload itself is still flushed, fsynced, and closed before atomic replace.
    if sys.platform == "win32":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


class CheckpointManager:
    """Synchronous checkpoint coordinator with async-ready best-snapshot capture."""

    def __init__(
        self,
        agent: CheckpointAgent,
        env: Any,
        device: torch.device,
        *,
        producer_torch_stream: torch.cuda.Stream | None = None,
        checkpoint_metadata: dict[str, object] | None = None,
        save_replay: bool = False,
    ) -> None:
        self._agent = agent
        self._env = unwrap_env(env)
        self._device = device
        self._save_replay = bool(save_replay)
        self._checkpoint_metadata = (
            None
            if checkpoint_metadata is None
            else clone_checkpoint_metadata(checkpoint_metadata)
        )
        self._producer_torch_stream = (
            producer_torch_stream if device.type == "cuda" else None
        )
        self._snapshot_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._cpu_copy_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._best_candidate_snapshot: dict[str, object] | None = None
        self._best_candidate_ready_event: torch.cuda.Event | None = None
        self._best_candidate_env_state: dict[str, object] | None = None
        self._best_candidate_serialized_with_producer = False
        self._best_snapshot_graph: torch.cuda.CUDAGraph | None = None
        self._best_snapshot_graph_failed = False
        self._eager_fallback = training_fallback_policy()
        self._best_snapshot_buffers: dict[str, object] | None = None
        self._eval_candidate_handles: list[CheckpointSnapshotHandle] = []
        self._checkpoint_gpu_intervals: list[
            tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._checkpoint_wait_intervals: list[tuple[int, int]] = []
        self._timing_anchor_event: torch.cuda.Event | None = None
        self._timing_anchor_ns = 0
        self.timing_calibration_uncertainty_ns = 0
        if self._producer_torch_stream is not None:
            # Prewarm before PPO update graph capture so both graphs see a stable
            # allocator / capture ordering from the start.
            self._initialize_best_snapshot_graph(self._producer_torch_stream)
        if device.type == "cuda":
            self._initialize_timing_anchor()

    def _initialize_timing_anchor(self) -> None:
        # Startup synchronization relates CUDA's event clock to the host clock.
        # Warm the lazy event allocation and clear producer backlog first so
        # neither widens the calibration bracket. Candidate capture never waits.
        anchor = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(self._device)
        anchor.record(stream)
        anchor.synchronize()
        before_ns = perf_counter_ns()
        anchor.record(stream)
        anchor.synchronize()
        after_ns = perf_counter_ns()
        self._timing_anchor_event = anchor
        self._timing_anchor_ns = (before_ns + after_ns) // 2
        self.timing_calibration_uncertainty_ns = (after_ns - before_ns + 1) // 2

    def _write_payload(self, path: str | Path, payload: dict[str, object]) -> None:
        _write_payload_torch_save(path, payload)

    def _agent_state(self, *, best: bool) -> dict[str, object]:
        if best:
            return self._agent.best_state_dict()
        checkpoint_state = getattr(self._agent, "checkpoint_state_dict", None)
        if callable(checkpoint_state):
            return checkpoint_state()
        return self._agent.state_dict()

    def _replay_sidecar_path(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        replay_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}.replay-{uuid.uuid4().hex}{checkpoint_path.suffix}"
        )
        return replay_path

    def _write_replay_sidecar(
        self,
        path: str | Path,
        payload: dict[str, object],
    ) -> Path:
        replay_state = getattr(self._agent, "replay_state_dict", None)
        if not callable(replay_state):
            raise TypeError(
                "experiment.checkpoint.replay=true requires an agent with "
                "replay_state_dict()."
            )
        replay_path = self._replay_sidecar_path(path)
        replay_payload = replay_state()
        if not isinstance(replay_payload, dict):
            raise TypeError("Agent replay_state_dict() must return a dictionary.")
        self._write_payload(replay_path, replay_payload)
        payload[REPLAY_SIDECAR_KEY] = replay_path.name
        return replay_path

    @staticmethod
    def _referenced_replay_sidecar(path: str | Path) -> Path | None:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            return None
        try:
            payload = load_checkpoint_payload(checkpoint_path)
        except (OSError, RuntimeError, EOFError):
            return None
        if not isinstance(payload, dict):
            return None
        filename = payload.get(REPLAY_SIDECAR_KEY)
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).is_absolute()
            or Path(filename).name != filename
        ):
            return None
        return checkpoint_path.parent / filename

    def _attach_checkpoint_metadata(self, payload: dict[str, object]) -> None:
        if self._checkpoint_metadata is not None:
            payload[CHECKPOINT_METADATA_KEY] = clone_checkpoint_metadata(
                self._checkpoint_metadata
            )

    def _environment_state(
        self, *, policy_ready_event: torch.cuda.Event | None = None
    ) -> tuple[dict[str, object] | None, torch.cuda.Event | None]:
        stream = self._producer_torch_stream or self._snapshot_stream
        if self._device.type != "cuda" or stream is None:
            state = self._env.get_checkpoint_state()
            if not state:
                return None, policy_ready_event
            return _clone_tree_preserve_device(state), policy_ready_event

        # Environment checkpoint hooks may clone live CUDA tensors themselves.
        # Run both that extraction and our detached snapshot on one stream so
        # the second copy cannot race the hook-created tensors. Resolve the
        # caller stream before entering the checkpoint stream: without an
        # explicit producer, current_stream() inside the context would return
        # the consumer itself and silently omit the required dependency.
        caller_stream = torch.cuda.current_stream(self._device)
        if caller_stream.cuda_stream != stream.cuda_stream:
            # Environment state is written by rollout/reset work on the caller
            # stream, even when policy updates use an explicit producer stream.
            # Order the snapshot after those already-queued writes.  The
            # reciprocal caller wait below is queued later, so this does not
            # create a cycle and neither dependency blocks the host.
            stream.wait_stream(caller_stream)
        if policy_ready_event is not None:
            # The policy may have been copied on a separate fallback stream.
            # Make the returned readiness cover both parts of the candidate,
            # with dependency waits outside the environment's timed work.
            stream.wait_event(policy_ready_event)
        with torch.cuda.stream(stream):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            state = self._env.get_checkpoint_state()
            snapshot = None if not state else _clone_tree_preserve_device(state)
            ready_event = torch.cuda.Event(enable_timing=True)
            ready_event.record(stream)
        self._checkpoint_gpu_intervals.append((start_event, ready_event))
        if caller_stream.cuda_stream != stream.cuda_stream:
            # The next rollout mutates environment state before its update
            # boundary. Order those caller-stream launches after the complete
            # snapshot without synchronizing the host, including when capture
            # uses an explicit producer but rollout uses the caller stream.
            caller_stream.wait_event(ready_event)
        return snapshot, ready_event

    def _wait_for_producer(
        self,
        consumer_stream: torch.cuda.Stream,
    ) -> None:
        producer_stream = self._producer_torch_stream
        if producer_stream is None:
            producer_stream = torch.cuda.current_stream(self._device)
        if consumer_stream.cuda_stream != producer_stream.cuda_stream:
            consumer_stream.wait_stream(producer_stream)

    @property
    def eager_fallback_stages(self) -> tuple[str, ...]:
        """Capture stages this manager fell back on, for the metrics to report.

        Training degrades rather than raising here, so without a reported flag a
        run that lost its snapshot graph looks in the metrics exactly like one
        that never needed it -- which is what would make a later throughput
        comparison misleading.
        """

        return tuple(self._eager_fallback.recorded)

    def _initialize_best_snapshot_graph(
        self,
        capture_stream: torch.cuda.Stream,
    ) -> bool:
        if self._device.type != "cuda" or self._best_snapshot_graph is not None:
            return False
        if self._best_snapshot_graph_failed:
            return False

        source = self._agent_state(best=True)
        buffers = _allocate_snapshot_tree(source, device=self._device)
        tensor_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        _collect_tensor_pairs(source, buffers, tensor_pairs)

        try:
            # Capture the snapshot graph on the producer stream itself so the
            # replay stays naturally ordered with PPO update graph launches.
            capture_stream.synchronize()
            with torch.cuda.stream(capture_stream):
                for dst, src in tensor_pairs:
                    dst.copy_(src)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    for dst, src in tensor_pairs:
                        dst.copy_(src)
        except Exception as exc:  # noqa: BLE001 - graph capture can fail in any
            # backend layer, and training continues either way.
            self._best_snapshot_graph_failed = True
            self._best_snapshot_graph = None
            self._best_snapshot_buffers = None
            # Training degrades rather than raising: a snapshot graph that will
            # not capture costs copy throughput, and ending a long run over it
            # would be worse. The policy records it so the flag can reach the
            # metrics instead of living only in the log.
            self._eager_fallback.degrade("best-checkpoint snapshot graph", exc)
            return False

        self._best_snapshot_graph = graph
        self._best_snapshot_buffers = buffers
        _LOGGER.info("Captured best-checkpoint snapshot graph on producer stream.")
        return True

    def _snapshot_state(
        self,
        *,
        best: bool,
        snapshot_stream: torch.cuda.Stream | None,
    ) -> tuple[dict[str, object], torch.cuda.Event | None, bool]:
        state = self._agent_state(best=best)
        if self._device.type != "cuda" or snapshot_stream is None:
            snapshot = _clone_tree_to_device(state, device=self._device)
            return snapshot, None, False

        self._wait_for_producer(snapshot_stream)
        with torch.cuda.stream(snapshot_stream):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(snapshot_stream)
            snapshot = _clone_tree_to_device(state, device=self._device)
            ready_event = torch.cuda.Event(enable_timing=True)
            ready_event.record(snapshot_stream)
        self._checkpoint_gpu_intervals.append((start_event, ready_event))
        serialized_with_producer = (
            self._producer_torch_stream is not None
            and snapshot_stream.cuda_stream == self._producer_torch_stream.cuda_stream
        )
        return snapshot, ready_event, serialized_with_producer

    def _capture_best_snapshot(
        self,
    ) -> tuple[dict[str, object], torch.cuda.Event | None, bool]:
        producer_stream = self._producer_torch_stream
        if (
            producer_stream is not None
            and self._best_snapshot_graph is not None
            and self._best_snapshot_buffers is not None
        ):
            with torch.cuda.stream(producer_stream):
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record(producer_stream)
                self._best_snapshot_graph.replay()
                ready_event = torch.cuda.Event(enable_timing=True)
                ready_event.record(producer_stream)
            self._checkpoint_gpu_intervals.append((start_event, ready_event))
            return self._best_snapshot_buffers, ready_event, True
        return self._snapshot_state(best=True, snapshot_stream=self._snapshot_stream)

    def _snapshot_to_cpu_payload(
        self,
        snapshot: object,
    ) -> tuple[dict[str, object], torch.cuda.Event | None]:
        if self._device.type != "cuda" or self._cpu_copy_stream is None:
            payload = _clone_tree_to_cpu(
                snapshot,
                non_blocking=False,
                pin_memory=False,
            )
            return payload, None

        with torch.cuda.stream(self._cpu_copy_stream):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(self._cpu_copy_stream)
            payload = _clone_tree_to_cpu(
                snapshot,
                non_blocking=True,
                pin_memory=True,
            )
            ready_event = torch.cuda.Event(enable_timing=True)
            ready_event.record(self._cpu_copy_stream)
        self._checkpoint_gpu_intervals.append((start_event, ready_event))
        return payload, ready_event

    def _save(self, path: str | Path, *, best: bool) -> None:
        if self._device.type == "cuda":
            producer = self._producer_torch_stream or torch.cuda.current_stream(
                self._device
            )
            self._synchronize_checkpoint_work(producer)
        payload = self._agent_state(best=best)
        env_state, _ = self._environment_state()
        if env_state is not None:
            payload["env_state"] = env_state
        self._wait_for_pending_gpu_work()
        self._attach_checkpoint_metadata(payload)
        previous_replay_path = (
            self._referenced_replay_sidecar(path) if not best else None
        )
        new_replay_path = None
        if self._save_replay and not best:
            new_replay_path = self._write_replay_sidecar(path, payload)
        try:
            self._write_payload(path, payload)
        except BaseException:
            if new_replay_path is not None:
                new_replay_path.unlink(missing_ok=True)
            raise
        if previous_replay_path is not None and previous_replay_path != new_replay_path:
            previous_replay_path.unlink(missing_ok=True)
            _fsync_parent_directory(previous_replay_path.parent)

    def save(self, path: str | Path) -> None:
        self._save(path, best=False)

    def save_best(self, path: str | Path) -> None:
        self._save(path, best=True)

    def capture_best_candidate(self) -> None:
        snapshot, ready_event, serialized_with_producer = self._capture_best_snapshot()
        env_state, ready_event = self._environment_state(policy_ready_event=ready_event)
        self._best_candidate_snapshot = snapshot
        self._best_candidate_ready_event = ready_event
        self._best_candidate_env_state = env_state
        self._best_candidate_serialized_with_producer = serialized_with_producer

    def capture_eval_candidate(
        self,
        *,
        metadata: dict[str, object] | None = None,
    ) -> CheckpointSnapshotHandle:
        snapshot_stream = self._producer_torch_stream or self._snapshot_stream
        snapshot, ready_event, serialized_with_producer = self._snapshot_state(
            best=True,
            snapshot_stream=snapshot_stream,
        )
        env_state, ready_event = self._environment_state(policy_ready_event=ready_event)
        handle = CheckpointSnapshotHandle(
            snapshot=snapshot,
            ready_event=ready_event,
            env_state=env_state,
            serialized_with_producer=serialized_with_producer,
            metadata={} if metadata is None else dict(metadata),
        )
        self._eval_candidate_handles.append(handle)
        return handle

    def _release_eval_candidate(self, handle: CheckpointSnapshotHandle) -> None:
        if handle.consumed or handle not in self._eval_candidate_handles:
            raise ValueError("Evaluation checkpoint snapshot handle is not active.")
        self._eval_candidate_handles.remove(handle)
        handle.consumed = True

    def promote_eval_candidate(self, handle: CheckpointSnapshotHandle) -> None:
        self._release_eval_candidate(handle)
        self._best_candidate_snapshot = handle.snapshot
        self._best_candidate_ready_event = handle.ready_event
        self._best_candidate_env_state = handle.env_state
        self._best_candidate_serialized_with_producer = handle.serialized_with_producer

    def discard_eval_candidate(self, handle: CheckpointSnapshotHandle) -> None:
        self._release_eval_candidate(handle)
        if handle.ready_event is not None and not handle.serialized_with_producer:
            self._synchronize_checkpoint_work(handle.ready_event)

    def flush_best_candidate(self, path: str | Path) -> bool:
        if self._best_candidate_snapshot is None:
            return False

        if self._best_candidate_ready_event is not None:
            self._synchronize_checkpoint_work(self._best_candidate_ready_event)
        # Settle any other pending checkpoint copies before writing the payload.
        self._wait_for_pending_gpu_work()

        payload, ready_event = self._snapshot_to_cpu_payload(
            self._best_candidate_snapshot
        )
        if ready_event is not None:
            self._synchronize_checkpoint_work(ready_event)
        if self._best_candidate_env_state is not None:
            payload["env_state"] = self._best_candidate_env_state

        self._attach_checkpoint_metadata(payload)
        self._write_payload(path, payload)
        self._best_candidate_snapshot = None
        self._best_candidate_ready_event = None
        self._best_candidate_env_state = None
        self._best_candidate_serialized_with_producer = False
        return True

    def wait_for_update_boundary(self) -> None:
        if (
            self._best_candidate_ready_event is not None
            and not self._best_candidate_serialized_with_producer
        ):
            self._synchronize_checkpoint_work(self._best_candidate_ready_event)
        for handle in self._eval_candidate_handles:
            if handle.ready_event is not None and not handle.serialized_with_producer:
                self._synchronize_checkpoint_work(handle.ready_event)

    def _synchronize_checkpoint_work(self, work: Any) -> None:
        before_ns = perf_counter_ns()
        work.synchronize()
        self._checkpoint_wait_intervals.append((before_ns, perf_counter_ns()))

    def _wait_for_pending_gpu_work(self) -> None:
        for _, end_event in self._checkpoint_gpu_intervals:
            self._synchronize_checkpoint_work(end_event)

    def consume_checkpoint_wait_intervals_ns(self) -> list[tuple[int, int]]:
        intervals = self._checkpoint_wait_intervals
        self._checkpoint_wait_intervals = []
        return intervals

    def consume_checkpoint_gpu_intervals_ns(
        self, *, synchronize: bool = False
    ) -> list[tuple[int, int]]:
        """Return completed GPU intervals on the calibrated host time axis.

        Event durations are exact at CUDA's event resolution; their host-axis
        placement has ``timing_calibration_uncertainty_ns`` startup uncertainty.
        Keeping intervals, instead of just durations, permits a union with CPU
        serialization work without counting their overlap twice.
        """
        completed: list[tuple[int, int]] = []
        intervals = self._checkpoint_gpu_intervals
        self._checkpoint_gpu_intervals = []
        for start_event, end_event in intervals:
            if not synchronize and not end_event.query():
                self._checkpoint_gpu_intervals.append((start_event, end_event))
                continue
            if synchronize:
                self._synchronize_checkpoint_work(end_event)
            anchor = self._timing_anchor_event
            if anchor is None:
                raise RuntimeError("CUDA checkpoint timing anchor is missing.")
            start_ns = self._timing_anchor_ns + round(
                anchor.elapsed_time(start_event) * 1_000_000
            )
            end_ns = start_ns + round(start_event.elapsed_time(end_event) * 1_000_000)
            completed.append((start_ns, end_ns))
        return completed

    def pending_checkpoint_gpu_interval_count(self) -> int:
        return len(self._checkpoint_gpu_intervals)

    def close(self) -> None:
        self._best_candidate_snapshot = None
        self._best_candidate_ready_event = None
        self._best_candidate_env_state = None
        self._best_candidate_serialized_with_producer = False
        self._best_snapshot_graph = None
        self._best_snapshot_buffers = None
        self._checkpoint_gpu_intervals = []
        self._eval_candidate_handles.clear()
