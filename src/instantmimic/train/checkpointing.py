from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch
from omegaconf import OmegaConf

from instantmimic.envs.config_access import get_value
from instantmimic.rl.checkpoint_io import load_checkpoint_payload
from instantmimic.rl.checkpointing import (
    _fsync_parent_directory,
    load_full_agent_checkpoint,
    normalize_checkpoint_load_mode,
)
from instantmimic.rl.normalization import adapt_observation_scaler_state

_LOGGER = logging.getLogger(__name__)
_TRAINING_STAGES = {"full", "musclenn_only"}


class TrainingCheckpointController:
    def __init__(
        self,
        *,
        checkpoint_manager: Any,
        checkpoints_dir: Path,
        persistence_disabled: bool = False,
    ) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.checkpoints_dir = Path(checkpoints_dir)
        self.persistence_disabled = bool(persistence_disabled)
        self.time_ns_acc = 0
        self._settled_time_ns = 0
        self._settled_foreground_overlap_ns = 0
        self._accounted_intervals: list[tuple[int, int]] = []
        self._foreground_intervals: list[tuple[int, int]] = []
        self._open_foreground_windows = 0
        self.best_pending = False
        self.best_written = False

    @property
    def best_path(self) -> Path:
        return self.checkpoints_dir / "best_agent.pt"

    def _timed(self, fn: Callable[[], Any]) -> Any:
        start_ns = perf_counter_ns()
        try:
            return fn()
        finally:
            end_ns = perf_counter_ns()
            # A synchronization can wait for unrelated producer backlog. Charge
            # its measured GPU checkpoint interval instead of that host wait.
            waits = self._consume_wait_intervals()
            pieces = [(start_ns, end_ns)]
            for wait_start, wait_end in waits:
                remainder = []
                for left, right in pieces:
                    if wait_end <= left or wait_start >= right:
                        remainder.append((left, right))
                    else:
                        if left < wait_start:
                            remainder.append((left, wait_start))
                        if wait_end < right:
                            remainder.append((wait_end, right))
                pieces = remainder
            self._account_intervals(pieces)
            self._collect_gpu_timing(synchronize=False)

    def _consume_wait_intervals(self) -> list[tuple[int, int]]:
        consume = getattr(
            self.checkpoint_manager, "consume_checkpoint_wait_intervals_ns", None
        )
        return consume() if callable(consume) else []

    def _account_intervals(self, intervals: list[tuple[int, int]]) -> None:
        if not intervals:
            return
        merged: list[tuple[int, int]] = []
        for start, end in sorted([*self._accounted_intervals, *intervals]):
            if end <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        self._accounted_intervals = merged
        self.time_ns_acc = self._settled_time_ns + sum(
            end - start for start, end in merged
        )

    def begin_eval_foreground(self) -> None:
        self._open_foreground_windows += 1

    def end_eval_foreground(self, start_ns: int, end_ns: int) -> None:
        self._foreground_intervals.append((start_ns, end_ns))
        self._open_foreground_windows -= 1

    @property
    def eval_checkpoint_overlap_time_ns(self) -> int:
        # Foreground windows are consecutive, non-overlapping host operations.
        return self._settled_foreground_overlap_ns + sum(
            max(0, min(end, right) - max(start, left))
            for start, end in self._foreground_intervals
            for left, right in self._accounted_intervals
        )

    def _settle_timing(self) -> None:
        if self._open_foreground_windows:
            return
        pending = getattr(
            self.checkpoint_manager, "pending_checkpoint_gpu_interval_count", None
        )
        if callable(pending) and pending() > 0:
            # Incomplete GPU intervals may overlap both the retained host work
            # and foreground windows. Compacting now would make those future
            # unions and intersections impossible to reconstruct exactly.
            return
        self._settled_foreground_overlap_ns = self.eval_checkpoint_overlap_time_ns
        self._settled_time_ns = self.time_ns_acc
        self._accounted_intervals.clear()
        self._foreground_intervals.clear()

    def _collect_gpu_timing(self, *, synchronize: bool) -> None:
        consume = getattr(
            self.checkpoint_manager, "consume_checkpoint_gpu_intervals_ns", None
        )
        if callable(consume):
            self._account_intervals(consume(synchronize=synchronize))
        self._consume_wait_intervals()

    def drain_timing(self) -> None:
        """Resolve pending event timing before final active-time reporting."""
        self._collect_gpu_timing(synchronize=True)
        self._settle_timing()

    def save(self, path: str | Path) -> Path:
        resolved = Path(path)
        self._timed(lambda: self.checkpoint_manager.save(resolved))
        return resolved

    def save_persistent(self, path: str | Path) -> Path | None:
        if self.persistence_disabled:
            return None
        return self.save(path)

    def iteration_path(self, iteration_index: int) -> Path:
        return self.checkpoints_dir / f"agent_iter_{int(iteration_index)}.pt"

    def save_iteration(self, iteration_index: int) -> Path | None:
        return self.save_persistent(self.iteration_path(iteration_index))

    def save_best(self, path: str | Path | None = None) -> Path | None:
        if self.persistence_disabled:
            return None
        resolved = self.best_path if path is None else Path(path)
        self._timed(lambda: self.checkpoint_manager.save_best(resolved))
        self.best_pending = False
        self.best_written = True
        return resolved

    def capture_best_candidate(self) -> None:
        if self.persistence_disabled:
            return
        self._timed(self.checkpoint_manager.capture_best_candidate)
        self.best_pending = True

    def capture_eval_candidate(self, *, metadata: dict[str, object]) -> object:
        return self._timed(
            lambda: self.checkpoint_manager.capture_eval_candidate(metadata=metadata)
        )

    def wait_for_update_boundary(self) -> None:
        # The event ledger measures actual GPU work. An empty boundary is not
        # checkpoint work, and synchronizing must not charge producer backlog.
        pending = getattr(
            self.checkpoint_manager, "pending_checkpoint_gpu_interval_count", None
        )
        if callable(pending) and pending() == 0:
            self._settle_timing()
            return
        self.checkpoint_manager.wait_for_update_boundary()
        # The manager synchronizes only side-stream snapshots that can race the
        # next update. Merely pending timing events, including producer-stream
        # snapshots, must remain nonblocking at an ordinary update boundary.
        self._collect_gpu_timing(synchronize=False)
        self._settle_timing()

    def promote_eval_candidate(self, handle: object) -> None:
        if self.persistence_disabled:
            self.discard_eval_candidate(handle)
            return
        self._timed(lambda: self.checkpoint_manager.promote_eval_candidate(handle))
        self.best_pending = True

    def promote_checkpoint_file(self, source: str | Path) -> Path | None:
        """Atomically promote the exact checkpoint file evaluated by a benchmark."""

        if self.persistence_disabled:
            return None
        source_path = Path(source)
        destination = self.best_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        def _promote() -> None:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
            try:
                shutil.copy2(source_path, temporary_path)
                # Windows fsync requires a descriptor opened for writing.
                sync_mode = "r+b" if os.name == "nt" else "rb"
                with temporary_path.open(sync_mode) as copied_file:
                    os.fsync(copied_file.fileno())
                os.replace(temporary_path, destination)
                _fsync_parent_directory(destination.parent)
            finally:
                temporary_path.unlink(missing_ok=True)

        self._timed(_promote)
        self.best_pending = False
        self.best_written = True
        return destination

    def discard_eval_candidate(self, handle: object) -> None:
        self._timed(lambda: self.checkpoint_manager.discard_eval_candidate(handle))

    def flush_best(self, path: str | Path | None = None) -> bool:
        if self.persistence_disabled:
            return False
        resolved = self.best_path if path is None else Path(path)
        flushed = bool(
            self._timed(lambda: self.checkpoint_manager.flush_best_candidate(resolved))
        )
        if flushed:
            self.best_pending = False
            self.best_written = True
        return flushed

    def flush_best_if_pending(self, path: str | Path | None = None) -> bool:
        if not self.best_pending:
            return False
        return self.flush_best(path)

    def ensure_clip_failure_checkpoint(
        self,
        *,
        iteration_index: int,
        checkpoint_due: bool,
    ) -> Path:
        checkpoint_path = self.iteration_path(iteration_index)
        if checkpoint_due:
            return checkpoint_path
        checkpoint_path = (
            self.checkpoints_dir
            / "clip_failure_benchmark_tmp"
            / f"agent_iter_{int(iteration_index)}.pt"
        )

        def _save_temporary_checkpoint() -> None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_manager.save(checkpoint_path)

        self._timed(_save_temporary_checkpoint)
        return checkpoint_path


def _normalize_training_stage(value: Any) -> str:
    text = "full" if value is None else str(value).strip().lower()
    if text not in _TRAINING_STAGES:
        supported = ", ".join(sorted(_TRAINING_STAGES))
        raise ValueError(
            f"Unsupported trainer.training_stage={value!r}. Supported values: {supported}."
        )
    return text


def _resolve_training_stage(
    *,
    cfg: Any,
    action_mode: str,
    resume_path: str | Path | None,
) -> str:
    stage_value = cfg.trainer.get("training_stage", "full")
    stage = _normalize_training_stage(stage_value)
    env_cfg = getattr(cfg, "env", None)
    if env_cfg is not None and _env_cfg_nested_get(
        env_cfg, "action", "muscle_nn", "frozen", default=False
    ):
        if stage == "musclenn_only":
            raise ValueError(
                "trainer.training_stage='musclenn_only' trains only MuscleNN, "
                "which env.action.muscle_nn.frozen keeps fixed."
            )
        if not resume_path:
            raise ValueError(
                "env.action.muscle_nn.frozen requires "
                "trainer.initialization.checkpoint to load a trained MuscleNN."
            )
    if stage != "musclenn_only":
        return stage
    if str(action_mode).strip().lower() != "mass":
        raise ValueError(
            "trainer.training_stage='musclenn_only' requires action_mode='mass'."
        )
    if not resume_path:
        raise ValueError(
            "trainer.training_stage='musclenn_only' requires "
            "trainer.initialization.checkpoint to load a frozen policy."
        )
    return stage


def _load_agent_checkpoint(
    agent: Any,
    checkpoint_path: str | Path,
    *,
    checkpoint_load_mode: str,
) -> None:
    resolved_mode = normalize_checkpoint_load_mode(checkpoint_load_mode)
    if resolved_mode == "full":
        load_full_agent_checkpoint(agent, checkpoint_path)
        return
    checkpoint_path = Path(checkpoint_path)
    # Read checkpoint tensors on the host before initializing policy weights.
    payload = load_checkpoint_payload(checkpoint_path)
    network_state = payload["network"]
    current_state = agent.network.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    widened_input_keys = {
        "trunk.0.weight",
        "actor_trunk.0.weight",
    }

    def _adapt_policy_init_tensor(
        key: str,
        *,
        current_tensor: torch.Tensor,
        checkpoint_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        if tuple(current_tensor.shape) == tuple(checkpoint_tensor.shape):
            return checkpoint_tensor
        if (
            key in widened_input_keys
            and current_tensor.ndim == 2
            and checkpoint_tensor.ndim == 2
            and int(current_tensor.shape[0]) == int(checkpoint_tensor.shape[0])
            and int(current_tensor.shape[1]) > int(checkpoint_tensor.shape[1])
        ):
            adapted = current_tensor.detach().clone()
            adapted[:, : checkpoint_tensor.shape[1]].copy_(
                checkpoint_tensor.to(
                    device=current_tensor.device,
                    dtype=current_tensor.dtype,
                )
            )
            return adapted
        return None

    for key, value in network_state.items():
        if str(key).startswith("value_head."):
            continue
        current_value = current_state.get(key)
        if current_value is None:
            continue
        adapted_value = _adapt_policy_init_tensor(
            key,
            current_tensor=current_value,
            checkpoint_tensor=value,
        )
        if adapted_value is None and tuple(current_value.shape) != tuple(value.shape):
            mismatched.append((key, tuple(current_value.shape), tuple(value.shape)))
            continue
        filtered_state[key] = adapted_value
    if mismatched:
        details = ", ".join(
            f"{key}: current={current_shape} checkpoint={checkpoint_shape}"
            for key, current_shape, checkpoint_shape in mismatched
        )
        raise RuntimeError(
            "trainer.initialization.type='policy_init' found incompatible non-value "
            f"network parameters while loading {checkpoint_path}: {details}"
        )

    incompatible = agent.network.load_state_dict(filtered_state, strict=False)
    allowed_missing = {
        "value_head.weight",
        "value_head.bias",
        "action_mean_scale",
        "action_std_scale",
        "action_log_std_scale",
    }
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    disallowed_missing = missing - allowed_missing
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Checkpoint network state mismatch while loading "
            f"{checkpoint_path} in policy_init mode: missing={sorted(disallowed_missing)} "
            f"unexpected={sorted(unexpected)}"
        )

    obs_scaler_state = payload.get("obs_scaler")
    if obs_scaler_state is not None:
        current_obs_scaler_state = agent.obs_scaler.state_dict()
        running_mean = obs_scaler_state.get("running_mean")
        running_variance = obs_scaler_state.get("running_variance")
        if obs_scaler_state is not None and (
            isinstance(running_mean, torch.Tensor)
            and isinstance(running_variance, torch.Tensor)
            and isinstance(current_obs_scaler_state.get("running_mean"), torch.Tensor)
            and isinstance(
                current_obs_scaler_state.get("running_variance"), torch.Tensor
            )
            and current_obs_scaler_state["running_mean"].ndim == 1
            and current_obs_scaler_state["running_variance"].ndim == 1
            and running_mean.ndim == 1
            and running_variance.ndim == 1
            and int(current_obs_scaler_state["running_mean"].shape[0])
            > int(running_mean.shape[0])
            and int(current_obs_scaler_state["running_variance"].shape[0])
            > int(running_variance.shape[0])
        ):

            def _transfer_leading(
                destination: torch.Tensor, source: torch.Tensor
            ) -> None:
                destination[: source.shape[0]].copy_(
                    source.to(device=destination.device, dtype=destination.dtype)
                )

            adapted_obs_scaler_state = adapt_observation_scaler_state(
                checkpoint_state=obs_scaler_state,
                current_state=current_obs_scaler_state,
                transfer=_transfer_leading,
            )
            if "current_count" in obs_scaler_state:
                adapted_obs_scaler_state["current_count"] = obs_scaler_state[
                    "current_count"
                ]
            agent.obs_scaler.load_state_dict(adapted_obs_scaler_state)
        elif obs_scaler_state is not None:
            agent.obs_scaler.load_state_dict(obs_scaler_state)
    _LOGGER.info(
        "Loaded checkpoint %s in policy_init mode: restored policy/trunk/obs_scaler, "
        "kept current value head, value scaler, and optimizer initialization.",
        checkpoint_path,
    )


def _resume_bundle_config_path(resume_path: str | Path) -> Path:
    resolved = Path(resume_path).expanduser().resolve()
    return (
        resolved.parent.parent / "run_bundle" / "config" / "train_config_resolved.yaml"
    )


def _env_cfg_nested_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    current = cfg
    for key in keys:
        current = get_value(current, key, None)
        if current is None:
            return default
    return current


def _uses_v1_env_schema(env_cfg: Any) -> bool:
    return get_value(env_cfg, "action", None) is not None


def _policy_observation_contract(env_cfg: Any) -> dict[str, Any]:
    if _uses_v1_env_schema(env_cfg):
        previous_action = _env_cfg_nested_get(
            env_cfg, "observations", "previous_action", default="action"
        )
        return {
            "action_space": int(
                _env_cfg_nested_get(env_cfg, "action", "space", default=0)
            ),
            "include_prev_actions_obs": previous_action is not None,
            "observe_applied_torque": bool(
                _env_cfg_nested_get(
                    env_cfg, "observations", "applied_torque", default=False
                )
            ),
        }
    return {
        "action_space": int(get_value(env_cfg, "action_space", 0)),
        "include_prev_actions_obs": bool(
            get_value(env_cfg, "include_prev_actions_obs", True)
        ),
        "observe_applied_torque": bool(
            get_value(env_cfg, "observe_applied_torque", False)
        ),
    }


def _validate_musclenn_only_resume_env_semantics(
    *,
    cfg: Any,
    resume_path: str | Path,
) -> None:
    bundle_cfg_path = _resume_bundle_config_path(resume_path)
    if not bundle_cfg_path.is_file():
        _LOGGER.warning(
            "Skipping musclenn_only env semantics validation because bundled config is missing: %s",
            bundle_cfg_path,
        )
        return

    resume_cfg = OmegaConf.load(bundle_cfg_path)
    current_env_cfg = getattr(cfg, "env", None)
    resume_env_cfg = getattr(resume_cfg, "env", None)
    current_policy_contract = _policy_observation_contract(current_env_cfg)
    resume_policy_contract = _policy_observation_contract(resume_env_cfg)
    for key, current_value in current_policy_contract.items():
        resume_value = resume_policy_contract[key]
        if current_value != resume_value:
            raise ValueError(
                "trainer.training_stage='musclenn_only' requires a checkpoint-compatible "
                "policy observation contract "
                f"for {key}: current={current_value!r}, checkpoint={resume_value!r} "
                f"(from {bundle_cfg_path})."
            )
