from __future__ import annotations

import logging
import math
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch
import warp as wp

from instantmimic.train.clip_failure_benchmark import (
    ClipFailureBenchmarkRunner,
    apply_clip_failure_benchmark_prior_to_env,
    build_clip_failure_benchmark_scalars,
    should_clone_shared_motion_tensors_for_clip_failure_benchmark,
    should_preserve_training_cuda_graphs_after_clip_failure_benchmark,
    should_rebuild_rollout_cuda_graph_after_clip_failure_benchmark,
    should_run_clip_failure_benchmark,
)
from instantmimic.train.metrics import (
    _ACTIVE_TRAIN_TIME_METRIC,
    _CHECKPOINT_TIME_METRIC,
    _TRAIN_FRAME_METRIC,
)

_LOGGER = logging.getLogger(__name__)


def _keep_aggregate_clip_failure_scalar(metric_name: str) -> bool:
    return (
        metric_name.startswith(("Clip Failure / Overall ", "Clip Failure / Adaptive "))
        or metric_name == "Clip Failure / Runtime (s)"
        or metric_name == "Clip Failure / Motion success rate"
        or metric_name
        in {_TRAIN_FRAME_METRIC, _ACTIVE_TRAIN_TIME_METRIC, _CHECKPOINT_TIME_METRIC}
    )


def _run_clip_failure_benchmark(
    *,
    config_name: str,
    checkpoint_path: Path,
    log_dir: Path,
    benchmark_cfg: Any,
    metrics_logger: Any,
    iteration_index: int,
    step_index: int,
    frames: int,
    active_train_time_s: float | None = None,
    checkpoint_time_s: float | None = None,
    env_raw: Any | None = None,
    runner: ClipFailureBenchmarkRunner | None = None,
) -> tuple[ClipFailureBenchmarkRunner, dict[str, object]]:
    if runner is None:
        runner = ClipFailureBenchmarkRunner(
            config_name=config_name,
            log_dir=log_dir,
            benchmark_cfg=benchmark_cfg,
            source_env_raw=env_raw,
        )
    record = runner.run(
        checkpoint_path=checkpoint_path,
        iteration_index=iteration_index,
        step_index=step_index,
        frames=frames,
        active_train_time_s=active_train_time_s,
        checkpoint_time_s=checkpoint_time_s,
    )
    record.setdefault("iteration", int(iteration_index))
    record.setdefault("step", int(step_index))
    record.setdefault("frames", int(frames))
    if active_train_time_s is not None:
        record["active_train_time_s"] = float(active_train_time_s)
    if checkpoint_time_s is not None:
        record["checkpoint_time_s"] = float(checkpoint_time_s)
    benchmark_prior_scalars: dict[str, float] = {}
    if env_raw is not None:
        try:
            benchmark_prior_scalars = (
                apply_clip_failure_benchmark_prior_to_env(
                    anchor_rows=getattr(runner, "last_anchor_rows", ()),
                    env_raw=env_raw,
                )
                or {}
            )
        except Exception:
            _LOGGER.exception(
                "Failed to apply clip failure benchmark prior at iteration %d.",
                iteration_index,
            )
    benchmark_scalars_all = build_clip_failure_benchmark_scalars(
        record,
        include_per_clip=False,
    )
    benchmark_scalars = {
        key: value
        for key, value in benchmark_scalars_all.items()
        if _keep_aggregate_clip_failure_scalar(key)
    }
    if benchmark_prior_scalars:
        benchmark_scalars.update(
            {
                "Clip Failure / Adaptive benchmark beta": float(
                    benchmark_prior_scalars["benchmark_beta"]
                ),
                "Clip Failure / Adaptive benchmark prior nonzero units": float(
                    benchmark_prior_scalars["benchmark_prior_nonzero_units"]
                ),
                "Clip Failure / Adaptive benchmark prior peak": float(
                    benchmark_prior_scalars["benchmark_prior_peak"]
                ),
            }
        )
    metrics_logger.log_scalars(
        benchmark_scalars,
        iteration_index,
        axis="iteration",
    )
    _LOGGER.info(
        "CLIP_FAILURE_BENCHMARK iteration=%d step=%d overall_failure=%.6f overall_survival=%.6f focus_gain=%.6f worst_clip=%s worst_clip_failure=%.6f runtime_s=%.3f output_root=%s",
        iteration_index,
        step_index,
        float(record["overall_failure_rate"]),
        float(record["overall_survival_ratio"]),
        float(record["failure_focus_gain"]),
        str(record["worst_clip_name"]),
        float(record["worst_clip_failure_rate"]),
        float(record["runtime_seconds"]),
        runner.output_root / f"iter_{iteration_index:06d}",
    )
    return runner, record


def _prewarm_clip_failure_benchmark_runner(
    *,
    runner: ClipFailureBenchmarkRunner | None,
    config_name: str,
    log_dir: Path,
    benchmark_cfg: Any,
    env_raw: Any,
    checkpoint_path: Path,
) -> ClipFailureBenchmarkRunner | None:
    if not should_preserve_training_cuda_graphs_after_clip_failure_benchmark(
        benchmark_cfg
    ):
        return runner

    if runner is None:
        runner = ClipFailureBenchmarkRunner(
            config_name=config_name,
            log_dir=log_dir,
            benchmark_cfg=benchmark_cfg,
            source_env_raw=env_raw,
        )
    runner.prewarm_graphs(checkpoint_path)
    return runner


def _cleanup_temporary_clip_failure_benchmark_checkpoint(
    checkpoint_path: Path,
    *,
    temporary: bool,
) -> None:
    if not temporary:
        return
    try:
        checkpoint_path.unlink(missing_ok=True)
    except OSError:
        _LOGGER.exception(
            "Failed to remove temporary clip failure benchmark checkpoint: %s",
            checkpoint_path,
        )
        return
    with suppress(OSError):
        checkpoint_path.parent.rmdir()


_MOTION_REFERENCE_POINTER_FIELDS = (
    "motion_qpos",
    "motion_qvel",
    "reference_body_state",
    "ref_com",
    "motion_qpos_mask",
    "motion_qvel_mask",
    "motion_body_mask",
    "loop_root_quat",
    "loop_root_pos",
    "motion_clip_start",
    "motion_clip_length",
    "motion_clip_reset_start_count",
    "motion_frame_to_clip_idx",
    "motion_clip_unit_start",
    "motion_clip_unit_count",
    "motion_clip_loop_root_quat",
    "motion_clip_loop_root_pos",
    "motion_xpos",
    "motion_xquat",
    "reference_body_pos",
    "reference_body_quat",
    "reference_body_linvel",
    "reward_body_idx",
    "reward_motion_qvel",
    "reward_reference_body_state",
    "reward_ref_com",
    "motion_ctrl",
)
_ADAPTIVE_SAMPLING_POINTER_FIELDS = (
    "dist",
    "cdf",
    "online_dist",
    "online_cdf",
    "runtime_dist_buffer",
    "runtime_cdf_buffer",
    "benchmark_prior",
    "benchmark_prior_enabled_tensor",
)


def _tensor_pointer_signature(value: torch.Tensor) -> dict[str, object]:
    return {
        "id": id(value),
        "data_ptr": int(value.data_ptr()),
        "shape": tuple(int(dim) for dim in value.shape),
        "stride": tuple(int(dim) for dim in value.stride()),
        "storage_offset": int(value.storage_offset()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _motion_reference_tensor_pointer_signature(
    env_raw: Any,
) -> dict[str, dict[str, object]]:
    signature: dict[str, dict[str, object]] = {}
    for field_name in _MOTION_REFERENCE_POINTER_FIELDS:
        value = getattr(env_raw, field_name, None)
        if isinstance(value, torch.Tensor):
            signature[field_name] = _tensor_pointer_signature(value)
    adaptive_state = getattr(env_raw, "adaptive_sampling_state", None)
    if adaptive_state is not None:
        for field_name in _ADAPTIVE_SAMPLING_POINTER_FIELDS:
            value = getattr(adaptive_state, field_name, None)
            if isinstance(value, torch.Tensor):
                signature[f"adaptive_sampling_state.{field_name}"] = (
                    _tensor_pointer_signature(value)
                )
    return signature


def _log_motion_reference_pointer_delta(
    before: dict[str, dict[str, object]] | None,
    after: dict[str, dict[str, object]] | None,
    *,
    context: str,
) -> None:
    if before is None or after is None:
        return
    changed_fields: dict[str, dict[str, object]] = {}
    for field_name in sorted(set(before) | set(after)):
        before_value = before.get(field_name)
        after_value = after.get(field_name)
        if before_value != after_value:
            changed_fields[field_name] = {
                "before": before_value,
                "after": after_value,
            }
    if changed_fields:
        _LOGGER.warning(
            "Motion/reference tensor pointer signatures changed during %s: %s",
            context,
            changed_fields,
        )
        return
    _LOGGER.info(
        "Motion/reference tensor pointer signatures unchanged during %s (%d tensors checked).",
        context,
        len(after),
    )


@dataclass(frozen=True)
class ClipFailureBenchmarkOutcome:
    ran: bool
    elapsed_ns: int = 0
    failed: bool = False
    best_updated: bool = False
    metric_value: float | None = None
    snapshot_iteration: int | None = None


class ClipFailureTrainingController:
    def __init__(
        self,
        *,
        config_name: str,
        log_dir: Path,
        benchmark_cfg: Any,
        env_raw: Any,
        device: torch.device | str,
        runner: ClipFailureBenchmarkRunner | None = None,
        select_best_checkpoint: bool = False,
    ) -> None:
        self.config_name = config_name
        self.log_dir = Path(log_dir)
        self.benchmark_cfg = benchmark_cfg
        self.env_raw = env_raw
        self.device = torch.device(device)
        self.runner = runner
        self.select_best_checkpoint = bool(select_best_checkpoint)
        self.best_metric_value: float | None = None
        self.best_iteration: int | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.benchmark_cfg.enabled)

    @property
    def preserve_training_graphs(self) -> bool:
        return should_preserve_training_cuda_graphs_after_clip_failure_benchmark(
            self.benchmark_cfg
        )

    def should_run(self, iteration_index: int) -> bool:
        return should_run_clip_failure_benchmark(
            iteration_index=iteration_index,
            config=self.benchmark_cfg,
        )

    def log_checkpoint_interval_notice(self, checkpoint_interval_rollouts: int) -> None:
        if not self.enabled or checkpoint_interval_rollouts <= 0:
            return
        if self.benchmark_cfg.interval % checkpoint_interval_rollouts == 0:
            return
        _LOGGER.info(
            "Clip failure benchmark interval=%d differs from checkpoint interval=%d; "
            "benchmark-only checkpoints will be saved on benchmark iterations.",
            self.benchmark_cfg.interval,
            checkpoint_interval_rollouts,
        )

    def prewarm(
        self,
        *,
        checkpoint_controller: Any,
        checkpoint_manager: Any,
        checkpoints_dir: Path,
    ) -> None:
        if not self.enabled or not self.preserve_training_graphs:
            return
        prewarm_checkpoint_path = (
            Path(checkpoints_dir)
            / "clip_failure_benchmark_tmp"
            / "agent_prewarm_initial.pt"
        )
        prewarm_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with torch.cuda.nvtx.range("save_checkpoint_for_clip_failure_prewarm"):
                checkpoint_controller.save(prewarm_checkpoint_path)
            checkpoint_controller.wait_for_update_boundary()
            with torch.cuda.nvtx.range("prewarm_clip_failure_benchmark"):
                self.runner = _prewarm_clip_failure_benchmark_runner(
                    runner=self.runner,
                    config_name=self.config_name,
                    log_dir=self.log_dir,
                    benchmark_cfg=self.benchmark_cfg,
                    env_raw=self.env_raw,
                    checkpoint_path=prewarm_checkpoint_path,
                )
            _LOGGER.info(
                "Prewarmed persistent clip failure benchmark runtime before training CUDA graph capture "
                "(shared_motion_clone_tensors=%s).",
                should_clone_shared_motion_tensors_for_clip_failure_benchmark(
                    self.benchmark_cfg
                ),
            )
        finally:
            _cleanup_temporary_clip_failure_benchmark_checkpoint(
                prewarm_checkpoint_path,
                temporary=True,
            )

    def run(
        self,
        *,
        checkpoint_controller: Any,
        checkpoint_manager: Any,
        rollout_runtime: Any,
        agent: Any,
        metrics_logger: Any,
        iteration_index: int,
        step_index: int,
        frames: int,
        checkpoint_due: bool,
        active_train_time_s: float | None,
        checkpoint_time_s: float | None,
    ) -> ClipFailureBenchmarkOutcome:
        if not self.enabled:
            return ClipFailureBenchmarkOutcome(ran=False)

        iteration_checkpoint_path = checkpoint_controller.iteration_path(
            iteration_index
        )
        checkpoint_temporary = False
        elapsed_ns = 0
        try:
            if not checkpoint_due:
                with torch.cuda.nvtx.range(
                    "save_checkpoint_for_clip_failure_benchmark"
                ):
                    iteration_checkpoint_path = (
                        checkpoint_controller.ensure_clip_failure_checkpoint(
                            iteration_index=iteration_index,
                            checkpoint_due=False,
                        )
                    )
                checkpoint_temporary = True
            checkpoint_controller.wait_for_update_boundary()
            self._sync_if_preserving_graphs()
            motion_pointer_signature_before = (
                _motion_reference_tensor_pointer_signature(self.env_raw)
                if self.preserve_training_graphs
                else None
            )

            benchmark_start_ns = perf_counter_ns()
            try:
                with torch.cuda.nvtx.range("clip_failure_benchmark"):
                    self.runner, benchmark_record = _run_clip_failure_benchmark(
                        config_name=self.config_name,
                        checkpoint_path=iteration_checkpoint_path,
                        log_dir=self.log_dir,
                        benchmark_cfg=self.benchmark_cfg,
                        metrics_logger=metrics_logger,
                        iteration_index=iteration_index,
                        step_index=step_index,
                        frames=frames,
                        active_train_time_s=active_train_time_s,
                        checkpoint_time_s=checkpoint_time_s,
                        env_raw=self.env_raw,
                        runner=self.runner,
                    )
            finally:
                elapsed_ns = perf_counter_ns() - benchmark_start_ns

            if self.preserve_training_graphs:
                self._sync_cuda_and_warp()
                _log_motion_reference_pointer_delta(
                    motion_pointer_signature_before,
                    _motion_reference_tensor_pointer_signature(self.env_raw),
                    context="shared-motion clip failure benchmark",
                )
                _LOGGER.info(
                    "Preserved training CUDA graphs after shared-motion clip failure benchmark."
                )
            if should_rebuild_rollout_cuda_graph_after_clip_failure_benchmark(
                self.benchmark_cfg
            ):
                self._sync_cuda_and_warp()
                rollout_runtime.invalidate_graphs()
                invalidate_agent_cuda_graphs = getattr(
                    agent, "invalidate_cuda_graphs", None
                )
                if callable(invalidate_agent_cuda_graphs):
                    invalidate_agent_cuda_graphs()
                _LOGGER.info(
                    "Invalidated training CUDA graphs after shared-motion clip failure benchmark; next rollout/update will recapture."
                )
            metric_value = float(
                benchmark_record.get("overall_fixed_horizon_reward_mean", math.nan)
            )
            best_updated = bool(
                self.select_best_checkpoint
                and math.isfinite(metric_value)
                and (
                    self.best_metric_value is None
                    or metric_value > self.best_metric_value
                )
            )
            if best_updated:
                checkpoint_controller.promote_checkpoint_file(iteration_checkpoint_path)
                self.best_metric_value = metric_value
                self.best_iteration = int(iteration_index)
                _LOGGER.info(
                    "CLIP_FAILURE_BEST_CHECKPOINT metric=fixed_horizon_reward_mean value=%.6f iteration=%d",
                    metric_value,
                    iteration_index,
                )
            if math.isfinite(metric_value):
                checkpoint_scalars = {
                    "Checkpoint / Candidate metric value": metric_value,
                }
                if self.best_metric_value is not None:
                    checkpoint_scalars["Checkpoint / Best metric value"] = (
                        self.best_metric_value
                    )
                metrics_logger.log_scalars(
                    checkpoint_scalars,
                    iteration_index,
                    axis="iteration",
                )
            return ClipFailureBenchmarkOutcome(
                ran=True,
                elapsed_ns=elapsed_ns,
                best_updated=best_updated,
                metric_value=(metric_value if math.isfinite(metric_value) else None),
                snapshot_iteration=int(iteration_index),
            )
        except Exception:
            _LOGGER.exception(
                "Clip failure benchmark failed at iteration=%d checkpoint=%s",
                iteration_index,
                iteration_checkpoint_path,
            )
            return ClipFailureBenchmarkOutcome(
                ran=True,
                elapsed_ns=elapsed_ns,
                failed=True,
            )
        finally:
            _cleanup_temporary_clip_failure_benchmark_checkpoint(
                iteration_checkpoint_path,
                temporary=checkpoint_temporary,
            )

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()

    def _sync_if_preserving_graphs(self) -> None:
        if self.preserve_training_graphs:
            self._sync_cuda_and_warp()

    def _sync_cuda_and_warp(self) -> None:
        if self.device.type != "cuda":
            return
        torch.cuda.synchronize(self.device)
        wp.synchronize()
