"""Rollout collection runtime for PPO training."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

import torch
import warp as wp

from instantmimic.envs.direct_rl_env import (
    StepInputs,
    StepIO,
    StepOutputs,
    external_step_capture_window,
)
from instantmimic.rl.agent_protocol import (
    CudaGraphCapturePreparationAgent,
    ObservationNormalizationAgent,
    TrainingAgent,
)
from instantmimic.rl.runtime import (
    agent_observation_output_views,
    copy_agent_observation_into,
)
from instantmimic.utils.torch_warp_capture import (
    capture_warp_graph,
    warp_stream_from_torch,
)

_LOGGER = logging.getLogger(__name__)
_TrainFullVideoPrepareStep = Callable[[int], None]
_TrainFullVideoCaptureStep = Callable[[int], None]


def _use_external_step_capture_window(
    *,
    use_torch_act_graph: bool,
    device: torch.device,
) -> bool:
    return bool(use_torch_act_graph and device.type == "cuda")


@wp.kernel(enable_backward=False)
def _update_episode_stats_kernel(
    rewards: wp.array(dtype=wp.float32),
    dones: wp.array(dtype=wp.bool),
    ep_reward_sum: wp.array(dtype=wp.float32),
    ep_timestep_sum: wp.array(dtype=wp.float32),
    ep_finished_reward: wp.array(dtype=wp.float32),
    ep_finished_timesteps: wp.array(dtype=wp.float32),
    ep_finished_count: wp.array(dtype=wp.int64),
):
    i = wp.tid()

    ep_reward_sum[i] = ep_reward_sum[i] + rewards[i]
    ep_timestep_sum[i] = ep_timestep_sum[i] + 1.0

    if dones[i]:
        ep_finished_reward[i] = ep_finished_reward[i] + ep_reward_sum[i]
        ep_finished_timesteps[i] = ep_finished_timesteps[i] + ep_timestep_sum[i]
        ep_finished_count[i] = ep_finished_count[i] + wp.int64(1)
        ep_reward_sum[i] = 0.0
        ep_timestep_sum[i] = 0.0


@dataclass(frozen=True)
class EpisodeTotals:
    count: int
    reward_total: float
    timestep_total: float

    def __iter__(self):
        yield self.count
        yield self.reward_total
        yield self.timestep_total


class EpisodeStatsAccumulator:
    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device,
    ) -> None:
        self.reward_sum = torch.zeros(num_envs, device=device)
        self.timestep_sum = torch.zeros(num_envs, device=device)
        self.finished_count = torch.zeros(num_envs, device=device, dtype=torch.int64)
        self.finished_reward = torch.zeros(num_envs, device=device)
        self.finished_timesteps = torch.zeros(num_envs, device=device)
        self.totals = torch.zeros(3, device=device, dtype=torch.float32)
        self.wp_device = wp.get_device(str(device))
        self.reward_sum_wp = wp.from_torch(self.reward_sum, dtype=wp.float32)
        self.timestep_sum_wp = wp.from_torch(self.timestep_sum, dtype=wp.float32)
        self.finished_count_wp = wp.from_torch(self.finished_count, dtype=wp.int64)
        self.finished_reward_wp = wp.from_torch(self.finished_reward, dtype=wp.float32)
        self.finished_timesteps_wp = wp.from_torch(
            self.finished_timesteps, dtype=wp.float32
        )

    def launch_update(
        self,
        *,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        num_envs: int,
    ) -> None:
        wp.launch(
            _update_episode_stats_kernel,
            dim=num_envs,
            inputs=[
                rewards,
                dones,
                self.reward_sum_wp,
                self.timestep_sum_wp,
                self.finished_reward_wp,
                self.finished_timesteps_wp,
                self.finished_count_wp,
            ],
            device=self.wp_device,
        )

    def consume_totals(self) -> EpisodeTotals:
        self.totals[0].copy_(self.finished_count.sum(dtype=torch.float32))
        self.totals[1].copy_(self.finished_reward.sum())
        self.totals[2].copy_(self.finished_timesteps.sum())
        totals_cpu = self.totals.cpu()
        self.finished_reward.zero_()
        self.finished_timesteps.zero_()
        self.finished_count.zero_()
        return EpisodeTotals(
            count=int(totals_cpu[0].item()),
            reward_total=float(totals_cpu[1].item()),
            timestep_total=float(totals_cpu[2].item()),
        )


class RolloutTiming:
    def __init__(
        self,
        *,
        device: torch.device,
        rollouts: int,
        env_step_timing_enabled: bool,
    ) -> None:
        self.device = device
        self.rollouts = int(rollouts)
        self.env_step_timing_enabled = bool(env_step_timing_enabled)
        self.env_step_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.env_step_cpu_fallback_ns: list[int] = []
        self.env_step_cpu_total_ns = 0
        self.env_step_rollout_estimate_ns_acc = 0
        self.env_step_rollout_estimate_count = 0
        self.rollout_start_event = (
            torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        )
        self.rollout_end_event = (
            torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        )
        self.rollout_timing_pending = False
        self.rollout_wall_start_ns: int | None = None
        self.rollout_wall_elapsed_ns = 0

    def record_rollout_start(self, stream: torch.cuda.Stream) -> None:
        if self.rollout_start_event is None:
            self.rollout_wall_start_ns = perf_counter_ns()
            self.rollout_timing_pending = True
            return
        self.rollout_start_event.record(stream)
        self.rollout_timing_pending = True

    def record_rollout_end(self, stream: torch.cuda.Stream) -> None:
        if self.rollout_end_event is None:
            assert self.rollout_wall_start_ns is not None
            self.rollout_wall_elapsed_ns = (
                perf_counter_ns() - self.rollout_wall_start_ns
            )
            return
        self.rollout_end_event.record(stream)

    def consume_rollout_timing_ns(self) -> int:
        if (
            not self.rollout_timing_pending
            or self.rollout_start_event is None
            or self.rollout_end_event is None
        ):
            if not self.rollout_timing_pending:
                return 0
            self.rollout_timing_pending = False
            self.rollout_wall_start_ns = None
            elapsed_ns = self.rollout_wall_elapsed_ns
            self.rollout_wall_elapsed_ns = 0
            return elapsed_ns
        self.rollout_timing_pending = False
        self.rollout_end_event.synchronize()
        return int(self.rollout_start_event.elapsed_time(self.rollout_end_event) * 1e6)

    def record_env_step_start(self, stream: torch.cuda.Stream | None):
        if not self.env_step_timing_enabled:
            return None
        cpu_start_ns = perf_counter_ns()
        if self.device.type != "cuda" or stream is None:
            return None, None, cpu_start_ns
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        return start_event, end_event, cpu_start_ns

    def record_env_step_end(
        self,
        token: tuple[torch.cuda.Event | None, torch.cuda.Event | None, int] | None,
        stream: torch.cuda.Stream | None,
    ) -> None:
        if token is None:
            return
        start_event, end_event, cpu_start_ns = token
        cpu_elapsed_ns = perf_counter_ns() - cpu_start_ns
        self.env_step_cpu_total_ns += cpu_elapsed_ns
        if start_event is None or end_event is None or stream is None:
            self.env_step_cpu_fallback_ns.append(cpu_elapsed_ns)
            return
        end_event.record(stream)
        self.env_step_event_pairs.append((start_event, end_event))

    def add_rollout_estimate(self, rollout_elapsed_ns: int) -> None:
        if not self.env_step_timing_enabled or rollout_elapsed_ns <= 0:
            return
        self.env_step_rollout_estimate_ns_acc += int(rollout_elapsed_ns)
        self.env_step_rollout_estimate_count += 1

    def log_env_step_timing_marker(self, iterations: int, *, num_envs: int) -> None:
        if not self.env_step_timing_enabled:
            return
        source = "cuda_event_env_step"
        estimated = False
        if self.env_step_event_pairs:
            self.env_step_event_pairs[-1][1].synchronize()
            values_ms = [
                float(start.elapsed_time(end))
                for start, end in self.env_step_event_pairs
            ]
        elif self.env_step_cpu_fallback_ns:
            source = "cpu_wall_env_step"
            values_ms = [value * 1e-6 for value in self.env_step_cpu_fallback_ns]
        elif self.env_step_rollout_estimate_count > 0:
            source = "cuda_event_rollout_body_step_estimate"
            estimated = True
            steps = int(self.env_step_rollout_estimate_count * self.rollouts)
            mean_ms = (self.env_step_rollout_estimate_ns_acc * 1e-6) / max(steps, 1)
            _LOGGER.info(
                "ENV_STEP_TIMING source=%s estimated=%s steps=%d num_envs=%d rollouts=%d iterations=%d total_ms=%.6f mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f cpu_total_ms=%.6f cpu_mean_ms=%.6f",
                source,
                str(estimated).lower(),
                steps,
                num_envs,
                self.rollouts,
                iterations,
                self.env_step_rollout_estimate_ns_acc * 1e-6,
                mean_ms,
                mean_ms,
                mean_ms,
                0.0,
                0.0,
            )
            return
        else:
            return

        values_ms.sort()
        total_ms = sum(values_ms)
        steps = len(values_ms)
        _LOGGER.info(
            "ENV_STEP_TIMING source=%s estimated=%s steps=%d num_envs=%d rollouts=%d iterations=%d total_ms=%.6f mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f cpu_total_ms=%.6f cpu_mean_ms=%.6f",
            source,
            str(estimated).lower(),
            steps,
            num_envs,
            self.rollouts,
            iterations,
            total_ms,
            total_ms / max(steps, 1),
            _percentile_from_sorted_ms(values_ms, 50.0),
            _percentile_from_sorted_ms(values_ms, 95.0),
            self.env_step_cpu_total_ns * 1e-6,
            (self.env_step_cpu_total_ns * 1e-6) / max(steps, 1),
        )


class RolloutRuntime:
    def __init__(
        self,
        *,
        env: Any,
        env_raw: Any,
        env_raw_for_agent: Any,
        agent: TrainingAgent,
        device: torch.device,
        num_envs: int,
        rollouts: int,
        nan_debug_enabled: bool,
        use_torch_act_graph: bool,
        detailed_nsys_nvtx: bool,
        rl_torch_stream: torch.cuda.Stream | None,
        time_limit_bootstrap: bool,
        train_full_video_enabled: bool = False,
        train_full_video_prepare_step: _TrainFullVideoPrepareStep | None = None,
        train_full_video_capture_step: _TrainFullVideoCaptureStep | None = None,
    ) -> None:
        self.env = env
        self.env_raw = env_raw
        self.env_raw_for_agent = env_raw_for_agent
        self.agent = agent
        self._cuda_graph_capture_agent = (
            agent if isinstance(agent, CudaGraphCapturePreparationAgent) else None
        )
        self._observation_normalization_agent = (
            agent if isinstance(agent, ObservationNormalizationAgent) else None
        )
        self.device = device
        self.num_envs = int(num_envs)
        self.rollouts = int(rollouts)
        self.time_limit_bootstrap = bool(time_limit_bootstrap)
        self.use_torch_act_graph = bool(use_torch_act_graph)
        self.detailed_nsys_nvtx = bool(detailed_nsys_nvtx)
        self.train_full_video_prepare_step = train_full_video_prepare_step
        self.train_full_video_capture_step = train_full_video_capture_step
        self.buf = agent.buffer
        initial_state_spec_source = getattr(
            env_raw_for_agent, "get_initial_state_spec", None
        )
        initial_state_enabled = bool(
            callable(initial_state_spec_source)
            and initial_state_spec_source() is not None
        )
        self._initial_state_metadata_source = (
            getattr(env_raw_for_agent, "initial_state_sample_metadata", None)
            if initial_state_enabled
            else None
        )
        self._clear_initial_state_pending = (
            getattr(env_raw_for_agent, "clear_initial_state_pending", None)
            if initial_state_enabled
            else None
        )
        if initial_state_enabled and (
            not callable(self._initial_state_metadata_source)
            or not callable(self._clear_initial_state_pending)
            or not callable(getattr(agent, "store_initial_state_samples", None))
        ):
            raise RuntimeError(
                "learned initial states require matching environment and agent hooks"
            )
        requires_policy_scores = getattr(
            env_raw_for_agent,
            "adaptive_sampling_requires_policy_scores",
            None,
        )
        self._adaptive_level_id_source = getattr(
            env_raw_for_agent,
            "adaptive_sampling_episode_start_unit_indices",
            None,
        )
        self.adaptive_sampling_level_ids = (
            torch.empty(
                (self.rollouts, self.num_envs),
                device=device,
                dtype=torch.long,
            )
            if callable(requires_policy_scores) and bool(requires_policy_scores())
            else None
        )
        if self.adaptive_sampling_level_ids is not None and not callable(
            self._adaptive_level_id_source
        ):
            raise RuntimeError(
                "motion PLR requires adaptive_sampling_episode_start_unit_indices()"
            )
        self._step_terminated = torch.empty(
            (self.rollouts, self.num_envs),
            device=device,
            dtype=torch.bool,
        )
        self._step_truncated = torch.empty_like(self._step_terminated)
        self._rollout_step_io_cache: list[StepIO | None] = [None] * self.rollouts
        self.rollout_env_use_cuda_graph = _configure_rollout_env_cuda_graph_mode(
            env_raw,
            nan_debug_enabled=nan_debug_enabled,
            train_full_video_enabled=bool(train_full_video_enabled),
        )
        self.external_step_capture_window_enabled = _use_external_step_capture_window(
            use_torch_act_graph=self.use_torch_act_graph,
            device=device,
        )
        self.stats = EpisodeStatsAccumulator(num_envs=self.num_envs, device=device)
        self.timing = RolloutTiming(
            device=device,
            rollouts=self.rollouts,
            env_step_timing_enabled=self.detailed_nsys_nvtx,
        )
        self._prepare_terminal_observation_buffer()
        self.use_step_rollout_graphs = bool(
            self.detailed_nsys_nvtx
            and self.use_torch_act_graph
            and device.type == "cuda"
        )
        self.rollout_graph: Any | None = None
        self.rollout_step_graphs: list[Any | None] | None = (
            [None] * self.rollouts if self.use_step_rollout_graphs else None
        )
        self.wp_stream = None
        self.torch_stream = None
        self._setup_graph_streams(rl_torch_stream)

    @property
    def env_step_timing_enabled(self) -> bool:
        return self.timing.env_step_timing_enabled

    def _setup_graph_streams(self, rl_torch_stream: torch.cuda.Stream | None) -> None:
        if not self.use_torch_act_graph:
            return
        if self.device.type != "cuda":
            _LOGGER.warning(
                "runtime.act_cuda_graph=true requested, but CUDA is unavailable; using eager act_into."
            )
            return
        assert rl_torch_stream is not None
        self.torch_stream = rl_torch_stream
        self.wp_stream = warp_stream_from_torch(self.torch_stream)
        if self.use_step_rollout_graphs:
            _LOGGER.info(
                "Detailed Nsight mode: will capture %d rollout step graphs on first rollout "
                "(act_into + step_with_io + episode stats for each step slot).",
                self.rollouts,
            )
        else:
            _LOGGER.info(
                "Will capture 1 fused rollout graph on first rollout "
                "(act_into + step_with_io + episode stats across %d steps).",
                self.rollouts,
            )

    def consume_episode_totals(self) -> EpisodeTotals:
        return self.stats.consume_totals()

    def consume_rollout_timing_ns(self) -> int:
        return self.timing.consume_rollout_timing_ns()

    def add_env_step_rollout_estimate(self, rollout_elapsed_ns: int) -> None:
        self.timing.add_rollout_estimate(rollout_elapsed_ns)

    def log_env_step_timing_marker(self, iterations: int) -> None:
        self.timing.log_env_step_timing_marker(iterations, num_envs=self.num_envs)

    def invalidate_graphs(self) -> None:
        self.rollout_graph = None
        self._rollout_step_io_cache = [None] * self.rollouts
        if self.rollout_step_graphs is not None:
            for graph_idx in range(len(self.rollout_step_graphs)):
                self.rollout_step_graphs[graph_idx] = None

    def prewarm(self) -> None:
        context = (
            external_step_capture_window(self.env_raw)
            if self.external_step_capture_window_enabled
            else nullcontext()
        )
        with context:
            self._rollout_body(prewarm_only=True)

    def collect(self) -> None:
        context = (
            external_step_capture_window(self.env_raw)
            if self.external_step_capture_window_enabled
            else nullcontext()
        )
        with context:
            self._rollout_body()
        self._mark_musclenn_rollout_collected()

    def _prepare_external_step_capture_window(self) -> None:
        if not self.external_step_capture_window_enabled:
            return
        prepare = getattr(self.env_raw, "prepare_step_capture_window", None)
        if callable(prepare):
            prepare()

    def _finalize_external_step_capture_window(self) -> None:
        if not self.external_step_capture_window_enabled:
            return
        finalize = getattr(self.env_raw, "finalize_step_capture_window", None)
        if callable(finalize):
            finalize()

    def _nvtx_range(self, name: str):
        if not self.detailed_nsys_nvtx:
            return nullcontext()
        return torch.cuda.nvtx.range(name)

    def _rollout_actions_for_step(self, step: int) -> Any:
        adaptive_level_ids = getattr(self, "adaptive_sampling_level_ids", None)
        if adaptive_level_ids is not None:
            assert callable(self._adaptive_level_id_source)
            adaptive_level_ids[step].copy_(self._adaptive_level_id_source())
        initial_state_metadata = (
            initial_state_metadata_source()
            if callable(
                initial_state_metadata_source := getattr(
                    self, "_initial_state_metadata_source", None
                )
            )
            else None
        )
        if initial_state_metadata is not None:
            (
                start_mask,
                samples,
                component_ids,
                old_log_prob,
                versions,
            ) = initial_state_metadata
            self.agent.store_initial_state_samples(
                step=step,
                mask=start_mask,
                samples=samples,
                component_ids=component_ids,
                old_log_prob=old_log_prob,
                versions=versions,
            )
            assert callable(self._clear_initial_state_pending)
            self._clear_initial_state_pending()
        self.agent.act_into(step)
        return self._resolve_rollout_action_binding(step)

    def _resolve_rollout_action_binding(self, step: int) -> torch.Tensor:
        return self.buf.actions[step]

    @staticmethod
    def _action_binding_key(actions: torch.Tensor) -> tuple[Any, ...]:
        return (
            actions.data_ptr(),
            actions.untyped_storage().data_ptr(),
            actions.storage_offset(),
            tuple(actions.shape),
            tuple(actions.stride()),
            actions.dtype,
            actions.device,
        )

    def _prewarm_rollout_step_io_cache(self) -> None:
        for step in range(self.rollouts):
            rollout_actions = self._resolve_rollout_action_binding(step)
            cached = self._rollout_step_io_cache[step]
            if cached is not None:
                if self._action_binding_key(
                    cached.inputs.actions
                ) != self._action_binding_key(rollout_actions):
                    self.invalidate_graphs()
                    raise RuntimeError(
                        "Rollout action storage changed after StepIO prewarm; "
                        "prewarm the rollout runtime again before graph capture."
                    )
                continue
            self._rollout_step_io_cache[step] = self._make_step_io(
                step,
                rollout_actions,
            )

    def _cached_rollout_step_io(self, step: int) -> StepIO:
        step_io = self._rollout_step_io_cache[step]
        if step_io is None:
            raise RuntimeError(
                "Rollout StepIO cache is not prewarmed before graph capture."
            )
        return step_io

    def _run_rollout_step_graph_ops(self, step: int) -> None:
        self._set_musclenn_rollout_step(step)
        with self._nvtx_range("rollout_policy"):
            self._rollout_actions_for_step(step)
        self._prepare_train_full_video_step(step)
        with self._nvtx_range("rollout_env_step"):
            outputs, extras = self.env_raw.step_with_io(
                self._cached_rollout_step_io(step),
                use_cuda_graph=False,
            )
            self.buf.dones[step, :, 0].copy_(self.env_raw.reset_buf)
            self._copy_terminal_observation(step)
            self._update_observation_normalization(self.buf.obs[step + 1])
        self._capture_train_full_video_step(step)
        self._maybe_apply_time_limit_bootstrap(
            step,
            fallback_time_outs=None,
        )
        with self._nvtx_range("rollout_episode_stats"):
            self.stats.launch_update(
                rewards=outputs.rewards,
                dones=self.env_raw.reset_buf,
                num_envs=self.num_envs,
            )

    def _nvtx_range(self, name: str):
        """A range only when the run asked for detailed NVTX.

        `_rollout_body` carried an unconditional `@torch.cuda.nvtx.range`
        decorator, so every rollout pushed and popped a range whether or not
        anyone was profiling, with no way to turn it off. This class already
        gates its other profiling on `detailed_nsys_nvtx`.
        """

        # `getattr` because tests build this runtime partially, the way the
        # playback fixtures do; a runtime that never set the flag is not
        # profiling, so no range is the right answer rather than an
        # AttributeError from inside the rollout body.
        if not getattr(self, "detailed_nsys_nvtx", False):
            return contextlib.nullcontext()
        return torch.cuda.nvtx.range(name)

    def _update_observation_normalization(self, obs: torch.Tensor) -> None:
        agent = self._observation_normalization_agent
        if agent is not None:
            agent.update_observation_normalization(obs)

    def _rollout_body(self, *, prewarm_only: bool = False) -> None:
        with self._nvtx_range("rollout_body"):
            self._rollout_body_inner(prewarm_only=prewarm_only)

    def _rollout_body_inner(self, *, prewarm_only: bool = False) -> None:
        self._prepare_musclenn_rollout_sampling()
        if self.use_torch_act_graph and any(
            step_io is None for step_io in self._rollout_step_io_cache
        ):
            self._prewarm_rollout_step_io_cache()

        if (
            self.use_step_rollout_graphs
            and self.rollout_step_graphs is not None
            and self.wp_stream is not None
            and self.torch_stream is not None
        ):
            if self._cuda_graph_capture_agent is not None:
                self._cuda_graph_capture_agent.prepare_cuda_libraries_for_graph_capture()
            self._run_step_graph_rollout(prewarm_only=prewarm_only)
            return

        if (
            self.use_torch_act_graph
            and self.rollout_graph is None
            and self.wp_stream is not None
            and self.torch_stream is not None
        ):
            if self._cuda_graph_capture_agent is not None:
                self._cuda_graph_capture_agent.prepare_cuda_libraries_for_graph_capture()
            self._capture_fused_rollout(prewarm_only=prewarm_only)
            return

        if prewarm_only:
            return

        if self.rollout_graph is not None:
            self._launch_fused_rollout()
            return

        self._run_eager_rollout()

    def _run_step_graph_rollout(self, *, prewarm_only: bool) -> None:
        assert self.rollout_step_graphs is not None
        assert self.wp_stream is not None
        assert self.torch_stream is not None
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        captured_any_step_graph = False
        if not prewarm_only:
            self.timing.record_rollout_start(self.torch_stream)
        with wp.ScopedStream(self.wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self.torch_stream):
                for step in range(self.rollouts):
                    step_range_name = f"rollout_step_graph_{step}"
                    step_graph = self.rollout_step_graphs[step]
                    if step_graph is None:
                        with torch.cuda.nvtx.range(f"capture_{step_range_name}"):
                            with capture_warp_graph(stream=self.wp_stream) as capture:
                                self._run_rollout_step_graph_ops(step)
                        self.rollout_step_graphs[step] = capture.graph
                        captured_any_step_graph = True
                        if not prewarm_only:
                            with torch.cuda.nvtx.range(f"launch_{step_range_name}"):
                                wp.capture_launch(capture.graph, stream=self.wp_stream)
                                wp.synchronize_stream(self.wp_stream)
                    elif not prewarm_only:
                        with torch.cuda.nvtx.range(f"replay_{step_range_name}"):
                            wp.capture_launch(step_graph, stream=self.wp_stream)
                            wp.synchronize_stream(self.wp_stream)
        if prewarm_only:
            self.torch_stream.synchronize()
        else:
            self.timing.record_rollout_end(self.torch_stream)
            torch.cuda.current_stream(self.device).wait_stream(self.torch_stream)
        if captured_any_step_graph:
            _LOGGER.info("Captured rollout step graphs for detailed Nsight mode")

    def _capture_fused_rollout(self, *, prewarm_only: bool) -> None:
        assert self.wp_stream is not None
        assert self.torch_stream is not None
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        with wp.ScopedStream(self.wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self.torch_stream):
                with torch.cuda.nvtx.range("capture_rollout_graph"):
                    with capture_warp_graph(stream=self.wp_stream) as capture:
                        for step in range(self.rollouts):
                            self._run_rollout_step_graph_ops(step)
                if not prewarm_only:
                    self.timing.record_rollout_start(self.torch_stream)
                    with torch.cuda.nvtx.range("launch_rollout_graph"):
                        wp.capture_launch(capture.graph, stream=self.wp_stream)
                        if self.detailed_nsys_nvtx:
                            wp.synchronize_stream(self.wp_stream)
                    self.timing.record_rollout_end(self.torch_stream)
        self.rollout_graph = capture.graph
        if prewarm_only:
            self.torch_stream.synchronize()
        else:
            torch.cuda.current_stream(self.device).wait_stream(self.torch_stream)
        _LOGGER.info("Captured fused rollout graph")

    def _launch_fused_rollout(self) -> None:
        assert self.wp_stream is not None
        assert self.torch_stream is not None
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        self.timing.record_rollout_start(self.torch_stream)
        with torch.cuda.nvtx.range("replay_rollout_graph"):
            wp.capture_launch(self.rollout_graph, stream=self.wp_stream)
            if self.detailed_nsys_nvtx:
                wp.synchronize_stream(self.wp_stream)
        self.timing.record_rollout_end(self.torch_stream)
        torch.cuda.current_stream(self.device).wait_stream(self.torch_stream)

    def _run_eager_rollout(self) -> None:
        current_stream = torch.cuda.current_stream(self.device)
        self.timing.record_rollout_start(current_stream)
        with torch.cuda.nvtx.range("rollout_eager"):
            for step in range(self.rollouts):
                step_range_name = (
                    f"rollout_step_{step}"
                    if self.detailed_nsys_nvtx
                    else "rollout_step"
                )
                with self._nvtx_range(step_range_name):
                    self._set_musclenn_rollout_step(step)
                    with self._nvtx_range("rollout_policy"):
                        rollout_actions = self._rollout_actions_for_step(step)
                    self._run_step_with_io_env_step(
                        step=step,
                        rollout_actions=rollout_actions,
                        current_stream=current_stream,
                    )
        self.timing.record_rollout_end(current_stream)

    def _run_step_with_io_env_step(
        self,
        *,
        step: int,
        rollout_actions: Any,
        current_stream: torch.cuda.Stream,
    ) -> None:
        env_step_timing_token = self.timing.record_env_step_start(current_stream)
        self._prepare_train_full_video_step(step)
        with self._nvtx_range("rollout_env_step"):
            outputs, extras = self.env_raw.step_with_io(
                self._make_step_io(step, rollout_actions),
                use_cuda_graph=self.rollout_env_use_cuda_graph,
            )
            self.buf.dones[step, :, 0].copy_(self.env_raw.reset_buf)
            self._copy_terminal_observation(step)
            self._update_observation_normalization(self.buf.obs[step + 1])
        self.timing.record_env_step_end(env_step_timing_token, current_stream)
        self._capture_train_full_video_step(step)
        self._maybe_apply_time_limit_bootstrap(step, fallback_time_outs=None)
        self._launch_episode_stats(
            rewards_ref=outputs.rewards,
            dones_ref=self.env_raw.reset_buf,
        )

    def _make_step_io(self, step: int, rollout_actions: Any) -> StepIO:
        return StepIO(
            inputs=StepInputs(actions=rollout_actions),
            outputs=StepOutputs(
                observations=agent_observation_output_views(
                    agent=self.agent,
                    destination=self.buf.obs[step + 1],
                ),
                rewards=self.buf.rewards[step, :, 0],
                terminated=self._step_terminated[step],
                truncated=self._step_truncated[step],
            ),
        )

    def _prepare_train_full_video_step(self, step: int) -> None:
        prepare_step = getattr(self, "train_full_video_prepare_step", None)
        if callable(prepare_step):
            prepare_step(int(step))

    def _capture_train_full_video_step(self, step: int) -> None:
        capture_step = getattr(self, "train_full_video_capture_step", None)
        if callable(capture_step):
            capture_step(int(step))

    def _prepare_terminal_observation_buffer(self) -> None:
        self._terminal_bootstrap_observations: torch.Tensor | None = None
        terminal_obs = getattr(self.buf, "terminal_obs", None)
        agent_bootstrap = getattr(
            self.agent,
            "store_time_limit_bootstrap_values",
            None,
        )
        needs_terminal_obs = isinstance(terminal_obs, torch.Tensor) or (
            self.time_limit_bootstrap and callable(agent_bootstrap)
        )
        if not needs_terminal_obs:
            return
        prepare = getattr(self.env_raw, "prepare_terminal_observation_buffer", None)
        if callable(prepare):
            prepare()
        if (
            self.time_limit_bootstrap
            and callable(agent_bootstrap)
            and bool(getattr(self.agent, "asymmetric_observations", False))
        ):
            self._terminal_bootstrap_observations = torch.empty_like(self.buf.obs[0])
            copy_agent_observation_into(
                agent=self.agent,
                observations=self.env_raw.terminal_obs_buf,
                destination=self._terminal_bootstrap_observations,
            )

    def _copy_terminal_observation(self, step: int) -> None:
        destination = getattr(self.buf, "terminal_obs", None)
        source = getattr(self.env_raw, "terminal_obs_buf", None)
        if not isinstance(destination, torch.Tensor) or not isinstance(source, Mapping):
            return
        source_policy = source.get("policy")
        if not isinstance(source_policy, torch.Tensor):
            return
        destination[int(step)].copy_(source_policy)

    def _launch_episode_stats(
        self,
        *,
        rewards_ref: torch.Tensor,
        dones_ref: torch.Tensor,
    ) -> None:
        with self._nvtx_range("rollout_episode_stats"):
            self.stats.launch_update(
                rewards=rewards_ref,
                dones=dones_ref,
                num_envs=self.num_envs,
            )

    def _maybe_apply_time_limit_bootstrap(
        self,
        step: int,
        *,
        fallback_time_outs: torch.Tensor | None,
    ) -> None:
        if not self.time_limit_bootstrap:
            return
        with self._nvtx_range("rollout_time_limit_bootstrap"):
            timeout_mask = _resolve_time_limit_bootstrap_mask(
                self.env_raw,
                fallback_time_outs=fallback_time_outs,
            ).to(dtype=self.buf.rewards.dtype)
            store_terminated = getattr(self.agent, "store_terminated_mask", None)
            if callable(store_terminated):
                store_terminated(step, self._step_terminated[step])
            store_bootstrap_values = getattr(
                self.agent,
                "store_time_limit_bootstrap_values",
                None,
            )
            terminal_obs = getattr(self.env_raw, "terminal_obs_buf", None)
            terminal_policy_obs = (
                terminal_obs.get("policy")
                if isinstance(terminal_obs, Mapping)
                else None
            )
            terminal_bootstrap = getattr(self, "_terminal_bootstrap_observations", None)
            if terminal_bootstrap is not None and isinstance(terminal_obs, Mapping):
                copy_agent_observation_into(
                    agent=self.agent,
                    observations=terminal_obs,
                    destination=terminal_bootstrap,
                )
                terminal_policy_obs = terminal_bootstrap
            if (
                callable(store_bootstrap_values)
                and isinstance(terminal_policy_obs, torch.Tensor)
                and bool(
                    store_bootstrap_values(
                        step,
                        terminal_policy_obs,
                        timeout_mask,
                    )
                )
            ):
                return
            store_bootstrap = getattr(
                self.agent,
                "store_time_limit_bootstrap_mask",
                None,
            )
            if callable(store_bootstrap) and bool(store_bootstrap(step, timeout_mask)):
                return
            self.buf.rewards[step, :, 0].add_(
                float(self.agent.discount_factor)
                * self.buf.values[step, :, 0]
                * timeout_mask
            )

    def _set_musclenn_rollout_step(self, step: int) -> None:
        set_musclenn_rollout_step = getattr(
            self.env_raw_for_agent, "set_musclenn_rollout_step", None
        )
        if callable(set_musclenn_rollout_step):
            set_musclenn_rollout_step(step)

    def _prepare_musclenn_rollout_sampling(self) -> None:
        prepare_musclenn_rollout_sampling = getattr(
            self.env_raw_for_agent, "prepare_musclenn_rollout_sampling", None
        )
        if callable(prepare_musclenn_rollout_sampling):
            prepare_musclenn_rollout_sampling()

    def _mark_musclenn_rollout_collected(self) -> None:
        mark_musclenn_rollout_collected = getattr(
            self.env_raw_for_agent, "mark_musclenn_rollout_collected", None
        )
        if callable(mark_musclenn_rollout_collected):
            mark_musclenn_rollout_collected()


def _configure_rollout_env_cuda_graph_mode(
    env_raw: Any,
    *,
    nan_debug_enabled: bool,
    train_full_video_enabled: bool = False,
) -> bool | None:
    if not (nan_debug_enabled or train_full_video_enabled):
        return None

    reasons: list[str] = []
    if nan_debug_enabled:
        reasons.append("nan_debug finite checks")
    if train_full_video_enabled:
        reasons.append("full train video capture")
    _LOGGER.info(
        "Preserving env.use_cuda_graph for %s; graph/compile must stay enabled and any incompatibility should be fixed in the graph-safe path.",
        " + ".join(reasons),
    )
    return None


def _resolve_time_limit_bootstrap_mask(
    env_raw: Any,
    *,
    fallback_time_outs: torch.Tensor | None,
) -> torch.Tensor:
    bootstrap_time_outs = getattr(env_raw, "bootstrap_time_outs", None)
    if isinstance(bootstrap_time_outs, torch.Tensor):
        return bootstrap_time_outs
    if fallback_time_outs is not None:
        return fallback_time_outs
    return env_raw.reset_time_outs


def _percentile_from_sorted_ms(values_ms: list[float], pct: float) -> float:
    if not values_ms:
        return float("nan")
    if len(values_ms) == 1:
        return values_ms[0]
    pos = (len(values_ms) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(values_ms) - 1)
    frac = pos - lo
    return values_ms[lo] * (1.0 - frac) + values_ms[hi] * frac
