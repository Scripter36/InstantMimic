"""Shared PPO training runtime used by the train entrypoint."""

from __future__ import annotations

import logging
import math
import sys
from collections import deque
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import gymnasium as gym
import torch
from omegaconf import OmegaConf

from instantmimic.bootstrap import (
    configure_process,
    prepare_environment_runtime,
)
from instantmimic.config.artifacts import build_checkpoint_metadata
from instantmimic.config.runtime_specs import mutable_environment_config
from instantmimic.config.schema import SemanticConfig
from instantmimic.envs.eval_metrics import eval_episode_step_limit
from instantmimic.rl.agent_protocol import TrainingAgent
from instantmimic.rl.checkpointing import (
    CheckpointManager,
    normalize_checkpoint_load_mode,
)
from instantmimic.rl.factory import (
    build_agent_from_semantic_config,
    resolve_agent_class_spec,
    resolve_agent_observation_dims,
)
from instantmimic.rl.observation_noise import bind_training_observation_noise
from instantmimic.rl.runtime import (
    bind_initial_state_agent_to_env,
    copy_agent_observation_into,
    get_env_control_runtime_spec,
    initial_state_agent_kwargs,
    load_env_checkpoint_state,
    load_env_policy_init_state,
    resolve_agent_runtime_spec,
    set_seed,
    sync_agent_action_scales_from_env,
)
from instantmimic.rl.runtime_setup import runtime_setup_time_ns
from instantmimic.train import logging as train_logging
from instantmimic.train import repro_bundle as train_repro_bundle
from instantmimic.train.benchmark_runner import RolloutBenchmarkRunner
from instantmimic.train.checkpointing import (
    TrainingCheckpointController,
    _load_agent_checkpoint,
    _resolve_training_stage,
    _validate_musclenn_only_resume_env_semantics,
)
from instantmimic.train.clip_failure import ClipFailureTrainingController
from instantmimic.train.console import (
    TrainingConsoleState,
    compact_console_enabled,
    create_training_console,
    format_duration,
    format_eta,
    format_eval_event_message,
    install_compact_console_logging,
    print_run_header,
    print_startup_summary,
)
from instantmimic.train.eval import TrainingEvalController, TrainingEvalOutcome
from instantmimic.train.full_video_recorder import (
    TrainFullVideoConfig,
    TrainFullVideoRecorder,
)
from instantmimic.train.metrics import build_training_iteration_scalars
from instantmimic.train.objective import (
    TrainingObjectiveState,
    _active_elapsed_seconds,
)
from instantmimic.train.playback import PostTrainingPlaybackRunner
from instantmimic.train.rollout import RolloutRuntime
from instantmimic.train.run_plan import (
    TrainingOrchestrationPlan,
    TrainVideoPlan,
    create_semantic_experiment_context,
)
from instantmimic.train.update import (
    _configure_musclenn_update_runtime,
    _maybe_update_adaptive_sampling_after_rollout,
    _maybe_update_motion_plr_after_policy_update,
    _run_training_update_cycle,
)
from instantmimic.train.video_recorder import TrainIterationVideoRecorder
from instantmimic.utils.experiment_logging import resolve_hydra_config_name
from instantmimic.utils.startup_profiler import StartupProfiler

_LOGGER = logging.getLogger(__name__)
_DEFAULT_EPISODE_AVERAGE_WINDOW = 100
_DEFAULT_TRAINING_FPS_WINDOW = 10
_UNSET = object()


def _resolve_final_iteration_index(
    *,
    last_iteration_index: int,
    total_rollouts: int,
) -> int:
    return last_iteration_index if last_iteration_index > 0 else total_rollouts


def _reset_training_environment(*, env: Any, agent: Any) -> tuple[Any, dict]:
    """Reset simulator state and notify agents with reset-sensitive history."""

    observations, extras = env.reset()
    on_environment_reset = getattr(agent, "on_environment_reset", None)
    if callable(on_environment_reset):
        on_environment_reset()
    return observations, extras


def _load_environment_resume_state(
    *,
    env: Any,
    checkpoint_path: str | Path,
    checkpoint_load_mode: str,
) -> bool:
    """Restore environment state: all of it for a full resume, and for
    policy_init only the part of the policy's inference path the environment
    owns (the MASS MuscleNN)."""

    if normalize_checkpoint_load_mode(checkpoint_load_mode) != "full":
        return load_env_policy_init_state(env, checkpoint_path)
    return load_env_checkpoint_state(env, checkpoint_path)


def _drain_pending_eval_for_shutdown(
    *,
    eval_controller: Any,
    completion_iteration: int,
    checkpoint_time_ns_acc: int,
) -> Any | None:
    """Finish a launched eval without allowing eval failure to block shutdown."""

    try:
        return eval_controller.drain_pending(
            completion_iteration=completion_iteration,
            checkpoint_time_ns_acc=checkpoint_time_ns_acc,
        )
    except Exception:
        _LOGGER.exception(
            "Pending training eval failed while stopping at iteration=%d; "
            "continuing graceful shutdown.",
            completion_iteration,
        )
        return None


def _run_eval_at_training_boundary(
    *,
    eval_controller: TrainingEvalController,
    objective_state: TrainingObjectiveState,
    now_ns: int,
    iteration_index: int,
    total_rollouts: int,
    step_index: int,
    frames: int,
    active_elapsed_ns: int,
    checkpoint_time_ns_acc: int,
    recording_time_ns_acc: int,
    runtime_setup_time_ns_acc: int,
    best_candidate_enabled: bool,
) -> TrainingEvalOutcome:
    """Use one time sample for stopping and terminal policy evaluation."""

    objective_state.maybe_mark_timeout(
        now_ns=now_ns,
        benchmark_eval_time_ns_acc=eval_controller.eval_time_ns_acc,
        checkpoint_time_ns_acc=checkpoint_time_ns_acc,
        recording_time_ns_acc=recording_time_ns_acc,
        runtime_setup_time_ns_acc=runtime_setup_time_ns_acc,
    )
    outcome = eval_controller.run_if_due(
        iteration_index=iteration_index,
        # The regular final-iteration path also waits for overlapped evaluation
        # and promotes the exact evaluated checkpoint snapshot.
        total_rollouts=(
            iteration_index if objective_state.timed_out else total_rollouts
        ),
        step_index=step_index,
        frames=frames,
        active_elapsed_ns=active_elapsed_ns,
        checkpoint_time_ns_acc=checkpoint_time_ns_acc,
        best_candidate_enabled=best_candidate_enabled,
    )
    if outcome.threshold_reached:
        # A passing evaluation retains precedence over this boundary's timeout.
        objective_state.timed_out = False
        objective_state.timeout_elapsed_seconds = None
    return outcome


class _RollingEpisodeMean:
    def __init__(self, window_size: int) -> None:
        self.window_size = max(1, int(window_size))
        self.current_size = 0
        self.reward_mean = 0.0
        self.timestep_mean = 0.0

    def update(self, reward_total: float, timestep_total: float, count: int) -> None:
        if count <= 0:
            return
        size = min(int(count), self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        inv_count = 1.0 / float(count)
        batch_reward_mean = reward_total * inv_count
        batch_timestep_mean = timestep_total * inv_count
        self.reward_mean = (
            self.reward_mean * old_size + batch_reward_mean * size
        ) / size_sum
        self.timestep_mean = (
            self.timestep_mean * old_size + batch_timestep_mean * size
        ) / size_sum
        self.current_size = size_sum

    def get_means(self) -> tuple[float | None, float | None]:
        if self.current_size == 0:
            return None, None
        return self.reward_mean, self.timestep_mean


class _TrainingFpsWindow:
    """Measure throughput over recent completed iteration boundaries."""

    def __init__(self, window_size: int) -> None:
        self.window_size = max(1, int(window_size))
        self._boundaries: deque[tuple[int, float]] = deque(maxlen=self.window_size)

    def value(self, *, frames: int, active_seconds: float) -> float | None:
        if not self._boundaries:
            return None
        start_frames, start_seconds = self._boundaries[0]
        elapsed = float(active_seconds) - start_seconds
        frame_count = int(frames) - start_frames
        if elapsed <= 0.0 or frame_count <= 0:
            return None
        return float(frame_count) / elapsed

    def record(self, *, frames: int, active_seconds: float) -> None:
        boundary = (int(frames), float(active_seconds))
        if self._boundaries and self._boundaries[-1][0] == boundary[0]:
            self._boundaries[-1] = boundary
            return
        self._boundaries.append(boundary)


def _format_compact_count(value: int) -> str:
    """Human-friendly compact count (K/M/G suffix)."""
    units = ("", "K", "M", "G", "T", "P")
    x = float(value)
    unit_idx = 0
    while x >= 1000.0 and unit_idx < len(units) - 1:
        x /= 1000.0
        unit_idx += 1
    if unit_idx == 0:
        return str(int(x))
    if x >= 100.0:
        return f"{x:.0f}{units[unit_idx]}"
    if x >= 10.0:
        return f"{x:.1f}{units[unit_idx]}"
    return f"{x:.2f}{units[unit_idx]}"


def _format_elapsed_seconds(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600.0)
    minutes = int((total_seconds % 3600.0) // 60.0)
    secs = total_seconds % 60.0
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:04.1f}s"
    if minutes > 0:
        return f"{minutes}m {secs:04.1f}s"
    return f"{secs:.1f}s"


def _metric_float(
    payload: dict[str, float | int | bool] | None,
    key: str,
) -> float | None:
    if payload is None or key not in payload:
        return None
    value = payload[key]
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    return None


def _format_console_metric(value: float | None, *, precision: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


def _close_training_resources(
    *,
    checkpoint_manager: Any,
    metrics_logger: Any,
    benchmark_runner: Any | None,
    clip_failure_controller: Any,
    train_video_recorder: Any | None,
    train_full_video_recorder: Any | None,
    env: Any,
) -> BaseException | None:
    close_order = (
        ("train_full_video_recorder", train_full_video_recorder),
        ("train_video_recorder", train_video_recorder),
        ("checkpoint_manager", checkpoint_manager),
        ("metrics_logger", metrics_logger),
        ("benchmark_runner", benchmark_runner),
        ("clip_failure_controller", clip_failure_controller),
        ("env", env),
    )
    first_error: BaseException | None = None
    for name, resource in close_order:
        if resource is None:
            continue
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as exc:
            _LOGGER.exception("Failed to close %s.", name)
            if first_error is None:
                first_error = exc
    return first_error


def run_training(cfg: SemanticConfig) -> None:
    configure_process()
    startup_profiler = StartupProfiler(
        range_factory=lambda name: torch.cuda.nvtx.range(f"startup:{name}")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = int(cfg.env.scene.num_envs)

    with startup_profiler.phase("log_dir_setup"):
        config_name = resolve_hydra_config_name(default="default")
        experiment_context, execution_context = create_semantic_experiment_context(
            cfg,
            device=str(device),
            config_name=config_name,
            num_envs=num_envs,
        )
        config_name = experiment_context.config_name
        experiment_name = experiment_context.experiment_name
        log_dir = experiment_context.log_dir
        checkpoints_dir = experiment_context.checkpoints_dir
        logging_backend = experiment_context.logging_backend
        experiment_group = experiment_context.group
        wandb_name = experiment_context.wandb_name
        setting_name = getattr(experiment_context, "setting_name", None)

    with startup_profiler.phase("seed_setup"):
        env_cfg = mutable_environment_config(
            cfg,
            seed=execution_context.resolved_seed,
            log_dir=log_dir,
        )
        runtime_cfg = OmegaConf.create(cfg.to_dict())
        runtime_cfg.env = env_cfg
        set_seed(execution_context.resolved_seed)

    orchestration = TrainingOrchestrationPlan.from_semantic_config(cfg)
    training_schedule = orchestration.schedule
    best_checkpoint_plan = orchestration.best_checkpoint
    checkpoint_disabled = training_schedule.checkpoint_disabled

    console_enabled = compact_console_enabled(cfg.trainer)
    install_compact_console_logging(log_dir=log_dir, enabled=console_enabled)
    print_run_header(
        config_name=config_name,
        experiment_name=experiment_name,
        log_dir=log_dir,
        enabled=console_enabled,
    )

    play_after_train = "post_training_playback" in cfg.trainer
    if checkpoint_disabled and play_after_train:
        raise ValueError(
            "trainer.post_training_playback requires persistent checkpoints; "
            "set experiment.checkpoint.disabled=false."
        )
    post_train_playback_runner = PostTrainingPlaybackRunner(cfg)
    post_train_playback_checkpoint: Path | None = None
    train_video_recorder: TrainIterationVideoRecorder | None = None
    train_full_video_recorder: TrainFullVideoRecorder | None = None
    recording_time_ns_acc = 0

    def _measure_recording_call(callback: Any, /, *args: Any, **kwargs: Any) -> Any:
        nonlocal recording_time_ns_acc
        recording_start_ns = perf_counter_ns()
        try:
            return callback(*args, **kwargs)
        finally:
            recording_time_ns_acc += perf_counter_ns() - recording_start_ns

    with startup_profiler.phase("save_training_repro_bundle"):
        train_repro_bundle._save_training_repro_bundle(
            cfg, log_dir, resolved_seed=execution_context.resolved_seed
        )
    initialization = cfg.trainer.get("initialization")
    resume_path = (
        str(initialization["checkpoint"])
        if hasattr(initialization, "__getitem__")
        else None
    )
    train_video_plan = TrainVideoPlan.from_semantic_config(cfg)
    train_video_requested = train_video_plan.requested
    train_video_interval_iterations = train_video_plan.interval_iterations
    train_video_length = train_video_plan.length
    recording = cfg.trainer.get("recording", {})
    train_full_video_requested = bool(
        hasattr(recording, "__contains__") and "full_run" in recording
    )
    train_full_video_config = (
        TrainFullVideoConfig.from_mapping(recording["full_run"])
        if train_full_video_requested
        else None
    )
    # Full-rollout recording renders directly from the training environment's
    # batched simulation model. Ordinary training has no render consumer, so
    # DirectRLEnv may ask MuJoCo to discard physics-neutral visual assets.
    OmegaConf.update(
        env_cfg,
        "preserve_sim_visual_assets",
        bool(train_full_video_requested),
        force_add=True,
    )

    with startup_profiler.phase("gym_make_env"):
        prepare_environment_runtime()
        env = gym.make(cfg.task, cfg=env_cfg, render_mode=None)

    try:
        with startup_profiler.phase("resolve_agent_runtime_spec"):
            scale_control_spec = get_env_control_runtime_spec(env)
            runtime_spec = resolve_agent_runtime_spec(
                env,
                control_spec=scale_control_spec,
            )
    except BaseException:
        _close_training_resources(
            checkpoint_manager=None,
            metrics_logger=None,
            benchmark_runner=None,
            clip_failure_controller=None,
            train_video_recorder=train_video_recorder,
            train_full_video_recorder=train_full_video_recorder,
            env=env,
        )
        raise
    env_raw_for_agent = runtime_spec.env_raw
    action_mode = runtime_spec.action_mode
    training_stage = _resolve_training_stage(
        cfg=cfg,
        action_mode=action_mode,
        resume_path=resume_path,
    )
    if training_stage == "musclenn_only":
        _validate_musclenn_only_resume_env_semantics(
            cfg=runtime_cfg,
            resume_path=resume_path,
        )
    policy_action_scale_snapshot = scale_control_spec.policy_action_scale_snapshot
    squash_actions = runtime_spec.squash_actions
    clamp_mean = runtime_spec.clamp_mean
    env_action_dim = int(env_cfg.action_space)
    agent_action_dim = env_action_dim
    rollouts = int(cfg.agent.rollouts)
    timesteps = int(orchestration.schedule.max_iterations) * rollouts
    benchmark_plan = orchestration.benchmark
    clip_failure_controller = ClipFailureTrainingController(
        config_name=config_name,
        log_dir=log_dir,
        benchmark_cfg=benchmark_plan.clip_failure_or_disabled,
        env_raw=env_raw_for_agent,
        device=device,
        select_best_checkpoint=(
            best_checkpoint_plan.uses_clip_failure_benchmark and not checkpoint_disabled
        ),
    )
    eval_plan = benchmark_plan.rollout_or_disabled
    stopping_plan = orchestration.stopping
    reward_stop_threshold = (
        stopping_plan.threshold if stopping_plan.type == "reward" else None
    )
    episode_length_stop_threshold = (
        stopping_plan.threshold if stopping_plan.type == "episode_length" else None
    )
    fail_on_reward_threshold_miss = bool(
        stopping_plan.fail_on_miss and stopping_plan.type == "reward"
    )
    fail_on_episode_length_threshold_miss = bool(
        stopping_plan.fail_on_miss and stopping_plan.type == "episode_length"
    )
    objective_timeout_seconds = stopping_plan.timeout_seconds
    fail_on_benchmark_threshold_miss = bool(
        stopping_plan.fail_on_miss and stopping_plan.type == "benchmark"
    )
    objective_modes_enabled = sum(
        int(enabled)
        for enabled in (
            reward_stop_threshold is not None,
            episode_length_stop_threshold is not None,
            eval_plan.threshold_enabled,
        )
    )
    if objective_modes_enabled > 1:
        raise ValueError(
            "Configure only one training objective threshold at a time: reward, "
            "episode length, or benchmark eval."
        )
    objective_threshold_enabled = (
        reward_stop_threshold is not None
        or episode_length_stop_threshold is not None
        or eval_plan.threshold_enabled
    )
    objective_timer_enabled = (
        objective_threshold_enabled or stopping_plan.type == "time_budget"
    )
    episode_threshold_enabled = (
        reward_stop_threshold is not None or episode_length_stop_threshold is not None
    )
    objective_warmup_rollouts = stopping_plan.warmup_iterations
    if objective_warmup_rollouts < 0:
        objective_warmup_rollouts = 0
    if objective_timeout_seconds is not None and objective_timeout_seconds <= 0.0:
        objective_timeout_seconds = None
    runtime_plan = orchestration.runtime
    use_torch_act_graph = bool(runtime_plan.act_cuda_graph)
    use_torch_update_graph = bool(runtime_plan.update_cuda_graph)
    use_torch_shuffle_graph = bool(runtime_plan.shuffle_cuda_graph)
    nsys_nvtx_mode = str(runtime_plan.nsys_nvtx_mode or "launch").strip().lower()
    detailed_nsys_nvtx = nsys_nvtx_mode == "detailed"
    time_limit_bootstrap = bool(cfg.agent.get("time_limit_bootstrap", False))
    compile_requested = bool(runtime_plan.compile)
    compile_enabled = compile_requested
    compile_mode = str(runtime_plan.compile_mode or "max-autotune-no-cudagraphs")
    shuffle_mode = str(cfg.agent.get("shuffle_mode", "per_epoch"))
    nan_debug_cfg = env_cfg.get("nan_debug", {})
    nan_debug_enabled = bool(nan_debug_cfg.get("enabled", False))

    if nan_debug_enabled and use_torch_act_graph:
        _LOGGER.info(
            "env.nan_debug.enabled=true while runtime.act_cuda_graph=true; "
            "preserving CUDA graph capture and expecting the debug path to remain graph-safe."
        )

    if train_full_video_requested and use_torch_act_graph:
        _LOGGER.info(
            "trainer.recording.full_run will stream actual train rollout frames; "
            "preserving CUDA graph capture for rollout and adding graph-captured "
            "state-copy kernels for video frames."
        )

    if train_video_requested and use_torch_act_graph:
        _LOGGER.info(
            "video=true will use inline iteration playback recording; preserving CUDA "
            "graph capture for the main training rollout path."
        )

    compile_act_into = bool(
        cfg.runtime.get(
            "compile_act_into",
            compile_requested and not use_torch_act_graph,
        )
    )
    if compile_requested and use_torch_act_graph and compile_act_into:
        _LOGGER.info(
            "runtime.compile_act_into=true with rollout Warp graph capture can be unstable; "
            "keeping compiled act_into because it was explicitly requested."
        )
    elif compile_requested and use_torch_act_graph and not compile_act_into:
        _LOGGER.info(
            "runtime.act_cuda_graph=true: leaving act_into eager inside the Warp rollout graph."
        )

    if compile_requested and (use_torch_act_graph or use_torch_update_graph):
        _LOGGER.info(
            "runtime.compile=true with CUDA graph capture enabled "
            "(act_graph=%s, update_graph=%s, shuffle_graph=%s).",
            use_torch_act_graph,
            use_torch_update_graph,
            use_torch_shuffle_graph,
        )
    if detailed_nsys_nvtx:
        _LOGGER.info(
            "trainer.nsys_nvtx_mode=detailed: keeping CUDA graphs enabled and "
            "blocking inside NVTX launch ranges so Nsight Systems shows wall-time "
            "for rollout/PPO phases."
        )
    if use_torch_shuffle_graph and not use_torch_update_graph:
        _LOGGER.info(
            "Torch minibatch shuffle graph requested without captured PPO update graph; "
            "the agent will fall back to eager permutation refresh."
        )
    if reward_stop_threshold is not None:
        _LOGGER.info(
            "Reward threshold early-stop enabled: threshold=%.6f, fail_on_miss=%s, objective_warmup_rollouts=%d",
            reward_stop_threshold,
            fail_on_reward_threshold_miss,
            objective_warmup_rollouts,
        )
    if episode_length_stop_threshold is not None:
        _LOGGER.info(
            "Episode-length threshold early-stop enabled: threshold=%.3f, fail_on_miss=%s, objective_warmup_rollouts=%d",
            episode_length_stop_threshold,
            fail_on_episode_length_threshold_miss,
            objective_warmup_rollouts,
        )
    if eval_plan.enabled:
        _LOGGER.info(
            "Benchmark eval enabled: interval_iterations=%d, interval_seconds=%s, play_steps=%d, frame=%s, deterministic_reset=%s, early_termination=%s, require_full_horizon=%s, episode_semantics=%s, use_best_candidate=%s, best_metric=%s, save_history=%s, cuda_graph=%s, execution=%s",
            eval_plan.interval,
            (
                f"{eval_plan.interval_seconds:.6f}"
                if eval_plan.interval_seconds is not None
                else "None"
            ),
            eval_plan.play_steps,
            eval_plan.metric_frame,
            eval_plan.deterministic_reset,
            eval_plan.enable_early_termination,
            eval_plan.require_full_horizon,
            eval_plan.episode_semantics,
            eval_plan.use_best_candidate,
            eval_plan.best_metric or "off",
            eval_plan.save_history,
            eval_plan.cuda_graph,
            eval_plan.execution,
        )
    _LOGGER.info(
        "Best checkpoint selection: configured_source=%s source=%s benchmark_protocol=%s metric=%s direction=%s persistence=%s",
        best_checkpoint_plan.configured_source,
        best_checkpoint_plan.source,
        best_checkpoint_plan.benchmark_protocol or "none",
        best_checkpoint_plan.metric,
        best_checkpoint_plan.direction,
        "disabled" if checkpoint_disabled else "enabled",
    )
    if eval_plan.threshold_enabled:
        _LOGGER.info(
            "Benchmark threshold objective enabled: num_envs=%d ape_mean<=%s global_mpjpe_mean<=%s ee_mean<=%s success_rate>=%s consecutive_passes=%d fail_on_miss=%s objective_warmup_rollouts=%d",
            eval_plan.num_envs,
            (
                f"{eval_plan.ape_mean_threshold:.6f}"
                if eval_plan.ape_mean_threshold is not None
                else "None"
            ),
            (
                f"{eval_plan.global_mpjpe_mean_threshold:.6f}"
                if eval_plan.global_mpjpe_mean_threshold is not None
                else "None"
            ),
            (
                f"{eval_plan.ee_mean_threshold:.6f}"
                if eval_plan.ee_mean_threshold is not None
                else "None"
            ),
            (
                f"{eval_plan.success_rate_threshold:.6f}"
                if eval_plan.success_rate_threshold is not None
                else "None"
            ),
            eval_plan.consecutive_passes_required,
            fail_on_benchmark_threshold_miss,
            objective_warmup_rollouts,
        )
    if objective_timeout_seconds is not None and objective_timer_enabled:
        _LOGGER.info(
            "Active training time excluding benchmark eval and checkpoint time is capped at %.3fs.",
            objective_timeout_seconds,
        )
    if time_limit_bootstrap:
        _LOGGER.info(
            "time_limit_bootstrap enabled: truncated(time-out) steps are bootstrapped in GAE."
        )
    if training_stage == "musclenn_only":
        _LOGGER.info(
            "Training stage: musclenn_only. PPO policy/value updates are frozen; "
            "rollouts continue online and only env-side MuscleNN optimization runs."
        )
    else:
        _LOGGER.info("Training stage: full.")
    if cfg.agent["class"] == "PPO":
        if runtime_plan.ppo_precision is not None:
            _LOGGER.info(
                "PPO shared policy precision: rollout=%s update=%s",
                runtime_plan.ppo_precision,
                runtime_plan.ppo_precision,
            )
        else:
            _LOGGER.info(
                "PPO legacy policy precision: rollout=model default, "
                "update-only CUDA autocast=%s",
                runtime_plan.mixed_precision,
            )
    _LOGGER.info(
        "PPO minibatch shuffle mode: %s (supported: per_epoch, per_update, off).",
        shuffle_mode,
    )
    rl_torch_stream = (
        torch.cuda.Stream(device=device)
        if device.type == "cuda" and (use_torch_act_graph or use_torch_update_graph)
        else None
    )
    if rl_torch_stream is not None and use_torch_act_graph and use_torch_update_graph:
        _LOGGER.info(
            "Using one shared CUDA stream for rollout and PPO update graph launches."
        )

    with startup_profiler.phase("agent_factory_setup"):
        agent_spec = resolve_agent_class_spec(cfg)
    with startup_profiler.phase("agent_from_cfg"):
        agent_obs_dim, asymmetric_kwargs = resolve_agent_observation_dims(
            env_raw_for_agent,
            fallback_observation_space=env_cfg.observation_space,
        )
        agent_extra_kwargs = dict(asymmetric_kwargs)
        agent_extra_kwargs.update(
            initial_state_agent_kwargs(
                env_raw_for_agent,
                algorithm=agent_spec.algorithm,
            )
        )
        agent = build_agent_from_semantic_config(
            cfg,
            agent_spec,
            obs_dim=agent_obs_dim,
            act_dim=agent_action_dim,
            num_envs=num_envs,
            device=device,
            runtime_spec=runtime_spec,
            policy_action_scale_snapshot=policy_action_scale_snapshot,
            update_torch_stream=(rl_torch_stream if use_torch_update_graph else None),
            extra_kwargs=agent_extra_kwargs or None,
        )
    initial_state_bound = bind_initial_state_agent_to_env(
        agent=agent,
        env=env_raw_for_agent,
        training=True,
    )
    configure_musclenn_rollout_storage = getattr(
        env_raw_for_agent, "configure_musclenn_rollout_storage", None
    )
    if callable(configure_musclenn_rollout_storage):
        configure_musclenn_rollout_storage(rollouts)
    if policy_action_scale_snapshot is not None:
        mean_scale = policy_action_scale_snapshot.mean_scale
        std_scale = policy_action_scale_snapshot.std_scale
        _LOGGER.info(
            "Configured policy action scales from env: mean[min=%.6f max=%.6f] std[min=%.6f max=%.6f]",
            float(mean_scale.min().item()) if mean_scale is not None else 1.0,
            float(mean_scale.max().item()) if mean_scale is not None else 1.0,
            float(std_scale.min().item()) if std_scale is not None else 1.0,
            float(std_scale.max().item()) if std_scale is not None else 1.0,
        )
        _LOGGER.info(
            "Configured policy action squash mode from env: %s", squash_actions
        )
        if clamp_mean:
            _LOGGER.info("Configured policy mean clamp mode from env: %s", clamp_mean)

    if resume_path:
        assert hasattr(initialization, "__getitem__")
        checkpoint_load_mode = normalize_checkpoint_load_mode(
            "full" if initialization["type"] == "resume" else "policy_init"
        )
        _LOGGER.info(
            "Loading model checkpoint from: %s (mode=%s)",
            resume_path,
            checkpoint_load_mode,
        )
        _load_agent_checkpoint(
            agent,
            resume_path,
            checkpoint_load_mode=checkpoint_load_mode,
        )
        if _load_environment_resume_state(
            env=env,
            checkpoint_path=resume_path,
            checkpoint_load_mode=checkpoint_load_mode,
        ):
            _LOGGER.info("Restored env checkpoint state from: %s", resume_path)
        if sync_agent_action_scales_from_env(agent, env_raw_for_agent):
            _LOGGER.info(
                "Resynced agent action scales from the active environment contract."
            )

    # Training rollouts only: the benchmark, playback and video envs are never
    # bound and keep clean observations. Bind after the checkpoint so the
    # normaliser statistics the noise scales with are the loaded ones.
    bind_training_observation_noise(agent=agent, env=env_raw_for_agent)

    with startup_profiler.phase("checkpoint_manager_init"):
        checkpoint_manager = CheckpointManager(
            agent=agent,
            env=env,
            device=device,
            producer_torch_stream=rl_torch_stream if use_torch_update_graph else None,
            checkpoint_metadata=build_checkpoint_metadata(),
            save_replay=bool(cfg.experiment.get("checkpoint", {}).get("replay", False)),
        )
    checkpoint_controller = TrainingCheckpointController(
        checkpoint_manager=checkpoint_manager,
        checkpoints_dir=checkpoints_dir,
        persistence_disabled=checkpoint_disabled,
    )

    if train_video_requested:
        _LOGGER.info(
            "Recording train videos via inline playback: interval=%d iteration(s), "
            "length=%d step(s), play_use_cuda_graph=%s.",
            train_video_interval_iterations,
            train_video_length,
            True,
        )

        def _play_train_video_checkpoint(
            *,
            checkpoint_path: Path,
            log_dir: Path,
            play_video: bool,
            play_video_length: int,
            play_use_cuda_graph: bool,
            **_: Any,
        ) -> dict[str, str]:
            return post_train_playback_runner.play_checkpoint_video(
                checkpoint_path=checkpoint_path,
                log_dir=log_dir,
                play_video=play_video,
                play_video_length=play_video_length,
                play_use_cuda_graph=play_use_cuda_graph,
            )

        def _save_train_video_checkpoint(path: Path) -> None:
            checkpoint_controller.save(path)

        train_video_recorder = TrainIterationVideoRecorder(
            log_dir=log_dir,
            checkpoints_dir=checkpoints_dir,
            interval_iterations=train_video_interval_iterations,
            video_length=train_video_length,
            save_checkpoint_fn=_save_train_video_checkpoint,
            playback_video_fn=_play_train_video_checkpoint,
            play_use_cuda_graph=True,
        )

    total_rollouts = training_schedule.max_iterations
    write_interval = training_schedule.write_interval
    checkpoint_interval = training_schedule.checkpoint_interval
    episode_metrics_interval = training_schedule.episode_metrics_interval
    best_checkpoint_interval = training_schedule.best_checkpoint_interval
    best_checkpoint_event_suffix = (
        "" if checkpoint_disabled else "  queued best checkpoint"
    )
    buf = agent.buffer
    env_raw = env.unwrapped  # bypass gym wrappers for direct _step_body()

    with startup_profiler.phase("env_reset"):
        obs, _ = _reset_training_environment(env=env, agent=agent)
    copy_agent_observation_into(
        agent=agent,
        observations=obs,
        destination=buf.obs[0],
    )

    benchmark_runner: RolloutBenchmarkRunner | None = None
    if eval_plan.enabled:
        with startup_profiler.phase("rollout_benchmark_runner_init"):
            benchmark_runner = RolloutBenchmarkRunner(
                cfg=runtime_cfg,
                agent=agent,
                source_env=env,
                log_dir=log_dir,
                device=device,
                act_dim=env_action_dim,
                num_envs=eval_plan.num_envs,
                play_steps=eval_plan.play_steps,
                frame=eval_plan.metric_frame,
                include_root_in_ape=eval_plan.include_root_in_ape,
                deterministic_reset=eval_plan.deterministic_reset,
                enable_early_termination=eval_plan.enable_early_termination,
                early_termination_mode=eval_plan.early_termination_mode,
                phc_termination_distance=eval_plan.phc_termination_distance,
                require_full_horizon=eval_plan.require_full_horizon,
                episode_semantics=eval_plan.episode_semantics,
                success_rate_threshold=eval_plan.success_rate_threshold,
                required_metric_names=(
                    *eval_plan.required_metric_names,
                    *(
                        (best_checkpoint_plan.metric,)
                        if best_checkpoint_plan.uses_rollout_benchmark
                        else ()
                    ),
                ),
                compile_enabled=compile_enabled,
                compile_mode=compile_mode,
                use_cuda_graph=eval_plan.cuda_graph,
                execution=eval_plan.execution,
                metric_names=eval_plan.metric_names,
                objective_metric=eval_plan.best_metric,
            )
    if train_full_video_requested:
        assert train_full_video_config is not None
        with startup_profiler.phase("train_full_video_setup"):
            train_full_video_recorder = TrainFullVideoRecorder(
                config=train_full_video_config,
                env=env,
                env_raw=env_raw,
                log_dir=log_dir,
                rollout_steps=rollouts,
                train_timesteps=timesteps,
                activation_getter=None,
            )
    with startup_profiler.phase("rollout_workspace_setup"):
        rollout_runtime = RolloutRuntime(
            env=env,
            env_raw=env_raw,
            env_raw_for_agent=env_raw_for_agent,
            agent=agent,
            device=device,
            num_envs=num_envs,
            rollouts=rollouts,
            nan_debug_enabled=nan_debug_enabled,
            use_torch_act_graph=use_torch_act_graph,
            detailed_nsys_nvtx=detailed_nsys_nvtx,
            rl_torch_stream=rl_torch_stream,
            time_limit_bootstrap=time_limit_bootstrap,
            train_full_video_enabled=train_full_video_recorder is not None,
            train_full_video_prepare_step=(
                lambda step: _measure_recording_call(
                    train_full_video_recorder.prepare_step,
                    step,
                )
                if train_full_video_recorder is not None
                else None
            ),
            train_full_video_capture_step=(
                lambda step: _measure_recording_call(
                    train_full_video_recorder.capture_step,
                    step,
                )
                if train_full_video_recorder is not None
                else None
            ),
        )
        episode_means = _RollingEpisodeMean(_DEFAULT_EPISODE_AVERAGE_WINDOW)
    _LOGGER.info(
        "Episode metrics use rolling mean over the last %d finished episodes.",
        _DEFAULT_EPISODE_AVERAGE_WINDOW,
    )
    last_mean_reward: float | None = None

    write_interval_rollouts = write_interval
    checkpoint_interval_rollouts = checkpoint_interval
    episode_metrics_interval_rollouts = episode_metrics_interval
    best_checkpoint_interval_rollouts = best_checkpoint_interval
    clip_failure_controller.log_checkpoint_interval_notice(checkpoint_interval_rollouts)
    _LOGGER.info(
        "Iteration schedule active (1 iteration = %d env step(s)).",
        rollouts,
    )
    _LOGGER.info(
        "Write interval: %d iteration(s), %d step(s).",
        write_interval_rollouts,
        write_interval_rollouts * rollouts,
    )
    _LOGGER.info(
        "Training FPS uses a rolling %d-iteration window after one warm-up iteration.",
        _DEFAULT_TRAINING_FPS_WINDOW,
    )
    _LOGGER.info(
        "Checkpoint interval: %d iteration(s), %d step(s).",
        checkpoint_interval_rollouts,
        checkpoint_interval_rollouts * rollouts,
    )
    _LOGGER.info(
        "Episode metrics sync interval: %d iteration(s), %d step(s).",
        episode_metrics_interval_rollouts,
        episode_metrics_interval_rollouts * rollouts,
    )
    _LOGGER.info(
        "Best checkpoint flush interval: %d iteration(s), %d step(s).",
        best_checkpoint_interval_rollouts,
        best_checkpoint_interval_rollouts * rollouts,
    )
    clip_failure_controller.prewarm(
        checkpoint_controller=checkpoint_controller,
        checkpoint_manager=checkpoint_manager,
        checkpoints_dir=checkpoints_dir,
    )

    with startup_profiler.phase("musclenn_update_graph_prewarm"):
        fast_muscle_controller = _configure_musclenn_update_runtime(
            agent=agent,
            env_raw_for_agent=env_raw_for_agent,
        )
        prewarm_musclenn_update_graph = getattr(
            fast_muscle_controller,
            "prewarm_musclenn_update_graph",
            None,
        )
        if callable(prewarm_musclenn_update_graph):
            prewarm_musclenn_update_graph()
    if benchmark_runner is not None:
        with startup_profiler.phase("tracking_benchmark_prewarm_graphs"):
            benchmark_runner.prewarm_graphs()
    with startup_profiler.phase("rollout_graph_prewarm"):
        rollout_runtime.prewarm()
    with startup_profiler.phase("update_graph_prewarm"):
        training_agent: TrainingAgent = agent
        training_agent.prewarm_update_graph()
    if initial_state_bound:
        clear_initial_state_rollout = getattr(
            agent, "clear_initial_state_rollout", None
        )
        assert callable(clear_initial_state_rollout)
        clear_initial_state_rollout()
        with startup_profiler.phase("initial_state_post_prewarm_reset"):
            obs, _ = _reset_training_environment(env=env, agent=agent)
            copy_agent_observation_into(
                agent=agent,
                observations=obs,
                destination=buf.obs[0],
            )

    writer_max_queue = int(cfg.experiment.get("writer_max_queue", 256))
    writer_flush_secs = int(cfg.experiment.get("writer_flush_secs", 120))
    with startup_profiler.phase("metrics_logger_init"):
        metrics_logger = train_logging.ExperimentMetricsLogger(
            backend=logging_backend,
            cfg=cfg,
            experiment_name=experiment_name,
            log_dir=log_dir,
            writer_max_queue=writer_max_queue,
            writer_flush_secs=writer_flush_secs,
            group=experiment_group,
            wandb_name=wandb_name,
            config_name=config_name,
            setting_name=setting_name,
        )
    eval_controller = TrainingEvalController(
        plan=eval_plan,
        runner=benchmark_runner,
        metrics_logger=metrics_logger,
        history_path=eval_plan.history_path(log_dir),
        checkpoint_controller=checkpoint_controller,
        checkpoint_metric=(
            best_checkpoint_plan.metric
            if best_checkpoint_plan.uses_rollout_benchmark
            else None
        ),
        checkpoint_metric_direction=(
            best_checkpoint_plan.direction
            if best_checkpoint_plan.uses_rollout_benchmark
            else None
        ),
    )
    _LOGGER.info("Training logger backend: %s", metrics_logger.backend)
    startup_profiler.log_summary(logger=logging.getLogger())
    print_startup_summary(startup_profiler.records, enabled=console_enabled)
    training_start_ns = perf_counter_ns()
    training_start_checkpoint_time_ns = checkpoint_controller.time_ns_acc
    training_start_recording_time_ns = recording_time_ns_acc
    training_start_runtime_setup_time_ns = runtime_setup_time_ns(agent)
    training_fps_window = _TrainingFpsWindow(_DEFAULT_TRAINING_FPS_WINDOW)
    try:
        last_losses: dict[str, float] | None = None
        update_count = 0
        best_mean_reward: float | None = None
        best_mean_reward_iteration: int | None = None
        best_mean_episode_timesteps: float | None = None
        last_mean_episode_timesteps: float | None = None
        best_eval_iteration: int | None = None
        last_iteration_index = 0
        objective_state = TrainingObjectiveState(
            enabled=objective_timer_enabled,
            warmup_rollouts=objective_warmup_rollouts,
            timeout_seconds=objective_timeout_seconds,
        )
        physics_time_ns_acc = 0
        update_time_ns_acc = 0
        timing_rollouts = 0
        max_episode_length = getattr(env_raw, "max_episode_length", None)
        max_episode_length_int = (
            int(max_episode_length) if max_episode_length is not None else None
        )
        # The console denominator is the longest episode the benchmark can run:
        # its step cap, shortened by a reference clip that ends naturally.
        eval_max_episode_length_int = (
            eval_episode_step_limit(env_raw, int(eval_plan.play_steps))
            if eval_plan.enabled
            else None
        )

        def _current_active_train_seconds(*, now_ns: int | None = None) -> float:
            return _active_elapsed_seconds(
                now_ns=perf_counter_ns() if now_ns is None else now_ns,
                start_ns=training_start_ns,
                benchmark_eval_time_ns_acc=eval_controller.eval_time_ns_acc,
                start_benchmark_eval_time_ns=0,
                checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
                start_checkpoint_time_ns=training_start_checkpoint_time_ns,
                recording_time_ns_acc=recording_time_ns_acc,
                start_recording_time_ns=training_start_recording_time_ns,
                runtime_setup_time_ns_acc=runtime_setup_time_ns(agent),
                start_runtime_setup_time_ns=training_start_runtime_setup_time_ns,
            )

        def _build_console_state(iteration_index: int) -> TrainingConsoleState:
            last_eval = eval_controller.last_eval
            best_eval = eval_controller.best_checkpoint_eval
            if best_checkpoint_plan.uses_rollout_benchmark and best_eval is not None:
                display_best_reward = _metric_float(best_eval, "reward_mean")
                display_best_episode_length = _metric_float(
                    best_eval, "episode_length_mean"
                )
                display_best_ape = _metric_float(best_eval, "ape_mean")
                display_best_iteration = best_eval_iteration
                display_best_max_episode_length = eval_max_episode_length_int
            else:
                display_best_reward = best_mean_reward
                display_best_episode_length = best_mean_episode_timesteps
                display_best_ape = None
                display_best_iteration = best_mean_reward_iteration
                display_best_max_episode_length = max_episode_length_int
            if best_checkpoint_plan.uses_rollout_benchmark:
                checkpoint_best_value = eval_controller.best_checkpoint_metric_value
                checkpoint_best_iteration = best_eval_iteration
                checkpoint_best_source = "tracking eval"
            elif best_checkpoint_plan.uses_clip_failure_benchmark:
                checkpoint_best_value = clip_failure_controller.best_metric_value
                checkpoint_best_iteration = clip_failure_controller.best_iteration
                checkpoint_best_source = "clip failure"
            else:
                checkpoint_best_value = best_mean_reward
                checkpoint_best_iteration = best_mean_reward_iteration
                checkpoint_best_source = "training"
            now_ns = perf_counter_ns()
            current_active_elapsed = _current_active_train_seconds(now_ns=now_ns)
            current_frames = int(iteration_index) * rollouts * num_envs
            current_fps = training_fps_window.value(
                frames=current_frames,
                active_seconds=current_active_elapsed,
            )

            return TrainingConsoleState(
                iteration=iteration_index,
                total_iterations=total_rollouts,
                eta=format_eta(
                    iteration=iteration_index,
                    total_iterations=total_rollouts,
                    start_time=training_start_ns * 1e-9,
                    now=now_ns * 1e-9,
                ),
                elapsed=format_duration((now_ns - training_start_ns) * 1e-9),
                fps=current_fps,
                train_reward=last_mean_reward,
                train_episode_length=last_mean_episode_timesteps,
                max_episode_length=max_episode_length_int,
                eval_reward=_metric_float(last_eval, "reward_mean"),
                eval_episode_length=_metric_float(last_eval, "episode_length_mean"),
                eval_max_episode_length=eval_max_episode_length_int,
                eval_ape=_metric_float(last_eval, "ape_mean"),
                best_reward=display_best_reward,
                best_iteration=display_best_iteration,
                best_episode_length=display_best_episode_length,
                best_max_episode_length=display_best_max_episode_length,
                best_ape=display_best_ape,
                best_checkpoint_source=checkpoint_best_source,
                best_checkpoint_metric=best_checkpoint_plan.metric,
                best_checkpoint_value=checkpoint_best_value,
                best_checkpoint_iteration=checkpoint_best_iteration,
            )

        with (
            create_training_console(
                total_iterations=total_rollouts,
                enabled=console_enabled,
            ) as train_console,
            torch.cuda.nvtx.range("training_loop"),
        ):
            train_console.event(
                "best",
                (
                    f"{best_checkpoint_plan.configured_source} -> "
                    f"{best_checkpoint_plan.source}"
                    + (
                        f"/{best_checkpoint_plan.benchmark_protocol}"
                        if best_checkpoint_plan.benchmark_protocol is not None
                        else ""
                    )
                    + f"  {best_checkpoint_plan.metric} {best_checkpoint_plan.direction}"
                ),
            )
            if eval_plan.evaluate_at_start:
                initial_eval_outcome = eval_controller.run_if_due(
                    iteration_index=0,
                    total_rollouts=total_rollouts,
                    step_index=0,
                    frames=0,
                    active_elapsed_ns=0,
                    checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
                    best_candidate_enabled=False,
                    force=True,
                )
                if initial_eval_outcome.ran and eval_controller.last_eval is not None:
                    train_console.event(
                        "eval",
                        format_eval_event_message(
                            iteration=0,
                            step=0,
                            frames=0,
                            active_train_seconds=0.0,
                            reward=_metric_float(
                                eval_controller.last_eval,
                                "reward_mean",
                            ),
                            episode_length=_metric_float(
                                eval_controller.last_eval,
                                "episode_length_mean",
                            ),
                            max_episode_length=eval_max_episode_length_int,
                            ape=_metric_float(eval_controller.last_eval, "ape_mean"),
                        ),
                    )
            for rollout_idx in range(total_rollouts):
                objective_state.maybe_start(
                    rollout_idx=rollout_idx,
                    now_ns=perf_counter_ns(),
                    benchmark_eval_time_ns_acc=eval_controller.eval_time_ns_acc,
                    checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
                    recording_time_ns_acc=recording_time_ns_acc,
                    runtime_setup_time_ns_acc=runtime_setup_time_ns(agent),
                )

                # -- Execute rollout ------------------------------------------
                rollout_runtime.collect()
                if train_full_video_recorder is not None:
                    with torch.cuda.nvtx.range("train_full_video_training_boundary"):
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                    with torch.cuda.nvtx.range("train_full_video_record_rollout"):
                        _measure_recording_call(
                            train_full_video_recorder.record_rollout
                        )
                _maybe_update_adaptive_sampling_after_rollout(
                    env_raw_for_agent,
                    wp_stream=rollout_runtime.wp_stream,
                    torch_stream=rollout_runtime.torch_stream,
                    device=device,
                )
                iteration_index = rollout_idx + 1
                step_index = (rollout_idx + 1) * rollouts
                frames = step_index * num_envs
                last_iteration_index = iteration_index

                metrics_sync_due = (
                    episode_metrics_interval_rollouts > 0
                    and (rollout_idx + 1) % episode_metrics_interval_rollouts == 0
                )
                checkpoint_due = (
                    checkpoint_interval_rollouts > 0
                    and (rollout_idx + 1) % checkpoint_interval_rollouts == 0
                )
                clip_failure_benchmark_due = clip_failure_controller.should_run(
                    iteration_index
                )
                best_checkpoint_flush_due = (
                    best_checkpoint_interval_rollouts > 0
                    and (rollout_idx + 1) % best_checkpoint_interval_rollouts == 0
                )

                if episode_threshold_enabled:
                    capture_best_candidate = False
                    with torch.cuda.nvtx.range("sync_episode_metrics"):
                        count, reward_total, timestep_total = (
                            rollout_runtime.consume_episode_totals()
                        )
                    train_logging._log_optional_done_reason_stats(
                        metrics_logger, env_raw, iteration_index
                    )
                    train_logging._log_optional_diagnostic_stats(
                        metrics_logger, env_raw, iteration_index
                    )
                    if count > 0:
                        episode_means.update(
                            reward_total=reward_total,
                            timestep_total=timestep_total,
                            count=count,
                        )
                        (
                            last_mean_reward,
                            last_mean_episode_timesteps,
                        ) = episode_means.get_means()
                        if last_mean_reward is not None and (
                            best_mean_reward is None
                            or last_mean_reward > best_mean_reward
                        ):
                            best_mean_reward = last_mean_reward
                            best_mean_reward_iteration = iteration_index
                            if best_checkpoint_plan.uses_training:
                                capture_best_candidate = True

                    if (
                        reward_stop_threshold is not None
                        and count > 0
                        and last_mean_reward is not None
                    ):
                        if last_mean_reward >= reward_stop_threshold:
                            objective_state.mark_reward_reached(
                                now_ns=perf_counter_ns(),
                                benchmark_eval_time_ns_acc=eval_controller.eval_time_ns_acc,
                                checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
                                recording_time_ns_acc=recording_time_ns_acc,
                                runtime_setup_time_ns_acc=runtime_setup_time_ns(agent),
                            )
                            _LOGGER.info(
                                "REWARD_THRESHOLD_REACHED threshold=%.6f reward=%.6f iteration=%d step=%d frames=%d",
                                reward_stop_threshold,
                                last_mean_reward,
                                iteration_index,
                                step_index,
                                frames,
                            )
                            _LOGGER.info(
                                "TIME_TO_REWARD_THRESHOLD_SECONDS=%.6f",
                                objective_state.time_to_reward_threshold_seconds,
                            )
                            metrics_logger.log_scalars(
                                {
                                    "Objective / Time to reward threshold (s)": (
                                        objective_state.time_to_reward_threshold_seconds
                                    )
                                },
                                iteration_index,
                                axis="iteration",
                            )

                    if (
                        episode_length_stop_threshold is not None
                        and count > 0
                        and last_mean_episode_timesteps is not None
                    ):
                        if (
                            best_mean_episode_timesteps is None
                            or last_mean_episode_timesteps > best_mean_episode_timesteps
                        ):
                            best_mean_episode_timesteps = last_mean_episode_timesteps
                            if best_mean_reward_iteration is None:
                                best_mean_reward_iteration = iteration_index
                        if last_mean_episode_timesteps >= episode_length_stop_threshold:
                            objective_state.mark_episode_length_reached(
                                now_ns=perf_counter_ns(),
                                benchmark_eval_time_ns_acc=eval_controller.eval_time_ns_acc,
                                checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
                                recording_time_ns_acc=recording_time_ns_acc,
                                runtime_setup_time_ns_acc=runtime_setup_time_ns(agent),
                            )
                            _LOGGER.info(
                                "EPISODE_LENGTH_THRESHOLD_REACHED threshold=%.3f episode_timesteps=%.6f iteration=%d step=%d frames=%d",
                                episode_length_stop_threshold,
                                last_mean_episode_timesteps,
                                iteration_index,
                                step_index,
                                frames,
                            )
                            _LOGGER.info(
                                "TIME_TO_EPISODE_LENGTH_THRESHOLD_SECONDS=%.6f",
                                objective_state.time_to_episode_length_threshold_seconds,
                            )
                            metrics_logger.log_scalars(
                                {
                                    "Objective / Time to episode length threshold (s)": (
                                        objective_state.time_to_episode_length_threshold_seconds
                                    )
                                },
                                iteration_index,
                                axis="iteration",
                            )

                    if capture_best_candidate:
                        with torch.cuda.nvtx.range("capture_best_agent"):
                            checkpoint_controller.capture_best_candidate()
                        train_console.event(
                            "new best",
                            (
                                f"reward {_format_console_metric(last_mean_reward)}  "
                                f"episode {_format_console_metric(last_mean_episode_timesteps, precision=0)}"
                                + (
                                    f" / {max_episode_length_int}"
                                    if max_episode_length_int is not None
                                    else ""
                                )
                                + f"  at iter {iteration_index:,}"
                                + best_checkpoint_event_suffix
                            ),
                        )

                    if objective_state.threshold_reached:
                        if checkpoint_controller.best_pending:
                            with torch.cuda.nvtx.range("save_best_agent"):
                                checkpoint_controller.flush_best_if_pending()
                        train_console.update(_build_console_state(iteration_index))
                        break

                # -- PPO update -----------------------------------------------
                with torch.cuda.nvtx.range("ppo_update_host"):
                    last_losses, update_count, update_elapsed_ns = (
                        _run_training_update_cycle(
                            training_stage=training_stage,
                            checkpoint_manager=checkpoint_manager,
                            agent=agent,
                            env_raw_for_agent=env_raw_for_agent,
                            rollout_obs=buf.obs[:rollouts],
                            update_count=update_count,
                            checkpoint_controller=checkpoint_controller,
                        )
                    )
                    rollout_elapsed_ns = rollout_runtime.consume_rollout_timing_ns()
                    physics_time_ns_acc += rollout_elapsed_ns
                    if (
                        rollout_runtime.env_step_timing_enabled
                        and rollout_elapsed_ns > 0
                    ):
                        rollout_runtime.add_env_step_rollout_estimate(
                            rollout_elapsed_ns
                        )
                    update_time_ns_acc += update_elapsed_ns
                    timing_rollouts += 1
                    _maybe_update_motion_plr_after_policy_update(
                        env_raw_for_agent,
                        agent=agent,
                        level_ids=rollout_runtime.adaptive_sampling_level_ids,
                        wp_stream=rollout_runtime.wp_stream,
                        torch_stream=rollout_runtime.torch_stream,
                        device=device,
                    )
                    # Carry over next obs for the next rollout
                    with torch.cuda.nvtx.range("copy_obs"):
                        buf.obs[0].copy_(buf.obs[rollouts])

                benchmark_checkpoint_time_ns = checkpoint_controller.time_ns_acc
                benchmark_runtime_setup_time_ns = runtime_setup_time_ns(agent)
                benchmark_now_ns = perf_counter_ns()
                benchmark_active_elapsed_ns = int(
                    _active_elapsed_seconds(
                        now_ns=benchmark_now_ns,
                        start_ns=training_start_ns,
                        benchmark_eval_time_ns_acc=eval_controller.eval_time_ns_acc,
                        start_benchmark_eval_time_ns=0,
                        checkpoint_time_ns_acc=benchmark_checkpoint_time_ns,
                        start_checkpoint_time_ns=training_start_checkpoint_time_ns,
                        recording_time_ns_acc=recording_time_ns_acc,
                        start_recording_time_ns=training_start_recording_time_ns,
                        runtime_setup_time_ns_acc=benchmark_runtime_setup_time_ns,
                        start_runtime_setup_time_ns=training_start_runtime_setup_time_ns,
                    )
                    * 1_000_000_000
                )
                eval_outcome = _run_eval_at_training_boundary(
                    eval_controller=eval_controller,
                    objective_state=objective_state,
                    now_ns=benchmark_now_ns,
                    iteration_index=iteration_index,
                    total_rollouts=total_rollouts,
                    step_index=step_index,
                    frames=frames,
                    active_elapsed_ns=benchmark_active_elapsed_ns,
                    checkpoint_time_ns_acc=benchmark_checkpoint_time_ns,
                    recording_time_ns_acc=recording_time_ns_acc,
                    runtime_setup_time_ns_acc=benchmark_runtime_setup_time_ns,
                    best_candidate_enabled=best_checkpoint_plan.uses_rollout_benchmark,
                )
                if eval_outcome.ran and eval_controller.last_eval is not None:
                    eval_iteration = (
                        eval_outcome.snapshot_iteration
                        if eval_outcome.snapshot_iteration is not None
                        else iteration_index
                    )
                    eval_step = (
                        eval_outcome.snapshot_step
                        if eval_outcome.snapshot_step is not None
                        else step_index
                    )
                    eval_frames = (
                        eval_outcome.snapshot_frames
                        if eval_outcome.snapshot_frames is not None
                        else frames
                    )
                    train_console.event(
                        "eval",
                        format_eval_event_message(
                            iteration=eval_iteration,
                            step=eval_step,
                            frames=eval_frames,
                            active_train_seconds=eval_outcome.active_elapsed_ns * 1e-9,
                            reward=_metric_float(
                                eval_controller.last_eval,
                                "reward_mean",
                            ),
                            episode_length=_metric_float(
                                eval_controller.last_eval,
                                "episode_length_mean",
                            ),
                            max_episode_length=eval_max_episode_length_int,
                            ape=_metric_float(eval_controller.last_eval, "ape_mean"),
                        ),
                    )
                if (
                    best_checkpoint_plan.uses_rollout_benchmark
                    and eval_outcome.ran
                    and eval_outcome.best_updated
                    and eval_controller.best_checkpoint_metric_value is not None
                ):
                    best_eval_iteration = (
                        eval_outcome.snapshot_iteration
                        if eval_outcome.snapshot_iteration is not None
                        else iteration_index
                    )
                    train_console.event(
                        "new best",
                        (
                            f"{best_checkpoint_plan.metric} "
                            f"{_format_console_metric(eval_controller.best_checkpoint_metric_value, precision=3)} "
                            f"at iter {best_eval_iteration:,}"
                            f"{best_checkpoint_event_suffix}"
                        ),
                    )
                if eval_outcome.threshold_reached:
                    objective_state.mark_benchmark_reached(
                        now_ns=perf_counter_ns(),
                        benchmark_eval_time_ns_acc=eval_controller.eval_time_ns_acc,
                        checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
                        recording_time_ns_acc=recording_time_ns_acc,
                        runtime_setup_time_ns_acc=runtime_setup_time_ns(agent),
                    )
                    _LOGGER.info(
                        "TIME_TO_TRACKING_BENCHMARK_SECONDS=%.6f",
                        objective_state.time_to_benchmark_threshold_seconds,
                    )
                    metrics_logger.log_scalars(
                        {
                            "Objective / Time to benchmark threshold (s)": (
                                objective_state.time_to_benchmark_threshold_seconds
                            )
                        },
                        iteration_index,
                        axis="iteration",
                    )

                if eval_outcome.capture_best_candidate:
                    with torch.cuda.nvtx.range("capture_best_agent"):
                        checkpoint_controller.capture_best_candidate()

                if objective_state.threshold_reached:
                    if checkpoint_controller.best_pending:
                        with torch.cuda.nvtx.range("save_best_agent"):
                            checkpoint_controller.flush_best_if_pending()
                    train_console.update(_build_console_state(iteration_index))
                    break

                if objective_state.timed_out:
                    if stopping_plan.type == "time_budget":
                        _LOGGER.info(
                            "ACTIVE_TIME_BUDGET_REACHED budget_seconds=%.6f elapsed_seconds=%.6f iteration=%d step=%d frames=%d",
                            objective_timeout_seconds,
                            objective_state.timeout_elapsed_seconds,
                            iteration_index,
                            step_index,
                            frames,
                        )
                    else:
                        _LOGGER.info(
                            "OBJECTIVE_TIMEOUT_REACHED timeout_seconds=%.6f elapsed_seconds=%.6f iteration=%d step=%d frames=%d",
                            objective_timeout_seconds,
                            objective_state.timeout_elapsed_seconds,
                            iteration_index,
                            step_index,
                            frames,
                        )
                    if checkpoint_controller.best_pending:
                        with torch.cuda.nvtx.range("save_best_agent"):
                            checkpoint_controller.flush_best_if_pending()
                    train_console.update(_build_console_state(iteration_index))
                    break

                if not episode_threshold_enabled and metrics_sync_due:
                    capture_best_candidate = False
                    with torch.cuda.nvtx.range("sync_episode_metrics"):
                        count, reward_total, timestep_total = (
                            rollout_runtime.consume_episode_totals()
                        )
                    train_logging._log_optional_done_reason_stats(
                        metrics_logger, env_raw, iteration_index
                    )
                    train_logging._log_optional_diagnostic_stats(
                        metrics_logger, env_raw, iteration_index
                    )
                    if count > 0:
                        episode_means.update(
                            reward_total=reward_total,
                            timestep_total=timestep_total,
                            count=count,
                        )
                        (
                            last_mean_reward,
                            last_mean_episode_timesteps,
                        ) = episode_means.get_means()
                        if (
                            best_mean_reward is None
                            or last_mean_reward > best_mean_reward
                        ):
                            best_mean_reward = last_mean_reward
                            best_mean_reward_iteration = iteration_index
                            if best_checkpoint_plan.uses_training:
                                capture_best_candidate = True

                    if capture_best_candidate:
                        with torch.cuda.nvtx.range("capture_best_agent"):
                            checkpoint_controller.capture_best_candidate()
                        train_console.event(
                            "new best",
                            (
                                f"reward {_format_console_metric(last_mean_reward)}  "
                                f"episode {_format_console_metric(last_mean_episode_timesteps, precision=0)}"
                                + (
                                    f" / {max_episode_length_int}"
                                    if max_episode_length_int is not None
                                    else ""
                                )
                                + f"  at iter {iteration_index:,}"
                                + best_checkpoint_event_suffix
                            ),
                        )

                if checkpoint_controller.best_pending and (
                    objective_state.threshold_reached or best_checkpoint_flush_due
                ):
                    with torch.cuda.nvtx.range("save_best_agent"):
                        checkpoint_controller.flush_best_if_pending()

                # -- Logging & checkpointing (per rollout) --------------------
                if (
                    write_interval_rollouts > 0
                    and (rollout_idx + 1) % write_interval_rollouts == 0
                ):
                    with torch.cuda.nvtx.range("logging"):
                        active_train_seconds = _current_active_train_seconds()
                        step_scalars, frame_scalars = build_training_iteration_scalars(
                            last_losses=last_losses,
                            agent=agent,
                            env_raw=env_raw,
                            frames=frames,
                            update_count=update_count,
                            last_mean_reward=last_mean_reward,
                            last_mean_episode_timesteps=last_mean_episode_timesteps,
                            timing_rollouts=timing_rollouts,
                            physics_time_ns_acc=physics_time_ns_acc,
                            update_time_ns_acc=update_time_ns_acc,
                            active_train_seconds=active_train_seconds,
                            training_fps=training_fps_window.value(
                                frames=frames,
                                active_seconds=active_train_seconds,
                            ),
                            checkpoint_time_seconds=checkpoint_controller.time_ns_acc
                            * 1e-9,
                        )
                        if timing_rollouts > 0:
                            physics_time_ns_acc = 0
                            update_time_ns_acc = 0
                            timing_rollouts = 0

                        # A degraded capture belongs in the metrics, not only
                        # in the log: otherwise a run that lost its snapshot
                        # graph is indistinguishable from one that never needed
                        # it, and a later throughput comparison is misleading.
                        fallback_stages = getattr(
                            checkpoint_controller.checkpoint_manager,
                            "eager_fallback_stages",
                            (),
                        )
                        step_scalars["diagnostics/eager_fallbacks"] = float(
                            len(fallback_stages)
                        )
                        with torch.cuda.nvtx.range("add_scalars"):
                            metrics_logger.log_scalars(
                                step_scalars,
                                iteration_index,
                                axis="iteration",
                            )
                            metrics_logger.log_scalars(
                                frame_scalars,
                                frames,
                                axis="frame",
                            )

                if checkpoint_due:
                    with torch.cuda.nvtx.range("save_checkpoint"):
                        checkpoint_path = checkpoint_controller.save_iteration(
                            iteration_index
                        )
                    train_console.event(
                        "checkpoint",
                        f"iter {iteration_index:,}  saved {checkpoint_path.name}",
                    )

                if clip_failure_benchmark_due:
                    clip_failure_trigger_ns = perf_counter_ns()
                    clip_failure_active_train_time_s = _active_elapsed_seconds(
                        now_ns=clip_failure_trigger_ns,
                        start_ns=training_start_ns,
                        benchmark_eval_time_ns_acc=eval_controller.eval_time_ns_acc,
                        start_benchmark_eval_time_ns=0,
                        checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
                        start_checkpoint_time_ns=training_start_checkpoint_time_ns,
                        recording_time_ns_acc=recording_time_ns_acc,
                        start_recording_time_ns=training_start_recording_time_ns,
                        runtime_setup_time_ns_acc=runtime_setup_time_ns(agent),
                        start_runtime_setup_time_ns=training_start_runtime_setup_time_ns,
                    )
                    clip_failure_outcome = clip_failure_controller.run(
                        checkpoint_controller=checkpoint_controller,
                        checkpoint_manager=checkpoint_manager,
                        rollout_runtime=rollout_runtime,
                        agent=agent,
                        metrics_logger=metrics_logger,
                        iteration_index=iteration_index,
                        step_index=step_index,
                        frames=frames,
                        checkpoint_due=checkpoint_due,
                        active_train_time_s=clip_failure_active_train_time_s,
                        checkpoint_time_s=checkpoint_controller.time_ns_acc * 1e-9,
                    )
                    eval_controller.add_foreground_time_ns(
                        clip_failure_outcome.elapsed_ns
                    )
                    if clip_failure_outcome.best_updated:
                        train_console.event(
                            "new best",
                            (
                                f"{best_checkpoint_plan.metric} "
                                f"{_format_console_metric(clip_failure_outcome.metric_value, precision=4)} "
                                f"at iter {iteration_index:,}"
                            ),
                        )

                if (
                    train_video_recorder is not None
                    and train_video_recorder.should_record(iteration_index)
                ):
                    checkpoint_controller.wait_for_update_boundary()
                    recording_start_ns = perf_counter_ns()
                    try:
                        train_video_recorder.record(
                            iteration_index=iteration_index,
                            step_index=step_index,
                        )
                    except Exception:
                        _LOGGER.exception(
                            "Train video recording failed at iteration=%d; disabling further train-video attempts.",
                            iteration_index,
                        )
                        train_video_recorder = None
                    finally:
                        recording_time_ns_acc += perf_counter_ns() - recording_start_ns

                train_console.update(_build_console_state(iteration_index))
                training_fps_window.record(
                    frames=frames,
                    active_seconds=_current_active_train_seconds(),
                )

        final_iteration_index = _resolve_final_iteration_index(
            last_iteration_index=last_iteration_index,
            total_rollouts=total_rollouts,
        )
        drained_eval = eval_controller.drain_pending(
            completion_iteration=final_iteration_index,
            checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
        )
        if drained_eval.best_updated:
            best_eval_iteration = drained_eval.snapshot_iteration
        if drained_eval.threshold_reached and not objective_state.threshold_reached:
            objective_state.mark_benchmark_reached(
                now_ns=perf_counter_ns(),
                benchmark_eval_time_ns_acc=eval_controller.eval_time_ns_acc,
                checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
                recording_time_ns_acc=recording_time_ns_acc,
                runtime_setup_time_ns_acc=runtime_setup_time_ns(agent),
            )
        if not episode_threshold_enabled:
            with torch.cuda.nvtx.range("sync_episode_metrics"):
                count, reward_total, timestep_total = (
                    rollout_runtime.consume_episode_totals()
                )
            train_logging._log_optional_done_reason_stats(
                metrics_logger, env_raw, final_iteration_index
            )
            train_logging._log_optional_diagnostic_stats(
                metrics_logger, env_raw, final_iteration_index
            )
            if count > 0:
                episode_means.update(
                    reward_total=reward_total,
                    timestep_total=timestep_total,
                    count=count,
                )
                last_mean_reward, last_mean_episode_timesteps = (
                    episode_means.get_means()
                )
                if best_mean_reward is None or last_mean_reward > best_mean_reward:
                    best_mean_reward = last_mean_reward
                    best_mean_reward_iteration = final_iteration_index
                    if best_checkpoint_plan.uses_training:
                        with torch.cuda.nvtx.range("capture_best_agent"):
                            checkpoint_controller.capture_best_candidate()

        final_step_index = final_iteration_index * rollouts
        final_agent_path = checkpoint_controller.save_iteration(final_iteration_index)
        if checkpoint_controller.best_pending:
            with torch.cuda.nvtx.range("save_best_agent"):
                checkpoint_controller.flush_best_if_pending()
        if not checkpoint_disabled and not checkpoint_controller.best_written:
            checkpoint_controller.save_best(checkpoints_dir / "best_agent.pt")
        checkpoint_controller.drain_timing()
        best_checkpoint_path = checkpoints_dir / "best_agent.pt"
        post_train_playback_checkpoint = (
            best_checkpoint_path if best_checkpoint_path.exists() else final_agent_path
        )
        training_complete_ns = perf_counter_ns()
        training_elapsed_seconds = (training_complete_ns - training_start_ns) * 1e-9
        active_elapsed_seconds = _active_elapsed_seconds(
            now_ns=training_complete_ns,
            start_ns=training_start_ns,
            benchmark_eval_time_ns_acc=eval_controller.eval_time_ns_acc,
            start_benchmark_eval_time_ns=0,
            checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
            start_checkpoint_time_ns=training_start_checkpoint_time_ns,
            recording_time_ns_acc=recording_time_ns_acc,
            start_recording_time_ns=training_start_recording_time_ns,
            runtime_setup_time_ns_acc=runtime_setup_time_ns(agent),
            start_runtime_setup_time_ns=training_start_runtime_setup_time_ns,
        )
        recording_elapsed_seconds = (
            max(
                0,
                recording_time_ns_acc - training_start_recording_time_ns,
            )
            * 1e-9
        )
        _LOGGER.info(
            "TRAINING_COMPLETE elapsed=%s active_elapsed=%s checkpoint_time=%s benchmark_eval_time=%s best_mean_reward=%s iteration=%d frames=%d recording_time=%s",
            _format_elapsed_seconds(training_elapsed_seconds),
            _format_elapsed_seconds(active_elapsed_seconds),
            _format_elapsed_seconds(checkpoint_controller.time_ns_acc * 1e-9),
            _format_elapsed_seconds(eval_controller.eval_time_ns_acc * 1e-9),
            (f"{best_mean_reward:.6f}" if best_mean_reward is not None else "None"),
            final_iteration_index,
            final_step_index * num_envs,
            _format_elapsed_seconds(recording_elapsed_seconds),
        )
        rollout_runtime.log_env_step_timing_marker(final_iteration_index)
        _LOGGER.info(
            "TRAINING_LOOP_COMPLETE elapsed=%.9fs active_elapsed=%.9fs iterations=%d rollouts=%d num_envs=%d frames=%d recording_time=%.9fs",
            training_elapsed_seconds,
            active_elapsed_seconds,
            final_iteration_index,
            rollouts,
            num_envs,
            final_step_index * num_envs,
            recording_elapsed_seconds,
        )
        if best_checkpoint_plan.uses_rollout_benchmark:
            completed_best_value = eval_controller.best_checkpoint_metric_value
            completed_best_iteration = best_eval_iteration
        elif best_checkpoint_plan.uses_clip_failure_benchmark:
            completed_best_value = clip_failure_controller.best_metric_value
            completed_best_iteration = clip_failure_controller.best_iteration
        else:
            completed_best_value = best_mean_reward
            completed_best_iteration = best_mean_reward_iteration
        _LOGGER.info(
            "BEST_CHECKPOINT source=%s benchmark_protocol=%s metric=%s direction=%s value=%s iteration=%s",
            best_checkpoint_plan.source,
            best_checkpoint_plan.benchmark_protocol or "none",
            best_checkpoint_plan.metric,
            best_checkpoint_plan.direction,
            (
                f"{completed_best_value:.6f}"
                if completed_best_value is not None
                else "None"
            ),
            (
                str(completed_best_iteration)
                if completed_best_iteration is not None
                else "None"
            ),
        )
        if best_checkpoint_plan.uses_rollout_benchmark:
            _LOGGER.info(
                "EVAL_BEST_CHECKPOINT metric=%s value=%s reward_mean=%s episode_length_mean=%s success_rate=%s ape_mean=%s global_mpjpe_mean=%s ee_mean=%s",
                best_checkpoint_plan.metric,
                (
                    f"{eval_controller.best_checkpoint_metric_value:.6f}"
                    if eval_controller.best_checkpoint_metric_value is not None
                    else "None"
                ),
                (
                    f"{float(eval_controller.best_checkpoint_eval['reward_mean']):.6f}"
                    if eval_controller.best_checkpoint_eval is not None
                    and eval_controller.best_checkpoint_eval.get("reward_mean", None)
                    is not None
                    else "None"
                ),
                (
                    f"{float(eval_controller.best_checkpoint_eval['episode_length_mean']):.6f}"
                    if eval_controller.best_checkpoint_eval is not None
                    and eval_controller.best_checkpoint_eval.get(
                        "episode_length_mean", None
                    )
                    is not None
                    else "None"
                ),
                (
                    f"{float(eval_controller.best_checkpoint_eval['success_rate']):.6f}"
                    if eval_controller.best_checkpoint_eval is not None
                    and eval_controller.best_checkpoint_eval.get("success_rate", None)
                    is not None
                    else "None"
                ),
                (
                    f"{float(eval_controller.best_checkpoint_eval['ape_mean']):.6f}"
                    if eval_controller.best_checkpoint_eval is not None
                    and eval_controller.best_checkpoint_eval.get("ape_mean", None)
                    is not None
                    else "None"
                ),
                (
                    f"{float(eval_controller.best_checkpoint_eval['global_mpjpe_mean']):.6f}"
                    if eval_controller.best_checkpoint_eval is not None
                    and eval_controller.best_checkpoint_eval.get(
                        "global_mpjpe_mean", None
                    )
                    is not None
                    else "None"
                ),
                (
                    f"{float(eval_controller.best_checkpoint_eval['ee_mean']):.6f}"
                    if eval_controller.best_checkpoint_eval is not None
                    and eval_controller.best_checkpoint_eval.get("ee_mean", None)
                    is not None
                    else "None"
                ),
            )
        if (
            objective_state.timed_out
            and objective_state.timeout_elapsed_seconds is not None
        ):
            if stopping_plan.type == "time_budget":
                _LOGGER.info(
                    "ACTIVE_TIME_BUDGET_SUMMARY budget_seconds=%.6f elapsed_seconds=%.6f final_iteration=%d",
                    objective_timeout_seconds,
                    objective_state.timeout_elapsed_seconds,
                    final_iteration_index,
                )
            else:
                _LOGGER.warning(
                    "OBJECTIVE_TIMEOUT_SUMMARY timeout_seconds=%.6f elapsed_seconds=%.6f final_iteration=%d",
                    objective_timeout_seconds,
                    objective_state.timeout_elapsed_seconds,
                    final_iteration_index,
                )
        if reward_stop_threshold is not None and not objective_state.threshold_reached:
            _LOGGER.warning(
                "REWARD_THRESHOLD_NOT_REACHED threshold=%.6f best_mean_reward=%s max_iteration=%d",
                reward_stop_threshold,
                f"{best_mean_reward:.6f}" if best_mean_reward is not None else "None",
                final_iteration_index,
            )
            if fail_on_reward_threshold_miss:
                raise RuntimeError(
                    "Reward threshold was not reached within the configured timesteps."
                )
        if (
            episode_length_stop_threshold is not None
            and not objective_state.threshold_reached
        ):
            _LOGGER.warning(
                "EPISODE_LENGTH_THRESHOLD_NOT_REACHED threshold=%.3f best_mean_episode_timesteps=%s max_iteration=%d",
                episode_length_stop_threshold,
                (
                    f"{best_mean_episode_timesteps:.6f}"
                    if best_mean_episode_timesteps is not None
                    else "None"
                ),
                final_iteration_index,
            )
            if fail_on_episode_length_threshold_miss:
                raise RuntimeError(
                    "Episode-length threshold was not reached within the configured timesteps."
                )
        if eval_plan.threshold_enabled and not objective_state.threshold_reached:
            _LOGGER.warning(
                "BENCHMARK_THRESHOLD_NOT_REACHED ape_threshold=%s global_mpjpe_threshold=%s ee_threshold=%s success_rate_threshold=%s consecutive_passes_required=%d best_valid_ape=%s best_valid_global_mpjpe=%s best_valid_ee=%s best_success_rate=%s last_valid=%s last_success_rate=%s last_failure_step=%s max_iteration=%d",
                (
                    f"{eval_plan.ape_mean_threshold:.6f}"
                    if eval_plan.ape_mean_threshold is not None
                    else "None"
                ),
                (
                    f"{eval_plan.global_mpjpe_mean_threshold:.6f}"
                    if eval_plan.global_mpjpe_mean_threshold is not None
                    else "None"
                ),
                (
                    f"{eval_plan.ee_mean_threshold:.6f}"
                    if eval_plan.ee_mean_threshold is not None
                    else "None"
                ),
                (
                    f"{eval_plan.success_rate_threshold:.6f}"
                    if eval_plan.success_rate_threshold is not None
                    else "None"
                ),
                eval_plan.consecutive_passes_required,
                (
                    f"{float(eval_controller.best_eval['ape_mean']):.6f}"
                    if eval_controller.best_eval is not None
                    else "None"
                ),
                (
                    f"{float(eval_controller.best_eval['global_mpjpe_mean']):.6f}"
                    if eval_controller.best_eval is not None
                    else "None"
                ),
                (
                    f"{float(eval_controller.best_eval['ee_mean']):.6f}"
                    if eval_controller.best_eval is not None
                    else "None"
                ),
                (
                    f"{float(eval_controller.best_eval['success_rate']):.6f}"
                    if eval_controller.best_eval is not None
                    and eval_controller.best_eval.get("success_rate", None) is not None
                    else "None"
                ),
                (
                    bool(eval_controller.last_eval["valid"])
                    if eval_controller.last_eval is not None
                    else "None"
                ),
                (
                    f"{float(eval_controller.last_eval['success_rate']):.6f}"
                    if eval_controller.last_eval is not None
                    and eval_controller.last_eval.get("success_rate", None) is not None
                    else "None"
                ),
                (
                    int(eval_controller.last_eval.get("failure_step", 0))
                    if eval_controller.last_eval is not None
                    else "None"
                ),
                final_iteration_index,
            )
            if fail_on_benchmark_threshold_miss:
                raise RuntimeError(
                    "Benchmark threshold was not reached within the configured timesteps."
                )
    except KeyboardInterrupt:
        interrupted_iteration_index = int(last_iteration_index)
        interrupted_step_index = interrupted_iteration_index * rollouts
        interrupted_frames = interrupted_step_index * num_envs
        _LOGGER.info(
            "KeyboardInterrupt received. Stopping gracefully at iteration=%d step=%d frames=%d.",
            interrupted_iteration_index,
            interrupted_step_index,
            interrupted_frames,
        )
        _drain_pending_eval_for_shutdown(
            eval_controller=eval_controller,
            completion_iteration=interrupted_iteration_index,
            checkpoint_time_ns_acc=checkpoint_controller.time_ns_acc,
        )
        if interrupted_iteration_index > 0 and not checkpoint_disabled:
            interrupted_agent_path = (
                checkpoints_dir
                / f"agent_interrupted_iter_{interrupted_iteration_index}.pt"
            )
            try:
                checkpoint_controller.save_persistent(interrupted_agent_path)
                if checkpoint_controller.best_pending:
                    checkpoint_controller.flush_best_if_pending()
                _LOGGER.info(
                    "Saved interrupted checkpoint to: %s",
                    interrupted_agent_path,
                )
            except Exception:
                _LOGGER.exception(
                    "Failed to save interrupted checkpoint at iteration=%d.",
                    interrupted_iteration_index,
                )
        return
    finally:
        active_error = sys.exception()
        timing_drain_error: Exception | None = None
        try:
            checkpoint_controller.drain_timing()
        except Exception as exc:
            timing_drain_error = exc
            _LOGGER.exception("Failed to drain final checkpoint timing.")
        cleanup_error = _close_training_resources(
            checkpoint_manager=checkpoint_manager,
            metrics_logger=metrics_logger,
            benchmark_runner=benchmark_runner,
            clip_failure_controller=clip_failure_controller,
            train_video_recorder=train_video_recorder,
            train_full_video_recorder=train_full_video_recorder,
            env=env,
        )
        if active_error is None and timing_drain_error is not None:
            raise timing_drain_error
        if cleanup_error is not None and active_error is None:
            raise cleanup_error

    if play_after_train and post_train_playback_checkpoint is not None:
        post_train_playback_runner.play_best_policy_after_training(
            checkpoint_path=post_train_playback_checkpoint,
            log_dir=log_dir,
        )
