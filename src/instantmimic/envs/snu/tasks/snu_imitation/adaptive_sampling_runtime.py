from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import torch
import warp as wp

from instantmimic.envs.direct_rl_env import (
    DirectRLEnv,
    make_stream_aligned_cuda_graph_streams,
)
from instantmimic.envs.snu.tasks.snu_imitation.adaptive_sampling import (
    ADAPTIVE_SAMPLING_METHOD_MOTION_PLR,
    ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX,
    ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX_SCALED,
    ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE,
    allocate_adaptive_sampling_buffers,
    allocate_disabled_adaptive_sampling_buffers,
    resolve_adaptive_sampling_config,
    solve_rank_exponent_for_target_ess,
)
from instantmimic.utils.torch_warp_capture import capture_warp_graph

_LOGGER = logging.getLogger(__name__)
__all__ = (
    "AdaptiveSamplingRuntimeState",
    "initialize",
    "capture_terminal_state_before_reset",
    "set_exp_dist",
    "set_adaptive_sampling_benchmark_prior",
    "clear_adaptive_sampling_benchmark_prior",
    "adaptive_sampling_uses_benchmark_prior",
    "select_adaptive_sampling_distribution_buffers",
    "blend_adaptive_sampling_distribution_torch",
    "refresh_adaptive_sampling_distribution",
    "ensure_adaptive_update_work_buffers",
    "launch_adaptive_sampling_distribution_update",
    "launch_motion_plr_distribution_update",
    "launch_sonic_failure_distribution_update",
    "update_adaptive_sampling_distribution_after_rollout",
    "update_motion_plr_distribution_after_policy_update",
    "get_adaptive_sampling_state",
    "get_checkpoint_state",
    "load_checkpoint_state",
)

_ADAPTIVE_SAMPLING_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass
class AdaptiveSamplingRuntimeState:
    """Persistent buffers and scheduling state for adaptive motion sampling."""

    sampling_enabled: bool = False
    sampling_disable_reason: str | None = None
    sampling_method: str = ""
    sampling_uses_episode_metrics: bool = False
    sampling_uses_sonic_statistics: bool = False
    metric_type: int = 0
    blame_mode: int = 0
    chunk_size: int = 1
    unit_count: int = 0
    ema_alpha: float = 0.0
    scaled_focus_strength: float = 0.0
    scaled_score_scale: float = 1.0
    scaled_stale_half_life_passes: float = 0.0
    temperature: float = 1.0
    decay_rate: float = 1.0
    random_sample_prob: float = 0.0
    min_metric: float = 0.0
    terminal_window_chunk_count: int = 1
    benchmark_beta: float = 0.0
    plr_focus_ess_ratio: float = 0.0
    plr_staleness_mix: float = 0.0
    sonic_uniform_sampling_rate: float = 0.0
    sonic_failure_rate_max_over_mean: float = 0.0
    sonic_pre_failure_window_steps: int = 0
    benchmark_prior_enabled: bool = False
    update_cuda_graph: bool = True
    sampling_update_graph: wp.Graph | None = None
    update_torch_stream: torch.cuda.Stream | None = None
    update_wp_stream: wp.Stream | None = None
    episode_start_unit_idx: torch.Tensor | None = None
    episode_terminal_unit_idx: torch.Tensor | None = None
    episode_terminal_clip_unit_start: torch.Tensor | None = None
    episode_terminal_clip_unit_count: torch.Tensor | None = None
    episode_reward_sum: torch.Tensor | None = None
    episode_length_before_reset: torch.Tensor | None = None
    metric_sum: torch.Tensor | None = None
    metric_count: torch.Tensor | None = None
    metric_event_count: torch.Tensor | None = None
    exp_metric: torch.Tensor | None = None
    online_dist: torch.Tensor | None = None
    online_cdf: torch.Tensor | None = None
    benchmark_prior: torch.Tensor | None = None
    benchmark_prior_enabled_tensor: torch.Tensor | None = None
    runtime_dist_buffer: torch.Tensor | None = None
    runtime_cdf_buffer: torch.Tensor | None = None
    dist: torch.Tensor | None = None
    cdf: torch.Tensor | None = None
    update_weights: torch.Tensor | None = None
    update_decayed_metric: torch.Tensor | None = None
    update_event_active: torch.Tensor | None = None
    update_count_active: torch.Tensor | None = None
    update_next_metric: torch.Tensor | None = None
    update_updated_metric: torch.Tensor | None = None
    update_scaled_metric: torch.Tensor | None = None
    update_max_metric: torch.Tensor | None = None
    update_mean_metric: torch.Tensor | None = None
    update_decay_active: torch.Tensor | None = None
    update_next_dist: torch.Tensor | None = None
    update_any_count: torch.Tensor | None = None
    update_should_update: torch.Tensor | None = None
    update_norm: torch.Tensor | None = None
    update_norm_clamped: torch.Tensor | None = None
    sonic_num_failures: torch.Tensor | None = None
    sonic_num_exposures: torch.Tensor | None = None
    sonic_bin_weight: torch.Tensor | None = None
    plr_rank_exponent: float = 0.0
    plr_seen: torch.Tensor | None = None
    plr_last_seen: torch.Tensor | None = None
    plr_update_count: torch.Tensor | None = None
    plr_has_mass: torch.Tensor | None = None
    plr_sorted_indices: torch.Tensor | None = None
    plr_advantage_magnitude: torch.Tensor | None = None
    plr_sample_weights: torch.Tensor | None = None
    plr_level_ids: torch.Tensor | None = None
    plr_returns: torch.Tensor | None = None
    plr_values: torch.Tensor | None = None


def initialize(env) -> None:
    runtime_state = AdaptiveSamplingRuntimeState()
    env.adaptive_sampling_state = runtime_state
    reset_settings = env.reset_settings
    adaptive_cfg = resolve_adaptive_sampling_config(
        env.cfg,
        motion_frame_count=env.motion_frame_count,
        random_init_enabled=reset_settings.random_init_enabled,
        deterministic_reset=reset_settings.deterministic_reset,
    )
    if adaptive_cfg.disable_reason is not None:
        _LOGGER.info(
            "adaptive_sampling: %s: disabling adaptive sampling.",
            adaptive_cfg.disable_reason,
        )

    runtime_state.sampling_enabled = adaptive_cfg.enabled
    runtime_state.sampling_disable_reason = adaptive_cfg.disable_reason
    runtime_state.sampling_method = adaptive_cfg.method
    reward_softmax_methods = (
        ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX,
        ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX_SCALED,
    )
    runtime_state.sampling_uses_episode_metrics = (
        adaptive_cfg.enabled and adaptive_cfg.method in reward_softmax_methods
    )
    runtime_state.sampling_uses_sonic_statistics = (
        adaptive_cfg.enabled
        and adaptive_cfg.method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE
    )
    runtime_state.metric_type = adaptive_cfg.metric_type
    runtime_state.blame_mode = adaptive_cfg.blame_mode
    runtime_state.chunk_size = adaptive_cfg.chunk_size
    runtime_state.unit_count = int(
        getattr(env, "motion_clip_total_unit_count", adaptive_cfg.unit_count)
    )
    runtime_state.ema_alpha = adaptive_cfg.ema_alpha
    runtime_state.scaled_focus_strength = adaptive_cfg.focus_strength
    runtime_state.scaled_score_scale = max(1.0, float(env.max_episode_length))
    runtime_state.scaled_stale_half_life_passes = adaptive_cfg.stale_half_life_passes
    if adaptive_cfg.method == ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX_SCALED:
        runtime_state.temperature = (
            adaptive_cfg.focus_strength / runtime_state.scaled_score_scale
        )
        runtime_state.decay_rate = 1.0
    else:
        runtime_state.temperature = adaptive_cfg.temperature
        runtime_state.decay_rate = adaptive_cfg.decay_rate
    runtime_state.random_sample_prob = (
        adaptive_cfg.random_sample_prob
        if adaptive_cfg.method in reward_softmax_methods
        else 0.0
    )
    runtime_state.min_metric = adaptive_cfg.min_metric
    runtime_state.terminal_window_chunk_count = adaptive_cfg.terminal_window_chunk_count
    runtime_state.benchmark_beta = adaptive_cfg.benchmark_beta
    runtime_state.plr_focus_ess_ratio = adaptive_cfg.focus_ess_ratio
    runtime_state.plr_staleness_mix = adaptive_cfg.staleness_mix
    runtime_state.sonic_uniform_sampling_rate = adaptive_cfg.sonic_uniform_sampling_rate
    runtime_state.sonic_failure_rate_max_over_mean = (
        adaptive_cfg.sonic_failure_rate_max_over_mean
    )
    runtime_state.sonic_pre_failure_window_steps = (
        adaptive_cfg.sonic_pre_failure_window_steps
        if adaptive_cfg.method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE
        else 0
    )
    runtime_state.benchmark_prior_enabled = False
    runtime_state.update_cuda_graph = True
    runtime_state.sampling_update_graph = None

    if runtime_state.sampling_enabled:
        adaptive_buffers = allocate_adaptive_sampling_buffers(
            num_envs=env.num_envs,
            unit_count=runtime_state.unit_count,
            device=env.device,
        )
    else:
        adaptive_buffers = allocate_disabled_adaptive_sampling_buffers(
            num_envs=env.num_envs,
            device=env.device,
        )
    runtime_state.episode_start_unit_idx = adaptive_buffers.episode_start_unit_idx
    runtime_state.episode_terminal_unit_idx = adaptive_buffers.episode_terminal_unit_idx
    runtime_state.episode_terminal_clip_unit_start = (
        adaptive_buffers.episode_terminal_clip_unit_start
    )
    runtime_state.episode_terminal_clip_unit_count = (
        adaptive_buffers.episode_terminal_clip_unit_count
    )
    runtime_state.episode_reward_sum = adaptive_buffers.episode_reward_sum
    runtime_state.episode_length_before_reset = (
        adaptive_buffers.episode_length_before_reset
    )
    runtime_state.metric_sum = adaptive_buffers.metric_sum
    runtime_state.metric_count = adaptive_buffers.metric_count
    runtime_state.metric_event_count = adaptive_buffers.metric_event_count
    runtime_state.exp_metric = adaptive_buffers.exp_metric
    runtime_state.online_dist = adaptive_buffers.online_dist
    runtime_state.online_cdf = adaptive_buffers.online_cdf
    runtime_state.benchmark_prior = adaptive_buffers.benchmark_prior
    runtime_state.benchmark_prior_enabled_tensor = (
        adaptive_buffers.benchmark_prior_enabled
    )
    runtime_state.runtime_dist_buffer = adaptive_buffers.dist
    runtime_state.runtime_cdf_buffer = adaptive_buffers.cdf
    runtime_state.dist = runtime_state.online_dist
    runtime_state.cdf = runtime_state.online_cdf
    if runtime_state.sampling_enabled:
        ensure_adaptive_update_work_buffers(env)
        if runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_MOTION_PLR:
            initialize_motion_plr_state(env)
        elif runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE:
            initialize_sonic_failure_state(env)
        refresh_adaptive_sampling_distribution(env)


def capture_terminal_state_before_reset(env) -> None:
    runtime_state = env.adaptive_sampling_state
    chunk = max(1, int(runtime_state.chunk_size))
    if env._multi_motion_enabled:
        clip_idx = torch.clamp(
            env.motion_clip_idx.to(dtype=torch.long),
            min=0,
            max=max(int(env.motion_clip_length.numel()) - 1, 0),
        )
        clip_length = env.motion_clip_length[clip_idx].to(dtype=torch.long)
        local_frame_idx = torch.remainder(
            env.ref_frame_idx.to(dtype=torch.long), clip_length
        )
        clip_unit_start = env.motion_clip_unit_start[clip_idx].to(dtype=torch.long)
        clip_unit_count = env.motion_clip_unit_count[clip_idx].to(dtype=torch.long)
        terminal_local_unit_idx = (
            torch.div(local_frame_idx, chunk, rounding_mode="floor")
            if chunk > 1
            else local_frame_idx
        )
        terminal_unit_idx = clip_unit_start + terminal_local_unit_idx
        runtime_state.episode_terminal_unit_idx.copy_(
            terminal_unit_idx.to(dtype=runtime_state.episode_terminal_unit_idx.dtype)
        )
        runtime_state.episode_terminal_clip_unit_start.copy_(
            clip_unit_start.to(
                dtype=runtime_state.episode_terminal_clip_unit_start.dtype
            )
        )
        runtime_state.episode_terminal_clip_unit_count.copy_(
            clip_unit_count.to(
                dtype=runtime_state.episode_terminal_clip_unit_count.dtype
            )
        )
        return

    local_frame_idx = torch.remainder(
        env.ref_frame_idx.to(dtype=torch.long), int(env.motion_frame_count)
    )
    terminal_unit_idx = (
        torch.div(local_frame_idx, chunk, rounding_mode="floor")
        if chunk > 1
        else local_frame_idx
    )
    runtime_state.episode_terminal_unit_idx.copy_(
        terminal_unit_idx.to(dtype=runtime_state.episode_terminal_unit_idx.dtype)
    )
    runtime_state.episode_terminal_clip_unit_start.zero_()
    runtime_state.episode_terminal_clip_unit_count.fill_(int(runtime_state.unit_count))


def set_exp_dist(self, distribution: torch.Tensor | np.ndarray):
    """Set the external adaptive-sampling probability distribution.

    Expects a probability vector over adaptive units (frames or chunks).
    """
    runtime_state = self.adaptive_sampling_state
    dist = torch.as_tensor(distribution, device=self.device, dtype=torch.float32)
    if dist.ndim > 1:
        dist = dist[0]
    dist = dist.reshape(-1)
    if dist.numel() != runtime_state.unit_count:
        raise ValueError(
            "distribution size mismatch: "
            f"expected {runtime_state.unit_count}, got {dist.numel()}"
        )
    runtime_state.online_dist.copy_(dist)
    runtime_state.online_dist.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    runtime_state.online_dist.clamp_(min=0.0)
    norm = float(runtime_state.online_dist.sum().item())
    if norm <= 0.0:
        runtime_state.online_dist.fill_(1.0 / float(runtime_state.unit_count))
    else:
        runtime_state.online_dist.div_(norm)
    torch.cumsum(
        runtime_state.online_dist,
        dim=0,
        out=runtime_state.online_cdf,
    )
    runtime_state.online_cdf[-1] = 1.0
    refresh_adaptive_sampling_distribution(self)


def set_adaptive_sampling_benchmark_prior(
    self, distribution: torch.Tensor | np.ndarray
) -> None:
    runtime_state = self.adaptive_sampling_state
    prior = torch.as_tensor(distribution, device=self.device, dtype=torch.float32)
    if prior.ndim > 1:
        prior = prior[0]
    prior = prior.reshape(-1)
    if prior.numel() != runtime_state.unit_count:
        raise ValueError(
            "benchmark prior size mismatch: "
            f"expected {runtime_state.unit_count}, got {prior.numel()}"
        )
    runtime_state.benchmark_prior.copy_(prior)
    runtime_state.benchmark_prior.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    runtime_state.benchmark_prior.clamp_(min=0.0)
    norm = float(runtime_state.benchmark_prior.sum().item())
    if norm <= 0.0:
        runtime_state.benchmark_prior.zero_()
        runtime_state.benchmark_prior_enabled = False
        runtime_state.benchmark_prior_enabled_tensor.zero_()
    else:
        runtime_state.benchmark_prior.div_(norm)
        runtime_state.benchmark_prior_enabled = True
        runtime_state.benchmark_prior_enabled_tensor.fill_(1)
    refresh_adaptive_sampling_distribution(self)


def clear_adaptive_sampling_benchmark_prior(self) -> None:
    runtime_state = self.adaptive_sampling_state
    runtime_state.benchmark_prior.zero_()
    runtime_state.benchmark_prior_enabled = False
    runtime_state.benchmark_prior_enabled_tensor.zero_()
    refresh_adaptive_sampling_distribution(self)


def adaptive_sampling_uses_benchmark_prior(self) -> bool:
    runtime_state = self.adaptive_sampling_state
    return (
        bool(runtime_state.benchmark_prior_enabled)
        and float(runtime_state.benchmark_beta) > 0.0
    )


def select_adaptive_sampling_distribution_buffers(self) -> bool:
    runtime_state = self.adaptive_sampling_state
    use_prior = adaptive_sampling_uses_benchmark_prior(self)
    if use_prior:
        runtime_state.dist = runtime_state.runtime_dist_buffer
        runtime_state.cdf = runtime_state.runtime_cdf_buffer
    else:
        runtime_state.dist = runtime_state.online_dist
        runtime_state.cdf = runtime_state.online_cdf
    return use_prior


def blend_adaptive_sampling_distribution_torch(self) -> None:
    runtime_state = self.adaptive_sampling_state
    if not adaptive_sampling_uses_benchmark_prior(self):
        return

    ensure_adaptive_update_work_buffers(self)
    beta = float(runtime_state.benchmark_beta)
    runtime_state.runtime_dist_buffer.copy_(runtime_state.online_dist)
    runtime_state.runtime_dist_buffer.mul_(1.0 - beta)
    runtime_state.runtime_dist_buffer.add_(
        runtime_state.benchmark_prior,
        alpha=beta,
    )
    runtime_state.runtime_dist_buffer.clamp_(min=0.0)
    torch.sum(
        runtime_state.runtime_dist_buffer,
        dim=0,
        out=runtime_state.update_norm,
    )
    torch.clamp(
        runtime_state.update_norm,
        min=1.0e-12,
        out=runtime_state.update_norm_clamped,
    )
    runtime_state.runtime_dist_buffer.div_(runtime_state.update_norm_clamped)
    torch.cumsum(
        runtime_state.runtime_dist_buffer,
        dim=0,
        out=runtime_state.runtime_cdf_buffer,
    )
    runtime_state.runtime_cdf_buffer[-1:].fill_(1.0)


def refresh_adaptive_sampling_distribution(self) -> None:
    runtime_state = self.adaptive_sampling_state
    select_adaptive_sampling_distribution_buffers(self)
    runtime_state.sampling_update_graph = None
    blend_adaptive_sampling_distribution_torch(self)


def ensure_adaptive_update_work_buffers(self) -> None:
    runtime_state = self.adaptive_sampling_state
    unit_count = int(runtime_state.unit_count)
    device = runtime_state.online_dist.device
    needs_allocate = (
        runtime_state.update_weights is None
        or runtime_state.update_weights.shape != (unit_count,)
        or runtime_state.update_weights.device != device
    )
    if not needs_allocate:
        return

    runtime_state.sampling_update_graph = None
    runtime_state.update_event_active = torch.empty(1, device=device, dtype=torch.bool)
    runtime_state.update_any_count = torch.empty((), device=device, dtype=torch.bool)
    runtime_state.update_should_update = torch.empty(1, device=device, dtype=torch.bool)
    runtime_state.update_count_active = torch.empty(
        unit_count, device=device, dtype=torch.bool
    )
    runtime_state.update_decay_active = torch.empty(
        unit_count, device=device, dtype=torch.bool
    )
    runtime_state.update_scaled_metric = torch.empty(
        unit_count, device=device, dtype=torch.float32
    )
    runtime_state.update_decayed_metric = torch.empty(
        unit_count, device=device, dtype=torch.float32
    )
    runtime_state.update_mean_metric = torch.empty(
        unit_count, device=device, dtype=torch.float32
    )
    runtime_state.update_updated_metric = torch.empty(
        unit_count, device=device, dtype=torch.float32
    )
    runtime_state.update_next_metric = torch.empty(
        unit_count, device=device, dtype=torch.float32
    )
    runtime_state.update_weights = torch.empty(
        unit_count, device=device, dtype=torch.float32
    )
    runtime_state.update_next_dist = torch.empty(
        unit_count, device=device, dtype=torch.float32
    )
    runtime_state.update_max_metric = torch.empty(
        (), device=device, dtype=torch.float32
    )
    runtime_state.update_norm = torch.empty((), device=device, dtype=torch.float32)
    runtime_state.update_norm_clamped = torch.empty(
        (), device=device, dtype=torch.float32
    )


def initialize_sonic_failure_state(self) -> None:
    """Initialize cumulative failure/exposure statistics and sequence weights."""
    runtime_state = self.adaptive_sampling_state
    unit_count = int(runtime_state.unit_count)
    device = runtime_state.online_dist.device
    runtime_state.sonic_num_failures = torch.ones(
        unit_count, device=device, dtype=torch.float32
    )
    runtime_state.sonic_num_exposures = torch.ones(
        unit_count, device=device, dtype=torch.float32
    )

    clip_unit_count = self.motion_clip_unit_count.to(device=device, dtype=torch.long)
    clip_unit_start = self.motion_clip_unit_start.to(device=device, dtype=torch.long)
    clip_length = self.motion_clip_length.to(device=device, dtype=torch.long)
    if int(clip_unit_count.sum().item()) != unit_count:
        raise ValueError(
            "SONIC adaptive unit metadata mismatch: "
            f"expected {unit_count}, got {int(clip_unit_count.sum().item())}"
        )
    repeated_unit_start = torch.repeat_interleave(clip_unit_start, clip_unit_count)
    repeated_unit_count = torch.repeat_interleave(clip_unit_count, clip_unit_count)
    repeated_clip_length = torch.repeat_interleave(clip_length, clip_unit_count)
    local_unit_idx = torch.arange(unit_count, device=device) - repeated_unit_start
    unit_frame_count = repeated_clip_length - local_unit_idx * int(
        runtime_state.chunk_size
    )
    unit_frame_count.clamp_(min=1, max=int(runtime_state.chunk_size))
    base_weight = unit_frame_count.to(torch.float32)
    base_weight.div_(base_weight.mean().clamp_min(1.0e-12))
    base_weight.div_(repeated_unit_count.to(torch.float32).clamp_min(1.0))
    runtime_state.sonic_bin_weight = base_weight

    runtime_state.exp_metric.fill_(1.0)
    runtime_state.online_dist.copy_(base_weight)
    runtime_state.online_dist.div_(runtime_state.online_dist.sum().clamp_min(1.0e-12))
    torch.cumsum(
        runtime_state.online_dist,
        dim=0,
        out=runtime_state.online_cdf,
    )
    runtime_state.online_cdf[-1:].fill_(1.0)


def initialize_motion_plr_state(self) -> None:
    """Allocate persistent PLR state and precompute dataset-size-specific ranks."""
    runtime_state = self.adaptive_sampling_state
    ensure_adaptive_update_work_buffers(self)
    unit_count = int(runtime_state.unit_count)
    device = runtime_state.online_dist.device
    exponent = solve_rank_exponent_for_target_ess(
        unit_count=unit_count,
        target_ess_ratio=float(runtime_state.plr_focus_ess_ratio),
    )
    runtime_state.plr_rank_exponent = float(exponent)
    rank_weights = runtime_state.update_scaled_metric
    torch.arange(
        1,
        unit_count + 1,
        device=device,
        dtype=torch.float32,
        out=rank_weights,
    )
    rank_weights.pow_(-float(exponent))
    rank_weights.div_(rank_weights.sum().clamp_min(1.0e-12))

    runtime_state.plr_seen = torch.zeros(
        unit_count,
        device=device,
        dtype=torch.bool,
    )
    runtime_state.plr_last_seen = torch.zeros(
        unit_count,
        device=device,
        dtype=torch.float32,
    )
    runtime_state.plr_update_count = torch.zeros(
        (),
        device=device,
        dtype=torch.float32,
    )
    runtime_state.plr_has_mass = torch.empty((), device=device, dtype=torch.bool)
    runtime_state.plr_sorted_indices = torch.empty(
        unit_count,
        device=device,
        dtype=torch.long,
    )
    runtime_state.plr_advantage_magnitude = None
    runtime_state.plr_sample_weights = None
    runtime_state.plr_level_ids = None
    runtime_state.plr_returns = None
    runtime_state.plr_values = None


def _bind_motion_plr_rollout_inputs(
    self,
    *,
    level_ids: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
) -> None:
    runtime_state = self.adaptive_sampling_state
    if level_ids.shape != returns.shape[:2]:
        raise ValueError(
            "motion PLR level_ids must match the rollout and environment dimensions: "
            f"got level_ids={tuple(level_ids.shape)}, returns={tuple(returns.shape)}"
        )
    if returns.shape != values.shape or returns.shape[-1:] != (1,):
        raise ValueError(
            "motion PLR returns and values must have matching [rollout, env, 1] "
            f"shapes; got returns={tuple(returns.shape)}, values={tuple(values.shape)}"
        )
    if (
        level_ids.device != runtime_state.online_dist.device
        or returns.device != runtime_state.online_dist.device
        or values.device != runtime_state.online_dist.device
    ):
        raise ValueError("motion PLR rollout tensors must be on the environment device")
    if level_ids.dtype != torch.long:
        raise ValueError("motion PLR level_ids must use torch.long")

    inputs_changed = (
        runtime_state.plr_level_ids is not level_ids
        or runtime_state.plr_returns is not returns
        or runtime_state.plr_values is not values
    )
    needs_allocate = (
        runtime_state.plr_advantage_magnitude is None
        or runtime_state.plr_advantage_magnitude.shape != returns.shape
        or runtime_state.plr_advantage_magnitude.device != returns.device
    )
    if inputs_changed:
        runtime_state.sampling_update_graph = None
    if needs_allocate:
        runtime_state.plr_advantage_magnitude = torch.empty_like(
            returns,
            dtype=torch.float32,
        )
        runtime_state.plr_sample_weights = torch.ones(
            level_ids.numel(),
            device=level_ids.device,
            dtype=torch.float32,
        )

    runtime_state.plr_level_ids = level_ids
    runtime_state.plr_returns = returns
    runtime_state.plr_values = values


def launch_motion_plr_distribution_update(self) -> None:
    """Update the PLR distribution using only preallocated device tensors."""
    runtime_state = self.adaptive_sampling_state
    ensure_adaptive_update_work_buffers(self)
    level_ids = runtime_state.plr_level_ids
    returns = runtime_state.plr_returns
    values = runtime_state.plr_values
    advantage_magnitude = runtime_state.plr_advantage_magnitude
    sample_weights = runtime_state.plr_sample_weights
    if (
        level_ids is None
        or returns is None
        or values is None
        or advantage_magnitude is None
        or sample_weights is None
    ):
        raise RuntimeError("motion PLR rollout inputs are not bound")

    metric_sum = runtime_state.metric_sum
    metric_count = runtime_state.metric_count
    observed = runtime_state.update_count_active
    mean_score = runtime_state.update_mean_metric
    scores = runtime_state.exp_metric
    rank_weights = runtime_state.update_scaled_metric
    sorted_scores = runtime_state.update_next_metric
    sorted_indices = runtime_state.plr_sorted_indices
    score_dist = runtime_state.update_weights
    age_dist = runtime_state.update_decayed_metric
    candidate_dist = runtime_state.update_updated_metric
    next_dist = runtime_state.update_next_dist
    norm = runtime_state.update_norm
    norm_clamped = runtime_state.update_norm_clamped
    any_count = runtime_state.update_any_count
    has_mass = runtime_state.plr_has_mass

    torch.sub(returns, values, out=advantage_magnitude)
    advantage_magnitude.abs_()
    advantage_magnitude.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    metric_sum.zero_()
    metric_count.zero_()
    flat_level_ids = level_ids.reshape(-1)
    metric_sum.scatter_add_(0, flat_level_ids, advantage_magnitude.reshape(-1))
    metric_count.scatter_add_(0, flat_level_ids, sample_weights)
    torch.gt(metric_count, 0.0, out=observed)
    torch.any(observed, out=any_count)
    torch.clamp(metric_count, min=1.0, out=mean_score)
    torch.div(metric_sum, mean_score, out=mean_score)
    torch.where(observed, mean_score, scores, out=scores)
    scores.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    runtime_state.plr_update_count.add_(1.0)
    torch.where(
        observed,
        runtime_state.plr_update_count,
        runtime_state.plr_last_seen,
        out=runtime_state.plr_last_seen,
    )
    torch.logical_or(
        runtime_state.plr_seen,
        observed,
        out=runtime_state.plr_seen,
    )

    torch.sort(
        scores,
        descending=True,
        stable=False,
        out=(sorted_scores, sorted_indices),
    )
    score_dist.zero_()
    score_dist.scatter_(0, sorted_indices, rank_weights)

    torch.sub(
        runtime_state.plr_update_count,
        runtime_state.plr_last_seen,
        out=age_dist,
    )
    age_dist.clamp_(min=0.0)
    torch.sum(age_dist, dim=0, out=norm)
    torch.gt(norm, 0.0, out=has_mass)
    torch.clamp(norm, min=1.0e-12, out=norm_clamped)
    age_dist.div_(norm_clamped)
    torch.where(has_mass, age_dist, score_dist, out=age_dist)
    candidate_dist.copy_(score_dist)
    candidate_dist.mul_(1.0 - float(runtime_state.plr_staleness_mix))
    candidate_dist.add_(
        age_dist,
        alpha=float(runtime_state.plr_staleness_mix),
    )

    torch.logical_not(runtime_state.plr_seen, out=observed)
    next_dist.copy_(observed)
    torch.sum(next_dist, dim=0, out=norm)
    torch.gt(norm, 0.0, out=has_mass)
    torch.clamp(norm, min=1.0e-12, out=norm_clamped)
    next_dist.div_(norm_clamped)
    torch.where(has_mass, next_dist, candidate_dist, out=next_dist)
    torch.where(any_count, next_dist, runtime_state.online_dist, out=next_dist)
    next_dist.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    torch.sum(next_dist, dim=0, out=norm)
    torch.clamp(norm, min=1.0e-12, out=norm_clamped)
    next_dist.div_(norm_clamped)
    runtime_state.online_dist.copy_(next_dist)
    torch.cumsum(
        runtime_state.online_dist,
        dim=0,
        out=runtime_state.online_cdf,
    )
    runtime_state.online_cdf[-1:].fill_(1.0)

    metric_sum.zero_()
    metric_count.zero_()
    runtime_state.metric_event_count.zero_()
    blend_adaptive_sampling_distribution_torch(self)


def launch_sonic_failure_distribution_update(self) -> None:
    """Recompute SONIC probabilities from cumulative device-side statistics."""
    runtime_state = self.adaptive_sampling_state
    ensure_adaptive_update_work_buffers(self)
    failure_rate = runtime_state.exp_metric
    clipped_failure_rate = runtime_state.update_weights
    next_dist = runtime_state.update_next_dist
    failure_rate_upper_bound = runtime_state.update_max_metric
    norm = runtime_state.update_norm
    norm_clamped = runtime_state.update_norm_clamped

    runtime_state.sonic_num_failures.nan_to_num_(nan=1.0, posinf=1.0, neginf=0.0)
    runtime_state.sonic_num_failures.clamp_(min=0.0)
    runtime_state.sonic_num_exposures.nan_to_num_(nan=1.0, posinf=1.0, neginf=1.0)
    runtime_state.sonic_num_exposures.clamp_(min=1.0e-12)
    torch.div(
        runtime_state.sonic_num_failures,
        runtime_state.sonic_num_exposures,
        out=failure_rate,
    )
    failure_rate.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    torch.mean(failure_rate, dim=0, out=failure_rate_upper_bound)
    failure_rate_upper_bound.mul_(
        max(0.0, float(runtime_state.sonic_failure_rate_max_over_mean))
    )
    torch.minimum(failure_rate, failure_rate_upper_bound, out=clipped_failure_rate)
    torch.sum(clipped_failure_rate, dim=0, out=norm)
    torch.clamp(norm, min=1.0e-12, out=norm_clamped)
    clipped_failure_rate.div_(norm_clamped)

    uniform_rate = min(
        1.0,
        max(0.0, float(runtime_state.sonic_uniform_sampling_rate)),
    )
    next_dist.copy_(clipped_failure_rate)
    next_dist.mul_(1.0 - uniform_rate)
    next_dist.add_(uniform_rate / float(runtime_state.unit_count))
    next_dist.mul_(runtime_state.sonic_bin_weight)
    next_dist.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    torch.sum(next_dist, dim=0, out=norm)
    torch.clamp(norm, min=1.0e-12, out=norm_clamped)
    next_dist.div_(norm_clamped)
    runtime_state.online_dist.copy_(next_dist)
    torch.cumsum(
        runtime_state.online_dist,
        dim=0,
        out=runtime_state.online_cdf,
    )
    runtime_state.online_cdf[-1:].fill_(1.0)
    blend_adaptive_sampling_distribution_torch(self)


def launch_adaptive_sampling_distribution_update(self) -> None:
    runtime_state = self.adaptive_sampling_state
    ensure_adaptive_update_work_buffers(self)
    event_active = runtime_state.update_event_active
    any_count = runtime_state.update_any_count
    should_update = runtime_state.update_should_update
    count_active = runtime_state.update_count_active
    decay_active = runtime_state.update_decay_active
    scaled_metric = runtime_state.update_scaled_metric
    decayed_metric = runtime_state.update_decayed_metric
    mean_metric = runtime_state.update_mean_metric
    updated_metric = runtime_state.update_updated_metric
    next_metric = runtime_state.update_next_metric
    weights = runtime_state.update_weights
    next_dist = runtime_state.update_next_dist
    max_metric = runtime_state.update_max_metric
    norm = runtime_state.update_norm
    norm_clamped = runtime_state.update_norm_clamped

    runtime_state.metric_sum.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    runtime_state.metric_count.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    runtime_state.metric_event_count.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    runtime_state.exp_metric.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    torch.gt(runtime_state.metric_event_count, 0.0, out=event_active)
    torch.gt(runtime_state.metric_count, 0.0, out=count_active)
    torch.any(count_active, out=any_count)
    torch.logical_and(event_active, any_count, out=should_update)
    alpha = max(0.0, min(1.0, float(runtime_state.ema_alpha)))

    if runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX_SCALED:
        torch.sum(runtime_state.metric_event_count, dim=0, out=norm)
        norm.mul_(
            -1.0
            / (
                float(runtime_state.unit_count)
                * float(runtime_state.scaled_stale_half_life_passes)
            )
        )
        norm.exp2_()
        torch.gt(
            runtime_state.exp_metric,
            float(runtime_state.min_metric),
            out=decay_active,
        )
        torch.mul(
            runtime_state.exp_metric,
            norm,
            out=scaled_metric,
        )
        torch.where(
            decay_active,
            scaled_metric,
            runtime_state.exp_metric,
            out=decayed_metric,
        )
    elif float(runtime_state.decay_rate) < 1.0:
        torch.gt(
            runtime_state.exp_metric,
            float(runtime_state.min_metric),
            out=decay_active,
        )
        torch.mul(
            runtime_state.exp_metric,
            float(runtime_state.decay_rate),
            out=scaled_metric,
        )
        torch.where(
            decay_active,
            scaled_metric,
            runtime_state.exp_metric,
            out=decayed_metric,
        )
    else:
        decayed_metric.copy_(runtime_state.exp_metric)

    torch.clamp(
        runtime_state.metric_count,
        min=1.0,
        out=mean_metric,
    )
    torch.div(runtime_state.metric_sum, mean_metric, out=mean_metric)
    torch.mul(decayed_metric, 1.0 - alpha, out=updated_metric)
    updated_metric.add_(mean_metric, alpha=alpha)
    torch.where(
        count_active,
        updated_metric,
        decayed_metric,
        out=updated_metric,
    )
    torch.where(
        event_active,
        updated_metric,
        runtime_state.exp_metric,
        out=next_metric,
    )
    runtime_state.exp_metric.copy_(next_metric)
    runtime_state.exp_metric.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    torch.max(runtime_state.exp_metric, out=max_metric)
    torch.sub(max_metric, runtime_state.exp_metric, out=weights)
    weights.mul_(float(runtime_state.temperature))
    torch.exp(weights, out=weights)
    torch.sum(weights, dim=0, out=norm)
    torch.clamp(norm, min=1.0e-12, out=norm_clamped)
    weights.div_(norm_clamped)
    weights.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    torch.sum(weights, dim=0, out=norm)
    torch.clamp(norm, min=1.0e-12, out=norm_clamped)
    weights.div_(norm_clamped)
    torch.where(
        should_update,
        weights,
        runtime_state.online_dist,
        out=next_dist,
    )
    runtime_state.online_dist.copy_(next_dist)
    torch.cumsum(
        runtime_state.online_dist,
        dim=0,
        out=runtime_state.online_cdf,
    )
    runtime_state.online_cdf[-1:].fill_(1.0)

    runtime_state.metric_sum.zero_()
    runtime_state.metric_count.zero_()
    runtime_state.metric_event_count.zero_()
    blend_adaptive_sampling_distribution_torch(self)


def update_adaptive_sampling_distribution_after_rollout(
    self,
    *,
    wp_stream: Any | None = None,
    torch_stream: torch.cuda.Stream | None = None,
) -> None:
    runtime_state = self.adaptive_sampling_state
    if not runtime_state.sampling_enabled:
        return
    method = runtime_state.sampling_method
    if method not in (
        ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX,
        ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX_SCALED,
        ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE,
    ):
        return
    launch_update = (
        launch_sonic_failure_distribution_update
        if method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE
        else launch_adaptive_sampling_distribution_update
    )

    use_cuda_graph = runtime_state.update_cuda_graph
    if self.device.type == "cuda" and use_cuda_graph:
        graph = runtime_state.sampling_update_graph
        if torch_stream is None and wp_stream is None:
            torch_stream = runtime_state.update_torch_stream
            wp_stream = runtime_state.update_wp_stream
            if torch_stream is None or wp_stream is None:
                torch_stream, wp_stream = make_stream_aligned_cuda_graph_streams(
                    self.device
                )
                runtime_state.update_torch_stream = torch_stream
                runtime_state.update_wp_stream = wp_stream
        elif torch_stream is None:
            torch_stream = torch.cuda.current_stream(self.device)
        elif wp_stream is None:
            wp_stream = wp.stream_from_torch(torch_stream)
        torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        if graph is None:
            with wp.ScopedStream(wp_stream, sync_enter=False, sync_exit=False):
                with torch.cuda.stream(torch_stream):
                    with capture_warp_graph(stream=wp_stream) as capture:
                        launch_update(self)
            graph = capture.graph
            runtime_state.sampling_update_graph = graph
        with wp.ScopedStream(wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(torch_stream):
                wp.capture_launch(graph, stream=wp_stream)
        return

    launch_update(self)


def update_motion_plr_distribution_after_policy_update(
    self,
    *,
    level_ids: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    wp_stream: Any | None = None,
    torch_stream: torch.cuda.Stream | None = None,
) -> None:
    runtime_state = self.adaptive_sampling_state
    if not runtime_state.sampling_enabled:
        return
    if runtime_state.sampling_method != ADAPTIVE_SAMPLING_METHOD_MOTION_PLR:
        return

    _bind_motion_plr_rollout_inputs(
        self,
        level_ids=level_ids,
        returns=returns,
        values=values,
    )
    use_cuda_graph = runtime_state.update_cuda_graph
    if self.device.type == "cuda" and use_cuda_graph:
        graph = runtime_state.sampling_update_graph
        if torch_stream is None and wp_stream is None:
            torch_stream = runtime_state.update_torch_stream
            wp_stream = runtime_state.update_wp_stream
            if torch_stream is None or wp_stream is None:
                torch_stream, wp_stream = make_stream_aligned_cuda_graph_streams(
                    self.device
                )
                runtime_state.update_torch_stream = torch_stream
                runtime_state.update_wp_stream = wp_stream
        elif torch_stream is None:
            torch_stream = torch.cuda.current_stream(self.device)
        elif wp_stream is None:
            wp_stream = wp.stream_from_torch(torch_stream)
        torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        if graph is None:
            with wp.ScopedStream(wp_stream, sync_enter=False, sync_exit=False):
                with torch.cuda.stream(torch_stream):
                    with capture_warp_graph(stream=wp_stream) as capture:
                        launch_motion_plr_distribution_update(self)
            graph = capture.graph
            runtime_state.sampling_update_graph = graph
        with wp.ScopedStream(wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(torch_stream):
                wp.capture_launch(graph, stream=wp_stream)
        return

    if self.device.type == "cuda" and torch_stream is not None:
        torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(torch_stream):
            launch_motion_plr_distribution_update(self)
        return

    launch_motion_plr_distribution_update(self)


def _checkpoint_int_tuple(value: object, *, fallback: int) -> tuple[int, ...]:
    if value is None:
        return (int(fallback),)
    if isinstance(value, torch.Tensor):
        values = value.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
    elif isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=np.int64).reshape(-1).tolist()
    else:
        raise TypeError("Adaptive-sampling motion partition must be array-like.")
    return tuple(int(item) for item in values)


def _adaptive_sampling_semantics(self) -> dict[str, object]:
    runtime_state = self.adaptive_sampling_state
    semantics: dict[str, object] = {
        "method": runtime_state.sampling_method,
        "metric_type": (
            "failure"
            if int(runtime_state.metric_type) == 2
            else ("length" if int(runtime_state.metric_type) == 1 else "reward")
        ),
        "blame_mode": (
            "terminal_window" if int(runtime_state.blame_mode) == 1 else "episode_start"
        ),
        "chunk_size": int(runtime_state.chunk_size),
        "unit_count": int(runtime_state.unit_count),
        "motion_frame_count": int(self.motion_frame_count),
        "multi_motion_enabled": bool(getattr(self, "_multi_motion_enabled", False)),
        "motion_clip_lengths": _checkpoint_int_tuple(
            getattr(self, "motion_clip_length", None),
            fallback=int(self.motion_frame_count),
        ),
        "motion_clip_unit_counts": _checkpoint_int_tuple(
            getattr(self, "motion_clip_unit_count", None),
            fallback=int(runtime_state.unit_count),
        ),
        "ema_alpha": float(runtime_state.ema_alpha),
        "temperature": float(runtime_state.temperature),
        "decay_rate": float(runtime_state.decay_rate),
        "random_sample_prob": float(runtime_state.random_sample_prob),
        "min_metric": float(runtime_state.min_metric),
        "terminal_window_chunk_count": int(runtime_state.terminal_window_chunk_count),
        "benchmark_beta": float(runtime_state.benchmark_beta),
    }
    if runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_MOTION_PLR:
        semantics.update(
            {
                "focus_ess_ratio": float(runtime_state.plr_focus_ess_ratio),
                "staleness_mix": float(runtime_state.plr_staleness_mix),
                "rank_exponent": float(runtime_state.plr_rank_exponent),
            }
        )
    elif (
        runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX_SCALED
    ):
        semantics.update(
            {
                "focus_strength": float(runtime_state.scaled_focus_strength),
                "score_scale": float(runtime_state.scaled_score_scale),
                "stale_half_life_passes": float(
                    runtime_state.scaled_stale_half_life_passes
                ),
            }
        )
    elif runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE:
        semantics.update(
            {
                "uniform_sampling_rate": float(
                    runtime_state.sonic_uniform_sampling_rate
                ),
                "failure_rate_max_over_mean": float(
                    runtime_state.sonic_failure_rate_max_over_mean
                ),
                "pre_failure_window_steps": int(
                    runtime_state.sonic_pre_failure_window_steps
                ),
            }
        )
    return semantics


def get_adaptive_sampling_state(self) -> dict[str, object] | None:
    runtime_state = self.adaptive_sampling_state
    if not runtime_state.sampling_enabled:
        return None

    def _checkpoint_clone(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().clone()

    state: dict[str, object] = {
        "schema_version": _ADAPTIVE_SAMPLING_CHECKPOINT_SCHEMA_VERSION,
        "enabled": True,
        **_adaptive_sampling_semantics(self),
        "benchmark_prior_enabled": bool(runtime_state.benchmark_prior_enabled),
        "exp_metric": _checkpoint_clone(runtime_state.exp_metric),
        "online_dist": _checkpoint_clone(runtime_state.online_dist),
        "benchmark_prior": _checkpoint_clone(runtime_state.benchmark_prior),
        "dist": _checkpoint_clone(runtime_state.dist),
    }
    if runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_MOTION_PLR:
        state.update(
            {
                "seen": _checkpoint_clone(runtime_state.plr_seen),
                "last_seen": _checkpoint_clone(runtime_state.plr_last_seen),
                "update_count": _checkpoint_clone(runtime_state.plr_update_count),
            }
        )
    elif runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE:
        state.update(
            {
                "num_failures": _checkpoint_clone(runtime_state.sonic_num_failures),
                "num_exposures": _checkpoint_clone(runtime_state.sonic_num_exposures),
            }
        )
    return state


def _checkpoint_semantic_values_match(saved: object, current: object) -> bool:
    if isinstance(current, float):
        if isinstance(saved, bool) or not isinstance(saved, Real):
            return False
        return math.isclose(float(saved), current, rel_tol=1.0e-9, abs_tol=1.0e-12)
    return type(saved) is type(current) and saved == current


def _validate_adaptive_sampling_semantics(
    env,
    adaptive_state: dict[str, object],
) -> None:
    schema_version = adaptive_state.get("schema_version", 0)
    if type(schema_version) is not int or schema_version not in (0, 1):
        raise ValueError(
            "Adaptive-sampling checkpoint field 'schema_version' is incompatible: "
            f"checkpoint={schema_version!r}, "
            f"current={_ADAPTIVE_SAMPLING_CHECKPOINT_SCHEMA_VERSION}."
        )

    mismatches: list[str] = []
    current_semantics = _adaptive_sampling_semantics(env)
    if schema_version == 0:
        for field in (
            "multi_motion_enabled",
            "motion_clip_lengths",
            "motion_clip_unit_counts",
        ):
            current_semantics.pop(field)
    for field, current_value in current_semantics.items():
        if field not in adaptive_state:
            mismatches.append(f"{field}: missing (current={current_value!r})")
            continue
        saved_value = adaptive_state[field]
        if not _checkpoint_semantic_values_match(saved_value, current_value):
            mismatches.append(
                f"{field}: checkpoint={saved_value!r}, current={current_value!r}"
            )
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise ValueError(
            f"Adaptive-sampling checkpoint semantics are incompatible:\n  - {details}"
        )


def _checkpoint_tensor(
    adaptive_state: dict[str, object],
    *,
    field: str,
    env,
    dtype: torch.dtype,
    expected_shape: tuple[int, ...],
    boolean: bool = False,
) -> torch.Tensor:
    if field not in adaptive_state:
        raise ValueError(f"Adaptive-sampling checkpoint field {field!r} is missing.")
    try:
        source = torch.as_tensor(adaptive_state[field])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Adaptive-sampling checkpoint field {field!r} is not a valid tensor."
        ) from exc
    if tuple(source.shape) != expected_shape:
        raise ValueError(
            f"Adaptive-sampling checkpoint field {field!r} shape mismatch: "
            f"expected {expected_shape}, got {tuple(source.shape)}."
        )
    if boolean and source.dtype is not torch.bool:
        raise ValueError(
            f"Adaptive-sampling checkpoint field {field!r} must be boolean."
        )
    if not boolean and source.dtype is torch.bool:
        raise ValueError(
            f"Adaptive-sampling checkpoint field {field!r} must be numeric."
        )
    return source.to(device=env.device, dtype=dtype).detach().clone()


def _validate_finite_checkpoint_tensor(
    tensor: torch.Tensor,
    *,
    field: str,
    nonnegative: bool = False,
) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(
            f"Adaptive-sampling checkpoint field {field!r} must be finite."
        )
    if nonnegative and bool((tensor < 0.0).any().item()):
        raise ValueError(
            f"Adaptive-sampling checkpoint field {field!r} must be non-negative."
        )


def _prepare_adaptive_sampling_restore(
    env,
    adaptive_state: dict[str, object],
) -> dict[str, Any]:
    runtime_state = env.adaptive_sampling_state
    _validate_adaptive_sampling_semantics(env, adaptive_state)
    unit_count = int(runtime_state.unit_count)
    prepared: dict[str, Any] = {
        "exp_metric": _checkpoint_tensor(
            adaptive_state,
            field="exp_metric",
            env=env,
            dtype=torch.float32,
            expected_shape=(unit_count,),
        ),
        "online_dist": _checkpoint_tensor(
            adaptive_state,
            field="online_dist",
            env=env,
            dtype=torch.float32,
            expected_shape=(unit_count,),
        ),
        "benchmark_prior": _checkpoint_tensor(
            adaptive_state,
            field="benchmark_prior",
            env=env,
            dtype=torch.float32,
            expected_shape=(unit_count,),
        ),
        "dist": _checkpoint_tensor(
            adaptive_state,
            field="dist",
            env=env,
            dtype=torch.float32,
            expected_shape=(unit_count,),
        ),
    }
    _validate_finite_checkpoint_tensor(prepared["exp_metric"], field="exp_metric")
    _validate_finite_checkpoint_tensor(
        prepared["online_dist"],
        field="online_dist",
        nonnegative=True,
    )
    if float(prepared["online_dist"].sum().item()) <= 0.0:
        raise ValueError(
            "Adaptive-sampling checkpoint field 'online_dist' must have positive mass."
        )
    _validate_finite_checkpoint_tensor(
        prepared["benchmark_prior"],
        field="benchmark_prior",
        nonnegative=True,
    )
    _validate_finite_checkpoint_tensor(
        prepared["dist"],
        field="dist",
        nonnegative=True,
    )
    if float(prepared["dist"].sum().item()) <= 0.0:
        raise ValueError(
            "Adaptive-sampling checkpoint field 'dist' must have positive mass."
        )

    benchmark_prior_enabled = adaptive_state.get("benchmark_prior_enabled")
    if type(benchmark_prior_enabled) is not bool:
        raise ValueError(
            "Adaptive-sampling checkpoint field 'benchmark_prior_enabled' must be "
            "a bool."
        )
    if (
        benchmark_prior_enabled
        and float(prepared["benchmark_prior"].sum().item()) <= 0.0
    ):
        raise ValueError(
            "Adaptive-sampling checkpoint field 'benchmark_prior' must have positive "
            "mass when benchmark_prior_enabled is true."
        )
    prepared["benchmark_prior_enabled"] = benchmark_prior_enabled

    if runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_MOTION_PLR:
        prepared["seen"] = _checkpoint_tensor(
            adaptive_state,
            field="seen",
            env=env,
            dtype=torch.bool,
            expected_shape=(unit_count,),
            boolean=True,
        )
        prepared["last_seen"] = _checkpoint_tensor(
            adaptive_state,
            field="last_seen",
            env=env,
            dtype=torch.float32,
            expected_shape=(unit_count,),
        )
        prepared["update_count"] = _checkpoint_tensor(
            adaptive_state,
            field="update_count",
            env=env,
            dtype=torch.float32,
            expected_shape=(),
        )
        _validate_finite_checkpoint_tensor(prepared["last_seen"], field="last_seen")
        _validate_finite_checkpoint_tensor(
            prepared["update_count"],
            field="update_count",
            nonnegative=True,
        )
        if not bool(
            torch.equal(prepared["update_count"], prepared["update_count"].floor())
        ):
            raise ValueError(
                "Adaptive-sampling checkpoint field 'update_count' must be an integer."
            )
        if bool((prepared["last_seen"] < 0.0).any().item()):
            raise ValueError(
                "Adaptive-sampling checkpoint field 'last_seen' must be non-negative."
            )
        if not bool(torch.equal(prepared["last_seen"], prepared["last_seen"].floor())):
            raise ValueError(
                "Adaptive-sampling checkpoint field 'last_seen' must contain "
                "integer values."
            )
        if bool((prepared["last_seen"] > prepared["update_count"]).any().item()):
            raise ValueError(
                "Adaptive-sampling checkpoint field 'last_seen' must not exceed "
                "'update_count'."
            )
    elif runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE:
        for field in ("num_failures", "num_exposures"):
            tensor = _checkpoint_tensor(
                adaptive_state,
                field=field,
                env=env,
                dtype=torch.float32,
                expected_shape=(unit_count,),
            )
            _validate_finite_checkpoint_tensor(
                tensor,
                field=field,
                nonnegative=True,
            )
            prepared[field] = tensor
    online_dist = prepared["online_dist"]
    online_dist = online_dist / online_dist.sum()
    prepared["online_dist"] = online_dist
    prepared["online_cdf"] = torch.cumsum(online_dist, dim=0)
    prepared["online_cdf"][-1] = 1.0

    benchmark_prior = prepared["benchmark_prior"]
    if benchmark_prior_enabled:
        benchmark_prior = benchmark_prior / benchmark_prior.sum()
    else:
        benchmark_prior = torch.zeros_like(benchmark_prior)
    prepared["benchmark_prior"] = benchmark_prior
    use_prior = benchmark_prior_enabled and float(runtime_state.benchmark_beta) > 0.0
    if use_prior:
        beta = float(runtime_state.benchmark_beta)
        runtime_dist = online_dist * (1.0 - beta) + benchmark_prior * beta
        runtime_dist = runtime_dist.clamp_min(0.0)
        runtime_dist = runtime_dist / runtime_dist.sum()
    else:
        runtime_dist = online_dist
    saved_dist = prepared.pop("dist")
    if not torch.allclose(saved_dist, runtime_dist, rtol=1.0e-5, atol=1.0e-7):
        raise ValueError(
            "Adaptive-sampling checkpoint field 'dist' is inconsistent with "
            "online_dist and benchmark_prior."
        )
    prepared["runtime_dist"] = runtime_dist
    prepared["runtime_cdf"] = torch.cumsum(runtime_dist, dim=0)
    prepared["runtime_cdf"][-1] = 1.0
    prepared["use_prior"] = use_prior
    return prepared


def get_checkpoint_state(env) -> dict[str, object]:
    state = DirectRLEnv.get_checkpoint_state(env)
    reset_runtime_state = getattr(env, "reset_runtime_state", None)
    reset_seed = getattr(reset_runtime_state, "seed", None)
    if isinstance(reset_seed, torch.Tensor):
        state["snu_reset_sampling"] = {
            "schema_version": 1,
            "seed": reset_seed.detach().clone(),
        }
    adaptive_state = get_adaptive_sampling_state(env)
    if adaptive_state is not None:
        state["adaptive_sampling"] = adaptive_state
    control_state_getter = getattr(env._control_pipeline, "get_checkpoint_state", None)
    if callable(control_state_getter):
        state.update(control_state_getter())
    return state


def _prepare_reset_sampling_restore(
    env,
    state: dict[str, object] | None,
) -> torch.Tensor | None:
    if not isinstance(state, dict) or "snu_reset_sampling" not in state:
        return None
    reset_state = state["snu_reset_sampling"]
    if not isinstance(reset_state, dict):
        raise ValueError("Checkpoint field 'snu_reset_sampling' must be a mapping.")
    schema_version = reset_state.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError(
            "SNU reset-sampling checkpoint field 'schema_version' is incompatible: "
            f"checkpoint={schema_version!r}, current=1."
        )
    if "seed" not in reset_state:
        raise ValueError("SNU reset-sampling checkpoint field 'seed' is missing.")
    reset_runtime_state = getattr(env, "reset_runtime_state", None)
    destination = getattr(reset_runtime_state, "seed", None)
    if not isinstance(destination, torch.Tensor):
        raise ValueError(
            "SNU reset-sampling restore destination 'seed' must be a tensor."
        )
    try:
        source = torch.as_tensor(reset_state["seed"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            "SNU reset-sampling checkpoint field 'seed' is not a valid tensor."
        ) from exc
    if source.shape != destination.shape:
        raise ValueError(
            "SNU reset-sampling checkpoint field 'seed' shape mismatch: "
            f"expected {tuple(destination.shape)}, got {tuple(source.shape)}."
        )
    if source.dtype != destination.dtype:
        raise ValueError(
            "SNU reset-sampling checkpoint field 'seed' dtype mismatch: "
            f"expected {destination.dtype}, got {source.dtype}."
        )
    return source.to(device=destination.device).detach().clone()


def _reset_disables_adaptive_sampling(env) -> bool | None:
    """Return whether the live sampler is disabled by the effective reset mode.

    ``reset_settings`` is resolved once during environment construction and is
    therefore authoritative.  The string reason remains a compatibility
    fallback for lightweight callers that predate the resolved reset settings.
    Returning ``None`` means that no authoritative reset settings are available.
    """

    reset_settings = getattr(env, "reset_settings", None)
    if reset_settings is not None:
        if bool(getattr(reset_settings, "deterministic_reset", False)):
            return True
        random_init_enabled = getattr(reset_settings, "random_init_enabled", None)
        if random_init_enabled is not None:
            return not bool(random_init_enabled)
        return None
    return None


def _prepare_checkpoint_adaptive_sampling_state(
    env,
    state: dict[str, object] | None,
) -> dict[str, Any] | None:
    runtime_state = env.adaptive_sampling_state
    if not isinstance(state, dict) or "adaptive_sampling" not in state:
        return None
    adaptive_state = state["adaptive_sampling"]
    if not isinstance(adaptive_state, dict):
        raise ValueError("Checkpoint field 'adaptive_sampling' must be a mapping.")
    if adaptive_state.get("enabled") is not True:
        raise ValueError("Adaptive-sampling checkpoint field 'enabled' must be true.")
    if runtime_state.sampling_enabled:
        return _prepare_adaptive_sampling_restore(env, adaptive_state)

    # Deterministic playback and fixed-start evaluation have no live sampler, so
    # their explicit control-state restore remains independent of this unused state.
    reset_disables_sampling = _reset_disables_adaptive_sampling(env)
    if reset_disables_sampling is True or (
        reset_disables_sampling is None
        and runtime_state.sampling_disable_reason
        in ("reset.random_start=false", "reset.deterministic=true")
    ):
        return None
    raise ValueError(
        "Adaptive-sampling checkpoint field 'enabled' is incompatible: "
        "checkpoint=True, current=False."
    )


def _validate_adaptive_restore_destinations(
    env,
    prepared: dict[str, Any],
) -> None:
    """Reject an invalid runtime layout before the control pipeline is mutated."""

    state = env.adaptive_sampling_state
    destinations = {
        "exp_metric": (state.exp_metric, prepared["exp_metric"]),
        "online_dist": (state.online_dist, prepared["online_dist"]),
        "online_cdf": (state.online_cdf, prepared["online_cdf"]),
        "benchmark_prior": (state.benchmark_prior, prepared["benchmark_prior"]),
        "runtime_dist_buffer": (
            state.runtime_dist_buffer,
            prepared["runtime_dist"],
        ),
        "runtime_cdf_buffer": (state.runtime_cdf_buffer, prepared["runtime_cdf"]),
    }
    if state.sampling_method == ADAPTIVE_SAMPLING_METHOD_MOTION_PLR:
        destinations.update(
            {
                "plr_seen": (state.plr_seen, prepared["seen"]),
                "plr_last_seen": (state.plr_last_seen, prepared["last_seen"]),
                "plr_update_count": (
                    state.plr_update_count,
                    prepared["update_count"],
                ),
            }
        )
    elif state.sampling_method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE:
        destinations.update(
            {
                "sonic_num_failures": (
                    state.sonic_num_failures,
                    prepared["num_failures"],
                ),
                "sonic_num_exposures": (
                    state.sonic_num_exposures,
                    prepared["num_exposures"],
                ),
            }
        )

    for field, (destination, source) in destinations.items():
        if not isinstance(destination, torch.Tensor):
            raise ValueError(
                f"Adaptive-sampling restore destination {field!r} must be a tensor."
            )
        if destination.shape != source.shape:
            raise ValueError(
                f"Adaptive-sampling restore destination {field!r} shape mismatch: "
                f"expected {tuple(source.shape)}, got {tuple(destination.shape)}."
            )
        if destination.device != source.device or destination.dtype != source.dtype:
            raise ValueError(
                f"Adaptive-sampling restore destination {field!r} layout mismatch: "
                f"expected {source.dtype} on {source.device}, got "
                f"{destination.dtype} on {destination.device}."
            )

    scalar_destination = state.benchmark_prior_enabled_tensor
    if (
        not isinstance(scalar_destination, torch.Tensor)
        or scalar_destination.numel() != 1
    ):
        raise ValueError(
            "Adaptive-sampling restore destination "
            "'benchmark_prior_enabled_tensor' must be a one-element tensor."
        )

    transient_destinations = {
        "metric_sum": state.metric_sum,
        "metric_count": state.metric_count,
        "metric_event_count": state.metric_event_count,
        "episode_reward_sum": state.episode_reward_sum,
        "episode_start_unit_idx": state.episode_start_unit_idx,
        "episode_terminal_unit_idx": state.episode_terminal_unit_idx,
        "episode_terminal_clip_unit_start": state.episode_terminal_clip_unit_start,
        "episode_terminal_clip_unit_count": state.episode_terminal_clip_unit_count,
    }
    for field, destination in transient_destinations.items():
        if not isinstance(destination, torch.Tensor):
            raise ValueError(
                f"Adaptive-sampling restore destination {field!r} must be a tensor."
            )


def load_checkpoint_state(env, state: dict[str, object] | None) -> bool:
    runtime_state = env.adaptive_sampling_state
    prepared_direct_state = DirectRLEnv._prepare_checkpoint_restore(env, state)
    prepared_reset_seed = _prepare_reset_sampling_restore(env, state)
    prepared_adaptive_state = _prepare_checkpoint_adaptive_sampling_state(env, state)
    if prepared_adaptive_state is not None:
        _validate_adaptive_restore_destinations(env, prepared_adaptive_state)

    control_restored = False
    control_state_loader = getattr(env._control_pipeline, "load_checkpoint_state", None)
    if callable(control_state_loader):
        control_restored = bool(control_state_loader(state))
    if prepared_direct_state is not None:
        DirectRLEnv._apply_checkpoint_restore(env, prepared_direct_state)
    if prepared_reset_seed is not None:
        env.reset_runtime_state.seed.copy_(prepared_reset_seed)
    base_restored = prepared_direct_state is not None or prepared_reset_seed is not None
    if prepared_adaptive_state is None:
        return control_restored or base_restored

    runtime_state.exp_metric.copy_(prepared_adaptive_state["exp_metric"])
    runtime_state.online_dist.copy_(prepared_adaptive_state["online_dist"])
    runtime_state.online_cdf.copy_(prepared_adaptive_state["online_cdf"])
    if runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_MOTION_PLR:
        runtime_state.plr_seen.copy_(prepared_adaptive_state["seen"])
        runtime_state.plr_last_seen.copy_(prepared_adaptive_state["last_seen"])
        runtime_state.plr_update_count.copy_(
            prepared_adaptive_state["update_count"].reshape(())
        )
    elif runtime_state.sampling_method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE:
        runtime_state.sonic_num_failures.copy_(prepared_adaptive_state["num_failures"])
        runtime_state.sonic_num_exposures.copy_(
            prepared_adaptive_state["num_exposures"]
        )
    runtime_state.benchmark_prior.copy_(prepared_adaptive_state["benchmark_prior"])
    runtime_state.benchmark_prior_enabled = prepared_adaptive_state[
        "benchmark_prior_enabled"
    ]
    if runtime_state.benchmark_prior_enabled:
        runtime_state.benchmark_prior_enabled_tensor.fill_(1)
    else:
        runtime_state.benchmark_prior_enabled_tensor.zero_()
    runtime_state.runtime_dist_buffer.copy_(prepared_adaptive_state["runtime_dist"])
    runtime_state.runtime_cdf_buffer.copy_(prepared_adaptive_state["runtime_cdf"])
    if prepared_adaptive_state["use_prior"]:
        runtime_state.dist = runtime_state.runtime_dist_buffer
        runtime_state.cdf = runtime_state.runtime_cdf_buffer
    else:
        runtime_state.dist = runtime_state.online_dist
        runtime_state.cdf = runtime_state.online_cdf
    runtime_state.sampling_update_graph = None
    runtime_state.metric_sum.zero_()
    runtime_state.metric_count.zero_()
    runtime_state.metric_event_count.zero_()
    runtime_state.episode_reward_sum.zero_()
    runtime_state.episode_start_unit_idx.zero_()
    runtime_state.episode_terminal_unit_idx.zero_()
    runtime_state.episode_terminal_clip_unit_start.zero_()
    runtime_state.episode_terminal_clip_unit_count.fill_(1)
    return True
