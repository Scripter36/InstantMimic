from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import warp as wp

from instantmimic.envs.config_access import get_block, get_value

ADAPTIVE_METRIC_TYPE_REWARD = 0
ADAPTIVE_METRIC_TYPE_LENGTH = 1
ADAPTIVE_METRIC_TYPE_FAILURE = 2

ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX = "reward_softmax"
ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX_SCALED = "reward_softmax_scaled"
ADAPTIVE_SAMPLING_METHOD_MOTION_PLR = "motion_plr"
ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE = "sonic_failure"

ADAPTIVE_BLAME_MODE_EPISODE_START = 0
ADAPTIVE_BLAME_MODE_TERMINAL_WINDOW = 1

# Slots of the packed reset-sampling settings array. All nine are resolved once
# by `initialize_adaptive_sampling_state` and never change during a run; they
# used to be rebuilt as nine Python ints and passed as nine launch scalars on
# every reset.
RESET_SETTING_RANDOM_INIT = 0
RESET_SETTING_START_FRAME_IDX = 1
RESET_SETTING_ADAPTIVE_ENABLED = 2
RESET_SETTING_ADAPTIVE_CHUNK_SIZE = 3
RESET_SETTING_ADAPTIVE_UNIT_COUNT = 4
RESET_SETTING_PRE_FAILURE_WINDOW_STEPS = 5
RESET_SETTING_TOTAL_MOTION_FRAME_COUNT = 6
RESET_SETTING_START_HORIZON_STEPS = 7
RESET_SETTING_FIXED_MOTION_IDX = 8
RESET_SETTING_COUNT = 9


@dataclass(frozen=True)
class AdaptiveSamplingConfig:
    enabled: bool
    method: str
    metric_type: int
    blame_mode: int
    chunk_size: int
    unit_count: int
    ema_alpha: float
    temperature: float
    decay_rate: float
    random_sample_prob: float
    min_metric: float
    terminal_window_chunk_count: int
    benchmark_beta: float
    focus_ess_ratio: float
    staleness_mix: float
    focus_strength: float
    stale_half_life_passes: float
    sonic_uniform_sampling_rate: float
    sonic_failure_rate_max_over_mean: float
    sonic_pre_failure_window_steps: int
    disable_reason: str | None = None


@dataclass(frozen=True)
class AdaptiveSamplingBuffers:
    episode_start_unit_idx: torch.Tensor
    episode_terminal_unit_idx: torch.Tensor
    episode_terminal_clip_unit_start: torch.Tensor
    episode_terminal_clip_unit_count: torch.Tensor
    episode_reward_sum: torch.Tensor
    episode_length_before_reset: torch.Tensor
    metric_sum: torch.Tensor
    metric_count: torch.Tensor
    metric_event_count: torch.Tensor
    exp_metric: torch.Tensor
    online_dist: torch.Tensor
    online_cdf: torch.Tensor
    benchmark_prior: torch.Tensor
    benchmark_prior_enabled: torch.Tensor
    dist: torch.Tensor
    cdf: torch.Tensor


def resolve_adaptive_sampling_method(adaptive: Any) -> str:
    method = (
        str(
            get_value(
                adaptive,
                "method",
                ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX,
            )
        )
        .strip()
        .lower()
    )
    if method not in (
        ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX,
        ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX_SCALED,
        ADAPTIVE_SAMPLING_METHOD_MOTION_PLR,
        ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE,
    ):
        raise ValueError(
            "adaptive_sampling.method must be one of "
            "{'reward_softmax', 'reward_softmax_scaled', 'motion_plr', "
            "'sonic_failure'}"
        )
    return method


def resolve_adaptive_sampling_chunk_size(cfg: Any, *, method: str) -> int:
    adaptive = get_block(cfg, "adaptive_sampling")
    if method == ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX:
        return max(1, int(get_value(adaptive, "chunk_size", 1)))

    bin_seconds = float(get_value(adaptive, "bin_seconds", 1.0))
    if not bin_seconds > 0.0:
        raise ValueError("adaptive_sampling.bin_seconds must be greater than zero")
    sim = get_block(cfg, "sim", required=False)
    sim_hz_value = get_value(sim, "hz", None) if sim is not None else None
    decimation_value = get_value(cfg, "decimation", None)
    if sim_hz_value is None or decimation_value is None:
        raise ValueError(
            "adaptive_sampling.method requires env.sim.hz and "
            "env.decimation to resolve bin_seconds"
        )
    sim_hz = float(sim_hz_value)
    decimation = int(decimation_value)
    if sim_hz <= 0.0 or decimation <= 0:
        raise ValueError("env.sim.hz and env.decimation must be greater than zero")
    return max(1, int(round(bin_seconds * sim_hz / float(decimation))))


@lru_cache(maxsize=128)
def solve_rank_exponent_for_target_ess(
    *,
    unit_count: int,
    target_ess_ratio: float,
) -> float:
    """Solve ``rank**(-exponent)`` for a target effective-sample ratio.

    The solution depends only on the number of adaptive units and the target,
    so runtime callers can compute it once and reuse fixed rank weights.
    """
    count = int(unit_count)
    target_ratio = float(target_ess_ratio)
    if count <= 0:
        raise ValueError("unit_count must be greater than zero")
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError("target_ess_ratio must be in (0, 1]")
    if count == 1 or target_ratio >= 1.0:
        return 0.0

    target_ess = max(1.0, target_ratio * float(count))
    log_ranks = np.log(np.arange(1, count + 1, dtype=np.float64))

    def _effective_sample_size(exponent: float) -> float:
        weights = np.exp(-float(exponent) * log_ranks)
        weight_sum = float(weights.sum())
        squared_sum = float(np.square(weights).sum())
        return weight_sum * weight_sum / max(squared_sum, np.finfo(np.float64).tiny)

    low = 0.0
    high = 1.0
    while high < 64.0 and _effective_sample_size(high) > target_ess:
        high *= 2.0
    if _effective_sample_size(high) > target_ess:
        return high

    for _ in range(40):
        midpoint = 0.5 * (low + high)
        if _effective_sample_size(midpoint) > target_ess:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def resolve_adaptive_sampling_config(
    cfg: Any,
    *,
    motion_frame_count: int,
    random_init_enabled: bool,
    deterministic_reset: bool,
) -> AdaptiveSamplingConfig:
    adaptive = get_block(cfg, "adaptive_sampling")
    enabled = bool(get_value(adaptive, "enabled", False))
    method = resolve_adaptive_sampling_method(adaptive)
    focus_ess_ratio = float(get_value(adaptive, "focus_ess_ratio", 0.25))
    if not 0.0 < focus_ess_ratio <= 1.0:
        raise ValueError("adaptive_sampling.focus_ess_ratio must be in (0, 1]")
    staleness_mix = float(get_value(adaptive, "staleness_mix", 0.1))
    if not 0.0 <= staleness_mix <= 1.0:
        raise ValueError("adaptive_sampling.staleness_mix must be in [0, 1]")
    metric_name = str(get_value(adaptive, "metric", "reward")).lower()
    if metric_name == "reward":
        metric_type = ADAPTIVE_METRIC_TYPE_REWARD
    elif metric_name == "length":
        metric_type = ADAPTIVE_METRIC_TYPE_LENGTH
    else:
        raise ValueError("adaptive_sampling.metric must be one of {'reward', 'length'}")
    if method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE:
        metric_type = ADAPTIVE_METRIC_TYPE_FAILURE
    blame_mode_name = str(get_value(adaptive, "blame", "episode_start")).lower()
    if blame_mode_name == "episode_start":
        blame_mode = ADAPTIVE_BLAME_MODE_EPISODE_START
    elif blame_mode_name == "terminal_window":
        blame_mode = ADAPTIVE_BLAME_MODE_TERMINAL_WINDOW
    else:
        raise ValueError(
            "adaptive_sampling.blame must be one of "
            "{'episode_start', 'terminal_window'}"
        )
    chunk_size = resolve_adaptive_sampling_chunk_size(cfg, method=method)
    unit_count = max(1, (motion_frame_count + chunk_size - 1) // chunk_size)
    ema_alpha = min(
        1.0,
        max(0.0, float(get_value(adaptive, "ema_alpha", 0.3))),
    )
    temperature = max(0.0, float(get_value(adaptive, "temperature", 0.1)))
    decay_rate = min(1.0, max(0.0, float(get_value(adaptive, "decay_rate", 0.995))))
    random_sample_prob = min(
        1.0,
        max(0.0, float(get_value(adaptive, "random_sample_prob", 0.0))),
    )
    min_metric = max(0.0, float(get_value(adaptive, "min_metric", 1.0e-6)))
    terminal_window_chunk_count = max(
        1,
        int(get_value(adaptive, "terminal_window_chunks", 4)),
    )
    benchmark_beta = min(
        1.0,
        max(0.0, float(get_value(adaptive, "benchmark_beta", 0.0))),
    )
    focus_strength = float(get_value(adaptive, "focus_strength", 24.0))
    if method == ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX_SCALED and (
        not np.isfinite(focus_strength) or focus_strength < 0.0
    ):
        raise ValueError(
            "adaptive_sampling.focus_strength must be finite and non-negative"
        )
    stale_half_life_passes = float(get_value(adaptive, "stale_half_life_passes", 400.0))
    if method == ADAPTIVE_SAMPLING_METHOD_REWARD_SOFTMAX_SCALED and (
        not np.isfinite(stale_half_life_passes) or stale_half_life_passes <= 0.0
    ):
        raise ValueError(
            "adaptive_sampling.stale_half_life_passes must be finite and greater "
            "than zero"
        )
    sonic_uniform_sampling_rate = float(
        get_value(adaptive, "uniform_sampling_rate", 0.1)
    )
    if not 0.0 <= sonic_uniform_sampling_rate <= 1.0:
        raise ValueError("adaptive_sampling.uniform_sampling_rate must be in [0, 1]")
    sonic_failure_rate_max_over_mean = float(
        get_value(adaptive, "failure_rate_max_over_mean", 200.0)
    )
    if sonic_failure_rate_max_over_mean <= 0.0:
        raise ValueError(
            "adaptive_sampling.failure_rate_max_over_mean must be greater than zero"
        )
    pre_failure_seconds = float(get_value(adaptive, "pre_failure_seconds", 4.0))
    if pre_failure_seconds < 0.0:
        raise ValueError("adaptive_sampling.pre_failure_seconds must be non-negative")
    sim = get_block(cfg, "sim", required=False)
    sim_hz_value = get_value(sim, "hz", None) if sim is not None else None
    decimation_value = get_value(cfg, "decimation", None)
    if method == ADAPTIVE_SAMPLING_METHOD_SONIC_FAILURE and (
        sim_hz_value is None or decimation_value is None
    ):
        raise ValueError(
            "adaptive_sampling.method=sonic_failure requires env.sim.hz and "
            "env.decimation to resolve pre_failure_seconds"
        )
    if sim_hz_value is None or decimation_value is None:
        sonic_pre_failure_window_steps = 0
    else:
        sim_hz = float(sim_hz_value)
        decimation = int(decimation_value)
        if sim_hz <= 0.0 or decimation <= 0:
            raise ValueError("env.sim.hz and env.decimation must be greater than zero")
        sonic_pre_failure_window_steps = max(
            0,
            int(round(pre_failure_seconds * sim_hz / float(decimation))),
        )

    disable_reason = None
    if enabled and not random_init_enabled:
        disable_reason = (
            "reset.deterministic=true"
            if deterministic_reset
            else "reset.random_start=false"
        )
        enabled = False

    return AdaptiveSamplingConfig(
        enabled=enabled,
        method=method,
        metric_type=metric_type,
        blame_mode=blame_mode,
        chunk_size=chunk_size,
        unit_count=unit_count,
        ema_alpha=ema_alpha,
        temperature=temperature,
        decay_rate=decay_rate,
        random_sample_prob=random_sample_prob,
        min_metric=min_metric,
        terminal_window_chunk_count=terminal_window_chunk_count,
        benchmark_beta=benchmark_beta,
        focus_ess_ratio=focus_ess_ratio,
        staleness_mix=staleness_mix,
        focus_strength=focus_strength,
        stale_half_life_passes=stale_half_life_passes,
        sonic_uniform_sampling_rate=sonic_uniform_sampling_rate,
        sonic_failure_rate_max_over_mean=sonic_failure_rate_max_over_mean,
        sonic_pre_failure_window_steps=sonic_pre_failure_window_steps,
        disable_reason=disable_reason,
    )


def allocate_adaptive_sampling_buffers(
    *,
    num_envs: int,
    unit_count: int,
    device: torch.device,
) -> AdaptiveSamplingBuffers:
    online_dist = torch.full(
        (unit_count,),
        1.0 / float(unit_count),
        device=device,
        dtype=torch.float32,
    )
    online_cdf = torch.cumsum(online_dist, dim=0)
    online_cdf[-1] = 1.0
    benchmark_prior = torch.zeros(unit_count, device=device, dtype=torch.float32)
    benchmark_prior_enabled = torch.zeros(1, device=device, dtype=torch.int32)
    dist = online_dist.clone()
    cdf = torch.cumsum(dist, dim=0)
    cdf[-1] = 1.0
    return AdaptiveSamplingBuffers(
        episode_start_unit_idx=torch.zeros(num_envs, device=device, dtype=torch.int32),
        episode_terminal_unit_idx=torch.zeros(
            num_envs, device=device, dtype=torch.int32
        ),
        episode_terminal_clip_unit_start=torch.zeros(
            num_envs, device=device, dtype=torch.int32
        ),
        episode_terminal_clip_unit_count=torch.ones(
            num_envs, device=device, dtype=torch.int32
        ),
        episode_reward_sum=torch.zeros(num_envs, device=device, dtype=torch.float32),
        episode_length_before_reset=torch.zeros(
            num_envs, device=device, dtype=torch.long
        ),
        metric_sum=torch.zeros(unit_count, device=device, dtype=torch.float32),
        metric_count=torch.zeros(unit_count, device=device, dtype=torch.float32),
        metric_event_count=torch.zeros(1, device=device, dtype=torch.float32),
        exp_metric=torch.zeros(unit_count, device=device, dtype=torch.float32),
        online_dist=online_dist,
        online_cdf=online_cdf,
        benchmark_prior=benchmark_prior,
        benchmark_prior_enabled=benchmark_prior_enabled,
        dist=dist,
        cdf=cdf,
    )


def allocate_disabled_adaptive_sampling_buffers(
    *,
    num_envs: int,
    device: torch.device,
) -> AdaptiveSamplingBuffers:
    episode_start_unit_idx = torch.zeros(num_envs, device=device, dtype=torch.int32)
    return AdaptiveSamplingBuffers(
        episode_start_unit_idx=episode_start_unit_idx,
        episode_terminal_unit_idx=torch.zeros(1, device=device, dtype=torch.int32),
        episode_terminal_clip_unit_start=torch.zeros(
            1, device=device, dtype=torch.int32
        ),
        episode_terminal_clip_unit_count=torch.ones(
            1, device=device, dtype=torch.int32
        ),
        episode_reward_sum=torch.zeros(1, device=device, dtype=torch.float32),
        episode_length_before_reset=torch.zeros(1, device=device, dtype=torch.long),
        metric_sum=torch.zeros(1, device=device, dtype=torch.float32),
        metric_count=torch.zeros(1, device=device, dtype=torch.float32),
        metric_event_count=torch.zeros(1, device=device, dtype=torch.float32),
        exp_metric=torch.zeros(1, device=device, dtype=torch.float32),
        online_dist=torch.ones(1, device=device, dtype=torch.float32),
        online_cdf=torch.ones(1, device=device, dtype=torch.float32),
        benchmark_prior=torch.zeros(1, device=device, dtype=torch.float32),
        benchmark_prior_enabled=torch.zeros(1, device=device, dtype=torch.int32),
        dist=torch.ones(1, device=device, dtype=torch.float32),
        cdf=torch.ones(1, device=device, dtype=torch.float32),
    )


def _normalize_probability_distribution(
    distribution: torch.Tensor,
    *,
    expected_size: int | None = None,
) -> torch.Tensor:
    dist = torch.as_tensor(distribution, dtype=torch.float32).reshape(-1).clone()
    if expected_size is not None and dist.numel() != expected_size:
        raise ValueError(
            f"distribution size mismatch: expected {expected_size}, got {dist.numel()}"
        )
    dist.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    dist.clamp_(min=0.0)
    norm = float(dist.sum().item())
    if norm <= 0.0:
        if dist.numel() <= 0:
            return dist
        dist.fill_(1.0 / float(dist.numel()))
    else:
        dist.div_(norm)
    return dist


def blend_benchmark_prior_distribution(
    *,
    online_dist: torch.Tensor,
    benchmark_prior: torch.Tensor,
    benchmark_beta: float,
) -> torch.Tensor:
    online = _normalize_probability_distribution(online_dist)
    beta = min(1.0, max(0.0, float(benchmark_beta)))
    if beta <= 0.0:
        return online

    prior = _normalize_probability_distribution(
        benchmark_prior,
        expected_size=online.numel(),
    )
    blended = (1.0 - beta) * online + beta * prior
    return _normalize_probability_distribution(blended, expected_size=online.numel())


@wp.kernel(enable_backward=False)
def accumulate_episode_reward_sum_kernel(
    reward: wp.array(dtype=float),
    episode_reward_sum: wp.array(dtype=float),
):
    env_idx = wp.tid()
    reward_value = reward[env_idx]
    if wp.isfinite(reward_value):
        episode_reward_sum[env_idx] = episode_reward_sum[env_idx] + reward_value


@wp.kernel(enable_backward=False)
def accumulate_sonic_failure_statistics_kernel(
    reset_terminated: wp.array(dtype=wp.bool),
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
    clip_unit_start: wp.array(dtype=int),
    clip_unit_count: wp.array(dtype=int),
    adaptive_chunk_size: int,
    num_failures: wp.array(dtype=float),
    num_exposures: wp.array(dtype=float),
):
    env_idx = wp.tid()
    clip_idx = motion_clip_idx[env_idx]
    length = clip_length[clip_idx]
    if length <= 0:
        return

    local_frame_idx = ref_frame_idx[env_idx]
    while local_frame_idx < 0:
        local_frame_idx += length
    while local_frame_idx >= length:
        local_frame_idx -= length

    chunk_size = adaptive_chunk_size
    if chunk_size <= 0:
        chunk_size = 1
    local_unit_idx = local_frame_idx // chunk_size
    unit_count = clip_unit_count[clip_idx]
    if unit_count <= 0:
        return
    if local_unit_idx >= unit_count:
        local_unit_idx = unit_count - 1
    global_unit_idx = clip_unit_start[clip_idx] + local_unit_idx

    unit_frame_count = length - local_unit_idx * chunk_size
    if unit_frame_count > chunk_size:
        unit_frame_count = chunk_size
    if unit_frame_count <= 0:
        unit_frame_count = 1
    wp.atomic_add(num_exposures, global_unit_idx, 1.0 / float(unit_frame_count))
    if reset_terminated[env_idx]:
        wp.atomic_add(num_failures, global_unit_idx, 1.0)


@wp.kernel(enable_backward=False)
def adaptive_accumulate_reset_metrics_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    reset_terminated: wp.array(dtype=wp.bool),
    reset_time_outs: wp.array(dtype=wp.bool),
    episode_length_buf: wp.array(dtype=wp.int64),
    adaptive_metric_type: int,
    reward_normalization_steps: int,
    episode_reward_sum: wp.array(dtype=float),
    adaptive_blame_mode: int,
    adaptive_terminal_window_chunk_count: int,
    episode_start_unit_idx: wp.array(dtype=int),
    episode_terminal_unit_idx: wp.array(dtype=int),
    episode_terminal_clip_unit_start: wp.array(dtype=int),
    episode_terminal_clip_unit_count: wp.array(dtype=int),
    adaptive_metric_sum: wp.array(dtype=float),
    adaptive_metric_count: wp.array(dtype=float),
    adaptive_metric_event_count: wp.array(dtype=float),
):
    env_idx = wp.tid()
    if not reset_mask[env_idx]:
        return

    ep_len = int(episode_length_buf[env_idx])
    if ep_len > 0:
        metric = episode_reward_sum[env_idx]
        if not wp.isfinite(metric):
            episode_reward_sum[env_idx] = 0.0
            return
        if adaptive_metric_type == ADAPTIVE_METRIC_TYPE_LENGTH:
            metric = float(ep_len)
            if (
                reset_time_outs[env_idx]
                and not reset_terminated[env_idx]
                and reward_normalization_steps > ep_len
            ):
                metric = float(reward_normalization_steps)
        elif (
            reset_time_outs[env_idx]
            and not reset_terminated[env_idx]
            and reward_normalization_steps > ep_len
        ):
            metric = metric * (float(reward_normalization_steps) / float(ep_len))

        if adaptive_blame_mode == ADAPTIVE_BLAME_MODE_TERMINAL_WINDOW:
            clip_unit_start = episode_terminal_clip_unit_start[env_idx]
            clip_unit_count = episode_terminal_clip_unit_count[env_idx]
            if clip_unit_count <= 0:
                clip_unit_count = 1
                clip_unit_start = 0

            window = adaptive_terminal_window_chunk_count
            if window <= 0:
                window = 1
            if window > clip_unit_count:
                window = clip_unit_count

            terminal_local_unit_idx = (
                episode_terminal_unit_idx[env_idx] - clip_unit_start
            )
            weight = 1.0 / float(window)
            for offset in range(window):
                blamed_local_unit_idx = terminal_local_unit_idx - offset
                while blamed_local_unit_idx < 0:
                    blamed_local_unit_idx += clip_unit_count
                while blamed_local_unit_idx >= clip_unit_count:
                    blamed_local_unit_idx -= clip_unit_count
                blamed_unit_idx = clip_unit_start + blamed_local_unit_idx
                wp.atomic_add(adaptive_metric_sum, blamed_unit_idx, metric * weight)
                wp.atomic_add(adaptive_metric_count, blamed_unit_idx, weight)
        else:
            start_unit = episode_start_unit_idx[env_idx]
            wp.atomic_add(adaptive_metric_sum, start_unit, metric)
            wp.atomic_add(adaptive_metric_count, start_unit, 1.0)

        wp.atomic_add(adaptive_metric_event_count, 0, 1.0)

    episode_reward_sum[env_idx] = 0.0


@wp.func
def valid_reset_start_count(clip_length: int, reset_start_horizon_steps: int) -> int:
    horizon = reset_start_horizon_steps
    if horizon < 1:
        horizon = 1

    count = clip_length - horizon + 1
    if count < 1:
        count = 1
    if count > clip_length:
        count = clip_length
    return count


@wp.func
def sample_adaptive_frame(
    state: wp.uint32,
    random_init: int,
    adaptive_enabled: int,
    adaptive_random_sample_prob: float,
    adaptive_chunk_size: int,
    adaptive_unit_count: int,
    adaptive_cdf: wp.array(dtype=float),
    motion_frame_count: int,
    reset_start_horizon_steps: int,
) -> int:
    fi = int(0)
    if random_init != 1:
        return fi

    valid_start_count = valid_reset_start_count(
        motion_frame_count,
        reset_start_horizon_steps,
    )

    use_uniform = adaptive_enabled == 0 or adaptive_unit_count <= 0
    if adaptive_enabled == 1 and adaptive_random_sample_prob > 0.0:
        use_uniform = use_uniform or (wp.randf(state) < adaptive_random_sample_prob)

    if use_uniform:
        return int(wp.randf(state) * float(valid_start_count))

    u = wp.randf(state)
    lo = int(0)
    hi = int(adaptive_unit_count - 1)
    while lo < hi:
        mid = int((lo + hi) // 2)
        if u <= adaptive_cdf[mid]:
            hi = mid
        else:
            lo = int(mid + 1)

    unit_idx = lo
    if adaptive_chunk_size <= 1:
        if unit_idx >= valid_start_count:
            return valid_start_count - 1
        return unit_idx

    start = unit_idx * adaptive_chunk_size
    if start >= valid_start_count:
        return valid_start_count - 1
    end = start + adaptive_chunk_size
    if end > valid_start_count:
        end = valid_start_count
    local = int(wp.randf(state) * float(end - start))
    return start + local


@wp.func
def sample_adaptive_unit(
    state: wp.uint32,
    adaptive_cdf: wp.array(dtype=float),
    adaptive_unit_count: int,
) -> int:
    u = wp.randf(state)
    lo = int(0)
    hi = int(adaptive_unit_count - 1)
    while lo < hi:
        mid = int((lo + hi) // 2)
        if u <= adaptive_cdf[mid]:
            hi = mid
        else:
            lo = int(mid + 1)
    return lo


@wp.func
def _find_clip_from_global_unit(
    global_unit_idx: int,
    clip_unit_start: wp.array(dtype=int),
    clip_unit_count: wp.array(dtype=int),
) -> int:
    clip_count = int(clip_unit_start.shape[0])
    lo = int(0)
    hi = int(clip_count - 1)
    while lo < hi:
        mid = int((lo + hi + 1) // 2)
        if clip_unit_start[mid] <= global_unit_idx:
            lo = mid
        else:
            hi = int(mid - 1)
    return lo


@wp.func
def _global_unit_to_global_frame_for_clip(
    state: wp.uint32,
    clip_idx: int,
    global_unit_idx: int,
    adaptive_chunk_size: int,
    reset_start_horizon_steps: int,
    clip_start: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
    clip_unit_start: wp.array(dtype=int),
) -> int:
    local_unit_idx = global_unit_idx - clip_unit_start[clip_idx]
    start = clip_start[clip_idx] + local_unit_idx * adaptive_chunk_size
    end = start + adaptive_chunk_size
    valid_start_count = valid_reset_start_count(
        clip_length[clip_idx],
        reset_start_horizon_steps,
    )
    clip_valid_end = clip_start[clip_idx] + valid_start_count
    if start >= clip_valid_end:
        return clip_valid_end - 1
    if end > clip_valid_end:
        end = clip_valid_end
    local = int(wp.randf(state) * float(end - start))
    return start + local


@wp.func
def _global_unit_to_global_frame(
    state: wp.uint32,
    global_unit_idx: int,
    adaptive_chunk_size: int,
    reset_start_horizon_steps: int,
    clip_start: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
    clip_unit_start: wp.array(dtype=int),
    clip_unit_count: wp.array(dtype=int),
) -> int:
    clip_idx = _find_clip_from_global_unit(
        global_unit_idx,
        clip_unit_start,
        clip_unit_count,
    )
    return _global_unit_to_global_frame_for_clip(
        state,
        clip_idx,
        global_unit_idx,
        adaptive_chunk_size,
        reset_start_horizon_steps,
        clip_start,
        clip_length,
        clip_unit_start,
    )


@wp.kernel(enable_backward=False)
def reset_from_motion_multi_select_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    reset_seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    reset_sampling_settings: wp.array(dtype=int),  # (RESET_SETTING_COUNT,)
    adaptive_random_sample_prob: float,
    adaptive_cdf: wp.array(dtype=float),
    analysis_reset_override_enabled: wp.array(dtype=wp.bool),
    analysis_reset_override_clip_idx: wp.array(dtype=int),
    analysis_reset_override_start_frame_idx: wp.array(dtype=int),
    clip_start: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
    frame_to_clip_idx: wp.array(dtype=int),
    clip_unit_start: wp.array(dtype=int),
    clip_unit_count: wp.array(dtype=int),
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    episode_start_unit_idx: wp.array(dtype=int),
    early_termination_timer: wp.array(dtype=int),
    reset_motion_frame_idx: wp.array(dtype=int),
):
    env_idx = wp.tid()
    if not reset_mask[env_idx]:
        return

    random_init = reset_sampling_settings[RESET_SETTING_RANDOM_INIT]
    start_frame_idx = reset_sampling_settings[RESET_SETTING_START_FRAME_IDX]
    adaptive_enabled = reset_sampling_settings[RESET_SETTING_ADAPTIVE_ENABLED]
    adaptive_chunk_size = reset_sampling_settings[RESET_SETTING_ADAPTIVE_CHUNK_SIZE]
    adaptive_unit_count = reset_sampling_settings[RESET_SETTING_ADAPTIVE_UNIT_COUNT]
    adaptive_pre_failure_window_steps = reset_sampling_settings[
        RESET_SETTING_PRE_FAILURE_WINDOW_STEPS
    ]
    total_motion_frame_count = reset_sampling_settings[
        RESET_SETTING_TOTAL_MOTION_FRAME_COUNT
    ]
    reset_start_horizon_steps = reset_sampling_settings[
        RESET_SETTING_START_HORIZON_STEPS
    ]
    fixed_motion_idx = reset_sampling_settings[RESET_SETTING_FIXED_MOTION_IDX]

    seed = reset_seed[0] + common_step_counter[0]
    state = wp.rand_init(seed, env_idx)
    clip_idx = int(0)
    local_frame_idx = int(0)
    global_frame_idx = int(0)
    global_unit_idx = int(0)

    if analysis_reset_override_enabled[env_idx]:
        clip_idx = analysis_reset_override_clip_idx[env_idx]
        local_frame_idx = analysis_reset_override_start_frame_idx[env_idx]
        global_frame_idx = clip_start[clip_idx] + local_frame_idx
        if adaptive_chunk_size > 1:
            global_unit_idx = clip_unit_start[clip_idx] + (
                local_frame_idx // adaptive_chunk_size
            )
        else:
            global_unit_idx = clip_unit_start[clip_idx] + local_frame_idx
    elif start_frame_idx >= 0:
        clip_idx = fixed_motion_idx if fixed_motion_idx >= 0 else int(0)
        local_frame_idx = start_frame_idx
        global_frame_idx = clip_start[clip_idx] + local_frame_idx
        if adaptive_chunk_size > 1:
            global_unit_idx = clip_unit_start[clip_idx] + (
                local_frame_idx // adaptive_chunk_size
            )
        else:
            global_unit_idx = clip_unit_start[clip_idx] + local_frame_idx
    elif random_init == 1:
        if fixed_motion_idx >= 0:
            clip_idx = fixed_motion_idx
            valid_start_count = valid_reset_start_count(
                clip_length[clip_idx],
                reset_start_horizon_steps,
            )
            local_frame_idx = int(wp.randf(state) * float(valid_start_count))
            global_frame_idx = clip_start[clip_idx] + local_frame_idx
            if adaptive_chunk_size > 1:
                global_unit_idx = clip_unit_start[clip_idx] + (
                    local_frame_idx // adaptive_chunk_size
                )
            else:
                global_unit_idx = clip_unit_start[clip_idx] + local_frame_idx
        else:
            use_uniform = adaptive_enabled == 0 or adaptive_unit_count <= 0
            if adaptive_enabled == 1 and adaptive_random_sample_prob > 0.0:
                use_uniform = use_uniform or (
                    wp.randf(state) < adaptive_random_sample_prob
                )

            if use_uniform:
                if adaptive_unit_count > 0:
                    global_unit_idx = int(wp.randf(state) * float(adaptive_unit_count))
                    if global_unit_idx >= adaptive_unit_count:
                        global_unit_idx = adaptive_unit_count - 1
                    clip_idx = _find_clip_from_global_unit(
                        global_unit_idx,
                        clip_unit_start,
                        clip_unit_count,
                    )
                    global_frame_idx = _global_unit_to_global_frame_for_clip(
                        state,
                        clip_idx,
                        global_unit_idx,
                        adaptive_chunk_size,
                        reset_start_horizon_steps,
                        clip_start,
                        clip_length,
                        clip_unit_start,
                    )
                    local_frame_idx = global_frame_idx - clip_start[clip_idx]
                else:
                    sampled_global_frame_idx = int(
                        wp.randf(state) * float(total_motion_frame_count)
                    )
                    clip_idx = frame_to_clip_idx[sampled_global_frame_idx]
                    valid_start_count = valid_reset_start_count(
                        clip_length[clip_idx],
                        reset_start_horizon_steps,
                    )
                    local_frame_idx = int(wp.randf(state) * float(valid_start_count))
                    global_frame_idx = clip_start[clip_idx] + local_frame_idx
                    if adaptive_chunk_size > 1:
                        global_unit_idx = clip_unit_start[clip_idx] + (
                            local_frame_idx // adaptive_chunk_size
                        )
                    else:
                        global_unit_idx = clip_unit_start[clip_idx] + local_frame_idx
            else:
                global_unit_idx = sample_adaptive_unit(
                    state,
                    adaptive_cdf,
                    adaptive_unit_count,
                )
                clip_idx = _find_clip_from_global_unit(
                    global_unit_idx,
                    clip_unit_start,
                    clip_unit_count,
                )
                global_frame_idx = _global_unit_to_global_frame_for_clip(
                    state,
                    clip_idx,
                    global_unit_idx,
                    adaptive_chunk_size,
                    reset_start_horizon_steps,
                    clip_start,
                    clip_length,
                    clip_unit_start,
                )
                local_frame_idx = global_frame_idx - clip_start[clip_idx]
    else:
        clip_idx = fixed_motion_idx if fixed_motion_idx >= 0 else int(0)
        global_frame_idx = clip_start[clip_idx]
        local_frame_idx = int(0)
        global_unit_idx = clip_unit_start[clip_idx]

    if (
        not analysis_reset_override_enabled[env_idx]
        and start_frame_idx < 0
        and random_init == 1
        and fixed_motion_idx < 0
        and adaptive_enabled == 1
        and adaptive_pre_failure_window_steps > 0
    ):
        offset = int(wp.randf(state) * float(adaptive_pre_failure_window_steps))
        local_frame_idx = local_frame_idx - offset
        if local_frame_idx < 0:
            local_frame_idx = 0
        global_frame_idx = clip_start[clip_idx] + local_frame_idx
        if adaptive_chunk_size > 1:
            global_unit_idx = clip_unit_start[clip_idx] + (
                local_frame_idx // adaptive_chunk_size
            )
        else:
            global_unit_idx = clip_unit_start[clip_idx] + local_frame_idx

    motion_clip_idx[env_idx] = clip_idx
    ref_frame_idx[env_idx] = local_frame_idx
    episode_start_unit_idx[env_idx] = global_unit_idx
    early_termination_timer[env_idx] = 0
    reset_motion_frame_idx[env_idx] = global_frame_idx


@wp.kernel(enable_backward=False)
def reset_from_motion_multi_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    reset_seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    random_init: int,
    start_frame_idx: int,
    adaptive_enabled: int,
    adaptive_random_sample_prob: float,
    adaptive_chunk_size: int,
    adaptive_unit_count: int,
    adaptive_cdf: wp.array(dtype=float),
    total_motion_frame_count: int,
    reset_start_horizon_steps: int,
    fixed_motion_idx: int,
    analysis_reset_override_enabled: wp.array(dtype=wp.bool),
    analysis_reset_override_clip_idx: wp.array(dtype=int),
    analysis_reset_override_start_frame_idx: wp.array(dtype=int),
    clip_start: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
    frame_to_clip_idx: wp.array(dtype=int),
    clip_unit_start: wp.array(dtype=int),
    clip_unit_count: wp.array(dtype=int),
    motion_qpos: wp.array2d(dtype=float),
    motion_qvel: wp.array2d(dtype=float),
    noise_std: float,
    spawn_offset: wp.vec3,
    nq: int,
    nv: int,
    nu: int,
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    episode_start_unit_idx: wp.array(dtype=int),
    early_termination_timer: wp.array(dtype=int),
    actions: wp.array2d(dtype=float),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    if not reset_mask[env_idx]:
        return

    seed = reset_seed[0] + common_step_counter[0]
    state = wp.rand_init(seed, env_idx)
    clip_idx = int(0)
    local_frame_idx = int(0)
    global_frame_idx = int(0)
    global_unit_idx = int(0)

    if analysis_reset_override_enabled[env_idx]:
        clip_idx = analysis_reset_override_clip_idx[env_idx]
        local_frame_idx = analysis_reset_override_start_frame_idx[env_idx]
        global_frame_idx = clip_start[clip_idx] + local_frame_idx
        if adaptive_chunk_size > 1:
            global_unit_idx = clip_unit_start[clip_idx] + (
                local_frame_idx // adaptive_chunk_size
            )
        else:
            global_unit_idx = clip_unit_start[clip_idx] + local_frame_idx
    elif start_frame_idx >= 0:
        clip_idx = fixed_motion_idx if fixed_motion_idx >= 0 else int(0)
        local_frame_idx = start_frame_idx
        global_frame_idx = clip_start[clip_idx] + local_frame_idx
        if adaptive_chunk_size > 1:
            global_unit_idx = clip_unit_start[clip_idx] + (
                local_frame_idx // adaptive_chunk_size
            )
        else:
            global_unit_idx = clip_unit_start[clip_idx] + local_frame_idx
    elif random_init == 1:
        if fixed_motion_idx >= 0:
            clip_idx = fixed_motion_idx
            valid_start_count = valid_reset_start_count(
                clip_length[clip_idx],
                reset_start_horizon_steps,
            )
            local_frame_idx = int(wp.randf(state) * float(valid_start_count))
            global_frame_idx = clip_start[clip_idx] + local_frame_idx
            if adaptive_chunk_size > 1:
                global_unit_idx = clip_unit_start[clip_idx] + (
                    local_frame_idx // adaptive_chunk_size
                )
            else:
                global_unit_idx = clip_unit_start[clip_idx] + local_frame_idx
        else:
            use_uniform = adaptive_enabled == 0 or adaptive_unit_count <= 0
            if adaptive_enabled == 1 and adaptive_random_sample_prob > 0.0:
                use_uniform = use_uniform or (
                    wp.randf(state) < adaptive_random_sample_prob
                )

            if use_uniform:
                if adaptive_unit_count > 0:
                    global_unit_idx = int(wp.randf(state) * float(adaptive_unit_count))
                    if global_unit_idx >= adaptive_unit_count:
                        global_unit_idx = adaptive_unit_count - 1
                    clip_idx = _find_clip_from_global_unit(
                        global_unit_idx,
                        clip_unit_start,
                        clip_unit_count,
                    )
                    global_frame_idx = _global_unit_to_global_frame(
                        state,
                        global_unit_idx,
                        adaptive_chunk_size,
                        reset_start_horizon_steps,
                        clip_start,
                        clip_length,
                        clip_unit_start,
                        clip_unit_count,
                    )
                    local_frame_idx = global_frame_idx - clip_start[clip_idx]
                else:
                    sampled_global_frame_idx = int(
                        wp.randf(state) * float(total_motion_frame_count)
                    )
                    clip_idx = frame_to_clip_idx[sampled_global_frame_idx]
                    valid_start_count = valid_reset_start_count(
                        clip_length[clip_idx],
                        reset_start_horizon_steps,
                    )
                    local_frame_idx = int(wp.randf(state) * float(valid_start_count))
                    global_frame_idx = clip_start[clip_idx] + local_frame_idx
                    if adaptive_chunk_size > 1:
                        global_unit_idx = clip_unit_start[clip_idx] + (
                            local_frame_idx // adaptive_chunk_size
                        )
                    else:
                        global_unit_idx = clip_unit_start[clip_idx] + local_frame_idx
            else:
                global_unit_idx = sample_adaptive_unit(
                    state,
                    adaptive_cdf,
                    adaptive_unit_count,
                )
                clip_idx = _find_clip_from_global_unit(
                    global_unit_idx,
                    clip_unit_start,
                    clip_unit_count,
                )
                global_frame_idx = _global_unit_to_global_frame(
                    state,
                    global_unit_idx,
                    adaptive_chunk_size,
                    reset_start_horizon_steps,
                    clip_start,
                    clip_length,
                    clip_unit_start,
                    clip_unit_count,
                )
                local_frame_idx = global_frame_idx - clip_start[clip_idx]
    else:
        clip_idx = fixed_motion_idx if fixed_motion_idx >= 0 else int(0)
        global_frame_idx = clip_start[clip_idx]
        local_frame_idx = int(0)
        global_unit_idx = clip_unit_start[clip_idx]

    motion_clip_idx[env_idx] = clip_idx
    ref_frame_idx[env_idx] = local_frame_idx
    episode_start_unit_idx[env_idx] = global_unit_idx
    early_termination_timer[env_idx] = 0

    for j in range(nu):
        actions[env_idx, j] = 0.0

    for j in range(nq):
        val = motion_qpos[global_frame_idx, j]
        if j == 0:
            val += spawn_offset.x
        elif j == 1:
            val += spawn_offset.y
        elif j == 2:
            val += spawn_offset.z
        if j >= 7:
            val += wp.randn(state) * noise_std
        qpos[env_idx, j] = val

    for j in range(nv):
        val = motion_qvel[global_frame_idx, j]
        if j >= 6:
            val += wp.randn(state) * noise_std
        qvel[env_idx, j] = val


@wp.kernel(enable_backward=False)
def reset_from_motion_select_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    reset_seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    random_init: int,
    start_frame_idx: int,
    adaptive_enabled: int,
    adaptive_random_sample_prob: float,
    adaptive_chunk_size: int,
    adaptive_unit_count: int,
    adaptive_cdf: wp.array(dtype=float),
    adaptive_pre_failure_window_steps: int,
    motion_frame_count: int,
    reset_start_horizon_steps: int,
    analysis_reset_override_enabled: wp.array(dtype=wp.bool),
    analysis_reset_override_start_frame_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    episode_start_unit_idx: wp.array(dtype=int),
    early_termination_timer: wp.array(dtype=int),
    reset_motion_frame_idx: wp.array(dtype=int),
):
    env_idx = wp.tid()
    if not reset_mask[env_idx]:
        return

    seed = reset_seed[0] + common_step_counter[0]
    state = wp.rand_init(seed, env_idx)
    fi = start_frame_idx
    if analysis_reset_override_enabled[env_idx]:
        fi = analysis_reset_override_start_frame_idx[env_idx]
    elif fi < 0:
        fi = sample_adaptive_frame(
            state,
            random_init,
            adaptive_enabled,
            adaptive_random_sample_prob,
            adaptive_chunk_size,
            adaptive_unit_count,
            adaptive_cdf,
            motion_frame_count,
            reset_start_horizon_steps,
        )
        if adaptive_enabled == 1 and adaptive_pre_failure_window_steps > 0:
            offset = int(wp.randf(state) * float(adaptive_pre_failure_window_steps))
            fi = fi - offset
            if fi < 0:
                fi = 0

    ref_frame_idx[env_idx] = fi
    if adaptive_chunk_size > 1:
        episode_start_unit_idx[env_idx] = fi // adaptive_chunk_size
    else:
        episode_start_unit_idx[env_idx] = fi
    early_termination_timer[env_idx] = 0
    reset_motion_frame_idx[env_idx] = fi


@wp.kernel(enable_backward=False)
def reset_from_motion_state_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    reset_seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    reset_motion_frame_idx: wp.array(dtype=int),
    motion_qpos: wp.array2d(dtype=float),
    motion_qvel: wp.array2d(dtype=float),
    noise_std: float,
    spawn_offset: wp.vec3,
    nq: int,
    nv: int,
    nu: int,
    root_qpos_count: int,
    root_dof_count: int,
    actions: wp.array2d(dtype=float),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
):
    env_idx, elem_idx = wp.tid()
    if not reset_mask[env_idx]:
        return

    seed = reset_seed[0] + common_step_counter[0]
    frame_idx = reset_motion_frame_idx[env_idx]
    if elem_idx < nu:
        actions[env_idx, elem_idx] = 0.0

    if elem_idx < nq:
        val = motion_qpos[frame_idx, elem_idx]
        if root_qpos_count == 7:
            if elem_idx == 0:
                val += spawn_offset.x
            elif elem_idx == 1:
                val += spawn_offset.y
            elif elem_idx == 2:
                val += spawn_offset.z
        if noise_std != 0.0 and elem_idx >= root_qpos_count:
            state = wp.rand_init(seed, env_idx * 4096 + elem_idx)
            val += wp.randn(state) * noise_std
        qpos[env_idx, elem_idx] = val

    if elem_idx < nv:
        val = motion_qvel[frame_idx, elem_idx]
        if noise_std != 0.0 and elem_idx >= root_dof_count:
            state = wp.rand_init(seed, env_idx * 4096 + 2048 + elem_idx)
            val += wp.randn(state) * noise_std
        qvel[env_idx, elem_idx] = val


@wp.kernel(enable_backward=False)
def reset_from_motion_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    reset_seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    random_init: int,
    start_frame_idx: int,
    adaptive_enabled: int,
    adaptive_random_sample_prob: float,
    adaptive_chunk_size: int,
    adaptive_unit_count: int,
    adaptive_cdf: wp.array(dtype=float),
    motion_frame_count: int,
    reset_start_horizon_steps: int,
    motion_qpos: wp.array2d(dtype=float),
    motion_qvel: wp.array2d(dtype=float),
    noise_std: float,
    spawn_offset: wp.vec3,
    nq: int,
    nv: int,
    nu: int,
    ref_frame_idx: wp.array(dtype=int),
    episode_start_unit_idx: wp.array(dtype=int),
    early_termination_timer: wp.array(dtype=int),
    actions: wp.array2d(dtype=float),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    if not reset_mask[env_idx]:
        return

    seed = reset_seed[0] + common_step_counter[0]
    state = wp.rand_init(seed, env_idx)
    fi = start_frame_idx
    if fi < 0:
        fi = sample_adaptive_frame(
            state,
            random_init,
            adaptive_enabled,
            adaptive_random_sample_prob,
            adaptive_chunk_size,
            adaptive_unit_count,
            adaptive_cdf,
            motion_frame_count,
            reset_start_horizon_steps,
        )

    ref_frame_idx[env_idx] = fi
    if adaptive_chunk_size > 1:
        episode_start_unit_idx[env_idx] = fi // adaptive_chunk_size
    else:
        episode_start_unit_idx[env_idx] = fi
    early_termination_timer[env_idx] = 0

    for j in range(nu):
        actions[env_idx, j] = 0.0

    for j in range(nq):
        val = motion_qpos[fi, j]
        if j == 0:
            val += spawn_offset.x
        elif j == 1:
            val += spawn_offset.y
        elif j == 2:
            val += spawn_offset.z
        if j >= 7:
            val += wp.randn(state) * noise_std
        qpos[env_idx, j] = val

    for j in range(nv):
        val = motion_qvel[fi, j]
        if j >= 6:
            val += wp.randn(state) * noise_std
        qvel[env_idx, j] = val


@wp.kernel(enable_backward=False)
def adaptive_update_distribution_kernel(
    metric_sum: wp.array(dtype=float),
    metric_count: wp.array(dtype=float),
    metric_event_count: wp.array(dtype=float),
    exp_metric: wp.array(dtype=float),
    dist: wp.array(dtype=float),
    cdf: wp.array(dtype=float),
    unit_count: int,
    ema_alpha: float,
    decay_rate: float,
    temperature: float,
    min_metric: float,
):
    if metric_event_count[0] <= 0.0:
        return

    alpha = wp.clamp(ema_alpha, 0.0, 1.0)
    keep = 1.0 - alpha
    max_metric = float(0.0)
    sum_weight = float(0.0)
    has_metric = int(0)

    for i in range(unit_count):
        old = exp_metric[i]
        if decay_rate < 1.0 and old > min_metric:
            old = old * decay_rate

        count = metric_count[i]
        if count > 0.0:
            mean_metric = metric_sum[i] / count
            old = keep * old + alpha * mean_metric
            has_metric = int(1)

        exp_metric[i] = old
        metric_sum[i] = 0.0
        metric_count[i] = 0.0

        if i == 0 or old > max_metric:
            max_metric = old

    if has_metric == 0:
        metric_event_count[0] = 0.0
        return

    for i in range(unit_count):
        w = wp.exp(temperature * (max_metric - exp_metric[i]))
        dist[i] = w
        sum_weight += w

    if sum_weight <= 0.0:
        p = 1.0 / float(unit_count)
        c = float(0.0)
        for i in range(unit_count):
            dist[i] = p
            c += p
            cdf[i] = c
    else:
        inv = 1.0 / sum_weight
        c = float(0.0)
        for i in range(unit_count):
            p = dist[i] * inv
            dist[i] = p
            c += p
            cdf[i] = c

    cdf[unit_count - 1] = 1.0
    metric_event_count[0] = 0.0


@wp.kernel(enable_backward=False)
def adaptive_refresh_sampling_distribution_kernel(
    online_dist: wp.array(dtype=float),
    benchmark_prior: wp.array(dtype=float),
    benchmark_prior_enabled: wp.array(dtype=int),
    dist: wp.array(dtype=float),
    cdf: wp.array(dtype=float),
    unit_count: int,
    benchmark_beta: float,
):
    beta = wp.clamp(benchmark_beta, 0.0, 1.0)
    sum_prob = float(0.0)
    c = float(0.0)
    enabled = benchmark_prior_enabled[0]

    for i in range(unit_count):
        p = online_dist[i]
        if enabled == 1 and beta > 0.0:
            p = (1.0 - beta) * p + beta * benchmark_prior[i]
        dist[i] = p
        sum_prob += p

    if sum_prob <= 0.0:
        p = 1.0 / float(unit_count)
        for i in range(unit_count):
            dist[i] = p
            c += p
            cdf[i] = c
    else:
        inv = 1.0 / sum_prob
        for i in range(unit_count):
            p = dist[i] * inv
            dist[i] = p
            c += p
            cdf[i] = c

    cdf[unit_count - 1] = 1.0
