"""Graph-safe PPO playback stepping shared by play and export tools."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
import warp as wp
from omegaconf import DictConfig

from instantmimic.envs.direct_rl_env import (
    StepInputs,
    StepIO,
    StepOutputs,
    external_step_capture_window,
)
from instantmimic.envs.eval_metrics import EvalEpisodeSpec
from instantmimic.envs.tracking import metrics_warp as tracking_metrics_warp_utils
from instantmimic.play.policy import policy_mean_into
from instantmimic.rl.eager_fallback import EagerFallbackPolicy
from instantmimic.rl.ppo.agent import PPOAgent
from instantmimic.rl.runtime import (
    agent_observation_output_views,
    copy_agent_observation_into,
)
from instantmimic.utils.network_precision import network_matmul_precision
from instantmimic.utils.torch_warp_capture import capture_warp_graph

_LOGGER = logging.getLogger(__name__)

_PLAYBACK_LENGTH_EPISODE_TOKENS = {
    "episode",
    "episode_length",
    "max_episode_length",
}

_PLAY_TIME_LIMIT_POLICY_EXTEND = "extend"
_PLAY_TIME_LIMIT_POLICY_RESPECT = "respect"
_PLAY_TIME_LIMIT_POLICIES = {
    _PLAY_TIME_LIMIT_POLICY_EXTEND,
    _PLAY_TIME_LIMIT_POLICY_RESPECT,
}
_PLAY_DONE_POLICY_RESET_CONTINUE = "reset_continue"
_PLAY_DONE_POLICY_STOP = "stop"
_PLAY_DONE_POLICIES = {
    _PLAY_DONE_POLICY_RESET_CONTINUE,
    _PLAY_DONE_POLICY_STOP,
}


def _initialize_goal_reach_counts(
    env_raw: Any,
    *,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor | None:
    getter = getattr(env_raw, "get_goal_reached_mask", None)
    if not callable(getter):
        return None
    mask = getter()
    if not isinstance(mask, torch.Tensor):
        raise TypeError("env.get_goal_reached_mask() must return a tensor")
    if tuple(mask.shape) != (int(num_envs),):
        raise ValueError(
            "env.get_goal_reached_mask() must return one value per environment, "
            f"got shape {tuple(mask.shape)} for {int(num_envs)} environments"
        )
    requested_device = torch.device(device)
    device_mismatch = mask.device.type != requested_device.type
    if requested_device.index is not None:
        device_mismatch = device_mismatch or mask.device.index != requested_device.index
    if device_mismatch:
        raise ValueError(
            "env.get_goal_reached_mask() must be on the playback device, "
            f"got {mask.device} instead of {requested_device}"
        )
    return torch.zeros(int(num_envs), device=mask.device, dtype=torch.int64)


def _accumulate_goal_reaches(
    env_raw: Any,
    counts: torch.Tensor | None,
) -> None:
    if counts is None:
        return
    mask = env_raw.get_goal_reached_mask()
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != tuple(counts.shape):
        raise RuntimeError(
            "env.get_goal_reached_mask() changed its tensor contract during playback"
        )
    counts.add_(mask)


def _summarize_goal_reaches(
    counts: torch.Tensor | None,
    *,
    eval_steps: int,
) -> dict[str, int | float | bool]:
    if counts is None:
        return {
            "goal_reach_metric_available": False,
            "goal_reach_eval_steps": int(eval_steps),
        }
    total = int(counts.sum().item())
    return {
        "goal_reach_metric_available": True,
        "goal_reach_total": total,
        "goal_reaches_per_env": float(total / max(1, int(counts.numel()))),
        "goal_reach_eval_steps": int(eval_steps),
        "goal_reach_num_envs": int(counts.numel()),
    }


def _cfg_contains_key(container: Any, key: str) -> bool:
    if container is None:
        return False
    if isinstance(container, DictConfig):
        return key in container
    return hasattr(container, key)


def _cfg_bool_value(container: Any, key: str, default: bool = False) -> bool:
    if container is None:
        return bool(default)
    if isinstance(container, DictConfig):
        if key not in container:
            return bool(default)
        return bool(container.get(key))
    return bool(getattr(container, key, default))


def _default_play_time_limit_policy(
    env_raw: Any | None,
) -> Literal["extend", "respect"]:
    cfg = getattr(env_raw, "cfg", None)
    termination = None if cfg is None else getattr(cfg, "termination", None)
    if termination is not None:
        if _cfg_bool_value(termination, "motion_end", default=False):
            return _PLAY_TIME_LIMIT_POLICY_RESPECT
        return _PLAY_TIME_LIMIT_POLICY_EXTEND
    if _cfg_contains_key(cfg, "terminate_at_motion_end"):
        if _cfg_bool_value(cfg, "terminate_at_motion_end", default=False):
            return _PLAY_TIME_LIMIT_POLICY_RESPECT
        return _PLAY_TIME_LIMIT_POLICY_EXTEND
    if env_raw is not None and (
        hasattr(env_raw, "motion_frame_count") or hasattr(env_raw, "motion_clip_length")
    ):
        return _PLAY_TIME_LIMIT_POLICY_EXTEND
    return _PLAY_TIME_LIMIT_POLICY_RESPECT


def _resolve_play_time_limit_policy(
    cfg: DictConfig,
    *,
    env_raw: Any | None = None,
) -> Literal["extend", "respect"]:
    raw_value = cfg.get("play_time_limit_policy", None)
    if raw_value is None:
        return _default_play_time_limit_policy(env_raw)
    raw = str(raw_value).strip().lower()
    if raw not in _PLAY_TIME_LIMIT_POLICIES:
        valid = ", ".join(sorted(_PLAY_TIME_LIMIT_POLICIES))
        raise ValueError(
            f"Unsupported play_time_limit_policy={raw!r}. Valid values: {valid}."
        )
    return raw  # type: ignore[return-value]


def _resolve_play_done_policy(
    cfg: DictConfig,
) -> Literal["reset_continue", "stop"]:
    raw = (
        str(cfg.get("play_done_policy", _PLAY_DONE_POLICY_RESET_CONTINUE))
        .strip()
        .lower()
    )
    if raw not in _PLAY_DONE_POLICIES:
        valid = ", ".join(sorted(_PLAY_DONE_POLICIES))
        raise ValueError(
            f"Unsupported play_done_policy={raw!r}. Valid values: {valid}."
        )
    return raw  # type: ignore[return-value]


def _playback_done_policy_stops(done_policy: str) -> bool:
    policy = str(done_policy).strip().lower()
    if policy == _PLAY_DONE_POLICY_STOP:
        return True
    if policy == _PLAY_DONE_POLICY_RESET_CONTINUE:
        return False
    valid = ", ".join(sorted(_PLAY_DONE_POLICIES))
    raise ValueError(f"Unsupported play_done_policy={policy!r}. Valid values: {valid}.")


def _resolve_play_use_cuda_graph(cfg: DictConfig) -> bool:
    value = cfg.get("play_use_cuda_graph", True)
    return True if value is None else bool(value)


@torch.no_grad()
def _policy_sample_into(
    agent: PPOAgent,
    obs_policy: torch.Tensor,
    out_actions: torch.Tensor,
) -> None:
    """Stochastic action sampling (same as training, but without grad)."""
    with network_matmul_precision():
        sampled_actions, _, _ = agent.act(obs_policy)
        out_actions.copy_(sampled_actions)


@torch.no_grad()
def _policy_sample_graph_safe_into(
    agent: PPOAgent,
    obs_policy: torch.Tensor,
    out_actions: torch.Tensor,
    out_log_probs: torch.Tensor,
    out_values: torch.Tensor,
) -> None:
    """Sample through the same preallocated noise and evaluation path as rollout."""
    with network_matmul_precision():
        scaled_obs = agent._policy_scaled_obs
        agent._obs_scale_eval_into_fn(obs_policy, scaled_obs)
        agent._act_into_fn(
            scaled_obs,
            out_actions,
            out_log_probs,
            out_values,
            agent._fill_policy_noise_(),
        )


def _maybe_build_compiled_playback_policy(
    *,
    agent: PPOAgent,
    device: torch.device,
    obs_policy: torch.Tensor,
    agent_actions: torch.Tensor,
    play_torch_stream: torch.cuda.Stream | None,
    stochastic_play: bool,
    eager_fallback: EagerFallbackPolicy,
    use_execution_mean: bool,
) -> dict[str, Any]:
    compiled: dict[str, Any] = {"mean": None, "execution_mean": None}
    if device.type != "cuda" or not hasattr(torch, "compile"):
        return compiled
    if stochastic_play:
        return compiled

    compile_mode = "max-autotune-no-cudagraphs"
    torch_stream_ctx = (
        torch.cuda.stream(play_torch_stream)
        if play_torch_stream is not None
        else nullcontext()
    )
    try:
        unscoped_mean = getattr(agent, "_policy_mean_into_unscoped", None)
        if not callable(unscoped_mean):
            unscoped_mean = agent.policy_mean_into
        unscoped_execution_mean = getattr(
            agent, "_policy_execution_mean_into_unscoped", None
        )
        if not callable(unscoped_execution_mean):
            unscoped_execution_mean = agent.policy_execution_mean_into

        @torch.no_grad()
        def _playback_policy_mean_compiled(
            obs: torch.Tensor,
            out_actions: torch.Tensor,
        ) -> None:
            unscoped_mean(obs, out_actions)

        @torch.no_grad()
        def _playback_policy_execution_mean_compiled(
            obs: torch.Tensor,
            out_actions: torch.Tensor,
        ) -> None:
            unscoped_execution_mean(
                obs,
                out_actions,
            )

        key = "execution_mean" if use_execution_mean else "mean"
        policy = (
            _playback_policy_execution_mean_compiled
            if use_execution_mean
            else _playback_policy_mean_compiled
        )
        compiled_policy = torch.compile(policy, mode=compile_mode, dynamic=False)

        @torch.no_grad()
        def _scoped_policy(obs: torch.Tensor, out_actions: torch.Tensor) -> None:
            # Keep process-state reads outside Dynamo's traced region.
            with network_matmul_precision():
                compiled_policy(obs, out_actions)

        compiled[key] = _scoped_policy
        warmup_obs = torch.zeros_like(obs_policy)
        warmup_actions = torch.empty_like(agent_actions)
        with torch_stream_ctx:
            compiled[key](warmup_obs, warmup_actions)
        if play_torch_stream is not None:
            play_torch_stream.synchronize()
        _LOGGER.info("Enabled torch.compile for deterministic playback policy path.")
    except Exception as exc:  # noqa: BLE001 - torch.compile surfaces backend
        # errors of many types; the policy decides whether to continue eagerly.
        eager_fallback.degrade("deterministic playback policy compile", exc)
        compiled["mean"] = None
        compiled["execution_mean"] = None
    return compiled


def _maybe_collect_tracking_metrics(
    env_raw: Any,
    *,
    frame: str = "root",
    include_root_in_ape: bool = False,
) -> dict[str, Any] | None:
    get_metrics = getattr(env_raw, "get_reference_tracking_metrics", None)
    if not callable(get_metrics):
        return None
    metrics = get_metrics(frame=frame, include_root_in_ape=include_root_in_ape)
    if not isinstance(metrics, dict):
        return None
    ape = metrics.get("ape")
    ee_error = metrics.get("ee_error")
    if not isinstance(ape, torch.Tensor) or not isinstance(ee_error, torch.Tensor):
        return None
    result: dict[str, Any] = {
        "ape": ape,
        "ee_error": ee_error,
    }
    sim_body_linvel = metrics.get("tracking_body_linvel")
    ref_body_linvel = metrics.get("reference_body_linvel")
    sim_root_quat = metrics.get("sim_root_quat")
    ref_root_quat = metrics.get("ref_root_quat")
    if (
        isinstance(sim_body_linvel, torch.Tensor)
        and isinstance(ref_body_linvel, torch.Tensor)
        and isinstance(sim_root_quat, torch.Tensor)
        and isinstance(ref_root_quat, torch.Tensor)
    ):
        result["tracking_body_linvel"] = sim_body_linvel
        result["reference_body_linvel"] = ref_body_linvel
        result["sim_root_quat"] = sim_root_quat
        result["ref_root_quat"] = ref_root_quat
    return result


def _update_tracking_metric_summary(
    summary: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> None:
    if metrics is None:
        return
    ape = metrics.get("ape")
    ee_error = metrics.get("ee_error")
    if not isinstance(ape, torch.Tensor) or not isinstance(ee_error, torch.Tensor):
        return

    summary["steps"] = int(summary["steps"]) + 1
    sim_body_linvel = metrics.get("tracking_body_linvel")
    ref_body_linvel = metrics.get("reference_body_linvel")
    sim_root_quat = metrics.get("sim_root_quat")
    ref_root_quat = metrics.get("ref_root_quat")
    scalar_tensors: list[torch.Tensor] = [ape.mean(), ee_error.mean()]
    scalar_names = ["ape", "ee_error"]
    vel_error: torch.Tensor | None = None
    accel_error: torch.Tensor | None = None
    precomputed_vel_error = metrics.get("vel_error")
    precomputed_accel_error = metrics.get("accel_error")
    accel_valid = bool(metrics.get("accel_valid", False))
    if isinstance(precomputed_vel_error, torch.Tensor):
        vel_error = precomputed_vel_error
        scalar_tensors.append(vel_error.mean())
        scalar_names.append("vel")
        if isinstance(precomputed_accel_error, torch.Tensor) and accel_valid:
            accel_error = precomputed_accel_error
            scalar_tensors.append(accel_error.mean())
            scalar_names.append("accel")
    elif (
        isinstance(sim_body_linvel, torch.Tensor)
        and isinstance(ref_body_linvel, torch.Tensor)
        and isinstance(sim_root_quat, torch.Tensor)
        and isinstance(ref_root_quat, torch.Tensor)
    ):
        vel_buffers = summary.get("vel_buffers")
        if (
            vel_buffers is None
            or int(vel_buffers.per_body_error.shape[0]) != int(sim_body_linvel.shape[0])
            or int(vel_buffers.per_body_error.shape[1]) != int(sim_body_linvel.shape[1])
        ):
            vel_buffers = tracking_metrics_warp_utils.create_metric_buffers(
                env_count=int(sim_body_linvel.shape[0]),
                body_count=int(sim_body_linvel.shape[1]),
                device=sim_body_linvel.device,
                dtype=sim_body_linvel.dtype,
            )
            summary["vel_buffers"] = vel_buffers
        tracking_metrics_warp_utils.compute_mean_velocity_error_into_warp(
            sim_body_linvel,
            ref_body_linvel,
            sim_root_quat=sim_root_quat,
            ref_root_quat=ref_root_quat,
            frame=str(summary.get("frame", "root")),
            out=vel_buffers,
        )
        vel_error = vel_buffers.mean_error
        scalar_tensors.append(vel_error.mean())
        scalar_names.append("vel")

        prev_sim_body_linvel = summary.get("prev_sim_body_linvel")
        prev_ref_body_linvel = summary.get("prev_ref_body_linvel")
        prev_sim_root_quat = summary.get("prev_sim_root_quat")
        prev_ref_root_quat = summary.get("prev_ref_root_quat")
        if (
            isinstance(prev_sim_body_linvel, torch.Tensor)
            and isinstance(prev_ref_body_linvel, torch.Tensor)
            and isinstance(prev_sim_root_quat, torch.Tensor)
            and isinstance(prev_ref_root_quat, torch.Tensor)
        ):
            accel_buffers = summary.get("accel_buffers")
            if (
                accel_buffers is None
                or int(accel_buffers.per_body_error.shape[0])
                != int(sim_body_linvel.shape[0])
                or int(accel_buffers.per_body_error.shape[1])
                != int(sim_body_linvel.shape[1])
            ):
                accel_buffers = tracking_metrics_warp_utils.create_metric_buffers(
                    env_count=int(sim_body_linvel.shape[0]),
                    body_count=int(sim_body_linvel.shape[1]),
                    device=sim_body_linvel.device,
                    dtype=sim_body_linvel.dtype,
                )
                summary["accel_buffers"] = accel_buffers
            tracking_metrics_warp_utils.compute_mean_acceleration_error_into_warp(
                sim_body_linvel,
                ref_body_linvel,
                prev_sim_body_linvel,
                prev_ref_body_linvel,
                dt=float(summary.get("dt", 1.0)),
                sim_root_quat=sim_root_quat,
                ref_root_quat=ref_root_quat,
                prev_sim_root_quat=prev_sim_root_quat,
                prev_ref_root_quat=prev_ref_root_quat,
                frame=str(summary.get("frame", "root")),
                out=accel_buffers,
            )
            accel_error = accel_buffers.mean_error
            scalar_tensors.append(accel_error.mean())
            scalar_names.append("accel")

        if (
            not isinstance(prev_sim_body_linvel, torch.Tensor)
            or prev_sim_body_linvel.shape != sim_body_linvel.shape
        ):
            prev_sim_body_linvel = torch.empty_like(sim_body_linvel)
            summary["prev_sim_body_linvel"] = prev_sim_body_linvel
        if (
            not isinstance(prev_ref_body_linvel, torch.Tensor)
            or prev_ref_body_linvel.shape != ref_body_linvel.shape
        ):
            prev_ref_body_linvel = torch.empty_like(ref_body_linvel)
            summary["prev_ref_body_linvel"] = prev_ref_body_linvel
        if (
            not isinstance(prev_sim_root_quat, torch.Tensor)
            or prev_sim_root_quat.shape != sim_root_quat.shape
        ):
            prev_sim_root_quat = torch.empty_like(sim_root_quat)
            summary["prev_sim_root_quat"] = prev_sim_root_quat
        if (
            not isinstance(prev_ref_root_quat, torch.Tensor)
            or prev_ref_root_quat.shape != ref_root_quat.shape
        ):
            prev_ref_root_quat = torch.empty_like(ref_root_quat)
            summary["prev_ref_root_quat"] = prev_ref_root_quat

        prev_sim_body_linvel.copy_(sim_body_linvel)
        prev_ref_body_linvel.copy_(ref_body_linvel)
        prev_sim_root_quat.copy_(sim_root_quat)
        prev_ref_root_quat.copy_(ref_root_quat)

    scalar_values = (
        torch.stack([tensor.to(torch.float32) for tensor in scalar_tensors])
        .detach()
        .cpu()
        .tolist()
    )
    scalar_map = {
        name: float(value)
        for name, value in zip(scalar_names, scalar_values, strict=True)
    }
    metrics["ape_scalar"] = scalar_map["ape"]
    metrics["ee_error_scalar"] = scalar_map["ee_error"]
    summary["ape_sum"] = float(summary["ape_sum"]) + scalar_map["ape"]
    summary["ee_sum"] = float(summary["ee_sum"]) + scalar_map["ee_error"]
    summary["ape_last"] = scalar_map["ape"]
    summary["ee_last"] = scalar_map["ee_error"]
    if "vel" in scalar_map:
        summary["vel_sum"] = float(summary["vel_sum"]) + scalar_map["vel"]
        summary["vel_last"] = scalar_map["vel"]
        summary["temporal_steps"] = int(summary["temporal_steps"]) + 1
        metrics["vel_scalar"] = scalar_map["vel"]
    if "accel" in scalar_map:
        summary["accel_sum"] = float(summary["accel_sum"]) + scalar_map["accel"]
        summary["accel_last"] = scalar_map["accel"]
        summary["accel_steps"] = int(summary["accel_steps"]) + 1
        metrics["accel_scalar"] = scalar_map["accel"]


def _tracking_metric_scalar(metrics: dict[str, Any] | None, key: str) -> float | None:
    if metrics is None:
        return None
    scalar = metrics.get(f"{key}_scalar")
    if isinstance(scalar, (float, int)):
        return float(scalar)
    value = metrics.get(key)
    if isinstance(value, torch.Tensor):
        return float(value.mean().detach().cpu().item())
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _finalize_tracking_metric_summary(
    summary: dict[str, Any],
) -> dict[str, float] | None:
    steps = int(summary["steps"])
    if steps <= 0:
        return None
    ape_last = summary["ape_last"]
    ee_last = summary["ee_last"]
    if ape_last is None or ee_last is None:
        return None
    return {
        "steps": float(steps),
        "ape_mean": float(summary["ape_sum"]) / float(steps),
        "ape_last": float(ape_last),
        "ee_mean": float(summary["ee_sum"]) / float(steps),
        "ee_last": float(ee_last),
        "vel_mean": (
            float(summary["vel_sum"]) / float(max(1, int(summary["temporal_steps"])))
            if int(summary.get("temporal_steps", 0)) > 0
            else math.nan
        ),
        "vel_last": (
            float(summary["vel_last"])
            if summary.get("vel_last") is not None
            else math.nan
        ),
        "accel_mean": (
            float(summary["accel_sum"]) / float(max(1, int(summary["accel_steps"])))
            if int(summary.get("accel_steps", 0)) > 0
            else math.nan
        ),
        "accel_last": (
            float(summary["accel_last"])
            if summary.get("accel_last") is not None
            else math.nan
        ),
    }


def _episode_done(terminated: torch.Tensor, truncated: torch.Tensor) -> bool:
    return bool(torch.logical_or(terminated, truncated).any().item())


def _episode_done_naturally(
    env_raw: Any, terminated: torch.Tensor, truncated: torch.Tensor
) -> bool:
    """Return whether every finished episode reached its natural completion.

    The environment's evaluation contract marks completions such as the end of
    a non-looping reference clip.  They satisfy a full-horizon requirement
    even before a longer requested playback length.
    """

    getter = getattr(env_raw, "get_eval_episode_spec", None)
    if not callable(getter):
        return False
    spec = getter()
    if not isinstance(spec, EvalEpisodeSpec):
        raise TypeError("env.get_eval_episode_spec() must return EvalEpisodeSpec")
    natural_completion = spec.natural_completion
    if natural_completion is None:
        return False
    done = torch.logical_or(terminated, truncated)
    natural = truncated & natural_completion & ~terminated
    return bool(done.any().item()) and not bool((done & ~natural).any().item())


def _update_playback_done_counters(
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    counters: dict[str, int],
) -> tuple[bool, bool]:
    terminated_done = bool(terminated.any().item())
    truncated_done = bool(truncated.any().item())
    if terminated_done or truncated_done:
        counters["episode_finish_count"] = (
            int(counters.get("episode_finish_count", 0)) + 1
        )
        counters["terminated_count"] = int(counters.get("terminated_count", 0)) + int(
            terminated_done
        )
        counters["truncated_count"] = int(counters.get("truncated_count", 0)) + int(
            truncated_done
        )
    return terminated_done, truncated_done


def _playback_nvtx_range(name: str, *, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.cuda.nvtx.range(name)


def _configure_playback_env_cuda_graph_mode(env_raw: Any, *, take_video: bool) -> None:
    """Preserve env step CUDA graph in playback modes."""
    cfg = getattr(env_raw, "cfg", None)
    if cfg is None or not bool(getattr(cfg, "use_cuda_graph", False)):
        return

    reasons: list[str] = []
    if take_video:
        reasons.append("video playback")
    if not reasons:
        return

    _LOGGER.info(
        "Preserving env.use_cuda_graph for %s; playback must remain graph-safe instead of disabling CUDA graph.",
        " + ".join(reasons),
    )


# Control helpers
def _can_use_fast_headless_playback(
    *, env_raw: Any, take_video: bool, play_length: int, stochastic_play: bool
) -> bool:
    if take_video or play_length <= 0:
        return False
    if stochastic_play:
        return False
    return callable(getattr(env_raw, "step_with_io", None))


def _can_use_playback_cuda_graph(
    *,
    requested: bool,
    device: torch.device,
    env_raw: Any,
    take_video: bool,
    play_length: int,
    stochastic_play: bool,
) -> bool:
    if not requested or device.type != "cuda":
        return False
    if take_video or play_length <= 0:
        return False
    return callable(getattr(env_raw, "step_with_io", None))


# CUDA graph helpers
@dataclass(frozen=True, slots=True)
class PlaybackGraphRunnerOptions:
    """Shared policy buffers and stream settings for playback graph runners."""

    env: Any
    env_raw: Any
    agent: PPOAgent
    device: torch.device
    obs_policy: torch.Tensor
    agent_actions: torch.Tensor
    env_actions: torch.Tensor
    play_torch_stream: Any
    play_wp_stream: Any
    stochastic_play: bool = False


class _PlaybackGraphRunnerBase:
    """Shared playback buffers and CUDA stream state."""

    def __init__(self, options: PlaybackGraphRunnerOptions) -> None:
        self._env = options.env
        self._env_raw = options.env_raw
        self._agent = options.agent
        self._device = options.device
        self._obs_policy = options.obs_policy
        self._agent_actions = options.agent_actions
        self._env_actions = options.env_actions
        self._torch_stream = options.play_torch_stream
        self._wp_stream = options.play_wp_stream
        self._stochastic_play = bool(options.stochastic_play)
        self._next_obs_policy = torch.empty_like(options.obs_policy)
        self._next_rewards = torch.empty(
            options.obs_policy.shape[0], device=options.device, dtype=torch.float32
        )
        self._next_terminated = torch.empty(
            options.obs_policy.shape[0], device=options.device, dtype=torch.bool
        )
        self._next_truncated = torch.empty_like(self._next_terminated)
        self._next_dones = torch.empty_like(self._next_terminated)
        self._step_io = StepIO(
            inputs=StepInputs(actions=self._env_actions),
            outputs=StepOutputs(
                observations=agent_observation_output_views(
                    agent=self._agent, destination=self._next_obs_policy
                ),
                rewards=self._next_rewards,
                terminated=self._next_terminated,
                truncated=self._next_truncated,
            ),
        )
        self._log_probs = torch.empty(
            options.obs_policy.shape[0], 1, device=options.device, dtype=torch.float32
        )
        self._values = torch.empty_like(self._log_probs)
        self._graph: Any | None = None
        self._nvtx_enabled = options.device.type == "cuda"

    def _nvtx_range(self, name: str):
        return _playback_nvtx_range(name, enabled=self._nvtx_enabled)

    def _synchronize(self) -> None:
        with self._nvtx_range("playback_sync_stream"):
            torch.cuda.current_stream(self._device).wait_stream(self._torch_stream)
            self._torch_stream.synchronize()


class _HeadlessPlaybackGraphRunner(_PlaybackGraphRunnerBase):
    def reset_from_env(self) -> None:
        with self._nvtx_range("playback_reset_env"):
            obs, _ = self._env.reset()
            copy_agent_observation_into(
                agent=self._agent, observations=obs, destination=self._obs_policy
            )
            self._next_obs_policy.zero_()
            self._next_rewards.zero_()
            self._next_terminated.zero_()
            self._next_truncated.zero_()
            self._next_dones.zero_()

    def _step_graph_ops(self) -> None:
        with self._nvtx_range("playback_policy"):
            if self._stochastic_play:
                _policy_sample_graph_safe_into(
                    self._agent,
                    self._obs_policy,
                    self._agent_actions,
                    self._log_probs,
                    self._values,
                )
            else:
                with network_matmul_precision():
                    self._agent.policy_execution_mean_into(
                        self._obs_policy,
                        self._agent_actions,
                    )
        with self._nvtx_range("playback_env_step"):
            self._env_raw.step_with_io(
                self._step_io,
                use_cuda_graph=False,
            )
            torch.logical_or(
                self._next_terminated,
                self._next_truncated,
                out=self._next_dones,
            )
        with self._nvtx_range("playback_obs_update"):
            self._obs_policy.copy_(self._next_obs_policy)

    def capture(self) -> None:
        if self._graph is not None:
            return

        self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self._torch_stream):
                with self._nvtx_range("playback_graph_warmup"):
                    for _ in range(2):
                        self._step_graph_ops()
        self._synchronize()

        self.reset_from_env()
        self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with external_step_capture_window(self._env_raw):
            with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
                with torch.cuda.stream(self._torch_stream):
                    with self._nvtx_range("playback_graph_capture"):
                        with capture_warp_graph(stream=self._wp_stream) as capture:
                            self._step_graph_ops()
            self._graph = capture.graph
            self._synchronize()

        # Capture executes the step body once; restore deterministic playback start.
        self.reset_from_env()

    def step(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._graph is None:
            raise RuntimeError("Playback CUDA graph has not been captured yet.")
        self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with external_step_capture_window(self._env_raw):
            with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
                with torch.cuda.stream(self._torch_stream):
                    with self._nvtx_range("playback_graph_replay"):
                        wp.capture_launch(self._graph, stream=self._wp_stream)
            self._synchronize()
        return self._next_rewards, self._next_dones


class _PlaybackTrackingMetricsGraphRunner:
    def __init__(
        self,
        *,
        env_raw: Any,
        device: torch.device,
        play_torch_stream: torch.cuda.Stream,
        play_wp_stream: Any,
        frame: str,
        include_root_in_ape: bool,
        dt: float,
    ) -> None:
        self._env_raw = env_raw
        self._device = device
        self._torch_stream = play_torch_stream
        self._wp_stream = play_wp_stream
        self._frame = str(frame)
        self._include_root_in_ape = bool(include_root_in_ape)
        self._dt = float(dt)
        self._graph: Any | None = None
        self._metrics: dict[str, Any] | None = None
        self._result: dict[str, Any] | None = None
        self._vel_buffers = None
        self._accel_buffers = None
        self._prev_sim_body_linvel: torch.Tensor | None = None
        self._prev_ref_body_linvel: torch.Tensor | None = None
        self._prev_sim_root_quat: torch.Tensor | None = None
        self._prev_ref_root_quat: torch.Tensor | None = None
        self._step_count = 0
        self._nvtx_enabled = device.type == "cuda"

    def _nvtx_range(self, name: str):
        return _playback_nvtx_range(name, enabled=self._nvtx_enabled)

    def _synchronize(self) -> None:
        with self._nvtx_range("playback_tracking_metrics_sync"):
            torch.cuda.current_stream(self._device).wait_stream(self._torch_stream)
            self._torch_stream.synchronize()

    def _ensure_buffers(self) -> None:
        if self._metrics is not None:
            return
        metrics = _maybe_collect_tracking_metrics(
            self._env_raw,
            frame=self._frame,
            include_root_in_ape=self._include_root_in_ape,
        )
        if metrics is None:
            raise RuntimeError(
                "Tracking metrics graph requested, but env did not return valid tracking metrics."
            )
        self._metrics = metrics
        self._result = dict(metrics)
        sim_body_linvel = metrics.get("tracking_body_linvel")
        ref_body_linvel = metrics.get("reference_body_linvel")
        sim_root_quat = metrics.get("sim_root_quat")
        ref_root_quat = metrics.get("ref_root_quat")
        if (
            isinstance(sim_body_linvel, torch.Tensor)
            and isinstance(ref_body_linvel, torch.Tensor)
            and isinstance(sim_root_quat, torch.Tensor)
            and isinstance(ref_root_quat, torch.Tensor)
        ):
            self._vel_buffers = tracking_metrics_warp_utils.create_metric_buffers(
                env_count=int(sim_body_linvel.shape[0]),
                body_count=int(sim_body_linvel.shape[1]),
                device=sim_body_linvel.device,
                dtype=sim_body_linvel.dtype,
            )
            self._accel_buffers = tracking_metrics_warp_utils.create_metric_buffers(
                env_count=int(sim_body_linvel.shape[0]),
                body_count=int(sim_body_linvel.shape[1]),
                device=sim_body_linvel.device,
                dtype=sim_body_linvel.dtype,
            )
            self._prev_sim_body_linvel = torch.zeros_like(sim_body_linvel)
            self._prev_ref_body_linvel = torch.zeros_like(ref_body_linvel)
            self._prev_sim_root_quat = torch.zeros_like(sim_root_quat)
            self._prev_ref_root_quat = torch.zeros_like(ref_root_quat)
            self._result["vel_error"] = self._vel_buffers.mean_error
            self._result["accel_error"] = self._accel_buffers.mean_error
            self._result["accel_valid"] = False

    def _reset_state(self) -> None:
        self._step_count = 0
        if self._prev_sim_body_linvel is not None:
            self._prev_sim_body_linvel.zero_()
        if self._prev_ref_body_linvel is not None:
            self._prev_ref_body_linvel.zero_()
        if self._prev_sim_root_quat is not None:
            self._prev_sim_root_quat.zero_()
        if self._prev_ref_root_quat is not None:
            self._prev_ref_root_quat.zero_()

    def _metric_ops(self) -> None:
        metrics = _maybe_collect_tracking_metrics(
            self._env_raw,
            frame=self._frame,
            include_root_in_ape=self._include_root_in_ape,
        )
        assert metrics is not None
        if self._metrics is None:
            self._metrics = metrics
        sim_body_linvel = metrics.get("tracking_body_linvel")
        ref_body_linvel = metrics.get("reference_body_linvel")
        sim_root_quat = metrics.get("sim_root_quat")
        ref_root_quat = metrics.get("ref_root_quat")
        if (
            self._vel_buffers is not None
            and self._accel_buffers is not None
            and self._prev_sim_body_linvel is not None
            and self._prev_ref_body_linvel is not None
            and self._prev_sim_root_quat is not None
            and self._prev_ref_root_quat is not None
            and isinstance(sim_body_linvel, torch.Tensor)
            and isinstance(ref_body_linvel, torch.Tensor)
            and isinstance(sim_root_quat, torch.Tensor)
            and isinstance(ref_root_quat, torch.Tensor)
        ):
            tracking_metrics_warp_utils.compute_mean_velocity_error_into_warp(
                sim_body_linvel,
                ref_body_linvel,
                sim_root_quat=sim_root_quat,
                ref_root_quat=ref_root_quat,
                frame=self._frame,
                out=self._vel_buffers,
            )
            tracking_metrics_warp_utils.compute_mean_acceleration_error_into_warp(
                sim_body_linvel,
                ref_body_linvel,
                self._prev_sim_body_linvel,
                self._prev_ref_body_linvel,
                dt=self._dt,
                sim_root_quat=sim_root_quat,
                ref_root_quat=ref_root_quat,
                prev_sim_root_quat=self._prev_sim_root_quat,
                prev_ref_root_quat=self._prev_ref_root_quat,
                frame=self._frame,
                out=self._accel_buffers,
            )
            self._prev_sim_body_linvel.copy_(sim_body_linvel)
            self._prev_ref_body_linvel.copy_(ref_body_linvel)
            self._prev_sim_root_quat.copy_(sim_root_quat)
            self._prev_ref_root_quat.copy_(ref_root_quat)

    def capture(self) -> None:
        if self._graph is not None:
            return
        self._ensure_buffers()
        self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self._torch_stream):
                with self._nvtx_range("playback_tracking_metrics_graph_warmup"):
                    self._metric_ops()
        self._synchronize()
        self._reset_state()
        self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self._torch_stream):
                with self._nvtx_range("playback_tracking_metrics_graph_capture"):
                    with capture_warp_graph(stream=self._wp_stream) as capture:
                        self._metric_ops()
        self._graph = capture.graph
        self._synchronize()
        self._reset_state()

    def step(self) -> dict[str, Any] | None:
        if self._graph is None:
            raise RuntimeError("Tracking metrics CUDA graph has not been captured yet.")
        self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self._torch_stream):
                with self._nvtx_range("playback_tracking_metrics_graph_replay"):
                    wp.capture_launch(self._graph, stream=self._wp_stream)
        self._synchronize()
        if self._result is None:
            return None
        self._result["accel_valid"] = self._step_count > 0
        self._step_count += 1
        return self._result


def _can_use_playback_video_device_graph(
    *, requested: bool, device: torch.device, take_video: bool, stochastic_play: bool
) -> bool:
    if not requested or device.type != "cuda":
        return False
    if not take_video:
        return False
    return True


def _resolve_playback_graph_mode(
    *,
    requested: bool,
    device: torch.device,
    env_raw: Any,
    take_video: bool,
    play_length: int,
    stochastic_play: bool,
) -> Literal["none", "headless", "video_device_step"]:
    if _can_use_playback_video_device_graph(
        requested=requested,
        device=device,
        take_video=take_video,
        stochastic_play=stochastic_play,
    ):
        return "video_device_step"
    if _can_use_playback_cuda_graph(
        requested=requested,
        device=device,
        env_raw=env_raw,
        take_video=take_video,
        play_length=play_length,
        stochastic_play=stochastic_play,
    ):
        return "headless"
    return "none"


def _resolve_playback_length(
    *,
    cfg: DictConfig,
    take_video: bool,
    headless_play_steps: int,
    env_raw: Any | None = None,
) -> int:
    if headless_play_steps > 0:
        return int(headless_play_steps)
    if take_video:
        raw_video_length = cfg.get("video_length", "episode")
        if str(raw_video_length).strip().lower() in _PLAYBACK_LENGTH_EPISODE_TOKENS:
            max_episode_length = getattr(env_raw, "max_episode_length", None)
            if max_episode_length is None:
                raise ValueError(
                    "video_length=episode requires an environment with max_episode_length."
                )
            return int(max_episode_length)
        return int(raw_video_length)
    return 0


def _maybe_extend_playback_episode_length(
    env_raw: Any,
    *,
    play_length: int,
    time_limit_policy: Literal["extend", "respect"] = _PLAY_TIME_LIMIT_POLICY_EXTEND,
) -> tuple[int | None, int | None]:
    policy = str(time_limit_policy).strip().lower()
    if policy not in _PLAY_TIME_LIMIT_POLICIES:
        valid = ", ".join(sorted(_PLAY_TIME_LIMIT_POLICIES))
        raise ValueError(
            f"Unsupported play_time_limit_policy={policy!r}. Valid values: {valid}."
        )
    max_episode_length = getattr(env_raw, "max_episode_length", None)
    if max_episode_length is None:
        return None, None
    prev_max_episode_length = int(max_episode_length)
    if (
        play_length < prev_max_episode_length
        or policy == _PLAY_TIME_LIMIT_POLICY_RESPECT
    ):
        return prev_max_episode_length, prev_max_episode_length
    if hasattr(env_raw, "cfg") and hasattr(env_raw.cfg, "episode_length_s"):
        target_episode_steps = int(play_length) + 2
        target_episode_length_s = target_episode_steps * float(env_raw.step_dt)
        prev_episode_length_s = float(env_raw.cfg.episode_length_s)
        if target_episode_length_s > prev_episode_length_s:
            env_raw.cfg.episode_length_s = target_episode_length_s
        return prev_max_episode_length, int(env_raw.max_episode_length)
    return prev_max_episode_length, prev_max_episode_length


class _PlaybackVideoStepGraphRunner(_PlaybackGraphRunnerBase):
    def __init__(
        self,
        options: PlaybackGraphRunnerOptions,
        *,
        compiled_playback_policy: Any,
        tracking_metrics_runner: Any,
    ) -> None:
        super().__init__(options)
        self._compiled_playback_policy = compiled_playback_policy
        self._tracking_metrics_runner = tracking_metrics_runner
        self._step_count = 0

    @property
    def terminated(self) -> torch.Tensor:
        return self._next_terminated

    @property
    def truncated(self) -> torch.Tensor:
        return self._next_truncated

    def reset_from_env(self) -> None:
        with self._nvtx_range("playback_reset_env"):
            obs, _ = self._env.reset()
            copy_agent_observation_into(
                agent=self._agent, observations=obs, destination=self._obs_policy
            )
            self._next_obs_policy.zero_()
            self._next_rewards.zero_()
            self._next_terminated.zero_()
            self._next_truncated.zero_()
            self._next_dones.zero_()
            if self._tracking_metrics_runner is not None:
                self._tracking_metrics_runner._reset_state()
            self._step_count = 0

    def _device_step_ops(self) -> None:
        if self._tracking_metrics_runner is not None:
            with self._nvtx_range("playback_tracking_metrics"):
                self._tracking_metrics_runner._metric_ops()
        with self._nvtx_range("playback_policy"):
            if self._stochastic_play:
                _policy_sample_graph_safe_into(
                    self._agent,
                    self._obs_policy,
                    self._agent_actions,
                    self._log_probs,
                    self._values,
                )
            else:
                compiled_execution_mean = self._compiled_playback_policy.get(
                    "execution_mean"
                )
                if callable(compiled_execution_mean):
                    compiled_execution_mean(self._obs_policy, self._agent_actions)
                else:
                    with network_matmul_precision():
                        self._agent.policy_execution_mean_into(
                            self._obs_policy,
                            self._agent_actions,
                        )
        with self._nvtx_range("playback_env_step"):
            self._env_raw.step_with_io(
                self._step_io,
                use_cuda_graph=False,
            )
            torch.logical_or(
                self._next_terminated,
                self._next_truncated,
                out=self._next_dones,
            )
        with self._nvtx_range("playback_obs_update"):
            self._obs_policy.copy_(self._next_obs_policy)

    def capture(self) -> None:
        if self._graph is not None:
            return
        if self._tracking_metrics_runner is not None:
            self._tracking_metrics_runner._ensure_buffers()
        self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self._torch_stream):
                with self._nvtx_range("playback_device_step_graph_warmup"):
                    for _ in range(2):
                        self._device_step_ops()
        self._synchronize()
        self.reset_from_env()
        self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with external_step_capture_window(self._env_raw):
            with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
                with torch.cuda.stream(self._torch_stream):
                    with self._nvtx_range("playback_device_step_graph_capture"):
                        with capture_warp_graph(stream=self._wp_stream) as capture:
                            self._device_step_ops()
            self._graph = capture.graph
            self._synchronize()
        self.reset_from_env()

    def step(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
        if self._graph is None:
            raise RuntimeError(
                "Playback video step CUDA graph has not been captured yet."
            )
        self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with external_step_capture_window(self._env_raw):
            with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
                with torch.cuda.stream(self._torch_stream):
                    with self._nvtx_range("playback_device_step_graph_replay"):
                        wp.capture_launch(self._graph, stream=self._wp_stream)
            self._synchronize()
        metrics = None
        if self._tracking_metrics_runner is not None:
            metrics = self._tracking_metrics_runner._result
            if metrics is not None:
                metrics["accel_valid"] = self._step_count > 0
        self._step_count += 1
        return self._next_rewards, self._next_dones, metrics


@torch.no_grad()
def _run_fast_headless_playback(
    *,
    env: Any,
    env_raw: Any,
    agent: PPOAgent,
    obs_policy: torch.Tensor,
    agent_actions: torch.Tensor,
    env_actions: torch.Tensor,
    play_length: int,
    auto_resets: bool,
    tracking_metric_frame: str,
    tracking_metric_include_root: bool,
    tracking_metric_summary: dict[str, Any],
    tracking_require_full_horizon: bool,
    playback_done_counters: dict[str, int] | None = None,
    graph_runner: _HeadlessPlaybackGraphRunner | None = None,
    play_done_policy: Literal[
        "reset_continue", "stop"
    ] = _PLAY_DONE_POLICY_RESET_CONTINUE,
) -> dict[str, Any]:
    next_obs_policy = torch.empty_like(obs_policy)
    current_observation_views = agent_observation_output_views(
        agent=agent, destination=obs_policy
    )
    next_observation_views = agent_observation_output_views(
        agent=agent, destination=next_obs_policy
    )
    next_rewards = torch.empty(
        obs_policy.shape[0], device=obs_policy.device, dtype=torch.float32
    )
    next_terminated = torch.empty(
        obs_policy.shape[0], device=obs_policy.device, dtype=torch.bool
    )
    next_truncated = torch.empty_like(next_terminated)
    next_dones = torch.empty_like(next_terminated)

    episode_reward = 0.0
    episode_length = 0
    best_episode_reward = 0.0
    best_episode_length = 0
    episode_index = 1
    tracking_failure_info: dict[str, float | int | bool] | None = None
    done_counters = playback_done_counters if playback_done_counters is not None else {}
    goal_reach_counts = _initialize_goal_reach_counts(
        env_raw,
        num_envs=int(obs_policy.shape[0]),
        device=obs_policy.device,
    )
    executed_steps = 0

    _LOGGER.info(
        "Starting fast headless PPO playback for %d steps via env.step_with_io().",
        play_length,
    )
    for play_step_idx in range(play_length):
        tracking_metrics = _maybe_collect_tracking_metrics(
            env_raw,
            frame=tracking_metric_frame,
            include_root_in_ape=tracking_metric_include_root,
        )
        _update_tracking_metric_summary(tracking_metric_summary, tracking_metrics)
        if graph_runner is not None:
            next_rewards, next_dones = graph_runner.step()
            step_terminated = graph_runner._next_terminated
            step_truncated = graph_runner._next_truncated
        else:
            policy_mean_into(agent, obs_policy, agent_actions)
            _, next_rewards, step_terminated, step_truncated, _ = (
                _step_playback_env_with_io(
                    env_raw=env_raw,
                    env_actions=env_actions,
                    next_obs_policy=next_obs_policy,
                    next_rewards=next_rewards,
                    next_dones=next_dones,
                    next_terminated=next_terminated,
                    next_truncated=next_truncated,
                    observation_views=next_observation_views,
                )
            )
            obs_policy, next_obs_policy = next_obs_policy, obs_policy
            current_observation_views, next_observation_views = (
                next_observation_views,
                current_observation_views,
            )
        step_reward = float(next_rewards.mean().item())
        _accumulate_goal_reaches(env_raw, goal_reach_counts)
        executed_steps += 1
        episode_reward += step_reward
        episode_length += 1

        if bool(next_dones.any().item()):
            done_step = play_step_idx + 1
            done_before_horizon = done_step < play_length
            terminated_done, truncated_done = _update_playback_done_counters(
                step_terminated,
                step_truncated,
                done_counters,
            )
            if episode_length > best_episode_length or (
                episode_length == best_episode_length
                and episode_reward > best_episode_reward
            ):
                best_episode_length = int(episode_length)
                best_episode_reward = float(episode_reward)
            if (
                done_before_horizon
                and tracking_require_full_horizon
                and not _episode_done_naturally(
                    env_raw, step_terminated, step_truncated
                )
            ):
                tracking_failure_info = {
                    "step": int(done_step),
                    "terminated": terminated_done,
                    "truncated": truncated_done,
                    "play_length": int(play_length),
                }
                _LOGGER.warning(
                    "Tracking evaluation invalidated: episode ended at step %d/%d before the requested horizon.",
                    done_step,
                    play_length,
                )
                break
            if _playback_done_policy_stops(play_done_policy):
                break
            episode_index += 1
            episode_reward = 0.0
            episode_length = 0
            if not auto_resets:
                if graph_runner is not None:
                    graph_runner.reset_from_env()
                else:
                    obs, _ = env.reset()
                    copy_agent_observation_into(
                        agent=agent, observations=obs, destination=obs_policy
                    )

    result: dict[str, Any] = {
        "episode_reward": float(episode_reward),
        "episode_length": int(episode_length),
        "best_episode_reward": float(best_episode_reward),
        "best_episode_length": int(best_episode_length),
        "episode_index": int(episode_index),
        "tracking_failure_info": tracking_failure_info,
    }
    result.update(_summarize_goal_reaches(goal_reach_counts, eval_steps=executed_steps))
    return result


def _step_playback_env_with_io(
    *,
    env_raw: Any,
    env_actions: torch.Tensor,
    next_obs_policy: torch.Tensor,
    next_rewards: torch.Tensor,
    next_dones: torch.Tensor,
    next_terminated: torch.Tensor,
    next_truncated: torch.Tensor,
    reset_done: bool = True,
    agent: Any = None,
    observation_views: Mapping[str, torch.Tensor] | None = None,
) -> tuple[
    Mapping[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict,
]:
    step_kwargs: dict[str, Any] = {"use_cuda_graph": False}
    if not bool(reset_done):
        step_kwargs["reset_done"] = False
    outputs, extras = env_raw.step_with_io(
        StepIO(
            inputs=StepInputs(actions=env_actions),
            outputs=StepOutputs(
                observations=(
                    observation_views
                    if observation_views is not None
                    else agent_observation_output_views(
                        agent=agent, destination=next_obs_policy
                    )
                ),
                rewards=next_rewards,
                terminated=next_terminated,
                truncated=next_truncated,
            ),
        ),
        **step_kwargs,
    )
    torch.logical_or(
        outputs.terminated,
        outputs.truncated,
        out=next_dones,
    )
    return (
        outputs.observations,
        outputs.rewards,
        outputs.terminated,
        outputs.truncated,
        extras,
    )
