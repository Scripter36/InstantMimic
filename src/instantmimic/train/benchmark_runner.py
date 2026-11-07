from __future__ import annotations

import copy
import logging
import math
import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
import warp as wp
from omegaconf import OmegaConf, open_dict

from instantmimic.envs.config_access import get_value
from instantmimic.envs.direct_rl_env import (
    StepInputs,
    StepIO,
    StepOutputs,
    external_step_capture_window,
)
from instantmimic.envs.eval_metrics import (
    EvalEpisodeSpec,
    EvalMetricAccumulator,
    EvalMetricSpec,
    EvalOutcomeSummary,
    load_eval_result_metric_specs,
    resolve_eval_episode_spec,
    resolve_eval_metric_specs,
    resolve_eval_result_metric_specs,
)
from instantmimic.envs.tracking import metrics as tracking_metrics
from instantmimic.envs.tracking import metrics_warp as tracking_metrics_warp
from instantmimic.rl.runtime import (
    agent_observation_output_views,
    copy_agent_observation_into,
)
from instantmimic.utils.network_precision import network_matmul_precision
from instantmimic.utils.policy_mean_contract import (
    matches_policy_mean_bound_callable,
    matches_policy_mean_methods,
)
from instantmimic.utils.torch_warp_capture import (
    capture_warp_graph,
    warp_stream_from_torch,
)

_LOGGER = logging.getLogger(__name__)
_BUILTIN_ROLLOUT_METRICS = {
    "score",
    "valid_rate",
    "success_rate",
    "reward_mean",
    "episode_length_mean",
}

_MOTION_GROUP_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

_AFFINE_STEP_TYPES = (torch.nn.Linear,)


def _is_affine_step(module: object) -> bool:
    return isinstance(module, _AFFINE_STEP_TYPES)


class _AffineParameterSource:
    """Where one preallocated affine step refreshes its parameters from."""

    def __init__(
        self,
        module: torch.nn.Module,
    ) -> None:
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(f"unsupported affine policy step: {type(module).__name__}")
        if module.bias is None:
            raise ValueError("policy affine steps require a bias")
        reference = module.weight
        self._weight = module.weight.detach()
        self._bias = module.bias.detach()
        self.out_features = int(module.out_features)
        self.in_features = int(module.in_features)
        self._device = reference.device
        self._dtype = reference.dtype

    def empty_weight(self) -> torch.Tensor:
        return torch.empty(
            self.out_features,
            self.in_features,
            device=self._device,
            dtype=self._dtype,
        )

    def empty_bias(self) -> torch.Tensor:
        return torch.empty(self.out_features, device=self._device, dtype=self._dtype)

    def write(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        weight.copy_(self._weight)
        bias.copy_(self._bias)


def _owned_policy_mean_matches(agent: Any, network: Any) -> bool:
    """Certify the ordinary policy and normalizer methods owned by this runtime."""
    from torch.nn.modules import module as module_hooks

    if module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks:
        return False
    if (
        network is not getattr(agent, "network", None)
        or not matches_policy_mean_methods(agent, "ppo")
        or not (
            matches_policy_mean_methods(network, "actor_critic")
            or matches_policy_mean_methods(network, "latent_prior")
        )
        or not any(
            matches_policy_mean_methods(getattr(agent, "obs_scaler", None), role)
            for role in ("standard_scaler", "rsl_scaler")
        )
        or not matches_policy_mean_bound_callable(
            agent,
            "ppo",
            "_obs_scale_eval_into",
            getattr(agent, "_obs_scale_eval_into_fn", None),
        )
    ):
        return False
    for root in (network, agent.obs_scaler):
        for module in root.modules():
            if module._forward_hooks or module._forward_pre_hooks:
                return False
    return True


def _native_policy_snapshot_matches(agent: Any, network: Any) -> bool:
    """Certify the FP32 MLP arithmetic implemented by the native snapshot."""
    if (
        not _owned_policy_mean_matches(agent, network)
        or not matches_policy_mean_methods(agent.obs_scaler, "standard_scaler")
        or bool(getattr(network, "policy_mlp_bf16", False))
    ):
        return False
    trunk = getattr(network, "actor_trunk", None)
    if trunk is None:
        trunk = getattr(network, "trunk", None)
    supported = (
        torch.nn.Sequential,
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.ELU,
        torch.nn.Identity,
        torch.nn.LeakyReLU,
        torch.nn.ReLU,
        torch.nn.SiLU,
        torch.nn.Tanh,
    )
    if trunk is None or any(
        type(module) not in supported
        or getattr(module.forward, "__func__", None) is not type(module).forward
        or getattr(module.forward, "__self__", None) is not module
        for module in trunk.modules()
    ):
        return False
    head = getattr(network, "policy_head", None)
    if head is None:
        return False
    head_type = type(head)
    return bool(
        head_type in _AFFINE_STEP_TYPES
        and getattr(head.forward, "__func__", None) is head_type.forward
        and getattr(head.forward, "__self__", None) is head
        and all(
            type(module) is torch.nn.Linear
            and getattr(module.forward, "__func__", None) is torch.nn.Linear.forward
            for module in head.children()
        )
    )


@dataclass(frozen=True)
class BenchmarkMotionGroup:
    name: str
    clip_index: int

    def __post_init__(self) -> None:
        if not _MOTION_GROUP_NAME_RE.fullmatch(self.name):
            raise ValueError(
                "Benchmark motion group names must start with a letter and contain "
                "only letters, digits, and underscores."
            )
        if self.clip_index < 0:
            raise ValueError("Benchmark motion group clip_index must be >= 0.")


def _benchmark_motion_groups(cfg: Any) -> tuple[BenchmarkMotionGroup, ...]:
    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        return ()
    benchmark_cfg = get_value(trainer_cfg, "benchmark", None)
    if benchmark_cfg is None:
        return ()
    raw_groups = get_value(benchmark_cfg, "motion_groups", None)
    if raw_groups is None:
        return ()
    if not isinstance(raw_groups, Sequence) or isinstance(
        raw_groups, (str, bytes, bytearray)
    ):
        raise ValueError("trainer.benchmark.motion_groups must be a sequence.")
    groups: list[BenchmarkMotionGroup] = []
    for index, item in enumerate(raw_groups):
        if not isinstance(item, Mapping) and not hasattr(item, "get"):
            raise ValueError(
                f"trainer.benchmark.motion_groups.{index} must be a mapping."
            )
        groups.append(
            BenchmarkMotionGroup(
                name=str(item.get("name", "")).strip(),
                clip_index=int(item.get("clip_index", -1)),
            )
        )
    if not groups:
        raise ValueError("trainer.benchmark.motion_groups must not be empty.")
    names = [group.name for group in groups]
    if len(set(names)) != len(names):
        raise ValueError("trainer.benchmark.motion_groups names must be unique.")
    clip_indices = [group.clip_index for group in groups]
    if len(set(clip_indices)) != len(clip_indices):
        raise ValueError(
            "trainer.benchmark.motion_groups clip_index values must be unique."
        )
    return tuple(groups)


def _benchmark_env_overrides(cfg: Any) -> Any | None:
    trainer_cfg = getattr(cfg, "trainer", None)
    if trainer_cfg is None:
        return None
    benchmark_cfg = get_value(trainer_cfg, "benchmark", None)
    if benchmark_cfg is None:
        return None
    raw_overrides = get_value(benchmark_cfg, "env_overrides", None)
    if raw_overrides is None:
        return None
    if OmegaConf.is_config(raw_overrides):
        if not OmegaConf.is_dict(raw_overrides):
            raise ValueError("trainer.benchmark.env_overrides must be a mapping.")
        override_data = OmegaConf.to_container(raw_overrides, resolve=True)
    elif isinstance(raw_overrides, Mapping):
        override_data = dict(raw_overrides)
    else:
        raise ValueError("trainer.benchmark.env_overrides must be a mapping.")
    return OmegaConf.create(override_data)


def _motion_group_assignment(
    groups: Sequence[BenchmarkMotionGroup],
    *,
    total_envs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not groups:
        raise ValueError("Motion-group assignment requires at least one group.")
    if total_envs < len(groups):
        raise ValueError(
            "trainer.benchmark.num_envs must be at least the number of motion groups."
        )
    group_ids = torch.arange(total_envs, device=device, dtype=torch.long).remainder(
        len(groups)
    )
    group_clip_indices = torch.tensor(
        [group.clip_index for group in groups],
        device=device,
        dtype=torch.int32,
    )
    clip_indices = group_clip_indices.index_select(0, group_ids)
    return group_ids, clip_indices


def _capture_eval_conditional_horizon(
    condition: Any,
    step_body: Any,
    **step_kwargs: Any,
) -> None:
    """Capture one device-controlled eval horizon without host unrolling."""
    wp.capture_while(condition, while_body=step_body, **step_kwargs)


def _resolve_motion_end_time_outs(
    env_raw: Any,
    *,
    fallback_time_outs: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    motion_end_time_outs = getattr(env_raw, "motion_end_time_outs", None)
    if (
        isinstance(motion_end_time_outs, torch.Tensor)
        and motion_end_time_outs.shape == fallback_time_outs.shape
    ):
        return motion_end_time_outs, True
    return fallback_time_outs, False


def _resolve_terminal_success_mask(
    env_raw: Any,
    *,
    fallback_terminated: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    reset_success = getattr(env_raw, "reset_success", None)
    if (
        isinstance(reset_success, torch.Tensor)
        and reset_success.shape == fallback_terminated.shape
        and reset_success.dtype == torch.bool
        and reset_success.device == fallback_terminated.device
    ):
        return reset_success, True
    return fallback_terminated, False


class _TrackingBenchmarkWorkspace:
    def __init__(
        self,
        *,
        device: torch.device,
        total_envs: int,
        obs_dim: int,
        policy_act_dim: int,
        env_act_dim: int,
        agent: Any = None,
    ) -> None:
        self.total_envs = int(total_envs)
        self.obs_dim = int(obs_dim)
        self.policy_act_dim = int(policy_act_dim)
        self.env_act_dim = int(env_act_dim)
        if self.policy_act_dim != self.env_act_dim:
            raise ValueError(
                "direct PPO policy and environment action widths must match"
            )

        self.obs_policy = torch.empty(
            self.total_envs,
            self.obs_dim,
            device=device,
            dtype=torch.float32,
        )
        self.policy_actions = torch.empty(
            self.total_envs,
            self.policy_act_dim,
            device=device,
            dtype=torch.float32,
        )
        self.env_actions = self.policy_actions
        self.next_rewards = torch.empty(
            self.total_envs,
            device=device,
            dtype=torch.float32,
        )
        self.next_terminated = torch.empty(
            self.total_envs,
            device=device,
            dtype=torch.bool,
        )
        self.next_truncated = torch.empty_like(self.next_terminated)
        self.next_dones = torch.empty(
            self.total_envs,
            device=device,
            dtype=torch.float32,
        )
        self.step_io = StepIO(
            inputs=StepInputs(actions=self.env_actions),
            outputs=StepOutputs(
                observations=agent_observation_output_views(
                    agent=agent, destination=self.obs_policy
                ),
                rewards=self.next_rewards,
                terminated=self.next_terminated,
                truncated=self.next_truncated,
            ),
        )

        self.active_mask = torch.empty(
            self.total_envs,
            device=device,
            dtype=torch.bool,
        )
        self.valid_mask = torch.empty_like(self.active_mask)
        self.success_mask = torch.empty_like(self.active_mask)
        self.reward_sum = torch.empty(
            self.total_envs,
            device=device,
            dtype=torch.float32,
        )
        self.ape_sum = torch.empty_like(self.reward_sum)
        self.global_mpjpe_sum = torch.empty_like(self.reward_sum)
        self.ee_sum = torch.empty_like(self.reward_sum)
        self.vel_sum = torch.empty_like(self.reward_sum)
        self.accel_sum = torch.empty_like(self.reward_sum)
        self.steps_per_env = torch.empty(
            self.total_envs,
            device=device,
            dtype=torch.int32,
        )
        self.accel_steps_per_env = torch.empty_like(self.steps_per_env)
        self.last_ape = torch.empty_like(self.reward_sum)
        self.last_global_mpjpe = torch.empty_like(self.reward_sum)
        self.last_ee = torch.empty_like(self.reward_sum)
        self.last_vel = torch.empty_like(self.reward_sum)
        self.last_accel = torch.empty_like(self.reward_sum)
        self.eval_condition = torch.empty(1, device=device, dtype=torch.int32)
        self.eval_condition_wp = wp.from_torch(
            self.eval_condition,
            dtype=wp.int32,
        )
        self.eval_step_index = torch.empty_like(self.eval_condition)
        self.eval_completed_steps = torch.empty_like(self.eval_condition)
        self.eval_first_invalid_step = torch.empty_like(self.eval_condition)
        self.eval_first_failure_step = torch.empty_like(self.eval_condition)
        self.eval_active_count = torch.empty_like(self.eval_condition)
        self.eval_terminated_any = torch.empty_like(self.eval_condition)
        self.eval_truncated_any = torch.empty_like(self.eval_condition)
        self.eval_reset_mask = torch.ones(
            self.total_envs,
            device=device,
            dtype=torch.bool,
        )
        self._temporal_body_count = -1
        self.vel_metric: tracking_metrics_warp.MetricBuffers | None = None
        self.accel_metric: tracking_metrics_warp.MetricBuffers | None = None
        self.prev_sim_body_linvel: torch.Tensor | None = None
        self.prev_ref_body_linvel: torch.Tensor | None = None
        self.prev_sim_root_quat: torch.Tensor | None = None
        self.prev_ref_root_quat: torch.Tensor | None = None

    def matches(
        self,
        *,
        total_envs: int,
        obs_dim: int,
        policy_act_dim: int,
        env_act_dim: int,
    ) -> bool:
        return bool(
            self.total_envs == int(total_envs)
            and self.obs_dim == int(obs_dim)
            and self.policy_act_dim == int(policy_act_dim)
            and self.env_act_dim == int(env_act_dim)
        )

    def reset_episode_accumulators(self) -> None:
        self.active_mask.fill_(True)
        self.valid_mask.fill_(True)
        self.success_mask.fill_(True)
        self.reward_sum.zero_()
        self.ape_sum.zero_()
        self.global_mpjpe_sum.zero_()
        self.ee_sum.zero_()
        self.vel_sum.zero_()
        self.accel_sum.zero_()
        self.steps_per_env.zero_()
        self.accel_steps_per_env.zero_()
        self.last_ape.zero_()
        self.last_global_mpjpe.zero_()
        self.last_ee.zero_()
        self.last_vel.zero_()
        self.last_accel.zero_()
        self.eval_condition.fill_(1)
        self.eval_step_index.zero_()
        self.eval_completed_steps.zero_()
        self.eval_first_invalid_step.zero_()
        self.eval_first_failure_step.zero_()
        self.eval_active_count.fill_(self.total_envs)
        self.eval_terminated_any.zero_()
        self.eval_truncated_any.zero_()

    def ensure_temporal_metric_workspace(
        self,
        sample_body_linvel: torch.Tensor,
    ) -> None:
        body_count = int(sample_body_linvel.shape[1])
        if self._temporal_body_count == body_count and self.vel_metric is not None:
            return
        self._temporal_body_count = body_count
        self.vel_metric = tracking_metrics_warp.create_metric_buffers(
            env_count=self.total_envs,
            body_count=body_count,
            device=sample_body_linvel.device,
            dtype=sample_body_linvel.dtype,
        )
        self.accel_metric = tracking_metrics_warp.create_metric_buffers(
            env_count=self.total_envs,
            body_count=body_count,
            device=sample_body_linvel.device,
            dtype=sample_body_linvel.dtype,
        )
        self.prev_sim_body_linvel = torch.empty_like(sample_body_linvel)
        self.prev_ref_body_linvel = torch.empty_like(sample_body_linvel)
        self.prev_sim_root_quat = torch.empty(
            self.total_envs,
            4,
            device=sample_body_linvel.device,
            dtype=sample_body_linvel.dtype,
        )
        self.prev_ref_root_quat = torch.empty_like(self.prev_sim_root_quat)


@dataclass
class TrackingBenchmarkJob:
    runner: Any
    workspace: _TrackingBenchmarkWorkspace
    snapshot_metadata: dict[str, int]
    start_event: Any
    complete_event: Any

    def done(self) -> bool:
        return bool(self.complete_event.query())

    def wait(self) -> None:
        self.complete_event.synchronize()

    @property
    def elapsed_ns(self) -> int:
        if not self.done():
            raise RuntimeError("Evaluation latency requested before job completion.")
        return int(
            round(float(self.start_event.elapsed_time(self.complete_event)) * 1e6)
        )

    def result(self) -> dict[str, float | int | bool]:
        if not self.done():
            raise RuntimeError("Evaluation result requested before job completion.")
        return self.runner._summarize_captured_workspace(self.workspace)


def _summarize_tracking_benchmark_workspace(
    workspace: _TrackingBenchmarkWorkspace,
    *,
    completed_steps: int,
    first_invalid_step: int,
    first_failure_step: int,
    terminated_any: bool,
    truncated_any: bool,
    tracking_metrics_available: bool,
    temporal_metrics_seen: bool,
    accel_metrics_seen: bool,
    episode_spec: EvalEpisodeSpec | Any | None = None,
) -> dict[str, float | int | bool]:
    total_envs = workspace.total_envs
    valid_mask = workspace.valid_mask
    success_mask = workspace.success_mask
    reward_sum = workspace.reward_sum
    steps_per_env = workspace.steps_per_env
    ape_sum = workspace.ape_sum
    global_mpjpe_sum = workspace.global_mpjpe_sum
    ee_sum = workspace.ee_sum
    vel_sum = workspace.vel_sum
    accel_sum = workspace.accel_sum
    accel_steps_per_env = workspace.accel_steps_per_env
    last_ape = workspace.last_ape
    last_global_mpjpe = workspace.last_global_mpjpe
    last_ee = workspace.last_ee
    last_vel = workspace.last_vel
    last_accel = workspace.last_accel

    if episode_spec is None:
        episode_spec = EvalEpisodeSpec(
            validity="full_horizon",
            success="full_horizon",
        )
    success_mode = episode_spec.success
    success_defined = success_mode is not None
    valid_count = int(valid_mask.sum().item())
    success_count = int(success_mask.sum().item()) if success_defined else None
    outcome = EvalOutcomeSummary(
        total_envs=total_envs,
        valid_count=valid_count,
        first_invalid_step=first_invalid_step,
        success_count=success_count,
        first_failure_step=first_failure_step,
    )
    reward_mean = float(reward_sum.mean().item())
    episode_length_mean = float(steps_per_env.to(dtype=torch.float32).mean().item())
    valid = valid_count > 0
    if valid:
        reward_mean_valid = float(reward_sum[valid_mask].mean().item())
        episode_length_mean_valid = float(
            steps_per_env[valid_mask].to(dtype=torch.float32).mean().item()
        )
        if tracking_metrics_available:
            valid_ape = ape_sum[valid_mask] / steps_per_env[valid_mask].clamp(min=1)
            valid_global_mpjpe = global_mpjpe_sum[valid_mask] / steps_per_env[
                valid_mask
            ].clamp(min=1)
            valid_ee = ee_sum[valid_mask] / steps_per_env[valid_mask].clamp(min=1)
            ape_mean = float(valid_ape.mean().item())
            global_mpjpe_mean = float(valid_global_mpjpe.mean().item())
            ee_mean = float(valid_ee.mean().item())
            ape_last = float(last_ape[valid_mask].mean().item())
            global_mpjpe_last = float(last_global_mpjpe[valid_mask].mean().item())
            ee_last = float(last_ee[valid_mask].mean().item())
            if temporal_metrics_seen:
                valid_vel = vel_sum[valid_mask] / steps_per_env[valid_mask].clamp(min=1)
                vel_mean = float(valid_vel.mean().item())
                vel_last = float(last_vel[valid_mask].mean().item())
            else:
                vel_mean = math.nan
                vel_last = math.nan
            valid_accel_mask = valid_mask & (accel_steps_per_env > 0)
            if accel_metrics_seen and bool(valid_accel_mask.any().item()):
                success_accel = accel_sum[valid_accel_mask] / accel_steps_per_env[
                    valid_accel_mask
                ].clamp(min=1)
                accel_mean = float(success_accel.mean().item())
                accel_last = float(last_accel[valid_accel_mask].mean().item())
            else:
                accel_mean = math.nan
                accel_last = math.nan
        else:
            ape_mean = math.nan
            global_mpjpe_mean = math.nan
            ee_mean = math.nan
            ape_last = math.nan
            global_mpjpe_last = math.nan
            ee_last = math.nan
            vel_mean = math.nan
            vel_last = math.nan
            accel_mean = math.nan
            accel_last = math.nan
    else:
        reward_mean_valid = math.nan
        episode_length_mean_valid = math.nan
        ape_mean = math.nan
        global_mpjpe_mean = math.nan
        ee_mean = math.nan
        ape_last = math.nan
        global_mpjpe_last = math.nan
        ee_last = math.nan
        vel_mean = math.nan
        vel_last = math.nan
        accel_mean = math.nan
        accel_last = math.nan

    result: dict[str, float | int | bool] = {
        "steps": int(completed_steps),
        "terminated": bool(terminated_any),
        "truncated": bool(truncated_any),
        "reward_mean": reward_mean,
        "reward_mean_valid": reward_mean_valid,
        "episode_length_mean": episode_length_mean,
        "episode_length_mean_valid": episode_length_mean_valid,
        "valid_only_metrics": True,
        "tracking_metrics_available": bool(tracking_metrics_available),
    }
    result.update(outcome.as_result())
    if success_count is not None:
        result.update(
            reward_mean_success=(
                float(reward_sum[success_mask].mean().item())
                if success_count > 0
                else math.nan
            ),
            episode_length_mean_success=(
                float(steps_per_env[success_mask].to(dtype=torch.float32).mean().item())
                if success_count > 0
                else math.nan
            ),
        )
    if tracking_metrics_available:
        result.update(
            ape_mean=ape_mean,
            ape_last=ape_last,
            global_mpjpe_mean=global_mpjpe_mean,
            global_mpjpe_last=global_mpjpe_last,
            ee_mean=ee_mean,
            ee_last=ee_last,
            vel_mean=vel_mean,
            vel_last=vel_last,
            accel_mean=accel_mean,
            accel_last=accel_last,
        )
    return result


def _summarize_tracking_benchmark_motion_groups(
    workspace: _TrackingBenchmarkWorkspace,
    *,
    groups: Sequence[BenchmarkMotionGroup],
    group_ids: torch.Tensor,
    tracking_metrics_available: bool,
    episode_spec: EvalEpisodeSpec,
    temporal_metrics_seen: bool = False,
    accel_metrics_seen: bool = False,
) -> dict[str, float | int | bool]:
    """Summarize one shared-policy rollout independently for each motion group."""

    if not groups:
        return {}
    if group_ids.numel() != workspace.total_envs:
        raise ValueError(
            "Benchmark motion-group assignment must contain one id per environment."
        )
    result: dict[str, float | int | bool] = {}
    success_defined = episode_spec.success is not None
    for group_index, group in enumerate(groups):
        group_mask = group_ids == group_index
        total_envs = int(group_mask.sum().item())
        if total_envs <= 0:
            raise ValueError(
                f"Benchmark motion group {group.name!r} has no environments."
            )
        valid_mask = workspace.valid_mask & group_mask
        valid_count = int(valid_mask.sum().item())
        prefix = f"{group.name}_"
        result[f"{prefix}total_envs"] = total_envs
        result[f"{prefix}valid_count"] = valid_count
        result[f"{prefix}invalid_count"] = total_envs - valid_count
        result[f"{prefix}valid_rate"] = valid_count / total_envs
        result[f"{prefix}valid"] = valid_count > 0
        result[f"{prefix}success_defined"] = success_defined
        if success_defined:
            success_count = int((workspace.success_mask & group_mask).sum().item())
            result[f"{prefix}success_count"] = success_count
            result[f"{prefix}failure_count"] = total_envs - success_count
            result[f"{prefix}success_rate"] = success_count / total_envs

        result[f"{prefix}reward_mean"] = float(
            workspace.reward_sum[group_mask].mean().item()
        )
        result[f"{prefix}episode_length_mean"] = float(
            workspace.steps_per_env[group_mask].to(dtype=torch.float32).mean().item()
        )
        if valid_count <= 0:
            continue
        steps = workspace.steps_per_env[valid_mask].clamp(min=1)
        result[f"{prefix}reward_mean_valid"] = float(
            workspace.reward_sum[valid_mask].mean().item()
        )
        result[f"{prefix}episode_length_mean_valid"] = float(
            steps.to(dtype=torch.float32).mean().item()
        )
        if tracking_metrics_available:
            result[f"{prefix}ape_mean"] = float(
                (workspace.ape_sum[valid_mask] / steps).mean().item()
            )
            result[f"{prefix}global_mpjpe_mean"] = float(
                (workspace.global_mpjpe_sum[valid_mask] / steps).mean().item()
            )
            result[f"{prefix}ee_mean"] = float(
                (workspace.ee_sum[valid_mask] / steps).mean().item()
            )
            result[f"{prefix}ape_last"] = float(
                workspace.last_ape[valid_mask].mean().item()
            )
            result[f"{prefix}global_mpjpe_last"] = float(
                workspace.last_global_mpjpe[valid_mask].mean().item()
            )
            result[f"{prefix}ee_last"] = float(
                workspace.last_ee[valid_mask].mean().item()
            )
            if temporal_metrics_seen:
                result[f"{prefix}vel_mean"] = float(
                    (
                        workspace.vel_sum[valid_mask]
                        / workspace.steps_per_env[valid_mask].clamp(min=1)
                    )
                    .mean()
                    .item()
                )
                result[f"{prefix}vel_last"] = float(
                    workspace.last_vel[valid_mask].mean().item()
                )
            valid_accel_mask = valid_mask & (workspace.accel_steps_per_env > 0)
            if accel_metrics_seen and bool(valid_accel_mask.any().item()):
                result[f"{prefix}accel_mean"] = float(
                    (
                        workspace.accel_sum[valid_accel_mask]
                        / workspace.accel_steps_per_env[valid_accel_mask].clamp(min=1)
                    )
                    .mean()
                    .item()
                )
                result[f"{prefix}accel_last"] = float(
                    workspace.last_accel[valid_accel_mask].mean().item()
                )
    return result


@wp.kernel(enable_backward=False)
def _tracking_eval_accumulate_metrics_kernel(
    condition: wp.array(dtype=int),
    active_mask: wp.array(dtype=bool),
    ape: wp.array(dtype=float),
    global_mpjpe: wp.array(dtype=float),
    ee: wp.array(dtype=float),
    vel: wp.array(dtype=float),
    accel: wp.array(dtype=float),
    step_index: wp.array(dtype=int),
    ape_sum: wp.array(dtype=float),
    global_mpjpe_sum: wp.array(dtype=float),
    ee_sum: wp.array(dtype=float),
    vel_sum: wp.array(dtype=float),
    accel_sum: wp.array(dtype=float),
    steps_per_env: wp.array(dtype=wp.int32),
    accel_steps_per_env: wp.array(dtype=wp.int32),
    last_ape: wp.array(dtype=float),
    last_global_mpjpe: wp.array(dtype=float),
    last_ee: wp.array(dtype=float),
    last_vel: wp.array(dtype=float),
    last_accel: wp.array(dtype=float),
):
    env_idx = wp.tid()
    if condition[0] == 0:
        return
    if active_mask[env_idx]:
        steps_per_env[env_idx] = steps_per_env[env_idx] + 1
        ape_sum[env_idx] = ape_sum[env_idx] + ape[env_idx]
        global_mpjpe_sum[env_idx] = global_mpjpe_sum[env_idx] + global_mpjpe[env_idx]
        ee_sum[env_idx] = ee_sum[env_idx] + ee[env_idx]
        vel_sum[env_idx] = vel_sum[env_idx] + vel[env_idx]
        last_ape[env_idx] = ape[env_idx]
        last_global_mpjpe[env_idx] = global_mpjpe[env_idx]
        last_ee[env_idx] = ee[env_idx]
        last_vel[env_idx] = vel[env_idx]
        if step_index[0] > 0:
            accel_sum[env_idx] = accel_sum[env_idx] + accel[env_idx]
            accel_steps_per_env[env_idx] = accel_steps_per_env[env_idx] + 1
            last_accel[env_idx] = accel[env_idx]


@wp.kernel(enable_backward=False)
def _tracking_eval_reset_active_count_kernel(
    condition: wp.array(dtype=int),
    active_count: wp.array(dtype=int),
):
    if condition[0] == 0:
        return
    active_count[0] = 0


@wp.kernel(enable_backward=False)
def _rollout_eval_increment_steps_kernel(
    condition: wp.array(dtype=int),
    active_mask: wp.array(dtype=bool),
    steps_per_env: wp.array(dtype=wp.int32),
):
    env_idx = wp.tid()
    if condition[0] != 0 and active_mask[env_idx]:
        steps_per_env[env_idx] = steps_per_env[env_idx] + 1


@wp.kernel(enable_backward=False)
def _tracking_eval_update_done_kernel(
    condition: wp.array(dtype=int),
    active_mask: wp.array(dtype=bool),
    valid_mask: wp.array(dtype=bool),
    success_mask: wp.array(dtype=bool),
    reward: wp.array(dtype=float),
    terminated: wp.array(dtype=bool),
    truncated: wp.array(dtype=bool),
    motion_end_time_outs: wp.array(dtype=wp.bool),
    terminal_success_mask: wp.array(dtype=wp.bool),
    reward_sum: wp.array(dtype=float),
    step_index: wp.array(dtype=int),
    completed_steps: wp.array(dtype=int),
    first_invalid_step: wp.array(dtype=int),
    first_failure_step: wp.array(dtype=int),
    active_count: wp.array(dtype=int),
    terminated_any: wp.array(dtype=int),
    truncated_any: wp.array(dtype=int),
    play_steps: int,
    validity_full_horizon: int,
    success_full_horizon: int,
    has_motion_end_time_outs: int,
    has_terminal_success_mask: int,
    terminal_success: int,
):
    env_idx = wp.tid()
    if condition[0] == 0:
        return
    was_active = active_mask[env_idx]
    if was_active:
        reward_sum[env_idx] = reward_sum[env_idx] + reward[env_idx]

    is_terminated = terminated[env_idx]
    is_truncated = truncated[env_idx]
    if is_terminated:
        terminated_any[0] = 1
    if is_truncated:
        truncated_any[0] = 1

    step_count = step_index[0] + 1
    done = is_terminated or is_truncated
    invalid = False
    unsuccessful = False
    if was_active and done:
        natural_completion = False
        if has_motion_end_time_outs != 0:
            natural_completion = (
                is_truncated and motion_end_time_outs[env_idx] and not is_terminated
            )
        if validity_full_horizon != 0 and step_count < play_steps:
            if not natural_completion:
                valid_mask[env_idx] = False
                invalid = True
        if terminal_success != 0:
            is_success = is_terminated
            if has_terminal_success_mask != 0:
                is_success = terminal_success_mask[env_idx]
            if not is_success:
                success_mask[env_idx] = False
                unsuccessful = True
        elif success_full_horizon != 0 and step_count < play_steps:
            if not natural_completion:
                success_mask[env_idx] = False
                unsuccessful = True
        if (
            validity_full_horizon != 0
            or success_full_horizon != 0
            or terminal_success != 0
            or step_count >= play_steps
        ):
            active_mask[env_idx] = False
    elif was_active and step_count >= play_steps:
        active_mask[env_idx] = False
        if terminal_success != 0:
            success_mask[env_idx] = False
            unsuccessful = True

    if invalid and first_invalid_step[0] == 0:
        first_invalid_step[0] = step_count
    if unsuccessful and first_failure_step[0] == 0:
        first_failure_step[0] = step_count

    if active_mask[env_idx]:
        wp.atomic_add(active_count, 0, 1)


@wp.kernel(enable_backward=False)
def _tracking_eval_finish_step_kernel(
    condition: wp.array(dtype=int),
    step_index: wp.array(dtype=int),
    completed_steps: wp.array(dtype=int),
    active_count: wp.array(dtype=int),
    play_steps: int,
):
    if condition[0] == 0:
        return
    next_step = step_index[0] + 1
    step_index[0] = next_step
    completed_steps[0] = next_step
    if next_step < play_steps and active_count[0] > 0:
        condition[0] = 1
    else:
        condition[0] = 0


@wp.func
def _tracking_eval_policy_activation(value: float, kind: int, param: float) -> float:
    if kind == 1:
        return wp.max(value, 0.0)
    if kind == 2:
        return value / (1.0 + wp.exp(0.0 - value))
    if kind == 3:
        return wp.tanh(value)
    if kind == 4:
        if value > 0.0:
            return value
        return param * (wp.exp(value) - 1.0)
    if kind == 5:
        if value > 0.0:
            return value
        return param * value
    return value


@wp.kernel(enable_backward=False)
def _tracking_eval_policy_normalize_kernel(
    obs: wp.array2d(dtype=wp.float32),
    mean: wp.array(dtype=wp.float32),
    denom: wp.array(dtype=wp.float32),
    scaler_enabled: int,
    clip_threshold: float,
    out: wp.array2d(dtype=wp.float32),
):
    row_idx, col_idx = wp.tid()
    value = obs[row_idx, col_idx]
    if scaler_enabled != 0:
        value = (value - mean[col_idx]) / denom[col_idx]
        value = wp.clamp(value, -clip_threshold, clip_threshold)
    out[row_idx, col_idx] = value


@wp.kernel(enable_backward=False)
def _tracking_eval_policy_linear_kernel(
    src: wp.array2d(dtype=wp.float32),
    weight: wp.array2d(dtype=wp.float32),
    bias: wp.array(dtype=wp.float32),
    activation_kind: int,
    activation_param: float,
    out: wp.array2d(dtype=wp.float32),
):
    row_idx, out_idx = wp.tid()
    input_width = int(src.shape[1])
    value = bias[out_idx]
    in_idx = int(0)
    while in_idx < input_width:
        value = value + src[row_idx, in_idx] * weight[out_idx, in_idx]
        in_idx = in_idx + 1
    out[row_idx, out_idx] = _tracking_eval_policy_activation(
        value,
        activation_kind,
        activation_param,
    )


@wp.kernel(enable_backward=False)
def _tracking_eval_policy_layer_norm_kernel(
    values: wp.array2d(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    bias: wp.array(dtype=wp.float32),
    epsilon: float,
    activation_kind: int,
    activation_param: float,
):
    row_idx = wp.tid()
    width = int(values.shape[1])
    mean = float(0.0)
    col_idx = int(0)
    while col_idx < width:
        mean = mean + values[row_idx, col_idx]
        col_idx = col_idx + 1
    mean = mean / float(width)
    variance = float(0.0)
    col_idx = int(0)
    while col_idx < width:
        centered = values[row_idx, col_idx] - mean
        variance = variance + centered * centered
        col_idx = col_idx + 1
    inverse_std = 1.0 / wp.sqrt(variance / float(width) + epsilon)
    col_idx = int(0)
    while col_idx < width:
        normalized = (values[row_idx, col_idx] - mean) * inverse_std
        normalized = normalized * weight[col_idx] + bias[col_idx]
        values[row_idx, col_idx] = _tracking_eval_policy_activation(
            normalized,
            activation_kind,
            activation_param,
        )
        col_idx = col_idx + 1


@wp.kernel(enable_backward=False)
def _tracking_eval_policy_residual_add_kernel(
    values: wp.array2d(dtype=wp.float32),
    residual: wp.array2d(dtype=wp.float32),
):
    row_idx, col_idx = wp.tid()
    values[row_idx, col_idx] = values[row_idx, col_idx] + residual[row_idx, col_idx]


@wp.kernel(enable_backward=False)
def _tracking_eval_policy_gather_columns_kernel(
    src: wp.array2d(dtype=wp.float32),
    columns: wp.array(dtype=wp.int64),
    out_offset: int,
    out: wp.array2d(dtype=wp.float32),
):
    row_idx, col_idx = wp.tid()
    out[row_idx, out_offset + col_idx] = src[row_idx, int(columns[col_idx])]


@wp.kernel(enable_backward=False)
def _tracking_eval_policy_copy_columns_kernel(
    src: wp.array2d(dtype=wp.float32),
    out_offset: int,
    out: wp.array2d(dtype=wp.float32),
):
    row_idx, col_idx = wp.tid()
    out[row_idx, out_offset + col_idx] = src[row_idx, col_idx]


@wp.kernel(enable_backward=False)
def _tracking_eval_policy_rms_norm_kernel(
    values: wp.array2d(dtype=wp.float32),
    weight: wp.array(dtype=wp.float32),
    epsilon: float,
):
    row_idx = wp.tid()
    width = int(values.shape[1])
    square_sum = float(0.0)
    col_idx = int(0)
    while col_idx < width:
        square_sum = square_sum + values[row_idx, col_idx] * values[row_idx, col_idx]
        col_idx = col_idx + 1
    inverse_rms = 1.0 / wp.sqrt(square_sum / float(width) + epsilon)
    col_idx = int(0)
    while col_idx < width:
        values[row_idx, col_idx] = (
            values[row_idx, col_idx] * inverse_rms * weight[col_idx]
        )
        col_idx = col_idx + 1


@wp.kernel(enable_backward=False)
def _tracking_eval_policy_output_kernel(
    src: wp.array2d(dtype=wp.float32),
    weight: wp.array2d(dtype=wp.float32),
    bias: wp.array(dtype=wp.float32),
    action_scale: wp.array(dtype=wp.float32),
    scale_enabled: int,
    squash_actions: int,
    clamp_mean: int,
    clamp_mean_min: float,
    clamp_mean_max: float,
    out: wp.array2d(dtype=wp.float32),
):
    row_idx, out_idx = wp.tid()
    input_width = int(src.shape[1])
    value = bias[out_idx]
    in_idx = int(0)
    while in_idx < input_width:
        value = value + src[row_idx, in_idx] * weight[out_idx, in_idx]
        in_idx = in_idx + 1
    if scale_enabled != 0:
        value = value * action_scale[out_idx]
    if squash_actions != 0:
        value = wp.tanh(value)
    elif clamp_mean != 0:
        value = wp.clamp(value, clamp_mean_min, clamp_mean_max)
    out[row_idx, out_idx] = value


@torch.no_grad()
def _collect_tracking_metric_tensors(
    env_raw: Any,
    *,
    frame: str = "root",
    include_root_in_ape: bool = False,
) -> dict[str, torch.Tensor] | None:
    get_metrics = getattr(env_raw, "get_reference_tracking_metrics", None)
    if not callable(get_metrics):
        return None
    metrics = get_metrics(frame=frame, include_root_in_ape=include_root_in_ape)
    if not isinstance(metrics, dict):
        return None
    ape = metrics.get("ape")
    global_mpjpe = metrics.get("global_mpjpe")
    ee_error = metrics.get("ee_error")
    if (
        not isinstance(ape, torch.Tensor)
        or not isinstance(global_mpjpe, torch.Tensor)
        or not isinstance(ee_error, torch.Tensor)
    ):
        return None
    result: dict[str, torch.Tensor] = {
        "ape": ape.reshape(-1),
        "global_mpjpe": global_mpjpe.reshape(-1),
        "ee_error": ee_error.reshape(-1),
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


def _build_tracking_benchmark_env(
    cfg: Any,
    *,
    log_dir: Path,
    num_envs: int,
    play_steps: int,
    deterministic_reset: bool,
    enable_early_termination: bool | None,
    early_termination_mode: str | None = None,
    phc_termination_distance: float | None = None,
    use_cuda_graph: bool = True,
) -> Any:
    eval_env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env, resolve=True))
    env_overrides = _benchmark_env_overrides(cfg)
    if env_overrides is not None:
        eval_env_cfg = OmegaConf.merge(eval_env_cfg, env_overrides)
    with open_dict(eval_env_cfg):
        eval_env_cfg.scene.num_envs = int(num_envs)
        eval_env_cfg.log_dir = str((log_dir / "benchmark_eval").resolve())
        eval_env_cfg.use_cuda_graph = bool(use_cuda_graph)
        reset = eval_env_cfg.get("reset", None)
        termination = eval_env_cfg.get("termination", None)
        if reset is not None:
            reset.deterministic = bool(deterministic_reset)
        else:
            eval_env_cfg.deterministic_reset = bool(deterministic_reset)
        if termination is not None:
            early = termination.get("early", None)
            if early is not None:
                if enable_early_termination is not None:
                    early.enabled = bool(enable_early_termination)
                if early_termination_mode is not None:
                    early.type = str(early_termination_mode)
                if phc_termination_distance is not None:
                    early.distance = float(phc_termination_distance)
        else:
            if enable_early_termination is not None:
                eval_env_cfg.enable_early_termination = bool(enable_early_termination)
            if early_termination_mode is not None:
                eval_env_cfg.early_termination_mode = str(early_termination_mode)
            if phc_termination_distance is not None:
                eval_env_cfg.phc_termination_distance = float(phc_termination_distance)

    eval_env = gym.make(cfg.task, cfg=eval_env_cfg, render_mode=None)
    eval_env_raw = eval_env.unwrapped
    max_episode_length = getattr(eval_env_raw, "max_episode_length", None)
    if max_episode_length is not None and play_steps >= int(max_episode_length):
        target_episode_steps = play_steps + 2
        target_episode_length_s = target_episode_steps * float(eval_env_raw.step_dt)
        if hasattr(eval_env_raw, "cfg") and hasattr(
            eval_env_raw.cfg, "episode_length_s"
        ):
            prev_episode_length_s = float(eval_env_raw.cfg.episode_length_s)
            if target_episode_length_s > prev_episode_length_s:
                eval_env_raw.cfg.episode_length_s = target_episode_length_s
    return eval_env


class _CompletePolicySnapshot:
    """Frozen policy and normalization state for exact complete-policy capture.

    The full ActorCritic forward preserves authored precision, including BF16
    autocast and intermediate rounding. Separate parameters also preserve the
    policy version while overlap evaluation runs beside the next update.
    """

    def __init__(
        self, *, agent: Any, batch_size: int, obs_dim: int, device: torch.device
    ) -> None:
        self.network = copy.deepcopy(agent.network)
        self.scaler = copy.deepcopy(agent.obs_scaler)
        self.scaled_obs = torch.empty(
            batch_size, obs_dim, device=device, dtype=torch.float32
        )

    @torch.no_grad()
    def refresh(self, agent: Any) -> None:
        # load_state_dict copies into existing tensors: captured addresses stay
        # fixed while each benchmark receives a fresh, internally frozen policy.
        self.network.load_state_dict(agent.network.state_dict(), strict=True)
        self.scaler.load_state_dict(agent.obs_scaler.state_dict(), strict=True)

    @torch.no_grad()
    def policy_mean_into(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> None:
        with network_matmul_precision():
            self._policy_mean_into_unscoped(observations, actions)

    def _policy_mean_into_unscoped(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> None:
        self.scaler(observations, train=False, out=self.scaled_obs)
        self.network.policy_mean_into(self.scaled_obs, actions)


class _BenchmarkPolicyRunner:
    class _PreallocatedMLPPolicy:
        def __init__(
            self,
            *,
            agent: Any,
            device: torch.device,
            obs_dim: int,
            batch_size: int,
            act_dim: int,
        ) -> None:
            self._agent = agent
            self._device = device
            self._obs_dim = int(obs_dim)
            self._batch_size = int(batch_size)
            self._act_dim = int(act_dim)
            self._network = getattr(agent, "network", None)
            self._obs_scaler = getattr(agent, "obs_scaler", None)
            if self._network is None:
                raise ValueError("agent has no network")
            self._obs_main_dim = (
                int(agent.actor_obs_dim)
                if bool(getattr(agent, "asymmetric_observations", False))
                else self._obs_dim
            )
            if self._obs_main_dim <= 0:
                raise ValueError("policy observation main dimension must be positive")
            self._squash_actions = bool(getattr(self._network, "squash_actions", False))
            self._clamp_mean = bool(getattr(self._network, "clamp_mean", False))
            self._clamp_mean_min = float(
                getattr(self._network, "clamp_mean_min", -math.inf)
            )
            self._clamp_mean_max = float(
                getattr(self._network, "clamp_mean_max", math.inf)
            )
            trunk = getattr(self._network, "actor_trunk", None)
            if trunk is None:
                trunk = getattr(self._network, "trunk", None)
            policy_head = getattr(self._network, "policy_head", None)
            if not isinstance(trunk, torch.nn.Sequential):
                raise ValueError("policy trunk must be an nn.Sequential")
            if _is_affine_step(policy_head):
                output_head = policy_head
                self._output_dim = self._act_dim
            else:
                raise ValueError("policy does not expose a supported output head")
            self._policy_head_source = _AffineParameterSource(output_head)
            self._policy_head_weight = self._policy_head_source.empty_weight()
            self._policy_head_bias = self._policy_head_source.empty_bias()
            self._trunk_steps: list[torch.nn.Module] = []
            self._linear_outputs: list[torch.Tensor] = []
            self._linear_sources: list[_AffineParameterSource] = []
            self._linear_weights: list[torch.Tensor] = []
            self._linear_biases: list[torch.Tensor] = []
            self._norm_sources: list[tuple[torch.Tensor, torch.Tensor]] = []
            self._norm_weights: list[torch.Tensor] = []
            self._norm_biases: list[torch.Tensor] = []
            self._warp_linear_steps: list[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    int,
                    int,
                    float,
                    torch.Tensor | None,
                    torch.Tensor | None,
                    float,
                ]
            ] = []
            current_width = self._obs_main_dim
            for module in trunk:
                if _is_affine_step(module):
                    if int(module.in_features) != int(current_width):
                        raise ValueError("policy trunk dimensions are inconsistent")
                    out = torch.empty(
                        self._batch_size,
                        int(module.out_features),
                        device=device,
                        dtype=torch.float32,
                    )
                    self._linear_outputs.append(out)
                    source = _AffineParameterSource(module)
                    self._linear_sources.append(source)
                    self._linear_weights.append(source.empty_weight())
                    self._linear_biases.append(source.empty_bias())
                    current_width = int(module.out_features)
                elif not isinstance(
                    module, torch.nn.LayerNorm
                ) and not self._is_supported_activation(module):
                    raise ValueError(
                        f"unsupported policy trunk module: {type(module).__name__}"
                    )
                self._trunk_steps.append(module)
            if int(output_head.in_features) != int(current_width):
                raise ValueError("policy head input dimension is inconsistent")
            if int(output_head.out_features) != self._output_dim:
                raise ValueError("policy head output dimension is inconsistent")
            self._build_warp_linear_steps()

            self._scaled_obs = torch.empty(
                self._batch_size,
                self._obs_main_dim,
                device=device,
                dtype=torch.float32,
            )
            self._scale_mean = torch.empty(
                self._obs_main_dim,
                device=device,
                dtype=torch.float32,
            )
            self._scale_denom = torch.empty_like(self._scale_mean)
            self._scaler_enabled = bool(
                self._obs_scaler is not None
                and bool(getattr(self._obs_scaler, "enabled", True))
            )
            self._scale_epsilon = (
                float(getattr(self._obs_scaler, "epsilon", 0.0))
                if self._obs_scaler is not None
                else 0.0
            )
            self._scale_clip = (
                float(getattr(self._obs_scaler, "clip_threshold", math.inf))
                if self._obs_scaler is not None
                else math.inf
            )
            action_mean_scale = getattr(self._network, "action_mean_scale", None)
            self._action_mean_scale_source = (
                action_mean_scale.detach()
                if isinstance(action_mean_scale, torch.Tensor)
                and tuple(action_mean_scale.shape) == (self._act_dim,)
                else None
            )
            self._action_mean_scale = (
                torch.empty_like(self._action_mean_scale_source)
                if self._action_mean_scale_source is not None
                else None
            )
            self._unit_action_scale = torch.ones(
                self._output_dim,
                device=device,
                dtype=torch.float32,
            )
            self._warp_action_scale = (
                self._action_mean_scale
                if self._action_mean_scale is not None
                else self._unit_action_scale
            )
            self._use_warp_policy = device.type == "cuda"
            self.refresh_state()

        @staticmethod
        def _is_supported_activation(module: torch.nn.Module) -> bool:
            return isinstance(
                module,
                (
                    torch.nn.ELU,
                    torch.nn.Identity,
                    torch.nn.LeakyReLU,
                    torch.nn.ReLU,
                    torch.nn.SiLU,
                    torch.nn.Tanh,
                ),
            )

        @staticmethod
        def _activation_spec(module: torch.nn.Module) -> tuple[int, float]:
            if isinstance(module, torch.nn.Identity):
                return 0, 0.0
            if isinstance(module, torch.nn.ReLU):
                return 1, 0.0
            if isinstance(module, torch.nn.SiLU):
                return 2, 0.0
            if isinstance(module, torch.nn.Tanh):
                return 3, 0.0
            if isinstance(module, torch.nn.ELU):
                return 4, float(module.alpha)
            if isinstance(module, torch.nn.LeakyReLU):
                return 5, float(module.negative_slope)
            raise RuntimeError(f"Unsupported activation: {type(module).__name__}")

        def _build_warp_linear_steps(self) -> None:
            linear_output_idx = 0
            module_idx = 0
            while module_idx < len(self._trunk_steps):
                module = self._trunk_steps[module_idx]
                if not _is_affine_step(module):
                    raise ValueError(
                        "Warp policy path requires each activation to follow a Linear"
                    )
                out = self._linear_outputs[linear_output_idx]
                weight = self._linear_weights[linear_output_idx]
                bias = self._linear_biases[linear_output_idx]
                linear_output_idx += 1
                norm_weight = None
                norm_bias = None
                norm_epsilon = 0.0
                activation_kind = 0
                activation_param = 0.0
                next_idx = module_idx + 1
                if next_idx < len(self._trunk_steps) and isinstance(
                    self._trunk_steps[next_idx], torch.nn.LayerNorm
                ):
                    norm = self._trunk_steps[next_idx]
                    if tuple(norm.normalized_shape) != (int(module.out_features),):
                        raise ValueError("unsupported policy LayerNorm shape")
                    if norm.weight is None or norm.bias is None:
                        raise ValueError("policy LayerNorm must use affine parameters")
                    norm_weight_source = norm.weight.detach()
                    norm_bias_source = norm.bias.detach()
                    self._norm_sources.append((norm_weight_source, norm_bias_source))
                    norm_weight = torch.empty_like(norm_weight_source)
                    norm_bias = torch.empty_like(norm_bias_source)
                    self._norm_weights.append(norm_weight)
                    self._norm_biases.append(norm_bias)
                    norm_epsilon = float(norm.eps)
                    next_idx += 1
                if next_idx < len(self._trunk_steps) and not _is_affine_step(
                    self._trunk_steps[next_idx]
                ):
                    activation_kind, activation_param = self._activation_spec(
                        self._trunk_steps[next_idx]
                    )
                    module_idx = next_idx + 1
                else:
                    module_idx = next_idx
                self._warp_linear_steps.append(
                    (
                        weight,
                        bias,
                        out,
                        int(module.out_features),
                        activation_kind,
                        activation_param,
                        norm_weight,
                        norm_bias,
                        norm_epsilon,
                    )
                )

        @staticmethod
        def _apply_activation(module: torch.nn.Module, tensor: torch.Tensor) -> None:
            if isinstance(module, torch.nn.Identity):
                return
            if isinstance(module, torch.nn.ReLU):
                tensor.relu_()
                return
            if isinstance(module, torch.nn.SiLU):
                torch.nn.functional.silu(tensor, inplace=True)
                return
            if isinstance(module, torch.nn.Tanh):
                tensor.tanh_()
                return
            if isinstance(module, torch.nn.ELU):
                torch.nn.functional.elu(
                    tensor,
                    alpha=float(module.alpha),
                    inplace=True,
                )
                return
            if isinstance(module, torch.nn.LeakyReLU):
                torch.nn.functional.leaky_relu(
                    tensor,
                    negative_slope=float(module.negative_slope),
                    inplace=True,
                )
                return
            raise RuntimeError(f"Unsupported activation: {type(module).__name__}")

        @torch.no_grad()
        def _forward_into_warp(
            self,
            obs_policy: torch.Tensor,
            out_values: torch.Tensor,
        ) -> None:
            launch_device = wp.device_from_torch(obs_policy.device)
            wp.launch(
                _tracking_eval_policy_normalize_kernel,
                dim=(self._batch_size, self._obs_main_dim),
                inputs=[
                    obs_policy,
                    self._scale_mean,
                    self._scale_denom,
                    int(self._scaler_enabled),
                    float(self._scale_clip),
                    self._scaled_obs,
                ],
                device=launch_device,
            )
            x = self._scaled_obs
            for (
                weight,
                bias,
                out,
                out_features,
                activation_kind,
                activation_param,
                norm_weight,
                norm_bias,
                norm_epsilon,
            ) in self._warp_linear_steps:
                wp.launch(
                    _tracking_eval_policy_linear_kernel,
                    dim=(self._batch_size, int(out_features)),
                    inputs=[
                        x,
                        weight,
                        bias,
                        int(activation_kind if norm_weight is None else 0),
                        float(activation_param),
                        out,
                    ],
                    device=launch_device,
                )
                if norm_weight is not None:
                    assert norm_bias is not None
                    wp.launch(
                        _tracking_eval_policy_layer_norm_kernel,
                        dim=self._batch_size,
                        inputs=[
                            out,
                            norm_weight,
                            norm_bias,
                            float(norm_epsilon),
                            int(activation_kind),
                            float(activation_param),
                        ],
                        device=launch_device,
                    )
                x = out
            wp.launch(
                _tracking_eval_policy_output_kernel,
                dim=(self._batch_size, self._output_dim),
                inputs=[
                    x,
                    self._policy_head_weight,
                    self._policy_head_bias,
                    self._warp_action_scale,
                    int(self._action_mean_scale is not None),
                    int(self._squash_actions),
                    int(self._clamp_mean),
                    float(self._clamp_mean_min),
                    float(self._clamp_mean_max),
                    out_values,
                ],
                device=launch_device,
            )

        def _validate_io(
            self,
            obs_policy: torch.Tensor,
            out_values: torch.Tensor,
        ) -> None:
            if tuple(obs_policy.shape) != (self._batch_size, self._obs_dim):
                raise RuntimeError(
                    "preallocated policy path received unexpected obs shape: "
                    f"expected {(self._batch_size, self._obs_dim)}, "
                    f"got {tuple(obs_policy.shape)}"
                )
            if tuple(out_values.shape) != (self._batch_size, self._output_dim):
                raise RuntimeError(
                    "preallocated policy path received unexpected output shape: "
                    f"expected {(self._batch_size, self._output_dim)}, "
                    f"got {tuple(out_values.shape)}"
                )

        def _forward_into_torch(
            self,
            obs_policy: torch.Tensor,
            out_values: torch.Tensor,
        ) -> None:
            self._scaled_obs.copy_(obs_policy[..., : self._obs_main_dim])
            if self._scaler_enabled:
                self._scaled_obs.sub_(self._scale_mean)
                self._scaled_obs.div_(self._scale_denom)
                self._scaled_obs.clamp_(-self._scale_clip, self._scale_clip)

            x = self._scaled_obs
            linear_output_idx = 0
            norm_idx = 0
            for module in self._trunk_steps:
                if _is_affine_step(module):
                    out = self._linear_outputs[linear_output_idx]
                    weight = self._linear_weights[linear_output_idx]
                    bias = self._linear_biases[linear_output_idx]
                    linear_output_idx += 1
                    torch.addmm(bias, x, weight.t(), out=out)
                    x = out
                    continue
                if isinstance(module, torch.nn.LayerNorm):
                    weight = self._norm_weights[norm_idx]
                    bias = self._norm_biases[norm_idx]
                    norm_idx += 1
                    x.copy_(
                        torch.nn.functional.layer_norm(
                            x,
                            module.normalized_shape,
                            weight,
                            bias,
                            module.eps,
                        )
                    )
                    continue
                self._apply_activation(module, x)

            torch.addmm(
                self._policy_head_bias,
                x,
                self._policy_head_weight.t(),
                out=out_values,
            )
            if self._action_mean_scale is not None:
                out_values.mul_(self._action_mean_scale)
            if self._squash_actions:
                out_values.tanh_()
            elif self._clamp_mean:
                out_values.clamp_(self._clamp_mean_min, self._clamp_mean_max)
            return

        @torch.no_grad()
        def refresh_state(self) -> None:
            for source, weight, bias in zip(
                self._linear_sources,
                self._linear_weights,
                self._linear_biases,
                strict=True,
            ):
                source.write(weight, bias)
            for (weight_source, bias_source), weight, bias in zip(
                self._norm_sources,
                self._norm_weights,
                self._norm_biases,
                strict=False,
            ):
                weight.copy_(weight_source)
                bias.copy_(bias_source)
            self._policy_head_source.write(
                self._policy_head_weight, self._policy_head_bias
            )
            if self._action_mean_scale is not None:
                assert self._action_mean_scale_source is not None
                self._action_mean_scale.copy_(self._action_mean_scale_source)
            if not self._scaler_enabled:
                return
            if self._obs_scaler is None:
                return
            running_mean = getattr(self._obs_scaler, "running_mean", None)
            running_variance = getattr(self._obs_scaler, "running_variance", None)
            if not isinstance(running_mean, torch.Tensor) or not isinstance(
                running_variance,
                torch.Tensor,
            ):
                raise RuntimeError("obs scaler does not expose running statistics")
            self._scale_mean.copy_(running_mean.reshape(-1)[: self._obs_main_dim])
            self._scale_denom.copy_(running_variance.reshape(-1)[: self._obs_main_dim])
            self._scale_denom.sqrt_()
            self._scale_denom.add_(self._scale_epsilon)

        @torch.no_grad()
        def policy_mean_into(
            self,
            obs_policy: torch.Tensor,
            out_actions: torch.Tensor,
        ) -> None:
            with network_matmul_precision():
                self._policy_mean_into_unscoped(obs_policy, out_actions)

        def _policy_mean_into_unscoped(
            self,
            obs_policy: torch.Tensor,
            out_actions: torch.Tensor,
        ) -> None:
            if not self.supports_policy_mean:
                raise RuntimeError(
                    "native policy snapshot output width does not match the policy action width"
                )
            self._validate_io(obs_policy, out_actions)
            if self._use_warp_policy:
                self._forward_into_warp(
                    obs_policy,
                    out_actions,
                )
                return
            self._forward_into_torch(
                obs_policy,
                out_actions,
            )

        @property
        def supports_policy_mean(self) -> bool:
            return bool(self._output_dim == self._act_dim)

    def __init__(
        self,
        *,
        agent: Any,
        device: torch.device,
        obs_dim: int,
        batch_size: int,
        compile_enabled: bool,
        compile_mode: str,
        use_cuda_graph: bool,
        torch_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self._agent = agent
        self._device = device
        self._obs_dim = int(obs_dim)
        self._batch_size = int(batch_size)
        act_dim = getattr(agent, "act_dim", None)
        if act_dim is None:
            network = getattr(agent, "network", None)
            act_dim = getattr(network, "act_dim", None)
        if act_dim is None:
            raise AttributeError(
                "Benchmark policy runner requires agent.act_dim to execute "
                "policy_mean_into()."
            )
        self._act_dim = int(act_dim)
        self._compile_enabled = bool(compile_enabled and hasattr(torch, "compile"))
        self._compile_mode = str(compile_mode)
        self._use_cuda_graph = bool(use_cuda_graph and device.type == "cuda")
        self._torch_stream = torch_stream
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_obs: torch.Tensor | None = None
        self._graph_actions: torch.Tensor | None = None

        self._preallocated_policy = self._build_preallocated_policy_runner()
        self._complete_policy_snapshot = (
            _CompletePolicySnapshot(
                agent=agent,
                batch_size=self._batch_size,
                obs_dim=self._obs_dim,
                device=self._device,
            )
            if _owned_policy_mean_matches(agent, agent.network)
            else None
        )
        self._policy_mean_into_fn = self._build_policy_mean_into_fn()
        self._warmup()
        if self._use_cuda_graph:
            self._capture_graph()

    @property
    def uses_cuda_graph(self) -> bool:
        return self._graph is not None

    @property
    def can_capture_complete_policy_mean(self) -> bool:
        """Allow an owned step capture without a standalone policy graph.

        The conditional benchmark captures the complete function directly;
        capture itself validates unsupported operations and reports failure.
        Only a certified frozen snapshot can outlive a training update.
        """
        return (
            self._device.type == "cuda" and self._complete_policy_snapshot is not None
        )

    @property
    def uses_preallocated_policy_path(self) -> bool:
        return self._preallocated_policy is not None

    @property
    def supports_preallocated_policy_mean(self) -> bool:
        return bool(
            self._preallocated_policy is not None
            and self._preallocated_policy.supports_policy_mean
            and _native_policy_snapshot_matches(
                self._agent, self._preallocated_policy._network
            )
        )

    def _build_preallocated_policy_runner(self) -> _PreallocatedMLPPolicy | None:
        if not _native_policy_snapshot_matches(self._agent, self._agent.network):
            # The native FP32 path requires its exact normalization arithmetic.
            # Other owned contracts use a complete frozen Torch policy.
            return None
        try:
            return self._PreallocatedMLPPolicy(
                agent=self._agent,
                device=self._device,
                obs_dim=self._obs_dim,
                batch_size=self._batch_size,
                act_dim=self._act_dim,
            )
        except Exception as exc:  # noqa: BLE001 - optional preallocated path falls back explicitly.
            _LOGGER.info(
                "Preallocated benchmark policy path unavailable: %s",
                exc,
            )
            return None

    def _build_policy_mean_into_fn(self):
        policy = self._complete_policy_snapshot or self._agent
        unscoped_policy_mean = getattr(policy, "_policy_mean_into_unscoped", None)
        if not callable(unscoped_policy_mean):
            unscoped_policy_mean = policy.policy_mean_into

        @torch.no_grad()
        def _policy_mean_into_public(
            obs_policy: torch.Tensor, out_actions: torch.Tensor
        ) -> torch.Tensor:
            policy.policy_mean_into(obs_policy, out_actions)
            return out_actions

        @torch.no_grad()
        def _policy_mean_into_compiled(
            obs_policy: torch.Tensor, out_actions: torch.Tensor
        ) -> torch.Tensor:
            unscoped_policy_mean(obs_policy, out_actions)
            return out_actions

        if self._compile_enabled:
            policy_mean_into = torch.compile(
                _policy_mean_into_compiled,
                mode=self._compile_mode,
                dynamic=False,
            )
        else:
            policy_mean_into = _policy_mean_into_public

        @torch.no_grad()
        def _scoped_policy_mean_into(
            obs_policy: torch.Tensor, out_actions: torch.Tensor
        ) -> torch.Tensor:
            # Keep process-state reads outside Dynamo's traced region.
            with network_matmul_precision():
                return policy_mean_into(obs_policy, out_actions)

        return _scoped_policy_mean_into

    def _warmup(self) -> None:
        warmup_obs = torch.zeros(
            self._batch_size, self._obs_dim, device=self._device, dtype=torch.float32
        )
        warmup_actions = torch.empty(
            self._batch_size, self._act_dim, device=self._device, dtype=torch.float32
        )
        preallocated = self._preallocated_policy
        warmup_policy_mean = bool(
            self._use_cuda_graph
            or preallocated is None
            or preallocated.supports_policy_mean
        )
        if warmup_policy_mean:
            if self._device.type == "cuda" and self._torch_stream is not None:
                self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
                with torch.cuda.stream(self._torch_stream):
                    _ = self._policy_mean_into_fn(warmup_obs, warmup_actions)
                torch.cuda.current_stream(self._device).wait_stream(self._torch_stream)
            else:
                _ = self._policy_mean_into_fn(warmup_obs, warmup_actions)
        if preallocated is not None:
            preallocated.refresh_state()
            if preallocated.supports_policy_mean:
                preallocated.policy_mean_into(warmup_obs, warmup_actions)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _capture_graph(self) -> None:
        assert self._device.type == "cuda"
        self._graph_obs = torch.zeros(
            self._batch_size,
            self._obs_dim,
            device=self._device,
            dtype=torch.float32,
        )
        self._graph_actions = torch.empty(
            self._batch_size,
            self._act_dim,
            device=self._device,
            dtype=torch.float32,
        )
        self._graph = torch.cuda.CUDAGraph()
        capture_stream = self._torch_stream or torch.cuda.Stream(device=self._device)
        capture_stream.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(capture_stream), torch.cuda.graph(self._graph):
            self._policy_mean_into_fn(self._graph_obs, self._graph_actions)
        torch.cuda.current_stream(self._device).wait_stream(capture_stream)
        capture_stream.synchronize()

    @torch.no_grad()
    def policy_mean_into(
        self,
        obs_policy: torch.Tensor,
        out_actions: torch.Tensor,
        *,
        use_cuda_graph: bool = True,
        require_preallocated: bool = False,
    ) -> None:
        if require_preallocated:
            if not self.supports_preallocated_policy_mean:
                raise RuntimeError(
                    "Captured benchmark eval requires an equivalent native policy snapshot."
                )
            assert self._preallocated_policy is not None
            self._preallocated_policy.policy_mean_into(obs_policy, out_actions)
            return
        if use_cuda_graph and self._graph is not None:
            assert self._graph_obs is not None
            assert self._graph_actions is not None
            self._graph_obs.copy_(obs_policy)
            self._graph.replay()
            out_actions.copy_(self._graph_actions)
            return
        self._policy_mean_into_fn(obs_policy, out_actions)

    def refresh_preallocated_policy_state(self) -> None:
        if self._preallocated_policy is not None:
            self._preallocated_policy.refresh_state()
        if self._complete_policy_snapshot is not None:
            self._complete_policy_snapshot.refresh(self._agent)


class RolloutBenchmarkRunner:
    def __init__(
        self,
        *,
        cfg: Any,
        agent: Any,
        source_env: Any | None,
        log_dir: Path,
        device: torch.device,
        act_dim: int,
        num_envs: int,
        play_steps: int,
        frame: str = "root",
        include_root_in_ape: bool = False,
        deterministic_reset: bool = True,
        enable_early_termination: bool | None = None,
        early_termination_mode: str | None = None,
        phc_termination_distance: float | None = None,
        require_full_horizon: bool = True,
        episode_semantics: str = "full_horizon",
        success_rate_threshold: float | None = None,
        required_metric_names: tuple[str, ...] = (),
        compile_enabled: bool = False,
        compile_mode: str = "max-autotune-no-cudagraphs",
        use_cuda_graph: bool = True,
        execution: str = "inline",
        metric_names: tuple[str, ...] | None = None,
        objective_metric: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._agent = agent
        self._obs_dim = int(agent.obs_dim)
        self._source_env = source_env
        self._log_dir = log_dir
        self._device = device
        self._env_act_dim = int(act_dim)
        self._policy_act_dim = self._env_act_dim
        self._num_envs = int(num_envs)
        self._play_steps = int(play_steps)
        self._frame = str(frame)
        self._include_root_in_ape = bool(include_root_in_ape)
        self._deterministic_reset = bool(deterministic_reset)
        self._enable_early_termination = (
            bool(enable_early_termination)
            if enable_early_termination is not None
            else None
        )
        self._early_termination_mode = (
            str(early_termination_mode) if early_termination_mode is not None else None
        )
        self._phc_termination_distance = (
            float(phc_termination_distance)
            if phc_termination_distance is not None
            else None
        )
        self._require_full_horizon = bool(require_full_horizon)
        self._episode_semantics = str(episode_semantics)
        self._success_rate_threshold = (
            float(success_rate_threshold)
            if success_rate_threshold is not None
            else None
        )
        self._compile_enabled = bool(compile_enabled)
        self._compile_mode = str(compile_mode)
        self._use_cuda_graph = bool(use_cuda_graph)
        self._execution = str(execution)
        self._requested_env_metric_names = (
            None
            if metric_names is None
            else tuple(
                name for name in metric_names if name not in _BUILTIN_ROLLOUT_METRICS
            )
        )
        self._objective_metric = (
            str(objective_metric) if objective_metric is not None else None
        )
        self._motion_groups = _benchmark_motion_groups(cfg)
        self._motion_group_ids: torch.Tensor | None = None
        required_names = tuple(str(name) for name in required_metric_names)
        if objective_metric is not None:
            required_names = (*required_names, str(objective_metric))
        self._required_metric_names = tuple(
            dict.fromkeys(
                name for name in required_names if name not in _BUILTIN_ROLLOUT_METRICS
            )
        )
        self._torch_stream = (
            torch.cuda.Stream(device=self._device)
            if self._device.type == "cuda"
            else None
        )
        self._wp_stream = (
            warp_stream_from_torch(self._torch_stream)
            if self._torch_stream is not None
            else None
        )
        self._env: Any | None = None
        self._policy_runner: _BenchmarkPolicyRunner | None = None
        self._tracking_metrics_available = False
        self._eval_episode_spec = EvalEpisodeSpec(
            validity="always",
            success=None,
        )
        self._eval_contract_registered = False
        self._episode_contract_from_env = False
        self._eval_metric_specs: dict[str, EvalMetricSpec] = {}
        self._requested_step_metric_names: tuple[str, ...] | None = None
        self._required_step_metric_names: tuple[str, ...] = ()
        self._eval_metric_accumulator: EvalMetricAccumulator | None = None
        self._workspace: _TrackingBenchmarkWorkspace | None = None
        self._captured_eval_graph: Any | None = None
        self._captured_eval_step_graph: Any | None = None
        self._captured_eval_disabled_reason: str | None = None
        self._initial_state_mean_snapshot: torch.Tensor | None = None
        self._initial_state_log_std_snapshot: torch.Tensor | None = None
        self._initial_state_version_snapshot: torch.Tensor | None = None

    def _initialize_initial_state_snapshot(self, env_raw: Any) -> None:
        learner = getattr(self._agent, "initial_state_learner", None)
        get_spec = getattr(env_raw, "get_initial_state_spec", None)
        spec = get_spec() if callable(get_spec) else None
        if learner is None and spec is None:
            return
        if learner is None or spec is None:
            raise RuntimeError(
                "benchmark initial-state adapter and learner must be enabled together"
            )
        if learner.spec.fingerprint != spec.fingerprint:
            raise ValueError(
                "benchmark initial-state fingerprint does not match the learner"
            )
        bind = getattr(env_raw, "bind_initial_state_parameters", None)
        if not callable(bind):
            raise RuntimeError(
                "benchmark environment initial-state parameter hook is unavailable"
            )
        self._initial_state_mean_snapshot = torch.empty_like(learner.mean)
        self._initial_state_log_std_snapshot = torch.empty_like(learner.log_std)
        self._initial_state_version_snapshot = torch.empty_like(learner.version)
        self._refresh_initial_state_snapshot()
        bind(
            mean=self._initial_state_mean_snapshot,
            log_std=self._initial_state_log_std_snapshot,
            version=self._initial_state_version_snapshot,
            training=False,
        )

    @torch.no_grad()
    def _refresh_initial_state_snapshot(self) -> None:
        mean_snapshot = getattr(self, "_initial_state_mean_snapshot", None)
        if mean_snapshot is None:
            return
        learner = self._agent.initial_state_learner
        assert self._initial_state_log_std_snapshot is not None
        assert self._initial_state_version_snapshot is not None
        mean_snapshot.copy_(learner.mean)
        self._initial_state_log_std_snapshot.copy_(learner.log_std)
        self._initial_state_version_snapshot.copy_(learner.version)

    def _sync_eval_env_runtime_state(self) -> bool:
        source_env = getattr(self, "_source_env", None)
        if self._env is None or source_env is None:
            return False

        source_raw = (
            source_env.unwrapped if hasattr(source_env, "unwrapped") else source_env
        )
        eval_raw = self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env
        copy_fn = getattr(eval_raw, "copy_runtime_state_from", None)
        if not callable(copy_fn):
            return False
        return bool(copy_fn(source_raw))

    def _apply_motion_group_reset_overrides(self, env_raw: Any) -> None:
        motion_groups = getattr(self, "_motion_groups", ())
        if not motion_groups:
            return
        set_overrides = getattr(env_raw, "set_analysis_reset_overrides", None)
        if not callable(set_overrides):
            raise ValueError(
                "trainer.benchmark.motion_groups requires an environment with "
                "set_analysis_reset_overrides()."
            )
        motion_clip_count = int(getattr(env_raw, "motion_clip_count", 0) or 0)
        if motion_clip_count <= 1:
            raise ValueError(
                "trainer.benchmark.motion_groups requires a multi-motion environment."
            )
        maximum_clip_index = max(group.clip_index for group in motion_groups)
        if maximum_clip_index >= motion_clip_count:
            raise ValueError(
                "trainer.benchmark.motion_groups clip_index is out of range for "
                f"{motion_clip_count} loaded clips."
            )
        group_ids, clip_indices = _motion_group_assignment(
            motion_groups,
            total_envs=self._num_envs,
            device=self._device,
        )
        self._motion_group_ids = group_ids
        set_overrides(
            clip_indices=clip_indices,
            start_frame_indices=torch.zeros_like(clip_indices),
        )

    def _append_motion_group_metrics(
        self,
        result: dict[str, float | int | bool],
        workspace: _TrackingBenchmarkWorkspace,
        episode_spec: EvalEpisodeSpec,
        *,
        temporal_metrics_seen: bool,
        accel_metrics_seen: bool,
    ) -> None:
        motion_groups = getattr(self, "_motion_groups", ())
        if not motion_groups:
            return
        motion_group_ids = getattr(self, "_motion_group_ids", None)
        if motion_group_ids is None:
            raise RuntimeError("Benchmark motion groups were not initialized.")
        result.update(
            _summarize_tracking_benchmark_motion_groups(
                workspace,
                groups=motion_groups,
                group_ids=motion_group_ids,
                tracking_metrics_available=self._tracking_metrics_available,
                episode_spec=episode_spec,
                temporal_metrics_seen=temporal_metrics_seen,
                accel_metrics_seen=accel_metrics_seen,
            )
        )

    def _synchronize_pending_work(self) -> None:
        if self._device.type != "cuda":
            return
        if self._torch_stream is not None:
            torch.cuda.current_stream(self._device).wait_stream(self._torch_stream)
        if self._wp_stream is not None:
            wp.synchronize_stream(self._wp_stream)
        if self._torch_stream is not None:
            self._torch_stream.synchronize()
        torch.cuda.synchronize(self._device)

    def close(self) -> None:
        self._synchronize_pending_work()
        self._policy_runner = None
        self._captured_eval_graph = None
        self._captured_eval_step_graph = None
        self._workspace = None
        self._eval_metric_accumulator = None
        self._eval_metric_specs = {}
        self._eval_episode_spec = EvalEpisodeSpec(
            validity="always",
            success=None,
        )
        self._eval_contract_registered = False
        self._episode_contract_from_env = False
        if self._env is not None:
            self._env.close()
            self._env = None

    def _ensure_workspace_shape(
        self,
        *,
        total_envs: int,
        obs_dim: int,
    ) -> _TrackingBenchmarkWorkspace:
        workspace = getattr(self, "_workspace", None)
        if workspace is None or not workspace.matches(
            total_envs=total_envs,
            obs_dim=obs_dim,
            policy_act_dim=self._policy_act_dim,
            env_act_dim=self._env_act_dim,
        ):
            workspace = _TrackingBenchmarkWorkspace(
                device=self._device,
                total_envs=total_envs,
                obs_dim=obs_dim,
                policy_act_dim=self._policy_act_dim,
                env_act_dim=self._env_act_dim,
                agent=getattr(self, "_agent", None),
            )
            self._workspace = workspace
            self._captured_eval_graph = None
            self._captured_eval_step_graph = None
            _LOGGER.info(
                "Allocated persistent rollout benchmark workspace: num_envs=%d obs_dim=%d policy_act_dim=%d env_act_dim=%d",
                total_envs,
                obs_dim,
                self._policy_act_dim,
                self._env_act_dim,
            )
        return workspace

    def _ensure_workspace(
        self, obs_policy: torch.Tensor
    ) -> _TrackingBenchmarkWorkspace:
        workspace = self._ensure_workspace_shape(
            total_envs=int(obs_policy.shape[0]),
            obs_dim=int(obs_policy.shape[1]),
        )
        workspace.obs_policy.copy_(obs_policy)
        return workspace

    def _ensure_observation_workspace(
        self, observations: Mapping[str, torch.Tensor]
    ) -> _TrackingBenchmarkWorkspace:
        agent = getattr(self, "_agent", None)
        obs_dim = (
            int(agent.obs_dim)
            if bool(getattr(agent, "asymmetric_observations", False))
            else int(observations["policy"].shape[1])
        )
        workspace = self._ensure_workspace_shape(
            total_envs=int(observations["policy"].shape[0]), obs_dim=obs_dim
        )
        copy_agent_observation_into(
            agent=agent, observations=observations, destination=workspace.obs_policy
        )
        return workspace

    def _legacy_episode_spec(self, env_raw: Any) -> EvalEpisodeSpec:
        terminal_success = (
            getattr(self, "_episode_semantics", "full_horizon") == "terminal_success"
        )
        reset_time_outs = getattr(env_raw, "reset_time_outs", None)
        if isinstance(reset_time_outs, torch.Tensor):
            natural_completion, has_natural_completion = _resolve_motion_end_time_outs(
                env_raw,
                fallback_time_outs=reset_time_outs,
            )
        else:
            natural_completion = None
            has_natural_completion = False
        reset_terminated = getattr(env_raw, "reset_terminated", None)
        if isinstance(reset_terminated, torch.Tensor):
            terminal_success_mask, has_terminal_success = (
                _resolve_terminal_success_mask(
                    env_raw,
                    fallback_terminated=reset_terminated,
                )
            )
        else:
            terminal_success_mask = None
            has_terminal_success = False
        return EvalEpisodeSpec(
            validity=(
                "always"
                if terminal_success
                or not bool(getattr(self, "_require_full_horizon", True))
                else "full_horizon"
            ),
            success="terminal_success" if terminal_success else "full_horizon",
            natural_completion=(natural_completion if has_natural_completion else None),
            terminal_success=(terminal_success_mask if has_terminal_success else None),
        )

    def _current_episode_spec(self, env_raw: Any) -> EvalEpisodeSpec:
        episode_spec = getattr(self, "_eval_episode_spec", None)
        if isinstance(episode_spec, EvalEpisodeSpec):
            return episode_spec
        return self._legacy_episode_spec(env_raw)

    def _validate_eval_contract(self, env_raw: Any) -> None:
        if not bool(getattr(self, "_eval_contract_registered", False)):
            return
        accumulator = getattr(self, "_eval_metric_accumulator", None)
        if accumulator is not None:
            current_specs = resolve_eval_metric_specs(
                env_raw,
                requested=self._requested_step_metric_names,
                required=self._required_step_metric_names,
                num_envs=self._num_envs,
                device=self._device,
            )
            accumulator.validate_sources(current_specs)
        if not bool(getattr(self, "_episode_contract_from_env", False)):
            return
        current_episode = resolve_eval_episode_spec(
            env_raw,
            num_envs=self._num_envs,
            device=self._device,
            fallback_validity=self._eval_episode_spec.validity,
            fallback_success=self._eval_episode_spec.success,
        )
        expected = self._eval_episode_spec
        if (
            current_episode.validity != expected.validity
            or current_episode.success != expected.success
            or current_episode.natural_completion is not expected.natural_completion
            or current_episode.terminal_success is not expected.terminal_success
        ):
            raise RuntimeError(
                "environment replaced its registered evaluation episode contract"
            )

    def _captured_eval_supported(self, env_raw: Any, step_with_io: Any) -> bool:
        if not bool(getattr(self, "_use_cuda_graph", False)):
            return False
        reason: str | None = None
        if self._device.type != "cuda":
            reason = "device is not CUDA"
        elif self._torch_stream is None or self._wp_stream is None:
            reason = "CUDA streams are unavailable"
        elif not callable(getattr(env_raw, "reset_into", None)):
            reason = "env does not expose reset_into()"
        elif not callable(step_with_io):
            reason = "env does not expose step_with_io()"
        elif callable(
            getattr(env_raw, "supports_allocation_free_step_capture", None)
        ) and not bool(env_raw.supports_allocation_free_step_capture()):
            reason = (
                "env MuJoCo Warp step does not expose allocation-free capture scratch"
            )
        elif self._policy_runner is not None and not (
            self._policy_runner.supports_preallocated_policy_mean
            or self._policy_runner.can_capture_complete_policy_mean
        ):
            reason = "policy path is not preallocated/capture-safe"

        if reason is None:
            self._captured_eval_disabled_reason = None
            return True
        self._captured_eval_disabled_reason = reason
        return False

    def _prepare_captured_eval_temporal_state(
        self,
        env_raw: Any,
        workspace: _TrackingBenchmarkWorkspace,
    ) -> dict[str, torch.Tensor]:
        metrics = _collect_tracking_metric_tensors(
            env_raw,
            frame=self._frame,
            include_root_in_ape=self._include_root_in_ape,
        )
        if metrics is None:
            raise RuntimeError(
                "Captured benchmark eval requested, but env does not expose "
                "get_reference_tracking_metrics()."
            )
        sim_body_linvel = metrics.get("tracking_body_linvel")
        ref_body_linvel = metrics.get("reference_body_linvel")
        sim_root_quat = metrics.get("sim_root_quat")
        ref_root_quat = metrics.get("ref_root_quat")
        if (
            not isinstance(sim_body_linvel, torch.Tensor)
            or not isinstance(ref_body_linvel, torch.Tensor)
            or not isinstance(sim_root_quat, torch.Tensor)
            or not isinstance(ref_root_quat, torch.Tensor)
        ):
            raise RuntimeError(
                "Captured benchmark eval requires APE, EE, velocity, and "
                "acceleration metric tensors from get_reference_tracking_metrics()."
            )
        workspace.ensure_temporal_metric_workspace(sim_body_linvel)
        assert workspace.prev_sim_body_linvel is not None
        assert workspace.prev_ref_body_linvel is not None
        assert workspace.prev_sim_root_quat is not None
        assert workspace.prev_ref_root_quat is not None
        workspace.prev_sim_body_linvel.copy_(sim_body_linvel)
        workspace.prev_ref_body_linvel.copy_(ref_body_linvel)
        workspace.prev_sim_root_quat.copy_(sim_root_quat)
        workspace.prev_ref_root_quat.copy_(ref_root_quat)
        return metrics

    def _compute_captured_eval_metrics_body(
        self,
        env_raw: Any,
        workspace: _TrackingBenchmarkWorkspace,
    ) -> dict[str, torch.Tensor]:
        metrics = _collect_tracking_metric_tensors(
            env_raw,
            frame=self._frame,
            include_root_in_ape=self._include_root_in_ape,
        )
        if metrics is None:
            raise RuntimeError(
                "Captured benchmark eval requested, but env does not expose "
                "get_reference_tracking_metrics()."
            )
        ape_tensor = metrics["ape"]
        global_mpjpe_tensor = metrics["global_mpjpe"]
        ee_tensor = metrics["ee_error"]
        if (
            ape_tensor.numel() != workspace.total_envs
            or global_mpjpe_tensor.numel() != workspace.total_envs
            or ee_tensor.numel() != workspace.total_envs
        ):
            raise RuntimeError(
                "Benchmark tracking metrics must return one value per environment."
            )
        sim_body_linvel = metrics.get("tracking_body_linvel")
        ref_body_linvel = metrics.get("reference_body_linvel")
        sim_root_quat = metrics.get("sim_root_quat")
        ref_root_quat = metrics.get("ref_root_quat")
        if (
            not isinstance(sim_body_linvel, torch.Tensor)
            or not isinstance(ref_body_linvel, torch.Tensor)
            or not isinstance(sim_root_quat, torch.Tensor)
            or not isinstance(ref_root_quat, torch.Tensor)
        ):
            raise RuntimeError(
                "Captured benchmark eval requires temporal tracking metric tensors."
            )
        workspace.ensure_temporal_metric_workspace(sim_body_linvel)
        assert workspace.vel_metric is not None
        assert workspace.accel_metric is not None
        assert workspace.prev_sim_body_linvel is not None
        assert workspace.prev_ref_body_linvel is not None
        assert workspace.prev_sim_root_quat is not None
        assert workspace.prev_ref_root_quat is not None
        tracking_metrics_warp.compute_mean_velocity_error_into_warp(
            sim_body_linvel,
            ref_body_linvel,
            sim_root_quat=sim_root_quat,
            ref_root_quat=ref_root_quat,
            frame=self._frame,
            out=workspace.vel_metric,
        )
        tracking_metrics_warp.compute_mean_acceleration_error_into_warp(
            sim_body_linvel,
            ref_body_linvel,
            workspace.prev_sim_body_linvel,
            workspace.prev_ref_body_linvel,
            dt=float(env_raw.step_dt),
            sim_root_quat=sim_root_quat,
            ref_root_quat=ref_root_quat,
            prev_sim_root_quat=workspace.prev_sim_root_quat,
            prev_ref_root_quat=workspace.prev_ref_root_quat,
            frame=self._frame,
            out=workspace.accel_metric,
        )
        return metrics

    def _captured_eval_step_body(
        self,
        *,
        env_raw: Any,
        workspace: _TrackingBenchmarkWorkspace,
        step_with_io: Any,
        require_preallocated_policy: bool = True,
    ) -> None:
        assert self._policy_runner is not None
        if self._tracking_metrics_available:
            metrics = self._compute_captured_eval_metrics_body(env_raw, workspace)
            assert workspace.vel_metric is not None
            assert workspace.accel_metric is not None
            wp.launch(
                _tracking_eval_accumulate_metrics_kernel,
                dim=workspace.total_envs,
                inputs=[
                    workspace.eval_condition,
                    workspace.active_mask,
                    metrics["ape"],
                    metrics["global_mpjpe"],
                    metrics["ee_error"],
                    workspace.vel_metric.mean_error,
                    workspace.accel_metric.mean_error,
                    workspace.eval_step_index,
                    workspace.ape_sum,
                    workspace.global_mpjpe_sum,
                    workspace.ee_sum,
                    workspace.vel_sum,
                    workspace.accel_sum,
                    workspace.steps_per_env,
                    workspace.accel_steps_per_env,
                    workspace.last_ape,
                    workspace.last_global_mpjpe,
                    workspace.last_ee,
                    workspace.last_vel,
                    workspace.last_accel,
                ],
                device=wp.device_from_torch(workspace.reward_sum.device),
            )
            assert workspace.prev_sim_body_linvel is not None
            assert workspace.prev_ref_body_linvel is not None
            assert workspace.prev_sim_root_quat is not None
            assert workspace.prev_ref_root_quat is not None
            workspace.prev_sim_body_linvel.copy_(metrics["tracking_body_linvel"])
            workspace.prev_ref_body_linvel.copy_(metrics["reference_body_linvel"])
            workspace.prev_sim_root_quat.copy_(metrics["sim_root_quat"])
            workspace.prev_ref_root_quat.copy_(metrics["ref_root_quat"])

        self._policy_runner.policy_mean_into(
            workspace.obs_policy,
            workspace.policy_actions,
            use_cuda_graph=False,
            require_preallocated=require_preallocated_policy,
        )
        step_with_io(
            workspace.step_io,
            use_cuda_graph=False,
            reset_done=True,
        )
        torch.logical_or(
            workspace.next_terminated,
            workspace.next_truncated,
            out=workspace.next_dones,
        )
        if not self._tracking_metrics_available:
            wp.launch(
                _rollout_eval_increment_steps_kernel,
                dim=workspace.total_envs,
                inputs=[
                    workspace.eval_condition,
                    workspace.active_mask,
                    workspace.steps_per_env,
                ],
                device=wp.device_from_torch(workspace.reward_sum.device),
            )
        eval_metric_accumulator = self._eval_metric_accumulator
        if eval_metric_accumulator is not None:
            eval_metric_accumulator.collect_post_step(workspace.active_mask)
        episode_spec = self._current_episode_spec(env_raw)
        motion_end_time_outs = (
            episode_spec.natural_completion
            if episode_spec.natural_completion is not None
            else env_raw.reset_time_outs
        )
        terminal_success_mask = (
            episode_spec.terminal_success
            if episode_spec.terminal_success is not None
            else env_raw.reset_terminated
        )
        wp.launch(
            _tracking_eval_reset_active_count_kernel,
            dim=1,
            inputs=[
                workspace.eval_condition,
                workspace.eval_active_count,
            ],
            device=wp.device_from_torch(workspace.reward_sum.device),
        )
        wp.launch(
            _tracking_eval_update_done_kernel,
            dim=workspace.total_envs,
            inputs=[
                workspace.eval_condition,
                workspace.active_mask,
                workspace.valid_mask,
                workspace.success_mask,
                workspace.next_rewards,
                env_raw.reset_terminated,
                env_raw.reset_time_outs,
                motion_end_time_outs,
                terminal_success_mask,
                workspace.reward_sum,
                workspace.eval_step_index,
                workspace.eval_completed_steps,
                workspace.eval_first_invalid_step,
                workspace.eval_first_failure_step,
                workspace.eval_active_count,
                workspace.eval_terminated_any,
                workspace.eval_truncated_any,
                int(self._play_steps),
                int(episode_spec.validity == "full_horizon"),
                int(episode_spec.success == "full_horizon"),
                int(episode_spec.natural_completion is not None),
                int(episode_spec.terminal_success is not None),
                int(episode_spec.success == "terminal_success"),
            ],
            device=wp.device_from_torch(workspace.reward_sum.device),
        )
        wp.launch(
            _tracking_eval_finish_step_kernel,
            dim=1,
            inputs=[
                workspace.eval_condition,
                workspace.eval_step_index,
                workspace.eval_completed_steps,
                workspace.eval_active_count,
                int(self._play_steps),
            ],
            device=wp.device_from_torch(workspace.reward_sum.device),
        )

    def _captured_eval_reset_body(
        self,
        *,
        env_raw: Any,
        workspace: _TrackingBenchmarkWorkspace,
    ) -> None:
        assert self._policy_runner is not None
        workspace.reset_episode_accumulators()
        if self._eval_metric_accumulator is not None:
            self._eval_metric_accumulator.reset()
        reset_into = getattr(env_raw, "reset_into", None)
        if not callable(reset_into):
            raise RuntimeError("Captured benchmark eval requires env.reset_into().")
        reset_result = reset_into(
            workspace.step_io.outputs.observations["policy"],
            reset_mask=workspace.eval_reset_mask,
        )
        if bool(getattr(self._agent, "asymmetric_observations", False)):
            observations, _ = reset_result
            workspace.step_io.outputs.observations["critic"].copy_(
                observations["critic"]
            )
        if bool(getattr(self, "_tracking_metrics_available", True)):
            self._prepare_captured_eval_temporal_state(env_raw, workspace)

    def _warmup_captured_eval_step_body(
        self,
        *,
        env_raw: Any,
        workspace: _TrackingBenchmarkWorkspace,
        step_with_io: Any,
    ) -> _TrackingBenchmarkWorkspace:
        assert self._env is not None
        assert self._policy_runner is not None
        assert self._torch_stream is not None
        assert self._wp_stream is not None
        self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self._torch_stream):
                with torch.cuda.nvtx.range("warmup_benchmark_eval_reset_body"):
                    self._captured_eval_reset_body(
                        env_raw=env_raw,
                        workspace=workspace,
                    )
                with torch.cuda.nvtx.range("warmup_benchmark_eval_step_body"):
                    self._captured_eval_step_body(
                        env_raw=env_raw,
                        workspace=workspace,
                        step_with_io=step_with_io,
                        require_preallocated_policy=(
                            self._policy_runner.supports_preallocated_policy_mean
                        ),
                    )
                with torch.cuda.nvtx.range("warmup_benchmark_eval_reset_body"):
                    self._captured_eval_reset_body(
                        env_raw=env_raw,
                        workspace=workspace,
                    )
        torch.cuda.current_stream(self._device).wait_stream(self._torch_stream)
        self._synchronize_pending_work()
        return workspace

    def _ensure_captured_eval_graph(
        self,
        *,
        env_raw: Any,
        workspace: _TrackingBenchmarkWorkspace,
        step_with_io: Any,
    ) -> bool:
        if self._captured_eval_graph is not None:
            return True
        if not self._captured_eval_supported(env_raw, step_with_io):
            reason = self._captured_eval_disabled_reason or "unsupported eval graph"
            raise RuntimeError(
                f"Benchmark eval CUDA graph was requested but is unavailable: {reason}. "
                "Set trainer.benchmark.cuda_graph: false to use eager eval "
                "(command-line override: ++trainer.benchmark.cuda_graph=false)."
            )
        assert self._torch_stream is not None
        assert self._wp_stream is not None
        assert self._policy_runner is not None
        workspace = self._warmup_captured_eval_step_body(
            env_raw=env_raw,
            workspace=workspace,
            step_with_io=step_with_io,
        )
        self._policy_runner.refresh_preallocated_policy_state()
        self._refresh_initial_state_snapshot()
        if bool(getattr(self, "_tracking_metrics_available", True)):
            self._prepare_captured_eval_temporal_state(env_raw, workspace)

        try:
            with external_step_capture_window(env_raw, unroll_solver_iterations=True):
                self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
                with (
                    wp.ScopedStream(
                        self._wp_stream,
                        sync_enter=False,
                        sync_exit=False,
                    ),
                    torch.cuda.stream(self._torch_stream),
                ):
                    # Environment stepping can allocate Torch temporaries even
                    # when the top-level policy uses preallocated inference.
                    # Own the whole step's allocation pool until the conditional
                    # graph has finished replaying it.
                    with capture_warp_graph(stream=self._wp_stream) as step_capture:
                        self._captured_eval_step_body(
                            env_raw=env_raw,
                            workspace=workspace,
                            step_with_io=step_with_io,
                            require_preallocated_policy=(
                                self._policy_runner.supports_preallocated_policy_mean
                            ),
                        )
                    self._captured_eval_step_graph = step_capture.graph
                    with (
                        torch.cuda.nvtx.range(
                            "capture_benchmark_eval_conditional_graph"
                        ),
                        capture_warp_graph(
                            stream=self._wp_stream
                        ) as conditional_capture,
                    ):
                        self._captured_eval_reset_body(
                            env_raw=env_raw,
                            workspace=workspace,
                        )
                        _capture_eval_conditional_horizon(
                            workspace.eval_condition_wp,
                            self._captured_eval_step_graph,
                        )
                    self._captured_eval_graph = conditional_capture.graph
                    with torch.cuda.nvtx.range(
                        "instantiate_benchmark_eval_conditional_graph"
                    ):
                        wp.capture_launch(
                            self._captured_eval_graph,
                            stream=self._wp_stream,
                        )
                self._torch_stream.synchronize()
        except Exception as exc:
            self._captured_eval_graph = None
            self._captured_eval_step_graph = None
            raise RuntimeError(
                f"Benchmark eval CUDA graph capture failed: {exc}"
            ) from exc
        _LOGGER.info("Captured and instantiated benchmark eval conditional graph")
        return True

    def _launch_captured_eval_graphs(self) -> None:
        assert self._captured_eval_graph is not None
        wp.capture_launch(
            self._captured_eval_graph,
            stream=self._wp_stream,
        )

    def _try_run_captured_eval(
        self,
        *,
        env_raw: Any,
        workspace: _TrackingBenchmarkWorkspace,
        step_with_io: Any,
    ) -> dict[str, float | int | bool] | None:
        if not self._captured_eval_supported(env_raw, step_with_io):
            return None
        if not self._ensure_captured_eval_graph(
            env_raw=env_raw,
            workspace=workspace,
            step_with_io=step_with_io,
        ):
            return None
        assert self._torch_stream is not None
        assert self._wp_stream is not None

        assert self._policy_runner is not None
        self._policy_runner.refresh_preallocated_policy_state()
        self._refresh_initial_state_snapshot()
        with external_step_capture_window(env_raw):
            self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
            with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
                with torch.cuda.stream(self._torch_stream):
                    with torch.cuda.nvtx.range("replay_benchmark_eval_graphs"):
                        self._launch_captured_eval_graphs()
            torch.cuda.current_stream(self._device).wait_stream(self._torch_stream)
            self._synchronize_pending_work()
        return self._summarize_captured_workspace(workspace)

    def _summarize_captured_workspace(
        self,
        workspace: _TrackingBenchmarkWorkspace,
    ) -> dict[str, float | int | bool]:
        episode_spec = getattr(self, "_eval_episode_spec", None)
        if not isinstance(episode_spec, EvalEpisodeSpec):
            episode_spec = EvalEpisodeSpec(
                validity="full_horizon",
                success="full_horizon",
            )
        tracking_metrics_available = bool(
            getattr(self, "_tracking_metrics_available", True)
        )
        result = _summarize_tracking_benchmark_workspace(
            workspace,
            completed_steps=int(workspace.eval_completed_steps.item()),
            first_invalid_step=int(workspace.eval_first_invalid_step.item()),
            first_failure_step=int(workspace.eval_first_failure_step.item()),
            terminated_any=bool(int(workspace.eval_terminated_any.item())),
            truncated_any=bool(int(workspace.eval_truncated_any.item())),
            tracking_metrics_available=tracking_metrics_available,
            temporal_metrics_seen=tracking_metrics_available,
            accel_metrics_seen=tracking_metrics_available,
            episode_spec=episode_spec,
        )
        eval_metric_accumulator = getattr(self, "_eval_metric_accumulator", None)
        if eval_metric_accumulator is not None:
            result.update(
                eval_metric_accumulator.summarize(valid_mask=workspace.valid_mask)
            )
        self._append_motion_group_metrics(
            result,
            workspace,
            episode_spec,
            temporal_metrics_seen=tracking_metrics_available,
            accel_metrics_seen=tracking_metrics_available,
        )
        self._filter_eval_result_metrics(result)
        result["captured_eval"] = True
        result["captured_eval_conditional_loop"] = True
        result["captured_eval_full_horizon_graph"] = False
        return result

    @torch.no_grad()
    def launch(
        self,
        *,
        snapshot_metadata: dict[str, int],
    ) -> TrackingBenchmarkJob:
        if getattr(self, "_execution", "inline") != "overlap":
            raise RuntimeError("Asynchronous benchmark launch requires overlap mode.")
        self._ensure_initialized()
        assert self._env is not None
        assert self._policy_runner is not None
        assert self._torch_stream is not None
        assert self._wp_stream is not None

        self._sync_eval_env_runtime_state()
        env_raw = self._env.unwrapped
        self._apply_motion_group_reset_overrides(env_raw)
        step_with_io = getattr(env_raw, "step_with_io", None)
        workspace = self._ensure_workspace_shape(
            total_envs=self._num_envs,
            obs_dim=self._obs_dim,
        )
        if not self._captured_eval_supported(env_raw, step_with_io):
            reason = self._captured_eval_disabled_reason or "unsupported captured eval"
            raise RuntimeError(f"Overlapped benchmark eval is unavailable: {reason}.")
        if not self._ensure_captured_eval_graph(
            env_raw=env_raw,
            workspace=workspace,
            step_with_io=step_with_io,
        ):
            reason = self._captured_eval_disabled_reason or "graph capture failed"
            raise RuntimeError(f"Overlapped benchmark eval is unavailable: {reason}.")

        self._policy_runner.refresh_preallocated_policy_state()
        self._refresh_initial_state_snapshot()

        start_event = torch.cuda.Event(enable_timing=True)
        complete_event = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(self._device)
        self._torch_stream.wait_stream(current_stream)
        with external_step_capture_window(env_raw):
            with wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False):
                with torch.cuda.stream(self._torch_stream):
                    start_event.record(self._torch_stream)
                    with torch.cuda.nvtx.range("replay_benchmark_eval_graph"):
                        self._launch_captured_eval_graphs()
                    complete_event.record(self._torch_stream)
        return TrackingBenchmarkJob(
            runner=self,
            workspace=workspace,
            snapshot_metadata={
                str(key): int(value) for key, value in snapshot_metadata.items()
            },
            start_event=start_event,
            complete_event=complete_event,
        )

    def _ensure_initialized(self) -> None:
        if self._env is not None and self._policy_runner is not None:
            return

        self._env = _build_tracking_benchmark_env(
            self._cfg,
            log_dir=self._log_dir,
            num_envs=self._num_envs,
            play_steps=self._play_steps,
            deterministic_reset=self._deterministic_reset,
            enable_early_termination=self._enable_early_termination,
            early_termination_mode=self._early_termination_mode,
            phc_termination_distance=self._phc_termination_distance,
            use_cuda_graph=self._use_cuda_graph,
        )
        env_raw = self._env.unwrapped
        self._initialize_initial_state_snapshot(env_raw)
        self._apply_motion_group_reset_overrides(env_raw)
        episode_getter = getattr(env_raw, "get_eval_episode_spec", None)
        self._episode_contract_from_env = callable(episode_getter)
        self._eval_episode_spec = (
            resolve_eval_episode_spec(
                env_raw,
                num_envs=self._num_envs,
                device=self._device,
            )
            if callable(episode_getter)
            else self._legacy_episode_spec(env_raw)
        )
        available_result_metric_specs = load_eval_result_metric_specs(env_raw)
        available_result_names = frozenset(available_result_metric_specs)
        requested_result_names = (
            None
            if self._requested_env_metric_names is None
            else tuple(
                name
                for name in self._requested_env_metric_names
                if name in available_result_names
            )
        )
        requested_step_names = (
            None
            if self._requested_env_metric_names is None
            else tuple(
                name
                for name in self._requested_env_metric_names
                if name not in available_result_names
            )
        )
        required_result_names = tuple(
            name
            for name in self._required_metric_names
            if name in available_result_names
        )
        required_step_names = tuple(
            name
            for name in self._required_metric_names
            if name not in available_result_names
        )
        self._requested_step_metric_names = requested_step_names
        self._required_step_metric_names = required_step_names
        self._eval_metric_specs = resolve_eval_metric_specs(
            env_raw,
            requested=requested_step_names,
            required=required_step_names,
            num_envs=self._num_envs,
            device=self._device,
        )
        self._eval_result_metric_specs = resolve_eval_result_metric_specs(
            env_raw,
            requested=requested_result_names,
            required=required_result_names,
        )
        overlapping_metric_names = set(self._eval_metric_specs).intersection(
            self._eval_result_metric_specs
        )
        if overlapping_metric_names:
            names = ", ".join(sorted(overlapping_metric_names))
            raise ValueError(f"environment defines duplicate eval metrics: {names}")
        if (
            getattr(self, "_objective_metric", None) == "success_rate"
            and self._eval_episode_spec.success is None
        ):
            raise ValueError(
                "benchmark objective success_rate requires the environment to "
                "define evaluation success"
            )
        if (
            self._success_rate_threshold is not None
            and self._eval_episode_spec.success is None
        ):
            raise ValueError(
                "benchmark success_rate threshold requires the environment to "
                "define evaluation success"
            )
        exposes_tracking_metrics = callable(
            getattr(env_raw, "get_reference_tracking_metrics", None)
        )
        self._tracking_metrics_available = bool(self._eval_result_metric_specs)
        if self._tracking_metrics_available and not exposes_tracking_metrics:
            raise RuntimeError(
                "Environment registered evaluation result metrics but does not "
                "expose get_reference_tracking_metrics()."
            )
        self._available_eval_result_metric_names = available_result_names
        self._eval_metric_accumulator = EvalMetricAccumulator(self._eval_metric_specs)
        self._eval_contract_registered = True

        self._policy_runner = _BenchmarkPolicyRunner(
            agent=self._agent,
            device=self._device,
            obs_dim=self._obs_dim,
            batch_size=self._num_envs,
            compile_enabled=self._compile_enabled,
            compile_mode=self._compile_mode,
            use_cuda_graph=False,
            torch_stream=self._torch_stream,
        )
        _LOGGER.info(
            "Initialized rollout benchmark runner lazily: compile=%s cuda_graph=%s num_envs=%d play_steps=%d stochastic_reset=%s success_rate_threshold=%s env_result_metrics=%s",
            self._compile_enabled,
            self._use_cuda_graph,
            self._num_envs,
            self._play_steps,
            not self._deterministic_reset,
            (
                f"{self._success_rate_threshold:.6f}"
                if self._success_rate_threshold is not None
                else "None"
            ),
            self._tracking_metrics_available,
        )

    @property
    def metric_names(self) -> tuple[str, ...]:
        return (
            *getattr(self, "_eval_metric_specs", {}),
            *getattr(self, "_eval_result_metric_specs", {}),
        )

    @property
    def metric_labels(self) -> dict[str, str]:
        labels = {
            name: spec.label
            for name, spec in getattr(self, "_eval_metric_specs", {}).items()
        }
        labels.update(
            {
                (spec.result_key or name): spec.label
                for name, spec in getattr(self, "_eval_result_metric_specs", {}).items()
            }
        )
        return labels

    @property
    def metric_statistics(self) -> dict[str, str]:
        statistics = {
            name: spec.statistic
            for name, spec in getattr(self, "_eval_metric_specs", {}).items()
        }
        statistics.update(
            {
                (spec.result_key or name): spec.statistic
                for name, spec in getattr(self, "_eval_result_metric_specs", {}).items()
            }
        )
        return statistics

    @property
    def metric_groups(self) -> tuple[str, ...]:
        return tuple(group.name for group in getattr(self, "_motion_groups", ()))

    @property
    def finite_metric_names(self) -> tuple[str, ...]:
        return (
            *getattr(self, "_eval_metric_specs", {}),
            *(
                (spec.result_key or name)
                for name, spec in getattr(self, "_eval_result_metric_specs", {}).items()
                if spec.finite_required
            ),
        )

    def _filter_eval_result_metrics(
        self,
        result: dict[str, float | int | bool],
    ) -> None:
        selected = set(getattr(self, "_eval_result_metric_specs", {}))
        for name in getattr(self, "_available_eval_result_metric_names", ()):
            if name not in selected:
                result.pop(name, None)

    @torch.no_grad()
    def prewarm_graphs(self) -> None:
        self._ensure_initialized()
        assert self._env is not None
        assert self._policy_runner is not None

        self._sync_eval_env_runtime_state()
        self._policy_runner.refresh_preallocated_policy_state()
        self._refresh_initial_state_snapshot()

        self._apply_motion_group_reset_overrides(self._env.unwrapped)
        obs, _ = self._env.reset()
        workspace = self._ensure_observation_workspace(obs)
        obs_policy = workspace.obs_policy
        policy_actions = workspace.policy_actions
        env_actions = workspace.env_actions
        env_raw = self._env.unwrapped
        self._validate_eval_contract(env_raw)
        step_with_io = getattr(env_raw, "step_with_io", None)
        use_fast_step_with_io = callable(step_with_io)

        torch_stream_ctx = (
            torch.cuda.stream(self._torch_stream)
            if self._torch_stream is not None
            else nullcontext()
        )
        wp_stream_ctx = (
            wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False)
            if self._wp_stream is not None
            else nullcontext()
        )
        if self._torch_stream is not None:
            self._torch_stream.wait_stream(torch.cuda.current_stream(self._device))
        with wp_stream_ctx, torch_stream_ctx:
            for _ in range(2):
                self._policy_runner.policy_mean_into(
                    obs_policy,
                    policy_actions,
                )
                if use_fast_step_with_io:
                    step_with_io(workspace.step_io)
                    torch.logical_or(
                        workspace.next_terminated,
                        workspace.next_truncated,
                        out=workspace.next_dones,
                    )
                    obs_policy = workspace.obs_policy
                else:
                    obs, _, _, _, _ = self._env.step(env_actions)
                    copy_agent_observation_into(
                        agent=getattr(self, "_agent", None),
                        observations=obs,
                        destination=workspace.obs_policy,
                    )
                    obs_policy = workspace.obs_policy

            if bool(getattr(self, "_use_cuda_graph", False)):
                workspace = self._ensure_workspace_shape(
                    total_envs=self._num_envs,
                    obs_dim=self._obs_dim,
                )
                step_with_io = getattr(env_raw, "step_with_io", None)
                captured = self._ensure_captured_eval_graph(
                    env_raw=env_raw,
                    workspace=workspace,
                    step_with_io=step_with_io,
                )
                assert captured

        self._synchronize_pending_work()

    @torch.no_grad()
    def run(self) -> dict[str, float | int | bool]:
        self._ensure_initialized()
        assert self._env is not None
        assert self._policy_runner is not None

        self._sync_eval_env_runtime_state()

        env_raw = self._env.unwrapped
        self._apply_motion_group_reset_overrides(env_raw)
        self._validate_eval_contract(env_raw)
        step_with_io = getattr(env_raw, "step_with_io", None)
        if bool(getattr(self, "_use_cuda_graph", False)):
            if not self._captured_eval_supported(env_raw, step_with_io):
                reason = self._captured_eval_disabled_reason or "unsupported eval graph"
                raise RuntimeError(
                    "Benchmark eval CUDA graph was requested but is unavailable: "
                    f"{reason}. Set trainer.benchmark.cuda_graph: false to use "
                    "eager eval (command-line override: "
                    "++trainer.benchmark.cuda_graph=false)."
                )
            captured_workspace = self._ensure_workspace_shape(
                total_envs=self._num_envs,
                obs_dim=self._obs_dim,
            )
            captured_result = self._try_run_captured_eval(
                env_raw=env_raw,
                workspace=captured_workspace,
                step_with_io=step_with_io,
            )
            if captured_result is not None:
                return captured_result
            raise RuntimeError("Benchmark eval CUDA graph did not produce a result.")

        # Captured and overlap execution refresh their snapshots in their own
        # launch paths. Eager execution must freeze the current policy as well.
        self._policy_runner.refresh_preallocated_policy_state()
        self._refresh_initial_state_snapshot()
        obs, _ = self._env.reset()
        workspace = self._ensure_observation_workspace(obs)
        obs_policy = workspace.obs_policy
        total_envs = workspace.total_envs
        policy_actions = workspace.policy_actions
        env_actions = workspace.env_actions
        use_fast_step_with_io = callable(step_with_io)
        workspace.reset_episode_accumulators()
        eval_metric_accumulator = getattr(self, "_eval_metric_accumulator", None)
        if eval_metric_accumulator is not None:
            eval_metric_accumulator.reset()
        active_mask = workspace.active_mask
        valid_mask = workspace.valid_mask
        success_mask = workspace.success_mask
        reward_sum = workspace.reward_sum
        ape_sum = workspace.ape_sum
        ee_sum = workspace.ee_sum
        vel_sum = workspace.vel_sum
        accel_sum = workspace.accel_sum
        steps_per_env = workspace.steps_per_env
        accel_steps_per_env = workspace.accel_steps_per_env
        last_ape = workspace.last_ape
        last_ee = workspace.last_ee
        last_vel = workspace.last_vel
        last_accel = workspace.last_accel
        prev_sim_body_linvel: torch.Tensor | None = None
        prev_ref_body_linvel: torch.Tensor | None = None
        prev_sim_root_quat: torch.Tensor | None = None
        prev_ref_root_quat: torch.Tensor | None = None
        temporal_metrics_seen = False
        accel_metrics_seen = False
        first_invalid_step = 0
        first_failure_step = 0
        completed_steps = 0
        terminated_any = False
        truncated_any = False
        episode_spec = self._current_episode_spec(env_raw)

        try:
            for step_idx in range(self._play_steps):
                steps_per_env.add_(active_mask.to(dtype=steps_per_env.dtype))
                if self._tracking_metrics_available:
                    metrics = _collect_tracking_metric_tensors(
                        env_raw,
                        frame=self._frame,
                        include_root_in_ape=self._include_root_in_ape,
                    )
                    if metrics is None:
                        raise RuntimeError(
                            "Benchmark eval requested, but env does not expose "
                            "get_reference_tracking_metrics()."
                        )
                    ape_tensor = metrics["ape"]
                    global_mpjpe_tensor = metrics["global_mpjpe"]
                    ee_tensor = metrics["ee_error"]
                    if (
                        ape_tensor.numel() != total_envs
                        or global_mpjpe_tensor.numel() != total_envs
                        or ee_tensor.numel() != total_envs
                    ):
                        raise RuntimeError(
                            "Benchmark tracking metrics must return one value per environment."
                        )
                    ape_sum.add_(ape_tensor * active_mask.to(dtype=ape_tensor.dtype))
                    workspace.global_mpjpe_sum.add_(
                        global_mpjpe_tensor
                        * active_mask.to(dtype=global_mpjpe_tensor.dtype)
                    )
                    ee_sum.add_(ee_tensor * active_mask.to(dtype=ee_tensor.dtype))
                    last_ape.copy_(torch.where(active_mask, ape_tensor, last_ape))
                    workspace.last_global_mpjpe.copy_(
                        torch.where(
                            active_mask,
                            global_mpjpe_tensor,
                            workspace.last_global_mpjpe,
                        )
                    )
                    last_ee.copy_(torch.where(active_mask, ee_tensor, last_ee))

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
                        temporal_metrics_seen = True
                        vel_tensor, _ = tracking_metrics.compute_mean_velocity_error(
                            sim_body_linvel,
                            ref_body_linvel,
                            sim_root_quat=sim_root_quat,
                            ref_root_quat=ref_root_quat,
                            frame=self._frame,
                        )
                        vel_sum.add_(
                            vel_tensor * active_mask.to(dtype=vel_tensor.dtype)
                        )
                        last_vel.copy_(torch.where(active_mask, vel_tensor, last_vel))

                        if (
                            prev_sim_body_linvel is not None
                            and prev_ref_body_linvel is not None
                            and prev_sim_root_quat is not None
                            and prev_ref_root_quat is not None
                        ):
                            accel_metrics_seen = True
                            accel_tensor, _ = (
                                tracking_metrics.compute_mean_acceleration_error(
                                    sim_body_linvel,
                                    ref_body_linvel,
                                    prev_sim_body_linvel,
                                    prev_ref_body_linvel,
                                    dt=float(env_raw.step_dt),
                                    sim_root_quat=sim_root_quat,
                                    ref_root_quat=ref_root_quat,
                                    prev_sim_root_quat=prev_sim_root_quat,
                                    prev_ref_root_quat=prev_ref_root_quat,
                                    frame=self._frame,
                                )
                            )
                            accel_sum.add_(
                                accel_tensor * active_mask.to(dtype=accel_tensor.dtype)
                            )
                            accel_steps_per_env.add_(
                                active_mask.to(dtype=accel_steps_per_env.dtype)
                            )
                            last_accel.copy_(
                                torch.where(active_mask, accel_tensor, last_accel)
                            )

                        prev_sim_body_linvel = sim_body_linvel.clone()
                        prev_ref_body_linvel = ref_body_linvel.clone()
                        prev_sim_root_quat = sim_root_quat.clone()
                        prev_ref_root_quat = ref_root_quat.clone()
                completed_steps = step_idx + 1
                torch_stream_ctx = (
                    torch.cuda.stream(self._torch_stream)
                    if self._torch_stream is not None
                    else nullcontext()
                )
                wp_stream_ctx = (
                    wp.ScopedStream(self._wp_stream, sync_enter=False, sync_exit=False)
                    if self._wp_stream is not None
                    else nullcontext()
                )
                if self._torch_stream is not None:
                    self._torch_stream.wait_stream(
                        torch.cuda.current_stream(self._device)
                    )
                with wp_stream_ctx, torch_stream_ctx:
                    self._policy_runner.policy_mean_into(
                        obs_policy,
                        policy_actions,
                    )
                    if use_fast_step_with_io:
                        step_with_io(workspace.step_io)
                        torch.logical_or(
                            workspace.next_terminated,
                            workspace.next_truncated,
                            out=workspace.next_dones,
                        )
                        reward_sum.add_(
                            workspace.next_rewards
                            * active_mask.to(dtype=workspace.next_rewards.dtype)
                        )
                        obs_policy = workspace.obs_policy
                        terminated = workspace.next_terminated
                        truncated = workspace.next_truncated
                    else:
                        obs, _reward, terminated, truncated, _ = self._env.step(
                            env_actions
                        )
                        reward_sum.add_(_reward * active_mask.to(dtype=_reward.dtype))
                        copy_agent_observation_into(
                            agent=getattr(self, "_agent", None),
                            observations=obs,
                            destination=workspace.obs_policy,
                        )
                        obs_policy = workspace.obs_policy
                if self._torch_stream is not None:
                    torch.cuda.current_stream(self._device).wait_stream(
                        self._torch_stream
                    )

                if eval_metric_accumulator is not None:
                    eval_metric_accumulator.collect_post_step(active_mask)

                done_mask = torch.logical_or(terminated, truncated)
                terminal_success_horizon = (
                    episode_spec.success == "terminal_success"
                    and completed_steps >= self._play_steps
                )
                if done_mask.any() or terminal_success_horizon:
                    terminated_any = terminated_any or bool(terminated.any().item())
                    truncated_any = truncated_any or bool(truncated.any().item())
                    active_done = torch.logical_and(active_mask, done_mask)
                    natural_completion = (
                        truncated & episode_spec.natural_completion & ~terminated
                        if episode_spec.natural_completion is not None
                        else torch.zeros_like(done_mask)
                    )
                    invalid_now = torch.zeros_like(done_mask)
                    unsuccessful_now = torch.zeros_like(done_mask)

                    if (
                        episode_spec.validity == "full_horizon"
                        and completed_steps < self._play_steps
                    ):
                        newly_invalid = active_done & ~natural_completion
                        valid_mask[newly_invalid] = False
                        invalid_now |= newly_invalid

                    if episode_spec.success == "terminal_success":
                        terminal_success_mask = (
                            episode_spec.terminal_success
                            if episode_spec.terminal_success is not None
                            else terminated
                        )
                        succeeded_now = torch.logical_and(
                            active_done,
                            terminal_success_mask,
                        )
                        if succeeded_now.any():
                            active_mask[succeeded_now] = False
                        terminal_failed = torch.logical_and(
                            active_done,
                            torch.logical_not(terminal_success_mask),
                        )
                        if terminal_failed.any():
                            success_mask[terminal_failed] = False
                            active_mask[terminal_failed] = False
                            unsuccessful_now |= terminal_failed
                        if completed_steps >= self._play_steps:
                            horizon_failed = active_mask.clone()
                            if horizon_failed.any():
                                success_mask[horizon_failed] = False
                                active_mask[horizon_failed] = False
                                unsuccessful_now |= horizon_failed
                    else:
                        if (
                            episode_spec.success == "full_horizon"
                            and completed_steps < self._play_steps
                        ):
                            unsuccessful_now = active_done & ~natural_completion
                            success_mask[unsuccessful_now] = False
                        if (
                            episode_spec.validity == "full_horizon"
                            or episode_spec.success == "full_horizon"
                        ):
                            completed_motion_end = active_done & natural_completion
                            if completed_motion_end.any():
                                active_mask[completed_motion_end] = False
                            stopped_now = active_done & ~natural_completion
                            active_mask[stopped_now] = False
                        elif completed_steps >= self._play_steps:
                            active_mask[done_mask] = False

                    if invalid_now.any() and first_invalid_step == 0:
                        first_invalid_step = completed_steps
                    if unsuccessful_now.any() and first_failure_step == 0:
                        first_failure_step = completed_steps

                if not active_mask.any():
                    break
        finally:
            pass

        result = _summarize_tracking_benchmark_workspace(
            workspace,
            completed_steps=completed_steps,
            first_invalid_step=first_invalid_step,
            first_failure_step=first_failure_step,
            terminated_any=terminated_any,
            truncated_any=truncated_any,
            tracking_metrics_available=self._tracking_metrics_available,
            temporal_metrics_seen=temporal_metrics_seen,
            accel_metrics_seen=accel_metrics_seen,
            episode_spec=episode_spec,
        )
        if eval_metric_accumulator is not None:
            result.update(
                eval_metric_accumulator.summarize(valid_mask=workspace.valid_mask)
            )
        self._append_motion_group_metrics(
            result,
            workspace,
            episode_spec,
            temporal_metrics_seen=temporal_metrics_seen,
            accel_metrics_seen=accel_metrics_seen,
        )
        self._filter_eval_result_metrics(result)
        self._synchronize_pending_work()
        return result


# Compatibility for external callers that imported the pre-rollout name.
TrackingBenchmarkRunner = RolloutBenchmarkRunner
