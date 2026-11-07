"""PPO with conditional policy state and captured auxiliary-loss updates."""

from __future__ import annotations

import copy
import inspect
import logging
import math
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext, suppress
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import warp as wp

from instantmimic.rl.agent_protocol import CheckpointRestoreMetadata
from instantmimic.rl.initial_state import InitialStateSpec
from instantmimic.rl.models.actor_critic import ActorCritic
from instantmimic.rl.normalization import (
    RslRlEmpiricalNormalization,
    RunningStandardScaler,
    build_observation_preprocessor,
    build_reward_preprocessor,
    resolve_observation_preprocessor_name,
)
from instantmimic.rl.optimizers import (
    CapturableMuon,
    HybridMuonAdamW,
    build_optimizer,
    prepare_optimizer_state,
    resolve_optimizer_name,
    validate_graph_optimizer_constraints,
)
from instantmimic.rl.sampling import fill_gaussian_noise_kernel, increment_seed_kernel
from instantmimic.rl.update_compile import full_update_compile_enabled
from instantmimic.utils.network_precision import network_matmul_precision
from instantmimic.utils.policy_mean_contract import register_policy_mean_methods
from instantmimic.utils.startup_profiler import StartupProfiler
from instantmimic.utils.torch_warp_capture import (
    capture_warp_graph,
    warp_stream_from_torch,
)

from .initial_state import InitialStateLearner
from .loss_scaler import CapturableLossScaler
from .losses import ppo_loss_terms

_LOGGER = logging.getLogger(__name__)
_LR_SCHEDULE_CONSTANT_NAMES = ("kl_high", "kl_low", "factor", "min_lr", "max_lr")

_UPDATE_KL_METRIC_INDEX = 4


def _nonfinite_update_metric_items(
    metrics: Mapping[str, float],
) -> list[tuple[str, float]]:
    bad_items: list[tuple[str, float]] = []
    for key, value in metrics.items():
        scalar = float(value)
        if not math.isfinite(scalar):
            bad_items.append((str(key), scalar))
    return bad_items


def _validate_minibatch_partition(
    flat_size: int, mini_batches: int, *, label: str
) -> int:
    if mini_batches <= 0:
        raise ValueError("`mini_batches` must be > 0")
    batch_size, remainder = divmod(flat_size, mini_batches)
    if batch_size == 0:
        raise ValueError("`mini_batches` is too large for rollout size")
    if remainder != 0:
        raise ValueError(
            f"{label} flat size ({flat_size}) must be divisible by `mini_batches` ({mini_batches}); otherwise PPO would drop {remainder} sample(s)."
        )
    return batch_size


_UPDATE_SUMMARY_KEYS = (
    "policy",
    "value",
    "entropy",
    "policy_aux",
    "kl",
    "clip_fraction",
    "lr",
    "std",
    "std_actual_mean",
    "std_actual_min",
    "std_actual_max",
    "raw_log_std_clamp_lower_fraction",
    "raw_log_std_clamp_upper_fraction",
    "raw_log_std_clamp_overshoot_rms",
    "raw_log_std_clamp_overshoot_max",
    "log_std_bound",
)
_UPDATE_CORE_METRIC_COUNT = 6
_UPDATE_STD_METRIC_COUNT = 8
_UPDATE_EXTRA_METRIC_OFFSET = _UPDATE_CORE_METRIC_COUNT + _UPDATE_STD_METRIC_COUNT
_UPDATE_SUMMARY_EXTRA_OFFSET = len(_UPDATE_SUMMARY_KEYS)


def _diagonal_gaussian_kl_mean(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    old_mean = old_mean.float()
    new_mean = new_mean.float()
    old_log_std = old_log_std.float().expand_as(old_mean)
    new_log_std = new_log_std.float().expand_as(new_mean)
    log_std_delta = new_log_std - old_log_std
    old_to_new_var_ratio = torch.exp(2.0 * (old_log_std - new_log_std))
    mean_delta_term = (old_mean - new_mean).square() * torch.exp(-2.0 * new_log_std)
    kl_per_dim = log_std_delta + 0.5 * (old_to_new_var_ratio + mean_delta_term) - 0.5
    return kl_per_dim.sum(dim=-1, keepdim=True).mean()


@wp.kernel(enable_backward=False)
def _compute_gae_kernel(
    rewards: wp.array3d(dtype=wp.float32),
    dones: wp.array3d(dtype=wp.float32),
    values: wp.array3d(dtype=wp.float32),
    returns: wp.array3d(dtype=wp.float32),
    advantages: wp.array3d(dtype=wp.float32),
    last_values: wp.array2d(dtype=wp.float32),
    discount_factor: float,
    lambda_coefficient: float,
    memory_size: int,
):
    env_idx = wp.tid()
    advantage = float(0.0)
    for step in reversed(range(memory_size)):
        if step < memory_size - 1:
            next_values = values[step + 1, env_idx, 0]
        else:
            next_values = last_values[env_idx, 0]
        advantage = (
            rewards[step, env_idx, 0]
            - values[step, env_idx, 0]
            + discount_factor
            * (1.0 - dones[step, env_idx, 0])
            * (next_values + lambda_coefficient * advantage)
        )
        advantages[step, env_idx, 0] = advantage
        returns[step, env_idx, 0] = values[step, env_idx, 0] + advantage


@wp.kernel(enable_backward=False)
def _update_lr_tensor_kernel(
    kl_value: wp.array(dtype=wp.float32),
    lr_value: wp.array(dtype=wp.float32),
    kl_threshold: float,
    lr_factor: float,
    min_lr: float,
    max_lr: float,
):
    kl = kl_value[0]
    lr = lr_value[0]
    if kl > kl_threshold * 2.0:
        lr = wp.max(lr / lr_factor, min_lr)
    elif kl < kl_threshold * 0.5:
        lr = wp.min(lr * lr_factor, max_lr)
    lr_value[0] = lr


class RolloutBuffer:
    """Fixed-size rollout buffer with GAE and sequential mini-batch slicing."""

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        device: torch.device,
        policy_aux_dim: int = 0,
        store_policy_obs: bool = False,
        store_policy_mean: bool = False,
        store_policy_log_std: bool = False,
        store_pre_squash_actions: bool = False,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.policy_aux_dim = int(policy_aux_dim)
        self.obs = torch.zeros(rollout_steps + 1, num_envs, obs_dim, device=device)
        self.policy_obs = (
            torch.zeros(rollout_steps, num_envs, obs_dim, device=device)
            if store_policy_obs
            else None
        )
        self.actions = torch.zeros(rollout_steps, num_envs, act_dim, device=device)
        self.pre_squash_actions = (
            torch.zeros_like(self.actions) if store_pre_squash_actions else None
        )
        self.policy_aux = (
            torch.zeros(rollout_steps, num_envs, self.policy_aux_dim, device=device)
            if self.policy_aux_dim > 0
            else None
        )
        self.log_probs = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.policy_mean = (
            torch.zeros(rollout_steps, num_envs, act_dim, device=device)
            if store_policy_mean
            else None
        )
        self.policy_log_std = (
            torch.zeros(rollout_steps, num_envs, act_dim, device=device)
            if store_policy_log_std
            else None
        )
        self.values = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.rewards = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.dones = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.advantages = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self.returns = torch.zeros(rollout_steps, num_envs, 1, device=device)
        self._use_warp_gae = self.device.type == "cuda"
        self._wp_device = (
            wp.get_device(str(self.device)) if self._use_warp_gae else None
        )
        self._rewards_wp = (
            wp.from_torch(self.rewards, dtype=wp.float32)
            if self._use_warp_gae
            else None
        )
        self._dones_wp = (
            wp.from_torch(self.dones, dtype=wp.float32) if self._use_warp_gae else None
        )
        self._values_wp = (
            wp.from_torch(self.values, dtype=wp.float32) if self._use_warp_gae else None
        )
        self._returns_wp = (
            wp.from_torch(self.returns, dtype=wp.float32)
            if self._use_warp_gae
            else None
        )
        self._advantages_wp = (
            wp.from_torch(self.advantages, dtype=wp.float32)
            if self._use_warp_gae
            else None
        )

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        wp_stream: Any | None = None,
        normalize_advantages: bool = True,
        rewards: torch.Tensor | None = None,
        rewards_wp: Any | None = None,
    ) -> None:
        rewards_source = self.rewards if rewards is None else rewards
        if self._use_warp_gae:
            assert self._wp_device is not None
            assert self._dones_wp is not None
            assert self._values_wp is not None
            assert self._returns_wp is not None
            assert self._advantages_wp is not None
            rewards_source_wp = (
                self._rewards_wp
                if rewards is None
                else rewards_wp
                if rewards_wp is not None
                else wp.from_torch(rewards_source, dtype=wp.float32)
            )
            assert rewards_source_wp is not None
            last_values_wp = wp.from_torch(last_values, dtype=wp.float32)
            target_wp_stream = wp_stream
            if target_wp_stream is None:
                current_torch_stream = torch.cuda.current_stream(self.device)
                target_wp_stream = warp_stream_from_torch(current_torch_stream)
            with wp.ScopedStream(target_wp_stream, sync_enter=False, sync_exit=False):
                wp.launch(
                    _compute_gae_kernel,
                    dim=self.num_envs,
                    inputs=[
                        rewards_source_wp,
                        self._dones_wp,
                        self._values_wp,
                        self._returns_wp,
                        self._advantages_wp,
                        last_values_wp,
                        float(gamma),
                        float(lam),
                        self.rollout_steps,
                    ],
                    device=self._wp_device,
                )
        else:
            advantage = torch.zeros_like(last_values)
            for i in reversed(range(self.rollout_steps)):
                next_values = (
                    self.values[i + 1] if i < self.rollout_steps - 1 else last_values
                )
                advantage = (
                    rewards_source[i]
                    - self.values[i]
                    + gamma * (1.0 - self.dones[i]) * (next_values + lam * advantage)
                )
                self.advantages[i].copy_(advantage)
            self.returns.copy_(self.advantages + self.values)
        if normalize_advantages:
            self.advantages.sub_(self.advantages.mean()).div_(
                self.advantages.std(correction=1 if self.advantages.numel() > 1 else 0)
                + 1e-08
            )

    def get_batches(self, mini_batches: int) -> Generator[slice, None, None]:
        flat_size = self.rollout_steps * self.num_envs
        batch_size = _validate_minibatch_partition(
            flat_size, mini_batches, label="Rollout buffer"
        )
        for i in range(mini_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            yield slice(start, end)


class PPOAgent:
    """Project PPO agent."""

    def _requires_update_aux_evaluate(self) -> bool:
        return (
            float(self.network_kwargs.get("activation_regularization_weight", 0.0))
            > 0.0
        )

    def _extra_update_metric_count(self) -> int:
        return 0

    def _extra_update_summary_keys(self) -> tuple[str, ...]:
        return ()

    def _update_summary_keys(self):
        return _UPDATE_SUMMARY_KEYS + self._extra_update_summary_keys()

    def _compute_extra_update_metrics(
        self,
        *,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        policy_aux_loss: torch.Tensor,
        transition_aux_loss: torch.Tensor,
        batch_next_obs: torch.Tensor | None,
        batch_dones: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        del (
            batch_obs,
            batch_actions,
            policy_aux_loss,
            transition_aux_loss,
            batch_next_obs,
            batch_dones,
        )
        return ()

    def _write_extra_update_summary_from_metrics(
        self, mean_metrics: torch.Tensor
    ) -> None:
        del mean_metrics

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_envs: int,
        device: torch.device | str,
        *,
        rollouts: int,
        learning_epochs: int,
        mini_batches: int,
        shuffle_mode: str = "per_epoch",
        discount_factor: float,
        lam: float,
        learning_rate: float,
        grad_norm_clip: float,
        ratio_clip: float,
        value_clip: float,
        clip_predicted_values: bool,
        entropy_loss_scale: float,
        value_loss_scale: float,
        value_loss_clip_mode: str = "clamp",
        kl_threshold: float,
        kl_estimator: str = "sampled",
        layers: Sequence[int] = (512, 512, 256),
        activation: str = "elu",
        actor_output_initialization_std: float | None = None,
        initial_log_std: float = 0.0,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        fixed_log_std: bool = False,
        fixed_sigma: bool = True,
        std_parameterization: str = "log",
        optimizer_name: str = "adam",
        optimizer_kwargs: dict[str, Any] | None = None,
        state_preprocessor: str | None = "RunningStandardScaler",
        state_preprocessor_kwargs: dict | None = None,
        value_preprocessor: str | None = "RunningStandardScaler",
        value_preprocessor_kwargs: dict | None = None,
        reward_preprocessor: str | None = None,
        reward_preprocessor_kwargs: dict | None = None,
        min_learning_rate: float = 1e-06,
        max_learning_rate: float = 0.01,
        lr_factor: float = 1.5,
        kl_scheduler_update_mode: str = "per_epoch",
        compile: bool = False,
        compile_mode: str = "max-autotune-no-cudagraphs",
        use_update_graph: bool = False,
        use_shuffle_graph: bool = False,
        update_torch_stream: torch.cuda.Stream | None = None,
        mixed_precision: bool | None = None,
        amp_dtype: str = "bfloat16",
        precision: str | None = None,
        loss_scale: float = 128.0,
        detailed_nvtx: bool = False,
        fail_on_nonfinite_update: bool = False,
        inference_only: bool = False,
        squash_actions: bool = False,
        clamp_mean: bool = False,
        clamp_mean_min: float = -1.0,
        clamp_mean_max: float = 1.0,
        log_std_bound_mode: str = "clamp",
        log_std_bound_loss_weight: float = 0.0,
        log_std_bound_loss_margin: float = 0.0,
        actor_obs_dim: int | None = None,
        critic_obs_dim: int | None = None,
        separate_grad_norm_clip: bool = False,
        initial_state_spec: InitialStateSpec | None = None,
        initial_state_config: Mapping[str, Any] | None = None,
        network_cls: type[nn.Module] = ActorCritic,
        network_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_envs = num_envs
        self.inference_only = bool(inference_only)
        self.obs_main_dim = self.obs_dim
        if self.obs_main_dim <= 0:
            raise ValueError("PPO observations must have a positive width.")
        if (actor_obs_dim is None) != (critic_obs_dim is None):
            raise ValueError(
                "actor_obs_dim and critic_obs_dim must be provided together"
            )
        self.asymmetric_observations = actor_obs_dim is not None
        self.actor_obs_dim = (
            self.obs_dim if actor_obs_dim is None else int(actor_obs_dim)
        )
        self.critic_obs_dim = 0 if critic_obs_dim is None else int(critic_obs_dim)
        if self.asymmetric_observations:
            if (
                self.actor_obs_dim <= 0
                or self.critic_obs_dim <= 0
                or self.actor_obs_dim + self.critic_obs_dim != self.obs_dim
            ):
                raise ValueError(
                    f"Asymmetric observation dimensions must be positive and sum to obs_dim={self.obs_dim}; got actor={self.actor_obs_dim}, critic={self.critic_obs_dim}"
                )
        self.network_kwargs = dict(network_kwargs or {})
        if precision is not None and precision not in {
            "float32",
            "bfloat16",
            "float16",
        }:
            raise ValueError("precision must be float32, bfloat16, or float16")
        self.precision = precision
        self._amp_dtype = (
            getattr(torch, precision)
            if precision is not None
            else torch.float32
            if self.inference_only
            else self._resolve_amp_dtype(amp_dtype)
        )
        if mixed_precision is not None:
            _LOGGER.warning(
                "PPO mixed_precision is deprecated. Omitting precision preserves legacy update-only autocast and the original rollout/playback policy precision. Set precision=float32, bfloat16, or float16 to opt into shared rollout/update precision; explicit precision takes precedence over mixed_precision."
            )
        self.loss_scaler = None
        if precision == "float16" and (not self.inference_only):
            self.loss_scaler = CapturableLossScaler(
                device=self.device, initial_scale=loss_scale
            )
        self.separate_grad_norm_clip = bool(separate_grad_norm_clip)
        has_separate_actor_critic = self.asymmetric_observations or bool(
            self.network_kwargs.get("separate", False)
        )
        if self.separate_grad_norm_clip and (not has_separate_actor_critic):
            raise ValueError(
                "separate_grad_norm_clip requires separate actor/critic models"
            )
        self.network_cls = network_cls
        init_profiler = StartupProfiler(
            range_factory=lambda name: torch.cuda.nvtx.range(f"ppo_agent_init:{name}")
            if self.device.type == "cuda"
            else nullcontext()
        )

        def _construct_policy_network() -> None:
            network_init_kwargs: dict[str, Any] = {
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "layers": layers,
                "activation": activation,
                "initial_log_std": initial_log_std,
                "min_log_std": min_log_std,
                "max_log_std": max_log_std,
                "squash_actions": squash_actions,
                "clamp_mean": clamp_mean,
                **self.network_kwargs,
            }
            if self.asymmetric_observations:
                network_init_kwargs.update(
                    {
                        "actor_obs_dim": self.actor_obs_dim,
                        "critic_obs_dim": self.critic_obs_dim,
                        "separate": True,
                    }
                )
            init_signature = None
            with suppress(TypeError, ValueError):
                init_signature = inspect.signature(self.network_cls.__init__)
            accepts_var_kwargs = bool(
                init_signature is not None
                and any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in init_signature.parameters.values()
                )
            )
            if init_signature is not None and (
                "log_std_bound_mode" in init_signature.parameters or accepts_var_kwargs
            ):
                network_init_kwargs["log_std_bound_mode"] = log_std_bound_mode
            if init_signature is not None and (
                "log_std_bound_loss_weight" in init_signature.parameters
                or accepts_var_kwargs
            ):
                network_init_kwargs["log_std_bound_loss_weight"] = (
                    log_std_bound_loss_weight
                )
                network_init_kwargs["log_std_bound_loss_margin"] = (
                    log_std_bound_loss_margin
                )
            if init_signature is not None and (
                "clamp_mean_min" in init_signature.parameters or accepts_var_kwargs
            ):
                network_init_kwargs["clamp_mean_min"] = clamp_mean_min
            if init_signature is not None and (
                "clamp_mean_max" in init_signature.parameters or accepts_var_kwargs
            ):
                network_init_kwargs["clamp_mean_max"] = clamp_mean_max
            if init_signature is not None and (
                "fixed_log_std" in init_signature.parameters or accepts_var_kwargs
            ):
                network_init_kwargs["fixed_log_std"] = fixed_log_std
            if init_signature is not None and (
                "fixed_sigma" in init_signature.parameters or accepts_var_kwargs
            ):
                network_init_kwargs["fixed_sigma"] = fixed_sigma
            if init_signature is not None and (
                "std_parameterization" in init_signature.parameters
                or accepts_var_kwargs
            ):
                network_init_kwargs["std_parameterization"] = std_parameterization
            elif str(std_parameterization).strip().lower() != "log":
                raise ValueError(
                    f"{self.network_cls.__name__} does not support std_parameterization={std_parameterization!r}"
                )
            with init_profiler.phase("network_construct"):
                self.network = self.network_cls(**network_init_kwargs)
                if actor_output_initialization_std is not None:
                    initialize_actor_output = getattr(
                        self.network, "initialize_actor_output", None
                    )
                    if not callable(initialize_actor_output):
                        raise ValueError(
                            f"{self.network_cls.__name__} does not support actor output initialization"
                        )
                    initialize_actor_output(actor_output_initialization_std)
            with init_profiler.phase("network_to_device"):
                self.network = self.network.to(self.device)
                if self.precision is not None:
                    set_precision = getattr(self.network, "set_policy_precision", None)
                    if callable(set_precision):
                        set_precision(self.precision)
                    elif self.precision != "float32":
                        raise ValueError(
                            "Reduced-precision PPO networks must expose set_policy_precision"
                        )

        def _construct_initial_state_learner() -> None:
            if initial_state_spec is None:
                if initial_state_config is not None:
                    raise ValueError(
                        "agent.initial_state requires an environment initial-state spec"
                    )
                self.initial_state_learner = None
                return
            self.initial_state_learner = InitialStateLearner(
                spec=initial_state_spec,
                num_envs=num_envs,
                rollout_steps=rollouts,
                device=self.device,
                config=initial_state_config or {},
                inference_only=self.inference_only,
            )

        if self.inference_only:
            self._use_update_graph = False
            self._use_shuffle_graph = False
            self._use_mixed_precision = False
            self._detailed_nvtx = False
            self._optimizer_graph_state_initialized = False
            _construct_policy_network()
            _construct_initial_state_learner()
            self._initialize_inference_runtime(
                state_preprocessor=state_preprocessor,
                state_preprocessor_kwargs=state_preprocessor_kwargs,
                value_preprocessor=value_preprocessor,
                value_preprocessor_kwargs=value_preprocessor_kwargs,
            )
            init_profiler.log_summary(
                logger=logging.getLogger(), prefix="PPO_AGENT_INFERENCE_INIT_PROFILE"
            )
            return
        self._kl_estimator = str(kl_estimator).strip().lower()
        if self._kl_estimator not in {"sampled", "analytic"}:
            raise ValueError(
                f"Unsupported agent.kl_estimator: {kl_estimator!r}. Use 'sampled' or 'analytic'."
            )
        self._uses_analytic_kl = self._kl_estimator == "analytic"
        self._use_update_graph = bool(
            not self.inference_only
            and use_update_graph
            and (self.device.type == "cuda")
        )
        self._optimizer_graph_state_initialized = False
        if (
            not self.inference_only
            and use_update_graph
            and (self.device.type != "cuda")
        ):
            _LOGGER.warning(
                "use_update_graph=true requested, but CUDA is unavailable; using eager update."
            )
        if (
            not self.inference_only
            and update_torch_stream is not None
            and (self.device.type != "cuda")
        ):
            _LOGGER.warning(
                "update_torch_stream was provided without CUDA; ignoring the external stream."
            )
            update_torch_stream = None
        self._detailed_nvtx = bool(detailed_nvtx)
        self.fail_on_nonfinite_update = bool(fail_on_nonfinite_update)
        self._use_mixed_precision = False
        if self.precision is None and mixed_precision:
            self._warn_if_fp16_without_loss_scaling(mixed_precision=True)
            if self.device.type != "cuda":
                _LOGGER.warning(
                    "mixed_precision=true requested, but CUDA is unavailable; using float32 update."
                )
            elif self._amp_dtype == torch.bfloat16 and (
                not torch.cuda.is_bf16_supported()
            ):
                _LOGGER.warning(
                    "mixed_precision=true with amp_dtype=bfloat16 requested, but bfloat16 is unsupported on this CUDA device; using float32 update."
                )
            else:
                self._use_mixed_precision = True
        self.optimizer_name = self._resolve_optimizer_name(optimizer_name)
        self.rollouts = rollouts
        self.learning_epochs = learning_epochs
        self.mini_batches = mini_batches
        self.shuffle_mode = self._resolve_shuffle_mode(shuffle_mode)
        self.shuffle_minibatches = self.shuffle_mode != "off"
        self._use_shuffle_graph = bool(
            not self.inference_only
            and use_shuffle_graph
            and self._use_update_graph
            and self.shuffle_minibatches
        )
        if (
            not self.inference_only
            and use_shuffle_graph
            and (not self._use_update_graph)
        ):
            _LOGGER.warning(
                "use_shuffle_graph=true requested without use_update_graph=true; using eager minibatch permutation refresh."
            )
        elif (
            not self.inference_only
            and use_shuffle_graph
            and (not self.shuffle_minibatches)
        ):
            _LOGGER.warning(
                "use_shuffle_graph=true requested with shuffle_mode=off; shuffle graph will remain disabled."
            )
        self.discount_factor = discount_factor
        self.lam = lam
        self.grad_norm_clip = grad_norm_clip
        self.ratio_clip = ratio_clip
        self.value_clip = value_clip
        self.clip_predicted_values = clip_predicted_values
        self.value_loss_clip_mode = str(value_loss_clip_mode).strip().lower()
        if self.value_loss_clip_mode not in {"clamp", "max"}:
            raise ValueError(
                f"Unsupported value_loss_clip_mode: {value_loss_clip_mode!r}. Use 'clamp' or 'max'."
            )
        self.entropy_loss_scale = entropy_loss_scale
        self.value_loss_scale = value_loss_scale
        self.log_std_bound_loss_weight = float(log_std_bound_loss_weight)
        self.log_std_bound_loss_margin = float(log_std_bound_loss_margin)
        if (
            not math.isfinite(self.log_std_bound_loss_weight)
            or self.log_std_bound_loss_weight < 0.0
        ):
            raise ValueError(
                "log_std_bound_loss_weight must be a finite non-negative value."
            )
        if (
            not math.isfinite(self.log_std_bound_loss_margin)
            or self.log_std_bound_loss_margin < 0.0
        ):
            raise ValueError(
                "log_std_bound_loss_margin must be a finite non-negative value."
            )
        self.kl_threshold = kl_threshold
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.lr_factor = lr_factor
        self.kl_scheduler_update_mode = str(kl_scheduler_update_mode).strip().lower()
        if self.kl_scheduler_update_mode not in {
            "per_update",
            "per_epoch",
            "per_minibatch",
        }:
            raise ValueError(
                f"Unsupported kl_scheduler_update_mode: {kl_scheduler_update_mode!r}. Use 'per_update', 'per_epoch', or 'per_minibatch'."
            )
        self._update_uses_aux_evaluate = bool(self._requires_update_aux_evaluate())
        _construct_policy_network()
        _construct_initial_state_learner()
        transition_aux_loss_fn = getattr(
            self.network, "transition_aux_regularization_loss", None
        )
        self._use_transition_aux_regularization = bool(
            callable(transition_aux_loss_fn)
            and float(
                getattr(self.network, "transition_aux_regularization_weight", 0.0)
            )
            > 0.0
        )
        with init_profiler.phase("optimizer_init"):
            optimizer_kwargs = dict(optimizer_kwargs or {})
            self._validate_graph_optimizer_constraints(optimizer_kwargs)
            self.optimizer = self._build_optimizer(
                learning_rate=learning_rate, optimizer_kwargs=optimizer_kwargs
            )
            self._update_wp_device = (
                wp.get_device(str(self.device)) if self._use_update_graph else None
            )
            self._optimizer_lr_wp: list[Any | None] = []
            self._lr_kl_tensor: torch.Tensor | None = None
            self._lr_kl_wp = None
            self._lr_schedule_constants: dict[
                torch.dtype,
                dict[str, torch.Tensor],
            ] = {}
            self._refresh_optimizer_lr_wrappers()
        with init_profiler.phase("optimizer_graph_state_init"):
            if self._use_update_graph:
                self._initialize_optimizer_state_for_graph()
        if self.loss_scaler is not None:
            self.loss_scaler.prepare_optimizer(self.optimizer)
        self._evaluate_with_std_fn = self.network.evaluate_with_std_stats
        self._evaluate_with_aux_std_fn = getattr(
            self.network, "evaluate_with_aux_losses_and_std_stats", None
        )
        self._evaluate_into_fn = self.network.evaluate_into
        self._evaluate_with_std_into_fn = self.network.evaluate_with_std_stats_into
        self._evaluate_with_aux_into_fn = getattr(
            self.network, "evaluate_with_aux_into", None
        )
        self._evaluate_with_aux_std_into_fn = getattr(
            self.network, "evaluate_with_aux_and_std_stats_into", None
        )
        self._act_into_fn = self.network.act_into
        self._uses_pre_squash_action_replay = bool(
            getattr(self.network, "squash_actions", False)
        )
        self._policy_aux_dim = int(getattr(self.network, "_policy_aux_dim", 0))
        self._analytic_kl_store_policy_log_std = bool(
            self._uses_analytic_kl
            and getattr(self.network, "_uses_state_dependent_sigma", False)
        )
        self._analytic_old_log_std_snapshot = (
            torch.empty(self.act_dim, device=self.device, dtype=torch.float32)
            if self._uses_analytic_kl and (not self._analytic_kl_store_policy_log_std)
            else None
        )
        self._snapshot_analytic_kl_log_std()
        self._flat_update_size = self.rollouts * self.num_envs
        self._update_batch_size = _validate_minibatch_partition(
            self._flat_update_size, self.mini_batches, label="PPO update"
        )
        self._update_eval_log_probs = torch.empty(
            self._update_batch_size, 1, device=self.device, dtype=torch.float32
        )
        self._update_eval_values = torch.empty(
            self._update_batch_size, 1, device=self.device, dtype=torch.float32
        )
        self._update_eval_entropy = torch.empty(
            self._update_batch_size, 1, device=self.device, dtype=torch.float32
        )
        self._update_eval_std_mean = torch.empty(
            (), device=self.device, dtype=torch.float32
        )
        self._update_eval_std_min = torch.empty(
            (), device=self.device, dtype=torch.float32
        )
        self._update_eval_std_max = torch.empty(
            (), device=self.device, dtype=torch.float32
        )
        self._update_eval_raw_log_std_bound_stats = torch.empty(
            5, device=self.device, dtype=torch.float32
        )
        self._update_eval_policy_mean = (
            torch.empty(
                self._update_batch_size,
                self.act_dim,
                device=self.device,
                dtype=torch.float32,
            )
            if self._uses_analytic_kl
            else None
        )
        self._update_eval_policy_log_std = (
            torch.empty_like(self._update_eval_policy_mean)
            if self._update_eval_policy_mean is not None
            else None
        )
        self._update_eval_policy_aux_loss = torch.empty(
            (), device=self.device, dtype=torch.float32
        )
        self._use_plain_update_minibatch_fast_path = bool(
            getattr(self.network, "_use_standard_policy_fast_path", False)
            and (not self._update_uses_aux_evaluate)
        )
        self._compile_enabled = bool(compile and hasattr(torch, "compile"))
        self._compile_mode = compile_mode
        self._compile_full_update = full_update_compile_enabled(
            compile_enabled=self._compile_enabled,
            update_graph_enabled=self._use_update_graph,
        )
        self._compiled_update_uses_return_path = False
        self._compiled_update_capture_prewarmed = False
        self._compiled_model_functions_warmed = False
        self._warmup_compiled_act_into = bool(self._compile_enabled)
        self._cuda_libraries_graph_capture_ready = False
        with init_profiler.phase("compile_model_wrappers"):
            if compile and self._use_update_graph:
                _LOGGER.info(
                    "use_update_graph=true with compile=true: enabling torch.compile alongside captured PPO update graphs."
                )
            if self._compile_enabled:
                self._evaluate_with_std_fn = torch.compile(
                    self.network.evaluate_with_std_stats,
                    mode=compile_mode,
                    dynamic=False,
                )
                self._compiled_update_uses_return_path = True
                if self._update_uses_aux_evaluate:
                    self._evaluate_with_aux_std_fn = torch.compile(
                        getattr(
                            self.network, "evaluate_with_aux_losses_and_std_stats", None
                        ),
                        mode=compile_mode,
                        dynamic=False,
                    )
                self._act_into_fn = torch.compile(
                    self.network.act_into, mode=compile_mode, dynamic=False
                )
        if self._compile_enabled:
            with init_profiler.phase("warmup_model_compile"):
                # act_into warmup needs the rollout buffer and policy workspace,
                # which are allocated below; warm it with real arguments there.
                self._warmup_compiled_functions(warmup_act_into=False)
                self._compiled_model_functions_warmed = True
        with init_profiler.phase("preprocessor_init"):
            state_preprocessor_kwargs = state_preprocessor_kwargs or {}
            value_preprocessor_kwargs = value_preprocessor_kwargs or {}
            reward_preprocessor_kwargs = reward_preprocessor_kwargs or {}
            self._validate_preprocessor_constraints(
                state_preprocessor=state_preprocessor,
                state_preprocessor_kwargs=state_preprocessor_kwargs,
                value_preprocessor=value_preprocessor,
                value_preprocessor_kwargs=value_preprocessor_kwargs,
            )
            self.obs_scaler = self._build_preprocessor(
                state_preprocessor,
                self.obs_dim,
                self._observation_preprocessor_kwargs(
                    state_preprocessor, state_preprocessor_kwargs
                ),
            ).to(self.device)
            self._observation_normalization_contract = self.obs_scaler.contract
            self.value_scaler = self._build_preprocessor(
                value_preprocessor, 1, value_preprocessor_kwargs
            ).to(self.device)
            self._value_head = self._resolve_popart_value_head()
            self.reward_preprocessor = build_reward_preprocessor(
                reward_preprocessor=reward_preprocessor,
                reward_preprocessor_kwargs=reward_preprocessor_kwargs,
                num_envs=self.num_envs,
                gamma=self.discount_factor,
                device=self.device,
            )
        with init_profiler.phase("compile_update_wrapper_setup"):
            self._obs_scale_eval_into_fn = self._obs_scale_eval_into
            self._obs_scale_update_into_fn = self._obs_scale_update_into
            self._prepare_update_views_fn = self._prepare_update_views
            self._prepare_update_views_post_bootstrap_fn = (
                self._prepare_update_views_post_bootstrap_compiled
            )
            self._run_update_minibatch_fn = self._run_update_minibatch
            self._evaluate_old_policy_minibatch_fn = self._evaluate_old_policy_minibatch
            self._compiled_update_wrappers_ready = False
            if self._compile_full_update:
                self._compile_update_wrappers_if_needed()
        with init_profiler.phase("rollout_buffer_alloc"):
            self.buffer = RolloutBuffer(
                rollout_steps=rollouts,
                num_envs=num_envs,
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=self.device,
                policy_aux_dim=self._policy_aux_dim,
                store_policy_obs=self._observation_normalization_contract.requires_policy_input_snapshot,
                store_policy_mean=self._uses_analytic_kl,
                store_policy_log_std=self._analytic_kl_store_policy_log_std,
                store_pre_squash_actions=self._uses_pre_squash_action_replay,
            )
            self._update_rewards = torch.empty_like(self.buffer.rewards)
            self._update_rewards.copy_(self.buffer.rewards)
            self._timeout_bootstrap_mask = torch.zeros_like(self.buffer.rewards)
            self._timeout_bootstrap_values = torch.zeros_like(self.buffer.rewards)
            self._timeout_bootstrap_dropped = torch.zeros_like(
                self.buffer.rewards, dtype=torch.bool
            )
            self.store_time_limit_bootstrap_mask = self._store_time_limit_bootstrap_mask
            self.store_time_limit_bootstrap_values = (
                self._store_time_limit_bootstrap_values
            )
            self._raw_update_returns = torch.empty_like(self.buffer.returns)
            self._value_popart_old_mean = torch.empty_like(
                self.value_scaler.running_mean, dtype=torch.float32
            )
            self._value_popart_old_scale = torch.empty_like(
                self.value_scaler.running_mean, dtype=torch.float32
            )
            self._update_rewards_wp = (
                wp.from_torch(self._update_rewards, dtype=wp.float32)
                if self.device.type == "cuda"
                else None
            )
        with init_profiler.phase("shuffle_workspace_alloc"):
            self._shuffle_permutation_count = (
                self.learning_epochs
                if self.shuffle_mode == "per_epoch"
                else 1
                if self.shuffle_mode == "per_update"
                else 0
            )
            base_epoch_permutation = (
                torch.arange(
                    self._flat_update_size, device=self.device, dtype=torch.int32
                )
                if self.shuffle_minibatches
                else None
            )
            self._epoch_permutations = (
                [
                    base_epoch_permutation.clone()
                    for _ in range(self._shuffle_permutation_count)
                ]
                if self.shuffle_minibatches
                else None
            )
            self._epoch_shuffled_obs = (
                torch.empty(self._flat_update_size, self.obs_dim, device=self.device)
                if self.shuffle_minibatches
                else None
            )
            self._epoch_shuffled_actions = (
                torch.empty(self._flat_update_size, self.act_dim, device=self.device)
                if self.shuffle_minibatches
                else None
            )
            self._epoch_shuffled_policy_aux = (
                torch.empty(
                    self._flat_update_size, self._policy_aux_dim, device=self.device
                )
                if self.shuffle_minibatches and self._policy_aux_dim > 0
                else None
            )
            self._epoch_shuffled_log_probs = (
                torch.empty(self._flat_update_size, 1, device=self.device)
                if self.shuffle_minibatches
                else None
            )
            self._epoch_shuffled_policy_mean = (
                torch.empty(self._flat_update_size, self.act_dim, device=self.device)
                if self.shuffle_minibatches and self._uses_analytic_kl
                else None
            )
            self._epoch_shuffled_policy_log_std = (
                torch.empty(self._flat_update_size, self.act_dim, device=self.device)
                if self.shuffle_minibatches and self._analytic_kl_store_policy_log_std
                else None
            )
            self._epoch_shuffled_values = (
                torch.empty(self._flat_update_size, 1, device=self.device)
                if self.shuffle_minibatches
                else None
            )
            self._epoch_shuffled_returns = (
                torch.empty(self._flat_update_size, 1, device=self.device)
                if self.shuffle_minibatches
                else None
            )
            self._epoch_shuffled_advantages = (
                torch.empty(self._flat_update_size, 1, device=self.device)
                if self.shuffle_minibatches
                else None
            )
            self._epoch_shuffled_next_obs = (
                torch.empty(self._flat_update_size, self.obs_dim, device=self.device)
                if self.shuffle_minibatches and self._use_transition_aux_regularization
                else None
            )
            self._epoch_shuffled_dones = (
                torch.empty(self._flat_update_size, 1, device=self.device)
                if self.shuffle_minibatches and self._use_transition_aux_regularization
                else None
            )
        with init_profiler.phase("update_graph_workspace_alloc"):
            self._update_graph: Any | None = None
            self._detailed_update_prepare_graph: Any | None = None
            self._detailed_update_epoch_prepare_graphs: list[Any | None] | None = (
                [None for _ in range(self.learning_epochs)]
                if self._use_update_graph
                and self._detailed_nvtx
                and (self.shuffle_mode == "per_epoch")
                else None
            )
            self._detailed_update_minibatch_graphs: list[list[Any | None]] | None = (
                [
                    [None for _ in range(self.mini_batches)]
                    for _ in range(self.learning_epochs)
                ]
                if self._use_update_graph and self._detailed_nvtx
                else None
            )
            self._update_torch_stream = (
                update_torch_stream
                if self._use_update_graph and update_torch_stream is not None
                else torch.cuda.Stream(device=self.device)
                if self._use_update_graph
                else None
            )
            self._shuffle_graph: torch.cuda.CUDAGraph | None = None
            self._shuffle_generator = (
                torch.Generator(device=self.device) if self._use_shuffle_graph else None
            )
            self._update_start_event = (
                torch.cuda.Event(enable_timing=True) if self._use_update_graph else None
            )
            self._update_end_event = (
                torch.cuda.Event(enable_timing=True) if self._use_update_graph else None
            )
            self._update_timing_pending = False
            self._update_mode = "full"
            self._value_train_params: set[nn.Parameter] = set()
            self._all_trainable_params: tuple[nn.Parameter, ...] = tuple(
                param for param in self.network.parameters() if param.requires_grad
            )
            self._policy_log_std_frozen = False
            self._update_wp_stream = (
                warp_stream_from_torch(self._update_torch_stream)
                if self._use_update_graph and self._update_torch_stream is not None
                else None
            )
            self._update_scaled_obs = (
                torch.empty(
                    self.rollouts * self.num_envs, self.obs_dim, device=self.device
                )
                if self._use_update_graph
                else None
            )
            self._update_scaled_next_obs = (
                torch.empty(
                    self.rollouts * self.num_envs, self.obs_dim, device=self.device
                )
                if self._use_update_graph and self._use_transition_aux_regularization
                else None
            )
            subclass_metric_count = int(self._extra_update_metric_count())
            if subclass_metric_count < 0:
                raise ValueError("_extra_update_metric_count() must be non-negative.")
            extra_update_metric_count = subclass_metric_count
            self._update_metrics = torch.zeros(
                self.learning_epochs * self.mini_batches,
                _UPDATE_EXTRA_METRIC_OFFSET + extra_update_metric_count,
                device=self.device,
            )
            update_summary_key_count = len(self._update_summary_keys())
            self._update_summary = torch.empty(
                update_summary_key_count, device=self.device, dtype=torch.float32
            )
            self._update_summary_host = torch.empty(
                update_summary_key_count,
                device="cpu",
                dtype=torch.float32,
                pin_memory=self.device.type == "cuda",
            )
        with init_profiler.phase("policy_workspace_alloc"):
            self._update_last_values = torch.empty(num_envs, 1, device=self.device)
            self._policy_scaled_obs = torch.empty(num_envs, obs_dim, device=self.device)
            self._bootstrap_actions = torch.empty(num_envs, act_dim, device=self.device)
            self._bootstrap_log_probs = torch.empty(num_envs, 1, device=self.device)
            self._bootstrap_eps = torch.zeros(
                num_envs, self.act_dim, device=self.device
            )
            self._bootstrap_policy_aux = (
                torch.zeros(num_envs, self._policy_aux_dim, device=self.device)
                if self._policy_aux_dim > 0
                else None
            )
            self._policy_eps = torch.empty(num_envs, self.act_dim, device=self.device)
            self._policy_aux_eps = (
                torch.empty(num_envs, self._policy_aux_dim, device=self.device)
                if self._policy_aux_dim > 0
                else None
            )
        with init_profiler.phase("warp_noise_workspace_init"):
            self._use_warp_noise = self.device.type == "cuda"
            if self._use_warp_noise:
                self._wp_device = wp.get_device(str(self.device))
                self._noise_seed = torch.randint(
                    low=0,
                    high=2**31 - 1,
                    size=(1,),
                    device=self.device,
                    dtype=torch.int32,
                )
                self._noise_seed_wp = wp.from_torch(self._noise_seed, dtype=wp.int32)
                self._policy_eps_wp = wp.from_torch(self._policy_eps, dtype=wp.float32)
                self._policy_aux_eps_wp = (
                    wp.from_torch(self._policy_aux_eps, dtype=wp.float32)
                    if self._policy_aux_eps is not None
                    else None
                )
            else:
                self._policy_aux_eps_wp = None
        if self._compile_enabled:
            with init_profiler.phase("warmup_compiled_act_into"):
                self._warmup_compiled_act_into_signatures()
        if self._compile_enabled and self._compile_full_update:
            with init_profiler.phase("compile_update_obs_scalers"):
                self._warmup_compiled_obs_scalers(warmup_update=self._use_update_graph)
        if self._compile_enabled and self._compile_full_update:
            with init_profiler.phase("compile_update_prepare_views"):
                self._warmup_compiled_prepare_update_views()
            with init_profiler.phase("compile_update_minibatch_warmup"):
                self._warmup_compiled_update_minibatch(max_minibatches=1)
            self._compiled_update_capture_prewarmed = True
        if self._shuffle_generator is not None:
            with init_profiler.phase("shuffle_graph_seed"):
                shuffle_seed = int(
                    torch.empty((), dtype=torch.int64).random_(0, 2**31 - 1).item()
                )
                self._shuffle_generator.manual_seed(shuffle_seed)
            with init_profiler.phase("shuffle_graph_capture"):
                self._capture_shuffle_graph()
        init_profiler.log_summary(
            logger=logging.getLogger(), prefix="PPO_AGENT_INIT_PROFILE"
        )

    def _initialize_inference_runtime(
        self,
        *,
        state_preprocessor: str | None,
        state_preprocessor_kwargs: dict | None,
        value_preprocessor: str | None,
        value_preprocessor_kwargs: dict | None,
    ) -> None:
        self.optimizer = None
        self.buffer = None
        self.reward_preprocessor = None
        self._update_wp_device = None
        self._optimizer_lr_wp = []
        self._lr_kl_tensor = None
        self._lr_kl_wp = None
        self._lr_schedule_constants = {}
        self._update_graph = None
        self._detailed_update_prepare_graph = None
        self._detailed_update_epoch_prepare_graphs = None
        self._detailed_update_minibatch_graphs = None
        self._update_torch_stream = None
        self._update_wp_stream = None
        self._shuffle_graph = None
        self._shuffle_generator = None
        self._flat_update_size = 0
        self._update_batch_size = 0
        self._compile_enabled = False
        self._compile_full_update = False
        self._compiled_update_capture_prewarmed = False
        self._compiled_model_functions_warmed = False
        self._compiled_update_wrappers_ready = False
        self._cuda_libraries_graph_capture_ready = False
        self._act_into_fn = self.network.act_into
        self._uses_pre_squash_action_replay = bool(
            getattr(self.network, "squash_actions", False)
        )
        self._policy_aux_dim = int(getattr(self.network, "_policy_aux_dim", 0))
        state_preprocessor_kwargs = state_preprocessor_kwargs or {}
        value_preprocessor_kwargs = value_preprocessor_kwargs or {}
        self.obs_scaler = self._build_preprocessor(
            state_preprocessor,
            self.obs_dim,
            self._observation_preprocessor_kwargs(
                state_preprocessor, state_preprocessor_kwargs
            ),
        ).to(self.device)
        self.value_scaler = self._build_preprocessor(
            value_preprocessor, 1, value_preprocessor_kwargs
        ).to(self.device)
        self._value_head = self._resolve_popart_value_head()
        self._obs_scale_eval_into_fn = self._obs_scale_eval_into
        self._policy_scaled_obs = torch.empty(
            self.num_envs, self.obs_dim, device=self.device
        )
        self._policy_eps = torch.empty(self.num_envs, self.act_dim, device=self.device)
        self._policy_aux_eps = (
            torch.empty(self.num_envs, self._policy_aux_dim, device=self.device)
            if self._policy_aux_dim > 0
            else None
        )
        self._use_warp_noise = self.device.type == "cuda"
        if self._use_warp_noise:
            self._wp_device = wp.get_device(str(self.device))
            self._noise_seed = torch.randint(
                low=0, high=2**31 - 1, size=(1,), device=self.device, dtype=torch.int32
            )
            self._noise_seed_wp = wp.from_torch(self._noise_seed, dtype=wp.int32)
            self._policy_eps_wp = wp.from_torch(self._policy_eps, dtype=wp.float32)
            self._policy_aux_eps_wp = (
                wp.from_torch(self._policy_aux_eps, dtype=wp.float32)
                if self._policy_aux_eps is not None
                else None
            )
        else:
            self._policy_aux_eps_wp = None

    def _require_training_runtime(self, operation: str) -> None:
        if self.inference_only:
            raise RuntimeError(
                f"PPO {operation} is unavailable in inference-only mode."
            )

    @torch.no_grad()
    def set_action_scales(
        self,
        *,
        mean_scale: torch.Tensor | None = None,
        std_scale: torch.Tensor | None = None,
    ) -> None:
        if mean_scale is not None:
            self.network.set_action_mean_scale(mean_scale)
        if std_scale is not None:
            self.network.set_action_std_scale(std_scale)

    @staticmethod
    def _resolve_shuffle_mode(shuffle_mode: str) -> str:
        key = str(shuffle_mode).strip().lower().replace("-", "_")
        if key in {"per_epoch", "epoch"}:
            return "per_epoch"
        if key in {"per_update", "update"}:
            return "per_update"
        if key in {"off", "none", "disabled", "false"}:
            return "off"
        raise ValueError(
            f"Unsupported shuffle_mode: {shuffle_mode}. Use one of: per_epoch, per_update, off."
        )

    @staticmethod
    def _resolve_optimizer_name(optimizer_name: str | None) -> str:
        return resolve_optimizer_name(optimizer_name)

    @staticmethod
    def _resolve_preprocessor_name(preprocessor_name: str | None) -> str:
        return resolve_observation_preprocessor_name(preprocessor_name)

    def _validate_graph_optimizer_constraints(
        self, optimizer_kwargs: dict[str, Any]
    ) -> None:
        validate_graph_optimizer_constraints(
            self.optimizer_name,
            optimizer_kwargs,
            use_update_graph=self._use_update_graph,
        )

    def _validate_preprocessor_constraints(
        self,
        *,
        state_preprocessor: str | None,
        state_preprocessor_kwargs: dict[str, Any],
        value_preprocessor: str | None,
        value_preprocessor_kwargs: dict[str, Any],
    ) -> None:
        if not self._use_update_graph:
            return
        for label, name, kwargs in (
            ("state", state_preprocessor, state_preprocessor_kwargs),
            ("value", value_preprocessor, value_preprocessor_kwargs),
        ):
            resolved_name = self._resolve_preprocessor_name(name)
            if (
                resolved_name in {"running_standard", "rsl_rl_empirical"}
                and kwargs.get("max_samples") is not None
            ):
                raise ValueError(
                    f"{label}_preprocessor max_samples is unsupported with runtime.update_cuda_graph=true."
                )

    def _build_optimizer(
        self, *, learning_rate: float, optimizer_kwargs: dict[str, Any]
    ) -> torch.optim.Optimizer:
        return build_optimizer(
            params=self.network.named_parameters(),
            optimizer_name=self.optimizer_name,
            learning_rate=learning_rate,
            optimizer_kwargs=optimizer_kwargs,
            capturable=self._use_update_graph
            or (self.loss_scaler is not None and self.device.type == "cuda"),
            device=self.device,
        )

    def _observation_preprocessor_kwargs(
        self,
        preprocessor_name: str | None,
        preprocessor_kwargs: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Freeze the statistics of the observation prefix a frozen actor reads."""
        kwargs = dict(preprocessor_kwargs or {})
        frozen_prefix_dim = int(
            getattr(self.network, "frozen_observation_prefix_dim", 0)
        )
        if frozen_prefix_dim <= 0:
            return kwargs
        resolve_observation_preprocessor_name(preprocessor_name)
        kwargs["frozen_prefix_dim"] = frozen_prefix_dim
        return kwargs

    def _build_preprocessor(
        self,
        preprocessor_name: str | None,
        size: int | Sequence[int],
        preprocessor_kwargs: dict[str, Any],
    ) -> RunningStandardScaler | RslRlEmpiricalNormalization:
        return build_observation_preprocessor(
            preprocessor=preprocessor_name,
            preprocessor_kwargs=preprocessor_kwargs,
            size=size,
            device=self.device,
        )

    def _resolve_popart_value_head(self) -> nn.Linear | None:
        if not self.value_scaler.enabled:
            return None
        value_head = getattr(self.network, "value_head", None)
        if not isinstance(value_head, nn.Linear) or value_head.out_features != 1:
            raise ValueError(
                "RunningStandardScaler value preprocessing requires a scalar nn.Linear value_head for PopArt reparameterization."
            )
        if value_head.bias is None:
            raise ValueError(
                "RunningStandardScaler value preprocessing requires a biased value_head for PopArt reparameterization."
            )
        return value_head

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str | torch.dtype) -> torch.dtype:
        if isinstance(amp_dtype, torch.dtype):
            if amp_dtype in (torch.bfloat16, torch.float16):
                return amp_dtype
            raise ValueError(f"Unsupported amp dtype: {amp_dtype}")
        key = str(amp_dtype).strip().lower()
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"fp16", "float16", "half"}:
            return torch.float16
        raise ValueError(
            f"Unsupported amp dtype: {amp_dtype}. Use one of: bfloat16, bf16, float16, fp16."
        )

    def _warn_if_fp16_without_loss_scaling(self, *, mixed_precision: bool) -> None:
        if not mixed_precision or self._amp_dtype != torch.float16:
            return
        _LOGGER.warning(
            "Legacy PPO mixed_precision with float16 does not apply loss scaling. Use precision=float16 for shared policy precision and dynamic loss scaling."
        )

    def _update_autocast_ctx(self):
        if self.precision is None and self._use_mixed_precision:
            return torch.amp.autocast(device_type="cuda", dtype=self._amp_dtype)
        return nullcontext()

    def _nvtx_range(self, name: str):
        if not self._detailed_nvtx:
            return nullcontext()
        return torch.cuda.nvtx.range(name)

    def _fill_policy_noise_(self) -> torch.Tensor:
        if self._use_warp_noise:
            wp.launch(
                increment_seed_kernel,
                dim=(1,),
                inputs=[self._noise_seed_wp],
                device=self._wp_device,
            )
            wp.launch(
                fill_gaussian_noise_kernel,
                dim=(self.num_envs, self.act_dim),
                inputs=[self._noise_seed_wp, self.act_dim],
                outputs=[self._policy_eps_wp],
                device=self._wp_device,
            )
            if self._policy_aux_eps_wp is not None and self._policy_aux_eps is not None:
                wp.launch(
                    increment_seed_kernel,
                    dim=(1,),
                    inputs=[self._noise_seed_wp],
                    device=self._wp_device,
                )
                wp.launch(
                    fill_gaussian_noise_kernel,
                    dim=(self.num_envs, self._policy_aux_dim),
                    inputs=[self._noise_seed_wp, self._policy_aux_dim],
                    outputs=[self._policy_aux_eps_wp],
                    device=self._wp_device,
                )
        else:
            self._policy_eps.normal_()
            if self._policy_aux_eps is not None:
                self._policy_aux_eps.normal_()
        return self._policy_eps

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with network_matmul_precision():
            scaled_obs = self._scaled_eval_obs(obs)
            actions, log_prob, values = self.network.act(scaled_obs)
            self._inverse_scale_values_(values)
            return (actions, log_prob, values)

    @torch.no_grad()
    def _obs_scale_into(
        self, obs: torch.Tensor, out: torch.Tensor, *, train: bool
    ) -> None:
        self.obs_scaler(obs, train=train, out=out)
        return
        out.copy_(obs)
        self.obs_scaler(
            obs[..., : self.obs_dim], train=train, out=out[..., : self.obs_dim]
        )

    @torch.no_grad()
    def _obs_scale_eval_into(self, obs: torch.Tensor, out: torch.Tensor) -> None:
        self._obs_scale_into(obs, out, train=False)

    @torch.no_grad()
    def _obs_scale_update_into(self, obs: torch.Tensor, out: torch.Tensor) -> None:
        self._obs_scale_into(obs, out, train=False)
        stats_obs = obs
        self.obs_scaler.observe_rollout_batch(stats_obs)

    @torch.no_grad()
    def update_observation_normalization(self, obs: torch.Tensor) -> bool:
        """Advance observation moments at the normalizer's declared cadence."""
        stats_obs = obs
        return self.obs_scaler.observe_rollout_step(stats_obs)

    @torch.no_grad()
    def _scaled_eval_obs(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device, dtype=torch.float32)
        if (
            obs.ndim == 2
            and obs.shape[0] == self.num_envs
            and (obs.shape[1] == self.obs_dim)
        ):
            self._obs_scale_eval_into_fn(obs, self._policy_scaled_obs)
            return self._policy_scaled_obs
        scaled_obs = torch.empty_like(obs)
        self._obs_scale_eval_into(obs, scaled_obs)
        return scaled_obs

    def _warmup_compiled_obs_scalers(self, *, warmup_update: bool) -> None:
        components: list[tuple[str, Callable[[], Any]]] = []
        if warmup_update:
            assert self._update_scaled_obs is not None
            for label, value in (
                ("observation mean", self.obs_scaler.running_mean),
                ("observation variance", self.obs_scaler.running_variance),
                ("observation count", self.obs_scaler.current_count),
                ("scaled observations", self._update_scaled_obs),
            ):
                saved = value.detach().clone()
                components.append(
                    (label, lambda value=value, saved=saved: value.copy_(saved))
                )
        try:
            if warmup_update:
                flat_obs = self.buffer.obs[: self.rollouts].reshape(-1, self.obs_dim)
                self._obs_scale_update_into_fn(flat_obs, self._update_scaled_obs)
                self._restore_warmup_components(components)
            self._synchronize_warmup_work()
        except Exception as error:
            self._restore_warmup_components(
                [("pending scaler warmup", self._synchronize_warmup_work), *components],
                error=error,
            )
            raise

    @torch.no_grad()
    def _warmup_compiled_prepare_update_views(self) -> None:
        values = [
            ("rollout values", self.buffer.values),
            ("rollout returns", self.buffer.returns),
            ("rollout advantages", self.buffer.advantages),
            ("bootstrap values", self._update_last_values),
            ("value mean", self.value_scaler.running_mean),
            ("value variance", self.value_scaler.running_variance),
            ("value count", self.value_scaler.current_count),
            ("raw returns", self._raw_update_returns),
            ("old value mean", self._value_popart_old_mean),
            ("old value scale", self._value_popart_old_scale),
            ("bootstrap observations", self._policy_scaled_obs),
            ("bootstrap actions", self._bootstrap_actions),
            ("bootstrap log probabilities", self._bootstrap_log_probs),
        ]
        if self._value_head is not None:
            values.extend(
                (
                    ("value-head weights", self._value_head.weight),
                    ("value-head bias", self._value_head.bias),
                )
            )
        values.extend(
            (
                ("network buffer " + name, value)
                for name, value in self.network.named_buffers()
            )
        )
        if self.initial_state_learner is not None:
            values.append(
                (
                    "initial-state return targets",
                    self.initial_state_learner.return_targets,
                )
            )
        components: list[tuple[str, Callable[[], Any]]] = []
        for label, value in values:
            saved = value.detach().clone()
            components.append(
                (label, lambda value=value, saved=saved: value.copy_(saved))
            )
        try:
            with network_matmul_precision():
                self._prepare_update_views_fn()
            self._restore_warmup_components(components)
            self._synchronize_warmup_work()
        except Exception as error:
            self._restore_warmup_components(
                [
                    ("pending prepare-view warmup", self._synchronize_warmup_work),
                    *components,
                ],
                error=error,
            )
            raise

    def _inverse_scale_values_(self, values: torch.Tensor) -> None:
        if not self.value_scaler.enabled:
            return
        values.mul_(
            torch.sqrt(self.value_scaler.running_variance.float()).clamp_min(
                self.value_scaler.epsilon
            )
        )
        values.add_(self.value_scaler.running_mean.float())

    @torch.no_grad()
    def _prepare_value_update_coordinates(self) -> None:
        """Move PPO value tensors and the critic head to one PopArt snapshot."""
        if not self.value_scaler.enabled:
            return
        assert self._value_head is not None
        self._raw_update_returns.copy_(self.buffer.returns)
        self._value_popart_old_mean.copy_(self.value_scaler.running_mean.float())
        self._value_popart_old_scale.copy_(
            torch.sqrt(self.value_scaler.running_variance.float()).clamp_min(
                self.value_scaler.epsilon
            )
        )
        self.value_scaler.update_stats(self._raw_update_returns)
        new_mean = self.value_scaler.running_mean.float()
        new_scale = torch.sqrt(self.value_scaler.running_variance.float()).clamp_min(
            self.value_scaler.epsilon
        )
        scale = self._value_popart_old_scale / new_scale
        shift = (self._value_popart_old_mean - new_mean) / new_scale
        self._value_head.weight.mul_(scale)
        self._value_head.bias.mul_(scale).add_(shift)
        self.buffer.values.sub_(new_mean).div_(new_scale)
        self.buffer.returns.copy_(self._raw_update_returns)
        self.buffer.returns.sub_(new_mean).div_(new_scale)

    @torch.no_grad()
    def act_into(self, step: int) -> None:
        """Read obs from buffer[step], write actions/log_probs/values into buffer[step] in-place.

        The NVTX range is gated. It used to be an unconditional
        `@torch.cuda.nvtx.range` decorator, which pushes and pops a range on
        every rollout step whether or not anyone is profiling, and cannot be
        turned off; `_nvtx_range` respects `runtime.nsys_nvtx_mode=detailed`
        like the rest of this class.
        """
        with self._nvtx_range("act_into"):
            self._act_into_body(step)

    def _act_into_body(
        self,
        step: int,
        *,
        eps: torch.Tensor | None = None,
    ) -> None:
        with network_matmul_precision():
            self._act_into_body_unscoped(step, eps=eps)

    def _act_into_body_unscoped(
        self,
        step: int,
        *,
        eps: torch.Tensor | None = None,
    ) -> None:
        buf = self.buffer
        self._obs_scale_eval_into_fn(buf.obs[step], self._policy_scaled_obs)
        if buf.policy_obs is not None:
            buf.policy_obs[step].copy_(self._policy_scaled_obs)
        policy_eps = eps if eps is not None else self._fill_policy_noise_()
        args, kwargs = self._act_into_call_arguments(
            step=step,
            scaled_obs=self._policy_scaled_obs,
            actions=buf.actions[step],
            log_probs=buf.log_probs[step],
            values=buf.values[step],
            eps=policy_eps,
        )
        self._act_into_fn(*args, **kwargs)
        self._inverse_scale_values_(buf.values[step])

    def _act_into_call_arguments(
        self,
        *,
        step: int,
        scaled_obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        eps: torch.Tensor,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """The exact rollout ``act_into`` call, shared with compile warmup.

        Optional positional and keyword arguments are chosen from the same
        persistent buffers the rollout uses, so warmup cannot specialize a
        different None/tensor pattern and force a recompile during capture.
        """
        buf = self.buffer
        replay_kwargs = (
            {"out_pre_squash_actions": buf.pre_squash_actions[step]}
            if buf.pre_squash_actions is not None
            else {}
        )
        policy_aux = None
        policy_mean = buf.policy_mean[step] if buf.policy_mean is not None else None
        policy_log_std = (
            buf.policy_log_std[step] if buf.policy_log_std is not None else None
        )
        if buf.policy_aux is not None and self._policy_aux_eps is not None:
            buf.policy_aux[step].copy_(self._policy_aux_eps)
            policy_aux = buf.policy_aux[step]
        kwargs = {
            "policy_aux": policy_aux,
            "out_policy_mean": policy_mean,
            "out_policy_log_std": policy_log_std,
            **replay_kwargs,
        }
        return (scaled_obs, actions, log_probs, values, eps), kwargs

    @torch.no_grad()
    def _bootstrap_values_fast(
        self, obs: torch.Tensor, out_values: torch.Tensor
    ) -> None:
        with network_matmul_precision():
            self._bootstrap_values_fast_unscoped(obs, out_values)

    def _bootstrap_values_fast_unscoped(
        self, obs: torch.Tensor, out_values: torch.Tensor
    ) -> None:
        self._obs_scale_eval_into_fn(obs, self._policy_scaled_obs)
        self._act_into_fn(
            self._policy_scaled_obs,
            self._bootstrap_actions,
            self._bootstrap_log_probs,
            out_values,
            self._bootstrap_eps,
            policy_aux=self._bootstrap_policy_aux,
        )
        self._inverse_scale_values_(out_values)

    @torch.no_grad()
    def bootstrap_values_into(
        self, obs: torch.Tensor, out_values: torch.Tensor
    ) -> None:
        """Compute bootstrap value estimates in-place for arbitrary observations."""
        with self._nvtx_range("bootstrap_values_into"):
            self._bootstrap_values_into_body(obs, out_values)

    def _bootstrap_values_into_body(
        self, obs: torch.Tensor, out_values: torch.Tensor
    ) -> None:
        with network_matmul_precision():
            self._bootstrap_values_into_unscoped(obs, out_values)

    def _bootstrap_values_into_unscoped(
        self, obs: torch.Tensor, out_values: torch.Tensor
    ) -> None:
        obs = obs.to(self.device)
        if (
            obs.ndim == 2
            and obs.shape[0] == self.num_envs
            and (obs.shape[1] == self.obs_dim)
        ):
            self._bootstrap_values_fast(obs, out_values)
            return
        scaled_obs = torch.empty_like(obs, dtype=torch.float32, device=self.device)
        self._obs_scale_eval_into(obs, scaled_obs)
        batch_size = int(scaled_obs.shape[0])
        bootstrap_actions = torch.empty(batch_size, self.act_dim, device=self.device)
        bootstrap_log_probs = torch.empty(batch_size, 1, device=self.device)
        bootstrap_eps = torch.zeros(batch_size, self.act_dim, device=self.device)
        self._act_into_fn(
            scaled_obs,
            bootstrap_actions,
            bootstrap_log_probs,
            out_values,
            bootstrap_eps,
            policy_aux=None,
        )
        self._inverse_scale_values_(out_values)

    def _refresh_optimizer_lr_wrappers(self) -> None:
        self._optimizer_lr_wp = []
        self._lr_kl_tensor = None
        self._lr_kl_wp = None
        if self._use_update_graph:
            self._lr_kl_tensor = torch.empty(1, device=self.device, dtype=torch.float32)
            self._lr_kl_wp = wp.from_torch(self._lr_kl_tensor, dtype=wp.float32)
            for group in self.optimizer.param_groups:
                group_lr = group["lr"]
                if torch.is_tensor(group_lr):
                    if group_lr.numel() != 1:
                        raise ValueError("Tensor lr must be 1-element")
                    if (
                        group_lr.device != self.device
                        or group_lr.dtype != torch.float32
                    ):
                        group_lr = group_lr.to(device=self.device, dtype=torch.float32)
                        group["lr"] = group_lr
                    self._optimizer_lr_wp.append(
                        wp.from_torch(group_lr.reshape(1), dtype=wp.float32)
                    )
                else:
                    self._optimizer_lr_wp.append(None)
        # Refresh after the normalization above: the constants are allocated
        # per learning-rate dtype, so a float64 tensor LR normalized to float32
        # must produce float32 constants. This runs at construction and on
        # resume, never inside a capture.
        self._refresh_lr_schedule_constants()

    def _update_learning_rate(self, kl_mean: torch.Tensor | float) -> None:
        """Apply the KL learning-rate schedule in device arithmetic.

        The schedule constants are device tensors allocated per learning-rate
        dtype at construction, so a captured minibatch reads them instead of
        moving a Python scalar onto the device inside the capture window; such
        a host-to-device copy aborts capture.
        """
        if self.kl_threshold <= 0.0 or self._update_mode == "value_bootstrap":
            return
        self._apply_learning_rate_schedule(kl_mean)

    def _lr_schedule_values(self) -> tuple[float, ...]:
        """The KL-schedule constants this agent currently schedules with."""
        return (
            self.kl_threshold * 2.0,
            self.kl_threshold / 2.0,
            self.lr_factor,
            self.min_learning_rate,
            self.max_learning_rate,
        )

    def _refresh_lr_schedule_constants(self) -> None:
        """Allocate or update the device scalars the KL schedule reads.

        One tensor per constant and per optimizer learning-rate dtype, created
        before any capture. The update path only reads them, so a captured
        minibatch never creates a tensor or mutates a guarded mapping. Call this
        outside capture after changing a schedule value (construction, resume).
        """
        optimizer = getattr(self, "optimizer", None)
        constants_by_dtype = getattr(self, "_lr_schedule_constants", None)
        if constants_by_dtype is None:
            constants_by_dtype = {}
            self._lr_schedule_constants = constants_by_dtype
        if optimizer is None:
            constants_by_dtype.clear()
            return
        dtypes = {
            group["lr"].dtype
            for group in optimizer.param_groups
            if torch.is_tensor(group["lr"])
        }
        # Optimizer construction leaves the learning rate a Python float and
        # graph preparation promotes it to float32. Allocate that dtype eagerly
        # so a caller that installs a tensor LR between the two never has to
        # create a schedule scalar from inside the update path.
        dtypes.add(torch.float32)
        values = self._lr_schedule_values()
        for dtype in dtypes:
            constants = constants_by_dtype.get(dtype)
            if constants is None:
                constants = {
                    name: torch.zeros((), device=self.device, dtype=dtype)
                    for name in _LR_SCHEDULE_CONSTANT_NAMES
                }
                constants_by_dtype[dtype] = constants
            for name, value in zip(_LR_SCHEDULE_CONSTANT_NAMES, values, strict=True):
                constants[name].fill_(float(value))

    def _apply_learning_rate_schedule(self, kl_mean: torch.Tensor | float) -> None:
        for group_idx, group in enumerate(self.optimizer.param_groups):
            group_lr = group["lr"]
            if torch.is_tensor(group_lr):
                lr_wp = (
                    self._optimizer_lr_wp[group_idx]
                    if group_idx < len(self._optimizer_lr_wp)
                    else None
                )
                if (
                    self.kl_scheduler_update_mode in {"per_update", "per_epoch"}
                    and lr_wp is not None
                    and (self._lr_kl_tensor is not None)
                    and (self._lr_kl_wp is not None)
                    and (self._update_wp_device is not None)
                ):
                    if torch.is_tensor(kl_mean):
                        kl_tensor = kl_mean.detach()
                        if kl_tensor.device != self.device:
                            kl_tensor = kl_tensor.to(self.device)
                        self._lr_kl_tensor.copy_(kl_tensor.reshape(1))
                    else:
                        self._lr_kl_tensor.fill_(float(kl_mean))
                    wp.launch(
                        _update_lr_tensor_kernel,
                        dim=1,
                        inputs=[
                            self._lr_kl_wp,
                            lr_wp,
                            self.kl_threshold,
                            self.lr_factor,
                            self.min_learning_rate,
                            self.max_learning_rate,
                        ],
                        device=self._update_wp_device,
                    )
                    continue
                if torch.is_tensor(kl_mean):
                    kl = kl_mean.to(device=group_lr.device, dtype=group_lr.dtype)
                else:
                    kl = torch.tensor(
                        kl_mean, device=group_lr.device, dtype=group_lr.dtype
                    )
                constants = self._lr_schedule_constants[group_lr.dtype]
                high = constants["kl_high"]
                low = constants["kl_low"]
                factor = constants["factor"]
                min_lr = constants["min_lr"]
                max_lr = constants["max_lr"]
                decreased = torch.maximum(group_lr / factor, min_lr)
                increased = torch.minimum(group_lr * factor, max_lr)
                increase_condition = kl < low
                if self.kl_scheduler_update_mode == "per_minibatch":
                    increase_condition = increase_condition & (kl > 0.0)
                new_lr = torch.where(
                    kl > high,
                    decreased,
                    torch.where(increase_condition, increased, group_lr),
                )
                group_lr.copy_(new_lr)
            else:
                kl = (
                    float(kl_mean.detach().cpu())
                    if torch.is_tensor(kl_mean)
                    else float(kl_mean)
                )
                lr = float(group_lr)
                if kl > self.kl_threshold * 2.0:
                    lr = max(lr / self.lr_factor, self.min_learning_rate)
                elif kl < self.kl_threshold / 2.0 and (
                    self.kl_scheduler_update_mode in {"per_update", "per_epoch"}
                    or kl > 0.0
                ):
                    lr = min(lr * self.lr_factor, self.max_learning_rate)
                group["lr"] = lr

    def _write_current_learning_rate(self, out_lr: torch.Tensor) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        if torch.is_tensor(lr):
            out_lr.copy_(lr.reshape_as(out_lr))
            return
        out_lr.fill_(float(lr))

    def _write_update_summary_values(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        policy_aux_loss: torch.Tensor,
        kl: torch.Tensor,
        clip_fraction: torch.Tensor,
        std_actual_mean: torch.Tensor,
        std_actual_min: torch.Tensor,
        std_actual_max: torch.Tensor,
        raw_log_std_clamp_lower_fraction: torch.Tensor,
        raw_log_std_clamp_upper_fraction: torch.Tensor,
        raw_log_std_clamp_overshoot_rms: torch.Tensor,
        raw_log_std_clamp_overshoot_max: torch.Tensor,
        log_std_bound_loss: torch.Tensor,
    ) -> None:
        self._update_summary[0].copy_(policy_loss)
        self._update_summary[1].copy_(value_loss)
        self._update_summary[2].copy_(entropy_loss)
        self._update_summary[3].copy_(policy_aux_loss)
        self._update_summary[4].copy_(kl)
        self._update_summary[5].copy_(clip_fraction)
        self._write_current_learning_rate(self._update_summary[6:7])
        self._update_summary[7].copy_(self.network.policy_std_mean())
        self._update_summary[8].copy_(std_actual_mean)
        self._update_summary[9].copy_(std_actual_min)
        self._update_summary[10].copy_(std_actual_max)
        self._update_summary[11].copy_(raw_log_std_clamp_lower_fraction)
        self._update_summary[12].copy_(raw_log_std_clamp_upper_fraction)
        self._update_summary[13].copy_(raw_log_std_clamp_overshoot_rms)
        self._update_summary[14].copy_(raw_log_std_clamp_overshoot_max)
        self._update_summary[15].copy_(log_std_bound_loss)

    def _write_update_summary_from_metrics(self, metrics: torch.Tensor) -> None:
        mean_metrics = metrics.mean(dim=0)
        self._write_update_summary_values(
            mean_metrics[0],
            mean_metrics[1],
            mean_metrics[2],
            mean_metrics[3],
            mean_metrics[_UPDATE_KL_METRIC_INDEX],
            mean_metrics[5],
            mean_metrics[6],
            metrics[:, 7].amin(),
            metrics[:, 8].amax(),
            mean_metrics[9],
            mean_metrics[10],
            mean_metrics[11].sqrt(),
            metrics[:, 12].amax(),
            mean_metrics[13],
        )
        self._write_extra_update_summary_from_metrics(mean_metrics)

    def _read_update_summary(self) -> dict[str, float]:
        self._update_summary_host.copy_(self._update_summary, non_blocking=False)
        summary_values = self._update_summary_host.tolist()
        summary = {
            key: float(value)
            for key, value in zip(
                self._update_summary_keys(), summary_values, strict=True
            )
        }
        if not self._update_uses_aux_evaluate:
            del summary["policy_aux"]
        return summary

    def prepare_cuda_libraries_for_graph_capture(self) -> None:
        """Create lazy CUDA library handles before entering external graph capture."""
        if self.device.type != "cuda" or self._cuda_libraries_graph_capture_ready:
            return
        with torch.no_grad():
            lhs = torch.empty((1, 1), device=self.device, dtype=torch.float32)
            rhs = torch.empty((1, 1), device=self.device, dtype=torch.float32)
            torch.mm(lhs, rhs)
        torch.cuda.synchronize(self.device)
        self._cuda_libraries_graph_capture_ready = True

    def _warmup_compiled_act_into_signatures(self, *, force: bool = False) -> None:
        """Warm ``act_into`` with the real rollout and bootstrap signatures.

        This runs after the rollout buffer and policy workspace exist and calls
        the same producers as the captured paths. Optional arguments therefore
        keep the exact None/tensor pattern of the rollout calls.
        """
        if not force and not self._warmup_compiled_act_into:
            return
        if self.buffer is None or getattr(self, "_policy_scaled_obs", None) is None:
            return
        with torch.no_grad():
            self._act_into_body(0, eps=self._policy_eps)
            self._bootstrap_values_fast(
                self.buffer.obs[0],
                torch.empty_like(self.buffer.values[0]),
            )

    def _warmup_compiled_functions(self, warmup_act_into: bool) -> None:
        """Compile graph-specialized callables before any external CUDA capture."""
        with network_matmul_precision():
            self._warmup_compiled_functions_unscoped(warmup_act_into)

    def _warmup_compiled_functions_unscoped(self, warmup_act_into: bool) -> None:
        update_wp_stream = getattr(self, "_update_wp_stream", None)
        update_wp_stream_ctx = (
            wp.ScopedStream(update_wp_stream, sync_enter=False, sync_exit=False)
            if self._use_update_graph and update_wp_stream is not None
            else nullcontext()
        )
        update_stream_ctx = (
            torch.cuda.stream(update_torch_stream)
            if self._use_update_graph
            and (update_torch_stream := getattr(self, "_update_torch_stream", None))
            is not None
            else nullcontext()
        )
        with update_wp_stream_ctx:
            with update_stream_ctx:
                if warmup_act_into:
                    self._warmup_compiled_act_into_signatures(force=True)

                warmup_batch = self._update_batch_size
                warmup_eval_obs = torch.zeros(
                    warmup_batch, self.obs_dim, device=self.device, dtype=torch.float32
                )
                warmup_eval_actions = torch.zeros(
                    warmup_batch, self.act_dim, device=self.device, dtype=torch.float32
                )
                warmup_eval_policy_aux = (
                    torch.zeros(
                        warmup_batch,
                        self._policy_aux_dim,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    if self._policy_aux_dim > 0
                    else None
                )
                warmup_replay_eval_kwargs = (
                    {"pre_squash_actions": warmup_eval_actions}
                    if self._uses_pre_squash_action_replay
                    else {}
                )
                warmup_raw_log_std_bound_stats = getattr(
                    self,
                    "_update_eval_raw_log_std_bound_stats",
                    torch.empty(5, device=self.device, dtype=torch.float32),
                )
                if self._compiled_update_uses_return_path:
                    if self._update_uses_aux_evaluate:
                        aux_kwargs = {}
                        if warmup_eval_policy_aux is not None:
                            aux_kwargs["policy_aux"] = warmup_eval_policy_aux
                        if self._update_eval_policy_mean is not None:
                            aux_kwargs["out_policy_mean"] = (
                                self._update_eval_policy_mean
                            )
                            aux_kwargs["out_policy_log_std"] = (
                                self._update_eval_policy_log_std
                            )
                        aux_kwargs["out_raw_log_std_bound_stats"] = (
                            warmup_raw_log_std_bound_stats
                        )
                        self._evaluate_with_aux_std_fn(
                            warmup_eval_obs,
                            warmup_eval_actions,
                            **aux_kwargs,
                            **warmup_replay_eval_kwargs,
                        )
                    elif warmup_eval_policy_aux is None:
                        self._evaluate_with_std_fn(
                            warmup_eval_obs,
                            warmup_eval_actions,
                            out_policy_mean=self._update_eval_policy_mean,
                            out_policy_log_std=self._update_eval_policy_log_std,
                            out_raw_log_std_bound_stats=warmup_raw_log_std_bound_stats,
                            **warmup_replay_eval_kwargs,
                        )
                    else:
                        self._evaluate_with_std_fn(
                            warmup_eval_obs,
                            warmup_eval_actions,
                            policy_aux=warmup_eval_policy_aux,
                            out_policy_mean=self._update_eval_policy_mean,
                            out_policy_log_std=self._update_eval_policy_log_std,
                            out_raw_log_std_bound_stats=warmup_raw_log_std_bound_stats,
                            **warmup_replay_eval_kwargs,
                        )
                    if self._use_update_graph:
                        self.optimizer.zero_grad(set_to_none=True)
                        with self._update_autocast_ctx():
                            if self._update_uses_aux_evaluate:
                                aux_kwargs = {}
                                if warmup_eval_policy_aux is not None:
                                    aux_kwargs["policy_aux"] = warmup_eval_policy_aux
                                if self._update_eval_policy_mean is not None:
                                    aux_kwargs["out_policy_mean"] = (
                                        self._update_eval_policy_mean
                                    )
                                    aux_kwargs["out_policy_log_std"] = (
                                        self._update_eval_policy_log_std
                                    )
                                aux_kwargs["out_raw_log_std_bound_stats"] = (
                                    warmup_raw_log_std_bound_stats
                                )
                                (
                                    warmup_log_probs,
                                    warmup_values,
                                    warmup_entropy,
                                    warmup_policy_aux_loss,
                                    _warmup_std_mean,
                                    _warmup_std_min,
                                    _warmup_std_max,
                                ) = self._evaluate_with_aux_std_fn(
                                    warmup_eval_obs,
                                    warmup_eval_actions,
                                    **aux_kwargs,
                                    **warmup_replay_eval_kwargs,
                                )
                            else:
                                if warmup_eval_policy_aux is None:
                                    (
                                        warmup_log_probs,
                                        warmup_values,
                                        warmup_entropy,
                                        _warmup_std_mean,
                                        _warmup_std_min,
                                        _warmup_std_max,
                                    ) = self._evaluate_with_std_fn(
                                        warmup_eval_obs,
                                        warmup_eval_actions,
                                        out_policy_mean=self._update_eval_policy_mean,
                                        out_policy_log_std=self._update_eval_policy_log_std,
                                        out_raw_log_std_bound_stats=warmup_raw_log_std_bound_stats,
                                        **warmup_replay_eval_kwargs,
                                    )
                                else:
                                    (
                                        warmup_log_probs,
                                        warmup_values,
                                        warmup_entropy,
                                        _warmup_std_mean,
                                        _warmup_std_min,
                                        _warmup_std_max,
                                    ) = self._evaluate_with_std_fn(
                                        warmup_eval_obs,
                                        warmup_eval_actions,
                                        policy_aux=warmup_eval_policy_aux,
                                        out_policy_mean=self._update_eval_policy_mean,
                                        out_policy_log_std=self._update_eval_policy_log_std,
                                        out_raw_log_std_bound_stats=warmup_raw_log_std_bound_stats,
                                        **warmup_replay_eval_kwargs,
                                    )
                                warmup_policy_aux_loss = self.network.zero_scalar()
                            warmup_loss = (
                                warmup_log_probs.float().mean()
                                + warmup_values.float().mean()
                                + warmup_entropy.float().mean()
                                + warmup_policy_aux_loss.float()
                                + warmup_raw_log_std_bound_stats[4].float()
                            )
                        warmup_loss.backward()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    warmup_out_log_probs = getattr(
                        self,
                        "_update_eval_log_probs",
                        torch.empty(
                            warmup_batch, 1, device=self.device, dtype=torch.float32
                        ),
                    )
                    warmup_out_values = getattr(
                        self,
                        "_update_eval_values",
                        torch.empty(
                            warmup_batch, 1, device=self.device, dtype=torch.float32
                        ),
                    )
                    warmup_out_entropy = getattr(
                        self,
                        "_update_eval_entropy",
                        torch.empty(
                            warmup_batch, 1, device=self.device, dtype=torch.float32
                        ),
                    )
                    warmup_out_policy_aux_loss = getattr(
                        self,
                        "_update_eval_policy_aux_loss",
                        torch.empty((), device=self.device, dtype=torch.float32),
                    )
                    warmup_out_std_mean = getattr(
                        self,
                        "_update_eval_std_mean",
                        torch.empty((), device=self.device, dtype=torch.float32),
                    )
                    warmup_out_std_min = getattr(
                        self,
                        "_update_eval_std_min",
                        torch.empty((), device=self.device, dtype=torch.float32),
                    )
                    warmup_out_std_max = getattr(
                        self,
                        "_update_eval_std_max",
                        torch.empty((), device=self.device, dtype=torch.float32),
                    )
                    self._evaluate_update_batch(
                        warmup_eval_obs,
                        warmup_eval_actions,
                        warmup_out_log_probs,
                        warmup_out_values,
                        warmup_out_entropy,
                        warmup_out_policy_aux_loss,
                        warmup_out_std_mean,
                        warmup_out_std_min,
                        warmup_out_std_max,
                        warmup_raw_log_std_bound_stats,
                        batch_policy_aux=warmup_eval_policy_aux,
                        out_policy_mean=self._update_eval_policy_mean,
                        out_policy_log_std=self._update_eval_policy_log_std,
                    )
                    if self._use_update_graph:
                        self.optimizer.zero_grad(set_to_none=True)
                        with self._update_autocast_ctx():
                            self._evaluate_update_batch(
                                warmup_eval_obs,
                                warmup_eval_actions,
                                warmup_out_log_probs,
                                warmup_out_values,
                                warmup_out_entropy,
                                warmup_out_policy_aux_loss,
                                warmup_out_std_mean,
                                warmup_out_std_min,
                                warmup_out_std_max,
                                warmup_raw_log_std_bound_stats,
                                batch_policy_aux=warmup_eval_policy_aux,
                                out_policy_mean=self._update_eval_policy_mean,
                                out_policy_log_std=self._update_eval_policy_log_std,
                            )
                            warmup_loss = (
                                warmup_out_log_probs.float().mean()
                                + warmup_out_values.float().mean()
                                + warmup_out_entropy.float().mean()
                                + warmup_out_policy_aux_loss.float()
                                + warmup_raw_log_std_bound_stats[4].float()
                            )
                        warmup_loss.backward()
                        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _evaluate_update_batch(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        out_log_probs: torch.Tensor,
        out_values: torch.Tensor,
        out_entropy: torch.Tensor,
        out_policy_aux_loss: torch.Tensor,
        out_std_mean: torch.Tensor | None = None,
        out_std_min: torch.Tensor | None = None,
        out_std_max: torch.Tensor | None = None,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
        *,
        batch_policy_aux: torch.Tensor | None = None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
    ) -> None:
        replay_kwargs = {}
        if self._uses_pre_squash_action_replay:
            replay_kwargs["pre_squash_actions"] = batch_actions
            batch_actions = batch_actions.tanh()
        collect_std_stats = (
            out_std_mean is not None
            and out_std_min is not None
            and (out_std_max is not None)
        )
        if self._update_uses_aux_evaluate:
            aux_kwargs = {}
            if batch_policy_aux is not None:
                aux_kwargs["policy_aux"] = batch_policy_aux
            if collect_std_stats:
                assert out_std_mean is not None
                assert out_std_min is not None
                assert out_std_max is not None
                self._evaluate_with_aux_std_into_fn(
                    batch_obs,
                    batch_actions,
                    out_log_probs,
                    out_values,
                    out_entropy,
                    out_policy_aux_loss,
                    out_std_mean,
                    out_std_min,
                    out_std_max,
                    out_policy_mean=out_policy_mean,
                    out_policy_log_std=out_policy_log_std,
                    out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
                    **aux_kwargs,
                    **replay_kwargs,
                )
            else:
                self._evaluate_with_aux_into_fn(
                    batch_obs,
                    batch_actions,
                    out_log_probs,
                    out_values,
                    out_entropy,
                    out_policy_aux_loss,
                    **aux_kwargs,
                    **replay_kwargs,
                )
            return
        if batch_policy_aux is None:
            if collect_std_stats:
                assert out_std_mean is not None
                assert out_std_min is not None
                assert out_std_max is not None
                self._evaluate_with_std_into_fn(
                    batch_obs,
                    batch_actions,
                    out_log_probs,
                    out_values,
                    out_entropy,
                    out_std_mean,
                    out_std_min,
                    out_std_max,
                    out_policy_mean=out_policy_mean,
                    out_policy_log_std=out_policy_log_std,
                    out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
                    **replay_kwargs,
                )
            else:
                self._evaluate_into_fn(
                    batch_obs,
                    batch_actions,
                    out_log_probs,
                    out_values,
                    out_entropy,
                    **replay_kwargs,
                )
        elif collect_std_stats:
            assert out_std_mean is not None
            assert out_std_min is not None
            assert out_std_max is not None
            self._evaluate_with_std_into_fn(
                batch_obs,
                batch_actions,
                out_log_probs,
                out_values,
                out_entropy,
                out_std_mean,
                out_std_min,
                out_std_max,
                out_policy_mean=out_policy_mean,
                out_policy_log_std=out_policy_log_std,
                out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
                policy_aux=batch_policy_aux,
                **replay_kwargs,
            )
        else:
            self._evaluate_into_fn(
                batch_obs,
                batch_actions,
                out_log_probs,
                out_values,
                out_entropy,
                policy_aux=batch_policy_aux,
                **replay_kwargs,
            )
        out_policy_aux_loss.zero_()

    def _warmup_compiled_update_minibatch(
        self, *, max_minibatches: int | None = None
    ) -> None:
        """Compile a real update while preserving its live training boundary.

        Optimizer and tensor storage stay fixed on successful setup. If setup fails,
        pending update-stream work finishes before rollback. Cleanup failures are
        recorded on the original error and the remaining restoration is attempted.
        Compiler caches are deliberately retained; training state is not.
        """
        with network_matmul_precision():
            self._warmup_compiled_update_minibatch_unscoped(
                max_minibatches=max_minibatches
            )

    def _warmup_compiled_update_minibatch_unscoped(
        self, *, max_minibatches: int | None = None
    ) -> None:
        parameters = tuple(self.network.parameters())
        saved_gradients = [
            (
                parameter.grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for parameter in parameters
        ]
        saved_optimizer_state = self._snapshot_optimizer_state()
        if self.loss_scaler is not None:
            saved_pre_scaler = copy.deepcopy(self.loss_scaler.state_dict())
        saved_preprocessors = [
            (
                "observation normalizer",
                self.obs_scaler,
                copy.deepcopy(self.obs_scaler.state_dict()),
            ),
            (
                "value normalizer",
                self.value_scaler,
                copy.deepcopy(self.value_scaler.state_dict()),
            ),
        ]
        if self.reward_preprocessor is not None:
            saved_preprocessors.append(
                (
                    "reward normalizer",
                    self.reward_preprocessor,
                    copy.deepcopy(self.reward_preprocessor.state_dict()),
                )
            )
        saved_tensors: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        seen_tensors: set[int] = set()

        def remember(label: str, value: torch.Tensor | None) -> None:
            if value is not None and id(value) not in seen_tensors:
                seen_tensors.add(id(value))
                saved_tensors.append((label, value, value.detach().clone()))

        for name, value in self.network.named_parameters():
            remember("network parameter " + name, value)
        for name, value in self.network.named_buffers():
            remember("network buffer " + name, value)
        for name in (
            "values",
            "returns",
            "advantages",
            "log_probs",
            "policy_mean",
            "policy_log_std",
            "policy_aux",
        ):
            remember("rollout " + name, getattr(self.buffer, name))
        for name in (
            "_update_last_values",
            "_update_summary",
            "_update_metrics",
            "_raw_update_returns",
            "_value_popart_old_mean",
            "_value_popart_old_scale",
            "_policy_scaled_obs",
            "_bootstrap_actions",
            "_bootstrap_log_probs",
            "_update_scaled_obs",
            "_epoch_shuffled_obs",
            "_epoch_shuffled_actions",
            "_epoch_shuffled_log_probs",
            "_epoch_shuffled_values",
            "_epoch_shuffled_returns",
            "_epoch_shuffled_advantages",
            "_epoch_shuffled_policy_mean",
            "_epoch_shuffled_policy_log_std",
            "_update_eval_log_probs",
            "_update_eval_values",
            "_update_eval_entropy",
            "_update_eval_std_mean",
            "_update_eval_std_min",
            "_update_eval_std_max",
            "_update_eval_policy_mean",
            "_update_eval_policy_log_std",
            "_update_eval_raw_log_std_bound_stats",
            "_analytic_old_log_std_snapshot",
            "_lr_kl_tensor",
            "_noise_seed",
            "_policy_aux_eps",
            "_bootstrap_policy_aux",
            "_epoch_shuffled_policy_aux",
            "_update_scaled_next_obs",
            "_epoch_shuffled_next_obs",
            "_epoch_shuffled_dones",
        ):
            remember(name, getattr(self, name, None))
        if self.initial_state_learner is not None:
            remember(
                "initial-state return targets",
                self.initial_state_learner.return_targets,
            )
        for index, permutation in enumerate(self._epoch_permutations or []):
            remember(f"epoch permutation {index}", permutation)
        remember("auxiliary loss", self._update_eval_policy_aux_loss)
        saved_lrs = [
            (
                group,
                group["lr"],
                group["lr"].detach().clone() if torch.is_tensor(group["lr"]) else None,
            )
            for group in self.optimizer.param_groups
        ]
        saved_cpu_rng = torch.get_rng_state().clone()
        saved_cuda_rng = (
            torch.cuda.get_rng_state(self.device).clone()
            if self.device.type == "cuda"
            else None
        )
        saved_shuffle_rng = (
            self._shuffle_generator.get_state().clone()
            if self._shuffle_generator is not None
            else None
        )
        update_stream = self._update_torch_stream if self._use_update_graph else None
        update_wp_stream = self._update_wp_stream if self._use_update_graph else None

        def synchronize_updates() -> None:
            self._synchronize_warmup_work(update_stream)

        def restore(error: Exception | None = None) -> None:
            def attempt(operation, label: str) -> None:
                self._restore_warmup_components(((label, operation),), error=error)

            with torch.no_grad():
                for label, value, saved in saved_tensors:
                    attempt(lambda value=value, saved=saved: value.copy_(saved), label)
                attempt(
                    lambda: self._restore_optimizer_state_inplace(
                        saved_optimizer_state
                    ),
                    "optimizer",
                )
                if self.loss_scaler is not None:
                    attempt(
                        lambda: self.loss_scaler.load_state_dict(saved_pre_scaler),
                        "loss scaler",
                    )
                for label, preprocessor, state in saved_preprocessors:
                    attempt(
                        lambda preprocessor=preprocessor,
                        state=state: preprocessor.load_state_dict(state),
                        label,
                    )
                for group, original_lr, saved_lr in saved_lrs:

                    def restore_lr(
                        group=group, original_lr=original_lr, saved_lr=saved_lr
                    ):
                        if torch.is_tensor(original_lr):
                            original_lr.copy_(saved_lr)
                        group["lr"] = original_lr

                    attempt(restore_lr, "learning rate")
                for parameter, (gradient, saved) in zip(
                    parameters, saved_gradients, strict=True
                ):

                    def restore_gradient(
                        parameter=parameter, gradient=gradient, saved=saved
                    ):
                        parameter.grad = gradient
                        if gradient is not None:
                            gradient.copy_(saved)

                    attempt(restore_gradient, "parameter gradient")
            attempt(lambda: torch.set_rng_state(saved_cpu_rng), "CPU random state")
            if saved_cuda_rng is not None:
                attempt(
                    lambda: torch.cuda.set_rng_state(saved_cuda_rng, self.device),
                    "CUDA random state",
                )
            if saved_shuffle_rng is not None:
                attempt(
                    lambda: self._shuffle_generator.set_state(saved_shuffle_rng),
                    "shuffle random state",
                )

        try:
            if update_stream is not None:
                update_stream.wait_stream(torch.cuda.current_stream(self.device))
            wp_context = (
                wp.ScopedStream(update_wp_stream, sync_enter=False, sync_exit=False)
                if update_wp_stream is not None
                else nullcontext()
            )
            torch_context = (
                torch.cuda.stream(update_stream)
                if update_stream is not None
                else nullcontext()
            )
            with wp_context, torch_context:
                if self._detailed_nvtx:
                    self._run_detailed_update_prepare()
                    for epoch in range(self.learning_epochs):
                        self._run_detailed_update_epoch_prepare(epoch)
                self._update_metrics.zero_()
                self._run_update_epoch_loops(max_minibatches=max_minibatches)
                restore()
            synchronize_updates()
        except Exception as error:
            self._restore_warmup_components(
                (("pending update-stream work", synchronize_updates),), error=error
            )
            restore(error)
            raise

    def _snapshot_optimizer_state(self) -> dict[torch.Tensor, dict[str, Any]]:
        return {
            param: {
                key: value.detach().clone()
                if torch.is_tensor(value)
                else copy.deepcopy(value)
                for key, value in state.items()
            }
            for param, state in self.optimizer.state.items()
        }

    @torch.no_grad()
    def _restore_optimizer_state_inplace(
        self, saved_state: dict[torch.Tensor, dict[str, Any]]
    ) -> None:
        optimizers = (
            tuple(
                optimizer
                for optimizer in (
                    self.optimizer.muon_optimizer,
                    self.optimizer.adamw_optimizer,
                )
                if optimizer is not None
            )
            if isinstance(self.optimizer, HybridMuonAdamW)
            else (self.optimizer,)
        )
        for optimizer in optimizers:
            for parameter in tuple(optimizer.state):
                if parameter not in saved_state:
                    del optimizer.state[parameter]
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    saved_param_state = saved_state.get(parameter)
                    if saved_param_state is None:
                        continue
                    state = optimizer.state[parameter]
                    for key in tuple(state):
                        if key not in saved_param_state:
                            del state[key]
                    for key, saved_value in saved_param_state.items():
                        current_value = state.get(key)
                        if torch.is_tensor(saved_value):
                            if (
                                torch.is_tensor(current_value)
                                and current_value.shape == saved_value.shape
                                and (current_value.dtype == saved_value.dtype)
                                and (current_value.device == saved_value.device)
                            ):
                                current_value.copy_(saved_value)
                            else:
                                state[key] = saved_value.detach().clone()
                        else:
                            state[key] = copy.deepcopy(saved_value)

    def _initialize_optimizer_state_for_graph(self) -> None:
        """Materialize Adam state tensors before graph capture.

        Without this warm initialization, the first captured optimizer step can
        include one-time state-init kernels, which may replay incorrectly.
        """
        if getattr(self, "_optimizer_graph_state_initialized", False):
            return
        if self._initialize_optimizer_state_without_step():
            self._optimizer_graph_state_initialized = True
            return
        with torch.no_grad():
            self.optimizer.zero_grad(set_to_none=True)
            for param in self.network.parameters():
                if param.requires_grad:
                    param.grad = torch.zeros_like(param)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    state = self.optimizer.state.get(param)
                    if not state:
                        continue
                    step = state.get("step")
                    if torch.is_tensor(step):
                        step.zero_()
                    else:
                        state["step"] = 0
                    momentum_buffer = state.get("momentum_buffer")
                    if torch.is_tensor(momentum_buffer):
                        momentum_buffer.zero_()
                    exp_avg = state.get("exp_avg")
                    if torch.is_tensor(exp_avg):
                        exp_avg.zero_()
                    exp_avg_sq = state.get("exp_avg_sq")
                    if torch.is_tensor(exp_avg_sq):
                        exp_avg_sq.zero_()
                    max_exp_avg_sq = state.get("max_exp_avg_sq")
                    if torch.is_tensor(max_exp_avg_sq):
                        max_exp_avg_sq.zero_()
        self._optimizer_graph_state_initialized = True

    def _ensure_optimizer_state_for_update_graph(self) -> None:
        if self._use_update_graph and (not self._optimizer_graph_state_initialized):
            self._initialize_optimizer_state_for_graph()

    def _initialize_optimizer_state_without_step(self) -> bool:
        optimizer = self.optimizer
        if isinstance(optimizer, HybridMuonAdamW):
            if optimizer.muon_optimizer is not None:
                self._initialize_muon_optimizer_state(optimizer.muon_optimizer)
            if optimizer.adamw_optimizer is not None:
                self._initialize_adam_optimizer_state(optimizer.adamw_optimizer)
            return True
        if isinstance(optimizer, CapturableMuon):
            self._initialize_muon_optimizer_state(optimizer)
            return True
        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            self._initialize_adam_optimizer_state(optimizer)
            return True
        return False

    @staticmethod
    def _initialize_muon_optimizer_state(optimizer: CapturableMuon) -> None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                state = optimizer.state[param]
                state.clear()
                state["momentum_buffer"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
        optimizer.warmup_ns_kernels()

    @staticmethod
    def _initialize_adam_optimizer_state(
        optimizer: torch.optim.Adam | torch.optim.AdamW,
    ) -> None:
        for group in optimizer.param_groups:
            capturable_or_fused = bool(
                group.get("capturable", False) or group.get("fused", False)
            )
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                state = optimizer.state[param]
                state.clear()
                state["step"] = torch.zeros(
                    (),
                    dtype=torch.float32,
                    device=param.device if capturable_or_fused else "cpu",
                )
                state["exp_avg"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )
                if bool(group.get("amsgrad", False)):
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

    def _invalidate_update_graphs(self) -> None:
        self._update_graph = None
        self._detailed_update_prepare_graph = None
        if self._detailed_update_epoch_prepare_graphs is not None:
            for idx in range(len(self._detailed_update_epoch_prepare_graphs)):
                self._detailed_update_epoch_prepare_graphs[idx] = None
        if self._detailed_update_minibatch_graphs is not None:
            for row in self._detailed_update_minibatch_graphs:
                for idx in range(len(row)):
                    row[idx] = None
        self._compiled_update_capture_prewarmed = False

    @staticmethod
    def _reset_running_standard_scaler_stats(scaler: RunningStandardScaler) -> None:
        scaler.running_mean.zero_()
        scaler.running_variance.fill_(1.0)
        scaler.current_count.fill_(1.0)

    def _value_scope_parameters(self, *, models_separate: bool) -> list[nn.Parameter]:
        if not models_separate and hasattr(self.network, "value_head"):
            return list(self.network.value_head.parameters())
        value_params = [
            param
            for name, param in self.network.named_parameters()
            if name.startswith("value") or ".value" in name
        ]
        if value_params:
            return value_params
        if hasattr(self.network, "value_head"):
            return list(self.network.value_head.parameters())
        raise RuntimeError(
            "Could not resolve value parameters for trainer.curriculum.dm2gait_switch."
        )

    def _reset_optimizer_state_for_params(self, params: Sequence[nn.Parameter]) -> None:
        for param in params:
            state = self.optimizer.state.get(param)
            if not state:
                continue
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    value.zero_()
                elif isinstance(value, int | float):
                    state[key] = type(value)(0)

    def reset_value_function_for_switch(self, *, models_separate: bool) -> None:
        value_params = self._value_scope_parameters(models_separate=models_separate)
        if not models_separate and hasattr(self.network, "value_head"):
            self.network.value_head.reset_parameters()
        else:
            reset_any = False
            for module_name, module in self.network.named_modules():
                if (
                    module_name.startswith("value")
                    or ".value" in module_name
                    or module_name == "value_head"
                ) and hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                    reset_any = True
            if not reset_any and hasattr(self.network, "value_head"):
                self.network.value_head.reset_parameters()
        self._reset_running_standard_scaler_stats(self.value_scaler)
        self._reset_optimizer_state_for_params(value_params)
        self._invalidate_update_graphs()

    def _log_std_parameters(self) -> tuple[nn.Parameter, ...]:
        policy_std_parameters = getattr(self.network, "policy_std_parameters", None)
        if callable(policy_std_parameters):
            return tuple(policy_std_parameters())
        params: list[nn.Parameter] = []
        param = getattr(self.network, "log_std", None)
        if isinstance(param, nn.Parameter):
            params.append(param)
        return tuple(params)

    def _set_policy_log_std_trainable(self, trainable: bool) -> None:
        for param in self._log_std_parameters():
            param.requires_grad_(trainable)

    def freeze_policy_log_std(self) -> None:
        if self._policy_log_std_frozen:
            return
        self._policy_log_std_frozen = True
        self._set_policy_log_std_trainable(False)
        self.optimizer.zero_grad(set_to_none=True)
        self._invalidate_update_graphs()

    def set_update_mode_for_switch(self, mode: str, *, models_separate: bool) -> None:
        key = str(mode).strip().lower()
        if key not in {"full", "value_bootstrap"}:
            raise ValueError(
                f"Unsupported PPO update mode {mode!r}; expected 'full' or 'value_bootstrap'."
            )
        if key == self._update_mode:
            return
        if key == "value_bootstrap":
            value_params = self._value_scope_parameters(models_separate=models_separate)
            self._value_train_params = set(value_params)
            for param in self._all_trainable_params:
                param.requires_grad_(param in self._value_train_params)
        else:
            self._value_train_params.clear()
            for param in self._all_trainable_params:
                param.requires_grad_(True)
            if self._policy_log_std_frozen:
                self._set_policy_log_std_trainable(False)
        self._update_mode = key
        self.optimizer.zero_grad(set_to_none=True)
        self._invalidate_update_graphs()

    def _clip_grad_norm_inplace(self) -> float | None:
        if self.grad_norm_clip <= 0:
            return None
        if self.separate_grad_norm_clip:
            parameter_groups = self.network.separate_ppo_parameter_groups()
        else:
            parameter_groups = (tuple(self.network.parameters()),)
        total_norms: list[torch.Tensor] = []
        for parameters in parameter_groups:
            group_norm = self._clip_parameter_group_grad_norm_inplace(parameters)
            if group_norm is not None:
                total_norms.append(group_norm)
        return None
        return float(torch.stack(total_norms).max().detach().item())

    def _clip_parameter_group_grad_norm_inplace(
        self, parameters: tuple[nn.Parameter, ...]
    ) -> torch.Tensor | None:
        grads = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        if not grads:
            return None
        total_norm_sq = torch.zeros((), device=self.device)
        for grad in grads:
            total_norm_sq = total_norm_sq + torch.sum(grad * grad)
        total_norm = torch.sqrt(total_norm_sq)
        clip_coef = torch.clamp(self.grad_norm_clip / (total_norm + 1e-06), max=1.0)
        for grad in grads:
            grad.mul_(clip_coef)
        return total_norm

    def _evaluate_update_policy(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_policy_aux: torch.Tensor | None,
        new_policy_mean: torch.Tensor | None,
        new_policy_log_std: torch.Tensor | None,
        raw_log_std_bound_stats: torch.Tensor,
        eval_raw_log_std_bound_stats: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """One minibatch's policy evaluation, shared by the PPO step and the
        old-policy recomputation so both see the same kernels and precision."""
        if self._compiled_update_uses_return_path:
            replay_kwargs = {}
            if self._uses_pre_squash_action_replay:
                replay_kwargs["pre_squash_actions"] = batch_actions
                batch_actions = batch_actions.tanh()

            if self._update_uses_aux_evaluate:
                aux_kwargs = {}
                if batch_policy_aux is not None:
                    aux_kwargs["policy_aux"] = batch_policy_aux
                if new_policy_mean is not None:
                    aux_kwargs["out_policy_mean"] = new_policy_mean
                    aux_kwargs["out_policy_log_std"] = new_policy_log_std
                aux_kwargs["out_raw_log_std_bound_stats"] = eval_raw_log_std_bound_stats
                (
                    new_log_probs,
                    predicted_values,
                    entropy,
                    policy_aux_loss,
                    std_actual_mean,
                    std_actual_min,
                    std_actual_max,
                ) = self._evaluate_with_aux_std_fn(
                    batch_obs, batch_actions, **aux_kwargs, **replay_kwargs
                )
            else:
                if batch_policy_aux is None:
                    (
                        new_log_probs,
                        predicted_values,
                        entropy,
                        std_actual_mean,
                        std_actual_min,
                        std_actual_max,
                    ) = self._evaluate_with_std_fn(
                        batch_obs,
                        batch_actions,
                        out_policy_mean=new_policy_mean,
                        out_policy_log_std=new_policy_log_std,
                        out_raw_log_std_bound_stats=eval_raw_log_std_bound_stats,
                        **replay_kwargs,
                    )
                else:
                    (
                        new_log_probs,
                        predicted_values,
                        entropy,
                        std_actual_mean,
                        std_actual_min,
                        std_actual_max,
                    ) = self._evaluate_with_std_fn(
                        batch_obs,
                        batch_actions,
                        out_policy_mean=new_policy_mean,
                        out_policy_log_std=new_policy_log_std,
                        out_raw_log_std_bound_stats=eval_raw_log_std_bound_stats,
                        policy_aux=batch_policy_aux,
                        **replay_kwargs,
                    )
                policy_aux_loss = self.network.zero_scalar()
        else:
            self._update_eval_log_probs.detach_()
            self._update_eval_values.detach_()
            self._update_eval_entropy.detach_()
            self._update_eval_std_mean.detach_()
            self._update_eval_std_min.detach_()
            self._update_eval_std_max.detach_()
            self._update_eval_policy_aux_loss.detach_()
            new_log_probs = self._update_eval_log_probs
            predicted_values = self._update_eval_values
            entropy = self._update_eval_entropy
            std_actual_mean = self._update_eval_std_mean
            std_actual_min = self._update_eval_std_min
            std_actual_max = self._update_eval_std_max
            policy_aux_loss = self._update_eval_policy_aux_loss
            self._evaluate_update_batch(
                batch_obs,
                batch_actions,
                new_log_probs,
                predicted_values,
                entropy,
                policy_aux_loss,
                std_actual_mean,
                std_actual_min,
                std_actual_max,
                eval_raw_log_std_bound_stats,
                batch_policy_aux=batch_policy_aux,
                out_policy_mean=new_policy_mean,
                out_policy_log_std=new_policy_log_std,
            )
        return (
            new_log_probs,
            predicted_values,
            entropy,
            policy_aux_loss,
            std_actual_mean,
            std_actual_min,
            std_actual_max,
        )

    @torch.no_grad()
    def _evaluate_old_policy_minibatch(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_policy_aux: torch.Tensor | None,
        out_log_probs: torch.Tensor,
        out_policy_mean: torch.Tensor | None,
        out_policy_log_std: torch.Tensor | None,
    ) -> None:
        """Write the pre-update policy's log-probabilities of one minibatch."""
        raw_log_std_bound_stats = self._update_eval_raw_log_std_bound_stats
        assert raw_log_std_bound_stats is not None
        with self._update_autocast_ctx():
            new_log_probs, *_ = self._evaluate_update_policy(
                batch_obs,
                batch_actions,
                batch_policy_aux,
                self._update_eval_policy_mean,
                self._update_eval_policy_log_std,
                raw_log_std_bound_stats,
                raw_log_std_bound_stats
                if bool(getattr(self.network, "_uses_state_dependent_sigma", False))
                else None,
            )
        out_log_probs.copy_(new_log_probs.float())
        if out_policy_mean is not None:
            assert self._update_eval_policy_mean is not None
            out_policy_mean.copy_(self._update_eval_policy_mean)
        if out_policy_log_std is not None:
            assert self._update_eval_policy_log_std is not None
            out_policy_log_std.copy_(self._update_eval_policy_log_std)

    @torch.no_grad()
    def _recompute_old_policy(self, scaled_obs: torch.Tensor) -> None:
        """Evaluate the old policy through the update path before optimizing.

        Collection and learning can select different kernels even at the same
        precision. Recompute likelihoods and distribution snapshots with the
        update's minibatches so an unchanged policy starts with unit ratios.
        """
        if self._update_mode == "value_bootstrap":
            return
        flat_actions, flat_log_probs, flat_values, flat_returns, flat_advantages = (
            self._flat_update_batch_views()
        )
        flat_policy_aux = self._flat_update_policy_aux_view()
        flat_policy_mean = self._flat_policy_mean_view()
        flat_policy_log_std = self._flat_policy_log_std_view()
        permutation = self._permutation_for_epoch(0)
        epoch_obs, epoch_actions, epoch_log_probs, _, _, _ = (
            self._prepare_epoch_batch_views(
                scaled_obs,
                flat_actions,
                flat_log_probs,
                flat_values,
                flat_returns,
                flat_advantages,
                permutation,
            )
        )
        epoch_policy_aux = self._prepare_epoch_policy_aux_view(
            flat_policy_aux, permutation
        )
        epoch_policy_mean, epoch_policy_log_std = (
            self._prepare_epoch_policy_distribution_views(
                flat_policy_mean, flat_policy_log_std, permutation
            )
        )
        for batch_slice in self.buffer.get_batches(self.mini_batches):
            self._detach_update_eval_buffers()
            self._evaluate_old_policy_minibatch_fn(
                epoch_obs[batch_slice],
                epoch_actions[batch_slice],
                epoch_policy_aux[batch_slice] if epoch_policy_aux is not None else None,
                epoch_log_probs[batch_slice],
                epoch_policy_mean[batch_slice]
                if epoch_policy_mean is not None
                else None,
                epoch_policy_log_std[batch_slice]
                if epoch_policy_log_std is not None
                else None,
            )
        if permutation is None:
            return
        flat_log_probs.index_put_((permutation,), epoch_log_probs)
        if flat_policy_mean is not None and epoch_policy_mean is not None:
            flat_policy_mean.index_put_((permutation,), epoch_policy_mean)
        if flat_policy_log_std is not None and epoch_policy_log_std is not None:
            flat_policy_log_std.index_put_((permutation,), epoch_policy_log_std)

    def _run_update_minibatch(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        batch_policy_aux: torch.Tensor | None,
        out_metrics: torch.Tensor,
        batch_next_obs: torch.Tensor | None = None,
        batch_dones: torch.Tensor | None = None,
        old_policy_mean: torch.Tensor | None = None,
        old_policy_log_std: torch.Tensor | None = None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
    ) -> None:
        batch_size = batch_obs.shape[0]
        if batch_size != self._update_batch_size:
            raise ValueError(
                f"PPO update minibatch size changed unexpectedly: expected {self._update_batch_size}, got {batch_size}."
            )
        if out_raw_log_std_bound_stats is None:
            raise RuntimeError("PPO update requires a log-std statistics buffer.")
        new_policy_mean = out_policy_mean
        new_policy_log_std = out_policy_log_std
        raw_log_std_bound_stats = out_raw_log_std_bound_stats
        eval_raw_log_std_bound_stats = (
            raw_log_std_bound_stats
            if bool(getattr(self.network, "_uses_state_dependent_sigma", False))
            else None
        )
        with self._nvtx_range("ppo_minibatch_evaluate"):
            with self._update_autocast_ctx():
                (
                    new_log_probs,
                    predicted_values,
                    entropy,
                    policy_aux_loss,
                    std_actual_mean,
                    std_actual_min,
                    std_actual_max,
                ) = self._evaluate_update_policy(
                    batch_obs,
                    batch_actions,
                    batch_policy_aux,
                    new_policy_mean,
                    new_policy_log_std,
                    raw_log_std_bound_stats,
                    eval_raw_log_std_bound_stats,
                )
        new_log_probs = new_log_probs.float()
        predicted_values = predicted_values.float()
        entropy = entropy.float()
        std_actual_mean = std_actual_mean.float()
        std_actual_min = std_actual_min.float()
        std_actual_max = std_actual_max.float()
        raw_log_std_bound_stats = raw_log_std_bound_stats.float()
        log_std_bound_loss = raw_log_std_bound_stats[4]
        policy_aux_loss = policy_aux_loss.float()
        if self._use_transition_aux_regularization:
            assert batch_next_obs is not None
            assert batch_dones is not None
            with self._nvtx_range("ppo_transition_aux_regularization"):
                with self._update_autocast_ctx():
                    transition_aux_loss = (
                        self.network.transition_aux_regularization_loss(
                            batch_obs, batch_next_obs, batch_dones
                        )
                    )
            policy_aux_loss = policy_aux_loss + transition_aux_loss.float()
        else:
            transition_aux_loss = self.network.zero_scalar()
        extra_update_metrics = self._compute_extra_update_metrics(
            batch_obs=batch_obs,
            batch_actions=(
                batch_actions.tanh()
                if self._uses_pre_squash_action_replay
                else batch_actions
            ),
            policy_aux_loss=policy_aux_loss - transition_aux_loss.float(),
            transition_aux_loss=transition_aux_loss.float(),
            batch_next_obs=batch_next_obs,
            batch_dones=batch_dones,
        )
        old_log_probs = old_log_probs.float()
        old_values = old_values.float()
        returns = returns.float()
        advantages = advantages.float()
        with self._nvtx_range("ppo_minibatch_loss"):
            ratio_delta = new_log_probs - old_log_probs
            kl = self._compute_kl_metric(
                ratio_delta=ratio_delta,
                old_policy_mean=old_policy_mean,
                old_policy_log_std=old_policy_log_std,
                new_policy_mean=new_policy_mean,
                new_policy_log_std=new_policy_log_std,
            )
            policy_loss, value_loss, entropy_loss, clip_fraction = ppo_loss_terms(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                predicted_values=predicted_values,
                old_values=old_values,
                returns=returns,
                advantages=advantages,
                entropy=entropy,
                ratio_clip=self.ratio_clip,
                value_clip=self.value_clip,
                clip_predicted_values=self.clip_predicted_values
                and self._update_mode != "value_bootstrap",
                value_loss_scale=self.value_loss_scale,
                entropy_loss_scale=self.entropy_loss_scale,
                value_loss_clip_mode=self.value_loss_clip_mode,
                zero_scalar=self.network.zero_scalar(),
            )
            if (
                self.kl_scheduler_update_mode == "per_minibatch"
                and self.kl_threshold > 0.0
            ):
                self._update_learning_rate(kl)
            if self._update_mode == "value_bootstrap":
                policy_loss = self.network.zero_scalar()
                entropy_loss = self.network.zero_scalar()
                policy_aux_loss = self.network.zero_scalar()
                log_std_bound_loss = self.network.zero_scalar()
        with self._nvtx_range("ppo_minibatch_backward"):
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = (
                policy_loss
                + value_loss
                + entropy_loss
                + policy_aux_loss
                + log_std_bound_loss
            )
            if self.loss_scaler is None:
                total_loss.backward()
            else:
                self.loss_scaler.scale_loss(total_loss).backward()
                self.loss_scaler.unscale_(self.optimizer)
            self._clip_grad_norm_inplace()
        with self._nvtx_range("ppo_minibatch_optimizer_step"):
            if self.loss_scaler is None:
                self.optimizer.step()
            else:
                self.loss_scaler.step(self.optimizer)
        out_metrics[0].copy_(policy_loss.detach())
        out_metrics[1].copy_(value_loss.detach())
        out_metrics[2].copy_(entropy_loss.detach())
        out_metrics[3].copy_(policy_aux_loss.detach())
        out_metrics[_UPDATE_KL_METRIC_INDEX].copy_(kl.detach())
        out_metrics[5].copy_(clip_fraction.detach())
        out_metrics[6].copy_(std_actual_mean.detach())
        out_metrics[7].copy_(std_actual_min.detach())
        out_metrics[8].copy_(std_actual_max.detach())
        out_metrics[9].copy_(raw_log_std_bound_stats[0].detach())
        out_metrics[10].copy_(raw_log_std_bound_stats[1].detach())
        out_metrics[11].copy_(raw_log_std_bound_stats[2].detach())
        out_metrics[12].copy_(raw_log_std_bound_stats[3].detach())
        out_metrics[13].copy_(log_std_bound_loss.detach())
        extra_metric_offset = _UPDATE_EXTRA_METRIC_OFFSET
        for extra_idx, extra_metric in enumerate(extra_update_metrics):
            out_metrics[extra_metric_offset + extra_idx].copy_(extra_metric.detach())

    @torch.no_grad()
    def _store_time_limit_bootstrap_mask(
        self, step: int, timeout_mask: torch.Tensor
    ) -> bool:
        self._timeout_bootstrap_mask[int(step)].copy_(
            timeout_mask.reshape(self.num_envs, 1)
        )
        self._timeout_bootstrap_values[int(step)].copy_(self.buffer.values[int(step)])
        self._keep_timeout_bootstrap_values(int(step))
        return True

    def _keep_timeout_bootstrap_values(self, step: int) -> None:
        torch.le(
            self._timeout_bootstrap_mask[step],
            0.0,
            out=self._timeout_bootstrap_dropped[step],
        )
        self._timeout_bootstrap_values[step].masked_fill_(
            self._timeout_bootstrap_dropped[step], 0.0
        )

    @torch.no_grad()
    def _store_time_limit_bootstrap_values(
        self, step: int, terminal_obs: torch.Tensor, timeout_mask: torch.Tensor
    ) -> bool:
        step = int(step)
        self._timeout_bootstrap_mask[step].copy_(timeout_mask.reshape(self.num_envs, 1))
        self.bootstrap_values_into(
            terminal_obs, out_values=self._timeout_bootstrap_values[step]
        )
        self._keep_timeout_bootstrap_values(step)
        return True

    @torch.no_grad()
    def _refresh_update_rewards(self, *, update_stats: bool = True) -> None:
        if self.reward_preprocessor is None:
            self._update_rewards.copy_(self.buffer.rewards)
        else:
            if update_stats:
                valid_mask = torch.isfinite(self.buffer.rewards).all(
                    dim=-1, keepdim=True
                ) & torch.isfinite(self.buffer.dones).all(dim=-1, keepdim=True)
                self.reward_preprocessor.update_rollout(
                    rewards=self.buffer.rewards,
                    dones=self.buffer.dones,
                    valid_mask=valid_mask,
                )
            self._update_rewards.copy_(
                self.reward_preprocessor.normalize(self.buffer.rewards)
            )
        self._update_rewards.add_(
            self._timeout_bootstrap_values, alpha=self.discount_factor
        )

    @torch.no_grad()
    def _prepare_update_views(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self.bootstrap_values_into(
            self.buffer.obs[self.rollouts], out_values=self._update_last_values
        )
        self.buffer.compute_returns(
            last_values=self._update_last_values,
            gamma=self.discount_factor,
            lam=self.lam,
            wp_stream=self._update_wp_stream,
            rewards=self._update_rewards,
            rewards_wp=self._update_rewards_wp,
        )
        if self.initial_state_learner is not None:
            self.initial_state_learner.capture_returns(self.buffer.returns)
        self._prepare_value_update_coordinates()
        flat_obs = self.buffer.obs[: self.rollouts].reshape(-1, self.obs_dim)
        # Reuse the action batching/shuffling path for exact tanh replay.
        action_source = (
            self.buffer.pre_squash_actions
            if self.buffer.pre_squash_actions is not None
            else self.buffer.actions
        )
        flat_actions = action_source.reshape(-1, self.act_dim)
        flat_log_probs = self.buffer.log_probs.reshape(-1, 1)
        flat_values = self.buffer.values.reshape(-1, 1)
        flat_returns = self.buffer.returns.reshape(-1, 1)
        flat_advantages = self.buffer.advantages.reshape(-1, 1)
        return (
            flat_obs,
            flat_actions,
            flat_log_probs,
            flat_values,
            flat_returns,
            flat_advantages,
        )

    @torch.no_grad()
    def _prepare_update_views_compiled(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._obs_scale_eval_into_fn(
            self.buffer.obs[self.rollouts], self._policy_scaled_obs
        )
        self._act_into_fn(
            self._policy_scaled_obs,
            self._bootstrap_actions,
            self._bootstrap_log_probs,
            self._update_last_values,
            self._bootstrap_eps,
            policy_aux=self._bootstrap_policy_aux,
        )
        self._inverse_scale_values_(self._update_last_values)
        return self._prepare_update_views_post_bootstrap_compiled()

    @torch.no_grad()
    def _prepare_update_views_post_bootstrap_compiled(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        advantage = torch.zeros_like(self._update_last_values)
        for i in reversed(range(self.rollouts)):
            next_values = (
                self.buffer.values[i + 1]
                if i < self.rollouts - 1
                else self._update_last_values
            )
            advantage = (
                self._update_rewards[i]
                - self.buffer.values[i]
                + self.discount_factor
                * (1.0 - self.buffer.dones[i])
                * (next_values + self.lam * advantage)
            )
            self.buffer.advantages[i].copy_(advantage)
        self.buffer.returns.copy_(self.buffer.advantages + self.buffer.values)
        if self.initial_state_learner is not None:
            self.initial_state_learner.capture_returns(self.buffer.returns)
        self.buffer.advantages.sub_(self.buffer.advantages.mean()).div_(
            self.buffer.advantages.std(
                correction=1 if self.buffer.advantages.numel() > 1 else 0
            )
            + 1e-08
        )
        self._prepare_value_update_coordinates()
        flat_obs = self.buffer.obs[: self.rollouts].reshape(-1, self.obs_dim)
        # Reuse the action batching/shuffling path for exact tanh replay.
        action_source = (
            self.buffer.pre_squash_actions
            if self.buffer.pre_squash_actions is not None
            else self.buffer.actions
        )
        flat_actions = action_source.reshape(-1, self.act_dim)
        flat_log_probs = self.buffer.log_probs.reshape(-1, 1)
        flat_values = self.buffer.values.reshape(-1, 1)
        flat_returns = self.buffer.returns.reshape(-1, 1)
        flat_advantages = self.buffer.advantages.reshape(-1, 1)
        return (
            flat_obs,
            flat_actions,
            flat_log_probs,
            flat_values,
            flat_returns,
            flat_advantages,
        )

    def _prepare_scaled_obs_for_update(self, flat_obs: torch.Tensor) -> torch.Tensor:
        policy_obs = self.buffer.policy_obs
        if (
            self._observation_normalization_contract.requires_policy_input_snapshot
            and policy_obs is None
        ):
            raise RuntimeError(
                "observation normalization contract requires rollout policy input snapshots, but the rollout buffer did not allocate them"
            )
        if policy_obs is not None:
            flat_policy_obs = policy_obs.reshape(-1, self.obs_dim)
            if self._use_update_graph:
                assert self._update_scaled_obs is not None
                self._update_scaled_obs.copy_(flat_policy_obs)
                return self._update_scaled_obs
            return flat_policy_obs
        if self._use_update_graph:
            assert self._update_scaled_obs is not None
            self._obs_scale_update_into_fn(flat_obs, self._update_scaled_obs)
            return self._update_scaled_obs
        scaled_obs = torch.empty_like(flat_obs, dtype=torch.float32, device=self.device)
        self._obs_scale_update_into(flat_obs, scaled_obs)
        return scaled_obs

    def _prepare_scaled_next_obs_for_update(
        self, flat_next_obs: torch.Tensor
    ) -> torch.Tensor | None:
        if not self._use_transition_aux_regularization:
            return None
        if self._use_update_graph:
            assert self._update_scaled_next_obs is not None
            self._obs_scale_eval_into_fn(flat_next_obs, self._update_scaled_next_obs)
            return self._update_scaled_next_obs
        scaled_next_obs = torch.empty_like(
            flat_next_obs, dtype=torch.float32, device=self.device
        )
        self._obs_scale_eval_into(flat_next_obs, scaled_next_obs)
        return scaled_next_obs

    @torch.no_grad()
    def policy_mean(self, obs: torch.Tensor) -> torch.Tensor:
        with network_matmul_precision():
            scaled_obs = self._scaled_eval_obs(obs)
            return self.network.policy_mean(scaled_obs)

    @torch.no_grad()
    def policy_mean_into(self, obs: torch.Tensor, out_actions: torch.Tensor) -> None:
        with network_matmul_precision():
            self._policy_mean_into_unscoped(obs, out_actions)

    def _policy_mean_into_unscoped(
        self, obs: torch.Tensor, out_actions: torch.Tensor
    ) -> None:
        scaled_obs = self._scaled_eval_obs(obs)
        self.network.policy_mean_into(scaled_obs, out_actions)

    @torch.no_grad()
    def policy_execution_mean_into(
        self, obs: torch.Tensor, out_actions: torch.Tensor
    ) -> None:
        with network_matmul_precision():
            self._policy_execution_mean_into_unscoped(obs, out_actions)

    def _policy_execution_mean_into_unscoped(
        self, obs: torch.Tensor, out_actions: torch.Tensor
    ) -> None:
        scaled_obs = self._scaled_eval_obs(obs)
        self.network.policy_execution_mean_into(scaled_obs, out_actions)

    @torch.no_grad()
    def _generate_epoch_permutations(
        self, generator: torch.Generator | None = None
    ) -> None:
        if self._epoch_permutations is None:
            return
        if self.shuffle_mode == "per_epoch":
            for permutation in self._epoch_permutations:
                torch.randperm(
                    self._flat_update_size, generator=generator, out=permutation
                )
            return
        torch.randperm(
            self._flat_update_size, generator=generator, out=self._epoch_permutations[0]
        )

    def _capture_shuffle_graph(self) -> None:
        assert self._shuffle_generator is not None
        assert self._update_torch_stream is not None
        self._update_torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._update_torch_stream):
            with torch.cuda.nvtx.range("warmup_shuffle_graph"):
                for _ in range(3):
                    self._generate_epoch_permutations(generator=self._shuffle_generator)
        self._update_torch_stream.synchronize()
        shuffle_graph = torch.cuda.CUDAGraph()
        shuffle_graph.register_generator_state(self._shuffle_generator)
        with torch.cuda.nvtx.range("capture_shuffle_graph"):
            with torch.cuda.graph(shuffle_graph, stream=self._update_torch_stream):
                self._generate_epoch_permutations(generator=self._shuffle_generator)
        self._shuffle_graph = shuffle_graph
        self._update_torch_stream.synchronize()
        _LOGGER.info(
            "Captured minibatch shuffle graph (%s, permutations=%d, dtype=int32).",
            self.shuffle_mode,
            self._shuffle_permutation_count,
        )

    def _ensure_shuffle_graph(self) -> None:
        if not self._use_shuffle_graph or self._shuffle_graph is not None:
            return
        self._capture_shuffle_graph()

    def invalidate_cuda_graphs(self) -> None:
        if self._update_wp_stream is not None:
            wp.synchronize_stream(self._update_wp_stream)
        if self._update_torch_stream is not None:
            self._update_torch_stream.synchronize()
        self._invalidate_update_graphs()
        self._shuffle_graph = None
        if self._use_shuffle_graph:
            self._capture_shuffle_graph()

    @torch.no_grad()
    def _refresh_epoch_permutations(self) -> None:
        if self._epoch_permutations is None:
            return
        self._ensure_shuffle_graph()
        if self._shuffle_graph is not None:
            with torch.cuda.nvtx.range("replay_shuffle_graph"):
                self._shuffle_graph.replay()
            return
        with torch.cuda.nvtx.range("generate_epoch_permutations"):
            self._generate_epoch_permutations()

    def _permutation_for_epoch(self, epoch: int) -> torch.Tensor | None:
        if self._epoch_permutations is None:
            return None
        if self.shuffle_mode == "per_epoch":
            return self._epoch_permutations[epoch]
        return self._epoch_permutations[0]

    def _prepare_epoch_batch_views(
        self,
        scaled_obs: torch.Tensor,
        flat_actions: torch.Tensor,
        flat_log_probs: torch.Tensor,
        flat_values: torch.Tensor,
        flat_returns: torch.Tensor,
        flat_advantages: torch.Tensor,
        permutation: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if permutation is None:
            return (
                scaled_obs,
                flat_actions,
                flat_log_probs,
                flat_values,
                flat_returns,
                flat_advantages,
            )
        assert self._epoch_shuffled_obs is not None
        assert self._epoch_shuffled_actions is not None
        assert self._epoch_shuffled_log_probs is not None
        assert self._epoch_shuffled_values is not None
        assert self._epoch_shuffled_returns is not None
        assert self._epoch_shuffled_advantages is not None
        torch.index_select(scaled_obs, 0, permutation, out=self._epoch_shuffled_obs)
        torch.index_select(
            flat_actions, 0, permutation, out=self._epoch_shuffled_actions
        )
        torch.index_select(
            flat_log_probs, 0, permutation, out=self._epoch_shuffled_log_probs
        )
        torch.index_select(flat_values, 0, permutation, out=self._epoch_shuffled_values)
        torch.index_select(
            flat_returns, 0, permutation, out=self._epoch_shuffled_returns
        )
        torch.index_select(
            flat_advantages, 0, permutation, out=self._epoch_shuffled_advantages
        )
        return (
            self._epoch_shuffled_obs,
            self._epoch_shuffled_actions,
            self._epoch_shuffled_log_probs,
            self._epoch_shuffled_values,
            self._epoch_shuffled_returns,
            self._epoch_shuffled_advantages,
        )

    def _prepare_epoch_policy_aux_view(
        self, flat_policy_aux: torch.Tensor | None, permutation: torch.Tensor | None
    ) -> torch.Tensor | None:
        if flat_policy_aux is None or permutation is None:
            return flat_policy_aux
        assert self._epoch_shuffled_policy_aux is not None
        torch.index_select(
            flat_policy_aux, 0, permutation, out=self._epoch_shuffled_policy_aux
        )
        return self._epoch_shuffled_policy_aux

    def _sync_update_graph_inputs(self) -> None:
        """Chain the update stream behind the caller's current stream."""
        assert self._update_torch_stream is not None
        producer_stream = torch.cuda.current_stream(self.device)
        if producer_stream.cuda_stream != self._update_torch_stream.cuda_stream:
            self._update_torch_stream.wait_stream(producer_stream)

    def _record_update_timing_start(self) -> None:
        if self._update_start_event is None or self._update_torch_stream is None:
            return
        self._update_start_event.record(self._update_torch_stream)
        self._update_timing_pending = True

    def _record_update_timing_end(self) -> None:
        if self._update_end_event is None or self._update_torch_stream is None:
            return
        self._update_end_event.record(self._update_torch_stream)

    def _consume_update_timing_ns(self) -> int:
        if (
            not self._update_timing_pending
            or self._update_start_event is None
            or self._update_end_event is None
        ):
            return 0
        self._update_timing_pending = False
        self._update_end_event.synchronize()
        return int(
            self._update_start_event.elapsed_time(self._update_end_event) * 1000000.0
        )

    def _has_detailed_update_graphs(self) -> bool:
        if (
            self._detailed_update_prepare_graph is None
            or self._detailed_update_minibatch_graphs is None
        ):
            return False
        if any(
            graph is None
            for row in self._detailed_update_minibatch_graphs
            for graph in row
        ):
            return False
        return not (
            self._detailed_update_epoch_prepare_graphs is not None
            and any(
                graph is None for graph in self._detailed_update_epoch_prepare_graphs
            )
        )

    def _update_batch_slice(self, batch_idx: int) -> slice:
        start = batch_idx * self._update_batch_size
        end = (batch_idx + 1) * self._update_batch_size
        return slice(start, end)

    def _flat_update_batch_views(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reuse the action batching/shuffling path for exact tanh replay.
        action_source = (
            self.buffer.pre_squash_actions
            if self.buffer.pre_squash_actions is not None
            else self.buffer.actions
        )
        flat_actions = action_source.reshape(-1, self.act_dim)
        return (
            flat_actions,
            self.buffer.log_probs.reshape(-1, 1),
            self.buffer.values.reshape(-1, 1),
            self.buffer.returns.reshape(-1, 1),
            self.buffer.advantages.reshape(-1, 1),
        )

    def _flat_update_policy_aux_view(self) -> torch.Tensor | None:
        if self.buffer.policy_aux is None:
            return None
        return self.buffer.policy_aux.reshape(-1, self._policy_aux_dim)

    def _flat_policy_mean_view(self) -> torch.Tensor | None:
        if self.buffer.policy_mean is None:
            return None
        return self.buffer.policy_mean.reshape(-1, self.act_dim)

    def _flat_policy_log_std_view(self) -> torch.Tensor | None:
        if self.buffer.policy_log_std is None:
            return None
        return self.buffer.policy_log_std.reshape(-1, self.act_dim)

    def _prepare_epoch_policy_distribution_views(
        self,
        flat_policy_mean: torch.Tensor | None,
        flat_policy_log_std: torch.Tensor | None,
        permutation: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if permutation is None:
            return (flat_policy_mean, flat_policy_log_std)
        policy_mean = None
        policy_log_std = None
        if flat_policy_mean is not None:
            assert self._epoch_shuffled_policy_mean is not None
            torch.index_select(
                flat_policy_mean, 0, permutation, out=self._epoch_shuffled_policy_mean
            )
            policy_mean = self._epoch_shuffled_policy_mean
        if flat_policy_log_std is not None:
            assert self._epoch_shuffled_policy_log_std is not None
            torch.index_select(
                flat_policy_log_std,
                0,
                permutation,
                out=self._epoch_shuffled_policy_log_std,
            )
            policy_log_std = self._epoch_shuffled_policy_log_std
        return (policy_mean, policy_log_std)

    @torch.no_grad()
    def _snapshot_analytic_kl_log_std(self) -> None:
        if self._analytic_old_log_std_snapshot is None:
            return
        policy_log_std_vector = getattr(self.network, "policy_log_std_vector", None)
        if not callable(policy_log_std_vector):
            raise RuntimeError(
                "agent.kl_estimator='analytic' requires the policy network to expose policy_log_std_vector()."
            )
        self._analytic_old_log_std_snapshot.copy_(
            policy_log_std_vector().detach().float()
        )

    @torch.no_grad()
    def _compute_kl_metric(
        self,
        *,
        ratio_delta: torch.Tensor,
        old_policy_mean: torch.Tensor | None,
        old_policy_log_std: torch.Tensor | None,
        new_policy_mean: torch.Tensor | None,
        new_policy_log_std: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self._uses_analytic_kl:
            return (ratio_delta.exp() - 1.0 - ratio_delta).mean()
        if old_policy_mean is None:
            raise RuntimeError(
                "agent.kl_estimator='analytic' requires stored rollout policy means."
            )
        if new_policy_mean is None or new_policy_log_std is None:
            raise RuntimeError(
                "agent.kl_estimator='analytic' requires policy distribution parameters from the update evaluation."
            )
        if old_policy_log_std is None:
            if self._analytic_old_log_std_snapshot is None:
                raise RuntimeError(
                    "agent.kl_estimator='analytic' requires old log_std data."
                )
            old_policy_log_std = self._analytic_old_log_std_snapshot.view(1, -1)
        return _diagonal_gaussian_kl_mean(
            old_policy_mean, old_policy_log_std, new_policy_mean, new_policy_log_std
        )

    def _flat_transition_aux_views(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if not self._use_transition_aux_regularization:
            return (None, None)
        return (
            self.buffer.obs[1 : self.rollouts + 1].reshape(-1, self.obs_dim),
            self.buffer.dones.reshape(-1, 1),
        )

    def _prepare_epoch_transition_aux_views(
        self,
        scaled_next_obs: torch.Tensor | None,
        flat_dones: torch.Tensor | None,
        permutation: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._use_transition_aux_regularization:
            return (None, None)
        assert scaled_next_obs is not None
        assert flat_dones is not None
        if permutation is None:
            return (scaled_next_obs, flat_dones)
        assert self._epoch_shuffled_next_obs is not None
        assert self._epoch_shuffled_dones is not None
        torch.index_select(
            scaled_next_obs, 0, permutation, out=self._epoch_shuffled_next_obs
        )
        torch.index_select(flat_dones, 0, permutation, out=self._epoch_shuffled_dones)
        return (self._epoch_shuffled_next_obs, self._epoch_shuffled_dones)

    @torch.no_grad()
    def _run_detailed_update_prepare(self) -> None:
        flat_obs = self.buffer.obs[: self.rollouts].reshape(-1, self.obs_dim)
        self._prepare_update_views()
        assert self._update_scaled_obs is not None
        flat_next_obs, flat_dones = self._flat_transition_aux_views()
        # Freeze both transition endpoints before current observations advance
        # the batch normalizer's moments for the next rollout.
        scaled_next_obs = (
            self._prepare_scaled_next_obs_for_update(flat_next_obs)
            if flat_next_obs is not None
            else None
        )
        scaled_obs = self._prepare_scaled_obs_for_update(flat_obs)
        if self.shuffle_mode == "per_update":
            flat_actions, flat_log_probs, flat_values, flat_returns, flat_advantages = (
                self._flat_update_batch_views()
            )
            flat_policy_aux = self._flat_update_policy_aux_view()
            flat_policy_mean = self._flat_policy_mean_view()
            flat_policy_log_std = self._flat_policy_log_std_view()
            self._prepare_epoch_batch_views(
                scaled_obs,
                flat_actions,
                flat_log_probs,
                flat_values,
                flat_returns,
                flat_advantages,
                self._permutation_for_epoch(0),
            )
            self._prepare_epoch_policy_aux_view(
                flat_policy_aux, self._permutation_for_epoch(0)
            )
            self._prepare_epoch_policy_distribution_views(
                flat_policy_mean, flat_policy_log_std, self._permutation_for_epoch(0)
            )
            self._prepare_epoch_transition_aux_views(
                scaled_next_obs, flat_dones, self._permutation_for_epoch(0)
            )
        self._recompute_old_policy(scaled_obs)

    @torch.no_grad()
    def _run_detailed_update_epoch_prepare(self, epoch: int) -> None:
        if self.shuffle_mode != "per_epoch":
            return
        assert self._update_scaled_obs is not None
        permutation = self._permutation_for_epoch(epoch)
        if permutation is None:
            return
        flat_actions, flat_log_probs, flat_values, flat_returns, flat_advantages = (
            self._flat_update_batch_views()
        )
        flat_policy_aux = self._flat_update_policy_aux_view()
        flat_policy_mean = self._flat_policy_mean_view()
        flat_policy_log_std = self._flat_policy_log_std_view()
        flat_next_obs, flat_dones = self._flat_transition_aux_views()
        scaled_next_obs = (
            self._update_scaled_next_obs
            if self._use_update_graph
            else self._prepare_scaled_next_obs_for_update(flat_next_obs)
            if flat_next_obs is not None
            else None
        )
        self._prepare_epoch_batch_views(
            self._update_scaled_obs,
            flat_actions,
            flat_log_probs,
            flat_values,
            flat_returns,
            flat_advantages,
            permutation,
        )
        self._prepare_epoch_policy_aux_view(flat_policy_aux, permutation)
        self._prepare_epoch_policy_distribution_views(
            flat_policy_mean, flat_policy_log_std, permutation
        )
        self._prepare_epoch_transition_aux_views(
            scaled_next_obs, flat_dones, permutation
        )

    def _run_detailed_update_minibatch(self, epoch: int, batch_idx: int) -> None:
        batch_slice = self._update_batch_slice(batch_idx)
        if self.shuffle_mode in {"per_update", "per_epoch"}:
            assert self._epoch_shuffled_obs is not None
            assert self._epoch_shuffled_actions is not None
            assert self._epoch_shuffled_log_probs is not None
            assert self._epoch_shuffled_values is not None
            assert self._epoch_shuffled_returns is not None
            assert self._epoch_shuffled_advantages is not None
            epoch_obs = self._epoch_shuffled_obs
            epoch_actions = self._epoch_shuffled_actions
            epoch_log_probs = self._epoch_shuffled_log_probs
            epoch_values = self._epoch_shuffled_values
            epoch_returns = self._epoch_shuffled_returns
            epoch_advantages = self._epoch_shuffled_advantages
            epoch_policy_aux = self._epoch_shuffled_policy_aux
            epoch_policy_mean = self._epoch_shuffled_policy_mean
            epoch_policy_log_std = self._epoch_shuffled_policy_log_std
            epoch_next_obs = self._epoch_shuffled_next_obs
            epoch_dones = self._epoch_shuffled_dones
        else:
            (
                epoch_actions,
                epoch_log_probs,
                epoch_values,
                epoch_returns,
                epoch_advantages,
            ) = self._flat_update_batch_views()
            assert self._update_scaled_obs is not None
            epoch_obs = self._update_scaled_obs
            epoch_policy_aux = self._flat_update_policy_aux_view()
            epoch_policy_mean = self._flat_policy_mean_view()
            epoch_policy_log_std = self._flat_policy_log_std_view()
            epoch_next_obs = self._update_scaled_next_obs
            epoch_dones = (
                self.buffer.dones.reshape(-1, 1)
                if self._use_transition_aux_regularization
                else None
            )
        graph_idx = epoch * self.mini_batches + batch_idx
        self._detach_update_eval_buffers()
        self._run_update_minibatch_fn(
            epoch_obs[batch_slice],
            epoch_actions[batch_slice],
            epoch_log_probs[batch_slice],
            epoch_values[batch_slice],
            epoch_returns[batch_slice],
            epoch_advantages[batch_slice],
            epoch_policy_aux[batch_slice] if epoch_policy_aux is not None else None,
            self._update_metrics[graph_idx],
            epoch_next_obs[batch_slice] if epoch_next_obs is not None else None,
            epoch_dones[batch_slice] if epoch_dones is not None else None,
            epoch_policy_mean[batch_slice] if epoch_policy_mean is not None else None,
            epoch_policy_log_std[batch_slice]
            if epoch_policy_log_std is not None
            else None,
            self._update_eval_policy_mean,
            self._update_eval_policy_log_std,
            self._update_eval_raw_log_std_bound_stats,
        )

    def _sync_detailed_update_stream(self) -> None:
        if self._detailed_nvtx and self._update_wp_stream is not None:
            wp.synchronize_stream(self._update_wp_stream)

    def _launch_detailed_update_graph_sequence(self, *, phase: str) -> None:
        assert self._update_wp_stream is not None
        assert self._detailed_update_prepare_graph is not None
        assert self._detailed_update_minibatch_graphs is not None
        with torch.cuda.nvtx.range(f"{phase}_update_prepare_graph"):
            wp.capture_launch(
                self._detailed_update_prepare_graph, stream=self._update_wp_stream
            )
            self._sync_detailed_update_stream()
        for epoch in range(self.learning_epochs):
            if self._detailed_update_epoch_prepare_graphs is not None:
                epoch_prepare_graph = self._detailed_update_epoch_prepare_graphs[epoch]
                assert epoch_prepare_graph is not None
                with torch.cuda.nvtx.range(
                    f"{phase}_update_epoch_prepare_graph_{epoch}"
                ):
                    wp.capture_launch(
                        epoch_prepare_graph, stream=self._update_wp_stream
                    )
                    self._sync_detailed_update_stream()
            for batch_idx in range(self.mini_batches):
                minibatch_graph = self._detailed_update_minibatch_graphs[epoch][
                    batch_idx
                ]
                assert minibatch_graph is not None
                with torch.cuda.nvtx.range(
                    f"{phase}_update_minibatch_graph_{epoch}_{batch_idx}"
                ):
                    wp.capture_launch(minibatch_graph, stream=self._update_wp_stream)
                    self._sync_detailed_update_stream()
            if self.kl_scheduler_update_mode == "per_epoch" and self.kl_threshold > 0.0:
                with self._nvtx_range("ppo_update_learning_rate"):
                    kl_slice = self._update_metrics[
                        epoch * self.mini_batches : (epoch + 1) * self.mini_batches,
                        _UPDATE_KL_METRIC_INDEX,
                    ]
                    self._update_learning_rate(kl_slice.mean())
                self._sync_detailed_update_stream()
        if self.kl_scheduler_update_mode == "per_update" and self.kl_threshold > 0.0:
            with self._nvtx_range("ppo_update_learning_rate"):
                self._update_learning_rate(
                    self._update_metrics[:, _UPDATE_KL_METRIC_INDEX].mean()
                )
            self._sync_detailed_update_stream()
        with self._nvtx_range("ppo_write_update_summary"):
            self._write_update_summary_from_metrics(self._update_metrics)
        self._sync_detailed_update_stream()

    def _capture_update_graph_detailed(
        self, *, launch_after_capture: bool
    ) -> dict[str, float] | None:
        assert self._update_wp_stream is not None
        assert self._update_torch_stream is not None
        assert self._update_scaled_obs is not None
        assert self._update_metrics is not None
        assert self._detailed_update_minibatch_graphs is not None
        with self._nvtx_range("ppo_refresh_update_rewards"):
            self._refresh_update_rewards(update_stats=launch_after_capture)
        with self._nvtx_range("ppo_sync_update_stream"):
            self._sync_update_graph_inputs()
        self._ensure_optimizer_state_for_update_graph()
        self._prewarm_compiled_update_capture_path()
        captured_prepare = False
        captured_epoch_prepare = 0
        captured_minibatches = 0
        with wp.ScopedStream(self._update_wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self._update_torch_stream):
                self._update_metrics.zero_()
                with self._nvtx_range("ppo_refresh_permutations"):
                    self._refresh_epoch_permutations()
                if self._detailed_update_prepare_graph is None:
                    with torch.cuda.nvtx.range("capture_update_prepare_graph"):
                        with capture_warp_graph(
                            stream=self._update_wp_stream
                        ) as capture:
                            self._run_detailed_update_prepare()
                    self._detailed_update_prepare_graph = capture.graph
                    captured_prepare = True
                if self._detailed_update_epoch_prepare_graphs is not None:
                    for epoch in range(self.learning_epochs):
                        if (
                            self._detailed_update_epoch_prepare_graphs[epoch]
                            is not None
                        ):
                            continue
                        with (
                            torch.cuda.nvtx.range(
                                f"capture_update_epoch_prepare_graph_{epoch}"
                            ),
                            capture_warp_graph(
                                stream=self._update_wp_stream
                            ) as capture,
                        ):
                            self._run_detailed_update_epoch_prepare(epoch)
                        self._detailed_update_epoch_prepare_graphs[epoch] = (
                            capture.graph
                        )
                        captured_epoch_prepare += 1
                for epoch in range(self.learning_epochs):
                    for batch_idx in range(self.mini_batches):
                        if (
                            self._detailed_update_minibatch_graphs[epoch][batch_idx]
                            is not None
                        ):
                            continue
                        with (
                            torch.cuda.nvtx.range(
                                f"capture_update_minibatch_graph_{epoch}_{batch_idx}"
                            ),
                            capture_warp_graph(
                                stream=self._update_wp_stream
                            ) as capture,
                        ):
                            self._run_detailed_update_minibatch(epoch, batch_idx)
                        self._detailed_update_minibatch_graphs[epoch][batch_idx] = (
                            capture.graph
                        )
                        captured_minibatches += 1
                if launch_after_capture:
                    self._record_update_timing_start()
                    self._launch_detailed_update_graph_sequence(phase="launch")
                    self._record_update_timing_end()
        if captured_prepare or captured_epoch_prepare > 0 or captured_minibatches > 0:
            _LOGGER.info(
                "Captured detailed PPO update graphs (prepare=%d, epoch_prepare=%d, minibatch=%d).",
                1 if captured_prepare else 0,
                captured_epoch_prepare,
                captured_minibatches,
            )
        if not launch_after_capture:
            return None
        metrics = self._read_update_summary()
        metrics["update_gpu_time_ns"] = float(self._consume_update_timing_ns())
        return metrics

    def _replay_update_graphs_detailed(self) -> dict[str, float]:
        if self._update_wp_stream is None:
            raise RuntimeError(
                "PPO detailed update graph replay requires a Warp update stream."
            )
        if self._update_torch_stream is None:
            raise RuntimeError(
                "PPO detailed update graph replay requires a Torch update stream."
            )
        if self._update_scaled_obs is None:
            raise RuntimeError(
                "PPO detailed update graph replay requires scaled observations."
            )
        if self._update_metrics is None:
            raise RuntimeError(
                "PPO detailed update graph replay requires an update metrics buffer."
            )
        if not self._has_detailed_update_graphs():
            raise RuntimeError(
                "PPO detailed update graph replay requires captured graphs."
            )
        with self._nvtx_range("ppo_refresh_update_rewards"):
            self._refresh_update_rewards()
        with self._nvtx_range("ppo_sync_update_stream"):
            self._sync_update_graph_inputs()
        with wp.ScopedStream(self._update_wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self._update_torch_stream):
                self._update_metrics.zero_()
                with self._nvtx_range("ppo_refresh_permutations"):
                    self._refresh_epoch_permutations()
                self._record_update_timing_start()
                self._launch_detailed_update_graph_sequence(phase="replay")
                self._record_update_timing_end()
        metrics = self._read_update_summary()
        metrics["update_gpu_time_ns"] = float(self._consume_update_timing_ns())
        return metrics

    def _run_update_epoch_loops(self, *, max_minibatches: int | None = None) -> None:
        processed_minibatches = 0
        with self._nvtx_range("ppo_prepare_update_views"):
            (
                flat_obs,
                flat_actions,
                flat_log_probs,
                flat_values,
                flat_returns,
                flat_advantages,
            ) = self._prepare_update_views_fn()
            flat_policy_aux = self._flat_update_policy_aux_view()
            flat_policy_mean = self._flat_policy_mean_view()
            flat_policy_log_std = self._flat_policy_log_std_view()
            flat_next_obs, flat_dones = self._flat_transition_aux_views()
        with self._nvtx_range("ppo_scale_obs"):
            # Scale next observations before the current batch commits moments.
            scaled_next_obs = (
                self._prepare_scaled_next_obs_for_update(flat_next_obs)
                if flat_next_obs is not None
                else None
            )
            scaled_obs = self._prepare_scaled_obs_for_update(flat_obs)
        with self._nvtx_range("ppo_prepare_per_update_views"):
            per_update_views = (
                self._prepare_epoch_batch_views(
                    scaled_obs,
                    flat_actions,
                    flat_log_probs,
                    flat_values,
                    flat_returns,
                    flat_advantages,
                    self._permutation_for_epoch(0),
                )
                if self.shuffle_mode == "per_update"
                else None
            )
            per_update_policy_aux = (
                self._prepare_epoch_policy_aux_view(
                    flat_policy_aux, self._permutation_for_epoch(0)
                )
                if self.shuffle_mode == "per_update"
                else None
            )
            per_update_policy_mean, per_update_policy_log_std = (
                self._prepare_epoch_policy_distribution_views(
                    flat_policy_mean,
                    flat_policy_log_std,
                    self._permutation_for_epoch(0),
                )
                if self.shuffle_mode == "per_update"
                else (None, None)
            )
            per_update_transition_aux = (
                self._prepare_epoch_transition_aux_views(
                    scaled_next_obs, flat_dones, self._permutation_for_epoch(0)
                )
                if self.shuffle_mode == "per_update"
                else (None, None)
            )
        with self._nvtx_range("ppo_recompute_old_policy"):
            self._recompute_old_policy(scaled_obs)
        for epoch in range(self.learning_epochs):
            epoch_range_name = (
                f"ppo_epoch_{epoch}" if self._detailed_nvtx else "ppo_epoch"
            )
            with self._nvtx_range(epoch_range_name):
                if per_update_views is not None:
                    (
                        epoch_obs,
                        epoch_actions,
                        epoch_log_probs,
                        epoch_values,
                        epoch_returns,
                        epoch_advantages,
                    ) = per_update_views
                    epoch_policy_aux = per_update_policy_aux
                    epoch_policy_mean = per_update_policy_mean
                    epoch_policy_log_std = per_update_policy_log_std
                    epoch_next_obs, epoch_dones = per_update_transition_aux
                else:
                    with self._nvtx_range("ppo_prepare_epoch_views"):
                        (
                            epoch_obs,
                            epoch_actions,
                            epoch_log_probs,
                            epoch_values,
                            epoch_returns,
                            epoch_advantages,
                        ) = self._prepare_epoch_batch_views(
                            scaled_obs,
                            flat_actions,
                            flat_log_probs,
                            flat_values,
                            flat_returns,
                            flat_advantages,
                            self._permutation_for_epoch(epoch),
                        )
                        epoch_policy_aux = self._prepare_epoch_policy_aux_view(
                            flat_policy_aux, self._permutation_for_epoch(epoch)
                        )
                        epoch_policy_mean, epoch_policy_log_std = (
                            self._prepare_epoch_policy_distribution_views(
                                flat_policy_mean,
                                flat_policy_log_std,
                                self._permutation_for_epoch(epoch),
                            )
                        )
                        epoch_next_obs, epoch_dones = (
                            self._prepare_epoch_transition_aux_views(
                                scaled_next_obs,
                                flat_dones,
                                self._permutation_for_epoch(epoch),
                            )
                        )
                    if self.shuffle_mode == "off":
                        epoch_policy_mean = flat_policy_mean
                        epoch_policy_log_std = flat_policy_log_std
                self._permutation_for_epoch(
                    epoch
                ) if self.shuffle_mode == "per_epoch" else None
                for batch_idx, batch_slice in enumerate(
                    self.buffer.get_batches(self.mini_batches)
                ):
                    graph_idx = epoch * self.mini_batches + batch_idx
                    batch_range_name = (
                        f"ppo_minibatch_{batch_idx}"
                        if self._detailed_nvtx
                        else "ppo_minibatch"
                    )
                    with self._nvtx_range(batch_range_name):
                        self._detach_update_eval_buffers()
                        self._run_update_minibatch_fn(
                            epoch_obs[batch_slice],
                            epoch_actions[batch_slice],
                            epoch_log_probs[batch_slice],
                            epoch_values[batch_slice],
                            epoch_returns[batch_slice],
                            epoch_advantages[batch_slice],
                            epoch_policy_aux[batch_slice]
                            if epoch_policy_aux is not None
                            else None,
                            self._update_metrics[graph_idx],
                            epoch_next_obs[batch_slice]
                            if epoch_next_obs is not None
                            else None,
                            epoch_dones[batch_slice]
                            if epoch_dones is not None
                            else None,
                            epoch_policy_mean[batch_slice]
                            if epoch_policy_mean is not None
                            else None,
                            epoch_policy_log_std[batch_slice]
                            if epoch_policy_log_std is not None
                            else None,
                            self._update_eval_policy_mean,
                            self._update_eval_policy_log_std,
                            self._update_eval_raw_log_std_bound_stats,
                        )
                        processed_minibatches += 1
                        if max_minibatches is not None and processed_minibatches >= int(
                            max_minibatches
                        ):
                            return
                if (
                    self.kl_scheduler_update_mode == "per_epoch"
                    and self.kl_threshold > 0.0
                ):
                    with self._nvtx_range("ppo_update_learning_rate"):
                        kl_slice = self._update_metrics[
                            epoch * self.mini_batches : (epoch + 1) * self.mini_batches,
                            _UPDATE_KL_METRIC_INDEX,
                        ]
                        self._update_learning_rate(kl_slice.mean())
        if self.kl_scheduler_update_mode == "per_update" and self.kl_threshold > 0.0:
            with self._nvtx_range("ppo_update_learning_rate"):
                self._update_learning_rate(
                    self._update_metrics[:, _UPDATE_KL_METRIC_INDEX].mean()
                )
        self._write_update_summary_from_metrics(self._update_metrics)

    def _detach_update_eval_buffers(self) -> None:
        """Clear autograd history before entering the compiled minibatch graph."""
        if self._update_eval_policy_mean is not None:
            assert self._update_eval_policy_log_std is not None
            self._update_eval_policy_mean.detach_()
            self._update_eval_policy_log_std.detach_()
        self._update_eval_raw_log_std_bound_stats.detach_()
        if not bool(getattr(self.network, "_uses_state_dependent_sigma", False)):
            self._update_eval_raw_log_std_bound_stats.zero_()

    def _capture_update_graphs(self) -> dict[str, float]:
        if self._detailed_nvtx:
            metrics = self._capture_update_graph_detailed(launch_after_capture=True)
            assert metrics is not None
            return metrics
        metrics = self._capture_update_graph(launch_after_capture=True)
        assert metrics is not None
        return metrics

    def _capture_update_graph(
        self, *, launch_after_capture: bool
    ) -> dict[str, float] | None:
        if self._update_wp_stream is None:
            raise RuntimeError(
                "PPO update graph capture requires a Warp update stream."
            )
        if self._update_torch_stream is None:
            raise RuntimeError(
                "PPO update graph capture requires a Torch update stream."
            )
        if self._update_scaled_obs is None:
            raise RuntimeError("PPO update graph capture requires scaled observations.")
        if self._update_metrics is None:
            raise RuntimeError(
                "PPO update graph capture requires an update metrics buffer."
            )
        with self._nvtx_range("ppo_refresh_update_rewards"):
            self._refresh_update_rewards(update_stats=launch_after_capture)
        with self._nvtx_range("ppo_sync_update_stream"):
            self._sync_update_graph_inputs()
        self._ensure_optimizer_state_for_update_graph()
        self._prewarm_compiled_update_capture_path()
        with wp.ScopedStream(self._update_wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self._update_torch_stream):
                self._update_metrics.zero_()
                with self._nvtx_range("ppo_refresh_permutations"):
                    self._refresh_epoch_permutations()
                with torch.cuda.nvtx.range("capture_update_graph"):
                    with capture_warp_graph(stream=self._update_wp_stream) as capture:
                        self._run_update_epoch_loops()
                if launch_after_capture:
                    self._record_update_timing_start()
                    with torch.cuda.nvtx.range("launch_update_graph"):
                        wp.capture_launch(capture.graph, stream=self._update_wp_stream)
                        if self._detailed_nvtx:
                            wp.synchronize_stream(self._update_wp_stream)
                    self._record_update_timing_end()
        self._update_graph = capture.graph
        if not self._detailed_nvtx:
            wp.synchronize_stream(self._update_wp_stream)
        _LOGGER.info("Captured fused PPO update graph")
        if not launch_after_capture:
            return None
        metrics = self._read_update_summary()
        metrics["update_gpu_time_ns"] = float(self._consume_update_timing_ns())
        return metrics

    def _replay_update_graphs(self) -> dict[str, float]:
        if self._update_graph is None:
            raise RuntimeError("PPO update graph replay requires a captured graph.")
        if self._update_wp_stream is None:
            raise RuntimeError("PPO update graph replay requires a Warp update stream.")
        if self._update_torch_stream is None:
            raise RuntimeError(
                "PPO update graph replay requires a Torch update stream."
            )
        if self._update_scaled_obs is None:
            raise RuntimeError("PPO update graph replay requires scaled observations.")
        if self._update_metrics is None:
            raise RuntimeError(
                "PPO update graph replay requires an update metrics buffer."
            )
        with self._nvtx_range("ppo_refresh_update_rewards"):
            self._refresh_update_rewards()
        with self._nvtx_range("ppo_sync_update_stream"):
            self._sync_update_graph_inputs()
        with wp.ScopedStream(self._update_wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(self._update_torch_stream):
                self._update_metrics.zero_()
                with self._nvtx_range("ppo_refresh_permutations"):
                    self._refresh_epoch_permutations()
                self._record_update_timing_start()
                with torch.cuda.nvtx.range("replay_update_graph"):
                    wp.capture_launch(self._update_graph, stream=self._update_wp_stream)
                    if self._detailed_nvtx:
                        wp.synchronize_stream(self._update_wp_stream)
                self._record_update_timing_end()
        if not self._detailed_nvtx:
            wp.synchronize_stream(self._update_wp_stream)
        metrics = self._read_update_summary()
        metrics["update_gpu_time_ns"] = float(self._consume_update_timing_ns())
        return metrics

    def _update_with_graph(self) -> dict[str, float]:
        if self._detailed_nvtx:
            if not self._has_detailed_update_graphs():
                return self._capture_update_graphs()
            return self._replay_update_graphs_detailed()
        if self._update_graph is None:
            return self._capture_update_graphs()
        return self._replay_update_graphs()

    def prewarm_update_graph(self) -> bool:
        with network_matmul_precision():
            return self._prewarm_update_graph_unscoped()

    def _prewarm_update_graph_unscoped(self) -> bool:
        if not self._use_update_graph:
            return False
        if self._detailed_nvtx:
            if self._has_detailed_update_graphs():
                return False
            self._capture_update_graph_detailed(launch_after_capture=False)
            return True
        if self._update_graph is not None:
            return False
        self._capture_update_graph(launch_after_capture=False)
        return True

    def _compile_update_wrappers_if_needed(self) -> None:
        if not self._compile_full_update or self._compiled_update_wrappers_ready:
            return
        self._obs_scale_update_into_fn = torch.compile(
            self._obs_scale_update_into, mode=self._compile_mode, dynamic=False
        )
        self._prepare_update_views_fn = torch.compile(
            self._prepare_update_views_compiled, mode=self._compile_mode, dynamic=False
        )
        self._run_update_minibatch_fn = torch.compile(
            self._run_update_minibatch, mode=self._compile_mode, dynamic=False
        )
        self._evaluate_old_policy_minibatch_fn = torch.compile(
            self._evaluate_old_policy_minibatch, mode=self._compile_mode, dynamic=False
        )
        self._compiled_update_wrappers_ready = True

    @contextmanager
    def _preserve_update_prewarm_sampling_state(
        self,
    ) -> Generator[None, None, None]:
        """Keep the capture prewarm from advancing sampling RNG or permutations.

        ``_refresh_epoch_permutations`` runs before the update warmup snapshots
        its own state, so its shuffle draw would otherwise survive the warmup
        restore. Preserve the surrounding state so a run that prewarms draws
        exactly the same permutations as one that does not.
        """
        shuffle_generator = getattr(self, "_shuffle_generator", None)
        shuffle_generator_state = (
            shuffle_generator.get_state() if shuffle_generator is not None else None
        )
        epoch_permutations = getattr(self, "_epoch_permutations", None)
        saved_epoch_permutations = (
            [permutation.detach().clone() for permutation in epoch_permutations]
            if epoch_permutations is not None
            else None
        )
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None
        )
        try:
            yield
        finally:
            if shuffle_generator_state is not None and shuffle_generator is not None:
                shuffle_generator.set_state(shuffle_generator_state)
            if saved_epoch_permutations is not None and epoch_permutations is not None:
                for permutation, saved_permutation in zip(
                    epoch_permutations,
                    saved_epoch_permutations,
                    strict=True,
                ):
                    permutation.copy_(saved_permutation)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, self.device)

    def _prewarm_compiled_update_capture_path(self) -> None:
        """Warm every compiled callable reachable from the captured update.

        The update wrappers compile because runtime.compile and the update CUDA
        graph are both enabled; that is also what makes the warmup mandatory.
        The update loop first reaches some compiled model callables (the no-grad
        + autocast specialization used by _recompute_old_policy) inside capture.
        Inductor would then compile and autotune there, and its benchmarking
        allocations invalidate the capturing stream. Warm the real update path
        first.
        """
        if not (self._compile_enabled and self._use_update_graph):
            return
        self._compile_update_wrappers_if_needed()
        if not self._compiled_model_functions_warmed:
            with self._nvtx_range("ppo_warmup_compiled_model_functions"):
                self._warmup_compiled_functions(
                    warmup_act_into=self._warmup_compiled_act_into
                )
            self._compiled_model_functions_warmed = True
        if self._compiled_update_capture_prewarmed:
            return
        with self._preserve_update_prewarm_sampling_state():
            try:
                with self._nvtx_range("ppo_prewarm_compiled_update_capture"):
                    self._refresh_epoch_permutations()
                    self._warmup_compiled_update_minibatch()
            except Exception as error:
                # Drain pending update-stream work before the sampling state is
                # restored by the surrounding context manager.
                self._restore_warmup_components(
                    [("pending prewarm work", self._synchronize_warmup_work)],
                    error=error,
                )
                raise
        self._compiled_update_capture_prewarmed = True

    def _update_eager(self) -> dict[str, float]:
        with self._nvtx_range("ppo_refresh_update_rewards"):
            self._refresh_update_rewards()
        with self._nvtx_range("ppo_refresh_permutations"):
            self._refresh_epoch_permutations()
        self._update_metrics.zero_()
        self._run_update_epoch_loops()
        return self._read_update_summary()

    def update(self) -> dict[str, float]:
        with network_matmul_precision():
            return self._update_unscoped()

    def _update_unscoped(self) -> dict[str, float]:
        self._require_training_runtime("update")
        if self.fail_on_nonfinite_update:
            self._raise_on_nonfinite_update_inputs()
        self._snapshot_analytic_kl_log_std()
        if self._use_update_graph:
            metrics = self._update_with_graph()
        else:
            metrics = self._update_eager()
        if self.fail_on_nonfinite_update:
            param_summary = self._nonfinite_parameter_summary()
            if param_summary:
                raise RuntimeError(
                    f"Non-finite PPO network parameters detected immediately after update: {param_summary}"
                )
            buffer_summary = self._nonfinite_module_buffer_summary()
            if buffer_summary:
                raise RuntimeError(
                    f"Non-finite PPO module buffers detected immediately after update: {buffer_summary}"
                )
        loss_scaler = getattr(self, "loss_scaler", None)
        if loss_scaler is not None:
            metrics = dict(metrics)
            metrics["loss_scale"] = float(loss_scaler.scale.item())
            metrics["loss_scaling_skipped_steps"] = float(
                loss_scaler.skipped_steps.item()
            )
        if self.reward_preprocessor is not None:
            metrics = dict(metrics)
            metrics["reward_scale"] = float(self.reward_preprocessor.reward_scale())
        if self.fail_on_nonfinite_update:
            self._raise_on_nonfinite_update_metrics(metrics)
        return metrics

    @torch.no_grad()
    def store_initial_state_samples(
        self,
        *,
        step: int,
        mask: torch.Tensor,
        samples: torch.Tensor,
        component_ids: torch.Tensor,
        old_log_prob: torch.Tensor,
        versions: torch.Tensor,
    ) -> None:
        learner = self.initial_state_learner
        if learner is None:
            raise RuntimeError("initial-state sample storage is not enabled")
        learner.store_start_samples(
            step=step,
            mask=mask,
            samples=samples,
            component_ids=component_ids,
            old_log_prob=old_log_prob,
            versions=versions,
        )

    def update_initial_state(self) -> dict[str, float]:
        with network_matmul_precision():
            learner = self.initial_state_learner
            return {} if learner is None else learner.update()

    @torch.no_grad()
    def clear_initial_state_rollout(self) -> None:
        learner = self.initial_state_learner
        if learner is not None:
            learner.clear_rollout()

    @torch.no_grad()
    def _raise_on_nonfinite_update_inputs(self) -> None:
        tensors: tuple[tuple[str, torch.Tensor | None], ...] = (
            ("obs", self.buffer.obs[:-1]),
            ("actions", self.buffer.actions),
            ("log_probs", self.buffer.log_probs),
            ("values", self.buffer.values),
            ("rewards", self.buffer.rewards),
            ("policy_aux", self.buffer.policy_aux),
        )
        bad: list[str] = []
        for name, value in tensors:
            if value is None:
                continue
            finite_mask = torch.isfinite(value)
            bad_count = int(torch.count_nonzero(~finite_mask).item())
            if bad_count:
                finite_abs_max = (
                    float(value[finite_mask].abs().max().item())
                    if bool(finite_mask.any().item())
                    else 1e309 - 1e309
                )
                detail = f"{name}={bad_count}/{value.numel()} finite_abs_max={finite_abs_max}"
                if name == "policy_aux":
                    bad_columns = torch.nonzero(
                        (~finite_mask).reshape(-1, value.shape[-1]).any(dim=0),
                        as_tuple=False,
                    ).flatten()
                    detail += f" bad_columns={bad_columns.detach().cpu().tolist()}"
                bad.append(detail)
        if bad:
            raise RuntimeError(
                "Non-finite PPO rollout inputs detected before update: "
                + ", ".join(bad)
            )

    def _raise_on_nonfinite_update_metrics(self, metrics: Mapping[str, float]) -> None:
        bad_metrics = _nonfinite_update_metric_items(metrics)
        if not bad_metrics:
            return
        metric_summary = ", ".join((f"{key}={value}" for key, value in bad_metrics))
        param_summary = self._nonfinite_parameter_summary()
        detail = (
            f"; nonfinite network parameters: {param_summary}" if param_summary else ""
        )
        raise RuntimeError(
            f"Non-finite PPO update metrics detected: {metric_summary}{detail}. The update has produced invalid training state; aborting before logging/checkpointing can hide the failure."
        )

    @torch.no_grad()
    def _nonfinite_parameter_summary(self, *, max_items: int = 8) -> str:
        items: list[str] = []
        total_bad = 0
        omitted = False
        for name, param in self.network.named_parameters():
            finite_mask = torch.isfinite(param)
            bad_count = int((~finite_mask).sum().item())
            if bad_count == 0:
                continue
            total_bad += bad_count
            if len(items) < max_items:
                items.append(f"{name}={bad_count}/{param.numel()}")
            else:
                omitted = True
        if not items:
            return ""
        suffix = " ..." if omitted else ""
        return ", ".join(items) + suffix

    @torch.no_grad()
    def _nonfinite_module_buffer_summary(self, *, max_items: int = 8) -> str:
        items: list[str] = []
        modules = (
            ("network", self.network),
            ("obs_scaler", self.obs_scaler),
            ("value_scaler", self.value_scaler),
        )
        for module_name, module in modules:
            for name, value in module.named_buffers():
                bad_count = int(torch.count_nonzero(~torch.isfinite(value)).item())
                if not bad_count:
                    continue
                detail = f"{module_name}.{name}={bad_count}/{value.numel()}"
                if module_name == "obs_scaler" and name == "running_variance":
                    stats_obs = self.buffer.obs[:-1, ..., : self.obs_dim].abs()
                    column_max = stats_obs.amax(dim=(0, 1))
                    top_values, top_indices = torch.topk(
                        column_max, k=min(12, column_max.numel())
                    )
                    detail += f" top_obs_columns={top_indices.detach().cpu().tolist()} top_obs_abs={top_values.detach().cpu().tolist()}"
                items.append(detail)
                if len(items) >= max_items:
                    return ", ".join(items)
        return ", ".join(items)

    def state_dict(self) -> dict[str, object]:
        if bool(getattr(self, "inference_only", False)):
            raise RuntimeError("PPO state_dict is unavailable in inference-only mode.")
        state = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_scaler": self.obs_scaler.state_dict(),
            "value_scaler": self.value_scaler.state_dict(),
        }
        if getattr(self, "loss_scaler", None) is not None:
            state["loss_scaler"] = self.loss_scaler.state_dict()
        if self.reward_preprocessor is not None:
            state["reward_preprocessor"] = self.reward_preprocessor.state_dict()
        initial_state_learner = getattr(self, "initial_state_learner", None)
        if initial_state_learner is not None:
            state["initial_state"] = initial_state_learner.full_state_dict()
        noise_seed = getattr(self, "_noise_seed", None)
        if noise_seed is not None:
            state["policy_noise_seed"] = noise_seed.detach().clone()
        if self._shuffle_generator is not None:
            state["shuffle_generator_state"] = self._shuffle_generator.get_state()
        state.update(PPOAgent._checkpoint_rng_state(self))
        return state

    def best_state_dict(self) -> dict[str, object]:
        state = {
            "network": self.network.state_dict(),
            "obs_scaler": self.obs_scaler.state_dict(),
            "value_scaler": self.value_scaler.state_dict(),
        }
        if self.reward_preprocessor is not None:
            state["reward_preprocessor"] = self.reward_preprocessor.live_state_dict()
        initial_state_learner = getattr(self, "initial_state_learner", None)
        if initial_state_learner is not None:
            state["initial_state"] = initial_state_learner.inference_state_dict()
        return state

    def _has_captured_update_graphs(self) -> bool:
        if getattr(self, "_update_graph", None) is not None:
            return True
        if getattr(self, "_detailed_update_prepare_graph", None) is not None:
            return True
        epoch_graphs = getattr(self, "_detailed_update_epoch_prepare_graphs", None)
        if epoch_graphs is not None and any(
            graph is not None for graph in epoch_graphs
        ):
            return True
        minibatch_graphs = getattr(self, "_detailed_update_minibatch_graphs", None)
        return bool(
            minibatch_graphs is not None
            and any(graph is not None for row in minibatch_graphs for graph in row)
        )

    def _checkpoint_rng_state(self) -> dict[str, torch.Tensor]:
        state = {"torch_rng_state": torch.get_rng_state()}
        device = torch.device(getattr(self, "device", "cpu"))
        if device.type == "cuda":
            state["torch_cuda_rng_state"] = torch.cuda.get_rng_state(device)
        return state

    def _restore_checkpoint_rng_state(self, state: Mapping[str, object]) -> None:
        cpu_rng = state.get("torch_rng_state")
        if cpu_rng is not None:
            torch.set_rng_state(cpu_rng.cpu())
        device = torch.device(getattr(self, "device", "cpu"))
        cuda_rng = state.get("torch_cuda_rng_state")
        if device.type == "cuda" and cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng.cpu(), device)

    def _checkpoint_transaction_state(self) -> dict[str, object]:
        noise_seed = getattr(self, "_noise_seed", None)
        return {
            **PPOAgent._checkpoint_rng_state(self),
            "loss_scaler": copy.deepcopy(self.loss_scaler.state_dict())
            if getattr(self, "loss_scaler", None) is not None
            else None,
            "network": copy.deepcopy(self.network.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict())
            if self.optimizer is not None
            else None,
            "obs_scaler": copy.deepcopy(self.obs_scaler.state_dict()),
            "value_scaler": copy.deepcopy(self.value_scaler.state_dict()),
            "reward_preprocessor": copy.deepcopy(self.reward_preprocessor.state_dict())
            if self.reward_preprocessor is not None
            else None,
            "initial_state": copy.deepcopy(initial_state_learner.full_state_dict())
            if (initial_state_learner := getattr(self, "initial_state_learner", None))
            is not None
            else None,
            "policy_noise_seed": noise_seed.detach().clone()
            if noise_seed is not None
            else None,
            "shuffle_generator_state": self._shuffle_generator.get_state().clone()
            if self._shuffle_generator is not None
            else None,
        }

    def _restore_checkpoint_transaction_state(
        self, state: Mapping[str, object]
    ) -> None:
        self.network.load_state_dict(state["network"])
        if getattr(self, "loss_scaler", None) is not None:
            self.loss_scaler.load_state_dict(state["loss_scaler"])
        self.obs_scaler.load_state_dict(state["obs_scaler"])
        self.value_scaler.load_state_dict(state["value_scaler"])
        reward_state = state["reward_preprocessor"]
        if reward_state is not None and self.reward_preprocessor is not None:
            self.reward_preprocessor.load_state_dict(reward_state)
        optimizer_state = state["optimizer"]
        if self.optimizer is not None and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        initial_state = state["initial_state"]
        initial_state_learner = getattr(self, "initial_state_learner", None)
        if initial_state is not None and initial_state_learner is not None:
            initial_state_learner.load_state_dict(initial_state, load_optimizer=True)
        noise_seed = getattr(self, "_noise_seed", None)
        saved_noise_seed = state["policy_noise_seed"]
        if noise_seed is not None and torch.is_tensor(saved_noise_seed):
            noise_seed.copy_(saved_noise_seed)
        saved_shuffle_state = state["shuffle_generator_state"]
        if saved_shuffle_state is not None and self._shuffle_generator is not None:
            self._shuffle_generator.set_state(saved_shuffle_state.cpu())
        if self.optimizer is not None:
            self._refresh_optimizer_lr_wrappers()
        PPOAgent._restore_checkpoint_rng_state(self, state)

    def _validate_full_resume_payload(
        self, payload: Mapping[str, object], *, path: Path
    ) -> None:
        """Reject policy-only payloads before mutating trainable agent state."""
        if bool(getattr(self, "inference_only", False)):
            return
        state_dict = getattr(self, "state_dict", None)
        current_full_payload = (
            state_dict() if callable(state_dict) else PPOAgent.state_dict(self)
        )
        required_keys = set(current_full_payload) - {
            "loss_scaler",
            "torch_rng_state",
            "torch_cuda_rng_state",
        }
        missing = sorted(key for key in required_keys if payload.get(key) is None)
        required_initial_state = current_full_payload.get("initial_state")
        checkpoint_initial_state = payload.get("initial_state")
        if isinstance(required_initial_state, Mapping):
            if not isinstance(checkpoint_initial_state, Mapping):
                if "initial_state" not in missing:
                    missing.append("initial_state")
            else:
                missing.extend(
                    PPOAgent._missing_required_mapping_paths(
                        required_initial_state,
                        checkpoint_initial_state,
                        prefix="initial_state",
                    )
                )
        missing.sort()
        if missing:
            raise ValueError(
                f"PPO training full resume requires a complete continuation payload; checkpoint {path} is missing required state: {missing}. Use trainer.initialization.type='policy_init' for policy-only initialization."
            )

    @staticmethod
    def _missing_required_mapping_paths(
        required: Mapping[object, object],
        checkpoint: Mapping[object, object],
        *,
        prefix: str,
    ) -> list[str]:
        missing: list[str] = []
        for key, required_value in required.items():
            path = f"{prefix}.{key}"
            checkpoint_value = checkpoint.get(key)
            if checkpoint_value is None:
                missing.append(path)
                continue
            if isinstance(required_value, Mapping):
                if not isinstance(checkpoint_value, Mapping):
                    missing.append(path)
                    continue
                missing.extend(
                    PPOAgent._missing_required_mapping_paths(
                        required_value, checkpoint_value, prefix=path
                    )
                )
        return missing

    def load(self, path: str | Path) -> CheckpointRestoreMetadata:
        path = Path(path)
        if path.is_dir():
            raise ValueError(f"Expected a checkpoint file, but got directory: {path}")
        if self._has_captured_update_graphs():
            raise RuntimeError(
                "PPO checkpoints must be loaded before update graph capture. Automatic mid-training graph invalidation is intentionally unsupported."
            )
        payload = torch.load(path, map_location=self.device)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"PPO checkpoint {path} must contain a mapping payload, got {type(payload).__name__}."
            )
        PPOAgent._validate_full_resume_payload(self, payload, path=path)
        transaction_state = self._checkpoint_transaction_state()
        optimizer_load_attempted = False
        optimizer_state = payload.get("optimizer")
        restore_metadata: CheckpointRestoreMetadata = {
            "optimizer_state_present": optimizer_state is not None,
            "optimizer_state_restored": False,
        }
        try:
            network_state = payload["network"]
            incompatible = self.network.load_state_dict(network_state, strict=False)
            allowed_missing = {
                "action_mean_scale",
                "action_std_scale",
                "action_log_std_scale",
            }
            allowed_unexpected = set()
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            disallowed_missing = missing - allowed_missing
            disallowed_unexpected = unexpected - allowed_unexpected
            if disallowed_unexpected or disallowed_missing:
                raise RuntimeError(
                    f"Checkpoint network state mismatch while loading {path}: missing={sorted(disallowed_missing)} unexpected={sorted(disallowed_unexpected)}"
                )
            if missing:
                _LOGGER.info(
                    "Checkpoint %s is missing optional derived buffers; keeping current defaults.",
                    path,
                )
            self.obs_scaler.load_state_dict(payload["obs_scaler"])
            self.value_scaler.load_state_dict(payload["value_scaler"])
            reward_preprocessor_state = payload.get("reward_preprocessor")
            if (
                reward_preprocessor_state is not None
                and self.reward_preprocessor is not None
            ):
                self.reward_preprocessor.load_state_dict(reward_preprocessor_state)
            initial_state_payload = payload.get("initial_state")
            initial_state_learner = getattr(self, "initial_state_learner", None)
            if initial_state_learner is not None:
                if initial_state_payload is None:
                    raise ValueError(
                        "checkpoint is missing the enabled initial-state learner"
                    )
                initial_state_learner.load_state_dict(
                    initial_state_payload, load_optimizer=not self.inference_only
                )
            elif initial_state_payload is not None:
                raise ValueError(
                    "checkpoint contains an initial-state learner but the environment has it disabled"
                )
            if optimizer_state is not None and self.optimizer is not None:
                optimizer_load_attempted = True
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except ValueError as exc:
                    raise ValueError(
                        f"PPO training full resume optimizer state is incompatible with the current agent for checkpoint {path}."
                    ) from exc
                else:
                    restore_metadata["optimizer_state_restored"] = True
            noise_seed = getattr(self, "_noise_seed", None)
            policy_noise_seed = payload.get("policy_noise_seed")
            if policy_noise_seed is not None and noise_seed is not None:
                noise_seed.copy_(policy_noise_seed)
            shuffle_generator_state = payload.get("shuffle_generator_state")
            if (
                shuffle_generator_state is not None
                and self._shuffle_generator is not None
            ):
                self._shuffle_generator.set_state(shuffle_generator_state.cpu())
            if getattr(self, "loss_scaler", None) is not None:
                scaler_state = payload.get("loss_scaler")
                if scaler_state is None:
                    self.loss_scaler.reset()
                else:
                    self.loss_scaler.load_state_dict(scaler_state)
                self.loss_scaler.prepare_optimizer(self.optimizer)
            if optimizer_load_attempted:
                if bool(getattr(self, "_use_update_graph", False)):
                    prepare_optimizer_state(self.optimizer, capturable=True)
                self._refresh_optimizer_lr_wrappers()
            self._compiled_update_capture_prewarmed = False
            if not bool(getattr(self, "inference_only", False)):
                PPOAgent._restore_checkpoint_rng_state(self, payload)
        except Exception:
            try:
                self._restore_checkpoint_transaction_state(transaction_state)
            except Exception as rollback_exc:
                raise RuntimeError(
                    f"Checkpoint load failed for {path}, and rollback also failed."
                ) from rollback_exc
            raise
        return restore_metadata

    def _synchronize_warmup_work(self, stream: torch.cuda.Stream | None = None) -> None:
        if stream is not None:
            stream.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def _restore_warmup_components(
        self,
        components: Sequence[tuple[str, Callable[[], Any]]],
        *,
        error: Exception | None = None,
    ) -> None:
        """Restore setup components, retaining the original error on cleanup failure."""
        for label, operation in components:
            if error is None:
                operation()
                continue
            try:
                operation()
            except Exception as cleanup_error:  # noqa: BLE001 - Restore all transaction components without replacing the original error.
                error.add_note(
                    f"Warmup rollback failed for {label}: {cleanup_error!r}. State restoration is best-effort."
                )


register_policy_mean_methods(
    PPOAgent,
    "ppo",
    ("policy_mean_into", "_scaled_eval_obs", "_obs_scale_eval_into", "_obs_scale_into"),
)
