"""Gaussian policy and value networks for PPO collection and optimization."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from instantmimic.utils.policy_mean_contract import register_policy_mean_methods

from .actor_critic_shared import (
    bounded_log_std,
    build_trunk,
    log_std_raw_from_bounded,
    physical_log_std,
    resolve_log_std_bound_mode,
    set_scale_buffer,
)
from .distributions import (
    LOG_2PI_HALF,
    diagonal_gaussian_entropy,
    gaussian_tanh_log_prob,
    tanh_log_abs_det_jacobian,
)

_LOGGER = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """Shared or separate Gaussian actor and scalar critic networks."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        layers: Sequence[int] = (512, 512, 256),
        activation: str = "elu",
        initial_log_std: float = 0.0,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        fixed_log_std: bool = False,
        fixed_sigma: bool = True,
        std_parameterization: str = "log",
        squash_actions: bool = False,
        clamp_mean: bool = False,
        clamp_mean_min: float = -1.0,
        clamp_mean_max: float = 1.0,
        log_std_bound_mode: str = "clamp",
        log_std_bound_loss_weight: float = 0.0,
        log_std_bound_loss_margin: float = 0.0,
        actor_layers: Sequence[int] | None = None,
        critic_layers: Sequence[int] | None = None,
        network_type: str = "mlp",
        actor_network_type: str | None = None,
        critic_network_type: str | None = None,
        use_layer_norm: bool = False,
        use_rms_norm: bool = False,
        residual_scale: float = 1.0,
        separate: bool = False,
        actor_obs_dim: int | None = None,
        critic_obs_dim: int | None = None,
        parameter_initialization: str = "default",
        policy_mlp_bf16: bool = False,
        activation_regularization_weight: float = 0.0,
        activation_regularization_target: float = 0.0,
    ) -> None:
        super().__init__()
        self.policy_mlp_bf16 = bool(policy_mlp_bf16)
        self.policy_mlp_dtype = torch.bfloat16 if policy_mlp_bf16 else None
        parameter_initialization_key = (
            str(parameter_initialization).strip().lower().replace("-", "_")
        )
        if parameter_initialization_key not in {"default", "rsl_rl"}:
            raise ValueError(
                "parameter_initialization must be 'default' or 'rsl_rl', got "
                f"{parameter_initialization!r}"
            )
        initial_cpu_rng_state = (
            torch.get_rng_state() if parameter_initialization_key == "rsl_rl" else None
        )
        if not bool(fixed_sigma) and bool(fixed_log_std):
            raise ValueError(
                "fixed log standard deviation requires "
                "models.actor.distribution.std=parameter."
            )
        self.std_parameterization = str(std_parameterization).strip().lower()
        if self.std_parameterization not in {"log", "scalar"}:
            raise ValueError(
                "std_parameterization must be 'log' or 'scalar', got "
                f"{std_parameterization!r}"
            )
        if self.std_parameterization == "scalar" and not bool(fixed_sigma):
            raise ValueError(
                "scalar std parameterization requires state-independent sigma"
            )
        if self.std_parameterization == "scalar" and log_std_bound_mode != "clamp":
            raise ValueError("scalar std parameterization requires clamp bounds")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        # Squared penalty on the policy mean, added to the PPO loss through the
        # auxiliary-loss evaluation path; used by activation-output policies.
        self.activation_regularization_weight = float(activation_regularization_weight)
        self.activation_regularization_target = float(activation_regularization_target)
        if (
            not math.isfinite(self.activation_regularization_weight)
            or self.activation_regularization_weight < 0.0
            or not math.isfinite(self.activation_regularization_target)
        ):
            raise ValueError(
                "activation regularization weight must be finite and non-negative "
                "and its target finite"
            )
        self.squash_actions = bool(squash_actions)
        self.clamp_mean = bool(clamp_mean)
        self.clamp_mean_min = float(clamp_mean_min)
        self.clamp_mean_max = float(clamp_mean_max)
        if self.clamp_mean:
            _LOGGER.info(
                "ActorCritic clamp_mean=True range=[%.2f, %.2f]",
                self.clamp_mean_min,
                self.clamp_mean_max,
            )
        self._squash_epsilon = 1.0e-6
        self.obs_main_dim = self.obs_dim
        if self.obs_dim <= 0 or self.act_dim <= 0:
            raise ValueError("Observation and action dimensions must be positive")
        if (actor_obs_dim is None) != (critic_obs_dim is None):
            raise ValueError(
                "actor_obs_dim and critic_obs_dim must be provided together"
            )
        self._asymmetric_observations = actor_obs_dim is not None
        if self._asymmetric_observations:
            self.actor_obs_dim = int(actor_obs_dim)
            self.critic_obs_dim = int(critic_obs_dim)
            if self.actor_obs_dim <= 0 or self.critic_obs_dim <= 0:
                raise ValueError(
                    "actor_obs_dim and critic_obs_dim must both be positive"
                )
            if self.actor_obs_dim + self.critic_obs_dim != self.obs_main_dim:
                raise ValueError(
                    "actor_obs_dim + critic_obs_dim must equal obs_dim; got "
                    f"{self.actor_obs_dim} + {self.critic_obs_dim} != "
                    f"{self.obs_main_dim}"
                )
            if not separate:
                raise ValueError(
                    "Asymmetric actor/critic observations require separate=True"
                )
        else:
            self.actor_obs_dim = self.obs_main_dim
            self.critic_obs_dim = self.obs_main_dim

        self._separate_actor_critic = bool(separate)
        resolved_network_type = str(network_type).strip().lower()
        resolved_actor_network_type = (
            str(
                actor_network_type
                if actor_network_type is not None
                else resolved_network_type
            )
            .strip()
            .lower()
        )
        resolved_critic_network_type = (
            str(
                critic_network_type
                if critic_network_type is not None
                else resolved_network_type
            )
            .strip()
            .lower()
        )
        if parameter_initialization_key == "rsl_rl":
            if not separate:
                raise ValueError(
                    "parameter_initialization='rsl_rl' requires separate=True"
                )
            if (
                resolved_actor_network_type != "mlp"
                or resolved_critic_network_type != "mlp"
                or use_layer_norm
                or use_rms_norm
                or not fixed_sigma
            ):
                raise ValueError(
                    "parameter_initialization='rsl_rl' requires separate plain "
                    "MLP actor/critic networks, fixed sigma, and no "
                    "layer/RMS normalization"
                )
        resolved_actor_layers = (
            list(actor_layers) if actor_layers is not None else list(layers)
        )
        resolved_critic_layers = (
            list(critic_layers) if critic_layers is not None else list(layers)
        )
        if self._separate_actor_critic:
            self.trunk, critic_out_dim = build_trunk(
                self.critic_obs_dim,
                resolved_critic_layers,
                activation,
                network_type=resolved_critic_network_type,
                use_layer_norm=use_layer_norm,
                use_rms_norm=use_rms_norm,
                residual_scale=residual_scale,
            )
            self.actor_trunk, actor_out_dim = build_trunk(
                self.actor_obs_dim,
                resolved_actor_layers,
                activation,
                network_type=resolved_actor_network_type,
                use_layer_norm=use_layer_norm,
                use_rms_norm=use_rms_norm,
                residual_scale=residual_scale,
            )
        else:
            self.trunk, critic_out_dim = build_trunk(
                self.obs_main_dim,
                layers,
                activation,
                network_type=resolved_network_type,
                use_layer_norm=use_layer_norm,
                use_rms_norm=use_rms_norm,
                residual_scale=residual_scale,
            )
            self.actor_trunk = self.trunk
            actor_out_dim = critic_out_dim
        self.actor_feature_dim = int(actor_out_dim)
        self.critic_feature_dim = int(critic_out_dim)
        self.value_head = nn.Linear(critic_out_dim, 1)

        self.fixed_sigma = bool(fixed_sigma)
        self._uses_state_dependent_sigma = not self.fixed_sigma
        self.log_std_head: nn.Linear | None = None
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self._log_std_bound_mode = resolve_log_std_bound_mode(log_std_bound_mode)
        self.log_std_bound_loss_weight = float(log_std_bound_loss_weight)
        self.log_std_bound_loss_margin = float(log_std_bound_loss_margin)
        if (
            not math.isfinite(self.log_std_bound_loss_weight)
            or self.log_std_bound_loss_weight < 0.0
        ):
            raise ValueError(
                "log_std_bound_loss_weight must be a finite non-negative value"
            )
        if (
            not math.isfinite(self.log_std_bound_loss_margin)
            or self.log_std_bound_loss_margin < 0.0
        ):
            raise ValueError(
                "log_std_bound_loss_margin must be a finite non-negative value"
            )
        if self.log_std_bound_loss_weight > 0.0:
            if self.fixed_sigma:
                raise ValueError("log_std bound loss requires state-dependent sigma")
            if self._log_std_bound_mode != "clamp":
                raise ValueError("log_std bound loss requires clamp bounds")
            if 2.0 * self.log_std_bound_loss_margin >= (
                self.max_log_std - self.min_log_std
            ):
                raise ValueError(
                    "log_std_bound_loss_margin must leave a non-empty bound interval"
                )
        raw_initial_log_std = log_std_raw_from_bounded(
            float(initial_log_std),
            min_log_std=self.min_log_std,
            max_log_std=self.max_log_std,
            bound_mode=self._log_std_bound_mode,
        )

        self.policy_head = nn.Linear(actor_out_dim, self.act_dim)

        self.register_buffer(
            "action_mean_scale", torch.ones(self.act_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "action_std_scale", torch.ones(self.act_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "action_log_std_scale", torch.zeros(self.act_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "_zero_scalar", torch.zeros((), dtype=torch.float32), persistent=False
        )

        initial_std_parameter = (
            math.exp(float(initial_log_std))
            if self.std_parameterization == "scalar"
            else raw_initial_log_std
        )
        self.log_std = nn.Parameter(
            torch.full((self.act_dim,), initial_std_parameter),
            requires_grad=bool(self.fixed_sigma and not bool(fixed_log_std)),
        )
        if self._uses_state_dependent_sigma:
            self.log_std_head = nn.Linear(actor_out_dim, self.act_dim)
            nn.init.zeros_(self.log_std_head.weight)
            nn.init.constant_(self.log_std_head.bias, raw_initial_log_std)
        self._use_standard_policy_fast_path = bool(
            self.policy_head is not None
            and not self._asymmetric_observations
            and not self.squash_actions
            and not self.clamp_mean
            and self.fixed_sigma
            and not self._separate_actor_critic
        )
        self._use_standard_policy_evaluate_fast_path = (
            self._use_standard_policy_fast_path
        )
        if initial_cpu_rng_state is not None:
            if any(parameter.device.type != "cpu" for parameter in self.parameters()):
                raise RuntimeError(
                    "parameter_initialization='rsl_rl' requires constructing the "
                    "network on CPU before moving it to the training device"
                )
            torch.set_rng_state(initial_cpu_rng_state)
            for module in self.actor_trunk.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            assert self.policy_head is not None
            self.policy_head.reset_parameters()
            for module in self.trunk.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            self.value_head.reset_parameters()

    def set_policy_precision(self, precision: str) -> None:
        """Configure one MLP dtype for collection and gradient-enabled evaluation."""
        self.policy_mlp_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[precision]
        self.policy_mlp_bf16 = self.policy_mlp_dtype == torch.bfloat16

    def _policy_mlp_dtype_in_effect(self, device_type: str) -> torch.dtype | None:
        """Return the dtype the policy MLP must enforce, or None for plain FP32.

        Explicit ``float32`` only needs its own context while an ambient autocast
        is active; otherwise it takes the plain FP32 path with fused output heads.
        """
        dtype = self.policy_mlp_dtype
        if dtype == torch.float32 and not torch.is_autocast_enabled(device_type):
            return None
        return dtype

    def _mlp_forward(self, module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Keep FP32 master weights and FP32 policy/physics boundaries.

        Use the same precision during collection and evaluation. A short-lived
        autocast context avoids retaining stale weight casts across optimizer
        steps and avoids no-grad casts leaking into gradient-enabled updates.
        """
        if self._policy_mlp_dtype_in_effect(inputs.device.type) is None:
            return module(inputs)
        with torch.autocast(
            device_type=inputs.device.type,
            dtype=self.policy_mlp_dtype,
            enabled=self.policy_mlp_dtype != torch.float32,
            cache_enabled=False,
        ):
            outputs = module(inputs)
        return outputs.float()

    def _actor_features(self, main_obs: torch.Tensor) -> torch.Tensor:
        if self._asymmetric_observations:
            main_obs = main_obs[..., : self.actor_obs_dim]
        return self._mlp_forward(self.actor_trunk, main_obs)

    def _critic_features(self, main_obs: torch.Tensor) -> torch.Tensor:
        if self._asymmetric_observations:
            main_obs = main_obs[
                ..., self.actor_obs_dim : self.actor_obs_dim + self.critic_obs_dim
            ]
        return self._mlp_forward(self.trunk, main_obs)

    def _actor_critic_features(
        self, main_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_features = self._actor_features(main_obs)
        if not self._separate_actor_critic:
            return actor_features, actor_features
        return actor_features, self._critic_features(main_obs)

    def policy_std_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters controlling the policy's learned standard deviation."""
        parameters = [self.log_std]
        if self.log_std_head is not None:
            parameters.extend(self.log_std_head.parameters())
        return tuple(parameters)

    def separate_ppo_parameter_groups(
        self,
    ) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
        """Return the disjoint actor and critic groups used by RSL-RL PPO."""
        if not self._separate_actor_critic:
            raise RuntimeError("Separate PPO gradient clipping requires separate=True")
        actor_prefixes = (
            "actor_trunk.",
            "policy_head.",
        )
        std_parameters = set(self.policy_std_parameters())
        critic_prefixes = ("trunk.", "value_head.")
        actor_parameters: list[nn.Parameter] = []
        critic_parameters: list[nn.Parameter] = []
        unassigned: list[str] = []
        for name, parameter in self.named_parameters():
            if parameter in std_parameters or name.startswith(actor_prefixes):
                actor_parameters.append(parameter)
            elif name.startswith(critic_prefixes):
                critic_parameters.append(parameter)
            else:
                unassigned.append(name)
        if unassigned:
            raise RuntimeError(
                "Separate PPO parameter grouping has unassigned parameters: "
                + ", ".join(unassigned)
            )
        return tuple(actor_parameters), tuple(critic_parameters)

    def initialize_actor_output(self, standard_deviation: float) -> None:
        """Initialize the current policy output head around zero."""
        standard_deviation = float(standard_deviation)
        if not math.isfinite(standard_deviation) or standard_deviation < 0.0:
            raise ValueError(
                "actor output initialization standard deviation must be finite "
                "and nonnegative"
            )
        head = self.policy_head
        if head is None:
            raise RuntimeError("Actor policy output head is not initialized")
        if standard_deviation == 0.0:
            nn.init.zeros_(head.weight)
        else:
            nn.init.normal_(head.weight, mean=0.0, std=standard_deviation)
        nn.init.zeros_(head.bias)

    def _policy_log_std(
        self,
        mean: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._uses_state_dependent_sigma:
            if features is None:
                raise RuntimeError(
                    "State-dependent sigma requires trunk features to compute log_std."
                )
            return self._state_dependent_policy_log_std(
                features,
                out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
            )
        if out_raw_log_std_bound_stats is not None:
            out_raw_log_std_bound_stats.zero_()
        return self._physical_log_std().expand_as(mean)

    def _std_stats_from_log_std(
        self, log_std: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std = log_std.exp()
        return self._std_stats_from_std(std)

    def _std_stats_from_std(
        self, std: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        std = std.detach().float()
        return std.mean(), std.amin(), std.amax()

    def _policy_std_stats_for_current_parameters(
        self, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._uses_state_dependent_sigma:
            std_mean = self.policy_std_mean().to(
                device=reference.device,
                dtype=torch.float32,
            )
            return std_mean, std_mean, std_mean
        log_std = self._physical_log_std().to(device=reference.device)
        return self._std_stats_from_log_std(log_std)

    def _policy_mean_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self._mlp_forward(self.policy_head, features) * self.action_mean_scale

    def _standard_policy_mean_from_features(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        assert self.policy_head is not None
        return self._mlp_forward(self.policy_head, features) * self.action_mean_scale

    def _standard_policy_mean_into_from_features(
        self,
        features: torch.Tensor,
        out_actions: torch.Tensor,
    ) -> None:
        assert self.policy_head is not None
        if self._policy_mlp_dtype_in_effect(features.device.type) is not None:
            out_actions.copy_(self._standard_policy_mean_from_features(features))
            return
        torch.addmm(
            self.policy_head.bias,
            features,
            self.policy_head.weight.t(),
            out=out_actions,
        )
        out_actions.mul_(self.action_mean_scale)

    def _physical_log_std(self) -> torch.Tensor:
        scale = self.action_log_std_scale
        if self.std_parameterization == "scalar":
            return (
                torch.log(
                    self.log_std.clamp(
                        min=math.exp(self.min_log_std),
                        max=math.exp(self.max_log_std),
                    )
                )
                + scale
            )
        return physical_log_std(
            self.log_std,
            min_log_std=self.min_log_std,
            max_log_std=self.max_log_std,
            log_std_scale=scale,
            bound_mode=self._log_std_bound_mode,
        )

    def _bound_log_std(self, raw_log_std: torch.Tensor) -> torch.Tensor:
        return bounded_log_std(
            raw_log_std,
            min_log_std=self.min_log_std,
            max_log_std=self.max_log_std,
            bound_mode=getattr(self, "_log_std_bound_mode", "clamp"),
        )

    def _state_dependent_policy_log_std(
        self,
        features: torch.Tensor,
        *,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.log_std_head is None:
            raise RuntimeError(
                "State-dependent sigma requested, but log_std_head is unavailable."
            )
        raw_log_std = self._mlp_forward(self.log_std_head, features)
        if out_raw_log_std_bound_stats is not None:
            if out_raw_log_std_bound_stats.numel() not in {4, 5}:
                raise ValueError(
                    "out_raw_log_std_bound_stats must contain four or five scalars."
                )
            out_raw_log_std_bound_stats.zero_()
            if self._log_std_bound_mode == "clamp":
                raw_stats = raw_log_std.detach().float()
                lower_overshoot = (self.min_log_std - raw_stats).clamp_min(0.0)
                upper_overshoot = (raw_stats - self.max_log_std).clamp_min(0.0)
                overshoot = lower_overshoot + upper_overshoot
                out_raw_log_std_bound_stats[0].copy_(
                    (raw_stats < self.min_log_std).float().mean()
                )
                out_raw_log_std_bound_stats[1].copy_(
                    (raw_stats > self.max_log_std).float().mean()
                )
                out_raw_log_std_bound_stats[2].copy_(overshoot.square().mean())
                out_raw_log_std_bound_stats[3].copy_(overshoot.amax())
            if out_raw_log_std_bound_stats.numel() == 5:
                if self.log_std_bound_loss_weight > 0.0:
                    head_only_raw_log_std = self._mlp_forward(
                        self.log_std_head, features.detach()
                    )
                    lower_limit = self.min_log_std + self.log_std_bound_loss_margin
                    upper_limit = self.max_log_std - self.log_std_bound_loss_margin
                    lower_violation = (lower_limit - head_only_raw_log_std).clamp_min(
                        0.0
                    )
                    upper_violation = (head_only_raw_log_std - upper_limit).clamp_min(
                        0.0
                    )
                    bound_loss = (
                        self.log_std_bound_loss_weight
                        * (lower_violation.square() + upper_violation.square()).mean()
                    )
                else:
                    bound_loss = self._graph_connected_zero_scalar(raw_log_std)
                out_raw_log_std_bound_stats[4].copy_(bound_loss)
        log_std = self._bound_log_std(raw_log_std)
        scale = self.action_log_std_scale.to(device=log_std.device, dtype=log_std.dtype)
        return log_std + scale.view(1, -1)

    def zero_scalar(self) -> torch.Tensor:
        return self._zero_scalar

    def _graph_connected_zero_scalar(
        self, source: torch.Tensor | None = None
    ) -> torch.Tensor:
        if source is None:
            return self._zero_scalar
        source_view = source.float().reshape(-1)
        if source_view.numel() == 0:
            return self._zero_scalar
        return source_view[:1].sum() * 0.0

    def set_action_mean_scale(self, scale: torch.Tensor) -> None:
        self.action_mean_scale.copy_(
            set_scale_buffer(
                self.action_mean_scale,
                scale,
                label="action mean scale",
                strictly_positive=False,
            )
        )

    def set_action_std_scale(self, scale: torch.Tensor) -> None:
        scale_f = set_scale_buffer(
            self.action_std_scale,
            scale,
            label="action std scale",
            strictly_positive=True,
        )
        self.action_std_scale.copy_(scale_f)
        self.action_log_std_scale.copy_(torch.log(scale_f))

    def policy_mean(self, obs: torch.Tensor) -> torch.Tensor:
        out_actions = torch.empty(
            obs.shape[:-1] + (self.act_dim,),
            device=obs.device,
            dtype=torch.float32,
        )
        self.policy_mean_into(obs, out_actions)
        return out_actions

    def policy_log_std_vector(self) -> torch.Tensor:
        if self._uses_state_dependent_sigma:
            raise RuntimeError(
                "State-dependent sigma does not have a rollout-invariant log_std vector."
            )
        return self._physical_log_std()

    def policy_distribution_params(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._actor_features(obs)
        mean = self._policy_mean_from_features(features)
        if self.clamp_mean:
            mean = mean.clamp(self.clamp_mean_min, self.clamp_mean_max)
        return mean, self._policy_log_std(mean, features=features)

    def policy_mean_for_loss(self, obs: torch.Tensor) -> torch.Tensor:
        if self._use_standard_policy_fast_path:
            mean = self._standard_policy_mean_from_features(
                self._mlp_forward(self.trunk, obs)
            )
        else:
            features = self._actor_features(obs)
            mean = self._policy_mean_from_features(features)
        if self.squash_actions:
            return mean.tanh()
        if self.clamp_mean:
            return mean.clamp(self.clamp_mean_min, self.clamp_mean_max)
        return mean

    def policy_mean_into(self, obs: torch.Tensor, out_actions: torch.Tensor) -> None:
        if self._use_standard_policy_fast_path:
            self._standard_policy_mean_into_from_features(
                self._mlp_forward(self.trunk, obs), out_actions
            )
            return
        features = self._actor_features(obs)
        out_actions[:] = self._policy_mean_from_features(features)
        if self.squash_actions:
            out_actions.tanh_()
        elif self.clamp_mean:
            out_actions.clamp_(self.clamp_mean_min, self.clamp_mean_max)

    @torch.no_grad()
    def policy_execution_mean_into(
        self, obs: torch.Tensor, out_actions: torch.Tensor
    ) -> None:
        self.policy_mean_into(obs, out_actions)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out_actions = torch.empty(
            obs.shape[:-1] + (self.act_dim,),
            device=obs.device,
            dtype=torch.float32,
        )
        out_log_probs = torch.empty(
            obs.shape[:-1] + (1,),
            device=obs.device,
            dtype=torch.float32,
        )
        out_values = torch.empty_like(out_log_probs)
        self.act_into(obs, out_actions, out_log_probs, out_values)
        return out_actions, out_log_probs, out_values

    def _evaluate_policy_terms(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_features, critic_features = self._actor_critic_features(obs)
        mean = self._policy_mean_from_features(actor_features)
        if self.clamp_mean:
            mean = mean.clamp(self.clamp_mean_min, self.clamp_mean_max)
        log_std = self._policy_log_std(mean, features=actor_features)
        log_prob = gaussian_tanh_log_prob(
            actions,
            mean,
            log_std,
            log_std.exp(),
            squash=self.squash_actions,
            epsilon=self._squash_epsilon,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )
        entropy = diagonal_gaussian_entropy(log_std)
        values = self._mlp_forward(self.value_head, critic_features)
        return log_prob, values, entropy

    def _evaluate_policy_terms_with_std_stats(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return self._evaluate_policy_terms_with_std_stats_and_mean(
            obs,
            actions,
            pre_squash_actions=pre_squash_actions,
            out_policy_mean=out_policy_mean,
            out_policy_log_std=out_policy_log_std,
            out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
        )[:6]

    def _evaluate_policy_terms_with_std_stats_and_mean(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        actor_features, critic_features = self._actor_critic_features(obs)
        mean = self._policy_mean_from_features(actor_features)
        raw_mean = mean
        if self.clamp_mean:
            mean = mean.clamp(self.clamp_mean_min, self.clamp_mean_max)
        log_std = self._policy_log_std(
            mean,
            features=actor_features,
            out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
        )
        if out_policy_mean is not None:
            if out_policy_log_std is None:
                raise ValueError(
                    "Policy mean and log-std outputs must be supplied together"
                )
            out_policy_mean.copy_(mean)
            out_policy_log_std.copy_(log_std)
        std = log_std.exp()
        std_mean, std_min, std_max = self._std_stats_from_std(std)
        log_prob = gaussian_tanh_log_prob(
            actions,
            mean,
            log_std,
            std,
            squash=self.squash_actions,
            epsilon=self._squash_epsilon,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )
        entropy = diagonal_gaussian_entropy(log_std)
        values = self._mlp_forward(self.value_head, critic_features)
        return log_prob, values, entropy, std_mean, std_min, std_max, raw_mean

    def _activation_regularization_loss(self, mean: torch.Tensor) -> torch.Tensor:
        return (
            self.activation_regularization_weight
            * (mean.float() - self.activation_regularization_target).square().mean()
        )

    def evaluate_with_aux_losses_and_std_stats(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """PPO evaluation terms plus the policy-mean regularization loss."""
        if policy_aux is not None:
            raise ValueError("The ordinary policy does not consume auxiliary inputs")
        log_prob, values, entropy, std_mean, std_min, std_max, mean = (
            self._evaluate_policy_terms_with_std_stats_and_mean(
                obs,
                actions,
                pre_squash_actions=pre_squash_actions,
                out_policy_mean=out_policy_mean,
                out_policy_log_std=out_policy_log_std,
                out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
            )
        )
        aux_loss = self._activation_regularization_loss(mean)
        return log_prob, values, entropy, aux_loss, std_mean, std_min, std_max

    def evaluate_with_aux_losses(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        terms = self.evaluate_with_aux_losses_and_std_stats(
            obs,
            actions,
            pre_squash_actions=pre_squash_actions,
            policy_aux=policy_aux,
        )
        return terms[:4]

    def evaluate_with_aux_and_std_stats_into(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        out_log_probs: torch.Tensor,
        out_values: torch.Tensor,
        out_entropy: torch.Tensor,
        out_aux_loss: torch.Tensor,
        out_std_mean: torch.Tensor,
        out_std_min: torch.Tensor,
        out_std_max: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
    ) -> None:
        log_prob, values, entropy, aux_loss, std_mean, std_min, std_max = (
            self.evaluate_with_aux_losses_and_std_stats(
                obs,
                actions,
                pre_squash_actions=pre_squash_actions,
                policy_aux=policy_aux,
                out_policy_mean=out_policy_mean,
                out_policy_log_std=out_policy_log_std,
                out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
            )
        )
        out_log_probs.copy_(log_prob)
        out_values.copy_(values)
        out_entropy.copy_(entropy)
        out_aux_loss.copy_(aux_loss)
        out_std_mean.copy_(std_mean)
        out_std_min.copy_(std_min)
        out_std_max.copy_(std_max)

    def evaluate_with_aux_into(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        out_log_probs: torch.Tensor,
        out_values: torch.Tensor,
        out_entropy: torch.Tensor,
        out_aux_loss: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
    ) -> None:
        log_prob, values, entropy, aux_loss = self.evaluate_with_aux_losses(
            obs,
            actions,
            pre_squash_actions=pre_squash_actions,
            policy_aux=policy_aux,
        )
        out_log_probs.copy_(log_prob)
        out_values.copy_(values)
        out_entropy.copy_(entropy)
        out_aux_loss.copy_(aux_loss)

    def evaluate_into(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        out_log_probs: torch.Tensor,
        out_values: torch.Tensor,
        out_entropy: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
    ) -> None:
        if policy_aux is not None:
            raise ValueError("The ordinary policy does not consume auxiliary inputs")
        log_prob, values, entropy = self._evaluate_policy_terms(
            obs,
            actions,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )
        out_log_probs.copy_(log_prob)
        out_values.copy_(values)
        out_entropy.copy_(entropy)

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if policy_aux is not None:
            raise ValueError("The ordinary policy does not consume auxiliary inputs")
        return self._evaluate_policy_terms(
            obs,
            actions,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )

    def evaluate_with_std_stats_into(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        out_log_probs: torch.Tensor,
        out_values: torch.Tensor,
        out_entropy: torch.Tensor,
        out_std_mean: torch.Tensor,
        out_std_min: torch.Tensor,
        out_std_max: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
    ) -> None:
        if policy_aux is not None:
            raise ValueError("The ordinary policy does not consume auxiliary inputs")
        log_prob, values, entropy, std_mean, std_min, std_max = (
            self._evaluate_policy_terms_with_std_stats(
                obs,
                actions,
                out_policy_mean=out_policy_mean,
                out_policy_log_std=out_policy_log_std,
                out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
                **(
                    {"pre_squash_actions": pre_squash_actions}
                    if pre_squash_actions is not None
                    else {}
                ),
            )
        )
        out_log_probs.copy_(log_prob)
        out_values.copy_(values)
        out_entropy.copy_(entropy)
        out_std_mean.copy_(std_mean)
        out_std_min.copy_(std_min)
        out_std_max.copy_(std_max)

    def evaluate_with_std_stats(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if policy_aux is not None:
            raise ValueError("The ordinary policy does not consume auxiliary inputs")
        return self._evaluate_policy_terms_with_std_stats(
            obs,
            actions,
            out_policy_mean=out_policy_mean,
            out_policy_log_std=out_policy_log_std,
            out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )

    def _act_into_standard_fast(
        self,
        obs: torch.Tensor,
        out_actions: torch.Tensor,
        out_log_probs: torch.Tensor,
        out_values: torch.Tensor,
        eps: torch.Tensor | None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
    ) -> None:
        if self._uses_state_dependent_sigma:
            raise RuntimeError(
                "_act_into_standard_fast is unavailable when fixed_sigma=false."
            )
        features = self._mlp_forward(self.trunk, obs)
        self._standard_policy_mean_into_from_features(features, out_actions)
        if out_policy_mean is not None:
            out_policy_mean.copy_(out_actions)
        log_std = self._physical_log_std()
        if out_policy_log_std is not None:
            out_policy_log_std.copy_(log_std.expand_as(out_actions))
        std = log_std.exp().expand_as(out_actions)
        if eps is None:
            eps = torch.randn_like(out_actions)
        out_actions.addcmul_(std, eps)
        out_log_probs.copy_(
            torch.sum(
                -0.5 * eps.square() - log_std - LOG_2PI_HALF,
                dim=-1,
                keepdim=True,
            )
        )
        if self._policy_mlp_dtype_in_effect(features.device.type) is not None:
            out_values.copy_(self._mlp_forward(self.value_head, features))
            return
        torch.addmm(
            self.value_head.bias,
            features,
            self.value_head.weight.t(),
            out=out_values,
        )

    def _sample_policy_into_outputs(
        self,
        *,
        features: torch.Tensor,
        value_features: torch.Tensor,
        mean: torch.Tensor,
        out_actions: torch.Tensor,
        out_log_probs: torch.Tensor,
        out_values: torch.Tensor,
        eps: torch.Tensor | None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_pre_squash_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.clamp_mean:
            mean = mean.clamp(self.clamp_mean_min, self.clamp_mean_max)
        out_actions.copy_(mean)
        if out_policy_mean is not None:
            out_policy_mean.copy_(mean)
        log_std = self._policy_log_std(out_actions, features=features)
        if out_policy_log_std is not None:
            out_policy_log_std.copy_(log_std)
        std = log_std.exp()
        if eps is None:
            eps = torch.randn_like(out_actions)
        out_actions.addcmul_(std, eps)
        log_det_jacobian = None
        if out_pre_squash_actions is not None:
            out_pre_squash_actions.copy_(out_actions)
        if self.squash_actions:
            log_det_jacobian = tanh_log_abs_det_jacobian(out_actions)
            eps = (out_actions - mean) / std
            out_actions.tanh_()
        out_log_probs.copy_(
            torch.sum(
                -0.5 * eps.square() - log_std - LOG_2PI_HALF,
                dim=-1,
                keepdim=True,
            )
        )
        if log_det_jacobian is not None:
            out_log_probs.sub_(torch.sum(log_det_jacobian, dim=-1, keepdim=True))
        out_values.copy_(self._mlp_forward(self.value_head, value_features))
        return mean

    def act_into(
        self,
        obs: torch.Tensor,
        out_actions: torch.Tensor,
        out_log_probs: torch.Tensor,
        out_values: torch.Tensor,
        eps: torch.Tensor | None = None,
        *,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_pre_squash_actions: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
    ) -> None:
        if policy_aux is not None:
            raise ValueError("The ordinary policy does not consume auxiliary inputs")
        if self._use_standard_policy_fast_path:
            self._act_into_standard_fast(
                obs,
                out_actions,
                out_log_probs,
                out_values,
                eps,
                out_policy_mean,
                out_policy_log_std,
            )
            if out_pre_squash_actions is not None:
                out_pre_squash_actions.copy_(out_actions)
            return
        features, value_features = self._actor_critic_features(obs)
        mean = self._policy_mean_from_features(features)
        self._sample_policy_into_outputs(
            features=features,
            value_features=value_features,
            mean=mean,
            out_actions=out_actions,
            out_log_probs=out_log_probs,
            out_values=out_values,
            eps=eps,
            out_policy_mean=out_policy_mean,
            out_policy_log_std=out_policy_log_std,
            **(
                {"out_pre_squash_actions": out_pre_squash_actions}
                if out_pre_squash_actions is not None
                else {}
            ),
        )

    def policy_std_mean(self) -> torch.Tensor:
        if self._uses_state_dependent_sigma:
            assert self.log_std_head is not None
            bias_log_std = self._bound_log_std(self.log_std_head.bias)
            scale = self.action_log_std_scale.to(
                device=bias_log_std.device, dtype=bias_log_std.dtype
            )
            return (bias_log_std + scale).exp().mean()
        return self._physical_log_std().exp().mean()


register_policy_mean_methods(
    ActorCritic,
    "actor_critic",
    (
        "policy_mean_into",
        "_actor_features",
        "_policy_mean_from_features",
        "_standard_policy_mean_into_from_features",
    ),
)
