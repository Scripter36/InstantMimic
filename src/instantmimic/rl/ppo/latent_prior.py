"""VAE policy, PPO regularization and frozen decoder runtime.

The posterior uses privileged imitation observations. The prior and decoder
use only the deployable partition, allowing the decoder to be reused by a
command policy. A rollout stores posterior noise so PPO evaluates the same
conditional policy during every minibatch update.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from instantmimic.config.artifacts import load_artifact_semantic_config
from instantmimic.config.schema import SemanticConfig
from instantmimic.envs.config_access import get_value as _cfg_get
from instantmimic.rl.build_specs import PPOAgentBuildSpec, lower_agent_build_spec
from instantmimic.rl.models.actor_critic import ActorCritic
from instantmimic.rl.models.actor_critic_shared import build_trunk
from instantmimic.rl.models.distributions import (
    diagonal_gaussian_entropy,
    gaussian_tanh_log_prob,
)
from instantmimic.utils.network_precision import network_matmul_precision
from instantmimic.utils.policy_mean_contract import register_policy_mean_methods

from .agent import (
    _UPDATE_EXTRA_METRIC_OFFSET,
    _UPDATE_SUMMARY_EXTRA_OFFSET,
    PPOAgent,
    RunningStandardScaler,
)

_DEFAULT_LATENT_MIN_LOGVAR = -20.0
_DEFAULT_LATENT_MAX_LOGVAR = 10.0


def _bounded_latent_logvar(
    logvar: torch.Tensor,
    *,
    min_logvar: float = _DEFAULT_LATENT_MIN_LOGVAR,
    max_logvar: float = _DEFAULT_LATENT_MAX_LOGVAR,
) -> torch.Tensor:
    logvar_f = logvar.float()
    bounded = logvar_f.clamp(float(min_logvar), float(max_logvar))
    return logvar_f + (bounded - logvar_f).detach()


def compute_latent_ar1_regularization(
    posterior_mu: torch.Tensor, transition_done: torch.Tensor, *, phi: float = 0.99
) -> torch.Tensor:
    if posterior_mu.shape[0] <= 1:
        return posterior_mu.sum() * 0.0
    error = posterior_mu[1:] - float(phi) * posterior_mu[:-1]
    mask = 1.0 - transition_done.to(device=error.device, dtype=error.dtype)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    error = error * mask.clamp(0.0, 1.0)
    return torch.linalg.vector_norm(error, dim=-1).mean()


def compute_latent_ar1_pair_regularization(
    current_posterior_mu: torch.Tensor,
    next_posterior_mu: torch.Tensor,
    transition_done: torch.Tensor,
    *,
    phi: float = 0.99,
) -> torch.Tensor:
    error = next_posterior_mu - float(phi) * current_posterior_mu
    mask = 1.0 - transition_done.to(device=error.device, dtype=error.dtype)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    error = error * mask.clamp(0.0, 1.0)
    return torch.linalg.vector_norm(error, dim=-1).mean()


def _sample_posterior_latent_from_stats(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    *,
    sample_alpha: float,
    eps: torch.Tensor | None = None,
    min_logvar: float = _DEFAULT_LATENT_MIN_LOGVAR,
    max_logvar: float = _DEFAULT_LATENT_MAX_LOGVAR,
) -> torch.Tensor:
    if sample_alpha <= 0.0:
        return posterior_mu
    if eps is None:
        eps = torch.randn_like(posterior_mu)
    else:
        eps = eps.to(device=posterior_mu.device, dtype=posterior_mu.dtype)
        if eps.shape != posterior_mu.shape:
            raise ValueError(
                f"Posterior latent eps shape mismatch: expected {tuple(posterior_mu.shape)}, got {tuple(eps.shape)}."
            )
    posterior_mu_f = posterior_mu.float()
    posterior_std = torch.exp(
        0.5
        * _bounded_latent_logvar(
            posterior_logvar, min_logvar=min_logvar, max_logvar=max_logvar
        )
    )
    sampled = posterior_mu_f + float(sample_alpha) * posterior_std * eps.float()
    return sampled.to(dtype=posterior_mu.dtype)


def _latent_kl_from_stats(
    *,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    min_logvar: float = _DEFAULT_LATENT_MIN_LOGVAR,
    max_logvar: float = _DEFAULT_LATENT_MAX_LOGVAR,
) -> torch.Tensor:
    prior_mu_f = prior_mu.float()
    posterior_mu_f = posterior_mu.float()
    prior_logvar_f = _bounded_latent_logvar(
        prior_logvar, min_logvar=min_logvar, max_logvar=max_logvar
    )
    posterior_logvar_f = _bounded_latent_logvar(
        posterior_logvar, min_logvar=min_logvar, max_logvar=max_logvar
    )
    kl_per_dim = 0.5 * (
        prior_logvar_f
        - posterior_logvar_f
        + torch.exp(posterior_logvar_f - prior_logvar_f)
        + (posterior_mu_f - prior_mu_f).square() * torch.exp(-prior_logvar_f)
        - 1.0
    )
    return kl_per_dim.sum(dim=-1, keepdim=True).mean()


def _evaluate_latent_policy_terms_from_latent(
    network: Any,
    *,
    pre_squash_actions: torch.Tensor | None = None,
    main_obs: torch.Tensor,
    policy_obs: torch.Tensor,
    posterior_latent: torch.Tensor,
    critic_latent: torch.Tensor | None = None,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_prob, values, entropy, _, _, _ = (
        _evaluate_latent_policy_terms_from_latent_with_std_stats(
            network,
            main_obs=main_obs,
            policy_obs=policy_obs,
            posterior_latent=posterior_latent,
            critic_latent=critic_latent,
            actions=actions,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )
    )
    return (log_prob, values, entropy)


def _evaluate_latent_policy_terms_from_latent_with_std_stats(
    network: Any,
    *,
    pre_squash_actions: torch.Tensor | None = None,
    main_obs: torch.Tensor,
    policy_obs: torch.Tensor,
    posterior_latent: torch.Tensor,
    critic_latent: torch.Tensor | None = None,
    actions: torch.Tensor,
    out_policy_mean: torch.Tensor | None = None,
    out_policy_log_std: torch.Tensor | None = None,
    out_raw_log_std_bound_stats: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    actor_features = network._actor_features_from_policy_obs_and_latent(
        policy_obs, posterior_latent
    )
    mean = network._policy_mean_from_features(actor_features)
    if network.clamp_mean:
        mean = mean.clamp(network.clamp_mean_min, network.clamp_mean_max)
    log_std = network._policy_log_std(
        mean,
        features=actor_features,
        out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
    )
    if out_policy_mean is not None:
        assert out_policy_log_std is not None
        out_policy_mean.copy_(mean)
        out_policy_log_std.copy_(log_std)
    std = log_std.exp()
    log_prob = gaussian_tanh_log_prob(
        actions,
        mean,
        log_std,
        std,
        squash=network.squash_actions,
        epsilon=network._squash_epsilon,
        **(
            {"pre_squash_actions": pre_squash_actions}
            if pre_squash_actions is not None
            else {}
        ),
    )
    entropy = diagonal_gaussian_entropy(log_std)
    value_features = _latent_critic_features_from_policy_input(
        network,
        main_obs=main_obs,
        policy_obs=policy_obs,
        latent=posterior_latent if critic_latent is None else critic_latent,
        actor_features=actor_features,
    )
    values = network._mlp_forward(network.value_head, value_features)
    std_mean, std_min, std_max = network._std_stats_from_std(std)
    return (log_prob, values, entropy, std_mean, std_min, std_max)


def _latent_critic_latent_from_posterior(
    network: Any, *, posterior_latent: torch.Tensor, posterior_mu: torch.Tensor
) -> torch.Tensor:
    if bool(getattr(network, "critic_use_detached_posterior_mu", False)):
        return posterior_mu.detach()
    return posterior_latent


def _latent_critic_features_from_policy_input(
    network: Any,
    *,
    main_obs: torch.Tensor,
    policy_obs: torch.Tensor,
    latent: torch.Tensor,
    actor_features: torch.Tensor | None,
) -> torch.Tensor:
    if bool(getattr(network, "use_full_obs_critic", True)):
        return network._mlp_forward(network.trunk, main_obs)
    if bool(getattr(network, "_share_latent_actor_critic", False)) and (
        not bool(getattr(network, "critic_use_detached_posterior_mu", False))
    ):
        if actor_features is None:
            raise RuntimeError("Shared latent actor-critic requires actor_features.")
        return actor_features
    return network._mlp_forward(network.trunk, torch.cat((policy_obs, latent), dim=-1))


def _evaluate_latent_policy_terms_reusing_posterior_with_std_stats(
    network: Any,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    pre_squash_actions: torch.Tensor | None = None,
    policy_aux: torch.Tensor | None = None,
    out_policy_mean: torch.Tensor | None = None,
    out_policy_log_std: torch.Tensor | None = None,
    out_raw_log_std_bound_stats: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    main_obs = obs
    policy_obs = network._extract_policy_obs(main_obs)
    privileged_obs = network._extract_privileged_obs(main_obs)
    posterior_latent, posterior_mu = network._sample_posterior_latent(
        privileged_obs, eps=policy_aux
    )
    return _evaluate_latent_policy_terms_from_latent_with_std_stats(
        network,
        main_obs=main_obs,
        policy_obs=policy_obs,
        posterior_latent=posterior_latent,
        critic_latent=_latent_critic_latent_from_posterior(
            network, posterior_latent=posterior_latent, posterior_mu=posterior_mu
        ),
        actions=actions,
        out_policy_mean=out_policy_mean,
        out_policy_log_std=out_policy_log_std,
        out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
        **(
            {"pre_squash_actions": pre_squash_actions}
            if pre_squash_actions is not None
            else {}
        ),
    )


def _evaluate_latent_with_aux_terms_reusing_posterior(
    network: Any,
    obs: torch.Tensor,
    actions: torch.Tensor,
    *,
    pre_squash_actions: torch.Tensor | None = None,
    policy_aux: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    main_obs = obs
    policy_obs = network._extract_policy_obs(main_obs)
    privileged_obs = network._extract_privileged_obs(main_obs)
    posterior_mu, posterior_logvar = network._posterior_stats_from_privileged_obs(
        privileged_obs
    )
    posterior_latent = _sample_posterior_latent_from_stats(
        posterior_mu,
        posterior_logvar,
        sample_alpha=network.latent_posterior_sample_alpha,
        eps=policy_aux,
        min_logvar=network.latent_min_logvar,
        max_logvar=network.latent_max_logvar,
    )
    log_prob, values, entropy = _evaluate_latent_policy_terms_from_latent(
        network,
        main_obs=main_obs,
        policy_obs=policy_obs,
        posterior_latent=posterior_latent,
        critic_latent=_latent_critic_latent_from_posterior(
            network, posterior_latent=posterior_latent, posterior_mu=posterior_mu
        ),
        actions=actions,
        **(
            {"pre_squash_actions": pre_squash_actions}
            if pre_squash_actions is not None
            else {}
        ),
    )
    prior_mu, prior_logvar = network._prior_stats_from_policy_obs(policy_obs)
    regularization_loss = (
        _latent_kl_from_stats(
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            min_logvar=network.latent_min_logvar,
            max_logvar=network.latent_max_logvar,
        )
        * network.latent_kl_weight
    )
    return (log_prob, values, entropy, regularization_loss)


def _evaluate_latent_with_aux_terms_reusing_posterior_and_std_stats(
    network: Any,
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
    main_obs = obs
    policy_obs = network._extract_policy_obs(main_obs)
    privileged_obs = network._extract_privileged_obs(main_obs)
    posterior_mu, posterior_logvar = network._posterior_stats_from_privileged_obs(
        privileged_obs
    )
    posterior_latent = _sample_posterior_latent_from_stats(
        posterior_mu,
        posterior_logvar,
        sample_alpha=network.latent_posterior_sample_alpha,
        eps=policy_aux,
        min_logvar=network.latent_min_logvar,
        max_logvar=network.latent_max_logvar,
    )
    log_prob, values, entropy, std_mean, std_min, std_max = (
        _evaluate_latent_policy_terms_from_latent_with_std_stats(
            network,
            main_obs=main_obs,
            policy_obs=policy_obs,
            posterior_latent=posterior_latent,
            critic_latent=_latent_critic_latent_from_posterior(
                network, posterior_latent=posterior_latent, posterior_mu=posterior_mu
            ),
            actions=actions,
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
    prior_mu, prior_logvar = network._prior_stats_from_policy_obs(policy_obs)
    regularization_loss = (
        _latent_kl_from_stats(
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            min_logvar=network.latent_min_logvar,
            max_logvar=network.latent_max_logvar,
        )
        * network.latent_kl_weight
    )
    return (log_prob, values, entropy, regularization_loss, std_mean, std_min, std_max)


def is_latent_prior_enabled(cfg: Any) -> bool:
    agent_cfg = getattr(cfg, "agent", None)
    latent_cfg = _cfg_get(agent_cfg, "latent_prior", None)
    return bool(_cfg_get(latent_cfg, "enabled", False))


def resolve_latent_prior_network_build_spec(
    *,
    runtime_spec: Any | None,
    network_cls: type[torch.nn.Module] | None = None,
    network_kwargs: Mapping[str, Any] | None = None,
    latent_prior_cfg: Mapping[str, Any] | None,
    actor_layers: Sequence[int],
    critic_layers: Sequence[int],
    models_separate: bool,
) -> tuple[type[torch.nn.Module] | None, dict[str, Any] | None]:
    latent_cfg = latent_prior_cfg
    if not bool(_cfg_get(latent_cfg, "enabled", False)):
        return (network_cls, None if network_kwargs is None else dict(network_kwargs))
    resolved_network_kwargs = None if network_kwargs is None else dict(network_kwargs)
    resolved_network_cls = network_cls
    if resolved_network_cls is None:
        resolved_network_cls = LatentPriorActorCritic
    if not issubclass(resolved_network_cls, LatentPriorActorCritic):
        return (resolved_network_cls, resolved_network_kwargs)
    if runtime_spec is None:
        raise ValueError("agent.latent_prior.enabled requires runtime_spec.")
    env_raw = getattr(runtime_spec, "env_raw", None)
    get_spec = getattr(env_raw, "get_latent_prior_obs_spec", None)
    if not callable(get_spec):
        raise ValueError(
            "agent.latent_prior.enabled requires env.get_latent_prior_obs_spec()."
        )
    obs_spec = dict(get_spec())
    policy_layers = list(actor_layers)
    value_layers = list(critic_layers)
    defaults = {
        "policy_obs_indices": list(obs_spec["policy_obs_indices"]),
        "privileged_obs_indices": list(obs_spec["privileged_obs_indices"]),
        "latent_dim": int(_cfg_get(latent_cfg, "latent_dim", 16)),
        "posterior_hidden_dims": list(
            _cfg_get(latent_cfg, "posterior_hidden_dims", [256, 256])
        ),
        "prior_hidden_dims": list(
            _cfg_get(latent_cfg, "prior_hidden_dims", [256, 256])
        ),
        "latent_kl_weight": float(_cfg_get(latent_cfg, "kl_weight", 0.001)),
        "latent_eval_mode": str(_cfg_get(latent_cfg, "eval_mode", "prior")),
        "latent_posterior_sample_alpha": float(
            _cfg_get(latent_cfg, "posterior_sample_alpha", 0.0)
        ),
        "latent_ar1_weight": float(_cfg_get(latent_cfg, "ar1_weight", 0.0)),
        "latent_ar1_phi": float(_cfg_get(latent_cfg, "ar1_phi", 0.99)),
        "use_full_obs_critic": bool(_cfg_get(latent_cfg, "use_full_obs_critic", True)),
        "critic_use_detached_posterior_mu": bool(
            _cfg_get(latent_cfg, "critic_use_detached_posterior_mu", False)
        ),
        "latent_min_logvar": float(
            _cfg_get(latent_cfg, "min_logvar", _DEFAULT_LATENT_MIN_LOGVAR)
        ),
        "latent_max_logvar": float(
            _cfg_get(latent_cfg, "max_logvar", _DEFAULT_LATENT_MAX_LOGVAR)
        ),
        "actor_layers": policy_layers,
        "critic_layers": value_layers,
        "separate": bool(models_separate),
    }
    if resolved_network_kwargs is None:
        resolved_network_kwargs = defaults
    else:
        for key, value in defaults.items():
            resolved_network_kwargs.setdefault(key, value)
    return (resolved_network_cls, resolved_network_kwargs)


def _artifact_ppo_model_settings(config: SemanticConfig) -> dict[str, object]:
    build_spec = lower_agent_build_spec(config)
    if not isinstance(build_spec, PPOAgentBuildSpec):
        raise RuntimeError("Low-level latent checkpoint must describe a PPO agent.")
    actor = build_spec.actor
    distribution = actor.distribution
    if distribution is None:
        raise RuntimeError(
            "Low-level PPO artifact is missing actor distribution settings."
        )
    architectures = (
        actor.architecture,
        build_spec.critic.architecture if build_spec.critic else "mlp",
    )
    if any(architecture != "mlp" for architecture in architectures):
        raise RuntimeError(
            "Frozen latent policies currently require flat MLP actor and critic networks"
        )
    preprocessor = (
        _cfg_get(config.agent, "state_preprocessor", None) or "RunningStandardScaler"
    )
    if preprocessor != "RunningStandardScaler":
        raise RuntimeError(
            "Frozen latent policies currently require RunningStandardScaler observations"
        )
    if distribution.parameterization != "log":
        raise RuntimeError(
            "Frozen latent policies currently require log standard-deviation parameters"
        )
    critic = build_spec.critic
    actor_kwargs = actor.constructor_kwargs
    critic_kwargs = critic.constructor_kwargs if critic is not None else actor_kwargs
    return {
        "squash_actions": distribution.action_transform == "tanh",
        "activation": actor_kwargs["activation"],
        "initial_log_std": distribution.initial_log_std,
        "min_log_std": distribution.log_std_bounds[0],
        "max_log_std": distribution.log_std_bounds[1],
        "fixed_log_std": distribution.fixed_log_std,
        "fixed_sigma": distribution.fixed_sigma,
        "log_std_bound_mode": distribution.bound_mode,
        "network_type": actor.architecture if build_spec.shared else "mlp",
        "actor_network_type": None if build_spec.shared else actor.architecture,
        "critic_network_type": None
        if build_spec.shared or critic is None
        else critic.architecture,
        "use_layer_norm": actor_kwargs["normalization"] == "layer_norm",
        "use_rms_norm": False,
        "residual_scale": actor_kwargs.get(
            "residual_scale", critic_kwargs.get("residual_scale", 1.0)
        ),
        "separate": not build_spec.shared,
    }


def _require_state_dict(
    payload: Mapping[str, Any], key: str
) -> dict[str, torch.Tensor]:
    state = payload.get(key)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint is missing a '{key}' state_dict.")
    return state


def _require_state_tensor(network_state: Mapping[str, Any], key: str) -> torch.Tensor:
    value = network_state.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Checkpoint network state is missing tensor '{key}'.")
    return value


def _tensor_to_index_list(network_state: Mapping[str, Any], key: str) -> list[int]:
    tensor = _require_state_tensor(network_state, key)
    if tensor.ndim != 1:
        raise RuntimeError(
            f"Checkpoint tensor '{key}' must be 1D, got shape {tuple(tensor.shape)}."
        )
    return [int(v) for v in tensor.detach().cpu().tolist()]


def _infer_trunk_hidden_dims(
    network_state: Mapping[str, Any], prefix: str
) -> list[int]:
    indexed_weights: list[tuple[int, torch.Tensor]] = []
    prefix_parts = prefix.split(".")
    for key, value in network_state.items():
        parts = key.split(".")
        if (
            parts[: len(prefix_parts)] == prefix_parts
            and len(parts) == len(prefix_parts) + 2
            and (parts[-1] == "weight")
            and parts[-2].isdigit()
            and isinstance(value, torch.Tensor)
            and (value.ndim == 2)
        ):
            indexed_weights.append((int(parts[-2]), value))
    dims = [int(weight.shape[0]) for _, weight in sorted(indexed_weights)]
    if not dims:
        raise RuntimeError(f"Checkpoint is missing MLP weights for prefix '{prefix}'.")
    return dims


def _validate_runtime_obs_contract(
    *,
    checkpoint_policy_obs_indices: Sequence[int],
    checkpoint_privileged_obs_indices: Sequence[int],
    runtime_obs_spec: Mapping[str, Any],
    obs_main_dim: int,
) -> None:
    runtime_policy_obs_indices = [
        int(v) for v in runtime_obs_spec["policy_obs_indices"]
    ]
    runtime_privileged_obs_indices = [
        int(v) for v in runtime_obs_spec["privileged_obs_indices"]
    ]
    if list(checkpoint_policy_obs_indices) != runtime_policy_obs_indices:
        raise RuntimeError(
            f"Low-level checkpoint policy_obs_indices do not match the current env runtime spec: checkpoint={list(checkpoint_policy_obs_indices)} runtime={runtime_policy_obs_indices}"
        )
    if list(checkpoint_privileged_obs_indices) != runtime_privileged_obs_indices:
        raise RuntimeError(
            f"Low-level checkpoint privileged_obs_indices do not match the current env runtime spec: checkpoint={list(checkpoint_privileged_obs_indices)} runtime={runtime_privileged_obs_indices}"
        )
    combined = list(checkpoint_policy_obs_indices) + list(
        checkpoint_privileged_obs_indices
    )
    if combined and max(combined) >= int(obs_main_dim):
        raise RuntimeError(
            f"Low-level checkpoint observation indices exceed the current env obs_main_dim: max_index={max(combined)} obs_main_dim={int(obs_main_dim)}"
        )


class LatentPriorActorCritic(ActorCritic):
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
        *,
        policy_obs_indices: Sequence[int],
        privileged_obs_indices: Sequence[int],
        latent_dim: int,
        posterior_hidden_dims: Sequence[int],
        prior_hidden_dims: Sequence[int],
        latent_kl_weight: float,
        latent_eval_mode: str = "prior",
        latent_posterior_sample_alpha: float = 0.0,
        latent_ar1_weight: float = 0.0,
        latent_ar1_phi: float = 0.99,
        use_full_obs_critic: bool = True,
        critic_use_detached_posterior_mu: bool = False,
        latent_min_logvar: float = _DEFAULT_LATENT_MIN_LOGVAR,
        latent_max_logvar: float = _DEFAULT_LATENT_MAX_LOGVAR,
        separate: bool = False,
        policy_mlp_bf16: bool = False,
    ) -> None:
        effective_separate = bool(separate or use_full_obs_critic)
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            layers=layers,
            activation=activation,
            initial_log_std=initial_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            fixed_log_std=fixed_log_std,
            fixed_sigma=fixed_sigma,
            squash_actions=squash_actions,
            clamp_mean=clamp_mean,
            clamp_mean_min=clamp_mean_min,
            clamp_mean_max=clamp_mean_max,
            log_std_bound_mode=log_std_bound_mode,
            log_std_bound_loss_weight=log_std_bound_loss_weight,
            log_std_bound_loss_margin=log_std_bound_loss_margin,
            actor_layers=actor_layers,
            critic_layers=critic_layers,
            network_type=network_type,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_layer_norm=use_layer_norm,
            use_rms_norm=use_rms_norm,
            residual_scale=residual_scale,
            separate=effective_separate,
            policy_mlp_bf16=policy_mlp_bf16,
        )
        self.obs_main_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        self.latent_kl_weight = float(latent_kl_weight)
        self.latent_eval_mode = str(latent_eval_mode).strip().lower()
        if self.latent_eval_mode not in {"prior", "posterior"}:
            raise ValueError(
                f"latent_eval_mode must be 'prior' or 'posterior', got {latent_eval_mode!r}."
            )
        self.latent_posterior_sample_alpha = float(latent_posterior_sample_alpha)
        if self.latent_posterior_sample_alpha < 0.0:
            raise ValueError(
                f"latent_posterior_sample_alpha must be >= 0, got {latent_posterior_sample_alpha!r}."
            )
        self.latent_ar1_weight = float(latent_ar1_weight)
        if not math.isfinite(self.latent_ar1_weight) or self.latent_ar1_weight < 0.0:
            raise ValueError(
                f"latent_ar1_weight must be a finite value >= 0, got {latent_ar1_weight!r}."
            )
        self.latent_ar1_phi = float(latent_ar1_phi)
        if not math.isfinite(self.latent_ar1_phi):
            raise ValueError(f"latent_ar1_phi must be finite, got {latent_ar1_phi!r}.")
        self.transition_aux_regularization_weight = self.latent_ar1_weight
        self.latent_min_logvar = float(latent_min_logvar)
        self.latent_max_logvar = float(latent_max_logvar)
        if (
            not math.isfinite(self.latent_min_logvar)
            or not math.isfinite(self.latent_max_logvar)
            or self.latent_min_logvar >= self.latent_max_logvar
        ):
            raise ValueError(
                f"latent logvar bounds must be finite and min_logvar < max_logvar, got [{latent_min_logvar!r}, {latent_max_logvar!r}]."
            )
        self._policy_aux_dim = (
            self.latent_dim if self.latent_posterior_sample_alpha > 0.0 else 0
        )
        policy_idx = torch.as_tensor(policy_obs_indices, dtype=torch.long)
        privileged_idx = torch.as_tensor(privileged_obs_indices, dtype=torch.long)
        if policy_idx.ndim != 1 or privileged_idx.ndim != 1:
            raise ValueError(
                "policy_obs_indices and privileged_obs_indices must be 1D."
            )
        if policy_idx.numel() <= 0 or privileged_idx.numel() <= 0:
            raise ValueError(
                "policy_obs_indices and privileged_obs_indices must be non-empty."
            )
        self.register_buffer("policy_obs_indices", policy_idx)
        self.register_buffer("privileged_obs_indices", privileged_idx)
        self.policy_obs_dim = int(policy_idx.numel())
        self.privileged_obs_dim = int(privileged_idx.numel())
        self.use_full_obs_critic = bool(use_full_obs_critic)
        self.critic_use_detached_posterior_mu = bool(critic_use_detached_posterior_mu)
        self._latent_separate_actor_critic = bool(separate)
        self._share_latent_actor_critic = bool(
            not self.use_full_obs_critic and (not self._latent_separate_actor_critic)
        )
        latent_policy_input_dim = self.policy_obs_dim + self.latent_dim
        resolved_actor_layers = (
            list(actor_layers) if actor_layers is not None else list(layers)
        )
        resolved_critic_layers = (
            list(critic_layers) if critic_layers is not None else list(layers)
        )
        resolved_actor_network_type = actor_network_type or network_type
        resolved_critic_network_type = critic_network_type or network_type
        if self.use_full_obs_critic:
            self.actor_trunk, actor_out_dim = build_trunk(
                latent_policy_input_dim,
                resolved_actor_layers,
                activation,
                network_type=resolved_actor_network_type,
                use_layer_norm=use_layer_norm,
                use_rms_norm=use_rms_norm,
                residual_scale=residual_scale,
            )
        else:
            trunk_layers = (
                list(layers)
                if self._share_latent_actor_critic
                else resolved_critic_layers
            )
            trunk_network_type = (
                network_type
                if self._share_latent_actor_critic
                else resolved_critic_network_type
            )
            self.trunk, critic_out_dim = build_trunk(
                latent_policy_input_dim,
                trunk_layers,
                activation,
                network_type=trunk_network_type,
                use_layer_norm=use_layer_norm,
                use_rms_norm=use_rms_norm,
                residual_scale=residual_scale,
            )
            self.value_head = torch.nn.Linear(critic_out_dim, 1)
            if self._share_latent_actor_critic:
                self.actor_trunk = self.trunk
                actor_out_dim = critic_out_dim
            else:
                self.actor_trunk, actor_out_dim = build_trunk(
                    latent_policy_input_dim,
                    resolved_actor_layers,
                    activation,
                    network_type=resolved_actor_network_type,
                    use_layer_norm=use_layer_norm,
                    use_rms_norm=use_rms_norm,
                    residual_scale=residual_scale,
                )
        self.actor_feature_dim = int(actor_out_dim)
        self.critic_feature_dim = int(self.value_head.in_features)
        self.policy_head = torch.nn.Linear(actor_out_dim, self.act_dim)
        self.posterior_trunk, posterior_out_dim = build_trunk(
            self.privileged_obs_dim, posterior_hidden_dims, activation
        )
        self.posterior_mu_head = torch.nn.Linear(posterior_out_dim, self.latent_dim)
        self.posterior_logvar_head = torch.nn.Linear(posterior_out_dim, self.latent_dim)
        self.prior_trunk, prior_out_dim = build_trunk(
            self.policy_obs_dim, prior_hidden_dims, activation
        )
        self.prior_mu_head = torch.nn.Linear(prior_out_dim, self.latent_dim)
        self.prior_logvar_head = torch.nn.Linear(prior_out_dim, self.latent_dim)
        self._use_standard_policy_fast_path = False
        self._use_standard_policy_evaluate_fast_path = False

    policy_obs_indices: torch.Tensor
    privileged_obs_indices: torch.Tensor
    prior_trunk: torch.nn.Module
    prior_mu_head: torch.nn.Module
    prior_logvar_head: torch.nn.Module
    posterior_trunk: torch.nn.Module
    posterior_mu_head: torch.nn.Module
    posterior_logvar_head: torch.nn.Module
    latent_min_logvar: float
    latent_max_logvar: float
    latent_posterior_sample_alpha: float
    latent_kl_weight: float
    latent_ar1_weight: float
    latent_ar1_phi: float
    zero_scalar: Callable[[], torch.Tensor]

    def _extract_policy_obs(self, main_obs: torch.Tensor) -> torch.Tensor:
        return torch.index_select(main_obs, -1, self.policy_obs_indices)

    def _extract_privileged_obs(self, main_obs: torch.Tensor) -> torch.Tensor:
        return torch.index_select(main_obs, -1, self.privileged_obs_indices)

    def _prior_stats_from_policy_obs(
        self, policy_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._mlp_forward(self.prior_trunk, policy_obs)
        return (
            self._mlp_forward(self.prior_mu_head, features),
            self._mlp_forward(self.prior_logvar_head, features),
        )

    def _posterior_stats_from_privileged_obs(
        self, privileged_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._mlp_forward(self.posterior_trunk, privileged_obs)
        return (
            self._mlp_forward(self.posterior_mu_head, features),
            self._mlp_forward(self.posterior_logvar_head, features),
        )

    def _sample_latent(
        self, mu: torch.Tensor, logvar: torch.Tensor, *, eps: torch.Tensor | None = None
    ) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(mu)
        else:
            eps = eps.to(device=mu.device, dtype=mu.dtype)
            if eps.shape != mu.shape:
                raise ValueError(
                    f"Latent eps shape mismatch: expected {tuple(mu.shape)}, got {tuple(eps.shape)}."
                )
        return _sample_posterior_latent_from_stats(
            mu,
            logvar,
            sample_alpha=1.0,
            eps=eps,
            min_logvar=self.latent_min_logvar,
            max_logvar=self.latent_max_logvar,
        )

    def _sample_posterior_latent(
        self, privileged_obs: torch.Tensor, *, eps: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        posterior_mu, posterior_logvar = self._posterior_stats_from_privileged_obs(
            privileged_obs
        )
        sampled_latent = _sample_posterior_latent_from_stats(
            posterior_mu,
            posterior_logvar,
            sample_alpha=self.latent_posterior_sample_alpha,
            eps=eps,
            min_logvar=self.latent_min_logvar,
            max_logvar=self.latent_max_logvar,
        )
        return (sampled_latent, posterior_mu)

    def _latent_kl_from_main_obs(self, main_obs: torch.Tensor) -> torch.Tensor:
        policy_obs = self._extract_policy_obs(main_obs)
        privileged_obs = self._extract_privileged_obs(main_obs)
        prior_mu, prior_logvar = self._prior_stats_from_policy_obs(policy_obs)
        posterior_mu, posterior_logvar = self._posterior_stats_from_privileged_obs(
            privileged_obs
        )
        return _latent_kl_from_stats(
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            min_logvar=self.latent_min_logvar,
            max_logvar=self.latent_max_logvar,
        )

    def weighted_latent_kl_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        main_obs = obs
        return self._latent_kl_from_main_obs(main_obs) * self.latent_kl_weight

    def _posterior_mu_from_main_obs(self, main_obs: torch.Tensor) -> torch.Tensor:
        privileged_obs = self._extract_privileged_obs(main_obs)
        posterior_mu, _ = self._posterior_stats_from_privileged_obs(privileged_obs)
        return posterior_mu

    def latent_ar1_regularization_from_obs_sequence(
        self, obs: torch.Tensor, transition_done: torch.Tensor
    ) -> torch.Tensor:
        main_obs = obs
        posterior_mu = self._posterior_mu_from_main_obs(main_obs)
        return compute_latent_ar1_regularization(
            posterior_mu, transition_done, phi=self.latent_ar1_phi
        )

    def transition_aux_regularization_loss(
        self, obs: torch.Tensor, next_obs: torch.Tensor, transition_done: torch.Tensor
    ) -> torch.Tensor:
        if self.latent_ar1_weight <= 0.0:
            return self.zero_scalar()
        main_obs = obs
        next_main_obs = next_obs
        posterior_mu = self._posterior_mu_from_main_obs(main_obs)
        next_posterior_mu = self._posterior_mu_from_main_obs(next_main_obs)
        ar1 = compute_latent_ar1_pair_regularization(
            posterior_mu, next_posterior_mu, transition_done, phi=self.latent_ar1_phi
        )
        return ar1 * self.latent_ar1_weight

    def weighted_latent_ar1_from_obs_sequence(
        self, obs: torch.Tensor, transition_done: torch.Tensor
    ) -> torch.Tensor:
        if self.latent_ar1_weight <= 0.0:
            return self.zero_scalar()
        return (
            self.latent_ar1_regularization_from_obs_sequence(obs, transition_done)
            * self.latent_ar1_weight
        )

    def _actor_features_from_policy_obs_and_latent(
        self, policy_obs: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        latent = latent.to(device=policy_obs.device, dtype=policy_obs.dtype)
        return self._mlp_forward(
            self.actor_trunk, torch.cat((policy_obs, latent), dim=-1)
        )

    def latent_policy_mean(
        self, obs: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        main_obs = obs
        policy_obs = self._extract_policy_obs(main_obs)
        features = self._actor_features_from_policy_obs_and_latent(policy_obs, latent)
        mean = self._policy_mean_from_features(features)
        if self.squash_actions:
            mean = mean.tanh()
        elif self.clamp_mean:
            mean = mean.clamp(self.clamp_mean_min, self.clamp_mean_max)
        return mean

    def _policy_mean_from_main_obs(
        self, main_obs: torch.Tensor, *, mode: str
    ) -> torch.Tensor:
        policy_obs = self._extract_policy_obs(main_obs)
        if mode == "prior":
            latent, _ = self._prior_stats_from_policy_obs(policy_obs)
        else:
            latent, _ = self._posterior_stats_from_privileged_obs(
                self._extract_privileged_obs(main_obs)
            )
        features = self._actor_features_from_policy_obs_and_latent(policy_obs, latent)
        return self._policy_mean_from_features(features)

    def policy_mean_into(self, obs: torch.Tensor, out_actions: torch.Tensor) -> None:
        main_obs = obs
        mean = self._policy_mean_from_main_obs(main_obs, mode=self.latent_eval_mode)
        out_actions.copy_(mean)
        if self.squash_actions:
            out_actions.tanh_()
        elif self.clamp_mean:
            out_actions.clamp_(self.clamp_mean_min, self.clamp_mean_max)

    def sample_prior_latent(
        self,
        obs: torch.Tensor,
        *,
        eps: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        main_obs = obs
        policy_obs = self._extract_policy_obs(main_obs)
        prior_mu, prior_logvar = self._prior_stats_from_policy_obs(policy_obs)
        if eps is None:
            eps = torch.randn_like(prior_mu)
        else:
            eps = eps.to(device=prior_mu.device, dtype=prior_mu.dtype)
            if eps.shape != prior_mu.shape:
                raise ValueError(
                    f"Prior latent eps shape mismatch: expected {tuple(prior_mu.shape)}, got {tuple(eps.shape)}."
                )
        temperature = float(temperature)
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0.")
        return _sample_posterior_latent_from_stats(
            prior_mu,
            prior_logvar,
            sample_alpha=temperature,
            eps=eps,
            min_logvar=self.latent_min_logvar,
            max_logvar=self.latent_max_logvar,
        )

    def prior_mean(self, obs: torch.Tensor) -> torch.Tensor:
        main_obs = obs
        policy_obs = self._extract_policy_obs(main_obs)
        prior_mu, _ = self._prior_stats_from_policy_obs(policy_obs)
        return prior_mu

    def policy_latent_mean_into(
        self, obs: torch.Tensor, latent: torch.Tensor, out_actions: torch.Tensor
    ) -> None:
        out_actions.copy_(self.latent_policy_mean(obs, latent))

    def policy_prior_sample_into(
        self,
        obs: torch.Tensor,
        out_actions: torch.Tensor,
        *,
        eps: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> None:
        main_obs = obs
        policy_obs = self._extract_policy_obs(main_obs)
        latent = self.sample_prior_latent(obs, eps=eps, temperature=temperature)
        actor_features = self._actor_features_from_policy_obs_and_latent(
            policy_obs, latent
        )
        mean = self._policy_mean_from_features(actor_features)
        out_actions.copy_(mean)
        if self.squash_actions:
            out_actions.tanh_()
        elif self.clamp_mean:
            out_actions.clamp_(self.clamp_mean_min, self.clamp_mean_max)

    def act_into(
        self,
        obs: torch.Tensor,
        out_actions: torch.Tensor,
        out_log_probs: torch.Tensor,
        out_values: torch.Tensor,
        eps: torch.Tensor | None = None,
        *,
        policy_aux: torch.Tensor | None = None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_pre_squash_actions: torch.Tensor | None = None,
    ) -> None:
        main_obs = obs
        policy_obs = self._extract_policy_obs(main_obs)
        privileged_obs = self._extract_privileged_obs(main_obs)
        posterior_latent, posterior_mu = self._sample_posterior_latent(
            privileged_obs, eps=policy_aux
        )
        actor_features = self._actor_features_from_policy_obs_and_latent(
            policy_obs, posterior_latent
        )
        value_features = _latent_critic_features_from_policy_input(
            self,
            main_obs=main_obs,
            policy_obs=policy_obs,
            latent=_latent_critic_latent_from_posterior(
                self, posterior_latent=posterior_latent, posterior_mu=posterior_mu
            ),
            actor_features=actor_features,
        )
        mean = self._policy_mean_from_features(actor_features)
        self._sample_policy_into_outputs(
            features=actor_features,
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

    def _evaluate_policy_terms(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        main_obs = obs
        policy_obs = self._extract_policy_obs(main_obs)
        privileged_obs = self._extract_privileged_obs(main_obs)
        posterior_latent, posterior_mu = self._sample_posterior_latent(
            privileged_obs, eps=policy_aux
        )
        actor_features = self._actor_features_from_policy_obs_and_latent(
            policy_obs, posterior_latent
        )
        mean = self._policy_mean_from_features(actor_features)
        if self.clamp_mean:
            mean = mean.clamp(self.clamp_mean_min, self.clamp_mean_max)
        log_std = self._policy_log_std(mean, features=actor_features)
        std = log_std.exp()
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
        value_features = _latent_critic_features_from_policy_input(
            self,
            main_obs=main_obs,
            policy_obs=policy_obs,
            latent=_latent_critic_latent_from_posterior(
                self, posterior_latent=posterior_latent, posterior_mu=posterior_mu
            ),
            actor_features=actor_features,
        )
        values = self._mlp_forward(self.value_head, value_features)
        return (log_prob, values, entropy)

    def _evaluate_policy_terms_with_std_stats(
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
    ]:
        return _evaluate_latent_policy_terms_reusing_posterior_with_std_stats(
            self,
            obs,
            actions,
            policy_aux=policy_aux,
            out_policy_mean=out_policy_mean,
            out_policy_log_std=out_policy_log_std,
            out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )

    def _evaluate_with_aux_terms(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _evaluate_latent_with_aux_terms_reusing_posterior(
            self,
            obs,
            actions,
            policy_aux=policy_aux,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )

    def _evaluate_with_aux_terms_with_std_stats(
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
        return _evaluate_latent_with_aux_terms_reusing_posterior_and_std_stats(
            self,
            obs,
            actions,
            policy_aux=policy_aux,
            out_policy_mean=out_policy_mean,
            out_policy_log_std=out_policy_log_std,
            out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )

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
        log_prob, values, entropy = self._evaluate_policy_terms(
            obs,
            actions,
            policy_aux=policy_aux,
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
        log_prob, values, entropy = self._evaluate_policy_terms(
            obs,
            actions,
            policy_aux=policy_aux,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )
        return (log_prob, values, entropy)

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
        policy_aux: torch.Tensor | None = None,
        out_policy_mean: torch.Tensor | None = None,
        out_policy_log_std: torch.Tensor | None = None,
        out_raw_log_std_bound_stats: torch.Tensor | None = None,
    ) -> None:
        log_prob, values, entropy, std_mean, std_min, std_max = (
            self._evaluate_policy_terms_with_std_stats(
                obs,
                actions,
                policy_aux=policy_aux,
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
    ]:
        log_prob, values, entropy, std_mean, std_min, std_max = (
            self._evaluate_policy_terms_with_std_stats(
                obs,
                actions,
                policy_aux=policy_aux,
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
        return (log_prob, values, entropy, std_mean, std_min, std_max)

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
        log_prob, values, entropy, regularization_loss = self._evaluate_with_aux_terms(
            obs,
            actions,
            policy_aux=policy_aux,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )
        out_log_probs.copy_(log_prob)
        out_values.copy_(values)
        out_entropy.copy_(entropy)
        out_aux_loss.copy_(regularization_loss)

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
        log_prob, values, entropy, regularization_loss, std_mean, std_min, std_max = (
            self._evaluate_with_aux_terms_with_std_stats(
                obs,
                actions,
                policy_aux=policy_aux,
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
        out_aux_loss.copy_(regularization_loss)
        out_std_mean.copy_(std_mean)
        out_std_min.copy_(std_min)
        out_std_max.copy_(std_max)

    def evaluate_with_aux_losses(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        *,
        pre_squash_actions: torch.Tensor | None = None,
        policy_aux: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._evaluate_with_aux_terms(
            obs,
            actions,
            policy_aux=policy_aux,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
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
        return self._evaluate_with_aux_terms_with_std_stats(
            obs,
            actions,
            policy_aux=policy_aux,
            out_policy_mean=out_policy_mean,
            out_policy_log_std=out_policy_log_std,
            out_raw_log_std_bound_stats=out_raw_log_std_bound_stats,
            **(
                {"pre_squash_actions": pre_squash_actions}
                if pre_squash_actions is not None
                else {}
            ),
        )


class LatentPriorPPOAgent(PPOAgent):
    def _requires_update_aux_evaluate(self) -> bool:
        return True

    def _extra_update_metric_count(self) -> int:
        return 2

    def _extra_update_summary_keys(self) -> tuple[str, ...]:
        return (
            "policy_aux_loss_total",
            "latent_prior_kl_weighted",
            "latent_prior_kl",
            "latent_prior_ar1_weighted",
            "latent_prior_ar1",
        )

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
        del batch_actions, batch_next_obs, batch_dones
        weighted_kl = policy_aux_loss
        return (weighted_kl.float(), transition_aux_loss.float())

    def _write_extra_update_summary_from_metrics(
        self, mean_metrics: torch.Tensor
    ) -> None:
        policy_aux_total = self._update_summary[3].clone()
        latent_kl_weighted = mean_metrics[_UPDATE_EXTRA_METRIC_OFFSET]
        latent_ar1_weighted = mean_metrics[_UPDATE_EXTRA_METRIC_OFFSET + 1]
        self._update_summary[3].copy_(
            policy_aux_total - latent_kl_weighted - latent_ar1_weighted
        )
        summary_offset = _UPDATE_SUMMARY_EXTRA_OFFSET
        self._update_summary[summary_offset].copy_(policy_aux_total)
        self._update_summary[summary_offset + 1].copy_(latent_kl_weighted)
        kl_weight = float(getattr(self.network, "latent_kl_weight", 0.0))
        if abs(kl_weight) > 1e-12:
            self._update_summary[summary_offset + 2].copy_(
                latent_kl_weighted / kl_weight
            )
        else:
            self._update_summary[summary_offset + 2].zero_()
        self._update_summary[summary_offset + 3].copy_(latent_ar1_weighted)
        ar1_weight = float(getattr(self.network, "latent_ar1_weight", 0.0))
        if abs(ar1_weight) > 1e-12:
            self._update_summary[summary_offset + 4].copy_(
                latent_ar1_weighted / ar1_weight
            )
        else:
            self._update_summary[summary_offset + 4].zero_()

    @torch.no_grad()
    def policy_prior_sample_into(
        self,
        obs: torch.Tensor,
        out_actions: torch.Tensor,
        *,
        eps: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> None:
        with network_matmul_precision():
            scaled_obs = self._scaled_eval_obs(obs)
            sample_fn = getattr(self.network, "policy_prior_sample_into", None)
            if not callable(sample_fn):
                raise RuntimeError(
                    "policy_prior_sample_into() requires a network that exposes policy_prior_sample_into()."
                )
            sample_fn(
                scaled_obs,
                out_actions,
                eps=None if eps is None else eps.to(device=scaled_obs.device),
                temperature=float(temperature),
            )

    @torch.no_grad()
    def sample_prior_latent(
        self,
        obs: torch.Tensor,
        *,
        eps: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        with network_matmul_precision():
            scaled_obs = self._scaled_eval_obs(obs)
            sample_fn = getattr(self.network, "sample_prior_latent", None)
            if not callable(sample_fn):
                raise RuntimeError(
                    "sample_prior_latent() requires a network that exposes sample_prior_latent()."
                )
            return sample_fn(
                scaled_obs,
                eps=None if eps is None else eps.to(device=scaled_obs.device),
                temperature=float(temperature),
            )

    @torch.no_grad()
    def prior_mean(self, obs: torch.Tensor) -> torch.Tensor:
        with network_matmul_precision():
            scaled_obs = self._scaled_eval_obs(obs)
            mean_fn = getattr(self.network, "prior_mean", None)
            if not callable(mean_fn):
                raise RuntimeError(
                    "prior_mean() requires a network that exposes prior_mean()."
                )
            return mean_fn(scaled_obs)

    @torch.no_grad()
    def policy_latent_mean_into(
        self, obs: torch.Tensor, latent: torch.Tensor, out_actions: torch.Tensor
    ) -> None:
        with network_matmul_precision():
            scaled_obs = self._scaled_eval_obs(obs)
            mean_fn = getattr(self.network, "policy_latent_mean_into", None)
            if not callable(mean_fn):
                raise RuntimeError(
                    "policy_latent_mean_into() requires a network that exposes policy_latent_mean_into()."
                )
            mean_fn(scaled_obs, latent.to(device=scaled_obs.device), out_actions)


class FrozenLatentPolicyAdapter:
    """Frozen Gaussian VAE decoder and prior reconstructed from a current checkpoint."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: torch.device | str,
        obs_dim: int,
        act_dim: int,
        layers: Sequence[int],
        critic_layers: Sequence[int] | None = None,
        activation: str,
        initial_log_std: float,
        min_log_std: float,
        max_log_std: float,
        fixed_log_std: bool = False,
        fixed_sigma: bool = True,
        log_std_bound_mode: str = "clamp",
        squash_actions: bool = False,
        policy_obs_indices: Sequence[int],
        privileged_obs_indices: Sequence[int],
        latent_dim: int,
        posterior_hidden_dims: Sequence[int],
        prior_hidden_dims: Sequence[int],
        latent_eval_mode: str = "prior",
        network_type: str = "mlp",
        actor_network_type: str | None = None,
        critic_network_type: str | None = None,
        use_layer_norm: bool = False,
        use_rms_norm: bool = False,
        residual_scale: float = 1.0,
        latent_min_logvar: float = _DEFAULT_LATENT_MIN_LOGVAR,
        latent_max_logvar: float = _DEFAULT_LATENT_MAX_LOGVAR,
        use_full_obs_critic: bool = True,
        separate: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.obs_main_dim = self.obs_dim
        if self.obs_dim <= 0:
            raise ValueError("Frozen policy requires a positive observation dimension")
        self.network = LatentPriorActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            layers=layers,
            actor_layers=layers,
            critic_layers=critic_layers,
            activation=activation,
            initial_log_std=initial_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            fixed_log_std=fixed_log_std,
            fixed_sigma=fixed_sigma,
            log_std_bound_mode=log_std_bound_mode,
            squash_actions=squash_actions,
            policy_obs_indices=policy_obs_indices,
            privileged_obs_indices=privileged_obs_indices,
            latent_dim=latent_dim,
            posterior_hidden_dims=posterior_hidden_dims,
            prior_hidden_dims=prior_hidden_dims,
            latent_kl_weight=0.0,
            latent_eval_mode=latent_eval_mode,
            network_type=network_type,
            actor_network_type=actor_network_type,
            critic_network_type=critic_network_type,
            use_layer_norm=use_layer_norm,
            use_rms_norm=use_rms_norm,
            residual_scale=residual_scale,
            latent_min_logvar=latent_min_logvar,
            latent_max_logvar=latent_max_logvar,
            use_full_obs_critic=use_full_obs_critic,
            separate=separate,
        ).to(self.device)
        self._load_and_freeze_checkpoint(
            checkpoint_path=checkpoint_path, policy_obs_indices=policy_obs_indices
        )

    @torch.no_grad()
    def prior_mean(self, obs: torch.Tensor) -> torch.Tensor:
        full_obs = obs.to(device=self.device, dtype=torch.float32)
        scaled_obs = self._scale_obs(full_obs)
        return self.network.prior_mean(scaled_obs)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        device: torch.device | str,
        obs_dim: int,
        act_dim: int,
        runtime_obs_spec: Mapping[str, Any],
    ) -> FrozenLatentPolicyAdapter:
        checkpoint_path = Path(checkpoint_path)
        resolved_cfg = load_artifact_semantic_config(checkpoint_path)
        model_settings = _artifact_ppo_model_settings(resolved_cfg)
        payload = torch.load(checkpoint_path, map_location="cpu")
        network_state = _require_state_dict(payload, "network")
        policy_obs_indices = _tensor_to_index_list(network_state, "policy_obs_indices")
        privileged_obs_indices = _tensor_to_index_list(
            network_state, "privileged_obs_indices"
        )
        obs_main_dim = int(obs_dim) - int(0)
        if obs_main_dim <= 0:
            raise RuntimeError(
                f"Low-level checkpoint requires a positive observation dimension, got {int(obs_dim)}."
            )
        _validate_runtime_obs_contract(
            checkpoint_policy_obs_indices=policy_obs_indices,
            checkpoint_privileged_obs_indices=privileged_obs_indices,
            runtime_obs_spec=runtime_obs_spec,
            obs_main_dim=obs_main_dim,
        )
        actor_layers = _infer_trunk_hidden_dims(network_state, "actor_trunk")
        critic_layers = _infer_trunk_hidden_dims(network_state, "trunk")
        posterior_hidden_dims = _infer_trunk_hidden_dims(
            network_state, "posterior_trunk"
        )
        prior_hidden_dims = _infer_trunk_hidden_dims(network_state, "prior_trunk")
        latent_dim = int(
            _require_state_tensor(network_state, "prior_mu_head.bias").numel()
        )
        checkpoint_act_dim = int(
            _require_state_tensor(network_state, "log_std").numel()
        )
        if checkpoint_act_dim != int(act_dim):
            raise RuntimeError(
                f"Low-level checkpoint action dimension does not match the current env: checkpoint={checkpoint_act_dim} runtime={int(act_dim)}"
            )
        actor_input_dim = int(
            _require_state_tensor(network_state, "actor_trunk.0.weight").shape[1]
        )
        if actor_input_dim != len(policy_obs_indices) + latent_dim:
            raise RuntimeError(
                f"Low-level checkpoint actor trunk input width does not match the latent observation contract: actor_input_dim={actor_input_dim} policy_obs_dim={len(policy_obs_indices)} latent_dim={latent_dim}"
            )
        posterior_input_dim = int(
            _require_state_tensor(network_state, "posterior_trunk.0.weight").shape[1]
        )
        if posterior_input_dim != len(privileged_obs_indices):
            raise RuntimeError(
                f"Low-level checkpoint posterior trunk input width does not match the privileged observation contract: checkpoint={posterior_input_dim} runtime={len(privileged_obs_indices)}"
            )
        prior_input_dim = int(
            _require_state_tensor(network_state, "prior_trunk.0.weight").shape[1]
        )
        if prior_input_dim != len(policy_obs_indices):
            raise RuntimeError(
                f"Low-level checkpoint prior trunk input width does not match the policy observation contract: checkpoint={prior_input_dim} runtime={len(policy_obs_indices)}"
            )
        latent_cfg = _cfg_get(
            getattr(resolved_cfg, "agent", None), "latent_prior", None
        )
        if not bool(_cfg_get(latent_cfg, "enabled", False)):
            raise RuntimeError(
                f"Resolved run config does not describe a latent-prior low-level policy for checkpoint {checkpoint_path}."
            )
        config_latent_dim = int(_cfg_get(latent_cfg, "latent_dim", latent_dim))
        if config_latent_dim != latent_dim:
            raise RuntimeError(
                f"Resolved run config latent_dim does not match the checkpoint network: config={config_latent_dim} checkpoint={latent_dim}"
            )
        return cls(
            checkpoint_path=checkpoint_path,
            device=device,
            obs_dim=obs_dim,
            act_dim=act_dim,
            layers=actor_layers,
            critic_layers=critic_layers,
            activation=str(model_settings["activation"]),
            initial_log_std=float(model_settings["initial_log_std"]),
            min_log_std=float(model_settings["min_log_std"]),
            max_log_std=float(model_settings["max_log_std"]),
            fixed_log_std=bool(model_settings["fixed_log_std"]),
            fixed_sigma=bool(model_settings["fixed_sigma"]),
            log_std_bound_mode=str(model_settings["log_std_bound_mode"]),
            squash_actions=bool(model_settings["squash_actions"]),
            policy_obs_indices=policy_obs_indices,
            privileged_obs_indices=privileged_obs_indices,
            latent_dim=latent_dim,
            posterior_hidden_dims=posterior_hidden_dims,
            prior_hidden_dims=prior_hidden_dims,
            latent_eval_mode=str(_cfg_get(latent_cfg, "eval_mode", "prior")),
            network_type=str(model_settings["network_type"]),
            actor_network_type=model_settings["actor_network_type"],
            critic_network_type=model_settings["critic_network_type"],
            use_layer_norm=bool(model_settings["use_layer_norm"]),
            use_rms_norm=bool(model_settings["use_rms_norm"]),
            residual_scale=float(model_settings["residual_scale"]),
            latent_min_logvar=float(
                _cfg_get(latent_cfg, "min_logvar", _DEFAULT_LATENT_MIN_LOGVAR)
            ),
            latent_max_logvar=float(
                _cfg_get(latent_cfg, "max_logvar", _DEFAULT_LATENT_MAX_LOGVAR)
            ),
            use_full_obs_critic=bool(_cfg_get(latent_cfg, "use_full_obs_critic", True)),
            separate=bool(model_settings["separate"]),
        )

    def _load_and_freeze_checkpoint(
        self, *, checkpoint_path: str | Path, policy_obs_indices: Sequence[int]
    ) -> None:
        self.obs_scaler = RunningStandardScaler(
            self.obs_main_dim, device=self.device
        ).to(self.device)
        payload = torch.load(Path(checkpoint_path), map_location=self.device)
        try:
            self.network.load_state_dict(_require_state_dict(payload, "network"))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Low-level checkpoint network state mismatch while loading {checkpoint_path}: {exc}"
            ) from exc
        self.obs_scaler.load_state_dict(_require_state_dict(payload, "obs_scaler"))
        self.network.eval()
        self.obs_scaler.eval()
        for param in self.network.parameters():
            param.requires_grad_(False)
        self.latent_dim = int(self.network.latent_dim)
        self.policy_obs_indices = [int(v) for v in policy_obs_indices]

    def _scale_obs(self, full_obs: torch.Tensor) -> torch.Tensor:
        scaled_obs = torch.empty_like(full_obs)
        self.obs_scaler(full_obs, train=False, out=scaled_obs)
        return scaled_obs

    @torch.no_grad()
    def policy_mean(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        full_obs = obs.to(device=self.device, dtype=torch.float32)
        latent = latent.to(device=self.device, dtype=torch.float32)
        scaled_obs = self._scale_obs(full_obs)
        return self.network.latent_policy_mean(scaled_obs, latent)


register_policy_mean_methods(
    LatentPriorActorCritic,
    "latent_prior",
    (
        "policy_mean_into",
        "_extract_policy_obs",
        "_extract_privileged_obs",
        "_prior_stats_from_policy_obs",
        "_posterior_stats_from_privileged_obs",
        "_actor_features_from_policy_obs_and_latent",
        "_policy_mean_from_main_obs",
        "_policy_mean_from_features",
        "_mlp_forward",
    ),
)
