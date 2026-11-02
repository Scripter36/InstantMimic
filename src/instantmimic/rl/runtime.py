from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from instantmimic.envs.control_pipeline import ControlRuntimeSpec
from instantmimic.rl.agent_protocol import ActionScaleAgent
from instantmimic.rl.checkpoint_io import load_checkpoint_payload
from instantmimic.rl.env_access import unwrap_env


@dataclass(frozen=True)
class AgentRuntimeSpec:
    env_raw: Any
    action_mode: str
    squash_actions: bool
    clamp_mean: bool
    policy_action_dim: int
    clamp_mean_min: float = -1.0
    clamp_mean_max: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initial_state_agent_kwargs(env: Any, *, algorithm: str) -> dict[str, Any]:
    """Return PPO construction data for an environment-owned reset adapter."""

    env_raw = unwrap_env(env)
    get_spec = getattr(env_raw, "get_initial_state_spec", None)
    spec = get_spec() if callable(get_spec) else None
    if spec is None:
        return {}
    if str(algorithm).strip().lower() != "ppo":
        raise ValueError("learned initial states currently require agent.class=PPO")
    return {"initial_state_spec": spec}


def bind_initial_state_agent_to_env(
    *,
    agent: Any,
    env: Any,
    training: bool,
) -> bool:
    """Share PPO initializer tensors with the simulator without copying them."""

    env_raw = unwrap_env(env)
    get_spec = getattr(env_raw, "get_initial_state_spec", None)
    spec = get_spec() if callable(get_spec) else None
    learner = getattr(agent, "initial_state_learner", None)
    if spec is None and learner is None:
        return False
    if spec is None or learner is None:
        raise RuntimeError(
            "initial-state learner and environment adapter must be enabled together"
        )
    if learner.spec.fingerprint != spec.fingerprint:
        raise ValueError(
            "initial-state learner fingerprint does not match the environment"
        )
    bind = getattr(env_raw, "bind_initial_state_parameters", None)
    if not callable(bind):
        raise RuntimeError("environment initial-state parameter hook is unavailable")
    bind(
        mean=learner.mean,
        log_std=learner.log_std,
        version=learner.version,
        training=bool(training),
    )
    return True


def agent_observation_output_views(
    *,
    agent: Any,
    destination: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Bind env observation outputs directly to PPO rollout storage."""
    if not bool(getattr(agent, "asymmetric_observations", False)):
        return {"policy": destination}

    actor_obs_dim = int(agent.actor_obs_dim)
    critic_obs_dim = int(agent.critic_obs_dim)
    if destination.shape[-1] != actor_obs_dim + critic_obs_dim:
        raise ValueError(
            "Asymmetric PPO observation destination width must equal "
            f"actor+critic ({actor_obs_dim}+{critic_obs_dim}), got "
            f"{destination.shape[-1]}"
        )
    return {
        "policy": destination[..., :actor_obs_dim],
        "critic": destination[..., actor_obs_dim : actor_obs_dim + critic_obs_dim],
    }


def copy_agent_observation_into(
    *,
    agent: Any,
    observations: Mapping[str, torch.Tensor],
    destination: torch.Tensor,
) -> torch.Tensor:
    """Copy actor/critic env observations into fixed PPO rollout storage."""
    policy = observations.get("policy")
    if not isinstance(policy, torch.Tensor):
        raise TypeError("Environment observations must contain a policy tensor")

    output_views = agent_observation_output_views(
        agent=agent,
        destination=destination,
    )
    policy_destination = output_views["policy"]
    if policy.shape != policy_destination.shape:
        raise ValueError(
            "Policy observation shape does not match PPO storage: "
            f"{tuple(policy.shape)} != {tuple(policy_destination.shape)}"
        )
    policy_destination.copy_(policy)

    critic_destination = output_views.get("critic")
    if critic_destination is not None:
        critic = observations.get("critic")
        if not isinstance(critic, torch.Tensor):
            raise TypeError("Asymmetric PPO observations must contain a critic tensor")
        if critic.shape != critic_destination.shape:
            raise ValueError(
                "Critic observation shape does not match PPO storage: "
                f"{tuple(critic.shape)} != {tuple(critic_destination.shape)}"
            )
        critic_destination.copy_(critic)
    return destination


def _load_env_state_with(
    env: Any,
    checkpoint_path: str | Path,
    loader_name: str,
) -> bool:
    load_fn = getattr(unwrap_env(env), loader_name, None)
    if not callable(load_fn):
        return False

    payload = load_checkpoint_payload(checkpoint_path)
    if not isinstance(payload, dict):
        return False
    env_state = payload.get("env_state")
    if not isinstance(env_state, dict):
        return False
    return bool(load_fn(env_state))


def load_env_checkpoint_state(
    env: Any,
    checkpoint_path: str | Path,
) -> bool:
    return _load_env_state_with(env, checkpoint_path, "load_checkpoint_state")


def load_env_policy_init_state(
    env: Any,
    checkpoint_path: str | Path,
) -> bool:
    """Restore the environment-owned part of a policy's inference path.

    Only MASS uses it today: its MuscleNN turns the policy command into
    activations. Continuation state (sampling, reset seeds, optimizers) stays
    fresh, as policy_init keeps it on the agent.
    """
    return _load_env_state_with(env, checkpoint_path, "load_policy_init_state")


def sync_agent_action_scales_from_env(agent: Any, env: Any) -> bool:
    synced = False
    control_spec = get_env_control_runtime_spec(env)
    policy_action_scale_snapshot = control_spec.policy_action_scale_snapshot
    if policy_action_scale_snapshot is not None:
        if isinstance(agent, ActionScaleAgent):
            agent.set_action_scales(
                mean_scale=policy_action_scale_snapshot.mean_scale,
                std_scale=policy_action_scale_snapshot.std_scale,
            )
            synced = True
    return synced


def get_env_control_runtime_spec(env: Any) -> ControlRuntimeSpec:
    env_raw = unwrap_env(env)
    get_control_runtime_spec = getattr(env_raw, "get_control_runtime_spec", None)
    if not callable(get_control_runtime_spec):
        raise TypeError(
            "Environment must implement get_control_runtime_spec() returning "
            "ControlRuntimeSpec."
        )
    control_spec = get_control_runtime_spec()
    if not isinstance(control_spec, ControlRuntimeSpec):
        raise TypeError(
            "Environment get_control_runtime_spec() must return ControlRuntimeSpec, "
            f"got {type(control_spec).__name__}."
        )
    return control_spec


def resolve_agent_runtime_spec(
    env: Any,
    *,
    control_spec: ControlRuntimeSpec | None = None,
) -> AgentRuntimeSpec:
    env_raw = unwrap_env(env)
    resolved_control_spec = (
        get_env_control_runtime_spec(env_raw) if control_spec is None else control_spec
    )
    action_mode = str(resolved_control_spec.action_mode).strip().lower()

    return AgentRuntimeSpec(
        env_raw=env_raw,
        action_mode=action_mode,
        squash_actions=bool(resolved_control_spec.squash_actions),
        clamp_mean=bool(resolved_control_spec.clamp_mean),
        clamp_mean_min=float(resolved_control_spec.clamp_mean_min),
        clamp_mean_max=float(resolved_control_spec.clamp_mean_max),
        policy_action_dim=int(resolved_control_spec.policy_action_dim),
    )
