from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from instantmimic.envs.control_pipeline import ActionScaleSnapshot
from instantmimic.rl.build_specs import (
    AgentBuildSpec,
    PPOAgentBuildSpec,
    lower_agent_build_spec,
)

AgentAlgorithm = str


@dataclass(frozen=True)
class AgentClassSpec:
    algorithm: AgentAlgorithm
    agent_cls: type[Any]
    extra_kwargs: dict[str, Any]


def resolve_agent_algorithm(cfg: Any) -> AgentAlgorithm:
    agent_cfg = getattr(cfg, "agent", None)
    if agent_cfg is None or "class" not in agent_cfg:
        raise ValueError("agent.class is required")
    if str(agent_cfg["class"]).strip().lower() != "ppo":
        raise ValueError("Unsupported agent.class; supported value: PPO")
    return "ppo"


def resolve_agent_class_spec(cfg: Any) -> AgentClassSpec:
    from instantmimic.rl.models.musubi_actor_critic import reject_unavailable_selectors

    reject_unavailable_selectors(cfg)
    algorithm = resolve_agent_algorithm(cfg)
    from instantmimic.rl.ppo.agent import PPOAgent
    from instantmimic.rl.ppo.latent_prior import (
        LatentPriorPPOAgent,
        is_latent_prior_enabled,
    )

    return AgentClassSpec(
        algorithm=algorithm,
        agent_cls=LatentPriorPPOAgent if is_latent_prior_enabled(cfg) else PPOAgent,
        extra_kwargs={},
    )


def build_agent_from_spec(
    build_spec: AgentBuildSpec,
    class_spec: AgentClassSpec,
    *,
    obs_dim: int,
    act_dim: int,
    num_envs: int,
    device: Any,
    runtime_spec: Any | None = None,
    policy_action_scale_snapshot: ActionScaleSnapshot | None = None,
    update_torch_stream: Any | None = None,
    network_cls: type[Any] | None = None,
    network_kwargs: dict[str, Any] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build one validated production PPO plan with its ordinary policy."""
    if not isinstance(build_spec, PPOAgentBuildSpec) or class_spec.algorithm != "ppo":
        raise ValueError("PPO build spec does not match resolved agent class")
    from instantmimic.rl.ppo.factory import build_ppo_agent_from_spec

    merged = dict(class_spec.extra_kwargs)
    if extra_kwargs:
        merged.update(extra_kwargs)
    return build_ppo_agent_from_spec(
        build_spec,
        agent_cls=class_spec.agent_cls,
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_envs=num_envs,
        device=device,
        runtime_spec=runtime_spec,
        policy_action_scale_snapshot=policy_action_scale_snapshot,
        update_torch_stream=update_torch_stream,
        network_cls=network_cls,
        network_kwargs=network_kwargs,
        extra_kwargs=merged,
    )


def resolve_agent_observation_dims(
    env_raw: Any,
    *,
    fallback_observation_space: int,
) -> tuple[int, dict[str, Any]]:
    """Total observation width for the agent, plus the asymmetric kwargs.

    An environment that separates actor and critic observations reports the two
    widths; the agent takes their sum and needs to know the split. Training and
    playback both had to ask, and both had to remember that
    `requires_separate_ppo_grad_norm_clip` rides along with the split.

    Returns an empty mapping for a symmetric environment, so a caller can merge
    it into whatever else it passes.
    """

    get_asymmetric_observation_dims = getattr(
        env_raw,
        "get_asymmetric_observation_dims",
        None,
    )
    if not callable(get_asymmetric_observation_dims):
        return int(fallback_observation_space), {}
    actor_obs_dim, critic_obs_dim = get_asymmetric_observation_dims()
    return int(actor_obs_dim) + int(critic_obs_dim), {
        "actor_obs_dim": int(actor_obs_dim),
        "critic_obs_dim": int(critic_obs_dim),
        "separate_grad_norm_clip": bool(
            getattr(env_raw, "requires_separate_ppo_grad_norm_clip", False)
        ),
    }


def build_agent_from_semantic_config(
    semantic_config: Any,
    class_spec: AgentClassSpec,
    *,
    obs_dim: int,
    act_dim: int,
    num_envs: int,
    device: Any,
    runtime_spec: Any | None = None,
    policy_action_scale_snapshot: ActionScaleSnapshot | None = None,
    update_torch_stream: Any | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Lower one semantic config exactly once, then build its typed plan."""
    return build_agent_from_spec(
        lower_agent_build_spec(semantic_config),
        class_spec,
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_envs=num_envs,
        device=device,
        runtime_spec=runtime_spec,
        policy_action_scale_snapshot=policy_action_scale_snapshot,
        update_torch_stream=update_torch_stream,
        extra_kwargs=extra_kwargs,
    )
