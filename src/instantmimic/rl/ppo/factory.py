from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import torch

from instantmimic.envs.control_pipeline import ActionScaleSnapshot
from instantmimic.rl.build_specs import PPOAgentBuildSpec
from instantmimic.rl.runtime import AgentRuntimeSpec

from .agent import PPOAgent
from .latent_prior import resolve_latent_prior_network_build_spec


def _constructor_keys(agent_type: type[Any] | None = None) -> frozenset[str]:
    classes = (
        (PPOAgent,)
        if agent_type is None
        else tuple(cls for cls in agent_type.__mro__ if cls is not object)
    )
    return frozenset(
        name
        for cls in classes
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
        and parameter.kind
        not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    )


def _merge_agent_kwargs_checked(
    target: dict[str, Any],
    values: Mapping[str, Any],
    *,
    agent_type: type[Any],
    owner: str,
) -> None:
    unknown = sorted(set(values) - _constructor_keys(agent_type))
    if unknown:
        raise ValueError(f"unconsumed {owner} kwargs: {unknown}")
    for key, value in values.items():
        if key in target and target[key] != value:
            raise ValueError(f"{owner} kwarg {key!r} conflicts with semantic value")
        target[key] = value


def build_ppo_agent_from_spec(
    spec: PPOAgentBuildSpec,
    *,
    obs_dim: int,
    act_dim: int,
    num_envs: int,
    device: torch.device | str,
    agent_cls: type[Any] | None = None,
    runtime_spec: AgentRuntimeSpec | None = None,
    policy_action_scale_snapshot: ActionScaleSnapshot | None = None,
    update_torch_stream: torch.cuda.Stream | None = None,
    network_cls: type[torch.nn.Module] | None = None,
    network_kwargs: Mapping[str, Any] | None = None,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Build PPO from an already validated schema-v1 construction plan."""
    from instantmimic.rl.models.musubi_actor_critic import reject_unavailable_network

    reject_unavailable_network(network_cls)
    if not isinstance(spec, PPOAgentBuildSpec):
        raise TypeError("spec must be a PPOAgentBuildSpec")
    kwargs: dict[str, Any] = dict(spec.constructor_kwargs)
    kwargs["initial_state_config"] = spec.initial_state
    unknown_semantic = sorted(set(kwargs) - _constructor_keys())
    if unknown_semantic:
        raise ValueError(f"unconsumed PPO semantic kwargs: {unknown_semantic}")
    kwargs.update(
        {
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "num_envs": int(num_envs),
            "device": device,
            "compile": spec.runtime.compile,
            "compile_mode": spec.runtime.compile_mode,
            "compile_act_into": spec.runtime.compile_act_into,
            "compile_update_fullgraph": spec.runtime.compile_update_fullgraph,
            "use_update_graph": spec.runtime.update_cuda_graph,
            "use_shuffle_graph": spec.runtime.shuffle_cuda_graph,
            "update_torch_stream": update_torch_stream,
            "mixed_precision": spec.runtime.get("mixed_precision"),
            "precision": spec.runtime.ppo_precision,
            "loss_scale": spec.runtime.loss_scale,
            "detailed_nvtx": spec.runtime.nsys_nvtx_mode == "detailed",
            "fail_on_nonfinite_update": spec.runtime.fail_on_nonfinite_update,
        }
    )

    resolved_network_cls = network_cls
    resolved_network_kwargs = dict(kwargs.pop("network_kwargs", {}))
    if network_kwargs:
        for key, value in network_kwargs.items():
            if key in resolved_network_kwargs and resolved_network_kwargs[key] != value:
                raise ValueError(
                    f"execution network kwarg {key!r} conflicts with semantic value"
                )
            resolved_network_kwargs[key] = value
    resolved_network_cls, latent_kwargs = resolve_latent_prior_network_build_spec(
        runtime_spec=runtime_spec,
        network_cls=resolved_network_cls,
        network_kwargs=resolved_network_kwargs,
        latent_prior_cfg=spec.latent_prior,
        actor_layers=spec.actor.constructor_kwargs["layers"],
        critic_layers=(
            spec.critic.constructor_kwargs["layers"]
            if spec.critic
            else spec.actor.constructor_kwargs["layers"]
        ),
        models_separate=not spec.shared,
    )
    resolved_network_kwargs = dict(latent_kwargs or {})
    if spec.runtime.policy_mlp_bf16:
        resolved_network_kwargs["policy_mlp_bf16"] = True
    if runtime_spec is not None:
        if bool(runtime_spec.squash_actions) != bool(kwargs["squash_actions"]):
            raise ValueError(
                "runtime squash_actions conflicts with semantic action_transform"
            )
        kwargs.update(
            {
                "clamp_mean": bool(runtime_spec.clamp_mean),
                "clamp_mean_min": float(runtime_spec.clamp_mean_min),
                "clamp_mean_max": float(runtime_spec.clamp_mean_max),
            }
        )
    if resolved_network_cls is not None:
        kwargs["network_cls"] = resolved_network_cls
    if resolved_network_kwargs:
        kwargs["network_kwargs"] = resolved_network_kwargs
    agent_type = PPOAgent if agent_cls is None else agent_cls
    if extra_kwargs:
        _merge_agent_kwargs_checked(
            kwargs,
            extra_kwargs,
            agent_type=agent_type,
            owner="PPO execution",
        )
    agent = agent_type(**kwargs)
    if policy_action_scale_snapshot is not None:
        agent.set_action_scales(
            mean_scale=policy_action_scale_snapshot.mean_scale,
            std_scale=policy_action_scale_snapshot.std_scale,
        )
    return agent
