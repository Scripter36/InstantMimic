"""Shared playback agent setup for play, export, and analysis tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from instantmimic.config.schema import SemanticConfig
from instantmimic.rl.checkpointing import load_full_agent_checkpoint
from instantmimic.rl.factory import (
    build_agent_from_semantic_config,
    resolve_agent_class_spec,
    resolve_agent_observation_dims,
)
from instantmimic.rl.runtime import (
    AgentRuntimeSpec,
    bind_initial_state_agent_to_env,
    get_env_control_runtime_spec,
    initial_state_agent_kwargs,
    load_env_checkpoint_state,
    resolve_agent_runtime_spec,
    sync_agent_action_scales_from_env,
)


@dataclass(frozen=True)
class PlaybackCheckpointState:
    checkpoint_path: Path | None
    env_state_restored: bool
    action_scales_synced: bool


@dataclass(frozen=True)
class PlaybackAgentRuntime:
    agent: Any
    runtime_spec: AgentRuntimeSpec
    env_raw: Any
    device: torch.device
    num_envs: int
    env_action_dim: int
    agent_action_dim: int
    checkpoint_state: PlaybackCheckpointState

    @property
    def env_state_restored(self) -> bool:
        return self.checkpoint_state.env_state_restored

    @property
    def action_scales_synced(self) -> bool:
        return self.checkpoint_state.action_scales_synced


def load_playback_checkpoint(
    *,
    agent: Any,
    env: Any,
    checkpoint_path: str | Path | None,
    eval_mode: bool = True,
    restore_env_state: bool = True,
    sync_action_scales: bool = True,
) -> PlaybackCheckpointState:
    if checkpoint_path is None:
        return PlaybackCheckpointState(
            checkpoint_path=None,
            env_state_restored=False,
            action_scales_synced=False,
        )

    resolved_path = load_full_agent_checkpoint(
        agent, Path(checkpoint_path).expanduser().resolve()
    )
    if eval_mode:
        agent.network.eval()

    env_state_restored = False
    action_scales_synced = False
    if restore_env_state:
        env_state_restored = load_env_checkpoint_state(env, resolved_path)
    if sync_action_scales:
        # The environment owns the physical action-scale contract. Agent
        # checkpoints also contain policy scale buffers, so synchronize even
        # when the checkpoint has no mutable environment payload.
        action_scales_synced = sync_agent_action_scales_from_env(agent, env)

    return PlaybackCheckpointState(
        checkpoint_path=resolved_path,
        env_state_restored=bool(env_state_restored),
        action_scales_synced=bool(action_scales_synced),
    )


def build_playback_agent_runtime(
    *,
    cfg: SemanticConfig,
    env: Any,
    checkpoint_path: str | Path | None,
    environment_cfg: Any | None = None,
    device: torch.device | None = None,
    eval_mode: bool = True,
    restore_env_state: bool = True,
    sync_action_scales: bool = True,
) -> PlaybackAgentRuntime:
    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    scale_control_spec = get_env_control_runtime_spec(env)
    runtime_spec = resolve_agent_runtime_spec(env, control_spec=scale_control_spec)
    env_raw = runtime_spec.env_raw
    resolved_env_cfg = environment_cfg if environment_cfg is not None else cfg.env
    num_envs = int(resolved_env_cfg.scene.num_envs)
    env_action_dim = int(resolved_env_cfg.action_space)
    agent_action_dim = int(runtime_spec.policy_action_dim)

    agent_spec = resolve_agent_class_spec(cfg)
    obs_dim, asymmetric_kwargs = resolve_agent_observation_dims(
        env_raw,
        fallback_observation_space=resolved_env_cfg.observation_space,
    )
    agent_extra_kwargs: dict[str, Any] = (
        {"inference_only": True} if agent_spec.algorithm == "ppo" else {}
    )
    agent_extra_kwargs.update(asymmetric_kwargs)
    agent_extra_kwargs.update(
        initial_state_agent_kwargs(env_raw, algorithm=agent_spec.algorithm)
    )
    agent = build_agent_from_semantic_config(
        cfg,
        agent_spec,
        obs_dim=obs_dim,
        act_dim=agent_action_dim,
        num_envs=num_envs,
        device=resolved_device,
        runtime_spec=runtime_spec,
        policy_action_scale_snapshot=scale_control_spec.policy_action_scale_snapshot,
        extra_kwargs=agent_extra_kwargs or None,
    )
    bind_initial_state_agent_to_env(
        agent=agent,
        env=env_raw,
        training=False,
    )
    checkpoint_state = load_playback_checkpoint(
        agent=agent,
        env=env,
        checkpoint_path=checkpoint_path,
        eval_mode=eval_mode,
        restore_env_state=restore_env_state,
        sync_action_scales=sync_action_scales,
    )
    return PlaybackAgentRuntime(
        agent=agent,
        runtime_spec=runtime_spec,
        env_raw=env_raw,
        device=resolved_device,
        num_envs=num_envs,
        env_action_dim=env_action_dim,
        agent_action_dim=agent_action_dim,
        checkpoint_state=checkpoint_state,
    )
