from __future__ import annotations

import torch

from instantmimic.envs.command.latent_low_level_state import LatentLowLevelState
from instantmimic.envs.config_access import get_block, get_value
from instantmimic.envs.snu.tasks.snu_imitation.snu_imitation_env import SNUImitationEnv
from instantmimic.rl.ppo.latent_prior import FrozenLatentPolicyAdapter


def initialize_low_level_policy(env) -> None:
    action = get_block(env.cfg, "action")
    if str(get_value(action, "type")) != "ctrl":
        raise ValueError("Frozen humanoid command control requires action.type=ctrl")
    if bool(get_value(action, "use_reference", False)):
        raise ValueError(
            "Frozen downstream control requires action.use_reference=false"
        )
    scale = float(get_value(action, "scale", 1.0))
    if scale <= 0:
        raise ValueError("Frozen command control requires a positive action scale")
    spec = env.get_latent_prior_obs_spec()
    indices = torch.tensor(
        spec["policy_obs_indices"], device=env.device, dtype=torch.long
    )
    obs_dim = int(spec["obs_main_dim"])
    act_dim = int(env.model.nu)
    env.cfg.low_level_action_space = act_dim
    adapter = FrozenLatentPolicyAdapter.from_checkpoint(
        checkpoint_path=env.cfg.low_level_checkpoint,
        device=env.device,
        obs_dim=obs_dim,
        act_dim=act_dim,
        runtime_obs_spec=spec,
    )
    latent_dim = adapter.latent_dim
    append_prior = bool(env.cfg.low_level_append_prior_to_obs)
    policy_dim = (
        len(spec["policy_obs_indices"]) + 4 + (latent_dim if append_prior else 0)
    )

    def zeros(width):
        return torch.zeros(
            (env.num_envs, width), device=env.device, dtype=torch.float32
        )

    low_obs = zeros(obs_dim)
    state = LatentLowLevelState(
        action_scale=scale,
        observation_spec=spec,
        deployable_observation_indices=indices,
        append_prior_to_policy_observation=append_prior,
        residual_scale=float(env.cfg.low_level_residual_scale),
        observation_buffer=low_obs,
        environment_action_buffer=zeros(act_dim),
        adapter=adapter,
        latent_action_buffer=zeros(latent_dim),
        previous_latent_action_buffer=zeros(latent_dim),
        prior_latent_buffer=zeros(latent_dim),
        total_latent_buffer=zeros(latent_dim),
        command_dim=4,
        policy_observation_dim=policy_dim,
        command_observation_buffer=zeros(4),
        observation_mapping={"policy": low_obs},
    )
    env.latent_low_level_state = state
    env.cfg.action_space = latent_dim
    env.obs_buf = {"policy": zeros(policy_dim)}
    env._obs_buf = env.obs_buf["policy"]
    env.finalize_derived_spaces(
        observation_spec=policy_dim, action_spec=latent_dim, observations=env.obs_buf
    )
    env.actions = state.environment_action_buffer


def refresh_low_level_obs(env) -> None:
    with env.bind_observation_outputs(env.latent_low_level_state.observation_mapping):
        SNUImitationEnv._get_observations(env)


def capture_previous_latent_action(env) -> None:
    state = env.latent_low_level_state
    state.previous_latent_action_buffer.copy_(state.latent_action_buffer)


def apply_latent_action(env, actions: torch.Tensor) -> None:
    state = env.latent_low_level_state
    state.latent_action_buffer.copy_(actions)
    refresh_low_level_obs(env)
    state.prior_latent_buffer.copy_(state.adapter.prior_mean(state.observation_buffer))
    state.total_latent_buffer.copy_(state.prior_latent_buffer)
    state.total_latent_buffer.add_(state.latent_action_buffer * state.residual_scale)
    state.environment_action_buffer.copy_(
        state.adapter.policy_mean(state.observation_buffer, state.total_latent_buffer)
    )
    state.environment_action_buffer.clamp_(
        -1.0 / state.action_scale, 1.0 / state.action_scale
    )
    state.environment_action_buffer.mul_(state.action_scale)
