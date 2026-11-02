from __future__ import annotations

import gymnasium as gym
import pytest
import torch
import warp as wp
from freemusco_runtime_fixtures import build_freemusco_run_cfg

import instantmimic.envs.freemusco.tasks  # noqa: F401
from instantmimic.envs.freemusco.tasks.freemusco_run.runtime import (
    FreemuscoRunRuntime,
)


@pytest.mark.parametrize(
    ("action_type", "species", "uses_fast_muscle"),
    [
        ("mujoco_activation", "chimanoid", False),
        ("mujoco_activation", "ostrich", False),
        ("torque", "chimanoid", True),
        ("torque", "ostrich", True),
    ],
    ids=[
        "chimanoid-mujoco",
        "ostrich-mujoco",
        "chimanoid-torque",
        "ostrich-torque",
    ],
)
def test_run_runtime_executes_shared_cpu_path(
    action_type: str,
    species: str,
    uses_fast_muscle: bool,
) -> None:
    cfg = build_freemusco_run_cfg(
        species=species,
        include_muscle_observation_blocks=False,
        include_heading_frame_body_state_obs=True,
        action_type=action_type,
    )
    if action_type == "mujoco_activation":
        cfg.action = {"type": action_type, "space": 120}

    env = gym.make("FreeMusco-Run-v0", cfg=cfg).unwrapped
    try:
        assert env.cfg.species == species
        assert type(env._run_runtime) is FreemuscoRunRuntime
        assert env._uses_fast_muscle is uses_fast_muscle

        observations, _ = env.reset()
        action = torch.zeros_like(env.actions)
        env._pre_physics_step(action)
        env._apply_action()
        observations = env._get_observations()
        rewards = env._get_rewards()
        terminated, time_out = env._get_dones()

        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(rewards).all()
        assert terminated.dtype == torch.bool
        assert time_out.dtype == torch.bool

        steps_before = env.common_step_counter.clone()
        observations, rewards, terminated, time_out, _ = env.step(action)
        assert env.common_step_counter == steps_before + 1
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(rewards).all()
        assert terminated.shape == time_out.shape == (env.num_envs,)
    finally:
        env.close()


def test_activation_actions_are_clamped_and_drive_muscle_forces() -> None:
    cfg = build_freemusco_run_cfg(
        species="ostrich",
        include_muscle_observation_blocks=False,
        include_heading_frame_body_state_obs=True,
    )
    cfg.action = {"type": "activation", "space": 120}

    env = gym.make("FreeMusco-Run-v0", cfg=cfg).unwrapped
    try:
        assert env._uses_fast_muscle
        assert env.actions.shape[-1] == env.fast_muscle.muscle_count
        env.reset()
        qfrc_applied = wp.to_torch(env.mjw_data.qfrc_applied)

        action = torch.full_like(env.actions, -0.5)
        action[..., ::2] = 2.0
        env._pre_physics_step(action)
        expected = torch.zeros_like(action)
        expected[..., ::2] = 1.0
        torch.testing.assert_close(env.actions, expected)
        env._apply_action()
        driven = qfrc_applied.clone()

        env._pre_physics_step(torch.zeros_like(action))
        env._apply_action()
        passive_only = qfrc_applied.clone()

        assert torch.isfinite(driven).all()
        assert torch.isfinite(passive_only).all()
        assert not torch.allclose(driven, passive_only)
    finally:
        env.close()
