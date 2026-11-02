from types import SimpleNamespace

import pytest
import torch

from instantmimic.envs.command.latent_command_runtime import LatentCommandRuntime
from instantmimic.envs.snu.tasks.snu_command.checkpoint_runtime import (
    infer_reset_motion_sources_from_low_level_cfg,
)


@pytest.mark.parametrize("direction_only", [False, True])
def test_velocity_command_observations_express_world_target_in_root_frame(
    direction_only,
):
    # Identical stationary body poses face opposite sides of the world target.
    root_pos = torch.zeros(2, 3)
    root_quat = torch.tensor([[2**-0.5, 0, 0, 2**-0.5], [2**-0.5, 0, 0, -(2**-0.5)]])
    command_dim = 2 if direction_only else 4
    low_level = SimpleNamespace(
        deployable_observation_indices=torch.tensor([0]),
        command_dim=command_dim,
        command_observation_buffer=torch.zeros(2, command_dim),
        observation_buffer=torch.zeros(2, 1),
        append_prior_to_policy_observation=False,
    )
    env = SimpleNamespace(
        latent_low_level_state=low_level,
        _obs_buf=torch.zeros(2, 1 + command_dim),
        _command_mode="velocity_heading",
        _target_heading=torch.zeros(2, 1),
        _target_speed=torch.ones(2, 1),
        _velocity_heading_direction_only_enabled=lambda: direction_only,
        _uses_fast_muscle_backend=lambda: False,
        refresh_low_level_obs=lambda: None,
        _root_pos_quat=lambda: (root_pos, root_quat),
        _planar_velocity=lambda: torch.tensor([[0.25, 0.0], [0.25, 0.0]]),
    )
    obs = LatentCommandRuntime(env).get_observations()["policy"]
    heading_start = 1 if direction_only else 2
    torch.testing.assert_close(
        obs[:, heading_start : heading_start + 2],
        torch.tensor([[0.0, -1.0], [0.0, 1.0]]),
        atol=1e-6,
        rtol=0,
    )
    if not direction_only:
        torch.testing.assert_close(obs[:, 1], torch.ones(2))
        # Speed error still uses the world target and world linear velocity.
        torch.testing.assert_close(obs[:, 4], torch.full((2,), 0.75))
    torch.testing.assert_close(env._target_heading, torch.zeros(2, 1))


@pytest.mark.parametrize("prefix", ["", "datasets/", "/tmp/external_data/"])
def test_amass_reset_source_preserves_original_dataset_prefix(prefix):
    root = prefix + "data/humanoid/motions/amass/train_set"
    actual = infer_reset_motion_sources_from_low_level_cfg(
        {"motion": {"source": [root + "/**/*.npz"]}},
        {"low_level_reset_motion_subsets": ["ACCAD/*Walking_c3d/*.npz"]},
    )
    assert actual == [root + "/ACCAD/*Walking_c3d/*.npz"]


def test_point_goal_observations_keep_local_displacement_distance_and_speed():
    root_pos = torch.zeros(1, 3)
    root_quat = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]])
    env = SimpleNamespace(
        latent_low_level_state=SimpleNamespace(
            deployable_observation_indices=torch.tensor([0]),
            command_dim=4,
            command_observation_buffer=torch.zeros(1, 4),
            observation_buffer=torch.zeros(1, 1),
            append_prior_to_policy_observation=False,
        ),
        num_envs=1,
        device=torch.device("cpu"),
        _obs_buf=torch.zeros(1, 5),
        _command_mode="point_goal",
        _goal_pos_xy=torch.tensor([[2.0, 0.0]]),
        _target_speed=torch.tensor([[1.5]]),
        _uses_fast_muscle_backend=lambda: False,
        refresh_low_level_obs=lambda: None,
        _root_pos_quat=lambda: (root_pos, root_quat),
    )
    obs = LatentCommandRuntime(env).get_observations()["policy"]
    torch.testing.assert_close(
        obs, torch.tensor([[0.0, 0.0, -2.0, 2.0, 1.5]]), atol=1e-6, rtol=0
    )
