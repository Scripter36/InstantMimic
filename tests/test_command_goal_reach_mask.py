"""Playback can read goal reaches without a benchmark having configured metrics."""

from types import SimpleNamespace

import pytest
import torch

from instantmimic.envs.command.command_runtime import CommandRuntime
from instantmimic.envs.command.latent_command_runtime import LatentCommandRuntime


class _Env(SimpleNamespace):
    def invalidate_step_graph_cache(self):
        self.invalidations += 1


@pytest.mark.parametrize("runtime_cls", [CommandRuntime, LatentCommandRuntime])
def test_goal_mask_enables_metrics_once_and_drops_captured_steps(runtime_cls):
    env = _Env(num_envs=3, _target_speed=torch.zeros(3), invalidations=0)
    runtime = runtime_cls(env)

    mask = runtime.get_goal_reached_mask()

    assert mask.shape == (3,) and mask.dtype == torch.bool and not mask.any()
    assert env.invalidations == 1
    assert runtime.get_goal_reached_mask() is mask
    assert env.invalidations == 1
