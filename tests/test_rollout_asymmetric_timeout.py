"""Timeout values use terminal privileged observations before automatic reset."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from instantmimic.rl.ppo.agent import PPOAgent
from instantmimic.train.rollout import RolloutRuntime


@pytest.mark.parametrize("asymmetric", [False, True])
def test_timeout_bootstrap_uses_complete_terminal_observation(asymmetric):
    agent = PPOAgent(
        obs_dim=6,
        act_dim=2,
        num_envs=4,
        device="cpu",
        rollouts=4,
        learning_epochs=1,
        mini_batches=1,
        discount_factor=0.95,
        lam=0.95,
        learning_rate=0.001,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.001,
        value_loss_scale=1.0,
        kl_threshold=0.0,
        layers=(12, 8),
        activation="silu",
        initial_log_std=-2.0,
        optimizer_name="adam",
        state_preprocessor="RslRlEmpiricalNormalization",
        **(
            {"actor_obs_dim": 3, "critic_obs_dim": 3, "separate_grad_norm_clip": True}
            if asymmetric
            else {}
        ),
    )
    terminal = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10
    groups = (
        {"policy": terminal[:, :3], "critic": terminal[:, 3:]}
        if asymmetric
        else {"policy": terminal}
    )
    timeouts = torch.tensor([True, False, True, False])
    runtime = RolloutRuntime.__new__(RolloutRuntime)
    runtime.agent = agent
    runtime.buf = agent.buffer
    runtime.time_limit_bootstrap = True
    runtime.env_raw = SimpleNamespace(
        terminal_obs_buf=groups,
        reset_time_outs=timeouts,
        prepare_terminal_observation_buffer=lambda: None,
    )
    runtime._step_terminated = torch.zeros((4, 4), dtype=torch.bool)
    runtime._nvtx_range = lambda name: nullcontext()
    runtime._prepare_terminal_observation_buffer()
    expected = torch.empty(4, 1)
    agent.bootstrap_values_into(terminal, expected)
    expected[~timeouts] = 0
    runtime._maybe_apply_time_limit_bootstrap(0, fallback_time_outs=None)
    torch.testing.assert_close(agent._timeout_bootstrap_values[0], expected)
    # The next policy state has already been reset and must not be overwritten.
    agent.buffer.obs.fill_(123.0)
    terminal.add_(1.0)
    agent.bootstrap_values_into(terminal, expected)
    expected[~timeouts] = 0
    runtime._maybe_apply_time_limit_bootstrap(1, fallback_time_outs=None)
    torch.testing.assert_close(agent._timeout_bootstrap_values[1], expected)
    assert torch.all(agent.buffer.obs == 123.0)
