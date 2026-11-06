"""A valid one-transition rollout must keep learning tensors finite."""

import pytest
import torch

from instantmimic.rl.ppo.agent import PPOAgent, RolloutBuffer


@pytest.mark.parametrize("normalize", [True, False])
def test_single_transition_returns_are_finite(normalize):
    buffer = RolloutBuffer(1, 1, 3, 1, torch.device("cpu"))
    buffer.values.zero_()
    buffer.rewards.fill_(1.0)
    buffer.dones.fill_(1.0)
    buffer.compute_returns(
        torch.zeros(1, 1), 0.99, 0.95, normalize_advantages=normalize
    )
    torch.testing.assert_close(buffer.returns, torch.ones_like(buffer.returns))
    expected = (
        torch.zeros_like(buffer.advantages)
        if normalize
        else torch.ones_like(buffer.advantages)
    )
    torch.testing.assert_close(buffer.advantages, expected)


def test_multiple_transitions_keep_sample_standard_deviation():
    buffer = RolloutBuffer(1, 3, 3, 1, torch.device("cpu"))
    rewards = torch.tensor([1.0, 2.0, 4.0]).view(1, 3, 1)
    buffer.values.zero_()
    buffer.rewards.copy_(rewards)
    buffer.dones.fill_(1.0)
    buffer.compute_returns(torch.zeros(3, 1), 0.99, 0.95)
    expected = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    torch.testing.assert_close(buffer.advantages, expected)


@pytest.mark.parametrize("compile_update", [False, True])
def test_single_sample_ppo_update_keeps_parameters_finite(compile_update):
    agent = PPOAgent(
        obs_dim=3,
        act_dim=1,
        num_envs=1,
        device="cpu",
        rollouts=1,
        learning_epochs=1,
        mini_batches=1,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=1e-4,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.0,
        value_loss_scale=1.0,
        kl_threshold=0.0,
        layers=(4,),
        state_preprocessor=None,
        value_preprocessor=None,
        compile=compile_update,
    )
    agent.buffer.obs.zero_()
    agent.act_into(0)
    agent.buffer.rewards.fill_(1.0)
    agent.buffer.dones.fill_(1.0)
    metrics = agent.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert all(torch.isfinite(p).all() for p in agent.network.parameters())
    torch.testing.assert_close(
        agent.buffer.advantages, torch.zeros_like(agent.buffer.advantages)
    )
    torch.testing.assert_close(
        agent.buffer.returns, torch.ones_like(agent.buffer.returns)
    )
