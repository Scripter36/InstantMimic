"""Unchanged policies retain unit PPO ratios across collection/update kernels."""

from __future__ import annotations

import pytest
import torch

from instantmimic.rl.ppo.agent import PPOAgent


def _agent(
    shuffle_mode: str,
    kl_estimator: str,
    *,
    precision: str | None = None,
    fixed_sigma: bool = True,
) -> PPOAgent:
    return PPOAgent(
        precision=precision,
        fixed_sigma=fixed_sigma,
        obs_dim=5,
        act_dim=3,
        num_envs=4,
        device="cpu",
        rollouts=4,
        learning_epochs=2,
        mini_batches=2,
        shuffle_mode=shuffle_mode,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=1.0e-12,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.0,
        value_loss_scale=1.0,
        kl_threshold=0.0,
        kl_estimator=kl_estimator,
        layers=(16,),
        state_preprocessor=None,
        value_preprocessor=None,
    )


@pytest.mark.parametrize("fixed_sigma", [True, False])
@pytest.mark.parametrize("precision", [None, "float32", "bfloat16", "float16"])
@pytest.mark.parametrize("shuffle_mode", ["off", "per_epoch", "per_update"])
@pytest.mark.parametrize("kl_estimator", ["sampled", "analytic"])
def test_an_unchanged_policy_has_ratio_one_despite_stale_rollout_log_probs(
    shuffle_mode: str, kl_estimator: str, precision: str | None, fixed_sigma: bool
) -> None:
    # The rollout's log-probabilities come from the act path, whose kernels and
    # precision can differ from the update's; stale values stand in for that.
    torch.manual_seed(0)
    agent = _agent(
        shuffle_mode, kl_estimator, precision=precision, fixed_sigma=fixed_sigma
    )
    agent.buffer.obs.copy_(torch.randn_like(agent.buffer.obs))
    for step in range(agent.rollouts):
        agent.act_into(step)
    agent.buffer.rewards.copy_(torch.randn_like(agent.buffer.rewards))
    agent.buffer.dones.zero_()
    agent.buffer.log_probs.add_(torch.randn_like(agent.buffer.log_probs))
    if agent.buffer.policy_mean is not None:
        agent.buffer.policy_mean.add_(0.5)
    if agent.buffer.policy_log_std is not None:
        agent.buffer.policy_log_std.add_(0.2)

    metrics = agent.update()

    assert metrics["clip_fraction"] == 0.0
    assert abs(metrics["kl"]) < 1.0e-6
    flat_obs = agent.buffer.obs[: agent.rollouts].reshape(-1, agent.obs_dim)
    flat_actions = agent.buffer.actions.reshape(-1, agent.act_dim)
    expected_mean = torch.empty_like(flat_actions)
    expected_log_std = torch.empty_like(flat_actions)
    with torch.no_grad():
        expected, *_ = agent.network.evaluate_with_std_stats(
            flat_obs,
            flat_actions,
            out_policy_mean=expected_mean,
            out_policy_log_std=expected_log_std,
        )
    torch.testing.assert_close(
        agent.buffer.log_probs.reshape(-1, 1), expected.reshape(-1, 1)
    )
    if agent.buffer.policy_mean is not None:
        torch.testing.assert_close(
            agent.buffer.policy_mean.reshape(-1, agent.act_dim), expected_mean
        )
    if agent.buffer.policy_log_std is not None:
        torch.testing.assert_close(
            agent.buffer.policy_log_std.reshape(-1, agent.act_dim), expected_log_std
        )


@pytest.mark.parametrize("precision", [None, "float32"])
def test_the_compiled_update_recomputes_through_its_compiled_evaluation(
    precision,
) -> None:
    torch.manual_seed(1)
    agent = PPOAgent(
        precision=precision,
        obs_dim=5,
        act_dim=3,
        num_envs=4,
        device="cpu",
        rollouts=4,
        learning_epochs=1,
        mini_batches=2,
        shuffle_mode="per_epoch",
        discount_factor=0.99,
        lam=0.95,
        learning_rate=1.0e-12,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.0,
        value_loss_scale=1.0,
        kl_threshold=0.0,
        layers=(16,),
        state_preprocessor=None,
        value_preprocessor=None,
        compile=True,
        compile_act_into=False,
        compile_update_fullgraph=True,
    )
    assert agent._evaluate_old_policy_minibatch_fn is not (
        agent._evaluate_old_policy_minibatch
    )
    agent.buffer.obs.copy_(torch.randn_like(agent.buffer.obs))
    for step in range(agent.rollouts):
        agent.act_into(step)
    agent.buffer.rewards.copy_(torch.randn_like(agent.buffer.rewards))
    agent.buffer.dones.zero_()
    agent.buffer.log_probs.add_(torch.randn_like(agent.buffer.log_probs))

    metrics = agent.update()

    assert metrics["clip_fraction"] == 0.0
    assert abs(metrics["kl"]) < 1.0e-6
