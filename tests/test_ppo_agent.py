"""CPU conformance for production PPO updates, persistent views and checkpoints."""

from __future__ import annotations

import copy

import pytest
import torch

from instantmimic.rl.ppo.agent import PPOAgent


def _agent(*, optimizer="adamw", shuffle="per_update", asymmetric=False, kl="sampled"):
    return PPOAgent(
        obs_dim=6,
        act_dim=2,
        num_envs=4,
        device="cpu",
        rollouts=4,
        learning_epochs=2,
        mini_batches=2,
        shuffle_mode=shuffle,
        discount_factor=0.95,
        lam=0.95,
        learning_rate=0.001,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.001,
        value_loss_scale=4.5,
        kl_threshold=0.0,
        kl_estimator=kl,
        layers=(12, 8),
        activation="silu",
        initial_log_std=-2.0,
        optimizer_name=optimizer,
        **(
            {"actor_obs_dim": 3, "critic_obs_dim": 3, "separate_grad_norm_clip": True}
            if asymmetric
            else {}
        ),
    )


def _collect(agent, observations):
    agent.buffer.obs[0].copy_(observations)
    for step in range(agent.rollouts):
        agent.act_into(step)
        following = (
            agent.buffer.obs[step]
            + torch.cat((agent.buffer.actions[step],) * 3, -1) * 0.01
            + 0.003
        )
        agent.buffer.obs[step + 1].copy_(following)
        agent.buffer.rewards[step].copy_(1 - following.square().sum(-1, keepdim=True))
        agent.buffer.dones[step].zero_()
        if step == 2:
            agent.buffer.dones[step, 0] = 1
    return agent.buffer.obs[-1].clone()


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
@pytest.mark.parametrize("shuffle", ["off", "per_update", "per_epoch"])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_optimizer_and_normalizer_checkpoint_repeats_the_next_controlled_update(
    tmp_path, optimizer, shuffle, asymmetric
):
    torch.manual_seed(17)
    agent = _agent(optimizer=optimizer, shuffle=shuffle, asymmetric=asymmetric)
    before = copy.deepcopy(agent.network.state_dict())
    observations = _collect(agent, torch.zeros(4, 6))
    assert all(torch.isfinite(torch.tensor(value)) for value in agent.update().values())
    assert any(
        not torch.equal(value, agent.network.state_dict()[key])
        for key, value in before.items()
    )
    checkpoint = tmp_path / "agent.pt"
    torch.save(agent.state_dict(), checkpoint)
    restored = _agent(optimizer=optimizer, shuffle=shuffle, asymmetric=asymmetric)
    assert restored.load(checkpoint)["optimizer_state_restored"]
    _equal(agent.state_dict(), restored.state_dict())
    # CPU sampling uses the caller's Torch RNG. Supply identical caller state
    # when comparing continuation of the two independently restored agents.
    rng = torch.get_rng_state()
    _collect(agent, observations)
    expected = agent.update()
    torch.set_rng_state(rng)
    _collect(restored, observations)
    assert restored.update() == expected
    _equal(agent.state_dict(), restored.state_dict())


@pytest.mark.parametrize("shuffle", ["off", "per_update", "per_epoch"])
@pytest.mark.parametrize("kl", ["sampled", "analytic"])
def test_detailed_update_operations_match_the_fused_update_on_cpu(shuffle, kl):
    """Exercise the same prepared operations without CUDA capture or replay."""
    torch.manual_seed(31)
    fused = _agent(shuffle=shuffle, kl=kl)
    _collect(fused, torch.zeros(4, 6))
    detailed = copy.deepcopy(fused)
    # The detailed path expects persistent graph workspaces. Give its operations
    # CPU storage; the normalizer, GAE, evaluator and Adam still execute normally.
    detailed._use_update_graph = True
    detailed._update_scaled_obs = torch.empty(16, 6)
    rng = torch.get_rng_state()
    expected = fused.update()
    torch.set_rng_state(rng)
    detailed._snapshot_analytic_kl_log_std()
    detailed._refresh_update_rewards()
    detailed._refresh_epoch_permutations()
    detailed._run_detailed_update_prepare()
    for epoch in range(detailed.learning_epochs):
        detailed._run_detailed_update_epoch_prepare(epoch)
        for batch in range(detailed.mini_batches):
            detailed._run_detailed_update_minibatch(epoch, batch)
    detailed._write_update_summary_from_metrics(detailed._update_metrics)
    assert detailed._read_update_summary() == expected
    _equal(fused.state_dict(), detailed.state_dict())
    torch.testing.assert_close(
        fused.buffer.returns, detailed.buffer.returns, rtol=0, atol=0
    )
    torch.testing.assert_close(
        fused.buffer.advantages, detailed.buffer.advantages, rtol=0, atol=0
    )


def test_kl_schedule_constants_are_preallocated_and_stable() -> None:
    agent = _agent()
    agent.kl_threshold = 0.02
    agent.lr_factor = 1.5
    agent.min_learning_rate = 1.0e-6
    agent.max_learning_rate = 1.0e-2
    for group in agent.optimizer.param_groups:
        group["lr"] = torch.tensor(1.0e-4, dtype=torch.float32)
    agent._refresh_optimizer_lr_wrappers()

    constants_by_dtype = agent._lr_schedule_constants
    constants = constants_by_dtype[torch.float32]
    tensor_ids = {name: id(tensor) for name, tensor in constants.items()}
    attributes = set(agent.__dict__)

    agent._update_learning_rate(torch.tensor(0.5))
    agent._update_learning_rate(torch.tensor(1.0e-4))

    # Nothing the update path touches may be created or replaced: a new
    # attribute, dict entry or tensor would invalidate Dynamo's guards and
    # force a recompile inside the captured update.
    assert agent._lr_schedule_constants is constants_by_dtype
    assert set(agent.__dict__) == attributes
    assert set(constants_by_dtype) == {torch.float32}
    assert {name: id(tensor) for name, tensor in constants.items()} == tensor_ids
    assert float(constants["kl_high"]) == pytest.approx(0.04)
    assert float(constants["kl_low"]) == pytest.approx(0.01)
    assert float(constants["factor"]) == pytest.approx(1.5)
    assert float(constants["min_lr"]) == pytest.approx(1.0e-6)
    assert float(constants["max_lr"]) == pytest.approx(1.0e-2)

    # A configuration change refreshes values in place, keeping identity.
    agent.kl_threshold = 0.01
    agent._refresh_lr_schedule_constants()
    assert float(constants["kl_high"]) == pytest.approx(0.02)
    assert float(constants["kl_low"]) == pytest.approx(0.005)
    assert {name: id(tensor) for name, tensor in constants.items()} == tensor_ids


def test_lr_schedule_constants_follow_tensor_lr_normalization() -> None:
    agent = _agent()
    # The capture path normalizes tensor LRs to float32 while it builds the
    # Warp handles; reproduce that configuration on CPU.
    agent._use_update_graph = True
    agent.kl_threshold = 0.02
    agent.lr_factor = 1.5
    agent.min_learning_rate = 1.0e-6
    agent.max_learning_rate = 1.0e-2
    for group in agent.optimizer.param_groups:
        group["lr"] = torch.tensor(1.0e-4, dtype=torch.float64)

    agent._refresh_optimizer_lr_wrappers()

    lr_dtypes = {group["lr"].dtype for group in agent.optimizer.param_groups}
    assert lr_dtypes == {torch.float32}
    # The constants are allocated from the normalized LRs. Refreshing them
    # before normalization left a float64 entry and scheduling raised
    # KeyError(torch.float32).
    assert set(agent._lr_schedule_constants) == {torch.float32}
    agent._update_learning_rate(torch.tensor(0.5))
    for group in agent.optimizer.param_groups:
        assert isinstance(group["lr"], torch.Tensor)
        assert group["lr"].dtype == torch.float32
