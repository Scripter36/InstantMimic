"""Observable numerical contracts of the production Gaussian policy."""

from __future__ import annotations

import copy

import pytest
import torch

from instantmimic.rl.models.actor_critic import ActorCritic


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize(
    "separate,fixed_sigma", [(False, True), (True, True), (True, False)]
)
@pytest.mark.parametrize("transform", ["none", "tanh", "clamp"])
def test_collection_evaluation_and_distribution_outputs_agree(
    bf16, separate, fixed_sigma, transform
):
    torch.manual_seed(11)
    model = ActorCritic(
        13,
        4,
        layers=(16, 12),
        activation="silu",
        initial_log_std=-1.3,
        separate=separate,
        fixed_sigma=fixed_sigma,
        policy_mlp_bf16=bf16,
        squash_actions=transform == "tanh",
        clamp_mean=transform == "clamp",
    )
    model.set_action_mean_scale(torch.linspace(0.8, 1.2, 4))
    model.set_action_std_scale(torch.linspace(0.5, 1.1, 4))
    obs, noise = torch.randn(7, 13), torch.randn(7, 4)
    actions, sampled_logprob, sampled_value = (
        torch.empty(7, 4),
        torch.empty(7, 1),
        torch.empty(7, 1),
    )
    mean, logstd, stats = torch.empty(7, 4), torch.empty(7, 4), torch.empty(5)
    with torch.no_grad():
        model.act_into(obs, actions, sampled_logprob, sampled_value, noise)
    logprob, value, entropy, std_mean, std_min, std_max = model.evaluate_with_std_stats(
        obs,
        actions,
        out_policy_mean=mean,
        out_policy_log_std=logstd,
        out_raw_log_std_bound_stats=stats,
    )
    torch.testing.assert_close(logprob, sampled_logprob, atol=5e-5, rtol=0)
    torch.testing.assert_close(value, sampled_value, atol=0, rtol=0)
    torch.testing.assert_close(
        entropy,
        (logstd + 0.5 * torch.log(torch.tensor(2 * torch.pi * torch.e))).sum(
            -1, keepdim=True
        ),
    )
    wanted_mean, wanted_logstd = model.policy_distribution_params(obs)
    torch.testing.assert_close(mean, wanted_mean, rtol=0, atol=0)
    torch.testing.assert_close(logstd, wanted_logstd, rtol=0, atol=0)
    std = logstd.exp()
    for actual, expected in zip(
        (std_mean, std_min, std_max), (std.mean(), std.amin(), std.amax()), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (-logprob.mean() + value.square().mean() - 0.01 * entropy.mean()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_std_bound_loss_updates_std_head_without_backpropagating_into_the_trunk():
    model = ActorCritic(
        6,
        2,
        layers=(8,),
        fixed_sigma=False,
        min_log_std=-3,
        max_log_std=1,
        initial_log_std=-1,
        log_std_bound_loss_weight=0.7,
        log_std_bound_loss_margin=0.2,
    )
    assert model.log_std_head is not None
    with torch.no_grad():
        model.log_std_head.bias.fill_(3.5)
    stats = torch.empty(5)
    model.evaluate_with_std_stats(
        torch.randn(3, 6), torch.randn(3, 2), out_raw_log_std_bound_stats=stats
    )
    assert stats[4] > 0
    stats[4].backward()
    assert model.log_std_head.bias.grad is not None
    assert model.log_std_head.bias.grad.norm() > 0
    assert all(p.grad is None for p in model.trunk.parameters())


@pytest.mark.parametrize("bf16", [False, True])
def test_fullgraph_evaluation_has_identical_outputs_and_gradients(bf16):
    torch.manual_seed(61)
    eager = ActorCritic(
        6, 2, layers=(8,), separate=True, fixed_sigma=False, policy_mlp_bf16=bf16
    )
    compiled = copy.deepcopy(eager)
    evaluate = torch.compile(compiled.evaluate, backend="aot_eager", fullgraph=True)
    obs, actions = torch.randn(3, 6), torch.randn(3, 2)
    original = eager.evaluate(obs, actions)
    actual = evaluate(obs, actions)
    for a, b in zip(original, actual, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for result in (original, actual):
        (
            -result[0].mean() + result[1].square().mean() - 0.01 * result[2].mean()
        ).backward()
    for a, b in zip(eager.parameters(), compiled.parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_checkpoint_restoration_preserves_the_next_optimizer_step():
    torch.manual_seed(31)
    first = ActorCritic(6, 2, layers=(8,), separate=True, fixed_sigma=False)
    optimizer = torch.optim.AdamW(first.parameters(), lr=3e-4)
    obs, actions = torch.randn(3, 6), torch.randn(3, 2)

    def update(model, opt):
        logprob, value, entropy = model.evaluate(obs, actions)
        opt.zero_grad(set_to_none=True)
        (-logprob.mean() + value.square().mean() - 0.01 * entropy.mean()).backward()
        opt.step()

    update(first, optimizer)
    state = copy.deepcopy(first.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    second = ActorCritic(6, 2, layers=(8,), separate=True, fixed_sigma=False)
    second.load_state_dict(state, strict=True)
    other_optimizer = torch.optim.AdamW(second.parameters(), lr=9e-3)
    other_optimizer.load_state_dict(optimizer_state)
    update(first, optimizer)
    update(second, other_optimizer)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
