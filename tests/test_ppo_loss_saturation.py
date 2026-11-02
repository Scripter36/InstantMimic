"""The constant part of PPO's clipped objective must have finite zero gradient."""

import pytest
import torch

from instantmimic.rl.ppo.losses import ppo_loss_terms


def policy_loss(log_ratio, advantage):
    zero = torch.zeros_like(log_ratio)
    return ppo_loss_terms(
        new_log_probs=log_ratio,
        old_log_probs=zero,
        predicted_values=zero,
        old_values=zero,
        returns=zero,
        advantages=torch.full_like(log_ratio, advantage),
        entropy=zero,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        value_loss_scale=1.0,
        entropy_loss_scale=0.0,
    )[0]


@pytest.mark.parametrize("log_ratio", [90.0, 100.0, 1000.0])
@pytest.mark.parametrize("advantage", [0.0, 1.0, 10.0])
def test_positive_advantage_upper_plateau_has_finite_zero_gradient(
    log_ratio, advantage
):
    value = torch.tensor([[log_ratio]], requires_grad=True)
    loss = policy_loss(value, advantage)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(-1.2 * advantage)
    assert torch.isfinite(value.grad).all()
    torch.testing.assert_close(value.grad, torch.zeros_like(value))


def test_negative_advantage_lower_plateau_has_finite_zero_gradient():
    value = torch.tensor([[-1000.0]], requires_grad=True)
    loss = policy_loss(value, -2.0)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(1.6)
    torch.testing.assert_close(value.grad, torch.zeros_like(value))


@pytest.mark.parametrize("clip", [0.0, 0.2, 1.0, 2.0])
def test_loss_and_gradient_match_the_original_objective_when_ratios_are_finite(clip):
    x = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64).repeat_interleave(5)
    x = x.reshape(-1, 1).requires_grad_(True)
    advantages = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0], dtype=torch.float64)
    advantages = advantages.repeat(101).reshape(-1, 1)
    zero = torch.zeros_like(x)
    actual = ppo_loss_terms(
        new_log_probs=x,
        old_log_probs=zero,
        predicted_values=zero,
        old_values=zero,
        returns=zero,
        advantages=advantages,
        entropy=zero,
        ratio_clip=clip,
        value_clip=0.2,
        clip_predicted_values=True,
        value_loss_scale=1.0,
        entropy_loss_scale=0.0,
    )[0]
    ratio = x.exp()
    expected = -torch.minimum(
        advantages * ratio,
        advantages * ratio.clamp(1.0 - clip, 1.0 + clip),
    ).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, retain_graph=True)[0],
        torch.autograd.grad(expected, x)[0],
    )


@pytest.mark.cuda
def test_compiled_positive_advantage_plateau_has_finite_zero_gradient():
    value = torch.full((256, 1), 100.0, device="cuda", requires_grad=True)
    compiled = torch.compile(lambda x: policy_loss(x, 1.0), fullgraph=True)
    loss = compiled(value)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(-1.2)
    torch.testing.assert_close(value.grad, torch.zeros_like(value))
