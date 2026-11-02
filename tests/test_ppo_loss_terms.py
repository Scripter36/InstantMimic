import torch

from instantmimic.rl.ppo.losses import ppo_loss_terms


def test_value_clipping_preserves_fast_ppo_clamp_then_mse_semantics() -> None:
    zero_scalar = torch.tensor(0.0)

    policy, value, entropy, clip_fraction = ppo_loss_terms(
        new_log_probs=torch.tensor([[0.0], [0.0]]),
        old_log_probs=torch.tensor([[0.0], [0.0]]),
        predicted_values=torch.tensor([[1.0], [-1.0]]),
        old_values=torch.zeros(2, 1),
        returns=torch.zeros(2, 1),
        advantages=torch.tensor([[1.0], [-1.0]]),
        entropy=torch.ones(2, 1),
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        value_loss_scale=2.0,
        entropy_loss_scale=0.0,
        zero_scalar=zero_scalar,
    )

    torch.testing.assert_close(policy, torch.tensor(0.0))
    torch.testing.assert_close(value, torch.tensor(0.08))
    assert entropy is zero_scalar
    torch.testing.assert_close(clip_fraction, torch.tensor(0.0))


def test_value_clipping_max_mode_keeps_unclipped_critic_gradient() -> None:
    predicted_values = torch.tensor([[2.0]], requires_grad=True)

    _, value, _, _ = ppo_loss_terms(
        new_log_probs=torch.zeros(1, 1),
        old_log_probs=torch.zeros(1, 1),
        predicted_values=predicted_values,
        old_values=torch.zeros(1, 1),
        returns=torch.ones(1, 1),
        advantages=torch.zeros(1, 1),
        entropy=torch.zeros(1, 1),
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        value_loss_scale=1.0,
        entropy_loss_scale=0.0,
        value_loss_clip_mode="max",
    )
    value.backward()

    torch.testing.assert_close(value, torch.tensor(1.0))
    torch.testing.assert_close(predicted_values.grad, torch.tensor([[2.0]]))
