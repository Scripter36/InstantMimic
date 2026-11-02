"""Policy-mean activation regularization for activation-output baselines."""

import pytest
import torch
from config_fixtures import valid_config

from instantmimic.config.loader import load_semantic_config
from instantmimic.config.validation import ConfigSchemaError
from instantmimic.rl.models.actor_critic import ActorCritic


def test_aux_loss_is_weighted_squared_policy_mean_around_target():
    torch.manual_seed(0)
    network = ActorCritic(
        3,
        4,
        layers=(8,),
        activation_regularization_weight=0.1,
        activation_regularization_target=0.2,
    )
    obs = torch.randn(5, 3)
    actions = torch.randn(5, 4)
    mean = torch.empty(5, 4)
    log_std = torch.empty(5, 4)

    _, _, _, aux_loss, *_ = network.evaluate_with_aux_losses_and_std_stats(
        obs, actions, out_policy_mean=mean, out_policy_log_std=log_std
    )

    torch.testing.assert_close(aux_loss, 0.1 * (mean - 0.2).square().mean())
    aux_loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in network.parameters()
    )


def test_aux_loss_is_zero_without_regularization():
    network = ActorCritic(3, 4, layers=(8,))
    _, _, _, aux_loss = network.evaluate_with_aux_losses(
        torch.randn(5, 3), torch.randn(5, 4)
    )
    assert aux_loss.item() == 0.0


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"weight": -0.1, "target": 0.0}, r"activation_regularization\.weight"),
        ({"weight": 0.1}, r"activation_regularization"),
    ],
)
def test_config_rejects_invalid_activation_regularization(block, message):
    cfg = valid_config()
    cfg["agent"]["activation_regularization"] = block
    with pytest.raises(ConfigSchemaError, match=message):
        load_semantic_config(cfg)


def test_config_accepts_activation_regularization():
    cfg = valid_config()
    cfg["agent"]["activation_regularization"] = {"weight": 0.1, "target": 0.0}
    load_semantic_config(cfg)
