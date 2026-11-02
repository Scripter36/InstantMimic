import pytest
import torch

from instantmimic.rl.ppo.agent import PPOAgent, RolloutBuffer


def _make_agent(
    *,
    num_envs: int,
    inference_only: bool = False,
    **overrides,
) -> PPOAgent:
    kwargs = dict(
        obs_dim=4,
        act_dim=2,
        num_envs=num_envs,
        device="cpu",
        rollouts=6,
        learning_epochs=1,
        mini_batches=4,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=3.0e-4,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.0,
        value_loss_scale=1.0,
        kl_threshold=0.0,
        layers=(4,),
        activation="elu",
        state_preprocessor=None,
        value_preprocessor=None,
    )
    if inference_only:
        kwargs["inference_only"] = True
    kwargs.update(overrides)
    return PPOAgent(**kwargs)


def test_rollout_buffer_rejects_non_divisible_minibatches() -> None:
    buffer = RolloutBuffer(
        rollout_steps=5,
        num_envs=2,
        obs_dim=3,
        act_dim=2,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="divisible"):
        list(buffer.get_batches(3))


def test_ppo_agent_rejects_non_divisible_minibatches_at_init() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _make_agent(num_envs=1)


def test_ppo_inference_only_loads_and_acts_without_training_state(tmp_path) -> None:
    training_agent = _make_agent(num_envs=2)
    checkpoint = tmp_path / "best_agent.pt"
    torch.save(training_agent.state_dict(), checkpoint)

    inference_agent = _make_agent(num_envs=1, inference_only=True)
    inference_agent.load(checkpoint)

    assert inference_agent.inference_only
    assert inference_agent.optimizer is None
    assert inference_agent.buffer is None
    actions, log_probs, values = inference_agent.act(torch.zeros(1, 4))
    assert actions.shape == (1, 2)
    assert log_probs.shape == (1, 1)
    assert values.shape == (1, 1)


def test_ppo_inference_only_ignores_training_only_configuration() -> None:
    agent = _make_agent(
        num_envs=1,
        inference_only=True,
        kl_estimator="unsupported",
        optimizer_name="unsupported",
        shuffle_mode="unsupported",
        value_loss_clip_mode="unsupported",
        amp_dtype="unsupported",
    )

    assert agent.inference_only
    assert agent.buffer is None
    assert agent.optimizer is None


def test_ppo_inference_only_rejects_training_operations() -> None:
    agent = _make_agent(num_envs=1, inference_only=True)

    with pytest.raises(RuntimeError, match="inference-only"):
        agent.update()
    with pytest.raises(RuntimeError, match="inference-only"):
        agent.state_dict()
