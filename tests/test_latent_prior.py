from __future__ import annotations

import pytest
import torch

from instantmimic.rl.ppo.latent_prior import LatentPriorActorCritic, LatentPriorPPOAgent


def network_kwargs():
    return dict(
        policy_obs_indices=[0, 1, 2, 3],
        privileged_obs_indices=[4, 5],
        latent_dim=2,
        posterior_hidden_dims=[8],
        prior_hidden_dims=[8],
        latent_kl_weight=0.1,
        latent_posterior_sample_alpha=1.0,
        latent_ar1_weight=0.05,
        latent_eval_mode="prior",
    )


def network():
    return LatentPriorActorCritic(obs_dim=6, act_dim=2, layers=[8], **network_kwargs())


def test_conditional_rollout_and_evaluation_replay_identical_posterior_noise():
    torch.manual_seed(21)
    n = network()
    obs = torch.randn(4, 6)
    noise = torch.randn(4, 2)
    actions = torch.empty(4, 2)
    log_probs = torch.empty(4, 1)
    values = torch.empty(4, 1)
    n.act_into(
        obs, actions, log_probs, values, eps=torch.zeros_like(actions), policy_aux=noise
    )
    evaluated = n.evaluate_with_aux_losses(obs, actions, policy_aux=noise)
    torch.testing.assert_close(log_probs, evaluated[0])
    torch.testing.assert_close(values, evaluated[1])
    loss = evaluated[0].mean() + evaluated[1].mean() + evaluated[3]
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in n.parameters())
    assert n.prior_mu_head.weight.grad is not None
    assert n.posterior_mu_head.weight.grad is not None


def test_privileged_reference_information_cannot_change_prior_or_decoder():
    n = network()
    obs = torch.randn(4, 6)
    changed = obs.clone()
    changed[:, 4:] += 100
    latent = torch.randn(4, 2)
    torch.testing.assert_close(n.prior_mean(obs), n.prior_mean(changed))
    torch.testing.assert_close(
        n.latent_policy_mean(obs, latent), n.latent_policy_mean(changed, latent)
    )


def agent(shuffle):
    return LatentPriorPPOAgent(
        obs_dim=6,
        act_dim=2,
        num_envs=4,
        device="cpu",
        rollouts=4,
        learning_epochs=2,
        mini_batches=2,
        shuffle_mode=shuffle,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=0.001,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.001,
        value_loss_scale=1.0,
        kl_threshold=0.0,
        layers=[8],
        optimizer_name="adamw",
        network_cls=LatentPriorActorCritic,
        network_kwargs=network_kwargs(),
    )


def collect(a):
    a.buffer.obs[0].copy_(torch.randn(4, 6))
    for step in range(a.rollouts):
        a.act_into(step)
        a.buffer.obs[step + 1].copy_(a.buffer.obs[step] + 0.01)
        a.buffer.rewards[step].fill_(1.0)
        a.buffer.dones[step].zero_()
    a.buffer.dones[1, 0] = 1


@pytest.mark.parametrize("shuffle", ["off", "per_update", "per_epoch"])
def test_latent_update_and_checkpoint_resume_keep_conditional_state(tmp_path, shuffle):
    torch.manual_seed(8)
    a = agent(shuffle)
    collect(a)
    assert a.buffer.policy_aux.shape == (4, 4, 2)
    metrics = a.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert metrics["latent_prior_kl"] >= 0
    assert metrics["latent_prior_ar1"] >= 0
    path = tmp_path / "agent.pt"
    torch.save(a.state_dict(), path)
    b = agent(shuffle)
    b.load(path)
    for key, value in a.network.state_dict().items():
        torch.testing.assert_close(value, b.network.state_dict()[key])
    collect(b)
    b.update()


def test_done_transition_has_no_ar1_gradient():
    n = network()
    obs = torch.randn(4, 6, requires_grad=True)
    following = torch.randn(4, 6, requires_grad=True)
    loss = n.transition_aux_regularization_loss(obs, following, torch.ones(4, 1))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(obs.grad) == 0
    assert torch.count_nonzero(following.grad) == 0


def artifact_config(squash):
    return dict(
        schema_version=1,
        seed=1,
        task="Synthetic-v0",
        env={"action": {"type": "ctrl", "scale": 1.0, "use_reference": False}},
        models={
            "shared": True,
            "actor": {
                "architecture": "mlp",
                "layers": [8],
                "activation": "elu",
                "normalization": "none",
                "distribution": {
                    "std": "parameter",
                    "initial_log_std": 0.0,
                    "log_std_bounds": [-20.0, 2.0],
                    "bound_mode": "clamp",
                    "action_transform": "tanh" if squash else "none",
                },
            },
        },
        agent={
            "class": "PPO",
            "rollouts": 4,
            "learning_epochs": 1,
            "mini_batches": 2,
            "discount_factor": 0.99,
            "gae_lambda": 0.95,
            "optimizer": {"type": "adamw"},
            "learning_rate": {"initial": 0.001},
            "latent_prior": {
                "enabled": True,
                "latent_dim": 2,
                "posterior_hidden_dims": [8],
                "prior_hidden_dims": [8],
                "use_full_obs_critic": True,
                "eval_mode": "prior",
            },
        },
        trainer={"max_iterations": 1},
        experiment={"directory": "synthetic", "logger": {"type": "tensorboard"}},
        runtime={},
        play={"num_envs": 1, "steps": 1, "action": {"mode": "mean"}},
    )


@pytest.mark.parametrize("squash", [False, True])
def test_frozen_checkpoint_preserves_action_transform_and_observation_partition(
    tmp_path, squash
):
    from omegaconf import OmegaConf

    from instantmimic.config.artifacts import build_checkpoint_metadata
    from instantmimic.config.loader import load_semantic_config
    from instantmimic.rl.ppo.agent import RunningStandardScaler
    from instantmimic.rl.ppo.latent_prior import FrozenLatentPolicyAdapter

    n = LatentPriorActorCritic(
        obs_dim=6, act_dim=2, layers=[8], squash_actions=squash, **network_kwargs()
    )
    with torch.no_grad():
        n.policy_head.bias.fill_(4.0)
    scaler = RunningStandardScaler(6, device="cpu")
    config = load_semantic_config(artifact_config(squash))
    bundle = tmp_path / "run_bundle/config/train_config_resolved.yaml"
    bundle.parent.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(config.to_dict()), bundle)
    path = tmp_path / "checkpoints/agent.pt"
    path.parent.mkdir()
    torch.save(
        {
            "network": n.state_dict(),
            "obs_scaler": scaler.state_dict(),
            "_instantmimic": build_checkpoint_metadata(),
        },
        path,
    )
    adapter = FrozenLatentPolicyAdapter.from_checkpoint(
        checkpoint_path=path,
        device="cpu",
        obs_dim=6,
        act_dim=2,
        runtime_obs_spec={
            "policy_obs_indices": [0, 1, 2, 3],
            "privileged_obs_indices": [4, 5],
        },
    )
    obs = torch.randn(4, 6)
    latent = torch.randn(4, 2)
    scaled = torch.empty_like(obs)
    scaler(obs, train=False, out=scaled)
    torch.testing.assert_close(
        adapter.policy_mean(obs, latent), n.latent_policy_mean(scaled, latent)
    )
    torch.testing.assert_close(adapter.prior_mean(obs), n.prior_mean(scaled))
    assert all(not p.requires_grad for p in adapter.network.parameters())
    with pytest.raises(RuntimeError, match="policy_obs_indices"):
        FrozenLatentPolicyAdapter.from_checkpoint(
            checkpoint_path=path,
            device="cpu",
            obs_dim=6,
            act_dim=2,
            runtime_obs_spec={
                "policy_obs_indices": [0, 1],
                "privileged_obs_indices": [2, 3, 4, 5],
            },
        )


def test_existing_initial_state_contract_can_construct_and_store_samples():
    from instantmimic.rl.initial_state import InitialStateSpec
    from instantmimic.rl.ppo.agent import PPOAgent

    specification = InitialStateSpec(
        num_components=1,
        latent_dim=2,
        coordinates="synthetic",
        residual_scale=(1.0, 1.0),
        fingerprint="synthetic",
    )
    a = PPOAgent(
        obs_dim=6,
        act_dim=2,
        num_envs=4,
        device="cpu",
        rollouts=4,
        learning_epochs=1,
        mini_batches=2,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=0.001,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.001,
        value_loss_scale=1.0,
        kl_threshold=0.0,
        layers=[8],
        optimizer_name="adamw",
        initial_state_spec=specification,
    )
    assert a.initial_state_learner is not None
    a.store_initial_state_samples(
        step=0,
        mask=torch.ones(4, dtype=torch.bool),
        samples=torch.zeros(4, 2),
        component_ids=torch.zeros(4, dtype=torch.long),
        old_log_prob=torch.zeros(4),
        versions=torch.zeros(4, dtype=torch.long),
    )


@pytest.mark.parametrize("variant", ["architecture", "normalizer"])
def test_frozen_checkpoint_rejects_unsupported_reconstruction_contract(variant):
    from instantmimic.config.loader import load_semantic_config
    from instantmimic.rl.ppo.latent_prior import _artifact_ppo_model_settings

    values = artifact_config(False)
    if variant == "architecture":
        values["models"]["actor"]["architecture"] = "residual_mlp"
        values["models"]["actor"]["residual_scale"] = 1.0
        message = "flat MLP"
    else:
        values["agent"]["state_preprocessor"] = "none"
        message = "RunningStandardScaler"
    with pytest.raises(RuntimeError, match=message):
        _artifact_ppo_model_settings(load_semantic_config(values))
