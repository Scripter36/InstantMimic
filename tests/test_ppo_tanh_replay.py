import pytest
import torch

from instantmimic.rl.models.actor_critic import ActorCritic
from instantmimic.rl.ppo.agent import PPOAgent
from instantmimic.rl.ppo.latent_prior import LatentPriorActorCritic, LatentPriorPPOAgent


def _make_agent(*, num_envs=2, latent=False, **overrides):
    options = dict(
        obs_dim=4,
        act_dim=2,
        num_envs=num_envs,
        device="cpu",
        rollouts=6,
        learning_epochs=1,
        mini_batches=4,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=3e-4,
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
    )
    if latent:
        options.update(
            network_cls=LatentPriorActorCritic,
            network_kwargs=dict(
                policy_obs_indices=[0, 1],
                privileged_obs_indices=[2, 3],
                latent_dim=2,
                posterior_hidden_dims=[4],
                prior_hidden_dims=[4],
                latent_posterior_sample_alpha=0.7,
                latent_kl_weight=0.1,
            ),
        )
    options.update(overrides)
    return (LatentPriorPPOAgent if latent else PPOAgent)(**options)


@pytest.mark.parametrize("mean", [0.2, 8.0, 10.0, -10.0])
@pytest.mark.parametrize("latent", [False, True])
def test_tanh_replay_preserves_sampled_density(mean, latent):
    torch.manual_seed(7)
    kwargs = dict(obs_dim=4, act_dim=2, layers=(4,), squash_actions=True)
    if latent:
        network = LatentPriorActorCritic(
            **kwargs,
            policy_obs_indices=[0, 1],
            privileged_obs_indices=[2, 3],
            latent_dim=2,
            posterior_hidden_dims=[4],
            prior_hidden_dims=[4],
            latent_posterior_sample_alpha=0.7,
            latent_kl_weight=0.0,
        )
    else:
        network = ActorCritic(**kwargs)
    with torch.no_grad():
        network.policy_head.weight.fill_(0.03)
        network.policy_head.bias.fill_(mean)
    obs = torch.randn(6, 4)
    actions = torch.empty(6, 2)
    raw = torch.empty_like(actions)
    collected = torch.empty(6, 1)
    values = torch.empty_like(collected)
    eps = torch.linspace(-0.4, 0.4, 12).reshape(6, 2)
    aux = torch.randn(6, 2) if latent else None
    with torch.no_grad():
        network.act_into(
            obs,
            actions,
            collected,
            values,
            eps,
            policy_aux=aux,
            out_pre_squash_actions=raw,
        )
    order = torch.tensor([5, 0, 3, 1, 4, 2])
    replay = network.evaluate_with_std_stats(
        obs[order],
        actions[order],
        pre_squash_actions=raw[order],
        policy_aux=aux[order] if latent else None,
    )[0]
    torch.testing.assert_close(
        (replay - collected[order]).exp(),
        torch.ones_like(replay),
        rtol=2e-5,
        atol=2e-5,
    )
    assert torch.isfinite(replay).all()
    assert torch.equal(actions, raw.tanh())
    plain_replay = network.evaluate(
        obs[order],
        actions[order],
        pre_squash_actions=raw[order],
        policy_aux=aux[order] if latent else None,
    )[0]
    torch.testing.assert_close(plain_replay, replay)
    if latent:
        changed_aux = network.evaluate(
            obs[order],
            actions[order],
            pre_squash_actions=raw[order],
            policy_aux=aux[order] + 1.0,
        )[0]
        assert not torch.allclose(changed_aux, replay)
        without_stats = network.evaluate_with_aux_losses(
            obs[order],
            actions[order],
            pre_squash_actions=raw[order],
            policy_aux=aux[order],
        )[0]
        torch.testing.assert_close(without_stats, replay)
        replay_aux = network.evaluate_with_aux_losses_and_std_stats(
            obs[order],
            actions[order],
            pre_squash_actions=raw[order],
            policy_aux=aux[order],
        )[0]
        torch.testing.assert_close(replay_aux, replay)


@pytest.mark.parametrize("shuffle_mode", ["none", "per_epoch", "per_update"])
@pytest.mark.parametrize("squash", [False, True])
def test_agent_tanh_replay_uses_action_batch_order(shuffle_mode, squash):
    agent = _make_agent(num_envs=2, squash_actions=squash, shuffle_mode=shuffle_mode)
    if not squash:
        assert agent.buffer.pre_squash_actions is None
        return
    with torch.no_grad():
        agent.network.policy_head.weight.zero_()
        agent.network.policy_head.bias.fill_(10.0)
        for step in range(agent.rollouts):
            agent.buffer.obs[step].normal_()
            agent.act_into(step)
    raw = agent.buffer.pre_squash_actions
    flat_actions, flat_log_probs, *_ = agent._flat_update_batch_views()
    assert flat_actions.data_ptr() == raw.data_ptr()
    order = torch.tensor([11, 0, 8, 3, 6, 4, 2, 9, 7, 10, 1, 5], dtype=torch.int32)
    # The existing action shuffle transports the raw samples with all other rows.
    if agent.shuffle_minibatches:
        epoch = agent._prepare_epoch_batch_views(
            agent.buffer.obs[:-1].reshape(-1, 4),
            *agent._flat_update_batch_views(),
            order,
        )
        batch_actions = epoch[1]
    else:
        order = torch.arange(12)
        batch_actions = flat_actions
    obs = agent.buffer.obs[:-1].reshape(-1, 4)[order.long()]
    replay = agent.network.evaluate(
        obs, batch_actions.tanh(), pre_squash_actions=batch_actions
    )[0]
    torch.testing.assert_close(
        (replay - flat_log_probs[order.long()]).exp(),
        torch.ones_like(replay),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("shuffle_mode", ["none", "per_epoch", "per_update"])
@pytest.mark.parametrize("return_path", [False, True])
@pytest.mark.parametrize("latent", [False, True])
def test_saturated_policy_update_preserves_rollout_likelihood(
    shuffle_mode, return_path, latent
):
    agent = _make_agent(
        num_envs=2,
        latent=latent,
        squash_actions=True,
        shuffle_mode=shuffle_mode,
        precision="float32",
        learning_rate=1e-12,
    )
    agent._compiled_update_uses_return_path = return_path
    with torch.no_grad():
        agent.network.policy_head.weight.fill_(0.03)
        agent.network.policy_head.bias.fill_(10.0)
        agent.buffer.obs.normal_()
        for step in range(agent.rollouts):
            agent.act_into(step)
        collected = agent.buffer.log_probs.clone()
        posterior_noise = agent.buffer.policy_aux.clone() if latent else None
        agent.buffer.rewards.normal_()
    metrics = agent.update()
    assert metrics["clip_fraction"] == 0.0
    assert abs(metrics["kl"]) < 1e-6
    torch.testing.assert_close(agent.buffer.log_probs, collected, rtol=0, atol=0)
    if latent:
        torch.testing.assert_close(
            agent.buffer.policy_aux, posterior_noise, rtol=0, atol=0
        )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "shuffle_mode,latent,detailed,fullgraph_update",
    [
        ("none", False, False, False),
        ("per_epoch", True, True, True),
        ("per_update", True, False, False),
        ("per_epoch", False, True, True),
    ],
)
def test_tanh_rollout_and_update_cuda_graph_replay_keeps_raw_storage(
    shuffle_mode, latent, detailed, fullgraph_update
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    agent = _make_agent(
        device="cuda",
        latent=latent,
        detailed_nvtx=detailed,
        squash_actions=True,
        shuffle_mode=shuffle_mode,
        precision="float32",
        learning_rate=1e-12,
        compile=True,
        use_update_graph=True,
    )
    with torch.no_grad():
        agent.network.policy_head.weight.zero_()
        agent.network.policy_head.bias.fill_(10.0)
    raw_pointer = agent.buffer.pre_squash_actions.data_ptr()
    # Warm the exact collector specialization before external rollout capture.
    for step in range(agent.rollouts):
        agent.act_into(step)
    torch.cuda.synchronize()
    rollout_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(rollout_graph):
        for step in range(agent.rollouts):
            agent.act_into(step)
    for _ in range(2):
        agent.buffer.obs.normal_()
        rollout_graph.replay()
        agent.buffer.rewards.normal_()
        collected = agent.buffer.log_probs.clone()
        metrics = agent.update()
        assert agent._has_captured_update_graphs()
        assert agent.buffer.pre_squash_actions.data_ptr() == raw_pointer
        assert metrics["clip_fraction"] == 0.0
        assert abs(metrics["kl"]) < 1e-5
        torch.testing.assert_close(agent.buffer.log_probs, collected, rtol=0, atol=0)


@pytest.mark.parametrize("squash", [False, True])
@pytest.mark.parametrize("fullgraph_update", [False, True])
def test_tanh_compile_warmup_precedes_rollout_buffer_allocation(
    squash, fullgraph_update
):
    agent = _make_agent(
        squash_actions=squash,
        precision="float32",
        learning_rate=1e-12,
        compile=True,
    )
    with torch.no_grad():
        agent.network.policy_head.weight.zero_()
        agent.network.policy_head.bias.fill_(10.0)
        agent.buffer.obs.normal_()
        for step in range(agent.rollouts):
            agent.act_into(step)
        agent.buffer.rewards.normal_()
        collected = agent.buffer.log_probs.clone()
    metrics = agent.update()
    assert metrics["clip_fraction"] == 0.0
    assert abs(metrics["kl"]) < 1e-6
    torch.testing.assert_close(agent.buffer.log_probs, collected, rtol=0, atol=0)


def test_raw_tanh_density_matches_gaussian_change_of_variables():
    from instantmimic.rl.models.distributions import gaussian_tanh_log_prob

    raw = torch.tensor([[-10.0], [-8.0], [0.3], [8.0], [10.0]])
    mean = torch.full_like(raw, 0.2)
    log_std = torch.full_like(raw, -0.4)
    actual = gaussian_tanh_log_prob(
        raw.tanh(),
        mean,
        log_std,
        log_std.exp(),
        squash=True,
        epsilon=1e-6,
        pre_squash_actions=raw,
    )
    # Float64 keeps the Jacobian representable, giving an independent oracle
    # where float32 actions have rounded to exactly +/-1.
    reference = torch.distributions.Normal(
        mean.double(), log_std.double().exp()
    ).log_prob(raw.double())
    reference -= (1.0 - raw.double().tanh().square()).log()
    torch.testing.assert_close(actual.double(), reference, rtol=1e-6, atol=1e-5)
    moved = gaussian_tanh_log_prob(
        raw.tanh(),
        mean + 0.4,
        log_std,
        log_std.exp(),
        squash=True,
        epsilon=1e-6,
        pre_squash_actions=raw,
    )
    reference_delta = torch.distributions.Normal(
        (mean + 0.4).double(), log_std.double().exp()
    ).log_prob(raw.double()) - torch.distributions.Normal(
        mean.double(), log_std.double().exp()
    ).log_prob(raw.double())
    torch.testing.assert_close(
        (moved - actual).double(), reference_delta, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("squash", [False, True])
@pytest.mark.parametrize("latent", [False, True])
@pytest.mark.parametrize("analytic_kl", [False, True])
def test_warmup_compiles_rollout_and_bootstrap_before_capture(
    squash, latent, analytic_kl
):
    torch._dynamo.reset()
    agent = _make_agent(
        latent=latent,
        squash_actions=squash,
        precision="float32",
        kl_estimator="analytic" if analytic_kl else "sampled",
    )
    compilations = []

    def counting_backend(graph, example_inputs):
        del example_inputs
        compilations.append(graph)
        return graph.forward

    agent._act_into_fn = torch.compile(
        agent.network.act_into,
        backend=counting_backend,
        fullgraph=True,
    )
    agent._warmup_compiled_functions(warmup_act_into=True)
    warmed_count = len(compilations)
    assert warmed_count > 0
    agent.bootstrap_values_into(agent.buffer.obs[0], torch.empty(agent.num_envs, 1))
    assert len(compilations) == warmed_count
    agent.act_into(0)
    assert len(compilations) == warmed_count
