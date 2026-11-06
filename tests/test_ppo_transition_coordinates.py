"""Transition regularizers must compare one observation coordinate system."""

import pytest
import torch

from instantmimic.rl.ppo.latent_prior import LatentPriorActorCritic, LatentPriorPPOAgent


def _agent(device: str, shuffle_mode: str, *, detailed: bool = False):
    torch.manual_seed(17)
    graph = device == "cuda"
    return LatentPriorPPOAgent(
        obs_dim=4,
        act_dim=1,
        num_envs=2,
        device=device,
        rollouts=4,
        learning_epochs=2,
        mini_batches=2,
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
        layers=(8,),
        optimizer_name="adam",
        state_preprocessor="RunningStandardScaler",
        value_preprocessor=None,
        shuffle_mode=shuffle_mode,
        network_cls=LatentPriorActorCritic,
        network_kwargs={
            "policy_obs_indices": [2, 3],
            "privileged_obs_indices": [0, 1],
            "latent_dim": 2,
            "posterior_hidden_dims": [8],
            "prior_hidden_dims": [8],
            "latent_kl_weight": 0.001,
            "latent_ar1_weight": 0.25,
            "latent_ar1_phi": 1.0,
        },
        compile=graph,
        compile_mode="max-autotune-no-cudagraphs",
        use_update_graph=graph,
        use_shuffle_graph=graph,
        update_torch_stream=torch.cuda.Stream() if graph else None,
        detailed_nvtx=detailed,
    )


def _check_stationary_transitions(agent):
    for _ in range(2):
        agent.buffer.obs.fill_(2.0)
        for step in range(agent.rollouts):
            agent.act_into(step)
        agent.buffer.rewards.fill_(1.0)
        agent.buffer.dones.zero_()
        before = float(agent.obs_scaler.current_count)
        metrics = agent.update()
        # At phi=1, identical consecutive inputs have zero AR1 residual even
        # as optimizer steps change the posterior and the scaler advances.
        assert metrics["latent_prior_ar1_weighted"] == pytest.approx(0.0, abs=1e-10)
        assert float(agent.obs_scaler.current_count) == before + 8


@pytest.mark.parametrize("shuffle_mode", ["off", "per_update", "per_epoch"])
def test_stationary_transition_has_no_normalization_induced_ar1(shuffle_mode):
    _check_stationary_transitions(_agent("cpu", shuffle_mode))


@pytest.mark.cuda
@pytest.mark.parametrize("detailed", [False, True])
def test_compiled_graph_preserves_stationary_transition_coordinates(detailed):
    agent = _agent("cuda", "per_epoch", detailed=detailed)
    _check_stationary_transitions(agent)
    assert agent._update_graph is not None or agent._has_detailed_update_graphs()
