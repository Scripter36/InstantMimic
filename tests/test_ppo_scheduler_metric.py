"""Adaptive learning rates must follow divergence rather than clipping frequency."""

import pytest
import torch

from instantmimic.rl.ppo.agent import PPOAgent


class _KnownDivergenceAgent(PPOAgent):
    def _compute_kl_metric(self, **kwargs):
        return super()._compute_kl_metric(**kwargs) + 4.0


def _check_scheduler(device, mode, *, detailed=False):
    graph = device == "cuda"
    agent = _KnownDivergenceAgent(
        obs_dim=4,
        act_dim=2,
        num_envs=2,
        device=device,
        rollouts=4,
        learning_epochs=2,
        mini_batches=2,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=1e-4,
        min_learning_rate=1e-7,
        max_learning_rate=1e-2,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.0,
        value_loss_scale=1.0,
        kl_threshold=0.1,
        lr_factor=1.5,
        kl_scheduler_update_mode=mode,
        layers=(8,),
        optimizer_name="adam",
        state_preprocessor=None,
        value_preprocessor=None,
        compile=graph,
        compile_act_into=False,
        compile_update_fullgraph=True,
        use_update_graph=graph,
        update_torch_stream=torch.cuda.Stream() if graph else None,
        detailed_nvtx=detailed,
    )
    # Separate the two signals deliberately. The actual update writes KL and
    # clip fraction, then its scheduler must consume the divergence signal.
    agent.buffer.obs.zero_()
    for step in range(agent.rollouts):
        agent.act_into(step)
    agent.buffer.rewards.zero_()
    agent.buffer.dones.zero_()
    metrics = agent.update()
    assert metrics["kl"] == pytest.approx(4.0, abs=1e-4)
    assert metrics["clip_fraction"] < 0.1
    reductions = agent.learning_epochs if mode == "per_epoch" else 1
    assert metrics["lr"] == pytest.approx(1e-4 / 1.5**reductions, rel=1e-5)


@pytest.mark.parametrize("mode", ["per_epoch", "per_update"])
def test_adaptive_scheduler_uses_kl_when_clip_fraction_disagrees(mode):
    _check_scheduler("cpu", mode)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["per_epoch", "per_update"])
@pytest.mark.parametrize("detailed", [False, True])
def test_graph_scheduler_uses_kl_when_clip_fraction_disagrees(mode, detailed):
    _check_scheduler("cuda", mode, detailed=detailed)
