"""Repeated clip evaluation must execute the newly loaded checkpoint policy."""

from types import SimpleNamespace

import pytest
import torch

from instantmimic.rl.ppo.agent import PPOAgent
from instantmimic.train.benchmark_runner import _BenchmarkPolicyRunner
from instantmimic.train.clip_failure_benchmark import ClipFailureBenchmarkRunner


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_reused_analysis_runtime_refreshes_the_frozen_policy_after_load(
    tmp_path, device
):
    agent = PPOAgent(
        obs_dim=5,
        act_dim=2,
        num_envs=2,
        device=device,
        rollouts=2,
        learning_epochs=1,
        mini_batches=1,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=0.001,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.0,
        value_loss_scale=1.0,
        kl_threshold=0.0,
        layers=(4,),
        squash_actions=False,
        state_preprocessor="RunningStandardScaler",
        value_preprocessor=None,
    )
    with torch.no_grad():
        agent.network.policy_head.weight.zero_()
        agent.network.policy_head.bias.fill_(1.0)
    policy_runner = _BenchmarkPolicyRunner(
        agent=agent,
        device=torch.device(device),
        obs_dim=5,
        batch_size=2,
        compile_enabled=False,
        compile_mode="default",
        use_cuda_graph=device == "cuda",
    )
    snapshot = policy_runner._preallocated_policy
    assert snapshot is not None
    obs = torch.zeros(2, 5, device=device)
    actions = torch.empty(2, 2, device=device)
    snapshot.policy_mean_into(obs, actions)
    torch.testing.assert_close(actions, torch.ones_like(actions))
    with torch.no_grad():
        agent.network.policy_head.bias.fill_(2.0)
    checkpoint = tmp_path / "next_agent.pt"
    torch.save(agent.state_dict(), checkpoint)
    with torch.no_grad():
        agent.network.policy_head.bias.fill_(1.0)
    runtime = SimpleNamespace(
        agent=agent, env_raw=object(), policy_runner=policy_runner
    )
    runner = ClipFailureBenchmarkRunner.__new__(ClipFailureBenchmarkRunner)
    runner._runtime = runtime
    runner._ensure_runtime(checkpoint)
    snapshot.policy_mean_into(obs, actions)
    torch.testing.assert_close(actions, torch.full_like(actions, 2.0))

    # The ordinary runner may use a separate full-policy CUDA graph snapshot.
    policy_runner.policy_mean_into(obs, actions)
    torch.testing.assert_close(actions, torch.full_like(actions, 2.0))
