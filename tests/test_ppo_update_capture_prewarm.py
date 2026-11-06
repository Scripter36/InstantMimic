from __future__ import annotations

from unittest import mock

import torch

from instantmimic.rl.ppo.agent import PPOAgent


def _bare_agent(
    *,
    compile_enabled: bool,
    compile_update_fullgraph: bool,
    use_update_graph: bool = True,
    compiled_model_functions_warmed: bool = True,
    capture_prewarmed: bool = False,
) -> PPOAgent:
    agent = object.__new__(PPOAgent)
    agent.device = torch.device("cpu")
    agent._compile_enabled = compile_enabled
    agent._compile_update_fullgraph = compile_update_fullgraph
    agent._use_update_graph = use_update_graph
    agent._compiled_model_functions_warmed = compiled_model_functions_warmed
    agent._compiled_update_capture_prewarmed = capture_prewarmed
    agent._warmup_compiled_act_into = True
    agent._detailed_nvtx = False
    agent._compile_update_wrappers_if_needed = mock.Mock()
    agent._warmup_compiled_functions = mock.Mock()
    agent._refresh_epoch_permutations = mock.Mock()
    agent._warmup_compiled_update_minibatch = mock.Mock()
    return agent


def test_prewarm_runs_update_warmup_without_fullgraph_compile() -> None:
    agent = _bare_agent(
        compile_enabled=True,
        compile_update_fullgraph=False,
    )

    agent._prewarm_compiled_update_capture_path()

    agent._refresh_epoch_permutations.assert_called_once_with()
    agent._warmup_compiled_update_minibatch.assert_called_once_with()
    agent._compile_update_wrappers_if_needed.assert_called_once_with()
    assert agent._compiled_update_capture_prewarmed is True


def test_prewarm_skips_everything_when_compile_disabled() -> None:
    agent = _bare_agent(
        compile_enabled=False,
        compile_update_fullgraph=False,
        compiled_model_functions_warmed=False,
    )

    agent._prewarm_compiled_update_capture_path()

    agent._warmup_compiled_update_minibatch.assert_not_called()
    agent._refresh_epoch_permutations.assert_not_called()
    agent._warmup_compiled_functions.assert_not_called()
    agent._compile_update_wrappers_if_needed.assert_not_called()
    assert agent._compiled_update_capture_prewarmed is False


def test_prewarm_warms_model_callables_before_update_warmup() -> None:
    agent = _bare_agent(
        compile_enabled=True,
        compile_update_fullgraph=False,
        compiled_model_functions_warmed=False,
    )
    order: list[str] = []
    agent._warmup_compiled_functions.side_effect = lambda **_: order.append("model")
    agent._warmup_compiled_update_minibatch.side_effect = lambda: order.append("update")

    agent._prewarm_compiled_update_capture_path()

    agent._warmup_compiled_functions.assert_called_once_with(warmup_act_into=True)
    assert order == ["model", "update"]


def test_prewarm_does_not_repeat_a_completed_update_warmup() -> None:
    agent = _bare_agent(
        compile_enabled=True,
        compile_update_fullgraph=False,
        capture_prewarmed=True,
    )

    agent._prewarm_compiled_update_capture_path()

    agent._warmup_compiled_update_minibatch.assert_not_called()
    agent._refresh_epoch_permutations.assert_not_called()


def test_prewarm_requires_the_captured_update_path() -> None:
    agent = _bare_agent(
        compile_enabled=True,
        compile_update_fullgraph=False,
        use_update_graph=False,
        compiled_model_functions_warmed=False,
    )

    agent._prewarm_compiled_update_capture_path()

    agent._warmup_compiled_update_minibatch.assert_not_called()
    agent._warmup_compiled_functions.assert_not_called()
    assert agent._compiled_update_capture_prewarmed is False


def test_prewarm_preserves_permutations_and_generator_state() -> None:
    agent = _bare_agent(
        compile_enabled=True,
        compile_update_fullgraph=False,
    )
    agent._epoch_permutations = [
        torch.arange(8, dtype=torch.int32),
        torch.arange(8, dtype=torch.int32),
    ]
    agent._shuffle_generator = torch.Generator().manual_seed(1234)

    def refresh_permutations() -> None:
        torch.randperm(
            8,
            generator=agent._shuffle_generator,
            out=agent._epoch_permutations[0],
        )
        torch.randperm(8, out=agent._epoch_permutations[1])

    def mutate_during_warmup() -> None:
        torch.rand(3)
        agent._shuffle_generator.manual_seed(999)
        agent._epoch_permutations[0].fill_(-1)

    agent._refresh_epoch_permutations = refresh_permutations
    agent._warmup_compiled_update_minibatch = mutate_during_warmup
    shuffle_generator_state = agent._shuffle_generator.get_state().clone()
    cpu_rng_state = torch.get_rng_state().clone()
    permutations = [permutation.clone() for permutation in agent._epoch_permutations]

    agent._prewarm_compiled_update_capture_path()

    assert torch.equal(agent._shuffle_generator.get_state(), shuffle_generator_state)
    assert torch.equal(torch.get_rng_state(), cpu_rng_state)
    for permutation, saved in zip(agent._epoch_permutations, permutations, strict=True):
        torch.testing.assert_close(permutation, saved, rtol=0, atol=0)


def test_preserve_sampling_state_restores_generator_permutations_and_rng() -> None:
    agent = _bare_agent(
        compile_enabled=True,
        compile_update_fullgraph=False,
    )
    agent._epoch_permutations = [torch.arange(6, dtype=torch.int32)]
    agent._shuffle_generator = torch.Generator().manual_seed(7)
    shuffle_generator_state = agent._shuffle_generator.get_state().clone()
    cpu_rng_state = torch.get_rng_state().clone()
    permutation = agent._epoch_permutations[0].clone()

    with agent._preserve_update_prewarm_sampling_state():
        torch.rand(5)
        agent._shuffle_generator.manual_seed(99)
        agent._epoch_permutations[0].fill_(-1)

    assert torch.equal(agent._shuffle_generator.get_state(), shuffle_generator_state)
    assert torch.equal(torch.get_rng_state(), cpu_rng_state)
    torch.testing.assert_close(
        agent._epoch_permutations[0], permutation, rtol=0, atol=0
    )


def test_actual_update_warmup_restores_network_buffers_in_place() -> None:
    agent = PPOAgent(
        obs_dim=4,
        act_dim=2,
        num_envs=2,
        device="cpu",
        rollouts=4,
        learning_epochs=1,
        mini_batches=1,
        shuffle_mode="per_epoch",
        discount_factor=0.99,
        lam=0.95,
        learning_rate=3e-4,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.01,
        value_loss_scale=1.0,
        kl_threshold=0.01,
        kl_estimator="analytic",
        fixed_sigma=False,
        layers=(8, 8),
    )
    agent.network.register_buffer(
        "warmup_transport_progress",
        torch.tensor(0.375, dtype=torch.float32),
    )
    agent.network.register_buffer(
        "warmup_transport_steps",
        torch.tensor([3, 4, 5], dtype=torch.int64),
    )
    buffers = [
        agent.network.get_buffer("warmup_transport_progress"),
        agent.network.get_buffer("warmup_transport_steps"),
    ]
    pointers = [buffer.data_ptr() for buffer in buffers]
    before = [buffer.detach().clone() for buffer in buffers]

    def advance_buffers(*, max_minibatches=None):
        assert max_minibatches == 1
        buffers[0].fill_(1.0)
        buffers[1].add_(1)

    agent._run_update_epoch_loops = advance_buffers
    agent._warmup_compiled_update_minibatch(max_minibatches=1)

    for buffer, pointer, saved in zip(buffers, pointers, before, strict=True):
        assert buffer.data_ptr() == pointer
        torch.testing.assert_close(buffer, saved, rtol=0, atol=0)
