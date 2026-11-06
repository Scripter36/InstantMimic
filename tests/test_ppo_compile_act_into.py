"""Compile controls the rollout ``act_into`` and its warmup matches rollout."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch

from instantmimic.rl.ppo.agent import PPOAgent


def _agent(*, compile: bool = False) -> PPOAgent:
    return PPOAgent(
        obs_dim=6,
        act_dim=2,
        num_envs=4,
        device="cpu",
        rollouts=4,
        learning_epochs=2,
        mini_batches=2,
        shuffle_mode="per_update",
        discount_factor=0.95,
        lam=0.95,
        learning_rate=0.001,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.001,
        value_loss_scale=4.5,
        kl_threshold=0.0,
        layers=(12, 8),
        activation="silu",
        initial_log_std=-2.0,
        optimizer_name="adamw",
        compile=compile,
    )


def test_compile_compiles_the_rollout_act_into() -> None:
    compiled_names: list[str] = []

    def recording_compile(fn: Any, **_kwargs: Any) -> Any:
        compiled_names.append(getattr(fn, "__name__", ""))
        return fn

    with patch("torch.compile", recording_compile):
        agent = _agent(compile=True)

    assert "act_into" in compiled_names
    assert agent._act_into_fn.__func__ is agent.network.act_into.__func__


def test_without_compile_the_rollout_act_into_stays_eager() -> None:
    compiled_names: list[str] = []

    def recording_compile(fn: Any, **_kwargs: Any) -> Any:
        compiled_names.append(getattr(fn, "__name__", ""))
        return fn

    with patch("torch.compile", recording_compile):
        agent = _agent(compile=False)

    assert "act_into" not in compiled_names
    assert agent._act_into_fn.__func__ is agent.network.act_into.__func__


def _argument_signature(value: Any) -> object:
    if value is None:
        return None
    if torch.is_tensor(value):
        return ("tensor", tuple(value.shape), value.dtype)
    return ("value", type(value).__name__)


def _call_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> object:
    return (
        tuple(_argument_signature(argument) for argument in args),
        {
            name: _argument_signature(argument)
            for name, argument in sorted(kwargs.items())
        },
    )


def _record_calls(agent: PPOAgent) -> list[object]:
    calls: list[object] = []

    def record(*args: Any, **kwargs: Any) -> None:
        calls.append(_call_signature(args, kwargs))

    agent._act_into_fn = record  # type: ignore[method-assign]
    return calls


def test_warmup_matches_the_rollout_and_bootstrap_signatures() -> None:
    agent = _agent()
    agent._warmup_compiled_act_into = True
    calls = _record_calls(agent)

    agent._warmup_compiled_act_into_signatures()
    warmup_calls = list(calls)
    assert len(warmup_calls) == 2

    calls.clear()
    with torch.no_grad():
        agent.act_into(0)
    rollout_calls = list(calls)
    calls.clear()
    agent._bootstrap_values_fast(
        agent.buffer.obs[0],
        torch.empty_like(agent.buffer.values[0]),
    )
    bootstrap_calls = list(calls)

    assert len(rollout_calls) == 1
    assert len(bootstrap_calls) == 1
    assert warmup_calls[0] == rollout_calls[0]
    assert warmup_calls[1] == bootstrap_calls[0]


def test_warmup_does_not_consume_rollout_noise_rng() -> None:
    agent = _agent()
    agent._warmup_compiled_act_into = True
    _record_calls(agent)
    before = torch.get_rng_state()

    agent._warmup_compiled_act_into_signatures()

    # Warmup reuses the persistent policy noise buffer instead of drawing, so
    # a warm run sees exactly the rollouts of one that did not warm up.
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.cuda
def test_compiled_act_into_does_not_recompile_inside_capture() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from instantmimic.envs.direct_rl_env import (
        _launch_stream_aligned_cuda_graph,
        _stream_aligned_cuda_graph_capture,
    )

    torch._dynamo.reset()
    agent = PPOAgent(
        obs_dim=6,
        act_dim=2,
        num_envs=4,
        device="cuda",
        rollouts=4,
        learning_epochs=1,
        mini_batches=1,
        shuffle_mode="per_update",
        discount_factor=0.95,
        lam=0.95,
        learning_rate=0.001,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.001,
        value_loss_scale=4.5,
        kl_threshold=0.0,
        layers=(12, 8),
        activation="silu",
        initial_log_std=-2.0,
        optimizer_name="adamw",
        compile=True,
    )
    agent._warmup_compiled_act_into_signatures(force=True)
    torch.cuda.synchronize(agent.device)

    previous = torch._dynamo.config.error_on_recompile
    torch._dynamo.config.error_on_recompile = True
    try:
        with _stream_aligned_cuda_graph_capture(agent.device) as capture:
            agent.act_into(0)
        _launch_stream_aligned_cuda_graph(capture.graph, agent.device)
        torch.cuda.synchronize(agent.device)
    finally:
        torch._dynamo.config.error_on_recompile = previous
