"""TF32 is scoped to learning-network computation, never to environment code."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from instantmimic.bootstrap import configure_process
from instantmimic.utils.network_precision import (
    IEEE_MATMUL_PRECISION,
    NETWORK_MATMUL_PRECISION,
    configure_ieee_float32_precision,
    network_matmul_precision,
)


def _cudnn_precisions() -> tuple[str, str]:
    return (
        torch.backends.cudnn.conv.fp32_precision,
        torch.backends.cudnn.rnn.fp32_precision,
    )


def _cudnn_allow_tf32() -> bool:
    return bool(torch.backends.cudnn.allow_tf32)


@pytest.fixture(autouse=True)
def _restore_process_precision():
    matmul = torch.get_float32_matmul_precision()
    allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    conv, rnn = _cudnn_precisions()
    yield
    torch.set_float32_matmul_precision(matmul)
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.conv.fp32_precision = conv
    torch.backends.cudnn.rnn.fp32_precision = rnn


def test_network_scope_enables_tf32_and_restores_ieee() -> None:
    configure_ieee_float32_precision()

    with network_matmul_precision():
        assert torch.get_float32_matmul_precision() == NETWORK_MATMUL_PRECISION
        assert torch.backends.cuda.matmul.fp32_precision == "tf32"
        assert _cudnn_precisions() == ("tf32", "tf32")
        assert _cudnn_allow_tf32() is True
        # Inductor reads this legacy getter for autotune keys; the new
        # per-backend setter to "tf32" would make it raise a mixed-API error.
        assert torch.get_float32_matmul_precision() == "high"

    assert torch.get_float32_matmul_precision() == IEEE_MATMUL_PRECISION
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"
    assert _cudnn_precisions() == ("ieee", "ieee")
    assert _cudnn_allow_tf32() is False


def test_network_scope_restores_after_an_exception() -> None:
    configure_ieee_float32_precision()

    with pytest.raises(RuntimeError, match="scoped failure"):
        with network_matmul_precision():
            assert torch.get_float32_matmul_precision() == "high"
            raise RuntimeError("scoped failure")

    assert torch.get_float32_matmul_precision() == "highest"
    assert _cudnn_precisions() == ("ieee", "ieee")


def test_nested_network_scopes_restore_the_outer_network_state() -> None:
    configure_ieee_float32_precision()

    with network_matmul_precision():
        with network_matmul_precision():
            assert torch.get_float32_matmul_precision() == "high"
        assert torch.get_float32_matmul_precision() == "high"
        assert _cudnn_precisions() == ("tf32", "tf32")

    assert torch.get_float32_matmul_precision() == "highest"


def test_configure_process_selects_ieee_float32() -> None:
    configure_process()

    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"
    assert _cudnn_precisions() == ("ieee", "ieee")
    assert _cudnn_allow_tf32() is False


def test_scope_outside_compiled_body_does_not_graph_break() -> None:
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 16),
    )
    compiled = torch.compile(model, mode="default")
    obs = torch.randn(8, 16)

    with network_matmul_precision():
        compiled(obs)
    for _ in range(3):
        # Environment-side math runs outside the scope, as it does between
        # rollout and update calls.
        _ = obs @ obs.transpose(0, 1)
        with network_matmul_precision():
            compiled(obs)

    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] == 1


class _ScopeRecorder:
    def __init__(self) -> None:
        self.depth = 0
        self.entries = 0

    @contextmanager
    def __call__(self):
        self.entries += 1
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1


def test_ppo_act_into_and_update_run_inside_the_network_scope(monkeypatch) -> None:
    from instantmimic.rl.ppo import agent as module

    recorder = _ScopeRecorder()
    monkeypatch.setattr(module, "network_matmul_precision", recorder)
    agent = object.__new__(module.PPOAgent)
    agent._nvtx_range = lambda _name: nullcontext()
    seen: list[int] = []
    agent._act_into_body_unscoped = lambda step, *, eps=None: seen.append(
        recorder.depth
    )

    agent.act_into(0)
    assert seen == [1]
    assert recorder.depth == 0

    agent._update_unscoped = lambda: seen.append(recorder.depth) or {}
    assert agent.update() == {}
    assert seen[-1] == 1
    assert recorder.depth == 0


def test_musclenn_update_runs_inside_the_network_scope(monkeypatch) -> None:
    from instantmimic.envs.controls.fast_muscle import training_runtime

    recorder = _ScopeRecorder()
    monkeypatch.setattr(training_runtime, "network_matmul_precision", recorder)
    host = object.__new__(training_runtime.FastMuscleTrainingMixin)
    seen: list[int] = []
    host._optimize_muscle_nn_unscoped = lambda: seen.append(recorder.depth) or None
    host._begin_muscle_nn_update_unscoped = lambda: seen.append(recorder.depth) or False

    assert host.optimize_muscle_nn() is None
    assert seen == [1]
    assert recorder.depth == 0

    assert host.begin_muscle_nn_update() is False
    assert seen[-1] == 1


def test_public_policy_entry_points_observe_tf32_without_callers() -> None:
    from instantmimic.rl.ppo.agent import PPOAgent

    obs = torch.zeros(1, 2)
    out = torch.empty(1, 2)

    def record(seen: list[str]) -> None:
        seen.append(torch.get_float32_matmul_precision())

    ppo_seen: list[str] = []
    ppo = object.__new__(PPOAgent)
    ppo._scaled_eval_obs = lambda inputs: inputs
    ppo.network = SimpleNamespace(
        policy_mean=lambda _obs: record(ppo_seen) or torch.zeros(1, 2),
        policy_mean_into=lambda _obs, _out: record(ppo_seen),
        policy_execution_mean_into=lambda _obs, _out, **_kw: record(ppo_seen),
    )
    configure_ieee_float32_precision()
    ppo.policy_mean(obs)
    ppo.policy_mean_into(obs, out)
    ppo.policy_execution_mean_into(obs, out)
    assert ppo_seen == ["high"] * 3
    assert torch.get_float32_matmul_precision() == "highest"


@pytest.mark.parametrize("execution_mean", [False, True])
def test_playback_compiled_wrapper_traces_the_unscoped_policy_body(
    execution_mean,
) -> None:
    from torch._dynamo.utils import counters

    from instantmimic.play.stepper import _maybe_build_compiled_playback_policy
    from instantmimic.rl.eager_fallback import EagerFallbackPolicy

    weight = torch.randn(3, 3)

    class Agent:
        def policy_mean_into(self, obs, out):
            raise AssertionError("compiled wrapper traced the scoped public method")

        def _policy_mean_into_unscoped(self, obs, out):
            out.copy_(obs @ weight)

        def policy_execution_mean_into(
            self,
            _obs,
            _out,
        ):
            raise AssertionError("compiled wrapper traced the scoped public method")

        def _policy_execution_mean_into_unscoped(
            self,
            obs,
            out,
        ):
            out.copy_(obs @ weight)

    torch._dynamo.reset()
    counters.clear()
    compiled = _maybe_build_compiled_playback_policy(
        agent=Agent(),
        device=SimpleNamespace(type="cuda"),
        obs_policy=torch.zeros(1, 3),
        agent_actions=torch.zeros(1, 3),
        play_torch_stream=None,
        stochastic_play=False,
        use_execution_mean=execution_mean,
        eager_fallback=EagerFallbackPolicy(allowed=False),
    )
    key = "execution_mean" if execution_mean else "mean"
    assert compiled[key] is not None
    out = torch.empty(1, 3)
    obs = torch.ones(1, 3)
    compiled[key](obs, out)
    torch.testing.assert_close(out, obs @ weight)
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] == 1


def test_benchmark_compiled_wrapper_traces_the_unscoped_policy_body() -> None:
    from torch._dynamo.utils import counters

    from instantmimic.train.benchmark_runner import _BenchmarkPolicyRunner

    weight = torch.randn(2, 1)

    class Agent:
        act_dim = 1
        network = None

        def policy_mean_into(self, _obs, _out):
            raise AssertionError("compiled wrapper traced the scoped public method")

        def _policy_mean_into_unscoped(self, obs, out):
            out.copy_(obs @ weight)

    torch._dynamo.reset()
    counters.clear()
    runner = _BenchmarkPolicyRunner(
        agent=Agent(),
        device=torch.device("cpu"),
        obs_dim=2,
        batch_size=2,
        compile_enabled=True,
        compile_mode="default",
        use_cuda_graph=False,
    )
    out = torch.empty(2, 1)
    obs = torch.ones(2, 2)
    runner.policy_mean_into(obs, out)
    torch.testing.assert_close(out, obs @ weight)
    assert sum(counters["graph_break"].values()) == 0
    assert counters["stats"]["unique_graphs"] == 1
