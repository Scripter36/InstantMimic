"""Execution switches derive from the core runtime instead of being authored."""

from __future__ import annotations

import pytest
from config_fixtures import valid_config

from instantmimic.rl.build_specs import lower_agent_build_spec
from instantmimic.rl.ppo.factory import build_ppo_agent_from_spec
from instantmimic.train.benchmark import TrainingBenchmarkPlan


class _CatchAllAgent:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _built_kwargs(**runtime: object) -> dict[str, object]:
    root = valid_config()
    root["runtime"] = runtime
    spec = lower_agent_build_spec(root)
    built = build_ppo_agent_from_spec(
        spec,
        agent_cls=_CatchAllAgent,
        obs_dim=11,
        act_dim=4,
        num_envs=2,
        device="cpu",
    )
    return built.kwargs


@pytest.mark.parametrize("update_graph", [False, True])
def test_ppo_derives_the_shuffle_graph_from_the_update_graph(
    update_graph: bool,
) -> None:
    kwargs = _built_kwargs(update_cuda_graph=update_graph, compile=False)

    assert kwargs["use_shuffle_graph"] is update_graph


@pytest.mark.parametrize("compile_enabled", [False, True])
def test_factory_has_no_separate_act_into_compile_switch(
    compile_enabled: bool,
) -> None:
    kwargs = _built_kwargs(compile=compile_enabled)

    assert kwargs["compile"] is compile_enabled
    assert "compile_act_into" not in kwargs


@pytest.mark.parametrize("enabled", [False, True])
def test_clip_failure_benchmark_inherits_the_run_execution_settings(
    enabled: bool,
) -> None:
    plan = TrainingBenchmarkPlan.from_semantic_trainer(
        {
            "clip_failure_benchmark": {
                "interval": 200,
                "mode": "anchor",
                "window_steps": 180,
                "batch_size": 4096,
            }
        },
        env_use_cuda_graph=enabled,
        compile=enabled,
    )

    assert plan.clip_failure is not None
    assert plan.clip_failure.use_cuda_graph is enabled
    assert plan.clip_failure.compile is enabled
