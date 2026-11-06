"""Only supported trainer extensions cross the semantic configuration boundary."""

import pytest
from config_fixtures import valid_config

from instantmimic.config.loader import load_semantic_config
from instantmimic.config.validation import ConfigSchemaError


@pytest.mark.parametrize("field", ["curriculum", "unsupported_extension"])
def test_trainer_rejects_unavailable_extensions(field):
    cfg = valid_config()
    cfg["trainer"][field] = {}
    with pytest.raises(ConfigSchemaError, match=rf"trainer\.{field}: unknown key"):
        load_semantic_config(cfg)


def test_removed_fullgraph_update_compile_key_is_rejected():
    cfg = valid_config()
    cfg["runtime"]["compile_update_fullgraph"] = True
    with pytest.raises(
        ConfigSchemaError,
        match=r"runtime\.compile_update_fullgraph: was removed; full update "
        r"compilation is implied by runtime\.compile together with "
        r"runtime\.update_cuda_graph",
    ):
        load_semantic_config(cfg)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        (
            "shuffle_cuda_graph",
            r"runtime\.shuffle_cuda_graph: was removed; PPO captures its "
            r"minibatch shuffle inside the update graph whenever "
            r"runtime\.update_cuda_graph=true",
        ),
        (
            "compile_act_into",
            r"runtime\.compile_act_into: was removed; runtime\.compile compiles "
            r"the rollout act_into path whenever compilation is enabled",
        ),
    ],
)
def test_removed_runtime_switch_is_rejected_with_its_replacement(key, message):
    cfg = valid_config()
    cfg["runtime"][key] = False

    with pytest.raises(ConfigSchemaError, match=message):
        load_semantic_config(cfg)


def test_removed_env_compile_mjw_step_key_is_rejected():
    cfg = valid_config()
    cfg["env"]["compile_mjw_step"] = False

    with pytest.raises(
        ConfigSchemaError,
        match=r"env\.compile_mjw_step: was removed; the repo-owned MuJoCo Warp "
        r"step suffix is always captured as a CUDA graph whenever "
        r"env\.use_cuda_graph is off",
    ):
        load_semantic_config(cfg)


def _clip_failure_config() -> dict:
    cfg = valid_config()
    del cfg["trainer"]["benchmark"]
    cfg["trainer"]["clip_failure_benchmark"] = {
        "interval": 2,
        "mode": "anchor",
        "window_steps": 20,
        "batch_size": 2,
    }
    return cfg


@pytest.mark.parametrize(
    ("key", "message"),
    [
        (
            "cuda_graph",
            r"trainer\.clip_failure_benchmark\.cuda_graph: was removed; the "
            r"clip-failure benchmark inherits env\.use_cuda_graph from the run",
        ),
        (
            "compile",
            r"trainer\.clip_failure_benchmark\.compile: was removed; the "
            r"clip-failure benchmark inherits runtime\.compile from the run",
        ),
    ],
)
def test_clip_failure_benchmark_rejects_its_removed_execution_switches(key, message):
    cfg = _clip_failure_config()
    cfg["trainer"]["clip_failure_benchmark"][key] = False

    with pytest.raises(ConfigSchemaError, match=message):
        load_semantic_config(cfg)


def test_clip_failure_benchmark_accepts_no_execution_overrides():
    cfg = _clip_failure_config()

    semantic = load_semantic_config(cfg)

    assert semantic.trainer.clip_failure_benchmark.window_steps == 20


def test_clip_failure_inline_execution_rejects_the_inherited_env_graph():
    cfg = _clip_failure_config()
    cfg["trainer"]["clip_failure_benchmark"]["execution"] = "inline"
    cfg["env"]["use_cuda_graph"] = True

    with pytest.raises(
        ConfigSchemaError,
        match=r"trainer\.clip_failure_benchmark\.execution: inline execution "
        r"conflicts with env\.use_cuda_graph=true",
    ):
        load_semantic_config(cfg)


def test_clip_failure_inline_execution_is_allowed_without_an_env_graph():
    cfg = _clip_failure_config()
    cfg["trainer"]["clip_failure_benchmark"]["execution"] = "inline"
    cfg["env"]["use_cuda_graph"] = False

    semantic = load_semantic_config(cfg)

    assert semantic.trainer.clip_failure_benchmark.execution == "inline"
