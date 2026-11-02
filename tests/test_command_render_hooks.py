"""Command environments implement every hook the shared command renderer calls."""

import ast
import inspect

import pytest

from instantmimic.envs.command import render_runtime
from instantmimic.envs.freemusco.tasks.freemusco_command_env import FreemuscoCommandEnv
from instantmimic.envs.snu.tasks.snu_command.snu_command_env import SNUCommandEnv


def _env_methods_called_by_renderer() -> set[str]:
    tree = ast.parse(inspect.getsource(render_runtime))
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "env"
    }


@pytest.mark.parametrize("env_cls", [SNUCommandEnv, FreemuscoCommandEnv])
def test_command_env_implements_renderer_hooks(env_cls):
    hooks = _env_methods_called_by_renderer()
    assert "prepare_render_root" in hooks
    missing = sorted(
        name for name in hooks if not callable(getattr(env_cls, name, None))
    )
    assert not missing, missing
