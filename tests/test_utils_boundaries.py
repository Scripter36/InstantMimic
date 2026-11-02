from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UTILS_ROOT = ROOT / "src" / "instantmimic" / "utils"
SRC_ROOT = ROOT / "src"
MJW_PRIVATE_BOUNDARY = (UTILS_ROOT / "mjw_kernels" / "private.py").relative_to(ROOT)

_TOP_LEVEL_MODULES = {
    "ball_joint_limits",
    "experiment_logging",
    "hydra_cli",
    "mjw_step_optimized",
    "mjw_model",
    "policy_mean_contract",
    "project_paths",
    "spaces",
    "startup_profiler",
    "timer",
    "torch_warp_capture",
    "warp_config",
    "warp_kernels",
}

_EXECUTABLE_MODULES: set[str] = set()

_HIGHER_LEVEL_IMPORTS = {
    "spaces": {"instantmimic.envs"},
}

_HIGHER_LEVEL_PACKAGES = {"envs", "motion", "play", "rl", "train"}


def _module_imports(tree: ast.AST, *, module_name: str) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                package = module_name.rpartition(".")[0]
                imports.add(
                    importlib.util.resolve_name(
                        f"{'.' * node.level}{node.module}", package
                    )
                )
            else:
                imports.add(node.module)
                if node.module == "instantmimic":
                    imports.update(
                        f"instantmimic.{alias.name}"
                        for alias in node.names
                        if alias.name in _HIGHER_LEVEL_PACKAGES
                    )
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call) and node.args:
            target = node.args[0]
            if not isinstance(target, ast.Constant) or not isinstance(
                target.value, str
            ):
                continue
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                or (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                    and node.func.attr == "import_module"
                )
            ):
                imports.add(target.value)
    return imports


def _higher_level_package(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "instantmimic":
        if parts[1] in _HIGHER_LEVEL_PACKAGES:
            return ".".join(parts[:2])
    return None


def _has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        try:
            expression = ast.unparse(node.test)
        except ValueError:
            continue
        if "__name__" in expression and "__main__" in expression:
            return True
    return False


def test_utils_top_level_modules_are_an_explicit_shrinking_allowlist() -> None:
    actual = {path.stem for path in UTILS_ROOT.glob("*.py")}
    assert actual == _TOP_LEVEL_MODULES


def test_mujoco_warp_private_imports_use_one_boundary() -> None:
    offenders = []
    for path in SRC_ROOT.rglob("*.py"):
        relative_path = path.relative_to(ROOT)
        if relative_path == MJW_PRIVATE_BOUNDARY:
            continue
        if "mujoco_warp._src" in path.read_text(encoding="utf-8"):
            offenders.append(relative_path.as_posix())

    assert offenders == []


def test_utils_does_not_gain_new_executable_modules() -> None:
    actual = {
        path.relative_to(UTILS_ROOT).with_suffix("").as_posix()
        for path in UTILS_ROOT.rglob("*.py")
        if _has_main_guard(ast.parse(path.read_text(encoding="utf-8")))
    }
    assert actual == _EXECUTABLE_MODULES


def test_utils_does_not_gain_new_higher_level_dependencies() -> None:
    actual: dict[str, set[str]] = {}
    for path in UTILS_ROOT.rglob("*.py"):
        relative_module = path.relative_to(UTILS_ROOT).with_suffix("")
        module_key = ".".join(relative_module.parts)
        module_name = f"instantmimic.utils.{module_key}"
        imports = {
            package
            for module in _module_imports(
                ast.parse(path.read_text(encoding="utf-8")),
                module_name=module_name,
            )
            if (package := _higher_level_package(module)) is not None
        }
        if imports:
            actual[module_key] = imports
    assert actual == _HIGHER_LEVEL_IMPORTS


def test_dependency_scan_handles_relative_indirect_and_dynamic_imports() -> None:
    tree = ast.parse(
        """
from ..envs import common
from instantmimic import play
import importlib

importlib.import_module("instantmimic.train.runtime")
__import__("instantmimic.rl.runtime")
"""
    )
    imports = _module_imports(tree, module_name="instantmimic.utils.example")
    assert {
        package
        for module in imports
        if (package := _higher_level_package(module)) is not None
    } == {
        "instantmimic.envs",
        "instantmimic.play",
        "instantmimic.rl",
        "instantmimic.train",
    }
