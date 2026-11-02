from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar

from hydra._internal.utils import get_args_parser

_CONFIG_PATH_OPTIONS = ("--config-path", "-cp")
_CONFIG_DIR_OPTIONS = ("--config-dir", "-cd")
_HYDRA_ARTIFACT_KEYS = (
    "hydra.run.dir",
    "hydra.sweep.dir",
    "hydra.sweep.subdir",
    "hydra.output_subdir",
    "hydra.job.chdir",
    "hydra/job_logging",
    "hydra/hydra_logging",
)

_T = TypeVar("_T")


def _contains_option(argv: Sequence[str], options: tuple[str, ...]) -> bool:
    return any(
        token == option or token.startswith(f"{option}=")
        for token in argv
        for option in options
    )


def add_project_config_dir(
    argv: Sequence[str],
    *,
    project_root: str | Path,
    framework_config_dir: str | Path,
) -> list[str]:
    """Add ``<project_root>/config`` without replacing the framework search root."""

    args = list(argv)
    if _contains_option(args, _CONFIG_PATH_OPTIONS + _CONFIG_DIR_OPTIONS):
        return args

    project_config_dir = (Path(project_root).resolve() / "config").resolve()
    framework_dir = Path(framework_config_dir).resolve()
    if project_config_dir == framework_dir or not project_config_dir.is_dir():
        return args
    # Hydra's positional overrides must follow all parser options. Appending
    # an option after overrides breaks parsing when our runtime adds more.
    return ["--config-dir", project_config_dir.as_posix(), *args]


def _is_hydra_artifact_override(token: str) -> bool:
    normalized = token.lstrip("+~")
    return any(
        normalized == key or normalized.startswith(f"{key}=")
        for key in _HYDRA_ARTIFACT_KEYS
    )


def hydra_no_output_args(
    argv: Sequence[str],
    *,
    runtime_dir: str | Path,
) -> list[str]:
    """Return Hydra CLI args that keep all framework artifacts ephemeral."""

    root = Path(runtime_dir).expanduser().resolve()
    args = [str(token) for token in argv if not _is_hydra_artifact_override(str(token))]
    artifact_overrides = [
        f"hydra.run.dir={(root / 'run').as_posix()}",
        f"hydra.sweep.dir={(root / 'sweep').as_posix()}",
        "hydra.sweep.subdir=${hydra.job.num}",
        "hydra.output_subdir=null",
        "hydra.job.chdir=false",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]
    # Hydra accepts one contiguous block of positional overrides, so ours must
    # join the user's block; `a=1 --cfg job` would otherwise gain a second one.
    user_overrides = get_args_parser().parse_known_args(args)[0].overrides
    insert_at = len(args)
    if user_overrides:
        insert_at = len(args) - args[::-1].index(user_overrides[-1])
    return [*args[:insert_at], *artifact_overrides, *args[insert_at:]]


def run_hydra_without_outputs(entrypoint: Callable[[], _T]) -> _T:
    """Run a decorated Hydra entrypoint without persistent framework output."""

    original_argv = list(sys.argv)
    with TemporaryDirectory(prefix="instantmimic-hydra-") as runtime_dir:
        sys.argv = [
            original_argv[0],
            *hydra_no_output_args(original_argv[1:], runtime_dir=runtime_dir),
        ]
        try:
            return entrypoint()
        finally:
            sys.argv = original_argv


__all__ = [
    "add_project_config_dir",
    "hydra_no_output_args",
    "run_hydra_without_outputs",
]
