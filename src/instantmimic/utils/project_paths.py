"""Where the repository, the invoking project, and per-user caches live.

`REPO_ROOT` is derived from this file's location, which lands on the checkout
root for a source checkout and on the interpreter's library directory for an
installed wheel. Anything that writes must therefore go through
`is_source_checkout()` rather than assuming `REPO_ROOT` is writable or ours.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from glob import glob, has_magic
from pathlib import Path

from hydra.utils import get_original_cwd

REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    return REPO_ROOT


def is_source_checkout() -> bool:
    """Whether `REPO_ROOT` really is this repository rather than site-packages.

    An installed wheel puts this module under ``<prefix>/lib/pythonX.Y/
    site-packages/instantmimic/utils/``, so ``parents[3]`` is the library
    directory. Both markers below ship only in the checkout.
    """

    return (REPO_ROOT / "pyproject.toml").is_file() and (
        REPO_ROOT / "config" / "config.yaml"
    ).is_file()


def framework_config_dir() -> Path:
    """Return the repository ``config/`` tree that Hydra composes from.

    Raises:
        RuntimeError: The package is installed rather than run from a checkout.
            `config/` is not packaged, so there is nothing to compose.
    """

    if is_source_checkout():
        return REPO_ROOT / "config"
    raise RuntimeError(
        "instantmimic needs its config/ tree, which is not part of the "
        "installed package. Run train and play from a source checkout:\n"
        "    git clone https://github.com/Scripter36/InstantMimic\n"
        "    cd InstantMimic && uv sync --locked\n"
        "    uv run train --config-name <config>\n"
        f"(looked for config/config.yaml under {REPO_ROOT})"
    )


def user_cache_root(*parts: str | Path) -> Path:
    configured_root = os.environ.get("XDG_CACHE_HOME")
    configured_path = Path(configured_root).expanduser() if configured_root else None
    base = (
        configured_path
        if configured_path is not None and configured_path.is_absolute()
        else Path.home() / ".cache"
    )
    path = base / "instantmimic"
    for part in parts:
        if str(part):
            path /= part
    return path.resolve()


def resolve_cache_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return user_cache_root(candidate)


def project_root() -> Path:
    """Return the invoking project root, falling back to the current directory."""

    original_cwd = hydra_original_cwd()
    return original_cwd if original_cwd is not None else Path.cwd().resolve()


def _writable_root(name: str, parts: tuple[str | Path, ...]) -> Path:
    # Only a checkout owns REPO_ROOT. Installed, it is the interpreter's library
    # directory, which must never be written to.
    base = REPO_ROOT if is_source_checkout() else project_root()
    path = base / name
    for part in parts:
        if str(part):
            path /= part
    return path.resolve()


def repo_analysis_root(*parts: str | Path) -> Path:
    """Task analysis root: ``<checkout>/workspaces``, else the invoking project."""

    return _writable_root("workspaces", parts)


def repo_artifact_root(*parts: str | Path) -> Path:
    """Task artifact root: ``<checkout>/workspaces``, else the invoking project."""

    return _writable_root("workspaces", parts)


def hydra_original_cwd() -> Path | None:
    try:
        return Path(get_original_cwd()).resolve()
    except (RuntimeError, ValueError):
        return None


def candidate_project_roots() -> list[Path]:
    roots = [project_root(), REPO_ROOT]

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_roots.append(resolved)
    return unique_roots


def _unique_resolved_paths(paths: Iterable[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(resolved)
    return unique_paths


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    for root in candidate_project_roots():
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (REPO_ROOT / candidate).resolve()


def resolve_project_path_pattern(
    path: str | Path,
    *,
    extra_roots: Iterable[str | Path] = (),
) -> list[Path]:
    candidate = Path(path).expanduser()
    roots = [
        *candidate_project_roots(),
        *(Path(root).expanduser() for root in extra_roots),
    ]
    unique_roots = _unique_resolved_paths(roots)

    if not has_magic(str(candidate)):
        if candidate.is_absolute():
            return [candidate.resolve()]
        candidates = [(root / candidate).resolve() for root in unique_roots]
        for resolved in candidates:
            if resolved.exists():
                return [resolved]
        if candidates:
            return [candidates[0]]
        return [resolve_project_path(candidate)]

    if candidate.is_absolute():
        matches = sorted(
            Path(match).resolve() for match in glob(str(candidate), recursive=True)
        )
        return _unique_resolved_paths(matches)

    matches: list[Path] = []
    for root in unique_roots:
        pattern = root / candidate
        root_matches = sorted(
            Path(match).resolve() for match in glob(str(pattern), recursive=True)
        )
        matches.extend(root_matches)
    return _unique_resolved_paths(matches)


def resolve_project_log_root(kind: str, experiment_directory: str | Path) -> Path:
    candidate = Path(experiment_directory).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    parts = [project_root(), "logs", kind]
    if str(candidate):
        parts.append(candidate)
    return Path(*parts).resolve()
