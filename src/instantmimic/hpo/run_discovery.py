"""Discover training run directories produced by an experiment."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def experiment_root(experiment_directory: str | Path) -> Path:
    """Return the filesystem root for an experiment directory setting."""
    candidate = Path(experiment_directory)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / "logs" / "ppo" / candidate


def _resolved_run_dirs(root: Path) -> tuple[Path, ...]:
    return tuple(path.resolve() for path in root.iterdir() if path.is_dir())


def list_run_dirs(experiment_directory: str | Path) -> set[Path]:
    """Return the resolved immediate run directories for an experiment."""
    root = experiment_root(experiment_directory)
    if not root.exists():
        return set()
    return set(_resolved_run_dirs(root))


def find_newest_run_dir(
    experiment_directory: str | Path, known: set[Path]
) -> Path | None:
    """Return the newest unknown run, falling back to the newest existing run."""
    root = experiment_root(experiment_directory)
    if not root.exists():
        return None
    run_dirs = _resolved_run_dirs(root)
    if not run_dirs:
        return None
    candidates = [path for path in run_dirs if path not in known]
    if not candidates:
        candidates = list(run_dirs)
    return max(candidates, key=lambda path: path.stat().st_mtime)
