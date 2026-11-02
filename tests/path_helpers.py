"""Shared filesystem paths used by tests."""

from pathlib import Path


def repo_root() -> Path:
    """Return the checkout root used by tests that need repository assets."""

    return Path(__file__).resolve().parents[1]


def fixtures_config_dir() -> Path:
    """Return the directory reserved for test-owned Hydra fixture configs."""

    return repo_root() / "tests" / "fixtures" / "config"
