from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from instantmimic.utils.project_paths import project_root, repo_root


def compose_config_file(
    config_path: str | Path,
    *,
    overrides: Iterable[str] = (),
) -> DictConfig:
    """Compose one Hydra YAML file from its applicable config search root."""

    resolved_path = Path(config_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {resolved_path}")
    repository_config_root = repo_root() / "config"
    try:
        config_name = (
            resolved_path.relative_to(repository_config_root).with_suffix("").as_posix()
        )
        config_dir = repository_config_root
        compose_overrides: list[str] = []
    except ValueError:
        project_config_root = (project_root() / "config").resolve()
        config_dir = (
            project_config_root
            if resolved_path.is_relative_to(project_config_root)
            else resolved_path.parent
        )
        config_name = resolved_path.relative_to(config_dir).with_suffix("").as_posix()
        compose_overrides = [
            f"hydra.searchpath=[{repository_config_root.resolve().as_uri()}]",
        ]
    compose_overrides.extend(str(item) for item in overrides)
    with initialize_config_dir(
        config_dir=config_dir.as_posix(),
        version_base=None,
    ):
        composed = compose(config_name=config_name, overrides=compose_overrides)
    if not isinstance(composed, DictConfig):
        raise TypeError(f"Composed config must be a mapping: {resolved_path}")
    return composed


__all__ = ["compose_config_file"]
