from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

_LOGGER = logging.getLogger(__name__)


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if OmegaConf.is_config(value):
        resolved = OmegaConf.to_container(value, resolve=True)
        if isinstance(resolved, dict):
            return dict(resolved)
        return {}
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def load_experiment_dotenv(*roots: str | Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        _LOGGER.warning("python-dotenv is not installed; skipping local .env loading.")
        return

    seen: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if root in seen:
            continue
        seen.add(root)
        for name in (".env", ".env.local"):
            path = root / name
            if path.exists():
                load_dotenv(path, override=False)


def resolve_hydra_config_name(*, default: str = "default") -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            config_name = str(HydraConfig.get().job.config_name).strip()
            if config_name:
                return Path(config_name).stem
    except Exception as exc:  # noqa: BLE001 - Hydra may be partially initialized in tests/tools.
        _LOGGER.debug(
            "Could not resolve Hydra config name; using default %r: %s",
            default,
            exc,
            exc_info=True,
        )
    return default


def build_wandb_init_kwargs(
    *,
    experiment_cfg: Any,
    run_name: str,
    run_dir: Path,
    default_project: str,
    sync_tensorboard: bool,
    config_payload: dict[str, Any] | None = None,
    group: str | None = None,
    job_type: str | None = "train",
) -> dict[str, Any]:
    plain_experiment = _to_plain_dict(experiment_cfg)
    if plain_experiment.get("type") == "wandb":
        wandb_kwargs = {
            key: value for key, value in plain_experiment.items() if key != "type"
        }
    else:
        wandb_kwargs = _to_plain_dict(experiment_cfg.get("wandb_kwargs", {}))
    wandb_kwargs.setdefault("project", default_project)
    wandb_kwargs.setdefault("name", run_name)
    wandb_kwargs.setdefault("dir", run_dir.as_posix())
    wandb_kwargs.setdefault("sync_tensorboard", sync_tensorboard)
    if group:
        wandb_kwargs.setdefault("group", group)
    if job_type:
        wandb_kwargs.setdefault("job_type", job_type)

    if config_payload is not None:
        merged_config = _to_plain_dict(wandb_kwargs.get("config", {}))
        merged_config.update(config_payload)
        wandb_kwargs["config"] = merged_config

    return wandb_kwargs


def build_mlflow_init_kwargs(
    *,
    experiment_cfg: Any,
    run_name: str,
    default_experiment: str = "InstantMimic",
    group: str | None = None,
    job_type: str | None = "train",
    config_name: str | None = None,
    setting_name: str | None = None,
) -> dict[str, Any]:
    plain_experiment = _to_plain_dict(experiment_cfg)
    if plain_experiment.get("type") == "mlflow":
        mlflow_kwargs = {
            key: value for key, value in plain_experiment.items() if key != "type"
        }
    else:
        mlflow_kwargs = _to_plain_dict(experiment_cfg.get("mlflow_kwargs", {}))
    tags = _to_plain_dict(mlflow_kwargs.pop("tags", {}))
    experiment_name = str(
        mlflow_kwargs.pop("experiment_name", default_experiment)
    ).strip()
    if not experiment_name:
        experiment_name = default_experiment

    mlflow_kwargs.setdefault("run_name", run_name)
    mlflow_kwargs["experiment_name"] = experiment_name
    if group:
        tags.setdefault("group", group)
    if job_type:
        tags.setdefault("job_type", job_type)
    if config_name:
        tags.setdefault("config_name", config_name)
    if setting_name:
        tags.setdefault("setting_name", setting_name)
    mlflow_kwargs["tags"] = tags
    return mlflow_kwargs
