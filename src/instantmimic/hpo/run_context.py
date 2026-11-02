from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class HPORunContext:
    task_path: Path
    task_cfg: DictConfig
    run_dir: Path | None
    state_dir: Path
    resumed: bool


def _copy_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _resolve_root_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (root / candidate).resolve()


def _relative_path_text(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _slugify(value: object) -> str:
    text = str(value or "").strip()
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.-")
    return slug.replace("-", "_").lower() or "hpo"


def task_slug(task_cfg: DictConfig, task_path: Path) -> str:
    name = str(task_cfg.get("name", "") or "").strip()
    return _slugify(name or task_path.stem)


def _config_path_for_name(root: Path, config_name: str) -> Path:
    raw_name = str(config_name).strip()
    if not raw_name:
        raise ValueError("HPO config name must be non-empty.")
    rel_path = Path(raw_name)
    if rel_path.suffix not in {".yaml", ".yml"}:
        rel_path = rel_path.with_suffix(".yaml")
    return (root / "config" / rel_path).resolve()


def hpo_config_path_for_name(root: Path, config_name: str) -> Path:
    return _config_path_for_name(root, config_name)


def split_hpo_runtime_cfg(cfg: DictConfig) -> tuple[DictConfig, DictConfig]:
    task_cfg = _copy_cfg(cfg)
    runtime_cfg: object
    if "runtime" in task_cfg:
        runtime_cfg = task_cfg.pop("runtime")
    else:
        runtime_cfg = OmegaConf.create({})
    if runtime_cfg is None:
        runtime_cfg = OmegaConf.create({})
    if not isinstance(runtime_cfg, DictConfig):
        raise ValueError("HPO runtime config must be a mapping.")
    return task_cfg, runtime_cfg


def _load_mapping(path: Path, overrides: Iterable[str]) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(
            f"HPO task file '{path}' must contain a mapping at the top level."
        )
    override_list = list(overrides)
    if override_list:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(override_list))
    return cfg


def _config_name_for_repo_config(root: Path, path: Path) -> str | None:
    config_root = (root / "config").resolve()
    try:
        relative_path = path.resolve().relative_to(config_root)
    except ValueError:
        return None
    if relative_path.suffix not in {".yaml", ".yml"}:
        return None
    return relative_path.with_suffix("").as_posix()


def _declares_hydra_defaults(path: Path) -> bool:
    cfg = OmegaConf.load(path)
    return isinstance(cfg, DictConfig) and cfg.get("defaults", None) is not None


def _load_repo_config_mapping(
    *,
    root: Path,
    path: Path,
    overrides: Iterable[str],
) -> DictConfig | None:
    config_name = _config_name_for_repo_config(root, path)
    if config_name is None:
        return None
    if not _declares_hydra_defaults(path):
        return None

    try:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(
            config_dir=str((root / "config").resolve()),
            version_base=None,
        ):
            cfg = compose(config_name=config_name, overrides=list(overrides))
    except Exception as exc:
        raise ValueError(
            f"Could not compose hpo config '{path}' with Hydra: {exc}"
        ) from exc

    if not isinstance(cfg, DictConfig):
        raise ValueError(f"HPO task config '{path}' must compose to a mapping.")
    return cfg


def load_hpo_task(
    *,
    root: Path,
    task: str | Path | None,
    config_name: str | None,
    overrides: Iterable[str] = (),
) -> tuple[Path, DictConfig]:
    if config_name:
        path = _config_path_for_name(root, config_name)
    elif task is not None:
        task_text = str(task).strip()
        if not task_text:
            raise ValueError("HPO task path or config name must be non-empty.")
        candidate = Path(task_text).expanduser()
        if candidate.suffix in {".yaml", ".yml"} or candidate.exists():
            path = _resolve_root_path(root, candidate)
        else:
            path = _config_path_for_name(root, task_text)
    else:
        raise ValueError(
            "Provide an hpo config YAML path or --config-name such as hpo/backflip_pd."
        )

    if not path.exists():
        raise FileNotFoundError(f"HPO task config does not exist: {path}")
    cfg = _load_repo_config_mapping(root=root, path=path, overrides=overrides)
    if cfg is None:
        cfg = _load_mapping(path, overrides)
    return path, cfg


def _timestamp_name() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def default_hpo_run_root(root: Path, task_cfg: DictConfig, task_path: Path) -> Path:
    """Where canonical controller, evaluator and training runs live."""
    return root / "logs" / "hpo" / task_slug(task_cfg, task_path)


def create_unique_hpo_run_dir(
    *,
    root: Path,
    task_cfg: DictConfig,
    task_path: Path,
    timestamp_name: str | None = None,
    max_attempts: int = 10000,
) -> Path:
    base_dir = default_hpo_run_root(root, task_cfg, task_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = str(timestamp_name or _timestamp_name()).strip() or _timestamp_name()

    for suffix in range(max_attempts):
        name = stem if suffix == 0 else f"{stem}_{suffix:04d}"
        candidate = base_dir / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate.resolve()

    raise FileExistsError(
        f"Could not create a unique hpo run directory under {base_dir} "
        f"for timestamp {stem!r} after {max_attempts} attempts."
    )


def experiment_directory_for_run(
    task_cfg: DictConfig, task_path: Path, run_dir: Path
) -> str:
    """Where this run's train outputs land, relative to the logs root."""

    return f"hpo/{task_slug(task_cfg, task_path)}/{run_dir.name}"


def _set_runtime_paths(
    *,
    root: Path,
    task_cfg: DictConfig,
    run_dir: Path,
    state_dir: Path,
) -> None:
    task_cfg["state_dir"] = _relative_path_text(root, state_dir)
    task_cfg["output_root"] = _relative_path_text(root, run_dir / "evaluator")


def prepare_hpo_runtime_task(
    *,
    root: Path,
    task_path: Path,
    task_cfg: DictConfig,
    resume: str | Path | None = None,
    timestamp_name: str | None = None,
) -> HPORunContext:
    root = root.resolve()
    task_path = task_path.resolve()
    runtime_cfg = _copy_cfg(task_cfg)

    if resume is not None:
        run_dir = _resolve_root_path(root, resume)
        run_dir.mkdir(parents=True, exist_ok=True)
        state_dir = (run_dir / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        _set_runtime_paths(
            root=root,
            task_cfg=runtime_cfg,
            run_dir=run_dir,
            state_dir=state_dir,
        )
        return HPORunContext(
            task_path=task_path,
            task_cfg=runtime_cfg,
            run_dir=run_dir,
            state_dir=state_dir,
            resumed=True,
        )

    configured_state_dir = runtime_cfg.get("state_dir", None)
    if configured_state_dir not in (None, ""):
        state_dir = _resolve_root_path(root, str(configured_state_dir))
        return HPORunContext(
            task_path=task_path,
            task_cfg=runtime_cfg,
            run_dir=None,
            state_dir=state_dir,
            resumed=False,
        )

    run_dir = create_unique_hpo_run_dir(
        root=root,
        task_cfg=runtime_cfg,
        task_path=task_path,
        timestamp_name=timestamp_name,
    )
    state_dir = (run_dir / "state").resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    _set_runtime_paths(
        root=root,
        task_cfg=runtime_cfg,
        run_dir=run_dir,
        state_dir=state_dir,
    )
    return HPORunContext(
        task_path=task_path,
        task_cfg=runtime_cfg,
        run_dir=run_dir,
        state_dir=state_dir,
        resumed=False,
    )
