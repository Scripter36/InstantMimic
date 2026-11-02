from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from instantmimic.hpo.common import cfg_bool
from instantmimic.hpo.common import section_cfg as _common_section_cfg
from instantmimic.hpo.controller_state import (
    CYCLE_ARTIFACT_FILES,
    LIFECYCLE_COMPLETED,
    RUN_LEVEL_STATE_FILES,
    is_failed,
)
from instantmimic.utils.experiment_logging import (
    build_wandb_init_kwargs,
    load_experiment_dotenv,
)

_SUPPORTED_BACKENDS = {"none", "tensorboard", "wandb", "mlflow"}
_TRACKING_STATE_FILE = "tracking.yaml"


def _section_cfg(task_cfg: DictConfig, key: str) -> DictConfig | None:
    return _common_section_cfg(task_cfg, key)


def _cfg_str(section: DictConfig | None, key: str, default: str) -> str:
    if section is None:
        return default
    value = section.get(key, None)
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return (
        result
        if result == result and result not in (float("inf"), float("-inf"))
        else None
    )


def _relpath(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _load_tracking_state(state_dir: Path) -> dict[str, object]:
    path = state_dir / _TRACKING_STATE_FILE
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logging.warning("Failed to read hpo tracking state: %s", exc, exc_info=True)
        return {}
    return dict(loaded) if isinstance(loaded, dict) else {}


def _write_tracking_state(
    state_dir: Path,
    *,
    backend: str,
    run_id: str,
    config: _TrackingConfig,
    tracking_uri: str = "",
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    backend_state = {
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "run_name": config.run_name,
        "group": config.group,
    }
    if config.scope:
        backend_state["scope"] = config.scope
    if backend == "mlflow" and tracking_uri:
        backend_state["tracking_destination_sha256"] = hashlib.sha256(
            tracking_uri.encode("utf-8")
        ).hexdigest()
    payload = {
        "backend": backend,
        backend: backend_state,
    }
    (state_dir / _TRACKING_STATE_FILE).write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _mlflow_tracking_uri(mlflow_module: Any) -> str:
    getter = getattr(mlflow_module, "get_tracking_uri", None)
    if not callable(getter):
        return ""
    try:
        return str(getter() or "").strip()
    except Exception as exc:  # noqa: BLE001 - tracking URI is best-effort metadata.
        logging.debug("Failed to read MLflow tracking URI: %s", exc, exc_info=True)
        return ""


def _stored_mlflow_run_id(
    *,
    state_dir: Path,
    config: _TrackingConfig,
    tracking_uri: str,
) -> str | None:
    state = _load_tracking_state(state_dir)
    if str(state.get("backend", "") or "").strip().lower() != "mlflow":
        return None
    mlflow_state = state.get("mlflow", None)
    if not isinstance(mlflow_state, dict):
        return None
    run_id = str(mlflow_state.get("run_id", "") or "").strip()
    if not run_id:
        return None
    stored_experiment = str(mlflow_state.get("experiment_name", "") or "").strip()
    if stored_experiment and stored_experiment != config.experiment_name:
        logging.warning(
            "Ignoring stored MLflow hpo run_id for experiment %r; current experiment is %r.",
            stored_experiment,
            config.experiment_name,
        )
        return None
    stored_scope = str(mlflow_state.get("scope", "") or "").strip()
    if config.scope and stored_scope != config.scope:
        logging.warning(
            "Ignoring stored MLflow hpo run_id for scope %r; current scope is %r.",
            stored_scope,
            config.scope,
        )
        return None
    stored_destination_sha256 = str(
        mlflow_state.get("tracking_destination_sha256", "") or ""
    ).strip()
    current_destination_sha256 = (
        hashlib.sha256(tracking_uri.encode("utf-8")).hexdigest() if tracking_uri else ""
    )
    stored_tracking_uri = str(mlflow_state.get("tracking_uri", "") or "").strip()
    destination_changed = bool(
        stored_destination_sha256
        and current_destination_sha256
        and stored_destination_sha256 != current_destination_sha256
    ) or bool(
        stored_tracking_uri and tracking_uri and stored_tracking_uri != tracking_uri
    )
    if destination_changed:
        logging.warning(
            "Ignoring stored MLflow hpo run_id because the configured "
            "tracking destination changed.",
        )
        return None
    return run_id


def _stored_wandb_run_id(
    *,
    state_dir: Path,
    config: _TrackingConfig,
) -> str | None:
    state = _load_tracking_state(state_dir)
    if str(state.get("backend", "") or "").strip().lower() != "wandb":
        return None
    wandb_state = state.get("wandb", None)
    if not isinstance(wandb_state, dict):
        return None
    run_id = str(wandb_state.get("run_id", "") or "").strip()
    if not run_id:
        return None
    stored_experiment = str(wandb_state.get("experiment_name", "") or "").strip()
    if stored_experiment and stored_experiment != config.experiment_name:
        return None
    stored_scope = str(wandb_state.get("scope", "") or "").strip()
    if config.scope and stored_scope != config.scope:
        return None
    return run_id


def _mlflow_filter_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _find_existing_mlflow_run_id(
    *,
    mlflow_module: Any,
    root: Path,
    task_path: Path,
    state_dir: Path,
    config: _TrackingConfig,
) -> str | None:
    search_runs = getattr(mlflow_module, "search_runs", None)
    if not callable(search_runs):
        return None
    state_dir_tag = _mlflow_filter_literal(_relpath(root, state_dir))
    task_path_tag = _mlflow_filter_literal(_relpath(root, task_path))
    filter_string = (
        "tags.job_type = 'hpo' "
        f"AND tags.state_dir = '{state_dir_tag}' "
        f"AND tags.task_path = '{task_path_tag}'"
    )
    if config.scope:
        scope_tag = _mlflow_filter_literal(config.scope)
        filter_string += f" AND tags.scope = '{scope_tag}'"
    try:
        runs = search_runs(
            experiment_names=[config.experiment_name],
            filter_string=filter_string,
            output_format="list",
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort.
        logging.warning(
            "Failed to search for an existing MLflow hpo run: %s",
            exc,
            exc_info=True,
        )
        return None
    if not runs:
        return None
    run_info = getattr(runs[0], "info", None)
    run_id = str(getattr(run_info, "run_id", "") or "").strip()
    if run_id:
        logging.info("Resuming discovered MLflow hpo run %s.", run_id)
        return run_id
    return None


def _task_name(task_cfg: DictConfig, task_path: Path) -> str:
    raw = str(task_cfg.get("name", "") or "").strip()
    return raw if raw else task_path.stem


def _default_group(task_name: str) -> str:
    for suffix in ("_r00", "_r01", "_r02", "_r03", "-r00", "-r01", "-r02", "-r03"):
        if task_name.endswith(suffix):
            return task_name[: -len(suffix)]
    return task_name


def _default_run_name(task_name: str, state_dir: Path, group: str) -> str:
    run_instance = ""
    if state_dir.name == "state" and state_dir.parent.name:
        run_instance = state_dir.parent.name
    if not run_instance:
        return task_name
    return f"{group}/{run_instance}" if group else run_instance


@dataclass(frozen=True)
class _TrackingConfig:
    backend: str
    experiment_name: str
    run_name: str
    group: str
    scope: str
    log_artifacts: bool
    log_llm_traces: bool
    log_dry_run: bool
    required: bool


class HPOTracker:
    def __init__(
        self,
        *,
        root: Path,
        task_path: Path,
        task_cfg: DictConfig,
        state_dir: Path,
        agent_label: str,
        config: _TrackingConfig,
        mlflow_module: Any | None = None,
        wandb_module: Any | None = None,
        wandb_run: Any | None = None,
        tensorboard_writer: Any | None = None,
        run_id: str | None = None,
    ) -> None:
        self.root = root.resolve()
        self.task_path = task_path.resolve()
        self.task_cfg = task_cfg
        self.state_dir = state_dir.resolve()
        self.agent_label = agent_label
        self.config = config
        self._mlflow = mlflow_module
        self._wandb = wandb_module
        self._wandb_run = wandb_run
        self._tensorboard_writer = tensorboard_writer
        self._run_id = run_id
        self._closed = False

    @classmethod
    def from_task(
        cls,
        *,
        root: Path,
        task_path: Path,
        task_cfg: DictConfig,
        state_dir: Path,
        agent_label: str,
        dry_run: bool,
    ) -> HPOTracker:
        tracking_cfg = _section_cfg(task_cfg, "hpo_tracking")
        if tracking_cfg is None:
            return cls.disabled(root, task_path, task_cfg, state_dir, agent_label)

        load_experiment_dotenv(root)
        task_name = _task_name(task_cfg, task_path)
        group = _cfg_str(tracking_cfg, "group", _default_group(task_name))
        config = _TrackingConfig(
            backend=_cfg_str(
                tracking_cfg,
                "backend",
                os.environ.get("INSTANTMIMIC_LOGGER", "tensorboard"),
            ).lower(),
            experiment_name=_cfg_str(
                tracking_cfg, "experiment_name", "instantmimic-hpo"
            ),
            run_name=_cfg_str(
                tracking_cfg,
                "run_name",
                _default_run_name(task_name, state_dir, group),
            ),
            group=group,
            scope=_cfg_str(tracking_cfg, "scope", ""),
            log_artifacts=cfg_bool(tracking_cfg, "log_artifacts", True),
            log_llm_traces=cfg_bool(tracking_cfg, "log_llm_traces", True),
            log_dry_run=cfg_bool(tracking_cfg, "log_dry_run", False),
            required=cfg_bool(tracking_cfg, "required", False),
        )
        if config.backend not in _SUPPORTED_BACKENDS:
            logging.warning(
                "Unsupported hpo tracking backend %r; disabling tracking.",
                config.backend,
            )
            return cls.disabled(root, task_path, task_cfg, state_dir, agent_label)
        if config.backend == "none" or (dry_run and not config.log_dry_run):
            return cls.disabled(root, task_path, task_cfg, state_dir, agent_label)
        if config.backend == "mlflow":
            return cls._start_mlflow(
                root=root,
                task_path=task_path,
                task_cfg=task_cfg,
                state_dir=state_dir,
                agent_label=agent_label,
                config=config,
            )
        if config.backend == "wandb":
            return cls._start_wandb(
                root=root,
                task_path=task_path,
                task_cfg=task_cfg,
                state_dir=state_dir,
                agent_label=agent_label,
                config=config,
            )
        if config.backend == "tensorboard":
            return cls._start_tensorboard(
                root=root,
                task_path=task_path,
                task_cfg=task_cfg,
                state_dir=state_dir,
                agent_label=agent_label,
                config=config,
            )
        return cls.disabled(root, task_path, task_cfg, state_dir, agent_label)

    @classmethod
    def disabled(
        cls,
        root: Path,
        task_path: Path,
        task_cfg: DictConfig,
        state_dir: Path,
        agent_label: str,
    ) -> HPOTracker:
        return cls(
            root=root,
            task_path=task_path,
            task_cfg=task_cfg,
            state_dir=state_dir,
            agent_label=agent_label,
            config=_TrackingConfig(
                backend="none",
                experiment_name="",
                run_name="",
                group="",
                scope="",
                log_artifacts=False,
                log_llm_traces=False,
                log_dry_run=False,
                required=False,
            ),
        )

    @classmethod
    def _start_mlflow(
        cls,
        *,
        root: Path,
        task_path: Path,
        task_cfg: DictConfig,
        state_dir: Path,
        agent_label: str,
        config: _TrackingConfig,
    ) -> HPOTracker:
        try:
            load_experiment_dotenv(root)
            import mlflow

            task_name = _task_name(task_cfg, task_path)
            mlflow.set_experiment(config.experiment_name)
            tracking_uri = _mlflow_tracking_uri(mlflow)
            stored_run_id = _stored_mlflow_run_id(
                state_dir=state_dir,
                config=config,
                tracking_uri=tracking_uri,
            )
            if stored_run_id is None:
                stored_run_id = _find_existing_mlflow_run_id(
                    mlflow_module=mlflow,
                    root=root,
                    task_path=task_path,
                    state_dir=state_dir,
                    config=config,
                )
            backfill_required = stored_run_id is None
            start_tags = {
                "agent": agent_label,
                "group": config.group,
                "job_type": "hpo",
                "state_dir": _relpath(root, state_dir),
                "task_name": task_name,
                "task_path": _relpath(root, task_path),
            }
            if config.scope:
                start_tags["scope"] = config.scope
            if stored_run_id:
                try:
                    active_run = mlflow.start_run(run_id=stored_run_id)
                except Exception as exc:  # noqa: BLE001 - stale persisted run ids fall back to a fresh run.
                    logging.warning(
                        "Failed to resume MLflow hpo run %s; starting a new run: %s",
                        stored_run_id,
                        exc,
                        exc_info=True,
                    )
                    active_run = mlflow.start_run(
                        run_name=config.run_name,
                        tags=start_tags,
                    )
                    backfill_required = True
            else:
                active_run = mlflow.start_run(
                    run_name=config.run_name,
                    tags=start_tags,
                )
            run_id = active_run.info.run_id
            _write_tracking_state(
                state_dir,
                backend="mlflow",
                run_id=run_id,
                config=config,
                tracking_uri=tracking_uri,
            )
            tracker = cls(
                root=root,
                task_path=task_path,
                task_cfg=task_cfg,
                state_dir=state_dir,
                agent_label=agent_label,
                config=config,
                mlflow_module=mlflow,
                run_id=run_id,
            )
            tracker._log_run_metadata()
            if backfill_required:
                tracker._backfill_existing_state()
            return tracker
        except Exception as exc:
            if config.required:
                raise RuntimeError(
                    "Required MLflow hpo tracking could not be initialized."
                ) from exc
            logging.warning(
                "Failed to initialize MLflow hpo tracking; disabling tracking: %s",
                exc,
                exc_info=True,
            )
            return cls.disabled(root, task_path, task_cfg, state_dir, agent_label)

    @classmethod
    def _start_wandb(
        cls,
        *,
        root: Path,
        task_path: Path,
        task_cfg: DictConfig,
        state_dir: Path,
        agent_label: str,
        config: _TrackingConfig,
    ) -> HPOTracker:
        try:
            import wandb

            task_name = _task_name(task_cfg, task_path)
            stored_run_id = _stored_wandb_run_id(
                state_dir=state_dir,
                config=config,
            )
            task_payload = OmegaConf.to_container(task_cfg, resolve=True)
            init_kwargs = build_wandb_init_kwargs(
                experiment_cfg=OmegaConf.create({"type": "wandb"}),
                run_name=config.run_name,
                run_dir=state_dir.parent,
                default_project=config.experiment_name,
                sync_tensorboard=False,
                config_payload={
                    "hpo": {
                        "agent": agent_label,
                        "scope": config.scope,
                        "state_dir": _relpath(root, state_dir),
                        "task_name": task_name,
                        "task_path": _relpath(root, task_path),
                        "task": task_payload,
                    }
                },
                group=config.group,
                job_type="hpo",
            )
            tags = list(init_kwargs.get("tags", []))
            tags.extend(
                value
                for value in ("hpo", config.scope, agent_label)
                if value and value not in tags
            )
            init_kwargs["tags"] = tags
            if stored_run_id:
                init_kwargs["id"] = stored_run_id
                init_kwargs["resume"] = "allow"
            wandb_run = wandb.init(**init_kwargs)
            run_id = str(getattr(wandb_run, "id", stored_run_id or "") or "").strip()
            if not run_id:
                raise RuntimeError("W&B did not return a run id.")
            _write_tracking_state(
                state_dir,
                backend="wandb",
                run_id=run_id,
                config=config,
            )
            tracker = cls(
                root=root,
                task_path=task_path,
                task_cfg=task_cfg,
                state_dir=state_dir,
                agent_label=agent_label,
                config=config,
                wandb_module=wandb,
                wandb_run=wandb_run,
                run_id=run_id,
            )
            tracker._log_run_metadata()
            if stored_run_id is None:
                tracker._backfill_existing_state()
            return tracker
        except Exception as exc:
            if config.required:
                raise RuntimeError(
                    "Required W&B hpo tracking could not be initialized."
                ) from exc
            logging.warning(
                "Failed to initialize W&B hpo tracking; disabling tracking: %s",
                exc,
                exc_info=True,
            )
            return cls.disabled(root, task_path, task_cfg, state_dir, agent_label)

    @classmethod
    def _start_tensorboard(
        cls,
        *,
        root: Path,
        task_path: Path,
        task_cfg: DictConfig,
        state_dir: Path,
        agent_label: str,
        config: _TrackingConfig,
    ) -> HPOTracker:
        try:
            from torch.utils.tensorboard import SummaryWriter

            stored_state = _load_tracking_state(state_dir)
            backfill_required = stored_state.get("backend") != "tensorboard"
            log_dir = state_dir / "tensorboard"
            writer = SummaryWriter(log_dir=log_dir.as_posix())
            _write_tracking_state(
                state_dir,
                backend="tensorboard",
                run_id="local",
                config=config,
            )
            tracker = cls(
                root=root,
                task_path=task_path,
                task_cfg=task_cfg,
                state_dir=state_dir,
                agent_label=agent_label,
                config=config,
                tensorboard_writer=writer,
                run_id="local",
            )
            if backfill_required:
                tracker._backfill_existing_state()
            return tracker
        except Exception as exc:
            if config.required:
                raise RuntimeError(
                    "Required TensorBoard hpo tracking could not be initialized."
                ) from exc
            logging.warning(
                "Failed to initialize TensorBoard hpo tracking; disabling tracking: %s",
                exc,
                exc_info=True,
            )
            return cls.disabled(root, task_path, task_cfg, state_dir, agent_label)

    @property
    def enabled(self) -> bool:
        return bool(
            (self.config.backend == "mlflow" and self._mlflow is not None)
            or (self.config.backend == "wandb" and self._wandb_run is not None)
            or (
                self.config.backend == "tensorboard"
                and self._tensorboard_writer is not None
            )
        )

    def _log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        if not metrics:
            return
        payload = dict(metrics)
        if self.config.backend == "mlflow":
            self._mlflow.log_metrics(payload, step=step, run_id=self._run_id)
        elif self.config.backend == "wandb":
            self._wandb_run.log(payload, step=step, commit=True)
        elif self.config.backend == "tensorboard":
            for name, value in payload.items():
                self._tensorboard_writer.add_scalar(name, value, step)

    def _write_cycle_outcome(
        self,
        *,
        cycle_index: int,
        metric_objective: str,
        outcome: Mapping[str, object],
        overwrite: bool,
    ) -> Path:
        cycle_dir = self.state_dir / "cycles" / f"{cycle_index:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        outcome_path = cycle_dir / "outcome.json"
        if outcome_path.exists() and not overwrite:
            return outcome_path
        outcome_path.write_text(
            json.dumps(
                {
                    "cycle": int(cycle_index),
                    "metric_objective": metric_objective,
                    "outcome": dict(outcome),
                },
                indent=2,
                sort_keys=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return outcome_path

    def _log_cycle_outcome(
        self,
        *,
        cycle_index: int,
        metric_objective: str,
        outcome: Mapping[str, object],
        overwrite_local: bool = True,
    ) -> None:
        outcome_path = self._write_cycle_outcome(
            cycle_index=cycle_index,
            metric_objective=metric_objective,
            outcome=outcome,
            overwrite=overwrite_local,
        )
        if self.config.backend == "mlflow":
            self._mlflow.log_dict(
                {
                    "cycle": int(cycle_index),
                    "metric_objective": metric_objective,
                    "outcome": dict(outcome),
                },
                f"hpo/cycles/{cycle_index:03d}/outcome.json",
                run_id=self._run_id,
            )
            return
        if self.config.backend == "wandb" and self.config.log_artifacts:
            self._log_artifact_if_present(
                outcome_path,
                f"hpo/cycles/{cycle_index:03d}",
            )

    def _log_run_metadata(self) -> None:
        if not self.enabled:
            return
        payload = {
            "agent": self.agent_label,
            "scope": self.config.scope,
            "state_dir": _relpath(self.root, self.state_dir),
            "task_path": _relpath(self.root, self.task_path),
            "task_sha256": _file_sha256(self.task_path),
            "task": OmegaConf.to_container(self.task_cfg, resolve=True),
        }
        try:
            if self.config.backend == "mlflow":
                self._mlflow.log_dict(
                    payload,
                    "hpo/task.json",
                    run_id=self._run_id,
                )
            if self.config.log_artifacts and self.task_path.exists():
                self._log_artifact_if_present(
                    self.task_path,
                    "hpo/task",
                )
        except Exception as exc:  # noqa: BLE001 - tracking must not break hpo.
            logging.warning("Failed to log hpo metadata: %s", exc, exc_info=True)

    def _backfill_existing_state(self) -> None:
        results_path = self.state_dir / "results.tsv"
        if not self.enabled or not results_path.exists():
            return
        try:
            csv.field_size_limit(sys.maxsize)
            with results_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            cycles_since_improvement: int | None = None
            for row in rows:
                try:
                    cycle_index = int(str(row.get("cycle", "") or "").strip())
                except ValueError:
                    continue
                # Per-strategy numbers live in one JSON column; the lifecycle
                # columns sit beside it.
                outcome: dict[str, object] = {}
                metrics_json = str(row.get("metrics_json", "") or "").strip()
                if metrics_json:
                    try:
                        loaded = json.loads(metrics_json)
                    except json.JSONDecodeError:
                        loaded = None
                    if isinstance(loaded, dict):
                        outcome.update(loaded)
                outcome.update(
                    {
                        key: value
                        for key, value in row.items()
                        if value not in (None, "") and not key.endswith("_json")
                    }
                )
                if "objective_value" not in outcome:
                    outcome["objective_value"] = row.get("objective_value", "")

                # `cycles_since_improvement` is the controller's counter; the
                # backfill reconstructs it from the recorded lifecycle alone.
                lifecycle = str(outcome.get("status", "")).strip().lower()
                if lifecycle == LIFECYCLE_COMPLETED:
                    cycles_since_improvement = 0
                elif cycles_since_improvement is not None:
                    cycles_since_improvement += 1
                metrics = self._build_cycle_metrics(
                    outcome=outcome,
                    cycles_since_improvement=cycles_since_improvement,
                )
                self._log_metrics(metrics, step=cycle_index)
                self._log_cycle_outcome(
                    cycle_index=cycle_index,
                    metric_objective="",
                    outcome=outcome,
                    overwrite_local=False,
                )
                if self.config.log_artifacts:
                    artifact_path = f"hpo/cycles/{cycle_index:03d}"
                    cycle_dir = self.state_dir / "cycles" / f"{cycle_index:03d}"
                    for file_name in CYCLE_ARTIFACT_FILES:
                        self._log_artifact_if_present(
                            cycle_dir / file_name,
                            artifact_path,
                        )
                    summary_path = self._resolve_recorded_artifact_path(
                        row.get("evidence_path", "")
                    )
                    if summary_path is not None:
                        self._log_artifact_if_present(summary_path, artifact_path)
                        for sibling_name in ("results.csv", "trials.jsonl"):
                            self._log_artifact_if_present(
                                summary_path.parent / sibling_name,
                                artifact_path,
                            )
            if self.config.log_artifacts:
                self._log_state_artifacts()
        except Exception as exc:  # noqa: BLE001 - tracking must not break hpo.
            logging.warning(
                "Failed to backfill hpo tracking state: %s",
                exc,
                exc_info=True,
            )

    def _resolve_recorded_artifact_path(self, value: object) -> Path | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        allowed_roots = (self.root, self.state_dir.parent)
        if not any(candidate.is_relative_to(base.resolve()) for base in allowed_roots):
            logging.warning(
                "Ignoring hpo artifact path outside the run or repository: %s",
                raw,
            )
            return None
        return candidate

    @contextlib.contextmanager
    def phase(
        self,
        *,
        cycle_index: int,
        phase_name: str,
        attributes: Mapping[str, object] | None = None,
    ) -> Iterator[object | None]:
        if not self.enabled or not self.config.log_llm_traces:
            yield None
            return
        if self.config.backend == "wandb":
            started_at = time.time()
            started_perf = time.perf_counter()
            try:
                yield None
            except Exception as exc:
                self._log_wandb_phase_trace(
                    cycle_index=cycle_index,
                    phase_name=phase_name,
                    attributes=attributes,
                    started_at=started_at,
                    duration_seconds=time.perf_counter() - started_perf,
                    error=repr(exc),
                )
                raise
            else:
                self._log_wandb_phase_trace(
                    cycle_index=cycle_index,
                    phase_name=phase_name,
                    attributes=attributes,
                    started_at=started_at,
                    duration_seconds=time.perf_counter() - started_perf,
                    error=None,
                )
            return
        if self.config.backend != "mlflow":
            yield None
            return
        span_attributes = {
            "hpo.cycle": int(cycle_index),
            "hpo.phase": phase_name,
        }
        if attributes:
            span_attributes.update(dict(attributes))
        try:
            span_context = self._mlflow.start_span(
                f"hpo.{phase_name}",
                span_type="CHAIN",
                attributes=span_attributes,
                run_id=self._run_id,
            )
        except Exception as exc:  # noqa: BLE001 - tracing must not break hpo.
            logging.warning("Failed to start MLflow span: %s", exc, exc_info=True)
            yield None
            return

        with span_context as span:
            try:
                yield span
            except Exception as exc:
                try:
                    span.record_exception(exc)
                except Exception as log_exc:  # noqa: BLE001 - tracing must not mask the original error.
                    logging.warning(
                        "Failed to record MLflow span exception: %s",
                        log_exc,
                        exc_info=True,
                    )
                raise
            else:
                try:
                    span.set_outputs({"status": "ok"})
                except Exception as exc:  # noqa: BLE001 - tracing must not break hpo.
                    logging.warning(
                        "Failed to record MLflow span outputs: %s",
                        exc,
                        exc_info=True,
                    )

    def _log_wandb_phase_trace(
        self,
        *,
        cycle_index: int,
        phase_name: str,
        attributes: Mapping[str, object] | None,
        started_at: float,
        duration_seconds: float,
        error: str | None,
    ) -> None:
        try:
            trace_dir = self.state_dir / "cycles" / f"{cycle_index:03d}" / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{phase_name}.json"
            trace_path.write_text(
                json.dumps(
                    {
                        "cycle": int(cycle_index),
                        "phase": phase_name,
                        "status": "error" if error else "ok",
                        "started_at_unix": started_at,
                        "duration_seconds": duration_seconds,
                        "attributes": dict(attributes or {}),
                        "error": error,
                    },
                    indent=2,
                    sort_keys=False,
                )
                + "\n",
                encoding="utf-8",
            )
            artifact = self._wandb.Artifact(
                f"hpo-trace-{cycle_index:03d}-{phase_name}",
                type="hpo-trace",
            )
            artifact.add_file(trace_path.as_posix(), name=trace_path.name)
            for key, value in dict(attributes or {}).items():
                if not str(key).endswith("_path"):
                    continue
                related_path = self._resolve_recorded_artifact_path(value)
                if (
                    related_path is not None
                    and related_path.exists()
                    and related_path.is_file()
                ):
                    artifact.add_file(
                        related_path.as_posix(),
                        name=related_path.name,
                    )
            self._wandb_run.log_artifact(artifact)
        except Exception as exc:  # noqa: BLE001 - tracing must not break hpo.
            logging.warning("Failed to log W&B hpo trace: %s", exc, exc_info=True)

    def log_cycle_result(
        self,
        *,
        cycle_index: int,
        metric_objective: str,
        outcome: Mapping[str, object],
        summary_path: Path | None,
        trials_path: Path | None,
        evaluator_log_path: Path | None,
        planning_message_path: Path | None,
        prepare_message_path: Path | None,
        final_message_path: Path | None,
        validation_log_path: Path | None,
        cycles_since_improvement: int | None,
    ) -> None:
        if not self.enabled:
            return
        try:
            metrics = self._build_cycle_metrics(
                outcome=outcome,
                cycles_since_improvement=cycles_since_improvement,
            )
            self._log_metrics(metrics, step=int(cycle_index))
            if self.config.log_artifacts:
                self._log_cycle_artifacts(
                    cycle_index=cycle_index,
                    summary_path=summary_path,
                    trials_path=trials_path,
                    evaluator_log_path=evaluator_log_path,
                    planning_message_path=planning_message_path,
                    prepare_message_path=prepare_message_path,
                    final_message_path=final_message_path,
                    validation_log_path=validation_log_path,
                )
                self._log_state_artifacts()
            self._log_cycle_outcome(
                cycle_index=cycle_index,
                metric_objective=metric_objective,
                outcome=outcome,
            )
        except Exception as exc:  # noqa: BLE001 - tracking must not break hpo.
            logging.warning(
                "Failed to log hpo cycle %s: %s",
                cycle_index,
                exc,
                exc_info=True,
            )

    def _build_cycle_metrics(
        self,
        *,
        outcome: Mapping[str, object],
        cycles_since_improvement: int | None,
    ) -> dict[str, float]:
        """Lifecycle one-hots, plus whatever the comparator asked to log.

        The lifecycle is common to both objective modes and is the only thing
        this module interprets. Every objective-specific number arrives already
        named by `comparator.tracking_metrics()`, so adding a strategy never
        means editing this function.
        """

        metrics: dict[str, float] = {}
        lifecycle = str(outcome.get("status", "") or "").strip().lower()
        completed = lifecycle == LIFECYCLE_COMPLETED
        metrics["hpo.status_completed"] = 1.0 if completed else 0.0
        metrics["hpo.status_failed"] = 1.0 if is_failed(lifecycle) else 0.0
        if cycles_since_improvement is not None:
            metrics["hpo.cycles_since_improvement"] = float(cycles_since_improvement)
        for key, value in outcome.items():
            if key in {"status", "decision"}:
                continue
            parsed = _finite_float(value)
            if parsed is not None:
                metrics[f"hpo.{key}"] = parsed
        return metrics

    def _log_cycle_artifacts(
        self,
        *,
        cycle_index: int,
        summary_path: Path | None,
        trials_path: Path | None,
        evaluator_log_path: Path | None,
        planning_message_path: Path | None,
        prepare_message_path: Path | None,
        final_message_path: Path | None,
        validation_log_path: Path | None,
    ) -> None:
        artifact_path = f"hpo/cycles/{cycle_index:03d}"
        for path in (
            summary_path,
            trials_path,
            evaluator_log_path,
            planning_message_path,
            prepare_message_path,
            final_message_path,
            validation_log_path,
        ):
            self._log_artifact_if_present(path, artifact_path)

    def _log_state_artifacts(self) -> None:
        """Upload the run-level state files, comparator artifacts included.

        The comparator names its own files, so its directory is taken whole
        rather than by a list this module would have to keep in step.
        """

        for pattern in RUN_LEVEL_STATE_FILES:
            for path in sorted(self.state_dir.glob(pattern)):
                self._log_artifact_if_present(path, "hpo/state")

    def _log_artifact_if_present(self, path: Path | None, artifact_path: str) -> None:
        if path is None:
            return
        candidate = path.resolve()
        if not candidate.exists() or not candidate.is_file():
            return
        if self.config.backend == "mlflow":
            self._mlflow.log_artifact(
                candidate.as_posix(),
                artifact_path=artifact_path,
                run_id=self._run_id,
            )
        elif self.config.backend == "wandb":
            artifact_name = "-".join(
                part.replace(".", "-")
                for part in (*Path(artifact_path).parts, candidate.name)
            )
            artifact = self._wandb.Artifact(
                artifact_name,
                type="hpo",
            )
            artifact.add_file(candidate.as_posix(), name=candidate.name)
            self._wandb_run.log_artifact(artifact)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._mlflow is not None:
                self._mlflow.end_run()
            elif self._wandb_run is not None:
                self._wandb_run.finish()
            elif self._tensorboard_writer is not None:
                self._tensorboard_writer.close()
        except Exception as exc:  # noqa: BLE001 - close should not mask the hpo outcome.
            logging.warning("Failed to close hpo tracking run: %s", exc, exc_info=True)
