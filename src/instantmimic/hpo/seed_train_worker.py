#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from instantmimic.config.config import load_semantic_config
from instantmimic.hpo.common import resolve_config_path_arg
from instantmimic.hpo.common import utc_now as _utc_now
from instantmimic.utils.experiment_logging import (
    load_experiment_dotenv,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SeedTrainRequest:
    config_path: str | None
    config_name: str
    overrides: list[str]
    log_file: str
    frames_per_run: int | None = None


@dataclass(frozen=True)
class SeedTrainResponse:
    wall_seconds: float | None
    throughput_fps: float | None
    returncode: int
    started_at: str
    finished_at: str
    error: str | None = None


def _default_run_training(cfg: DictConfig) -> None:
    from instantmimic.train.runtime import run_training

    semantic = load_semantic_config(cfg)
    run_training(semantic)


def _compose_train_cfg(
    *,
    config_path: str | None,
    config_name: str,
    overrides: list[str],
) -> DictConfig:
    resolved_config_path = resolve_config_path_arg(config_path)
    if resolved_config_path is None:
        resolved_config_path = str((_REPO_ROOT / "config").resolve())
    with initialize_config_dir(config_dir=resolved_config_path, version_base=None):
        return compose(
            config_name=config_name,
            overrides=list(overrides),
            return_hydra_config=True,
        )


def _reset_hydra_config() -> None:
    HydraConfig.instance().cfg = None


def _remove_added_logging_handlers(
    before_handlers: tuple[logging.Handler, ...],
) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler in before_handlers:
            continue
        root.removeHandler(handler)
        try:
            handler.flush()
        except (OSError, ValueError):
            # Handler cleanup may race with already-closed log files after failures.
            pass
        try:
            handler.close()
        except (OSError, ValueError):
            # Best-effort cleanup must not mask the training exception being reported.
            pass


@contextlib.contextmanager
def _redirect_process_output(log_stream):
    log_stream.flush()
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    try:
        os.dup2(log_stream.fileno(), 1)
        os.dup2(log_stream.fileno(), 2)
        with (
            contextlib.redirect_stdout(log_stream),
            contextlib.redirect_stderr(log_stream),
        ):
            yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            log_stream.flush()
        finally:
            os.dup2(original_stdout_fd, 1)
            os.dup2(original_stderr_fd, 2)
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)


def _prepare_runtime_cfg(cfg: DictConfig) -> None:
    if "hydra" not in cfg:
        return
    HydraConfig.instance().set_config(cfg)
    with open_dict(cfg):
        cfg.pop("hydra", None)


def _run_seed_request(
    request: SeedTrainRequest,
    *,
    compose_train_cfg_fn: Callable[..., DictConfig] = _compose_train_cfg,
    run_training_fn: Callable[[DictConfig], None] | None = None,
) -> SeedTrainResponse:
    started_at = _utc_now()
    start = time.perf_counter()
    log_file = Path(request.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    run_training = run_training_fn or _default_run_training
    returncode = 0
    error: str | None = None

    root = logging.getLogger()
    before_handlers = tuple(root.handlers)

    with log_file.open("w", encoding="utf-8") as log_stream:
        try:
            with _redirect_process_output(log_stream):
                cfg = compose_train_cfg_fn(
                    config_path=request.config_path,
                    config_name=request.config_name,
                    overrides=list(request.overrides),
                )
                _prepare_runtime_cfg(cfg)
                run_training(cfg)
        except Exception as exc:  # noqa: BLE001 - worker boundary reports failures to parent process.
            returncode = 1
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(file=log_stream)
        finally:
            try:
                log_stream.flush()
            except (OSError, ValueError):
                # The log stream may already be closed while reporting fatal worker errors.
                pass
            _reset_hydra_config()
            _remove_added_logging_handlers(before_handlers)

    # Training agents can own CUDA replay buffers through cycles (compiled
    # bound methods, graph callbacks). Reclaim them before the next seed in
    # this long-lived process, regardless of success or failure.
    gc.collect()

    wall = time.perf_counter() - start
    throughput = None
    if request.frames_per_run is not None and wall > 0.0:
        throughput = float(request.frames_per_run) / wall
    return SeedTrainResponse(
        wall_seconds=wall,
        throughput_fps=throughput,
        returncode=returncode,
        started_at=started_at,
        finished_at=_utc_now(),
        error=error,
    )


def _request_from_payload(payload: dict[str, Any]) -> SeedTrainRequest:
    overrides = payload.get("overrides", [])
    if not isinstance(overrides, list) or not all(
        isinstance(item, str) for item in overrides
    ):
        raise ValueError("request.overrides must be a list of strings")
    frames_per_run = payload.get("frames_per_run")
    return SeedTrainRequest(
        config_path=(
            str(payload["config_path"])
            if payload.get("config_path") is not None
            else None
        ),
        config_name=str(payload["config_name"]),
        overrides=list(overrides),
        log_file=str(payload["log_file"]),
        frames_per_run=(int(frames_per_run) if frames_per_run is not None else None),
    )


def _error_response(error: str) -> SeedTrainResponse:
    now = _utc_now()
    return SeedTrainResponse(
        wall_seconds=None,
        throughput_fps=None,
        returncode=1,
        started_at=now,
        finished_at=now,
        error=error,
    )


def main() -> int:
    load_experiment_dotenv(_REPO_ROOT)
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            request = _request_from_payload(payload)
            response = _run_seed_request(request)
        except Exception as exc:  # noqa: BLE001 - protocol boundary converts malformed requests to JSON errors.
            response = _error_response(f"{type(exc).__name__}: {exc}")
        print(json.dumps(asdict(response), ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
