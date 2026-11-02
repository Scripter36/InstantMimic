"""HPO batch evaluator runtime."""

# Batch evaluator
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from instantmimic.hpo import evidence, run_discovery
from instantmimic.hpo.candidate_executor import (
    SeedTrainWorker,
    build_candidate_runtime_env,
    candidate_cache_root,
    hydra_overrides_from_command,
    terminate_process_group,
)
from instantmimic.hpo.common import (
    default_play_overrides as _common_default_play_overrides,
)
from instantmimic.hpo.common import (
    ensure_list as _common_ensure_list,
)
from instantmimic.hpo.common import (
    parse_keyval_payload as _common_parse_keyval_payload,
)
from instantmimic.hpo.common import (
    parse_wandb_run_dir as _common_parse_wandb_run_dir,
)
from instantmimic.hpo.common import (
    resolve_repo_path as _common_resolve_repo_path,
)
from instantmimic.hpo.common import (
    safe_float as _common_safe_float,
)
from instantmimic.hpo.common import (
    safe_int as _common_safe_int,
)
from instantmimic.hpo.common import (
    slugify as _common_slugify,
)
from instantmimic.hpo.evaluator_launch import (
    DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
    owned_process_group_termination_grace_seconds,
    process_group_manifest_path,
    register_process_group,
    unregister_process_group,
)
from instantmimic.hpo.experiment import OverrideSet

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_CONFIG_ROOT = REPO_ROOT / "config"
PLAYBACK_RESULT_RE = re.compile(r"PLAYBACK_RESULT(?:_JSON)?\s+(.*)")
BENCHMARK_EVAL_RE = re.compile(r"BENCHMARK_EVAL\s+(.*)")
KEYVAL_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]*)")
TRAINING_COMPLETE_RE = re.compile(
    r"TRAINING_COMPLETE.*best_mean_reward=([0-9eE+\-.]+|None)"
)
WANDB_RUN_DIR_RE = re.compile(
    r"wandb: Run data is saved locally in (.*?/wandb/run-[^\s]+)"
)
TRACKING_METRICS_SUMMARY_RE = re.compile(r"TRACKING_METRICS_SUMMARY\s+(.*)")
EPISODE_FINISHED_RE = re.compile(
    r"Episode\s+(\d+)\s+finished:\s+length=(\d+)\s+total_reward=([-+0-9.eE]+)"
)


@dataclass(frozen=True)
class MotionDefaults:
    motion_name: str
    config_path: Path
    config_name: str


@dataclass(frozen=True)
class MotionRunSpec:
    motion_name: str
    config_path: Path
    config_name: str
    overrides: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    rationale: str
    motions: dict[str, MotionRunSpec]
    # The node this candidate branched from, recorded so the search tree can
    # be reconstructed from evidence alone.
    parent: Any = "root"


@dataclass(frozen=True)
class ConfigSpec:
    config_name: str
    config_path: Path
    config_file: Path
    config_label: str
    config_dir: str
    config_prefix: str
    config_slug: str
    motion: str


@dataclass
class Job:
    experiment_name: str
    motion_name: str
    config: ConfigSpec
    seed: int
    group: str
    experiment_directory: str
    train_log: Path
    train_overrides: tuple[str, ...]
    parent: Any = None
    process: subprocess.Popen[str] | None = None
    process_group_id: int | None = None
    started_at: float | None = None
    wall_seconds: float | None = None
    known_run_dirs: set[Path] = field(default_factory=set)
    run_dir: Path | None = None
    best_checkpoint: Path | None = None
    train_returncode: int | None = None
    train_best_mean_reward: float | None = None
    play_log: Path | None = None
    play_returncode: int | None = None
    video_path: Path | None = None
    episode_length: int | None = None
    episode_reward: float | None = None
    ape_mean: float | None = None
    ee_mean: float | None = None
    vel_mean: float | None = None
    accel_mean: float | None = None
    episode_finish_count: int = 0
    fall_count: int = 0
    terminated_count: int = 0
    truncated_count: int = 0
    playback_result: dict[str, str] = field(default_factory=dict)
    benchmark_eval_count: int = 0
    benchmark_reward_mean_last: float | None = None
    benchmark_reward_mean_recent: float | None = None
    benchmark_episode_length_mean_last: float | None = None
    benchmark_episode_length_mean_recent: float | None = None
    benchmark_success_rate_last: float | None = None
    benchmark_success_rate_recent: float | None = None
    benchmark_ape_mean_last: float | None = None
    benchmark_ape_mean_recent: float | None = None
    benchmark_ee_mean_last: float | None = None
    benchmark_ee_mean_recent: float | None = None
    benchmark_failure_count_last: int | None = None
    benchmark_threshold_frames: dict[str, int] = field(default_factory=dict)
    benchmark_curve: list[dict[str, float]] = field(default_factory=list)
    train_retry_count: int = 0
    train_command: list[str] = field(default_factory=list)
    play_command: list[str] = field(default_factory=list)
    train_evidence: evidence.TrainEvidence = field(
        default_factory=evidence.TrainEvidence
    )
    playback_evidence: evidence.PlaybackEvidence | None = None


def _ensure_list(value: Any, field_name: str) -> list[Any]:
    return _common_ensure_list(value, field_name)


def _resolve_repo_path(path: str | Path) -> Path:
    return _common_resolve_repo_path(path, REPO_ROOT)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file '{path}' must contain a top-level mapping.")
    return payload


def _load_task(task_path: Path) -> DictConfig:
    cfg = OmegaConf.load(task_path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Task file '{task_path}' must contain a top-level mapping.")
    return cfg


def _slugify(text: str) -> str:
    return _common_slugify(text, empty_fallback="item")


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _safe_float(value: str | None) -> float | None:
    return _common_safe_float(value)


def _safe_int(value: str | None) -> int | None:
    return _common_safe_int(value)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _threshold_key(threshold: float) -> str:
    numeric = float(threshold)
    if numeric.is_integer():
        return str(int(numeric))
    return ("%g" % numeric).replace(".", "p")


@dataclass(frozen=True)
class ExecutionPolicy:
    """How trials run, independent of how their results are compared.

    Scalar hpo retried any non-zero return code and enforced a
    per-trial timeout; Pareto retried only on two recognized log signatures and
    had no timeout at all. Both are now the same configurable policy.
    """

    retries_per_seed: int = 1
    retry_triggers: tuple[str, ...] = ("cublas", "worker_eof")
    trial_timeout_seconds: float | None = None
    stop_on_fail: bool = False
    max_parallel: int = 1

    def retries_on(self, trigger: str) -> bool:
        return "any_failure" in self.retry_triggers or trigger in self.retry_triggers


def _execution_policy(batch_cfg: DictConfig) -> ExecutionPolicy:
    triggers_raw = batch_cfg.get("retry_triggers", None)
    triggers = (
        ("cublas", "worker_eof")
        if triggers_raw is None
        else tuple(str(item).strip() for item in triggers_raw if str(item).strip())
    )
    timeout_raw = batch_cfg.get("trial_timeout_seconds", None)
    return ExecutionPolicy(
        retries_per_seed=max(0, int(batch_cfg.get("retries_per_seed", 1))),
        retry_triggers=triggers,
        trial_timeout_seconds=(
            None if timeout_raw in (None, "") else float(timeout_raw)
        ),
        stop_on_fail=bool(batch_cfg.get("stop_on_fail", False)),
        max_parallel=_batch_max_parallel(batch_cfg),
    )


def _train_failure_trigger(job: Job) -> str:
    """Name the retryable signature a failed trial left behind."""

    if _train_log_has_cublas_error(job.train_log):
        return "cublas"
    if _train_log_has_worker_failure(job.train_log):
        return "worker_eof"
    return "any_failure"


# Retry policy
def _default_torchinductor_cache_dir() -> Path:
    return Path(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/instantmimic-torchinductor")
    ).resolve()


def _train_log_has_cublas_error(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return (
        "cublas_status" in text
        or "cublascreate(handle)" in text
        or "cublaslt" in text
        or "cublas error" in text
    )


def _purge_torchinductor_cache_dir(cache_dir: Path | None = None) -> Path:
    target = (cache_dir or _default_torchinductor_cache_dir()).resolve()
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
    elif target.exists():
        target.unlink(missing_ok=True)
    return target


def _archive_failed_train_log(
    path: Path,
    *,
    retry_index: int,
    reason: str = "cublas",
) -> Path | None:
    if not path.exists():
        return None
    reason_slug = _slugify(reason)
    archived = path.with_name(
        f"{path.stem}.{reason_slug}_retry{retry_index}{path.suffix}"
    )
    suffix_index = 1
    while archived.exists():
        archived = path.with_name(
            f"{path.stem}.{reason_slug}_retry{retry_index}_{suffix_index}{path.suffix}"
        )
        suffix_index += 1
    path.rename(archived)
    return archived


def _train_log_has_worker_failure(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    return "[candidate-executor] worker failed:" in text


def _reset_job_after_train_retry(job: Job) -> None:
    job.process = None
    job.run_dir = None
    job.best_checkpoint = None
    job.train_returncode = None
    job.train_best_mean_reward = None
    job.play_log = None
    job.play_returncode = None
    job.video_path = None
    job.episode_length = None
    job.episode_reward = None
    job.ape_mean = None
    job.ee_mean = None
    job.vel_mean = None
    job.accel_mean = None
    job.episode_finish_count = 0
    job.fall_count = 0
    job.terminated_count = 0
    job.truncated_count = 0
    job.playback_result = {}
    job.benchmark_eval_count = 0
    job.benchmark_reward_mean_last = None
    job.benchmark_reward_mean_recent = None
    job.benchmark_episode_length_mean_last = None
    job.benchmark_episode_length_mean_recent = None
    job.benchmark_success_rate_last = None
    job.benchmark_success_rate_recent = None
    job.benchmark_ape_mean_last = None
    job.benchmark_ape_mean_recent = None
    job.benchmark_ee_mean_last = None
    job.benchmark_ee_mean_recent = None
    job.benchmark_failure_count_last = None
    job.benchmark_threshold_frames = {}
    job.wall_seconds = None
    job.train_evidence = evidence.TrainEvidence()
    job.playback_evidence = None


def _maybe_retry_train(
    job: Job,
    *,
    fixed_train_overrides: list[str],
    dry_run: bool,
    policy: ExecutionPolicy = ExecutionPolicy(),
    cache_dir: Path | None = None,
    launch_job: Any | None = None,
) -> bool:
    """Relaunch a failed trial when the policy says its failure is retryable."""

    if int(job.train_returncode or 0) == 0:
        return False
    if job.train_retry_count >= policy.retries_per_seed:
        return False
    trigger = _train_failure_trigger(job)
    if not policy.retries_on(trigger):
        return False

    next_retry_index = job.train_retry_count + 1
    # A CUBLAS failure poisons the compiled-kernel cache; the other triggers do
    # not, so only that one pays for a purge.
    purged_dir: Path | None = None
    if trigger == "cublas":
        if cache_dir is None:
            cache_dir = candidate_cache_root(job.train_log.stem) / "torchinductor"
        purged_dir = _purge_torchinductor_cache_dir(cache_dir)
    archived_log = _archive_failed_train_log(
        job.train_log, retry_index=next_retry_index, reason=trigger
    )
    job.train_retry_count = next_retry_index
    _reset_job_after_train_retry(job)
    print(
        f"[hpo:evaluator] detected {trigger} failure for "
        f"experiment={job.experiment_name} motion={job.motion_name} seed={job.seed}; "
        + (f"purged torchinductor cache at {purged_dir} and " if purged_dir else "")
        + "retrying train "
        f"(attempt {job.train_retry_count}/{policy.retries_per_seed})."
    )
    if archived_log is not None:
        print(f"  archived failed train log to {archived_log}")
    if launch_job is None:
        launch_job = _launch_job
    launch_job(job, fixed_train_overrides, dry_run=dry_run)
    return dry_run or job.process is not None


# Configuration resolution
def _load_motion_defaults(task_cfg: DictConfig) -> dict[str, MotionDefaults]:
    batch_cfg = task_cfg.get("batch", None)
    if batch_cfg is None:
        raise ValueError("Task is missing required 'batch' section.")
    motions_cfg = batch_cfg.get("motions", None)
    if not isinstance(motions_cfg, (dict, DictConfig)):
        raise ValueError("batch.motions must be a mapping.")

    resolved: dict[str, MotionDefaults] = {}
    for motion_name, item in motions_cfg.items():
        if not isinstance(item, (dict, DictConfig)):
            raise ValueError(f"batch.motions.{motion_name} must be a mapping.")
        config_path = item.get("config_path", None)
        config_name = item.get("config_name", None)
        if not config_path or not config_name:
            raise ValueError(
                f"batch.motions.{motion_name} requires config_path and config_name."
            )
        resolved[str(motion_name)] = MotionDefaults(
            motion_name=str(motion_name),
            config_path=_resolve_repo_path(str(config_path)),
            config_name=str(config_name),
        )
    if not resolved:
        raise ValueError("batch.motions must not be empty.")
    return resolved


def _validate_motion_keys(
    experiment_name: str,
    provided: dict[str, Any],
    required: dict[str, MotionDefaults],
) -> None:
    provided_keys = set(provided.keys())
    required_keys = set(required.keys())
    missing = sorted(required_keys - provided_keys)
    extras = sorted(provided_keys - required_keys)
    if missing:
        raise ValueError(
            f"Experiment '{experiment_name}' is missing required motions: {', '.join(missing)}"
        )
    if extras:
        raise ValueError(
            f"Experiment '{experiment_name}' has unknown motions: {', '.join(extras)}"
        )


def _forbidden_train_override_prefixes(task_cfg: DictConfig) -> tuple[str, ...]:
    raw = _ensure_list(
        task_cfg.get("forbidden_prefixes"),
        "forbidden_prefixes",
    )
    return tuple(str(item).strip() for item in raw if str(item).strip())


def allowed_train_override_paths(task_cfg: DictConfig) -> tuple[str, ...] | None:
    raw = task_cfg.get("parameter_space", None)
    if raw is None:
        return None
    if not isinstance(raw, (list, ListConfig)):
        raise ValueError(
            "parameter_space must be a YAML list of exact Hydra parameter paths."
        )
    paths = tuple(str(item).strip() for item in raw)
    if not paths or any(not path for path in paths):
        raise ValueError("parameter_space must contain non-empty Hydra paths.")
    if len(set(paths)) != len(paths):
        raise ValueError("parameter_space contains duplicate Hydra paths.")
    if any("=" in path or path.startswith(("+", "~")) for path in paths):
        raise ValueError(
            "parameter_space entries must be paths, not Hydra override expressions."
        )
    return paths


def _train_override_path(override: str) -> str:
    normalized = override.strip()
    while normalized.startswith("+"):
        normalized = normalized[1:]
    normalized = normalized.removeprefix("~")
    path = normalized.split("=", 1)[0].strip()
    if not path:
        raise ValueError(f"Invalid empty train override path in '{override}'.")
    return path


def _validate_train_overrides(
    *,
    task_cfg: DictConfig,
    experiment_name: str,
    motion_name: str,
    overrides: tuple[str, ...],
) -> None:
    allowed_paths = allowed_train_override_paths(task_cfg)
    forbidden_prefixes = _forbidden_train_override_prefixes(task_cfg)
    for override in overrides:
        path = _train_override_path(override)
        for prefix in forbidden_prefixes:
            if path == prefix or path.startswith(f"{prefix}."):
                raise ValueError(
                    f"Experiment '{experiment_name}' motion '{motion_name}' uses forbidden override '{override}'. "
                    f"Forbidden prefix: '{prefix}'."
                )
        if allowed_paths is not None and path not in allowed_paths:
            raise ValueError(
                f"Experiment '{experiment_name}' motion '{motion_name}' uses "
                f"override '{override}', whose path '{path}' is not in the exact "
                "parameter_space list."
            )


def _motion_configs_locked(task_cfg: DictConfig) -> bool:
    batch_cfg = task_cfg.get("batch", None)
    if batch_cfg is None:
        return False
    return bool(batch_cfg.get("lock_motion_configs", False))


def load_proposal_experiments(
    task_cfg: DictConfig,
    proposal_path: Path,
) -> list[ExperimentSpec]:
    payload = _load_yaml(proposal_path)
    experiments_raw = payload.get("experiments", None)
    if not isinstance(experiments_raw, list) or not experiments_raw:
        raise ValueError("Proposal must contain a non-empty 'experiments' list.")

    batch_cfg = task_cfg.batch
    max_experiments = batch_cfg.get("max_experiments_per_cycle", None)
    if max_experiments not in (None, "") and len(experiments_raw) > int(
        max_experiments
    ):
        raise ValueError(
            f"Proposal has {len(experiments_raw)} experiments but task allows at most {int(max_experiments)}."
        )
    min_experiments = batch_cfg.get("min_experiments_per_cycle", None)
    if min_experiments not in (None, "") and len(experiments_raw) < int(
        min_experiments
    ):
        raise ValueError(
            f"Proposal has {len(experiments_raw)} experiments but task requires at least {int(min_experiments)}."
        )

    defaults = _load_motion_defaults(task_cfg)
    seen_names: set[str] = set()
    experiments: list[ExperimentSpec] = []
    for item in experiments_raw:
        if not isinstance(item, dict):
            raise ValueError("Each proposal experiment must be a mapping.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Each proposal experiment requires a non-empty name.")
        if name in seen_names:
            raise ValueError(f"Duplicate experiment name '{name}' in proposal.")
        seen_names.add(name)

        motions_raw = item.get("motions", None)
        if not isinstance(motions_raw, dict):
            raise ValueError(f"Experiment '{name}' must contain a motions mapping.")
        _validate_motion_keys(name, motions_raw, defaults)

        motions: dict[str, MotionRunSpec] = {}
        for motion_name, motion_payload in motions_raw.items():
            default = defaults[motion_name]
            if not isinstance(motion_payload, dict):
                raise ValueError(
                    f"Experiment '{name}' motion '{motion_name}' must be a mapping."
                )
            config_path_raw = motion_payload.get("config_path", default.config_path)
            config_name_raw = motion_payload.get("config_name", default.config_name)
            config_path = _resolve_repo_path(str(config_path_raw))
            config_name = str(config_name_raw)
            if _motion_configs_locked(task_cfg) and (
                config_path != default.config_path or config_name != default.config_name
            ):
                raise ValueError(
                    f"Experiment '{name}' motion '{motion_name}' changes locked config "
                    f"from '{default.config_path}/{default.config_name}' to "
                    f"'{config_path}/{config_name}'."
                )
            overrides = tuple(
                str(item)
                for item in _ensure_list(
                    motion_payload.get("overrides"),
                    f"experiments[{name}].motions[{motion_name}].overrides",
                )
            )
            _validate_train_overrides(
                task_cfg=task_cfg,
                experiment_name=name,
                motion_name=motion_name,
                overrides=overrides,
            )
            effective = item.get("effective_overrides")
            if effective is not None:
                if not isinstance(effective, dict) or set(effective) != set(defaults):
                    raise ValueError(
                        f"Experiment '{name}' requires effective overrides for every motion."
                    )
                overrides = tuple(
                    str(value)
                    for value in _ensure_list(
                        effective[motion_name],
                        f"experiments[{name}].effective_overrides.{motion_name}",
                    )
                )
                _validate_train_overrides(
                    task_cfg=task_cfg,
                    experiment_name=name,
                    motion_name=motion_name,
                    overrides=overrides,
                )
            motions[motion_name] = MotionRunSpec(
                motion_name=motion_name,
                config_path=config_path,
                config_name=config_name,
                overrides=overrides,
            )

        experiments.append(
            ExperimentSpec(
                name=name,
                rationale=str(item.get("rationale", "")).strip(),
                motions=motions,
                parent=item.get("parent", payload.get("base", "root")),
            )
        )
    return experiments


def _resolve_config_spec(config_path: Path, config_name: str) -> ConfigSpec:
    path = _resolve_repo_path(config_path)
    config_file = (path / f"{config_name}.yaml").resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Could not find config file: {config_file}")
    if config_file.is_relative_to(REPO_CONFIG_ROOT):
        config_label = (
            config_file.relative_to(REPO_CONFIG_ROOT).with_suffix("").as_posix()
        )
        hydra_config_path = REPO_CONFIG_ROOT
        hydra_config_name = config_label
    elif config_file.is_relative_to(REPO_ROOT):
        config_label = config_file.relative_to(REPO_ROOT).with_suffix("").as_posix()
        hydra_config_path = path
        hydra_config_name = config_name
    else:
        config_label = config_file.with_suffix("").name
        hydra_config_path = path
        hydra_config_name = config_name
    config_dir = config_label.rsplit("/", 1)[0] if "/" in config_label else ""
    config_prefix = f"{config_dir}/" if config_dir else ""
    return ConfigSpec(
        config_name=hydra_config_name,
        config_path=hydra_config_path,
        config_file=config_file,
        config_label=config_label,
        config_dir=config_dir,
        config_prefix=config_prefix,
        config_slug=_slugify(config_label),
        motion=config_file.stem.split("_", 1)[0],
    )


def _render_group(task_name: str, experiment_name: str, motion_name: str) -> str:
    return _slugify(f"{task_name}/{experiment_name}/{motion_name}_ppo")


def _render_experiment_directory(
    task_name: str, experiment_name: str, motion_name: str
) -> str:
    return f"{_slugify(task_name)}/{_slugify(experiment_name)}/{motion_name}"


def _default_play_overrides(config_file: Path) -> list[str]:
    return _common_default_play_overrides(config_file, REPO_CONFIG_ROOT)


# Command construction
def _build_train_command(job: Job, fixed_train_overrides: list[str]) -> list[str]:
    settings = OverrideSet.of(fixed_train_overrides).merge(
        OverrideSet.of(job.train_overrides)
    )
    # The batch owns seed identity even when fixed settings contain prefixed seeds.
    settings = settings.merge(
        OverrideSet.of(
            [
                f"seed={job.seed}",
                f"experiment.directory={job.experiment_directory}",
                f"++experiment.group={job.group}",
            ]
        )
    )
    return [
        "uv",
        "run",
        "train",
        "--config-path",
        str(job.config.config_path),
        "--config-name",
        job.config.config_name,
        *settings.items,
    ]


def _build_play_command(
    job: Job,
    *,
    play_video: bool,
    play_video_length: int,
    play_overrides: list[str],
    disable_default_play_overrides: bool,
) -> list[str]:
    auto_overrides = (
        []
        if disable_default_play_overrides
        else _default_play_overrides(job.config.config_file)
    )
    if not play_video:
        steps_raw = (
            OverrideSet.of(play_overrides)
            .as_map()
            .get("play.steps", str(play_video_length))
        )
        try:
            steps = int(steps_raw)
        except ValueError as exc:
            raise ValueError(
                "Nonvideo evaluator playback requires positive play.steps."
            ) from exc
        if steps <= 0:
            raise ValueError(
                "Nonvideo evaluator playback requires positive play.steps."
            )
        # The CLI otherwise selects an interactive viewer when video is off.
        # Explicit playback overrides below retain their authored horizon.
        auto_overrides.append(f"++play.steps={play_video_length}")
    command = [
        "uv",
        "run",
        "play",
        "--config-path",
        str(job.config.config_path),
        "--config-name",
        job.config.config_name,
        *auto_overrides,
        f"++play.checkpoint={job.best_checkpoint.as_posix()}",
        *play_overrides,
    ]
    if play_video:
        command.extend(
            [
                "play.video.enabled=true",
                f"play.video.length={play_video_length}",
                "++play.video.overlay_rewards=true",
                "++play.video.overlay_history_seconds=3.0",
                "++play.video.overlay_panel_width=360",
                "++play.output.emit_result_payload=true",
            ]
        )
    else:
        command.extend(
            [
                "play.video.enabled=false",
                "++play.output.emit_result_payload=true",
            ]
        )
    return command


# Metric extraction
def _parse_keyval_payload(path: Path, pattern: re.Pattern[str]) -> dict[str, str]:
    return _common_parse_keyval_payload(path, pattern, KEYVAL_RE)


def _parse_keyval_payloads(
    path: Path, pattern: re.Pattern[str]
) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    payloads: list[dict[str, str]] = []
    for line in text.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        payloads.append(
            {key: value for key, value in KEYVAL_RE.findall(match.group(1))}
        )
    return payloads


def _parse_train_best_mean_reward(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        match = TRAINING_COMPLETE_RE.search(line)
        if match is None:
            continue
        raw = match.group(1)
        if raw == "None":
            return None
        return float(raw)
    return None


def _parse_wandb_run_dir(path: Path) -> Path | None:
    return _common_parse_wandb_run_dir(path, WANDB_RUN_DIR_RE)


def _recent_mean(values: list[float], window: int) -> float | None:
    if not values:
        return None
    count = max(1, int(window))
    recent_values = values[-count:]
    return float(sum(recent_values) / len(recent_values))


def _train_benchmark_log_metrics(
    path: Path,
    recent_window: int,
    reward_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    payloads = _parse_keyval_payloads(path, BENCHMARK_EVAL_RE)
    reward_values: list[float] = []
    reward_frames: list[int] = []
    episode_length_values: list[float] = []
    success_rate_values: list[float] = []
    ape_values: list[float] = []
    ee_values: list[float] = []
    last_payload: dict[str, str] | None = None
    benchmark_curve: list[dict[str, float]] = []

    for payload in payloads:
        numeric_payload: dict[str, float] = {}
        for key, raw_value in payload.items():
            try:
                numeric_value = float(raw_value)
            except ValueError:
                continue
            if math.isfinite(numeric_value):
                numeric_payload[key] = numeric_value
        benchmark_curve.append(numeric_payload)
        reward = _safe_float(payload.get("current_reward_mean"))
        frames = _safe_int(payload.get("frames"))
        episode_length = _safe_float(payload.get("current_episode_length_mean"))
        success_rate = _safe_float(payload.get("current_success_rate"))
        ape_mean = _safe_float(payload.get("current_ape_mean"))
        ee_mean = _safe_float(payload.get("current_ee_mean"))
        if reward is not None:
            reward_values.append(reward)
            reward_frames.append(int(frames) if frames is not None else -1)
        if episode_length is not None:
            episode_length_values.append(episode_length)
        if success_rate is not None:
            success_rate_values.append(success_rate)
        if ape_mean is not None:
            ape_values.append(ape_mean)
        if ee_mean is not None:
            ee_values.append(ee_mean)
        last_payload = payload

    total_envs = (
        _safe_int(last_payload.get("total_envs")) if last_payload is not None else None
    )
    success_count = (
        _safe_int(last_payload.get("current_success_count"))
        if last_payload is not None
        else None
    )
    failure_count_last = None
    if total_envs is not None and success_count is not None:
        failure_count_last = max(0, int(total_envs) - int(success_count))

    threshold_frames: dict[str, int] = {}
    for threshold in reward_thresholds or []:
        key = _threshold_key(float(threshold))
        for frame, reward in zip(reward_frames, reward_values, strict=False):
            if frame < 0:
                continue
            if float(reward) >= float(threshold):
                threshold_frames[key] = int(frame)
                break

    return {
        "benchmark_eval_count": len(payloads),
        "benchmark_reward_mean_last": reward_values[-1] if reward_values else None,
        "benchmark_reward_mean_recent": _recent_mean(reward_values, recent_window),
        "benchmark_episode_length_mean_last": (
            episode_length_values[-1] if episode_length_values else None
        ),
        "benchmark_episode_length_mean_recent": _recent_mean(
            episode_length_values, recent_window
        ),
        "benchmark_success_rate_last": (
            success_rate_values[-1] if success_rate_values else None
        ),
        "benchmark_success_rate_recent": _recent_mean(
            success_rate_values, recent_window
        ),
        "benchmark_ape_mean_last": ape_values[-1] if ape_values else None,
        "benchmark_ape_mean_recent": _recent_mean(ape_values, recent_window),
        "benchmark_ee_mean_last": ee_values[-1] if ee_values else None,
        "benchmark_ee_mean_recent": _recent_mean(ee_values, recent_window),
        "benchmark_failure_count_last": failure_count_last,
        "benchmark_threshold_frames": threshold_frames,
        "benchmark_curve": benchmark_curve,
    }


def _benchmark_eval_source(task_cfg: DictConfig) -> str:
    batch_cfg = task_cfg.get("batch", None)
    if batch_cfg is None:
        return "playback"
    raw = str(batch_cfg.get("eval_source", "playback")).strip().lower()
    if raw not in {"playback", "train_benchmark"}:
        raise ValueError(
            "batch.eval_source must be either 'playback' or 'train_benchmark'."
        )
    return raw


def _train_benchmark_recent_window(task_cfg: DictConfig) -> int:
    batch_cfg = task_cfg.get("batch", None)
    if batch_cfg is None:
        return 3
    return max(1, int(batch_cfg.get("train_benchmark_recent_window", 3)))


def _populate_train_benchmark_metrics(
    job: Job,
    *,
    recent_window: int,
    reward_thresholds: list[float] | None = None,
) -> None:
    benchmark_log = _train_benchmark_log_path(job)
    job.train_evidence = evidence.parse_train_log(
        benchmark_log,
        recent_window=recent_window,
        reward_thresholds=reward_thresholds,
    )
    metrics = {
        **job.train_evidence.summary,
        "benchmark_curve": job.train_evidence.benchmark_curve,
        "benchmark_threshold_frames": job.train_evidence.benchmark_threshold_frames,
    }
    job.benchmark_eval_count = int(metrics.get("benchmark_eval_count") or 0)
    job.benchmark_reward_mean_last = metrics.get("benchmark_reward_mean_last")
    job.benchmark_reward_mean_recent = metrics.get("benchmark_reward_mean_recent")
    job.benchmark_episode_length_mean_last = metrics.get(
        "benchmark_episode_length_mean_last"
    )
    job.benchmark_episode_length_mean_recent = metrics.get(
        "benchmark_episode_length_mean_recent"
    )
    job.benchmark_success_rate_last = metrics.get("benchmark_success_rate_last")
    job.benchmark_success_rate_recent = metrics.get("benchmark_success_rate_recent")
    job.benchmark_ape_mean_last = metrics.get("benchmark_ape_mean_last")
    job.benchmark_ape_mean_recent = metrics.get("benchmark_ape_mean_recent")
    job.benchmark_ee_mean_last = metrics.get("benchmark_ee_mean_last")
    job.benchmark_ee_mean_recent = metrics.get("benchmark_ee_mean_recent")
    job.benchmark_failure_count_last = metrics.get("benchmark_failure_count_last")
    job.benchmark_threshold_frames = dict(
        metrics.get("benchmark_threshold_frames") or {}
    )
    job.benchmark_curve = [
        dict(point) for point in metrics.get("benchmark_curve") or []
    ]


def _train_benchmark_log_path(job: Job) -> Path:
    if job.run_dir is not None:
        run_log = job.run_dir / "train.log"
        if run_log.is_file():
            return run_log
    return job.train_log


def _play_log_metrics(
    path: Path,
    *,
    playback_payload: dict[str, str] | None = None,
    play_steps: int | None,
    episode_length: int | None,
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    tracking_payload = {}
    for line in reversed(text.splitlines()):
        match = TRACKING_METRICS_SUMMARY_RE.search(line)
        if match is None:
            continue
        tracking_payload = {
            key: value for key, value in KEYVAL_RE.findall(match.group(1))
        }
        break
    payload = dict(playback_payload or {})
    terminated_count = _safe_int(payload.get("terminated_count"))
    truncated_count = _safe_int(payload.get("truncated_count"))
    episode_finish_count = _safe_int(payload.get("episode_finish_count"))
    if terminated_count is not None or truncated_count is not None:
        terminated_count = 0 if terminated_count is None else int(terminated_count)
        truncated_count = 0 if truncated_count is None else int(truncated_count)
        if episode_finish_count is None:
            episode_finish_count = terminated_count + truncated_count
        fall_count = terminated_count
    else:
        episode_finish_count = len(list(EPISODE_FINISHED_RE.finditer(text)))
        reached_horizon = (
            play_steps is not None
            and episode_length is not None
            and episode_length >= play_steps - 1
        )
        fall_count = (
            max(0, episode_finish_count - 1)
            if reached_horizon
            else episode_finish_count
        )
        terminated_count = fall_count
        truncated_count = max(0, episode_finish_count - fall_count)
    return {
        "ape_mean": _safe_float(tracking_payload.get("ape_mean")),
        "ee_mean": _safe_float(tracking_payload.get("ee_mean")),
        "vel_mean": _safe_float(tracking_payload.get("vel_mean")),
        "accel_mean": _safe_float(tracking_payload.get("accel_mean")),
        "episode_finish_count": episode_finish_count,
        "fall_count": fall_count,
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
    }


def _launch_job(job: Job, fixed_train_overrides: list[str], dry_run: bool) -> None:
    job.known_run_dirs = run_discovery.list_run_dirs(job.experiment_directory)
    job.train_command = _build_train_command(job, fixed_train_overrides)
    print(
        f"[hpo:evaluator] launch experiment={job.experiment_name} motion={job.motion_name} seed={job.seed}"
    )
    print("  " + shlex.join(job.train_command))
    if dry_run:
        return
    job.train_log.parent.mkdir(parents=True, exist_ok=True)
    popen_kwargs: dict[str, object] = {
        "cwd": REPO_ROOT,
        "env": build_candidate_runtime_env(
            cache_key=job.train_log.stem,
            headless_rendering=True,
        ),
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    process: subprocess.Popen[str] | None = None
    try:
        with job.train_log.open("w", encoding="utf-8") as handle:
            popen_kwargs["stdout"] = handle
            process = subprocess.Popen(job.train_command, **popen_kwargs)
            job.process = process
            job.started_at = time.monotonic()
            job.process_group_id = process.pid if os.name == "posix" else None
            _register_job_process_group(job.process_group_id)
    except BaseException:
        if process is not None:
            try:
                _terminate_process_group(
                    process,
                    job.process_group_id,
                    grace_seconds=owned_process_group_termination_grace_seconds(),
                )
            finally:
                _unregister_job_process_group(job.process_group_id)
        job.process = None
        job.process_group_id = None
        raise


def _process_group_manifest_path() -> Path | None:
    return process_group_manifest_path()


def _register_job_process_group(process_group_id: int | None) -> None:
    register_process_group(process_group_id, _process_group_manifest_path())


def _unregister_job_process_group(process_group_id: int | None) -> None:
    unregister_process_group(process_group_id, _process_group_manifest_path())


def _finalize_train_job(job: Job, returncode: int) -> None:
    if job.started_at is not None:
        job.wall_seconds = time.monotonic() - job.started_at
    _terminate_process_group(
        job.process,
        job.process_group_id,
        grace_seconds=owned_process_group_termination_grace_seconds(),
    )
    _unregister_job_process_group(job.process_group_id)
    job.train_returncode = int(returncode)
    job.train_best_mean_reward = _parse_train_best_mean_reward(job.train_log)
    job.run_dir = run_discovery.find_newest_run_dir(
        job.experiment_directory, job.known_run_dirs
    )
    if job.run_dir is None:
        job.run_dir = _parse_wandb_run_dir(job.train_log)
    if job.run_dir is not None:
        candidate = job.run_dir / "checkpoints" / "best_agent.pt"
        if candidate.exists():
            job.best_checkpoint = candidate


def _run_play(
    job: Job,
    *,
    play_video: bool,
    play_video_length: int,
    play_overrides: list[str],
    disable_default_play_overrides: bool,
    play_steps: int | None,
    dry_run: bool,
) -> None:
    if job.best_checkpoint is None:
        return
    job.play_log = job.train_log.with_name(job.train_log.stem + "_play.log")
    job.play_command = _build_play_command(
        job,
        play_video=play_video,
        play_video_length=play_video_length,
        play_overrides=play_overrides,
        disable_default_play_overrides=disable_default_play_overrides,
    )
    if not play_video:
        play_steps = int(
            OverrideSet.of(hydra_overrides_from_command(job.play_command)).as_map()[
                "play.steps"
            ]
        )
    print(
        f"[hpo:evaluator] play experiment={job.experiment_name} motion={job.motion_name} seed={job.seed}"
    )
    print("  " + shlex.join(job.play_command))
    if dry_run:
        return
    assert job.play_log is not None
    try:
        with job.play_log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                job.play_command,
                cwd=REPO_ROOT,
                env=build_candidate_runtime_env(
                    cache_key=job.train_log.stem,
                    headless_rendering=True,
                ),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
            )
    except subprocess.TimeoutExpired:
        with job.play_log.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n[hpo:evaluator] playback timed out after "
                f"{DEFAULT_SUBPROCESS_TIMEOUT_SECONDS:.1f}s\n"
            )
        job.play_returncode = 124
        return
    job.play_returncode = int(completed.returncode)
    if completed.returncode != 0:
        return
    payload = _parse_keyval_payload(job.play_log, PLAYBACK_RESULT_RE)
    job.playback_result = payload
    job.episode_length = _safe_int(payload.get("episode_length"))
    job.episode_reward = _safe_float(payload.get("episode_reward"))
    job.playback_evidence = evidence.parse_play_log(
        job.play_log,
        playback_payload=payload,
        play_steps=play_steps,
        episode_length=job.episode_length,
    )
    metrics = job.playback_evidence.as_dict()
    job.ape_mean = metrics["ape_mean"]
    job.ee_mean = metrics["ee_mean"]
    job.vel_mean = metrics["vel_mean"]
    job.accel_mean = metrics["accel_mean"]
    job.episode_finish_count = int(metrics["episode_finish_count"])
    job.fall_count = int(metrics["fall_count"])
    job.terminated_count = int(metrics["terminated_count"])
    job.truncated_count = int(metrics["truncated_count"])
    video_output_path = payload.get("video_output_path")
    if video_output_path:
        job.video_path = Path(video_output_path).resolve()


def _job_row(job: Job) -> dict[str, Any]:
    return {
        "experiment": job.experiment_name,
        "motion": job.motion_name,
        "config": job.config.config_label,
        "config_name": job.config.config_name,
        "seed": job.seed,
        "group": job.group,
        "experiment_directory": job.experiment_directory,
        "train_returncode": job.train_returncode,
        "play_returncode": job.play_returncode,
        "train_best_mean_reward": job.train_best_mean_reward,
        "episode_length": job.episode_length,
        "episode_reward": job.episode_reward,
        "ape_mean": job.ape_mean,
        "ee_mean": job.ee_mean,
        "vel_mean": job.vel_mean,
        "accel_mean": job.accel_mean,
        "episode_finish_count": job.episode_finish_count,
        "fall_count": job.fall_count,
        "terminated_count": job.terminated_count,
        "truncated_count": job.truncated_count,
        "benchmark_eval_count": job.benchmark_eval_count,
        "benchmark_reward_mean_last": job.benchmark_reward_mean_last,
        "benchmark_reward_mean_recent": job.benchmark_reward_mean_recent,
        "benchmark_episode_length_mean_last": job.benchmark_episode_length_mean_last,
        "benchmark_episode_length_mean_recent": job.benchmark_episode_length_mean_recent,
        "benchmark_success_rate_last": job.benchmark_success_rate_last,
        "benchmark_success_rate_recent": job.benchmark_success_rate_recent,
        "benchmark_ape_mean_last": job.benchmark_ape_mean_last,
        "benchmark_ape_mean_recent": job.benchmark_ape_mean_recent,
        "benchmark_ee_mean_last": job.benchmark_ee_mean_last,
        "benchmark_ee_mean_recent": job.benchmark_ee_mean_recent,
        "benchmark_failure_count_last": job.benchmark_failure_count_last,
        "benchmark_threshold_frames": dict(job.benchmark_threshold_frames),
        "benchmark_curve": [dict(point) for point in job.benchmark_curve],
        "run_dir": str(job.run_dir) if job.run_dir is not None else "",
        "best_checkpoint": str(job.best_checkpoint)
        if job.best_checkpoint is not None
        else "",
        "video_path": str(job.video_path) if job.video_path is not None else "",
        "train_log": str(job.train_log),
        "play_log": str(job.play_log) if job.play_log is not None else "",
        "train_command": shlex.join(job.train_command) if job.train_command else "",
        "play_command": shlex.join(job.play_command) if job.play_command else "",
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _aggregate_values(values: list[float], method: str) -> float:
    aggregate = _mean(values) if method == "mean" else _median(values)
    if aggregate is None:
        raise ValueError("Cannot aggregate an empty metric sequence.")
    return float(aggregate)


def _summarize_rows(
    rows: list[dict[str, Any]], *, eval_source: str = "playback"
) -> dict[str, Any]:
    rewards = [
        float(row["episode_reward"])
        for row in rows
        if row.get("episode_reward") is not None
    ]
    lengths = [
        float(row["episode_length"])
        for row in rows
        if row.get("episode_length") is not None
    ]
    apes = [float(row["ape_mean"]) for row in rows if row.get("ape_mean") is not None]
    ees = [float(row["ee_mean"]) for row in rows if row.get("ee_mean") is not None]
    vels = [float(row["vel_mean"]) for row in rows if row.get("vel_mean") is not None]
    accels = [
        float(row["accel_mean"]) for row in rows if row.get("accel_mean") is not None
    ]
    benchmark_reward_last = [
        float(row["benchmark_reward_mean_last"])
        for row in rows
        if row.get("benchmark_reward_mean_last") is not None
    ]
    benchmark_reward_recent = [
        float(row["benchmark_reward_mean_recent"])
        for row in rows
        if row.get("benchmark_reward_mean_recent") is not None
    ]
    benchmark_episode_length_last = [
        float(row["benchmark_episode_length_mean_last"])
        for row in rows
        if row.get("benchmark_episode_length_mean_last") is not None
    ]
    benchmark_episode_length_recent = [
        float(row["benchmark_episode_length_mean_recent"])
        for row in rows
        if row.get("benchmark_episode_length_mean_recent") is not None
    ]
    benchmark_success_rate_last = [
        float(row["benchmark_success_rate_last"])
        for row in rows
        if row.get("benchmark_success_rate_last") is not None
    ]
    benchmark_success_rate_recent = [
        float(row["benchmark_success_rate_recent"])
        for row in rows
        if row.get("benchmark_success_rate_recent") is not None
    ]
    benchmark_ape_last = [
        float(row["benchmark_ape_mean_last"])
        for row in rows
        if row.get("benchmark_ape_mean_last") is not None
    ]
    benchmark_ape_recent = [
        float(row["benchmark_ape_mean_recent"])
        for row in rows
        if row.get("benchmark_ape_mean_recent") is not None
    ]
    benchmark_ee_last = [
        float(row["benchmark_ee_mean_last"])
        for row in rows
        if row.get("benchmark_ee_mean_last") is not None
    ]
    benchmark_ee_recent = [
        float(row["benchmark_ee_mean_recent"])
        for row in rows
        if row.get("benchmark_ee_mean_recent") is not None
    ]
    benchmark_eval_counts = [
        float(row["benchmark_eval_count"])
        for row in rows
        if row.get("benchmark_eval_count") is not None
    ]
    threshold_key_set: set[str] = set()
    for row in rows:
        payload = row.get("benchmark_threshold_frames") or {}
        if isinstance(payload, dict):
            threshold_key_set.update(str(key) for key in payload)
    benchmark_failure_count_last = sum(
        int(row.get("benchmark_failure_count_last") or 0)
        for row in rows
        if row.get("benchmark_failure_count_last") is not None
    )
    total_falls = sum(int(row.get("fall_count") or 0) for row in rows)
    total_episode_finishes = sum(
        int(row.get("episode_finish_count") or 0) for row in rows
    )
    total_terminated = sum(int(row.get("terminated_count") or 0) for row in rows)
    total_truncated = sum(int(row.get("truncated_count") or 0) for row in rows)
    summary = {
        "run_count": len(rows),
        "train_runs_ok": sum(int(row.get("train_returncode") == 0) for row in rows),
        "play_runs_ok": sum(int(row.get("play_returncode") == 0) for row in rows),
        "mean_reward": _mean(rewards),
        "mean_episode_length": _mean(lengths),
        "mean_ape": _mean(apes),
        "mean_ee": _mean(ees),
        "mean_vel": _mean(vels),
        "mean_accel": _mean(accels),
        "total_falls": int(total_falls),
        "mean_falls": float(total_falls / len(rows)) if rows else None,
        "total_episode_finishes": int(total_episode_finishes),
        "total_terminated": int(total_terminated),
        "total_truncated": int(total_truncated),
        "mean_benchmark_reward_mean_last": _mean(benchmark_reward_last),
        "mean_benchmark_reward_mean_recent": _mean(benchmark_reward_recent),
        "mean_benchmark_episode_length_mean_last": _mean(benchmark_episode_length_last),
        "mean_benchmark_episode_length_mean_recent": _mean(
            benchmark_episode_length_recent
        ),
        "mean_benchmark_success_rate_last": _mean(benchmark_success_rate_last),
        "mean_benchmark_success_rate_recent": _mean(benchmark_success_rate_recent),
        "mean_benchmark_ape_mean_last": _mean(benchmark_ape_last),
        "mean_benchmark_ape_mean_recent": _mean(benchmark_ape_recent),
        "mean_benchmark_ee_mean_last": _mean(benchmark_ee_last),
        "mean_benchmark_ee_mean_recent": _mean(benchmark_ee_recent),
        "mean_benchmark_eval_count": _mean(benchmark_eval_counts),
        "total_benchmark_failure_count_last": int(benchmark_failure_count_last),
        "mean_benchmark_failure_count_last": (
            float(benchmark_failure_count_last / len(rows)) if rows else None
        ),
    }

    def _threshold_sort_value(key: str) -> float:
        return float(key.replace("p", "."))

    for threshold_key in sorted(
        threshold_key_set, key=_threshold_sort_value, reverse=True
    ):
        threshold_frames = [
            float((row.get("benchmark_threshold_frames") or {}).get(threshold_key))
            for row in rows
            if isinstance(row.get("benchmark_threshold_frames"), dict)
            and (row.get("benchmark_threshold_frames") or {}).get(threshold_key)
            is not None
        ]
        summary[f"benchmark_threshold_{threshold_key}_reached_count"] = len(
            threshold_frames
        )
        summary[f"benchmark_threshold_{threshold_key}_frames_median"] = _median(
            threshold_frames
        )
    if eval_source == "train_benchmark":
        summary["mean_reward"] = (
            summary["mean_benchmark_reward_mean_recent"]
            if summary["mean_benchmark_reward_mean_recent"] is not None
            else summary["mean_benchmark_reward_mean_last"]
        )
        summary["mean_episode_length"] = (
            summary["mean_benchmark_episode_length_mean_recent"]
            if summary["mean_benchmark_episode_length_mean_recent"] is not None
            else summary["mean_benchmark_episode_length_mean_last"]
        )
        summary["mean_ape"] = (
            summary["mean_benchmark_ape_mean_recent"]
            if summary["mean_benchmark_ape_mean_recent"] is not None
            else summary["mean_benchmark_ape_mean_last"]
        )
        summary["mean_ee"] = (
            summary["mean_benchmark_ee_mean_recent"]
            if summary["mean_benchmark_ee_mean_recent"] is not None
            else summary["mean_benchmark_ee_mean_last"]
        )
        summary["total_falls"] = summary["total_benchmark_failure_count_last"]
        summary["mean_falls"] = summary["mean_benchmark_failure_count_last"]
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Evaluation runtime
def _terminate_process_group(
    process: subprocess.Popen[str] | None,
    process_group_id: int | None,
    *,
    grace_seconds: float = 5.0,
) -> None:
    """One owned-group terminator, shared with the candidate executor."""

    if process is None:
        return
    terminate_process_group(
        process,
        grace_seconds,
        process_group_id=process_group_id,
    )


def _terminate_running_jobs(jobs: list[Job]) -> None:
    for job in jobs:
        try:
            _terminate_process_group(
                job.process,
                job.process_group_id,
                grace_seconds=owned_process_group_termination_grace_seconds(),
            )
        finally:
            _unregister_job_process_group(job.process_group_id)


def _seed_batch_mode(batch_cfg: DictConfig) -> str:
    seeds = [int(item) for item in _ensure_list(batch_cfg.get("seeds"), "batch.seeds")]
    default_mode = "require" if len(seeds) > 1 else "off"
    raw = str(batch_cfg.get("seed_batch", default_mode) or default_mode).strip().lower()
    if raw not in {"off", "auto", "require"}:
        raise ValueError("batch.seed_batch must be 'off', 'auto', or 'require'.")
    max_parallel = _batch_max_parallel(batch_cfg)
    if len(seeds) > 1 and raw == "off":
        raise ValueError("Multi-seed hpo cannot disable persistent seed execution.")
    persistent_supported = max_parallel == 1 and len(seeds) > 1
    if raw == "require" and not persistent_supported:
        raise ValueError(
            "batch.seed_batch=require needs batch.max_parallel=1 and at least two seeds."
        )
    if raw == "off" or not persistent_supported:
        return "isolated"
    return "persistent"


def _batch_max_parallel(batch_cfg: DictConfig) -> int:
    max_parallel = int(batch_cfg.get("max_parallel", 1))
    if max_parallel < 1:
        raise ValueError("batch.max_parallel must be >= 1.")
    return max_parallel


def _compile_cache_scope(batch_cfg: DictConfig) -> str:
    scope = str(batch_cfg.get("compile_cache_scope", "isolated")).strip().lower()
    if scope not in {"isolated", "task"}:
        raise ValueError("batch.compile_cache_scope must be 'isolated' or 'task'.")
    return scope


_TIMEOUT_RETURNCODE = 124


def _trial_timed_out(job: Job, policy: ExecutionPolicy) -> bool:
    if policy.trial_timeout_seconds is None or job.started_at is None:
        return False
    return (time.monotonic() - job.started_at) > policy.trial_timeout_seconds


def _run_jobs_with_persistent_seed_workers(
    jobs: list[Job],
    *,
    fixed_train_overrides: list[str],
    compile_cache_scope: str = "isolated",
    task_cache_key: str | None = None,
    policy: ExecutionPolicy = ExecutionPolicy(),
) -> list[Job]:
    grouped: dict[tuple[str, str, str], list[Job]] = {}
    for job in jobs:
        key = (
            job.motion_name,
            str(job.config.config_path),
            str(job.config.config_name),
        )
        grouped.setdefault(key, []).append(job)

    completed: list[Job] = []
    for (motion_name, config_path, config_name), group in grouped.items():
        if compile_cache_scope == "task":
            if task_cache_key is None:
                raise ValueError(
                    "task_cache_key is required for task-scoped compiler caches."
                )
            cache_key = f"{task_cache_key}-{motion_name}-{config_name}"
        else:
            cache_key = f"playback-{group[0].experiment_name}-{motion_name}"
        print(
            "[hpo:evaluator] seed_execution=persistent "
            f"experiment={group[0].experiment_name} motion={motion_name} "
            f"seeds={','.join(str(job.seed) for job in group)} "
            f"compile_cache_scope={compile_cache_scope}"
        )
        with SeedTrainWorker(
            config_path=config_path,
            config_name=config_name,
            cache_key=cache_key,
            cache_scope=compile_cache_scope,
        ) as worker:
            for job in group:
                while True:
                    job.known_run_dirs = run_discovery.list_run_dirs(
                        job.experiment_directory
                    )
                    job.train_command = _build_train_command(job, fixed_train_overrides)
                    print(
                        "[hpo:evaluator] launch "
                        f"experiment={job.experiment_name} motion={job.motion_name} "
                        f"seed={job.seed}"
                    )
                    print("  " + shlex.join(job.train_command))
                    job.train_log.parent.mkdir(parents=True, exist_ok=True)
                    wall, _fps, returncode, _started_at, _finished_at = (
                        worker.run_trial(
                            command=job.train_command,
                            log_file=job.train_log,
                            dry_run=False,
                            frames_per_run=None,
                            # The worker honours this and returns 124; passing
                            # None here is what made `batch.trial_timeout_seconds`
                            # a no-op for every multi-seed task, which is every
                            # task that reaches this path.
                            trial_timeout_seconds=policy.trial_timeout_seconds,
                            overrides=hydra_overrides_from_command(job.train_command),
                        )
                    )
                    _finalize_train_job(job, returncode)
                    # The persistent worker measures the trial itself.
                    job.wall_seconds = wall
                    if returncode == 0:
                        break
                    if job.train_retry_count >= policy.retries_per_seed:
                        break
                    if not policy.retries_on(_train_failure_trigger(job)):
                        break

                    next_retry_index = job.train_retry_count + 1
                    archived_log = _archive_failed_train_log(
                        job.train_log,
                        retry_index=next_retry_index,
                        reason="worker",
                    )
                    job.train_retry_count = next_retry_index
                    _reset_job_after_train_retry(job)
                    print(
                        "[hpo:evaluator] persistent seed worker failed for "
                        f"experiment={job.experiment_name} motion={job.motion_name} "
                        f"seed={job.seed}; retrying once in a fresh worker process."
                    )
                    if archived_log is not None:
                        print(f"  archived failed train log to {archived_log}")
                completed.append(job)
                if policy.stop_on_fail and job.train_returncode != 0:
                    print(
                        "[hpo:evaluator] stop_on_fail: abandoning "
                        f"experiment={job.experiment_name} after motion="
                        f"{job.motion_name} seed={job.seed} failed."
                    )
                    return completed
    return completed


def _run_jobs_for_experiment(
    task_cfg: DictConfig,
    experiment: ExperimentSpec,
    output_root: Path,
    *,
    dry_run: bool,
) -> list[Job]:
    batch_cfg = task_cfg.batch
    eval_source = _benchmark_eval_source(task_cfg)
    train_benchmark_recent_window = _train_benchmark_recent_window(task_cfg)
    seeds = [int(item) for item in _ensure_list(batch_cfg.get("seeds"), "batch.seeds")]
    policy = _execution_policy(batch_cfg)
    max_parallel = policy.max_parallel
    seed_batch_mode = _seed_batch_mode(batch_cfg)
    compile_cache_scope = _compile_cache_scope(batch_cfg)
    if compile_cache_scope == "task" and seed_batch_mode != "persistent":
        raise ValueError(
            "batch.compile_cache_scope=task requires persistent seed execution "
            "(batch.max_parallel=1 and at least two seeds)."
        )
    playback_cfg = batch_cfg.get("playback", None) or {}
    play_video = bool(playback_cfg.get("video", True))
    play_video_length = int(playback_cfg.get("video_length", 1800))
    play_steps_raw = task_cfg.get("play", {}).get("steps", None)
    play_steps = None if play_steps_raw in (None, "") else int(play_steps_raw)
    disable_default_play_overrides = bool(
        playback_cfg.get("disable_default_overrides", False)
    )
    fixed_train_overrides = [
        str(item)
        for item in _ensure_list(
            batch_cfg.get("fixed_train_overrides", []),
            "batch.fixed_train_overrides",
        )
    ]
    play_overrides = [
        str(item)
        for item in _ensure_list(
            playback_cfg.get("overrides"), "batch.playback.overrides"
        )
    ]

    log_root = output_root / "launcher_logs" / _slugify(experiment.name)
    log_root.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = []
    task_name = str(task_cfg.get("name", "hpo"))
    fixed_values = OverrideSet.of(fixed_train_overrides).as_map()
    run_directory = fixed_values.get("experiment.directory", _slugify(task_name))

    for seed in seeds:
        for motion_name, motion_spec in experiment.motions.items():
            config = _resolve_config_spec(
                motion_spec.config_path, motion_spec.config_name
            )
            jobs.append(
                Job(
                    experiment_name=experiment.name,
                    motion_name=motion_name,
                    config=config,
                    seed=seed,
                    group=_render_group(task_name, experiment.name, motion_name),
                    experiment_directory=str(
                        Path(run_directory) / _slugify(experiment.name) / motion_name
                    ),
                    train_log=log_root / f"{motion_name}_seed{seed}.train.log",
                    train_overrides=motion_spec.overrides,
                    parent=experiment.parent,
                )
            )

    if dry_run:
        print(
            f"[hpo:evaluator] seed_execution={seed_batch_mode} "
            f"compile_cache_scope={compile_cache_scope}"
        )
        for job in jobs:
            _launch_job(job, fixed_train_overrides, dry_run=True)
            if eval_source == "playback":
                job.best_checkpoint = Path("best_agent.pt")
                _run_play(
                    job,
                    play_video=play_video,
                    play_video_length=play_video_length,
                    play_overrides=play_overrides,
                    disable_default_play_overrides=disable_default_play_overrides,
                    play_steps=play_steps,
                    dry_run=True,
                )
                job.best_checkpoint = None
        return jobs

    if seed_batch_mode == "persistent":
        completed = _run_jobs_with_persistent_seed_workers(
            jobs,
            fixed_train_overrides=fixed_train_overrides,
            compile_cache_scope=compile_cache_scope,
            task_cache_key=task_name,
            policy=policy,
        )
        for job in completed:
            if job.train_returncode != 0:
                continue
            if eval_source == "playback" and job.best_checkpoint is not None:
                _run_play(
                    job,
                    play_video=play_video,
                    play_video_length=play_video_length,
                    play_overrides=play_overrides,
                    disable_default_play_overrides=disable_default_play_overrides,
                    play_steps=play_steps,
                    dry_run=False,
                )
            elif eval_source == "train_benchmark":
                _populate_train_benchmark_metrics(
                    job,
                    recent_window=train_benchmark_recent_window,
                )
        return completed

    pending = jobs.copy()
    running: list[Job] = []
    completed: list[Job] = []
    try:
        while pending or running:
            while pending and len(running) < max(1, max_parallel):
                job = pending.pop(0)
                try:
                    _launch_job(job, fixed_train_overrides, dry_run=False)
                    if job.process is not None:
                        running.append(job)
                except BaseException:
                    _terminate_running_jobs([job])
                    raise

            if not running:
                continue
            time.sleep(5.0)
            still_running: list[Job] = []
            for job in running:
                assert job.process is not None
                returncode = job.process.poll()
                if returncode is None:
                    if _trial_timed_out(job, policy):
                        _terminate_running_jobs([job])
                        returncode = _TIMEOUT_RETURNCODE
                    else:
                        still_running.append(job)
                        continue
                _finalize_train_job(job, returncode)
                if _maybe_retry_train(
                    job,
                    fixed_train_overrides=fixed_train_overrides,
                    dry_run=False,
                    policy=policy,
                ):
                    still_running.append(job)
                    continue
                completed.append(job)
            running = still_running
            if policy.stop_on_fail and any(
                int(job.train_returncode or 0) != 0 for job in completed
            ):
                _terminate_running_jobs(running)
                pending.clear()
                running = []
    except KeyboardInterrupt:
        _terminate_running_jobs(running)
        raise
    except BaseException:
        _terminate_running_jobs(running)
        raise

    for job in completed:
        if job.train_returncode != 0:
            continue
        if eval_source == "playback" and job.best_checkpoint is not None:
            _run_play(
                job,
                play_video=play_video,
                play_video_length=play_video_length,
                play_overrides=play_overrides,
                disable_default_play_overrides=disable_default_play_overrides,
                play_steps=play_steps,
                dry_run=False,
            )
        elif eval_source == "train_benchmark":
            _populate_train_benchmark_metrics(
                job,
                recent_window=train_benchmark_recent_window,
            )
    return completed


def _summarize_experiment(
    task_cfg: DictConfig, experiment: ExperimentSpec, jobs: list[Job], batch_root: Path
) -> dict[str, Any]:
    rows = [_job_row(job) for job in jobs]
    per_motion: dict[str, Any] = {}
    eval_source = _benchmark_eval_source(task_cfg)
    for motion_name in experiment.motions:
        motion_rows = [row for row in rows if row["motion"] == motion_name]
        per_motion[motion_name] = _summarize_rows(motion_rows, eval_source=eval_source)
        per_motion[motion_name]["runs"] = motion_rows
    aggregate = _summarize_rows(rows, eval_source=eval_source)
    aggregate["batch_root"] = str(batch_root)
    aggregate["eval_source"] = eval_source
    aggregate["seed_execution"] = _seed_batch_mode(task_cfg.batch)
    payload = {
        "name": experiment.name,
        "rationale": experiment.rationale,
        "aggregate": aggregate,
        "motions": per_motion,
        "runs": rows,
    }
    return payload


def _trial_records(jobs: list[Job]) -> list[evidence.TrialRecord]:
    """One record per candidate x motion x seed, as executed."""

    return [
        evidence.TrialRecord(
            candidate=job.experiment_name,
            motion=job.motion_name,
            seed=job.seed,
            attempt=job.train_retry_count,
            returncode=job.train_returncode,
            train_command=list(job.train_command),
            train_log=str(job.train_log),
            run_dir=None if job.run_dir is None else str(job.run_dir),
            best_checkpoint=(
                None if job.best_checkpoint is None else str(job.best_checkpoint)
            ),
            wall_seconds=job.wall_seconds,
            effective_overrides=list(job.train_overrides),
            train=job.train_evidence.as_dict(),
            playback=(
                None
                if job.playback_evidence is None
                else job.playback_evidence.as_dict()
            ),
            failure=(
                None
                if int(job.train_returncode or 0) == 0
                else f"train returncode {job.train_returncode}"
            ),
        )
        for job in jobs
    ]


def _write_evidence_files(
    output_root: Path,
    *,
    task_path: Path,
    proposal_path: Path,
    jobs_by_candidate: dict[str, list[Job]],
    required_seeds: tuple[int, ...],
    policy: ExecutionPolicy,
    dry_run: bool,
) -> None:
    """Write the execution-defined evidence both comparators read.

    Completeness here is an execution fact -- every required seed returned zero
    within the retry budget -- and never an objective decision.
    """

    records: list[evidence.TrialRecord] = []
    candidates: list[dict[str, Any]] = []
    for name, jobs in jobs_by_candidate.items():
        candidate_records = _trial_records(jobs)
        records.extend(candidate_records)
        succeeded = {
            (job.motion_name, job.seed) for job in jobs if job.train_returncode == 0
        }
        required_trials = {
            (motion, seed)
            for motion in {job.motion_name for job in jobs}
            for seed in required_seeds
        }
        failure_reasons = sorted(
            {
                record.failure
                for record in candidate_records
                if record.failure is not None
            }
        )
        candidates.append(
            {
                "name": name,
                "parent": _candidate_parent(jobs),
                "motions": sorted({job.motion_name for job in jobs}),
                "effective_overrides": {
                    job.motion_name: list(job.train_overrides) for job in jobs
                },
                "trial_ids": [
                    f"{record.candidate}/{record.motion}/{record.seed}"
                    for record in candidate_records
                ],
                # A dry run planned the trials but ran none, so nothing is
                # complete: completeness is an execution fact.
                "complete": (
                    not dry_run
                    and bool(jobs)
                    and required_trials.issubset(succeeded)
                    and all(job.train_returncode == 0 for job in jobs)
                ),
                "failure_reasons": failure_reasons,
            }
        )

    trials_path = output_root / "trials.jsonl"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    with trials_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), sort_keys=False) + "\n")

    _write_json(
        output_root / "evidence.json",
        {
            "schema_version": evidence.EVIDENCE_SCHEMA_VERSION,
            "task": str(task_path),
            "proposal": str(proposal_path),
            "output_root": str(output_root),
            "dry_run": bool(dry_run),
            "execution": {
                "retries_per_seed": policy.retries_per_seed,
                "retry_triggers": list(policy.retry_triggers),
                "trial_timeout_seconds": policy.trial_timeout_seconds,
                "stop_on_fail": policy.stop_on_fail,
                "max_parallel": policy.max_parallel,
                "seeds": list(required_seeds),
            },
            "candidates": candidates,
        },
    )


def _candidate_parent(jobs: list[Job]) -> Any:
    for job in jobs:
        if job.parent is not None:
            return job.parent
    return "root"


def _read_candidate_completeness(output_root: Path) -> list[dict[str, Any]]:
    payload = json.loads((output_root / "evidence.json").read_text(encoding="utf-8"))
    return list(payload.get("candidates", []))


def _write_summary_files(output_root: Path, payload: dict[str, Any]) -> None:
    """Write the human-readable side of one evaluation.

    The frontier tables this used to render belonged to the comparator and went
    with it; what is left is the flattened per-seed table plus the aggregates.
    """

    all_rows: list[dict[str, Any]] = []
    for experiment in payload.get("experiments", []):
        all_rows.extend(experiment.get("runs", []))
    _write_csv(output_root / "results.csv", all_rows)
    _write_json(output_root / "summary.json", payload)

    lines = [f"# {payload.get('title', 'HPO evaluation')}", ""]
    lines.append("| experiment | mean_reward | mean_ape | mean_ee | total_falls |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for experiment in payload.get("experiments", []):
        aggregate = experiment.get("aggregate", {})

        def cell(key: str) -> str:
            value = aggregate.get(key)
            return "" if value is None else f"{float(value):.6f}"

        lines.append(
            f"| {experiment['name']} | {cell('mean_reward')} | "
            f"{cell('mean_ape')} | {cell('mean_ee')} | "
            f"{aggregate.get('total_falls', '')} |"
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic evaluator for agent-authored experiment bundles. "
            "Runs the configured training and evaluation protocol for each proposed experiment."
        )
    )
    parser.add_argument("--task", type=Path, required=True, help="Task YAML path.")
    parser.add_argument(
        "--proposal",
        type=Path,
        required=True,
        help="Proposal YAML path generated by the planning agent.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where logs and summaries are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and write planned summaries without executing them.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the task and proposal contract without launching jobs.",
    )
    return parser.parse_args()


def _evaluator_exit_code(*, best_experiment: str | None, dry_run: bool) -> int:
    return 0 if dry_run or best_experiment is not None else 1


def _handle_termination_signal(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def main() -> int:
    if os.name == "posix":
        signal.signal(signal.SIGTERM, _handle_termination_signal)
    args = _parse_args()
    task_path = _resolve_repo_path(args.task)
    proposal_path = _resolve_repo_path(args.proposal)
    output_root = _resolve_repo_path(args.output_root)

    task_cfg = _load_task(task_path)
    experiments = load_proposal_experiments(task_cfg, proposal_path)
    if args.validate_only:
        print("proposal contract: PASS")
        return 0

    # Execution first, then evidence, then comparison. Evidence is written
    # before any objective is applied so a comparator failure never costs the
    # record of what actually ran.
    jobs_by_candidate: dict[str, list[Job]] = {}
    experiment_roots: dict[str, Path] = {}
    for experiment in experiments:
        experiment_root = output_root / _slugify(experiment.name)
        experiment_roots[experiment.name] = experiment_root
        jobs_by_candidate[experiment.name] = _run_jobs_for_experiment(
            task_cfg, experiment, experiment_root, dry_run=bool(args.dry_run)
        )

    batch_cfg = task_cfg.batch
    _write_evidence_files(
        output_root,
        task_path=task_path,
        proposal_path=proposal_path,
        jobs_by_candidate=jobs_by_candidate,
        required_seeds=tuple(
            int(seed) for seed in _ensure_list(batch_cfg.get("seeds"), "batch.seeds")
        ),
        policy=_execution_policy(batch_cfg),
        dry_run=bool(args.dry_run),
    )

    experiment_payloads = [
        _summarize_experiment(
            task_cfg,
            experiment,
            jobs_by_candidate[experiment.name],
            experiment_roots[experiment.name],
        )
        for experiment in experiments
    ]

    # Comparison happens in the controller. The evaluator reports only whether
    # a candidate ran to completion, which is an execution fact.
    complete = [
        candidate["name"]
        for candidate in _read_candidate_completeness(output_root)
        if candidate["complete"]
    ]
    best_experiment = complete[0] if complete else None

    summary_payload = {
        "title": str(task_cfg.get("name", task_path.stem)),
        "task": str(task_path),
        "proposal": str(proposal_path),
        "output_root": str(output_root),
        "best_experiment": best_experiment,
        "experiments": experiment_payloads,
    }
    _write_summary_files(output_root, summary_payload)

    print(f"[hpo:evaluator] output_root={output_root}")
    print(f"[hpo:evaluator] evidence_json={(output_root / 'evidence.json').resolve()}")
    print(f"[hpo:evaluator] summary_json={(output_root / 'summary.json').resolve()}")
    print(f"[hpo:evaluator] best_experiment={best_experiment or ''}")
    returncode = _evaluator_exit_code(
        best_experiment=best_experiment,
        dry_run=bool(args.dry_run),
    )
    if returncode != 0:
        print(
            "[hpo:evaluator] no candidate produced valid complete evidence.",
            file=sys.stderr,
        )
    return returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
