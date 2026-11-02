"""The one hpo task schema, validator, and runtime options.

``objective.mode`` selects the comparator family and nothing else. Every other
section -- ``batch``, ``parameter_space``, ``forbidden_prefixes``, ``planner``,
``hpo_tracking``, ``runtime`` -- is shared by both modes and is
validated identically here.

Strategy-specific keys under ``objective`` are deliberately not enumerated: the
comparator that a ``(mode, strategy)`` pair selects owns them. This module only
checks that the pair is coherent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, ListConfig

from instantmimic.hpo.evaluator_launch import (
    EvaluatorLaunchConfig,
    resolve_evaluator_launch_config,
)
from instantmimic.hpo.evidence import (
    parse_motion_group_evidence,
    parse_motion_group_seed_reduce,
)

# Comparator families. `objective/__init__.py` turns a pair into a comparator;
# this table only decides whether the pair is spellable.
SCALAR_STRATEGIES = (
    "wall_clock",
    "time_to_reward_threshold",
    "time_to_tracking_benchmark",
    "frames_to_tracking_benchmark",
    "tracking_benchmark_auc",
)
# Spellable before the unification and scored by machinery that read the old
# scalar evaluator's aggregate rather than a trial's evidence. `trial_objective`
# has no branch for either, so they would score as `time_to_reward_threshold`
# and rank by a metric nobody asked for. No shipped config used them.
UNIMPLEMENTED_SCALAR_STRATEGIES: Mapping[str, str] = {
    "weighted_sum": (
        "it scored a weighted aggregate the unified evaluator does not emit"
    ),
    "reward_threshold_ladder": (
        "it scored a threshold ladder the unified evaluator does not emit"
    ),
}
FRONTIER_STRATEGIES = (
    "pareto",
    "learning_curve_pareto",
    "axes_pareto",
)
STRATEGIES_BY_MODE: Mapping[str, tuple[str, ...]] = {
    "scalar": SCALAR_STRATEGIES,
    "pareto": FRONTIER_STRATEGIES,
}

TOP_LEVEL_KEYS = frozenset(
    {
        "name",
        "description",
        "objective",
        "parameter_space",
        "forbidden_prefixes",
        "batch",
        "planner",
        "hpo_tracking",
    }
)
# Written into the task mapping by `run_context` once the run directory exists.
INJECTED_TOP_LEVEL_KEYS = frozenset({"state_dir", "output_root"})

BATCH_KEYS = frozenset(
    {
        "motions",
        "seeds",
        "seed_batch",
        "max_parallel",
        "retries_per_seed",
        "retry_triggers",
        "trial_timeout_seconds",
        "stop_on_fail",
        "compile_cache_scope",
        "fixed_train_overrides",
        "eval_source",
        "train_benchmark_recent_window",
        "playback",
        "min_experiments_per_cycle",
        "max_experiments_per_cycle",
        "lock_motion_configs",
    }
)
PLAYBACK_KEYS = frozenset(
    {"video", "video_length", "overrides", "disable_default_overrides"}
)
MOTION_KEYS = frozenset({"config_path", "config_name"})

PLANNER_KEYS = frozenset(
    {
        # agent_runtime selection
        "platform",
        "model",
        "reasoning_effort",
        "sandbox",
        "command",
        "codex_config",
        "host",
        "auto_start_ollama",
        "agent",
        "variant",
        "output_format",
        "pure",
        "skip_permissions",
        # protocol
        "timeout_seconds",
        "timeout_retries",
        "retry_delay_seconds",
        "recent_cycles",
        "instructions",
        "initial_candidates",
        "search_base_selection",
        "max_same_dimension_streak",
        "deep_review",
        "creativity",
        # Per-motion-group evidence rendering for multi-motion tasks.
        "motion_group_evidence",
        "motion_group_seed_reduce",
    }
)
RUNTIME_KEYS = frozenset({"cycles", "resume", "dry_run", "evaluator"})
RUNTIME_EVALUATOR_KEYS = frozenset(
    {"launcher", "max_parallel", "slurm_host", "hq_submit_args", "slurm_sbatch_args"}
)

# Planner defaults that both the controller and the tracker label need before
# any cycle body runs.
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_PLANNER_TIMEOUT_SECONDS = 1800.0
DEFAULT_PLANNER_RETRY_DELAY_SECONDS = 15.0

SEED_BATCH_MODES = frozenset({"off", "auto", "require"})
EVAL_SOURCES = frozenset({"train_benchmark", "playback"})
COMPILE_CACHE_SCOPES = frozenset({"isolated", "task"})
RETRY_TRIGGERS = frozenset({"any_failure", "cublas", "worker_eof"})

# Keys the two controllers used before unification. They are errors now, with a
# pointer to their replacement, so a stale task file fails loudly instead of
# running with half its intent dropped.
RETIRED_TOP_LEVEL_KEYS: Mapping[str, str] = {
    "allowed_params": "use top-level `parameter_space`",
    "base_overrides": "use `batch.fixed_train_overrides`",
    "benchmark_ape_mean_threshold": "use `objective` keys and `batch.fixed_train_overrides`",
    "benchmark_auc_error_cap": "use `objective` keys",
    "benchmark_consecutive_passes_required": "use `batch.fixed_train_overrides`",
    "benchmark_cuda_graph": "use `batch.fixed_train_overrides`",
    "benchmark_deterministic_reset": "use `batch.fixed_train_overrides`",
    "benchmark_ee_mean_threshold": "use `objective` keys and `batch.fixed_train_overrides`",
    "benchmark_eval_interval": "use `batch.fixed_train_overrides`",
    "benchmark_eval_interval_seconds": "use `batch.fixed_train_overrides`",
    "benchmark_global_mpjpe_mean_threshold": "use `objective` keys and `batch.fixed_train_overrides`",
    "benchmark_include_root_in_ape": "use `batch.fixed_train_overrides`",
    "benchmark_metric_frame": "use `batch.fixed_train_overrides`",
    "benchmark_num_envs": "use `batch.fixed_train_overrides`",
    "benchmark_objective_mode": "use `objective.strategy`",
    "benchmark_objective_unit": "use `objective.strategy`",
    "benchmark_play_steps": "use `batch.fixed_train_overrides`",
    "benchmark_require_full_horizon": "use `batch.fixed_train_overrides`",
    "benchmark_success_rate_threshold": "use `objective` keys and `batch.fixed_train_overrides`",
    "codex": "use `planner`",
    "config_name": "use `batch.motions.<motion>.config_name`",
    "config_path": "use `batch.motions.<motion>.config_path`",
    "continue_without_human": "unread; delete it",
    "deep_review": "use `planner.deep_review`",
    "editable_files": "the controller owns every file it writes; delete it",
    "evaluator": "use `runtime.evaluator`",
    "frozen_params": "use top-level `forbidden_prefixes`",
    "frozen_params_extra": "use top-level `forbidden_prefixes`",
    "local_incumbent_advance": "use `objective.local_incumbent_advance`",
    "max_iterations": "use `batch.fixed_train_overrides`",
    "metric_improvement_margin": "use `objective.improvement_margin`",
    "objective_range_penalty_lambda": "use `objective.range_penalty_lambda`",
    "objective_timeout_seconds": "use `batch.fixed_train_overrides`",
    "output_dir": "the controller sets `output_root`",
    "parameter_manifest": "use top-level `parameter_space` and `forbidden_prefixes`",
    "params": "the planner proposes candidates; delete it",
    "planning": "use `planner`",
    "randomize_search_seed": "use `batch.seeds`",
    "repeats": "use `batch.seeds`",
    "retry_failures_per_repeat": "use `batch.retries_per_seed`",
    "reward_threshold": "use `objective` keys",
    "search_script": "the controller owns the evaluator; delete it",
    "seed": "use `batch.seeds`",
    "seed_batch": "use `batch.seed_batch`",
    "state_prompt": "use `planner`",
    "stop_on_fail": "use `batch.stop_on_fail`",
    "task_goal": "use `planner.instructions`",
    "train_script": "the controller owns the train command; delete it",
    "trial_timeout_seconds": "use `batch.trial_timeout_seconds`",
}
RETIRED_BATCH_KEYS: Mapping[str, str] = {
    "disable_default_play_overrides": "use `batch.playback.disable_default_overrides`",
    "fixed_overrides": "use `batch.fixed_train_overrides`",
    "forbidden_train_override_prefixes": "use top-level `forbidden_prefixes`",
    "play_overrides": "use `batch.playback.overrides`",
    "play_video": "use `batch.playback.video`",
    "play_video_length": "use `batch.playback.video_length`",
}
RETIRED_PLANNER_KEYS: Mapping[str, str] = {
    "deep_review_metric_context": "unread; delete it",
    "deep_review_metric_context_candidates": "unread; delete it",
    "deep_review_metric_context_cycles": "unread; delete it",
    "response_mode": "the planner always answers in a response; delete it",
}
# `metric_context` rendered the scalar evaluator's `search_summary` block, which
# the `params` sweep produced and the unified evaluator does not. The per-
# candidate numbers it summarized are in every recent-results line now.
RETIRED_DEEP_REVIEW_KEYS: Mapping[str, str] = {
    "metric_context": "unread; the recent-results lines carry per-candidate metrics",
    "metric_context_candidates": "unread; delete it",
    "metric_context_cycles": "use `planner.deep_review.results_rows`",
}
RETIRED_RUNTIME_KEYS: Mapping[str, str] = {
    "local_agent_timeout_seconds": "use `planner.timeout_seconds`",
    "model": "use `planner.model`",
}

DEEP_REVIEW_KEYS = frozenset({"interval_cycles", "results_rows"})
CREATIVITY_KEYS = frozenset({"interval_cycles", "plateau_cycles"})
SEARCH_BASE_SELECTION_KEYS = frozenset({"enabled", "max_options"})


@dataclass(frozen=True)
class RuntimeOptions:
    """The one runtime option set. Both modes run the same controller."""

    cycles: int | None
    dry_run: bool
    evaluator: EvaluatorLaunchConfig


def _is_mapping(value: object) -> bool:
    return isinstance(value, (DictConfig, dict))


def _is_sequence(value: object) -> bool:
    return isinstance(value, (ListConfig, list, tuple))


def _keys(value: object) -> list[str]:
    if isinstance(value, DictConfig):
        return [str(key) for key in value.keys()]
    if isinstance(value, dict):
        return [str(key) for key in value]
    return []


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, DictConfig):
        return value.get(key, default)
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def _check_unknown(
    errors: list[str],
    prefix: str,
    section: object,
    allowed: Iterable[str],
    retired: Mapping[str, str] = {},
) -> None:
    allowed_set = set(allowed)
    for key in _keys(section):
        if key in retired:
            errors.append(f"{prefix}.{key} is retired; {retired[key]}.")
        elif key not in allowed_set:
            errors.append(f"{prefix}.{key} is not a known key.")


def _check_str_list(
    errors: list[str], label: str, value: object, *, allow_empty: bool
) -> None:
    if not _is_sequence(value):
        errors.append(f"{label} must be a list of strings.")
        return
    items = list(value)
    if not items and not allow_empty:
        errors.append(f"{label} must not be empty.")
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{label}[{index}] must be a non-empty string.")


def _check_positive_int(
    errors: list[str], label: str, value: object, *, minimum: int = 1
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} must be an integer.")
        return
    if value < minimum:
        errors.append(f"{label} must be >= {minimum}.")


def _check_bool(errors: list[str], label: str, value: object) -> None:
    if value is not None and not isinstance(value, bool):
        errors.append(f"{label} must be a boolean.")


def _check_choice(
    errors: list[str], label: str, value: object, choices: Iterable[str]
) -> None:
    if value is None:
        return
    allowed = sorted(choices)
    if not isinstance(value, str) or value not in allowed:
        errors.append(f"{label} must be one of: {', '.join(allowed)}.")


def _validate_objective(errors: list[str], objective: object) -> None:
    if not _is_mapping(objective):
        errors.append("objective must be a mapping.")
        return
    mode = _get(objective, "mode")
    if not isinstance(mode, str) or mode not in STRATEGIES_BY_MODE:
        errors.append("objective.mode must be 'scalar' or 'pareto'.")
        return
    strategies = STRATEGIES_BY_MODE[mode]
    strategy = _get(objective, "strategy")
    if strategy is None:
        errors.append(
            f"objective.strategy is required; mode '{mode}' accepts: "
            f"{', '.join(strategies)}."
        )
        return
    if not isinstance(strategy, str):
        errors.append("objective.strategy must be a string.")
        return
    if strategy in strategies:
        return
    if strategy in UNIMPLEMENTED_SCALAR_STRATEGIES:
        errors.append(
            f"objective.strategy '{strategy}' is not implemented: "
            f"{UNIMPLEMENTED_SCALAR_STRATEGIES[strategy]}. It would silently "
            "rank by a different metric, so it is refused rather than guessed."
        )
        return
    other_mode = "pareto" if mode == "scalar" else "scalar"
    if strategy in STRATEGIES_BY_MODE[other_mode]:
        errors.append(
            f"objective.strategy '{strategy}' belongs to objective.mode "
            f"'{other_mode}', not '{mode}'."
        )
        return
    errors.append(
        f"objective.strategy '{strategy}' is unknown; mode '{mode}' accepts: "
        f"{', '.join(strategies)}."
    )


def _validate_motions(errors: list[str], motions: object) -> None:
    if not _is_mapping(motions):
        errors.append("batch.motions must be a mapping of motion name to config.")
        return
    names = _keys(motions)
    if not names:
        errors.append("batch.motions must define at least one motion.")
        return
    for name in names:
        entry = _get(motions, name)
        label = f"batch.motions.{name}"
        if not _is_mapping(entry):
            errors.append(f"{label} must be a mapping.")
            continue
        _check_unknown(errors, label, entry, MOTION_KEYS)
        for key in ("config_path", "config_name"):
            value = _get(entry, key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{label}.{key} must be a non-empty string.")


def _validate_batch(errors: list[str], batch: object) -> None:
    if not _is_mapping(batch):
        errors.append("batch must be a mapping.")
        return
    _check_unknown(errors, "batch", batch, BATCH_KEYS, RETIRED_BATCH_KEYS)
    _validate_motions(errors, _get(batch, "motions"))

    seeds = _get(batch, "seeds")
    if not _is_sequence(seeds) or not list(seeds):
        errors.append("batch.seeds must be a non-empty list of integers.")
    else:
        seed_list = list(seeds)
        for index, seed in enumerate(seed_list):
            if isinstance(seed, bool) or not isinstance(seed, int):
                errors.append(f"batch.seeds[{index}] must be an integer.")
        if len(set(map(str, seed_list))) != len(seed_list):
            errors.append("batch.seeds must not repeat a seed.")

    _check_choice(
        errors, "batch.seed_batch", _get(batch, "seed_batch"), SEED_BATCH_MODES
    )
    _check_choice(errors, "batch.eval_source", _get(batch, "eval_source"), EVAL_SOURCES)
    _check_choice(
        errors,
        "batch.compile_cache_scope",
        _get(batch, "compile_cache_scope"),
        COMPILE_CACHE_SCOPES,
    )
    _check_positive_int(errors, "batch.max_parallel", _get(batch, "max_parallel"))
    _check_positive_int(
        errors, "batch.retries_per_seed", _get(batch, "retries_per_seed"), minimum=0
    )
    _check_positive_int(
        errors,
        "batch.train_benchmark_recent_window",
        _get(batch, "train_benchmark_recent_window"),
    )
    _check_bool(errors, "batch.stop_on_fail", _get(batch, "stop_on_fail"))
    _check_bool(errors, "batch.lock_motion_configs", _get(batch, "lock_motion_configs"))

    trial_timeout = _get(batch, "trial_timeout_seconds")
    if trial_timeout is not None:
        if isinstance(trial_timeout, bool) or not isinstance(
            trial_timeout, (int, float)
        ):
            errors.append("batch.trial_timeout_seconds must be a number or null.")
        elif float(trial_timeout) <= 0.0:
            errors.append("batch.trial_timeout_seconds must be > 0 when set.")

    triggers = _get(batch, "retry_triggers")
    if triggers is not None:
        _check_str_list(errors, "batch.retry_triggers", triggers, allow_empty=True)
        if _is_sequence(triggers):
            for item in triggers:
                if isinstance(item, str) and item not in RETRY_TRIGGERS:
                    errors.append(
                        f"batch.retry_triggers entry '{item}' must be one of: "
                        f"{', '.join(sorted(RETRY_TRIGGERS))}."
                    )

    fixed = _get(batch, "fixed_train_overrides")
    if fixed is not None:
        _check_str_list(errors, "batch.fixed_train_overrides", fixed, allow_empty=True)

    minimum = _get(batch, "min_experiments_per_cycle")
    maximum = _get(batch, "max_experiments_per_cycle")
    _check_positive_int(errors, "batch.min_experiments_per_cycle", minimum)
    _check_positive_int(errors, "batch.max_experiments_per_cycle", maximum)
    if (
        isinstance(minimum, int)
        and isinstance(maximum, int)
        and not isinstance(minimum, bool)
        and not isinstance(maximum, bool)
        and minimum > maximum
    ):
        errors.append(
            "batch.min_experiments_per_cycle must not exceed "
            "batch.max_experiments_per_cycle."
        )

    playback = _get(batch, "playback")
    if playback is not None:
        if not _is_mapping(playback):
            errors.append("batch.playback must be a mapping.")
        else:
            _check_unknown(errors, "batch.playback", playback, PLAYBACK_KEYS)
            _check_bool(errors, "batch.playback.video", _get(playback, "video"))
            _check_bool(
                errors,
                "batch.playback.disable_default_overrides",
                _get(playback, "disable_default_overrides"),
            )
            _check_positive_int(
                errors, "batch.playback.video_length", _get(playback, "video_length")
            )
            overrides = _get(playback, "overrides")
            if overrides is not None:
                _check_str_list(
                    errors, "batch.playback.overrides", overrides, allow_empty=True
                )


def _validate_planner(errors: list[str], planner: object) -> None:
    if planner is None:
        return
    if not _is_mapping(planner):
        errors.append("planner must be a mapping.")
        return
    _check_unknown(errors, "planner", planner, PLANNER_KEYS, RETIRED_PLANNER_KEYS)
    _check_positive_int(
        errors, "planner.timeout_retries", _get(planner, "timeout_retries"), minimum=0
    )
    _check_positive_int(errors, "planner.recent_cycles", _get(planner, "recent_cycles"))
    _check_positive_int(
        errors,
        "planner.max_same_dimension_streak",
        _get(planner, "max_same_dimension_streak"),
    )
    for key in ("timeout_seconds", "retry_delay_seconds"):
        value = _get(planner, key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"planner.{key} must be a number.")
        elif float(value) < 0.0:
            errors.append(f"planner.{key} must be >= 0.")

    for key, allowed, retired in (
        ("deep_review", DEEP_REVIEW_KEYS, RETIRED_DEEP_REVIEW_KEYS),
        ("creativity", CREATIVITY_KEYS, {}),
        ("search_base_selection", SEARCH_BASE_SELECTION_KEYS, {}),
    ):
        section = _get(planner, key)
        if section is None:
            continue
        if not _is_mapping(section):
            errors.append(f"planner.{key} must be a mapping.")
            continue
        _check_unknown(errors, f"planner.{key}", section, allowed, retired)

    # The planner parses these when it builds its prompt; validating here means
    # a config audit catches a bad spelling instead of cycle 1 doing it.
    for parse in (parse_motion_group_evidence, parse_motion_group_seed_reduce):
        try:
            parse(planner)
        except ValueError as exc:
            errors.append(str(exc))

    candidates = _get(planner, "initial_candidates")
    if candidates is None:
        return
    if not _is_sequence(candidates):
        errors.append("planner.initial_candidates must be a list.")
        return
    for index, candidate in enumerate(candidates):
        label = f"planner.initial_candidates[{index}]"
        if not _is_mapping(candidate):
            errors.append(f"{label} must be a mapping.")
            continue
        name = _get(candidate, "name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{label}.name must be a non-empty string.")
        motions = _get(candidate, "motions")
        if not _is_mapping(motions):
            errors.append(f"{label}.motions must be a mapping.")
            continue
        for motion in _keys(motions):
            entry = _get(motions, motion)
            if not _is_mapping(entry):
                errors.append(f"{label}.motions.{motion} must be a mapping.")
                continue
            overrides = _get(entry, "overrides")
            if overrides is None:
                continue
            _check_str_list(
                errors,
                f"{label}.motions.{motion}.overrides",
                overrides,
                allow_empty=True,
            )


def _validate_runtime(errors: list[str], runtime: object) -> None:
    if runtime is None:
        return
    if not _is_mapping(runtime):
        errors.append("runtime must be a mapping.")
        return
    _check_unknown(errors, "runtime", runtime, RUNTIME_KEYS, RETIRED_RUNTIME_KEYS)
    _check_positive_int(errors, "runtime.cycles", _get(runtime, "cycles"))
    _check_bool(errors, "runtime.dry_run", _get(runtime, "dry_run"))
    evaluator = _get(runtime, "evaluator")
    if evaluator is None:
        return
    if not _is_mapping(evaluator):
        errors.append("runtime.evaluator must be a mapping.")
        return
    _check_unknown(errors, "runtime.evaluator", evaluator, RUNTIME_EVALUATOR_KEYS)


def validate_task(
    task_path: Path,
    task_cfg: DictConfig,
    runtime_cfg: DictConfig | None = None,
) -> list[str]:
    """Return every schema error in ``task_cfg``; an empty list means valid."""

    errors: list[str] = []
    if not _is_mapping(task_cfg):
        return [f"{task_path}: hpo task must be a mapping."]

    _check_unknown(
        errors,
        "task",
        task_cfg,
        TOP_LEVEL_KEYS | INJECTED_TOP_LEVEL_KEYS | {"runtime"},
        RETIRED_TOP_LEVEL_KEYS,
    )

    name = _get(task_cfg, "name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string.")
    description = _get(task_cfg, "description")
    if description is not None and not isinstance(description, str):
        errors.append("description must be a string.")

    _validate_objective(errors, _get(task_cfg, "objective"))
    _check_str_list(
        errors, "parameter_space", _get(task_cfg, "parameter_space"), allow_empty=False
    )
    forbidden = _get(task_cfg, "forbidden_prefixes")
    if forbidden is not None:
        _check_str_list(errors, "forbidden_prefixes", forbidden, allow_empty=True)
    _validate_batch(errors, _get(task_cfg, "batch"))
    _validate_planner(errors, _get(task_cfg, "planner"))
    _validate_runtime(
        errors,
        runtime_cfg if runtime_cfg is not None else _get(task_cfg, "runtime"),
    )

    return [f"{task_path}: {error}" for error in errors]


def resolve_runtime_options(runtime_cfg: DictConfig) -> RuntimeOptions:
    """Resolve ``runtime.*`` once for the controller loop."""

    cycles = _get(runtime_cfg, "cycles")
    if cycles in (None, ""):
        resolved_cycles = None
    else:
        resolved_cycles = int(cycles)
        if resolved_cycles < 1:
            raise ValueError("runtime.cycles must be >= 1.")

    dry_run_raw = _get(runtime_cfg, "dry_run", False)
    if isinstance(dry_run_raw, str):
        dry_run = dry_run_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        dry_run = bool(dry_run_raw)

    evaluator_cfg = _get(runtime_cfg, "evaluator")
    if evaluator_cfg is not None and not _is_mapping(evaluator_cfg):
        raise ValueError("runtime.evaluator must be a mapping.")
    launch_config = resolve_evaluator_launch_config(
        runtime_cfg,
        launcher=None,
        max_parallel=None,
    )

    return RuntimeOptions(
        cycles=resolved_cycles,
        dry_run=dry_run,
        evaluator=launch_config,
    )
