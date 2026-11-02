"""Synthetic hpo task configs for tests.

One builder for the one task schema. Tests that need a valid task ask for it
here and override only the part they are exercising, so a schema change lands
in a single place and no test depends on a committed ``config/`` variant.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

SCALAR_OBJECTIVE: dict[str, Any] = {
    "mode": "scalar",
    "strategy": "wall_clock",
    "improvement_margin": 0.0,
}
PARETO_OBJECTIVE: dict[str, Any] = {
    "mode": "pareto",
    "strategy": "learning_curve_pareto",
    "final_quality": {"key": "current_reward_mean", "epsilon": 0.0},
    "anytime_quality": {"epsilon": 0.0},
    "selection": "balanced",
}


def objective_for_mode(mode: str) -> dict[str, Any]:
    return dict(SCALAR_OBJECTIVE if mode == "scalar" else PARETO_OBJECTIVE)


def task_mapping(
    mode: str = "scalar",
    *,
    name: str = "synthetic-task",
    motion: str = "demo",
    motion_config_path: str = "config/demo",
    motion_config_name: str = "demo",
    seeds: tuple[int, ...] = (1, 2),
    **overrides: Any,
) -> dict[str, Any]:
    """Return a minimal task that passes ``schema.validate_task``."""

    task: dict[str, Any] = {
        "name": name,
        "description": "Synthetic task used by tests.",
        "objective": objective_for_mode(mode),
        "parameter_space": ["agent.learning_rate.initial"],
        "forbidden_prefixes": ["trainer", "experiment", "seed"],
        "batch": {
            "motions": {
                motion: {
                    "config_path": motion_config_path,
                    "config_name": motion_config_name,
                }
            },
            "seeds": list(seeds),
            "seed_batch": "off",
            "max_parallel": 1,
            "min_experiments_per_cycle": 1,
            "max_experiments_per_cycle": 1,
            "eval_source": "train_benchmark",
            "fixed_train_overrides": ["trainer.max_iterations=2"],
        },
        # No agent runs and no failure is retried in tests, so neither the
        # planner timeout nor the post-failure sleep should cost wall time.
        "planner": {
            "platform": "openai",
            "timeout_seconds": 1.0,
            "retry_delay_seconds": 0.0,
        },
        "hpo_tracking": {"backend": "none", "required": False},
    }
    for key, value in overrides.items():
        if value is None:
            task.pop(key, None)
            continue
        # `objective` is swapped whole so a test can state exactly which
        # comparator keys it means; other sections merge onto the defaults.
        if (
            key != "objective"
            and isinstance(value, dict)
            and isinstance(task.get(key), dict)
        ):
            task[key] = {**task[key], **value}
        else:
            task[key] = value
    return task


def task_cfg(mode: str = "scalar", **overrides: Any) -> DictConfig:
    return OmegaConf.create(task_mapping(mode, **overrides))


def runtime_mapping(**overrides: Any) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "cycles": None,
        "resume": None,
        "dry_run": False,
        "evaluator": {
            "launcher": None,
            "max_parallel": None,
            "slurm_host": None,
            "slurm_sbatch_args": None,
            "hq_submit_args": None,
        },
    }
    runtime.update(overrides)
    return runtime


def runtime_cfg(**overrides: Any) -> DictConfig:
    return OmegaConf.create(runtime_mapping(**overrides))
