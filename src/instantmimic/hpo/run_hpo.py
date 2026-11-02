#!/usr/bin/env python3
"""HPO controller for scalar and Pareto objective modes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from instantmimic.hpo.controller_state import (
    controller_state_lock,
)
from instantmimic.hpo.cycle import run_one_cycle
from instantmimic.hpo.planner import planner_label
from instantmimic.hpo.run_context import (
    HPORunContext,
    hpo_config_path_for_name,
    prepare_hpo_runtime_task,
    split_hpo_runtime_cfg,
)
from instantmimic.hpo.schema import (
    RuntimeOptions,
    resolve_runtime_options,
    validate_task,
)
from instantmimic.hpo.tracking import HPOTracker
from instantmimic.utils.experiment_logging import load_experiment_dotenv
from instantmimic.utils.hydra_cli import run_hydra_without_outputs

REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = str(REPO_ROOT / "config")


@dataclass(frozen=True)
class CycleRunResult:
    returncode: int
    before_cycle: int
    after_cycle: int


def _objective_mode(task_cfg: DictConfig) -> str:
    """Read the mode for labelling. `validate_task` already checked it."""

    objective_cfg = task_cfg.get("objective", None)
    if not isinstance(objective_cfg, DictConfig):
        return ""
    return str(objective_cfg.get("mode", "") or "").strip().lower()


def _status_cycle(state_dir: Path) -> int:
    status_path = state_dir / "status.yaml"
    if not status_path.exists():
        return 0
    status = OmegaConf.load(status_path)
    if not isinstance(status, DictConfig):
        return 0
    return int(status.get("cycle", 0) or 0)


def run_controller_cycle(
    *,
    root: Path,
    context: HPORunContext,
    runtime_options: RuntimeOptions,
    tracker: HPOTracker,
) -> CycleRunResult:
    before_cycle = _status_cycle(context.state_dir)
    returncode = run_one_cycle(
        root=root,
        run_context=context,
        runtime_options=runtime_options,
        tracker=tracker,
    )
    after_cycle = _status_cycle(context.state_dir)
    return CycleRunResult(
        returncode=int(returncode),
        before_cycle=before_cycle,
        after_cycle=after_cycle,
    )


def _run_controller(
    *, task_path: Path, task_cfg: DictConfig, runtime_cfg: DictConfig
) -> int:
    mode = _objective_mode(task_cfg)
    schema_errors = validate_task(task_path, task_cfg, runtime_cfg)
    if schema_errors:
        print("\n".join(schema_errors), file=sys.stderr)
        return 2
    run_options = resolve_runtime_options(runtime_cfg)
    target_cycle = run_options.cycles
    dry_run = run_options.dry_run
    resume_raw = runtime_cfg.get("resume", None)
    resume = None if resume_raw in (None, "") else Path(str(resume_raw))
    context = prepare_hpo_runtime_task(
        root=REPO_ROOT,
        task_path=task_path,
        task_cfg=task_cfg,
        resume=resume,
    )
    print(f"[hpo] mode={mode} run_dir={context.run_dir or context.state_dir.parent}")

    with controller_state_lock(context.state_dir) as lock_acquired:
        if not lock_acquired:
            print(
                "[hpo] another controller is already running for "
                f"{context.state_dir}; exiting",
                file=sys.stderr,
            )
            return 2

        # The controller loop owns cycle iteration; every cycle body runs once.
        runtime_options = RuntimeOptions(
            cycles=1,
            dry_run=run_options.dry_run,
            evaluator=run_options.evaluator,
        )
        tracker = HPOTracker.from_task(
            root=REPO_ROOT,
            task_path=context.task_path,
            task_cfg=context.task_cfg,
            state_dir=context.state_dir,
            agent_label=planner_label(context.task_cfg),
            dry_run=dry_run,
        )

        executed_cycles = 0
        try:
            initial_cycle = _status_cycle(context.state_dir)
            if target_cycle is not None and initial_cycle >= target_cycle:
                print(
                    f"[hpo] cycle target {target_cycle} already reached "
                    f"(current={initial_cycle})."
                )
                return 0
            while (
                target_cycle is None or _status_cycle(context.state_dir) < target_cycle
            ):
                result = run_controller_cycle(
                    root=REPO_ROOT,
                    context=context,
                    runtime_options=runtime_options,
                    tracker=tracker,
                )
                if result.returncode != 0:
                    return result.returncode
                executed_cycles += 1
                if dry_run:
                    print("[hpo] dry-run inspection completed.")
                    return 0
                if result.after_cycle != result.before_cycle + 1:
                    print(
                        "[hpo] a cycle must advance the cycle index by "
                        f"exactly one; observed {result.before_cycle} -> "
                        f"{result.after_cycle}.",
                        file=sys.stderr,
                    )
                    return 2
        finally:
            tracker.close()

    print(
        f"[hpo] reached cycle {_status_cycle(context.state_dir)} "
        f"after executing {executed_cycles} cycle(s)"
    )
    return 0


def _hydra_config_name() -> str | None:
    if not HydraConfig.initialized():
        return None
    config_name = str(HydraConfig.get().job.config_name or "").strip()
    return config_name or None


def _run_hydra_main(cfg: DictConfig) -> int:
    try:
        config_name = _hydra_config_name()
        if config_name is None:
            raise ValueError(
                "Provide an hpo Hydra config with --config-name hpo/<name>."
            )
        task_path = hpo_config_path_for_name(REPO_ROOT, config_name)
        if not task_path.exists():
            raise FileNotFoundError(f"HPO task config does not exist: {task_path}")
        task_cfg, runtime_cfg = split_hpo_runtime_cfg(cfg)
        return _run_controller(
            task_path=task_path,
            task_cfg=task_cfg,
            runtime_cfg=runtime_cfg,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary reports task errors.
        print(str(exc), file=sys.stderr)
        return 2


@hydra.main(config_path=_CONFIG_DIR, config_name=None, version_base="1.3")
def main(cfg: DictConfig) -> None:
    raise SystemExit(_run_hydra_main(cfg))


def cli() -> None:
    load_experiment_dotenv(REPO_ROOT)
    run_hydra_without_outputs(main)


if __name__ == "__main__":
    cli()
