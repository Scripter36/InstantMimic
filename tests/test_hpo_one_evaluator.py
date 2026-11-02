"""One evaluator, one evidence schema, for both objective modes.

The point of the unification is that ``objective.mode`` cannot change what
runs. These tests take one task body, vary only its ``objective`` block, and
assert that the planned trials and the recorded evidence come out identical --
which is the property that lets a single comparator interface sit on top.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any

import pytest
import test_hpo_cycle as _cycle_tests
import yaml
from hpo_task_fixtures import objective_for_mode

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
EVALUATE = "src/instantmimic/hpo/evaluate.py"

BATCH: dict[str, Any] = {
    "motions": {
        "ant": {"config_path": "config/ant", "config_name": "isaac_ant_direct"}
    },
    "seeds": [1, 2],
    "seed_batch": "require",
    "max_parallel": 1,
    "min_experiments_per_cycle": 1,
    "max_experiments_per_cycle": 2,
    "eval_source": "train_benchmark",
    "lock_motion_configs": True,
    "retries_per_seed": 2,
    "retry_triggers": ["any_failure"],
    "trial_timeout_seconds": 90,
    "stop_on_fail": True,
    "fixed_train_overrides": ["trainer.max_iterations=2"],
}
EXPERIMENTS = [
    {
        "name": "a",
        "parent": "root",
        "motions": {"ant": {"overrides": ["agent.rollouts=16"]}},
    },
    {
        "name": "b",
        "parent": [1, "a"],
        "motions": {"ant": {"overrides": ["agent.rollouts=32"]}},
    },
]
# A Pareto objective needs a budget and quality axes; a scalar one needs
# neither. Everything outside this block is identical in both tasks.
PARETO_OBJECTIVE = {
    **objective_for_mode("pareto"),
    "active_time_budget_seconds": 20.0,
    "terminal_time_tolerance_seconds": 0.25,
    "final_quality": {
        "key": "current_reward_mean",
        "aggregate": "median",
        "statistic": "best_per_seed",
        "epsilon": 0.0,
    },
    "anytime_quality": {"epsilon": 0.0},
    "guards": [],
}


def _task(objective: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "demo",
        "parameter_space": ["agent.rollouts"],
        "forbidden_prefixes": ["trainer", "seed"],
        "batch": BATCH,
        "objective": objective,
    }


def _evaluate(tmp_path: pathlib.Path, label: str, objective: dict[str, Any]):
    task_path = tmp_path / f"task_{label}.yaml"
    task = _task(objective)
    config_dir = tmp_path / "train-config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "synthetic.yaml").write_text("defaults: [_self_]\n")
    task["batch"] = {
        **BATCH,
        "motions": {
            "ant": {"config_path": str(config_dir), "config_name": "synthetic"}
        },
    }
    task_path.write_text(yaml.safe_dump(task), encoding="utf-8")
    proposal_path = tmp_path / "proposal.yaml"
    proposal_path.write_text(
        yaml.safe_dump({"base": "root", "experiments": EXPERIMENTS}), encoding="utf-8"
    )
    output_root = tmp_path / f"cycle_{label}"
    completed = subprocess.run(
        [
            sys.executable,
            EVALUATE,
            "--task",
            str(task_path),
            "--proposal",
            str(proposal_path),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return output_root, completed.stdout


def _trials(output_root: pathlib.Path) -> list[dict[str, Any]]:
    lines = (output_root / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    for record in records:
        # Timing and the output directory are the only per-run differences.
        record.pop("wall_seconds", None)
        record["train_log"] = str(
            pathlib.Path(record["train_log"]).relative_to(output_root)
        )
    return records


@pytest.fixture(name="both_modes")
def _both_modes(tmp_path: pathlib.Path):
    pareto_root, pareto_stdout = _evaluate(tmp_path, "pareto", PARETO_OBJECTIVE)
    scalar_root, scalar_stdout = _evaluate(
        tmp_path, "scalar", objective_for_mode("scalar")
    )
    return (pareto_root, pareto_stdout), (scalar_root, scalar_stdout)


class TestBothModesPlanTheSameWork:
    def test_trial_records_are_identical(self, both_modes) -> None:
        (pareto_root, _), (scalar_root, _) = both_modes

        assert _trials(pareto_root) == _trials(scalar_root)

    def test_planned_train_commands_are_identical(self, both_modes) -> None:
        (pareto_root, _), (scalar_root, _) = both_modes

        pareto_commands = [record["train_command"] for record in _trials(pareto_root)]
        scalar_commands = [record["train_command"] for record in _trials(scalar_root)]

        assert pareto_commands == scalar_commands
        assert len(pareto_commands) == len(BATCH["seeds"]) * len(EXPERIMENTS)

    def test_evidence_differs_only_in_the_task_it_names(self, both_modes) -> None:
        (pareto_root, _), (scalar_root, _) = both_modes

        pareto = json.loads((pareto_root / "evidence.json").read_text())
        scalar = json.loads((scalar_root / "evidence.json").read_text())
        for payload in (pareto, scalar):
            payload.pop("task")
            payload.pop("output_root")

        assert pareto == scalar


class TestEvidenceContract:
    def test_execution_settings_are_recorded_as_resolved(self, both_modes) -> None:
        (pareto_root, _), _ = both_modes

        execution = json.loads((pareto_root / "evidence.json").read_text())["execution"]

        assert execution == {
            "retries_per_seed": 2,
            "retry_triggers": ["any_failure"],
            "trial_timeout_seconds": 90.0,
            "stop_on_fail": True,
            "max_parallel": 1,
            "seeds": [1, 2],
        }

    def test_each_candidate_records_its_parent_and_trials(self, both_modes) -> None:
        (pareto_root, _), _ = both_modes

        candidates = json.loads((pareto_root / "evidence.json").read_text())[
            "candidates"
        ]

        assert [candidate["name"] for candidate in candidates] == ["a", "b"]
        assert [candidate["parent"] for candidate in candidates] == [
            "root",
            [1, "a"],
        ]
        assert candidates[0]["trial_ids"] == ["a/ant/1", "a/ant/2"]
        assert candidates[0]["effective_overrides"] == {"ant": ["agent.rollouts=16"]}

    def test_a_dry_run_completes_no_candidate(self, both_modes) -> None:
        (pareto_root, _), _ = both_modes

        payload = json.loads((pareto_root / "evidence.json").read_text())

        assert payload["dry_run"] is True
        assert [candidate["complete"] for candidate in payload["candidates"]] == [
            False,
            False,
        ]

    def test_the_evaluator_does_not_interpret_the_objective(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Comparison moved to the controller; the evaluator must not depend on it.

        An objective block the evaluator could never score used to fail the run
        and cost the execution record. It now runs to completion and writes the
        same evidence, because nothing in the evaluator reads it.
        """

        opaque = {"mode": "pareto", "strategy": "learning_curve_pareto"}
        output_root, _ = _evaluate(tmp_path, "opaque", opaque)

        payload = json.loads((output_root / "evidence.json").read_text())

        assert len(_trials(output_root)) == len(BATCH["seeds"]) * len(EXPERIMENTS)
        assert [candidate["name"] for candidate in payload["candidates"]] == ["a", "b"]


class TestTheControllerFixtureMatchesTheRealEvaluator:
    """The seam that hid a mode-wide failure for a whole refactor.

    `test_hpo_cycle.py` stubs the evaluator and writes `summary.json`
    itself. While that stub invented its own shape, the scalar comparator read
    a document nothing produced and scored every completed cycle as a crash --
    with all 597 tests green, because the only end-to-end test agreed with the
    comparator instead of with the evaluator. This asserts the stub against the
    real writer, so the two cannot drift apart again.
    """

    def test_the_fixture_writes_the_keys_the_evaluator_writes(
        self, tmp_path: pathlib.Path
    ) -> None:
        output_root, _ = _evaluate(tmp_path, "shape", objective_for_mode("scalar"))
        real = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))

        stub_root = tmp_path / "stub"
        _cycle_tests.write_evaluator_evidence(stub_root)
        stub = json.loads((stub_root / "summary.json").read_text(encoding="utf-8"))

        assert set(stub) == set(real)
        assert set(stub["experiments"][0]) == set(real["experiments"][0])

    def test_the_fixture_trial_record_matches_the_real_one(
        self, tmp_path: pathlib.Path
    ) -> None:
        output_root, _ = _evaluate(tmp_path, "trialshape", objective_for_mode("scalar"))
        # The raw line, not `_trials`: that helper strips `wall_seconds` so it
        # can compare runs across modes, which is precisely why a missing
        # `wall_seconds` once went unnoticed until a GPU run.
        real = json.loads(
            (output_root / "trials.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )

        stub_root = tmp_path / "stub_trials"
        _cycle_tests.write_evaluator_evidence(stub_root)
        stub = json.loads(
            (stub_root / "trials.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )

        # The stub may carry fewer fields, but never a field the evaluator
        # does not write -- that is how an invented contract creeps back in.
        assert set(stub) <= set(real), sorted(set(stub) - set(real))
        for required in ("candidate", "seed", "returncode", "wall_seconds", "train"):
            assert required in stub
