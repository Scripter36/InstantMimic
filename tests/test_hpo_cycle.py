"""The one cycle runtime, driven in both objective modes.

Every test here runs the real controller with a fake agent and a fake
evaluator, so the phases, the lifecycle, and the state files are exercised for
real. The parametrization is the point: the same test body runs under
``objective.mode: scalar`` and ``objective.mode: pareto``, because after the
unification there is nothing left for the mode to change except the comparator.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest
import yaml
from config_fixtures import valid_config
from hpo_task_fixtures import task_mapping
from omegaconf import OmegaConf

from instantmimic.hpo import cycle as cycle_module
from instantmimic.hpo import planner as planner_module
from instantmimic.hpo.controller_state import (
    load_controller_status,
    read_controller_results,
    write_controller_status,
)
from instantmimic.hpo.evaluator_launch import EvaluatorLaunchConfig
from instantmimic.hpo.run_context import HPORunContext
from instantmimic.hpo.schema import RuntimeOptions
from instantmimic.hpo.tracking import HPOTracker

MODES = ["scalar", "pareto"]

STRATEGY = "\n".join(
    [
        "# Current Strategy",
        "## Best Baseline",
        "none",
        "## What Seems Promising",
        "none",
        "## What To Avoid",
        "none",
        "## Next Experiment",
        "none",
        "## Wildcards",
        "none",
        "",
    ]
)
PROPOSAL = json.dumps(
    {
        "base": "root",
        "rationale": "try a wider rollout",
        "experiments": [
            {
                "name": "wider",
                "rationale": "more rollouts",
                "motions": {
                    "demo": {"overrides": ["agent.learning_rate.initial=1e-4"]}
                },
            }
        ],
    }
)


def _objective_for(mode: str) -> dict[str, Any]:
    if mode == "scalar":
        return {"mode": "scalar", "strategy": "wall_clock"}
    # A complete objective: the comparator builds its own frontier points from
    # the evaluator's per-seed rows, so both axes need a key that the rows
    # actually carry. The half-specified version this used to hold only worked
    # because the fixture handed the comparator a pre-built frontier.
    return {
        "mode": "pareto",
        "strategy": "pareto",
        # `active_train_time` is the default key and a real `BENCHMARK_EVAL`
        # field (train/eval.py:1221).
        "time": {"epsilon": 0.0},
        "quality": {"key": "current_reward_mean", "goal": "maximize", "epsilon": 0.0},
        "selection": "knee",
    }


def write_evaluator_evidence(
    run_dir: pathlib.Path, mode: str = "scalar", *, wall_seconds: float = 12.0
) -> None:
    """Write what `evaluate.py` writes, in the shape it writes it.

    This is the seam where the controller meets the evaluator, so a
    fixture that invents its own shape hides exactly the bug it should
    catch: the comparators derive their numbers from these files, and
    a comparator reading a document nothing produces scores every
    cycle as a failure while every test stays green. The summary key
    set is asserted against the real writer in
    `test_the_fixture_matches_what_the_evaluator_writes`.
    """

    del mode
    run_dir.mkdir(parents=True, exist_ok=True)
    # `eval_index` is a real `BENCHMARK_EVAL` field (train/eval.py:1217) and the
    # frontier aggregation keys its per-seed alignment on it: a curve without
    # one yields no points at all, so the first cycle would report `dominated`
    # against an empty frontier.
    curve = [
        {
            "eval_index": 1.0,
            "frames": 1000.0,
            "current_reward_mean": 40.0,
            "active_train_time": 6.0,
        },
        {
            "eval_index": 2.0,
            "frames": 2000.0,
            "current_reward_mean": 80.0,
            "active_train_time": 12.0,
        },
    ]
    (run_dir / "trials.jsonl").write_text(
        json.dumps(
            {
                "candidate": "wider",
                "motion": "demo",
                "seed": 1,
                "attempt": 0,
                "returncode": 0,
                "wall_seconds": wall_seconds,
                "train": {"benchmark_curve": curve},
                "failure": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "evidence.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidates": [{"name": "wider", "complete": True}],
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "title": "synthetic-task",
        "task": str(run_dir / "task.yaml"),
        "proposal": str(run_dir / "proposal.yaml"),
        "output_root": str(run_dir),
        "best_experiment": "wider",
        "experiments": [
            {
                "name": "wider",
                "rationale": "",
                "aggregate": {"eval_source": "train_benchmark"},
                "motions": {},
                "runs": [
                    {
                        "experiment": "wider",
                        "motion": "demo",
                        "seed": 1,
                        "train_returncode": 0,
                        "benchmark_curve": curve,
                        "benchmark_threshold_frames": {},
                    }
                ],
            }
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture(name="harness")
def _harness(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Run the real controller with the agent and evaluator stubbed out."""

    class Harness:
        def __init__(self) -> None:
            self.root = tmp_path
            self.state_dir = tmp_path / "run" / "state"
            self.state_dir.mkdir(parents=True)
            self.task_path = tmp_path / "config" / "hpo" / "demo.yaml"
            self.task_path.parent.mkdir(parents=True)

            train_dir = tmp_path / "config" / "demo"
            train_dir.mkdir()
            (train_dir / "demo.yaml").write_text(yaml.safe_dump(valid_config()))
            (tmp_path / "docs" / "reference").mkdir(parents=True)
            (tmp_path / "docs" / "reference" / "hpo-task-schema.md").write_text(
                "# Program\n", encoding="utf-8"
            )
            self.agent_answers: dict[str, str] = {
                "planning": STRATEGY,
                "prepare": PROPOSAL,
            }
            self.agent_returncodes: dict[str, int] = {}
            self.phases: list[str] = []
            self.evaluator_returncode = 0
            self.evidence_writer = self._default_evidence
            self.logged_cycles: list[dict[str, Any]] = []
            # Lets a test interrupt exactly where the upload happens.
            self.tracker_hook: Any = None

        def _default_evidence(self, run_dir: pathlib.Path, mode: str) -> None:
            write_evaluator_evidence(run_dir, mode)

        def run(self, mode: str, *, dry_run: bool = False, cycles: int = 1) -> int:
            task = task_mapping(mode, objective=_objective_for(mode))
            task["state_dir"] = str(self.state_dir)
            task["output_root"] = str(tmp_path / "run" / "evaluator")
            self.task_path.write_text(
                yaml.safe_dump(task, sort_keys=False), encoding="utf-8"
            )
            task_cfg = OmegaConf.create(task)
            context = HPORunContext(
                task_path=self.task_path,
                task_cfg=task_cfg,
                run_dir=tmp_path / "run",
                state_dir=self.state_dir,
                resumed=False,
            )
            tracker = HPOTracker.disabled(
                self.root, self.task_path, task_cfg, self.state_dir, "test"
            )
            # Record what the controller hands the tracker, without enabling a
            # backend: a cycle that reaches disk must also reach tracking.
            logged = self.logged_cycles

            def log_cycle_result(**kwargs: Any) -> None:
                logged.append(kwargs)
                if self.tracker_hook is not None:
                    self.tracker_hook(**kwargs)

            tracker.log_cycle_result = log_cycle_result  # type: ignore[method-assign]
            returncode = 0
            for _ in range(cycles):
                returncode = cycle_module.run_one_cycle(
                    root=self.root,
                    run_context=context,
                    runtime_options=RuntimeOptions(
                        cycles=1,
                        dry_run=dry_run,
                        evaluator=EvaluatorLaunchConfig(),
                    ),
                    tracker=tracker,
                )
                if returncode != 0:
                    break
            return returncode

        def status(self) -> dict[str, Any]:
            return load_controller_status(self.state_dir / "status.yaml")

        def rows(self) -> list[dict[str, str]]:
            return read_controller_results(self.state_dir / "results.tsv")

    harness = Harness()

    def fake_run_phase(**kwargs: Any) -> int:
        phase = str(kwargs["phase_name"])
        harness.phases.append(phase)
        message_path = kwargs["message_path"]
        message_path.parent.mkdir(parents=True, exist_ok=True)
        message_path.write_text(harness.agent_answers.get(phase, ""), encoding="utf-8")
        return harness.agent_returncodes.get(phase, 0)

    def fake_run_evaluator(**kwargs: Any):
        run_dir = (
            kwargs["output_root"] / f"cycle_{int(kwargs['cycle_index']):03d}"
        ).resolve()
        harness.phases.append("evaluate")
        if harness.evaluator_returncode == 0:
            mode = str(kwargs["evaluator_task_path"].read_text(encoding="utf-8"))
            harness.evidence_writer(
                run_dir, "pareto" if "mode: pareto" in mode else "scalar"
            )
        log_path = kwargs["cycle_root"] / "evaluator.log"
        log_path.write_text("fake evaluator\n", encoding="utf-8")
        return harness.evaluator_returncode, log_path, run_dir

    monkeypatch.setattr(planner_module, "run_phase", fake_run_phase)
    monkeypatch.setattr(cycle_module.planner_module, "run_phase", fake_run_phase)
    monkeypatch.setattr(cycle_module, "_run_evaluator", fake_run_evaluator)
    return harness


@pytest.mark.parametrize("mode", MODES)
def test_resume_rejects_any_objective_change_before_planning(
    harness, monkeypatch, mode
):
    assert harness.run(mode) == 0
    original = _objective_for

    def changed_objective(selected_mode):
        objective = original(selected_mode)
        objective[
            "selection" if selected_mode == "pareto" else "improvement_margin"
        ] = "quality" if selected_mode == "pareto" else 100.0
        return objective

    monkeypatch.setattr(__import__(__name__), "_objective_for", changed_objective)
    harness.phases.clear()
    with pytest.raises(ValueError, match="objective.*changed|Objective.*changed"):
        harness.run(mode)
    assert harness.phases == []
    assert harness.status()["cycle"] == 1
    assert len(harness.rows()) == 1


class TestOneCycleRuntime:
    @pytest.mark.parametrize("mode", MODES)
    def test_both_modes_run_the_same_phases(self, harness, mode: str) -> None:
        assert harness.run(mode) == 0

        assert harness.phases == ["planning", "prepare", "evaluate"]

    @pytest.mark.parametrize("mode", MODES)
    def test_a_completed_cycle_advances_the_index(self, harness, mode: str) -> None:
        harness.run(mode)
        status = harness.status()

        assert status["cycle"] == 1
        assert status["active_cycle"] is None
        assert status["active_phase"] == ""
        assert status["last_status"] == "completed"

    @pytest.mark.parametrize("mode", MODES)
    def test_lifecycle_and_decision_are_recorded_separately(
        self, harness, mode: str
    ) -> None:
        harness.run(mode)
        rows = harness.rows()

        assert [row["status"] for row in rows] == ["completed"]
        assert rows[0]["decision"]
        assert rows[0]["decision"] != rows[0]["status"]
        assert rows[0]["mode"] == mode

    @pytest.mark.parametrize("mode", MODES)
    def test_one_row_per_candidate_with_its_parent_and_overrides(
        self, harness, mode: str
    ) -> None:
        harness.run(mode)
        rows = harness.rows()

        assert [row["candidate"] for row in rows] == ["wider"]
        assert rows[0]["parent"] == "root"
        assert json.loads(rows[0]["effective_overrides_json"]) == {
            "demo": ["agent.learning_rate.initial=1e-4"]
        }

    @pytest.mark.parametrize("mode", MODES)
    def test_every_state_file_is_written(self, harness, mode: str) -> None:
        harness.run(mode)

        for name in ("status.yaml", "results.tsv", "progress.md", "best_result.md"):
            assert (harness.state_dir / name).is_file()
        assert (harness.state_dir / "cycles" / "001" / "final.md").is_file()
        assert (harness.state_dir / "cycles" / "001" / "proposal.yaml").is_file()

    @pytest.mark.parametrize("mode", MODES)
    def test_progress_carries_the_documented_bullets(self, harness, mode: str) -> None:
        harness.run(mode)
        text = (harness.state_dir / "progress.md").read_text(encoding="utf-8")

        assert "## Cycle 1" in text
        for bullet in ("- Status:", "- Decision:", "- Proposal:", "- Evidence:"):
            assert bullet in text

    @pytest.mark.parametrize("mode", MODES)
    def test_the_proposal_is_persisted_for_the_record(self, harness, mode: str) -> None:
        harness.run(mode)
        proposal = yaml.safe_load(
            (harness.state_dir / "cycles" / "001" / "proposal.yaml").read_text(
                encoding="utf-8"
            )
        )

        assert proposal["base"] == "root"
        assert [item["name"] for item in proposal["experiments"]] == ["wider"]
        assert proposal["experiments"][0]["effective_overrides"] == {
            "demo": ["agent.learning_rate.initial=1e-4"]
        }


class TestFailureLifecycle:
    @pytest.mark.parametrize("mode", MODES)
    @pytest.mark.parametrize("phase", ["planning", "prepare"])
    def test_an_agent_failure_consumes_one_cycle(
        self, harness, mode: str, phase: str
    ) -> None:
        """A failed cycle is evidence, so it costs an index like any other."""

        harness.agent_returncodes[phase] = 7

        assert harness.run(mode) == 0
        status = harness.status()

        assert status["cycle"] == 1
        assert status["last_status"] == f"failed:{phase}"
        assert "exited with code 7" in status["last_error"]

    @pytest.mark.parametrize("mode", MODES)
    def test_a_malformed_proposal_fails_the_cycle_without_raising(
        self, harness, mode: str
    ) -> None:
        harness.agent_answers["prepare"] = "this is not JSON"

        assert harness.run(mode) == 0
        assert harness.status()["last_status"].startswith("failed:")

    @pytest.mark.parametrize("mode", MODES)
    def test_a_forbidden_override_is_rejected_before_anything_runs(
        self, harness, mode: str
    ) -> None:
        harness.agent_answers["prepare"] = json.dumps(
            {
                "base": "root",
                "experiments": [
                    {
                        "name": "bad",
                        "motions": {
                            "demo": {"overrides": ["trainer.max_iterations=1"]}
                        },
                    }
                ],
            }
        )

        assert harness.run(mode) == 0
        assert "evaluate" not in harness.phases
        assert harness.status()["last_status"] == "failed:validation"
        validation = (
            harness.state_dir / "cycles" / "001" / "validation.log"
        ).read_text(encoding="utf-8")
        assert "forbidden override" in validation

    @pytest.mark.parametrize("mode", MODES)
    def test_validation_feedback_reaches_the_next_cycles_prompt(
        self, harness, mode: str
    ) -> None:
        """The rejection has to be readable by the cycle that follows it.

        This used to read `cycles/<this>/validation.log`, which cannot exist
        yet -- the rejection consumed cycle N and the prompt for N+1 looked in
        N+1's own directory. Only `errors[0]` survived, through `last_error`.
        """

        rejected = json.dumps(
            {
                "base": "root",
                "experiments": [
                    {
                        "name": "bad",
                        "motions": {
                            "demo": {
                                "overrides": [
                                    "trainer.max_iterations=1",
                                    "experiment.logger.type=wandb",
                                ]
                            }
                        },
                    }
                ],
            }
        )
        harness.agent_answers["prepare"] = rejected
        assert harness.run(mode) == 0
        assert harness.status()["last_status"] == "failed:validation"

        harness.agent_answers["prepare"] = PROPOSAL
        assert harness.run(mode) == 0

        prepare_prompt = (
            harness.state_dir / "cycles" / "002" / "proposal.prompt.md"
        ).read_text(encoding="utf-8")
        assert "## Previous failures" in prepare_prompt
        # `status.last_error` carries only the first error, so the second one
        # can only be here if the full log was re-fed.
        assert "trainer.max_iterations=1" in prepare_prompt
        assert "experiment.logger.type=wandb" in prepare_prompt

    def test_per_motion_bases_are_not_added_to_common_fixed_settings(self, tmp_path):
        from instantmimic.hpo.experiment import OverrideSet

        task = OmegaConf.create(task_mapping("scalar"))
        base = {
            "walk": OverrideSet.of(["agent.learning_rate.initial=0.001"]),
            "run": OverrideSet.of(["agent.learning_rate.initial=0.002"]),
        }
        path = cycle_module._write_evaluator_task(
            task, base=base, experiment_directory="demo", cycle_root=tmp_path
        )
        fixed = list(OmegaConf.load(path).batch.fixed_train_overrides)
        assert fixed == ["trainer.max_iterations=2", "experiment.directory=demo"]

    @pytest.mark.parametrize("mode", MODES)
    def test_an_evaluator_failure_is_a_failed_cycle_not_a_crash_of_the_run(
        self, harness, mode: str
    ) -> None:
        harness.evaluator_returncode = 3

        assert harness.run(mode) == 0
        assert harness.status()["last_status"] == "failed:evaluate"


class TestDryRunAndResume:
    @pytest.mark.parametrize("mode", MODES)
    def test_a_dry_run_does_not_consume_a_cycle(self, harness, mode: str) -> None:
        assert harness.run(mode, dry_run=True) == 0
        status = harness.status()

        assert status["cycle"] == 0
        assert status["last_status"] == "dry_run"

    @pytest.mark.parametrize("mode", MODES)
    def test_an_interrupted_cycle_is_restarted_not_skipped(
        self, harness, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        """Ctrl-C leaves `active_cycle` set, so the resume redoes that cycle."""

        def interrupt(**_kwargs: Any) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(cycle_module.planner_module, "run_phase", interrupt)
        with pytest.raises(KeyboardInterrupt):
            harness.run(mode)

        status = harness.status()
        assert status["active_cycle"] == 1
        assert status["cycle"] == 0

    @pytest.mark.parametrize("mode", MODES)
    def test_a_second_cycle_follows_the_first(self, harness, mode: str) -> None:
        harness.run(mode, cycles=2)

        assert harness.status()["cycle"] == 2
        assert [row["cycle"] for row in harness.rows()] == ["1", "2"]


class TestDeterministicCandidates:
    @pytest.mark.parametrize("mode", MODES)
    def test_an_initial_candidate_skips_both_agent_turns(
        self, harness, tmp_path: pathlib.Path, mode: str
    ) -> None:
        """A configured first cycle is deterministic, so no agent is asked."""

        task = task_mapping(mode, objective=_objective_for(mode))
        task["state_dir"] = str(harness.state_dir)
        task["output_root"] = str(tmp_path / "run" / "evaluator")
        task["planner"] = {
            **task["planner"],
            "initial_candidates": [
                {
                    "name": "untouched_baseline",
                    "rationale": "establish the baseline",
                    "motions": {"demo": {"overrides": []}},
                }
            ],
        }
        harness.task_path.write_text(
            yaml.safe_dump(task, sort_keys=False), encoding="utf-8"
        )
        task_cfg = OmegaConf.create(task)
        context = HPORunContext(
            task_path=harness.task_path,
            task_cfg=task_cfg,
            run_dir=tmp_path / "run",
            state_dir=harness.state_dir,
            resumed=False,
        )
        tracker = HPOTracker.disabled(
            harness.root, harness.task_path, task_cfg, harness.state_dir, "test"
        )

        returncode = cycle_module.run_one_cycle(
            root=harness.root,
            run_context=context,
            runtime_options=RuntimeOptions(
                cycles=1, dry_run=False, evaluator=EvaluatorLaunchConfig()
            ),
            tracker=tracker,
        )

        assert returncode == 0
        assert harness.phases == ["evaluate"]
        assert [row["candidate"] for row in harness.rows()] == ["untouched_baseline"]


class TestEveryCycleReachesTracking:
    """A failed cycle used to leave a hole in the metric series.

    `tracker.log_cycle_result` was called on the success path only, so a run
    that failed three cycles logged nothing for them: the step index jumped,
    and `hpo.status_failed` was never 1.0 for a real failure -- the
    one signal a plateau watcher would look for. Logging now happens inside
    `_record_cycle`, so a cycle cannot be written to disk without being logged.
    """

    @pytest.mark.parametrize("mode", MODES)
    def test_a_completed_cycle_is_logged(self, harness, mode: str) -> None:
        assert harness.run(mode) == 0

        assert len(harness.logged_cycles) == 1
        logged = harness.logged_cycles[0]
        assert logged["outcome"]["status"] == "completed"
        assert logged["outcome"]["decision"]
        assert logged["cycle_index"] == 1

    @pytest.mark.parametrize("mode", MODES)
    def test_a_failed_cycle_is_logged_too(self, harness, mode: str) -> None:
        harness.agent_returncodes["planning"] = 3

        assert harness.run(mode) == 0

        assert len(harness.logged_cycles) == 1
        logged = harness.logged_cycles[0]
        assert logged["outcome"]["status"] == "failed:planning"
        assert logged["outcome"]["decision"] == ""

    @pytest.mark.parametrize("mode", MODES)
    def test_the_cycle_artifacts_are_offered_to_the_tracker(
        self, harness, mode: str
    ) -> None:
        """Every artifact path used to be passed as None, so none uploaded."""

        assert harness.run(mode) == 0

        logged = harness.logged_cycles[0]
        for key in (
            "planning_message_path",
            "prepare_message_path",
            "final_message_path",
            "summary_path",
            "trials_path",
        ):
            assert logged[key] is not None, key
            assert pathlib.Path(logged[key]).is_file(), key
        # Nothing was rejected this cycle, so there is no validation log.
        assert logged["validation_log_path"] is None


class TestAChangedObjectiveStopsTheRun:
    """Resuming with a different objective is fatal, not a bad cycle.

    The archive refuses to merge points measured against a different objective,
    but it only discovers that on insertion -- after the cycle has trained. The
    generic `except ValueError` around finalize turned that into one failed
    cycle, so a 40-cycle run would train 40 times to reach the same answer 40
    times. The strategy itself needs no evidence, so it is checked before the
    cycle starts; anything only detectable on insertion now stops the run.
    """

    def test_a_different_strategy_is_refused_before_any_training(
        self, tmp_path: pathlib.Path
    ) -> None:
        from instantmimic.hpo.objective import build_comparator
        from instantmimic.hpo.objective.base import (
            IncompatibleObjectiveError,
        )

        state_dir = tmp_path / "state"
        (state_dir / "objective").mkdir(parents=True)
        (state_dir / "objective" / "pareto_frontier.json").write_text(
            json.dumps({"mode": "learning_curve", "frontier": []}), encoding="utf-8"
        )

        comparator = build_comparator(
            {"mode": "pareto", "strategy": "axes_pareto"}, state_dir=state_dir
        )
        comparator.load_state(state_dir / "objective", {})

        with pytest.raises(IncompatibleObjectiveError, match="different objective"):
            comparator.check_compatible()

    def test_a_matching_strategy_and_a_fresh_run_both_pass(
        self, tmp_path: pathlib.Path
    ) -> None:
        from instantmimic.hpo.objective import build_comparator

        state_dir = tmp_path / "state"
        (state_dir / "objective").mkdir(parents=True)
        comparator = build_comparator(
            {"mode": "pareto", "strategy": "learning_curve_pareto"},
            state_dir=state_dir,
        )
        comparator.load_state(state_dir / "objective", {})
        # No archive yet: nothing to be incompatible with.
        comparator.check_compatible()

        (state_dir / "objective" / "pareto_frontier.json").write_text(
            json.dumps({"mode": "learning_curve", "frontier": []}), encoding="utf-8"
        )
        comparator.check_compatible()

    def test_the_controller_stops_rather_than_consuming_the_budget(
        self, harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from instantmimic.hpo.objective.base import (
            IncompatibleObjectiveError,
        )
        from instantmimic.hpo.objective.frontier import FrontierComparator

        def refuse(self, evidence):
            raise IncompatibleObjectiveError("axes changed while resuming")

        monkeypatch.setattr(FrontierComparator, "compare", refuse)

        # Non-zero stops `run_hpo`'s loop; a failed cycle returns 0.
        assert harness.run("pareto") == 2
        assert harness.status()["last_status"] == "failed:finalize"
        assert "axes changed" in harness.status()["last_error"]


class TestResumeAndThePlateauCounter:
    """Two behaviours the suite named but never exercised.

    `active_cycle` is the resume pointer: an interrupted cycle is restarted at
    its own index rather than skipped. `cycles_since_improvement` is what the
    prompt calls a plateau on. Both survived being mutated away, because every
    test either asserted the state an interrupt leaves behind without ever
    resuming from it, or passed the counter in as a literal.
    """

    @pytest.mark.parametrize("mode", MODES)
    def test_an_interrupted_cycle_is_resumed_at_its_own_index(
        self, harness, mode: str
    ) -> None:
        harness.run(mode)  # cycle 1 completes, so `cycle` is 1
        status = harness.status()
        status["active_cycle"] = 2
        status["active_phase"] = "evaluate"
        write_controller_status(harness.state_dir / "status.yaml", status, mode=mode)

        assert harness.run(mode) == 0

        # Cycle 2 is redone, not skipped to 3.
        assert harness.status()["cycle"] == 2
        assert [int(row["cycle"]) for row in harness.rows()] == [1, 2]
        assert (harness.state_dir / "cycles" / "002").is_dir()
        assert not (harness.state_dir / "cycles" / "003").exists()

    @pytest.mark.parametrize("mode", MODES)
    def test_ctrl_c_after_a_cycle_is_recorded_does_not_redo_it(
        self, harness, mode: str
    ) -> None:
        """The window between recording a cycle and finishing its upload.

        `_record_cycle` writes the row, advances `cycle`, and clears
        `active_cycle`; the tracker upload that follows can take a while. A
        Ctrl-C landing there used to re-mark the finished cycle active, so the
        next run redid it, appended a second row for the same cycle, and then
        exited 2 on "a cycle must advance the cycle index by exactly one".
        """

        assert harness.run(mode) == 0

        def interrupt(**_kwargs: Any) -> None:
            raise KeyboardInterrupt

        harness.tracker_hook = interrupt
        with pytest.raises(KeyboardInterrupt):
            harness.run(mode)

        # Cycle 2 was recorded before the interrupt, so it must not be redone.
        assert harness.status()["cycle"] == 2
        assert harness.status()["active_cycle"] is None
        assert [int(row["cycle"]) for row in harness.rows()] == [1, 2]

        harness.tracker_hook = None
        assert harness.run(mode) == 0
        assert [int(row["cycle"]) for row in harness.rows()] == [1, 2, 3]

    @pytest.mark.parametrize("mode", MODES)
    def test_cycles_since_improvement_accumulates_and_resets(
        self, harness, mode: str
    ) -> None:
        assert harness.run(mode) == 0
        # The first cycle is always an improvement: nothing to beat.
        assert harness.status()["cycles_since_improvement"] == 0

        # A cycle the planner cannot complete does not improve anything.
        harness.agent_returncodes["planning"] = 3
        harness.run(mode)
        assert harness.status()["cycles_since_improvement"] == 1
        harness.run(mode)
        assert harness.status()["cycles_since_improvement"] == 2

        harness.agent_returncodes.pop("planning")
        # A genuinely faster result: the fixture otherwise replays the same
        # number, which ties the previous best and is correctly discarded.
        harness.evidence_writer = lambda run_dir, mode: write_evaluator_evidence(
            run_dir, mode, wall_seconds=6.0
        )
        harness.agent_answers["prepare"] = json.dumps(
            {
                "base": "root",
                "experiments": [
                    {
                        "name": "different",
                        "motions": {
                            "demo": {"overrides": ["agent.learning_rate.initial=5e-5"]}
                        },
                    }
                ],
            }
        )
        harness.run(mode)

        assert harness.status()["cycles_since_improvement"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_invalid_composed_train_setting_fails_before_evaluation(harness, mode):
    response = json.loads(PROPOSAL)
    response["experiments"][0]["motions"]["demo"]["overrides"] = [
        "agent.learning_rate.initial=-1"
    ]
    harness.agent_answers["prepare"] = json.dumps(response)
    assert harness.run(mode) == 0
    assert harness.status()["last_status"] == "failed:validation"
    assert "evaluate" not in harness.phases
    assert (
        "learning_rate" in (harness.state_dir / "cycles/001/validation.log").read_text()
    )


@pytest.mark.parametrize("mode", MODES)
def test_interrupt_before_status_commit_resumes_without_duplicate_results(
    harness, monkeypatch, mode
):
    write_status = cycle_module.write_controller_status

    def interrupted_write(path, payload, **kwargs):
        if payload["cycle"] == 1 and payload["active_cycle"] is None:
            raise KeyboardInterrupt
        return write_status(path, payload, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cycle_module, "write_controller_status", interrupted_write)
        with pytest.raises(KeyboardInterrupt):
            harness.run(mode)
    assert harness.status()["cycle"] == 0
    assert harness.run(mode) == 0
    assert harness.status()["cycle"] == 1
    assert harness.status()["last_status"] == "completed"
    assert harness.status()["last_decision"] == (
        "keep" if mode == "scalar" else "frontier_changed"
    )
    assert len(harness.rows()) == 1
    assert (harness.state_dir / "progress.md").read_text().count("## Cycle 1") == 1


def test_duplicate_override_sets_ignore_prefix_and_order():
    from instantmimic.hpo.experiment import Candidate, OverrideSet, RootSpec

    root = RootSpec(
        motions={},
        fixed_overrides=OverrideSet(),
        seeds=(1,),
        parameter_space=("agent.a", "agent.b"),
        forbidden_prefixes=(),
    )
    first = Candidate(
        name="first", effective={"demo": OverrideSet.of(["agent.a=1", "++agent.b=2"])}
    )
    second = Candidate(
        name="second", effective={"demo": OverrideSet.of(["agent.b=2", "+agent.a=1"])}
    )
    cfg = OmegaConf.create({"batch": {"max_experiments_per_cycle": 2}})
    errors = cycle_module._validate_candidates((first, second), root, cfg, ())
    assert any("repeats" in error for error in errors)
