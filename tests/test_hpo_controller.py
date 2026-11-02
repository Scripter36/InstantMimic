from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from hpo_task_fixtures import runtime_cfg, task_mapping
from omegaconf import OmegaConf


def test_runtime_paths_use_canonical_task_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from instantmimic.hpo import cycle
    from instantmimic.hpo.run_context import (
        prepare_hpo_runtime_task,
    )

    del monkeypatch
    task_path = tmp_path / "config" / "hpo" / "fallback.yaml"
    task_cfg = OmegaConf.create({"name": "../Mixed-Case Task"})

    context = prepare_hpo_runtime_task(
        root=tmp_path,
        task_path=task_path,
        task_cfg=task_cfg,
        timestamp_name="2026-09-03_12-00-00",
    )
    expected_run_dir = (
        tmp_path / "logs" / "hpo" / "mixed_case_task" / "2026-09-03_12-00-00"
    ).resolve()

    assert context.run_dir == expected_run_dir
    assert context.state_dir == expected_run_dir / "state"
    assert cycle._resolve_output_root(tmp_path, context.task_cfg, task_path) == (
        expected_run_dir / "evaluator"
    )
    context.state_dir.relative_to(tmp_path)
    assert not (tmp_path / "workspaces").exists()

    # A task without a run directory yet falls back inside the repository.
    default_output_root = cycle._resolve_output_root(tmp_path, task_cfg, task_path)

    assert default_output_root == (tmp_path / "logs/hpo/mixed_case_task/evaluator")


def test_objective_engine_selects_scalar_minimum() -> None:
    from instantmimic.hpo.objective.selection import (
        ObjectiveAxis,
        ObjectiveSpec,
        select_objective,
    )

    result = select_objective(
        [
            {"experiment": "worse", "loss": 2.0},
            {"experiment": "best", "loss": 1.0},
        ],
        ObjectiveSpec(
            mode="scalar",
            axes=(ObjectiveAxis(key="loss", goal="minimize"),),
        ),
    )

    assert result["selected"]["experiment"] == "best"
    assert [point["experiment"] for point in result["frontier"]] == ["best"]


def test_objective_engine_selects_epsilon_pareto_frontier() -> None:
    from instantmimic.hpo.objective.selection import (
        ObjectiveAxis,
        ObjectiveSpec,
        select_objective,
    )

    result = select_objective(
        [
            {"experiment": "final", "final": 1.0, "auc": 2.0},
            {"experiment": "anytime", "final": 2.0, "auc": 1.0},
            {"experiment": "dominated", "final": 3.0, "auc": 3.0},
        ],
        ObjectiveSpec(
            mode="pareto",
            axes=(
                ObjectiveAxis(key="final", goal="minimize"),
                ObjectiveAxis(key="auc", goal="minimize"),
            ),
            selection="balanced",
        ),
    )

    assert {point["experiment"] for point in result["frontier"]} == {
        "final",
        "anytime",
    }
    assert result["selected"]["experiment"] in {"final", "anytime"}


def test_batch_seed_runner_reuses_one_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from instantmimic.hpo import evaluate as evaluator

    task_cfg = OmegaConf.create(
        {
            "name": "test",
            "batch": {
                "seeds": [1, 2, 3],
                "max_parallel": 1,
                "seed_batch": "require",
                "eval_source": "train_benchmark",
                "fixed_train_overrides": [],
                "play_overrides": [],
            },
            "objective": {
                "strategy": "weighted_sum",
                "metrics": [{"key": "mean_reward", "goal": "maximize"}],
            },
        }
    )
    experiment = evaluator.ExperimentSpec(
        name="candidate",
        rationale="test",
        motions={
            "motion": evaluator.MotionRunSpec(
                motion_name="motion",
                config_path=Path("/tmp/config"),
                config_name="task",
                overrides=(),
            )
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_resolve_config_spec",
        lambda *_args: SimpleNamespace(
            config_path=Path("/tmp/config"),
            config_name="task",
        ),
    )
    monkeypatch.setattr(evaluator.run_discovery, "list_run_dirs", lambda _path: set())
    monkeypatch.setattr(
        evaluator.run_discovery,
        "find_newest_run_dir",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        evaluator,
        "_populate_train_benchmark_metrics",
        lambda job, **_kwargs: setattr(job, "benchmark_eval_count", 1),
    )

    workers: list[object] = []

    class FakeWorker:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.seeds: list[int] = []
            workers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def run_trial(self, **kwargs):
            overrides = kwargs["overrides"]
            seed = int(
                next(
                    item.split("=", 1)[1]
                    for item in overrides
                    if item.startswith("seed=")
                )
            )
            self.seeds.append(seed)
            kwargs["log_file"].write_text(f"seed={seed}\n", encoding="utf-8")
            return (1.0, None, 0, "start", "finish")

    monkeypatch.setattr(evaluator, "SeedTrainWorker", FakeWorker, raising=False)
    monkeypatch.setattr(
        evaluator.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "persistent seed batching must not spawn one train process per seed"
            )
        ),
    )

    jobs = evaluator._run_jobs_for_experiment(
        task_cfg, experiment, tmp_path, dry_run=False
    )

    assert len(workers) == 1
    assert workers[0].seeds == [1, 2, 3]
    assert [job.seed for job in jobs] == [1, 2, 3]
    assert all(job.train_returncode == 0 for job in jobs)


def test_controller_mode_is_explicit_not_inferred_from_batch_shape() -> None:
    """The mode is read, never guessed, and never inferred from the shape.

    A task's shape used to imply its mode: a `batch:` section meant Pareto and
    its absence meant scalar. Both modes have a `batch:` now, so the only
    statement of the mode is `objective.mode`, and the schema is what rejects a
    missing or unknown one.
    """

    from instantmimic.hpo import run_hpo as controller
    from instantmimic.hpo.schema import validate_task

    assert (
        controller._objective_mode(
            OmegaConf.create({"objective": {"mode": "scalar"}, "batch": {}})
        )
        == "scalar"
    )
    assert (
        controller._objective_mode(OmegaConf.create({"objective": {"mode": "pareto"}}))
        == "pareto"
    )
    assert controller._objective_mode(OmegaConf.create({"batch": {}})) == ""

    errors = validate_task(Path("task.yaml"), OmegaConf.create({"batch": {}}))
    assert any("objective must be a mapping" in error for error in errors)
    errors = validate_task(
        Path("task.yaml"), OmegaConf.create({"objective": {"strategy": "wall_clock"}})
    )
    assert any("objective.mode must be" in error for error in errors)


@pytest.mark.parametrize("mode", ["scalar", "pareto"])
def test_controller_owns_cycle_loop_for_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from instantmimic.hpo import run_hpo as controller

    calls: list[tuple[str, int]] = []
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(task_mapping(mode))), encoding="utf-8"
    )

    def fake_prepare(**kwargs):
        del kwargs
        return SimpleNamespace(
            task_path=task_path,
            task_cfg=OmegaConf.create(task_mapping(mode, state_dir=str(state_dir))),
            run_dir=tmp_path,
            state_dir=state_dir,
        )

    def fake_cycle(**kwargs):
        mode = str(kwargs["context"].task_cfg.objective.mode)
        cycle = len(calls) + 1
        calls.append((mode, cycle))
        (state_dir / "status.yaml").write_text(f"cycle: {cycle}\n", encoding="utf-8")
        return controller.CycleRunResult(
            returncode=0,
            before_cycle=cycle - 1,
            after_cycle=cycle,
        )

    monkeypatch.setattr(controller, "prepare_hpo_runtime_task", fake_prepare)
    monkeypatch.setattr(
        controller,
        "run_controller_cycle",
        fake_cycle,
    )

    result = controller._run_controller(
        task_path=task_path,
        task_cfg=OmegaConf.create(task_mapping(mode)),
        runtime_cfg=runtime_cfg(cycles=3),
    )

    assert result == 0
    assert calls == [(mode, 1), (mode, 2), (mode, 3)]


@pytest.mark.parametrize("mode", ["scalar", "pareto"])
@pytest.mark.parametrize(
    ("initial_cycle", "target_cycle", "expected_calls"),
    [(2, 3, [3]), (3, 3, [])],
)
def test_controller_resume_runs_only_until_absolute_cycle_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    initial_cycle: int,
    target_cycle: int,
    expected_calls: list[int],
) -> None:
    from instantmimic.hpo import run_hpo as controller

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "status.yaml").write_text(
        f"cycle: {initial_cycle}\n", encoding="utf-8"
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(task_mapping(mode))), encoding="utf-8"
    )
    calls: list[int] = []

    def fake_prepare(**_kwargs):
        return SimpleNamespace(
            task_path=task_path,
            task_cfg=OmegaConf.create(task_mapping(mode, state_dir=str(state_dir))),
            run_dir=tmp_path,
            state_dir=state_dir,
        )

    def fake_cycle(**_kwargs):
        before = int(OmegaConf.load(state_dir / "status.yaml").get("cycle", 0))
        after = before + 1
        calls.append(after)
        (state_dir / "status.yaml").write_text(f"cycle: {after}\n", encoding="utf-8")
        return controller.CycleRunResult(0, before, after)

    monkeypatch.setattr(controller, "prepare_hpo_runtime_task", fake_prepare)
    monkeypatch.setattr(controller, "run_controller_cycle", fake_cycle)

    result = controller._run_controller(
        task_path=task_path,
        task_cfg=OmegaConf.create(task_mapping(mode)),
        runtime_cfg=runtime_cfg(cycles=target_cycle, resume=str(tmp_path)),
    )

    assert result == 0
    assert calls == expected_calls


@pytest.mark.parametrize("mode", ["scalar", "pareto"])
def test_controller_rejects_adapter_that_skips_a_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from instantmimic.hpo import run_hpo as controller

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "status.yaml").write_text("cycle: 0\n", encoding="utf-8")
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(task_mapping(mode))), encoding="utf-8"
    )

    monkeypatch.setattr(
        controller,
        "prepare_hpo_runtime_task",
        lambda **_kwargs: SimpleNamespace(
            task_path=task_path,
            task_cfg=OmegaConf.create(task_mapping(mode, state_dir=str(state_dir))),
            run_dir=tmp_path,
            state_dir=state_dir,
        ),
    )

    calls: list[dict[str, object]] = []

    def skip_cycle(**kwargs):
        calls.append(kwargs)
        (state_dir / "status.yaml").write_text("cycle: 2\n", encoding="utf-8")
        return controller.CycleRunResult(returncode=0, before_cycle=0, after_cycle=2)

    monkeypatch.setattr(controller, "run_controller_cycle", skip_cycle)

    result = controller._run_controller(
        task_path=task_path,
        # A task the validator accepts. This used to be a two-key stub that
        # `validate_task` rejected, so the controller returned 2 from the schema
        # branch and the loop below was never entered: the assertion passed
        # without the invariant ever being exercised.
        task_cfg=OmegaConf.create(task_mapping(mode, state_dir=str(state_dir))),
        runtime_cfg=OmegaConf.create({"cycles": 1, "dry_run": False}),
    )

    assert calls, "the cycle was never run, so the invariant was not tested"
    assert result == 2


def test_common_controller_lock_is_shared_and_exclusive(tmp_path: Path) -> None:
    from instantmimic.hpo.controller_state import controller_state_lock

    with controller_state_lock(tmp_path) as first_acquired:
        with controller_state_lock(tmp_path) as second_acquired:
            assert first_acquired is True
            assert second_acquired is False

    with controller_state_lock(tmp_path) as acquired_after_release:
        assert acquired_after_release is True


def test_the_cycle_runtime_is_not_a_second_entrypoint() -> None:
    """`run_hpo` owns the loop; the cycle body runs one cycle."""

    from instantmimic.hpo import cycle

    assert not hasattr(cycle, "_run_hydra_main")
    assert not hasattr(cycle, "main")


def test_one_standing_program_for_both_modes() -> None:
    from instantmimic.hpo import cycle

    root = Path(cycle.__file__).resolve().parents[3]

    program = cycle._program_path(
        root, OmegaConf.create({"objective": {"mode": "scalar"}})
    )
    assert program.is_file()
    assert program == cycle._program_path(
        root, OmegaConf.create({"objective": {"mode": "pareto"}})
    )


def test_default_controller_strategy_is_mode_independent(tmp_path: Path) -> None:
    from instantmimic.hpo.controller_state import ensure_controller_state

    strategy_texts = []
    for mode in ("scalar", "pareto"):
        state_dir = tmp_path / mode
        ensure_controller_state(
            state_dir=state_dir,
            task_path=tmp_path / "task.yaml",
            task_name="demo",
            mode=mode,
        )
        strategy_texts.append((state_dir / "strategy.md").read_text(encoding="utf-8"))

    assert strategy_texts[0] == strategy_texts[1]


@pytest.mark.parametrize("mode", ["scalar", "pareto"])
def test_state_layout_and_status_envelope_are_the_same_in_both_modes(
    tmp_path: Path, mode: str
) -> None:
    """One state layer means one layout; the mode only labels the run."""

    from instantmimic.hpo import controller_state

    state_dir = tmp_path / mode / "state"
    status = controller_state.ensure_controller_state(
        state_dir=state_dir,
        task_path=tmp_path / "task.yaml",
        task_name="demo",
        mode=mode,
    )

    assert status["schema_version"] == controller_state.STATUS_SCHEMA_VERSION
    assert status["mode"] == mode
    assert set(status) == set(controller_state.status_defaults(mode))
    for name in (
        "status.yaml",
        "strategy.md",
        "progress.md",
        "best_result.md",
        "results.tsv",
    ):
        assert (state_dir / name).is_file()
    header = (state_dir / "results.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split("\t")) == controller_state.RESULTS_TSV_FIELDS


def test_lifecycle_and_decision_are_separate_fields(tmp_path: Path) -> None:
    """A comparator's word and the cycle's fate are different facts.

    They shared one column before, which is how a scalar `advance` -- a
    perfectly successful cycle -- ended up logged as a failure.
    """

    from instantmimic.hpo import controller_state

    status_path = tmp_path / "status.yaml"
    controller_state.write_controller_status(
        status_path,
        {"last_status": "completed", "last_decision": "advance"},
        mode="scalar",
    )
    loaded = controller_state.load_controller_status(status_path, mode="scalar")

    assert loaded["last_status"] == "completed"
    assert loaded["last_decision"] == "advance"


def test_objective_state_is_opaque_to_the_controller(tmp_path: Path) -> None:
    """Whatever the comparator stores comes back unchanged and unmerged."""

    from instantmimic.hpo import controller_state

    status_path = tmp_path / "status.yaml"
    payload = {"best_metric_seconds": 3.0, "incumbent_overrides": {"a": "1"}}
    controller_state.write_controller_status(
        status_path,
        {"cycle": 4, "objective_state": payload},
        mode="scalar",
    )
    loaded = controller_state.load_controller_status(status_path, mode="scalar")

    assert loaded["objective_state"] == payload
    # No lifecycle key leaked into the comparator's blob.
    assert "cycle" not in loaded["objective_state"]
    assert "last_status" not in loaded["objective_state"]


def test_a_legacy_state_directory_refuses_to_resume(tmp_path: Path) -> None:
    """Section 9 Q4: legacy runs stay readable, but are never migrated."""

    from instantmimic.hpo import controller_state

    status_path = tmp_path / "status.yaml"
    status_path.write_text(
        "schema_version: 1\nmode: scalar\ncycle: 12\n", encoding="utf-8"
    )

    with pytest.raises(
        controller_state.IncompatibleStateError, match="schema_version 1"
    ):
        controller_state.load_controller_status(status_path, mode="scalar")

    # The original is untouched: it is still history.
    assert "cycle: 12" in status_path.read_text(encoding="utf-8")


def test_a_legacy_results_header_refuses_to_resume(tmp_path: Path) -> None:
    from instantmimic.hpo import controller_state

    results_path = tmp_path / "results.tsv"
    results_path.write_text("cycle\tmode\tstatus\tresult_json\n", encoding="utf-8")

    with pytest.raises(controller_state.IncompatibleStateError, match="results header"):
        controller_state.read_controller_results(results_path)
