"""The unified hpo task schema.

One schema serves both objective modes. These tests hold that line: the same
task body is valid under ``objective.mode: scalar`` and ``objective.mode:
pareto``, every retired key from either old controller is rejected by name, and
a strategy may not be paired with the wrong mode.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from hpo_task_fixtures import runtime_cfg, task_cfg, task_mapping
from omegaconf import OmegaConf

from instantmimic.hpo import schema

TASK_PATH = pathlib.Path("config/hpo/synthetic.yaml")


def _errors(task: dict[str, Any], runtime: dict[str, Any] | None = None) -> list[str]:
    return [
        error.split(": ", 1)[1]
        for error in schema.validate_task(
            TASK_PATH,
            OmegaConf.create(task),
            OmegaConf.create(runtime) if runtime is not None else None,
        )
    ]


class TestOneSchemaForBothModes:
    @pytest.mark.parametrize("mode", ["scalar", "pareto"])
    def test_the_shipped_shape_validates_in_either_mode(self, mode: str) -> None:
        assert schema.validate_task(TASK_PATH, task_cfg(mode), runtime_cfg()) == []

    def test_only_the_objective_block_differs_between_modes(self) -> None:
        scalar = task_mapping("scalar")
        pareto = task_mapping("pareto")

        differing = {
            key
            for key in set(scalar) | set(pareto)
            if scalar.get(key) != pareto.get(key)
        }

        assert differing == {"objective"}
        assert _errors(scalar) == []
        assert _errors(pareto) == []

    def test_errors_are_prefixed_with_the_task_path(self) -> None:
        errors = schema.validate_task(
            TASK_PATH, OmegaConf.create(task_mapping(name="")), runtime_cfg()
        )

        assert errors == [f"{TASK_PATH}: name must be a non-empty string."]


class TestObjective:
    def test_missing_objective_is_reported(self) -> None:
        assert "objective must be a mapping." in _errors(task_mapping(objective=None))

    @pytest.mark.parametrize("mode", ["", "frontier", "Scalar"])
    def test_unknown_mode_is_rejected(self, mode: str) -> None:
        assert "objective.mode must be 'scalar' or 'pareto'." in _errors(
            task_mapping(objective={"mode": mode, "strategy": "wall_clock"})
        )

    def test_missing_strategy_is_rejected_with_the_mode_s_choices(self) -> None:
        errors = _errors(task_mapping(objective={"mode": "scalar"}))

        assert len(errors) == 1
        assert errors[0].startswith("objective.strategy is required")
        assert "wall_clock" in errors[0]

    def test_a_pareto_strategy_under_scalar_mode_names_the_right_mode(self) -> None:
        errors = _errors(
            task_mapping(objective={"mode": "scalar", "strategy": "axes_pareto"})
        )

        assert errors == [
            "objective.strategy 'axes_pareto' belongs to objective.mode "
            "'pareto', not 'scalar'."
        ]

    def test_a_scalar_strategy_under_pareto_mode_names_the_right_mode(self) -> None:
        errors = _errors(
            task_mapping(objective={"mode": "pareto", "strategy": "wall_clock"})
        )

        assert errors == [
            "objective.strategy 'wall_clock' belongs to objective.mode "
            "'scalar', not 'pareto'."
        ]

    def test_an_unknown_strategy_lists_the_mode_s_choices(self) -> None:
        errors = _errors(
            task_mapping(objective={"mode": "pareto", "strategy": "knee_only"})
        )

        assert len(errors) == 1
        assert "objective.strategy 'knee_only' is unknown" in errors[0]
        assert "learning_curve_pareto" in errors[0]

    def test_strategy_specific_keys_are_left_to_the_comparator(self) -> None:
        task = task_mapping(
            "scalar",
            objective={
                "mode": "scalar",
                "strategy": "tracking_benchmark_auc",
                "thresholds": {"ape_mean": 0.05},
                "auc_error_cap": 0.06,
                "some_future_comparator_key": 3,
            },
        )

        assert _errors(task) == []

    @pytest.mark.parametrize(
        "strategy",
        [
            "wall_clock",
            "time_to_reward_threshold",
            "time_to_tracking_benchmark",
            "frames_to_tracking_benchmark",
            "tracking_benchmark_auc",
        ],
    )
    def test_every_scalar_strategy_is_spellable(self, strategy: str) -> None:
        assert (
            _errors(task_mapping(objective={"mode": "scalar", "strategy": strategy}))
            == []
        )

    def test_the_listed_strategies_are_exactly_the_ones_with_a_scorer(self) -> None:
        """A strategy the scorer cannot score must not be spellable.

        `curves.trial_objective` falls through to the time-to-reward-threshold
        marker for a name it does not know, so an unscored strategy would rank
        by a metric nobody configured, silently and plausibly.
        """

        from instantmimic.hpo.objective.scalar import (
            SCALAR_STRATEGY_OBJECTIVES,
        )

        assert set(schema.SCALAR_STRATEGIES) == set(SCALAR_STRATEGY_OBJECTIVES)

    @pytest.mark.parametrize("strategy", ["weighted_sum", "reward_threshold_ladder"])
    def test_an_unimplemented_strategy_is_refused_with_its_reason(
        self, strategy: str
    ) -> None:
        """Both scored an aggregate the unified evaluator does not emit."""

        errors = _errors(
            task_mapping(objective={"mode": "scalar", "strategy": strategy})
        )

        assert any("is not implemented" in error for error in errors), errors
        assert any("refused rather than guessed" in error for error in errors)

    @pytest.mark.parametrize(
        "strategy", ["pareto", "learning_curve_pareto", "axes_pareto"]
    )
    def test_every_frontier_strategy_is_spellable(self, strategy: str) -> None:
        assert (
            _errors(task_mapping(objective={"mode": "pareto", "strategy": strategy}))
            == []
        )


class TestRetiredKeys:
    """Every key either old controller read must fail by name."""

    @pytest.mark.parametrize("key", sorted(schema.RETIRED_TOP_LEVEL_KEYS))
    def test_retired_top_level_key_is_rejected(self, key: str) -> None:
        errors = _errors(task_mapping(**{key: "whatever"}))

        assert any(error.startswith(f"task.{key} is retired;") for error in errors)

    @pytest.mark.parametrize("key", sorted(schema.RETIRED_BATCH_KEYS))
    def test_retired_batch_key_is_rejected(self, key: str) -> None:
        errors = _errors(task_mapping(batch={key: "whatever"}))

        assert any(error.startswith(f"batch.{key} is retired;") for error in errors)

    @pytest.mark.parametrize("key", sorted(schema.RETIRED_PLANNER_KEYS))
    def test_retired_planner_key_is_rejected(self, key: str) -> None:
        errors = _errors(task_mapping(planner={key: "whatever"}))

        assert any(error.startswith(f"planner.{key} is retired;") for error in errors)

    @pytest.mark.parametrize("key", sorted(schema.RETIRED_RUNTIME_KEYS))
    def test_retired_runtime_key_is_rejected(self, key: str) -> None:
        errors = _errors(task_mapping(), {"cycles": 1, key: "whatever"})

        assert any(error.startswith(f"runtime.{key} is retired;") for error in errors)

    def test_the_retired_message_names_the_replacement(self) -> None:
        assert _errors(task_mapping(config_path="config/demo")) == [
            "task.config_path is retired; use `batch.motions.<motion>.config_path`."
        ]

    def test_an_unknown_key_is_rejected_without_a_replacement(self) -> None:
        assert _errors(task_mapping(mystery=1)) == ["task.mystery is not a known key."]

    @pytest.mark.parametrize(
        ("section", "key"),
        [
            ("batch", "mystery"),
            ("planner", "mystery"),
        ],
    )
    def test_unknown_section_keys_are_rejected(self, section: str, key: str) -> None:
        errors = _errors(task_mapping(**{section: {key: 1}}))

        assert f"{section}.{key} is not a known key." in errors

    def test_injected_runtime_paths_are_accepted(self) -> None:
        task = task_mapping(state_dir="runs/x/state", output_root="runs/x/evaluator")

        assert _errors(task) == []


class TestBatch:
    def test_motions_must_name_a_config(self) -> None:
        errors = _errors(task_mapping(batch={"motions": {"demo": {}}}))

        assert "batch.motions.demo.config_path must be a non-empty string." in errors
        assert "batch.motions.demo.config_name must be a non-empty string." in errors

    def test_an_empty_motion_set_is_rejected(self) -> None:
        assert "batch.motions must define at least one motion." in _errors(
            task_mapping(batch={"motions": {}})
        )

    def test_multiple_motions_are_allowed(self) -> None:
        task = task_mapping(
            "pareto",
            batch={
                "motions": {
                    "walk": {"config_path": "config/a", "config_name": "walk"},
                    "run": {"config_path": "config/a", "config_name": "run"},
                }
            },
        )

        assert _errors(task) == []

    def test_seeds_must_be_a_non_empty_list_of_distinct_integers(self) -> None:
        assert "batch.seeds must be a non-empty list of integers." in _errors(
            task_mapping(batch={"seeds": []})
        )
        assert "batch.seeds[0] must be an integer." in _errors(
            task_mapping(batch={"seeds": ["one"]})
        )
        assert "batch.seeds must not repeat a seed." in _errors(
            task_mapping(batch={"seeds": [1, 1]})
        )

    @pytest.mark.parametrize(
        ("key", "value", "expected"),
        [
            ("seed_batch", "sometimes", "batch.seed_batch must be one of"),
            ("eval_source", "vibes", "batch.eval_source must be one of"),
            ("compile_cache_scope", "global", "batch.compile_cache_scope must be one"),
            ("max_parallel", 0, "batch.max_parallel must be >= 1."),
            ("retries_per_seed", -1, "batch.retries_per_seed must be >= 0."),
            ("stop_on_fail", "yes", "batch.stop_on_fail must be a boolean."),
            (
                "trial_timeout_seconds",
                0,
                "batch.trial_timeout_seconds must be > 0 when set.",
            ),
        ],
    )
    def test_batch_value_constraints(
        self, key: str, value: object, expected: str
    ) -> None:
        errors = _errors(task_mapping(batch={key: value}))

        assert any(error.startswith(expected) for error in errors), errors

    def test_a_null_trial_timeout_means_no_timeout(self) -> None:
        assert _errors(task_mapping(batch={"trial_timeout_seconds": None})) == []

    def test_retry_triggers_are_checked_against_the_known_set(self) -> None:
        assert _errors(task_mapping(batch={"retry_triggers": ["cublas"]})) == []
        errors = _errors(task_mapping(batch={"retry_triggers": ["vibes"]}))
        assert any("retry_triggers entry 'vibes'" in error for error in errors)

    def test_experiment_bounds_must_be_ordered(self) -> None:
        errors = _errors(
            task_mapping(
                batch={"min_experiments_per_cycle": 4, "max_experiments_per_cycle": 2}
            )
        )

        assert (
            "batch.min_experiments_per_cycle must not exceed "
            "batch.max_experiments_per_cycle." in errors
        )

    def test_playback_keys_are_namespaced(self) -> None:
        task = task_mapping(
            batch={
                "playback": {
                    "video": True,
                    "video_length": 1800,
                    "overrides": ["play.steps=10"],
                    "disable_default_overrides": False,
                }
            }
        )

        assert _errors(task) == []
        assert "batch.playback.length is not a known key." in _errors(
            task_mapping(batch={"playback": {"length": 10}})
        )

    def test_fixed_train_overrides_must_be_strings(self) -> None:
        assert "batch.fixed_train_overrides[0] must be a non-empty string." in _errors(
            task_mapping(batch={"fixed_train_overrides": [7]})
        )


class TestSearchSpace:
    def test_parameter_space_is_required_and_non_empty(self) -> None:
        assert "parameter_space must be a list of strings." in _errors(
            task_mapping(parameter_space=None)
        )
        assert "parameter_space must not be empty." in _errors(
            task_mapping(parameter_space=[])
        )

    def test_forbidden_prefixes_may_be_empty(self) -> None:
        assert _errors(task_mapping(forbidden_prefixes=[])) == []


class TestPlanner:
    def test_planner_is_optional(self) -> None:
        assert _errors(task_mapping(planner=None)) == []

    def test_nested_planner_sections_reject_unknown_keys(self) -> None:
        assert "planner.deep_review.rows is not a known key." in _errors(
            task_mapping(planner={"deep_review": {"rows": 3}})
        )
        assert "planner.creativity.every is not a known key." in _errors(
            task_mapping(planner={"creativity": {"every": 3}})
        )

    def test_initial_candidates_carry_a_name_and_per_motion_overrides(self) -> None:
        task = task_mapping(
            planner={
                "initial_candidates": [
                    {
                        "name": "baseline",
                        "rationale": "start here",
                        "motions": {"demo": {"overrides": []}},
                    }
                ]
            }
        )

        assert _errors(task) == []

    def test_an_initial_candidate_without_a_name_is_rejected(self) -> None:
        errors = _errors(
            task_mapping(planner={"initial_candidates": [{"motions": {"demo": {}}}]})
        )

        assert (
            "planner.initial_candidates[0].name must be a non-empty string." in errors
        )

    def test_planner_timing_must_be_non_negative(self) -> None:
        assert "planner.timeout_seconds must be >= 0." in _errors(
            task_mapping(planner={"timeout_seconds": -1})
        )
        assert "planner.timeout_retries must be >= 0." in _errors(
            task_mapping(planner={"timeout_retries": -1})
        )


class TestRuntime:
    def test_runtime_is_validated_once(self) -> None:
        assert "runtime.cycles must be >= 1." in _errors(task_mapping(), {"cycles": 0})

    def test_unknown_evaluator_keys_are_rejected(self) -> None:
        errors = _errors(task_mapping(), {"evaluator": {"nodes": 2}})

        assert "runtime.evaluator.nodes is not a known key." in errors

    def test_runtime_inside_the_task_mapping_is_validated_too(self) -> None:
        errors = _errors(task_mapping(runtime={"cycles": 0}))

        assert "runtime.cycles must be >= 1." in errors


class TestRuntimeOptions:
    def test_defaults_resolve_to_one_local_evaluator(self) -> None:
        options = schema.resolve_runtime_options(runtime_cfg())

        assert options.cycles is None
        assert options.dry_run is False
        assert options.evaluator.launcher == "local"
        assert options.evaluator.max_parallel == 1

    def test_cycles_and_dry_run_are_read_from_the_config(self) -> None:
        options = schema.resolve_runtime_options(runtime_cfg(cycles=7, dry_run=True))

        assert options.cycles == 7
        assert options.dry_run is True

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("on", True), ("0", False), ("no", False)],
    )
    def test_dry_run_accepts_a_string_override(self, raw: str, expected: bool) -> None:
        assert schema.resolve_runtime_options(runtime_cfg(dry_run=raw)).dry_run is (
            expected
        )

    def test_a_cycle_target_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="runtime.cycles must be >= 1"):
            schema.resolve_runtime_options(runtime_cfg(cycles=0))

    def test_the_evaluator_launcher_comes_from_runtime_evaluator(self) -> None:
        options = schema.resolve_runtime_options(
            runtime_cfg(evaluator={"launcher": "local", "max_parallel": 1})
        )

        assert options.evaluator.launcher == "local"


class TestMotionGroupEvidenceIsValidatedAtAuditTime:
    """A misspelled statistic should fail the config audit, not cycle 1.

    `planner.motion_group_evidence` is parsed when the planner builds its
    prompt, which is after the run directory exists and the lock is held. The
    validator parses it too, so `validate_task_configs.py` catches it.
    """

    def test_a_valid_spec_passes(self) -> None:
        task = task_mapping(
            "pareto",
            planner={
                "motion_group_evidence": {
                    "reward_mean": ["terminal", "normalized_auc"]
                },
                "motion_group_seed_reduce": "median",
            },
        )
        assert schema.validate_task("task.yaml", OmegaConf.create(task)) == []

    def test_an_unknown_statistic_is_reported(self) -> None:
        task = task_mapping(
            "pareto", planner={"motion_group_evidence": {"reward_mean": ["p95"]}}
        )
        errors = schema.validate_task("task.yaml", OmegaConf.create(task))
        assert any("supports only" in error and "p95" in error for error in errors)

    def test_an_unknown_seed_reduce_is_reported(self) -> None:
        task = task_mapping("pareto", planner={"motion_group_seed_reduce": "p95"})
        errors = schema.validate_task("task.yaml", OmegaConf.create(task))
        assert any("mean' or 'median" in error for error in errors)
