"""The frontier (Pareto) comparator, on synthetic evaluator summaries.

Three archive shapes share one implementation: they accumulate points across
cycles, recompute the frontier, pick representatives, rotate the planner
anchor, and project a best result. These tests were written against the three
separate updaters in the Pareto controller and carry the same expected values
over unchanged, which is what shows the move into ``FrontierComparator``
preserved the behavior.

Dispatch is on ``summary["pareto"]["mode"]`` -- absent (time/quality),
``"axes"``, or ``"learning_curve"``.
"""

from __future__ import annotations

import json
import pathlib
import unittest
from typing import Any

import pytest
from omegaconf import OmegaConf

from instantmimic.hpo.objective import frontier as objective_frontier
from instantmimic.hpo.objective.frontier import (
    _aggregate_axes_pareto_curve,
    _aggregate_pareto_curve,
    _build_axes_pareto_summary,
    _build_learning_curve_pareto_summary,
    _build_pareto_summary,
)


def _archive(state_dir: pathlib.Path) -> dict[str, Any]:
    return json.loads(
        (state_dir / "objective" / "pareto_frontier.json").read_text(encoding="utf-8")
    )


def _state_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _time_quality_summary(
    tmp_path: pathlib.Path,
    frontier: list[dict[str, Any]],
    *,
    selection: str = "knee",
    time_epsilon: float = 0.0,
    quality_epsilon: float = 0.0,
    quality_goal: str = "maximize",
) -> dict[str, Any]:
    return {
        "output_root": str(tmp_path / "evaluation"),
        "pareto": {
            "time": {
                "source_key": "active_train_time",
                "aggregate": "median",
                "epsilon": time_epsilon,
            },
            "quality": {
                "source_key": "current_reward_mean",
                "goal": quality_goal,
                "aggregate": "mean",
                "epsilon": quality_epsilon,
            },
            "selection": selection,
            "frontier": frontier,
        },
        "experiments": [],
    }


def _axes_summary(
    tmp_path: pathlib.Path,
    points: list[dict[str, Any]],
    *,
    selection: str = "balanced",
    axes: list[dict[str, Any]] | None = None,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "output_root": str(tmp_path / "evaluation"),
        "pareto": {
            "mode": "axes",
            "axes": axes
            or [
                {"key": "active_train_time", "goal": "minimize", "epsilon": 0.0},
                {"key": "quality", "goal": "maximize", "epsilon": 0.0},
            ],
            "budget": budget or {},
            "selection": selection,
            "points": points,
        },
        "experiments": [],
    }


def _learning_curve_summary(
    tmp_path: pathlib.Path,
    frontier: list[dict[str, Any]],
    *,
    selection: str = "balanced",
    floor: float | None = None,
    goal: str = "maximize",
) -> dict[str, Any]:
    return {
        "output_root": str(tmp_path / "evaluation"),
        "pareto": {
            "mode": "learning_curve",
            "final_quality": {
                "source_key": "current_reward_mean",
                "goal": goal,
                "aggregate": "mean",
                "epsilon": 0.0,
            },
            "anytime_quality": {"goal": goal, "epsilon": 0.0, "floor": floor},
            "budget": {},
            "active_time_budget": {},
            "selection": selection,
            "frontier": frontier,
        },
        "experiments": [],
    }


class TestTimeQualityArchive:
    """The original two-axis (time, quality) archive."""

    def test_frontier_accumulates_across_three_cycles(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {"best_score": None}

        objective_frontier._update_best_result(
            state_dir,
            status,
            1,
            _time_quality_summary(
                tmp_path,
                [
                    {
                        "experiment": "fast",
                        "eval_index": 1,
                        "active_train_time": 9.0,
                        "quality": 80.0,
                    },
                    {
                        "experiment": "knee",
                        "eval_index": 1,
                        "active_train_time": 10.0,
                        "quality": 90.0,
                    },
                ],
            ),
        )
        objective_frontier._update_best_result(
            state_dir,
            status,
            2,
            _time_quality_summary(
                tmp_path,
                [
                    {
                        "experiment": "quality",
                        "eval_index": 1,
                        "active_train_time": 12.0,
                        "quality": 100.0,
                    },
                    {
                        "experiment": "dominated",
                        "eval_index": 1,
                        "active_train_time": 13.0,
                        "quality": 95.0,
                    },
                ],
            ),
        )
        objective_frontier._update_best_result(
            state_dir,
            status,
            3,
            _time_quality_summary(
                tmp_path,
                [
                    {
                        "experiment": "also_dominated",
                        "eval_index": 1,
                        "active_train_time": 11.0,
                        "quality": 85.0,
                    },
                ],
            ),
        )

        archive = _archive(state_dir)
        assert [point["experiment"] for point in archive["frontier"]] == [
            "fast",
            "knee",
            "quality",
        ]
        assert [point["cycle"] for point in archive["frontier"]] == [1, 1, 2]
        assert archive["representatives"]["fastest"]["experiment"] == "fast"
        assert archive["representatives"]["quality"]["experiment"] == "quality"
        assert archive["representatives"]["knee"]["experiment"] == "knee"
        assert archive["selected"]["experiment"] == "knee"

    def test_status_records_the_selected_representative_without_a_score(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {"best_score": 1.0}

        objective_frontier._update_best_result(
            state_dir,
            status,
            4,
            _time_quality_summary(
                tmp_path,
                [
                    {
                        "experiment": "only",
                        "eval_index": 1,
                        "active_train_time": 5.0,
                        "quality": 50.0,
                    },
                ],
            ),
        )

        assert status["best_cycle"] == 4
        assert status["best_experiment"] == "only"
        assert status["best_score"] is None
        assert status["best_summary_path"].endswith("evaluation/summary.json")
        assert status["pareto_archive_path"].endswith("objective/pareto_frontier.json")

    @pytest.mark.parametrize(
        ("selection", "expected"),
        [("fastest", "fast"), ("quality", "slow"), ("knee", "fast")],
    )
    def test_selection_policy_picks_the_named_representative(
        self, tmp_path: pathlib.Path, selection: str, expected: str
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {"best_score": None}

        objective_frontier._update_best_result(
            state_dir,
            status,
            1,
            _time_quality_summary(
                tmp_path,
                [
                    {
                        "experiment": "fast",
                        "eval_index": 1,
                        "active_train_time": 1.0,
                        "quality": 90.0,
                    },
                    {
                        "experiment": "slow",
                        "eval_index": 1,
                        "active_train_time": 100.0,
                        "quality": 91.0,
                    },
                ],
                selection=selection,
            ),
        )

        assert _archive(state_dir)["selected"]["experiment"] == expected

    def test_unknown_selection_policy_is_rejected(self, tmp_path: pathlib.Path) -> None:
        state_dir = _state_dir(tmp_path)

        with pytest.raises(ValueError, match="Unsupported Pareto selection policy"):
            objective_frontier._update_best_result(
                state_dir,
                {"best_score": None},
                1,
                _time_quality_summary(
                    tmp_path,
                    [
                        {
                            "experiment": "a",
                            "eval_index": 1,
                            "active_train_time": 1.0,
                            "quality": 1.0,
                        }
                    ],
                    selection="cheapest",
                ),
            )

    def test_epsilon_keeps_near_equivalent_points_on_the_frontier(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)

        objective_frontier._update_best_result(
            state_dir,
            {"best_score": None},
            1,
            _time_quality_summary(
                tmp_path,
                [
                    {
                        "experiment": "a",
                        "eval_index": 1,
                        "active_train_time": 10.0,
                        "quality": 90.0,
                    },
                    {
                        "experiment": "b",
                        "eval_index": 1,
                        "active_train_time": 10.4,
                        "quality": 89.8,
                    },
                ],
                time_epsilon=0.5,
                quality_epsilon=1.0,
            ),
        )

        assert [point["experiment"] for point in _archive(state_dir)["frontier"]] == [
            "a",
            "b",
        ]

    def test_changing_the_objective_on_resume_raises(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        points = [
            {
                "experiment": "a",
                "eval_index": 1,
                "active_train_time": 1.0,
                "quality": 1.0,
            }
        ]
        objective_frontier._update_best_result(
            state_dir,
            {"best_score": None},
            1,
            _time_quality_summary(tmp_path, points),
        )

        with pytest.raises(ValueError, match="Pareto objective changed"):
            objective_frontier._update_best_result(
                state_dir,
                {"best_score": None},
                2,
                _time_quality_summary(tmp_path, points, selection="fastest"),
            )

    def test_best_result_renders_the_frontier(self, tmp_path: pathlib.Path) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, object] = {"best_score": None}

        objective_frontier._update_best_result(
            state_dir,
            status,
            1,
            _time_quality_summary(
                tmp_path,
                [
                    {
                        "experiment": "fast",
                        "eval_index": 1,
                        "active_train_time": 9.0,
                        "quality": 80.0,
                    },
                    {
                        "experiment": "knee",
                        "eval_index": 2,
                        "active_train_time": 10.0,
                        "quality": 90.0,
                    },
                ],
            ),
        )

        # `controller_state.write_best_result` is the only writer of
        # `best_result.md`; the archive hands it the body.
        assert not (state_dir / "best_result.md").exists()
        text = str(status["best_result_text"])
        assert text.startswith("# Pareto Result\n")
        assert "The frontier is authoritative" in text
        assert "- Selection view: `knee`" in text
        # With only two frontier points the knee representative is the fastest.
        assert "- Selected experiment: `fast`" in text
        assert "- Active train time: `9.0`" in text
        assert "- Fixed-evaluator quality: `80.0`" in text
        assert "- cycle=1 experiment=fast eval_index=1 time=9.0 quality=80.0" in text
        assert "- cycle=1 experiment=knee eval_index=2 time=10.0 quality=90.0" in text


class TestAxesArchive:
    """The generalized N-axis archive with anchor rotation."""

    def test_frontier_and_points_accumulate_separately(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {}

        objective_frontier._update_best_result(
            state_dir,
            status,
            1,
            _axes_summary(
                tmp_path,
                [
                    {
                        "experiment": "fast",
                        "eval_index": 1,
                        "active_train_time": 9.0,
                        "quality": 80.0,
                    },
                    {
                        "experiment": "mid",
                        "eval_index": 1,
                        "active_train_time": 10.0,
                        "quality": 90.0,
                    },
                ],
            ),
        )
        objective_frontier._update_best_result(
            state_dir,
            status,
            2,
            _axes_summary(
                tmp_path,
                [
                    {
                        "experiment": "dominated",
                        "eval_index": 1,
                        "active_train_time": 11.0,
                        "quality": 85.0,
                    },
                ],
            ),
        )

        archive = _archive(state_dir)
        assert [point["experiment"] for point in archive["points"]] == [
            "fast",
            "mid",
            "dominated",
        ]
        assert [point["experiment"] for point in archive["frontier"]] == [
            "fast",
            "mid",
        ]

    def test_anchor_rotates_through_balanced_then_each_axis(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {}
        points = [
            {
                "experiment": "fast",
                "eval_index": 1,
                "active_train_time": 1.0,
                "quality": 10.0,
            },
            {
                "experiment": "good",
                "eval_index": 1,
                "active_train_time": 100.0,
                "quality": 99.0,
            },
        ]

        roles: list[str] = []
        for cycle in (1, 2, 3, 4):
            objective_frontier._update_best_result(
                state_dir, status, cycle, _axes_summary(tmp_path, points)
            )
            roles.append(_archive(state_dir)["active_anchor"]["role"])

        assert roles == ["balanced", "active_train_time", "quality", "balanced"]
        assert status["active_anchor"]["role"] == "balanced"

    def test_anchor_points_at_the_axis_champion(self, tmp_path: pathlib.Path) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {}
        points = [
            {
                "experiment": "fast",
                "eval_index": 1,
                "active_train_time": 1.0,
                "quality": 10.0,
            },
            {
                "experiment": "good",
                "eval_index": 1,
                "active_train_time": 100.0,
                "quality": 99.0,
            },
        ]

        objective_frontier._update_best_result(
            state_dir, status, 2, _axes_summary(tmp_path, points)
        )

        anchor = _archive(state_dir)["active_anchor"]
        assert anchor["role"] == "active_train_time"
        assert anchor["point"]["experiment"] == "fast"

    def test_fewer_than_two_axes_is_rejected(self, tmp_path: pathlib.Path) -> None:
        state_dir = _state_dir(tmp_path)

        with pytest.raises(ValueError, match="at least two axes"):
            objective_frontier._update_best_result(
                state_dir,
                {},
                1,
                _axes_summary(
                    tmp_path,
                    [{"experiment": "a", "eval_index": 1, "quality": 1.0}],
                    axes=[{"key": "quality", "goal": "maximize", "epsilon": 0.0}],
                ),
            )

    def test_switching_archive_mode_on_resume_raises(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        points = [
            {
                "experiment": "a",
                "eval_index": 1,
                "active_train_time": 1.0,
                "quality": 1.0,
            }
        ]
        objective_frontier._update_best_result(
            state_dir, {}, 1, _axes_summary(tmp_path, points)
        )

        with pytest.raises(ValueError, match="Pareto mode changed"):
            objective_frontier._update_best_result(
                state_dir,
                {},
                2,
                _learning_curve_summary(
                    tmp_path,
                    [{"experiment": "a", "final_quality": 1.0, "anytime_quality": 1.0}],
                ),
            )

    def test_changed_axes_on_resume_raises(self, tmp_path: pathlib.Path) -> None:
        state_dir = _state_dir(tmp_path)
        points = [
            {
                "experiment": "a",
                "eval_index": 1,
                "active_train_time": 1.0,
                "quality": 1.0,
            }
        ]
        objective_frontier._update_best_result(
            state_dir, {}, 1, _axes_summary(tmp_path, points)
        )

        with pytest.raises(ValueError, match="Pareto objective changed"):
            objective_frontier._update_best_result(
                state_dir,
                {},
                2,
                _axes_summary(tmp_path, points, budget={"frames": 10}),
            )

    def test_frontier_markdown_and_best_projection_are_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {}

        objective_frontier._update_best_result(
            state_dir,
            status,
            1,
            _axes_summary(
                tmp_path,
                [
                    {
                        "experiment": "fast",
                        "eval_index": 1,
                        "active_train_time": 1.0,
                        "quality": 10.0,
                    },
                    {
                        "experiment": "good",
                        "eval_index": 1,
                        "active_train_time": 100.0,
                        "quality": 99.0,
                    },
                ],
            ),
        )

        frontier_text = (state_dir / "objective" / "pareto_frontier.md").read_text(
            encoding="utf-8"
        )
        assert frontier_text.startswith("# Pareto Frontier\n")
        assert (
            "- cycle=1 experiment=fast active_train_time=1.0 quality=10.0"
            in frontier_text
        )
        assert "## Recommendations" in frontier_text
        assert "- active_train_time: `fast`" in frontier_text
        assert "## Active anchor" in frontier_text

        assert not (state_dir / "best_result.md").exists()
        best_text = str(status["best_result_text"])
        assert best_text.startswith("# Best Result\n")
        assert (
            "The authoritative result is `objective/pareto_frontier.md`." in best_text
        )
        assert "- Selection policy: `balanced`" in best_text
        assert "- active_train_time: `1.0`" in best_text


class TestLearningCurveArchive:
    """final_quality vs anytime_quality, with the constrained selection."""

    def test_balanced_selection_across_three_cycles(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {}

        objective_frontier._update_best_result(
            state_dir,
            status,
            1,
            _learning_curve_summary(
                tmp_path,
                [
                    {
                        "experiment": "slow_high",
                        "final_quality": 100.0,
                        "anytime_quality": 63.0,
                    },
                ],
            ),
        )
        objective_frontier._update_best_result(
            state_dir,
            status,
            2,
            _learning_curve_summary(
                tmp_path,
                [
                    {
                        "experiment": "fast_lower",
                        "final_quality": 95.0,
                        "anytime_quality": 79.0,
                    },
                ],
            ),
        )
        objective_frontier._update_best_result(
            state_dir,
            status,
            3,
            _learning_curve_summary(
                tmp_path,
                [
                    {
                        "experiment": "dominated",
                        "final_quality": 90.0,
                        "anytime_quality": 60.0,
                    },
                ],
            ),
        )

        archive = _archive(state_dir)
        assert [point["experiment"] for point in archive["frontier"]] == [
            "slow_high",
            "fast_lower",
        ]
        assert archive["recommendations"]["final_quality"]["experiment"] == (
            "slow_high"
        )
        assert archive["recommendations"]["anytime_quality"]["experiment"] == (
            "fast_lower"
        )

    def test_final_quality_constrained_respects_the_anytime_floor(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {}

        objective_frontier._update_best_result(
            state_dir,
            status,
            2,
            _learning_curve_summary(
                tmp_path,
                [
                    {
                        "experiment": "slow_high",
                        "final_quality": 100.0,
                        "anytime_quality": 63.0,
                    },
                    {
                        "experiment": "fast_lower",
                        "final_quality": 95.0,
                        "anytime_quality": 79.0,
                    },
                ],
                selection="final_quality_constrained",
                floor=75.0,
            ),
        )

        archive = _archive(state_dir)
        assert archive["recommendations"]["final_quality"]["experiment"] == (
            "slow_high"
        )
        assert archive["selected"]["experiment"] == "fast_lower"
        assert archive["active_anchor"]["role"] == "final_quality"
        assert archive["active_anchor"]["point"]["experiment"] == "slow_high"
        assert status["best_experiment"] == "fast_lower"

    def test_floor_above_every_point_leaves_no_constrained_selection(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {}

        objective_frontier._update_best_result(
            state_dir,
            status,
            1,
            _learning_curve_summary(
                tmp_path,
                [
                    {
                        "experiment": "slow_high",
                        "final_quality": 100.0,
                        "anytime_quality": 63.0,
                    },
                    {
                        "experiment": "fast_lower",
                        "final_quality": 95.0,
                        "anytime_quality": 79.0,
                    },
                ],
                selection="final_quality_constrained",
                floor=200.0,
            ),
        )

        archive = _archive(state_dir)
        assert "final_quality_constrained" not in archive["recommendations"]
        assert archive["selected"] is None
        assert "best_experiment" not in status

    def test_minimized_objectives_flip_the_champions(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {}

        objective_frontier._update_best_result(
            state_dir,
            status,
            1,
            _learning_curve_summary(
                tmp_path,
                [
                    {
                        "experiment": "best_final",
                        "final_quality": 0.05,
                        "anytime_quality": 0.30,
                    },
                    {
                        "experiment": "best_anytime",
                        "final_quality": 0.07,
                        "anytime_quality": 0.10,
                    },
                ],
                goal="minimize",
            ),
        )

        archive = _archive(state_dir)
        assert archive["recommendations"]["final_quality"]["experiment"] == (
            "best_final"
        )
        assert archive["recommendations"]["anytime_quality"]["experiment"] == (
            "best_anytime"
        )

    def test_anchor_rotates_through_the_three_learning_curve_roles(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {}
        frontier = [
            {
                "experiment": "slow_high",
                "final_quality": 100.0,
                "anytime_quality": 63.0,
            },
            {
                "experiment": "fast_lower",
                "final_quality": 95.0,
                "anytime_quality": 79.0,
            },
        ]

        roles: list[str] = []
        for cycle in (1, 2, 3, 4):
            objective_frontier._update_best_result(
                state_dir,
                status,
                cycle,
                _learning_curve_summary(tmp_path, frontier),
            )
            roles.append(_archive(state_dir)["active_anchor"]["role"])

        assert roles == [
            "balanced",
            "final_quality",
            "anytime_quality",
            "balanced",
        ]

    def test_changed_floor_on_resume_raises(self, tmp_path: pathlib.Path) -> None:
        state_dir = _state_dir(tmp_path)
        frontier = [{"experiment": "a", "final_quality": 1.0, "anytime_quality": 1.0}]
        objective_frontier._update_best_result(
            state_dir, {}, 1, _learning_curve_summary(tmp_path, frontier, floor=1.0)
        )

        with pytest.raises(ValueError, match="Pareto objective changed"):
            objective_frontier._update_best_result(
                state_dir,
                {},
                2,
                _learning_curve_summary(tmp_path, frontier, floor=2.0),
            )

    def test_best_projection_names_both_quality_axes(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, object] = {}

        objective_frontier._update_best_result(
            state_dir,
            status,
            1,
            _learning_curve_summary(
                tmp_path,
                [
                    {
                        "experiment": "only",
                        "final_quality": 10.0,
                        "anytime_quality": 5.0,
                    },
                ],
            ),
        )

        assert not (state_dir / "best_result.md").exists()
        text = str(status["best_result_text"])
        assert "- Final quality: `10.0`" in text
        assert "- Anytime quality: `5.0`" in text
        assert "- Selection policy: `balanced`" in text


class TestScalarFallbackInsideParetoController:
    """``_update_best_result`` still scores experiments when no Pareto block."""

    def test_highest_scoring_valid_experiment_becomes_best(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {"best_score": None}
        payload = {
            "output_root": str(tmp_path / "evaluation"),
            "experiments": [
                {
                    "name": "low",
                    "aggregate": {"mean_reward": 1.0},
                    "selection": {"valid": True, "score": 1.0},
                },
                {
                    "name": "high",
                    "aggregate": {"mean_reward": 2.0},
                    "selection": {"valid": True, "score": 2.0},
                },
                {
                    "name": "invalid",
                    "aggregate": {"mean_reward": 9.0},
                    "selection": {"valid": False, "score": 9.0},
                },
            ],
        }

        objective_frontier._update_best_result(state_dir, status, 1, payload)

        assert status["best_experiment"] == "high"
        assert status["best_score"] == 2.0
        assert not (state_dir / "best_result.md").exists()
        assert "# Best Result" in str(status["best_result_text"])

    def test_a_worse_later_cycle_does_not_replace_the_best(
        self, tmp_path: pathlib.Path
    ) -> None:
        state_dir = _state_dir(tmp_path)
        status: dict[str, Any] = {"best_score": None}

        def payload(name: str, score: float) -> dict[str, Any]:
            return {
                "output_root": str(tmp_path / "evaluation"),
                "experiments": [
                    {
                        "name": name,
                        "aggregate": {},
                        "selection": {"valid": True, "score": score},
                    }
                ],
            }

        objective_frontier._update_best_result(
            state_dir, status, 1, payload("first", 5.0)
        )
        objective_frontier._update_best_result(
            state_dir, status, 2, payload("second", 4.0)
        )

        assert status["best_experiment"] == "first"
        assert status["best_score"] == 5.0
        assert status["best_cycle"] == 1


# --- comparison over evaluator aggregates -----------------------------------
#
# These moved here with the code they exercise: building frontier points out of
# per-seed rows, and scoring an aggregate for the weighted-sum and
# reward-threshold-ladder strategies. The expected values are unchanged from
# when they lived beside the evaluator.


class FrontierEvaluatorParameterSpaceTest(unittest.TestCase):
    def test_axes_pareto_supports_three_failure_censored_terminal_metrics(
        self,
    ) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["env.reward.alive"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1, 2],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "axes_pareto",
                    "frame_budget": 2_000,
                    "terminal_frame_tolerance": 0,
                    "selection": "balanced",
                    "axes": [
                        {
                            "key": f"{motion}_global_mpjpe",
                            "goal": "minimize",
                            "epsilon": 0.005,
                            "value": {
                                "type": "terminal",
                                "metric": f"current_{motion}_global_mpjpe_mean",
                                "window": 1,
                                "seed_reduce": "mean",
                            },
                            "failure_censor": {
                                "success_rate_metric": (
                                    f"current_{motion}_success_rate"
                                ),
                                "failure_value": 1.0,
                            },
                        }
                        for motion in ("walk", "run", "backflip")
                    ],
                },
            }
        )
        rows = [
            {
                "seed": 1,
                "train_returncode": 0,
                "benchmark_curve": [
                    {
                        "eval_index": 1,
                        "frames": 1_000,
                        "current_walk_success_rate": 1.0,
                        "current_walk_global_mpjpe_mean": 0.10,
                        "current_run_success_rate": 0.0,
                        "current_backflip_success_rate": 1.0,
                        "current_backflip_global_mpjpe_mean": 0.30,
                    },
                    {
                        "eval_index": 2,
                        "frames": 2_000,
                        "current_walk_success_rate": 1.0,
                        "current_walk_global_mpjpe_mean": 0.08,
                        "current_run_success_rate": 1.0,
                        "current_run_global_mpjpe_mean": 0.12,
                        "current_backflip_success_rate": 1.0,
                        "current_backflip_global_mpjpe_mean": 0.25,
                    },
                ],
            },
            {
                "seed": 2,
                "train_returncode": 0,
                "benchmark_curve": [
                    {
                        "eval_index": 1,
                        "frames": 1_000,
                        "current_walk_success_rate": 1.0,
                        "current_walk_global_mpjpe_mean": 0.12,
                        "current_run_success_rate": 0.5,
                        "current_run_global_mpjpe_mean": 0.20,
                        "current_backflip_success_rate": 0.0,
                    },
                    {
                        "eval_index": 2,
                        "frames": 2_000,
                        "current_walk_success_rate": 1.0,
                        "current_walk_global_mpjpe_mean": 0.10,
                        "current_run_success_rate": 1.0,
                        "current_run_global_mpjpe_mean": 0.14,
                        "current_backflip_success_rate": 1.0,
                        "current_backflip_global_mpjpe_mean": 0.35,
                    },
                ],
            },
        ]

        curve, invalid_reasons = _aggregate_axes_pareto_curve(rows, task_cfg)
        summary = _build_axes_pareto_summary(
            [{"name": "candidate", "pareto_curve": curve}], task_cfg
        )

        self.assertEqual(invalid_reasons, [])
        self.assertAlmostEqual(curve[0]["run_global_mpjpe"], 0.8)
        self.assertAlmostEqual(curve[0]["backflip_global_mpjpe"], 0.65)
        self.assertEqual(
            [axis["key"] for axis in summary["axes"]],
            ["walk_global_mpjpe", "run_global_mpjpe", "backflip_global_mpjpe"],
        )
        self.assertEqual(summary["mode"], "axes")
        self.assertEqual(summary["selected"]["experiment"], "candidate")
        self.assertAlmostEqual(summary["selected"]["walk_global_mpjpe"], 0.09)
        self.assertAlmostEqual(summary["selected"]["run_global_mpjpe"], 0.13)
        self.assertAlmostEqual(summary["selected"]["backflip_global_mpjpe"], 0.30)

    def test_axes_pareto_rejects_missing_censored_metric_after_any_success(
        self,
    ) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["env.reward.alive"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "axes_pareto",
                    "frame_budget": 100,
                    "axes": [
                        {
                            "key": axis,
                            "goal": "maximize",
                            "value": {
                                "type": "terminal",
                                "metric": f"current_{axis}",
                                "seed_reduce": "mean",
                            },
                            "failure_censor": {
                                "success_rate_metric": f"current_{axis}_success_rate",
                                "failure_value": 0.0,
                            },
                        }
                        for axis in ("task_quality", "style_quality")
                    ],
                },
            }
        )
        rows = [
            {
                "seed": 1,
                "train_returncode": 0,
                "benchmark_curve": [
                    {
                        "eval_index": 1,
                        "frames": 100,
                        "current_task_quality_success_rate": 0.5,
                        "current_style_quality_success_rate": 0.0,
                    }
                ],
            }
        ]

        curve, invalid_reasons = _aggregate_axes_pareto_curve(rows, task_cfg)

        self.assertEqual(curve, [])
        self.assertEqual(
            invalid_reasons,
            ["eval_index_1_missing_axis_metric:task_quality"],
        )


class FrontierEvaluatorMetricsTest(unittest.TestCase):
    def test_pareto_curve_uses_median_time_and_mean_fixed_quality(self) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["agent.learning_rate.initial"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1, 2, 3],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "pareto",
                    "time": {"key": "active_train_time", "epsilon": 0.0},
                    "quality": {
                        "key": "current_reward_mean",
                        "goal": "maximize",
                        "aggregate": "mean",
                        "epsilon": 0.0,
                    },
                    "selection": "knee",
                },
            }
        )
        rows = [
            {
                "seed": seed,
                "train_returncode": 0,
                "benchmark_curve": [
                    {
                        "eval_index": 1.0,
                        "active_train_time": time_1,
                        "current_reward_mean": quality_1,
                    },
                    {
                        "eval_index": 2.0,
                        "active_train_time": time_2,
                        "current_reward_mean": quality_2,
                    },
                ],
            }
            for seed, time_1, quality_1, time_2, quality_2 in [
                (1, 9.0, 80.0, 19.0, 100.0),
                (2, 10.0, 90.0, 20.0, 110.0),
                (3, 50.0, 100.0, 60.0, 120.0),
            ]
        ]

        curve, invalid_reasons = _aggregate_pareto_curve(rows, task_cfg)

        self.assertEqual(invalid_reasons, [])
        self.assertEqual(len(curve), 2)
        self.assertEqual(curve[0]["active_train_time"], 10.0)
        self.assertEqual(curve[0]["quality"], 90.0)
        self.assertEqual(curve[1]["active_train_time"], 20.0)
        self.assertEqual(curve[1]["quality"], 110.0)
        self.assertEqual(curve[0]["run_count"], 3)
        self.assertEqual(curve[0]["time_max_over_median"], 5.0)
        self.assertEqual(curve[0]["quality_spread"], 20.0)

    def test_pareto_curve_rejects_mismatched_seed_evaluation_points(self) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["agent.learning_rate.initial"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1, 2, 3],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "pareto",
                    "time": {"key": "active_train_time"},
                    "quality": {
                        "key": "current_reward_mean",
                        "goal": "maximize",
                    },
                },
            }
        )
        rows = [
            {
                "seed": seed,
                "train_returncode": 0,
                "benchmark_curve": [
                    {
                        "eval_index": 1.0,
                        "active_train_time": 10.0,
                        "current_reward_mean": 90.0,
                    }
                ]
                + (
                    [
                        {
                            "eval_index": 2.0,
                            "active_train_time": 20.0,
                            "current_reward_mean": 110.0,
                        }
                    ]
                    if seed != 3
                    else []
                ),
            }
            for seed in [1, 2, 3]
        ]

        curve, invalid_reasons = _aggregate_pareto_curve(rows, task_cfg)

        self.assertEqual(curve, [])
        self.assertEqual(invalid_reasons, ["mismatched_eval_indices"])

    def test_pareto_summary_keeps_frontier_and_named_representatives(self) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["agent.learning_rate.initial"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1, 2, 3],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "pareto",
                    "time": {"key": "active_train_time", "epsilon": 0.0},
                    "quality": {
                        "key": "current_reward_mean",
                        "goal": "maximize",
                        "aggregate": "mean",
                        "epsilon": 0.0,
                    },
                    "selection": "knee",
                },
            }
        )
        experiments = [
            {
                "name": "candidate_a",
                "pareto_curve": [
                    {"eval_index": 1, "active_train_time": 9.0, "quality": 80.0},
                    {
                        "eval_index": 2,
                        "active_train_time": 10.0,
                        "quality": 90.0,
                    },
                ],
            },
            {
                "name": "candidate_b",
                "pareto_curve": [
                    {
                        "eval_index": 1,
                        "active_train_time": 12.0,
                        "quality": 100.0,
                    },
                    {
                        "eval_index": 2,
                        "active_train_time": 13.0,
                        "quality": 95.0,
                    },
                ],
            },
        ]

        summary = _build_pareto_summary(experiments, task_cfg)

        self.assertEqual(
            [point["experiment"] for point in summary["frontier"]],
            ["candidate_a", "candidate_a", "candidate_b"],
        )
        self.assertEqual(
            summary["representatives"]["fastest"]["experiment"], "candidate_a"
        )
        self.assertEqual(
            summary["representatives"]["quality"]["experiment"], "candidate_b"
        )
        self.assertEqual(summary["selected"]["experiment"], "candidate_a")

    def test_learning_curve_pareto_emits_one_final_and_anytime_point_per_candidate(
        self,
    ) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["agent.learning_rate.initial"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1, 2],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "learning_curve_pareto",
                    "active_time_budget_seconds": 30.0,
                    "terminal_time_tolerance_seconds": 1.0,
                    "final_quality": {
                        "key": "current_reward_mean",
                        "aggregate": "mean",
                        "epsilon": 0.0,
                    },
                    "anytime_quality": {"epsilon": 0.0},
                    "selection": "final_quality_constrained",
                    "anytime_quality_floor": 75.0,
                },
            }
        )
        experiments = [
            {
                "name": "slow_high",
                "pareto_curve": [
                    {"eval_index": 1, "active_train_time": 10.0, "quality": 40.0},
                    {"eval_index": 2, "active_train_time": 20.0, "quality": 80.0},
                    {
                        "eval_index": 3,
                        "active_train_time": 30.0,
                        "quality": 100.0,
                    },
                ],
            },
            {
                "name": "fast_lower",
                "pareto_curve": [
                    {"eval_index": 1, "active_train_time": 10.0, "quality": 70.0},
                    {"eval_index": 2, "active_train_time": 20.0, "quality": 85.0},
                    {"eval_index": 3, "active_train_time": 30.0, "quality": 95.0},
                ],
            },
            {
                "name": "dominated",
                "pareto_curve": [
                    {"eval_index": 1, "active_train_time": 10.0, "quality": 30.0},
                    {"eval_index": 2, "active_train_time": 20.0, "quality": 60.0},
                    {"eval_index": 3, "active_train_time": 30.0, "quality": 90.0},
                ],
            },
        ]

        summary = _build_learning_curve_pareto_summary(experiments, task_cfg)

        self.assertEqual(len(summary["points"]), 3)
        self.assertEqual(
            [point["experiment"] for point in summary["frontier"]],
            ["slow_high", "fast_lower"],
        )
        slow_high = next(
            point for point in summary["points"] if point["experiment"] == "slow_high"
        )
        self.assertEqual(slow_high["final_quality"], 100.0)
        self.assertAlmostEqual(slow_high["anytime_quality"], 1900.0 / 30.0)
        self.assertEqual(
            summary["recommendations"]["final_quality"]["experiment"],
            "slow_high",
        )
        self.assertEqual(
            summary["recommendations"]["anytime_quality"]["experiment"],
            "fast_lower",
        )
        self.assertEqual(summary["selected"]["experiment"], "fast_lower")

    def test_learning_curve_pareto_supports_a_fixed_frame_budget(self) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["agent.learning_rate.initial"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1, 2],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "learning_curve_pareto",
                    "frame_budget": 2_000,
                    "terminal_frame_tolerance": 0,
                    "final_quality": {
                        "key": "current_reward_mean",
                        "aggregate": "median",
                        "statistic": "best_per_seed",
                    },
                    "anytime_quality": {},
                    "selection": "balanced",
                },
            }
        )
        rows = [
            {
                "seed": 1,
                "train_returncode": 0,
                "benchmark_curve": [
                    {
                        "eval_index": 1,
                        "frames": 1_000,
                        "active_train_time": 10.0,
                        "current_reward_mean": 40.0,
                    },
                    {
                        "eval_index": 2,
                        "frames": 2_000,
                        "active_train_time": 20.0,
                        "current_reward_mean": 80.0,
                    },
                ],
            },
            {
                "seed": 2,
                "train_returncode": 0,
                "benchmark_curve": [
                    {
                        "eval_index": 1,
                        "frames": 1_000,
                        "active_train_time": 12.0,
                        "current_reward_mean": 60.0,
                    },
                    {
                        "eval_index": 2,
                        "frames": 2_000,
                        "active_train_time": 24.0,
                        "current_reward_mean": 100.0,
                    },
                ],
            },
        ]

        curve, invalid_reasons = _aggregate_pareto_curve(rows, task_cfg)
        summary = _build_learning_curve_pareto_summary(
            [{"name": "candidate", "pareto_curve": curve}], task_cfg
        )

        self.assertEqual(invalid_reasons, [])
        self.assertEqual(curve[-1]["frames"], 2_000.0)
        self.assertEqual(summary["budget"]["key"], "frames")
        self.assertEqual(summary["points"][0]["terminal_frames"], 2_000.0)
        self.assertEqual(summary["points"][0]["final_quality"], 90.0)
        self.assertEqual(summary["points"][0]["anytime_quality"], 60.0)

    def test_learning_curve_pareto_rejects_a_candidate_outside_the_time_budget(
        self,
    ) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["agent.learning_rate.initial"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1, 2, 3],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "learning_curve_pareto",
                    "active_time_budget_seconds": 30.0,
                    "terminal_time_tolerance_seconds": 1.0,
                    "final_quality": {
                        "key": "current_reward_mean",
                        "aggregate": "mean",
                    },
                    "anytime_quality": {},
                },
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "candidate_a.*terminal active train time.*fixed budget",
        ):
            _build_learning_curve_pareto_summary(
                [
                    {
                        "name": "candidate_a",
                        "pareto_curve": [
                            {
                                "eval_index": 1,
                                "active_train_time": 10.0,
                                "quality": 40.0,
                            },
                            {
                                "eval_index": 2,
                                "active_train_time": 35.0,
                                "quality": 90.0,
                            },
                        ],
                    }
                ],
                task_cfg,
            )

    def test_learning_curve_pareto_can_use_median_seedwise_best_quality(self) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["agent.learning_rate.initial"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1, 2, 3],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "learning_curve_pareto",
                    "active_time_budget_seconds": 20.0,
                    "terminal_time_tolerance_seconds": 0.1,
                    "final_quality": {
                        "key": "current_reward_mean",
                        "aggregate": "median",
                        "statistic": "best_per_seed",
                    },
                    "anytime_quality": {},
                },
            }
        )

        summary = _build_learning_curve_pareto_summary(
            [
                {
                    "name": "candidate",
                    "pareto_curve": [
                        {
                            "eval_index": 1,
                            "active_train_time": 10.0,
                            "quality": 50.0,
                            "quality_values": [100.0, 40.0, 50.0],
                        },
                        {
                            "eval_index": 2,
                            "active_train_time": 20.0,
                            "quality": 20.0,
                            "quality_values": [10.0, 80.0, 20.0],
                        },
                    ],
                }
            ],
            task_cfg,
        )

        point = summary["points"][0]
        self.assertEqual(point["final_quality"], 80.0)
        self.assertEqual(point["best_quality_values"], [100.0, 80.0, 50.0])
        self.assertEqual(point["best_eval_indices"], [1, 2, 1])
        self.assertEqual(point["terminal_quality"], 20.0)
        self.assertEqual(summary["final_quality"]["statistic"], "best_per_seed")

    def test_learning_curve_pareto_minimizes_failure_censored_mpjpe(self) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["env.reward.alive"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1, 2],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "learning_curve_pareto",
                    "frame_budget": 2_000,
                    "terminal_frame_tolerance": 0,
                    "final_quality": {
                        "key": "current_global_mpjpe_mean",
                        "goal": "minimize",
                        "aggregate": "mean",
                        "failure_censor": {
                            "success_rate_key": "current_success_rate",
                            "failure_value": 1.0,
                        },
                    },
                    "anytime_quality": {},
                    "selection": "balanced",
                },
            }
        )
        rows = [
            {
                "seed": 1,
                "train_returncode": 0,
                "benchmark_curve": [
                    {
                        "eval_index": 1,
                        "frames": 1_000,
                        "current_success_rate": 0.0,
                    },
                    {
                        "eval_index": 2,
                        "frames": 2_000,
                        "current_success_rate": 1.0,
                        "current_global_mpjpe_mean": 0.05,
                    },
                ],
            },
            {
                "seed": 2,
                "train_returncode": 0,
                "benchmark_curve": [
                    {
                        "eval_index": 1,
                        "frames": 1_000,
                        "current_success_rate": 0.5,
                        "current_global_mpjpe_mean": 0.10,
                    },
                    {
                        "eval_index": 2,
                        "frames": 2_000,
                        "current_success_rate": 1.0,
                        "current_global_mpjpe_mean": 0.07,
                    },
                ],
            },
        ]

        curve, invalid_reasons = _aggregate_pareto_curve(rows, task_cfg)
        summary = _build_learning_curve_pareto_summary(
            [{"name": "candidate", "pareto_curve": curve}], task_cfg
        )

        self.assertEqual(invalid_reasons, [])
        self.assertEqual(curve[0]["quality_values"], [1.0, 0.55])
        self.assertEqual(curve[0]["quality"], 0.775)
        self.assertEqual(summary["final_quality"]["goal"], "minimize")
        self.assertEqual(summary["anytime_quality"]["goal"], "minimize")
        self.assertAlmostEqual(summary["points"][0]["final_quality"], 0.06)

    def test_learning_curve_pareto_minimize_recommendations_follow_both_axes(
        self,
    ) -> None:
        task_cfg = OmegaConf.create(
            {
                "parameter_space": ["env.reward.alive"],
                "batch": {
                    "eval_source": "train_benchmark",
                    "seeds": [1, 2],
                    "lock_motion_configs": True,
                },
                "objective": {
                    "strategy": "learning_curve_pareto",
                    "frame_budget": 2_000,
                    "terminal_frame_tolerance": 0,
                    "final_quality": {
                        "key": "current_global_mpjpe_mean",
                        "goal": "minimize",
                        "aggregate": "mean",
                    },
                    "anytime_quality": {},
                    "selection": "balanced",
                },
            }
        )
        experiments = [
            {
                "name": "best_final",
                "pareto_curve": [
                    {"eval_index": 1, "frames": 1_000, "quality": 0.50},
                    {"eval_index": 2, "frames": 2_000, "quality": 0.05},
                ],
            },
            {
                "name": "best_anytime",
                "pareto_curve": [
                    {"eval_index": 1, "frames": 1_000, "quality": 0.08},
                    {"eval_index": 2, "frames": 2_000, "quality": 0.07},
                ],
            },
        ]

        summary = _build_learning_curve_pareto_summary(experiments, task_cfg)

        self.assertEqual(
            summary["recommendations"]["final_quality"]["experiment"],
            "best_final",
        )
        self.assertEqual(
            summary["recommendations"]["anytime_quality"]["experiment"],
            "best_anytime",
        )
