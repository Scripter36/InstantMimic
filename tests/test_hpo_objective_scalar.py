"""Characterization of the scalar hpo comparator.

These tests freeze the decisions the scalar comparison path produces today so
the unification refactor can move that math behind
``instantmimic.hpo.objective`` without changing it. Every expected
value is a literal derived from the current implementation on synthetic
evidence; the tests are re-pointed at ``ScalarComparator`` later and must keep
passing unchanged.

Two functions carry the comparison today:

* ``ScalarComparator.outcome`` -- cycle verdict (keep / advance / discard /
  crash) against the recorded best, including the improvement margin, the tie
  epsilon, the secondary tiebreak, and local incumbent advance.
* ``scalar_cycle._select_scalar_winner`` -- winner among a sweep's candidates,
  including the risk adjustment by objective range.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from instantmimic.hpo.objective.scalar import (
    ScalarComparator,
    ScalarContext,
    ScalarObjective,
    ScalarState,
    _select_scalar_winner,
)

# Flat task keys the comparator no longer reads, mapped to what it does read.
# Keeping this table in the test lets every case below keep the exact inputs it
# was written with while the comparator takes only its own config.
_OBJECTIVE_FROM_FLAT = {
    "metric_improvement_margin": "improvement_margin",
    "objective_range_penalty_lambda": "range_penalty_lambda",
    "local_incumbent_advance": "local_incumbent_advance",
    "benchmark_objective_mode": "threshold_mode",
}


def _task_cfg(**overrides: object) -> DictConfig:
    return OmegaConf.create(dict(overrides))


def _metric_objective(task_cfg: DictConfig) -> str:
    """Which objective the task's thresholds imply."""

    if any(
        task_cfg.get(key, None) is not None
        for key in (
            "benchmark_ape_mean_threshold",
            "benchmark_global_mpjpe_mean_threshold",
            "benchmark_ee_mean_threshold",
            "benchmark_success_rate_threshold",
        )
    ):
        mode = str(task_cfg.get("benchmark_objective_mode", "threshold") or "")
        if mode.strip().lower() == "auc":
            return "tracking_benchmark_auc"
        unit = str(task_cfg.get("benchmark_objective_unit", "seconds") or "")
        if unit.strip().lower() in {"frames", "tuples"}:
            return "frames_to_tracking_benchmark"
        return "time_to_tracking_benchmark_excluding_warmup"
    if task_cfg.get("reward_threshold", None) is not None:
        return "time_to_reward_threshold_excluding_warmup"
    return "wall_clock"


def _comparator(
    task_cfg: DictConfig,
    status_cfg: DictConfig,
    root: pathlib.Path,
) -> ScalarComparator:
    """Build the comparator the controller would build for this task."""

    fields: dict[str, object] = {"metric_objective": _metric_objective(task_cfg)}
    for flat_key, field_name in _OBJECTIVE_FROM_FLAT.items():
        value = task_cfg.get(flat_key, None)
        if value is not None:
            fields[field_name] = value
    objective = ScalarObjective(**fields)  # type: ignore[arg-type]

    # The controller injects no context; only the rendering half survives.
    del root
    context = ScalarContext(
        config_summary=lambda overrides: ", ".join(
            f"{key}={value}" for key, value in overrides.items()
        ),
    )
    best_value = status_cfg.get("best_metric_seconds", None)
    best_tiebreak = status_cfg.get("best_metric_tiebreak", None)
    incumbent = status_cfg.get("incumbent_overrides", None)
    state = ScalarState(
        best_value=None if best_value in (None, "") else float(best_value),
        best_tiebreak=None if best_tiebreak in (None, "") else float(best_tiebreak),
        best_config_summary=str(status_cfg.get("best_config_summary", "") or ""),
        incumbent_overrides=(
            {str(key): str(value) for key, value in dict(incumbent).items()}
            if incumbent
            else {}
        ),
        incumbent_config_summary=str(
            status_cfg.get("incumbent_config_summary", "") or ""
        ),
        metric_objective=str(status_cfg.get("metric_objective", "") or ""),
    )
    return ScalarComparator(objective, context=context, state=state)


def _compose_config(
    root: pathlib.Path, task_cfg: DictConfig, overrides: dict[str, str]
) -> tuple[DictConfig | None, str | None]:
    """Compose the candidate's effective train config, as the controller does."""

    config_name = str(task_cfg.get("config_name", "") or "")
    config_path = str(task_cfg.get("config_path", "") or "")
    if not config_name or not config_path:
        return None, "task has no train config"
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str((root / config_path).resolve()), version_base=None
    ):
        cfg = compose(config_name=config_name)
    for key, raw_value in overrides.items():
        OmegaConf.update(cfg, key, yaml.safe_load(str(raw_value)), merge=False)
    return cfg, None


def _compose(
    root: pathlib.Path,
    task_cfg: DictConfig,
    overrides: dict[str, str],
    dotted_key: str,
) -> object | None:
    cfg, error = _compose_config(root, task_cfg, overrides)
    if error is not None or cfg is None:
        return None
    return OmegaConf.select(cfg, dotted_key, default=None)


def _synthetic_train_config(root: pathlib.Path, num_envs: int = 2048) -> None:
    """Write a minimal composable Hydra config under ``root``.

    ``ScalarComparator.local_incumbent_advance`` resolves the incumbent's
    effective value by composing the task's train config, so the advance path
    needs a real config directory. A synthetic one keeps the test independent
    of any committed ``config/`` variant.
    """
    config_dir = root / "cfg"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "tiny.yaml").write_text(
        f"env:\n  scene:\n    num_envs: {num_envs}\nagent:\n  ratio_clip: 0.1\n",
        encoding="utf-8",
    )


def _run_outcome(
    tmp_path: pathlib.Path,
    *,
    task_cfg: DictConfig,
    status_cfg: DictConfig,
    summary_doc: object | None,
    prepared_incumbent_overrides: dict[str, str] | None = None,
    evaluator_returncode: int = 0,
    root: pathlib.Path | None = None,
    compact_error_excerpt: str = "",
) -> dict[str, object]:
    comparator = _comparator(
        task_cfg, status_cfg, root if root is not None else tmp_path
    )
    return comparator.outcome(
        summary_doc=summary_doc,
        prepared_incumbent_overrides=dict(prepared_incumbent_overrides or {}),
        evidence_text=str(tmp_path / "summary.json"),
        compact_error_excerpt=compact_error_excerpt,
        failed=evaluator_returncode != 0,
    )


def _sweep_summary(
    *,
    param: str = "agent.ratio_clip",
    winner_value: str,
    winner_metric: float,
    candidates: list[dict[str, object]],
    winner_tiebreak: float | None = None,
    winner_range: float | None = None,
) -> dict[str, object]:
    winner: dict[str, object] = {
        "value": winner_value,
        "median_objective_seconds": winner_metric,
        "risk_adjusted_objective_seconds": winner_metric,
    }
    if winner_tiebreak is not None:
        winner["median_objective_tiebreak_score"] = winner_tiebreak
    if winner_range is not None:
        winner["objective_seconds_range"] = winner_range
    return {
        "objective": "wall_clock",
        "selected_overrides": {param: winner_value},
        "search_summary": [
            {"param": param, "winner": winner, "candidates": candidates}
        ],
    }


class TestBaselineOutcome:
    def test_crash_when_evaluator_returncode_is_nonzero(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(),
            status_cfg=OmegaConf.create({}),
            summary_doc={"objective_seconds": 10.0},
            evaluator_returncode=1,
            compact_error_excerpt="evaluator exited 1",
        )

        assert outcome["status"] == "crash"
        assert outcome["metric_seconds"] == 0.0
        assert outcome["should_keep"] is False
        assert outcome["observed_metric_text"] == "crash - evaluator exited 1"
        assert outcome["delta_text"] == "N/A (crash)"
        assert outcome["decision_text"] == (
            "discard - restored the pre-cycle experiment files after the crash"
        )

    def test_crash_when_summary_is_not_a_mapping(self, tmp_path: pathlib.Path) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(),
            status_cfg=OmegaConf.create({}),
            summary_doc=None,
            compact_error_excerpt="missing summary.json",
        )

        assert outcome["status"] == "crash"
        assert outcome["last_result_text"] == "crash - missing summary.json"

    @pytest.mark.parametrize(
        "metric", [None, -1.0, float("nan"), float("inf"), -float("inf")]
    )
    def test_invalid_metric_is_treated_as_a_crash(
        self, tmp_path: pathlib.Path, metric: float | None
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(),
            status_cfg=OmegaConf.create({}),
            summary_doc={"objective_seconds": metric},
        )

        assert outcome["status"] == "crash"

    def test_zero_metric_keeps_the_first_valid_cycle(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(),
            status_cfg=OmegaConf.create({}),
            summary_doc={"objective_seconds": 0.0},
        )

        assert outcome["status"] == "keep"
        assert outcome["metric_seconds"] == 0.0
        assert outcome["should_keep"] is True

    def test_failed_zero_metric_run_stays_a_crash(self, tmp_path: pathlib.Path) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(),
            status_cfg=OmegaConf.create({}),
            summary_doc={"objective_seconds": 0.0},
            evaluator_returncode=1,
        )

        assert outcome["status"] == "crash"
        assert outcome["should_keep"] is False

    @pytest.mark.parametrize(
        ("metric", "tiebreak", "expected"),
        [
            (1.0, 0.4, "discard"),
            (0.0, 0.5, "discard"),
            (0.0, 0.6, "discard"),
            (0.0, 0.4, "keep"),
        ],
    )
    def test_zero_best_preserves_comparison_and_tiebreak_rules(
        self, tmp_path: pathlib.Path, metric: float, tiebreak: float, expected: str
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(),
            status_cfg=OmegaConf.create(
                {"best_metric_seconds": 0.0, "best_metric_tiebreak": 0.5}
            ),
            summary_doc={
                "objective_seconds": metric,
                "objective_tiebreak_score": tiebreak,
            },
        )

        assert outcome["status"] == expected

    def test_first_valid_cycle_keeps_without_a_previous_best(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(),
            status_cfg=OmegaConf.create({}),
            summary_doc={
                "objective": "wall_clock",
                "objective_seconds": 12.5,
                "runs": [
                    {"objective_seconds": 12.0},
                    {"objective_seconds": 13.0},
                ],
            },
        )

        assert outcome["status"] == "keep"
        assert outcome["should_keep"] is True
        assert outcome["metric_seconds"] == 12.5
        assert outcome["delta_text"] == "N/A (first valid recorded cycle)"
        assert outcome["observed_metric_text"] == (
            "median WALL_SECONDS = 12.500000s from repeats 12.000000s, 13.000000s."
        )
        assert outcome["last_result_text"] == "keep - new best metric 12.500000s"
        assert outcome["description"] == "baseline run with current config"


class TestImprovementMargin:
    """``metric < previous_best - metric_improvement_margin`` is the keep rule."""

    @pytest.mark.parametrize(
        ("metric", "margin", "expected_status"),
        [
            (9.0, 0.0, "keep"),
            (9.999999, 0.0, "keep"),
            (10.0, 0.0, "discard"),
            (9.6, 0.5, "discard"),
            (9.4, 0.5, "keep"),
            (10.5, 0.0, "discard"),
        ],
    )
    def test_margin_decides_keep_versus_discard(
        self,
        tmp_path: pathlib.Path,
        metric: float,
        margin: float,
        expected_status: str,
    ) -> None:
        task_cfg = _task_cfg(
            local_incumbent_advance=False,
            metric_improvement_margin=margin,
        )
        outcome = _run_outcome(
            tmp_path,
            task_cfg=task_cfg,
            status_cfg=OmegaConf.create({"best_metric_seconds": 10.0}),
            summary_doc={"objective": "wall_clock", "objective_seconds": metric},
        )

        assert outcome["status"] == expected_status

    def test_negative_margin_is_rejected(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            _run_outcome(
                tmp_path,
                task_cfg=_task_cfg(metric_improvement_margin=-1.0),
                status_cfg=OmegaConf.create({"best_metric_seconds": 10.0}),
                summary_doc={"objective": "wall_clock", "objective_seconds": 9.0},
            )

    def test_improvement_text_reports_the_delta_against_the_previous_best(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(local_incumbent_advance=False),
            status_cfg=OmegaConf.create({"best_metric_seconds": 10.0}),
            summary_doc={"objective": "wall_clock", "objective_seconds": 8.25},
        )

        assert outcome["delta_text"] == (
            "better by 1.750000s versus the previous best (10.000000s)."
        )
        assert outcome["decision_text"] == (
            "keep - the cycle produced a new best result"
        )

    def test_regression_text_reports_the_gap_against_the_previous_best(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(local_incumbent_advance=False),
            status_cfg=OmegaConf.create({"best_metric_seconds": 10.0}),
            summary_doc={"objective": "wall_clock", "objective_seconds": 11.5},
        )

        assert outcome["delta_text"] == (
            "worse by 1.500000s versus the previous best (10.000000s)."
        )
        assert outcome["last_result_text"] == (
            "discard - metric 11.500000s was not better than the best 10.000000s"
        )
        assert outcome["description"] == (
            "baseline run with current config discarded as worse than best"
        )


class TestTieBreak:
    """A tie within 1e-6 seconds is broken by the secondary benchmark score."""

    def test_tied_metric_with_better_tiebreak_keeps(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(local_incumbent_advance=False),
            status_cfg=OmegaConf.create(
                {"best_metric_seconds": 10.0, "best_metric_tiebreak": 0.5}
            ),
            summary_doc={
                "objective": "wall_clock",
                "objective_seconds": 10.0000005,
                "objective_tiebreak_score": 0.4,
            },
        )

        assert outcome["status"] == "keep"
        assert outcome["delta_text"] == (
            "same primary time, but better secondary benchmark error score "
            "(0.400000 vs 0.500000)."
        )
        assert outcome["last_result_text"] == (
            "keep - matched the best primary time and improved the secondary "
            "benchmark error score"
        )

    def test_tied_metric_with_worse_tiebreak_discards(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(local_incumbent_advance=False),
            status_cfg=OmegaConf.create(
                {"best_metric_seconds": 10.0, "best_metric_tiebreak": 0.4}
            ),
            summary_doc={
                "objective": "wall_clock",
                "objective_seconds": 10.0000005,
                "objective_tiebreak_score": 0.5,
            },
        )

        assert outcome["status"] == "discard"

    def test_tie_epsilon_is_one_microsecond(self, tmp_path: pathlib.Path) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(local_incumbent_advance=False),
            status_cfg=OmegaConf.create(
                {"best_metric_seconds": 10.0, "best_metric_tiebreak": 0.5}
            ),
            summary_doc={
                "objective": "wall_clock",
                "objective_seconds": 10.000002,
                "objective_tiebreak_score": 0.1,
            },
        )

        assert outcome["status"] == "discard"


class TestMetricObjectiveChange:
    def test_changed_objective_drops_the_previous_best_and_says_so(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(reward_threshold=5.0),
            status_cfg=OmegaConf.create(
                {
                    "best_metric_seconds": 3.0,
                    "metric_objective": "wall_clock",
                }
            ),
            summary_doc={
                "objective": "time_to_reward_threshold_excluding_warmup",
                "objective_seconds": 42.0,
            },
        )

        assert outcome["status"] == "keep"
        assert outcome["delta_text"] == (
            "N/A (metric objective changed from wall_clock to "
            "time_to_reward_threshold_excluding_warmup)."
        )

    def test_frames_objective_formats_the_metric_in_frames(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(
                benchmark_ape_mean_threshold=0.05,
                benchmark_objective_unit="frames",
                local_incumbent_advance=False,
            ),
            status_cfg=OmegaConf.create(
                {
                    "best_metric_seconds": 100.0,
                    "metric_objective": "frames_to_tracking_benchmark",
                }
            ),
            summary_doc={
                "objective": "frames_to_tracking_benchmark",
                "objective_seconds": 80.0,
            },
        )

        assert outcome["status"] == "keep"
        assert outcome["observed_metric_text"] == (
            "FRAMES_TO_TRACKING_BENCHMARK = 80.000000 frames."
        )
        assert outcome["delta_text"] == (
            "better by 20.000000 frames versus the previous best (100.000000 frames)."
        )


class TestLineSearchOutcome:
    def test_sweep_winner_drives_the_metric_and_description(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(local_incumbent_advance=False),
            status_cfg=OmegaConf.create({}),
            summary_doc=_sweep_summary(
                winner_value="0.1",
                winner_metric=30.0,
                winner_range=2.0,
                candidates=[
                    {"value": "0.1", "median_objective_seconds": 30.0},
                    {"value": "0.2", "median_objective_seconds": 33.0},
                ],
            ),
        )

        assert outcome["status"] == "keep"
        assert outcome["metric_seconds"] == 30.0
        assert outcome["description"] == (
            "agent.ratio_clip line search over 0.1,0.2 kept 0.1"
        )
        assert outcome["observed_metric_text"] == (
            "winner agent.ratio_clip=0.1 with risk-adjusted WALL_SECONDS = "
            "30.000000s (median=30.000000s, range=2.000000s)."
        )
        assert outcome["incumbent_overrides"] == {"agent.ratio_clip": "0.1"}

    def test_sweep_loss_discards_and_preserves_the_previous_incumbent(
        self, tmp_path: pathlib.Path
    ) -> None:
        outcome = _run_outcome(
            tmp_path,
            task_cfg=_task_cfg(local_incumbent_advance=False),
            status_cfg=OmegaConf.create(
                {
                    "best_metric_seconds": 20.0,
                    "incumbent_overrides": {"agent.ratio_clip": "0.3"},
                    "incumbent_config_summary": "agent.ratio_clip=0.3",
                    "best_config_summary": "agent.ratio_clip=0.3",
                }
            ),
            prepared_incumbent_overrides={"agent.ratio_clip": "0.3"},
            summary_doc=_sweep_summary(
                winner_value="0.1",
                winner_metric=30.0,
                candidates=[
                    {"value": "0.1", "median_objective_seconds": 30.0},
                    {"value": "0.2", "median_objective_seconds": 33.0},
                ],
            ),
        )

        assert outcome["status"] == "discard"
        assert outcome["incumbent_overrides"] == {"agent.ratio_clip": "0.3"}
        assert outcome["best_config_summary"] == "agent.ratio_clip=0.3"
        assert outcome["description"] == (
            "agent.ratio_clip line search over 0.1,0.2 discarded as worse than best"
        )


class TestAdvanceIsRefusedNotIgnored:
    """`advance` needed the incumbent re-measured in the same cycle.

    The scalar sweep re-ran the incumbent's own value as one of its candidates,
    so "the winner beat the incumbent" was a within-cycle fact. The unified
    cycle inherits its base through `batch.fixed_train_overrides`, never re-runs
    it, and refuses a proposal that repeats an earlier override set -- so no
    cycle contains anything to compare the incumbent against.

    What `advance` should mean instead is a design question. Until it is
    answered, asking for it is an error: a flag that silently does nothing is
    worse than one that says it cannot.
    """

    def test_asking_for_advance_is_an_error_that_explains_itself(self) -> None:
        with pytest.raises(
            ValueError, match="local_incumbent_advance is not available"
        ):
            ScalarObjective.from_config(
                {"strategy": "wall_clock", "local_incumbent_advance": True}
            )

    def test_the_default_and_an_explicit_false_are_both_accepted(self) -> None:
        assert (
            ScalarObjective.from_config(
                {"strategy": "wall_clock"}
            ).local_incumbent_advance
            is False
        )
        assert (
            ScalarObjective.from_config(
                {"strategy": "wall_clock", "local_incumbent_advance": False}
            ).local_incumbent_advance
            is False
        )

    def test_the_comparator_no_longer_carries_the_removed_machinery(self) -> None:
        """The Hydra questions it asked belonged to a model that is gone."""

        assert not hasattr(ScalarComparator, "local_incumbent_advance")
        context = ScalarContext()
        assert not hasattr(context, "resolve_effective_value")
        assert not hasattr(context, "compose_effective_config")


class TestWinnerSelection:
    """``_select_winner_candidate`` ranks by risk-adjusted objective."""

    def test_lowest_objective_wins_without_a_range_penalty(self) -> None:
        candidates = [
            {"value": "a", "median_objective_seconds": 4.0},
            {"value": "b", "median_objective_seconds": 3.0},
            {"value": "c", "median_objective_seconds": 3.5},
        ]

        winner = _select_scalar_winner(candidates)

        assert winner["value"] == "b"

    def test_wall_seconds_are_used_when_no_objective_is_recorded(self) -> None:
        candidates = [
            {"value": "a", "median_wall_seconds": 5.0},
            {"value": "b", "median_wall_seconds": 4.0},
        ]

        winner = _select_scalar_winner(candidates)

        assert winner["value"] == "b"

    def test_range_penalty_prefers_the_stable_candidate(self) -> None:
        candidates = [
            {
                "value": "fast_unstable",
                "median_objective_seconds": 3.0,
                "objective_seconds_range": 4.0,
            },
            {
                "value": "slower_stable",
                "median_objective_seconds": 4.0,
                "objective_seconds_range": 0.1,
            },
        ]

        assert (
            _select_scalar_winner(candidates, range_penalty_lambda=0.0)["value"]
            == "fast_unstable"
        )
        assert (
            _select_scalar_winner(candidates, range_penalty_lambda=0.5)["value"]
            == "slower_stable"
        )

    def test_range_penalty_derives_the_range_from_seed_results(self) -> None:
        candidates = [
            {
                "value": "fast_unstable",
                "median_objective_seconds": 3.0,
                "seed_results": [
                    {"objective_seconds": 1.0},
                    {"objective_seconds": 5.0},
                ],
            },
            {
                "value": "slower_stable",
                "median_objective_seconds": 4.0,
                "seed_results": [
                    {"objective_seconds": 3.95},
                    {"objective_seconds": 4.05},
                ],
            },
        ]

        winner = _select_scalar_winner(candidates, range_penalty_lambda=0.5)

        assert winner["value"] == "slower_stable"

    def test_stored_risk_adjusted_score_wins_over_recomputation(self) -> None:
        candidates = [
            {
                "value": "a",
                "median_objective_seconds": 3.0,
                "objective_seconds_range": 10.0,
                "risk_adjusted_objective_seconds": 3.0,
            },
            {
                "value": "b",
                "median_objective_seconds": 4.0,
                "objective_seconds_range": 0.0,
                "risk_adjusted_objective_seconds": 4.0,
            },
        ]

        winner = _select_scalar_winner(candidates, range_penalty_lambda=1.0)

        assert winner["value"] == "a"

    def test_ties_are_broken_by_tiebreak_score_then_value_text(self) -> None:
        candidates = [
            {
                "value": "b",
                "median_objective_seconds": 3.0,
                "median_objective_tiebreak_score": 0.5,
            },
            {
                "value": "a",
                "median_objective_seconds": 3.0,
                "median_objective_tiebreak_score": 0.5,
            },
            {
                "value": "c",
                "median_objective_seconds": 3.0,
                "median_objective_tiebreak_score": 0.2,
            },
        ]

        assert _select_scalar_winner(candidates)["value"] == "c"
        assert _select_scalar_winner(candidates[:2])["value"] == "a"

    def test_objectives_are_compared_at_microsecond_resolution(self) -> None:
        candidates = [
            {
                "value": "b",
                "median_objective_seconds": 3.0000001,
                "median_objective_tiebreak_score": 0.9,
            },
            {
                "value": "a",
                "median_objective_seconds": 3.0,
                "median_objective_tiebreak_score": 0.1,
            },
        ]

        winner = _select_scalar_winner(candidates)

        assert winner["value"] == "a"

    def test_candidates_without_an_objective_sort_last(self) -> None:
        candidates = [
            {"value": "missing"},
            {"value": "present", "median_objective_seconds": 99.0},
        ]

        winner = _select_scalar_winner(candidates)

        assert winner["value"] == "present"

    def test_empty_candidate_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _select_scalar_winner([])


class TestScoringTheEvaluatorsOwnEvidence:
    """The scalar family's number is derived here, from real trial records.

    `ScalarComparator` used to read a document the unified evaluator never
    writes -- top-level `objective_seconds`, or a `search_summary` block the
    deleted scalar evaluator produced. Every successful cycle therefore scored
    as `crash`: no metric, no best, no incumbent, and the planner told it had
    crashed. These tests drive `compare` with the record `evaluate.py` actually
    emits, so the scorer cannot go missing again.
    """

    @staticmethod
    def _trial(candidate, seed, *, wall=None, rc=0, failure=None, train=None):
        return {
            "candidate": candidate,
            "motion": "demo",
            "seed": seed,
            "attempt": 0,
            "returncode": rc,
            "wall_seconds": wall,
            "train": train or {},
            "failure": failure,
        }

    @staticmethod
    def _comparator(tmp_path, **objective):
        from instantmimic.hpo.objective import build_comparator

        (tmp_path / "objective").mkdir(parents=True, exist_ok=True)
        comparator = build_comparator(
            {"mode": "scalar", "strategy": "wall_clock", **objective},
            state_dir=tmp_path,
        )
        comparator.load_state(tmp_path / "objective", {})
        return comparator

    def _evidence(self, cycle, trials, names):
        from instantmimic.hpo.experiment import Candidate
        from instantmimic.hpo.objective.base import Evidence

        return Evidence(
            cycle=cycle,
            lifecycle="completed",
            summary={"experiments": []},
            trials=trials,
            candidates=tuple(Candidate(name=name) for name in names),
        )

    def test_a_successful_cycle_is_scored_not_called_a_crash(self, tmp_path) -> None:
        comparator = self._comparator(tmp_path, improvement_margin=0.0)

        verdict = comparator.compare(
            self._evidence(
                1,
                [
                    self._trial("base", seed, wall=wall)
                    for seed, wall in ((1, 110.0), (2, 120.0), (3, 130.0))
                ],
                ["base"],
            )
        )

        assert verdict.decision == "keep"
        # The median across seeds, not the first or the mean.
        assert verdict.objective_value == 120.0
        assert verdict.per_candidate["base"]["objective_value"] == 120.0

    def test_the_fastest_candidate_of_a_sweep_wins(self, tmp_path) -> None:
        comparator = self._comparator(tmp_path, improvement_margin=0.0)

        verdict = comparator.compare(
            self._evidence(
                1,
                [self._trial("slow", seed, wall=150.0) for seed in (1, 2, 3)]
                + [self._trial("fast", seed, wall=100.0) for seed in (1, 2, 3)],
                ["slow", "fast"],
            )
        )

        assert verdict.decision == "keep"
        assert verdict.objective_value == 100.0
        assert verdict.representative["candidate"] == "fast"
        assert set(verdict.per_candidate) == {"slow", "fast"}

    def test_a_worse_cycle_is_discarded_not_crashed(self, tmp_path) -> None:
        comparator = self._comparator(tmp_path, improvement_margin=0.0)
        comparator.compare(
            self._evidence(1, [self._trial("a", s, wall=100.0) for s in (1, 2)], ["a"])
        )

        verdict = comparator.compare(
            self._evidence(2, [self._trial("b", s, wall=200.0) for s in (1, 2)], ["b"])
        )

        assert verdict.decision == "discard"
        assert verdict.objective_value == 200.0

    def test_a_candidate_missing_a_seed_is_not_ranked(self, tmp_path) -> None:
        """Two seeds against three is not a comparison."""

        comparator = self._comparator(tmp_path)

        verdict = comparator.compare(
            self._evidence(
                1,
                [
                    self._trial("a", 1, wall=90.0),
                    self._trial("a", 2, rc=1, failure="CUBLAS_STATUS_NOT_INITIALIZED"),
                    self._trial("a", 3, wall=90.0),
                ],
                ["a"],
            )
        )

        assert verdict.decision == "crash"
        assert verdict.objective_value is None

    def test_seed_spread_is_penalised_when_the_task_asks(self, tmp_path) -> None:
        """`objective.range_penalty_lambda` had no reader at all."""

        steady = [self._trial("steady", s, wall=w) for s, w in ((1, 100.0), (2, 100.0))]
        jumpy = [self._trial("jumpy", s, wall=w) for s, w in ((1, 60.0), (2, 130.0))]

        without = self._comparator(tmp_path / "a", improvement_margin=0.0)
        with_penalty = self._comparator(
            tmp_path / "b", improvement_margin=0.0, range_penalty_lambda=1.0
        )

        # Median 95 beats 100, so the unstable candidate wins on raw value.
        assert (
            without.compare(
                self._evidence(1, steady + jumpy, ["steady", "jumpy"])
            ).representative["candidate"]
            == "jumpy"
        )
        # 95 + 1.0 * 70 spread loses to 100 + 0.
        assert (
            with_penalty.compare(
                self._evidence(1, steady + jumpy, ["steady", "jumpy"])
            ).representative["candidate"]
            == "steady"
        )

    def test_the_best_node_survives_a_status_round_trip(self, tmp_path) -> None:
        """A fresh comparator is built every cycle and loaded from status.

        Without this, `recommend_bases()` was always empty and a proposal that
        asked to branch from `recommended` silently got `root`.
        """

        comparator = self._comparator(tmp_path, improvement_margin=0.0)
        comparator.compare(
            self._evidence(4, [self._trial("w", s, wall=10.0) for s in (1, 2)], ["w"])
        )
        assert comparator.recommend_bases() == ((4, "w"),)

        revived = self._comparator(tmp_path, improvement_margin=0.0)
        revived.load_state(
            tmp_path / "objective", comparator.dump_state(tmp_path / "objective")
        )

        assert revived.recommend_bases() == ((4, "w"),)


class TestThresholdStrategiesScoreTheirMarkers:
    """The shipped scalar configs rank by a threshold crossing, not wall time.

    `wall_clock` reads `wall_seconds` off the trial record; every other scalar
    strategy reads a marker the trainer emits into its log and `evidence.py`
    parses into `TrainEvidence`. Both halves were unreachable before the
    scorer existed, so both are pinned here.
    """

    @staticmethod
    def _comparator(tmp_path, strategy, **objective):
        from instantmimic.hpo.objective import build_comparator

        (tmp_path / "objective").mkdir(parents=True, exist_ok=True)
        comparator = build_comparator(
            {
                "mode": "scalar",
                "strategy": strategy,
                "improvement_margin": 0.0,
                **objective,
            },
            state_dir=tmp_path,
        )
        comparator.load_state(tmp_path / "objective", {})
        return comparator

    @staticmethod
    def _evidence(trials):
        from instantmimic.hpo.experiment import Candidate
        from instantmimic.hpo.objective.base import Evidence

        return Evidence(
            cycle=1,
            lifecycle="completed",
            summary={"experiments": []},
            trials=trials,
            candidates=(Candidate(name="only"),),
        )

    @staticmethod
    def _trial(seed, train):
        return {
            "candidate": "only",
            "motion": "demo",
            "seed": seed,
            "returncode": 0,
            "wall_seconds": 999.0,
            "train": train,
            "failure": None,
        }

    def test_time_to_tracking_benchmark_reads_the_trainer_marker(
        self, tmp_path: pathlib.Path
    ) -> None:
        comparator = self._comparator(
            tmp_path,
            "time_to_tracking_benchmark",
            thresholds={"success_rate": 0.9},
        )

        verdict = comparator.compare(
            self._evidence(
                [
                    self._trial(
                        seed,
                        {
                            "threshold_events": {
                                "time_to_tracking_benchmark_seconds": seconds
                            },
                            "reached_tracking_benchmark": {"success_rate": 0.95},
                        },
                    )
                    for seed, seconds in ((1, 30.0), (2, 50.0), (3, 40.0))
                ]
            )
        )

        assert verdict.decision == "keep"
        # The marker's median, not the 999s process wall on the same records.
        assert verdict.objective_value == 40.0

    def test_benchmark_reached_before_warmup_keeps_zero_elapsed_time(
        self, tmp_path: pathlib.Path
    ) -> None:
        from instantmimic.hpo.evidence import parse_train_log
        from instantmimic.hpo.objective.scalar import score_evidence
        from instantmimic.train.objective import TrainingObjectiveState

        objective_state = TrainingObjectiveState(
            enabled=True, warmup_rollouts=1, timeout_seconds=None
        )
        assert (
            objective_state.maybe_start(
                rollout_idx=0,
                now_ns=100,
                benchmark_eval_time_ns_acc=0,
                checkpoint_time_ns_acc=0,
                recording_time_ns_acc=0,
            )
            is False
        )
        seconds = objective_state.mark_benchmark_reached(
            now_ns=200,
            benchmark_eval_time_ns_acc=10,
            checkpoint_time_ns_acc=0,
            recording_time_ns_acc=0,
        )
        assert seconds == 0.0
        assert objective_state.threshold_reached is True
        assert objective_state.timer_started is False
        log_path = tmp_path / "train.log"
        log_path.write_text(
            "TRACKING_BENCHMARK_REACHED success_rate=1.0 frames=64\n"
            f"TIME_TO_TRACKING_BENCHMARK_SECONDS={seconds:.6f}\n"
            "TRAINING_COMPLETE best_mean_reward=2.0\n",
            encoding="utf-8",
        )
        train = parse_train_log(log_path)
        assert train.threshold_events["time_to_tracking_benchmark_seconds"] == 0.0
        comparator = self._comparator(
            tmp_path, "time_to_tracking_benchmark", thresholds={"success_rate": 0.9}
        )
        evidence = self._evidence([self._trial(1, train.as_dict())])
        scored = score_evidence(evidence, comparator.objective)
        assert scored is not None
        assert scored["objective_seconds"] == 0.0

        verdict = comparator.compare(evidence)

        assert verdict.decision == "keep"
        assert verdict.objective_value == 0.0
        assert comparator.state.best_value == 0.0

    def test_frames_to_tracking_benchmark_reads_the_reached_frame(
        self, tmp_path: pathlib.Path
    ) -> None:
        comparator = self._comparator(
            tmp_path,
            "frames_to_tracking_benchmark",
            thresholds={"success_rate": 0.9},
        )

        verdict = comparator.compare(
            self._evidence(
                [
                    self._trial(
                        seed,
                        {
                            "reached_tracking_benchmark": {
                                "frames": frames,
                                "success_rate": 0.95,
                            }
                        },
                    )
                    for seed, frames in ((1, 1000.0), (2, 3000.0), (3, 2000.0))
                ]
            )
        )

        assert verdict.decision == "keep"
        assert verdict.objective_value == 2000.0

    def test_a_run_that_never_reached_the_benchmark_is_not_ranked(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Ranking a miss as if it were a result is how a search goes wrong."""

        comparator = self._comparator(
            tmp_path,
            "time_to_tracking_benchmark",
            thresholds={"success_rate": 0.9},
        )

        verdict = comparator.compare(
            self._evidence(
                [self._trial(seed, {"threshold_events": {}}) for seed in (1, 2)]
            )
        )

        assert verdict.decision == "crash"
        assert verdict.objective_value is None
