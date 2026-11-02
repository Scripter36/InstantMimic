"""The one train-log parser, and the derivations built on it.

Two parsers used to read the same training logs: one for the scalar evaluator
and one for the Pareto evaluator. There is one now -- ``evidence.parse_train_log``
-- and every objective-specific number is derived from its output by
``objective.curves``. These tests were written against the two old parsers and
carry the same expected values over unchanged, which is what shows the move
preserved the math.
"""

from __future__ import annotations

import math
import pathlib

import pytest

from instantmimic.hpo import evaluate, evidence
from instantmimic.hpo.objective import curves

# A recorded-shape training log: three benchmark evaluations, a reached
# tracking benchmark, the two time-to markers, and the completion line.
TRAIN_LOG = """\
[trainer] starting run
BENCHMARK_EVAL active_train_time=10.0 frames=1000 current_reward_mean=1.0 \
current_episode_length_mean=50.0 current_success_rate=0.20 \
current_ape_mean=0.30 current_ee_mean=0.40 current_global_mpjpe_mean=0.60 \
total_envs=64 current_success_count=12
[trainer] iteration 10
BENCHMARK_EVAL active_train_time=20.0 frames=2000 current_reward_mean=3.0 \
current_episode_length_mean=70.0 current_success_rate=0.50 \
current_ape_mean=0.20 current_ee_mean=0.25 current_global_mpjpe_mean=0.40 \
total_envs=64 current_success_count=32
BENCHMARK_EVAL active_train_time=30.0 frames=3000 current_reward_mean=5.0 \
current_episode_length_mean=90.0 current_success_rate=0.90 \
current_ape_mean=0.04 current_ee_mean=0.05 current_global_mpjpe_mean=0.08 \
total_envs=64 current_success_count=58
TRACKING_BENCHMARK_REACHED frames=3000 success_rate=0.90 ape_mean=0.04 \
ee_mean=0.05 global_mpjpe_mean=0.08
TIME_TO_REWARD_THRESHOLD_SECONDS=18.5
TIME_TO_TRACKING_BENCHMARK_SECONDS=30.25
TRAINING_COMPLETE iterations=375 best_mean_reward=5.25
"""


@pytest.fixture(name="train_log")
def _train_log(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "train.log"
    path.write_text(TRAIN_LOG, encoding="utf-8")
    return path


class TestObjectiveNaming:
    """Which objective the flat scalar keys resolve to."""

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, "wall_clock"),
            ({"reward_threshold": 5.0}, "time_to_reward_threshold_excluding_warmup"),
            (
                {"benchmark_ape_threshold": 0.05},
                "time_to_tracking_benchmark_excluding_warmup",
            ),
            (
                {
                    "benchmark_ape_threshold": 0.05,
                    "benchmark_objective_unit": "frames",
                },
                "frames_to_tracking_benchmark",
            ),
            (
                {
                    "benchmark_ape_threshold": 0.05,
                    "benchmark_objective_mode": "auc",
                },
                "tracking_benchmark_auc",
            ),
            (
                {"reward_threshold": 5.0, "benchmark_success_rate_threshold": 0.9},
                "time_to_tracking_benchmark_excluding_warmup",
            ),
        ],
    )
    def test_objective_name_from_thresholds(
        self, kwargs: dict[str, object], expected: str
    ) -> None:
        defaults: dict[str, object] = {
            "reward_threshold": None,
            "benchmark_ape_threshold": None,
            "benchmark_ee_threshold": None,
            "benchmark_success_rate_threshold": None,
        }
        assert curves.objective_name_for(**{**defaults, **kwargs}) == expected

    @pytest.mark.parametrize(
        ("objective_name", "label"),
        [
            ("wall_clock", "WALL_SECONDS"),
            (
                "time_to_reward_threshold_excluding_warmup",
                "TIME_TO_REWARD_THRESHOLD_SECONDS",
            ),
            (
                "time_to_tracking_benchmark_excluding_warmup",
                "TIME_TO_TRACKING_BENCHMARK_SECONDS",
            ),
            ("frames_to_tracking_benchmark", "FRAMES_TO_TRACKING_BENCHMARK"),
            ("tracking_benchmark_auc", "TRACKING_BENCHMARK_ERROR_AUC"),
        ],
    )
    def test_objective_label(self, objective_name: str, label: str) -> None:
        assert curves.objective_label(objective_name) == label

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, "seconds"),
            ("seconds", "seconds"),
            ("time", "seconds"),
            ("frames", "frames"),
            ("tuples", "frames"),
            ("FRAMES", "frames"),
        ],
    )
    def test_unit_normalization(self, raw: str | None, expected: str) -> None:
        assert curves.normalize_benchmark_objective_unit(raw) == expected

    def test_unknown_unit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="'benchmark_objective_unit'"):
            curves.normalize_benchmark_objective_unit("meters")


class TestTimeAndFrameExtraction:
    def test_time_to_reward_threshold_is_read_from_the_marker(
        self, train_log: pathlib.Path
    ) -> None:
        train = evidence.parse_train_log(train_log)

        assert train.threshold_events["time_to_reward_threshold_seconds"] == 18.5

    def test_time_to_tracking_benchmark_is_read_from_the_marker(
        self, train_log: pathlib.Path
    ) -> None:
        train = evidence.parse_train_log(train_log)

        assert train.threshold_events["time_to_tracking_benchmark_seconds"] == 30.25

    def test_the_last_marker_wins(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "train.log"
        path.write_text(
            "TIME_TO_REWARD_THRESHOLD_SECONDS=10.0\n"
            "TIME_TO_REWARD_THRESHOLD_SECONDS=20.0\n",
            encoding="utf-8",
        )

        train = evidence.parse_train_log(path)

        assert train.threshold_events["time_to_reward_threshold_seconds"] == 20.0

    def test_a_missing_log_yields_no_value(self, tmp_path: pathlib.Path) -> None:
        train = evidence.parse_train_log(tmp_path / "absent.log")

        assert train.threshold_events == {}
        assert train.benchmark_curve == []

    def test_frames_to_tracking_benchmark(self, train_log: pathlib.Path) -> None:
        train = evidence.parse_train_log(train_log)

        assert train.reached_tracking_benchmark["frames"] == 3000.0

    def test_frames_ignores_a_reached_line_without_frames(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "train.log"
        path.write_text(
            "TRACKING_BENCHMARK_REACHED success_rate=0.9 ape_mean=0.04\n",
            encoding="utf-8",
        )

        train = evidence.parse_train_log(path)

        assert "frames" not in train.reached_tracking_benchmark

    def test_threshold_metrics_come_from_the_reached_line(
        self, train_log: pathlib.Path
    ) -> None:
        reached = evidence.parse_train_log(train_log).reached_tracking_benchmark

        assert (
            reached["success_rate"],
            reached["ape_mean"],
            reached["global_mpjpe_mean"],
            reached["ee_mean"],
        ) == (0.90, 0.04, 0.08, 0.05)


class TestIntervalBucket:
    @pytest.mark.parametrize(
        ("objective_seconds", "interval_seconds", "expected"),
        [
            (None, 10.0, None),
            (30.0, None, None),
            (30.0, 0.0, None),
            (0.0, 10.0, 0),
            (-5.0, 10.0, 0),
            (10.0, 10.0, 1),
            (10.000001, 10.0, 2),
            (25.0, 10.0, 3),
        ],
    )
    def test_bucket_is_the_ceiling_of_the_interval_count(
        self,
        objective_seconds: float | None,
        interval_seconds: float | None,
        expected: int | None,
    ) -> None:
        assert (
            curves.objective_interval_bucket(objective_seconds, interval_seconds)
            == expected
        )


class TestBenchmarkScore:
    """``_compute_benchmark_score`` is the scalar tiebreak score."""

    def test_thresholded_terms_are_normalized_and_maxed(self) -> None:
        score = curves.benchmark_score(
            success_rate=0.5,
            ape_mean=0.10,
            ee_mean=0.10,
            benchmark_success_rate_threshold=0.9,
            benchmark_ape_threshold=0.05,
            benchmark_ee_threshold=0.20,
        )

        # max(0.9/0.5, 0.10/0.05, 0.10/0.20) == 2.0
        assert score == 2.0

    def test_untresholded_metrics_fall_back_to_raw_values(self) -> None:
        score = curves.benchmark_score(
            success_rate=0.5,
            ape_mean=0.10,
            global_mpjpe_mean=0.25,
            ee_mean=None,
        )

        # max(1/0.5, 0.10, 0.25) == 2.0
        assert score == 2.0

    def test_no_metrics_yields_no_score(self) -> None:
        assert curves.benchmark_score(success_rate=None, ape_mean=None) is None

    def test_non_finite_metrics_are_dropped(self) -> None:
        score = curves.benchmark_score(
            success_rate=float("nan"),
            ape_mean=0.10,
            benchmark_ape_threshold=0.05,
        )

        assert score == 2.0

    def test_best_benchmark_eval_is_the_lowest_scoring_point(
        self, train_log: pathlib.Path
    ) -> None:
        best = curves.best_benchmark_metrics(
            evidence.parse_train_log(train_log),
            curves.Thresholds(success_rate=0.9, ape_mean=0.05),
        )

        assert best is not None
        assert best.success_rate == 0.90
        assert best.ape_mean == 0.04
        assert best.ee_mean == 0.05
        assert best.score == pytest.approx(1.0)

    def test_eval_best_checkpoint_lines_participate(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "train.log"
        path.write_text(
            "BENCHMARK_EVAL active_train_time=1.0 current_success_rate=0.5 "
            "current_ape_mean=0.10\n"
            "EVAL_BEST_CHECKPOINT success_rate=0.95 ape_mean=0.02\n",
            encoding="utf-8",
        )

        best = curves.best_benchmark_metrics(
            evidence.parse_train_log(path),
            curves.Thresholds(success_rate=0.9, ape_mean=0.05),
        )

        assert best is not None
        assert best.ape_mean == 0.02
        assert best.score == pytest.approx(0.9 / 0.95)

    def test_gate_prefixed_payloads_win_over_current(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "train.log"
        path.write_text(
            "BENCHMARK_EVAL active_train_time=1.0 current_ape_mean=0.99 "
            "gate_ape_mean=0.01 gate_success_rate=1.0\n",
            encoding="utf-8",
        )

        best = curves.best_benchmark_metrics(
            evidence.parse_train_log(path), curves.Thresholds(ape_mean=0.05)
        )

        assert best is not None
        assert best.ape_mean == 0.01


class TestBenchmarkAUC:
    def test_auc_is_the_time_average_of_the_best_so_far_error(
        self, train_log: pathlib.Path
    ) -> None:
        result = curves.benchmark_auc(
            evidence.parse_train_log(train_log),
            curves.Thresholds(ape_mean=0.05),
            horizon_seconds=None,
        )

        assert result is not None
        # Point scores are the APE means clamped to the error cap (0.05 * 1.2
        # = 0.06), so the best-so-far series at t=10,20,30 is 0.06, 0.06, 0.04.
        # The integral takes each interval at its right endpoint's score:
        # 10*0.06 + 10*0.06 + 10*0.04, averaged over the 30s horizon.
        assert result.auc_score == pytest.approx((0.6 + 0.6 + 0.4) / 30.0)
        assert result.best_metrics.ape_mean == 0.04

    def test_horizon_extends_the_integral_with_the_last_score(
        self, train_log: pathlib.Path
    ) -> None:
        result = curves.benchmark_auc(
            evidence.parse_train_log(train_log),
            curves.Thresholds(ape_mean=0.05),
            horizon_seconds=60.0,
        )

        assert result is not None
        # The tail from the last evaluation to the horizon is held at 0.04.
        assert result.auc_score == pytest.approx((0.6 + 0.6 + 0.4 + 0.04 * 30.0) / 60.0)

    def test_a_horizon_shorter_than_the_run_is_widened(
        self, train_log: pathlib.Path
    ) -> None:
        clipped = curves.benchmark_auc(
            evidence.parse_train_log(train_log),
            curves.Thresholds(ape_mean=0.05),
            horizon_seconds=5.0,
        )
        natural = curves.benchmark_auc(
            evidence.parse_train_log(train_log),
            curves.Thresholds(ape_mean=0.05),
            horizon_seconds=None,
        )

        assert clipped is not None and natural is not None
        assert clipped.auc_score == pytest.approx(natural.auc_score)

    def test_error_cap_defaults_to_the_ape_threshold_times_1_2(self) -> None:
        assert curves.resolve_auc_error_cap(
            benchmark_auc_error_cap=None,
            benchmark_ape_threshold=0.05,
        ) == pytest.approx(0.06)

    def test_error_cap_falls_back_through_mpjpe_then_ee_then_one(self) -> None:
        assert curves.resolve_auc_error_cap(
            benchmark_auc_error_cap=None,
            benchmark_ape_threshold=None,
            benchmark_global_mpjpe_threshold=0.10,
        ) == pytest.approx(0.12)
        assert curves.resolve_auc_error_cap(
            benchmark_auc_error_cap=None,
            benchmark_ape_threshold=None,
            benchmark_ee_threshold=0.20,
        ) == pytest.approx(0.24)
        assert (
            curves.resolve_auc_error_cap(
                benchmark_auc_error_cap=None, benchmark_ape_threshold=None
            )
            == 1.0
        )

    def test_point_score_is_capped_by_the_error_cap(self) -> None:
        score = curves.auc_point_score(
            success_rate=None,
            ape_mean=10.0,
            benchmark_ape_threshold=0.05,
            benchmark_auc_error_cap=0.06,
        )

        assert score == pytest.approx(0.06)

    def test_points_without_an_active_train_time_are_skipped(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "train.log"
        path.write_text("BENCHMARK_EVAL current_ape_mean=0.10\n", encoding="utf-8")

        assert (
            curves.benchmark_auc(
                evidence.parse_train_log(path),
                curves.Thresholds(ape_mean=0.05),
                horizon_seconds=None,
            )
            is None
        )


class TestBatchBenchmarkCurve:
    """The Pareto-side parser over the same log."""

    def test_curve_carries_every_numeric_field_of_every_eval(
        self, train_log: pathlib.Path
    ) -> None:
        metrics = evaluate._train_benchmark_log_metrics(train_log, recent_window=3)

        assert metrics["benchmark_eval_count"] == 3
        curve = metrics["benchmark_curve"]
        assert len(curve) == 3
        assert curve[0]["active_train_time"] == 10.0
        assert curve[0]["frames"] == 1000.0
        assert curve[0]["current_reward_mean"] == 1.0
        assert curve[-1]["current_ape_mean"] == 0.04
        assert curve[-1]["current_global_mpjpe_mean"] == 0.08

    def test_last_and_recent_window_means(self, train_log: pathlib.Path) -> None:
        metrics = evaluate._train_benchmark_log_metrics(train_log, recent_window=2)

        assert metrics["benchmark_reward_mean_last"] == 5.0
        assert metrics["benchmark_reward_mean_recent"] == pytest.approx(4.0)
        assert metrics["benchmark_episode_length_mean_last"] == 90.0
        assert metrics["benchmark_episode_length_mean_recent"] == pytest.approx(80.0)
        assert metrics["benchmark_success_rate_last"] == 0.90
        assert metrics["benchmark_success_rate_recent"] == pytest.approx(0.70)
        assert metrics["benchmark_ape_mean_last"] == 0.04
        assert metrics["benchmark_ape_mean_recent"] == pytest.approx(0.12)
        assert metrics["benchmark_ee_mean_last"] == 0.05
        assert metrics["benchmark_ee_mean_recent"] == pytest.approx(0.15)

    def test_a_window_wider_than_the_curve_averages_everything(
        self, train_log: pathlib.Path
    ) -> None:
        metrics = evaluate._train_benchmark_log_metrics(train_log, recent_window=99)

        assert metrics["benchmark_reward_mean_recent"] == pytest.approx(3.0)

    def test_failure_count_comes_from_the_last_eval(
        self, train_log: pathlib.Path
    ) -> None:
        metrics = evaluate._train_benchmark_log_metrics(train_log, recent_window=3)

        assert metrics["benchmark_failure_count_last"] == 6

    def test_threshold_frames_record_the_first_crossing(
        self, train_log: pathlib.Path
    ) -> None:
        metrics = evaluate._train_benchmark_log_metrics(
            train_log, recent_window=3, reward_thresholds=[1.0, 2.5, 4.0, 99.0]
        )

        assert metrics["benchmark_threshold_frames"] == {
            "1": 1000,
            "2p5": 2000,
            "4": 3000,
        }

    def test_an_empty_log_yields_empty_metrics(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "train.log"
        path.write_text("nothing here\n", encoding="utf-8")

        metrics = evaluate._train_benchmark_log_metrics(path, recent_window=3)

        assert metrics["benchmark_eval_count"] == 0
        assert metrics["benchmark_curve"] == []
        assert metrics["benchmark_reward_mean_last"] is None
        assert metrics["benchmark_reward_mean_recent"] is None
        assert metrics["benchmark_failure_count_last"] is None

    def test_non_finite_curve_entries_are_dropped(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "train.log"
        path.write_text(
            "BENCHMARK_EVAL active_train_time=1.0 current_reward_mean=nan "
            "current_ape_mean=0.10\n",
            encoding="utf-8",
        )

        curve = evaluate._train_benchmark_log_metrics(path, recent_window=3)[
            "benchmark_curve"
        ]

        assert "current_reward_mean" not in curve[0]
        assert curve[0]["current_ape_mean"] == 0.10

    @pytest.mark.parametrize(
        ("threshold", "key"),
        [(1.0, "1"), (2.5, "2p5"), (0.25, "0p25"), (10.0, "10")],
    )
    def test_threshold_key_formatting(self, threshold: float, key: str) -> None:
        assert evaluate._threshold_key(threshold) == key


class TestTrainingComplete:
    def test_best_mean_reward_is_read_from_the_last_marker(
        self, train_log: pathlib.Path
    ) -> None:
        assert evaluate._parse_train_best_mean_reward(train_log) == 5.25

    def test_an_explicit_none_reads_as_no_reward(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "train.log"
        path.write_text(
            "TRAINING_COMPLETE iterations=1 best_mean_reward=None\n",
            encoding="utf-8",
        )

        assert evaluate._parse_train_best_mean_reward(path) is None

    def test_a_log_without_the_marker_reads_as_no_reward(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "train.log"
        path.write_text("no completion here\n", encoding="utf-8")

        assert evaluate._parse_train_best_mean_reward(path) is None


class TestPlaybackMetrics:
    def test_payload_counts_drive_falls_when_present(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "play.log"
        path.write_text(
            "TRACKING_METRICS_SUMMARY ape_mean=0.03 ee_mean=0.04 vel_mean=0.5 "
            "accel_mean=1.5\n",
            encoding="utf-8",
        )

        metrics = evaluate._play_log_metrics(
            path,
            playback_payload={"terminated_count": "2", "truncated_count": "6"},
            play_steps=1800,
            episode_length=1800,
        )

        assert metrics["ape_mean"] == 0.03
        assert metrics["ee_mean"] == 0.04
        assert metrics["vel_mean"] == 0.5
        assert metrics["accel_mean"] == 1.5
        assert metrics["terminated_count"] == 2
        assert metrics["truncated_count"] == 6
        assert metrics["episode_finish_count"] == 8
        assert metrics["fall_count"] == 2

    def test_episode_lines_are_counted_when_no_payload_is_emitted(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "play.log"
        path.write_text(
            "Episode 1 finished: length=100 total_reward=5.0\n"
            "Episode 2 finished: length=120 total_reward=6.0\n"
            "Episode 3 finished: length=1800 total_reward=9.0\n"
            "TRACKING_METRICS_SUMMARY ape_mean=0.03 ee_mean=0.04\n",
            encoding="utf-8",
        )

        metrics = evaluate._play_log_metrics(
            path, playback_payload=None, play_steps=1800, episode_length=1800
        )

        assert metrics["episode_finish_count"] == 3
        # Reaching the horizon means the final truncation is not a fall.
        assert metrics["fall_count"] == 2
        assert metrics["truncated_count"] == 1

    def test_without_reaching_the_horizon_every_finish_is_a_fall(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "play.log"
        path.write_text(
            "Episode 1 finished: length=100 total_reward=5.0\n"
            "Episode 2 finished: length=120 total_reward=6.0\n",
            encoding="utf-8",
        )

        metrics = evaluate._play_log_metrics(
            path, playback_payload=None, play_steps=1800, episode_length=120
        )

        assert metrics["episode_finish_count"] == 2
        assert metrics["fall_count"] == 2
        assert metrics["truncated_count"] == 0

    def test_the_last_tracking_summary_wins(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "play.log"
        path.write_text(
            "TRACKING_METRICS_SUMMARY ape_mean=0.99\n"
            "TRACKING_METRICS_SUMMARY ape_mean=0.01\n",
            encoding="utf-8",
        )

        metrics = evaluate._play_log_metrics(
            path,
            playback_payload={"terminated_count": "0"},
            play_steps=None,
            episode_length=None,
        )

        assert metrics["ape_mean"] == 0.01


class TestOneParserServesBoth:
    """The derivations and the running summary read the same parse."""

    def test_best_benchmark_point_matches_the_summary_last_values(
        self, train_log: pathlib.Path
    ) -> None:
        train = evidence.parse_train_log(train_log, recent_window=1)
        best = curves.best_benchmark_metrics(
            train, curves.Thresholds(success_rate=0.9, ape_mean=0.05)
        )

        assert best is not None
        assert best.ape_mean == train.summary["benchmark_ape_mean_last"]
        assert best.ee_mean == train.summary["benchmark_ee_mean_last"]
        assert best.success_rate == train.summary["benchmark_success_rate_last"]

    def test_reached_marker_matches_the_last_curve_frame(
        self, train_log: pathlib.Path
    ) -> None:
        train = evidence.parse_train_log(train_log, recent_window=1)
        frames = train.reached_tracking_benchmark["frames"]

        assert math.isclose(frames, train.benchmark_curve[-1]["frames"])
