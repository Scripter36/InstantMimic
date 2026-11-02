from __future__ import annotations

from dataclasses import dataclass


def _active_elapsed_seconds(
    *,
    now_ns: int,
    start_ns: int,
    benchmark_eval_time_ns_acc: int,
    start_benchmark_eval_time_ns: int,
    checkpoint_time_ns_acc: int,
    start_checkpoint_time_ns: int,
    recording_time_ns_acc: int,
    start_recording_time_ns: int,
    runtime_setup_time_ns_acc: int = 0,
    start_runtime_setup_time_ns: int = 0,
) -> float:
    eval_ns_since_start = max(
        0,
        benchmark_eval_time_ns_acc - start_benchmark_eval_time_ns,
    )
    checkpoint_ns_since_start = max(
        0,
        checkpoint_time_ns_acc - start_checkpoint_time_ns,
    )
    recording_ns_since_start = max(
        0,
        recording_time_ns_acc - start_recording_time_ns,
    )
    runtime_setup_ns_since_start = max(
        0,
        runtime_setup_time_ns_acc - start_runtime_setup_time_ns,
    )
    active_ns = max(
        0,
        now_ns
        - start_ns
        - eval_ns_since_start
        - checkpoint_ns_since_start
        - recording_ns_since_start
        - runtime_setup_ns_since_start,
    )
    return active_ns * 1e-9


def _objective_elapsed_seconds(
    *,
    now_ns: int,
    objective_timer_started: bool,
    objective_start_ns: int,
    benchmark_eval_time_ns_acc: int,
    objective_start_benchmark_eval_time_ns: int,
    checkpoint_time_ns_acc: int,
    objective_start_checkpoint_time_ns: int,
    recording_time_ns_acc: int,
    objective_start_recording_time_ns: int,
    runtime_setup_time_ns_acc: int = 0,
    objective_start_runtime_setup_time_ns: int = 0,
) -> float:
    if not objective_timer_started:
        return 0.0
    return _active_elapsed_seconds(
        now_ns=now_ns,
        start_ns=objective_start_ns,
        benchmark_eval_time_ns_acc=benchmark_eval_time_ns_acc,
        start_benchmark_eval_time_ns=objective_start_benchmark_eval_time_ns,
        checkpoint_time_ns_acc=checkpoint_time_ns_acc,
        start_checkpoint_time_ns=objective_start_checkpoint_time_ns,
        recording_time_ns_acc=recording_time_ns_acc,
        start_recording_time_ns=objective_start_recording_time_ns,
        runtime_setup_time_ns_acc=runtime_setup_time_ns_acc,
        start_runtime_setup_time_ns=objective_start_runtime_setup_time_ns,
    )


@dataclass
class TrainingObjectiveState:
    enabled: bool
    warmup_rollouts: int
    timeout_seconds: float | None
    timer_started: bool = False
    start_ns: int = 0
    start_benchmark_eval_time_ns: int = 0
    start_checkpoint_time_ns: int = 0
    start_recording_time_ns: int = 0
    start_runtime_setup_time_ns: int = 0
    threshold_reached: bool = False
    timed_out: bool = False
    timeout_elapsed_seconds: float | None = None
    time_to_reward_threshold_seconds: float | None = None
    time_to_episode_length_threshold_seconds: float | None = None
    time_to_benchmark_threshold_seconds: float | None = None

    def maybe_start(
        self,
        *,
        rollout_idx: int,
        now_ns: int,
        benchmark_eval_time_ns_acc: int,
        checkpoint_time_ns_acc: int,
        recording_time_ns_acc: int,
        runtime_setup_time_ns_acc: int = 0,
    ) -> bool:
        if (
            not self.enabled
            or self.timer_started
            or int(rollout_idx) < int(self.warmup_rollouts)
        ):
            return False
        self.start_ns = int(now_ns)
        self.start_benchmark_eval_time_ns = int(benchmark_eval_time_ns_acc)
        self.start_checkpoint_time_ns = int(checkpoint_time_ns_acc)
        self.start_recording_time_ns = int(recording_time_ns_acc)
        self.start_runtime_setup_time_ns = int(runtime_setup_time_ns_acc)
        self.timer_started = True
        return True

    def elapsed_seconds(
        self,
        *,
        now_ns: int,
        benchmark_eval_time_ns_acc: int,
        checkpoint_time_ns_acc: int,
        recording_time_ns_acc: int,
        runtime_setup_time_ns_acc: int = 0,
    ) -> float:
        return _objective_elapsed_seconds(
            now_ns=now_ns,
            objective_timer_started=self.timer_started,
            objective_start_ns=self.start_ns,
            benchmark_eval_time_ns_acc=benchmark_eval_time_ns_acc,
            objective_start_benchmark_eval_time_ns=self.start_benchmark_eval_time_ns,
            checkpoint_time_ns_acc=checkpoint_time_ns_acc,
            objective_start_checkpoint_time_ns=self.start_checkpoint_time_ns,
            recording_time_ns_acc=recording_time_ns_acc,
            objective_start_recording_time_ns=self.start_recording_time_ns,
            runtime_setup_time_ns_acc=runtime_setup_time_ns_acc,
            objective_start_runtime_setup_time_ns=self.start_runtime_setup_time_ns,
        )

    def mark_reward_reached(
        self,
        *,
        now_ns: int,
        benchmark_eval_time_ns_acc: int,
        checkpoint_time_ns_acc: int,
        recording_time_ns_acc: int,
        runtime_setup_time_ns_acc: int = 0,
    ) -> float:
        elapsed = self.elapsed_seconds(
            now_ns=now_ns,
            benchmark_eval_time_ns_acc=benchmark_eval_time_ns_acc,
            checkpoint_time_ns_acc=checkpoint_time_ns_acc,
            recording_time_ns_acc=recording_time_ns_acc,
            runtime_setup_time_ns_acc=runtime_setup_time_ns_acc,
        )
        self.threshold_reached = True
        self.time_to_reward_threshold_seconds = elapsed
        return elapsed

    def mark_episode_length_reached(
        self,
        *,
        now_ns: int,
        benchmark_eval_time_ns_acc: int,
        checkpoint_time_ns_acc: int,
        recording_time_ns_acc: int,
        runtime_setup_time_ns_acc: int = 0,
    ) -> float:
        elapsed = self.elapsed_seconds(
            now_ns=now_ns,
            benchmark_eval_time_ns_acc=benchmark_eval_time_ns_acc,
            checkpoint_time_ns_acc=checkpoint_time_ns_acc,
            recording_time_ns_acc=recording_time_ns_acc,
            runtime_setup_time_ns_acc=runtime_setup_time_ns_acc,
        )
        self.threshold_reached = True
        self.time_to_episode_length_threshold_seconds = elapsed
        return elapsed

    def mark_benchmark_reached(
        self,
        *,
        now_ns: int,
        benchmark_eval_time_ns_acc: int,
        checkpoint_time_ns_acc: int,
        recording_time_ns_acc: int,
        runtime_setup_time_ns_acc: int = 0,
    ) -> float:
        elapsed = self.elapsed_seconds(
            now_ns=now_ns,
            benchmark_eval_time_ns_acc=benchmark_eval_time_ns_acc,
            checkpoint_time_ns_acc=checkpoint_time_ns_acc,
            recording_time_ns_acc=recording_time_ns_acc,
            runtime_setup_time_ns_acc=runtime_setup_time_ns_acc,
        )
        self.threshold_reached = True
        self.time_to_benchmark_threshold_seconds = elapsed
        return elapsed

    def maybe_mark_timeout(
        self,
        *,
        now_ns: int,
        benchmark_eval_time_ns_acc: int,
        checkpoint_time_ns_acc: int,
        recording_time_ns_acc: int,
        runtime_setup_time_ns_acc: int = 0,
    ) -> bool:
        if (
            not self.enabled
            or self.threshold_reached
            or self.timeout_seconds is None
            or self.timed_out
        ):
            return False
        elapsed = self.elapsed_seconds(
            now_ns=now_ns,
            benchmark_eval_time_ns_acc=benchmark_eval_time_ns_acc,
            checkpoint_time_ns_acc=checkpoint_time_ns_acc,
            recording_time_ns_acc=recording_time_ns_acc,
            runtime_setup_time_ns_acc=runtime_setup_time_ns_acc,
        )
        if elapsed < float(self.timeout_seconds):
            return False
        self.timed_out = True
        self.timeout_elapsed_seconds = elapsed
        return True
