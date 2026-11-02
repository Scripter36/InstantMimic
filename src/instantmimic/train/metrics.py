from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_TRAIN_FRAME_METRIC = "Trainer / Frame"
_ACTIVE_TRAIN_TIME_METRIC = "Trainer / Active train time (s)"
_ACTIVE_TRAIN_FPS_METRIC = "Trainer / FPS"
_CHECKPOINT_TIME_METRIC = "Trainer / Checkpoint time (s)"

EVAL_METRIC_STATISTICS = frozenset({"mean", "rate", "last", "count", "flag", "step"})

_ROLLOUT_RESULT_METRIC_STATISTICS = {
    "valid_rate": "rate",
    "success_rate": "rate",
    "reward_mean": "mean",
    "episode_length_mean": "mean",
}


@dataclass(frozen=True, slots=True)
class EvalMetricLoggingPolicy:
    statistics: frozenset[str] = frozenset({"mean", "rate"})
    exclude: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unsupported = self.statistics - EVAL_METRIC_STATISTICS
        if unsupported:
            raise ValueError(
                "Unsupported eval logging statistics: " + ", ".join(sorted(unsupported))
            )


def _split_eval_metric_group(
    name: str,
    groups: Sequence[str],
) -> tuple[str | None, str]:
    for group in sorted((str(item) for item in groups), key=len, reverse=True):
        group_prefix = f"{group}_"
        if name.startswith(group_prefix):
            return group, name[len(group_prefix) :]
    return None, name


def _append_eval_result_scalars(
    scalars: dict[str, float],
    result: Mapping[str, float | int | bool],
    *,
    labels: Mapping[str, str],
    statistics: Mapping[str, str],
    groups: Sequence[str],
    policy: EvalMetricLoggingPolicy,
    prefix: str = "Eval / ",
) -> dict[str, dict[str, str]]:
    """Append UI-facing eval metrics according to environment-owned metadata."""

    resolved_statistics = dict(_ROLLOUT_RESULT_METRIC_STATISTICS)
    resolved_statistics.update(statistics)
    grouped_metric_names: set[str] = set()
    for raw_name in result:
        group, base_name = _split_eval_metric_group(str(raw_name), groups)
        if group is not None:
            grouped_metric_names.add(base_name)
    grouped_metrics: dict[str, dict[str, str]] = {}
    for raw_name, value in result.items():
        name = str(raw_name)
        group, base_name = _split_eval_metric_group(name, groups)
        if group is None and base_name in grouped_metric_names:
            continue
        if group is not None and base_name not in grouped_metric_names:
            continue
        statistic = resolved_statistics.get(base_name)
        if statistic not in policy.statistics:
            continue
        if name in policy.exclude or base_name in policy.exclude:
            continue
        if not isinstance(value, bool | int | float):
            continue
        label = labels.get(base_name, _label_from_metric_key(base_name))
        group_label = None if group is None else _label_from_metric_key(group)
        group_prefix = "" if group_label is None else f"{group_label} / "
        scalar_tag = f"{prefix}{group_prefix}{label}"
        scalars[scalar_tag] = float(value)
        if group_label is not None:
            grouped_metrics.setdefault(f"{prefix}{label}", {})[group_label] = scalar_tag
    return grouped_metrics


def _build_rollout_benchmark_scalars(
    *,
    benchmark_pass_streak: int,
    benchmark_current_score: float | None,
    best_benchmark_metric_value: float | None,
    benchmark_threshold_enabled: bool,
    benchmark_eval_elapsed_ns: int,
    benchmark_active_elapsed_ns: int,
    checkpoint_time_ns_acc: int,
    frames: int,
) -> dict[str, float]:
    scalars = {
        _TRAIN_FRAME_METRIC: float(frames),
        _ACTIVE_TRAIN_TIME_METRIC: benchmark_active_elapsed_ns * 1e-9,
        _CHECKPOINT_TIME_METRIC: checkpoint_time_ns_acc * 1e-9,
        "Eval / Best metric value": (
            float(best_benchmark_metric_value)
            if best_benchmark_metric_value is not None
            else math.nan
        ),
        "Timing / Eval runtime (ms)": benchmark_eval_elapsed_ns * 1e-6,
    }
    if benchmark_threshold_enabled:
        scalars["Eval / Pass streak"] = float(benchmark_pass_streak)
        scalars["Eval / Score"] = (
            float(benchmark_current_score)
            if benchmark_current_score is not None
            else math.nan
        )
    return scalars


# Compatibility for callers using the original tracking-only helper name.
_build_tracking_benchmark_scalars = _build_rollout_benchmark_scalars


def _append_env_eval_metric_scalars(
    scalars: dict[str, float],
    result: Mapping[str, float | int | bool],
    labels: Mapping[str, str],
    *,
    prefix: str = "Eval / ",
) -> None:
    for name, label in labels.items():
        value = result.get(name)
        if isinstance(value, (bool, int | float)):
            scalars[f"{prefix}{label}"] = float(value)


def _append_optional_command_diagnostic_scalars(
    step_scalars: dict[str, float],
    frame_scalars: dict[str, float],
    env_raw: Any,
) -> dict[str, float]:
    get_metrics = getattr(env_raw, "get_command_diagnostic_means", None)
    if not callable(get_metrics):
        return {}
    raw_metrics = get_metrics()
    if not isinstance(raw_metrics, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if isinstance(value, (bool, int | float)):
            scalar = float(value)
        else:
            continue
        label = str(key).replace("_", " ")
        label = label[:1].upper() + label[1:]
        label = label.replace(" l2 ", " L2 ")
        if label.endswith(" l2"):
            label = label[:-3] + " L2"
        step_scalars[f"Command / {label}"] = scalar
        frame_scalars[f"Command / {label} (frames)"] = scalar
        result[str(key)] = scalar
    return result


def _label_from_metric_key(key: object) -> str:
    label = str(key).replace("_", " ")
    label = label[:1].upper() + label[1:]
    label = label.replace(" l2 ", " L2 ")
    if label.endswith(" l2"):
        label = label[:-3] + " L2"
    return label


def _append_optional_env_diagnostic_scalars(
    step_scalars: dict[str, float],
    frame_scalars: dict[str, float],
    env_raw: Any,
) -> dict[str, float]:
    get_metrics = getattr(env_raw, "get_training_diagnostic_means", None)
    if not callable(get_metrics):
        return {}
    raw_metrics = get_metrics()
    if not isinstance(raw_metrics, dict):
        return {}
    result: dict[str, float] = {}
    for key, value in raw_metrics.items():
        if isinstance(value, (bool, int | float)):
            scalar = float(value)
        else:
            continue
        label = _label_from_metric_key(key)
        step_scalars[f"Env / {label}"] = scalar
        frame_scalars[f"Env / {label} (frames)"] = scalar
        result[str(key)] = scalar
    return result


def _musclenn_update_diagnostic_scalars(
    last_losses: dict[str, float],
) -> dict[str, float]:
    """MuscleNN optimizer and raw-input diagnostics of the latest update."""
    scalars: dict[str, float] = {}
    if "musclenn_bound_loss" in last_losses:
        scalars["Loss / MuscleNN bound loss"] = last_losses["musclenn_bound_loss"]
    if "musclenn_grad_norm" in last_losses:
        scalars["Info / MuscleNN grad norm"] = last_losses["musclenn_grad_norm"]
    if "musclenn_grad_clip_fraction" in last_losses:
        scalars["Info / MuscleNN grad clip fraction"] = last_losses[
            "musclenn_grad_clip_fraction"
        ]
    musclenn_gpu_time_ns = float(last_losses.get("musclenn_gpu_time_ns", 0.0))
    if musclenn_gpu_time_ns > 0.0:
        # GPU time of the latest MuscleNN update, on its own stream: it runs
        # concurrently with the policy update, whose time is reported apart.
        scalars["Timing / MuscleNN update (ms)"] = musclenn_gpu_time_ns * 1.0e-6
    prefix = "musclenn_input_"
    for key, value in last_losses.items():
        if not key.startswith(prefix):
            continue
        block_and_stat = key[len(prefix) :]
        for suffix, label in (("_mean_abs", "mean abs"), ("_std", "std")):
            if block_and_stat.endswith(suffix):
                block = block_and_stat[: -len(suffix)]
                scalars[f"Info / MuscleNN input {block} {label}"] = value
                break
    return scalars


def build_training_iteration_scalars(
    *,
    last_losses: dict[str, float] | None,
    agent: Any,
    env_raw: Any,
    frames: int,
    update_count: int,
    last_mean_reward: float | None,
    last_mean_episode_timesteps: float | None,
    timing_rollouts: int,
    physics_time_ns_acc: int,
    update_time_ns_acc: int,
    active_train_seconds: float,
    checkpoint_time_seconds: float,
    training_fps: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    step_scalars: dict[str, float] = {}
    frame_scalars: dict[str, float] = {}

    if last_losses is not None:
        step_scalars["Loss / Policy loss"] = last_losses["policy"]
        if "value" in last_losses:
            step_scalars["Loss / Value loss"] = last_losses["value"]
        if "entropy" in last_losses:
            step_scalars["Loss / Entropy loss"] = last_losses["entropy"]
        if "policy_aux" in last_losses:
            step_scalars["Loss / Policy auxiliary loss"] = last_losses["policy_aux"]
        if "log_std_bound" in last_losses:
            step_scalars["Loss / Log std bound loss"] = last_losses["log_std_bound"]
        if "kl" in last_losses:
            step_scalars["Learning / Approx KL"] = last_losses["kl"]
        if "clip_fraction" in last_losses:
            step_scalars["Policy / Clip fraction"] = last_losses["clip_fraction"]
        step_scalars["Learning / Learning rate"] = last_losses["lr"]
        step_scalars["Policy / Standard deviation"] = last_losses["std"]
        if "std_actual_mean" in last_losses:
            step_scalars["Policy / Actual standard deviation mean"] = last_losses[
                "std_actual_mean"
            ]
            step_scalars["Policy / Actual standard deviation min"] = last_losses[
                "std_actual_min"
            ]
            step_scalars["Policy / Actual standard deviation max"] = last_losses[
                "std_actual_max"
            ]
        if "raw_log_std_clamp_lower_fraction" in last_losses:
            step_scalars["Policy / Raw log std clamp lower fraction"] = last_losses[
                "raw_log_std_clamp_lower_fraction"
            ]
            step_scalars["Policy / Raw log std clamp upper fraction"] = last_losses[
                "raw_log_std_clamp_upper_fraction"
            ]
            step_scalars["Policy / Raw log std clamp overshoot RMS"] = last_losses[
                "raw_log_std_clamp_overshoot_rms"
            ]
            step_scalars["Policy / Raw log std clamp overshoot max"] = last_losses[
                "raw_log_std_clamp_overshoot_max"
            ]
        if "action_rms" in last_losses:
            step_scalars["Action / RMS"] = last_losses["action_rms"]
            frame_scalars["Action / RMS (frames)"] = last_losses["action_rms"]
        if "action_abs_max" in last_losses:
            step_scalars["Action / Abs max"] = last_losses["action_abs_max"]
            frame_scalars["Action / Abs max (frames)"] = last_losses["action_abs_max"]
        if "action_abs_gt1_frac" in last_losses:
            step_scalars["Action / Abs > 1 fraction"] = last_losses[
                "action_abs_gt1_frac"
            ]
            frame_scalars["Action / Abs > 1 fraction (frames)"] = last_losses[
                "action_abs_gt1_frac"
            ]
        if "debug_total_grad_norm_pre_clip" in last_losses:
            step_scalars["Debug / Total grad norm pre-clip"] = last_losses[
                "debug_total_grad_norm_pre_clip"
            ]
        if "musclenn_loss" in last_losses:
            step_scalars["Loss / MuscleNN loss"] = last_losses["musclenn_loss"]
            step_scalars["Loss / MuscleNN target loss"] = last_losses[
                "musclenn_target_loss"
            ]
            step_scalars["Loss / MuscleNN regularization loss"] = last_losses[
                "musclenn_reg_loss"
            ]
            if "musclenn_bound_loss" in last_losses:
                step_scalars["Loss / MuscleNN bound loss"] = last_losses[
                    "musclenn_bound_loss"
                ]
            step_scalars["Info / MuscleNN buffer size"] = last_losses[
                "musclenn_buffer_size"
            ]
            step_scalars.update(_musclenn_update_diagnostic_scalars(last_losses))

        # Same metrics with frames as x-axis
        frame_scalars["Loss / Policy loss (frames)"] = last_losses["policy"]
        if "value" in last_losses:
            frame_scalars["Loss / Value loss (frames)"] = last_losses["value"]
        if "entropy" in last_losses:
            frame_scalars["Loss / Entropy loss (frames)"] = last_losses["entropy"]
        if "policy_aux" in last_losses:
            frame_scalars["Loss / Policy auxiliary loss (frames)"] = last_losses[
                "policy_aux"
            ]
        if "log_std_bound" in last_losses:
            frame_scalars["Loss / Log std bound loss (frames)"] = last_losses[
                "log_std_bound"
            ]
        if "kl" in last_losses:
            frame_scalars["Learning / Approx KL (frames)"] = last_losses["kl"]
        if "clip_fraction" in last_losses:
            frame_scalars["Policy / Clip fraction (frames)"] = last_losses[
                "clip_fraction"
            ]
        frame_scalars["Learning / Learning rate (frames)"] = last_losses["lr"]
        if "debug_total_grad_norm_pre_clip" in last_losses:
            frame_scalars["Debug / Total grad norm pre-clip (frames)"] = last_losses[
                "debug_total_grad_norm_pre_clip"
            ]
        if "musclenn_loss" in last_losses:
            frame_scalars["Loss / MuscleNN loss (frames)"] = last_losses[
                "musclenn_loss"
            ]
            frame_scalars["Loss / MuscleNN target loss (frames)"] = last_losses[
                "musclenn_target_loss"
            ]
            frame_scalars["Loss / MuscleNN regularization loss (frames)"] = last_losses[
                "musclenn_reg_loss"
            ]
            if "musclenn_bound_loss" in last_losses:
                frame_scalars["Loss / MuscleNN bound loss (frames)"] = last_losses[
                    "musclenn_bound_loss"
                ]

    if timing_rollouts > 0:
        avg_rollout_ms = (physics_time_ns_acc / timing_rollouts) * 1e-6
        avg_update_ms = (update_time_ns_acc / timing_rollouts) * 1e-6
        step_scalars["Timing / Rollout body (ms)"] = avg_rollout_ms
        step_scalars["Timing / PPO update (ms)"] = avg_update_ms
        frame_scalars["Timing / Rollout body (ms) (frames)"] = avg_rollout_ms
        frame_scalars["Timing / PPO update (ms) (frames)"] = avg_update_ms

    if last_mean_reward is not None and last_mean_episode_timesteps is not None:
        step_scalars["Reward / Total reward (mean)"] = last_mean_reward
        step_scalars["Episode / Total timesteps (mean)"] = last_mean_episode_timesteps
        frame_scalars["Reward / Total reward (mean) (frames)"] = last_mean_reward
        frame_scalars["Episode / Total timesteps (mean) (frames)"] = (
            last_mean_episode_timesteps
        )

    step_scalars["Info / Frames"] = float(frames)
    step_scalars["Info / Updates"] = float(update_count)
    step_scalars[_ACTIVE_TRAIN_TIME_METRIC] = float(active_train_seconds)
    if training_fps is not None:
        step_scalars[_ACTIVE_TRAIN_FPS_METRIC] = float(training_fps)
    step_scalars[_CHECKPOINT_TIME_METRIC] = float(checkpoint_time_seconds)
    _append_optional_command_diagnostic_scalars(
        step_scalars,
        frame_scalars,
        env_raw,
    )
    _append_optional_env_diagnostic_scalars(
        step_scalars,
        frame_scalars,
        env_raw,
    )

    return step_scalars, frame_scalars
