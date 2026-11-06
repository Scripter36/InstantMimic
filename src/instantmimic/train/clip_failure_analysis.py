from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAX_FAILURE_PLOT_CLIPS = 200
MAX_TIMELINE_HEATMAP_CLIPS = 200
MAX_TIMELINE_HEATMAP_CELLS = 200_000
DEFAULT_CLIP_REPLAY_TIMEOUT_SECONDS = 1800.0

TRACKING_METRIC_KEYS = (
    "ape_mean",
    "global_mpjpe_mean",
    "ee_mean",
    "vel_mean",
    "accel_mean",
    "phc_mpjpe_mean",
    "phc_global_mpjpe_mean",
    "phc_vel_mean",
    "phc_accel_mean",
)


@dataclass(frozen=True)
class AnchorSpec:
    clip_idx: int
    start_frame_idx: int


@dataclass(frozen=True)
class FullMotionSpec:
    clip_idx: int
    horizon_steps: int


@dataclass(frozen=True)
class ScanAnchorSpec:
    clip_idx: int
    start_frame_idx: int
    reset_start_frame_idx: int
    lead_in_steps_effective: int


@dataclass
class LoadedAnalysisRuntime:
    cfg: Any
    env: Any
    env_raw: Any
    agent: Any
    device: Any
    batch_size: int
    obs_dim: int
    env_action_dim: int
    policy_action_dim: int
    policy_runner: Any
    policy_actions: Any
    env_actions: Any
    next_obs_policy: Any
    next_rewards: Any
    next_terminated: Any
    next_truncated: Any
    next_dones: Any
    step_io: Any
    checkpoint_path: Path
    run_log_dir: Path

    def synchronize_pending_work(self) -> None:
        import torch
        import warp as wp

        device = torch.device(self.device)
        if device.type != "cuda":
            return
        torch.cuda.synchronize(device)
        wp.synchronize()

    def close(self) -> None:
        self.synchronize_pending_work()
        self.env.close()


def _format_anchor_scan_exception_context(
    *,
    batch_start: int,
    batch_anchors: Sequence[ScanAnchorSpec],
    clip_names: Sequence[str],
    clip_lengths: Sequence[int],
    step_idx: int | None,
    max_items: int = 24,
) -> str:
    batch_size = len(batch_anchors)
    batch_stop = int(batch_start) + int(batch_size)
    unique_clip_indices = sorted({int(anchor.clip_idx) for anchor in batch_anchors})
    unique_clip_sample = [
        {
            "clip_idx": int(clip_idx),
            "clip_name": str(clip_names[int(clip_idx)]),
            "clip_length": int(clip_lengths[int(clip_idx)]),
        }
        for clip_idx in unique_clip_indices[:max_items]
    ]

    if batch_size <= max_items:
        sampled_anchors = list(enumerate(batch_anchors))
    else:
        head_count = max_items // 2
        tail_count = max_items - head_count
        sampled_anchors = list(enumerate(batch_anchors[:head_count]))
        sampled_anchors.extend(
            (batch_size - tail_count + offset, anchor)
            for offset, anchor in enumerate(batch_anchors[-tail_count:])
        )

    anchor_sample = [
        {
            "batch_local_idx": int(batch_idx),
            "global_anchor_idx": int(batch_start) + int(batch_idx),
            "clip_idx": int(anchor.clip_idx),
            "clip_name": str(clip_names[int(anchor.clip_idx)]),
            "clip_length": int(clip_lengths[int(anchor.clip_idx)]),
            "start_frame_idx": int(anchor.start_frame_idx),
            "reset_start_frame_idx": int(anchor.reset_start_frame_idx),
            "lead_in_steps_effective": int(anchor.lead_in_steps_effective),
        }
        for batch_idx, anchor in sampled_anchors
    ]
    payload = {
        "batch_start": int(batch_start),
        "batch_stop": int(batch_stop),
        "batch_size": int(batch_size),
        "step_idx": None if step_idx is None else int(step_idx),
        "unique_clip_count": int(len(unique_clip_indices)),
        "unique_clip_sample": unique_clip_sample,
        "anchor_sample": anchor_sample,
    }
    return json.dumps(payload, ensure_ascii=True)


def prepare_analysis_runtime_cfg(
    *,
    checkpoint_path: str | Path,
    batch_size: int,
    seed: int | None = None,
    use_cuda_graph: bool = False,
    compile: bool = False,
) -> Any:
    from omegaconf import open_dict

    from instantmimic.play.config import load_artifact_playback_runtime

    runtime = load_artifact_playback_runtime(
        checkpoint_path,
        num_envs=batch_size,
        seed=seed,
        use_cuda_graph=use_cuda_graph,
        runtime_overrides={
            "act_cuda_graph": False,
            "update_cuda_graph": False,
            "compile": bool(compile),
        },
    )
    runtime_cfg = runtime.execution_config

    with open_dict(runtime_cfg):
        runtime_cfg.env.scene.num_envs = int(batch_size)
        runtime_cfg.env.reset.random_start = False
        runtime_cfg.env.reset.deterministic = True
        runtime_cfg.env.reset.motion_noise_std = 0.0
        runtime_cfg.env.adaptive_sampling.enabled = False
        runtime_cfg.env.use_cuda_graph = bool(use_cuda_graph)

    return runtime


def build_anchor_table(
    *,
    clip_lengths: list[int],
    anchor_stride: int,
    window_steps: int | None = None,
) -> list[AnchorSpec]:
    stride = max(1, int(anchor_stride))
    anchors: list[AnchorSpec] = []
    for clip_idx, raw_length in enumerate(clip_lengths):
        clip_length = max(1, int(raw_length))
        reset_start_count = _valid_reset_start_count(
            clip_length=clip_length,
            horizon_steps=window_steps,
        )
        for start_frame_idx in range(0, reset_start_count, stride):
            anchors.append(
                AnchorSpec(
                    clip_idx=int(clip_idx),
                    start_frame_idx=int(start_frame_idx),
                )
            )
    return anchors


def prepare_scan_anchors(
    anchors: Sequence[AnchorSpec],
    *,
    lead_in_steps: int,
) -> list[ScanAnchorSpec]:
    requested_lead_in = max(0, int(lead_in_steps))
    planned: list[ScanAnchorSpec] = []
    for anchor in anchors:
        start_frame_idx = int(anchor.start_frame_idx)
        reset_start_frame_idx = max(0, start_frame_idx - requested_lead_in)
        planned.append(
            ScanAnchorSpec(
                clip_idx=int(anchor.clip_idx),
                start_frame_idx=start_frame_idx,
                reset_start_frame_idx=reset_start_frame_idx,
                lead_in_steps_effective=start_frame_idx - reset_start_frame_idx,
            )
        )
    return planned


def finite_anchor_window_steps(
    *,
    clip_length: int,
    start_frame_idx: int,
    requested_window_steps: int,
) -> int:
    """Return measurable transition count before a finite clip reaches its end."""
    remaining_transitions = int(clip_length) - 1 - int(start_frame_idx)
    return max(1, min(int(requested_window_steps), remaining_transitions))


def summarize_anchor_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        clip_idx = int(row["clip_idx"])
        clip_name = str(row["clip_name"])
        grouped[(clip_idx, clip_name)].append(row)

    summary_rows: list[dict[str, object]] = []
    for (clip_idx, clip_name), clip_rows in sorted(grouped.items()):
        anchor_count = len(clip_rows)
        failure_count = sum(int(item["failed"]) for item in clip_rows)
        pre_anchor_failure_count = sum(
            int(item.get("pre_anchor_failure", 0)) for item in clip_rows
        )
        survival_steps = [float(item["survival_steps"]) for item in clip_rows]
        survival_ratios = [
            float(item["survival_steps"])
            / max(1.0, float(item.get("effective_window_steps", item["window_steps"])))
            for item in clip_rows
        ]
        episode_reward_sums = [
            float(item.get("episode_reward_sum", 0.0)) for item in clip_rows
        ]
        reward_sum_total = sum(episode_reward_sums)
        measured_reward_steps = sum(float(item["survival_steps"]) for item in clip_rows)
        effective_window_steps_total = sum(
            float(item.get("effective_window_steps", item["window_steps"]))
            for item in clip_rows
        )
        row = {
            "clip_idx": clip_idx,
            "clip_name": clip_name,
            "anchor_count": anchor_count,
            "failure_count": failure_count,
            "pre_anchor_failure_count": pre_anchor_failure_count,
            "failure_rate": float(failure_count) / float(anchor_count),
            "pre_anchor_failure_rate": float(pre_anchor_failure_count)
            / float(anchor_count),
            "mean_survival_steps": sum(survival_steps) / float(anchor_count),
            "mean_survival_ratio": sum(survival_ratios) / float(anchor_count),
            "mean_episode_reward_sum": reward_sum_total / float(anchor_count),
            "mean_reward_per_step": reward_sum_total / max(1.0, measured_reward_steps),
            "fixed_horizon_reward_mean": reward_sum_total
            / max(1.0, effective_window_steps_total),
            "reward_sum_total": reward_sum_total,
            "effective_window_steps_total": effective_window_steps_total,
        }
        if any(any(key in item for key in TRACKING_METRIC_KEYS) for item in clip_rows):
            tracking_success_rows = [
                item
                for item in clip_rows
                if int(item.get("completed", 0)) == 1
                and int(item.get("failed", 0)) == 0
            ]
            row["tracking_success_count"] = len(tracking_success_rows)
            for key in TRACKING_METRIC_KEYS:
                values = [
                    float(item[key])
                    for item in tracking_success_rows
                    if key in item and np.isfinite(float(item[key]))
                ]
                row[key] = float(np.mean(values)) if values else float("nan")
        summary_rows.append(row)
    return summary_rows


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(path: Path, summary_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    include_tracking = any(
        any(key in row for key in TRACKING_METRIC_KEYS) for row in summary_rows
    )
    lines = [
        "# LAFAN Clip Failure Analysis",
        "",
        (
            "| clip | anchors | failures | pre_anchor_failures | failure_rate | "
            "pre_anchor_failure_rate | mean_survival_steps | mean_survival_ratio | "
            "mean_reward | mean_episode_reward | tracking_successes | APE | "
            "global_MPJPE | EE | velocity | acceleration | PHC_MPJPE | "
            "PHC_global_MPJPE | PHC_velocity | PHC_acceleration |"
            if include_tracking
            else "| clip | anchors | failures | pre_anchor_failures | failure_rate | pre_anchor_failure_rate | mean_survival_steps | mean_survival_ratio | mean_reward | mean_episode_reward |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            if include_tracking
            else "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for row in summary_rows:
        base = (
            "| {clip_name} | {anchor_count} | {failure_count} | {pre_anchor_failure_count} | "
            "{failure_rate:.4f} | {pre_anchor_failure_rate:.4f} | "
            "{mean_survival_steps:.2f} | {mean_survival_ratio:.4f} | "
            "{mean_reward_per_step:.4f} | {mean_episode_reward_sum:.2f} |"
        ).format(**row)
        if include_tracking:
            base = (
                base[:-1] + " {tracking_success_count} | {ape_mean:.6f} | "
                "{global_mpjpe_mean:.6f} | {ee_mean:.6f} | {vel_mean:.6f} | "
                "{accel_mean:.6f} | {phc_mpjpe_mean:.6f} | "
                "{phc_global_mpjpe_mean:.6f} | {phc_vel_mean:.6f} | "
                "{phc_accel_mean:.6f} |"
            ).format(
                tracking_success_count=int(row.get("tracking_success_count", 0)),
                ape_mean=float(row.get("ape_mean", float("nan"))),
                global_mpjpe_mean=float(row.get("global_mpjpe_mean", float("nan"))),
                ee_mean=float(row.get("ee_mean", float("nan"))),
                vel_mean=float(row.get("vel_mean", float("nan"))),
                accel_mean=float(row.get("accel_mean", float("nan"))),
                phc_mpjpe_mean=float(row.get("phc_mpjpe_mean", float("nan"))),
                phc_global_mpjpe_mean=float(
                    row.get("phc_global_mpjpe_mean", float("nan"))
                ),
                phc_vel_mean=float(row.get("phc_vel_mean", float("nan"))),
                phc_accel_mean=float(row.get("phc_accel_mean", float("nan"))),
            )
        lines.append(base)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_plots(
    *,
    output_dir: Path,
    summary_rows: list[dict[str, object]],
    anchor_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_rows:
        all_summary_sorted = sorted(
            summary_rows,
            key=lambda row: float(row["failure_rate"]),
            reverse=True,
        )
        summary_sorted = all_summary_sorted[:MAX_FAILURE_PLOT_CLIPS]
        labels = [str(row["clip_name"]) for row in summary_sorted]
        failure_rates = [float(row["failure_rate"]) for row in summary_sorted]
        anchor_counts = [int(row["anchor_count"]) for row in summary_sorted]

        fig, ax = plt.subplots(figsize=(10.5, max(3.0, 0.55 * len(labels))))
        bars = ax.barh(labels, failure_rates, color="#D66A4E")
        ax.invert_yaxis()
        ax.set_xlabel("failure rate")
        title = "Clip Failure Rate"
        if len(all_summary_sorted) > len(summary_sorted):
            title += f" (top {len(summary_sorted)} of {len(all_summary_sorted)})"
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, axis="x", alpha=0.2)
        for bar, count in zip(bars, anchor_counts, strict=False):
            ax.text(
                min(0.98, float(bar.get_width()) + 0.02),
                bar.get_y() + bar.get_height() / 2.0,
                f"n={count}",
                va="center",
                ha="left",
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(output_dir / "failure_rate.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if anchor_rows:
        clip_names = list(dict.fromkeys(str(row["clip_name"]) for row in anchor_rows))
        original_clip_count = len(clip_names)
        if len(clip_names) > MAX_TIMELINE_HEATMAP_CLIPS:
            ranked_clip_names = [
                str(row["clip_name"])
                for row in sorted(
                    summary_rows,
                    key=lambda row: float(row["failure_rate"]),
                    reverse=True,
                )
            ]
            if ranked_clip_names:
                clip_names = ranked_clip_names[:MAX_TIMELINE_HEATMAP_CLIPS]
            else:
                clip_names = clip_names[:MAX_TIMELINE_HEATMAP_CLIPS]
        selected_clip_names = set(clip_names)
        selected_anchor_rows = [
            row for row in anchor_rows if str(row["clip_name"]) in selected_clip_names
        ]
        clip_to_row = {name: idx for idx, name in enumerate(clip_names)}
        unique_frames = sorted(
            {int(row["start_frame_idx"]) for row in selected_anchor_rows}
        )
        if not unique_frames:
            return
        if len(clip_names) * len(unique_frames) > MAX_TIMELINE_HEATMAP_CELLS:
            max_clips = max(1, MAX_TIMELINE_HEATMAP_CELLS // len(unique_frames))
            clip_names = clip_names[:max_clips]
            selected_clip_names = set(clip_names)
            selected_anchor_rows = [
                row
                for row in selected_anchor_rows
                if str(row["clip_name"]) in selected_clip_names
            ]
            clip_to_row = {name: idx for idx, name in enumerate(clip_names)}
        frame_to_col = {frame: idx for idx, frame in enumerate(unique_frames)}
        heatmap = np.full(
            (len(clip_names), len(unique_frames)), np.nan, dtype=np.float32
        )
        for row in selected_anchor_rows:
            clip_name = str(row["clip_name"])
            start_frame_idx = int(row["start_frame_idx"])
            heatmap[clip_to_row[clip_name], frame_to_col[start_frame_idx]] = float(
                row["survival_ratio"]
            )

        fig, ax = plt.subplots(
            figsize=(
                min(18.0, max(6.0, len(unique_frames) * 0.04)),
                max(3.0, len(clip_names) * 0.55),
            )
        )
        image = ax.imshow(
            heatmap,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        title = "Clip Timeline Survival Ratio"
        if original_clip_count > len(clip_names):
            title += f" (top {len(clip_names)} of {original_clip_count})"
        ax.set_title(title)
        ax.set_xlabel("anchor start frame")
        ax.set_yticks(np.arange(len(clip_names)))
        ax.set_yticklabels(clip_names)
        if len(unique_frames) <= 24:
            ax.set_xticks(np.arange(len(unique_frames)))
            ax.set_xticklabels(
                [str(frame) for frame in unique_frames], rotation=45, ha="right"
            )
        else:
            xtick_positions = np.linspace(
                0, len(unique_frames) - 1, num=min(10, len(unique_frames))
            )
            xtick_positions = np.unique(xtick_positions.astype(int))
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(
                [str(unique_frames[idx]) for idx in xtick_positions],
                rotation=45,
                ha="right",
            )
        fig.colorbar(image, ax=ax, label="survival ratio")
        fig.tight_layout()
        fig.savefig(output_dir / "timeline_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def _clip_name_from_motion_path(path_like: str) -> str:
    return Path(str(path_like)).name


def _clip_info_from_runtime_env(env_raw: Any) -> tuple[list[int], list[str]]:
    if bool(getattr(env_raw, "_multi_motion_enabled", False)):
        clip_lengths = [
            int(item) for item in env_raw.motion_clip_length.detach().cpu().tolist()
        ]
        clip_names = [
            _clip_name_from_motion_path(path)
            for path in env_raw._resolved_motion_sources
        ]
        return clip_lengths, clip_names

    motion_frame_count = getattr(env_raw, "motion_frame_count", None)
    if motion_frame_count is None:
        motion_clip_length = getattr(env_raw, "motion_clip_length", None)
        if motion_clip_length is not None:
            clip_lengths = [
                int(item) for item in motion_clip_length.detach().cpu().tolist()
            ]
            if clip_lengths:
                motion_frame_count = clip_lengths[0]
    if motion_frame_count is None:
        raise RuntimeError(
            "LAFAN clip failure analysis requires a loaded motion.source."
        )

    resolved_sources = tuple(getattr(env_raw, "_resolved_motion_sources", ()))
    motion_file = resolved_sources[0] if resolved_sources else ""
    clip_name = _clip_name_from_motion_path(motion_file) if motion_file else "motion_0"
    return [int(motion_frame_count)], [clip_name]


def _normalize_analysis_mode(raw_value: str) -> str:
    mode = str(raw_value).strip().lower()
    if mode in {"", "anchor"}:
        return "anchor"
    if mode == "full_motion":
        return "full_motion"
    raise ValueError(f"analysis mode must be anchor or full_motion; got {raw_value!r}.")


def build_anchor_table_for_clip_indices(
    *,
    clip_lengths: Sequence[int],
    clip_indices: Sequence[int],
    anchor_stride: int,
    window_steps: int | None = None,
) -> list[AnchorSpec]:
    stride = max(1, int(anchor_stride))
    anchors: list[AnchorSpec] = []
    for clip_idx in clip_indices:
        clip_length = max(1, int(clip_lengths[int(clip_idx)]))
        reset_start_count = _valid_reset_start_count(
            clip_length=clip_length,
            horizon_steps=window_steps,
        )
        for start_frame_idx in range(0, reset_start_count, stride):
            anchors.append(
                AnchorSpec(
                    clip_idx=int(clip_idx),
                    start_frame_idx=int(start_frame_idx),
                )
            )
    return anchors


def build_full_motion_table_for_clip_indices(
    *,
    clip_lengths: Sequence[int],
    clip_indices: Sequence[int],
) -> list[FullMotionSpec]:
    motions: list[FullMotionSpec] = []
    for clip_idx in clip_indices:
        clip_length = max(1, int(clip_lengths[int(clip_idx)]))
        horizon_steps = max(1, clip_length - 1)
        motions.append(
            FullMotionSpec(
                clip_idx=int(clip_idx),
                horizon_steps=int(horizon_steps),
            )
        )
    return sorted(
        motions, key=lambda item: (int(item.horizon_steps), int(item.clip_idx))
    )


def _valid_reset_start_count(
    *,
    clip_length: int,
    horizon_steps: int | None,
) -> int:
    clip_length = max(1, int(clip_length))
    if horizon_steps is None:
        return clip_length
    horizon = max(1, int(horizon_steps))
    return min(clip_length, max(1, clip_length - horizon + 1))


def _resolve_output_root(
    *,
    checkpoint_path: Path,
    config_name: str,
    output_root: str | Path | None,
) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    run_dir = checkpoint_path.parent.parent
    return (
        Path("analysis") / "lafan_clip_failure" / str(config_name) / run_dir.name
    ).resolve()


def _parse_playback_result(play_log: Path) -> dict[str, str]:
    from instantmimic.play.results import parse_playback_result_payload

    return (
        parse_playback_result_payload(
            play_log.read_text(encoding="utf-8", errors="replace")
        )
        or {}
    )


def _build_agent(
    *,
    semantic_cfg: Any,
    environment_cfg: Any,
    env: Any,
    checkpoint_path: str | Path,
    device: Any,
) -> tuple[Any, Any, int, int]:
    from instantmimic.play.setup import build_playback_agent_runtime

    setup = build_playback_agent_runtime(
        cfg=semantic_cfg,
        env=env,
        checkpoint_path=checkpoint_path,
        environment_cfg=environment_cfg,
        device=device,
    )
    return (
        setup.agent,
        setup.runtime_spec,
        setup.env_action_dim,
        setup.agent_action_dim,
    )


def _make_analysis_env(
    runtime_cfg: Any,
    *,
    shared_motion_source_env: Any | None,
    shared_motion_clone_tensors: bool,
) -> Any:
    import gymnasium as gym

    from instantmimic.bootstrap import prepare_environment_runtime

    prepare_environment_runtime()
    shared_motion_kwargs = {}
    if shared_motion_source_env is not None:
        shared_motion_kwargs = {
            "shared_motion_reference": shared_motion_source_env,
            "clone_shared_motion_reference": bool(shared_motion_clone_tensors),
        }
    return gym.make(
        runtime_cfg.task,
        cfg=runtime_cfg.env,
        render_mode=None,
        **shared_motion_kwargs,
    )


def load_analysis_runtime(
    *,
    checkpoint_path: str | Path,
    batch_size: int,
    seed: int | None = None,
    use_cuda_graph: bool = False,
    compile: bool = False,
    shared_motion_source_env: Any | None = None,
    shared_motion_clone_tensors: bool = False,
) -> LoadedAnalysisRuntime:
    import torch

    from instantmimic.envs.direct_rl_env import StepInputs, StepIO, StepOutputs
    from instantmimic.rl.runtime import (
        set_seed,
    )
    from instantmimic.train.benchmark_runner import _BenchmarkPolicyRunner

    playback_runtime = prepare_analysis_runtime_cfg(
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        seed=seed,
        use_cuda_graph=use_cuda_graph,
        compile=compile,
    )
    runtime_cfg = playback_runtime.execution_config
    checkpoint_path_resolved = playback_runtime.checkpoint_path
    run_log_dir = playback_runtime.log_dir

    set_seed(int(runtime_cfg.seed))
    env = _make_analysis_env(
        runtime_cfg,
        shared_motion_source_env=shared_motion_source_env,
        shared_motion_clone_tensors=shared_motion_clone_tensors,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, runtime_spec, env_action_dim, agent_action_dim = _build_agent(
        semantic_cfg=playback_runtime.semantic_config,
        environment_cfg=runtime_cfg.env,
        env=env,
        checkpoint_path=checkpoint_path_resolved,
        device=device,
    )
    env_raw = runtime_spec.env_raw

    policy_runner = _BenchmarkPolicyRunner(
        agent=agent,
        device=device,
        obs_dim=int(runtime_cfg.env.observation_space),
        batch_size=int(batch_size),
        compile_enabled=bool(
            playback_runtime.semantic_config.runtime.get("compile", False)
        ),
        compile_mode=str(
            playback_runtime.semantic_config.runtime.get(
                "compile_mode", "max-autotune-no-cudagraphs"
            )
        ),
        use_cuda_graph=bool(getattr(env_raw.cfg, "use_cuda_graph", False)),
    )
    policy_actions = torch.empty(int(batch_size), agent_action_dim, device=device)
    env_actions = policy_actions
    next_obs_policy = torch.empty(
        int(batch_size),
        int(runtime_cfg.env.observation_space),
        device=device,
    )
    next_rewards = torch.empty(int(batch_size), device=device)
    next_terminated = torch.empty(int(batch_size), device=device, dtype=torch.bool)
    next_truncated = torch.empty_like(next_terminated)
    next_dones = torch.empty(int(batch_size), device=device)
    step_io = StepIO(
        inputs=StepInputs(actions=env_actions),
        outputs=StepOutputs(
            observations={"policy": next_obs_policy},
            rewards=next_rewards,
            terminated=next_terminated,
            truncated=next_truncated,
        ),
    )
    return LoadedAnalysisRuntime(
        cfg=runtime_cfg,
        env=env,
        env_raw=env_raw,
        agent=agent,
        device=device,
        batch_size=int(batch_size),
        obs_dim=int(runtime_cfg.env.observation_space),
        env_action_dim=env_action_dim,
        policy_action_dim=agent_action_dim,
        policy_runner=policy_runner,
        policy_actions=policy_actions,
        env_actions=env_actions,
        next_obs_policy=next_obs_policy,
        next_rewards=next_rewards,
        next_terminated=next_terminated,
        next_truncated=next_truncated,
        next_dones=next_dones,
        step_io=step_io,
        checkpoint_path=checkpoint_path_resolved,
        run_log_dir=run_log_dir,
    )


def reload_analysis_runtime_checkpoint(
    runtime: LoadedAnalysisRuntime,
    checkpoint_path: str | Path,
) -> LoadedAnalysisRuntime:
    from instantmimic.play.setup import load_playback_checkpoint

    checkpoint_path_resolved = Path(checkpoint_path).expanduser().resolve()
    load_playback_checkpoint(
        agent=runtime.agent,
        env=runtime.env_raw,
        checkpoint_path=checkpoint_path_resolved,
    )
    runtime.checkpoint_path = checkpoint_path_resolved
    runtime.run_log_dir = checkpoint_path_resolved.parent.parent.resolve()
    return runtime


def prewarm_analysis_runtime_graphs(runtime: LoadedAnalysisRuntime) -> None:
    import torch

    step_with_io = getattr(runtime.env_raw, "step_with_io", None)
    use_fast_step_with_io = callable(step_with_io)

    try:
        with torch.no_grad():
            obs, _ = runtime.env.reset()
            obs_policy = obs["policy"]
            for _ in range(2):
                runtime.policy_runner.policy_mean_into(
                    obs_policy,
                    runtime.policy_actions,
                )
                if use_fast_step_with_io:
                    step_with_io(runtime.step_io)
                    torch.logical_or(
                        runtime.next_terminated,
                        runtime.next_truncated,
                        out=runtime.next_dones,
                    )
                    obs_policy = runtime.next_obs_policy
                else:
                    obs, reward_tensor, _terminated, _truncated, _ = runtime.env.step(
                        runtime.env_actions
                    )
                    obs_policy = obs["policy"]
                    runtime.next_rewards.copy_(reward_tensor)
    finally:
        runtime.env_raw.clear_analysis_reset_overrides()
        runtime.synchronize_pending_work()


def _maybe_extend_episode_length(env_raw: Any, *, window_steps: int) -> None:
    max_episode_length = getattr(env_raw, "max_episode_length", None)
    if max_episode_length is None or int(max_episode_length) >= int(window_steps):
        return
    target_episode_steps = int(window_steps) + 2
    target_episode_length_s = target_episode_steps * float(env_raw.step_dt)
    if hasattr(env_raw, "cfg") and hasattr(env_raw.cfg, "episode_length_s"):
        previous_episode_length_s = float(env_raw.cfg.episode_length_s)
        env_raw.cfg.episode_length_s = max(
            previous_episode_length_s,
            target_episode_length_s,
        )
        if float(env_raw.cfg.episode_length_s) > previous_episode_length_s:
            # The env step CUDA graph bakes scalar done inputs such as
            # max_episode_length at capture time.  Full-motion benchmark prewarm can
            # capture the graph before we know the longest horizon, so force a
            # recapture after extending the benchmark-only episode cap.
            if hasattr(env_raw, "_step_graph"):
                env_raw._step_graph = None
            graph_cache = getattr(env_raw, "_step_graph_cache", None)
            if graph_cache is not None:
                graph_cache.clear()


def _compute_phc_position_velocity_error_mm(
    sim_body_pos: Any,
    ref_body_pos: Any,
    prev_sim_body_pos: Any,
    prev_ref_body_pos: Any,
) -> Any:
    sim_delta = sim_body_pos - prev_sim_body_pos
    ref_delta = ref_body_pos - prev_ref_body_pos
    return (sim_delta - ref_delta).norm(dim=-1).mean(dim=-1).mul(1000.0)


def _compute_phc_global_mpjpe_mm(
    sim_body_pos: Any,
    ref_body_pos: Any,
) -> Any:
    return (sim_body_pos - ref_body_pos).norm(dim=-1).mean(dim=-1).mul(1000.0)


def _compute_phc_root_relative_mpjpe_mm(
    sim_body_pos: Any,
    ref_body_pos: Any,
    sim_root_pos: Any,
    ref_root_pos: Any,
) -> Any:
    sim_local = sim_body_pos - sim_root_pos.unsqueeze(1)
    ref_local = ref_body_pos - ref_root_pos.unsqueeze(1)
    return (sim_local - ref_local).norm(dim=-1).mean(dim=-1).mul(1000.0)


def _compute_phc_position_acceleration_error_mm(
    sim_body_pos: Any,
    ref_body_pos: Any,
    prev_sim_body_pos: Any,
    prev_ref_body_pos: Any,
    prev2_sim_body_pos: Any,
    prev2_ref_body_pos: Any,
) -> Any:
    sim_second_delta = sim_body_pos - (2.0 * prev_sim_body_pos) + prev2_sim_body_pos
    ref_second_delta = ref_body_pos - (2.0 * prev_ref_body_pos) + prev2_ref_body_pos
    return (sim_second_delta - ref_second_delta).norm(dim=-1).mean(dim=-1).mul(1000.0)


def run_anchor_scan(
    runtime: LoadedAnalysisRuntime,
    *,
    anchors: Sequence[ScanAnchorSpec],
    clip_names: Sequence[str],
    clip_lengths: Sequence[int],
    window_steps: int,
) -> list[dict[str, object]]:
    import torch

    from instantmimic.envs.tracking import metrics_warp as tracking_metrics_warp

    step_with_io = getattr(runtime.env_raw, "step_with_io", None)
    use_fast_step_with_io = callable(step_with_io)
    get_tracking_metrics = getattr(
        runtime.env_raw, "get_reference_tracking_metrics", None
    )
    tracking_metrics_available = callable(get_tracking_metrics)
    runtime_cfg = getattr(runtime, "cfg", {})
    metric_frame = str(runtime_cfg.get("tracking_metric_frame", "root"))
    include_root_in_ape = bool(
        runtime_cfg.get("tracking_metric_include_root_in_ape", False)
    )
    rows: list[dict[str, object]] = []
    max_lead_in_steps = max(
        (int(anchor.lead_in_steps_effective) for anchor in anchors),
        default=0,
    )
    _maybe_extend_episode_length(
        runtime.env_raw,
        window_steps=int(window_steps) + max_lead_in_steps,
    )

    scan_error: BaseException | None = None
    current_batch_start: int | None = None
    current_batch_anchors: Sequence[ScanAnchorSpec] = ()
    current_step_idx: int | None = None
    try:
        with torch.no_grad():
            for batch_start in range(0, len(anchors), runtime.batch_size):
                batch_anchors = anchors[batch_start : batch_start + runtime.batch_size]
                current_batch_start = int(batch_start)
                current_batch_anchors = batch_anchors
                current_step_idx = None
                clip_indices = torch.zeros(
                    runtime.batch_size, device=runtime.device, dtype=torch.int32
                )
                start_frame_indices = torch.zeros(
                    runtime.batch_size, device=runtime.device, dtype=torch.int32
                )
                lead_in_steps_effective = torch.zeros(
                    runtime.batch_size, device=runtime.device, dtype=torch.int32
                )
                effective_window_steps = torch.ones(
                    runtime.batch_size, device=runtime.device, dtype=torch.int32
                )
                enabled = torch.zeros(
                    runtime.batch_size, device=runtime.device, dtype=torch.bool
                )
                for batch_idx, anchor in enumerate(batch_anchors):
                    clip_indices[batch_idx] = int(anchor.clip_idx)
                    start_frame_indices[batch_idx] = int(anchor.reset_start_frame_idx)
                    lead_in_steps_effective[batch_idx] = int(
                        anchor.lead_in_steps_effective
                    )
                    effective_window_steps[batch_idx] = finite_anchor_window_steps(
                        clip_length=int(clip_lengths[int(anchor.clip_idx)]),
                        start_frame_idx=int(anchor.start_frame_idx),
                        requested_window_steps=int(window_steps),
                    )
                    enabled[batch_idx] = True

                runtime.env_raw.set_analysis_reset_overrides(
                    clip_indices=clip_indices,
                    start_frame_indices=start_frame_indices,
                    enabled=enabled,
                )
                obs, _ = runtime.env.reset()
                obs_policy = obs["policy"]
                active_mask = enabled.clone()
                terminated_mask = torch.zeros_like(enabled)
                truncated_mask = torch.zeros_like(enabled)
                completed_mask = torch.zeros_like(enabled)
                pre_anchor_failure_mask = torch.zeros_like(enabled)
                survival_steps = torch.zeros(
                    runtime.batch_size, device=runtime.device, dtype=torch.int32
                )
                reward_sum = torch.zeros(
                    runtime.batch_size, device=runtime.device, dtype=torch.float32
                )
                terminal_reward = torch.zeros(
                    runtime.batch_size, device=runtime.device, dtype=torch.float32
                )
                tracking_metric_steps = torch.zeros(
                    runtime.batch_size, device=runtime.device, dtype=torch.int32
                )
                accel_metric_steps = torch.zeros_like(tracking_metric_steps)
                phc_mpjpe_metric_steps = torch.zeros_like(tracking_metric_steps)
                phc_global_mpjpe_metric_steps = torch.zeros_like(tracking_metric_steps)
                phc_vel_metric_steps = torch.zeros_like(tracking_metric_steps)
                phc_accel_metric_steps = torch.zeros_like(tracking_metric_steps)
                ape_sum = torch.zeros(
                    runtime.batch_size, device=runtime.device, dtype=torch.float32
                )
                global_mpjpe_sum = torch.zeros_like(ape_sum)
                ee_sum = torch.zeros_like(ape_sum)
                vel_sum = torch.zeros_like(ape_sum)
                accel_sum = torch.zeros_like(ape_sum)
                phc_mpjpe_sum = torch.zeros_like(ape_sum)
                phc_global_mpjpe_sum = torch.zeros_like(ape_sum)
                phc_vel_sum = torch.zeros_like(ape_sum)
                phc_accel_sum = torch.zeros_like(ape_sum)
                prev_seen = torch.zeros_like(enabled)
                phc_prev_seen = torch.zeros_like(enabled)
                phc_prev2_seen = torch.zeros_like(enabled)
                prev_sim_body_linvel = None
                prev_ref_body_linvel = None
                prev_sim_root_quat = None
                prev_ref_root_quat = None
                prev_sim_body_pos = None
                prev_ref_body_pos = None
                prev2_sim_body_pos = None
                prev2_ref_body_pos = None
                vel_metric = None
                accel_metric = None
                metric_body_count = -1
                phc_metric_body_count = -1
                tracking_metrics_seen = False
                temporal_metrics_seen = False
                accel_metrics_seen = False
                phc_position_metrics_seen = False
                phc_mpjpe_metrics_seen = False
                phc_global_mpjpe_metrics_seen = False
                phc_vel_metrics_seen = False
                phc_accel_metrics_seen = False
                total_scan_steps = int(window_steps) + int(
                    lead_in_steps_effective.max().item()
                )

                for step_idx in range(total_scan_steps):
                    current_step_idx = int(step_idx)
                    if not bool(active_mask.any().item()):
                        break

                    measurement_mask = active_mask & (
                        lead_in_steps_effective <= int(step_idx)
                    )
                    if tracking_metrics_available:
                        assert callable(get_tracking_metrics)
                        metrics = get_tracking_metrics(
                            frame=metric_frame,
                            include_root_in_ape=include_root_in_ape,
                        )
                        if not isinstance(metrics, dict):
                            raise RuntimeError(
                                "Clip failure benchmark tracking metrics must be a dict."
                            )
                        ape_tensor = metrics.get("ape")
                        global_mpjpe_tensor = metrics.get("global_mpjpe")
                        ee_tensor = metrics.get("ee_error")
                        if (
                            not isinstance(ape_tensor, torch.Tensor)
                            or not isinstance(global_mpjpe_tensor, torch.Tensor)
                            or not isinstance(ee_tensor, torch.Tensor)
                        ):
                            raise RuntimeError(
                                "Clip failure benchmark tracking metrics require "
                                "ape, global_mpjpe, and ee_error tensors."
                            )
                        tracking_metrics_seen = True
                        measurement_float = measurement_mask.to(dtype=ape_tensor.dtype)
                        ape_sum.add_(ape_tensor.reshape(-1) * measurement_float)
                        global_mpjpe_sum.add_(
                            global_mpjpe_tensor.reshape(-1)
                            * measurement_mask.to(dtype=global_mpjpe_tensor.dtype)
                        )
                        ee_sum.add_(
                            ee_tensor.reshape(-1)
                            * measurement_mask.to(dtype=ee_tensor.dtype)
                        )
                        tracking_metric_steps.add_(
                            measurement_mask.to(dtype=tracking_metric_steps.dtype)
                        )

                        sim_body_linvel = metrics.get("tracking_body_linvel")
                        ref_body_linvel = metrics.get("reference_body_linvel")
                        sim_root_quat = metrics.get("sim_root_quat")
                        ref_root_quat = metrics.get("ref_root_quat")
                        if (
                            isinstance(sim_body_linvel, torch.Tensor)
                            and isinstance(ref_body_linvel, torch.Tensor)
                            and isinstance(sim_root_quat, torch.Tensor)
                            and isinstance(ref_root_quat, torch.Tensor)
                        ):
                            temporal_metrics_seen = True
                            body_count = int(sim_body_linvel.shape[1])
                            if vel_metric is None or metric_body_count != body_count:
                                metric_body_count = body_count
                                vel_metric = (
                                    tracking_metrics_warp.create_metric_buffers(
                                        env_count=runtime.batch_size,
                                        body_count=body_count,
                                        device=sim_body_linvel.device,
                                        dtype=sim_body_linvel.dtype,
                                    )
                                )
                                accel_metric = (
                                    tracking_metrics_warp.create_metric_buffers(
                                        env_count=runtime.batch_size,
                                        body_count=body_count,
                                        device=sim_body_linvel.device,
                                        dtype=sim_body_linvel.dtype,
                                    )
                                )
                                prev_sim_body_linvel = None
                                prev_ref_body_linvel = None
                                prev_sim_root_quat = None
                                prev_ref_root_quat = None
                                prev_seen.zero_()
                            assert vel_metric is not None
                            tracking_metrics_warp.compute_mean_velocity_error_into_warp(
                                sim_body_linvel,
                                ref_body_linvel,
                                sim_root_quat=sim_root_quat,
                                ref_root_quat=ref_root_quat,
                                frame=metric_frame,
                                out=vel_metric,
                            )
                            vel_sum.add_(
                                vel_metric.mean_error
                                * measurement_mask.to(dtype=vel_metric.mean_error.dtype)
                            )
                            if (
                                prev_sim_body_linvel is not None
                                and prev_ref_body_linvel is not None
                                and prev_sim_root_quat is not None
                                and prev_ref_root_quat is not None
                            ):
                                assert accel_metric is not None
                                tracking_metrics_warp.compute_mean_acceleration_error_into_warp(
                                    sim_body_linvel,
                                    ref_body_linvel,
                                    prev_sim_body_linvel,
                                    prev_ref_body_linvel,
                                    dt=float(runtime.env_raw.step_dt),
                                    sim_root_quat=sim_root_quat,
                                    ref_root_quat=ref_root_quat,
                                    prev_sim_root_quat=prev_sim_root_quat,
                                    prev_ref_root_quat=prev_ref_root_quat,
                                    frame=metric_frame,
                                    out=accel_metric,
                                )
                                accel_mask = measurement_mask & prev_seen
                                accel_sum.add_(
                                    accel_metric.mean_error
                                    * accel_mask.to(dtype=accel_metric.mean_error.dtype)
                                )
                                accel_metric_steps.add_(
                                    accel_mask.to(dtype=accel_metric_steps.dtype)
                                )
                                accel_metrics_seen = True
                            prev_sim_body_linvel = sim_body_linvel.clone()
                            prev_ref_body_linvel = ref_body_linvel.clone()
                            prev_sim_root_quat = sim_root_quat.clone()
                            prev_ref_root_quat = ref_root_quat.clone()
                            prev_seen |= active_mask

                        sim_body_pos = metrics.get("tracking_body_pos")
                        ref_body_pos = metrics.get("reference_body_pos")
                        if isinstance(sim_body_pos, torch.Tensor) and isinstance(
                            ref_body_pos, torch.Tensor
                        ):
                            if sim_body_pos.shape != ref_body_pos.shape:
                                raise RuntimeError(
                                    "PHC-style tracking metrics require matching "
                                    "tracking_body_pos and reference_body_pos shapes."
                                )
                            if sim_body_pos.ndim != 3 or sim_body_pos.shape[-1] != 3:
                                raise RuntimeError(
                                    "PHC-style tracking metrics require body position "
                                    "tensors shaped [env, body, 3]."
                                )
                            phc_position_metrics_seen = True
                            phc_global_mpjpe_error = _compute_phc_global_mpjpe_mm(
                                sim_body_pos,
                                ref_body_pos,
                            )
                            phc_global_mpjpe_sum.add_(
                                phc_global_mpjpe_error
                                * measurement_mask.to(
                                    dtype=phc_global_mpjpe_error.dtype
                                )
                            )
                            phc_global_mpjpe_metric_steps.add_(
                                measurement_mask.to(
                                    dtype=phc_global_mpjpe_metric_steps.dtype
                                )
                            )
                            phc_global_mpjpe_metrics_seen = True
                            sim_root_pos = metrics.get("tracking_root_pos")
                            ref_root_pos = metrics.get("reference_root_pos")
                            if isinstance(sim_root_pos, torch.Tensor) and isinstance(
                                ref_root_pos, torch.Tensor
                            ):
                                if sim_root_pos.shape != ref_root_pos.shape:
                                    raise RuntimeError(
                                        "PHC-style MPJPE requires matching "
                                        "tracking_root_pos and reference_root_pos shapes."
                                    )
                                if (
                                    sim_root_pos.ndim != 2
                                    or sim_root_pos.shape[-1] != 3
                                ):
                                    raise RuntimeError(
                                        "PHC-style MPJPE requires root position tensors "
                                        "shaped [env, 3]."
                                    )
                                phc_mpjpe_error = _compute_phc_root_relative_mpjpe_mm(
                                    sim_body_pos,
                                    ref_body_pos,
                                    sim_root_pos,
                                    ref_root_pos,
                                )
                                phc_mpjpe_sum.add_(
                                    phc_mpjpe_error
                                    * measurement_mask.to(dtype=phc_mpjpe_error.dtype)
                                )
                                phc_mpjpe_metric_steps.add_(
                                    measurement_mask.to(
                                        dtype=phc_mpjpe_metric_steps.dtype
                                    )
                                )
                                phc_mpjpe_metrics_seen = True
                            phc_body_count = int(sim_body_pos.shape[1])
                            if phc_metric_body_count != phc_body_count:
                                phc_metric_body_count = phc_body_count
                                prev_sim_body_pos = None
                                prev_ref_body_pos = None
                                prev2_sim_body_pos = None
                                prev2_ref_body_pos = None
                                phc_prev_seen.zero_()
                                phc_prev2_seen.zero_()
                            if (
                                prev_sim_body_pos is not None
                                and prev_ref_body_pos is not None
                            ):
                                phc_vel_mask = measurement_mask & phc_prev_seen
                                phc_vel_error = _compute_phc_position_velocity_error_mm(
                                    sim_body_pos,
                                    ref_body_pos,
                                    prev_sim_body_pos,
                                    prev_ref_body_pos,
                                )
                                phc_vel_sum.add_(
                                    phc_vel_error
                                    * phc_vel_mask.to(dtype=phc_vel_error.dtype)
                                )
                                phc_vel_metric_steps.add_(
                                    phc_vel_mask.to(dtype=phc_vel_metric_steps.dtype)
                                )
                                phc_vel_metrics_seen = True
                                if (
                                    prev2_sim_body_pos is not None
                                    and prev2_ref_body_pos is not None
                                ):
                                    phc_accel_mask = (
                                        measurement_mask
                                        & phc_prev_seen
                                        & phc_prev2_seen
                                    )
                                    phc_accel_error = (
                                        _compute_phc_position_acceleration_error_mm(
                                            sim_body_pos,
                                            ref_body_pos,
                                            prev_sim_body_pos,
                                            prev_ref_body_pos,
                                            prev2_sim_body_pos,
                                            prev2_ref_body_pos,
                                        )
                                    )
                                    phc_accel_sum.add_(
                                        phc_accel_error
                                        * phc_accel_mask.to(dtype=phc_accel_error.dtype)
                                    )
                                    phc_accel_metric_steps.add_(
                                        phc_accel_mask.to(
                                            dtype=phc_accel_metric_steps.dtype
                                        )
                                    )
                                    phc_accel_metrics_seen = True
                                prev2_sim_body_pos = prev_sim_body_pos
                                prev2_ref_body_pos = prev_ref_body_pos
                                phc_prev2_seen.copy_(phc_prev_seen)
                            prev_sim_body_pos = sim_body_pos.clone()
                            prev_ref_body_pos = ref_body_pos.clone()
                            phc_prev_seen |= active_mask

                    runtime.policy_runner.policy_mean_into(
                        obs_policy,
                        runtime.policy_actions,
                    )
                    if use_fast_step_with_io:
                        step_with_io(runtime.step_io)
                        torch.logical_or(
                            runtime.next_terminated,
                            runtime.next_truncated,
                            out=runtime.next_dones,
                        )
                        obs_policy = runtime.next_obs_policy
                        reward_tensor = runtime.next_rewards
                    else:
                        obs, reward_tensor, _terminated, _truncated, _ = (
                            runtime.env.step(runtime.env_actions)
                        )
                        obs_policy = obs["policy"]
                        runtime.next_rewards.copy_(reward_tensor)
                        reward_tensor = runtime.next_rewards

                    if use_fast_step_with_io:
                        terminated = runtime.next_terminated
                        truncated = runtime.next_truncated
                    else:
                        terminated = runtime.env_raw.reset_terminated
                        truncated = runtime.env_raw.reset_time_outs
                    reward_sum.add_(
                        reward_tensor * measurement_mask.to(dtype=reward_tensor.dtype)
                    )
                    terminal_reward.copy_(
                        torch.where(measurement_mask, reward_tensor, terminal_reward)
                    )
                    survival_steps.add_(measurement_mask.to(dtype=survival_steps.dtype))
                    done_now = active_mask & torch.logical_or(terminated, truncated)
                    pre_anchor_failure_mask |= done_now & ~measurement_mask
                    terminated_mask |= done_now & terminated
                    completed_now = active_mask & (
                        survival_steps >= effective_window_steps
                    )
                    truncated_mask |= done_now & truncated & ~completed_now
                    completed_mask |= completed_now
                    active_mask &= ~(done_now | completed_now)

                for batch_idx, anchor in enumerate(batch_anchors):
                    survival_steps_value = int(survival_steps[batch_idx].item())
                    effective_window_steps_value = int(
                        effective_window_steps[batch_idx].item()
                    )
                    row = {
                        "clip_idx": int(anchor.clip_idx),
                        "clip_name": str(clip_names[int(anchor.clip_idx)]),
                        "clip_length": int(clip_lengths[int(anchor.clip_idx)]),
                        "start_frame_idx": int(anchor.start_frame_idx),
                        "reset_start_frame_idx": int(anchor.reset_start_frame_idx),
                        "lead_in_steps_effective": int(anchor.lead_in_steps_effective),
                        "window_steps": int(window_steps),
                        "effective_window_steps": effective_window_steps_value,
                        "survival_steps": survival_steps_value,
                        "survival_ratio": float(survival_steps_value)
                        / float(max(1, effective_window_steps_value)),
                        "failed": int(
                            bool(
                                terminated_mask[batch_idx].item()
                                or truncated_mask[batch_idx].item()
                            )
                        ),
                        "pre_anchor_failure": int(
                            bool(pre_anchor_failure_mask[batch_idx].item())
                        ),
                        "completed": int(bool(completed_mask[batch_idx].item())),
                        "terminated": int(bool(terminated_mask[batch_idx].item())),
                        "truncated": int(bool(truncated_mask[batch_idx].item())),
                        "terminal_reward": float(terminal_reward[batch_idx].item()),
                        "episode_reward_sum": float(reward_sum[batch_idx].item()),
                    }
                    if tracking_metrics_seen:
                        metric_steps_value = int(
                            tracking_metric_steps[batch_idx].item()
                        )
                        accel_steps_value = int(accel_metric_steps[batch_idx].item())
                        phc_mpjpe_steps_value = int(
                            phc_mpjpe_metric_steps[batch_idx].item()
                        )
                        phc_global_mpjpe_steps_value = int(
                            phc_global_mpjpe_metric_steps[batch_idx].item()
                        )
                        phc_vel_steps_value = int(
                            phc_vel_metric_steps[batch_idx].item()
                        )
                        phc_accel_steps_value = int(
                            phc_accel_metric_steps[batch_idx].item()
                        )
                        denom = float(metric_steps_value)
                        row.update(
                            {
                                "tracking_metric_steps": metric_steps_value,
                                "accel_metric_steps": accel_steps_value,
                                "phc_mpjpe_metric_steps": phc_mpjpe_steps_value,
                                "phc_global_mpjpe_metric_steps": (
                                    phc_global_mpjpe_steps_value
                                ),
                                "phc_vel_metric_steps": phc_vel_steps_value,
                                "phc_accel_metric_steps": phc_accel_steps_value,
                                "ape_mean": (
                                    float(ape_sum[batch_idx].item()) / denom
                                    if metric_steps_value > 0
                                    else float("nan")
                                ),
                                "global_mpjpe_mean": (
                                    float(global_mpjpe_sum[batch_idx].item()) / denom
                                    if metric_steps_value > 0
                                    else float("nan")
                                ),
                                "ee_mean": (
                                    float(ee_sum[batch_idx].item()) / denom
                                    if metric_steps_value > 0
                                    else float("nan")
                                ),
                                "vel_mean": (
                                    float(vel_sum[batch_idx].item()) / denom
                                    if temporal_metrics_seen and metric_steps_value > 0
                                    else float("nan")
                                ),
                                "accel_mean": (
                                    float(accel_sum[batch_idx].item())
                                    / float(accel_steps_value)
                                    if accel_metrics_seen and accel_steps_value > 0
                                    else float("nan")
                                ),
                                "phc_mpjpe_mean": (
                                    float(phc_mpjpe_sum[batch_idx].item())
                                    / float(phc_mpjpe_steps_value)
                                    if phc_position_metrics_seen
                                    and phc_mpjpe_metrics_seen
                                    and phc_mpjpe_steps_value > 0
                                    else float("nan")
                                ),
                                "phc_global_mpjpe_mean": (
                                    float(phc_global_mpjpe_sum[batch_idx].item())
                                    / float(phc_global_mpjpe_steps_value)
                                    if phc_position_metrics_seen
                                    and phc_global_mpjpe_metrics_seen
                                    and phc_global_mpjpe_steps_value > 0
                                    else float("nan")
                                ),
                                "phc_vel_mean": (
                                    float(phc_vel_sum[batch_idx].item())
                                    / float(phc_vel_steps_value)
                                    if phc_position_metrics_seen
                                    and phc_vel_metrics_seen
                                    and phc_vel_steps_value > 0
                                    else float("nan")
                                ),
                                "phc_accel_mean": (
                                    float(phc_accel_sum[batch_idx].item())
                                    / float(phc_accel_steps_value)
                                    if phc_position_metrics_seen
                                    and phc_accel_metrics_seen
                                    and phc_accel_steps_value > 0
                                    else float("nan")
                                ),
                            }
                        )
                    rows.append(row)

    except BaseException as exc:
        scan_error = exc
        if current_batch_start is not None:
            context = _format_anchor_scan_exception_context(
                batch_start=current_batch_start,
                batch_anchors=current_batch_anchors,
                clip_names=clip_names,
                clip_lengths=clip_lengths,
                step_idx=current_step_idx,
            )
            raise RuntimeError(f"Anchor scan failed with context: {context}") from exc
        raise
    finally:
        try:
            runtime.env_raw.clear_analysis_reset_overrides()
            synchronize_pending_work = getattr(
                runtime, "synchronize_pending_work", None
            )
            if callable(synchronize_pending_work):
                synchronize_pending_work()
        except BaseException:
            if scan_error is None:
                raise
            logging.getLogger(__name__).exception(
                "Ignoring clip failure benchmark cleanup failure after scan error."
            )
    return rows


def run_full_motion_scan(
    runtime: LoadedAnalysisRuntime,
    *,
    motions: Sequence[FullMotionSpec],
    clip_names: Sequence[str],
    clip_lengths: Sequence[int],
) -> list[dict[str, object]]:
    if not motions:
        return []
    max_window_steps = max(int(item.horizon_steps) for item in motions)
    anchors = [
        ScanAnchorSpec(
            clip_idx=int(item.clip_idx),
            start_frame_idx=0,
            reset_start_frame_idx=0,
            lead_in_steps_effective=0,
        )
        for item in motions
    ]
    rows = run_anchor_scan(
        runtime,
        anchors=anchors,
        clip_names=clip_names,
        clip_lengths=clip_lengths,
        window_steps=int(max_window_steps),
    )
    horizon_by_clip = {int(item.clip_idx): int(item.horizon_steps) for item in motions}
    for row in rows:
        horizon_steps = horizon_by_clip[int(row["clip_idx"])]
        survival_steps = int(row["survival_steps"])
        row["window_steps"] = horizon_steps
        row["effective_window_steps"] = horizon_steps
        row["survival_ratio"] = float(survival_steps) / float(max(1, horizon_steps))
    return rows


def write_analysis_artifacts(
    *,
    output_root: Path,
    anchor_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_root / "results.csv", anchor_rows)
    write_csv_rows(output_root / "summary.csv", summary_rows)
    write_summary_markdown(output_root / "summary.md", summary_rows)
    write_failure_plots(
        output_dir=output_root / "plots",
        summary_rows=summary_rows,
        anchor_rows=anchor_rows,
    )
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def replay_failure_videos(
    *,
    checkpoint_path: str | Path,
    output_root: Path,
    anchor_rows: Sequence[dict[str, object]],
    top_k: int,
    video_length: int,
    subprocess_timeout_seconds: float = DEFAULT_CLIP_REPLAY_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    if top_k <= 0:
        return []

    checkpoint_path_resolved = Path(checkpoint_path).expanduser().resolve()
    video_dir = output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    worst_rows = sorted(
        (row for row in anchor_rows if int(row["failed"]) == 1),
        key=lambda row: (
            int(row["survival_steps"]),
            int(row["clip_idx"]),
            int(row["start_frame_idx"]),
        ),
    )[: int(top_k)]
    video_rows: list[dict[str, object]] = []
    for row in worst_rows:
        clip_idx = int(row["clip_idx"])
        start_frame_idx = int(row["start_frame_idx"])
        replay_start_frame_idx = int(row.get("reset_start_frame_idx", start_frame_idx))
        lead_in_steps_effective = int(row.get("lead_in_steps_effective", 0))
        play_log = video_dir / f"clip{clip_idx:02d}_frame{start_frame_idx:06d}.log"
        command = [
            "uv",
            "run",
            "play",
            f"++play.checkpoint={checkpoint_path_resolved.as_posix()}",
            "play.num_envs=1",
            "++play.tracking.mode=reference",
            "++play.tracking.deterministic_reset=true",
            f"++play.tracking.motion_index={clip_idx}",
            f"++play.tracking.start_frame={replay_start_frame_idx}",
            "play.video.enabled=true",
            f"play.video.length={int(video_length) + lead_in_steps_effective}",
            "++play.video.overlay_rewards=true",
            "++play.video.overlay_history_seconds=3.0",
            "++play.video.overlay_panel_width=360",
            "++play.output.emit_result_payload=true",
        ]
        with play_log.open("w", encoding="utf-8") as handle:
            timeout = _normalize_process_timeout(
                subprocess_timeout_seconds, owner="clip failure replay"
            )
            try:
                completed = subprocess.run(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"clip failure replay timed out after {timeout:.1f}s"
                ) from exc
        payload = _parse_playback_result(play_log)
        video_rows.append(
            {
                "clip_idx": clip_idx,
                "clip_name": str(row["clip_name"]),
                "start_frame_idx": start_frame_idx,
                "replay_start_frame_idx": replay_start_frame_idx,
                "lead_in_steps_effective": lead_in_steps_effective,
                "play_log": play_log.as_posix(),
                "play_returncode": int(completed.returncode),
                "video_output_path": payload.get("video_output_path", ""),
            }
        )
    if video_rows:
        write_csv_rows(video_dir / "results.csv", video_rows)
    return video_rows


def run_lafan_clip_failure_analysis(
    *,
    config_name: str,
    checkpoint_path: str | Path,
    mode: str = "anchor",
    anchor_stride: int = 16,
    window_steps: int = 180,
    lead_in_steps: int = 0,
    batch_size: int = 2048,
    clip_indices: Sequence[int] | None = None,
    output_root: str | Path | None = None,
    seed: int | None = None,
    video: bool = False,
    top_k_failure_videos: int = 3,
    video_length: int | None = None,
    subprocess_timeout_seconds: float = DEFAULT_CLIP_REPLAY_TIMEOUT_SECONDS,
) -> dict[str, object]:
    analysis_mode = _normalize_analysis_mode(mode)
    checkpoint_path_resolved = Path(checkpoint_path).expanduser().resolve()
    output_root_resolved = _resolve_output_root(
        checkpoint_path=checkpoint_path_resolved,
        config_name=str(config_name),
        output_root=output_root,
    )
    output_root_resolved.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    runtime = load_analysis_runtime(
        checkpoint_path=checkpoint_path_resolved,
        batch_size=int(batch_size),
        seed=seed,
    )
    try:
        clip_lengths, clip_names = _clip_info_from_runtime_env(runtime.env_raw)
        selected_clip_indices = (
            [int(idx) for idx in clip_indices]
            if clip_indices is not None and len(clip_indices) > 0
            else list(range(len(clip_lengths)))
        )
        if analysis_mode == "full_motion":
            motions = build_full_motion_table_for_clip_indices(
                clip_lengths=clip_lengths,
                clip_indices=selected_clip_indices,
            )
            anchor_rows = run_full_motion_scan(
                runtime,
                motions=motions,
                clip_names=clip_names,
                clip_lengths=clip_lengths,
            )
        else:
            anchors = build_anchor_table_for_clip_indices(
                clip_lengths=clip_lengths,
                clip_indices=selected_clip_indices,
                anchor_stride=int(anchor_stride),
                window_steps=int(window_steps),
            )
            scan_anchors = prepare_scan_anchors(
                anchors,
                lead_in_steps=int(lead_in_steps),
            )
            anchor_rows = run_anchor_scan(
                runtime,
                anchors=scan_anchors,
                clip_names=clip_names,
                clip_lengths=clip_lengths,
                window_steps=int(window_steps),
            )
        summary_rows = summarize_anchor_rows(anchor_rows)
        metadata = {
            "config_name": str(config_name),
            "checkpoint_path": checkpoint_path_resolved.as_posix(),
            "mode": analysis_mode,
            "anchor_stride": int(anchor_stride),
            "window_steps": int(window_steps),
            "lead_in_steps": int(lead_in_steps),
            "batch_size": int(batch_size),
            "clip_count": len(clip_names),
            "anchor_count": len(anchor_rows),
            "motion_count": len(anchor_rows) if analysis_mode == "full_motion" else 0,
            "selected_clip_indices": selected_clip_indices,
            "runtime_seconds": time.perf_counter() - start_time,
        }
        write_analysis_artifacts(
            output_root=output_root_resolved,
            anchor_rows=anchor_rows,
            summary_rows=summary_rows,
            metadata=metadata,
        )
        video_rows: list[dict[str, object]] = []
        if video:
            video_rows = replay_failure_videos(
                checkpoint_path=checkpoint_path_resolved,
                output_root=output_root_resolved,
                anchor_rows=anchor_rows,
                top_k=int(top_k_failure_videos),
                video_length=int(video_length or window_steps),
                subprocess_timeout_seconds=subprocess_timeout_seconds,
            )
        return {
            "output_root": output_root_resolved,
            "anchor_rows": anchor_rows,
            "summary_rows": summary_rows,
            "video_rows": video_rows,
            "metadata": metadata,
        }
    finally:
        runtime.close()


def run_lafan_clip_failure_analysis_with_runtime(
    *,
    runtime: LoadedAnalysisRuntime,
    config_name: str,
    checkpoint_path: str | Path,
    mode: str = "anchor",
    anchor_stride: int = 16,
    window_steps: int = 180,
    lead_in_steps: int = 0,
    clip_indices: Sequence[int] | None = None,
    output_root: str | Path | None = None,
    video: bool = False,
    top_k_failure_videos: int = 3,
    video_length: int | None = None,
    subprocess_timeout_seconds: float = DEFAULT_CLIP_REPLAY_TIMEOUT_SECONDS,
) -> dict[str, object]:
    analysis_mode = _normalize_analysis_mode(mode)
    checkpoint_path_resolved = Path(checkpoint_path).expanduser().resolve()
    if Path(runtime.checkpoint_path).expanduser().resolve() != checkpoint_path_resolved:
        reload_analysis_runtime_checkpoint(runtime, checkpoint_path_resolved)
    output_root_resolved = _resolve_output_root(
        checkpoint_path=checkpoint_path_resolved,
        config_name=str(config_name),
        output_root=output_root,
    )
    output_root_resolved.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    clip_lengths, clip_names = _clip_info_from_runtime_env(runtime.env_raw)
    selected_clip_indices = (
        [int(idx) for idx in clip_indices]
        if clip_indices is not None and len(clip_indices) > 0
        else list(range(len(clip_lengths)))
    )
    if analysis_mode == "full_motion":
        motions = build_full_motion_table_for_clip_indices(
            clip_lengths=clip_lengths,
            clip_indices=selected_clip_indices,
        )
        anchor_rows = run_full_motion_scan(
            runtime,
            motions=motions,
            clip_names=clip_names,
            clip_lengths=clip_lengths,
        )
    else:
        anchors = build_anchor_table_for_clip_indices(
            clip_lengths=clip_lengths,
            clip_indices=selected_clip_indices,
            anchor_stride=int(anchor_stride),
            window_steps=int(window_steps),
        )
        scan_anchors = prepare_scan_anchors(
            anchors,
            lead_in_steps=int(lead_in_steps),
        )
        anchor_rows = run_anchor_scan(
            runtime,
            anchors=scan_anchors,
            clip_names=clip_names,
            clip_lengths=clip_lengths,
            window_steps=int(window_steps),
        )
    summary_rows = summarize_anchor_rows(anchor_rows)
    metadata = {
        "config_name": str(config_name),
        "checkpoint_path": checkpoint_path_resolved.as_posix(),
        "mode": analysis_mode,
        "anchor_stride": int(anchor_stride),
        "window_steps": int(window_steps),
        "lead_in_steps": int(lead_in_steps),
        "batch_size": int(runtime.batch_size),
        "clip_count": len(clip_names),
        "anchor_count": len(anchor_rows),
        "motion_count": len(anchor_rows) if analysis_mode == "full_motion" else 0,
        "selected_clip_indices": selected_clip_indices,
        "runtime_seconds": time.perf_counter() - start_time,
    }
    write_analysis_artifacts(
        output_root=output_root_resolved,
        anchor_rows=anchor_rows,
        summary_rows=summary_rows,
        metadata=metadata,
    )
    video_rows: list[dict[str, object]] = []
    if video:
        video_rows = replay_failure_videos(
            checkpoint_path=checkpoint_path_resolved,
            output_root=output_root_resolved,
            anchor_rows=anchor_rows,
            top_k=int(top_k_failure_videos),
            video_length=int(video_length or window_steps),
            subprocess_timeout_seconds=subprocess_timeout_seconds,
        )
    return {
        "output_root": output_root_resolved,
        "anchor_rows": anchor_rows,
        "summary_rows": summary_rows,
        "video_rows": video_rows,
        "metadata": metadata,
    }


def _normalize_process_timeout(value: object, *, owner: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner} timeout must be finite and positive") from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(f"{owner} timeout must be finite and positive")
    return timeout
