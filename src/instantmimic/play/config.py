"""Public playback config helpers shared by play-side tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, open_dict

from instantmimic.config.artifacts import load_artifact_semantic_record
from instantmimic.config.loader import load_semantic_config
from instantmimic.config.play_config import (
    PlayBenchmarkConfig,
    PlayCameraConfig,
    PlayOutputConfig,
    PlayTrackingConfig,
    PlayVideoConfig,
    PlayViewerConfig,
    WarpVideoConfig,
)
from instantmimic.config.runtime_specs import mutable_environment_config
from instantmimic.config.schema import SemanticConfig
from instantmimic.play.resolver import (
    bundle_config_path,
    cli_dotlist_overrides,
    cli_override_value,
    resolve_artifact_paths,
    resolve_checkpoint_path,
    resolve_mjcf_path_for_play,
    task_dotlist_overrides,
    validate_playback_dotlist_overrides,
)

__all__ = [
    "ArtifactPlaybackRuntime",
    "apply_play_free_run_overrides",
    "bundle_config_path",
    "cli_dotlist_overrides",
    "cli_override_value",
    "load_artifact_playback_runtime",
    "resolve_checkpoint_path",
    "resolve_artifact_paths",
    "resolve_mjcf_path_for_play",
    "task_dotlist_overrides",
    "validate_playback_dotlist_overrides",
]


@dataclass(frozen=True, slots=True)
class ArtifactPlaybackRuntime:
    semantic_config: SemanticConfig
    execution_config: DictConfig
    checkpoint_path: Path
    log_dir: Path
    seed: int


def load_artifact_playback_runtime(
    checkpoint_path: str | Path,
    *,
    num_envs: int,
    steps: int | None = None,
    seed: int | None = None,
    use_cuda_graph: bool = False,
    action_mode: str = "mean",
    runtime_overrides: Mapping[str, bool | str] | None = None,
    log_dir: str | Path | None = None,
) -> ArtifactPlaybackRuntime:
    """Build the canonical non-video execution config for an artifact utility."""

    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    artifact = load_artifact_semantic_record(resolved_checkpoint)
    resolved_log_dir = (
        Path(log_dir).expanduser().resolve()
        if log_dir is not None
        else (
            artifact.bundle_config_path.parent.parent.parent.resolve()
            if artifact.bundle_config_path is not None
            else resolved_checkpoint.parent.parent.resolve()
        )
    )
    resolved_seed = (
        int(seed)
        if seed is not None
        else (
            int(artifact.resolved_seed)
            if artifact.resolved_seed is not None
            else max(0, int(artifact.config.seed))
        )
    )
    values = artifact.config.to_dict()
    values["seed"] = resolved_seed
    if runtime_overrides:
        values["runtime"] = {**values["runtime"], **dict(runtime_overrides)}
    values["play"] = {
        "checkpoint": resolved_checkpoint.as_posix(),
        "num_envs": int(num_envs),
        "steps": int(steps) if steps is not None and int(steps) > 0 else None,
        "use_cuda_graph": bool(use_cuda_graph),
        "action": {"mode": action_mode},
        "video": {"enabled": False},
    }
    semantic = load_semantic_config(values)
    execution = lower_playback_execution_config(
        semantic,
        seed=resolved_seed,
        log_dir=resolved_log_dir,
    )
    execution = resolve_artifact_paths(
        execution,
        resolved_log_dir,
        bundle_cfg_path=bundle_config_path(resolved_log_dir),
    )
    with open_dict(execution):
        execution.env.scene.num_envs = int(num_envs)
        execution.checkpoint = resolved_checkpoint.as_posix()
    return ArtifactPlaybackRuntime(
        semantic_config=semantic,
        execution_config=execution,
        checkpoint_path=resolved_checkpoint,
        log_dir=resolved_log_dir,
        seed=resolved_seed,
    )


def lower_playback_execution_config(
    config: SemanticConfig,
    *,
    seed: int,
    log_dir: str | Path,
) -> DictConfig:
    """Lower v1 playback controls for the flat renderer/stepper execution API."""

    runtime_cfg = DictConfig(config.to_dict())
    env_cfg = mutable_environment_config(config, seed=seed, log_dir=log_dir)
    play = config.play
    action = play.action
    if action is None or play.num_envs is None:
        raise ValueError("play.action and play.num_envs are required")
    action_mode = action.mode
    tracking = play.tracking or PlayTrackingConfig()
    viewer = play.viewer or PlayViewerConfig()
    video = play.video or PlayVideoConfig()
    benchmark = play.benchmark or PlayBenchmarkConfig()
    output = play.output or PlayOutputConfig()
    with open_dict(runtime_cfg):
        runtime_cfg.env = env_cfg
        runtime_cfg.seed = int(seed)
        runtime_cfg.checkpoint = play.checkpoint
        runtime_cfg.play_num_envs = play.num_envs
        runtime_cfg.play_steps = play.steps or 0
        runtime_cfg.play_time_limit_policy = play.time_limit_policy
        runtime_cfg.play_done_policy = play.done_policy
        runtime_cfg.play_use_cuda_graph = play.use_cuda_graph
        runtime_cfg.play_allow_eager_fallback = play.allow_eager_fallback
        runtime_cfg.play_action_mode = action_mode
        runtime_cfg.stochastic = action_mode == "sample"
        tracking_mode = tracking.mode
        runtime_cfg.play_free_run = tracking_mode == "free_run"
        runtime_cfg.tracking_metric_frame = tracking.metric_frame
        runtime_cfg.tracking_metric_include_root_in_ape = tracking.include_root_in_ape
        if "require_full_horizon" in tracking:
            runtime_cfg.tracking_require_full_horizon = bool(
                tracking["require_full_horizon"]
            )
        if tracking:
            reset = env_cfg.get("reset")
            if reset is None:
                env_cfg.reset = {}
                reset = env_cfg.reset
            with open_dict(reset):
                if "deterministic_reset" in tracking:
                    reset.deterministic = bool(tracking["deterministic_reset"])
                    if bool(tracking["deterministic_reset"]):
                        if "random_start" in reset:
                            reset.random_start = False
                        if "motion_noise_std" in reset:
                            reset.motion_noise_std = 0.0
                if "motion_index" in tracking:
                    reset.motion_index = tracking["motion_index"]
                if "start_frame" in tracking:
                    reset.start_frame = tracking["start_frame"]
            if tracking.deterministic_reset:
                adaptive_sampling = env_cfg.get("adaptive_sampling")
                if adaptive_sampling is not None and "enabled" in adaptive_sampling:
                    adaptive_sampling.enabled = False
            if tracking.termination == "disabled":
                termination = env_cfg.get("termination")
                if termination is not None:
                    with open_dict(termination):
                        if "min_torso_height" in termination:
                            termination.min_torso_height = -1000000000.0
                        if "motion_end" in termination:
                            termination.motion_end = False
                        early = termination.get("early")
                        if early is not None:
                            with open_dict(early):
                                early.enabled = False
                else:
                    for key, value in (
                        ("enable_early_termination", False),
                        ("termination_height", -1000000000.0),
                        ("early_termination_reward_threshold", -1000000000.0),
                        ("early_termination_time_threshold", 1000000000),
                        ("terminate_at_motion_end", False),
                    ):
                        if key in env_cfg:
                            env_cfg[key] = value
        runtime_cfg.viewer_backend = viewer.backend
        runtime_cfg.viewer_reset_on_done = viewer.reset_on_done
        if "port" in viewer:
            runtime_cfg.viewer_port = int(viewer["port"])
        if "resolution" in viewer:
            env_viewer = env_cfg.get("viewer")
            if env_viewer is None:
                env_cfg.viewer = {}
                env_viewer = env_cfg.viewer
            with open_dict(env_viewer):
                env_viewer.resolution = list(viewer["resolution"])
        runtime_cfg.video = video.enabled
        raw_video_length = video.length or play.steps
        if raw_video_length is None:
            raw_video_length = "episode" if runtime_cfg.video else 0
        runtime_cfg.video_length = (
            str(raw_video_length)
            if str(raw_video_length).strip().lower() == "episode"
            else int(raw_video_length)
        )
        for semantic_key, execution_key in (
            ("render_backend", "video_render_backend"),
            ("encoder_backend", "video_encoder_backend"),
            ("codec", "video_codec"),
            ("tile_envs", "video_tile_envs"),
            ("tile_cols", "video_tile_cols"),
            ("overlay_rewards", "video_overlay_rewards"),
            ("overlay_history_seconds", "video_overlay_history_seconds"),
            ("overlay_panel_width", "video_overlay_panel_width"),
            ("remux_timeout_seconds", "video_remux_timeout_seconds"),
        ):
            if semantic_key in video:
                runtime_cfg[execution_key] = video[semantic_key]
        warp = video.warp or WarpVideoConfig()
        for semantic_key, execution_key in (
            ("use_textures", "video_warp_use_textures"),
            ("floor_shader_mode", "video_warp_floor_shader_mode"),
            ("render_block_dim", "video_warp_render_block_dim"),
            ("lbs_mesh_mode", "video_warp_lbs_mesh_mode"),
            ("lbs_mesh_bundle", "video_warp_lbs_mesh_bundle"),
            ("lbs_mesh_shading_mode", "video_warp_lbs_mesh_shading_mode"),
            ("lbs_activation_mapping", "video_warp_lbs_activation_mapping"),
        ):
            if semantic_key in warp:
                runtime_cfg[execution_key] = warp[semantic_key]
        runtime_cfg.video_benchmark_sync = bool(benchmark)
        if benchmark:
            runtime_cfg.video_benchmark_warmup_frames = benchmark.warmup_frames
            runtime_cfg.video_benchmark_stage_sync = benchmark.stage_sync
        if "directory" in output and output["directory"] is not None:
            runtime_cfg.video_output_dir = str(output["directory"])
        runtime_cfg.emit_playback_result_payload = output.emit_result_payload
        camera = play.camera or PlayCameraConfig()
        for semantic_key, execution_key in (
            ("azimuth_deg", "playback_camera_azimuth_deg"),
            ("elevation_deg", "playback_camera_elevation_deg"),
            ("distance", "playback_camera_distance"),
        ):
            if semantic_key in camera:
                runtime_cfg[execution_key] = camera[semantic_key]
        if "lookat_ema_alpha" in camera:
            runtime_cfg.cam_lookat_ema_alpha = camera["lookat_ema_alpha"]
        if "views" in camera:
            runtime_cfg.playback_camera_views = config.to_dict()["play"]["camera"][
                "views"
            ]
        runtime_cfg.env = env_cfg
    return runtime_cfg


def apply_play_free_run_overrides(cfg: DictConfig) -> None:
    if not bool(cfg.get("play_free_run", False)):
        return

    with open_dict(cfg):
        cfg.tracking_require_full_horizon = False

    env_cfg = cfg.get("env", None)
    if env_cfg is None:
        return
    action_cfg = env_cfg.get("action", {})
    if bool(action_cfg.get("use_reference", False)):
        raise ValueError(
            "env.action.use_reference=true is incompatible with play_free_run=true. "
            "Reference-action policies add reference motion controls inside the env, "
            "so use a checkpoint trained with env.action.use_reference=false for "
            "free-run playback."
        )

    preserve_height_termination = False

    float_zero_keys = (
        "imitation_pose_reward_weight",
        "imitation_rot_reward_weight",
    )
    with open_dict(env_cfg):
        for key in float_zero_keys:
            if key in env_cfg:
                env_cfg[key] = 0.0
        reward_cfg = env_cfg.get("reward", None)
        if reward_cfg is not None:
            reward_cfg.alive = 0.0
            reward_cfg.com_weight = 0.0
            tracking = reward_cfg.get("tracking", None)
            if tracking is not None:
                if str(tracking.get("type", "")).lower() == "body_state":
                    for key in (
                        "position",
                        "rotation",
                        "linear_velocity",
                        "angular_velocity",
                    ):
                        tracking.weights[key] = 0.0
                else:
                    tracking.pose.weight = 0.0
                    tracking.velocity.weight = 0.0
            end_effector = reward_cfg.get("end_effector", None)
            if end_effector is not None:
                end_effector.position_weight = 0.0
                end_effector.rotation_weight = 0.0
        termination = env_cfg.get("termination", None)
        if termination is not None:
            if not preserve_height_termination:
                termination.min_torso_height = -1000000000.0
            termination.motion_end = False
            early = termination.get("early", None)
            if early is not None:
                early.enabled = False
        else:
            if "enable_early_termination" in env_cfg:
                env_cfg.enable_early_termination = False
            if "termination_height" in env_cfg and not preserve_height_termination:
                env_cfg.termination_height = -1000000000.0
            if "early_termination_reward_threshold" in env_cfg:
                env_cfg.early_termination_reward_threshold = -1000000000.0
            if "early_termination_time_threshold" in env_cfg:
                env_cfg.early_termination_time_threshold = 1000000000
            if "terminate_at_motion_end" in env_cfg:
                env_cfg.terminate_at_motion_end = False
        if "imitation_reward_threshold" in env_cfg:
            env_cfg.imitation_reward_threshold = -1000000000.0
        motion = env_cfg.get("motion", None)
        if motion is not None:
            loop = motion.get("loop", None)
            if loop is not None:
                loop.smooth = True
        if "motion_loop" in env_cfg:
            env_cfg.motion_loop = True
        if "reward_combination" in env_cfg:
            env_cfg.reward_combination = "raw_sum"
