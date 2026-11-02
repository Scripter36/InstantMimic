from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from instantmimic.envs.direct_rl_env_cfg import (
    DirectRLEnvCfg,
    InteractiveSceneCfg,
    SimulationCfg,
    ViewerCfg,
)


@dataclass(kw_only=True)
class FreemuscoRunEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = field(
        default_factory=lambda: ViewerCfg(resolution=(1080, 1080))
    )
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            mjcf_path="",
            integrator="implicitfast",
            render_interval=10,
            njmax=800,
            nconmax=500,
            hz=400.0,
        )
    )
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(num_envs=1024)
    )
    species: str = ""
    decimation: int = 10
    episode_length_s: float = 25.0
    observation_space: int | None = None
    action_space: int = 120
    use_cuda_graph: bool = False
    compile_mjw_step: bool = False
    log_dir: str = ""
    deterministic_reset: bool = False
    randomize_reset_joints: bool = True
    reset_joint_range_scale: float = 0.2

    root_body_name: str = ""
    head_body_name: str = ""
    torso_body_name: str | None = None
    foot_body_names: list[str] = field(default_factory=list)
    pelvis_geom_name: str = "pelvis"
    root_tilt_joint_names: list[str] = field(default_factory=list)
    root_yaw_joint_name: str = ""
    forward_joint_name: str | None = None
    qpos_observation_skip: str = "first"
    torso_tilt_source: str = "root_tilt_xy"
    forward_velocity_sensor_name: str = "torso_subtreelinvel"
    reset_keyframe_name: str | None = None
    forward_velocity_source: str = "qvel_root_x"
    supports_dynamic_target_forward_velocity: bool = False
    observation_include_target_forward_velocity: bool = True
    observation_include_velocity_error: bool = True
    termination_torso_tilt_mode: str = "xy_norm"

    reward_mode: str = "track"
    target_forward_velocity: float = 0.0
    target_forward_velocity_min: float | None = None
    target_forward_velocity_max: float | None = None
    target_forward_velocity_override: float | None = None
    reward_track_sigma: float = 1.0
    reward_velocity_scale: float = 1.0
    reward_upright_scale: float = 0.5
    reward_head_height_target: float = 1.0
    reward_head_height_sigma: float = 0.10
    reward_head_height_scale: float = 0.10
    reward_direction_scale: float = 0.0
    reward_pose_regularization_mode: str = "joint_qpos"
    reward_pose_regularization_scale: float = 0.02
    reward_pose_regularization_body_pos_weight: float = 1.0
    reward_pose_regularization_body_rot_weight: float = 1.0
    reward_pelvis_height_threshold: float = 0.6
    reward_pelvis_height_penalty_scale: float = 0.0
    min_head_height: float = 0.0
    min_pelvis_height: float = 0.0
    max_torso_tilt_rad: float = 1.0
    max_joint_velocity: float = 0.0

    action: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "mujoco_activation",
            "space": 120,
        }
    )
    muscle: dict[str, Any] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)
    activation_visualization: dict[str, Any] = field(default_factory=dict)
    include_muscle_observation_blocks: bool = True
    include_heading_frame_body_state_obs: bool = False
    character_render_mode: str = "mesh"
