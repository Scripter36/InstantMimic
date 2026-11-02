from __future__ import annotations

from dataclasses import dataclass, field

from instantmimic.envs.snu.tasks.snu_imitation.snu_imitation_env_cfg import (
    SNUImitationEnvCfg,
)


@dataclass
class SNUCommandEnvCfg(SNUImitationEnvCfg):
    """Unified command env config."""

    command_mode: str = "velocity_heading"
    low_level_checkpoint: str | None = None
    low_level_action_space: int | None = None
    low_level_residual_scale: float = 1.0
    low_level_append_prior_to_obs: bool = True
    command_residual_l2_weight: float = 0.0
    command_residual_rate_weight: float = 0.0
    command_fall_penalty: float = 0.0
    low_level_config_autoload: bool = False
    low_level_reset_motion_subsets: list[str] = field(
        default_factory=lambda: ["ACCAD/*Walking_c3d/*.npz", "ACCAD/*Running_c3d/*.npz"]
    )
    command_alive_reward: float = 0.05
    command_upright_weight: float = 0.1
    command_torso_height_weight: float = 0.0
    target_speed_min: float = 0.5
    target_speed_max: float = 3.5
    fixed_target_speed: float | None = None
    fixed_target_heading: float | None = None
    velocity_heading_reward_mode: str = "forward_speed_heading"
    velocity_command_resample_interval_min_s: float = 0.0
    velocity_command_resample_interval_max_s: float = 0.0
    velocity_command_arrow_tail_mesh_path: str | None = None
    velocity_command_arrow_body_mesh_path: str | None = None
    velocity_command_arrow_head_mesh_path: str | None = None
    velocity_command_arrow_offset: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    velocity_command_arrow_scale: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )
    velocity_heading_speed_sigma: float = 0.75
    velocity_heading_heading_weight: float = 0.25
    point_goal_distance_min: float = 1.0
    point_goal_distance_max: float = 5.0
    point_goal_world_radius_max: float | None = None
    point_goal_radius: float = 0.4
    point_goal_marker_mesh_path: str | None = None
    point_goal_marker_texture_path: str | None = None
    point_goal_marker_offset: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    point_goal_marker_scale: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )
    point_goal_marker_rgba: list[float] | None = None
    point_goal_progress_weight: float = 1.0
    point_goal_facing_weight: float = 0.1
    point_goal_bonus: float = 2.0
    point_goal_stall_penalty_weight: float = 0.05
    point_goal_min_progress_for_stall: float = 0.01
