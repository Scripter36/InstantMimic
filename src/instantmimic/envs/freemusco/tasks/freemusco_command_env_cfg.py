from __future__ import annotations

from dataclasses import dataclass, field

from instantmimic.envs.freemusco.tasks.freemusco_imitation_env_cfg import (
    FreemuscoImitationEnvCfg,
)


@dataclass(kw_only=True)
class FreemuscoCommandEnvCfg(FreemuscoImitationEnvCfg):
    species: str = "ostrich"
    imitation_reward_threshold: float = 0.0
    command_mode: str = "velocity_heading"  # velocity_heading | point_goal

    termination_height: float = 0.6
    enable_early_termination: bool = True
    early_termination_reward_threshold: float = 0.05
    early_termination_time_threshold: int = 4

    velocity_command_arrow_tail_mesh_path: str | None = None
    velocity_command_arrow_body_mesh_path: str | None = None
    velocity_command_arrow_head_mesh_path: str | None = None
    velocity_command_arrow_offset: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    velocity_command_arrow_scale: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )

    point_goal_marker_mesh_path: str | None = None
    point_goal_marker_texture_path: str | None = None
    point_goal_marker_offset: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    point_goal_marker_scale: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )
    point_goal_marker_rgba: list[float] | None = None

    command_alive_reward: float = 0.05
    command_upright_weight: float = 0.1
    command_torso_height_weight: float = 0.0

    target_speed_min: float = 0.5
    target_speed_max: float = 3.5
    fixed_target_speed: float | None = None
    fixed_target_heading: float | None = None
    velocity_heading_direction_only: bool = False
    velocity_heading_reward_mode: str = "forward_speed_heading"
    velocity_command_resample_interval_min_s: float = 0.0
    velocity_command_resample_interval_max_s: float = 0.0
    velocity_heading_speed_sigma: float = 0.75
    velocity_heading_heading_weight: float = 0.25

    point_goal_distance_min: float = 1.0
    point_goal_distance_max: float = 5.0
    point_goal_radius: float = 0.4
    point_goal_progress_weight: float = 1.0
    point_goal_facing_weight: float = 0.1
    point_goal_bonus: float = 2.0
    point_goal_stall_penalty_weight: float = 0.05
    point_goal_min_progress_for_stall: float = 0.01
