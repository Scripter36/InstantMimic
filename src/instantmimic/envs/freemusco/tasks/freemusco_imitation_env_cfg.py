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
class FreemuscoImitationEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = field(
        default_factory=lambda: ViewerCfg(resolution=(1080, 1080))
    )
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            mjcf_path="",
            integrator="implicitfast",
            render_interval=5,
            njmax=512,
            nconmax=36,
            hz=400.0,
        )
    )
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(num_envs=1024)
    )
    decimation: int = 10
    episode_length_s: float = 25.0
    observation_space: int | None = None
    action_space: int = 120
    use_cuda_graph: bool = False
    terminate_on_nonfinite: bool = True
    log_dir: str = ""

    species: str = ""
    imitation_profile: dict[str, Any] = field(default_factory=dict)
    motion_clip: str = ""
    motion_clips: list[str] = field(default_factory=list)
    motion_loop: bool = True
    motion_data_dir: str = ""
    mirror_motion_clips: bool = False
    mirror_exclude_motion_clips: list[str] = field(default_factory=list)
    fixed_motion_idx: int | None = None
    start_frame_idx: int = 0
    smooth_cycle_blend_frames: int = 30
    include_phase_in_policy_obs: bool = True
    reward: dict[str, Any] = field(
        default_factory=lambda: {
            "alive": 0.05,
            "tracking": {
                "type": "joint",
                "pose": {
                    "weight": 3.0,
                    "mode": "default",
                    "include_root": False,
                },
                "velocity": {
                    "weight": 0.02,
                    "mode": "qvel",
                    "include_root": True,
                },
            },
            "com_weight": 10.0,
            "end_effector": {
                "position_weight": 8.0,
                "rotation_weight": 2.0,
                "position_root_relative": True,
            },
            "combination": {"type": "product"},
        }
    )
    loop_boundary_vel_extrapolate: bool = False
    loop_boundary_vel_from_first_frame: bool = False
    smooth_cycle: bool = True
    velocity_reward_seam_fade_frames: int = 0
    imitation_reward_threshold: float = 0.01
    deterministic_reset: bool = False
    random_init: bool = True
    ghost_offset: list[float] = field(default_factory=lambda: [0.0, 1.5, 0.0])

    include_muscle_observation_blocks: bool = True
    action: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "mujoco_activation",
            "space": 120,
        }
    )
    muscle: dict[str, Any] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)
    activation_visualization: dict[str, Any] = field(default_factory=dict)
    character_render_mode: str = "mesh"
