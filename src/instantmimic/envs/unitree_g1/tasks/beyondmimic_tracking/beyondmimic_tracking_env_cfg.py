from __future__ import annotations

from dataclasses import dataclass, field

from instantmimic.envs.direct_rl_env_cfg import (
    DirectRLEnvCfg,
    InteractiveSceneCfg,
    SimulationCfg,
    ViewerCfg,
)


@dataclass
class MotionCfg:
    file: str = "data/unitree_g1/beyondmimic/motions/converted/side_kick_motion.npz"
    sampling_mode: str = "adaptive"
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001


@dataclass
class ResetCfg:
    position_range: tuple[tuple[float, float], ...] = (
        (-0.05, 0.05),
        (-0.05, 0.05),
        (-0.01, 0.01),
    )
    euler_range: tuple[tuple[float, float], ...] = (
        (-0.1, 0.1),
        (-0.1, 0.1),
        (-0.2, 0.2),
    )
    velocity_range: tuple[tuple[float, float], ...] = (
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.2, 0.2),
        (-0.52, 0.52),
        (-0.52, 0.52),
        (-0.78, 0.78),
    )
    joint_position_range: tuple[float, float] = (-0.1, 0.1)


@dataclass
class RandomizationCfg:
    enabled: bool = True
    push_enabled: bool = True
    foot_friction_range: tuple[float, float] = (0.3, 1.2)
    encoder_bias_range: tuple[float, float] = (-0.01, 0.01)
    torso_com_offset_range: tuple[tuple[float, float], ...] = (
        (-0.025, 0.025),
        (-0.05, 0.05),
        (-0.05, 0.05),
    )
    push_interval_range_s: tuple[float, float] = (1.0, 3.0)
    push_velocity_range: tuple[tuple[float, float], ...] = (
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.2, 0.2),
        (-0.52, 0.52),
        (-0.52, 0.52),
        (-0.78, 0.78),
    )


@dataclass
class ObservationCfg:
    corruption: bool = True


@dataclass(kw_only=True)
class BeyondMimicTrackingEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = field(
        default_factory=lambda: ViewerCfg(resolution=(1080, 1080))
    )
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            mjcf_path="data/unitree_g1/beyondmimic/mjcf/scene.xml",
            integrator="implicitfast",
            render_interval=4,
            njmax=250,
            nconmax=35,
            hz=200.0,
            iterations=10,
            ls_iterations=20,
        )
    )
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(num_envs=4096)
    )
    decimation: int = 4
    episode_length_s: float = 10.0
    observation_space: int = 160
    state_space: int = 286
    action_space: int = 29
    use_cuda_graph: bool = False
    terminate_on_nonfinite: bool = True
    track_done_reasons: bool = True
    execution_mode: str = "reference"
    log_dir: str = ""

    motion: MotionCfg = field(default_factory=MotionCfg)
    reset: ResetCfg = field(default_factory=ResetCfg)
    randomization: RandomizationCfg = field(default_factory=RandomizationCfg)
    observations: ObservationCfg = field(default_factory=ObservationCfg)
