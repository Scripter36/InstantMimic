from __future__ import annotations

from dataclasses import dataclass, field

from instantmimic.envs.direct_rl_env_cfg import (
    DirectRLEnvCfg,
    InteractiveSceneCfg,
    SimulationCfg,
    ViewerCfg,
)


@dataclass(kw_only=True)
class AntEnvCfg(DirectRLEnvCfg):
    # Ant is represented entirely by primitive collision geoms, with no
    # separate visual-mesh group to render.
    character_render_mode: str = "collision"
    viewer: ViewerCfg = field(
        default_factory=lambda: ViewerCfg(resolution=(1080, 1080))
    )
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            mjcf_path="data/ant/nv_ant.xml",
            integrator="implicitfast",
            render_interval=2,
            njmax=200,
            nconmax=50,
            hz=120.0,
            iterations=50,
        )
    )
    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(num_envs=4096)
    )
    decimation: int = 2
    episode_length_s: float = 15.0
    action_space: int = 8
    observation_space: int = 36
    state_space: int = 0

    action_scale: float = 0.5
    joint_gears: tuple[float, ...] = (15.0,) * 8
    heading_weight: float = 0.5
    up_weight: float = 0.1
    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.005
    alive_reward_scale: float = 0.5
    dof_vel_scale: float = 0.2
    death_cost: float = -2.0
    termination_height: float = 0.31
    angular_velocity_scale: float = 1.0
    contact_force_scale: float = 0.1

    use_cuda_graph: bool = True
    compile_mjw_step: bool = False
    terminate_on_nonfinite: bool = True
    use_parallelized_kernel_launch: bool = True
    use_optimized_kernel: bool = True
    use_native_task_kernels: bool = True
    log_dir: str = ""
