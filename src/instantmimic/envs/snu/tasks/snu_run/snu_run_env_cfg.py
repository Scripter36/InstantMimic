from dataclasses import dataclass

from instantmimic.envs.direct_rl_env_cfg import DirectRLEnvCfg


@dataclass
class SNURunEnvCfg(DirectRLEnvCfg):
    joint_coordinate_mjcf_path: str

    heading_weight: float
    up_weight: float

    energy_cost_scale: float
    actions_cost_scale: float
    alive_reward_scale: float
    dof_vel_scale: float

    death_cost: float
    termination_height: float

    angular_velocity_scale: float
