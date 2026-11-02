from dataclasses import MISSING, dataclass, field
from typing import Any

from instantmimic.envs.direct_rl_env_cfg import DirectRLEnvCfg, SimulationCfg


@dataclass
class SNUSimulationCfg(SimulationCfg):
    """SNU-family skeleton and muscle source settings."""

    skel_path: str | None = None
    muscle_path: str | None = None
    meshdir: str | None = None
    self_collision: bool = False
    three_dof_as_hinge: bool = False


@dataclass
class SNUImitationEnvCfg(DirectRLEnvCfg):
    sim: SNUSimulationCfg = MISSING
    motion: dict[str, Any]
    action: dict[str, Any]

    reward: dict[str, Any]
    reset: dict[str, Any]
    termination: dict[str, Any]
    observations: dict[str, Any]
    adaptive_sampling: dict[str, Any]
    initial_state: dict[str, Any] = field(default_factory=dict)

    root_body_names: list[str] = field(default_factory=lambda: ["Pelvis", "pelvis"])
    torso_body_names: list[str] = field(default_factory=lambda: ["Torso", "torso"])
    ee_body_name_groups: list[list[str]] = field(
        default_factory=lambda: [
            ["TalusR", "R_Ankle", "RightFoot", "R_Foot", "right_foot"],
            ["TalusL", "L_Ankle", "LeftFoot", "L_Foot", "left_foot"],
            ["HandR", "R_Wrist", "RightHand", "R_Hand", "right_hand"],
            ["HandL", "L_Wrist", "LeftHand", "L_Hand", "left_hand"],
            ["Head", "head"],
        ]
    )
    muscle: dict[str, Any] = field(default_factory=dict)
    activation_visualization: dict[str, Any] = field(default_factory=dict)
    nan_debug: dict[str, Any] = field(default_factory=dict)
    max_joint_velocity: float = -1.0
    character_render_mode: str = "mesh"
    spawn_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ghost_offset: list[float] = field(default_factory=lambda: [0.0, 1.5, 0.0])
