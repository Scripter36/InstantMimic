"""Small FreeMusco configurations for environment behavior tests."""

from typing import Literal

import mujoco

from instantmimic.envs.direct_rl_env_cfg import InteractiveSceneCfg
from instantmimic.envs.freemusco.tasks.freemusco_imitation_env_cfg import (
    FreemuscoImitationEnvCfg,
)
from instantmimic.envs.freemusco.tasks.freemusco_run_env_cfg import FreemuscoRunEnvCfg
from instantmimic.utils.project_paths import resolve_project_path

FreemuscoSpecies = Literal["chimanoid", "ostrich"]


def _run_model_values(species: FreemuscoSpecies) -> dict[str, object]:
    if species == "ostrich":
        return {
            "root_body_name": "ostrich",
            "head_body_name": "head",
            "foot_body_names": ["r_pes", "l_pes"],
            "root_tilt_joint_names": ["root_rot_x", "root_rot_y"],
            "root_yaw_joint_name": "root_rot_z",
            "supports_dynamic_target_forward_velocity": True,
            "mjcf_path": "data/freemusco/animal/ostrich/ostrich.xml",
        }
    return {
        "root_body_name": "pelvis",
        "head_body_name": "thorax",
        "torso_body_name": "thorax",
        "foot_body_names": ["toes_r", "toes_l"],
        "root_yaw_joint_name": "pelvis_rotation",
        "forward_joint_name": "pelvis_tx",
        "qpos_observation_skip": "forward_joint",
        "torso_tilt_source": "torso_upright",
        "mjcf_path": "data/freemusco/animal/chimanoid/chimanoid.xml",
    }


def build_freemusco_run_cfg(
    *,
    species: FreemuscoSpecies,
    include_muscle_observation_blocks: bool,
    include_heading_frame_body_state_obs: bool = False,
    action_type: Literal["mujoco_activation", "torque"] = "mujoco_activation",
) -> FreemuscoRunEnvCfg:
    model_values = _run_model_values(species)
    mjcf_path = str(model_values.pop("mjcf_path"))
    cfg = FreemuscoRunEnvCfg(
        seed=1234,
        scene=InteractiveSceneCfg(num_envs=1),
        species=species,
        action={"type": "mujoco_activation", "space": 120},
        observations={"previous_action": None, "applied_torque": False},
        include_muscle_observation_blocks=include_muscle_observation_blocks,
        include_heading_frame_body_state_obs=include_heading_frame_body_state_obs,
        **model_values,
    )
    cfg.sim.mjcf_path = mjcf_path
    if action_type == "torque":
        model = mujoco.MjModel.from_xml_path(str(resolve_project_path(mjcf_path)))
        cfg.action = {"type": "torque", "space": model.nv - 6, "scale": 1.0}
    cfg.activation_visualization = {"enabled": False}
    return cfg


def build_freemusco_imitation_cfg(
    *,
    species: FreemuscoSpecies,
    mass: bool = False,
) -> FreemuscoImitationEnvCfg:
    values = _run_model_values(species)
    path = str(values["mjcf_path"])
    model = mujoco.MjModel.from_xml_path(str(resolve_project_path(path)))
    ostrich = species == "ostrich"
    profile = {
        "root_body_name": values["root_body_name"],
        "head_body_name": values["head_body_name"],
        "foot_body_names": values["foot_body_names"],
        "pelvis_geom_name": "pelvis",
        "end_effector_body_names": values["foot_body_names"],
        "talus_body_names": [] if ostrich else ["talus_r", "talus_l"],
        "toe_body_names": [] if ostrich else ["toes_r", "toes_l"],
        "reference_body_start": 1,
        "reference_body_count_offset": 2 if ostrich else 1,
        "root_qpos_position_indices": [0, 1, 2],
        "root_qpos_rotation_indices": [3, 4, 5],
        "loop_vertical_axis": 2,
        "phase_observation_mode": "configurable" if ostrich else "required",
        "loop_transform_mode": "extrapolated_root_yaw"
        if ostrich
        else "endpoint_root_pose",
        "mirror_mode": "none",
        "smoothing_mode": "none",
        "ghost_root_xpath": ".//body[@name='ostrich']"
        if ostrich
        else ".//body[@name='pelvis']",
        "ghost_root_description": "Test character root body",
    }
    action = {"type": "torque", "space": model.nv - 6, "scale": 1.0}
    if mass:
        action = {
            "type": "mass",
            "space": model.nv - 6,
            "policy_output": {"type": "torque", "scale": 1.0},
            "muscle_nn": {"hidden_dims": [8], "batch_size": 2, "buffer_capacity": 16},
        }
    cfg = FreemuscoImitationEnvCfg(
        seed=1234,
        scene=InteractiveSceneCfg(num_envs=1),
        species=species,
        imitation_profile=profile,
        action=action,
        motion_clip="cyclic" if ostrich else "rest_loop_2frame",
        motion_data_dir=f"data/freemusco/animal/{species}/mocap",
        random_init=False,
        smooth_cycle=False,
        observations={
            "previous_action": "action",
            "applied_torque": True,
            "muscle_geometry": True,
        },
        activation_visualization={"enabled": False},
        include_muscle_observation_blocks=False,
    )
    cfg.sim.mjcf_path = path
    return cfg
