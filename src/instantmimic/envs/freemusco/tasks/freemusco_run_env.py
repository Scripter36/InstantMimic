from __future__ import annotations

import logging
from typing import Any

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from instantmimic.envs.common import warp_to_torch
from instantmimic.envs.config_access import get_block, get_value
from instantmimic.envs.config_access import get_value as _cfg_get
from instantmimic.envs.controls.fast_muscle.bindings import (
    FastMuscleExecutionBindings,
)
from instantmimic.envs.controls.fast_muscle.config_models import (
    ActionSettings,
    ActivationVisualizationSettings,
    FastMuscleObservationSettings,
    MuscleSettings,
    set_runtime_action_space,
)
from instantmimic.envs.controls.fast_muscle.controller import (
    FastMuscleController,
)
from instantmimic.envs.controls.fast_muscle.env_bridge import (
    FastMuscleEnvMixin,
)
from instantmimic.envs.controls.fast_muscle.render_builder import (
    build_plain_render_model,
)
from instantmimic.envs.controls.fast_muscle.render_utils import (
    apply_character_geom_render_visibility,
)
from instantmimic.envs.controls.fast_muscle.settings import FastMuscleControlSettings
from instantmimic.envs.direct_rl_env import DirectRLEnv
from instantmimic.envs.model_index import dof_index, qpos_index, resolve_id
from instantmimic.motion import quaternion as torch_quaternion
from instantmimic.muscle.parsing.mjcf import FastMuscleBootstrap
from instantmimic.rendering.character import resolve_character_render_mode

from .freemusco_run.runtime import (
    REWARD_WEIGHT_COUNT,
    REWARD_WEIGHT_DIRECTION_SCALE,
    REWARD_WEIGHT_HEAD_HEIGHT_SCALE,
    REWARD_WEIGHT_HEAD_HEIGHT_SIGMA,
    REWARD_WEIGHT_HEAD_HEIGHT_TARGET,
    REWARD_WEIGHT_PELVIS_HEIGHT_PENALTY_SCALE,
    REWARD_WEIGHT_PELVIS_HEIGHT_THRESHOLD,
    REWARD_WEIGHT_POSE_BODY_POS_WEIGHT,
    REWARD_WEIGHT_POSE_BODY_ROT_WEIGHT,
    REWARD_WEIGHT_POSE_REGULARIZATION_SCALE,
    REWARD_WEIGHT_TRACK_SIGMA,
    REWARD_WEIGHT_UPRIGHT_SCALE,
    REWARD_WEIGHT_VELOCITY_SCALE,
    TORSO_TILT_MODE_ROOT_ROT_Y_ABS,
    TORSO_TILT_MODE_XY_NORM,
    TORSO_TILT_SOURCE_BODY,
    TORSO_TILT_SOURCE_ROOT,
    FreemuscoRunRuntime,
    build_freemusco_run_kernel_bundle,
)
from .freemusco_run_env_cfg import FreemuscoRunEnvCfg
from .profile import FreemuscoRunLayout, FreemuscoRunProfile

_LOGGER = logging.getLogger(__name__)
POSE_REG_MODE_JOINT_QPOS = 0
POSE_REG_MODE_HEADING_FRAME_BODY_POSE = 1
REWARD_MODE_TRACK = 0
REWARD_MODE_MAXIMIZE = 1
REWARD_MODE_TRACK_DYNAMIC = 2
REWARD_MODE_RAW_FORWARD_VELOCITY = 3
FORWARD_VELOCITY_SOURCE_QVEL_ROOT_X = 0
FORWARD_VELOCITY_SOURCE_SENSOR_SUBTREELINVEL_X = 1
MUJOCO_ACTIVATION_MODE = "mujoco_activation"
FREEMUSCO_RUN_ACTION_MODES = {
    MUJOCO_ACTIVATION_MODE,
    "activation",
    "torque",
}


def _resolve_action_mode(cfg: Any) -> str:
    action = get_block(cfg, "action")
    action_mode = str(get_value(action, "type"))
    return _validate_action_mode(action_mode)


def _validate_action_mode(action_mode: str) -> str:
    if action_mode not in FREEMUSCO_RUN_ACTION_MODES:
        supported = sorted(FREEMUSCO_RUN_ACTION_MODES)
        raise ValueError(
            "FreeMusco Run env.action.type must be one of "
            f"{supported}, got {action_mode!r}."
        )
    return action_mode


class FreemuscoRunEnv(FastMuscleEnvMixin, DirectRLEnv):
    cfg: FreemuscoRunEnvCfg

    def _init_reset_behavior(self) -> None:
        self._deterministic_reset = bool(
            _cfg_get(self.cfg, "deterministic_reset", False)
        )
        self._reset_randomize_joints_enabled = bool(
            _cfg_get(self.cfg, "randomize_reset_joints", True)
        )
        self._reset_joint_range_scale = float(
            _cfg_get(self.cfg, "reset_joint_range_scale", 0.2)
        )
        if not self._deterministic_reset:
            return

        if self._reset_randomize_joints_enabled:
            _LOGGER.info(
                "deterministic_reset: overriding randomize_reset_joints=True -> False "
                "for deterministic playback/eval."
            )
        if self._reset_joint_range_scale != 0.0:
            _LOGGER.info(
                "deterministic_reset: overriding reset_joint_range_scale=%.6f -> 0.0 "
                "for deterministic playback/eval.",
                self._reset_joint_range_scale,
            )
        self._reset_randomize_joints_enabled = False
        self._reset_joint_range_scale = 0.0

    def _init_target_forward_velocity_behavior(self) -> None:
        self._fixed_target_forward_velocity = float(self.cfg.target_forward_velocity)
        mode = str(self.cfg.reward_mode).strip().lower()
        self._dynamic_target_forward_velocity_enabled = mode == "track_dynamic"
        if (
            self._dynamic_target_forward_velocity_enabled
            and not self._profile.supports_dynamic_target_forward_velocity
        ):
            raise ValueError(
                "reward_mode='track_dynamic' requires "
                "supports_dynamic_target_forward_velocity=True."
            )
        min_value = _cfg_get(self.cfg, "target_forward_velocity_min", None)
        max_value = _cfg_get(self.cfg, "target_forward_velocity_max", None)
        override_value = _cfg_get(self.cfg, "target_forward_velocity_override", None)

        self._target_forward_velocity_min = (
            self._fixed_target_forward_velocity
            if min_value is None
            else float(min_value)
        )
        self._target_forward_velocity_max = (
            self._fixed_target_forward_velocity
            if max_value is None
            else float(max_value)
        )
        if self._target_forward_velocity_max < self._target_forward_velocity_min:
            raise ValueError(
                "target_forward_velocity_max must be greater than or equal to "
                "target_forward_velocity_min."
            )
        self._has_target_forward_velocity_override = override_value is not None
        self._target_forward_velocity_override = (
            0.0 if override_value is None else float(override_value)
        )

    def _init_observation_behavior(self) -> None:
        self._include_target_forward_velocity_obs = bool(
            _cfg_get(self.cfg, "observation_include_target_forward_velocity", True)
        )
        self._include_velocity_error_obs = bool(
            _cfg_get(self.cfg, "observation_include_velocity_error", True)
        )

        if self._profile.forward_velocity_source == "qvel_root_x":
            self._forward_velocity_source_int = FORWARD_VELOCITY_SOURCE_QVEL_ROOT_X
        elif self._profile.forward_velocity_source == "sensor_subtreelinvel_x":
            self._forward_velocity_source_int = (
                FORWARD_VELOCITY_SOURCE_SENSOR_SUBTREELINVEL_X
            )
        else:
            raise ValueError(
                "Unsupported forward_velocity_source="
                f"{_cfg_get(self.cfg, 'forward_velocity_source', None)!r}"
            )

        if self._profile.termination_torso_tilt_mode == "xy_norm":
            self._termination_torso_tilt_mode_int = TORSO_TILT_MODE_XY_NORM
        elif self._profile.termination_torso_tilt_mode == "root_rot_y_abs":
            self._termination_torso_tilt_mode_int = TORSO_TILT_MODE_ROOT_ROT_Y_ABS
        else:
            raise ValueError(
                "Unsupported termination_torso_tilt_mode="
                f"{_cfg_get(self.cfg, 'termination_torso_tilt_mode', None)!r}"
            )

        if self._profile.torso_tilt_source == "torso_upright":
            self._torso_tilt_source_int = TORSO_TILT_SOURCE_BODY
        elif self._profile.torso_tilt_source == "root_tilt_xy":
            self._torso_tilt_source_int = TORSO_TILT_SOURCE_ROOT
        else:
            raise ValueError(
                "Unsupported torso_tilt_source="
                f"{_cfg_get(self.cfg, 'torso_tilt_source', None)!r}"
            )

    def __init__(
        self,
        cfg: FreemuscoRunEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        action = ActionSettings.from_env_cfg(cfg)
        self._action_mode = _validate_action_mode(action.type)
        set_runtime_action_space(cfg, action.space)
        self._uses_fast_muscle = self._action_mode != MUJOCO_ACTIVATION_MODE
        self._fast_muscle_settings = (
            FastMuscleControlSettings.from_validated(
                action=action,
                muscle=MuscleSettings.from_env_cfg(cfg),
                observations=FastMuscleObservationSettings.from_env_cfg(cfg),
                visualization=ActivationVisualizationSettings.from_env_cfg(cfg),
                character_render_mode=resolve_character_render_mode(cfg),
            )
            if self._uses_fast_muscle
            else None
        )
        self._run_runtime = FreemuscoRunRuntime()
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._profile = FreemuscoRunProfile.from_cfg(cfg)
        FreemuscoRunEnv._init_reset_behavior(self)
        FreemuscoRunEnv._init_target_forward_velocity_behavior(self)
        FreemuscoRunEnv._init_observation_behavior(self)
        self._init_model_indices()
        self._init_reset_and_reward_metadata()
        self._init_observation_buffers()
        if self._uses_fast_muscle:
            self._init_fast_muscle_runtime()
        else:
            self._include_muscle_observation_blocks = False
            self._include_heading_frame_body_state_obs = False
            self.finalize_derived_spaces(
                observation_spec=int(self._obs_dim),
                action_spec=self.cfg.action_space,
                observations=self.obs_buf,
            )
        self._kernel_bundle = build_freemusco_run_kernel_bundle(
            self._layout,
            uses_fast_muscle=self._uses_fast_muscle,
            include_muscle_observation_blocks=self._include_muscle_observation_blocks,
        )

    def _build_sim_spec(self, mjcf_path: str) -> mujoco.MjSpec:
        if not self._uses_fast_muscle:
            return super()._build_sim_spec(mjcf_path)
        self._fast_muscle_bootstrap = FastMuscleBootstrap.from_mjcf(
            mjcf_path,
            rest_keyframe_name=_cfg_get(self.cfg, "reset_keyframe_name", None),
        )
        self._source_mjcf_path = self._fast_muscle_bootstrap.source_mjcf_path
        self._parsed_mjcf_muscle_data = self._fast_muscle_bootstrap.parsed_mjcf_data
        return self._fast_muscle_bootstrap.sim_spec

    def _build_render_model(self):
        if not self._uses_fast_muscle:
            return super()._build_render_model()
        self._resolved_source_mjcf_path = str(self._source_mjcf_path)
        return build_plain_render_model(
            source_mjcf_path=self._resolved_source_mjcf_path,
            sim_model=self.model,
            resolution=tuple(self.cfg.viewer.resolution),
        )

    def _init_fast_muscle_runtime(self) -> None:
        settings = self._require_fast_muscle_settings()
        self.fast_muscle = FastMuscleController.attach_to_host(
            self,
            self.cfg,
            self._fast_muscle_bootstrap,
            settings=settings,
        )
        self._register_post_physics_substep_hook(self.fast_muscle.post_physics_substep)

        self.cfg.action_space = int(self.fast_muscle.policy_action_dim)
        # The base provisional builder may have allocated the authored action
        # width before the controller was attached. The controller owns this
        # execution buffer; derived-space finalization leaves it untouched.
        self.actions = torch.zeros(
            (self.num_envs, int(self.fast_muscle.policy_action_dim)),
            device=self.device,
            dtype=torch.float32,
        )
        self._include_muscle_observation_blocks = bool(
            _cfg_get(self.cfg, "include_muscle_observation_blocks", True)
        )
        self._include_heading_frame_body_state_obs = bool(
            _cfg_get(self.cfg, "include_heading_frame_body_state_obs", False)
        )

        muscle_count = int(self.fast_muscle.muscle_count)
        self._body_state_obs_dim = 0
        self._body_state_obs_ofs = 1 + 1 + 2 + (self._nq - 1) + self._nv
        if self._include_muscle_observation_blocks:
            self._body_state_obs_ofs += 4 * muscle_count
        self._forward_velocity_obs_ofs = int(self._body_state_obs_ofs)
        self._body_state_obs_ofs += 1
        self._target_forward_velocity_obs_ofs = None
        if self._include_target_forward_velocity_obs:
            self._target_forward_velocity_obs_ofs = int(self._body_state_obs_ofs)
            self._body_state_obs_ofs += 1
        if self._include_velocity_error_obs:
            self._body_state_obs_ofs += 1
        if self._include_heading_frame_body_state_obs:
            self._body_state_obs_dim = int(self.model.nbody) * 15
        self._obs_dim = int(self._body_state_obs_ofs + self._body_state_obs_dim)
        self._obs_main_dim = int(self._obs_dim)

        self._obs_suffix_dim = (
            self.fast_muscle.prev_action_obs_dim
            + self.fast_muscle.applied_torque_obs_dim
            + self.fast_muscle.muscle_geometry_observation_dim
        )
        self.fast_muscle.configure_observation_layout(self._obs_dim)
        total_obs_dim = int(self._obs_dim + self._obs_suffix_dim)
        self._obs_buf = torch.zeros(
            (self.num_envs, total_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.obs_buf = {"policy": self._obs_buf}
        self.finalize_derived_spaces(
            observation_spec=total_obs_dim,
            action_spec=self.cfg.action_space,
            observations=self.obs_buf,
        )
        self.fast_muscle.bind_execution(FastMuscleExecutionBindings.from_host(self))
        self.fast_muscle.finalize_runtime_state()
        apply_character_geom_render_visibility(self)
        self.fast_muscle.attach_render_assets(
            render_mode=self.render_mode,
            render_model=getattr(self, "render_model", self.model),
        )

    def _require_fast_muscle_settings(self) -> FastMuscleControlSettings:
        settings = self._fast_muscle_settings
        if settings is None:
            raise RuntimeError(
                "FastMuscle settings are unavailable for this action mode."
            )
        return settings

    def _init_model_indices(self) -> None:
        profile = self._profile
        self.root_body_id = resolve_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, profile.root_body_name
        )
        self.head_body_id = resolve_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, profile.head_body_name
        )
        self.foot_body_id_0 = resolve_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, profile.foot_body_names[0]
        )
        self.foot_body_id_1 = resolve_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, profile.foot_body_names[1]
        )
        self.pelvis_geom_id = resolve_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, profile.pelvis_geom_name
        )
        self.torso_body_id = (
            resolve_id(self.model, mujoco.mjtObj.mjOBJ_BODY, profile.torso_body_name)
            if profile.torso_body_name is not None
            else self.head_body_id
        )
        self._root_yaw_qpos_idx = qpos_index(self.model, profile.root_yaw_joint_name)

        self._root_tilt_qpos_idx_0 = -1
        self._root_tilt_qpos_idx_1 = -1
        if profile.root_tilt_joint_names:
            if len(profile.root_tilt_joint_names) != 2:
                raise ValueError(
                    "root_tilt_joint_names must contain exactly two joint names."
                )
            self._root_tilt_qpos_idx_0 = qpos_index(
                self.model, profile.root_tilt_joint_names[0]
            )
            self._root_tilt_qpos_idx_1 = qpos_index(
                self.model, profile.root_tilt_joint_names[1]
            )

        self._fwd_vel_qvel_idx = 0
        self._fwd_pos_qpos_idx = 0
        if profile.forward_joint_name is not None:
            self._fwd_vel_qvel_idx = dof_index(self.model, profile.forward_joint_name)
            self._fwd_pos_qpos_idx = qpos_index(self.model, profile.forward_joint_name)
        elif profile.qpos_observation_skip != "first":
            raise ValueError(
                "qpos_observation_skip='forward_joint' requires forward_joint_name."
            )

        self._forward_velocity_sensor_adr = -1
        if (
            self._forward_velocity_source_int
            == FORWARD_VELOCITY_SOURCE_SENSOR_SUBTREELINVEL_X
        ):
            sensor_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                profile.forward_velocity_sensor_name,
            )
            if sensor_id < 0:
                raise ValueError(
                    f"Missing sensor: {profile.forward_velocity_sensor_name}"
                )
            self._forward_velocity_sensor_adr = int(self.model.sensor_adr[sensor_id])

        self._nq = int(self.model.nq)
        self._nv = int(self.model.nv)
        self._na = int(self.model.na)
        self._nu = int(self.model.nu)
        self._qpos_observation_skip_idx = (
            0 if profile.qpos_observation_skip == "first" else self._fwd_pos_qpos_idx
        )
        self._layout = FreemuscoRunLayout(
            nq=self._nq,
            nv=self._nv,
            na=self._na,
            nu=self._nu,
            nbody=int(self.model.nbody),
            qpos_observation_skip_idx=self._qpos_observation_skip_idx,
            root_tilt_qpos_idx_0=self._root_tilt_qpos_idx_0,
            root_tilt_qpos_idx_1=self._root_tilt_qpos_idx_1,
        )
        self._layout.validate()

    def _init_reset_and_reward_metadata(self) -> None:
        reset_qpos = self.model.qpos0
        reset_keyframe_name = _cfg_get(self.cfg, "reset_keyframe_name", None)
        if reset_keyframe_name:
            key_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_KEY,
                str(reset_keyframe_name),
            )
            if key_id < 0:
                raise ValueError(f"Missing keyframe: {reset_keyframe_name}")
            reset_qpos = self.model.key_qpos[key_id]
        reset_qpos_t = torch.as_tensor(
            reset_qpos,
            dtype=torch.float32,
            device=self.device,
        )
        self._reset_qpos_wp = wp.from_torch(reset_qpos_t.contiguous())

        non_root_joint_qpos_idx_t = torch.as_tensor(
            self.model.jnt_qposadr[6:],
            dtype=torch.int32,
            device=self.device,
        )
        non_root_joint_low_t = (
            torch.as_tensor(
                self.model.jnt_range[6:, 0],
                dtype=torch.float32,
                device=self.device,
            )
            / 5.0
        )
        non_root_joint_high_t = (
            torch.as_tensor(
                self.model.jnt_range[6:, 1],
                dtype=torch.float32,
                device=self.device,
            )
            / 5.0
        )
        self._non_root_joint_qpos_idx_wp = wp.from_torch(
            non_root_joint_qpos_idx_t.contiguous()
        )
        self._non_root_joint_low_wp = wp.from_torch(non_root_joint_low_t.contiguous())
        self._non_root_joint_high_wp = wp.from_torch(non_root_joint_high_t.contiguous())
        self._num_non_root_joints = int(non_root_joint_qpos_idx_t.numel())
        (
            rest_body_pos_heading_t,
            rest_body_quat_heading_t,
        ) = self._build_heading_frame_rest_body_pose_tensors()
        self._rest_body_pos_heading_wp = wp.from_torch(
            rest_body_pos_heading_t.contiguous(),
            dtype=wp.vec3,
        )
        self._rest_body_quat_heading_wp = wp.from_torch(
            rest_body_quat_heading_t.contiguous(),
            dtype=wp.quat,
        )

        self._torso_upright_local = wp.vec3(0.0, 0.0, 1.0)
        if self._profile.torso_tilt_source == "torso_upright":
            torso_upright_local_t = self._build_torso_upright_local_axis(
                self.torso_body_id
            )
            self._torso_upright_local = wp.vec3(
                float(torso_upright_local_t[0].item()),
                float(torso_upright_local_t[1].item()),
                float(torso_upright_local_t[2].item()),
            )

        mode = str(self.cfg.reward_mode).strip().lower()
        if mode == "track":
            self._reward_mode_int = REWARD_MODE_TRACK
        elif mode == "track_dynamic":
            self._reward_mode_int = REWARD_MODE_TRACK_DYNAMIC
        elif mode == "raw_forward_velocity":
            self._reward_mode_int = REWARD_MODE_RAW_FORWARD_VELOCITY
        elif mode == "maximize":
            self._reward_mode_int = REWARD_MODE_MAXIMIZE
        else:
            raise ValueError(f"Unsupported reward_mode={self.cfg.reward_mode!r}")

        pose_mode = (
            str(_cfg_get(self.cfg, "reward_pose_regularization_mode", "joint_qpos"))
            .strip()
            .lower()
        )
        if pose_mode == "joint_qpos":
            self._reward_pose_regularization_mode_int = POSE_REG_MODE_JOINT_QPOS
        elif pose_mode == "heading_frame_body_pose":
            self._reward_pose_regularization_mode_int = (
                POSE_REG_MODE_HEADING_FRAME_BODY_POSE
            )
        else:
            raise ValueError(
                "Unsupported reward_pose_regularization_mode="
                f"{_cfg_get(self.cfg, 'reward_pose_regularization_mode', None)!r}"
            )

        self._reward_track_sigma = max(float(self.cfg.reward_track_sigma), 1.0e-6)
        self._reward_head_height_sigma = max(
            float(self.cfg.reward_head_height_sigma), 1.0e-6
        )
        self._init_reward_weight_buffer()

    def _init_reward_weight_buffer(self) -> None:
        """Stage the twelve reward weights on the device once.

        They are configuration, constant for the run, and used to be rebuilt as
        twelve Python floats on every reward evaluation.
        """
        weights = [0.0] * REWARD_WEIGHT_COUNT
        weights[REWARD_WEIGHT_TRACK_SIGMA] = self._reward_track_sigma
        weights[REWARD_WEIGHT_VELOCITY_SCALE] = float(self.cfg.reward_velocity_scale)
        weights[REWARD_WEIGHT_UPRIGHT_SCALE] = float(self.cfg.reward_upright_scale)
        weights[REWARD_WEIGHT_HEAD_HEIGHT_TARGET] = float(
            self.cfg.reward_head_height_target
        )
        weights[REWARD_WEIGHT_HEAD_HEIGHT_SIGMA] = self._reward_head_height_sigma
        weights[REWARD_WEIGHT_HEAD_HEIGHT_SCALE] = float(
            self.cfg.reward_head_height_scale
        )
        weights[REWARD_WEIGHT_DIRECTION_SCALE] = float(
            _cfg_get(self.cfg, "reward_direction_scale", 0.0)
        )
        weights[REWARD_WEIGHT_PELVIS_HEIGHT_THRESHOLD] = float(
            _cfg_get(self.cfg, "reward_pelvis_height_threshold", 0.6)
        )
        weights[REWARD_WEIGHT_PELVIS_HEIGHT_PENALTY_SCALE] = float(
            _cfg_get(self.cfg, "reward_pelvis_height_penalty_scale", 0.0)
        )
        weights[REWARD_WEIGHT_POSE_REGULARIZATION_SCALE] = float(
            _cfg_get(self.cfg, "reward_pose_regularization_scale", 0.02)
        )
        weights[REWARD_WEIGHT_POSE_BODY_POS_WEIGHT] = float(
            _cfg_get(self.cfg, "reward_pose_regularization_body_pos_weight", 1.0)
        )
        weights[REWARD_WEIGHT_POSE_BODY_ROT_WEIGHT] = float(
            _cfg_get(self.cfg, "reward_pose_regularization_body_rot_weight", 1.0)
        )
        self._reward_weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=self.device,
        )
        self._reward_weights_wp = wp.from_torch(
            self._reward_weights,
            dtype=wp.float32,
        )

    def _init_observation_buffers(self) -> None:
        self._obs_dim = (
            1
            + 1
            + 2
            + (self._nq - 1)
            + self._nv
            + self._na
            + self._nu
            + self._nu
            + self._nu
            + 1
        )
        self._forward_velocity_obs_ofs = int(self._obs_dim - 1)
        self._target_forward_velocity_obs_ofs: int | None = None
        if self._include_target_forward_velocity_obs:
            self._target_forward_velocity_obs_ofs = int(self._obs_dim)
            self._obs_dim += 1
        if self._include_velocity_error_obs:
            self._obs_dim += 1
        self._obs_buf = torch.zeros(
            (self.num_envs, int(self._obs_dim)),
            dtype=torch.float32,
            device=self.device,
        )
        self.obs_buf = {"policy": self._obs_buf}

        self.qpos_view = warp_to_torch(self.mjw_data.qpos)
        self.qvel_view = warp_to_torch(self.mjw_data.qvel)
        self.act_view = warp_to_torch(self.mjw_data.act)
        self.sensordata_view = warp_to_torch(self.mjw_data.sensordata)
        self.xpos_view = warp_to_torch(self.mjw_data.xpos)
        self.xquat_view = warp_to_torch(self.mjw_data.xquat)
        self.geom_xpos_view = warp_to_torch(self.mjw_data.geom_xpos)
        self.actuator_force_view = warp_to_torch(self.mjw_data.actuator_force)
        self.actuator_length_view = warp_to_torch(self.mjw_data.actuator_length)
        self.actuator_velocity_view = warp_to_torch(self.mjw_data.actuator_velocity)
        self._target_forward_velocity_obs = torch.full(
            (self.num_envs, 1),
            self._fixed_target_forward_velocity,
            dtype=torch.float32,
            device=self.device,
        )
        self._forward_velocity_obs = torch.zeros(
            (self.num_envs, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self._target_forward_velocity_obs_wp = wp.from_torch(
            self._target_forward_velocity_obs.contiguous()
        )
        self._forward_velocity_obs_wp = wp.from_torch(
            self._forward_velocity_obs.contiguous()
        )
        self._configure_done_reason_tracking(
            ("Head height", "Pelvis height", "Torso tilt", "Joint velocity")
        )

    def _resolve_observation_output(
        self,
        out_obs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if out_obs is not None:
            return out_obs, {"policy": out_obs}
        return self.obs_buf["policy"], self.obs_buf

    def _forward_velocity(self) -> torch.Tensor:
        if self._forward_velocity_source_int == FORWARD_VELOCITY_SOURCE_QVEL_ROOT_X:
            return self.qvel_view[
                :, self._fwd_vel_qvel_idx : self._fwd_vel_qvel_idx + 1
            ]
        adr = int(self._forward_velocity_sensor_adr)
        return self.sensordata_view[:, adr : adr + 1]

    def _update_forward_velocity_observation(self) -> None:
        self._forward_velocity_obs.copy_(self._forward_velocity())

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._uses_fast_muscle:
            self.fast_muscle.pre_physics_step(actions)
            return
        if actions.shape[-1] != int(self.cfg.action_space):
            raise ValueError(
                f"Expected action dimension {int(self.cfg.action_space)}, got {actions.shape[-1]}."
            )
        self.actions.copy_(actions)

    def _apply_action(self) -> None:
        if self._uses_fast_muscle:
            self.fast_muscle.apply_action()
            return
        self._run_runtime._apply_action(self)

    def _get_observations(self, out_obs: torch.Tensor | None = None) -> dict:
        return self._run_runtime._get_observations(self, out_obs)

    def _get_rewards(self) -> torch.Tensor:
        return self._run_runtime._get_rewards(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._run_runtime._get_dones(self)

    def _reset_idx(self, reset_mask: torch.Tensor):
        super()._reset_idx(reset_mask)
        mjw.reset_data(self.mjw_model, self.mjw_data, wp.from_torch(reset_mask))
        self._launch_reset_kernel(reset_mask)
        if self._uses_fast_muscle:
            self.fast_muscle.reset_idx(reset_mask)

    def _launch_reset_kernel(self, reset_mask: torch.Tensor) -> None:
        self._run_runtime._launch_reset_kernel(self, reset_mask)

    def _update_render_data(self):
        if self._uses_fast_muscle:
            self.fast_muscle.update_render_tendon_rgba()
            return None
        return super()._update_render_data()

    def configure_musclenn_rollout_storage(self, rollout_steps: int) -> None:
        if self._uses_fast_muscle:
            self.fast_muscle.configure_musclenn_rollout_storage(rollout_steps)

    def prepare_musclenn_rollout_sampling(self) -> None:
        if self._uses_fast_muscle:
            self.fast_muscle.prepare_musclenn_rollout_sampling()

    def set_musclenn_rollout_step(self, step: int) -> None:
        if self._uses_fast_muscle:
            self.fast_muscle.set_musclenn_rollout_step(step)

    def mark_musclenn_rollout_collected(self) -> None:
        if self._uses_fast_muscle:
            self.fast_muscle.mark_musclenn_rollout_collected()

    def optimize_muscle_nn(self) -> dict[str, float] | None:
        if not self._uses_fast_muscle:
            return None
        return self.fast_muscle.optimize_muscle_nn()

    def get_checkpoint_state(self) -> dict[str, object]:
        state = super().get_checkpoint_state()
        if self._uses_fast_muscle:
            state.update(self.fast_muscle.get_checkpoint_state())
        return state

    def load_checkpoint_state(self, state: dict[str, object] | None) -> bool:
        restored = bool(super().load_checkpoint_state(state))
        if not self._uses_fast_muscle:
            return restored
        return bool(self.fast_muscle.load_checkpoint_state(state)) or restored

    def _build_heading_frame_rest_body_pose_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        helper_data = mujoco.MjData(self.model)
        reset_keyframe_name = _cfg_get(self.cfg, "reset_keyframe_name", None)
        if reset_keyframe_name:
            key_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_KEY,
                str(reset_keyframe_name),
            )
            if key_id < 0:
                raise ValueError(f"Missing keyframe: {reset_keyframe_name}")
            helper_data.qpos[:] = self.model.key_qpos[key_id]
        else:
            helper_data.qpos[:] = self.model.qpos0
        helper_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, helper_data)

        body_pos = torch.as_tensor(
            helper_data.xpos,
            dtype=torch.float32,
            device=self.device,
        ).clone()
        body_quat = torch.as_tensor(
            helper_data.xquat,
            dtype=torch.float32,
            device=self.device,
        ).clone()

        body_quat = body_quat / torch.linalg.vector_norm(
            body_quat, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        body_quat = torch.where(body_quat[:, :1] < 0.0, -body_quat, body_quat)

        root_q = body_quat[self.root_body_id : self.root_body_id + 1]
        forward = torch.tensor(
            [[1.0, 0.0, 0.0]],
            dtype=body_quat.dtype,
            device=body_quat.device,
        )
        root_forward = torch_quaternion.quat_rotate(root_q, forward)
        root_yaw = torch.atan2(root_forward[:, 1], root_forward[:, 0])
        half_yaw = 0.5 * root_yaw
        yaw_quat = torch.stack(
            (
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ),
            dim=-1,
        )
        yaw_inv = torch_quaternion.quat_conj(yaw_quat)
        yaw_inv_expand = yaw_inv.expand(body_quat.shape[0], -1)
        root_pos = body_pos[self.root_body_id : self.root_body_id + 1].expand(
            body_pos.shape[0], -1
        )

        local_body_pos = torch_quaternion.quat_rotate(
            yaw_inv_expand, body_pos - root_pos
        )
        local_body_quat = torch_quaternion.quat_mul(yaw_inv_expand, body_quat)
        local_body_quat = local_body_quat / torch.linalg.vector_norm(
            local_body_quat, dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        local_body_quat = torch.where(
            local_body_quat[:, :1] < 0.0,
            -local_body_quat,
            local_body_quat,
        )
        return local_body_pos.contiguous(), local_body_quat.contiguous()

    def _build_torso_upright_local_axis(self, body_id: int) -> torch.Tensor:
        helper_data = mujoco.MjData(self.model)
        helper_data.qpos[:] = self.model.qpos0
        helper_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, helper_data)

        body_rot = torch.as_tensor(
            helper_data.xmat[body_id].reshape(3, 3),
            dtype=torch.float32,
            device=self.device,
        ).clone()
        world_up = torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=body_rot.dtype,
            device=body_rot.device,
        )
        local_axis = torch.matmul(body_rot.transpose(0, 1), world_up)
        return local_axis / torch.linalg.vector_norm(local_axis).clamp_min(1.0e-8)
