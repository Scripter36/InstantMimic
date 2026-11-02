"""Readable public BeyondMimic Stage-1 reference environment."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from instantmimic.envs.config_access import get_value
from instantmimic.envs.direct_rl_env import DirectRLEnv

from . import optimized_runtime
from .beyondmimic_tracking_env_cfg import BeyondMimicTrackingEnvCfg
from .constants import (
    ACTUATOR_NAMES,
    ANCHOR_BODY_NAME,
    END_EFFECTOR_BODY_NAMES,
    FOOT_GEOM_NAMES,
    JOINT_NAMES,
    OBSERVATION_NOISE_RANGES,
    REWARD_TERM_NAMES,
    ROOT_BODY_NAME,
    TRACKED_BODY_NAMES,
)
from .metrics import BenchmarkReference, BenchmarkRobotState
from .model_randomization import expand_startup_randomization_fields
from .motion import load_motion_tensors
from .reference import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
    AdaptiveSamplerState,
    ObservationNoise,
    assemble_actor_observation,
    assemble_critic_observation,
    compute_reward,
    compute_terminations,
    default_relative_action_targets,
    failure_bins,
    motion_relative_to_robot_anchor,
    quat_apply_inverse,
    quat_inverse,
    quat_multiply,
    quaternion_to_matrix,
    sampling_probabilities,
    update_failure_ema,
)

_OBSERVATION_NOISE_TERMS = (
    ("anchor_position", 3),
    ("anchor_orientation", 6),
    ("base_linear_velocity", 3),
    ("base_angular_velocity", 3),
    ("joint_position", ACTION_DIM),
    ("joint_velocity", ACTION_DIM),
)


@wp.kernel(enable_backward=False)
def _observation_noise_kernel(
    seed: int,
    common_step_counter: wp.array(dtype=wp.int32),
    half_range: wp.array(dtype=wp.float32),
    noise: wp.array2d(dtype=wp.float32),
):
    """Uniform observation noise keyed by (seed, step, env, column).

    A counter-based draw makes the reference observation a pure function of
    the step: the terminal capture and the policy input see the same noise.
    """
    env_idx, column = wp.tid()
    state = wp.rand_init(
        seed + common_step_counter[0] * 17 + 1009,
        env_idx * noise.shape[1] + column,
    )
    bound = half_range[column]
    noise[env_idx, column] = bound * (2.0 * wp.randf(state) - 1.0)


def _uniform(
    shape: tuple[int, ...],
    bounds: tuple[float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    lower, upper = (float(value) for value in bounds)
    return torch.empty(shape, device=device, dtype=dtype).uniform_(lower, upper)


def _qualified(name: str) -> str:
    return name if name.startswith("robot/") else f"robot/{name}"


class BeyondMimicTrackingEnv(DirectRLEnv):
    """Torch reference path matching the pinned mjlab manager ordering."""

    cfg: BeyondMimicTrackingEnvCfg
    optimized_runtime_state: optimized_runtime.BeyondMimicOptimizedRuntimeState | None
    # Task-wide first-reset transition shared by ordinary reset and optimized
    # reset finalization; it is intentionally Env-owned rather than optional
    # optimized runtime state.
    _initial_reset_complete: bool
    parallelize_dones_and_rewards: ClassVar[bool] = False
    requires_separate_ppo_grad_norm_clip: ClassVar[bool] = True

    def get_asymmetric_observation_dims(self) -> tuple[int, int]:
        return ACTOR_OBSERVATION_DIM, CRITIC_OBSERVATION_DIM

    def __init__(
        self,
        cfg: BeyondMimicTrackingEnvCfg,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.optimized_runtime_state = None
        self._optimized_task_path = str(cfg.execution_mode) == "optimized"
        if str(cfg.execution_mode) not in {"reference", "optimized"}:
            raise ValueError(
                "env.execution_mode must be either 'reference' or 'optimized'"
            )
        if self._optimized_task_path:
            required_flags = {
                "use_cuda_graph": cfg.use_cuda_graph,
                "use_parallelized_kernel_launch": (cfg.use_parallelized_kernel_launch),
                "use_optimized_kernel": cfg.use_optimized_kernel,
            }
            disabled = [name for name, enabled in required_flags.items() if not enabled]
            if disabled:
                raise ValueError(
                    "The optimized BeyondMimic execution mode requires: "
                    + ", ".join(disabled)
                )
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        if (
            int(self.cfg.observation_space) != ACTOR_OBSERVATION_DIM
            or int(self.cfg.state_space) != CRITIC_OBSERVATION_DIM
            or int(self.cfg.action_space) != ACTION_DIM
        ):
            raise ValueError(
                "BeyondMimic G1 spaces must be policy=160, critic=286, action=29"
            )
        if not math.isclose(self.physics_dt, 0.005, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Official physics dt is 0.005, got {self.physics_dt}")
        if int(self.cfg.decimation) != 4:
            raise ValueError(f"Official decimation is 4, got {self.cfg.decimation}")

        self._initialize_views()
        self._initialize_model_indices()
        self._initialize_model_contract()
        self._motion = load_motion_tensors(
            str(self.cfg.motion.file), device=self.device
        )
        self._initialize_task_buffers()
        self._apply_startup_randomization()
        if self._optimized_task_path:
            optimized_runtime.initialize(self)

    def _can_parallelize_dones_and_rewards(self) -> bool:
        return self._optimized_task_path

    def _initialize_views(self) -> None:
        self._qpos = wp.to_torch(self.mjw_data.qpos)
        self._qvel = wp.to_torch(self.mjw_data.qvel)
        self._ctrl = wp.to_torch(self.mjw_data.ctrl)
        self._xpos = wp.to_torch(self.mjw_data.xpos)
        self._xquat = wp.to_torch(self.mjw_data.xquat)
        self._cvel = wp.to_torch(self.mjw_data.cvel)
        self._subtree_com = wp.to_torch(self.mjw_data.subtree_com)
        self._sensordata = wp.to_torch(self.mjw_data.sensordata)

    def _object_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Canonical G1 model is missing {name!r}")
        return int(object_id)

    def _initialize_model_indices(self) -> None:
        self._joint_ids = torch.tensor(
            [
                self._object_id(mujoco.mjtObj.mjOBJ_JOINT, _qualified(name))
                for name in JOINT_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._joint_qpos_indices = torch.as_tensor(
            self.model.jnt_qposadr[self._joint_ids.cpu().numpy()],
            dtype=torch.long,
            device=self.device,
        )
        self._joint_qvel_indices = torch.as_tensor(
            self.model.jnt_dofadr[self._joint_ids.cpu().numpy()],
            dtype=torch.long,
            device=self.device,
        )
        self._body_ids = torch.tensor(
            [
                self._object_id(mujoco.mjtObj.mjOBJ_BODY, _qualified(name))
                for name in TRACKED_BODY_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._anchor_body_id = self._object_id(
            mujoco.mjtObj.mjOBJ_BODY, _qualified(ANCHOR_BODY_NAME)
        )
        self._root_body_id = self._object_id(
            mujoco.mjtObj.mjOBJ_BODY, _qualified(ROOT_BODY_NAME)
        )
        self._end_effector_body_ids = torch.tensor(
            [
                self._object_id(mujoco.mjtObj.mjOBJ_BODY, _qualified(name))
                for name in END_EFFECTOR_BODY_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )
        robot_body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for body_id in range(self.model.nbody)
            if (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            ).startswith("robot/")
        ]
        motion_body_index = {
            name.removeprefix("robot/"): index
            for index, name in enumerate(robot_body_names)
        }
        self._motion_body_indices = torch.tensor(
            [motion_body_index[name] for name in TRACKED_BODY_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self._motion_anchor_body_index = motion_body_index[ANCHOR_BODY_NAME]
        self._motion_root_body_index = motion_body_index[ROOT_BODY_NAME]

        actuator_ids = [
            self._object_id(mujoco.mjtObj.mjOBJ_ACTUATOR, _qualified(name))
            for name in ACTUATOR_NAMES
        ]
        if actuator_ids != list(range(ACTION_DIM)):
            raise ValueError(
                "Canonical G1 actuator ordering changed; regenerate parity evidence"
            )
        joint_id_to_order = {
            int(joint_id): order
            for order, joint_id in enumerate(self._joint_ids.tolist())
        }
        self._actuator_to_joint_order = torch.tensor(
            [
                joint_id_to_order[int(self.model.actuator_trnid[actuator_id, 0])]
                for actuator_id in actuator_ids
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._foot_geom_ids = torch.tensor(
            [
                self._object_id(mujoco.mjtObj.mjOBJ_GEOM, _qualified(name))
                for name in FOOT_GEOM_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._imu_linear_slice = self._sensor_slice("robot/imu_lin_vel", 3)
        self._imu_angular_slice = self._sensor_slice("robot/imu_ang_vel", 3)
        self._self_collision_force_slice = self._sensor_slice(
            "self_collision_pelvis_force", 3
        )

    def _sensor_slice(self, name: str, expected_dimension: int) -> slice:
        sensor_id = self._object_id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        dimension = int(self.model.sensor_dim[sensor_id])
        if dimension != expected_dimension:
            raise ValueError(
                f"Sensor {name!r} dimension is {dimension}, expected {expected_dimension}"
            )
        address = int(self.model.sensor_adr[sensor_id])
        return slice(address, address + dimension)

    def _initialize_model_contract(self) -> None:
        key_id = self._object_id(mujoco.mjtObj.mjOBJ_KEY, "init_state")
        key_qpos = torch.as_tensor(
            self.model.key_qpos[key_id],
            device=self.device,
            dtype=torch.float32,
        )
        self._default_joint_position = key_qpos[self._joint_qpos_indices].contiguous()
        self._default_joint_velocity = torch.zeros_like(self._default_joint_position)
        stiffness = torch.as_tensor(
            self.model.actuator_gainprm[:, 0],
            device=self.device,
            dtype=torch.float32,
        )
        effort = torch.as_tensor(
            self.model.actuator_forcerange[:, 1],
            device=self.device,
            dtype=torch.float32,
        )
        actuator_action_scale = 0.25 * effort / stiffness
        self._action_scale = torch.empty_like(actuator_action_scale)
        self._action_scale[self._actuator_to_joint_order] = actuator_action_scale
        joint_range = torch.as_tensor(
            self.model.jnt_range[self._joint_ids.cpu().numpy()],
            device=self.device,
            dtype=torch.float32,
        )
        center = 0.5 * (joint_range[:, 0] + joint_range[:, 1])
        extent = joint_range[:, 1] - joint_range[:, 0]
        self._soft_joint_lower = (center - 0.45 * extent).contiguous()
        self._soft_joint_upper = (center + 0.45 * extent).contiguous()

    def _initialize_task_buffers(self) -> None:
        shape = (self.num_envs,)
        self.actions = torch.zeros((*shape, ACTION_DIM), device=self.device)
        self._previous_actions = torch.zeros_like(self.actions)
        self._target_joint_position = torch.zeros_like(self.actions)
        self._target_ctrl = torch.zeros_like(self.actions)
        self._encoder_bias = torch.zeros_like(self.actions)
        self._time_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self._adaptive = AdaptiveSamplerState.create(
            motion_frame_count=self._motion.frame_count,
            sampling_fps=1.0 / self.step_dt,
            device=self.device,
            dtype=torch.float32,
        )
        if bool(get_value(self.cfg.randomization, "enabled", True)) and bool(
            get_value(self.cfg.randomization, "push_enabled", True)
        ):
            self._push_time_left = _uniform(
                shape,
                tuple(self.cfg.randomization.push_interval_range_s),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self._push_time_left = torch.zeros(shape, device=self.device)
        self._collision_force_history = torch.zeros(
            (self.num_envs, 1, self.cfg.decimation, 3), device=self.device
        )
        self._target_body_position = torch.zeros(
            (self.num_envs, len(TRACKED_BODY_NAMES), 3), device=self.device
        )
        self._target_body_quaternion = torch.zeros(
            (self.num_envs, len(TRACKED_BODY_NAMES), 4), device=self.device
        )
        self._target_body_quaternion[..., 0] = 1.0
        self._actor_observation = torch.zeros(
            (self.num_envs, ACTOR_OBSERVATION_DIM), device=self.device
        )
        self._critic_observation = torch.zeros(
            (self.num_envs, CRITIC_OBSERVATION_DIM), device=self.device
        )
        self.obs_buf = {
            "policy": self._actor_observation,
            "critic": self._critic_observation,
        }
        self._reward_terms = torch.zeros(
            (self.num_envs, len(REWARD_TERM_NAMES)), device=self.device
        )
        self._configure_done_reason_tracking(
            ("anchor_height", "anchor_orientation", "end_effector_height")
        )
        self._initial_reset_complete = False

    def _apply_startup_randomization(self) -> None:
        randomization = self.cfg.randomization
        if not bool(get_value(randomization, "enabled", True)):
            return
        expand_startup_randomization_fields(self.mjw_model, num_worlds=self.num_envs)
        friction = wp.to_torch(self.mjw_model.geom_friction)
        friction_sample = _uniform(
            (self.num_envs, 1),
            tuple(randomization.foot_friction_range),
            device=self.device,
            dtype=friction.dtype,
        )
        friction[:, self._foot_geom_ids, 0] = friction_sample
        body_ipos = wp.to_torch(self.mjw_model.body_ipos)
        offsets = torch.cat(
            [
                _uniform(
                    (self.num_envs, 1),
                    tuple(bounds),
                    device=self.device,
                    dtype=body_ipos.dtype,
                )
                for bounds in randomization.torso_com_offset_range
            ],
            dim=-1,
        )
        body_ipos[:, self._anchor_body_id] += offsets
        self._encoder_bias.uniform_(*randomization.encoder_bias_range)
        mjw.set_const(self.mjw_model, self.mjw_data)

    def _current_motion(self, value: torch.Tensor) -> torch.Tensor:
        return value[self._time_steps]

    def _robot_body_velocities(self) -> tuple[torch.Tensor, torch.Tensor]:
        position = self._xpos[:, self._body_ids]
        subtree_com = self._subtree_com[:, self._root_body_id, None, :]
        cvel = self._cvel[:, self._body_ids]
        angular = cvel[..., :3]
        linear = cvel[..., 3:6] - torch.cross(angular, subtree_com - position, dim=-1)
        return linear, angular

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if tuple(actions.shape) != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"Expected actions {(self.num_envs, ACTION_DIM)}, got {tuple(actions.shape)}"
            )
        if self._optimized_task_path:
            optimized_runtime.pre_physics_step(self, actions)
            return
        self._previous_actions.copy_(self.actions)
        self.actions.copy_(actions)
        self._target_joint_position.copy_(
            default_relative_action_targets(
                self.actions,
                self._action_scale,
                self._default_joint_position,
                self._encoder_bias,
            )
        )
        self._target_ctrl.copy_(
            self._target_joint_position[:, self._actuator_to_joint_order]
        )

    def _apply_action(self) -> None:
        if self._optimized_task_path:
            return
        self._ctrl.copy_(self._target_ctrl)

    def _post_physics_substep(self, substep_idx: int) -> None:
        if self._optimized_task_path:
            optimized_runtime.capture_collision_force(self, substep_idx)
            return
        self._collision_force_history[:, 0, substep_idx].copy_(
            self._sensordata[:, self._self_collision_force_slice]
        )

    def _anchor_observation_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot_position = self._xpos[:, self._anchor_body_id]
        robot_quaternion = self._xquat[:, self._anchor_body_id]
        reference_position = self._current_motion(self._motion.body_position_world)[
            :, self._motion_anchor_body_index
        ]
        reference_quaternion = self._current_motion(self._motion.body_quaternion_world)[
            :, self._motion_anchor_body_index
        ]
        position = quat_apply_inverse(
            robot_quaternion, reference_position - robot_position
        )
        orientation = quat_multiply(
            quat_inverse(robot_quaternion), reference_quaternion
        )
        orientation_6d = quaternion_to_matrix(orientation)[..., :2].reshape(
            self.num_envs, 6
        )
        return position, orientation_6d

    def _robot_body_observation_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        anchor_position = self._xpos[:, self._anchor_body_id, None, :]
        anchor_quaternion = self._xquat[:, self._anchor_body_id, None, :]
        position = quat_apply_inverse(
            anchor_quaternion, self._xpos[:, self._body_ids] - anchor_position
        ).reshape(self.num_envs, -1)
        orientation = quat_multiply(
            quat_inverse(anchor_quaternion), self._xquat[:, self._body_ids]
        )
        orientation_6d = quaternion_to_matrix(orientation)[..., :2].reshape(
            self.num_envs, -1
        )
        return position, orientation_6d

    def _observation_noise(self) -> ObservationNoise | None:
        if not bool(get_value(self.cfg.observations, "corruption", True)):
            return None
        buffers = getattr(self, "_observation_noise_buffers", None)
        if buffers is None:
            half_range = torch.cat(
                [
                    torch.full((width,), OBSERVATION_NOISE_RANGES[name])
                    for name, width in _OBSERVATION_NOISE_TERMS
                ]
            ).to(device=self.device, dtype=torch.float32)
            noise = torch.empty(
                (self.num_envs, int(half_range.numel())),
                device=self.device,
                dtype=torch.float32,
            )
            buffers = (half_range, noise)
            self._observation_noise_buffers = buffers
        half_range, noise = buffers
        wp.launch(
            _observation_noise_kernel,
            dim=tuple(noise.shape),
            inputs=[
                0 if self.cfg.seed is None else int(self.cfg.seed),
                wp.from_torch(self.common_step_counter, dtype=wp.int32),
                wp.from_torch(half_range, dtype=wp.float32),
            ],
            outputs=[wp.from_torch(noise, dtype=wp.float32)],
            device=str(noise.device),
        )
        columns: dict[str, torch.Tensor] = {}
        offset = 0
        for name, width in _OBSERVATION_NOISE_TERMS:
            columns[name] = noise[:, offset : offset + width]
            offset += width
        return ObservationNoise(**columns)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self._optimized_task_path:
            return optimized_runtime.get_observations(self)
        anchor_position, anchor_orientation = self._anchor_observation_terms()
        body_position, body_orientation = self._robot_body_observation_terms()
        command = torch.cat(
            (
                self._current_motion(self._motion.joint_position),
                self._current_motion(self._motion.joint_velocity),
            ),
            dim=-1,
        )
        joint_position = self._qpos[:, self._joint_qpos_indices]
        joint_velocity = self._qvel[:, self._joint_qvel_indices]
        actor = assemble_actor_observation(
            command=command,
            anchor_position=anchor_position,
            anchor_orientation=anchor_orientation,
            base_linear_velocity=self._sensordata[:, self._imu_linear_slice],
            base_angular_velocity=self._sensordata[:, self._imu_angular_slice],
            joint_position=joint_position + self._encoder_bias,
            default_joint_position=self._default_joint_position.expand(
                self.num_envs, -1
            ),
            joint_velocity=joint_velocity,
            default_joint_velocity=self._default_joint_velocity.expand(
                self.num_envs, -1
            ),
            action=self.actions,
            noise=self._observation_noise(),
        )
        critic = assemble_critic_observation(
            command=command,
            anchor_position=anchor_position,
            anchor_orientation=anchor_orientation,
            body_position=body_position,
            body_orientation=body_orientation,
            base_linear_velocity=self._sensordata[:, self._imu_linear_slice],
            base_angular_velocity=self._sensordata[:, self._imu_angular_slice],
            joint_position=joint_position,
            default_joint_position=self._default_joint_position.expand(
                self.num_envs, -1
            ),
            joint_velocity=joint_velocity,
            default_joint_velocity=self._default_joint_velocity.expand(
                self.num_envs, -1
            ),
            action=self.actions,
        )
        self.obs_buf["policy"].copy_(actor)
        self.obs_buf["critic"].copy_(critic)
        return self.obs_buf

    def _get_rewards(self) -> torch.Tensor:
        if self._optimized_task_path:
            return optimized_runtime.get_rewards(self)
        robot_linear_velocity, robot_angular_velocity = self._robot_body_velocities()
        motion_body_position = self._current_motion(self._motion.body_position_world)[
            :, self._motion_body_indices
        ]
        motion_body_quaternion = self._current_motion(
            self._motion.body_quaternion_world
        )[:, self._motion_body_indices]
        result = compute_reward(
            robot_anchor_position=self._xpos[:, self._anchor_body_id],
            target_anchor_position=self._current_motion(
                self._motion.body_position_world
            )[:, self._motion_anchor_body_index],
            robot_anchor_quaternion=self._xquat[:, self._anchor_body_id],
            target_anchor_quaternion=self._current_motion(
                self._motion.body_quaternion_world
            )[:, self._motion_anchor_body_index],
            robot_body_position=self._xpos[:, self._body_ids],
            target_body_position=self._target_body_position,
            robot_body_quaternion=self._xquat[:, self._body_ids],
            target_body_quaternion=self._target_body_quaternion,
            robot_body_linear_velocity=robot_linear_velocity,
            target_body_linear_velocity=self._current_motion(
                self._motion.body_linear_velocity_world
            )[:, self._motion_body_indices],
            robot_body_angular_velocity=robot_angular_velocity,
            target_body_angular_velocity=self._current_motion(
                self._motion.body_angular_velocity_world
            )[:, self._motion_body_indices],
            action=self.actions,
            previous_action=self._previous_actions,
            joint_position=self._qpos[:, self._joint_qpos_indices],
            soft_joint_lower=self._soft_joint_lower,
            soft_joint_upper=self._soft_joint_upper,
            self_collision_force_history=self._collision_force_history,
            step_duration=self.step_dt,
        )
        del motion_body_position, motion_body_quaternion
        for index, name in enumerate(REWARD_TERM_NAMES):
            self._reward_terms[:, index].copy_(result.contributions[name])
        self.reward_buf.copy_(result.total)
        return self.reward_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._optimized_task_path:
            return optimized_runtime.get_dones(self)
        reference_anchor_quaternion = self._current_motion(
            self._motion.body_quaternion_world
        )[:, self._motion_anchor_body_index]
        gravity = self._xpos.new_tensor((0.0, 0.0, -1.0)).expand(self.num_envs, -1)
        result = compute_terminations(
            elapsed_steps=self.episode_length_buf,
            max_episode_steps=self.max_episode_length,
            robot_anchor_height=self._xpos[:, self._anchor_body_id, 2],
            target_anchor_height=self._current_motion(self._motion.body_position_world)[
                :, self._motion_anchor_body_index, 2
            ],
            robot_projected_gravity_z=quat_apply_inverse(
                self._xquat[:, self._anchor_body_id], gravity
            )[:, 2],
            target_projected_gravity_z=quat_apply_inverse(
                reference_anchor_quaternion, gravity
            )[:, 2],
            robot_end_effector_height=self._xpos[:, self._end_effector_body_ids, 2],
            target_end_effector_height=self._target_body_position[
                :,
                [TRACKED_BODY_NAMES.index(name) for name in END_EFFECTOR_BODY_NAMES],
                2,
            ],
        )
        self.reset_terminated.copy_(result.terminated)
        self.reset_time_outs.copy_(result.timeout)
        if self._track_done_reasons:
            self._done_reason_mask_buf[0].copy_(result.anchor_height)
            self._done_reason_mask_buf[1].copy_(result.anchor_orientation)
            self._done_reason_mask_buf[2].copy_(result.end_effector_height)
        return self.reset_terminated, self.reset_time_outs

    def _sample_motion_frames(self, env_ids: torch.Tensor) -> None:
        mode = str(self.cfg.motion.sampling_mode)
        if mode == "start":
            self._time_steps[env_ids] = 0
            return
        if mode == "uniform":
            self._time_steps[env_ids] = torch.randint(
                0,
                self._motion.frame_count,
                (len(env_ids),),
                device=self.device,
            )
            return
        if mode != "adaptive":
            raise ValueError(f"Unsupported motion sampling mode {mode!r}")
        failed = self.reset_terminated[env_ids]
        if torch.any(failed):
            bins = failure_bins(
                self._time_steps,
                self._motion.frame_count,
                self._adaptive.bin_count,
            )
            self._adaptive.current_bin_failed.copy_(
                torch.bincount(
                    bins[env_ids][failed], minlength=self._adaptive.bin_count
                ).to(torch.float32)
            )
        probabilities = sampling_probabilities(
            self._adaptive.bin_failed_count,
            uniform_ratio=float(self.cfg.motion.adaptive_uniform_ratio),
            kernel_size=int(self.cfg.motion.adaptive_kernel_size),
            adaptive_lambda=float(self.cfg.motion.adaptive_lambda),
        )
        sampled_bins = torch.multinomial(probabilities, len(env_ids), replacement=True)
        within_bin = torch.rand(len(env_ids), device=self.device)
        self._time_steps[env_ids] = (
            (sampled_bins + within_bin)
            / self._adaptive.bin_count
            * (self._motion.frame_count - 1)
        ).to(torch.long)

    def _write_reference_state(self, env_ids: torch.Tensor) -> None:
        frames = self._time_steps[env_ids]
        reset_cfg = self.cfg.reset
        position_delta = torch.cat(
            [
                _uniform(
                    (len(env_ids), 1),
                    tuple(bounds),
                    device=self.device,
                    dtype=torch.float32,
                )
                for bounds in reset_cfg.position_range
            ],
            dim=-1,
        )
        euler_delta = torch.cat(
            [
                _uniform(
                    (len(env_ids), 1),
                    tuple(bounds),
                    device=self.device,
                    dtype=torch.float32,
                )
                for bounds in reset_cfg.euler_range
            ],
            dim=-1,
        )
        velocity_delta = torch.cat(
            [
                _uniform(
                    (len(env_ids), 1),
                    tuple(bounds),
                    device=self.device,
                    dtype=torch.float32,
                )
                for bounds in reset_cfg.velocity_range
            ],
            dim=-1,
        )
        joint_delta = _uniform(
            (len(env_ids), ACTION_DIM),
            tuple(reset_cfg.joint_position_range),
            device=self.device,
            dtype=torch.float32,
        )
        from .reference import perturb_reference_state

        state = perturb_reference_state(
            reference_root_position=self._motion.body_position_world[
                frames, self._motion_root_body_index
            ],
            reference_root_quaternion=self._motion.body_quaternion_world[
                frames, self._motion_root_body_index
            ],
            reference_root_linear_velocity=self._motion.body_linear_velocity_world[
                frames, self._motion_root_body_index
            ],
            reference_root_angular_velocity=self._motion.body_angular_velocity_world[
                frames, self._motion_root_body_index
            ],
            reference_joint_position=self._motion.joint_position[frames],
            reference_joint_velocity=self._motion.joint_velocity[frames],
            position_delta=position_delta,
            euler_delta=euler_delta,
            root_velocity_delta=velocity_delta,
            joint_position_delta=joint_delta,
            joint_lower=self._soft_joint_lower,
            joint_upper=self._soft_joint_upper,
        )
        self._qpos[env_ids, :3] = state.root_position
        self._qpos[env_ids, 3:7] = state.root_quaternion
        self._qpos[env_ids[:, None], self._joint_qpos_indices] = state.joint_position
        self._qvel[env_ids, :3] = state.root_linear_velocity_world
        self._qvel[env_ids, 3:6] = state.root_angular_velocity_body
        self._qvel[env_ids[:, None], self._joint_qvel_indices] = state.joint_velocity

    def _resample_and_write(self, env_ids: torch.Tensor, *, reset_data: bool) -> None:
        if len(env_ids) == 0:
            return
        if reset_data:
            reset_mask = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            reset_mask[env_ids] = True
            mjw.reset_data(
                self.mjw_model,
                self.mjw_data,
                wp.from_torch(reset_mask, dtype=wp.bool),
            )
        self._sample_motion_frames(env_ids)
        self._write_reference_state(env_ids)

    def _update_command(self) -> None:
        self._time_steps.add_(1)
        motion_end_ids = torch.where(self._time_steps >= self._motion.frame_count)[0]
        if len(motion_end_ids) > 0:
            self._resample_and_write(motion_end_ids, reset_data=False)
            # Not a reset path, so the base's post-reset state prefix does not
            # cover this: a motion ending mid-episode rewrites reference state
            # that the command terms below read straight away.
            self._run_mjw_forward()
        reference_position = self._current_motion(self._motion.body_position_world)
        reference_quaternion = self._current_motion(self._motion.body_quaternion_world)
        target_position, target_quaternion = motion_relative_to_robot_anchor(
            robot_anchor_position=self._xpos[:, self._anchor_body_id],
            robot_anchor_quaternion=self._xquat[:, self._anchor_body_id],
            reference_anchor_position=reference_position[
                :, self._motion_anchor_body_index
            ],
            reference_anchor_quaternion=reference_quaternion[
                :, self._motion_anchor_body_index
            ],
            reference_body_position=reference_position[:, self._motion_body_indices],
            reference_body_quaternion=reference_quaternion[
                :, self._motion_body_indices
            ],
        )
        self._target_body_position.copy_(target_position)
        self._target_body_quaternion.copy_(target_quaternion)
        if str(self.cfg.motion.sampling_mode) == "adaptive":
            self._adaptive.bin_failed_count.copy_(
                update_failure_ema(
                    self._adaptive.bin_failed_count,
                    self._adaptive.current_bin_failed,
                    alpha=float(self.cfg.motion.adaptive_alpha),
                )
            )
            self._adaptive.current_bin_failed.zero_()

    def _apply_interval_pushes(self) -> None:
        randomization = self.cfg.randomization
        if not bool(get_value(randomization, "enabled", True)) or not bool(
            get_value(randomization, "push_enabled", True)
        ):
            return
        self._push_time_left.sub_(self.step_dt)
        env_ids = torch.where(self._push_time_left < 1e-6)[0]
        if len(env_ids) == 0:
            return
        self._push_time_left[env_ids] = _uniform(
            (len(env_ids),),
            tuple(randomization.push_interval_range_s),
            device=self.device,
            dtype=torch.float32,
        )
        root_position = self._xpos[env_ids, self._root_body_id]
        root_subtree_com = self._subtree_com[env_ids, self._root_body_id]
        cvel = self._cvel[env_ids, self._root_body_id]
        angular_world = cvel[:, :3]
        linear_world = cvel[:, 3:6] - torch.cross(
            angular_world, root_subtree_com - root_position, dim=-1
        )
        velocity_delta = torch.cat(
            [
                _uniform(
                    (len(env_ids), 1),
                    tuple(bounds),
                    device=self.device,
                    dtype=torch.float32,
                )
                for bounds in randomization.push_velocity_range
            ],
            dim=-1,
        )
        linear_world += velocity_delta[:, :3]
        angular_world += velocity_delta[:, 3:]
        self._qvel[env_ids, :3] = linear_world
        self._qvel[env_ids, 3:6] = quat_apply_inverse(
            self._qpos[env_ids, 3:7], angular_world
        )

    def _reset_idx(self, reset_mask: torch.Tensor) -> None:
        if self._optimized_task_path:
            optimized_runtime.reset_idx(self, reset_mask)
            return
        env_ids = torch.where(reset_mask)[0]
        super()._reset_idx(reset_mask)
        if len(env_ids) > 0:
            self._resample_and_write(env_ids, reset_data=True)
            self.actions[env_ids] = 0.0
            self._previous_actions[env_ids] = 0.0
            self._target_joint_position[env_ids] = 0.0
            self._target_ctrl[env_ids] = 0.0
            self._collision_force_history[env_ids] = 0.0
        # Kept deliberately. The base runs `_run_mjw_state_prefix()` after
        # `_reset_idx` returns, which is too late for `_update_command()` on the
        # next line: it reads `self._xpos`/`self._xquat` for the anchor body, so
        # the derived state has to be current before this hook finishes.
        self._run_mjw_forward()
        self._update_command()
        if self._initial_reset_complete:
            self._apply_interval_pushes()
        else:
            self._initial_reset_complete = True

    def _post_reset_state_prefix(self, reset_mask: torch.Tensor) -> bool:
        del reset_mask
        if self._optimized_task_path:
            return optimized_runtime.finalize_reset_state(self)
        return False

    def get_reference_diagnostics(self) -> dict[str, torch.Tensor]:
        return {
            name: self._reward_terms[:, index]
            for index, name in enumerate(REWARD_TERM_NAMES)
        }

    @property
    def benchmark_end_effector_indices(self) -> tuple[int, ...]:
        """End-effector indices in the pinned tracked-body ordering."""
        return tuple(TRACKED_BODY_NAMES.index(name) for name in END_EFFECTOR_BODY_NAMES)

    def snapshot_benchmark_reference(self) -> BenchmarkReference:
        """Snapshot the current reference before the upcoming scored step."""
        frames = self._time_steps
        return BenchmarkReference(
            body_position_world=self._motion.body_position_world[
                frames[:, None],
                self._motion_body_indices,
            ].clone(),
            body_position_relative_world=self._target_body_position.clone(),
            body_quaternion_relative_world=self._target_body_quaternion.clone(),
            joint_velocity=self._motion.joint_velocity[frames].clone(),
        )

    def get_benchmark_robot_state(self) -> BenchmarkRobotState:
        """Return the post-step robot state consumed by the benchmark metrics."""
        return BenchmarkRobotState(
            body_position_world=self._xpos[:, self._body_ids],
            body_quaternion_world=self._xquat[:, self._body_ids],
            joint_velocity=self._qvel[:, self._joint_qvel_indices],
        )
