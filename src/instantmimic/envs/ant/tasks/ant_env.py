from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from instantmimic.envs.direct_rl_env import DirectRLEnv, StepOutputs

from .ant_constants import (
    DEFAULT_JOINT_POSITIONS,
    DEFAULT_ROOT_POSITION,
    DEFAULT_ROOT_QUATERNION,
    SOURCE_MJCF_JOINT_ORDER,
    action_to_actuator_indices,
)
from .ant_env_cfg import AntEnvCfg
from .ant_kernels import (
    apply_ant_action_kernel,
    compute_ant_observation_kernel,
    compute_ant_task_kernel,
    reset_ant_state_kernel,
    stage_ant_actions_kernel,
)
from .ant_math import (
    build_observation,
    compute_intermediate_values,
    compute_rewards,
    quat_rotate,
)


class AntEnv(DirectRLEnv):
    cfg: AntEnvCfg

    # Reset writes authoritative simulator state; DirectRLEnv finalizes the
    # universal state prefix before reset observations are produced.

    def __init__(
        self,
        cfg: AntEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        if cfg.action_space != 8 or cfg.observation_space != 36:
            raise ValueError(
                "Isaac Ant requires action_space=8 and observation_space=36; "
                f"got action_space={cfg.action_space!r}, "
                f"observation_space={cfg.observation_space!r}."
            )
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self.qpos_view = wp.to_torch(self.mjw_data.qpos)
        self.qvel_view = wp.to_torch(self.mjw_data.qvel)
        self.ctrl_view = wp.to_torch(self.mjw_data.ctrl)

        self._init_model_indices()
        self._init_reset_state()
        self._init_task_buffers()
        self._use_native_task_kernels = bool(
            self.cfg.use_native_task_kernels
            and torch.device(self.device).type == "cuda"
        )
        if self._use_native_task_kernels:
            if self.parallelize_dones_and_rewards:
                raise ValueError(
                    "AntEnv cannot parallelize dones and rewards while "
                    "use_native_task_kernels is set: compute_ant_task_kernel "
                    "writes the reward from _get_dones, and _get_rewards only "
                    "returns the buffer it filled."
                )
            self._init_native_task_views()

    def _can_parallelize_dones_and_rewards(self) -> bool:
        """Never on the native path: rewards are produced by the dones hook.

        `_get_dones` launches `compute_ant_task_kernel`, which writes the reward
        along with the termination flags, and `_get_rewards` then just returns
        `self.reward_buf`. Running the two hooks concurrently would let the
        reward be read before the kernel that fills it has run. The eager path
        computes each independently and keeps the base class's answer.
        """

        if self._use_native_task_kernels:
            return False
        return super()._can_parallelize_dones_and_rewards()

    def _init_native_task_views(self) -> None:
        self._joint_qpos_idx_i32 = self._joint_qpos_idx.to(dtype=torch.int32)
        self._joint_qvel_idx_i32 = self._joint_qvel_idx.to(dtype=torch.int32)
        self._source_to_actuator_idx_i32 = self._source_to_actuator_idx.to(
            dtype=torch.int32
        )
        self._joint_qpos_idx_wp = wp.from_torch(
            self._joint_qpos_idx_i32, dtype=wp.int32
        )
        self._joint_qvel_idx_wp = wp.from_torch(
            self._joint_qvel_idx_i32, dtype=wp.int32
        )
        self._source_to_actuator_idx_wp = wp.from_torch(
            self._source_to_actuator_idx_i32, dtype=wp.int32
        )
        self._joint_lower_limits_wp = wp.from_torch(
            self._joint_lower_limits, dtype=wp.float32
        )
        self._joint_upper_limits_wp = wp.from_torch(
            self._joint_upper_limits, dtype=wp.float32
        )
        self._targets_wp = wp.from_torch(self._targets, dtype=wp.float32)
        self._actions_wp = wp.from_torch(self.actions, dtype=wp.float32)
        self._reset_buf_wp = wp.from_torch(self.reset_buf, dtype=wp.bool)
        self._motor_effort_ratio_wp = wp.from_torch(
            self._motor_effort_ratio, dtype=wp.float32
        )
        self._episode_length_buf_wp = wp.from_torch(
            self.episode_length_buf, dtype=wp.int64
        )
        self._potentials_wp = wp.from_torch(self.potentials, dtype=wp.float32)
        self._previous_potentials_wp = wp.from_torch(
            self.previous_potentials, dtype=wp.float32
        )
        self._default_qpos_wp = wp.from_torch(self._default_qpos, dtype=wp.float32)
        self._default_qvel_wp = wp.from_torch(self._default_qvel, dtype=wp.float32)
        self._default_ctrl_wp = wp.from_torch(self._default_ctrl, dtype=wp.float32)
        self._native_reward_wp = wp.from_torch(self.reward_buf, dtype=wp.float32)
        self._native_terminated_wp = wp.from_torch(self.reset_terminated, dtype=wp.bool)
        self._native_truncated_wp = wp.from_torch(self.reset_time_outs, dtype=wp.bool)
        self._native_observation_wp = wp.from_torch(
            self.obs_buf["policy"], dtype=wp.float32
        )

    def _bind_step_outputs(self, outputs: StepOutputs) -> None:
        super()._bind_step_outputs(outputs)
        if self._use_native_task_kernels:
            self._native_reward_wp = wp.from_torch(self.reward_buf, dtype=wp.float32)
            self._native_terminated_wp = wp.from_torch(
                self.reset_terminated, dtype=wp.bool
            )
            self._native_truncated_wp = wp.from_torch(
                self.reset_time_outs, dtype=wp.bool
            )
            self._native_observation_wp = wp.from_torch(
                outputs.observations["policy"], dtype=wp.float32
            )

    def _init_model_indices(self) -> None:
        if self.model.nu != 8:
            raise ValueError(f"Isaac Ant requires 8 actuators, found {self.model.nu}.")

        joint_ids = []
        joint_qpos_indices = []
        joint_qvel_indices = []
        for name in SOURCE_MJCF_JOINT_ORDER:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Missing Ant joint: {name}")
            joint_ids.append(joint_id)
            joint_qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            joint_qvel_indices.append(int(self.model.jnt_dofadr[joint_id]))

        self._joint_qpos_idx = torch.tensor(
            joint_qpos_indices, dtype=torch.long, device=self.device
        )
        self._joint_qvel_idx = torch.tensor(
            joint_qvel_indices, dtype=torch.long, device=self.device
        )
        joint_ranges = torch.as_tensor(
            self.model.jnt_range[joint_ids], dtype=torch.float32, device=self.device
        )
        self._joint_lower_limits = joint_ranges[:, 0].contiguous()
        self._joint_upper_limits = joint_ranges[:, 1].contiguous()

        actuator_joint_names = tuple(
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(self.model.actuator_trnid[actuator_id, 0]),
            )
            for actuator_id in range(self.model.nu)
        )
        self._source_to_actuator_idx = torch.tensor(
            action_to_actuator_indices(actuator_joint_names),
            dtype=torch.long,
            device=self.device,
        )

    def _init_reset_state(self) -> None:
        self._default_qpos = torch.zeros(
            self.model.nq, dtype=torch.float32, device=self.device
        )
        self._default_qpos[0:3] = torch.tensor(
            DEFAULT_ROOT_POSITION, dtype=torch.float32, device=self.device
        )
        self._default_qpos[3:7] = torch.tensor(
            DEFAULT_ROOT_QUATERNION, dtype=torch.float32, device=self.device
        )
        self._default_qpos[self._joint_qpos_idx] = torch.tensor(
            DEFAULT_JOINT_POSITIONS, dtype=torch.float32, device=self.device
        )
        self._default_qvel = torch.zeros(
            self.model.nv, dtype=torch.float32, device=self.device
        )
        self._default_ctrl = torch.zeros(
            self.model.nu, dtype=torch.float32, device=self.device
        )

    def _init_task_buffers(self) -> None:
        self._targets = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._targets[:, 0] = 1000.0
        self._start_rotation = torch.tensor(
            DEFAULT_ROOT_QUATERNION, dtype=torch.float32, device=self.device
        )
        self._inverse_start_rotation = self._start_rotation.unsqueeze(0).repeat(
            self.num_envs, 1
        )
        self._heading_basis = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._heading_basis[:, 0] = 1.0
        self._up_basis = torch.zeros_like(self._heading_basis)
        self._up_basis[:, 2] = 1.0

        self.potentials = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.previous_potentials = torch.zeros_like(self.potentials)
        self._motor_effort_ratio = torch.ones(
            8, dtype=torch.float32, device=self.device
        )
        self._intermediates: tuple[torch.Tensor, ...] | None = None
        self._obs_buf = torch.empty(
            (self.num_envs, 36), dtype=torch.float32, device=self.device
        )
        self.obs_buf = {"policy": self._obs_buf}
        self._default_step_outputs = StepOutputs(
            observations=self.obs_buf,
            rewards=self._default_reward_buf,
            terminated=self._default_terminated_buf,
            truncated=self._default_truncated_buf,
        )

    def _compute_intermediate_values(self) -> tuple[torch.Tensor, ...]:
        torso_position = self.qpos_view[:, :3]
        torso_rotation = self.qpos_view[:, 3:7]
        velocity = self.qvel_view[:, :3]
        angular_velocity = quat_rotate(torso_rotation, self.qvel_view[:, 3:6])
        joint_positions = self.qpos_view.index_select(1, self._joint_qpos_idx)
        self._intermediates = compute_intermediate_values(
            self._targets,
            torso_position,
            torso_rotation,
            velocity,
            angular_velocity,
            joint_positions,
            self._joint_lower_limits,
            self._joint_upper_limits,
            self._inverse_start_rotation,
            self._heading_basis,
            self._up_basis,
            self.potentials,
            self.previous_potentials,
            self.physics_dt,
        )
        return self._intermediates

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != 8:
            raise ValueError(f"Expected action dimension 8, got {actions.shape[-1]}.")
        if self._use_native_task_kernels:
            external_actions_wp = wp.from_torch(actions, dtype=wp.float32)
            wp.launch(
                stage_ant_actions_kernel,
                dim=(self.num_envs, 8),
                inputs=[external_actions_wp],
                outputs=[self._actions_wp],
            )
            return
        self.actions.copy_(actions)

    def _apply_action(self) -> None:
        if self._use_native_task_kernels:
            wp.launch(
                apply_ant_action_kernel,
                dim=(self.num_envs, 8),
                inputs=[
                    self._actions_wp,
                    self._source_to_actuator_idx_wp,
                    float(self.cfg.action_scale),
                ],
                outputs=[self.mjw_data.ctrl],
            )
            return
        self.ctrl_view.zero_()
        self.ctrl_view.index_copy_(
            1,
            self._source_to_actuator_idx,
            self.actions * float(self.cfg.action_scale),
        )

    def _get_observations(self, out_obs: torch.Tensor | None = None) -> dict:
        target = self.obs_buf["policy"] if out_obs is None else out_obs
        if self._use_native_task_kernels:
            observation_wp = self._native_observation_wp
            if out_obs is not None and out_obs is not self.obs_buf["policy"]:
                observation_wp = wp.from_torch(out_obs, dtype=wp.float32)
            wp.launch(
                compute_ant_observation_kernel,
                dim=(self.num_envs,),
                inputs=[
                    self.mjw_data.qpos,
                    self.mjw_data.qvel,
                    self._targets_wp,
                    self._joint_qpos_idx_wp,
                    self._joint_qvel_idx_wp,
                    self._joint_lower_limits_wp,
                    self._joint_upper_limits_wp,
                    self._actions_wp,
                    float(self.cfg.angular_velocity_scale),
                    float(self.cfg.dof_vel_scale),
                ],
                outputs=[observation_wp],
            )
            if out_obs is not None:
                return {"policy": out_obs}
            return self.obs_buf

        intermediates = (
            self._intermediates
            if self._intermediates is not None
            else self._compute_intermediate_values()
        )
        joint_velocities = self.qvel_view.index_select(1, self._joint_qvel_idx)
        target.copy_(
            build_observation(
                torso_z=self.qpos_view[:, 2],
                velocity_local=intermediates[4],
                angular_velocity_local=intermediates[5],
                yaw=intermediates[8],
                roll=intermediates[6],
                angle_to_target=intermediates[9],
                up_projection=intermediates[0],
                heading_projection=intermediates[1],
                joint_positions_scaled=intermediates[10],
                joint_velocities=joint_velocities,
                actions=self.actions,
                angular_velocity_scale=float(self.cfg.angular_velocity_scale),
                joint_velocity_scale=float(self.cfg.dof_vel_scale),
            )
        )
        if out_obs is not None:
            return {"policy": out_obs}
        return self.obs_buf

    def _get_rewards(self) -> torch.Tensor:
        if self._use_native_task_kernels:
            # Already filled by `compute_ant_task_kernel` in `_get_dones`; see
            # `_can_parallelize_dones_and_rewards` for why the order is fixed.
            return self.reward_buf
        intermediates = (
            self._intermediates
            if self._intermediates is not None
            else self._compute_intermediate_values()
        )
        joint_velocities = self.qvel_view.index_select(1, self._joint_qvel_idx)
        self.reward_buf.copy_(
            compute_rewards(
                actions=self.actions,
                reset_terminated=self.reset_terminated,
                up_projection=intermediates[0],
                heading_projection=intermediates[1],
                joint_velocities=joint_velocities,
                joint_positions_scaled=intermediates[10],
                potentials=intermediates[12],
                previous_potentials=intermediates[11],
                heading_weight=float(self.cfg.heading_weight),
                up_weight=float(self.cfg.up_weight),
                actions_cost_scale=float(self.cfg.actions_cost_scale),
                energy_cost_scale=float(self.cfg.energy_cost_scale),
                joint_velocity_scale=float(self.cfg.dof_vel_scale),
                death_cost=float(self.cfg.death_cost),
                alive_reward_scale=float(self.cfg.alive_reward_scale),
                motor_effort_ratio=self._motor_effort_ratio,
            )
        )
        return self.reward_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_native_task_kernels:
            wp.launch(
                compute_ant_task_kernel,
                dim=(self.num_envs,),
                inputs=[
                    self.mjw_data.qpos,
                    self.mjw_data.qvel,
                    self._targets_wp,
                    self._joint_qpos_idx_wp,
                    self._joint_qvel_idx_wp,
                    self._joint_lower_limits_wp,
                    self._joint_upper_limits_wp,
                    self._actions_wp,
                    self._motor_effort_ratio_wp,
                    self._episode_length_buf_wp,
                    int(self.max_episode_length),
                    float(self.physics_dt),
                    float(self.cfg.termination_height),
                    float(self.cfg.heading_weight),
                    float(self.cfg.up_weight),
                    float(self.cfg.actions_cost_scale),
                    float(self.cfg.energy_cost_scale),
                    float(self.cfg.dof_vel_scale),
                    float(self.cfg.death_cost),
                    float(self.cfg.alive_reward_scale),
                ],
                outputs=[
                    self._native_reward_wp,
                    self._native_terminated_wp,
                    self._native_truncated_wp,
                    self._previous_potentials_wp,
                    self._potentials_wp,
                ],
            )
            return self.reset_terminated, self.reset_time_outs
        self._compute_intermediate_values()
        self.reset_terminated.copy_(
            self.qpos_view[:, 2] < float(self.cfg.termination_height)
        )
        self.reset_time_outs.copy_(self.episode_length_buf >= self.max_episode_length)
        return self.reset_terminated, self.reset_time_outs

    def _reset_idx(self, reset_mask: torch.Tensor):
        super()._reset_idx(reset_mask)
        if self._use_native_task_kernels and reset_mask is self.reset_buf:
            reset_mask_wp = self._reset_buf_wp
        else:
            reset_mask_wp = wp.from_torch(reset_mask, dtype=wp.bool)
        mjw.reset_data(self.mjw_model, self.mjw_data, reset_mask_wp)
        self.actions.masked_fill_(reset_mask.unsqueeze(-1), 0.0)
        if self._use_native_task_kernels:
            wp.launch(
                reset_ant_state_kernel,
                dim=(self.num_envs,),
                inputs=[
                    reset_mask_wp,
                    self._default_qpos_wp,
                    self._default_qvel_wp,
                    self._default_ctrl_wp,
                    self._targets_wp,
                    float(self.physics_dt),
                ],
                outputs=[
                    self.mjw_data.qpos,
                    self.mjw_data.qvel,
                    self.mjw_data.ctrl,
                    self._previous_potentials_wp,
                    self._potentials_wp,
                ],
            )
            self._intermediates = None
            return
        mask = reset_mask.unsqueeze(-1)
        self.qpos_view.copy_(
            torch.where(mask, self._default_qpos.unsqueeze(0), self.qpos_view)
        )
        self.qvel_view.copy_(
            torch.where(mask, self._default_qvel.unsqueeze(0), self.qvel_view)
        )
        self.ctrl_view.copy_(
            torch.where(mask, self._default_ctrl.unsqueeze(0), self.ctrl_view)
        )
        reset_to_target = self._targets - self._default_qpos[:3].unsqueeze(0)
        reset_to_target[:, 2] = 0.0
        reset_potential = (
            -torch.linalg.vector_norm(reset_to_target, dim=-1) / self.physics_dt
        )
        self.potentials.copy_(torch.where(reset_mask, reset_potential, self.potentials))
        self._compute_intermediate_values()


__all__ = ["AntEnv"]
