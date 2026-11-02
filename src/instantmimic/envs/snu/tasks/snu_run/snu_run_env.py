"""SNU humanoid running task, ported from Isaac Lab's locomotion environment.

Portions derived from Isaac Lab (BSD-3-Clause); see THIRD_PARTY_NOTICES.md.
`compute_heading_and_up`, `compute_rot`, `compute_intermediate_values`,
`compute_rewards`, `normalize_angle` and the observation concatenation follow
`isaaclab_tasks/direct/locomotion/locomotion_env.py` statement for statement,
with the math backend swapped to MuJoCo Warp and pymotion. The reward
coefficients and termination thresholds are that benchmark's definition.
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from pymotion.rotations import quat_torch

from instantmimic.envs.direct_rl_env import DirectRLEnv

from .control_resolver import make_default_control_cfg
from .joint_state import JointStateProjection
from .snu_run_env_cfg import SNURunEnvCfg

_ROOT_OBSERVATION_DIM = 12
_JOINT_COORDINATE_DIM = 50
_TASK_OBSERVATION_DIM = (
    _ROOT_OBSERVATION_DIM + 2 * _JOINT_COORDINATE_DIM + _JOINT_COORDINATE_DIM
)


def get_euler_xyz(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qw, qx, qy, qz = 0, 1, 2, 3
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = (
        q[:, qw] * q[:, qw]
        - q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        + q[:, qz] * q[:, qz]
    )
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * (torch.pi / 2.0),
        torch.asin(sinp),
    )

    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = (
        q[:, qw] * q[:, qw]
        + q[:, qx] * q[:, qx]
        - q[:, qy] * q[:, qy]
        - q[:, qz] * q[:, qz]
    )
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll % (2 * torch.pi), pitch % (2 * torch.pi), yaw % (2 * torch.pi)


def compute_heading_and_up(
    torso_rotation: torch.Tensor,
    inv_start_rot: torch.Tensor,
    to_target: torch.Tensor,
    vec0: torch.Tensor,
    vec1: torch.Tensor,
    up_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_envs = torso_rotation.shape[0]
    target_dirs = torch.nn.functional.normalize(to_target, dim=-1)

    torso_quat = quat_torch.mul(torso_rotation, inv_start_rot)
    up_vec = quat_torch.mul_vec(torso_quat, vec1).view(num_envs, 3)
    heading_vec = quat_torch.mul_vec(torso_quat, vec0).view(num_envs, 3)
    up_proj = up_vec[:, up_idx]
    heading_proj = torch.bmm(
        heading_vec.view(num_envs, 1, 3),
        target_dirs.view(num_envs, 3, 1),
    ).view(num_envs)
    return torso_quat, up_proj, heading_proj, up_vec, heading_vec


def compute_rot(torso_quat, velocity, ang_velocity, targets, torso_positions):
    torso_quat_inv = quat_torch.conjugate(torso_quat)
    vel_loc = quat_torch.mul_vec(torso_quat_inv, velocity)
    angvel_loc = quat_torch.mul_vec(torso_quat_inv, ang_velocity)
    roll, pitch, yaw = get_euler_xyz(torso_quat)
    walk_target_angle = torch.atan2(
        targets[:, 1] - torso_positions[:, 1],
        targets[:, 0] - torso_positions[:, 0],
    )
    return vel_loc, angvel_loc, roll, pitch, yaw, walk_target_angle - yaw


@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class SNURunEnv(DirectRLEnv):
    cfg: SNURunEnvCfg

    def __init__(
        self,
        cfg: SNURunEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        configured_observation_space = int(cfg.observation_space)
        if configured_observation_space != _TASK_OBSERVATION_DIM:
            raise ValueError(
                "SNU Run requires observation_space="
                f"{_TASK_OBSERVATION_DIM}, got {configured_observation_space}."
            )
        if getattr(cfg, "control", None) is None:
            self._control_pipeline = make_default_control_cfg(
                cfg
            ).build_control_pipeline()
        self._task_obs_dim = _TASK_OBSERVATION_DIM
        self._obs_main_dim = _TASK_OBSERVATION_DIM
        super().__init__(cfg, render_mode, **kwargs)

        coordinate_mjcf_path = self._resolve_cfg_path_in_place(
            self.cfg,
            "joint_coordinate_mjcf_path",
        )
        coordinate_model = mujoco.MjModel.from_xml_path(coordinate_mjcf_path)
        self._joint_state_projection = JointStateProjection.from_models(
            model=self.model,
            coordinate_model=coordinate_model,
            device=self.device,
        )
        if self._joint_state_projection.coordinate_count != _JOINT_COORDINATE_DIM:
            raise ValueError(
                "SNU Run requires exactly "
                f"{_JOINT_COORDINATE_DIM} joint coordinates, got "
                f"{self._joint_state_projection.coordinate_count}."
            )
        self.joint_range = torch.stack(
            (
                self._joint_state_projection.lower_limits,
                self._joint_state_projection.upper_limits,
            ),
            dim=-1,
        )
        if int(self.actions.shape[-1]) != _JOINT_COORDINATE_DIM:
            raise ValueError(
                "SNU Run requires a 50-dimensional policy action for its stable "
                f"previous-action observation, got {int(self.actions.shape[-1])}."
            )

        self.torso_body_idx = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "Torso",
        )
        if self.torso_body_idx < 0:
            raise ValueError("SNU Run model is missing the required 'Torso' body.")

        xpos_view = wp.to_torch(self.mjw_data.xpos)
        xquat_view = wp.to_torch(self.mjw_data.xquat)
        self.default_torso_position = xpos_view[:, self.torso_body_idx].clone()
        self.default_torso_rotation = xquat_view[:, self.torso_body_idx].clone()
        self._xpos_view = xpos_view
        self._xquat_view = xquat_view
        self._qpos_view = wp.to_torch(self.mjw_data.qpos)
        self._qvel_view = wp.to_torch(self.mjw_data.qvel)
        self._qfrc_actuator_view = wp.to_torch(self.mjw_data.qfrc_actuator)
        self._qfrc_applied_view = wp.to_torch(self.mjw_data.qfrc_applied)
        self.dof_pos = torch.zeros(
            (self.num_envs, _JOINT_COORDINATE_DIM),
            device=self.device,
            dtype=torch.float32,
        )
        self.dof_vel = torch.zeros_like(self.dof_pos)
        default_qpos = torch.as_tensor(
            self.model.qpos0,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        default_qvel = torch.zeros(
            (1, int(self.model.nv)),
            device=self.device,
            dtype=torch.float32,
        )
        default_dof_pos, _ = self._joint_state_projection.project(
            default_qpos,
            default_qvel,
        )
        self.default_dof_pos_scaled = (
            2.0 * default_dof_pos[0]
            - self._joint_state_projection.lower_limits
            - self._joint_state_projection.upper_limits
        ) / (
            self._joint_state_projection.upper_limits
            - self._joint_state_projection.lower_limits
        )
        self._mechanical_power_sum = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )

        self.potentials = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor(
            [1000, 0, 0],
            dtype=torch.float32,
            device=self.device,
        ).repeat((self.num_envs, 1))
        self.start_rotation = torch.tensor(
            [1, 0, 0, 0],
            device=self.device,
            dtype=torch.float32,
        )
        self.up_vec = torch.tensor(
            [0, 0, 1],
            dtype=torch.float32,
            device=self.device,
        ).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor(
            [1, 0, 0],
            dtype=torch.float32,
            device=self.device,
        ).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_torch.conjugate(self.start_rotation).repeat(
            (self.num_envs, 1)
        )
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self._obs_buf = torch.zeros(
            (self.num_envs, int(self.cfg.observation_space)),
            dtype=torch.float32,
            device=self.device,
        )
        self.obs_buf = {"policy": self._obs_buf}

        finalizer = getattr(self._control_pipeline, "finalize_task_state", None)
        if callable(finalizer):
            finalizer(self)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._mechanical_power_sum.zero_()
        return DirectRLEnv._pre_physics_step(self, actions)

    def _apply_action(self) -> None:
        return DirectRLEnv._apply_action(self)

    def _post_physics_substep(self, substep_idx: int) -> None:
        del substep_idx
        applied_torque = (
            self._qfrc_actuator_view[:, 6:] + self._qfrc_applied_view[:, 6:]
        )
        instantaneous_power = torch.mean(
            torch.abs(applied_torque * self._qvel_view[:, 6:]),
            dim=-1,
        )
        self._mechanical_power_sum.add_(instantaneous_power)

    def _compute_intermediate_values(
        self,
        after_reset: bool = False,
        reset_mask: torch.Tensor | None = None,
    ) -> None:
        self.torso_position = self._xpos_view[:, self.torso_body_idx].clone()
        self.torso_rotation = self._xquat_view[:, self.torso_body_idx].clone()
        if after_reset and reset_mask is not None:
            mask_2d = reset_mask.unsqueeze(-1)
            self.torso_position = torch.where(
                mask_2d,
                self.default_torso_position,
                self.torso_position,
            )
            self.torso_rotation = torch.where(
                mask_2d,
                self.default_torso_rotation,
                self.torso_rotation,
            )
            saved_prev_potentials = self.prev_potentials.clone()

        self.velocity = self._qvel_view[:, :3].clone()
        self.ang_velocity = self._qvel_view[:, 3:6].clone()
        self._joint_state_projection.project_into(
            self._qpos_view,
            self._qvel_view,
            out_pos=self.dof_pos,
            out_vel=self.dof_vel,
        )
        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self._joint_state_projection.lower_limits,
            self._joint_state_projection.upper_limits,
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.physics_dt,
        )

        if after_reset and reset_mask is not None:
            torch.where(
                reset_mask,
                self.prev_potentials,
                saved_prev_potentials,
                out=self.prev_potentials,
            )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        target = self.obs_buf["policy"]
        prev_action_getter = getattr(
            self._control_pipeline,
            "get_prev_action_observation_tensor",
            None,
        )
        prev_actions = (
            prev_action_getter() if callable(prev_action_getter) else self.actions
        )
        if int(prev_actions.shape[-1]) != _JOINT_COORDINATE_DIM:
            prev_actions = self.actions
        elif callable(prev_action_getter):
            prev_actions = self._joint_state_projection.project_policy_actions(
                prev_actions
            )
        tensors = (
            self.torso_position[:, 2].view(-1, 1),
            self.vel_loc,
            self.angvel_loc * self.cfg.angular_velocity_scale,
            normalize_angle(self.yaw).unsqueeze(-1),
            normalize_angle(self.roll).unsqueeze(-1),
            normalize_angle(self.angle_to_target).unsqueeze(-1),
            self.up_proj.unsqueeze(-1),
            self.heading_proj.unsqueeze(-1),
            self.dof_pos_scaled,
            self.dof_vel * self.cfg.dof_vel_scale,
            prev_actions,
        )
        torch.cat(tensors, dim=-1, out=target[:, : self._task_obs_dim])
        extension_writer = getattr(
            self._control_pipeline,
            "write_observation_extension",
            None,
        )
        if callable(extension_writer):
            extension_writer(target, int(self._obs_main_dim))
        return self.obs_buf

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.reset_terminated,
            self.actions,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_pos_scaled,
            self.default_dof_pos_scaled,
            self._mechanical_power_sum,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.cfg.decimation,
            self.step_dt,
        )
        self.reward_buf.copy_(total_reward)
        return self.reward_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        torch.ge(
            self.episode_length_buf,
            self.max_episode_length,
            out=self.reset_time_outs,
        )
        torch.lt(
            self.torso_position[:, 2],
            self.cfg.termination_height,
            out=self.reset_terminated,
        )
        return self.reset_terminated, self.reset_time_outs

    def _reset_idx(self, reset_mask: torch.Tensor) -> None:
        super()._reset_idx(reset_mask)
        mjw.reset_data(self.mjw_model, self.mjw_data, wp.from_torch(reset_mask))
        self._run_mjw_kinematics()
        self._run_mjw_com_state()
        self._control_pipeline.reset_idx(reset_mask)

        to_target = self.targets - self.default_torso_position
        to_target[:, 2] = 0.0
        new_potentials = -torch.norm(to_target, p=2, dim=-1)
        torch.where(
            reset_mask,
            new_potentials,
            self.potentials,
            out=self.potentials,
        )
        self._compute_intermediate_values(after_reset=True, reset_mask=reset_mask)

    def _update_render_data(self) -> None:
        updater = getattr(self._control_pipeline, "update_render_data", None)
        if callable(updater):
            updater()

    def _append_dynamic_render_geoms(
        self,
        *,
        scene,
        render_model,
        render_data,
    ) -> None:
        del render_model
        appender = getattr(
            self._control_pipeline,
            "append_dynamic_render_geoms",
            None,
        )
        if callable(appender):
            appender(scene=scene, render_data=render_data)

    def _dynamic_render_maxgeom(self) -> int:
        getter = getattr(self._control_pipeline, "dynamic_render_maxgeom", None)
        return int(getter()) if callable(getter) else 0

    def dynamic_gl_render_sources(self) -> list[object]:
        getter = getattr(self._control_pipeline, "dynamic_gl_render_sources", None)
        return list(getter()) if callable(getter) else []


def compute_rewards(
    reset_terminated: torch.Tensor,
    actions: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    default_dof_pos_scaled: torch.Tensor,
    mechanical_power_sum: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    decimation: int,
    dt: float,
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(
        heading_proj > 0.8,
        heading_weight_tensor,
        heading_weight * heading_proj / 0.8,
    )
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)
    upper_limit_cost = (dof_pos_scaled > 0.98) & (default_dof_pos_scaled <= 0.98)
    lower_limit_cost = (dof_pos_scaled < -0.98) & (default_dof_pos_scaled >= -0.98)
    dof_at_limit_cost = torch.sum(upper_limit_cost | lower_limit_cost, dim=-1)
    actions_cost = torch.sum(actions * actions, dim=-1)
    mechanical_power_cost = mechanical_power_sum * dof_vel_scale / float(decimation)
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = (potentials - prev_potentials) / dt
    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - dof_at_limit_cost
        - actions_cost_scale * actions_cost
        - energy_cost_scale * mechanical_power_cost
    )
    return torch.where(
        reset_terminated,
        torch.ones_like(total_reward) * death_cost,
        total_reward,
    )


def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    del dt
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation,
        inv_start_rot,
        to_target,
        basis_vec0,
        basis_vec1,
        2,
    )
    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat,
        velocity,
        ang_velocity,
        targets,
        torso_position,
    )
    dof_pos_scaled = (2.0 * dof_pos - dof_lower_limits - dof_upper_limits) / (
        dof_upper_limits - dof_lower_limits
    )
    prev_potentials.copy_(potentials)
    torch.norm(to_target, p=2, dim=-1, out=potentials)
    potentials.neg_()
    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
