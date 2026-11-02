from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import torch
import warp as wp

from instantmimic.envs.snu.tasks.snu_imitation import nan_debug, observation_runtime
from instantmimic.envs.snu.tasks.snu_imitation.adaptive_sampling import (
    accumulate_episode_reward_sum_kernel,
    accumulate_sonic_failure_statistics_kernel,
)
from instantmimic.envs.tracking import metrics as tracking_metrics
from instantmimic.envs.tracking import metrics_warp as tracking_metrics_warp
from instantmimic.envs.tracking.action_smoothness import (
    ActionSmoothnessState,
    create_action_smoothness_state,
    reset_action_smoothness_history,
    update_action_smoothness_error,
)
from instantmimic.envs.tracking.reward_config import (
    RewardSettings,
    joint_root_error_weight,
    resolve_ee_position_layout,
    resolve_ee_rotation_body_mask,
    resolve_joint_dof_scales,
    resolve_reward_body_weight,
    resolve_reward_settings,
)
from instantmimic.envs.tracking.reward_kernels import (
    REWARD_KERNEL_BLOCK_DIM,
    RewardKernelSpec,
    build_reward_dof_scales,
    compute_body_state_frame_sigma_multipliers,
    launch_compute_rewards,
    make_compute_rewards_kernel,
    resolve_reward_kernel_scalar_weights,
)
from instantmimic.utils.startup_profiler import StartupProfiler, startup_nvtx_range


@dataclass
class RewardRuntimeState:
    reward: torch.Tensor
    reward_terms: torch.Tensor
    reward_term_names: tuple[str, ...]
    substep_mean_enabled: bool
    substep_reward_accumulator: torch.Tensor
    substep_reward_terms_accumulator: torch.Tensor
    substep_reward_sample: torch.Tensor
    substep_reward_terms_sample: torch.Tensor
    substep_reference_frame_index: torch.Tensor
    compute_rewards_kernel: Any | None = None
    tracking_workspace_excluding_root: (
        tracking_metrics_warp.ReferenceTrackingMetricWorkspace | None
    ) = None
    tracking_workspace_including_root: (
        tracking_metrics_warp.ReferenceTrackingMetricWorkspace | None
    ) = None
    # Per-env mean squared command change read by the reward kernel; zeros
    # (never read) without env.reward.action_smoothness.
    action_smoothness_error: torch.Tensor | None = None
    action_smoothness: ActionSmoothnessState | None = None


def initialize_reward_kernel(env) -> None:
    startup_profiler = StartupProfiler(
        range_factory=lambda name: startup_nvtx_range(f"reward_kernel:{name}")
    )
    settings = env.reward_settings
    with startup_profiler.phase("constant_tuple_keys"):
        loop_root_q_tuple = tuple(float(v) for v in env.loop_root_quat_tuple)
        loop_root_t_tuple = tuple(float(v) for v in env.loop_root_pos_tuple)
        ee_body_idx_tuple = tuple(int(v) for v in env.ee_body_idx_tuple)
        reward_body_idx_tuple = tuple(int(v) for v in env.reward_body_idx_tuple)
        body_parent_idx_tuple = tuple(int(v) for v in env.body_parent_idx_tuple)
        hinge_body_idx_tuple = tuple(int(v) for v in env.hinge_body_indices_tuple)
        hinge_parent_body_idx_tuple = tuple(
            int(v) for v in env.hinge_parent_body_indices_tuple
        )
        hinge_axis_local_tuple = tuple(
            tuple(float(c) for c in row) for row in env.hinge_axis_local_tuple
        )
        hinge_multi_body_mask_tuple = tuple(
            int(v) for v in env.hinge_multi_body_mask_tuple
        )
    with startup_profiler.phase("reward_kernel_spec"):
        reward_weights = resolve_reward_kernel_scalar_weights(env.reward_settings)
        root_body_name = (
            mujoco.mj_id2name(
                env.model, mujoco.mjtObj.mjOBJ_BODY, int(env.root_body_idx)
            )
            if settings.pose_body_weights is not None
            or settings.velocity_body_weights is not None
            else None
        ) or ""
        ee_position_layout = resolve_ee_position_layout(
            settings, env.model, ee_body_idx_tuple
        )
        reward_kernel_spec = RewardKernelSpec(
            motion_frame_count=env.reward_motion_frame_count,
            n_hinge=len(env.hinge_qpos_indices),
            n_ball=env.n_ball_joints,
            nv=env.model.nv,
            root_body_idx=int(env.root_body_idx),
            ee_body_idx=ee_body_idx_tuple,
            reward_body_idx=reward_body_idx_tuple,
            pose_w=reward_weights.pose_w,
            vel_w=reward_weights.vel_w,
            com_w=reward_weights.com_w,
            ee_w=reward_weights.ee_w,
            ee_rot_w=reward_weights.ee_rot_w,
            alive_reward=reward_weights.alive_reward,
            reward_combination_mode=int(settings.combination_mode),
            body_state_reward=bool(settings.body_state_reward),
            body_state_reward_frame_mode=int(settings.body_state_frame_mode),
            joint_pose_reward_mode=int(settings.joint_pose_mode),
            joint_velocity_reward_mode=int(settings.joint_velocity_mode),
            body_parent_idx=body_parent_idx_tuple,
            hinge_body_idx=hinge_body_idx_tuple,
            hinge_parent_body_idx=hinge_parent_body_idx_tuple,
            hinge_axis_local=hinge_axis_local_tuple,
            hinge_multi_body_mask=hinge_multi_body_mask_tuple,
            mix_w_pose=float(settings.mix_w_pose),
            mix_w_vel=float(settings.mix_w_vel),
            mix_w_body_pos=float(settings.mix_w_body_pos),
            mix_w_body_rot=float(settings.mix_w_body_rot),
            mix_w_body_linvel=float(settings.mix_w_body_linvel),
            mix_w_body_angvel=float(settings.mix_w_body_angvel),
            mix_w_com=float(settings.mix_w_com),
            mix_w_ee_pos=float(settings.mix_w_ee_pos),
            mix_w_ee_rot=float(settings.mix_w_ee_rot),
            pose_include_root=bool(settings.pose_include_root),
            vel_include_root=bool(settings.velocity_include_root),
            ee_pos_root_relative=bool(settings.ee_pos_root_relative),
            velocity_reward_seam_fade_frames=int(settings.velocity_seam_fade_frames),
            vel_valid_all=bool(settings.vel_valid_all),
            loop_root_q=loop_root_q_tuple,
            loop_root_t=loop_root_t_tuple,
            root_qpos_count=int(getattr(env, "root_qpos_count", 7)),
            root_dof_count=int(getattr(env, "root_dof_count", 6)),
            passive_dof_count=len(env.dof_roles.passive_dof),
            reward_power_q=float(settings.power_q),
            reward_power_beta=float(settings.power_beta),
            reward_power_epsilon=float(settings.power_epsilon),
            body_state_pos_reward_weight=float(settings.body_state_pos_w),
            body_state_rot_reward_weight=float(settings.body_state_rot_w),
            body_state_linvel_reward_weight=float(settings.body_state_linvel_w),
            body_state_angvel_reward_weight=float(settings.body_state_angvel_w),
            body_state_frame_sigma_enabled=settings.frame_sigma is not None,
            reward_body_weight=resolve_reward_body_weight(
                settings, env.model, reward_body_idx_tuple
            ),
            reward_scale=float(settings.combination_scale),
            pose_root_error_scale=joint_root_error_weight(
                settings.pose_body_weights, root_body_name
            ),
            vel_root_error_scale=joint_root_error_weight(
                settings.velocity_body_weights, root_body_name
            ),
            ee_rot_body_mask=resolve_ee_rotation_body_mask(
                settings, env.model, ee_body_idx_tuple
            ),
            ee_rot_mean_per_body=settings.ee_rotation_mean == "body",
            ee_pos_body_idx=ee_position_layout.body_idx,
            ee_pos_body_offset=ee_position_layout.body_offset,
            ee_pos_anchor_body_idx=ee_position_layout.anchor_body_idx,
            ee_pos_anchor_offset=ee_position_layout.anchor_offset,
            action_smoothness_enabled=settings.action_smoothness_enabled,
            action_smoothness_weight=float(settings.action_smoothness_weight or 0.0),
            mix_w_action_smoothness=float(settings.mix_w_action_smoothness),
        )
    with startup_profiler.phase("reward_kernel"):
        env.reward_runtime_state.compute_rewards_kernel = make_compute_rewards_kernel(
            reward_kernel_spec
        )
    startup_profiler.log_summary(prefix="REWARD_KERNEL_STARTUP_PROFILE")


__all__ = (
    "RewardSettings",
    "RewardRuntimeState",
    "resolve_reward_settings",
    "initialize_reward_kernel",
    "initialize_reward_state",
    "get_rewards",
    "accumulate_reward_metrics",
    "prepare_substep_rewards",
    "accumulate_substep_reward",
    "finalize_substep_rewards",
    "write_substep_reward_ref_frame_idx",
    "get_training_diagnostic_means",
    "get_reference_tracking_metrics",
    "get_video_overlay_plots",
)


def get_video_overlay_plots(env, env_idx: int = 0) -> dict[str, object]:
    if env_idx < 0 or env_idx >= env.num_envs:
        return {}

    state = env.reward_runtime_state
    r_values = state.reward_terms[env_idx].detach().cpu().tolist()
    total_value = float(state.reward[env_idx].detach().cpu().item())
    series = dict(zip(state.reward_term_names, r_values, strict=True))
    series["total_reward"] = total_value

    return {
        "title": "Reward Debug Overlay",
        "plots": [
            {
                "title": "reward terms + total",
                "series": series,
                "colors": {
                    "r_q": (231, 76, 60),
                    "r_qd": (243, 156, 18),
                    "r_body_pos": (231, 76, 60),
                    "r_body_rot": (192, 57, 43),
                    "r_body_linvel": (243, 156, 18),
                    "r_body_angvel": (211, 84, 0),
                    "r_com": (46, 204, 113),
                    "r_ee": (52, 152, 219),
                    "r_ee_rot": (155, 89, 182),
                    "r_action_smooth": (26, 188, 156),
                    "total_reward": (236, 240, 241),
                },
            },
        ],
    }


def write_substep_reward_ref_frame_idx(
    ref_frame_idx: torch.Tensor,
    out: torch.Tensor,
    *,
    decimation: int,
    substep_idx: int,
) -> None:
    out.copy_(ref_frame_idx)
    out.sub_(1)
    out.mul_(int(decimation))
    out.add_(int(substep_idx) + 1)


def _init_body_state_frame_sigma_multiplier(self) -> None:
    self.body_state_frame_sigma_multiplier = torch.ones(
        int(self.reward_motion_frame_count),
        4,
        device=self.device,
        dtype=torch.float32,
    )
    settings = self.reward_settings
    if not settings.body_state_reward:
        return
    if settings.frame_sigma is None:
        return

    (
        pos_weight,
        rot_weight,
        linvel_weight,
        angvel_weight,
    ) = settings.body_state_component_weights
    self.body_state_frame_sigma_multiplier = (
        compute_body_state_frame_sigma_multipliers(
            self.reward_reference_body_state,
            self.reward_body_idx,
            self.reward_motion_clip_start,
            self.reward_motion_clip_length,
            pos_weight=pos_weight,
            rot_weight=rot_weight,
            linvel_weight=linvel_weight,
            angvel_weight=angvel_weight,
            reference_delta_scale=float(settings.frame_sigma[0]),
            min_weight_multiplier=float(settings.frame_sigma[1]),
            reward_body_weight=(
                resolve_reward_body_weight(
                    settings,
                    self.model,
                    tuple(int(v) for v in self.reward_body_idx.detach().cpu().tolist()),
                )
                if settings.body_weights
                else None
            ),
        )
        .to(device=self.device, dtype=torch.float32)
        .contiguous()
    )


def _resolve_muscle_activation_tensor(self) -> torch.Tensor | None:
    controller = getattr(self, "fast_muscle", None)
    if controller is None:
        return None
    value = controller.get_current_muscle_activation(kind="state")
    return value if isinstance(value, torch.Tensor) and value.ndim == 2 else None


def get_training_diagnostic_means(self) -> dict[str, float]:
    metrics: dict[str, float] = {}
    activation = _resolve_muscle_activation_tensor(self)
    if isinstance(activation, torch.Tensor) and activation.numel() > 0:
        activation_f = activation.float()
        metrics["muscle_activation_mean"] = float(activation_f.mean().item())
        metrics["muscle_activation_square_mean"] = float(
            activation_f.square().mean().item()
        )
        metrics["muscle_activation_abs_max"] = float(activation_f.abs().max().item())
    return metrics


def _apply_body_weights(self) -> None:
    """Replace a term's built-in DOF scales with its body_weights mapping."""
    settings = self.reward_settings
    if settings.pose_body_weights is None and settings.velocity_body_weights is None:
        return
    common = dict(
        joint_ids=tuple(int(j) for j in self.dof_roles.joint_ids),
        root_body_idx=int(self.root_body_idx),
        root_dof_count=int(getattr(self, "root_dof_count", 6)),
        tracked_dof_mask=tuple(bool(v) for v in self.joint_dof_mask.tolist()),
    )
    for weights, attr, term in (
        (settings.pose_body_weights, "dof_q_scale", "pose"),
        (settings.velocity_body_weights, "dof_qd_scale", "velocity"),
    ):
        if weights is None:
            continue
        values = resolve_joint_dof_scales(
            weights,
            self.model,
            key=f"env.reward.tracking.{term}.body_weights",
            **common,
        )
        setattr(
            self,
            attr,
            torch.tensor(values, device=self.device, dtype=torch.float32),
        )


def initialize_reward_state(self):
    startup_profiler = StartupProfiler(
        range_factory=lambda name: startup_nvtx_range(f"reward_state:{name}")
    )
    # The non-root DOF space [root, nv); passive DOFs in it keep zero scales.
    joint_dof_count = self.model.nv - int(getattr(self, "root_dof_count", 6))
    with startup_profiler.phase("reward_scales"):
        joint_indices = getattr(self, "joint_dof_indices_tuple", None)
        talus_indices = getattr(self, "talus_dof_indices_tuple", None)
        toe_indices = getattr(self, "toe_dof_indices_tuple", None)
        if (
            joint_indices is not None
            and talus_indices is not None
            and toe_indices is not None
        ):
            dof_q_scale = [0.0] * joint_dof_count
            dof_qd_scale = [0.0] * joint_dof_count
            for dof_idx in joint_indices:
                dof_q_scale[int(dof_idx)] = 1.0
                dof_qd_scale[int(dof_idx)] = 1.0
            for dof_idx in talus_indices:
                dof_q_scale[int(dof_idx)] = 0.5
                dof_qd_scale[int(dof_idx)] = 0.5
            for dof_idx in toe_indices:
                dof_q_scale[int(dof_idx)] = 0.01
                dof_qd_scale[int(dof_idx)] = 1.0
            self.dof_q_scale = torch.tensor(
                dof_q_scale, device=self.device, dtype=torch.float32
            )
            self.dof_qd_scale = torch.tensor(
                dof_qd_scale, device=self.device, dtype=torch.float32
            )
        else:
            self.dof_q_scale, self.dof_qd_scale = build_reward_dof_scales(
                joint_dof_mask=self.joint_dof_mask,
                talus_dof_indices=self.talus_dof_indices,
                toe_dof_indices=self.toe_dof_indices,
            )
        _apply_body_weights(self)

    with startup_profiler.phase("index_casts"):
        self.hinge_qpos_idx_i32 = self.hinge_qpos_indices.to(torch.int32)
        self.hinge_dof_idx_i32 = self.hinge_dof_indices.to(torch.int32)
        self.ee_body_idx_i32 = self.ee_body_idx.to(torch.int32)
        self.future_pose_body_idx_i32 = self.future_pose_body_idx.to(torch.int32)

    with startup_profiler.phase("joint_velocity_projection"):
        hinge_dof_indices = getattr(self, "hinge_dof_indices_tuple", None)
        hinge_body_indices = getattr(self, "hinge_body_indices_tuple", None)
        hinge_parent_body_indices = getattr(
            self, "hinge_parent_body_indices_tuple", None
        )
        hinge_axis_local = getattr(self, "hinge_axis_local_tuple", None)
        hinge_multi_body_mask = getattr(self, "hinge_multi_body_mask_tuple", None)
        if (
            hinge_dof_indices is not None
            and hinge_body_indices is not None
            and hinge_parent_body_indices is not None
            and hinge_axis_local is not None
            and hinge_multi_body_mask is not None
        ):
            use_local = [0] * joint_dof_count
            body_idx = [0] * joint_dof_count
            parent_body_idx = [0] * joint_dof_count
            axis_values = [[0.0, 0.0, 0.0] for _ in range(joint_dof_count)]
            for local_idx, dof_idx in enumerate(hinge_dof_indices):
                dof_idx = int(dof_idx)
                body_idx[dof_idx] = int(hinge_body_indices[local_idx])
                parent_body_idx[dof_idx] = int(hinge_parent_body_indices[local_idx])
                axis_values[dof_idx] = [
                    float(component) for component in hinge_axis_local[local_idx]
                ]
                if self.observation_settings.joint_velocity_mode == 1:
                    use_local[dof_idx] = 1
                elif self.observation_settings.joint_velocity_mode == 2:
                    use_local[dof_idx] = int(bool(hinge_multi_body_mask[local_idx]))
            self.obs_joint_vel_use_local_i32 = torch.tensor(
                use_local, device=self.device, dtype=torch.int32
            )
            self.obs_joint_vel_body_idx_i32 = torch.tensor(
                body_idx, device=self.device, dtype=torch.int32
            )
            self.obs_joint_vel_parent_body_idx_i32 = torch.tensor(
                parent_body_idx, device=self.device, dtype=torch.int32
            )
            self.obs_joint_vel_axis = torch.tensor(
                axis_values, device=self.device, dtype=torch.float32
            )
        else:
            self.obs_joint_vel_use_local_i32 = torch.zeros(
                joint_dof_count, device=self.device, dtype=torch.int32
            )
            self.obs_joint_vel_body_idx_i32 = torch.zeros(
                joint_dof_count, device=self.device, dtype=torch.int32
            )
            self.obs_joint_vel_parent_body_idx_i32 = torch.zeros(
                joint_dof_count, device=self.device, dtype=torch.int32
            )
            self.obs_joint_vel_axis = torch.zeros(
                joint_dof_count, 3, device=self.device, dtype=torch.float32
            )
            observation_runtime.configure_obs_joint_velocity_projection(self)

    with startup_profiler.phase("ball_joint_indices"):
        if self.n_ball_joints > 0:
            self.ball_qpos_adr_i32 = self.ball_qpos_gather[::4].to(torch.int32)
            self.ball_dof_adr_i32 = self.ball_dof_scatter[::3].to(torch.int32)
        else:
            self.ball_qpos_adr_i32 = torch.zeros(
                0, device=self.device, dtype=torch.int32
            )
            self.ball_dof_adr_i32 = torch.zeros(
                0, device=self.device, dtype=torch.int32
            )

    with startup_profiler.phase("reward_buffers"):
        reward = torch.zeros(self.num_envs, device=self.device)
        reward_term_count = 7 if self.reward_settings.body_state_reward else 5
        if self.reward_settings.action_smoothness_enabled:
            reward_term_count += 1
        reward_terms = torch.zeros(self.num_envs, reward_term_count, device=self.device)
    if self.reward_settings.body_state_reward:
        reward_term_names = (
            "r_body_pos",
            "r_body_rot",
            "r_body_linvel",
            "r_body_angvel",
            "r_com",
            "r_ee",
            "r_ee_rot",
        )
    else:
        reward_term_names = (
            "r_q",
            "r_qd",
            "r_com",
            "r_ee",
            "r_ee_rot",
        )
    if self.reward_settings.action_smoothness_enabled:
        reward_term_names = (*reward_term_names, "r_action_smooth")
    self.reward_runtime_state = RewardRuntimeState(
        reward=reward,
        reward_terms=reward_terms,
        reward_term_names=reward_term_names,
        substep_mean_enabled=self.reward_settings.substep_mean,
        substep_reward_accumulator=torch.zeros_like(reward),
        substep_reward_terms_accumulator=torch.zeros_like(reward_terms),
        substep_reward_sample=torch.zeros_like(reward),
        substep_reward_terms_sample=torch.zeros_like(reward_terms),
        substep_reference_frame_index=torch.zeros_like(self.ref_frame_idx),
    )
    with startup_profiler.phase("action_smoothness"):
        _init_action_smoothness(self)
    self.configure_reward_runtime_buffers(
        reward=reward,
        reward_terms=reward_terms,
        reward_term_names=reward_term_names,
    )
    with startup_profiler.phase("body_state_frame_sigma"):
        _init_body_state_frame_sigma_multiplier(self)
    startup_profiler.log_summary(prefix="REWARD_STARTUP_PROFILE")


def _init_action_smoothness(self) -> None:
    """Bind env.reward.action_smoothness to the controller's per-step command."""
    state = self.reward_runtime_state
    settings = self.reward_settings
    if not settings.action_smoothness_enabled:
        state.action_smoothness_error = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        return
    controller = getattr(self, "fast_muscle", None)
    if controller is None:
        raise ValueError(
            "env.reward.action_smoothness requires a fast-muscle action mode "
            "(torque or mass with policy_output.type pose)"
        )
    try:
        command = controller.get_per_step_command(settings.action_smoothness_source)
    except ValueError as exc:
        raise ValueError(f"env.reward.action_smoothness: {exc}") from exc
    # A torque change over the PD stiffness is the equivalent PD-target change;
    # a PD target change is scored as is.
    gain = (
        float(settings.action_smoothness_torque_gain)
        if command.source == "torque"
        else 1.0
    )
    smoothness = create_action_smoothness_state(
        source=command.source,
        values=command.values,
        column_scale=command.column_scale,
        column_index=command.column_index,
        gain=gain,
    )
    state.action_smoothness = smoothness
    state.action_smoothness_error = smoothness.error
    self.register_reset_state_finalizer(
        lambda reset_mask: reset_action_smoothness_history(smoothness, reset_mask)
    )


def _update_action_smoothness(self) -> None:
    smoothness = self.reward_runtime_state.action_smoothness
    if smoothness is not None:
        update_action_smoothness_error(smoothness)


def _compute_rewards_into(
    self,
    *,
    ref_frame_idx: torch.Tensor,
    reward_r_terms: torch.Tensor | None = None,
    reward: torch.Tensor | None = None,
) -> torch.Tensor:
    state = self.reward_runtime_state
    reward_r_terms_out = (
        state.reward_terms if reward_r_terms is None else reward_r_terms
    )
    reward_out = state.reward if reward is None else reward
    if state.compute_rewards_kernel is None:
        raise RuntimeError("Reward kernel is not initialized.")
    launch_compute_rewards(
        state.compute_rewards_kernel,
        dim=(self.num_envs,),
        inputs=[
            self.mjw_data.qpos,
            self.mjw_data.qvel,
            self.mjw_data.xpos,
            self.mjw_data.xquat,
            self.mjw_data.cvel,
            self.mjw_data.subtree_com,
            self.reward_motion_qpos,
            self.reward_motion_qvel,
            self.reward_reference_body_state,
            self.reward_ref_com,
            self.motion_clip_idx,
            ref_frame_idx,
            self.reward_motion_clip_start,
            self.reward_motion_clip_length,
            self.reward_motion_clip_loop_root_quat,
            self.reward_motion_clip_loop_root_pos,
            self.hinge_qpos_idx_i32,
            self.hinge_dof_idx_i32,
            self.ball_qpos_adr_i32,
            self.ball_dof_adr_i32,
            self.dof_q_scale,
            self.dof_qd_scale,
            self.body_state_frame_sigma_multiplier,
            state.action_smoothness_error,
        ],
        outputs=[
            reward_r_terms_out,
            reward_out,
        ],
        block_dim=REWARD_KERNEL_BLOCK_DIM,
    )
    return reward_out


def prepare_substep_rewards(self) -> None:
    state = self.reward_runtime_state
    state.substep_reward_accumulator.zero_()
    state.substep_reward_terms_accumulator.zero_()


def accumulate_substep_reward(self, substep_idx: int) -> None:
    state = self.reward_runtime_state
    if int(substep_idx) == 0:
        # The step's command is in place once its first substep applied it.
        _update_action_smoothness(self)
    write_substep_reward_ref_frame_idx(
        self.ref_frame_idx,
        state.substep_reference_frame_index,
        decimation=int(self.cfg.decimation),
        substep_idx=int(substep_idx),
    )
    _compute_rewards_into(
        self,
        ref_frame_idx=state.substep_reference_frame_index,
        reward_r_terms=state.substep_reward_terms_sample,
        reward=state.substep_reward_sample,
    )
    state.substep_reward_accumulator.add_(state.substep_reward_sample)
    state.substep_reward_terms_accumulator.add_(state.substep_reward_terms_sample)


def finalize_substep_rewards(self) -> torch.Tensor:
    state = self.reward_runtime_state
    inv_count = 1.0 / float(max(1, int(self.cfg.decimation)))
    state.reward.copy_(state.substep_reward_accumulator)
    state.reward.mul_(inv_count)
    state.reward_terms.copy_(state.substep_reward_terms_accumulator)
    state.reward_terms.mul_(inv_count)
    return state.reward


def get_rewards(self) -> torch.Tensor:
    state = self.reward_runtime_state
    if state.substep_mean_enabled:
        finalize_substep_rewards(self)
    else:
        _update_action_smoothness(self)
        _compute_rewards_into(
            self,
            ref_frame_idx=self.ref_frame_idx,
        )
    nan_debug.nan_debug_record_raw_reward_nonfinite(self)
    self.total_reward = state.reward
    return state.reward


def accumulate_reward_metrics(self) -> None:
    """Account for the completed transition after all termination checks."""
    if self.adaptive_sampling_state.sampling_uses_sonic_statistics:
        wp.launch(
            accumulate_sonic_failure_statistics_kernel,
            dim=(self.num_envs,),
            inputs=[
                self.reset_terminated,
                self.motion_clip_idx,
                self.ref_frame_idx,
                self.motion_clip_length,
                self.motion_clip_unit_start,
                self.motion_clip_unit_count,
                int(self.adaptive_sampling_state.chunk_size),
            ],
            outputs=[
                self.adaptive_sampling_state.sonic_num_failures,
                self.adaptive_sampling_state.sonic_num_exposures,
            ],
        )
    elif self.adaptive_sampling_state.sampling_uses_episode_metrics:
        wp.launch(
            accumulate_episode_reward_sum_kernel,
            dim=(self.num_envs,),
            inputs=[self.reward_buf],
            outputs=[self.adaptive_sampling_state.episode_reward_sum],
        )


def get_reference_tracking_metrics(
    self,
    *,
    frame: tracking_metrics.ReferenceFrame = "root",
    include_root_in_ape: bool = False,
) -> dict[str, torch.Tensor]:
    body_pos = wp.to_torch(self.mjw_data.xpos)
    body_quat = wp.to_torch(self.mjw_data.xquat)
    body_cvel = wp.to_torch(self.mjw_data.cvel)
    body_linvel = body_cvel[:, :, 3:6]
    ape_body_idx = (
        self._tracking_ape_body_idx_including_root
        if include_root_in_ape
        else self._tracking_ape_body_idx_excluding_root
    )
    state = self.reward_runtime_state
    workspace = (
        state.tracking_workspace_including_root
        if include_root_in_ape
        else state.tracking_workspace_excluding_root
    )
    if (
        workspace is None
        or int(workspace.ape.per_body_error.shape[0]) != int(body_pos.shape[0])
        or int(workspace.ape.per_body_error.shape[1]) != int(ape_body_idx.shape[0])
        or int(workspace.ee.per_body_error.shape[1]) != int(self.ee_body_idx.shape[0])
    ):
        workspace = tracking_metrics_warp.create_reference_tracking_metric_workspace(
            env_count=int(body_pos.shape[0]),
            ape_body_count=int(ape_body_idx.shape[0]),
            ee_body_count=int(self.ee_body_idx.shape[0]),
            device=body_pos.device,
            dtype=body_pos.dtype,
        )
        if include_root_in_ape:
            state.tracking_workspace_including_root = workspace
        else:
            state.tracking_workspace_excluding_root = workspace
    tracking_metrics_warp.compute_reference_tracking_metrics_into_warp(
        body_pos=body_pos,
        body_quat=body_quat,
        body_linvel=body_linvel,
        reference_body_pos=self.reference_body_pos,
        reference_body_quat=self.reference_body_quat,
        reference_body_linvel=self.reference_body_linvel,
        ref_frame_idx=self.ref_frame_idx,
        root_body_idx=int(self.root_body_idx),
        ape_body_idx=ape_body_idx,
        ee_body_idx=self.ee_body_idx,
        loop_root_quat=(None if self._multi_motion_enabled else self.loop_root_quat),
        loop_root_pos=(None if self._multi_motion_enabled else self.loop_root_pos),
        motion_clip_idx=(self.motion_clip_idx if self._multi_motion_enabled else None),
        clip_start=(self.motion_clip_start if self._multi_motion_enabled else None),
        clip_length=(self.motion_clip_length if self._multi_motion_enabled else None),
        clip_loop_root_quat=(
            self.motion_clip_loop_root_quat if self._multi_motion_enabled else None
        ),
        clip_loop_root_pos=(
            self.motion_clip_loop_root_pos if self._multi_motion_enabled else None
        ),
        frame=frame,
        workspace=workspace,
    )
    return {
        "ape": workspace.ape.mean_error,
        "ape_per_body": workspace.ape.per_body_error,
        "global_mpjpe": workspace.global_mpjpe.mean_error,
        "global_mpjpe_per_body": workspace.global_mpjpe.per_body_error,
        "ee_error": workspace.ee.mean_error,
        "ee_error_per_body": workspace.ee.per_body_error,
        "tracking_body_pos": workspace.sim_ape_body_pos,
        "reference_body_pos": workspace.ref_ape_body_pos,
        "tracking_root_pos": workspace.sim_root_pos,
        "reference_root_pos": workspace.ref_root_pos,
        "tracking_body_linvel": workspace.sim_tracking_body_linvel,
        "reference_body_linvel": workspace.ref_tracking_body_linvel,
        "sim_root_quat": workspace.sim_root_quat,
        "ref_root_quat": workspace.ref_root_quat,
    }
