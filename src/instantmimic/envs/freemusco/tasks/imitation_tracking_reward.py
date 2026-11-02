from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from instantmimic.envs.tracking.reward_config import (
    joint_root_error_weight,
    resolve_ee_position_layout,
    resolve_ee_rotation_body_mask,
    resolve_joint_dof_scales,
    resolve_reward_body_weight,
    resolve_reward_settings,
)
from instantmimic.envs.tracking.reward_kernels import (
    RewardKernelSpec,
    make_compute_rewards_kernel,
)
from instantmimic.envs.tracking.reward_kernels import (
    build_reward_dof_scales as build_snu_reward_dof_scales,
)


@dataclass
class JointRewardMetadata:
    joint_dof_mask: torch.Tensor
    joint_qpos_mask: torch.Tensor
    hinge_qpos_indices: torch.Tensor
    hinge_dof_indices: torch.Tensor
    hinge_body_indices: torch.Tensor
    hinge_parent_body_indices: torch.Tensor
    hinge_multi_body_mask: torch.Tensor
    hinge_axis_local: torch.Tensor
    ball_qpos_gather: torch.Tensor
    ball_dof_scatter: torch.Tensor
    n_ball_joints: int
    talus_dof_indices: torch.Tensor
    toe_dof_indices: torch.Tensor
    body_parent_indices: torch.Tensor


def build_joint_reward_metadata(
    *,
    model: mujoco.MjModel,
    device: torch.device,
    talus_body_names: Sequence[str] = (),
    toe_body_names: Sequence[str] = (),
) -> JointRewardMetadata:
    ball_qpos_gather: list[int] = []
    ball_dof_scatter: list[int] = []
    talus_dof_local: list[int] = []
    toe_dof_local: list[int] = []
    hinge_qpos_indices: list[int] = []
    hinge_dof_indices: list[int] = []
    hinge_body_indices: list[int] = []
    hinge_parent_body_indices: list[int] = []
    hinge_axis_local: list[tuple[float, float, float]] = []
    hinge_body_counts: dict[int, int] = {}
    n_ball = 0

    joint_dof_mask = torch.zeros(model.nv - 6, device=device, dtype=torch.bool)
    joint_qpos_mask = torch.zeros(model.nq - 7, device=device, dtype=torch.bool)
    talus_name_set = set(str(name) for name in talus_body_names)
    toe_name_set = set(str(name) for name in toe_body_names)

    for jnt_id in range(model.njnt):
        jnt_type = model.jnt_type[jnt_id]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            continue

        qpos_adr = int(model.jnt_qposadr[jnt_id])
        dof_adr = int(model.jnt_dofadr[jnt_id])
        body_id = int(model.jnt_bodyid[jnt_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        local_dof = dof_adr - 6
        local_qpos = qpos_adr - 7

        if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            ball_qpos_gather.extend(
                [qpos_adr, qpos_adr + 1, qpos_adr + 2, qpos_adr + 3]
            )
            ball_dof_scatter.extend([local_dof, local_dof + 1, local_dof + 2])
            joint_dof_mask[local_dof : local_dof + 3] = True
            joint_qpos_mask[local_qpos : local_qpos + 4] = True
            n_ball += 1
            if body_name in talus_name_set:
                talus_dof_local.extend([local_dof, local_dof + 1, local_dof + 2])
            if body_name in toe_name_set:
                toe_dof_local.extend([local_dof, local_dof + 1, local_dof + 2])
            continue

        if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
            joint_dof_mask[local_dof] = True
            joint_qpos_mask[local_qpos] = True
            hinge_qpos_indices.append(qpos_adr)
            hinge_dof_indices.append(local_dof)
            hinge_body_indices.append(body_id)
            hinge_parent_body_indices.append(int(model.body_parentid[body_id]))
            hinge_axis_local.append(
                tuple(float(v) for v in model.jnt_axis[jnt_id].tolist())
            )
            hinge_body_counts[body_id] = hinge_body_counts.get(body_id, 0) + 1
            if body_name in talus_name_set:
                talus_dof_local.append(local_dof)
            if body_name in toe_name_set:
                toe_dof_local.append(local_dof)

    return JointRewardMetadata(
        joint_dof_mask=joint_dof_mask,
        joint_qpos_mask=joint_qpos_mask,
        hinge_qpos_indices=torch.tensor(
            hinge_qpos_indices, device=device, dtype=torch.long
        ),
        hinge_dof_indices=torch.tensor(
            hinge_dof_indices, device=device, dtype=torch.long
        ),
        hinge_body_indices=torch.tensor(
            hinge_body_indices, device=device, dtype=torch.long
        ),
        hinge_parent_body_indices=torch.tensor(
            hinge_parent_body_indices, device=device, dtype=torch.long
        ),
        hinge_multi_body_mask=torch.tensor(
            [hinge_body_counts.get(body_id, 0) > 1 for body_id in hinge_body_indices],
            device=device,
            dtype=torch.bool,
        ),
        hinge_axis_local=(
            torch.tensor(hinge_axis_local, device=device, dtype=torch.float32)
            if hinge_axis_local
            else torch.zeros((0, 3), device=device, dtype=torch.float32)
        ),
        ball_qpos_gather=torch.tensor(
            ball_qpos_gather, device=device, dtype=torch.long
        ),
        ball_dof_scatter=torch.tensor(
            ball_dof_scatter, device=device, dtype=torch.long
        ),
        n_ball_joints=n_ball,
        talus_dof_indices=torch.tensor(
            talus_dof_local, device=device, dtype=torch.long
        ),
        toe_dof_indices=torch.tensor(toe_dof_local, device=device, dtype=torch.long),
        body_parent_indices=torch.tensor(
            model.body_parentid.tolist(), device=device, dtype=torch.long
        ),
    )


def build_reward_body_indices(
    *,
    model: mujoco.MjModel,
    motion_body_count: int,
    device: torch.device,
) -> torch.Tensor:
    body_start = int(model.nbody - motion_body_count)
    return torch.arange(body_start, model.nbody, device=device, dtype=torch.long)


def build_tracking_reward_kernel(
    *,
    cfg,
    model: mujoco.MjModel,
    device: torch.device,
    root_body_idx: int,
    ee_body_idx: torch.Tensor,
    reward_body_idx: torch.Tensor,
    joint_meta: JointRewardMetadata,
    loop_root_quat: np.ndarray,
    loop_root_translation: np.ndarray,
    motion_frame_count: int,
):
    settings = resolve_reward_settings(cfg)
    if settings.action_smoothness_enabled:
        raise ValueError(
            "env.reward.action_smoothness is not supported by the freemusco "
            "imitation family; it is implemented for the SNU imitation family"
        )
    ee_body_idx_tuple = tuple(int(v) for v in ee_body_idx.detach().cpu().tolist())
    ee_position_layout = resolve_ee_position_layout(settings, model, ee_body_idx_tuple)
    root_body_name = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(root_body_idx)) or ""
    )
    spec = RewardKernelSpec(
        motion_frame_count=int(motion_frame_count),
        n_hinge=int(joint_meta.hinge_qpos_indices.numel()),
        n_ball=int(joint_meta.n_ball_joints),
        nv=int(model.nv),
        root_body_idx=int(root_body_idx),
        ee_body_idx=ee_body_idx_tuple,
        reward_body_idx=tuple(int(v) for v in reward_body_idx.detach().cpu().tolist()),
        pose_w=settings.pose_w,
        vel_w=settings.vel_w,
        com_w=settings.com_w,
        ee_w=settings.ee_w,
        ee_rot_w=settings.ee_rot_w,
        alive_reward=settings.alive_reward,
        reward_combination_mode=settings.combination_mode,
        body_state_reward=settings.body_state_reward,
        body_state_reward_frame_mode=settings.body_state_frame_mode,
        joint_pose_reward_mode=settings.joint_pose_mode,
        joint_velocity_reward_mode=settings.joint_velocity_mode,
        body_parent_idx=tuple(
            int(v) for v in joint_meta.body_parent_indices.detach().cpu().tolist()
        ),
        hinge_body_idx=tuple(
            int(v) for v in joint_meta.hinge_body_indices.detach().cpu().tolist()
        ),
        hinge_parent_body_idx=tuple(
            int(v) for v in joint_meta.hinge_parent_body_indices.detach().cpu().tolist()
        ),
        hinge_axis_local=tuple(
            tuple(float(c) for c in row)
            for row in joint_meta.hinge_axis_local.detach().cpu().tolist()
        ),
        hinge_multi_body_mask=tuple(
            int(v) for v in joint_meta.hinge_multi_body_mask.detach().cpu().tolist()
        ),
        mix_w_pose=settings.mix_w_pose,
        mix_w_vel=settings.mix_w_vel,
        mix_w_body_pos=settings.mix_w_body_pos,
        mix_w_body_rot=settings.mix_w_body_rot,
        mix_w_body_linvel=settings.mix_w_body_linvel,
        mix_w_body_angvel=settings.mix_w_body_angvel,
        mix_w_com=settings.mix_w_com,
        mix_w_ee_pos=settings.mix_w_ee_pos,
        mix_w_ee_rot=settings.mix_w_ee_rot,
        pose_include_root=settings.pose_include_root,
        vel_include_root=settings.velocity_include_root,
        ee_pos_root_relative=settings.ee_pos_root_relative,
        velocity_reward_seam_fade_frames=settings.velocity_seam_fade_frames,
        vel_valid_all=bool(
            cfg.get("loop_boundary_vel_from_first_frame", False)
            or cfg.get("loop_boundary_vel_extrapolate", False)
            or cfg.get("smooth_cycle", False)
        ),
        loop_root_q=tuple(float(v) for v in loop_root_quat.tolist()),
        loop_root_t=tuple(float(v) for v in loop_root_translation.tolist()),
        reward_power_q=settings.power_q,
        reward_power_beta=settings.power_beta,
        reward_power_epsilon=settings.power_epsilon,
        body_state_pos_reward_weight=settings.body_state_pos_w,
        body_state_rot_reward_weight=settings.body_state_rot_w,
        body_state_linvel_reward_weight=settings.body_state_linvel_w,
        body_state_angvel_reward_weight=settings.body_state_angvel_w,
        body_state_frame_sigma_enabled=settings.frame_sigma is not None,
        reward_body_weight=resolve_reward_body_weight(
            settings,
            model,
            tuple(int(v) for v in reward_body_idx.detach().cpu().tolist()),
        ),
        reward_scale=settings.combination_scale,
        pose_root_error_scale=joint_root_error_weight(
            settings.pose_body_weights, root_body_name
        ),
        vel_root_error_scale=joint_root_error_weight(
            settings.velocity_body_weights, root_body_name
        ),
        ee_rot_body_mask=resolve_ee_rotation_body_mask(
            settings,
            model,
            tuple(int(v) for v in ee_body_idx.detach().cpu().tolist()),
        ),
        ee_rot_mean_per_body=settings.ee_rotation_mean == "body",
        ee_pos_body_idx=ee_position_layout.body_idx,
        ee_pos_body_offset=ee_position_layout.body_offset,
        ee_pos_anchor_body_idx=ee_position_layout.anchor_body_idx,
        ee_pos_anchor_offset=ee_position_layout.anchor_offset,
    )
    return make_compute_rewards_kernel(spec)


def build_reward_dof_scales(
    *,
    joint_meta: JointRewardMetadata,
    cfg,
    model: mujoco.MjModel,
    root_body_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dof_q_scale, dof_qd_scale = build_snu_reward_dof_scales(
        joint_dof_mask=joint_meta.joint_dof_mask,
        talus_dof_indices=joint_meta.talus_dof_indices,
        toe_dof_indices=joint_meta.toe_dof_indices,
    )
    settings = resolve_reward_settings(cfg)
    # The joint metadata covers every non-free joint after a 6-DOF root.
    joint_ids = tuple(
        jnt_id
        for jnt_id in range(model.njnt)
        if int(model.jnt_type[jnt_id]) != int(mujoco.mjtJoint.mjJNT_FREE)
    )
    tracked = tuple(bool(v) for v in joint_meta.joint_dof_mask.tolist())
    scales = []
    for term, body_weights, default in (
        ("pose", settings.pose_body_weights, dof_q_scale),
        ("velocity", settings.velocity_body_weights, dof_qd_scale),
    ):
        if body_weights is None:
            scales.append(default)
            continue
        values = resolve_joint_dof_scales(
            body_weights,
            model,
            joint_ids=joint_ids,
            root_body_idx=int(root_body_idx),
            root_dof_count=6,
            tracked_dof_mask=tracked,
            key=f"env.reward.tracking.{term}.body_weights",
        )
        scales.append(torch.tensor(values, device=default.device, dtype=torch.float32))
    return scales[0], scales[1]
