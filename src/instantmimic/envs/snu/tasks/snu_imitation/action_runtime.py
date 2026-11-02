from __future__ import annotations

import mujoco
import torch
import warp as wp

from instantmimic.envs.tracking.reward_kernels import (
    quat_diff_to_axis_angle,
)
from instantmimic.motion.warp_utils import clip_global_frame

__all__ = (
    "compute_motion_ctrl",
    "apply_ctrl_action",
)


@wp.kernel(enable_backward=False)
def _apply_action_kernel(
    joint_range: wp.array(dtype=wp.vec2),
    actions: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
):
    env_idx, joint_idx = wp.tid()
    range = joint_range[joint_idx]
    action = (actions[env_idx, joint_idx] + 1.0) / 2.0
    ctrl[env_idx, joint_idx] = wp.lerp(range[0], range[1], action)


@wp.kernel(enable_backward=False)
def _apply_action_ref_kernel(
    joint_range: wp.array(dtype=wp.vec2),
    motion_ctrl: wp.array2d(dtype=wp.float32),
    ref_frame_idx: wp.array(dtype=int),
    motion_frame_count: int,
    actions: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
):
    env_idx, joint_idx = wp.tid()
    r = joint_range[joint_idx]
    fi = ref_frame_idx[env_idx] % motion_frame_count
    half_range = 0.5 * (r[1] - r[0])
    target = motion_ctrl[fi, joint_idx] + actions[env_idx, joint_idx] * half_range
    ctrl[env_idx, joint_idx] = wp.clamp(target, r[0], r[1])


@wp.kernel(enable_backward=False)
def _apply_action_ref_multi_kernel(
    joint_range: wp.array(dtype=wp.vec2),
    motion_ctrl: wp.array2d(dtype=wp.float32),
    motion_clip_idx: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    clip_start: wp.array(dtype=int),
    clip_length: wp.array(dtype=int),
    actions: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
):
    env_idx, joint_idx = wp.tid()
    clip_idx = motion_clip_idx[env_idx]
    fi = clip_global_frame(
        ref_frame_idx[env_idx],
        clip_idx,
        clip_start,
        clip_length,
    )
    r = joint_range[joint_idx]
    half_range = 0.5 * (r[1] - r[0])
    target = motion_ctrl[fi, joint_idx] + actions[env_idx, joint_idx] * half_range
    ctrl[env_idx, joint_idx] = wp.clamp(target, r[0], r[1])


def compute_motion_ctrl(env):
    num_frames = env.motion_qpos.shape[0]
    ctrl = torch.zeros(num_frames, env.model.nu, device=env.device)
    qpos0 = torch.tensor(env.model.qpos0, device=env.device, dtype=torch.float32)

    for i in range(env.model.nu):
        joint_id = env.model.actuator_trnid[i, 0]
        jnt_type = env.model.jnt_type[joint_id]
        qpos_addr = env.model.jnt_qposadr[joint_id]

        if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
            ctrl[:, i] = env.motion_qpos[:, qpos_addr] - qpos0[qpos_addr]
        elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            q = env.motion_qpos[:, qpos_addr : qpos_addr + 4]
            q0 = qpos0[qpos_addr : qpos_addr + 4].unsqueeze(0).expand(num_frames, -1)
            axis_angle = quat_diff_to_axis_angle(q0, q)
            gear_xyz = torch.tensor(
                env.model.actuator_gear[i, :3],
                device=env.device,
                dtype=torch.float32,
            )
            ctrl[:, i] = (axis_angle * gear_xyz.unsqueeze(0)).sum(dim=1)

    return ctrl


def apply_ctrl_action(env, *, use_ref_action: bool) -> None:
    if use_ref_action:
        if env._multi_motion_enabled:
            wp.launch(
                _apply_action_ref_multi_kernel,
                dim=(env.num_envs, env.model.nu),
                inputs=[
                    env.joint_range,
                    env.motion_ctrl,
                    env.motion_clip_idx,
                    env.ref_frame_idx,
                    env.motion_clip_start,
                    env.motion_clip_length,
                    env.actions,
                ],
                outputs=[env.mjw_data.ctrl],
            )
        else:
            wp.launch(
                _apply_action_ref_kernel,
                dim=(env.num_envs, env.model.nu),
                inputs=[
                    env.joint_range,
                    env.motion_ctrl,
                    env.ref_frame_idx,
                    env.motion_frame_count,
                    env.actions,
                ],
                outputs=[env.mjw_data.ctrl],
            )
    else:
        wp.launch(
            _apply_action_kernel,
            dim=(env.num_envs, env.model.nu),
            inputs=[env.joint_range, env.actions],
            outputs=[env.mjw_data.ctrl],
        )
