"""Runtime wiring for the fixed-shape BeyondMimic Warp hot path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco_warp as mjw
import torch
import warp as wp

from .constants import END_EFFECTOR_BODY_NAMES, TRACKED_BODY_NAMES
from .optimized_kernels import (
    action_kernel,
    build_sampling_cdf_kernel,
    collision_history_kernel,
    command_target_kernel,
    increment_motion_and_mark_end_kernel,
    interval_push_kernel,
    observation_kernel,
    reset_reference_state_kernel,
    reward_kernel,
    sample_motion_frames_kernel,
    termination_kernel,
    update_current_failures_kernel,
    update_failure_ema_kernel,
)
from .reference import ACTION_DIM

_SAMPLING_MODES = {
    "start": 0,
    "uniform": 1,
    "adaptive": 2,
}


@dataclass(frozen=True)
class OptimizedTensorState:
    seed: torch.Tensor
    motion_end_mask: torch.Tensor
    sampling_cdf: torch.Tensor
    joint_qpos_indices: torch.Tensor
    joint_qvel_indices: torch.Tensor
    body_ids: torch.Tensor
    motion_body_indices: torch.Tensor
    end_effector_body_ids: torch.Tensor
    end_effector_tracked_indices: torch.Tensor
    actuator_to_joint_order: torch.Tensor
    reset_position_bounds: torch.Tensor
    reset_euler_bounds: torch.Tensor
    reset_velocity_bounds: torch.Tensor
    reset_joint_bounds: torch.Tensor
    push_interval_bounds: torch.Tensor
    push_velocity_bounds: torch.Tensor


@dataclass(frozen=True)
class OptimizedWarpViewState:
    seed: wp.array
    motion_end_mask: wp.array
    reset_buf: wp.array
    sampling_cdf: wp.array
    joint_qpos_indices: wp.array
    joint_qvel_indices: wp.array
    body_ids: wp.array
    motion_body_indices: wp.array
    end_effector_body_ids: wp.array
    end_effector_tracked_indices: wp.array
    actuator_to_joint_order: wp.array
    reset_position_bounds: wp.array
    reset_euler_bounds: wp.array
    reset_velocity_bounds: wp.array
    reset_joint_bounds: wp.array
    push_interval_bounds: wp.array
    push_velocity_bounds: wp.array
    time_steps: wp.array
    default_joint_position: wp.array
    action_scale: wp.array
    soft_joint_lower: wp.array
    soft_joint_upper: wp.array
    encoder_bias: wp.array
    actions: wp.array
    previous_actions: wp.array
    target_joint_position: wp.array
    target_ctrl: wp.array
    push_time_left: wp.array
    collision_force_history: wp.array
    target_body_position: wp.array
    target_body_quaternion: wp.array
    reward_terms: wp.array
    adaptive_failed: wp.array
    adaptive_current: wp.array
    motion_joint_position: wp.array
    motion_joint_velocity: wp.array
    motion_body_position: wp.array
    motion_body_quaternion: wp.array
    motion_body_linear_velocity: wp.array
    motion_body_angular_velocity: wp.array
    qpos: wp.array
    qvel: wp.array
    ctrl: wp.array
    xpos: wp.array
    xquat: wp.array
    cvel: wp.array
    subtree_com: wp.array
    sensordata: wp.array


@dataclass(frozen=True)
class BeyondMimicOptimizedRuntimeState:
    sampling_mode: int
    tensors: OptimizedTensorState
    views: OptimizedWarpViewState


def _float_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=torch.float32).contiguous()


def initialize(env: Any) -> None:
    mode = str(env.cfg.motion.sampling_mode)
    if mode not in _SAMPLING_MODES:
        raise ValueError(f"Unsupported motion sampling mode {mode!r}")
    if mode == "adaptive" and int(env.cfg.motion.adaptive_kernel_size) != 1:
        raise ValueError(
            "The optimized BeyondMimic path currently requires "
            "env.motion.adaptive_kernel_size=1, matching the public benchmark."
        )

    tensors = OptimizedTensorState(
        seed=torch.tensor(
            [0 if env.cfg.seed is None else int(env.cfg.seed)],
            device=env.device,
            dtype=torch.int32,
        ),
        motion_end_mask=torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        ),
        sampling_cdf=torch.zeros(
            env._adaptive.bin_count,
            device=env.device,
            dtype=torch.float32,
        ),
        joint_qpos_indices=env._joint_qpos_indices.to(dtype=torch.int32).contiguous(),
        joint_qvel_indices=env._joint_qvel_indices.to(dtype=torch.int32).contiguous(),
        body_ids=env._body_ids.to(dtype=torch.int32).contiguous(),
        motion_body_indices=env._motion_body_indices.to(dtype=torch.int32).contiguous(),
        end_effector_body_ids=env._end_effector_body_ids.to(
            dtype=torch.int32
        ).contiguous(),
        end_effector_tracked_indices=torch.tensor(
            [TRACKED_BODY_NAMES.index(name) for name in END_EFFECTOR_BODY_NAMES],
            device=env.device,
            dtype=torch.int32,
        ),
        actuator_to_joint_order=env._actuator_to_joint_order.to(
            dtype=torch.int32
        ).contiguous(),
        reset_position_bounds=_float_tensor(
            env.cfg.reset.position_range, device=env.device
        ),
        reset_euler_bounds=_float_tensor(env.cfg.reset.euler_range, device=env.device),
        reset_velocity_bounds=_float_tensor(
            env.cfg.reset.velocity_range, device=env.device
        ),
        reset_joint_bounds=_float_tensor(
            env.cfg.reset.joint_position_range, device=env.device
        ),
        push_interval_bounds=_float_tensor(
            env.cfg.randomization.push_interval_range_s, device=env.device
        ),
        push_velocity_bounds=_float_tensor(
            env.cfg.randomization.push_velocity_range, device=env.device
        ),
    )
    views = OptimizedWarpViewState(
        seed=wp.from_torch(tensors.seed),
        motion_end_mask=wp.from_torch(tensors.motion_end_mask),
        reset_buf=wp.from_torch(env.reset_buf),
        sampling_cdf=wp.from_torch(tensors.sampling_cdf),
        joint_qpos_indices=wp.from_torch(tensors.joint_qpos_indices),
        joint_qvel_indices=wp.from_torch(tensors.joint_qvel_indices),
        body_ids=wp.from_torch(tensors.body_ids),
        motion_body_indices=wp.from_torch(tensors.motion_body_indices),
        end_effector_body_ids=wp.from_torch(tensors.end_effector_body_ids),
        end_effector_tracked_indices=wp.from_torch(
            tensors.end_effector_tracked_indices
        ),
        actuator_to_joint_order=wp.from_torch(tensors.actuator_to_joint_order),
        reset_position_bounds=wp.from_torch(tensors.reset_position_bounds),
        reset_euler_bounds=wp.from_torch(tensors.reset_euler_bounds),
        reset_velocity_bounds=wp.from_torch(tensors.reset_velocity_bounds),
        reset_joint_bounds=wp.from_torch(tensors.reset_joint_bounds),
        push_interval_bounds=wp.from_torch(tensors.push_interval_bounds),
        push_velocity_bounds=wp.from_torch(tensors.push_velocity_bounds),
        time_steps=wp.from_torch(env._time_steps),
        default_joint_position=wp.from_torch(env._default_joint_position),
        action_scale=wp.from_torch(env._action_scale),
        soft_joint_lower=wp.from_torch(env._soft_joint_lower),
        soft_joint_upper=wp.from_torch(env._soft_joint_upper),
        encoder_bias=wp.from_torch(env._encoder_bias),
        actions=wp.from_torch(env.actions),
        previous_actions=wp.from_torch(env._previous_actions),
        target_joint_position=wp.from_torch(env._target_joint_position),
        target_ctrl=wp.from_torch(env._target_ctrl),
        push_time_left=wp.from_torch(env._push_time_left),
        collision_force_history=wp.from_torch(env._collision_force_history),
        target_body_position=wp.from_torch(env._target_body_position),
        target_body_quaternion=wp.from_torch(env._target_body_quaternion),
        reward_terms=wp.from_torch(env._reward_terms),
        adaptive_failed=wp.from_torch(env._adaptive.bin_failed_count),
        adaptive_current=wp.from_torch(env._adaptive.current_bin_failed),
        motion_joint_position=wp.from_torch(env._motion.joint_position),
        motion_joint_velocity=wp.from_torch(env._motion.joint_velocity),
        motion_body_position=wp.from_torch(env._motion.body_position_world),
        motion_body_quaternion=wp.from_torch(env._motion.body_quaternion_world),
        motion_body_linear_velocity=wp.from_torch(
            env._motion.body_linear_velocity_world
        ),
        motion_body_angular_velocity=wp.from_torch(
            env._motion.body_angular_velocity_world
        ),
        qpos=wp.from_torch(env._qpos),
        qvel=wp.from_torch(env._qvel),
        ctrl=wp.from_torch(env._ctrl),
        xpos=wp.from_torch(env._xpos),
        xquat=wp.from_torch(env._xquat),
        cvel=wp.from_torch(env._cvel),
        subtree_com=wp.from_torch(env._subtree_com),
        sensordata=wp.from_torch(env._sensordata),
    )
    env.optimized_runtime_state = BeyondMimicOptimizedRuntimeState(
        sampling_mode=_SAMPLING_MODES[mode],
        tensors=tensors,
        views=views,
    )


def pre_physics_step(env: Any, incoming_action: torch.Tensor) -> None:
    wp.launch(
        action_kernel,
        dim=env.num_envs,
        inputs=[
            incoming_action,
            env.optimized_runtime_state.views.action_scale,
            env.optimized_runtime_state.views.default_joint_position,
            env.optimized_runtime_state.views.encoder_bias,
            env.optimized_runtime_state.views.actuator_to_joint_order,
            env.optimized_runtime_state.views.previous_actions,
            env.optimized_runtime_state.views.actions,
            env.optimized_runtime_state.views.target_joint_position,
            env.optimized_runtime_state.views.target_ctrl,
            env.optimized_runtime_state.views.ctrl,
            ACTION_DIM,
        ],
    )


def capture_collision_force(env: Any, substep_idx: int) -> None:
    wp.launch(
        collision_history_kernel,
        dim=env.num_envs,
        inputs=[
            env.optimized_runtime_state.views.sensordata,
            env._self_collision_force_slice.start,
            substep_idx,
            env.optimized_runtime_state.views.collision_force_history,
        ],
    )


def get_observations(env: Any) -> dict[str, torch.Tensor]:
    wp.launch(
        observation_kernel,
        dim=env.num_envs,
        inputs=[
            env.optimized_runtime_state.views.seed,
            env.common_step_counter,
            env.optimized_runtime_state.views.time_steps,
            env.optimized_runtime_state.views.motion_joint_position,
            env.optimized_runtime_state.views.motion_joint_velocity,
            env.optimized_runtime_state.views.motion_body_position,
            env.optimized_runtime_state.views.motion_body_quaternion,
            env._motion_anchor_body_index,
            env.optimized_runtime_state.views.body_ids,
            env.optimized_runtime_state.views.joint_qpos_indices,
            env.optimized_runtime_state.views.joint_qvel_indices,
            env._anchor_body_id,
            env._imu_linear_slice.start,
            env._imu_angular_slice.start,
            env.optimized_runtime_state.views.qpos,
            env.optimized_runtime_state.views.qvel,
            env.optimized_runtime_state.views.xpos,
            env.optimized_runtime_state.views.xquat,
            env.optimized_runtime_state.views.sensordata,
            env.optimized_runtime_state.views.default_joint_position,
            env.optimized_runtime_state.views.encoder_bias,
            env.optimized_runtime_state.views.actions,
            int(bool(env.cfg.observations.corruption)),
            env.obs_buf["policy"],
            env.obs_buf["critic"],
            ACTION_DIM,
            len(TRACKED_BODY_NAMES),
        ],
    )
    return env.obs_buf


def get_rewards(env: Any) -> torch.Tensor:
    wp.launch(
        reward_kernel,
        dim=env.num_envs,
        inputs=[
            env.optimized_runtime_state.views.time_steps,
            env.optimized_runtime_state.views.motion_body_position,
            env.optimized_runtime_state.views.motion_body_quaternion,
            env.optimized_runtime_state.views.motion_body_linear_velocity,
            env.optimized_runtime_state.views.motion_body_angular_velocity,
            env.optimized_runtime_state.views.motion_body_indices,
            env._motion_anchor_body_index,
            env.optimized_runtime_state.views.body_ids,
            env._root_body_id,
            env._anchor_body_id,
            env.optimized_runtime_state.views.joint_qpos_indices,
            env.optimized_runtime_state.views.qpos,
            env.optimized_runtime_state.views.xpos,
            env.optimized_runtime_state.views.xquat,
            env.optimized_runtime_state.views.cvel,
            env.optimized_runtime_state.views.subtree_com,
            env.optimized_runtime_state.views.target_body_position,
            env.optimized_runtime_state.views.target_body_quaternion,
            env.optimized_runtime_state.views.actions,
            env.optimized_runtime_state.views.previous_actions,
            env.optimized_runtime_state.views.soft_joint_lower,
            env.optimized_runtime_state.views.soft_joint_upper,
            env.optimized_runtime_state.views.collision_force_history,
            env.optimized_runtime_state.views.reward_terms,
            env.reward_buf,
            ACTION_DIM,
            len(TRACKED_BODY_NAMES),
            env.cfg.decimation,
            env.step_dt,
        ],
    )
    return env.reward_buf


def get_dones(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    wp.launch(
        termination_kernel,
        dim=env.num_envs,
        inputs=[
            env.episode_length_buf,
            env.max_episode_length,
            env.optimized_runtime_state.views.time_steps,
            env.optimized_runtime_state.views.motion_body_position,
            env.optimized_runtime_state.views.motion_body_quaternion,
            env._motion_anchor_body_index,
            env._anchor_body_id,
            env.optimized_runtime_state.views.end_effector_body_ids,
            env.optimized_runtime_state.views.end_effector_tracked_indices,
            env.optimized_runtime_state.views.xpos,
            env.optimized_runtime_state.views.xquat,
            env.optimized_runtime_state.views.target_body_position,
            env.reset_terminated,
            env.reset_time_outs,
            env._done_reason_mask_buf,
            int(env._track_done_reasons),
            len(END_EFFECTOR_BODY_NAMES),
        ],
    )
    return env.reset_terminated, env.reset_time_outs


def _accumulate_failures(env: Any, sample_mask: torch.Tensor) -> None:
    if env.optimized_runtime_state.sampling_mode != _SAMPLING_MODES["adaptive"]:
        return
    wp.launch(
        update_current_failures_kernel,
        dim=env._adaptive.bin_count,
        inputs=[
            sample_mask,
            env.reset_terminated,
            env.optimized_runtime_state.views.time_steps,
            env.num_envs,
            env._motion.frame_count,
            env._adaptive.bin_count,
            env.optimized_runtime_state.views.adaptive_current,
        ],
    )


def _sample_motion_frames(
    env: Any,
    sample_mask: torch.Tensor,
    *,
    salt: int,
) -> None:
    if env.optimized_runtime_state.sampling_mode == _SAMPLING_MODES["adaptive"]:
        wp.launch(
            build_sampling_cdf_kernel,
            dim=1,
            inputs=[
                env.optimized_runtime_state.views.adaptive_failed,
                float(env.cfg.motion.adaptive_uniform_ratio),
                env._adaptive.bin_count,
                env.optimized_runtime_state.views.sampling_cdf,
            ],
        )
    wp.launch(
        sample_motion_frames_kernel,
        dim=env.num_envs,
        inputs=[
            sample_mask,
            env.optimized_runtime_state.views.seed,
            env.common_step_counter,
            salt,
            env.optimized_runtime_state.sampling_mode,
            env.optimized_runtime_state.views.sampling_cdf,
            env._motion.frame_count,
            env._adaptive.bin_count,
            env.optimized_runtime_state.views.time_steps,
        ],
    )


def _write_reference_state(
    env: Any,
    reset_mask: torch.Tensor,
    *,
    salt: int,
    reset_episode_state: bool,
) -> None:
    wp.launch(
        reset_reference_state_kernel,
        dim=env.num_envs,
        inputs=[
            reset_mask,
            env.optimized_runtime_state.views.seed,
            env.common_step_counter,
            salt,
            env.optimized_runtime_state.views.time_steps,
            env.optimized_runtime_state.views.motion_joint_position,
            env.optimized_runtime_state.views.motion_joint_velocity,
            env.optimized_runtime_state.views.motion_body_position,
            env.optimized_runtime_state.views.motion_body_quaternion,
            env.optimized_runtime_state.views.motion_body_linear_velocity,
            env.optimized_runtime_state.views.motion_body_angular_velocity,
            env._motion_root_body_index,
            env.optimized_runtime_state.views.joint_qpos_indices,
            env.optimized_runtime_state.views.joint_qvel_indices,
            env.optimized_runtime_state.views.reset_position_bounds,
            env.optimized_runtime_state.views.reset_euler_bounds,
            env.optimized_runtime_state.views.reset_velocity_bounds,
            env.optimized_runtime_state.views.reset_joint_bounds,
            env.optimized_runtime_state.views.soft_joint_lower,
            env.optimized_runtime_state.views.soft_joint_upper,
            int(reset_episode_state),
            env.optimized_runtime_state.views.qpos,
            env.optimized_runtime_state.views.qvel,
            env.optimized_runtime_state.views.actions,
            env.optimized_runtime_state.views.previous_actions,
            env.optimized_runtime_state.views.target_joint_position,
            env.optimized_runtime_state.views.target_ctrl,
            env.optimized_runtime_state.views.collision_force_history,
            ACTION_DIM,
            env.cfg.decimation,
        ],
    )


def reset_idx(env: Any, reset_mask: torch.Tensor) -> None:
    # Base reset is already a fixed-size Warp launch.
    from instantmimic.envs.direct_rl_env import DirectRLEnv

    reset_mask_wp = (
        env.optimized_runtime_state.views.reset_buf
        if reset_mask is env.reset_buf
        else wp.from_torch(reset_mask)
    )
    DirectRLEnv._reset_idx(env, reset_mask_wp)
    mjw.reset_data(env.mjw_model, env.mjw_data, reset_mask_wp)
    _accumulate_failures(env, reset_mask_wp)
    _sample_motion_frames(env, reset_mask_wp, salt=2003)
    _write_reference_state(
        env,
        reset_mask_wp,
        salt=2011,
        reset_episode_state=True,
    )

    wp.launch(
        increment_motion_and_mark_end_kernel,
        dim=env.num_envs,
        inputs=[
            env.optimized_runtime_state.views.time_steps,
            env._motion.frame_count,
            env.optimized_runtime_state.views.motion_end_mask,
        ],
    )
    _accumulate_failures(env, env.optimized_runtime_state.views.motion_end_mask)
    _sample_motion_frames(
        env,
        env.optimized_runtime_state.views.motion_end_mask,
        salt=2027,
    )
    _write_reference_state(
        env,
        env.optimized_runtime_state.views.motion_end_mask,
        salt=2029,
        reset_episode_state=False,
    )

    if env.optimized_runtime_state.sampling_mode == _SAMPLING_MODES["adaptive"]:
        wp.launch(
            update_failure_ema_kernel,
            dim=env._adaptive.bin_count,
            inputs=[
                env.optimized_runtime_state.views.adaptive_failed,
                env.optimized_runtime_state.views.adaptive_current,
                float(env.cfg.motion.adaptive_alpha),
                env._adaptive.bin_count,
            ],
        )


def finalize_reset_state(env: Any) -> bool:
    """Finish optimized reset work that requires a fresh state prefix."""
    wp.launch(
        command_target_kernel,
        dim=env.num_envs,
        inputs=[
            env.optimized_runtime_state.views.time_steps,
            env.optimized_runtime_state.views.motion_body_position,
            env.optimized_runtime_state.views.motion_body_quaternion,
            env.optimized_runtime_state.views.motion_body_indices,
            env._motion_anchor_body_index,
            env._anchor_body_id,
            env.optimized_runtime_state.views.xpos,
            env.optimized_runtime_state.views.xquat,
            env.optimized_runtime_state.views.target_body_position,
            env.optimized_runtime_state.views.target_body_quaternion,
            len(TRACKED_BODY_NAMES),
        ],
    )
    # `_initial_reset_complete` belongs to the task and is shared with the
    # ordinary reset path; optimized runtime state must not reset this flag.
    if env._initial_reset_complete:
        pushes_mutate_state = bool(
            env.cfg.randomization.enabled and env.cfg.randomization.push_enabled
        )
        apply_interval_pushes(env)
        return pushes_mutate_state
    else:
        env._initial_reset_complete = True
        return False


def apply_interval_pushes(env: Any) -> None:
    if not bool(env.cfg.randomization.enabled) or not bool(
        env.cfg.randomization.push_enabled
    ):
        return
    wp.launch(
        interval_push_kernel,
        dim=env.num_envs,
        inputs=[
            env.optimized_runtime_state.views.seed,
            env.common_step_counter,
            env.optimized_runtime_state.views.push_interval_bounds,
            env.optimized_runtime_state.views.push_velocity_bounds,
            env._root_body_id,
            env.step_dt,
            env.optimized_runtime_state.views.qpos,
            env.optimized_runtime_state.views.qvel,
            env.optimized_runtime_state.views.xpos,
            env.optimized_runtime_state.views.cvel,
            env.optimized_runtime_state.views.subtree_com,
            env.optimized_runtime_state.views.push_time_left,
        ],
    )
