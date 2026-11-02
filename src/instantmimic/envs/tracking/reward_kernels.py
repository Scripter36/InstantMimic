from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

from instantmimic.envs.config_access import get_value
from instantmimic.envs.tracking import metrics as tracking_metrics
from instantmimic.envs.tracking.warp_features import (
    local_mjw_quat as local_quat_wp,
)
from instantmimic.envs.tracking.warp_features import (
    read_quat_from_2d,
    read_vec3_from_2d,
)
from instantmimic.envs.tracking.warp_features import (
    safe_quat_sub_mjw as safe_quat_sub_wp,
)
from instantmimic.motion.warp_math import normalize_mjw_quat as normalize_quat_wp
from instantmimic.motion.warp_utils import (
    apply_loop_rotation_vec,
    apply_loop_transform_quat,
    apply_loop_transform_vec,
    clip_cycle_count,
    clip_local_frame,
)
from instantmimic.utils.mjw_kernels.private import mul_quat, quat_inv, rot_vec_quat
from instantmimic.utils.warp_kernels import expose_kernel_name

_REWARD_KERNEL_CACHE: dict[tuple[Any, ...], Any] = {}
REWARD_KERNEL_BLOCK_DIM = 32


@dataclass(frozen=True)
class RewardKernelSpec:
    motion_frame_count: int
    n_hinge: int
    n_ball: int
    nv: int
    root_body_idx: int
    ee_body_idx: tuple[int, ...]
    reward_body_idx: tuple[int, ...]
    pose_w: float
    vel_w: float
    com_w: float
    ee_w: float
    ee_rot_w: float
    alive_reward: float
    reward_combination_mode: int
    body_state_reward: bool
    body_state_reward_frame_mode: int
    joint_pose_reward_mode: int
    joint_velocity_reward_mode: int
    body_parent_idx: tuple[int, ...]
    hinge_body_idx: tuple[int, ...]
    hinge_parent_body_idx: tuple[int, ...]
    hinge_axis_local: tuple[tuple[float, float, float], ...]
    hinge_multi_body_mask: tuple[int, ...]
    mix_w_pose: float
    mix_w_vel: float
    mix_w_body_pos: float
    mix_w_body_rot: float
    mix_w_body_linvel: float
    mix_w_body_angvel: float
    mix_w_com: float
    mix_w_ee_pos: float
    mix_w_ee_rot: float
    pose_include_root: bool
    vel_include_root: bool
    ee_pos_root_relative: bool
    velocity_reward_seam_fade_frames: int
    vel_valid_all: bool
    loop_root_q: tuple[float, float, float, float]
    loop_root_t: tuple[float, float, float]
    root_qpos_count: int = 7
    root_dof_count: int = 6
    # Passive DOFs (instantmimic.envs.dof_roles) inside [root, nv): their
    # dof_q/qd scales are zero and the joint means divide by nv - passive.
    passive_dof_count: int = 0
    reward_power_q: float = 1.0
    reward_power_beta: float = 1.0
    reward_power_epsilon: float = 1.0e-6
    body_state_pos_reward_weight: float | None = None
    body_state_rot_reward_weight: float | None = None
    body_state_linvel_reward_weight: float | None = None
    body_state_angvel_reward_weight: float | None = None
    body_state_frame_sigma_enabled: bool = False
    # Per reward body (aligned with reward_body_idx); None weighs every body 1.
    reward_body_weight: tuple[float, ...] | None = None
    # Factor on the combined tracking reward, applied in every combination mode.
    reward_scale: float = 1.0
    # Multipliers on the squared root errors added by pose/velocity include_root.
    pose_root_error_scale: float = 2.0
    vel_root_error_scale: float = 2.0
    # Per end effector (aligned with ee_body_idx): 1 scores its rotation, 0
    # skips it. None scores every end effector.
    ee_rot_body_mask: tuple[int, ...] | None = None
    # False: the rotation error is a mean over the 3 rotation-vector components
    # of the scored bodies; True: a mean of the squared angle over the bodies.
    ee_rot_mean_per_body: bool = False
    # Bodies the end-effector position term scores, with the body-frame point
    # compared per body (zeros: the body origin; body_ipos: its COM). None
    # scores the end effectors (ee_body_idx) at their origins.
    ee_pos_body_idx: tuple[int, ...] | None = None
    ee_pos_body_offset: tuple[tuple[float, float, float], ...] | None = None
    # Anchor body subtracted from the scored positions on both sides, axes
    # world-aligned; -1 compares world (or root-relative) positions.
    ee_pos_anchor_body_idx: int = -1
    ee_pos_anchor_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Action-smoothness term exp(-weight * action_smoothness_error[env]); it
    # joins the combination like the other terms and is reported after them.
    action_smoothness_enabled: bool = False
    action_smoothness_weight: float = 0.0
    mix_w_action_smoothness: float = 1.0

    def cache_key(self) -> tuple[Any, ...]:
        return (
            int(self.motion_frame_count),
            int(self.n_hinge),
            int(self.n_ball),
            int(self.nv),
            int(self.root_body_idx),
            tuple(int(v) for v in self.ee_body_idx),
            tuple(int(v) for v in self.reward_body_idx),
            float(self.pose_w),
            float(self.vel_w),
            float(self.com_w),
            float(self.ee_w),
            float(self.ee_rot_w),
            float(self.alive_reward),
            int(self.reward_combination_mode),
            bool(self.body_state_reward),
            int(self.body_state_reward_frame_mode),
            int(self.joint_pose_reward_mode),
            int(self.joint_velocity_reward_mode),
            tuple(int(v) for v in self.body_parent_idx),
            tuple(int(v) for v in self.hinge_body_idx),
            tuple(int(v) for v in self.hinge_parent_body_idx),
            tuple(tuple(float(c) for c in row) for row in self.hinge_axis_local),
            tuple(int(v) for v in self.hinge_multi_body_mask),
            float(self.mix_w_pose),
            float(self.mix_w_vel),
            float(self.mix_w_body_pos),
            float(self.mix_w_body_rot),
            float(self.mix_w_body_linvel),
            float(self.mix_w_body_angvel),
            float(self.mix_w_com),
            float(self.mix_w_ee_pos),
            float(self.mix_w_ee_rot),
            bool(self.pose_include_root),
            bool(self.vel_include_root),
            bool(self.ee_pos_root_relative),
            int(self.velocity_reward_seam_fade_frames),
            bool(self.vel_valid_all),
            tuple(float(v) for v in self.loop_root_q),
            tuple(float(v) for v in self.loop_root_t),
            int(self.root_qpos_count),
            int(self.root_dof_count),
            int(self.passive_dof_count),
            float(self.reward_power_q),
            float(self.reward_power_beta),
            float(self.reward_power_epsilon),
            (
                None
                if self.body_state_pos_reward_weight is None
                else float(self.body_state_pos_reward_weight)
            ),
            (
                None
                if self.body_state_rot_reward_weight is None
                else float(self.body_state_rot_reward_weight)
            ),
            (
                None
                if self.body_state_linvel_reward_weight is None
                else float(self.body_state_linvel_reward_weight)
            ),
            (
                None
                if self.body_state_angvel_reward_weight is None
                else float(self.body_state_angvel_reward_weight)
            ),
            bool(self.body_state_frame_sigma_enabled),
            (
                None
                if self.reward_body_weight is None
                else tuple(float(v) for v in self.reward_body_weight)
            ),
            float(self.reward_scale),
            float(self.pose_root_error_scale),
            float(self.vel_root_error_scale),
            (
                None
                if self.ee_rot_body_mask is None
                else tuple(int(bool(v)) for v in self.ee_rot_body_mask)
            ),
            bool(self.ee_rot_mean_per_body),
            (
                None
                if self.ee_pos_body_idx is None
                else tuple(int(v) for v in self.ee_pos_body_idx)
            ),
            (
                None
                if self.ee_pos_body_offset is None
                else tuple(
                    tuple(float(c) for c in row) for row in self.ee_pos_body_offset
                )
            ),
            int(self.ee_pos_anchor_body_idx),
            tuple(float(c) for c in self.ee_pos_anchor_offset),
            bool(self.action_smoothness_enabled),
            float(self.action_smoothness_weight),
            float(self.mix_w_action_smoothness),
        )


@dataclass(frozen=True)
class RewardKernelScalarWeights:
    pose_w: float
    vel_w: float
    com_w: float
    ee_w: float
    ee_rot_w: float
    alive_reward: float


def resolve_reward_kernel_scalar_weights(settings: Any) -> RewardKernelScalarWeights:
    return RewardKernelScalarWeights(
        pose_w=float(get_value(settings, "pose_w", 0.0)),
        vel_w=float(get_value(settings, "vel_w", 0.0)),
        com_w=float(get_value(settings, "com_w", 0.0)),
        ee_w=float(get_value(settings, "ee_w", 0.0)),
        ee_rot_w=float(get_value(settings, "ee_rot_w", 0.0)),
        alive_reward=float(get_value(settings, "alive_reward", 0.0)),
    )


def resolve_reward_combination_mode(mode: str) -> int:
    mode_key = str(mode).lower()
    if mode_key == "weighted_sum":
        return 1
    if mode_key == "raw_sum":
        return 2
    if mode_key == "power_sum":
        return 3
    if mode_key == "product":
        return 0
    raise ValueError(
        "Unsupported reward_combination "
        f"{mode!r}. Valid values: product, raw_sum, weighted_sum, power_sum."
    )


def resolve_body_state_reward_frame_mode(mode: str) -> int:
    return 1 if str(mode).lower() == "parent_local" else 0


def root_position_uses_qpos(root_qpos_count: int) -> bool:
    """Return whether the model exposes its root position as a free-joint qpos."""
    return int(root_qpos_count) == 7


def build_reward_dof_scales(
    *,
    joint_dof_mask: torch.Tensor,
    talus_dof_indices: torch.Tensor,
    toe_dof_indices: torch.Tensor,
    q_talus_scale: float = 0.5,
    q_toe_scale: float = 0.01,
    qd_talus_scale: float = 0.5,
    qd_toe_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Built-in per-DOF difference scales (the kernel squares scaled differences)."""
    scales = []
    for talus_scale, toe_scale in (
        (q_talus_scale, q_toe_scale),
        (qd_talus_scale, qd_toe_scale),
    ):
        scale = torch.zeros(
            int(joint_dof_mask.numel()),
            device=joint_dof_mask.device,
            dtype=torch.float32,
        )
        scale[joint_dof_mask] = 1.0
        if talus_dof_indices.numel() > 0:
            scale[talus_dof_indices] = float(talus_scale)
        if toe_dof_indices.numel() > 0:
            scale[toe_dof_indices] = float(toe_scale)
        scales.append(scale)
    return scales[0], scales[1]


def quat_diff_to_axis_angle(q_ref: torch.Tensor, q_sim: torch.Tensor) -> torch.Tensor:
    return tracking_metrics.quat_diff_to_axis_angle(q_ref, q_sim)


def _frame_delta_to_multiplier(
    frame_delta_mse: torch.Tensor,
    *,
    weight: float,
    reference_delta_scale: float,
    min_weight_multiplier: float,
) -> torch.Tensor:
    if float(weight) <= 0.0:
        return torch.ones_like(frame_delta_mse)
    base_variance = 1.0 / float(weight)
    sigma_sq = torch.maximum(
        torch.full_like(frame_delta_mse, base_variance),
        frame_delta_mse * float(reference_delta_scale),
    )
    multiplier = base_variance / sigma_sq
    if float(min_weight_multiplier) > 0.0:
        multiplier = multiplier.clamp_min(float(min_weight_multiplier))
    return multiplier


def _assign_pair_delta_to_frames(pair_delta_mse: torch.Tensor) -> torch.Tensor:
    frame_delta_mse = torch.zeros(
        pair_delta_mse.shape[0] + 1,
        device=pair_delta_mse.device,
        dtype=pair_delta_mse.dtype,
    )
    frame_delta_mse[:-1] = torch.maximum(frame_delta_mse[:-1], pair_delta_mse)
    frame_delta_mse[1:] = torch.maximum(frame_delta_mse[1:], pair_delta_mse)
    return frame_delta_mse


def validate_reward_body_weight(
    weights: tuple[float, ...] | list[float], body_count: int
) -> tuple[float, ...]:
    """Per-reward-body weights: one finite, non-negative value per body, sum > 0."""

    values = tuple(float(v) for v in weights)
    if len(values) != int(body_count):
        raise ValueError(
            f"reward_body_weight has {len(values)} entries for {body_count} reward bodies."
        )
    if not all(math.isfinite(v) and v >= 0.0 for v in values) or sum(values) <= 0.0:
        raise ValueError(
            "reward_body_weight entries must be finite and >= 0, with a positive sum."
        )
    return values


def compute_body_state_frame_sigma_multipliers(
    reference_body_state: torch.Tensor,
    reward_body_idx: torch.Tensor,
    clip_start: torch.Tensor,
    clip_length: torch.Tensor,
    *,
    pos_weight: float,
    rot_weight: float,
    linvel_weight: float,
    angvel_weight: float,
    reference_delta_scale: float = 1.0,
    min_weight_multiplier: float = 0.0,
    reward_body_weight: tuple[float, ...] | None = None,
) -> torch.Tensor:
    if float(reference_delta_scale) < 0.0:
        raise ValueError("reference_delta_scale must be >= 0.")
    if not 0.0 <= float(min_weight_multiplier) <= 1.0:
        raise ValueError("min_weight_multiplier must be in [0, 1].")

    frame_count = int(reference_body_state.shape[0])
    multipliers = torch.ones(
        frame_count,
        4,
        device=reference_body_state.device,
        dtype=torch.float32,
    )
    if frame_count == 0 or int(reward_body_idx.numel()) == 0:
        return multipliers

    reward_body_idx = reward_body_idx.to(
        device=reference_body_state.device, dtype=torch.long
    )
    body_weight = (
        None
        if reward_body_weight is None
        else torch.tensor(
            validate_reward_body_weight(
                reward_body_weight, int(reward_body_idx.numel())
            ),
            device=reference_body_state.device,
            dtype=reference_body_state.dtype,
        )
    )

    def body_mean(squared: torch.Tensor) -> torch.Tensor:
        """Mean over bodies and components, weighted per body when configured."""
        if body_weight is None:
            return squared.mean(dim=(1, 2))
        return (squared * body_weight[None, :, None]).sum(dim=(1, 2)) / (
            body_weight.sum() * squared.shape[2]
        )

    clip_start_cpu = clip_start.detach().cpu().to(torch.long)
    clip_length_cpu = clip_length.detach().cpu().to(torch.long)
    for start_tensor, length_tensor in zip(
        clip_start_cpu, clip_length_cpu, strict=False
    ):
        start = int(start_tensor.item())
        length = int(length_tensor.item())
        if length <= 1:
            continue
        stop = start + length
        ref = reference_body_state[start:stop].index_select(1, reward_body_idx)

        pos_pair_mse = body_mean((ref[1:, :, :3] - ref[:-1, :, :3]).pow(2))
        rot_pair_aa = quat_diff_to_axis_angle(ref[:-1, :, 3:7], ref[1:, :, 3:7])
        rot_pair_mse = body_mean(rot_pair_aa.pow(2))
        linvel_pair_mse = body_mean((ref[1:, :, 7:10] - ref[:-1, :, 7:10]).pow(2))
        angvel_pair_mse = body_mean((ref[1:, :, 10:13] - ref[:-1, :, 10:13]).pow(2))

        multipliers[start:stop, 0] = _frame_delta_to_multiplier(
            _assign_pair_delta_to_frames(pos_pair_mse),
            weight=float(pos_weight),
            reference_delta_scale=float(reference_delta_scale),
            min_weight_multiplier=float(min_weight_multiplier),
        )
        multipliers[start:stop, 1] = _frame_delta_to_multiplier(
            _assign_pair_delta_to_frames(rot_pair_mse),
            weight=float(rot_weight),
            reference_delta_scale=float(reference_delta_scale),
            min_weight_multiplier=float(min_weight_multiplier),
        )
        multipliers[start:stop, 2] = _frame_delta_to_multiplier(
            _assign_pair_delta_to_frames(linvel_pair_mse),
            weight=float(linvel_weight),
            reference_delta_scale=float(reference_delta_scale),
            min_weight_multiplier=float(min_weight_multiplier),
        )
        multipliers[start:stop, 3] = _frame_delta_to_multiplier(
            _assign_pair_delta_to_frames(angvel_pair_mse),
            weight=float(angvel_weight),
            reference_delta_scale=float(reference_delta_scale),
            min_weight_multiplier=float(min_weight_multiplier),
        )

    return multipliers


# ============================================================
# Reward kernel
# ============================================================


def make_compute_rewards_kernel(spec: RewardKernelSpec):
    if float(spec.reward_power_beta) < 0.0:
        raise ValueError("reward_power_beta must be >= 0.")
    if float(spec.reward_power_epsilon) <= 0.0:
        raise ValueError("reward_power_epsilon must be > 0.")

    key = spec.cache_key()
    cached = _REWARD_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    N_HINGE = int(spec.n_hinge)
    N_BALL = int(spec.n_ball)
    NV = int(spec.nv)
    ROOT_QPOS_COUNT = int(spec.root_qpos_count)
    ROOT_POSITION_USES_QPOS = root_position_uses_qpos(ROOT_QPOS_COUNT)
    ROOT_DOF_COUNT = int(spec.root_dof_count)
    JOINT_DOF_COUNT = NV - ROOT_DOF_COUNT
    ROOT_BODY_IDX = int(spec.root_body_idx)
    EE_BODY_IDX = tuple(int(v) for v in spec.ee_body_idx)
    REWARD_BODY_IDX = tuple(int(v) for v in spec.reward_body_idx)
    NUM_EE = len(EE_BODY_IDX)
    NUM_REWARD_BODIES = len(REWARD_BODY_IDX)
    POSE_W = float(spec.pose_w)
    VEL_W = float(spec.vel_w)
    COM_W = float(spec.com_w)
    EE_W = float(spec.ee_w)
    EE_ROT_W = float(spec.ee_rot_w)
    ALIVE_REWARD = float(spec.alive_reward)
    REWARD_MODE = int(spec.reward_combination_mode)
    REWARD_POWER_Q = float(spec.reward_power_q)
    REWARD_POWER_BETA = float(spec.reward_power_beta)
    REWARD_POWER_EPS = float(spec.reward_power_epsilon)
    REWARD_POWER_Q_IS_EXP = abs(REWARD_POWER_Q - 1.0) <= 1.0e-8
    REWARD_POWER_Q_IS_LOW = REWARD_POWER_Q < 1.0
    BODY_STATE_REWARD = bool(spec.body_state_reward)
    BODY_STATE_FRAME_SIGMA_ENABLED = bool(spec.body_state_frame_sigma_enabled)
    BODY_STATE_REWARD_FRAME_MODE = int(spec.body_state_reward_frame_mode)
    JOINT_POSE_REWARD_MODE = int(spec.joint_pose_reward_mode)
    JOINT_VELOCITY_REWARD_MODE = int(spec.joint_velocity_reward_mode)
    BODY_PARENT_IDX = tuple(int(v) for v in spec.body_parent_idx)
    HINGE_BODY_IDX = tuple(int(v) for v in spec.hinge_body_idx)
    HINGE_PARENT_BODY_IDX = tuple(int(v) for v in spec.hinge_parent_body_idx)
    HINGE_AXIS_LOCAL = tuple(
        tuple(float(c) for c in row) for row in spec.hinge_axis_local
    )
    HINGE_MULTI_BODY_MASK = tuple(int(v) for v in spec.hinge_multi_body_mask)
    MIX_W_POSE = float(spec.mix_w_pose)
    MIX_W_VEL = float(spec.mix_w_vel)
    MIX_W_BODY_POS = float(spec.mix_w_body_pos)
    MIX_W_BODY_ROT = float(spec.mix_w_body_rot)
    MIX_W_BODY_LINVEL = float(spec.mix_w_body_linvel)
    MIX_W_BODY_ANGVEL = float(spec.mix_w_body_angvel)
    MIX_W_COM = float(spec.mix_w_com)
    MIX_W_EE_POS = float(spec.mix_w_ee_pos)
    MIX_W_EE_ROT = float(spec.mix_w_ee_rot)
    POSE_INCLUDE_ROOT = bool(spec.pose_include_root)
    VEL_INCLUDE_ROOT = bool(spec.vel_include_root)
    MIX_W_SUM = MIX_W_POSE + MIX_W_VEL + MIX_W_COM + MIX_W_EE_POS + MIX_W_EE_ROT
    EE_POS_ROOT_RELATIVE = bool(spec.ee_pos_root_relative)
    VEL_SEAM_FADE_FRAMES = int(spec.velocity_reward_seam_fade_frames)
    VEL_VALID_ALL = bool(spec.vel_valid_all)
    PASSIVE_DOF_COUNT = int(spec.passive_dof_count)
    INV_NUM_Q = 1.0 / float(NV - PASSIVE_DOF_COUNT)
    INV_NUM_QD = 1.0 / float(NV - PASSIVE_DOF_COUNT)
    REWARD_SCALE = float(spec.reward_scale)
    if not math.isfinite(REWARD_SCALE) or REWARD_SCALE <= 0.0:
        raise ValueError("reward_scale must be finite and > 0.")
    POSE_ROOT_ERROR_SCALE = float(spec.pose_root_error_scale)
    VEL_ROOT_ERROR_SCALE = float(spec.vel_root_error_scale)
    # The rotation term scores the masked end effectors; its error is a mean
    # over their rotation-vector components (3 per body) or, per body, of
    # their squared rotation angle.
    EE_ROT_MASK = (
        (1,) * NUM_EE
        if spec.ee_rot_body_mask is None
        else tuple(int(bool(v)) for v in spec.ee_rot_body_mask)
    )
    if len(EE_ROT_MASK) != NUM_EE:
        raise ValueError(
            f"ee_rot_body_mask has {len(EE_ROT_MASK)} entries for {NUM_EE} "
            "end effectors."
        )
    NUM_EE_ROT = sum(EE_ROT_MASK)
    if NUM_EE > 0 and NUM_EE_ROT == 0:
        raise ValueError("ee_rot_body_mask selects no end effector.")
    INV_NUM_EE_ROT = (
        0.0
        if NUM_EE_ROT == 0
        else 1.0 / (float(NUM_EE_ROT) * (1.0 if spec.ee_rot_mean_per_body else 3.0))
    )
    # The position term scores EE_POS_BODY_IDX at one body-frame point each
    # (EE_POS_OFFSET), optionally relative to an anchor body's point.
    EE_POS_BODY_IDX = (
        EE_BODY_IDX
        if spec.ee_pos_body_idx is None
        else tuple(int(v) for v in spec.ee_pos_body_idx)
    )
    NUM_EE_POS = len(EE_POS_BODY_IDX)
    if spec.ee_pos_body_idx is not None and NUM_EE_POS == 0:
        raise ValueError("ee_pos_body_idx selects no body.")
    EE_POS_OFFSET = (
        ((0.0, 0.0, 0.0),) * NUM_EE_POS
        if spec.ee_pos_body_offset is None
        else tuple(tuple(float(c) for c in row) for row in spec.ee_pos_body_offset)
    )
    if len(EE_POS_OFFSET) != NUM_EE_POS or any(len(r) != 3 for r in EE_POS_OFFSET):
        raise ValueError(
            f"ee_pos_body_offset needs one 3-vector for each of {NUM_EE_POS} "
            "position bodies."
        )
    EE_POS_HAS_OFFSET = tuple(int(any(c != 0.0 for c in r)) for r in EE_POS_OFFSET)
    EE_POS_ANCHOR = int(spec.ee_pos_anchor_body_idx)
    EE_POS_ANCHORED = EE_POS_ANCHOR >= 0
    EE_POS_ANCHOR_OFFSET = tuple(float(c) for c in spec.ee_pos_anchor_offset)
    if len(EE_POS_ANCHOR_OFFSET) != 3:
        raise ValueError("ee_pos_anchor_offset must be a 3-vector.")
    if EE_POS_ANCHORED and EE_POS_ROOT_RELATIVE:
        raise ValueError(
            "ee_pos_anchor_body_idx and ee_pos_root_relative exclude each other."
        )
    INV_NUM_EE3 = 1.0 / (float(NUM_EE_POS) * 3.0) if NUM_EE_POS > 0 else 0.0
    ACTION_SMOOTHNESS = bool(spec.action_smoothness_enabled)
    ACTION_SMOOTHNESS_W = float(spec.action_smoothness_weight)
    if ACTION_SMOOTHNESS and (
        not math.isfinite(ACTION_SMOOTHNESS_W) or ACTION_SMOOTHNESS_W < 0.0
    ):
        raise ValueError("action_smoothness_weight must be finite and >= 0.")
    MIX_W_ACTION_SMOOTH = (
        float(spec.mix_w_action_smoothness) if ACTION_SMOOTHNESS else 0.0
    )
    # Reported terms: the 5 joint (7 body-state) terms, then action smoothness.
    ACTION_SMOOTH_TERM = 7 if BODY_STATE_REWARD else 5
    # The equal-mix fallback applies when every tracking weight is 0; the
    # smoothness weight then joins the equal mix instead of standing alone.
    TRACK_MIX_W_SUM = MIX_W_SUM
    MIX_W_SUM = MIX_W_SUM + MIX_W_ACTION_SMOOTH
    INV_NUM_BODY3 = (
        1.0 / (float(NUM_REWARD_BODIES) * 3.0) if NUM_REWARD_BODIES > 0 else 0.0
    )
    # Body-state errors are a weighted mean over the reward bodies: each body's
    # squared error is scaled by its weight and the sum by 1 / (3 sum(weights)).
    BODY_WEIGHTED = spec.reward_body_weight is not None
    REWARD_BODY_WEIGHT = (
        validate_reward_body_weight(spec.reward_body_weight, NUM_REWARD_BODIES)
        if BODY_WEIGHTED
        else (1.0,)
    )
    if BODY_WEIGHTED:
        INV_NUM_BODY3 = 1.0 / (sum(REWARD_BODY_WEIGHT) * 3.0)
    (
        BODY_STATE_POS_W,
        BODY_STATE_ROT_W,
        BODY_STATE_LINVEL_W,
        BODY_STATE_ANGVEL_W,
    ) = (
        float(spec.body_state_pos_reward_weight or 0.0),
        float(spec.body_state_rot_reward_weight or 0.0),
        float(spec.body_state_linvel_reward_weight or 0.0),
        float(spec.body_state_angvel_reward_weight or 0.0),
    )
    BODY_TRACK_MIX_W_SUM = (
        MIX_W_BODY_POS
        + MIX_W_BODY_ROT
        + MIX_W_BODY_LINVEL
        + MIX_W_BODY_ANGVEL
        + MIX_W_COM
        + MIX_W_EE_POS
        + MIX_W_EE_ROT
    )
    BODY_MIX_W_SUM = BODY_TRACK_MIX_W_SUM + MIX_W_ACTION_SMOOTH

    # Runtime body indices allow lanes to stride over arbitrary skeleton sizes.
    BODY_IDS = wp.constant(
        wp.types.vector(length=max(1, NUM_REWARD_BODIES), dtype=int)(
            REWARD_BODY_IDX or (0,)
        )
    )
    BODY_WEIGHTS = wp.constant(
        wp.types.vector(length=max(1, len(REWARD_BODY_WEIGHT)), dtype=float)(
            REWARD_BODY_WEIGHT
        )
    )
    BODY_PARENTS = wp.constant(
        wp.types.vector(length=max(1, len(BODY_PARENT_IDX)), dtype=int)(
            BODY_PARENT_IDX or (0,)
        )
    )

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def compute_imitation_rewards_kernel(
        qpos: wp.array2d(dtype=float),
        qvel: wp.array2d(dtype=float),
        xpos: wp.array2d(dtype=wp.vec3),
        xquat: wp.array2d(dtype=wp.quat),
        cvel: wp.array2d(dtype=wp.spatial_vector),
        subtree_com: wp.array2d(dtype=wp.vec3),
        motion_qpos: wp.array2d(dtype=float),
        motion_qvel: wp.array2d(dtype=float),
        ref_body_state: wp.array3d(dtype=float),
        ref_com: wp.array(dtype=wp.vec3),
        motion_clip_idx: wp.array(dtype=int),
        ref_frame_idx: wp.array(dtype=int),
        clip_start: wp.array(dtype=int),
        clip_length: wp.array(dtype=int),
        clip_loop_root_quat: wp.array2d(dtype=float),
        clip_loop_root_pos: wp.array2d(dtype=float),
        hinge_qpos_idx: wp.array(dtype=int),
        hinge_dof_idx: wp.array(dtype=int),
        ball_qpos_adr: wp.array(dtype=int),
        ball_dof_adr: wp.array(dtype=int),
        dof_q_scale: wp.array(dtype=float),
        dof_qd_scale: wp.array(dtype=float),
        body_state_frame_sigma_multiplier: wp.array2d(dtype=float),
        action_smoothness_error: wp.array(dtype=float),
        reward_r_terms: wp.array2d(dtype=float),
        reward: wp.array(dtype=float),
    ):
        lane = int(0)
        if wp.static(BODY_STATE_REWARD):
            env_idx, lane = wp.tid()
        else:
            env_idx = wp.tid()
        clip_idx = motion_clip_idx[env_idx]
        fi = ref_frame_idx[env_idx]
        fi_local = clip_local_frame(fi, clip_idx, clip_length)
        fi_cycles = clip_cycle_count(fi, clip_idx, clip_length)
        fi_ref = clip_start[clip_idx] + fi_local
        clip_frame_last = clip_length[clip_idx] - 1
        loop_root_q = normalize_quat_wp(
            read_quat_from_2d(clip_loop_root_quat, clip_idx, 0)
        )
        loop_root_t = read_vec3_from_2d(clip_loop_root_pos, clip_idx, 0)

        vel_mult = 1.0
        if wp.static(not VEL_VALID_ALL) and fi_local == clip_frame_last:
            vel_mult = 0.0
        if wp.static(VEL_SEAM_FADE_FRAMES > 0):
            fade_denom = float(wp.max(VEL_SEAM_FADE_FRAMES - 1, 1))
            dist_to_start = fi_local
            dist_to_end = clip_frame_last - fi_local
            dist_to_seam = wp.min(dist_to_start, dist_to_end)
            seam_mult = wp.min(1.0, float(dist_to_seam) / fade_denom)
            vel_mult = vel_mult * seam_mult

        raw_q = float(0.0)
        raw_qd = float(0.0)
        raw_body_pos = float(0.0)
        raw_body_rot = float(0.0)
        raw_body_linvel = float(0.0)
        raw_body_angvel = float(0.0)
        if wp.static(ROOT_QPOS_COUNT == 7):
            ref_root_pos = wp.vec3(
                motion_qpos[fi_ref, 0],
                motion_qpos[fi_ref, 1],
                motion_qpos[fi_ref, 2],
            )
        else:
            ref_root_pos = wp.vec3(
                ref_body_state[fi_ref, ROOT_BODY_IDX, 0],
                ref_body_state[fi_ref, ROOT_BODY_IDX, 1],
                ref_body_state[fi_ref, ROOT_BODY_IDX, 2],
            )
        ref_root_pos = apply_loop_transform_vec(
            ref_root_pos, loop_root_q, loop_root_t, fi_cycles
        )

        if wp.static(BODY_STATE_REWARD):
            body_pos_diff_sq = float(0.0)
            body_rot_diff_sq = float(0.0)
            body_lv_diff_sq = float(0.0)
            body_av_diff_sq = float(0.0)

            for b in range(lane, NUM_REWARD_BODIES, wp.block_dim()):
                bi = BODY_IDS[b]
                bw = float(1.0)
                if wp.static(BODY_WEIGHTED):
                    bw = BODY_WEIGHTS[b]

                sim_bp = xpos[env_idx, bi]
                sim_bq = normalize_quat_wp(xquat[env_idx, bi])
                sim_bv = cvel[env_idx, bi]

                ref_bp = wp.vec3(
                    ref_body_state[fi_ref, bi, 0],
                    ref_body_state[fi_ref, bi, 1],
                    ref_body_state[fi_ref, bi, 2],
                )
                ref_bq = normalize_quat_wp(
                    wp.quat(
                        ref_body_state[fi_ref, bi, 3],
                        ref_body_state[fi_ref, bi, 4],
                        ref_body_state[fi_ref, bi, 5],
                        ref_body_state[fi_ref, bi, 6],
                    )
                )
                ref_blv = wp.vec3(
                    ref_body_state[fi_ref, bi, 7],
                    ref_body_state[fi_ref, bi, 8],
                    ref_body_state[fi_ref, bi, 9],
                )
                ref_bav = wp.vec3(
                    ref_body_state[fi_ref, bi, 10],
                    ref_body_state[fi_ref, bi, 11],
                    ref_body_state[fi_ref, bi, 12],
                )

                ref_bp = apply_loop_transform_vec(
                    ref_bp, loop_root_q, loop_root_t, fi_cycles
                )
                ref_bq = normalize_quat_wp(
                    apply_loop_transform_quat(ref_bq, loop_root_q, fi_cycles)
                )
                ref_blv = apply_loop_rotation_vec(ref_blv, loop_root_q, fi_cycles)
                ref_bav = apply_loop_rotation_vec(ref_bav, loop_root_q, fi_cycles)

                sim_cmp_bp = sim_bp
                sim_cmp_bq = sim_bq
                sim_cmp_blv = wp.vec3(sim_bv[3], sim_bv[4], sim_bv[5])
                sim_cmp_bav = wp.vec3(sim_bv[0], sim_bv[1], sim_bv[2])
                ref_cmp_bp = ref_bp
                ref_cmp_bq = ref_bq
                ref_cmp_blv = ref_blv
                ref_cmp_bav = ref_bav

                if wp.static(BODY_STATE_REWARD_FRAME_MODE == 1):
                    if bi != wp.static(ROOT_BODY_IDX):
                        pi = BODY_PARENTS[bi]

                        sim_pp = xpos[env_idx, pi]
                        sim_pq = normalize_quat_wp(xquat[env_idx, pi])
                        sim_pv = cvel[env_idx, pi]
                        sim_plv = wp.vec3(sim_pv[3], sim_pv[4], sim_pv[5])
                        sim_pav = wp.vec3(sim_pv[0], sim_pv[1], sim_pv[2])
                        sim_pq_inv = quat_inv(sim_pq)
                        sim_rel = sim_bp - sim_pp
                        sim_cmp_bp = rot_vec_quat(sim_rel, sim_pq_inv)
                        sim_cmp_bq = normalize_quat_wp(mul_quat(sim_pq_inv, sim_bq))
                        sim_cmp_blv = rot_vec_quat(
                            (sim_cmp_blv - sim_plv)
                            + wp.cross(
                                sim_cmp_bav - sim_pav,
                                sim_bp - subtree_com[env_idx, ROOT_BODY_IDX],
                            ),
                            sim_pq_inv,
                        )
                        sim_cmp_bav = rot_vec_quat(sim_cmp_bav - sim_pav, sim_pq_inv)

                        ref_pp = wp.vec3(
                            ref_body_state[fi_ref, pi, 0],
                            ref_body_state[fi_ref, pi, 1],
                            ref_body_state[fi_ref, pi, 2],
                        )
                        ref_pq = normalize_quat_wp(
                            wp.quat(
                                ref_body_state[fi_ref, pi, 3],
                                ref_body_state[fi_ref, pi, 4],
                                ref_body_state[fi_ref, pi, 5],
                                ref_body_state[fi_ref, pi, 6],
                            )
                        )
                        ref_plv = wp.vec3(
                            ref_body_state[fi_ref, pi, 7],
                            ref_body_state[fi_ref, pi, 8],
                            ref_body_state[fi_ref, pi, 9],
                        )
                        ref_pav = wp.vec3(
                            ref_body_state[fi_ref, pi, 10],
                            ref_body_state[fi_ref, pi, 11],
                            ref_body_state[fi_ref, pi, 12],
                        )
                        ref_pp = apply_loop_transform_vec(
                            ref_pp, loop_root_q, loop_root_t, fi_cycles
                        )
                        ref_pq = normalize_quat_wp(
                            apply_loop_transform_quat(ref_pq, loop_root_q, fi_cycles)
                        )
                        ref_plv = apply_loop_rotation_vec(
                            ref_plv, loop_root_q, fi_cycles
                        )
                        ref_pav = apply_loop_rotation_vec(
                            ref_pav, loop_root_q, fi_cycles
                        )
                        ref_pq_inv = quat_inv(ref_pq)
                        ref_rel = ref_bp - ref_pp
                        ref_cmp_bp = rot_vec_quat(ref_rel, ref_pq_inv)
                        ref_cmp_bq = normalize_quat_wp(mul_quat(ref_pq_inv, ref_bq))
                        # cvel's linear components share the root subtree
                        # COM origin. Convert their difference to the child's
                        # point before expressing velocity in the parent frame.
                        ref_origin = apply_loop_transform_vec(
                            ref_com[fi_ref], loop_root_q, loop_root_t, fi_cycles
                        )
                        ref_cmp_blv = rot_vec_quat(
                            (ref_cmp_blv - ref_plv)
                            + wp.cross(ref_cmp_bav - ref_pav, ref_bp - ref_origin),
                            ref_pq_inv,
                        )
                        ref_cmp_bav = rot_vec_quat(ref_cmp_bav - ref_pav, ref_pq_inv)

                for c in range(wp.static(3)):
                    pd = sim_cmp_bp[c] - ref_cmp_bp[c]
                    body_pos_diff_sq += bw * pd * pd

                body_aa = safe_quat_sub_wp(sim_cmp_bq, ref_cmp_bq)
                body_rot_diff_sq += bw * (
                    body_aa[0] * body_aa[0]
                    + body_aa[1] * body_aa[1]
                    + body_aa[2] * body_aa[2]
                )

                for c in range(wp.static(3)):
                    lvd = sim_cmp_blv[c] - ref_cmp_blv[c]
                    body_lv_diff_sq += bw * lvd * lvd
                    avd = sim_cmp_bav[c] - ref_cmp_bav[c]
                    body_av_diff_sq += bw * avd * avd

            body_pos_diff_sq = wp.tile_extract(
                wp.tile_sum(wp.tile(body_pos_diff_sq)), 0
            )
            body_rot_diff_sq = wp.tile_extract(
                wp.tile_sum(wp.tile(body_rot_diff_sq)), 0
            )
            body_lv_diff_sq = wp.tile_extract(wp.tile_sum(wp.tile(body_lv_diff_sq)), 0)
            body_av_diff_sq = wp.tile_extract(wp.tile_sum(wp.tile(body_av_diff_sq)), 0)

            body_pos_err = body_pos_diff_sq * INV_NUM_BODY3
            body_rot_err = body_rot_diff_sq * INV_NUM_BODY3
            body_linvel_err = body_lv_diff_sq * INV_NUM_BODY3 * vel_mult
            body_angvel_err = body_av_diff_sq * INV_NUM_BODY3 * vel_mult

            body_pos_mult = float(1.0)
            body_rot_mult = float(1.0)
            body_linvel_mult = float(1.0)
            body_angvel_mult = float(1.0)
            if wp.static(BODY_STATE_FRAME_SIGMA_ENABLED):
                body_pos_mult = body_state_frame_sigma_multiplier[fi_ref, 0]
                body_rot_mult = body_state_frame_sigma_multiplier[fi_ref, 1]
                body_linvel_mult = body_state_frame_sigma_multiplier[fi_ref, 2]
                body_angvel_mult = body_state_frame_sigma_multiplier[fi_ref, 3]

            raw_body_pos = -BODY_STATE_POS_W * body_pos_err * body_pos_mult
            raw_body_rot = -BODY_STATE_ROT_W * body_rot_err * body_rot_mult
            raw_body_linvel = -BODY_STATE_LINVEL_W * body_linvel_err * body_linvel_mult
            raw_body_angvel = -BODY_STATE_ANGVEL_W * body_angvel_err * body_angvel_mult
            raw_q = raw_body_pos + raw_body_rot
            raw_qd = raw_body_linvel + raw_body_angvel
        else:
            q_diff_sq_sum = float(0.0)

            for i in range(wp.static(N_HINGE)):
                qi = hinge_qpos_idx[i]
                di = hinge_dof_idx[i]
                diff = (qpos[env_idx, qi] - motion_qpos[fi_ref, qi]) * dof_q_scale[di]
                if wp.static(JOINT_POSE_REWARD_MODE == 1):
                    if bool(wp.static(HINGE_MULTI_BODY_MASK[i])):
                        bi = wp.static(HINGE_BODY_IDX[i])
                        pi = wp.static(HINGE_PARENT_BODY_IDX[i])

                        sim_bq = normalize_quat_wp(xquat[env_idx, bi])
                        sim_pq = normalize_quat_wp(xquat[env_idx, pi])
                        sim_local_q = local_quat_wp(sim_pq, sim_bq)

                        ref_bq_base = normalize_quat_wp(
                            wp.quat(
                                ref_body_state[fi_ref, bi, 3],
                                ref_body_state[fi_ref, bi, 4],
                                ref_body_state[fi_ref, bi, 5],
                                ref_body_state[fi_ref, bi, 6],
                            )
                        )
                        ref_pq_base = normalize_quat_wp(
                            wp.quat(
                                ref_body_state[fi_ref, pi, 3],
                                ref_body_state[fi_ref, pi, 4],
                                ref_body_state[fi_ref, pi, 5],
                                ref_body_state[fi_ref, pi, 6],
                            )
                        )
                        ref_bq = normalize_quat_wp(
                            apply_loop_transform_quat(
                                ref_bq_base, loop_root_q, fi_cycles
                            )
                        )
                        ref_pq = normalize_quat_wp(
                            apply_loop_transform_quat(
                                ref_pq_base, loop_root_q, fi_cycles
                            )
                        )
                        ref_local_q = local_quat_wp(ref_pq, ref_bq)
                        local_aa = safe_quat_sub_wp(sim_local_q, ref_local_q)
                        axis = wp.vec3(
                            wp.static(HINGE_AXIS_LOCAL[i][0]),
                            wp.static(HINGE_AXIS_LOCAL[i][1]),
                            wp.static(HINGE_AXIS_LOCAL[i][2]),
                        )
                        diff = wp.dot(local_aa, axis) * dof_q_scale[di]
                q_diff_sq_sum += diff * diff

            for i in range(wp.static(N_BALL)):
                qa = ball_qpos_adr[i]
                da = ball_dof_adr[i]
                sim_q = normalize_quat_wp(read_quat_from_2d(qpos, env_idx, qa))
                ref_q = normalize_quat_wp(read_quat_from_2d(motion_qpos, fi_ref, qa))
                aa = safe_quat_sub_wp(sim_q, ref_q)
                for c in range(wp.static(3)):
                    diff = aa[c] * dof_q_scale[da + c]
                    q_diff_sq_sum += diff * diff

            root_pos_diff_sq = float(0.0)
            if wp.static(ROOT_QPOS_COUNT == 7):
                sim_root_pos = wp.vec3(
                    qpos[env_idx, 0], qpos[env_idx, 1], qpos[env_idx, 2]
                )
                sim_root_q = normalize_quat_wp(read_quat_from_2d(qpos, env_idx, 3))
                ref_root_q_base = normalize_quat_wp(
                    read_quat_from_2d(motion_qpos, fi_ref, 3)
                )
            else:
                sim_root_pos = xpos[env_idx, ROOT_BODY_IDX]
                sim_root_q = normalize_quat_wp(xquat[env_idx, ROOT_BODY_IDX])
                ref_root_q_base = normalize_quat_wp(
                    wp.quat(
                        ref_body_state[fi_ref, ROOT_BODY_IDX, 3],
                        ref_body_state[fi_ref, ROOT_BODY_IDX, 4],
                        ref_body_state[fi_ref, ROOT_BODY_IDX, 5],
                        ref_body_state[fi_ref, ROOT_BODY_IDX, 6],
                    )
                )
            for c in range(wp.static(3)):
                rp = sim_root_pos[c] - ref_root_pos[c]
                root_pos_diff_sq += rp * rp

            ref_root_q = normalize_quat_wp(
                apply_loop_transform_quat(ref_root_q_base, loop_root_q, fi_cycles)
            )
            root_aa = safe_quat_sub_wp(sim_root_q, ref_root_q)
            root_quat_diff_sq = (
                root_aa[0] * root_aa[0]
                + root_aa[1] * root_aa[1]
                + root_aa[2] * root_aa[2]
            )
            root_q_diff_sq = root_pos_diff_sq + root_quat_diff_sq

            qd_diff_sq_sum = float(0.0)
            d = int(0)
            while d < JOINT_DOF_COUNT:
                diff = (
                    qvel[env_idx, d + ROOT_DOF_COUNT]
                    - motion_qvel[fi_ref, d + ROOT_DOF_COUNT]
                ) * dof_qd_scale[d]
                qd_diff_sq_sum += diff * diff
                d += 1

            if wp.static(JOINT_VELOCITY_REWARD_MODE >= 1):
                for i in range(wp.static(N_HINGE)):
                    use_local_angvel = True
                    if wp.static(JOINT_VELOCITY_REWARD_MODE == 2):
                        use_local_angvel = bool(wp.static(HINGE_MULTI_BODY_MASK[i]))
                    if use_local_angvel:
                        bi = wp.static(HINGE_BODY_IDX[i])
                        pi = wp.static(HINGE_PARENT_BODY_IDX[i])
                        di = hinge_dof_idx[i]

                        scale = dof_qd_scale[di]
                        old_diff = (
                            qvel[env_idx, di + ROOT_DOF_COUNT]
                            - motion_qvel[fi_ref, di + ROOT_DOF_COUNT]
                        ) * scale
                        qd_diff_sq_sum -= old_diff * old_diff

                        sim_bv = cvel[env_idx, bi]
                        sim_pv = cvel[env_idx, pi]
                        sim_bq = normalize_quat_wp(xquat[env_idx, bi])
                        sim_rel_av_local = rot_vec_quat(
                            wp.vec3(
                                sim_bv[0] - sim_pv[0],
                                sim_bv[1] - sim_pv[1],
                                sim_bv[2] - sim_pv[2],
                            ),
                            quat_inv(sim_bq),
                        )

                        ref_bq_base = normalize_quat_wp(
                            wp.quat(
                                ref_body_state[fi_ref, bi, 3],
                                ref_body_state[fi_ref, bi, 4],
                                ref_body_state[fi_ref, bi, 5],
                                ref_body_state[fi_ref, bi, 6],
                            )
                        )
                        ref_bq = normalize_quat_wp(
                            apply_loop_transform_quat(
                                ref_bq_base, loop_root_q, fi_cycles
                            )
                        )
                        ref_bav = apply_loop_rotation_vec(
                            wp.vec3(
                                ref_body_state[fi_ref, bi, 10],
                                ref_body_state[fi_ref, bi, 11],
                                ref_body_state[fi_ref, bi, 12],
                            ),
                            loop_root_q,
                            fi_cycles,
                        )
                        ref_pav = apply_loop_rotation_vec(
                            wp.vec3(
                                ref_body_state[fi_ref, pi, 10],
                                ref_body_state[fi_ref, pi, 11],
                                ref_body_state[fi_ref, pi, 12],
                            ),
                            loop_root_q,
                            fi_cycles,
                        )
                        ref_rel_av_local = rot_vec_quat(
                            wp.vec3(
                                ref_bav[0] - ref_pav[0],
                                ref_bav[1] - ref_pav[1],
                                ref_bav[2] - ref_pav[2],
                            ),
                            quat_inv(ref_bq),
                        )

                        axis = wp.vec3(
                            wp.static(HINGE_AXIS_LOCAL[i][0]),
                            wp.static(HINGE_AXIS_LOCAL[i][1]),
                            wp.static(HINGE_AXIS_LOCAL[i][2]),
                        )
                        rel_av_diff_local = sim_rel_av_local - ref_rel_av_local
                        new_diff = wp.dot(rel_av_diff_local, axis) * scale
                        qd_diff_sq_sum += new_diff * new_diff
            qd_diff_sq_sum = qd_diff_sq_sum * vel_mult

            root_cv = cvel[env_idx, ROOT_BODY_IDX]
            ref_root_linvel = wp.vec3(
                ref_body_state[fi_ref, ROOT_BODY_IDX, 7],
                ref_body_state[fi_ref, ROOT_BODY_IDX, 8],
                ref_body_state[fi_ref, ROOT_BODY_IDX, 9],
            )
            ref_root_angvel = wp.vec3(
                ref_body_state[fi_ref, ROOT_BODY_IDX, 10],
                ref_body_state[fi_ref, ROOT_BODY_IDX, 11],
                ref_body_state[fi_ref, ROOT_BODY_IDX, 12],
            )
            ref_root_linvel = apply_loop_rotation_vec(
                ref_root_linvel, loop_root_q, fi_cycles
            )
            ref_root_angvel = apply_loop_rotation_vec(
                ref_root_angvel, loop_root_q, fi_cycles
            )
            root_qd_diff_sq = float(0.0)
            for c in range(wp.static(3)):
                ld = root_cv[3 + c] - ref_root_linvel[c]
                root_qd_diff_sq += ld * ld
            for c in range(wp.static(3)):
                ad = (root_cv[c] - ref_root_angvel[c]) * 0.1
                root_qd_diff_sq += ad * ad
            root_qd_diff_sq = root_qd_diff_sq * vel_mult

            pose_err = q_diff_sq_sum
            if wp.static(POSE_INCLUDE_ROOT):
                pose_err = pose_err + root_q_diff_sq * POSE_ROOT_ERROR_SCALE

            vel_err = qd_diff_sq_sum
            if wp.static(VEL_INCLUDE_ROOT):
                vel_err = vel_err + root_qd_diff_sq * VEL_ROOT_ERROR_SCALE

            raw_q = -POSE_W * pose_err * INV_NUM_Q
            raw_qd = -VEL_W * vel_err * INV_NUM_QD

        sim_com = subtree_com[env_idx, ROOT_BODY_IDX]
        ref_com_base = ref_com[fi_ref]
        ref_com_world = apply_loop_transform_vec(
            ref_com_base, loop_root_q, loop_root_t, fi_cycles
        )
        cx = sim_com[0] - ref_com_world[0]
        cy = sim_com[1] - ref_com_world[1]
        cz = sim_com[2] - ref_com_world[2]
        com_diff_sq = cx * cx + cy * cy + cz * cz

        ee_pos_diff_sq = float(0.0)
        ee_rot_diff_sq = float(0.0)
        if wp.static(ROOT_POSITION_USES_QPOS):
            sim_root_pos = wp.vec3(qpos[env_idx, 0], qpos[env_idx, 1], qpos[env_idx, 2])
        else:
            sim_root_pos = xpos[env_idx, ROOT_BODY_IDX]
        sim_anchor_pos = wp.vec3(0.0, 0.0, 0.0)
        ref_anchor_pos = wp.vec3(0.0, 0.0, 0.0)
        if wp.static(EE_POS_ANCHORED):
            anchor_off = wp.vec3(
                wp.static(EE_POS_ANCHOR_OFFSET[0]),
                wp.static(EE_POS_ANCHOR_OFFSET[1]),
                wp.static(EE_POS_ANCHOR_OFFSET[2]),
            )
            sim_anchor_pos = xpos[env_idx, EE_POS_ANCHOR] + rot_vec_quat(
                anchor_off, normalize_quat_wp(xquat[env_idx, EE_POS_ANCHOR])
            )
            ref_anchor_q = normalize_quat_wp(
                wp.quat(
                    ref_body_state[fi_ref, EE_POS_ANCHOR, 3],
                    ref_body_state[fi_ref, EE_POS_ANCHOR, 4],
                    ref_body_state[fi_ref, EE_POS_ANCHOR, 5],
                    ref_body_state[fi_ref, EE_POS_ANCHOR, 6],
                )
            )
            ref_anchor_pos = apply_loop_transform_vec(
                wp.vec3(
                    ref_body_state[fi_ref, EE_POS_ANCHOR, 0],
                    ref_body_state[fi_ref, EE_POS_ANCHOR, 1],
                    ref_body_state[fi_ref, EE_POS_ANCHOR, 2],
                )
                + rot_vec_quat(anchor_off, ref_anchor_q),
                loop_root_q,
                loop_root_t,
                fi_cycles,
            )
        for e in range(wp.static(NUM_EE_POS)):
            pbi = wp.static(EE_POS_BODY_IDX[e])
            ee_p = xpos[env_idx, pbi]
            ref_ee_pos = wp.vec3(
                ref_body_state[fi_ref, pbi, 0],
                ref_body_state[fi_ref, pbi, 1],
                ref_body_state[fi_ref, pbi, 2],
            )
            if wp.static(EE_POS_HAS_OFFSET[e] == 1):
                point_off = wp.vec3(
                    wp.static(EE_POS_OFFSET[e][0]),
                    wp.static(EE_POS_OFFSET[e][1]),
                    wp.static(EE_POS_OFFSET[e][2]),
                )
                ee_p = ee_p + rot_vec_quat(
                    point_off, normalize_quat_wp(xquat[env_idx, pbi])
                )
                ref_ee_pos = ref_ee_pos + rot_vec_quat(
                    point_off,
                    normalize_quat_wp(
                        wp.quat(
                            ref_body_state[fi_ref, pbi, 3],
                            ref_body_state[fi_ref, pbi, 4],
                            ref_body_state[fi_ref, pbi, 5],
                            ref_body_state[fi_ref, pbi, 6],
                        )
                    ),
                )
            ref_ee_pos = apply_loop_transform_vec(
                ref_ee_pos, loop_root_q, loop_root_t, fi_cycles
            )
            sim_ee_pos_cmp = ee_p
            ref_ee_pos_cmp = ref_ee_pos
            if wp.static(EE_POS_ROOT_RELATIVE):
                sim_ee_pos_cmp = ee_p - sim_root_pos
                ref_ee_pos_cmp = ref_ee_pos - ref_root_pos
            if wp.static(EE_POS_ANCHORED):
                sim_ee_pos_cmp = ee_p - sim_anchor_pos
                ref_ee_pos_cmp = ref_ee_pos - ref_anchor_pos
            for c in range(wp.static(3)):
                pd = sim_ee_pos_cmp[c] - ref_ee_pos_cmp[c]
                ee_pos_diff_sq += pd * pd

        for e in range(wp.static(NUM_EE)):
            bi = wp.static(EE_BODY_IDX[e])
            if wp.static(EE_ROT_MASK[e] == 1):
                sim_eq = normalize_quat_wp(xquat[env_idx, bi])
                ref_eq_base = normalize_quat_wp(
                    wp.quat(
                        ref_body_state[fi_ref, bi, 3],
                        ref_body_state[fi_ref, bi, 4],
                        ref_body_state[fi_ref, bi, 5],
                        ref_body_state[fi_ref, bi, 6],
                    )
                )
                ref_eq = normalize_quat_wp(
                    apply_loop_transform_quat(ref_eq_base, loop_root_q, fi_cycles)
                )
                ee_aa = safe_quat_sub_wp(sim_eq, ref_eq)
                ee_rot_diff_sq += (
                    ee_aa[0] * ee_aa[0] + ee_aa[1] * ee_aa[1] + ee_aa[2] * ee_aa[2]
                )

        raw_com = -COM_W * com_diff_sq / 3.0
        raw_ee = -EE_W * ee_pos_diff_sq * INV_NUM_EE3
        raw_ee_rot = -EE_ROT_W * ee_rot_diff_sq * INV_NUM_EE_ROT

        r_q = wp.exp(raw_q)
        r_qd = wp.exp(raw_qd)
        r_body_pos = wp.exp(raw_body_pos)
        r_body_rot = wp.exp(raw_body_rot)
        r_body_linvel = wp.exp(raw_body_linvel)
        r_body_angvel = wp.exp(raw_body_angvel)
        r_com = wp.exp(raw_com)
        r_ee = wp.exp(raw_ee)
        r_ee_rot = wp.exp(raw_ee_rot)
        raw_action_smooth = float(0.0)
        if wp.static(ACTION_SMOOTHNESS):
            raw_action_smooth = -ACTION_SMOOTHNESS_W * action_smoothness_error[env_idx]
        r_action_smooth = wp.exp(raw_action_smooth)

        ra = ALIVE_REWARD
        omra = 1.0 - ra

        t_q = ra + omra * r_q
        t_ee = ra + omra * r_ee
        t_ee_rot = ra + omra * r_ee_rot
        t_com = ra + omra * r_com
        t_qd = ra + omra * r_qd
        t_body_pos = ra + omra * r_body_pos
        t_body_rot = ra + omra * r_body_rot
        t_body_linvel = ra + omra * r_body_linvel
        t_body_angvel = ra + omra * r_body_angvel
        t_action_smooth = ra + omra * r_action_smooth

        r_total = float(0.0)
        raw_total = float(0.0)
        if wp.static(BODY_STATE_REWARD):
            raw_total = (
                raw_body_pos
                + raw_body_rot
                + raw_body_linvel
                + raw_body_angvel
                + raw_com
                + raw_ee
                + raw_ee_rot
                + raw_action_smooth
            )
            if wp.static(REWARD_MODE == 2 or REWARD_MODE == 3):
                if lane == 0:
                    reward_r_terms[env_idx, 0] = raw_body_pos
                    reward_r_terms[env_idx, 1] = raw_body_rot
                    reward_r_terms[env_idx, 2] = raw_body_linvel
                    reward_r_terms[env_idx, 3] = raw_body_angvel
                    reward_r_terms[env_idx, 4] = raw_com
                    reward_r_terms[env_idx, 5] = raw_ee
                    reward_r_terms[env_idx, 6] = raw_ee_rot
                    if wp.static(ACTION_SMOOTHNESS):
                        reward_r_terms[env_idx, ACTION_SMOOTH_TERM] = raw_action_smooth
                if wp.static(REWARD_MODE == 2):
                    r_total = ALIVE_REWARD + raw_total
            else:
                if lane == 0:
                    reward_r_terms[env_idx, 0] = r_body_pos
                    reward_r_terms[env_idx, 1] = r_body_rot
                    reward_r_terms[env_idx, 2] = r_body_linvel
                    reward_r_terms[env_idx, 3] = r_body_angvel
                    reward_r_terms[env_idx, 4] = r_com
                    reward_r_terms[env_idx, 5] = r_ee
                    reward_r_terms[env_idx, 6] = r_ee_rot
                    if wp.static(ACTION_SMOOTHNESS):
                        reward_r_terms[env_idx, ACTION_SMOOTH_TERM] = r_action_smooth
        else:
            raw_total = (
                raw_q + raw_qd + raw_com + raw_ee + raw_ee_rot + raw_action_smooth
            )
            if wp.static(REWARD_MODE == 2 or REWARD_MODE == 3):
                if lane == 0:
                    reward_r_terms[env_idx, 0] = raw_q
                    reward_r_terms[env_idx, 1] = raw_qd
                    reward_r_terms[env_idx, 2] = raw_com
                    reward_r_terms[env_idx, 3] = raw_ee
                    reward_r_terms[env_idx, 4] = raw_ee_rot
                    if wp.static(ACTION_SMOOTHNESS):
                        reward_r_terms[env_idx, ACTION_SMOOTH_TERM] = raw_action_smooth
                if wp.static(REWARD_MODE == 2):
                    r_total = ALIVE_REWARD + raw_total
            else:
                if lane == 0:
                    reward_r_terms[env_idx, 0] = r_q
                    reward_r_terms[env_idx, 1] = r_qd
                    reward_r_terms[env_idx, 2] = r_com
                    reward_r_terms[env_idx, 3] = r_ee
                    reward_r_terms[env_idx, 4] = r_ee_rot
                    if wp.static(ACTION_SMOOTHNESS):
                        reward_r_terms[env_idx, ACTION_SMOOTH_TERM] = r_action_smooth

        if wp.static(REWARD_MODE == 3):
            error_total = -raw_total
            if error_total < 0.0:
                error_total = 0.0
            scaled_error = REWARD_POWER_BETA * error_total

            if wp.static(REWARD_POWER_Q_IS_EXP):
                r_total = wp.max(REWARD_POWER_EPS, wp.exp(-scaled_error))
            elif wp.static(REWARD_POWER_Q_IS_LOW):
                base = 1.0 - (1.0 - REWARD_POWER_Q) * scaled_error
                base = wp.max(REWARD_POWER_EPS, base)
                r_total = wp.max(
                    REWARD_POWER_EPS,
                    wp.pow(base, 1.0 / (1.0 - REWARD_POWER_Q)),
                )
            else:
                base = 1.0 + (REWARD_POWER_Q - 1.0) * scaled_error
                base = wp.max(REWARD_POWER_EPS, base)
                r_total = wp.max(
                    REWARD_POWER_EPS,
                    wp.pow(base, -1.0 / (REWARD_POWER_Q - 1.0)),
                )
        elif wp.static(REWARD_MODE == 1):
            if wp.static(BODY_STATE_REWARD):
                if wp.static(BODY_TRACK_MIX_W_SUM > 0.0):
                    r_total = (
                        MIX_W_BODY_POS * t_body_pos
                        + MIX_W_BODY_ROT * t_body_rot
                        + MIX_W_BODY_LINVEL * t_body_linvel
                        + MIX_W_BODY_ANGVEL * t_body_angvel
                        + MIX_W_COM * t_com
                        + MIX_W_EE_POS * t_ee
                        + MIX_W_EE_ROT * t_ee_rot
                        + MIX_W_ACTION_SMOOTH * t_action_smooth
                    ) / BODY_MIX_W_SUM
                else:
                    r_total = (
                        t_body_pos
                        + t_body_rot
                        + t_body_linvel
                        + t_body_angvel
                        + t_com
                        + t_ee
                        + t_ee_rot
                    )
                    if wp.static(ACTION_SMOOTHNESS):
                        r_total = (r_total + t_action_smooth) / 8.0
                    else:
                        r_total = r_total / 7.0
            else:
                if wp.static(TRACK_MIX_W_SUM > 0.0):
                    r_total = (
                        MIX_W_POSE * t_q
                        + MIX_W_VEL * t_qd
                        + MIX_W_COM * t_com
                        + MIX_W_EE_POS * t_ee
                        + MIX_W_EE_ROT * t_ee_rot
                        + MIX_W_ACTION_SMOOTH * t_action_smooth
                    ) / MIX_W_SUM
                else:
                    if wp.static(ACTION_SMOOTHNESS):
                        r_total = (
                            t_q + t_ee + t_ee_rot + t_com + t_qd + t_action_smooth
                        ) / 6.0
                    else:
                        r_total = 0.2 * (t_q + t_ee + t_ee_rot + t_com + t_qd)
        elif wp.static(REWARD_MODE == 0):
            if wp.static(BODY_STATE_REWARD):
                r_total = (
                    t_body_pos
                    * t_body_rot
                    * t_body_linvel
                    * t_body_angvel
                    * t_com
                    * t_ee
                    * t_ee_rot
                )
            else:
                r_total = t_q * t_ee * t_ee_rot * t_com * t_qd
            if wp.static(ACTION_SMOOTHNESS):
                r_total = r_total * t_action_smooth

        if wp.static(REWARD_SCALE != 1.0):
            r_total = r_total * REWARD_SCALE

        if lane == 0:
            reward[env_idx] = r_total

    compute_imitation_rewards_kernel.body_state_cooperative = BODY_STATE_REWARD
    _REWARD_KERNEL_CACHE[key] = compute_imitation_rewards_kernel
    return compute_imitation_rewards_kernel


def launch_compute_rewards(
    kernel, *, dim, inputs, outputs, block_dim=REWARD_KERNEL_BLOCK_DIM, device=None
):
    """Launch the shared reward implementation with its required work layout."""
    launch = (
        wp.launch_tiled
        if getattr(kernel, "body_state_cooperative", False)
        else wp.launch
    )
    return launch(
        kernel,
        dim=dim,
        inputs=inputs,
        outputs=outputs,
        block_dim=block_dim,
        device=device,
    )
