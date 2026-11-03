from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import torch
import warp as wp

from instantmimic.envs.config_access import get_block, get_value
from instantmimic.envs.direct_rl_env import make_stream_aligned_cuda_graph_streams
from instantmimic.envs.snu.tasks.snu_imitation.future_observation_kernels import (
    FutureObservationKernelSpec,
    build_loop_cycle_transform_tables,
    future_observation_feature_dim,
    future_observation_uses_phc_v6,
    make_future_observation_kernel,
    resolve_future_observation_mode,
)
from instantmimic.envs.snu.tasks.snu_imitation.warp_utils import (
    heading_yaw_quat,
    local_quat_wp,
    normalize_quat_wp,
    ortho6d_component,
    read_quat_from_2d,
    read_vec3_from_2d,
    write_ortho6d,
    write_quat4,
    write_vec3,
)
from instantmimic.envs.tracking.observation_modes import (
    resolve_joint_pose_mode,
    resolve_joint_velocity_mode,
)
from instantmimic.motion.warp_utils import (
    apply_loop_transform_quat,
    apply_loop_transform_vec,
    clip_cycle_count,
    clip_local_frame,
    loop_cycle_count,
    loop_local_frame,
)
from instantmimic.utils.mjw_kernels.private import mul_quat, quat_inv, rot_vec_quat
from instantmimic.utils.startup_profiler import StartupProfiler, startup_nvtx_range
from instantmimic.utils.warp_kernels import expose_kernel_name

# ============================================================
# Observation kernel
# ============================================================


@dataclass(frozen=True)
class ObservationLayout:
    joint_q_count: int
    joint_v_count: int
    num_future_pose_bodies: int
    num_future_frames: int
    future_observation_mode: str
    joint_pos_ofs: int
    joint_vel_ofs: int
    body_pos_ofs: int
    body_rot_ofs: int
    body_lv_ofs: int
    body_av_ofs: int
    future_obs_ofs: int
    future_obs_dim: int
    action_ofs: int
    obs_dim: int
    root_target_dim: int = 9


@dataclass(frozen=True)
class ObservationKernels:
    layout: ObservationLayout
    root_target: Any
    joint_pos: Any
    joint_vel: Any
    body_state: Any
    future_obs: Any | None
    prev_actions: Any


@dataclass(frozen=True)
class ObservationSettings:
    joint_pose_mode: int
    joint_velocity_mode: int
    future_frames: int
    future_frame_step: int
    future_mode: str
    include_root_target: bool
    include_joint_state: bool
    body_source: str = "all"


@dataclass
class ObservationRuntimeState:
    """Persistent buffers, kernels, and streams for observation generation."""

    joint_position_count: int = 0
    root_position_xz: torch.Tensor | None = None
    root_yaw_inverse: torch.Tensor | None = None
    loop_cycle_table_size: int = 0
    loop_cycle_transform_quat: torch.Tensor | None = None
    loop_cycle_transform_pos: torch.Tensor | None = None
    kernels: ObservationKernels | None = None
    parallel_torch_stream: torch.cuda.Stream | None = None
    parallel_wp_stream: wp.Stream | None = None
    future_parallel_torch_stream: torch.cuda.Stream | None = None
    future_parallel_wp_stream: wp.Stream | None = None


def resolve_observation_settings(
    cfg,
    *,
    joint_pose_reward_mode: int,
    joint_velocity_reward_mode: int,
) -> ObservationSettings:
    observations = get_block(cfg, "observations")
    obs_joint_pose_mode = str(get_value(observations, "joint_pose", "qpos")).lower()
    if obs_joint_pose_mode == "match_reward":
        resolved_joint_pose_mode = int(joint_pose_reward_mode)
    else:
        resolved_joint_pose_mode = resolve_joint_pose_mode(
            obs_joint_pose_mode,
            allow_6d=True,
        )

    obs_joint_velocity_mode = str(
        get_value(observations, "joint_velocity", "qvel")
    ).lower()
    if obs_joint_velocity_mode == "match_reward":
        resolved_joint_velocity_mode = int(joint_velocity_reward_mode)
    else:
        resolved_joint_velocity_mode = resolve_joint_velocity_mode(
            obs_joint_velocity_mode
        )
    return ObservationSettings(
        joint_pose_mode=resolved_joint_pose_mode,
        joint_velocity_mode=resolved_joint_velocity_mode,
        future_frames=int(get_value(observations, "future_frames")),
        future_frame_step=int(get_value(observations, "future_frame_step")),
        future_mode=str(get_value(observations, "future_mode", "phc_v6")),
        include_root_target=bool(get_value(observations, "include_root_target", True)),
        include_joint_state=bool(get_value(observations, "include_joint_state", True)),
        body_source=str(get_value(observations, "body_source", "all")).lower(),
    )


def initialize_observation_kernels(env) -> None:
    startup_profiler = StartupProfiler(
        range_factory=lambda name: startup_nvtx_range(f"observation_kernels:{name}")
    )
    with startup_profiler.phase("observation_kernels"):
        env.observation_runtime_state.kernels = (
            build_observation_kernels_for_prev_action_dim(
                env,
                prev_action_dim=int(env.model.nu),
            )
        )
    startup_profiler.log_summary(prefix="OBSERVATION_KERNELS_STARTUP_PROFILE")


def make_observation_layout(
    *,
    nbody: int,
    nq: int,
    nv: int,
    nu: int,
    num_future_pose_bodies: int,
    num_future_frames: int,
    future_observation_mode: str,
    include_root_target_obs: bool,
    include_joint_state: bool,
    joint_q_count_override: int | None = None,
    joint_v_count_override: int | None = None,
) -> ObservationLayout:
    if include_joint_state:
        joint_q_count = (
            int(joint_q_count_override)
            if joint_q_count_override is not None
            else int(nq) - 7
        )
        joint_v_count = (
            int(joint_v_count_override)
            if joint_v_count_override is not None
            else int(nv) - 6
        )
    else:
        joint_q_count = 0
        joint_v_count = 0
    resolved_future_observation_mode = resolve_future_observation_mode(
        future_observation_mode
    )
    root_target_dim = 9 if bool(include_root_target_obs) else 0
    joint_pos_ofs = root_target_dim
    joint_vel_ofs = joint_pos_ofs + joint_q_count
    body_pos_ofs = joint_vel_ofs + joint_v_count
    body_rot_ofs = body_pos_ofs + int(nbody) * 3
    body_lv_ofs = body_rot_ofs + int(nbody) * 6
    body_av_ofs = body_lv_ofs + int(nbody) * 3
    future_obs_ofs = body_av_ofs + int(nbody) * 3
    future_obs_dim = future_observation_feature_dim(
        mode=resolved_future_observation_mode,
        num_future_pose_bodies=int(num_future_pose_bodies),
        num_future_frames=int(num_future_frames),
    )
    action_ofs = future_obs_ofs + future_obs_dim
    obs_dim = action_ofs + int(nu)

    return ObservationLayout(
        joint_q_count=joint_q_count,
        joint_v_count=joint_v_count,
        num_future_pose_bodies=int(num_future_pose_bodies),
        num_future_frames=int(num_future_frames),
        future_observation_mode=resolved_future_observation_mode,
        joint_pos_ofs=joint_pos_ofs,
        joint_vel_ofs=joint_vel_ofs,
        body_pos_ofs=body_pos_ofs,
        body_rot_ofs=body_rot_ofs,
        body_lv_ofs=body_lv_ofs,
        body_av_ofs=body_av_ofs,
        future_obs_ofs=future_obs_ofs,
        future_obs_dim=future_obs_dim,
        action_ofs=action_ofs,
        obs_dim=obs_dim,
        root_target_dim=root_target_dim,
    )


_OBSERVATION_KERNEL_CACHE: dict[tuple[Any, ...], ObservationKernels] = {}


def make_observation_kernels(
    *,
    motion_frame_count: int,
    layout: ObservationLayout,
    nbody: int,
    nu: int,
    root_body_idx: int,
    future_pose_body_idx: tuple[int, ...],
    future_frame_step: int,
    loop_root_q: tuple[float, float, float, float],
    loop_root_t: tuple[float, float, float],
    loop_cycle_table_size: int,
    joint_velocity_obs_mode: int,
    observation_body_idx: tuple[int, ...] | None = None,
    root_qpos_count: int = 7,
    root_dof_count: int = 6,
    multi_motion: bool = False,
):
    if observation_body_idx is None:
        observation_body_idx = tuple(range(int(nbody)))
    key = (
        int(motion_frame_count),
        layout,
        int(nbody),
        int(nu),
        int(root_body_idx),
        tuple(int(v) for v in future_pose_body_idx),
        int(future_frame_step),
        tuple(float(v) for v in loop_root_q),
        tuple(float(v) for v in loop_root_t),
        int(loop_cycle_table_size),
        int(joint_velocity_obs_mode),
        tuple(int(v) for v in observation_body_idx),
        int(root_qpos_count),
        int(root_dof_count),
        bool(multi_motion),
    )
    cached = _OBSERVATION_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    MOTION_FRAME_COUNT = int(motion_frame_count)
    ROOT_BODY_IDX = int(root_body_idx)
    ROOT_TARGET_DIM = int(layout.root_target_dim)
    JOINT_POS_OFS = layout.joint_pos_ofs
    JOINT_VEL_OFS = layout.joint_vel_ofs
    BODY_POS_OFS = layout.body_pos_ofs
    BODY_ROT_OFS = layout.body_rot_ofs
    BODY_LV_OFS = layout.body_lv_ofs
    BODY_AV_OFS = layout.body_av_ofs
    ACTION_OFS = layout.action_ofs
    JOINT_VELOCITY_OBS_MODE = int(joint_velocity_obs_mode)
    ROOT_QPOS_COUNT = int(root_qpos_count)
    ROOT_DOF_COUNT = int(root_dof_count)
    LOOP_ROOT_Q = wp.quat(
        float(loop_root_q[0]),
        float(loop_root_q[1]),
        float(loop_root_q[2]),
        float(loop_root_q[3]),
    )
    LOOP_ROOT_T = wp.vec3(
        float(loop_root_t[0]),
        float(loop_root_t[1]),
        float(loop_root_t[2]),
    )

    if not multi_motion:

        @wp.kernel(enable_backward=False)
        @expose_kernel_name
        def write_root_target_observation_single_motion_kernel(
            qpos: wp.array2d(dtype=float),
            motion_qpos: wp.array2d(dtype=float),
            xpos: wp.array2d(dtype=wp.vec3),
            xquat: wp.array2d(dtype=wp.quat),
            reference_body_state: wp.array3d(dtype=float),
            ref_frame_idx: wp.array(dtype=int),
            root_pos_xz_buf: wp.array2d(dtype=float),
            root_yaw_inv_buf: wp.array2d(dtype=float),
            obs: wp.array2d(dtype=float),
        ):
            env_idx = wp.tid()
            fi = ref_frame_idx[env_idx]
            fi_local = loop_local_frame(fi, MOTION_FRAME_COUNT)
            fi_cycles = loop_cycle_count(fi, MOTION_FRAME_COUNT)

            if wp.static(ROOT_QPOS_COUNT == 7):
                root_q = normalize_quat_wp(read_quat_from_2d(qpos, env_idx, 3))
                root_pos_xz = wp.vec3(qpos[env_idx, 0], qpos[env_idx, 1], 0.0)
                base_ref_root_q = normalize_quat_wp(
                    read_quat_from_2d(motion_qpos, fi_local, 3)
                )
                base_ref_root_pos = wp.vec3(
                    motion_qpos[fi_local, 0],
                    motion_qpos[fi_local, 1],
                    motion_qpos[fi_local, 2],
                )
            else:
                root_q = normalize_quat_wp(xquat[env_idx, ROOT_BODY_IDX])
                sim_root_pos = xpos[env_idx, ROOT_BODY_IDX]
                root_pos_xz = wp.vec3(sim_root_pos[0], sim_root_pos[1], 0.0)
                base_ref_root_q = normalize_quat_wp(
                    wp.quat(
                        reference_body_state[fi_local, ROOT_BODY_IDX, 3],
                        reference_body_state[fi_local, ROOT_BODY_IDX, 4],
                        reference_body_state[fi_local, ROOT_BODY_IDX, 5],
                        reference_body_state[fi_local, ROOT_BODY_IDX, 6],
                    )
                )
                base_ref_root_pos = wp.vec3(
                    reference_body_state[fi_local, ROOT_BODY_IDX, 0],
                    reference_body_state[fi_local, ROOT_BODY_IDX, 1],
                    reference_body_state[fi_local, ROOT_BODY_IDX, 2],
                )
            yaw_q = heading_yaw_quat(root_q)
            yaw_inv = quat_inv(yaw_q)
            ref_root_q = normalize_quat_wp(
                apply_loop_transform_quat(base_ref_root_q, LOOP_ROOT_Q, fi_cycles)
            )
            ref_yaw_q = heading_yaw_quat(ref_root_q)
            ref_root_pos = apply_loop_transform_vec(
                base_ref_root_pos, LOOP_ROOT_Q, LOOP_ROOT_T, fi_cycles
            )
            ref_root_xz = wp.vec3(ref_root_pos[0], ref_root_pos[1], 0.0)

            target_pos = rot_vec_quat(ref_root_xz - root_pos_xz, yaw_inv)
            target_rot_q = normalize_quat_wp(mul_quat(yaw_inv, ref_yaw_q))

            write_vec3(root_pos_xz_buf, env_idx, 0, root_pos_xz, 1.0)
            write_quat4(root_yaw_inv_buf, env_idx, 0, yaw_inv)
            if ROOT_TARGET_DIM > 0:
                write_vec3(obs, env_idx, 0, target_pos, 1.0)
                write_ortho6d(obs, env_idx, 3, target_rot_q)

        root_target_kernel = write_root_target_observation_single_motion_kernel

    else:

        @wp.kernel(enable_backward=False)
        @expose_kernel_name
        def write_root_target_observation_multi_motion_kernel(
            qpos: wp.array2d(dtype=float),
            motion_qpos: wp.array2d(dtype=float),
            xpos: wp.array2d(dtype=wp.vec3),
            xquat: wp.array2d(dtype=wp.quat),
            reference_body_state: wp.array3d(dtype=float),
            motion_clip_idx: wp.array(dtype=int),
            ref_frame_idx: wp.array(dtype=int),
            clip_start: wp.array(dtype=int),
            clip_length: wp.array(dtype=int),
            clip_loop_root_quat: wp.array2d(dtype=float),
            clip_loop_root_pos: wp.array2d(dtype=float),
            root_pos_xz_buf: wp.array2d(dtype=float),
            root_yaw_inv_buf: wp.array2d(dtype=float),
            obs: wp.array2d(dtype=float),
        ):
            env_idx = wp.tid()
            clip_idx = motion_clip_idx[env_idx]
            fi = ref_frame_idx[env_idx]
            fi_local = clip_local_frame(fi, clip_idx, clip_length)
            fi_cycles = clip_cycle_count(fi, clip_idx, clip_length)
            fi_global = clip_start[clip_idx] + fi_local
            loop_root_q = normalize_quat_wp(
                read_quat_from_2d(clip_loop_root_quat, clip_idx, 0)
            )
            loop_root_t = read_vec3_from_2d(clip_loop_root_pos, clip_idx, 0)

            if wp.static(ROOT_QPOS_COUNT == 7):
                root_q = normalize_quat_wp(read_quat_from_2d(qpos, env_idx, 3))
                root_pos_xz = wp.vec3(qpos[env_idx, 0], qpos[env_idx, 1], 0.0)
                base_ref_root_q = normalize_quat_wp(
                    read_quat_from_2d(motion_qpos, fi_global, 3)
                )
                base_ref_root_pos = wp.vec3(
                    motion_qpos[fi_global, 0],
                    motion_qpos[fi_global, 1],
                    motion_qpos[fi_global, 2],
                )
            else:
                root_q = normalize_quat_wp(xquat[env_idx, ROOT_BODY_IDX])
                sim_root_pos = xpos[env_idx, ROOT_BODY_IDX]
                root_pos_xz = wp.vec3(sim_root_pos[0], sim_root_pos[1], 0.0)
                base_ref_root_q = normalize_quat_wp(
                    wp.quat(
                        reference_body_state[fi_global, ROOT_BODY_IDX, 3],
                        reference_body_state[fi_global, ROOT_BODY_IDX, 4],
                        reference_body_state[fi_global, ROOT_BODY_IDX, 5],
                        reference_body_state[fi_global, ROOT_BODY_IDX, 6],
                    )
                )
                base_ref_root_pos = wp.vec3(
                    reference_body_state[fi_global, ROOT_BODY_IDX, 0],
                    reference_body_state[fi_global, ROOT_BODY_IDX, 1],
                    reference_body_state[fi_global, ROOT_BODY_IDX, 2],
                )
            yaw_q = heading_yaw_quat(root_q)
            yaw_inv = quat_inv(yaw_q)
            ref_root_q = normalize_quat_wp(
                apply_loop_transform_quat(base_ref_root_q, loop_root_q, fi_cycles)
            )
            ref_yaw_q = heading_yaw_quat(ref_root_q)
            ref_root_pos = apply_loop_transform_vec(
                base_ref_root_pos, loop_root_q, loop_root_t, fi_cycles
            )
            ref_root_xz = wp.vec3(ref_root_pos[0], ref_root_pos[1], 0.0)

            target_pos = rot_vec_quat(ref_root_xz - root_pos_xz, yaw_inv)
            target_rot_q = normalize_quat_wp(mul_quat(yaw_inv, ref_yaw_q))

            write_vec3(root_pos_xz_buf, env_idx, 0, root_pos_xz, 1.0)
            write_quat4(root_yaw_inv_buf, env_idx, 0, yaw_inv)
            if ROOT_TARGET_DIM > 0:
                write_vec3(obs, env_idx, 0, target_pos, 1.0)
                write_ortho6d(obs, env_idx, 3, target_rot_q)

        root_target_kernel = write_root_target_observation_multi_motion_kernel

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def write_joint_position_observation_kernel(
        qpos: wp.array2d(dtype=float),
        xquat: wp.array2d(dtype=wp.quat),
        obs_joint_pos_qpos_idx: wp.array(dtype=int),
        obs_joint_pos_body_idx: wp.array(dtype=int),
        obs_joint_pos_parent_body_idx: wp.array(dtype=int),
        obs_joint_pos_component_idx: wp.array(dtype=int),
        obs_joint_pos_encoding: wp.array(dtype=int),
        obs: wp.array2d(dtype=float),
    ):
        env_idx, joint_idx = wp.tid()
        joint_value = qpos[env_idx, obs_joint_pos_qpos_idx[joint_idx]]
        encoding = obs_joint_pos_encoding[joint_idx]
        if encoding != 0:
            bi = obs_joint_pos_body_idx[joint_idx]
            pi = obs_joint_pos_parent_body_idx[joint_idx]
            comp = obs_joint_pos_component_idx[joint_idx]
            sim_bq = normalize_quat_wp(xquat[env_idx, bi])
            sim_pq = normalize_quat_wp(xquat[env_idx, pi])
            sim_local_q = local_quat_wp(sim_pq, sim_bq)
            if encoding == 1:
                joint_value = sim_local_q[comp]
            else:
                joint_value = ortho6d_component(sim_local_q, comp)
        obs[env_idx, JOINT_POS_OFS + joint_idx] = joint_value

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def write_joint_velocity_observation_kernel(
        qvel: wp.array2d(dtype=float),
        xquat: wp.array2d(dtype=wp.quat),
        cvel: wp.array2d(dtype=wp.spatial_vector),
        obs_joint_vel_local_idx: wp.array(dtype=int),
        obs_joint_vel_use_local: wp.array(dtype=int),
        obs_joint_vel_body_idx: wp.array(dtype=int),
        obs_joint_vel_parent_body_idx: wp.array(dtype=int),
        obs_joint_vel_axis: wp.array(dtype=wp.vec3),
        obs: wp.array2d(dtype=float),
    ):
        env_idx, vel_slot = wp.tid()
        # Slot -> non-root DOF index: passive DOFs are not observed.
        vel_idx = obs_joint_vel_local_idx[vel_slot]
        vel_value = qvel[env_idx, vel_idx + ROOT_DOF_COUNT]
        if wp.static(JOINT_VELOCITY_OBS_MODE >= 1):
            if obs_joint_vel_use_local[vel_idx] != 0:
                bi = obs_joint_vel_body_idx[vel_idx]
                pi = obs_joint_vel_parent_body_idx[vel_idx]
                axis = obs_joint_vel_axis[vel_idx]
                sim_bq = normalize_quat_wp(xquat[env_idx, bi])
                sim_bv = cvel[env_idx, bi]
                sim_pv = cvel[env_idx, pi]
                sim_rel_av_local = rot_vec_quat(
                    wp.vec3(
                        sim_bv[0] - sim_pv[0],
                        sim_bv[1] - sim_pv[1],
                        sim_bv[2] - sim_pv[2],
                    ),
                    quat_inv(sim_bq),
                )
                vel_value = wp.dot(sim_rel_av_local, axis)
        obs[env_idx, JOINT_VEL_OFS + vel_slot] = vel_value * 0.1

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def write_body_state_observation_kernel(
        observation_body_idx: wp.array(dtype=wp.int32),
        xpos: wp.array2d(dtype=wp.vec3),
        xquat: wp.array2d(dtype=wp.quat),
        cvel: wp.array2d(dtype=wp.spatial_vector),
        root_pos_xz_buf: wp.array2d(dtype=float),
        root_yaw_inv_buf: wp.array2d(dtype=float),
        obs: wp.array2d(dtype=float),
    ):
        env_idx, observation_body_slot = wp.tid()
        body_idx = observation_body_idx[observation_body_slot]
        yaw_inv = normalize_quat_wp(read_quat_from_2d(root_yaw_inv_buf, env_idx, 0))
        root_pos_xz = wp.vec3(
            root_pos_xz_buf[env_idx, 0],
            root_pos_xz_buf[env_idx, 1],
            root_pos_xz_buf[env_idx, 2],
        )

        bp = xpos[env_idx, body_idx]
        bq = normalize_quat_wp(xquat[env_idx, body_idx])
        cv = cvel[env_idx, body_idx]
        bv = wp.vec3(cv[3], cv[4], cv[5])
        bw = wp.vec3(cv[0], cv[1], cv[2])

        pos_rel = rot_vec_quat(bp - root_pos_xz, yaw_inv)
        quat_rel = normalize_quat_wp(mul_quat(yaw_inv, bq))
        linvel_rel = rot_vec_quat(bv, yaw_inv)
        angvel_rel = rot_vec_quat(bw, yaw_inv)

        write_vec3(obs, env_idx, BODY_POS_OFS + observation_body_slot * 3, pos_rel, 1.0)
        write_ortho6d(obs, env_idx, BODY_ROT_OFS + observation_body_slot * 6, quat_rel)
        write_vec3(
            obs, env_idx, BODY_LV_OFS + observation_body_slot * 3, linvel_rel, 0.1
        )
        write_vec3(
            obs, env_idx, BODY_AV_OFS + observation_body_slot * 3, angvel_rel, 0.1
        )

    future_obs_kernel = make_future_observation_kernel(
        FutureObservationKernelSpec(
            motion_frame_count=motion_frame_count,
            future_obs_ofs=int(layout.future_obs_ofs),
            future_obs_dim=int(layout.future_obs_dim),
            future_observation_mode=layout.future_observation_mode,
            num_future_frames=int(layout.num_future_frames),
            root_body_idx=int(root_body_idx),
            future_pose_body_idx=tuple(int(v) for v in future_pose_body_idx),
            future_frame_step=int(future_frame_step),
            loop_root_q=tuple(float(v) for v in loop_root_q),
            loop_root_t=tuple(float(v) for v in loop_root_t),
            loop_cycle_table_size=int(loop_cycle_table_size),
            multi_motion=bool(multi_motion),
        )
    )

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def write_previous_action_observation_kernel(
        actions: wp.array2d(dtype=float),
        obs: wp.array2d(dtype=float),
    ):
        env_idx, action_idx = wp.tid()
        obs[env_idx, ACTION_OFS + action_idx] = actions[env_idx, action_idx]

    kernels = ObservationKernels(
        layout=layout,
        root_target=root_target_kernel,
        joint_pos=write_joint_position_observation_kernel,
        joint_vel=write_joint_velocity_observation_kernel,
        body_state=write_body_state_observation_kernel,
        future_obs=future_obs_kernel,
        prev_actions=write_previous_action_observation_kernel,
    )
    _OBSERVATION_KERNEL_CACHE[key] = kernels
    return kernels


def build_joint_position_observation_indices(self):
    joint_pos_qpos_idx: list[int] = []
    joint_pos_body_idx: list[int] = []
    joint_pos_parent_body_idx: list[int] = []
    joint_pos_component_idx: list[int] = []
    joint_pos_encoding: list[int] = []

    if self.observation_settings.include_joint_state:
        seen_multi_hinge_body: set[int] = set()
        multi_hinge_body_mask = set(
            self.hinge_body_indices[self.hinge_multi_body_mask].detach().cpu().tolist()
        )
        # Character joints only: the root and passive coordinates
        # (instantmimic.envs.dof_roles) are not joint observations.
        for jnt_id in self.dof_roles.joint_ids:
            jnt_type = self.model.jnt_type[jnt_id]
            qpos_adr = int(self.model.jnt_qposadr[jnt_id])
            body_id = int(self.model.jnt_bodyid[jnt_id])
            parent_body_id = int(self.model.body_parentid[body_id])

            if (
                self.observation_settings.joint_pose_mode in (1, 2)
                and jnt_type == mujoco.mjtJoint.mjJNT_HINGE
                and body_id in multi_hinge_body_mask
            ):
                if body_id in seen_multi_hinge_body:
                    continue
                seen_multi_hinge_body.add(body_id)
                local_dim = 6 if self.observation_settings.joint_pose_mode == 2 else 4
                encoding = 2 if self.observation_settings.joint_pose_mode == 2 else 1
                for comp in range(local_dim):
                    joint_pos_qpos_idx.append(0)
                    joint_pos_body_idx.append(body_id)
                    joint_pos_parent_body_idx.append(parent_body_id)
                    joint_pos_component_idx.append(comp)
                    joint_pos_encoding.append(encoding)
                continue

            if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                if self.observation_settings.joint_pose_mode == 2:
                    for comp in range(6):
                        joint_pos_qpos_idx.append(0)
                        joint_pos_body_idx.append(body_id)
                        joint_pos_parent_body_idx.append(parent_body_id)
                        joint_pos_component_idx.append(comp)
                        joint_pos_encoding.append(2)
                    continue
                for comp in range(4):
                    joint_pos_qpos_idx.append(qpos_adr + comp)
                    joint_pos_body_idx.append(0)
                    joint_pos_parent_body_idx.append(0)
                    joint_pos_component_idx.append(0)
                    joint_pos_encoding.append(0)
            elif jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                joint_pos_qpos_idx.append(qpos_adr)
                joint_pos_body_idx.append(0)
                joint_pos_parent_body_idx.append(0)
                joint_pos_component_idx.append(0)
                joint_pos_encoding.append(0)

    self.obs_joint_pos_qpos_idx_i32 = torch.tensor(
        joint_pos_qpos_idx, device=self.device, dtype=torch.int32
    )
    self.obs_joint_pos_body_idx_i32 = torch.tensor(
        joint_pos_body_idx, device=self.device, dtype=torch.int32
    )
    self.obs_joint_pos_parent_body_idx_i32 = torch.tensor(
        joint_pos_parent_body_idx, device=self.device, dtype=torch.int32
    )
    self.obs_joint_pos_component_idx_i32 = torch.tensor(
        joint_pos_component_idx, device=self.device, dtype=torch.int32
    )
    self.obs_joint_pos_encoding_i32 = torch.tensor(
        joint_pos_encoding, device=self.device, dtype=torch.int32
    )
    self.observation_runtime_state.joint_position_count = int(
        self.obs_joint_pos_qpos_idx_i32.numel()
    )


def configure_obs_joint_velocity_projection(self):
    self.obs_joint_vel_body_idx_i32.zero_()
    self.obs_joint_vel_parent_body_idx_i32.zero_()
    self.obs_joint_vel_axis.zero_()
    self.obs_joint_vel_use_local_i32.zero_()

    if self.hinge_dof_indices.numel() <= 0:
        return

    self.obs_joint_vel_body_idx_i32[self.hinge_dof_indices] = (
        self.hinge_body_indices.to(torch.int32)
    )
    self.obs_joint_vel_parent_body_idx_i32[self.hinge_dof_indices] = (
        self.hinge_parent_body_indices.to(torch.int32)
    )
    self.obs_joint_vel_axis[self.hinge_dof_indices] = self.hinge_axis_local
    if self.observation_settings.joint_velocity_mode == 1:
        self.obs_joint_vel_use_local_i32[self.hinge_dof_indices] = 1
    elif self.observation_settings.joint_velocity_mode == 2:
        self.obs_joint_vel_use_local_i32[self.hinge_dof_indices] = (
            self.hinge_multi_body_mask.to(torch.int32)
        )


def get_observation_parallel_streams(self) -> tuple[torch.cuda.Stream, wp.Stream]:
    state = self.observation_runtime_state
    if state.parallel_torch_stream is None or state.parallel_wp_stream is None:
        device = getattr(self, "device", torch.device("cuda"))
        (
            state.parallel_torch_stream,
            state.parallel_wp_stream,
        ) = make_stream_aligned_cuda_graph_streams(device)
    return state.parallel_torch_stream, state.parallel_wp_stream


def get_future_observation_parallel_streams(
    self,
) -> tuple[torch.cuda.Stream, wp.Stream]:
    state = self.observation_runtime_state
    if (
        state.future_parallel_torch_stream is None
        or state.future_parallel_wp_stream is None
    ):
        device = getattr(self, "device", torch.device("cuda"))
        (
            state.future_parallel_torch_stream,
            state.future_parallel_wp_stream,
        ) = make_stream_aligned_cuda_graph_streams(device)
    return (
        state.future_parallel_torch_stream,
        state.future_parallel_wp_stream,
    )


def write_joint_observation_slices(self, target: torch.Tensor) -> None:
    if self._observation_layout.joint_q_count > 0:
        wp.launch(
            self.observation_runtime_state.kernels.joint_pos,
            dim=(self.num_envs, self._observation_layout.joint_q_count),
            inputs=[
                self.mjw_data.qpos,
                self.mjw_data.xquat,
                self.obs_joint_pos_qpos_idx_i32,
                self.obs_joint_pos_body_idx_i32,
                self.obs_joint_pos_parent_body_idx_i32,
                self.obs_joint_pos_component_idx_i32,
                self.obs_joint_pos_encoding_i32,
            ],
            outputs=[target],
        )
    if self._observation_layout.joint_v_count > 0:
        wp.launch(
            self.observation_runtime_state.kernels.joint_vel,
            dim=(self.num_envs, self._observation_layout.joint_v_count),
            inputs=[
                self.mjw_data.qvel,
                self.mjw_data.xquat,
                self.mjw_data.cvel,
                self.joint_dof_local_i32,
                self.obs_joint_vel_use_local_i32,
                self.obs_joint_vel_body_idx_i32,
                self.obs_joint_vel_parent_body_idx_i32,
                self.obs_joint_vel_axis,
            ],
            outputs=[target],
        )


def write_prev_action_observation_slice(
    self, target: torch.Tensor, *, prev_actions: torch.Tensor
) -> None:
    if int(self._observation_layout.obs_dim) > int(self._observation_layout.action_ofs):
        wp.launch(
            self.observation_runtime_state.kernels.prev_actions,
            dim=(self.num_envs, int(prev_actions.shape[-1])),
            inputs=[prev_actions],
            outputs=[target],
        )


def write_parallel_observation_slices(
    self, target: torch.Tensor, *, prev_actions: torch.Tensor
) -> None:
    write_joint_observation_slices(self, target)
    write_prev_action_observation_slice(self, target, prev_actions=prev_actions)


def write_core_observations(
    self, target: torch.Tensor, *, prev_actions: torch.Tensor
) -> None:
    env_device = torch.device(getattr(self, "device", target.device))
    parallel_slices = env_device.type == "cuda"
    future_obs_pending = False
    if parallel_slices:
        current_stream = torch.cuda.current_stream(env_device)
        obs_torch_stream, obs_wp_stream = get_observation_parallel_streams(
            self,
        )
        obs_torch_stream.wait_stream(current_stream)
        with wp.ScopedStream(obs_wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(obs_torch_stream):
                write_parallel_observation_slices(
                    self, target, prev_actions=prev_actions
                )

    if self._multi_motion_enabled:
        wp.launch(
            self.observation_runtime_state.kernels.root_target,
            dim=(self.num_envs,),
            inputs=[
                self.mjw_data.qpos,
                self.motion_qpos,
                self.mjw_data.xpos,
                self.mjw_data.xquat,
                getattr(self, "reference_body_state", self.motion_qpos),
                self.motion_clip_idx,
                self.ref_frame_idx,
                self.motion_clip_start,
                self.motion_clip_length,
                self.motion_clip_loop_root_quat,
                self.motion_clip_loop_root_pos,
            ],
            outputs=[
                self.observation_runtime_state.root_position_xz,
                self.observation_runtime_state.root_yaw_inverse,
                target,
            ],
        )
    else:
        wp.launch(
            self.observation_runtime_state.kernels.root_target,
            dim=(self.num_envs,),
            inputs=[
                self.mjw_data.qpos,
                self.motion_qpos,
                self.mjw_data.xpos,
                self.mjw_data.xquat,
                getattr(self, "reference_body_state", self.motion_qpos),
                self.ref_frame_idx,
            ],
            outputs=[
                self.observation_runtime_state.root_position_xz,
                self.observation_runtime_state.root_yaw_inverse,
                target,
            ],
        )
    if parallel_slices and getattr(
        self.observation_runtime_state.kernels, "future_obs", None
    ):
        (
            future_torch_stream,
            future_wp_stream,
        ) = get_future_observation_parallel_streams(
            self,
        )
        future_torch_stream.wait_stream(current_stream)
        with wp.ScopedStream(future_wp_stream, sync_enter=False, sync_exit=False):
            with torch.cuda.stream(future_torch_stream):
                launch_future_observation(self, target)
        future_obs_pending = True
    if not parallel_slices:
        write_joint_observation_slices(self, target)
    observation_body_idx_tuple = self.observation_body_idx_tuple
    observation_body_idx = self.observation_body_idx
    wp.launch(
        self.observation_runtime_state.kernels.body_state,
        dim=(self.num_envs, len(observation_body_idx_tuple)),
        inputs=[
            observation_body_idx,
            self.mjw_data.xpos,
            self.mjw_data.xquat,
            self.mjw_data.cvel,
            self.observation_runtime_state.root_position_xz,
            self.observation_runtime_state.root_yaw_inverse,
        ],
        outputs=[target],
    )
    if not future_obs_pending:
        launch_future_observation(self, target)
    if parallel_slices:
        if future_obs_pending:
            current_stream.wait_stream(future_torch_stream)
        current_stream.wait_stream(obs_torch_stream)
    else:
        write_prev_action_observation_slice(self, target, prev_actions=prev_actions)


def get_observations(self) -> dict:
    target = self.obs_buf["policy"]
    obs_main_dim = int(getattr(self, "_obs_main_dim", target.shape[-1]))
    target_main = (
        target if target.shape[-1] == obs_main_dim else target[:, :obs_main_dim]
    )
    control_pipeline = getattr(self, "_control_pipeline", None)
    get_prev_actions = getattr(
        control_pipeline, "get_prev_action_observation_tensor", None
    )
    prev_actions = get_prev_actions() if callable(get_prev_actions) else self.actions
    write_core_observations(self, target_main, prev_actions=prev_actions)
    write_extension = getattr(control_pipeline, "write_observation_extension", None)
    if callable(write_extension):
        write_extension(target, obs_main_dim)
    return self.obs_buf


__all__ = (
    "ObservationLayout",
    "ObservationKernels",
    "ObservationRuntimeState",
    "ObservationSettings",
    "resolve_observation_settings",
    "make_observation_layout",
    "make_observation_kernels",
    "resolve_joint_pose_mode",
    "resolve_joint_velocity_mode",
    "initialize_observation_state",
    "initialize_observation_kernels",
    "sync_runtime_spaces_to_model",
    "finalize_task_control_pipeline",
    "initialize_observation_core",
    "initialize_loop_cycle_transform_cache",
    "get_future_observation_mode",
    "build_observation_kernels_for_prev_action_dim",
    "launch_future_observation",
    "build_joint_position_observation_indices",
    "configure_obs_joint_velocity_projection",
    "get_observation_parallel_streams",
    "get_future_observation_parallel_streams",
    "write_joint_observation_slices",
    "write_prev_action_observation_slice",
    "write_parallel_observation_slices",
    "write_core_observations",
    "get_observations",
)


def initialize_observation_state(self):
    self.observation_runtime_state = ObservationRuntimeState()
    startup_profiler = StartupProfiler(
        range_factory=lambda name: startup_nvtx_range(f"observation_state:{name}")
    )
    with startup_profiler.phase("resolve_prev_action_dim"):
        get_prev_action_obs_dim = getattr(
            self._control_pipeline, "get_prev_action_observation_dim", None
        )
        prev_action_dim = (
            int(get_prev_action_obs_dim())
            if callable(get_prev_action_obs_dim)
            else int(self.model.nu)
        )
    with startup_profiler.phase("initialize_observation_core"):
        initialize_observation_core(self, prev_action_dim=prev_action_dim)
    with startup_profiler.phase("sync_runtime_spaces"):
        self._obs_main_dim = int(self._observation_layout.obs_dim)
        sync_runtime_spaces_to_model(
            self,
        )
    with startup_profiler.phase("obs_buffer_alloc"):
        self._obs_buf = torch.zeros(
            self.num_envs,
            int(self._obs_main_dim),
            device=self.device,
            dtype=torch.float32,
        )
    startup_profiler.log_summary(prefix="OBSERVATION_STARTUP_PROFILE")


def sync_runtime_spaces_to_model(self) -> None:
    action_dim = int(self.model.nu)
    obs_dim = int(self._observation_layout.obs_dim)
    if (
        not bool(getattr(self._control_pipeline, "is_enabled", False))
        and int(self.cfg.action_space) != action_dim
    ):
        self.cfg.action_space = action_dim
    self._obs_main_dim = obs_dim


def finalize_task_control_pipeline(self) -> None:
    startup_profiler = StartupProfiler(
        range_factory=lambda name: startup_nvtx_range(f"task_control_finalize:{name}")
    )
    with startup_profiler.phase("configure_spaces"):
        self._control_pipeline.configure_spaces(self)

    with startup_profiler.phase("obs_buffer_resize"):
        obs_dim = int(getattr(self, "_obs_main_dim", self._observation_layout.obs_dim))
        controller = getattr(self, "fast_muscle", None)
        if controller is not None:
            obs_dim += int(controller.muscle_geometry_observation_dim)
        if self._obs_buf.shape != (self.num_envs, obs_dim):
            self._obs_buf = torch.zeros(
                self.num_envs,
                obs_dim,
                device=self.device,
                dtype=torch.float32,
            )

    self.obs_buf = {"policy": self._obs_buf}
    with startup_profiler.phase("finalize_derived_spaces"):
        self.finalize_derived_spaces(
            observation_spec=int(self._obs_buf.shape[-1]),
            action_spec=self.cfg.action_space,
            observations=self.obs_buf,
        )
    with startup_profiler.phase("control_finalize_task_state"):
        finalizer = getattr(self._control_pipeline, "finalize_task_state", None)
        if callable(finalizer):
            finalizer(self)
    startup_profiler.log_summary(prefix="TASK_CONTROL_STARTUP_PROFILE")


def initialize_observation_core(self, *, prev_action_dim: int) -> None:
    startup_profiler = StartupProfiler(
        range_factory=lambda name: startup_nvtx_range(f"observation_core:{name}")
    )
    with startup_profiler.phase("joint_position_indices"):
        build_joint_position_observation_indices(
            self,
        )
    with startup_profiler.phase("future_body_indices"):
        future_observation_mode = get_future_observation_mode(
            self,
        )
        reward_body_idx_tuple = getattr(self, "reward_body_idx_tuple", None)
        if reward_body_idx_tuple is None:
            reward_body_idx_tuple = tuple(
                int(v) for v in self.reward_body_idx.detach().cpu().tolist()
            )
            self.reward_body_idx_tuple = reward_body_idx_tuple
        root_body_idx = int(self.root_body_idx)
        if root_body_idx in reward_body_idx_tuple:
            self.future_pose_body_idx = self.reward_body_idx
            future_pose_body_idx_tuple = reward_body_idx_tuple
        else:
            self.future_pose_body_idx = torch.cat(
                (
                    torch.tensor(
                        [root_body_idx],
                        device=self.device,
                        dtype=self.reward_body_idx.dtype,
                    ),
                    self.reward_body_idx,
                ),
            )
            future_pose_body_idx_tuple = (root_body_idx, *reward_body_idx_tuple)
        self.future_pose_body_idx_tuple = tuple(
            int(v) for v in future_pose_body_idx_tuple
        )
        body_source = self.observation_settings.body_source
        if body_source == "all":
            # Every body but the passive ones (instantmimic.envs.dof_roles):
            # they have no reference and are not observed.
            passive_bodies = frozenset(self.dof_roles.passive_bodies)
            observation_body_idx_tuple = tuple(
                body
                for body in range(int(self.model.nbody))
                if body not in passive_bodies
            )
        elif body_source == "reward":
            observation_body_idx_tuple = reward_body_idx_tuple
        else:
            raise ValueError(
                "observations.body_source must be either 'all' or 'reward'."
            )
        if not observation_body_idx_tuple:
            raise ValueError("Body-state observations require at least one body.")
        self.observation_body_idx_tuple = tuple(
            int(v) for v in observation_body_idx_tuple
        )
        self.observation_body_idx = torch.tensor(
            self.observation_body_idx_tuple,
            device=self.device,
            dtype=torch.int32,
        )
        if future_observation_uses_phc_v6(future_observation_mode) and (
            self.future_pose_body_idx.numel() == 0
        ):
            raise ValueError(
                "future_observation_mode requires future pose bodies, but none were found."
            )
    with startup_profiler.phase("layout_and_buffers"):
        self._observation_layout = make_observation_layout(
            nbody=len(self.observation_body_idx_tuple),
            nq=self.model.nq,
            nv=self.model.nv,
            nu=int(prev_action_dim),
            num_future_pose_bodies=len(self.future_pose_body_idx),
            num_future_frames=self.observation_settings.future_frames,
            future_observation_mode=future_observation_mode,
            include_root_target_obs=self.observation_settings.include_root_target,
            include_joint_state=self.observation_settings.include_joint_state,
            joint_q_count_override=self.observation_runtime_state.joint_position_count,
            joint_v_count_override=len(self.dof_roles.joint_dof),
        )
        self.observation_runtime_state.root_position_xz = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )
        self.observation_runtime_state.root_yaw_inverse = torch.zeros(
            self.num_envs, 4, device=self.device, dtype=torch.float32
        )
    with startup_profiler.phase("loop_cycle_cache"):
        initialize_loop_cycle_transform_cache(
            self,
        )
    startup_profiler.log_summary(prefix="OBSERVATION_CORE_STARTUP_PROFILE")


def initialize_loop_cycle_transform_cache(self) -> None:
    if bool(getattr(self, "_multi_motion_enabled", False)):
        self.observation_runtime_state.loop_cycle_table_size = 0
        self.observation_runtime_state.loop_cycle_transform_quat = torch.empty(
            0, 4, device=self.device, dtype=torch.float32
        )
        self.observation_runtime_state.loop_cycle_transform_pos = torch.empty(
            0, 3, device=self.device, dtype=torch.float32
        )
        return

    max_cycles = 0
    if (
        self._observation_layout.future_obs_dim > 0
        and future_observation_uses_phc_v6(
            self._observation_layout.future_observation_mode
        )
        and int(self.motion_frame_count) > 0
    ):
        num_future_frames = max(0, self.observation_settings.future_frames)
        future_frame_step = self.observation_settings.future_frame_step
        if num_future_frames == 0:
            max_future_offset = 0
        else:
            max_future_offset = 1 + (num_future_frames - 1) * future_frame_step
        try:
            max_episode_length = int(self.max_episode_length)
        except (AttributeError, KeyError, TypeError):
            max_episode_length = int(self.motion_frame_count)
        max_future_frame_idx = (
            int(self.motion_frame_count) - 1 + max_episode_length + max_future_offset
        )
        max_cycles = max(0, max_future_frame_idx // int(self.motion_frame_count))

    loop_root_quat = torch.tensor(
        getattr(self, "loop_root_quat_tuple", None)
        or tuple(float(v) for v in self.loop_root_quat.detach().cpu().tolist()),
        dtype=torch.float32,
    )
    loop_root_pos = torch.tensor(
        getattr(self, "loop_root_pos_tuple", None)
        or tuple(float(v) for v in self.loop_root_pos.detach().cpu().tolist()),
        dtype=torch.float32,
    )
    cycle_quat, cycle_pos = build_loop_cycle_transform_tables(
        loop_root_quat=loop_root_quat,
        loop_root_pos=loop_root_pos,
        max_cycles=max_cycles,
    )
    self.observation_runtime_state.loop_cycle_transform_quat = cycle_quat.to(
        device=self.device, non_blocking=True
    )
    self.observation_runtime_state.loop_cycle_transform_pos = cycle_pos.to(
        device=self.device, non_blocking=True
    )
    self.observation_runtime_state.loop_cycle_table_size = int(
        self.observation_runtime_state.loop_cycle_transform_quat.shape[0]
    )


def get_future_observation_mode(self) -> str:
    return self.observation_settings.future_mode


def build_observation_kernels_for_prev_action_dim(
    self, *, prev_action_dim: int
) -> ObservationKernels:
    loop_root_q_tuple = getattr(self, "loop_root_quat_tuple", None)
    if loop_root_q_tuple is None:
        loop_root_q_tuple = tuple(
            float(v) for v in self.loop_root_quat.detach().cpu().tolist()
        )
    loop_root_t_tuple = getattr(self, "loop_root_pos_tuple", None)
    if loop_root_t_tuple is None:
        loop_root_t_tuple = tuple(
            float(v) for v in self.loop_root_pos.detach().cpu().tolist()
        )
    future_pose_body_idx_tuple = getattr(self, "future_pose_body_idx_tuple", None)
    if future_pose_body_idx_tuple is None:
        future_pose_body_idx_tuple = tuple(
            int(v) for v in self.future_pose_body_idx_i32.detach().cpu().tolist()
        )
    return make_observation_kernels(
        motion_frame_count=self.motion_frame_count,
        layout=self._observation_layout,
        nbody=self.model.nbody,
        nu=int(prev_action_dim),
        root_body_idx=int(self.root_body_idx),
        future_pose_body_idx=future_pose_body_idx_tuple,
        future_frame_step=self.observation_settings.future_frame_step,
        loop_root_q=loop_root_q_tuple,
        loop_root_t=loop_root_t_tuple,
        loop_cycle_table_size=int(self.observation_runtime_state.loop_cycle_table_size),
        joint_velocity_obs_mode=int(self.observation_settings.joint_velocity_mode),
        observation_body_idx=self.observation_body_idx_tuple,
        root_qpos_count=int(getattr(self, "root_qpos_count", 7)),
        root_dof_count=int(getattr(self, "root_dof_count", 6)),
        multi_motion=bool(self._multi_motion_enabled),
    )


def launch_future_observation(self, target: torch.Tensor) -> None:
    if self.observation_runtime_state.kernels.future_obs is None:
        return
    if self._multi_motion_enabled:
        wp.launch(
            self.observation_runtime_state.kernels.future_obs,
            dim=(self.num_envs,),
            inputs=[
                self.reference_body_state,
                self.motion_clip_idx,
                self.ref_frame_idx,
                self.motion_clip_start,
                self.motion_clip_length,
                self.motion_clip_loop_root_quat,
                self.motion_clip_loop_root_pos,
                self.mjw_data.xpos,
                self.mjw_data.xquat,
                self.mjw_data.cvel,
                self.observation_runtime_state.root_yaw_inverse,
            ],
            outputs=[target],
        )
    else:
        layout = self._observation_layout
        wp.launch(
            self.observation_runtime_state.kernels.future_obs,
            dim=(
                self.num_envs,
                int(layout.num_future_frames),
                int(layout.num_future_pose_bodies),
            ),
            inputs=[
                self.reference_body_state,
                self.ref_frame_idx,
                self.future_pose_body_idx_i32,
                self.observation_runtime_state.loop_cycle_transform_quat,
                self.observation_runtime_state.loop_cycle_transform_pos,
                self.mjw_data.xpos,
                self.mjw_data.xquat,
                self.mjw_data.cvel,
                self.observation_runtime_state.root_yaw_inverse,
            ],
            outputs=[target],
        )
