from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from instantmimic.envs.config_access import get_value
from instantmimic.envs.controls.fast_muscle.render_builder import (
    build_ghost_render_model,
)
from instantmimic.envs.controls.fast_muscle.render_utils import (
    apply_character_geom_render_visibility,
)
from instantmimic.envs.freemusco.tasks.freemusco_imitation.mocap_clip import (
    MocapClip,
    MocapLibrary,
    compute_loop_root_transform_np,
    mirror_mocap_clip,
    smooth_mocap_library,
)
from instantmimic.envs.model_index import resolve_id
from instantmimic.envs.tracking.observation_spec import build_deployable_obs_spec
from instantmimic.envs.tracking.warp_features import (
    write_ortho6d,
    write_tan_norm,
    write_vec3,
)
from instantmimic.motion.quaternion import quat_mul_np
from instantmimic.motion.warp_math import heading_yaw_mjw_quat as heading_yaw_quat
from instantmimic.motion.warp_math import normalize_mjw_quat as normalize_quat_wp
from instantmimic.motion.warp_utils import (
    apply_loop_rotation_vec,
    apply_loop_transform_quat,
    apply_loop_transform_vec,
    loop_cycle_count,
    loop_local_frame,
)
from instantmimic.utils.mjw_kernels.private import mul_quat, quat_inv, rot_vec_quat
from instantmimic.utils.warp_kernels import expose_kernel_name


def _required_profile_name(profile: Any, key: str) -> str:
    value = str(get_value(profile, key, "")).strip()
    if not value:
        raise ValueError(f"Freemusco imitation profile requires {key!r}.")
    return value


def _profile_names(
    profile: Any,
    key: str,
    *,
    expected_count: int | None = None,
) -> tuple[str, ...]:
    values = tuple(str(value).strip() for value in (get_value(profile, key, ()) or ()))
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            f"Freemusco imitation profile {key!r} must contain exactly "
            f"{expected_count} names, got {len(values)}."
        )
    if any(not value for value in values):
        raise ValueError(f"Freemusco imitation profile {key!r} contains an empty name.")
    return values


def _profile_ints(
    profile: Any,
    key: str,
    *,
    expected_count: int,
) -> tuple[int, ...]:
    values = tuple(int(value) for value in (get_value(profile, key, ()) or ()))
    if len(values) != expected_count:
        raise ValueError(
            f"Freemusco imitation profile {key!r} must contain exactly "
            f"{expected_count} indices, got {len(values)}."
        )
    if any(value < 0 for value in values):
        raise ValueError(
            f"Freemusco imitation profile {key!r} contains a negative index."
        )
    return values


@dataclass(frozen=True)
class FreemuscoImitationProfile:
    """Data-only body and motion capabilities consumed by the generic runtime."""

    root_body_name: str
    head_body_name: str
    foot_body_names: tuple[str, str]
    pelvis_geom_name: str
    end_effector_body_names: tuple[str, ...]
    talus_body_names: tuple[str, ...]
    toe_body_names: tuple[str, ...]
    reference_body_start: int
    reference_body_count_offset: int
    root_qpos_position_indices: tuple[int, int, int]
    root_qpos_rotation_indices: tuple[int, int, int]
    loop_vertical_axis: int
    phase_observation_mode: str
    loop_transform_mode: str
    mirror_mode: str
    mirror_left_prefix: str
    mirror_right_prefix: str
    mirror_clip_suffix: str
    smoothing_mode: str
    ghost_root_xpath: str
    ghost_root_description: str

    @classmethod
    def from_cfg(cls, cfg: Any) -> FreemuscoImitationProfile:
        raw_profile = get_value(cfg, "imitation_profile", None)
        if raw_profile is None:
            raise ValueError(
                "FreemuscoImitationEnv requires env.imitation_profile capabilities."
            )

        profile = cls(
            root_body_name=_required_profile_name(raw_profile, "root_body_name"),
            head_body_name=_required_profile_name(raw_profile, "head_body_name"),
            foot_body_names=_profile_names(
                raw_profile, "foot_body_names", expected_count=2
            ),
            pelvis_geom_name=_required_profile_name(raw_profile, "pelvis_geom_name"),
            end_effector_body_names=_profile_names(
                raw_profile, "end_effector_body_names"
            ),
            talus_body_names=_profile_names(raw_profile, "talus_body_names"),
            toe_body_names=_profile_names(raw_profile, "toe_body_names"),
            reference_body_start=int(get_value(raw_profile, "reference_body_start", 1)),
            reference_body_count_offset=int(
                get_value(raw_profile, "reference_body_count_offset", 0)
            ),
            root_qpos_position_indices=_profile_ints(
                raw_profile, "root_qpos_position_indices", expected_count=3
            ),
            root_qpos_rotation_indices=_profile_ints(
                raw_profile, "root_qpos_rotation_indices", expected_count=3
            ),
            loop_vertical_axis=int(get_value(raw_profile, "loop_vertical_axis", 2)),
            phase_observation_mode=str(
                get_value(raw_profile, "phase_observation_mode", "configurable")
            )
            .strip()
            .lower(),
            loop_transform_mode=str(
                get_value(raw_profile, "loop_transform_mode", "endpoint_root_pose")
            )
            .strip()
            .lower(),
            mirror_mode=str(get_value(raw_profile, "mirror_mode", "none"))
            .strip()
            .lower(),
            mirror_left_prefix=str(get_value(raw_profile, "mirror_left_prefix", "l_")),
            mirror_right_prefix=str(
                get_value(raw_profile, "mirror_right_prefix", "r_")
            ),
            mirror_clip_suffix=str(
                get_value(raw_profile, "mirror_clip_suffix", "__mirror")
            ),
            smoothing_mode=str(get_value(raw_profile, "smoothing_mode", "none"))
            .strip()
            .lower(),
            ghost_root_xpath=_required_profile_name(raw_profile, "ghost_root_xpath"),
            ghost_root_description=_required_profile_name(
                raw_profile, "ghost_root_description"
            ),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if self.reference_body_start < 0:
            raise ValueError("reference_body_start must be non-negative.")
        if self.reference_body_count_offset < 0:
            raise ValueError("reference_body_count_offset must be non-negative.")
        if not 0 <= self.loop_vertical_axis < 3:
            raise ValueError("loop_vertical_axis must be one of 0, 1, or 2.")
        if len(set(self.root_qpos_position_indices)) != 3:
            raise ValueError("root_qpos_position_indices must contain unique indices.")
        if len(set(self.root_qpos_rotation_indices)) != 3:
            raise ValueError("root_qpos_rotation_indices must contain unique indices.")
        if self.phase_observation_mode not in {"required", "configurable"}:
            raise ValueError(
                "phase_observation_mode must be 'required' or 'configurable', got "
                f"{self.phase_observation_mode!r}."
            )
        if self.loop_transform_mode not in {
            "endpoint_root_pose",
            "extrapolated_root_yaw",
        }:
            raise ValueError(
                "Unsupported imitation loop_transform_mode: "
                f"{self.loop_transform_mode!r}."
            )
        if self.mirror_mode not in {"none", "paired_joint_reflection"}:
            raise ValueError(
                f"Unsupported imitation mirror_mode: {self.mirror_mode!r}."
            )
        if not self.mirror_left_prefix or not self.mirror_right_prefix:
            raise ValueError("Mirror joint prefixes must be non-empty.")
        if self.mirror_left_prefix == self.mirror_right_prefix:
            raise ValueError("Mirror joint prefixes must be distinct.")
        if self.smoothing_mode not in {"none", "extrapolated_root_cycle"}:
            raise ValueError(
                f"Unsupported imitation smoothing_mode: {self.smoothing_mode!r}."
            )
        if not self.end_effector_body_names:
            raise ValueError("end_effector_body_names must contain at least one name.")


@wp.func
def _selected_clip_length(
    motion_clip_idx: wp.array(dtype=wp.int32),
    clip_length: wp.array(dtype=wp.int32),
    env_idx: int,
) -> int:
    clip_idx = int(motion_clip_idx[env_idx])
    return int(clip_length[clip_idx])


@wp.func
def _clamp_or_wrap_frame_idx(frame_idx: int, frame_count: int, motion_loop: int) -> int:
    if motion_loop != 0:
        return loop_local_frame(frame_idx, frame_count)
    if frame_idx < 0:
        return 0
    if frame_idx >= frame_count:
        return frame_count - 1
    return frame_idx


@wp.func
def _frame_cycle_count(frame_idx: int, frame_count: int, motion_loop: int) -> int:
    if motion_loop == 0:
        return 0
    return loop_cycle_count(frame_idx, frame_count)


@wp.kernel(enable_backward=False)
def advance_motion_frame_idx_kernel(
    frame_idx: wp.array(dtype=wp.int32),
    motion_clip_idx: wp.array(dtype=wp.int32),
    clip_length: wp.array(dtype=wp.int32),
    motion_loop: int,
):
    env_idx = wp.tid()
    frame_count = _selected_clip_length(motion_clip_idx, clip_length, env_idx)
    next_idx = frame_idx[env_idx] + 1
    if motion_loop == 0 and next_idx >= frame_count:
        next_idx = frame_count - 1
    frame_idx[env_idx] = next_idx


@wp.kernel(enable_backward=False)
def _imitation_termination_kernel(
    frame_idx: wp.array(dtype=wp.int32),
    motion_clip_idx: wp.array(dtype=wp.int32),
    clip_length: wp.array(dtype=wp.int32),
    reward_buf: wp.array(dtype=float),
    episode_length_buf: wp.array(dtype=wp.int64),
    motion_loop: int,
    reward_threshold: float,
    max_episode_length: int,
    terminated: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()
    frame_count = _selected_clip_length(motion_clip_idx, clip_length, env_idx)
    done = False
    if motion_loop == 0 and frame_idx[env_idx] >= frame_count - 1:
        done = True
    if reward_threshold > 0.0 and reward_buf[env_idx] < reward_threshold:
        done = True
    terminated[env_idx] = done
    time_out[env_idx] = int(episode_length_buf[env_idx]) >= max_episode_length


@wp.kernel(enable_backward=False)
def _reset_motion_state_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    random_seed: wp.array(dtype=int),
    motion_clip_idx: wp.array(dtype=wp.int32),
    clip_count: int,
    fixed_motion_idx: int,
    motion_qpos: wp.array2d(dtype=float),
    motion_qvel: wp.array2d(dtype=float),
    clip_start: wp.array(dtype=wp.int32),
    clip_length: wp.array(dtype=wp.int32),
    random_init: int,
    motion_loop: int,
    start_frame_idx: int,
    nq: int,
    nv: int,
    frame_idx_out: wp.array(dtype=wp.int32),
    qpos_out: wp.array2d(dtype=float),
    qvel_out: wp.array2d(dtype=float),
):
    env_idx = wp.tid()
    if not reset_mask[env_idx]:
        return

    clip_idx = int(motion_clip_idx[env_idx])
    state = wp.rand_init(random_seed[0], env_idx)
    if fixed_motion_idx >= 0:
        clip_idx = fixed_motion_idx
    else:
        clip_idx = int(wp.randf(state) * float(clip_count))
        if clip_idx >= clip_count:
            clip_idx = clip_count - 1
    motion_clip_idx[env_idx] = clip_idx
    frame_count = int(clip_length[clip_idx])
    frame_start = int(clip_start[clip_idx])

    frame_idx = 0
    if random_init != 0 and frame_count > 1:
        upper = frame_count if motion_loop != 0 else frame_count - 1
        if upper > 1:
            frame_idx = int(wp.randf(state) * float(upper))
            if frame_idx >= upper:
                frame_idx = upper - 1
    else:
        frame_idx = start_frame_idx
        if frame_idx < 0:
            frame_idx = 0
        if frame_idx >= frame_count:
            frame_idx = frame_count - 1

    frame_idx_out[env_idx] = frame_idx
    global_frame_idx = frame_start + frame_idx
    for j in range(nq):
        qpos_out[env_idx, j] = motion_qpos[global_frame_idx, j]
    for j in range(nv):
        qvel_out[env_idx, j] = motion_qvel[global_frame_idx, j]


def _make_observation_kernels(
    *,
    nbody: int,
    nbody_obs: int,
    root_body_idx: int,
    pelvis_geom_id: int,
    foot_body_id_0: int,
    foot_body_id_1: int,
    nq: int,
    nv: int,
    muscle_count: int,
    include_muscle: bool,
    include_phase: bool,
    motion_loop: bool,
    body_state_ofs: int,
    future_ref_ofs: int,
    phase_ofs: int,
):
    NBODY = int(nbody)
    NBODY_OBS = int(nbody_obs)
    ROOT_BODY_IDX = int(root_body_idx)
    PELVIS_GEOM_ID = int(pelvis_geom_id)
    FOOT_BODY_ID_0 = int(foot_body_id_0)
    FOOT_BODY_ID_1 = int(foot_body_id_1)
    NQ = int(nq)
    NV = int(nv)
    MUSCLE_COUNT = int(muscle_count)
    INCLUDE_MUSCLE = bool(include_muscle)
    INCLUDE_PHASE = bool(include_phase)
    MOTION_LOOP = int(motion_loop)
    BODY_STATE_OFS = int(body_state_ofs)
    FUTURE_REF_OFS = int(future_ref_ofs)
    PHASE_OFS = int(phase_ofs)
    TWO_PI = float(6.283185307179586)

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def obs_proprioception_kernel(
        xpos: wp.array2d(dtype=wp.vec3),
        xquat: wp.array2d(dtype=wp.quat),
        geom_xpos: wp.array2d(dtype=wp.vec3),
        qpos: wp.array2d(dtype=float),
        qvel: wp.array2d(dtype=float),
        cvel: wp.array2d(dtype=wp.spatial_vector),
        muscle_activation: wp.array2d(dtype=wp.float32),
        muscle_force: wp.array2d(dtype=wp.float32),
        muscle_length: wp.array2d(dtype=wp.float32),
        muscle_velocity: wp.array2d(dtype=wp.float32),
        frame_idx: wp.array(dtype=wp.int32),
        motion_clip_idx: wp.array(dtype=wp.int32),
        clip_length: wp.array(dtype=wp.int32),
        obs: wp.array2d(dtype=float),
    ):
        env_idx = wp.tid()
        selected_clip_length = _selected_clip_length(
            motion_clip_idx, clip_length, env_idx
        )
        local_frame_idx = _clamp_or_wrap_frame_idx(
            int(frame_idx[env_idx]), selected_clip_length, MOTION_LOOP
        )

        ofs = 0
        obs[env_idx, ofs] = geom_xpos[env_idx, PELVIS_GEOM_ID][2]
        ofs += 1
        obs[env_idx, ofs] = xpos[env_idx, FOOT_BODY_ID_0][2]
        ofs += 1
        obs[env_idx, ofs] = xpos[env_idx, FOOT_BODY_ID_1][2]
        ofs += 1
        for j in range(wp.static(NQ)):
            obs[env_idx, ofs] = qpos[env_idx, j]
            ofs += 1
        for j in range(wp.static(NV)):
            obs[env_idx, ofs] = wp.clamp(qvel[env_idx, j], -100.0, 100.0)
            ofs += 1

        if wp.static(INCLUDE_MUSCLE):
            for j in range(wp.static(MUSCLE_COUNT)):
                obs[env_idx, ofs] = muscle_activation[env_idx, j]
                ofs += 1
            for j in range(wp.static(MUSCLE_COUNT)):
                obs[env_idx, ofs] = wp.clamp(
                    muscle_force[env_idx, j] / 1000.0, -100.0, 100.0
                )
                ofs += 1
            for j in range(wp.static(MUSCLE_COUNT)):
                obs[env_idx, ofs] = muscle_length[env_idx, j]
                ofs += 1
            for j in range(wp.static(MUSCLE_COUNT)):
                obs[env_idx, ofs] = wp.clamp(muscle_velocity[env_idx, j], -100.0, 100.0)
                ofs += 1

        root_q = normalize_quat_wp(xquat[env_idx, ROOT_BODY_IDX])
        yaw_q = heading_yaw_quat(root_q)
        yaw_inv = quat_inv(yaw_q)
        root_pos = xpos[env_idx, ROOT_BODY_IDX]
        root_pos_xz = wp.vec3(root_pos[0], root_pos[1], 0.0)

        ofs = wp.static(BODY_STATE_OFS)
        for bi in range(wp.static(NBODY)):
            bp = xpos[env_idx, bi]
            bq = normalize_quat_wp(xquat[env_idx, bi])
            cv = cvel[env_idx, bi]
            bv = wp.vec3(cv[3], cv[4], cv[5])
            bw = wp.vec3(cv[0], cv[1], cv[2])

            pos_rel = rot_vec_quat(bp - root_pos_xz, yaw_inv)
            quat_rel = normalize_quat_wp(mul_quat(yaw_inv, bq))
            linvel_rel = rot_vec_quat(bv, yaw_inv)
            angvel_rel = rot_vec_quat(bw, yaw_inv)

            write_vec3(obs, env_idx, ofs + bi * 15 + 0, pos_rel, 1.0)
            write_ortho6d(obs, env_idx, ofs + bi * 15 + 3, quat_rel)
            write_vec3(obs, env_idx, ofs + bi * 15 + 9, linvel_rel, 0.1)
            write_vec3(obs, env_idx, ofs + bi * 15 + 12, angvel_rel, 0.1)

        if wp.static(INCLUDE_PHASE):
            denom = float(
                selected_clip_length
                if wp.static(MOTION_LOOP != 0)
                else max(selected_clip_length - 1, 1)
            )
            phase = TWO_PI * float(local_frame_idx) / denom
            obs[env_idx, wp.static(PHASE_OFS)] = wp.sin(phase)
            obs[env_idx, wp.static(PHASE_OFS) + 1] = wp.cos(phase)

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def obs_future_ref_kernel(
        xpos: wp.array2d(dtype=wp.vec3),
        xquat: wp.array2d(dtype=wp.quat),
        cvel: wp.array2d(dtype=wp.spatial_vector),
        ref_xpos: wp.array2d(dtype=wp.vec3),
        ref_xquat: wp.array2d(dtype=wp.quat),
        ref_cvel: wp.array2d(dtype=wp.spatial_vector),
        frame_idx: wp.array(dtype=wp.int32),
        motion_clip_idx: wp.array(dtype=wp.int32),
        clip_start: wp.array(dtype=wp.int32),
        clip_length: wp.array(dtype=wp.int32),
        clip_loop_root_quat: wp.array(dtype=wp.quat),
        clip_loop_root_pos: wp.array(dtype=wp.vec3),
        obs: wp.array2d(dtype=float),
    ):
        env_idx = wp.tid()
        clip_idx = int(motion_clip_idx[env_idx])
        selected_clip_length = int(clip_length[clip_idx])
        selected_clip_start = int(clip_start[clip_idx])

        fi = int(frame_idx[env_idx])
        next_fi = fi + 1
        local_idx = _clamp_or_wrap_frame_idx(next_fi, selected_clip_length, MOTION_LOOP)
        cycles = _frame_cycle_count(next_fi, selected_clip_length, MOTION_LOOP)
        global_idx = selected_clip_start + local_idx
        loop_root_q = normalize_quat_wp(clip_loop_root_quat[clip_idx])
        loop_root_t = clip_loop_root_pos[clip_idx]

        root_q = normalize_quat_wp(xquat[env_idx, ROOT_BODY_IDX])
        yaw_q = heading_yaw_quat(root_q)
        yaw_inv = quat_inv(yaw_q)
        heading = yaw_q
        root_pos = xpos[env_idx, ROOT_BODY_IDX]

        ofs = wp.static(FUTURE_REF_OFS)
        for bi in range(wp.static(NBODY_OBS)):
            body_idx = bi + 1
            ref_bp = ref_xpos[global_idx, body_idx]
            ref_bq = normalize_quat_wp(ref_xquat[global_idx, body_idx])
            ref_cv = ref_cvel[global_idx, body_idx]
            ref_lv = wp.vec3(ref_cv[3], ref_cv[4], ref_cv[5])
            ref_av = wp.vec3(ref_cv[0], ref_cv[1], ref_cv[2])

            if cycles > 0:
                ref_bp = apply_loop_transform_vec(
                    ref_bp, loop_root_q, loop_root_t, cycles
                )
                ref_bq = normalize_quat_wp(
                    apply_loop_transform_quat(ref_bq, loop_root_q, cycles)
                )
                ref_lv = apply_loop_rotation_vec(ref_lv, loop_root_q, cycles)
                ref_av = apply_loop_rotation_vec(ref_av, loop_root_q, cycles)

            sim_bp = xpos[env_idx, body_idx]
            sim_bq = normalize_quat_wp(xquat[env_idx, body_idx])
            sim_cv = cvel[env_idx, body_idx]
            sim_lv = wp.vec3(sim_cv[3], sim_cv[4], sim_cv[5])
            sim_av = wp.vec3(sim_cv[0], sim_cv[1], sim_cv[2])

            diff_pos = rot_vec_quat(ref_bp - sim_bp, yaw_inv)
            diff_rot = normalize_quat_wp(mul_quat(ref_bq, quat_inv(sim_bq)))
            diff_rot = normalize_quat_wp(mul_quat(mul_quat(yaw_inv, diff_rot), heading))
            diff_lv = rot_vec_quat(ref_lv - sim_lv, yaw_inv)
            diff_av = rot_vec_quat(ref_av - sim_av, yaw_inv)

            local_ref_pos = rot_vec_quat(ref_bp - root_pos, yaw_inv)
            local_ref_rot = normalize_quat_wp(mul_quat(yaw_inv, ref_bq))

            body_ofs = ofs + bi * 24
            write_vec3(obs, env_idx, body_ofs + 0, diff_pos, 1.0)
            write_tan_norm(obs, env_idx, body_ofs + 3, diff_rot)
            write_vec3(obs, env_idx, body_ofs + 9, diff_lv, 0.1)
            write_vec3(obs, env_idx, body_ofs + 12, diff_av, 0.1)
            write_vec3(obs, env_idx, body_ofs + 15, local_ref_pos, 1.0)
            write_tan_norm(obs, env_idx, body_ofs + 18, local_ref_rot)

    return obs_proprioception_kernel, obs_future_ref_kernel


def _compute_endpoint_loop_transform(
    self,
    qpos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    helper_data = mujoco.MjData(self.model)
    helper_data.qpos[:] = np.asarray(qpos[0], dtype=np.float64)
    helper_data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, helper_data)
    root_pos_0 = helper_data.xpos[self.root_body_id].copy()
    root_rot_0 = helper_data.xmat[self.root_body_id].reshape(3, 3).copy()

    helper_data.qpos[:] = np.asarray(qpos[-1], dtype=np.float64)
    helper_data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, helper_data)
    root_pos_1 = helper_data.xpos[self.root_body_id].copy()
    root_rot_1 = helper_data.xmat[self.root_body_id].reshape(3, 3).copy()

    loop_rot = root_rot_1 @ root_rot_0.T
    loop_translation = root_pos_1 - loop_rot @ root_pos_0
    return loop_rot, loop_translation


def _mat_to_quat_np(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
    return quat.astype(np.float32, copy=False)


def _apply_loop_transform_positions_np(
    xpos: np.ndarray,
    loop_rotmat: np.ndarray,
    loop_translation: np.ndarray,
    cycles: int,
) -> np.ndarray:
    out = np.asarray(xpos, dtype=np.float32).copy()
    for _ in range(max(0, int(cycles))):
        out = out @ loop_rotmat.T + loop_translation[None, :]
    return out


def _apply_loop_transform_quats_np(
    xquat: np.ndarray,
    loop_quat: np.ndarray,
    cycles: int,
) -> np.ndarray:
    # Renormalizes like the Warp `apply_loop_transform_quat` it mirrors; without
    # it the two drift apart over long loops and neither stays unit-length.
    out = _normalize_quats_np(np.asarray(xquat, dtype=np.float32).copy())
    for _ in range(max(0, int(cycles))):
        out = _normalize_quats_np(
            np.stack(
                [quat_mul_np(loop_quat, quat, dtype=np.float32) for quat in out],
                axis=0,
            )
        )
    return out


def _normalize_quats_np(quats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quats.astype(np.float64), axis=-1, keepdims=True)
    safe = norms > 1.0e-6
    out = np.where(safe, quats / np.where(safe, norms, 1.0), 0.0).astype(np.float32)
    out[..., 0] = np.where(safe[..., 0], out[..., 0], 1.0)
    return np.where(out[..., :1] < 0.0, -out, out)


def _resolve_motion_clip_names(self) -> list[str]:
    motion_clips_cfg = self.cfg.get("motion_clips", [])
    if motion_clips_cfg is not None and len(motion_clips_cfg) > 0:
        return [str(name) for name in motion_clips_cfg]
    return [str(self.cfg.motion_clip)]


def _load_motion_library(self, profile: FreemuscoImitationProfile) -> MocapLibrary:
    motion_clips = [
        MocapClip.load(
            motion_data_dir=self._motion_data_dir,
            clip_name=clip_name,
            device=self.device,
        )
        for clip_name in _resolve_motion_clip_names(self)
    ]
    if bool(self.cfg.get("mirror_motion_clips", False)):
        if profile.mirror_mode != "paired_joint_reflection":
            raise ValueError(
                "mirror_motion_clips requires the profile's paired_joint_reflection "
                "capability."
            )
        mirror_exclude = {
            str(name) for name in self.cfg.get("mirror_exclude_motion_clips", [])
        }
        motion_clips.extend(
            [
                mirror_mocap_clip(
                    clip,
                    model=self.model,
                    device=self.device,
                    left_prefix=profile.mirror_left_prefix,
                    right_prefix=profile.mirror_right_prefix,
                    clip_suffix=profile.mirror_clip_suffix,
                    reference_body_start=profile.reference_body_start,
                    reference_body_count_offset=profile.reference_body_count_offset,
                )
                for clip in motion_clips
                if str(clip.clip_name) not in mirror_exclude
            ]
        )

    library = MocapLibrary.from_clips(clips=motion_clips, device=self.device)
    if self._motion_loop and bool(self.cfg.get("smooth_cycle", False)):
        if profile.smoothing_mode == "extrapolated_root_cycle":
            library = smooth_mocap_library(
                library,
                model=self.model,
                root_body_name=profile.root_body_name,
                control_dt=float(self.step_dt),
                smooth_cycle_blend_frames=int(
                    self.cfg.get("smooth_cycle_blend_frames", 30)
                ),
                device=self.device,
                reference_body_start=profile.reference_body_start,
                reference_body_count_offset=profile.reference_body_count_offset,
                root_qpos_position_indices=profile.root_qpos_position_indices,
                root_qpos_rotation_indices=profile.root_qpos_rotation_indices,
                vertical_axis=profile.loop_vertical_axis,
            )
        elif profile.smoothing_mode != "none":
            raise ValueError(
                f"Unsupported imitation smoothing capability: {profile.smoothing_mode!r}."
            )
    return library


def _library_first_clip(library: MocapLibrary) -> MocapClip:
    clip_start = int(library.clip_start[0].item())
    clip_length = int(library.clip_length[0].item())
    return MocapClip(
        clip_name=str(library.clip_names[0]),
        qpos=library.qpos[clip_start : clip_start + clip_length].contiguous(),
        qvel=library.qvel[clip_start : clip_start + clip_length].contiguous(),
        xipos=library.xipos[clip_start : clip_start + clip_length].contiguous(),
        ximat=library.ximat[clip_start : clip_start + clip_length].contiguous(),
        xquat=library.xquat[clip_start : clip_start + clip_length].contiguous(),
    )


def _compute_clip_loop_reference_transforms(
    self,
    profile: FreemuscoImitationProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clip_count = int(self.motion_clip_length.numel())
    loop_rot = np.zeros((clip_count, 3, 3), dtype=np.float32)
    loop_translation = np.zeros((clip_count, 3), dtype=np.float32)
    loop_quat = np.zeros((clip_count, 4), dtype=np.float32)
    root_body_pos = np.asarray(self.model.body_pos[self.root_body_id], dtype=np.float64)

    for clip_idx in range(clip_count):
        clip_start = int(self.motion_clip_start[clip_idx].item())
        clip_length = int(self.motion_clip_length[clip_idx].item())
        clip_qpos = (
            self._motion_qpos_t[clip_start : clip_start + clip_length]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        if profile.loop_transform_mode == "extrapolated_root_yaw":
            clip_loop_rot, clip_loop_translation = compute_loop_root_transform_np(
                clip_qpos,
                root_body_pos=root_body_pos,
                vertical_axis=profile.loop_vertical_axis,
                root_qpos_position_indices=profile.root_qpos_position_indices,
                root_qpos_rotation_indices=profile.root_qpos_rotation_indices,
            )
        else:
            clip_loop_rot, clip_loop_translation = _compute_endpoint_loop_transform(
                self, clip_qpos
            )
        loop_rot[clip_idx] = np.asarray(clip_loop_rot, dtype=np.float32)
        loop_translation[clip_idx] = np.asarray(clip_loop_translation, dtype=np.float32)
        loop_quat[clip_idx] = _mat_to_quat_np(clip_loop_rot)

    return loop_rot, loop_translation, loop_quat


def initialize(self, cfg, render_mode: str | None = None, **kwargs) -> None:
    profile = FreemuscoImitationProfile.from_cfg(cfg)
    self._imitation_profile = profile
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

    self._init_fast_muscle_runtime()
    self._init_motion_runtime_flags()
    self._motion_data_dir = self._resolve_motion_data_dir(str(self.cfg.motion_data_dir))
    self._motion_library = _load_motion_library(self, profile)
    self._multi_motion_enabled = int(self._motion_library.clip_count) > 1
    self.motion_clip_start = self._motion_library.clip_start.to(
        device=self.device, dtype=torch.int32
    ).contiguous()
    self.motion_clip_length = self._motion_library.clip_length.to(
        device=self.device, dtype=torch.int32
    ).contiguous()
    self._motion_clip = _library_first_clip(self._motion_library)

    self._start_frame_idx = int(self.cfg.get("start_frame_idx", 0))
    if self._start_frame_idx < 0:
        raise ValueError("start_frame_idx must be >= 0.")
    fixed_motion_idx_cfg = self.cfg.get("fixed_motion_idx", None)
    self._fixed_motion_idx = (
        None if fixed_motion_idx_cfg is None else int(fixed_motion_idx_cfg)
    )
    if self._fixed_motion_idx is not None:
        if self._fixed_motion_idx < 0:
            raise ValueError("fixed_motion_idx must be >= 0.")
        if self._fixed_motion_idx >= int(self.motion_clip_length.numel()):
            raise ValueError(
                f"fixed_motion_idx={self._fixed_motion_idx} is out of range for "
                f"{int(self.motion_clip_length.numel())} mocap clips."
            )

    self._motion_qpos_t = self._motion_library.qpos
    self._motion_qvel_t = self._motion_library.qvel
    self._validate_motion_shapes(
        qpos=self._motion_qpos_t,
        qvel=self._motion_qvel_t,
        xipos=self._motion_library.xipos,
        ximat=self._motion_library.ximat,
        body_count=int(self.model.nbody - profile.reference_body_count_offset),
    )
    self._motion_reference_payload = self._compute_reference_render_body_states()
    if self._motion_reference_payload.cvel is None:
        raise RuntimeError("FreeMusco render reference payload is missing cvel.")
    self._motion_xpos_cpu = self._motion_reference_payload.xpos.numpy()
    self._motion_xquat_cpu = self._motion_reference_payload.xquat.numpy()
    (
        self._clip_loop_root_rotmat_np,
        self._clip_loop_root_translation_np,
        self._clip_loop_root_quat_np,
    ) = _compute_clip_loop_reference_transforms(self, profile)
    self._loop_root_rotmat_np = self._clip_loop_root_rotmat_np[0]
    self._loop_root_translation_np = self._clip_loop_root_translation_np[0]
    self._loop_root_quat_np = self._clip_loop_root_quat_np[0]

    self._init_mujoco_tensor_views()
    self._motion_frame_idx = torch.zeros(
        self.num_envs, device=self.device, dtype=torch.int32
    )
    self._motion_frame_idx_wp = wp.from_torch(self._motion_frame_idx, dtype=wp.int32)
    self.motion_clip_idx = torch.zeros(
        self.num_envs, device=self.device, dtype=torch.int32
    )
    if self._fixed_motion_idx is not None:
        self.motion_clip_idx.fill_(int(self._fixed_motion_idx))
    self.motion_clip_idx_wp = wp.from_torch(self.motion_clip_idx, dtype=wp.int32)
    self.motion_clip_start_wp = wp.from_torch(self.motion_clip_start, dtype=wp.int32)
    self.motion_clip_length_wp = wp.from_torch(self.motion_clip_length, dtype=wp.int32)
    self._motion_qpos_wp = wp.from_torch(self._motion_qpos_t, dtype=wp.float32)
    self._motion_qvel_wp = wp.from_torch(self._motion_qvel_t, dtype=wp.float32)
    self._motion_xpos_t = self._motion_reference_payload.xpos.to(
        device=self.device, dtype=torch.float32
    ).contiguous()
    self._motion_xquat_t = self._motion_reference_payload.xquat.to(
        device=self.device, dtype=torch.float32
    ).contiguous()
    self._motion_cvel_t = self._motion_reference_payload.cvel.to(
        device=self.device, dtype=torch.float32
    ).contiguous()
    self._motion_xpos_wp = wp.from_torch(self._motion_xpos_t, dtype=wp.vec3)
    self._motion_xquat_wp = wp.from_torch(self._motion_xquat_t, dtype=wp.quat)
    self._motion_cvel_wp = wp.from_torch(self._motion_cvel_t, dtype=wp.spatial_vector)
    self.motion_clip_loop_root_quat = torch.from_numpy(self._clip_loop_root_quat_np).to(
        device=self.device, dtype=torch.float32
    )
    self.motion_clip_loop_root_pos = torch.from_numpy(
        self._clip_loop_root_translation_np
    ).to(device=self.device, dtype=torch.float32)
    self.motion_clip_loop_root_quat_wp = wp.from_torch(
        self.motion_clip_loop_root_quat, dtype=wp.quat
    )
    self.motion_clip_loop_root_pos_wp = wp.from_torch(
        self.motion_clip_loop_root_pos, dtype=wp.vec3
    )
    self._motion_body_count = int(self._motion_library.xipos.shape[1])
    self._init_tracking_reward_runtime(
        ee_body_ids=tuple(
            resolve_id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in profile.end_effector_body_names
        ),
        talus_body_names=profile.talus_body_names,
        toe_body_names=profile.toe_body_names,
        loop_root_quat=self._loop_root_quat_np,
        loop_root_translation=self._loop_root_translation_np,
        motion_frame_count=int(self._motion_qpos_t.shape[0]),
    )
    del self._motion_reference_payload

    nbody = int(self.model.nbody)
    include_phase = profile.phase_observation_mode == "required" or bool(
        self.cfg.get("include_phase_in_policy_obs", True)
    )
    phase_dim = 2 if include_phase else 0
    self._init_observation_and_reward_buffers(
        nbody_obs=int(self.model.nbody - profile.reference_body_count_offset),
        phase_dim=phase_dim,
    )
    mc = int(self.fast_muscle.muscle_count)
    self._obs_proprio_kernel, self._obs_future_ref_kernel = _make_observation_kernels(
        nbody=nbody,
        nbody_obs=self._nbody_obs,
        root_body_idx=self.root_body_id,
        pelvis_geom_id=self.pelvis_geom_id,
        foot_body_id_0=self.foot_body_id_0,
        foot_body_id_1=self.foot_body_id_1,
        nq=self._nq,
        nv=self._nv,
        muscle_count=mc,
        include_muscle=self._include_muscle_observation_blocks,
        include_phase=include_phase,
        motion_loop=self._motion_loop,
        body_state_ofs=self._body_state_ofs,
        future_ref_ofs=self._future_ref_ofs,
        phase_ofs=self._phase_ofs,
    )

    self._include_phase_in_policy_obs = include_phase
    self._finalize_fast_muscle_runtime()
    apply_character_geom_render_visibility(self)
    self.fast_muscle.attach_render_assets(
        render_mode=self.render_mode,
        render_model=getattr(self, "render_model", self.model),
    )


def compute_clip_loop_reference_transforms(self):
    return _compute_clip_loop_reference_transforms(self, self._imitation_profile)


def compute_loop_reference_transform(self):
    loop_rot, loop_translation, loop_quat = _compute_clip_loop_reference_transforms(
        self, self._imitation_profile
    )
    return loop_rot[0], loop_translation[0], loop_quat[0]


def pre_physics_step(self, actions: torch.Tensor) -> None:
    self.fast_muscle.pre_physics_step(actions)
    wp.launch(
        advance_motion_frame_idx_kernel,
        dim=(self.num_envs,),
        inputs=[
            self._motion_frame_idx_wp,
            self.motion_clip_idx_wp,
            self.motion_clip_length_wp,
            int(self._motion_loop),
        ],
    )


def launch_obs_proprio_kernel(
    self,
    target: torch.Tensor,
    *,
    muscle_activation: torch.Tensor,
    muscle_force: torch.Tensor,
    muscle_length: torch.Tensor,
    muscle_velocity: torch.Tensor,
) -> None:
    wp.launch(
        self._obs_proprio_kernel,
        dim=(self.num_envs,),
        inputs=[
            self.mjw_data.xpos,
            self.mjw_data.xquat,
            self.mjw_data.geom_xpos,
            self.mjw_data.qpos,
            self.mjw_data.qvel,
            self.mjw_data.cvel,
            muscle_activation,
            muscle_force,
            muscle_length,
            muscle_velocity,
            self._motion_frame_idx_wp,
            self.motion_clip_idx_wp,
            self.motion_clip_length_wp,
        ],
        outputs=[target[:, : self._obs_dim]],
    )


def launch_obs_future_ref_kernel(self, target: torch.Tensor) -> None:
    wp.launch(
        self._obs_future_ref_kernel,
        dim=(self.num_envs,),
        inputs=[
            self.mjw_data.xpos,
            self.mjw_data.xquat,
            self.mjw_data.cvel,
            self._motion_xpos_wp,
            self._motion_xquat_wp,
            self._motion_cvel_wp,
            self._motion_frame_idx_wp,
            self.motion_clip_idx_wp,
            self.motion_clip_start_wp,
            self.motion_clip_length_wp,
            self.motion_clip_loop_root_quat_wp,
            self.motion_clip_loop_root_pos_wp,
        ],
        outputs=[target[:, : self._obs_dim]],
    )


def get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    if self._imitation_reward_threshold > 0.0:
        self._compute_termination_rewards()
    wp.launch(
        _imitation_termination_kernel,
        dim=(self.num_envs,),
        inputs=[
            self._motion_frame_idx_wp,
            self.motion_clip_idx_wp,
            self.motion_clip_length_wp,
            self._reward_buf,
            self.episode_length_buf,
            int(self._motion_loop),
            self._imitation_reward_threshold,
            self.max_episode_length,
        ],
        outputs=[self.reset_terminated, self.reset_time_outs],
    )
    return self.reset_terminated, self.reset_time_outs


def reset_idx(self, reset_mask: torch.Tensor):
    mjw.reset_data(self.mjw_model, self.mjw_data, wp.from_torch(reset_mask))
    wp.launch(
        _reset_motion_state_kernel,
        dim=(self.num_envs,),
        inputs=[
            reset_mask,
            self._sample_reset_seed(),
            self.motion_clip_idx_wp,
            int(self.motion_clip_length.numel()),
            int(-1 if self._fixed_motion_idx is None else self._fixed_motion_idx),
            self._motion_qpos_wp,
            self._motion_qvel_wp,
            self.motion_clip_start_wp,
            self.motion_clip_length_wp,
            int(self._random_init),
            int(self._motion_loop),
            int(self._start_frame_idx),
            self._nq,
            self._nv,
        ],
        outputs=[
            self._motion_frame_idx_wp,
            self.mjw_data.qpos,
            self.mjw_data.qvel,
        ],
    )
    self.fast_muscle.reset_idx(reset_mask)


def build_render_model(self):
    profile = self._imitation_profile
    self._resolved_source_mjcf_path = str(self._source_mjcf_path)
    result = build_ghost_render_model(
        source_mjcf_path=self._resolved_source_mjcf_path,
        sim_model=self.model,
        resolution=tuple(self.cfg.viewer.resolution),
        ghost_root_xpath=profile.ghost_root_xpath,
        ghost_root_description=profile.ghost_root_description,
        ghost_offset=self.cfg.get("ghost_offset", [0.0, 1.5, 0.0]),
        ghost_offset_dtype=np.float32,
        sim_body_resolver=lambda name: resolve_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, name
        ),
    )
    if result.ghost_binding is None:
        raise ValueError("Expected ghost render binding for imitation profile.")
    self._ghost_mocap_ids = result.ghost_binding.ghost_mocap_ids
    self._ghost_sim_body_ids = result.ghost_binding.ghost_sim_body_ids
    self._ghost_offset_np = result.ghost_binding.ghost_offset_np
    return result.render_model, result.render_data


def update_render_data(self):
    if not hasattr(self, "_ghost_mocap_ids"):
        return
    clip_idx = int(self.motion_clip_idx[0].item())
    clip_start = int(self.motion_clip_start[clip_idx].item())
    clip_length = int(self.motion_clip_length[clip_idx].item())
    global_frame_idx = int(self._motion_frame_idx[0].item())
    if self._motion_loop:
        local_frame_idx = global_frame_idx % clip_length
        cycles = global_frame_idx // clip_length
    else:
        local_frame_idx = min(global_frame_idx, clip_length - 1)
        cycles = 0
    frame_ref = clip_start + local_frame_idx

    xpos = self._motion_xpos_cpu[frame_ref]
    xquat = self._motion_xquat_cpu[frame_ref]
    if cycles > 0:
        xpos = _apply_loop_transform_positions_np(
            xpos,
            self._clip_loop_root_rotmat_np[clip_idx],
            self._clip_loop_root_translation_np[clip_idx],
            cycles,
        )
        xquat = _apply_loop_transform_quats_np(
            xquat, self._clip_loop_root_quat_np[clip_idx], cycles
        )

    self.render_data.mocap_pos[self._ghost_mocap_ids] = (
        xpos[self._ghost_sim_body_ids] + self._ghost_offset_np
    )
    self.render_data.mocap_quat[self._ghost_mocap_ids] = xquat[self._ghost_sim_body_ids]
    self.fast_muscle.update_render_tendon_rgba()


def get_deployable_obs_spec(self) -> dict[str, object]:
    return build_deployable_obs_spec(
        observation_layout=self._observation_layout,
        obs_main_dim=int(self._obs_main_dim),
        include_phase_in_policy_obs=bool(self._include_phase_in_policy_obs),
        phase_ofs=int(self._phase_ofs),
        phase_dim=int(self._observation_layout.phase_dim),
    )


species = "freemusco"

clamp_or_wrap_frame_idx = _clamp_or_wrap_frame_idx
frame_cycle_count = _frame_cycle_count
mat_to_quat_np = _mat_to_quat_np
apply_loop_transform_positions_np = _apply_loop_transform_positions_np
apply_loop_transform_quats_np = _apply_loop_transform_quats_np
resolve_motion_clip_names = _resolve_motion_clip_names


__all__ = [
    "FreemuscoImitationProfile",
    "advance_motion_frame_idx_kernel",
    "apply_loop_transform_positions_np",
    "apply_loop_transform_quats_np",
    "clamp_or_wrap_frame_idx",
    "compute_clip_loop_reference_transforms",
    "compute_loop_reference_transform",
    "frame_cycle_count",
    "get_dones",
    "get_deployable_obs_spec",
    "initialize",
    "launch_obs_future_ref_kernel",
    "launch_obs_proprio_kernel",
    "pre_physics_step",
    "reset_idx",
    "resolve_motion_clip_names",
    "species",
    "build_render_model",
    "update_render_data",
]
