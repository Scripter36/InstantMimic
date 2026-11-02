from __future__ import annotations

import hashlib
import math
from functools import cache
from pathlib import Path

import warp as wp

from instantmimic.utils.mjw_kernels.private import rot_vec_quat
from instantmimic.utils.warp_kernels import expose_kernel_name

from .muscle_qfrc_projector import (
    active_force_length_model_wp,
    force_magnitude_from_activation_model_wp,
    lbs_muscle_path_velocity_wp,
    muscle_force_velocity_curve_wp,
    native_load_vec2i_wp,
    native_load_vec4_wp,
    native_load_vec4i_wp,
    normalized_muscle_length_wp,
    passive_force_length_model_wp,
    segment_generalized_force_contribution_wp,
    segment_length_wp,
    unit_force_magnitude_from_length_norm_wp,
)
from .muscle_qfrc_projector import (
    normalize_quat_wp as _normalize_quat_wp,
)
from .waypoint_wrap import WaypointWrapKernelState

_NATIVE_SNIPPET_DIR = Path(__file__).resolve().parent

# Isolate tiled block-dimension specializations from scalar kernel compilation.
_TILED_KERNEL_MODULE = f"{__name__}.tiled"


def _native_snippet_with_line_mapping(file_name: str) -> str:
    path = _NATIVE_SNIPPET_DIR / file_name
    return f'#line 1 "{path.as_posix()}"\n{path.read_text(encoding="utf-8")}'


@wp.func
def _unit_force_from_precomputed_coeff_wp(
    muscle_length: float,
    coeff: wp.vec4,
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    active_force_model_params: wp.vec3,
) -> wp.vec2:
    l_m_norm = wp.clamp(
        muscle_length * coeff[0] + coeff[1],
        muscle_length_norm_min,
        muscle_length_norm_max,
    )
    unit_force = coeff[2] * active_force_length_model_wp(
        l_m_norm, active_force_model_params
    )
    return wp.vec2(l_m_norm, unit_force)


@wp.kernel(enable_backward=False)
def _normalize_body_quats_kernel(
    body_quat: wp.array2d(dtype=wp.quat),
    normalized_body_quat: wp.array2d(dtype=wp.quat),
):
    env_idx, body_idx = wp.tid()
    normalized_body_quat[env_idx, body_idx] = _normalize_quat_wp(
        body_quat[env_idx, body_idx]
    )


@wp.kernel(enable_backward=False)
def _transform_waypoints_from_torch_kernel(
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    muscle_waypoint_body: wp.array(dtype=wp.int32),
    muscle_waypoints_local: wp.array(dtype=wp.vec3),
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
):
    env_idx, waypoint_idx = wp.tid()
    body_idx = muscle_waypoint_body[waypoint_idx]
    p = body_pos[env_idx, body_idx]
    q = _normalize_quat_wp(body_quat[env_idx, body_idx])
    local = muscle_waypoints_local[waypoint_idx]
    transformed_waypoints[env_idx, waypoint_idx] = p + rot_vec_quat(local, q)


@wp.kernel(enable_backward=False)
def _transform_lbs_waypoints_from_torch_kernel(
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    waypoint_attachment_start: wp.array(dtype=wp.int32),
    waypoint_attachment_body: wp.array(dtype=wp.int32),
    waypoint_attachment_local: wp.array(dtype=wp.vec3),
    waypoint_attachment_weight: wp.array(dtype=wp.float32),
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
):
    env_idx, waypoint_idx = wp.tid()
    begin = waypoint_attachment_start[waypoint_idx]
    end = waypoint_attachment_start[waypoint_idx + 1]
    total = wp.vec3(0.0, 0.0, 0.0)
    for attach_idx in range(begin, end):
        body_idx = waypoint_attachment_body[attach_idx]
        p = body_pos[env_idx, body_idx]
        q = _normalize_quat_wp(body_quat[env_idx, body_idx])
        local = waypoint_attachment_local[attach_idx]
        total += waypoint_attachment_weight[attach_idx] * (p + rot_vec_quat(local, q))
    transformed_waypoints[env_idx, waypoint_idx] = total


@wp.kernel(enable_backward=False)
def _transform_lbs_waypoints_fixed2_normalized_quat_from_torch_kernel(
    body_pos: wp.array2d(dtype=wp.vec3),
    normalized_body_quat: wp.array2d(dtype=wp.quat),
    waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
    waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
    waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
):
    env_idx, waypoint_idx = wp.tid()
    body2 = native_load_vec2i_wp(waypoint_attachment_body2, waypoint_idx)
    local_weight0 = native_load_vec4_wp(
        waypoint_attachment_local_weight0,
        waypoint_idx,
    )
    local_weight1 = native_load_vec4_wp(
        waypoint_attachment_local_weight1,
        waypoint_idx,
    )

    body0 = body2[0]
    q0 = normalized_body_quat[env_idx, body0]
    local0 = wp.vec3(local_weight0[0], local_weight0[1], local_weight0[2])
    point0 = body_pos[env_idx, body0] + rot_vec_quat(local0, q0)

    body1 = body2[1]
    q1 = normalized_body_quat[env_idx, body1]
    local1 = wp.vec3(local_weight1[0], local_weight1[1], local_weight1[2])
    point1 = body_pos[env_idx, body1] + rot_vec_quat(local1, q1)

    transformed_waypoints[env_idx, waypoint_idx] = (
        local_weight0[3] * point0 + local_weight1[3] * point1
    )


@wp.func
def _lbs_fixed2_waypoint_position_wp(
    env_idx: int,
    waypoint_idx: int,
    body_pos: wp.array2d(dtype=wp.vec3),
    normalized_body_quat: wp.array2d(dtype=wp.quat),
    waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
    waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
    waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
) -> wp.vec3:
    body2 = native_load_vec2i_wp(waypoint_attachment_body2, waypoint_idx)
    local_weight0 = native_load_vec4_wp(
        waypoint_attachment_local_weight0,
        waypoint_idx,
    )
    local_weight1 = native_load_vec4_wp(
        waypoint_attachment_local_weight1,
        waypoint_idx,
    )

    body0 = body2[0]
    local0 = wp.vec3(local_weight0[0], local_weight0[1], local_weight0[2])
    point0 = body_pos[env_idx, body0] + rot_vec_quat(
        local0,
        normalized_body_quat[env_idx, body0],
    )

    body1 = body2[1]
    local1 = wp.vec3(local_weight1[0], local_weight1[1], local_weight1[2])
    point1 = body_pos[env_idx, body1] + rot_vec_quat(
        local1,
        normalized_body_quat[env_idx, body1],
    )
    return local_weight0[3] * point0 + local_weight1[3] * point1


@wp.func
def _lbs_fixed1_waypoint_position_wp(
    env_idx: int,
    waypoint_idx: int,
    body_pos: wp.array2d(dtype=wp.vec3),
    normalized_body_quat: wp.array2d(dtype=wp.quat),
    waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
    waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
) -> wp.vec3:
    body_idx = native_load_vec2i_wp(waypoint_attachment_body2, waypoint_idx)[0]
    local_weight = native_load_vec4_wp(
        waypoint_attachment_local_weight0,
        waypoint_idx,
    )
    local = wp.vec3(local_weight[0], local_weight[1], local_weight[2])
    point = body_pos[env_idx, body_idx] + rot_vec_quat(
        local,
        normalized_body_quat[env_idx, body_idx],
    )
    return local_weight[3] * point


@wp.func
def _lbs_fixed2_attachment_point_normalized_wp(
    env_idx: int,
    body_idx: int,
    local_weight: wp.vec4,
    body_pos: wp.array2d(dtype=wp.vec3),
    normalized_body_quat: wp.array2d(dtype=wp.quat),
) -> wp.vec3:
    local = wp.vec3(local_weight[0], local_weight[1], local_weight[2])
    return body_pos[env_idx, body_idx] + rot_vec_quat(
        local,
        normalized_body_quat[env_idx, body_idx],
    )


@wp.kernel(enable_backward=False)
def _compute_lbs_fixed2_unit_force_from_geometry_kernel(
    body_pos: wp.array2d(dtype=wp.vec3),
    normalized_body_quat: wp.array2d(dtype=wp.quat),
    muscle_segment_offsets: wp.array(dtype=wp.int32),
    muscle_force_length_coeff: wp.array(dtype=wp.vec4),
    waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
    waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
    waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    active_force_model_params: wp.vec3,
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    segment_idx = muscle_segment_offsets[muscle_idx]
    segment_end = muscle_segment_offsets[muscle_idx + 1]
    muscle_length = float(0.0)
    if segment_idx < segment_end:
        waypoint_idx = segment_idx + muscle_idx
        previous_point = _lbs_fixed2_waypoint_position_wp(
            env_idx,
            waypoint_idx,
            body_pos,
            normalized_body_quat,
            waypoint_attachment_body2,
            waypoint_attachment_local_weight0,
            waypoint_attachment_local_weight1,
        )
        while segment_idx < segment_end:
            waypoint_idx = segment_idx + muscle_idx + 1
            next_point = _lbs_fixed2_waypoint_position_wp(
                env_idx,
                waypoint_idx,
                body_pos,
                normalized_body_quat,
                waypoint_attachment_body2,
                waypoint_attachment_local_weight0,
                waypoint_attachment_local_weight1,
            )
            muscle_length += wp.length(next_point - previous_point)
            previous_point = next_point
            segment_idx += 1

    coeff = native_load_vec4_wp(muscle_force_length_coeff, muscle_idx)
    unit_force_magnitude[env_idx, muscle_idx] = _unit_force_from_precomputed_coeff_wp(
        muscle_length,
        coeff,
        muscle_length_norm_min,
        muscle_length_norm_max,
        active_force_model_params,
    )[1]


@wp.kernel(enable_backward=False)
def _compute_lbs_fixed1_unit_force_from_geometry_kernel(
    body_pos: wp.array2d(dtype=wp.vec3),
    normalized_body_quat: wp.array2d(dtype=wp.quat),
    muscle_segment_offsets: wp.array(dtype=wp.int32),
    muscle_force_length_coeff: wp.array(dtype=wp.vec4),
    waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
    waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    active_force_model_params: wp.vec3,
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    segment_idx = muscle_segment_offsets[muscle_idx]
    segment_end = muscle_segment_offsets[muscle_idx + 1]
    muscle_length = float(0.0)
    if segment_idx < segment_end:
        waypoint_idx = segment_idx + muscle_idx
        previous_point = _lbs_fixed1_waypoint_position_wp(
            env_idx,
            waypoint_idx,
            body_pos,
            normalized_body_quat,
            waypoint_attachment_body2,
            waypoint_attachment_local_weight0,
        )
        while segment_idx < segment_end:
            waypoint_idx = segment_idx + muscle_idx + 1
            next_point = _lbs_fixed1_waypoint_position_wp(
                env_idx,
                waypoint_idx,
                body_pos,
                normalized_body_quat,
                waypoint_attachment_body2,
                waypoint_attachment_local_weight0,
            )
            muscle_length += wp.length(next_point - previous_point)
            previous_point = next_point
            segment_idx += 1

    coeff = native_load_vec4_wp(muscle_force_length_coeff, muscle_idx)
    unit_force_magnitude[env_idx, muscle_idx] = _unit_force_from_precomputed_coeff_wp(
        muscle_length,
        coeff,
        muscle_length_norm_min,
        muscle_length_norm_max,
        active_force_model_params,
    )[1]


@wp.kernel(enable_backward=False)
def _precompute_lbs_fixed2_force_magnitude_cache_kernel(
    body_pos: wp.array2d(dtype=wp.vec3),
    normalized_body_quat: wp.array2d(dtype=wp.quat),
    muscle_segment_offsets: wp.array(dtype=wp.int32),
    muscle_force_length_coeff: wp.array(dtype=wp.vec4),
    waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
    waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
    waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    include_passive: int,
    active_force_model_params: wp.vec3,
    passive_force_model_params: wp.vec3,
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    passive_force_magnitude: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    segment_idx = muscle_segment_offsets[muscle_idx]
    segment_end = muscle_segment_offsets[muscle_idx + 1]
    muscle_length = float(0.0)
    if segment_idx < segment_end:
        waypoint_idx = segment_idx + muscle_idx
        previous_point = _lbs_fixed2_waypoint_position_wp(
            env_idx,
            waypoint_idx,
            body_pos,
            normalized_body_quat,
            waypoint_attachment_body2,
            waypoint_attachment_local_weight0,
            waypoint_attachment_local_weight1,
        )
        while segment_idx < segment_end:
            waypoint_idx = segment_idx + muscle_idx + 1
            next_point = _lbs_fixed2_waypoint_position_wp(
                env_idx,
                waypoint_idx,
                body_pos,
                normalized_body_quat,
                waypoint_attachment_body2,
                waypoint_attachment_local_weight0,
                waypoint_attachment_local_weight1,
            )
            muscle_length += wp.length(next_point - previous_point)
            previous_point = next_point
            segment_idx += 1

    coeff = native_load_vec4_wp(muscle_force_length_coeff, muscle_idx)
    norm_unit = _unit_force_from_precomputed_coeff_wp(
        muscle_length,
        coeff,
        muscle_length_norm_min,
        muscle_length_norm_max,
        active_force_model_params,
    )
    unit_force_magnitude[env_idx, muscle_idx] = norm_unit[1]
    passive_force = float(0.0)
    if include_passive != 0:
        passive_force = coeff[2] * passive_force_length_model_wp(
            norm_unit[0],
            passive_force_model_params,
        )
    passive_force_magnitude[env_idx, muscle_idx] = passive_force


@wp.kernel(enable_backward=False)
def _precompute_lbs_fixed1_force_magnitude_cache_kernel(
    body_pos: wp.array2d(dtype=wp.vec3),
    normalized_body_quat: wp.array2d(dtype=wp.quat),
    muscle_segment_offsets: wp.array(dtype=wp.int32),
    muscle_force_length_coeff: wp.array(dtype=wp.vec4),
    waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
    waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    include_passive: int,
    active_force_model_params: wp.vec3,
    passive_force_model_params: wp.vec3,
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    passive_force_magnitude: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    segment_idx = muscle_segment_offsets[muscle_idx]
    segment_end = muscle_segment_offsets[muscle_idx + 1]
    muscle_length = float(0.0)
    if segment_idx < segment_end:
        waypoint_idx = segment_idx + muscle_idx
        previous_point = _lbs_fixed1_waypoint_position_wp(
            env_idx,
            waypoint_idx,
            body_pos,
            normalized_body_quat,
            waypoint_attachment_body2,
            waypoint_attachment_local_weight0,
        )
        while segment_idx < segment_end:
            waypoint_idx = segment_idx + muscle_idx + 1
            next_point = _lbs_fixed1_waypoint_position_wp(
                env_idx,
                waypoint_idx,
                body_pos,
                normalized_body_quat,
                waypoint_attachment_body2,
                waypoint_attachment_local_weight0,
            )
            muscle_length += wp.length(next_point - previous_point)
            previous_point = next_point
            segment_idx += 1

    coeff = native_load_vec4_wp(muscle_force_length_coeff, muscle_idx)
    norm_unit = _unit_force_from_precomputed_coeff_wp(
        muscle_length,
        coeff,
        muscle_length_norm_min,
        muscle_length_norm_max,
        active_force_model_params,
    )
    unit_force_magnitude[env_idx, muscle_idx] = norm_unit[1]
    passive_force = float(0.0)
    if include_passive != 0:
        passive_force = coeff[2] * passive_force_length_model_wp(
            norm_unit[0],
            passive_force_model_params,
        )
    passive_force_magnitude[env_idx, muscle_idx] = passive_force


@wp.func
def waypoint_muscle_path_length_wp(
    env_idx: int,
    muscle_idx: int,
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    all_segment_start_idx: wp.array(dtype=wp.int32),
    all_muscle_segment_offsets: wp.array(dtype=wp.int32),
) -> float:
    """Polyline length over every segment of one muscle."""
    segment_begin = all_muscle_segment_offsets[muscle_idx]
    segment_end = all_muscle_segment_offsets[muscle_idx + 1]
    muscle_length = float(0.0)
    for segment_idx in range(segment_begin, segment_end):
        waypoint_idx = all_segment_start_idx[segment_idx]
        transformed_waypoints[env_idx, waypoint_idx]
        transformed_waypoints[env_idx, waypoint_idx + 1]
        muscle_length = muscle_length + segment_length_wp(
            env_idx,
            waypoint_idx,
            transformed_waypoints,
        )
    return muscle_length


@wp.func
def muscle_path_velocity_wp(
    env_idx: int,
    muscle_idx: int,
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    qvel_source: wp.array2d(dtype=wp.float32),
    muscle_active_entry_offsets: wp.array(dtype=wp.int32),
    muscle_active_segment_idx: wp.array(dtype=wp.int32),
    muscle_active_torque_idx: wp.array(dtype=wp.int32),
    muscle_active_start_coeff: wp.array(dtype=wp.float32),
    muscle_active_end_coeff: wp.array(dtype=wp.float32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    segment_start_root_body_idx: wp.array(dtype=wp.int32),
    segment_end_root_body_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    qvel_offset: int,
) -> float:
    """Single-body waypoint path velocity ``-sum_d moment_d qvel_d``.

    ``qvel_source[env, qvel_offset + dof]`` is the full generalized velocity.
    """
    velocity = float(0.0)
    entry_begin = muscle_active_entry_offsets[muscle_idx]
    entry_end = muscle_active_entry_offsets[muscle_idx + 1]
    for entry_idx in range(entry_begin, entry_end):
        torque_idx = muscle_active_torque_idx[entry_idx]
        full_dof = torque_idx + root_dof_count
        angular_cdof = wp.vec3(
            cdof[env_idx, full_dof, 0],
            cdof[env_idx, full_dof, 1],
            cdof[env_idx, full_dof, 2],
        )
        linear_cdof = wp.vec3(
            cdof[env_idx, full_dof, 3],
            cdof[env_idx, full_dof, 4],
            cdof[env_idx, full_dof, 5],
        )
        segment_idx = muscle_active_segment_idx[entry_idx]
        start_waypoint_idx = apply_segment_start_idx[segment_idx]
        end_waypoint_idx = apply_segment_end_idx[segment_idx]
        moment = segment_generalized_force_contribution_wp(
            linear_cdof,
            angular_cdof,
            transformed_waypoints[env_idx, start_waypoint_idx],
            transformed_waypoints[env_idx, end_waypoint_idx],
            subtree_com[env_idx, segment_start_root_body_idx[segment_idx]],
            subtree_com[env_idx, segment_end_root_body_idx[segment_idx]],
            muscle_active_start_coeff[entry_idx],
            muscle_active_end_coeff[entry_idx],
            1.0,
        )
        velocity = velocity - moment * qvel_source[env_idx, qvel_offset + full_dof]
    return velocity


@wp.kernel(enable_backward=False)
def _compute_waypoint_muscle_path_kinematics_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    all_segment_start_idx: wp.array(dtype=wp.int32),
    all_muscle_segment_offsets: wp.array(dtype=wp.int32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    raw_context: wp.array2d(dtype=wp.float32),
    muscle_active_entry_offsets: wp.array(dtype=wp.int32),
    muscle_active_segment_idx: wp.array(dtype=wp.int32),
    muscle_active_torque_idx: wp.array(dtype=wp.int32),
    muscle_active_start_coeff: wp.array(dtype=wp.float32),
    muscle_active_end_coeff: wp.array(dtype=wp.float32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    segment_start_root_body_idx: wp.array(dtype=wp.int32),
    segment_end_root_body_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    qvel_context_offset: int,
    path_length: wp.array2d(dtype=wp.float32),
    path_velocity: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    path_length[env_idx, muscle_idx] = waypoint_muscle_path_length_wp(
        env_idx,
        muscle_idx,
        transformed_waypoints,
        all_segment_start_idx,
        all_muscle_segment_offsets,
    )
    path_velocity[env_idx, muscle_idx] = muscle_path_velocity_wp(
        env_idx,
        muscle_idx,
        transformed_waypoints,
        subtree_com,
        cdof,
        raw_context,
        muscle_active_entry_offsets,
        muscle_active_segment_idx,
        muscle_active_torque_idx,
        muscle_active_start_coeff,
        muscle_active_end_coeff,
        apply_segment_start_idx,
        apply_segment_end_idx,
        segment_start_root_body_idx,
        segment_end_root_body_idx,
        root_dof_count,
        qvel_context_offset,
    )


@wp.kernel(enable_backward=False)
def _compute_lbs_muscle_path_kinematics_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    all_segment_start_idx: wp.array(dtype=wp.int32),
    all_muscle_segment_offsets: wp.array(dtype=wp.int32),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    raw_context: wp.array2d(dtype=wp.float32),
    muscle_active_entry_offsets: wp.array(dtype=wp.int32),
    muscle_active_segment_idx: wp.array(dtype=wp.int32),
    muscle_active_torque_idx: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    body_rootid: wp.array(dtype=wp.int32),
    active_attach_offsets: wp.array(dtype=wp.int32),
    active_attach_body: wp.array(dtype=wp.int32),
    active_attach_local: wp.array(dtype=wp.vec3),
    active_attach_coeff_weight: wp.array(dtype=wp.float32),
    waypoint_count: int,
    root_dof_count: int,
    qvel_context_offset: int,
    path_length: wp.array2d(dtype=wp.float32),
    path_velocity: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    path_length[env_idx, muscle_idx] = waypoint_muscle_path_length_wp(
        env_idx,
        muscle_idx,
        transformed_waypoints,
        all_segment_start_idx,
        all_muscle_segment_offsets,
    )
    path_velocity[env_idx, muscle_idx] = lbs_muscle_path_velocity_wp(
        env_idx,
        muscle_idx,
        transformed_waypoints,
        body_pos,
        body_quat,
        subtree_com,
        cdof,
        raw_context,
        muscle_active_entry_offsets,
        muscle_active_segment_idx,
        muscle_active_torque_idx,
        apply_segment_start_idx,
        apply_segment_end_idx,
        body_rootid,
        active_attach_offsets,
        active_attach_body,
        active_attach_local,
        active_attach_coeff_weight,
        waypoint_count,
        root_dof_count,
        qvel_context_offset,
    )


@wp.kernel(enable_backward=False)
def _compute_force_magnitudes_from_waypoints_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    all_segment_start_idx: wp.array(dtype=wp.int32),
    all_muscle_segment_offsets: wp.array(dtype=wp.int32),
    muscle_params: wp.array2d(dtype=wp.float32),
    muscle_rest_length: wp.array(dtype=wp.float32),
    muscle_force_scale: float,
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    include_passive_force_magnitude: float,
    active_force_model_params: wp.vec3,
    passive_force_model_params: wp.vec3,
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    passive_force_magnitude: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()

    muscle_length = waypoint_muscle_path_length_wp(
        env_idx,
        muscle_idx,
        transformed_waypoints,
        all_segment_start_idx,
        all_muscle_segment_offsets,
    )

    f0 = muscle_params[muscle_idx, 0]
    l_m0 = muscle_params[muscle_idx, 1]
    l_t0 = muscle_params[muscle_idx, 2]
    pen_angle = muscle_params[muscle_idx, 3]
    l_m_norm = normalized_muscle_length_wp(
        muscle_length,
        l_m0,
        l_t0,
        muscle_rest_length[muscle_idx],
        muscle_length_norm_min,
        muscle_length_norm_max,
    )
    unit_force_magnitude[env_idx, muscle_idx] = (
        unit_force_magnitude_from_length_norm_wp(
            f0,
            pen_angle,
            muscle_force_scale,
            l_m_norm,
            active_force_model_params,
        )
    )
    passive_force_value = float(0.0)
    if include_passive_force_magnitude != 0.0:
        passive_force_value = force_magnitude_from_activation_model_wp(
            0.0,
            0.0,
            0.0,
            1.0,
            f0,
            pen_angle,
            muscle_force_scale,
            l_m_norm,
            active_force_model_params,
            passive_force_model_params,
        )
    passive_force_magnitude[env_idx, muscle_idx] = passive_force_value


@wp.kernel(enable_backward=False)
def _compute_force_magnitudes_contiguous_waypoints_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    all_muscle_segment_offsets: wp.array(dtype=wp.int32),
    muscle_params: wp.array2d(dtype=wp.float32),
    muscle_rest_length: wp.array(dtype=wp.float32),
    muscle_force_scale: float,
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    include_passive_force_magnitude: float,
    active_force_model_params: wp.vec3,
    passive_force_model_params: wp.vec3,
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    passive_force_magnitude: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()

    segment_idx = all_muscle_segment_offsets[muscle_idx]
    segment_end = all_muscle_segment_offsets[muscle_idx + 1]

    muscle_length = float(0.0)
    while segment_idx < segment_end:
        waypoint_idx = segment_idx + muscle_idx
        transformed_waypoints[env_idx, waypoint_idx]
        transformed_waypoints[env_idx, waypoint_idx + 1]
        muscle_length += segment_length_wp(
            env_idx,
            waypoint_idx,
            transformed_waypoints,
        )
        segment_idx += 1

    f0 = muscle_params[muscle_idx, 0]
    l_m0 = muscle_params[muscle_idx, 1]
    l_t0 = muscle_params[muscle_idx, 2]
    pen_angle = muscle_params[muscle_idx, 3]
    l_m_norm = normalized_muscle_length_wp(
        muscle_length,
        l_m0,
        l_t0,
        muscle_rest_length[muscle_idx],
        muscle_length_norm_min,
        muscle_length_norm_max,
    )
    unit_force_magnitude[env_idx, muscle_idx] = (
        unit_force_magnitude_from_length_norm_wp(
            f0,
            pen_angle,
            muscle_force_scale,
            l_m_norm,
            active_force_model_params,
        )
    )
    passive_force_value = float(0.0)
    if include_passive_force_magnitude != 0.0:
        passive_force_value = force_magnitude_from_activation_model_wp(
            0.0,
            0.0,
            0.0,
            1.0,
            f0,
            pen_angle,
            muscle_force_scale,
            l_m_norm,
            active_force_model_params,
            passive_force_model_params,
        )
    passive_force_magnitude[env_idx, muscle_idx] = passive_force_value


@wp.kernel(enable_backward=False)
def _apply_muscle_force_velocity_to_unit_force_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    raw_context: wp.array2d(dtype=wp.float32),
    muscle_active_entry_offsets: wp.array(dtype=wp.int32),
    muscle_active_segment_idx: wp.array(dtype=wp.int32),
    muscle_active_torque_idx: wp.array(dtype=wp.int32),
    muscle_active_start_coeff: wp.array(dtype=wp.float32),
    muscle_active_end_coeff: wp.array(dtype=wp.float32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    segment_start_root_body_idx: wp.array(dtype=wp.int32),
    segment_end_root_body_idx: wp.array(dtype=wp.int32),
    muscle_params: wp.array2d(dtype=wp.float32),
    muscle_rest_length: wp.array(dtype=wp.float32),
    force_velocity_length_scale: wp.array(dtype=wp.float32),
    root_dof_count: int,
    qvel_context_offset: int,
    vmax: float,
    fvmax: float,
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    velocity = muscle_path_velocity_wp(
        env_idx,
        muscle_idx,
        transformed_waypoints,
        subtree_com,
        cdof,
        raw_context,
        muscle_active_entry_offsets,
        muscle_active_segment_idx,
        muscle_active_torque_idx,
        muscle_active_start_coeff,
        muscle_active_end_coeff,
        apply_segment_start_idx,
        apply_segment_end_idx,
        segment_start_root_body_idx,
        segment_end_root_body_idx,
        root_dof_count,
        qvel_context_offset,
    )

    velocity_scale = wp.max(
        force_velocity_length_scale[muscle_idx] * vmax,
        1.0e-6,
    )
    multiplier = muscle_force_velocity_curve_wp(
        velocity / velocity_scale,
        fvmax,
    )[0]
    unit_force_magnitude[env_idx, muscle_idx] *= multiplier


@wp.kernel(enable_backward=False)
def _accumulate_lbs_muscle_lengths_from_segments_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    all_segment_start_idx: wp.array(dtype=wp.int32),
    all_segment_muscle_idx: wp.array(dtype=wp.int32),
    muscle_length: wp.array2d(dtype=wp.float32),
):
    env_idx, segment_idx = wp.tid()
    waypoint_idx = all_segment_start_idx[segment_idx]
    transformed_waypoints[env_idx, waypoint_idx]
    transformed_waypoints[env_idx, waypoint_idx + 1]
    segment_length = segment_length_wp(
        env_idx,
        waypoint_idx,
        transformed_waypoints,
    )
    muscle_idx = all_segment_muscle_idx[segment_idx]
    wp.atomic_add(muscle_length, env_idx, muscle_idx, segment_length)


@wp.kernel(enable_backward=False)
def _accumulate_lbs_muscle_lengths_contiguous_waypoints_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    all_segment_muscle_idx: wp.array(dtype=wp.int32),
    muscle_length: wp.array2d(dtype=wp.float32),
):
    env_idx, segment_idx = wp.tid()
    muscle_idx = all_segment_muscle_idx[segment_idx]
    waypoint_idx = segment_idx + muscle_idx
    transformed_waypoints[env_idx, waypoint_idx]
    transformed_waypoints[env_idx, waypoint_idx + 1]
    segment_length = segment_length_wp(
        env_idx,
        waypoint_idx,
        transformed_waypoints,
    )
    wp.atomic_add(muscle_length, env_idx, muscle_idx, segment_length)


@wp.kernel(enable_backward=False)
def _accumulate_lbs_muscle_lengths_muscle_contiguous_waypoints_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    muscle_segment_offsets: wp.array(dtype=wp.int32),
    muscle_length: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    segment_idx = muscle_segment_offsets[muscle_idx]
    segment_end = muscle_segment_offsets[muscle_idx + 1]
    total = float(0.0)
    while segment_idx < segment_end:
        waypoint_idx = segment_idx + muscle_idx
        transformed_waypoints[env_idx, waypoint_idx]
        transformed_waypoints[env_idx, waypoint_idx + 1]
        total += segment_length_wp(
            env_idx,
            waypoint_idx,
            transformed_waypoints,
        )
        segment_idx += 1
    muscle_length[env_idx, muscle_idx] = total


@cache
def make_lbs_force_by_muscle_from_length_kernel(
    include_passive_force: bool,
    passive_force_model_id: int = 0,
    passive_force_param0: float = 5.5,
    passive_force_param1: float = 0.3,
):
    include_passive_force = bool(include_passive_force)
    include_passive_flag = 1 if include_passive_force else 0
    passive_model_id = float(passive_force_model_id)
    passive_param0 = float(passive_force_param0)
    passive_param1 = float(passive_force_param1)

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _compute_lbs_force_by_muscle_from_length_kernel(
        muscle_length: wp.array2d(dtype=wp.float32),
        muscle_params: wp.array2d(dtype=wp.float32),
        muscle_rest_length: wp.array(dtype=wp.float32),
        activation: wp.array2d(dtype=wp.float32),
        muscle_force_scale: float,
        muscle_length_norm_min: float,
        muscle_length_norm_max: float,
        active_force_model_params: wp.vec3,
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        passive_force_magnitude: wp.array2d(dtype=wp.float32),
        force_by_muscle: wp.array2d(dtype=wp.float32),
    ):
        env_idx, muscle_idx = wp.tid()
        f0 = muscle_params[muscle_idx, 0]
        l_m0 = muscle_params[muscle_idx, 1]
        l_t0 = muscle_params[muscle_idx, 2]
        pen_angle = muscle_params[muscle_idx, 3]
        l_m_norm = normalized_muscle_length_wp(
            muscle_length[env_idx, muscle_idx],
            l_m0,
            l_t0,
            muscle_rest_length[muscle_idx],
            muscle_length_norm_min,
            muscle_length_norm_max,
        )
        unit_force = unit_force_magnitude_from_length_norm_wp(
            f0,
            pen_angle,
            muscle_force_scale,
            l_m_norm,
            active_force_model_params,
        )
        passive_force = float(0.0)
        if wp.static(include_passive_flag) != 0:
            passive_force = (
                f0
                * passive_force_length_model_wp(
                    l_m_norm,
                    wp.vec3(passive_model_id, passive_param0, passive_param1),
                )
                * wp.cos(pen_angle)
                * muscle_force_scale
            )
        unit_force_magnitude[env_idx, muscle_idx] = unit_force
        passive_force_magnitude[env_idx, muscle_idx] = passive_force
        force_by_muscle[env_idx, muscle_idx] = (
            unit_force * activation[env_idx, muscle_idx] + passive_force
        )

    return _compute_lbs_force_by_muscle_from_length_kernel


@cache
def make_lbs_force_by_muscle_from_length_precomputed_kernel(
    include_passive_force: bool,
    passive_force_model_id: int = 0,
    passive_force_param0: float = 5.5,
    passive_force_param1: float = 0.3,
):
    include_passive_force = bool(include_passive_force)
    include_passive_flag = 1 if include_passive_force else 0
    passive_model_id = float(passive_force_model_id)
    passive_param0 = float(passive_force_param0)
    passive_param1 = float(passive_force_param1)

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _compute_lbs_force_by_muscle_from_length_precomputed_kernel(
        muscle_length: wp.array2d(dtype=wp.float32),
        muscle_force_length_coeff: wp.array(dtype=wp.vec4),
        activation: wp.array2d(dtype=wp.float32),
        muscle_length_norm_min: float,
        muscle_length_norm_max: float,
        active_force_model_params: wp.vec3,
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        force_by_muscle: wp.array2d(dtype=wp.float32),
    ):
        env_idx, muscle_idx = wp.tid()
        coeff = native_load_vec4_wp(muscle_force_length_coeff, muscle_idx)
        norm_unit = _unit_force_from_precomputed_coeff_wp(
            muscle_length[env_idx, muscle_idx],
            coeff,
            muscle_length_norm_min,
            muscle_length_norm_max,
            active_force_model_params,
        )
        passive_force = float(0.0)
        if wp.static(include_passive_flag) != 0:
            passive_force = coeff[2] * passive_force_length_model_wp(
                norm_unit[0],
                wp.vec3(passive_model_id, passive_param0, passive_param1),
            )
        unit_force_magnitude[env_idx, muscle_idx] = norm_unit[1]
        force_by_muscle[env_idx, muscle_idx] = (
            norm_unit[1] * activation[env_idx, muscle_idx] + passive_force
        )

    return _compute_lbs_force_by_muscle_from_length_precomputed_kernel


@cache
def make_lbs_force_by_muscle_contiguous_waypoints_kernel(
    include_passive_force: bool,
    passive_force_model_id: int = 0,
    passive_force_param0: float = 5.5,
    passive_force_param1: float = 0.3,
):
    include_passive_force = bool(include_passive_force)
    include_passive_flag = 1 if include_passive_force else 0
    passive_model_id = float(passive_force_model_id)
    passive_param0 = float(passive_force_param0)
    passive_param1 = float(passive_force_param1)

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _compute_lbs_force_by_muscle_contiguous_waypoints_kernel(
        transformed_waypoints: wp.array2d(dtype=wp.vec3),
        muscle_segment_offsets: wp.array(dtype=wp.int32),
        muscle_params: wp.array2d(dtype=wp.float32),
        muscle_rest_length: wp.array(dtype=wp.float32),
        activation: wp.array2d(dtype=wp.float32),
        muscle_force_scale: float,
        muscle_length_norm_min: float,
        muscle_length_norm_max: float,
        active_force_model_params: wp.vec3,
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        force_by_muscle: wp.array2d(dtype=wp.float32),
    ):
        env_idx, muscle_idx = wp.tid()
        segment_idx = muscle_segment_offsets[muscle_idx]
        segment_end = muscle_segment_offsets[muscle_idx + 1]
        muscle_length = float(0.0)
        while segment_idx < segment_end:
            waypoint_idx = segment_idx + muscle_idx
            transformed_waypoints[env_idx, waypoint_idx]
            transformed_waypoints[env_idx, waypoint_idx + 1]
            muscle_length += segment_length_wp(
                env_idx,
                waypoint_idx,
                transformed_waypoints,
            )
            segment_idx += 1

        f0 = muscle_params[muscle_idx, 0]
        l_m0 = muscle_params[muscle_idx, 1]
        l_t0 = muscle_params[muscle_idx, 2]
        pen_angle = muscle_params[muscle_idx, 3]
        l_m_norm = normalized_muscle_length_wp(
            muscle_length,
            l_m0,
            l_t0,
            muscle_rest_length[muscle_idx],
            muscle_length_norm_min,
            muscle_length_norm_max,
        )
        unit_force = unit_force_magnitude_from_length_norm_wp(
            f0,
            pen_angle,
            muscle_force_scale,
            l_m_norm,
            active_force_model_params,
        )
        passive_force = float(0.0)
        if wp.static(include_passive_flag) != 0:
            passive_force = (
                f0
                * passive_force_length_model_wp(
                    l_m_norm,
                    wp.vec3(passive_model_id, passive_param0, passive_param1),
                )
                * wp.cos(pen_angle)
                * muscle_force_scale
            )
        unit_force_magnitude[env_idx, muscle_idx] = unit_force
        force_by_muscle[env_idx, muscle_idx] = (
            unit_force * activation[env_idx, muscle_idx] + passive_force
        )

    return _compute_lbs_force_by_muscle_contiguous_waypoints_kernel


@cache
def make_lbs_force_by_muscle_contiguous_waypoints_precomputed_kernel(
    include_passive_force: bool,
    passive_force_model_id: int = 0,
    passive_force_param0: float = 5.5,
    passive_force_param1: float = 0.3,
):
    include_passive_force = bool(include_passive_force)
    include_passive_flag = 1 if include_passive_force else 0
    passive_model_id = float(passive_force_model_id)
    passive_param0 = float(passive_force_param0)
    passive_param1 = float(passive_force_param1)

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _compute_lbs_force_by_muscle_contiguous_waypoints_precomputed_kernel(
        transformed_waypoints: wp.array2d(dtype=wp.vec3),
        muscle_segment_offsets: wp.array(dtype=wp.int32),
        muscle_force_length_coeff: wp.array(dtype=wp.vec4),
        activation: wp.array2d(dtype=wp.float32),
        muscle_length_norm_min: float,
        muscle_length_norm_max: float,
        active_force_model_params: wp.vec3,
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        force_by_muscle: wp.array2d(dtype=wp.float32),
    ):
        env_idx, muscle_idx = wp.tid()
        segment_idx = muscle_segment_offsets[muscle_idx]
        segment_end = muscle_segment_offsets[muscle_idx + 1]
        muscle_length = float(0.0)
        while segment_idx < segment_end:
            waypoint_idx = segment_idx + muscle_idx
            transformed_waypoints[env_idx, waypoint_idx]
            transformed_waypoints[env_idx, waypoint_idx + 1]
            muscle_length += segment_length_wp(
                env_idx,
                waypoint_idx,
                transformed_waypoints,
            )
            segment_idx += 1

        coeff = native_load_vec4_wp(muscle_force_length_coeff, muscle_idx)
        norm_unit = _unit_force_from_precomputed_coeff_wp(
            muscle_length,
            coeff,
            muscle_length_norm_min,
            muscle_length_norm_max,
            active_force_model_params,
        )
        passive_force = float(0.0)
        if wp.static(include_passive_flag) != 0:
            passive_force = coeff[2] * passive_force_length_model_wp(
                norm_unit[0],
                wp.vec3(passive_model_id, passive_param0, passive_param1),
            )
        unit_force_magnitude[env_idx, muscle_idx] = norm_unit[1]
        force_by_muscle[env_idx, muscle_idx] = (
            norm_unit[1] * activation[env_idx, muscle_idx] + passive_force
        )

    return _compute_lbs_force_by_muscle_contiguous_waypoints_precomputed_kernel


@cache
def make_lbs_segment_length_group_partials_kernel(group_width: int):
    group_width = max(1, int(group_width))

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _compute_lbs_segment_length_group_partials_kernel(
        transformed_waypoints: wp.array2d(dtype=wp.vec3),
        muscle_segment_offsets: wp.array(dtype=wp.int32),
        segment_length_group_partials: wp.array2d(dtype=wp.float32),
    ):
        env_idx, muscle_idx, lane_idx = wp.tid()
        segment_idx = muscle_segment_offsets[muscle_idx] + lane_idx
        segment_end = muscle_segment_offsets[muscle_idx + 1]
        total = float(0.0)
        while segment_idx < segment_end:
            waypoint_idx = segment_idx + muscle_idx
            transformed_waypoints[env_idx, waypoint_idx]
            transformed_waypoints[env_idx, waypoint_idx + 1]
            total += segment_length_wp(
                env_idx,
                waypoint_idx,
                transformed_waypoints,
            )
            segment_idx += wp.static(group_width)
        partial_idx = muscle_idx * wp.static(group_width) + lane_idx
        segment_length_group_partials[env_idx, partial_idx] = total

    return _compute_lbs_segment_length_group_partials_kernel


@cache
def make_lbs_force_by_muscle_from_group_partials_precomputed_kernel(
    include_passive_force: bool,
    group_width: int,
    passive_force_model_id: int = 0,
    passive_force_param0: float = 5.5,
    passive_force_param1: float = 0.3,
):
    include_passive_force = bool(include_passive_force)
    include_passive_flag = 1 if include_passive_force else 0
    group_width = max(1, int(group_width))
    passive_model_id = float(passive_force_model_id)
    passive_param0 = float(passive_force_param0)
    passive_param1 = float(passive_force_param1)

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _compute_lbs_force_by_muscle_from_group_partials_precomputed_kernel(
        segment_length_group_partials: wp.array2d(dtype=wp.float32),
        muscle_force_length_coeff: wp.array(dtype=wp.vec4),
        activation: wp.array2d(dtype=wp.float32),
        muscle_length_norm_min: float,
        muscle_length_norm_max: float,
        active_force_model_params: wp.vec3,
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        force_by_muscle: wp.array2d(dtype=wp.float32),
    ):
        env_idx, muscle_idx = wp.tid()
        base_idx = muscle_idx * wp.static(group_width)
        muscle_length = float(0.0)
        lane_idx = int(0)
        while lane_idx < wp.static(group_width):
            muscle_length += segment_length_group_partials[env_idx, base_idx + lane_idx]
            lane_idx += 1

        coeff = native_load_vec4_wp(muscle_force_length_coeff, muscle_idx)
        norm_unit = _unit_force_from_precomputed_coeff_wp(
            muscle_length,
            coeff,
            muscle_length_norm_min,
            muscle_length_norm_max,
            active_force_model_params,
        )
        passive_force = float(0.0)
        if wp.static(include_passive_flag) != 0:
            passive_force = coeff[2] * passive_force_length_model_wp(
                norm_unit[0],
                wp.vec3(passive_model_id, passive_param0, passive_param1),
            )
        unit_force_magnitude[env_idx, muscle_idx] = norm_unit[1]
        force_by_muscle[env_idx, muscle_idx] = (
            norm_unit[1] * activation[env_idx, muscle_idx] + passive_force
        )

    return _compute_lbs_force_by_muscle_from_group_partials_precomputed_kernel


@wp.kernel(enable_backward=False)
def _compute_lbs_segment_length_pair_partials_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    muscle_segment_offsets: wp.array(dtype=wp.int32),
    segment_length_pair_partials: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx, lane_idx = wp.tid()
    segment_idx = muscle_segment_offsets[muscle_idx] + lane_idx * 3
    segment_end = muscle_segment_offsets[muscle_idx + 1]
    total = float(0.0)
    while segment_idx < segment_end:
        waypoint_idx = segment_idx + muscle_idx
        transformed_waypoints[env_idx, waypoint_idx]
        transformed_waypoints[env_idx, waypoint_idx + 1]
        total += segment_length_wp(
            env_idx,
            waypoint_idx,
            transformed_waypoints,
        )
        next_segment_idx = segment_idx + 1
        if next_segment_idx < segment_end:
            transformed_waypoints[env_idx, waypoint_idx + 2]
            total += segment_length_wp(
                env_idx,
                waypoint_idx + 1,
                transformed_waypoints,
            )
            third_segment_idx = segment_idx + 2
            if third_segment_idx < segment_end:
                transformed_waypoints[env_idx, waypoint_idx + 3]
                total += segment_length_wp(
                    env_idx,
                    waypoint_idx + 2,
                    transformed_waypoints,
                )
        segment_idx += 6
    segment_length_pair_partials[env_idx, muscle_idx * 2 + lane_idx] = total


@cache
def make_lbs_force_by_muscle_from_pair_partials_precomputed_kernel(
    include_passive_force: bool,
    passive_force_model_id: int = 0,
    passive_force_param0: float = 5.5,
    passive_force_param1: float = 0.3,
):
    include_passive_force = bool(include_passive_force)
    include_passive_flag = 1 if include_passive_force else 0
    passive_model_id = float(passive_force_model_id)
    passive_param0 = float(passive_force_param0)
    passive_param1 = float(passive_force_param1)

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _compute_lbs_force_by_muscle_from_pair_partials_precomputed_kernel(
        segment_length_pair_partials: wp.array2d(dtype=wp.float32),
        muscle_force_length_coeff: wp.array(dtype=wp.vec4),
        activation: wp.array2d(dtype=wp.float32),
        muscle_length_norm_min: float,
        muscle_length_norm_max: float,
        active_force_model_params: wp.vec3,
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        force_by_muscle: wp.array2d(dtype=wp.float32),
    ):
        env_idx, muscle_idx = wp.tid()
        base_idx = muscle_idx * 2
        muscle_length = (
            segment_length_pair_partials[env_idx, base_idx]
            + segment_length_pair_partials[env_idx, base_idx + 1]
        )
        coeff = native_load_vec4_wp(muscle_force_length_coeff, muscle_idx)
        norm_unit = _unit_force_from_precomputed_coeff_wp(
            muscle_length,
            coeff,
            muscle_length_norm_min,
            muscle_length_norm_max,
            active_force_model_params,
        )
        passive_force = float(0.0)
        if wp.static(include_passive_flag) != 0:
            passive_force = coeff[2] * passive_force_length_model_wp(
                norm_unit[0],
                wp.vec3(passive_model_id, passive_param0, passive_param1),
            )
        unit_force_magnitude[env_idx, muscle_idx] = norm_unit[1]
        force_by_muscle[env_idx, muscle_idx] = (
            norm_unit[1] * activation[env_idx, muscle_idx] + passive_force
        )

    return _compute_lbs_force_by_muscle_from_pair_partials_precomputed_kernel


@wp.kernel(enable_backward=False)
def _compute_muscle_dynamics_contribution_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    apply_muscle_segment_offsets: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    segment_start_root_body_idx: wp.array(dtype=wp.int32),
    segment_end_root_body_idx: wp.array(dtype=wp.int32),
    segment_start_dof_mask: wp.array2d(dtype=wp.float32),
    segment_end_dof_mask: wp.array2d(dtype=wp.float32),
    root_dof_count: int,
    contribution: wp.array3d(dtype=wp.float32),
):
    env_idx, torque_idx, muscle_idx = wp.tid()

    segment_begin = apply_muscle_segment_offsets[muscle_idx]
    segment_end = apply_muscle_segment_offsets[muscle_idx + 1]
    total = float(0.0)
    full_torque_idx = torque_idx + root_dof_count
    force_magnitude = unit_force_magnitude[env_idx, muscle_idx]

    linear_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 3],
        cdof[env_idx, full_torque_idx, 4],
        cdof[env_idx, full_torque_idx, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 0],
        cdof[env_idx, full_torque_idx, 1],
        cdof[env_idx, full_torque_idx, 2],
    )

    for segment_idx in range(segment_begin, segment_end):
        start_waypoint_idx = apply_segment_start_idx[segment_idx]
        end_waypoint_idx = apply_segment_end_idx[segment_idx]
        p0 = transformed_waypoints[env_idx, start_waypoint_idx]
        p1 = transformed_waypoints[env_idx, end_waypoint_idx]

        subtree_com_start = subtree_com[
            env_idx, segment_start_root_body_idx[segment_idx]
        ]
        subtree_com_end = subtree_com[env_idx, segment_end_root_body_idx[segment_idx]]
        total += segment_generalized_force_contribution_wp(
            linear_cdof,
            angular_cdof,
            p0,
            p1,
            subtree_com_start,
            subtree_com_end,
            segment_start_dof_mask[torque_idx, segment_idx],
            segment_end_dof_mask[torque_idx, segment_idx],
            force_magnitude,
        )

    contribution[env_idx, torque_idx, muscle_idx] = total


@wp.kernel(enable_backward=False)
def _compute_muscle_dynamics_contribution_sparse_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    torque_active_torque_idx: wp.array(dtype=wp.int32),
    torque_active_segment_idx: wp.array(dtype=wp.int32),
    torque_active_muscle_idx: wp.array(dtype=wp.int32),
    torque_active_start_coeff: wp.array(dtype=wp.float32),
    torque_active_end_coeff: wp.array(dtype=wp.float32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    segment_start_root_body_idx: wp.array(dtype=wp.int32),
    segment_end_root_body_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    contribution: wp.array3d(dtype=wp.float32),
):
    env_idx, entry_idx = wp.tid()

    torque_idx = torque_active_torque_idx[entry_idx]
    muscle_idx = torque_active_muscle_idx[entry_idx]
    segment_idx = torque_active_segment_idx[entry_idx]
    full_torque_idx = torque_idx + root_dof_count

    linear_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 3],
        cdof[env_idx, full_torque_idx, 4],
        cdof[env_idx, full_torque_idx, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 0],
        cdof[env_idx, full_torque_idx, 1],
        cdof[env_idx, full_torque_idx, 2],
    )

    start_waypoint_idx = apply_segment_start_idx[segment_idx]
    end_waypoint_idx = apply_segment_end_idx[segment_idx]
    p0 = transformed_waypoints[env_idx, start_waypoint_idx]
    p1 = transformed_waypoints[env_idx, end_waypoint_idx]

    subtree_com_start = subtree_com[env_idx, segment_start_root_body_idx[segment_idx]]
    subtree_com_end = subtree_com[env_idx, segment_end_root_body_idx[segment_idx]]
    total = segment_generalized_force_contribution_wp(
        linear_cdof,
        angular_cdof,
        p0,
        p1,
        subtree_com_start,
        subtree_com_end,
        torque_active_start_coeff[entry_idx],
        torque_active_end_coeff[entry_idx],
        unit_force_magnitude[env_idx, muscle_idx],
    )
    wp.atomic_add(contribution, env_idx, torque_idx, muscle_idx, total)


@wp.kernel(enable_backward=False)
def _compute_muscle_dynamics_sparse_bounds_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    torque_active_entry_offsets: wp.array(dtype=wp.int32),
    torque_active_segment_idx: wp.array(dtype=wp.int32),
    torque_active_muscle_idx: wp.array(dtype=wp.int32),
    torque_active_start_coeff: wp.array(dtype=wp.float32),
    torque_active_end_coeff: wp.array(dtype=wp.float32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    segment_start_root_body_idx: wp.array(dtype=wp.int32),
    segment_end_root_body_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
):
    env_idx, torque_idx = wp.tid()
    entry_begin = torque_active_entry_offsets[torque_idx]
    entry_end = torque_active_entry_offsets[torque_idx + 1]
    current_muscle = int(-1)
    current_total = float(0.0)
    lower_total = float(0.0)
    upper_total = float(0.0)
    full_torque_idx = torque_idx + root_dof_count
    linear_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 3],
        cdof[env_idx, full_torque_idx, 4],
        cdof[env_idx, full_torque_idx, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 0],
        cdof[env_idx, full_torque_idx, 1],
        cdof[env_idx, full_torque_idx, 2],
    )

    for entry_idx in range(entry_begin, entry_end):
        muscle_idx = torque_active_muscle_idx[entry_idx]
        if current_muscle >= 0 and muscle_idx != current_muscle:
            if current_total < 0.0:
                lower_total += current_total
            else:
                upper_total += current_total
            current_total = 0.0
        current_muscle = muscle_idx
        segment_idx = torque_active_segment_idx[entry_idx]
        start_waypoint_idx = apply_segment_start_idx[segment_idx]
        end_waypoint_idx = apply_segment_end_idx[segment_idx]
        p0 = transformed_waypoints[env_idx, start_waypoint_idx]
        p1 = transformed_waypoints[env_idx, end_waypoint_idx]
        subtree_com_start = subtree_com[
            env_idx, segment_start_root_body_idx[segment_idx]
        ]
        subtree_com_end = subtree_com[env_idx, segment_end_root_body_idx[segment_idx]]
        current_total += segment_generalized_force_contribution_wp(
            linear_cdof,
            angular_cdof,
            p0,
            p1,
            subtree_com_start,
            subtree_com_end,
            torque_active_start_coeff[entry_idx],
            torque_active_end_coeff[entry_idx],
            unit_force_magnitude[env_idx, muscle_idx],
        )

    if current_muscle >= 0:
        if current_total < 0.0:
            lower_total += current_total
        else:
            upper_total += current_total

    lower[env_idx, torque_idx] = lower_total
    upper[env_idx, torque_idx] = upper_total


@wp.func
def _lbs_segment_entry_contribution_active_points_wp(
    env_idx: int,
    torque_idx: int,
    start_wp: int,
    end_wp: int,
    force_magnitude: float,
    linear_cdof: wp.vec3,
    angular_cdof: wp.vec3,
    p_start: wp.vec3,
    p_end: wp.vec3,
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    body_rootid: wp.array(dtype=wp.int32),
    start_begin: int,
    start_end: int,
    end_begin: int,
    end_end: int,
    active_attach_body: wp.array(dtype=wp.int32),
    active_attach_local: wp.array(dtype=wp.vec3),
    active_attach_coeff_weight: wp.array(dtype=wp.float32),
) -> float:
    segment = p_end - p_start
    segment_length = wp.length(segment)
    if segment_length < 1.0e-8:
        return 0.0

    seg_force = force_magnitude * segment / segment_length

    total = float(0.0)
    for active_idx in range(start_begin, start_end):
        body_idx = active_attach_body[active_idx]
        q = _normalize_quat_wp(body_quat[env_idx, body_idx])
        point = body_pos[env_idx, body_idx] + rot_vec_quat(
            active_attach_local[active_idx], q
        )
        root_idx = body_rootid[body_idx]
        term = linear_cdof + wp.cross(
            angular_cdof,
            point - subtree_com[env_idx, root_idx],
        )
        total += active_attach_coeff_weight[active_idx] * wp.dot(term, seg_force)

    for active_idx in range(end_begin, end_end):
        body_idx = active_attach_body[active_idx]
        q = _normalize_quat_wp(body_quat[env_idx, body_idx])
        point = body_pos[env_idx, body_idx] + rot_vec_quat(
            active_attach_local[active_idx], q
        )
        root_idx = body_rootid[body_idx]
        term = linear_cdof + wp.cross(
            angular_cdof,
            point - subtree_com[env_idx, root_idx],
        )
        total -= active_attach_coeff_weight[active_idx] * wp.dot(term, seg_force)
    return total


@wp.func
def _lbs_incident_segment_force_from_force_wp(
    env_idx: int,
    segment_idx: int,
    sign: int,
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    force_by_muscle: wp.array2d(dtype=wp.float32),
    segment_waypoints: wp.array(dtype=wp.vec2i),
    segment_muscle_idx: wp.array(dtype=wp.int32),
) -> wp.vec3:
    if segment_idx < 0 or sign == 0:
        return wp.vec3(0.0, 0.0, 0.0)

    waypoints = segment_waypoints[segment_idx]
    p_start = transformed_waypoints[env_idx, waypoints[0]]
    p_end = transformed_waypoints[env_idx, waypoints[1]]
    segment = p_end - p_start
    segment_length = wp.length(segment)
    if segment_length < 1.0e-8:
        return wp.vec3(0.0, 0.0, 0.0)

    muscle_idx = segment_muscle_idx[segment_idx]
    force_magnitude = force_by_muscle[env_idx, muscle_idx]
    return float(sign) * force_magnitude * segment / segment_length


@cache
def make_lbs_waypoint_force_from_force_kernel():
    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _compute_lbs_waypoint_force_from_force_kernel(
        transformed_waypoints: wp.array2d(dtype=wp.vec3),
        force_by_muscle: wp.array2d(dtype=wp.float32),
        waypoint_segment_idx2: wp.array(dtype=wp.vec2i),
        waypoint_segment_sign2: wp.array(dtype=wp.vec2i),
        segment_waypoints: wp.array(dtype=wp.vec2i),
        segment_muscle_idx: wp.array(dtype=wp.int32),
        waypoint_force: wp.array2d(dtype=wp.vec3),
    ):
        env_idx, waypoint_idx = wp.tid()
        segment_idx2 = waypoint_segment_idx2[waypoint_idx]
        segment_sign2 = waypoint_segment_sign2[waypoint_idx]
        total = _lbs_incident_segment_force_from_force_wp(
            env_idx,
            segment_idx2[0],
            segment_sign2[0],
            transformed_waypoints,
            force_by_muscle,
            segment_waypoints,
            segment_muscle_idx,
        )
        total += _lbs_incident_segment_force_from_force_wp(
            env_idx,
            segment_idx2[1],
            segment_sign2[1],
            transformed_waypoints,
            force_by_muscle,
            segment_waypoints,
            segment_muscle_idx,
        )
        waypoint_force[env_idx, waypoint_idx] = total

    return _compute_lbs_waypoint_force_from_force_kernel


@wp.func
def _lbs_segment_entry_contribution_active_wp(
    env_idx: int,
    torque_idx: int,
    segment_idx: int,
    force_magnitude: float,
    linear_cdof: wp.vec3,
    angular_cdof: wp.vec3,
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    body_rootid: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    active_attach_offsets: wp.array(dtype=wp.int32),
    active_attach_body: wp.array(dtype=wp.int32),
    active_attach_local: wp.array(dtype=wp.vec3),
    active_attach_coeff_weight: wp.array(dtype=wp.float32),
    waypoint_count: int,
) -> float:
    start_wp = apply_segment_start_idx[segment_idx]
    end_wp = apply_segment_end_idx[segment_idx]
    p_start = transformed_waypoints[env_idx, start_wp]
    p_end = transformed_waypoints[env_idx, end_wp]
    start_offset_idx = torque_idx * waypoint_count + start_wp
    end_offset_idx = torque_idx * waypoint_count + end_wp
    return _lbs_segment_entry_contribution_active_points_wp(
        env_idx,
        torque_idx,
        start_wp,
        end_wp,
        force_magnitude,
        linear_cdof,
        angular_cdof,
        p_start,
        p_end,
        body_pos,
        body_quat,
        subtree_com,
        body_rootid,
        active_attach_offsets[start_offset_idx],
        active_attach_offsets[start_offset_idx + 1],
        active_attach_offsets[end_offset_idx],
        active_attach_offsets[end_offset_idx + 1],
        active_attach_body,
        active_attach_local,
        active_attach_coeff_weight,
    )


@cache
def make_muscle_dynamics_from_activation_sparse_kernel(include_passive_force: bool):
    """Build the kernel with or without the passive force term.

    `include_passive_force` is baked in by `wp.static`, so the two variants are
    separate specializations inside one Warp module -- same `kernel.key`,
    different kernel hashes -- rather than one kernel with a runtime branch.
    The passive-off variant's load sits behind a statically false branch and
    never runs, so its caller passes `None`; Warp packs that as a null array
    descriptor.
    """

    include_passive_flag = 1 if bool(include_passive_force) else 0

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _compute_muscle_dynamics_from_activation_sparse_kernel(
        transformed_waypoints: wp.array2d(dtype=wp.vec3),
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        passive_force_magnitude: wp.array2d(dtype=wp.float32),
        activation: wp.array2d(dtype=wp.float32),
        subtree_com: wp.array2d(dtype=wp.vec3),
        cdof: wp.array3d(dtype=wp.float32),
        torque_active_entry_offsets: wp.array(dtype=wp.int32),
        torque_active_segment_idx: wp.array(dtype=wp.int32),
        torque_active_muscle_idx: wp.array(dtype=wp.int32),
        torque_active_start_coeff: wp.array(dtype=wp.float32),
        torque_active_end_coeff: wp.array(dtype=wp.float32),
        apply_segment_start_idx: wp.array(dtype=wp.int32),
        apply_segment_end_idx: wp.array(dtype=wp.int32),
        segment_start_root_body_idx: wp.array(dtype=wp.int32),
        segment_end_root_body_idx: wp.array(dtype=wp.int32),
        root_dof_count: int,
        mean: wp.array2d(dtype=wp.float32),
    ):
        env_idx, torque_idx = wp.tid()
        full_torque_idx = torque_idx + root_dof_count
        linear_cdof = wp.vec3(
            cdof[env_idx, full_torque_idx, 3],
            cdof[env_idx, full_torque_idx, 4],
            cdof[env_idx, full_torque_idx, 5],
        )
        angular_cdof = wp.vec3(
            cdof[env_idx, full_torque_idx, 0],
            cdof[env_idx, full_torque_idx, 1],
            cdof[env_idx, full_torque_idx, 2],
        )

        total = float(0.0)
        entry_begin = torque_active_entry_offsets[torque_idx]
        entry_end = torque_active_entry_offsets[torque_idx + 1]
        for entry_idx in range(entry_begin, entry_end):
            segment_idx = torque_active_segment_idx[entry_idx]
            muscle_idx = torque_active_muscle_idx[entry_idx]
            force_magnitude = (
                unit_force_magnitude[env_idx, muscle_idx]
                * activation[env_idx, muscle_idx]
            )
            if wp.static(include_passive_flag) != 0:
                force_magnitude += passive_force_magnitude[env_idx, muscle_idx]
            if force_magnitude == 0.0:
                continue
            start_waypoint_idx = apply_segment_start_idx[segment_idx]
            end_waypoint_idx = apply_segment_end_idx[segment_idx]
            p0 = transformed_waypoints[env_idx, start_waypoint_idx]
            p1 = transformed_waypoints[env_idx, end_waypoint_idx]

            subtree_com_start = subtree_com[
                env_idx, segment_start_root_body_idx[segment_idx]
            ]
            subtree_com_end = subtree_com[
                env_idx, segment_end_root_body_idx[segment_idx]
            ]
            total += segment_generalized_force_contribution_wp(
                linear_cdof,
                angular_cdof,
                p0,
                p1,
                subtree_com_start,
                subtree_com_end,
                torque_active_start_coeff[entry_idx],
                torque_active_end_coeff[entry_idx],
                force_magnitude,
            )

        mean[env_idx, torque_idx] = total

    return _compute_muscle_dynamics_from_activation_sparse_kernel


@wp.kernel(enable_backward=False)
def _compute_muscle_dynamics_activation_grad_sparse_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    muscle_active_entry_offsets: wp.array(dtype=wp.int32),
    muscle_active_segment_idx: wp.array(dtype=wp.int32),
    muscle_active_torque_idx: wp.array(dtype=wp.int32),
    muscle_active_start_coeff: wp.array(dtype=wp.float32),
    muscle_active_end_coeff: wp.array(dtype=wp.float32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    segment_start_root_body_idx: wp.array(dtype=wp.int32),
    segment_end_root_body_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    grad_output: wp.array2d(dtype=wp.float32),
    grad_activation: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    force_magnitude = unit_force_magnitude[env_idx, muscle_idx]
    if force_magnitude == 0.0:
        return

    grad = float(0.0)
    entry_begin = muscle_active_entry_offsets[muscle_idx]
    entry_end = muscle_active_entry_offsets[muscle_idx + 1]
    for entry_idx in range(entry_begin, entry_end):
        segment_idx = muscle_active_segment_idx[entry_idx]
        start_waypoint_idx = apply_segment_start_idx[segment_idx]
        end_waypoint_idx = apply_segment_end_idx[segment_idx]
        p0 = transformed_waypoints[env_idx, start_waypoint_idx]
        p1 = transformed_waypoints[env_idx, end_waypoint_idx]
        torque_idx = muscle_active_torque_idx[entry_idx]
        full_torque_idx = torque_idx + root_dof_count
        linear_cdof = wp.vec3(
            cdof[env_idx, full_torque_idx, 3],
            cdof[env_idx, full_torque_idx, 4],
            cdof[env_idx, full_torque_idx, 5],
        )
        angular_cdof = wp.vec3(
            cdof[env_idx, full_torque_idx, 0],
            cdof[env_idx, full_torque_idx, 1],
            cdof[env_idx, full_torque_idx, 2],
        )
        subtree_com_start = subtree_com[
            env_idx, segment_start_root_body_idx[segment_idx]
        ]
        subtree_com_end = subtree_com[env_idx, segment_end_root_body_idx[segment_idx]]
        grad += grad_output[
            env_idx, torque_idx
        ] * segment_generalized_force_contribution_wp(
            linear_cdof,
            angular_cdof,
            p0,
            p1,
            subtree_com_start,
            subtree_com_end,
            muscle_active_start_coeff[entry_idx],
            muscle_active_end_coeff[entry_idx],
            force_magnitude,
        )
    grad_activation[env_idx, muscle_idx] = grad


@cache
def make_accumulate_lbs_body_wrench_from_activation_kernel(include_passive_force: bool):
    """Build the kernel with or without the passive force term.

    `include_passive_force` is baked in by `wp.static`, so the two variants are
    separate specializations inside one Warp module -- same `kernel.key`,
    different kernel hashes -- rather than one kernel with a runtime branch.
    The passive-off variant's load sits behind a statically false branch and
    never runs, so its caller passes `None`; Warp packs that as a null array
    descriptor.
    """

    include_passive_flag = 1 if bool(include_passive_force) else 0

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _accumulate_lbs_body_wrench_from_activation_kernel(
        transformed_waypoints: wp.array2d(dtype=wp.vec3),
        body_pos: wp.array2d(dtype=wp.vec3),
        body_quat: wp.array2d(dtype=wp.quat),
        xipos: wp.array2d(dtype=wp.vec3),
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        passive_force_magnitude: wp.array2d(dtype=wp.float32),
        activation: wp.array2d(dtype=wp.float32),
        apply_segment_muscle_idx: wp.array(dtype=wp.int32),
        apply_segment_start_idx: wp.array(dtype=wp.int32),
        apply_segment_end_idx: wp.array(dtype=wp.int32),
        waypoint_attachment_start: wp.array(dtype=wp.int32),
        waypoint_attachment_body: wp.array(dtype=wp.int32),
        waypoint_attachment_local: wp.array(dtype=wp.vec3),
        waypoint_attachment_weight: wp.array(dtype=wp.float32),
        body_force: wp.array3d(dtype=wp.float32),
        body_torque: wp.array3d(dtype=wp.float32),
    ):
        env_idx, segment_idx = wp.tid()
        muscle_idx = apply_segment_muscle_idx[segment_idx]
        force_magnitude = (
            unit_force_magnitude[env_idx, muscle_idx] * activation[env_idx, muscle_idx]
        )
        if wp.static(include_passive_flag) != 0:
            force_magnitude += passive_force_magnitude[env_idx, muscle_idx]
        if force_magnitude == 0.0:
            return

        start_wp = apply_segment_start_idx[segment_idx]
        end_wp = apply_segment_end_idx[segment_idx]
        p_start = transformed_waypoints[env_idx, start_wp]
        p_end = transformed_waypoints[env_idx, end_wp]
        segment = p_end - p_start
        segment_length = wp.length(segment)
        if segment_length < 1.0e-8:
            return
        seg_force = force_magnitude * segment / segment_length

        start_begin = waypoint_attachment_start[start_wp]
        start_end = waypoint_attachment_start[start_wp + 1]
        for attach_idx in range(start_begin, start_end):
            body_idx = waypoint_attachment_body[attach_idx]
            q = _normalize_quat_wp(body_quat[env_idx, body_idx])
            point = body_pos[env_idx, body_idx] + rot_vec_quat(
                waypoint_attachment_local[attach_idx], q
            )
            signed_force = waypoint_attachment_weight[attach_idx] * seg_force
            body_com = xipos[env_idx, body_idx]
            torque = wp.cross(point - body_com, signed_force)
            wp.atomic_add(body_force, env_idx, body_idx, 0, signed_force[0])
            wp.atomic_add(body_force, env_idx, body_idx, 1, signed_force[1])
            wp.atomic_add(body_force, env_idx, body_idx, 2, signed_force[2])
            wp.atomic_add(body_torque, env_idx, body_idx, 0, torque[0])
            wp.atomic_add(body_torque, env_idx, body_idx, 1, torque[1])
            wp.atomic_add(body_torque, env_idx, body_idx, 2, torque[2])

        end_begin = waypoint_attachment_start[end_wp]
        end_end = waypoint_attachment_start[end_wp + 1]
        for attach_idx in range(end_begin, end_end):
            body_idx = waypoint_attachment_body[attach_idx]
            q = _normalize_quat_wp(body_quat[env_idx, body_idx])
            point = body_pos[env_idx, body_idx] + rot_vec_quat(
                waypoint_attachment_local[attach_idx], q
            )
            signed_force = -waypoint_attachment_weight[attach_idx] * seg_force
            body_com = xipos[env_idx, body_idx]
            torque = wp.cross(point - body_com, signed_force)
            wp.atomic_add(body_force, env_idx, body_idx, 0, signed_force[0])
            wp.atomic_add(body_force, env_idx, body_idx, 1, signed_force[1])
            wp.atomic_add(body_force, env_idx, body_idx, 2, signed_force[2])
            wp.atomic_add(body_torque, env_idx, body_idx, 0, torque[0])
            wp.atomic_add(body_torque, env_idx, body_idx, 1, torque[1])
            wp.atomic_add(body_torque, env_idx, body_idx, 2, torque[2])

    return _accumulate_lbs_body_wrench_from_activation_kernel


@cache
def make_accumulate_body_wrench_direct_from_activation_kernel(
    include_passive_force: bool,
):
    """Build the kernel with or without the passive force term.

    `include_passive_force` is baked in by `wp.static`, so the two variants are
    separate specializations inside one Warp module -- same `kernel.key`,
    different kernel hashes -- rather than one kernel with a runtime branch.
    The passive-off variant's load sits behind a statically false branch and
    never runs, so its caller passes `None`; Warp packs that as a null array
    descriptor.
    """

    include_passive_flag = 1 if bool(include_passive_force) else 0

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _accumulate_body_wrench_direct_from_activation_kernel(
        transformed_waypoints: wp.array2d(dtype=wp.vec3),
        xipos: wp.array2d(dtype=wp.vec3),
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        passive_force_magnitude: wp.array2d(dtype=wp.float32),
        activation: wp.array2d(dtype=wp.float32),
        apply_segment_muscle_idx: wp.array(dtype=wp.int32),
        apply_segment_start_idx: wp.array(dtype=wp.int32),
        apply_segment_end_idx: wp.array(dtype=wp.int32),
        apply_segment_body_start_idx: wp.array(dtype=wp.int32),
        apply_segment_body_end_idx: wp.array(dtype=wp.int32),
        body_force: wp.array3d(dtype=wp.float32),
        body_torque: wp.array3d(dtype=wp.float32),
    ):
        env_idx, segment_idx = wp.tid()
        muscle_idx = apply_segment_muscle_idx[segment_idx]
        force_magnitude = (
            unit_force_magnitude[env_idx, muscle_idx] * activation[env_idx, muscle_idx]
        )
        if wp.static(include_passive_flag) != 0:
            force_magnitude += passive_force_magnitude[env_idx, muscle_idx]
        if force_magnitude == 0.0:
            return

        start_wp = apply_segment_start_idx[segment_idx]
        end_wp = apply_segment_end_idx[segment_idx]
        p_start = transformed_waypoints[env_idx, start_wp]
        p_end = transformed_waypoints[env_idx, end_wp]
        segment = p_end - p_start
        segment_length = wp.length(segment)
        if segment_length < 1.0e-8:
            return
        seg_force = force_magnitude * segment / segment_length

        start_body = apply_segment_body_start_idx[segment_idx]
        start_com = xipos[env_idx, start_body]
        start_torque = wp.cross(p_start - start_com, seg_force)
        wp.atomic_add(body_force, env_idx, start_body, 0, seg_force[0])
        wp.atomic_add(body_force, env_idx, start_body, 1, seg_force[1])
        wp.atomic_add(body_force, env_idx, start_body, 2, seg_force[2])
        wp.atomic_add(body_torque, env_idx, start_body, 0, start_torque[0])
        wp.atomic_add(body_torque, env_idx, start_body, 1, start_torque[1])
        wp.atomic_add(body_torque, env_idx, start_body, 2, start_torque[2])

        end_body = apply_segment_body_end_idx[segment_idx]
        end_force = -seg_force
        end_com = xipos[env_idx, end_body]
        end_torque = wp.cross(p_end - end_com, end_force)
        wp.atomic_add(body_force, env_idx, end_body, 0, end_force[0])
        wp.atomic_add(body_force, env_idx, end_body, 1, end_force[1])
        wp.atomic_add(body_force, env_idx, end_body, 2, end_force[2])
        wp.atomic_add(body_torque, env_idx, end_body, 0, end_torque[0])
        wp.atomic_add(body_torque, env_idx, end_body, 1, end_torque[1])
        wp.atomic_add(body_torque, env_idx, end_body, 2, end_torque[2])

    return _accumulate_body_wrench_direct_from_activation_kernel


@wp.kernel(enable_backward=False, module=_TILED_KERNEL_MODULE)
def _accumulate_lbs_body_wrench_segment_event_chunk_partials_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    xipos: wp.array2d(dtype=wp.vec3),
    force_by_muscle: wp.array2d(dtype=wp.float32),
    body_segment_event_chunk_info: wp.array(dtype=wp.vec3i),
    body_segment_event_packed_indices: wp.array(dtype=wp.vec4i),
    body_segment_event_local_force_coeff: wp.array(dtype=wp.vec4),
    partial_wrench: wp.array3d(dtype=wp.float32),
):
    env_idx, chunk_idx, lane_idx = wp.tid()
    chunk = body_segment_event_chunk_info[chunk_idx]
    body_idx = chunk[0]
    event_idx = chunk[1] + lane_idx
    event_end = chunk[2]

    body_q = _normalize_quat_wp(body_quat[env_idx, body_idx])
    body_p = body_pos[env_idx, body_idx]
    body_com = xipos[env_idx, body_idx]
    body_p_from_com = body_p - body_com
    total_force = wp.vec3(0.0, 0.0, 0.0)
    total_torque = wp.vec3(0.0, 0.0, 0.0)
    while event_idx < event_end:
        packed_indices = native_load_vec4i_wp(
            body_segment_event_packed_indices,
            event_idx,
        )
        local_force_coeff = native_load_vec4_wp(
            body_segment_event_local_force_coeff,
            event_idx,
        )
        start_wp = packed_indices[0]
        end_wp = packed_indices[1]
        muscle_idx = packed_indices[2]
        force_magnitude = force_by_muscle[env_idx, muscle_idx]
        if force_magnitude != 0.0:
            p_start = transformed_waypoints[env_idx, start_wp]
            p_end = transformed_waypoints[env_idx, end_wp]
            segment = p_end - p_start
            segment_length = wp.length(segment)
            if segment_length >= 1.0e-8:
                force_vec = force_magnitude * segment / segment_length
                local_coeff = wp.vec3(
                    local_force_coeff[0],
                    local_force_coeff[1],
                    local_force_coeff[2],
                )
                force_coeff = local_force_coeff[3]
                signed_force = force_coeff * force_vec
                moment_arm = force_coeff * body_p_from_com + rot_vec_quat(
                    local_coeff, body_q
                )
                total_force += signed_force
                total_torque += wp.cross(moment_arm, force_vec)
        event_idx += wp.block_dim()

    fx_tile = wp.tile(total_force[0])
    fy_tile = wp.tile(total_force[1])
    fz_tile = wp.tile(total_force[2])
    tx_tile = wp.tile(total_torque[0])
    ty_tile = wp.tile(total_torque[1])
    tz_tile = wp.tile(total_torque[2])
    fx_sum = wp.tile_sum(fx_tile)
    fy_sum = wp.tile_sum(fy_tile)
    fz_sum = wp.tile_sum(fz_tile)
    tx_sum = wp.tile_sum(tx_tile)
    ty_sum = wp.tile_sum(ty_tile)
    tz_sum = wp.tile_sum(tz_tile)
    if lane_idx == 0:
        partial_wrench[env_idx, chunk_idx, 0] = wp.tile_extract(fx_sum, 0)
        partial_wrench[env_idx, chunk_idx, 1] = wp.tile_extract(fy_sum, 0)
        partial_wrench[env_idx, chunk_idx, 2] = wp.tile_extract(fz_sum, 0)
        partial_wrench[env_idx, chunk_idx, 3] = wp.tile_extract(tx_sum, 0)
        partial_wrench[env_idx, chunk_idx, 4] = wp.tile_extract(ty_sum, 0)
        partial_wrench[env_idx, chunk_idx, 5] = wp.tile_extract(tz_sum, 0)


@wp.kernel(enable_backward=False, module=_TILED_KERNEL_MODULE)
def _accumulate_lbs_body_wrench_segment_start_event_chunk_partials_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    xipos: wp.array2d(dtype=wp.vec3),
    force_by_muscle: wp.array2d(dtype=wp.float32),
    body_segment_event_chunk_info: wp.array(dtype=wp.vec3i),
    body_segment_event_start_muscle: wp.array(dtype=wp.vec2i),
    body_segment_event_local_force_coeff: wp.array(dtype=wp.vec4),
    partial_wrench: wp.array3d(dtype=wp.float32),
):
    env_idx, chunk_idx, lane_idx = wp.tid()
    chunk = body_segment_event_chunk_info[chunk_idx]
    body_idx = chunk[0]
    event_idx = chunk[1] + lane_idx
    event_end = chunk[2]

    body_q = _normalize_quat_wp(body_quat[env_idx, body_idx])
    body_p = body_pos[env_idx, body_idx]
    body_com = xipos[env_idx, body_idx]
    body_p_from_com = body_p - body_com
    total_force = wp.vec3(0.0, 0.0, 0.0)
    total_torque = wp.vec3(0.0, 0.0, 0.0)
    while event_idx < event_end:
        start_muscle = native_load_vec2i_wp(
            body_segment_event_start_muscle,
            event_idx,
        )
        local_force_coeff = native_load_vec4_wp(
            body_segment_event_local_force_coeff,
            event_idx,
        )
        start_wp = start_muscle[0]
        muscle_idx = start_muscle[1]
        force_magnitude = force_by_muscle[env_idx, muscle_idx]
        p_start = transformed_waypoints[env_idx, start_wp]
        p_end = transformed_waypoints[env_idx, start_wp + 1]
        segment = p_end - p_start
        segment_length_sq = wp.dot(segment, segment)
        inv_segment_length = 1.0 / wp.sqrt(wp.max(segment_length_sq, 1.0e-16))
        force_vec = force_magnitude * segment * inv_segment_length
        local_coeff = wp.vec3(
            local_force_coeff[0],
            local_force_coeff[1],
            local_force_coeff[2],
        )
        force_coeff = local_force_coeff[3]
        signed_force = force_coeff * force_vec
        moment_arm = force_coeff * body_p_from_com + rot_vec_quat(local_coeff, body_q)
        total_force += signed_force
        total_torque += wp.cross(moment_arm, force_vec)
        event_idx += wp.block_dim()

    fx_tile = wp.tile(total_force[0])
    fy_tile = wp.tile(total_force[1])
    fz_tile = wp.tile(total_force[2])
    tx_tile = wp.tile(total_torque[0])
    ty_tile = wp.tile(total_torque[1])
    tz_tile = wp.tile(total_torque[2])
    fx_sum = wp.tile_sum(fx_tile)
    fy_sum = wp.tile_sum(fy_tile)
    fz_sum = wp.tile_sum(fz_tile)
    tx_sum = wp.tile_sum(tx_tile)
    ty_sum = wp.tile_sum(ty_tile)
    tz_sum = wp.tile_sum(tz_tile)
    if lane_idx == 0:
        partial_wrench[env_idx, chunk_idx, 0] = wp.tile_extract(fx_sum, 0)
        partial_wrench[env_idx, chunk_idx, 1] = wp.tile_extract(fy_sum, 0)
        partial_wrench[env_idx, chunk_idx, 2] = wp.tile_extract(fz_sum, 0)
        partial_wrench[env_idx, chunk_idx, 3] = wp.tile_extract(tx_sum, 0)
        partial_wrench[env_idx, chunk_idx, 4] = wp.tile_extract(ty_sum, 0)
        partial_wrench[env_idx, chunk_idx, 5] = wp.tile_extract(tz_sum, 0)


def _native_float_literal(value: float) -> str:
    literal = f"{float(value):.17g}"
    if "." not in literal and "e" not in literal:
        literal += ".0"
    return f"{literal}f"


# Body slots per native tile, reduced in shared memory. The packed tile event
# holds the local start point (bits 0-8), the tile muscle (bits 9-13) and the
# body slot (bits 14 and up).
NATIVE_TILE_MAX_BODIES = 32


def _specialize_native_tile_limits(
    native_source: str,
    *,
    muscle_cap: int,
    waypoint_cap: int,
    body_cap: int,
    block_dim: int,
) -> str:
    limits = {
        "muscle_cap": int(muscle_cap),
        "waypoint_cap": int(waypoint_cap),
        "body_cap": int(body_cap),
        "block_dim": int(block_dim),
    }
    if not 1 <= limits["muscle_cap"] <= 32:
        raise ValueError("native tile muscle_cap must be in [1, 32]")
    if not 2 <= limits["waypoint_cap"] <= 288:
        raise ValueError("native tile waypoint_cap must be in [2, 288]")
    if not 1 <= limits["body_cap"] <= NATIVE_TILE_MAX_BODIES:
        raise ValueError(
            f"native tile body_cap must be in [1, {NATIVE_TILE_MAX_BODIES}]"
        )
    if limits["block_dim"] not in {32, 64, 128, 256}:
        raise ValueError("native tile block_dim must be one of 32, 64, 128, or 256")
    replacements = {
        "constexpr int kMaxTileMuscles = 32;": (
            f"constexpr int kMaxTileMuscles = {limits['muscle_cap']};"
        ),
        "constexpr int kMaxTileWaypoints = 288;": (
            f"constexpr int kMaxTileWaypoints = {limits['waypoint_cap']};"
        ),
        "constexpr int kMaxTileBodies = 4;": (
            f"constexpr int kMaxTileBodies = {limits['body_cap']};"
        ),
        "constexpr int kMaxBlockDim = 128;": (
            f"constexpr int kMaxBlockDim = {limits['block_dim']};"
        ),
    }
    for original, specialized in replacements.items():
        if original not in native_source:
            raise RuntimeError(
                f"Native tile source is missing specialization token {original!r}"
            )
        native_source = native_source.replace(original, specialized)
    return native_source


@cache
def _make_accumulate_lbs_body_wrench_native_tile_full_force_kernel(
    attachment_arity: int,
    include_passive_force: bool,
    passive_force_model_id: int = 0,
    passive_force_param0: float = 5.5,
    passive_force_param1: float = 0.3,
    muscle_cap: int = 32,
    waypoint_cap: int = 288,
    body_cap: int = 4,
    block_dim: int = 128,
    wrapped: bool = False,
    active_force_model: tuple[float, float, float] = (0.0, 0.45, 0.0),
    force_velocity: bool = False,
):
    """``active_force_model`` is the (type, param0, param1) of
    ``active_force_length_model_wp``: (0, gamma, -) Gaussian or
    (1, lmin, lmax) MuJoCo, compiled in as constants.

    ``force_velocity`` builds the environment variant that evaluates each
    single-body path's lengthening velocity in the tile and scales the active
    force by MuJoCo's force-velocity curve; it reads activation in model
    muscle order."""
    if int(attachment_arity) not in {1, 2}:
        raise ValueError("native tile attachment_arity must be 1 or 2")
    active_type, active_param0, active_param1 = (float(v) for v in active_force_model)
    if active_type < 0.5:
        native_active_model = 0
        native_active_param0 = 1.0 / active_param0
        native_active_param1 = 0.0
    else:
        native_active_model = 1
        native_active_param0 = active_param0
        native_active_param1 = active_param1
    if not include_passive_force:
        native_passive_model = 0
        native_passive_param0 = 0.0
        native_passive_param1 = 0.0
    elif int(passive_force_model_id) == 0:
        native_passive_model = 1
        native_passive_param0 = float(passive_force_param0) / float(
            passive_force_param1
        )
        native_passive_param1 = 1.0 / math.expm1(float(passive_force_param0))
    elif int(passive_force_model_id) == 1:
        native_passive_model = 2
        native_passive_param0 = 2.0 / (float(passive_force_param0) - 1.0)
        native_passive_param1 = float(passive_force_param1)
    else:
        raise ValueError(
            f"Unsupported passive force model id {passive_force_model_id!r}"
        )

    specialization_key = (
        int(attachment_arity),
        native_passive_model,
        native_passive_param0,
        native_passive_param1,
        int(muscle_cap),
        int(waypoint_cap),
        int(body_cap),
        int(block_dim),
        bool(wrapped),
        native_active_model,
        native_active_param0,
        native_active_param1,
    )
    if force_velocity:
        if int(attachment_arity) != 1:
            raise ValueError("native tile force-velocity requires single-body paths")
        specialization_key = (*specialization_key, "force_velocity")
    specialization_suffix = hashlib.sha1(
        repr(specialization_key).encode("utf-8")
    ).hexdigest()[:12]
    native_source = _native_snippet_with_line_mapping(
        "lbs_body_wrench_native_tile_full_force_fixed_arity.cuh"
    )
    native_source = native_source.replace(
        "__INSTANTMIMIC_FORCE_VELOCITY__", str(int(bool(force_velocity)))
    )
    native_source = (
        native_source.replace("__INSTANTMIMIC_ACTIVE_MODEL__", str(native_active_model))
        .replace(
            "__INSTANTMIMIC_ACTIVE_PARAM0__",
            _native_float_literal(native_active_param0),
        )
        .replace(
            "__INSTANTMIMIC_ACTIVE_PARAM1__",
            _native_float_literal(native_active_param1),
        )
    )
    native_source = native_source.replace(
        "__INSTANTMIMIC_ATTACHMENT_ARITY__",
        str(int(attachment_arity)),
    )
    native_source = _specialize_native_tile_limits(
        native_source,
        muscle_cap=muscle_cap,
        waypoint_cap=waypoint_cap,
        body_cap=body_cap,
        block_dim=block_dim,
    )
    native_source = (
        native_source.replace(
            "__INSTANTMIMIC_PASSIVE_MODEL__",
            str(native_passive_model),
        )
        .replace(
            "__INSTANTMIMIC_PASSIVE_PARAM0__",
            _native_float_literal(native_passive_param0),
        )
        .replace(
            "__INSTANTMIMIC_PASSIVE_PARAM1__",
            _native_float_literal(native_passive_param1),
        )
    )

    native_source = native_source.replace("__INSTANTMIMIC_WRAPPED__", str(int(wrapped)))
    if force_velocity:
        return _make_native_tile_full_force_velocity_kernel(
            native_source,
            attachment_arity=int(attachment_arity),
            specialization_suffix=specialization_suffix,
        )

    def native_accumulate_lbs_body_wrench_native_tile_full_force_specialized(
        body_pos: wp.array2d(dtype=wp.vec3),
        wrapped_points: wp.array2d(dtype=wp.vec3),
        point_source: wp.array(dtype=wp.int32),
        wrap_state: WaypointWrapKernelState,
        normalized_body_quat: wp.array2d(dtype=wp.quat),
        xipos: wp.array2d(dtype=wp.vec3),
        muscle_segment_offsets: wp.array(dtype=wp.int32),
        muscle_force_length_coeff: wp.array(dtype=wp.vec4),
        activation: wp.array2d(dtype=wp.float32),
        muscle_length_norm_min: float,
        muscle_length_norm_max: float,
        tile_info: wp.array(dtype=wp.vec4i),
        tile_extra_info: wp.array(dtype=wp.vec4i),
        tile_body_ids: wp.array(dtype=wp.int32),
        tile_body_output_idx: wp.array(dtype=wp.int32),
        tile_event_start_slot_muscle: wp.array(dtype=wp.int32),
        tile_event_local_force_coeff: wp.array(dtype=wp.vec4),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
        partial_wrench: wp.array3d(dtype=wp.float32),
        env_idx: int,
        tile_idx: int,
        lane_idx: int,
    ): ...

    native_name = (
        "native_accumulate_lbs_wrenches_full_force_"
        f"fixed{int(attachment_arity)}_"
        f"{specialization_suffix}"
    )
    native_accumulate_lbs_body_wrench_native_tile_full_force_specialized.__name__ = (
        native_name
    )
    native_accumulate_lbs_body_wrench_native_tile_full_force_specialized.__qualname__ = native_name
    native_specialized = wp.func_native(native_source)(
        native_accumulate_lbs_body_wrench_native_tile_full_force_specialized
    )

    def _accumulate_lbs_body_wrench_native_tile_full_force_specialized_kernel(
        body_pos: wp.array2d(dtype=wp.vec3),
        wrapped_points: wp.array2d(dtype=wp.vec3),
        point_source: wp.array(dtype=wp.int32),
        wrap_state: WaypointWrapKernelState,
        normalized_body_quat: wp.array2d(dtype=wp.quat),
        xipos: wp.array2d(dtype=wp.vec3),
        muscle_segment_offsets: wp.array(dtype=wp.int32),
        muscle_force_length_coeff: wp.array(dtype=wp.vec4),
        activation: wp.array2d(dtype=wp.float32),
        muscle_length_norm_min: float,
        muscle_length_norm_max: float,
        tile_info: wp.array(dtype=wp.vec4i),
        tile_extra_info: wp.array(dtype=wp.vec4i),
        tile_body_ids: wp.array(dtype=wp.int32),
        tile_body_output_idx: wp.array(dtype=wp.int32),
        tile_event_start_slot_muscle: wp.array(dtype=wp.int32),
        tile_event_local_force_coeff: wp.array(dtype=wp.vec4),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
        partial_wrench: wp.array3d(dtype=wp.float32),
    ):
        env_idx, tile_idx, lane_idx = wp.tid()
        native_specialized(
            body_pos,
            wrapped_points,
            point_source,
            wrap_state,
            normalized_body_quat,
            xipos,
            muscle_segment_offsets,
            muscle_force_length_coeff,
            activation,
            muscle_length_norm_min,
            muscle_length_norm_max,
            tile_info,
            tile_extra_info,
            tile_body_ids,
            tile_body_output_idx,
            tile_event_start_slot_muscle,
            tile_event_local_force_coeff,
            waypoint_attachment_body2,
            waypoint_attachment_local_weight0,
            waypoint_attachment_local_weight1,
            partial_wrench,
            env_idx,
            tile_idx,
            lane_idx,
        )

    kernel_name = (
        "accumulate_lbs_wrenches_full_force_"
        f"fixed{int(attachment_arity)}_"
        f"{specialization_suffix}_kernel"
    )
    _accumulate_lbs_body_wrench_native_tile_full_force_specialized_kernel.__name__ = (
        kernel_name
    )
    _accumulate_lbs_body_wrench_native_tile_full_force_specialized_kernel.__qualname__ = kernel_name
    return wp.kernel(enable_backward=False, module=_TILED_KERNEL_MODULE)(
        _accumulate_lbs_body_wrench_native_tile_full_force_specialized_kernel
    )


def _make_native_tile_full_force_velocity_kernel(
    native_source: str,
    *,
    attachment_arity: int,
    specialization_suffix: str,
):
    def native_accumulate_lbs_body_wrench_native_tile_full_force_velocity(
        body_pos: wp.array2d(dtype=wp.vec3),
        wrapped_points: wp.array2d(dtype=wp.vec3),
        point_source: wp.array(dtype=wp.int32),
        wrap_state: WaypointWrapKernelState,
        normalized_body_quat: wp.array2d(dtype=wp.quat),
        xipos: wp.array2d(dtype=wp.vec3),
        muscle_segment_offsets: wp.array(dtype=wp.int32),
        muscle_force_length_coeff: wp.array(dtype=wp.vec4),
        activation: wp.array2d(dtype=wp.float32),
        muscle_length_norm_min: float,
        muscle_length_norm_max: float,
        tile_info: wp.array(dtype=wp.vec4i),
        tile_extra_info: wp.array(dtype=wp.vec4i),
        tile_body_ids: wp.array(dtype=wp.int32),
        tile_body_output_idx: wp.array(dtype=wp.int32),
        tile_event_start_slot_muscle: wp.array(dtype=wp.int32),
        tile_event_local_force_coeff: wp.array(dtype=wp.vec4),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
        muscle_order: wp.array(dtype=wp.int32),
        body_cvel: wp.array2d(dtype=wp.spatial_vector),
        subtree_com: wp.array2d(dtype=wp.vec3),
        body_rootid: wp.array(dtype=wp.int32),
        velocity_scale: wp.array(dtype=wp.float32),
        fvmax: float,
        clamp_activation: int,
        partial_wrench: wp.array3d(dtype=wp.float32),
        env_idx: int,
        tile_idx: int,
        lane_idx: int,
    ): ...

    native_name = (
        "native_accumulate_lbs_wrenches_full_force_velocity_"
        f"fixed{int(attachment_arity)}_{specialization_suffix}"
    )
    native_accumulate_lbs_body_wrench_native_tile_full_force_velocity.__name__ = (
        native_name
    )
    native_accumulate_lbs_body_wrench_native_tile_full_force_velocity.__qualname__ = (
        native_name
    )
    native_specialized = wp.func_native(native_source)(
        native_accumulate_lbs_body_wrench_native_tile_full_force_velocity
    )

    def _accumulate_lbs_body_wrench_native_tile_full_force_velocity_kernel(
        body_pos: wp.array2d(dtype=wp.vec3),
        wrapped_points: wp.array2d(dtype=wp.vec3),
        point_source: wp.array(dtype=wp.int32),
        wrap_state: WaypointWrapKernelState,
        normalized_body_quat: wp.array2d(dtype=wp.quat),
        xipos: wp.array2d(dtype=wp.vec3),
        muscle_segment_offsets: wp.array(dtype=wp.int32),
        muscle_force_length_coeff: wp.array(dtype=wp.vec4),
        activation: wp.array2d(dtype=wp.float32),
        muscle_length_norm_min: float,
        muscle_length_norm_max: float,
        tile_info: wp.array(dtype=wp.vec4i),
        tile_extra_info: wp.array(dtype=wp.vec4i),
        tile_body_ids: wp.array(dtype=wp.int32),
        tile_body_output_idx: wp.array(dtype=wp.int32),
        tile_event_start_slot_muscle: wp.array(dtype=wp.int32),
        tile_event_local_force_coeff: wp.array(dtype=wp.vec4),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
        muscle_order: wp.array(dtype=wp.int32),
        body_cvel: wp.array2d(dtype=wp.spatial_vector),
        subtree_com: wp.array2d(dtype=wp.vec3),
        body_rootid: wp.array(dtype=wp.int32),
        velocity_scale: wp.array(dtype=wp.float32),
        fvmax: float,
        clamp_activation: int,
        partial_wrench: wp.array3d(dtype=wp.float32),
    ):
        env_idx, tile_idx, lane_idx = wp.tid()
        native_specialized(
            body_pos,
            wrapped_points,
            point_source,
            wrap_state,
            normalized_body_quat,
            xipos,
            muscle_segment_offsets,
            muscle_force_length_coeff,
            activation,
            muscle_length_norm_min,
            muscle_length_norm_max,
            tile_info,
            tile_extra_info,
            tile_body_ids,
            tile_body_output_idx,
            tile_event_start_slot_muscle,
            tile_event_local_force_coeff,
            waypoint_attachment_body2,
            waypoint_attachment_local_weight0,
            waypoint_attachment_local_weight1,
            muscle_order,
            body_cvel,
            subtree_com,
            body_rootid,
            velocity_scale,
            fvmax,
            clamp_activation,
            partial_wrench,
            env_idx,
            tile_idx,
            lane_idx,
        )

    kernel_name = (
        "accumulate_lbs_wrenches_full_force_velocity_"
        f"fixed{int(attachment_arity)}_{specialization_suffix}_kernel"
    )
    _accumulate_lbs_body_wrench_native_tile_full_force_velocity_kernel.__name__ = (
        kernel_name
    )
    _accumulate_lbs_body_wrench_native_tile_full_force_velocity_kernel.__qualname__ = (
        kernel_name
    )
    return wp.kernel(enable_backward=False, module=_TILED_KERNEL_MODULE)(
        _accumulate_lbs_body_wrench_native_tile_full_force_velocity_kernel
    )


def make_accumulate_lbs_body_wrench_native_tile_full_force_fixed1_kernel(
    include_passive_force: bool,
    passive_force_model_id: int = 0,
    passive_force_param0: float = 5.5,
    passive_force_param1: float = 0.3,
    muscle_cap: int = 32,
    waypoint_cap: int = 288,
    body_cap: int = 4,
    block_dim: int = 128,
    wrapped: bool = False,
    active_force_model: tuple[float, float, float] = (0.0, 0.45, 0.0),
):
    return _make_accumulate_lbs_body_wrench_native_tile_full_force_kernel(
        1,
        include_passive_force,
        passive_force_model_id,
        passive_force_param0,
        passive_force_param1,
        muscle_cap,
        waypoint_cap,
        body_cap,
        block_dim,
        wrapped,
        tuple(float(v) for v in active_force_model),
    )


def make_accumulate_lbs_body_wrench_native_tile_full_force_velocity_fixed1_kernel(
    include_passive_force: bool,
    passive_force_model_id: int = 0,
    passive_force_param0: float = 5.5,
    passive_force_param1: float = 0.3,
    muscle_cap: int = 32,
    waypoint_cap: int = 288,
    body_cap: int = 4,
    block_dim: int = 128,
    wrapped: bool = False,
    active_force_model: tuple[float, float, float] = (0.0, 0.45, 0.0),
):
    return _make_accumulate_lbs_body_wrench_native_tile_full_force_kernel(
        1,
        include_passive_force,
        passive_force_model_id,
        passive_force_param0,
        passive_force_param1,
        muscle_cap,
        waypoint_cap,
        body_cap,
        block_dim,
        wrapped,
        tuple(float(v) for v in active_force_model),
        True,
    )


def make_accumulate_lbs_body_wrench_native_tile_full_force_fixed2_kernel(
    include_passive_force: bool,
    passive_force_model_id: int = 0,
    passive_force_param0: float = 5.5,
    passive_force_param1: float = 0.3,
    muscle_cap: int = 32,
    waypoint_cap: int = 288,
    body_cap: int = 4,
    block_dim: int = 128,
    wrapped: bool = False,
    active_force_model: tuple[float, float, float] = (0.0, 0.45, 0.0),
):
    return _make_accumulate_lbs_body_wrench_native_tile_full_force_kernel(
        2,
        include_passive_force,
        passive_force_model_id,
        passive_force_param0,
        passive_force_param1,
        muscle_cap,
        waypoint_cap,
        body_cap,
        block_dim,
        wrapped,
        tuple(float(v) for v in active_force_model),
    )


@cache
def _make_accumulate_lbs_body_wrench_native_tile_cached_force_kernel(
    attachment_arity: int,
    muscle_cap: int = 32,
    waypoint_cap: int = 288,
    body_cap: int = 4,
    block_dim: int = 128,
    direct_force: bool = False,
    wrapped: bool = False,
):
    if int(attachment_arity) not in {1, 2}:
        raise ValueError("native tile attachment_arity must be 1 or 2")
    specialization_key = (
        int(attachment_arity),
        int(muscle_cap),
        int(waypoint_cap),
        int(body_cap),
        int(block_dim),
        bool(direct_force),
        bool(wrapped),
    )
    specialization_suffix = hashlib.sha1(
        repr(specialization_key).encode("utf-8")
    ).hexdigest()[:12]
    native_source = _specialize_native_tile_limits(
        _native_snippet_with_line_mapping(
            "lbs_body_wrench_native_tile_cached_force_fixed_arity.cuh"
        ),
        muscle_cap=muscle_cap,
        waypoint_cap=waypoint_cap,
        body_cap=body_cap,
        block_dim=block_dim,
    )
    native_source = native_source.replace(
        "__INSTANTMIMIC_ATTACHMENT_ARITY__",
        str(int(attachment_arity)),
    )
    native_source = native_source.replace(
        "__INSTANTMIMIC_DIRECT_FORCE__",
        "1" if direct_force else "0",
    )

    native_source = native_source.replace("__INSTANTMIMIC_WRAPPED__", str(int(wrapped)))

    def native_accumulate_lbs_body_wrench_native_tile_cached_force_specialized(
        body_pos: wp.array2d(dtype=wp.vec3),
        wrapped_points: wp.array2d(dtype=wp.vec3),
        point_source: wp.array(dtype=wp.int32),
        wrap_state: WaypointWrapKernelState,
        normalized_body_quat: wp.array2d(dtype=wp.quat),
        xipos: wp.array2d(dtype=wp.vec3),
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        passive_force_magnitude: wp.array2d(dtype=wp.float32),
        activation: wp.array2d(dtype=wp.float32),
        tile_info: wp.array(dtype=wp.vec4i),
        tile_extra_info: wp.array(dtype=wp.vec4i),
        tile_body_ids: wp.array(dtype=wp.int32),
        tile_body_output_idx: wp.array(dtype=wp.int32),
        tile_event_start_slot_muscle: wp.array(dtype=wp.int32),
        tile_event_local_force_coeff: wp.array(dtype=wp.vec4),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
        partial_wrench: wp.array3d(dtype=wp.float32),
        include_passive: int,
        env_idx: int,
        tile_idx: int,
        lane_idx: int,
    ): ...

    native_name = (
        "native_accumulate_lbs_wrenches_cached_force_"
        f"fixed{int(attachment_arity)}_"
        f"{specialization_suffix}"
    )
    native_accumulate_lbs_body_wrench_native_tile_cached_force_specialized.__name__ = (
        native_name
    )
    native_accumulate_lbs_body_wrench_native_tile_cached_force_specialized.__qualname__ = native_name
    native_specialized = wp.func_native(native_source)(
        native_accumulate_lbs_body_wrench_native_tile_cached_force_specialized
    )

    def _accumulate_lbs_body_wrench_native_tile_cached_force_specialized_kernel(
        body_pos: wp.array2d(dtype=wp.vec3),
        wrapped_points: wp.array2d(dtype=wp.vec3),
        point_source: wp.array(dtype=wp.int32),
        wrap_state: WaypointWrapKernelState,
        normalized_body_quat: wp.array2d(dtype=wp.quat),
        xipos: wp.array2d(dtype=wp.vec3),
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        passive_force_magnitude: wp.array2d(dtype=wp.float32),
        activation: wp.array2d(dtype=wp.float32),
        tile_info: wp.array(dtype=wp.vec4i),
        tile_extra_info: wp.array(dtype=wp.vec4i),
        tile_body_ids: wp.array(dtype=wp.int32),
        tile_body_output_idx: wp.array(dtype=wp.int32),
        tile_event_start_slot_muscle: wp.array(dtype=wp.int32),
        tile_event_local_force_coeff: wp.array(dtype=wp.vec4),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
        include_passive: int,
        partial_wrench: wp.array3d(dtype=wp.float32),
    ):
        env_idx, tile_idx, lane_idx = wp.tid()
        native_specialized(
            body_pos,
            wrapped_points,
            point_source,
            wrap_state,
            normalized_body_quat,
            xipos,
            unit_force_magnitude,
            passive_force_magnitude,
            activation,
            tile_info,
            tile_extra_info,
            tile_body_ids,
            tile_body_output_idx,
            tile_event_start_slot_muscle,
            tile_event_local_force_coeff,
            waypoint_attachment_body2,
            waypoint_attachment_local_weight0,
            waypoint_attachment_local_weight1,
            partial_wrench,
            include_passive,
            env_idx,
            tile_idx,
            lane_idx,
        )

    kernel_name = (
        "accumulate_lbs_wrenches_cached_force_"
        f"fixed{int(attachment_arity)}_"
        f"{specialization_suffix}_kernel"
    )
    _accumulate_lbs_body_wrench_native_tile_cached_force_specialized_kernel.__name__ = (
        kernel_name
    )
    _accumulate_lbs_body_wrench_native_tile_cached_force_specialized_kernel.__qualname__ = kernel_name
    return wp.kernel(enable_backward=False, module=_TILED_KERNEL_MODULE)(
        _accumulate_lbs_body_wrench_native_tile_cached_force_specialized_kernel
    )


def make_accumulate_lbs_body_wrench_native_tile_cached_force_fixed1_kernel(
    muscle_cap: int = 32,
    waypoint_cap: int = 288,
    body_cap: int = 4,
    block_dim: int = 128,
    direct_force: bool = False,
    wrapped: bool = False,
):
    return _make_accumulate_lbs_body_wrench_native_tile_cached_force_kernel(
        1,
        muscle_cap,
        waypoint_cap,
        body_cap,
        block_dim,
        direct_force,
        wrapped,
    )


def make_accumulate_lbs_body_wrench_native_tile_cached_force_fixed2_kernel(
    muscle_cap: int = 32,
    waypoint_cap: int = 288,
    body_cap: int = 4,
    block_dim: int = 128,
    direct_force: bool = False,
    wrapped: bool = False,
):
    return _make_accumulate_lbs_body_wrench_native_tile_cached_force_kernel(
        2,
        muscle_cap,
        waypoint_cap,
        body_cap,
        block_dim,
        direct_force,
        wrapped,
    )


@wp.kernel(enable_backward=False)
def _reduce_lbs_body_wrench_event_partials_kernel(
    partial_wrench: wp.array3d(dtype=wp.float32),
    body_event_chunk_offsets: wp.array(dtype=wp.int32),
    body_force: wp.array3d(dtype=wp.float32),
    body_torque: wp.array3d(dtype=wp.float32),
):
    env_idx, body_idx = wp.tid()
    chunk_begin = body_event_chunk_offsets[body_idx]
    chunk_end = body_event_chunk_offsets[body_idx + 1]
    fx = float(0.0)
    fy = float(0.0)
    fz = float(0.0)
    tx = float(0.0)
    ty = float(0.0)
    tz = float(0.0)
    for chunk_idx in range(chunk_begin, chunk_end):
        fx += partial_wrench[env_idx, chunk_idx, 0]
        fy += partial_wrench[env_idx, chunk_idx, 1]
        fz += partial_wrench[env_idx, chunk_idx, 2]
        tx += partial_wrench[env_idx, chunk_idx, 3]
        ty += partial_wrench[env_idx, chunk_idx, 4]
        tz += partial_wrench[env_idx, chunk_idx, 5]
    body_force[env_idx, body_idx, 0] = fx
    body_force[env_idx, body_idx, 1] = fy
    body_force[env_idx, body_idx, 2] = fz
    body_torque[env_idx, body_idx, 0] = tx
    body_torque[env_idx, body_idx, 1] = ty
    body_torque[env_idx, body_idx, 2] = tz


@wp.kernel(enable_backward=False)
def _project_body_wrench_to_torque_kernel(
    body_force: wp.array3d(dtype=wp.float32),
    body_torque: wp.array3d(dtype=wp.float32),
    xipos: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    body_rootid: wp.array(dtype=wp.int32),
    dof_subtree_mask: wp.array2d(dtype=wp.float32),
    body_count: int,
    root_dof_count: int,
    output_dof_offset: int,
    mean: wp.array2d(dtype=wp.float32),
):
    env_idx, torque_idx = wp.tid()
    full_torque_idx = torque_idx + root_dof_count
    linear_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 3],
        cdof[env_idx, full_torque_idx, 4],
        cdof[env_idx, full_torque_idx, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 0],
        cdof[env_idx, full_torque_idx, 1],
        cdof[env_idx, full_torque_idx, 2],
    )
    total = float(0.0)
    for body_idx in range(body_count):
        coeff = dof_subtree_mask[full_torque_idx, body_idx]
        if coeff == 0.0:
            continue
        force = wp.vec3(
            body_force[env_idx, body_idx, 0],
            body_force[env_idx, body_idx, 1],
            body_force[env_idx, body_idx, 2],
        )
        torque = wp.vec3(
            body_torque[env_idx, body_idx, 0],
            body_torque[env_idx, body_idx, 1],
            body_torque[env_idx, body_idx, 2],
        )
        root_idx = body_rootid[body_idx]
        torque_about_subtree = torque + wp.cross(
            xipos[env_idx, body_idx] - subtree_com[env_idx, root_idx], force
        )
        total += coeff * (
            wp.dot(linear_cdof, force) + wp.dot(angular_cdof, torque_about_subtree)
        )
    mean[env_idx, torque_idx + output_dof_offset] = total


@wp.kernel(enable_backward=False)
def _compute_lbs_body_adjoint_kernel(
    grad_output: wp.array2d(dtype=wp.float32),
    cdof: wp.array3d(dtype=wp.float32),
    dof_subtree_mask: wp.array2d(dtype=wp.float32),
    body_count: int,
    torque_dim: int,
    root_dof_count: int,
    body_linear_adjoint: wp.array3d(dtype=wp.float32),
    body_angular_adjoint: wp.array3d(dtype=wp.float32),
):
    env_idx, body_idx = wp.tid()
    linear = wp.vec3(0.0, 0.0, 0.0)
    angular = wp.vec3(0.0, 0.0, 0.0)
    for torque_idx in range(torque_dim):
        full_torque_idx = torque_idx + root_dof_count
        scale = (
            grad_output[env_idx, torque_idx]
            * dof_subtree_mask[full_torque_idx, body_idx]
        )
        linear += scale * wp.vec3(
            cdof[env_idx, full_torque_idx, 3],
            cdof[env_idx, full_torque_idx, 4],
            cdof[env_idx, full_torque_idx, 5],
        )
        angular += scale * wp.vec3(
            cdof[env_idx, full_torque_idx, 0],
            cdof[env_idx, full_torque_idx, 1],
            cdof[env_idx, full_torque_idx, 2],
        )
    body_linear_adjoint[env_idx, body_idx, 0] = linear[0]
    body_linear_adjoint[env_idx, body_idx, 1] = linear[1]
    body_linear_adjoint[env_idx, body_idx, 2] = linear[2]
    body_angular_adjoint[env_idx, body_idx, 0] = angular[0]
    body_angular_adjoint[env_idx, body_idx, 1] = angular[1]
    body_angular_adjoint[env_idx, body_idx, 2] = angular[2]


@wp.kernel(enable_backward=False)
def _accumulate_lbs_activation_grad_from_body_adjoint_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    body_rootid: wp.array(dtype=wp.int32),
    body_linear_adjoint: wp.array3d(dtype=wp.float32),
    body_angular_adjoint: wp.array3d(dtype=wp.float32),
    apply_segment_muscle_idx: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    waypoint_attachment_start: wp.array(dtype=wp.int32),
    waypoint_attachment_body: wp.array(dtype=wp.int32),
    waypoint_attachment_local: wp.array(dtype=wp.vec3),
    waypoint_attachment_weight: wp.array(dtype=wp.float32),
    grad_activation: wp.array2d(dtype=wp.float32),
):
    env_idx, segment_idx = wp.tid()
    muscle_idx = apply_segment_muscle_idx[segment_idx]
    unit_force = unit_force_magnitude[env_idx, muscle_idx]
    if unit_force == 0.0:
        return

    start_wp = apply_segment_start_idx[segment_idx]
    end_wp = apply_segment_end_idx[segment_idx]
    p_start = transformed_waypoints[env_idx, start_wp]
    p_end = transformed_waypoints[env_idx, end_wp]
    segment = p_end - p_start
    segment_length = wp.length(segment)
    if segment_length < 1.0e-8:
        return
    seg_force = unit_force * segment / segment_length
    total = float(0.0)

    start_begin = waypoint_attachment_start[start_wp]
    start_end = waypoint_attachment_start[start_wp + 1]
    for attach_idx in range(start_begin, start_end):
        body_idx = waypoint_attachment_body[attach_idx]
        q = _normalize_quat_wp(body_quat[env_idx, body_idx])
        point = body_pos[env_idx, body_idx] + rot_vec_quat(
            waypoint_attachment_local[attach_idx], q
        )
        root_idx = body_rootid[body_idx]
        linear = wp.vec3(
            body_linear_adjoint[env_idx, body_idx, 0],
            body_linear_adjoint[env_idx, body_idx, 1],
            body_linear_adjoint[env_idx, body_idx, 2],
        )
        angular = wp.vec3(
            body_angular_adjoint[env_idx, body_idx, 0],
            body_angular_adjoint[env_idx, body_idx, 1],
            body_angular_adjoint[env_idx, body_idx, 2],
        )
        term = linear + wp.cross(angular, point - subtree_com[env_idx, root_idx])
        total += waypoint_attachment_weight[attach_idx] * wp.dot(term, seg_force)

    end_begin = waypoint_attachment_start[end_wp]
    end_end = waypoint_attachment_start[end_wp + 1]
    for attach_idx in range(end_begin, end_end):
        body_idx = waypoint_attachment_body[attach_idx]
        q = _normalize_quat_wp(body_quat[env_idx, body_idx])
        point = body_pos[env_idx, body_idx] + rot_vec_quat(
            waypoint_attachment_local[attach_idx], q
        )
        root_idx = body_rootid[body_idx]
        linear = wp.vec3(
            body_linear_adjoint[env_idx, body_idx, 0],
            body_linear_adjoint[env_idx, body_idx, 1],
            body_linear_adjoint[env_idx, body_idx, 2],
        )
        angular = wp.vec3(
            body_angular_adjoint[env_idx, body_idx, 0],
            body_angular_adjoint[env_idx, body_idx, 1],
            body_angular_adjoint[env_idx, body_idx, 2],
        )
        term = linear + wp.cross(angular, point - subtree_com[env_idx, root_idx])
        total -= waypoint_attachment_weight[attach_idx] * wp.dot(term, seg_force)

    wp.atomic_add(grad_activation, env_idx, muscle_idx, total)


@wp.kernel(enable_backward=False)
def _accumulate_activation_grad_direct_from_body_adjoint_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    body_rootid: wp.array(dtype=wp.int32),
    body_linear_adjoint: wp.array3d(dtype=wp.float32),
    body_angular_adjoint: wp.array3d(dtype=wp.float32),
    apply_segment_muscle_idx: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    apply_segment_body_start_idx: wp.array(dtype=wp.int32),
    apply_segment_body_end_idx: wp.array(dtype=wp.int32),
    grad_activation: wp.array2d(dtype=wp.float32),
):
    env_idx, segment_idx = wp.tid()
    muscle_idx = apply_segment_muscle_idx[segment_idx]
    unit_force = unit_force_magnitude[env_idx, muscle_idx]
    if unit_force == 0.0:
        return

    start_wp = apply_segment_start_idx[segment_idx]
    end_wp = apply_segment_end_idx[segment_idx]
    p_start = transformed_waypoints[env_idx, start_wp]
    p_end = transformed_waypoints[env_idx, end_wp]
    segment = p_end - p_start
    segment_length = wp.length(segment)
    if segment_length < 1.0e-8:
        return
    seg_force = unit_force * segment / segment_length

    start_body = apply_segment_body_start_idx[segment_idx]
    start_linear = wp.vec3(
        body_linear_adjoint[env_idx, start_body, 0],
        body_linear_adjoint[env_idx, start_body, 1],
        body_linear_adjoint[env_idx, start_body, 2],
    )
    start_angular = wp.vec3(
        body_angular_adjoint[env_idx, start_body, 0],
        body_angular_adjoint[env_idx, start_body, 1],
        body_angular_adjoint[env_idx, start_body, 2],
    )
    start_term = start_linear + wp.cross(
        start_angular,
        p_start - subtree_com[env_idx, body_rootid[start_body]],
    )

    end_body = apply_segment_body_end_idx[segment_idx]
    end_linear = wp.vec3(
        body_linear_adjoint[env_idx, end_body, 0],
        body_linear_adjoint[env_idx, end_body, 1],
        body_linear_adjoint[env_idx, end_body, 2],
    )
    end_angular = wp.vec3(
        body_angular_adjoint[env_idx, end_body, 0],
        body_angular_adjoint[env_idx, end_body, 1],
        body_angular_adjoint[env_idx, end_body, 2],
    )
    end_term = end_linear + wp.cross(
        end_angular,
        p_end - subtree_com[env_idx, body_rootid[end_body]],
    )

    total = wp.dot(start_term - end_term, seg_force)
    wp.atomic_add(grad_activation, env_idx, muscle_idx, total)


@wp.func
def _lbs_activation_grad_fixed2_attachment_term_wp(
    body_idx: int,
    local_weight: wp.vec4,
    coeff_sign: float,
    env_idx: int,
    root_body_idx: int,
    seg_force: wp.vec3,
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    body_linear_adjoint: wp.array3d(dtype=wp.float32),
    body_angular_adjoint: wp.array3d(dtype=wp.float32),
) -> float:
    q = _normalize_quat_wp(body_quat[env_idx, body_idx])
    local = wp.vec3(local_weight[0], local_weight[1], local_weight[2])
    point = body_pos[env_idx, body_idx] + rot_vec_quat(local, q)
    linear = wp.vec3(
        body_linear_adjoint[env_idx, body_idx, 0],
        body_linear_adjoint[env_idx, body_idx, 1],
        body_linear_adjoint[env_idx, body_idx, 2],
    )
    angular = wp.vec3(
        body_angular_adjoint[env_idx, body_idx, 0],
        body_angular_adjoint[env_idx, body_idx, 1],
        body_angular_adjoint[env_idx, body_idx, 2],
    )
    term = linear + wp.cross(angular, point - subtree_com[env_idx, root_body_idx])
    return coeff_sign * local_weight[3] * wp.dot(term, seg_force)


@wp.func
def _lbs_activation_grad_fixed2_attachment_point_term_wp(
    body_idx: int,
    point: wp.vec3,
    weight: float,
    coeff_sign: float,
    env_idx: int,
    root_body_idx: int,
    seg_force: wp.vec3,
    subtree_com: wp.array2d(dtype=wp.vec3),
    body_linear_adjoint: wp.array3d(dtype=wp.float32),
    body_angular_adjoint: wp.array3d(dtype=wp.float32),
) -> float:
    linear = wp.vec3(
        body_linear_adjoint[env_idx, body_idx, 0],
        body_linear_adjoint[env_idx, body_idx, 1],
        body_linear_adjoint[env_idx, body_idx, 2],
    )
    angular = wp.vec3(
        body_angular_adjoint[env_idx, body_idx, 0],
        body_angular_adjoint[env_idx, body_idx, 1],
        body_angular_adjoint[env_idx, body_idx, 2],
    )
    term = linear + wp.cross(angular, point - subtree_com[env_idx, root_body_idx])
    return coeff_sign * weight * wp.dot(term, seg_force)


@wp.func
def _lbs_activation_grad_fixed2_attachment_point_term_root_wp(
    body_idx: int,
    point: wp.vec3,
    weight: float,
    coeff_sign: float,
    env_idx: int,
    body_rootid: wp.array(dtype=wp.int32),
    seg_force: wp.vec3,
    subtree_com: wp.array2d(dtype=wp.vec3),
    body_linear_adjoint: wp.array3d(dtype=wp.float32),
    body_angular_adjoint: wp.array3d(dtype=wp.float32),
) -> float:
    linear = wp.vec3(
        body_linear_adjoint[env_idx, body_idx, 0],
        body_linear_adjoint[env_idx, body_idx, 1],
        body_linear_adjoint[env_idx, body_idx, 2],
    )
    angular = wp.vec3(
        body_angular_adjoint[env_idx, body_idx, 0],
        body_angular_adjoint[env_idx, body_idx, 1],
        body_angular_adjoint[env_idx, body_idx, 2],
    )
    root_body_idx = body_rootid[body_idx]
    term = linear + wp.cross(angular, point - subtree_com[env_idx, root_body_idx])
    return coeff_sign * weight * wp.dot(term, seg_force)


@cache
def make_accumulate_lbs_activation_grad_fixed1_direct_start_from_body_adjoint_kernel(
    single_root_body_idx: int,
    use_segment_start_formula: bool = False,
):
    root_body_idx = int(single_root_body_idx)
    use_segment_start_formula_flag = 1 if bool(use_segment_start_formula) else 0

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def accumulate_lbs_activation_grad_from_body_adjoint_kernel(
        body_pos: wp.array2d(dtype=wp.vec3),
        normalized_body_quat: wp.array2d(dtype=wp.quat),
        subtree_com: wp.array2d(dtype=wp.vec3),
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        body_linear_adjoint: wp.array3d(dtype=wp.float32),
        body_angular_adjoint: wp.array3d(dtype=wp.float32),
        apply_segment_muscle_idx: wp.array(dtype=wp.int32),
        apply_segment_start_idx: wp.array(dtype=wp.int32),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        grad_activation: wp.array2d(dtype=wp.float32),
    ):
        env_idx, segment_idx = wp.tid()
        muscle_idx = apply_segment_muscle_idx[segment_idx]
        unit_force = unit_force_magnitude[env_idx, muscle_idx]

        start_wp = apply_segment_start_idx[segment_idx]
        if wp.static(use_segment_start_formula_flag) != 0:
            start_wp = segment_idx + muscle_idx
        end_wp = start_wp + 1
        start_body = native_load_vec2i_wp(
            waypoint_attachment_body2,
            start_wp,
        )[0]
        end_body = native_load_vec2i_wp(
            waypoint_attachment_body2,
            end_wp,
        )[0]
        start_local_weight = native_load_vec4_wp(
            waypoint_attachment_local_weight0,
            start_wp,
        )
        end_local_weight = native_load_vec4_wp(
            waypoint_attachment_local_weight0,
            end_wp,
        )
        start_point = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            start_body,
            start_local_weight,
            body_pos,
            normalized_body_quat,
        )
        end_point = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            end_body,
            end_local_weight,
            body_pos,
            normalized_body_quat,
        )
        p_start = start_local_weight[3] * start_point
        p_end = end_local_weight[3] * end_point
        segment = p_end - p_start
        inv_segment_length = 1.0 / wp.sqrt(wp.max(wp.dot(segment, segment), 1.0e-16))
        seg_force = unit_force * segment * inv_segment_length

        total = _lbs_activation_grad_fixed2_attachment_point_term_wp(
            start_body,
            start_point,
            start_local_weight[3],
            1.0,
            env_idx,
            wp.static(root_body_idx),
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_point_term_wp(
            end_body,
            end_point,
            end_local_weight[3],
            -1.0,
            env_idx,
            wp.static(root_body_idx),
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        wp.atomic_add(grad_activation, env_idx, muscle_idx, total)

    return accumulate_lbs_activation_grad_from_body_adjoint_kernel


@cache
def make_accumulate_lbs_activation_grad_fixed1_direct_start_from_body_adjoint_multi_root_kernel(
    use_segment_start_formula: bool = False,
):
    use_segment_start_formula_flag = 1 if bool(use_segment_start_formula) else 0

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _accumulate_lbs_activation_grad_fixed1_direct_start_from_body_adjoint_multi_root_kernel(
        body_pos: wp.array2d(dtype=wp.vec3),
        normalized_body_quat: wp.array2d(dtype=wp.quat),
        subtree_com: wp.array2d(dtype=wp.vec3),
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        body_rootid: wp.array(dtype=wp.int32),
        body_linear_adjoint: wp.array3d(dtype=wp.float32),
        body_angular_adjoint: wp.array3d(dtype=wp.float32),
        apply_segment_muscle_idx: wp.array(dtype=wp.int32),
        apply_segment_start_idx: wp.array(dtype=wp.int32),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        grad_activation: wp.array2d(dtype=wp.float32),
    ):
        env_idx, segment_idx = wp.tid()
        muscle_idx = apply_segment_muscle_idx[segment_idx]
        unit_force = unit_force_magnitude[env_idx, muscle_idx]

        start_wp = apply_segment_start_idx[segment_idx]
        if wp.static(use_segment_start_formula_flag) != 0:
            start_wp = segment_idx + muscle_idx
        end_wp = start_wp + 1
        start_body = native_load_vec2i_wp(
            waypoint_attachment_body2,
            start_wp,
        )[0]
        end_body = native_load_vec2i_wp(
            waypoint_attachment_body2,
            end_wp,
        )[0]
        start_local_weight = native_load_vec4_wp(
            waypoint_attachment_local_weight0,
            start_wp,
        )
        end_local_weight = native_load_vec4_wp(
            waypoint_attachment_local_weight0,
            end_wp,
        )
        start_point = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            start_body,
            start_local_weight,
            body_pos,
            normalized_body_quat,
        )
        end_point = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            end_body,
            end_local_weight,
            body_pos,
            normalized_body_quat,
        )
        p_start = start_local_weight[3] * start_point
        p_end = end_local_weight[3] * end_point
        segment = p_end - p_start
        inv_segment_length = 1.0 / wp.sqrt(wp.max(wp.dot(segment, segment), 1.0e-16))
        seg_force = unit_force * segment * inv_segment_length

        total = _lbs_activation_grad_fixed2_attachment_point_term_root_wp(
            start_body,
            start_point,
            start_local_weight[3],
            1.0,
            env_idx,
            body_rootid,
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_point_term_root_wp(
            end_body,
            end_point,
            end_local_weight[3],
            -1.0,
            env_idx,
            body_rootid,
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        wp.atomic_add(grad_activation, env_idx, muscle_idx, total)

    return _accumulate_lbs_activation_grad_fixed1_direct_start_from_body_adjoint_multi_root_kernel


@cache
def make_accumulate_lbs_activation_grad_fixed2_start_from_body_adjoint_kernel(
    single_root_body_idx: int,
    use_segment_start_formula: bool = False,
):
    root_body_idx = int(single_root_body_idx)
    use_segment_start_formula_flag = 1 if bool(use_segment_start_formula) else 0

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _accumulate_lbs_activation_grad_fixed2_start_from_body_adjoint_kernel(
        transformed_waypoints: wp.array2d(dtype=wp.vec3),
        body_pos: wp.array2d(dtype=wp.vec3),
        body_quat: wp.array2d(dtype=wp.quat),
        subtree_com: wp.array2d(dtype=wp.vec3),
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        body_linear_adjoint: wp.array3d(dtype=wp.float32),
        body_angular_adjoint: wp.array3d(dtype=wp.float32),
        apply_segment_muscle_idx: wp.array(dtype=wp.int32),
        apply_segment_start_idx: wp.array(dtype=wp.int32),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
        grad_activation: wp.array2d(dtype=wp.float32),
    ):
        env_idx, segment_idx = wp.tid()
        muscle_idx = apply_segment_muscle_idx[segment_idx]
        unit_force = unit_force_magnitude[env_idx, muscle_idx]

        start_wp = apply_segment_start_idx[segment_idx]
        if wp.static(use_segment_start_formula_flag) != 0:
            start_wp = segment_idx + muscle_idx
        p_start = transformed_waypoints[env_idx, start_wp]
        p_end = transformed_waypoints[env_idx, start_wp + 1]
        segment = p_end - p_start
        segment_length_sq = wp.dot(segment, segment)
        inv_segment_length = 1.0 / wp.sqrt(wp.max(segment_length_sq, 1.0e-16))
        seg_force = unit_force * segment * inv_segment_length

        start_body2 = native_load_vec2i_wp(waypoint_attachment_body2, start_wp)
        start_local_weight0 = native_load_vec4_wp(
            waypoint_attachment_local_weight0,
            start_wp,
        )
        start_local_weight1 = native_load_vec4_wp(
            waypoint_attachment_local_weight1,
            start_wp,
        )
        end_wp = start_wp + 1
        end_body2 = native_load_vec2i_wp(waypoint_attachment_body2, end_wp)
        end_local_weight0 = native_load_vec4_wp(
            waypoint_attachment_local_weight0,
            end_wp,
        )
        end_local_weight1 = native_load_vec4_wp(
            waypoint_attachment_local_weight1,
            end_wp,
        )

        total = _lbs_activation_grad_fixed2_attachment_term_wp(
            start_body2[0],
            start_local_weight0,
            1.0,
            env_idx,
            wp.static(root_body_idx),
            seg_force,
            body_pos,
            body_quat,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_term_wp(
            start_body2[1],
            start_local_weight1,
            1.0,
            env_idx,
            wp.static(root_body_idx),
            seg_force,
            body_pos,
            body_quat,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_term_wp(
            end_body2[0],
            end_local_weight0,
            -1.0,
            env_idx,
            wp.static(root_body_idx),
            seg_force,
            body_pos,
            body_quat,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_term_wp(
            end_body2[1],
            end_local_weight1,
            -1.0,
            env_idx,
            wp.static(root_body_idx),
            seg_force,
            body_pos,
            body_quat,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )

        wp.atomic_add(grad_activation, env_idx, muscle_idx, total)

    return _accumulate_lbs_activation_grad_fixed2_start_from_body_adjoint_kernel


@cache
def make_accumulate_lbs_activation_grad_fixed2_direct_start_from_body_adjoint_kernel(
    single_root_body_idx: int,
    use_segment_start_formula: bool = False,
):
    root_body_idx = int(single_root_body_idx)
    use_segment_start_formula_flag = 1 if bool(use_segment_start_formula) else 0

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _accumulate_lbs_activation_grad_fixed2_direct_start_from_body_adjoint_kernel(
        body_pos: wp.array2d(dtype=wp.vec3),
        normalized_body_quat: wp.array2d(dtype=wp.quat),
        subtree_com: wp.array2d(dtype=wp.vec3),
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        body_linear_adjoint: wp.array3d(dtype=wp.float32),
        body_angular_adjoint: wp.array3d(dtype=wp.float32),
        apply_segment_muscle_idx: wp.array(dtype=wp.int32),
        apply_segment_start_idx: wp.array(dtype=wp.int32),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
        grad_activation: wp.array2d(dtype=wp.float32),
    ):
        env_idx, segment_idx = wp.tid()
        muscle_idx = apply_segment_muscle_idx[segment_idx]
        unit_force = unit_force_magnitude[env_idx, muscle_idx]

        start_wp = apply_segment_start_idx[segment_idx]
        if wp.static(use_segment_start_formula_flag) != 0:
            start_wp = segment_idx + muscle_idx
        start_body2 = native_load_vec2i_wp(waypoint_attachment_body2, start_wp)
        start_local_weight0 = native_load_vec4_wp(
            waypoint_attachment_local_weight0,
            start_wp,
        )
        start_local_weight1 = native_load_vec4_wp(
            waypoint_attachment_local_weight1,
            start_wp,
        )
        end_wp = start_wp + 1
        end_body2 = native_load_vec2i_wp(waypoint_attachment_body2, end_wp)
        end_local_weight0 = native_load_vec4_wp(
            waypoint_attachment_local_weight0,
            end_wp,
        )
        end_local_weight1 = native_load_vec4_wp(
            waypoint_attachment_local_weight1,
            end_wp,
        )

        start_point0 = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            start_body2[0],
            start_local_weight0,
            body_pos,
            normalized_body_quat,
        )
        start_point1 = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            start_body2[1],
            start_local_weight1,
            body_pos,
            normalized_body_quat,
        )
        end_point0 = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            end_body2[0],
            end_local_weight0,
            body_pos,
            normalized_body_quat,
        )
        end_point1 = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            end_body2[1],
            end_local_weight1,
            body_pos,
            normalized_body_quat,
        )
        p_start = (
            start_local_weight0[3] * start_point0
            + start_local_weight1[3] * start_point1
        )
        p_end = end_local_weight0[3] * end_point0 + end_local_weight1[3] * end_point1
        segment = p_end - p_start
        segment_length_sq = wp.dot(segment, segment)
        inv_segment_length = 1.0 / wp.sqrt(wp.max(segment_length_sq, 1.0e-16))
        seg_force = unit_force * segment * inv_segment_length

        total = _lbs_activation_grad_fixed2_attachment_point_term_wp(
            start_body2[0],
            start_point0,
            start_local_weight0[3],
            1.0,
            env_idx,
            wp.static(root_body_idx),
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_point_term_wp(
            start_body2[1],
            start_point1,
            start_local_weight1[3],
            1.0,
            env_idx,
            wp.static(root_body_idx),
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_point_term_wp(
            end_body2[0],
            end_point0,
            end_local_weight0[3],
            -1.0,
            env_idx,
            wp.static(root_body_idx),
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_point_term_wp(
            end_body2[1],
            end_point1,
            end_local_weight1[3],
            -1.0,
            env_idx,
            wp.static(root_body_idx),
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )

        wp.atomic_add(grad_activation, env_idx, muscle_idx, total)

    return _accumulate_lbs_activation_grad_fixed2_direct_start_from_body_adjoint_kernel


@cache
def make_accumulate_lbs_activation_grad_fixed2_direct_start_from_body_adjoint_multi_root_kernel(
    use_segment_start_formula: bool = False,
):
    use_segment_start_formula_flag = 1 if bool(use_segment_start_formula) else 0

    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def _accumulate_lbs_activation_grad_fixed2_direct_start_from_body_adjoint_multi_root_kernel(
        body_pos: wp.array2d(dtype=wp.vec3),
        normalized_body_quat: wp.array2d(dtype=wp.quat),
        subtree_com: wp.array2d(dtype=wp.vec3),
        unit_force_magnitude: wp.array2d(dtype=wp.float32),
        body_rootid: wp.array(dtype=wp.int32),
        body_linear_adjoint: wp.array3d(dtype=wp.float32),
        body_angular_adjoint: wp.array3d(dtype=wp.float32),
        apply_segment_muscle_idx: wp.array(dtype=wp.int32),
        apply_segment_start_idx: wp.array(dtype=wp.int32),
        waypoint_attachment_body2: wp.array(dtype=wp.vec2i),
        waypoint_attachment_local_weight0: wp.array(dtype=wp.vec4),
        waypoint_attachment_local_weight1: wp.array(dtype=wp.vec4),
        grad_activation: wp.array2d(dtype=wp.float32),
    ):
        env_idx, segment_idx = wp.tid()
        muscle_idx = apply_segment_muscle_idx[segment_idx]
        unit_force = unit_force_magnitude[env_idx, muscle_idx]

        start_wp = apply_segment_start_idx[segment_idx]
        if wp.static(use_segment_start_formula_flag) != 0:
            start_wp = segment_idx + muscle_idx
        start_body2 = native_load_vec2i_wp(waypoint_attachment_body2, start_wp)
        start_local_weight0 = native_load_vec4_wp(
            waypoint_attachment_local_weight0,
            start_wp,
        )
        start_local_weight1 = native_load_vec4_wp(
            waypoint_attachment_local_weight1,
            start_wp,
        )
        end_wp = start_wp + 1
        end_body2 = native_load_vec2i_wp(waypoint_attachment_body2, end_wp)
        end_local_weight0 = native_load_vec4_wp(
            waypoint_attachment_local_weight0,
            end_wp,
        )
        end_local_weight1 = native_load_vec4_wp(
            waypoint_attachment_local_weight1,
            end_wp,
        )

        start_point0 = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            start_body2[0],
            start_local_weight0,
            body_pos,
            normalized_body_quat,
        )
        start_point1 = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            start_body2[1],
            start_local_weight1,
            body_pos,
            normalized_body_quat,
        )
        end_point0 = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            end_body2[0],
            end_local_weight0,
            body_pos,
            normalized_body_quat,
        )
        end_point1 = _lbs_fixed2_attachment_point_normalized_wp(
            env_idx,
            end_body2[1],
            end_local_weight1,
            body_pos,
            normalized_body_quat,
        )
        p_start = (
            start_local_weight0[3] * start_point0
            + start_local_weight1[3] * start_point1
        )
        p_end = end_local_weight0[3] * end_point0 + end_local_weight1[3] * end_point1
        segment = p_end - p_start
        segment_length_sq = wp.dot(segment, segment)
        inv_segment_length = 1.0 / wp.sqrt(wp.max(segment_length_sq, 1.0e-16))
        seg_force = unit_force * segment * inv_segment_length

        total = _lbs_activation_grad_fixed2_attachment_point_term_root_wp(
            start_body2[0],
            start_point0,
            start_local_weight0[3],
            1.0,
            env_idx,
            body_rootid,
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_point_term_root_wp(
            start_body2[1],
            start_point1,
            start_local_weight1[3],
            1.0,
            env_idx,
            body_rootid,
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_point_term_root_wp(
            end_body2[0],
            end_point0,
            end_local_weight0[3],
            -1.0,
            env_idx,
            body_rootid,
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )
        total += _lbs_activation_grad_fixed2_attachment_point_term_root_wp(
            end_body2[1],
            end_point1,
            end_local_weight1[3],
            -1.0,
            env_idx,
            body_rootid,
            seg_force,
            subtree_com,
            body_linear_adjoint,
            body_angular_adjoint,
        )

        wp.atomic_add(grad_activation, env_idx, muscle_idx, total)

    return _accumulate_lbs_activation_grad_fixed2_direct_start_from_body_adjoint_multi_root_kernel


@wp.kernel(enable_backward=False)
def _compute_muscle_dynamics_contribution_lbs_sparse_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    body_rootid: wp.array(dtype=wp.int32),
    dof_subtree_mask: wp.array2d(dtype=wp.float32),
    torque_active_torque_idx: wp.array(dtype=wp.int32),
    torque_active_segment_idx: wp.array(dtype=wp.int32),
    torque_active_muscle_idx: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    waypoint_attachment_start: wp.array(dtype=wp.int32),
    waypoint_attachment_body: wp.array(dtype=wp.int32),
    waypoint_attachment_local: wp.array(dtype=wp.vec3),
    waypoint_attachment_weight: wp.array(dtype=wp.float32),
    active_attach_offsets: wp.array(dtype=wp.int32),
    active_attach_body: wp.array(dtype=wp.int32),
    active_attach_local: wp.array(dtype=wp.vec3),
    active_attach_coeff_weight: wp.array(dtype=wp.float32),
    waypoint_count: int,
    root_dof_count: int,
    contribution: wp.array3d(dtype=wp.float32),
):
    env_idx, entry_idx = wp.tid()
    torque_idx = torque_active_torque_idx[entry_idx]
    muscle_idx = torque_active_muscle_idx[entry_idx]
    segment_idx = torque_active_segment_idx[entry_idx]
    full_torque_idx = torque_idx + root_dof_count
    linear_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 3],
        cdof[env_idx, full_torque_idx, 4],
        cdof[env_idx, full_torque_idx, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 0],
        cdof[env_idx, full_torque_idx, 1],
        cdof[env_idx, full_torque_idx, 2],
    )
    total = _lbs_segment_entry_contribution_active_wp(
        env_idx,
        torque_idx,
        segment_idx,
        unit_force_magnitude[env_idx, muscle_idx],
        linear_cdof,
        angular_cdof,
        transformed_waypoints,
        body_pos,
        body_quat,
        subtree_com,
        body_rootid,
        apply_segment_start_idx,
        apply_segment_end_idx,
        active_attach_offsets,
        active_attach_body,
        active_attach_local,
        active_attach_coeff_weight,
        waypoint_count,
    )
    wp.atomic_add(contribution, env_idx, torque_idx, muscle_idx, total)


@wp.kernel(enable_backward=False)
def _compute_muscle_dynamics_lbs_sparse_bounds_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    body_rootid: wp.array(dtype=wp.int32),
    dof_subtree_mask: wp.array2d(dtype=wp.float32),
    torque_active_entry_offsets: wp.array(dtype=wp.int32),
    torque_active_segment_idx: wp.array(dtype=wp.int32),
    torque_active_muscle_idx: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    waypoint_attachment_start: wp.array(dtype=wp.int32),
    waypoint_attachment_body: wp.array(dtype=wp.int32),
    waypoint_attachment_local: wp.array(dtype=wp.vec3),
    waypoint_attachment_weight: wp.array(dtype=wp.float32),
    active_attach_offsets: wp.array(dtype=wp.int32),
    active_attach_body: wp.array(dtype=wp.int32),
    active_attach_local: wp.array(dtype=wp.vec3),
    active_attach_coeff_weight: wp.array(dtype=wp.float32),
    waypoint_count: int,
    root_dof_count: int,
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
):
    env_idx, torque_idx = wp.tid()
    entry_begin = torque_active_entry_offsets[torque_idx]
    entry_end = torque_active_entry_offsets[torque_idx + 1]
    current_muscle = int(-1)
    current_total = float(0.0)
    lower_total = float(0.0)
    upper_total = float(0.0)
    full_torque_idx = torque_idx + root_dof_count
    linear_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 3],
        cdof[env_idx, full_torque_idx, 4],
        cdof[env_idx, full_torque_idx, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 0],
        cdof[env_idx, full_torque_idx, 1],
        cdof[env_idx, full_torque_idx, 2],
    )

    for entry_idx in range(entry_begin, entry_end):
        muscle_idx = torque_active_muscle_idx[entry_idx]
        if current_muscle >= 0 and muscle_idx != current_muscle:
            if current_total < 0.0:
                lower_total += current_total
            else:
                upper_total += current_total
            current_total = 0.0
        current_muscle = muscle_idx
        segment_idx = torque_active_segment_idx[entry_idx]
        current_total += _lbs_segment_entry_contribution_active_wp(
            env_idx,
            torque_idx,
            segment_idx,
            unit_force_magnitude[env_idx, muscle_idx],
            linear_cdof,
            angular_cdof,
            transformed_waypoints,
            body_pos,
            body_quat,
            subtree_com,
            body_rootid,
            apply_segment_start_idx,
            apply_segment_end_idx,
            active_attach_offsets,
            active_attach_body,
            active_attach_local,
            active_attach_coeff_weight,
            waypoint_count,
        )

    if current_muscle >= 0:
        if current_total < 0.0:
            lower_total += current_total
        else:
            upper_total += current_total

    lower[env_idx, torque_idx] = lower_total
    upper[env_idx, torque_idx] = upper_total


@wp.kernel(enable_backward=False, module=_TILED_KERNEL_MODULE)
def _compute_lbs_torque_bounds_tile_partials_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    body_rootid: wp.array(dtype=wp.int32),
    torque_chunk_torque_idx: wp.array(dtype=wp.int32),
    torque_chunk_group_start: wp.array(dtype=wp.int32),
    torque_chunk_group_end: wp.array(dtype=wp.int32),
    group_entry_start: wp.array(dtype=wp.int32),
    group_entry_end: wp.array(dtype=wp.int32),
    torque_active_segment_idx: wp.array(dtype=wp.int32),
    torque_active_muscle_idx: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    active_attach_offsets: wp.array(dtype=wp.int32),
    active_attach_body: wp.array(dtype=wp.int32),
    active_attach_local: wp.array(dtype=wp.vec3),
    active_attach_coeff_weight: wp.array(dtype=wp.float32),
    waypoint_count: int,
    root_dof_count: int,
    partial_lower: wp.array2d(dtype=wp.float32),
    partial_upper: wp.array2d(dtype=wp.float32),
):
    env_idx, chunk_idx, lane_idx = wp.tid()
    torque_idx = torque_chunk_torque_idx[chunk_idx]
    group_idx = torque_chunk_group_start[chunk_idx] + lane_idx
    group_end = torque_chunk_group_end[chunk_idx]
    full_torque_idx = torque_idx + root_dof_count
    linear_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 3],
        cdof[env_idx, full_torque_idx, 4],
        cdof[env_idx, full_torque_idx, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_torque_idx, 0],
        cdof[env_idx, full_torque_idx, 1],
        cdof[env_idx, full_torque_idx, 2],
    )
    lower_total = float(0.0)
    upper_total = float(0.0)
    while group_idx < group_end:
        entry_begin = group_entry_start[group_idx]
        entry_end = group_entry_end[group_idx]
        muscle_idx = torque_active_muscle_idx[entry_begin]
        group_total = float(0.0)
        previous_end_wp = int(-1)
        previous_p_end = wp.vec3(0.0, 0.0, 0.0)
        previous_end_begin = int(0)
        previous_end_end = int(0)
        for entry_idx in range(entry_begin, entry_end):
            segment_idx = torque_active_segment_idx[entry_idx]
            start_wp = apply_segment_start_idx[segment_idx]
            end_wp = apply_segment_end_idx[segment_idx]
            p_start = wp.vec3(0.0, 0.0, 0.0)
            start_begin = int(0)
            start_end = int(0)
            if start_wp == previous_end_wp:
                p_start = previous_p_end
                start_begin = previous_end_begin
                start_end = previous_end_end
            else:
                p_start = transformed_waypoints[env_idx, start_wp]
                start_offset_idx = torque_idx * waypoint_count + start_wp
                start_begin = active_attach_offsets[start_offset_idx]
                start_end = active_attach_offsets[start_offset_idx + 1]

            p_end = transformed_waypoints[env_idx, end_wp]
            end_offset_idx = torque_idx * waypoint_count + end_wp
            end_begin = active_attach_offsets[end_offset_idx]
            end_end = active_attach_offsets[end_offset_idx + 1]
            group_total += _lbs_segment_entry_contribution_active_points_wp(
                env_idx,
                torque_idx,
                start_wp,
                end_wp,
                unit_force_magnitude[env_idx, muscle_idx],
                linear_cdof,
                angular_cdof,
                p_start,
                p_end,
                body_pos,
                body_quat,
                subtree_com,
                body_rootid,
                start_begin,
                start_end,
                end_begin,
                end_end,
                active_attach_body,
                active_attach_local,
                active_attach_coeff_weight,
            )
            previous_end_wp = end_wp
            previous_p_end = p_end
            previous_end_begin = end_begin
            previous_end_end = end_end
        if group_total < 0.0:
            lower_total += group_total
        else:
            upper_total += group_total
        group_idx += wp.block_dim()

    lower_tile = wp.tile(lower_total)
    upper_tile = wp.tile(upper_total)
    lower_sum = wp.tile_sum(lower_tile)
    upper_sum = wp.tile_sum(upper_tile)
    if lane_idx == 0:
        partial_lower[env_idx, chunk_idx] = wp.tile_extract(lower_sum, 0)
        partial_upper[env_idx, chunk_idx] = wp.tile_extract(upper_sum, 0)


@wp.kernel(enable_backward=False)
def _reduce_lbs_torque_bounds_chunk_partials_kernel(
    partial_lower: wp.array2d(dtype=wp.float32),
    partial_upper: wp.array2d(dtype=wp.float32),
    torque_chunk_offsets: wp.array(dtype=wp.int32),
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
):
    env_idx, torque_idx = wp.tid()
    chunk_begin = torque_chunk_offsets[torque_idx]
    chunk_end = torque_chunk_offsets[torque_idx + 1]
    lower_total = float(0.0)
    upper_total = float(0.0)
    for chunk_idx in range(chunk_begin, chunk_end):
        lower_total += partial_lower[env_idx, chunk_idx]
        upper_total += partial_upper[env_idx, chunk_idx]
    lower[env_idx, torque_idx] = lower_total
    upper[env_idx, torque_idx] = upper_total


@cache
def make_lbs_torque_waypoint_force_active_attach_pair_vec4_native_partials_kernel(
    include_passive_force: bool,
    single_root_body_idx: int,
):
    single_root_body_idx = int(single_root_body_idx)

    if single_root_body_idx >= 0:

        @wp.kernel(enable_backward=False, module=_TILED_KERNEL_MODULE)
        @expose_kernel_name
        def _compute_lbs_torque_waypoint_force_active_attach_pair_vec4_native_partials_single_root_kernel(
            waypoint_force: wp.array2d(dtype=wp.vec3),
            body_pos: wp.array2d(dtype=wp.vec3),
            body_quat: wp.array2d(dtype=wp.quat),
            subtree_com: wp.array2d(dtype=wp.vec3),
            cdof: wp.array3d(dtype=wp.float32),
            active_attach_chunk_info: wp.array(dtype=wp.vec3i),
            active_attach_idx_body: wp.array(dtype=wp.vec2i),
            active_attach_local_coeff: wp.array(dtype=wp.vec4),
            root_dof_count: int,
            partial: wp.array2d(dtype=wp.float32),
        ):
            env_idx, chunk_idx, lane_idx = wp.tid()
            chunk = active_attach_chunk_info[chunk_idx]
            torque_idx = chunk[0]
            active_idx = chunk[1] + lane_idx
            active_end = chunk[2]
            full_torque_idx = torque_idx + root_dof_count
            linear_cdof = wp.vec3(
                cdof[env_idx, full_torque_idx, 3],
                cdof[env_idx, full_torque_idx, 4],
                cdof[env_idx, full_torque_idx, 5],
            )
            angular_cdof = wp.vec3(
                cdof[env_idx, full_torque_idx, 0],
                cdof[env_idx, full_torque_idx, 1],
                cdof[env_idx, full_torque_idx, 2],
            )
            root_com = subtree_com[env_idx, wp.static(single_root_body_idx)]
            total = float(0.0)
            while active_idx < active_end:
                idx_body = native_load_vec2i_wp(active_attach_idx_body, active_idx)
                packed = native_load_vec4_wp(active_attach_local_coeff, active_idx)
                waypoint_idx = idx_body[0]
                body_idx = idx_body[1]
                signed_force = waypoint_force[env_idx, waypoint_idx]
                q = _normalize_quat_wp(body_quat[env_idx, body_idx])
                local = wp.vec3(packed[0], packed[1], packed[2])
                point = body_pos[env_idx, body_idx] + rot_vec_quat(local, q)
                term = linear_cdof + wp.cross(angular_cdof, point - root_com)
                total += packed[3] * wp.dot(term, signed_force)
                active_idx += wp.block_dim()

            total_tile = wp.tile(total)
            sum_tile = wp.tile_sum(total_tile)
            if lane_idx == 0:
                partial[env_idx, chunk_idx] = wp.tile_extract(sum_tile, 0)

        return _compute_lbs_torque_waypoint_force_active_attach_pair_vec4_native_partials_single_root_kernel

    @wp.kernel(enable_backward=False, module=_TILED_KERNEL_MODULE)
    @expose_kernel_name
    def _compute_lbs_torque_waypoint_force_active_attach_pair_vec4_native_partials_multi_root_kernel(
        waypoint_force: wp.array2d(dtype=wp.vec3),
        body_pos: wp.array2d(dtype=wp.vec3),
        body_quat: wp.array2d(dtype=wp.quat),
        subtree_com: wp.array2d(dtype=wp.vec3),
        cdof: wp.array3d(dtype=wp.float32),
        body_rootid: wp.array(dtype=wp.int32),
        active_attach_chunk_info: wp.array(dtype=wp.vec3i),
        active_attach_idx_body: wp.array(dtype=wp.vec2i),
        active_attach_local_coeff: wp.array(dtype=wp.vec4),
        root_dof_count: int,
        partial: wp.array2d(dtype=wp.float32),
    ):
        env_idx, chunk_idx, lane_idx = wp.tid()
        chunk = active_attach_chunk_info[chunk_idx]
        torque_idx = chunk[0]
        active_idx = chunk[1] + lane_idx
        active_end = chunk[2]
        full_torque_idx = torque_idx + root_dof_count
        linear_cdof = wp.vec3(
            cdof[env_idx, full_torque_idx, 3],
            cdof[env_idx, full_torque_idx, 4],
            cdof[env_idx, full_torque_idx, 5],
        )
        angular_cdof = wp.vec3(
            cdof[env_idx, full_torque_idx, 0],
            cdof[env_idx, full_torque_idx, 1],
            cdof[env_idx, full_torque_idx, 2],
        )
        total = float(0.0)
        while active_idx < active_end:
            idx_body = native_load_vec2i_wp(active_attach_idx_body, active_idx)
            packed = native_load_vec4_wp(active_attach_local_coeff, active_idx)
            waypoint_idx = idx_body[0]
            body_idx = idx_body[1]
            signed_force = waypoint_force[env_idx, waypoint_idx]
            q = _normalize_quat_wp(body_quat[env_idx, body_idx])
            local = wp.vec3(packed[0], packed[1], packed[2])
            point = body_pos[env_idx, body_idx] + rot_vec_quat(local, q)
            root_idx = body_rootid[body_idx]
            term = linear_cdof + wp.cross(
                angular_cdof,
                point - subtree_com[env_idx, root_idx],
            )
            total += packed[3] * wp.dot(term, signed_force)
            active_idx += wp.block_dim()

        total_tile = wp.tile(total)
        sum_tile = wp.tile_sum(total_tile)
        if lane_idx == 0:
            partial[env_idx, chunk_idx] = wp.tile_extract(sum_tile, 0)

    return _compute_lbs_torque_waypoint_force_active_attach_pair_vec4_native_partials_multi_root_kernel


@wp.kernel(enable_backward=False)
def _reduce_lbs_torque_chunk_partials_kernel(
    partial: wp.array2d(dtype=wp.float32),
    torque_chunk_offsets: wp.array(dtype=wp.int32),
    mean: wp.array2d(dtype=wp.float32),
):
    env_idx, torque_idx = wp.tid()
    chunk_begin = torque_chunk_offsets[torque_idx]
    chunk_end = torque_chunk_offsets[torque_idx + 1]
    total = float(0.0)
    for chunk_idx in range(chunk_begin, chunk_end):
        total += partial[env_idx, chunk_idx]
    mean[env_idx, torque_idx] = total


@wp.kernel(enable_backward=False)
def _reduce_activation_interval_bounds_kernel(
    contribution: wp.array3d(dtype=wp.float32),
    muscle_count: int,
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
):
    env_idx, torque_idx = wp.tid()
    lower_total = float(0.0)
    upper_total = float(0.0)
    for muscle_idx in range(muscle_count):
        value = contribution[env_idx, torque_idx, muscle_idx]
        lower_total += wp.min(value, 0.0)
        upper_total += wp.max(value, 0.0)
    lower[env_idx, torque_idx] = lower_total
    upper[env_idx, torque_idx] = upper_total


@wp.kernel(enable_backward=False)
def _select_activation_interval_bounds_kernel(
    lower_full: wp.array2d(dtype=wp.float32),
    upper_full: wp.array2d(dtype=wp.float32),
    active_torque_indices: wp.array(dtype=wp.int32),
    lower_active: wp.array2d(dtype=wp.float32),
    upper_active: wp.array2d(dtype=wp.float32),
):
    env_idx, active_idx = wp.tid()
    torque_idx = active_torque_indices[active_idx]
    lower_active[env_idx, active_idx] = lower_full[env_idx, torque_idx]
    upper_active[env_idx, active_idx] = upper_full[env_idx, torque_idx]


@wp.kernel(enable_backward=False)
def _add_passive_offset_to_bounds_kernel(
    passive_offset: wp.array2d(dtype=wp.float32),
    lower: wp.array2d(dtype=wp.float32),
    upper: wp.array2d(dtype=wp.float32),
):
    env_idx, torque_idx = wp.tid()
    offset = passive_offset[env_idx, torque_idx]
    lower[env_idx, torque_idx] += offset
    upper[env_idx, torque_idx] += offset
