#if defined(__CUDA_ARCH__)
constexpr int kMaxTileMuscles = 32;
constexpr int kMaxTileWaypoints = 288;
constexpr int kMaxTileBodies = 4;
constexpr int kMaxBlockDim = 128;
constexpr int kWarpSize = 32;
// Active force-length model: 0 Gaussian exp(-(l - 1)^2 / gamma) with
// param0 = 1 / gamma; 1 MuJoCo's piecewise quadratic on [lmin, lmax] with
// param0 = lmin, param1 = lmax (active_force_length_model_wp).
constexpr int kActiveModel = __INSTANTMIMIC_ACTIVE_MODEL__;
constexpr float kActiveParam0 = __INSTANTMIMIC_ACTIVE_PARAM0__;
constexpr float kActiveParam1 = __INSTANTMIMIC_ACTIVE_PARAM1__;
constexpr int kPassiveModel = __INSTANTMIMIC_PASSIVE_MODEL__;
constexpr float kPassiveParam0 = __INSTANTMIMIC_PASSIVE_PARAM0__;
constexpr float kPassiveParam1 = __INSTANTMIMIC_PASSIVE_PARAM1__;

#define ARR2_PTR(arr, type, i, j) \
    reinterpret_cast<type*>(reinterpret_cast<char*>((arr).data) + \
        static_cast<int64_t>(i) * (arr).strides[0] + \
        static_cast<int64_t>(j) * (arr).strides[1])
#define ARR3_FLOAT_PTR(arr, i, j, k) \
    reinterpret_cast<float*>(reinterpret_cast<char*>((arr).data) + \
        static_cast<int64_t>(i) * (arr).strides[0] + \
        static_cast<int64_t>(j) * (arr).strides[1] + \
        static_cast<int64_t>(k) * (arr).strides[2])
#define ROT_VEC_QUAT(vx, vy, vz, q, ox, oy, oz) \
    do { \
        const float _s = (q)[0]; \
        const float _ux = (q)[1]; \
        const float _uy = (q)[2]; \
        const float _uz = (q)[3]; \
        const float _tx = 2.0f * (_uy * (vz) - _uz * (vy)); \
        const float _ty = 2.0f * (_uz * (vx) - _ux * (vz)); \
        const float _tz = 2.0f * (_ux * (vy) - _uy * (vx)); \
        (ox) = (vx) + _s * _tx + (_uy * _tz - _uz * _ty); \
        (oy) = (vy) + _s * _ty + (_uz * _tx - _ux * _tz); \
        (oz) = (vz) + _s * _tz + (_ux * _ty - _uy * _tx); \
    } while (0)

__shared__ float s_wp_x[kMaxTileWaypoints];
__shared__ float s_wp_y[kMaxTileWaypoints];
__shared__ float s_wp_z[kMaxTileWaypoints];
__shared__ float s_force[kMaxTileMuscles];
__shared__ float s_body_q[kMaxTileBodies][4];
__shared__ float s_body_pcom[kMaxTileBodies][3];
#if __INSTANTMIMIC_WRAPPED__ == 1
__shared__ int s_arc[kMaxTileWaypoints];
__shared__ float s_body_com[kMaxTileBodies][3];
#endif
// Per body slot wrench (force xyz, torque xyz), accumulated by warp runs.
__shared__ float s_acc[kMaxTileBodies * 6];
#if __INSTANTMIMIC_FORCE_VELOCITY__ == 1
// Per tile point: the body it moves with.
__shared__ unsigned short s_wp_body[kMaxTileWaypoints];
// Lengthening rate of the straight segment starting at each tile point,
// evaluated at the solved fixed-radius tangents.
__shared__ float s_seg_vel[kMaxTileWaypoints];
// Velocity of a world point moving with ``body`` (body_point_velocity_wp).
#define BODY_POINT_VELOCITY(body, px, py, pz, ox, oy, oz) \
    do { \
        const wp::spatial_vector _v = *ARR2_PTR(body_cvel, wp::spatial_vector, env_idx, (body)); \
        const wp::vec3 _c = *ARR2_PTR(subtree_com, wp::vec3, env_idx, body_rootid.data[(body)]); \
        const float _rx = (px) - _c[0]; \
        const float _ry = (py) - _c[1]; \
        const float _rz = (pz) - _c[2]; \
        (ox) = _v[3] + (_v[1] * _rz - _v[2] * _ry); \
        (oy) = _v[4] + (_v[2] * _rx - _v[0] * _rz); \
        (oz) = _v[5] + (_v[0] * _ry - _v[1] * _rx); \
    } while (0)
#endif

const wp::vec4i tile = tile_info.data[tile_idx];
const wp::vec4i extra = tile_extra_info.data[tile_idx];
const int muscle_begin = tile[0];
const int muscle_end = tile[1];
const int body_begin = tile[2];
const int event_begin = tile[3];
const int waypoint_begin = extra[0];
const int waypoint_count = extra[1];
const int body_count = extra[2];
const int event_end = extra[3];
const int muscle_count = muscle_end - muscle_begin;
for (int i = lane_idx; i < body_count * 6; i += blockDim.x) {
    s_acc[i] = 0.0f;
}

for (int local_wp = lane_idx; local_wp < waypoint_count; local_wp += blockDim.x) {
    const int global_wp = waypoint_begin + local_wp;
#if __INSTANTMIMIC_WRAPPED__ == 1
    const int original_wp = point_source.data[global_wp];
    s_arc[local_wp] = wrap_state.point_arc.data[original_wp];
    const wp::vec3 point = *ARR2_PTR(wrapped_points, wp::vec3, env_idx, original_wp);
    s_wp_x[local_wp] = point[0];
    s_wp_y[local_wp] = point[1];
    s_wp_z[local_wp] = point[2];
#if __INSTANTMIMIC_FORCE_VELOCITY__ == 1
    s_wp_body[local_wp] = static_cast<unsigned short>(
        waypoint_attachment_body2.data[global_wp][0]);
#endif
    continue;
#endif
    {
    const wp::vec2i body2 = waypoint_attachment_body2.data[global_wp];
    const wp::vec4 lw0 = waypoint_attachment_local_weight0.data[global_wp];

    const int body0 = body2[0];
    const wp::vec3 bp0 = *ARR2_PTR(body_pos, wp::vec3, env_idx, body0);
    const wp::quat q0 = *ARR2_PTR(normalized_body_quat, wp::quat, env_idx, body0);
    float rx0, ry0, rz0;
    ROT_VEC_QUAT(lw0[0], lw0[1], lw0[2], q0, rx0, ry0, rz0);

#if __INSTANTMIMIC_ATTACHMENT_ARITY__ == 1
    s_wp_x[local_wp] = lw0[3] * (bp0[0] + rx0);
    s_wp_y[local_wp] = lw0[3] * (bp0[1] + ry0);
    s_wp_z[local_wp] = lw0[3] * (bp0[2] + rz0);
#if __INSTANTMIMIC_FORCE_VELOCITY__ == 1
    s_wp_body[local_wp] = static_cast<unsigned short>(body0);
#endif
#else
    const wp::vec4 lw1 = waypoint_attachment_local_weight1.data[global_wp];
    const int body1 = body2[1];
    const wp::vec3 bp1 = *ARR2_PTR(body_pos, wp::vec3, env_idx, body1);
    const wp::quat q1 = *ARR2_PTR(normalized_body_quat, wp::quat, env_idx, body1);
    float rx1, ry1, rz1;
    ROT_VEC_QUAT(lw1[0], lw1[1], lw1[2], q1, rx1, ry1, rz1);

    s_wp_x[local_wp] = lw0[3] * (bp0[0] + rx0) + lw1[3] * (bp1[0] + rx1);
    s_wp_y[local_wp] = lw0[3] * (bp0[1] + ry0) + lw1[3] * (bp1[1] + ry1);
    s_wp_z[local_wp] = lw0[3] * (bp0[2] + rz0) + lw1[3] * (bp1[2] + rz1);
#endif
    }
}

if (lane_idx < body_count) {
    const int body_idx = tile_body_ids.data[body_begin + lane_idx];

    const wp::quat q = *ARR2_PTR(normalized_body_quat, wp::quat, env_idx, body_idx);
    const wp::vec3 bp = *ARR2_PTR(body_pos, wp::vec3, env_idx, body_idx);
    const wp::vec3 com = *ARR2_PTR(xipos, wp::vec3, env_idx, body_idx);
    s_body_q[lane_idx][0] = q[0];
    s_body_q[lane_idx][1] = q[1];
    s_body_q[lane_idx][2] = q[2];
    s_body_q[lane_idx][3] = q[3];
    s_body_pcom[lane_idx][0] = bp[0] - com[0];
    s_body_pcom[lane_idx][1] = bp[1] - com[1];
    s_body_pcom[lane_idx][2] = bp[2] - com[2];
#if __INSTANTMIMIC_WRAPPED__ == 1
    s_body_com[lane_idx][0] = com[0];
    s_body_com[lane_idx][1] = com[1];
    s_body_com[lane_idx][2] = com[2];
#endif
}

__syncthreads();

#if __INSTANTMIMIC_FORCE_VELOCITY__ == 1
// Path velocity per segment as in eval_muscle_force_magnitude_velocity_kernel,
// in parallel over tile points;
// segments that cross muscles or lie on an arc are never read.
for (int local_wp = lane_idx; local_wp + 1 < waypoint_count; local_wp += blockDim.x) {
    const float dx = s_wp_x[local_wp + 1] - s_wp_x[local_wp];
    const float dy = s_wp_y[local_wp + 1] - s_wp_y[local_wp];
    const float dz = s_wp_z[local_wp + 1] - s_wp_z[local_wp];
    const float seg_len = sqrtf(dx * dx + dy * dy + dz * dz);
    float seg_vel = 0.0f;
#if __INSTANTMIMIC_WRAPPED__ == 1
    if (s_arc[local_wp] >= 0) {
        s_seg_vel[local_wp] = 0.0f;
        continue;
    }
#endif
    const int body0 = s_wp_body[local_wp];
    const int body1 = s_wp_body[local_wp + 1];
    if (body0 == body1) {
        // Both ends move rigidly with one body: no lengthening rate.
        s_seg_vel[local_wp] = 0.0f;
        continue;
    }
    if (seg_len >= 1.0e-8f) {
        float v0x, v0y, v0z, v1x, v1y, v1z;
        BODY_POINT_VELOCITY(body0,
            s_wp_x[local_wp], s_wp_y[local_wp], s_wp_z[local_wp], v0x, v0y, v0z);
        BODY_POINT_VELOCITY(body1,
            s_wp_x[local_wp + 1], s_wp_y[local_wp + 1], s_wp_z[local_wp + 1],
            v1x, v1y, v1z);
        seg_vel = (dx * (v1x - v0x) + dy * (v1y - v0y) + dz * (v1z - v0z)) / seg_len;
    }
    s_seg_vel[local_wp] = seg_vel;
}
__syncthreads();
#endif

if (lane_idx < muscle_count) {
    const int muscle_idx = muscle_begin + lane_idx;
    int segment_idx = muscle_segment_offsets.data[muscle_idx];
    const int segment_end = muscle_segment_offsets.data[muscle_idx + 1];
    float muscle_length = 0.0f;
#if __INSTANTMIMIC_FORCE_VELOCITY__ == 1
    float muscle_velocity = 0.0f;
#endif
    while (segment_idx < segment_end) {
        const int local_start_wp = segment_idx + muscle_idx - waypoint_begin;
        const float dx = s_wp_x[local_start_wp + 1] - s_wp_x[local_start_wp];
        const float dy = s_wp_y[local_start_wp + 1] - s_wp_y[local_start_wp];
        const float dz = s_wp_z[local_start_wp + 1] - s_wp_z[local_start_wp];
        const float len_sq = dx * dx + dy * dy + dz * dz;
#if __INSTANTMIMIC_WRAPPED__ == 1
        const int instance = s_arc[local_start_wp];
        if (instance >= 0) {
            muscle_length += *ARR2_PTR(wrap_state.arc, float, env_idx, instance);
        } else
#endif
        {
            muscle_length += len_sq * rsqrtf(fmaxf(len_sq, 1.0e-16f));
#if __INSTANTMIMIC_FORCE_VELOCITY__ == 1
            muscle_velocity += s_seg_vel[local_start_wp];
#endif
        }
        ++segment_idx;
    }

    const wp::vec4 coeff = muscle_force_length_coeff.data[muscle_idx];
    float l_norm = muscle_length * coeff[0] + coeff[1];
    l_norm = fminf(fmaxf(l_norm, muscle_length_norm_min), muscle_length_norm_max);
    const float l_delta = l_norm - 1.0f;
#if __INSTANTMIMIC_ACTIVE_MODEL__ == 0
    const float active = expf(-(l_delta * l_delta) * kActiveParam0);
#else
    float active = 0.0f;
    if (l_norm >= kActiveParam0 && l_norm <= kActiveParam1) {
        const float lower_mid = 0.5f * (kActiveParam0 + 1.0f);
        const float upper_mid = 0.5f * (1.0f + kActiveParam1);
        if (l_norm <= lower_mid) {
            const float x = (l_norm - kActiveParam0) / fmaxf(lower_mid - kActiveParam0, 1.0e-8f);
            active = 0.5f * x * x;
        } else if (l_norm <= 1.0f) {
            const float x = (1.0f - l_norm) / fmaxf(1.0f - lower_mid, 1.0e-8f);
            active = 1.0f - 0.5f * x * x;
        } else if (l_norm <= upper_mid) {
            const float x = (l_norm - 1.0f) / fmaxf(upper_mid - 1.0f, 1.0e-8f);
            active = 1.0f - 0.5f * x * x;
        } else {
            const float x = (kActiveParam1 - l_norm) / fmaxf(kActiveParam1 - upper_mid, 1.0e-8f);
            active = 0.5f * x * x;
        }
    }
#endif
#if __INSTANTMIMIC_FORCE_VELOCITY__ == 1
    // Activation and velocity scale arrive in model muscle order; MuJoCo's
    // force-velocity curve scales the active force (muscle_force_velocity_curve_wp).
    const int model_muscle = muscle_order.data[muscle_idx];
    float muscle_activation = *ARR2_PTR(activation, float, env_idx, model_muscle);
    if (clamp_activation != 0) {
        muscle_activation = fminf(fmaxf(muscle_activation, 0.0f), 1.0f);
    }
    const float normalized_velocity = muscle_velocity / velocity_scale.data[model_muscle];
    const float fv_rise = fvmax - 1.0f;
    float fv_curve;
    if (normalized_velocity <= -1.0f) {
        fv_curve = 0.0f;
    } else if (normalized_velocity <= 0.0f) {
        const float shifted = normalized_velocity + 1.0f;
        fv_curve = shifted * shifted;
    } else if (normalized_velocity <= fv_rise) {
        const float remaining = fv_rise - normalized_velocity;
        fv_curve = fvmax - remaining * remaining / fv_rise;
    } else {
        fv_curve = fvmax;
    }
    float force_scale = active * (muscle_activation * fv_curve);
#else
    float force_scale = active * activation.data[
        static_cast<int64_t>(env_idx) * activation.strides[0] / sizeof(float) +
        static_cast<int64_t>(muscle_idx) * activation.strides[1] / sizeof(float)
    ];
#endif
#if __INSTANTMIMIC_PASSIVE_MODEL__ == 1
    if (l_delta >= 0.0f) {
        float x = kPassiveParam0 * l_delta;
        x = fminf(x, 80.0f);
        force_scale += (expf(x) - 1.0f) * kPassiveParam1;
    }
#elif __INSTANTMIMIC_PASSIVE_MODEL__ == 2
    if (l_delta > 0.0f) {
        const float stretch = l_delta * kPassiveParam0;
        if (stretch <= 1.0f) {
            force_scale += 0.5f * kPassiveParam1 * stretch * stretch;
        } else {
            force_scale += kPassiveParam1 * (stretch - 0.5f);
        }
    }
#endif
    s_force[lane_idx] = coeff[2] * force_scale;
}

__syncthreads();

// Events are sorted by body slot within a tile, so each warp reduces its 32
// consecutive events by contiguous slot runs and one lane per run adds the
// run's wrench to shared memory. Any number of body slots fits.
const int warp_lane = lane_idx & (kWarpSize - 1);
for (int event_base = event_begin; event_base < event_end; event_base += blockDim.x) {
    const int event_idx = event_base + lane_idx;
    int body_slot = -1;
    float signed_fx = 0.0f, signed_fy = 0.0f, signed_fz = 0.0f;
    float tvx = 0.0f, tvy = 0.0f, tvz = 0.0f;
    if (event_idx < event_end) {
    const unsigned int ev = tile_event_start_slot_muscle.data[event_idx];
    const int local_start_wp = ev & 511;
    const int local_muscle_idx = (ev >> 9) & 31;
    body_slot = static_cast<int>(ev >> 14);
    const float fmag = s_force[local_muscle_idx];
    const float sx = s_wp_x[local_start_wp + 1] - s_wp_x[local_start_wp];
    const float sy = s_wp_y[local_start_wp + 1] - s_wp_y[local_start_wp];
    const float sz = s_wp_z[local_start_wp + 1] - s_wp_z[local_start_wp];
    const float inv_len = rsqrtf(fmaxf(sx * sx + sy * sy + sz * sz, 1.0e-16f));
    const float fvx = fmag * sx * inv_len;
    const float fvy = fmag * sy * inv_len;
    const float fvz = fmag * sz * inv_len;
    const wp::vec4 coeff = tile_event_local_force_coeff.data[event_idx];
    signed_fx = coeff[3] * fvx;
    signed_fy = coeff[3] * fvy;
    signed_fz = coeff[3] * fvz;
    float max, may, maz;
#if __INSTANTMIMIC_WRAPPED__ == 1
    // Use the same world-space point for the direction and lever arm.
    const int endpoint_wp = local_start_wp + (coeff[3] < 0.0f ? 1 : 0);
    {
        max = coeff[3] * (s_wp_x[endpoint_wp] - s_body_com[body_slot][0]);
        may = coeff[3] * (s_wp_y[endpoint_wp] - s_body_com[body_slot][1]);
        maz = coeff[3] * (s_wp_z[endpoint_wp] - s_body_com[body_slot][2]);
    }
#else
    {
        const wp::quat bq = wp::quat(
            s_body_q[body_slot][0],
            s_body_q[body_slot][1],
            s_body_q[body_slot][2],
            s_body_q[body_slot][3]
        );
        float rx, ry, rz;
        ROT_VEC_QUAT(coeff[0], coeff[1], coeff[2], bq, rx, ry, rz);
        max = coeff[3] * s_body_pcom[body_slot][0] + rx;
        may = coeff[3] * s_body_pcom[body_slot][1] + ry;
        maz = coeff[3] * s_body_pcom[body_slot][2] + rz;
    }
#endif
    tvx = may * fvz - maz * fvy;
    tvy = maz * fvx - max * fvz;
    tvz = max * fvy - may * fvx;

    }

    #pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int other_slot = __shfl_up_sync(0xffffffffu, body_slot, offset);
        const float ofx = __shfl_up_sync(0xffffffffu, signed_fx, offset);
        const float ofy = __shfl_up_sync(0xffffffffu, signed_fy, offset);
        const float ofz = __shfl_up_sync(0xffffffffu, signed_fz, offset);
        const float otx = __shfl_up_sync(0xffffffffu, tvx, offset);
        const float oty = __shfl_up_sync(0xffffffffu, tvy, offset);
        const float otz = __shfl_up_sync(0xffffffffu, tvz, offset);
        if (warp_lane >= offset && other_slot == body_slot) {
            signed_fx += ofx; signed_fy += ofy; signed_fz += ofz;
            tvx += otx; tvy += oty; tvz += otz;
        }
    }
    const int next_slot = __shfl_down_sync(0xffffffffu, body_slot, 1);
    if (body_slot >= 0 && (warp_lane == kWarpSize - 1 || next_slot != body_slot)) {
        float* acc = s_acc + body_slot * 6;
        atomicAdd(acc + 0, signed_fx);
        atomicAdd(acc + 1, signed_fy);
        atomicAdd(acc + 2, signed_fz);
        atomicAdd(acc + 3, tvx);
        atomicAdd(acc + 4, tvy);
        atomicAdd(acc + 5, tvz);
    }
}
__syncthreads();

for (int i = lane_idx; i < body_count * 6; i += blockDim.x) {
    const int slot = i / 6;
    const int comp = i - slot * 6;
    const int output_idx = tile_body_output_idx.data[body_begin + slot];
    *ARR3_FLOAT_PTR(partial_wrench, env_idx, output_idx, comp) = s_acc[i];
}

#if __INSTANTMIMIC_FORCE_VELOCITY__ == 1
#undef BODY_POINT_VELOCITY
#endif
#undef ROT_VEC_QUAT
#undef ARR3_FLOAT_PTR
#undef ARR2_PTR
#else
// CPU fallback is unused by the CUDA-only benchmark path.
#endif
