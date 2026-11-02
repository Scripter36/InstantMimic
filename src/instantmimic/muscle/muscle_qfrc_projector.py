from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch
import warp as wp

from instantmimic.utils.mjw_kernels.private import rot_vec_quat
from instantmimic.utils.startup_profiler import startup_nvtx_wrapped


@wp.func
def segment_length_wp(
    env_idx: int, start: int, points: wp.array2d(dtype=wp.vec3)
) -> float:
    return wp.length(points[env_idx, start + 1] - points[env_idx, start])


@wp.func
def body_point_velocity_wp(
    env_idx: int,
    body: int,
    point: wp.vec3,
    subtree_com: wp.array2d(dtype=wp.vec3),
    body_rootid: wp.array(dtype=wp.int32),
    body_cvel: wp.array2d(dtype=wp.spatial_vector),
) -> wp.vec3:
    velocity = body_cvel[env_idx, body]
    angular = wp.vec3(velocity[0], velocity[1], velocity[2])
    linear = wp.vec3(velocity[3], velocity[4], velocity[5])
    return linear + wp.cross(angular, point - subtree_com[env_idx, body_rootid[body]])


@wp.func_native("""
#if defined(__CUDA_ARCH__)
int32_t x;
int32_t y;
asm volatile(
    "ld.global.v2.u32 {%0, %1}, [%2];"
    : "=r"(x), "=r"(y)
    : "l"(data.data + i)
);
return wp::vec2i(x, y);
#else
return data.data[i];
#endif
""")
def native_load_vec2i_wp(data: wp.array(dtype=wp.vec2i), i: int) -> wp.vec2i: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
int32_t x;
int32_t y;
int32_t z;
int32_t w;
asm volatile(
    "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
    : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
    : "l"(data.data + i)
);
return wp::vec4i(x, y, z, w);
#else
return data.data[i];
#endif
""")
def native_load_vec4i_wp(data: wp.array(dtype=wp.vec4i), i: int) -> wp.vec4i: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
float x;
float y;
float z;
float w;
asm volatile(
    "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
    : "l"(data.data + i)
);
return wp::vec4(x, y, z, w);
#else
return data.data[i];
#endif
""")
def native_load_vec4_wp(data: wp.array(dtype=wp.vec4), i: int) -> wp.vec4: ...


@wp.func
def normalize_quat_wp(q: wp.quat) -> wp.quat:
    inv_norm = 1.0 / wp.sqrt(
        q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] + 1.0e-8
    )
    return wp.quat(
        q[0] * inv_norm,
        q[1] * inv_norm,
        q[2] * inv_norm,
        q[3] * inv_norm,
    )


@wp.func
def active_force_length_wp(normalized_length: float, gamma: float) -> float:
    return wp.exp(-((normalized_length - 1.0) * (normalized_length - 1.0)) / gamma)


@wp.func
def mujoco_active_force_length_wp(
    normalized_length: float,
    lmin: float,
    lmax: float,
) -> float:
    if normalized_length < lmin or normalized_length > lmax:
        return 0.0
    lower_midpoint = 0.5 * (lmin + 1.0)
    upper_midpoint = 0.5 * (1.0 + lmax)
    if normalized_length <= lower_midpoint:
        x = (normalized_length - lmin) / wp.max(lower_midpoint - lmin, 1.0e-8)
        return 0.5 * x * x
    if normalized_length <= 1.0:
        x = (1.0 - normalized_length) / wp.max(1.0 - lower_midpoint, 1.0e-8)
        return 1.0 - 0.5 * x * x
    if normalized_length <= upper_midpoint:
        x = (normalized_length - 1.0) / wp.max(upper_midpoint - 1.0, 1.0e-8)
        return 1.0 - 0.5 * x * x
    x = (lmax - normalized_length) / wp.max(lmax - upper_midpoint, 1.0e-8)
    return 0.5 * x * x


@wp.func
def active_force_length_model_wp(
    normalized_length: float,
    model_params: wp.vec3,
) -> float:
    if model_params[0] < 0.5:
        return active_force_length_wp(normalized_length, model_params[1])
    return mujoco_active_force_length_wp(
        normalized_length,
        model_params[1],
        model_params[2],
    )


@wp.func
def passive_force_length_wp(
    normalized_length: float, k_pe: float, e_mo: float
) -> float:
    if normalized_length < 1.0:
        return 0.0
    x = k_pe * (normalized_length - 1.0) / e_mo
    x = wp.clamp(x, -80.0, 80.0)
    return (wp.exp(x) - 1.0) / (wp.exp(k_pe) - 1.0)


@wp.func
def mujoco_passive_force_length_wp(
    normalized_length: float,
    lmax: float,
    fpmax: float,
) -> float:
    if normalized_length <= 1.0:
        return 0.0
    midpoint = 0.5 * (1.0 + lmax)
    width = wp.max(midpoint - 1.0, 1.0e-8)
    if normalized_length <= midpoint:
        stretch = (normalized_length - 1.0) / width
        return 0.5 * fpmax * stretch * stretch
    return fpmax * (0.5 + (normalized_length - midpoint) / width)


@wp.func
def passive_force_length_model_wp(
    normalized_length: float,
    model_params: wp.vec3,
) -> float:
    if model_params[0] < 0.5:
        return passive_force_length_wp(
            normalized_length,
            model_params[1],
            model_params[2],
        )
    return mujoco_passive_force_length_wp(
        normalized_length,
        model_params[1],
        model_params[2],
    )


@wp.func
def normalized_muscle_length_wp(
    muscle_length: float,
    l_m0: float,
    l_t0: float,
    muscle_rest_length: float,
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
) -> float:
    muscle_tendon_length0 = wp.max(muscle_rest_length, 1.0e-6)
    l_m = (muscle_length / muscle_tendon_length0) - l_t0
    return wp.clamp(
        l_m / wp.max(l_m0, 1.0e-6),
        muscle_length_norm_min,
        muscle_length_norm_max,
    )


@wp.func
def unit_force_magnitude_from_length_norm_wp(
    f0: float,
    pen_angle: float,
    muscle_force_scale: float,
    normalized_length: float,
    active_force_model_params: wp.vec3,
) -> float:
    return (
        f0
        * active_force_length_model_wp(normalized_length, active_force_model_params)
        * wp.cos(pen_angle)
        * muscle_force_scale
    )


@startup_nvtx_wrapped("metadata:build_muscle_force_length_coeff")
def build_muscle_force_length_coeff(
    muscle_params: torch.Tensor,
    muscle_rest_length: torch.Tensor,
    muscle_force_scale: float,
) -> torch.Tensor:
    """Pack static force-length constants as vec4-friendly rows.

    Row layout: (norm_scale, norm_bias, force_gain, unused).
    """
    params = muscle_params.detach().to(dtype=torch.float32)
    rest = muscle_rest_length.detach().to(dtype=torch.float32)
    l_m0 = params[:, 1].clamp_min(1.0e-6)
    norm_scale = 1.0 / (rest.clamp_min(1.0e-6) * l_m0)
    norm_bias = -params[:, 2] / l_m0
    force_gain = params[:, 0] * torch.cos(params[:, 3]) * float(muscle_force_scale)
    unused = torch.zeros_like(force_gain)
    return torch.stack(
        [norm_scale, norm_bias, force_gain, unused],
        dim=1,
    ).contiguous()


@wp.func
def force_magnitude_from_activation_model_wp(
    activation: float,
    clamp_activation: float,
    include_active: float,
    include_passive: float,
    f0: float,
    pen_angle: float,
    muscle_force_scale: float,
    normalized_length: float,
    active_force_model_params: wp.vec3,
    passive_force_model_params: wp.vec3,
) -> float:
    if clamp_activation != 0.0:
        activation = wp.clamp(activation, 0.0, 1.0)
    f_a = active_force_length_model_wp(
        normalized_length,
        active_force_model_params,
    )
    f_p = float(0.0)
    if include_passive != 0.0:
        f_p = passive_force_length_model_wp(
            normalized_length,
            passive_force_model_params,
        )
    return (
        f0
        * (include_active * (f_a * activation) + include_passive * f_p)
        * wp.cos(pen_angle)
        * muscle_force_scale
    )


@wp.func
def segment_generalized_force_contribution_wp(
    linear_cdof: wp.vec3,
    angular_cdof: wp.vec3,
    p_start: wp.vec3,
    p_end: wp.vec3,
    subtree_com_start: wp.vec3,
    subtree_com_end: wp.vec3,
    start_coeff: float,
    end_coeff: float,
    force_magnitude: float,
) -> float:
    segment = p_end - p_start
    segment_length = wp.length(segment)
    if segment_length < 1.0e-8:
        return 0.0

    seg_dir = segment / segment_length
    seg_force = force_magnitude * seg_dir
    start_term = linear_cdof + wp.cross(angular_cdof, p_start - subtree_com_start)
    end_term = linear_cdof + wp.cross(angular_cdof, p_end - subtree_com_end)
    return wp.dot(
        start_coeff * start_term - end_coeff * end_term,
        seg_force,
    )


@wp.func
def endpoint_generalized_force_contribution_wp(
    linear_cdof: wp.vec3,
    angular_cdof: wp.vec3,
    point: wp.vec3,
    subtree_com: wp.vec3,
    coeff: float,
    weight: float,
    signed_force: wp.vec3,
) -> float:
    term = linear_cdof + wp.cross(angular_cdof, point - subtree_com)
    return coeff * weight * wp.dot(term, signed_force)


@startup_nvtx_wrapped("metadata:build_muscle_segment_offsets")
def build_muscle_segment_offsets(
    segment_muscle_idx: torch.Tensor,
    muscle_count: int,
) -> torch.Tensor:
    counts = torch.bincount(segment_muscle_idx, minlength=muscle_count)
    offsets = torch.zeros(
        muscle_count + 1,
        device=segment_muscle_idx.device,
        dtype=torch.int32,
    )
    offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    return offsets


@startup_nvtx_wrapped("metadata:build_dof_subtree_mask")
def build_dof_subtree_mask(
    body_parentid: torch.Tensor,
    dof_bodyid: torch.Tensor,
    body_count: int,
    full_torque_dim: int,
) -> torch.Tensor:
    body_parent = [int(v) for v in body_parentid.detach().cpu().tolist()]
    ancestor = torch.zeros((body_count, body_count), dtype=torch.bool)
    for body_idx in range(body_count):
        parent_idx = body_idx
        while parent_idx != 0:
            ancestor[body_idx, parent_idx] = True
            parent_idx = body_parent[parent_idx]

    dof_body = dof_bodyid[:full_torque_dim].detach().cpu().to(torch.long)
    mask = ancestor.index_select(1, dof_body).transpose(0, 1).to(torch.float32)
    root_dof = dof_body == 0
    if bool(root_dof.any().item()):
        mask[root_dof] = 1.0
    return mask.to(device=body_parentid.device)


def build_body_ancestor_dof_lists(
    dof_subtree_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CSR lists of the DOFs that move each body, in ascending DOF order.

    ``dof_subtree_mask`` is the (dof, body) 0/1 mask of build_dof_subtree_mask;
    summing only its nonzero entries in this order reproduces the dense sum.
    """
    mask = dof_subtree_mask.detach().cpu()
    if not bool(((mask == 0.0) | (mask == 1.0)).all()):
        raise ValueError("dof_subtree_mask must contain only 0 and 1")
    moving = mask.transpose(0, 1) != 0.0
    offsets = torch.zeros(moving.shape[0] + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(moving.sum(dim=1), dim=0).to(torch.int32)
    dof_idx = moving.nonzero()[:, 1].to(torch.int32)
    device = dof_subtree_mask.device
    return offsets.to(device=device), dof_idx.to(device=device)


@startup_nvtx_wrapped("metadata:build_torque_sparse_segment_entries")
def build_torque_sparse_segment_entries(
    segment_start_dof_mask: torch.Tensor,
    segment_end_dof_mask: torch.Tensor,
    apply_segment_muscle_idx: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torque_dim, _ = segment_start_dof_mask.shape
    start_cpu = segment_start_dof_mask.detach().cpu()
    end_cpu = segment_end_dof_mask.detach().cpu()
    muscle_cpu = apply_segment_muscle_idx.detach().cpu()

    offsets = [0]
    torque_idx_parts: list[torch.Tensor] = []
    segment_idx_parts: list[torch.Tensor] = []
    muscle_idx_parts: list[torch.Tensor] = []
    start_coeff_parts: list[torch.Tensor] = []
    end_coeff_parts: list[torch.Tensor] = []

    for torque_idx in range(torque_dim):
        active = (start_cpu[torque_idx] != 0) | (end_cpu[torque_idx] != 0)
        active_idx = active.nonzero(as_tuple=False).flatten().to(torch.int32)
        active_idx_long = active_idx.long()
        torque_idx_parts.append(torch.full_like(active_idx, int(torque_idx)))
        segment_idx_parts.append(active_idx)
        muscle_idx_parts.append(
            muscle_cpu.index_select(0, active_idx_long).to(torch.int32)
        )
        start_coeff_parts.append(
            start_cpu[torque_idx].index_select(0, active_idx_long).float()
        )
        end_coeff_parts.append(
            end_cpu[torque_idx].index_select(0, active_idx_long).float()
        )
        offsets.append(offsets[-1] + int(active_idx.numel()))

    flat_segment_idx = (
        torch.cat(segment_idx_parts, dim=0)
        if segment_idx_parts
        else torch.zeros(0, dtype=torch.int32)
    )
    flat_torque_idx = (
        torch.cat(torque_idx_parts, dim=0)
        if torque_idx_parts
        else torch.zeros(0, dtype=torch.int32)
    )
    flat_muscle_idx = (
        torch.cat(muscle_idx_parts, dim=0)
        if muscle_idx_parts
        else torch.zeros(0, dtype=torch.int32)
    )
    flat_start_coeff = (
        torch.cat(start_coeff_parts, dim=0)
        if start_coeff_parts
        else torch.zeros(0, dtype=torch.float32)
    )
    flat_end_coeff = (
        torch.cat(end_coeff_parts, dim=0)
        if end_coeff_parts
        else torch.zeros(0, dtype=torch.float32)
    )
    offset_tensor = torch.tensor(offsets, dtype=torch.int32)
    return (
        offset_tensor,
        flat_torque_idx,
        flat_segment_idx,
        flat_muscle_idx,
        flat_start_coeff,
        flat_end_coeff,
    )


@startup_nvtx_wrapped("metadata:build_muscle_sparse_torque_entries")
def build_muscle_sparse_torque_entries(
    torque_entry_offsets: torch.Tensor,
    torque_segment_idx: torch.Tensor,
    torque_start_coeff: torch.Tensor,
    torque_end_coeff: torch.Tensor,
    apply_segment_muscle_idx: torch.Tensor,
    muscle_count: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    offsets = torque_entry_offsets.detach().cpu().to(torch.long)
    segment_idx = torque_segment_idx.detach().cpu().to(torch.long)
    start_coeff = torque_start_coeff.detach().cpu().to(torch.float32)
    end_coeff = torque_end_coeff.detach().cpu().to(torch.float32)
    segment_to_muscle = apply_segment_muscle_idx.detach().cpu().to(torch.long)
    torque_dim = max(int(offsets.numel()) - 1, 0)
    entry_count = int(segment_idx.numel())
    if torque_dim <= 0 or entry_count == 0:
        empty_i32 = torch.zeros(0, dtype=torch.int32)
        return (
            torch.zeros(muscle_count + 1, dtype=torch.int32),
            empty_i32.clone(),
            empty_i32.clone(),
            torch.zeros(0, dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )

    counts = offsets[1:] - offsets[:-1]
    torque_idx = torch.repeat_interleave(
        torch.arange(torque_dim, dtype=torch.long),
        counts,
    )
    muscle_idx = segment_to_muscle.index_select(0, segment_idx)
    segment_stride = torque_dim + 1
    muscle_stride = (int(segment_idx.max().item()) + 2) * segment_stride
    order = torch.argsort(
        muscle_idx * muscle_stride + segment_idx * segment_stride + torque_idx
    )
    sorted_muscle = muscle_idx.index_select(0, order)
    muscle_counts = torch.bincount(sorted_muscle, minlength=muscle_count)
    out_offsets = torch.zeros(muscle_count + 1, dtype=torch.int32)
    out_offsets[1:] = torch.cumsum(muscle_counts, dim=0).to(torch.int32)
    return (
        out_offsets,
        segment_idx.index_select(0, order).to(torch.int32),
        torque_idx.index_select(0, order).to(torch.int32),
        start_coeff.index_select(0, order),
        end_coeff.index_select(0, order),
    )


@dataclass(frozen=True)
class MuscleDofSparseEntries:
    muscle_dof_offsets: torch.Tensor
    muscle_dof_muscle_idx: torch.Tensor
    muscle_dof_torque_idx: torch.Tensor
    muscle_dof_segment_offsets: torch.Tensor
    segment_idx: torch.Tensor
    start_coeff: torch.Tensor
    end_coeff: torch.Tensor
    pair_entry_i: torch.Tensor
    pair_entry_j: torch.Tensor
    pair_muscle_idx: torch.Tensor


@dataclass(frozen=True)
class PreparedMuscleImplicitQderivPairs:
    pair_elem_id: torch.Tensor
    pair_entry_i: torch.Tensor
    pair_entry_j: torch.Tensor
    pair_muscle_idx: torch.Tensor
    dropped_pair_count: int


@startup_nvtx_wrapped("metadata:prepare_muscle_implicit_qderiv_pairs")
def prepare_muscle_implicit_qderiv_pairs(
    *,
    pair_entry_i: torch.Tensor,
    pair_entry_j: torch.Tensor,
    pair_muscle_idx: torch.Tensor,
    muscle_dof_torque_idx: torch.Tensor,
    matrix_elemid: torch.Tensor,
    root_dof_count: int,
) -> PreparedMuscleImplicitQderivPairs:
    """Resolve the packed mass-matrix element for every valid damping pair."""
    pair_i = pair_entry_i.detach().cpu().to(torch.long)
    pair_j = pair_entry_j.detach().cpu().to(torch.long)
    pair_muscle = pair_muscle_idx.detach().cpu().to(torch.long)
    dof_idx = muscle_dof_torque_idx.detach().cpu().to(torch.long)
    elemid = matrix_elemid.detach().cpu().to(torch.long)
    if pair_i.shape != pair_j.shape or pair_i.shape != pair_muscle.shape:
        raise ValueError("Implicit muscle pair arrays must have matching shapes")

    if pair_i.numel() == 0:
        empty_i32 = torch.zeros(0, dtype=torch.int32)
        return PreparedMuscleImplicitQderivPairs(
            pair_elem_id=empty_i32.clone(),
            pair_entry_i=empty_i32.clone(),
            pair_entry_j=empty_i32.clone(),
            pair_muscle_idx=empty_i32.clone(),
            dropped_pair_count=0,
        )

    full_dof_i = dof_idx.index_select(0, pair_i) + int(root_dof_count)
    full_dof_j = dof_idx.index_select(0, pair_j) + int(root_dof_count)
    mapped_elemid = elemid[full_dof_i, full_dof_j]
    valid = mapped_elemid >= 0
    dropped_pair_count = int(torch.count_nonzero(~valid).item())
    valid_idx = valid.nonzero(as_tuple=False).flatten()
    mapped_elemid = mapped_elemid.index_select(0, valid_idx)
    pair_i = pair_i.index_select(0, valid_idx)
    pair_j = pair_j.index_select(0, valid_idx)
    pair_muscle = pair_muscle.index_select(0, valid_idx)

    return PreparedMuscleImplicitQderivPairs(
        pair_elem_id=mapped_elemid.to(torch.int32),
        pair_entry_i=pair_i.to(torch.int32),
        pair_entry_j=pair_j.to(torch.int32),
        pair_muscle_idx=pair_muscle.to(torch.int32),
        dropped_pair_count=dropped_pair_count,
    )


# Pairs one lane sums before the per-element reduction of the gathered qDeriv.
# Measured on the SNU sprint tables (11,924 pairs over 396 elements, 4,096
# worlds): chunks of 8 and 16 tie, 32 pads the tables by 46%.
_QDERIV_GATHER_CHUNK = 16
_QDERIV_GATHER_BLOCK_DIM = 256
# One block keeps a world's moment row, its slope-weighted copy and one partial
# per chunk in static shared memory.
_QDERIV_GATHER_MAX_SHARED_BYTES = 48 * 1024
# Two entry indices share one int32; the padding sentinel is the entry count.
_QDERIV_GATHER_MAX_ENTRIES = 0x7FFF


@dataclass(frozen=True)
class MuscleImplicitQderivGather:
    """Implicit muscle qDeriv gathered per M element in a fixed summation order.

    Pairs are grouped by their M element and cut into chunks of
    ``_QDERIV_GATHER_CHUNK``; ``chunk_pairs`` packs each pair's two moment
    entries as ``i | j << 16`` in chunk-interleaved order (pair ``k`` of chunk
    ``c`` at ``k * chunk_count + c``), padded with the entry count.
    """

    kernel: object
    cuda: bool
    entry_muscle: wp.array
    elem_ids: wp.array
    chunk_offsets: wp.array
    chunk_pairs: wp.array


@startup_nvtx_wrapped("metadata:build_muscle_implicit_qderiv_gather")
def build_muscle_implicit_qderiv_gather(
    prepared: PreparedMuscleImplicitQderivPairs,
    entry_muscle_idx: torch.Tensor,
    device: object,
) -> MuscleImplicitQderivGather | None:
    """The gather tables, or ``None`` when they exceed its index or shared limits."""
    elem = prepared.pair_elem_id.detach().cpu().to(torch.long)
    entry_muscle = entry_muscle_idx.detach().cpu().to(torch.long)
    entry_count = int(entry_muscle.numel())
    if elem.numel() == 0 or entry_count >= _QDERIV_GATHER_MAX_ENTRIES:
        return None
    pair_i = prepared.pair_entry_i.detach().cpu().to(torch.long)
    pair_j = prepared.pair_entry_j.detach().cpu().to(torch.long)
    pair_muscle = prepared.pair_muscle_idx.detach().cpu().to(torch.long)
    # The gather weights entry i by its own muscle's slope.
    if not (
        torch.equal(entry_muscle[pair_i], pair_muscle)
        and torch.equal(entry_muscle[pair_j], pair_muscle)
    ):
        raise ValueError("every implicit qDeriv pair must join entries of its muscle")

    order = torch.argsort(elem, stable=True)
    elem, pair_i, pair_j = elem[order], pair_i[order], pair_j[order]
    elem_ids, counts = torch.unique_consecutive(elem, return_counts=True)
    chunk = _QDERIV_GATHER_CHUNK
    chunks = (counts + chunk - 1) // chunk
    chunk_offsets = torch.zeros(chunks.numel() + 1, dtype=torch.long)
    chunk_offsets[1:] = torch.cumsum(chunks, dim=0)
    chunk_count = int(chunk_offsets[-1])
    starts = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(elem.numel()) - torch.repeat_interleave(starts, counts)
    row = torch.repeat_interleave(chunk_offsets[:-1], counts) + rank // chunk
    column = rank % chunk
    packed = torch.full((chunk_count, chunk), entry_count | entry_count << 16)
    packed[row, column] = pair_i | pair_j << 16

    device = wp.get_device(str(device))
    cuda = device.is_cuda
    shared_bytes = 4 * (2 * (entry_count + 1) + chunk_count)
    if cuda and shared_bytes > _QDERIV_GATHER_MAX_SHARED_BYTES:
        return None

    def device_i32(values: torch.Tensor) -> wp.array:
        return wp.array(
            values.to(torch.int32).contiguous().numpy(), dtype=wp.int32, device=device
        )

    return MuscleImplicitQderivGather(
        kernel=_muscle_implicit_qderiv_gather_kernel(entry_count, chunk_count, cuda),
        cuda=cuda,
        entry_muscle=device_i32(entry_muscle),
        elem_ids=device_i32(elem_ids),
        chunk_offsets=device_i32(chunk_offsets),
        chunk_pairs=device_i32(packed.t().reshape(-1)),
    )


@cache
def _muscle_implicit_qderiv_gather_kernel(
    entry_count: int, chunk_count: int, cuda: bool
):
    chunk = _QDERIV_GATHER_CHUNK
    if not cuda:
        # One thread per (world, element) in the CUDA kernel's summation order.
        @wp.kernel(module="unique", enable_backward=False)
        def serial_gather(
            moment: wp.array2d(dtype=wp.float32),
            force_velocity_slope: wp.array2d(dtype=wp.float32),
            entry_muscle: wp.array(dtype=wp.int32),
            elem_ids: wp.array(dtype=wp.int32),
            chunk_offsets: wp.array(dtype=wp.int32),
            chunk_pairs: wp.array(dtype=wp.int32),
            timestep: float,
            qderiv: wp.array2d(dtype=wp.float32),
        ):
            world, slot = wp.tid()
            acc = float(0.0)
            for c in range(chunk_offsets[slot], chunk_offsets[slot + 1]):
                total = float(0.0)
                for k in range(chunk):
                    packed = chunk_pairs[k * chunk_count + c]
                    i = packed & 0xFFFF
                    j = packed >> 16
                    if i < entry_count:
                        weighted = (
                            force_velocity_slope[world, entry_muscle[i]]
                            * moment[world, i]
                        )
                        total += weighted * moment[world, j]
                acc += total
            elem = elem_ids[slot]
            qderiv[world, elem] = qderiv[world, elem] + timestep * acc

        return serial_gather

    def shared_snippet(size: int) -> str:
        return (
            f"__shared__ float data[{size}];\n"
            f"wp::array_t<float> result(data, {size});\n"
            "return result;\n"
        )

    moment_snippet = shared_snippet(entry_count + 1)
    weighted_snippet = shared_snippet(entry_count + 1)
    partial_snippet = shared_snippet(chunk_count)

    @wp.func_native(snippet=moment_snippet)
    def shared_moment(dummy: int) -> wp.array(dtype=float): ...

    @wp.func_native(snippet=weighted_snippet)
    def shared_weighted(dummy: int) -> wp.array(dtype=float): ...

    @wp.func_native(snippet=partial_snippet)
    def shared_partial(dummy: int) -> wp.array(dtype=float): ...

    @wp.func_native(snippet="__syncthreads();")
    def block_sync(): ...

    # One block per world: the moment row and its slope-weighted copy are staged
    # in shared memory, each lane sums whole chunks (coalesced index loads), and
    # each element adds its chunk partials in order, so no atomics are needed.
    @wp.kernel(module="unique", enable_backward=False)
    def block_gather(
        moment: wp.array2d(dtype=wp.float32),
        force_velocity_slope: wp.array2d(dtype=wp.float32),
        entry_muscle: wp.array(dtype=wp.int32),
        elem_ids: wp.array(dtype=wp.int32),
        chunk_offsets: wp.array(dtype=wp.int32),
        chunk_pairs: wp.array(dtype=wp.int32),
        timestep: float,
        qderiv: wp.array2d(dtype=wp.float32),
    ):
        world, lane = wp.tid()
        width = wp.block_dim()
        m = shared_moment(0)
        weighted = shared_weighted(0)
        partial = shared_partial(0)
        for e in range(lane, entry_count, width):
            value = moment[world, e]
            m[e] = value
            weighted[e] = force_velocity_slope[world, entry_muscle[e]] * value
        if lane == 0:
            m[entry_count] = 0.0
            weighted[entry_count] = 0.0
        block_sync()
        for c in range(lane, chunk_count, width):
            total = float(0.0)
            for k in range(chunk):
                packed = chunk_pairs[k * chunk_count + c]
                total += weighted[packed & 0xFFFF] * m[packed >> 16]
            partial[c] = total
        block_sync()
        for slot in range(lane, elem_ids.shape[0], width):
            acc = float(0.0)
            for c in range(chunk_offsets[slot], chunk_offsets[slot + 1]):
                acc += partial[c]
            elem = elem_ids[slot]
            qderiv[world, elem] = qderiv[world, elem] + timestep * acc

    return block_gather


def apply_muscle_implicit_qderiv_gather(
    gather: MuscleImplicitQderivGather,
    moment: wp.array,
    force_velocity_slope: wp.array,
    timestep: float,
    qderiv: wp.array,
) -> None:
    """Add ``timestep * sum_m slope_m r_m r_m^T`` to qDeriv in M structure."""
    inputs = [
        moment,
        force_velocity_slope,
        gather.entry_muscle,
        gather.elem_ids,
        gather.chunk_offsets,
        gather.chunk_pairs,
        timestep,
    ]
    nworld = moment.shape[0]
    if gather.cuda:
        wp.launch_tiled(
            gather.kernel,
            dim=[nworld],
            inputs=inputs,
            outputs=[qderiv],
            block_dim=_QDERIV_GATHER_BLOCK_DIM,
        )
    else:
        wp.launch(
            gather.kernel,
            dim=(nworld, gather.elem_ids.shape[0]),
            inputs=inputs,
            outputs=[qderiv],
        )


@startup_nvtx_wrapped("metadata:build_muscle_dof_sparse_entries")
def build_muscle_dof_sparse_entries(
    torque_active_entry_offsets: torch.Tensor,
    torque_active_segment_idx: torch.Tensor,
    torque_active_muscle_idx: torch.Tensor,
    torque_active_start_coeff: torch.Tensor,
    torque_active_end_coeff: torch.Tensor,
    muscle_count: int,
    additional_muscle_dofs: torch.Tensor | None = None,
) -> MuscleDofSparseEntries:
    """Group active segments by muscle/DOF and build packed lower-triangle pairs."""
    if additional_muscle_dofs is not None and additional_muscle_dofs.numel():
        base = build_muscle_dof_sparse_entries(
            torque_active_entry_offsets,
            torque_active_segment_idx,
            torque_active_muscle_idx,
            torque_active_start_coeff,
            torque_active_end_coeff,
            muscle_count,
        )
        # A spline's driver need not move any body carrying the muscle. Such
        # a row has no rigid segments; its moment is supplied by virtual work.
        existing = {
            (int(muscle), int(dof)): entry
            for entry, (muscle, dof) in enumerate(
                zip(base.muscle_dof_muscle_idx, base.muscle_dof_torque_idx, strict=True)
            )
        }
        extra = additional_muscle_dofs.detach().cpu().to(torch.int64).reshape(-1, 2)
        torque_dim = max(int(torque_active_entry_offsets.numel()) - 1, 0)
        if bool(
            (
                (extra[:, 0] < 0)
                | (extra[:, 0] >= muscle_count)
                | (extra[:, 1] < 0)
                | (extra[:, 1] >= torque_dim)
            ).any()
        ):
            raise ValueError("Additional muscle/DOF entries are out of range.")
        keys = sorted(existing.keys() | {tuple(row) for row in extra.tolist()})
        if keys == list(existing):
            return base
        muscle_offsets = [0]
        segment_offsets = [0]
        segment_indices: list[int] = []
        for muscle in range(muscle_count):
            for key in (key for key in keys if key[0] == muscle):
                entry = existing.get(key)
                if entry is not None:
                    begin, end = base.muscle_dof_segment_offsets[
                        entry : entry + 2
                    ].tolist()
                    segment_indices.extend(range(begin, end))
                segment_offsets.append(len(segment_indices))
            muscle_offsets.append(len(segment_offsets) - 1)
        index = torch.tensor(segment_indices, dtype=torch.long)
        pairs = [
            (i, j, muscle)
            for muscle in range(muscle_count)
            for i in range(muscle_offsets[muscle], muscle_offsets[muscle + 1])
            for j in range(muscle_offsets[muscle], i + 1)
        ]
        return MuscleDofSparseEntries(
            muscle_dof_offsets=torch.tensor(muscle_offsets, dtype=torch.int32),
            muscle_dof_muscle_idx=torch.tensor(
                [key[0] for key in keys], dtype=torch.int32
            ),
            muscle_dof_torque_idx=torch.tensor(
                [key[1] for key in keys], dtype=torch.int32
            ),
            muscle_dof_segment_offsets=torch.tensor(segment_offsets, dtype=torch.int32),
            segment_idx=base.segment_idx[index],
            start_coeff=base.start_coeff[index],
            end_coeff=base.end_coeff[index],
            pair_entry_i=torch.tensor([p[0] for p in pairs], dtype=torch.int32),
            pair_entry_j=torch.tensor([p[1] for p in pairs], dtype=torch.int32),
            pair_muscle_idx=torch.tensor([p[2] for p in pairs], dtype=torch.int32),
        )
    offsets = torque_active_entry_offsets.detach().cpu().to(torch.long)
    segment_idx = torque_active_segment_idx.detach().cpu().to(torch.long)
    muscle_idx = torque_active_muscle_idx.detach().cpu().to(torch.long)
    start_coeff = torque_active_start_coeff.detach().cpu().to(torch.float32)
    end_coeff = torque_active_end_coeff.detach().cpu().to(torch.float32)
    torque_dim = max(int(offsets.numel()) - 1, 0)
    entry_count = int(segment_idx.numel())
    if torque_dim == 0 or entry_count == 0:
        empty_i32 = torch.zeros(0, dtype=torch.int32)
        return MuscleDofSparseEntries(
            muscle_dof_offsets=torch.zeros(muscle_count + 1, dtype=torch.int32),
            muscle_dof_muscle_idx=empty_i32.clone(),
            muscle_dof_torque_idx=empty_i32.clone(),
            muscle_dof_segment_offsets=torch.zeros(1, dtype=torch.int32),
            segment_idx=empty_i32.clone(),
            start_coeff=torch.zeros(0, dtype=torch.float32),
            end_coeff=torch.zeros(0, dtype=torch.float32),
            pair_entry_i=empty_i32.clone(),
            pair_entry_j=empty_i32.clone(),
            pair_muscle_idx=empty_i32.clone(),
        )

    counts = offsets[1:] - offsets[:-1]
    torque_idx = torch.repeat_interleave(
        torch.arange(torque_dim, dtype=torch.long),
        counts,
    )
    segment_stride = int(segment_idx.max().item()) + 1
    torque_stride = segment_stride
    muscle_stride = (torque_dim + 1) * torque_stride
    order = torch.argsort(
        muscle_idx * muscle_stride + torque_idx * torque_stride + segment_idx
    )
    sorted_muscle = muscle_idx.index_select(0, order)
    sorted_torque = torque_idx.index_select(0, order)
    sorted_segment = segment_idx.index_select(0, order)
    sorted_start = start_coeff.index_select(0, order)
    sorted_end = end_coeff.index_select(0, order)

    new_group = torch.ones(entry_count, dtype=torch.bool)
    if entry_count > 1:
        new_group[1:] = (sorted_muscle[1:] != sorted_muscle[:-1]) | (
            sorted_torque[1:] != sorted_torque[:-1]
        )
    group_start = new_group.nonzero(as_tuple=False).flatten()
    group_end = torch.cat(
        (group_start[1:], torch.tensor([entry_count], dtype=torch.long))
    )
    group_muscle = sorted_muscle.index_select(0, group_start)
    group_torque = sorted_torque.index_select(0, group_start)

    muscle_group_counts = torch.bincount(group_muscle, minlength=muscle_count)
    muscle_dof_offsets = torch.zeros(muscle_count + 1, dtype=torch.int32)
    muscle_dof_offsets[1:] = torch.cumsum(muscle_group_counts, dim=0).to(torch.int32)
    muscle_dof_segment_offsets = torch.cat((group_start, group_end[-1:])).to(
        torch.int32
    )

    pair_i: list[int] = []
    pair_j: list[int] = []
    pair_muscle: list[int] = []
    group_offsets = muscle_dof_offsets.to(torch.long)
    for muscle in range(muscle_count):
        begin = int(group_offsets[muscle].item())
        end = int(group_offsets[muscle + 1].item())
        for entry_i in range(begin, end):
            for entry_j in range(begin, entry_i + 1):
                pair_i.append(entry_i)
                pair_j.append(entry_j)
                pair_muscle.append(muscle)

    return MuscleDofSparseEntries(
        muscle_dof_offsets=muscle_dof_offsets,
        muscle_dof_muscle_idx=group_muscle.to(torch.int32),
        muscle_dof_torque_idx=group_torque.to(torch.int32),
        muscle_dof_segment_offsets=muscle_dof_segment_offsets,
        segment_idx=sorted_segment.to(torch.int32),
        start_coeff=sorted_start,
        end_coeff=sorted_end,
        pair_entry_i=torch.tensor(pair_i, dtype=torch.int32),
        pair_entry_j=torch.tensor(pair_j, dtype=torch.int32),
        pair_muscle_idx=torch.tensor(pair_muscle, dtype=torch.int32),
    )


@startup_nvtx_wrapped("metadata:build_dense_muscle_dof_entries")
def build_dense_muscle_dof_entries(
    muscle_count: int,
    torque_dim: int,
) -> MuscleDofSparseEntries:
    """Build packed muscle/DOF entries for geometry with dynamic body routing."""
    muscle_count = max(int(muscle_count), 0)
    torque_dim = max(int(torque_dim), 0)
    entry_count = muscle_count * torque_dim
    if torque_dim > 0:
        muscle_dof_offsets = torch.arange(
            0, entry_count + 1, torque_dim, dtype=torch.int32
        )
    else:
        muscle_dof_offsets = torch.zeros(muscle_count + 1, dtype=torch.int32)
    muscle_idx = torch.repeat_interleave(
        torch.arange(muscle_count, dtype=torch.int32), torque_dim
    )
    torque_idx = torch.arange(torque_dim, dtype=torch.int32).repeat(muscle_count)
    local_pairs = torch.tril_indices(torque_dim, torque_dim)
    pairs_per_muscle = int(local_pairs.shape[1])
    pair_muscle = torch.repeat_interleave(
        torch.arange(muscle_count, dtype=torch.int32),
        pairs_per_muscle,
    )
    pair_base = pair_muscle * torque_dim
    pair_i = pair_base + local_pairs[0].to(torch.int32).repeat(muscle_count)
    pair_j = pair_base + local_pairs[1].to(torch.int32).repeat(muscle_count)
    empty_i32 = torch.zeros(0, dtype=torch.int32)
    return MuscleDofSparseEntries(
        muscle_dof_offsets=muscle_dof_offsets,
        muscle_dof_muscle_idx=muscle_idx,
        muscle_dof_torque_idx=torque_idx,
        muscle_dof_segment_offsets=torch.zeros(entry_count + 1, dtype=torch.int32),
        segment_idx=empty_i32,
        start_coeff=torch.zeros(0, dtype=torch.float32),
        end_coeff=torch.zeros(0, dtype=torch.float32),
        pair_entry_i=pair_i,
        pair_entry_j=pair_j,
        pair_muscle_idx=pair_muscle,
    )


@startup_nvtx_wrapped("metadata:build_torque_sparse_lbs_segment_entries")
def build_torque_sparse_lbs_segment_entries(
    torque_dof_subtree_mask: torch.Tensor,
    waypoint_attachment_start: torch.Tensor,
    waypoint_attachment_body: torch.Tensor,
    apply_segment_start_idx: torch.Tensor,
    apply_segment_end_idx: torch.Tensor,
    apply_segment_muscle_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torque_dim = int(torque_dof_subtree_mask.shape[0])
    mask_cpu = torque_dof_subtree_mask.detach().cpu().to(torch.bool)
    attach_start = waypoint_attachment_start.detach().cpu().to(torch.long)
    attach_body = waypoint_attachment_body.detach().cpu().to(torch.long)
    seg_start = apply_segment_start_idx.detach().cpu().to(torch.long)
    seg_end = apply_segment_end_idx.detach().cpu().to(torch.long)
    seg_muscle = apply_segment_muscle_idx.detach().cpu().to(torch.long)
    waypoint_count = max(0, int(attach_start.numel()) - 1)
    segment_count = int(seg_start.numel())

    offsets: list[int] = [0]
    torque_idx_parts: list[torch.Tensor] = []
    segment_idx_parts: list[torch.Tensor] = []
    muscle_idx_parts: list[torch.Tensor] = []
    if (
        torque_dim <= 0
        or waypoint_count <= 0
        or segment_count <= 0
        or int(attach_body.numel()) == 0
    ):
        return (
            torch.zeros(torque_dim + 1, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
        )

    attach_counts = attach_start[1:] - attach_start[:-1]
    waypoint_for_attachment = torch.repeat_interleave(
        torch.arange(waypoint_count, dtype=torch.long),
        attach_counts,
    )
    segment_sort_stride = segment_count + 1
    for torque_idx in range(torque_dim):
        active_attachment = mask_cpu[torque_idx].index_select(0, attach_body)
        if not bool(active_attachment.any().item()):
            offsets.append(offsets[-1])
            continue
        waypoint_active = torch.zeros(waypoint_count, dtype=torch.bool)
        waypoint_active[waypoint_for_attachment[active_attachment]] = True
        active_segment = waypoint_active.index_select(
            0, seg_start
        ) | waypoint_active.index_select(0, seg_end)
        active_segment_idx = active_segment.nonzero(as_tuple=False).flatten()
        if int(active_segment_idx.numel()) == 0:
            offsets.append(offsets[-1])
            continue

        active_muscle = seg_muscle.index_select(0, active_segment_idx)
        order = torch.argsort(active_muscle * segment_sort_stride + active_segment_idx)
        active_segment_idx = active_segment_idx.index_select(0, order).to(torch.int32)
        active_muscle = active_muscle.index_select(0, order).to(torch.int32)
        segment_idx_parts.append(active_segment_idx)
        muscle_idx_parts.append(active_muscle)
        torque_idx_parts.append(
            torch.full_like(active_segment_idx, int(torque_idx), dtype=torch.int32)
        )
        offsets.append(offsets[-1] + int(active_segment_idx.numel()))

    return (
        torch.tensor(offsets, dtype=torch.int32),
        torch.cat(torque_idx_parts)
        if torque_idx_parts
        else torch.zeros(0, dtype=torch.int32),
        torch.cat(segment_idx_parts)
        if segment_idx_parts
        else torch.zeros(0, dtype=torch.int32),
        torch.cat(muscle_idx_parts)
        if muscle_idx_parts
        else torch.zeros(0, dtype=torch.int32),
    )


@startup_nvtx_wrapped("metadata:build_lbs_active_attachment_lists")
def build_lbs_active_attachment_lists(
    *,
    dof_subtree_mask: torch.Tensor,
    attachment_start: torch.Tensor,
    attachment_body: torch.Tensor,
    attachment_local: torch.Tensor,
    attachment_weight: torch.Tensor,
    root_dof_count: int,
    torque_dim: int,
    waypoint_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if waypoint_count <= 0 or torque_dim <= 0 or int(attachment_body.numel()) == 0:
        return (
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(0, dtype=torch.int32),
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros(0, dtype=torch.float32),
        )

    mask_cpu = dof_subtree_mask.detach().cpu().float()
    start_cpu = attachment_start.detach().cpu().long()
    body_cpu = attachment_body.detach().cpu().long()
    local_cpu = attachment_local.detach().cpu().float()
    weight_cpu = attachment_weight.detach().cpu().float()
    counts_per_waypoint = start_cpu[1:] - start_cpu[:-1]
    waypoint_for_attachment = torch.repeat_interleave(
        torch.arange(waypoint_count, dtype=torch.long),
        counts_per_waypoint,
    )

    offsets = torch.empty(torque_dim * waypoint_count + 1, dtype=torch.int32)
    offsets[0] = 0
    body_chunks: list[torch.Tensor] = []
    local_chunks: list[torch.Tensor] = []
    coeff_chunks: list[torch.Tensor] = []
    cursor = 0
    row = 0
    for torque_idx in range(torque_dim):
        full_torque_idx = torque_idx + root_dof_count
        coeff = mask_cpu[full_torque_idx, body_cpu] * weight_cpu
        active = coeff != 0.0
        active_wp = waypoint_for_attachment[active]
        active_counts = torch.bincount(
            active_wp,
            minlength=waypoint_count,
        ).to(torch.int32)
        row_end = row + waypoint_count
        offsets[row + 1 : row_end + 1] = cursor + torch.cumsum(
            active_counts,
            dim=0,
        )
        active_count = int(active_counts.sum().item())
        if active_count > 0:
            body_chunks.append(body_cpu[active].to(torch.int32))
            local_chunks.append(local_cpu[active])
            coeff_chunks.append(coeff[active])
        cursor += active_count
        row = row_end

    active_body = (
        torch.cat(body_chunks).to(torch.int32)
        if body_chunks
        else torch.zeros(0, dtype=torch.int32)
    )
    active_local = (
        torch.cat(local_chunks).to(torch.float32)
        if local_chunks
        else torch.zeros((0, 3), dtype=torch.float32)
    )
    active_coeff_weight = (
        torch.cat(coeff_chunks).to(torch.float32)
        if coeff_chunks
        else torch.zeros(0, dtype=torch.float32)
    )
    return offsets, active_body, active_local, active_coeff_weight


@startup_nvtx_wrapped("metadata:build_lbs_active_local_coeff_vec4")
def build_lbs_active_local_coeff_vec4(
    active_attach_local: torch.Tensor,
    active_attach_coeff_weight: torch.Tensor,
) -> torch.Tensor:
    local = active_attach_local.to(dtype=torch.float32)
    coeff = active_attach_coeff_weight.to(
        device=local.device,
        dtype=torch.float32,
    )
    if local.ndim != 2 or int(local.shape[1]) != 3:
        raise ValueError(
            f"active_attach_local must have shape (N, 3), got {tuple(local.shape)}."
        )
    if coeff.ndim != 1 or int(coeff.shape[0]) != int(local.shape[0]):
        raise ValueError(
            "active_attach_coeff_weight must have shape (N,), "
            f"got {tuple(coeff.shape)} for local shape {tuple(local.shape)}."
        )
    return torch.cat([local, coeff[:, None]], dim=1).contiguous()


@startup_nvtx_wrapped("metadata:build_lbs_active_idx_body_vec2")
def build_lbs_active_idx_body_vec2(
    active_attach_waypoint_idx: torch.Tensor,
    active_attach_body: torch.Tensor,
) -> torch.Tensor:
    waypoint_idx = active_attach_waypoint_idx.to(dtype=torch.int32)
    body = active_attach_body.to(device=waypoint_idx.device, dtype=torch.int32)
    if waypoint_idx.ndim != 1:
        raise ValueError(
            "active_attach_waypoint_idx must have shape (N,), "
            f"got {tuple(waypoint_idx.shape)}."
        )
    if body.ndim != 1 or int(body.shape[0]) != int(waypoint_idx.shape[0]):
        raise ValueError(
            "active_attach_body must have shape (N,), "
            f"got {tuple(body.shape)} for waypoint_idx shape {tuple(waypoint_idx.shape)}."
        )
    return torch.stack([waypoint_idx, body], dim=1).contiguous()


@startup_nvtx_wrapped("metadata:build_lbs_waypoint_attachment_fixed2_metadata")
def build_lbs_waypoint_attachment_fixed2_metadata(
    *,
    attachment_start: torch.Tensor,
    attachment_body: torch.Tensor,
    attachment_local: torch.Tensor,
    attachment_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = attachment_body.device
    starts = attachment_start.detach().cpu().to(torch.long)
    bodies = attachment_body.detach().cpu().to(torch.int32)
    locals_cpu = attachment_local.detach().cpu().to(torch.float32)
    weights = attachment_weight.detach().cpu().to(torch.float32)
    waypoint_count = max(0, int(starts.numel()) - 1)
    if waypoint_count == 0:
        return (
            torch.zeros((0, 2), dtype=torch.int32, device=device),
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros((0, 4), dtype=torch.float32, device=device),
        )

    counts = starts[1:] - starts[:-1]
    invalid = (counts < 1) | (counts > 2)
    if bool(invalid.any().item()):
        waypoint_idx = int(invalid.nonzero(as_tuple=False)[0].item())
        raise ValueError(
            "fixed2 waypoint attachment metadata requires one or two "
            f"attachments per waypoint, got {int(counts[waypoint_idx].item())} "
            f"for waypoint {waypoint_idx}."
        )

    slot0 = starts[:-1]
    has_slot1 = counts > 1
    slot1 = torch.where(has_slot1, slot0 + 1, slot0)

    body2 = torch.stack(
        [
            bodies.index_select(0, slot0),
            bodies.index_select(0, slot1),
        ],
        dim=1,
    ).contiguous()
    local_weight0 = torch.cat(
        [
            locals_cpu.index_select(0, slot0),
            weights.index_select(0, slot0)[:, None],
        ],
        dim=1,
    ).contiguous()
    local_weight1 = torch.zeros((waypoint_count, 4), dtype=torch.float32)
    slot1_rows = has_slot1.nonzero(as_tuple=False).flatten()
    if int(slot1_rows.numel()) > 0:
        slot1_attach = slot1.index_select(0, slot1_rows)
        local_weight1.index_copy_(
            0,
            slot1_rows,
            torch.cat(
                [
                    locals_cpu.index_select(0, slot1_attach),
                    weights.index_select(0, slot1_attach)[:, None],
                ],
                dim=1,
            ),
        )

    return (
        body2.to(device=device),
        local_weight0.to(device=device),
        local_weight1.to(device=device),
    )


@startup_nvtx_wrapped("metadata:build_lbs_waypoint_force_metadata")
def build_lbs_waypoint_force_metadata(
    *,
    apply_segment_start_idx: torch.Tensor,
    apply_segment_end_idx: torch.Tensor,
    waypoint_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = apply_segment_start_idx.device
    waypoint_count = int(waypoint_count)
    if waypoint_count <= 0:
        return (
            torch.zeros((0, 2), dtype=torch.int32, device=device),
            torch.zeros((0, 2), dtype=torch.int32, device=device),
        )

    segment_idx2 = torch.full((waypoint_count, 2), -1, dtype=torch.int32)
    segment_sign2 = torch.zeros((waypoint_count, 2), dtype=torch.int32)
    starts = apply_segment_start_idx.detach().cpu().to(torch.long)
    ends = apply_segment_end_idx.detach().cpu().to(torch.long)
    segment_count = int(starts.numel())
    if segment_count == 0:
        return segment_idx2.to(device=device), segment_sign2.to(device=device)

    waypoint_idx = torch.cat([starts, ends], dim=0)
    invalid = (waypoint_idx < 0) | (waypoint_idx >= waypoint_count)
    if bool(invalid.any().item()):
        event_idx = int(invalid.nonzero(as_tuple=False)[0].item())
        segment_idx = event_idx % segment_count
        raise ValueError(
            "segment waypoint index is out of range for waypoint_force metadata: "
            f"segment={segment_idx} waypoint={int(waypoint_idx[event_idx].item())} "
            f"waypoint_count={waypoint_count}."
        )

    counts = torch.bincount(waypoint_idx, minlength=waypoint_count)
    if bool((counts > 2).any().item()):
        waypoint = int((counts > 2).nonzero(as_tuple=False)[0].item())
        raise ValueError(
            "waypoint_force metadata currently supports at most two incident "
            f"segments per waypoint; waypoint {waypoint} has more."
        )

    segments = torch.arange(segment_count, dtype=torch.int32)
    event_segment = torch.cat([segments, segments], dim=0)
    event_sign = torch.cat(
        [
            torch.ones(segment_count, dtype=torch.int32),
            -torch.ones(segment_count, dtype=torch.int32),
        ],
        dim=0,
    )
    event_order = torch.cat(
        [
            2 * torch.arange(segment_count, dtype=torch.long),
            2 * torch.arange(segment_count, dtype=torch.long) + 1,
        ],
        dim=0,
    )
    order = torch.argsort(waypoint_idx * (2 * segment_count + 1) + event_order)
    sorted_waypoint = waypoint_idx.index_select(0, order)
    sorted_segment = event_segment.index_select(0, order)
    sorted_sign = event_sign.index_select(0, order)
    slot = torch.arange(int(order.numel()), dtype=torch.long) - torch.repeat_interleave(
        torch.cumsum(counts, dim=0) - counts,
        counts,
    )
    segment_idx2[sorted_waypoint, slot] = sorted_segment
    segment_sign2[sorted_waypoint, slot] = sorted_sign
    return segment_idx2.to(device=device), segment_sign2.to(device=device)


@startup_nvtx_wrapped("metadata:build_lbs_active_attachment_projection_metadata")
def build_lbs_active_attachment_projection_metadata(
    *,
    active_attach_offsets: torch.Tensor,
    torque_dim: int,
    waypoint_count: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = active_attach_offsets.device
    torque_dim = int(torque_dim)
    waypoint_count = int(waypoint_count)
    chunk_size = max(1, int(chunk_size))
    expected_offsets = torque_dim * waypoint_count + 1
    if int(active_attach_offsets.numel()) != expected_offsets:
        raise ValueError(
            "active_attach_offsets shape does not match torque_dim * waypoint_count + 1: "
            f"got {int(active_attach_offsets.numel())}, expected {expected_offsets}."
        )

    offsets_cpu = active_attach_offsets.detach().cpu().to(torch.int64)
    counts = offsets_cpu[1:] - offsets_cpu[:-1]
    waypoint_rows = torch.arange(waypoint_count, dtype=torch.int32).repeat(torque_dim)
    active_waypoint_idx = torch.repeat_interleave(waypoint_rows, counts).contiguous()

    row_start = torch.arange(torque_dim, dtype=torch.long) * waypoint_count
    active_begin = offsets_cpu.index_select(0, row_start)
    active_end = offsets_cpu.index_select(0, row_start + waypoint_count)
    active_count = active_end - active_begin
    chunks_per_torque = (active_count + chunk_size - 1) // chunk_size
    chunk_offsets_tensor = torch.zeros(torque_dim + 1, dtype=torch.int32)
    chunk_offsets_tensor[1:] = torch.cumsum(chunks_per_torque, dim=0).to(torch.int32)
    chunk_count = int(chunks_per_torque.sum().item())
    if chunk_count > 0:
        chunk_torque = torch.repeat_interleave(
            torch.arange(torque_dim, dtype=torch.long),
            chunks_per_torque,
        )
        chunk_local = torch.arange(
            chunk_count, dtype=torch.long
        ) - torch.repeat_interleave(
            chunk_offsets_tensor[:-1].to(torch.long),
            chunks_per_torque,
        )
        chunk_begin = (
            active_begin.index_select(0, chunk_torque) + chunk_local * chunk_size
        )
        chunk_end = torch.minimum(
            chunk_begin + chunk_size,
            active_end.index_select(0, chunk_torque),
        )
        chunk_info_tensor = torch.stack(
            [
                chunk_torque.to(torch.int32),
                chunk_begin.to(torch.int32),
                chunk_end.to(torch.int32),
            ],
            dim=1,
        ).contiguous()
    else:
        chunk_info_tensor = torch.zeros((0, 3), dtype=torch.int32)
    return (
        active_waypoint_idx.to(device=device),
        chunk_offsets_tensor.to(device=device),
        chunk_info_tensor.to(device=device),
    )


@startup_nvtx_wrapped("metadata:build_lbs_body_wrench_segment_event_metadata")
def build_lbs_body_wrench_segment_event_metadata(
    *,
    apply_segment_start_idx: torch.Tensor,
    apply_segment_end_idx: torch.Tensor,
    apply_segment_muscle_idx: torch.Tensor,
    waypoint_attachment_start: torch.Tensor,
    waypoint_attachment_body: torch.Tensor,
    waypoint_attachment_local: torch.Tensor,
    waypoint_attachment_weight: torch.Tensor,
    body_count: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = waypoint_attachment_body.device
    body_count = int(body_count)
    chunk_size = max(1, int(chunk_size))
    if body_count <= 0:
        return (
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.zeros((0, 3), dtype=torch.int32, device=device),
            torch.zeros((0, 4), dtype=torch.int32, device=device),
            torch.zeros((0, 4), dtype=torch.float32, device=device),
        )

    starts = apply_segment_start_idx.detach().cpu().to(torch.long)
    ends = apply_segment_end_idx.detach().cpu().to(torch.long)
    muscles = apply_segment_muscle_idx.detach().cpu().to(torch.long)
    attach_start = waypoint_attachment_start.detach().cpu().to(torch.long)
    attach_body = waypoint_attachment_body.detach().cpu().to(torch.long)
    attach_local = waypoint_attachment_local.detach().cpu().to(torch.float32)
    attach_weight = waypoint_attachment_weight.detach().cpu().to(torch.float32)
    segment_count = int(starts.numel())
    if segment_count == 0:
        return (
            torch.zeros(body_count + 1, dtype=torch.int32, device=device),
            torch.zeros((0, 3), dtype=torch.int32, device=device),
            torch.zeros((0, 4), dtype=torch.int32, device=device),
            torch.zeros((0, 4), dtype=torch.float32, device=device),
        )

    segment_idx = torch.arange(segment_count, dtype=torch.long)
    endpoint_segment = segment_idx.repeat_interleave(2)
    endpoint_waypoint = torch.stack([starts, ends], dim=1).reshape(-1)
    endpoint_sign = torch.tensor([1.0, -1.0], dtype=torch.float32).repeat(segment_count)
    endpoint_begin = attach_start.index_select(0, endpoint_waypoint)
    endpoint_end = attach_start.index_select(0, endpoint_waypoint + 1)
    endpoint_counts = endpoint_end - endpoint_begin
    attachment_event_count = int(endpoint_counts.sum().item())
    if attachment_event_count == 0:
        return (
            torch.zeros(body_count + 1, dtype=torch.int32, device=device),
            torch.zeros((0, 3), dtype=torch.int32, device=device),
            torch.zeros((0, 4), dtype=torch.int32, device=device),
            torch.zeros((0, 4), dtype=torch.float32, device=device),
        )

    endpoint_offsets = torch.cumsum(endpoint_counts, dim=0) - endpoint_counts
    attachment_relative = torch.arange(
        attachment_event_count, dtype=torch.long
    ) - torch.repeat_interleave(endpoint_offsets, endpoint_counts)
    attachment_idx = (
        torch.repeat_interleave(endpoint_begin, endpoint_counts) + attachment_relative
    )
    event_segment = torch.repeat_interleave(endpoint_segment, endpoint_counts)
    event_body = attach_body.index_select(0, attachment_idx)
    invalid_body = (event_body < 0) | (event_body >= body_count)
    if bool(invalid_body.any().item()):
        bad_idx = int(invalid_body.nonzero(as_tuple=False).flatten()[0].item())
        bad = int(event_body[bad_idx].item())
        raise ValueError(
            "waypoint attachment body is out of range for body-wrench "
            f"metadata: body={bad} body_count={body_count}."
        )
    event_coeff = torch.repeat_interleave(endpoint_sign, endpoint_counts)
    event_coeff = event_coeff * attach_weight.index_select(0, attachment_idx)
    event_values = torch.cat(
        [
            attach_local.index_select(0, attachment_idx) * event_coeff[:, None],
            event_coeff[:, None],
        ],
        dim=1,
    )

    group_key = event_segment * body_count + event_body
    unique_key, inverse = torch.unique(group_key, sorted=True, return_inverse=True)
    local_force_coeff_tensor = torch.zeros(
        (int(unique_key.numel()), 4), dtype=torch.float32
    )
    local_force_coeff_tensor.index_add_(0, inverse, event_values)
    grouped_segment = unique_key // body_count
    grouped_body = unique_key % body_count
    sort_key = grouped_body * (segment_count + 1) + grouped_segment
    order = torch.argsort(sort_key)
    grouped_segment = grouped_segment.index_select(0, order)
    grouped_body = grouped_body.index_select(0, order)
    local_force_coeff_tensor = local_force_coeff_tensor.index_select(0, order)
    packed_tensor = torch.stack(
        [
            starts.index_select(0, grouped_segment).to(torch.int32),
            ends.index_select(0, grouped_segment).to(torch.int32),
            muscles.index_select(0, grouped_segment).to(torch.int32),
            torch.zeros_like(grouped_segment, dtype=torch.int32),
        ],
        dim=1,
    ).contiguous()

    body_counts = torch.bincount(grouped_body, minlength=body_count).tolist()
    body_event_offsets = [0]
    for count in body_counts:
        body_event_offsets.append(body_event_offsets[-1] + int(count))

    chunk_offsets: list[int] = [0]
    chunk_info: list[list[int]] = []
    for body_idx in range(body_count):
        body_begin = body_event_offsets[body_idx]
        body_end = body_event_offsets[body_idx + 1]
        for chunk_begin in range(body_begin, body_end, chunk_size):
            chunk_info.append(
                [
                    body_idx,
                    chunk_begin,
                    min(chunk_begin + chunk_size, body_end),
                ]
            )
        chunk_offsets.append(len(chunk_info))

    chunk_offsets_tensor = torch.tensor(chunk_offsets, dtype=torch.int32)
    chunk_info_tensor = (
        torch.tensor(chunk_info, dtype=torch.int32)
        if chunk_info
        else torch.zeros((0, 3), dtype=torch.int32)
    )
    return (
        chunk_offsets_tensor.to(device=device),
        chunk_info_tensor.to(device=device),
        packed_tensor.to(device=device),
        local_force_coeff_tensor.to(device=device),
    )


@startup_nvtx_wrapped("metadata:build_lbs_native_body_wrench_tile_metadata")
def build_lbs_native_body_wrench_tile_metadata(
    *,
    muscle_segment_offsets: torch.Tensor,
    body_event_chunk_info: torch.Tensor,
    body_event_start_muscle: torch.Tensor,
    body_event_local_force_coeff: torch.Tensor,
    muscle_count: int,
    body_count: int,
    muscle_cap: int,
    segment_cap: int,
    body_event_cap: int,
    body_cap: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    device = body_event_start_muscle.device
    muscle_count = int(muscle_count)
    body_count = int(body_count)
    muscle_cap = max(1, int(muscle_cap))
    segment_cap = max(1, int(segment_cap))
    body_event_cap = max(1, int(body_event_cap))
    body_cap = max(1, int(body_cap))
    waypoint_cap = muscle_cap + segment_cap

    offsets = muscle_segment_offsets.detach().cpu().to(torch.int64)
    start_muscle = body_event_start_muscle.detach().cpu().to(torch.int64)
    local_force_coeff = body_event_local_force_coeff.detach().cpu().to(torch.float32)
    chunk_info = body_event_chunk_info.detach().cpu().to(torch.int64)

    event_count = int(start_muscle.shape[0])
    if event_count == 0:
        return (
            torch.zeros((0, 4), dtype=torch.int32, device=device),
            torch.zeros((0, 4), dtype=torch.int32, device=device),
            torch.zeros((0,), dtype=torch.int32, device=device),
            torch.zeros((0,), dtype=torch.int32, device=device),
            torch.zeros((0, 4), dtype=torch.int32, device=device),
            torch.zeros((0, 4), dtype=torch.float32, device=device),
            torch.zeros(body_count + 1, dtype=torch.int32, device=device),
        )
    chunk_counts = chunk_info[:, 2] - chunk_info[:, 1]
    event_body = torch.repeat_interleave(chunk_info[:, 0], chunk_counts)
    if int(event_body.numel()) != event_count:
        raise ValueError(
            "body event chunk metadata does not cover the native tile events: "
            f"events={event_count}, chunked={int(event_body.numel())}."
        )

    segment_counts = (offsets[1:] - offsets[:-1]).to(torch.int64)
    bodies_by_muscle: list[set[int]] = [set() for _ in range(muscle_count)]
    event_muscle = start_muscle[:, 1]
    event_counts = torch.bincount(event_muscle, minlength=muscle_count).to(torch.int64)
    unique_muscle_body = torch.unique(
        event_muscle * body_count + event_body,
        sorted=True,
    )
    unique_muscle = unique_muscle_body // body_count
    unique_body = unique_muscle_body % body_count
    for muscle_idx, body_idx in zip(
        unique_muscle.tolist(),
        unique_body.tolist(),
        strict=True,
    ):
        bodies_by_muscle[int(muscle_idx)].add(int(body_idx))

    event_order = torch.arange(event_count, dtype=torch.int64)
    muscle_order = torch.argsort(event_muscle * (event_count + 1) + event_order)
    event_muscle_sorted = event_muscle.index_select(0, muscle_order)
    event_body_sorted = event_body.index_select(0, muscle_order)
    event_start_wp_sorted = start_muscle[:, 0].index_select(0, muscle_order)
    local_force_coeff_sorted = local_force_coeff.index_select(0, muscle_order)
    event_offsets = torch.zeros(muscle_count + 1, dtype=torch.int64)
    event_offsets[1:] = torch.cumsum(event_counts, dim=0)

    tile_ranges: list[tuple[int, int, int, set[int], int, int]] = []
    muscle_idx = 0
    while muscle_idx < muscle_count:
        begin = muscle_idx
        segments = 0
        events = 0
        bodies: set[int] = set()
        while muscle_idx < muscle_count:
            next_segments = segments + int(segment_counts[muscle_idx].item())
            next_events = events + int(event_counts[muscle_idx].item())
            next_bodies = bodies | bodies_by_muscle[muscle_idx]
            next_muscles = muscle_idx - begin + 1
            would_exceed = (
                next_muscles > muscle_cap
                or next_segments > segment_cap
                or next_events > body_event_cap
                or len(next_bodies) > body_cap
            )
            if muscle_idx > begin and would_exceed:
                break
            segments = next_segments
            events = next_events
            bodies = next_bodies
            muscle_idx += 1

        end = muscle_idx
        waypoint_count = segments + (end - begin)
        if waypoint_count > waypoint_cap:
            raise ValueError(
                "native body-wrench tile waypoint cap is too small: "
                f"tile=({begin},{end}) waypoints={waypoint_count} cap={waypoint_cap}."
            )
        if segments > segment_cap:
            raise ValueError(
                "native body-wrench tile segment cap is too small: "
                f"tile=({begin},{end}) segments={segments} cap={segment_cap}."
            )
        event_begin = int(event_offsets[begin])
        event_end = int(event_offsets[end])
        if len(bodies) <= body_cap and events <= body_event_cap:
            tile_ranges.append((begin, end, segments, bodies, event_begin, event_end))
        else:
            # One muscle may span more bodies/events than fit in a native tile.
            # Repeat its complete length evaluation, partition only its force
            # events, and let the existing body-slot reduction sum the tiles.
            # Ordinary tiles retain their previous ordering and arithmetic.
            cursor = event_begin
            while cursor < event_end:
                chunk_begin = cursor
                chunk_bodies: set[int] = set()
                while cursor < event_end and cursor - chunk_begin < body_event_cap:
                    body = int(event_body_sorted[cursor])
                    if body not in chunk_bodies and len(chunk_bodies) == body_cap:
                        break
                    chunk_bodies.add(body)
                    cursor += 1
                tile_ranges.append(
                    (begin, end, segments, chunk_bodies, chunk_begin, cursor)
                )

    tile_info_rows: list[list[int]] = []
    tile_extra_rows: list[list[int]] = []
    tile_body_ids: list[int] = []
    tile_event_parts: list[torch.Tensor] = []
    tile_event_coeff_parts: list[torch.Tensor] = []
    tile_body_row_body: list[int] = []
    event_cursor = 0

    for begin, end, segments, bodies, event_slice_begin, event_slice_end in tile_ranges:
        local_bodies = sorted(bodies)
        body_begin = len(tile_body_ids)
        tile_body_ids.extend(local_bodies)
        tile_body_row_body.extend(local_bodies)
        event_begin = event_cursor
        waypoint_begin = int(offsets[begin].item()) + begin
        waypoint_count = segments + (end - begin)
        if event_slice_end > event_slice_begin:
            body_to_slot = torch.full((body_count,), -1, dtype=torch.int64)
            if local_bodies:
                body_to_slot[torch.tensor(local_bodies, dtype=torch.long)] = (
                    torch.arange(len(local_bodies), dtype=torch.int64)
                )
            event_body_slice = event_body_sorted[event_slice_begin:event_slice_end]
            body_slot = body_to_slot.index_select(0, event_body_slice)
            # The kernel reduces each warp's events by contiguous body-slot
            # runs, so a tile's events are ordered by slot.
            slot_order = torch.argsort(body_slot, stable=True)
            body_slot = body_slot.index_select(0, slot_order)
            local_muscle = (
                event_muscle_sorted[event_slice_begin:event_slice_end] - begin
            ).index_select(0, slot_order)
            local_start_wp = (
                event_start_wp_sorted[event_slice_begin:event_slice_end]
                - waypoint_begin
            ).index_select(0, slot_order)
            tile_event_parts.append(
                (
                    local_start_wp.to(torch.int64)
                    | (local_muscle.to(torch.int64) << 9)
                    | (body_slot.to(torch.int64) << 14)
                )
                .to(torch.int32)
                .contiguous()
            )
            tile_event_coeff_parts.append(
                local_force_coeff_sorted[
                    event_slice_begin:event_slice_end
                ].index_select(0, slot_order)
            )
        event_end = event_begin + (event_slice_end - event_slice_begin)
        event_cursor = event_end
        tile_info_rows.append([begin, end, body_begin, event_begin])
        tile_extra_rows.append(
            [waypoint_begin, waypoint_count, len(local_bodies), event_end]
        )

    body_slots: list[list[int]] = [[] for _ in range(body_count)]
    for flat_slot, body_idx in enumerate(tile_body_row_body):
        body_slots[body_idx].append(flat_slot)
    tile_body_output_idx = [0] * len(tile_body_ids)
    body_slot_offsets: list[int] = [0]
    output_idx = 0
    for slots in body_slots:
        for flat_slot in slots:
            tile_body_output_idx[flat_slot] = output_idx
            output_idx += 1
        body_slot_offsets.append(output_idx)

    tile_info = (
        torch.tensor(tile_info_rows, dtype=torch.int32)
        if tile_info_rows
        else torch.zeros((0, 4), dtype=torch.int32)
    )
    tile_extra_info = (
        torch.tensor(tile_extra_rows, dtype=torch.int32)
        if tile_extra_rows
        else torch.zeros((0, 4), dtype=torch.int32)
    )
    tile_body_ids_tensor = (
        torch.tensor(tile_body_ids, dtype=torch.int32)
        if tile_body_ids
        else torch.zeros((0,), dtype=torch.int32)
    )
    tile_body_output_idx_tensor = (
        torch.tensor(tile_body_output_idx, dtype=torch.int32)
        if tile_body_output_idx
        else torch.zeros((0,), dtype=torch.int32)
    )
    tile_event_tensor = (
        torch.cat(tile_event_parts, dim=0).to(torch.int32)
        if tile_event_parts
        else torch.zeros((0,), dtype=torch.int32)
    )
    tile_event_coeff_tensor = (
        torch.cat(tile_event_coeff_parts, dim=0).to(torch.float32)
        if tile_event_coeff_parts
        else torch.zeros((0, 4), dtype=torch.float32)
    )
    body_slot_offsets_tensor = torch.tensor(body_slot_offsets, dtype=torch.int32)
    return (
        tile_info.to(device=device),
        tile_extra_info.to(device=device),
        tile_body_ids_tensor.to(device=device),
        tile_body_output_idx_tensor.to(device=device),
        tile_event_tensor.to(device=device),
        tile_event_coeff_tensor.to(device=device),
        body_slot_offsets_tensor.to(device=device),
    )


@dataclass(frozen=True)
class LBSNativeBodyWrenchSortedMetadata:
    muscle_order: torch.Tensor
    muscle_segment_offsets: torch.Tensor
    muscle_force_length_coeff: torch.Tensor
    waypoint_attachment_body2: torch.Tensor
    waypoint_attachment_local_weight0: torch.Tensor
    waypoint_attachment_local_weight1: torch.Tensor
    tile_info: torch.Tensor
    tile_extra_info: torch.Tensor
    tile_body_ids: torch.Tensor
    tile_body_output_idx: torch.Tensor
    tile_event_start_slot_muscle: torch.Tensor
    tile_event_local_force_coeff: torch.Tensor
    tile_body_slot_offsets: torch.Tensor


@startup_nvtx_wrapped("metadata:build_lbs_native_body_wrench_sorted_metadata")
def build_lbs_native_body_wrench_sorted_metadata(
    *,
    muscle_segment_offsets: torch.Tensor,
    muscle_force_length_coeff: torch.Tensor,
    waypoint_attachment_body2: torch.Tensor,
    waypoint_attachment_local_weight0: torch.Tensor,
    waypoint_attachment_local_weight1: torch.Tensor,
    body_event_chunk_info: torch.Tensor,
    body_event_start_muscle: torch.Tensor,
    body_event_local_force_coeff: torch.Tensor,
    muscle_count: int,
    body_count: int,
    muscle_cap: int,
    segment_cap: int,
    body_event_cap: int,
    body_cap: int,
) -> LBSNativeBodyWrenchSortedMetadata:
    """Pack native-tile-only muscle data in body-local order.

    ``muscle_order`` maps the internal native-tile index to the public muscle
    index. Callers keep activations and their gradients in public order.
    """
    device = muscle_force_length_coeff.device
    muscle_count = int(muscle_count)
    body_count = int(body_count)
    offsets = muscle_segment_offsets.detach().cpu().to(torch.int64)
    chunk_info = body_event_chunk_info.detach().cpu().to(torch.int64)
    start_muscle = body_event_start_muscle.detach().cpu().to(torch.int64)

    event_count = int(start_muscle.shape[0])
    chunk_counts = chunk_info[:, 2] - chunk_info[:, 1]
    event_body = torch.repeat_interleave(chunk_info[:, 0], chunk_counts)
    if int(event_body.numel()) != event_count:
        raise ValueError(
            "body event chunk metadata does not cover sorted native tile events: "
            f"events={event_count}, chunked={int(event_body.numel())}."
        )
    bodies_by_muscle: list[set[int]] = [set() for _ in range(muscle_count)]
    for muscle_idx, body_idx in zip(
        start_muscle[:, 1].tolist(),
        event_body.tolist(),
        strict=True,
    ):
        bodies_by_muscle[int(muscle_idx)].add(int(body_idx))
    order_list = sorted(
        range(muscle_count),
        key=lambda muscle_idx: (
            tuple(sorted(bodies_by_muscle[muscle_idx])),
            muscle_idx,
        ),
    )
    muscle_order_cpu = torch.tensor(order_list, dtype=torch.int64)
    public_to_sorted = torch.empty(muscle_count, dtype=torch.int64)
    public_to_sorted[muscle_order_cpu] = torch.arange(
        muscle_count,
        dtype=torch.int64,
    )

    segment_counts = offsets[1:] - offsets[:-1]
    sorted_segment_counts = segment_counts.index_select(0, muscle_order_cpu)
    sorted_offsets = torch.zeros(muscle_count + 1, dtype=torch.int64)
    sorted_offsets[1:] = torch.cumsum(sorted_segment_counts, dim=0)
    public_waypoint_offsets = offsets + torch.arange(
        muscle_count + 1,
        dtype=torch.int64,
    )
    sorted_waypoint_offsets = sorted_offsets + torch.arange(
        muscle_count + 1,
        dtype=torch.int64,
    )
    waypoint_order_parts = [
        torch.arange(
            int(public_waypoint_offsets[public_idx].item()),
            int(public_waypoint_offsets[public_idx + 1].item()),
            dtype=torch.int64,
        )
        for public_idx in order_list
    ]
    waypoint_order_cpu = (
        torch.cat(waypoint_order_parts)
        if waypoint_order_parts
        else torch.zeros(0, dtype=torch.int64)
    )

    source_event_muscle = start_muscle[:, 1]
    sorted_event_muscle = public_to_sorted.index_select(0, source_event_muscle)
    relative_start_waypoint = start_muscle[:, 0] - public_waypoint_offsets.index_select(
        0,
        source_event_muscle,
    )
    sorted_event_start_waypoint = (
        sorted_waypoint_offsets.index_select(
            0,
            sorted_event_muscle,
        )
        + relative_start_waypoint
    )
    sorted_event_start_muscle = torch.stack(
        (sorted_event_start_waypoint, sorted_event_muscle),
        dim=1,
    ).to(torch.int32)

    (
        tile_info,
        tile_extra_info,
        tile_body_ids,
        tile_body_output_idx,
        tile_event_start_slot_muscle,
        tile_event_local_force_coeff,
        tile_body_slot_offsets,
    ) = build_lbs_native_body_wrench_tile_metadata(
        muscle_segment_offsets=sorted_offsets.to(torch.int32),
        body_event_chunk_info=body_event_chunk_info,
        body_event_start_muscle=sorted_event_start_muscle.to(
            device=body_event_start_muscle.device
        ),
        body_event_local_force_coeff=body_event_local_force_coeff,
        muscle_count=muscle_count,
        body_count=body_count,
        muscle_cap=muscle_cap,
        segment_cap=segment_cap,
        body_event_cap=body_event_cap,
        body_cap=body_cap,
    )
    muscle_order = muscle_order_cpu.to(device=device)
    waypoint_order = waypoint_order_cpu.to(device=waypoint_attachment_body2.device)
    return LBSNativeBodyWrenchSortedMetadata(
        muscle_order=muscle_order,
        muscle_segment_offsets=sorted_offsets.to(
            device=device,
            dtype=torch.int32,
        ),
        muscle_force_length_coeff=muscle_force_length_coeff.index_select(
            0,
            muscle_order,
        ),
        waypoint_attachment_body2=waypoint_attachment_body2.index_select(
            0,
            waypoint_order,
        ),
        waypoint_attachment_local_weight0=(
            waypoint_attachment_local_weight0.index_select(0, waypoint_order)
        ),
        waypoint_attachment_local_weight1=(
            waypoint_attachment_local_weight1.index_select(0, waypoint_order)
        ),
        tile_info=tile_info,
        tile_extra_info=tile_extra_info,
        tile_body_ids=tile_body_ids,
        tile_body_output_idx=tile_body_output_idx,
        tile_event_start_slot_muscle=tile_event_start_slot_muscle,
        tile_event_local_force_coeff=tile_event_local_force_coeff,
        tile_body_slot_offsets=tile_body_slot_offsets,
    )


@startup_nvtx_wrapped("metadata:build_lbs_forward_packed_metadata")
def build_lbs_forward_packed_metadata(
    *,
    chunk_torque_idx: torch.Tensor,
    chunk_entry_start: torch.Tensor,
    chunk_entry_end: torch.Tensor,
    forward_segment_idx: torch.Tensor,
    forward_muscle_idx: torch.Tensor,
    apply_segment_start_idx: torch.Tensor,
    apply_segment_end_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_info = torch.stack(
        [
            chunk_torque_idx.to(torch.int32),
            chunk_entry_start.to(torch.int32),
            chunk_entry_end.to(torch.int32),
        ],
        dim=1,
    ).contiguous()
    entry_info = torch.stack(
        [
            forward_segment_idx.to(torch.int32),
            forward_muscle_idx.to(torch.int32),
        ],
        dim=1,
    ).contiguous()
    segment_waypoints = torch.stack(
        [
            apply_segment_start_idx.to(torch.int32),
            apply_segment_end_idx.to(torch.int32),
        ],
        dim=1,
    ).contiguous()
    return chunk_info, entry_info, segment_waypoints


@startup_nvtx_wrapped("metadata:build_lbs_forward_endpoint_ranges_metadata")
def build_lbs_forward_endpoint_ranges_metadata(
    *,
    forward_torque_idx: torch.Tensor,
    forward_segment_idx: torch.Tensor,
    apply_segment_start_idx: torch.Tensor,
    apply_segment_end_idx: torch.Tensor,
    active_attach_offsets: torch.Tensor,
    waypoint_count: int,
) -> torch.Tensor:
    device = active_attach_offsets.device
    if int(forward_segment_idx.numel()) == 0:
        return torch.zeros((0, 4), dtype=torch.int32, device=device)

    torque_idx = forward_torque_idx.to(device=device, dtype=torch.long)
    segment_idx = forward_segment_idx.to(device=device, dtype=torch.long)
    start_wp = apply_segment_start_idx.to(device=device, dtype=torch.long).index_select(
        0,
        segment_idx,
    )
    end_wp = apply_segment_end_idx.to(device=device, dtype=torch.long).index_select(
        0,
        segment_idx,
    )
    waypoint_count = int(waypoint_count)
    start_offset = torque_idx * waypoint_count + start_wp
    end_offset = torque_idx * waypoint_count + end_wp
    return (
        torch.stack(
            [
                active_attach_offsets.index_select(0, start_offset),
                active_attach_offsets.index_select(0, start_offset + 1),
                active_attach_offsets.index_select(0, end_offset),
                active_attach_offsets.index_select(0, end_offset + 1),
            ],
            dim=1,
        )
        .contiguous()
        .to(torch.int32)
    )


def _sort_lbs_torque_entries_by_torque_muscle_segment(
    *,
    torque_active_entry_offsets: torch.Tensor,
    torque_active_segment_idx: torch.Tensor,
    torque_active_muscle_idx: torch.Tensor,
    torque_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = torque_active_entry_offsets.detach().cpu().to(torch.long)
    segment_idx = torque_active_segment_idx.detach().cpu().to(torch.long)
    muscle_idx = torque_active_muscle_idx.detach().cpu().to(torch.long)
    torque_dim = int(torque_dim)
    entry_count = int(segment_idx.numel())
    if torque_dim <= 0 or entry_count == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return (
            empty,
            empty.clone(),
            empty.clone(),
            torch.zeros(
                max(torque_dim, 0) + 1,
                dtype=torch.long,
            ),
        )

    counts = offsets[1:] - offsets[:-1]
    torque_idx = torch.repeat_interleave(
        torch.arange(torque_dim, dtype=torch.long),
        counts,
    )
    segment_stride = int(segment_idx.max().item()) + 2
    muscle_stride = (int(muscle_idx.max().item()) + 2) * segment_stride
    order = torch.argsort(
        torque_idx * muscle_stride + muscle_idx * segment_stride + segment_idx
    )
    sorted_torque_idx = torque_idx.index_select(0, order)
    sorted_segment_idx = segment_idx.index_select(0, order)
    sorted_muscle_idx = muscle_idx.index_select(0, order)
    forward_entry_offsets = torch.zeros(torque_dim + 1, dtype=torch.long)
    forward_entry_offsets[1:] = torch.cumsum(counts, dim=0)
    return (
        sorted_torque_idx,
        sorted_segment_idx,
        sorted_muscle_idx,
        forward_entry_offsets,
    )


def _build_lbs_chunk_ranges(
    *,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_size = max(1, int(chunk_size))
    chunks_per_row = (counts + chunk_size - 1) // chunk_size
    chunk_offsets = torch.zeros(int(counts.numel()) + 1, dtype=torch.int32)
    chunk_offsets[1:] = torch.cumsum(chunks_per_row, dim=0).to(torch.int32)
    chunk_count = int(chunks_per_row.sum().item())
    if chunk_count == 0:
        empty = torch.zeros(0, dtype=torch.int32)
        return chunk_offsets, empty.clone(), empty.clone(), empty.clone()
    chunk_row = torch.repeat_interleave(
        torch.arange(int(counts.numel()), dtype=torch.long),
        chunks_per_row,
    )
    chunk_local = torch.arange(chunk_count, dtype=torch.long) - torch.repeat_interleave(
        chunk_offsets[:-1].to(torch.long),
        chunks_per_row,
    )
    chunk_begin = offsets.index_select(0, chunk_row) + chunk_local * chunk_size
    chunk_end = torch.minimum(
        chunk_begin + chunk_size,
        offsets.index_select(0, chunk_row) + counts.index_select(0, chunk_row),
    )
    return (
        chunk_offsets,
        chunk_row.to(torch.int32),
        chunk_begin.to(torch.int32),
        chunk_end.to(torch.int32),
    )


@startup_nvtx_wrapped("metadata:build_lbs_torque_forward_chunks")
def build_lbs_torque_forward_chunks(
    *,
    torque_active_entry_offsets: torch.Tensor,
    torque_active_segment_idx: torch.Tensor,
    torque_active_muscle_idx: torch.Tensor,
    torque_dim: int,
    chunk_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (
        forward_torque_idx,
        forward_segment_idx,
        forward_muscle_idx,
        forward_entry_offsets,
    ) = _sort_lbs_torque_entries_by_torque_muscle_segment(
        torque_active_entry_offsets=torque_active_entry_offsets,
        torque_active_segment_idx=torque_active_segment_idx,
        torque_active_muscle_idx=torque_active_muscle_idx,
        torque_dim=torque_dim,
    )
    counts = forward_entry_offsets[1:] - forward_entry_offsets[:-1]
    (
        chunk_offsets,
        chunk_torque_idx,
        chunk_entry_start,
        chunk_entry_end,
    ) = _build_lbs_chunk_ranges(
        counts=counts,
        offsets=forward_entry_offsets[:-1],
        chunk_size=chunk_size,
    )

    return (
        forward_torque_idx.to(torch.int32),
        forward_segment_idx.to(torch.int32),
        forward_muscle_idx.to(torch.int32),
        chunk_offsets,
        chunk_torque_idx,
        chunk_entry_start,
        chunk_entry_end,
    )


@startup_nvtx_wrapped("metadata:build_lbs_torque_bounds_chunks")
def build_lbs_torque_bounds_chunks(
    *,
    torque_active_entry_offsets: torch.Tensor,
    torque_active_segment_idx: torch.Tensor,
    torque_active_muscle_idx: torch.Tensor,
    torque_dim: int,
    chunk_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    (
        forward_torque_idx,
        forward_segment_idx,
        forward_muscle_idx,
        _forward_entry_offsets,
    ) = _sort_lbs_torque_entries_by_torque_muscle_segment(
        torque_active_entry_offsets=torque_active_entry_offsets,
        torque_active_segment_idx=torque_active_segment_idx,
        torque_active_muscle_idx=torque_active_muscle_idx,
        torque_dim=torque_dim,
    )
    entry_count = int(forward_torque_idx.numel())
    if entry_count == 0:
        empty = torch.zeros(0, dtype=torch.int32)
        return (
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            torch.zeros(torque_dim + 1, dtype=torch.int32),
            empty.clone(),
            empty.clone(),
            empty.clone(),
        )

    group_start_mask = torch.ones(entry_count, dtype=torch.bool)
    group_start_mask[1:] = (forward_torque_idx[1:] != forward_torque_idx[:-1]) | (
        forward_muscle_idx[1:] != forward_muscle_idx[:-1]
    )
    group_entry_start_long = group_start_mask.nonzero(as_tuple=False).flatten()
    group_entry_end_long = torch.cat(
        [
            group_entry_start_long[1:],
            torch.tensor([entry_count], dtype=torch.long),
        ],
        dim=0,
    )
    group_torque_idx = forward_torque_idx.index_select(0, group_entry_start_long)
    group_counts = torch.bincount(group_torque_idx, minlength=torque_dim)
    group_offsets = torch.zeros(torque_dim + 1, dtype=torch.long)
    group_offsets[1:] = torch.cumsum(group_counts, dim=0)
    (
        chunk_offsets,
        chunk_torque_idx,
        chunk_group_start,
        chunk_group_end,
    ) = _build_lbs_chunk_ranges(
        counts=group_counts,
        offsets=group_offsets[:-1],
        chunk_size=chunk_size,
    )

    return (
        forward_torque_idx.to(torch.int32),
        forward_segment_idx.to(torch.int32),
        forward_muscle_idx.to(torch.int32),
        group_entry_start_long.to(torch.int32),
        group_entry_end_long.to(torch.int32),
        chunk_offsets,
        chunk_torque_idx,
        chunk_group_start,
        chunk_group_end,
    )


@startup_nvtx_wrapped("metadata:build_muscle_sparse_lbs_torque_entries")
def build_muscle_sparse_lbs_torque_entries(
    torque_entry_offsets: torch.Tensor,
    torque_segment_idx: torch.Tensor,
    apply_segment_muscle_idx: torch.Tensor,
    muscle_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = torque_entry_offsets.detach().cpu().to(torch.long)
    segment_idx = torque_segment_idx.detach().cpu().to(torch.long)
    segment_to_muscle = apply_segment_muscle_idx.detach().cpu().to(torch.long)
    torque_dim = max(int(offsets.numel()) - 1, 0)
    entry_count = int(segment_idx.numel())
    if torque_dim <= 0 or entry_count == 0:
        empty_i32 = torch.zeros(0, dtype=torch.int32)
        return (
            torch.zeros(muscle_count + 1, dtype=torch.int32),
            empty_i32.clone(),
            empty_i32.clone(),
        )

    counts = offsets[1:] - offsets[:-1]
    torque_idx = torch.repeat_interleave(
        torch.arange(torque_dim, dtype=torch.long),
        counts,
    )
    muscle_idx = segment_to_muscle.index_select(0, segment_idx)
    segment_stride = torque_dim + 1
    muscle_stride = (int(segment_idx.max().item()) + 2) * segment_stride
    order = torch.argsort(
        muscle_idx * muscle_stride + segment_idx * segment_stride + torque_idx
    )
    sorted_muscle = muscle_idx.index_select(0, order)
    muscle_counts = torch.bincount(sorted_muscle, minlength=muscle_count)
    out_offsets = torch.zeros(muscle_count + 1, dtype=torch.int32)
    out_offsets[1:] = torch.cumsum(muscle_counts, dim=0).to(torch.int32)
    return (
        out_offsets,
        segment_idx.index_select(0, order).to(torch.int32),
        torque_idx.index_select(0, order).to(torch.int32),
    )


@wp.kernel(enable_backward=False)
def transform_waypoints_kernel(
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    muscle_waypoint_body: wp.array(dtype=wp.int32),
    muscle_waypoints_local: wp.array(dtype=wp.vec3),
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
):
    env_idx, waypoint_idx = wp.tid()
    body_idx = muscle_waypoint_body[waypoint_idx]

    p = body_pos[env_idx, body_idx]
    q = normalize_quat_wp(body_quat[env_idx, body_idx])
    local = muscle_waypoints_local[waypoint_idx]

    transformed_waypoints[env_idx, waypoint_idx] = p + rot_vec_quat(local, q)


@wp.kernel(enable_backward=False)
def transform_lbs_waypoints_kernel(
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
        q = normalize_quat_wp(body_quat[env_idx, body_idx])
        local = waypoint_attachment_local[attach_idx]
        total += waypoint_attachment_weight[attach_idx] * (p + rot_vec_quat(local, q))
    transformed_waypoints[env_idx, waypoint_idx] = total


@wp.kernel(enable_backward=False)
def eval_muscle_force_magnitude_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    muscle_waypoint_start: wp.array(dtype=wp.int32),
    muscle_params: wp.array2d(dtype=wp.float32),
    muscle_actions: wp.array2d(dtype=wp.float32),
    muscle_rest_length: wp.array(dtype=wp.float32),
    muscle_force_scale: float,
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    clamp_activation: float,
    include_active: float,
    include_passive: float,
    active_force_model_params: wp.vec3,
    passive_force_model_params: wp.vec3,
    muscle_length: wp.array2d(dtype=wp.float32),
    muscle_force_magnitude: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()

    m_start = muscle_waypoint_start[muscle_idx]
    m_end = muscle_waypoint_start[muscle_idx + 1]

    muscle_tendon_length = float(0.0)
    for i in range(m_start, m_end - 1):
        transformed_waypoints[env_idx, i]
        transformed_waypoints[env_idx, i + 1]
        muscle_tendon_length += segment_length_wp(env_idx, i, transformed_waypoints)

    muscle_length[env_idx, muscle_idx] = muscle_tendon_length

    activation = muscle_actions[env_idx, muscle_idx]
    f0 = muscle_params[muscle_idx, 0]
    l_m0 = muscle_params[muscle_idx, 1]
    l_t0 = muscle_params[muscle_idx, 2]
    pen_angle = muscle_params[muscle_idx, 3]
    l_m_norm = normalized_muscle_length_wp(
        muscle_tendon_length,
        l_m0,
        l_t0,
        muscle_rest_length[muscle_idx],
        muscle_length_norm_min,
        muscle_length_norm_max,
    )
    muscle_force_magnitude[env_idx, muscle_idx] = (
        force_magnitude_from_activation_model_wp(
            activation,
            clamp_activation,
            include_active,
            include_passive,
            f0,
            pen_angle,
            muscle_force_scale,
            l_m_norm,
            active_force_model_params,
            passive_force_model_params,
        )
    )


@wp.kernel(enable_backward=False)
def eval_muscle_force_magnitude_velocity_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    muscle_waypoint_start: wp.array(dtype=wp.int32),
    muscle_waypoint_body: wp.array(dtype=wp.int32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    body_rootid: wp.array(dtype=wp.int32),
    body_cvel: wp.array2d(dtype=wp.spatial_vector),
    muscle_params: wp.array2d(dtype=wp.float32),
    muscle_actions: wp.array2d(dtype=wp.float32),
    muscle_rest_length: wp.array(dtype=wp.float32),
    muscle_force_scale: float,
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    clamp_activation: float,
    include_active: float,
    include_passive: float,
    active_force_model_params: wp.vec3,
    passive_force_model_params: wp.vec3,
    muscle_length: wp.array2d(dtype=wp.float32),
    muscle_force_magnitude: wp.array2d(dtype=wp.float32),
    muscle_velocity: wp.array2d(dtype=wp.float32),
):
    """eval_muscle_force_magnitude_kernel and waypoint_tendon_velocity_kernel
    in one pass over each single-body path, with the same arithmetic."""
    env_idx, muscle_idx = wp.tid()
    m_start = muscle_waypoint_start[muscle_idx]
    m_end = muscle_waypoint_start[muscle_idx + 1]
    muscle_tendon_length = float(0.0)
    velocity = float(0.0)
    for start in range(m_start, m_end - 1):
        p0 = transformed_waypoints[env_idx, start]
        p1 = transformed_waypoints[env_idx, start + 1]
        delta = p1 - p0
        length = wp.length(delta)
        muscle_tendon_length += length
        b0 = muscle_waypoint_body[start]
        b1 = muscle_waypoint_body[start + 1]
        if length >= 1e-8:
            v0 = body_point_velocity_wp(
                env_idx, b0, p0, subtree_com, body_rootid, body_cvel
            )
            v1 = body_point_velocity_wp(
                env_idx, b1, p1, subtree_com, body_rootid, body_cvel
            )
            velocity += wp.dot(delta / length, v1 - v0)
    muscle_length[env_idx, muscle_idx] = muscle_tendon_length
    muscle_velocity[env_idx, muscle_idx] = velocity
    l_m_norm = normalized_muscle_length_wp(
        muscle_tendon_length,
        muscle_params[muscle_idx, 1],
        muscle_params[muscle_idx, 2],
        muscle_rest_length[muscle_idx],
        muscle_length_norm_min,
        muscle_length_norm_max,
    )
    muscle_force_magnitude[env_idx, muscle_idx] = (
        force_magnitude_from_activation_model_wp(
            muscle_actions[env_idx, muscle_idx],
            clamp_activation,
            include_active,
            include_passive,
            muscle_params[muscle_idx, 0],
            muscle_params[muscle_idx, 3],
            muscle_force_scale,
            l_m_norm,
            active_force_model_params,
            passive_force_model_params,
        )
    )


@wp.func
def muscle_force_velocity_curve_wp(
    normalized_velocity: float,
    fvmax: float,
) -> wp.vec2:
    """MuJoCo's muscle force-velocity multiplier and d(multiplier)/dV."""
    y = fvmax - 1.0
    if normalized_velocity <= -1.0:
        return wp.vec2(0.0, 0.0)
    if normalized_velocity <= 0.0:
        shifted = normalized_velocity + 1.0
        return wp.vec2(shifted * shifted, 2.0 * shifted)
    if normalized_velocity <= y:
        remaining = y - normalized_velocity
        return wp.vec2(
            fvmax - remaining * remaining / y,
            2.0 * remaining / y,
        )
    return wp.vec2(fvmax, 0.0)


@wp.func
def muscle_force_velocity_response_wp(
    length: float,
    velocity: float,
    activation: float,
    f0: float,
    l_m0: float,
    l_t0: float,
    pen_angle: float,
    rest_length: float,
    length_scale: float,
    muscle_force_scale: float,
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    vmax: float,
    fvmax: float,
    clamp_activation: float,
    include_active: float,
    active_force_model_params: wp.vec3,
) -> wp.vec3:
    velocity_scale = wp.max(length_scale * vmax, 1.0e-6)
    curve = muscle_force_velocity_curve_wp(velocity / velocity_scale, fvmax)
    active = activation
    if clamp_activation != 0.0:
        active = wp.clamp(active, 0.0, 1.0)
    l_m_norm = normalized_muscle_length_wp(
        length,
        l_m0,
        l_t0,
        rest_length,
        muscle_length_norm_min,
        muscle_length_norm_max,
    )
    unit_force = (
        f0
        * active_force_length_model_wp(l_m_norm, active_force_model_params)
        * wp.cos(pen_angle)
        * muscle_force_scale
    )
    return wp.vec3(
        active * curve[0],
        curve[0],
        include_active * active * unit_force * curve[1] / velocity_scale,
    )


@wp.kernel(enable_backward=False)
def eval_muscle_force_velocity_curve_kernel(
    normalized_velocity: wp.array(dtype=wp.float32),
    fvmax: float,
    multiplier: wp.array(dtype=wp.float32),
    derivative: wp.array(dtype=wp.float32),
):
    idx = wp.tid()
    result = muscle_force_velocity_curve_wp(normalized_velocity[idx], fvmax)
    multiplier[idx] = result[0]
    derivative[idx] = result[1]


@wp.kernel(enable_backward=False)
def compute_body_cvel_kernel(
    cdof: wp.array3d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    body_dof_offsets: wp.array(dtype=wp.int32),
    body_dof_idx: wp.array(dtype=wp.int32),
    body_cvel: wp.array2d(dtype=wp.spatial_vector),
):
    """Reconstruct MuJoCo COM-based body spatial velocities from cdof*qvel
    over each body's ancestor DOFs (build_body_ancestor_dof_lists)."""
    env_idx, body_idx = wp.tid()
    angular = wp.vec3(0.0, 0.0, 0.0)
    linear = wp.vec3(0.0, 0.0, 0.0)
    for entry_idx in range(body_dof_offsets[body_idx], body_dof_offsets[body_idx + 1]):
        dof_idx = body_dof_idx[entry_idx]
        velocity = qvel[env_idx, dof_idx]
        angular += velocity * wp.vec3(
            cdof[env_idx, dof_idx, 0],
            cdof[env_idx, dof_idx, 1],
            cdof[env_idx, dof_idx, 2],
        )
        linear += velocity * wp.vec3(
            cdof[env_idx, dof_idx, 3],
            cdof[env_idx, dof_idx, 4],
            cdof[env_idx, dof_idx, 5],
        )
    body_cvel[env_idx, body_idx] = wp.spatial_vector(
        angular[0], angular[1], angular[2], linear[0], linear[1], linear[2]
    )


@wp.kernel(enable_backward=False)
def eval_muscle_sparse_moment_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    muscle_dof_torque_idx: wp.array(dtype=wp.int32),
    muscle_dof_segment_offsets: wp.array(dtype=wp.int32),
    segment_idx: wp.array(dtype=wp.int32),
    start_coeff: wp.array(dtype=wp.float32),
    end_coeff: wp.array(dtype=wp.float32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    segment_start_root_body_idx: wp.array(dtype=wp.int32),
    segment_end_root_body_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    moment: wp.array2d(dtype=wp.float32),
):
    env_idx, entry_idx = wp.tid()
    full_dof = muscle_dof_torque_idx[entry_idx] + root_dof_count
    cdof_angular = wp.vec3(
        cdof[env_idx, full_dof, 0],
        cdof[env_idx, full_dof, 1],
        cdof[env_idx, full_dof, 2],
    )
    cdof_linear = wp.vec3(
        cdof[env_idx, full_dof, 3],
        cdof[env_idx, full_dof, 4],
        cdof[env_idx, full_dof, 5],
    )
    total = float(0.0)
    begin = muscle_dof_segment_offsets[entry_idx]
    end = muscle_dof_segment_offsets[entry_idx + 1]
    for sparse_idx in range(begin, end):
        apply_idx = segment_idx[sparse_idx]
        total += segment_generalized_force_contribution_wp(
            cdof_linear,
            cdof_angular,
            transformed_waypoints[env_idx, apply_segment_start_idx[apply_idx]],
            transformed_waypoints[env_idx, apply_segment_end_idx[apply_idx]],
            subtree_com[env_idx, segment_start_root_body_idx[apply_idx]],
            subtree_com[env_idx, segment_end_root_body_idx[apply_idx]],
            start_coeff[sparse_idx],
            end_coeff[sparse_idx],
            1.0,
        )
    moment[env_idx, entry_idx] = total


# A block per world evaluates each applied segment's direction and lever terms
# once in shared memory; the per-entry kernel repeats them for every dof the
# segment crosses (SNU: 2,890 evaluations of 388 segments per world).
_SPARSE_MOMENT_BLOCK_DIM = 256
_SPARSE_MOMENT_MAX_SHARED_BYTES = 48 * 1024


@cache
def _sparse_moment_block_kernel(segment_count: int, nv: int):
    def shared_snippet(size: int) -> str:
        return (
            f"__shared__ float data[{size}];\n"
            f"wp::array_t<float> result(data, {size});\n"
            "return result;\n"
        )

    segment_snippet = shared_snippet(9 * segment_count)
    cdof_snippet = shared_snippet(6 * nv)

    @wp.func_native(snippet=segment_snippet)
    def shared_segments(dummy: int) -> wp.array(dtype=float): ...

    @wp.func_native(snippet=cdof_snippet)
    def shared_cdof(dummy: int) -> wp.array(dtype=float): ...

    @wp.func_native(snippet="__syncthreads();")
    def block_sync(): ...

    # With u the unit segment direction and r = p - subtree_com,
    # (angular x r) . u = angular . (r x u), so a segment needs only u,
    # w0 = r_start x u and w1 = r_end x u (all zero for a degenerate segment).
    @wp.kernel(module="unique", enable_backward=False)
    def sparse_moment_block(
        transformed_waypoints: wp.array2d(dtype=wp.vec3),
        subtree_com: wp.array2d(dtype=wp.vec3),
        cdof: wp.array3d(dtype=wp.float32),
        muscle_dof_torque_idx: wp.array(dtype=wp.int32),
        muscle_dof_segment_offsets: wp.array(dtype=wp.int32),
        segment_idx: wp.array(dtype=wp.int32),
        start_coeff: wp.array(dtype=wp.float32),
        end_coeff: wp.array(dtype=wp.float32),
        apply_segment_start_idx: wp.array(dtype=wp.int32),
        apply_segment_end_idx: wp.array(dtype=wp.int32),
        segment_start_root_body_idx: wp.array(dtype=wp.int32),
        segment_end_root_body_idx: wp.array(dtype=wp.int32),
        root_dof_count: int,
        moment: wp.array2d(dtype=wp.float32),
    ):
        world, lane = wp.tid()
        width = wp.block_dim()
        segment = shared_segments(0)
        cd = shared_cdof(0)
        for s in range(lane, segment_count, width):
            p_start = transformed_waypoints[world, apply_segment_start_idx[s]]
            p_end = transformed_waypoints[world, apply_segment_end_idx[s]]
            delta = p_end - p_start
            length = wp.length(delta)
            u = wp.vec3(0.0)
            w0 = wp.vec3(0.0)
            w1 = wp.vec3(0.0)
            if length >= 1.0e-8:
                u = delta / length
                w0 = wp.cross(
                    p_start - subtree_com[world, segment_start_root_body_idx[s]], u
                )
                w1 = wp.cross(
                    p_end - subtree_com[world, segment_end_root_body_idx[s]], u
                )
            for k in range(3):
                segment[9 * s + k] = u[k]
                segment[9 * s + 3 + k] = w0[k]
                segment[9 * s + 6 + k] = w1[k]
        for e in range(lane, 6 * nv, width):
            cd[e] = cdof[world, e // 6, e % 6]
        block_sync()
        for entry in range(lane, muscle_dof_torque_idx.shape[0], width):
            base = 6 * (muscle_dof_torque_idx[entry] + root_dof_count)
            angular = wp.vec3(cd[base], cd[base + 1], cd[base + 2])
            linear = wp.vec3(cd[base + 3], cd[base + 4], cd[base + 5])
            total = float(0.0)
            for sparse_idx in range(
                muscle_dof_segment_offsets[entry], muscle_dof_segment_offsets[entry + 1]
            ):
                s9 = 9 * segment_idx[sparse_idx]
                a = start_coeff[sparse_idx]
                b = end_coeff[sparse_idx]
                u = wp.vec3(segment[s9], segment[s9 + 1], segment[s9 + 2])
                w0 = wp.vec3(segment[s9 + 3], segment[s9 + 4], segment[s9 + 5])
                w1 = wp.vec3(segment[s9 + 6], segment[s9 + 7], segment[s9 + 8])
                total += (a - b) * wp.dot(linear, u) + wp.dot(angular, a * w0 - b * w1)
            moment[world, entry] = total

    return sparse_moment_block


def launch_muscle_sparse_moment(
    transformed_waypoints: wp.array,
    subtree_com: wp.array,
    cdof: object,
    muscle_dof_torque_idx: wp.array,
    muscle_dof_segment_offsets: wp.array,
    segment_idx: wp.array,
    start_coeff: wp.array,
    end_coeff: wp.array,
    apply_segment_start_idx: wp.array,
    apply_segment_end_idx: wp.array,
    segment_start_root_body_idx: wp.array,
    segment_end_root_body_idx: wp.array,
    root_dof_count: int,
    moment: wp.array,
) -> None:
    """Sparse muscle moment rows: the block kernel on CUDA when it fits."""
    inputs = [
        transformed_waypoints,
        subtree_com,
        cdof,
        muscle_dof_torque_idx,
        muscle_dof_segment_offsets,
        segment_idx,
        start_coeff,
        end_coeff,
        apply_segment_start_idx,
        apply_segment_end_idx,
        segment_start_root_body_idx,
        segment_end_root_body_idx,
        root_dof_count,
    ]
    segment_count = int(apply_segment_start_idx.shape[0])
    nv = int(cdof.shape[1])
    shared_bytes = 4 * (9 * segment_count + 6 * nv)
    if (
        moment.device.is_cuda
        and segment_count > 0
        and shared_bytes <= _SPARSE_MOMENT_MAX_SHARED_BYTES
    ):
        wp.launch_tiled(
            _sparse_moment_block_kernel(segment_count, nv),
            dim=[moment.shape[0]],
            inputs=inputs,
            outputs=[moment],
            block_dim=_SPARSE_MOMENT_BLOCK_DIM,
        )
        return
    wp.launch(
        eval_muscle_sparse_moment_kernel,
        dim=moment.shape,
        inputs=inputs,
        outputs=[moment],
    )


@wp.kernel(enable_backward=False)
def eval_muscle_force_velocity_state_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    muscle_waypoint_start: wp.array(dtype=wp.int32),
    muscle_params: wp.array2d(dtype=wp.float32),
    muscle_rest_length: wp.array(dtype=wp.float32),
    force_velocity_length_scale: wp.array(dtype=wp.float32),
    activation: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    moment: wp.array2d(dtype=wp.float32),
    muscle_dof_offsets: wp.array(dtype=wp.int32),
    muscle_dof_torque_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    muscle_force_scale: float,
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    vmax: float,
    fvmax: float,
    clamp_activation: float,
    include_active: float,
    active_force_model_params: wp.vec3,
    muscle_length: wp.array2d(dtype=wp.float32),
    effective_activation: wp.array2d(dtype=wp.float32),
    length_velocity: wp.array2d(dtype=wp.float32),
    force_velocity_multiplier: wp.array2d(dtype=wp.float32),
    force_velocity_slope: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    waypoint_begin = muscle_waypoint_start[muscle_idx]
    waypoint_end = muscle_waypoint_start[muscle_idx + 1]
    length = float(0.0)
    for waypoint_idx in range(waypoint_begin, waypoint_end - 1):
        length += segment_length_wp(
            env_idx,
            waypoint_idx,
            transformed_waypoints,
        )
    muscle_length[env_idx, muscle_idx] = length

    velocity = float(0.0)
    begin = muscle_dof_offsets[muscle_idx]
    end = muscle_dof_offsets[muscle_idx + 1]
    for entry_idx in range(begin, end):
        full_dof = muscle_dof_torque_idx[entry_idx] + root_dof_count
        velocity -= moment[env_idx, entry_idx] * qvel[env_idx, full_dof]

    response = muscle_force_velocity_response_wp(
        length,
        velocity,
        activation[env_idx, muscle_idx],
        muscle_params[muscle_idx, 0],
        muscle_params[muscle_idx, 1],
        muscle_params[muscle_idx, 2],
        muscle_params[muscle_idx, 3],
        muscle_rest_length[muscle_idx],
        force_velocity_length_scale[muscle_idx],
        muscle_force_scale,
        muscle_length_norm_min,
        muscle_length_norm_max,
        vmax,
        fvmax,
        clamp_activation,
        include_active,
        active_force_model_params,
    )
    effective_activation[env_idx, muscle_idx] = response[0]
    length_velocity[env_idx, muscle_idx] = velocity
    force_velocity_multiplier[env_idx, muscle_idx] = response[1]
    force_velocity_slope[env_idx, muscle_idx] = response[2]


@wp.kernel(enable_backward=False)
def apply_muscle_implicit_qderiv_kernel(
    moment: wp.array2d(dtype=wp.float32),
    force_velocity_slope: wp.array2d(dtype=wp.float32),
    pair_entry_i: wp.array(dtype=wp.int32),
    pair_entry_j: wp.array(dtype=wp.int32),
    pair_muscle_idx: wp.array(dtype=wp.int32),
    muscle_dof_torque_idx: wp.array(dtype=wp.int32),
    M_elemid: wp.array2d(dtype=wp.int32),
    root_dof_count: int,
    timestep: float,
    qderiv: wp.array2d(dtype=wp.float32),
):
    env_idx, pair_idx = wp.tid()
    entry_i = pair_entry_i[pair_idx]
    entry_j = pair_entry_j[pair_idx]
    dof_i = muscle_dof_torque_idx[entry_i] + root_dof_count
    dof_j = muscle_dof_torque_idx[entry_j] + root_dof_count
    elem_id = M_elemid[dof_i, dof_j]
    if elem_id >= 0:
        muscle_idx = pair_muscle_idx[pair_idx]
        contribution = (
            timestep
            * force_velocity_slope[env_idx, muscle_idx]
            * moment[env_idx, entry_i]
            * moment[env_idx, entry_j]
        )
        wp.atomic_add(qderiv, env_idx, elem_id, contribution)


@wp.kernel(enable_backward=False)
def apply_precomputed_muscle_implicit_qderiv_kernel(
    moment: wp.array2d(dtype=wp.float32),
    force_velocity_slope: wp.array2d(dtype=wp.float32),
    pair_elem_id: wp.array(dtype=wp.int32),
    pair_entry_i: wp.array(dtype=wp.int32),
    pair_entry_j: wp.array(dtype=wp.int32),
    pair_muscle_idx: wp.array(dtype=wp.int32),
    timestep: float,
    qderiv: wp.array2d(dtype=wp.float32),
):
    env_idx, pair_idx = wp.tid()
    entry_i = pair_entry_i[pair_idx]
    entry_j = pair_entry_j[pair_idx]
    muscle_idx = pair_muscle_idx[pair_idx]
    contribution = (
        timestep
        * force_velocity_slope[env_idx, muscle_idx]
        * moment[env_idx, entry_i]
        * moment[env_idx, entry_j]
    )
    wp.atomic_add(qderiv, env_idx, pair_elem_id[pair_idx], contribution)


@wp.kernel(enable_backward=False)
def project_muscle_sparse_moment_qfrc_kernel(
    moment: wp.array2d(dtype=wp.float32),
    muscle_force_magnitude: wp.array2d(dtype=wp.float32),
    torque_entry_offsets: wp.array(dtype=wp.int32),
    torque_entry_idx: wp.array(dtype=wp.int32),
    muscle_dof_muscle_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    qfrc_applied: wp.array2d(dtype=wp.float32),
):
    env_idx, torque_idx = wp.tid()
    total = float(0.0)
    begin = torque_entry_offsets[torque_idx]
    end = torque_entry_offsets[torque_idx + 1]
    for sorted_idx in range(begin, end):
        entry_idx = torque_entry_idx[sorted_idx]
        muscle_idx = muscle_dof_muscle_idx[entry_idx]
        total += (
            moment[env_idx, entry_idx] * muscle_force_magnitude[env_idx, muscle_idx]
        )
    qfrc_applied[env_idx, torque_idx + root_dof_count] = total


@wp.kernel(enable_backward=False)
def eval_muscle_qfrc_kernel(
    muscle_force_magnitude: wp.array2d(dtype=wp.float32),
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    xipos: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    torque_active_entry_offsets: wp.array(dtype=wp.int32),
    torque_active_segment_idx: wp.array(dtype=wp.int32),
    torque_active_muscle_idx: wp.array(dtype=wp.int32),
    torque_active_start_coeff: wp.array(dtype=wp.float32),
    torque_active_end_coeff: wp.array(dtype=wp.float32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    apply_segment_body_start_idx: wp.array(dtype=wp.int32),
    apply_segment_body_end_idx: wp.array(dtype=wp.int32),
    segment_start_root_body_idx: wp.array(dtype=wp.int32),
    segment_end_root_body_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    qfrc_applied: wp.array2d(dtype=wp.float32),
):
    env_idx, dof_idx = wp.tid()
    full_dof = dof_idx + root_dof_count

    linear_cdof = wp.vec3(
        cdof[env_idx, full_dof, 3],
        cdof[env_idx, full_dof, 4],
        cdof[env_idx, full_dof, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_dof, 0],
        cdof[env_idx, full_dof, 1],
        cdof[env_idx, full_dof, 2],
    )

    total = float(0.0)
    entry_begin = torque_active_entry_offsets[dof_idx]
    entry_end = torque_active_entry_offsets[dof_idx + 1]
    for entry_idx in range(entry_begin, entry_end):
        muscle_idx = torque_active_muscle_idx[entry_idx]
        force_magnitude = muscle_force_magnitude[env_idx, muscle_idx]
        if force_magnitude == 0.0:
            continue

        segment_idx = torque_active_segment_idx[entry_idx]
        p_start = transformed_waypoints[env_idx, apply_segment_start_idx[segment_idx]]
        p_end = transformed_waypoints[env_idx, apply_segment_end_idx[segment_idx]]
        subtree_com_start = subtree_com[
            env_idx, segment_start_root_body_idx[segment_idx]
        ]
        subtree_com_end = subtree_com[env_idx, segment_end_root_body_idx[segment_idx]]
        total += segment_generalized_force_contribution_wp(
            linear_cdof,
            angular_cdof,
            p_start,
            p_end,
            subtree_com_start,
            subtree_com_end,
            torque_active_start_coeff[entry_idx],
            torque_active_end_coeff[entry_idx],
            force_magnitude,
        )

    qfrc_applied[env_idx, full_dof] = total


@wp.kernel(enable_backward=False)
def eval_muscle_lbs_qfrc_kernel(
    muscle_force_magnitude: wp.array2d(dtype=wp.float32),
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
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
    root_dof_count: int,
    qfrc_applied: wp.array2d(dtype=wp.float32),
):
    env_idx, dof_idx = wp.tid()
    full_dof = dof_idx + root_dof_count

    linear_cdof = wp.vec3(
        cdof[env_idx, full_dof, 3],
        cdof[env_idx, full_dof, 4],
        cdof[env_idx, full_dof, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_dof, 0],
        cdof[env_idx, full_dof, 1],
        cdof[env_idx, full_dof, 2],
    )

    total = float(0.0)
    entry_begin = torque_active_entry_offsets[dof_idx]
    entry_end = torque_active_entry_offsets[dof_idx + 1]
    for entry_idx in range(entry_begin, entry_end):
        muscle_idx = torque_active_muscle_idx[entry_idx]
        force_magnitude = muscle_force_magnitude[env_idx, muscle_idx]
        if force_magnitude == 0.0:
            continue

        segment_idx = torque_active_segment_idx[entry_idx]
        start_wp = apply_segment_start_idx[segment_idx]
        end_wp = apply_segment_end_idx[segment_idx]
        p_start = transformed_waypoints[env_idx, start_wp]
        p_end = transformed_waypoints[env_idx, end_wp]
        segment = p_end - p_start
        segment_length = wp.length(segment)
        if segment_length < 1.0e-8:
            continue
        seg_force = force_magnitude * segment / segment_length

        start_begin = waypoint_attachment_start[start_wp]
        start_end = waypoint_attachment_start[start_wp + 1]
        for attach_idx in range(start_begin, start_end):
            body_idx = waypoint_attachment_body[attach_idx]
            coeff = dof_subtree_mask[full_dof, body_idx]
            if coeff == 0.0:
                continue
            q = normalize_quat_wp(body_quat[env_idx, body_idx])
            point = body_pos[env_idx, body_idx] + rot_vec_quat(
                waypoint_attachment_local[attach_idx], q
            )
            total += endpoint_generalized_force_contribution_wp(
                linear_cdof,
                angular_cdof,
                point,
                subtree_com[env_idx, body_rootid[body_idx]],
                coeff,
                waypoint_attachment_weight[attach_idx],
                seg_force,
            )

        end_begin = waypoint_attachment_start[end_wp]
        end_end = waypoint_attachment_start[end_wp + 1]
        for attach_idx in range(end_begin, end_end):
            body_idx = waypoint_attachment_body[attach_idx]
            coeff = dof_subtree_mask[full_dof, body_idx]
            if coeff == 0.0:
                continue
            q = normalize_quat_wp(body_quat[env_idx, body_idx])
            point = body_pos[env_idx, body_idx] + rot_vec_quat(
                waypoint_attachment_local[attach_idx], q
            )
            total += endpoint_generalized_force_contribution_wp(
                linear_cdof,
                angular_cdof,
                point,
                subtree_com[env_idx, body_rootid[body_idx]],
                coeff,
                waypoint_attachment_weight[attach_idx],
                -seg_force,
            )

    qfrc_applied[env_idx, full_dof] = total


@wp.func
def lbs_segment_entry_contribution_active_points_wp(
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
        q = normalize_quat_wp(body_quat[env_idx, body_idx])
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
        q = normalize_quat_wp(body_quat[env_idx, body_idx])
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
def lbs_segment_entry_contribution_active_pair_vec4_native_points_wp(
    env_idx: int,
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
    active_attach_idx_body: wp.array(dtype=wp.vec2i),
    active_attach_local_coeff: wp.array(dtype=wp.vec4),
) -> float:
    segment = p_end - p_start
    segment_length = wp.length(segment)
    if segment_length < 1.0e-8:
        return 0.0

    seg_force = force_magnitude * segment / segment_length
    total = float(0.0)

    for active_idx in range(start_begin, start_end):
        idx_body = native_load_vec2i_wp(active_attach_idx_body, active_idx)
        packed = native_load_vec4_wp(active_attach_local_coeff, active_idx)
        body_idx = idx_body[1]
        q = normalize_quat_wp(body_quat[env_idx, body_idx])
        local = wp.vec3(packed[0], packed[1], packed[2])
        point = body_pos[env_idx, body_idx] + rot_vec_quat(local, q)
        root_idx = body_rootid[body_idx]
        term = linear_cdof + wp.cross(
            angular_cdof,
            point - subtree_com[env_idx, root_idx],
        )
        total += packed[3] * wp.dot(term, seg_force)

    for active_idx in range(end_begin, end_end):
        idx_body = native_load_vec2i_wp(active_attach_idx_body, active_idx)
        packed = native_load_vec4_wp(active_attach_local_coeff, active_idx)
        body_idx = idx_body[1]
        q = normalize_quat_wp(body_quat[env_idx, body_idx])
        local = wp.vec3(packed[0], packed[1], packed[2])
        point = body_pos[env_idx, body_idx] + rot_vec_quat(local, q)
        root_idx = body_rootid[body_idx]
        term = linear_cdof + wp.cross(
            angular_cdof,
            point - subtree_com[env_idx, root_idx],
        )
        total -= packed[3] * wp.dot(term, seg_force)
    return total


@wp.func
def lbs_segment_entry_contribution_active_wp(
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
    return lbs_segment_entry_contribution_active_points_wp(
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


@wp.kernel(enable_backward=False)
def eval_lbs_muscle_sparse_moment_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    body_rootid: wp.array(dtype=wp.int32),
    muscle_dof_torque_idx: wp.array(dtype=wp.int32),
    muscle_dof_segment_offsets: wp.array(dtype=wp.int32),
    segment_idx: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    active_attach_offsets: wp.array(dtype=wp.int32),
    active_attach_idx_body: wp.array(dtype=wp.vec2i),
    active_attach_local_coeff: wp.array(dtype=wp.vec4),
    waypoint_count: int,
    root_dof_count: int,
    moment: wp.array2d(dtype=wp.float32),
):
    env_idx, entry_idx = wp.tid()
    torque_idx = muscle_dof_torque_idx[entry_idx]
    full_dof = torque_idx + root_dof_count
    linear_cdof = wp.vec3(
        cdof[env_idx, full_dof, 3],
        cdof[env_idx, full_dof, 4],
        cdof[env_idx, full_dof, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_dof, 0],
        cdof[env_idx, full_dof, 1],
        cdof[env_idx, full_dof, 2],
    )
    total = float(0.0)
    begin = muscle_dof_segment_offsets[entry_idx]
    end = muscle_dof_segment_offsets[entry_idx + 1]
    for sparse_idx in range(begin, end):
        apply_idx = segment_idx[sparse_idx]
        start_wp = apply_segment_start_idx[apply_idx]
        end_wp = apply_segment_end_idx[apply_idx]
        start_offset_idx = torque_idx * waypoint_count + start_wp
        end_offset_idx = torque_idx * waypoint_count + end_wp
        total += lbs_segment_entry_contribution_active_pair_vec4_native_points_wp(
            env_idx,
            1.0,
            linear_cdof,
            angular_cdof,
            transformed_waypoints[env_idx, start_wp],
            transformed_waypoints[env_idx, end_wp],
            body_pos,
            body_quat,
            subtree_com,
            body_rootid,
            active_attach_offsets[start_offset_idx],
            active_attach_offsets[start_offset_idx + 1],
            active_attach_offsets[end_offset_idx],
            active_attach_offsets[end_offset_idx + 1],
            active_attach_idx_body,
            active_attach_local_coeff,
        )
    moment[env_idx, entry_idx] = total


@wp.kernel(enable_backward=False)
def clear_muscle_sparse_moment_kernel(
    moment: wp.array2d(dtype=wp.float32),
):
    env_idx, entry_idx = wp.tid()
    moment[env_idx, entry_idx] = 0.0


@wp.kernel(enable_backward=False)
def eval_lbs_muscle_sparse_moment_atomic_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    body_rootid: wp.array(dtype=wp.int32),
    sparse_entry_idx: wp.array(dtype=wp.int32),
    muscle_dof_torque_idx: wp.array(dtype=wp.int32),
    segment_idx: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    active_attach_offsets: wp.array(dtype=wp.int32),
    active_attach_idx_body: wp.array(dtype=wp.vec2i),
    active_attach_local_coeff: wp.array(dtype=wp.vec4),
    waypoint_count: int,
    root_dof_count: int,
    moment: wp.array2d(dtype=wp.float32),
):
    env_idx, sparse_idx = wp.tid()
    entry_idx = sparse_entry_idx[sparse_idx]
    torque_idx = muscle_dof_torque_idx[entry_idx]
    full_dof = torque_idx + root_dof_count
    linear_cdof = wp.vec3(
        cdof[env_idx, full_dof, 3],
        cdof[env_idx, full_dof, 4],
        cdof[env_idx, full_dof, 5],
    )
    angular_cdof = wp.vec3(
        cdof[env_idx, full_dof, 0],
        cdof[env_idx, full_dof, 1],
        cdof[env_idx, full_dof, 2],
    )
    apply_idx = segment_idx[sparse_idx]
    start_wp = apply_segment_start_idx[apply_idx]
    end_wp = apply_segment_end_idx[apply_idx]
    start_offset_idx = torque_idx * waypoint_count + start_wp
    end_offset_idx = torque_idx * waypoint_count + end_wp
    contribution = lbs_segment_entry_contribution_active_pair_vec4_native_points_wp(
        env_idx,
        1.0,
        linear_cdof,
        angular_cdof,
        transformed_waypoints[env_idx, start_wp],
        transformed_waypoints[env_idx, end_wp],
        body_pos,
        body_quat,
        subtree_com,
        body_rootid,
        active_attach_offsets[start_offset_idx],
        active_attach_offsets[start_offset_idx + 1],
        active_attach_offsets[end_offset_idx],
        active_attach_offsets[end_offset_idx + 1],
        active_attach_idx_body,
        active_attach_local_coeff,
    )
    wp.atomic_add(moment, env_idx, entry_idx, contribution)


@wp.func
def lbs_muscle_path_velocity_wp(
    env_idx: int,
    muscle_idx: int,
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    qvel_source: wp.array2d(dtype=wp.float32),
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
    qvel_offset: int,
) -> float:
    """LBS musculotendon path velocity ``-sum_d moment_d qvel_d``.

    ``qvel_source[env, qvel_offset + dof]`` is the full generalized velocity.
    The moment is the unit-force generalized force, so a positive tendon force
    shortens the path along it.
    """
    velocity = float(0.0)
    begin = muscle_active_entry_offsets[muscle_idx]
    end = muscle_active_entry_offsets[muscle_idx + 1]
    for entry_idx in range(begin, end):
        torque_idx = muscle_active_torque_idx[entry_idx]
        full_dof = torque_idx + root_dof_count
        linear_cdof = wp.vec3(
            cdof[env_idx, full_dof, 3],
            cdof[env_idx, full_dof, 4],
            cdof[env_idx, full_dof, 5],
        )
        angular_cdof = wp.vec3(
            cdof[env_idx, full_dof, 0],
            cdof[env_idx, full_dof, 1],
            cdof[env_idx, full_dof, 2],
        )
        moment = lbs_segment_entry_contribution_active_wp(
            env_idx,
            torque_idx,
            muscle_active_segment_idx[entry_idx],
            1.0,
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
        velocity = velocity - moment * qvel_source[env_idx, qvel_offset + full_dof]
    return velocity


@wp.kernel(enable_backward=False)
def apply_lbs_muscle_force_velocity_to_unit_force_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
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
    force_velocity_length_scale: wp.array(dtype=wp.float32),
    waypoint_count: int,
    root_dof_count: int,
    geometry_context_dim: int,
    vmax: float,
    fvmax: float,
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    velocity = lbs_muscle_path_velocity_wp(
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
        geometry_context_dim,
    )
    velocity_scale = wp.max(
        force_velocity_length_scale[muscle_idx] * vmax,
        1.0e-6,
    )
    multiplier = muscle_force_velocity_curve_wp(velocity / velocity_scale, fvmax)[0]
    unit_force_magnitude[env_idx, muscle_idx] *= multiplier


@wp.kernel(enable_backward=False)
def apply_lbs_muscle_force_velocity_to_force_cache_kernel(
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    raw_context: wp.array2d(dtype=wp.float32),
    activation: wp.array2d(dtype=wp.float32),
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
    force_velocity_length_scale: wp.array(dtype=wp.float32),
    waypoint_count: int,
    root_dof_count: int,
    geometry_context_dim: int,
    vmax: float,
    fvmax: float,
    unit_force_magnitude: wp.array2d(dtype=wp.float32),
    force_by_muscle: wp.array2d(dtype=wp.float32),
):
    env_idx, muscle_idx = wp.tid()
    velocity = lbs_muscle_path_velocity_wp(
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
        geometry_context_dim,
    )
    velocity_scale = wp.max(
        force_velocity_length_scale[muscle_idx] * vmax,
        1.0e-6,
    )
    multiplier = muscle_force_velocity_curve_wp(velocity / velocity_scale, fvmax)[0]
    unit_force = unit_force_magnitude[env_idx, muscle_idx]
    active_force = activation[env_idx, muscle_idx] * unit_force
    force_by_muscle[env_idx, muscle_idx] += (multiplier - 1.0) * active_force
    unit_force_magnitude[env_idx, muscle_idx] = multiplier * unit_force


@wp.kernel(enable_backward=False)
def eval_muscle_lbs_qfrc_pair_vec4_native_tiled_partials_kernel(
    muscle_force_magnitude: wp.array2d(dtype=wp.float32),
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_quat: wp.array2d(dtype=wp.quat),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    body_rootid: wp.array(dtype=wp.int32),
    torque_chunk_torque_idx: wp.array(dtype=wp.int32),
    torque_chunk_entry_start: wp.array(dtype=wp.int32),
    torque_chunk_entry_end: wp.array(dtype=wp.int32),
    torque_active_segment_idx: wp.array(dtype=wp.int32),
    torque_active_muscle_idx: wp.array(dtype=wp.int32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    active_attach_offsets: wp.array(dtype=wp.int32),
    active_attach_idx_body: wp.array(dtype=wp.vec2i),
    active_attach_local_coeff: wp.array(dtype=wp.vec4),
    waypoint_count: int,
    root_dof_count: int,
    partial: wp.array2d(dtype=wp.float32),
):
    env_idx, chunk_idx, lane_idx = wp.tid()
    torque_idx = torque_chunk_torque_idx[chunk_idx]
    entry_begin = torque_chunk_entry_start[chunk_idx]
    entry_end = torque_chunk_entry_end[chunk_idx]
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
    tile_entry_begin = entry_begin
    while tile_entry_begin < entry_end:
        tile_entry_end = wp.min(tile_entry_begin + wp.block_dim(), entry_end)
        entry_idx = tile_entry_begin + lane_idx
        valid_entry = entry_idx < tile_entry_end
        segment_idx = int(0)
        muscle_idx = int(0)
        force_magnitude = float(0.0)
        start_wp = int(0)
        end_wp = int(0)
        start_begin = int(0)
        start_end = int(0)
        p_start = wp.vec3(0.0, 0.0, 0.0)
        if valid_entry:
            muscle_idx = torque_active_muscle_idx[entry_idx]
            force_magnitude = muscle_force_magnitude[env_idx, muscle_idx]
            segment_idx = torque_active_segment_idx[entry_idx]
            start_wp = apply_segment_start_idx[segment_idx]
            end_wp = apply_segment_end_idx[segment_idx]
            p_start = transformed_waypoints[env_idx, start_wp]
            start_offset_idx = torque_idx * waypoint_count + start_wp
            start_begin = active_attach_offsets[start_offset_idx]
            start_end = active_attach_offsets[start_offset_idx + 1]

        start_wp_tile = wp.tile(start_wp)
        start_begin_tile = wp.tile(start_begin)
        start_end_tile = wp.tile(start_end)
        p_start_tile = wp.tile(p_start, preserve_type=True)

        next_lane = lane_idx + 1
        if next_lane >= wp.block_dim():
            next_lane = lane_idx
        next_start_wp = wp.tile_extract(start_wp_tile, next_lane)
        next_start_begin = wp.tile_extract(start_begin_tile, next_lane)
        next_start_end = wp.tile_extract(start_end_tile, next_lane)
        next_p_start = wp.tile_extract(p_start_tile, next_lane)

        if valid_entry and force_magnitude != 0.0:
            p_end = wp.vec3(0.0, 0.0, 0.0)
            end_begin = int(0)
            end_end = int(0)
            if entry_idx + 1 < tile_entry_end and next_start_wp == end_wp:
                p_end = next_p_start
                end_begin = next_start_begin
                end_end = next_start_end
            else:
                p_end = transformed_waypoints[env_idx, end_wp]
                end_offset_idx = torque_idx * waypoint_count + end_wp
                end_begin = active_attach_offsets[end_offset_idx]
                end_end = active_attach_offsets[end_offset_idx + 1]
            total += lbs_segment_entry_contribution_active_pair_vec4_native_points_wp(
                env_idx,
                force_magnitude,
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
                active_attach_idx_body,
                active_attach_local_coeff,
            )
        tile_entry_begin += wp.block_dim()

    total_tile = wp.tile(total)
    sum_tile = wp.tile_sum(total_tile)
    if lane_idx == 0:
        partial[env_idx, chunk_idx] = wp.tile_extract(sum_tile, 0)


@wp.kernel(enable_backward=False)
def reduce_lbs_qfrc_chunk_partials_kernel(
    partial: wp.array2d(dtype=wp.float32),
    torque_chunk_offsets: wp.array(dtype=wp.int32),
    root_dof_count: int,
    qfrc_applied: wp.array2d(dtype=wp.float32),
):
    env_idx, torque_idx = wp.tid()
    chunk_begin = torque_chunk_offsets[torque_idx]
    chunk_end = torque_chunk_offsets[torque_idx + 1]
    total = float(0.0)
    for chunk_idx in range(chunk_begin, chunk_end):
        total += partial[env_idx, chunk_idx]
    qfrc_applied[env_idx, torque_idx + root_dof_count] = total


@wp.kernel(enable_backward=False)
def write_generalized_force_actions_kernel(
    actions: wp.array2d(dtype=wp.float32),
    torque_scale: wp.array(dtype=wp.float32),
    apply_scale: float,
    root_dof_count: int,
    qfrc_applied: wp.array2d(dtype=wp.float32),
):
    env_idx, torque_idx = wp.tid()
    value = actions[env_idx, torque_idx]
    if apply_scale != 0.0:
        value *= torque_scale[torque_idx]
    qfrc_applied[env_idx, torque_idx + root_dof_count] += value


@wp.kernel(enable_backward=False)
def write_generalized_force_actions_indexed_kernel(
    actions: wp.array2d(dtype=wp.float32),
    torque_scale: wp.array(dtype=wp.float32),
    active_indices: wp.array(dtype=wp.int32),
    apply_scale: float,
    root_dof_count: int,
    qfrc_applied: wp.array2d(dtype=wp.float32),
):
    env_idx, compact_idx = wp.tid()
    full_idx = active_indices[compact_idx]
    value = actions[env_idx, compact_idx]
    if apply_scale != 0.0:
        value *= torque_scale[full_idx]
    qfrc_applied[env_idx, full_idx + root_dof_count] += value


@wp.kernel(enable_backward=False)
def extract_applied_torque_kernel(
    qfrc_smooth: wp.array2d(dtype=wp.float32),
    qfrc_bias: wp.array2d(dtype=wp.float32),
    torque_scale: wp.array(dtype=wp.float32),
    root_dof_count: int,
    out_torque: wp.array2d(dtype=wp.float32),
):
    env_idx, dof_idx = wp.tid()
    full_idx = dof_idx + root_dof_count
    raw = qfrc_smooth[env_idx, full_idx] - qfrc_bias[env_idx, full_idx]
    scale = torque_scale[dof_idx]
    if scale > 1.0e-8:
        out_torque[env_idx, dof_idx] = raw / scale
    else:
        out_torque[env_idx, dof_idx] = raw


@wp.kernel(enable_backward=False)
def sparse_moment_length_velocity_kernel(
    qvel: wp.array2d(dtype=wp.float32),
    moment: wp.array2d(dtype=wp.float32),
    muscle_dof_offsets: wp.array(dtype=wp.int32),
    muscle_dof_torque_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    length_velocity: wp.array2d(dtype=wp.float32),
):
    """Instantaneous muscle-tendon velocity from the sparse moment arms.

    Moments follow the projector convention (generalized force per unit
    tendon force, i.e. ``-dL/dq``), so ``dL/dt = -moment . qvel``.
    """
    env_idx, muscle_idx = wp.tid()
    velocity = float(0.0)
    for entry_idx in range(
        muscle_dof_offsets[muscle_idx], muscle_dof_offsets[muscle_idx + 1]
    ):
        full_dof = muscle_dof_torque_idx[entry_idx] + root_dof_count
        velocity -= moment[env_idx, entry_idx] * qvel[env_idx, full_dof]
    length_velocity[env_idx, muscle_idx] = velocity


@wp.kernel(enable_backward=False)
def clip_torque_feasible_bounds_kernel(
    muscle_length: wp.array2d(dtype=wp.float32),
    muscle_params: wp.array2d(dtype=wp.float32),
    muscle_rest_length: wp.array(dtype=wp.float32),
    muscle_force_scale: float,
    muscle_length_norm_min: float,
    muscle_length_norm_max: float,
    active_force_model_params: wp.vec3,
    torque_active_entry_offsets: wp.array(dtype=wp.int32),
    torque_active_segment_idx: wp.array(dtype=wp.int32),
    torque_active_muscle_idx: wp.array(dtype=wp.int32),
    torque_active_start_coeff: wp.array(dtype=wp.float32),
    torque_active_end_coeff: wp.array(dtype=wp.float32),
    apply_segment_start_idx: wp.array(dtype=wp.int32),
    apply_segment_end_idx: wp.array(dtype=wp.int32),
    apply_segment_body_start_idx: wp.array(dtype=wp.int32),
    apply_segment_body_end_idx: wp.array(dtype=wp.int32),
    segment_start_root_body_idx: wp.array(dtype=wp.int32),
    segment_end_root_body_idx: wp.array(dtype=wp.int32),
    transformed_waypoints: wp.array2d(dtype=wp.vec3),
    xipos: wp.array2d(dtype=wp.vec3),
    subtree_com: wp.array2d(dtype=wp.vec3),
    cdof: wp.array3d(dtype=wp.float32),
    torque_scale: wp.array(dtype=wp.float32),
    torque_scale_inv: wp.array(dtype=wp.float32),
    root_dof_count: int,
    actions: wp.array2d(dtype=wp.float32),
):
    env_idx, dof_idx = wp.tid()
    full_dof = dof_idx + root_dof_count

    cdof_ax = cdof[env_idx, full_dof, 0]
    cdof_ay = cdof[env_idx, full_dof, 1]
    cdof_az = cdof[env_idx, full_dof, 2]
    cdof_lx = cdof[env_idx, full_dof, 3]
    cdof_ly = cdof[env_idx, full_dof, 4]
    cdof_lz = cdof[env_idx, full_dof, 5]
    cdof_angular = wp.vec3(cdof_ax, cdof_ay, cdof_az)
    cdof_linear = wp.vec3(cdof_lx, cdof_ly, cdof_lz)

    lower = float(0.0)
    upper = float(0.0)
    current_muscle = int(-1)
    muscle_contrib = float(0.0)
    entry_begin = torque_active_entry_offsets[dof_idx]
    entry_end = torque_active_entry_offsets[dof_idx + 1]

    for entry_idx in range(entry_begin, entry_end):
        m_idx = torque_active_muscle_idx[entry_idx]

        if m_idx != current_muscle:
            if current_muscle >= 0:
                if muscle_contrib > 0.0:
                    upper += muscle_contrib
                elif muscle_contrib < 0.0:
                    lower += muscle_contrib
            current_muscle = m_idx
            muscle_contrib = 0.0

        f0 = muscle_params[m_idx, 0]
        l_m0 = muscle_params[m_idx, 1]
        l_t0 = muscle_params[m_idx, 2]
        pen_angle = muscle_params[m_idx, 3]
        muscle_tendon_length = muscle_length[env_idx, m_idx]
        l_m_norm = normalized_muscle_length_wp(
            muscle_tendon_length,
            l_m0,
            l_t0,
            muscle_rest_length[m_idx],
            muscle_length_norm_min,
            muscle_length_norm_max,
        )
        unit_force_mag = unit_force_magnitude_from_length_norm_wp(
            f0,
            pen_angle,
            muscle_force_scale,
            l_m_norm,
            active_force_model_params,
        )

        segment_idx = torque_active_segment_idx[entry_idx]
        p_start = transformed_waypoints[env_idx, apply_segment_start_idx[segment_idx]]
        p_end = transformed_waypoints[env_idx, apply_segment_end_idx[segment_idx]]
        subtree_com_s = subtree_com[env_idx, segment_start_root_body_idx[segment_idx]]
        subtree_com_e = subtree_com[env_idx, segment_end_root_body_idx[segment_idx]]
        muscle_contrib += segment_generalized_force_contribution_wp(
            cdof_linear,
            cdof_angular,
            p_start,
            p_end,
            subtree_com_s,
            subtree_com_e,
            torque_active_start_coeff[entry_idx],
            torque_active_end_coeff[entry_idx],
            unit_force_mag,
        )

    if current_muscle >= 0:
        if muscle_contrib > 0.0:
            upper += muscle_contrib
        elif muscle_contrib < 0.0:
            lower += muscle_contrib

    scale = torque_scale[dof_idx]
    scale_inv_val = torque_scale_inv[dof_idx]
    physical = actions[env_idx, dof_idx] * scale
    physical = wp.clamp(physical, lower, upper)
    actions[env_idx, dof_idx] = physical * scale_inv_val


__all__ = [
    "active_force_length_wp",
    "apply_lbs_muscle_force_velocity_to_force_cache_kernel",
    "apply_lbs_muscle_force_velocity_to_unit_force_kernel",
    "apply_muscle_implicit_qderiv_gather",
    "apply_muscle_implicit_qderiv_kernel",
    "apply_precomputed_muscle_implicit_qderiv_kernel",
    "MuscleImplicitQderivGather",
    "build_dense_muscle_dof_entries",
    "build_lbs_active_idx_body_vec2",
    "build_lbs_active_attachment_lists",
    "build_lbs_active_attachment_projection_metadata",
    "build_lbs_active_local_coeff_vec4",
    "build_lbs_body_wrench_segment_event_metadata",
    "build_lbs_native_body_wrench_sorted_metadata",
    "build_lbs_native_body_wrench_tile_metadata",
    "build_muscle_implicit_qderiv_gather",
    "build_lbs_forward_endpoint_ranges_metadata",
    "build_lbs_forward_packed_metadata",
    "build_lbs_waypoint_attachment_fixed2_metadata",
    "build_lbs_waypoint_force_metadata",
    "build_lbs_torque_forward_chunks",
    "build_body_ancestor_dof_lists",
    "build_dof_subtree_mask",
    "build_muscle_force_length_coeff",
    "build_muscle_dof_sparse_entries",
    "build_muscle_segment_offsets",
    "build_muscle_sparse_torque_entries",
    "build_torque_sparse_segment_entries",
    "clear_muscle_sparse_moment_kernel",
    "clip_torque_feasible_bounds_kernel",
    "eval_muscle_force_magnitude_kernel",
    "eval_muscle_force_magnitude_velocity_kernel",
    "compute_body_cvel_kernel",
    "eval_muscle_force_velocity_curve_kernel",
    "eval_muscle_force_velocity_state_kernel",
    "eval_lbs_muscle_sparse_moment_kernel",
    "eval_lbs_muscle_sparse_moment_atomic_kernel",
    "eval_muscle_lbs_qfrc_pair_vec4_native_tiled_partials_kernel",
    "eval_muscle_qfrc_kernel",
    "eval_muscle_sparse_moment_kernel",
    "launch_muscle_sparse_moment",
    "extract_applied_torque_kernel",
    "normalize_quat_wp",
    "native_load_vec2i_wp",
    "native_load_vec4i_wp",
    "native_load_vec4_wp",
    "normalized_muscle_length_wp",
    "passive_force_length_wp",
    "project_muscle_sparse_moment_qfrc_kernel",
    "prepare_muscle_implicit_qderiv_pairs",
    "reduce_lbs_qfrc_chunk_partials_kernel",
    "segment_generalized_force_contribution_wp",
    "build_torque_sparse_lbs_segment_entries",
    "build_muscle_sparse_lbs_torque_entries",
    "eval_muscle_lbs_qfrc_kernel",
    "transform_lbs_waypoints_kernel",
    "transform_waypoints_kernel",
    "unit_force_magnitude_from_length_norm_wp",
    "sparse_moment_length_velocity_kernel",
    "write_generalized_force_actions_indexed_kernel",
    "write_generalized_force_actions_kernel",
]
