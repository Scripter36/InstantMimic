from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import warp as wp

from instantmimic.muscle.parsing.mjcf import load_pickle_cache, mjcf_cache_dir
from instantmimic.utils.startup_profiler import startup_nvtx_wrapped

from .muscle_dynamics_kernels import (
    NATIVE_TILE_MAX_BODIES,
    _accumulate_activation_grad_direct_from_body_adjoint_kernel,
    _accumulate_lbs_activation_grad_from_body_adjoint_kernel,
    _accumulate_lbs_body_wrench_segment_event_chunk_partials_kernel,
    _accumulate_lbs_body_wrench_segment_start_event_chunk_partials_kernel,
    _accumulate_lbs_muscle_lengths_contiguous_waypoints_kernel,
    _accumulate_lbs_muscle_lengths_from_segments_kernel,
    _accumulate_lbs_muscle_lengths_muscle_contiguous_waypoints_kernel,
    _apply_muscle_force_velocity_to_unit_force_kernel,
    _compute_force_magnitudes_contiguous_waypoints_kernel,
    _compute_force_magnitudes_from_waypoints_kernel,
    _compute_lbs_body_adjoint_kernel,
    _compute_lbs_fixed1_unit_force_from_geometry_kernel,
    _compute_lbs_fixed2_unit_force_from_geometry_kernel,
    _compute_lbs_segment_length_pair_partials_kernel,
    _compute_lbs_torque_bounds_tile_partials_kernel,
    _compute_muscle_dynamics_activation_grad_sparse_kernel,
    _compute_muscle_dynamics_contribution_kernel,
    _compute_muscle_dynamics_contribution_lbs_sparse_kernel,
    _compute_muscle_dynamics_contribution_sparse_kernel,
    _compute_muscle_dynamics_lbs_sparse_bounds_kernel,
    _compute_muscle_dynamics_sparse_bounds_kernel,
    _normalize_body_quats_kernel,
    _precompute_lbs_fixed1_force_magnitude_cache_kernel,
    _precompute_lbs_fixed2_force_magnitude_cache_kernel,
    _reduce_activation_interval_bounds_kernel,
    _reduce_lbs_body_wrench_event_partials_kernel,
    _reduce_lbs_torque_bounds_chunk_partials_kernel,
    _reduce_lbs_torque_chunk_partials_kernel,
    _select_activation_interval_bounds_kernel,
    _transform_lbs_waypoints_fixed2_normalized_quat_from_torch_kernel,
    _transform_lbs_waypoints_from_torch_kernel,
    _transform_waypoints_from_torch_kernel,
    make_accumulate_body_wrench_direct_from_activation_kernel,
    make_accumulate_lbs_activation_grad_fixed1_direct_start_from_body_adjoint_kernel,
    make_accumulate_lbs_activation_grad_fixed1_direct_start_from_body_adjoint_multi_root_kernel,
    make_accumulate_lbs_activation_grad_fixed2_direct_start_from_body_adjoint_kernel,
    make_accumulate_lbs_activation_grad_fixed2_direct_start_from_body_adjoint_multi_root_kernel,
    make_accumulate_lbs_activation_grad_fixed2_start_from_body_adjoint_kernel,
    make_accumulate_lbs_body_wrench_from_activation_kernel,
    make_accumulate_lbs_body_wrench_native_tile_cached_force_fixed1_kernel,
    make_accumulate_lbs_body_wrench_native_tile_cached_force_fixed2_kernel,
    make_accumulate_lbs_body_wrench_native_tile_full_force_fixed1_kernel,
    make_accumulate_lbs_body_wrench_native_tile_full_force_fixed2_kernel,
    make_accumulate_lbs_body_wrench_native_tile_full_force_velocity_fixed1_kernel,
    make_lbs_force_by_muscle_contiguous_waypoints_precomputed_kernel,
    make_lbs_force_by_muscle_from_group_partials_precomputed_kernel,
    make_lbs_force_by_muscle_from_length_kernel,
    make_lbs_force_by_muscle_from_length_precomputed_kernel,
    make_lbs_force_by_muscle_from_pair_partials_precomputed_kernel,
    make_lbs_segment_length_group_partials_kernel,
    make_lbs_torque_waypoint_force_active_attach_pair_vec4_native_partials_kernel,
    make_lbs_waypoint_force_from_force_kernel,
    make_muscle_dynamics_from_activation_sparse_kernel,
)
from .muscle_qfrc_projector import (
    apply_lbs_muscle_force_velocity_to_force_cache_kernel,
    apply_lbs_muscle_force_velocity_to_unit_force_kernel,
    build_dof_subtree_mask,
    build_lbs_active_attachment_lists,
    build_lbs_active_attachment_projection_metadata,
    build_lbs_active_idx_body_vec2,
    build_lbs_active_local_coeff_vec4,
    build_lbs_body_wrench_segment_event_metadata,
    build_lbs_forward_packed_metadata,
    build_lbs_native_body_wrench_sorted_metadata,
    build_lbs_native_body_wrench_tile_metadata,
    build_lbs_torque_bounds_chunks,
    build_lbs_torque_forward_chunks,
    build_lbs_waypoint_attachment_fixed2_metadata,
    build_lbs_waypoint_force_metadata,
    build_muscle_force_length_coeff,
    build_muscle_segment_offsets,
    build_muscle_sparse_lbs_torque_entries,
    build_muscle_sparse_torque_entries,
    build_torque_sparse_lbs_segment_entries,
    build_torque_sparse_segment_entries,
)
from .reference import (
    active_force_length_torch,
    passive_force_length_torch,
    project_body_wrench_to_torque,
    quat_normalize,
    quat_rotate,
    segment_force_magnitude_to_body_wrench,
)
from .runtime_specs import (
    MuscleLayerRuntimeSpec,
)
from .waypoint_wrap import (
    WaypointWrapping,
    WaypointWrapState,
    force_components_kernel,
    path_lengths_kernel,
)
from .wrench_projection import (
    build_wrench_projection_topology,
    launch_project_body_wrench_to_torque,
)

# Fast paths for LBS waypoint attachments.
_FORCE_PRECOMPUTE_LBS_ATTACHMENT_LAYOUTS = {"waypoint_force", "body_wrench"}
_LBS_ATTACHMENT_LAYOUTS = _FORCE_PRECOMPUTE_LBS_ATTACHMENT_LAYOUTS
_LAYER_LBS_METADATA_CACHE_KIND = "muscle_dynamics_layer_lbs_metadata"
_LAYER_LBS_METADATA_TORCH_CACHE_VERSION = 1
_LAYER_LBS_BASE_METADATA_TENSOR_KEYS = (
    "muscle_force_length_coeff",
    "all_muscle_segment_offsets",
    "apply_muscle_segment_offsets",
)
_LAYER_LBS_ACTIVE_METADATA_TENSOR_KEYS = (
    "lbs_active_attach_offsets",
    "lbs_active_attach_body",
    "lbs_active_attach_local",
    "lbs_active_attach_coeff_weight",
    "lbs_waypoint_attachment_body2",
    "lbs_waypoint_attachment_local_weight0",
    "lbs_waypoint_attachment_local_weight1",
    "torque_active_entry_offsets",
    "torque_active_torque_idx",
    "torque_active_segment_idx",
    "torque_active_muscle_idx",
    "muscle_active_entry_offsets",
    "muscle_active_segment_idx",
    "muscle_active_torque_idx",
    "lbs_forward_segment_idx",
    "lbs_forward_muscle_idx",
    "lbs_torque_chunk_offsets",
    "lbs_torque_chunk_torque_idx",
    "lbs_torque_chunk_entry_start",
    "lbs_torque_chunk_entry_end",
    "lbs_bounds_forward_torque_idx",
    "lbs_bounds_forward_segment_idx",
    "lbs_bounds_forward_muscle_idx",
    "lbs_bounds_group_entry_start",
    "lbs_bounds_group_entry_end",
    "lbs_bounds_chunk_offsets",
    "lbs_bounds_chunk_torque_idx",
    "lbs_bounds_chunk_group_start",
    "lbs_bounds_chunk_group_end",
    "lbs_segment_waypoints",
)
_LAYER_LBS_STATIC_METADATA_TENSOR_KEYS = (
    "lbs_waypoint_segment_idx2",
    "lbs_waypoint_segment_sign2",
    "lbs_active_attach_waypoint_idx",
    "lbs_active_attach_chunk_offsets",
    "lbs_active_attach_chunk_info",
    "lbs_body_event_chunk_offsets",
    "lbs_body_event_chunk_info",
    "lbs_body_event_packed_indices",
    "lbs_body_event_local_force_coeff",
    "lbs_body_segment_event_start_muscle",
    "lbs_native_body_tile_info",
    "lbs_native_body_tile_extra_info",
    "lbs_native_body_tile_body_ids",
    "lbs_native_body_tile_body_output_idx",
    "lbs_native_body_tile_event_start_slot_muscle",
    "lbs_native_body_tile_event_local_force_coeff",
    "lbs_native_body_tile_body_slot_offsets",
)
_LOG = logging.getLogger(__name__)


def _metadata_buffer(
    tensor: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out = tensor.detach()
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype=dtype)
    return out.contiguous()


class _LazyIntList:
    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor.detach()
        self._values: list[int] | None = None

    def _materialize(self) -> list[int]:
        values = self._values
        if values is None:
            values = [int(v) for v in self._tensor.detach().cpu().tolist()]
            self._values = values
        return values

    def __getitem__(self, idx: int) -> int:
        return self._materialize()[idx]

    def __len__(self) -> int:
        return int(self._tensor.numel()) if self._values is None else len(self._values)


def _layer_lbs_torch_cache_path(key: str) -> Path | None:
    cache_dir = mjcf_cache_dir()
    if cache_dir is None:
        return None
    return cache_dir / f"{_LAYER_LBS_METADATA_CACHE_KIND}-{key}.pt"


def _store_layer_lbs_metadata_cache(key: str, value: dict[str, object]) -> None:
    cache_path = _layer_lbs_torch_cache_path(key)
    if cache_path is None:
        return
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            torch.save(
                {
                    "format_version": _LAYER_LBS_METADATA_TORCH_CACHE_VERSION,
                    "kind": _LAYER_LBS_METADATA_CACHE_KIND,
                    "value": value,
                },
                handle,
            )
        os.replace(tmp_path, cache_path)
    except (OSError, RuntimeError):
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)


def _load_layer_lbs_metadata_cache(key: str | None) -> dict[str, object] | None:
    if key is None:
        return None
    cache_path = _layer_lbs_torch_cache_path(key)
    if cache_path is not None and cache_path.exists():
        try:
            payload = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, RuntimeError, ValueError):
            payload = None
        if (
            isinstance(payload, dict)
            and payload.get("format_version") == _LAYER_LBS_METADATA_TORCH_CACHE_VERSION
            and payload.get("kind") == _LAYER_LBS_METADATA_CACHE_KIND
            and isinstance(payload.get("value"), dict)
        ):
            return payload["value"]

    value = load_pickle_cache(_LAYER_LBS_METADATA_CACHE_KIND, key)
    if isinstance(value, dict):
        _store_layer_lbs_metadata_cache(key, value)
        return value
    return None


class _WaypointGeometry(NamedTuple):
    unit_force_magnitude: torch.Tensor
    passive_force_magnitude: torch.Tensor | None
    seg_start: torch.Tensor | None
    seg_end: torch.Tensor | None
    seg_dir: torch.Tensor | None
    transformed_waypoints: torch.Tensor
    xpos: torch.Tensor
    xquat: torch.Tensor
    xipos: torch.Tensor
    subtree_com: torch.Tensor
    cdof: torch.Tensor
    muscle_length: torch.Tensor


class _SegmentProjectionTerms(NamedTuple):
    unit_seg_force_t: torch.Tensor
    neg_unit_seg_force_t: torch.Tensor
    unit_torque_start_t: torch.Tensor
    unit_torque_end_t: torch.Tensor
    offset_cross_force_start_t: torch.Tensor
    offset_cross_force_end_t: torch.Tensor
    muscle_index: torch.Tensor


class _WarpMuscleDynamicsFromActivation(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        activation: torch.Tensor,
        raw_context: torch.Tensor,
        layer: MuscleDynamicsLayer,
        include_passive_force: bool,
    ) -> torch.Tensor:
        mean, transformed_waypoints, unit_force_magnitude = (
            layer._warp_muscle_dynamics_from_activation_forward(
                activation=activation,
                raw_context=raw_context,
                include_passive_force=include_passive_force,
            )
        )
        ctx.layer = layer
        ctx.scratch_generation = layer._geometry_scratch_generation
        if getattr(layer, "_uses_native_body_wrench_projection", False) and not getattr(
            layer, "muscle_force_velocity_enabled", False
        ):
            transformed_waypoints = raw_context.new_empty(0)
            unit_force_magnitude = raw_context.new_empty(0)
        ctx.save_for_backward(
            raw_context,
            transformed_waypoints,
            unit_force_magnitude,
        )
        return mean

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        raw_context, transformed_waypoints, unit_force_magnitude = ctx.saved_tensors
        grad_activation = ctx.layer._warp_muscle_dynamics_from_activation_backward(
            raw_context=raw_context,
            grad_output=grad_output,
            transformed_waypoints=transformed_waypoints,
            unit_force_magnitude=unit_force_magnitude,
            scratch_generation=ctx.scratch_generation,
        )
        return grad_activation, None, None, None


class _WarpMuscleDynamicsFromActivationCachedLBSForce(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        activation: torch.Tensor,
        raw_context: torch.Tensor,
        unit_force_magnitude: torch.Tensor,
        passive_force_magnitude: torch.Tensor,
        layer: MuscleDynamicsLayer,
        include_passive_force: bool,
    ) -> torch.Tensor:
        mean = layer._warp_muscle_dynamics_from_activation_forward_cached_lbs_force(
            activation=activation,
            raw_context=raw_context,
            unit_force_magnitude=unit_force_magnitude,
            passive_force_magnitude=passive_force_magnitude,
            include_passive_force=include_passive_force,
        )
        ctx.layer = layer
        ctx.save_for_backward(raw_context, unit_force_magnitude)
        return mean

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        raw_context, unit_force_magnitude = ctx.saved_tensors
        grad_activation = ctx.layer._warp_muscle_dynamics_from_activation_backward(
            raw_context=raw_context,
            grad_output=grad_output,
            transformed_waypoints=None,
            unit_force_magnitude=unit_force_magnitude,
        )
        return grad_activation, None, None, None, None, None


@torch.compiler.disable
def _warp_muscle_dynamics_from_activation_eager(
    activation: torch.Tensor,
    raw_context: torch.Tensor,
    layer: MuscleDynamicsLayer,
    *,
    include_passive_force: bool,
) -> torch.Tensor:
    """Keep Warp custom autograd outside Dynamo fullgraph traces.

    Coordinator training can compile the surrounding tensor math while the
    Warp-backed muscle forward/backward remains eager and capture-safe.
    """

    return _WarpMuscleDynamicsFromActivation.apply(
        activation,
        raw_context,
        layer,
        include_passive_force,
    )


@torch.compiler.disable
def _warp_muscle_dynamics_from_activation_cached_lbs_force_eager(
    activation: torch.Tensor,
    raw_context: torch.Tensor,
    unit_force_magnitude: torch.Tensor,
    passive_force_magnitude: torch.Tensor,
    layer: MuscleDynamicsLayer,
    *,
    include_passive_force: bool,
) -> torch.Tensor:
    return _WarpMuscleDynamicsFromActivationCachedLBSForce.apply(
        activation,
        raw_context,
        unit_force_magnitude,
        passive_force_magnitude,
        layer,
        include_passive_force,
    )


class MuscleDynamicsLayer(nn.Module):
    """Differentiable projection from muscle activations to joint torques.

    Supported production CUDA execution is Warp-only. The pure-Torch branches
    are deterministic CPU numerical oracles for tests, not runtime fallbacks.
    """

    @staticmethod
    def _build_muscle_segment_offsets(
        segment_muscle_idx: torch.Tensor,
        muscle_count: int,
    ) -> torch.Tensor:
        return build_muscle_segment_offsets(
            segment_muscle_idx=segment_muscle_idx,
            muscle_count=muscle_count,
        )

    @staticmethod
    def _build_torque_sparse_segment_entries(
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
        return build_torque_sparse_segment_entries(
            segment_start_dof_mask=segment_start_dof_mask,
            segment_end_dof_mask=segment_end_dof_mask,
            apply_segment_muscle_idx=apply_segment_muscle_idx,
        )

    @staticmethod
    def _build_muscle_sparse_torque_entries(
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
        return build_muscle_sparse_torque_entries(
            torque_entry_offsets=torque_entry_offsets,
            torque_segment_idx=torque_segment_idx,
            torque_start_coeff=torque_start_coeff,
            torque_end_coeff=torque_end_coeff,
            apply_segment_muscle_idx=apply_segment_muscle_idx,
            muscle_count=muscle_count,
        )

    #: Which implementation `forward` runs. "auto" keeps the historical choice
    #: -- Warp on CUDA, the Torch reference elsewhere -- and the two named
    #: values pin one of them, which is how the equivalence tests put the same
    #: inputs through both on one device.
    FORWARD_BACKENDS = ("auto", "reference", "warp")

    @startup_nvtx_wrapped("muscle_dynamics_layer:init")
    def __init__(self, spec: MuscleLayerRuntimeSpec) -> None:
        super().__init__()
        self._forward_backend = "auto"
        init_profile_records: list[tuple[str, float]] = []
        init_profile_start = perf_counter()
        init_profile_last = init_profile_start

        def _record_init_phase(name: str) -> None:
            nonlocal init_profile_last
            now = perf_counter()
            init_profile_records.append((name, now - init_profile_last))
            init_profile_last = now

        dimensions = spec.dimensions
        output = spec.output
        mechanics = spec.mechanics
        projection = spec.projection
        topology = spec.topology

        self.body_count = dimensions.body_count
        self.muscle_count = dimensions.muscle_count
        self.action_dim = output.action_dim
        self.torque_dim = dimensions.torque_dim
        self.context_dim = dimensions.obs_context_dim
        self.geometry_backend = projection.geometry_backend.strip().lower()
        self.muscle_force_scale = mechanics.force_scale
        self.muscle_active_force_type = mechanics.active_force_type
        self.muscle_active_force_gamma = mechanics.active_force_gamma
        self.muscle_active_force_lmin = mechanics.active_force_lmin
        self.muscle_active_force_lmax = mechanics.active_force_lmax
        if self.muscle_active_force_type not in {"gaussian", "mujoco"}:
            raise ValueError("muscle_active_force_type must be 'gaussian' or 'mujoco'")
        if self.muscle_active_force_gamma <= 0.0:
            raise ValueError("Gaussian active-force gamma must be positive")
        if not (
            0.0 < self.muscle_active_force_lmin < 1.0
            and self.muscle_active_force_lmax > 1.0
        ):
            raise ValueError(
                "MuJoCo active-force lmin/lmax must satisfy 0 < lmin < 1 < lmax"
            )
        self._muscle_active_force_model_params_wp = wp.vec3(
            0.0 if self.muscle_active_force_type == "gaussian" else 1.0,
            (
                self.muscle_active_force_gamma
                if self.muscle_active_force_type == "gaussian"
                else self.muscle_active_force_lmin
            ),
            (
                0.0
                if self.muscle_active_force_type == "gaussian"
                else self.muscle_active_force_lmax
            ),
        )
        self.muscle_force_velocity_enabled = mechanics.force_velocity_enabled
        self.muscle_force_velocity_vmax = mechanics.force_velocity_vmax
        self.muscle_force_velocity_fvmax = mechanics.force_velocity_fvmax
        self.muscle_length_norm_min = mechanics.length_norm_min
        self.muscle_length_norm_max = mechanics.length_norm_max
        self.muscle_passive_force_type = mechanics.passive_force_type
        self.muscle_passive_force_k_pe = mechanics.passive_force_k_pe
        self.muscle_passive_force_e_mo = mechanics.passive_force_e_mo
        self.muscle_passive_force_lmax = mechanics.passive_force_lmax
        self.muscle_passive_force_fpmax = mechanics.passive_force_fpmax
        if self.muscle_passive_force_type not in {"exponential", "mujoco"}:
            raise ValueError(
                "muscle_passive_force_type must be 'exponential' or 'mujoco'"
            )
        if (
            self.muscle_passive_force_k_pe <= 0.0
            or self.muscle_passive_force_e_mo <= 0.0
        ):
            raise ValueError("exponential passive-force k_pe and e_mo must be positive")
        if (
            self.muscle_passive_force_lmax <= 1.0
            or self.muscle_passive_force_fpmax < 0.0
        ):
            raise ValueError(
                "MuJoCo passive-force lmax must exceed 1 and fpmax must be nonnegative"
            )
        self.muscle_passive_force_model_id = (
            0 if self.muscle_passive_force_type == "exponential" else 1
        )
        self.muscle_passive_force_param0 = (
            self.muscle_passive_force_k_pe
            if self.muscle_passive_force_model_id == 0
            else self.muscle_passive_force_lmax
        )
        self.muscle_passive_force_param1 = (
            self.muscle_passive_force_e_mo
            if self.muscle_passive_force_model_id == 0
            else self.muscle_passive_force_fpmax
        )
        self._muscle_passive_force_model_params_wp = wp.vec3(
            float(self.muscle_passive_force_model_id),
            self.muscle_passive_force_param0,
            self.muscle_passive_force_param1,
        )
        self.include_passive_force_in_output = mechanics.include_passive_force_in_output
        self.sparse_body_wrench_forward = projection.sparse_body_wrench_forward
        self.bound_chunk_size = output.bound_chunk_size
        self._xpos_width = self.body_count * 3
        self._xquat_width = self.body_count * 4
        self._xipos_width = self.body_count * 3
        self.full_torque_dim = dimensions.full_torque_dim
        self.root_dof_count = self.full_torque_dim - self.torque_dim
        self._subtree_com_width = self.body_count * 3
        self._cdof_width = self.full_torque_dim * 6
        self._geometry_context_dim = (
            self._xpos_width
            + self._xquat_width
            + self._xipos_width
            + self._subtree_com_width
            + self._cdof_width
        )
        expected_context_dim = self._geometry_context_dim
        if self.muscle_force_velocity_enabled:
            expected_context_dim += self.full_torque_dim
        if self.context_dim != expected_context_dim:
            raise ValueError(
                "Invalid muscle dynamics geometry context size: "
                f"expected {expected_context_dim}, got {self.context_dim}."
            )
        _record_init_phase("config_and_validation")
        self.uses_lbs_waypoint_attachments = topology.uses_lbs_waypoint_attachments
        # Bumped by every geometry preparation. A forward saves geometry
        # scratch views for its backward; a later preparation overwrites them.
        self._geometry_scratch_generation = 0
        self._waypoint_wrapping = WaypointWrapping(
            topology.wraps,
            int(topology.muscle_waypoints_local.shape[0]),
            topology.muscle_params.device,
        )
        self._waypoint_wrap_states = {}
        self.register_buffer(
            "muscle_waypoint_start_i32",
            torch.cat(
                (
                    torch.zeros(
                        1, device=topology.muscle_params.device, dtype=torch.int32
                    ),
                    torch.bincount(
                        topology.all_segment_muscle_idx.long(),
                        minlength=self.muscle_count,
                    )
                    .to(torch.int32)
                    .add(1)
                    .cumsum(0)
                    .to(torch.int32),
                )
            ),
            persistent=False,
        )
        native_body_wrench_requested = bool(
            self.geometry_backend == "waypoint"
            and projection.lbs_attachment_layout.strip().lower() == "body_wrench"
            and projection.lbs_body_wrench_forward_mode.strip().lower() == "native_tile"
        )
        lbs_metadata_cache_key = projection.static_metadata_cache_key
        cached_lbs_metadata = (
            _load_layer_lbs_metadata_cache(lbs_metadata_cache_key)
            if (self.uses_lbs_waypoint_attachments or native_body_wrench_requested)
            else None
        )
        cached_lbs_base_metadata_ready = isinstance(cached_lbs_metadata, dict) and all(
            torch.is_tensor(cached_lbs_metadata.get(key))
            for key in _LAYER_LBS_BASE_METADATA_TENSOR_KEYS
        )
        cached_lbs_active_metadata_ready = (
            isinstance(cached_lbs_metadata, dict)
            and all(
                torch.is_tensor(cached_lbs_metadata.get(key))
                for key in _LAYER_LBS_ACTIVE_METADATA_TENSOR_KEYS
            )
            and isinstance(
                cached_lbs_metadata.get("lbs_waypoint_attachment_fixed2"),
                bool,
            )
        )
        _record_init_phase("lbs_metadata_cache_load")

        self.register_buffer(
            "muscle_params",
            _metadata_buffer(topology.muscle_params, torch.float32),
        )
        self.register_buffer(
            "muscle_rest_length",
            _metadata_buffer(topology.muscle_rest_length, torch.float32),
        )
        self.register_buffer(
            "muscle_force_velocity_length_scale",
            _metadata_buffer(
                topology.muscle_force_velocity_length_scale,
                torch.float32,
            ),
        )
        # The static cache is shared across force settings. Rebuild these
        # mechanics-dependent coefficients before sorting native tile metadata.
        muscle_force_length_coeff = build_muscle_force_length_coeff(
            self.muscle_params,
            self.muscle_rest_length,
            self.muscle_force_scale,
        )
        self.register_buffer(
            "muscle_force_length_coeff",
            muscle_force_length_coeff,
            persistent=False,
        )
        _record_init_phase("base_force_metadata")
        self.register_buffer(
            "muscle_waypoints_local",
            _metadata_buffer(topology.muscle_waypoints_local, torch.float32),
        )
        self.register_buffer(
            "muscle_waypoint_body",
            _metadata_buffer(topology.muscle_waypoint_body, torch.long),
        )
        self.register_buffer(
            "muscle_waypoint_body_i32",
            _metadata_buffer(topology.muscle_waypoint_body, torch.int32),
            persistent=False,
        )
        waypoint_count = int(self.muscle_waypoints_local.shape[0])
        attachment_start = topology.muscle_waypoint_attachment_start
        attachment_body = topology.muscle_waypoint_attachment_body
        attachment_local = topology.muscle_waypoint_attachment_local
        attachment_weight = topology.muscle_waypoint_attachment_weight
        self.register_buffer(
            "muscle_waypoint_attachment_start",
            _metadata_buffer(attachment_start, torch.long),
            persistent=False,
        )
        self.register_buffer(
            "muscle_waypoint_attachment_start_i32",
            _metadata_buffer(attachment_start, torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "muscle_waypoint_attachment_body",
            _metadata_buffer(attachment_body, torch.long),
            persistent=False,
        )
        self.register_buffer(
            "muscle_waypoint_attachment_body_i32",
            _metadata_buffer(attachment_body, torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "muscle_waypoint_attachment_local",
            _metadata_buffer(attachment_local, torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "muscle_waypoint_attachment_weight",
            _metadata_buffer(attachment_weight, torch.float32),
            persistent=False,
        )
        _record_init_phase("base_waypoint_attachment_buffers")
        self.register_buffer(
            "all_segment_muscle_idx",
            _metadata_buffer(topology.all_segment_muscle_idx, torch.long),
        )
        self.register_buffer(
            "all_segment_muscle_idx_i32",
            _metadata_buffer(topology.all_segment_muscle_idx, torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "all_segment_start_idx",
            _metadata_buffer(topology.all_segment_start_idx, torch.long),
        )
        self.register_buffer(
            "all_segment_start_idx_i32",
            _metadata_buffer(topology.all_segment_start_idx, torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "apply_segment_muscle_idx",
            _metadata_buffer(topology.apply_segment_muscle_idx, torch.long),
        )
        self.register_buffer(
            "apply_segment_muscle_idx_i32",
            _metadata_buffer(topology.apply_segment_muscle_idx, torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "apply_segment_start_idx",
            _metadata_buffer(topology.apply_segment_start_idx, torch.long),
        )
        self.register_buffer(
            "apply_segment_start_idx_i32",
            _metadata_buffer(topology.apply_segment_start_idx, torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "apply_segment_end_idx",
            _metadata_buffer(topology.apply_segment_end_idx, torch.long),
        )
        self.register_buffer(
            "apply_segment_end_idx_i32",
            _metadata_buffer(topology.apply_segment_end_idx, torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "apply_segment_body_start_idx",
            _metadata_buffer(topology.apply_segment_body_start_idx, torch.long),
        )
        self.register_buffer(
            "apply_segment_body_start_idx_i32",
            _metadata_buffer(topology.apply_segment_body_start_idx, torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "apply_segment_body_end_idx",
            _metadata_buffer(topology.apply_segment_body_end_idx, torch.long),
        )
        self.register_buffer(
            "apply_segment_body_end_idx_i32",
            _metadata_buffer(topology.apply_segment_body_end_idx, torch.int32),
            persistent=False,
        )
        if cached_lbs_base_metadata_ready and isinstance(cached_lbs_metadata, dict):
            all_muscle_segment_offsets = cached_lbs_metadata[
                "all_muscle_segment_offsets"
            ]
            apply_muscle_segment_offsets = cached_lbs_metadata[
                "apply_muscle_segment_offsets"
            ]
        else:
            all_muscle_segment_offsets = self._build_muscle_segment_offsets(
                self.all_segment_muscle_idx,
                self.muscle_count,
            )
            apply_muscle_segment_offsets = self._build_muscle_segment_offsets(
                self.apply_segment_muscle_idx,
                self.muscle_count,
            )
        self.register_buffer(
            "all_muscle_segment_offsets",
            all_muscle_segment_offsets,
            persistent=False,
        )
        self.register_buffer(
            "apply_muscle_segment_offsets",
            apply_muscle_segment_offsets,
            persistent=False,
        )
        _record_init_phase("base_segment_metadata")
        self.register_buffer(
            "body_parentid",
            _metadata_buffer(topology.body_parentid, torch.long),
        )
        self.register_buffer(
            "body_rootid",
            _metadata_buffer(topology.body_rootid, torch.long),
        )
        self.register_buffer(
            "body_rootid_i32",
            _metadata_buffer(topology.body_rootid, torch.int32),
            persistent=False,
        )
        body_rootid_cpu = topology.body_rootid.detach().cpu().long()
        if int(body_rootid_cpu.numel()) > 0 and bool(
            torch.all(body_rootid_cpu == body_rootid_cpu[0]).item()
        ):
            self.lbs_single_root_body_idx = int(body_rootid_cpu[0].item())
        else:
            self.lbs_single_root_body_idx = -1
        self.register_buffer(
            "dof_bodyid",
            _metadata_buffer(topology.dof_bodyid, torch.long),
        )
        self.register_buffer(
            "dof_subtree_mask",
            self._build_dof_subtree_mask(
                self.body_parentid,
                self.dof_bodyid,
                self.body_count,
                self.full_torque_dim,
            ),
        )
        _record_init_phase("base_body_dof_metadata")
        self.bound_chunk_size = min(self.bound_chunk_size, max(self.torque_dim, 1))
        self.lbs_forward_chunk_size = max(
            1,
            projection.lbs_forward_chunk_size,
        )
        self.lbs_forward_tile_block_dim = max(
            1,
            projection.lbs_forward_tile_block_dim,
        )
        self.lbs_body_event_chunk_size = max(
            1,
            projection.lbs_body_event_chunk_size,
        )
        self.lbs_forward_reduction = projection.lbs_forward_reduction.strip().lower()
        self.lbs_attachment_layout = projection.lbs_attachment_layout.strip().lower()
        self.lbs_body_wrench_forward_mode = (
            projection.lbs_body_wrench_forward_mode.strip().lower()
        )
        self.lbs_body_wrench_native_tile_muscle_cap = max(
            1,
            projection.lbs_body_wrench_native_tile_muscle_cap,
        )
        self.lbs_body_wrench_native_tile_segment_cap = max(
            1,
            projection.lbs_body_wrench_native_tile_segment_cap,
        )
        self.lbs_body_wrench_native_tile_event_cap = max(
            1,
            projection.lbs_body_wrench_native_tile_event_cap,
        )
        self.lbs_body_wrench_native_tile_body_cap = max(
            1,
            projection.lbs_body_wrench_native_tile_body_cap,
        )
        if self.lbs_forward_reduction not in {"partial", "serial"}:
            raise ValueError(
                "lbs_forward_reduction must be 'partial' or 'serial', "
                f"got {self.lbs_forward_reduction!r}."
            )
        if self.lbs_attachment_layout not in _LBS_ATTACHMENT_LAYOUTS:
            raise ValueError(
                "lbs_attachment_layout must be one of "
                f"{sorted(_LBS_ATTACHMENT_LAYOUTS)!r}, "
                f"got {self.lbs_attachment_layout!r}."
            )
        if self.lbs_body_wrench_forward_mode not in {"pairchain", "native_tile"}:
            raise ValueError(
                "lbs_body_wrench_forward_mode must be 'pairchain' or 'native_tile', "
                f"got {self.lbs_body_wrench_forward_mode!r}."
            )
        if self.lbs_body_wrench_forward_mode == "native_tile":
            if self.lbs_attachment_layout != "body_wrench":
                raise ValueError(
                    "lbs_body_wrench_forward_mode='native_tile' requires "
                    "lbs_attachment_layout='body_wrench'."
                )
            if self.lbs_body_wrench_native_tile_muscle_cap > 32:
                raise ValueError("native_tile supports at most 32 muscles per tile.")
            if (
                self.lbs_body_wrench_native_tile_muscle_cap
                + self.lbs_body_wrench_native_tile_segment_cap
                > 288
            ):
                raise ValueError("native_tile supports at most 288 waypoints per tile.")
            if self.lbs_body_wrench_native_tile_body_cap > NATIVE_TILE_MAX_BODIES:
                raise ValueError(
                    f"native_tile supports at most {NATIVE_TILE_MAX_BODIES} "
                    "bodies per tile."
                )
            if self.lbs_forward_tile_block_dim not in {32, 64, 128, 256}:
                raise ValueError(
                    "native_tile requires lbs_forward_tile_block_dim to be one of "
                    "32, 64, 128, or 256."
                )
        self._lbs_native_body_wrench_full_force_kernels = None
        self._lbs_native_body_wrench_full_force_velocity_kernels = None
        self._lbs_native_full_force_velocity_factory_args = None
        self._lbs_native_body_wrench_cached_force_kernel = None
        self._lbs_native_body_wrench_direct_force_kernel = None
        self.lbs_native_body_wrench_compiled_muscle_cap = 0
        self.lbs_native_body_wrench_compiled_waypoint_cap = 0
        self.lbs_native_body_wrench_compiled_body_cap = 0
        if (
            self.uses_lbs_waypoint_attachments
            and self.lbs_forward_reduction != "partial"
        ):
            raise ValueError(
                "LBS waypoint attachments require lbs_forward_reduction='partial'."
            )
        if self.lbs_forward_tile_block_dim > self.lbs_forward_chunk_size:
            self.lbs_forward_tile_block_dim = self.lbs_forward_chunk_size
        _record_init_phase("lbs_mode_validation")
        segment_idx_cpu = torch.arange(
            int(self.apply_segment_start_idx_i32.numel()),
            dtype=torch.int64,
        )
        segment_muscle_cpu = (
            self.apply_segment_muscle_idx_i32.detach().cpu().to(torch.int64)
        )
        segment_start_cpu = (
            self.apply_segment_start_idx_i32.detach().cpu().to(torch.int64)
        )
        self.lbs_segment_start_is_segment_plus_muscle = bool(
            torch.all(segment_start_cpu == segment_idx_cpu + segment_muscle_cpu).item()
        )
        all_segment_idx_cpu = torch.arange(
            int(self.all_segment_start_idx_i32.numel()),
            dtype=torch.int64,
        )
        all_segment_muscle_cpu = (
            self.all_segment_muscle_idx_i32.detach().cpu().to(torch.int64)
        )
        all_segment_start_cpu = (
            self.all_segment_start_idx_i32.detach().cpu().to(torch.int64)
        )
        self.all_segment_start_is_segment_plus_muscle = bool(
            torch.all(
                all_segment_start_cpu == all_segment_idx_cpu + all_segment_muscle_cpu
            ).item()
        )
        self.lbs_muscle_length_by_muscle_min_batch = max(
            1,
            projection.lbs_muscle_length_by_muscle_min_batch,
        )
        self.lbs_body_wrench_force_segment_group_width = 2
        attachment_start_cpu = (
            self.muscle_waypoint_attachment_start.detach().cpu().long()
        )
        expected_attachment_start = torch.arange(waypoint_count + 1, dtype=torch.long)
        single_waypoint_attachment = bool(
            torch.equal(attachment_start_cpu, expected_attachment_start)
        )
        self.lbs_waypoint_attachment_fixed1 = single_waypoint_attachment
        self._uses_attachment_projection = bool(
            self.uses_lbs_waypoint_attachments
            or (native_body_wrench_requested and single_waypoint_attachment)
        )
        attachment_body_cpu = self.muscle_waypoint_attachment_body.detach().cpu().long()
        attachment_local_cpu = (
            self.muscle_waypoint_attachment_local.detach().cpu().float()
        )
        attachment_weight_cpu = (
            self.muscle_waypoint_attachment_weight.detach().cpu().float()
        )
        waypoint_body_cpu = self.muscle_waypoint_body.detach().cpu().long()
        waypoint_local_cpu = self.muscle_waypoints_local.detach().cpu().float()
        direct_attachment_matches_waypoint = bool(
            single_waypoint_attachment
            and int(attachment_body_cpu.numel()) == waypoint_count
            and torch.equal(attachment_body_cpu, waypoint_body_cpu)
            and torch.equal(attachment_local_cpu, waypoint_local_cpu)
            and torch.all(attachment_weight_cpu == 1.0).item()
        )
        if int(self.apply_segment_start_idx.numel()) > 0:
            apply_start_cpu = self.apply_segment_start_idx.detach().cpu().long()
            apply_end_cpu = self.apply_segment_end_idx.detach().cpu().long()
            apply_body_start_cpu = (
                self.apply_segment_body_start_idx.detach().cpu().long()
            )
            apply_body_end_cpu = self.apply_segment_body_end_idx.detach().cpu().long()
            direct_segment_bodies_match = bool(
                torch.equal(
                    waypoint_body_cpu.index_select(0, apply_start_cpu),
                    apply_body_start_cpu,
                )
                and torch.equal(
                    waypoint_body_cpu.index_select(0, apply_end_cpu),
                    apply_body_end_cpu,
                )
            )
        else:
            direct_segment_bodies_match = True
        sparse_body_wrench_direct_available = bool(
            self.sparse_body_wrench_forward
            and not self.uses_lbs_waypoint_attachments
            and direct_attachment_matches_waypoint
            and direct_segment_bodies_match
        )
        # Tangents move within their wrapping body. Their force lever arms must
        # come from solved world points, never the static placeholder attachments.
        if self._waypoint_wrapping.count and not sparse_body_wrench_direct_available:
            self.sparse_body_wrench_forward = False
        self.sparse_body_wrench_direct = bool(
            sparse_body_wrench_direct_available
            and (projection.sparse_body_wrench_direct or self._waypoint_wrapping.count)
        )
        self._sparse_body_wrench_direct_backward_requested = bool(
            projection.sparse_body_wrench_direct_backward
            or self._waypoint_wrapping.count
        )
        self.sparse_body_wrench_direct_backward = bool(
            sparse_body_wrench_direct_available
            and self._sparse_body_wrench_direct_backward_requested
        )
        _record_init_phase("direct_path_analysis")
        if self._uses_attachment_projection:
            if int(attachment_body_cpu.numel()) > 0:
                attachment_root_cpu = body_rootid_cpu.index_select(
                    0,
                    attachment_body_cpu,
                )
                if bool(
                    torch.all(attachment_root_cpu == attachment_root_cpu[0]).item()
                ):
                    self.lbs_single_root_body_idx = int(attachment_root_cpu[0].item())
                else:
                    self.lbs_single_root_body_idx = -1
            if cached_lbs_active_metadata_ready:
                if not isinstance(cached_lbs_metadata, dict):
                    raise ValueError(
                        "Cached LBS metadata payload must be a dictionary."
                    )
                lbs_active_attach_offsets = cached_lbs_metadata[
                    "lbs_active_attach_offsets"
                ]
                lbs_active_attach_body = cached_lbs_metadata["lbs_active_attach_body"]
                lbs_active_attach_local = cached_lbs_metadata["lbs_active_attach_local"]
                lbs_active_attach_coeff_weight = cached_lbs_metadata[
                    "lbs_active_attach_coeff_weight"
                ]
                lbs_waypoint_attachment_body2 = cached_lbs_metadata[
                    "lbs_waypoint_attachment_body2"
                ]
                lbs_waypoint_attachment_local_weight0 = cached_lbs_metadata[
                    "lbs_waypoint_attachment_local_weight0"
                ]
                lbs_waypoint_attachment_local_weight1 = cached_lbs_metadata[
                    "lbs_waypoint_attachment_local_weight1"
                ]
                self.lbs_waypoint_attachment_fixed2 = bool(
                    cached_lbs_metadata["lbs_waypoint_attachment_fixed2"]
                )
            else:
                (
                    lbs_active_attach_offsets,
                    lbs_active_attach_body,
                    lbs_active_attach_local,
                    lbs_active_attach_coeff_weight,
                ) = self._build_lbs_active_attachment_lists(
                    dof_subtree_mask=self.dof_subtree_mask,
                    attachment_start=self.muscle_waypoint_attachment_start,
                    attachment_body=self.muscle_waypoint_attachment_body,
                    attachment_local=self.muscle_waypoint_attachment_local,
                    attachment_weight=self.muscle_waypoint_attachment_weight,
                    root_dof_count=self.root_dof_count,
                    torque_dim=self.torque_dim,
                    waypoint_count=waypoint_count,
                )
                try:
                    (
                        lbs_waypoint_attachment_body2,
                        lbs_waypoint_attachment_local_weight0,
                        lbs_waypoint_attachment_local_weight1,
                    ) = build_lbs_waypoint_attachment_fixed2_metadata(
                        attachment_start=self.muscle_waypoint_attachment_start_i32,
                        attachment_body=self.muscle_waypoint_attachment_body_i32,
                        attachment_local=self.muscle_waypoint_attachment_local,
                        attachment_weight=self.muscle_waypoint_attachment_weight,
                    )
                    self.lbs_waypoint_attachment_fixed2 = True
                except ValueError:
                    lbs_waypoint_attachment_body2 = torch.zeros(
                        (0, 2), dtype=torch.int32
                    )
                    lbs_waypoint_attachment_local_weight0 = torch.zeros(
                        (0, 4), dtype=torch.float32
                    )
                    lbs_waypoint_attachment_local_weight1 = torch.zeros(
                        (0, 4), dtype=torch.float32
                    )
                    self.lbs_waypoint_attachment_fixed2 = False
        else:
            lbs_active_attach_offsets = torch.zeros(1, dtype=torch.int32)
            lbs_active_attach_body = torch.zeros(0, dtype=torch.int32)
            lbs_active_attach_local = torch.zeros((0, 3), dtype=torch.float32)
            lbs_active_attach_coeff_weight = torch.zeros(0, dtype=torch.float32)
            lbs_waypoint_attachment_body2 = torch.zeros((0, 2), dtype=torch.int32)
            lbs_waypoint_attachment_local_weight0 = torch.zeros(
                (0, 4), dtype=torch.float32
            )
            lbs_waypoint_attachment_local_weight1 = torch.zeros(
                (0, 4), dtype=torch.float32
            )
            self.lbs_waypoint_attachment_fixed2 = False
        self.lbs_native_waypoint_attachment_arity = (
            1
            if self.lbs_waypoint_attachment_fixed1
            else (2 if self.lbs_waypoint_attachment_fixed2 else 0)
        )
        self._uses_native_body_wrench_projection = bool(
            native_body_wrench_requested
            and self.lbs_native_waypoint_attachment_arity > 0
        )
        lbs_active_attach_local_coeff = build_lbs_active_local_coeff_vec4(
            lbs_active_attach_local,
            lbs_active_attach_coeff_weight,
        )
        self.register_buffer(
            "lbs_active_attach_offsets",
            lbs_active_attach_offsets.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_active_attach_body",
            lbs_active_attach_body.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_active_attach_local",
            lbs_active_attach_local.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_active_attach_coeff_weight",
            lbs_active_attach_coeff_weight.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_active_attach_local_coeff",
            lbs_active_attach_local_coeff.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_waypoint_attachment_body2",
            lbs_waypoint_attachment_body2.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_waypoint_attachment_local_weight0",
            lbs_waypoint_attachment_local_weight0.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_waypoint_attachment_local_weight1",
            lbs_waypoint_attachment_local_weight1.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.lbs_waypoint_count = int(waypoint_count)
        _record_init_phase("lbs_attachment_metadata")
        if (
            self.uses_lbs_waypoint_attachments
            and self.apply_segment_start_idx.numel() > 0
        ):
            empty_long = torch.zeros(0, dtype=torch.long)
            self.register_buffer(
                "segment_start_root_body_idx", empty_long.clone(), persistent=False
            )
            self.register_buffer(
                "segment_start_root_body_idx_i32",
                torch.zeros(0, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "segment_end_root_body_idx", empty_long.clone(), persistent=False
            )
            self.register_buffer(
                "segment_end_root_body_idx_i32",
                torch.zeros(0, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "segment_start_dof_mask",
                torch.zeros(self.torque_dim, 0, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "segment_end_dof_mask",
                torch.zeros(self.torque_dim, 0, dtype=torch.float32),
                persistent=False,
            )
            if cached_lbs_active_metadata_ready:
                if not isinstance(cached_lbs_metadata, dict):
                    raise ValueError(
                        "Cached LBS metadata payload must be a dictionary."
                    )
                torque_active_entry_offsets = cached_lbs_metadata[
                    "torque_active_entry_offsets"
                ]
                torque_active_torque_idx = cached_lbs_metadata[
                    "torque_active_torque_idx"
                ]
                torque_active_segment_idx = cached_lbs_metadata[
                    "torque_active_segment_idx"
                ]
                torque_active_muscle_idx = cached_lbs_metadata[
                    "torque_active_muscle_idx"
                ]
            else:
                (
                    torque_active_entry_offsets,
                    torque_active_torque_idx,
                    torque_active_segment_idx,
                    torque_active_muscle_idx,
                ) = build_torque_sparse_lbs_segment_entries(
                    torque_dof_subtree_mask=self.dof_subtree_mask[
                        self.root_dof_count :
                    ],
                    waypoint_attachment_start=self.muscle_waypoint_attachment_start,
                    waypoint_attachment_body=self.muscle_waypoint_attachment_body,
                    apply_segment_start_idx=self.apply_segment_start_idx,
                    apply_segment_end_idx=self.apply_segment_end_idx,
                    apply_segment_muscle_idx=self.apply_segment_muscle_idx,
                )
            torque_active_start_coeff = torch.zeros_like(
                torque_active_segment_idx, dtype=torch.float32
            )
            torque_active_end_coeff = torch.zeros_like(
                torque_active_segment_idx, dtype=torch.float32
            )
            if cached_lbs_active_metadata_ready:
                if not isinstance(cached_lbs_metadata, dict):
                    raise ValueError(
                        "Cached LBS metadata payload must be a dictionary."
                    )
                muscle_active_entry_offsets = cached_lbs_metadata[
                    "muscle_active_entry_offsets"
                ]
                muscle_active_segment_idx = cached_lbs_metadata[
                    "muscle_active_segment_idx"
                ]
                muscle_active_torque_idx = cached_lbs_metadata[
                    "muscle_active_torque_idx"
                ]
            else:
                (
                    muscle_active_entry_offsets,
                    muscle_active_segment_idx,
                    muscle_active_torque_idx,
                ) = build_muscle_sparse_lbs_torque_entries(
                    torque_entry_offsets=torque_active_entry_offsets,
                    torque_segment_idx=torque_active_segment_idx,
                    apply_segment_muscle_idx=self.apply_segment_muscle_idx,
                    muscle_count=self.muscle_count,
                )
            muscle_active_start_coeff = torch.zeros_like(
                muscle_active_segment_idx, dtype=torch.float32
            )
            muscle_active_end_coeff = torch.zeros_like(
                muscle_active_segment_idx, dtype=torch.float32
            )
        elif self.apply_segment_body_start_idx.numel() > 0:
            self.register_buffer(
                "segment_start_root_body_idx",
                self.body_rootid.index_select(0, self.apply_segment_body_start_idx),
                persistent=False,
            )
            self.register_buffer(
                "segment_start_root_body_idx_i32",
                self.body_rootid.index_select(0, self.apply_segment_body_start_idx).to(
                    torch.int32
                ),
                persistent=False,
            )
            self.register_buffer(
                "segment_end_root_body_idx",
                self.body_rootid.index_select(0, self.apply_segment_body_end_idx),
                persistent=False,
            )
            self.register_buffer(
                "segment_end_root_body_idx_i32",
                self.body_rootid.index_select(0, self.apply_segment_body_end_idx).to(
                    torch.int32
                ),
                persistent=False,
            )
            self.register_buffer(
                "segment_start_dof_mask",
                self.dof_subtree_mask[
                    self.root_dof_count :, self.apply_segment_body_start_idx
                ],
                persistent=False,
            )
            self.register_buffer(
                "segment_end_dof_mask",
                self.dof_subtree_mask[
                    self.root_dof_count :, self.apply_segment_body_end_idx
                ],
                persistent=False,
            )
            (
                torque_active_entry_offsets,
                torque_active_torque_idx,
                torque_active_segment_idx,
                torque_active_muscle_idx,
                torque_active_start_coeff,
                torque_active_end_coeff,
            ) = self._build_torque_sparse_segment_entries(
                self.segment_start_dof_mask,
                self.segment_end_dof_mask,
                self.apply_segment_muscle_idx,
            )
            (
                muscle_active_entry_offsets,
                muscle_active_segment_idx,
                muscle_active_torque_idx,
                muscle_active_start_coeff,
                muscle_active_end_coeff,
            ) = self._build_muscle_sparse_torque_entries(
                torque_active_entry_offsets,
                torque_active_segment_idx,
                torque_active_start_coeff,
                torque_active_end_coeff,
                self.apply_segment_muscle_idx,
                self.muscle_count,
            )
        else:
            self.register_buffer(
                "segment_start_root_body_idx",
                torch.zeros(0, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "segment_start_root_body_idx_i32",
                torch.zeros(0, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "segment_end_root_body_idx",
                torch.zeros(0, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "segment_end_root_body_idx_i32",
                torch.zeros(0, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "segment_start_dof_mask",
                torch.zeros(self.torque_dim, 0, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "segment_end_dof_mask",
                torch.zeros(self.torque_dim, 0, dtype=torch.float32),
                persistent=False,
            )
            torque_active_entry_offsets = torch.zeros(
                self.torque_dim + 1, dtype=torch.int32
            )
            torque_active_torque_idx = torch.zeros(0, dtype=torch.int32)
            torque_active_segment_idx = torch.zeros(0, dtype=torch.int32)
            torque_active_muscle_idx = torch.zeros(0, dtype=torch.int32)
            torque_active_start_coeff = torch.zeros(0, dtype=torch.float32)
            torque_active_end_coeff = torch.zeros(0, dtype=torch.float32)
            muscle_active_entry_offsets = torch.zeros(
                self.muscle_count + 1, dtype=torch.int32
            )
            muscle_active_segment_idx = torch.zeros(0, dtype=torch.int32)
            muscle_active_torque_idx = torch.zeros(0, dtype=torch.int32)
            muscle_active_start_coeff = torch.zeros(0, dtype=torch.float32)
            muscle_active_end_coeff = torch.zeros(0, dtype=torch.float32)
        self.register_buffer(
            "torque_active_entry_offsets",
            torque_active_entry_offsets.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "torque_active_torque_idx",
            torque_active_torque_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "torque_active_segment_idx",
            torque_active_segment_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "torque_active_muscle_idx",
            torque_active_muscle_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "torque_active_start_coeff",
            torque_active_start_coeff.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "torque_active_end_coeff",
            torque_active_end_coeff.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "muscle_active_entry_offsets",
            muscle_active_entry_offsets.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "muscle_active_segment_idx",
            muscle_active_segment_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "muscle_active_torque_idx",
            muscle_active_torque_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "muscle_active_start_coeff",
            muscle_active_start_coeff.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "muscle_active_end_coeff",
            muscle_active_end_coeff.to(device=self.muscle_params.device),
            persistent=False,
        )
        _record_init_phase("sparse_projection_metadata")
        if (
            self.uses_lbs_waypoint_attachments
            and int(torque_active_segment_idx.numel()) > 0
        ):
            if cached_lbs_active_metadata_ready:
                if not isinstance(cached_lbs_metadata, dict):
                    raise ValueError(
                        "Cached LBS metadata payload must be a dictionary."
                    )
                lbs_forward_segment_idx = cached_lbs_metadata["lbs_forward_segment_idx"]
                lbs_forward_muscle_idx = cached_lbs_metadata["lbs_forward_muscle_idx"]
                lbs_torque_chunk_offsets = cached_lbs_metadata[
                    "lbs_torque_chunk_offsets"
                ]
                lbs_torque_chunk_torque_idx = cached_lbs_metadata[
                    "lbs_torque_chunk_torque_idx"
                ]
                lbs_torque_chunk_entry_start = cached_lbs_metadata[
                    "lbs_torque_chunk_entry_start"
                ]
                lbs_torque_chunk_entry_end = cached_lbs_metadata[
                    "lbs_torque_chunk_entry_end"
                ]
                lbs_bounds_forward_torque_idx = cached_lbs_metadata[
                    "lbs_bounds_forward_torque_idx"
                ]
                lbs_bounds_forward_segment_idx = cached_lbs_metadata[
                    "lbs_bounds_forward_segment_idx"
                ]
                lbs_bounds_forward_muscle_idx = cached_lbs_metadata[
                    "lbs_bounds_forward_muscle_idx"
                ]
                lbs_bounds_group_entry_start = cached_lbs_metadata[
                    "lbs_bounds_group_entry_start"
                ]
                lbs_bounds_group_entry_end = cached_lbs_metadata[
                    "lbs_bounds_group_entry_end"
                ]
                lbs_bounds_chunk_offsets = cached_lbs_metadata[
                    "lbs_bounds_chunk_offsets"
                ]
                lbs_bounds_chunk_torque_idx = cached_lbs_metadata[
                    "lbs_bounds_chunk_torque_idx"
                ]
                lbs_bounds_chunk_group_start = cached_lbs_metadata[
                    "lbs_bounds_chunk_group_start"
                ]
                lbs_bounds_chunk_group_end = cached_lbs_metadata[
                    "lbs_bounds_chunk_group_end"
                ]
            else:
                (
                    _lbs_forward_torque_idx,
                    lbs_forward_segment_idx,
                    lbs_forward_muscle_idx,
                    lbs_torque_chunk_offsets,
                    lbs_torque_chunk_torque_idx,
                    lbs_torque_chunk_entry_start,
                    lbs_torque_chunk_entry_end,
                ) = build_lbs_torque_forward_chunks(
                    torque_active_entry_offsets=torque_active_entry_offsets,
                    torque_active_segment_idx=torque_active_segment_idx,
                    torque_active_muscle_idx=torque_active_muscle_idx,
                    torque_dim=self.torque_dim,
                    chunk_size=self.lbs_forward_chunk_size,
                )
                (
                    lbs_bounds_forward_torque_idx,
                    lbs_bounds_forward_segment_idx,
                    lbs_bounds_forward_muscle_idx,
                    lbs_bounds_group_entry_start,
                    lbs_bounds_group_entry_end,
                    lbs_bounds_chunk_offsets,
                    lbs_bounds_chunk_torque_idx,
                    lbs_bounds_chunk_group_start,
                    lbs_bounds_chunk_group_end,
                ) = build_lbs_torque_bounds_chunks(
                    torque_active_entry_offsets=torque_active_entry_offsets,
                    torque_active_segment_idx=torque_active_segment_idx,
                    torque_active_muscle_idx=torque_active_muscle_idx,
                    torque_dim=self.torque_dim,
                    chunk_size=self.lbs_forward_chunk_size,
                )
        else:
            lbs_forward_segment_idx = torch.zeros(0, dtype=torch.int32)
            lbs_forward_muscle_idx = torch.zeros(0, dtype=torch.int32)
            lbs_torque_chunk_offsets = torch.zeros(
                self.torque_dim + 1, dtype=torch.int32
            )
            lbs_torque_chunk_torque_idx = torch.zeros(0, dtype=torch.int32)
            lbs_torque_chunk_entry_start = torch.zeros(0, dtype=torch.int32)
            lbs_torque_chunk_entry_end = torch.zeros(0, dtype=torch.int32)
            lbs_bounds_forward_torque_idx = torch.zeros(0, dtype=torch.int32)
            lbs_bounds_forward_segment_idx = torch.zeros(0, dtype=torch.int32)
            lbs_bounds_forward_muscle_idx = torch.zeros(0, dtype=torch.int32)
            lbs_bounds_group_entry_start = torch.zeros(0, dtype=torch.int32)
            lbs_bounds_group_entry_end = torch.zeros(0, dtype=torch.int32)
            lbs_bounds_chunk_offsets = torch.zeros(
                self.torque_dim + 1, dtype=torch.int32
            )
            lbs_bounds_chunk_torque_idx = torch.zeros(0, dtype=torch.int32)
            lbs_bounds_chunk_group_start = torch.zeros(0, dtype=torch.int32)
            lbs_bounds_chunk_group_end = torch.zeros(0, dtype=torch.int32)
        if cached_lbs_active_metadata_ready:
            if not isinstance(cached_lbs_metadata, dict):
                raise ValueError("Cached LBS metadata payload must be a dictionary.")
            lbs_segment_waypoints = cached_lbs_metadata["lbs_segment_waypoints"]
        else:
            (
                _lbs_torque_chunk_info,
                _lbs_forward_entry_info,
                lbs_segment_waypoints,
            ) = build_lbs_forward_packed_metadata(
                chunk_torque_idx=lbs_torque_chunk_torque_idx,
                chunk_entry_start=lbs_torque_chunk_entry_start,
                chunk_entry_end=lbs_torque_chunk_entry_end,
                forward_segment_idx=lbs_forward_segment_idx,
                forward_muscle_idx=lbs_forward_muscle_idx,
                apply_segment_start_idx=self.apply_segment_start_idx_i32,
                apply_segment_end_idx=self.apply_segment_end_idx_i32,
            )
        _record_init_phase("lbs_torque_chunk_metadata")
        if self._uses_attachment_projection:
            if (
                isinstance(cached_lbs_metadata, dict)
                and all(
                    torch.is_tensor(cached_lbs_metadata.get(key))
                    for key in _LAYER_LBS_STATIC_METADATA_TENSOR_KEYS
                )
                and isinstance(
                    cached_lbs_metadata.get("lbs_body_segment_event_start_contiguous"),
                    bool,
                )
            ):
                lbs_waypoint_segment_idx2 = cached_lbs_metadata[
                    "lbs_waypoint_segment_idx2"
                ]
                lbs_waypoint_segment_sign2 = cached_lbs_metadata[
                    "lbs_waypoint_segment_sign2"
                ]
                lbs_active_attach_waypoint_idx = cached_lbs_metadata[
                    "lbs_active_attach_waypoint_idx"
                ]
                lbs_active_attach_chunk_offsets = cached_lbs_metadata[
                    "lbs_active_attach_chunk_offsets"
                ]
                lbs_active_attach_chunk_info = cached_lbs_metadata[
                    "lbs_active_attach_chunk_info"
                ]
                lbs_body_event_chunk_offsets = cached_lbs_metadata[
                    "lbs_body_event_chunk_offsets"
                ]
                lbs_body_event_chunk_info = cached_lbs_metadata[
                    "lbs_body_event_chunk_info"
                ]
                lbs_body_event_packed_indices = cached_lbs_metadata[
                    "lbs_body_event_packed_indices"
                ]
                lbs_body_event_local_force_coeff = cached_lbs_metadata[
                    "lbs_body_event_local_force_coeff"
                ]
                lbs_body_segment_event_start_muscle = cached_lbs_metadata[
                    "lbs_body_segment_event_start_muscle"
                ]
                lbs_native_body_tile_info = cached_lbs_metadata[
                    "lbs_native_body_tile_info"
                ]
                lbs_native_body_tile_extra_info = cached_lbs_metadata[
                    "lbs_native_body_tile_extra_info"
                ]
                lbs_native_body_tile_body_ids = cached_lbs_metadata[
                    "lbs_native_body_tile_body_ids"
                ]
                lbs_native_body_tile_body_output_idx = cached_lbs_metadata[
                    "lbs_native_body_tile_body_output_idx"
                ]
                lbs_native_body_tile_event_start_slot_muscle = cached_lbs_metadata[
                    "lbs_native_body_tile_event_start_slot_muscle"
                ]
                lbs_native_body_tile_event_local_force_coeff = cached_lbs_metadata[
                    "lbs_native_body_tile_event_local_force_coeff"
                ]
                lbs_native_body_tile_body_slot_offsets = cached_lbs_metadata[
                    "lbs_native_body_tile_body_slot_offsets"
                ]
                self.lbs_body_segment_event_start_contiguous = cached_lbs_metadata[
                    "lbs_body_segment_event_start_contiguous"
                ]
            else:
                (
                    lbs_waypoint_segment_idx2,
                    lbs_waypoint_segment_sign2,
                ) = build_lbs_waypoint_force_metadata(
                    apply_segment_start_idx=self.apply_segment_start_idx_i32,
                    apply_segment_end_idx=self.apply_segment_end_idx_i32,
                    waypoint_count=self.lbs_waypoint_count,
                )
                (
                    lbs_active_attach_waypoint_idx,
                    lbs_active_attach_chunk_offsets,
                    lbs_active_attach_chunk_info,
                ) = build_lbs_active_attachment_projection_metadata(
                    active_attach_offsets=lbs_active_attach_offsets,
                    torque_dim=self.torque_dim,
                    waypoint_count=self.lbs_waypoint_count,
                    chunk_size=self.lbs_forward_chunk_size,
                )
                (
                    lbs_body_event_chunk_offsets,
                    lbs_body_event_chunk_info,
                    lbs_body_event_packed_indices,
                    lbs_body_event_local_force_coeff,
                ) = build_lbs_body_wrench_segment_event_metadata(
                    apply_segment_start_idx=self.apply_segment_start_idx_i32,
                    apply_segment_end_idx=self.apply_segment_end_idx_i32,
                    apply_segment_muscle_idx=self.apply_segment_muscle_idx_i32,
                    waypoint_attachment_start=self.muscle_waypoint_attachment_start_i32,
                    waypoint_attachment_body=self.muscle_waypoint_attachment_body_i32,
                    waypoint_attachment_local=self.muscle_waypoint_attachment_local,
                    waypoint_attachment_weight=self.muscle_waypoint_attachment_weight,
                    body_count=self.body_count,
                    chunk_size=self.lbs_body_event_chunk_size,
                )
                self.lbs_body_segment_event_start_contiguous = bool(
                    torch.all(
                        self.apply_segment_end_idx_i32
                        - self.apply_segment_start_idx_i32
                        == 1
                    ).item()
                )
                if self.lbs_body_segment_event_start_contiguous:
                    lbs_body_segment_event_start_muscle = torch.stack(
                        [
                            lbs_body_event_packed_indices[:, 0].to(torch.int32),
                            lbs_body_event_packed_indices[:, 2].to(torch.int32),
                        ],
                        dim=1,
                    ).contiguous()
                else:
                    lbs_body_segment_event_start_muscle = torch.zeros(
                        (0, 2), dtype=torch.int32
                    )
                if self.lbs_body_wrench_forward_mode == "native_tile":
                    if self.lbs_native_waypoint_attachment_arity == 0:
                        raise ValueError(
                            "lbs_body_wrench_forward_mode='native_tile' requires "
                            "one or two attachments per waypoint."
                        )
                    if not self.lbs_body_segment_event_start_contiguous:
                        raise ValueError(
                            "lbs_body_wrench_forward_mode='native_tile' requires "
                            "segment end waypoint to be start+1."
                        )
                    (
                        lbs_native_body_tile_info,
                        lbs_native_body_tile_extra_info,
                        lbs_native_body_tile_body_ids,
                        lbs_native_body_tile_body_output_idx,
                        lbs_native_body_tile_event_start_slot_muscle,
                        lbs_native_body_tile_event_local_force_coeff,
                        lbs_native_body_tile_body_slot_offsets,
                    ) = build_lbs_native_body_wrench_tile_metadata(
                        muscle_segment_offsets=self.all_muscle_segment_offsets,
                        body_event_chunk_info=lbs_body_event_chunk_info,
                        body_event_start_muscle=lbs_body_segment_event_start_muscle,
                        body_event_local_force_coeff=lbs_body_event_local_force_coeff,
                        muscle_count=self.muscle_count,
                        body_count=self.body_count,
                        muscle_cap=self.lbs_body_wrench_native_tile_muscle_cap,
                        segment_cap=self.lbs_body_wrench_native_tile_segment_cap,
                        body_event_cap=self.lbs_body_wrench_native_tile_event_cap,
                        body_cap=self.lbs_body_wrench_native_tile_body_cap,
                    )
                else:
                    lbs_native_body_tile_info = torch.zeros((0, 4), dtype=torch.int32)
                    lbs_native_body_tile_extra_info = torch.zeros(
                        (0, 4), dtype=torch.int32
                    )
                    lbs_native_body_tile_body_ids = torch.zeros(0, dtype=torch.int32)
                    lbs_native_body_tile_body_output_idx = torch.zeros(
                        0, dtype=torch.int32
                    )
                    lbs_native_body_tile_event_start_slot_muscle = torch.zeros(
                        (0,), dtype=torch.int32
                    )
                    lbs_native_body_tile_event_local_force_coeff = torch.zeros(
                        (0, 4), dtype=torch.float32
                    )
                    lbs_native_body_tile_body_slot_offsets = torch.zeros(
                        self.body_count + 1, dtype=torch.int32
                    )
                if lbs_metadata_cache_key is not None:
                    metadata_to_cache = {
                        "muscle_force_length_coeff": self.muscle_force_length_coeff,
                        "all_muscle_segment_offsets": all_muscle_segment_offsets,
                        "apply_muscle_segment_offsets": apply_muscle_segment_offsets,
                        "lbs_active_attach_offsets": lbs_active_attach_offsets,
                        "lbs_active_attach_body": lbs_active_attach_body,
                        "lbs_active_attach_local": lbs_active_attach_local,
                        "lbs_active_attach_coeff_weight": lbs_active_attach_coeff_weight,
                        "lbs_waypoint_attachment_body2": lbs_waypoint_attachment_body2,
                        "lbs_waypoint_attachment_local_weight0": lbs_waypoint_attachment_local_weight0,
                        "lbs_waypoint_attachment_local_weight1": lbs_waypoint_attachment_local_weight1,
                        "torque_active_entry_offsets": torque_active_entry_offsets,
                        "torque_active_torque_idx": torque_active_torque_idx,
                        "torque_active_segment_idx": torque_active_segment_idx,
                        "torque_active_muscle_idx": torque_active_muscle_idx,
                        "muscle_active_entry_offsets": muscle_active_entry_offsets,
                        "muscle_active_segment_idx": muscle_active_segment_idx,
                        "muscle_active_torque_idx": muscle_active_torque_idx,
                        "lbs_forward_segment_idx": lbs_forward_segment_idx,
                        "lbs_forward_muscle_idx": lbs_forward_muscle_idx,
                        "lbs_torque_chunk_offsets": lbs_torque_chunk_offsets,
                        "lbs_torque_chunk_torque_idx": lbs_torque_chunk_torque_idx,
                        "lbs_torque_chunk_entry_start": lbs_torque_chunk_entry_start,
                        "lbs_torque_chunk_entry_end": lbs_torque_chunk_entry_end,
                        "lbs_bounds_forward_torque_idx": lbs_bounds_forward_torque_idx,
                        "lbs_bounds_forward_segment_idx": lbs_bounds_forward_segment_idx,
                        "lbs_bounds_forward_muscle_idx": lbs_bounds_forward_muscle_idx,
                        "lbs_bounds_group_entry_start": lbs_bounds_group_entry_start,
                        "lbs_bounds_group_entry_end": lbs_bounds_group_entry_end,
                        "lbs_bounds_chunk_offsets": lbs_bounds_chunk_offsets,
                        "lbs_bounds_chunk_torque_idx": lbs_bounds_chunk_torque_idx,
                        "lbs_bounds_chunk_group_start": lbs_bounds_chunk_group_start,
                        "lbs_bounds_chunk_group_end": lbs_bounds_chunk_group_end,
                        "lbs_segment_waypoints": lbs_segment_waypoints,
                        "lbs_waypoint_segment_idx2": lbs_waypoint_segment_idx2,
                        "lbs_waypoint_segment_sign2": lbs_waypoint_segment_sign2,
                        "lbs_active_attach_waypoint_idx": lbs_active_attach_waypoint_idx,
                        "lbs_active_attach_chunk_offsets": lbs_active_attach_chunk_offsets,
                        "lbs_active_attach_chunk_info": lbs_active_attach_chunk_info,
                        "lbs_body_event_chunk_offsets": lbs_body_event_chunk_offsets,
                        "lbs_body_event_chunk_info": lbs_body_event_chunk_info,
                        "lbs_body_event_packed_indices": lbs_body_event_packed_indices,
                        "lbs_body_event_local_force_coeff": lbs_body_event_local_force_coeff,
                        "lbs_body_segment_event_start_muscle": lbs_body_segment_event_start_muscle,
                        "lbs_native_body_tile_info": lbs_native_body_tile_info,
                        "lbs_native_body_tile_extra_info": lbs_native_body_tile_extra_info,
                        "lbs_native_body_tile_body_ids": lbs_native_body_tile_body_ids,
                        "lbs_native_body_tile_body_output_idx": lbs_native_body_tile_body_output_idx,
                        "lbs_native_body_tile_event_start_slot_muscle": lbs_native_body_tile_event_start_slot_muscle,
                        "lbs_native_body_tile_event_local_force_coeff": lbs_native_body_tile_event_local_force_coeff,
                        "lbs_native_body_tile_body_slot_offsets": lbs_native_body_tile_body_slot_offsets,
                    }
                    _store_layer_lbs_metadata_cache(
                        lbs_metadata_cache_key,
                        {
                            "lbs_waypoint_attachment_fixed2": (
                                self.lbs_waypoint_attachment_fixed2
                            ),
                            "lbs_body_segment_event_start_contiguous": (
                                self.lbs_body_segment_event_start_contiguous
                            ),
                            **{
                                key: value.detach().cpu()
                                for key, value in metadata_to_cache.items()
                            },
                        },
                    )
        else:
            lbs_waypoint_segment_idx2 = torch.zeros((0, 2), dtype=torch.int32)
            lbs_waypoint_segment_sign2 = torch.zeros((0, 2), dtype=torch.int32)
            lbs_active_attach_waypoint_idx = torch.zeros(0, dtype=torch.int32)
            lbs_active_attach_chunk_offsets = torch.zeros(
                self.torque_dim + 1, dtype=torch.int32
            )
            lbs_active_attach_chunk_info = torch.zeros((0, 3), dtype=torch.int32)
            lbs_body_event_chunk_offsets = torch.zeros(
                self.body_count + 1, dtype=torch.int32
            )
            lbs_body_event_chunk_info = torch.zeros((0, 3), dtype=torch.int32)
            lbs_body_event_packed_indices = torch.zeros((0, 4), dtype=torch.int32)
            lbs_body_event_local_force_coeff = torch.zeros((0, 4), dtype=torch.float32)
            lbs_body_segment_event_start_muscle = torch.zeros((0, 2), dtype=torch.int32)
            self.lbs_body_segment_event_start_contiguous = False
            lbs_native_body_tile_info = torch.zeros((0, 4), dtype=torch.int32)
            lbs_native_body_tile_extra_info = torch.zeros((0, 4), dtype=torch.int32)
            lbs_native_body_tile_body_ids = torch.zeros(0, dtype=torch.int32)
            lbs_native_body_tile_body_output_idx = torch.zeros(0, dtype=torch.int32)
            lbs_native_body_tile_event_start_slot_muscle = torch.zeros(
                (0,), dtype=torch.int32
            )
            lbs_native_body_tile_event_local_force_coeff = torch.zeros(
                (0, 4), dtype=torch.float32
            )
            lbs_native_body_tile_body_slot_offsets = torch.zeros(
                self.body_count + 1, dtype=torch.int32
            )
        if (
            self.uses_lbs_waypoint_attachments
            and lbs_metadata_cache_key is not None
            and (
                not cached_lbs_base_metadata_ready
                or not cached_lbs_active_metadata_ready
            )
        ):
            metadata_to_cache = {
                "muscle_force_length_coeff": self.muscle_force_length_coeff,
                "all_muscle_segment_offsets": all_muscle_segment_offsets,
                "apply_muscle_segment_offsets": apply_muscle_segment_offsets,
                "lbs_active_attach_offsets": lbs_active_attach_offsets,
                "lbs_active_attach_body": lbs_active_attach_body,
                "lbs_active_attach_local": lbs_active_attach_local,
                "lbs_active_attach_coeff_weight": lbs_active_attach_coeff_weight,
                "lbs_waypoint_attachment_body2": lbs_waypoint_attachment_body2,
                "lbs_waypoint_attachment_local_weight0": (
                    lbs_waypoint_attachment_local_weight0
                ),
                "lbs_waypoint_attachment_local_weight1": (
                    lbs_waypoint_attachment_local_weight1
                ),
                "torque_active_entry_offsets": torque_active_entry_offsets,
                "torque_active_torque_idx": torque_active_torque_idx,
                "torque_active_segment_idx": torque_active_segment_idx,
                "torque_active_muscle_idx": torque_active_muscle_idx,
                "muscle_active_entry_offsets": muscle_active_entry_offsets,
                "muscle_active_segment_idx": muscle_active_segment_idx,
                "muscle_active_torque_idx": muscle_active_torque_idx,
                "lbs_forward_segment_idx": lbs_forward_segment_idx,
                "lbs_forward_muscle_idx": lbs_forward_muscle_idx,
                "lbs_torque_chunk_offsets": lbs_torque_chunk_offsets,
                "lbs_torque_chunk_torque_idx": lbs_torque_chunk_torque_idx,
                "lbs_torque_chunk_entry_start": lbs_torque_chunk_entry_start,
                "lbs_torque_chunk_entry_end": lbs_torque_chunk_entry_end,
                "lbs_bounds_forward_torque_idx": lbs_bounds_forward_torque_idx,
                "lbs_bounds_forward_segment_idx": lbs_bounds_forward_segment_idx,
                "lbs_bounds_forward_muscle_idx": lbs_bounds_forward_muscle_idx,
                "lbs_bounds_group_entry_start": lbs_bounds_group_entry_start,
                "lbs_bounds_group_entry_end": lbs_bounds_group_entry_end,
                "lbs_bounds_chunk_offsets": lbs_bounds_chunk_offsets,
                "lbs_bounds_chunk_torque_idx": lbs_bounds_chunk_torque_idx,
                "lbs_bounds_chunk_group_start": lbs_bounds_chunk_group_start,
                "lbs_bounds_chunk_group_end": lbs_bounds_chunk_group_end,
                "lbs_segment_waypoints": lbs_segment_waypoints,
                "lbs_waypoint_segment_idx2": lbs_waypoint_segment_idx2,
                "lbs_waypoint_segment_sign2": lbs_waypoint_segment_sign2,
                "lbs_active_attach_waypoint_idx": lbs_active_attach_waypoint_idx,
                "lbs_active_attach_chunk_offsets": lbs_active_attach_chunk_offsets,
                "lbs_active_attach_chunk_info": lbs_active_attach_chunk_info,
                "lbs_body_event_chunk_offsets": lbs_body_event_chunk_offsets,
                "lbs_body_event_chunk_info": lbs_body_event_chunk_info,
                "lbs_body_event_packed_indices": lbs_body_event_packed_indices,
                "lbs_body_event_local_force_coeff": lbs_body_event_local_force_coeff,
                "lbs_body_segment_event_start_muscle": (
                    lbs_body_segment_event_start_muscle
                ),
                "lbs_native_body_tile_info": lbs_native_body_tile_info,
                "lbs_native_body_tile_extra_info": lbs_native_body_tile_extra_info,
                "lbs_native_body_tile_body_ids": lbs_native_body_tile_body_ids,
                "lbs_native_body_tile_body_output_idx": (
                    lbs_native_body_tile_body_output_idx
                ),
                "lbs_native_body_tile_event_start_slot_muscle": (
                    lbs_native_body_tile_event_start_slot_muscle
                ),
                "lbs_native_body_tile_event_local_force_coeff": (
                    lbs_native_body_tile_event_local_force_coeff
                ),
                "lbs_native_body_tile_body_slot_offsets": (
                    lbs_native_body_tile_body_slot_offsets
                ),
            }
            _store_layer_lbs_metadata_cache(
                lbs_metadata_cache_key,
                {
                    "lbs_waypoint_attachment_fixed2": (
                        self.lbs_waypoint_attachment_fixed2
                    ),
                    "lbs_body_segment_event_start_contiguous": (
                        self.lbs_body_segment_event_start_contiguous
                    ),
                    **{
                        key: value.detach().cpu()
                        for key, value in metadata_to_cache.items()
                    },
                },
            )
        if (
            self._uses_native_body_wrench_projection
            and self.lbs_body_segment_event_start_contiguous
        ):
            native_sorted_metadata = build_lbs_native_body_wrench_sorted_metadata(
                muscle_segment_offsets=self.all_muscle_segment_offsets,
                muscle_force_length_coeff=self.muscle_force_length_coeff,
                waypoint_attachment_body2=self.lbs_waypoint_attachment_body2,
                waypoint_attachment_local_weight0=(
                    self.lbs_waypoint_attachment_local_weight0
                ),
                waypoint_attachment_local_weight1=(
                    self.lbs_waypoint_attachment_local_weight1
                ),
                body_event_chunk_info=lbs_body_event_chunk_info,
                body_event_start_muscle=lbs_body_segment_event_start_muscle,
                body_event_local_force_coeff=lbs_body_event_local_force_coeff,
                muscle_count=self.muscle_count,
                body_count=self.body_count,
                muscle_cap=self.lbs_body_wrench_native_tile_muscle_cap,
                segment_cap=self.lbs_body_wrench_native_tile_segment_cap,
                body_event_cap=self.lbs_body_wrench_native_tile_event_cap,
                body_cap=self.lbs_body_wrench_native_tile_body_cap,
            )
            lbs_native_muscle_order = native_sorted_metadata.muscle_order
            lbs_native_muscle_segment_offsets = (
                native_sorted_metadata.muscle_segment_offsets
            )
            lbs_native_muscle_force_length_coeff = (
                native_sorted_metadata.muscle_force_length_coeff
            )
            lbs_native_waypoint_attachment_body2 = (
                native_sorted_metadata.waypoint_attachment_body2
            )
            lbs_native_waypoint_attachment_local_weight0 = (
                native_sorted_metadata.waypoint_attachment_local_weight0
            )
            lbs_native_waypoint_attachment_local_weight1 = (
                native_sorted_metadata.waypoint_attachment_local_weight1
            )
            lbs_native_body_tile_info = native_sorted_metadata.tile_info
            lbs_native_body_tile_extra_info = native_sorted_metadata.tile_extra_info
            lbs_native_body_tile_body_ids = native_sorted_metadata.tile_body_ids
            lbs_native_body_tile_body_output_idx = (
                native_sorted_metadata.tile_body_output_idx
            )
            lbs_native_body_tile_event_start_slot_muscle = (
                native_sorted_metadata.tile_event_start_slot_muscle
            )
            lbs_native_body_tile_event_local_force_coeff = (
                native_sorted_metadata.tile_event_local_force_coeff
            )
            lbs_native_body_tile_body_slot_offsets = (
                native_sorted_metadata.tile_body_slot_offsets
            )
        else:
            lbs_native_muscle_order = torch.arange(
                self.muscle_count,
                dtype=torch.int64,
                device=self.muscle_params.device,
            )
            lbs_native_muscle_segment_offsets = self.all_muscle_segment_offsets
            lbs_native_muscle_force_length_coeff = self.muscle_force_length_coeff
            lbs_native_waypoint_attachment_body2 = self.lbs_waypoint_attachment_body2
            lbs_native_waypoint_attachment_local_weight0 = (
                self.lbs_waypoint_attachment_local_weight0
            )
            lbs_native_waypoint_attachment_local_weight1 = (
                self.lbs_waypoint_attachment_local_weight1
            )
        if (
            self._uses_native_body_wrench_projection
            and int(lbs_native_body_tile_info.shape[0]) > 0
        ):
            native_tile_info_cpu = (
                lbs_native_body_tile_info.detach().cpu().to(torch.int64)
            )
            native_tile_extra_cpu = (
                lbs_native_body_tile_extra_info.detach().cpu().to(torch.int64)
            )
            self.lbs_native_body_wrench_compiled_muscle_cap = int(
                (native_tile_info_cpu[:, 1] - native_tile_info_cpu[:, 0]).max()
            )
            self.lbs_native_body_wrench_compiled_waypoint_cap = int(
                native_tile_extra_cpu[:, 1].max()
            )
            self.lbs_native_body_wrench_compiled_body_cap = int(
                native_tile_extra_cpu[:, 2].max()
            )
            native_kernel_args = (
                self.lbs_native_body_wrench_compiled_muscle_cap,
                self.lbs_native_body_wrench_compiled_waypoint_cap,
                self.lbs_native_body_wrench_compiled_body_cap,
                self.lbs_forward_tile_block_dim,
            )
            full_force_factory = (
                make_accumulate_lbs_body_wrench_native_tile_full_force_fixed1_kernel
                if self.lbs_native_waypoint_attachment_arity == 1
                else make_accumulate_lbs_body_wrench_native_tile_full_force_fixed2_kernel
            )
            cached_force_factory = (
                make_accumulate_lbs_body_wrench_native_tile_cached_force_fixed1_kernel
                if self.lbs_native_waypoint_attachment_arity == 1
                else make_accumulate_lbs_body_wrench_native_tile_cached_force_fixed2_kernel
            )
            self._lbs_native_body_wrench_full_force_kernels = (
                full_force_factory(
                    False,
                    self.muscle_passive_force_model_id,
                    self.muscle_passive_force_param0,
                    self.muscle_passive_force_param1,
                    *native_kernel_args,
                    wrapped=bool(self._waypoint_wrapping.count),
                    active_force_model=tuple(
                        float(self._muscle_active_force_model_params_wp[i])
                        for i in range(3)
                    ),
                ),
                full_force_factory(
                    True,
                    self.muscle_passive_force_model_id,
                    self.muscle_passive_force_param0,
                    self.muscle_passive_force_param1,
                    *native_kernel_args,
                    wrapped=bool(self._waypoint_wrapping.count),
                    active_force_model=tuple(
                        float(self._muscle_active_force_model_params_wp[i])
                        for i in range(3)
                    ),
                ),
            )
            if self.lbs_native_waypoint_attachment_arity == 1:
                # Built on request by the environment force-velocity path.
                self._lbs_native_full_force_velocity_factory_args = (
                    self.muscle_passive_force_model_id,
                    self.muscle_passive_force_param0,
                    self.muscle_passive_force_param1,
                    *native_kernel_args,
                )
            self._lbs_native_body_wrench_cached_force_kernel = cached_force_factory(
                *native_kernel_args, wrapped=bool(self._waypoint_wrapping.count)
            )
            self._lbs_native_body_wrench_direct_force_kernel = cached_force_factory(
                *native_kernel_args,
                direct_force=True,
                wrapped=bool(self._waypoint_wrapping.count),
            )
        point_source = [0]
        if self._waypoint_wrapping.count:
            offsets = self.all_muscle_segment_offsets.detach().cpu().tolist()
            point_source = [
                point
                for muscle in lbs_native_muscle_order.detach().cpu().tolist()
                for point in range(
                    offsets[muscle] + muscle, offsets[muscle + 1] + muscle + 1
                )
            ]
        self.register_buffer(
            "lbs_native_point_source",
            torch.tensor(
                point_source, dtype=torch.int32, device=self.muscle_params.device
            ),
            persistent=False,
        )
        _record_init_phase("lbs_static_metadata")
        lbs_active_attach_idx_body = build_lbs_active_idx_body_vec2(
            lbs_active_attach_waypoint_idx,
            lbs_active_attach_body,
        )
        self.register_buffer(
            "lbs_forward_segment_idx",
            lbs_forward_segment_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_forward_muscle_idx",
            lbs_forward_muscle_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_torque_chunk_offsets",
            lbs_torque_chunk_offsets.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_torque_chunk_torque_idx",
            lbs_torque_chunk_torque_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_torque_chunk_entry_start",
            lbs_torque_chunk_entry_start.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_torque_chunk_entry_end",
            lbs_torque_chunk_entry_end.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_segment_waypoints",
            lbs_segment_waypoints.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_waypoint_segment_idx2",
            lbs_waypoint_segment_idx2.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_waypoint_segment_sign2",
            lbs_waypoint_segment_sign2.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_active_attach_idx_body",
            lbs_active_attach_idx_body.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_active_attach_chunk_offsets",
            lbs_active_attach_chunk_offsets.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_active_attach_chunk_info",
            lbs_active_attach_chunk_info.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_body_event_chunk_offsets",
            lbs_body_event_chunk_offsets.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_body_event_chunk_info",
            lbs_body_event_chunk_info.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_body_event_packed_indices",
            lbs_body_event_packed_indices.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_body_segment_event_start_muscle",
            lbs_body_segment_event_start_muscle.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_body_event_local_force_coeff",
            lbs_body_event_local_force_coeff.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_muscle_order",
            lbs_native_muscle_order.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_muscle_order_i32",
            lbs_native_muscle_order.to(
                device=self.muscle_params.device, dtype=torch.int32
            ),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_muscle_segment_offsets",
            lbs_native_muscle_segment_offsets.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_muscle_force_length_coeff",
            lbs_native_muscle_force_length_coeff.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_waypoint_attachment_body2",
            lbs_native_waypoint_attachment_body2.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_waypoint_attachment_local_weight0",
            lbs_native_waypoint_attachment_local_weight0.to(
                device=self.muscle_params.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_waypoint_attachment_local_weight1",
            lbs_native_waypoint_attachment_local_weight1.to(
                device=self.muscle_params.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_body_tile_info",
            lbs_native_body_tile_info.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_body_tile_extra_info",
            lbs_native_body_tile_extra_info.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_body_tile_body_ids",
            lbs_native_body_tile_body_ids.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_body_tile_body_output_idx",
            lbs_native_body_tile_body_output_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_body_tile_event_start_slot_muscle",
            lbs_native_body_tile_event_start_slot_muscle.to(
                device=self.muscle_params.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_body_tile_event_local_force_coeff",
            lbs_native_body_tile_event_local_force_coeff.to(
                device=self.muscle_params.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "lbs_native_body_tile_body_slot_offsets",
            lbs_native_body_tile_body_slot_offsets.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.lbs_torque_chunk_count = int(lbs_torque_chunk_torque_idx.numel())
        self.lbs_active_attach_chunk_count = int(lbs_active_attach_chunk_info.shape[0])
        self.lbs_body_event_chunk_count = int(lbs_body_event_chunk_info.shape[0])
        self.lbs_native_body_tile_count = int(lbs_native_body_tile_info.shape[0])
        self.lbs_native_body_tile_body_slot_count = int(
            lbs_native_body_tile_body_ids.numel()
        )
        self.native_lbs_body_wrench_direct_backward = bool(
            self._uses_native_body_wrench_projection
            and not self._waypoint_wrapping.count
            and self.lbs_body_segment_event_start_contiguous
            and self._sparse_body_wrench_direct_backward_requested
        )
        # Build the backward kernel with the forward ones. Creating it on the
        # first backward changes the kernel module's hash after the forward
        # tile launch loaded it, so the next forward reloads the module; inside
        # a CUDA graph capture that reload's unload is illegal.
        self._native_lbs_direct_backward_kernel = (
            self._build_native_lbs_direct_backward_kernel()
            if self.native_lbs_body_wrench_direct_backward
            else None
        )
        self.register_buffer(
            "lbs_bounds_forward_torque_idx",
            lbs_bounds_forward_torque_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_bounds_forward_segment_idx",
            lbs_bounds_forward_segment_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_bounds_forward_muscle_idx",
            lbs_bounds_forward_muscle_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_bounds_group_entry_start",
            lbs_bounds_group_entry_start.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_bounds_group_entry_end",
            lbs_bounds_group_entry_end.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_bounds_chunk_offsets",
            lbs_bounds_chunk_offsets.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_bounds_chunk_torque_idx",
            lbs_bounds_chunk_torque_idx.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_bounds_chunk_group_start",
            lbs_bounds_chunk_group_start.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.register_buffer(
            "lbs_bounds_chunk_group_end",
            lbs_bounds_chunk_group_end.to(device=self.muscle_params.device),
            persistent=False,
        )
        self.lbs_bounds_chunk_count = int(lbs_bounds_chunk_torque_idx.numel())
        _record_init_phase("lbs_static_gpu_buffers")
        active_torque_indices = output.active_torque_indices
        if active_torque_indices is not None:
            self.register_buffer(
                "active_torque_indices",
                active_torque_indices.detach().clone().long(),
                persistent=False,
            )
            self.register_buffer(
                "active_torque_indices_i32",
                active_torque_indices.detach().clone().to(torch.int32),
                persistent=False,
            )
            self.active_output_dim = int(active_torque_indices.numel())
        else:
            self.active_torque_indices = None
            self.active_torque_indices_i32 = None
            self.active_output_dim = self.torque_dim
        _record_init_phase("active_output_metadata")
        self._body_rootid_py = _LazyIntList(self.body_rootid)
        self._torque_active_entry_offsets_py = _LazyIntList(
            self.torque_active_entry_offsets
        )
        self._torque_active_segment_idx_py = _LazyIntList(
            self.torque_active_segment_idx
        )
        self._torque_active_muscle_idx_py = _LazyIntList(self.torque_active_muscle_idx)
        self._waypoint_attachment_start_py = _LazyIntList(
            self.muscle_waypoint_attachment_start
        )
        self._waypoint_attachment_body_py = _LazyIntList(
            self.muscle_waypoint_attachment_body
        )
        self._warp_geometry_workspace: dict[
            tuple[str, int, str], dict[str, torch.Tensor]
        ] = {}
        self._warp_torque_jacobian_workspace: dict[
            tuple[str, int], dict[str, torch.Tensor]
        ] = {}
        self._warp_torque_bounds_workspace: dict[
            tuple[str, int], dict[str, torch.Tensor]
        ] = {}
        self._wrench_projection_topologies = {}
        if self._uses_attachment_projection:
            projection_device = str(self.body_parentid.device)
            self._wrench_projection_topologies[projection_device] = (
                build_wrench_projection_topology(
                    self.body_parentid,
                    self.dof_bodyid,
                    self.body_count,
                    projection_device,
                )
            )
        _record_init_phase("final_setup")
        for name, seconds in init_profile_records:
            _LOG.info(
                "MUSCLE_DYNAMICS_LAYER_INIT_PROFILE phase=%s seconds=%.6f",
                name,
                seconds,
            )
        _LOG.info(
            "MUSCLE_DYNAMICS_LAYER_INIT_PROFILE total_seconds=%.6f phases=%d",
            perf_counter() - init_profile_start,
            len(init_profile_records),
        )

    def _build_native_lbs_direct_backward_kernel(self) -> Any:
        fixed1 = self.lbs_native_waypoint_attachment_arity == 1
        segment_plus_muscle = bool(self.lbs_segment_start_is_segment_plus_muscle)
        if self.lbs_single_root_body_idx >= 0:
            factory = (
                make_accumulate_lbs_activation_grad_fixed1_direct_start_from_body_adjoint_kernel
                if fixed1
                else make_accumulate_lbs_activation_grad_fixed2_direct_start_from_body_adjoint_kernel
            )
            return factory(int(self.lbs_single_root_body_idx), segment_plus_muscle)
        factory = (
            make_accumulate_lbs_activation_grad_fixed1_direct_start_from_body_adjoint_multi_root_kernel
            if fixed1
            else make_accumulate_lbs_activation_grad_fixed2_direct_start_from_body_adjoint_multi_root_kernel
        )
        return factory(segment_plus_muscle)

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._waypoint_wrap_states.clear()
        cache = self.__dict__.get("_wrench_projection_topologies")
        if cache is not None:
            cache.clear()
            if self._uses_attachment_projection:
                device = str(self.body_parentid.device)
                cache[device] = build_wrench_projection_topology(
                    self.body_parentid,
                    self.dof_bodyid,
                    self.body_count,
                    device,
                )
        return result

    @staticmethod
    def _build_dof_subtree_mask(
        body_parentid: torch.Tensor,
        dof_bodyid: torch.Tensor,
        body_count: int,
        full_torque_dim: int,
    ) -> torch.Tensor:
        return build_dof_subtree_mask(
            body_parentid=body_parentid,
            dof_bodyid=dof_bodyid,
            body_count=body_count,
            full_torque_dim=full_torque_dim,
        )

    @staticmethod
    def _build_lbs_active_attachment_lists(
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
        return build_lbs_active_attachment_lists(
            dof_subtree_mask=dof_subtree_mask,
            attachment_start=attachment_start,
            attachment_body=attachment_body,
            attachment_local=attachment_local,
            attachment_weight=attachment_weight,
            root_dof_count=root_dof_count,
            torque_dim=torque_dim,
            waypoint_count=waypoint_count,
        )

    @staticmethod
    def _activation_interval_bounds(
        muscle_dynamics_contribution: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive = torch.clamp(muscle_dynamics_contribution, min=0.0)
        negative = torch.clamp(muscle_dynamics_contribution, max=0.0)
        lower = positive.sum(dim=-1) * 0.0 + negative.sum(dim=-1) * 1.0
        upper = positive.sum(dim=-1) * 1.0 + negative.sum(dim=-1) * 0.0
        return lower, upper

    def _project_body_wrench_reference(
        self,
        body_force: torch.Tensor,
        body_torque: torch.Tensor,
        xipos: torch.Tensor,
        subtree_com: torch.Tensor,
        cdof: torch.Tensor,
    ) -> torch.Tensor:
        return project_body_wrench_to_torque(
            body_force=body_force,
            body_torque=body_torque,
            xipos=xipos,
            subtree_com=subtree_com,
            cdof=cdof,
            body_count=self.body_count,
            full_torque_dim=self.full_torque_dim,
            root_dof_count=self.root_dof_count,
            body_rootid=self._body_rootid_py,
            dof_subtree_mask=self.dof_subtree_mask,
        )

    def _build_segment_projection_terms(
        self,
        unit_force_magnitude: torch.Tensor,
        seg_start: torch.Tensor | None,
        seg_end: torch.Tensor | None,
        seg_dir: torch.Tensor | None,
        xipos: torch.Tensor,
        subtree_com: torch.Tensor,
    ) -> _SegmentProjectionTerms | None:
        if (
            seg_start is None
            or seg_end is None
            or seg_dir is None
            or self.apply_segment_start_idx.numel() == 0
        ):
            return None

        batch_size = unit_force_magnitude.shape[0]
        unit_seg_force = (
            unit_force_magnitude.index_select(
                1, self.apply_segment_muscle_idx
            ).unsqueeze(-1)
            * seg_dir
        )
        neg_unit_seg_force = -unit_seg_force
        body_start_idx = self.apply_segment_body_start_idx
        body_end_idx = self.apply_segment_body_end_idx
        com_start = xipos.index_select(1, body_start_idx)
        com_end = xipos.index_select(1, body_end_idx)
        unit_torque_start = torch.cross(seg_start - com_start, unit_seg_force, dim=-1)
        unit_torque_end = torch.cross(seg_end - com_end, neg_unit_seg_force, dim=-1)

        subtree_com_start = subtree_com.index_select(
            1, self.segment_start_root_body_idx
        )
        subtree_com_end = subtree_com.index_select(1, self.segment_end_root_body_idx)
        offset_start = com_start - subtree_com_start
        offset_end = com_end - subtree_com_end
        muscle_index = self.apply_segment_muscle_idx.view(1, 1, -1).expand(
            batch_size, self.bound_chunk_size, -1
        )
        return _SegmentProjectionTerms(
            unit_seg_force_t=unit_seg_force.transpose(1, 2),
            neg_unit_seg_force_t=neg_unit_seg_force.transpose(1, 2),
            unit_torque_start_t=unit_torque_start.transpose(1, 2),
            unit_torque_end_t=unit_torque_end.transpose(1, 2),
            offset_cross_force_start_t=torch.cross(
                offset_start, unit_seg_force, dim=-1
            ).transpose(1, 2),
            offset_cross_force_end_t=torch.cross(
                offset_end, neg_unit_seg_force, dim=-1
            ).transpose(1, 2),
            muscle_index=muscle_index,
        )

    def _segment_projection_chunk(
        self,
        terms: _SegmentProjectionTerms,
        cdof: torch.Tensor,
        torque_start_idx: int,
        torque_end_idx: int,
    ) -> torch.Tensor:
        full_slice = slice(
            torque_start_idx + self.root_dof_count,
            torque_end_idx + self.root_dof_count,
        )
        torque_slice = slice(torque_start_idx, torque_end_idx)
        linear_cdof = cdof[:, full_slice, 3:]
        angular_cdof = cdof[:, full_slice, :3]
        segment_contrib = (
            torch.matmul(linear_cdof, terms.unit_seg_force_t)
            + torch.matmul(angular_cdof, terms.unit_torque_start_t)
            + torch.matmul(angular_cdof, terms.offset_cross_force_start_t)
        ) * self.segment_start_dof_mask[torque_slice].unsqueeze(0)
        segment_contrib.add_(
            (
                torch.matmul(linear_cdof, terms.neg_unit_seg_force_t)
                + torch.matmul(angular_cdof, terms.unit_torque_end_t)
                + torch.matmul(angular_cdof, terms.offset_cross_force_end_t)
            )
            * self.segment_end_dof_mask[torque_slice].unsqueeze(0)
        )
        return segment_contrib

    def _compute_muscle_dynamics_bounds_from_segments(
        self,
        unit_force_magnitude: torch.Tensor,
        seg_start: torch.Tensor | None,
        seg_end: torch.Tensor | None,
        seg_dir: torch.Tensor | None,
        xipos: torch.Tensor,
        subtree_com: torch.Tensor,
        cdof: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = unit_force_magnitude.shape[0]
        lower = torch.zeros(
            batch_size,
            self.torque_dim,
            device=unit_force_magnitude.device,
            dtype=unit_force_magnitude.dtype,
        )
        upper = torch.zeros_like(lower)
        terms = self._build_segment_projection_terms(
            unit_force_magnitude,
            seg_start,
            seg_end,
            seg_dir,
            xipos,
            subtree_com,
        )
        if terms is None:
            return lower, upper

        muscle_contrib = torch.zeros(
            batch_size,
            self.bound_chunk_size,
            self.muscle_count,
            device=unit_force_magnitude.device,
            dtype=unit_force_magnitude.dtype,
        )
        for torque_start_idx in range(0, self.torque_dim, self.bound_chunk_size):
            torque_end_idx = min(
                torque_start_idx + self.bound_chunk_size, self.torque_dim
            )
            chunk_len = torque_end_idx - torque_start_idx
            torque_slice = slice(torque_start_idx, torque_end_idx)

            segment_contrib = self._segment_projection_chunk(
                terms, cdof, torque_start_idx, torque_end_idx
            )
            muscle_contrib_chunk = muscle_contrib[:, :chunk_len]
            muscle_contrib_chunk.zero_()
            muscle_contrib_chunk.scatter_add_(
                2, terms.muscle_index[:, :chunk_len], segment_contrib
            )
            lower_chunk, upper_chunk = self._activation_interval_bounds(
                muscle_contrib_chunk
            )
            lower[:, torque_slice].copy_(lower_chunk)
            upper[:, torque_slice].copy_(upper_chunk)
        return lower, upper

    def _lbs_segment_entry_contribution_torch(
        self,
        *,
        torque_idx: int,
        segment_idx: int,
        force_magnitude: torch.Tensor,
        transformed_waypoints: torch.Tensor,
        xpos: torch.Tensor,
        xquat: torch.Tensor,
        subtree_com: torch.Tensor,
        cdof: torch.Tensor,
    ) -> torch.Tensor:
        start_wp = int(self.apply_segment_start_idx[segment_idx])
        end_wp = int(self.apply_segment_end_idx[segment_idx])
        p_start = transformed_waypoints[:, start_wp]
        p_end = transformed_waypoints[:, end_wp]
        segment = p_end - p_start
        segment_length = torch.linalg.norm(segment, dim=-1, keepdim=True)
        valid = segment_length.squeeze(-1) >= 1.0e-8
        seg_force = (
            force_magnitude.unsqueeze(-1) * segment / segment_length.clamp_min(1.0e-8)
        )
        full_torque_idx = torque_idx + self.root_dof_count
        linear_cdof = cdof[:, full_torque_idx, 3:]
        angular_cdof = cdof[:, full_torque_idx, :3]
        total = torch.zeros_like(force_magnitude)

        start_begin = self._waypoint_attachment_start_py[start_wp]
        start_end = self._waypoint_attachment_start_py[start_wp + 1]
        for attach_idx in range(start_begin, start_end):
            body_idx = self._waypoint_attachment_body_py[attach_idx]
            local = self.muscle_waypoint_attachment_local[attach_idx].view(1, 3)
            point = xpos[:, body_idx] + quat_rotate(
                xquat[:, body_idx], local.expand(xpos.shape[0], -1)
            )
            root_idx = self._body_rootid_py[body_idx]
            coeff = self.dof_subtree_mask[full_torque_idx, body_idx].to(
                dtype=force_magnitude.dtype
            )
            weight = self.muscle_waypoint_attachment_weight[attach_idx].to(
                dtype=force_magnitude.dtype
            )
            term = linear_cdof + torch.cross(
                angular_cdof, point - subtree_com[:, root_idx], dim=-1
            )
            total = total + coeff * weight * torch.sum(term * seg_force, dim=-1)

        end_begin = self._waypoint_attachment_start_py[end_wp]
        end_end = self._waypoint_attachment_start_py[end_wp + 1]
        for attach_idx in range(end_begin, end_end):
            body_idx = self._waypoint_attachment_body_py[attach_idx]
            local = self.muscle_waypoint_attachment_local[attach_idx].view(1, 3)
            point = xpos[:, body_idx] + quat_rotate(
                xquat[:, body_idx], local.expand(xpos.shape[0], -1)
            )
            root_idx = self._body_rootid_py[body_idx]
            coeff = self.dof_subtree_mask[full_torque_idx, body_idx].to(
                dtype=force_magnitude.dtype
            )
            weight = self.muscle_waypoint_attachment_weight[attach_idx].to(
                dtype=force_magnitude.dtype
            )
            term = linear_cdof + torch.cross(
                angular_cdof, point - subtree_com[:, root_idx], dim=-1
            )
            total = total - coeff * weight * torch.sum(term * seg_force, dim=-1)

        return torch.where(valid, total, torch.zeros_like(total))

    def _compute_lbs_torque_from_force_from_entries(
        self,
        force_magnitude: torch.Tensor,
        geometry: _WaypointGeometry,
    ) -> torch.Tensor:
        batch_size = force_magnitude.shape[0]
        mean = torch.zeros(
            batch_size,
            self.torque_dim,
            device=force_magnitude.device,
            dtype=force_magnitude.dtype,
        )
        if self.torque_active_segment_idx.numel() == 0:
            return mean

        for torque_idx in range(self.torque_dim):
            entry_begin = self._torque_active_entry_offsets_py[torque_idx]
            entry_end = self._torque_active_entry_offsets_py[torque_idx + 1]
            if entry_begin == entry_end:
                continue
            total = torch.zeros(
                batch_size, device=force_magnitude.device, dtype=force_magnitude.dtype
            )
            for entry_idx in range(entry_begin, entry_end):
                muscle_idx = self._torque_active_muscle_idx_py[entry_idx]
                segment_idx = self._torque_active_segment_idx_py[entry_idx]
                total = total + self._lbs_segment_entry_contribution_torch(
                    torque_idx=torque_idx,
                    segment_idx=segment_idx,
                    force_magnitude=force_magnitude[:, muscle_idx],
                    transformed_waypoints=geometry.transformed_waypoints,
                    xpos=geometry.xpos,
                    xquat=geometry.xquat,
                    subtree_com=geometry.subtree_com,
                    cdof=geometry.cdof,
                )
            mean[:, torque_idx].copy_(total)
        return mean

    def _compute_lbs_muscle_dynamics_contribution_from_entries(
        self,
        unit_force_magnitude: torch.Tensor,
        geometry: _WaypointGeometry,
    ) -> torch.Tensor:
        batch_size = unit_force_magnitude.shape[0]
        contribution = torch.zeros(
            batch_size,
            self.torque_dim,
            self.muscle_count,
            device=unit_force_magnitude.device,
            dtype=unit_force_magnitude.dtype,
        )
        if self.torque_active_segment_idx.numel() == 0:
            return contribution

        for torque_idx in range(self.torque_dim):
            entry_begin = self._torque_active_entry_offsets_py[torque_idx]
            entry_end = self._torque_active_entry_offsets_py[torque_idx + 1]
            for entry_idx in range(entry_begin, entry_end):
                muscle_idx = self._torque_active_muscle_idx_py[entry_idx]
                segment_idx = self._torque_active_segment_idx_py[entry_idx]
                contribution[:, torque_idx, muscle_idx] += (
                    self._lbs_segment_entry_contribution_torch(
                        torque_idx=torque_idx,
                        segment_idx=segment_idx,
                        force_magnitude=unit_force_magnitude[:, muscle_idx],
                        transformed_waypoints=geometry.transformed_waypoints,
                        xpos=geometry.xpos,
                        xquat=geometry.xquat,
                        subtree_com=geometry.subtree_com,
                        cdof=geometry.cdof,
                    )
                )
        return contribution

    def _compute_lbs_muscle_dynamics_bounds_from_entries(
        self,
        unit_force_magnitude: torch.Tensor,
        geometry: _WaypointGeometry,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = unit_force_magnitude.shape[0]
        lower = torch.zeros(
            batch_size,
            self.torque_dim,
            device=unit_force_magnitude.device,
            dtype=unit_force_magnitude.dtype,
        )
        upper = torch.zeros_like(lower)
        if self.torque_active_segment_idx.numel() == 0:
            return lower, upper

        for torque_idx in range(self.torque_dim):
            entry_begin = self._torque_active_entry_offsets_py[torque_idx]
            entry_end = self._torque_active_entry_offsets_py[torque_idx + 1]
            current_muscle = -1
            current_total: torch.Tensor | None = None
            for entry_idx in range(entry_begin, entry_end):
                muscle_idx = self._torque_active_muscle_idx_py[entry_idx]
                if current_total is not None and muscle_idx != current_muscle:
                    lower[:, torque_idx] += torch.clamp(current_total, max=0.0)
                    upper[:, torque_idx] += torch.clamp(current_total, min=0.0)
                    current_total = None
                segment_idx = self._torque_active_segment_idx_py[entry_idx]
                contribution = self._lbs_segment_entry_contribution_torch(
                    torque_idx=torque_idx,
                    segment_idx=segment_idx,
                    force_magnitude=unit_force_magnitude[:, muscle_idx],
                    transformed_waypoints=geometry.transformed_waypoints,
                    xpos=geometry.xpos,
                    xquat=geometry.xquat,
                    subtree_com=geometry.subtree_com,
                    cdof=geometry.cdof,
                )
                current_muscle = muscle_idx
                current_total = (
                    contribution
                    if current_total is None
                    else current_total + contribution
                )
            if current_total is not None:
                lower[:, torque_idx] += torch.clamp(current_total, max=0.0)
                upper[:, torque_idx] += torch.clamp(current_total, min=0.0)
        return lower, upper

    def torque_from_activation(
        self,
        activation: torch.Tensor,
        raw_context: torch.Tensor | None,
        *,
        include_passive_force: bool | None = None,
        lbs_unit_force_magnitude: torch.Tensor | None = None,
        lbs_passive_force_magnitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._torque_from_activation_impl(
            activation.float(),
            raw_context,
            include_passive_force=include_passive_force,
            lbs_unit_force_magnitude=lbs_unit_force_magnitude,
            lbs_passive_force_magnitude=lbs_passive_force_magnitude,
        )

    def _passive_force_length(
        self,
        normalized_length: torch.Tensor,
    ) -> torch.Tensor:
        return passive_force_length_torch(
            normalized_length,
            force_type=self.muscle_passive_force_type,
            k_pe=self.muscle_passive_force_k_pe,
            e_mo=self.muscle_passive_force_e_mo,
            lmax=self.muscle_passive_force_lmax,
            fpmax=self.muscle_passive_force_fpmax,
        )

    def _active_force_length(
        self,
        normalized_length: torch.Tensor,
    ) -> torch.Tensor:
        return active_force_length_torch(
            normalized_length,
            force_type=self.muscle_active_force_type,
            gamma=self.muscle_active_force_gamma,
            lmin=self.muscle_active_force_lmin,
            lmax=self.muscle_active_force_lmax,
        )

    def _waypoint_unit_moment_torch(
        self,
        geometry: _WaypointGeometry,
    ) -> torch.Tensor:
        """Unit-force generalized force per muscle, shape (B, torque_dim, muscles)."""
        ones = torch.ones_like(geometry.unit_force_magnitude)
        if self.uses_lbs_waypoint_attachments:
            return self._compute_lbs_muscle_dynamics_contribution_from_entries(
                ones,
                geometry,
            )
        moment = self._compute_full_muscle_dynamics_contribution(
            unit_force_magnitude=ones,
            seg_start=geometry.seg_start,
            seg_end=geometry.seg_end,
            seg_dir=geometry.seg_dir,
            xipos=geometry.xipos,
            subtree_com=geometry.subtree_com,
            cdof=geometry.cdof,
        )
        return moment

    def _waypoint_path_velocity_torch(
        self,
        raw_context: torch.Tensor,
        moment: torch.Tensor,
    ) -> torch.Tensor:
        """Path velocity ``-sum_d moment_d qvel_d`` from the context's qvel."""
        qvel = raw_context[
            :,
            self._geometry_context_dim : self._geometry_context_dim
            + self.full_torque_dim,
        ]
        root_dof_count = self.full_torque_dim - self.torque_dim
        return -torch.sum(
            moment * qvel[:, root_dof_count:].unsqueeze(-1),
            dim=1,
        )

    def _waypoint_force_velocity_multiplier(
        self,
        raw_context: torch.Tensor,
        geometry: _WaypointGeometry,
    ) -> torch.Tensor:
        length_velocity = self._waypoint_path_velocity_torch(
            raw_context,
            self._waypoint_unit_moment_torch(geometry),
        )
        velocity_scale = (
            self.muscle_force_velocity_length_scale * self.muscle_force_velocity_vmax
        ).clamp_min(1.0e-6)
        normalized_velocity = length_velocity / velocity_scale.unsqueeze(0)
        fvmax = self.muscle_force_velocity_fvmax
        eccentric_width = fvmax - 1.0
        concentric = (normalized_velocity + 1.0).square()
        remaining = eccentric_width - normalized_velocity
        eccentric = fvmax - remaining.square() / eccentric_width
        return torch.where(
            normalized_velocity <= -1.0,
            torch.zeros_like(normalized_velocity),
            torch.where(
                normalized_velocity <= 0.0,
                concentric,
                torch.where(
                    normalized_velocity <= eccentric_width,
                    eccentric,
                    torch.full_like(normalized_velocity, fvmax),
                ),
            ),
        )

    def _segment_force_magnitude_to_body_wrench(
        self,
        force_magnitude: torch.Tensor,
        *,
        seg_start: torch.Tensor | None,
        seg_end: torch.Tensor | None,
        seg_dir: torch.Tensor | None,
        xipos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return segment_force_magnitude_to_body_wrench(
            force_magnitude,
            seg_start=seg_start,
            seg_end=seg_end,
            seg_dir=seg_dir,
            xipos=xipos,
            body_count=self.body_count,
            apply_segment_muscle_idx=self.apply_segment_muscle_idx,
            apply_segment_start_idx=self.apply_segment_start_idx,
            apply_segment_body_start_idx=self.apply_segment_body_start_idx,
            apply_segment_body_end_idx=self.apply_segment_body_end_idx,
        )

    def _reserve_native_wrap_workspace(self, workspace, batch_size, device):
        if not self._waypoint_wrapping.count:
            return None
        state = self._get_waypoint_wrap_state(batch_size, device)
        if "transformed_waypoints" not in workspace:
            workspace["transformed_waypoints"] = torch.empty(
                (batch_size, self.muscle_waypoints_local.shape[0], 3),
                device=device,
                dtype=torch.float32,
            )
        return state

    def _transform_warp_waypoints(self, workspace, body_pos, body_quat):
        """Share world coordinates across force, cache and adjoint paths.

        For fixed2 attachments the caller first fills xquat_normalized.
        Equivalent local transforms can round differently after large scene
        translations and change the directions of short segments.
        """
        if self._uses_attachment_projection and self.lbs_waypoint_attachment_fixed2:
            kernel = _transform_lbs_waypoints_fixed2_normalized_quat_from_torch_kernel
            inputs = [
                wp.from_torch(body_pos, dtype=wp.vec3),
                wp.from_torch(workspace["xquat_normalized"], dtype=wp.quat),
                wp.from_torch(self.lbs_waypoint_attachment_body2, dtype=wp.vec2i),
                wp.from_torch(
                    self.lbs_waypoint_attachment_local_weight0, dtype=wp.vec4
                ),
                wp.from_torch(
                    self.lbs_waypoint_attachment_local_weight1, dtype=wp.vec4
                ),
            ]
        elif self._uses_attachment_projection:
            kernel = _transform_lbs_waypoints_from_torch_kernel
            inputs = [
                wp.from_torch(body_pos, dtype=wp.vec3),
                wp.from_torch(body_quat, dtype=wp.quat),
                wp.from_torch(self.muscle_waypoint_attachment_start_i32),
                wp.from_torch(self.muscle_waypoint_attachment_body_i32),
                wp.from_torch(self.muscle_waypoint_attachment_local, dtype=wp.vec3),
                wp.from_torch(self.muscle_waypoint_attachment_weight),
            ]
        else:
            kernel = _transform_waypoints_from_torch_kernel
            inputs = [
                wp.from_torch(body_pos, dtype=wp.vec3),
                wp.from_torch(body_quat, dtype=wp.quat),
                wp.from_torch(self.muscle_waypoint_body_i32),
                wp.from_torch(self.muscle_waypoints_local, dtype=wp.vec3),
            ]
        wp.launch(
            kernel,
            dim=(body_pos.shape[0], self.muscle_waypoints_local.shape[0]),
            inputs=inputs,
            outputs=[wp.from_torch(workspace["transformed_waypoints"], dtype=wp.vec3)],
            device=str(body_pos.device),
        )

    def _prepare_native_wrap_geometry(self, workspace, body_pos, body_quat):
        state = self._reserve_native_wrap_workspace(
            workspace, body_pos.shape[0], body_pos.device
        )
        if state is None:
            return
        self._geometry_scratch_generation += 1
        self._transform_warp_waypoints(workspace, body_pos, body_quat)
        self._waypoint_wrapping.transform(
            wp.from_torch(body_pos, dtype=wp.vec3),
            wp.from_torch(body_quat, dtype=wp.quat),
            wp.from_torch(workspace["transformed_waypoints"], dtype=wp.vec3),
            state,
        )

    def _native_wrap_inputs(self, workspace, batch_size):
        state = (
            self._get_waypoint_wrap_state(batch_size, self.muscle_params.device)
            if self._waypoint_wrapping.count
            else None
        )
        points = workspace.get("transformed_waypoints")
        return [
            wp.from_torch(points, dtype=wp.vec3) if points is not None else None,
            wp.from_torch(self.lbs_native_point_source),
            self._waypoint_wrapping.kernel_state(state),
        ]

    def prepare_waypoint_wrap_workspace(self, *, batch_size: int) -> None:
        """Reserve the caller's batch geometry before its first graph capture."""
        if self._waypoint_wrapping.count:
            self._get_waypoint_wrap_state(batch_size, self.muscle_params.device)

    def _get_waypoint_wrap_state(self, batch_size, device):
        key = self._warp_workspace_key(device=device, batch_size=batch_size)
        state = self._waypoint_wrap_states.get(key)
        if state is None:
            if wp.get_device(str(device)).is_capturing:
                raise RuntimeError(
                    "Wrapped geometry must be prepared before graph capture."
                )
            state = WaypointWrapState(
                self._waypoint_wrapping.count, batch_size, device, self.muscle_count
            )
            self._waypoint_wrap_states[key] = state
        return state

    def _prepare_wrapped_points(self, points, xpos, xquat):
        state = self._get_waypoint_wrap_state(points.shape[0], points.device)
        self._waypoint_wrapping.transform(
            wp.from_torch(xpos, dtype=wp.vec3),
            wp.from_torch(xquat, dtype=wp.quat),
            wp.from_torch(points, dtype=wp.vec3),
            state,
        )
        wp.launch(
            path_lengths_kernel,
            dim=(points.shape[0], self.muscle_count),
            inputs=[
                wp.from_torch(points, dtype=wp.vec3),
                wp.from_torch(self.muscle_waypoint_start_i32),
                self._waypoint_wrapping.point_arc,
                state.arc,
            ],
            outputs=[state.lengths],
            device=str(points.device),
        )
        return state

    def _compute_waypoint_geometry(
        self,
        raw_context: torch.Tensor,
        *,
        include_passive_force: bool = False,
    ) -> _WaypointGeometry:
        raw_context = raw_context.float()
        batch_size = raw_context.shape[0]
        xpos = raw_context[:, : self._xpos_width].reshape(
            batch_size, self.body_count, 3
        )
        xquat = raw_context[
            :, self._xpos_width : self._xpos_width + self._xquat_width
        ].reshape(batch_size, self.body_count, 4)
        offset = self._xpos_width + self._xquat_width
        xipos = raw_context[:, offset : offset + self._xipos_width].reshape(
            batch_size, self.body_count, 3
        )
        offset += self._xipos_width
        subtree_com = raw_context[:, offset : offset + self._subtree_com_width].reshape(
            batch_size, self.body_count, 3
        )
        offset += self._subtree_com_width
        cdof = raw_context[:, offset : offset + self._cdof_width].reshape(
            batch_size, self.full_torque_dim, 6
        )
        xquat = quat_normalize(xquat)

        if self.uses_lbs_waypoint_attachments:
            attachment_body = self.muscle_waypoint_attachment_body
            attachment_pos = xpos.index_select(1, attachment_body)
            attachment_quat = xquat.index_select(1, attachment_body)
            attachment_world = attachment_pos + quat_rotate(
                attachment_quat,
                self.muscle_waypoint_attachment_local.unsqueeze(0).expand(
                    batch_size, -1, -1
                ),
            )
            attachment_world = (
                attachment_world * self.muscle_waypoint_attachment_weight.view(1, -1, 1)
            )
            waypoint_idx = torch.repeat_interleave(
                torch.arange(
                    self.muscle_waypoints_local.shape[0],
                    device=raw_context.device,
                    dtype=torch.long,
                ),
                (
                    self.muscle_waypoint_attachment_start[1:]
                    - self.muscle_waypoint_attachment_start[:-1]
                ).to(torch.long),
            )
            transformed_waypoints = torch.zeros(
                batch_size,
                self.muscle_waypoints_local.shape[0],
                3,
                device=raw_context.device,
                dtype=raw_context.dtype,
            )
            transformed_waypoints.scatter_add_(
                1,
                waypoint_idx.view(1, -1, 1).expand(batch_size, -1, 3),
                attachment_world,
            )
        else:
            waypoint_body = self.muscle_waypoint_body
            waypoint_pos = xpos.index_select(1, waypoint_body)
            waypoint_quat = xquat.index_select(1, waypoint_body)
            transformed_waypoints = waypoint_pos + quat_rotate(
                waypoint_quat,
                self.muscle_waypoints_local.unsqueeze(0).expand(batch_size, -1, -1),
            )

        muscle_length = torch.zeros(
            batch_size,
            self.muscle_count,
            device=raw_context.device,
            dtype=raw_context.dtype,
        )
        if self.all_segment_start_idx.numel() > 0:
            all_seg_start = transformed_waypoints.index_select(
                1, self.all_segment_start_idx
            )
            all_seg_end = transformed_waypoints.index_select(
                1, self.all_segment_start_idx + 1
            )
            all_seg_len = torch.linalg.norm(all_seg_end - all_seg_start, dim=-1)
            muscle_length.scatter_add_(
                1,
                self.all_segment_muscle_idx.unsqueeze(0).expand(batch_size, -1),
                all_seg_len,
            )

        if self._waypoint_wrapping.count:
            state = self._prepare_wrapped_points(transformed_waypoints, xpos, xquat)
            muscle_length = state.lengths_tensor
        f0 = self.muscle_params[:, 0]
        l_m0 = self.muscle_params[:, 1].clamp_min(1.0e-6)
        l_t0 = self.muscle_params[:, 2]
        pen_angle = self.muscle_params[:, 3]
        muscle_tendon_length0 = self.muscle_rest_length.clamp_min(1.0e-6)
        l_m = (muscle_length / muscle_tendon_length0.unsqueeze(0)) - l_t0.unsqueeze(0)
        l_m_norm = l_m / l_m0.unsqueeze(0)
        l_m_norm_clamped = torch.clamp(
            l_m_norm,
            min=self.muscle_length_norm_min,
            max=self.muscle_length_norm_max,
        )
        # Match the env kernel: active force-length uses the clamped normalized
        # muscle length so jump/high-ROM frames do not underflow to zero only in
        # the physical muscle helper.
        f_a = self._active_force_length(l_m_norm_clamped)
        unit_force_magnitude = (
            f0.unsqueeze(0)
            * f_a
            * torch.cos(pen_angle).unsqueeze(0)
            * self.muscle_force_scale
        )
        passive_force_magnitude: torch.Tensor | None = None
        if include_passive_force:
            f_p = self._passive_force_length(l_m_norm_clamped)
            passive_force_magnitude = (
                f0.unsqueeze(0)
                * f_p
                * torch.cos(pen_angle).unsqueeze(0)
                * self.muscle_force_scale
            )

        seg_start: torch.Tensor | None = None
        seg_end: torch.Tensor | None = None
        seg_dir: torch.Tensor | None = None
        if self.apply_segment_start_idx.numel() > 0:
            seg_start = transformed_waypoints.index_select(
                1, self.apply_segment_start_idx
            )
            seg_end = transformed_waypoints.index_select(1, self.apply_segment_end_idx)
            seg_vec = seg_end - seg_start
            seg_len = torch.linalg.norm(seg_vec, dim=-1, keepdim=True).clamp_min(1.0e-8)
            seg_dir = seg_vec / seg_len

        return _WaypointGeometry(
            unit_force_magnitude=unit_force_magnitude,
            passive_force_magnitude=passive_force_magnitude,
            seg_start=seg_start,
            seg_end=seg_end,
            seg_dir=seg_dir,
            transformed_waypoints=transformed_waypoints,
            xpos=xpos,
            xquat=xquat,
            xipos=xipos,
            subtree_com=subtree_com,
            cdof=cdof,
            muscle_length=muscle_length,
        )

    def _torque_from_activation_impl(
        self,
        activation: torch.Tensor,
        raw_context: torch.Tensor | None,
        *,
        include_passive_force: bool | None = None,
        lbs_unit_force_magnitude: torch.Tensor | None = None,
        lbs_passive_force_magnitude: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_supported_physical_geometry()
        if raw_context is None:
            raise ValueError("Muscle dynamics requires a raw geometry context tensor.")
        if raw_context.shape[-1] != self.context_dim:
            raise ValueError(
                "Unexpected raw muscle context width: "
                f"expected {self.context_dim}, got {raw_context.shape[-1]}."
            )
        if activation.shape[-1] != self.muscle_count:
            raise ValueError(
                "Unexpected activation width: "
                f"expected {self.muscle_count}, got {activation.shape[-1]}."
            )

        # Keep the muscle proxy in float32 even under autocast so scatter_add and
        # geometry accumulation stay dtype-consistent across compile/graph paths.
        activation = activation.float()
        include_passive_force = (
            self.include_passive_force_in_output
            if include_passive_force is None
            else bool(include_passive_force)
        )
        activation = activation.clamp(0.0, 1.0)
        raw_context = raw_context.float()
        if self._can_use_warp_muscle_dynamics_forward(raw_context):
            if lbs_unit_force_magnitude is not None:
                if lbs_passive_force_magnitude is None:
                    lbs_passive_force_magnitude = raw_context.new_empty(0)
                mean = _warp_muscle_dynamics_from_activation_cached_lbs_force_eager(
                    activation,
                    raw_context,
                    lbs_unit_force_magnitude,
                    lbs_passive_force_magnitude,
                    self,
                    include_passive_force=include_passive_force,
                )
            else:
                mean = _warp_muscle_dynamics_from_activation_eager(
                    activation,
                    raw_context,
                    self,
                    include_passive_force=include_passive_force,
                )
            return mean

        geometry = self._compute_waypoint_geometry(
            raw_context,
            include_passive_force=include_passive_force,
        )
        active_unit_force_magnitude = geometry.unit_force_magnitude
        if self.muscle_force_velocity_enabled:
            active_unit_force_magnitude = (
                active_unit_force_magnitude
                * self._waypoint_force_velocity_multiplier(raw_context, geometry)
            )
        force_magnitude = active_unit_force_magnitude * activation
        if include_passive_force and geometry.passive_force_magnitude is not None:
            force_magnitude = force_magnitude + geometry.passive_force_magnitude

        if self.uses_lbs_waypoint_attachments:
            mean = self._compute_lbs_torque_from_force_from_entries(
                force_magnitude,
                geometry,
            )
            return mean

        body_force, body_torque = self._segment_force_magnitude_to_body_wrench(
            force_magnitude,
            seg_start=geometry.seg_start,
            seg_end=geometry.seg_end,
            seg_dir=geometry.seg_dir,
            xipos=geometry.xipos,
        )
        mean = self._project_body_wrench_reference(
            body_force=body_force,
            body_torque=body_torque,
            xipos=geometry.xipos,
            subtree_com=geometry.subtree_com,
            cdof=geometry.cdof,
        )
        return mean

    def _can_use_warp_torque_jacobian(self, raw_context: torch.Tensor) -> bool:
        return (
            self.forward_backend != "reference"
            and self.geometry_backend == "waypoint"
            and raw_context.device.type == "cuda"
        )

    @property
    def forward_backend(self) -> str:
        return self._forward_backend

    @forward_backend.setter
    def forward_backend(self, backend: str) -> None:
        self.set_forward_backend(backend)

    def set_forward_backend(self, backend: str) -> None:
        """Pin `forward` to one implementation, or back to "auto".

        Selecting "warp" does not make the kernels run where they cannot: the
        device still has to be CUDA and the geometry backend still has to have
        a kernel path, so this narrows the choice rather than widening it.
        """

        if backend not in self.FORWARD_BACKENDS:
            raise ValueError(
                "muscle forward backend must be one of "
                f"{self.FORWARD_BACKENDS}, got {backend!r}."
            )
        self._forward_backend = backend

    def _supports_warp_muscle_dynamics_forward(
        self, device: torch.device | None = None
    ) -> bool:
        if self.forward_backend == "reference":
            return False
        if self.geometry_backend != "waypoint":
            return False
        if device is None:
            device = self.muscle_params.device
        return torch.device(device).type == "cuda"

    def _require_supported_physical_geometry(self) -> None:
        if self.geometry_backend != "waypoint":
            raise RuntimeError(
                "Muscle mechanics requires body-fixed waypoint geometry."
            )

    def _can_use_warp_muscle_dynamics_forward(
        self,
        raw_context: torch.Tensor,
    ) -> bool:
        return self._supports_warp_muscle_dynamics_forward(raw_context.device)

    @staticmethod
    def _warp_workspace_key(
        *,
        device: torch.device,
        batch_size: int,
        workspace_kind: str = "full",
    ) -> tuple[str, int, str]:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            return (
                f"cuda:{torch.cuda.current_device()}",
                int(batch_size),
                str(workspace_kind),
            )
        return (str(device), int(batch_size), str(workspace_kind))

    def _get_warp_geometry_workspace(
        self,
        batch_size: int,
        device: torch.device,
        *,
        minimal_native_lbs: bool = False,
    ) -> dict[str, torch.Tensor]:
        key = self._warp_workspace_key(
            device=device,
            batch_size=batch_size,
            workspace_kind="native_lbs_min" if minimal_native_lbs else "full",
        )
        omit_passive_force_workspace = (
            self._uses_attachment_projection
            and self.lbs_forward_reduction == "partial"
            and self.lbs_attachment_layout == "body_wrench"
        )
        workspace = self._warp_geometry_workspace.get(key)
        if workspace is not None:
            if (
                not omit_passive_force_workspace
                and "passive_force_magnitude" not in workspace
            ):
                workspace["passive_force_magnitude"] = torch.empty(
                    batch_size,
                    self.muscle_count,
                    device=device,
                    dtype=torch.float32,
                )
            if (
                self.lbs_torque_chunk_count > 0
                and self.lbs_forward_reduction == "partial"
                and not self.uses_lbs_waypoint_attachments
                and "lbs_torque_chunk_partial" not in workspace
            ):
                workspace["lbs_torque_chunk_partial"] = torch.empty(
                    batch_size,
                    self.lbs_torque_chunk_count,
                    device=device,
                    dtype=torch.float32,
                )
            if (
                self._uses_attachment_projection
                and self.lbs_forward_reduction == "partial"
                and self.lbs_waypoint_attachment_fixed2
                and "xquat_normalized" not in workspace
            ):
                workspace["xquat_normalized"] = torch.empty(
                    batch_size,
                    self.body_count,
                    4,
                    device=device,
                    dtype=torch.float32,
                )
            if (
                self._uses_attachment_projection
                and self.lbs_forward_reduction == "partial"
            ):
                if not minimal_native_lbs and "lbs_muscle_length" not in workspace:
                    workspace["lbs_muscle_length"] = torch.empty(
                        batch_size,
                        self.muscle_count,
                        device=device,
                        dtype=torch.float32,
                    )
                if not minimal_native_lbs and "lbs_force_by_muscle" not in workspace:
                    workspace["lbs_force_by_muscle"] = torch.empty(
                        batch_size,
                        self.muscle_count,
                        device=device,
                        dtype=torch.float32,
                    )
                if (
                    self.lbs_attachment_layout == "body_wrench"
                    and self.lbs_segment_start_is_segment_plus_muscle
                    and not minimal_native_lbs
                    and "lbs_segment_length_group_partials" not in workspace
                ):
                    workspace["lbs_segment_length_group_partials"] = torch.empty(
                        batch_size,
                        self.muscle_count
                        * self.lbs_body_wrench_force_segment_group_width,
                        device=device,
                        dtype=torch.float32,
                    )
                if (
                    self.lbs_attachment_layout == "waypoint_force"
                    and "lbs_waypoint_force" not in workspace
                ):
                    workspace["lbs_waypoint_force"] = torch.empty(
                        batch_size,
                        int(self.lbs_waypoint_count),
                        3,
                        device=device,
                        dtype=torch.float32,
                    )
                if (
                    self.lbs_attachment_layout == "waypoint_force"
                    and self.lbs_active_attach_chunk_count > 0
                    and "lbs_active_attach_chunk_partial" not in workspace
                ):
                    workspace["lbs_active_attach_chunk_partial"] = torch.empty(
                        batch_size,
                        self.lbs_active_attach_chunk_count,
                        device=device,
                        dtype=torch.float32,
                    )
                if (
                    self.lbs_attachment_layout == "body_wrench"
                    and self.lbs_body_event_chunk_count > 0
                ):
                    if (
                        self.lbs_body_wrench_forward_mode == "native_tile"
                        and self.lbs_native_body_tile_body_slot_count > 0
                        and "lbs_native_body_wrench_tile_partial" not in workspace
                    ):
                        workspace["lbs_native_body_wrench_tile_partial"] = torch.empty(
                            batch_size,
                            self.lbs_native_body_tile_body_slot_count,
                            6,
                            device=device,
                            dtype=torch.float32,
                        )
                    if (
                        not minimal_native_lbs
                        and "lbs_body_wrench_chunk_partial" not in workspace
                    ):
                        workspace["lbs_body_wrench_chunk_partial"] = torch.empty(
                            batch_size,
                            self.lbs_body_event_chunk_count,
                            6,
                            device=device,
                            dtype=torch.float32,
                        )
                    if "lbs_body_force" not in workspace:
                        workspace["lbs_body_force"] = torch.empty(
                            batch_size,
                            self.body_count,
                            3,
                            device=device,
                            dtype=torch.float32,
                        )
                    if "lbs_body_torque" not in workspace:
                        workspace["lbs_body_torque"] = torch.empty(
                            batch_size,
                            self.body_count,
                            3,
                            device=device,
                            dtype=torch.float32,
                        )
            return workspace

        waypoint_count = int(self.muscle_waypoints_local.shape[0])
        workspace = {
            "unit_force_magnitude": torch.empty(
                batch_size,
                self.muscle_count,
                device=device,
                dtype=torch.float32,
            ),
            "body_linear_adjoint": torch.empty(
                batch_size,
                self.body_count,
                3,
                device=device,
                dtype=torch.float32,
            ),
            "body_angular_adjoint": torch.empty(
                batch_size,
                self.body_count,
                3,
                device=device,
                dtype=torch.float32,
            ),
            "sparse_body_force": torch.empty(
                batch_size,
                self.body_count,
                3,
                device=device,
                dtype=torch.float32,
            ),
            "sparse_body_torque": torch.empty(
                batch_size,
                self.body_count,
                3,
                device=device,
                dtype=torch.float32,
            ),
        }
        if not minimal_native_lbs:
            workspace["transformed_waypoints"] = torch.empty(
                batch_size,
                waypoint_count,
                3,
                device=device,
                dtype=torch.float32,
            )
        if not omit_passive_force_workspace:
            workspace["passive_force_magnitude"] = torch.empty(
                batch_size,
                self.muscle_count,
                device=device,
                dtype=torch.float32,
            )
        if (
            self._uses_attachment_projection
            and self.lbs_forward_reduction == "partial"
            and self.lbs_waypoint_attachment_fixed2
        ):
            workspace["xquat_normalized"] = torch.empty(
                batch_size,
                self.body_count,
                4,
                device=device,
                dtype=torch.float32,
            )
        if self._uses_attachment_projection and self.lbs_forward_reduction == "partial":
            if not minimal_native_lbs:
                workspace["lbs_muscle_length"] = torch.empty(
                    batch_size,
                    self.muscle_count,
                    device=device,
                    dtype=torch.float32,
                )
                workspace["lbs_force_by_muscle"] = torch.empty(
                    batch_size,
                    self.muscle_count,
                    device=device,
                    dtype=torch.float32,
                )
            if (
                self.lbs_attachment_layout == "body_wrench"
                and self.lbs_segment_start_is_segment_plus_muscle
                and not minimal_native_lbs
            ):
                workspace["lbs_segment_length_group_partials"] = torch.empty(
                    batch_size,
                    self.muscle_count * self.lbs_body_wrench_force_segment_group_width,
                    device=device,
                    dtype=torch.float32,
                )
            if self.lbs_attachment_layout == "waypoint_force":
                workspace["lbs_waypoint_force"] = torch.empty(
                    batch_size,
                    int(self.lbs_waypoint_count),
                    3,
                    device=device,
                    dtype=torch.float32,
                )
            if (
                self.lbs_attachment_layout == "waypoint_force"
                and self.lbs_active_attach_chunk_count > 0
            ):
                workspace["lbs_active_attach_chunk_partial"] = torch.empty(
                    batch_size,
                    self.lbs_active_attach_chunk_count,
                    device=device,
                    dtype=torch.float32,
                )
            if (
                self.lbs_attachment_layout == "body_wrench"
                and self.lbs_body_event_chunk_count > 0
            ):
                if (
                    self.lbs_body_wrench_forward_mode == "native_tile"
                    and self.lbs_native_body_tile_body_slot_count > 0
                ):
                    workspace["lbs_native_body_wrench_tile_partial"] = torch.empty(
                        batch_size,
                        self.lbs_native_body_tile_body_slot_count,
                        6,
                        device=device,
                        dtype=torch.float32,
                    )
                if not minimal_native_lbs:
                    workspace["lbs_body_wrench_chunk_partial"] = torch.empty(
                        batch_size,
                        self.lbs_body_event_chunk_count,
                        6,
                        device=device,
                        dtype=torch.float32,
                    )
                workspace["lbs_body_force"] = torch.empty(
                    batch_size,
                    self.body_count,
                    3,
                    device=device,
                    dtype=torch.float32,
                )
                workspace["lbs_body_torque"] = torch.empty(
                    batch_size,
                    self.body_count,
                    3,
                    device=device,
                    dtype=torch.float32,
                )
        elif (
            self.lbs_torque_chunk_count > 0 and self.lbs_forward_reduction == "partial"
        ):
            workspace["lbs_torque_chunk_partial"] = torch.empty(
                batch_size,
                self.lbs_torque_chunk_count,
                device=device,
                dtype=torch.float32,
            )
        self._warp_geometry_workspace[key] = workspace
        return workspace

    def _get_warp_lbs_force_cache_workspace(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        key = self._warp_workspace_key(
            device=device,
            batch_size=batch_size,
            workspace_kind="lbs_force_cache",
        )
        workspace = self._warp_geometry_workspace.get(key)
        if workspace is None:
            workspace = {
                "xquat_normalized": torch.empty(
                    batch_size,
                    self.body_count,
                    4,
                    device=device,
                    dtype=torch.float32,
                )
            }
            self._warp_geometry_workspace[key] = workspace
            return workspace
        if "xquat_normalized" not in workspace:
            workspace["xquat_normalized"] = torch.empty(
                batch_size,
                self.body_count,
                4,
                device=device,
                dtype=torch.float32,
            )
        return workspace

    def _prepare_warp_lbs_force_cache_geometry(
        self,
        raw_context: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        raw_context = raw_context.float()
        wp.init()
        batch_size = raw_context.shape[0]
        workspace = self._get_warp_lbs_force_cache_workspace(
            batch_size=batch_size,
            device=raw_context.device,
        )
        xpos = raw_context[:, : self._xpos_width].reshape(
            batch_size,
            self.body_count,
            3,
        )
        xquat = raw_context[
            :, self._xpos_width : self._xpos_width + self._xquat_width
        ].reshape(batch_size, self.body_count, 4)
        wp.launch(
            _normalize_body_quats_kernel,
            dim=(batch_size, self.body_count),
            inputs=[wp.from_torch(xquat, dtype=wp.quat)],
            outputs=[wp.from_torch(workspace["xquat_normalized"], dtype=wp.quat)],
            device=str(raw_context.device),
        )
        return workspace, xpos

    def _wrench_projection_topology(self, device: str):
        """Return device-resident tree metadata, or ``None`` for the legacy kernel."""
        cache = self.__dict__.setdefault("_wrench_projection_topologies", {})
        key = str(device)
        if key not in cache:
            if wp.get_device(key).is_capturing:
                raise RuntimeError(
                    "wrench projection topology must be prepared before graph capture"
                )
            cache[key] = build_wrench_projection_topology(
                self.body_parentid,
                self.dof_bodyid,
                self.body_count,
                device,
            )
        return cache[key]

    def _launch_project_body_wrench_to_torque(
        self,
        *,
        body_force: torch.Tensor,
        body_torque: torch.Tensor,
        xipos: torch.Tensor,
        subtree_com: torch.Tensor,
        cdof: torch.Tensor,
        mean: torch.Tensor,
        batch_size: int,
        device: str,
        output_dof_offset: int = 0,
    ) -> None:
        launch_project_body_wrench_to_torque(
            self._wrench_projection_topology(device),
            body_force=body_force,
            body_torque=body_torque,
            xipos=xipos,
            subtree_com=subtree_com,
            cdof=cdof,
            body_rootid=self.body_rootid_i32,
            dof_subtree_mask=self.dof_subtree_mask,
            mean=mean,
            body_count=self.body_count,
            torque_dim=self.torque_dim,
            root_dof_count=int(self.full_torque_dim - self.torque_dim),
            output_dof_offset=int(output_dof_offset),
            batch_size=batch_size,
            device=device,
        )

    def supports_native_body_wrench_full_force_forward(self) -> bool:
        """Return whether this layer owns a runnable native-tile forward kernel."""
        return bool(
            self.geometry_backend == "waypoint"
            and self._uses_native_body_wrench_projection
            and not self.muscle_force_velocity_enabled
            and self.lbs_native_body_tile_count > 0
            and self._lbs_native_body_wrench_full_force_kernels is not None
        )

    def prepare_native_body_wrench_full_force_workspace(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> bool:
        """Allocate the persistent workspace used by the native-tile launcher."""
        if not self.supports_native_body_wrench_full_force_forward():
            return False
        device = torch.device(device)
        if device.type != "cuda":
            return False
        workspace = self._get_warp_geometry_workspace(
            batch_size=int(batch_size),
            device=device,
            minimal_native_lbs=True,
        )
        self._reserve_native_wrap_workspace(workspace, int(batch_size), device)
        if "lbs_native_sorted_activation" not in workspace:
            workspace["lbs_native_sorted_activation"] = torch.empty(
                int(batch_size),
                self.muscle_count,
                device=device,
                dtype=torch.float32,
            )
        return True

    def _launch_native_body_wrench_full_force_into(
        self,
        *,
        workspace: dict[str, torch.Tensor],
        activation: torch.Tensor,
        body_pos: torch.Tensor,
        body_quat: torch.Tensor,
        xipos: torch.Tensor,
        subtree_com: torch.Tensor,
        cdof: torch.Tensor,
        output: torch.Tensor,
        include_passive_force: bool,
        output_dof_offset: int,
    ) -> None:
        kernels = self._lbs_native_body_wrench_full_force_kernels
        if kernels is None:
            raise RuntimeError("native-tile full-force kernel was not initialized")
        batch_size = int(activation.shape[0])
        device = str(activation.device)
        sorted_activation = self._prepare_native_tile_sorted_muscle_input(
            workspace,
            activation,
            key="lbs_native_sorted_activation",
        )
        self._prepare_native_wrap_geometry(workspace, body_pos, body_quat)
        partial = workspace["lbs_native_body_wrench_tile_partial"]
        wp.launch_tiled(
            kernels[1 if include_passive_force else 0],
            dim=(batch_size, self.lbs_native_body_tile_count),
            inputs=[
                wp.from_torch(body_pos, dtype=wp.vec3),
                *self._native_wrap_inputs(workspace, batch_size),
                wp.from_torch(
                    workspace["xquat_normalized"]
                    if self.lbs_waypoint_attachment_fixed2
                    else body_quat,
                    dtype=wp.quat,
                ),
                wp.from_torch(xipos, dtype=wp.vec3),
                wp.from_torch(
                    self.lbs_native_muscle_segment_offsets,
                    dtype=wp.int32,
                ),
                wp.from_torch(
                    self.lbs_native_muscle_force_length_coeff,
                    dtype=wp.vec4,
                ),
                wp.from_torch(
                    sorted_activation,
                    dtype=wp.float32,
                    requires_grad=False,
                ),
                self.muscle_length_norm_min,
                self.muscle_length_norm_max,
                wp.from_torch(self.lbs_native_body_tile_info, dtype=wp.vec4i),
                wp.from_torch(self.lbs_native_body_tile_extra_info, dtype=wp.vec4i),
                wp.from_torch(self.lbs_native_body_tile_body_ids, dtype=wp.int32),
                wp.from_torch(
                    self.lbs_native_body_tile_body_output_idx, dtype=wp.int32
                ),
                wp.from_torch(
                    self.lbs_native_body_tile_event_start_slot_muscle,
                    dtype=wp.int32,
                ),
                wp.from_torch(
                    self.lbs_native_body_tile_event_local_force_coeff,
                    dtype=wp.vec4,
                ),
                wp.from_torch(
                    self.lbs_native_waypoint_attachment_body2,
                    dtype=wp.vec2i,
                ),
                wp.from_torch(
                    self.lbs_native_waypoint_attachment_local_weight0,
                    dtype=wp.vec4,
                ),
                wp.from_torch(
                    self.lbs_native_waypoint_attachment_local_weight1,
                    dtype=wp.vec4,
                ),
            ],
            outputs=[wp.from_torch(partial, dtype=wp.float32)],
            device=device,
            block_dim=self.lbs_forward_tile_block_dim,
        )
        self._project_native_tile_partials_into(
            workspace,
            xipos=xipos,
            subtree_com=subtree_com,
            cdof=cdof,
            output=output,
            batch_size=batch_size,
            device=device,
            output_dof_offset=output_dof_offset,
        )

    def _project_native_tile_partials_into(
        self,
        workspace: dict[str, torch.Tensor],
        *,
        xipos: torch.Tensor,
        subtree_com: torch.Tensor,
        cdof: torch.Tensor,
        output: torch.Tensor,
        batch_size: int,
        device: str,
        output_dof_offset: int,
    ) -> None:
        """Reduce the native tiles' per-slot wrenches and project them to DOFs."""
        wp.launch(
            _reduce_lbs_body_wrench_event_partials_kernel,
            dim=(batch_size, self.body_count),
            inputs=[
                wp.from_torch(
                    workspace["lbs_native_body_wrench_tile_partial"],
                    dtype=wp.float32,
                ),
                wp.from_torch(
                    self.lbs_native_body_tile_body_slot_offsets,
                    dtype=wp.int32,
                ),
            ],
            outputs=[
                wp.from_torch(workspace["lbs_body_force"], dtype=wp.float32),
                wp.from_torch(workspace["lbs_body_torque"], dtype=wp.float32),
            ],
            device=device,
        )
        self._launch_project_body_wrench_to_torque(
            body_force=workspace["lbs_body_force"],
            body_torque=workspace["lbs_body_torque"],
            xipos=xipos,
            subtree_com=subtree_com,
            cdof=cdof,
            mean=output,
            batch_size=batch_size,
            device=device,
            output_dof_offset=int(output_dof_offset),
        )

    def launch_native_body_wrench_full_force_into(
        self,
        *,
        activation: torch.Tensor,
        body_pos: torch.Tensor,
        body_quat: torch.Tensor,
        xipos: torch.Tensor,
        subtree_com: torch.Tensor,
        cdof: torch.Tensor,
        output: torch.Tensor,
        include_passive_force: bool,
        output_dof_offset: int = 0,
    ) -> None:
        """Launch the shared allocation-free native-tile force projection path."""
        if not self.supports_native_body_wrench_full_force_forward():
            raise RuntimeError("native-tile full-force forward is unavailable")
        if activation.dtype != torch.float32 or activation.device.type != "cuda":
            raise ValueError("native-tile activation must be CUDA float32")
        batch_size = int(activation.shape[0])
        workspace = self._get_warp_geometry_workspace(
            batch_size=batch_size,
            device=activation.device,
            minimal_native_lbs=True,
        )
        if self.lbs_waypoint_attachment_fixed2:
            normalized_body_quat = workspace["xquat_normalized"]
            wp.launch(
                _normalize_body_quats_kernel,
                dim=(batch_size, self.body_count),
                inputs=[wp.from_torch(body_quat, dtype=wp.quat)],
                outputs=[wp.from_torch(normalized_body_quat, dtype=wp.quat)],
                device=str(activation.device),
            )
        self._launch_native_body_wrench_full_force_into(
            workspace=workspace,
            activation=activation,
            body_pos=body_pos,
            body_quat=body_quat,
            xipos=xipos,
            subtree_com=subtree_com,
            cdof=cdof,
            output=output,
            include_passive_force=include_passive_force,
            output_dof_offset=int(output_dof_offset),
        )

    def supports_native_body_wrench_full_force_velocity_forward(self) -> bool:
        """Return whether the native tiles can evaluate explicit force-velocity."""
        return bool(
            self.geometry_backend == "waypoint"
            and self._uses_native_body_wrench_projection
            and self.lbs_native_body_tile_count > 0
            and self._lbs_native_full_force_velocity_factory_args is not None
        )

    def prepare_native_body_wrench_full_force_velocity_workspace(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> bool:
        """Build the force-velocity native-tile kernels and their workspace."""
        if not self.supports_native_body_wrench_full_force_velocity_forward():
            return False
        if self.body_count > 65536:
            # The tile keeps each point's body id in 16 bits.
            raise ValueError(
                "native-tile force-velocity supports at most 65536 bodies, got "
                f"{self.body_count}"
            )
        device = torch.device(device)
        if device.type != "cuda":
            return False
        if self._lbs_native_body_wrench_full_force_velocity_kernels is None:
            active_force_model = tuple(
                float(self._muscle_active_force_model_params_wp[i]) for i in range(3)
            )
            self._lbs_native_body_wrench_full_force_velocity_kernels = tuple(
                make_accumulate_lbs_body_wrench_native_tile_full_force_velocity_fixed1_kernel(
                    include_passive,
                    *self._lbs_native_full_force_velocity_factory_args,
                    wrapped=bool(self._waypoint_wrapping.count),
                    active_force_model=active_force_model,
                )
                for include_passive in (False, True)
            )
        workspace = self._get_warp_geometry_workspace(
            batch_size=int(batch_size),
            device=device,
            minimal_native_lbs=True,
        )
        self._reserve_native_wrap_workspace(workspace, int(batch_size), device)
        return True

    def launch_native_body_wrench_full_force_velocity_into(
        self,
        *,
        activation: torch.Tensor,
        body_pos: torch.Tensor,
        body_quat: torch.Tensor,
        xipos: torch.Tensor,
        subtree_com: torch.Tensor,
        cdof: torch.Tensor,
        body_cvel: wp.array,
        body_rootid: torch.Tensor,
        velocity_scale: torch.Tensor,
        fvmax: float,
        clamp_activation: bool,
        output: torch.Tensor,
        include_passive_force: bool,
        output_dof_offset: int = 0,
    ) -> None:
        """Explicit force-velocity muscle force projected by the native tiles.

        ``activation`` and ``velocity_scale`` are in model muscle order; the
        path velocity comes from ``body_cvel`` (COM-based body velocities),
        so no moment arms are formed.
        """
        if self._lbs_native_body_wrench_full_force_velocity_kernels is None:
            raise RuntimeError(
                "native-tile force-velocity kernels were not prepared; call "
                "prepare_native_body_wrench_full_force_velocity_workspace first"
            )
        if activation.dtype != torch.float32 or activation.device.type != "cuda":
            raise ValueError("native-tile activation must be CUDA float32")
        batch_size = int(activation.shape[0])
        device = str(activation.device)
        workspace = self._get_warp_geometry_workspace(
            batch_size=batch_size,
            device=activation.device,
            minimal_native_lbs=True,
        )
        kernel_body_quat = body_quat
        if self.lbs_waypoint_attachment_fixed2:
            kernel_body_quat = workspace["xquat_normalized"]
            wp.launch(
                _normalize_body_quats_kernel,
                dim=(batch_size, self.body_count),
                inputs=[wp.from_torch(body_quat, dtype=wp.quat)],
                outputs=[wp.from_torch(kernel_body_quat, dtype=wp.quat)],
                device=device,
            )
        partial = workspace["lbs_native_body_wrench_tile_partial"]
        self._prepare_native_wrap_geometry(workspace, body_pos, body_quat)
        wp.launch_tiled(
            self._lbs_native_body_wrench_full_force_velocity_kernels[
                1 if include_passive_force else 0
            ],
            dim=(batch_size, self.lbs_native_body_tile_count),
            inputs=[
                wp.from_torch(body_pos, dtype=wp.vec3),
                *self._native_wrap_inputs(workspace, batch_size),
                wp.from_torch(kernel_body_quat, dtype=wp.quat),
                wp.from_torch(xipos, dtype=wp.vec3),
                wp.from_torch(self.lbs_native_muscle_segment_offsets, dtype=wp.int32),
                wp.from_torch(self.lbs_native_muscle_force_length_coeff, dtype=wp.vec4),
                wp.from_torch(activation, dtype=wp.float32, requires_grad=False),
                self.muscle_length_norm_min,
                self.muscle_length_norm_max,
                wp.from_torch(self.lbs_native_body_tile_info, dtype=wp.vec4i),
                wp.from_torch(self.lbs_native_body_tile_extra_info, dtype=wp.vec4i),
                wp.from_torch(self.lbs_native_body_tile_body_ids, dtype=wp.int32),
                wp.from_torch(
                    self.lbs_native_body_tile_body_output_idx, dtype=wp.int32
                ),
                wp.from_torch(
                    self.lbs_native_body_tile_event_start_slot_muscle,
                    dtype=wp.int32,
                ),
                wp.from_torch(
                    self.lbs_native_body_tile_event_local_force_coeff,
                    dtype=wp.vec4,
                ),
                wp.from_torch(
                    self.lbs_native_waypoint_attachment_body2, dtype=wp.vec2i
                ),
                wp.from_torch(
                    self.lbs_native_waypoint_attachment_local_weight0, dtype=wp.vec4
                ),
                wp.from_torch(
                    self.lbs_native_waypoint_attachment_local_weight1, dtype=wp.vec4
                ),
                wp.from_torch(self.lbs_native_muscle_order_i32, dtype=wp.int32),
                body_cvel,
                wp.from_torch(subtree_com, dtype=wp.vec3),
                wp.from_torch(body_rootid, dtype=wp.int32),
                wp.from_torch(velocity_scale, dtype=wp.float32),
                float(fvmax),
                1 if clamp_activation else 0,
            ],
            outputs=[wp.from_torch(partial, dtype=wp.float32)],
            device=device,
            block_dim=self.lbs_forward_tile_block_dim,
        )
        self._project_native_tile_partials_into(
            workspace,
            xipos=xipos,
            subtree_com=subtree_com,
            cdof=cdof,
            output=output,
            batch_size=batch_size,
            device=device,
            output_dof_offset=output_dof_offset,
        )

    def _get_warp_torque_jacobian_workspace(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        key = self._warp_workspace_key(device=device, batch_size=batch_size)
        workspace = self._warp_torque_jacobian_workspace.get(key)
        if workspace is not None:
            return workspace

        workspace = {
            "contribution": torch.empty(
                batch_size,
                self.torque_dim,
                self.muscle_count,
                device=device,
                dtype=torch.float32,
            ),
        }
        if self.active_torque_indices is not None:
            workspace["active_contribution"] = torch.empty(
                batch_size,
                self.active_output_dim,
                self.muscle_count,
                device=device,
                dtype=torch.float32,
            )
        self._warp_torque_jacobian_workspace[key] = workspace
        return workspace

    def _get_warp_torque_bounds_workspace(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        key = self._warp_workspace_key(device=device, batch_size=batch_size)
        workspace = self._warp_torque_bounds_workspace.get(key)
        if workspace is not None:
            if (
                self.uses_lbs_waypoint_attachments
                and self.lbs_forward_reduction == "partial"
                and self.lbs_bounds_chunk_count > 0
                and "partial_lower" not in workspace
            ):
                workspace["partial_lower"] = torch.empty(
                    batch_size,
                    self.lbs_bounds_chunk_count,
                    device=device,
                    dtype=torch.float32,
                )
                workspace["partial_upper"] = torch.empty(
                    batch_size,
                    self.lbs_bounds_chunk_count,
                    device=device,
                    dtype=torch.float32,
                )
            return workspace

        workspace = {
            "lower": torch.empty(
                batch_size,
                self.torque_dim,
                device=device,
                dtype=torch.float32,
            ),
            "upper": torch.empty(
                batch_size,
                self.torque_dim,
                device=device,
                dtype=torch.float32,
            ),
        }
        if self.active_torque_indices is not None:
            workspace["active_lower"] = torch.empty(
                batch_size,
                self.active_output_dim,
                device=device,
                dtype=torch.float32,
            )
            workspace["active_upper"] = torch.empty(
                batch_size,
                self.active_output_dim,
                device=device,
                dtype=torch.float32,
            )
        if (
            self.uses_lbs_waypoint_attachments
            and self.lbs_forward_reduction == "partial"
            and self.lbs_bounds_chunk_count > 0
        ):
            workspace["partial_lower"] = torch.empty(
                batch_size,
                self.lbs_bounds_chunk_count,
                device=device,
                dtype=torch.float32,
            )
            workspace["partial_upper"] = torch.empty(
                batch_size,
                self.lbs_bounds_chunk_count,
                device=device,
                dtype=torch.float32,
            )
        self._warp_torque_bounds_workspace[key] = workspace
        return workspace

    def _prepare_warp_waypoint_geometry(
        self,
        raw_context: torch.Tensor,
        *,
        include_passive_force_magnitude: bool,
        compute_force_magnitudes: bool = True,
        skip_lbs_waypoint_transform: bool = False,
        apply_force_velocity: bool = True,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        raw_context = raw_context.float()
        wp.init()
        batch_size = raw_context.shape[0]
        # Native tiles transform their body-fixed waypoints directly.
        skip_lbs_waypoint_transform = bool(skip_lbs_waypoint_transform)
        if compute_force_magnitudes or not skip_lbs_waypoint_transform:
            # This call rewrites the transformed points or force magnitudes a
            # forward may have saved for its backward.
            self._geometry_scratch_generation += 1
        workspace = self._get_warp_geometry_workspace(
            batch_size=batch_size,
            device=raw_context.device,
            minimal_native_lbs=(
                skip_lbs_waypoint_transform and self._uses_native_body_wrench_projection
            ),
        )

        xpos = raw_context[:, : self._xpos_width].reshape(
            batch_size, self.body_count, 3
        )
        xquat = raw_context[
            :, self._xpos_width : self._xpos_width + self._xquat_width
        ].reshape(batch_size, self.body_count, 4)
        offset = self._xpos_width + self._xquat_width + self._xipos_width
        subtree_com = raw_context[:, offset : offset + self._subtree_com_width].reshape(
            batch_size, self.body_count, 3
        )
        offset += self._subtree_com_width
        cdof = raw_context[:, offset : offset + self._cdof_width].reshape(
            batch_size, self.full_torque_dim, 6
        )

        xipos = raw_context[
            :,
            self._xpos_width + self._xquat_width : self._xpos_width
            + self._xquat_width
            + self._xipos_width,
        ].reshape(batch_size, self.body_count, 3)
        workspace["xpos"] = xpos
        workspace["xquat"] = xquat
        workspace["xipos"] = xipos
        device = str(raw_context.device)
        if self._uses_attachment_projection and self.lbs_waypoint_attachment_fixed2:
            wp.launch(
                _normalize_body_quats_kernel,
                dim=(batch_size, self.body_count),
                inputs=[wp.from_torch(xquat, dtype=wp.quat)],
                outputs=[wp.from_torch(workspace["xquat_normalized"], dtype=wp.quat)],
                device=device,
            )
        if not skip_lbs_waypoint_transform:
            self._transform_warp_waypoints(workspace, xpos, xquat)
        wrapped_state = None
        if self._waypoint_wrapping.count and not skip_lbs_waypoint_transform:
            wrapped_state = self._prepare_wrapped_points(
                workspace["transformed_waypoints"], xpos, xquat
            )
        if compute_force_magnitudes:
            if "passive_force_magnitude" not in workspace:
                workspace["passive_force_magnitude"] = torch.empty(
                    batch_size,
                    self.muscle_count,
                    device=raw_context.device,
                    dtype=torch.float32,
                )
            if wrapped_state is not None:
                wp.launch(
                    force_components_kernel,
                    dim=(batch_size, self.muscle_count),
                    inputs=[
                        wrapped_state.lengths,
                        wp.from_torch(self.muscle_params),
                        wp.from_torch(self.muscle_rest_length),
                        self.muscle_force_scale,
                        self.muscle_length_norm_min,
                        self.muscle_length_norm_max,
                        1.0 if include_passive_force_magnitude else 0.0,
                        self._muscle_active_force_model_params_wp,
                        self._muscle_passive_force_model_params_wp,
                    ],
                    outputs=[
                        wp.from_torch(workspace["unit_force_magnitude"]),
                        wp.from_torch(workspace["passive_force_magnitude"]),
                    ],
                    device=device,
                )
            elif (
                not self.uses_lbs_waypoint_attachments
                and self.all_segment_start_is_segment_plus_muscle
            ):
                wp.launch(
                    _compute_force_magnitudes_contiguous_waypoints_kernel,
                    dim=(batch_size, self.muscle_count),
                    inputs=[
                        wp.from_torch(
                            workspace["transformed_waypoints"], dtype=wp.vec3
                        ),
                        wp.from_torch(self.all_muscle_segment_offsets, dtype=wp.int32),
                        wp.from_torch(self.muscle_params, dtype=wp.float32),
                        wp.from_torch(self.muscle_rest_length, dtype=wp.float32),
                        self.muscle_force_scale,
                        self.muscle_length_norm_min,
                        self.muscle_length_norm_max,
                        1.0 if include_passive_force_magnitude else 0.0,
                        self._muscle_active_force_model_params_wp,
                        self._muscle_passive_force_model_params_wp,
                    ],
                    outputs=[
                        wp.from_torch(
                            workspace["unit_force_magnitude"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            workspace["passive_force_magnitude"], dtype=wp.float32
                        ),
                    ],
                    device=device,
                )
            else:
                wp.launch(
                    _compute_force_magnitudes_from_waypoints_kernel,
                    dim=(batch_size, self.muscle_count),
                    inputs=[
                        wp.from_torch(
                            workspace["transformed_waypoints"], dtype=wp.vec3
                        ),
                        wp.from_torch(self.all_segment_start_idx_i32, dtype=wp.int32),
                        wp.from_torch(self.all_muscle_segment_offsets, dtype=wp.int32),
                        wp.from_torch(self.muscle_params, dtype=wp.float32),
                        wp.from_torch(self.muscle_rest_length, dtype=wp.float32),
                        self.muscle_force_scale,
                        self.muscle_length_norm_min,
                        self.muscle_length_norm_max,
                        1.0 if include_passive_force_magnitude else 0.0,
                        self._muscle_active_force_model_params_wp,
                        self._muscle_passive_force_model_params_wp,
                    ],
                    outputs=[
                        wp.from_torch(
                            workspace["unit_force_magnitude"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            workspace["passive_force_magnitude"], dtype=wp.float32
                        ),
                    ],
                    device=device,
                )
        if (
            compute_force_magnitudes
            and apply_force_velocity
            and self.muscle_force_velocity_enabled
        ):
            if self.uses_lbs_waypoint_attachments:
                wp.launch(
                    apply_lbs_muscle_force_velocity_to_unit_force_kernel,
                    dim=(batch_size, self.muscle_count),
                    inputs=[
                        wp.from_torch(
                            workspace["transformed_waypoints"], dtype=wp.vec3
                        ),
                        wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                        wp.from_torch(workspace["xquat"], dtype=wp.quat),
                        wp.from_torch(subtree_com, dtype=wp.vec3),
                        wp.from_torch(cdof, dtype=wp.float32),
                        wp.from_torch(raw_context, dtype=wp.float32),
                        wp.from_torch(self.muscle_active_entry_offsets, dtype=wp.int32),
                        wp.from_torch(self.muscle_active_segment_idx, dtype=wp.int32),
                        wp.from_torch(self.muscle_active_torque_idx, dtype=wp.int32),
                        wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                        wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                        wp.from_torch(self.body_rootid_i32, dtype=wp.int32),
                        wp.from_torch(self.lbs_active_attach_offsets, dtype=wp.int32),
                        wp.from_torch(self.lbs_active_attach_body, dtype=wp.int32),
                        wp.from_torch(self.lbs_active_attach_local, dtype=wp.vec3),
                        wp.from_torch(
                            self.lbs_active_attach_coeff_weight, dtype=wp.float32
                        ),
                        wp.from_torch(
                            self.muscle_force_velocity_length_scale,
                            dtype=wp.float32,
                        ),
                        int(self.lbs_waypoint_count),
                        int(self.full_torque_dim - self.torque_dim),
                        int(self._geometry_context_dim),
                        self.muscle_force_velocity_vmax,
                        self.muscle_force_velocity_fvmax,
                    ],
                    outputs=[
                        wp.from_torch(
                            workspace["unit_force_magnitude"], dtype=wp.float32
                        )
                    ],
                    device=device,
                )
            else:
                wp.launch(
                    _apply_muscle_force_velocity_to_unit_force_kernel,
                    dim=(batch_size, self.muscle_count),
                    inputs=[
                        wp.from_torch(
                            workspace["transformed_waypoints"], dtype=wp.vec3
                        ),
                        wp.from_torch(subtree_com, dtype=wp.vec3),
                        wp.from_torch(cdof, dtype=wp.float32),
                        wp.from_torch(raw_context, dtype=wp.float32),
                        wp.from_torch(self.muscle_active_entry_offsets, dtype=wp.int32),
                        wp.from_torch(self.muscle_active_segment_idx, dtype=wp.int32),
                        wp.from_torch(self.muscle_active_torque_idx, dtype=wp.int32),
                        wp.from_torch(self.muscle_active_start_coeff, dtype=wp.float32),
                        wp.from_torch(self.muscle_active_end_coeff, dtype=wp.float32),
                        wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                        wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                        wp.from_torch(
                            self.segment_start_root_body_idx_i32, dtype=wp.int32
                        ),
                        wp.from_torch(
                            self.segment_end_root_body_idx_i32, dtype=wp.int32
                        ),
                        wp.from_torch(self.muscle_params, dtype=wp.float32),
                        wp.from_torch(self.muscle_rest_length, dtype=wp.float32),
                        wp.from_torch(
                            self.muscle_force_velocity_length_scale,
                            dtype=wp.float32,
                        ),
                        int(self.full_torque_dim - self.torque_dim),
                        int(self._geometry_context_dim),
                        self.muscle_force_velocity_vmax,
                        self.muscle_force_velocity_fvmax,
                    ],
                    outputs=[
                        wp.from_torch(
                            workspace["unit_force_magnitude"], dtype=wp.float32
                        )
                    ],
                    device=device,
                )
        return workspace, subtree_com, cdof

    def _compute_lbs_muscle_dynamics_bounds_warp(
        self,
        raw_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_context = raw_context.float()
        batch_size = raw_context.shape[0]
        geometry_workspace, subtree_com, cdof = self._prepare_warp_waypoint_geometry(
            raw_context,
            include_passive_force_magnitude=False,
            apply_force_velocity=True,
        )
        bounds_workspace = self._get_warp_torque_bounds_workspace(
            batch_size=batch_size,
            device=raw_context.device,
        )
        lower = bounds_workspace["lower"]
        upper = bounds_workspace["upper"]
        if self.lbs_forward_reduction == "partial" and self.lbs_bounds_chunk_count > 0:
            partial_lower = bounds_workspace["partial_lower"]
            partial_upper = bounds_workspace["partial_upper"]
            wp.launch_tiled(
                _compute_lbs_torque_bounds_tile_partials_kernel,
                dim=(batch_size, self.lbs_bounds_chunk_count),
                inputs=[
                    wp.from_torch(
                        geometry_workspace["transformed_waypoints"], dtype=wp.vec3
                    ),
                    wp.from_torch(geometry_workspace["xpos"], dtype=wp.vec3),
                    wp.from_torch(geometry_workspace["xquat"], dtype=wp.quat),
                    wp.from_torch(
                        geometry_workspace["unit_force_magnitude"], dtype=wp.float32
                    ),
                    wp.from_torch(subtree_com, dtype=wp.vec3),
                    wp.from_torch(cdof, dtype=wp.float32),
                    wp.from_torch(self.body_rootid_i32, dtype=wp.int32),
                    wp.from_torch(self.lbs_bounds_chunk_torque_idx, dtype=wp.int32),
                    wp.from_torch(self.lbs_bounds_chunk_group_start, dtype=wp.int32),
                    wp.from_torch(self.lbs_bounds_chunk_group_end, dtype=wp.int32),
                    wp.from_torch(self.lbs_bounds_group_entry_start, dtype=wp.int32),
                    wp.from_torch(self.lbs_bounds_group_entry_end, dtype=wp.int32),
                    wp.from_torch(self.lbs_bounds_forward_segment_idx, dtype=wp.int32),
                    wp.from_torch(self.lbs_bounds_forward_muscle_idx, dtype=wp.int32),
                    wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                    wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                    wp.from_torch(self.lbs_active_attach_offsets, dtype=wp.int32),
                    wp.from_torch(self.lbs_active_attach_body, dtype=wp.int32),
                    wp.from_torch(self.lbs_active_attach_local, dtype=wp.vec3),
                    wp.from_torch(
                        self.lbs_active_attach_coeff_weight, dtype=wp.float32
                    ),
                    int(self.lbs_waypoint_count),
                    int(self.root_dof_count),
                ],
                outputs=[
                    wp.from_torch(partial_lower, dtype=wp.float32),
                    wp.from_torch(partial_upper, dtype=wp.float32),
                ],
                device=str(raw_context.device),
                block_dim=self.lbs_forward_tile_block_dim,
            )
            wp.launch(
                _reduce_lbs_torque_bounds_chunk_partials_kernel,
                dim=(batch_size, self.torque_dim),
                inputs=[
                    wp.from_torch(partial_lower, dtype=wp.float32),
                    wp.from_torch(partial_upper, dtype=wp.float32),
                    wp.from_torch(self.lbs_bounds_chunk_offsets, dtype=wp.int32),
                ],
                outputs=[
                    wp.from_torch(lower, dtype=wp.float32),
                    wp.from_torch(upper, dtype=wp.float32),
                ],
                device=str(raw_context.device),
            )
            return lower, upper
        wp.launch(
            _compute_muscle_dynamics_lbs_sparse_bounds_kernel,
            dim=(batch_size, self.torque_dim),
            inputs=[
                wp.from_torch(
                    geometry_workspace["transformed_waypoints"], dtype=wp.vec3
                ),
                wp.from_torch(geometry_workspace["xpos"], dtype=wp.vec3),
                wp.from_torch(geometry_workspace["xquat"], dtype=wp.quat),
                wp.from_torch(
                    geometry_workspace["unit_force_magnitude"], dtype=wp.float32
                ),
                wp.from_torch(subtree_com, dtype=wp.vec3),
                wp.from_torch(cdof, dtype=wp.float32),
                wp.from_torch(self.body_rootid_i32, dtype=wp.int32),
                wp.from_torch(self.dof_subtree_mask, dtype=wp.float32),
                wp.from_torch(self.torque_active_entry_offsets, dtype=wp.int32),
                wp.from_torch(self.torque_active_segment_idx, dtype=wp.int32),
                wp.from_torch(self.torque_active_muscle_idx, dtype=wp.int32),
                wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                wp.from_torch(
                    self.muscle_waypoint_attachment_start_i32, dtype=wp.int32
                ),
                wp.from_torch(self.muscle_waypoint_attachment_body_i32, dtype=wp.int32),
                wp.from_torch(self.muscle_waypoint_attachment_local, dtype=wp.vec3),
                wp.from_torch(self.muscle_waypoint_attachment_weight, dtype=wp.float32),
                wp.from_torch(self.lbs_active_attach_offsets, dtype=wp.int32),
                wp.from_torch(self.lbs_active_attach_body, dtype=wp.int32),
                wp.from_torch(self.lbs_active_attach_local, dtype=wp.vec3),
                wp.from_torch(self.lbs_active_attach_coeff_weight, dtype=wp.float32),
                int(self.lbs_waypoint_count),
                int(self.root_dof_count),
            ],
            outputs=[
                wp.from_torch(lower, dtype=wp.float32),
                wp.from_torch(upper, dtype=wp.float32),
            ],
            device=str(raw_context.device),
        )
        return lower, upper

    def _compute_muscle_dynamics_bounds_warp(
        self,
        raw_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_context = raw_context.float()
        batch_size = raw_context.shape[0]
        geometry_workspace, subtree_com, cdof = self._prepare_warp_waypoint_geometry(
            raw_context,
            include_passive_force_magnitude=False,
            apply_force_velocity=True,
        )
        bounds_workspace = self._get_warp_torque_bounds_workspace(
            batch_size=batch_size,
            device=raw_context.device,
        )
        lower = bounds_workspace["lower"]
        upper = bounds_workspace["upper"]
        wp.launch(
            _compute_muscle_dynamics_sparse_bounds_kernel,
            dim=(batch_size, self.torque_dim),
            inputs=[
                wp.from_torch(
                    geometry_workspace["transformed_waypoints"], dtype=wp.vec3
                ),
                wp.from_torch(
                    geometry_workspace["unit_force_magnitude"], dtype=wp.float32
                ),
                wp.from_torch(subtree_com, dtype=wp.vec3),
                wp.from_torch(cdof, dtype=wp.float32),
                wp.from_torch(self.torque_active_entry_offsets, dtype=wp.int32),
                wp.from_torch(self.torque_active_segment_idx, dtype=wp.int32),
                wp.from_torch(self.torque_active_muscle_idx, dtype=wp.int32),
                wp.from_torch(self.torque_active_start_coeff, dtype=wp.float32),
                wp.from_torch(self.torque_active_end_coeff, dtype=wp.float32),
                wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                wp.from_torch(self.segment_start_root_body_idx_i32, dtype=wp.int32),
                wp.from_torch(self.segment_end_root_body_idx_i32, dtype=wp.int32),
                int(self.root_dof_count),
            ],
            outputs=[
                wp.from_torch(lower, dtype=wp.float32),
                wp.from_torch(upper, dtype=wp.float32),
            ],
            device=str(raw_context.device),
        )
        return lower, upper

    def _reduce_activation_interval_bounds_warp(
        self,
        contribution: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, output_dim, _ = contribution.shape
        workspace = self._get_warp_torque_bounds_workspace(
            batch_size=batch_size,
            device=contribution.device,
        )
        lower = workspace["lower"][:, :output_dim]
        upper = workspace["upper"][:, :output_dim]
        wp.launch(
            _reduce_activation_interval_bounds_kernel,
            dim=(batch_size, output_dim),
            inputs=[
                wp.from_torch(contribution, dtype=wp.float32),
                self.muscle_count,
            ],
            outputs=[
                wp.from_torch(lower, dtype=wp.float32),
                wp.from_torch(upper, dtype=wp.float32),
            ],
            device=str(contribution.device),
        )
        return lower, upper

    def _select_activation_interval_bounds_warp(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.active_torque_indices is None:
            return lower, upper
        workspace = self._get_warp_torque_bounds_workspace(
            batch_size=int(lower.shape[0]),
            device=lower.device,
        )
        active_lower = workspace["active_lower"]
        active_upper = workspace["active_upper"]
        wp.launch(
            _select_activation_interval_bounds_kernel,
            dim=(int(lower.shape[0]), self.active_output_dim),
            inputs=[
                wp.from_torch(lower, dtype=wp.float32),
                wp.from_torch(upper, dtype=wp.float32),
                wp.from_torch(self.active_torque_indices_i32, dtype=wp.int32),
            ],
            outputs=[
                wp.from_torch(active_lower, dtype=wp.float32),
                wp.from_torch(active_upper, dtype=wp.float32),
            ],
            device=str(lower.device),
        )
        return active_lower, active_upper

    def _compute_activation_torque_bounds_warp(
        self,
        raw_context: torch.Tensor,
        *,
        select_active: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.uses_lbs_waypoint_attachments:
            lower, upper = self._compute_lbs_muscle_dynamics_bounds_warp(raw_context)
        else:
            lower, upper = self._compute_muscle_dynamics_bounds_warp(raw_context)
        if select_active:
            return self._select_activation_interval_bounds_warp(lower, upper)
        return lower, upper

    def _unpack_warp_muscle_dynamics_context(
        self,
        raw_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_context = raw_context.float()
        batch_size = raw_context.shape[0]
        offset = self._xpos_width + self._xquat_width + self._xipos_width
        subtree_com = raw_context[:, offset : offset + self._subtree_com_width].reshape(
            batch_size, self.body_count, 3
        )
        offset += self._subtree_com_width
        cdof = raw_context[:, offset : offset + self._cdof_width].reshape(
            batch_size, self.full_torque_dim, 6
        )
        return subtree_com, cdof

    def _compute_full_muscle_dynamics_contribution_warp(
        self,
        raw_context: torch.Tensor,
    ) -> torch.Tensor:
        """Per-muscle generalized force at unit activation, (B, torque_dim, M)."""
        raw_context = raw_context.float()
        batch_size = raw_context.shape[0]
        geometry_workspace, subtree_com, cdof = self._prepare_warp_waypoint_geometry(
            raw_context,
            include_passive_force_magnitude=False,
            compute_force_magnitudes=True,
        )
        unit_force_magnitude = geometry_workspace["unit_force_magnitude"]
        jacobian_workspace = self._get_warp_torque_jacobian_workspace(
            batch_size=batch_size,
            device=raw_context.device,
        )
        device = str(raw_context.device)
        contribution = jacobian_workspace["contribution"]
        if self.uses_lbs_waypoint_attachments:
            contribution.zero_()
            if self.torque_active_torque_idx.numel() > 0:
                wp.launch(
                    _compute_muscle_dynamics_contribution_lbs_sparse_kernel,
                    dim=(batch_size, int(self.torque_active_torque_idx.numel())),
                    inputs=[
                        wp.from_torch(
                            geometry_workspace["transformed_waypoints"], dtype=wp.vec3
                        ),
                        wp.from_torch(geometry_workspace["xpos"], dtype=wp.vec3),
                        wp.from_torch(geometry_workspace["xquat"], dtype=wp.quat),
                        wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                        wp.from_torch(subtree_com, dtype=wp.vec3),
                        wp.from_torch(cdof, dtype=wp.float32),
                        wp.from_torch(self.body_rootid_i32, dtype=wp.int32),
                        wp.from_torch(self.dof_subtree_mask, dtype=wp.float32),
                        wp.from_torch(self.torque_active_torque_idx, dtype=wp.int32),
                        wp.from_torch(self.torque_active_segment_idx, dtype=wp.int32),
                        wp.from_torch(self.torque_active_muscle_idx, dtype=wp.int32),
                        wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                        wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_start_i32, dtype=wp.int32
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_body_i32, dtype=wp.int32
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_local, dtype=wp.vec3
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_weight, dtype=wp.float32
                        ),
                        wp.from_torch(self.lbs_active_attach_offsets, dtype=wp.int32),
                        wp.from_torch(self.lbs_active_attach_body, dtype=wp.int32),
                        wp.from_torch(self.lbs_active_attach_local, dtype=wp.vec3),
                        wp.from_torch(
                            self.lbs_active_attach_coeff_weight, dtype=wp.float32
                        ),
                        int(self.lbs_waypoint_count),
                        int(self.root_dof_count),
                    ],
                    outputs=[wp.from_torch(contribution, dtype=wp.float32)],
                    device=device,
                )
            return contribution
        if self.torque_active_torque_idx.numel() > 0:
            contribution.zero_()
            wp.launch(
                _compute_muscle_dynamics_contribution_sparse_kernel,
                dim=(batch_size, int(self.torque_active_torque_idx.numel())),
                inputs=[
                    wp.from_torch(
                        geometry_workspace["transformed_waypoints"], dtype=wp.vec3
                    ),
                    wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                    wp.from_torch(subtree_com, dtype=wp.vec3),
                    wp.from_torch(cdof, dtype=wp.float32),
                    wp.from_torch(self.torque_active_torque_idx, dtype=wp.int32),
                    wp.from_torch(self.torque_active_segment_idx, dtype=wp.int32),
                    wp.from_torch(self.torque_active_muscle_idx, dtype=wp.int32),
                    wp.from_torch(self.torque_active_start_coeff, dtype=wp.float32),
                    wp.from_torch(self.torque_active_end_coeff, dtype=wp.float32),
                    wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                    wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                    wp.from_torch(self.segment_start_root_body_idx_i32, dtype=wp.int32),
                    wp.from_torch(self.segment_end_root_body_idx_i32, dtype=wp.int32),
                    int(self.root_dof_count),
                ],
                outputs=[wp.from_torch(contribution, dtype=wp.float32)],
                device=device,
            )
        else:
            wp.launch(
                _compute_muscle_dynamics_contribution_kernel,
                dim=(batch_size, self.torque_dim, self.muscle_count),
                inputs=[
                    wp.from_torch(
                        geometry_workspace["transformed_waypoints"], dtype=wp.vec3
                    ),
                    wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                    wp.from_torch(subtree_com, dtype=wp.vec3),
                    wp.from_torch(cdof, dtype=wp.float32),
                    wp.from_torch(self.apply_muscle_segment_offsets, dtype=wp.int32),
                    wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                    wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                    wp.from_torch(self.segment_start_root_body_idx_i32, dtype=wp.int32),
                    wp.from_torch(self.segment_end_root_body_idx_i32, dtype=wp.int32),
                    wp.from_torch(self.segment_start_dof_mask, dtype=wp.float32),
                    wp.from_torch(self.segment_end_dof_mask, dtype=wp.float32),
                    int(self.root_dof_count),
                ],
                outputs=[wp.from_torch(contribution, dtype=wp.float32)],
                device=device,
            )

        return contribution

    def _warp_muscle_dynamics_from_activation_forward(
        self,
        activation: torch.Tensor,
        raw_context: torch.Tensor,
        *,
        include_passive_force: bool,
        mean_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        activation = activation.float()
        raw_context = raw_context.float()
        batch_size = activation.shape[0]
        use_lbs_force_precompute = (
            self.uses_lbs_waypoint_attachments
            and self.lbs_attachment_layout in _FORCE_PRECOMPUTE_LBS_ATTACHMENT_LAYOUTS
            and self.lbs_torque_chunk_count > 0
        )
        use_native_body_wrench_forward = (
            self._uses_native_body_wrench_projection
            and self.lbs_native_body_tile_count > 0
        )
        use_native_full_force = (
            use_native_body_wrench_forward and not self.muscle_force_velocity_enabled
        )
        use_native_fv_cached_force = (
            use_native_body_wrench_forward and self.muscle_force_velocity_enabled
        )
        workspace, subtree_com, cdof = self._prepare_warp_waypoint_geometry(
            raw_context,
            include_passive_force_magnitude=include_passive_force,
            compute_force_magnitudes=not (
                use_lbs_force_precompute or use_native_full_force
            ),
            skip_lbs_waypoint_transform=use_native_full_force,
            apply_force_velocity=not use_lbs_force_precompute,
        )
        mean = (
            mean_out
            if mean_out is not None
            else torch.empty(
                batch_size,
                self.torque_dim,
                device=activation.device,
                dtype=torch.float32,
            )
        )
        device = str(raw_context.device)
        if self._uses_attachment_projection:
            if self.lbs_torque_chunk_count == 0 and not use_native_body_wrench_forward:
                mean.zero_()
                return (
                    mean,
                    workspace.get("transformed_waypoints", raw_context.new_empty(0)),
                    workspace["unit_force_magnitude"],
                )
            if use_native_full_force:
                self._launch_native_body_wrench_full_force_into(
                    workspace=workspace,
                    activation=activation,
                    body_pos=workspace["xpos"],
                    body_quat=workspace["xquat"],
                    xipos=workspace["xipos"],
                    subtree_com=subtree_com,
                    cdof=cdof,
                    output=mean,
                    include_passive_force=include_passive_force,
                    output_dof_offset=0,
                )
                return (
                    mean,
                    workspace.get("transformed_waypoints", raw_context.new_empty(0)),
                    workspace["unit_force_magnitude"],
                )
            if use_native_fv_cached_force and not use_lbs_force_precompute:
                self._launch_native_body_wrench_cached_force_into(
                    workspace=workspace,
                    activation=activation,
                    unit_force_magnitude=workspace["unit_force_magnitude"],
                    passive_force_magnitude=(
                        workspace["passive_force_magnitude"]
                        if include_passive_force
                        else workspace["unit_force_magnitude"]
                    ),
                    subtree_com=subtree_com,
                    cdof=cdof,
                    mean=mean,
                    include_passive_force=include_passive_force,
                )
                return (
                    mean,
                    workspace["transformed_waypoints"],
                    workspace["unit_force_magnitude"],
                )
            if self.lbs_attachment_layout in _FORCE_PRECOMPUTE_LBS_ATTACHMENT_LAYOUTS:
                muscle_length = workspace["lbs_muscle_length"]
                use_muscle_length_by_muscle = (
                    self.lbs_segment_start_is_segment_plus_muscle
                    and batch_size >= self.lbs_muscle_length_by_muscle_min_batch
                )
                use_fused_body_wrench_force = (
                    self.lbs_attachment_layout == "body_wrench"
                    and use_muscle_length_by_muscle
                )
                use_grouped_body_wrench_force = (
                    use_fused_body_wrench_force
                    and self.lbs_body_wrench_force_segment_group_width > 1
                )
                if use_grouped_body_wrench_force:
                    segment_group_width = self.lbs_body_wrench_force_segment_group_width
                    segment_length_group_partials = workspace[
                        "lbs_segment_length_group_partials"
                    ]
                    if segment_group_width == 2:
                        wp.launch(
                            _compute_lbs_segment_length_pair_partials_kernel,
                            dim=(batch_size, self.muscle_count, 2),
                            inputs=[
                                wp.from_torch(
                                    workspace["transformed_waypoints"], dtype=wp.vec3
                                ),
                                wp.from_torch(
                                    self.all_muscle_segment_offsets, dtype=wp.int32
                                ),
                            ],
                            outputs=[
                                wp.from_torch(
                                    segment_length_group_partials, dtype=wp.float32
                                )
                            ],
                            device=device,
                        )
                        force_from_partials_kernel = make_lbs_force_by_muscle_from_pair_partials_precomputed_kernel(
                            bool(include_passive_force),
                            self.muscle_passive_force_model_id,
                            self.muscle_passive_force_param0,
                            self.muscle_passive_force_param1,
                        )
                        force_from_partials_inputs = [
                            wp.from_torch(
                                segment_length_group_partials, dtype=wp.float32
                            ),
                            wp.from_torch(
                                self.muscle_force_length_coeff, dtype=wp.vec4
                            ),
                            wp.from_torch(
                                activation, dtype=wp.float32, requires_grad=False
                            ),
                            self.muscle_length_norm_min,
                            self.muscle_length_norm_max,
                            self._muscle_active_force_model_params_wp,
                        ]
                    else:
                        wp.launch(
                            make_lbs_segment_length_group_partials_kernel(
                                segment_group_width
                            ),
                            dim=(batch_size, self.muscle_count, segment_group_width),
                            inputs=[
                                wp.from_torch(
                                    workspace["transformed_waypoints"], dtype=wp.vec3
                                ),
                                wp.from_torch(
                                    self.all_muscle_segment_offsets, dtype=wp.int32
                                ),
                            ],
                            outputs=[
                                wp.from_torch(
                                    segment_length_group_partials, dtype=wp.float32
                                )
                            ],
                            device=device,
                        )
                        force_from_partials_kernel = make_lbs_force_by_muscle_from_group_partials_precomputed_kernel(
                            bool(include_passive_force),
                            segment_group_width,
                            self.muscle_passive_force_model_id,
                            self.muscle_passive_force_param0,
                            self.muscle_passive_force_param1,
                        )
                        force_from_partials_inputs = [
                            wp.from_torch(
                                segment_length_group_partials, dtype=wp.float32
                            ),
                            wp.from_torch(
                                self.muscle_force_length_coeff, dtype=wp.vec4
                            ),
                            wp.from_torch(
                                activation, dtype=wp.float32, requires_grad=False
                            ),
                            self.muscle_length_norm_min,
                            self.muscle_length_norm_max,
                            self._muscle_active_force_model_params_wp,
                        ]
                    wp.launch(
                        force_from_partials_kernel,
                        dim=(batch_size, self.muscle_count),
                        inputs=force_from_partials_inputs,
                        outputs=[
                            wp.from_torch(
                                workspace["unit_force_magnitude"], dtype=wp.float32
                            ),
                            wp.from_torch(
                                workspace["lbs_force_by_muscle"], dtype=wp.float32
                            ),
                        ],
                        device=device,
                    )
                elif use_fused_body_wrench_force:
                    wp.launch(
                        make_lbs_force_by_muscle_contiguous_waypoints_precomputed_kernel(
                            bool(include_passive_force),
                            self.muscle_passive_force_model_id,
                            self.muscle_passive_force_param0,
                            self.muscle_passive_force_param1,
                        ),
                        dim=(batch_size, self.muscle_count),
                        inputs=[
                            wp.from_torch(
                                workspace["transformed_waypoints"], dtype=wp.vec3
                            ),
                            wp.from_torch(
                                self.all_muscle_segment_offsets, dtype=wp.int32
                            ),
                            wp.from_torch(
                                self.muscle_force_length_coeff, dtype=wp.vec4
                            ),
                            wp.from_torch(
                                activation, dtype=wp.float32, requires_grad=False
                            ),
                            self.muscle_length_norm_min,
                            self.muscle_length_norm_max,
                            self._muscle_active_force_model_params_wp,
                        ],
                        outputs=[
                            wp.from_torch(
                                workspace["unit_force_magnitude"], dtype=wp.float32
                            ),
                            wp.from_torch(
                                workspace["lbs_force_by_muscle"], dtype=wp.float32
                            ),
                        ],
                        device=device,
                    )
                else:
                    if use_muscle_length_by_muscle:
                        wp.launch(
                            _accumulate_lbs_muscle_lengths_muscle_contiguous_waypoints_kernel,
                            dim=(batch_size, self.muscle_count),
                            inputs=[
                                wp.from_torch(
                                    workspace["transformed_waypoints"], dtype=wp.vec3
                                ),
                                wp.from_torch(
                                    self.all_muscle_segment_offsets, dtype=wp.int32
                                ),
                            ],
                            outputs=[wp.from_torch(muscle_length, dtype=wp.float32)],
                            device=device,
                        )
                    else:
                        muscle_length.zero_()
                    if (
                        self.lbs_segment_start_is_segment_plus_muscle
                        and not use_muscle_length_by_muscle
                    ):
                        wp.launch(
                            _accumulate_lbs_muscle_lengths_contiguous_waypoints_kernel,
                            dim=(
                                batch_size,
                                int(self.apply_segment_muscle_idx_i32.numel()),
                            ),
                            inputs=[
                                wp.from_torch(
                                    workspace["transformed_waypoints"], dtype=wp.vec3
                                ),
                                wp.from_torch(
                                    self.apply_segment_muscle_idx_i32, dtype=wp.int32
                                ),
                            ],
                            outputs=[wp.from_torch(muscle_length, dtype=wp.float32)],
                            device=device,
                        )
                    elif not use_muscle_length_by_muscle:
                        wp.launch(
                            _accumulate_lbs_muscle_lengths_from_segments_kernel,
                            dim=(
                                batch_size,
                                int(self.apply_segment_muscle_idx_i32.numel()),
                            ),
                            inputs=[
                                wp.from_torch(
                                    workspace["transformed_waypoints"], dtype=wp.vec3
                                ),
                                wp.from_torch(
                                    self.apply_segment_start_idx_i32, dtype=wp.int32
                                ),
                                wp.from_torch(
                                    self.apply_segment_muscle_idx_i32, dtype=wp.int32
                                ),
                            ],
                            outputs=[wp.from_torch(muscle_length, dtype=wp.float32)],
                            device=device,
                        )
                    force_inputs = [
                        wp.from_torch(muscle_length, dtype=wp.float32),
                        wp.from_torch(self.muscle_params, dtype=wp.float32),
                        wp.from_torch(self.muscle_rest_length, dtype=wp.float32),
                        wp.from_torch(
                            activation, dtype=wp.float32, requires_grad=False
                        ),
                        self.muscle_force_scale,
                        self.muscle_length_norm_min,
                        self.muscle_length_norm_max,
                        self._muscle_active_force_model_params_wp,
                    ]
                    if self.lbs_attachment_layout == "body_wrench":
                        wp.launch(
                            make_lbs_force_by_muscle_from_length_precomputed_kernel(
                                bool(include_passive_force),
                                self.muscle_passive_force_model_id,
                                self.muscle_passive_force_param0,
                                self.muscle_passive_force_param1,
                            ),
                            dim=(batch_size, self.muscle_count),
                            inputs=[
                                wp.from_torch(muscle_length, dtype=wp.float32),
                                wp.from_torch(
                                    self.muscle_force_length_coeff, dtype=wp.vec4
                                ),
                                wp.from_torch(
                                    activation,
                                    dtype=wp.float32,
                                    requires_grad=False,
                                ),
                                self.muscle_length_norm_min,
                                self.muscle_length_norm_max,
                                self._muscle_active_force_model_params_wp,
                            ],
                            outputs=[
                                wp.from_torch(
                                    workspace["unit_force_magnitude"],
                                    dtype=wp.float32,
                                ),
                                wp.from_torch(
                                    workspace["lbs_force_by_muscle"],
                                    dtype=wp.float32,
                                ),
                            ],
                            device=device,
                        )
                    else:
                        wp.launch(
                            make_lbs_force_by_muscle_from_length_kernel(
                                bool(include_passive_force),
                                self.muscle_passive_force_model_id,
                                self.muscle_passive_force_param0,
                                self.muscle_passive_force_param1,
                            ),
                            dim=(batch_size, self.muscle_count),
                            inputs=force_inputs,
                            outputs=[
                                wp.from_torch(
                                    workspace["unit_force_magnitude"],
                                    dtype=wp.float32,
                                ),
                                wp.from_torch(
                                    workspace["passive_force_magnitude"],
                                    dtype=wp.float32,
                                ),
                                wp.from_torch(
                                    workspace["lbs_force_by_muscle"],
                                    dtype=wp.float32,
                                ),
                            ],
                            device=device,
                        )
                if self.muscle_force_velocity_enabled:
                    wp.launch(
                        apply_lbs_muscle_force_velocity_to_force_cache_kernel,
                        dim=(batch_size, self.muscle_count),
                        inputs=[
                            wp.from_torch(
                                workspace["transformed_waypoints"], dtype=wp.vec3
                            ),
                            wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                            wp.from_torch(workspace["xquat"], dtype=wp.quat),
                            wp.from_torch(subtree_com, dtype=wp.vec3),
                            wp.from_torch(cdof, dtype=wp.float32),
                            wp.from_torch(raw_context, dtype=wp.float32),
                            wp.from_torch(activation, dtype=wp.float32),
                            wp.from_torch(
                                self.muscle_active_entry_offsets, dtype=wp.int32
                            ),
                            wp.from_torch(
                                self.muscle_active_segment_idx, dtype=wp.int32
                            ),
                            wp.from_torch(
                                self.muscle_active_torque_idx, dtype=wp.int32
                            ),
                            wp.from_torch(
                                self.apply_segment_start_idx_i32, dtype=wp.int32
                            ),
                            wp.from_torch(
                                self.apply_segment_end_idx_i32, dtype=wp.int32
                            ),
                            wp.from_torch(self.body_rootid_i32, dtype=wp.int32),
                            wp.from_torch(
                                self.lbs_active_attach_offsets, dtype=wp.int32
                            ),
                            wp.from_torch(self.lbs_active_attach_body, dtype=wp.int32),
                            wp.from_torch(self.lbs_active_attach_local, dtype=wp.vec3),
                            wp.from_torch(
                                self.lbs_active_attach_coeff_weight, dtype=wp.float32
                            ),
                            wp.from_torch(
                                self.muscle_force_velocity_length_scale,
                                dtype=wp.float32,
                            ),
                            int(self.lbs_waypoint_count),
                            int(self.full_torque_dim - self.torque_dim),
                            int(self._geometry_context_dim),
                            self.muscle_force_velocity_vmax,
                            self.muscle_force_velocity_fvmax,
                        ],
                        outputs=[
                            wp.from_torch(
                                workspace["unit_force_magnitude"], dtype=wp.float32
                            ),
                            wp.from_torch(
                                workspace["lbs_force_by_muscle"], dtype=wp.float32
                            ),
                        ],
                        device=device,
                    )
                if use_native_fv_cached_force:
                    self._launch_native_body_wrench_cached_force_into(
                        workspace=workspace,
                        activation=activation,
                        unit_force_magnitude=workspace["lbs_force_by_muscle"],
                        passive_force_magnitude=workspace["lbs_force_by_muscle"],
                        subtree_com=subtree_com,
                        cdof=cdof,
                        mean=mean,
                        include_passive_force=False,
                        force_is_precomputed=True,
                    )
                    return (
                        mean,
                        workspace["transformed_waypoints"],
                        workspace["unit_force_magnitude"],
                    )
                if self.lbs_attachment_layout == "body_wrench":
                    if self.lbs_body_event_chunk_count == 0:
                        mean.zero_()
                        return (
                            mean,
                            workspace["transformed_waypoints"],
                            workspace["unit_force_magnitude"],
                        )
                    partial = workspace["lbs_body_wrench_chunk_partial"]
                    body_wrench_inputs: list[Any] = [
                        wp.from_torch(
                            workspace["transformed_waypoints"], dtype=wp.vec3
                        ),
                        wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                        wp.from_torch(workspace["xquat"], dtype=wp.quat),
                        wp.from_torch(workspace["xipos"], dtype=wp.vec3),
                        wp.from_torch(
                            workspace["lbs_force_by_muscle"], dtype=wp.float32
                        ),
                        wp.from_torch(self.lbs_body_event_chunk_info, dtype=wp.vec3i),
                    ]
                    if self.lbs_body_segment_event_start_contiguous:
                        body_wrench_inputs.extend(
                            [
                                wp.from_torch(
                                    self.lbs_body_segment_event_start_muscle,
                                    dtype=wp.vec2i,
                                ),
                                wp.from_torch(
                                    self.lbs_body_event_local_force_coeff,
                                    dtype=wp.vec4,
                                ),
                            ]
                        )
                        body_wrench_kernel = _accumulate_lbs_body_wrench_segment_start_event_chunk_partials_kernel
                    else:
                        body_wrench_inputs.extend(
                            [
                                wp.from_torch(
                                    self.lbs_body_event_packed_indices,
                                    dtype=wp.vec4i,
                                ),
                                wp.from_torch(
                                    self.lbs_body_event_local_force_coeff,
                                    dtype=wp.vec4,
                                ),
                            ]
                        )
                        body_wrench_kernel = _accumulate_lbs_body_wrench_segment_event_chunk_partials_kernel
                    wp.launch_tiled(
                        body_wrench_kernel,
                        dim=(batch_size, self.lbs_body_event_chunk_count),
                        inputs=body_wrench_inputs,
                        outputs=[wp.from_torch(partial, dtype=wp.float32)],
                        device=device,
                        block_dim=self.lbs_forward_tile_block_dim,
                    )
                    wp.launch(
                        _reduce_lbs_body_wrench_event_partials_kernel,
                        dim=(batch_size, self.body_count),
                        inputs=[
                            wp.from_torch(partial, dtype=wp.float32),
                            wp.from_torch(
                                self.lbs_body_event_chunk_offsets, dtype=wp.int32
                            ),
                        ],
                        outputs=[
                            wp.from_torch(
                                workspace["lbs_body_force"], dtype=wp.float32
                            ),
                            wp.from_torch(
                                workspace["lbs_body_torque"], dtype=wp.float32
                            ),
                        ],
                        device=device,
                    )
                    self._launch_project_body_wrench_to_torque(
                        body_force=workspace["lbs_body_force"],
                        body_torque=workspace["lbs_body_torque"],
                        xipos=workspace["xipos"],
                        subtree_com=subtree_com,
                        cdof=cdof,
                        mean=mean,
                        batch_size=batch_size,
                        device=device,
                    )
                    return (
                        mean,
                        workspace["transformed_waypoints"],
                        workspace["unit_force_magnitude"],
                    )
                if self.lbs_active_attach_chunk_count == 0:
                    mean.zero_()
                    return (
                        mean,
                        workspace["transformed_waypoints"],
                        workspace["unit_force_magnitude"],
                    )
                waypoint_force = workspace["lbs_waypoint_force"]
                wp.launch(
                    make_lbs_waypoint_force_from_force_kernel(),
                    dim=(batch_size, self.lbs_waypoint_count),
                    inputs=[
                        wp.from_torch(
                            workspace["transformed_waypoints"], dtype=wp.vec3
                        ),
                        wp.from_torch(
                            workspace["lbs_force_by_muscle"], dtype=wp.float32
                        ),
                        wp.from_torch(self.lbs_waypoint_segment_idx2, dtype=wp.vec2i),
                        wp.from_torch(self.lbs_waypoint_segment_sign2, dtype=wp.vec2i),
                        wp.from_torch(self.lbs_segment_waypoints, dtype=wp.vec2i),
                        wp.from_torch(
                            self.apply_segment_muscle_idx_i32, dtype=wp.int32
                        ),
                    ],
                    outputs=[wp.from_torch(waypoint_force, dtype=wp.vec3)],
                    device=device,
                )
                partial = workspace["lbs_active_attach_chunk_partial"]
                active_attach_inputs: list[Any] = [
                    wp.from_torch(waypoint_force, dtype=wp.vec3),
                    wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                    wp.from_torch(workspace["xquat"], dtype=wp.quat),
                    wp.from_torch(subtree_com, dtype=wp.vec3),
                    wp.from_torch(cdof, dtype=wp.float32),
                ]
                if self.lbs_single_root_body_idx < 0:
                    active_attach_inputs.append(
                        wp.from_torch(self.body_rootid_i32, dtype=wp.int32)
                    )
                active_attach_inputs.extend(
                    [
                        wp.from_torch(
                            self.lbs_active_attach_chunk_info,
                            dtype=wp.vec3i,
                        ),
                        wp.from_torch(
                            self.lbs_active_attach_idx_body,
                            dtype=wp.vec2i,
                        ),
                        wp.from_torch(
                            self.lbs_active_attach_local_coeff,
                            dtype=wp.vec4,
                        ),
                    ]
                )
                active_attach_inputs.append(int(self.root_dof_count))
                wp.launch_tiled(
                    make_lbs_torque_waypoint_force_active_attach_pair_vec4_native_partials_kernel(
                        bool(include_passive_force),
                        int(self.lbs_single_root_body_idx),
                    ),
                    dim=(batch_size, self.lbs_active_attach_chunk_count),
                    inputs=active_attach_inputs,
                    outputs=[wp.from_torch(partial, dtype=wp.float32)],
                    device=device,
                    block_dim=self.lbs_forward_tile_block_dim,
                )
                wp.launch(
                    _reduce_lbs_torque_chunk_partials_kernel,
                    dim=(batch_size, self.torque_dim),
                    inputs=[
                        wp.from_torch(partial, dtype=wp.float32),
                        wp.from_torch(
                            self.lbs_active_attach_chunk_offsets, dtype=wp.int32
                        ),
                    ],
                    outputs=[wp.from_torch(mean, dtype=wp.float32)],
                    device=device,
                )
                return (
                    mean,
                    workspace["transformed_waypoints"],
                    workspace["unit_force_magnitude"],
                )
        if self.sparse_body_wrench_forward:
            body_force = workspace["sparse_body_force"]
            body_torque = workspace["sparse_body_torque"]
            body_force.zero_()
            body_torque.zero_()
            if self.sparse_body_wrench_direct:
                body_wrench_kernel = (
                    make_accumulate_body_wrench_direct_from_activation_kernel(
                        include_passive_force
                    )
                )
                body_wrench_inputs: list[Any] = [
                    wp.from_torch(workspace["transformed_waypoints"], dtype=wp.vec3),
                    wp.from_torch(workspace["xipos"], dtype=wp.vec3),
                    wp.from_torch(workspace["unit_force_magnitude"], dtype=wp.float32),
                ]
                # The passive-off specialisation declares the argument but
                # never reads it, so there is nothing to pass.
                body_wrench_inputs.append(
                    wp.from_torch(
                        workspace["passive_force_magnitude"],
                        dtype=wp.float32,
                    )
                    if include_passive_force
                    else None
                )
                body_wrench_inputs.extend(
                    [
                        wp.from_torch(
                            activation,
                            dtype=wp.float32,
                            requires_grad=False,
                        ),
                        wp.from_torch(
                            self.apply_segment_muscle_idx_i32, dtype=wp.int32
                        ),
                        wp.from_torch(
                            self.apply_segment_start_idx_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                        wp.from_torch(
                            self.apply_segment_body_start_idx_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.apply_segment_body_end_idx_i32,
                            dtype=wp.int32,
                        ),
                    ]
                )
            else:
                body_wrench_kernel = (
                    make_accumulate_lbs_body_wrench_from_activation_kernel(
                        include_passive_force
                    )
                )
                body_wrench_inputs = [
                    wp.from_torch(workspace["transformed_waypoints"], dtype=wp.vec3),
                    wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                    wp.from_torch(workspace["xquat"], dtype=wp.quat),
                    wp.from_torch(workspace["xipos"], dtype=wp.vec3),
                    wp.from_torch(workspace["unit_force_magnitude"], dtype=wp.float32),
                ]
                # The passive-off specialisation declares the argument but
                # never reads it, so there is nothing to pass.
                body_wrench_inputs.append(
                    wp.from_torch(
                        workspace["passive_force_magnitude"],
                        dtype=wp.float32,
                    )
                    if include_passive_force
                    else None
                )
                body_wrench_inputs.extend(
                    [
                        wp.from_torch(
                            activation,
                            dtype=wp.float32,
                            requires_grad=False,
                        ),
                        wp.from_torch(
                            self.apply_segment_muscle_idx_i32, dtype=wp.int32
                        ),
                        wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                        wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_start_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_body_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_local, dtype=wp.vec3
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_weight,
                            dtype=wp.float32,
                        ),
                    ]
                )
            wp.launch(
                body_wrench_kernel,
                dim=(batch_size, int(self.apply_segment_muscle_idx_i32.numel())),
                inputs=body_wrench_inputs,
                outputs=[
                    wp.from_torch(body_force, dtype=wp.float32),
                    wp.from_torch(body_torque, dtype=wp.float32),
                ],
                device=device,
            )
            self._launch_project_body_wrench_to_torque(
                body_force=body_force,
                body_torque=body_torque,
                xipos=workspace["xipos"],
                subtree_com=subtree_com,
                cdof=cdof,
                mean=mean,
                batch_size=batch_size,
                device=device,
            )
            return (
                mean,
                workspace["transformed_waypoints"],
                workspace["unit_force_magnitude"],
            )
        kernel = make_muscle_dynamics_from_activation_sparse_kernel(
            include_passive_force
        )
        kernel_inputs: list[Any] = [
            wp.from_torch(workspace["transformed_waypoints"], dtype=wp.vec3),
            wp.from_torch(workspace["unit_force_magnitude"], dtype=wp.float32),
        ]
        kernel_inputs.append(
            wp.from_torch(workspace["passive_force_magnitude"], dtype=wp.float32)
            if include_passive_force
            else None
        )
        kernel_inputs.extend(
            [
                wp.from_torch(activation, dtype=wp.float32, requires_grad=False),
                wp.from_torch(subtree_com, dtype=wp.vec3),
                wp.from_torch(cdof, dtype=wp.float32),
                wp.from_torch(self.torque_active_entry_offsets, dtype=wp.int32),
                wp.from_torch(self.torque_active_segment_idx, dtype=wp.int32),
                wp.from_torch(self.torque_active_muscle_idx, dtype=wp.int32),
                wp.from_torch(self.torque_active_start_coeff, dtype=wp.float32),
                wp.from_torch(self.torque_active_end_coeff, dtype=wp.float32),
                wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                wp.from_torch(self.segment_start_root_body_idx_i32, dtype=wp.int32),
                wp.from_torch(self.segment_end_root_body_idx_i32, dtype=wp.int32),
                int(self.root_dof_count),
            ]
        )
        wp.launch(
            kernel,
            dim=(batch_size, self.torque_dim),
            inputs=kernel_inputs,
            outputs=[wp.from_torch(mean, dtype=wp.float32)],
            device=device,
        )
        return (
            mean,
            workspace["transformed_waypoints"],
            workspace["unit_force_magnitude"],
        )

    def _prepare_native_tile_sorted_muscle_input(
        self,
        workspace: dict[str, torch.Tensor],
        source: torch.Tensor,
        *,
        key: str,
    ) -> torch.Tensor:
        out = workspace.get(key)
        if out is None or out.shape != source.shape:
            out = torch.empty_like(source, dtype=torch.float32)
            workspace[key] = out
        # Native-tile launches pass this persistent scratch to Warp with
        # requires_grad=False; the layer's custom backward computes activation
        # gradients separately. Keep the ordering copy on that boundary so a
        # A source carrying grad_fn remains a legal input.
        with torch.no_grad():
            torch.index_select(
                source,
                1,
                self.lbs_native_muscle_order,
                out=out,
            )
        return out

    def _prepare_warp_native_tile_backward_geometry(
        self,
        raw_context: torch.Tensor,
        *,
        compute_unit_force_magnitude: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        raw_context = raw_context.float()
        wp.init()
        batch_size = raw_context.shape[0]
        workspace = self._get_warp_geometry_workspace(
            batch_size=batch_size,
            device=raw_context.device,
            minimal_native_lbs=True,
        )
        xpos = raw_context[:, : self._xpos_width].reshape(
            batch_size, self.body_count, 3
        )
        xquat = raw_context[
            :, self._xpos_width : self._xpos_width + self._xquat_width
        ].reshape(batch_size, self.body_count, 4)
        xipos = raw_context[
            :,
            self._xpos_width + self._xquat_width : self._xpos_width
            + self._xquat_width
            + self._xipos_width,
        ].reshape(batch_size, self.body_count, 3)
        offset = self._xpos_width + self._xquat_width + self._xipos_width
        subtree_com = raw_context[:, offset : offset + self._subtree_com_width].reshape(
            batch_size, self.body_count, 3
        )
        offset += self._subtree_com_width
        cdof = raw_context[:, offset : offset + self._cdof_width].reshape(
            batch_size, self.full_torque_dim, 6
        )
        workspace["xpos"] = xpos
        workspace["xquat"] = xquat
        workspace["xipos"] = xipos
        device = str(raw_context.device)
        wp.launch(
            _normalize_body_quats_kernel,
            dim=(batch_size, self.body_count),
            inputs=[wp.from_torch(xquat, dtype=wp.quat)],
            outputs=[wp.from_torch(workspace["xquat_normalized"], dtype=wp.quat)],
            device=device,
        )
        if compute_unit_force_magnitude:
            unit_force_kernel = (
                _compute_lbs_fixed1_unit_force_from_geometry_kernel
                if self.lbs_native_waypoint_attachment_arity == 1
                else _compute_lbs_fixed2_unit_force_from_geometry_kernel
            )
            unit_force_inputs: list[Any] = [
                wp.from_torch(xpos, dtype=wp.vec3),
                wp.from_torch(workspace["xquat_normalized"], dtype=wp.quat),
                wp.from_torch(self.all_muscle_segment_offsets, dtype=wp.int32),
                wp.from_torch(self.muscle_force_length_coeff, dtype=wp.vec4),
                wp.from_torch(self.lbs_waypoint_attachment_body2, dtype=wp.vec2i),
                wp.from_torch(
                    self.lbs_waypoint_attachment_local_weight0, dtype=wp.vec4
                ),
            ]
            if self.lbs_native_waypoint_attachment_arity == 2:
                unit_force_inputs.append(
                    wp.from_torch(
                        self.lbs_waypoint_attachment_local_weight1,
                        dtype=wp.vec4,
                    )
                )
            unit_force_inputs.extend(
                [
                    self.muscle_length_norm_min,
                    self.muscle_length_norm_max,
                    self._muscle_active_force_model_params_wp,
                ]
            )
            wp.launch(
                unit_force_kernel,
                dim=(batch_size, self.muscle_count),
                inputs=unit_force_inputs,
                outputs=[
                    wp.from_torch(workspace["unit_force_magnitude"], dtype=wp.float32)
                ],
                device=device,
            )
        return workspace, subtree_com, cdof

    def precompute_lbs_force_magnitude_cache(
        self,
        raw_context: torch.Tensor,
        *,
        include_passive_force: bool,
        unit_force_magnitude: torch.Tensor | None = None,
        passive_force_magnitude: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_supported_physical_geometry()
        raw_context = raw_context.float()
        batch_size = raw_context.shape[0]
        if unit_force_magnitude is None:
            unit_force_magnitude = torch.empty(
                batch_size,
                self.muscle_count,
                device=raw_context.device,
                dtype=torch.float32,
            )
        if passive_force_magnitude is None:
            passive_force_magnitude = torch.empty_like(unit_force_magnitude)
        use_fixed_arity_native_precompute = bool(
            self.geometry_backend == "waypoint"
            and self._uses_native_body_wrench_projection
            and not self._waypoint_wrapping.count
            and not self.muscle_force_velocity_enabled
            and self.lbs_native_waypoint_attachment_arity in {1, 2}
        )
        if not use_fixed_arity_native_precompute:
            chunk_size = self._force_cache_precompute_chunk_size(batch_size)
            if chunk_size < batch_size:
                for start in range(0, batch_size, chunk_size):
                    end = min(start + chunk_size, batch_size)
                    self.precompute_lbs_force_magnitude_cache(
                        raw_context[start:end],
                        include_passive_force=include_passive_force,
                        unit_force_magnitude=unit_force_magnitude[start:end],
                        passive_force_magnitude=passive_force_magnitude[start:end],
                    )
                return unit_force_magnitude, passive_force_magnitude
        if use_fixed_arity_native_precompute:
            workspace, xpos = self._prepare_warp_lbs_force_cache_geometry(raw_context)
            precompute_kernel = (
                _precompute_lbs_fixed1_force_magnitude_cache_kernel
                if self.lbs_native_waypoint_attachment_arity == 1
                else _precompute_lbs_fixed2_force_magnitude_cache_kernel
            )
            precompute_inputs: list[Any] = [
                wp.from_torch(xpos, dtype=wp.vec3),
                wp.from_torch(workspace["xquat_normalized"], dtype=wp.quat),
                wp.from_torch(self.all_muscle_segment_offsets, dtype=wp.int32),
                wp.from_torch(self.muscle_force_length_coeff, dtype=wp.vec4),
                wp.from_torch(self.lbs_waypoint_attachment_body2, dtype=wp.vec2i),
                wp.from_torch(
                    self.lbs_waypoint_attachment_local_weight0,
                    dtype=wp.vec4,
                ),
            ]
            if self.lbs_native_waypoint_attachment_arity == 2:
                precompute_inputs.append(
                    wp.from_torch(
                        self.lbs_waypoint_attachment_local_weight1,
                        dtype=wp.vec4,
                    )
                )
            precompute_inputs.extend(
                [
                    self.muscle_length_norm_min,
                    self.muscle_length_norm_max,
                    1 if include_passive_force else 0,
                    self._muscle_active_force_model_params_wp,
                    self._muscle_passive_force_model_params_wp,
                ]
            )
            wp.launch(
                precompute_kernel,
                dim=(batch_size, self.muscle_count),
                inputs=precompute_inputs,
                outputs=[
                    wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                    wp.from_torch(passive_force_magnitude, dtype=wp.float32),
                ],
                device=str(raw_context.device),
            )
            return unit_force_magnitude, passive_force_magnitude

        workspace, _, _ = self._prepare_warp_waypoint_geometry(
            raw_context,
            include_passive_force_magnitude=include_passive_force,
            compute_force_magnitudes=True,
            apply_force_velocity=True,
        )
        unit_force_magnitude.copy_(workspace["unit_force_magnitude"])
        passive_force_magnitude.copy_(workspace["passive_force_magnitude"])
        return unit_force_magnitude, passive_force_magnitude

    def _force_cache_precompute_chunk_size(self, batch_size: int) -> int:
        """Bound persistent precompute scratch without changing cache payloads."""
        batch_size = int(batch_size)
        if batch_size <= 1:
            return max(batch_size, 1)

        muscle_count = int(self.muscle_count)
        body_count = int(self.body_count)
        waypoint_count = int(self.muscle_waypoints_local.shape[0])
        bytes_per_sample = 12 * waypoint_count + 16 * muscle_count + 48 * body_count
        if self._uses_attachment_projection:
            bytes_per_sample += 8 * muscle_count + 16 * body_count
            if self.lbs_attachment_layout == "waypoint_force":
                bytes_per_sample += 12 * int(self.lbs_waypoint_count) + 4 * int(
                    self.lbs_active_attach_chunk_count
                )
            elif self.lbs_attachment_layout == "body_wrench":
                bytes_per_sample += (
                    24 * int(self.lbs_body_event_chunk_count)
                    + 24 * int(self.lbs_native_body_tile_body_slot_count)
                    + 24 * body_count
                )
            if self.lbs_segment_start_is_segment_plus_muscle:
                bytes_per_sample += (
                    4
                    * muscle_count
                    * int(self.lbs_body_wrench_force_segment_group_width)
                )

        target_scratch_bytes = 256 * 1024 * 1024
        return min(
            batch_size,
            1024,
            max(1, target_scratch_bytes // max(bytes_per_sample, 1)),
        )

    def _launch_native_body_wrench_cached_force_into(
        self,
        *,
        workspace: dict[str, torch.Tensor],
        activation: torch.Tensor,
        unit_force_magnitude: torch.Tensor,
        passive_force_magnitude: torch.Tensor,
        subtree_com: torch.Tensor,
        cdof: torch.Tensor,
        mean: torch.Tensor,
        include_passive_force: bool,
        force_is_precomputed: bool = False,
    ) -> None:
        batch_size = int(activation.shape[0])
        device = str(activation.device)
        partial = workspace["lbs_native_body_wrench_tile_partial"]
        unit_force_magnitude = self._prepare_native_tile_sorted_muscle_input(
            workspace,
            unit_force_magnitude,
            key=(
                "lbs_native_sorted_force"
                if force_is_precomputed
                else "lbs_native_sorted_unit_force"
            ),
        )
        if not force_is_precomputed:
            activation = self._prepare_native_tile_sorted_muscle_input(
                workspace,
                activation,
                key="lbs_native_sorted_activation",
            )
        if (
            not force_is_precomputed
            and include_passive_force
            and passive_force_magnitude.numel() > 0
        ):
            passive_force_magnitude = self._prepare_native_tile_sorted_muscle_input(
                workspace,
                passive_force_magnitude,
                key="lbs_native_sorted_passive_force",
            )
        kernel = (
            self._lbs_native_body_wrench_direct_force_kernel
            if force_is_precomputed
            else self._lbs_native_body_wrench_cached_force_kernel
        )
        if kernel is None:
            raise RuntimeError("native-tile cached-force kernel was not initialized")
        wp.launch_tiled(
            kernel,
            dim=(batch_size, self.lbs_native_body_tile_count),
            inputs=[
                wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                *self._native_wrap_inputs(workspace, batch_size),
                wp.from_torch(workspace["xquat_normalized"], dtype=wp.quat),
                wp.from_torch(workspace["xipos"], dtype=wp.vec3),
                wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                wp.from_torch(
                    (
                        passive_force_magnitude
                        if not force_is_precomputed
                        and include_passive_force
                        and passive_force_magnitude.numel() > 0
                        else unit_force_magnitude
                    ),
                    dtype=wp.float32,
                ),
                wp.from_torch(activation, dtype=wp.float32, requires_grad=False),
                wp.from_torch(self.lbs_native_body_tile_info, dtype=wp.vec4i),
                wp.from_torch(self.lbs_native_body_tile_extra_info, dtype=wp.vec4i),
                wp.from_torch(self.lbs_native_body_tile_body_ids, dtype=wp.int32),
                wp.from_torch(
                    self.lbs_native_body_tile_body_output_idx,
                    dtype=wp.int32,
                ),
                wp.from_torch(
                    self.lbs_native_body_tile_event_start_slot_muscle,
                    dtype=wp.int32,
                ),
                wp.from_torch(
                    self.lbs_native_body_tile_event_local_force_coeff,
                    dtype=wp.vec4,
                ),
                wp.from_torch(
                    self.lbs_native_waypoint_attachment_body2,
                    dtype=wp.vec2i,
                ),
                wp.from_torch(
                    self.lbs_native_waypoint_attachment_local_weight0,
                    dtype=wp.vec4,
                ),
                wp.from_torch(
                    self.lbs_native_waypoint_attachment_local_weight1,
                    dtype=wp.vec4,
                ),
                1 if include_passive_force and not force_is_precomputed else 0,
            ],
            outputs=[wp.from_torch(partial, dtype=wp.float32)],
            device=device,
            block_dim=self.lbs_forward_tile_block_dim,
        )
        wp.launch(
            _reduce_lbs_body_wrench_event_partials_kernel,
            dim=(batch_size, self.body_count),
            inputs=[
                wp.from_torch(partial, dtype=wp.float32),
                wp.from_torch(
                    self.lbs_native_body_tile_body_slot_offsets,
                    dtype=wp.int32,
                ),
            ],
            outputs=[
                wp.from_torch(workspace["lbs_body_force"], dtype=wp.float32),
                wp.from_torch(workspace["lbs_body_torque"], dtype=wp.float32),
            ],
            device=device,
        )
        self._launch_project_body_wrench_to_torque(
            body_force=workspace["lbs_body_force"],
            body_torque=workspace["lbs_body_torque"],
            xipos=workspace["xipos"],
            subtree_com=subtree_com,
            cdof=cdof,
            mean=mean,
            batch_size=batch_size,
            device=device,
        )

    def _warp_muscle_dynamics_from_activation_forward_cached_lbs_force(
        self,
        activation: torch.Tensor,
        raw_context: torch.Tensor,
        *,
        unit_force_magnitude: torch.Tensor,
        passive_force_magnitude: torch.Tensor,
        include_passive_force: bool,
        mean_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        activation = activation.float()
        raw_context = raw_context.float()
        batch_size = activation.shape[0]
        use_native_body_wrench = (
            self._uses_native_body_wrench_projection
            and self.lbs_native_body_tile_count > 0
        )
        workspace, subtree_com, cdof = self._prepare_warp_waypoint_geometry(
            raw_context,
            include_passive_force_magnitude=False,
            compute_force_magnitudes=False,
            skip_lbs_waypoint_transform=use_native_body_wrench
            and not self._waypoint_wrapping.count,
        )
        mean = (
            mean_out
            if mean_out is not None
            else torch.empty(
                batch_size,
                self.torque_dim,
                device=activation.device,
                dtype=torch.float32,
            )
        )
        if use_native_body_wrench:
            self._launch_native_body_wrench_cached_force_into(
                workspace=workspace,
                activation=activation,
                unit_force_magnitude=unit_force_magnitude,
                passive_force_magnitude=passive_force_magnitude,
                subtree_com=subtree_com,
                cdof=cdof,
                mean=mean,
                include_passive_force=include_passive_force,
            )
            return mean

        use_passive_force = bool(
            include_passive_force and passive_force_magnitude.numel() > 0
        )
        device = str(raw_context.device)
        if self._uses_attachment_projection:
            force_by_muscle = workspace["lbs_force_by_muscle"]
            torch.mul(unit_force_magnitude, activation, out=force_by_muscle)
            if use_passive_force:
                torch.add(
                    force_by_muscle,
                    passive_force_magnitude,
                    out=force_by_muscle,
                )
            if self.lbs_attachment_layout == "body_wrench":
                if self.lbs_body_event_chunk_count == 0:
                    mean.zero_()
                    return mean
                partial = workspace["lbs_body_wrench_chunk_partial"]
                body_wrench_inputs: list[Any] = [
                    wp.from_torch(workspace["transformed_waypoints"], dtype=wp.vec3),
                    wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                    wp.from_torch(workspace["xquat"], dtype=wp.quat),
                    wp.from_torch(workspace["xipos"], dtype=wp.vec3),
                    wp.from_torch(force_by_muscle, dtype=wp.float32),
                    wp.from_torch(self.lbs_body_event_chunk_info, dtype=wp.vec3i),
                ]
                if self.lbs_body_segment_event_start_contiguous:
                    body_wrench_inputs.extend(
                        [
                            wp.from_torch(
                                self.lbs_body_segment_event_start_muscle,
                                dtype=wp.vec2i,
                            ),
                            wp.from_torch(
                                self.lbs_body_event_local_force_coeff,
                                dtype=wp.vec4,
                            ),
                        ]
                    )
                    body_wrench_kernel = _accumulate_lbs_body_wrench_segment_start_event_chunk_partials_kernel
                else:
                    body_wrench_inputs.extend(
                        [
                            wp.from_torch(
                                self.lbs_body_event_packed_indices,
                                dtype=wp.vec4i,
                            ),
                            wp.from_torch(
                                self.lbs_body_event_local_force_coeff,
                                dtype=wp.vec4,
                            ),
                        ]
                    )
                    body_wrench_kernel = (
                        _accumulate_lbs_body_wrench_segment_event_chunk_partials_kernel
                    )
                wp.launch_tiled(
                    body_wrench_kernel,
                    dim=(batch_size, self.lbs_body_event_chunk_count),
                    inputs=body_wrench_inputs,
                    outputs=[wp.from_torch(partial, dtype=wp.float32)],
                    device=device,
                    block_dim=self.lbs_forward_tile_block_dim,
                )
                wp.launch(
                    _reduce_lbs_body_wrench_event_partials_kernel,
                    dim=(batch_size, self.body_count),
                    inputs=[
                        wp.from_torch(partial, dtype=wp.float32),
                        wp.from_torch(
                            self.lbs_body_event_chunk_offsets,
                            dtype=wp.int32,
                        ),
                    ],
                    outputs=[
                        wp.from_torch(workspace["lbs_body_force"], dtype=wp.float32),
                        wp.from_torch(workspace["lbs_body_torque"], dtype=wp.float32),
                    ],
                    device=device,
                )
                self._launch_project_body_wrench_to_torque(
                    body_force=workspace["lbs_body_force"],
                    body_torque=workspace["lbs_body_torque"],
                    xipos=workspace["xipos"],
                    subtree_com=subtree_com,
                    cdof=cdof,
                    mean=mean,
                    batch_size=batch_size,
                    device=device,
                )
                return mean

            if self.lbs_active_attach_chunk_count == 0:
                mean.zero_()
                return mean
            waypoint_force = workspace["lbs_waypoint_force"]
            wp.launch(
                make_lbs_waypoint_force_from_force_kernel(),
                dim=(batch_size, self.lbs_waypoint_count),
                inputs=[
                    wp.from_torch(workspace["transformed_waypoints"], dtype=wp.vec3),
                    wp.from_torch(force_by_muscle, dtype=wp.float32),
                    wp.from_torch(self.lbs_waypoint_segment_idx2, dtype=wp.vec2i),
                    wp.from_torch(self.lbs_waypoint_segment_sign2, dtype=wp.vec2i),
                    wp.from_torch(self.lbs_segment_waypoints, dtype=wp.vec2i),
                    wp.from_torch(
                        self.apply_segment_muscle_idx_i32,
                        dtype=wp.int32,
                    ),
                ],
                outputs=[wp.from_torch(waypoint_force, dtype=wp.vec3)],
                device=device,
            )
            partial = workspace["lbs_active_attach_chunk_partial"]
            active_attach_inputs: list[Any] = [
                wp.from_torch(waypoint_force, dtype=wp.vec3),
                wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                wp.from_torch(workspace["xquat"], dtype=wp.quat),
                wp.from_torch(subtree_com, dtype=wp.vec3),
                wp.from_torch(cdof, dtype=wp.float32),
            ]
            if self.lbs_single_root_body_idx < 0:
                active_attach_inputs.append(
                    wp.from_torch(self.body_rootid_i32, dtype=wp.int32)
                )
            active_attach_inputs.extend(
                [
                    wp.from_torch(
                        self.lbs_active_attach_chunk_info,
                        dtype=wp.vec3i,
                    ),
                    wp.from_torch(
                        self.lbs_active_attach_idx_body,
                        dtype=wp.vec2i,
                    ),
                    wp.from_torch(
                        self.lbs_active_attach_local_coeff,
                        dtype=wp.vec4,
                    ),
                ]
            )
            active_attach_inputs.append(int(self.root_dof_count))
            wp.launch_tiled(
                make_lbs_torque_waypoint_force_active_attach_pair_vec4_native_partials_kernel(
                    use_passive_force,
                    int(self.lbs_single_root_body_idx),
                ),
                dim=(batch_size, self.lbs_active_attach_chunk_count),
                inputs=active_attach_inputs,
                outputs=[wp.from_torch(partial, dtype=wp.float32)],
                device=device,
                block_dim=self.lbs_forward_tile_block_dim,
            )
            wp.launch(
                _reduce_lbs_torque_chunk_partials_kernel,
                dim=(batch_size, self.torque_dim),
                inputs=[
                    wp.from_torch(partial, dtype=wp.float32),
                    wp.from_torch(
                        self.lbs_active_attach_chunk_offsets,
                        dtype=wp.int32,
                    ),
                ],
                outputs=[wp.from_torch(mean, dtype=wp.float32)],
                device=device,
            )
            return mean

        if self.sparse_body_wrench_forward:
            body_force = workspace["sparse_body_force"]
            body_torque = workspace["sparse_body_torque"]
            body_force.zero_()
            body_torque.zero_()
            if self.sparse_body_wrench_direct:
                body_wrench_kernel = (
                    make_accumulate_body_wrench_direct_from_activation_kernel(
                        use_passive_force
                    )
                )
                body_wrench_inputs = [
                    wp.from_torch(workspace["transformed_waypoints"], dtype=wp.vec3),
                    wp.from_torch(workspace["xipos"], dtype=wp.vec3),
                    wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                ]
                body_wrench_inputs.append(
                    wp.from_torch(passive_force_magnitude, dtype=wp.float32)
                    if use_passive_force
                    else None
                )
                body_wrench_inputs.extend(
                    [
                        wp.from_torch(
                            activation,
                            dtype=wp.float32,
                            requires_grad=False,
                        ),
                        wp.from_torch(
                            self.apply_segment_muscle_idx_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.apply_segment_start_idx_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.apply_segment_end_idx_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.apply_segment_body_start_idx_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.apply_segment_body_end_idx_i32,
                            dtype=wp.int32,
                        ),
                    ]
                )
            else:
                body_wrench_kernel = (
                    make_accumulate_lbs_body_wrench_from_activation_kernel(
                        use_passive_force
                    )
                )
                body_wrench_inputs = [
                    wp.from_torch(workspace["transformed_waypoints"], dtype=wp.vec3),
                    wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                    wp.from_torch(workspace["xquat"], dtype=wp.quat),
                    wp.from_torch(workspace["xipos"], dtype=wp.vec3),
                    wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                ]
                body_wrench_inputs.append(
                    wp.from_torch(passive_force_magnitude, dtype=wp.float32)
                    if use_passive_force
                    else None
                )
                body_wrench_inputs.extend(
                    [
                        wp.from_torch(
                            activation,
                            dtype=wp.float32,
                            requires_grad=False,
                        ),
                        wp.from_torch(
                            self.apply_segment_muscle_idx_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.apply_segment_start_idx_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.apply_segment_end_idx_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_start_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_body_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_local,
                            dtype=wp.vec3,
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_weight,
                            dtype=wp.float32,
                        ),
                    ]
                )
            wp.launch(
                body_wrench_kernel,
                dim=(batch_size, int(self.apply_segment_muscle_idx_i32.numel())),
                inputs=body_wrench_inputs,
                outputs=[
                    wp.from_torch(body_force, dtype=wp.float32),
                    wp.from_torch(body_torque, dtype=wp.float32),
                ],
                device=device,
            )
            self._launch_project_body_wrench_to_torque(
                body_force=body_force,
                body_torque=body_torque,
                xipos=workspace["xipos"],
                subtree_com=subtree_com,
                cdof=cdof,
                mean=mean,
                batch_size=batch_size,
                device=device,
            )
            return mean

        kernel = make_muscle_dynamics_from_activation_sparse_kernel(use_passive_force)
        kernel_inputs: list[Any] = [
            wp.from_torch(workspace["transformed_waypoints"], dtype=wp.vec3),
            wp.from_torch(unit_force_magnitude, dtype=wp.float32),
        ]
        kernel_inputs.append(
            wp.from_torch(passive_force_magnitude, dtype=wp.float32)
            if use_passive_force
            else None
        )
        kernel_inputs.extend(
            [
                wp.from_torch(activation, dtype=wp.float32, requires_grad=False),
                wp.from_torch(subtree_com, dtype=wp.vec3),
                wp.from_torch(cdof, dtype=wp.float32),
                wp.from_torch(self.torque_active_entry_offsets, dtype=wp.int32),
                wp.from_torch(self.torque_active_segment_idx, dtype=wp.int32),
                wp.from_torch(self.torque_active_muscle_idx, dtype=wp.int32),
                wp.from_torch(self.torque_active_start_coeff, dtype=wp.float32),
                wp.from_torch(self.torque_active_end_coeff, dtype=wp.float32),
                wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                wp.from_torch(self.segment_start_root_body_idx_i32, dtype=wp.int32),
                wp.from_torch(self.segment_end_root_body_idx_i32, dtype=wp.int32),
                int(self.root_dof_count),
            ]
        )
        wp.launch(
            kernel,
            dim=(batch_size, self.torque_dim),
            inputs=kernel_inputs,
            outputs=[wp.from_torch(mean, dtype=wp.float32)],
            device=device,
        )
        return mean

    def _warp_muscle_dynamics_from_activation_backward(
        self,
        raw_context: torch.Tensor,
        grad_output: torch.Tensor,
        *,
        transformed_waypoints: torch.Tensor | None = None,
        unit_force_magnitude: torch.Tensor | None = None,
        scratch_generation: int | None = None,
    ) -> torch.Tensor:
        raw_context = raw_context.float()
        grad_output = grad_output.float().contiguous()
        if (
            scratch_generation is not None
            and scratch_generation != self._geometry_scratch_generation
        ):
            # The saved views alias geometry scratch that a later forward
            # rewrote; rebuild both from this backward's own context.
            transformed_waypoints = None
            unit_force_magnitude = None
        batch_size = raw_context.shape[0]
        use_native_lbs_backward = (
            self._uses_native_body_wrench_projection
            and self.lbs_body_segment_event_start_contiguous
            and self.native_lbs_body_wrench_direct_backward
        )
        missing_unit_force = (
            unit_force_magnitude is None or unit_force_magnitude.numel() == 0
        )
        missing_transformed_waypoints = (
            transformed_waypoints is None or transformed_waypoints.numel() == 0
        )
        if missing_unit_force and self.muscle_force_velocity_enabled:
            # The native backward rebuilds the unit force without the
            # force-velocity factor; the generic geometry path applies it.
            use_native_lbs_backward = False
        if use_native_lbs_backward:
            workspace, subtree_com, cdof = (
                self._prepare_warp_native_tile_backward_geometry(
                    raw_context,
                    compute_unit_force_magnitude=missing_unit_force,
                )
            )
            transformed_waypoints = None
            if missing_unit_force:
                unit_force_magnitude = workspace["unit_force_magnitude"]
        elif missing_transformed_waypoints or missing_unit_force:
            workspace, subtree_com, cdof = self._prepare_warp_waypoint_geometry(
                raw_context,
                include_passive_force_magnitude=False,
                compute_force_magnitudes=missing_unit_force,
            )
            transformed_waypoints = workspace["transformed_waypoints"]
            if missing_unit_force:
                unit_force_magnitude = workspace["unit_force_magnitude"]
        else:
            subtree_com, cdof = self._unpack_warp_muscle_dynamics_context(raw_context)
        grad_activation = torch.empty(
            batch_size,
            self.muscle_count,
            device=raw_context.device,
            dtype=torch.float32,
        )
        grad_activation.zero_()
        if self.muscle_active_segment_idx.numel() == 0:
            return grad_activation
        device = str(raw_context.device)
        if self._uses_attachment_projection and not self._waypoint_wrapping.count:
            workspace = self._get_warp_geometry_workspace(
                batch_size=batch_size,
                device=raw_context.device,
                minimal_native_lbs=use_native_lbs_backward,
            )
            xpos = raw_context[:, : self._xpos_width].reshape(
                batch_size, self.body_count, 3
            )
            xquat = raw_context[
                :, self._xpos_width : self._xpos_width + self._xquat_width
            ].reshape(batch_size, self.body_count, 4)
            wp.launch(
                _compute_lbs_body_adjoint_kernel,
                dim=(batch_size, self.body_count),
                inputs=[
                    wp.from_torch(grad_output, dtype=wp.float32, requires_grad=False),
                    wp.from_torch(cdof, dtype=wp.float32),
                    wp.from_torch(self.dof_subtree_mask, dtype=wp.float32),
                    int(self.body_count),
                    int(self.torque_dim),
                    int(self.root_dof_count),
                ],
                outputs=[
                    wp.from_torch(workspace["body_linear_adjoint"], dtype=wp.float32),
                    wp.from_torch(workspace["body_angular_adjoint"], dtype=wp.float32),
                ],
                device=device,
            )
            if use_native_lbs_backward:
                fixed1 = self.lbs_native_waypoint_attachment_arity == 1
                direct_backward_kernel = self._native_lbs_direct_backward_kernel
                if self.lbs_single_root_body_idx >= 0:
                    direct_backward_inputs: list[Any] = [
                        wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                        wp.from_torch(workspace["xquat_normalized"], dtype=wp.quat),
                        wp.from_torch(subtree_com, dtype=wp.vec3),
                        wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                    ]
                else:
                    direct_backward_inputs = [
                        wp.from_torch(workspace["xpos"], dtype=wp.vec3),
                        wp.from_torch(workspace["xquat_normalized"], dtype=wp.quat),
                        wp.from_torch(subtree_com, dtype=wp.vec3),
                        wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                        wp.from_torch(self.body_rootid_i32, dtype=wp.int32),
                    ]
                direct_backward_inputs.extend(
                    [
                        wp.from_torch(
                            workspace["body_linear_adjoint"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            workspace["body_angular_adjoint"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            self.apply_segment_muscle_idx_i32, dtype=wp.int32
                        ),
                        wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                        wp.from_torch(
                            self.lbs_waypoint_attachment_body2,
                            dtype=wp.vec2i,
                        ),
                        wp.from_torch(
                            self.lbs_waypoint_attachment_local_weight0,
                            dtype=wp.vec4,
                        ),
                    ]
                )
                if not fixed1:
                    direct_backward_inputs.append(
                        wp.from_torch(
                            self.lbs_waypoint_attachment_local_weight1,
                            dtype=wp.vec4,
                        )
                    )
                wp.launch(
                    direct_backward_kernel,
                    dim=(batch_size, int(self.apply_segment_muscle_idx_i32.numel())),
                    inputs=direct_backward_inputs,
                    outputs=[wp.from_torch(grad_activation, dtype=wp.float32)],
                    device=device,
                )
            elif (
                self.lbs_body_segment_event_start_contiguous
                and self.lbs_single_root_body_idx >= 0
                and self.lbs_waypoint_attachment_fixed2
            ):
                wp.launch(
                    make_accumulate_lbs_activation_grad_fixed2_start_from_body_adjoint_kernel(
                        int(self.lbs_single_root_body_idx),
                        bool(self.lbs_segment_start_is_segment_plus_muscle),
                    ),
                    dim=(batch_size, int(self.apply_segment_muscle_idx_i32.numel())),
                    inputs=[
                        wp.from_torch(transformed_waypoints, dtype=wp.vec3),
                        wp.from_torch(xpos, dtype=wp.vec3),
                        wp.from_torch(xquat, dtype=wp.quat),
                        wp.from_torch(subtree_com, dtype=wp.vec3),
                        wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                        wp.from_torch(
                            workspace["body_linear_adjoint"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            workspace["body_angular_adjoint"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            self.apply_segment_muscle_idx_i32, dtype=wp.int32
                        ),
                        wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                        wp.from_torch(
                            self.lbs_waypoint_attachment_body2,
                            dtype=wp.vec2i,
                        ),
                        wp.from_torch(
                            self.lbs_waypoint_attachment_local_weight0,
                            dtype=wp.vec4,
                        ),
                        wp.from_torch(
                            self.lbs_waypoint_attachment_local_weight1,
                            dtype=wp.vec4,
                        ),
                    ],
                    outputs=[wp.from_torch(grad_activation, dtype=wp.float32)],
                    device=device,
                )
            else:
                wp.launch(
                    _accumulate_lbs_activation_grad_from_body_adjoint_kernel,
                    dim=(batch_size, int(self.apply_segment_muscle_idx_i32.numel())),
                    inputs=[
                        wp.from_torch(transformed_waypoints, dtype=wp.vec3),
                        wp.from_torch(xpos, dtype=wp.vec3),
                        wp.from_torch(xquat, dtype=wp.quat),
                        wp.from_torch(subtree_com, dtype=wp.vec3),
                        wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                        wp.from_torch(self.body_rootid_i32, dtype=wp.int32),
                        wp.from_torch(
                            workspace["body_linear_adjoint"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            workspace["body_angular_adjoint"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            self.apply_segment_muscle_idx_i32, dtype=wp.int32
                        ),
                        wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                        wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_start_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_body_i32, dtype=wp.int32
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_local, dtype=wp.vec3
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_weight,
                            dtype=wp.float32,
                        ),
                    ],
                    outputs=[wp.from_torch(grad_activation, dtype=wp.float32)],
                    device=device,
                )
            return grad_activation
        if self.sparse_body_wrench_forward:
            workspace = self._get_warp_geometry_workspace(
                batch_size=batch_size,
                device=raw_context.device,
            )
            wp.launch(
                _compute_lbs_body_adjoint_kernel,
                dim=(batch_size, self.body_count),
                inputs=[
                    wp.from_torch(grad_output, dtype=wp.float32, requires_grad=False),
                    wp.from_torch(cdof, dtype=wp.float32),
                    wp.from_torch(self.dof_subtree_mask, dtype=wp.float32),
                    int(self.body_count),
                    int(self.torque_dim),
                    int(self.root_dof_count),
                ],
                outputs=[
                    wp.from_torch(workspace["body_linear_adjoint"], dtype=wp.float32),
                    wp.from_torch(workspace["body_angular_adjoint"], dtype=wp.float32),
                ],
                device=device,
            )
            if self.sparse_body_wrench_direct_backward:
                wp.launch(
                    _accumulate_activation_grad_direct_from_body_adjoint_kernel,
                    dim=(batch_size, int(self.apply_segment_muscle_idx_i32.numel())),
                    inputs=[
                        wp.from_torch(transformed_waypoints, dtype=wp.vec3),
                        wp.from_torch(subtree_com, dtype=wp.vec3),
                        wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                        wp.from_torch(self.body_rootid_i32, dtype=wp.int32),
                        wp.from_torch(
                            workspace["body_linear_adjoint"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            workspace["body_angular_adjoint"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            self.apply_segment_muscle_idx_i32, dtype=wp.int32
                        ),
                        wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                        wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                        wp.from_torch(
                            self.apply_segment_body_start_idx_i32, dtype=wp.int32
                        ),
                        wp.from_torch(
                            self.apply_segment_body_end_idx_i32, dtype=wp.int32
                        ),
                    ],
                    outputs=[wp.from_torch(grad_activation, dtype=wp.float32)],
                    device=device,
                )
            else:
                xpos = raw_context[:, : self._xpos_width].reshape(
                    batch_size, self.body_count, 3
                )
                xquat = raw_context[
                    :, self._xpos_width : self._xpos_width + self._xquat_width
                ].reshape(batch_size, self.body_count, 4)
                wp.launch(
                    _accumulate_lbs_activation_grad_from_body_adjoint_kernel,
                    dim=(batch_size, int(self.apply_segment_muscle_idx_i32.numel())),
                    inputs=[
                        wp.from_torch(transformed_waypoints, dtype=wp.vec3),
                        wp.from_torch(xpos, dtype=wp.vec3),
                        wp.from_torch(xquat, dtype=wp.quat),
                        wp.from_torch(subtree_com, dtype=wp.vec3),
                        wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                        wp.from_torch(self.body_rootid_i32, dtype=wp.int32),
                        wp.from_torch(
                            workspace["body_linear_adjoint"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            workspace["body_angular_adjoint"], dtype=wp.float32
                        ),
                        wp.from_torch(
                            self.apply_segment_muscle_idx_i32, dtype=wp.int32
                        ),
                        wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                        wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_start_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_body_i32,
                            dtype=wp.int32,
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_local, dtype=wp.vec3
                        ),
                        wp.from_torch(
                            self.muscle_waypoint_attachment_weight,
                            dtype=wp.float32,
                        ),
                    ],
                    outputs=[wp.from_torch(grad_activation, dtype=wp.float32)],
                    device=device,
                )
            return grad_activation
        wp.launch(
            _compute_muscle_dynamics_activation_grad_sparse_kernel,
            dim=(batch_size, self.muscle_count),
            inputs=[
                wp.from_torch(transformed_waypoints, dtype=wp.vec3),
                wp.from_torch(unit_force_magnitude, dtype=wp.float32),
                wp.from_torch(subtree_com, dtype=wp.vec3),
                wp.from_torch(cdof, dtype=wp.float32),
                wp.from_torch(self.muscle_active_entry_offsets, dtype=wp.int32),
                wp.from_torch(self.muscle_active_segment_idx, dtype=wp.int32),
                wp.from_torch(self.muscle_active_torque_idx, dtype=wp.int32),
                wp.from_torch(self.muscle_active_start_coeff, dtype=wp.float32),
                wp.from_torch(self.muscle_active_end_coeff, dtype=wp.float32),
                wp.from_torch(self.apply_segment_start_idx_i32, dtype=wp.int32),
                wp.from_torch(self.apply_segment_end_idx_i32, dtype=wp.int32),
                wp.from_torch(self.segment_start_root_body_idx_i32, dtype=wp.int32),
                wp.from_torch(self.segment_end_root_body_idx_i32, dtype=wp.int32),
                int(self.root_dof_count),
                wp.from_torch(grad_output, dtype=wp.float32, requires_grad=False),
            ],
            outputs=[wp.from_torch(grad_activation, dtype=wp.float32)],
            device=device,
        )
        return grad_activation

    def _compute_full_muscle_dynamics_contribution(
        self,
        unit_force_magnitude: torch.Tensor,
        seg_start: torch.Tensor | None,
        seg_end: torch.Tensor | None,
        seg_dir: torch.Tensor | None,
        xipos: torch.Tensor,
        subtree_com: torch.Tensor,
        cdof: torch.Tensor,
    ) -> torch.Tensor:
        """Return the full contribution matrix C of shape (B, torque_dim, muscle_count).

        C[b, d, m] is the contribution of muscle m to DOF d per unit physical
        activation.  Unlike _compute_muscle_dynamics_bounds_from_segments this
        method retains the full (torque_dim × muscle_count) matrix rather than
        collapsing it to lower/upper bounds.
        """
        batch_size = unit_force_magnitude.shape[0]
        C = torch.zeros(
            batch_size,
            self.torque_dim,
            self.muscle_count,
            device=unit_force_magnitude.device,
            dtype=unit_force_magnitude.dtype,
        )
        terms = self._build_segment_projection_terms(
            unit_force_magnitude,
            seg_start,
            seg_end,
            seg_dir,
            xipos,
            subtree_com,
        )
        if terms is None:
            return C

        muscle_contrib = torch.zeros(
            batch_size,
            self.bound_chunk_size,
            self.muscle_count,
            device=unit_force_magnitude.device,
            dtype=unit_force_magnitude.dtype,
        )
        for torque_start_idx in range(0, self.torque_dim, self.bound_chunk_size):
            torque_end_idx = min(
                torque_start_idx + self.bound_chunk_size, self.torque_dim
            )
            chunk_len = torque_end_idx - torque_start_idx
            torque_slice = slice(torque_start_idx, torque_end_idx)

            segment_contrib = self._segment_projection_chunk(
                terms, cdof, torque_start_idx, torque_end_idx
            )
            muscle_contrib_chunk = muscle_contrib[:, :chunk_len]
            muscle_contrib_chunk.zero_()
            muscle_contrib_chunk.scatter_add_(
                2, terms.muscle_index[:, :chunk_len], segment_contrib
            )
            C[:, torque_slice].copy_(muscle_contrib_chunk)
        return C

    def compute_activation_torque_jacobian(
        self, raw_context: torch.Tensor
    ) -> torch.Tensor:
        """Return the physical activation Jacobian used by the MASS controller.

        Returns J_act of shape (B, torque_dim, muscle_count) — or
        (B, active_output_dim, muscle_count) when active_torque_indices is set.
        """
        self._require_supported_physical_geometry()
        raw_context = raw_context.float()
        if self._can_use_warp_torque_jacobian(raw_context):
            J = self._compute_full_muscle_dynamics_contribution_warp(raw_context)
        else:
            geometry = self._compute_waypoint_geometry(raw_context)
            unit_force_magnitude = geometry.unit_force_magnitude
            if self.muscle_force_velocity_enabled:
                unit_force_magnitude = (
                    unit_force_magnitude
                    * self._waypoint_force_velocity_multiplier(raw_context, geometry)
                )
            if self.uses_lbs_waypoint_attachments:
                J = self._compute_lbs_muscle_dynamics_contribution_from_entries(
                    unit_force_magnitude,
                    geometry,
                )
            else:
                J = self._compute_full_muscle_dynamics_contribution(
                    unit_force_magnitude=unit_force_magnitude,
                    seg_start=geometry.seg_start,
                    seg_end=geometry.seg_end,
                    seg_dir=geometry.seg_dir,
                    xipos=geometry.xipos,
                    subtree_com=geometry.subtree_com,
                    cdof=geometry.cdof,
                )
        if self.active_torque_indices is not None:
            J = J.index_select(1, self.active_torque_indices)
        return J

    def compute_activation_torque_bounds(
        self,
        raw_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return feasible torque bounds without materializing the full Jacobian.

        The bounds are over physical activation in [0, 1] and match the lower/upper
        values obtained from compute_activation_torque_jacobian(...), but the
        waypoint backend reduces each torque chunk immediately.
        """
        self._require_supported_physical_geometry()
        raw_context = raw_context.float()
        if raw_context.shape[-1] != self.context_dim:
            raise ValueError(
                "Unexpected raw muscle context width: "
                f"expected {self.context_dim}, got {raw_context.shape[-1]}."
            )
        if self._supports_warp_muscle_dynamics_forward(raw_context.device):
            return self._compute_activation_torque_bounds_warp(
                raw_context,
                select_active=True,
            )
        geometry = self._compute_waypoint_geometry(raw_context)
        unit_force_magnitude = geometry.unit_force_magnitude
        if self.muscle_force_velocity_enabled:
            unit_force_magnitude = (
                unit_force_magnitude
                * self._waypoint_force_velocity_multiplier(raw_context, geometry)
            )
        if self.uses_lbs_waypoint_attachments:
            lower, upper = self._compute_lbs_muscle_dynamics_bounds_from_entries(
                unit_force_magnitude,
                geometry,
            )
        else:
            lower, upper = self._compute_muscle_dynamics_bounds_from_segments(
                unit_force_magnitude=unit_force_magnitude,
                seg_start=geometry.seg_start,
                seg_end=geometry.seg_end,
                seg_dir=geometry.seg_dir,
                xipos=geometry.xipos,
                subtree_com=geometry.subtree_com,
                cdof=geometry.cdof,
            )
        if self.active_torque_indices is not None:
            lower = lower.index_select(1, self.active_torque_indices)
            upper = upper.index_select(1, self.active_torque_indices)
        return lower, upper


__all__ = ["MuscleDynamicsLayer"]
