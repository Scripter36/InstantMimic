from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import warp as wp

from instantmimic.envs.control_pipeline import (
    ActionScaleSnapshot,
    PolicyTransformSemantics,
)
from instantmimic.envs.controls.fast_muscle.buffer_ops import (
    copy_2d_,
    masked_zero_2d_,
    zero_,
)
from instantmimic.muscle.muscle_dynamics import MuscleDynamicsLayer
from instantmimic.muscle.muscle_qfrc_projector import (
    apply_muscle_implicit_qderiv_gather,
    apply_precomputed_muscle_implicit_qderiv_kernel,
    build_body_ancestor_dof_lists,
    build_dof_subtree_mask,
    build_lbs_active_attachment_lists,
    build_lbs_active_attachment_projection_metadata,
    build_lbs_active_idx_body_vec2,
    build_lbs_active_local_coeff_vec4,
    build_lbs_torque_forward_chunks,
    build_muscle_dof_sparse_entries,
    build_muscle_implicit_qderiv_gather,
    build_muscle_sparse_torque_entries,
    build_torque_sparse_lbs_segment_entries,
    build_torque_sparse_segment_entries,
    clear_muscle_sparse_moment_kernel,
    clip_torque_feasible_bounds_kernel,
    eval_lbs_muscle_sparse_moment_atomic_kernel,
    eval_lbs_muscle_sparse_moment_kernel,
    eval_muscle_force_magnitude_kernel,
    eval_muscle_force_velocity_state_kernel,
    eval_muscle_lbs_qfrc_kernel,
    eval_muscle_lbs_qfrc_pair_vec4_native_tiled_partials_kernel,
    eval_muscle_qfrc_kernel,
    launch_muscle_sparse_moment,
    prepare_muscle_implicit_qderiv_pairs,
    project_muscle_sparse_moment_qfrc_kernel,
    reduce_lbs_qfrc_chunk_partials_kernel,
    transform_lbs_waypoints_kernel,
    transform_waypoints_kernel,
)
from instantmimic.muscle.parsing.mjcf import (
    cache_key,
    collect_xml_dependency_signatures,
    file_signature,
)
from instantmimic.muscle.runtime_specs import (
    MuscleDimensionsSpec,
    MuscleLayerRuntimeSpec,
    MuscleMechanicsRuntimeSpec,
    MuscleOutputSpec,
    MuscleProjectionRuntimeSpec,
    MuscleTopologySpec,
)
from instantmimic.muscle.waypoint_wrap import (
    force_from_lengths_kernel,
    force_velocity_activation_kernel,
    path_lengths_kernel,
)
from instantmimic.utils.startup_profiler import startup_nvtx_wrapped

logger = logging.getLogger(__name__)


class FastMuscleDynamicsMixin:
    @startup_nvtx_wrapped("fast_muscle:build_muscle_segment_state")
    def _build_muscle_segment_state(self) -> None:
        waypoint_start = self.muscle_waypoint_start_i32.detach().cpu().to(torch.long)
        waypoint_body = self.muscle_waypoint_body_i32.detach().cpu().to(torch.long)
        segment_counts = (waypoint_start[1:] - waypoint_start[:-1] - 1).clamp_min(0)
        total_segments = int(segment_counts.sum().item())
        if total_segments > 0:
            muscle_idx_cpu = torch.repeat_interleave(
                torch.arange(self.muscle_count, dtype=torch.long),
                segment_counts,
            )
            segment_offsets = torch.cumsum(segment_counts, dim=0) - segment_counts
            local_segment_idx = torch.arange(total_segments, dtype=torch.long)
            local_segment_idx -= torch.repeat_interleave(
                segment_offsets,
                segment_counts,
            )
            segment_start_idx_cpu = (
                torch.repeat_interleave(waypoint_start[:-1], segment_counts)
                + local_segment_idx
            )
            segment_end_idx_cpu = segment_start_idx_cpu + 1
            body_start_idx_cpu = waypoint_body.index_select(0, segment_start_idx_cpu)
            body_end_idx_cpu = waypoint_body.index_select(0, segment_end_idx_cpu)
            if bool(getattr(self, "_uses_lbs_waypoint_attachments", False)):
                apply_mask = torch.ones(total_segments, dtype=torch.bool)
            else:
                apply_mask = body_start_idx_cpu != body_end_idx_cpu
        else:
            muscle_idx_cpu = torch.zeros(0, dtype=torch.long)
            segment_start_idx_cpu = torch.zeros(0, dtype=torch.long)
            segment_end_idx_cpu = torch.zeros(0, dtype=torch.long)
            body_start_idx_cpu = torch.zeros(0, dtype=torch.long)
            body_end_idx_cpu = torch.zeros(0, dtype=torch.long)
            apply_mask = torch.zeros(0, dtype=torch.bool)

        self._muscle_all_segment_muscle_idx = muscle_idx_cpu.to(
            device=self.device,
            dtype=torch.long,
        )
        self._muscle_all_segment_start_idx = segment_start_idx_cpu.to(
            device=self.device,
            dtype=torch.long,
        )
        self._muscle_apply_segment_muscle_idx = muscle_idx_cpu[apply_mask].to(
            device=self.device,
            dtype=torch.long,
        )
        self._muscle_apply_segment_start_idx = segment_start_idx_cpu[apply_mask].to(
            device=self.device,
            dtype=torch.long,
        )
        self._muscle_apply_segment_end_idx = segment_end_idx_cpu[apply_mask].to(
            device=self.device,
            dtype=torch.long,
        )
        self._muscle_apply_segment_body_start_idx = body_start_idx_cpu[apply_mask].to(
            device=self.device,
            dtype=torch.long,
        )
        self._muscle_apply_segment_body_end_idx = body_end_idx_cpu[apply_mask].to(
            device=self.device,
            dtype=torch.long,
        )

    def _build_muscle_layer_contract(
        self,
        *,
        obs_context_dim: int | None = None,
        include_passive_force_in_output: bool | None = None,
        select_active_torque_dofs: bool = True,
    ) -> MuscleLayerRuntimeSpec:
        """Build the physical MASS helper contract.

        ``select_active_torque_dofs=False`` keeps every
        non-root torque coordinate in the output even when a zero torque scale
        masks some of them out of the policy action.
        """
        muscle_settings = self.settings.muscle
        muscle_dynamics = muscle_settings.dynamics
        lbs = muscle_settings.lbs
        body_wrench = lbs.body_wrench
        body_wrench_native_tile = body_wrench.native_tile
        active_force = muscle_settings.active_force
        passive_force = muscle_settings.passive_force
        force_velocity = muscle_settings.force_velocity
        action_dim = int(self.model.nv - self._root_dof_count)
        resolved_obs_context_dim = (
            int(obs_context_dim)
            if obs_context_dim is not None
            else int(self.helper_obs_context_dim)
        )

        include_passive = (
            bool(self._muscle_include_passive != 0.0)
            if include_passive_force_in_output is None
            else bool(include_passive_force_in_output)
        )
        layer = MuscleLayerRuntimeSpec(
            dimensions=MuscleDimensionsSpec(
                obs_context_dim=resolved_obs_context_dim,
                body_count=int(self.model.nbody),
                full_torque_dim=int(self.model.nv),
                torque_dim=action_dim,
                muscle_count=int(self.muscle_count),
            ),
            output=MuscleOutputSpec(
                action_dim=action_dim,
                active_torque_indices=(
                    self._active_torque_indices.to(torch.long)
                    if select_active_torque_dofs and self._has_masked_torque_dofs
                    else None
                ),
                bound_chunk_size=self.settings.effective.bounds.chunk_size,
            ),
            mechanics=MuscleMechanicsRuntimeSpec(
                dynamics_model=str(muscle_dynamics.model),
                force_scale=float(muscle_settings.force_scale),
                active_force_type=str(active_force.type),
                active_force_gamma=float(active_force.gamma),
                active_force_lmin=float(active_force.lmin),
                active_force_lmax=float(active_force.lmax),
                force_velocity_enabled=bool(
                    self._uses_mass_coordinator and force_velocity.enabled
                ),
                force_velocity_vmax=float(force_velocity.vmax),
                force_velocity_fvmax=float(force_velocity.fvmax),
                passive_force_type=str(passive_force.type),
                passive_force_k_pe=float(passive_force.k_pe),
                passive_force_e_mo=float(passive_force.e_mo),
                passive_force_lmax=float(passive_force.lmax),
                passive_force_fpmax=float(passive_force.fpmax),
                length_norm_min=float(self._muscle_length_norm_min),
                length_norm_max=float(self._muscle_length_norm_max),
                include_passive_force_in_output=include_passive,
            ),
            projection=MuscleProjectionRuntimeSpec(
                geometry_backend=str(self._geometry_backend),
                sparse_body_wrench_forward=bool(lbs.sparse_forward),
                sparse_body_wrench_direct=bool(lbs.sparse_direct),
                sparse_body_wrench_direct_backward=bool(lbs.sparse_direct_backward),
                lbs_forward_reduction=lbs.forward_reduction,
                lbs_forward_chunk_size=lbs.forward_chunk_size,
                lbs_forward_tile_block_dim=lbs.forward_tile_block_dim,
                lbs_body_event_chunk_size=lbs.body_event_chunk_size,
                lbs_muscle_length_by_muscle_min_batch=(
                    lbs.muscle_length_by_muscle_min_batch
                ),
                lbs_attachment_layout=lbs.attachment_layout,
                lbs_body_wrench_forward_mode=muscle_settings.body_wrench_forward_mode,
                lbs_body_wrench_native_tile_muscle_cap=(
                    body_wrench_native_tile.muscle_cap
                ),
                lbs_body_wrench_native_tile_segment_cap=(
                    body_wrench_native_tile.segment_cap
                ),
                lbs_body_wrench_native_tile_event_cap=(
                    body_wrench_native_tile.event_cap
                ),
                lbs_body_wrench_native_tile_body_cap=(body_wrench_native_tile.body_cap),
                static_metadata_cache_key=(
                    self._muscle_dynamics_layer_static_metadata_cache_key()
                ),
            ),
            topology=MuscleTopologySpec(
                wraps=self._waypoint_wrap_spec,
                muscle_params=self.muscle_params,
                muscle_rest_length=self.muscle_rest_length,
                muscle_force_velocity_length_scale=(
                    self.muscle_force_velocity_length_scale
                ),
                muscle_waypoints_local=self.muscle_waypoints_local,
                muscle_waypoint_body=self.muscle_waypoint_body_i32.to(torch.long),
                uses_lbs_waypoint_attachments=bool(
                    getattr(self, "_uses_lbs_waypoint_attachments", False)
                ),
                muscle_waypoint_attachment_start=(
                    self.muscle_waypoint_attachment_start_i32.to(torch.long)
                ),
                muscle_waypoint_attachment_body=(
                    self.muscle_waypoint_attachment_body_i32.to(torch.long)
                ),
                muscle_waypoint_attachment_local=self.muscle_waypoint_attachment_local,
                muscle_waypoint_attachment_weight=(
                    self.muscle_waypoint_attachment_weight
                ),
                all_segment_muscle_idx=self._muscle_all_segment_muscle_idx,
                all_segment_start_idx=self._muscle_all_segment_start_idx,
                apply_segment_muscle_idx=self._muscle_apply_segment_muscle_idx,
                apply_segment_start_idx=self._muscle_apply_segment_start_idx,
                apply_segment_end_idx=self._muscle_apply_segment_end_idx,
                apply_segment_body_start_idx=(
                    self._muscle_apply_segment_body_start_idx
                ),
                apply_segment_body_end_idx=self._muscle_apply_segment_body_end_idx,
                body_parentid=wp.to_torch(self.mjw_model.body_parentid)
                .detach()
                .clone()
                .to(torch.long),
                body_rootid=wp.to_torch(self.mjw_model.body_rootid)
                .detach()
                .clone()
                .to(torch.long),
                dof_bodyid=wp.to_torch(self.mjw_model.dof_bodyid)
                .detach()
                .clone()
                .to(torch.long),
            ),
        )
        return layer

    def _build_muscle_helper_runtime_spec(
        self,
        *,
        obs_context_dim: int,
        include_passive_force_in_output: bool | None = None,
        select_active_torque_dofs: bool = True,
    ) -> MuscleLayerRuntimeSpec:
        return self._build_muscle_layer_contract(
            obs_context_dim=obs_context_dim,
            include_passive_force_in_output=include_passive_force_in_output,
            select_active_torque_dofs=select_active_torque_dofs,
        )

    @startup_nvtx_wrapped("fast_muscle:initialize_torque_bound_state")
    def _initialize_torque_bound_state(self, *, force_projector: bool = False) -> None:
        lbs = self.settings.muscle.lbs
        self._torque_action_scale_inv = torch.reciprocal(
            torch.clamp(self._torque_action_scale, min=1.0e-6)
        )
        self._lbs_qfrc_forward_reduction = lbs.forward_reduction
        self._lbs_qfrc_forward_chunk_size = lbs.forward_chunk_size
        self._lbs_qfrc_forward_tile_block_dim = lbs.forward_tile_block_dim
        self._lbs_qfrc_forward_segment_idx_i32 = None
        self._lbs_qfrc_forward_muscle_idx_i32 = None
        self._lbs_qfrc_chunk_offsets_i32 = None
        self._lbs_qfrc_chunk_torque_idx_i32 = None
        self._lbs_qfrc_chunk_entry_start_i32 = None
        self._lbs_qfrc_chunk_entry_end_i32 = None
        self._lbs_qfrc_active_attach_offsets_i32 = None
        self._lbs_qfrc_active_attach_idx_body_i32 = None
        self._lbs_qfrc_active_attach_local_coeff = None
        self._lbs_qfrc_chunk_partial = None
        self._lbs_qfrc_torque_chunk_count = 0
        body_rootid = wp.to_torch(self.mjw_model.body_rootid).detach().clone()
        self._body_rootid_i32 = body_rootid.to(device=self.device, dtype=torch.int32)
        build_projector = bool(
            force_projector or self._needs_env_muscle_qfrc_projector_state()
        )
        self._env_muscle_qfrc_projector_ready = build_projector
        if not build_projector:
            self._torque_dof_subtree_mask = torch.zeros(
                0, 0, device=self.device, dtype=torch.float32
            )
            self._apply_segment_start_wp_i32 = None
            self._apply_segment_end_wp_i32 = None
            self._apply_segment_body_start_i32 = None
            self._apply_segment_body_end_i32 = None
            self._segment_start_root_body_idx_i32 = None
            self._segment_end_root_body_idx_i32 = None
            self._torque_active_entry_offsets_i32 = None
            self._torque_active_segment_idx_i32 = None
            self._torque_active_muscle_idx_i32 = None
            self._torque_active_start_coeff = None
            self._torque_active_end_coeff = None
            return

        body_parentid = wp.to_torch(self.mjw_model.body_parentid).detach().clone()
        dof_bodyid = wp.to_torch(self.mjw_model.dof_bodyid).detach().clone()
        dof_subtree_mask = build_dof_subtree_mask(
            body_parentid=body_parentid.to(torch.long),
            dof_bodyid=dof_bodyid.to(torch.long),
            body_count=int(self.model.nbody),
            full_torque_dim=int(self.model.nv),
        ).to(device=self.device)
        self._torque_dof_subtree_mask = dof_subtree_mask
        (
            self._body_cvel_dof_offsets_i32,
            self._body_cvel_dof_idx_i32,
        ) = build_body_ancestor_dof_lists(dof_subtree_mask)
        if not self._needs_muscle_qfrc_projector_state():
            self._apply_segment_start_wp_i32 = None
            self._apply_segment_end_wp_i32 = None
            self._apply_segment_body_start_i32 = None
            self._apply_segment_body_end_i32 = None
            self._segment_start_root_body_idx_i32 = None
            self._segment_end_root_body_idx_i32 = None
            self._torque_active_entry_offsets_i32 = None
            self._torque_active_segment_idx_i32 = None
            self._torque_active_muscle_idx_i32 = None
            self._torque_active_start_coeff = None
            self._torque_active_end_coeff = None
            return
        self._apply_segment_start_wp_i32 = self._muscle_apply_segment_start_idx.to(
            torch.int32
        )
        self._apply_segment_end_wp_i32 = self._muscle_apply_segment_end_idx.to(
            torch.int32
        )
        self._apply_segment_body_start_i32 = (
            self._muscle_apply_segment_body_start_idx.to(torch.int32)
        )
        self._apply_segment_body_end_i32 = self._muscle_apply_segment_body_end_idx.to(
            torch.int32
        )
        self._segment_start_root_body_idx_i32 = body_rootid.index_select(
            0, self._muscle_apply_segment_body_start_idx
        ).to(device=self.device, dtype=torch.int32)
        self._segment_end_root_body_idx_i32 = body_rootid.index_select(
            0, self._muscle_apply_segment_body_end_idx
        ).to(device=self.device, dtype=torch.int32)
        if bool(getattr(self, "_uses_lbs_waypoint_attachments", False)):
            (
                torque_active_entry_offsets,
                _torque_active_torque_idx,
                torque_active_segment_idx,
                torque_active_muscle_idx,
            ) = build_torque_sparse_lbs_segment_entries(
                torque_dof_subtree_mask=dof_subtree_mask[self._root_dof_count :],
                waypoint_attachment_start=self.muscle_waypoint_attachment_start_i32,
                waypoint_attachment_body=self.muscle_waypoint_attachment_body_i32,
                apply_segment_start_idx=self._muscle_apply_segment_start_idx,
                apply_segment_end_idx=self._muscle_apply_segment_end_idx,
                apply_segment_muscle_idx=self._muscle_apply_segment_muscle_idx,
            )
            torque_active_start_coeff = torch.zeros_like(
                torque_active_segment_idx, dtype=torch.float32
            )
            torque_active_end_coeff = torch.zeros_like(
                torque_active_segment_idx, dtype=torch.float32
            )
            (
                active_attach_offsets,
                active_attach_body,
                active_attach_local,
                active_attach_coeff_weight,
            ) = build_lbs_active_attachment_lists(
                dof_subtree_mask=dof_subtree_mask,
                attachment_start=self.muscle_waypoint_attachment_start_i32,
                attachment_body=self.muscle_waypoint_attachment_body_i32,
                attachment_local=self.muscle_waypoint_attachment_local,
                attachment_weight=self.muscle_waypoint_attachment_weight,
                root_dof_count=int(self._root_dof_count),
                torque_dim=int(self.model.nv - self._root_dof_count),
                waypoint_count=int(self.muscle_waypoints_local.shape[0]),
            )
            (
                _forward_torque_idx,
                forward_segment_idx,
                forward_muscle_idx,
                chunk_offsets,
                chunk_torque_idx,
                chunk_entry_start,
                chunk_entry_end,
            ) = build_lbs_torque_forward_chunks(
                torque_active_entry_offsets=torque_active_entry_offsets,
                torque_active_segment_idx=torque_active_segment_idx,
                torque_active_muscle_idx=torque_active_muscle_idx,
                torque_dim=int(self.model.nv - self._root_dof_count),
                chunk_size=int(self._lbs_qfrc_forward_chunk_size),
            )
            active_attach_waypoint_idx, _active_chunk_offsets, _active_chunk_info = (
                build_lbs_active_attachment_projection_metadata(
                    active_attach_offsets=active_attach_offsets,
                    torque_dim=int(self.model.nv - self._root_dof_count),
                    waypoint_count=int(self.muscle_waypoints_local.shape[0]),
                    chunk_size=int(self._lbs_qfrc_forward_chunk_size),
                )
            )
            active_attach_idx_body = build_lbs_active_idx_body_vec2(
                active_attach_waypoint_idx,
                active_attach_body,
            )
            active_attach_local_coeff = build_lbs_active_local_coeff_vec4(
                active_attach_local,
                active_attach_coeff_weight,
            )
            self._lbs_qfrc_forward_segment_idx_i32 = forward_segment_idx.to(
                device=self.device, dtype=torch.int32
            )
            self._lbs_qfrc_forward_muscle_idx_i32 = forward_muscle_idx.to(
                device=self.device, dtype=torch.int32
            )
            self._lbs_qfrc_chunk_offsets_i32 = chunk_offsets.to(
                device=self.device, dtype=torch.int32
            )
            self._lbs_qfrc_chunk_torque_idx_i32 = chunk_torque_idx.to(
                device=self.device, dtype=torch.int32
            )
            self._lbs_qfrc_chunk_entry_start_i32 = chunk_entry_start.to(
                device=self.device, dtype=torch.int32
            )
            self._lbs_qfrc_chunk_entry_end_i32 = chunk_entry_end.to(
                device=self.device, dtype=torch.int32
            )
            self._lbs_qfrc_active_attach_offsets_i32 = active_attach_offsets.to(
                device=self.device, dtype=torch.int32
            )
            self._lbs_qfrc_active_attach_idx_body_i32 = active_attach_idx_body.to(
                device=self.device, dtype=torch.int32
            )
            self._lbs_qfrc_active_attach_local_coeff = active_attach_local_coeff.to(
                device=self.device, dtype=torch.float32
            )
            self._lbs_qfrc_torque_chunk_count = int(chunk_torque_idx.numel())
            self._lbs_qfrc_chunk_partial = torch.empty(
                (self.num_envs, self._lbs_qfrc_torque_chunk_count),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            segment_start_dof_mask = dof_subtree_mask[
                self._root_dof_count :,
                self._muscle_apply_segment_body_start_idx,
            ]
            segment_end_dof_mask = dof_subtree_mask[
                self._root_dof_count :,
                self._muscle_apply_segment_body_end_idx,
            ]
            (
                torque_active_entry_offsets,
                _torque_active_torque_idx,
                torque_active_segment_idx,
                torque_active_muscle_idx,
                torque_active_start_coeff,
                torque_active_end_coeff,
            ) = build_torque_sparse_segment_entries(
                segment_start_dof_mask=segment_start_dof_mask,
                segment_end_dof_mask=segment_end_dof_mask,
                apply_segment_muscle_idx=self._muscle_apply_segment_muscle_idx,
            )
            (
                _muscle_active_entry_offsets,
                _muscle_active_segment_idx,
                _muscle_active_torque_idx,
                _muscle_active_start_coeff,
                _muscle_active_end_coeff,
            ) = build_muscle_sparse_torque_entries(
                torque_entry_offsets=torque_active_entry_offsets,
                torque_segment_idx=torque_active_segment_idx,
                torque_start_coeff=torque_active_start_coeff,
                torque_end_coeff=torque_active_end_coeff,
                apply_segment_muscle_idx=self._muscle_apply_segment_muscle_idx,
                muscle_count=self.muscle_count,
            )
        self._torque_active_entry_offsets_i32 = torque_active_entry_offsets.to(
            device=self.device, dtype=torch.int32
        )
        self._torque_active_segment_idx_i32 = torque_active_segment_idx.to(
            device=self.device, dtype=torch.int32
        )
        self._torque_active_muscle_idx_i32 = torque_active_muscle_idx.to(
            device=self.device, dtype=torch.int32
        )
        self._torque_active_start_coeff = torque_active_start_coeff.to(
            device=self.device, dtype=torch.float32
        )
        self._torque_active_end_coeff = torque_active_end_coeff.to(
            device=self.device, dtype=torch.float32
        )

    def _ensure_env_muscle_qfrc_projector_state(self) -> None:
        # Its own flag: the segment tables stay None when no projector needs
        # them, and per-call rebuilds would sync with the host every step.
        if getattr(self, "_env_muscle_qfrc_projector_ready", False):
            return
        self._initialize_torque_bound_state(force_projector=True)

    @startup_nvtx_wrapped("fast_muscle:initialize_force_velocity_state")
    def _initialize_muscle_force_velocity_state(self) -> None:
        self._muscle_force_velocity_enabled = bool(
            self.settings.muscle.force_velocity.enabled
        )
        if not self._muscle_force_velocity_enabled:
            return

        self._ensure_env_muscle_qfrc_projector_state()
        metadata = build_muscle_dof_sparse_entries(
            torque_active_entry_offsets=self._torque_active_entry_offsets_i32,
            torque_active_segment_idx=self._torque_active_segment_idx_i32,
            torque_active_muscle_idx=self._torque_active_muscle_idx_i32,
            torque_active_start_coeff=self._torque_active_start_coeff,
            torque_active_end_coeff=self._torque_active_end_coeff,
            muscle_count=self.muscle_count,
        )

        def device_i32(value: torch.Tensor) -> torch.Tensor:
            return value.to(device=self.device, dtype=torch.int32)

        self._muscle_dof_offsets_i32 = device_i32(metadata.muscle_dof_offsets)
        self._muscle_dof_muscle_idx_i32 = device_i32(metadata.muscle_dof_muscle_idx)
        self._muscle_dof_torque_idx_i32 = device_i32(metadata.muscle_dof_torque_idx)
        self._muscle_dof_segment_offsets_i32 = device_i32(
            metadata.muscle_dof_segment_offsets
        )
        self._muscle_dof_segment_idx_i32 = device_i32(metadata.segment_idx)
        segment_counts = (
            metadata.muscle_dof_segment_offsets[1:]
            - metadata.muscle_dof_segment_offsets[:-1]
        ).to(torch.long)
        self._muscle_dof_sparse_entry_idx_i32 = device_i32(
            torch.repeat_interleave(
                torch.arange(
                    int(metadata.muscle_dof_torque_idx.numel()),
                    dtype=torch.int32,
                ),
                segment_counts,
            )
        )
        self._muscle_dof_start_coeff = metadata.start_coeff.to(
            device=self.device, dtype=torch.float32
        )
        self._muscle_dof_end_coeff = metadata.end_coeff.to(
            device=self.device, dtype=torch.float32
        )
        self._muscle_implicit_pair_count = int(metadata.pair_entry_i.numel())
        prepared_qderiv = prepare_muscle_implicit_qderiv_pairs(
            pair_entry_i=metadata.pair_entry_i,
            pair_entry_j=metadata.pair_entry_j,
            pair_muscle_idx=metadata.pair_muscle_idx,
            muscle_dof_torque_idx=metadata.muscle_dof_torque_idx,
            matrix_elemid=wp.to_torch(self.mjw_model.M_elemid),
            root_dof_count=int(self._root_dof_count),
        )
        self._muscle_implicit_dropped_pair_count = prepared_qderiv.dropped_pair_count
        self._muscle_implicit_prepared_pair_count = int(
            prepared_qderiv.pair_elem_id.numel()
        )
        self._muscle_implicit_pair_elem_id_i32 = device_i32(
            prepared_qderiv.pair_elem_id
        )
        self._muscle_dof_pair_entry_i_i32 = device_i32(prepared_qderiv.pair_entry_i)
        self._muscle_dof_pair_entry_j_i32 = device_i32(prepared_qderiv.pair_entry_j)
        self._muscle_dof_pair_muscle_idx_i32 = device_i32(
            prepared_qderiv.pair_muscle_idx
        )
        # Atomic per-pair scatter only when the gather tables exceed its limits.
        self._muscle_implicit_qderiv_gather = build_muscle_implicit_qderiv_gather(
            prepared_qderiv,
            entry_muscle_idx=metadata.muscle_dof_muscle_idx,
            device=self.device,
        )
        torque_dim = int(self.model.nv - self._root_dof_count)
        torque_order = torch.argsort(
            metadata.muscle_dof_torque_idx.to(torch.long),
            stable=True,
        )
        torque_counts = torch.bincount(
            metadata.muscle_dof_torque_idx.to(torch.long),
            minlength=torque_dim,
        )
        torque_offsets = torch.zeros(torque_dim + 1, dtype=torch.int32)
        torque_offsets[1:] = torch.cumsum(torque_counts, dim=0).to(torch.int32)
        self._muscle_dof_torque_entry_offsets_i32 = device_i32(torque_offsets)
        self._muscle_dof_torque_entry_idx_i32 = device_i32(torque_order)

        entry_count = int(self._muscle_dof_torque_idx_i32.numel())
        self._muscle_sparse_moment = torch.empty(
            (self.num_envs, entry_count),
            device=self.device,
            dtype=torch.float32,
        )
        self._muscle_force_velocity_effective_activation = torch.empty_like(
            self.muscle_length
        )
        self._muscle_length_velocity = torch.empty_like(self.muscle_length)
        self._muscle_force_velocity_multiplier = torch.empty_like(self.muscle_length)
        # The implicit derivative hook can run before the first replay payload.
        # A direct torque action has no activation-derived force-velocity slope.
        self._muscle_force_velocity_slope = torch.zeros_like(self.muscle_length)

        self._muscle_sparse_moment_wp = wp.from_torch(
            self._muscle_sparse_moment, dtype=wp.float32
        )
        self._muscle_force_velocity_effective_activation_wp = wp.from_torch(
            self._muscle_force_velocity_effective_activation,
            dtype=wp.float32,
        )
        self._muscle_length_velocity_wp = wp.from_torch(
            self._muscle_length_velocity, dtype=wp.float32
        )
        self._muscle_force_velocity_multiplier_wp = wp.from_torch(
            self._muscle_force_velocity_multiplier, dtype=wp.float32
        )
        self._muscle_force_velocity_slope_wp = wp.from_torch(
            self._muscle_force_velocity_slope, dtype=wp.float32
        )
        self._muscle_dof_offsets_i32_wp = wp.from_torch(
            self._muscle_dof_offsets_i32, dtype=wp.int32
        )
        self._muscle_dof_muscle_idx_i32_wp = wp.from_torch(
            self._muscle_dof_muscle_idx_i32, dtype=wp.int32
        )
        self._muscle_dof_torque_idx_i32_wp = wp.from_torch(
            self._muscle_dof_torque_idx_i32, dtype=wp.int32
        )
        self._muscle_dof_segment_offsets_i32_wp = wp.from_torch(
            self._muscle_dof_segment_offsets_i32, dtype=wp.int32
        )
        self._muscle_dof_segment_idx_i32_wp = wp.from_torch(
            self._muscle_dof_segment_idx_i32, dtype=wp.int32
        )
        self._muscle_dof_sparse_entry_idx_i32_wp = wp.from_torch(
            self._muscle_dof_sparse_entry_idx_i32, dtype=wp.int32
        )
        self._muscle_dof_start_coeff_wp = wp.from_torch(
            self._muscle_dof_start_coeff, dtype=wp.float32
        )
        self._muscle_dof_end_coeff_wp = wp.from_torch(
            self._muscle_dof_end_coeff, dtype=wp.float32
        )
        self._muscle_dof_pair_entry_i_i32_wp = wp.from_torch(
            self._muscle_dof_pair_entry_i_i32, dtype=wp.int32
        )
        self._muscle_dof_pair_entry_j_i32_wp = wp.from_torch(
            self._muscle_dof_pair_entry_j_i32, dtype=wp.int32
        )
        self._muscle_dof_pair_muscle_idx_i32_wp = wp.from_torch(
            self._muscle_dof_pair_muscle_idx_i32, dtype=wp.int32
        )
        self._muscle_implicit_pair_elem_id_i32_wp = wp.from_torch(
            self._muscle_implicit_pair_elem_id_i32, dtype=wp.int32
        )
        self._muscle_dof_torque_entry_offsets_i32_wp = wp.from_torch(
            self._muscle_dof_torque_entry_offsets_i32, dtype=wp.int32
        )
        self._muscle_dof_torque_entry_idx_i32_wp = wp.from_torch(
            self._muscle_dof_torque_entry_idx_i32, dtype=wp.int32
        )

        self._install_muscle_implicit_velocity_derivative()

    def _install_muscle_implicit_velocity_derivative(self) -> None:
        if not self.settings.muscle.force_velocity.implicit:
            return
        if self.integrator != "implicitfast":
            raise ValueError(
                "muscle.force_velocity.implicit=true requires "
                "sim.integrator=implicitfast"
            )
        # Keep qDeriv within mass-matrix sparsity, matching MuJoCo Warp.
        # Damping pairs outside M_elemid are dropped; the muscle force itself
        # is applied in full.
        if self._muscle_implicit_dropped_pair_count > 0:
            logger.info(
                "implicit muscle force-velocity keeps M sparsity; %d cross-branch "
                "damping pairs have no M element and are dropped",
                self._muscle_implicit_dropped_pair_count,
            )
        installer = (
            self._require_execution_bindings().install_implicit_velocity_derivative_hook
        )
        if installer is None:
            raise RuntimeError(
                "implicit fast-muscle force-velocity requires a host derivative-hook "
                "installer."
            )
        installer(self._apply_muscle_force_velocity_implicit_derivative)

    def _apply_muscle_force_velocity_implicit_derivative(
        self,
        qderiv: wp.array,
    ) -> None:
        pair_count = int(self._muscle_implicit_prepared_pair_count)
        if pair_count == 0:
            return
        gather = self._muscle_implicit_qderiv_gather
        if gather is not None:
            apply_muscle_implicit_qderiv_gather(
                gather,
                self._muscle_sparse_moment_wp,
                self._muscle_force_velocity_slope_wp,
                float(self.model.opt.timestep),
                qderiv,
            )
            return
        wp.launch(
            apply_precomputed_muscle_implicit_qderiv_kernel,
            dim=(self.num_envs, pair_count),
            inputs=[
                self._muscle_sparse_moment_wp,
                self._muscle_force_velocity_slope_wp,
                self._muscle_implicit_pair_elem_id_i32_wp,
                self._muscle_dof_pair_entry_i_i32_wp,
                self._muscle_dof_pair_entry_j_i32_wp,
                self._muscle_dof_pair_muscle_idx_i32_wp,
                float(self.model.opt.timestep),
            ],
            outputs=[qderiv],
        )

    def _refresh_waypoint_sparse_moment(self) -> None:
        entry_count = int(self._muscle_dof_torque_idx_i32.numel())
        if bool(getattr(self, "_uses_lbs_waypoint_attachments", False)):
            common_inputs = [
                wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                self.mjw_data.xpos,
                self.mjw_data.xquat,
                self.mjw_data.subtree_com,
                self._cdof_view,
                wp.from_torch(self._body_rootid_i32, dtype=wp.int32),
            ]
            geometry_inputs = [
                self._muscle_dof_segment_idx_i32_wp,
                wp.from_torch(self._apply_segment_start_wp_i32, dtype=wp.int32),
                wp.from_torch(self._apply_segment_end_wp_i32, dtype=wp.int32),
                wp.from_torch(self._lbs_qfrc_active_attach_offsets_i32, dtype=wp.int32),
                wp.from_torch(
                    self._lbs_qfrc_active_attach_idx_body_i32, dtype=wp.vec2i
                ),
                wp.from_torch(self._lbs_qfrc_active_attach_local_coeff, dtype=wp.vec4),
                int(self.muscle_waypoints_local.shape[0]),
                int(self._root_dof_count),
            ]
            use_flat_atomic = (
                self.settings.muscle.body_wrench_forward_mode == "native_tile"
            )
            if use_flat_atomic:
                wp.launch(
                    clear_muscle_sparse_moment_kernel,
                    dim=(self.num_envs, entry_count),
                    outputs=[self._muscle_sparse_moment_wp],
                )
                sparse_count = int(self._muscle_dof_segment_idx_i32.numel())
                wp.launch(
                    eval_lbs_muscle_sparse_moment_atomic_kernel,
                    dim=(self.num_envs, sparse_count),
                    inputs=[
                        *common_inputs,
                        self._muscle_dof_sparse_entry_idx_i32_wp,
                        self._muscle_dof_torque_idx_i32_wp,
                        *geometry_inputs,
                    ],
                    outputs=[self._muscle_sparse_moment_wp],
                )
            else:
                wp.launch(
                    eval_lbs_muscle_sparse_moment_kernel,
                    dim=(self.num_envs, entry_count),
                    inputs=[
                        *common_inputs,
                        self._muscle_dof_torque_idx_i32_wp,
                        self._muscle_dof_segment_offsets_i32_wp,
                        *geometry_inputs,
                    ],
                    outputs=[self._muscle_sparse_moment_wp],
                )
            return
        launch_muscle_sparse_moment(
            wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
            self.mjw_data.subtree_com,
            self._cdof_view,
            self._muscle_dof_torque_idx_i32_wp,
            self._muscle_dof_segment_offsets_i32_wp,
            self._muscle_dof_segment_idx_i32_wp,
            self._muscle_dof_start_coeff_wp,
            self._muscle_dof_end_coeff_wp,
            wp.from_torch(self._apply_segment_start_wp_i32, dtype=wp.int32),
            wp.from_torch(self._apply_segment_end_wp_i32, dtype=wp.int32),
            wp.from_torch(self._segment_start_root_body_idx_i32, dtype=wp.int32),
            wp.from_torch(self._segment_end_root_body_idx_i32, dtype=wp.int32),
            int(self._root_dof_count),
            self._muscle_sparse_moment_wp,
        )

    def post_physics_substep_reads_geometry(self) -> bool:
        """Instantaneous muscle force has no endpoint-geometry commit."""
        return False

    def post_physics_substep_writes_state(self) -> bool:
        return False

    def post_physics_substep(self, substep_idx: int) -> None:
        del substep_idx

    def _update_muscle_force_velocity_state(
        self,
        activation: torch.Tensor,
        *,
        include_active: float,
    ) -> None:
        self._refresh_waypoint_sparse_moment()
        wp.launch(
            eval_muscle_force_velocity_state_kernel,
            dim=(self.num_envs, self.muscle_count),
            inputs=[
                wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                wp.from_torch(self.muscle_waypoint_start_i32, dtype=wp.int32),
                self._muscle_params_wp,
                self._muscle_rest_length_wp,
                self._muscle_force_velocity_length_scale_wp,
                wp.from_torch(activation, dtype=wp.float32),
                self.mjw_data.qvel,
                self._muscle_sparse_moment_wp,
                self._muscle_dof_offsets_i32_wp,
                self._muscle_dof_torque_idx_i32_wp,
                int(self._root_dof_count),
                float(self.settings.muscle.force_scale),
                float(self._muscle_length_norm_min),
                float(self._muscle_length_norm_max),
                float(self.settings.muscle.force_velocity.vmax),
                float(self.settings.muscle.force_velocity.fvmax),
                float(self._activation_clamp_enabled),
                float(include_active),
                self._muscle_active_force_model_params,
            ],
            outputs=[
                self._muscle_length_wp,
                self._muscle_force_velocity_effective_activation_wp,
                self._muscle_length_velocity_wp,
                self._muscle_force_velocity_multiplier_wp,
                self._muscle_force_velocity_slope_wp,
            ],
        )

    def _clear_muscle_force_velocity_implicit_derivative_state(self) -> None:
        if not self.settings.muscle.force_velocity.implicit:
            return
        self._muscle_force_velocity_slope.zero_()

    def _project_muscle_force_velocity_qfrc(self) -> None:
        torque_dim = int(self.model.nv - self._root_dof_count)
        wp.launch(
            project_muscle_sparse_moment_qfrc_kernel,
            dim=(self.num_envs, torque_dim),
            inputs=[
                self._muscle_sparse_moment_wp,
                self._muscle_force_magnitude_wp,
                self._muscle_dof_torque_entry_offsets_i32_wp,
                self._muscle_dof_torque_entry_idx_i32_wp,
                self._muscle_dof_muscle_idx_i32_wp,
                int(self._root_dof_count),
            ],
            outputs=[self._qfrc_applied_view_wp],
        )

    @startup_nvtx_wrapped("fast_muscle:initialize_muscle_dynamics_helper_state")
    def _initialize_muscle_dynamics_helper_state(self) -> None:
        self._muscle_dynamics_helper = None
        self._muscle_dynamics_helper_spec = None
        transient_auto_scale_helper = bool(
            self._action_mode == "torque"
            and getattr(self, "_torque_action_scale_auto_from_rest_bounds", False)
        )
        if not (
            transient_auto_scale_helper or self._env_native_tile_projection_requested()
        ):
            return
        self._muscle_dynamics_helper_spec = self._build_muscle_helper_runtime_spec(
            obs_context_dim=int(self.helper_obs_context_dim)
        )
        if not transient_auto_scale_helper:
            self._ensure_muscle_dynamics_helper()

    @startup_nvtx_wrapped("fast_muscle:ensure_muscle_dynamics_helper")
    def _ensure_muscle_dynamics_helper(
        self,
    ) -> MuscleDynamicsLayer | None:
        helper = getattr(self, "_muscle_dynamics_helper", None)
        if helper is not None:
            return helper
        helper_spec = getattr(self, "_muscle_dynamics_helper_spec", None)
        if helper_spec is None:
            return None
        helper = MuscleDynamicsLayer(helper_spec).to(self.device)
        helper.prepare_waypoint_wrap_workspace(batch_size=self.num_envs)
        helper.eval()
        self._muscle_dynamics_helper = helper
        if self.runtime_state_finalized:
            self._prepare_env_native_tile_projection()
        return helper

    def _uses_generic_activation_qfrc_projection(self) -> bool:
        return bool(self._policy_outputs_activation or self._uses_mass_coordinator)

    def _env_native_tile_projection_requested(self) -> bool:
        settings = getattr(self, "settings", None)
        if settings is None:
            return False
        muscle = settings.muscle
        force_velocity = muscle.force_velocity
        return bool(
            self._uses_generic_activation_qfrc_projection()
            and (
                not force_velocity.enabled
                or self._env_native_tile_force_velocity_supported()
            )
            and muscle.lbs.attachment_layout == "body_wrench"
            and muscle.body_wrench_forward_mode == "native_tile"
        )

    def _env_native_tile_force_velocity_supported(self) -> bool:
        """Explicit force-velocity needs only the path velocity, which body
        velocities give for single-body paths; implicit force-velocity keeps
        the moment arms for the solver derivative."""
        return bool(
            not self.settings.muscle.force_velocity.implicit
            and getattr(self, "_observes_waypoint_path_velocity", False)
        )

    def _prepare_env_native_tile_projection(self) -> bool:
        helper = getattr(self, "_muscle_dynamics_helper", None)
        force_velocity = bool(self.settings.muscle.force_velocity.enabled)
        ready = False
        if (
            helper is not None
            and self._env_native_tile_projection_requested()
            and int(self._root_dof_count)
            == int(helper.full_torque_dim - helper.torque_dim)
        ):
            prepare_workspace = (
                helper.prepare_native_body_wrench_full_force_velocity_workspace
                if force_velocity
                else helper.prepare_native_body_wrench_full_force_workspace
            )
            ready = bool(
                prepare_workspace(batch_size=self.num_envs, device=self.device)
            )
        if ready and force_velocity:
            self._ensure_env_muscle_qfrc_projector_state()
            self._muscle_force_velocity_scale = torch.clamp(
                self.muscle_force_velocity_length_scale
                * float(self.settings.muscle.force_velocity.vmax),
                min=1.0e-6,
            )
        self._env_native_tile_projection_ready = ready
        return ready

    def _try_apply_activation_to_qfrc_native_tile(
        self,
        activation: torch.Tensor,
        *,
        include_active: float,
        include_passive: float,
        require_muscle_state: bool = False,
    ) -> bool:
        if not bool(getattr(self, "_env_native_tile_projection_ready", False)):
            return False
        if require_muscle_state:
            return False
        if float(include_active) != 1.0 or float(include_passive) not in (0.0, 1.0):
            return False
        helper = self._muscle_dynamics_helper
        if helper is None:
            return False
        if self.settings.muscle.force_velocity.enabled:
            self._apply_force_velocity_activation_to_qfrc_native_tile(
                helper,
                activation,
                include_passive=include_passive,
            )
            return True
        native_activation = activation
        if self._activation_clamp_enabled != 0.0:
            self._activation_apply_buffer.copy_(activation)
            self._activation_apply_buffer.clamp_(
                self._ACTIVATION_LOWER_BOUND,
                self._ACTIVATION_UPPER_BOUND,
            )
            native_activation = self._activation_apply_buffer
        helper.launch_native_body_wrench_full_force_into(
            activation=native_activation,
            body_pos=self._xpos_view,
            body_quat=self._xquat_view,
            xipos=self._xipos_view,
            subtree_com=self._subtree_com_view,
            cdof=self._cdof_view,
            output=self._qfrc_applied_view,
            include_passive_force=bool(include_passive),
            output_dof_offset=self._root_dof_count,
        )
        return True

    def _apply_force_velocity_activation_to_qfrc_native_tile(
        self,
        helper: MuscleDynamicsLayer,
        activation: torch.Tensor,
        *,
        include_passive: float,
    ) -> None:
        """Explicit force-velocity force evaluated and projected in the native
        tiles; the path velocity comes from body velocities, not moment arms."""
        self._launch_waypoint_body_cvel()
        helper.launch_native_body_wrench_full_force_velocity_into(
            activation=activation,
            body_pos=self._xpos_view,
            body_quat=self._xquat_view,
            xipos=self._xipos_view,
            subtree_com=self._subtree_com_view,
            cdof=self._cdof_view,
            body_cvel=self._waypoint_body_cvel,
            body_rootid=self._body_rootid_i32,
            velocity_scale=self._muscle_force_velocity_scale,
            fvmax=float(self.settings.muscle.force_velocity.fvmax),
            clamp_activation=self._activation_clamp_enabled != 0.0,
            output=self._qfrc_applied_view,
            include_passive_force=bool(include_passive),
            output_dof_offset=self._root_dof_count,
        )

    def _muscle_dynamics_layer_static_metadata_cache_key(self) -> str | None:
        source_path = getattr(self, "_source_mjcf_path", "")
        if not source_path:
            return None
        try:
            # Every included file, not just the top-level MJCF: an edit to an
            # included body or tendon file changes the waypoint attachments.
            source_signature = collect_xml_dependency_signatures(Path(source_path))
        except (OSError, RuntimeError, ET.ParseError):
            return None
        metadata_signature = None
        metadata_path = self.settings.muscle.metadata_path
        if metadata_path:
            try:
                metadata_signature = file_signature(Path(metadata_path))
            except (OSError, RuntimeError):
                return None
        active_torque_indices = (
            tuple(int(v) for v in self._active_torque_indices.detach().cpu().tolist())
            if getattr(self, "_has_masked_torque_dofs", False)
            else None
        )
        payload = {
            "source": source_signature,
            "native_tile_event_partition_version": 2,
            "metadata": metadata_signature,
            "geometry_backend": str(self._geometry_backend),
            "body_count": int(self.model.nbody),
            "full_torque_dim": int(self.model.nv),
            "torque_dim": int(self.model.nv - self._root_dof_count),
            "muscle_count": int(self.muscle_count),
            "active_torque_indices": active_torque_indices,
            "uses_lbs_waypoint_attachments": bool(
                getattr(self, "_uses_lbs_waypoint_attachments", False)
            ),
            "lbs_forward_chunk_size": self.settings.muscle.lbs.forward_chunk_size,
            "lbs_body_event_chunk_size": self.settings.muscle.lbs.body_event_chunk_size,
            "lbs_attachment_layout": self.settings.muscle.lbs.attachment_layout,
            "lbs_body_wrench_forward_mode": (
                self.settings.muscle.body_wrench_forward_mode
            ),
            "lbs_body_wrench_native_tile_muscle_cap": (
                self.settings.muscle.lbs.body_wrench.native_tile.muscle_cap
            ),
            "lbs_body_wrench_native_tile_segment_cap": (
                self.settings.muscle.lbs.body_wrench.native_tile.segment_cap
            ),
            "lbs_body_wrench_native_tile_event_cap": (
                self.settings.muscle.lbs.body_wrench.native_tile.event_cap
            ),
            "lbs_body_wrench_native_tile_body_cap": (
                self.settings.muscle.lbs.body_wrench.native_tile.body_cap
            ),
        }
        return cache_key("muscle_dynamics_layer_static_metadata", payload)

    def get_policy_action_scale_snapshot(self) -> ActionScaleSnapshot | None:
        if not self._policy_outputs_torque:
            # MASS pose control: pose residuals use the ordinary
            # unsquashed Gaussian policy. The residual scale is applied by the
            # environment after sampling; it is not a hard action bound.
            return None
        # Torque commands are scaled in the environment; the policy keeps
        # unit scales over its (possibly masked) action width.
        unit_scale = torch.ones_like(self._torque_action_scale)
        if getattr(self, "_has_masked_torque_dofs", False):
            unit_scale = unit_scale[self._active_torque_indices.long()]
        return ActionScaleSnapshot(
            mean_scale=unit_scale,
            std_scale=unit_scale.clone(),
        )

    def get_policy_transform_semantics(self) -> PolicyTransformSemantics:
        return PolicyTransformSemantics()

    @staticmethod
    def _copy_float_block_(
        target: torch.Tensor,
        offset: int,
        source: torch.Tensor,
    ) -> int:
        width = int(source.shape[1])
        copy_2d_(target.narrow(1, int(offset), width), source)
        return int(offset) + width

    @staticmethod
    def _masked_zero_2d_(values: torch.Tensor, reset_mask: torch.Tensor) -> None:
        masked_zero_2d_(values, reset_mask)

    def _clear_applied_force_buffers(self) -> None:
        zero_(self._xfrc_applied_flat_view)
        zero_(self._qfrc_applied_view)

    def _transform_muscle_waypoints(self) -> None:
        if bool(getattr(self, "_uses_lbs_waypoint_attachments", False)):
            wp.launch(
                transform_lbs_waypoints_kernel,
                dim=(self.num_envs, self.muscle_waypoints_local.shape[0]),
                inputs=[
                    self.mjw_data.xpos,
                    self.mjw_data.xquat,
                    wp.from_torch(
                        self.muscle_waypoint_attachment_start_i32, dtype=wp.int32
                    ),
                    wp.from_torch(
                        self.muscle_waypoint_attachment_body_i32, dtype=wp.int32
                    ),
                    wp.from_torch(self.muscle_waypoint_attachment_local, dtype=wp.vec3),
                    wp.from_torch(
                        self.muscle_waypoint_attachment_weight, dtype=wp.float32
                    ),
                ],
                outputs=[wp.from_torch(self.transformed_waypoints, dtype=wp.vec3)],
            )
            return
        wp.launch(
            transform_waypoints_kernel,
            dim=(self.num_envs, self.muscle_waypoints_local.shape[0]),
            inputs=[
                self.mjw_data.xpos,
                self.mjw_data.xquat,
                wp.from_torch(self.muscle_waypoint_body_i32, dtype=wp.int32),
                wp.from_torch(self.muscle_waypoints_local, dtype=wp.vec3),
            ],
            outputs=[wp.from_torch(self.transformed_waypoints, dtype=wp.vec3)],
        )

        self._waypoint_wrapping.transform(
            self.mjw_data.xpos,
            self.mjw_data.xquat,
            wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
            self._waypoint_wrap_state,
        )

    def _project_muscle_force_magnitude_to_qfrc(self) -> None:
        self._ensure_env_muscle_qfrc_projector_state()
        muscle_force_magnitude_wp = getattr(self, "_muscle_force_magnitude_wp", None)
        if muscle_force_magnitude_wp is None:
            muscle_force_magnitude_wp = wp.from_torch(
                self._muscle_force_magnitude, dtype=wp.float32
            )
        if bool(getattr(self, "_uses_lbs_waypoint_attachments", False)):
            if (
                getattr(self, "_lbs_qfrc_forward_reduction", "serial") != "serial"
                and int(getattr(self, "_lbs_qfrc_torque_chunk_count", 0)) > 0
            ):
                qfrc_inputs = [
                    muscle_force_magnitude_wp,
                    wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                    self.mjw_data.xpos,
                    self.mjw_data.xquat,
                    self.mjw_data.subtree_com,
                    self._cdof_view,
                    wp.from_torch(self._body_rootid_i32, dtype=wp.int32),
                    wp.from_torch(self._lbs_qfrc_chunk_torque_idx_i32, dtype=wp.int32),
                    wp.from_torch(self._lbs_qfrc_chunk_entry_start_i32, dtype=wp.int32),
                    wp.from_torch(self._lbs_qfrc_chunk_entry_end_i32, dtype=wp.int32),
                    wp.from_torch(
                        self._lbs_qfrc_forward_segment_idx_i32, dtype=wp.int32
                    ),
                    wp.from_torch(
                        self._lbs_qfrc_forward_muscle_idx_i32, dtype=wp.int32
                    ),
                    wp.from_torch(self._apply_segment_start_wp_i32, dtype=wp.int32),
                    wp.from_torch(self._apply_segment_end_wp_i32, dtype=wp.int32),
                    wp.from_torch(
                        self._lbs_qfrc_active_attach_offsets_i32, dtype=wp.int32
                    ),
                    wp.from_torch(
                        self._lbs_qfrc_active_attach_idx_body_i32, dtype=wp.vec2i
                    ),
                    wp.from_torch(
                        self._lbs_qfrc_active_attach_local_coeff, dtype=wp.vec4
                    ),
                    int(self.muscle_waypoints_local.shape[0]),
                    int(self._root_dof_count),
                ]
                wp.launch_tiled(
                    eval_muscle_lbs_qfrc_pair_vec4_native_tiled_partials_kernel,
                    dim=(self.num_envs, int(self._lbs_qfrc_torque_chunk_count)),
                    inputs=qfrc_inputs,
                    outputs=[
                        wp.from_torch(self._lbs_qfrc_chunk_partial, dtype=wp.float32)
                    ],
                    block_dim=int(self._lbs_qfrc_forward_tile_block_dim),
                )
                wp.launch(
                    reduce_lbs_qfrc_chunk_partials_kernel,
                    dim=(
                        self.num_envs,
                        int(self.model.nv - self._root_dof_count),
                    ),
                    inputs=[
                        wp.from_torch(self._lbs_qfrc_chunk_partial, dtype=wp.float32),
                        wp.from_torch(self._lbs_qfrc_chunk_offsets_i32, dtype=wp.int32),
                        int(self._root_dof_count),
                    ],
                    outputs=[self._qfrc_applied_view],
                )
                return
            wp.launch(
                eval_muscle_lbs_qfrc_kernel,
                dim=(self.num_envs, int(self.model.nv - self._root_dof_count)),
                inputs=[
                    muscle_force_magnitude_wp,
                    wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                    self.mjw_data.xpos,
                    self.mjw_data.xquat,
                    self.mjw_data.subtree_com,
                    self._cdof_view,
                    wp.from_torch(self._body_rootid_i32, dtype=wp.int32),
                    wp.from_torch(self._torque_dof_subtree_mask, dtype=wp.float32),
                    wp.from_torch(
                        self._torque_active_entry_offsets_i32, dtype=wp.int32
                    ),
                    wp.from_torch(self._torque_active_segment_idx_i32, dtype=wp.int32),
                    wp.from_torch(self._torque_active_muscle_idx_i32, dtype=wp.int32),
                    wp.from_torch(self._apply_segment_start_wp_i32, dtype=wp.int32),
                    wp.from_torch(self._apply_segment_end_wp_i32, dtype=wp.int32),
                    wp.from_torch(
                        self.muscle_waypoint_attachment_start_i32, dtype=wp.int32
                    ),
                    wp.from_torch(
                        self.muscle_waypoint_attachment_body_i32, dtype=wp.int32
                    ),
                    wp.from_torch(self.muscle_waypoint_attachment_local, dtype=wp.vec3),
                    wp.from_torch(
                        self.muscle_waypoint_attachment_weight, dtype=wp.float32
                    ),
                    int(self._root_dof_count),
                ],
                outputs=[self._qfrc_applied_view],
            )
            return
        wp.launch(
            eval_muscle_qfrc_kernel,
            dim=(self.num_envs, int(self.model.nv - self._root_dof_count)),
            inputs=[
                muscle_force_magnitude_wp,
                wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                self.mjw_data.xipos,
                self.mjw_data.subtree_com,
                self._cdof_view,
                wp.from_torch(self._torque_active_entry_offsets_i32, dtype=wp.int32),
                wp.from_torch(self._torque_active_segment_idx_i32, dtype=wp.int32),
                wp.from_torch(self._torque_active_muscle_idx_i32, dtype=wp.int32),
                wp.from_torch(self._torque_active_start_coeff, dtype=wp.float32),
                wp.from_torch(self._torque_active_end_coeff, dtype=wp.float32),
                wp.from_torch(self._apply_segment_start_wp_i32, dtype=wp.int32),
                wp.from_torch(self._apply_segment_end_wp_i32, dtype=wp.int32),
                wp.from_torch(self._apply_segment_body_start_i32, dtype=wp.int32),
                wp.from_torch(self._apply_segment_body_end_i32, dtype=wp.int32),
                wp.from_torch(self._segment_start_root_body_idx_i32, dtype=wp.int32),
                wp.from_torch(self._segment_end_root_body_idx_i32, dtype=wp.int32),
                int(self._root_dof_count),
            ],
            outputs=[self._qfrc_applied_view],
        )

    def _evaluate_wrapped_muscle_force(
        self, activation, *, clamp, include_active, include_passive
    ):
        wrapping = self._waypoint_wrapping
        state = self._waypoint_wrap_state
        wp.launch(
            path_lengths_kernel,
            dim=(self.num_envs, self.muscle_count),
            inputs=[
                wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                wp.from_torch(self.muscle_waypoint_start_i32),
                wrapping.point_arc,
                state.arc,
            ],
            outputs=[wp.from_torch(self.muscle_length)],
            device=str(self.device),
        )
        wp.launch(
            force_from_lengths_kernel,
            dim=(self.num_envs, self.muscle_count),
            inputs=[
                wp.from_torch(self.muscle_length),
                wp.from_torch(self.muscle_params),
                wp.from_torch(activation),
                wp.from_torch(self.muscle_rest_length),
                float(self.settings.muscle.force_scale),
                float(self._muscle_length_norm_min),
                float(self._muscle_length_norm_max),
                float(clamp),
                float(include_active),
                float(include_passive),
                self._muscle_active_force_model_params,
                self._muscle_passive_force_model_params,
            ],
            outputs=[wp.from_torch(self._muscle_force_magnitude)],
            device=str(self.device),
        )

    def _apply_activation_to_qfrc(
        self,
        activation: torch.Tensor,
        *,
        include_active: float,
        include_passive: float,
        require_muscle_state: bool = False,
    ) -> None:
        self._clear_applied_force_buffers()
        if self._try_apply_activation_to_qfrc_native_tile(
            activation,
            include_active=include_active,
            include_passive=include_passive,
            require_muscle_state=require_muscle_state,
        ):
            return
        self._transform_muscle_waypoints()
        if self._waypoint_wrapping.count:
            force_activation = activation
            clamp = self._activation_clamp_enabled
            if self.settings.muscle.force_velocity.enabled:
                self._refresh_waypoint_sparse_moment()
                wp.launch(
                    force_velocity_activation_kernel,
                    dim=(self.num_envs, self.muscle_count),
                    inputs=[
                        wp.from_torch(activation),
                        self.mjw_data.qvel,
                        self._muscle_sparse_moment_wp,
                        self._muscle_dof_offsets_i32_wp,
                        self._muscle_dof_torque_idx_i32_wp,
                        self._muscle_force_velocity_length_scale_wp,
                        int(self._root_dof_count),
                        float(self.settings.muscle.force_velocity.vmax),
                        float(self.settings.muscle.force_velocity.fvmax),
                        float(clamp),
                    ],
                    outputs=[self._muscle_force_velocity_effective_activation_wp],
                    device=str(self.device),
                )
                force_activation = self._muscle_force_velocity_effective_activation
                clamp = 0.0
            self._evaluate_wrapped_muscle_force(
                force_activation,
                clamp=clamp,
                include_active=include_active,
                include_passive=include_passive,
            )
            if self.settings.muscle.force_velocity.enabled:
                self._project_muscle_force_velocity_qfrc()
            else:
                self._project_muscle_force_magnitude_to_qfrc()
            return
        if self.settings.muscle.force_velocity.enabled:
            self._update_muscle_force_velocity_state(
                activation,
                include_active=include_active,
            )
            wp.launch(
                eval_muscle_force_magnitude_kernel,
                dim=(self.num_envs, self.muscle_count),
                inputs=[
                    wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                    wp.from_torch(self.muscle_waypoint_start_i32, dtype=wp.int32),
                    self._muscle_params_wp,
                    self._muscle_force_velocity_effective_activation_wp,
                    self._muscle_rest_length_wp,
                    float(self.settings.muscle.force_scale),
                    float(self._muscle_length_norm_min),
                    float(self._muscle_length_norm_max),
                    0.0,
                    float(include_active),
                    float(include_passive),
                    self._muscle_active_force_model_params,
                    self._muscle_passive_force_model_params,
                ],
                outputs=[
                    self._muscle_length_wp,
                    self._muscle_force_magnitude_wp,
                ],
            )
        else:
            wp.launch(
                eval_muscle_force_magnitude_kernel,
                dim=(self.num_envs, self.muscle_count),
                inputs=[
                    wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                    wp.from_torch(self.muscle_waypoint_start_i32, dtype=wp.int32),
                    wp.from_torch(self.muscle_params, dtype=wp.float32),
                    wp.from_torch(activation, dtype=wp.float32),
                    wp.from_torch(self.muscle_rest_length, dtype=wp.float32),
                    float(self.settings.muscle.force_scale),
                    float(self._muscle_length_norm_min),
                    float(self._muscle_length_norm_max),
                    float(self._activation_clamp_enabled),
                    float(include_active),
                    float(include_passive),
                    self._muscle_active_force_model_params,
                    self._muscle_passive_force_model_params,
                ],
                outputs=[
                    wp.from_torch(self.muscle_length, dtype=wp.float32),
                    wp.from_torch(self._muscle_force_magnitude, dtype=wp.float32),
                ],
            )
        if self.settings.muscle.force_velocity.enabled:
            self._project_muscle_force_velocity_qfrc()
        else:
            self._project_muscle_force_magnitude_to_qfrc()

    def _prepare_torque_passive_muscle_state(
        self,
        *,
        include_passive: float | None = None,
    ) -> None:
        self._apply_activation_to_qfrc(
            self._zero_muscle_actions,
            include_active=0.0,
            include_passive=self._muscle_include_passive
            if include_passive is None
            else float(include_passive),
            require_muscle_state=True,
        )

    def _update_prev_action_observation(self) -> None:
        if self._include_prev_actions_obs:
            self._prev_action_obs.copy_(self.actions)

    def _clip_torque_actions_to_feasible_bounds(
        self,
        actions_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_source = self.actions if actions_override is None else actions_override
        self._prepare_torque_passive_muscle_state()
        wp.launch(
            clip_torque_feasible_bounds_kernel,
            dim=(self.num_envs, int(self.model.nv - self._root_dof_count)),
            inputs=[
                wp.from_torch(self.muscle_length, dtype=wp.float32),
                wp.from_torch(self.muscle_params, dtype=wp.float32),
                wp.from_torch(self.muscle_rest_length, dtype=wp.float32),
                float(self.settings.muscle.force_scale),
                float(self._muscle_length_norm_min),
                float(self._muscle_length_norm_max),
                self._muscle_active_force_model_params,
                wp.from_torch(self._torque_active_entry_offsets_i32, dtype=wp.int32),
                wp.from_torch(self._torque_active_segment_idx_i32, dtype=wp.int32),
                wp.from_torch(self._torque_active_muscle_idx_i32, dtype=wp.int32),
                wp.from_torch(self._torque_active_start_coeff, dtype=wp.float32),
                wp.from_torch(self._torque_active_end_coeff, dtype=wp.float32),
                wp.from_torch(self._apply_segment_start_wp_i32, dtype=wp.int32),
                wp.from_torch(self._apply_segment_end_wp_i32, dtype=wp.int32),
                wp.from_torch(self._apply_segment_body_start_i32, dtype=wp.int32),
                wp.from_torch(self._apply_segment_body_end_i32, dtype=wp.int32),
                wp.from_torch(self._segment_start_root_body_idx_i32, dtype=wp.int32),
                wp.from_torch(self._segment_end_root_body_idx_i32, dtype=wp.int32),
                wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                self.mjw_data.xipos,
                self.mjw_data.subtree_com,
                self._cdof_view,
                wp.from_torch(self._torque_action_scale, dtype=wp.float32),
                wp.from_torch(self._torque_action_scale_inv, dtype=wp.float32),
                int(self._root_dof_count),
                wp.from_torch(action_source, dtype=wp.float32),
            ],
        )
        return action_source
