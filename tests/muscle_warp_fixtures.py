"""Small muscle layers and geometry contexts for the Warp custom-op tests."""

from __future__ import annotations

import torch

from instantmimic.muscle.muscle_dynamics import MuscleDynamicsLayer
from instantmimic.muscle.runtime_specs import (
    MuscleDimensionsSpec,
    MuscleLayerRuntimeSpec,
    MuscleMechanicsRuntimeSpec,
    MuscleOutputSpec,
    MuscleProjectionRuntimeSpec,
    MuscleTopologySpec,
)


def make_muscle_layer(
    device: str | torch.device = "cpu",
    *,
    muscle_count: int = 3,
    torque_dim: int = 2,
    root_dof_count: int = 6,
    uses_lbs: bool = False,
    native_tile: bool = False,
    force_velocity: bool = False,
) -> MuscleDynamicsLayer:
    """A two-body, waypoint-projected layer with one segment per muscle.

    The lower waypoint sits one metre above the root and the upper one above
    the child body, so a rotation about the child's z axis gives a nonzero
    moment arm even at zero body displacement.
    """
    full_torque_dim = int(torque_dim) + int(root_dof_count)
    muscles = int(muscle_count)
    index = torch.arange(muscles, dtype=torch.long, device=device)
    segment_starts = index * 2
    muscle_params = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).repeat(muscles, 1)
    muscle_rest_length = torch.ones(muscles, dtype=torch.float32, device=device)
    muscle_waypoints_local = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    ).repeat(muscles, 1)
    muscle_waypoint_body = torch.tensor([0, 1], dtype=torch.long, device=device).repeat(
        muscles
    )
    context_dim = 2 * 3 + 2 * 4 + 2 * 3 + 2 * 3 + full_torque_dim * 6
    if force_velocity:
        context_dim += full_torque_dim
    topology = MuscleTopologySpec(
        muscle_params=muscle_params,
        muscle_rest_length=muscle_rest_length,
        muscle_force_velocity_length_scale=muscle_rest_length * muscle_params[:, 1],
        muscle_waypoints_local=muscle_waypoints_local,
        muscle_waypoint_body=muscle_waypoint_body,
        uses_lbs_waypoint_attachments=bool(uses_lbs),
        muscle_waypoint_attachment_start=torch.arange(
            muscle_waypoint_body.numel() + 1, dtype=torch.long, device=device
        ),
        muscle_waypoint_attachment_body=muscle_waypoint_body,
        muscle_waypoint_attachment_local=muscle_waypoints_local,
        muscle_waypoint_attachment_weight=torch.ones(
            muscle_waypoint_body.numel(), dtype=torch.float32, device=device
        ),
        all_segment_muscle_idx=index,
        all_segment_start_idx=segment_starts,
        apply_segment_muscle_idx=index,
        apply_segment_start_idx=segment_starts,
        apply_segment_end_idx=segment_starts + 1,
        apply_segment_body_start_idx=torch.zeros_like(index),
        apply_segment_body_end_idx=torch.ones_like(index),
        body_parentid=torch.tensor([0, 0], dtype=torch.long, device=device),
        body_rootid=torch.tensor([0, 0], dtype=torch.long, device=device),
        dof_bodyid=torch.tensor(
            [0] * (full_torque_dim - 1) + [1],
            dtype=torch.long,
            device=device,
        ),
    )
    spec = MuscleLayerRuntimeSpec(
        dimensions=MuscleDimensionsSpec(
            obs_context_dim=context_dim,
            body_count=2,
            full_torque_dim=full_torque_dim,
            torque_dim=int(torque_dim),
            muscle_count=muscles,
        ),
        output=MuscleOutputSpec(
            action_dim=int(torque_dim),
            active_torque_indices=None,
            bound_chunk_size=8,
        ),
        mechanics=MuscleMechanicsRuntimeSpec(
            dynamics_model="instantaneous",
            force_scale=1.0,
            active_force_type="gaussian",
            active_force_gamma=0.45,
            active_force_lmin=0.5,
            active_force_lmax=1.6,
            force_velocity_enabled=bool(force_velocity),
            force_velocity_vmax=1.5,
            force_velocity_fvmax=1.5,
            passive_force_type="exponential",
            passive_force_k_pe=5.5,
            passive_force_e_mo=0.3,
            passive_force_lmax=1.6,
            passive_force_fpmax=1.3,
            length_norm_min=0.2,
            length_norm_max=1.4,
            include_passive_force_in_output=False,
        ),
        projection=MuscleProjectionRuntimeSpec(
            geometry_backend="waypoint",
            sparse_body_wrench_forward=True,
            sparse_body_wrench_direct=False,
            sparse_body_wrench_direct_backward=True,
            lbs_forward_reduction="partial",
            lbs_forward_chunk_size=2048,
            lbs_forward_tile_block_dim=128,
            lbs_body_event_chunk_size=8192,
            lbs_muscle_length_by_muscle_min_batch=512,
            lbs_attachment_layout="body_wrench",
            lbs_body_wrench_forward_mode="native_tile" if native_tile else "pairchain",
            lbs_body_wrench_native_tile_muscle_cap=32,
            lbs_body_wrench_native_tile_segment_cap=256,
            lbs_body_wrench_native_tile_event_cap=512,
            lbs_body_wrench_native_tile_body_cap=4,
            static_metadata_cache_key=None,
        ),
        topology=topology,
    )
    return MuscleDynamicsLayer(spec)


def make_force_velocity_layer(
    device: str | torch.device = "cpu",
    *,
    native_tile: bool = False,
) -> MuscleDynamicsLayer:
    return make_muscle_layer(
        device,
        uses_lbs=True,
        native_tile=native_tile,
        force_velocity=True,
    )


def make_context(
    layer: MuscleDynamicsLayer,
    batch: int = 4,
    *,
    device: str | torch.device | None = None,
    velocity: float = 0.0,
) -> torch.Tensor:
    device = layer.muscle_params.device if device is None else device
    body = torch.zeros((batch, layer.body_count, 3), device=device)
    body[:, 1, 0] = 1.0
    quat = torch.zeros((batch, layer.body_count, 4), device=device)
    quat[..., 0] = 1.0
    cdof = torch.zeros((batch, layer.full_torque_dim, 6), device=device)
    last_torque_dof = layer.root_dof_count + layer.torque_dim - 1
    cdof[:, last_torque_dof, 2] = 1.0
    parts = [
        body.flatten(1),
        quat.flatten(1),
        body.flatten(1),
        body.flatten(1),
        cdof.flatten(1),
    ]
    if layer.muscle_force_velocity_enabled:
        qvel = torch.zeros((batch, layer.full_torque_dim), device=device)
        qvel[:, last_torque_dof] = float(velocity)
        parts.append(qvel)
    return torch.cat(parts, dim=1)


def with_warp_forward(layer: MuscleDynamicsLayer) -> MuscleDynamicsLayer:
    """Route the CPU layer through the Warp implementation for op tests."""
    layer._can_use_warp_muscle_dynamics_forward = lambda raw_context: True
    return layer
