from __future__ import annotations

from dataclasses import dataclass

import torch

_LEGACY_FREE_ROOT_DOF_COUNT = 6


# Default normalized fiber length; authored MASS exponential passive-force
# configurations use an upper limit of 1.3, where that curve reaches 1 Fmax.
_DEFAULT_MUSCLE_LENGTH_RANGE: dict[str, tuple[float, float]] = {
    "instantaneous": (0.2, 1.8),
}


def resolve_muscle_length_range(
    length_range: tuple[float, float] | None,
    dynamics_model: str,
) -> tuple[float, float]:
    """The authored ``muscle.length_range``, else the dynamics model's default."""

    if length_range is not None:
        return length_range
    try:
        return _DEFAULT_MUSCLE_LENGTH_RANGE[dynamics_model]
    except KeyError as exc:
        raise ValueError(
            f"No default muscle.length_range for dynamics model {dynamics_model!r}"
        ) from exc


@dataclass(frozen=True, slots=True)
class MuscleDimensionsSpec:
    obs_context_dim: int
    body_count: int
    full_torque_dim: int
    torque_dim: int
    muscle_count: int

    def __post_init__(self) -> None:
        if self.body_count <= 0 or self.muscle_count <= 0 or self.torque_dim <= 0:
            raise ValueError(
                "muscle body, muscle, and torque dimensions must be positive"
            )
        if self.full_torque_dim < self.torque_dim:
            raise ValueError("full_torque_dim must be at least torque_dim")
        if self.obs_context_dim < 0:
            raise ValueError("obs_context_dim must be non-negative")


@dataclass(frozen=True, slots=True, eq=False)
class MuscleOutputSpec:
    action_dim: int
    active_torque_indices: torch.Tensor | None
    bound_chunk_size: int

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("muscle output action_dim must be positive")
        if self.bound_chunk_size <= 0:
            raise ValueError("muscle output bound_chunk_size must be positive")


@dataclass(frozen=True, slots=True)
class MuscleMechanicsRuntimeSpec:
    dynamics_model: str
    force_scale: float
    active_force_type: str
    active_force_gamma: float
    active_force_lmin: float
    active_force_lmax: float
    force_velocity_enabled: bool
    force_velocity_vmax: float
    force_velocity_fvmax: float
    passive_force_type: str
    passive_force_k_pe: float
    passive_force_e_mo: float
    passive_force_lmax: float
    passive_force_fpmax: float
    length_norm_min: float
    length_norm_max: float
    include_passive_force_in_output: bool

    def __post_init__(self) -> None:
        dynamics_model = self.dynamics_model.strip().lower()
        if dynamics_model != "instantaneous":
            raise ValueError(
                f"Unsupported physical_dynamics_model {self.dynamics_model!r}."
            )
        if self.force_velocity_enabled:
            if self.force_velocity_vmax <= 0.0:
                raise ValueError("muscle_force_velocity_vmax must be positive")
            if self.force_velocity_fvmax <= 1.0:
                raise ValueError("muscle_force_velocity_fvmax must be greater than 1")


@dataclass(frozen=True, slots=True)
class MuscleProjectionRuntimeSpec:
    geometry_backend: str
    sparse_body_wrench_forward: bool
    sparse_body_wrench_direct: bool
    sparse_body_wrench_direct_backward: bool
    lbs_forward_reduction: str
    lbs_forward_chunk_size: int
    lbs_forward_tile_block_dim: int
    lbs_body_event_chunk_size: int
    lbs_muscle_length_by_muscle_min_batch: int
    lbs_attachment_layout: str
    lbs_body_wrench_forward_mode: str
    lbs_body_wrench_native_tile_muscle_cap: int
    lbs_body_wrench_native_tile_segment_cap: int
    lbs_body_wrench_native_tile_event_cap: int
    lbs_body_wrench_native_tile_body_cap: int
    static_metadata_cache_key: str | None

    def __post_init__(self) -> None:
        if self.geometry_backend != "waypoint":
            raise ValueError("geometry_backend must be 'waypoint'")


@dataclass(frozen=True, slots=True, eq=False)
class WaypointWrapSpec:
    """Fixed-radius site/geom/site paths in body-local coordinates."""

    body: torch.Tensor
    side_body: torch.Tensor
    kind: torch.Tensor
    center: torch.Tensor
    quaternion: torch.Tensor
    radius: torch.Tensor
    side_local: torch.Tensor
    start_point: torch.Tensor

    def __post_init__(self):
        count = self.radius.numel()
        for name in ("body", "side_body", "kind", "radius", "start_point"):
            _require_vector_shape(getattr(self, name), count, name="wrap." + name)
        for name, width in (("center", 3), ("quaternion", 4), ("side_local", 3)):
            _require_matrix_shape(
                getattr(self, name), count, width, name="wrap." + name
            )
        for name in ("body", "side_body", "kind", "start_point"):
            if getattr(self, name).dtype not in (torch.int32, torch.int64):
                raise TypeError("wrap." + name + " must use an integer index dtype")
        for name in ("center", "quaternion", "radius", "side_local"):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError("wrap." + name + " must be finite")
        if bool((self.radius <= 0).any()):
            raise ValueError("Wrap radii must be positive")
        norms = torch.linalg.vector_norm(self.quaternion, dim=1)
        if not bool(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
        ):
            raise ValueError("Wrap quaternions must be normalized wxyz rotations")
        if bool((self.body < 0).any()) or bool((self.side_body < -1).any()):
            raise ValueError("Wrap body indices are invalid")


@dataclass(frozen=True, slots=True, eq=False)
class MuscleTopologySpec:
    muscle_params: torch.Tensor
    muscle_rest_length: torch.Tensor
    muscle_force_velocity_length_scale: torch.Tensor
    muscle_waypoints_local: torch.Tensor
    muscle_waypoint_body: torch.Tensor
    uses_lbs_waypoint_attachments: bool
    muscle_waypoint_attachment_start: torch.Tensor
    muscle_waypoint_attachment_body: torch.Tensor
    muscle_waypoint_attachment_local: torch.Tensor
    muscle_waypoint_attachment_weight: torch.Tensor
    all_segment_muscle_idx: torch.Tensor
    all_segment_start_idx: torch.Tensor
    apply_segment_muscle_idx: torch.Tensor
    apply_segment_start_idx: torch.Tensor
    apply_segment_end_idx: torch.Tensor
    apply_segment_body_start_idx: torch.Tensor
    apply_segment_body_end_idx: torch.Tensor
    body_parentid: torch.Tensor
    body_rootid: torch.Tensor
    dof_bodyid: torch.Tensor
    wraps: WaypointWrapSpec | None = None


def _require_vector_shape(tensor: torch.Tensor, expected: int, *, name: str) -> None:
    if tensor.ndim != 1 or tensor.shape[0] != expected:
        raise ValueError(f"{name} must have shape ({expected},)")


def _require_matrix_shape(
    tensor: torch.Tensor,
    rows: int,
    columns: int,
    *,
    name: str,
) -> None:
    if tensor.ndim != 2 or tuple(tensor.shape) != (rows, columns):
        raise ValueError(f"{name} must have shape ({rows}, {columns})")


def _require_index_vector(
    tensor: torch.Tensor,
    expected: int,
    *,
    upper: int,
    name: str,
) -> None:
    _require_vector_shape(tensor, expected, name=name)
    if tensor.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use an integer index dtype")
    if tensor.numel() == 0:
        return
    minimum = int(tensor.amin().item())
    maximum = int(tensor.amax().item())
    if minimum < 0 or maximum >= upper:
        raise ValueError(f"{name} values must be in [0, {upper})")


def _require_offsets(
    tensor: torch.Tensor,
    expected: int,
    *,
    final: int,
    name: str,
) -> None:
    _require_index_vector(tensor, expected, upper=final + 1, name=name)
    if int(tensor[0].item()) != 0 or int(tensor[-1].item()) != final:
        raise ValueError(f"{name} must start at zero and end at {final}")
    if bool(torch.any(tensor[1:] < tensor[:-1]).item()):
        raise ValueError(f"{name} must be non-decreasing")


def _validate_muscle_topology(
    dimensions: MuscleDimensionsSpec,
    topology: MuscleTopologySpec,
) -> None:
    muscle_count = dimensions.muscle_count
    body_count = dimensions.body_count
    waypoint_count = int(topology.muscle_waypoints_local.shape[0])
    attachment_count = int(topology.muscle_waypoint_attachment_body.numel())

    _require_matrix_shape(
        topology.muscle_params,
        muscle_count,
        5,
        name="muscle_params",
    )
    _require_vector_shape(
        topology.muscle_rest_length,
        muscle_count,
        name="muscle_rest_length",
    )
    _require_vector_shape(
        topology.muscle_force_velocity_length_scale,
        muscle_count,
        name="muscle_force_velocity_length_scale",
    )
    _require_matrix_shape(
        topology.muscle_waypoints_local,
        waypoint_count,
        3,
        name="muscle_waypoints_local",
    )
    _require_index_vector(
        topology.muscle_waypoint_body,
        waypoint_count,
        upper=body_count,
        name="muscle_waypoint_body",
    )
    _require_offsets(
        topology.muscle_waypoint_attachment_start,
        waypoint_count + 1,
        final=attachment_count,
        name="muscle_waypoint_attachment_start",
    )
    _require_matrix_shape(
        topology.muscle_waypoint_attachment_local,
        attachment_count,
        3,
        name="muscle_waypoint_attachment_local",
    )
    _require_vector_shape(
        topology.muscle_waypoint_attachment_weight,
        attachment_count,
        name="muscle_waypoint_attachment_weight",
    )
    _require_index_vector(
        topology.muscle_waypoint_attachment_body,
        attachment_count,
        upper=body_count,
        name="muscle_waypoint_attachment_body",
    )

    all_segment_count = int(topology.all_segment_muscle_idx.numel())
    _require_index_vector(
        topology.all_segment_muscle_idx,
        all_segment_count,
        upper=muscle_count,
        name="all_segment_muscle_idx",
    )
    _require_index_vector(
        topology.all_segment_start_idx,
        all_segment_count,
        upper=max(waypoint_count - 1, 0),
        name="all_segment_start_idx",
    )

    apply_segment_count = int(topology.apply_segment_muscle_idx.numel())
    _require_index_vector(
        topology.apply_segment_muscle_idx,
        apply_segment_count,
        upper=muscle_count,
        name="apply_segment_muscle_idx",
    )
    for name, tensor in (
        ("apply_segment_start_idx", topology.apply_segment_start_idx),
        ("apply_segment_end_idx", topology.apply_segment_end_idx),
    ):
        _require_index_vector(
            tensor,
            apply_segment_count,
            upper=waypoint_count,
            name=name,
        )
    for name, tensor in (
        ("apply_segment_body_start_idx", topology.apply_segment_body_start_idx),
        ("apply_segment_body_end_idx", topology.apply_segment_body_end_idx),
    ):
        _require_index_vector(
            tensor,
            apply_segment_count,
            upper=body_count,
            name=name,
        )

    _require_index_vector(
        topology.body_parentid,
        body_count,
        upper=body_count,
        name="body_parentid",
    )
    _require_index_vector(
        topology.body_rootid,
        body_count,
        upper=body_count,
        name="body_rootid",
    )
    _require_index_vector(
        topology.dof_bodyid,
        dimensions.full_torque_dim,
        upper=body_count,
        name="dof_bodyid",
    )

    wraps = topology.wraps
    if wraps is not None:
        for body in (wraps.body, wraps.side_body):
            if bool((body >= body_count).any()):
                raise ValueError("Wrap body index exceeds the physical model")
        starts = wraps.start_point.long()
        if bool(((starts < 0) | (starts + 3 >= waypoint_count)).any()):
            raise ValueError("A wrap requires four consecutive points A/T1/T2/B")
        if topology.uses_lbs_waypoint_attachments:
            raise ValueError("Wrapped paths require body-fixed attachments")
        if not torch.equal(
            topology.muscle_waypoint_body[starts + 1].long(), wraps.body.long()
        ) or not torch.equal(
            topology.muscle_waypoint_body[starts + 2].long(), wraps.body.long()
        ):
            raise ValueError("Wrap tangents must be attached to the wrapping body")
        # All three edges must belong to one muscle, with canonical contiguous paths.
        edge_muscle = torch.full(
            (waypoint_count,), -1, dtype=torch.long, device=starts.device
        )
        edge_muscle[topology.all_segment_start_idx.long()] = (
            topology.all_segment_muscle_idx.long()
        )
        if (
            bool((edge_muscle[starts] < 0).any())
            or not torch.equal(edge_muscle[starts], edge_muscle[starts + 1])
            or not torch.equal(edge_muscle[starts], edge_muscle[starts + 2])
        ):
            raise ValueError("Wrap tangent slots must lie within one muscle path")
        counts = torch.bincount(
            topology.all_segment_muscle_idx.long(), minlength=muscle_count
        )
        offsets = torch.cat((counts.new_zeros(1), (counts + 1).cumsum(0)))
        expected_starts = torch.cat(
            [
                torch.arange(
                    int(offsets[i]), int(offsets[i + 1]) - 1, device=starts.device
                )
                for i in range(muscle_count)
            ]
        )
        expected_muscles = torch.repeat_interleave(
            torch.arange(muscle_count, device=starts.device), counts
        )
        if (
            int(offsets[-1]) != waypoint_count
            or not torch.equal(topology.all_segment_start_idx.long(), expected_starts)
            or not torch.equal(topology.all_segment_muscle_idx.long(), expected_muscles)
        ):
            raise ValueError(
                "Wrapped muscles require contiguous ordered waypoint paths"
            )


@dataclass(frozen=True, slots=True, eq=False)
class MuscleLayerRuntimeSpec:
    dimensions: MuscleDimensionsSpec
    output: MuscleOutputSpec
    mechanics: MuscleMechanicsRuntimeSpec
    projection: MuscleProjectionRuntimeSpec
    topology: MuscleTopologySpec

    def __post_init__(self) -> None:
        active_torque_indices = self.output.active_torque_indices
        if active_torque_indices is not None:
            active_count = int(active_torque_indices.numel())
            _require_index_vector(
                active_torque_indices,
                active_count,
                upper=self.dimensions.torque_dim,
                name="active_torque_indices",
            )
            if torch.unique(active_torque_indices).numel() != active_count:
                raise ValueError("active_torque_indices must not contain duplicates")
        _validate_muscle_topology(self.dimensions, self.topology)


__all__ = (
    "MuscleDimensionsSpec",
    "MuscleLayerRuntimeSpec",
    "MuscleMechanicsRuntimeSpec",
    "MuscleOutputSpec",
    "MuscleProjectionRuntimeSpec",
    "MuscleTopologySpec",
    "WaypointWrapSpec",
    "resolve_muscle_length_range",
)
