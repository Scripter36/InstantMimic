from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp

from instantmimic.envs.common import warp_to_torch
from instantmimic.envs.dof_roles import (
    DofRolesDeclaration,
    ModelDofRoles,
    derive_dof_roles,
    root_body_from_names,
)
from instantmimic.muscle.parsing.mjcf import FastMuscleBootstrap
from instantmimic.utils.startup_profiler import (
    StartupProfiler,
    startup_nvtx_range,
    startup_nvtx_wrapped,
)

from .bindings import (
    FastMuscleActionBufferBindings,
    FastMuscleExecutionBindings,
    FastMuscleJointTopologyBindings,
    FastMuscleMotionReferenceBindings,
    FastMuscleRuntimeInputs,
    FastMuscleSimulationBindings,
)
from .config_models import (
    MassActionSettings,
    MassPosePolicyOutputSettings,
)
from .constants import (
    VALID_ACTION_MODES,
)
from .geometry_runtime import FastMuscleGeometryMixin
from .observation_runtime import FastMuscleObservationMixin
from .settings import FastMuscleControlSettings, activation_interval_bounds
from .step_runtime import FastMuscleStepMixin
from .torque_runtime import FastMuscleDynamicsMixin
from .training_runtime import FastMuscleTrainingMixin

__all__ = [
    "FastMuscleControlSettings",
    "FastMuscleController",
    "FastMuscleMechanicsView",
    "FastMuscleRenderGeometryView",
    "activation_interval_bounds",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FastMuscleMechanicsView:
    """Borrowed live mechanics tensors; callers must not mutate them."""

    muscle_names: tuple[str, ...]
    parameters: torch.Tensor
    rest_length: torch.Tensor


@dataclass(frozen=True, slots=True)
class PerStepCommand:
    """Borrowed per-control-step command: column j is values[:, j] * scale[index[j]]."""

    source: str
    values: torch.Tensor
    column_scale: torch.Tensor
    column_index: torch.Tensor


@dataclass(frozen=True, slots=True)
class FastMuscleRenderGeometryView:
    """Borrowed render buffers valid until the controller updates them again."""

    muscle_names: tuple[str, ...]
    points: torch.Tensor
    point_offsets: torch.Tensor
    point_valid: torch.Tensor | None
    activation: torch.Tensor


def _resolve_dof_roles(
    model: mujoco.MjModel,
    *,
    root_body_names: tuple[str, ...] | None = None,
    declaration: DofRolesDeclaration | None = None,
) -> ModelDofRoles:
    """Coordinate roles of the controlled model (instantmimic.envs.dof_roles).

    The torque space is the non-root DOF space ``[root, nv)``: the root DOFs
    must lead and be non-empty. Passive DOFs inside it stay unactuated (zero
    torque scale) and keep their free dynamics.
    """
    if int(model.njnt) <= 0:
        raise ValueError("FastMuscle requires a model with at least one joint.")
    roles = derive_dof_roles(
        model,
        root_body=root_body_from_names(model, root_body_names),
        declaration=declaration,
    )
    if roles.root_dof_count <= 0:
        raise ValueError("FastMuscle requires a root body with DOFs.")
    roles.require_root_leads("FastMuscle")
    return roles


class FastMuscleController(
    FastMuscleGeometryMixin,
    FastMuscleObservationMixin,
    FastMuscleDynamicsMixin,
    FastMuscleTrainingMixin,
    FastMuscleStepMixin,
):
    """FastMuscle runtime attached through :meth:`attach_to_host`.

    The raw constructor is an internal assembly boundary for validated bindings;
    first-party environments and integrations must use ``attach_to_host``.
    """

    _IMMUTABLE_INPUT_ATTRS = frozenset(
        {
            "_runtime_inputs",
            "_action_buffer_bindings",
            "model",
            "mjw_model",
            "mjw_data",
            "device",
            "num_envs",
            "physics_dt",
            "step_dt",
            "control_interval_substeps",
            "integrator",
            "optimized_kernel_enabled",
        }
    )
    _ACTIVATION_LOWER_BOUND = 0.0
    _ACTIVATION_UPPER_BOUND = 1.0

    def __init__(
        self,
        *,
        bootstrap: FastMuscleBootstrap,
        action_buffers: FastMuscleActionBufferBindings,
        simulation: FastMuscleSimulationBindings,
        runtime_inputs: FastMuscleRuntimeInputs,
        settings: FastMuscleControlSettings,
    ) -> None:
        object.__setattr__(self, "bootstrap", bootstrap)
        object.__setattr__(self, "_action_buffer_bindings", action_buffers)
        object.__setattr__(self, "_runtime_inputs", runtime_inputs)
        object.__setattr__(self, "model", simulation.model)
        object.__setattr__(self, "mjw_model", simulation.mjw_model)
        object.__setattr__(self, "mjw_data", simulation.mjw_data)
        object.__setattr__(self, "device", simulation.device)
        object.__setattr__(self, "num_envs", simulation.num_envs)
        object.__setattr__(self, "physics_dt", simulation.physics_dt)
        object.__setattr__(self, "step_dt", simulation.step_dt)
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "_execution_bindings", None)
        object.__setattr__(self, "_joint_topology", None)
        object.__setattr__(self, "_motion_reference", None)
        object.__setattr__(self, "_runtime_state_finalized", False)
        self._initialize_host_control_state()

    def bind_joint_topology(self, topology: FastMuscleJointTopologyBindings) -> None:
        if self._runtime_state_finalized:
            raise RuntimeError(
                "FastMuscle joint topology cannot change after runtime finalization."
            )
        object.__setattr__(self, "_joint_topology", topology)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._IMMUTABLE_INPUT_ATTRS:
            raise AttributeError(
                f"{name} is captured when FastMuscle attaches and cannot be replaced"
            )
        object.__setattr__(self, name, value)

    @classmethod
    def attach_to_host(
        cls,
        env: Any,
        cfg: Any,
        bootstrap: FastMuscleBootstrap,
        *,
        settings: FastMuscleControlSettings,
    ) -> FastMuscleController:
        simulation = FastMuscleSimulationBindings.from_host(env)
        action_buffers = FastMuscleActionBufferBindings.from_host(env)
        runtime_inputs = FastMuscleRuntimeInputs.from_cfg(
            cfg, substep_kind=env.substep_kind
        )
        controller = cls(
            bootstrap=bootstrap,
            action_buffers=action_buffers,
            simulation=simulation,
            runtime_inputs=runtime_inputs,
            settings=settings,
        )
        return controller

    def bind_action_buffer(self, actions: torch.Tensor) -> None:
        """Publish the host-owned action tensor and its Warp alias atomically.

        Callers must replace the tensor through this method instead of changing the
        storage of an already-bound tensor in place.
        """
        if not isinstance(actions, torch.Tensor):
            raise TypeError("FastMuscle action buffer must be a torch.Tensor.")
        action_buffers = self._action_buffer_bindings
        current = self.__dict__.get("_bound_action_buffer")
        current_fingerprint = self.__dict__.get("_bound_action_buffer_fingerprint")
        fingerprint = (
            actions.data_ptr(),
            actions.storage_offset(),
            tuple(actions.shape),
            tuple(actions.stride()),
            actions.dtype,
            actions.device,
        )
        graph_capture_started = action_buffers.graph_capture_started()
        if (
            current is actions
            and current_fingerprint is not None
            and current_fingerprint != fingerprint
            and graph_capture_started
        ):
            storage_guard = self.__dict__["_bound_action_storage_guard"]
            actions.set_(storage_guard)
            raise RuntimeError(
                "FastMuscle action buffer storage cannot change after graph capture; "
                "the previous storage binding was restored."
            )
        expected_shape = (int(self.num_envs), int(self._policy_action_dim))
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                "FastMuscle action buffer shape must be "
                f"{expected_shape}, got {tuple(actions.shape)}."
            )
        if actions.dtype is not torch.float32:
            raise TypeError(
                "FastMuscle action buffer dtype must be torch.float32, "
                f"got {actions.dtype}."
            )
        expected_device = self.__dict__.get("_action_buffer_device")
        if expected_device is None:
            host_actions = action_buffers.current_actions()
            if isinstance(host_actions, torch.Tensor):
                expected_device = host_actions.device
            else:
                expected_device = torch.device(self.device)
                if expected_device.type == "cuda" and expected_device.index is None:
                    expected_device = torch.device("cuda", torch.cuda.current_device())
        if actions.device != expected_device:
            raise ValueError(
                "FastMuscle action buffer device must match the controller device, "
                f"got {actions.device} and {self.device}."
            )
        if not actions.is_contiguous():
            raise ValueError("FastMuscle action buffer must be contiguous.")

        if (
            current is actions
            and current_fingerprint == fingerprint
            and action_buffers.current_actions() is actions
        ):
            return
        if current is not None and graph_capture_started:
            raise RuntimeError(
                "FastMuscle action buffer cannot change after graph capture."
            )

        actions_wp = wp.from_torch(actions, dtype=wp.float32)
        if int(actions_wp.ptr) != int(actions.data_ptr()):
            raise RuntimeError("FastMuscle action Warp alias does not share storage.")
        storage_guard = actions.detach()
        action_buffers.publish_actions(actions)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "_bound_action_buffer", actions)
        object.__setattr__(self, "_bound_action_buffer_fingerprint", fingerprint)
        object.__setattr__(self, "_bound_action_storage_guard", storage_guard)
        object.__setattr__(self, "_action_buffer_device", actions.device)
        object.__setattr__(self, "_fast_muscle_actions_wp", actions_wp)

    def bind_execution(self, bindings: FastMuscleExecutionBindings) -> None:
        """Bind repository-owned runtime operations before state finalization."""

        if self.runtime_state_finalized:
            raise RuntimeError(
                "FastMuscle execution operations cannot be rebound after finalization."
            )
        current = self.__dict__.get("_execution_bindings")
        if current is bindings:
            return
        if current is not None:
            raise RuntimeError("FastMuscle execution operations are already bound.")
        object.__setattr__(self, "_execution_bindings", bindings)

    def bind_dof_roles(self, roles: ModelDofRoles) -> None:
        """Validate the task's coordinate roles against the controller's.

        Both sides derive them from the same model and declaration; a mismatch
        (another root body, other passive coordinates) is a wiring error.
        """
        if self._runtime_state_finalized:
            raise RuntimeError("FastMuscle DOF roles cannot change after finalization.")
        if int(roles.root_dof_count) != int(self._root_dof_count):
            raise ValueError(
                "FastMuscle task root DOF count does not match the controller's: "
                f"task={roles.root_dof_count}, controller={self._root_dof_count}."
            )
        own = getattr(self, "_dof_roles", None)
        if own is not None and (
            tuple(roles.root_dof) != tuple(own.root_dof)
            or tuple(roles.passive_dof) != tuple(own.passive_dof)
        ):
            raise ValueError(
                "FastMuscle task DOF roles differ from the controller's: root "
                f"{list(roles.root_dof)} vs {list(own.root_dof)}, passive "
                f"{list(roles.passive_dof)} vs {list(own.passive_dof)}."
            )

    def bind_motion_reference(
        self,
        bindings: FastMuscleMotionReferenceBindings | None,
    ) -> None:
        """Bind one optional live motion capability before state finalization."""

        if self.runtime_state_finalized:
            raise RuntimeError(
                "FastMuscle motion reference cannot be rebound after finalization."
            )
        current = self.__dict__.get("_motion_reference")
        if current is bindings:
            return
        if current is not None:
            raise RuntimeError("FastMuscle motion reference is already bound.")
        if bindings is not None:
            expected_device = torch.device(self.device)
            if expected_device.type == "cuda" and expected_device.index is None:
                expected_device = torch.device("cuda", torch.cuda.current_device())
            tensor_names = (
                "motion_qpos",
                "motion_ctrl",
                "reference_body_state",
                "ref_frame_idx",
                "motion_clip_idx",
                "motion_clip_start",
                "motion_clip_length",
            )
            values = {name: getattr(bindings, name) for name in tensor_names}
            invalid_types = [
                name
                for name, value in values.items()
                if value is not None and not isinstance(value, torch.Tensor)
            ]
            if invalid_types:
                raise TypeError(
                    "FastMuscle motion fields must be torch.Tensor or None; "
                    f"invalid {invalid_types}."
                )
            if not isinstance(bindings.ref_frame_idx, torch.Tensor):
                raise TypeError("FastMuscle ref_frame_idx must be a torch.Tensor.")
            tensors = {
                name: value
                for name, value in values.items()
                if isinstance(value, torch.Tensor)
            }
            wrong_device = [
                name
                for name, tensor in tensors.items()
                if tensor.device != expected_device
            ]
            if wrong_device:
                raise ValueError(
                    "FastMuscle motion tensors must share the simulation device; "
                    f"wrong device for {wrong_device}."
                )
            wrong_float_dtype = [
                name
                for name in ("motion_qpos", "motion_ctrl", "reference_body_state")
                if (tensor := tensors.get(name)) is not None
                and tensor.dtype is not torch.float32
            ]
            wrong_index_dtype = [
                name
                for name in (
                    "ref_frame_idx",
                    "motion_clip_idx",
                    "motion_clip_start",
                    "motion_clip_length",
                )
                if (tensor := tensors.get(name)) is not None
                and tensor.dtype is not torch.int32
            ]
            if wrong_float_dtype or wrong_index_dtype:
                raise TypeError(
                    "FastMuscle motion tensors have incompatible dtypes; "
                    f"float32 required for {wrong_float_dtype}, "
                    f"int32 required for {wrong_index_dtype}."
                )
            if bindings.ref_frame_idx.shape != (self.num_envs,):
                raise ValueError(
                    "FastMuscle ref_frame_idx shape must be "
                    f"({self.num_envs},), got {tuple(bindings.ref_frame_idx.shape)}."
                )
            for name in ("motion_qpos", "motion_ctrl", "reference_body_state"):
                tensor = tensors.get(name)
                if tensor is not None and (
                    tensor.ndim < 2
                    or int(tensor.shape[0]) != bindings.motion_frame_count
                ):
                    raise ValueError(
                        f"FastMuscle {name} must start with "
                        f"({bindings.motion_frame_count}, ...), got "
                        f"{tuple(tensor.shape)}."
                    )
            reference_body_state = tensors.get("reference_body_state")
            if reference_body_state is not None and (
                reference_body_state.ndim != 3
                or int(reference_body_state.shape[1]) <= 0
                or int(reference_body_state.shape[2]) < 7
            ):
                raise ValueError(
                    "FastMuscle reference_body_state must have shape "
                    f"(frames, bodies, >=7), got {tuple(reference_body_state.shape)}."
                )
            motion_ctrl = tensors.get("motion_ctrl")
            if motion_ctrl is not None and int(motion_ctrl.shape[1]) != int(
                self.model.nu
            ):
                raise ValueError(
                    "FastMuscle motion_ctrl width must match model.nu, got "
                    f"{motion_ctrl.shape[1]} and {self.model.nu}."
                )
            clip_idx = tensors.get("motion_clip_idx")
            if clip_idx is not None and clip_idx.shape != (self.num_envs,):
                raise ValueError(
                    "FastMuscle motion_clip_idx shape must be "
                    f"({self.num_envs},), got {tuple(clip_idx.shape)}."
                )
            clip_start = tensors.get("motion_clip_start")
            clip_length = tensors.get("motion_clip_length")
            if (clip_start is None) != (clip_length is None):
                raise ValueError(
                    "FastMuscle motion_clip_start and motion_clip_length must "
                    "be supplied together."
                )
            if clip_start is not None and clip_length is not None:
                if (
                    clip_start.ndim != 1
                    or clip_length.ndim != 1
                    or clip_start.shape != clip_length.shape
                    or clip_start.numel() == 0
                ):
                    raise ValueError(
                        "FastMuscle motion clip start/length tensors must be "
                        "non-empty one-dimensional arrays with equal shape."
                    )
                if bool(torch.any(clip_start < 0).item()) or bool(
                    torch.any(clip_length <= 0).item()
                ):
                    raise ValueError(
                        "FastMuscle motion clips require non-negative starts and "
                        "positive lengths."
                    )
                if bool(
                    torch.any(
                        clip_start.to(torch.int64) + clip_length.to(torch.int64)
                        > bindings.motion_frame_count
                    ).item()
                ):
                    raise ValueError(
                        "FastMuscle motion clips exceed motion_frame_count."
                    )
                if clip_idx is not None and (
                    bool(torch.any(clip_idx < 0).item())
                    or bool(torch.any(clip_idx >= clip_length.numel()).item())
                ):
                    raise ValueError("FastMuscle motion_clip_idx is out of range.")
        object.__setattr__(self, "_motion_reference", bindings)

    def _require_execution_bindings(self) -> FastMuscleExecutionBindings:
        bindings = self._execution_bindings
        if bindings is None:
            raise RuntimeError(
                "FastMuscle execution operations must be bound before finalization."
            )
        return bindings

    def _require_motion_reference(
        self,
        purpose: str,
    ) -> FastMuscleMotionReferenceBindings:
        bindings = self._motion_reference
        if bindings is None:
            raise RuntimeError(f"{purpose} requires a bound motion reference.")
        return bindings

    def _require_motion_clip_bindings(
        self,
        purpose: str,
    ) -> FastMuscleMotionReferenceBindings:
        bindings = self._require_motion_reference(purpose)
        missing = [
            name
            for name in (
                "motion_clip_idx",
                "motion_clip_start",
                "motion_clip_length",
            )
            if getattr(bindings, name) is None
        ]
        if missing:
            raise RuntimeError(
                f"{purpose} requires motion clip tensors; missing {missing}."
            )
        return bindings

    def _validate_motion_requirements(self) -> None:
        if self._uses_mass_pose_coordinator and self._mass_pose_uses_motion_reference():
            motion = self._require_motion_reference("MASS pose output")
            if motion.motion_ctrl is None:
                raise RuntimeError("MASS pose output requires motion control targets.")
            if motion.multi_motion_enabled:
                self._require_motion_clip_bindings("multi-motion MASS pose output")

    def configure_observation_layout(self, main_dim: int) -> None:
        """Set the controller-owned main observation width."""

        resolved = int(main_dim)
        if resolved < 0:
            raise ValueError(
                f"FastMuscle main observation width must be >= 0, got {main_dim}."
            )
        object.__setattr__(self, "_obs_main_dim", resolved)

    def get_mechanics_view(self) -> FastMuscleMechanicsView:
        """Return borrowed live tensors describing muscle mechanics."""

        return FastMuscleMechanicsView(
            muscle_names=tuple(self.muscle_names),
            parameters=self.muscle_params,
            rest_length=self.muscle_rest_length,
        )

    def get_active_torque_indices(self) -> torch.Tensor:
        """Return the borrowed live index tensor for policy-controlled torques."""

        return self._active_torque_indices

    def get_applied_generalized_force(self) -> torch.Tensor:
        """Return the borrowed live generalized-force buffer."""

        return self._qfrc_applied_view

    def get_torque_action_scale(self) -> torch.Tensor:
        """Return the borrowed live torque-scale tensor."""

        return self._torque_action_scale

    def get_per_step_command(self, source: str) -> PerStepCommand:
        """Return the borrowed per-control-step command buffer for ``source``.

        ``"torque"``: the commanded joint torque of the torque-output modes
        (``torque``), ``actions[:, j] * scale[index[j]]`` in N m,
        read through the live torque scale. ``"pd_target"``:
        the absolute Stable-PD target of the MASS pose output, in actuator
        coordinates. ``"auto"`` picks ``torque`` for direct-torque control and
        ``pd_target`` for MASS pose output. Other modes raise ``ValueError``.
        Both buffers hold the current step's command once its first physics
        substep has applied it.
        """

        mode = str(self._action_mode)
        torque_ok = mode == "torque"
        pd_ok = bool(self._uses_mass_pose_coordinator)
        if source == "auto":
            source = "torque" if torque_ok else "pd_target" if pd_ok else ""
        if source == "torque" and torque_ok:
            return PerStepCommand(
                source="torque",
                values=self.actions,
                column_scale=self._torque_action_scale,
                column_index=self._active_torque_indices,
            )
        if source == "pd_target" and pd_ok:
            controller = self._stable_pd_controller
            if controller is None:
                raise RuntimeError("MASS pose output has no Stable-PD controller yet")
            width = int(controller.target_ctrl.shape[1])
            return PerStepCommand(
                source="pd_target",
                values=controller.target_ctrl,
                column_scale=torch.ones(width, device=self.device, dtype=torch.float32),
                column_index=torch.arange(width, device=self.device, dtype=torch.int32),
            )
        raise ValueError(
            f"no {source or 'per-step'} command for env.action.type {mode!r}: "
            "'torque' needs direct-torque control, 'pd_target' needs MASS with "
            "policy_output.type pose"
        )

    def refresh_muscle_dynamics_context_view(self) -> torch.Tensor:
        """Refresh and return borrowed MuscleNN context scratch."""

        return self._build_musclenn_raw_context()

    def refresh_render_geometry_view(
        self,
        *,
        env_idx: int = 0,
    ) -> FastMuscleRenderGeometryView:
        """Refresh and borrow one environment's backend-neutral muscle geometry."""

        resolved_env_idx = int(env_idx)
        if resolved_env_idx < 0 or resolved_env_idx >= int(self.num_envs):
            raise IndexError(
                f"FastMuscle render env index {resolved_env_idx} is outside "
                f"[0, {self.num_envs})."
            )
        activation = self.get_observation_muscle_activation()
        if activation is None:
            activation = self._zero_muscle_actions

        self._transform_muscle_waypoints()
        return FastMuscleRenderGeometryView(
            muscle_names=tuple(self.muscle_names),
            points=self.transformed_waypoints[resolved_env_idx],
            point_offsets=self.muscle_waypoint_start_i32,
            # Every waypoint of a fast-muscle path is a valid point.
            point_valid=None,
            activation=activation[resolved_env_idx],
        )

    def _render_geometry_sources(self) -> dict[str, Any]:
        """Buffers changed by body-fixed path rendering and activation colors."""

        sources: dict[str, Any] = {}
        for name in (
            "transformed_waypoints",
            "_activation_apply_buffer",
        ):
            value = getattr(self, name, None)
            if isinstance(value, (torch.Tensor, wp.array)):
                sources[name] = value
        return sources

    @contextmanager
    def preserve_render_geometry_state(self) -> Iterator[None]:
        """Leave every buffer changed by geometry rendering bitwise unchanged.

        A caller that renders a state the step did not leave (an export of a
        physics substep, or of the endpoint after the step's last apply) wraps
        its refreshes in this, so the next step reads the controller exactly as
        it would have without the render, whether or not it recomputes these
        buffers first. The copies live in buffers the first call allocates and
        later calls reuse; the step path never calls this. Blocks do not nest:
        an inner block would overwrite the outer block's copies.
        """

        sources = self._render_geometry_sources()
        cached = getattr(self, "_render_geometry_saved", None)
        if (
            cached is None
            or cached[0].keys() != sources.keys()
            or any(cached[0][name] is not value for name, value in sources.items())
        ):
            views = tuple(
                value if isinstance(value, torch.Tensor) else wp.to_torch(value)
                for value in sources.values()
                if (value.numel() if isinstance(value, torch.Tensor) else value.size)
                > 0
            )
            cached = (sources, views, tuple(torch.empty_like(v) for v in views))
            self._render_geometry_saved = cached
        _, views, saved = cached
        for view, copy in zip(views, saved, strict=True):
            copy.copy_(view)
        try:
            yield
        finally:
            for view, copy in zip(views, saved, strict=True):
                view.copy_(copy)

    @property
    def policy_action_dim(self) -> int:
        return int(self._policy_action_dim)

    @property
    def control_interval_substeps(self) -> int:
        return self._runtime_inputs.control_interval_substeps

    @property
    def integrator(self) -> str:
        return self._runtime_inputs.integrator

    @property
    def optimized_kernel_enabled(self) -> bool:
        return self._runtime_inputs.optimized_kernel_enabled

    @property
    def passive_torque_indices(self) -> tuple[int, ...]:
        """Passive DOFs (env.dof_roles) as torque-space indices."""
        return tuple(getattr(self, "_passive_torque_indices", ()))

    @property
    def character_torque_indices(self) -> tuple[int, ...]:
        """Character joint DOFs as torque-space indices (passive excluded)."""
        torque_dim = int(self.model.nv - self._root_dof_count)
        passive = set(self.passive_torque_indices)
        return tuple(i for i in range(torque_dim) if i not in passive)

    @property
    def action_mode(self) -> str:
        return str(self._action_mode)

    @property
    def requires_native_actuation_disabled(self) -> bool:
        return bool(self._uses_mass_pose_coordinator)

    @property
    def character_render_mode(self) -> str:
        return str(self._character_render_mode)

    @property
    def source_mjcf_path(self) -> str:
        return str(self._source_mjcf_path)

    @property
    def helper_obs_context_dim(self) -> int:
        return int(
            self._geometry_context_dim() + self._muscle_force_velocity_context_dim()
        )

    @property
    def muscle_geometry_observation_dim(self) -> int:
        if self._uses_mass_coordinator and self.settings.observations.muscle_geometry:
            return self._geometry_context_dim()
        return 0

    def write_muscle_geometry_observation(
        self,
        target: torch.Tensor,
        offset: int,
    ) -> int:
        """Append the body kinematics used by the ordinary MASS policy."""
        if self.muscle_geometry_observation_dim == 0:
            return offset
        return self._write_geometry_context_block(target, offset)

    def get_muscle_geometry_observation(self) -> torch.Tensor | None:
        if self.muscle_geometry_observation_dim == 0:
            return None
        target = self._muscle_geometry_observation_buffer
        if target is None:
            raise RuntimeError("Muscle geometry observations are not finalized")
        self._write_geometry_context_block(target, 0)
        return target

    def _resolve_musclenn_substep_sampling_mode(self) -> str:
        action = self.settings.action
        if not isinstance(action, MassActionSettings):
            return "all"
        mode = action.muscle_nn.substep_sampling_mode
        if mode not in ("all", "random"):
            raise ValueError(
                "musclenn_substep_sampling_mode must be one of {'all', 'random'}."
            )
        return mode

    def _reuse_musclenn_activation_across_substeps_enabled(self) -> bool:
        action = self.settings.action
        return bool(
            isinstance(action, MassActionSettings)
            and action.muscle_nn.reuse_activation_across_substeps
        )

    def _reject_slot_indexed_bound_paths(self) -> None:
        """Bound paths whose kernels read the policy action at its DOF index.

        With masked DOFs (passive DOFs or zero command scales) the
        policy action is compact: slot i is the i-th active DOF, not DOF i. The
        direct-torque feasible clip addresses
        ``actions[env, dof]`` over the whole non-root space, so they would read
        and write the wrong DOFs or past the action row.
        """
        unsupported = []
        # MASS builds a full-width torque command from the compact action and
        # clips that, so its clip addresses the DOFs correctly.
        if self._clip_torque_to_feasible_bounds and not self._uses_mass_coordinator:
            unsupported.append("action.bounds.env_clip")
        if unsupported:
            raise ValueError(
                f"{', '.join(unsupported)} cannot run with masked torque DOFs "
                f"({int(self.model.nv) - int(self._root_dof_count) - self._active_torque_count} "
                "passive or equality-masked): these paths index the policy action by "
                "DOF, and the policy action covers only the active DOFs."
            )

    def _helper_force_velocity_context_enabled(self) -> bool:
        return bool(self.settings.muscle.force_velocity.enabled)

    def _muscle_force_velocity_context_dim(self) -> int:
        return (
            int(self.model.nv) if self._helper_force_velocity_context_enabled() else 0
        )

    def get_prev_action_observation_tensor(self) -> torch.Tensor:
        return self._prev_action_obs

    @property
    def runtime_state_finalized(self) -> bool:
        return bool(self.__dict__.get("_runtime_state_finalized", False))

    @property
    def obs_main_dim(self) -> int:
        return int(getattr(self, "_obs_main_dim", 0))

    @property
    def prev_action_obs_dim(self) -> int:
        include_prev_actions = getattr(self, "_include_prev_actions_obs", None)
        if include_prev_actions is None:
            include_prev_actions = getattr(self, "_prev_action_obs", None) is not None
        if not bool(include_prev_actions):
            return 0
        return int(self._policy_action_dim)

    @property
    def applied_torque_obs_dim(self) -> int:
        return int(self._applied_torque_obs_dim)

    def _geometry_context_dim(self) -> int:
        return int(self.model.nbody) * 13 + int(self.model.nv) * 6

    def _uses_muscle_layer(self) -> bool:
        return self._action_mode != "torque"

    @property
    def _policy_outputs_activation(self) -> bool:
        return self._action_mode == "activation"

    @property
    def _policy_outputs_torque(self) -> bool:
        return self._action_mode == "torque" or bool(
            self._uses_mass_coordinator and not self._uses_mass_pose_coordinator
        )

    @property
    def _uses_mass_coordinator(self) -> bool:
        return self._action_mode == "mass"

    @property
    def _uses_mass_pose_coordinator(self) -> bool:
        action = self.settings.action
        return isinstance(action, MassActionSettings) and isinstance(
            action.policy_output,
            MassPosePolicyOutputSettings,
        )

    def _mass_pose_uses_motion_reference(self) -> bool:
        action = self.settings.action
        return (
            isinstance(action, MassActionSettings)
            and isinstance(action.policy_output, MassPosePolicyOutputSettings)
            and action.policy_output.reference == "motion"
        )

    def _uses_torque_feasible_bounds(self) -> bool:
        return bool(
            (self._policy_outputs_torque or self._uses_mass_pose_coordinator)
            and self._clip_torque_to_feasible_bounds
        )

    def _needs_torque_bound_state(self) -> bool:
        return bool(self._uses_mass_coordinator or self._uses_torque_feasible_bounds())

    def _needs_muscle_qfrc_projector_state(self) -> bool:
        return bool(
            self._policy_outputs_activation
            or self._needs_torque_bound_state()
            or self._uses_torque_passive_muscle_forces()
        )

    def _needs_env_muscle_qfrc_projector_state(self) -> bool:
        return bool(
            self._policy_outputs_activation
            or self._uses_mass_coordinator
            or self._uses_torque_feasible_bounds()
            or self._uses_torque_passive_muscle_forces()
        )

    def _uses_torque_passive_muscle_forces(self) -> bool:
        return bool(
            self._action_mode == "torque"
            and not self._uses_torque_feasible_bounds()
            and self._muscle_include_passive != 0.0
        )

    @startup_nvtx_wrapped("fast_muscle:initialize_host_control_state")
    def _initialize_host_control_state(self) -> None:
        action = self.settings.action
        muscle = self.settings.muscle
        observations = self.settings.observations
        visualization = self.settings.visualization
        bounds = self.settings.effective.bounds

        inputs = self._runtime_inputs
        self._dof_roles = _resolve_dof_roles(
            self.model,
            root_body_names=inputs.root_body_names,
            declaration=inputs.dof_roles,
        )
        self._root_dof_count = int(self._dof_roles.root_dof_count)
        self._passive_torque_indices = tuple(self._dof_roles.passive_dof_local)

        self._source_mjcf_path = str(self.bootstrap.source_mjcf_path)
        self._parsed_mjcf_muscle_data = self.bootstrap.parsed_mjcf_data
        self._geometry_backend = str(
            getattr(self.bootstrap, "geometry_backend", "waypoint")
        )
        if self._geometry_backend != "waypoint":
            raise ValueError(
                f"Unsupported fast-muscle geometry backend {self._geometry_backend!r}."
            )

        self._action_mode = action.type
        if self._action_mode not in VALID_ACTION_MODES:
            raise ValueError(
                f"action_mode must be one of {VALID_ACTION_MODES}, got {self._action_mode!r}."
            )

        length_min, length_max = self.settings.effective.length_range
        self._muscle_length_norm_min = float(length_min)
        self._muscle_length_norm_max = float(length_max)
        if not (
            self._muscle_length_norm_min < self._muscle_length_norm_max
            and self._muscle_length_norm_min > 0.0
        ):
            raise ValueError("muscle_length_norm_min/max must satisfy 0.0 < min < max.")

        self._muscle_include_passive = 1.0 if muscle.passive_force.env else 0.0
        self._muscle_active_force_type = muscle.active_force.type
        self._muscle_active_force_model_params = wp.vec3(
            0.0 if self._muscle_active_force_type == "gaussian" else 1.0,
            (
                float(muscle.active_force.gamma)
                if self._muscle_active_force_type == "gaussian"
                else float(muscle.active_force.lmin)
            ),
            (
                0.0
                if self._muscle_active_force_type == "gaussian"
                else float(muscle.active_force.lmax)
            ),
        )
        self._muscle_passive_force_type = muscle.passive_force.type
        self._muscle_passive_force_model_params = wp.vec3(
            0.0 if self._muscle_passive_force_type == "exponential" else 1.0,
            (
                float(muscle.passive_force.k_pe)
                if self._muscle_passive_force_type == "exponential"
                else float(muscle.passive_force.lmax)
            ),
            (
                float(muscle.passive_force.e_mo)
                if self._muscle_passive_force_type == "exponential"
                else float(muscle.passive_force.fpmax)
            ),
        )

        self._activation_vis_enabled = visualization.enabled
        self._character_render_mode = self.settings.character_render_mode
        self._tendon_visualizer: Any | None = None
        self._activation_clamp_enabled = 1.0

        low_color = np.asarray(visualization.color_low, dtype=np.float64).reshape(-1)
        high_color = np.asarray(visualization.color_high, dtype=np.float64).reshape(-1)
        if low_color.size not in (3, 4) or high_color.size not in (3, 4):
            raise ValueError(
                "activation_color_low/high must each contain exactly 3 or 4 values."
            )
        self._activation_color_low = np.clip(low_color, 0.0, 1.0)
        self._activation_color_high = np.clip(high_color, 0.0, 1.0)
        self._activation_color_under = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        self._activation_color_over = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        self._activation_color_alpha = float(visualization.alpha)
        if not (0.0 <= self._activation_color_alpha <= 1.0):
            raise ValueError("activation_color_alpha must be in [0, 1].")
        self._activation_line_width = float(visualization.line_width)
        if self._activation_line_width <= 0.0:
            raise ValueError("activation_line_width must be positive.")
        self._activation_color_interpolation = visualization.interpolation
        if self._activation_color_interpolation not in {"rgb", "hsv"}:
            raise ValueError("activation_color_interpolation must be 'rgb' or 'hsv'.")
        self._activation_line_composite_mode = (
            visualization.composite_mode.strip().lower()
        )
        if self._activation_line_composite_mode not in {
            "alpha",
            "max_activation",
            "unit_mean_alpha",
        }:
            raise ValueError(
                "activation_line_composite_mode must be 'alpha', 'max_activation', "
                "or 'unit_mean_alpha'."
            )

        self._load_muscle_data()

        self._clip_torque_to_feasible_bounds = bounds.env_clip

        torque_dim = int(self.model.nv - self._root_dof_count)
        if self._uses_mass_pose_coordinator:
            self._policy_action_dim = int(self.model.nu)
        elif self._policy_outputs_activation:
            self._policy_action_dim = self.muscle_count
        else:
            self._policy_action_dim = torque_dim

        # Zero command scales remove the coordinate from the policy action.
        self._torque_action_scale_cfg = self._derive_torque_action_scale()
        active_mask = self._torque_action_scale_cfg.abs() > 1.0e-8
        self._active_torque_indices = (
            active_mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        )
        self._active_torque_count = int(self._active_torque_indices.numel())
        self._has_masked_torque_dofs = self._active_torque_count < torque_dim
        if (
            self._policy_outputs_torque
            and not self._uses_mass_pose_coordinator
            and self._has_masked_torque_dofs
        ):
            logger.info(
                "torque_action_scale has %d zero entries → policy action dim "
                "reduced from %d to %d.",
                torque_dim - self._active_torque_count,
                torque_dim,
                self._active_torque_count,
            )
            self._policy_action_dim = self._active_torque_count
            self._reject_slot_indexed_bound_paths()

        self._observe_applied_torque = observations.applied_torque
        # Applied torques are observed on the character joint DOFs only.
        self._applied_torque_obs_dim = (
            len(self.character_torque_indices) if self._observe_applied_torque else 0
        )
        self._include_prev_actions_obs = observations.previous_action is not None

    @startup_nvtx_wrapped("fast_muscle:finalize_runtime_state")
    def finalize_runtime_state(self) -> None:
        if self.runtime_state_finalized:
            return
        self._require_execution_bindings()
        self._validate_motion_requirements()
        actions = self._action_buffer_bindings.current_actions()
        if actions is None:
            raise RuntimeError(
                "FastMuscle requires the host action buffer before finalization."
            )
        self.bind_action_buffer(actions)
        startup_profiler = StartupProfiler(
            range_factory=lambda name: startup_nvtx_range(
                f"fast_muscle_finalize:{name}"
            )
        )
        with startup_profiler.phase("core_views"):
            self._prev_action_obs = torch.zeros(
                (self.num_envs, int(self.prev_action_obs_dim)),
                device=self.device,
                dtype=torch.float32,
            )
            self._xpos_view = warp_to_torch(self.mjw_data.xpos)
            self._xquat_view = warp_to_torch(self.mjw_data.xquat)
            self._xipos_view = warp_to_torch(self.mjw_data.xipos)
            self._subtree_com_view = warp_to_torch(self.mjw_data.subtree_com)
            self._cdof_view = warp_to_torch(self.mjw_data.cdof)
            self._muscle_force_velocity_qvel_context_view = (
                warp_to_torch(self.mjw_data.qvel)
                if (self._helper_force_velocity_context_enabled())
                else None
            )
            self._xpos_flat_view = self._xpos_view.reshape(
                self.num_envs, int(self.model.nbody) * 3
            )
            self._xquat_flat_view = self._xquat_view.reshape(
                self.num_envs, int(self.model.nbody) * 4
            )
            self._xipos_flat_view = self._xipos_view.reshape(
                self.num_envs, int(self.model.nbody) * 3
            )
            self._subtree_com_flat_view = self._subtree_com_view.reshape(
                self.num_envs, int(self.model.nbody) * 3
            )
            self._cdof_flat_view = self._cdof_view.reshape(
                self.num_envs, int(self.model.nv) * 6
            )
            self._muscle_geometry_observation_buffer = (
                torch.empty(
                    (self.num_envs, self.muscle_geometry_observation_dim),
                    device=self.device,
                    dtype=torch.float32,
                )
                if self.muscle_geometry_observation_dim > 0
                else None
            )
            xfrc_applied_view = warp_to_torch(self.mjw_data.xfrc_applied)
            self._xfrc_applied_flat_view = xfrc_applied_view.reshape(
                self.num_envs,
                int(self.model.nbody) * 6,
            )
            self._qfrc_applied_view = warp_to_torch(self.mjw_data.qfrc_applied)
        if self._observe_applied_torque:
            self._qfrc_smooth_view = warp_to_torch(self.mjw_data.qfrc_smooth)
            self._qfrc_bias_view = warp_to_torch(self.mjw_data.qfrc_bias)
            self._applied_torque_buf = torch.zeros(
                (self.num_envs, int(self._applied_torque_obs_dim)),
                device=self.device,
                dtype=torch.float32,
            )
            self._applied_torque_torque_idx_wp = (
                wp.array(
                    self.character_torque_indices,
                    dtype=wp.int32,
                    device=str(self.device),
                )
                if self.passive_torque_indices
                else None
            )
        else:
            self._applied_torque_buf = None
        with startup_profiler.phase("activation_and_obs_buffers"):
            self._torque_action_scale = self._torque_action_scale_cfg
            self._torque_action_scale_wp = wp.from_torch(
                self._torque_action_scale, dtype=wp.float32
            )
            self._torque_action_scale_inv = torch.reciprocal(
                torch.clamp(self._torque_action_scale, min=1.0e-6)
            )
            self._active_torque_indices_wp = wp.from_torch(
                self._active_torque_indices, dtype=wp.int32
            )
            self._activation_apply_buffer = torch.zeros(
                (self.num_envs, self.muscle_count),
                device=self.device,
                dtype=torch.float32,
            )
            self._muscle_obs_velocity = torch.zeros(
                (self.num_envs, self.muscle_count),
                device=self.device,
                dtype=torch.float32,
            )
        with startup_profiler.phase("muscle_runtime_buffers"):
            self._activation_apply_buffer_wp = wp.from_torch(
                self._activation_apply_buffer, dtype=wp.float32
            )
            self._muscle_params_wp = wp.from_torch(self.muscle_params, dtype=wp.float32)
            self._muscle_rest_length_wp = wp.from_torch(
                self.muscle_rest_length, dtype=wp.float32
            )
            self._muscle_force_velocity_length_scale_wp = wp.from_torch(
                self.muscle_force_velocity_length_scale,
                dtype=wp.float32,
            )
            self._muscle_length_wp = wp.from_torch(self.muscle_length, dtype=wp.float32)
            self._muscle_force_magnitude_wp = wp.from_torch(
                self._muscle_force_magnitude, dtype=wp.float32
            )
        self._qfrc_applied_view_wp = wp.from_torch(
            self._qfrc_applied_view, dtype=wp.float32
        )
        auto_direct_torque_scale = bool(
            self._action_mode == "torque"
            and self._torque_action_scale_auto_from_rest_bounds
        )
        with startup_profiler.phase("muscle_segment_state"):
            if (
                self._uses_muscle_layer()
                or self._uses_torque_feasible_bounds()
                or auto_direct_torque_scale
            ):
                self._build_muscle_segment_state()
        with startup_profiler.phase("torque_bound_state"):
            self._initialize_torque_bound_state(
                force_projector=auto_direct_torque_scale
            )
            self._initialize_muscle_force_velocity_state()
        with startup_profiler.phase("mass_state"):
            self._initialize_mass_state()
        with startup_profiler.phase("muscle_dynamics_helper_state"):
            self._initialize_muscle_dynamics_helper_state()
            self._prepare_env_native_tile_projection()
        with startup_profiler.phase("auto_torque_action_scale"):
            self._initialize_auto_torque_action_scale_from_rest_bounds()
            self._refresh_musclenn_sparse_scale()
        object.__setattr__(self, "_runtime_state_finalized", True)
        startup_profiler.log_summary(
            logger=logger,
            prefix="FAST_MUSCLE_FINALIZE_PROFILE",
        )
