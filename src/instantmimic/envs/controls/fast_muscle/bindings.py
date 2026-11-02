from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from instantmimic.envs.config_access import get_value
from instantmimic.envs.dof_roles import DofRolesDeclaration


@dataclass(frozen=True, slots=True)
class FastMuscleSimulationBindings:
    """Host-owned simulation objects and scalar facts borrowed at attachment."""

    model: Any
    mjw_model: Any
    mjw_data: Any
    device: torch.device
    num_envs: int
    physics_dt: float
    step_dt: float

    @classmethod
    def from_host(cls, host: Any) -> FastMuscleSimulationBindings:
        device = torch.device(host.device)
        num_envs = int(host.num_envs)
        physics_dt = float(host.physics_dt)
        step_dt = float(host.step_dt)
        if num_envs <= 0:
            raise ValueError(f"FastMuscle requires num_envs > 0, got {num_envs}.")
        if not math.isfinite(physics_dt) or physics_dt <= 0.0:
            raise ValueError(
                f"FastMuscle requires a positive finite physics_dt, got {physics_dt}."
            )
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError(
                f"FastMuscle requires a positive finite step_dt, got {step_dt}."
            )
        return cls(
            model=host.model,
            mjw_model=host.mjw_model,
            mjw_data=host.mjw_data,
            device=device,
            num_envs=num_envs,
            physics_dt=physics_dt,
            step_dt=step_dt,
        )


@dataclass(frozen=True, slots=True)
class FastMuscleRuntimeInputs:
    """Normalized scalar inputs captured from host config at attachment."""

    control_interval_substeps: int
    # The host's substep kind (implicitfast, euler, rk4 or instantaneous), not
    # the configured sim.integrator name.
    integrator: str
    optimized_kernel_enabled: bool
    # Coordinate roles (instantmimic.envs.dof_roles): the controller derives
    # them from the compiled model with the host's root body names and the
    # env.dof_roles declaration, the same inputs the host task uses.
    root_body_names: tuple[str, ...] | None = None
    dof_roles: DofRolesDeclaration = DofRolesDeclaration()

    @classmethod
    def from_cfg(cls, cfg: Any, *, substep_kind: str) -> FastMuscleRuntimeInputs:
        """``substep_kind`` is the host's ``DirectRLEnv.substep_kind``."""
        control_interval_substeps = int(cfg.decimation)
        if control_interval_substeps <= 0:
            raise ValueError(
                "FastMuscle requires env.decimation > 0, got "
                f"{control_interval_substeps}."
            )
        root_body_names = get_value(cfg, "root_body_names", None)
        if isinstance(root_body_names, str):
            root_body_names = (root_body_names,)
        elif root_body_names is not None:
            root_body_names = tuple(str(name) for name in root_body_names)
        return cls(
            control_interval_substeps=control_interval_substeps,
            integrator=str(substep_kind),
            optimized_kernel_enabled=bool(getattr(cfg, "use_optimized_kernel", False)),
            root_body_names=root_body_names,
            dof_roles=DofRolesDeclaration.from_cfg(cfg),
        )


@dataclass(frozen=True, slots=True)
class FastMuscleActionBufferBindings:
    """Explicit host operations needed to publish and protect action storage."""

    current_actions: Callable[[], torch.Tensor | None]
    publish_actions: Callable[[torch.Tensor], None]
    graph_capture_started: Callable[[], bool]

    @classmethod
    def from_host(cls, host: Any) -> FastMuscleActionBufferBindings:
        def current_actions() -> torch.Tensor | None:
            actions = getattr(host, "actions", None)
            return actions if isinstance(actions, torch.Tensor) else None

        def publish_actions(actions: torch.Tensor) -> None:
            host.actions = actions

        def graph_capture_started() -> bool:
            return (
                bool(getattr(host, "_step_lifecycle_registration_frozen", False))
                or bool(getattr(host, "_step_graph_cache", None))
                or getattr(host, "_step_graph", None) is not None
            )

        return cls(
            current_actions=current_actions,
            publish_actions=publish_actions,
            graph_capture_started=graph_capture_started,
        )


@dataclass(frozen=True, slots=True)
class FastMuscleJointTopologyBindings:
    """Immutable joint metadata snapshot supplied after host model mapping."""

    hinge_dof_indices: torch.Tensor
    hinge_body_indices: torch.Tensor
    hinge_parent_body_indices: torch.Tensor
    hinge_axis_local: torch.Tensor
    ball_dof_scatter: torch.Tensor

    @classmethod
    def from_host(cls, host: Any) -> FastMuscleJointTopologyBindings:
        def snapshot(name: str, *, dtype: torch.dtype) -> torch.Tensor:
            value = getattr(host, name, None)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"FastMuscle joint topology requires host.{name}")
            return value.detach().clone().to(dtype=dtype)

        result = cls(
            hinge_dof_indices=snapshot("hinge_dof_indices", dtype=torch.long),
            hinge_body_indices=snapshot("hinge_body_indices", dtype=torch.long),
            hinge_parent_body_indices=snapshot(
                "hinge_parent_body_indices", dtype=torch.long
            ),
            hinge_axis_local=snapshot("hinge_axis_local", dtype=torch.float32),
            ball_dof_scatter=snapshot("ball_dof_scatter", dtype=torch.long),
        )
        hinge_count = int(result.hinge_dof_indices.numel())
        if tuple(result.hinge_axis_local.shape) != (hinge_count, 3):
            raise ValueError(
                "FastMuscle hinge_axis_local must have shape "
                f"({hinge_count}, 3), got {tuple(result.hinge_axis_local.shape)}."
            )
        for name in ("hinge_body_indices", "hinge_parent_body_indices"):
            if int(getattr(result, name).numel()) != hinge_count:
                raise ValueError(
                    f"FastMuscle {name} must contain {hinge_count} entries."
                )
        if int(result.ball_dof_scatter.numel()) % 3 != 0:
            raise ValueError(
                "FastMuscle ball_dof_scatter must contain three entries per joint."
            )
        return result


@dataclass(frozen=True, slots=True)
class FastMuscleExecutionBindings:
    """Repository-owned runtime operations available after host initialization."""

    refresh_full_dynamics: Callable[[], None]
    refresh_control_dynamics: Callable[[], None]
    install_implicit_velocity_derivative_hook: (
        Callable[[Callable[..., None]], None] | None
    )

    @classmethod
    def from_host(cls, host: Any) -> FastMuscleExecutionBindings:
        operation_names = (
            "_run_mjw_forward",
            "_run_mjw_control_forward",
        )
        operations = {name: getattr(host, name, None) for name in operation_names}
        missing = [
            name for name, operation in operations.items() if not callable(operation)
        ]
        if missing:
            raise RuntimeError(
                "FastMuscle execution binding requires repository-owned host "
                f"operations; missing {missing}."
            )
        return cls(
            refresh_full_dynamics=operations["_run_mjw_forward"],
            refresh_control_dynamics=operations["_run_mjw_control_forward"],
            install_implicit_velocity_derivative_hook=getattr(
                host,
                "_install_mjw_implicit_velocity_derivative_hook",
                None,
            ),
        )


@dataclass(frozen=True, slots=True)
class FastMuscleMotionReferenceBindings:
    """Live motion-reference tensors borrowed as one optional capability."""

    motion_qpos: torch.Tensor | None
    motion_ctrl: torch.Tensor | None
    reference_body_state: torch.Tensor | None
    ref_frame_idx: torch.Tensor
    motion_frame_count: int
    multi_motion_enabled: bool
    motion_clip_idx: torch.Tensor | None = None
    motion_clip_start: torch.Tensor | None = None
    motion_clip_length: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.motion_frame_count <= 0:
            raise ValueError(
                "FastMuscle motion_frame_count must be > 0, got "
                f"{self.motion_frame_count}."
            )
        if self.multi_motion_enabled:
            missing = [
                name
                for name in (
                    "motion_clip_idx",
                    "motion_clip_start",
                    "motion_clip_length",
                )
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"FastMuscle multi-motion binding is incomplete; missing {missing}."
                )
