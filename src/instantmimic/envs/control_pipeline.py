from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import mujoco
import torch


@dataclass(frozen=True, slots=True)
class ActionScaleSnapshot:
    """Current policy scales exposed by an environment control pipeline.

    Producers must detach its tensors from evolving controller-owned buffers.
    Consumers must request a fresh snapshot after an environment-side scale
    update instead of caching this point-in-time value.
    """

    mean_scale: torch.Tensor | None = None
    std_scale: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class PolicyTransformSemantics:
    squash_actions: bool = False
    clamp_mean: bool = False
    clamp_mean_min: float = -1.0
    clamp_mean_max: float = 1.0


@dataclass(frozen=True)
class ControlRuntimeSpec:
    action_mode: str
    policy_action_dim: int
    policy_action_scale_snapshot: ActionScaleSnapshot | None = None
    squash_actions: bool = False
    clamp_mean: bool = False
    clamp_mean_min: float = -1.0
    clamp_mean_max: float = 1.0


def make_direct_control_runtime_spec(
    *,
    action_dim: int,
    action_mode: str = "direct",
    policy_action_scale_snapshot: ActionScaleSnapshot | None = None,
    policy_transform: PolicyTransformSemantics = PolicyTransformSemantics(),
) -> ControlRuntimeSpec:
    """Describe a policy that acts directly on an environment action space."""

    return ControlRuntimeSpec(
        action_mode=str(action_mode),
        policy_action_dim=int(action_dim),
        policy_action_scale_snapshot=policy_action_scale_snapshot,
        squash_actions=policy_transform.squash_actions,
        clamp_mean=policy_transform.clamp_mean,
        clamp_mean_min=policy_transform.clamp_mean_min,
        clamp_mean_max=policy_transform.clamp_mean_max,
    )


@runtime_checkable
class ControlPipeline(Protocol):
    @property
    def is_enabled(self) -> bool: ...

    def build_sim_spec(
        self,
        mjcf_path: str,
        default_builder: Callable[[str], mujoco.MjSpec],
    ) -> mujoco.MjSpec: ...

    def external_sim_geom_dependencies(self) -> frozenset[str]:
        """Named geoms read outside MuJoCo; valid after ``build_sim_spec``."""
        ...

    def finalize_after_model_compile(self, env: Any) -> None: ...

    def configure_spaces(self, env: Any) -> None: ...

    def pre_physics_step(self, actions: torch.Tensor) -> None: ...

    def apply_action(self) -> None: ...

    def reset_idx(self, reset_mask: torch.Tensor) -> None: ...

    def runtime_spec(self) -> ControlRuntimeSpec | None: ...

    def copy_runtime_state_from(self, source: ControlPipeline) -> bool: ...

    def post_physics_substep(self, substep_idx: int) -> None: ...

    def post_physics_substep_reads_geometry(self) -> bool:
        """Whether ``post_physics_substep`` reads the refreshed endpoint geometry."""
        ...

    def post_physics_substep_writes_state(self) -> bool:
        """Whether ``post_physics_substep`` changes state the next substep reads."""
        ...

    def full_video_activation_width(self) -> int: ...

    def get_full_video_activation(self) -> torch.Tensor | None: ...


class NoControlPipeline:
    @property
    def is_enabled(self) -> bool:
        return False

    def build_sim_spec(
        self,
        mjcf_path: str,
        default_builder: Callable[[str], mujoco.MjSpec],
    ) -> mujoco.MjSpec:
        return default_builder(mjcf_path)

    def external_sim_geom_dependencies(self) -> frozenset[str]:
        return frozenset()

    def finalize_after_model_compile(self, env: Any) -> None:
        return None

    def configure_spaces(self, env: Any) -> None:
        return None

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        raise NotImplementedError("No control pipeline is enabled.")

    def apply_action(self) -> None:
        raise NotImplementedError("No control pipeline is enabled.")

    def reset_idx(self, reset_mask: torch.Tensor) -> None:
        return None

    def runtime_spec(self) -> ControlRuntimeSpec | None:
        return None

    def copy_runtime_state_from(self, source: ControlPipeline) -> bool:
        del source
        return False

    def post_physics_substep(self, substep_idx: int) -> None:
        del substep_idx

    def _overrides_post_physics_substep(self) -> bool:
        return (
            type(self).post_physics_substep
            is not NoControlPipeline.post_physics_substep
        )

    def post_physics_substep_reads_geometry(self) -> bool:
        # An overriding pipeline must declare what its callback reads.
        return self._overrides_post_physics_substep()

    def post_physics_substep_writes_state(self) -> bool:
        return self._overrides_post_physics_substep()

    def full_video_activation_width(self) -> int:
        return 0

    def get_full_video_activation(self) -> torch.Tensor | None:
        return None


def resolve_control_pipeline(control_cfg: Any) -> ControlPipeline:
    if control_cfg is None:
        return NoControlPipeline()
    if isinstance(control_cfg, ControlPipeline):
        return control_cfg
    factory = getattr(control_cfg, "build_control_pipeline", None)
    if callable(factory):
        pipeline = factory()
        if not isinstance(pipeline, ControlPipeline):
            raise TypeError(
                "control.build_control_pipeline() must return a ControlPipeline."
            )
        return pipeline
    raise TypeError(
        "cfg.control must be None, a ControlPipeline, or expose "
        "build_control_pipeline()."
    )
