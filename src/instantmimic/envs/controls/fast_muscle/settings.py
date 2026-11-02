from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from instantmimic.muscle.runtime_specs import resolve_muscle_length_range
from instantmimic.rendering.character import resolve_character_render_mode

from .config_models import (
    ActionConfig,
    ActionSettings,
    ActivationVisualizationSettings,
    FastMuscleObservationSettings,
    MassActionSettings,
    MassPosePolicyOutputSettings,
    MuscleNNSettings,
    MuscleSettings,
    TorqueActionSettings,
    TorqueBoundsSettings,
)
from .constants import FORCE_VELOCITY_ACTION_MODES

_ACTIVATION_LOWER_BOUND = 0.0
_ACTIVATION_UPPER_BOUND = 1.0


def activation_interval_bounds(
    muscle_dynamics_contribution: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = torch.clamp(muscle_dynamics_contribution, min=0.0)
    negative = torch.clamp(muscle_dynamics_contribution, max=0.0)
    lower = (
        positive.sum(dim=-1) * _ACTIVATION_LOWER_BOUND
        + negative.sum(dim=-1) * _ACTIVATION_UPPER_BOUND
    )
    upper = (
        positive.sum(dim=-1) * _ACTIVATION_UPPER_BOUND
        + negative.sum(dim=-1) * _ACTIVATION_LOWER_BOUND
    )
    return lower, upper


@dataclass(frozen=True, slots=True)
class AutoScale:
    """A scale resolved from runtime mechanics rather than authored numbers."""


@dataclass(frozen=True, slots=True)
class ScalarScale:
    value: float


@dataclass(frozen=True, slots=True)
class VectorScale:
    values: tuple[float, ...]


type AuthoredScale = AutoScale | ScalarScale | VectorScale
type ExplicitScale = ScalarScale | VectorScale


@dataclass(frozen=True, slots=True)
class FastMuscleEffectiveSemantics:
    """Mode-aware defaults genuinely shared by multiple runtime consumers."""

    bounds: TorqueBoundsSettings
    action_scale: AuthoredScale | None
    # muscle.length_range, or the default of muscle.dynamics.model when unset.
    length_range: tuple[float, float]
    # MASS MuscleNN settings with the muscle-model defaults filled in; None for
    # every other action type.
    mass_muscle_nn: MuscleNNSettings | None = None


def _resolve_bounds(action: ActionConfig) -> TorqueBoundsSettings:
    if isinstance(
        action,
        (
            TorqueActionSettings,
            MassActionSettings,
        ),
    ):
        return action.bounds
    return TorqueBoundsSettings()


def _authored_scale(action: ActionConfig) -> str | float | tuple[float, ...]:
    if isinstance(action, MassActionSettings):
        return action.policy_output.scale
    if isinstance(
        action,
        (TorqueActionSettings,),
    ):
        return action.scale
    return 1.0


def _project_authored_scale(
    value: str | float | tuple[float, ...],
) -> AuthoredScale:
    if value == "auto":
        return AutoScale()
    if isinstance(value, tuple):
        return VectorScale(value)
    return ScalarScale(float(value))


def _resolve_action_scale(action: ActionConfig) -> AuthoredScale | None:
    scale = _authored_scale(action)
    mass_pose = isinstance(action, MassActionSettings) and isinstance(
        action.policy_output,
        MassPosePolicyOutputSettings,
    )
    if mass_pose:
        return None
    return _project_authored_scale(scale)


def _resolve_mass_muscle_nn(
    action: ActionConfig,
    muscle: MuscleSettings,
) -> MuscleNNSettings | None:
    if not isinstance(action, MassActionSettings):
        return None
    updates = {}
    if action.muscle_nn.output_activation is None:
        updates["output_activation"] = (
            "tanh_relu"
            if isinstance(action.policy_output, MassPosePolicyOutputSettings)
            else "sine"
        )
    if action.muscle_nn.target_includes_passive_force is None:
        updates["target_includes_passive_force"] = False
    return action.muscle_nn.model_copy(update=updates)


def _validate_cross_capability_contracts(
    *,
    action: ActionConfig,
    muscle: MuscleSettings,
) -> None:
    force_velocity = muscle.force_velocity
    lbs = muscle.lbs
    if (
        lbs.body_wrench.forward_mode == "native_tile"
        and lbs.attachment_layout != "body_wrench"
    ):
        raise ValueError(
            "muscle.lbs.body_wrench.forward_mode='native_tile' requires "
            "muscle.lbs.attachment_layout='body_wrench'; use pairchain for other layouts"
        )

    if force_velocity.enabled and action.type not in FORCE_VELOCITY_ACTION_MODES:
        raise NotImplementedError(
            "muscle.force_velocity.enabled currently supports only action.type='mass'"
        )


@dataclass(frozen=True, slots=True)
class FastMuscleControlSettings:
    action: ActionConfig
    muscle: MuscleSettings
    observations: FastMuscleObservationSettings
    visualization: ActivationVisualizationSettings
    character_render_mode: str
    effective: FastMuscleEffectiveSemantics

    @classmethod
    def from_validated(
        cls,
        *,
        action: ActionConfig,
        muscle: MuscleSettings,
        observations: FastMuscleObservationSettings,
        visualization: ActivationVisualizationSettings,
        character_render_mode: str,
    ) -> FastMuscleControlSettings:
        effective = FastMuscleEffectiveSemantics(
            bounds=_resolve_bounds(action),
            action_scale=_resolve_action_scale(action),
            length_range=resolve_muscle_length_range(
                muscle.length_range, muscle.dynamics.model
            ),
            mass_muscle_nn=_resolve_mass_muscle_nn(action, muscle),
        )
        _validate_cross_capability_contracts(
            action=action,
            muscle=muscle,
        )
        return cls(
            action=action,
            muscle=muscle,
            observations=observations,
            visualization=visualization,
            character_render_mode=str(character_render_mode),
            effective=effective,
        )

    @classmethod
    def from_cfg(cls, cfg: Any) -> FastMuscleControlSettings:
        return cls.from_validated(
            action=ActionSettings.from_env_cfg(cfg),
            muscle=MuscleSettings.from_env_cfg(cfg),
            observations=FastMuscleObservationSettings.from_env_cfg(cfg),
            visualization=ActivationVisualizationSettings.from_env_cfg(cfg),
            character_render_mode=resolve_character_render_mode(cfg),
        )


__all__ = (
    "AuthoredScale",
    "AutoScale",
    "ExplicitScale",
    "FastMuscleControlSettings",
    "FastMuscleEffectiveSemantics",
    "ScalarScale",
    "VectorScale",
    "activation_interval_bounds",
)
