from __future__ import annotations

from typing import TYPE_CHECKING, Any

from instantmimic.envs.controls.fast_muscle.config_models import ActionSettings
from instantmimic.envs.snu.tasks.snu_imitation.ctrl_control import SNUCtrlControlCfg

if TYPE_CHECKING:
    from instantmimic.envs.controls.fast_muscle.control import FastMuscleControlCfg

_FAST_MUSCLE_ACTION_MODES = frozenset({"activation", "torque", "mass"})


def uses_fast_muscle_control(cfg: Any) -> bool:
    return ActionSettings.from_env_cfg(cfg).type in _FAST_MUSCLE_ACTION_MODES


def make_default_control_cfg(
    cfg: Any,
) -> SNUCtrlControlCfg | FastMuscleControlCfg:
    action = ActionSettings.from_env_cfg(cfg)
    if action.type in _FAST_MUSCLE_ACTION_MODES:
        from instantmimic.envs.controls.fast_muscle.config_models import (
            ActivationVisualizationSettings,
            FastMuscleObservationSettings,
            MuscleSettings,
        )
        from instantmimic.envs.controls.fast_muscle.control import (
            FastMuscleBootstrapSettings,
            FastMuscleControlCfg,
        )
        from instantmimic.envs.controls.fast_muscle.settings import (
            FastMuscleControlSettings,
        )
        from instantmimic.rendering.character import resolve_character_render_mode

        from .render_adapter import SNUFastMuscleRenderAdapter

        settings = FastMuscleControlSettings.from_validated(
            action=action,
            muscle=MuscleSettings.from_env_cfg(cfg),
            observations=FastMuscleObservationSettings.from_env_cfg(cfg),
            visualization=ActivationVisualizationSettings.from_env_cfg(cfg),
            character_render_mode=resolve_character_render_mode(cfg),
        )
        return FastMuscleControlCfg(
            settings=settings,
            bootstrap=FastMuscleBootstrapSettings.from_settings(settings),
            render_adapter=SNUFastMuscleRenderAdapter(),
        )
    if action.type != "ctrl":
        raise ValueError(f"Unsupported env.action.type={action.type!r}.")
    return SNUCtrlControlCfg(
        action_dim=action.space,
        action_scale=action.scale,
        use_ref_action=action.use_reference,
    )
