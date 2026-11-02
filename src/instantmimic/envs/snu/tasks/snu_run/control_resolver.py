from __future__ import annotations

from typing import Any

from instantmimic.envs.controls.fast_muscle.config_models import (
    ActionSettings,
    ActivationVisualizationSettings,
    FastMuscleObservationSettings,
    MuscleSettings,
)
from instantmimic.envs.controls.fast_muscle.control import (
    FastMuscleBootstrapSettings,
    FastMuscleControlCfg,
)
from instantmimic.envs.controls.fast_muscle.settings import FastMuscleControlSettings
from instantmimic.envs.snu.tasks.snu_imitation.ctrl_control import SNUCtrlControlCfg
from instantmimic.rendering.character import resolve_character_render_mode


def make_default_control_cfg(cfg: Any) -> SNUCtrlControlCfg | FastMuscleControlCfg:
    action = ActionSettings.from_env_cfg(cfg)
    action_mode = action.type
    if action_mode == "ctrl":
        return SNUCtrlControlCfg(
            action_dim=action.space,
            action_scale=action.scale,
            use_ref_action=False,
            action_mode="pd",
            clamp_actions=False,
        )
    if action_mode == "torque":
        muscle = MuscleSettings.from_env_cfg(cfg)
        settings = FastMuscleControlSettings.from_validated(
            action=action,
            muscle=muscle,
            observations=FastMuscleObservationSettings.from_env_cfg(cfg),
            visualization=ActivationVisualizationSettings.from_env_cfg(cfg),
            character_render_mode=resolve_character_render_mode(cfg),
        )
        return FastMuscleControlCfg(
            settings=settings,
            bootstrap=FastMuscleBootstrapSettings.from_settings(settings),
        )
    raise ValueError(
        f"Unsupported SNU Run env.action.type={action_mode!r}; expected one of "
        "'ctrl' or 'torque'."
    )
