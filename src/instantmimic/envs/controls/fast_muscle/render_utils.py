from __future__ import annotations

from typing import Any

from instantmimic.rendering.character import (
    apply_character_render_mode_to_model,
    normalize_character_render_mode,
    resolve_character_render_mode,
)


def _resolve_env_character_render_mode(env: Any) -> str:
    if hasattr(env, "_character_render_mode"):
        return normalize_character_render_mode(env._character_render_mode)
    cfg = getattr(env, "cfg", None)
    if cfg is not None:
        return resolve_character_render_mode(cfg)
    controller = getattr(env, "fast_muscle", None)
    if controller is not None:
        return normalize_character_render_mode(controller.character_render_mode)
    return "mesh"


def apply_character_geom_render_visibility(env: Any) -> bool:
    render_model = getattr(env, "render_model", None)
    if render_model is None:
        render_model = getattr(env, "model", None)
    if render_model is None:
        return False
    return apply_character_render_mode_to_model(
        render_model,
        _resolve_env_character_render_mode(env),
        env_raw=env,
    )
