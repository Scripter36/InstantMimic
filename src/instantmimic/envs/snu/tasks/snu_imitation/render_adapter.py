from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from instantmimic.envs.config_access import get_value
from instantmimic.envs.controls.fast_muscle.render_builder import (
    build_ghost_render_model,
)


def _resolve_fast_muscle_ghost_root_xpath(env: Any) -> str:
    bvh_map = getattr(getattr(env, "motion_settings", None), "bvh_map", {}) or {}
    hips_body = get_value(bvh_map, "Hips", None)
    if hips_body:
        return f"body[@name='{hips_body}']"
    return "body[@name='Pelvis']"


@dataclass(frozen=True, slots=True)
class SNUFastMuscleRenderAdapter:
    """SNU imitation ghost-render policy for the shared fast-muscle control."""

    def build_render_model(self, control: Any, env: Any):
        bootstrap = getattr(control, "bootstrap", None)
        if bootstrap is None:
            return None
        source_mjcf_path = str(bootstrap.source_mjcf_path)
        env._resolved_source_mjcf_path = source_mjcf_path
        result = build_ghost_render_model(
            source_mjcf_path=source_mjcf_path,
            sim_model=env.model,
            resolution=tuple(env.cfg.viewer.resolution),
            ghost_root_xpath=_resolve_fast_muscle_ghost_root_xpath(env),
            ghost_root_description="fast-muscle ghost root body",
            ghost_offset=env.cfg.ghost_offset,
            ghost_offset_dtype=np.float64,
        )
        if result.ghost_binding is None:
            raise ValueError(
                "Expected ghost render binding for imitation render model."
            )
        env._ghost_mocap_ids = result.ghost_binding.ghost_mocap_ids
        env._ghost_sim_body_ids = result.ghost_binding.ghost_sim_body_ids
        env._ghost_offset_np = result.ghost_binding.ghost_offset_np
        return result.render_model, result.render_data


__all__ = ["SNUFastMuscleRenderAdapter", "_resolve_fast_muscle_ghost_root_xpath"]
