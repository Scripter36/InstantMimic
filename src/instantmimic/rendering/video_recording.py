from __future__ import annotations

from typing import Any

import numpy as np


def resolve_render_output_fps(
    env: Any,
    *,
    env_raw: Any,
    step_dt: float,
) -> float:
    get_render_output_fps = getattr(env_raw, "get_render_output_fps", None)
    if callable(get_render_output_fps):
        fps_raw = get_render_output_fps()
    else:
        metadata = getattr(env, "metadata", {})
        fps_raw = metadata.get("render_fps", 1.0 / max(step_dt, 1.0e-6))
    try:
        fps = float(fps_raw)
    except (TypeError, ValueError):
        fps = 30.0
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    return fps


def normalize_render_frames(render_output: Any) -> list[np.ndarray]:
    if render_output is None:
        return []
    if isinstance(render_output, list):
        return [frame for frame in render_output if isinstance(frame, np.ndarray)]
    if isinstance(render_output, np.ndarray):
        return [render_output]
    return []
