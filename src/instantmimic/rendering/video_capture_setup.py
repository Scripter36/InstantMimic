from __future__ import annotations

import logging
from typing import Any

_LOGGER = logging.getLogger(__name__)


def enable_substep_render_capture_for_video(
    env_raw: Any,
    *,
    cfg_env: Any | None = None,
) -> bool:
    """Enable substep render capture when the env supports it.

    This keeps debug/video tools aligned with playback, so recorded video fps
    follows the env's effective render cadence when ``render_interval`` is
    smaller than ``decimation``.
    """

    set_collect_render_substeps = getattr(env_raw, "set_collect_render_substeps", None)
    if not callable(set_collect_render_substeps):
        return False

    enabled = bool(set_collect_render_substeps(True))
    if enabled:
        render_interval = getattr(getattr(cfg_env, "sim", None), "render_interval", "?")
        decimation = getattr(cfg_env, "decimation", "?")
        _LOGGER.info(
            "Enabled substep render capture for playback/video "
            "(render_interval=%s, decimation=%s).",
            render_interval,
            decimation,
        )
    return enabled
