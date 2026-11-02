from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SubstepRenderSchedule:
    """Render samples within one policy step and their interpolation fractions."""

    frame_indices: tuple[int, ...]
    interpolation_alphas: tuple[float, ...]


def resolve_substep_render_schedule(
    *,
    decimation: int,
    render_interval: int,
    phase_before_step: int,
) -> SubstepRenderSchedule:
    """Return the canonical render samples for one decimated policy step."""
    decimation = max(1, int(decimation))
    render_interval = max(1, int(render_interval))
    phase_before_step = int(phase_before_step) % render_interval
    frame_indices = tuple(
        local_idx
        for local_idx in range(decimation)
        if ((phase_before_step + local_idx + 1) % render_interval) == 0
    )
    return SubstepRenderSchedule(
        frame_indices=frame_indices,
        interpolation_alphas=tuple(
            float(substep_idx + 1) / float(decimation) for substep_idx in frame_indices
        ),
    )
