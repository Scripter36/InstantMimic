from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MotionClipMetadata:
    """Device-resident timeline metadata for one or more motion clips."""

    clip_start: torch.Tensor
    clip_length: torch.Tensor
    clip_reset_start_count: torch.Tensor
    frame_to_clip_idx: torch.Tensor
    clip_unit_start: torch.Tensor
    clip_unit_count: torch.Tensor
    total_frame_count: int
    total_unit_count: int


def build_clip_offsets(
    *,
    clip_frame_counts: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build only concatenated clip offsets and lengths on ``device``."""
    if len(clip_frame_counts) <= 0:
        raise ValueError("clip_frame_counts must contain at least one motion clip.")

    normalized_counts = [max(1, int(count)) for count in clip_frame_counts]
    clip_start_values: list[int] = []
    total_frames = 0
    for frame_count in normalized_counts:
        clip_start_values.append(total_frames)
        total_frames += frame_count

    clip_start = torch.tensor(
        clip_start_values,
        device=device,
        dtype=torch.int32,
    ).contiguous()
    clip_length = torch.tensor(
        normalized_counts,
        device=device,
        dtype=torch.int32,
    ).contiguous()
    return clip_start, clip_length


def build_motion_clip_metadata(
    *,
    clip_frame_counts: list[int],
    chunk_size: int,
    device: torch.device,
    reset_start_horizon_steps: int = 1,
) -> MotionClipMetadata:
    """Build SNU-compatible clip metadata directly on ``device``."""
    if len(clip_frame_counts) <= 0:
        raise ValueError("clip_frame_counts must contain at least one motion clip.")

    normalized_counts = [max(1, int(count)) for count in clip_frame_counts]
    chunk_size = max(1, int(chunk_size))
    reset_horizon = max(1, int(reset_start_horizon_steps))
    clip_start, clip_length = build_clip_offsets(
        clip_frame_counts=normalized_counts,
        device=device,
    )
    clip_reset_start_count_list: list[int] = []
    clip_unit_start_list: list[int] = []
    clip_unit_count_list: list[int] = []

    total_frames = sum(normalized_counts)
    total_units = 0
    for frame_count in normalized_counts:
        reset_start_count = max(1, frame_count - reset_horizon + 1)
        reset_start_count = min(frame_count, reset_start_count)
        clip_reset_start_count_list.append(reset_start_count)
        unit_count = max(1, (reset_start_count + chunk_size - 1) // chunk_size)
        clip_unit_start_list.append(total_units)
        clip_unit_count_list.append(unit_count)
        total_units += unit_count

    frame_to_clip_idx = torch.empty(
        total_frames,
        device=device,
        dtype=torch.int32,
    )
    frame_start = 0
    for clip_idx, frame_count in enumerate(normalized_counts):
        frame_to_clip_idx[frame_start : frame_start + frame_count] = int(clip_idx)
        frame_start += frame_count

    return MotionClipMetadata(
        clip_start=clip_start,
        clip_length=clip_length,
        clip_reset_start_count=torch.tensor(
            clip_reset_start_count_list,
            device=device,
            dtype=torch.int32,
        ),
        frame_to_clip_idx=frame_to_clip_idx.contiguous(),
        clip_unit_start=torch.tensor(
            clip_unit_start_list,
            device=device,
            dtype=torch.int32,
        ),
        clip_unit_count=torch.tensor(
            clip_unit_count_list,
            device=device,
            dtype=torch.int32,
        ),
        total_frame_count=int(total_frames),
        total_unit_count=int(total_units),
    )


def resolve_clip_reference_frames(
    *,
    ref_frame_idx: torch.Tensor,
    motion_clip_idx: torch.Tensor,
    clip_start: torch.Tensor,
    clip_length: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve SNU multi-clip reference frames into global/local/cycle indices."""
    ref_idx_long = ref_frame_idx.to(torch.long)
    clip_idx_long = motion_clip_idx.to(torch.long)
    clip_length_long = clip_length.to(torch.long).index_select(0, clip_idx_long)
    clip_start_long = clip_start.to(torch.long).index_select(0, clip_idx_long)
    local_idx = torch.remainder(ref_idx_long, clip_length_long)
    cycle_idx = torch.div(ref_idx_long, clip_length_long, rounding_mode="floor")
    global_idx = clip_start_long + local_idx
    return global_idx, local_idx, cycle_idx
