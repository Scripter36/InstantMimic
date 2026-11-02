from __future__ import annotations

import torch

from instantmimic.motion import (
    build_motion_clip_metadata,
    resolve_clip_reference_frames,
)


def test_motion_timeline_matches_snu_frozen_metadata_contract() -> None:
    metadata = build_motion_clip_metadata(
        clip_frame_counts=[5, 3, 4],
        chunk_size=2,
        reset_start_horizon_steps=1,
        device=torch.device("cpu"),
    )

    assert metadata.total_frame_count == 12
    assert metadata.total_unit_count == 7
    assert metadata.clip_start.dtype == torch.int32
    assert metadata.clip_start.is_contiguous()
    assert torch.equal(metadata.clip_start, torch.tensor([0, 5, 8], dtype=torch.int32))
    assert torch.equal(metadata.clip_length, torch.tensor([5, 3, 4], dtype=torch.int32))
    assert torch.equal(
        metadata.clip_reset_start_count,
        torch.tensor([5, 3, 4], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.frame_to_clip_idx,
        torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.clip_unit_start,
        torch.tensor([0, 3, 5], dtype=torch.int32),
    )
    assert torch.equal(
        metadata.clip_unit_count,
        torch.tensor([3, 2, 2], dtype=torch.int32),
    )


def test_motion_frame_resolution_matches_snu_frozen_multi_clip_contract() -> None:
    global_idx, local_idx, cycle_idx = resolve_clip_reference_frames(
        ref_frame_idx=torch.tensor([0, 5, 2, 7], dtype=torch.int32),
        motion_clip_idx=torch.tensor([0, 0, 1, 2], dtype=torch.int32),
        clip_start=torch.tensor([0, 5, 8], dtype=torch.int32),
        clip_length=torch.tensor([5, 3, 4], dtype=torch.int32),
    )

    assert torch.equal(global_idx, torch.tensor([0, 0, 7, 11], dtype=torch.long))
    assert torch.equal(local_idx, torch.tensor([0, 0, 2, 3], dtype=torch.long))
    assert torch.equal(cycle_idx, torch.tensor([0, 1, 0, 1], dtype=torch.long))
