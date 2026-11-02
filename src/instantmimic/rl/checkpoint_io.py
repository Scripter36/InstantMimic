"""CPU readers for trusted training artifacts, including legacy checkpoints."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from zipfile import is_zipfile

import torch

REPLAY_SIDECAR_KEY = "replay_sidecar"


def load_checkpoint_payload(path: str | Path) -> Any:
    """Read a trusted checkpoint without eagerly copying unused tensor storage.

    Full off-policy checkpoints can contain replay buffers much larger than the
    policy. Memory mapping keeps their storage on disk until a consumer actually
    reads a tensor, so metadata and environment-state readers need not load the
    replay. PyTorch's older non-ZIP serialization cannot be memory mapped.

    Keep pickle compatibility explicit: historical environment and optimizer
    state may include objects outside the weights-only unpickler's allowlist.
    As with the artifact metadata loader, only use this on trusted checkpoints.
    """
    checkpoint = Path(path)
    return torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=is_zipfile(checkpoint),
    )


@contextmanager
def load_replay_checkpoint_payload(
    checkpoint_path: str | Path,
    payload: dict[str, Any],
) -> Iterator[dict[str, Any] | None]:
    """Resolve legacy inline replay or a trusted sibling replay sidecar."""

    inline = payload.get("replay")
    sidecar_filename = payload.get(REPLAY_SIDECAR_KEY)
    if inline is not None and sidecar_filename is not None:
        raise ValueError(
            "Checkpoint cannot contain both inline replay and a replay sidecar."
        )
    if inline is not None:
        if not isinstance(inline, dict):
            raise TypeError("Checkpoint replay payload must be a dictionary.")
        yield inline
        return
    if sidecar_filename is None:
        yield None
        return
    if (
        not isinstance(sidecar_filename, str)
        or not sidecar_filename
        or Path(sidecar_filename).is_absolute()
        or Path(sidecar_filename).name != sidecar_filename
    ):
        raise ValueError("Checkpoint replay sidecar must be a sibling filename.")
    sidecar_path = Path(checkpoint_path).parent / sidecar_filename
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Replay sidecar referenced by checkpoint is missing: {sidecar_path}"
        )
    sidecar = load_checkpoint_payload(sidecar_path)
    if not isinstance(sidecar, dict):
        raise TypeError("Replay sidecar payload must be a dictionary.")
    yield sidecar
