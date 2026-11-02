"""Runtime interfaces shared by reinforcement-learning algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable

import torch


class CheckpointRestoreMetadata(TypedDict):
    """Metadata returned when a training checkpoint restores optional state."""

    optimizer_state_present: bool
    optimizer_state_restored: bool


@runtime_checkable
class TrainingAgent(Protocol):
    """Required operations used by the algorithm-independent training loop."""

    buffer: Any
    discount_factor: float

    def act_into(self, step: int) -> None: ...

    def update(self) -> dict[str, float]: ...

    def load(self, path: str | Path) -> CheckpointRestoreMetadata | None: ...

    def prewarm_update_graph(self) -> bool: ...


class CheckpointAgent(Protocol):
    """Agent state consumed by :class:`CheckpointManager`."""

    def state_dict(self) -> dict[str, object]: ...

    def best_state_dict(self) -> dict[str, object]: ...


class PlaybackPolicyAgent(Protocol):
    """Required deterministic policy operations used during playback."""

    device: torch.device

    def policy_mean_into(
        self,
        obs: torch.Tensor,
        out_actions: torch.Tensor,
    ) -> None: ...

    def policy_execution_mean_into(
        self,
        obs: torch.Tensor,
        out_actions: torch.Tensor,
    ) -> None: ...


@runtime_checkable
class ActionScaleAgent(Protocol):
    def set_action_scales(
        self,
        *,
        mean_scale: torch.Tensor | None = None,
        std_scale: torch.Tensor | None = None,
    ) -> None: ...


@runtime_checkable
class ObservationNormalizationAgent(Protocol):
    def update_observation_normalization(self, obs: torch.Tensor) -> bool: ...


@runtime_checkable
class CudaGraphCapturePreparationAgent(Protocol):
    def prepare_cuda_libraries_for_graph_capture(self) -> None: ...


__all__ = [
    "ActionScaleAgent",
    "CheckpointRestoreMetadata",
    "CheckpointAgent",
    "CudaGraphCapturePreparationAgent",
    "ObservationNormalizationAgent",
    "PlaybackPolicyAgent",
    "TrainingAgent",
]
