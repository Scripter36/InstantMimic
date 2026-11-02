from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch


class LatentLowLevelAdapter(Protocol):
    latent_dim: int

    def prior_mean(self, obs: torch.Tensor) -> torch.Tensor: ...

    def policy_mean(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor: ...


@dataclass
class LatentLowLevelState:
    """Persistent policy buffers shared by latent command environments."""

    action_scale: float
    observation_spec: dict[str, object]
    deployable_observation_indices: torch.Tensor
    append_prior_to_policy_observation: bool
    residual_scale: float
    observation_buffer: torch.Tensor
    environment_action_buffer: torch.Tensor
    adapter: LatentLowLevelAdapter
    latent_action_buffer: torch.Tensor
    prior_latent_buffer: torch.Tensor
    total_latent_buffer: torch.Tensor
    command_dim: int
    policy_observation_dim: int
    command_observation_buffer: torch.Tensor
    previous_latent_action_buffer: torch.Tensor | None = None
    observation_mapping: dict[str, torch.Tensor] = field(default_factory=dict)
