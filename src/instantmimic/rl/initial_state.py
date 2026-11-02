"""Algorithm-independent contracts for learned simulator initial states."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class InitialStateSpec:
    """Shape and simulator-coordinate contract shared by an env and an agent."""

    num_components: int
    latent_dim: int
    coordinates: str
    residual_scale: tuple[float, ...]
    fingerprint: str
    sample_arity: int = 1
    component_layout: str = "independent"
    reference_pose_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.num_components < 1:
            raise ValueError("initial-state num_components must be positive")
        if self.latent_dim < 1:
            raise ValueError("initial-state latent_dim must be positive")
        if self.sample_arity < 1:
            raise ValueError("initial-state sample_arity must be positive")
        if self.component_layout not in {"independent", "adjacent_frames"}:
            raise ValueError(
                "initial-state component_layout must be one of "
                "{'independent', 'adjacent_frames'}"
            )
        if self.component_layout == "adjacent_frames" and self.sample_arity != 2:
            raise ValueError("adjacent_frames initial states require sample_arity=2")
        if len(self.residual_scale) != self.latent_dim:
            raise ValueError(
                "initial-state residual_scale length must equal latent_dim: "
                f"{len(self.residual_scale)} != {self.latent_dim}"
            )
        if any(
            not math.isfinite(scale) or scale <= 0.0 for scale in self.residual_scale
        ):
            raise ValueError(
                "initial-state residual scales must be finite and positive"
            )
        if not self.fingerprint:
            raise ValueError("initial-state fingerprint must not be empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["InitialStateSpec"]
