"""Device-side FP16 loss scaling for whole-update CUDA graph capture."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from instantmimic.rl.optimizers import (
    HybridMuonAdamW,
    optimizer_step_if_finite,
    prepare_optimizer_state,
)


class CapturableLossScaler(torch.nn.Module):
    """Scale gradients and skip optimizer writes without reading device scalars.

    The native fused Adam/AdamW kernel consumes the overflow flag; Muon masks
    parameter and momentum writes. No parameter/state backup is taken, and
    ordinary BF16/FP32 optimizer stepping remains unchanged.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        initial_scale: float = 128.0,
        growth_interval: int = 2000,
    ) -> None:
        super().__init__()
        if not math.isfinite(initial_scale) or not 0 < initial_scale <= 2**24:
            raise ValueError("loss_scale must be finite and in (0, 2**24]")
        if growth_interval < 1:
            raise ValueError("growth_interval must be positive")
        self.initial_scale = float(initial_scale)
        self.growth_interval = int(growth_interval)
        self.register_buffer(
            "scale", torch.tensor(initial_scale, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "growth_tracker", torch.zeros((), dtype=torch.int64, device=device)
        )
        self.register_buffer(
            "found_inf", torch.zeros((), dtype=torch.bool, device=device)
        )
        self.register_buffer(
            "skipped_steps", torch.zeros((), dtype=torch.int64, device=device)
        )

    @torch.no_grad()
    def prepare_optimizer(
        self, optimizer: torch.optim.Optimizer | HybridMuonAdamW
    ) -> None:
        """Materialize state before compilation/capture, preserving loaded state."""
        prepare_optimizer_state(optimizer, capturable=self.scale.device.type == "cuda")

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    @torch.no_grad()
    def unscale_(self, optimizer: torch.optim.Optimizer | HybridMuonAdamW) -> None:
        self.found_inf.zero_()
        inverse_scale = self.scale.reciprocal()
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    self.found_inf.logical_or_(~torch.isfinite(param.grad).all())
                    param.grad.mul_(inverse_scale)
                    self.found_inf.logical_or_(~torch.isfinite(param.grad).all())
        # Sanitize the entire transaction before clipping and candidate stepping.
        # Zeroing just a bad entry, or multiplying NaN by zero, is insufficient.
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.copy_(torch.where(self.found_inf, 0.0, param.grad))

    @torch.no_grad()
    def step(self, optimizer: torch.optim.Optimizer | HybridMuonAdamW) -> None:
        optimizer_step_if_finite(optimizer, self.found_inf.float())
        self.skipped_steps.add_(self.found_inf)
        next_tracker = self.growth_tracker + 1
        grow = next_tracker >= self.growth_interval
        next_scale = torch.where(
            self.found_inf,
            self.scale * 0.5,
            torch.where(grow, self.scale * 2.0, self.scale),
        )
        self.scale.copy_(next_scale.clamp(torch.finfo(torch.float32).tiny, 2**24))
        self.growth_tracker.copy_(torch.where(self.found_inf | grow, 0, next_tracker))

    @torch.no_grad()
    def reset(self) -> None:
        self.scale.fill_(self.initial_scale)
        self.growth_tracker.zero_()
        self.found_inf.zero_()
        self.skipped_steps.zero_()

    def load_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict=True, assign=False
    ):
        # Loading happens outside capture. Validate before mutating live tensors.
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("Invalid loss_scaler checkpoint fields")
        for key, value in state_dict.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype
            ):
                raise ValueError(f"Invalid loss_scaler.{key} checkpoint tensor")
        scale = float(state_dict["scale"].item())
        tracker = int(state_dict["growth_tracker"].item())
        if (
            not math.isfinite(scale)
            or not 0 < scale <= 2**24
            or not 0 <= tracker < self.growth_interval
            or state_dict["skipped_steps"].item() < 0
        ):
            raise ValueError("Invalid loss_scaler checkpoint values")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
