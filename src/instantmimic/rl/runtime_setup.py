from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter_ns
from typing import Any

import torch

from instantmimic.rl.optimizers import CapturableMuon, HybridMuonAdamW


def runtime_setup_time_ns(owner: Any) -> int:
    return int(getattr(owner, "_runtime_setup_time_ns", 0))


@contextmanager
def measure_runtime_setup(owner: Any, device: torch.device) -> Iterator[None]:
    """Attribute one-time runtime preparation without hiding queued training work."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_ns = perf_counter_ns()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        owner._runtime_setup_time_ns = runtime_setup_time_ns(owner) + (
            perf_counter_ns() - start_ns
        )


def initialize_optimizer_state(
    optimizer: torch.optim.Optimizer | HybridMuonAdamW,
) -> None:
    """Materialize optimizer state without performing an optimizer step.

    CUDA graph capture requires every state tensor to exist, and every compiled
    Muon Newton-Schulz kernel to be built, before the first captured step.
    """

    if isinstance(optimizer, HybridMuonAdamW):
        if optimizer.muon_optimizer is not None:
            initialize_optimizer_state(optimizer.muon_optimizer)
        if optimizer.adamw_optimizer is not None:
            initialize_optimizer_state(optimizer.adamw_optimizer)
        return
    if isinstance(optimizer, CapturableMuon):
        _initialize_muon_state(optimizer)
        return
    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        _initialize_adam_state(optimizer)
        return
    raise TypeError(f"cannot initialize state for optimizer {type(optimizer).__name__}")


def _initialize_muon_state(optimizer: CapturableMuon) -> None:
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not parameter.requires_grad:
                continue
            state = optimizer.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
    optimizer.warmup_ns_kernels()


def _initialize_adam_state(
    optimizer: torch.optim.Adam | torch.optim.AdamW,
) -> None:
    for group in optimizer.param_groups:
        capturable_or_fused = bool(
            group.get("capturable", False) or group.get("fused", False)
        )
        for parameter in group["params"]:
            if not parameter.requires_grad:
                continue
            state = optimizer.state[parameter]
            if state:
                continue
            state["step"] = torch.zeros(
                (),
                dtype=torch.float32,
                device=parameter.device if capturable_or_fused else "cpu",
            )
            state["exp_avg"] = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
            state["exp_avg_sq"] = torch.zeros_like(
                parameter,
                memory_format=torch.preserve_format,
            )
            if bool(group.get("amsgrad", False)):
                state["max_exp_avg_sq"] = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
