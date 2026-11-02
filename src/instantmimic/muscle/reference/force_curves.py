"""Normalized Hill force-length curves, written to be read.

`muscle_qfrc_projector.py` holds the Warp statement of the same two curves;
these are what the CPU layer path evaluates and what a reader checks the
kernels against.
"""

from __future__ import annotations

import math

import torch


def passive_force_length_torch(
    normalized_length: torch.Tensor,
    *,
    force_type: str,
    k_pe: float,
    e_mo: float,
    lmax: float,
    fpmax: float,
) -> torch.Tensor:
    """Evaluate a normalized passive muscle force-length curve."""
    if force_type == "exponential":
        x = (float(k_pe) * (normalized_length - 1.0) / float(e_mo)).clamp(
            min=-80.0,
            max=80.0,
        )
        passive = torch.expm1(x) / math.expm1(float(k_pe))
    elif force_type == "mujoco":
        midpoint = 0.5 * (1.0 + float(lmax))
        width = midpoint - 1.0
        quadratic = 0.5 * float(fpmax) * ((normalized_length - 1.0) / width).square()
        linear = float(fpmax) * (0.5 + (normalized_length - midpoint) / width)
        passive = torch.where(normalized_length <= midpoint, quadratic, linear)
    else:
        raise ValueError(
            f"Unsupported passive force type {force_type!r}; "
            "expected 'exponential' or 'mujoco'."
        )
    return torch.where(
        normalized_length > 1.0,
        passive,
        torch.zeros_like(passive),
    )


def active_force_length_torch(
    normalized_length: torch.Tensor,
    *,
    force_type: str,
    gamma: float,
    lmin: float,
    lmax: float,
) -> torch.Tensor:
    """Evaluate the configured normalized active muscle force-length curve."""
    if force_type == "gaussian":
        return torch.exp(-((normalized_length - 1.0).square()) / float(gamma))
    if force_type != "mujoco":
        raise ValueError(
            f"Unsupported active force type {force_type!r}; "
            "expected 'gaussian' or 'mujoco'."
        )

    lower_midpoint = 0.5 * (float(lmin) + 1.0)
    upper_midpoint = 0.5 * (1.0 + float(lmax))
    lower_rise = (
        0.5
        * ((normalized_length - float(lmin)) / (lower_midpoint - float(lmin))).square()
    )
    lower_fall = (
        1.0 - 0.5 * ((1.0 - normalized_length) / (1.0 - lower_midpoint)).square()
    )
    upper_fall = (
        1.0 - 0.5 * ((normalized_length - 1.0) / (upper_midpoint - 1.0)).square()
    )
    upper_tail = (
        0.5
        * ((float(lmax) - normalized_length) / (float(lmax) - upper_midpoint)).square()
    )
    active = torch.where(
        normalized_length <= lower_midpoint,
        lower_rise,
        torch.where(
            normalized_length <= 1.0,
            lower_fall,
            torch.where(
                normalized_length <= upper_midpoint,
                upper_fall,
                upper_tail,
            ),
        ),
    )
    in_range = (normalized_length >= float(lmin)) & (normalized_length <= float(lmax))
    return torch.where(in_range, active, torch.zeros_like(active))
