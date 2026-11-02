"""Diagonal-Gaussian density, entropy and the optional tanh correction.

These stateless tensor operations are shared by policy collection and learning,
and can be inlined into compiled policy evaluation.
"""

from __future__ import annotations

import math

import torch

LOG_2PI = math.log(2.0 * math.pi)
LOG_2PI_HALF = 0.5 * LOG_2PI
NORMAL_ENTROPY_CONST = LOG_2PI_HALF + 0.5
"""Per-dimension differential entropy of a unit Gaussian, less `log_std`."""


def gaussian_tanh_log_prob(
    actions: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    std: torch.Tensor,
    *,
    squash: bool,
    epsilon: float,
    pre_squash_actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log-probability of `actions` under `Normal(mean, std)`, summed over the
    last dimension and keeping it.

    With `squash` and `pre_squash_actions`, use the original Gaussian sample
    and a stable log Jacobian, even when float32 tanh saturates. Otherwise
    `actions` are tanh-squashed: they are pulled inside
    `(-1, 1)` by `epsilon`, inverted back to pre-tanh space, and the
    log-determinant of the tanh Jacobian is subtracted. Without it the actions
    are already in pre-tanh space and nothing is subtracted.
    """

    if squash and pre_squash_actions is not None:
        noise = (pre_squash_actions - mean) / std
        log_det_jacobian = tanh_log_abs_det_jacobian(pre_squash_actions)
    elif squash:
        actions = actions.clamp(min=-(1.0 - epsilon), max=(1.0 - epsilon))
        # atanh, written so the two halves stay accurate near +/-1.
        pre_tanh = 0.5 * (torch.log1p(actions) - torch.log1p(-actions))
        noise = (pre_tanh - mean) / std
        log_det_jacobian = torch.log(torch.clamp(1.0 - actions.square(), min=epsilon))
    else:
        noise = (actions - mean) / std
        log_det_jacobian = None
    log_prob = torch.sum(
        -0.5 * noise.square() - log_std - LOG_2PI_HALF,
        dim=-1,
        keepdim=True,
    )
    if log_det_jacobian is not None:
        log_prob.sub_(torch.sum(log_det_jacobian, dim=-1, keepdim=True))
    return log_prob


def tanh_log_abs_det_jacobian(raw_actions: torch.Tensor) -> torch.Tensor:
    """Stable log(1 - tanh(x)**2), including saturated float32 actions."""

    return 2.0 * (
        math.log(2.0) - raw_actions - torch.nn.functional.softplus(-2.0 * raw_actions)
    )


def diagonal_gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Entropy of `Normal(_, exp(log_std))`, summed over the last dimension."""

    return torch.sum(log_std + NORMAL_ENTROPY_CONST, dim=-1, keepdim=True)
