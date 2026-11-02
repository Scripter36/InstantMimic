"""Pure PPO loss terms shared by eager and compiled update paths."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ppo_loss_terms(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    predicted_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    ratio_clip: float,
    value_clip: float,
    clip_predicted_values: bool,
    value_loss_scale: float,
    entropy_loss_scale: float,
    value_loss_clip_mode: str = "clamp",
    zero_scalar: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return policy, value, entropy losses and PPO clip fraction.

    Value clipping preserves MuJoCo Warp's historical clamp-then-MSE semantics.
    """

    log_ratio = new_log_probs - old_log_probs
    # The unbounded ratio is only a nondifferentiable reporting statistic.
    ratio = log_ratio.detach().exp()
    clip_fraction = ((ratio - 1.0).abs() > ratio_clip).float().mean()
    # Clip the selected objective before exponentiating: an overflowing ratio
    # in the constant positive-advantage branch otherwise gives inf * 0 in
    # backward, even when the clipped loss itself is finite.
    log_upper = math.log1p(ratio_clip)
    log_lower = math.log1p(-ratio_clip) if ratio_clip < 1.0 else -math.inf
    objective_log_ratio = torch.where(
        advantages >= 0.0,
        log_ratio.clamp(max=log_upper),
        log_ratio.clamp(min=log_lower),
    )
    policy_loss = -(advantages * objective_log_ratio.exp()).mean()

    if clip_predicted_values:
        clipped_values = old_values + (predicted_values - old_values).clamp(
            -value_clip,
            value_clip,
        )
        clipped_error = F.mse_loss(returns, clipped_values, reduction="none")
        if value_loss_clip_mode == "max":
            unclipped_error = F.mse_loss(
                returns,
                predicted_values,
                reduction="none",
            )
            clipped_error = torch.maximum(unclipped_error, clipped_error)
        value_loss = value_loss_scale * clipped_error.mean()
    else:
        value_loss = value_loss_scale * F.mse_loss(returns, predicted_values)

    if entropy_loss_scale:
        entropy_loss = -entropy_loss_scale * entropy.mean()
    elif zero_scalar is not None:
        entropy_loss = zero_scalar
    else:
        entropy_loss = entropy.new_zeros(())

    return policy_loss, value_loss, entropy_loss, clip_fraction
