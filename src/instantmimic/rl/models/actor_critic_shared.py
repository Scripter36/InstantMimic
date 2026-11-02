"""Shared neural-network primitives for the simulator-neutral PPO package."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

_LOG_STD_BOUND_MODES = {"clamp", "tanh"}


def make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    if key == "mish":
        return nn.Mish()
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unsupported activation: {name}")


def _make_norm(
    hidden_dim: int,
    *,
    use_layer_norm: bool = False,
    use_rms_norm: bool = False,
) -> nn.Module:
    if bool(use_layer_norm) and bool(use_rms_norm):
        raise ValueError("use_layer_norm and use_rms_norm are mutually exclusive.")
    if use_layer_norm:
        return nn.LayerNorm(int(hidden_dim))
    if use_rms_norm:
        return nn.RMSNorm(int(hidden_dim))
    return nn.Identity()


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        activation: str,
        *,
        use_layer_norm: bool = False,
        use_rms_norm: bool = False,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        dim = int(hidden_dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = _make_norm(
            dim,
            use_layer_norm=use_layer_norm,
            use_rms_norm=use_rms_norm,
        )
        self.activation = make_activation(activation)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = _make_norm(
            dim,
            use_layer_norm=use_layer_norm,
            use_rms_norm=use_rms_norm,
        )
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return residual + out * self.residual_scale


def build_trunk(
    input_dim: int,
    layers: Sequence[int],
    activation: str,
    *,
    network_type: str = "mlp",
    use_layer_norm: bool = False,
    use_rms_norm: bool = False,
    residual_scale: float = 1.0,
) -> tuple[nn.Sequential, int]:
    hidden_layers = [int(layer) for layer in layers]
    if not hidden_layers:
        raise ValueError("`layers` must not be empty")

    key = str(network_type).strip().lower()
    if key not in {"mlp", "residual_mlp"}:
        raise ValueError(
            f"Unsupported network_type: {network_type}. Use 'mlp' or 'residual_mlp'."
        )
    if bool(use_layer_norm) and bool(use_rms_norm):
        raise ValueError("use_layer_norm and use_rms_norm are mutually exclusive.")

    modules: list[nn.Module] = []
    in_dim = int(input_dim)
    if key == "mlp":
        for hidden_dim in hidden_layers:
            modules.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm or use_rms_norm:
                modules.append(
                    _make_norm(
                        hidden_dim,
                        use_layer_norm=use_layer_norm,
                        use_rms_norm=use_rms_norm,
                    )
                )
            modules.append(make_activation(activation))
            in_dim = hidden_dim
        return nn.Sequential(*modules), in_dim

    for hidden_dim in hidden_layers:
        modules.append(nn.Linear(in_dim, hidden_dim))
        if use_layer_norm or use_rms_norm:
            modules.append(
                _make_norm(
                    hidden_dim,
                    use_layer_norm=use_layer_norm,
                    use_rms_norm=use_rms_norm,
                )
            )
        modules.append(make_activation(activation))
        modules.append(
            ResidualMLPBlock(
                hidden_dim,
                activation,
                use_layer_norm=use_layer_norm,
                use_rms_norm=use_rms_norm,
                residual_scale=residual_scale,
            )
        )
        in_dim = hidden_dim
    return nn.Sequential(*modules), in_dim


def set_scale_buffer(
    buffer: torch.Tensor,
    scale: torch.Tensor,
    *,
    label: str,
    strictly_positive: bool,
) -> torch.Tensor:
    if scale.shape != buffer.shape:
        raise ValueError(
            f"Unexpected {label} shape: expected {tuple(buffer.shape)}, got {tuple(scale.shape)}."
        )
    if not torch.isfinite(scale).all():
        raise ValueError(f"{label} must be finite.")
    if strictly_positive and torch.any(scale <= 0):
        raise ValueError(f"{label} must be strictly positive.")
    return scale.to(device=buffer.device, dtype=torch.float32)


def resolve_log_std_bound_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key not in _LOG_STD_BOUND_MODES:
        raise ValueError(
            f"Unsupported log_std_bound_mode: {mode!r}. Use 'clamp' or 'tanh'."
        )
    return key


def log_std_raw_from_bounded(
    initial_log_std: float,
    *,
    min_log_std: float,
    max_log_std: float,
    bound_mode: str,
) -> float:
    mode = resolve_log_std_bound_mode(bound_mode)
    initial = float(initial_log_std)
    minimum = float(min_log_std)
    maximum = float(max_log_std)
    if not all(math.isfinite(value) for value in (initial, minimum, maximum)):
        raise ValueError("log std bounds and initial_log_std must be finite.")
    if maximum < minimum:
        raise ValueError("max_log_std must be greater than or equal to min_log_std.")
    if mode == "clamp":
        return initial
    if maximum <= minimum:
        raise ValueError(
            "log_std_bound_mode='tanh' requires max_log_std > min_log_std."
        )
    if not minimum < initial < maximum:
        raise ValueError(
            "log_std_bound_mode='tanh' requires initial_log_std to be strictly "
            "inside [min_log_std, max_log_std]."
        )
    mid = 0.5 * (minimum + maximum)
    half_range = 0.5 * (maximum - minimum)
    normalized = (initial - mid) / half_range
    return math.atanh(normalized)


def bounded_log_std(
    raw_log_std: torch.Tensor,
    *,
    min_log_std: float,
    max_log_std: float,
    bound_mode: str,
) -> torch.Tensor:
    mode = resolve_log_std_bound_mode(bound_mode)
    minimum = float(min_log_std)
    maximum = float(max_log_std)
    if maximum < minimum:
        raise ValueError("max_log_std must be greater than or equal to min_log_std.")
    if mode == "clamp":
        return torch.clamp(raw_log_std, min=minimum, max=maximum)
    if maximum <= minimum:
        return torch.full_like(raw_log_std, minimum)
    mid = 0.5 * (minimum + maximum)
    half_range = 0.5 * (maximum - minimum)
    return mid + half_range * torch.tanh(raw_log_std)


def physical_log_std(
    log_std: torch.Tensor,
    *,
    min_log_std: float,
    max_log_std: float,
    log_std_scale: torch.Tensor,
    bound_mode: str = "clamp",
) -> torch.Tensor:
    return (
        bounded_log_std(
            log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            bound_mode=bound_mode,
        )
        + log_std_scale
    )
