"""Sparse muscle coordination networks and input normalization."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MuscleNNInputNormalizer(nn.Module):
    """Per-column running mean and variance of the MuscleNN input.

    ``forward`` returns ``clamp((x - mean) / max(std, min_std), -clip, clip)``
    from the statistics of every batch passed to ``update``: the cumulative
    population moments, merged batch by batch (Chan et al.) in float64. Before
    the first update the mean is zero and the deviation one, so the input
    passes through clamped. A constant column has zero variance and normalizes
    to exactly zero; ``min_std`` bounds the gain of a nearly constant one.

    Every buffer is updated in place, so CUDA graphs that read them replay
    with the current statistics, and the buffers travel with the network's
    ``state_dict``. ``forward`` reads only the float32 ``shift`` and
    ``inv_std`` derived from the float64 moments.
    """

    def __init__(self, input_dim: int, *, clip: float, min_std: float) -> None:
        super().__init__()
        if clip <= 0.0 or min_std <= 0.0:
            raise ValueError(
                "MuscleNN input normalization clip and min_std must be positive"
            )
        self.clip = float(clip)
        self.min_std = float(min_std)
        dim = int(input_dim)
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("variance", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("shift", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("inv_std", torch.ones(dim, dtype=torch.float32))
        self._refresh()

    @torch.no_grad()
    def _refresh(self) -> None:
        self.shift.copy_(self.mean)
        std = torch.sqrt(self.variance).clamp_min_(self.min_std)
        self.inv_std.copy_(std.reciprocal_())

    @torch.no_grad()
    def update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_variance: torch.Tensor,
        batch_count: int,
    ) -> None:
        """Merge one batch's per-column population moments."""
        if batch_count <= 0:
            return
        count_b = float(batch_count)
        total = self.count + count_b
        delta = batch_mean.to(torch.float64) - self.mean
        # Population M2 of the union, then the new mean, both in place.
        self.variance.mul_(self.count)
        self.variance.add_(batch_variance.to(torch.float64), alpha=count_b)
        self.variance.add_(delta.square().mul_(self.count * count_b / total))
        self.variance.div_(total)
        self.mean.add_(delta.mul_(count_b / total))
        self.count.copy_(total)
        self._refresh()

    @torch.no_grad()
    def update(self, inputs: torch.Tensor) -> None:
        """Merge the moments of a ``(samples, input_dim)`` batch."""
        variance, mean = torch.var_mean(inputs, dim=0, unbiased=False)
        self.update_from_moments(mean, variance, int(inputs.shape[0]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.clamp((inputs - self.shift) * self.inv_std, -self.clip, self.clip)


RAW_ZERO_IS_REST_OUTPUT_ACTIVATIONS = frozenset({"tanh_relu", "clamp", "identity"})


class MassMuscleNN(nn.Module):
    """Sparse torque-descriptor -> activation coordinator."""

    def __init__(
        self,
        input_dim: int,
        muscle_count: int,
        hidden_dims: Sequence[int],
        output_activation: str = "tanh_relu",
        input_normalizer: MuscleNNInputNormalizer | None = None,
    ) -> None:
        super().__init__()
        # Registered even when absent, so a network without normalization
        # keeps its historical state_dict keys.
        self.register_module("input_normalizer", input_normalizer)
        activation_key = str(output_activation).strip().lower().replace("-", "_")
        if activation_key not in {"tanh_relu", "sigmoid", "identity", "clamp", "sine"}:
            raise ValueError(
                "Unsupported MassMuscleNN output_activation: "
                f"{output_activation!r}. Use 'clamp', 'sine', 'tanh_relu', "
                "'sigmoid', or 'identity'."
            )
        dims = [int(input_dim), *[int(v) for v in hidden_dims], int(muscle_count)]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1], strict=False):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if activation_key == "sigmoid":
            layers.append(nn.Sigmoid())
        elif activation_key in {"identity", "clamp", "sine"}:
            pass
        else:
            # Match MASS: tanh followed by ReLU yields a bounded [0, 1] output.
            layers.append(nn.Tanh())
            layers.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layers)
        self.output_activation = activation_key
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def zero_output_layer_(self) -> None:
        """Initialize a residual coordinator as the exact identity command."""
        for module in reversed(list(self.net.children())):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                return
        raise RuntimeError("MassMuscleNN has no linear output layer")

    def forward_wo_activation(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        if self.input_normalizer is not None:
            output = self.input_normalizer(output)
        skip_tail = 0
        if self.output_activation == "sigmoid":
            skip_tail = 1
        elif self.output_activation == "tanh_relu":
            skip_tail = 2
        modules = list(self.net.children())
        for module in modules[: len(modules) - skip_tail if skip_tail else None]:
            output = module(output)
        return output

    def _apply_output_activation(self, raw_output: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "clamp":
            return torch.clamp(raw_output, min=0.0, max=1.0)
        if self.output_activation == "sine":
            return 0.5 * (1.0 + torch.sin(raw_output))
        if self.output_activation == "sigmoid":
            return torch.sigmoid(raw_output)
        if self.output_activation == "tanh_relu":
            return torch.relu(torch.tanh(raw_output))
        return raw_output

    def forward_with_preact(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_output = self.forward_wo_activation(inputs)
        return self._apply_output_activation(raw_output), raw_output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activation, _ = self.forward_with_preact(inputs)
        return activation
