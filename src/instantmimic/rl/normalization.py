"""State, value, and reward normalization shared by RL agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from instantmimic.utils.policy_mean_contract import register_policy_mean_methods


@dataclass(frozen=True, slots=True)
class ObservationNormalizationContract:
    """Temporal coordinate contract shared by rollout and agent code.

    A normalizer that advances statistics after every environment step changes
    the coordinates seen by the behavior policy within one rollout. On-policy
    updates must therefore replay the exact normalized policy inputs rather
    than re-normalizing raw observations with later statistics.
    """

    statistics_update: Literal["disabled", "rollout_step", "rollout_batch"]
    policy_input_replay: Literal["recompute", "snapshot"]

    def __post_init__(self) -> None:
        if self.statistics_update not in {
            "disabled",
            "rollout_step",
            "rollout_batch",
        }:
            raise ValueError(
                "unsupported observation statistics update cadence: "
                f"{self.statistics_update!r}"
            )
        if self.policy_input_replay not in {"recompute", "snapshot"}:
            raise ValueError(
                f"unsupported policy input replay mode: {self.policy_input_replay!r}"
            )
        if (
            self.statistics_update == "rollout_step"
            and self.policy_input_replay != "snapshot"
        ):
            raise ValueError(
                "per-step observation normalization requires policy input snapshots"
            )

    @property
    def requires_policy_input_snapshot(self) -> bool:
        return self.policy_input_replay == "snapshot"


class RunningStandardScaler(nn.Module):
    """Running mean / variance normalizer using a parallel variance update.

    ``frozen_prefix_dim`` keeps the statistics of the leading features fixed
    (a frozen pretrained policy reads them); updates and ``current_count`` then
    cover only the trailing features.
    """

    def __init__(
        self,
        size: int | Sequence[int],
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        enabled: bool = True,
        max_samples: int | None = None,
        device: torch.device | str | None = None,
        frozen_prefix_dim: int = 0,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold
        self.enabled = enabled
        self.max_samples = (
            float(max_samples) if max_samples is not None and max_samples >= 0 else None
        )
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.contract = ObservationNormalizationContract(
            statistics_update="rollout_batch" if self.enabled else "disabled",
            policy_input_replay="recompute",
        )

        shape = (size,) if isinstance(size, int) else tuple(size)
        self.frozen_prefix_dim = int(frozen_prefix_dim)
        if self.frozen_prefix_dim < 0 or (
            self.frozen_prefix_dim > 0
            and (len(shape) != 1 or self.frozen_prefix_dim > int(shape[0]))
        ):
            raise ValueError(
                f"frozen_prefix_dim={self.frozen_prefix_dim} must lie within the "
                f"one-dimensional feature shape {shape}"
            )
        self.register_buffer(
            "running_mean", torch.zeros(shape, dtype=torch.float64, device=self.device)
        )
        self.register_buffer(
            "running_variance",
            torch.ones(shape, dtype=torch.float64, device=self.device),
        )
        self.register_buffer(
            "current_count", torch.ones((), dtype=torch.float64, device=self.device)
        )

    def _parallel_variance(
        self, input_mean: torch.Tensor, input_var: torch.Tensor, input_count: int
    ) -> None:
        running_mean = self.running_mean[self.frozen_prefix_dim :]
        running_variance = self.running_variance[self.frozen_prefix_dim :]
        delta = input_mean.double() - running_mean
        total_count = self.current_count + input_count
        m2 = (
            (running_variance * self.current_count)
            + (input_var.double() * input_count)
            + delta.pow(2) * self.current_count * input_count / total_count
        )
        # Keep buffer storage stable for CUDA graph replay by updating in-place.
        running_mean.add_(delta * input_count / total_count)
        running_variance.copy_(m2 / total_count)
        self.current_count.copy_(total_count)

    @torch.no_grad()
    def update_stats(self, x: torch.Tensor) -> None:
        """Commit a batch's population moments without producing normalized output."""
        if not self.enabled:
            return
        if self.max_samples is not None:
            effective_count = float(self.current_count.item()) - 1.0
            if effective_count >= self.max_samples:
                return
        if self.frozen_prefix_dim > 0:
            if self.frozen_prefix_dim == int(self.running_mean.shape[0]):
                return
            x = x[..., self.frozen_prefix_dim :]
        if x.dim() == 3:
            self._parallel_variance(
                torch.mean(x, dim=(0, 1)),
                torch.var(x, dim=(0, 1), unbiased=False),
                x.shape[0] * x.shape[1],
            )
            return
        self._parallel_variance(
            torch.mean(x, dim=0),
            torch.var(x, dim=0, unbiased=False),
            x.shape[0],
        )

    @torch.no_grad()
    def observe_rollout_step(self, x: torch.Tensor) -> bool:
        """Leave batch-updated statistics frozen throughout a rollout."""
        del x
        return False

    @torch.no_grad()
    def observe_rollout_batch(self, x: torch.Tensor) -> bool:
        """Advance statistics once after materializing rollout coordinates."""
        if not self.enabled:
            return False
        self.update_stats(x)
        return True

    def _compute(
        self,
        x: torch.Tensor,
        train: bool = False,
        inverse: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            if out is None:
                return x
            out.copy_(x)
            return out

        if train:
            self.update_stats(x)

        running_std = torch.sqrt(self.running_variance.float())
        running_mean = self.running_mean.float()
        if inverse:
            if out is None:
                return (
                    running_std
                    * torch.clamp(x, min=-self.clip_threshold, max=self.clip_threshold)
                    + running_mean
                )
            out.copy_(x)
            out.clamp_(-self.clip_threshold, self.clip_threshold)
            out.mul_(running_std)
            out.add_(running_mean)
            return out

        if out is None:
            return torch.clamp(
                (x - running_mean) / (running_std + self.epsilon),
                min=-self.clip_threshold,
                max=self.clip_threshold,
            )
        out.copy_(x)
        out.sub_(running_mean)
        out.div_(running_std + self.epsilon)
        out.clamp_(-self.clip_threshold, self.clip_threshold)
        return out

    def forward(
        self,
        x: torch.Tensor,
        train: bool = False,
        inverse: bool = False,
        no_grad: bool = True,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if no_grad:
            with torch.no_grad():
                return self._compute(x, train=train, inverse=inverse, out=out)
        return self._compute(x, train=train, inverse=inverse, out=out)


class RslRlEmpiricalNormalization(nn.Module):
    """RSL-RL 5.4 empirical observation normalization semantics."""

    def __init__(
        self,
        size: int | Sequence[int],
        *,
        device: torch.device | str | None = None,
        epsilon: float = 1.0e-2,
    ) -> None:
        super().__init__()
        shape = (size,) if isinstance(size, int) else tuple(size)
        resolved_device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.epsilon = float(epsilon)
        self.enabled = True
        self.contract = ObservationNormalizationContract(
            statistics_update="rollout_step",
            policy_input_replay="snapshot",
        )
        self.register_buffer(
            "running_mean",
            torch.zeros(shape, dtype=torch.float32, device=resolved_device),
        )
        self.register_buffer(
            "running_variance",
            torch.ones(shape, dtype=torch.float32, device=resolved_device),
        )
        self.register_buffer(
            "running_std",
            torch.ones(shape, dtype=torch.float32, device=resolved_device),
        )
        self.register_buffer(
            "current_count",
            torch.zeros((), dtype=torch.long, device=resolved_device),
        )

    @torch.no_grad()
    def update_stats(self, x: torch.Tensor) -> None:
        if x.dim() == 3:
            stats_dims = (0, 1)
            count_x = x.shape[0] * x.shape[1]
        else:
            stats_dims = 0
            count_x = x.shape[0]
        self.current_count.add_(count_x)
        rate = count_x / self.current_count
        input_var = torch.var(x, dim=stats_dims, unbiased=False)
        input_mean = torch.mean(x, dim=stats_dims)
        delta_mean = input_mean - self.running_mean
        self.running_mean.add_(rate * delta_mean)
        self.running_variance.add_(
            rate
            * (
                input_var
                - self.running_variance
                + delta_mean * (input_mean - self.running_mean)
            )
        )
        self.running_std.copy_(torch.sqrt(self.running_variance))

    @torch.no_grad()
    def observe_rollout_step(self, x: torch.Tensor) -> bool:
        """Advance RSL-RL moments after each environment transition."""
        self.update_stats(x)
        return True

    @torch.no_grad()
    def observe_rollout_batch(self, x: torch.Tensor) -> bool:
        """Avoid double-counting samples already observed step by step."""
        del x
        return False

    def forward(
        self,
        x: torch.Tensor,
        train: bool = False,
        inverse: bool = False,
        no_grad: bool = True,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del no_grad
        if train:
            self.update_stats(x)
        if out is None:
            if inverse:
                return x * (self.running_std + self.epsilon) + self.running_mean
            return (x - self.running_mean) / (self.running_std + self.epsilon)
        out.copy_(x)
        if inverse:
            out.mul_(self.running_std + self.epsilon)
            out.add_(self.running_mean)
        else:
            out.sub_(self.running_mean)
            out.div_(self.running_std + self.epsilon)
        return out


class RunningReturnRewardScaler:
    """Scale rewards by a running discounted-return magnitude estimate."""

    def __init__(
        self,
        *,
        num_envs: int,
        gamma: float,
        g_max: float,
        device: torch.device,
        eps: float = 1.0e-8,
    ) -> None:
        self.gamma = float(gamma)
        self.g_max = float(g_max)
        self.eps = float(eps)
        self.device = device
        self.return_trace = torch.zeros(
            int(num_envs), 1, dtype=torch.float32, device=device
        )
        self.return_abs_max = torch.zeros(1, dtype=torch.float32, device=device)
        self.running_mean = torch.zeros(1, dtype=torch.float32, device=device)
        self.running_var = torch.ones(1, dtype=torch.float32, device=device)
        self.running_count = torch.zeros(1, dtype=torch.float32, device=device)

    @torch.no_grad()
    def _update_running(self, samples: torch.Tensor) -> None:
        samples = samples.reshape(-1, 1).float()
        sample_count = float(samples.shape[0])
        if sample_count <= 0.0:
            return
        sample_mean = samples.mean(dim=0)
        sample_var = samples.var(dim=0, unbiased=False)
        delta = sample_mean - self.running_mean
        total_count = self.running_count + sample_count
        ratio = sample_count / total_count.clamp_min(1.0e-8)
        new_mean = self.running_mean + delta * ratio
        m_a = self.running_var * (self.running_count + 1.0e-4)
        m_b = sample_var * sample_count
        m2 = m_a + m_b + delta.square() * self.running_count * ratio
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(m2 / total_count.clamp_min(1.0e-8))
        self.running_count.copy_(total_count)

    @torch.no_grad()
    def update_rollout(
        self,
        *,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        step_count = int(rewards.shape[0])
        for step in range(step_count):
            valid = valid_mask[step].reshape(-1, 1)
            step_reward = torch.where(
                valid,
                rewards[step].reshape(-1, 1).float(),
                torch.zeros_like(self.return_trace),
            )
            step_done = torch.where(
                valid,
                dones[step].reshape(-1, 1).float(),
                torch.ones_like(self.return_trace),
            )
            next_trace = self.return_trace.mul(self.gamma).add(step_reward)
            self.return_trace.copy_(
                torch.where(valid, next_trace, torch.zeros_like(self.return_trace))
            )
            self.return_abs_max.copy_(
                torch.maximum(
                    self.return_abs_max, self.return_trace.abs().max().view(1)
                )
            )
            self._update_running(self.return_trace[valid.reshape(-1)])
            self.return_trace.mul_(1.0 - step_done)

    def reward_scale(self) -> torch.Tensor:
        var_denominator = torch.sqrt(self.running_var + self.eps)
        max_denominator = self.return_abs_max / max(self.g_max, 1.0e-8)
        return torch.maximum(var_denominator, max_denominator).clamp_min(self.eps)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        return rewards / self.reward_scale().to(
            device=rewards.device, dtype=rewards.dtype
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "return_trace": self.return_trace.detach().clone(),
            "return_abs_max": self.return_abs_max.detach().clone(),
            "running_mean": self.running_mean.detach().clone(),
            "running_var": self.running_var.detach().clone(),
            "running_count": self.running_count.detach().clone(),
        }

    def live_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "return_trace": self.return_trace.detach(),
            "return_abs_max": self.return_abs_max.detach(),
            "running_mean": self.running_mean.detach(),
            "running_var": self.running_var.detach(),
            "running_count": self.running_count.detach(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        return_trace = state.get("return_trace")
        if isinstance(return_trace, torch.Tensor):
            loaded_trace = return_trace.to(self.device)
            if tuple(loaded_trace.shape) == tuple(self.return_trace.shape):
                self.return_trace.copy_(loaded_trace)
            else:
                self.return_trace.zero_()
        else:
            self.return_trace.zero_()
        self.return_abs_max.copy_(
            state["return_abs_max"].to(self.device).reshape_as(self.return_abs_max)
        )
        self.running_mean.copy_(
            state["running_mean"].to(self.device).reshape_as(self.running_mean)
        )
        self.running_var.copy_(
            state["running_var"].to(self.device).reshape_as(self.running_var)
        )
        self.running_count.copy_(
            state["running_count"].to(self.device).reshape_as(self.running_count)
        )


def resolve_observation_preprocessor_name(preprocessor_name: str | None) -> str:
    """Canonicalise an observation-preprocessor name, or raise.

    Returns one of `"identity"`, `"running_standard"` or `"rsl_rl_empirical"`.
    Both algorithms resolve through here so that a config naming
    `RslRlEmpiricalNormalization` cannot mean empirical normalization under one
    and no normalization under the other, and so that a typo is an error rather
    than a silently disabled scaler.

    This accepts more spellings than a config can carry, on purpose.
    `config/validation.py` gates `agent.state_preprocessor` to the three exact
    names `RunningStandardScaler`, `RslRlEmpiricalNormalization` and `none`, so
    the extra aliases here serve programmatic callers that build a spec directly
    and preserve what PPO's resolver already accepted. Do not narrow this to
    match the config enum, and do not widen the config enum to match this.
    """

    if preprocessor_name is None:
        return "identity"
    key = str(preprocessor_name).strip().lower().replace("-", "_")
    if key in {"none", "null", "identity", "identity_scaler", "off", "disabled"}:
        return "identity"
    if key in {
        "runningstandardscaler",
        "running_standard_scaler",
        "running_standard",
    }:
        return "running_standard"
    if key in {
        "rslrlempiricalnormalization",
        "rsl_rl_empirical_normalization",
    }:
        # Deliberately not accepting the canonical `rsl_rl_empirical` as input.
        # PPO's resolver did not, and widening it here would be an alias nobody
        # asked for -- note the asymmetry with `running_standard`, which PPO did
        # accept and which is preserved above for the same reason.
        return "rsl_rl_empirical"
    raise ValueError(
        "Unsupported preprocessor: "
        f"{preprocessor_name}. Use one of: RunningStandardScaler, "
        "RslRlEmpiricalNormalization, Identity."
    )


def build_observation_preprocessor(
    *,
    preprocessor: str | None,
    preprocessor_kwargs: Mapping[str, Any] | None,
    size: int | Sequence[int],
    device: torch.device,
) -> RunningStandardScaler | RslRlEmpiricalNormalization:
    """Build the preprocessor a name asks for.

    `identity` returns a disabled `RunningStandardScaler` rather than `None`, so
    callers keep one call signature. Both classes accept
    `(x, train, inverse, no_grad, out)`, but they are interchangeable only for
    the forward-normalizing use: `RslRlEmpiricalNormalization.forward` discards
    `no_grad` (it does not wrap the computation), and its `inverse=True` branch
    multiplies by `running_std + epsilon` without the clipping
    `RunningStandardScaler` applies. A caller that depends on either parameter
    must not assume the name it was given selects an equivalent object.
    """

    resolved = resolve_observation_preprocessor_name(preprocessor)
    kwargs = dict(preprocessor_kwargs or {})
    if resolved == "rsl_rl_empirical":
        return RslRlEmpiricalNormalization(size, device=device, **kwargs)
    return RunningStandardScaler(
        size,
        enabled=resolved == "running_standard",
        device=device,
        **kwargs,
    )


def adapt_observation_scaler_state(
    *,
    checkpoint_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    transfer: Callable[[torch.Tensor, torch.Tensor], None],
) -> dict[str, Any]:
    """Carry a narrower checkpoint's per-feature statistics into a wider scaler.

    `transfer(destination, source)` writes one buffer's checkpoint values into a
    clone of the current buffer; callers differ in how they place them (a leading
    slice, an index remap) but not in which buffers must move.

    Every per-feature buffer is carried, not only `running_mean` and
    `running_variance`. `RslRlEmpiricalNormalization` divides by `running_std`,
    so naming those two left the buffer that actually normalizes the observation
    at its freshly-initialized 1.0 while variance held the trained value -- the
    widths still matched, so `load_state_dict` succeeded and the policy silently
    saw observations scaled by ~1 instead of by the checkpoint's deviation.
    """

    adapted = dict(current_state)
    for key, current_value in current_state.items():
        source = checkpoint_state.get(key)
        if not isinstance(current_value, torch.Tensor):
            continue
        if not isinstance(source, torch.Tensor):
            continue
        if current_value.ndim != 1 or source.ndim != 1:
            continue
        destination = current_value.clone()
        transfer(destination, source)
        adapted[key] = destination
    return adapted


def resolve_reward_preprocessor_name(preprocessor_name: str | None) -> str:
    if preprocessor_name is None:
        return "identity"
    key = str(preprocessor_name).strip().lower().replace("-", "_")
    if key in {"none", "null", "identity", "identity_scaler", "off", "disabled"}:
        return "identity"
    if key in {
        "runningreturnscaler",
        "running_return_scaler",
        "running_return_reward_scaler",
    }:
        return "running_return"
    raise ValueError(
        "Unsupported reward_preprocessor: "
        f"{preprocessor_name}. Use one of: RunningReturnScaler, Identity."
    )


def build_reward_preprocessor(
    *,
    reward_preprocessor: str | None,
    reward_preprocessor_kwargs: Mapping[str, Any] | None,
    num_envs: int,
    gamma: float,
    device: torch.device,
) -> RunningReturnRewardScaler | None:
    resolved = resolve_reward_preprocessor_name(reward_preprocessor)
    if resolved == "identity":
        return None
    kwargs = dict(reward_preprocessor_kwargs or {})
    g_max = float(kwargs.pop("g_max", 5.0))
    eps = float(kwargs.pop("eps", 1.0e-8))
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise ValueError(f"Unknown reward_preprocessor_kwargs: {unknown}")
    return RunningReturnRewardScaler(
        num_envs=int(num_envs),
        gamma=float(gamma),
        g_max=g_max,
        device=device,
        eps=eps,
    )


register_policy_mean_methods(
    RunningStandardScaler,
    "standard_scaler",
    ("forward", "_compute"),
)

register_policy_mean_methods(
    RslRlEmpiricalNormalization,
    "rsl_scaler",
    ("forward",),
)
