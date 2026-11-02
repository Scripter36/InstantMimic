"""PPO learner for per-frame Gaussian reset residuals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from instantmimic.rl.initial_state import InitialStateSpec


def diagonal_gaussian_log_prob(
    sample: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    """Return the summed diagonal-Gaussian log probability per sample."""

    normalized = (sample - mean) * torch.exp(-log_std)
    return -0.5 * (normalized.square() + 2.0 * log_std + math.log(2.0 * math.pi)).sum(
        dim=-1
    )


def _reference_standard_normal_kl(
    mean: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (mean.square() + torch.exp(2.0 * log_std) - 1.0 - 2.0 * log_std).sum(
        dim=-1
    )


class _MaskedTableAdam:
    """Adam state whose rows advance only when their component is observed."""

    def __init__(
        self,
        mean: nn.Parameter,
        log_std: nn.Parameter,
        *,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1.0e-8,
    ) -> None:
        self.mean = mean
        self.log_std = log_std
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.steps = torch.zeros(mean.shape[0], device=mean.device, dtype=torch.long)
        self.mean_exp_avg = torch.zeros_like(mean)
        self.mean_exp_avg_sq = torch.zeros_like(mean)
        self.log_std_exp_avg = torch.zeros_like(log_std)
        self.log_std_exp_avg_sq = torch.zeros_like(log_std)

    @torch.no_grad()
    def step(self, observed: torch.Tensor) -> None:
        if self.mean.grad is None or self.log_std.grad is None:
            raise RuntimeError("initial-state gradients are unavailable")
        observed_rows = observed[:, None]
        observed_float = observed_rows.to(dtype=self.mean.dtype)
        self.steps.add_(observed.to(dtype=torch.long))
        for parameter, exp_avg, exp_avg_sq in (
            (self.mean, self.mean_exp_avg, self.mean_exp_avg_sq),
            (self.log_std, self.log_std_exp_avg, self.log_std_exp_avg_sq),
        ):
            gradient = parameter.grad
            updated_exp_avg = exp_avg * self.beta1 + gradient * (1.0 - self.beta1)
            updated_exp_avg_sq = exp_avg_sq * self.beta2 + gradient.square() * (
                1.0 - self.beta2
            )
            exp_avg.copy_(torch.where(observed_rows, updated_exp_avg, exp_avg))
            exp_avg_sq.copy_(torch.where(observed_rows, updated_exp_avg_sq, exp_avg_sq))
            safe_steps = self.steps.clamp_min(1).to(dtype=parameter.dtype)
            bias_correction1 = 1.0 - torch.pow(self.beta1, safe_steps)
            bias_correction2 = 1.0 - torch.pow(self.beta2, safe_steps)
            denominator = exp_avg_sq.sqrt() / bias_correction2.sqrt()[:, None]
            denominator.add_(self.eps)
            update = exp_avg / bias_correction1[:, None] / denominator
            parameter.add_(
                update * observed_float,
                alpha=-self.learning_rate,
            )
            gradient.zero_()

    def state_dict(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "steps": self.steps,
            "mean_exp_avg": self.mean_exp_avg,
            "mean_exp_avg_sq": self.mean_exp_avg_sq,
            "log_std_exp_avg": self.log_std_exp_avg,
            "log_std_exp_avg_sq": self.log_std_exp_avg_sq,
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        for name in ("learning_rate", "beta1", "beta2", "eps"):
            if name in state:
                setattr(self, name, float(state[name]))
        for name in (
            "steps",
            "mean_exp_avg",
            "mean_exp_avg_sq",
            "log_std_exp_avg",
            "log_std_exp_avg_sq",
        ):
            value = state.get(name)
            target = getattr(self, name)
            if not torch.is_tensor(value) or value.shape != target.shape:
                raise ValueError(
                    f"initial-state optimizer tensor {name!r} is incompatible"
                )
            target.copy_(value)


class InitialStateLearner(nn.Module):
    """Learn Gaussian reset residuals from PPO return targets.

    The environment samples from ``mean`` and ``log_std`` at reset time.  The
    rollout records that sample at the first policy action and this learner
    consumes only the matching TD(lambda) return.  The simulator remains
    non-differentiable; gradients use the likelihood-ratio estimator.
    """

    def __init__(
        self,
        *,
        spec: InitialStateSpec,
        num_envs: int,
        rollout_steps: int,
        device: torch.device,
        config: Mapping[str, Any],
        inference_only: bool = False,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.num_envs = int(num_envs)
        self.rollout_steps = int(rollout_steps)
        self.inference_only = bool(inference_only)
        if self.num_envs < 1 or self.rollout_steps < 1:
            raise ValueError("initial-state rollout dimensions must be positive")

        initial_log_std = float(config.get("initial_log_std", 0.0))
        bounds = config.get("log_std_bounds", (-5.0, 2.0))
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("initial_state.log_std_bounds must contain two values")
        self.min_log_std, self.max_log_std = (float(bounds[0]), float(bounds[1]))
        if not self.min_log_std < self.max_log_std:
            raise ValueError("initial_state.log_std_bounds must be increasing")
        if not self.min_log_std <= initial_log_std <= self.max_log_std:
            raise ValueError("initial_state.initial_log_std is outside its bounds")

        for name, value in (
            ("initial_log_std", initial_log_std),
            ("log_std_bounds", self.min_log_std),
            ("log_std_bounds", self.max_log_std),
        ):
            if not math.isfinite(value):
                raise ValueError(f"initial_state.{name} must be finite")

        self.ratio_clip = float(config.get("ratio_clip", 0.2))
        self.reference_kl_weight = float(config.get("reference_kl_weight", 0.0))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.baseline_ema = float(config.get("baseline_ema", 0.95))
        sample_age = float(config.get("max_sample_age", 1))
        if not math.isfinite(sample_age) or not sample_age.is_integer():
            raise ValueError("initial_state.max_sample_age must be a finite integer")
        self.max_sample_age = int(sample_age)
        for name in (
            "ratio_clip",
            "reference_kl_weight",
            "max_grad_norm",
            "baseline_ema",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"initial_state.{name} must be finite")
        if self.ratio_clip < 0.0:
            raise ValueError("initial_state.ratio_clip must be non-negative")
        if self.reference_kl_weight < 0.0:
            raise ValueError("initial_state.reference_kl_weight must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("initial_state.max_grad_norm must be positive")
        if not 0.0 <= self.baseline_ema < 1.0:
            raise ValueError("initial_state.baseline_ema must be in [0, 1)")
        if self.max_sample_age < 0:
            raise ValueError("initial_state.max_sample_age must be non-negative")

        shape = (spec.num_components, spec.latent_dim)
        self.mean = nn.Parameter(torch.zeros(shape, device=device))
        self.log_std = nn.Parameter(torch.full(shape, initial_log_std, device=device))
        self.register_buffer(
            "version", torch.zeros((), device=device, dtype=torch.long)
        )
        self.register_buffer(
            "baseline", torch.zeros(spec.num_components, device=device)
        )
        self.register_buffer(
            "baseline_initialized",
            torch.zeros(spec.num_components, device=device, dtype=torch.bool),
        )

        rollout_shape = (self.rollout_steps, self.num_envs)
        sample_arity = int(spec.sample_arity)
        self.register_buffer(
            "start_mask",
            torch.zeros(rollout_shape, device=device, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "samples",
            torch.zeros((*rollout_shape, sample_arity, spec.latent_dim), device=device),
            persistent=False,
        )
        self.register_buffer(
            "component_ids",
            torch.zeros(
                (*rollout_shape, sample_arity), device=device, dtype=torch.long
            ),
            persistent=False,
        )
        self.register_buffer(
            "old_log_prob",
            torch.zeros(rollout_shape, device=device),
            persistent=False,
        )
        self.register_buffer(
            "sample_versions",
            torch.zeros(rollout_shape, device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "return_targets",
            torch.zeros((*rollout_shape, 1), device=device),
            persistent=False,
        )

        self.optimizer: _MaskedTableAdam | None
        if self.inference_only:
            self.optimizer = None
        else:
            learning_rate = float(config.get("learning_rate", 1.0e-3))
            if not math.isfinite(learning_rate) or learning_rate <= 0.0:
                raise ValueError(
                    "initial_state.learning_rate must be finite and positive"
                )
            self.optimizer = _MaskedTableAdam(
                self.mean,
                self.log_std,
                learning_rate=learning_rate,
            )

        self.register_buffer(
            "component_counts",
            torch.zeros(spec.num_components, device=device),
            persistent=False,
        )
        self.register_buffer(
            "component_return_sums",
            torch.zeros(spec.num_components, device=device),
            persistent=False,
        )
        self.register_buffer(
            "participating_component_counts",
            torch.zeros(spec.num_components, device=device),
            persistent=False,
        )

    @torch.no_grad()
    def store_start_samples(
        self,
        *,
        step: int,
        mask: torch.Tensor,
        samples: torch.Tensor,
        component_ids: torch.Tensor,
        old_log_prob: torch.Tensor,
        versions: torch.Tensor,
    ) -> None:
        if not 0 <= step < self.rollout_steps:
            raise IndexError(f"initial-state rollout step is out of range: {step}")
        expected_vector = (self.num_envs,)
        expected_samples = (
            self.num_envs,
            self.spec.sample_arity,
            self.spec.latent_dim,
        )
        expected_component_ids = (self.num_envs, self.spec.sample_arity)
        if self.spec.sample_arity == 1:
            if samples.shape == (self.num_envs, self.spec.latent_dim):
                samples = samples.unsqueeze(1)
            if component_ids.shape == expected_vector:
                component_ids = component_ids.unsqueeze(1)
        if mask.shape != expected_vector or mask.dtype != torch.bool:
            raise ValueError("initial-state mask has an incompatible layout")
        if samples.shape != expected_samples:
            raise ValueError("initial-state samples have an incompatible layout")
        if component_ids.shape != expected_component_ids:
            raise ValueError("initial-state component_ids has an incompatible layout")
        for name, value in (("old_log_prob", old_log_prob), ("versions", versions)):
            if value.shape != expected_vector:
                raise ValueError(f"initial-state {name} has an incompatible layout")
        self.start_mask[step].copy_(mask)
        self.samples[step].copy_(samples)
        self.component_ids[step].copy_(component_ids)
        self.old_log_prob[step].copy_(old_log_prob)
        self.sample_versions[step].copy_(versions)

    @torch.no_grad()
    def capture_returns(self, returns: torch.Tensor) -> None:
        if returns.shape != self.return_targets.shape:
            raise ValueError(
                "initial-state returns have an incompatible layout: "
                f"{tuple(returns.shape)} != {tuple(self.return_targets.shape)}"
            )
        self.return_targets.copy_(returns)

    @torch.no_grad()
    def clear_rollout(self) -> None:
        self.start_mask.zero_()
        self.samples.zero_()
        self.component_ids.zero_()
        self.old_log_prob.zero_()
        self.sample_versions.zero_()
        self.return_targets.zero_()

    def update(self) -> dict[str, float]:
        if self.optimizer is None:
            return {}
        age = self.version - self.sample_versions
        valid = self.start_mask & (age >= 0) & (age <= self.max_sample_age)
        start_component_ids = self.component_ids[..., 0]
        valid &= (start_component_ids >= 0) & (
            start_component_ids < self.spec.num_components
        )
        valid_flat = valid.reshape(-1)
        valid_float = valid_flat.to(dtype=self.mean.dtype)
        sample_count = valid_float.sum()
        denominator = sample_count.clamp_min(1.0)
        samples = self.samples.reshape(-1, self.spec.sample_arity, self.spec.latent_dim)
        raw_component_ids = self.component_ids.reshape(-1, self.spec.sample_arity)
        participating = (
            valid_flat[:, None]
            & (raw_component_ids >= 0)
            & (raw_component_ids < self.spec.num_components)
        )
        # Inactive slots carry zero weight, so any row is safe.  Spread them over
        # the table instead of one row so the gradient accumulation does not
        # contend on a single address.
        inactive_component_ids = (
            torch.arange(
                raw_component_ids.numel(),
                device=raw_component_ids.device,
                dtype=raw_component_ids.dtype,
            ).view_as(raw_component_ids)
            % self.spec.num_components
        )
        component_ids = torch.where(
            participating,
            raw_component_ids,
            inactive_component_ids,
        )
        old_log_prob = self.old_log_prob.reshape(-1)
        returns = self.return_targets.reshape(-1)
        # Gather with index_select: its backward accumulates atomically, while
        # advanced indexing backpropagates through a sorted index_put that
        # serializes every duplicate row, i.e. every reset of a shared frame.
        flat_component_ids = component_ids.reshape(-1)
        gathered_shape = (*component_ids.shape, self.spec.latent_dim)
        selected_mean = self.mean.index_select(0, flat_component_ids).view(
            gathered_shape
        )
        selected_log_std = self.log_std.index_select(0, flat_component_ids).view(
            gathered_shape
        )
        safe_samples = torch.where(
            participating[:, :, None],
            samples,
            selected_mean.detach(),
        )
        component_log_prob = diagonal_gaussian_log_prob(
            safe_samples, selected_mean, selected_log_std
        )
        new_log_prob = (
            component_log_prob * participating.to(dtype=component_log_prob.dtype)
        ).sum(dim=-1)
        safe_old_log_prob = torch.where(
            valid_flat, old_log_prob, torch.zeros_like(old_log_prob)
        )
        log_ratio = torch.where(
            valid_flat,
            new_log_prob - safe_old_log_prob,
            torch.zeros_like(new_log_prob),
        )
        ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
        start_ids = component_ids[:, 0]
        baseline = self.baseline.index_select(0, start_ids)
        initialized = self.baseline_initialized.index_select(0, start_ids)
        safe_returns = torch.where(valid_flat, returns, torch.zeros_like(returns))
        advantage = safe_returns - torch.where(
            initialized, baseline, torch.zeros_like(baseline)
        )
        advantage = advantage.detach()
        unclipped = ratio * advantage
        clipped = ratio.clamp(1.0 - self.ratio_clip, 1.0 + self.ratio_clip) * advantage
        policy_loss = -(torch.minimum(unclipped, clipped) * valid_float).sum()
        policy_loss = policy_loss / denominator

        self.component_counts.zero_()
        self.component_return_sums.zero_()
        self.participating_component_counts.zero_()
        self.component_counts.scatter_add_(0, start_ids, valid_float)
        self.component_return_sums.scatter_add_(
            0, start_ids, safe_returns * valid_float
        )
        self.participating_component_counts.scatter_add_(
            0,
            component_ids.reshape(-1),
            participating.reshape(-1).to(dtype=self.mean.dtype),
        )
        observed_components = self.participating_component_counts > 0.0
        observed_float = observed_components.to(dtype=self.mean.dtype)
        observed_count = observed_float.sum().clamp_min(1.0)
        reference_kl = (
            _reference_standard_normal_kl(self.mean, self.log_std) * observed_float
        ).sum() / observed_count
        loss = policy_loss + self.reference_kl_weight * reference_kl
        if self.mean.grad is not None:
            self.mean.grad.zero_()
        if self.log_std.grad is not None:
            self.log_std.grad.zero_()
        loss.backward()
        assert self.mean.grad is not None
        assert self.log_std.grad is not None
        grad_norm = torch.sqrt(
            self.mean.grad.square().sum() + self.log_std.grad.square().sum()
        )
        grad_scale = (self.max_grad_norm / (grad_norm + 1.0e-8)).clamp(max=1.0)
        self.mean.grad.mul_(grad_scale)
        self.log_std.grad.mul_(grad_scale)
        self.optimizer.step(observed_components)
        with torch.no_grad():
            self.log_std.clamp_(self.min_log_std, self.max_log_std)
            batch_mean = self.component_return_sums / self.component_counts.clamp_min(
                1.0
            )
            ema_baseline = self.baseline * self.baseline_ema + batch_mean * (
                1.0 - self.baseline_ema
            )
            next_baseline = torch.where(
                self.baseline_initialized,
                ema_baseline,
                batch_mean,
            )
            self.baseline.copy_(
                torch.where(
                    self.component_counts > 0.0,
                    next_baseline,
                    self.baseline,
                )
            )
            self.baseline_initialized.logical_or_(self.component_counts > 0.0)
            self.version.add_((sample_count > 0.0).to(dtype=torch.long))
            return_mean = (safe_returns * valid_float).sum() / denominator
            metric_values = torch.stack(
                (
                    loss.detach(),
                    policy_loss.detach(),
                    reference_kl.detach(),
                    return_mean,
                    sample_count,
                    self.mean.norm(),
                    self.log_std.exp().mean(),
                    grad_norm,
                    self.version.to(dtype=self.mean.dtype),
                )
            ).cpu()
            self.clear_rollout()
        if metric_values[4].item() == 0.0:
            return {}
        metric_names = (
            "initial_state_loss",
            "initial_state_policy_loss",
            "initial_state_reference_kl",
            "initial_state_return_mean",
            "initial_state_sample_count",
            "initial_state_mean_norm",
            "initial_state_std_mean",
            "initial_state_grad_norm",
            "initial_state_version",
        )
        return dict(zip(metric_names, metric_values.tolist(), strict=True))

    def full_state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "schema_version": 1,
            "spec": self.spec.to_dict(),
            "model": super().state_dict(),
        }
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def inference_state_dict(self) -> dict[str, object]:
        # CheckpointManager captures its CUDA snapshot graph once and replays it
        # throughout training.  Keep these sources tied to the live module
        # storage, just like nn.Module.state_dict(), rather than capturing
        # short-lived clones from the initial call.
        return {
            "schema_version": 1,
            "spec": self.spec.to_dict(),
            "model": {
                "mean": self.mean.detach(),
                "log_std": self.log_std.detach(),
                "version": self.version.detach(),
                "baseline": self.baseline.detach(),
                "baseline_initialized": self.baseline_initialized.detach(),
            },
        }

    def load_state_dict(  # type: ignore[override]
        self,
        state_dict: Mapping[str, object],
        strict: bool = True,
        *,
        load_optimizer: bool = False,
    ) -> Any:
        del strict
        spec = state_dict.get("spec")
        if not isinstance(spec, Mapping):
            raise ValueError("initial-state checkpoint is missing its spec")
        if spec.get("fingerprint") != self.spec.fingerprint:
            raise ValueError(
                "initial-state checkpoint fingerprint does not match the environment: "
                f"checkpoint={spec.get('fingerprint')!r}, "
                f"environment={self.spec.fingerprint!r}"
            )
        reference_pose_fingerprint = spec.get("reference_pose_fingerprint")
        if (
            reference_pose_fingerprint
            and reference_pose_fingerprint != self.spec.reference_pose_fingerprint
        ):
            raise ValueError(
                "initial-state checkpoint reference poses do not match the environment"
            )
        model = state_dict.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("initial-state checkpoint is missing model state")
        result = super().load_state_dict(model, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                "initial-state checkpoint model mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        optimizer = state_dict.get("optimizer")
        if load_optimizer and optimizer is not None:
            if self.optimizer is None:
                raise ValueError("cannot load initializer optimizer in inference mode")
            self.optimizer.load_state_dict(optimizer)  # type: ignore[arg-type]
        self.clear_rollout()
        return result


__all__ = ["InitialStateLearner", "diagonal_gaussian_log_prob"]
