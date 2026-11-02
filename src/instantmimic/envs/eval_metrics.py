from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

EvalMetricReduction = Literal["event_count", "step_mean"]
EvalMetricDirection = Literal["maximize", "minimize"]
EvalMetricStatistic = Literal["mean", "rate", "last", "count", "flag", "step"]
EvalValidityMode = Literal["always", "full_horizon"]
EvalSuccessMode = Literal["full_horizon", "terminal_success"]

_POST_STEP_RESERVED_NAMES = {
    "valid",
    "valid_rate",
    "valid_count",
    "invalid_count",
    "steps",
    "first_invalid_step",
    "failure_step",
    "terminated",
    "truncated",
    "reward_mean",
    "reward_mean_valid",
    "reward_mean_success",
    "episode_length_mean",
    "episode_length_mean_valid",
    "episode_length_mean_success",
    "total_envs",
    "valid_only_metrics",
    "success_defined",
    "success_rate",
    "success_count",
    "failure_count",
    "tracking_metrics_available",
    "captured_eval",
    "captured_eval_conditional_loop",
    "captured_eval_full_horizon_graph",
    "score",
}


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    if actual.type != expected.type:
        return False
    if expected.index is not None:
        return actual.index == expected.index
    if expected.type != "cuda" or actual.index is None:
        return True
    if not torch.cuda.is_available():
        return False
    return actual.index == torch.cuda.current_device()


@dataclass(frozen=True)
class EvalOutcomeSummary:
    """Canonical episode counts and their derived rollout-result fields."""

    total_envs: int
    valid_count: int
    first_invalid_step: int = 0
    success_count: int | None = None
    first_failure_step: int = 0

    def __post_init__(self) -> None:
        total_envs = int(self.total_envs)
        valid_count = int(self.valid_count)
        success_count = self.success_count
        if total_envs < 0:
            raise ValueError("eval total_envs must be non-negative")
        if valid_count < 0 or valid_count > total_envs:
            raise ValueError("eval valid_count must be between zero and total_envs")
        if int(self.first_invalid_step) < 0 or int(self.first_failure_step) < 0:
            raise ValueError("eval outcome steps must be non-negative")
        if success_count is None:
            if int(self.first_failure_step) != 0:
                raise ValueError(
                    "eval first_failure_step requires task success to be defined"
                )
        elif int(success_count) < 0 or int(success_count) > total_envs:
            raise ValueError("eval success_count must be between zero and total_envs")

    def as_result(self) -> dict[str, float | int | bool]:
        total_envs = int(self.total_envs)
        valid_count = int(self.valid_count)
        result: dict[str, float | int | bool] = {
            "valid": valid_count > 0,
            "valid_rate": float(valid_count / max(1, total_envs)),
            "valid_count": valid_count,
            "first_invalid_step": int(self.first_invalid_step),
            "total_envs": total_envs,
            "success_defined": self.success_count is not None,
        }
        if self.success_count is not None:
            success_count = int(self.success_count)
            result.update(
                success_rate=float(success_count / max(1, total_envs)),
                success_count=success_count,
                failure_count=total_envs - success_count,
                failure_step=int(self.first_failure_step),
            )
        return result


@dataclass(frozen=True)
class EvalEpisodeSpec:
    """Environment-owned rollout validity and optional task success semantics.

    ``natural_horizon_steps`` is the most control steps any episode can run
    before ``natural_completion`` ends it, for example a non-looping reference
    clip that terminates at its last frame.  ``None`` means episodes have no
    such end, as with a looping reference.
    """

    validity: EvalValidityMode
    success: EvalSuccessMode | None
    natural_completion: torch.Tensor | None = None
    terminal_success: torch.Tensor | None = None
    natural_horizon_steps: int | None = None


@dataclass(frozen=True)
class EvalMetricSpec:
    value: torch.Tensor
    reduction: EvalMetricReduction
    label: str
    default: bool = True
    statistic: EvalMetricStatistic = "mean"


@dataclass(frozen=True)
class EvalResultMetricSpec:
    """Environment-owned metadata for a metric produced by an eval adapter."""

    label: str
    result_key: str | None = None
    direction: EvalMetricDirection | None = None
    default: bool = True
    finite_required: bool = True
    statistic: EvalMetricStatistic = "mean"


def load_eval_result_metric_specs(env_raw: Any) -> dict[str, EvalResultMetricSpec]:
    getter = getattr(env_raw, "get_eval_result_metric_specs", None)
    if not callable(getter):
        return {}
    raw_specs = getter()
    if not isinstance(raw_specs, Mapping):
        raise TypeError("env.get_eval_result_metric_specs() must return a mapping")

    available: dict[str, EvalResultMetricSpec] = {}
    for raw_name, raw_spec in raw_specs.items():
        name = str(raw_name)
        if not name or name in available:
            raise ValueError(f"invalid or duplicate eval result metric name {name!r}")
        if not isinstance(raw_spec, EvalResultMetricSpec):
            raise TypeError(
                f"eval result metric {name!r} must use EvalResultMetricSpec"
            )
        if not isinstance(raw_spec.label, str) or not raw_spec.label.strip():
            raise ValueError(f"eval result metric {name!r} label must be non-empty")
        if raw_spec.result_key is not None and (
            not isinstance(raw_spec.result_key, str) or not raw_spec.result_key.strip()
        ):
            raise ValueError(
                f"eval result metric {name!r} result_key must be non-empty"
            )
        if raw_spec.direction not in {None, "maximize", "minimize"}:
            raise ValueError(
                f"eval result metric {name!r} has invalid direction "
                f"{raw_spec.direction!r}"
            )
        if not isinstance(raw_spec.default, bool):
            raise ValueError(f"eval result metric {name!r} default must be bool")
        if not isinstance(raw_spec.finite_required, bool):
            raise ValueError(
                f"eval result metric {name!r} finite_required must be bool"
            )
        if raw_spec.statistic not in {
            "mean",
            "rate",
            "last",
            "count",
            "flag",
            "step",
        }:
            raise ValueError(
                f"eval result metric {name!r} has invalid statistic "
                f"{raw_spec.statistic!r}"
            )
        if name in _POST_STEP_RESERVED_NAMES:
            raise ValueError(
                f"eval result metric {name!r} uses a reserved eval result name"
            )
        available[name] = raw_spec
    return available


def resolve_eval_result_metric_specs(
    env_raw: Any,
    *,
    requested: Sequence[str] | None,
    required: Sequence[str] = (),
) -> dict[str, EvalResultMetricSpec]:
    requested_names = tuple(requested or ())
    required_names = tuple(required)
    available = load_eval_result_metric_specs(env_raw)
    if not available and (requested_names or required_names):
        raise ValueError(
            "environment does not provide evaluation result metrics; requested: "
            + ", ".join(str(name) for name in (*requested_names, *required_names))
        )

    resolved = (
        {name: spec for name, spec in available.items() if spec.default}
        if requested is None
        else {}
    )
    for raw_name in (*requested_names, *required_names):
        name = str(raw_name)
        if name not in available:
            supported = ", ".join(sorted(available)) or "none"
            raise ValueError(
                f"unknown eval result metric {name!r}; env supports: {supported}"
            )
        if name not in resolved:
            resolved[name] = available[name]
    return resolved


def resolve_eval_metric_specs(
    env_raw: Any,
    *,
    requested: Sequence[str] | None,
    required: Sequence[str] = (),
    num_envs: int,
    device: torch.device | None = None,
) -> dict[str, EvalMetricSpec]:
    requested_names = tuple(requested or ())
    required_names = tuple(required)
    getter = getattr(env_raw, "get_eval_metric_specs", None)
    if not callable(getter):
        if requested_names or required_names:
            raise ValueError(
                "environment does not provide evaluation metrics; requested: "
                + ", ".join(str(name) for name in (*requested_names, *required_names))
            )
        return {}
    raw_specs = getter()
    if not isinstance(raw_specs, Mapping):
        raise TypeError("env.get_eval_metric_specs() must return a mapping")

    available: dict[str, EvalMetricSpec] = {}
    for raw_name, raw_spec in raw_specs.items():
        name = str(raw_name)
        if not name or name in available:
            raise ValueError(f"invalid or duplicate eval metric name {name!r}")
        if not isinstance(raw_spec, EvalMetricSpec):
            raise TypeError(f"eval metric {name!r} must use EvalMetricSpec")
        if raw_spec.reduction not in {"event_count", "step_mean"}:
            raise ValueError(
                f"eval metric {name!r} has unknown reduction {raw_spec.reduction!r}"
            )
        if not isinstance(raw_spec.default, bool):
            raise ValueError(f"eval metric {name!r} default must be bool")
        if not isinstance(raw_spec.label, str) or not raw_spec.label.strip():
            raise ValueError(f"eval metric {name!r} label must be non-empty")
        if raw_spec.statistic not in {
            "mean",
            "rate",
            "last",
            "count",
            "flag",
            "step",
        }:
            raise ValueError(
                f"eval metric {name!r} has invalid statistic {raw_spec.statistic!r}"
            )
        if name in _POST_STEP_RESERVED_NAMES:
            raise ValueError(f"eval metric {name!r} uses a reserved eval result name")
        value = raw_spec.value
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"eval metric {name!r} must expose a tensor value")
        if tuple(value.shape) not in {(int(num_envs),), (int(num_envs), 1)}:
            raise ValueError(
                f"eval metric {name!r} must expose one value per environment, "
                f"got shape {tuple(value.shape)}"
            )
        if device is not None and not _device_matches(value.device, device):
            raise ValueError(
                f"eval metric {name!r} must be on {device}, got {value.device}"
            )
        available[name] = raw_spec

    resolved = (
        {name: spec for name, spec in available.items() if spec.default}
        if requested is None
        else {}
    )
    for raw_name in (*requested_names, *required_names):
        name = str(raw_name)
        if name not in available:
            supported = ", ".join(sorted(available)) or "none"
            raise ValueError(f"unknown eval metric {name!r}; env supports: {supported}")
        if name not in resolved:
            resolved[name] = available[name]
    return resolved


def resolve_eval_episode_spec(
    env_raw: Any,
    *,
    num_envs: int,
    device: torch.device,
    fallback_validity: EvalValidityMode = "always",
    fallback_success: EvalSuccessMode | None = None,
) -> EvalEpisodeSpec:
    getter = getattr(env_raw, "get_eval_episode_spec", None)
    spec = (
        getter()
        if callable(getter)
        else EvalEpisodeSpec(
            validity=fallback_validity,
            success=fallback_success,
        )
    )
    if not isinstance(spec, EvalEpisodeSpec):
        raise TypeError("env.get_eval_episode_spec() must return EvalEpisodeSpec")
    if spec.validity not in {"always", "full_horizon"}:
        raise ValueError(f"unknown eval validity mode {spec.validity!r}")
    if spec.success not in {None, "full_horizon", "terminal_success"}:
        raise ValueError(f"unknown eval success mode {spec.success!r}")

    for name, value in (
        ("natural_completion", spec.natural_completion),
        ("terminal_success", spec.terminal_success),
    ):
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"eval episode {name} must be a tensor")
        if tuple(value.shape) != (int(num_envs),):
            raise ValueError(
                f"eval episode {name} must have shape ({int(num_envs)},), "
                f"got {tuple(value.shape)}"
            )
        if value.dtype != torch.bool:
            raise ValueError(f"eval episode {name} must use bool dtype")
        if not _device_matches(value.device, device):
            raise ValueError(
                f"eval episode {name} must be on {device}, got {value.device}"
            )
    if spec.success == "terminal_success" and spec.terminal_success is None:
        raise ValueError(
            "terminal_success evaluation requires a persistent terminal_success tensor"
        )
    _validate_natural_horizon_steps(spec)
    return spec


def _validate_natural_horizon_steps(spec: EvalEpisodeSpec) -> None:
    horizon = spec.natural_horizon_steps
    if horizon is None:
        return
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError(
            "eval episode natural_horizon_steps must be a positive int, "
            f"got {horizon!r}"
        )
    if spec.natural_completion is None:
        raise ValueError(
            "eval episode natural_horizon_steps requires a natural_completion tensor"
        )


def eval_episode_step_limit(env_raw: Any, requested_steps: int) -> int:
    """Return the most steps an evaluation episode of ``requested_steps`` can run.

    A reference that ends naturally before the requested cap shortens it; an
    environment without that contract keeps the requested value.
    """

    requested = int(requested_steps)
    getter = getattr(env_raw, "get_eval_episode_spec", None)
    if not callable(getter):
        return requested
    spec = getter()
    if not isinstance(spec, EvalEpisodeSpec):
        raise TypeError("env.get_eval_episode_spec() must return EvalEpisodeSpec")
    _validate_natural_horizon_steps(spec)
    if spec.natural_horizon_steps is None:
        return requested
    return min(requested, spec.natural_horizon_steps)


class EvalMetricAccumulator:
    def __init__(self, specs: Mapping[str, EvalMetricSpec]) -> None:
        self.specs = dict(specs)
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, torch.Tensor] = {}
        self._scratch: dict[str, torch.Tensor] = {}
        # Collection runs inside captured eval graphs, so the inverted mask is
        # persistent; a temporary would be freed and reused before replay.
        self._inactive: torch.Tensor | None = None
        for name, spec in self.specs.items():
            value = spec.value.reshape(-1)
            self._sums[name] = torch.zeros_like(value, dtype=torch.float32)
            self._scratch[name] = torch.empty_like(value, dtype=torch.float32)
            if spec.reduction == "step_mean":
                self._counts[name] = torch.zeros_like(value, dtype=torch.int32)
            if self._inactive is None:
                self._inactive = torch.empty_like(value, dtype=torch.bool)

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(self.specs)

    def reset(self) -> None:
        for value in self._sums.values():
            value.zero_()
        for value in self._counts.values():
            value.zero_()

    def validate_sources(self, specs: Mapping[str, EvalMetricSpec]) -> None:
        for name, expected in self.specs.items():
            current = specs.get(name)
            if (
                current is None
                or current.value is not expected.value
                or current.reduction != expected.reduction
            ):
                raise RuntimeError(
                    f"eval metric {name!r} changed its registered contract; "
                    "metric tensors and aggregation must remain persistent"
                )

    def collect_post_step(self, active_mask: torch.Tensor) -> None:
        active = active_mask.reshape(-1)
        if self._inactive is None:
            return
        torch.logical_not(active, out=self._inactive)
        for name, spec in self.specs.items():
            scratch = self._scratch[name]
            scratch.copy_(spec.value.reshape(-1))
            # masked_fill keeps a NaN from an inactive world out of the sum.
            scratch.masked_fill_(self._inactive, 0.0)
            self._sums[name].add_(scratch)
            if spec.reduction == "step_mean":
                self._counts[name].add_(active)

    def summarize(
        self,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        if valid_mask is not None:
            valid_mask = valid_mask.reshape(-1)
            if valid_mask.dtype != torch.bool:
                raise ValueError("eval metric valid_mask must use bool dtype")
        for name, spec in self.specs.items():
            metric_valid = (
                torch.ones_like(self._sums[name], dtype=torch.bool)
                if valid_mask is None
                else valid_mask
            )
            if tuple(metric_valid.shape) != tuple(self._sums[name].shape):
                raise ValueError(
                    f"eval metric valid_mask shape {tuple(metric_valid.shape)} does "
                    f"not match metric {name!r} shape {tuple(self._sums[name].shape)}"
                )
            if metric_valid.device != self._sums[name].device:
                raise ValueError(
                    f"eval metric valid_mask must be on {self._sums[name].device}, "
                    f"got {metric_valid.device}"
                )
            if spec.reduction == "event_count":
                result[name] = (
                    float(self._sums[name][metric_valid].mean().item())
                    if bool(metric_valid.any().item())
                    else float("nan")
                )
                continue
            counts = self._counts[name]
            valid = metric_valid & (counts > 0)
            if not bool(valid.any().item()):
                result[name] = float("nan")
                continue
            per_env = self._sums[name][valid] / counts[valid]
            result[name] = float(per_env.mean().item())
        return result

    def labels(self) -> dict[str, str]:
        return {name: spec.label for name, spec in self.specs.items()}
