"""Training-only Gaussian noise on the policy observation.

`env.observation_noise` declares the noise; `DirectRLEnv` parses it at
construction but adds nothing until the training runtime enables it on its
rollout environment (`DirectRLEnv.enable_observation_noise`). Benchmark,
playback and every other environment instance therefore see clean
observations without any gate of their own.

The noise covers every feature of the ``"policy"`` group, including muscle
geometry features when enabled. The ``"critic"`` group stays clean.

``units: normalized`` scales the draw of feature ``j`` by the normaliser's
denominator ``sqrt(running_variance[j]) + epsilon``, read live from the agent's
statistics buffer, so after normalisation the policy input receives exactly
``std * N(0, 1)`` on every feature (before the normaliser's clip). ``units:
raw`` adds ``std * N(0, 1)`` in observation units.

Until the normaliser holds statistics (its sample count is still the
initial one) nothing is added, so a fresh run's first statistics update sees
clean observations instead of baking ``std**2`` of raw-unit noise into every
feature's variance. The gate is read on the device at every launch.

The draw is counter based: feature ``j`` of env ``i`` uses
``rand_init(int(rand_init(slot_seed + common_step_counter[0])), i * width + j)``,
where the slot seed is fixed per emission point (step, terminal, reset).
Nothing on the host changes between steps, so the launch is CUDA-graph safe
and allocates nothing; a restored ``common_step_counter`` reproduces the
sequence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

OBSERVATION_NOISE_CONFIG_KEY = "observation_noise"
OBSERVATION_NOISE_UNITS = ("normalized", "raw")
_OBSERVATION_NOISE_KEYS = frozenset({"std", "units", "enabled"})

# Emission points of a policy observation. Each draws from its own seed so the
# terminal observation and the post-reset observation of one step differ.
OBSERVATION_NOISE_SLOT_STEP = 0
OBSERVATION_NOISE_SLOT_TERMINAL = 1
OBSERVATION_NOISE_SLOT_RESET = 2
_OBSERVATION_NOISE_SLOTS = (
    OBSERVATION_NOISE_SLOT_STEP,
    OBSERVATION_NOISE_SLOT_TERMINAL,
    OBSERVATION_NOISE_SLOT_RESET,
)
# Keeps slot seeds below 2**30 so adding the int32 step counter cannot overflow
# within any realistic run.
_SLOT_SEED_MASK = (1 << 30) - 1
_INT32_LIMIT = 1 << 31


@dataclass(frozen=True, slots=True)
class ObservationNoiseSettings:
    """Parsed `env.observation_noise` block."""

    std: float
    units: str = "normalized"
    enabled: bool = True

    @property
    def active(self) -> bool:
        return self.enabled and self.std > 0.0


def parse_observation_noise_settings(block: Any) -> ObservationNoiseSettings | None:
    """Validate an `env.observation_noise` mapping; `None` when it is absent."""

    if block is None:
        return None
    if not isinstance(block, Mapping) and not callable(getattr(block, "keys", None)):
        raise TypeError("env.observation_noise must be a mapping")
    keys = {str(key) for key in block.keys()}
    unknown = keys - _OBSERVATION_NOISE_KEYS
    if unknown:
        raise ValueError(
            f"unknown env.observation_noise keys: {sorted(unknown)}; "
            f"expected a subset of {sorted(_OBSERVATION_NOISE_KEYS)}"
        )
    if "std" not in keys:
        raise ValueError("env.observation_noise.std is required")
    raw_std = block["std"]
    if isinstance(raw_std, bool) or not isinstance(raw_std, int | float):
        raise TypeError(
            f"env.observation_noise.std must be a number, got {type(raw_std).__name__}"
        )
    std = float(raw_std)
    if not math.isfinite(std) or std < 0.0:
        raise ValueError(
            f"env.observation_noise.std must be finite and >= 0, got {std}"
        )
    units = str(block.get("units", "normalized"))
    if units not in OBSERVATION_NOISE_UNITS:
        raise ValueError(
            f"env.observation_noise.units must be one of {list(OBSERVATION_NOISE_UNITS)}, "
            f"got {units!r}"
        )
    enabled = block.get("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError(
            "env.observation_noise.enabled must be a bool, "
            f"got {type(enabled).__name__}"
        )
    return ObservationNoiseSettings(std=std, units=units, enabled=enabled)


def observation_noise_slot_seed(seed: int | None, slot: int) -> int:
    """Seed of one emission slot, decorrelated from the env's other streams."""

    base = 0 if seed is None or int(seed) < 0 else int(seed)
    mixed = (base * 0x9E3779B1 + (slot + 1) * 0x85EBCA77 + 0x6F6E6F69) & 0xFFFFFFFF
    mixed ^= mixed >> 15
    mixed = (mixed * 0x2C1B3C6D) & 0xFFFFFFFF
    mixed ^= mixed >> 12
    return mixed & _SLOT_SEED_MASK


@wp.func
def _noise_state(
    slot_seed: wp.int32,
    step_counter: wp.int32,
    env_idx: wp.int32,
    feature_idx: wp.int32,
    width: wp.int32,
):
    # Hash the per-step seed before mixing in the offset: Warp's
    # rand_init(seed, offset) is pcg(seed + pcg(offset)), so feeding the raw
    # counter would make some offsets replay another offset's sequence shifted
    # by a fixed number of steps.
    step_seed = wp.int32(wp.rand_init(slot_seed + step_counter))
    return wp.rand_init(step_seed, env_idx * width + feature_idx)


@wp.kernel(enable_backward=False)
def _add_observation_noise_f32_kernel(
    observations: wp.array2d(dtype=wp.float32),
    feature_variance: wp.array(dtype=wp.float32),
    statistics_count: wp.array(dtype=wp.int64),
    min_count: wp.int64,
    step_counter: wp.array(dtype=wp.int32),
    std: wp.float32,
    epsilon: wp.float32,
    slot_seed: wp.int32,
    width: wp.int32,
):
    env_idx, feature_idx = wp.tid()
    if statistics_count[0] <= min_count:
        return
    state = _noise_state(slot_seed, step_counter[0], env_idx, feature_idx, width)
    scale = std * (wp.sqrt(feature_variance[feature_idx]) + epsilon)
    observations[env_idx, feature_idx] = observations[
        env_idx, feature_idx
    ] + scale * wp.randn(state)


@wp.kernel(enable_backward=False)
def _add_observation_noise_f64_kernel(
    observations: wp.array2d(dtype=wp.float32),
    feature_variance: wp.array(dtype=wp.float64),
    statistics_count: wp.array(dtype=wp.float64),
    min_count: wp.float64,
    step_counter: wp.array(dtype=wp.int32),
    std: wp.float32,
    epsilon: wp.float32,
    slot_seed: wp.int32,
    width: wp.int32,
):
    env_idx, feature_idx = wp.tid()
    if statistics_count[0] <= min_count:
        return
    state = _noise_state(slot_seed, step_counter[0], env_idx, feature_idx, width)
    # RunningStandardScaler casts its float64 variance to float32 before sqrt.
    variance = wp.float32(feature_variance[feature_idx])
    scale = std * (wp.sqrt(variance) + epsilon)
    observations[env_idx, feature_idx] = observations[
        env_idx, feature_idx
    ] + scale * wp.randn(state)


# Statistics-count dtype each variance dtype pairs with: RslRlEmpiricalNormalization
# keeps float32 moments and an int64 count, RunningStandardScaler float64 both.
_COUNT_DTYPE = {torch.float32: torch.int64, torch.float64: torch.float64}


class ObservationNoiseRuntime:
    """Persistent launch state for one environment's policy-observation noise."""

    def __init__(
        self,
        *,
        settings: ObservationNoiseSettings,
        num_envs: int,
        width: int,
        step_counter: torch.Tensor,
        seed: int | None,
        feature_variance: torch.Tensor | None = None,
        epsilon: float = 0.0,
        statistics_count: torch.Tensor | None = None,
        min_count: float = 0.0,
    ) -> None:
        """
        Args:
            feature_variance: Per-feature variance read on every launch (by
                reference); `None` means raw units.
            epsilon: Added to the standard deviation, as the normaliser does.
            statistics_count: Sample count of the statistics, read on every
                launch. While ``statistics_count[0] <= min_count`` the
                statistics hold no data yet and nothing is added, so a fresh
                normaliser's first update sees clean observations. `None`
                applies the noise unconditionally.
        """
        if not settings.active:
            raise ValueError("observation noise runtime requires an active setting")
        self.settings = settings
        self.num_envs = int(num_envs)
        self.width = int(width)
        if self.num_envs <= 0 or self.width <= 0:
            raise ValueError(
                "observation noise needs a positive env count and feature width, got "
                f"num_envs={self.num_envs}, width={self.width}"
            )
        if self.num_envs * self.width >= _INT32_LIMIT:
            raise ValueError(
                "observation noise RNG offsets must fit int32: "
                f"num_envs * width = {self.num_envs * self.width}"
            )
        if (
            not isinstance(step_counter, torch.Tensor)
            or step_counter.dtype != torch.int32
            or step_counter.numel() < 1
        ):
            raise TypeError("observation noise step counter must be an int32 tensor")
        self.device = step_counter.device
        self.step_counter = step_counter
        if feature_variance is None:
            # Raw units: unit denominators, allocated once.
            feature_variance = torch.ones(
                self.width, dtype=torch.float32, device=self.device
            )
            epsilon = 0.0
        if not isinstance(feature_variance, torch.Tensor):
            raise TypeError("observation noise feature_variance must be a tensor")
        if feature_variance.ndim != 1 or feature_variance.numel() < self.width:
            raise ValueError(
                "observation noise feature_variance must be one-dimensional with at "
                f"least {self.width} features, got shape {tuple(feature_variance.shape)}"
            )
        if feature_variance.device != self.device:
            raise ValueError(
                "observation noise feature_variance must live on the env device "
                f"{self.device}, got {feature_variance.device}"
            )
        if feature_variance.dtype == torch.float32:
            self._kernel = _add_observation_noise_f32_kernel
        elif feature_variance.dtype == torch.float64:
            self._kernel = _add_observation_noise_f64_kernel
        else:
            raise TypeError(
                "observation noise feature_variance must be float32 or float64, "
                f"got {feature_variance.dtype}"
            )
        count_dtype = _COUNT_DTYPE[feature_variance.dtype]
        if statistics_count is None:
            # Ungated: a persistent count that always exceeds min_count.
            statistics_count = torch.ones(1, dtype=count_dtype, device=self.device)
            min_count = 0.0
        if (
            not isinstance(statistics_count, torch.Tensor)
            or statistics_count.dtype != count_dtype
            or statistics_count.numel() != 1
            or statistics_count.device != self.device
        ):
            raise TypeError(
                f"observation noise statistics_count for {feature_variance.dtype} "
                f"variance must be a one-element {count_dtype} tensor on "
                f"{self.device}"
            )
        min_count = float(min_count)
        if not math.isfinite(min_count):
            raise ValueError(
                f"observation noise min_count must be finite, got {min_count}"
            )
        epsilon = float(epsilon)
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError(f"observation noise epsilon must be >= 0, got {epsilon}")
        # Held by reference: normaliser updates are copied into this storage in
        # place, so captured launches keep reading the live statistics.
        self.feature_variance = feature_variance
        self.epsilon = epsilon
        self.statistics_count = statistics_count.reshape(1)
        self.min_count = (
            int(min_count) if count_dtype == torch.int64 else float(min_count)
        )
        self.slot_seeds = tuple(
            observation_noise_slot_seed(seed, slot) for slot in _OBSERVATION_NOISE_SLOTS
        )
        # Compile before any step graph capture can record the first launch.
        wp.load_module(module=__name__, device=str(self.device))

    def apply(self, policy: torch.Tensor, *, slot: int) -> None:
        """Add one draw to ``policy[:, :width]`` in place."""

        if slot not in _OBSERVATION_NOISE_SLOTS:
            raise ValueError(f"unknown observation noise slot {slot}")
        if (
            not isinstance(policy, torch.Tensor)
            or policy.ndim != 2
            or policy.shape[0] != self.num_envs
            or policy.shape[1] < self.width
        ):
            shape = tuple(policy.shape) if isinstance(policy, torch.Tensor) else None
            raise ValueError(
                "observation noise expects a policy tensor of shape "
                f"({self.num_envs}, >= {self.width}), got {shape}"
            )
        if policy.dtype != torch.float32 or policy.device != self.device:
            raise ValueError(
                "observation noise expects a float32 policy tensor on "
                f"{self.device}, got {policy.dtype} on {policy.device}"
            )
        wp.launch(
            self._kernel,
            dim=(self.num_envs, self.width),
            inputs=[
                policy,
                self.feature_variance,
                self.statistics_count,
                self.min_count,
                self.step_counter,
                float(self.settings.std),
                self.epsilon,
                int(self.slot_seeds[slot]),
                self.width,
            ],
            device=str(self.device),
        )


__all__ = (
    "OBSERVATION_NOISE_CONFIG_KEY",
    "OBSERVATION_NOISE_SLOT_RESET",
    "OBSERVATION_NOISE_SLOT_STEP",
    "OBSERVATION_NOISE_SLOT_TERMINAL",
    "OBSERVATION_NOISE_UNITS",
    "ObservationNoiseRuntime",
    "ObservationNoiseSettings",
    "observation_noise_slot_seed",
    "parse_observation_noise_settings",
)
