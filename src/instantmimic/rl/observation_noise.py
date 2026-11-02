"""Bind the training env's observation noise to the agent's normaliser."""

from __future__ import annotations

from logging import getLogger
from typing import Any

import torch

from instantmimic.rl.normalization import (
    RslRlEmpiricalNormalization,
    RunningStandardScaler,
)

_LOGGER = getLogger(__name__)


def _unwrap(env: Any) -> Any:
    return env.unwrapped if hasattr(env, "unwrapped") else env


def bind_training_observation_noise(*, agent: Any, env: Any) -> bool:
    """Enable `env.observation_noise` on the training rollout environment.

    Call once on the env the training loop steps, after the agent and any
    checkpoint it loads exist and before the first reset. Every other env
    instance (benchmark, playback, video) is never bound and stays clean.

    ``units: normalized`` reads the agent's ``obs_scaler`` statistics by
    reference, so the noise follows the normaliser as it updates (or stays
    fixed where its statistics are frozen). While a normaliser that updates
    still holds its initial sample count, no noise is added, so its first
    statistics come from clean observations; loaded statistics open the gate
    at once. With a disabled (identity) normaliser the policy input is the raw
    observation, and the noise is applied in raw units.

    Returns:
        Whether the environment now adds noise.
    """

    env_raw = _unwrap(env)
    settings = getattr(env_raw, "observation_noise_settings", None)
    if settings is None or not settings.active:
        return False
    enable = getattr(env_raw, "enable_observation_noise", None)
    if not callable(enable):
        raise TypeError(
            f"{type(env_raw).__name__} declares env.observation_noise but cannot "
            "enable it"
        )
    if settings.units == "raw":
        return bool(enable())
    scaler = getattr(agent, "obs_scaler", None)
    variance = getattr(scaler, "running_variance", None)
    if not isinstance(variance, torch.Tensor):
        raise ValueError(
            "env.observation_noise.units=normalized needs an agent observation "
            f"normaliser with running_variance; {type(agent).__name__} has none. "
            "Use env.observation_noise.units=raw."
        )
    if not bool(getattr(scaler, "enabled", True)):
        _LOGGER.info(
            "Observation normaliser is disabled; observation noise is in raw units."
        )
        return bool(
            enable(
                feature_variance=torch.ones(
                    variance.shape, dtype=torch.float32, device=variance.device
                ),
                epsilon=0.0,
            )
        )
    statistics_count, min_count = _statistics_gate(scaler)
    return bool(
        enable(
            feature_variance=variance,
            epsilon=float(getattr(scaler, "epsilon", 0.0)),
            statistics_count=statistics_count,
            min_count=min_count,
        )
    )


def _statistics_gate(scaler: Any) -> tuple[torch.Tensor | None, float]:
    """The count buffer and its initial value for a normaliser that updates.

    `None` for a normaliser whose statistics never change (a fully frozen
    prefix, ``max_samples <= 0``) or whose count convention is unknown: there
    is nothing to keep clean, and gating would leave the noise off forever.
    """

    if isinstance(scaler, RunningStandardScaler):
        width = int(scaler.running_variance.shape[0])
        max_samples = scaler.max_samples
        if int(scaler.frozen_prefix_dim) >= width or (
            max_samples is not None and max_samples <= 0
        ):
            return None, 0.0
        # Constructed and reset with count 1 before any sample.
        return scaler.current_count, 1.0
    if isinstance(scaler, RslRlEmpiricalNormalization):
        return scaler.current_count, 0.0
    return None, 0.0


__all__ = ("bind_training_observation_noise",)
