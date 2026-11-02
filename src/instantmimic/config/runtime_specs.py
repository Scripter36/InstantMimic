from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

# Preserve the existing execution-spec import while sharing its field declarations.
from instantmimic.config.runtime_config import RuntimeConfig as RuntimeExecutionSpec
from instantmimic.config.schema import SemanticConfig

__all__ = [
    "ExecutionContext",
    "RuntimeExecutionSpec",
    "mutable_environment_config",
    "resolve_seed",
]


def resolve_seed(
    requested: int,
    *,
    seed_factory: Callable[[], int] | None = None,
) -> int:
    """Turn a requested seed into the one a run will use.

    `-1` means "pick one". Playback used to pick with
    `random.randint(0, 10000)`, which drew from the global `random` state and
    from 10001 values, while training drew from `secrets` over the full 31-bit
    range -- so two runs that both asked for a random seed were reproducible to
    different degrees, and playback collided roughly 200000 times more often.
    Both go through here.
    """

    if requested != -1:
        return requested
    factory = seed_factory or (lambda: secrets.randbelow(2**31))
    return factory()


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Generated execution metadata kept separate from semantic configuration."""

    resolved_seed: int
    device: str
    run_name: str
    log_dir: Path
    logger_type: str
    timestamp: str
    transitions_per_iteration: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.resolved_seed, int)
            or isinstance(self.resolved_seed, bool)
            or self.resolved_seed < 0
        ):
            raise ValueError("resolved seed must be a nonnegative integer")
        for name in ("device", "run_name", "logger_type", "timestamp"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.log_dir, Path):
            raise TypeError("log_dir must be a pathlib.Path")
        if (
            not isinstance(self.transitions_per_iteration, int)
            or isinstance(self.transitions_per_iteration, bool)
            or self.transitions_per_iteration <= 0
        ):
            raise ValueError("transitions_per_iteration must be a positive integer")

    @classmethod
    def from_semantic_config(
        cls,
        config: SemanticConfig,
        *,
        device: str,
        run_name: str,
        log_dir: Path,
        timestamp: str,
        num_envs: int,
        seed_factory: Callable[[], int] | None = None,
    ) -> ExecutionContext:
        resolved_seed = resolve_seed(config.seed, seed_factory=seed_factory)
        logger = config.experiment.logger
        if logger is None:
            raise ValueError("experiment.logger is required")
        logger_type = logger.type
        rollouts = config.agent["rollouts"]
        if not isinstance(num_envs, int) or isinstance(num_envs, bool):
            raise TypeError("num_envs must be an integer")
        if num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(rollouts, int) or isinstance(rollouts, bool):
            raise TypeError("agent.rollouts must be an integer")
        if rollouts <= 0:
            raise ValueError("agent.rollouts must be a positive integer")
        return cls(
            resolved_seed=resolved_seed,
            device=device,
            run_name=run_name,
            log_dir=log_dir,
            logger_type=logger_type,
            timestamp=timestamp,
            transitions_per_iteration=num_envs * rollouts,
        )


def mutable_environment_config(
    config: SemanticConfig,
    *,
    seed: int,
    log_dir: str | Path,
) -> DictConfig:
    """Build the detached mutable config owned by one environment instance."""

    env_cfg = OmegaConf.create(config.to_dict()["env"])
    if not isinstance(env_cfg, DictConfig):
        raise TypeError("semantic env root must produce a DictConfig")
    with open_dict(env_cfg):
        env_cfg.seed = int(seed)
        env_cfg.log_dir = Path(log_dir).as_posix()
    return env_cfg
