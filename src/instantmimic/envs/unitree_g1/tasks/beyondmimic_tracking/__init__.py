"""BeyondMimic Stage-1 tracking task for the 29-DoF Unitree G1."""

import gymnasium as gym

from .reference import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
    TRACKED_BODY_COUNT,
)

gym.register(
    id="Unitree-G1-BeyondMimic-Tracking-v0",
    entry_point=(
        "instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking."
        "beyondmimic_tracking_env:BeyondMimicTrackingEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking."
            "beyondmimic_tracking_env_cfg:BeyondMimicTrackingEnvCfg"
        )
    },
)

__all__ = [
    "ACTION_DIM",
    "ACTOR_OBSERVATION_DIM",
    "CRITIC_OBSERVATION_DIM",
    "TRACKED_BODY_COUNT",
]
