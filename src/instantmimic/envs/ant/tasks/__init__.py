"""Ant task registration and runtime implementation.

The registered id matches Isaac Lab's so the two can be compared directly. It
names the benchmark being reproduced, not an affiliation; see
THIRD_PARTY_NOTICES.md.
"""

import gymnasium as gym

gym.register(
    id="Isaac-Ant-Direct-v0",
    entry_point=f"{__name__}.ant_env:AntEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_env_cfg:AntEnvCfg",
    },
)

__all__ = ["ant_env", "ant_env_cfg", "ant_math", "ant_constants"]
