import gymnasium as gym

gym.register(
    id="FreeMusco-Run-v0",
    entry_point="instantmimic.envs.freemusco.tasks.freemusco_run_env:FreemuscoRunEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "instantmimic.envs.freemusco.tasks.freemusco_run_env_cfg:FreemuscoRunEnvCfg"
        ),
    },
)
