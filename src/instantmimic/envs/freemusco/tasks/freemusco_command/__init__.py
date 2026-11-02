import gymnasium as gym

gym.register(
    id="FreeMusco-OstrichCommand-FastMuscle-v0",
    entry_point="instantmimic.envs.freemusco.tasks.freemusco_command_env:FreemuscoCommandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "instantmimic.envs.freemusco.tasks.freemusco_command_env_cfg:FreemuscoCommandEnvCfg"
        ),
    },
)
