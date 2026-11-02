import gymnasium as gym

gym.register(
    id="SNU-Command-v0",
    entry_point=f"{__name__}.snu_command_env:SNUCommandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.snu_command_env_cfg:SNUCommandEnvCfg",
    },
)
