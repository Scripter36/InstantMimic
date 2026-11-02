import gymnasium as gym

gym.register(
    id="SNU-Run-v0",
    entry_point=f"{__name__}.snu_run_env:SNURunEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.snu_run_env_cfg:SNURunEnvCfg",
    },
)
