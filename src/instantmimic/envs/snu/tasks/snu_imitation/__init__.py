import gymnasium as gym

gym.register(
    id="SNU-Imitation-v0",
    entry_point=f"{__name__}.snu_imitation_env:SNUImitationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.snu_imitation_env_cfg:SNUImitationEnvCfg",
    },
)
