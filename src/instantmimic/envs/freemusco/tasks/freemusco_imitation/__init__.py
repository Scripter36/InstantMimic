import gymnasium as gym

_IMITATION_ENTRY_POINT = (
    "instantmimic.envs.freemusco.tasks.freemusco_imitation_env:FreemuscoImitationEnv"
)
_IMITATION_CFG_ENTRY_POINT = "instantmimic.envs.freemusco.tasks.freemusco_imitation_env_cfg:FreemuscoImitationEnvCfg"

for _task_id in (
    "FreeMusco-OstrichImitation-FastMuscle-v0",
    "FreeMusco-ChimanoidImitation-FastMuscle-v0",
):
    gym.register(
        id=_task_id,
        entry_point=_IMITATION_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _IMITATION_CFG_ENTRY_POINT},
    )
