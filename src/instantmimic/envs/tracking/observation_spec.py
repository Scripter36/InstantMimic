"""Partition task observations into deployable and privileged inputs."""

from typing import Any


def build_deployable_obs_spec(
    *,
    observation_layout: Any,
    obs_main_dim: int,
    include_phase_in_policy_obs: bool = True,
    phase_ofs: int | None = None,
    phase_dim: int = 0,
) -> dict[str, Any]:
    root_target_dim = int(
        getattr(observation_layout, "root_target_dim", observation_layout.joint_pos_ofs)
    )
    future_obs_ofs = int(observation_layout.future_obs_ofs)
    future_obs_dim = int(observation_layout.future_obs_dim)
    privileged_obs_indices = list(range(root_target_dim))
    if future_obs_dim > 0:
        privileged_obs_indices.extend(
            list(range(future_obs_ofs, future_obs_ofs + future_obs_dim))
        )
    if not include_phase_in_policy_obs and phase_ofs is not None and phase_dim > 0:
        privileged_obs_indices.extend(
            list(range(int(phase_ofs), int(phase_ofs + phase_dim)))
        )
    privileged_obs_set = set(privileged_obs_indices)
    policy_obs_indices = [
        idx for idx in range(int(obs_main_dim)) if idx not in privileged_obs_set
    ]
    return {
        "policy_obs_indices": policy_obs_indices,
        "privileged_obs_indices": privileged_obs_indices,
        "policy_obs_dim": len(policy_obs_indices),
        "privileged_obs_dim": len(privileged_obs_indices),
        "root_target_dim": root_target_dim,
        "future_obs_ofs": future_obs_ofs,
        "future_obs_dim": future_obs_dim,
        "obs_main_dim": int(obs_main_dim),
    }
