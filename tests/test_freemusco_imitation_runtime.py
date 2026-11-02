from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest
import torch

from instantmimic.envs.freemusco.tasks.freemusco_imitation import (
    runtime,
)
from instantmimic.envs.freemusco.tasks.freemusco_imitation_env import (
    FreemuscoImitationEnv,
    resolve_imitation_runtime,
)


def test_imitation_species_profiles_resolve_to_one_generic_runtime() -> None:
    initial_species = runtime.species

    chimanoid = resolve_imitation_runtime(SimpleNamespace(species="chimanoid"))
    ostrich = resolve_imitation_runtime(SimpleNamespace(species="ostrich"))

    assert isinstance(chimanoid, ModuleType)
    assert chimanoid is runtime
    assert ostrich is runtime
    assert runtime.species == initial_species


@pytest.mark.parametrize(
    "species,mass",
    [("chimanoid", False), ("chimanoid", True), ("ostrich", False), ("ostrich", True)],
)
def test_imitation_fast_controller_steps_and_writes_observations(species, mass):
    from freemusco_runtime_fixtures import build_freemusco_imitation_cfg

    cfg = build_freemusco_imitation_cfg(species=species, mass=mass)
    env = FreemuscoImitationEnv(cfg)
    try:
        observations, _ = env.reset()
        geometry_dim = env.fast_muscle.muscle_geometry_observation_dim
        assert (geometry_dim > 0) is mass
        assert env._obs_suffix_dim == (
            env.fast_muscle.prev_action_obs_dim
            + env.fast_muscle.applied_torque_obs_dim
            + geometry_dim
        )
        before = env.common_step_counter.clone()
        observations, reward, terminated, timeout, _ = env.step(
            torch.zeros_like(env.actions)
        )
        assert env.common_step_counter == before + 1
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(reward).all()
        assert terminated.shape == timeout.shape == (env.num_envs,)
        if geometry_dim:
            torch.testing.assert_close(
                observations["policy"][:, -geometry_dim:],
                env.fast_muscle.get_muscle_geometry_observation(),
            )
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["velocity_heading", "point_goal"])
def test_command_mass_steps_with_ordinary_geometry_observations(mode):
    from dataclasses import fields

    from freemusco_runtime_fixtures import build_freemusco_imitation_cfg

    from instantmimic.envs.freemusco.tasks.freemusco_command_env import (
        FreemuscoCommandEnv,
    )
    from instantmimic.envs.freemusco.tasks.freemusco_command_env_cfg import (
        FreemuscoCommandEnvCfg,
    )

    base = build_freemusco_imitation_cfg(species="ostrich", mass=True)
    cfg = FreemuscoCommandEnvCfg(
        **{f.name: getattr(base, f.name) for f in fields(base) if f.init}
    )
    cfg.command_mode = mode
    env = FreemuscoCommandEnv(cfg)
    try:
        observations, _ = env.reset()
        assert torch.isfinite(observations["policy"]).all()
        observations, rewards, _, _, _ = env.step(torch.zeros_like(env.actions))
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(rewards).all()
        geometry = env.fast_muscle.get_muscle_geometry_observation()
        # Command features precede the shared observation suffix.
        assert geometry.shape[-1] > 0
        assert env._obs_suffix_dim >= geometry.shape[-1]
    finally:
        env.close()


@pytest.mark.cuda
@pytest.mark.parametrize("graph", [False, True])
def test_native_run_actuator_observations_remain_current(graph):
    import mujoco_warp as mjw
    import warp as wp
    from freemusco_runtime_fixtures import build_freemusco_run_cfg

    from instantmimic.envs.freemusco.tasks.freemusco_run_env import FreemuscoRunEnv

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wp.init()
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    warp_stream = wp.stream_from_torch(stream)
    with torch.cuda.stream(stream), wp.ScopedStream(warp_stream):
        cfg = build_freemusco_run_cfg(
            species="chimanoid", include_muscle_observation_blocks=True
        )
        cfg.scene.num_envs = 8
        cfg.decimation = 4
        cfg.use_cuda_graph = graph
        env = FreemuscoRunEnv(cfg)
        try:
            env.prepare_terminal_observation_buffer()
            env.reset(seed=0)
            action = torch.empty_like(env.actions)
            for step in range(12):
                if step in (4, 9):
                    env.episode_length_buf[::3] = env.max_episode_length - 1
                action.fill_(0.3 + 0.1 * (step % 3))
                observations, _, _, _, _ = env.step(action)
                assert torch.isfinite(observations["policy"]).all()
            env.episode_length_buf[::3] = env.max_episode_length - 1
            outputs, _ = env.step_with_io(env._default_step_io, reset_done=False)
            observations = outputs.observations
            # Probe only after the uninterrupted native trajectory.
            before = observations["policy"].clone()
            state = {
                name: wp.to_torch(getattr(env.mjw_data, name)).clone()
                for name in ("qpos", "qvel", "act", "actuator_force")
            }
            mjw.forward(env.mjw_model, env.mjw_data)
            for name, expected in state.items():
                torch.testing.assert_close(
                    wp.to_torch(getattr(env.mjw_data, name)),
                    expected,
                    rtol=2e-4,
                    atol=2e-4,
                )
            refreshed = env._get_observations()["policy"]
            torch.testing.assert_close(refreshed, before, rtol=2e-4, atol=2e-4)
            assert (env._step_graph is not None) is graph
        finally:
            env.close()
    current.wait_stream(stream)
