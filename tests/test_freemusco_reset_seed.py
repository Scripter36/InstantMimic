"""FreeMusco reset sampling follows the public seed and RNG lifecycle."""

import mujoco
import pytest
import torch
import warp as wp
from freemusco_runtime_fixtures import (
    build_freemusco_imitation_cfg,
    build_freemusco_run_cfg,
)

from instantmimic.envs.freemusco.tasks.freemusco_imitation_env import (
    FreemuscoImitationEnv,
)
from instantmimic.envs.freemusco.tasks.freemusco_run_env import FreemuscoRunEnv
from instantmimic.utils.project_paths import resolve_project_path


def _environment(family, seed):
    if family == "run":
        cfg = build_freemusco_run_cfg(
            species="ostrich", include_muscle_observation_blocks=False
        )
        cfg.action = {"type": "mujoco_activation", "space": 120}
        env_type = FreemuscoRunEnv
    else:
        cfg = build_freemusco_imitation_cfg(species="ostrich")
        model = mujoco.MjModel.from_xml_path(
            str(resolve_project_path(cfg.sim.mjcf_path))
        )
        cfg.action = {"type": "torque", "space": model.nv - 6, "scale": 1.0}
        cfg.random_init = True
        env_type = FreemuscoImitationEnv
    cfg.seed = seed
    cfg.scene.num_envs = 4
    return env_type(cfg)


def _reset_qpos(env, seed=None):
    env.reset(seed=seed)
    return wp.to_torch(env.mjw_data.qpos).clone()


@pytest.mark.parametrize("family", ["run", "imitation"])
def test_reset_reseeds_explicitly_and_advances_without_seed(family):
    env = _environment(family, 123)
    try:
        first = _reset_qpos(env, 11)
        next_draw = _reset_qpos(env)
        different_seed = _reset_qpos(env, 29)
        assert not torch.equal(first, different_seed)
        assert not torch.equal(first, next_draw)
        torch.testing.assert_close(_reset_qpos(env, 11), first, rtol=0, atol=0)
        torch.testing.assert_close(_reset_qpos(env), next_draw, rtol=0, atol=0)
        # Physics time is not the random stream: explicit reseeding reproduces
        # the reset even after advancing the policy-step counter.
        env.common_step_counter.add_(17)
        torch.testing.assert_close(_reset_qpos(env, 11), first, rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.parametrize("family", ["run", "imitation"])
def test_constructor_seed_controls_reset_sequence(family):
    sequences = []
    for seed in (11, 29, 11):
        env = _environment(family, seed)
        try:
            sequences.append(torch.stack([_reset_qpos(env), _reset_qpos(env)]))
        finally:
            env.close()
    assert not torch.equal(sequences[0], sequences[1])
    torch.testing.assert_close(sequences[0], sequences[2], rtol=0, atol=0)


@pytest.mark.parametrize("family", ["run", "imitation"])
def test_reset_checkpoint_restores_next_draw_without_aliasing(family):
    env = _environment(family, 11)
    try:
        _reset_qpos(env)
        state = env.get_checkpoint_state()
        saved_seed = state["freemusco_reset"]["seed"].clone()
        expected = _reset_qpos(env)
        torch.testing.assert_close(state["freemusco_reset"]["seed"], saved_seed)
        assert env.load_checkpoint_state(state)
        torch.testing.assert_close(_reset_qpos(env), expected, rtol=0, atol=0)
        state["freemusco_reset"]["seed"] = torch.zeros(2, dtype=torch.int32)
        env.common_step_counter.fill_(17)
        with pytest.raises(ValueError, match="seed"):
            env.load_checkpoint_state(state)
        assert env.common_step_counter.item() == 17
    finally:
        env.close()


@pytest.mark.cuda
@pytest.mark.parametrize("family", ["run", "imitation"])
def test_reset_seed_advances_on_warp_graph_replay_and_reseeds(family):
    from instantmimic.utils.torch_warp_capture import capture_warp_graph

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    warp_stream = wp.stream_from_torch(stream)
    with torch.cuda.stream(stream), wp.ScopedStream(warp_stream):
        env = _environment(family, 11)
        try:
            first = _reset_qpos(env, 11)
            second = _reset_qpos(env)
            output = torch.empty_like(env.obs_buf["policy"])
            # Warm every native reset/prefix path before capture.
            env.reset_into(output)
            env.seed(11)
            with capture_warp_graph(stream=warp_stream) as capture:
                env.reset_into(output)
            wp.capture_launch(capture.graph, stream=warp_stream)
            torch.testing.assert_close(
                wp.to_torch(env.mjw_data.qpos), first, rtol=0, atol=0
            )
            wp.capture_launch(capture.graph, stream=warp_stream)
            torch.testing.assert_close(
                wp.to_torch(env.mjw_data.qpos), second, rtol=0, atol=0
            )
            env.seed(11)
            wp.capture_launch(capture.graph, stream=warp_stream)
            torch.testing.assert_close(
                wp.to_torch(env.mjw_data.qpos), first, rtol=0, atol=0
            )
        finally:
            env.close()
    current.wait_stream(stream)
