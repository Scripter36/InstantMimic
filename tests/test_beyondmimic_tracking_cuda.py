"""Numerical oracle for the real CUDA graph task path, with prepared local assets."""

from __future__ import annotations

import json

import pytest
import torch
import warp as wp

from instantmimic.envs.direct_rl_env_cfg import InteractiveSceneCfg
from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.beyondmimic_tracking_env import (
    BeyondMimicTrackingEnv,
)
from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.beyondmimic_tracking_env_cfg import (
    BeyondMimicTrackingEnvCfg,
    MotionCfg,
    ObservationCfg,
    RandomizationCfg,
    ResetCfg,
)


@pytest.fixture
def cuda_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wp.init()
    # Share the stream, as the training rollout does for Torch/Warp interop.
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        yield
    current.wait_stream(stream)


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.local_data
def test_cuda_graph_matches_reference_across_partial_resets(cuda_stream):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    def config(optimized):
        return BeyondMimicTrackingEnvCfg(
            scene=InteractiveSceneCfg(num_envs=4096),
            execution_mode="optimized" if optimized else "reference",
            use_cuda_graph=optimized,
            motion=MotionCfg(sampling_mode="start"),
            reset=ResetCfg(
                position_range=((0.0, 0.0),) * 3,
                euler_range=((0.0, 0.0),) * 3,
                velocity_range=((0.0, 0.0),) * 6,
                joint_position_range=(0.0, 0.0),
            ),
            randomization=RandomizationCfg(enabled=False, push_enabled=False),
            observations=ObservationCfg(corruption=False),
            seed=23,
        )

    reference = BeyondMimicTrackingEnv(config(False))
    optimized = BeyondMimicTrackingEnv(config(True))
    maxima = {}

    def compare(name, actual, expected, tolerance):
        difference = float((actual - expected).abs().max())
        maxima[name] = max(maxima.get(name, 0.0), difference)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.0)

    try:
        # Lift the reference into free flight: independent contact rollouts
        # can diverge even between two instances of the same implementation. Task formulas and
        # graph reset transitions use a deterministic fixture without contact.
        for env in (reference, optimized):
            env._motion.body_position_world[..., 2].add_(2.0)
        reference.reset()
        optimized.reset()
        for step in range(24):
            if step in (7, 15):
                for env in (reference, optimized):
                    env.episode_length_buf[::3] = env.max_episode_length - 1
            action = (
                (
                    0.05
                    * torch.sin(
                        torch.arange(29, device=reference.device) * 0.17 + step * 0.031
                    )
                )
                .expand(4096, -1)
                .contiguous()
            )
            expected = reference.step(action)
            torch.cuda.synchronize()
            actual = optimized.step(action)
            torch.cuda.synchronize()
            compare("qpos", optimized._qpos, reference._qpos, 1e-4)
            compare("qvel", optimized._qvel, reference._qvel, 1e-3)
            for name in ("policy", "critic"):
                compare(name, actual[0][name], expected[0][name], 1e-3)
            compare("reward", actual[1], expected[1], 1e-4)
            compare(
                "reward_terms", optimized._reward_terms, reference._reward_terms, 1e-4
            )
            torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
            torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)
            if step in (7, 15):
                assert bool(actual[3][::3].all())
            assert optimized._step_graph is not None
        print(
            json.dumps(
                {
                    "num_envs": 4096,
                    "steps": 24,
                    "partial_reset_steps": [7, 15],
                    "fixture": "reference lifted 2m; no ground contact",
                    "max_absolute_error": maxima,
                },
                sort_keys=True,
            )
        )
    finally:
        reference.close()
        optimized.close()


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.local_data
def test_cuda_graph_task_formulas_on_shared_contact_states(cuda_stream):
    from instantmimic.utils.torch_warp_capture import capture_warp_graph

    reference = BeyondMimicTrackingEnv(
        BeyondMimicTrackingEnvCfg(
            scene=InteractiveSceneCfg(num_envs=4096),
            randomization=RandomizationCfg(enabled=False, push_enabled=False),
            observations=ObservationCfg(corruption=False),
            seed=23,
        )
    )
    optimized = BeyondMimicTrackingEnv(
        BeyondMimicTrackingEnvCfg(
            scene=InteractiveSceneCfg(num_envs=4096),
            execution_mode="optimized",
            use_cuda_graph=True,
            randomization=RandomizationCfg(enabled=False, push_enabled=False),
            observations=ObservationCfg(corruption=False),
            seed=23,
        )
    )
    inputs = (
        "_qpos",
        "_qvel",
        "_xpos",
        "_xquat",
        "_cvel",
        "_subtree_com",
        "_sensordata",
        "_time_steps",
        "common_step_counter",
        "episode_length_buf",
        "actions",
        "_previous_actions",
        "_target_body_position",
        "_target_body_quaternion",
        "_collision_force_history",
        "_encoder_bias",
    )
    maxima = {}
    try:
        reference.reset()
        optimized.reset()
        action = torch.zeros((4096, 29), device=optimized.device)
        graph = None
        contacts_seen = 0
        for step in range(24):
            optimized.step(action)
            torch.cuda.synchronize()
            contacts_seen = max(contacts_seen, int(optimized.mjw_data.nacon.numpy()[0]))
            # An identical physical input isolates task math from independent
            # contact/solver trajectories; the oracle never advances physics.
            for name in inputs:
                getattr(reference, name).copy_(getattr(optimized, name))
            if graph is None:
                with capture_warp_graph(stream=wp.get_stream()) as capture:
                    optimized._get_observations()
                    optimized._get_rewards()
                    optimized._get_dones()
                graph = capture.graph
            wp.capture_launch(graph)
            expected_obs = reference._get_observations()
            expected_reward = reference._get_rewards()
            expected_done = reference._get_dones()
            for name, actual, expected, tolerance in (
                ("policy", optimized.obs_buf["policy"], expected_obs["policy"], 1e-3),
                ("critic", optimized.obs_buf["critic"], expected_obs["critic"], 1e-3),
                ("reward", optimized.reward_buf, expected_reward, 1e-4),
                (
                    "reward_terms",
                    optimized._reward_terms,
                    reference._reward_terms,
                    1e-4,
                ),
            ):
                maxima[name] = max(
                    maxima.get(name, 0.0), float((actual - expected).abs().max())
                )
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=0.0)
            torch.testing.assert_close(
                optimized.reset_terminated, expected_done[0], atol=0, rtol=0
            )
            torch.testing.assert_close(
                optimized.reset_time_outs, expected_done[1], atol=0, rtol=0
            )
        assert contacts_seen > 0
        assert optimized._step_graph is not None
        print(
            json.dumps(
                {
                    "fixture": "shared contact states",
                    "num_envs": 4096,
                    "steps": 24,
                    "contacts_seen": contacts_seen,
                    "max_absolute_error": maxima,
                },
                sort_keys=True,
            )
        )
    finally:
        reference.close()
        optimized.close()
