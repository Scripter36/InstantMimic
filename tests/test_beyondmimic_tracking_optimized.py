from __future__ import annotations

import pytest
import torch
import warp as wp

from instantmimic.envs.direct_rl_env_cfg import InteractiveSceneCfg
from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking import (
    optimized_runtime,
)
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
from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.reference import (
    ACTION_DIM,
)


@pytest.fixture(autouse=True)
def _warp_cpu_device():
    with wp.ScopedDevice("cpu"):
        yield


def _cpu_reference_environment(monkeypatch) -> BeyondMimicTrackingEnv:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = BeyondMimicTrackingEnvCfg(
        scene=InteractiveSceneCfg(num_envs=3),
        observations=ObservationCfg(corruption=False),
        seed=13,
    )
    environment = BeyondMimicTrackingEnv(cfg)
    environment.reset()
    return environment


def _deterministic_cfg() -> BeyondMimicTrackingEnvCfg:
    return BeyondMimicTrackingEnvCfg(
        scene=InteractiveSceneCfg(num_envs=1),
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


@pytest.mark.integration
@pytest.mark.local_data
def test_optimized_runtime_state_owns_views_without_replacing_source_buffers(
    monkeypatch,
) -> None:
    environment = _cpu_reference_environment(monkeypatch)
    try:
        source_actions = environment.actions
        source_qpos = environment._qpos

        environment._optimized_task_path = True
        optimized_runtime.initialize(environment)

        state = environment.optimized_runtime_state
        assert state is not None
        assert state.views.actions.ptr == source_actions.data_ptr()
        assert state.views.qpos.ptr == source_qpos.data_ptr()
        assert not hasattr(environment, "_optimized_actions_wp")
        assert not hasattr(environment, "_optimized_qpos_wp")
    finally:
        environment.close()


@pytest.mark.integration
@pytest.mark.local_data
def test_optimized_warp_action_observation_reward_and_done_match_reference(
    monkeypatch,
) -> None:
    environment = _cpu_reference_environment(monkeypatch)
    try:
        incoming_action = torch.linspace(
            -0.25,
            0.25,
            environment.num_envs * ACTION_DIM,
            device=environment.device,
        ).reshape(environment.num_envs, ACTION_DIM)
        environment._pre_physics_step(incoming_action)
        environment._apply_action()
        reference_action = environment.actions.clone()
        reference_previous_action = environment._previous_actions.clone()
        reference_target_joint_position = environment._target_joint_position.clone()
        reference_target_ctrl = environment._target_ctrl.clone()
        reference_ctrl = environment._ctrl.clone()

        environment.actions.zero_()
        environment._previous_actions.zero_()
        environment._target_joint_position.zero_()
        environment._target_ctrl.zero_()
        environment._ctrl.zero_()
        environment._optimized_task_path = True
        optimized_runtime.initialize(environment)
        environment._pre_physics_step(incoming_action)
        torch.testing.assert_close(environment.actions, reference_action)
        torch.testing.assert_close(
            environment._previous_actions,
            reference_previous_action,
        )
        torch.testing.assert_close(
            environment._target_joint_position,
            reference_target_joint_position,
        )
        torch.testing.assert_close(environment._target_ctrl, reference_target_ctrl)
        torch.testing.assert_close(environment._ctrl, reference_ctrl)

        environment._optimized_task_path = False
        reference_observations = {
            name: value.clone()
            for name, value in environment._get_observations().items()
        }
        reference_reward = environment._get_rewards().clone()
        reference_terms = environment._reward_terms.clone()
        reference_terminated, reference_truncated = environment._get_dones()
        reference_terminated = reference_terminated.clone()
        reference_truncated = reference_truncated.clone()

        environment._optimized_task_path = True
        optimized_observations = environment._get_observations()
        optimized_reward = environment._get_rewards()
        optimized_terminated, optimized_truncated = environment._get_dones()
        torch.testing.assert_close(
            optimized_observations["policy"],
            reference_observations["policy"],
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            optimized_observations["critic"],
            reference_observations["critic"],
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            optimized_reward,
            reference_reward,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(
            environment._reward_terms,
            reference_terms,
            atol=2.0e-5,
            rtol=2.0e-5,
        )
        torch.testing.assert_close(optimized_terminated, reference_terminated)
        torch.testing.assert_close(optimized_truncated, reference_truncated)
    finally:
        environment.close()


@pytest.mark.integration
@pytest.mark.local_data
def test_optimized_masked_reset_preserves_unselected_worlds(monkeypatch) -> None:
    environment = _cpu_reference_environment(monkeypatch)
    try:
        environment._optimized_task_path = True
        optimized_runtime.initialize(environment)
        qpos_before = environment._qpos.clone()
        qvel_before = environment._qvel.clone()
        time_before = environment._time_steps.clone()
        reset_mask = torch.tensor(
            [True, False, False],
            device=environment.device,
            dtype=torch.bool,
        )

        environment._reset_idx(reset_mask)
        environment._finalize_reset_state(reset_mask)

        torch.testing.assert_close(environment._qpos[1:], qpos_before[1:])
        torch.testing.assert_close(environment._qvel[1:], qvel_before[1:])
        torch.testing.assert_close(
            environment._time_steps[1:],
            time_before[1:] + 1,
        )
        assert torch.isfinite(environment._qpos).all()
        assert torch.isfinite(environment._qvel).all()
        assert int(environment._time_steps[0]) < environment._motion.frame_count
    finally:
        environment.close()


@pytest.mark.integration
@pytest.mark.local_data
def test_episode_reset_preserves_existing_push_timers_on_both_paths(
    monkeypatch,
) -> None:
    """Resetting an episode must not restart its interval push schedule."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reset_mask = torch.tensor([True, False, True], dtype=torch.bool)
    initial_timers = torch.tensor([2.25, 2.5, 2.75], dtype=torch.float32)

    for optimized in (False, True):
        environment = _cpu_reference_environment(monkeypatch)
        try:
            if optimized:
                environment._optimized_task_path = True
                optimized_runtime.initialize(environment)
            environment._push_time_left.copy_(initial_timers)
            environment._reset_idx(reset_mask)
            environment._finalize_reset_state(reset_mask)

            # _reset_idx also applies the normal interval event once, so every
            # world's timer advances by one environment step. The reset itself
            # must not replace the selected worlds' timers with new samples.
            expected = initial_timers - environment.step_dt
            torch.testing.assert_close(environment._push_time_left, expected)

            # An ordinary interval event still owns re-sampling: force one
            # world to expire and verify only that world's timer is refreshed.
            environment._push_time_left[0] = 0.0
            if optimized:
                optimized_runtime.apply_interval_pushes(environment)
            else:
                environment._apply_interval_pushes()
            assert 1.0 <= float(environment._push_time_left[0]) <= 3.0
            torch.testing.assert_close(
                environment._push_time_left[1],
                initial_timers[1] - 2.0 * environment.step_dt,
            )
        finally:
            environment.close()


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.integration
@pytest.mark.local_data
def test_first_reset_suppresses_pushes_on_both_paths(monkeypatch, optimized) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    environment = BeyondMimicTrackingEnv(
        BeyondMimicTrackingEnvCfg(
            scene=InteractiveSceneCfg(num_envs=3),
            observations=ObservationCfg(corruption=False),
            seed=13,
        )
    )
    try:
        if optimized:
            environment._optimized_task_path = True
            optimized_runtime.initialize(environment)
        environment._push_time_left.zero_()
        assert environment._initial_reset_complete is False

        environment.reset()

        assert environment._initial_reset_complete is True
        torch.testing.assert_close(
            environment._push_time_left,
            torch.zeros_like(environment._push_time_left),
        )

        environment._push_time_left.zero_()
        environment.reset()
        assert bool((environment._push_time_left >= 1.0).all())
    finally:
        environment.close()


@pytest.mark.integration
@pytest.mark.local_data
def test_disabled_pushes_do_not_initialize_interval_timers(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = BeyondMimicTrackingEnvCfg(
        scene=InteractiveSceneCfg(num_envs=2),
        randomization=RandomizationCfg(enabled=True, push_enabled=False),
        observations=ObservationCfg(corruption=False),
        seed=13,
    )
    environment = BeyondMimicTrackingEnv(cfg)
    try:
        environment.reset()
        torch.testing.assert_close(
            environment._push_time_left,
            torch.zeros_like(environment._push_time_left),
        )
    finally:
        environment.close()


@pytest.mark.integration
@pytest.mark.local_data
def test_optimized_task_hot_path_matches_reference_rollout(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reference = BeyondMimicTrackingEnv(_deterministic_cfg())
    optimized = BeyondMimicTrackingEnv(_deterministic_cfg())
    try:
        reference.reset()
        optimized._optimized_task_path = True
        optimized_runtime.initialize(optimized)
        optimized.reset()

        for step in range(100):
            phase = torch.tensor(float(step), device=reference.device)
            action = (
                0.2
                * torch.sin(
                    torch.arange(ACTION_DIM, device=reference.device) * 0.17
                    + phase * 0.031
                )
            ).reshape(1, ACTION_DIM)
            reference_output = reference.step(action)
            optimized_output = optimized.step(action)
            if step not in {0, 9, 99}:
                continue
            for name in ("policy", "critic"):
                torch.testing.assert_close(
                    optimized_output[0][name],
                    reference_output[0][name],
                    atol=8.0e-5,
                    rtol=2.0e-5,
                )
            torch.testing.assert_close(
                optimized_output[1],
                reference_output[1],
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            torch.testing.assert_close(
                optimized._reward_terms,
                reference._reward_terms,
                atol=2.0e-5,
                rtol=2.0e-5,
            )
            torch.testing.assert_close(
                optimized_output[2],
                reference_output[2],
            )
            torch.testing.assert_close(
                optimized_output[3],
                reference_output[3],
            )
            torch.testing.assert_close(
                optimized._qpos,
                reference._qpos,
                atol=2.0e-6,
                rtol=2.0e-6,
            )
            torch.testing.assert_close(
                optimized._qvel,
                reference._qvel,
                atol=8.0e-5,
                rtol=2.0e-5,
            )
    finally:
        reference.close()
        optimized.close()


@pytest.mark.integration
@pytest.mark.local_data
def test_optimized_adaptive_failure_update_preserves_mjlab_overwrite_semantics(
    monkeypatch,
) -> None:
    environment = _cpu_reference_environment(monkeypatch)
    try:
        environment._optimized_task_path = True
        optimized_runtime.initialize(environment)
        current = environment._adaptive.current_bin_failed
        current.copy_(torch.arange(current.numel(), dtype=current.dtype) + 10.0)
        sample_mask = torch.tensor([True, False, True], dtype=torch.bool)
        environment.reset_terminated.zero_()

        optimized_runtime._accumulate_failures(environment, sample_mask)
        torch.testing.assert_close(
            current,
            torch.arange(current.numel(), dtype=current.dtype) + 10.0,
        )

        environment._time_steps.copy_(
            torch.tensor([0, 80, environment._motion.frame_count - 1])
        )
        environment.reset_terminated.copy_(
            torch.tensor([True, False, True], dtype=torch.bool)
        )
        optimized_runtime._accumulate_failures(environment, sample_mask)
        expected = torch.zeros_like(current)
        expected[0] = 1.0
        expected[-1] = 1.0
        torch.testing.assert_close(current, expected)
    finally:
        environment.close()


@pytest.mark.integration
@pytest.mark.local_data
def test_optimized_motion_wrap_preserves_episode_local_state(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    environment = BeyondMimicTrackingEnv(_deterministic_cfg())
    try:
        environment._optimized_task_path = True
        optimized_runtime.initialize(environment)
        environment.reset()
        environment._time_steps.fill_(environment._motion.frame_count - 1)
        environment.actions.fill_(0.11)
        environment._previous_actions.fill_(-0.12)
        environment._target_joint_position.fill_(0.13)
        environment._target_ctrl.fill_(-0.14)
        environment._collision_force_history.fill_(0.15)
        expected = {
            "actions": environment.actions.clone(),
            "previous": environment._previous_actions.clone(),
            "target_joint": environment._target_joint_position.clone(),
            "target_ctrl": environment._target_ctrl.clone(),
            "collision": environment._collision_force_history.clone(),
        }

        reset_mask = torch.zeros(1, dtype=torch.bool)
        environment._reset_idx(reset_mask)
        environment._finalize_reset_state(reset_mask)

        assert int(environment._time_steps[0]) == 0
        torch.testing.assert_close(environment.actions, expected["actions"])
        torch.testing.assert_close(
            environment._previous_actions,
            expected["previous"],
        )
        torch.testing.assert_close(
            environment._target_joint_position,
            expected["target_joint"],
        )
        torch.testing.assert_close(
            environment._target_ctrl,
            expected["target_ctrl"],
        )
        torch.testing.assert_close(
            environment._collision_force_history,
            expected["collision"],
        )
    finally:
        environment.close()
