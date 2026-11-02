from __future__ import annotations

import pytest
import torch
import warp as wp

from instantmimic.envs.direct_rl_env_cfg import InteractiveSceneCfg
from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.benchmark_protocol import (
    EVALUATION_SAMPLING_MODE,
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
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
)


@pytest.fixture
def warp_cpu_device():
    with wp.ScopedDevice("cpu"):
        yield


def test_reference_config_has_50_hz_asymmetric_policy_contract() -> None:
    cfg = BeyondMimicTrackingEnvCfg(scene=InteractiveSceneCfg(num_envs=2))

    assert cfg.execution_mode == "reference"
    assert cfg.sim.hz / cfg.decimation == pytest.approx(50.0)
    assert cfg.action_space == ACTION_DIM
    assert cfg.observation_space == ACTOR_OBSERVATION_DIM
    assert cfg.state_space == CRITIC_OBSERVATION_DIM
    assert cfg.state_space != cfg.observation_space


@pytest.mark.integration
@pytest.mark.local_data
def test_reference_environment_resets_and_steps_on_cpu(
    monkeypatch,
    warp_cpu_device,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = BeyondMimicTrackingEnvCfg(
        scene=InteractiveSceneCfg(num_envs=2),
        seed=7,
    )
    environment = BeyondMimicTrackingEnv(cfg)
    try:
        assert environment.physics_dt == pytest.approx(1.0 / cfg.sim.hz)
        assert environment.step_dt == pytest.approx(1.0 / 50.0)
        assert environment.get_asymmetric_observation_dims() == (
            ACTOR_OBSERVATION_DIM,
            CRITIC_OBSERVATION_DIM,
        )
        observations, _ = environment.reset()
        assert observations["policy"].shape == (2, ACTOR_OBSERVATION_DIM)
        assert observations["critic"].shape == (2, CRITIC_OBSERVATION_DIM)
        assert environment.mjw_model.geom_friction.shape[0] == 2
        assert environment.mjw_model.body_ipos.shape[0] == 2
        environment._pre_physics_step(
            torch.zeros((2, ACTION_DIM), device=environment.device)
        )
        torch.testing.assert_close(
            environment._target_joint_position,
            environment._default_joint_position - environment._encoder_bias,
        )
        torch.testing.assert_close(
            environment._target_ctrl,
            environment._target_joint_position[:, environment._actuator_to_joint_order],
        )

        next_observations, reward, terminated, truncated, _ = environment.step(
            torch.zeros((2, ACTION_DIM), device=environment.device)
        )
        assert next_observations["policy"].shape == (2, ACTOR_OBSERVATION_DIM)
        assert next_observations["critic"].shape == (2, CRITIC_OBSERVATION_DIM)
        assert torch.isfinite(next_observations["policy"]).all()
        assert torch.isfinite(next_observations["critic"]).all()
        assert torch.isfinite(reward).all()
        assert terminated.dtype == torch.bool
        assert truncated.dtype == torch.bool
        term_sum = torch.stack(
            tuple(environment.get_reference_diagnostics().values()), dim=-1
        ).sum(dim=-1)
        torch.testing.assert_close(reward, term_sum)
    finally:
        environment.close()


@pytest.mark.integration
@pytest.mark.local_data
def test_eval_environment_preserves_startup_randomization_but_removes_push_and_noise(
    monkeypatch,
    warp_cpu_device,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = BeyondMimicTrackingEnvCfg(
        scene=InteractiveSceneCfg(num_envs=2),
        motion=MotionCfg(sampling_mode=EVALUATION_SAMPLING_MODE),
        reset=ResetCfg(
            position_range=((0.0, 0.0),) * 3,
            euler_range=((0.0, 0.0),) * 3,
            velocity_range=((0.0, 0.0),) * 6,
            joint_position_range=(0.0, 0.0),
        ),
        randomization=RandomizationCfg(
            enabled=True,
            push_enabled=False,
            foot_friction_range=(0.75, 0.75),
            encoder_bias_range=(0.01, 0.01),
            torso_com_offset_range=((0.01, 0.01),) * 3,
        ),
        observations=ObservationCfg(corruption=False),
        seed=7,
    )

    environment = BeyondMimicTrackingEnv(cfg)
    try:
        environment._time_steps.fill_(environment._motion.frame_count - 1)
        observations, _ = environment.reset()
        torch.testing.assert_close(
            environment._time_steps,
            torch.ones_like(environment._time_steps),
        )
        assert torch.count_nonzero(environment._push_time_left) == 0
        assert environment._observation_noise() is None
        torch.testing.assert_close(
            environment._encoder_bias,
            torch.full_like(environment._encoder_bias, 0.01),
        )
        friction = wp.to_torch(environment.mjw_model.geom_friction)
        torch.testing.assert_close(
            friction[:, environment._foot_geom_ids, 0],
            torch.full(
                (environment.num_envs, len(environment._foot_geom_ids)),
                0.75,
                device=environment.device,
                dtype=friction.dtype,
            ),
        )
        assert observations["policy"].shape == (2, ACTOR_OBSERVATION_DIM)
    finally:
        environment.close()
