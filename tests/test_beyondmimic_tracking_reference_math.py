from __future__ import annotations

import math

import torch

from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.reference import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
    TRACKED_BODY_COUNT,
    AdaptiveSamplerState,
    ObservationNoise,
    assemble_actor_observation,
    assemble_critic_observation,
    compute_reward,
    compute_terminations,
    default_relative_action_targets,
    failure_bins,
    motion_relative_to_robot_anchor,
    perturb_reference_state,
    sampling_probabilities,
    update_failure_ema,
)


def _filled(batch: int, width: int, value: float) -> torch.Tensor:
    return torch.full((batch, width), value, dtype=torch.float64)


def test_observation_contract_preserves_official_term_order_and_dimensions() -> None:
    batch = 2
    actor = assemble_actor_observation(
        command=_filled(batch, 58, 1.0),
        anchor_position=_filled(batch, 3, 2.0),
        anchor_orientation=_filled(batch, 6, 3.0),
        base_linear_velocity=_filled(batch, 3, 4.0),
        base_angular_velocity=_filled(batch, 3, 5.0),
        joint_position=_filled(batch, ACTION_DIM, 7.0),
        default_joint_position=_filled(batch, ACTION_DIM, 1.0),
        joint_velocity=_filled(batch, ACTION_DIM, 8.0),
        default_joint_velocity=_filled(batch, ACTION_DIM, 2.0),
        action=_filled(batch, ACTION_DIM, 9.0),
    )
    assert actor.shape == (batch, ACTOR_OBSERVATION_DIM)
    expected_actor = torch.cat(
        (
            _filled(batch, 58, 1.0),
            _filled(batch, 3, 2.0),
            _filled(batch, 6, 3.0),
            _filled(batch, 3, 4.0),
            _filled(batch, 3, 5.0),
            _filled(batch, ACTION_DIM, 6.0),
            _filled(batch, ACTION_DIM, 6.0),
            _filled(batch, ACTION_DIM, 9.0),
        ),
        dim=-1,
    )
    torch.testing.assert_close(actor, expected_actor)

    critic = assemble_critic_observation(
        command=_filled(batch, 58, 1.0),
        anchor_position=_filled(batch, 3, 2.0),
        anchor_orientation=_filled(batch, 6, 3.0),
        body_position=_filled(batch, TRACKED_BODY_COUNT * 3, 4.0),
        body_orientation=_filled(batch, TRACKED_BODY_COUNT * 6, 5.0),
        base_linear_velocity=_filled(batch, 3, 6.0),
        base_angular_velocity=_filled(batch, 3, 7.0),
        joint_position=_filled(batch, ACTION_DIM, 8.0),
        default_joint_position=_filled(batch, ACTION_DIM, 2.0),
        joint_velocity=_filled(batch, ACTION_DIM, 9.0),
        default_joint_velocity=_filled(batch, ACTION_DIM, 3.0),
        action=_filled(batch, ACTION_DIM, 10.0),
    )
    assert critic.shape == (batch, CRITIC_OBSERVATION_DIM)
    expected_critic = torch.cat(
        (
            _filled(batch, 58, 1.0),
            _filled(batch, 3, 2.0),
            _filled(batch, 6, 3.0),
            _filled(batch, TRACKED_BODY_COUNT * 3, 4.0),
            _filled(batch, TRACKED_BODY_COUNT * 6, 5.0),
            _filled(batch, 3, 6.0),
            _filled(batch, 3, 7.0),
            _filled(batch, ACTION_DIM, 6.0),
            _filled(batch, ACTION_DIM, 6.0),
            _filled(batch, ACTION_DIM, 10.0),
        ),
        dim=-1,
    )
    torch.testing.assert_close(critic, expected_critic)


def test_actor_noise_affects_only_official_noisy_terms() -> None:
    batch = 1
    zeros = lambda width: _filled(batch, width, 0.0)
    noise = ObservationNoise(
        anchor_position=_filled(batch, 3, 1.0),
        anchor_orientation=_filled(batch, 6, 2.0),
        base_linear_velocity=_filled(batch, 3, 3.0),
        base_angular_velocity=_filled(batch, 3, 4.0),
        joint_position=_filled(batch, ACTION_DIM, 5.0),
        joint_velocity=_filled(batch, ACTION_DIM, 6.0),
    )
    obs = assemble_actor_observation(
        command=zeros(58),
        anchor_position=zeros(3),
        anchor_orientation=zeros(6),
        base_linear_velocity=zeros(3),
        base_angular_velocity=zeros(3),
        joint_position=zeros(ACTION_DIM),
        default_joint_position=zeros(ACTION_DIM),
        joint_velocity=zeros(ACTION_DIM),
        default_joint_velocity=zeros(ACTION_DIM),
        action=zeros(ACTION_DIM),
        noise=noise,
    )
    expected = torch.cat(
        (
            zeros(58),
            _filled(batch, 3, 1.0),
            _filled(batch, 6, 2.0),
            _filled(batch, 3, 3.0),
            _filled(batch, 3, 4.0),
            _filled(batch, ACTION_DIM, 5.0),
            _filled(batch, ACTION_DIM, 6.0),
            zeros(ACTION_DIM),
        ),
        dim=-1,
    )
    torch.testing.assert_close(obs, expected)


def test_motion_reference_is_yaw_aligned_to_robot_anchor() -> None:
    half_sqrt = math.sqrt(0.5)
    robot_position = torch.tensor([[1.0, 2.0, 0.8]], dtype=torch.float64)
    robot_quaternion = torch.tensor(
        [[half_sqrt, 0.0, 0.0, half_sqrt]], dtype=torch.float64
    )
    reference_anchor_position = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    reference_anchor_quaternion = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    reference_body_position = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]], dtype=torch.float64
    )
    reference_body_quaternion = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )

    position, quaternion = motion_relative_to_robot_anchor(
        robot_anchor_position=robot_position,
        robot_anchor_quaternion=robot_quaternion,
        reference_anchor_position=reference_anchor_position,
        reference_anchor_quaternion=reference_anchor_quaternion,
        reference_body_position=reference_body_position,
        reference_body_quaternion=reference_body_quaternion,
    )

    torch.testing.assert_close(
        position,
        torch.tensor([[[1.0, 2.0, 1.0], [1.0, 3.0, 1.0]]], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    expected_quaternion = robot_quaternion[:, None, :].expand_as(quaternion)
    torch.testing.assert_close(quaternion, expected_quaternion, atol=1e-12, rtol=1e-12)


def test_default_relative_actions_include_encoder_bias() -> None:
    action = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    scale = torch.tensor([0.25, 0.5], dtype=torch.float64)
    default = torch.tensor([0.1, -0.2], dtype=torch.float64)
    bias = torch.tensor([[0.01, -0.03]], dtype=torch.float64)
    target = default_relative_action_targets(action, scale, default, bias)
    torch.testing.assert_close(
        target, torch.tensor([[0.59, -0.67]], dtype=torch.float64)
    )


def test_reward_exposes_raw_terms_and_applies_step_duration_once() -> None:
    batch = 1
    anchor_position = torch.zeros((batch, 3), dtype=torch.float64)
    anchor_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    body_position = torch.zeros((batch, TRACKED_BODY_COUNT, 3), dtype=torch.float64)
    body_quaternion = torch.zeros((batch, TRACKED_BODY_COUNT, 4), dtype=torch.float64)
    body_quaternion[..., 0] = 1.0
    body_velocity = torch.zeros((batch, TRACKED_BODY_COUNT, 3), dtype=torch.float64)
    action = torch.ones((batch, ACTION_DIM), dtype=torch.float64)
    previous_action = torch.zeros_like(action)
    joint_position = torch.tensor([[-2.0, 0.0, 3.0]], dtype=torch.float64)
    soft_lower = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float64)
    soft_upper = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    collision_history = torch.zeros((batch, 4, 3), dtype=torch.float64)
    collision_history[0, 1, 0] = 11.0
    collision_history[0, 3, 2] = 20.0

    result = compute_reward(
        robot_anchor_position=anchor_position,
        target_anchor_position=anchor_position,
        robot_anchor_quaternion=anchor_quaternion,
        target_anchor_quaternion=anchor_quaternion,
        robot_body_position=body_position,
        target_body_position=body_position,
        robot_body_quaternion=body_quaternion,
        target_body_quaternion=body_quaternion,
        robot_body_linear_velocity=body_velocity,
        target_body_linear_velocity=body_velocity,
        robot_body_angular_velocity=body_velocity,
        target_body_angular_velocity=body_velocity,
        action=action,
        previous_action=previous_action,
        joint_position=joint_position,
        soft_joint_lower=soft_lower,
        soft_joint_upper=soft_upper,
        self_collision_force_history=collision_history,
        step_duration=0.02,
    )

    for name in (
        "anchor_position",
        "anchor_orientation",
        "body_position",
        "body_orientation",
        "body_linear_velocity",
        "body_angular_velocity",
    ):
        torch.testing.assert_close(
            result.raw[name], torch.ones(batch, dtype=torch.float64)
        )
    torch.testing.assert_close(
        result.raw["action_rate"],
        torch.tensor([float(ACTION_DIM)], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.raw["joint_limit"], torch.tensor([3.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.raw["self_collision"], torch.tensor([2.0], dtype=torch.float64)
    )
    expected = (0.5 + 0.5 + 1.0 + 1.0 + 1.0 + 1.0 - 2.9 - 30.0 - 20.0) * 0.02
    torch.testing.assert_close(
        result.total, torch.tensor([expected], dtype=torch.float64)
    )


def test_terminations_keep_timeout_separate_from_failure_reasons() -> None:
    result = compute_terminations(
        elapsed_steps=torch.tensor([500, 1, 1, 1, 1]),
        max_episode_steps=500,
        robot_anchor_height=torch.tensor([0.0, 1.3, 1.0, 1.0, 1.0]),
        target_anchor_height=torch.ones(5),
        robot_projected_gravity_z=torch.tensor([-1.0, -1.0, 0.0, -1.0, -1.0]),
        target_projected_gravity_z=-1.0,
        robot_end_effector_height=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.3, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        target_end_effector_height=torch.zeros((5, 4)),
    )
    assert result.timeout.tolist() == [True, False, False, False, False]
    assert result.anchor_height.tolist() == [True, True, False, False, False]
    assert result.anchor_orientation.tolist() == [False, False, True, False, False]
    assert result.end_effector_height.tolist() == [False, False, False, True, False]
    assert result.terminated.tolist() == [True, True, True, True, False]


def test_adaptive_sampler_matches_six_bin_reference_update() -> None:
    state = AdaptiveSamplerState.create(
        motion_frame_count=297,
        sampling_fps=50.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert state.bin_count == 6
    bins = failure_bins(torch.tensor([0, 49, 50, 296]), 297, state.bin_count)
    assert bins.tolist() == [0, 0, 1, 5]

    failed = torch.tensor([2.0, 0.0, 1.0, 0.0, 0.0, 3.0], dtype=torch.float64)
    ema = update_failure_ema(torch.zeros_like(failed), failed, alpha=0.001)
    torch.testing.assert_close(ema, failed * 0.001)
    probabilities = sampling_probabilities(
        ema,
        uniform_ratio=0.1,
        kernel_size=1,
    )
    torch.testing.assert_close(
        probabilities,
        (ema + 0.1 / 6.0) / (ema + 0.1 / 6.0).sum(),
    )


def test_reset_perturbation_rotates_world_angular_velocity_into_root_frame() -> None:
    half_sqrt = math.sqrt(0.5)
    state = perturb_reference_state(
        reference_root_position=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        reference_root_quaternion=torch.tensor(
            [[half_sqrt, 0.0, 0.0, half_sqrt]], dtype=torch.float64
        ),
        reference_root_linear_velocity=torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float64
        ),
        reference_root_angular_velocity=torch.tensor(
            [[0.0, 1.0, 0.0]], dtype=torch.float64
        ),
        reference_joint_position=torch.zeros((1, 2), dtype=torch.float64),
        reference_joint_velocity=torch.ones((1, 2), dtype=torch.float64),
        position_delta=torch.tensor([[0.1, -0.2, 0.01]], dtype=torch.float64),
        euler_delta=torch.zeros((1, 3), dtype=torch.float64),
        root_velocity_delta=torch.tensor(
            [[0.2, 0.3, 0.4, 0.0, 0.0, 1.0]], dtype=torch.float64
        ),
        joint_position_delta=torch.tensor([[0.2, -0.3]], dtype=torch.float64),
        joint_lower=torch.tensor([[-0.1, -0.2]], dtype=torch.float64),
        joint_upper=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        state.root_position, torch.tensor([[0.1, -0.2, 1.01]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        state.joint_position, torch.tensor([[0.1, -0.2]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        state.root_linear_velocity_world,
        torch.tensor([[1.2, 0.3, 0.4]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        state.root_angular_velocity_body,
        torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        state.joint_velocity, torch.ones((1, 2), dtype=torch.float64)
    )
