from __future__ import annotations

import math

import pytest
import torch
import warp as wp

from instantmimic.envs.direct_rl_env_cfg import InteractiveSceneCfg
from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.benchmark_protocol import (
    EvaluatedCheckpoint,
    convergence_iteration,
    meets_quality_threshold,
    select_best_checkpoint,
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
from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.metrics import (
    BenchmarkReference,
    BeyondMimicBenchmarkAccumulator,
    compute_ee_orientation_error,
    compute_ee_position_error,
    compute_joint_velocity_error,
    compute_mpkpe,
    compute_root_relative_mpkpe,
)


@pytest.fixture
def warp_cpu_device():
    with wp.ScopedDevice("cpu"):
        yield


def _reference(*, num_envs: int = 2, num_bodies: int = 3) -> BenchmarkReference:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    return BenchmarkReference(
        body_position_world=torch.zeros(num_envs, num_bodies, 3),
        body_position_relative_world=torch.zeros(num_envs, num_bodies, 3),
        body_quaternion_relative_world=identity.expand(num_envs, num_bodies, 4).clone(),
        joint_velocity=torch.zeros(num_envs, 2),
    )


def test_pinned_mjlab_metric_definitions() -> None:
    reference = _reference()
    robot_position = torch.zeros_like(reference.body_position_world)
    robot_position[0, :, 0] = 1.0
    robot_position[1, :, 0] = 2.0
    robot_joint_velocity = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

    torch.testing.assert_close(
        compute_mpkpe(reference.body_position_world, robot_position),
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        compute_root_relative_mpkpe(
            reference.body_position_relative_world,
            robot_position,
        ),
        torch.tensor([1.0, 2.0]),
    )
    torch.testing.assert_close(
        compute_joint_velocity_error(
            reference.joint_velocity,
            robot_joint_velocity,
        ),
        torch.tensor([math.sqrt(12.5), math.sqrt(2.0)]),
    )
    torch.testing.assert_close(
        compute_ee_position_error(
            reference.body_position_relative_world,
            robot_position,
            end_effector_indices=(1, 2),
        ),
        torch.tensor([1.0, 2.0]),
    )


def test_end_effector_orientation_error_is_shortest_path_radians() -> None:
    reference = _reference(num_envs=1)
    half_angle = math.pi / 4.0
    rotated = torch.tensor(
        [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]
    ).expand_as(reference.body_quaternion_relative_world)

    error = compute_ee_orientation_error(
        reference.body_quaternion_relative_world,
        rotated,
        end_effector_indices=(1,),
    )

    torch.testing.assert_close(error, torch.tensor([math.pi / 2.0]))


def test_accumulator_averages_each_episode_over_only_its_active_steps() -> None:
    accumulator = BeyondMimicBenchmarkAccumulator(
        num_envs=2,
        end_effector_indices=(1, 2),
        device=torch.device("cpu"),
    )
    reference = _reference()
    robot_quaternion = reference.body_quaternion_relative_world.clone()

    first_position = torch.zeros_like(reference.body_position_world)
    first_position[0, :, 0] = 1.0
    first_position[1, :, 0] = 2.0
    accumulator.update(
        reference=reference,
        robot_body_position_world=first_position,
        robot_body_quaternion_world=robot_quaternion,
        robot_joint_velocity=torch.zeros_like(reference.joint_velocity),
    )
    accumulator.mark_dones(
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([False, False]),
    )

    second_position = torch.zeros_like(reference.body_position_world)
    second_position[0, :, 0] = 3.0
    second_position[1, :, 0] = 100.0
    accumulator.update(
        reference=reference,
        robot_body_position_world=second_position,
        robot_body_quaternion_world=robot_quaternion,
        robot_joint_velocity=torch.zeros_like(reference.joint_velocity),
    )
    accumulator.mark_dones(
        terminated=torch.tensor([False, False]),
        truncated=torch.tensor([True, False]),
    )

    metrics = accumulator.finalize()
    assert metrics["success_rate"] == 0.5
    assert metrics["mpkpe"] == 2.0
    assert metrics["r_mpkpe"] == 2.0
    assert metrics["ee_pos_error"] == 2.0
    assert metrics["ee_ori_error"] == 0.0
    assert metrics["joint_vel_error"] == 0.0
    torch.testing.assert_close(accumulator.active_steps, torch.tensor([2.0, 1.0]))


def test_success_requires_timeout_without_termination() -> None:
    accumulator = BeyondMimicBenchmarkAccumulator(
        num_envs=3,
        end_effector_indices=(0,),
        device=torch.device("cpu"),
    )

    newly_done = accumulator.mark_dones(
        terminated=torch.tensor([False, True, True]),
        truncated=torch.tensor([True, False, True]),
    )

    torch.testing.assert_close(newly_done, torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(
        accumulator.success,
        torch.tensor([True, False, False]),
    )


def _benchmark_metrics(
    *,
    success_rate: float = 0.97,
    mpkpe: float = 0.10,
    r_mpkpe: float = 0.03,
    ee_pos_error: float = 0.045,
    ee_ori_error: float = 0.17,
    joint_vel_error: float = 0.5,
) -> dict[str, float]:
    return {
        "success_rate": success_rate,
        "mpkpe": mpkpe,
        "r_mpkpe": r_mpkpe,
        "ee_pos_error": ee_pos_error,
        "ee_ori_error": ee_ori_error,
        "joint_vel_error": joint_vel_error,
    }


def test_quality_threshold_requires_every_public_sanity_gate() -> None:
    assert meets_quality_threshold(_benchmark_metrics())
    assert not meets_quality_threshold(_benchmark_metrics(success_rate=0.959))
    assert not meets_quality_threshold(_benchmark_metrics(r_mpkpe=0.034))


def test_best_checkpoint_prefers_gate_then_public_metric_order() -> None:
    high_success_but_failing = EvaluatedCheckpoint(
        iteration=500,
        path="failing.pt",
        metrics=_benchmark_metrics(success_rate=0.99, mpkpe=0.121),
    )
    passing = EvaluatedCheckpoint(
        iteration=1000,
        path="passing.pt",
        metrics=_benchmark_metrics(success_rate=0.96, mpkpe=0.11),
    )
    lower_error_tie = EvaluatedCheckpoint(
        iteration=1500,
        path="lower-error.pt",
        metrics=_benchmark_metrics(success_rate=0.96, mpkpe=0.09),
    )

    selected = select_best_checkpoint(
        [high_success_but_failing, passing, lower_error_tie]
    )

    assert selected is lower_error_tie


def test_convergence_is_first_saved_checkpoint_meeting_gate() -> None:
    checkpoints = [
        EvaluatedCheckpoint(
            iteration=500,
            path="500.pt",
            metrics=_benchmark_metrics(success_rate=0.95),
        ),
        EvaluatedCheckpoint(
            iteration=1500,
            path="1500.pt",
            metrics=_benchmark_metrics(),
        ),
        EvaluatedCheckpoint(
            iteration=1000,
            path="1000.pt",
            metrics=_benchmark_metrics(),
        ),
    ]

    assert convergence_iteration(checkpoints) == 1000


@pytest.mark.integration
@pytest.mark.local_data
def test_environment_snapshots_pre_step_benchmark_reference(
    monkeypatch,
    warp_cpu_device,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = BeyondMimicTrackingEnvCfg(
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
        seed=7,
    )
    environment = BeyondMimicTrackingEnv(cfg)
    try:
        environment.reset(seed=7)
        frames = environment._time_steps.clone()
        reference = environment.snapshot_benchmark_reference()
        expected_world = environment._motion.body_position_world[
            frames[:, None],
            environment._motion_body_indices,
        ]
        torch.testing.assert_close(reference.body_position_world, expected_world)
        torch.testing.assert_close(
            reference.body_position_relative_world,
            environment._target_body_position,
        )
        torch.testing.assert_close(
            reference.body_quaternion_relative_world,
            environment._target_body_quaternion,
        )

        environment._target_body_position.add_(100.0)
        environment._target_body_quaternion.zero_()
        environment._time_steps.add_(1)
        torch.testing.assert_close(reference.body_position_world, expected_world)
        assert torch.all(reference.body_position_relative_world < 100.0)
        assert torch.any(reference.body_quaternion_relative_world != 0.0)

        robot = environment.get_benchmark_robot_state()
        torch.testing.assert_close(
            robot.body_position_world,
            environment._xpos[:, environment._body_ids],
        )
        torch.testing.assert_close(
            robot.body_quaternion_world,
            environment._xquat[:, environment._body_ids],
        )
        torch.testing.assert_close(
            robot.joint_velocity,
            environment._qvel[:, environment._joint_qvel_indices],
        )
        assert environment.benchmark_end_effector_indices == (3, 6, 10, 13)
    finally:
        environment.close()
