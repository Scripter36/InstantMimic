"""Bit-exact parity for the restructured BeyondMimic observation kernel launch."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.constants import (
    OBSERVATION_NOISE_RANGES,
)
from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.optimized_kernels import (
    _normalized_rotation_columns,
    _quat_apply_inverse,
    _quat_inverse,
    _quat_multiply,
    _read_vec3,
    _read_vec4,
    interval_push_kernel,
    observation_group_count,
    observation_kernel,
    reset_reference_state_kernel,
)
from instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.reference import (
    ACTION_DIM,
    TRACKED_BODY_COUNT,
)

DEVICE = "cpu"
MOTION_FRAME_COUNT = 41
MOTION_BODY_COUNT = 32
QPOS_WIDTH = 45
QVEL_WIDTH = 43
SENSOR_WIDTH = 32
IMU_LINEAR_ADDRESS = 4
IMU_ANGULAR_ADDRESS = 9
ANCHOR_BODY_ID = 1
MOTION_ANCHOR_BODY_INDEX = 2
ACTOR_WIDTH = 160
CRITIC_WIDTH = 286


@wp.func
def _frozen_uniform(state: wp.uint32, lower: float, upper: float) -> float:
    """Frozen copy of the pre-fix helper: takes ``state`` by value."""
    return lower + (upper - lower) * wp.randf(state)


@wp.kernel(enable_backward=False)
def _frozen_observation_kernel(
    seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    time_steps: wp.array(dtype=wp.int64),
    motion_joint_position: wp.array2d(dtype=float),
    motion_joint_velocity: wp.array2d(dtype=float),
    motion_body_position: wp.array3d(dtype=float),
    motion_body_quaternion: wp.array3d(dtype=float),
    motion_anchor_body_index: int,
    body_ids: wp.array(dtype=int),
    joint_qpos_indices: wp.array(dtype=int),
    joint_qvel_indices: wp.array(dtype=int),
    anchor_body_id: int,
    imu_linear_address: int,
    imu_angular_address: int,
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    xpos: wp.array3d(dtype=float),
    xquat: wp.array3d(dtype=float),
    sensordata: wp.array2d(dtype=float),
    default_joint_position: wp.array(dtype=float),
    encoder_bias: wp.array2d(dtype=float),
    action: wp.array2d(dtype=float),
    corruption_enabled: int,
    actor_observation: wp.array2d(dtype=float),
    critic_observation: wp.array2d(dtype=float),
    action_dim: int,
    tracked_body_count: int,
):
    """Frozen copy of the pre-restructure one-thread-per-environment kernel."""
    env_idx = wp.tid()
    frame_idx = int(time_steps[env_idx])
    state = wp.rand_init(
        seed[0] + common_step_counter[0] * 17 + 1009,
        env_idx,
    )

    for joint_idx in range(action_dim):
        command_position = motion_joint_position[frame_idx, joint_idx]
        command_velocity = motion_joint_velocity[frame_idx, joint_idx]
        actor_observation[env_idx, joint_idx] = command_position
        actor_observation[env_idx, action_dim + joint_idx] = command_velocity
        critic_observation[env_idx, joint_idx] = command_position
        critic_observation[env_idx, action_dim + joint_idx] = command_velocity

    robot_anchor_position = _read_vec3(xpos, env_idx, anchor_body_id)
    robot_anchor_quaternion = _read_vec4(xquat, env_idx, anchor_body_id)
    reference_anchor_position = _read_vec3(
        motion_body_position, frame_idx, motion_anchor_body_index
    )
    reference_anchor_quaternion = _read_vec4(
        motion_body_quaternion, frame_idx, motion_anchor_body_index
    )
    anchor_position = _quat_apply_inverse(
        robot_anchor_quaternion,
        reference_anchor_position - robot_anchor_position,
    )
    anchor_orientation = _normalized_rotation_columns(
        _quat_multiply(
            _quat_inverse(robot_anchor_quaternion),
            reference_anchor_quaternion,
        )
    )
    for axis in range(3):
        actor_value = anchor_position[axis]
        if corruption_enabled == 1:
            actor_value += _frozen_uniform(state, -0.25, 0.25)
        actor_observation[env_idx, 58 + axis] = actor_value
        critic_observation[env_idx, 58 + axis] = anchor_position[axis]
    for component in range(6):
        actor_value = anchor_orientation[component]
        if corruption_enabled == 1:
            actor_value += _frozen_uniform(state, -0.05, 0.05)
        actor_observation[env_idx, 61 + component] = actor_value
        critic_observation[env_idx, 61 + component] = anchor_orientation[component]

    for body_order in range(tracked_body_count):
        body_id = body_ids[body_order]
        body_position = _quat_apply_inverse(
            robot_anchor_quaternion,
            _read_vec3(xpos, env_idx, body_id) - robot_anchor_position,
        )
        body_orientation = _normalized_rotation_columns(
            _quat_multiply(
                _quat_inverse(robot_anchor_quaternion),
                _read_vec4(xquat, env_idx, body_id),
            )
        )
        for axis in range(3):
            critic_observation[env_idx, 67 + body_order * 3 + axis] = body_position[
                axis
            ]
        for component in range(6):
            critic_observation[env_idx, 109 + body_order * 6 + component] = (
                body_orientation[component]
            )

    for axis in range(3):
        linear_velocity = sensordata[env_idx, imu_linear_address + axis]
        angular_velocity = sensordata[env_idx, imu_angular_address + axis]
        actor_linear = linear_velocity
        actor_angular = angular_velocity
        if corruption_enabled == 1:
            actor_linear += _frozen_uniform(state, -0.5, 0.5)
            actor_angular += _frozen_uniform(state, -0.2, 0.2)
        actor_observation[env_idx, 67 + axis] = actor_linear
        actor_observation[env_idx, 70 + axis] = actor_angular
        critic_observation[env_idx, 193 + axis] = linear_velocity
        critic_observation[env_idx, 196 + axis] = angular_velocity

    for joint_idx in range(action_dim):
        joint_position = (
            qpos[env_idx, joint_qpos_indices[joint_idx]]
            - default_joint_position[joint_idx]
        )
        joint_velocity = qvel[env_idx, joint_qvel_indices[joint_idx]]
        actor_joint_position = joint_position + encoder_bias[env_idx, joint_idx]
        actor_joint_velocity = joint_velocity
        if corruption_enabled == 1:
            actor_joint_position += _frozen_uniform(state, -0.01, 0.01)
            actor_joint_velocity += _frozen_uniform(state, -0.5, 0.5)
        actor_observation[env_idx, 73 + joint_idx] = actor_joint_position
        actor_observation[env_idx, 102 + joint_idx] = actor_joint_velocity
        actor_observation[env_idx, 131 + joint_idx] = action[env_idx, joint_idx]
        critic_observation[env_idx, 199 + joint_idx] = joint_position
        critic_observation[env_idx, 228 + joint_idx] = joint_velocity
        critic_observation[env_idx, 257 + joint_idx] = action[env_idx, joint_idx]


def _wp(array: np.ndarray, dtype: type = wp.float32) -> wp.array:
    return wp.array(np.ascontiguousarray(array), dtype=dtype, device=DEVICE)


def _random_quaternions(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    values = rng.normal(size=(*shape, 4))
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    return values.astype(np.float32)


def _build_inputs(
    rng: np.random.Generator,
    num_envs: int,
) -> tuple[list, list]:
    """Build observation inputs shared by the corruption on/off launches."""

    def floats(*shape: int) -> np.ndarray:
        return rng.uniform(-2.0, 2.0, size=shape).astype(np.float32)

    time_steps = rng.integers(0, MOTION_FRAME_COUNT, size=num_envs).astype(np.int64)
    body_ids = np.arange(3, 3 + TRACKED_BODY_COUNT, dtype=np.int32)
    joint_qpos_indices = np.arange(2, 2 + ACTION_DIM, dtype=np.int32)
    joint_qvel_indices = np.arange(6, 6 + ACTION_DIM, dtype=np.int32)

    shared = [
        _wp(np.array([12345], dtype=np.int32), wp.int32),
        _wp(np.array([7], dtype=np.int32), wp.int32),
        _wp(time_steps, wp.int64),
        _wp(floats(MOTION_FRAME_COUNT, ACTION_DIM)),
        _wp(floats(MOTION_FRAME_COUNT, ACTION_DIM)),
        _wp(floats(MOTION_FRAME_COUNT, MOTION_BODY_COUNT, 3)),
        _wp(_random_quaternions(rng, (MOTION_FRAME_COUNT, MOTION_BODY_COUNT))),
        MOTION_ANCHOR_BODY_INDEX,
        _wp(body_ids, wp.int32),
        _wp(joint_qpos_indices, wp.int32),
        _wp(joint_qvel_indices, wp.int32),
        ANCHOR_BODY_ID,
        IMU_LINEAR_ADDRESS,
        IMU_ANGULAR_ADDRESS,
        _wp(floats(num_envs, QPOS_WIDTH)),
        _wp(floats(num_envs, QVEL_WIDTH)),
        _wp(floats(num_envs, MOTION_BODY_COUNT, 3)),
        _wp(_random_quaternions(rng, (num_envs, MOTION_BODY_COUNT))),
        _wp(floats(num_envs, SENSOR_WIDTH)),
        _wp(floats(ACTION_DIM)),
        _wp(floats(num_envs, ACTION_DIM)),
        _wp(floats(num_envs, ACTION_DIM)),
    ]
    return shared, [ACTION_DIM, TRACKED_BODY_COUNT]


def _assert_bit_equal(actual: wp.array, expected: wp.array, label: str) -> None:
    actual_bits = actual.numpy().view(np.uint32)
    expected_bits = expected.numpy().view(np.uint32)
    if np.array_equal(actual_bits, expected_bits):
        return
    mismatch = np.argwhere(actual_bits != expected_bits)
    first = tuple(int(index) for index in mismatch[0])
    raise AssertionError(
        f"{label} differs from the frozen oracle at {first}: "
        f"new={actual.numpy()[first]!r} ({actual_bits[first]:#010x}), "
        f"oracle={expected.numpy()[first]!r} ({expected_bits[first]:#010x}); "
        f"{len(mismatch)} of {actual_bits.size} values differ"
    )


@pytest.mark.parametrize("num_envs", [1, 7, 33, 129, 200])
@pytest.mark.parametrize("data_seed", [3, 11])
def test_observation_kernel_matches_frozen_oracle_bit_exact(
    num_envs: int, data_seed: int
) -> None:
    """Corruption off still matches the pre-restructure kernel bit for bit."""
    with wp.ScopedDevice(DEVICE):
        rng = np.random.default_rng(data_seed)
        actor_seed = rng.uniform(-1.0, 1.0, size=(num_envs, ACTOR_WIDTH)).astype(
            np.float32
        )
        critic_seed = rng.uniform(-1.0, 1.0, size=(num_envs, CRITIC_WIDTH)).astype(
            np.float32
        )
        shared, tail = _build_inputs(rng, num_envs)
        oracle_outputs = [_wp(actor_seed.copy()), _wp(critic_seed.copy())]
        new_outputs = [_wp(actor_seed.copy()), _wp(critic_seed.copy())]

        wp.launch(
            _frozen_observation_kernel,
            dim=num_envs,
            inputs=[*shared, 0, *oracle_outputs, *tail],
            device=DEVICE,
        )
        wp.launch(
            observation_kernel,
            dim=(num_envs, observation_group_count(ACTION_DIM, TRACKED_BODY_COUNT)),
            inputs=[*shared, 0, *new_outputs, *tail],
            device=DEVICE,
        )

        _assert_bit_equal(new_outputs[0], oracle_outputs[0], "actor observation")
        _assert_bit_equal(new_outputs[1], oracle_outputs[1], "critic observation")


def test_observation_group_count_covers_every_row_element() -> None:
    groups = observation_group_count(ACTION_DIM, TRACKED_BODY_COUNT)
    assert groups == ACTION_DIM + 9 + 6 + ACTION_DIM + 9 * TRACKED_BODY_COUNT
    assert groups <= ACTOR_WIDTH + CRITIC_WIDTH


RNG_SEED = 12345
RNG_STEP = 7
RNG_ENV_COUNT = 4096
# Counter ranges hard-coded by the three kernels' `_slot_uniform` calls.
OBSERVATION_SLOT_COUNT = 15 + 2 * ACTION_DIM
RESET_SLOT_COUNT = 12 + ACTION_DIM
PUSH_SLOT_COUNT = 7


def _bool_array(values: np.ndarray) -> wp.array:
    return wp.array(values.astype(np.bool_), dtype=wp.bool, device=DEVICE)


def _int_array(values: np.ndarray) -> wp.array:
    return wp.array(
        np.ascontiguousarray(values, dtype=np.int32), dtype=wp.int32, device=DEVICE
    )


def _assert_rng_columns(
    draws: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: tuple[slice, ...],
    label: str,
) -> None:
    """Draws are distinct per element, uncorrelated across envs, in bounds."""
    env_count, column_count = draws.shape
    assert env_count >= 1000
    assert draws.shape[1] == lower.size == upper.size

    # (a) Different elements of one environment draw different values. Every
    # group shares one bound, so the pre-fix by-value helper collapsed it to a
    # single repeated value.
    for env_idx in range(8):
        for group in groups:
            values = draws[env_idx, group]
            distinct = np.unique(values).size
            assert distinct == values.size, (
                f"{label}: env {env_idx} repeated draws inside a {values.size}-draw "
                f"group with values {values.tolist()}"
            )

    # (b) Draws for one element across environments are independent: the
    # correlation matrix of the columns is near diagonal.
    correlation = np.corrcoef(draws.T)
    off_diagonal = np.abs(correlation - np.eye(column_count))
    worst = float(off_diagonal.max())
    assert worst < 0.1, f"{label}: cross-env correlation reached {worst:.3f}"

    # (c) Every draw is inside its bounds and the column means sit at the
    # center of those bounds.
    assert np.all(draws >= lower - 1.0e-6), f"{label}: draw below lower bound"
    assert np.all(draws <= upper + 1.0e-6), f"{label}: draw above upper bound"
    half_range = 0.5 * (upper - lower)
    tolerance = np.maximum(0.002, 0.05 * half_range)
    excess = np.abs(draws.mean(axis=0) - 0.5 * (lower + upper)) - tolerance
    assert np.all(excess <= 0.0), f"{label}: column mean left the bound center"


def _slot_offsets(slot_count: int, env_count: int) -> np.ndarray:
    """Return every ``env * slot_count + slot`` counter offset, row-major."""
    env_indices = np.arange(env_count, dtype=np.int64)[:, None]
    slots = np.arange(slot_count, dtype=np.int64)[None, :]
    return env_indices * slot_count + slots


def _assert_no_cross_env_slot_alias(
    draws: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    label: str,
) -> None:
    """Env e's final slot must not share an RNG state with env e+1's slot 0.

    ``draws`` must start with slot 0 and end with the highest counter slot.
    """
    normalized = (draws - lower) / (upper - lower)
    alias = np.abs(normalized[:-1, -1] - normalized[1:, 0]) <= 1.0e-5
    aliased = np.flatnonzero(alias)
    assert aliased.size == 0, (
        f"{label}: env e final slot aliases env e+1 first slot at "
        f"{aliased[:8].tolist()}"
    )


def _euler_zyx_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Invert the reset kernel's roll-pitch-yaw quaternion product."""
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = (quaternion[:, index] for index in range(4))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=1)


def _rotate_wxyz(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate world-frame vectors by ``quaternion`` (body-to-world)."""
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    scalar = quaternion[:, 0:1]
    vector_part = quaternion[:, 1:4]
    cross = np.cross(vector_part, vectors)
    return vectors + scalar * (2.0 * cross) + np.cross(vector_part, 2.0 * cross)


def _reset_draws(
    env_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[slice, ...]]:
    identity_quaternion = np.zeros((MOTION_FRAME_COUNT, 4, 4), dtype=np.float32)
    identity_quaternion[:, :, 0] = 1.0
    joint_qpos_indices = np.arange(7, 7 + ACTION_DIM, dtype=np.int32)
    joint_qvel_indices = np.arange(7, 7 + ACTION_DIM, dtype=np.int32)
    qpos = wp.zeros((env_count, 40), dtype=wp.float32, device=DEVICE)
    qvel = wp.zeros((env_count, 40), dtype=wp.float32, device=DEVICE)
    wp.launch(
        reset_reference_state_kernel,
        dim=env_count,
        inputs=[
            _bool_array(np.ones(env_count)),
            _int_array(np.array([RNG_SEED])),
            _int_array(np.array([RNG_STEP])),
            17,
            _wp(np.zeros(env_count, dtype=np.int64), wp.int64),
            _wp(np.zeros((MOTION_FRAME_COUNT, ACTION_DIM), dtype=np.float32)),
            _wp(np.zeros((MOTION_FRAME_COUNT, ACTION_DIM), dtype=np.float32)),
            _wp(np.zeros((MOTION_FRAME_COUNT, 4, 3), dtype=np.float32)),
            _wp(identity_quaternion),
            _wp(np.zeros((MOTION_FRAME_COUNT, 4, 3), dtype=np.float32)),
            _wp(np.zeros((MOTION_FRAME_COUNT, 4, 3), dtype=np.float32)),
            0,
            _int_array(joint_qpos_indices),
            _int_array(joint_qvel_indices),
            _wp(np.array([[-1.0, 1.0]] * 3, dtype=np.float32)),
            _wp(np.array([[-0.4, 0.4]] * 3, dtype=np.float32)),
            _wp(np.array([[-1.0, 1.0]] * 6, dtype=np.float32)),
            _wp(np.array([-1.0, 1.0], dtype=np.float32)),
            _wp(np.full(ACTION_DIM, -10.0, dtype=np.float32)),
            _wp(np.full(ACTION_DIM, 10.0, dtype=np.float32)),
            0,
            qpos,
            qvel,
            wp.zeros((env_count, ACTION_DIM), dtype=wp.float32, device=DEVICE),
            wp.zeros((env_count, ACTION_DIM), dtype=wp.float32, device=DEVICE),
            wp.zeros((env_count, ACTION_DIM), dtype=wp.float32, device=DEVICE),
            wp.zeros((env_count, ACTION_DIM), dtype=wp.float32, device=DEVICE),
            wp.zeros((env_count, 1, 1, 3), dtype=wp.float32, device=DEVICE),
            ACTION_DIM,
            1,
        ],
        device=DEVICE,
    )
    qpos_values = qpos.numpy().astype(np.float64)
    qvel_values = qvel.numpy().astype(np.float64)
    root_quaternion = qpos_values[:, 3:7]
    # The reset kernel composes the euler delta with the reference quaternion;
    # the reference is identity here, so the stored quaternion is the draw.
    euler_delta = _euler_zyx_from_quaternion_wxyz(root_quaternion)
    # qvel holds the body-frame angular delta; rotate it back to world frame.
    angular_velocity_delta = _rotate_wxyz(root_quaternion, qvel_values[:, 3:6])
    draws = np.concatenate(
        (
            qpos_values[:, 0:3],
            euler_delta,
            qvel_values[:, 0:3],
            angular_velocity_delta,
            qpos_values[:, joint_qpos_indices],
        ),
        axis=1,
    )
    lower = np.concatenate(
        (
            np.full(3, -1.0),
            np.full(3, -0.4),
            np.full(3, -1.0),
            np.full(3, -1.0),
            np.full(ACTION_DIM, -1.0),
        )
    )
    upper = np.concatenate(
        (
            np.full(3, 1.0),
            np.full(3, 0.4),
            np.full(3, 1.0),
            np.full(3, 1.0),
            np.full(ACTION_DIM, 1.0),
        )
    )
    groups = (
        slice(0, 3),
        slice(3, 6),
        slice(6, 9),
        slice(9, 12),
        slice(12, 12 + ACTION_DIM),
    )
    return draws, lower, upper, groups


def _push_draws(
    env_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[slice, ...]]:
    qpos = wp.zeros((env_count, 7), dtype=wp.float32, device=DEVICE)
    qpos_numpy = qpos.numpy()
    qpos_numpy[:, 3] = 1.0
    qpos = _wp(qpos_numpy)
    qvel = wp.zeros((env_count, 6), dtype=wp.float32, device=DEVICE)
    push_time_left = wp.zeros(env_count, dtype=wp.float32, device=DEVICE)
    wp.launch(
        interval_push_kernel,
        dim=env_count,
        inputs=[
            _int_array(np.array([RNG_SEED])),
            _int_array(np.array([RNG_STEP])),
            _wp(np.array([1.0, 2.0], dtype=np.float32)),
            _wp(np.array([[-1.0, 1.0]] * 6, dtype=np.float32)),
            0,
            0.02,
            qpos,
            qvel,
            _wp(np.zeros((env_count, 2, 3), dtype=np.float32)),
            _wp(np.zeros((env_count, 2, 6), dtype=np.float32)),
            _wp(np.zeros((env_count, 2, 3), dtype=np.float32)),
            push_time_left,
        ],
        device=DEVICE,
    )
    # Slot order: interval first, then linear and angular velocity axes.
    draws = np.concatenate(
        (push_time_left.numpy()[:, None], qvel.numpy()[:, 0:3], qvel.numpy()[:, 3:6]),
        axis=1,
    ).astype(np.float64)
    lower = np.concatenate(([1.0], np.full(6, -1.0)))
    upper = np.concatenate(([2.0], np.full(6, 1.0)))
    groups = (slice(0, 1), slice(1, 4), slice(4, 7))
    return draws, lower, upper, groups


def test_reset_draws_are_independent() -> None:
    with wp.ScopedDevice(DEVICE):
        draws, lower, upper, groups = _reset_draws(RNG_ENV_COUNT)
        _assert_rng_columns(draws, lower, upper, groups, "reset")
        _assert_no_cross_env_slot_alias(draws, lower, upper, "reset")


def test_push_draws_are_independent() -> None:
    with wp.ScopedDevice(DEVICE):
        draws, lower, upper, groups = _push_draws(RNG_ENV_COUNT)
        _assert_rng_columns(draws, lower, upper, groups, "push")
        _assert_no_cross_env_slot_alias(draws, lower, upper, "push")


def test_slot_offsets_do_not_alias_between_environments() -> None:
    for label, slot_count in (
        ("observation", OBSERVATION_SLOT_COUNT),
        ("reset", RESET_SLOT_COUNT),
        ("push", PUSH_SLOT_COUNT),
    ):
        offsets = _slot_offsets(slot_count, RNG_ENV_COUNT)
        assert offsets.shape == (RNG_ENV_COUNT, slot_count)
        assert np.unique(offsets).size == offsets.size, (
            f"{label}: {offsets.size - np.unique(offsets).size} duplicate "
            f"(env, slot) offsets"
        )


def test_observation_corruption_draws_are_independent() -> None:
    env_count = RNG_ENV_COUNT
    noise_columns = ((58, 61), (61, 67), (67, 70), (70, 73), (73, 102), (102, 131))
    half_ranges = np.concatenate(
        [
            np.full(width, OBSERVATION_NOISE_RANGES[name])
            for name, width in (
                ("anchor_position", 3),
                ("anchor_orientation", 6),
                ("base_linear_velocity", 3),
                ("base_angular_velocity", 3),
                ("joint_position", ACTION_DIM),
                ("joint_velocity", ACTION_DIM),
            )
        ]
    ).astype(np.float64)
    groups: list[slice] = []
    offset = 0
    for start, end in noise_columns:
        groups.append(slice(offset, offset + end - start))
        offset += end - start

    with wp.ScopedDevice(DEVICE):
        shared, tail = _build_inputs(np.random.default_rng(23), env_count)
        group_count = observation_group_count(ACTION_DIM, TRACKED_BODY_COUNT)
        outputs = {}
        for corruption in (0, 1):
            actor = wp.zeros((env_count, ACTOR_WIDTH), dtype=wp.float32, device=DEVICE)
            critic = wp.zeros(
                (env_count, CRITIC_WIDTH), dtype=wp.float32, device=DEVICE
            )
            wp.launch(
                observation_kernel,
                dim=(env_count, group_count),
                inputs=[*shared, corruption, actor, critic, *tail],
                device=DEVICE,
            )
            outputs[corruption] = (actor.numpy(), critic.numpy())

        clean_actor, clean_critic = outputs[0]
        noisy_actor, noisy_critic = outputs[1]
        assert np.array_equal(
            noisy_critic.view(np.uint32), clean_critic.view(np.uint32)
        ), "corruption must not touch critic observations"
        noise = np.concatenate(
            [noisy_actor[:, start:end] for start, end in noise_columns], axis=1
        ).astype(np.float64) - np.concatenate(
            [clean_actor[:, start:end] for start, end in noise_columns], axis=1
        ).astype(np.float64)
        _assert_rng_columns(
            noise,
            -half_ranges,
            half_ranges,
            tuple(groups),
            "observation corruption",
        )
        # The concatenation starts at anchor-position slot 0 and ends at the
        # final joint-velocity slot, which is the highest noise slot.
        _assert_no_cross_env_slot_alias(
            noise, -half_ranges, half_ranges, "observation corruption"
        )
