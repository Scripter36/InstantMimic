from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import pytest
import torch
import warp as wp

from instantmimic.envs.ant.tasks.ant_constants import (
    ACTUATOR_JOINT_ORDER,
    DEFAULT_JOINT_POSITIONS,
    DEFAULT_ROOT_POSITION,
    MJCF_MODEL_JOINT_ORDER,
    SOURCE_JOINT_ORDER,
    SOURCE_MJCF_JOINT_ORDER,
    SOURCE_TO_MJCF_JOINT,
    action_to_actuator_indices,
    map_actions_to_actuator_ctrl,
)
from instantmimic.envs.ant.tasks.ant_env_cfg import AntEnvCfg
from instantmimic.envs.ant.tasks.ant_math import (
    build_observation,
    compute_intermediate_values,
    compute_rewards,
    quat_rotate_inverse,
)
from instantmimic.play.stepper import (
    _maybe_extend_playback_episode_length,
    _playback_done_policy_stops,
)


def test_source_joint_order_maps_around_nv_ant_actuator_order() -> None:
    assert SOURCE_JOINT_ORDER == (
        "front_left_leg",
        "front_right_leg",
        "left_back_leg",
        "right_back_leg",
        "front_left_foot",
        "front_right_foot",
        "left_back_foot",
        "right_back_foot",
    )
    assert MJCF_MODEL_JOINT_ORDER == (
        "hip_1",
        "ankle_1",
        "hip_2",
        "ankle_2",
        "hip_3",
        "ankle_3",
        "hip_4",
        "ankle_4",
    )
    assert SOURCE_MJCF_JOINT_ORDER == (
        "hip_1",
        "hip_2",
        "hip_3",
        "hip_4",
        "ankle_1",
        "ankle_2",
        "ankle_3",
        "ankle_4",
    )
    assert (
        tuple(SOURCE_TO_MJCF_JOINT[source_name] for source_name in SOURCE_JOINT_ORDER)
        == SOURCE_MJCF_JOINT_ORDER
    )
    assert action_to_actuator_indices(ACTUATOR_JOINT_ORDER) == (2, 4, 6, 0, 3, 5, 7, 1)

    actions = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    ctrl = map_actions_to_actuator_ctrl(
        actions,
        torch.tensor(action_to_actuator_indices(ACTUATOR_JOINT_ORDER)),
        action_scale=0.5,
    )
    torch.testing.assert_close(
        ctrl,
        torch.tensor([[1.5, 3.5, 0.0, 2.0, 0.5, 2.5, 1.0, 3.0]]),
    )


def test_default_joint_positions_are_source_faithful() -> None:
    assert (
        0.0,
        0.0,
        0.0,
        0.0,
        math.pi / 4,
        -math.pi / 4,
        -math.pi / 4,
        math.pi / 4,
    ) == DEFAULT_JOINT_POSITIONS


@pytest.mark.parametrize("dimension", ("action_space", "observation_space"))
def test_ant_rejects_non_source_dimensions(dimension: str) -> None:
    from instantmimic.envs.ant.tasks.ant_env import AntEnv

    cfg = AntEnvCfg()
    setattr(cfg, dimension, 7)
    with pytest.raises(ValueError, match="action_space=8 and observation_space=36"):
        AntEnv(cfg)


def test_source_episode_limit_is_respected_for_900_control_steps() -> None:
    class _PlaybackEnv:
        max_episode_length = 900
        step_dt = 1.0 / 60.0
        cfg = SimpleNamespace(episode_length_s=15.0)

    env = _PlaybackEnv()
    previous_max, resolved_max = _maybe_extend_playback_episode_length(
        env,
        play_length=900,
        time_limit_policy="respect",
    )

    assert math.ceil(env.cfg.episode_length_s / env.step_dt) == 900
    assert (previous_max, resolved_max) == (900, 900)
    assert env.cfg.episode_length_s == pytest.approx(15.0)
    assert _playback_done_policy_stops("stop")


def test_rotation_intermediate_values_preserve_source_target_angle_pitfall() -> None:
    targets = torch.tensor([[10.0, 0.0, 2.0]])
    torso_position = torch.tensor([[0.0, 0.0, 0.5]])
    torso_rotation = torch.tensor(
        [[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]]
    )
    velocity = torch.tensor([[1.0, 2.0, 3.0]])
    ang_velocity = torch.tensor([[0.5, -0.25, 0.75]])
    dof_pos = torch.zeros((1, 8))
    lower = torch.full((8,), -1.0)
    upper = torch.full((8,), 1.0)
    potentials = torch.tensor([-120.0])
    previous = torch.tensor([-120.0])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    basis_x = torch.tensor([[1.0, 0.0, 0.0]])
    basis_z = torch.tensor([[0.0, 0.0, 1.0]])

    result = compute_intermediate_values(
        targets,
        torso_position,
        torso_rotation,
        velocity,
        ang_velocity,
        dof_pos,
        lower,
        upper,
        identity,
        basis_x,
        basis_z,
        potentials,
        previous,
        1.0 / 120.0,
    )

    _, _, _, _, _, _, roll, _, yaw, angle_to_target, _, previous_out, potential_out = (
        result
    )
    assert roll.item() == pytest.approx(0.0, abs=1e-6)
    assert yaw.item() == pytest.approx(math.pi / 2, abs=1e-6)
    expected_walk_angle = math.atan2(2.0 - 0.5, 10.0)
    assert angle_to_target.item() == pytest.approx(
        expected_walk_angle - math.pi / 2, abs=1e-6
    )
    assert previous_out.item() == pytest.approx(-120.0)
    assert potential_out.item() == pytest.approx(-math.sqrt(10.0**2 + 0.0**2) * 120.0)


def test_inverse_rotation_matches_source_a_minus_b_plus_c() -> None:
    quaternion = torch.tensor(
        [[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]]
    )
    vector = torch.tensor([[1.0, 2.0, 3.0]])

    rotated = quat_rotate_inverse(quaternion, vector)

    torch.testing.assert_close(rotated, torch.tensor([[2.0, -1.0, 3.0]]))


def test_reset_potential_lifecycle_has_zero_initial_progress() -> None:
    targets = torch.tensor([[1000.0, 0.0, 0.0]])
    position = torch.tensor([[0.0, 0.0, 0.5]])
    flattened_to_target = targets - position
    flattened_to_target[:, 2] = 0.0
    potential = -torch.linalg.vector_norm(flattened_to_target, dim=-1) / (1.0 / 120.0)
    previous = torch.zeros_like(potential)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    result = compute_intermediate_values(
        targets,
        position,
        identity,
        torch.zeros((1, 3)),
        torch.zeros((1, 3)),
        torch.zeros((1, 8)),
        torch.full((8,), -1.0),
        torch.full((8,), 1.0),
        identity,
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        potential,
        previous,
        1.0 / 120.0,
    )
    assert result[-1].item() == pytest.approx(potential.item())
    assert result[-2].item() == pytest.approx(potential.item())
    assert (result[-1] - result[-2]).item() == pytest.approx(0.0)


def test_reset_observation_clears_previous_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from instantmimic.envs.ant.tasks.ant_env import AntEnv

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp-cache"))
    cfg = AntEnvCfg()
    cfg.scene.num_envs = 1
    cfg.use_cuda_graph = False
    cfg.use_parallelized_kernel_launch = False
    cfg.use_optimized_kernel = False
    env = AntEnv(cfg)
    try:
        env.reset()
        torch.testing.assert_close(
            env._joint_qpos_idx,
            torch.tensor([7, 9, 11, 13, 8, 10, 12, 14], device=env.device),
        )
        torch.testing.assert_close(
            env._joint_qvel_idx,
            torch.tensor([6, 8, 10, 12, 7, 9, 11, 13], device=env.device),
        )
        torch.testing.assert_close(
            env.qpos_view.index_select(1, env._joint_qpos_idx),
            torch.tensor([DEFAULT_JOINT_POSITIONS], device=env.device),
        )

        expected_scaled_qpos = torch.tensor(
            [[-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]],
            device=env.device,
        )
        source_qpos = env._joint_lower_limits + (expected_scaled_qpos + 1.0) * 0.5 * (
            env._joint_upper_limits - env._joint_lower_limits
        )
        source_qvel = torch.arange(
            1, 9, dtype=torch.float32, device=env.device
        ).reshape(1, 8)
        env.qpos_view[:, env._joint_qpos_idx] = source_qpos
        env.qvel_view[:, env._joint_qvel_idx] = source_qvel
        env.qpos_view[:, 3:7] = torch.tensor(
            [[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]],
            device=env.device,
        )
        body_angular_velocity = torch.tensor([[1.0, 2.0, 3.0]], device=env.device)
        env.qvel_view[:, 3:6] = body_angular_velocity
        env._compute_intermediate_values()
        ordered_observation = env._get_observations()["policy"]
        torch.testing.assert_close(
            ordered_observation[:, 4:7],
            body_angular_velocity * float(cfg.angular_velocity_scale),
        )
        torch.testing.assert_close(ordered_observation[:, 12:20], expected_scaled_qpos)
        torch.testing.assert_close(ordered_observation[:, 20:28], source_qvel * 0.2)

        action = torch.tensor(
            [[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]],
            device=env.device,
        )
        env.actions.copy_(action)
        torso_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        xpos = wp.to_torch(env.mjw_data.xpos)
        env.qpos_view[:, 0] = 5.0
        env._run_mjw_kinematics()
        shifted_torso_x = xpos[0, torso_id, 0].clone()
        reset_mask = torch.ones(1, dtype=torch.bool, device=env.device)
        monkeypatch.setattr(
            env,
            "_run_mjw_kinematics",
            lambda: pytest.fail(
                "Ant reset must defer derived state to the base prefix"
            ),
        )
        env._reset_idx(reset_mask)
        env._finalize_reset_state(reset_mask)

        observation = env._get_observations()["policy"]
        torch.testing.assert_close(observation[:, -8:], torch.zeros_like(action))
        assert shifted_torso_x.item() == pytest.approx(5.0)
        assert xpos[0, torso_id, 0].item() == pytest.approx(
            float(DEFAULT_ROOT_POSITION[0])
        )

        env.episode_length_buf.fill_(env.max_episode_length - 1)
        _, before_limit = env._get_dones()
        assert not bool(before_limit.item())
        auto_reset_observation, _, _, truncated, _ = env.step(action)
        assert bool(truncated.item())
        torch.testing.assert_close(
            auto_reset_observation["policy"][:, -8:],
            torch.zeros_like(action),
        )
    finally:
        env.close()


def test_observation_order_and_scaling() -> None:
    obs = build_observation(
        torso_z=torch.tensor([0.5]),
        velocity_local=torch.tensor([[1.0, 2.0, 3.0]]),
        angular_velocity_local=torch.tensor([[4.0, 5.0, 6.0]]),
        yaw=torch.tensor([0.7]),
        roll=torch.tensor([0.8]),
        angle_to_target=torch.tensor([0.9]),
        up_projection=torch.tensor([10.0]),
        heading_projection=torch.tensor([11.0]),
        joint_positions_scaled=torch.arange(10, 18, dtype=torch.float32).reshape(1, 8),
        joint_velocities=torch.arange(5, 45, 5, dtype=torch.float32).reshape(1, 8),
        actions=torch.full((1, 8), 3.0),
    )
    expected = torch.tensor(
        [
            [
                0.5,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                0.7,
                0.8,
                0.9,
                10.0,
                11.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
            ]
        ],
    )
    torch.testing.assert_close(obs, expected)


def test_reward_formula_and_terminated_replacement() -> None:
    actions = torch.tensor([[1.0, -2.0]])
    reward = compute_rewards(
        actions=actions,
        reset_terminated=torch.tensor([False]),
        up_projection=torch.tensor([0.94]),
        heading_projection=torch.tensor([0.7]),
        joint_velocities=torch.tensor([[2.0, -3.0]]),
        joint_positions_scaled=torch.tensor([[0.99, 0.5]]),
        potentials=torch.tensor([-9.0]),
        previous_potentials=torch.tensor([-12.0]),
    )
    expected = (
        3.0
        + 0.5
        + 0.1
        + 0.5 * 0.7 / 0.8
        - 0.005 * (1.0 + 4.0)
        - 0.05 * (abs(1.0 * 2.0 * 0.2) + abs(-2.0 * -3.0 * 0.2))
        - 1.0
    )
    assert reward.item() == pytest.approx(expected)

    replaced = compute_rewards(
        actions=actions,
        reset_terminated=torch.tensor([True]),
        up_projection=torch.tensor([1.0]),
        heading_projection=torch.tensor([1.0]),
        joint_velocities=torch.zeros((1, 2)),
        joint_positions_scaled=torch.zeros((1, 2)),
        potentials=torch.tensor([100.0]),
        previous_potentials=torch.tensor([-100.0]),
    )
    assert replaced.item() == pytest.approx(-2.0)


def test_ant_asset_has_expected_topology_and_order() -> None:
    asset_path = Path("data/ant/nv_ant.xml")
    model = mujoco.MjModel.from_xml_path(str(asset_path))
    assert model.nq == 15
    assert model.nv == 14
    assert model.nu == 8
    assert [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, 9)
    ] == list(MJCF_MODEL_JOINT_ORDER)
    actuator_joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        for joint_id in model.actuator_trnid[:, 0]
    )
    assert actuator_joint_names == ACTUATOR_JOINT_ORDER

    source_actions = torch.tensor([[-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]])
    source_to_actuator_idx = torch.tensor(
        action_to_actuator_indices(ACTUATOR_JOINT_ORDER), dtype=torch.long
    )
    source_ctrl = map_actions_to_actuator_ctrl(
        source_actions,
        source_to_actuator_idx,
        action_scale=0.5,
    )
    data = mujoco.MjData(model)
    data.ctrl[:] = source_ctrl[0].numpy()
    mujoco.mj_forward(model, data)
    source_dof_indices = torch.tensor(
        [
            int(
                model.jnt_dofadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
            )
            for name in SOURCE_MJCF_JOINT_ORDER
        ],
        dtype=torch.long,
    )
    source_forces = torch.as_tensor(data.qfrc_actuator).index_select(
        0, source_dof_indices
    )
    force_per_action = source_forces / source_actions[0]
    assert torch.all(force_per_action > 0)
    torch.testing.assert_close(
        force_per_action,
        force_per_action[0].expand_as(force_per_action),
    )


def test_ant_task_registration_exposes_config_entry_point() -> None:
    import instantmimic.envs.ant.tasks  # noqa: F401

    spec = gym.spec("Isaac-Ant-Direct-v0")
    assert spec.entry_point.endswith("ant_env:AntEnv")
    assert spec.kwargs["env_cfg_entry_point"].endswith("ant_env_cfg:AntEnvCfg")
