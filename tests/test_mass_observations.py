"""Ordinary MASS observations backed by the actual SNU body model."""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch

from instantmimic.envs.controls.fast_muscle.bindings import (
    FastMuscleActionBufferBindings,
    FastMuscleExecutionBindings,
    FastMuscleRuntimeInputs,
    FastMuscleSimulationBindings,
)
from instantmimic.envs.controls.fast_muscle.config_models import (
    FastMuscleObservationSettings,
)
from instantmimic.envs.controls.fast_muscle.control import FastMuscleControlCfg
from instantmimic.envs.controls.fast_muscle.controller import FastMuscleController
from instantmimic.envs.controls.fast_muscle.settings import FastMuscleControlSettings
from instantmimic.muscle.parsing import mjcf


@pytest.fixture
def actual_controller(monkeypatch):
    monkeypatch.setattr(mjcf, "mjcf_cache_dir", lambda: None)
    root = Path(__file__).resolve().parents[1]

    def make(mode: str, geometry: bool = True):
        pose = mode == "pose"
        output = (
            {
                "type": "pose",
                "reference": "rest",
                "stable_pd": {
                    "mjcf_path": str(root / "data/snu_humanoid/mjcf/humanoid_pd.xml")
                },
            }
            if pose
            else {"type": "torque", "scale": 1.0}
        )
        action = (
            {"type": "torque", "space": 50, "scale": 1.0}
            if mode == "direct"
            else {
                "type": "mass",
                "space": 50,
                "policy_output": output,
                "muscle_nn": {"hidden_dims": [8]},
            }
        )
        settings = FastMuscleControlSettings.from_cfg(
            {
                "action": action,
                "observations": {"previous_action": None, "muscle_geometry": geometry},
                "activation_visualization": {"enabled": False},
            }
        )
        pipeline = FastMuscleControlCfg(settings=settings).build_control_pipeline()
        simulation_spec = pipeline.build_sim_spec(
            str(root / "data/snu_humanoid/mjcf/humanoid.xml"),
            mujoco.MjSpec.from_file,
        )
        bootstrap = pipeline.bootstrap
        assert bootstrap is not None
        model = simulation_spec.compile()
        data = mujoco.MjData(model)
        velocity = np.linspace(-0.15, 0.15, model.nv)
        velocity[:6] = 0
        mujoco.mj_integratePos(model, data.qpos, velocity, 0.35)
        mujoco.mj_forward(model, data)
        warp_model, warp_data = (
            mjw.put_model(model),
            mjw.put_data(model, data, nworld=2),
        )
        actions = {}
        controller = FastMuscleController(
            bootstrap=bootstrap,
            action_buffers=FastMuscleActionBufferBindings(
                lambda: actions.get("value"),
                lambda value: actions.update(value=value),
                lambda: False,
            ),
            simulation=FastMuscleSimulationBindings(
                model, warp_model, warp_data, torch.device("cpu"), 2, 1 / 120, 1 / 30
            ),
            runtime_inputs=FastMuscleRuntimeInputs(4, "implicitfast"),
            settings=settings,
        )
        controller.bind_action_buffer(torch.zeros(2, controller.policy_action_dim))
        controller.bind_execution(
            FastMuscleExecutionBindings(lambda: None, lambda: None)
        )
        controller.finalize_runtime_state()
        return controller, data

    return make


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["pose", "torque"])
def test_mass_geometry_observation_preserves_native_body_values_and_order(
    actual_controller, mode
):
    controller, data = actual_controller(mode)
    expected = torch.from_numpy(
        np.concatenate(
            [
                getattr(data, name).reshape(-1)
                for name in ("xpos", "xquat", "xipos", "subtree_com", "cdof")
            ]
        ).astype(np.float32)
    ).repeat(2, 1)
    assert controller.muscle_count == 284
    assert controller.muscle_geometry_observation_dim == expected.shape[1] == 648
    actual = controller.get_muscle_geometry_observation()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    target = torch.full((2, 655), -17.0)
    assert controller.write_muscle_geometry_observation(target, 3) == 651
    torch.testing.assert_close(target[:, 3:651], expected, rtol=0, atol=0)
    assert torch.all(target[:, :3] == -17) and torch.all(target[:, 651:] == -17)
    torch.testing.assert_close(
        controller._build_musclenn_raw_context(), expected, rtol=0, atol=0
    )


@pytest.mark.integration
def test_disabling_policy_geometry_keeps_coordinator_physics_context(actual_controller):
    controller, _data = actual_controller("torque", geometry=False)
    assert controller.muscle_geometry_observation_dim == 0
    assert controller.get_muscle_geometry_observation() is None
    assert controller._build_musclenn_raw_context().shape == (2, 648)
    target = torch.full((2, 5), -17.0)
    assert controller.write_muscle_geometry_observation(target, 3) == 3
    assert torch.all(target == -17)


@pytest.mark.integration
def test_direct_torque_has_no_policy_muscle_geometry_suffix(actual_controller):
    controller, _data = actual_controller("direct")
    assert controller.policy_action_dim == 50
    assert controller.muscle_geometry_observation_dim == 0
    assert controller.get_muscle_geometry_observation() is None


@pytest.mark.parametrize("invalid", [1, "true", None])
def test_muscle_geometry_setting_requires_a_boolean(invalid):
    with pytest.raises(ValueError):
        FastMuscleObservationSettings(muscle_geometry=invalid)


@pytest.mark.integration
def test_actual_mass_pose_stable_pd_matches_full_inertia_equation(actual_controller):
    controller, data = actual_controller("pose")
    spd = controller._stable_pd_controller
    actions = torch.linspace(-0.05, 0.05, 100).reshape(2, 50)
    output = torch.empty(2, controller.model.nv)
    inertia = np.zeros((controller.model.nv, controller.model.nv))
    mujoco.mj_fullM(controller.model, data, inertia)
    expected = torch.zeros_like(output, dtype=torch.float64)
    p, v, damping = (torch.zeros_like(expected) for _ in range(3))
    active = spd.dof_to_actuator >= 0
    actuator = spd.dof_to_actuator[active].long()
    lengths = torch.from_numpy(data.actuator_length.copy())
    velocity = torch.from_numpy(data.actuator_velocity.copy())
    p[:, active] = (actions.double() + spd.rest_ctrl.double() - lengths).index_select(
        1, actuator
    ) * spd.kp.double().index_select(0, actuator)
    v[:, active] = (-velocity * spd.kv.double()).index_select(0, actuator)
    damping[:, active] = spd.kv.double().index_select(0, actuator)
    system = torch.from_numpy(inertia).repeat(2, 1, 1)
    diagonal = system.diagonal(dim1=1, dim2=2)
    diagonal[:, active] += (
        spd.physics_dt * damping[:, active] + spd.settings.regularization
    )
    rhs = (
        -torch.from_numpy(data.qfrc_bias.copy())
        + p
        + v
        + torch.from_numpy(data.qfrc_constraint.copy())
    )
    acceleration = torch.from_numpy(
        np.linalg.solve(system.numpy(), rhs.numpy()[..., None])[..., 0]
    )
    expected = (
        p + v - spd.physics_dt * damping * acceleration
    ) * spd.settings.torque_scale
    mass_before = spd.mass_matrix.clone()
    actual = spd.compute_torque(actions, output)
    # Single precision control versus a double precision independent solve.
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(spd.mass_matrix, mass_before, rtol=0, atol=0)
