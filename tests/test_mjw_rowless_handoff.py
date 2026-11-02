"""Rowless custom force must reach implicit integration through a fresh M*qacc."""

from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import instantmimic.utils.mjw_step_optimized as optimized


@wp.kernel
def _add_passive(force: wp.array2d(dtype=float), passive: wp.array2d(dtype=float)):
    world, dof = wp.tid()
    passive[world, dof] += force[world, dof]


@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.parametrize("constraint_free", [False, True])
@pytest.mark.parametrize("use_optimized_kernel", [False, True])
def test_rowless_custom_force_refreshes_implicit_rhs(
    jacobian, constraint_free, use_optimized_kernel
):
    host_model = mujoco.MjModel.from_xml_string(
        f"""<mujoco>
          <option jacobian="{jacobian}" integrator="implicitfast"
                  timestep=".004" gravity="0 0 0"/>
          <worldbody><body>
            <joint type="hinge" axis="0 1 0"/>
            <geom type="box" size=".1 .02 .03" mass=".3"
                  contype="0" conaffinity="0"/>
            <body pos=".2 0 0">
              <joint type="hinge" axis="0 0 1"/>
              <geom type="box" size=".08 .03 .02" mass=".2"
                    contype="0" conaffinity="0"/>
            </body>
          </body></worldbody>
        </mujoco>"""
    )
    host_data = mujoco.MjData(host_model)
    host_data.qpos[:] = [0.3, -0.2]
    initial = host_data.qpos.copy()
    force = np.array([0.1, -0.07])
    host_data.qfrc_applied[:] = force
    mujoco.mj_forward(host_model, host_data)
    mass = np.zeros((host_model.nv, host_model.nv))
    mujoco.mj_fullM(host_model, host_data, mass)
    expected_acc = np.linalg.solve(mass, force)
    host_data.qfrc_applied[:] = 0

    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(host_model)
        data = mjw.put_data(host_model, host_data, nworld=2, nconmax=0, njmax=0)
        scratch = optimized.make_optimized_fwd_position_scratch(model, data, "cpu")
        assert scratch.solver_context is scratch.solver_nsolving is None
        # Rowless capacity must be valid even without the stronger structural
        # promise that no row can ever arise in this model.
        scratch.constraint_free = constraint_free
        supplied = wp.array(np.tile(force, (2, 1)), dtype=float, device="cpu")
        native_passive = scratch.passive_call

        def passive(m, d):
            native_passive(m, d)
            wp.launch(
                _add_passive, dim=(2, host_model.nv), inputs=[supplied, d.qfrc_passive]
            )

        scratch.passive_call = passive
        # Poison the implicit RHS so a stale buffer cannot accidentally pass.
        data.efc.Ma.fill_(123.0)
        data.qfrc_constraint.fill_(17.0)
        data.solver_niter.fill_(9)
        optimized.step_optimized_fwd_position(
            model, data, (), scratch, use_optimized_kernel=use_optimized_kernel
        )
        np.testing.assert_allclose(
            data.qfrc_passive.numpy(), np.tile(force, (2, 1)), atol=1e-7
        )
        np.testing.assert_allclose(
            data.efc.Ma.numpy(), np.tile(force, (2, 1)), rtol=2e-5, atol=1e-7
        )
        np.testing.assert_allclose(
            data.qacc.numpy(), np.tile(expected_acc, (2, 1)), rtol=2e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            data.qvel.numpy(),
            np.tile(0.004 * expected_acc, (2, 1)),
            rtol=2e-5,
            atol=1e-7,
        )
        expected_pos = initial + 0.004**2 * expected_acc
        np.testing.assert_allclose(
            data.qpos.numpy(), np.tile(expected_pos, (2, 1)), rtol=2e-5, atol=1e-7
        )
        np.testing.assert_allclose(
            data.qacc.numpy() @ mass.T, data.efc.Ma.numpy(), rtol=2e-5, atol=1e-7
        )
        assert not np.any(data.qfrc_constraint.numpy())
        assert not np.any(data.solver_niter.numpy())


@pytest.mark.parametrize("has_context", [False, True])
def test_nonrowless_action_suffix_retains_solver_dispatch(monkeypatch, has_context):
    calls = []
    model = SimpleNamespace(
        nv=2,
        opt=SimpleNamespace(disableflags=0),
        callback=SimpleNamespace(control=None),
    )
    data = SimpleNamespace(njmax=4)
    scratch = SimpleNamespace(
        constraint_free=False,
        solver_context=object() if has_context else None,
        solver_nsolving=object() if has_context else None,
        actuation_call=lambda *_: None,
    )
    monkeypatch.setattr(optimized, "_resolve_use_optimized_kernel", lambda *_: False)
    monkeypatch.setattr(
        optimized.mjw_forward, "fwd_acceleration", lambda *_, **__: None
    )
    monkeypatch.setattr(
        optimized, "_start_optimized_implicit_deriv_before_solve", lambda *_, **__: None
    )
    monkeypatch.setattr(
        optimized, "solve_with_scratch", lambda *_, **__: calls.append("persistent")
    )
    monkeypatch.setattr(optimized.solver, "solve", lambda *_: calls.append("upstream"))
    monkeypatch.setattr(optimized.sensor, "sensor_acc", lambda *_, **__: None)
    optimized._forward_action_suffix(
        model,
        data,
        (),
        scratch,
        optimized.OptimizedFwdVelocityResult(),
        use_optimized_kernel=False,
    )
    assert calls == ["persistent" if has_context else "upstream"]
