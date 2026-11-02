"""Repository RK4 exposes physical trial clocks without extra time advances."""

from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

import instantmimic.utils.mjw_step_optimized as optimized


@wp.kernel
def _time_force(
    time: wp.array(dtype=float),
    inertia: float,
    passive: wp.array2d(dtype=float),
    clocks: wp.array2d(dtype=float),
    counts: wp.array(dtype=int),
):
    world = wp.tid()
    stage = counts[world]
    clocks[world, stage] = time[world]
    counts[world] = stage + 1
    passive[world, 0] += inertia * time[world]


@pytest.mark.parametrize("joint", ["hinge", "free"])
@pytest.mark.parametrize(
    "jacobian,optimized_kernel", [("dense", False), ("sparse", True)]
)
def test_rk4_custom_force_uses_trial_time(
    joint, jacobian, optimized_kernel, monkeypatch
):
    joint_xml = "<freejoint/>" if joint == "free" else '<joint axis="0 1 0"/>'
    host_model = mujoco.MjModel.from_xml_string(
        f'<mujoco><option integrator="RK4" jacobian="{jacobian}" gravity="0 0 0"/>'
        f'<worldbody><body>{joint_xml}<geom type="sphere" size=".1" mass=".3" '
        'contype="0" conaffinity="0"/></body></worldbody></mujoco>'
    )
    host_data = mujoco.MjData(host_model)
    host_data.qvel[0] = 0.013
    mujoco.mj_forward(host_model, host_data)
    mass = np.zeros((host_model.nv, host_model.nv))
    mujoco.mj_fullM(host_model, host_data, mass)
    initial = host_data.qpos.copy()
    t0 = np.array([0.2, 0.5])
    h = np.array([0.04, 0.06])
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(host_model)
        model.opt.graph_conditional = True
        model.opt.timestep = wp.array(h, dtype=float, device="cpu")
        data = mjw.put_data(host_model, host_data, nworld=2, nconmax=0, njmax=0)
        wp.copy(data.time, wp.array(t0, dtype=float, device="cpu"))
        scratch = optimized.make_optimized_fwd_position_scratch(model, data, "cpu")
        assert scratch.constraint_free
        clocks = wp.zeros((2, 8), dtype=float, device="cpu")
        counts = wp.zeros(2, dtype=int, device="cpu")
        native_passive = scratch.passive_call

        def passive(m, d):
            native_passive(m, d)
            wp.launch(
                _time_force,
                dim=2,
                inputs=[d.time, float(mass[0, 0]), d.qfrc_passive, clocks, counts],
            )

        scratch.passive_call = passive
        data.qfrc_passive.fill_(123)
        optimized.step_optimized_fwd_position(
            model, data, (), scratch, use_optimized_kernel=optimized_kernel
        )
        np.testing.assert_array_equal(counts.numpy(), [4, 4])
        np.testing.assert_allclose(
            clocks.numpy()[:, :4],
            t0[:, None] + h[:, None] * [0, 0.5, 0.5, 1],
            atol=1e-7,
        )
        np.testing.assert_allclose(data.time.numpy(), t0 + h, atol=1e-7)
        np.testing.assert_allclose(
            data.qvel.numpy()[:, 0], 0.013 + t0 * h + 0.5 * h**2, atol=1e-7
        )
        np.testing.assert_allclose(
            data.qpos.numpy()[:, 0],
            initial[0] + 0.013 * h + 0.5 * t0 * h**2 + h**3 / 6,
            atol=1e-7,
        )
        storage = [
            data.qpos,
            data.qvel,
            data.time,
            scratch.rk4.time_t0,
            scratch.rk4.qpos_t0,
            scratch.rk4.qvel_t0,
            scratch.rk4.qvel_rk,
            scratch.rk4.qacc_rk,
        ]
        pointers = [a.ptr for a in storage]

        def unexpected_allocation(*args, **kwargs):
            pytest.fail("Warmed repository RK4 allocated during a physical step")

        with monkeypatch.context() as patch:
            for name in ["empty", "zeros", "clone", "empty_like", "zeros_like"]:
                patch.setattr(wp, name, unexpected_allocation)
            optimized.step_optimized_fwd_position(
                model, data, (), scratch, use_optimized_kernel=optimized_kernel
            )
        assert [a.ptr for a in storage] == pointers
        np.testing.assert_array_equal(counts.numpy(), [8, 8])
        np.testing.assert_allclose(
            clocks.numpy()[:, 4:],
            t0[:, None] + h[:, None] * [1, 1.5, 1.5, 2],
            atol=1e-7,
        )
        np.testing.assert_allclose(data.time.numpy(), t0 + 2 * h, atol=1e-7)
        np.testing.assert_allclose(
            data.qvel.numpy()[:, 0], 0.013 + t0 * 2 * h + 0.5 * (2 * h) ** 2, atol=1e-7
        )


def test_conditional_rk4_capture_requires_structural_rowless_clock_scratch():
    model = SimpleNamespace(
        opt=SimpleNamespace(
            integrator=optimized.IntegratorType.RK4,
            graph_conditional=True,
            enableflags=0,
        )
    )
    scratch = SimpleNamespace(
        rk4=SimpleNamespace(time_t0=object()), constraint_free=True
    )
    assert optimized.supports_allocation_free_implicit_capture(
        model, scratch, use_optimized_kernel=True
    )
    scratch.constraint_free = False
    assert not optimized.supports_allocation_free_implicit_capture(
        model, scratch, use_optimized_kernel=True
    )
    scratch.constraint_free = True
    model.opt.enableflags = optimized.EnableBit.SLEEP
    assert not optimized.supports_allocation_free_implicit_capture(
        model, scratch, use_optimized_kernel=True
    )
    model.opt.enableflags = 0
    scratch.rk4.time_t0 = None
    assert not optimized.supports_allocation_free_implicit_capture(
        model, scratch, use_optimized_kernel=True
    )
