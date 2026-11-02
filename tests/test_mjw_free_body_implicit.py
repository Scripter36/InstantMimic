"""Implicit free-body correction preserves additive custom damping and graphs."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

from instantmimic.utils.mjw_step_optimized import (
    ImplicitQHContributor,
    PreAdvanceModifier,
    make_optimized_fwd_position_scratch,
    make_optimized_fwd_position_streams,
    step_optimized_fwd_position,
)


@wp.kernel
def _damping_qh(
    elemid: wp.array2d[int], h: float, damping: float, qh: wp.array2d[float]
):
    world, dof = wp.tid()
    qh[world, elemid[dof, dof]] += h * damping


@wp.kernel
def _damping_force(
    damping: float, velocity: wp.array2d[float], force: wp.array2d[float]
):
    world, dof = wp.tid()
    force[world, dof] = -damping * velocity[world, dof]


def _check(device, optimized, damping, *, capture=False, parallel=False):
    def host_model():
        return mujoco.MjModel.from_xml_string(
            """
            <mujoco><option timestep=".005" integrator="implicitfast" jacobian="dense"/>
              <worldbody><body pos="0 0 1"><freejoint/>
                <geom type="box" size=".1 .2 .3" pos=".04 .02 0" mass="1"/>
              </body></worldbody>
            </mujoco>
            """
        )

    custom_host = host_model()
    oracle_host = host_model()
    custom_host.dof_damping[:] = 0.02
    oracle_host.dof_damping[:] = 0.02 + damping
    host = mujoco.MjData(custom_host)
    host.qvel[:] = [0.1, -0.2, 0.3, 2.0, -3.0, 4.0]
    with wp.ScopedDevice(device):
        m = mjw.put_model(custom_host)
        oracle_m = mjw.put_model(oracle_host)
        d = mjw.put_data(custom_host, host, nworld=2, nconmax=4, njmax=8)
        oracle = mjw.put_data(oracle_host, host, nworld=2, nconmax=4, njmax=8)
        scratch = make_optimized_fwd_position_scratch(
            m, d, device, include_optimized_kernel_scratch=optimized
        )
        assert m.body_freeadr.size == 1
        scratch.implicit_qh_contributors = (
            ImplicitQHContributor(
                "damping",
                lambda qh: wp.launch(
                    _damping_qh, dim=(2, 6), inputs=[m.M_elemid, 0.005, damping, qh]
                ),
            ),
        )
        seen_acceleration = wp.empty_like(d.qacc)

        def modifier(_m, _d, qacc, qvel):
            wp.copy(seen_acceleration, qacc)
            return qacc, qvel

        scratch.pre_advance_modifier = PreAdvanceModifier("observe", modifier)
        streams = make_optimized_fwd_position_streams(device) if parallel else ()

        def step():
            wp.launch(
                _damping_force, dim=(2, 6), inputs=[damping, d.qvel, d.qfrc_applied]
            )
            step_optimized_fwd_position(
                m, d, streams, scratch, use_optimized_kernel=optimized
            )

        if capture:
            step()
            d.qpos.assign(np.tile(host.qpos, (2, 1)).astype(np.float32))
            d.qvel.assign(np.tile(host.qvel, (2, 1)).astype(np.float32))
            d.time.zero_()
            with wp.ScopedCapture(device=device) as captured:
                step()
        for _ in range(4):
            previous_velocity = oracle.qvel.numpy().copy()
            mjw.step(oracle_m, oracle)
            if capture:
                wp.capture_launch(captured.graph)
            else:
                step()
        wp.synchronize()
        for name in ("qpos", "qvel"):
            np.testing.assert_allclose(
                getattr(d, name).numpy(),
                getattr(oracle, name).numpy(),
                atol=2e-5,
                rtol=2e-5,
            )
        assert np.any(seen_acceleration.numpy() != 0.0)
        np.testing.assert_allclose(
            seen_acceleration.numpy(),
            (oracle.qvel.numpy() - previous_velocity) / 0.005,
            atol=2e-4,
            rtol=2e-4,
        )


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("damping", [0.0, 0.3])
def test_custom_free_body_damping_matches_native(optimized, damping):
    _check("cpu", optimized, damping)


@pytest.mark.cuda
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_custom_free_body_correction_captures_and_replays(optimized, parallel):
    _check("cuda:0", optimized, 0.3, capture=True, parallel=parallel)
