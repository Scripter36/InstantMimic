"""MuJoCo Warp model/data contracts this repository relies on."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

from instantmimic.utils.mjw_kernels import private as mjw_private


def test_private_boundary_exports_pinned_mujoco_warp_symbols() -> None:
    from mujoco_warp._src import forward, smooth, solver, types
    from mujoco_warp._src.math import mul_quat

    assert mjw.__version__ == "3.16.0"
    assert mjw_private.mjw_forward is forward
    assert mjw_private.smooth is smooth
    assert mjw_private.solver is solver
    assert mjw_private.types is types
    assert mjw_private.mul_quat is mul_quat


def _hinge_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option jacobian="dense"/>
          <worldbody>
            <body>
              <joint type="hinge"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _two_hinge_tendon_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <site name="origin" pos="0 0 0"/>
            <body>
              <joint type="hinge"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
              <site name="moving" pos="0.1 0 0.3"/>
              <body pos="0 0 0.3">
                <joint type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.2"/>
              </body>
            </body>
          </worldbody>
          <tendon>
            <spatial name="tendon">
              <site site="origin"/>
              <site site="moving"/>
            </spatial>
          </tendon>
        </mujoco>
        """
    )


def test_put_data_accepts_sparse_capacity_argument() -> None:
    model = _hinge_model()
    data = mujoco.MjData(model)
    warp_model = mjw.put_model(model)

    warp_data = mjw.put_data(
        model,
        data,
        nworld=2,
        nconmax=4,
        njmax=8,
        njmax_nnz=8,
    )

    assert warp_data.M.shape == (2, model.nM)
    assert warp_model.M_tiles
    assert warp_data.qLD.shape[1] >= warp_model.qLD_block_total


def test_reset_and_get_data_delegate_to_multiworld_layout(tmp_path) -> None:
    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    model = _hinge_model()
    host = mujoco.MjData(model)
    warp_model = mjw.put_model(model)
    warp_data = mjw.put_data(model, host, nworld=2, nconmax=4, njmax=8)
    qpos = np.array([[0.3], [0.7]], dtype=np.float32)
    warp_data.qpos.assign(qpos)

    mjw.reset_data(warp_model, warp_data, wp.array([False, True], dtype=bool))
    wp.synchronize()

    np.testing.assert_allclose(warp_data.qpos.numpy()[0], qpos[0])
    np.testing.assert_allclose(warp_data.qpos.numpy()[1], model.qpos0)
    result = mujoco.MjData(model)
    mjw.get_data_into(result, model, warp_data, world_id=0)
    np.testing.assert_allclose(result.qpos, qpos[0])


def test_put_data_preserves_sparse_tendon_and_compact_capacity() -> None:
    model = _two_hinge_tendon_model()
    host = mujoco.MjData(model)
    host.qpos[:] = [0.2, -0.3]
    mujoco.mj_forward(model, host)

    warp_data = mjw.put_data(
        model,
        host,
        nworld=2,
        nconmax=4,
        njmax=8,
        nvmax=1,
    )

    assert warp_data.ten_J.shape == (2, model.nJten)
    np.testing.assert_allclose(warp_data.ten_J.numpy()[0], host.ten_J)
    assert warp_data.nvmax == 1
    assert warp_data.nvmax_pad >= warp_data.nvmax
    assert warp_data.cM.shape[1:] == (warp_data.nvmax_pad, warp_data.nvmax_pad)


@pytest.mark.parametrize("inputs,nu", [("pos vel ff", 3), ("none", 0)])
def test_control_inputs_and_actuator_storage_have_independent_dimensions(inputs, nu):
    history = 'delay="0.02" nsample="3"' if nu else ""
    model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody><body><joint name="hinge"/>
            <geom type="sphere" size="0.1" mass="1"/>
          </body></worldbody>
          <actuator>
            <dcmotor joint="hinge" motorconst="1" resistance="1"
                     input="{inputs}" {history}/>
          </actuator>
          <sensor><jointpos joint="hinge" delay="0.02" nsample="3"/></sensor>
        </mujoco>
        """
    )
    m = mjw.put_model(model)
    d = mjw.put_data(model, mujoco.MjData(model), nworld=2, nconmax=4, njmax=8)

    assert m.nu == nu
    assert m.nactuator == 1
    assert d.ctrl.shape == (2, nu)
    assert d.actuator_force.shape == (2, 1)
    assert d.actuator_velocity.shape == (2, 1)
    if nu:
        np.testing.assert_array_equal(m.actuator_ctrladr.numpy(), [0])
    np.testing.assert_array_equal(m.actuator_ctrlnum.numpy(), [nu])
    assert m.actuator_history.shape == (1, 1)
    assert m.actuator_history.dtype is wp.vec2i
    assert m.actuator_historyadr.shape == (1, 1)
    assert m.sensor_history.shape == (1, 1)
    assert m.sensor_history.dtype is wp.vec2i
    assert m.sensor_delay.shape == (1, 1)
