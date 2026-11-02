"""Dense MuJoCo Warp models beyond the upstream 60-DOF cap (CPU)."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

from instantmimic.utils.mjw_model import DENSE_NV_MAX, put_model


def _boxes(n_free: int, n_hinge: int) -> mujoco.MjModel:
    bodies = "".join(
        f'<body pos="{0.3 * i} 0 {0.032 + 0.002 * i}"><freejoint/>'
        f'<geom type="box" size="0.05 0.04 0.03"/></body>'
        for i in range(n_free)
    )
    chain = (
        "".join(
            '<body pos="0.1 0 0"><joint type="hinge" axis="0 1 0"/>'
            '<geom type="capsule" size="0.02 0.04"/>'
            for _ in range(n_hinge)
        )
        + "</body>" * n_hinge
    )
    xml = f"""<mujoco><option jacobian="dense" solver="Newton"/><worldbody>
      <geom type="plane" size="5 5 .1"/>{bodies}
      <body pos="-1 0 0.5"><freejoint/><geom size="0.05"/>{chain}</body>
    </worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def test_dense_models_up_to_the_raised_cap_match_cpu_mujoco():
    mjm = _boxes(10, 4)  # 11 free bodies + 4 hinges: nv = 70
    assert 60 < mjm.nv <= DENSE_NV_MAX
    with pytest.raises(ValueError, match="Dense is unsupported"):
        mjw.put_model(mjm)
    with wp.ScopedDevice("cpu"):
        m = put_model(mjm)
        assert not m.is_sparse and m.nv_pad == 80
        cpu = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, cpu)
        d = mjw.put_data(mjm, cpu, nworld=1, nconmax=64, njmax=512)
        for _ in range(40):  # falls onto the floor and the other boxes
            mjw.step(m, d)
            mujoco.mj_step(mjm, cpu)
        np.testing.assert_allclose(d.qpos.numpy()[0], cpu.qpos, atol=1e-5)
        assert int(d.nacon.numpy()[0]) > 0


def test_small_or_sparse_models_use_upstream_put_model():
    small = _boxes(2, 4)
    assert small.nv <= 60
    with wp.ScopedDevice("cpu"):
        assert put_model(small).nv_pad == mjw.put_model(small).nv_pad
    big = _boxes(10, 4)
    big.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    with wp.ScopedDevice("cpu"):
        assert put_model(big).is_sparse


def test_dense_models_beyond_the_raised_cap_are_rejected():
    mjm = _boxes(12, 4)  # nv = 82
    with pytest.raises(ValueError, match=f"nv <= {DENSE_NV_MAX}"):
        put_model(mjm)


def test_wide_dense_models_with_sleeping_islands_are_rejected():
    mjm = _boxes(10, 4)
    mjm.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP
    with pytest.raises(ValueError, match="sleeping islands"):
        put_model(mjm)
    mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
    with wp.ScopedDevice("cpu"):
        assert put_model(mjm).nv_pad == 80


def test_wide_dense_models_widen_the_hessian_block_only_at_nv_pad_80():
    from mujoco_warp._src import types

    from instantmimic.utils.mjw_model import WIDE_JTDAJ_BLOCK_DIM

    default = types.BlockDim().update_gradient_JTDAJ_dense
    with wp.ScopedDevice("cpu"):
        wide = put_model(_boxes(10, 4))  # nv 70 -> nv_pad 80
        narrow = put_model(_boxes(9, 2))  # nv 62 -> nv_pad 64
    assert wide.nv_pad == 80 and narrow.nv_pad == 64
    assert wide.block_dim.update_gradient_JTDAJ_dense == WIDE_JTDAJ_BLOCK_DIM
    assert narrow.block_dim.update_gradient_JTDAJ_dense == default
