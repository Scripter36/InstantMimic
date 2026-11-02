"""Implicit-fast smooth velocity derivative gates actuation and damping apart (CPU).

MuJoCo adds the actuator velocity derivative unless ACTUATION is disabled and
the damping derivative unless DAMPER is disabled. MuJoCo Warp 3.16 gates both
on ``not (ACTUATION | DAMPER)``, so a model run with native actuation off (MASS
pose control) lost its joint damping from ``M - h*D``.
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

from instantmimic.utils.mjw_model import put_model
from instantmimic.utils.mjw_step_optimized import (
    make_optimized_fwd_position_scratch,
    step_optimized_fwd_position,
)

_XML = """<mujoco><option integrator="implicitfast" timestep="0.01">{flags}</option>
<worldbody><body><joint name="a" type="hinge" axis="0 1 0" damping="30"/>
  <geom type="capsule" size="0.03 0.1" contype="0" conaffinity="0" mass="0.5"/>
  <body pos="0 0 -0.2"><joint name="b" type="hinge" axis="1 0 0" damping="12"/>
    <geom type="capsule" size="0.03 0.1" contype="0" conaffinity="0" mass="0.3"/>
  </body></body></worldbody>
<actuator><velocity joint="a" kv="8"/><velocity joint="b" kv="5"/></actuator>
</mujoco>"""


@pytest.mark.parametrize(
    "flags",
    [
        "",
        '<flag actuation="disable"/>',
        '<flag damper="disable"/>',
        '<flag actuation="disable" damper="disable"/>',
    ],
    ids=["all", "no_actuation", "no_damper", "neither"],
)
def test_implicitfast_step_matches_mujoco_under_each_disable_flag(flags):
    mjm = mujoco.MjModel.from_xml_string(_XML.format(flags=flags))
    cpu = mujoco.MjData(mjm)
    cpu.qvel[:] = [2.0, -3.0]
    cpu.ctrl[:] = [0.5, -0.4]
    mujoco.mj_forward(mjm, cpu)
    with wp.ScopedDevice("cpu"):
        m = put_model(mjm)
        d = mjw.put_data(mjm, cpu, nworld=1, nconmax=8, njmax=16)
        scratch = make_optimized_fwd_position_scratch(m, d, "cpu")
        for _ in range(5):
            step_optimized_fwd_position(
                m, d, (), scratch=scratch, use_optimized_kernel=True
            )
            mujoco.mj_step(mjm, cpu)
        np.testing.assert_allclose(d.qvel.numpy()[0], cpu.qvel, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(d.qpos.numpy()[0], cpu.qpos, rtol=1e-5, atol=1e-6)
