"""Cooperative velocity propagation preserves joint/depth and graph semantics."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

from instantmimic.utils.mjw_kernels.private import smooth
from instantmimic.utils.mjw_kernels.tree_fusion import create_tree_reduction_scratch
from instantmimic.utils.mjw_kernels.velocity_fusion import com_vel


@pytest.mark.cuda
def test_body_velocities_match_native_joint_branches_and_replay():
    model_cpu = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body><freejoint/><geom type="sphere" size="0.1"/>
      <body pos="0 0 0.2"><joint type="ball"/><geom type="sphere" size="0.1"/>
        <body pos="0 0 0.2"><joint type="hinge"/><joint type="slide" axis="1 0 0"/><geom type="sphere" size="0.1"/></body>
      </body>
      <body pos="0.2 0 0"><geom type="sphere" size="0.1"/>
        <body pos="0.2 0 0"><joint type="hinge"/><geom type="sphere" size="0.1"/></body>
      </body>
    </body>
    <body pos="1 0 0"><freejoint/><geom type="sphere" size="0.1"/></body>
    </worldbody></mujoco>""")
    with wp.ScopedDevice("cuda"):
        model = mjw.put_model(model_cpu)
        data = mjw.put_data(
            model_cpu, mujoco.MjData(model_cpu), nworld=3, nconmax=32, njmax=64
        )
        scratch = create_tree_reduction_scratch(model)
        smooth.kinematics(model, data)
        smooth.com_pos(model, data)
        rng = np.random.default_rng(22)
        for capture in (False, True):
            data.qvel.assign(rng.normal(size=data.qvel.shape).astype(np.float32))
            smooth.com_vel(model, data)
            reference = [data.cvel.numpy(), data.cdof_dot.numpy()]
            data.cvel.fill_(wp.spatial_vector(123.0))
            data.cdof_dot.fill_(wp.spatial_vector(123.0))
            if capture:
                with wp.ScopedCapture() as cap:
                    com_vel(model, data, scratch)
                wp.capture_launch(cap.graph)
            else:
                com_vel(model, data, scratch)
            for actual, expected in zip(
                (data.cvel, data.cdof_dot), reference, strict=True
            ):
                np.testing.assert_allclose(
                    actual.numpy(), expected, rtol=1e-6, atol=1e-6
                )
