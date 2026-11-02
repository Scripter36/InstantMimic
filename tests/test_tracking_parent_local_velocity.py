"""Parent-local velocity tracking is invariant to rigid root rotation."""

from dataclasses import MISSING, fields

import mujoco
import numpy as np
import pytest
import warp as wp

from instantmimic.envs.tracking.reward_kernels import (
    RewardKernelSpec,
    launch_compute_rewards,
    make_compute_rewards_kernel,
)


@pytest.mark.parametrize("sim_spin,ref_spin", [(1.0, 0.0), (0.0, 1.0)])
@pytest.mark.parametrize("frame", ["parent_local", "world"])
@pytest.mark.parametrize("relative_spin", [0.0, 0.7])
@pytest.mark.parametrize("loop_cycle", [0, 1])
def test_rigid_root_rotation_preserves_parent_local_linear_velocity(
    sim_spin, ref_spin, frame, relative_spin, loop_cycle
):
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/>"
        '<geom type="sphere" size=".1" mass="1"/>'
        '<body pos="1 0 0"><joint type="hinge" axis="0 0 1" pos=".2 0 0"/>'
        '<geom type="sphere" size=".1" mass="1"/>'
        "</body></body></worldbody></mujoco>"
    )
    sim, ref = mujoco.MjData(model), mujoco.MjData(model)
    sim.qvel[5], ref.qvel[5] = sim_spin, ref_spin
    sim.qvel[6], ref.qvel[6] = relative_spin, -relative_spin
    mujoco.mj_forward(model, sim)
    mujoco.mj_forward(model, ref)
    # Zero unused reward components, retaining the production specification's
    # defaults for all optional numerical and compatibility settings.
    spec_values = {
        field.name: 0
        for field in fields(RewardKernelSpec)
        if field.default is MISSING and field.default_factory is MISSING
    }
    spec_values.update(
        motion_frame_count=1,
        nv=model.nv,
        root_body_idx=1,
        ee_body_idx=(),
        reward_body_idx=(2,),
        body_parent_idx=(0, 0, 1),
        hinge_body_idx=(),
        hinge_parent_body_idx=(),
        hinge_axis_local=(),
        hinge_multi_body_mask=(),
        body_state_reward=True,
        body_state_reward_frame_mode=int(frame == "parent_local"),
        vel_valid_all=True,
        loop_root_q=(1.0, 0.0, 0.0, 0.0),
        loop_root_t=(0.0, 0.0, 0.0),
        reward_combination_mode=2,
        body_state_linvel_reward_weight=1.0,
    )
    kernel = make_compute_rewards_kernel(RewardKernelSpec(**spec_values))

    def array(value, dtype=wp.float32):
        return wp.array(np.asarray(value), dtype=dtype, device="cpu")

    ref_state = np.concatenate(
        [ref.xpos, ref.xquat, ref.cvel[:, 3:], ref.cvel[:, :3]], axis=1
    )[None]
    index = array([0], wp.int32)
    empty = array([], wp.int32)
    inputs = [
        array(sim.qpos[None]),
        array(sim.qvel[None]),
        array(sim.xpos[None], wp.vec3),
        array(sim.xquat[None], wp.quat),
        array(sim.cvel[None], wp.spatial_vector),
        array(sim.subtree_com[None], wp.vec3),
        array(ref.qpos[None]),
        array(ref.qvel[None]),
        array(ref_state),
        array(ref.subtree_com[1:2], wp.vec3),
        index,
        array([loop_cycle], wp.int32),
        index,
        array([1], wp.int32),
        array([[2**-0.5, 0.0, 0.0, 2**-0.5]]),
        array([[3.0, 4.0, 0.0]]),
        empty,
        empty,
        empty,
        empty,
        array(np.ones(model.nv)),
        array(np.ones(model.nv)),
        array(np.ones((1, 4))),
        array([0.0]),
    ]
    terms = wp.zeros((1, 7), dtype=float, device="cpu")
    reward = wp.zeros(1, dtype=float, device="cpu")
    launch_compute_rewards(
        kernel, dim=(1,), inputs=inputs, outputs=[terms, reward], device="cpu"
    )

    def local_point_velocity(data):
        velocities = []
        for body in (1, 2):
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0
            )
            velocities.append(velocity)
        parent, child = velocities
        relative = (
            child[3:] - parent[3:] - np.cross(parent[:3], data.xpos[2] - data.xpos[1])
        )
        return data.xmat[1].reshape(3, 3).T @ relative

    if frame == "parent_local":
        # Independent MuJoCo point-velocity oracle, including an off-origin
        # hinge. A global loop transform cannot change parent-local velocity.
        difference = local_point_velocity(sim) - local_point_velocity(ref)
    else:
        # Preserve the existing world-frame cvel convention and loop rotation.
        ref_velocity = ref.cvel[2, 3:]
        if loop_cycle:
            ref_velocity = np.array(
                [-ref_velocity[1], ref_velocity[0], ref_velocity[2]]
            )
        difference = sim.cvel[2, 3:] - ref_velocity
    expected = -np.square(difference).mean()
    assert terms.numpy()[0, 2] == pytest.approx(expected, abs=1e-6)
