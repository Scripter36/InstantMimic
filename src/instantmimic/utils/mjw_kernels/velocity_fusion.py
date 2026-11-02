"""Depth-ordered body velocities with one cooperative block per world.

Derived from mujoco_warp smooth.py, Copyright 2025 The Newton Developers,
licensed under the Apache License, Version 2.0. Joint arithmetic is retained;
only independent bodies at a common tree depth execute concurrently.
"""

import warp as wp

from instantmimic.utils.mjw_kernels.private import JointType, math, smooth
from instantmimic.utils.mjw_kernels.tree_fusion import sync


@wp.kernel(enable_backward=False)
def body_velocity_kernel(
    body_parentid: wp.array[int],
    body_jntnum: wp.array[int],
    body_jntadr: wp.array[int],
    body_dofadr: wp.array[int],
    jnt_type: wp.array[int],
    bodies: wp.array[int],
    offsets: wp.array[int],
    levels: int,
    qvel_in: wp.array2d[float],
    cdof_in: wp.array2d[wp.spatial_vector],
    cvel_out: wp.array2d[wp.spatial_vector],
    cdof_dot_out: wp.array2d[wp.spatial_vector],
):
    worldid, lane = wp.tid()
    qvel = qvel_in[worldid]
    cdof = cdof_in[worldid]
    for level in range(levels):
        start, end = offsets[level], offsets[level + 1]
        for entry in range(start + lane, end, wp.block_dim()):
            bodyid = bodies[entry]
            if bodyid == 0:
                cvel_out[worldid, bodyid] = wp.spatial_vector()
            else:
                pid = body_parentid[bodyid]
                cvel = cvel_out[worldid, pid]
                dofid = body_dofadr[bodyid]
                jntid = body_jntadr[bodyid]
                jntnum = body_jntnum[bodyid]

                for j in range(jntid, jntid + jntnum):
                    jnttype = jnt_type[j]

                    if jnttype == JointType.FREE:
                        cvel += cdof[dofid + 0] * qvel[dofid + 0]
                        cvel += cdof[dofid + 1] * qvel[dofid + 1]
                        cvel += cdof[dofid + 2] * qvel[dofid + 2]

                        cdof_dot_out[worldid, dofid + 0] = wp.spatial_vector(
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        )
                        cdof_dot_out[worldid, dofid + 1] = wp.spatial_vector(
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        )
                        cdof_dot_out[worldid, dofid + 2] = wp.spatial_vector(
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        )
                        cdof_dot_out[worldid, dofid + 3] = math.motion_cross(
                            cvel, cdof[dofid + 3]
                        )
                        cdof_dot_out[worldid, dofid + 4] = math.motion_cross(
                            cvel, cdof[dofid + 4]
                        )
                        cdof_dot_out[worldid, dofid + 5] = math.motion_cross(
                            cvel, cdof[dofid + 5]
                        )

                        cvel += cdof[dofid + 3] * qvel[dofid + 3]
                        cvel += cdof[dofid + 4] * qvel[dofid + 4]
                        cvel += cdof[dofid + 5] * qvel[dofid + 5]

                        dofid += 6
                    elif jnttype == JointType.BALL:
                        cdof_dot_out[worldid, dofid + 0] = math.motion_cross(
                            cvel, cdof[dofid + 0]
                        )
                        cdof_dot_out[worldid, dofid + 1] = math.motion_cross(
                            cvel, cdof[dofid + 1]
                        )
                        cdof_dot_out[worldid, dofid + 2] = math.motion_cross(
                            cvel, cdof[dofid + 2]
                        )

                        cvel += cdof[dofid + 0] * qvel[dofid + 0]
                        cvel += cdof[dofid + 1] * qvel[dofid + 1]
                        cvel += cdof[dofid + 2] * qvel[dofid + 2]

                        dofid += 3
                    else:
                        cdof_dot_out[worldid, dofid] = math.motion_cross(
                            cvel, cdof[dofid]
                        )
                        cvel += cdof[dofid] * qvel[dofid]

                        dofid += 1

                cvel_out[worldid, bodyid] = cvel
        sync()


def com_vel(m, d, tree):
    if tree is None:
        smooth.com_vel(m, d)
        return
    wp.launch_tiled(
        body_velocity_kernel,
        dim=d.nworld,
        inputs=[
            m.body_parentid,
            m.body_jntnum,
            m.body_jntadr,
            m.body_dofadr,
            m.jnt_type,
            tree.bodies,
            tree.offsets,
            tree.levels,
            d.qvel,
            d.cdof,
        ],
        outputs=[d.cvel, d.cdof_dot],
        block_dim=32,
    )
