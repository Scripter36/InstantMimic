"""Body rotation and inertial frames sharing the same position/quaternion reads."""

import warp as wp

from instantmimic.utils.mjw_kernels.private import mul_quat, quat_to_mat, rot_vec_quat


@wp.kernel(enable_backward=False)
def body_frames_kernel(
    body_ipos: wp.array2d(dtype=wp.vec3),
    body_iquat: wp.array2d(dtype=wp.quat),
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    xmat: wp.array2d(dtype=wp.mat33),
    xipos: wp.array2d(dtype=wp.vec3),
    ximat: wp.array2d(dtype=wp.mat33),
):
    world, body = wp.tid()
    rotation = xquat[world, body]
    xmat[world, body] = quat_to_mat(rotation)
    xipos[world, body] = xpos[world, body] + rot_vec_quat(
        body_ipos[world % body_ipos.shape[0], body], rotation
    )
    ximat[world, body] = quat_to_mat(
        mul_quat(rotation, body_iquat[world % body_iquat.shape[0], body])
    )
