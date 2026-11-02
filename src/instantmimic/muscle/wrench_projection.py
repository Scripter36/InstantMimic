"""Project accumulated body wrenches onto joint torques with a tree reduction.

The legacy kernel evaluates one DOF per thread and scans every body through a
dense descendant mask, so its cost grows with ``ndof * nbody``. Here one
cooperative block per world stores each body's force and root-referenced moment
in shared memory, folds the six wrench components bottom-up along the body
tree, and finally takes one dot product per DOF with its ``cdof`` row. Bodies
at the same depth atomically add into their parents in parallel. The result
equals the masked sum up to floating-point summation order.

Topology metadata is built once per (topology, device) outside CUDA graph
capture. Models with more bodies than the largest shared-memory tile, and CPU
devices, keep the legacy kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from instantmimic.utils.warp_kernels import expose_kernel_name

from .muscle_dynamics_kernels import _project_body_wrench_to_torque_kernel

BLOCK_DIM = 64
BODY_CAPS = (64, 256)


@wp.func_native(snippet="WP_TILE_SYNC();")
def _sync():
    pass


@wp.func_native(
    snippet=r"""
__shared__ float data[64 * 6];
wp::array_t<float> result(data, nworld, 64, 6);
result.strides[0] = 0;
return result;
"""
)
def _scratch_64(nworld: int) -> wp.array3d[float]: ...


@wp.func_native(
    snippet=r"""
__shared__ float data[256 * 6];
wp::array_t<float> result(data, nworld, 256, 6);
result.strides[0] = 0;
return result;
"""
)
def _scratch_256(nworld: int) -> wp.array3d[float]: ...


def _make_body_wrench_projection_kernel(scratch):
    @wp.kernel(enable_backward=False)
    @expose_kernel_name
    def project_body_wrenches_to_dofs_kernel(
        body_force: wp.array3d[float],
        body_torque: wp.array3d[float],
        xipos: wp.array2d[wp.vec3],
        subtree_com: wp.array2d[wp.vec3],
        cdof: wp.array3d[float],
        body_rootid: wp.array[int],
        dof_bodyid: wp.array[int],
        body_parentid: wp.array[int],
        level_bodies: wp.array[int],
        level_offsets: wp.array[int],
        levels: int,
        body_count: int,
        root_dof_count: int,
        torque_dim: int,
        output_dof_offset: int,
        mean: wp.array2d[float],
    ):
        world, lane = wp.tid()
        values = wp.static(scratch)(mean.shape[0])
        for body in range(lane, body_count, wp.block_dim()):
            force = wp.vec3(
                body_force[world, body, 0],
                body_force[world, body, 1],
                body_force[world, body, 2],
            )
            moment = wp.vec3(
                body_torque[world, body, 0],
                body_torque[world, body, 1],
                body_torque[world, body, 2],
            )
            moment += wp.cross(
                xipos[world, body] - subtree_com[world, body_rootid[body]], force
            )
            for component in range(3):
                values[world, body, component] = force[component]
                values[world, body, component + 3] = moment[component]
        _sync()
        # Fold each non-root level into its parents, deepest first. Sources are
        # complete before the level starts; atomics resolve siblings that share
        # a parent without serializing the child traversal.
        for level in range(levels - 1, 0, -1):
            start = level_offsets[level]
            count = level_offsets[level + 1] - start
            for entry in range(lane, count * 6, wp.block_dim()):
                body = level_bodies[start + entry // 6]
                component = entry % 6
                wp.atomic_add(
                    values,
                    world,
                    body_parentid[body],
                    component,
                    values[world, body, component],
                )
            _sync()
        for torque_idx in range(lane, torque_dim, wp.block_dim()):
            dof = torque_idx + root_dof_count
            body = dof_bodyid[dof]
            force = wp.vec3(
                values[world, body, 0], values[world, body, 1], values[world, body, 2]
            )
            moment = wp.vec3(
                values[world, body, 3], values[world, body, 4], values[world, body, 5]
            )
            linear_cdof = wp.vec3(
                cdof[world, dof, 3], cdof[world, dof, 4], cdof[world, dof, 5]
            )
            angular_cdof = wp.vec3(
                cdof[world, dof, 0], cdof[world, dof, 1], cdof[world, dof, 2]
            )
            mean[world, torque_idx + output_dof_offset] = wp.dot(
                linear_cdof, force
            ) + wp.dot(angular_cdof, moment)

    return project_body_wrenches_to_dofs_kernel


_PROJECT_KERNELS = {
    64: _make_body_wrench_projection_kernel(_scratch_64),
    256: _make_body_wrench_projection_kernel(_scratch_256),
}


@dataclass(frozen=True)
class WrenchProjectionTopology:
    """Device-resident body-tree ordering for the cooperative projection."""

    dof_bodyid: wp.array
    body_parentid: wp.array
    level_bodies: wp.array
    level_offsets: wp.array
    levels: int
    body_count: int
    body_cap: int


def body_cap_for(body_count: int) -> int | None:
    """Return the smallest shared-memory tile that holds ``body_count`` bodies."""
    for cap in BODY_CAPS:
        if body_count <= cap:
            return cap
    return None


def tree_levels(
    body_parentid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return depth-sorted bodies, depth offsets, and a child list for a tree."""
    parents = np.asarray(body_parentid, dtype=np.int32)
    depth = np.zeros(parents.shape[0], dtype=np.int32)
    for body in range(1, parents.shape[0]):
        parent = int(parents[body])
        if parent >= body:
            raise ValueError("body_parentid must reference an earlier body")
        depth[body] = depth[parent] + 1
    order = np.argsort(depth, kind="stable").astype(np.int32)
    counts = np.bincount(depth, minlength=int(depth.max()) + 1 if depth.size else 1)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    children = np.arange(1, parents.shape[0], dtype=np.int32)
    child_bodies = children[np.argsort(parents[1:], kind="stable")]
    child_counts = np.bincount(parents[1:], minlength=parents.shape[0])
    child_offsets = np.concatenate([[0], np.cumsum(child_counts)]).astype(np.int32)
    return order, offsets, child_bodies, child_offsets


def build_wrench_projection_topology(
    body_parentid: torch.Tensor,
    dof_bodyid: torch.Tensor,
    body_count: int,
    device: torch.device | str,
) -> WrenchProjectionTopology | None:
    """Build projection metadata, or ``None`` when the legacy kernel must run."""
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return None
    body_cap = body_cap_for(int(body_count))
    if body_cap is None:
        return None
    wp_device = wp.get_device(str(torch_device))
    if wp_device.is_capturing:
        raise RuntimeError(
            "wrench projection topology must be prepared before graph capture"
        )
    parents = body_parentid.detach().cpu().numpy().astype(np.int32)
    order, offsets, _, _ = tree_levels(parents)
    return WrenchProjectionTopology(
        dof_bodyid=wp.array(
            dof_bodyid.detach().cpu().numpy().astype(np.int32),
            dtype=int,
            device=wp_device,
        ),
        body_parentid=wp.array(parents, dtype=int, device=wp_device),
        level_bodies=wp.array(order, dtype=int, device=wp_device),
        level_offsets=wp.array(offsets, dtype=int, device=wp_device),
        levels=int(offsets.shape[0] - 1),
        body_count=int(body_count),
        body_cap=body_cap,
    )


def launch_project_body_wrench_to_torque(
    topology: WrenchProjectionTopology | None,
    *,
    body_force: torch.Tensor,
    body_torque: torch.Tensor,
    xipos: torch.Tensor,
    subtree_com: torch.Tensor,
    cdof: torch.Tensor,
    body_rootid: torch.Tensor,
    dof_subtree_mask: torch.Tensor,
    mean: torch.Tensor,
    body_count: int,
    torque_dim: int,
    root_dof_count: int,
    output_dof_offset: int,
    batch_size: int,
    device: str,
) -> None:
    """Project body wrenches onto ``mean``; use the tree kernel when eligible."""
    if topology is None:
        wp.launch(
            _project_body_wrench_to_torque_kernel,
            dim=(batch_size, torque_dim),
            inputs=[
                wp.from_torch(body_force, dtype=wp.float32),
                wp.from_torch(body_torque, dtype=wp.float32),
                wp.from_torch(xipos, dtype=wp.vec3),
                wp.from_torch(subtree_com, dtype=wp.vec3),
                wp.from_torch(cdof, dtype=wp.float32),
                wp.from_torch(body_rootid, dtype=wp.int32),
                wp.from_torch(dof_subtree_mask, dtype=wp.float32),
                body_count,
                root_dof_count,
                int(output_dof_offset),
            ],
            outputs=[wp.from_torch(mean, dtype=wp.float32)],
            device=device,
        )
        return
    wp.launch_tiled(
        _PROJECT_KERNELS[topology.body_cap],
        dim=batch_size,
        block_dim=BLOCK_DIM,
        inputs=[
            wp.from_torch(body_force, dtype=wp.float32),
            wp.from_torch(body_torque, dtype=wp.float32),
            wp.from_torch(xipos, dtype=wp.vec3),
            wp.from_torch(subtree_com, dtype=wp.vec3),
            wp.from_torch(cdof, dtype=wp.float32),
            wp.from_torch(body_rootid, dtype=wp.int32),
            topology.dof_bodyid,
            topology.body_parentid,
            topology.level_bodies,
            topology.level_offsets,
            topology.levels,
            body_count,
            root_dof_count,
            torque_dim,
            int(output_dof_offset),
        ],
        outputs=[wp.from_torch(mean, dtype=wp.float32)],
        device=device,
    )
