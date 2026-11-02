"""Bottom-up rigid-body reductions with one cooperative block per world.

Copyright 2025 The Newton Developers, licensed under the Apache License,
Version 2.0. Reproduces the stage sequence of mujoco_warp smooth.py.
Local changes walk ``m.body_tree`` inside a single cooperative launch instead of
one launch per tree level.

Topology belongs to the model's persistent scratch; no process-global state or
allocation occurs during forward, reset, or CUDA graph capture.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from instantmimic.utils.mjw_kernels.private import smooth
from instantmimic.utils.warp_kernels import expose_kernel_name


@wp.func_native(snippet="WP_TILE_SYNC();")
def sync():
    pass


@dataclass(frozen=True)
class TreeReductionScratch:
    bodies: wp.array
    offsets: wp.array
    levels: int
    deterministic: bool = False
    cooperative: bool = True


def create_tree_reduction_scratch(m, *, deterministic=False, cooperative=True):
    if not m.body_tree:
        return None
    levels = [level.numpy() for level in m.body_tree]
    device = m.body_parentid.device
    return TreeReductionScratch(
        wp.array(np.concatenate(levels).astype(np.int32), dtype=int, device=device),
        wp.array(
            np.cumsum([0] + [len(level) for level in levels]).astype(np.int32),
            dtype=int,
            device=device,
        ),
        len(levels),
        deterministic,
        cooperative,
    )


@wp.kernel(enable_backward=False)
@expose_kernel_name
def accumulate_body_tree_kernel(
    parent: wp.array[int],
    bodies: wp.array[int],
    offsets: wp.array[int],
    levels: int,
    components: int,
    skip_root_parent: bool,
    values: wp.array3d[float],
):
    world, lane = wp.tid()
    for level in range(levels - 1, -1, -1):
        start = offsets[level]
        count = offsets[level + 1] - start
        for entry in range(lane, count * components, wp.block_dim()):
            body = bodies[start + entry // components]
            component = entry % components
            parent_id = parent[body]
            if body != 0 and (not skip_root_parent or parent_id != 0):
                wp.atomic_add(
                    values, world, parent_id, component, values[world, body, component]
                )
        sync()


@wp.kernel(enable_backward=False)
@expose_kernel_name
def accumulate_body_tree_deterministic_kernel(
    parent: wp.array[int],
    bodies: wp.array[int],
    offsets: wp.array[int],
    levels: int,
    components: int,
    skip_root_parent: bool,
    values: wp.array3d[float],
):
    """One writer per parent/component, adding children in topology order.

    Adjacent levels contain disjoint bodies, so readers never race writers.
    The level barrier makes each completed parent available to its parent.
    """
    world, lane = wp.tid()
    for level in range(levels - 1, 0, -1):
        start, end = offsets[level], offsets[level + 1]
        parent_start = offsets[level - 1]
        parent_count = offsets[level] - parent_start
        for entry in range(lane, parent_count * components, wp.block_dim()):
            parent_id = bodies[parent_start + entry // components]
            component = entry % components
            if not skip_root_parent or parent_id != 0:
                value = values[world, parent_id, component]
                for child_entry in range(start, end):
                    body = bodies[child_entry]
                    if body != 0 and parent[body] == parent_id:
                        value += values[world, body, component]
                values[world, parent_id, component] = value
        sync()


@wp.kernel(enable_backward=False)
def accumulate_body_tree_level_deterministic_kernel(
    parent: wp.array[int],
    bodies: wp.array[int],
    offsets: wp.array[int],
    level: int,
    components: int,
    skip_root_parent: bool,
    values: wp.array3d[float],
):
    world, entry = wp.tid()
    parent_count = offsets[level] - offsets[level - 1]
    if entry >= parent_count * components:
        return
    parent_id = bodies[offsets[level - 1] + entry // components]
    component = entry % components
    if not skip_root_parent or parent_id != 0:
        value = values[world, parent_id, component]
        for child_entry in range(offsets[level], offsets[level + 1]):
            body = bodies[child_entry]
            if body != 0 and parent[body] == parent_id:
                value += values[world, body, component]
        values[world, parent_id, component] = value


def accumulate(m, d, values, components, skip_root_parent, scratch):
    if scratch.deterministic and not scratch.cooperative:
        for level in range(scratch.levels - 1, 0, -1):
            wp.launch(
                accumulate_body_tree_level_deterministic_kernel,
                dim=(d.nworld, values.shape[1] * components),
                inputs=[
                    m.body_parentid,
                    scratch.bodies,
                    scratch.offsets,
                    level,
                    components,
                    skip_root_parent,
                    values.view(wp.float32),
                ],
            )
        return
    wp.launch_tiled(
        accumulate_body_tree_deterministic_kernel
        if scratch.deterministic
        else accumulate_body_tree_kernel,
        dim=d.nworld,
        block_dim=64,
        inputs=[
            m.body_parentid,
            scratch.bodies,
            scratch.offsets,
            scratch.levels,
            components,
            skip_root_parent,
            values.view(wp.float32),
        ],
    )


def subtree_com_accumulate(m, d, scratch=None):
    if scratch is not None:
        accumulate(m, d, d.subtree_com, 3, False, scratch)
    else:
        for body_tree in reversed(m.body_tree):
            wp.launch(
                smooth._subtree_com_acc,
                dim=(d.nworld, body_tree.size),
                inputs=[m.body_parentid, d.subtree_com, body_tree],
                outputs=[d.subtree_com],
            )


def crb(m, d, scratch=None):
    if scratch is None:
        smooth.crb(m, d)
        return
    wp.copy(d.crb, d.cinert)
    accumulate(m, d, d.crb, 10, True, scratch)
    d.M.zero_()
    wp.launch(
        smooth._M,
        dim=(d.nworld, m.nv),
        inputs=[
            m.dof_bodyid,
            m.dof_parentid,
            m.dof_armature,
            m.M_rownnz,
            m.M_rowadr,
            d.cdof,
            d.crb,
        ],
        outputs=[d.M],
    )


def rne(m, d, scratch=None):
    if scratch is None:
        smooth.rne(m, d)
        return
    smooth._rne_cacc_world(m, d)
    smooth._rne_cacc_forward(m, d, flg_acc=False)
    smooth._rne_cfrc(m, d)
    accumulate(m, d, d.cfrc_int, 6, False, scratch)
    wp.launch(
        smooth._qfrc_bias,
        dim=(d.nworld, m.nv),
        inputs=[m.dof_bodyid, d.cdof, d.cfrc_int],
        outputs=[d.qfrc_bias],
    )
