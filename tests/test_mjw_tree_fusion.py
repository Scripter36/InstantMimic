"""Captured bottom-up reductions on topology shapes independent of task assets."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from instantmimic.utils.mjw_kernels import tree_fusion as fusion


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["atomic", "ordered_cooperative", "ordered_levels"])
@pytest.mark.parametrize("topology", ["wide", "deep", "multiple_roots"])
@pytest.mark.parametrize("components,skip_root", [(3, False), (6, False), (10, True)])
def test_tree_reduction_matches_depth_ordered_reference(
    topology, components, skip_root, mode
):
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    count = 97
    if topology == "wide":
        parents = np.zeros(count, dtype=np.int32)
    elif topology == "deep":
        parents = np.maximum(np.arange(count, dtype=np.int32) - 1, 0)
    else:
        parents = np.maximum(np.arange(count, dtype=np.int32) - 3, 0)
    depths = np.zeros(count, dtype=np.int32)
    for body in range(1, count):
        depths[body] = depths[parents[body]] + 1
    levels = [
        np.flatnonzero(depths == depth).astype(np.int32)
        for depth in range(int(depths.max()) + 1)
    ]
    initial = (
        np.random.default_rng(7)
        .uniform(0.1, 1.0, (3, count, components))
        .astype(np.float32)
    )
    reference = initial.copy()
    for level in reversed(levels):
        for body in level:
            parent = parents[body]
            if body and (not skip_root or parent):
                reference[:, parent, :] += reference[:, body, :]
    with wp.ScopedDevice("cuda"):
        model = SimpleNamespace(
            body_parentid=wp.array(parents, dtype=int),
            body_tree=[wp.array(level, dtype=int) for level in levels],
        )
        scratch = fusion.create_tree_reduction_scratch(
            model,
            deterministic=mode != "atomic",
            cooperative=mode != "ordered_levels",
        )
        data = SimpleNamespace(nworld=3)
        values = wp.array(initial, dtype=float)
        fusion.accumulate(model, data, values, components, skip_root, scratch)
        values.assign(initial)
        with wp.ScopedCapture() as capture:
            fusion.accumulate(model, data, values, components, skip_root, scratch)
        values.assign(initial)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(values.numpy(), reference, rtol=2e-6, atol=2e-6)
        if mode != "atomic":
            np.testing.assert_array_equal(values.numpy(), reference)
            for _ in range(3):
                values.assign(initial)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(values.numpy(), reference)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["atomic", "ordered_cooperative", "ordered_levels"])
def test_fused_wrappers_match_upstream_stages_on_vector_dtypes(mode):
    """Exercise the vec3/vec10/spatial_vector reinterpretation on a real model."""
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    import mujoco
    import mujoco_warp as mjw
    from mujoco_warp._src import smooth

    def link(depth):
        if depth == 0:
            return ""
        child = link(depth - 1)
        return (
            '<body pos="0 0 0.1"><joint type="ball"/>'
            '<geom type="capsule" size="0.02 0.04" mass="0.4"/>'
            f"{child}{child}</body>"
        )

    xml = (
        "<mujoco><worldbody>"
        '<body pos="0 0 1"><freejoint/><geom type="sphere" size="0.05" mass="1"/>'
        f"{link(3)}</body>"
        '<body pos="1 0 1"><freejoint/><geom type="sphere" size="0.05" mass="2"/>'
        f"{link(2)}</body>"
        "</worldbody></mujoco>"
    )
    cpu = mujoco.MjModel.from_xml_string(xml)
    with wp.ScopedDevice("cuda"):
        m = mjw.put_model(cpu)
        d = mjw.put_data(cpu, mujoco.MjData(cpu), nworld=4)
        rng = np.random.default_rng(3)
        d.qvel.assign(rng.normal(0, 0.5, (d.nworld, m.nv)).astype(np.float32))
        mjw.forward(m, d)
        scratch = fusion.create_tree_reduction_scratch(
            m,
            deterministic=mode != "atomic",
            cooperative=mode != "ordered_levels",
        )
        assert scratch is not None and scratch.levels == len(m.body_tree)

        def run(stage, fused):
            outputs = []
            for use_fused in (False, True):
                if stage == "com":
                    wp.launch(
                        smooth._subtree_com_init,
                        dim=(d.nworld, m.nbody),
                        inputs=[m.body_mass, d.xipos],
                        outputs=[d.subtree_com],
                    )
                    fusion.subtree_com_accumulate(m, d, scratch if use_fused else None)
                    outputs.append([d.subtree_com.numpy().copy()])
                elif stage == "crb":
                    fused(m, d, scratch if use_fused else None)
                    outputs.append([d.crb.numpy().copy(), d.M.numpy().copy()])
                else:
                    fused(m, d, scratch if use_fused else None)
                    outputs.append(
                        [d.cfrc_int.numpy().copy(), d.qfrc_bias.numpy().copy()]
                    )
            return outputs

        for stage, fused in (("com", None), ("crb", fusion.crb), ("rne", fusion.rne)):
            reference, candidate = run(stage, fused)
            for ref, cand in zip(reference, candidate, strict=False):
                assert np.abs(ref).max() > 0.0
                np.testing.assert_allclose(cand, ref, rtol=1e-5, atol=1e-5)


def test_ordered_level_reduction_matches_float32_cancellation_order_on_cpu():
    parents = np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int32)
    levels = [np.array(x, dtype=np.int32) for x in ([0], [1, 2, 3], [4, 5, 6, 7])]
    initial = np.array([[[1], [1e8], [1], [-1e8], [1], [-1], [2], [3]]], np.float32)
    expected = initial.copy()
    for level in reversed(levels):
        for body in level:
            if body:
                expected[:, parents[body]] += expected[:, body]
    model = SimpleNamespace(
        nbody=len(parents),
        body_parentid=wp.array(parents, dtype=int, device="cpu"),
        body_tree=[wp.array(x, dtype=int, device="cpu") for x in levels],
    )
    scratch = fusion.create_tree_reduction_scratch(
        model, deterministic=True, cooperative=False
    )
    values = wp.array(initial, dtype=float, device="cpu")
    fusion.accumulate(model, SimpleNamespace(nworld=1), values, 1, False, scratch)
    np.testing.assert_array_equal(values.numpy(), expected)


def test_reference_tree_selection_preserves_legacy_kernel_flag():
    from instantmimic.utils.mjw_step_optimized import _select_tree_reduction

    atomic = SimpleNamespace(deterministic=False)
    ordered = SimpleNamespace(deterministic=True, cooperative=False)
    assert _select_tree_reduction(SimpleNamespace(tree_reduction=atomic), False) is None
    assert (
        _select_tree_reduction(SimpleNamespace(tree_reduction=atomic), True) is atomic
    )
    assert (
        _select_tree_reduction(SimpleNamespace(tree_reduction=ordered), False)
        is ordered
    )
    assert _select_tree_reduction(None, True) is None
