"""Fused body fields retain the pinned native arithmetic and broadcast layout."""

import numpy as np
import pytest
import warp as wp

from instantmimic.utils.mjw_kernels.kinematics_fusion import body_frames_kernel
from instantmimic.utils.mjw_kernels.private import smooth


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("model_worlds", [1, 3])
def test_body_frames_match_native_stages_and_replay(device, model_worlds):
    rng = np.random.default_rng(51)
    shape = (3, 9)

    def quaternions(n):
        q = rng.normal(size=(n, shape[1], 4)).astype(np.float32)
        return wp.array(
            q / np.linalg.norm(q, axis=-1, keepdims=True), dtype=wp.quat, device=device
        )

    positions = wp.array(rng.normal(size=(*shape, 3)), dtype=wp.vec3, device=device)
    rotations = quaternions(shape[0])
    ipos = wp.array(
        rng.normal(size=(model_worlds, shape[1], 3)), dtype=wp.vec3, device=device
    )
    iquat = quaternions(model_worlds)
    reference = [
        wp.empty(shape, dtype=t, device=device) for t in (wp.mat33, wp.vec3, wp.mat33)
    ]
    actual = [wp.empty_like(a) for a in reference]

    def native():
        wp.launch(
            smooth._compute_body_matrices,
            shape,
            [rotations],
            [reference[0]],
            device=device,
        )
        wp.launch(
            smooth._compute_body_inertial_frames,
            shape,
            [ipos, iquat, positions, rotations],
            reference[1:],
            device=device,
        )

    def fused():
        wp.launch(
            body_frames_kernel,
            shape,
            [ipos, iquat, positions, rotations],
            actual,
            device=device,
        )

    native()
    fused()
    graph = None
    if device == "cuda":
        with wp.ScopedCapture(device=device) as capture:
            fused()
        graph = capture.graph
    for shift in (0.0, 0.3):
        positions.fill_(wp.vec3(shift, 0.2, -0.1))
        native()
        if graph is None:
            fused()
        else:
            wp.capture_launch(graph)
        for a, b in zip(actual, reference, strict=True):
            if device == "cpu":
                np.testing.assert_array_equal(a.numpy(), b.numpy())
            else:
                # Fusion can change Float32 rounding on GPU. Bound each
                # field by two epsilons; full integrator checks remain strict.
                epsilon = 2 * np.finfo(np.float32).eps
                np.testing.assert_allclose(
                    a.numpy(), b.numpy(), atol=epsilon, rtol=epsilon
                )
