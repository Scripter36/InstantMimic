"""Sampling kernels shared across RL algorithms."""

import warp as wp


@wp.kernel(enable_backward=False)
def increment_seed_kernel(seed: wp.array(dtype=wp.int32)):
    seed[0] = seed[0] + wp.int32(1)


@wp.kernel(enable_backward=False)
def fill_gaussian_noise_kernel(
    seed: wp.array(dtype=wp.int32),
    width: int,
    out: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    state = wp.rand_init(seed[0], i * width + j)
    out[i, j] = wp.randn(state)
