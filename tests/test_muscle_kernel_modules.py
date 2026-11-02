"""Keep tiled launch specializations out of the shared scalar module."""

import pytest

from instantmimic.muscle import muscle_dynamics_kernels as kernels


def test_tiled_kernels_do_not_specialize_the_scalar_module():
    scalar_module = kernels._normalize_body_quats_kernel.module
    tiled = (
        kernels._accumulate_lbs_body_wrench_segment_event_chunk_partials_kernel,
        kernels._accumulate_lbs_body_wrench_segment_start_event_chunk_partials_kernel,
        kernels._compute_lbs_torque_bounds_tile_partials_kernel,
    )
    tiled_module = tiled[0].module
    assert tiled_module is not scalar_module
    assert all(kernel.module is tiled_module for kernel in tiled)
    assert all(kernel not in scalar_module.kernels.values() for kernel in tiled)


@pytest.mark.parametrize("include_passive", [False, True])
@pytest.mark.parametrize("block_dim", [32, 128])
def test_generated_tiled_kernels_share_the_tiled_compilation_boundary(
    include_passive, block_dim
):
    tiled_module = kernels._compute_lbs_torque_bounds_tile_partials_kernel.module
    scalar_module = kernels._normalize_body_quats_kernel.module
    generated = [
        kernels.make_accumulate_lbs_body_wrench_native_tile_full_force_fixed1_kernel(
            include_passive, block_dim=block_dim
        ),
        kernels.make_accumulate_lbs_body_wrench_native_tile_full_force_fixed2_kernel(
            include_passive, block_dim=block_dim
        ),
        kernels.make_accumulate_lbs_body_wrench_native_tile_full_force_velocity_fixed1_kernel(
            include_passive, block_dim=block_dim
        ),
        kernels.make_accumulate_lbs_body_wrench_native_tile_cached_force_fixed1_kernel(
            block_dim=block_dim
        ),
        kernels.make_accumulate_lbs_body_wrench_native_tile_cached_force_fixed2_kernel(
            block_dim=block_dim, direct_force=True
        ),
    ]
    for root in (-1, 0):
        generated.append(
            kernels.make_lbs_torque_waypoint_force_active_attach_pair_vec4_native_partials_kernel(
                include_passive, root
            )
        )
    for kernel in generated:
        assert kernel.module is tiled_module
        assert kernel.module is not scalar_module
        assert kernel.options["enable_backward"] is False
