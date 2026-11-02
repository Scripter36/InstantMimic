"""Persistent controller operations preserve masked and strided data exactly."""

import numpy as np
import pytest
import torch

from instantmimic.envs.controls.fast_muscle.buffer_ops import (
    copy_2d_,
    masked_zero_2d_,
    zero_,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_native_masked_zero_keeps_unselected_world_and_strides(dtype):
    data = torch.ones((3, 8), dtype=dtype)
    view = data[:, ::2]
    masked_zero_2d_(view, torch.tensor([True, False, True]))
    expected = torch.ones_like(data)
    expected[[0, 2], ::2] = 0
    np.testing.assert_array_equal(data.numpy(), expected.numpy())


def test_native_copy_handles_both_strides_and_float_cast():
    source = torch.arange(24, dtype=torch.int32).reshape(3, 8)[:, ::2]
    target = torch.full((3, 8), float("nan"))
    copy_2d_(target[:, 1::2], source)
    np.testing.assert_array_equal(target[:, 1::2].numpy(), source.numpy())
    assert torch.isnan(target[:, ::2]).all()
    zero_(target)
    assert torch.count_nonzero(target) == 0


def test_native_empty_buffers_are_noops():
    empty = torch.empty((3, 0))
    copy_2d_(empty, empty)
    masked_zero_2d_(empty, torch.tensor([True, False, True]))
    zero_(empty)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_native_kernels_match_cpu_reference_on_strided_buffers(dtype):
    import warp as wp

    from instantmimic.envs.controls.fast_muscle.buffer_ops import (
        copy_2d_kernel,
        masked_zero_2d_kernel,
    )

    data = torch.ones((3, 8), dtype=dtype)
    mask = torch.tensor([True, False, True])
    view = data[:, ::2]
    wp.launch(
        masked_zero_2d_kernel,
        dim=tuple(view.shape),
        inputs=[wp.from_torch(view), wp.from_torch(mask)],
        device="cpu",
    )
    expected = torch.ones_like(data)
    expected[[0, 2], ::2] = 0
    np.testing.assert_array_equal(data.numpy(), expected.numpy())
    target = torch.empty_like(data)
    wp.launch(
        copy_2d_kernel,
        dim=tuple(view.shape),
        inputs=[wp.from_torch(view), wp.from_torch(target[:, 1::2])],
        device="cpu",
    )
    np.testing.assert_array_equal(target[:, 1::2].numpy(), view.numpy())


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_cuda_wrappers_capture_strided_cast_mask_and_replay_changed_inputs(dtype):
    import warp as wp

    wp.init()

    source_storage = torch.ones((6, 8), dtype=dtype, device="cuda")
    source = source_storage[::2, ::2]
    target_storage = torch.full((3, 8), -999.0, device="cuda")
    target = target_storage[:, 1::2]
    mask = torch.tensor([True, False, True], device="cuda")
    cleared = torch.ones((3, 5), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def operations():
        copy_2d_(target, source)
        masked_zero_2d_(target, mask)
        zero_(cleared)

    # Use the same stream for Torch writes and Warp launches, as required by
    # callers that share persistent tensor storage across both runtimes.
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        operations()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            operations()
        for values, selected in (
            (torch.ones((3, 4), dtype=dtype), [False, True, False]),
            (torch.zeros((3, 4), dtype=dtype), [True, False, True]),
        ):
            source.copy_(values)
            mask.copy_(torch.tensor(selected, device="cuda"))
            target_storage.fill_(-999)
            cleared.fill_(7)
            graph.replay()
            stream.synchronize()
            expected = torch.full((3, 8), -999.0)
            expected[:, 1::2] = values.float()
            expected[torch.tensor(selected), 1::2] = 0
            torch.testing.assert_close(target_storage.cpu(), expected)
            torch.testing.assert_close(cleared.cpu(), torch.zeros_like(cleared.cpu()))
