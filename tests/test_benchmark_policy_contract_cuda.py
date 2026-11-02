"""Complete policy temporaries remain owned inside a conditional child graph."""

import gc

import pytest
import torch
import warp as wp

from instantmimic.train.benchmark_runner import _capture_eval_conditional_horizon
from instantmimic.utils.torch_warp_capture import capture_warp_graph


@wp.kernel(enable_backward=False)
def _refresh_condition(
    remaining: wp.array(dtype=wp.int32), condition: wp.array(dtype=wp.int32)
):
    condition[0] = wp.int32(remaining[0] > 0)


@wp.kernel(enable_backward=False)
def _finish_step(
    remaining: wp.array(dtype=wp.int32), condition: wp.array(dtype=wp.int32)
):
    remaining[0] = remaining[0] - 1
    condition[0] = wp.int32(remaining[0] > 0)


@pytest.mark.cuda
def test_complete_policy_child_keeps_torch_temporaries_and_reads_current_inputs():
    wp.init()
    stream = torch.cuda.Stream()
    warp_stream = wp.stream_from_torch(stream)
    observation = torch.randn(4, 7, device="cuda")
    parameter = torch.randn(7, device="cuda")
    total = torch.zeros((), device="cuda")
    horizon = torch.tensor([3], device="cuda", dtype=torch.int32)
    remaining = torch.empty_like(horizon)
    condition = torch.empty_like(horizon)
    rw, cw = wp.from_torch(remaining), wp.from_torch(condition)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), wp.ScopedStream(warp_stream):
        # Warm every operation before capture; no graph/compile fallback.
        total.add_(torch.sin(observation + parameter).sum())
        wp.launch(_refresh_condition, dim=1, inputs=[rw, cw])
        wp.launch(_finish_step, dim=1, inputs=[rw, cw])
        stream.synchronize()
        with capture_warp_graph(stream=warp_stream) as child:
            total.add_(torch.sin(observation + parameter).sum())
            wp.launch(_finish_step, dim=1, inputs=[rw, cw])
        with capture_warp_graph(stream=warp_stream) as parent:
            remaining.copy_(horizon)
            total.zero_()
            wp.launch(_refresh_condition, dim=1, inputs=[rw, cw])
            _capture_eval_conditional_horizon(cw, child.graph)
        gc.collect()
        for steps in (3, 1, 0, 5):
            horizon.fill_(steps)
            observation.add_(0.1)
            parameter.mul_(0.9)
            wp.capture_launch(parent.graph, stream=warp_stream)
            stream.synchronize()
            expected = steps * torch.sin(observation + parameter).sum()
            torch.testing.assert_close(total, expected)
            assert int(remaining) == 0
        allocated = torch.cuda.memory_allocated()
        for _ in range(8):
            wp.capture_launch(parent.graph, stream=warp_stream)
        stream.synchronize()
        assert torch.cuda.memory_allocated() == allocated
    torch.cuda.current_stream().wait_stream(stream)
