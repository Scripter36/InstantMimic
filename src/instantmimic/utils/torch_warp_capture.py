"""Torch-owned capture storage with Warp graph execution."""

import ctypes
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import warp as wp


class _TorchOwnedWarpGraph(wp.Graph):
    def __init__(self, captured: wp.Graph, owner: torch.cuda.CUDAGraph):
        from cuda.bindings import driver

        super().__init__(captured.device, captured.capture_id)
        self.module_execs = captured.module_execs
        self._deterministic_buffer_refs = captured._deterministic_buffer_refs
        self.graph = ctypes.c_void_p(owner.raw_cuda_graph())
        self._torch_owner = owner
        self._driver = driver

    def __del__(self):
        # Torch owns the source graph and its allocation pool. Warp owns only
        # the executable it instantiates for replay; release that first.
        self.graph = None
        graph_exec = getattr(self, "graph_exec", None)
        self.graph_exec = None
        try:
            if graph_exec is not None:
                with self.device.context_guard:
                    (result,) = self._driver.cuGraphExecDestroy(
                        self._driver.CUgraphExec(graph_exec.value)
                    )
                    if result != self._driver.CUresult.CUDA_SUCCESS:
                        warnings.warn(
                            f"Could not destroy mixed CUDA graph executable: {result.name}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
        except (AttributeError, TypeError):
            # CUDA/module handles may already be gone during interpreter exit.
            return
        super().__del__()


def _initialize_blas_for_capture(torch_stream: torch.cuda.Stream) -> None:
    """Create this thread's cuBLAS/cuBLASLt handles and the stream's workspace.

    Torch creates them on the first matmul, and handle creation is not
    capturable: a captured region that holds a thread's first matmul fails
    with CUBLAS_STATUS_NOT_INITIALIZED. Whether an earlier matmul ran depends
    on what preceded the capture (for example a torch.compile autotune that a
    warm cache skips), so warm them up explicitly on the capture stream.
    """
    with torch.cuda.stream(torch_stream):
        probe = torch.zeros((2, 2), device=torch_stream.device, dtype=torch.float32)
        torch.nn.functional.linear(probe, probe, probe[0])
        torch.mm(probe, probe)


@contextmanager
def capture_warp_graph(*, stream: wp.Stream) -> Iterator[wp.ScopedCapture]:
    """Keep Torch temporaries alive while Warp replays the mixed graph."""
    owner = torch.cuda.CUDAGraph(keep_graph=True)
    torch_stream = wp.stream_to_torch(stream)
    _initialize_blas_for_capture(torch_stream)
    with torch.cuda.graph(
        owner, stream=torch_stream, capture_error_mode="thread_local"
    ):
        with wp.ScopedCapture(stream=stream, external=True) as capture:
            yield capture
    capture.graph = _TorchOwnedWarpGraph(capture.graph, owner)
