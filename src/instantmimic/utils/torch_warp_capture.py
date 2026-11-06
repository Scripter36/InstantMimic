"""Torch-owned capture storage with Warp graph execution.

Also owns the process-wide Warp stream registry: Warp registers an external
CUDA stream handle when a wp.Stream wraps it and unregisters the handle when
that wrapper is collected. Two wrappers for one handle therefore race, and
dropping either makes the next capture fail with "Warp error: unknown stream".
Every long-lived Warp stream in first-party code comes from
warp_stream_from_torch, so a handle has exactly one wrapper and it stays alive
for the process lifetime.
"""

import ctypes
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import warp as wp

# Wrappers this module owns, keyed by CUDA stream handle. An entry is never
# replaced: replacing it would collect the previous wrapper and unregister the
# handle under every other wrapper sharing it.
_WARP_STREAMS: dict[int, Any] = {}


def stream_handle(stream: Any) -> int:
    """CUDA handle of a Warp or torch stream; the default stream is 0."""
    handle = getattr(stream, "cuda_stream", None)
    return 0 if handle is None else int(handle)


def warp_stream_for_handle(
    handle: int,
    current: Any,
    create: Callable[[int], Any],
) -> Any:
    """The one Warp stream for a CUDA handle.

    current is Warp's stream for the device. A handle already in the registry
    always returns the retained wrapper, even when Warp's current stream is a
    later duplicate for the same handle: handing back the duplicate would leave
    the caller holding a wrapper the registry does not retain, and collecting
    it would unregister the handle under the cached stream.

    Otherwise the first wrapper for the handle wins. When Warp already owns a
    stream for the handle, that wrapper is adopted and retained here rather
    than creating a second registration of the same handle.
    """
    cached = _WARP_STREAMS.get(int(handle))
    if cached is not None:
        return cached
    if current is not None and stream_handle(current) == int(handle):
        _WARP_STREAMS[int(handle)] = current
        return current
    created = create(int(handle))
    _WARP_STREAMS[int(handle)] = created
    return created


def warp_stream_from_torch(torch_stream: Any) -> Any:
    """The persistent wp.Stream for a torch CUDA stream's handle."""
    warp_device = wp.get_device(str(torch_stream.device))
    handle = stream_handle(torch_stream)

    def create(created_handle: int) -> Any:
        stream = wp.Stream(warp_device, cuda_stream=created_handle)
        # Keep the torch stream alive as long as this wrapper exists.
        stream._torch_stream = torch_stream
        return stream

    return warp_stream_for_handle(handle, wp.get_stream(warp_device), create)


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
