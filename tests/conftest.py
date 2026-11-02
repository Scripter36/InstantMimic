"""Bound test-worker memory growth and enforce explicit CUDA markers.

Collection after large allocations prevents cyclic environment state from
accumulating across tests. CPU tests cannot initialize CUDA contexts; tests
that require a device declare the CUDA marker explicitly.
"""

from __future__ import annotations

import contextlib
import gc
import importlib.util
import itertools
import os
import pathlib
import re
import sys

import pytest
from path_helpers import fixtures_config_dir, repo_root


@pytest.fixture(name="repo_root", scope="session")
def _repo_root_fixture() -> pathlib.Path:
    return repo_root()


@pytest.fixture(name="fixtures_config_dir", scope="session")
def _fixtures_config_dir_fixture() -> pathlib.Path:
    return fixtures_config_dir()


# Collect once RSS has grown this far past the last post-collection reading.
# Small enough to catch a single env build, large enough that ordinary tests
# never trigger it.
_GROWTH_LIMIT_BYTES = 1_073_741_824

_STATM = pathlib.Path("/proc/self/statm")
_PAGE_SIZE = 4096


def _resident_bytes() -> int | None:
    """Current RSS, or None where /proc is unavailable."""

    try:
        return int(_STATM.read_text().split()[1]) * _PAGE_SIZE
    except (OSError, IndexError, ValueError):
        return None


class _MemoryFloor:
    def __init__(self) -> None:
        self.baseline = _resident_bytes()

    def collect_if_grown(self) -> None:
        current = _resident_bytes()
        if current is None or self.baseline is None:
            # No RSS reading available: fall back to collecting every time
            # rather than silently letting the peak run away.
            gc.collect()
            self.baseline = None
            return
        if current - self.baseline < _GROWTH_LIMIT_BYTES:
            return
        gc.collect()
        self.baseline = _resident_bytes()


_floor = _MemoryFloor()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    del item, nextitem
    _floor.collect_if_grown()


# Peak RSS a worker needs while building one large musculoskeletal env,
# measured per file with everything else quiet: 6.07GB for
# larger environment modules down to ~2GB for the lightest module, and
# ~3.5GB for a single ostrich test on its own.
#
# This floor is native, not Python: `tracemalloc` attributes nothing above 20MB
# to Python objects in such a test, because the memory is the CUDA context, the
# Warp JIT compiling kernels in-process, and MuJoCo's C model. No test-side
# change removes it, so the largest safe worker count is a property of the
# machine -- which is why this is checked rather than hard-coded.
_WORKER_BUDGET_BYTES = 4 * 1_073_741_824

_MEMINFO = pathlib.Path("/proc/meminfo")


def _available_bytes() -> int | None:
    try:
        for line in _MEMINFO.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def _requested_workers(config: pytest.Config) -> int | None:
    value = getattr(config.option, "numprocesses", None)
    if value in (None, 0):
        return None
    if isinstance(value, int):
        return value
    if value in ("auto", "logical"):
        return os.cpu_count() or 1
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _selection_excludes_cuda(config: pytest.Config) -> bool:
    """Whether the `-m` expression deselects every test with the `cuda` marker.

    A `cuda` test may carry other markers too, so `-m integration` still selects
    one marked `integration` and `cuda`. The expression excludes all `cuda`
    tests only if no assignment of its other marker names makes it true while
    `cuda` is present.
    """

    markexpr = getattr(config.option, "markexpr", "") or ""
    if not markexpr:
        return False
    # pytest has no public API for evaluating a mark expression; this is the
    # evaluator `-m` itself uses.
    from _pytest.mark.expression import Expression

    expression = Expression.compile(markexpr)
    names = sorted(
        set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", markexpr))
        - {"and", "or", "not", "True", "False", "None", "cuda"}
    )
    for present in itertools.product((False, True), repeat=len(names)):
        markers = {
            "cuda",
            *(name for name, on in zip(names, present, strict=True) if on),
        }
        if expression.evaluate(lambda name, markers=markers, **kwargs: name in markers):
            return False
    return True


def pytest_configure(config: pytest.Config) -> None:
    """Hide the GPU from runs without `cuda` tests, then check the worker count.

    Tests without the `cuda` marker must not touch the GPU. When the selection
    excludes every `cuda` test (for example `-m "not cuda"`), the device is
    hidden before torch or Warp initialise CUDA, in the controller and in every
    xdist worker, so an accidental CUDA launch runs on the CPU or fails instead
    of reaching the driver. On a host with HMM enabled, a Warp launch that
    defaults to `cuda:0` while its arrays live in pageable host memory is not
    rejected: the GPU faults the host pages in, and one such launch from this
    suite hung the NVIDIA UVM driver until reboot.

    Over the memory budget, workers are killed mid-test (`worker 'gwN'
    crashed`) rather than failing an assertion, and with `--dist loadgroup` a
    kill can abort the session and lose thousands of results. A clear refusal
    beats discovering that from a truncated summary.
    """

    if _selection_excludes_cuda(config):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Lightweight repository checks do not install Warp. When it is installed,
    # every test still requires launch arrays to live on the launch device;
    # errors from a broken installation must propagate.
    if importlib.util.find_spec("warp") is not None:
        from instantmimic.utils.warp_config import configure_warp_launch_checks

        configure_warp_launch_checks()
    if hasattr(config, "workerinput"):  # running inside a worker
        return
    requested = _requested_workers(config)
    available = _available_bytes()
    if requested is None or available is None or requested <= 1:
        return
    safe = max(1, available // _WORKER_BUDGET_BYTES)
    if requested <= safe:
        return
    raise pytest.UsageError(
        f"-n {requested} needs about {requested * _WORKER_BUDGET_BYTES / 1_073_741_824:.0f}GB "
        f"but only {available / 1_073_741_824:.0f}GB is available. The heaviest "
        f"tests build a musculoskeletal model and peak near "
        f"{_WORKER_BUDGET_BYTES / 1_073_741_824:.0f}GB per worker, and over "
        f"budget the kernel kills workers instead of tests failing. "
        f"Use -n {safe} or fewer here."
    )


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Give the GPU to one worker at a time.

    There is one physical device, and the `cuda` tests do not share it well:
    they build contexts, capture CUDA graphs and allocate device memory, and
    several at once fail in ways none of them fail alone. Running the whole
    suite at `-n 8` produced exactly one such failure
    (`test_real_cuda_playback_replays_owned_inputs_and_preserves_output_slots`),
    while that test passes by itself and its whole file passes at `-n 8`
    three times over.

    Putting every `cuda` test in one xdist group serialises them onto a single
    worker while the CPU tests keep parallelising. This needs
    `--dist loadgroup`; without it the marker is inert and nothing changes.
    """

    del session, config
    for item in items:
        if item.get_closest_marker("cuda") is not None:
            item.add_marker(pytest.mark.xdist_group("cuda_device"))


def _refuse_cuda_initialization() -> None:
    raise RuntimeError("CUDA is unavailable to tests without the `cuda` marker")


def _retained_warp_cuda_contexts(warp) -> set[str]:
    return {
        device.alias
        for device in warp.get_cuda_devices()
        if getattr(device, "_context", None) is not None
    }


def _forget_torch_device_probes(torch) -> None:
    """Drop torch's memoised device probes taken while CUDA was hidden.

    `torch.utils._triton.has_triton()` and its neighbours cache their first
    answer for the process. Computed during an unmarked test, that answer is
    "no CUDA", and a later `cuda` test in the same worker then fails to
    compile with `TritonMissing`.
    """

    triton_probes = sys.modules.get("torch.utils._triton")
    if triton_probes is None:
        return
    for probe in vars(triton_probes).values():
        cache_clear = getattr(probe, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


@contextlib.contextmanager
def _torch_without_cuda(torch):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        if not torch.cuda.is_initialized():
            # torch.compile probes the CUDA device for its cache key even for
            # CPU graphs and treats this RuntimeError as "no CUDA".
            patch.setattr(torch.cuda, "_lazy_init", _refuse_cuda_initialization)
        try:
            yield
        finally:
            patch.undo()
            _forget_torch_device_probes(torch)


@contextlib.contextmanager
def _warp_without_cuda(warp, nodeid: str):
    # Scoping the device initialises Warp, so apply the repository's cache
    # settings first; a later initialize_warp_runtime() call is a no-op.
    # Initialisation queries the driver but creates no CUDA context.
    from instantmimic.utils.warp_config import initialize_warp_runtime

    initialize_warp_runtime()
    contexts_before = _retained_warp_cuda_contexts(warp)
    with warp.ScopedDevice("cpu"):
        yield
    created = _retained_warp_cuda_contexts(warp) - contexts_before
    if created:
        pytest.fail(
            f"{nodeid} created Warp CUDA contexts {sorted(created)} "
            "without the `cuda` marker",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _keep_unmarked_tests_off_the_gpu(request: pytest.FixtureRequest):
    """Run tests without the `cuda` marker as if no GPU existed.

    This covers runs that also select `cuda` tests, where the device stays
    visible to the process. `torch.cuda.is_available()` reports False, so tests
    that pick `cuda` when it is available take the CPU path, and initialising
    torch CUDA raises. Warp launches default to the CPU, where the session's
    strict launch-array checks reject GPU-resident arguments, and a test that
    still creates a Warp CUDA context fails. Either failure means the test
    needs the `cuda` marker. A process whose CUDA
    state a `cuda` test already initialised cannot be guarded this way; the
    hidden device of a `-m "not cuda"` run covers that case.
    """

    if request.node.get_closest_marker("cuda") is not None:
        yield
        return
    with contextlib.ExitStack() as stack:
        torch = sys.modules.get("torch")
        if torch is not None:
            stack.enter_context(_torch_without_cuda(torch))
        warp = sys.modules.get("warp")
        if warp is not None:
            stack.enter_context(_warp_without_cuda(warp, request.node.nodeid))
        yield
