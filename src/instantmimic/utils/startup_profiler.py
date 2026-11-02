"""Optional startup timing and NVTX helpers with no simulator dependencies."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from typing import ContextManager, TypeVar

_F = TypeVar("_F", bound=Callable[..., object])


def startup_nvtx_range(name: str) -> ContextManager[object]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.nvtx.range(f"startup:{name}")
    except Exception as exc:  # noqa: BLE001 - optional CUDA/NVTX probe must not affect startup.
        logging.getLogger(__name__).debug(
            "Startup NVTX range unavailable; continuing without NVTX: %s",
            exc,
            exc_info=True,
        )
    return nullcontext()


def startup_nvtx_wrapped(name: str) -> Callable[[_F], _F]:
    def _decorator(func: _F) -> _F:
        @wraps(func)
        def _wrapped(*args, **kwargs):
            with startup_nvtx_range(name):
                return func(*args, **kwargs)

        return _wrapped  # type: ignore[return-value]

    return _decorator


@dataclass(frozen=True)
class StartupProfileRecord:
    name: str
    seconds: float


class StartupProfiler:
    """Coarse wall-time profiler for process startup and training prewarm phases."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = perf_counter,
        range_factory: Callable[[str], ContextManager[object]] | None = None,
    ) -> None:
        self._clock = clock
        self._range_factory = range_factory
        self._records: list[StartupProfileRecord] = []

    @property
    def records(self) -> tuple[StartupProfileRecord, ...]:
        return tuple(self._records)

    def phase(self, name: str) -> ContextManager[None]:
        profiler = self

        class _Phase:
            def __enter__(self) -> None:
                self._range = (
                    profiler._range_factory(name)
                    if profiler._range_factory is not None
                    else nullcontext()
                )
                self._range.__enter__()
                self._start = profiler._clock()
                return None

            def __exit__(self, exc_type, exc_value, traceback) -> bool:
                elapsed = profiler._clock() - self._start
                profiler._records.append(StartupProfileRecord(name, elapsed))
                return bool(self._range.__exit__(exc_type, exc_value, traceback))

        return _Phase()

    def total_seconds(self) -> float:
        return sum(record.seconds for record in self._records)

    def log_summary(
        self,
        *,
        logger: logging.Logger | None = None,
        prefix: str = "TRAIN_STARTUP_PROFILE",
    ) -> None:
        log = logger or logging.getLogger(__name__)
        for record in self._records:
            log.info("%s phase=%s seconds=%.6f", prefix, record.name, record.seconds)
        log.info(
            "%s total_seconds=%.6f phases=%d",
            prefix,
            self.total_seconds(),
            len(self._records),
        )
