from __future__ import annotations

import logging
from contextlib import ContextDecorator
from time import perf_counter


class Timer(ContextDecorator):
    """Simple context-manager/decorator to log elapsed time."""

    _records: dict[str, list[float]] = {}

    def __init__(
        self,
        message: str = "",
        metric: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.message = message
        self.metric = metric
        self.logger = logger or logging.getLogger(__name__)
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> Timer:
        self._start = perf_counter()
        if self.message:
            self.logger.info(self.message)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        self.elapsed = perf_counter() - self._start
        if self.metric:
            self._records.setdefault(self.metric, []).append(self.elapsed)

        if self.message:
            self.logger.info(f"{self.message} - {self.elapsed:.3f}s")
        else:
            self.logger.info(f"Timer finished in {self.elapsed:.3f}s")

        # Do not suppress exceptions
        return False

    @classmethod
    def get_metric(cls, name: str) -> list[float]:
        """Return recorded durations for a metric."""
        return cls._records.get(name, []).copy()

    @classmethod
    def summary(cls) -> dict[str, list[float]]:
        """Return a shallow copy of all recorded metrics."""
        return {key: value.copy() for key, value in cls._records.items()}

    @classmethod
    def reset(cls, name: str | None = None) -> None:
        """Reset recorded metrics."""
        if name is None:
            cls._records.clear()
        else:
            cls._records.pop(name, None)
