"""One policy for degrading a failed graph or compile capture to eager.

Four sites used to swallow any capture failure, log it, and continue eagerly:
playback's tracking-metrics graph, its video device-step graph, its compiled
deterministic policy path, and PPO's best-checkpoint snapshot graph. A run whose
graph never captured then looked like a run that never asked for one, except in
the log -- which `AGENTS.md` rules out for the playback and benchmark paths,
since "Do not disable CUDA graph or compile paths to make a change or benchmark
pass" is unenforceable when the code disables them silently.

It lives here rather than under `utils/`, whose top-level modules are an
explicit shrinking allowlist that `tests/test_utils_boundaries.py` enforces.
`rl/` is the home for algorithm-independent helpers, and playback already
imports from it.

The policy separates the two audiences. Playback and benchmarking raise by
default and degrade only when the run asks for it with
`play.allow_eager_fallback: true`, recording what degraded so the result payload
can carry it. Training keeps degrading -- losing a checkpoint snapshot graph
should not end a run that is otherwise fine -- but records it the same way, so
the flag reaches the metrics rather than only the log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class EagerFallbackError(RuntimeError):
    """A capture failed and this run does not permit falling back to eager."""


@dataclass
class EagerFallbackPolicy:
    """Decide, and record, whether one capture failure may degrade to eager.

    `allowed=False` re-raises as `EagerFallbackError`, chaining the original.
    `allowed=True` logs and records, leaving the caller to install its eager
    path. Either way `recorded` accumulates the stage names, in order, so a
    caller can report them.
    """

    allowed: bool
    recorded: list[str] = field(default_factory=list)

    def degrade(self, stage: str, exc: BaseException) -> None:
        """Record a fallback for `stage`, or raise when it is not permitted."""

        if not self.allowed:
            raise EagerFallbackError(
                f"Failed to capture {stage} and eager fallback is not enabled. "
                "Set play.allow_eager_fallback=true to run without it, or fix "
                "the capture."
            ) from exc
        self.recorded.append(stage)
        logger.warning(
            "Failed to capture %s; falling back to eager. This run permits it "
            "via play.allow_eager_fallback.",
            stage,
            exc_info=exc,
        )

    def as_payload_value(self) -> str:
        """The stages that degraded, for a result payload. Empty when none did."""

        return ",".join(self.recorded)


def training_fallback_policy() -> EagerFallbackPolicy:
    """A policy for training paths: always degrades, always records.

    Training does not raise here. A best-checkpoint snapshot graph that will not
    capture costs some copy throughput and nothing else, and ending a long run
    over it would be worse than continuing. The record is what makes it visible.
    """

    return EagerFallbackPolicy(allowed=True)
