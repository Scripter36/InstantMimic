"""Selection for the singular training benchmark protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from instantmimic.train.clip_failure_benchmark import (
    ClipFailureBenchmarkConfig,
    resolve_semantic_clip_failure_benchmark_config,
)
from instantmimic.train.eval import TrainingEvalPlan


@dataclass(frozen=True)
class TrainingBenchmarkPlan:
    rollout: TrainingEvalPlan | None
    clip_failure: ClipFailureBenchmarkConfig | None

    @property
    def rollout_or_disabled(self) -> TrainingEvalPlan:
        return self.rollout or TrainingEvalPlan.disabled()

    @property
    def clip_failure_or_disabled(self) -> ClipFailureBenchmarkConfig:
        return self.clip_failure or ClipFailureBenchmarkConfig.disabled()

    @classmethod
    def from_semantic_trainer(
        cls,
        trainer: Mapping[str, Any],
        *,
        env_use_cuda_graph: bool = False,
        compile: bool = False,
    ) -> TrainingBenchmarkPlan:
        """Resolve the active evaluation protocol for one run.

        A clip-failure benchmark inherits the run's execution settings rather
        than authoring its own: its step graph follows ``env.use_cuda_graph``
        and its policy/analysis compile follows ``runtime.compile``.
        """

        benchmark = trainer.get("benchmark")
        clip_failure_benchmark = trainer.get("clip_failure_benchmark")
        if benchmark is not None and clip_failure_benchmark is not None:
            raise ValueError(
                "Configure only one of trainer.benchmark and "
                "trainer.clip_failure_benchmark."
            )
        if benchmark is not None and not isinstance(benchmark, Mapping):
            raise TypeError("trainer.benchmark must be a mapping")
        if clip_failure_benchmark is not None and not isinstance(
            clip_failure_benchmark, Mapping
        ):
            raise TypeError("trainer.clip_failure_benchmark must be a mapping")
        return cls(
            rollout=(
                TrainingEvalPlan.from_semantic_benchmark(benchmark)
                if benchmark is not None
                else None
            ),
            clip_failure=(
                resolve_semantic_clip_failure_benchmark_config(
                    clip_failure_benchmark,
                    use_cuda_graph=bool(env_use_cuda_graph),
                    compile=bool(compile),
                )
                if clip_failure_benchmark is not None
                else None
            ),
        )
