"""Shared runtime configuration, independent of Hydra and RL implementations."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

if TYPE_CHECKING:
    from instantmimic.config.schema import SemanticConfig

from instantmimic.config.typed_mapping import TypedMapping

CompileMode = Literal[
    "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
]
NvtxMode = Literal["launch", "detailed"]
DEFAULT_COMPILE_MODE: CompileMode = "max-autotune-no-cudagraphs"


class RuntimeConfig(TypedMapping):
    """Effective typed defaults with a mapping view of authored fields only.

    Mapping access and serialization preserve omissions for existing YAML and
    artifact consumers. Attribute access supplies the defaults used by factories.
    Every field is scalar, so freezing the model freezes its complete value tree.
    """

    act_cuda_graph: bool = False
    update_cuda_graph: bool = False
    shuffle_cuda_graph: bool = False
    compile: bool = False
    compile_act_into: bool = False
    compile_update_fullgraph: bool = False
    compile_mode: CompileMode = DEFAULT_COMPILE_MODE
    precision: Literal["float32", "bfloat16", "float16"] | None = None
    loss_scale: float = Field(default=128.0, gt=0, le=2**24, allow_inf_nan=False)
    mixed_precision: bool = False
    policy_mlp_bf16: bool = False
    fail_on_nonfinite_update: bool = False
    nsys_nvtx_mode: NvtxMode = "launch"

    @property
    def ppo_precision(self) -> str | None:
        """An explicit shared policy dtype, or None for the legacy AMP path."""
        if self.precision is not None:
            return self.precision
        return "bfloat16" if self.policy_mlp_bf16 else None

    @model_validator(mode="after")
    def warn_deprecated_mixed_precision(self) -> RuntimeConfig:
        if "mixed_precision" in self.model_fields_set:
            warnings.warn(
                "runtime.mixed_precision is deprecated. Without runtime.precision, "
                "its legacy algorithm-specific behavior is preserved (PPO: "
                "update-only autocast). For PPO, use runtime.precision=float32, "
                "bfloat16, or float16 to share rollout/update precision; an "
                "explicit precision overrides mixed_precision. Other algorithms "
                "currently retain their existing mixed_precision behavior.",
                FutureWarning,
                stacklevel=2,
            )
        return self

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> RuntimeConfig:
        if isinstance(values, cls):
            return cls.model_validate(values)
        return cls.model_validate(dict(values))

    @classmethod
    def from_semantic_config(cls, config: SemanticConfig) -> RuntimeConfig:
        return cls.from_mapping(config.runtime)
