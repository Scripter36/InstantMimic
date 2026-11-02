from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
)

from instantmimic.config.agent_config import AgentConfig, parse_agent
from instantmimic.config.experiment_config import ExperimentConfig
from instantmimic.config.model_config import ModelsConfig
from instantmimic.config.play_config import PlayConfig
from instantmimic.config.runtime_config import RuntimeConfig
from instantmimic.config.trainer_config import TrainerConfig
from instantmimic.config.values import (
    FrozenMapping as FrozenMapping,
)
from instantmimic.config.values import (
    FrozenValue as FrozenValue,
)
from instantmimic.config.values import (
    ScalarValue as ScalarValue,
)
from instantmimic.config.values import (
    ThawedValue as ThawedValue,
)
from instantmimic.config.values import (
    freeze_value as freeze_value,
)
from instantmimic.config.values import (
    thaw_value as thaw_value,
)


@dataclass(frozen=True, slots=True)
class SequenceField:
    """Declarative schema for a homogeneous environment-extension sequence."""

    item_schema: object
    min_length: int = 0
    max_length: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("min_length", self.min_length),
            ("max_length", self.max_length),
        ):
            if isinstance(value, bool):
                raise TypeError(f"sequence {name} constraint cannot be a boolean")
            if value is not None and not isinstance(value, int):
                raise TypeError(f"sequence {name} constraint must be an integer")
        if self.min_length < 0:
            raise TypeError("sequence min_length constraint must be nonnegative")
        if self.max_length is not None and self.max_length < self.min_length:
            raise TypeError(
                "sequence max_length constraint must not be below min_length"
            )


@dataclass(frozen=True, slots=True)
class IntegerField:
    """Declarative schema for a range-constrained integer."""

    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if isinstance(value, bool):
                raise TypeError(f"integer {name} constraint cannot be a boolean")
            if value is not None and not isinstance(value, int):
                raise TypeError(f"integer {name} constraint must be an integer")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise TypeError("integer minimum constraint must not exceed maximum")


@dataclass(frozen=True, slots=True)
class NumberField:
    """Declarative schema for a finite range-constrained number."""

    minimum: int | float | None = None
    maximum: int | float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False

    def __post_init__(self) -> None:
        constraints: dict[str, float] = {}
        for name, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if isinstance(value, bool):
                raise TypeError(f"number {name} constraint cannot be a boolean")
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise TypeError(f"number {name} constraint must be numeric")
            try:
                numeric = float(value)
            except OverflowError as exc:
                raise TypeError(f"number {name} constraint must be finite") from exc
            if not isfinite(numeric):
                raise TypeError(f"number {name} constraint must be finite")
            constraints[name] = numeric
        for name, value in (
            ("exclusive_minimum", self.exclusive_minimum),
            ("exclusive_maximum", self.exclusive_maximum),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        if (
            "minimum" in constraints
            and "maximum" in constraints
            and constraints["minimum"] > constraints["maximum"]
        ):
            raise TypeError("number minimum constraint must not exceed maximum")
        if (
            constraints.get("minimum") == constraints.get("maximum")
            and "minimum" in constraints
            and (self.exclusive_minimum or self.exclusive_maximum)
        ):
            raise TypeError("number constraints define an empty domain")


@dataclass(frozen=True, slots=True)
class OptionalField:
    """Mark one environment-extension field as optional."""

    schema: object


@dataclass(frozen=True, slots=True)
class NullableField:
    """Allow an explicitly authored null in addition to one typed value."""

    schema: object


@dataclass(frozen=True, slots=True)
class StringMapField:
    """Declarative schema for a mapping with string keys and typed values."""

    value_schema: object


@dataclass(frozen=True, slots=True)
class EnumField:
    """Declarative schema for a closed scalar value set."""

    values: tuple[ScalarValue, ...]

    def __init__(self, *values: ScalarValue) -> None:
        unique: list[ScalarValue] = []
        for value in values:
            if value is not None and type(value) not in (str, int, float, bool):
                raise TypeError("enum values must be scalar values")
            if type(value) is float and not isfinite(value):
                raise TypeError("numeric enum values must be finite")
            if not any(
                type(value) is type(candidate) and value == candidate
                for candidate in unique
            ):
                unique.append(value)
        object.__setattr__(self, "values", tuple(unique))


@dataclass(frozen=True, slots=True)
class BoundsField:
    """Declarative schema for a finite ordered two-number sequence."""


@dataclass(frozen=True, slots=True)
class SymmetricBoundsField:
    """Declarative schema for an ordered numeric pair shaped exactly as [-v, v]."""


@dataclass(frozen=True, slots=True)
class TaggedUnionField:
    """Declarative closed tagged union for environment extensions."""

    discriminator: str
    variants: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        if not isinstance(self.discriminator, str) or not self.discriminator:
            raise TypeError("tagged-union discriminator must be a non-empty string")
        for tag, fields in self.variants.items():
            if not isinstance(tag, str) or not tag:
                raise TypeError("tagged-union tags must be non-empty strings")
            if self.discriminator in fields:
                raise TypeError(
                    "tagged-union discriminator "
                    f"{self.discriminator!r} collision in variant {tag!r}"
                )
        variants = {
            tag: MappingProxyType(dict(fields)) for tag, fields in self.variants.items()
        }
        object.__setattr__(self, "variants", MappingProxyType(variants))


@dataclass(frozen=True, slots=True, init=False)
class UntaggedUnionField:
    """Declarative structural union requiring exactly one matching variant."""

    variants: tuple[Mapping[str, object], ...]

    def __init__(self, *variants: Mapping[str, object]) -> None:
        if len(variants) < 2:
            raise TypeError("untagged union requires at least two variants")
        frozen = tuple(MappingProxyType(dict(variant)) for variant in variants)
        object.__setattr__(self, "variants", frozen)


class SemanticConfig(BaseModel):
    """Resolved, validated, deeply immutable schema-version-1 configuration."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    schema_version: Literal[1]
    seed: int
    task: str
    env: FrozenMapping
    models: ModelsConfig
    agent: AgentConfig
    trainer: TrainerConfig
    experiment: ExperimentConfig
    runtime: RuntimeConfig
    play: PlayConfig

    @field_validator(
        "env",
        mode="before",
    )
    @classmethod
    def _freeze_mapping(cls, value: object) -> FrozenMapping:
        if not isinstance(value, Mapping):
            raise TypeError("configuration root must be a mapping")
        frozen = freeze_value(dict(value))  # type: ignore[arg-type]
        if not isinstance(frozen, FrozenMapping):
            raise TypeError("configuration root must be a mapping")
        return frozen

    @field_serializer(
        "env",
    )
    def _serialize_mapping(self, value: FrozenMapping) -> ThawedValue:
        return thaw_value(value)

    @field_validator("agent", mode="before")
    @classmethod
    def _parse_agent(cls, value: object) -> AgentConfig:
        if not isinstance(value, Mapping):
            raise TypeError("agent must be a mapping")
        return parse_agent(value)

    @field_serializer("models", "agent", "runtime", "trainer", "experiment", "play")
    def _serialize_typed(
        self,
        value: ModelsConfig
        | AgentConfig
        | RuntimeConfig
        | TrainerConfig
        | ExperimentConfig
        | PlayConfig,
    ) -> dict[str, object]:
        return value.to_dict()

    @classmethod
    def from_mapping(cls, values: Mapping[str, ThawedValue]) -> SemanticConfig:
        return cls.model_validate(dict(values))

    def to_dict(self) -> dict[str, ThawedValue]:
        return thaw_semantic_config(self)


def thaw_semantic_config(config: SemanticConfig) -> dict[str, ThawedValue]:
    """Return a detached plain-Python serialization of a semantic config."""

    return config.model_dump(mode="python")  # type: ignore[return-value]
