"""Typed policy and optimizer blocks; domain validation stays in validation.py."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from instantmimic.config.typed_mapping import TypedMapping

type Number = int | float


class PolicyMapping(TypedMapping):
    @model_validator(mode="before")
    @classmethod
    def reject_authored_nulls(cls, values: object) -> object:
        if isinstance(values, Mapping):
            for key, value in values.items():
                if value is None:
                    raise ValueError(f"{key} must not be null")
        return values

    @field_validator("adamw", "bound_loss", mode="before", check_fields=False)
    @classmethod
    def detach_nested_model(cls, value: object) -> object:
        return value.to_dict() if isinstance(value, TypedMapping) else value

    @field_validator("betas", "log_std_bounds", mode="before", check_fields=False)
    @classmethod
    def freeze_pair(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class AdamWCompanionConfig(PolicyMapping):
    betas: tuple[Number, Number] | None = None
    eps: Number | None = None
    weight_decay: Number | None = None
    amsgrad: bool | None = None


class AdamOptimizerConfig(AdamWCompanionConfig):
    type: Literal["adam"]


class AdamWOptimizerConfig(AdamWCompanionConfig):
    type: Literal["adamw"]


class MuonOptimizerConfig(PolicyMapping):
    type: Literal["muon"]
    momentum: Number | None = None
    nesterov: bool | None = None
    adjust_lr_fn: Literal["match_rms_adamw"] | None = None
    ns_steps: int | None = None
    eps: Number | None = None
    weight_decay: Number | None = None
    adamw: AdamWCompanionConfig | None = None


type OptimizerConfig = Annotated[
    AdamOptimizerConfig | AdamWOptimizerConfig | MuonOptimizerConfig,
    Field(discriminator="type"),
]
OPTIMIZER_MODELS = {
    "adam": AdamOptimizerConfig,
    "adamw": AdamWOptimizerConfig,
    "muon": MuonOptimizerConfig,
}
_OPTIMIZER_ADAPTER = TypeAdapter(OptimizerConfig)


def parse_optimizer(value: Mapping[str, object]) -> OptimizerConfig:
    if isinstance(value, tuple(OPTIMIZER_MODELS.values())):
        return _OPTIMIZER_ADAPTER.validate_python(value.to_dict())
    return _OPTIMIZER_ADAPTER.validate_python(dict(value))


class BoundLossConfig(PolicyMapping):
    weight: Number
    margin: Number


class DistributionConfig(PolicyMapping):
    std: Literal["parameter", "fixed", "network"]
    parameterization: Literal["log", "scalar"] = "log"
    initial_log_std: Number
    log_std_bounds: tuple[Number, Number]
    bound_mode: Literal["clamp", "tanh"]
    action_transform: Literal["none", "tanh"]
    bound_loss: BoundLossConfig | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> DistributionConfig:
        if isinstance(value, cls):
            return cls.model_validate(value.to_dict())
        return cls.model_validate(dict(value))
