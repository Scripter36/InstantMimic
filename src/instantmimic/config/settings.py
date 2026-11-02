"""Common normalization for authored immutable configuration blocks."""

from collections.abc import Mapping

from pydantic import ConfigDict, model_validator

from instantmimic.config.typed_mapping import TypedMapping


class Settings(TypedMapping):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        # Configuration sequences are tuples in every declared block.
        if isinstance(value, Mapping):
            return {
                key: tuple(item) if isinstance(item, list) else item
                for key, item in value.items()
            }
        return value


def field_names(model: type[TypedMapping]) -> frozenset[str]:
    """Serialized names, including reserved Python identifiers such as class."""
    return frozenset(field.alias or name for name, field in model.model_fields.items())


def required_field_names(model: type[TypedMapping]) -> frozenset[str]:
    return frozenset(
        field.alias or name
        for name, field in model.model_fields.items()
        if field.is_required()
    )
