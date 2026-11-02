"""Immutable typed blocks exposing only authored fields through Mapping."""

from collections.abc import Iterator, Mapping

from pydantic import BaseModel, ConfigDict, ModelWrapValidatorHandler, model_validator


class ImmutableConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, revalidate_instances="always"
    )

    @property
    def model_fields_set(self) -> set[str]:
        # Pydantic exposes its internal mutable set even on frozen models.
        # Consumers must not be able to change which authored fields we export.
        return set(self.__pydantic_fields_set__)


class TypedMapping(ImmutableConfig, Mapping[str, object]):
    @model_validator(mode="wrap")
    @classmethod
    def validate_authored_values(
        cls, value: object, handler: ModelWrapValidatorHandler
    ):
        # Instance revalidation otherwise supplies defaults as though authored.
        # Detach first, including nested blocks, and validate even unchecked copies.
        if isinstance(value, TypedMapping):
            value = value.to_dict()
        return handler(value)

    def __getitem__(self, key: str) -> object:
        for name, field in type(self).model_fields.items():
            if (field.alias or name) == key and name in self.model_fields_set:
                return getattr(self, name)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        unknown = self.model_fields_set - type(self).model_fields.keys()
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)!r}")
        return (
            field.alias or name
            for name, field in type(self).model_fields.items()
            if name in self.model_fields_set
        )

    def __len__(self) -> int:
        return len(self.model_fields_set)

    def to_dict(self) -> dict[str, object]:
        def thaw(value: object) -> object:
            if isinstance(value, TypedMapping):
                return value.to_dict()
            if isinstance(value, Mapping):
                return {key: thaw(item) for key, item in value.items()}
            if isinstance(value, tuple):
                return [thaw(item) for item in value]
            return value

        return {key: thaw(value) for key, value in self.items()}
