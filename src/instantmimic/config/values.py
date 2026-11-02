"""Deeply immutable extension values and detached plain-data conversion."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType

from instantmimic.config.typed_mapping import TypedMapping

type ScalarValue = str | int | float | bool | None
type FrozenValue = ScalarValue | FrozenMapping | TypedMapping | tuple[FrozenValue, ...]
type ThawedValue = ScalarValue | dict[str, ThawedValue] | list[ThawedValue]


class FrozenMapping(Mapping[str, FrozenValue]):
    """An attribute-readable mapping whose complete value tree is immutable."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, FrozenValue]) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(values)))

    def __getitem__(self, key: str) -> FrozenValue:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, name: str) -> FrozenValue:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("frozen configuration mappings do not support assignment")

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._data)!r})"


def freeze_value(value: ThawedValue | FrozenValue) -> FrozenValue:
    if isinstance(value, Mapping):
        return FrozenMapping({key: freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    return value


def thaw_value(value: FrozenValue) -> ThawedValue:
    if isinstance(value, TypedMapping):
        return value.to_dict()
    if isinstance(value, FrozenMapping):
        return {key: thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    return value
