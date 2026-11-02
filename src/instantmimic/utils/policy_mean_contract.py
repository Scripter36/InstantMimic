"""Native callable identities eligible for a benchmark policy snapshot.

Owners register dependencies while their defining module is still importing.
Unknown replacements, including metadata-copying wrappers, remain uncertified.
"""

from collections.abc import Callable, Sequence
from inspect import getattr_static
from types import MappingProxyType
from typing import Any

_CONTRACTS: dict[str, tuple[type, MappingProxyType]] = {}


def register_policy_mean_methods(owner: type, role: str, names: Sequence[str]) -> None:
    methods = {
        name: (
            getattr(owner, name),
            isinstance(getattr_static(owner, name), staticmethod),
        )
        for name in names
    }
    if role in _CONTRACTS:
        previous_owner, previous = _CONTRACTS[role]
        if previous_owner is not owner or dict(previous) != methods:
            raise RuntimeError(f"Conflicting native policy contract: {role}")
        return
    _CONTRACTS[role] = owner, MappingProxyType(methods)


def matches_policy_mean_methods(instance: Any, role: str) -> bool:
    record = _CONTRACTS.get(role)
    if record is None or not isinstance(instance, record[0]):
        return False
    for name, (expected, is_static) in record[1].items():
        candidate = getattr(instance, name, None)
        if is_static:
            if candidate is not expected:
                return False
        elif (
            getattr(candidate, "__func__", None) is not expected
            or getattr(candidate, "__self__", None) is not instance
        ):
            return False
    return True


def matches_policy_mean_bound_callable(
    instance: Any, role: str, name: str, candidate: Callable | None
) -> bool:
    record = _CONTRACTS.get(role)
    return bool(
        record is not None
        and isinstance(instance, record[0])
        and getattr(candidate, "__self__", None) is instance
        and name in record[1]
        and not record[1][name][1]
        and getattr(candidate, "__func__", None) is record[1][name][0]
    )
