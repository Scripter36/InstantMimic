"""Placeholder for MUSUBI, which this release does not include."""

from collections.abc import Mapping
from typing import NoReturn

UNAVAILABLE_MESSAGE = (
    "MUSUBI is not included in this release because its patent application is "
    "still in progress. We apologize for the inconvenience and plan to release "
    "it once the process allows. Thank you for your understanding."
)


class MusubiUnavailableError(NotImplementedError):
    """Raised when a configuration selects MUSUBI."""


def raise_unavailable() -> NoReturn:
    raise MusubiUnavailableError(UNAVAILABLE_MESSAGE)


def reject_unavailable_selectors(config: object) -> None:
    """Reject unavailable requests before configuration can allocate a device."""
    for path in (("env", "action", "type"), ("models", "actor", "architecture")):
        value = config
        for field in path:
            value = (
                value.get(field)
                if isinstance(value, Mapping)
                else getattr(value, field, None)
            )
        if isinstance(value, str) and value.strip().lower() == "musubi":
            raise_unavailable()


class MusubiActorCritic:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise_unavailable()


def reject_unavailable_network(network_cls: type | None) -> None:
    if isinstance(network_cls, type) and issubclass(network_cls, MusubiActorCritic):
        raise_unavailable()
