from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_MISSING = object()
_NO_DEFAULT = object()


def get_value(cfg: Any, key: str, default: Any = _NO_DEFAULT) -> Any:
    getter = getattr(cfg, "get", None)
    if callable(getter):
        if default is _NO_DEFAULT:
            value = getter(key, _MISSING)
        else:
            value = getter(key, default)
    elif isinstance(cfg, Mapping):
        if default is _NO_DEFAULT:
            value = cfg.get(key, _MISSING)
        else:
            value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, _MISSING if default is _NO_DEFAULT else default)
    if value is _MISSING:
        raise ValueError(f"missing required environment config key: {key}")
    return value


def get_block(cfg: Any, key: str, *, required: bool = True) -> Any | None:
    value = get_value(cfg, key, None)
    if value is None:
        if required:
            raise ValueError(f"missing required environment config block: {key}")
        return None
    if not isinstance(value, Mapping) and not callable(getattr(value, "get", None)):
        raise TypeError(f"environment config block {key!r} must be a mapping")
    return value


__all__ = ("get_block", "get_value")
