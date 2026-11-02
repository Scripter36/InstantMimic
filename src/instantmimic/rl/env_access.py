from __future__ import annotations

import inspect
from typing import Any

_MISSING = object()


def unwrap_env(env: Any) -> Any:
    """Return an explicitly exposed environment target without probing mocks."""

    if inspect.getattr_static(env, "unwrapped", _MISSING) is _MISSING:
        return env
    return env.unwrapped or env
