"""Type checks shared by the config surfaces that read untrusted mappings.

`config/validation.py`, `config/loader.py`, and `rl/build_specs.py` each grew
their own `_bool`, `_int`, `_number`, `_string`, `_mapping`, and `_sequence`.
The checks were the same; what differed was the exception each module raises
and the wording it uses. Multiple copies of one
`isinstance` chain is how they drifted -- `rl/build_specs.py` accepted `inf` and
`nan` where the other two rejected them.

So the check lives here once and the caller keeps its own error. A caller binds
`Coercions` to a reporter that raises its exception type, and the reporter
receives the canonical phrase for the failure; a module whose wording differs
maps the phrase before raising. Nothing here raises on its own.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import NoReturn

BOOLEAN = "expected a boolean"
INTEGER = "expected an integer"
NUMBER = "expected a number"
FINITE = "must be finite"
STRING = "expected a string"
EMPTY = "must not be empty"
MAPPING = "expected a mapping"
MAPPING_KEYS = "mapping keys must be strings"
SEQUENCE = "expected a sequence"

type Reporter = Callable[[str, str, BaseException | None], NoReturn]
"""Raises the caller's error for `(path, phrase)`. Never returns.

The third argument is the failure being translated, when there is one; a
reporter passes it to `raise_translated` so the original survives as
`__cause__`.
"""


def raise_translated(error: BaseException, cause: BaseException | None) -> NoReturn:
    """Raise `error`, chaining `cause` only when there is one.

    `raise error from None` is not the same statement with `cause` unset: it
    suppresses whatever exception the caller happened to be handling. So the
    two cases cannot share one `raise`.
    """

    if cause is None:
        raise error
    raise error from cause


class Coercions:
    """The shared checks bound to one module's `Reporter`."""

    __slots__ = ("_fail",)

    def __init__(self, fail: Reporter) -> None:
        self._fail = fail

    def boolean(self, value: object, path: str) -> bool:
        if isinstance(value, bool):
            return value
        self._fail(path, BOOLEAN, None)

    def integer(self, value: object, path: str) -> int:
        # `bool` is an `int` subclass, and every caller means to reject it.
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        self._fail(path, INTEGER, None)

    def number(self, value: object, path: str) -> int | float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        self._fail(path, NUMBER, None)

    def finite(self, value: object, path: str) -> int | float:
        """`number()` that also rejects `inf` and `nan`.

        An `int` too large to convert to a float raises `OverflowError` here
        rather than comparing unequal, so it fails as non-finite too.
        """

        numeric = self.number(value, path)
        try:
            finite = math.isfinite(numeric)
        except OverflowError as exc:
            self._fail(path, FINITE, exc)
        if not finite:
            self._fail(path, FINITE, None)
        return numeric

    def string(self, value: object, path: str, *, nonempty: bool = False) -> str:
        if not isinstance(value, str):
            self._fail(path, STRING, None)
        if nonempty and not value:
            self._fail(path, EMPTY, None)
        return value

    def mapping(
        self, value: object, path: str, *, string_keys: bool = False
    ) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            self._fail(path, MAPPING, None)
        if string_keys:
            for key in value:
                if not isinstance(key, str):
                    self._fail(path, MAPPING_KEYS, None)
        return value

    def sequence(self, value: object, path: str) -> Sequence[object]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return value
        self._fail(path, SEQUENCE, None)


def closed_keys(
    mapping: Mapping[str, object],
    allowed: frozenset[str] | set[str],
    required: frozenset[str] | set[str] = frozenset(),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return `(unknown, missing)` for a closed key set.

    `unknown` keeps the mapping's own order so a caller can report the first
    offending key at its own path; `missing` is sorted because a required key
    absent from the mapping has no position to report it at.
    """

    unknown = tuple(key for key in mapping if key not in allowed)
    missing = tuple(sorted(key for key in required if key not in mapping))
    return unknown, missing
