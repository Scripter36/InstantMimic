"""Lightweight playback result wire format for runtime and analysis callers."""

from __future__ import annotations

import json
import re

_RESULT_RE = re.compile(r"PLAYBACK_RESULT(_JSON)?[ \t]+(.*)")


def parse_playback_result_payload(text: str) -> dict[str, str] | None:
    """Read the latest JSON result or a legacy key=value result line."""
    for line in reversed(text.splitlines()):
        match = _RESULT_RE.search(line)
        if match is None:
            continue
        if match.group(1):
            payload = json.loads(match.group(2))
            if not isinstance(payload, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in payload.items()
            ):
                raise TypeError("playback JSON result must map strings to strings")
            return payload
        payload = {}
        for token in match.group(2).split():
            if "=" in token:
                key, value = token.split("=", 1)
                payload[key] = value
        return payload
    return None
