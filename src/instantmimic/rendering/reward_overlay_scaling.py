from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_ESTIMATED_TEXT_CHAR_WIDTH = 6
_LEGEND_TEXT_PADDING = 26


def _finite_series_values(series_map: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for raw_values in series_map.values():
        for raw_value in raw_values:
            value = float(raw_value)
            if math.isfinite(value):
                values.append(value)
    return values


def resolve_overlay_y_range(series_map: Mapping[str, Any]) -> tuple[float, float]:
    values = _finite_series_values(series_map)
    if not values:
        return 0.0, 1.0

    min_value = min(values)
    max_value = max(values)
    if min_value >= 0.0 and max_value <= 1.0:
        return 0.0, 1.0

    y_min = min(0.0, min_value)
    y_max = max(0.0, max_value)
    span = y_max - y_min
    if span <= 0.0:
        magnitude = max(1.0, abs(y_min), abs(y_max))
        y_min -= 0.5 * magnitude
        y_max += 0.5 * magnitude
        span = y_max - y_min

    padding = 0.05 * span
    if y_min < 0.0:
        y_min -= padding
    if y_max > 0.0:
        y_max += padding
    if y_max <= y_min:
        return 0.0, 1.0
    return float(y_min), float(y_max)


def resolve_overlay_legend_width(
    *,
    left: int,
    right: int,
    series_map: Mapping[str, Any],
    min_width: int = 142,
    max_width: int = 300,
) -> int:
    chart_width = max(1, int(right) - int(left))
    max_allowed = max(
        96, min(int(max_width), int(chart_width * 0.52), chart_width - 40)
    )
    desired = int(min_width)
    for name, raw_values in series_map.items():
        values = list(raw_values)
        if not values:
            continue
        label = f"{name}: {float(values[-1]):.3f}"
        desired = max(
            desired,
            len(label) * _ESTIMATED_TEXT_CHAR_WIDTH + _LEGEND_TEXT_PADDING,
        )
    return max(96, min(max_allowed, desired))


def format_overlay_legend_label(
    name: str,
    value: float,
    *,
    legend_width: int,
) -> str:
    value_text = f"{float(value):.3f}"
    suffix = f": {value_text}"
    max_chars = max(
        len(suffix) + 1,
        (max(1, int(legend_width)) - _LEGEND_TEXT_PADDING)
        // _ESTIMATED_TEXT_CHAR_WIDTH,
    )
    label = f"{name}{suffix}"
    if len(label) <= max_chars:
        return label

    name_budget = max(1, max_chars - len(suffix))
    if name_budget <= 3:
        short_name = str(name)[:name_budget]
    else:
        short_name = f"{str(name)[: name_budget - 3]}..."
    return f"{short_name}{suffix}"
