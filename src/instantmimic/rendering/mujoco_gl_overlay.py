from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from instantmimic.rendering.reward_overlay_scaling import (
    format_overlay_legend_label,
    resolve_overlay_legend_width,
    resolve_overlay_y_range,
)


@dataclass
class MujocoGLOverlayConfig:
    panel_width: int
    history_seconds: float


_PANEL_BG = np.array([13.0 / 255.0, 17.0 / 255.0, 23.0 / 255.0, 1.0], dtype=np.float32)
_CHART_BG = np.array([0.08, 0.10, 0.13, 1.0], dtype=np.float32)
_LEGEND_BG = np.array([0.02, 0.03, 0.04, 0.75], dtype=np.float32)
_TEXT_RGB = np.array([0.95, 0.96, 0.98], dtype=np.float32)
_GRID_RGB = np.array([0.20, 0.22, 0.25], dtype=np.float32)


def _figure_line_xs(count: int, history_seconds: float) -> np.ndarray:
    if count <= 1:
        return np.array([0.0], dtype=np.float32)
    return np.linspace(
        -float(history_seconds),
        0.0,
        num=int(count),
        dtype=np.float32,
    )


def _build_figure(
    *,
    title: str,
    series_map: dict[str, deque[float]],
    colors_map: dict[str, tuple[int, int, int]] | None,
    history_seconds: float,
) -> mujoco.MjvFigure:
    fig = mujoco.MjvFigure()
    mujoco.mjv_defaultFigure(fig)
    fig.title = str(title)
    y_min, y_max = resolve_overlay_y_range(series_map)
    fig.xlabel = "time (s)"
    fig.range[:] = np.array(
        [[-float(history_seconds), 0.0], [float(y_min), float(y_max)]],
        dtype=np.float32,
    )
    fig.gridsize[:] = np.array([3, 3], dtype=np.int32)
    fig.flg_extend = 0
    fig.flg_barplot = 0
    fig.flg_legend = 0
    fig.flg_ticklabel = 1
    fig.figurergba[:] = _CHART_BG
    fig.panergba[:] = _CHART_BG
    fig.legendrgba[:] = _LEGEND_BG
    fig.textrgb[:] = _TEXT_RGB
    fig.gridrgb[:] = _GRID_RGB

    for line_idx, (name, values_deque) in enumerate(series_map.items()):
        if line_idx >= int(fig.linepnt.shape[0]):
            break
        values = np.asarray(list(values_deque), dtype=np.float32)
        if values.size == 0:
            continue
        xs = _figure_line_xs(int(values.size), float(history_seconds))
        fig.linepnt[line_idx] = int(values.size)
        fig.linename[line_idx] = str(name).encode("utf-8", errors="ignore")[:99]
        color = (
            colors_map.get(name, (230, 233, 237))
            if colors_map is not None
            else (230, 233, 237)
        )
        fig.linergb[line_idx] = np.asarray(color, dtype=np.float32) / 255.0
        fig.linedata[line_idx, : 2 * int(values.size) : 2] = xs
        fig.linedata[line_idx, 1 : 2 * int(values.size) : 2] = values

    return fig


class MujocoGLOverlayRenderer:
    def __init__(self, *, panel_width: int, history_seconds: float) -> None:
        self._config = MujocoGLOverlayConfig(
            panel_width=max(220, int(panel_width)),
            history_seconds=float(history_seconds),
        )

    @property
    def panel_width(self) -> int:
        return int(self._config.panel_width)

    def draw(
        self,
        *,
        context: mujoco.MjrContext,
        full_height: int,
        scene_width: int,
        overlay_state: dict[str, Any],
        episode_length: int,
        step_reward: float,
    ) -> None:
        panel_left = int(scene_width)
        panel_width = int(self._config.panel_width)
        panel_inner_left = panel_left + 12
        panel_inner_right = panel_left + panel_width - 12
        header_height = 92
        bottom_padding = 14
        chart_gap = 12
        legend_line_height = 18
        header_bottom = max(0, int(full_height) - header_height)
        chart_region_top = max(bottom_padding + 1, header_bottom - 8)

        panel_rect = mujoco.MjrRect(
            panel_left,
            0,
            panel_width,
            int(full_height),
        )
        mujoco.mjr_rectangle(
            panel_rect,
            float(_PANEL_BG[0]),
            float(_PANEL_BG[1]),
            float(_PANEL_BG[2]),
            float(_PANEL_BG[3]),
        )

        title = str(overlay_state.get("title") or "Reward Terms")
        left_text = "\n".join(
            [
                title,
                f"episode_step: {int(episode_length)}",
                f"total_reward: {float(step_reward):.3f}",
            ]
        )
        header_rect = mujoco.MjrRect(
            panel_inner_left,
            header_bottom,
            max(1, panel_inner_right - panel_inner_left),
            header_height,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_50,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            header_rect,
            left_text,
            "",
            context,
        )

        plots = overlay_state.get("plots", [])
        if not plots:
            return

        available_height = max(2, chart_region_top - bottom_padding)
        total_gap = chart_gap * max(0, len(plots) - 1)
        chart_height = max(1, (available_height - total_gap) // max(1, len(plots)))
        chart_y = bottom_padding
        for plot_idx, plot in enumerate(plots):
            series_map = dict(plot.get("series", {}))
            legend_width = resolve_overlay_legend_width(
                left=panel_inner_left,
                right=panel_inner_right,
                series_map=series_map,
            )
            is_last = plot_idx == len(plots) - 1
            chart_bottom = (
                chart_region_top
                if is_last
                else min(chart_region_top, chart_y + chart_height)
            )
            fig_rect = mujoco.MjrRect(
                panel_inner_left,
                int(chart_y),
                max(1, panel_inner_right - panel_inner_left - legend_width),
                max(1, int(chart_bottom - chart_y)),
            )
            fig = _build_figure(
                title=str(plot.get("title", f"plot_{plot_idx}")),
                series_map=series_map,
                colors_map=plot.get("colors"),
                history_seconds=float(self._config.history_seconds),
            )
            mujoco.mjr_figure(fig_rect, fig, context)

            legend_items = list(series_map.items())
            if legend_items:
                legend_rect = mujoco.MjrRect(
                    panel_inner_right - legend_width + 10,
                    int(chart_y) + 22,
                    max(1, legend_width - 14),
                    max(1, int(chart_bottom - chart_y) - 28),
                )
                legend_text = "\n".join(
                    "  "
                    + format_overlay_legend_label(
                        str(name),
                        float(values[-1]),
                        legend_width=legend_width,
                    )
                    for name, values in legend_items
                    if len(values) > 0
                )
                if legend_text:
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_50,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        legend_rect,
                        legend_text,
                        "",
                        context,
                    )
                    for item_idx, (name, values) in enumerate(legend_items):
                        if len(values) == 0:
                            continue
                        color = (
                            plot.get("colors", {}).get(name)
                            if isinstance(plot.get("colors"), dict)
                            else None
                        )
                        if color is None:
                            color = (230, 233, 237)
                        swatch_rect = mujoco.MjrRect(
                            int(legend_rect.left),
                            int(
                                legend_rect.bottom
                                + legend_rect.height
                                - 16
                                - item_idx * legend_line_height
                            ),
                            10,
                            10,
                        )
                        mujoco.mjr_rectangle(
                            swatch_rect,
                            float(color[0]) / 255.0,
                            float(color[1]) / 255.0,
                            float(color[2]) / 255.0,
                            1.0,
                        )
            chart_y = chart_bottom + chart_gap
