from __future__ import annotations

import colorsys
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

logger = logging.getLogger(__name__)


_COLOR_INTERPOLATION_MODES = frozenset({"rgb", "hsv"})
_COMPOSITE_MODES = frozenset({"alpha", "max_activation", "unit_mean_alpha"})
_FIBER_SUFFIX_RE = re.compile(r"_fiber\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class MuscleLineGLRenderSource:
    endpoint_body: np.ndarray
    endpoint_local: np.ndarray
    endpoint_muscle: np.ndarray
    endpoint_unit: np.ndarray
    unit_vertex_start: np.ndarray
    unit_vertex_count: np.ndarray
    muscle_count: int
    activation_resolver: Callable[[], torch.Tensor | None]
    activation_clamp_enabled_resolver: Callable[[], float]
    activation_lower_bound: float
    activation_upper_bound: float
    color_low_rgba: np.ndarray
    color_high_rgba: np.ndarray
    color_under_rgba: np.ndarray
    color_over_rgba: np.ndarray
    line_width: float
    color_interpolation: str
    composite_mode: str
    endpoint_attachment_start: np.ndarray | None = None
    endpoint_attachment_body: np.ndarray | None = None
    endpoint_attachment_local_weight: np.ndarray | None = None

    endpoint_world_resolver: Callable | None = None

    @property
    def vertex_count(self) -> int:
        return int(self.endpoint_body.size)

    @property
    def segment_count(self) -> int:
        return self.vertex_count // 2

    @property
    def unit_count(self) -> int:
        return int(self.unit_vertex_start.size)

    @property
    def uses_lbs_endpoints(self) -> bool:
        return (
            self.endpoint_attachment_start is not None
            and self.endpoint_attachment_body is not None
            and self.endpoint_attachment_local_weight is not None
        )

    @property
    def max_referenced_body(self) -> int:
        if self.uses_lbs_endpoints:
            assert self.endpoint_attachment_body is not None
            return int(self.endpoint_attachment_body.max(initial=-1))
        return int(self.endpoint_body.max(initial=-1))

    def resolve_activation_tensor(self) -> torch.Tensor:
        activation = self.activation_resolver()
        if activation is None:
            return torch.zeros((self.muscle_count,), dtype=torch.float32)
        activations = activation[0].detach()
        if self.activation_clamp_enabled_resolver() != 0.0:
            activations = activations.clamp(
                self.activation_lower_bound,
                self.activation_upper_bound,
            )
        if activations.shape[0] != self.muscle_count:
            return torch.zeros((self.muscle_count,), dtype=torch.float32)
        return activations.to(dtype=torch.float32)


def _coerce_rgba(
    color: np.ndarray,
    *,
    fallback_alpha: float,
    name: str,
) -> np.ndarray:
    values = np.asarray(color, dtype=np.float64).reshape(-1)
    if values.size == 3:
        values = np.concatenate(
            [values, np.asarray([float(fallback_alpha)], dtype=np.float64)]
        )
    elif values.size != 4:
        raise ValueError(f"{name} must contain exactly 3 or 4 values.")
    return np.clip(values, 0.0, 1.0)


def _normalize_color_interpolation(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _COLOR_INTERPOLATION_MODES:
        raise ValueError("activation_color_interpolation must be 'rgb' or 'hsv'.")
    return normalized


def _normalize_composite_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _COMPOSITE_MODES:
        raise ValueError(
            "activation_line_composite_mode must be 'alpha', 'max_activation', "
            "or 'unit_mean_alpha'."
        )
    return normalized


def _derive_muscle_unit_indices(
    muscle_names: list[str] | None,
    muscle_count: int,
) -> np.ndarray:
    names = list(muscle_names or [])
    unit_to_index: dict[str, int] = {}
    indices = np.empty((int(muscle_count),), dtype=np.int32)
    for muscle_idx in range(int(muscle_count)):
        if muscle_idx < len(names) and names[muscle_idx]:
            muscle_name = str(names[muscle_idx])
        else:
            muscle_name = f"muscle_{muscle_idx}"
        unit_name = _FIBER_SUFFIX_RE.sub("", muscle_name)
        unit_index = unit_to_index.setdefault(unit_name, len(unit_to_index))
        indices[muscle_idx] = int(unit_index)
    return indices


def _unit_vertex_ranges(segment_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if segment_unit.size == 0:
        empty = np.zeros((0,), dtype=np.int32)
        return empty, empty
    unit_change = np.flatnonzero(
        np.concatenate(
            [
                np.ones((1,), dtype=bool),
                segment_unit[1:] != segment_unit[:-1],
            ]
        )
    )
    next_change = np.concatenate(
        [unit_change[1:], np.asarray([segment_unit.size], dtype=np.int64)]
    )
    return (
        (unit_change * 2).astype(np.int32, copy=False),
        ((next_change - unit_change) * 2).astype(np.int32, copy=False),
    )


def _interpolate_rgba(
    low_rgba: np.ndarray,
    high_rgba: np.ndarray,
    values: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    t = np.asarray(values, dtype=np.float64).reshape(-1)
    if mode == "rgb":
        out = low_rgba[None, :] * (1.0 - t[:, None]) + high_rgba[None, :] * t[:, None]
        return out.astype(np.float32, copy=False)

    low_hsv = colorsys.rgb_to_hsv(*low_rgba[:3])
    high_hsv = colorsys.rgb_to_hsv(*high_rgba[:3])
    hue_delta = (high_hsv[0] - low_hsv[0] + 0.5) % 1.0 - 0.5
    hue = (low_hsv[0] + hue_delta * t) % 1.0
    saturation = low_hsv[1] * (1.0 - t) + high_hsv[1] * t
    value = low_hsv[2] * (1.0 - t) + high_hsv[2] * t
    out = np.empty((t.size, 4), dtype=np.float32)
    for idx in range(t.size):
        out[idx, :3] = colorsys.hsv_to_rgb(
            float(hue[idx]),
            float(saturation[idx]),
            float(value[idx]),
        )
    out[:, 3] = low_rgba[3] * (1.0 - t) + high_rgba[3] * t
    return out


def _rotate_wxyz_batch(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    qvec = quat[:, 1:4]
    t = 2.0 * np.cross(qvec, vec)
    return vec + quat[:, 0:1] * t + np.cross(qvec, t)


class MuscleLineSceneVisualizer:
    def __init__(
        self,
        *,
        muscle_count: int,
        activation_resolver: Callable[[], torch.Tensor | None],
        activation_clamp_enabled_resolver: Callable[[], float],
        activation_lower_bound: float,
        activation_upper_bound: float,
        color_low: np.ndarray,
        color_high: np.ndarray,
        color_under: np.ndarray,
        color_over: np.ndarray,
        alpha: float,
        waypoint_body: np.ndarray,
        waypoints_local: np.ndarray,
        waypoint_start: np.ndarray,
        waypoint_world_resolver: Callable | None = None,
        waypoint_attachment_start: np.ndarray | None = None,
        waypoint_attachment_body: np.ndarray | None = None,
        waypoint_attachment_local: np.ndarray | None = None,
        waypoint_attachment_weight: np.ndarray | None = None,
        line_width: float = 3.0,
        color_interpolation: str = "rgb",
        composite_mode: str = "alpha",
        muscle_names: list[str] | None = None,
    ) -> None:
        self._muscle_count = int(muscle_count)
        self._activation_resolver = activation_resolver
        self._activation_clamp_enabled_resolver = activation_clamp_enabled_resolver
        self._activation_lower_bound = float(activation_lower_bound)
        self._activation_upper_bound = float(activation_upper_bound)
        self._alpha = float(alpha)
        self._line_width = float(line_width)
        self._color_interpolation = _normalize_color_interpolation(color_interpolation)
        self._composite_mode = _normalize_composite_mode(composite_mode)
        self._muscle_unit_index = _derive_muscle_unit_indices(
            muscle_names,
            self._muscle_count,
        )
        self._color_low_rgba = _coerce_rgba(
            color_low,
            fallback_alpha=self._alpha,
            name="color_low",
        )
        self._color_high_rgba = _coerce_rgba(
            color_high,
            fallback_alpha=self._alpha,
            name="color_high",
        )
        self._color_under_rgba = _coerce_rgba(
            color_under,
            fallback_alpha=self._alpha,
            name="color_under",
        )
        self._color_over_rgba = _coerce_rgba(
            color_over,
            fallback_alpha=self._alpha,
            name="color_over",
        )
        self._waypoint_world_resolver = waypoint_world_resolver
        self._endpoint_indices_by_device = {}
        self._waypoint_body = np.asarray(waypoint_body, dtype=np.int32).reshape(-1)
        self._waypoints_local = np.asarray(waypoints_local, dtype=np.float64).reshape(
            -1, 3
        )
        self._waypoint_start = np.asarray(waypoint_start, dtype=np.int32).reshape(-1)
        self._waypoint_attachment_start: np.ndarray | None = None
        self._waypoint_attachment_body: np.ndarray | None = None
        self._waypoint_attachment_local: np.ndarray | None = None
        self._waypoint_attachment_weight: np.ndarray | None = None
        attachment_args = (
            waypoint_attachment_start,
            waypoint_attachment_body,
            waypoint_attachment_local,
            waypoint_attachment_weight,
        )
        if any(arg is not None for arg in attachment_args):
            if not all(arg is not None for arg in attachment_args):
                raise ValueError(
                    "LBS waypoint rendering requires start, body, local, and weight arrays."
                )
            attachment_start = np.asarray(
                waypoint_attachment_start, dtype=np.int32
            ).reshape(-1)
            attachment_body = np.asarray(
                waypoint_attachment_body, dtype=np.int32
            ).reshape(-1)
            attachment_local = np.asarray(
                waypoint_attachment_local, dtype=np.float64
            ).reshape(-1, 3)
            attachment_weight = np.asarray(
                waypoint_attachment_weight, dtype=np.float64
            ).reshape(-1)
            if attachment_start.size != self._waypoints_local.shape[0] + 1:
                raise ValueError(
                    "LBS waypoint attachment start size must equal waypoint_count + 1."
                )
            if int(attachment_start[0]) != 0 or int(attachment_start[-1]) != int(
                attachment_body.size
            ):
                raise ValueError("Invalid LBS waypoint attachment start offsets.")
            if (
                attachment_local.shape[0] != attachment_body.size
                or attachment_weight.size != attachment_body.size
            ):
                raise ValueError(
                    "LBS waypoint attachment arrays must have equal length."
                )
            self._waypoint_attachment_start = attachment_start
            self._waypoint_attachment_body = attachment_body
            self._waypoint_attachment_local = attachment_local
            self._waypoint_attachment_weight = attachment_weight

        segment_start: list[int] = []
        segment_end: list[int] = []
        segment_muscle: list[int] = []
        segment_unit: list[int] = []
        for muscle_idx in range(min(self._muscle_count, self._waypoint_start.size - 1)):
            start = int(self._waypoint_start[muscle_idx])
            end = int(self._waypoint_start[muscle_idx + 1])
            for waypoint_idx in range(start, max(start, end - 1)):
                segment_start.append(waypoint_idx)
                segment_end.append(waypoint_idx + 1)
                segment_muscle.append(muscle_idx)
                segment_unit.append(int(self._muscle_unit_index[muscle_idx]))

        self._segment_start = np.asarray(segment_start, dtype=np.int32)
        self._segment_end = np.asarray(segment_end, dtype=np.int32)
        self._segment_muscle = np.asarray(segment_muscle, dtype=np.int32)
        self._segment_unit = np.asarray(segment_unit, dtype=np.int32)
        if self._composite_mode == "unit_mean_alpha" and self._segment_unit.size > 0:
            order = np.argsort(self._segment_unit, kind="stable")
            self._segment_start = self._segment_start[order]
            self._segment_end = self._segment_end[order]
            self._segment_muscle = self._segment_muscle[order]
            self._segment_unit = self._segment_unit[order]
        self._unit_vertex_start, self._unit_vertex_count = _unit_vertex_ranges(
            self._segment_unit
        )
        endpoint_waypoint = np.empty((self._segment_start.size * 2,), dtype=np.int32)
        endpoint_waypoint[0::2] = self._segment_start
        endpoint_waypoint[1::2] = self._segment_end
        self._endpoint_waypoint = endpoint_waypoint
        self._endpoint_body = self._waypoint_body[endpoint_waypoint].astype(
            np.int32,
            copy=False,
        )
        self._endpoint_local = self._waypoints_local[endpoint_waypoint].astype(
            np.float32,
            copy=False,
        )
        self._endpoint_muscle = np.repeat(self._segment_muscle, 2).astype(
            np.int32,
            copy=False,
        )
        self._endpoint_unit = np.repeat(self._segment_unit, 2).astype(
            np.int32,
            copy=False,
        )
        self._endpoint_attachment_start: np.ndarray | None = None
        self._endpoint_attachment_body: np.ndarray | None = None
        self._endpoint_attachment_local_weight: np.ndarray | None = None
        if self._waypoint_attachment_start is not None:
            endpoint_attachment_start = [0]
            endpoint_attachment_body: list[int] = []
            endpoint_attachment_local_weight: list[np.ndarray] = []
            assert self._waypoint_attachment_body is not None
            assert self._waypoint_attachment_local is not None
            assert self._waypoint_attachment_weight is not None
            for waypoint_idx in endpoint_waypoint:
                begin = int(self._waypoint_attachment_start[int(waypoint_idx)])
                end = int(self._waypoint_attachment_start[int(waypoint_idx) + 1])
                endpoint_attachment_body.extend(
                    int(v) for v in self._waypoint_attachment_body[begin:end]
                )
                local = self._waypoint_attachment_local[begin:end].astype(
                    np.float32,
                    copy=False,
                )
                weight = self._waypoint_attachment_weight[begin:end].astype(
                    np.float32,
                    copy=False,
                )
                local_weight = np.empty((end - begin, 4), dtype=np.float32)
                local_weight[:, :3] = local
                local_weight[:, 3] = weight
                endpoint_attachment_local_weight.append(local_weight)
                endpoint_attachment_start.append(len(endpoint_attachment_body))
            self._endpoint_attachment_start = np.asarray(
                endpoint_attachment_start,
                dtype=np.int32,
            )
            self._endpoint_attachment_body = np.asarray(
                endpoint_attachment_body,
                dtype=np.int32,
            )
            if endpoint_attachment_local_weight:
                self._endpoint_attachment_local_weight = np.ascontiguousarray(
                    np.vstack(endpoint_attachment_local_weight),
                    dtype=np.float32,
                )
            else:
                self._endpoint_attachment_local_weight = np.zeros(
                    (0, 4),
                    dtype=np.float32,
                )
        self._warned_scene_capacity = False

    @property
    def segment_count(self) -> int:
        return int(self._segment_start.size)

    def build_gl_render_source(self) -> MuscleLineGLRenderSource:
        return MuscleLineGLRenderSource(
            endpoint_world_resolver=self._world_endpoints
            if self._waypoint_world_resolver is not None
            else None,
            endpoint_body=self._endpoint_body,
            endpoint_local=self._endpoint_local,
            endpoint_muscle=self._endpoint_muscle,
            endpoint_unit=self._endpoint_unit,
            unit_vertex_start=self._unit_vertex_start,
            unit_vertex_count=self._unit_vertex_count,
            muscle_count=self._muscle_count,
            activation_resolver=self._activation_resolver,
            activation_clamp_enabled_resolver=self._activation_clamp_enabled_resolver,
            activation_lower_bound=self._activation_lower_bound,
            activation_upper_bound=self._activation_upper_bound,
            color_low_rgba=self._color_low_rgba,
            color_high_rgba=self._color_high_rgba,
            color_under_rgba=self._color_under_rgba,
            color_over_rgba=self._color_over_rgba,
            line_width=self._line_width,
            color_interpolation=self._color_interpolation,
            composite_mode=self._composite_mode,
            endpoint_attachment_start=self._endpoint_attachment_start,
            endpoint_attachment_body=self._endpoint_attachment_body,
            endpoint_attachment_local_weight=self._endpoint_attachment_local_weight,
        )

    def _resolve_activation_tensor(self) -> torch.Tensor | None:
        activation = self._activation_resolver()
        if activation is None:
            return None
        activations = activation[0].detach()
        if self._activation_clamp_enabled_resolver() != 0.0:
            activations = activations.clamp(
                self._activation_lower_bound,
                self._activation_upper_bound,
            )
        if activations.is_cuda:
            activations = activations.cpu()
        if activations.shape[0] != self._muscle_count:
            return None
        return activations.to(dtype=torch.float32)

    def _resolve_activation_np(self) -> np.ndarray | None:
        activations = self._resolve_activation_tensor()
        if activations is None:
            return None
        if activations.is_cuda:
            activations = activations.cpu()
        return activations.numpy()

    def _resolve_muscle_rgba(self) -> np.ndarray:
        activation = self._resolve_activation_np()
        if activation is None:
            activation = np.zeros((self._muscle_count,), dtype=np.float32)

        clipped = np.clip(
            activation,
            self._activation_lower_bound,
            self._activation_upper_bound,
        )
        rgba = _interpolate_rgba(
            self._color_low_rgba,
            self._color_high_rgba,
            clipped,
            mode=self._color_interpolation,
        )
        under_mag = np.maximum(-activation, 0.0)
        over_mag = np.maximum(activation - 1.0, 0.0)
        under_mask = (under_mag > 0.0) & (under_mag >= over_mag)
        over_mask = over_mag > under_mag
        rgba[under_mask, :] = self._color_under_rgba
        rgba[over_mask, :] = self._color_over_rgba
        return rgba

    def _world_endpoints(self, data):
        points = self._waypoint_world_resolver(data)
        key = (str(points.device), points.shape[0])
        cached = self._endpoint_indices_by_device.get(key)
        if cached is None:
            indices = torch.as_tensor(
                self._endpoint_waypoint, dtype=torch.long, device=points.device
            )
            output = torch.empty(
                (points.shape[0], len(indices), 3), device=points.device
            )
            cached = (indices, output)
            self._endpoint_indices_by_device[key] = cached
        indices, output = cached
        torch.index_select(points, 1, indices, out=output)
        return output

    def _world_waypoints(
        self,
        render_data: mujoco.MjData,
    ) -> np.ndarray:
        if self._waypoint_world_resolver is not None:
            return self._waypoint_world_resolver(render_data)[0].numpy()
        if self._waypoint_attachment_start is not None:
            assert self._waypoint_attachment_body is not None
            assert self._waypoint_attachment_local is not None
            assert self._waypoint_attachment_weight is not None
            body_ids = self._waypoint_attachment_body
            body_pos = np.asarray(render_data.xpos, dtype=np.float64)[body_ids]
            body_quat = np.asarray(render_data.xquat, dtype=np.float64)[body_ids]
            attachment_world = body_pos + _rotate_wxyz_batch(
                body_quat,
                self._waypoint_attachment_local,
            )
            attachment_world *= self._waypoint_attachment_weight[:, None]
            points = np.zeros_like(self._waypoints_local, dtype=np.float64)
            counts = (
                self._waypoint_attachment_start[1:]
                - self._waypoint_attachment_start[:-1]
            )
            waypoint_idx = np.repeat(
                np.arange(self._waypoints_local.shape[0], dtype=np.int32),
                counts,
            )
            np.add.at(points, waypoint_idx, attachment_world)
            return points
        body_ids = self._waypoint_body
        body_pos = np.asarray(render_data.xpos, dtype=np.float64)[body_ids]
        body_quat = np.asarray(render_data.xquat, dtype=np.float64)[body_ids]
        return body_pos + _rotate_wxyz_batch(body_quat, self._waypoints_local)

    def append_to_scene(
        self,
        *,
        scene: mujoco.MjvScene,
        render_data: mujoco.MjData,
    ) -> None:
        if self._segment_start.size == 0:
            return
        remaining = int(scene.maxgeom) - int(scene.ngeom)
        if remaining <= 0:
            return

        count = min(int(self._segment_start.size), remaining)
        if count < int(self._segment_start.size) and not self._warned_scene_capacity:
            logger.warning(
                "Muscle line render truncated: scene has room for %d/%d segments.",
                count,
                int(self._segment_start.size),
            )
            self._warned_scene_capacity = True

        points = self._world_waypoints(render_data)
        muscle_rgba = self._resolve_muscle_rgba()
        size = np.zeros(3, dtype=np.float64)
        pos = np.zeros(3, dtype=np.float64)
        mat = np.eye(3, dtype=np.float64).reshape(9)
        geom_type = int(mujoco.mjtGeom.mjGEOM_LINE)

        for segment_idx in range(count):
            geom_idx = int(scene.ngeom)
            geom = scene.geoms[geom_idx]
            rgba = muscle_rgba[int(self._segment_muscle[segment_idx])]
            mujoco.mjv_initGeom(geom, geom_type, size, pos, mat, rgba)
            mujoco.mjv_connector(
                geom,
                geom_type,
                self._line_width,
                points[int(self._segment_start[segment_idx])],
                points[int(self._segment_end[segment_idx])],
            )
            geom.rgba[:] = rgba
            scene.ngeom = geom_idx + 1


class TendonActivationVisualizer:
    def __init__(
        self,
        *,
        muscle_names: list[str],
        muscle_count: int,
        activation_resolver: Callable[[], torch.Tensor | None],
        activation_clamp_enabled_resolver: Callable[[], float],
        activation_lower_bound: float,
        activation_upper_bound: float,
        color_low: np.ndarray,
        color_high: np.ndarray,
        color_under: np.ndarray,
        color_over: np.ndarray,
        alpha: float,
        waypoint_body: np.ndarray | None = None,
        waypoints_local: np.ndarray | None = None,
        waypoint_start: np.ndarray | None = None,
        waypoint_world_resolver: Callable | None = None,
        waypoint_attachment_start: np.ndarray | None = None,
        waypoint_attachment_body: np.ndarray | None = None,
        waypoint_attachment_local: np.ndarray | None = None,
        waypoint_attachment_weight: np.ndarray | None = None,
        line_width: float = 3.0,
        color_interpolation: str = "rgb",
        composite_mode: str = "alpha",
    ) -> None:
        self._muscle_names = list(muscle_names)
        self._muscle_count = int(muscle_count)
        self._activation_resolver = activation_resolver
        self._activation_clamp_enabled_resolver = activation_clamp_enabled_resolver
        self._activation_lower_bound = float(activation_lower_bound)
        self._activation_upper_bound = float(activation_upper_bound)
        self._alpha = float(alpha)
        self._color_interpolation = _normalize_color_interpolation(color_interpolation)
        self._composite_mode = _normalize_composite_mode(composite_mode)
        self._color_low_rgba = _coerce_rgba(
            color_low,
            fallback_alpha=self._alpha,
            name="color_low",
        )
        self._color_high_rgba = _coerce_rgba(
            color_high,
            fallback_alpha=self._alpha,
            name="color_high",
        )
        self._color_under_rgba = _coerce_rgba(
            color_under,
            fallback_alpha=self._alpha,
            name="color_under",
        )
        self._color_over_rgba = _coerce_rgba(
            color_over,
            fallback_alpha=self._alpha,
            name="color_over",
        )
        self._enabled = True
        self._render_model: mujoco.MjModel | None = None
        self._base_rgba: np.ndarray | None = None
        self._muscle_to_render_tendon = np.zeros((0,), dtype=np.int32)
        self._mapped_render_tendon_ids = np.zeros((0,), dtype=np.int32)
        self._use_line_scene = False
        self._line_visualizer = (
            MuscleLineSceneVisualizer(
                muscle_count=muscle_count,
                activation_resolver=activation_resolver,
                activation_clamp_enabled_resolver=activation_clamp_enabled_resolver,
                activation_lower_bound=activation_lower_bound,
                activation_upper_bound=activation_upper_bound,
                color_low=color_low,
                color_high=color_high,
                color_under=color_under,
                color_over=color_over,
                alpha=alpha,
                waypoint_body=waypoint_body,
                waypoints_local=waypoints_local,
                waypoint_start=waypoint_start,
                waypoint_world_resolver=waypoint_world_resolver,
                waypoint_attachment_start=waypoint_attachment_start,
                waypoint_attachment_body=waypoint_attachment_body,
                waypoint_attachment_local=waypoint_attachment_local,
                waypoint_attachment_weight=waypoint_attachment_weight,
                line_width=line_width,
                color_interpolation=color_interpolation,
                composite_mode=composite_mode,
                muscle_names=muscle_names,
            )
            if waypoint_start is not None
            and waypoint_body is not None
            and waypoints_local is not None
            else None
        )

    def attach(self, render_model: mujoco.MjModel) -> None:
        self._render_model = render_model
        self._base_rgba = render_model.tendon_rgba.copy()

        if render_model.ntendon <= 0:
            self._use_line_scene = self._line_visualizer is not None
            self._enabled = self._use_line_scene
            return

        self._use_line_scene = False
        self._enabled = True
        self._muscle_to_render_tendon = np.full(
            (self._muscle_count,),
            -1,
            dtype=np.int32,
        )
        for muscle_idx, muscle_name in enumerate(self._muscle_names):
            tendon_render_id = mujoco.mj_name2id(
                render_model,
                mujoco.mjtObj.mjOBJ_TENDON,
                muscle_name,
            )
            if tendon_render_id >= 0:
                self._muscle_to_render_tendon[muscle_idx] = int(tendon_render_id)

        mapped = self._muscle_to_render_tendon[self._muscle_to_render_tendon >= 0]
        if mapped.size == 0 and render_model.ntendon == self._muscle_count:
            self._muscle_to_render_tendon[:] = np.arange(
                self._muscle_count,
                dtype=np.int32,
            )
            mapped = self._muscle_to_render_tendon

        self._mapped_render_tendon_ids = np.unique(mapped)
        if self._mapped_render_tendon_ids.size == 0:
            self._enabled = False

    def update(self) -> None:
        if self._use_line_scene:
            return
        if not self._enabled or self._render_model is None or self._base_rgba is None:
            return

        activation = self._activation_resolver()
        if activation is None:
            self._render_model.tendon_rgba[:] = self._base_rgba
            return

        activations = activation[0].detach()
        if self._activation_clamp_enabled_resolver() != 0.0:
            activations = activations.clamp(
                self._activation_lower_bound,
                self._activation_upper_bound,
            )
        if activations.is_cuda:
            activations = activations.cpu()
        activations_np = activations.numpy()
        if activations_np.shape[0] != self._muscle_count:
            self._render_model.tendon_rgba[:] = self._base_rgba
            return

        if activations_np.min() < 0.0 or activations_np.max() > 1.0:
            logger.warning(
                "Activation out of [0,1]: min=%.4f max=%.4f",
                float(activations_np.min()),
                float(activations_np.max()),
            )

        tendon_act = np.zeros(self._render_model.ntendon, dtype=np.float64)
        tendon_under = np.zeros_like(tendon_act)
        tendon_over = np.zeros_like(tendon_act)
        valid = self._muscle_to_render_tendon >= 0
        if np.any(valid):
            mapped_valid = self._muscle_to_render_tendon[valid]
            activation_valid = activations_np[valid]
            np.maximum.at(
                tendon_act,
                mapped_valid,
                np.clip(
                    activation_valid,
                    self._activation_lower_bound,
                    self._activation_upper_bound,
                ),
            )
            np.maximum.at(
                tendon_under,
                mapped_valid,
                np.maximum(-activation_valid, 0.0),
            )
            np.maximum.at(
                tendon_over,
                mapped_valid,
                np.maximum(activation_valid - 1.0, 0.0),
            )

        out_rgba = self._base_rgba.copy()
        tendon_ids = self._mapped_render_tendon_ids
        activation_values = tendon_act[tendon_ids]
        out_rgba[tendon_ids, :] = _interpolate_rgba(
            self._color_low_rgba,
            self._color_high_rgba,
            activation_values,
            mode=self._color_interpolation,
        )
        under_mag = tendon_under[tendon_ids]
        over_mag = tendon_over[tendon_ids]
        under_mask = (under_mag > 0.0) & (under_mag >= over_mag)
        over_mask = over_mag > under_mag
        out_rgba[tendon_ids[under_mask], :] = self._color_under_rgba
        out_rgba[tendon_ids[over_mask], :] = self._color_over_rgba
        self._render_model.tendon_rgba[:] = out_rgba

    def append_to_scene(
        self,
        *,
        scene: mujoco.MjvScene,
        render_data: mujoco.MjData,
    ) -> None:
        if self._use_line_scene and self._line_visualizer is not None:
            self._line_visualizer.append_to_scene(
                scene=scene,
                render_data=render_data,
            )

    def scene_geom_capacity_hint(self) -> int:
        if self._line_visualizer is None:
            return 0
        return self._line_visualizer.segment_count

    def gl_render_sources(self) -> list[MuscleLineGLRenderSource]:
        if self._use_line_scene and self._line_visualizer is not None:
            return [self._line_visualizer.build_gl_render_source()]
        return []
