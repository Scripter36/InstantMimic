"""Viser-backed interactive playback support for PPO play."""

from __future__ import annotations

import colorsys
import html
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from instantmimic.envs.config_access import get_value as _cfg_get
from instantmimic.rendering.character import (
    classify_character_geom_semantic,
    resolve_character_render_mode,
)

logger = logging.getLogger(__name__)

_VISER_SPEEDS = (1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0)
_SEMANTIC_GROUP_LABELS = {
    "body": "Body mesh",
    "collision": "Collision mesh",
    "ghost": "Ghost skeleton",
    "scene": "Scene / task objects",
}
_DEFAULT_VISER_BACKGROUND_RGB = (13, 17, 23)
_DEFAULT_VISER_GRID_CELL_RGB = (42, 48, 58)
_DEFAULT_VISER_GRID_SECTION_RGB = (76, 86, 102)
_DEFAULT_VISER_GRID_PLANE_RGB = (18, 22, 28)
_DEFAULT_VISER_GRID_PLANE_OPACITY = 0.72
_DEFAULT_VISER_GRID_SHADOW_OPACITY = 0.22
_DEFAULT_VISER_ENVIRONMENT_INTENSITY = 0.8
_DEFAULT_VISER_BODY_MESH_RGB = (146, 156, 172)
_DEFAULT_VISER_BODY_MESH_OPACITY = 1.0
_DEFAULT_VISER_COLLISION_MESH_RGB = (82, 94, 112)
_DEFAULT_VISER_COLLISION_MESH_OPACITY = 0.38
_DEFAULT_VISER_GHOST_MESH_RGB = (92, 138, 196)
_DEFAULT_VISER_GHOST_MESH_OPACITY = 0.55


def resolve_viewer_backend(cfg: Any) -> str:
    backend = str(cfg.get("viewer_backend", "mujoco")).strip().lower()
    if backend not in {"mujoco", "viser"}:
        raise ValueError("viewer_backend must be one of: mujoco, viser.")
    return backend


def _color_to_css(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 7 and text.startswith("#"):
            return text
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            rgb = tuple(max(0, min(255, int(channel))) for channel in value)
        except (TypeError, ValueError):
            return None
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    return None


def build_uplot_data_from_overlay_plot(
    plot: dict[str, Any],
    *,
    history_seconds: float,
) -> tuple[np.ndarray, ...]:
    series_map = plot.get("series", {})
    if not isinstance(series_map, dict) or not series_map:
        return (np.zeros((0,), dtype=np.float64),)

    labels = [str(label) for label in series_map]
    values_by_label = {
        label: np.asarray(list(series_map[label]), dtype=np.float64) for label in labels
    }
    sample_count = max(
        (int(values.size) for values in values_by_label.values()),
        default=0,
    )
    if sample_count <= 0:
        return (np.zeros((0,), dtype=np.float64),)

    step_seconds = float(history_seconds) / float(max(1, sample_count))
    x_data = np.linspace(
        -step_seconds * float(sample_count - 1),
        0.0,
        num=sample_count,
        dtype=np.float64,
    )
    data: list[np.ndarray] = [x_data]
    for label in labels:
        values = values_by_label[label]
        padded = np.full((sample_count,), np.nan, dtype=np.float64)
        if values.size:
            clipped = values[-sample_count:]
            padded[-int(clipped.size) :] = clipped
        data.append(padded)
    return tuple(data)


def _build_uplot_series(plot: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    series_map = plot.get("series", {})
    colors = plot.get("colors", {})
    series: list[dict[str, Any]] = [{"label": "time"}]
    if not isinstance(series_map, dict):
        return tuple(series)
    for label in series_map:
        item: dict[str, Any] = {"label": str(label), "width": 2}
        if isinstance(colors, dict):
            color = _color_to_css(colors.get(label))
            if color is not None:
                item["stroke"] = color
        series.append(item)
    return tuple(series)


def _coerce_rgb_u8(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 7 and text.startswith("#"):
            try:
                return (
                    int(text[1:3], 16),
                    int(text[3:5], 16),
                    int(text[5:7], 16),
                )
            except ValueError:
                return default
        return default

    try:
        channels = list(value)
    except TypeError:
        return default
    if len(channels) != 3:
        return default
    try:
        return tuple(max(0, min(255, int(channel))) for channel in channels)
    except (TypeError, ValueError):
        return default


def _coerce_float(
    value: Any,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(default)
    if minimum is not None:
        resolved = max(float(minimum), resolved)
    if maximum is not None:
        resolved = min(float(maximum), resolved)
    return resolved


def _mujoco_name(model: Any, obj_type: Any, item_id: int, fallback: str) -> str:
    import mujoco

    name = mujoco.mj_id2name(model, obj_type, int(item_id))
    return str(name) if name else fallback


def _body_name(model: Any, body_id: int) -> str:
    try:
        import mujoco

        return _mujoco_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(body_id),
            f"body_{int(body_id)}",
        )
    except Exception:  # noqa: BLE001 - viewer name lookup fallback must tolerate backend variants.
        return ""


def _body_is_ghost(model: Any, body_id: int) -> bool:
    body_mocapid = getattr(model, "body_mocapid", None)
    if body_mocapid is not None and int(body_mocapid[int(body_id)]) >= 0:
        return True
    return _body_name(model, int(body_id)).startswith("ghost_")


def classify_viser_geom_semantic_group(
    model: Any,
    geom_ids: list[int] | np.ndarray,
    *,
    body_id: int,
    env_raw: Any | None = None,
) -> str:
    semantic = classify_character_geom_semantic(
        model,
        geom_ids,
        body_id=int(body_id),
        env_raw=env_raw,
    )
    if semantic == "scene":
        return "scene"
    if semantic == "collision":
        return "collision"
    if _body_is_ghost(model, int(body_id)):
        return "ghost"
    return "body"


def _geom_ids_for_body_group(
    model: Any,
    *,
    body_ids: np.ndarray,
    group_id: int,
) -> np.ndarray:
    geom_bodyid = np.asarray(model.geom_bodyid)
    geom_group = np.asarray(model.geom_group)
    body_ids_np = np.asarray(body_ids, dtype=np.int32).reshape(-1)
    mask = np.isin(geom_bodyid, body_ids_np) & (geom_group == int(group_id))
    return np.flatnonzero(mask).astype(np.int32)


def resolve_viser_semantic_group_defaults(env_raw: Any) -> dict[str, bool]:
    cfg = getattr(env_raw, "cfg", None)
    mode = resolve_character_render_mode(cfg)
    show_mesh = mode in {"mesh", "both"}
    show_collision = mode in {"collision", "both"}
    return {
        "body": show_mesh,
        "collision": show_collision,
        "ghost": show_mesh,
        "scene": True,
    }


def _is_fixed_body_for_mjviser(model: Any, body_id: int) -> bool:
    try:
        from mjviser.scene import is_fixed_body

        return bool(is_fixed_body(model, int(body_id)))
    except Exception:  # noqa: BLE001 - viewer fixed-body helper is optional/backend-specific.
        return int(body_id) == 0


def _iter_mjviser_fixed_plane_grid_specs(model: Any):
    import mujoco

    geom_type = np.asarray(model.geom_type)
    geom_bodyid = np.asarray(model.geom_bodyid)
    for geom_id in range(int(getattr(model, "ngeom", 0))):
        if int(geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        body_id = int(geom_bodyid[geom_id])
        if not _is_fixed_body_for_mjviser(model, body_id):
            continue
        body_name = _mujoco_name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            f"body_{body_id}",
        )
        geom_name = _mujoco_name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
            f"geom_{geom_id}",
        )
        yield (
            f"/fixed_bodies/{body_name}/{geom_name}",
            np.asarray(model.geom_pos[geom_id], dtype=np.float64),
            np.asarray(model.geom_quat[geom_id], dtype=np.float64),
        )


def configure_viser_scene_theme(scene: Any, model: Any, cfg: Any) -> None:
    if not bool(_cfg_get(cfg, "viser_dark_theme", True)):
        return

    background_rgb = _coerce_rgb_u8(
        _cfg_get(cfg, "viser_background_rgb", _DEFAULT_VISER_BACKGROUND_RGB),
        _DEFAULT_VISER_BACKGROUND_RGB,
    )
    set_background_image = getattr(scene, "set_background_image", None)
    if callable(set_background_image):
        background_image = np.asarray([[background_rgb]], dtype=np.uint8)
        set_background_image(background_image, format="png")

    configure_environment_map = getattr(scene, "configure_environment_map", None)
    if callable(configure_environment_map):
        configure_environment_map(
            background=False,
            environment_intensity=_coerce_float(
                _cfg_get(
                    cfg,
                    "viser_environment_intensity",
                    _DEFAULT_VISER_ENVIRONMENT_INTENSITY,
                ),
                _DEFAULT_VISER_ENVIRONMENT_INTENSITY,
                minimum=0.0,
            ),
        )

    add_grid = getattr(scene, "add_grid", None)
    if not callable(add_grid):
        return

    remove_by_name = getattr(scene, "remove_by_name", None)
    cell_color = _coerce_rgb_u8(
        _cfg_get(cfg, "viser_grid_cell_rgb", _DEFAULT_VISER_GRID_CELL_RGB),
        _DEFAULT_VISER_GRID_CELL_RGB,
    )
    section_color = _coerce_rgb_u8(
        _cfg_get(cfg, "viser_grid_section_rgb", _DEFAULT_VISER_GRID_SECTION_RGB),
        _DEFAULT_VISER_GRID_SECTION_RGB,
    )
    plane_color = _coerce_rgb_u8(
        _cfg_get(cfg, "viser_grid_plane_rgb", _DEFAULT_VISER_GRID_PLANE_RGB),
        _DEFAULT_VISER_GRID_PLANE_RGB,
    )
    plane_opacity = _coerce_float(
        _cfg_get(
            cfg,
            "viser_grid_plane_opacity",
            _DEFAULT_VISER_GRID_PLANE_OPACITY,
        ),
        _DEFAULT_VISER_GRID_PLANE_OPACITY,
        minimum=0.0,
        maximum=1.0,
    )
    shadow_opacity = _coerce_float(
        _cfg_get(
            cfg,
            "viser_grid_shadow_opacity",
            _DEFAULT_VISER_GRID_SHADOW_OPACITY,
        ),
        _DEFAULT_VISER_GRID_SHADOW_OPACITY,
        minimum=0.0,
        maximum=1.0,
    )

    for name, position, wxyz in _iter_mjviser_fixed_plane_grid_specs(model):
        if callable(remove_by_name):
            remove_by_name(name)
        add_grid(
            name,
            infinite_grid=True,
            fade_distance=50.0,
            shadow_opacity=shadow_opacity,
            plane_opacity=plane_opacity,
            plane_color=plane_color,
            cell_color=cell_color,
            section_color=section_color,
            position=position,
            wxyz=wxyz,
        )


def _semantic_mesh_rgb_and_opacity(
    semantic_group: str,
    cfg: Any,
) -> tuple[tuple[int, int, int], float]:
    if semantic_group == "collision":
        return (
            _coerce_rgb_u8(
                _cfg_get(
                    cfg,
                    "viser_collision_mesh_rgb",
                    _DEFAULT_VISER_COLLISION_MESH_RGB,
                ),
                _DEFAULT_VISER_COLLISION_MESH_RGB,
            ),
            _coerce_float(
                _cfg_get(
                    cfg,
                    "viser_collision_mesh_opacity",
                    _DEFAULT_VISER_COLLISION_MESH_OPACITY,
                ),
                _DEFAULT_VISER_COLLISION_MESH_OPACITY,
                minimum=0.0,
                maximum=1.0,
            ),
        )
    if semantic_group == "ghost":
        return (
            _coerce_rgb_u8(
                _cfg_get(cfg, "viser_ghost_mesh_rgb", _DEFAULT_VISER_GHOST_MESH_RGB),
                _DEFAULT_VISER_GHOST_MESH_RGB,
            ),
            _coerce_float(
                _cfg_get(
                    cfg,
                    "viser_ghost_mesh_opacity",
                    _DEFAULT_VISER_GHOST_MESH_OPACITY,
                ),
                _DEFAULT_VISER_GHOST_MESH_OPACITY,
                minimum=0.0,
                maximum=1.0,
            ),
        )
    return (
        _coerce_rgb_u8(
            _cfg_get(cfg, "viser_body_mesh_rgb", _DEFAULT_VISER_BODY_MESH_RGB),
            _DEFAULT_VISER_BODY_MESH_RGB,
        ),
        _coerce_float(
            _cfg_get(cfg, "viser_body_mesh_opacity", _DEFAULT_VISER_BODY_MESH_OPACITY),
            _DEFAULT_VISER_BODY_MESH_OPACITY,
            minimum=0.0,
            maximum=1.0,
        ),
    )


def _subgroup_index_from_handle_name(handle: Any) -> int:
    name = str(getattr(handle, "name", ""))
    leaf = name.rsplit("/", 1)[-1]
    if not leaf.startswith("sub"):
        return 0
    try:
        return max(0, int(leaf[3:]))
    except ValueError:
        return 0


def _semantic_mesh_batched_opacities(
    *,
    batch_count: int,
    opacity: float,
) -> np.ndarray | None:
    if opacity >= 0.999:
        return None
    return np.full((int(batch_count),), float(opacity), dtype=np.float32)


def configure_viser_mujoco_scene_mesh_style(
    mujoco_scene: Any,
    cfg: Any,
    *,
    env_raw: Any | None = None,
) -> None:
    if not bool(_cfg_get(cfg, "viser_dark_theme", True)):
        return
    if not bool(_cfg_get(cfg, "viser_recolor_meshes", True)):
        return

    server_scene = getattr(getattr(mujoco_scene, "server", None), "scene", None)
    add_batched_meshes_simple = getattr(
        server_scene,
        "add_batched_meshes_simple",
        None,
    )
    if not callable(add_batched_meshes_simple):
        return

    from mjviser.conversions import group_geoms_by_visual_compat, merge_geoms

    model = mujoco_scene.mj_model
    mesh_groups = list(getattr(mujoco_scene, "_mesh_groups", []))
    for mesh_group in mesh_groups:
        body_ids = np.asarray(mesh_group.body_ids, dtype=np.int32).reshape(-1)
        if body_ids.size == 0:
            continue
        body_id = int(body_ids[0])
        geom_ids = _geom_ids_for_body_group(
            model,
            body_ids=np.asarray([body_id], dtype=np.int32),
            group_id=int(mesh_group.group_id),
        )
        if geom_ids.size == 0:
            continue
        subgroups = group_geoms_by_visual_compat(model, geom_ids.tolist())
        subgroup_index = _subgroup_index_from_handle_name(mesh_group.handle)
        if subgroup_index >= len(subgroups):
            continue

        subgroup_geom_ids = subgroups[subgroup_index]
        semantic_group = classify_viser_geom_semantic_group(
            model,
            subgroup_geom_ids,
            body_id=body_id,
            env_raw=env_raw,
        )
        color, opacity = _semantic_mesh_rgb_and_opacity(semantic_group, cfg)
        old_handle = mesh_group.handle
        batched_wxyzs = np.asarray(old_handle.batched_wxyzs, dtype=np.float32)
        batched_positions = np.asarray(old_handle.batched_positions, dtype=np.float32)
        batch_count = int(batched_positions.shape[0])
        mesh = merge_geoms(model, subgroup_geom_ids)

        remove = getattr(old_handle, "remove", None)
        if callable(remove):
            remove()
        mesh_group.handle = add_batched_meshes_simple(
            old_handle.name,
            np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.uint32),
            batched_wxyzs=batched_wxyzs,
            batched_positions=batched_positions,
            batched_scales=getattr(old_handle, "batched_scales", None),
            batched_colors=np.tile(
                np.asarray(color, dtype=np.uint8),
                (batch_count, 1),
            ),
            batched_opacities=_semantic_mesh_batched_opacities(
                batch_count=batch_count,
                opacity=opacity,
            ),
            lod=getattr(old_handle, "lod", "auto"),
            material=str(_cfg_get(cfg, "viser_mesh_material", "standard")),
            flat_shading=bool(_cfg_get(cfg, "viser_mesh_flat_shading", False)),
            cast_shadow=bool(getattr(old_handle, "cast_shadow", True)),
            receive_shadow=bool(getattr(old_handle, "receive_shadow", True)),
            visible=bool(getattr(old_handle, "visible", True)),
        )


def resolve_viser_render_model(env_raw: Any) -> tuple[Any, Any]:
    render_model = getattr(env_raw, "render_model", None)
    render_data = getattr(env_raw, "render_data", None)
    if render_model is not None and render_data is not None:
        return render_model, render_data

    build_render_model = getattr(env_raw, "_build_render_model", None)
    if callable(build_render_model):
        render_result = build_render_model()
        if render_result is not None:
            render_model, render_data = render_result
        else:
            render_model = env_raw.model
            render_data = env_raw.data
    else:
        render_model = env_raw.model
        render_data = env_raw.data

    env_raw.render_model = render_model
    env_raw.render_data = render_data
    return render_model, render_data


def _iter_fast_muscle_tendon_providers(env_raw: Any) -> list[Any]:
    candidates = [
        env_raw,
        getattr(env_raw, "fast_muscle", None),
        getattr(env_raw, "_control_pipeline", None),
        getattr(getattr(env_raw, "_control_pipeline", None), "controller", None),
    ]
    providers: list[Any] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        providers.append(candidate)
    return providers


def _attach_fast_muscle_tendon_provider(provider: Any, render_model: Any) -> None:
    if getattr(provider, "_activation_vis_enabled", True) is False:
        return
    try:
        provider.render_model = render_model
    except Exception as exc:  # noqa: BLE001 - activation visualization is optional viewer state.
        logger.debug(
            "Could not attach render_model to tendon provider: %s",
            exc,
            exc_info=True,
        )

    setup_activation_visualization = getattr(
        provider,
        "_setup_activation_visualization",
        None,
    )
    if callable(setup_activation_visualization):
        setup_activation_visualization()
        return

    ensure_tendon_visualizer = getattr(provider, "_ensure_tendon_visualizer", None)
    if callable(ensure_tendon_visualizer):
        visualizer = ensure_tendon_visualizer()
        attach = getattr(visualizer, "attach", None)
        if callable(attach):
            attach(render_model)


def _looks_like_fast_muscle_line_source(source: Any) -> bool:
    return all(
        hasattr(source, attr)
        for attr in (
            "endpoint_body",
            "endpoint_local",
            "endpoint_muscle",
            "resolve_activation_tensor",
            "segment_count",
        )
    )


def resolve_fast_muscle_tendon_sources(env_raw: Any, render_model: Any) -> list[Any]:
    for provider in _iter_fast_muscle_tendon_providers(env_raw):
        _attach_fast_muscle_tendon_provider(provider, render_model)

    sources: list[Any] = []
    seen: set[int] = set()
    for provider in _iter_fast_muscle_tendon_providers(env_raw):
        for getter_name in ("dynamic_gl_render_sources",):
            source_getter = getattr(provider, getter_name, None)
            if not callable(source_getter):
                continue
            for source in source_getter():
                source_id = id(source)
                if source_id in seen:
                    continue
                if not _looks_like_fast_muscle_line_source(source):
                    continue
                seen.add(source_id)
                sources.append(source)
    return sources


@dataclass(frozen=True)
class ViserLineSegments:
    points: np.ndarray
    colors: np.ndarray
    line_width: float


def _rotate_wxyz_batch(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    qvec = quat[:, 1:4]
    t = 2.0 * np.cross(qvec, vec)
    return vec + quat[:, 0:1] * t + np.cross(qvec, t)


def _interpolate_rgba(
    low_rgba: np.ndarray,
    high_rgba: np.ndarray,
    values: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    t = np.asarray(values, dtype=np.float64).reshape(-1)
    if str(mode).strip().lower() == "hsv":
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

    out = low_rgba[None, :] * (1.0 - t[:, None]) + high_rgba[None, :] * t[:, None]
    return out.astype(np.float32, copy=False)


def _resolve_fast_muscle_source_points(source: Any, render_data: Any) -> np.ndarray:
    resolver = getattr(source, "endpoint_world_resolver", None)
    if resolver is not None:
        return resolver(render_data)[0].detach().cpu().numpy()
    vertex_count = int(getattr(source, "vertex_count", 0))
    if vertex_count <= 0:
        vertex_count = int(np.asarray(source.endpoint_body).size)

    if bool(getattr(source, "uses_lbs_endpoints", False)):
        attachment_start = np.asarray(source.endpoint_attachment_start, dtype=np.int32)
        attachment_body = np.asarray(source.endpoint_attachment_body, dtype=np.int32)
        local_weight = np.asarray(
            source.endpoint_attachment_local_weight,
            dtype=np.float64,
        )
        body_pos = np.asarray(render_data.xpos, dtype=np.float64)[attachment_body]
        body_quat = np.asarray(render_data.xquat, dtype=np.float64)[attachment_body]
        attachment_world = body_pos + _rotate_wxyz_batch(
            body_quat,
            local_weight[:, :3],
        )
        attachment_world *= local_weight[:, 3:4]
        points = np.zeros((vertex_count, 3), dtype=np.float64)
        counts = attachment_start[1:] - attachment_start[:-1]
        endpoint_idx = np.repeat(np.arange(vertex_count, dtype=np.int32), counts)
        np.add.at(points, endpoint_idx, attachment_world)
        return points.astype(np.float32, copy=False)

    body_ids = np.asarray(source.endpoint_body, dtype=np.int32).reshape(-1)
    local = np.asarray(source.endpoint_local, dtype=np.float64).reshape(-1, 3)
    body_pos = np.asarray(render_data.xpos, dtype=np.float64)[body_ids]
    body_quat = np.asarray(render_data.xquat, dtype=np.float64)[body_ids]
    points = body_pos + _rotate_wxyz_batch(body_quat, local)
    return points.astype(np.float32, copy=False)


def _resolve_fast_muscle_source_rgba(source: Any) -> np.ndarray:
    activation = source.resolve_activation_tensor()
    if torch.is_tensor(activation):
        activation_np = activation.detach().cpu().numpy()
    else:
        activation_np = np.asarray(activation, dtype=np.float32)
    activation_np = np.asarray(activation_np, dtype=np.float32).reshape(-1)

    muscle_count = int(getattr(source, "muscle_count", activation_np.size))
    if activation_np.size != muscle_count:
        activation_np = np.zeros((muscle_count,), dtype=np.float32)

    lower = float(getattr(source, "activation_lower_bound", 0.0))
    upper = float(getattr(source, "activation_upper_bound", 1.0))
    clipped = np.clip(activation_np, lower, upper)
    rgba = _interpolate_rgba(
        np.asarray(source.color_low_rgba, dtype=np.float32),
        np.asarray(source.color_high_rgba, dtype=np.float32),
        clipped,
        mode=str(getattr(source, "color_interpolation", "rgb")),
    )
    under_mag = np.maximum(-activation_np, 0.0)
    over_mag = np.maximum(activation_np - 1.0, 0.0)
    under_mask = (under_mag > 0.0) & (under_mag >= over_mag)
    over_mask = over_mag > under_mag
    rgba[under_mask, :] = np.asarray(source.color_under_rgba, dtype=np.float32)
    rgba[over_mask, :] = np.asarray(source.color_over_rgba, dtype=np.float32)
    return rgba


def _effective_fast_muscle_line_width(source_width: float) -> float:
    return max(1.0, 0.5 * float(source_width))


def build_viser_line_segments_from_fast_muscle_source(
    source: Any,
    render_data: Any,
) -> ViserLineSegments:
    segment_count = int(getattr(source, "segment_count", 0))
    points = _resolve_fast_muscle_source_points(source, render_data).reshape(
        segment_count,
        2,
        3,
    )

    rgba = _resolve_fast_muscle_source_rgba(source)
    endpoint_muscle = np.asarray(source.endpoint_muscle, dtype=np.int32).reshape(-1)
    endpoint_muscle = np.clip(endpoint_muscle, 0, max(0, int(rgba.shape[0]) - 1))
    endpoint_rgb = np.clip(rgba[endpoint_muscle, :3] * 255.0, 0.0, 255.0)
    colors = endpoint_rgb.astype(np.uint8).reshape(segment_count, 2, 3)
    return ViserLineSegments(
        points=points,
        colors=colors,
        line_width=_effective_fast_muscle_line_width(
            float(getattr(source, "line_width", 1.0))
        ),
    )


def _ensure_frame_batch(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2:
        return array
    raise ValueError(
        f"{name} must have shape (dim,) or (frames, dim), got {array.shape}"
    )


def stage_viser_render_state_batch(env_raw: Any) -> tuple[np.ndarray, np.ndarray]:
    state_getter = getattr(env_raw, "stage_playback_render_state_batch_cpu", None)
    if callable(state_getter):
        qpos, qvel = state_getter()
        return (
            _ensure_frame_batch(qpos, name="qpos"),
            _ensure_frame_batch(qvel, name="qvel"),
        )

    data = getattr(env_raw, "mjw_data", None)
    qpos = _warp_array_to_numpy(getattr(data, "qpos", None))
    qvel = _warp_array_to_numpy(getattr(data, "qvel", None))
    if qpos is None or qvel is None:
        raise RuntimeError("Unable to stage render state for viser playback.")
    qpos_batch = _ensure_frame_batch(qpos, name="qpos")
    qvel_batch = _ensure_frame_batch(qvel, name="qvel")
    return qpos_batch[:1], qvel_batch[:1]


class ViserSemanticVisibilityController:
    def __init__(
        self,
        *,
        scene: Any,
        defaults: dict[str, bool],
        env_raw: Any | None = None,
    ) -> None:
        self._scene = scene
        self._env_raw = env_raw
        self._semantic_group_cache: dict[tuple[int, tuple[int, ...]], str] = {}
        self._visible = {
            key: bool(defaults.get(key, True)) for key in _SEMANTIC_GROUP_LABELS
        }
        self._original_sync_visibilities = getattr(scene, "_sync_visibilities", None)
        self._install_sync_wrapper()
        self._enable_mujoco_geom_groups_by_default()
        self.refresh()

    def _install_sync_wrapper(self) -> None:
        if not callable(self._original_sync_visibilities):
            return

        def _sync_with_semantic_visibility() -> None:
            self._original_sync_visibilities()
            self.apply()

        self._scene._sync_visibilities = _sync_with_semantic_visibility

    def _enable_mujoco_geom_groups_by_default(self) -> None:
        geom_groups_visible = getattr(self._scene, "geom_groups_visible", None)
        if isinstance(geom_groups_visible, list):
            for group_idx in range(min(6, len(geom_groups_visible))):
                geom_groups_visible[group_idx] = True

    @property
    def visible(self) -> dict[str, bool]:
        return dict(self._visible)

    def set_visible(self, group: str, visible: bool) -> None:
        if group not in self._visible:
            raise ValueError(f"Unknown viser semantic group: {group}")
        self._visible[group] = bool(visible)
        self.refresh()

    def refresh(self) -> None:
        if callable(self._original_sync_visibilities):
            self._original_sync_visibilities()
        self.apply()

    def apply(self) -> None:
        for mesh_group in getattr(self._scene, "_mesh_groups", []):
            group = self._semantic_group_for_mesh_group(mesh_group)
            if not self._visible.get(group, True):
                mesh_group.handle.visible = False

        fixed_handles = getattr(self._scene, "_fixed_geom_handles", {})
        for (body_id, group_id, _), handle in fixed_handles.items():
            geom_ids = _geom_ids_for_body_group(
                self._scene.mj_model,
                body_ids=np.asarray([body_id], dtype=np.int32),
                group_id=int(group_id),
            )
            group = self._classify_geom_group(geom_ids, body_id=int(body_id))
            if not self._visible.get(group, True):
                handle.visible = False

    def _classify_geom_group(
        self,
        geom_ids: list[int] | np.ndarray,
        *,
        body_id: int,
    ) -> str:
        geom_ids_np = np.asarray(geom_ids, dtype=np.int32).reshape(-1)
        cache_key = (
            int(body_id),
            tuple(int(geom_id) for geom_id in geom_ids_np),
        )
        semantic_group = self._semantic_group_cache.get(cache_key)
        if semantic_group is None:
            semantic_group = classify_viser_geom_semantic_group(
                self._scene.mj_model,
                geom_ids_np,
                body_id=int(body_id),
                env_raw=self._env_raw,
            )
            self._semantic_group_cache[cache_key] = semantic_group
        return semantic_group

    def _semantic_group_for_mesh_group(self, mesh_group: Any) -> str:
        body_ids = np.asarray(mesh_group.body_ids, dtype=np.int32).reshape(-1)
        geom_ids = _geom_ids_for_body_group(
            self._scene.mj_model,
            body_ids=body_ids,
            group_id=int(mesh_group.group_id),
        )
        if geom_ids.size == 0 and getattr(mesh_group, "mocap_ids", None) is not None:
            return "ghost"
        body_id = int(body_ids[0]) if body_ids.size else 0
        return self._classify_geom_group(geom_ids, body_id=body_id)

    def create_gui(self, gui: Any) -> None:
        with gui.add_folder("Semantic"):
            for group, label in _SEMANTIC_GROUP_LABELS.items():
                checkbox = gui.add_checkbox(
                    label,
                    initial_value=self._visible[group],
                    hint=f"Show/hide {label.lower()} geometry.",
                )

                @checkbox.on_update
                def _(event: Any, semantic_group: str = group) -> None:
                    self.set_visible(semantic_group, bool(event.target.value))


class ViserGhostOffsetController:
    def __init__(
        self,
        *,
        env_raw: Any,
        update_callback: Any | None = None,
    ) -> None:
        self._env_raw = env_raw
        self._update_callback = update_callback
        self._offset = self._resolve_offset()

    @property
    def available(self) -> bool:
        return self._offset is not None

    @property
    def offset(self) -> np.ndarray | None:
        if self._offset is None:
            return None
        return np.array(self._offset, copy=True)

    def _resolve_offset(self) -> np.ndarray | None:
        raw_offset = getattr(self._env_raw, "_ghost_offset_np", None)
        if raw_offset is None:
            cfg = getattr(self._env_raw, "cfg", None)
            raw_offset = _cfg_get(cfg, "ghost_offset", None)
        if raw_offset is None:
            return None
        offset = np.asarray(raw_offset, dtype=np.float64).reshape(-1)
        if offset.size != 3:
            return None
        self._env_raw._ghost_offset_np = np.array(offset, copy=True)
        return np.array(offset, copy=True)

    def set_axis(self, axis: int, value: float) -> None:
        if self._offset is None:
            return
        self._offset[int(axis)] = float(value)
        self._env_raw._ghost_offset_np = np.array(self._offset, copy=True)
        if callable(self._update_callback):
            self._update_callback()

    def create_gui(self, gui: Any) -> bool:
        if self._offset is None:
            return False
        with gui.add_folder("Ghost offset"):
            for axis, label in enumerate(("X", "Y", "Z")):
                value = float(self._offset[axis])
                slider = gui.add_slider(
                    label,
                    min=min(-5.0, value - 1.0),
                    max=max(5.0, value + 1.0),
                    step=0.01,
                    initial_value=value,
                )

                @slider.on_update
                def _(event: Any, axis_idx: int = axis) -> None:
                    self.set_axis(axis_idx, float(event.target.value))

        return True


class ViserFastMuscleTendonController:
    def __init__(
        self,
        *,
        scene: Any,
        env_raw: Any,
        render_model: Any,
        scene_offset_getter: Any | None = None,
    ) -> None:
        self._scene = scene
        self._scene_offset_getter = scene_offset_getter
        self._sources = resolve_fast_muscle_tendon_sources(env_raw, render_model)
        self._visible = True
        self._handles: dict[int, Any] = {}
        self._line_width = max(
            (
                _effective_fast_muscle_line_width(
                    float(getattr(source, "line_width", 1.0))
                )
                for source in self._sources
            ),
            default=1.0,
        )

    @property
    def available(self) -> bool:
        return bool(self._sources)

    @property
    def visible(self) -> bool:
        return bool(self._visible)

    def set_visible(self, visible: bool) -> None:
        self._visible = bool(visible)
        for handle in self._handles.values():
            handle.visible = self._visible

    def set_line_width(self, line_width: float) -> None:
        self._line_width = max(0.1, float(line_width))
        for handle in self._handles.values():
            handle.line_width = self._line_width

    def _scene_offset(self) -> np.ndarray:
        if not callable(self._scene_offset_getter):
            return np.zeros((3,), dtype=np.float32)
        offset = np.asarray(self._scene_offset_getter(), dtype=np.float32).reshape(-1)
        if offset.size != 3:
            return np.zeros((3,), dtype=np.float32)
        return offset

    def create_gui(self, gui: Any) -> bool:
        if not self.available:
            return False

        with gui.add_folder("Fast muscle tendons"):
            checkbox = gui.add_checkbox(
                "Visible",
                initial_value=self._visible,
                hint="Show/hide fast-muscle tendon line visualization.",
            )

            @checkbox.on_update
            def _(event: Any) -> None:
                self.set_visible(bool(event.target.value))

            slider = gui.add_slider(
                "Line width",
                min=0.5,
                max=12.0,
                step=0.1,
                initial_value=self._line_width,
                hint="Screen-space width for fast-muscle tendon lines.",
            )

            @slider.on_update
            def _(event: Any) -> None:
                self.set_line_width(float(event.target.value))

        return True

    def update(self, render_data: Any) -> None:
        for source_idx, source in enumerate(self._sources):
            try:
                segments = build_viser_line_segments_from_fast_muscle_source(
                    source,
                    render_data,
                )
            except Exception:
                logger.exception(
                    "Failed to update fast-muscle tendon line source %d.",
                    source_idx,
                )
                continue

            points = segments.points + self._scene_offset()[None, None, :]
            handle = self._handles.get(source_idx)
            if handle is None:
                handle = self._scene.add_line_segments(
                    f"/fast_muscle_tendons/source_{source_idx}",
                    points=points,
                    colors=segments.colors,
                    line_width=self._line_width,
                    visible=self._visible,
                )
                self._handles[source_idx] = handle
                continue

            handle.points = points
            handle.colors = segments.colors
            handle.line_width = self._line_width
            handle.visible = self._visible


@dataclass
class _RewardPlotHandle:
    labels: tuple[str, ...]
    handle: Any


class ViserRewardPlotter:
    def __init__(self, *, gui: Any, history_seconds: float) -> None:
        self._gui = gui
        self._history_seconds = float(history_seconds)
        self._plots: dict[int, _RewardPlotHandle] = {}

    def clear(self) -> None:
        for state in self._plots.values():
            remove = getattr(state.handle, "remove", None)
            if callable(remove):
                remove()
        self._plots.clear()

    def update(self, overlay_state: dict[str, Any]) -> None:
        plots = overlay_state.get("plots", [])
        if not isinstance(plots, list):
            self.clear()
            return

        live_indices: set[int] = set()
        for plot_idx, plot in enumerate(plots):
            if not isinstance(plot, dict):
                continue
            series_map = plot.get("series", {})
            if not isinstance(series_map, dict) or not series_map:
                continue

            labels = tuple(str(label) for label in series_map)
            data = build_uplot_data_from_overlay_plot(
                plot,
                history_seconds=self._history_seconds,
            )
            state = self._plots.get(plot_idx)
            if state is None or state.labels != labels:
                if state is not None:
                    remove = getattr(state.handle, "remove", None)
                    if callable(remove):
                        remove()
                handle = self._gui.add_uplot(
                    data=data,
                    series=_build_uplot_series(plot),
                    title=str(plot.get("title", f"plot_{plot_idx}")),
                    legend={"show": True},
                    scales={"x": {"time": False, "auto": True}},
                    aspect=2.5,
                )
                self._plots[plot_idx] = _RewardPlotHandle(
                    labels=labels,
                    handle=handle,
                )
            else:
                state.handle.data = data
            live_indices.add(plot_idx)

        for plot_idx in list(self._plots):
            if plot_idx not in live_indices:
                state = self._plots.pop(plot_idx)
                remove = getattr(state.handle, "remove", None)
                if callable(remove):
                    remove()


def _warp_array_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    try:
        import warp as wp

        return wp.to_torch(value).detach().cpu().numpy()
    except Exception:
        logger.exception("Failed to stage MuJoCo-Warp array for viser playback.")
        return None


def _normalize_body_xmat(body_xmat: np.ndarray) -> np.ndarray:
    if body_xmat.ndim == 4:
        return body_xmat
    if body_xmat.ndim == 3 and body_xmat.shape[-1] == 9:
        return body_xmat.reshape(body_xmat.shape[0], body_xmat.shape[1], 3, 3)
    raise ValueError(
        f"Unsupported body_xmat shape for viser playback: {body_xmat.shape}"
    )


def _format_speed(multiplier: float) -> str:
    if multiplier == 1.0:
        return "1x"
    inv = 1.0 / multiplier
    inv_rounded = round(inv)
    if abs(inv - inv_rounded) < 1e-9 and inv_rounded > 0:
        return f"1/{inv_rounded}x"
    return f"{multiplier:.3g}x"


class ViserPlaybackSession:
    def __init__(
        self,
        *,
        cfg: Any,
        env_raw: Any,
        step_dt: float,
        overlay_enabled: bool,
        overlay_history_seconds: float,
        server: Any | None = None,
        scene_cls: Any | None = None,
    ) -> None:
        import viser
        from mjviser import ViserMujocoScene

        self._env_raw = env_raw
        self._step_dt = float(step_dt)
        self._paused = False
        self._single_step_requests = 0
        self._reset_requested = False
        self._stop_requested = False
        self._speed_idx = _VISER_SPEEDS.index(1.0)
        self._render_model, self._render_data = resolve_viser_render_model(env_raw)

        port = int(cfg.get("viewer_port", 8080))
        self.server = server if server is not None else viser.ViserServer(port=port)
        num_envs = 1
        self.scene = (scene_cls or ViserMujocoScene)(
            self.server,
            self._render_model,
            num_envs=num_envs,
        )
        configure_viser_scene_theme(self.server.scene, self._render_model, cfg)
        configure_viser_mujoco_scene_mesh_style(
            self.scene,
            cfg,
            env_raw=env_raw,
        )
        self._semantic_visibility = ViserSemanticVisibilityController(
            scene=self.scene,
            defaults=resolve_viser_semantic_group_defaults(env_raw),
            env_raw=env_raw,
        )
        self._fast_muscle_tendons = ViserFastMuscleTendonController(
            scene=self.server.scene,
            env_raw=env_raw,
            render_model=self._render_model,
            scene_offset_getter=lambda: getattr(
                self.scene,
                "_scene_offset",
                np.zeros(3),
            ),
        )
        self._ghost_offset = ViserGhostOffsetController(
            env_raw=env_raw,
            update_callback=lambda: self.update_scene(),
        )
        self._status_html: Any | None = None
        self._reward_plotter: ViserRewardPlotter | None = None
        self._setup_gui(
            overlay_enabled=overlay_enabled,
            overlay_history_seconds=overlay_history_seconds,
        )

    @property
    def speed(self) -> float:
        return float(_VISER_SPEEDS[self._speed_idx])

    @property
    def paused(self) -> bool:
        return bool(self._paused)

    def is_running(self) -> bool:
        return not self._stop_requested

    def should_step(self) -> bool:
        return (not self._paused) or self._single_step_requests > 0

    def mark_step_consumed(self) -> None:
        if self._single_step_requests > 0:
            self._single_step_requests -= 1

    def consume_reset_request(self) -> bool:
        requested = bool(self._reset_requested)
        self._reset_requested = False
        return requested

    def close(self) -> None:
        stop = getattr(self.server, "stop", None)
        if callable(stop):
            stop()

    def _setup_gui(
        self,
        *,
        overlay_enabled: bool,
        overlay_history_seconds: float,
    ) -> None:
        gui = self.server.gui
        tabs = gui.add_tab_group()

        with tabs.add_tab("Playback"):
            self._status_html = gui.add_html("")
            pause_btn = gui.add_button("Pause")

            @pause_btn.on_click
            def _(_) -> None:
                self._paused = not self._paused
                pause_btn.label = "Pause" if not self._paused else "Play"

            step_btn = gui.add_button("Step")

            @step_btn.on_click
            def _(_) -> None:
                self._single_step_requests += 1

            reset_btn = gui.add_button("Reset")

            @reset_btn.on_click
            def _(_) -> None:
                self._reset_requested = True

            stop_btn = gui.add_button("Stop")

            @stop_btn.on_click
            def _(_) -> None:
                self._stop_requested = True

            speed_btns = gui.add_button_group(
                "Speed",
                options=["Slower", "1x", "Faster"],
            )

            @speed_btns.on_click
            def _(event: Any) -> None:
                value = getattr(getattr(event, "target", None), "value", None)
                if value == "Slower":
                    self._speed_idx = max(0, self._speed_idx - 1)
                elif value == "Faster":
                    self._speed_idx = min(len(_VISER_SPEEDS) - 1, self._speed_idx + 1)
                else:
                    self._speed_idx = _VISER_SPEEDS.index(1.0)

        if overlay_enabled:
            with tabs.add_tab("Rewards"):
                self._reward_plotter = ViserRewardPlotter(
                    gui=gui,
                    history_seconds=overlay_history_seconds,
                )

        with tabs.add_tab("Scene"):
            self.scene.create_scene_gui()
        with tabs.add_tab("Visualization"):
            self.scene.create_overlay_gui()
            self._fast_muscle_tendons.create_gui(gui)
        with tabs.add_tab("Groups"):
            self._semantic_visibility.create_gui(gui)
            self._ghost_offset.create_gui(gui)
            self.scene.create_groups_gui()

    def update_scene(self, *, animate_substeps: bool = False) -> int:
        try:
            return self._update_render_scene(animate_substeps=animate_substeps)
        except Exception:
            logger.exception(
                "Failed to update viser render-model scene; falling back to mjw_data."
            )
            return self._update_mjw_scene()

    def _update_mjw_scene(self) -> int:
        data = self._env_raw.mjw_data
        body_xpos = _warp_array_to_numpy(getattr(data, "xpos", None))
        body_xmat = _warp_array_to_numpy(getattr(data, "xmat", None))
        if body_xpos is None or body_xmat is None:
            return 0

        qpos = _warp_array_to_numpy(getattr(data, "qpos", None))
        qvel = _warp_array_to_numpy(getattr(data, "qvel", None))
        mocap_pos = _warp_array_to_numpy(getattr(data, "mocap_pos", None))
        mocap_quat = _warp_array_to_numpy(getattr(data, "mocap_quat", None))
        self.scene.update_from_arrays(
            body_xpos=body_xpos,
            body_xmat=_normalize_body_xmat(body_xmat),
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            qpos=qpos,
            qvel=qvel,
        )
        return 1

    def _update_render_scene(self, *, animate_substeps: bool) -> int:
        import mujoco

        qpos_batch, qvel_batch = stage_viser_render_state_batch(self._env_raw)
        frame_count = max(1, min(int(qpos_batch.shape[0]), int(qvel_batch.shape[0])))
        for frame_idx in range(frame_count):
            qpos = np.asarray(qpos_batch[frame_idx])
            qvel = np.asarray(qvel_batch[frame_idx])
            self._render_data.qpos[:] = qpos[: int(self._render_model.nq)]
            self._render_data.qvel[:] = qvel[: int(self._render_model.nv)]
            update_render_data = getattr(self._env_raw, "_update_render_data", None)
            if callable(update_render_data):
                update_render_data()
            mujoco.mj_forward(self._render_model, self._render_data)
            self.scene.update_from_mjdata(self._render_data)
            self._fast_muscle_tendons.update(self._render_data)
            if animate_substeps and frame_idx + 1 < frame_count:
                time.sleep(
                    (self._step_dt / float(frame_count))
                    / max(float(self.speed), 1.0e-6)
                )
        return frame_count

    def update_overlay(self, overlay_state: dict[str, Any]) -> None:
        if self._reward_plotter is not None:
            self._reward_plotter.update(overlay_state)

    def clear_overlay(self) -> None:
        if self._reward_plotter is not None:
            self._reward_plotter.clear()

    def update_status(
        self,
        *,
        step: int,
        step_reward: float,
        episode_length: int,
        episode_reward: float,
        tracking_metrics: dict[str, float | None] | None = None,
    ) -> None:
        if self._status_html is None:
            return
        tracking_metrics = tracking_metrics or {}
        ape = tracking_metrics.get("ape")
        ee_error = tracking_metrics.get("ee_error")
        ape_text = "-" if ape is None else f"{float(ape):.6f}"
        ee_text = "-" if ee_error is None else f"{float(ee_error):.6f}"
        state = "Paused" if self._paused else "Running"
        self._status_html.content = f"""
        <div style="font-size:0.9em;line-height:1.35;padding:0.25em 0.75em;">
          <strong>Status:</strong> {html.escape(state)}<br/>
          <strong>Step:</strong> {int(step)}<br/>
          <strong>Speed:</strong> {html.escape(_format_speed(self.speed))}
          ({self.speed:.2f}x target)<br/>
          <strong>Control dt:</strong> {self._step_dt:.5f}s<br/>
          <strong>Reward:</strong> {float(step_reward):.6f}<br/>
          <strong>Episode:</strong> {int(episode_length)}
          steps / {float(episode_reward):.6f} reward<br/>
          <strong>APE:</strong> {ape_text}<br/>
          <strong>EE:</strong> {ee_text}
        </div>
        """
