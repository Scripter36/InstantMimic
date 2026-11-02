from __future__ import annotations

import colorsys
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp

from instantmimic.utils.project_paths import resolve_project_path

logger = logging.getLogger(__name__)

_ACTIVATION_POSITIVE_LOW = np.asarray([1.0, 0.38, 0.12], dtype=np.float32)
_ACTIVATION_POSITIVE_MID = np.asarray([1.0, 0.68, 0.05], dtype=np.float32)
_ACTIVATION_POSITIVE_HIGH = np.asarray([1.0, 0.04, 0.01], dtype=np.float32)
_ACTIVATION_NEGATIVE_LOW = np.asarray([0.12, 0.45, 1.0], dtype=np.float32)
_ACTIVATION_NEGATIVE_MID = np.asarray([0.2, 0.85, 1.0], dtype=np.float32)
_ACTIVATION_NEGATIVE_HIGH = np.asarray([0.02, 0.08, 1.0], dtype=np.float32)
_ACTIVATION_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class _LBSActivationMapping:
    activation_channel_count: int
    mapped_mesh_count: int
    mesh_channel_starts: np.ndarray
    mesh_channel_indices: np.ndarray


@dataclass(frozen=True)
class _LBSActivationColorSettings:
    color_low_rgba: np.ndarray
    color_high_rgba: np.ndarray
    color_interpolation: str


def _activation_scheme_color_cpu(strength: float, *, sign: float = 1.0) -> np.ndarray:
    """Mirror the zip export activation-surface RGB ramp without fade/top-K state."""
    value = float(np.clip(strength, 0.0, 1.0))
    if sign < 0.0:
        low = _ACTIVATION_NEGATIVE_LOW
        mid = _ACTIVATION_NEGATIVE_MID
        high = _ACTIVATION_NEGATIVE_HIGH
    else:
        low = _ACTIVATION_POSITIVE_LOW
        mid = _ACTIVATION_POSITIVE_MID
        high = _ACTIVATION_POSITIVE_HIGH
    if value <= 0.5:
        raw = low * (1.0 - value * 2.0) + mid * (value * 2.0)
    else:
        t = (value - 0.5) * 2.0
        raw = mid * (1.0 - t) + high * t
    saturation = 0.45 + 0.55 * value
    luma = float(np.dot(raw, _ACTIVATION_LUMA))
    return (luma + (raw - luma) * saturation).astype(np.float32)


def _resolve_lbs_mesh_mode(mode: str | None) -> str:
    value = (
        "off" if mode is None or str(mode).strip() == "" else str(mode).strip().lower()
    )
    if value in {"false", "0", "none", "null"}:
        value = "off"
    if value in {"off", "skin", "muscle"}:
        return value
    raise ValueError("video_warp_lbs_mesh_mode must be one of: off, skin, muscle.")


def _resolve_lbs_mesh_shading_mode(mode: str | None) -> str:
    value = (
        "smooth"
        if mode is None or str(mode).strip() == ""
        else str(mode).strip().lower()
    )
    if value in {"flat", "smooth"}:
        return value
    raise ValueError("video_warp_lbs_mesh_shading_mode must be one of: flat, smooth.")


def _resolve_lbs_mesh_bundle_path(path: str | Path | None) -> Path:
    if path is None or not str(path).strip():
        raise ValueError(
            "LBS mesh rendering requires an explicit video_warp_lbs_mesh_bundle path."
        )
    return resolve_project_path(Path(path).expanduser())


def _resolve_lbs_activation_mapping_path(path: str | Path | None) -> Path | None:
    if path is None or not str(path).strip():
        return None
    return resolve_project_path(Path(path).expanduser())


def _coerce_lbs_rgba(
    value: Any,
    *,
    fallback: tuple[float, float, float, float],
    fallback_alpha: float,
    name: str,
) -> np.ndarray:
    if value is None:
        values = np.asarray(fallback, dtype=np.float32)
    else:
        values = np.asarray(value, dtype=np.float32).reshape(-1)
    if values.size == 3:
        values = np.concatenate(
            [values, np.asarray([float(fallback_alpha)], dtype=np.float32)]
        )
    elif values.size != 4:
        raise ValueError(f"{name} must contain exactly 3 or 4 values.")
    return np.clip(values.astype(np.float32), 0.0, 1.0)


def _resolve_lbs_activation_color_settings(env_raw: Any) -> _LBSActivationColorSettings:
    cfg = getattr(env_raw, "cfg", None)
    from instantmimic.envs.controls.fast_muscle.config_models import (
        ActivationVisualizationSettings,
    )

    visualization = ActivationVisualizationSettings.from_env_cfg(cfg)
    fallback_alpha = visualization.alpha
    low = _coerce_lbs_rgba(
        visualization.color_low,
        fallback=(0.20, 0.25, 0.90, 0.30),
        fallback_alpha=fallback_alpha,
        name="activation_visualization.color_low",
    )
    high = _coerce_lbs_rgba(
        visualization.color_high,
        fallback=(1.00, 0.15, 0.10, 1.0),
        fallback_alpha=fallback_alpha,
        name="activation_visualization.color_high",
    )
    return _LBSActivationColorSettings(
        color_low_rgba=low,
        color_high_rgba=high,
        color_interpolation=visualization.interpolation,
    )


def _read_lbs_activation_mapping_payload(
    mapping_path: Path,
) -> dict[str, np.ndarray]:
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"LBS activation mapping does not exist: {mapping_path.as_posix()}"
        )
    if mapping_path.suffix.lower() == ".json":
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        return {
            "muscle_names": np.asarray(
                payload.get("muscleNames", payload.get("muscle_names", [])),
                dtype=object,
            ),
            "activation_channel_names": np.asarray(
                payload.get(
                    "activationChannelNames",
                    payload.get("activation_channel_names", []),
                ),
                dtype=object,
            ),
            "channel_to_muscle_index": np.asarray(
                payload.get(
                    "channelToMuscleIndex",
                    payload.get("channel_to_muscle_index", []),
                ),
                dtype=np.int32,
            ),
        }
    with np.load(mapping_path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _load_lbs_activation_mapping(
    mapping_path: str | Path,
    *,
    selected_mesh_names: list[str],
) -> _LBSActivationMapping:
    payload = _read_lbs_activation_mapping_payload(Path(mapping_path))
    if "muscle_names" not in payload or "channel_to_muscle_index" not in payload:
        raise ValueError(
            "LBS activation mapping requires muscle_names and channel_to_muscle_index."
        )
    muscle_names = [str(name) for name in payload["muscle_names"].tolist()]
    channel_to_muscle_index = np.asarray(
        payload["channel_to_muscle_index"], dtype=np.int32
    ).reshape(-1)
    selected_index_by_name = {
        str(name): int(index) for index, name in enumerate(selected_mesh_names)
    }
    channels_by_mesh: list[list[int]] = [[] for _ in selected_mesh_names]
    for channel_index, muscle_index in enumerate(channel_to_muscle_index.tolist()):
        if int(muscle_index) < 0:
            continue
        if int(muscle_index) >= len(muscle_names):
            raise ValueError(
                "channel_to_muscle_index contains an index outside muscle_names."
            )
        selected_mesh_index = selected_index_by_name.get(
            muscle_names[int(muscle_index)]
        )
        if selected_mesh_index is not None:
            channels_by_mesh[selected_mesh_index].append(int(channel_index))

    starts = [0]
    flat_channels: list[int] = []
    for channels in channels_by_mesh:
        flat_channels.extend(channels)
        starts.append(len(flat_channels))
    return _LBSActivationMapping(
        activation_channel_count=int(len(channel_to_muscle_index)),
        mapped_mesh_count=sum(1 for channels in channels_by_mesh if channels),
        mesh_channel_starts=np.asarray(starts, dtype=np.int32),
        mesh_channel_indices=np.asarray(flat_channels, dtype=np.int32),
    )


@wp.func
def _rotate_wxyz(q: wp.quat, v: wp.vec3) -> wp.vec3:
    qv = wp.vec3(q[1], q[2], q[3])
    t = 2.0 * wp.cross(qv, v)
    return v + q[0] * t + wp.cross(qv, t)


@wp.func
def _muscle_endpoint_world(
    worldid: int,
    endpoint_idx: int,
    use_lbs: bool,
    endpoint_body: wp.array[int],
    endpoint_local: wp.array[wp.vec3],
    endpoint_world: wp.array2d[wp.vec3],
    use_world: bool,
    endpoint_attachment_start: wp.array[int],
    endpoint_attachment_body: wp.array[int],
    endpoint_attachment_local_weight: wp.array[wp.vec4],
    body_xpos: wp.array2d[wp.vec3],
    body_xquat: wp.array2d[wp.quat],
) -> wp.vec3:
    if use_world:
        return endpoint_world[worldid, endpoint_idx]
    if use_lbs:
        begin = endpoint_attachment_start[endpoint_idx]
        end = endpoint_attachment_start[endpoint_idx + 1]
        out = wp.vec3(0.0, 0.0, 0.0)
        for attachment_idx in range(begin, end):
            body_id = endpoint_attachment_body[attachment_idx]
            local_weight = endpoint_attachment_local_weight[attachment_idx]
            local = wp.vec3(local_weight[0], local_weight[1], local_weight[2])
            world = body_xpos[worldid, body_id] + _rotate_wxyz(
                body_xquat[worldid, body_id],
                local,
            )
            out = out + world * local_weight[3]
        return out

    body_id = endpoint_body[endpoint_idx]
    return body_xpos[worldid, body_id] + _rotate_wxyz(
        body_xquat[worldid, body_id],
        endpoint_local[endpoint_idx],
    )


@wp.func
def _lbs_mesh_vertex_world(
    worldid: int,
    vertex_idx: int,
    vertex_body_ids: wp.array2d[int],
    vertex_local_weight: wp.array2d[wp.vec4],
    body_xpos: wp.array2d[wp.vec3],
    body_xquat: wp.array2d[wp.quat],
) -> wp.vec3:
    out = wp.vec3(0.0, 0.0, 0.0)
    for slot in range(4):
        local_weight = vertex_local_weight[vertex_idx, slot]
        weight = local_weight[3]
        if weight <= 0.0:
            continue
        body_id = vertex_body_ids[vertex_idx, slot]
        local = wp.vec3(local_weight[0], local_weight[1], local_weight[2])
        world = body_xpos[worldid, body_id] + _rotate_wxyz(
            body_xquat[worldid, body_id],
            local,
        )
        out = out + world * weight
    return out


@wp.func
def _lbs_mesh_vertex_normal_world(
    worldid: int,
    vertex_idx: int,
    vertex_body_ids: wp.array2d[int],
    vertex_local_weight: wp.array2d[wp.vec4],
    vertex_local_normal: wp.array2d[wp.vec3],
    body_xquat: wp.array2d[wp.quat],
) -> wp.vec3:
    out = wp.vec3(0.0, 0.0, 0.0)
    for slot in range(4):
        local_weight = vertex_local_weight[vertex_idx, slot]
        weight = local_weight[3]
        if weight <= 0.0:
            continue
        body_id = vertex_body_ids[vertex_idx, slot]
        local_normal = vertex_local_normal[vertex_idx, slot]
        out = out + _rotate_wxyz(body_xquat[worldid, body_id], local_normal) * weight
    normal_len = wp.length(out)
    if normal_len <= 1.0e-6:
        return wp.vec3(0.0, 0.0, 1.0)
    return out / normal_len


@wp.func
def _project_muscle_point(
    point: wp.vec3,
    cam_pos: wp.vec3,
    cam_xmat: wp.mat33,
    width: int,
    height: int,
    fovy: float,
) -> wp.vec3:
    delta = point - cam_pos
    local = wp.transpose(cam_xmat) @ delta
    depth = -local[2]
    if depth <= 1.0e-6:
        return wp.vec3(-1.0e9, -1.0e9, -1.0)
    tan_y = wp.tan(0.5 * fovy * wp.static(wp.pi / 180.0))
    tan_x = tan_y * (float(width) / float(height))
    x_ndc = local[0] / (depth * tan_x)
    y_ndc = local[1] / (depth * tan_y)
    x = (x_ndc * 0.5 + 0.5) * float(width) - 0.5
    y = (0.5 - y_ndc * 0.5) * float(height) - 0.5
    return wp.vec3(x, y, depth)


@wp.func
def _pack_rgb_bytes(r: int, g: int, b: int) -> wp.uint32:
    return wp.uint32(
        (255 << 24)
        | (wp.min(255, wp.max(0, r)) << 16)
        | (wp.min(255, wp.max(0, g)) << 8)
        | wp.min(255, wp.max(0, b))
    )


@wp.func
def _hsv_to_rgb(h: float, s: float, v: float) -> wp.vec3:
    h6 = h * 6.0
    i = int(wp.floor(h6))
    f = h6 - float(i)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    imod = i - (i / 6) * 6
    if imod == 0:
        return wp.vec3(v, t, p)
    if imod == 1:
        return wp.vec3(q, v, p)
    if imod == 2:
        return wp.vec3(p, v, t)
    if imod == 3:
        return wp.vec3(p, q, v)
    if imod == 4:
        return wp.vec3(t, p, v)
    return wp.vec3(v, p, q)


@wp.kernel(enable_backward=False)
def _draw_muscle_lines_kernel(
    body_xpos: wp.array2d[wp.vec3],
    body_xquat: wp.array2d[wp.quat],
    cam_xpos: wp.array2d[wp.vec3],
    cam_xmat: wp.array2d[wp.mat33],
    rgb_data: wp.array2d[wp.uint32],
    rgb_adr: wp.array[int],
    depth_data: wp.array2d[float],
    depth_adr: wp.array[int],
    camera_id: int,
    world_id: int,
    width: int,
    height: int,
    fovy: float,
    endpoint_body: wp.array[int],
    endpoint_local: wp.array[wp.vec3],
    endpoint_world: wp.array2d[wp.vec3],
    use_world: bool,
    endpoint_muscle: wp.array[int],
    endpoint_attachment_start: wp.array[int],
    endpoint_attachment_body: wp.array[int],
    endpoint_attachment_local_weight: wp.array[wp.vec4],
    use_lbs: bool,
    activation: wp.array[float],
    color_low: wp.vec4,
    color_high: wp.vec4,
    color_under: wp.vec4,
    color_over: wp.vec4,
    color_low_hsv: wp.vec3,
    color_high_hsv: wp.vec3,
    hue_delta: float,
    color_mode: int,
    activation_lower: float,
    activation_upper: float,
    line_width: float,
    depth_bias: float,
) -> None:
    segment_idx, sample_idx = wp.tid()
    v0 = segment_idx * 2
    v1 = v0 + 1
    p0_world = _muscle_endpoint_world(
        world_id,
        v0,
        use_lbs,
        endpoint_body,
        endpoint_local,
        endpoint_world,
        use_world,
        endpoint_attachment_start,
        endpoint_attachment_body,
        endpoint_attachment_local_weight,
        body_xpos,
        body_xquat,
    )
    p1_world = _muscle_endpoint_world(
        world_id,
        v1,
        use_lbs,
        endpoint_body,
        endpoint_local,
        endpoint_world,
        use_world,
        endpoint_attachment_start,
        endpoint_attachment_body,
        endpoint_attachment_local_weight,
        body_xpos,
        body_xquat,
    )
    p0 = _project_muscle_point(
        p0_world,
        cam_xpos[world_id, camera_id],
        cam_xmat[world_id, camera_id],
        width,
        height,
        fovy,
    )
    p1 = _project_muscle_point(
        p1_world,
        cam_xpos[world_id, camera_id],
        cam_xmat[world_id, camera_id],
        width,
        height,
        fovy,
    )
    if p0[2] <= 0.0 or p1[2] <= 0.0:
        return

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    steps = int(wp.ceil(wp.max(wp.abs(dx), wp.abs(dy))))
    if steps < 1:
        steps = 1
    if sample_idx > steps:
        return

    t = float(sample_idx) / float(steps)
    px = int(wp.round(p0[0] + dx * t))
    py = int(wp.round(p0[1] + dy * t))
    line_depth = p0[2] * (1.0 - t) + p1[2] * t
    radius = int(wp.ceil(wp.max(1.0, 0.5 * line_width) * 0.5))

    muscle_idx = endpoint_muscle[v0]
    raw = activation[muscle_idx]
    a = wp.clamp(raw, activation_lower, activation_upper)
    color = color_low * (1.0 - a) + color_high * a
    if color_mode == 1:
        hue = color_low_hsv[0] + hue_delta * a
        hue = hue - wp.floor(hue)
        saturation = color_low_hsv[1] * (1.0 - a) + color_high_hsv[1] * a
        value = color_low_hsv[2] * (1.0 - a) + color_high_hsv[2] * a
        rgb = _hsv_to_rgb(hue, saturation, value)
        alpha_hsv = color_low[3] * (1.0 - a) + color_high[3] * a
        color = wp.vec4(rgb[0], rgb[1], rgb[2], alpha_hsv)
    under = wp.max(-raw, 0.0)
    over = wp.max(raw - 1.0, 0.0)
    if under > 0.0 and under >= over:
        color = color_under
    if over > under:
        color = color_over

    alpha = wp.clamp(color[3], 0.0, 1.0)
    rgb_adr_offset = rgb_adr[0]
    depth_adr_offset = depth_adr[0]
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            if ox * ox + oy * oy > radius * radius:
                continue
            x = px + ox
            y = py + oy
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            local_pixelid = y * width + x
            if depth_adr_offset >= 0:
                scene_depth = depth_data[world_id, depth_adr_offset + local_pixelid]
                if scene_depth > 0.0 and line_depth > scene_depth + depth_bias:
                    continue
            pixelid = rgb_adr_offset + local_pixelid
            dst = rgb_data[world_id, pixelid]
            dst_b = float(dst & wp.uint32(0xFF))
            dst_g = float((dst >> wp.uint32(8)) & wp.uint32(0xFF))
            dst_r = float((dst >> wp.uint32(16)) & wp.uint32(0xFF))
            out_r = int(dst_r * (1.0 - alpha) + color[0] * 255.0 * alpha)
            out_g = int(dst_g * (1.0 - alpha) + color[1] * 255.0 * alpha)
            out_b = int(dst_b * (1.0 - alpha) + color[2] * 255.0 * alpha)
            rgb_data[world_id, pixelid] = _pack_rgb_bytes(out_r, out_g, out_b)


@wp.kernel(enable_backward=False)
def _draw_muscle_lines_batched_serial_kernel(
    body_xpos: wp.array2d[wp.vec3],
    body_xquat: wp.array2d[wp.quat],
    cam_xpos: wp.array2d[wp.vec3],
    cam_xmat: wp.array2d[wp.mat33],
    rgb_data: wp.array2d[wp.uint32],
    rgb_adr: wp.array[int],
    depth_data: wp.array2d[float],
    depth_adr: wp.array[int],
    camera_id: int,
    width: int,
    height: int,
    fovy: float,
    endpoint_body: wp.array[int],
    endpoint_local: wp.array[wp.vec3],
    endpoint_world: wp.array2d[wp.vec3],
    use_world: bool,
    endpoint_muscle: wp.array[int],
    endpoint_attachment_start: wp.array[int],
    endpoint_attachment_body: wp.array[int],
    endpoint_attachment_local_weight: wp.array[wp.vec4],
    use_lbs: bool,
    activation: wp.array2d[float],
    color_low: wp.vec4,
    color_high: wp.vec4,
    color_under: wp.vec4,
    color_over: wp.vec4,
    color_low_hsv: wp.vec3,
    color_high_hsv: wp.vec3,
    hue_delta: float,
    color_mode: int,
    activation_lower: float,
    activation_upper: float,
    line_width: float,
    depth_bias: float,
    sample_count: int,
) -> None:
    world_id, segment_idx = wp.tid()
    v0 = segment_idx * 2
    v1 = v0 + 1
    p0_world = _muscle_endpoint_world(
        world_id,
        v0,
        use_lbs,
        endpoint_body,
        endpoint_local,
        endpoint_world,
        use_world,
        endpoint_attachment_start,
        endpoint_attachment_body,
        endpoint_attachment_local_weight,
        body_xpos,
        body_xquat,
    )
    p1_world = _muscle_endpoint_world(
        world_id,
        v1,
        use_lbs,
        endpoint_body,
        endpoint_local,
        endpoint_world,
        use_world,
        endpoint_attachment_start,
        endpoint_attachment_body,
        endpoint_attachment_local_weight,
        body_xpos,
        body_xquat,
    )
    p0 = _project_muscle_point(
        p0_world,
        cam_xpos[world_id, camera_id],
        cam_xmat[world_id, camera_id],
        width,
        height,
        fovy,
    )
    p1 = _project_muscle_point(
        p1_world,
        cam_xpos[world_id, camera_id],
        cam_xmat[world_id, camera_id],
        width,
        height,
        fovy,
    )
    if p0[2] <= 0.0 or p1[2] <= 0.0:
        return

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    steps = int(wp.ceil(wp.max(wp.abs(dx), wp.abs(dy))))
    if steps < 1:
        steps = 1

    muscle_idx = endpoint_muscle[v0]
    raw = activation[world_id, muscle_idx]
    a = wp.clamp(raw, activation_lower, activation_upper)
    color = color_low * (1.0 - a) + color_high * a
    if color_mode == 1:
        hue = color_low_hsv[0] + hue_delta * a
        hue = hue - wp.floor(hue)
        saturation = color_low_hsv[1] * (1.0 - a) + color_high_hsv[1] * a
        value = color_low_hsv[2] * (1.0 - a) + color_high_hsv[2] * a
        rgb = _hsv_to_rgb(hue, saturation, value)
        alpha_hsv = color_low[3] * (1.0 - a) + color_high[3] * a
        color = wp.vec4(rgb[0], rgb[1], rgb[2], alpha_hsv)
    under = wp.max(-raw, 0.0)
    over = wp.max(raw - 1.0, 0.0)
    if under > 0.0 and under >= over:
        color = color_under
    if over > under:
        color = color_over

    alpha = wp.clamp(color[3], 0.0, 1.0)
    radius = int(wp.ceil(wp.max(1.0, 0.5 * line_width) * 0.5))
    rgb_adr_offset = rgb_adr[0]
    depth_adr_offset = depth_adr[0]
    sample_idx = int(0)
    while sample_idx <= steps and sample_idx < sample_count:
        t = float(sample_idx) / float(steps)
        px = int(wp.round(p0[0] + dx * t))
        py = int(wp.round(p0[1] + dy * t))
        line_depth = p0[2] * (1.0 - t) + p1[2] * t
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                if ox * ox + oy * oy > radius * radius:
                    continue
                x = px + ox
                y = py + oy
                if x < 0 or x >= width or y < 0 or y >= height:
                    continue
                local_pixelid = y * width + x
                if depth_adr_offset >= 0:
                    scene_depth = depth_data[world_id, depth_adr_offset + local_pixelid]
                    if scene_depth > 0.0 and line_depth > scene_depth + depth_bias:
                        continue
                pixelid = rgb_adr_offset + local_pixelid
                dst = rgb_data[world_id, pixelid]
                dst_b = float(dst & wp.uint32(0xFF))
                dst_g = float((dst >> wp.uint32(8)) & wp.uint32(0xFF))
                dst_r = float((dst >> wp.uint32(16)) & wp.uint32(0xFF))
                out_r = int(dst_r * (1.0 - alpha) + color[0] * 255.0 * alpha)
                out_g = int(dst_g * (1.0 - alpha) + color[1] * 255.0 * alpha)
                out_b = int(dst_b * (1.0 - alpha) + color[2] * 255.0 * alpha)
                rgb_data[world_id, pixelid] = _pack_rgb_bytes(out_r, out_g, out_b)
        sample_idx += 1


@wp.kernel(enable_backward=False)
def _clear_lbs_mesh_depth_kernel(
    far_depth: float,
    depth_out: wp.array3d[float],
) -> None:
    worldid, yid, xid = wp.tid()
    depth_out[worldid, yid, xid] = far_depth


@wp.func
def _inside_lbs_mesh_triangle(
    px: float,
    py: float,
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
) -> wp.vec4:
    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if wp.abs(denom) <= 1.0e-8:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / denom
    w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / denom
    w2 = 1.0 - w0 - w1
    inside = float(0.0)
    if w0 >= -1.0e-5 and w1 >= -1.0e-5 and w2 >= -1.0e-5:
        inside = 1.0
    return wp.vec4(w0, w1, w2, inside)


@wp.kernel(enable_backward=False)
def _draw_lbs_mesh_depth_kernel(
    body_xpos: wp.array2d[wp.vec3],
    body_xquat: wp.array2d[wp.quat],
    cam_xpos: wp.array2d[wp.vec3],
    cam_xmat: wp.array2d[wp.mat33],
    camera_id: int,
    width: int,
    height: int,
    fovy: float,
    faces: wp.array[wp.vec3i],
    vertex_body_ids: wp.array2d[int],
    vertex_local_weight: wp.array2d[wp.vec4],
    overlay_depth: wp.array3d[float],
) -> None:
    worldid, face_idx = wp.tid()
    face = faces[face_idx]
    p0_world = _lbs_mesh_vertex_world(
        worldid, face[0], vertex_body_ids, vertex_local_weight, body_xpos, body_xquat
    )
    p1_world = _lbs_mesh_vertex_world(
        worldid, face[1], vertex_body_ids, vertex_local_weight, body_xpos, body_xquat
    )
    p2_world = _lbs_mesh_vertex_world(
        worldid, face[2], vertex_body_ids, vertex_local_weight, body_xpos, body_xquat
    )
    cam_pos = cam_xpos[worldid, camera_id]
    cam_mat = cam_xmat[worldid, camera_id]
    p0 = _project_muscle_point(p0_world, cam_pos, cam_mat, width, height, fovy)
    p1 = _project_muscle_point(p1_world, cam_pos, cam_mat, width, height, fovy)
    p2 = _project_muscle_point(p2_world, cam_pos, cam_mat, width, height, fovy)
    if p0[2] <= 0.0 or p1[2] <= 0.0 or p2[2] <= 0.0:
        return
    min_x = int(wp.floor(wp.min(p0[0], wp.min(p1[0], p2[0]))))
    max_x = int(wp.ceil(wp.max(p0[0], wp.max(p1[0], p2[0]))))
    min_y = int(wp.floor(wp.min(p0[1], wp.min(p1[1], p2[1]))))
    max_y = int(wp.ceil(wp.max(p0[1], wp.max(p1[1], p2[1]))))
    min_x = wp.max(0, wp.min(width - 1, min_x))
    max_x = wp.max(0, wp.min(width - 1, max_x))
    min_y = wp.max(0, wp.min(height - 1, min_y))
    max_y = wp.max(0, wp.min(height - 1, max_y))
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            bary = _inside_lbs_mesh_triangle(float(x) + 0.5, float(y) + 0.5, p0, p1, p2)
            if bary[3] <= 0.0:
                continue
            depth = bary[0] * p0[2] + bary[1] * p1[2] + bary[2] * p2[2]
            if depth > 0.0:
                wp.atomic_min(overlay_depth, worldid, y, x, depth)


@wp.kernel(enable_backward=False)
def _update_lbs_activation_mesh_colors_kernel(
    activation: wp.array2d(dtype=float),
    activation_world_count: int,
    mesh_channel_starts: wp.array(dtype=int),
    mesh_channel_indices: wp.array(dtype=int),
    color_low: wp.vec4,
    color_high: wp.vec4,
    color_low_hsv: wp.vec3,
    color_high_hsv: wp.vec3,
    hue_delta: float,
    color_mode: int,
    mesh_colors: wp.array2d(dtype=wp.vec4),
) -> None:
    worldid, mesh_idx = wp.tid()
    material_color = color_low
    if worldid < activation_world_count:
        start = mesh_channel_starts[mesh_idx]
        end = mesh_channel_starts[mesh_idx + 1]
        strength = float(0.0)
        for item_idx in range(start, end):
            channel_idx = mesh_channel_indices[item_idx]
            value = activation[worldid, channel_idx]
            abs_value = wp.abs(value)
            if abs_value > strength:
                strength = abs_value
        strength = wp.min(float(1.0), wp.max(float(0.0), strength))
        if strength > 1.0e-6:
            material_color = _lbs_activation_color_interp(
                strength,
                color_low,
                color_high,
                color_low_hsv,
                color_high_hsv,
                hue_delta,
                color_mode,
            )
    mesh_colors[worldid, mesh_idx] = material_color


@wp.func
def _lbs_activation_color_interp(
    strength: float,
    color_low: wp.vec4,
    color_high: wp.vec4,
    color_low_hsv: wp.vec3,
    color_high_hsv: wp.vec3,
    hue_delta: float,
    color_mode: int,
) -> wp.vec4:
    value = wp.min(float(1.0), wp.max(float(0.0), strength))
    color = color_low * (float(1.0) - value) + color_high * value
    if color_mode == 1:
        hue = color_low_hsv[0] + hue_delta * value
        hue = hue - wp.floor(hue)
        saturation = color_low_hsv[1] * (float(1.0) - value) + color_high_hsv[1] * value
        hsv_value = color_low_hsv[2] * (float(1.0) - value) + color_high_hsv[2] * value
        rgb = _hsv_to_rgb(hue, saturation, hsv_value)
        alpha = color_low[3] * (float(1.0) - value) + color_high[3] * value
        color = wp.vec4(rgb[0], rgb[1], rgb[2], alpha)
    return color


@wp.kernel(enable_backward=False)
def _draw_lbs_mesh_color_kernel(
    body_xpos: wp.array2d[wp.vec3],
    body_xquat: wp.array2d[wp.quat],
    cam_xpos: wp.array2d[wp.vec3],
    cam_xmat: wp.array2d[wp.mat33],
    rgb_data: wp.array2d[wp.uint32],
    rgb_adr: wp.array[int],
    depth_data: wp.array2d[float],
    depth_adr: wp.array[int],
    camera_id: int,
    width: int,
    height: int,
    fovy: float,
    faces: wp.array[wp.vec3i],
    vertex_body_ids: wp.array2d[int],
    vertex_local_weight: wp.array2d[wp.vec4],
    vertex_local_normal: wp.array2d[wp.vec3],
    overlay_depth: wp.array3d[float],
    face_mesh_indices: wp.array(dtype=int),
    mesh_colors: wp.array2d(dtype=wp.vec4),
    use_mesh_colors: bool,
    base_color: wp.vec4,
    shading_mode: int,
) -> None:
    worldid, face_idx = wp.tid()
    face = faces[face_idx]
    p0_world = _lbs_mesh_vertex_world(
        worldid, face[0], vertex_body_ids, vertex_local_weight, body_xpos, body_xquat
    )
    p1_world = _lbs_mesh_vertex_world(
        worldid, face[1], vertex_body_ids, vertex_local_weight, body_xpos, body_xquat
    )
    p2_world = _lbs_mesh_vertex_world(
        worldid, face[2], vertex_body_ids, vertex_local_weight, body_xpos, body_xquat
    )
    tri_normal = wp.normalize(wp.cross(p1_world - p0_world, p2_world - p0_world))
    light_dir = wp.normalize(wp.vec3(-0.35, -0.45, 0.82))
    n0_world = tri_normal
    n1_world = tri_normal
    n2_world = tri_normal
    if shading_mode != 0:
        n0_world = _lbs_mesh_vertex_normal_world(
            worldid,
            face[0],
            vertex_body_ids,
            vertex_local_weight,
            vertex_local_normal,
            body_xquat,
        )
        n1_world = _lbs_mesh_vertex_normal_world(
            worldid,
            face[1],
            vertex_body_ids,
            vertex_local_weight,
            vertex_local_normal,
            body_xquat,
        )
        n2_world = _lbs_mesh_vertex_normal_world(
            worldid,
            face[2],
            vertex_body_ids,
            vertex_local_weight,
            vertex_local_normal,
            body_xquat,
        )
    cam_pos = cam_xpos[worldid, camera_id]
    cam_mat = cam_xmat[worldid, camera_id]
    p0 = _project_muscle_point(p0_world, cam_pos, cam_mat, width, height, fovy)
    p1 = _project_muscle_point(p1_world, cam_pos, cam_mat, width, height, fovy)
    p2 = _project_muscle_point(p2_world, cam_pos, cam_mat, width, height, fovy)
    if p0[2] <= 0.0 or p1[2] <= 0.0 or p2[2] <= 0.0:
        return
    min_x = int(wp.floor(wp.min(p0[0], wp.min(p1[0], p2[0]))))
    max_x = int(wp.ceil(wp.max(p0[0], wp.max(p1[0], p2[0]))))
    min_y = int(wp.floor(wp.min(p0[1], wp.min(p1[1], p2[1]))))
    max_y = int(wp.ceil(wp.max(p0[1], wp.max(p1[1], p2[1]))))
    min_x = wp.max(0, wp.min(width - 1, min_x))
    max_x = wp.max(0, wp.min(width - 1, max_x))
    min_y = wp.max(0, wp.min(height - 1, min_y))
    max_y = wp.max(0, wp.min(height - 1, max_y))
    rgb_offset = rgb_adr[0]
    scene_depth_offset = depth_adr[0]
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            bary = _inside_lbs_mesh_triangle(float(x) + 0.5, float(y) + 0.5, p0, p1, p2)
            if bary[3] <= 0.0:
                continue
            depth = bary[0] * p0[2] + bary[1] * p1[2] + bary[2] * p2[2]
            if wp.abs(depth - overlay_depth[worldid, y, x]) > 1.0e-3:
                continue
            pixelid = y * width + x
            scene_depth = depth_data[worldid, scene_depth_offset + pixelid]
            if scene_depth > 0.0 and depth > scene_depth + 1.0e-3:
                continue
            normal = tri_normal
            if shading_mode != 0:
                normal = wp.normalize(
                    bary[0] * n0_world + bary[1] * n1_world + bary[2] * n2_world
                )
            shade = 0.35 + 0.65 * wp.abs(wp.dot(normal, light_dir))
            material_color = base_color
            activation_colored = bool(False)
            if use_mesh_colors:
                material_color = mesh_colors[worldid, face_mesh_indices[face_idx]]
                activation_colored = bool(True)
            color = wp.vec3(material_color[0], material_color[1], material_color[2])
            if not activation_colored:
                color = wp.min(color * shade, wp.vec3(1.0, 1.0, 1.0))
            alpha = wp.clamp(material_color[3], 0.0, 1.0)
            dst = rgb_data[worldid, rgb_offset + pixelid]
            dst_b = float(dst & wp.uint32(0xFF))
            dst_g = float((dst >> wp.uint32(8)) & wp.uint32(0xFF))
            dst_r = float((dst >> wp.uint32(16)) & wp.uint32(0xFF))
            out_r = int(dst_r * (1.0 - alpha) + color[0] * 255.0 * alpha)
            out_g = int(dst_g * (1.0 - alpha) + color[1] * 255.0 * alpha)
            out_b = int(dst_b * (1.0 - alpha) + color[2] * 255.0 * alpha)
            rgb_data[worldid, rgb_offset + pixelid] = _pack_rgb_bytes(
                out_r,
                out_g,
                out_b,
            )


def _hsv_uniforms(
    low_rgba: np.ndarray,
    high_rgba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    low_hsv = np.asarray(colorsys.rgb_to_hsv(*low_rgba[:3]), dtype=np.float32)
    high_hsv = np.asarray(colorsys.rgb_to_hsv(*high_rgba[:3]), dtype=np.float32)
    hue_delta = (float(high_hsv[0]) - float(low_hsv[0]) + 0.5) % 1.0 - 0.5
    return low_hsv, high_hsv, float(hue_delta)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _bundle_world_to_viewer_matrix(bundle: dict[str, Any]) -> np.ndarray:
    calibration = bundle.get("calibration") or {}
    source = calibration.get("source") or {}
    matrix = source.get("coordinateTransform")
    if matrix is None:
        return np.eye(3, dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64).reshape(3, 3)


def _bundle_bind_rotation_matrices(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    calibration = bundle.get("calibration") or {}
    bone_fit = calibration.get("boneMeshFit") or {}
    qpos_bind = bone_fit.get("qposBindPose") or {}
    rotations = (
        qpos_bind.get("globalRotations") or bone_fit.get("bindRotationMatrices") or {}
    )
    return {
        str(name): np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        for name, matrix in rotations.items()
    }


def _bundle_qpos_bind_positions(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    calibration = bundle.get("calibration") or {}
    bone_fit = calibration.get("boneMeshFit") or {}
    qpos_bind = bone_fit.get("qposBindPose") or {}
    positions = qpos_bind.get("globalPositions") or {}
    return {
        str(name): np.asarray(position, dtype=np.float64).reshape(3)
        for name, position in positions.items()
    }


def _bundle_has_qpos_bind_pose(bundle: dict[str, Any]) -> bool:
    calibration = bundle.get("calibration") or {}
    bone_fit = calibration.get("boneMeshFit") or {}
    qpos_bind = bone_fit.get("qposBindPose") or {}
    return isinstance(qpos_bind, dict) and bool(qpos_bind)


def _bundle_bone_mesh_fit_transforms(
    bundle: dict[str, Any],
) -> dict[str, dict[str, np.ndarray | float]]:
    calibration = bundle.get("calibration") or {}
    bone_fit = calibration.get("boneMeshFit") or {}
    if bone_fit.get("qposBindPose"):
        return {}
    per_bone = bone_fit.get("perBone") or {}
    transforms: dict[str, dict[str, np.ndarray | float]] = {}
    for bone_name, payload in per_bone.items():
        if not isinstance(payload, dict) or payload.get("status") != "fit":
            continue
        scale = float(payload.get("scale", 1.0))
        rotation = np.asarray(payload.get("rotation", np.eye(3)), dtype=np.float64)
        translation = np.asarray(
            payload.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        if rotation.shape != (3, 3) or translation.shape != (3,) or scale <= 1.0e-12:
            continue
        transforms[str(bone_name)] = {
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
        }
    return transforms


def _load_lbs_mesh_bundle_with_manifest(bundle_path: Path) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest_path = bundle_path.parent / "manifest.json"
    if not manifest_path.exists():
        return bundle
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("boneNames", "bindPositions", "calibration"):
        if key not in bundle and key in manifest:
            bundle[key] = manifest[key]
    return bundle


class _WarpLBSMeshOverlay:
    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        bundle_path: str | Path,
        mode: str,
        shading_mode: str | None = None,
        activation_mapping_path: str | Path | None = None,
        activation_tensor: torch.Tensor | None = None,
        activation_color_settings: _LBSActivationColorSettings | None = None,
    ) -> None:
        self.mode = _resolve_lbs_mesh_mode(mode)
        if self.mode == "off":
            raise ValueError("_WarpLBSMeshOverlay requires mode skin or muscle.")
        self.shading_mode = _resolve_lbs_mesh_shading_mode(shading_mode)
        self.bundle_path = _resolve_lbs_mesh_bundle_path(bundle_path)
        if not self.bundle_path.exists():
            raise FileNotFoundError(
                f"LBS mesh bundle does not exist: {self.bundle_path.as_posix()}"
            )
        bundle = _load_lbs_mesh_bundle_with_manifest(self.bundle_path)
        selected_meshes = [
            mesh
            for mesh in bundle.get("meshes", [])
            if str(mesh.get("type", "")).lower() == self.mode
        ]
        if not selected_meshes:
            raise ValueError(
                f"No {self.mode!r} meshes found in {self.bundle_path.as_posix()}."
            )
        (
            vertices,
            faces,
            normals,
            bone_indices,
            bone_weights,
            mesh_names,
            face_mesh_indices,
        ) = self._pack_meshes(selected_meshes)
        self.mesh_names = mesh_names
        self.mesh_count = int(len(mesh_names))
        self.vertex_count = int(len(vertices))
        self.face_count = int(len(faces))
        (
            self.vertex_body_ids_np,
            self.vertex_local_weight_np,
            self.vertex_local_normal_np,
        ) = self._build_vertex_attachments(
            model=model,
            bundle=bundle,
            vertices=vertices,
            normals=normals,
            bone_indices=bone_indices,
            bone_weights=bone_weights,
        )
        self.faces = wp.array(np.asarray(faces, dtype=np.int32), dtype=wp.vec3i)
        self.face_mesh_indices = wp.array(
            np.asarray(face_mesh_indices, dtype=np.int32),
            dtype=int,
        )
        self.vertex_body_ids = wp.array(
            self.vertex_body_ids_np,
            dtype=int,
        )
        self.vertex_local_weight = wp.array(
            self.vertex_local_weight_np,
            dtype=wp.vec4,
        )
        self.vertex_local_normal = wp.array(
            self.vertex_local_normal_np,
            dtype=wp.vec3,
        )
        base_rgba = np.asarray(
            [0.78, 0.09, 0.07, 1.0]
            if self.mode == "muscle"
            else [0.70, 0.53, 0.41, 1.0],
            dtype=np.float32,
        )
        self.base_color = wp.vec4(base_rgba)
        if activation_color_settings is None:
            activation_color_settings = _LBSActivationColorSettings(
                color_low_rgba=np.asarray(
                    [0.20, 0.25, 0.90, 0.3],
                    dtype=np.float32,
                ),
                color_high_rgba=np.asarray([1.00, 0.15, 0.10, 1.0], dtype=np.float32),
                color_interpolation="hsv",
            )
        self.activation_color_low_np = np.asarray(
            activation_color_settings.color_low_rgba,
            dtype=np.float32,
        )
        self.activation_color_high_np = np.asarray(
            activation_color_settings.color_high_rgba,
            dtype=np.float32,
        )
        self.activation_color_interpolation = (
            activation_color_settings.color_interpolation
        )
        self.activation_color_low = wp.vec4(self.activation_color_low_np)
        self.activation_color_high = wp.vec4(self.activation_color_high_np)
        low_hsv, high_hsv, hue_delta = _hsv_uniforms(
            self.activation_color_low_np,
            self.activation_color_high_np,
        )
        self.activation_color_low_hsv = wp.vec3(low_hsv)
        self.activation_color_high_hsv = wp.vec3(high_hsv)
        self.activation_hue_delta = float(hue_delta)
        self.activation_color_mode = (
            1 if self.activation_color_interpolation == "hsv" else 0
        )
        self.activation_tensor = activation_tensor
        self.activation_mapping: _LBSActivationMapping | None = None
        self.activation_mesh_channel_starts: Any | None = None
        self.activation_mesh_channel_indices: Any | None = None
        resolved_mapping_path = _resolve_lbs_activation_mapping_path(
            activation_mapping_path
        )
        if resolved_mapping_path is not None:
            if self.mode != "muscle":
                raise ValueError(
                    "LBS activation mapping is supported only for muscle mode."
                )
            self.activation_mapping = _load_lbs_activation_mapping(
                resolved_mapping_path,
                selected_mesh_names=self.mesh_names,
            )
            self.activation_mesh_channel_starts = wp.array(
                self.activation_mapping.mesh_channel_starts,
                dtype=int,
            )
            self.activation_mesh_channel_indices = wp.array(
                self.activation_mapping.mesh_channel_indices,
                dtype=int,
            )
        self._depth_buffer: Any | None = None
        self._depth_buffer_shape: tuple[int, int, int] | None = None
        self._mesh_color_buffer: Any | None = None
        self._mesh_color_buffer_shape: tuple[int, int] | None = None
        self._activation_buffer: Any | None = None
        self._activation_buffer_shape: tuple[int, int] | None = None
        self._prepared_use_mesh_colors = False

    @property
    def prepared_graph_signature(self) -> tuple[Any, ...]:
        """Return graph-relevant buffer identity and fixed draw branches."""
        return (
            self._depth_buffer_shape,
            None if self._depth_buffer is None else int(self._depth_buffer.ptr),
            self._mesh_color_buffer_shape,
            None
            if self._mesh_color_buffer is None
            else int(self._mesh_color_buffer.ptr),
            self._activation_buffer_shape,
            None
            if self._activation_buffer is None
            else int(self._activation_buffer.ptr),
            bool(self._prepared_use_mesh_colors),
        )

    @staticmethod
    def _compute_vertex_normals(
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        normals = np.zeros_like(vertices, dtype=np.float64)
        for face in np.asarray(faces, dtype=np.int32):
            p0 = vertices[int(face[0])]
            p1 = vertices[int(face[1])]
            p2 = vertices[int(face[2])]
            face_normal = np.cross(p1 - p0, p2 - p0)
            norm = float(np.linalg.norm(face_normal))
            if norm <= 1.0e-12:
                continue
            face_normal = face_normal / norm
            normals[int(face[0])] += face_normal
            normals[int(face[1])] += face_normal
            normals[int(face[2])] += face_normal
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        valid = lengths[:, 0] > 1.0e-12
        normals[valid] /= lengths[valid]
        normals[~valid] = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        return normals

    @staticmethod
    def _weld_split_vertex_normals(
        normals: np.ndarray,
        source_vertex_indices: list[int] | np.ndarray | None,
    ) -> np.ndarray:
        if source_vertex_indices is None:
            return normals
        source_ids = np.asarray(source_vertex_indices, dtype=np.int64).reshape(-1)
        if len(source_ids) != len(normals):
            return normals
        unique_ids, inverse = np.unique(source_ids, return_inverse=True)
        group_count = len(unique_ids)
        welded_groups = np.zeros((group_count, 3), dtype=np.float64)
        counts = np.zeros((group_count, 1), dtype=np.float64)
        np.add.at(welded_groups, inverse, normals)
        np.add.at(counts[:, 0], inverse, 1.0)
        welded_groups /= np.maximum(counts, 1.0)
        lengths = np.linalg.norm(welded_groups, axis=1, keepdims=True)
        valid = lengths[:, 0] > 1.0e-12
        welded_groups[valid] /= lengths[valid]
        welded_groups[~valid] = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        return welded_groups[inverse]

    @staticmethod
    def _pack_meshes(
        meshes: list[dict[str, Any]],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[str],
        np.ndarray,
    ]:
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        face_mesh_indices: list[int] = []
        normals: list[list[float]] = []
        bone_indices: list[list[int]] = []
        bone_weights: list[list[float]] = []
        mesh_names: list[str] = []
        vertex_offset = 0
        for mesh in meshes:
            mesh_vertices = mesh.get("vertices") or []
            mesh_faces = mesh.get("faces") or []
            mesh_normals = mesh.get("normals") or []
            mesh_indices = mesh.get("boneIndices") or []
            mesh_weights = mesh.get("boneWeights") or []
            if not mesh_vertices or not mesh_faces:
                continue
            mesh_index = len(mesh_names)
            mesh_names.append(str(mesh.get("name", f"mesh_{mesh_index}")))
            if len(mesh_indices) != len(mesh_vertices) or len(mesh_weights) != len(
                mesh_vertices
            ):
                raise ValueError(
                    f"Mesh {mesh.get('name', '<unnamed>')} has inconsistent LBS rows."
                )
            if mesh_normals and len(mesh_normals) != len(mesh_vertices):
                raise ValueError(
                    f"Mesh {mesh.get('name', '<unnamed>')} has inconsistent normal rows."
                )
            vertices.extend(mesh_vertices)
            bone_indices.extend(mesh_indices)
            bone_weights.extend(mesh_weights)
            source_vertex_indices = mesh.get("sourceVertexIndices")
            if mesh_normals:
                mesh_normal_array = np.asarray(mesh_normals, dtype=np.float64)
            else:
                mesh_normal_array = _WarpLBSMeshOverlay._compute_vertex_normals(
                    np.asarray(mesh_vertices, dtype=np.float64),
                    np.asarray(mesh_faces, dtype=np.int32),
                )
            mesh_normal_array = _WarpLBSMeshOverlay._weld_split_vertex_normals(
                mesh_normal_array,
                source_vertex_indices,
            )
            normals.extend(mesh_normal_array.tolist())
            for face in mesh_faces:
                faces.append(
                    [
                        int(face[0]) + vertex_offset,
                        int(face[1]) + vertex_offset,
                        int(face[2]) + vertex_offset,
                    ]
                )
                face_mesh_indices.append(mesh_index)
            vertex_offset += len(mesh_vertices)
        if not vertices or not faces:
            raise ValueError(
                "Selected LBS mesh payloads contain no renderable triangles."
            )
        return (
            np.asarray(vertices, dtype=np.float64),
            np.asarray(faces, dtype=np.int32),
            np.asarray(normals, dtype=np.float64),
            np.asarray(bone_indices, dtype=np.int32),
            np.asarray(bone_weights, dtype=np.float64),
            mesh_names,
            np.asarray(face_mesh_indices, dtype=np.int32),
        )

    @staticmethod
    def _normalize_vec3(
        value: np.ndarray,
        fallback: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-12:
            return np.asarray(fallback, dtype=np.float64)
        return vector / norm

    @staticmethod
    def _local_lbs_attachment(
        *,
        vertex_viewer: np.ndarray,
        normal_viewer: np.ndarray,
        bone_index: int,
        viewer_to_world: np.ndarray,
        rest_pos_world: dict[int, np.ndarray],
        rest_rot_world: dict[int, np.ndarray],
        fit_pos_world: dict[int, np.ndarray],
        fit_rot_world: dict[int, np.ndarray],
        mesh_fit: dict[str, np.ndarray | float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if mesh_fit is not None:
            fit_rotation = np.asarray(mesh_fit["rotation"], dtype=np.float64)
            source_viewer = (
                vertex_viewer - np.asarray(mesh_fit["translation"], dtype=np.float64)
            ) @ fit_rotation
            source_viewer = source_viewer / float(mesh_fit["scale"])
            source_world = viewer_to_world @ source_viewer
            local = rest_rot_world[bone_index].T @ (
                source_world - rest_pos_world[bone_index]
            )

            source_normal = normal_viewer @ fit_rotation
            normal_world = viewer_to_world @ source_normal
            local_normal = rest_rot_world[bone_index].T @ normal_world
            return local, _WarpLBSMeshOverlay._normalize_vec3(local_normal)

        vertex_world = viewer_to_world @ vertex_viewer
        local = fit_rot_world[bone_index].T @ (vertex_world - fit_pos_world[bone_index])
        normal_world = viewer_to_world @ normal_viewer
        local_normal = fit_rot_world[bone_index].T @ normal_world
        return local, _WarpLBSMeshOverlay._normalize_vec3(local_normal)

    @staticmethod
    def _build_vertex_attachments(
        *,
        model: mujoco.MjModel,
        bundle: dict[str, Any],
        vertices: np.ndarray,
        normals: np.ndarray,
        bone_indices: np.ndarray,
        bone_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bone_names = [str(name) for name in bundle.get("boneNames", [])]
        bind_positions = bundle.get("bindPositions") or {}
        qpos_bind_positions = _bundle_qpos_bind_positions(bundle)
        bind_rotations_are_absolute = _bundle_has_qpos_bind_pose(bundle)
        world_to_viewer = _bundle_world_to_viewer_matrix(bundle)
        viewer_to_world = np.linalg.inv(world_to_viewer)
        bind_rotations_viewer = _bundle_bind_rotation_matrices(bundle)
        mesh_fit_transforms = _bundle_bone_mesh_fit_transforms(bundle)
        rest_data = mujoco.MjData(model)
        mujoco.mj_forward(model, rest_data)
        body_ids: dict[int, int] = {}
        fit_pos_world: dict[int, np.ndarray] = {}
        fit_rot_world: dict[int, np.ndarray] = {}
        rest_pos_world: dict[int, np.ndarray] = {}
        rest_rot_world: dict[int, np.ndarray] = {}
        mesh_fit_by_bone_index: dict[int, dict[str, np.ndarray | float]] = {}
        for bone_index, bone_name in enumerate(bone_names):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bone_name)
            if body_id < 0:
                raise ValueError(
                    f"Bone {bone_name!r} is not a body in the render model."
                )
            if bone_name not in bind_positions:
                raise ValueError(
                    f"Bone {bone_name!r} is missing from bundle bindPositions."
                )
            body_ids[bone_index] = int(body_id)
            rest_pos_world[bone_index] = np.asarray(
                rest_data.xpos[body_id], dtype=np.float64
            )
            rest_rot = _quat_wxyz_to_matrix(np.asarray(rest_data.xquat[body_id]))
            rest_rot_world[bone_index] = rest_rot
            fit_pos_viewer = qpos_bind_positions.get(
                bone_name,
                np.asarray(bind_positions[bone_name], dtype=np.float64),
            )
            fit_pos_world[bone_index] = viewer_to_world @ np.asarray(
                fit_pos_viewer,
                dtype=np.float64,
            )
            delta_viewer = bind_rotations_viewer.get(
                bone_name, np.eye(3, dtype=np.float64)
            )
            delta_world = viewer_to_world @ delta_viewer @ world_to_viewer
            fit_rot_world[bone_index] = (
                delta_world if bind_rotations_are_absolute else delta_world @ rest_rot
            )
            if bone_name in mesh_fit_transforms:
                mesh_fit_by_bone_index[bone_index] = mesh_fit_transforms[bone_name]

        vertex_body_ids = np.zeros((len(vertices), 4), dtype=np.int32)
        vertex_local_weight = np.zeros((len(vertices), 4, 4), dtype=np.float32)
        vertex_local_normal = np.zeros((len(vertices), 4, 3), dtype=np.float32)
        for vertex_idx, vertex_viewer in enumerate(vertices):
            weight_sum = float(np.sum(np.maximum(bone_weights[vertex_idx], 0.0)))
            if weight_sum <= 1.0e-8:
                vertex_body_ids[vertex_idx, 0] = body_ids.get(0, 0)
                vertex_local_weight[vertex_idx, 0, 3] = 1.0
                vertex_local_normal[vertex_idx, 0] = np.asarray(
                    [0.0, 0.0, 1.0], dtype=np.float32
                )
                continue
            vertex_normal_viewer = _WarpLBSMeshOverlay._normalize_vec3(
                normals[vertex_idx]
            )
            for slot in range(4):
                bone_index = int(bone_indices[vertex_idx, slot])
                weight = max(0.0, float(bone_weights[vertex_idx, slot])) / weight_sum
                if weight <= 0.0:
                    vertex_body_ids[vertex_idx, slot] = body_ids.get(0, 0)
                    continue
                if bone_index not in body_ids:
                    raise ValueError(
                        f"Bone index {bone_index} is outside bundle boneNames."
                    )
                vertex_body_ids[vertex_idx, slot] = body_ids[bone_index]
                local, local_normal = _WarpLBSMeshOverlay._local_lbs_attachment(
                    vertex_viewer=np.asarray(vertex_viewer, dtype=np.float64),
                    normal_viewer=vertex_normal_viewer,
                    bone_index=bone_index,
                    viewer_to_world=viewer_to_world,
                    rest_pos_world=rest_pos_world,
                    rest_rot_world=rest_rot_world,
                    fit_pos_world=fit_pos_world,
                    fit_rot_world=fit_rot_world,
                    mesh_fit=mesh_fit_by_bone_index.get(bone_index),
                )
                vertex_local_weight[vertex_idx, slot, :3] = local.astype(np.float32)
                vertex_local_weight[vertex_idx, slot, 3] = float(weight)
                vertex_local_normal[vertex_idx, slot] = local_normal.astype(np.float32)
        return vertex_body_ids, vertex_local_weight, vertex_local_normal

    def _ensure_depth_buffer(
        self, data: Any, world_count: int, height: int, width: int
    ) -> Any:
        shape = (int(world_count), int(height), int(width))
        device = data.xpos.device
        if self._depth_buffer is None or self._depth_buffer_shape != shape:
            self._depth_buffer = wp.empty(shape, dtype=float, device=device)
            self._depth_buffer_shape = shape
        return self._depth_buffer

    def _ensure_mesh_color_buffer(self, data: Any, world_count: int) -> Any:
        shape = (int(world_count), int(self.mesh_count))
        device = data.xpos.device
        if self._mesh_color_buffer is None or self._mesh_color_buffer_shape != shape:
            self._mesh_color_buffer = wp.empty(shape, dtype=wp.vec4, device=device)
            self._mesh_color_buffer_shape = shape
        return self._mesh_color_buffer

    def _stage_activation_tensor(
        self,
        data: Any,
        activation_batch: torch.Tensor | None,
        *,
        world_count: int,
    ) -> int:
        if self.activation_mapping is None:
            return 0
        activation = (
            activation_batch if activation_batch is not None else self.activation_tensor
        )
        if activation is None:
            return 0
        if not torch.is_tensor(activation):
            activation = torch.as_tensor(activation, dtype=torch.float32)
        if activation.ndim == 1:
            activation = activation.reshape(1, -1)
        if activation.ndim != 2:
            raise ValueError("LBS activation tensor must be rank-1 or rank-2.")
        if int(activation.shape[1]) != int(
            self.activation_mapping.activation_channel_count
        ):
            raise ValueError(
                "LBS activation tensor channel count does not match activation mapping."
            )
        activation = activation[: int(world_count)]
        if activation.dtype != torch.float32:
            activation = activation.to(dtype=torch.float32)
        if not activation.is_cuda and torch.cuda.is_available():
            activation = activation.to("cuda")
        if not activation.is_contiguous():
            activation = activation.contiguous()
        activation_shape = tuple(int(value) for value in activation.shape)
        if (
            self._activation_buffer is None
            or self._activation_buffer_shape != activation_shape
        ):
            self._activation_buffer = wp.empty(
                activation_shape,
                dtype=wp.float32,
                device=data.xpos.device,
            )
            self._activation_buffer_shape = activation_shape
        wp.copy(
            self._activation_buffer,
            wp.from_torch(activation, dtype=wp.float32, requires_grad=False),
        )
        return int(activation.shape[0])

    def draw_batch(
        self,
        *,
        data: Any,
        rc: Any,
        camera_id: int,
        width: int,
        height: int,
        fovy: float,
        world_count: int,
        activation_batch: torch.Tensor | None = None,
    ) -> None:
        world_count = max(1, int(world_count))
        self.prepare_batch(
            data=data,
            world_count=world_count,
            width=width,
            height=height,
            activation_batch=activation_batch,
        )
        self.draw_prepared_batch(
            data=data,
            rc=rc,
            camera_id=camera_id,
            width=width,
            height=height,
            fovy=fovy,
            world_count=world_count,
        )

    def prepare_batch(
        self,
        *,
        data: Any,
        world_count: int,
        width: int,
        height: int,
        activation_batch: torch.Tensor | None = None,
    ) -> None:
        """Allocate scratch and stage activation into stable device storage."""
        world_count = max(1, int(world_count))
        self._ensure_depth_buffer(data, world_count, height, width)
        self._ensure_mesh_color_buffer(data, world_count)
        activation_world_count = self._stage_activation_tensor(
            data,
            activation_batch,
            world_count=world_count,
        )
        self._prepared_use_mesh_colors = activation_world_count > 0

    def draw_prepared_batch(
        self,
        *,
        data: Any,
        rc: Any,
        camera_id: int,
        width: int,
        height: int,
        fovy: float,
        world_count: int,
    ) -> None:
        """Draw only from persistent buffers; safe to include in a CUDA graph."""
        overlay_depth = self._depth_buffer
        mesh_colors = self._mesh_color_buffer
        if overlay_depth is None or mesh_colors is None:
            raise RuntimeError(
                "prepare_batch must be called before draw_prepared_batch."
            )
        if self._prepared_use_mesh_colors:
            assert self._activation_buffer is not None
            assert self.activation_mesh_channel_starts is not None
            assert self.activation_mesh_channel_indices is not None
            wp.launch(
                _update_lbs_activation_mesh_colors_kernel,
                dim=(int(world_count), int(self.mesh_count)),
                inputs=[
                    self._activation_buffer,
                    int(self._activation_buffer_shape[0]),
                    self.activation_mesh_channel_starts,
                    self.activation_mesh_channel_indices,
                    self.activation_color_low,
                    self.activation_color_high,
                    self.activation_color_low_hsv,
                    self.activation_color_high_hsv,
                    float(self.activation_hue_delta),
                    int(self.activation_color_mode),
                ],
                outputs=[mesh_colors],
            )
        wp.launch(
            _clear_lbs_mesh_depth_kernel,
            dim=(world_count, int(height), int(width)),
            inputs=[1.0e20],
            outputs=[overlay_depth],
        )
        wp.launch(
            _draw_lbs_mesh_depth_kernel,
            dim=(world_count, self.face_count),
            inputs=[
                data.xpos,
                data.xquat,
                data.cam_xpos,
                data.cam_xmat,
                int(camera_id),
                int(width),
                int(height),
                float(fovy),
                self.faces,
                self.vertex_body_ids,
                self.vertex_local_weight,
            ],
            outputs=[overlay_depth],
        )
        wp.launch(
            _draw_lbs_mesh_color_kernel,
            dim=(world_count, self.face_count),
            inputs=[
                data.xpos,
                data.xquat,
                data.cam_xpos,
                data.cam_xmat,
                rc.rgb_data,
                rc.rgb_adr,
                rc.depth_data,
                rc.depth_adr,
                int(camera_id),
                int(width),
                int(height),
                float(fovy),
                self.faces,
                self.vertex_body_ids,
                self.vertex_local_weight,
                self.vertex_local_normal,
                overlay_depth,
                self.face_mesh_indices,
                mesh_colors,
                bool(self._prepared_use_mesh_colors),
                self.base_color,
                0 if self.shading_mode == "flat" else 1,
            ],
        )

    def draw(
        self,
        *,
        data: Any,
        rc: Any,
        camera_id: int,
        width: int,
        height: int,
        fovy: float,
        env_idx: int = 0,
        world_id: int = 0,
        activation: torch.Tensor | None = None,
    ) -> None:
        del env_idx, world_id
        self.draw_batch(
            data=data,
            rc=rc,
            camera_id=camera_id,
            width=width,
            height=height,
            fovy=fovy,
            world_count=1,
            activation_batch=activation,
        )


class _WarpMuscleLineOverlay:
    def __init__(self, source: Any) -> None:
        self._source = source
        self.segment_count = int(source.segment_count)
        self.endpoint_body = wp.array(
            np.asarray(source.endpoint_body, dtype=np.int32),
            dtype=int,
        )
        self.endpoint_local = wp.array(
            np.asarray(source.endpoint_local, dtype=np.float32),
            dtype=wp.vec3,
        )
        self.endpoint_muscle = wp.array(
            np.asarray(source.endpoint_muscle, dtype=np.int32),
            dtype=int,
        )
        if bool(getattr(source, "uses_lbs_endpoints", False)):
            self.use_lbs = True
            self.endpoint_attachment_start = wp.array(
                np.asarray(source.endpoint_attachment_start, dtype=np.int32),
                dtype=int,
            )
            self.endpoint_attachment_body = wp.array(
                np.asarray(source.endpoint_attachment_body, dtype=np.int32),
                dtype=int,
            )
            self.endpoint_attachment_local_weight = wp.array(
                np.asarray(
                    source.endpoint_attachment_local_weight,
                    dtype=np.float32,
                ),
                dtype=wp.vec4,
            )
        else:
            self.use_lbs = False
            self.endpoint_attachment_start = wp.array([0], dtype=int)
            self.endpoint_attachment_body = wp.array([0], dtype=int)
            self.endpoint_attachment_local_weight = wp.array(
                np.zeros((1, 4), dtype=np.float32),
                dtype=wp.vec4,
            )
        color_low = np.asarray(source.color_low_rgba, dtype=np.float32)
        color_high = np.asarray(source.color_high_rgba, dtype=np.float32)
        self.color_low = wp.vec4(color_low)
        self.color_high = wp.vec4(color_high)
        self.color_under = wp.vec4(
            np.asarray(source.color_under_rgba, dtype=np.float32)
        )
        self.color_over = wp.vec4(np.asarray(source.color_over_rgba, dtype=np.float32))
        low_hsv, high_hsv, hue_delta = _hsv_uniforms(color_low, color_high)
        self.color_low_hsv = wp.vec3(low_hsv)
        self.color_high_hsv = wp.vec3(high_hsv)
        self.hue_delta = float(hue_delta)
        self.color_mode = (
            1 if getattr(source, "color_interpolation", "rgb") == "hsv" else 0
        )
        self.line_width = float(source.line_width)
        self._endpoint_world = wp.zeros(
            (1, 1), dtype=wp.vec3, device=self.endpoint_body.device
        )
        self._uses_world = False
        self._activation_batch: Any | None = None
        self._activation_batch_shape: tuple[int, int] | None = None
        self._activation_single: Any | None = None

    @property
    def prepared_graph_signature(self) -> tuple[Any, ...]:
        """Return graph-relevant buffer identity and fixed draw branches."""
        return (
            self._uses_world,
            int(self._endpoint_world.ptr),
            self._activation_batch_shape,
            None if self._activation_batch is None else int(self._activation_batch.ptr),
            None
            if self._activation_single is None
            else int(self._activation_single.ptr),
        )

    def draw(
        self,
        *,
        data: Any,
        rc: Any,
        camera_id: int,
        width: int,
        height: int,
        fovy: float,
        env_idx: int = 0,
        world_id: int = 0,
        activation: torch.Tensor | None = None,
    ) -> None:
        self.prepare_single(
            data=data,
            env_idx=env_idx,
            activation=activation,
        )
        self.draw_prepared_single(
            data=data,
            rc=rc,
            camera_id=camera_id,
            width=width,
            height=height,
            fovy=fovy,
            world_id=world_id,
        )

    def prepare_single(
        self,
        *,
        data: Any,
        env_idx: int,
        activation: torch.Tensor | None = None,
    ) -> None:
        """Stage one source environment without changing its selection semantics."""
        device = getattr(data.xpos, "device", self.endpoint_body.device)
        resolver = getattr(self._source, "endpoint_world_resolver", None)
        self._uses_world = resolver is not None
        if resolver is not None:
            self._endpoint_world = wp.from_torch(resolver(data), dtype=wp.vec3)
        values = (
            self._resolve_activation_tensor(env_idx)
            if activation is None
            else activation
        )
        if not torch.is_tensor(values):
            values = torch.as_tensor(values)
        values = values.detach().reshape(-1)
        if int(values.shape[0]) != int(self._source.muscle_count):
            values = torch.zeros(
                int(self._source.muscle_count),
                dtype=torch.float32,
                device=values.device,
            )
        clamp_resolver = getattr(
            self._source, "activation_clamp_enabled_resolver", None
        )
        if (
            activation is not None
            and callable(clamp_resolver)
            and clamp_resolver() != 0.0
        ):
            values = values.clamp(
                float(self._source.activation_lower_bound),
                float(self._source.activation_upper_bound),
            )
        values = self._coerce_activation_device(values)
        if self._activation_single is None:
            self._activation_single = wp.empty(
                int(self._source.muscle_count), dtype=wp.float32, device=device
            )
        wp.copy(
            self._activation_single,
            wp.from_torch(values, dtype=wp.float32, requires_grad=False),
        )

    def draw_prepared_single(
        self,
        *,
        data: Any,
        rc: Any,
        camera_id: int,
        width: int,
        height: int,
        fovy: float,
        world_id: int = 0,
    ) -> None:
        if self._activation_single is None:
            raise RuntimeError(
                "prepare_single must be called before draw_prepared_single."
            )
        wp.launch(
            _draw_muscle_lines_kernel,
            dim=(self.segment_count, max(1, int(max(width, height)))),
            inputs=[
                data.xpos,
                data.xquat,
                data.cam_xpos,
                data.cam_xmat,
                rc.rgb_data,
                rc.rgb_adr,
                rc.depth_data,
                rc.depth_adr,
                int(camera_id),
                int(world_id),
                int(width),
                int(height),
                float(fovy),
                self.endpoint_body,
                self.endpoint_local,
                self._endpoint_world,
                self._uses_world,
                self.endpoint_muscle,
                self.endpoint_attachment_start,
                self.endpoint_attachment_body,
                self.endpoint_attachment_local_weight,
                bool(self.use_lbs),
                self._activation_single,
                self.color_low,
                self.color_high,
                self.color_under,
                self.color_over,
                self.color_low_hsv,
                self.color_high_hsv,
                self.hue_delta,
                int(self.color_mode),
                float(self._source.activation_lower_bound),
                float(self._source.activation_upper_bound),
                self.line_width,
                1.0e-3,
            ],
        )

    def draw_batch(
        self,
        *,
        data: Any,
        rc: Any,
        camera_id: int,
        width: int,
        height: int,
        fovy: float,
        world_count: int,
        activation_batch: torch.Tensor | None = None,
    ) -> None:
        world_count = max(1, int(world_count))
        self.prepare_batch(
            data=data,
            world_count=world_count,
            activation_batch=activation_batch,
        )
        self.draw_prepared_batch(
            data=data,
            rc=rc,
            camera_id=camera_id,
            width=width,
            height=height,
            fovy=fovy,
            world_count=world_count,
        )

    def prepare_batch(
        self,
        *,
        data: Any,
        world_count: int,
        activation_batch: torch.Tensor | None = None,
    ) -> None:
        """Stage changing inputs into stable buffers outside graph capture."""
        world_count = max(1, int(world_count))
        device = getattr(data.xpos, "device", self.endpoint_body.device)
        resolver = getattr(self._source, "endpoint_world_resolver", None)
        self._uses_world = resolver is not None
        if resolver is not None:
            self._endpoint_world = wp.from_torch(resolver(data), dtype=wp.vec3)
        activation_shape = (world_count, int(self._source.muscle_count))
        if (
            self._activation_batch is None
            or self._activation_batch_shape != activation_shape
        ):
            self._activation_batch = wp.empty(
                activation_shape, dtype=wp.float32, device=device
            )
            self._activation_batch_shape = activation_shape
        activation = self._resolve_activation_batch_tensor(
            world_count,
            activation_batch=activation_batch,
        )
        if activation is None:
            self._activation_batch.zero_()
        else:
            wp.copy(
                self._activation_batch,
                wp.from_torch(activation, dtype=wp.float32, requires_grad=False),
            )

    def draw_prepared_batch(
        self,
        *,
        data: Any,
        rc: Any,
        camera_id: int,
        width: int,
        height: int,
        fovy: float,
        world_count: int,
    ) -> None:
        """Draw only from stable buffers; safe to include in a CUDA graph."""
        if self._activation_batch is None:
            raise RuntimeError(
                "prepare_batch must be called before draw_prepared_batch."
            )
        sample_count = max(1, int(max(width, height)))
        wp.launch(
            _draw_muscle_lines_batched_serial_kernel,
            dim=(world_count, self.segment_count),
            inputs=[
                data.xpos,
                data.xquat,
                data.cam_xpos,
                data.cam_xmat,
                rc.rgb_data,
                rc.rgb_adr,
                rc.depth_data,
                rc.depth_adr,
                int(camera_id),
                int(width),
                int(height),
                float(fovy),
                self.endpoint_body,
                self.endpoint_local,
                self._endpoint_world,
                self._uses_world,
                self.endpoint_muscle,
                self.endpoint_attachment_start,
                self.endpoint_attachment_body,
                self.endpoint_attachment_local_weight,
                bool(self.use_lbs),
                self._activation_batch,
                self.color_low,
                self.color_high,
                self.color_under,
                self.color_over,
                self.color_low_hsv,
                self.color_high_hsv,
                self.hue_delta,
                int(self.color_mode),
                float(self._source.activation_lower_bound),
                float(self._source.activation_upper_bound),
                self.line_width,
                1.0e-3,
                int(sample_count),
            ],
        )

    def _coerce_activation_device(self, activation: torch.Tensor) -> torch.Tensor:
        activation = activation.to(dtype=torch.float32)
        if not activation.is_cuda and torch.cuda.is_available():
            activation = activation.to("cuda")
        return activation.contiguous()

    def _resolve_activation_batch_tensor(
        self,
        world_count: int,
        *,
        activation_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        world_count = max(1, int(world_count))
        muscle_count = int(self._source.muscle_count)
        resolver = getattr(self._source, "activation_resolver", None)
        if activation_batch is not None:
            activation = activation_batch
        elif callable(resolver):
            activation = resolver()
        else:
            activation = self._source.resolve_activation_tensor()
        if activation is None:
            return None
        if not torch.is_tensor(activation):
            activation = torch.as_tensor(activation)
        activation = activation.detach()
        if activation.ndim == 1:
            if int(activation.shape[0]) != muscle_count:
                return None
            activation = activation.reshape(1, muscle_count).expand(world_count, -1)
        elif activation.ndim >= 2:
            activation = activation.reshape(int(activation.shape[0]), -1)
            if int(activation.shape[1]) != muscle_count:
                return None
            if int(activation.shape[0]) < world_count:
                env_idx = torch.arange(world_count, device=activation.device).clamp(
                    max=int(activation.shape[0]) - 1
                )
                activation = activation.index_select(0, env_idx)
            else:
                activation = activation[:world_count]
        else:
            return None
        clamp_resolver = getattr(
            self._source,
            "activation_clamp_enabled_resolver",
            None,
        )
        if callable(clamp_resolver) and clamp_resolver() != 0.0:
            activation = activation.clamp(
                float(self._source.activation_lower_bound),
                float(self._source.activation_upper_bound),
            )
        return self._coerce_activation_device(activation)

    def _resolve_activation_tensor(self, env_idx: int) -> torch.Tensor:
        resolver = getattr(self._source, "activation_resolver", None)
        if callable(resolver):
            activation = resolver()
            if activation is None:
                return torch.zeros(
                    (int(self._source.muscle_count),), dtype=torch.float32
                )
            source_env_count = int(activation.shape[0]) if activation.ndim > 1 else 1
            selected_env = max(0, min(int(env_idx), source_env_count - 1))
            activations = (
                activation[selected_env].detach()
                if activation.ndim > 1
                else activation.detach()
            )
            clamp_resolver = getattr(
                self._source,
                "activation_clamp_enabled_resolver",
                None,
            )
            if callable(clamp_resolver) and clamp_resolver() != 0.0:
                activations = activations.clamp(
                    float(self._source.activation_lower_bound),
                    float(self._source.activation_upper_bound),
                )
            return activations.to(dtype=torch.float32)
        return self._source.resolve_activation_tensor()


def _resolve_dynamic_muscle_line_overlays(env_raw: Any) -> list[_WarpMuscleLineOverlay]:
    source_getter = getattr(env_raw, "dynamic_gl_render_sources", None)
    if not callable(source_getter):
        return []
    overlays: list[_WarpMuscleLineOverlay] = []
    try:
        sources = list(source_getter())
    except Exception:
        logger.exception("Failed to inspect dynamic muscle-line render sources.")
        return []
    for source in sources:
        try:
            overlays.append(_WarpMuscleLineOverlay(source))
        except Exception:
            logger.exception("Failed to initialize Warp muscle-line overlay.")
    return overlays
