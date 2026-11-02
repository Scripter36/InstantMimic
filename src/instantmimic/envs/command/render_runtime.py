from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
from pymotion.rotations import quat_torch

from instantmimic.envs.command.render_assets import (
    BLENDER_TEMPLATE_COMMAND_GOAL_MARKER,
    COMMAND_VISUAL_GEOM_GROUP,
    append_point_goal_marker_render_assets,
    command_template_visual_spec,
    scale_mesh_vertices_along_geom_x,
    transform_mesh_vertices_to_geom_frame,
)
from instantmimic.envs.controls.fast_muscle.render_builder import (
    compile_render_root,
    configure_render_resolution,
    get_worldbody_or_raise,
    load_render_root,
)
from instantmimic.envs.tracking.metrics import get_heading_frame
from instantmimic.utils.project_paths import resolve_project_path

logger = logging.getLogger(__name__)


@dataclass
class ArrowPartRenderState:
    mocap_id: int | None = None
    bound_min_x: float = 0.0
    bound_max_x: float = 0.0
    bounds_initialized: bool = False
    local_position: np.ndarray | None = None
    local_quaternion: np.ndarray | None = None


@dataclass
class CommandRenderState:
    resolved_source_mjcf_path: str | None = None
    ghost_mocap_ids: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    ghost_sim_body_ids: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.int32)
    )
    ghost_offset: np.ndarray = field(
        default_factory=lambda: np.zeros((3,), dtype=np.float64)
    )
    arrow_parts: dict[str, ArrowPartRenderState] = field(
        default_factory=lambda: {
            name: ArrowPartRenderState() for name in ("tail", "body", "head")
        }
    )
    arrow_offset: np.ndarray = field(
        default_factory=lambda: np.zeros((3,), dtype=np.float64)
    )
    point_goal_marker_mocap_id: int | None = None
    point_goal_marker_offset: np.ndarray = field(
        default_factory=lambda: np.zeros((3,), dtype=np.float64)
    )
    arrow_body_mesh_vertex_slice: slice = field(default_factory=lambda: slice(0, 0))
    arrow_body_mesh_base_vertices: np.ndarray | None = None
    arrow_body_mesh_id: int | None = None
    arrow_body_mesh_position: np.ndarray | None = None
    arrow_body_mesh_quaternion: np.ndarray | None = None
    arrow_body_last_scale: float | None = None
    dirty_mesh_ids: set[int] = field(default_factory=set)


def initialize_render_state(env, cfg=None) -> CommandRenderState:
    cfg = env.cfg if cfg is None else cfg
    state = CommandRenderState(
        ghost_offset=np.array(
            getattr(cfg, "ghost_offset", [0.0, 0.0, 0.0]), dtype=np.float64
        ),
        arrow_offset=np.array(
            getattr(cfg, "velocity_command_arrow_offset", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ),
        point_goal_marker_offset=np.array(
            getattr(cfg, "point_goal_marker_offset", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ),
    )
    env.command_render_state = state
    return state


def render_env_index(env) -> int:
    num_envs = max(1, int(env._resolved_command_num_envs()))
    idx = int(getattr(env, "_render_capture_env_idx", 0))
    return max(0, min(idx, num_envs - 1))


def get_blender_reference_enabled(env) -> bool:
    return False


def get_blender_special_visual_specs(env) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    if env._point_goal_marker_enabled():
        specs.append(
            command_template_visual_spec(
                visual_name="command_point_goal_marker",
                template_object_name=BLENDER_TEMPLATE_COMMAND_GOAL_MARKER,
            )
        )
    if env.render_extra_visuals_enabled():
        template_object_name = env.render_extra_visual_template_name()
        specs.extend(
            [
                command_template_visual_spec(
                    visual_name=f"command_projectile_{slot_idx}",
                    template_object_name=template_object_name,
                )
                for slot_idx in range(env.render_extra_visual_count())
            ]
        )
    return specs


def get_blender_special_visual_state(env) -> dict[str, dict[str, np.ndarray]]:
    state: dict[str, dict[str, np.ndarray]] = {}
    env_idx = render_env_index(env)
    if env._velocity_command_arrow_enabled():
        state.update(compute_velocity_command_arrow_visual_state(env, env_idx))
    if env._point_goal_marker_enabled():
        state["command_point_goal_marker"] = compute_point_goal_marker_visual_state(
            env, env_idx
        )
    if env.render_extra_visuals_enabled():
        state.update(env.render_extra_visual_state(env_idx))
    return state


def get_blender_special_reduced_state(env) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    env_idx = render_env_index(env)
    if env._velocity_command_arrow_enabled():
        state.update(compute_velocity_command_arrow_reduced_state(env, env_idx))
    return state


def build_render_model(env):
    state = initialize_render_state(env)
    source_mjcf_path = env.resolve_render_source_mjcf_path()
    state.resolved_source_mjcf_path = source_mjcf_path
    root = load_render_root(source_mjcf_path)
    env.prepare_render_root(root)
    configure_render_resolution(root, tuple(env.cfg.viewer.resolution))
    worldbody = get_worldbody_or_raise(root, source_mjcf_path)

    env.append_render_worldbody_extras(worldbody)

    arrow_mesh_names: dict[str, str] = {}
    if env._velocity_command_arrow_enabled():
        asset_elem = root.find("asset")
        if asset_elem is None:
            asset_elem = ET.SubElement(root, "asset")
        scale = [
            float(v)
            for v in getattr(env.cfg, "velocity_command_arrow_scale", [1.0, 1.0, 1.0])
        ]
        scale_str = " ".join(str(v) for v in scale)
        for part in ("tail", "body", "head"):
            part_mesh_path = env._resolved_velocity_command_arrow_part_mesh_path(part)
            if part_mesh_path is None:
                raise ValueError(
                    f"velocity_command_arrow_{part}_mesh_path must be set."
                )
            if not resolve_project_path(part_mesh_path).is_file():
                raise ValueError(
                    f"velocity_command_arrow_{part}_mesh_path does not exist: "
                    f"{part_mesh_path}"
                )
            mesh_name = f"command_velocity_arrow_{part}_mesh"
            arrow_mesh_names[part] = mesh_name
            mesh_elem = ET.SubElement(asset_elem, "mesh")
            mesh_elem.set("name", mesh_name)
            mesh_elem.set("file", part_mesh_path)
            mesh_elem.set("scale", scale_str)

            arrow_body = ET.SubElement(worldbody, "body")
            arrow_body.set("name", f"command_velocity_arrow_{part}")
            arrow_body.set("mocap", "true")
            arrow_geom = ET.SubElement(arrow_body, "geom")
            arrow_geom.set("name", f"command_velocity_arrow_{part}_geom")
            arrow_geom.set("type", "mesh")
            arrow_geom.set("mesh", mesh_name)
            arrow_geom.set("group", COMMAND_VISUAL_GEOM_GROUP)
            arrow_geom.set("rgba", "0.15 0.95 0.35 0.90")
            arrow_geom.set("contype", "0")
            arrow_geom.set("conaffinity", "0")
            arrow_geom.set("mass", "0")

    point_goal_marker_mesh_name: str | None = None
    if env._point_goal_marker_enabled():
        asset_elem = root.find("asset")
        if asset_elem is None:
            asset_elem = ET.SubElement(root, "asset")
        marker_mesh_path = env._resolved_point_goal_marker_mesh_path()
        if marker_mesh_path is None:
            raise ValueError("point_goal_marker_mesh_path must be set.")
        marker_path = resolve_project_path(marker_mesh_path)
        if not marker_path.is_file():
            raise ValueError(
                f"point_goal_marker_mesh_path does not exist: {marker_mesh_path}"
            )
        if marker_path.suffix.lower() != ".obj":
            raise ValueError("point_goal_marker_mesh_path must point to an OBJ mesh.")
        marker_texture_path = env._resolved_point_goal_marker_texture_path()
        texture_path = None
        if marker_texture_path is not None:
            texture_path = resolve_project_path(marker_texture_path)
            if texture_path.suffix.lower() != ".png":
                raise ValueError(
                    "point_goal_marker_texture_path must point to a PNG texture."
                )
            if not texture_path.is_file():
                # The texture is optional; without it the marker uses
                # point_goal_marker_rgba.
                logger.warning(
                    "point_goal_marker_texture_path not found, falling back to "
                    "point_goal_marker_rgba: %s",
                    marker_texture_path,
                )
                marker_texture_path = None
                texture_path = None
        scale = [
            float(v)
            for v in getattr(env.cfg, "point_goal_marker_scale", [1.0, 1.0, 1.0])
        ]
        point_goal_marker_mesh_name = append_point_goal_marker_render_assets(
            asset_elem=asset_elem,
            worldbody=worldbody,
            mesh_path=str(marker_path),
            texture_path=(None if marker_texture_path is None else str(texture_path)),
            scale=scale,
            rgba=getattr(env.cfg, "point_goal_marker_rgba", None),
        )

    render_model, render_data = compile_render_root(
        root=root,
        sim_model=env.model,
        source_mjcf_path=source_mjcf_path,
    )

    if arrow_mesh_names:
        for part in ("tail", "body", "head"):
            part_state = state.arrow_parts[part]
            body_id = mujoco.mj_name2id(
                render_model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"command_velocity_arrow_{part}",
            )
            mocap_id = int(render_model.body_mocapid[body_id])
            part_state.mocap_id = mocap_id

            mesh_id = mujoco.mj_name2id(
                render_model,
                mujoco.mjtObj.mjOBJ_MESH,
                arrow_mesh_names[part],
            )
            vert_adr = int(render_model.mesh_vertadr[mesh_id])
            vert_num = int(render_model.mesh_vertnum[mesh_id])
            vert_slice = slice(vert_adr, vert_adr + vert_num)
            verts = np.array(render_model.mesh_vert[vert_slice], copy=True)
            mesh_pos = np.array(render_model.mesh_pos[mesh_id], copy=True)
            mesh_quat = np.array(render_model.mesh_quat[mesh_id], copy=True)
            verts_geom = transform_mesh_vertices_to_geom_frame(
                verts,
                mesh_pos,
                mesh_quat,
            )
            min_x = float(verts_geom[:, 0].min())
            max_x = float(verts_geom[:, 0].max())
            part_state.bound_min_x = min_x
            part_state.bound_max_x = max_x
            part_state.bounds_initialized = True
            geom_id = mujoco.mj_name2id(
                render_model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"command_velocity_arrow_{part}_geom",
            )
            part_state.local_position = np.array(
                render_model.geom_pos[geom_id], copy=True
            )
            part_state.local_quaternion = np.array(
                render_model.geom_quat[geom_id], copy=True
            )
            if part == "body":
                state.arrow_body_mesh_id = int(mesh_id)
                state.arrow_body_mesh_vertex_slice = vert_slice
                state.arrow_body_mesh_base_vertices = verts
                state.arrow_body_mesh_position = mesh_pos
                state.arrow_body_mesh_quaternion = mesh_quat

    if point_goal_marker_mesh_name is not None:
        body_id = mujoco.mj_name2id(
            render_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "command_point_goal_marker",
        )
        state.point_goal_marker_mocap_id = int(render_model.body_mocapid[body_id])

    return render_model, render_data


def update_render_data(env):
    if hasattr(env, "fast_muscle"):
        env.fast_muscle.update_render_tendon_rgba()
    tail_mocap_id = env.command_render_state.arrow_parts["tail"].mocap_id
    body_mocap_id = env.command_render_state.arrow_parts["body"].mocap_id
    head_mocap_id = env.command_render_state.arrow_parts["head"].mocap_id
    if (
        tail_mocap_id is not None
        and body_mocap_id is not None
        and head_mocap_id is not None
    ):
        update_velocity_command_arrow_render_data(env)
    point_goal_marker_mocap_id = env.command_render_state.point_goal_marker_mocap_id
    if point_goal_marker_mocap_id is not None:
        update_point_goal_marker_render_data(env)


def update_velocity_command_arrow_render_data(env) -> None:
    tail_mocap_id = env.command_render_state.arrow_parts["tail"].mocap_id
    body_mocap_id = env.command_render_state.arrow_parts["body"].mocap_id
    head_mocap_id = env.command_render_state.arrow_parts["head"].mocap_id
    if tail_mocap_id is None or body_mocap_id is None or head_mocap_id is None:
        return
    env_idx = render_env_index(env)

    render_data = getattr(env, "render_data", None)
    if (
        render_data is not None
        and render_data is not getattr(env, "data", None)
        and hasattr(render_data, "qpos")
    ):
        render_device = getattr(env, "device", torch.device("cpu"))
        root_pos = torch.as_tensor(
            np.asarray(render_data.qpos[:3], dtype=np.float32),
            device=render_device,
        ).unsqueeze(0)
        root_quat = torch.as_tensor(
            np.asarray(render_data.qpos[3:7], dtype=np.float32),
            device=render_device,
        ).unsqueeze(0)
    else:
        root_pos, root_quat = env._root_pos_quat()
    _, root_yaw_quat = get_heading_frame(root_pos, root_quat)
    root_pos_np = root_pos[0].detach().cpu().numpy()
    state = compute_velocity_command_arrow_visual_state(
        env,
        env_idx,
        root_pos_np=root_pos_np,
        root_yaw_quat=root_yaw_quat[:1],
    )
    tail_mocap_pos = state["command_velocity_arrow_tail"]["pos"]
    body_mocap_pos = state["command_velocity_arrow_body"]["pos"]
    head_mocap_pos = state["command_velocity_arrow_head"]["pos"]
    quat = state["command_velocity_arrow_tail"]["quat"]
    command_speed = float(state["command_velocity_arrow_body"]["scale"][0])
    body_min = env.command_render_state.arrow_parts["body"].bound_min_x

    env.render_data.mocap_pos[tail_mocap_id] = tail_mocap_pos
    env.render_data.mocap_pos[body_mocap_id] = body_mocap_pos
    env.render_data.mocap_pos[head_mocap_id] = head_mocap_pos
    env.render_data.mocap_quat[tail_mocap_id] = quat
    env.render_data.mocap_quat[body_mocap_id] = quat
    env.render_data.mocap_quat[head_mocap_id] = quat

    base_vertices = env.command_render_state.arrow_body_mesh_base_vertices
    mesh_pos = env.command_render_state.arrow_body_mesh_position
    mesh_quat = env.command_render_state.arrow_body_mesh_quaternion
    if base_vertices is None or mesh_pos is None or mesh_quat is None:
        return
    previous_scale = env.command_render_state.arrow_body_last_scale
    if previous_scale is None or abs(float(previous_scale) - command_speed) > 1.0e-6:
        updated_vertices = scale_mesh_vertices_along_geom_x(
            base_vertices,
            mesh_pos,
            mesh_quat,
            pivot_x=body_min,
            scale=command_speed,
        )
        env.render_model.mesh_vert[
            env.command_render_state.arrow_body_mesh_vertex_slice
        ] = updated_vertices.astype(env.render_model.mesh_vert.dtype, copy=False)
        env.command_render_state.arrow_body_last_scale = float(command_speed)
        body_mesh_id = env.command_render_state.arrow_body_mesh_id
        if body_mesh_id is not None:
            env.command_render_state.dirty_mesh_ids.add(int(body_mesh_id))


def update_point_goal_marker_render_data(env) -> None:
    marker_mocap_id = env.command_render_state.point_goal_marker_mocap_id
    if marker_mocap_id is None:
        return
    env_idx = render_env_index(env)
    state = compute_point_goal_marker_visual_state(env, env_idx)
    env.render_data.mocap_pos[marker_mocap_id] = state["pos"]
    env.render_data.mocap_quat[marker_mocap_id] = state["quat"]


def command_arrow_length(env, env_idx: int) -> float:
    """How long the velocity arrow should be drawn.

    Normally the commanded speed. A task whose velocity command carries only a
    heading has no magnitude to show, so its arrow is unit length -- FreeMusco's
    `velocity_heading_direction_only` is that case.
    """

    if env.velocity_command_arrow_is_unit_length():
        return 1.0
    return max(0.0, float(env._target_speed[env_idx, 0].detach().cpu().item()))


def compute_velocity_command_arrow_visual_state(
    env,
    env_idx: int,
    *,
    root_pos_np: np.ndarray | None = None,
    root_yaw_quat: torch.Tensor | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    env._ensure_velocity_command_arrow_visual_defaults()
    env_idx = int(env_idx)
    if root_pos_np is None or root_yaw_quat is None:
        root_pos, root_quat = env._root_pos_quat()
        _, resolved_root_yaw_quat = get_heading_frame(
            root_pos[env_idx : env_idx + 1],
            root_quat[env_idx : env_idx + 1],
        )
        root_pos_np = root_pos[env_idx].detach().cpu().numpy()
        root_yaw_quat = resolved_root_yaw_quat
    offset_xy_local = torch.tensor(
        [
            [
                float(env.command_render_state.arrow_offset[0]),
                float(env.command_render_state.arrow_offset[1]),
                0.0,
            ]
        ],
        device=root_yaw_quat.device,
        dtype=root_yaw_quat.dtype,
    )
    offset_xy_world = (
        quat_torch.mul_vec(root_yaw_quat[:1], offset_xy_local)[0].detach().cpu().numpy()
    )
    heading = float(env._target_heading[env_idx, 0].detach().cpu().item())
    command_speed = command_arrow_length(env, env_idx)
    quat = np.array(
        [math.cos(heading * 0.5), 0.0, 0.0, math.sin(heading * 0.5)],
        dtype=np.float64,
    )
    x_dir = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=np.float64)
    origin = np.array(root_pos_np, copy=True, dtype=np.float64)
    origin[:2] += offset_xy_world[:2]
    origin[2] = float(env.command_render_state.arrow_offset[2])

    tail = env.command_render_state.arrow_parts["tail"]
    body = env.command_render_state.arrow_parts["body"]
    head = env.command_render_state.arrow_parts["head"]
    tail_min = tail.bound_min_x
    tail_max = tail.bound_max_x
    body_min = body.bound_min_x
    body_max = body.bound_max_x
    head_min = head.bound_min_x
    base_body_length = body_max - body_min
    scaled_body_length = base_body_length * command_speed

    tail_pos = origin - tail_min * x_dir
    body_world_min = origin + (tail_max - tail_min) * x_dir
    body_pos = body_world_min - body_min * x_dir
    head_world_min = body_world_min + scaled_body_length * x_dir
    head_pos = head_world_min - head_min * x_dir

    return {
        "command_velocity_arrow_tail": {
            "pos": tail_pos,
            "quat": quat,
            "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        },
        "command_velocity_arrow_body": {
            "pos": body_pos,
            "quat": quat,
            "scale": np.array([command_speed, 1.0, 1.0], dtype=np.float64),
        },
        "command_velocity_arrow_head": {
            "pos": head_pos,
            "quat": quat,
            "scale": np.array([1.0, 1.0, 1.0], dtype=np.float64),
        },
    }


def compute_velocity_command_arrow_reduced_state(
    env,
    env_idx: int,
    *,
    root_pos_np: np.ndarray | None = None,
    root_yaw_quat: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    env._ensure_velocity_command_arrow_visual_defaults()
    env_idx = int(env_idx)
    if root_pos_np is None or root_yaw_quat is None:
        root_pos, root_quat = env._root_pos_quat()
        _, resolved_root_yaw_quat = get_heading_frame(
            root_pos[env_idx : env_idx + 1],
            root_quat[env_idx : env_idx + 1],
        )
        root_pos_np = root_pos[env_idx].detach().cpu().numpy()
        root_yaw_quat = resolved_root_yaw_quat

    offset_xy_local = torch.tensor(
        [
            [
                float(env.command_render_state.arrow_offset[0]),
                float(env.command_render_state.arrow_offset[1]),
                0.0,
            ]
        ],
        device=root_yaw_quat.device,
        dtype=root_yaw_quat.dtype,
    )
    offset_xy_world = (
        quat_torch.mul_vec(root_yaw_quat[:1], offset_xy_local)[0].detach().cpu().numpy()
    )
    heading = float(env._target_heading[env_idx, 0].detach().cpu().item())
    command_speed = command_arrow_length(env, env_idx)

    origin = np.array(root_pos_np, copy=True, dtype=np.float64)
    origin[:2] += offset_xy_world[:2]
    origin[2] = float(env.command_render_state.arrow_offset[2])

    # Blender templates use +Y as the arrow-forward axis in local space.
    blender_heading = heading - (math.pi * 0.5)
    quat = np.array(
        [
            math.cos(blender_heading * 0.5),
            0.0,
            0.0,
            math.sin(blender_heading * 0.5),
        ],
        dtype=np.float32,
    )

    return {
        "command_velocity_arrow_origin": origin.astype(np.float32, copy=False),
        "command_velocity_arrow_quat": quat,
        "command_velocity_arrow_body_scale": np.asarray(
            command_speed,
            dtype=np.float32,
        ),
    }


def compute_point_goal_marker_visual_state(
    env,
    env_idx: int,
) -> dict[str, np.ndarray]:
    goal_xy = env._goal_pos_xy[int(env_idx)].detach().cpu().numpy()
    offset = env.command_render_state.point_goal_marker_offset
    return {
        "pos": np.array(
            [
                float(goal_xy[0]) + float(offset[0]),
                float(goal_xy[1]) + float(offset[1]),
                float(offset[2]),
            ],
            dtype=np.float64,
        ),
        "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "scale": np.array(
            getattr(env.cfg, "point_goal_marker_scale", [1.0, 1.0, 1.0]),
            dtype=np.float64,
        ),
    }


def upload_dynamic_render_meshes(env, *, mjr_context) -> None:
    if mjr_context is None:
        return
    render_model = getattr(env, "render_model", None)
    if render_model is None:
        return
    dirty_ids = sorted(
        int(mesh_id) for mesh_id in env.command_render_state.dirty_mesh_ids
    )
    env.command_render_state.dirty_mesh_ids.clear()
    for mesh_id in dirty_ids:
        mujoco.mjr_uploadMesh(render_model, mjr_context, mesh_id)
