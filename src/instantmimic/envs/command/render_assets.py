from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

BLENDER_TEMPLATE_COMMAND_GOAL_MARKER = "COMMAND_GOAL_MARKER_TEMPLATE"
BLENDER_TEMPLATE_COMMAND_VELOCITY_ARROW_TAIL = "COMMAND_VELOCITY_ARROW_TAIL_TEMPLATE"
BLENDER_TEMPLATE_COMMAND_VELOCITY_ARROW_BODY = "COMMAND_VELOCITY_ARROW_BODY_TEMPLATE"
BLENDER_TEMPLATE_COMMAND_VELOCITY_ARROW_HEAD = "COMMAND_VELOCITY_ARROW_HEAD_TEMPLATE"
BLENDER_TEMPLATE_COMMAND_PROJECTILE_BALL = "COMMAND_PROJECTILE_BALL_TEMPLATE"
BLENDER_TEMPLATE_COMMAND_PROJECTILE_BOX = "COMMAND_PROJECTILE_BOX_TEMPLATE"
COMMAND_VISUAL_GEOM_GROUP = "4"


def quat_to_rotmat_np(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def transform_mesh_vertices_to_geom_frame(
    vertices_mesh_frame: np.ndarray,
    mesh_pos: np.ndarray,
    mesh_quat: np.ndarray,
) -> np.ndarray:
    rot = quat_to_rotmat_np(mesh_quat)
    verts = np.asarray(vertices_mesh_frame, dtype=np.float64)
    pos = np.asarray(mesh_pos, dtype=np.float64)
    return verts @ rot.T + pos


def transform_geom_vertices_to_mesh_frame(
    vertices_geom_frame: np.ndarray,
    mesh_pos: np.ndarray,
    mesh_quat: np.ndarray,
) -> np.ndarray:
    rot = quat_to_rotmat_np(mesh_quat)
    verts = np.asarray(vertices_geom_frame, dtype=np.float64)
    pos = np.asarray(mesh_pos, dtype=np.float64)
    return (verts - pos) @ rot


def scale_mesh_vertices_along_geom_x(
    vertices_mesh_frame: np.ndarray,
    mesh_pos: np.ndarray,
    mesh_quat: np.ndarray,
    *,
    pivot_x: float,
    scale: float,
) -> np.ndarray:
    verts_geom = transform_mesh_vertices_to_geom_frame(
        vertices_mesh_frame,
        mesh_pos,
        mesh_quat,
    )
    verts_geom = np.array(verts_geom, copy=True)
    scale = max(float(scale), 0.0)
    verts_geom[:, 0] = float(pivot_x) + (verts_geom[:, 0] - float(pivot_x)) * scale
    return transform_geom_vertices_to_mesh_frame(verts_geom, mesh_pos, mesh_quat)


def load_wavefront_vertices_np(mesh_path: str) -> np.ndarray:
    vertices: list[list[float]] = []
    with open(mesh_path, encoding="utf-8") as mesh_file:
        for line in mesh_file:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"OBJ mesh contains no vertices: {mesh_path}")
    return np.asarray(vertices, dtype=np.float64)


def append_point_goal_marker_render_assets(
    *,
    asset_elem: ET.Element,
    worldbody: ET.Element,
    mesh_path: str,
    texture_path: str | None,
    scale: list[float],
    rgba: list[float] | None,
) -> str:
    mesh_name = "command_point_goal_marker_mesh"
    mesh_elem = ET.SubElement(asset_elem, "mesh")
    mesh_elem.set("name", mesh_name)
    mesh_elem.set("file", mesh_path)
    mesh_elem.set("scale", " ".join(str(float(v)) for v in scale))

    material_name: str | None = None
    if texture_path is not None:
        texture_name = "command_point_goal_marker_texture"
        material_name = "command_point_goal_marker_material"
        texture_elem = ET.SubElement(asset_elem, "texture")
        texture_elem.set("name", texture_name)
        texture_elem.set("type", "2d")
        texture_elem.set("file", texture_path)

        material_elem = ET.SubElement(asset_elem, "material")
        material_elem.set("name", material_name)
        material_elem.set("texture", texture_name)
        material_elem.set("texuniform", "false")

    marker_body = ET.SubElement(worldbody, "body")
    marker_body.set("name", "command_point_goal_marker")
    marker_body.set("mocap", "true")
    marker_geom = ET.SubElement(marker_body, "geom")
    marker_geom.set("name", "command_point_goal_marker_geom")
    marker_geom.set("type", "mesh")
    marker_geom.set("mesh", mesh_name)
    marker_geom.set("group", COMMAND_VISUAL_GEOM_GROUP)
    if material_name is not None:
        marker_geom.set("material", material_name)
    elif rgba is not None:
        marker_geom.set("rgba", " ".join(str(float(v)) for v in rgba))
    marker_geom.set("contype", "0")
    marker_geom.set("conaffinity", "0")
    marker_geom.set("mass", "0")
    return mesh_name


def command_template_visual_spec(
    *,
    visual_name: str,
    template_object_name: str,
) -> dict[str, object]:
    return {
        "visual_name": visual_name,
        "geom_name": visual_name,
        "body_name": "",
        "body_index": -1,
        "geom_type": "template_object",
        "visual_type": "template_object",
        "template_object_name": template_object_name,
        "local_pos": (0.0, 0.0, 0.0),
        "local_quat": (1.0, 0.0, 0.0, 0.0),
        "size": (1.0, 1.0, 1.0),
        "rgba": None,
        "mesh_name": None,
        "mesh_asset_source": None,
        "mesh_scale": None,
    }
