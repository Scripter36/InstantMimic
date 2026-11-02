from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from instantmimic.muscle.parsing.mjcf import (
    absolutize_mjcf_asset_paths,
    collect_referenced_site_names,
    load_expanded_mjcf_root,
    remove_all_actuator_elements,
    remove_all_tendon_elements,
    remove_unreferenced_named_sites,
)


@dataclass(slots=True)
class GhostRenderBinding:
    ghost_mocap_ids: np.ndarray
    ghost_sim_body_ids: np.ndarray
    ghost_offset_np: np.ndarray


@dataclass(slots=True)
class RenderBuildResult:
    render_model: mujoco.MjModel
    render_data: mujoco.MjData
    ghost_binding: GhostRenderBinding | None = None


def load_render_root(source_mjcf_path: str) -> ET.Element:
    root, _ = load_expanded_mjcf_root(source_mjcf_path)
    ensure_grid_floor_material(root)
    ensure_playback_camera_element(root)
    return root


def ensure_playback_camera_element(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None or worldbody.find(".//camera") is not None:
        return
    visual_global = root.find("visual/global")
    fovy = "45" if visual_global is None else visual_global.get("fovy", "45")
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "__instantmimic_playback_camera",
            "pos": "0 0 0",
            "fovy": str(fovy),
        },
    )


def get_worldbody_or_raise(root: ET.Element, source_mjcf_path: str) -> ET.Element:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody not found in source MJCF: {source_mjcf_path}")
    return worldbody


def find_body_or_raise(
    worldbody: ET.Element,
    *,
    xpath: str,
    source_mjcf_path: str,
    description: str,
) -> ET.Element:
    body = worldbody.find(xpath)
    if body is None:
        raise ValueError(f"{description} not found in source MJCF: {source_mjcf_path}")
    return body


def strip_render_actuators(root: ET.Element) -> None:
    remove_all_actuator_elements(root)


def strip_render_muscle_artifacts(root: ET.Element) -> None:
    strip_render_actuators(root)
    remove_all_tendon_elements(root)
    remove_unreferenced_named_sites(
        root,
        referenced_site_names=collect_referenced_site_names(root),
    )


def configure_render_resolution(root: ET.Element, resolution: tuple[int, int]) -> None:
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_elem = visual.find("global")
    if global_elem is None:
        global_elem = ET.SubElement(visual, "global")
    global_elem.set("offwidth", str(int(resolution[0])))
    global_elem.set("offheight", str(int(resolution[1])))


def ensure_grid_floor_material(root: ET.Element) -> None:
    floor_geom = root.find(".//geom[@name='floor']")
    if floor_geom is None:
        return

    assets = root.findall("asset")
    if not assets:
        assets = [ET.SubElement(root, "asset")]
    asset = assets[0]
    for asset_section in assets:
        for tex in list(asset_section.findall("texture[@name='grid']")):
            asset_section.remove(tex)
        for mat in list(asset_section.findall("material[@name='grid']")):
            asset_section.remove(mat)
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "grid",
            "type": "2d",
            "builtin": "checker",
            "width": "512",
            "height": "512",
            "rgb1": ".1 .2 .3",
            "rgb2": ".2 .3 .4",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "grid",
            "texture": "grid",
            "texrepeat": "1 1",
            "texuniform": "true",
            "reflectance": ".2",
        },
    )
    floor_geom.set("material", "grid")


def compile_render_root(
    *,
    root: ET.Element,
    sim_model: mujoco.MjModel,
    source_mjcf_path: str,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    absolutize_mjcf_asset_paths(
        root,
        Path(source_mjcf_path).expanduser().resolve(),
    )
    xml_string = ET.tostring(root, encoding="unicode")
    render_spec = mujoco.MjSpec.from_string(xml_string)
    render_model = mujoco.MjSpec.compile(render_spec)
    render_data = mujoco.MjData(render_model)

    if render_model.nq != sim_model.nq or render_model.nv != sim_model.nv:
        raise ValueError(
            "Render MJCF must match simulation kinematics "
            f"(sim nq/nv={sim_model.nq}/{sim_model.nv}, "
            f"render nq/nv={render_model.nq}/{render_model.nv})."
        )
    return render_model, render_data


def build_plain_render_model(
    *,
    source_mjcf_path: str,
    sim_model: mujoco.MjModel,
    resolution: tuple[int, int],
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    root, _ = load_expanded_mjcf_root(source_mjcf_path)
    ensure_playback_camera_element(root)
    configure_render_resolution(
        root,
        (
            max(int(sim_model.vis.global_.offwidth), int(resolution[0])),
            max(int(sim_model.vis.global_.offheight), int(resolution[1])),
        ),
    )
    return compile_render_root(
        root=root,
        sim_model=sim_model,
        source_mjcf_path=source_mjcf_path,
    )


def append_ghost_mocap_subtree(
    *,
    worldbody: ET.Element,
    source_body: ET.Element,
    mesh_rgba: str = "1 0.3 0.3 0.20",
    primitive_rgba: str = "1 0.15 0.15 0.45",
) -> list[str]:
    ghost_body_names: list[str] = []

    def _add_mocap_body(body_elem: ET.Element) -> None:
        name = body_elem.get("name")
        if name is None:
            return
        ghost_body_names.append(name)

        mocap = ET.SubElement(worldbody, "body")
        mocap.set("name", f"ghost_{name}")
        mocap.set("mocap", "true")

        for geom in body_elem.findall("geom"):
            ghost_geom = copy.deepcopy(geom)
            geom_name = geom.get("name")
            if geom_name is not None:
                ghost_geom.set("name", f"ghost_{geom_name}")
            ghost_geom.set(
                "rgba",
                mesh_rgba if geom.get("mesh") is not None else primitive_rgba,
            )
            ghost_geom.set("contype", "0")
            ghost_geom.set("conaffinity", "0")
            ghost_geom.set("mass", "0")
            mocap.append(ghost_geom)

        for child in body_elem.findall("body"):
            _add_mocap_body(child)

    _add_mocap_body(source_body)
    return ghost_body_names


def resolve_ghost_binding(
    *,
    render_model: mujoco.MjModel,
    sim_model: mujoco.MjModel,
    ghost_body_names: list[str],
    ghost_offset: list[float] | tuple[float, float, float],
    ghost_offset_dtype: object = np.float64,
    sim_body_resolver: Callable[[str], int] | None = None,
) -> GhostRenderBinding:
    if sim_body_resolver is None:
        sim_body_resolver = lambda name: int(
            mujoco.mj_name2id(sim_model, mujoco.mjtObj.mjOBJ_BODY, name)
        )

    ghost_mocap_ids: list[int] = []
    ghost_sim_body_ids: list[int] = []
    for name in ghost_body_names:
        ghost_body_id = int(
            mujoco.mj_name2id(
                render_model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"ghost_{name}",
            )
        )
        if ghost_body_id < 0:
            raise ValueError(f"Ghost body 'ghost_{name}' not found in render model.")

        mocap_id = int(render_model.body_mocapid[ghost_body_id])
        if mocap_id < 0:
            raise ValueError(f"Ghost body 'ghost_{name}' is not a mocap body.")

        sim_body_id = int(sim_body_resolver(name))
        if sim_body_id < 0:
            raise ValueError(
                f"Body '{name}' required for ghost rendering not found in sim model."
            )

        ghost_mocap_ids.append(mocap_id)
        ghost_sim_body_ids.append(sim_body_id)

    return GhostRenderBinding(
        ghost_mocap_ids=np.asarray(ghost_mocap_ids, dtype=np.int32),
        ghost_sim_body_ids=np.asarray(ghost_sim_body_ids, dtype=np.int32),
        ghost_offset_np=np.asarray(ghost_offset, dtype=ghost_offset_dtype),
    )


def build_ghost_render_model(
    *,
    source_mjcf_path: str,
    sim_model: mujoco.MjModel,
    resolution: tuple[int, int],
    ghost_root_xpath: str,
    ghost_root_description: str,
    ghost_offset: list[float] | tuple[float, float, float],
    ghost_offset_dtype: object = np.float64,
    sim_body_resolver: Callable[[str], int] | None = None,
) -> RenderBuildResult:
    root = load_render_root(source_mjcf_path)
    strip_render_muscle_artifacts(root)
    configure_render_resolution(root, resolution)
    worldbody = get_worldbody_or_raise(root, source_mjcf_path)
    ghost_root = find_body_or_raise(
        worldbody,
        xpath=ghost_root_xpath,
        source_mjcf_path=source_mjcf_path,
        description=ghost_root_description,
    )
    ghost_body_names = append_ghost_mocap_subtree(
        worldbody=worldbody,
        source_body=ghost_root,
    )
    render_model, render_data = compile_render_root(
        root=root,
        sim_model=sim_model,
        source_mjcf_path=source_mjcf_path,
    )
    ghost_binding = resolve_ghost_binding(
        render_model=render_model,
        sim_model=sim_model,
        ghost_body_names=ghost_body_names,
        ghost_offset=ghost_offset,
        ghost_offset_dtype=ghost_offset_dtype,
        sim_body_resolver=sim_body_resolver,
    )
    return RenderBuildResult(
        render_model=render_model,
        render_data=render_data,
        ghost_binding=ghost_binding,
    )
