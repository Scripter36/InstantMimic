from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from pymotion.rotations import quat_torch

from instantmimic.envs.config_access import get_value as _cfg_get
from instantmimic.envs.controls.fast_muscle.render_builder import (
    append_ghost_mocap_subtree,
    configure_render_resolution,
    load_render_root,
    resolve_ghost_binding,
)

__all__ = (
    "ReferenceGhostRenderBundle",
    "resolve_reference_ghost_pose",
    "build_render_model",
    "build_reference_ghost_render_bundle",
    "append_dynamic_render_geoms",
    "dynamic_gl_render_sources",
    "dynamic_render_maxgeom",
    "set_reference_ghost_pose_on_render_data",
    "update_render_data",
)


@dataclass(frozen=True)
class ReferenceGhostRenderBundle:
    model: mujoco.MjModel
    data: mujoco.MjData
    ghost_mocap_ids: np.ndarray
    ghost_sim_body_ids: np.ndarray
    ghost_offset: np.ndarray


def resolve_reference_ghost_pose(
    *,
    ref_idx: int,
    motion_frame_count: int,
    motion_xpos: torch.Tensor,
    motion_xquat: torch.Tensor,
    loop_root_quat: torch.Tensor,
    loop_root_pos: torch.Tensor,
    motion_clip_idx: int | None = None,
    clip_start: torch.Tensor | None = None,
    clip_length: torch.Tensor | None = None,
    clip_loop_root_quat: torch.Tensor | None = None,
    clip_loop_root_pos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if motion_clip_idx is None:
        local_idx = int(ref_idx) % int(motion_frame_count)
        cycles = int(ref_idx) // int(motion_frame_count)
        frame_idx = local_idx
        loop_root_quat_t = loop_root_quat
        loop_root_pos_t = loop_root_pos
    else:
        if (
            clip_start is None
            or clip_length is None
            or clip_loop_root_quat is None
            or clip_loop_root_pos is None
        ):
            raise ValueError(
                "Multi-motion ghost pose resolution requires clip metadata tensors."
            )
        clip_idx = int(motion_clip_idx)
        clip_start_cpu = clip_start.detach().cpu()
        clip_length_cpu = clip_length.detach().cpu()
        local_idx = int(ref_idx) % int(clip_length_cpu[clip_idx].item())
        cycles = int(ref_idx) // int(clip_length_cpu[clip_idx].item())
        frame_idx = int(clip_start_cpu[clip_idx].item()) + local_idx
        loop_root_quat_t = clip_loop_root_quat[clip_idx]
        loop_root_pos_t = clip_loop_root_pos[clip_idx]

    xpos_t = motion_xpos[frame_idx].clone()
    xquat_t = motion_xquat[frame_idx].clone()
    if cycles > 0:
        loop_q = loop_root_quat_t[None, :].expand(xquat_t.shape[0], -1)
        loop_t = loop_root_pos_t[None, :].expand(xpos_t.shape[0], -1)
        for _ in range(cycles):
            xpos_t = quat_torch.mul_vec(loop_q, xpos_t) + loop_t
            xquat_t = quat_torch.mul(loop_q, xquat_t)
    return xpos_t, xquat_t


def build_render_model(self):
    """Build render model with transparent mocap ghost bodies for reference visualization."""
    control_render_builder = getattr(self._control_pipeline, "build_render_model", None)
    if callable(control_render_builder):
        render_result = control_render_builder(self)
        if render_result is not None:
            return render_result
    ghost_bundle = build_reference_ghost_render_bundle(self)
    self._ghost_mocap_ids = ghost_bundle.ghost_mocap_ids
    self._ghost_sim_body_ids = ghost_bundle.ghost_sim_body_ids
    self._ghost_offset_np = ghost_bundle.ghost_offset
    return ghost_bundle.model, ghost_bundle.data


def _as_name_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(name) for name in value]


def _resolve_reference_ghost_root_candidates(self) -> list[str]:
    candidates = _as_name_list(_cfg_get(self.cfg, "root_body_names", None))
    bvh_map = self.motion_settings.bvh_map
    if isinstance(bvh_map, dict):
        hips_body_name = bvh_map.get("Hips")
        if hips_body_name is not None:
            candidates.append(str(hips_body_name))
    candidates.extend(["Pelvis", "pelvis"])

    unique_candidates: list[str] = []
    for name in candidates:
        if name not in unique_candidates:
            unique_candidates.append(name)
    return unique_candidates


def build_reference_ghost_render_bundle(self):
    """Build an isolated render model/data pair with reference ghost bodies."""
    mjcf_path = self._resolve_cfg_path_in_place(self.cfg.sim, "mjcf_path")
    root = load_render_root(mjcf_path)
    configure_render_resolution(
        root,
        (int(self.cfg.viewer.resolution[0]), int(self.cfg.viewer.resolution[1])),
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody not found in render MJCF: {mjcf_path}")

    def _find_body_elem(candidates: list[str]) -> ET.Element:
        for name in candidates:
            body_elem = worldbody.find(f".//body[@name='{name}']")
            if body_elem is not None:
                return body_elem
        raise ValueError(
            f"None of render-body names were found in MJCF: {candidates} ({mjcf_path})"
        )

    pelvis = _find_body_elem(_resolve_reference_ghost_root_candidates(self))

    # Collect body names and add mocap bodies with transparent ghost geoms.
    # Include both visual meshes and primitive collision boxes so reference
    # body boxes are visible during playback/video capture.
    ghost_body_names = append_ghost_mocap_subtree(
        worldbody=worldbody,
        source_body=pelvis,
    )

    # Compile render model
    xml_string = ET.tostring(root, encoding="unicode")
    render_spec = mujoco.MjSpec.from_string(xml_string)
    render_model = mujoco.MjSpec.compile(render_spec)
    render_data = mujoco.MjData(render_model)

    # Build mocap ID <-> sim body ID mapping
    ghost_binding = resolve_ghost_binding(
        render_model=render_model,
        sim_model=self.model,
        ghost_body_names=ghost_body_names,
        ghost_offset=self.cfg.ghost_offset,
    )

    return ReferenceGhostRenderBundle(
        model=render_model,
        data=render_data,
        ghost_mocap_ids=ghost_binding.ghost_mocap_ids,
        ghost_sim_body_ids=ghost_binding.ghost_sim_body_ids,
        ghost_offset=ghost_binding.ghost_offset_np,
    )


def set_reference_ghost_pose_on_render_data(
    self,
    *,
    render_data: mujoco.MjData,
    ghost_mocap_ids: np.ndarray,
    ghost_sim_body_ids: np.ndarray,
    ghost_offset_np: np.ndarray,
    ref_idx: int,
    motion_clip_idx: int | None,
) -> None:
    if self._multi_motion_enabled:
        xpos_t, xquat_t = resolve_reference_ghost_pose(
            ref_idx=ref_idx,
            motion_frame_count=int(self.motion_frame_count),
            motion_xpos=self.motion_xpos,
            motion_xquat=self.motion_xquat,
            loop_root_quat=self.loop_root_quat,
            loop_root_pos=self.loop_root_pos,
            motion_clip_idx=int(motion_clip_idx),
            clip_start=self.motion_clip_start,
            clip_length=self.motion_clip_length,
            clip_loop_root_quat=self.motion_clip_loop_root_quat,
            clip_loop_root_pos=self.motion_clip_loop_root_pos,
        )
    else:
        xpos_t, xquat_t = resolve_reference_ghost_pose(
            ref_idx=ref_idx,
            motion_frame_count=int(self.motion_frame_count),
            motion_xpos=self.motion_xpos,
            motion_xquat=self.motion_xquat,
            loop_root_quat=self.loop_root_quat,
            loop_root_pos=self.loop_root_pos,
        )

    xpos = xpos_t.cpu().numpy()
    xquat = xquat_t.cpu().numpy()

    render_data.mocap_pos[ghost_mocap_ids] = (
        xpos[ghost_sim_body_ids, :3] + ghost_offset_np
    )
    render_data.mocap_quat[ghost_mocap_ids] = xquat[ghost_sim_body_ids]


def update_render_data(self):
    """Set ghost mocap bodies to reference motion state with offset."""
    set_reference_ghost_pose_on_render_data(
        self,
        render_data=self.render_data,
        ghost_mocap_ids=self._ghost_mocap_ids,
        ghost_sim_body_ids=self._ghost_sim_body_ids,
        ghost_offset_np=self._ghost_offset_np,
        ref_idx=int(self.ref_frame_idx[0].item()),
        motion_clip_idx=(
            int(self.motion_clip_idx[0].item()) if self._multi_motion_enabled else None
        ),
    )
    control_render_updater = getattr(self._control_pipeline, "update_render_data", None)
    if callable(control_render_updater):
        control_render_updater()


def append_dynamic_render_geoms(self, *, scene, render_model, render_data):
    del render_model
    append_dynamic_render_geoms = getattr(
        self._control_pipeline,
        "append_dynamic_render_geoms",
        None,
    )
    if callable(append_dynamic_render_geoms):
        append_dynamic_render_geoms(scene=scene, render_data=render_data)


def dynamic_gl_render_sources(self) -> list[object]:
    dynamic_gl_render_sources = getattr(
        self._control_pipeline,
        "dynamic_gl_render_sources",
        None,
    )
    if not callable(dynamic_gl_render_sources):
        return []
    return list(dynamic_gl_render_sources())


def dynamic_render_maxgeom(self) -> int:
    dynamic_render_maxgeom = getattr(
        self._control_pipeline,
        "dynamic_render_maxgeom",
        None,
    )
    if not callable(dynamic_render_maxgeom):
        return 0
    return int(dynamic_render_maxgeom())
