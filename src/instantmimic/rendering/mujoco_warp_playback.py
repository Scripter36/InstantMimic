from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjwr
import numpy as np
import torch
import warp as wp

from instantmimic.rendering.character import (
    apply_character_render_mode_to_model,
    normalize_character_geom_groups,
)
from instantmimic.rendering.muscle_overlay import (
    _resolve_dynamic_muscle_line_overlays,
    _resolve_lbs_activation_color_settings,
    _resolve_lbs_mesh_bundle_path,
    _resolve_lbs_mesh_mode,
    _resolve_lbs_mesh_shading_mode,
    _WarpLBSMeshOverlay,
)
from instantmimic.rendering.render_model import clone_playback_render_model
from instantmimic.utils.mjw_kernels.private import mul_quat, rot_vec_quat, smooth
from instantmimic.utils.mjw_model import put_model
from instantmimic.utils.torch_warp_capture import warp_stream_from_torch

logger = logging.getLogger(__name__)

_MUJOCO_GL_CLEAR_RGBA = (13.0 / 255.0, 17.0 / 255.0, 23.0 / 255.0, 1.0)
_REQUIRED_MUJOCO_WARP_RENDERER_CALLABLES = (
    "create_render_context",
    "get_rgb",
    "put_data",
    "put_model",
    "refit_bvh",
    "render",
)
_ACTIVATION_POSITIVE_LOW = np.asarray([1.0, 0.38, 0.12], dtype=np.float32)
_ACTIVATION_POSITIVE_MID = np.asarray([1.0, 0.68, 0.05], dtype=np.float32)
_ACTIVATION_POSITIVE_HIGH = np.asarray([1.0, 0.04, 0.01], dtype=np.float32)
_ACTIVATION_NEGATIVE_LOW = np.asarray([0.12, 0.45, 1.0], dtype=np.float32)
_ACTIVATION_NEGATIVE_MID = np.asarray([0.2, 0.85, 1.0], dtype=np.float32)
_ACTIVATION_NEGATIVE_HIGH = np.asarray([0.02, 0.08, 1.0], dtype=np.float32)
_ACTIVATION_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _time_cuda_stage_ms(
    stage_timings_ms: dict[str, float],
    name: str,
    device: torch.device | None,
    callback: Any,
) -> Any:
    if device is None or device.type != "cuda":
        return callback()
    torch.cuda.synchronize(device)
    start_ns = time.perf_counter_ns()
    result = callback()
    torch.cuda.synchronize(device)
    end_ns = time.perf_counter_ns()
    stage_timings_ms[name] = max(0.0, float(end_ns - start_ns) / 1_000_000.0)
    return result


@dataclass
class MujocoWarpRenderBatch:
    qpos: np.ndarray | torch.Tensor
    qvel: np.ndarray | torch.Tensor
    activation: np.ndarray | torch.Tensor | None = None
    mocap_pos: np.ndarray | torch.Tensor | None = None
    mocap_quat: np.ndarray | torch.Tensor | None = None


@dataclass
class _CudaRenderGraphEntry:
    graph: Any
    signature: tuple[Any, ...]
    qpos: torch.Tensor
    qvel: torch.Tensor
    mocap_pos: torch.Tensor | None
    mocap_quat: torch.Tensor | None


def mujoco_warp_playback_is_available() -> bool:
    """Return whether upstream MuJoCo-Warp has its required renderer API.

    Import failures are handled by the lazy caller in ``play.render``. Keeping
    this check to attribute lookups avoids device discovery or CUDA context
    creation on the normal MuJoCo GL path.
    """
    return all(
        callable(getattr(mjwr, name, None))
        for name in _REQUIRED_MUJOCO_WARP_RENDERER_CALLABLES
    )


def _camera_name(model: mujoco.MjModel, camera_id: int) -> str | None:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, int(camera_id))
    return str(name) if name is not None else None


def _model_has_skybox(model: object) -> bool:
    tex_type = getattr(model, "tex_type", None)
    if tex_type is None or int(getattr(model, "ntex", 0)) <= 0:
        return False
    return bool(np.any(np.asarray(tex_type) == mujoco.mjtTexture.mjTEXTURE_SKYBOX))


def _require_cuda_render_tensor(name: str, value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise RuntimeError(
            "MuJoCo-Warp pynvcodec playback requires qpos/qvel torch tensors; "
            "CPU numpy staging is not supported on the CUDA video path."
        )
    if torch.cuda.is_available() and not value.is_cuda:
        raise RuntimeError(
            "MuJoCo-Warp pynvcodec playback requires qpos/qvel CUDA tensors; "
            f"{name} is on {value.device}."
        )
    return value


def _resolve_floor_shader_mode(mode: str | None) -> str:
    value = "texture" if mode is None else str(mode).strip().lower()
    if value == "texture":
        return value
    if value == "procedural_grid":
        raise ValueError(
            "video_warp_floor_shader_mode='procedural_grid' is no longer supported; "
            "use 'texture'."
        )
    raise ValueError("video_warp_floor_shader_mode must be 'texture'.")


def _apply_render_model_character_mode(
    model: mujoco.MjModel,
    mode: str,
    *,
    env_raw: Any | None = None,
) -> None:
    apply_character_render_mode_to_model(model, mode, env_raw=env_raw)


def _resolve_warp_camera_id(
    model: mujoco.MjModel,
    camera: str | int | mujoco.MjvCamera | None,
) -> int:
    if int(model.ncam) <= 0:
        if camera is None or isinstance(camera, mujoco.MjvCamera):
            return 0
        raise RuntimeError(
            "Named or integer MuJoCo-Warp playback cameras require at least one "
            "camera in the render model."
        )

    if camera is None:
        return 0

    if isinstance(camera, str):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if camera_id < 0:
            raise ValueError(f'Camera "{camera}" not found in render model.')
        return int(camera_id)

    if isinstance(camera, mujoco.MjvCamera):
        is_fixed = camera.type in (
            mujoco.mjtCamera.mjCAMERA_FIXED,
            int(mujoco.mjtCamera.mjCAMERA_FIXED),
        )
        fixedcamid = int(getattr(camera, "fixedcamid", -1))
        if is_fixed and 0 <= fixedcamid < int(model.ncam):
            return fixedcamid
        logger.info(
            "MuJoCo-Warp playback will render through camera slot 0 (%s) "
            "and override its pose from the MjvCamera.",
            _camera_name(model, 0),
        )
        return 0

    camera_id = int(camera)
    if camera_id < 0 or camera_id >= int(model.ncam):
        raise ValueError(
            f"Camera id {camera_id} is outside render model range [0, {model.ncam})."
        )
    return camera_id


def _mjv_camera_pose(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    scene: mujoco.MjvScene,
    option: mujoco.MjvOption,
    perturb: mujoco.MjvPerturb,
) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mjv_updateScene(
        model,
        data,
        option,
        perturb,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    gl_camera = scene.camera[0]
    pos = np.asarray(gl_camera.pos, dtype=np.float32).copy()
    forward = np.asarray(gl_camera.forward, dtype=np.float32)
    up = np.asarray(gl_camera.up, dtype=np.float32)

    backward = -forward
    backward /= np.linalg.norm(backward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(backward, right)
    true_up /= np.linalg.norm(true_up)
    xmat = np.column_stack([right, true_up, backward]).astype(np.float32)
    return pos, xmat


def _copy_camera_pose_to_warp_data(
    warp_data: Any,
    *,
    camera_id: int,
    pos: np.ndarray,
    xmat: np.ndarray,
) -> None:
    cam_xpos = warp_data.cam_xpos.numpy()
    cam_xmat = warp_data.cam_xmat.numpy()
    cam_xpos[:, int(camera_id), :] = np.asarray(pos, dtype=np.float32)
    cam_xmat[:, int(camera_id), :, :] = np.asarray(xmat, dtype=np.float32)
    warp_data.cam_xpos.assign(cam_xpos)
    warp_data.cam_xmat.assign(cam_xmat)


@wp.kernel(enable_backward=False)
def _pack_warp_rgb_to_bgra_kernel(
    packed: wp.array2d[wp.uint32],
    rgb_adr: wp.array[int],
    camera_index: int,
    frame_out: wp.array3d[wp.uint8],
) -> None:
    yid, xid = wp.tid()
    width = frame_out.shape[1]
    pixelid = yid * width + xid
    val = packed[0, rgb_adr[camera_index] + pixelid]
    frame_out[yid, xid, 0] = wp.uint8(val & wp.uint32(0xFF))
    frame_out[yid, xid, 1] = wp.uint8((val >> wp.uint32(8)) & wp.uint32(0xFF))
    frame_out[yid, xid, 2] = wp.uint8((val >> wp.uint32(16)) & wp.uint32(0xFF))
    frame_out[yid, xid, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def _pack_warp_rgb_to_bgra_tile_kernel(
    packed: wp.array2d[wp.uint32],
    rgb_adr: wp.array[int],
    camera_index: int,
    tile_width: int,
    tile_x: int,
    tile_y: int,
    frame_out: wp.array3d[wp.uint8],
) -> None:
    yid, xid = wp.tid()
    pixelid = yid * tile_width + xid
    val = packed[0, rgb_adr[camera_index] + pixelid]
    out_y = tile_y + yid
    out_x = tile_x + xid
    frame_out[out_y, out_x, 0] = wp.uint8(val & wp.uint32(0xFF))
    frame_out[out_y, out_x, 1] = wp.uint8((val >> wp.uint32(8)) & wp.uint32(0xFF))
    frame_out[out_y, out_x, 2] = wp.uint8((val >> wp.uint32(16)) & wp.uint32(0xFF))
    frame_out[out_y, out_x, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def _pack_warp_rgb_to_bgra_tiles_kernel(
    packed: wp.array2d[wp.uint32],
    rgb_adr: wp.array[int],
    camera_index: int,
    tile_width: int,
    tile_height: int,
    tile_cols: int,
    frame_out: wp.array3d[wp.uint8],
) -> None:
    worldid, yid, xid = wp.tid()
    pixelid = yid * tile_width + xid
    val = packed[worldid, rgb_adr[camera_index] + pixelid]
    tile_row = worldid / tile_cols
    tile_col = worldid - tile_row * tile_cols
    out_y = tile_row * tile_height + yid
    out_x = tile_col * tile_width + xid
    frame_out[out_y, out_x, 0] = wp.uint8(val & wp.uint32(0xFF))
    frame_out[out_y, out_x, 1] = wp.uint8((val >> wp.uint32(8)) & wp.uint32(0xFF))
    frame_out[out_y, out_x, 2] = wp.uint8((val >> wp.uint32(16)) & wp.uint32(0xFF))
    frame_out[out_y, out_x, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def _clear_bgra_tiles_kernel(
    tile_start: int,
    tile_width: int,
    tile_height: int,
    tile_cols: int,
    frame_out: wp.array3d[wp.uint8],
) -> None:
    tile_offset, yid, xid = wp.tid()
    tile_idx = tile_start + tile_offset
    tile_row = tile_idx / tile_cols
    tile_col = tile_idx - tile_row * tile_cols
    out_y = tile_row * tile_height + yid
    out_x = tile_col * tile_width + xid
    frame_out[out_y, out_x, 0] = wp.uint8(0)
    frame_out[out_y, out_x, 1] = wp.uint8(0)
    frame_out[out_y, out_x, 2] = wp.uint8(0)
    frame_out[out_y, out_x, 3] = wp.uint8(255)


@wp.kernel(enable_backward=False)
def _set_tracking_camera_pose_kernel(
    body_xpos: wp.array2d[wp.vec3],
    cam_xpos: wp.array2d[wp.vec3],
    cam_xmat: wp.array2d[wp.mat33],
    camera_id: int,
    trackbodyid: int,
    camera_params: wp.array[float],
    use_lookat: bool,
    smooth_lookat: wp.array[wp.vec3],
    smooth_valid: wp.array[int],
    use_smooth: bool,
) -> None:
    worldid = wp.tid()
    target = wp.vec3(camera_params[3], camera_params[4], camera_params[5])
    if not use_lookat:
        target = body_xpos[worldid, trackbodyid]
        if use_smooth:
            if smooth_valid[worldid] != 0:
                target = smooth_lookat[worldid] + camera_params[6] * (
                    target - smooth_lookat[worldid]
                )
            smooth_lookat[worldid] = target
            smooth_valid[worldid] = 1

    elev = camera_params[1] * wp.static(wp.pi / 180.0)
    azim = camera_params[2] * wp.static(wp.pi / 180.0)
    cos_elev = wp.cos(elev)
    sin_elev = wp.sin(elev)
    cos_azim = wp.cos(azim)
    sin_azim = wp.sin(azim)
    forward = wp.normalize(wp.vec3(cos_elev * cos_azim, cos_elev * sin_azim, sin_elev))
    up_hint = wp.normalize(
        wp.vec3(-sin_elev * cos_azim, -sin_elev * sin_azim, cos_elev)
    )
    backward = -forward
    right = wp.normalize(wp.cross(forward, up_hint))
    true_up = wp.normalize(wp.cross(backward, right))
    cam_xpos[worldid, camera_id] = target - forward * camera_params[0]
    cam_xmat[worldid, camera_id] = wp.mat33(
        right[0],
        true_up[0],
        backward[0],
        right[1],
        true_up[1],
        backward[1],
        right[2],
        true_up[2],
        backward[2],
    )


@wp.func
def _read_vec3_3d(arr: wp.array3d[float], i: int, j: int) -> wp.vec3:
    return wp.vec3(arr[i, j, 0], arr[i, j, 1], arr[i, j, 2])


@wp.func
def _read_quat_3d(arr: wp.array3d[float], i: int, j: int) -> wp.quat:
    return wp.quat(arr[i, j, 0], arr[i, j, 1], arr[i, j, 2], arr[i, j, 3])


@wp.func
def _read_quat_2d(arr: wp.array2d[float], i: int) -> wp.quat:
    return wp.quat(arr[i, 0], arr[i, 1], arr[i, 2], arr[i, 3])


@wp.func
def _read_vec3_2d(arr: wp.array2d[float], i: int) -> wp.vec3:
    return wp.vec3(arr[i, 0], arr[i, 1], arr[i, 2])


@wp.func
def _apply_loop_pose_vec(
    pos: wp.vec3,
    loop_q: wp.quat,
    loop_t: wp.vec3,
    cycles: int,
) -> wp.vec3:
    out = pos
    cycle = int(0)
    while cycle < cycles:
        out = rot_vec_quat(out, loop_q) + loop_t
        cycle = cycle + 1
    return out


@wp.func
def _apply_loop_pose_quat(q: wp.quat, loop_q: wp.quat, cycles: int) -> wp.quat:
    out = q
    cycle = int(0)
    while cycle < cycles:
        out = mul_quat(loop_q, out)
        cycle = cycle + 1
    return out


@wp.kernel(enable_backward=False)
def _set_reference_ghost_mocap_kernel(
    motion_xpos: wp.array3d[float],
    motion_xquat: wp.array3d[float],
    ref_frame_idx: wp.array[int],
    motion_clip_idx: wp.array[int],
    clip_start: wp.array[int],
    clip_length: wp.array[int],
    clip_loop_root_quat: wp.array2d[float],
    clip_loop_root_pos: wp.array2d[float],
    ghost_mocap_ids: wp.array[int],
    ghost_sim_body_ids: wp.array[int],
    ghost_offset: wp.array[wp.vec3],
    mocap_pos: wp.array2d[wp.vec3],
    mocap_quat: wp.array2d[wp.quat],
) -> None:
    worldid, ghost_idx = wp.tid()
    clip_idx = motion_clip_idx[worldid]
    clip_len = clip_length[clip_idx]
    ref_idx = ref_frame_idx[worldid]
    local_idx = ref_idx % clip_len
    cycles = ref_idx // clip_len
    frame_idx = clip_start[clip_idx] + local_idx
    body_id = ghost_sim_body_ids[ghost_idx]

    pos = _read_vec3_3d(motion_xpos, frame_idx, body_id)
    quat = _read_quat_3d(motion_xquat, frame_idx, body_id)
    loop_q = _read_quat_2d(clip_loop_root_quat, clip_idx)
    loop_t = _read_vec3_2d(clip_loop_root_pos, clip_idx)
    if cycles > 0:
        pos = _apply_loop_pose_vec(pos, loop_q, loop_t, cycles)
        quat = _apply_loop_pose_quat(quat, loop_q, cycles)

    mocap_id = ghost_mocap_ids[ghost_idx]
    mocap_pos[worldid, mocap_id] = pos + ghost_offset[0]
    mocap_quat[worldid, mocap_id] = quat


def _resolve_camera_fovy(
    model: mujoco.MjModel,
    camera_state: mujoco.MjvCamera,
    camera_id: int,
) -> float:
    is_fixed = camera_state.type in (
        mujoco.mjtCamera.mjCAMERA_FIXED,
        int(mujoco.mjtCamera.mjCAMERA_FIXED),
    )
    if is_fixed and 0 <= int(camera_id) < int(model.ncam):
        return float(model.cam_fovy[int(camera_id)])
    return float(model.vis.global_.fovy)


class MujocoWarpPlaybackRenderer:
    def __init__(
        self,
        *,
        env_raw: Any,
        model: mujoco.MjModel,
        width: int,
        height: int,
        data: mujoco.MjData | None = None,
        camera: str | int | mujoco.MjvCamera | None = None,
        overlay_renderer: Any | None = None,
        cam_lookat_ema_alpha: float | None = None,
        tile_env_count: int = 1,
        tile_cols: int | None = None,
        use_textures: bool = True,
        floor_shader_mode: str = "texture",
        render_block_dim: int = 64,
        lbs_mesh_mode: str | None = None,
        lbs_mesh_bundle_path: str | Path | None = None,
        lbs_mesh_shading_mode: str | None = None,
        lbs_activation_mapping_path: str | Path | None = None,
        lbs_activation_tensor: torch.Tensor | None = None,
        character_render_mode: str = "mesh",
    ) -> None:
        if not mujoco_warp_playback_is_available():
            raise RuntimeError("MuJoCo-Warp playback backend is unavailable.")
        if overlay_renderer is not None:
            raise RuntimeError(
                "MuJoCo-Warp playback does not support GL reward overlays yet. "
                "Use video_render_backend=mujoco_gl for overlay videos."
            )

        model, data = clone_playback_render_model(model, data)
        normalize_character_geom_groups(model, env_raw=env_raw)
        ghost_geom_ids = [
            geom_id
            for geom_id in range(int(model.ngeom))
            if (
                _body_name := mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[geom_id]),
                )
            )
            and str(_body_name).startswith("ghost_")
        ]
        ghost_rgba = model.geom_rgba[ghost_geom_ids].copy()
        ghost_matid = model.geom_matid[ghost_geom_ids].copy()
        _apply_render_model_character_mode(
            model,
            character_render_mode,
            env_raw=env_raw,
        )
        model.geom_rgba[ghost_geom_ids] = ghost_rgba
        model.geom_matid[ghost_geom_ids] = ghost_matid
        model.geom_group[np.asarray(model.geom_rgba)[:, 3] <= 0.0] = 0
        self._env_raw = env_raw
        self.model = model
        self.data = data if data is not None else mujoco.MjData(model)
        self.width = int(width)
        self.height = int(height)
        self._tile_env_count = max(1, int(tile_env_count))
        if tile_cols is None or int(tile_cols) <= 0:
            resolved_tile_cols = int(np.ceil(np.sqrt(float(self._tile_env_count))))
        else:
            resolved_tile_cols = int(tile_cols)
        self._tile_cols = max(1, min(resolved_tile_cols, self._tile_env_count))
        self._tile_rows = int(np.ceil(self._tile_env_count / float(self._tile_cols)))
        self._output_width = self.width * self._tile_cols
        self._output_height = self.height * self._tile_rows
        self._camera = camera
        self._use_textures = bool(use_textures)
        self._floor_shader_mode = _resolve_floor_shader_mode(floor_shader_mode)
        self._render_block_dim = int(render_block_dim)
        self._lbs_mesh_mode = _resolve_lbs_mesh_mode(lbs_mesh_mode)
        self._lbs_mesh_shading_mode = _resolve_lbs_mesh_shading_mode(
            lbs_mesh_shading_mode
        )
        self._cam_lookat_ema_alpha = cam_lookat_ema_alpha
        self._smooth_lookat: np.ndarray | None = None
        self._camera_id = _resolve_warp_camera_id(model, camera)
        self._mjv_override_camera_id = self._camera_id
        self._render_camera_index = 0
        self._render_skybox = _model_has_skybox(model)
        self._camera_active = [idx == self._camera_id for idx in range(int(model.ncam))]
        self._scene = mujoco.MjvScene(
            self.model,
            maxgeom=max(4096, int(self.model.ngeom) * 24 + 1024),
        )
        self._option = mujoco.MjvOption()
        self._perturb = mujoco.MjvPerturb()
        self._camera_state = mujoco.MjvCamera()
        self._rgb = wp.empty((1, self.height, self.width), dtype=wp.vec3)
        self._cuda_bgra_frame_wp: wp.array | None = None
        self._cuda_bgra_frame_pool: list[torch.Tensor | None] = []
        self._cuda_bgra_frame_wp_pool: list[wp.array | None] = []
        self._cuda_render_graphs: dict[int, _CudaRenderGraphEntry] = {}
        self._cuda_graph_capture_count = 0
        self._lbs_mesh_overlay = (
            _WarpLBSMeshOverlay(
                model=self.model,
                bundle_path=_resolve_lbs_mesh_bundle_path(lbs_mesh_bundle_path),
                mode=self._lbs_mesh_mode,
                shading_mode=self._lbs_mesh_shading_mode,
                activation_mapping_path=lbs_activation_mapping_path,
                activation_tensor=lbs_activation_tensor,
                activation_color_settings=_resolve_lbs_activation_color_settings(
                    env_raw
                ),
            )
            if self._lbs_mesh_mode != "off"
            else None
        )
        self._muscle_line_overlays = (
            []
            if self._lbs_mesh_overlay is not None
            else _resolve_dynamic_muscle_line_overlays(env_raw)
        )
        if self._muscle_line_overlays:
            logger.info(
                "Initialized %d Warp muscle-line overlay source(s).",
                len(self._muscle_line_overlays),
            )
        self._m = None
        self._d = None
        self._tile_m = None
        self._tile_d = None
        self._tile_rc = None
        self._tile_context_world_count = 0
        self._tile_context_signature: tuple[Any, ...] | None = None
        self._ghost_mocap_ids_wp = None
        self._ghost_sim_body_ids_wp = None
        self._ghost_offset_wp = None
        self._ghost_graph_tensors: dict[str, torch.Tensor] = {}
        self._ghost_graph_arrays: dict[str, wp.array] = {}
        self._ghost_metadata_tensors: dict[str, torch.Tensor] = {}
        self._ghost_metadata_arrays: dict[str, wp.array] = {}
        self._graph_smooth_lookat = None
        self._graph_smooth_valid = None
        self._graph_camera_params_t: torch.Tensor | None = None
        self._graph_camera_params_wp: wp.array | None = None
        self._graph_camera_params_host = np.zeros(7, dtype=np.float32)
        self._render_wp_stream: wp.Stream | None = None
        self._render_torch_stream: torch.cuda.ExternalStream | None = None
        # `render_substep_batch` takes numpy and returns numpy -- the contract
        # `train/full_video_recorder.py` depends on -- but routes the work
        # through the CUDA path when a device is available, converting back at
        # the end. Capturing that decision here instead of reading
        # `torch.cuda.is_available()` per batch keeps the two implementations of
        # one public contract independently testable, and takes a global read
        # out of a per-batch path. The value is what it always was.
        self._prefer_cuda_batch_render = torch.cuda.is_available()
        # `_render_batched_tiled_frame_cuda` likewise has two implementations:
        # an eager, per-stage-timed one and a CUDA-graph-captured one. They call
        # different overlay entry points and report different stage sets, so
        # pinning the choice is what lets either be tested on any host.
        self._use_cuda_render_graph = torch.cuda.is_available()
        self._producer_ready_event: torch.cuda.Event | None = None
        self._render_ready_event: torch.cuda.Event | None = None
        self._batched_needs_subtree_com = self._batched_render_needs_subtree_com()
        self._rc = None
        self._m = put_model(self.model)
        self._m.block_dim.render = self._render_block_dim
        self._d = mjwr.put_data(self.model, self.data, nworld=1)

    def _needs_overlay_depth(self) -> bool:
        return bool(getattr(self, "_muscle_line_overlays", [])) or (
            getattr(self, "_lbs_mesh_overlay", None) is not None
        )

    def _configure_camera(self) -> None:
        ema_active = self._cam_lookat_ema_alpha is not None
        if self._camera is None:
            self._camera_state.trackbodyid = int(
                getattr(self._env_raw, "torso_body_idx", 1)
            )
            self._camera_state.distance = 3.0
            self._camera_state.elevation = -20.0
            self._camera_state.azimuth = 135.0
            if ema_active:
                self._camera_state.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self._camera_state.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            return

        if isinstance(self._camera, mujoco.MjvCamera):
            self._camera_state.fixedcamid = int(self._camera.fixedcamid)
            trackbodyid = int(getattr(self._camera, "trackbodyid", -1))
            tracking_body_name = getattr(
                self._env_raw,
                "_render_tracking_body_name",
                None,
            )
            if tracking_body_name:
                mapped_trackbodyid = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    tracking_body_name,
                )
                if mapped_trackbodyid >= 0:
                    trackbodyid = int(mapped_trackbodyid)
            self._camera_state.trackbodyid = int(trackbodyid)
            self._camera_state.distance = float(self._camera.distance)
            self._camera_state.elevation = float(self._camera.elevation)
            self._camera_state.azimuth = float(self._camera.azimuth)
            if hasattr(self._camera_state, "orthographic") and hasattr(
                self._camera,
                "orthographic",
            ):
                self._camera_state.orthographic = self._camera.orthographic
            if hasattr(self._camera_state, "lookat") and hasattr(
                self._camera, "lookat"
            ):
                self._camera_state.lookat[:] = self._camera.lookat
            is_tracking = self._camera.type in (
                mujoco.mjtCamera.mjCAMERA_TRACKING,
                int(mujoco.mjtCamera.mjCAMERA_TRACKING),
            )
            if ema_active and is_tracking:
                self._camera_state.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self._camera_state.type = self._camera.type
            is_fixed = self._camera_state.type in (
                mujoco.mjtCamera.mjCAMERA_FIXED,
                int(mujoco.mjtCamera.mjCAMERA_FIXED),
            )
            if is_fixed:
                selected_camera_id = int(self._camera_state.fixedcamid)
                if not 0 <= selected_camera_id < int(self.model.ncam):
                    raise ValueError(
                        "MjvCamera.fixedcamid must select an existing render camera."
                    )
            else:
                selected_camera_id = int(self._mjv_override_camera_id)
            if selected_camera_id != self._camera_id:
                self._camera_id = selected_camera_id
                self._camera_active = [
                    idx == selected_camera_id for idx in range(int(self.model.ncam))
                ]
            return

        if isinstance(self._camera, str):
            camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                self._camera,
            )
            if camera_id < 0:
                raise ValueError(f'Camera "{self._camera}" not found in render model.')
        else:
            camera_id = int(self._camera)
        self._camera_state.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera_state.fixedcamid = camera_id

    def _update_smooth_lookat(self) -> None:
        if self._cam_lookat_ema_alpha is None:
            return
        body_pos = self.data.xpos[self._camera_state.trackbodyid].copy()
        if self._smooth_lookat is None:
            self._smooth_lookat = body_pos
        else:
            alpha = self._cam_lookat_ema_alpha
            self._smooth_lookat += alpha * (body_pos - self._smooth_lookat)
        self._camera_state.lookat[:] = self._smooth_lookat

    def reset_smooth_lookat(self) -> None:
        self._smooth_lookat = None
        if self._graph_smooth_valid is not None:
            smooth_valid_t = wp.to_torch(self._graph_smooth_valid)
            if smooth_valid_t.is_cuda:
                torch_stream = torch.cuda.current_stream(smooth_valid_t.device)
                with wp.ScopedStream(
                    warp_stream_from_torch(torch_stream),
                    sync_enter=False,
                    sync_exit=False,
                ):
                    self._graph_smooth_valid.zero_()
            else:
                self._graph_smooth_valid.zero_()

    def close(self) -> None:
        self._d = None
        self._m = None
        self._rc = None
        self._tile_d = None
        self._tile_m = None
        self._tile_rc = None
        self._tile_context_world_count = 0
        self._tile_context_signature = None
        self._ghost_mocap_ids_wp = None
        self._ghost_sim_body_ids_wp = None
        self._ghost_offset_wp = None
        self._ghost_graph_tensors = {}
        self._ghost_graph_arrays = {}
        self._ghost_metadata_tensors = {}
        self._ghost_metadata_arrays = {}
        self._graph_smooth_lookat = None
        self._graph_smooth_valid = None
        self._graph_camera_params_t = None
        self._graph_camera_params_wp = None
        self._render_wp_stream = None
        self._render_torch_stream = None
        self._producer_ready_event = None
        self._render_ready_event = None
        self._cuda_bgra_frame_wp = None
        self._cuda_bgra_frame_pool = []
        self._cuda_bgra_frame_wp_pool = []
        self._cuda_render_graphs = {}

    def __del__(self) -> None:
        try:
            self.close()
        except Exception as exc:  # noqa: BLE001 - destructors must not raise during GC.
            logger.debug(
                "MuJoCo Warp playback cleanup failed: %s",
                exc,
                exc_info=True,
            )

    def _ensure_single_render_context(self) -> Any:
        if self._rc is None:
            self._rc = mjwr.create_render_context(
                self.model,
                nworld=1,
                cam_res=(self.width, self.height),
                render_rgb=True,
                render_depth=self._needs_overlay_depth(),
                render_seg=False,
                use_textures=bool(getattr(self, "_use_textures", True)),
                use_shadows=True,
                enabled_geom_groups=[1, 2, 3, 4, 5],
                background_color=_MUJOCO_GL_CLEAR_RGBA,
                cam_active=self._camera_active,
                render_skybox=self._render_skybox,
            )
        return self._rc

    def _ensure_cuda_frame(self, buffer_index: int = 0) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError("MuJoCo-Warp pynvcodec playback requires a CUDA device.")
        buffer_index = max(0, int(buffer_index))
        shape = (
            int(getattr(self, "_output_height", self.height)),
            int(getattr(self, "_output_width", self.width)),
            4,
        )
        if not hasattr(self, "_cuda_bgra_frame_pool"):
            self._cuda_bgra_frame_pool = []
            self._cuda_bgra_frame_wp_pool = []
        while len(self._cuda_bgra_frame_pool) <= buffer_index:
            self._cuda_bgra_frame_pool.append(None)
            self._cuda_bgra_frame_wp_pool.append(None)
        frame = self._cuda_bgra_frame_pool[buffer_index]
        if frame is None or tuple(frame.shape) != shape:
            frame = torch.empty(shape, device="cuda", dtype=torch.uint8)
            self._cuda_bgra_frame_pool[buffer_index] = frame
            self._cuda_bgra_frame_wp_pool[buffer_index] = wp.from_torch(
                frame,
                dtype=wp.uint8,
                requires_grad=False,
            )
        self._cuda_bgra_frame_wp = self._cuda_bgra_frame_wp_pool[buffer_index]
        return frame

    def _clear_unused_cuda_tiles(self, tile_limit: int) -> None:
        tile_cols = int(getattr(self, "_tile_cols", 1))
        tile_rows = int(
            getattr(
                self,
                "_tile_rows",
                int(
                    np.ceil(
                        float(int(getattr(self, "_tile_env_count", 1)))
                        / float(max(1, tile_cols))
                    )
                ),
            )
        )
        capacity = tile_cols * tile_rows
        tile_limit = max(0, min(int(tile_limit), capacity))
        if tile_limit >= capacity:
            return
        assert self._cuda_bgra_frame_wp is not None
        wp.launch(
            _clear_bgra_tiles_kernel,
            dim=(capacity - tile_limit, self.height, self.width),
            inputs=[
                int(tile_limit),
                int(self.width),
                int(self.height),
                int(getattr(self, "_tile_cols", 1)),
            ],
            outputs=[self._cuda_bgra_frame_wp],
        )

    def _render_current_frame_to_context(
        self,
        *,
        env_idx: int = 0,
        activation: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        rc = self._ensure_single_render_context()
        assert self._m is not None and self._d is not None
        pos, xmat = _mjv_camera_pose(
            model=self.model,
            data=self.data,
            camera=self._camera_state,
            scene=self._scene,
            option=self._option,
            perturb=self._perturb,
        )
        for name, converter in (
            ("qpos", self._vec_stack_field),
            ("qvel", self._vec_stack_field),
            ("mocap_pos", self._vec_stack_field),
            ("mocap_quat", self._vec_stack_field),
            ("xpos", self._vec_stack_field),
            ("xquat", self._vec_stack_field),
            ("xmat", self._mat33_stack_field),
            ("geom_xpos", self._vec_stack_field),
            ("geom_xmat", self._mat33_stack_field),
            ("site_xpos", self._vec_stack_field),
            ("site_xmat", self._mat33_stack_field),
            ("light_xpos", self._vec_stack_field),
            ("light_xdir", self._vec_stack_field),
            ("subtree_com", self._vec_stack_field),
            ("flexvert_xpos", self._vec_stack_field),
        ):
            target = getattr(self._d, name, None)
            if target is not None:
                target.assign(converter(getattr(self.data, name))[None, ...])
        self._d.cam_xpos.assign(self._vec_stack_field(self.data.cam_xpos)[None, ...])
        self._d.cam_xmat.assign(self._mat33_stack_field(self.data.cam_xmat)[None, ...])
        _copy_camera_pose_to_warp_data(
            self._d,
            camera_id=self._camera_id,
            pos=pos,
            xmat=xmat,
        )
        mjwr.refit_bvh(self._m, self._d, rc)
        mjwr.render(self._m, self._d, rc)
        if self._muscle_line_overlays:
            fovy = _resolve_camera_fovy(
                self.model,
                self._camera_state,
                self._camera_id,
            )
            for overlay in self._muscle_line_overlays:
                overlay.draw(
                    data=self._d,
                    rc=rc,
                    camera_id=self._camera_id,
                    width=self.width,
                    height=self.height,
                    fovy=fovy,
                    env_idx=env_idx,
                    activation=activation,
                )
        if getattr(self, "_lbs_mesh_overlay", None) is not None:
            fovy = _resolve_camera_fovy(
                self.model,
                self._camera_state,
                self._camera_id,
            )
            self._lbs_mesh_overlay.draw(
                data=self._d,
                rc=rc,
                camera_id=self._camera_id,
                width=self.width,
                height=self.height,
                fovy=fovy,
                env_idx=env_idx,
                activation=activation,
            )

    def _render_current_frame(
        self,
        *,
        env_idx: int = 0,
        activation: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        self._render_current_frame_to_context(env_idx=env_idx, activation=activation)
        rc = self._ensure_single_render_context()
        mjwr.get_rgb(rc, int(getattr(self, "_render_camera_index", 0)), self._rgb)
        rgb = np.asarray(self._rgb.numpy()[0], dtype=np.float32)
        return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    def _render_current_frame_cuda(
        self,
        *,
        env_idx: int = 0,
        buffer_index: int = 0,
    ) -> torch.Tensor:
        frame = self._ensure_cuda_frame(buffer_index)
        self._render_current_frame_to_context(env_idx=env_idx)
        rc = self._ensure_single_render_context()
        assert self._cuda_bgra_frame_wp is not None
        wp.launch(
            _pack_warp_rgb_to_bgra_kernel,
            dim=(self.height, self.width),
            inputs=[
                rc.rgb_data,
                rc.rgb_adr,
                int(getattr(self, "_render_camera_index", 0)),
            ],
            outputs=[self._cuda_bgra_frame_wp],
        )
        return frame

    def _ensure_batched_tile_context(self, world_count: int) -> tuple[Any, Any, Any]:
        world_count = max(1, int(world_count))
        context_signature = (
            world_count,
            self.width,
            self.height,
            tuple(bool(value) for value in (self._camera_active or [])),
            float(self.model.vis.global_.fovy),
            tuple(np.asarray(self.model.cam_fovy, dtype=np.float64)),
            tuple(
                np.asarray(getattr(self.model, "cam_orthographic", []), dtype=np.int32)
            ),
        )
        if (
            self._tile_context_world_count == world_count
            and self._tile_m is not None
            and self._tile_d is not None
            and self._tile_rc is not None
            and getattr(self, "_tile_context_signature", None) == context_signature
        ):
            return self._tile_m, self._tile_d, self._tile_rc

        self._cuda_render_graphs = {}
        self._tile_m = put_model(self.model)
        block_dim = getattr(self._tile_m, "block_dim", None)
        if block_dim is not None:
            block_dim.render = int(getattr(self, "_render_block_dim", 64))
        self._tile_d = mjwr.put_data(self.model, self.data, nworld=world_count)
        self._tile_rc = mjwr.create_render_context(
            self.model,
            nworld=world_count,
            cam_res=(self.width, self.height),
            render_rgb=True,
            render_depth=self._needs_overlay_depth(),
            render_seg=False,
            use_textures=bool(getattr(self, "_use_textures", True)),
            use_shadows=True,
            enabled_geom_groups=[1, 2, 3, 4, 5],
            background_color=_MUJOCO_GL_CLEAR_RGBA,
            cam_active=self._camera_active,
            render_skybox=getattr(
                self,
                "_render_skybox",
                _model_has_skybox(self.model),
            ),
        )
        tile_xpos = getattr(self._tile_d, "xpos", None)
        tile_device = getattr(tile_xpos, "device", None)
        self._graph_smooth_lookat = wp.zeros(
            world_count, dtype=wp.vec3, device=tile_device
        )
        self._graph_smooth_valid = wp.zeros(
            world_count, dtype=wp.int32, device=tile_device
        )
        camera_device = (
            wp.to_torch(self._tile_d.xpos).device
            if hasattr(self._tile_d, "xpos")
            else torch.device("cpu")
        )
        self._graph_camera_params_t = torch.empty(
            7, dtype=torch.float32, device=camera_device
        )
        self._graph_camera_params_wp = wp.from_torch(
            self._graph_camera_params_t, dtype=wp.float32, requires_grad=False
        )
        self._tile_context_world_count = world_count
        self._tile_context_signature = context_signature
        return self._tile_m, self._tile_d, self._tile_rc

    @staticmethod
    def _mat33_stack_field(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).reshape((-1, 3, 3)).copy()

    @staticmethod
    def _vec_stack_field(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).copy()

    @staticmethod
    def _assign_warp_array(target: Any, values: list[np.ndarray]) -> None:
        if target is None:
            return
        target.assign(np.stack(values, axis=0))

    def _copy_torch_rows_to_warp_array(
        self,
        target: Any,
        source: torch.Tensor,
        *,
        row_count: int,
        col_count: int,
    ) -> None:
        source_t = source[:row_count, :col_count]
        if source_t.dtype != torch.float32 or not source_t.is_contiguous():
            raise RuntimeError(
                "Renderer graph inputs must be contiguous float32 staging tensors."
            )
        wp.copy(
            target,
            wp.from_torch(
                source_t,
                dtype=getattr(target, "dtype", wp.float32),
                requires_grad=False,
            ),
        )

    def _ensure_reference_ghost_arrays(self, data: Any) -> int:
        env_raw = self._env_raw
        if not hasattr(self, "_ghost_metadata_tensors"):
            self._ghost_metadata_tensors = {}
        if not hasattr(self, "_ghost_metadata_arrays"):
            self._ghost_metadata_arrays = {}
        ghost_mocap_ids = getattr(env_raw, "_ghost_mocap_ids", None)
        ghost_sim_body_ids = getattr(env_raw, "_ghost_sim_body_ids", None)
        ghost_offset = getattr(env_raw, "_ghost_offset_np", None)
        if (
            ghost_mocap_ids is None
            or ghost_sim_body_ids is None
            or ghost_offset is None
        ):
            self._ghost_mocap_ids_wp = None
            self._ghost_sim_body_ids_wp = None
            self._ghost_offset_wp = None
            self._ghost_metadata_tensors = {}
            self._ghost_metadata_arrays = {}
            return 0
        ghost_mocap_ids_np = np.asarray(ghost_mocap_ids, dtype=np.int32)
        ghost_sim_body_ids_np = np.asarray(ghost_sim_body_ids, dtype=np.int32)
        if ghost_mocap_ids_np.size == 0 or ghost_sim_body_ids_np.size == 0:
            self._ghost_mocap_ids_wp = None
            self._ghost_sim_body_ids_wp = None
            self._ghost_offset_wp = None
            self._ghost_metadata_tensors = {}
            self._ghost_metadata_arrays = {}
            return 0
        if ghost_mocap_ids_np.shape != ghost_sim_body_ids_np.shape:
            raise ValueError(
                "Reference ghost mocap/body id arrays must have equal shape."
            )

        torch_device = wp.to_torch(data.mocap_pos).device
        values = {
            "mocap_ids": torch.as_tensor(ghost_mocap_ids_np, dtype=torch.int32),
            "body_ids": torch.as_tensor(ghost_sim_body_ids_np, dtype=torch.int32),
            "offset": torch.as_tensor(
                np.asarray(ghost_offset, dtype=np.float32).reshape(1, 3)
            ),
        }
        warp_dtypes = {
            "mocap_ids": wp.int32,
            "body_ids": wp.int32,
            "offset": wp.vec3,
        }
        for name, source in values.items():
            owned = self._ghost_metadata_tensors.get(name)
            if (
                owned is None
                or owned.shape != source.shape
                or owned.device != torch_device
            ):
                owned = torch.empty(
                    source.shape, dtype=source.dtype, device=torch_device
                )
                self._ghost_metadata_tensors[name] = owned
                self._ghost_metadata_arrays[name] = wp.from_torch(
                    owned, dtype=warp_dtypes[name], requires_grad=False
                )
            owned.copy_(source)
        self._ghost_mocap_ids_wp = self._ghost_metadata_arrays["mocap_ids"]
        self._ghost_sim_body_ids_wp = self._ghost_metadata_arrays["body_ids"]
        self._ghost_offset_wp = self._ghost_metadata_arrays["offset"]
        return int(ghost_mocap_ids_np.size)

    @staticmethod
    def _torch_i32_view(tensor: torch.Tensor, row_count: int) -> wp.array:
        return wp.from_torch(
            tensor[:row_count].to(dtype=torch.int32).contiguous(),
            dtype=wp.int32,
            requires_grad=False,
        )

    @staticmethod
    def _torch_f32_view(tensor: torch.Tensor) -> wp.array:
        return wp.from_torch(
            tensor.to(dtype=torch.float32).contiguous(),
            dtype=wp.float32,
            requires_grad=False,
        )

    def _stage_reference_ghost_mocap_gpu(self, data: Any, *, world_count: int) -> None:
        metadata_tensors = getattr(self, "_ghost_metadata_tensors", {})
        ghost_count = (
            int(metadata_tensors["mocap_ids"].numel())
            if getattr(self, "_ghost_graph_arrays", {})
            and "mocap_ids" in metadata_tensors
            else self._ensure_reference_ghost_arrays(data)
        )
        if ghost_count <= 0:
            return
        env_raw = self._env_raw
        required = (
            "motion_xpos",
            "motion_xquat",
            "ref_frame_idx",
            "motion_clip_idx",
            "motion_clip_start",
            "motion_clip_length",
            "motion_clip_loop_root_quat",
            "motion_clip_loop_root_pos",
        )
        if not all(torch.is_tensor(getattr(env_raw, name, None)) for name in required):
            return
        assert self._ghost_mocap_ids_wp is not None
        assert self._ghost_sim_body_ids_wp is not None
        assert self._ghost_offset_wp is not None
        graph_arrays = getattr(self, "_ghost_graph_arrays", {})
        wp.launch(
            _set_reference_ghost_mocap_kernel,
            dim=(int(world_count), ghost_count),
            inputs=[
                graph_arrays["motion_xpos"]
                if "motion_xpos" in graph_arrays
                else self._torch_f32_view(env_raw.motion_xpos),
                graph_arrays["motion_xquat"]
                if "motion_xquat" in graph_arrays
                else self._torch_f32_view(env_raw.motion_xquat),
                graph_arrays["ref_frame_idx"]
                if "ref_frame_idx" in graph_arrays
                else self._torch_i32_view(env_raw.ref_frame_idx, int(world_count)),
                graph_arrays["motion_clip_idx"]
                if "motion_clip_idx" in graph_arrays
                else self._torch_i32_view(env_raw.motion_clip_idx, int(world_count)),
                graph_arrays["motion_clip_start"]
                if "motion_clip_start" in graph_arrays
                else self._torch_i32_view(
                    env_raw.motion_clip_start, int(env_raw.motion_clip_start.numel())
                ),
                graph_arrays["motion_clip_length"]
                if "motion_clip_length" in graph_arrays
                else self._torch_i32_view(
                    env_raw.motion_clip_length, int(env_raw.motion_clip_length.numel())
                ),
                graph_arrays["motion_clip_loop_root_quat"]
                if "motion_clip_loop_root_quat" in graph_arrays
                else self._torch_f32_view(env_raw.motion_clip_loop_root_quat),
                graph_arrays["motion_clip_loop_root_pos"]
                if "motion_clip_loop_root_pos" in graph_arrays
                else self._torch_f32_view(env_raw.motion_clip_loop_root_pos),
                self._ghost_mocap_ids_wp,
                self._ghost_sim_body_ids_wp,
                self._ghost_offset_wp,
            ],
            outputs=[data.mocap_pos, data.mocap_quat],
        )

    def _prepare_reference_ghost_graph_inputs(
        self, data: Any, world_count: int
    ) -> tuple[Any, ...]:
        env_raw = self._env_raw
        specs = {
            "motion_xpos": (torch.float32, None),
            "motion_xquat": (torch.float32, None),
            "ref_frame_idx": (torch.int32, world_count),
            "motion_clip_idx": (torch.int32, world_count),
            "motion_clip_start": (torch.int32, None),
            "motion_clip_length": (torch.int32, None),
            "motion_clip_loop_root_quat": (torch.float32, None),
            "motion_clip_loop_root_pos": (torch.float32, None),
        }
        if not all(torch.is_tensor(getattr(env_raw, name, None)) for name in specs):
            self._ghost_graph_tensors = {}
            self._ghost_graph_arrays = {}
            self._ensure_reference_ghost_arrays(data)
            return ("unavailable",)
        arrays: dict[str, wp.array] = {}
        for name, (dtype, row_count) in specs.items():
            source = getattr(env_raw, name)
            if row_count is not None:
                source = source[:row_count]
            source = source.detach().to(dtype=dtype).contiguous()
            owned = self._ghost_graph_tensors.get(name)
            if (
                owned is None
                or owned.shape != source.shape
                or owned.device != source.device
            ):
                owned = torch.empty_like(source)
                self._ghost_graph_tensors[name] = owned
            owned.copy_(source)
            arrays[name] = wp.from_torch(
                owned,
                dtype=wp.int32 if dtype == torch.int32 else wp.float32,
                requires_grad=False,
            )
        self._ghost_graph_arrays = arrays
        ghost_count = self._ensure_reference_ghost_arrays(data)
        return (
            "reference",
            ghost_count,
            tuple(
                (name, tuple(tensor.shape), str(tensor.device), tensor.data_ptr())
                for name, tensor in sorted(self._ghost_graph_tensors.items())
            ),
            tuple(
                (name, tuple(tensor.shape), str(tensor.device), tensor.data_ptr())
                for name, tensor in sorted(self._ghost_metadata_tensors.items())
            ),
        )

    def _stage_graph_camera_params(self) -> None:
        assert self._graph_camera_params_t is not None
        host = self._graph_camera_params_host
        host[0] = float(self._camera_state.distance)
        host[1] = float(self._camera_state.elevation)
        host[2] = float(self._camera_state.azimuth)
        host[3:6] = np.asarray(self._camera_state.lookat, dtype=np.float32)
        host[6] = float(self._cam_lookat_ema_alpha or 0.0)
        self._graph_camera_params_t.copy_(torch.from_numpy(host))

    def _override_batched_camera_pose_gpu(self, data: Any, *, world_count: int) -> None:
        is_fixed = self._camera_state.type in (
            mujoco.mjtCamera.mjCAMERA_FIXED,
            int(mujoco.mjtCamera.mjCAMERA_FIXED),
        )
        if is_fixed:
            return
        trackbodyid = int(getattr(self._camera_state, "trackbodyid", -1))
        use_lookat = trackbodyid < 0 or trackbodyid >= int(self.model.nbody)
        assert self._graph_smooth_lookat is not None
        assert self._graph_smooth_valid is not None
        assert self._graph_camera_params_wp is not None
        wp.launch(
            _set_tracking_camera_pose_kernel,
            dim=int(world_count),
            inputs=[
                data.xpos,
                data.cam_xpos,
                data.cam_xmat,
                int(self._camera_id),
                max(0, min(trackbodyid, int(self.model.nbody) - 1)),
                self._graph_camera_params_wp,
                bool(use_lookat),
                self._graph_smooth_lookat,
                self._graph_smooth_valid,
                self._cam_lookat_ema_alpha is not None,
            ],
        )

    def _batched_render_needs_subtree_com(self) -> bool:
        com_modes = {
            int(mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM),
            int(mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM),
        }
        if int(getattr(self.model, "ncam", 0)) > 0:
            if any(int(mode) in com_modes for mode in np.asarray(self.model.cam_mode)):
                return True
        return bool(
            int(getattr(self.model, "nlight", 0)) > 0
            and any(
                int(mode) in com_modes for mode in np.asarray(self.model.light_mode)
            )
        )

    def _stage_batched_tile_data_gpu(
        self,
        model: Any,
        data: Any,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        *,
        world_count: int,
        mocap_pos: torch.Tensor | None = None,
        mocap_quat: torch.Tensor | None = None,
    ) -> None:
        del qvel
        self._copy_torch_rows_to_warp_array(
            data.qpos,
            qpos,
            row_count=int(world_count),
            col_count=int(self.model.nq),
        )
        if mocap_pos is not None and getattr(data, "mocap_pos", None) is not None:
            self._copy_torch_rows_to_warp_array(
                data.mocap_pos,
                mocap_pos,
                row_count=int(world_count),
                col_count=int(mocap_pos.shape[1]),
            )
        if mocap_quat is not None and getattr(data, "mocap_quat", None) is not None:
            self._copy_torch_rows_to_warp_array(
                data.mocap_quat,
                mocap_quat,
                row_count=int(world_count),
                col_count=int(mocap_quat.shape[1]),
            )
        if mocap_pos is None and mocap_quat is None:
            self._stage_reference_ghost_mocap_gpu(data, world_count=int(world_count))
        smooth.kinematics(model, data)
        if self._batched_needs_subtree_com:
            smooth.com_pos(model, data)
        smooth.camlight(model, data)
        if int(getattr(model, "nflex", 0)) > 0:
            smooth.flex(model, data)
        self._override_batched_camera_pose_gpu(data, world_count=int(world_count))

    def _stage_batched_tile_data(
        self,
        warp_data: Any,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        world_count: int,
        mocap_pos: np.ndarray | None = None,
        mocap_quat: np.ndarray | None = None,
    ) -> None:
        qpos_values: list[np.ndarray] = []
        qvel_values: list[np.ndarray] = []
        mocap_pos_values: list[np.ndarray] = []
        mocap_quat_values: list[np.ndarray] = []
        xpos_values: list[np.ndarray] = []
        xquat_values: list[np.ndarray] = []
        xmat_values: list[np.ndarray] = []
        geom_xpos_values: list[np.ndarray] = []
        geom_xmat_values: list[np.ndarray] = []
        site_xpos_values: list[np.ndarray] = []
        site_xmat_values: list[np.ndarray] = []
        cam_xpos_values: list[np.ndarray] = []
        cam_xmat_values: list[np.ndarray] = []
        light_xpos_values: list[np.ndarray] = []
        light_xdir_values: list[np.ndarray] = []
        subtree_com_values: list[np.ndarray] = []
        flexvert_xpos_values: list[np.ndarray] = []

        camera_count = int(self.model.ncam)
        for world_id in range(world_count):
            world_mocap_pos = None if mocap_pos is None else mocap_pos[world_id]
            world_mocap_quat = None if mocap_quat is None else mocap_quat[world_id]
            self._set_state_from_arrays(
                qpos[world_id],
                qvel[world_id],
                mocap_pos=world_mocap_pos,
                mocap_quat=world_mocap_quat,
            )
            pos, xmat = _mjv_camera_pose(
                model=self.model,
                data=self.data,
                camera=self._camera_state,
                scene=self._scene,
                option=self._option,
                perturb=self._perturb,
            )

            qpos_values.append(np.asarray(self.data.qpos, dtype=np.float32).copy())
            qvel_values.append(np.asarray(self.data.qvel, dtype=np.float32).copy())
            mocap_pos_values.append(
                np.asarray(self.data.mocap_pos, dtype=np.float32).copy()
            )
            mocap_quat_values.append(
                np.asarray(self.data.mocap_quat, dtype=np.float32).copy()
            )
            xpos_values.append(self._vec_stack_field(self.data.xpos))
            xquat_values.append(np.asarray(self.data.xquat, dtype=np.float32).copy())
            xmat_values.append(self._mat33_stack_field(self.data.xmat))
            geom_xpos_values.append(self._vec_stack_field(self.data.geom_xpos))
            geom_xmat_values.append(self._mat33_stack_field(self.data.geom_xmat))
            site_xpos_values.append(self._vec_stack_field(self.data.site_xpos))
            site_xmat_values.append(self._mat33_stack_field(self.data.site_xmat))

            cam_xpos = np.zeros((camera_count, 3), dtype=np.float32)
            cam_xmat = np.tile(
                np.eye(3, dtype=np.float32),
                (camera_count, 1, 1),
            )
            if camera_count > 0:
                cam_xpos[:] = self._vec_stack_field(self.data.cam_xpos)
                cam_xmat[:] = self._mat33_stack_field(self.data.cam_xmat)
            cam_xpos[int(self._camera_id)] = np.asarray(pos, dtype=np.float32)
            cam_xmat[int(self._camera_id)] = np.asarray(xmat, dtype=np.float32)
            cam_xpos_values.append(cam_xpos)
            cam_xmat_values.append(cam_xmat)

            light_xpos_values.append(self._vec_stack_field(self.data.light_xpos))
            light_xdir_values.append(self._vec_stack_field(self.data.light_xdir))
            subtree_com_values.append(self._vec_stack_field(self.data.subtree_com))
            flexvert_xpos_values.append(self._vec_stack_field(self.data.flexvert_xpos))

        self._assign_warp_array(warp_data.qpos, qpos_values)
        self._assign_warp_array(warp_data.qvel, qvel_values)
        self._assign_warp_array(warp_data.mocap_pos, mocap_pos_values)
        self._assign_warp_array(warp_data.mocap_quat, mocap_quat_values)
        self._assign_warp_array(warp_data.xpos, xpos_values)
        self._assign_warp_array(warp_data.xquat, xquat_values)
        self._assign_warp_array(warp_data.xmat, xmat_values)
        self._assign_warp_array(warp_data.geom_xpos, geom_xpos_values)
        self._assign_warp_array(warp_data.geom_xmat, geom_xmat_values)
        self._assign_warp_array(warp_data.site_xpos, site_xpos_values)
        self._assign_warp_array(warp_data.site_xmat, site_xmat_values)
        self._assign_warp_array(warp_data.cam_xpos, cam_xpos_values)
        self._assign_warp_array(warp_data.cam_xmat, cam_xmat_values)
        self._assign_warp_array(warp_data.light_xpos, light_xpos_values)
        self._assign_warp_array(warp_data.light_xdir, light_xdir_values)
        self._assign_warp_array(warp_data.subtree_com, subtree_com_values)
        self._assign_warp_array(warp_data.flexvert_xpos, flexvert_xpos_values)

    def _set_state_from_arrays(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        mocap_pos: np.ndarray | None = None,
        mocap_quat: np.ndarray | None = None,
    ) -> None:
        self.data.qpos[: self.model.nq] = qpos[: self.model.nq]
        self.data.qvel[: self.model.nv] = qvel[: self.model.nv]
        if mocap_pos is not None:
            self.data.mocap_pos[:] = mocap_pos
        if mocap_quat is not None:
            self.data.mocap_quat[:] = mocap_quat
        update_render_data = getattr(self._env_raw, "_update_render_data", None)
        if callable(update_render_data):
            update_render_data()
        mujoco.mj_forward(self.model, self.data)
        self._update_smooth_lookat()

    def _render_tiled_frame(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        mocap_pos: np.ndarray | None = None,
        mocap_quat: np.ndarray | None = None,
        activation: np.ndarray | None = None,
    ) -> np.ndarray:
        tile_count = int(qpos.shape[0])
        frame = np.zeros(
            (
                int(getattr(self, "_output_height", self.height)),
                int(getattr(self, "_output_width", self.width)),
                3,
            ),
            dtype=np.uint8,
        )
        for tile_idx in range(
            min(tile_count, int(getattr(self, "_tile_env_count", 1)))
        ):
            tile_mocap_pos = None if mocap_pos is None else mocap_pos[tile_idx]
            tile_mocap_quat = None if mocap_quat is None else mocap_quat[tile_idx]
            self._set_state_from_arrays(
                qpos[tile_idx],
                qvel[tile_idx],
                mocap_pos=tile_mocap_pos,
                mocap_quat=tile_mocap_quat,
            )
            tile = self._render_current_frame(
                env_idx=tile_idx,
                activation=None if activation is None else activation[tile_idx],
            )
            row = tile_idx // int(getattr(self, "_tile_cols", 1))
            col = tile_idx % int(getattr(self, "_tile_cols", 1))
            y0 = row * self.height
            x0 = col * self.width
            frame[y0 : y0 + self.height, x0 : x0 + self.width, :] = tile
        return frame

    def _render_tiled_frame_cuda(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        *,
        buffer_index: int = 0,
        mocap_pos: np.ndarray | None = None,
        mocap_quat: np.ndarray | None = None,
    ) -> torch.Tensor:
        frame = self._ensure_cuda_frame(buffer_index)
        assert self._cuda_bgra_frame_wp is not None
        tile_limit = min(int(qpos.shape[0]), int(getattr(self, "_tile_env_count", 1)))
        for tile_idx in range(tile_limit):
            tile_mocap_pos = None if mocap_pos is None else mocap_pos[tile_idx]
            tile_mocap_quat = None if mocap_quat is None else mocap_quat[tile_idx]
            self._set_state_from_arrays(
                qpos[tile_idx],
                qvel[tile_idx],
                mocap_pos=tile_mocap_pos,
                mocap_quat=tile_mocap_quat,
            )
            self._render_current_frame_to_context(env_idx=tile_idx)
            rc = self._ensure_single_render_context()
            row = tile_idx // int(getattr(self, "_tile_cols", 1))
            col = tile_idx % int(getattr(self, "_tile_cols", 1))
            wp.launch(
                _pack_warp_rgb_to_bgra_tile_kernel,
                dim=(self.height, self.width),
                inputs=[
                    rc.rgb_data,
                    rc.rgb_adr,
                    int(getattr(self, "_render_camera_index", 0)),
                    int(self.width),
                    int(col * self.width),
                    int(row * self.height),
                ],
                outputs=[self._cuda_bgra_frame_wp],
            )
        self._clear_unused_cuda_tiles(tile_limit)
        return frame

    def _render_batched_tiled_frame_cuda(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        *,
        buffer_index: int = 0,
        benchmark_sync_device: torch.device | None = None,
        activation_batch: torch.Tensor | None = None,
        mocap_pos: torch.Tensor | None = None,
        mocap_quat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qpos = _require_cuda_render_tensor("qpos", qpos)
        qvel = _require_cuda_render_tensor("qvel", qvel)
        frame = self._ensure_cuda_frame(buffer_index)
        assert self._cuda_bgra_frame_wp is not None

        tile_limit = min(int(qpos.shape[0]), int(getattr(self, "_tile_env_count", 1)))
        model, data, rc = self._ensure_batched_tile_context(tile_limit)
        if not self._use_cuda_render_graph:
            timings: dict[str, float] = {}
            timed_device = benchmark_sync_device
            _time_cuda_stage_ms(
                timings,
                "stage_state",
                timed_device,
                lambda: self._stage_batched_tile_data_gpu(
                    model,
                    data,
                    qpos,
                    qvel,
                    world_count=tile_limit,
                    mocap_pos=mocap_pos,
                    mocap_quat=mocap_quat,
                ),
            )
            _time_cuda_stage_ms(
                timings,
                "refit_bvh",
                timed_device,
                lambda: mjwr.refit_bvh(model, data, rc),
            )
            _time_cuda_stage_ms(
                timings, "render", timed_device, lambda: mjwr.render(model, data, rc)
            )
            fovy = _resolve_camera_fovy(self.model, self._camera_state, self._camera_id)
            for overlay in self._muscle_line_overlays:
                overlay.draw_batch(
                    data=data,
                    rc=rc,
                    camera_id=self._camera_id,
                    width=self.width,
                    height=self.height,
                    fovy=fovy,
                    world_count=tile_limit,
                    activation_batch=activation_batch,
                )
            if getattr(self, "_lbs_mesh_overlay", None) is not None:
                self._lbs_mesh_overlay.draw_batch(
                    data=data,
                    rc=rc,
                    camera_id=self._camera_id,
                    width=self.width,
                    height=self.height,
                    fovy=fovy,
                    world_count=tile_limit,
                    activation_batch=activation_batch,
                )
            _time_cuda_stage_ms(
                timings,
                "pack",
                timed_device,
                lambda: wp.launch(
                    _pack_warp_rgb_to_bgra_tiles_kernel,
                    dim=(tile_limit, self.height, self.width),
                    inputs=[
                        rc.rgb_data,
                        rc.rgb_adr,
                        int(getattr(self, "_render_camera_index", 0)),
                        self.width,
                        self.height,
                        self._tile_cols,
                    ],
                    outputs=[self._cuda_bgra_frame_wp],
                ),
            )
            _time_cuda_stage_ms(
                timings,
                "clear_unused",
                timed_device,
                lambda: self._clear_unused_cuda_tiles(tile_limit),
            )
            if benchmark_sync_device is not None:
                self._last_cuda_stage_timings_ms = timings
                history = getattr(self, "_cuda_stage_timing_history_ms", None)
                if not isinstance(history, list):
                    history = []
                    self._cuda_stage_timing_history_ms = history
                history.append(dict(timings))
            return frame
        graphs = getattr(self, "_cuda_render_graphs", None)
        if graphs is None:
            graphs = {}
            self._cuda_render_graphs = graphs
        entry = graphs.get(int(buffer_index))
        qpos_shape = (tile_limit, int(self.model.nq))
        qvel_shape = (tile_limit, int(self.model.nv))
        if entry is None or tuple(entry.qpos.shape) != qpos_shape:
            entry = _CudaRenderGraphEntry(
                graph=None,
                signature=(),
                qpos=torch.empty(qpos_shape, device=qpos.device, dtype=torch.float32),
                qvel=torch.empty(qvel_shape, device=qvel.device, dtype=torch.float32),
                mocap_pos=(
                    None
                    if mocap_pos is None
                    else torch.empty_like(mocap_pos, dtype=torch.float32)
                ),
                mocap_quat=(
                    None
                    if mocap_quat is None
                    else torch.empty_like(mocap_quat, dtype=torch.float32)
                ),
            )
            self._cuda_render_graphs[int(buffer_index)] = entry
        entry.qpos.copy_(qpos[:tile_limit, : self.model.nq])
        entry.qvel.copy_(qvel[:tile_limit, : self.model.nv])
        if mocap_pos is not None:
            if entry.mocap_pos is None or entry.mocap_pos.shape != mocap_pos.shape:
                entry.mocap_pos = torch.empty_like(mocap_pos, dtype=torch.float32)
                entry.graph = None
            entry.mocap_pos.copy_(mocap_pos)
        elif entry.mocap_pos is not None:
            entry.mocap_pos = None
            entry.graph = None
        if mocap_quat is not None:
            if entry.mocap_quat is None or entry.mocap_quat.shape != mocap_quat.shape:
                entry.mocap_quat = torch.empty_like(mocap_quat, dtype=torch.float32)
                entry.graph = None
            entry.mocap_quat.copy_(mocap_quat)
        elif entry.mocap_quat is not None:
            entry.mocap_quat = None
            entry.graph = None
        self._stage_graph_camera_params()
        use_reference_ghost = entry.mocap_pos is None and entry.mocap_quat is None
        ghost_signature = (
            self._prepare_reference_ghost_graph_inputs(data, tile_limit)
            if use_reference_ghost
            else ("explicit_mocap",)
        )
        fovy = _resolve_camera_fovy(self.model, self._camera_state, self._camera_id)
        if self._muscle_line_overlays:
            for overlay in self._muscle_line_overlays:
                overlay.prepare_batch(
                    data=data,
                    world_count=tile_limit,
                    activation_batch=activation_batch,
                )
        if getattr(self, "_lbs_mesh_overlay", None) is not None:
            self._lbs_mesh_overlay.prepare_batch(
                data=data,
                world_count=tile_limit,
                width=self.width,
                height=self.height,
                activation_batch=activation_batch,
            )
        overlay_signature = tuple(
            overlay.prepared_graph_signature for overlay in self._muscle_line_overlays
        ) + (
            (self._lbs_mesh_overlay.prepared_graph_signature,)
            if self._lbs_mesh_overlay is not None
            else ()
        )
        signature = (
            id(model),
            id(data),
            id(rc),
            id(self._cuda_bgra_frame_wp),
            tile_limit,
            self.width,
            self.height,
            self._tile_cols,
            self._camera_id,
            int(self._camera_state.type),
            int(getattr(self._camera_state, "fixedcamid", -1)),
            int(getattr(self._camera_state, "trackbodyid", -1)),
            bool(self._cam_lookat_ema_alpha is not None),
            float(fovy),
            entry.mocap_pos is not None,
            entry.mocap_quat is not None,
            ghost_signature,
            overlay_signature,
        )

        def _launch_render_chain() -> None:
            self._stage_batched_tile_data_gpu(
                model,
                data,
                entry.qpos,
                entry.qvel,
                world_count=tile_limit,
                mocap_pos=entry.mocap_pos,
                mocap_quat=entry.mocap_quat,
            )
            mjwr.refit_bvh(model, data, rc)
            mjwr.render(model, data, rc)
            for overlay in self._muscle_line_overlays:
                overlay.draw_prepared_batch(
                    data=data,
                    rc=rc,
                    camera_id=self._camera_id,
                    width=self.width,
                    height=self.height,
                    fovy=fovy,
                    world_count=tile_limit,
                )
            if self._lbs_mesh_overlay is not None:
                self._lbs_mesh_overlay.draw_prepared_batch(
                    data=data,
                    rc=rc,
                    camera_id=self._camera_id,
                    width=self.width,
                    height=self.height,
                    fovy=fovy,
                    world_count=tile_limit,
                )
            wp.launch(
                _pack_warp_rgb_to_bgra_tiles_kernel,
                dim=(tile_limit, self.height, self.width),
                inputs=[
                    rc.rgb_data,
                    rc.rgb_adr,
                    int(getattr(self, "_render_camera_index", 0)),
                    int(self.width),
                    int(self.height),
                    int(getattr(self, "_tile_cols", 1)),
                ],
                outputs=[self._cuda_bgra_frame_wp],
            )
            self._clear_unused_cuda_tiles(tile_limit)

        if entry.graph is None or entry.signature != signature:
            smooth_before = wp.to_torch(self._graph_smooth_lookat).clone()
            smooth_valid_before = wp.to_torch(self._graph_smooth_valid).clone()
            _launch_render_chain()
            with wp.ScopedCapture() as capture:
                _launch_render_chain()
            wp.copy(
                self._graph_smooth_lookat,
                wp.from_torch(smooth_before, dtype=wp.vec3, requires_grad=False),
            )
            wp.copy(
                self._graph_smooth_valid,
                wp.from_torch(smooth_valid_before, dtype=wp.int32, requires_grad=False),
            )
            entry.graph = capture.graph
            entry.signature = signature
            self._cuda_graph_capture_count += 1
        if benchmark_sync_device is None:
            wp.capture_launch(entry.graph)
        else:
            stage_timings_ms: dict[str, float] = {}
            self._last_cuda_stage_timings_ms = stage_timings_ms
            _time_cuda_stage_ms(
                stage_timings_ms,
                "render_graph",
                benchmark_sync_device,
                lambda: wp.capture_launch(entry.graph),
            )
            history = getattr(self, "_cuda_stage_timing_history_ms", None)
            if not isinstance(history, list):
                history = []
                self._cuda_stage_timing_history_ms = history
            history.append(dict(stage_timings_ms))
        return frame

    def render_substep_batch(
        self,
        batch: MujocoWarpRenderBatch,
        *,
        overlay_state: dict[str, Any] | None = None,
        episode_length: int = 0,
        step_reward: float = 0.0,
    ) -> np.ndarray:
        del episode_length, step_reward
        if overlay_state is not None:
            raise RuntimeError(
                "MuJoCo-Warp playback does not support reward overlays yet."
            )

        qpos_batch = np.asarray(batch.qpos)
        qvel_batch = np.asarray(batch.qvel)
        if qpos_batch.ndim not in {2, 3} or qvel_batch.ndim != qpos_batch.ndim:
            raise ValueError("qpos/qvel batches must both be rank-2 or rank-3 arrays.")
        if qpos_batch.shape[0] != qvel_batch.shape[0]:
            raise ValueError("qpos/qvel batch sizes must match.")
        if qpos_batch.ndim == 3 and qpos_batch.shape[1] != qvel_batch.shape[1]:
            raise ValueError("qpos/qvel tile batch sizes must match.")

        if self._prefer_cuda_batch_render:

            def _cuda_value(value: Any | None) -> torch.Tensor | None:
                if value is None:
                    return None
                return torch.as_tensor(value, device="cuda", dtype=torch.float32)

            cuda_frames = self.render_substep_batch_cuda(
                MujocoWarpRenderBatch(
                    qpos=_cuda_value(qpos_batch),
                    qvel=_cuda_value(qvel_batch),
                    activation=_cuda_value(batch.activation),
                    mocap_pos=_cuda_value(batch.mocap_pos),
                    mocap_quat=_cuda_value(batch.mocap_quat),
                )
            )
            return np.stack(
                [frame[..., (2, 1, 0)].detach().cpu().numpy() for frame in cuda_frames],
                axis=0,
            )

        frame_count = int(qpos_batch.shape[0])
        frames = np.empty(
            (
                frame_count,
                int(getattr(self, "_output_height", self.height)),
                int(getattr(self, "_output_width", self.width)),
                3,
            ),
            dtype=np.uint8,
        )
        self._configure_camera()
        mocap_pos_batch = (
            None if batch.mocap_pos is None else np.asarray(batch.mocap_pos)
        )
        mocap_quat_batch = (
            None if batch.mocap_quat is None else np.asarray(batch.mocap_quat)
        )
        activation_batch = (
            None if batch.activation is None else np.asarray(batch.activation)
        )
        if qpos_batch.ndim == 3:
            if activation_batch is not None:
                if activation_batch.ndim != 3:
                    raise ValueError(
                        "activation batch must be rank-3 when qpos/qvel are rank-3."
                    )
                if activation_batch.shape[:2] != qpos_batch.shape[:2]:
                    raise ValueError(
                        "activation tile batch shape must match qpos/qvel batch shape."
                    )
            if (
                mocap_pos_batch is not None
                and mocap_pos_batch.shape[:2] != qpos_batch.shape[:2]
            ):
                raise ValueError("mocap_pos tile batch shape must match qpos/qvel.")
            if (
                mocap_quat_batch is not None
                and mocap_quat_batch.shape[:2] != qpos_batch.shape[:2]
            ):
                raise ValueError("mocap_quat tile batch shape must match qpos/qvel.")
            for idx in range(frame_count):
                frames[idx] = self._render_tiled_frame(
                    qpos_batch[idx],
                    qvel_batch[idx],
                    mocap_pos=(
                        None if mocap_pos_batch is None else mocap_pos_batch[idx]
                    ),
                    mocap_quat=(
                        None if mocap_quat_batch is None else mocap_quat_batch[idx]
                    ),
                    activation=(
                        None if activation_batch is None else activation_batch[idx]
                    ),
                )
            return frames

        if activation_batch is not None:
            if activation_batch.ndim != 2:
                raise ValueError(
                    "activation batch must be rank-2 when qpos/qvel are rank-2."
                )
            if activation_batch.shape[0] != qpos_batch.shape[0]:
                raise ValueError("activation batch size must match qpos/qvel.")

        for idx in range(frame_count):
            frames[idx] = self._render_tiled_frame(
                qpos_batch[idx : idx + 1],
                qvel_batch[idx : idx + 1],
                mocap_pos=(
                    None if mocap_pos_batch is None else mocap_pos_batch[idx : idx + 1]
                ),
                mocap_quat=(
                    None
                    if mocap_quat_batch is None
                    else mocap_quat_batch[idx : idx + 1]
                ),
                activation=(
                    None
                    if activation_batch is None
                    else activation_batch[idx : idx + 1]
                ),
            )
        return frames

    def render_substep_batch_cuda(
        self,
        batch: MujocoWarpRenderBatch,
        *,
        overlay_state: dict[str, Any] | None = None,
        episode_length: int = 0,
        step_reward: float = 0.0,
        benchmark_sync_device: torch.device | None = None,
    ) -> list[torch.Tensor]:
        qpos = _require_cuda_render_tensor("qpos", batch.qpos)
        if not qpos.is_cuda:
            return self._render_substep_batch_cuda_impl(
                batch,
                overlay_state=overlay_state,
                episode_length=episode_length,
                step_reward=step_reward,
                benchmark_sync_device=benchmark_sync_device,
            )
        caller_stream = torch.cuda.current_stream(qpos.device)
        if self._render_wp_stream is None or self._render_torch_stream is None:
            self._render_wp_stream = wp.Stream(device=str(qpos.device))
            self._render_torch_stream = torch.cuda.ExternalStream(
                self._render_wp_stream.cuda_stream,
                device=qpos.device,
            )
            self._producer_ready_event = torch.cuda.Event()
            self._render_ready_event = torch.cuda.Event()
        assert self._producer_ready_event is not None
        assert self._render_ready_event is not None
        self._producer_ready_event.record(caller_stream)
        self._render_torch_stream.wait_event(self._producer_ready_event)
        with (
            torch.cuda.stream(self._render_torch_stream),
            wp.ScopedStream(
                self._render_wp_stream,
                sync_enter=False,
                sync_exit=False,
            ),
        ):
            frames = self._render_substep_batch_cuda_impl(
                batch,
                overlay_state=overlay_state,
                episode_length=episode_length,
                step_reward=step_reward,
                benchmark_sync_device=benchmark_sync_device,
            )
            self._render_ready_event.record(self._render_torch_stream)
        caller_stream.wait_event(self._render_ready_event)
        return frames

    def _render_substep_batch_cuda_impl(
        self,
        batch: MujocoWarpRenderBatch,
        *,
        overlay_state: dict[str, Any] | None = None,
        episode_length: int = 0,
        step_reward: float = 0.0,
        benchmark_sync_device: torch.device | None = None,
    ) -> list[torch.Tensor]:
        del episode_length, step_reward
        if overlay_state is not None:
            raise RuntimeError(
                "MuJoCo-Warp playback does not support reward overlays yet."
            )

        qpos_batch = _require_cuda_render_tensor("qpos", batch.qpos)
        qvel_batch = _require_cuda_render_tensor("qvel", batch.qvel)
        if qpos_batch.ndim not in {2, 3} or qvel_batch.ndim != qpos_batch.ndim:
            raise ValueError("qpos/qvel batches must both be rank-2 or rank-3 arrays.")
        if qpos_batch.shape[0] != qvel_batch.shape[0]:
            raise ValueError("qpos/qvel batch sizes must match.")
        if qpos_batch.ndim == 3 and qpos_batch.shape[1] != qvel_batch.shape[1]:
            raise ValueError("qpos/qvel tile batch sizes must match.")

        frames: list[torch.Tensor] = []
        if benchmark_sync_device is not None:
            self._cuda_stage_timing_history_ms = []
        else:
            self._cuda_stage_timing_history_ms = None
        self._configure_camera()
        if qpos_batch.ndim == 3:
            activation_batch = None
            if batch.activation is not None:
                activation_batch = _require_cuda_render_tensor(
                    "activation",
                    batch.activation,
                )
                if activation_batch.ndim != 3:
                    raise ValueError(
                        "activation batch must be rank-3 when qpos/qvel are rank-3."
                    )
                if activation_batch.shape[:2] != qpos_batch.shape[:2]:
                    raise ValueError(
                        "activation tile batch shape must match qpos/qvel batch shape."
                    )
            mocap_pos_batch = None
            if batch.mocap_pos is not None:
                mocap_pos_batch = _require_cuda_render_tensor(
                    "mocap_pos",
                    batch.mocap_pos,
                )
                if mocap_pos_batch.ndim != 4:
                    raise ValueError(
                        "mocap_pos batch must be rank-4 when qpos/qvel are rank-3."
                    )
                if mocap_pos_batch.shape[:2] != qpos_batch.shape[:2]:
                    raise ValueError(
                        "mocap_pos tile batch shape must match qpos/qvel batch shape."
                    )
            mocap_quat_batch = None
            if batch.mocap_quat is not None:
                mocap_quat_batch = _require_cuda_render_tensor(
                    "mocap_quat",
                    batch.mocap_quat,
                )
                if mocap_quat_batch.ndim != 4:
                    raise ValueError(
                        "mocap_quat batch must be rank-4 when qpos/qvel are rank-3."
                    )
                if mocap_quat_batch.shape[:2] != qpos_batch.shape[:2]:
                    raise ValueError(
                        "mocap_quat tile batch shape must match qpos/qvel batch shape."
                    )
            for idx in range(int(qpos_batch.shape[0])):
                frames.append(
                    self._render_batched_tiled_frame_cuda(
                        qpos_batch[idx],
                        qvel_batch[idx],
                        buffer_index=idx,
                        benchmark_sync_device=benchmark_sync_device,
                        activation_batch=(
                            None if activation_batch is None else activation_batch[idx]
                        ),
                        mocap_pos=(
                            None if mocap_pos_batch is None else mocap_pos_batch[idx]
                        ),
                        mocap_quat=(
                            None if mocap_quat_batch is None else mocap_quat_batch[idx]
                        ),
                    )
                )
            return frames

        activation_batch = None
        if batch.activation is not None:
            activation_batch = _require_cuda_render_tensor(
                "activation", batch.activation
            )
            if activation_batch.ndim != 2:
                raise ValueError(
                    "activation batch must be rank-2 when qpos/qvel are rank-2."
                )
            if int(activation_batch.shape[0]) != int(qpos_batch.shape[0]):
                raise ValueError(
                    "activation batch size must match qpos/qvel batch size."
                )
        mocap_pos_batch = None
        if batch.mocap_pos is not None:
            mocap_pos_batch = _require_cuda_render_tensor("mocap_pos", batch.mocap_pos)
            if mocap_pos_batch.ndim != 3:
                raise ValueError(
                    "mocap_pos batch must be rank-3 when qpos/qvel are rank-2."
                )
            if int(mocap_pos_batch.shape[0]) != int(qpos_batch.shape[0]):
                raise ValueError(
                    "mocap_pos batch size must match qpos/qvel batch size."
                )
        mocap_quat_batch = None
        if batch.mocap_quat is not None:
            mocap_quat_batch = _require_cuda_render_tensor(
                "mocap_quat",
                batch.mocap_quat,
            )
            if mocap_quat_batch.ndim != 3:
                raise ValueError(
                    "mocap_quat batch must be rank-3 when qpos/qvel are rank-2."
                )
            if int(mocap_quat_batch.shape[0]) != int(qpos_batch.shape[0]):
                raise ValueError(
                    "mocap_quat batch size must match qpos/qvel batch size."
                )
        for idx in range(int(qpos_batch.shape[0])):
            frames.append(
                self._render_batched_tiled_frame_cuda(
                    qpos_batch[idx : idx + 1],
                    qvel_batch[idx : idx + 1],
                    buffer_index=idx,
                    benchmark_sync_device=benchmark_sync_device,
                    activation_batch=(
                        None
                        if activation_batch is None
                        else activation_batch[idx : idx + 1]
                    ),
                    mocap_pos=(
                        None
                        if mocap_pos_batch is None
                        else mocap_pos_batch[idx : idx + 1]
                    ),
                    mocap_quat=(
                        None
                        if mocap_quat_batch is None
                        else mocap_quat_batch[idx : idx + 1]
                    ),
                )
            )
        return frames
