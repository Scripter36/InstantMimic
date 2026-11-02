from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import torch
from cuda.bindings import runtime as cudart
from mujoco import gl_context as mujoco_gl_context
from OpenGL import GL

from instantmimic.rendering.character import (
    character_geom_group_visibility,
    normalize_character_geom_groups,
    resolve_character_render_mode_for_model,
)
from instantmimic.rendering.mujoco_gl_overlay import MujocoGLOverlayRenderer

logger = logging.getLogger(__name__)


class CudaGLInteropError(RuntimeError):
    """Raised when a MuJoCo OpenGL resource cannot be shared with CUDA."""

    def __init__(self, *, operation: str, error: int) -> None:
        self.error_code = int(error)
        cuda_error = cudart.cudaError_t(self.error_code)
        error_name = cuda_error.name
        status, error_string = cudart.cudaGetErrorString(cuda_error)
        if status == cudart.cudaError_t.cudaSuccess and error_string is not None:
            detail = error_string.decode("utf-8")
        else:
            detail = "unknown CUDA error"
        super().__init__(f"{operation}: {self.error_code} ({error_name}: {detail})")


def _mujoco_gl_nvtx_range(name: str, *, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.cuda.nvtx.range(name)


def mujoco_gl_playback_is_available() -> bool:
    return hasattr(mujoco_gl_context, "GLContext")


def _resolve_mujoco_gl_scene_maxgeom(
    *,
    model: mujoco.MjModel,
    explicit_maxgeom: int | None,
    env_raw: Any,
) -> int:
    if explicit_maxgeom is not None:
        return int(explicit_maxgeom)

    base_maxgeom = max(4096, int(model.ngeom) * 24 + 1024)
    dynamic_gl_sources = getattr(env_raw, "dynamic_gl_render_sources", None)
    if callable(dynamic_gl_sources):
        try:
            if list(dynamic_gl_sources()):
                return base_maxgeom
        except Exception:
            logger.exception("Failed to inspect dynamic GL render sources.")

    dynamic_render_maxgeom = getattr(env_raw, "_dynamic_render_maxgeom", None)
    if not callable(dynamic_render_maxgeom):
        return base_maxgeom

    return base_maxgeom + max(0, int(dynamic_render_maxgeom()))


def _apply_character_render_mode(
    option: mujoco.MjvOption,
    model: mujoco.MjModel,
    render_mode: str,
    *,
    env_raw: Any | None = None,
) -> None:
    """Set MuJoCo geom groups for playback character rendering."""
    render_mode = resolve_character_render_mode_for_model(
        model,
        render_mode,
        env_raw=env_raw,
    )
    normalize_character_geom_groups(model, env_raw=env_raw)
    option.geomgroup[1] = 1
    option.flags[int(mujoco.mjtVisFlag.mjVIS_TRANSPARENT)] = 0
    option.flags[int(mujoco.mjtVisFlag.mjVIS_CONVEXHULL)] = 0
    for group, visible in character_geom_group_visibility(render_mode).items():
        option.geomgroup[group] = int(visible)


@dataclass
class MujocoGLRenderBatch:
    qpos: np.ndarray
    qvel: np.ndarray
    mocap_pos: np.ndarray | None = None
    mocap_quat: np.ndarray | None = None


def resolve_camera_body_tracking(model, camera_count, body_names=(), offsets=()):
    """Resolve optional per-view centroids before opening graphics resources."""
    names = body_names or ((),) * camera_count
    shifts = offsets or ((0.0, 0.0, 0.0),) * camera_count
    if len(names) != camera_count or len(shifts) != camera_count:
        raise ValueError("camera tracking layout must match the camera count")
    result = []
    for bodies, offset in zip(names, shifts, strict=True):
        ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in bodies
        ]
        if any(body < 0 for body in ids):
            raise ValueError("camera track_bodies names an unknown model body")
        shift = np.asarray(offset, dtype=float)
        if shift.shape != (3,) or not np.isfinite(shift).all():
            raise ValueError("camera lookat_offset must contain three finite values")
        if not ids and np.any(shift):
            raise ValueError("camera lookat_offset requires track_bodies")
        result.append((np.asarray(ids, dtype=int), shift))
    return tuple(result)


def update_body_tracking_camera(camera, data, body_ids, offset):
    """Follow rendered body positions without changing view pose or physics."""
    if not len(body_ids):
        return
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.trackbodyid = -1
    camera.lookat[:] = np.mean(data.xpos[body_ids], axis=0) + offset


class MujocoGLPlaybackRenderer:
    def __init__(
        self,
        *,
        env_raw: Any,
        model: mujoco.MjModel,
        width: int,
        height: int,
        data: mujoco.MjData | None = None,
        camera: str | int | mujoco.MjvCamera | None = None,
        cameras: Sequence[str | int | mujoco.MjvCamera | None] | None = None,
        camera_layout: str = "horizontal",
        camera_track_bodies: Sequence[Sequence[str]] = (),
        camera_lookat_offsets: Sequence[Sequence[float]] = (),
        maxgeom: int | None = None,
        overlay_renderer: MujocoGLOverlayRenderer | None = None,
        cam_lookat_ema_alpha: float | None = None,
        character_render_mode: str = "mesh",
    ) -> None:
        if not mujoco_gl_playback_is_available():
            raise RuntimeError("MuJoCo GL playback backend is unavailable.")
        self._env_raw = env_raw
        self._nvtx_enabled = torch.cuda.is_available()
        self.model = model
        self.data = data if data is not None else mujoco.MjData(model)
        self.width = int(width)
        self.height = int(height)
        self._camera = camera
        self._cameras = tuple(cameras) if cameras is not None else (camera,)
        if not self._cameras:
            raise ValueError("At least one playback camera is required.")
        self._camera_tracks = resolve_camera_body_tracking(
            model, len(self._cameras), camera_track_bodies, camera_lookat_offsets
        )
        self._camera_layout = str(camera_layout).strip().lower()
        if self._camera_layout not in {"horizontal", "vertical"}:
            raise ValueError("camera_layout must be 'horizontal' or 'vertical'.")
        self._cam_lookat_ema_alpha = cam_lookat_ema_alpha
        self._smooth_lookat: np.ndarray | None = None
        self._smooth_lookats: list[np.ndarray | None] = [None] * len(self._cameras)
        self._overlay_renderer = overlay_renderer
        camera_cols = len(self._cameras) if self._camera_layout == "horizontal" else 1
        camera_rows = len(self._cameras) if self._camera_layout == "vertical" else 1
        self._scene_width = int(width) * camera_cols
        self._output_height = int(height) * camera_rows
        self._panel_width = int(getattr(overlay_renderer, "panel_width", 0))
        self._output_width = int(self._scene_width) + int(self._panel_width)
        self.model.vis.global_.offwidth = int(self._output_width)
        self.model.vis.global_.offheight = int(self._output_height)
        self._gl = mujoco_gl_context.GLContext(self._output_width, self._output_height)
        self._gl.make_current()
        self._scene = mujoco.MjvScene(
            self.model,
            maxgeom=_resolve_mujoco_gl_scene_maxgeom(
                model=self.model,
                explicit_maxgeom=maxgeom,
                env_raw=env_raw,
            ),
        )
        self._option = mujoco.MjvOption()
        _apply_character_render_mode(
            self._option,
            self.model,
            character_render_mode,
            env_raw=env_raw,
        )
        self._perturb = mujoco.MjvPerturb()
        self._camera_states = [mujoco.MjvCamera() for _ in self._cameras]
        self._camera_state = self._camera_states[0]
        self._context = mujoco.MjrContext(
            self.model,
            mujoco.mjtFontScale.mjFONTSCALE_100,
        )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)
        self._full_rect = mujoco.MjrRect(
            0,
            0,
            self._output_width,
            self._output_height,
        )
        self._scene_rects = []
        for index in range(len(self._cameras)):
            if self._camera_layout == "horizontal":
                x, y = index * int(width), 0
            else:
                # Config order is top-to-bottom; MuJoCo rectangles use a
                # bottom-left origin.
                x, y = 0, (len(self._cameras) - index - 1) * int(height)
            self._scene_rects.append(mujoco.MjrRect(x, y, int(width), int(height)))
        self._scene_rect = self._scene_rects[0]
        self._cuda_rgba_frame: torch.Tensor | None = None
        self._cuda_argb_frame: torch.Tensor | None = None
        self._cuda_gl_resource = None
        self._dynamic_gl_renderers = self._create_dynamic_gl_renderers()
        if len(self._cameras) > 1 and self._dynamic_gl_renderers:
            raise RuntimeError(
                "Multi-camera playback does not yet support dynamic GL render sources."
            )

    def enable_cuda_interop(self) -> None:
        """Register the resolved GL renderbuffer for CUDA-backed video encoding."""
        if self._cuda_gl_resource is not None:
            return
        err, resource = cudart.cudaGraphicsGLRegisterImage(
            int(self._context.offColor_r),
            GL.GL_RENDERBUFFER,
            cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly,
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise CudaGLInteropError(
                operation="Failed to register GL renderbuffer with CUDA",
                error=int(err),
            )
        self._cuda_gl_resource = resource

    def _create_dynamic_gl_renderers(self) -> list[object]:
        source_getter = getattr(self._env_raw, "dynamic_gl_render_sources", None)
        if not callable(source_getter):
            return []
        sources = list(source_getter())
        if not sources:
            return []

        from instantmimic.rendering.muscle_line_gl_renderer import MuscleLineGLRenderer

        renderers: list[object] = []
        for source in sources:
            try:
                renderers.append(MuscleLineGLRenderer(source))
            except Exception:
                logger.exception("Failed to initialize dynamic GL renderer.")
        if renderers:
            logger.info("Initialized %d dynamic GL renderer(s).", len(renderers))
        return renderers

    def _append_dynamic_render_geoms(self) -> None:
        if getattr(self, "_dynamic_gl_renderers", None):
            append_aux_dynamic_render_geoms = getattr(
                self._env_raw,
                "_append_aux_dynamic_render_geoms",
                None,
            )
            if callable(append_aux_dynamic_render_geoms):
                append_aux_dynamic_render_geoms(
                    scene=self._scene,
                    render_model=self.model,
                    render_data=self.data,
                )
            return
        append_dynamic_render_geoms = getattr(
            self._env_raw,
            "_append_dynamic_render_geoms",
            None,
        )
        if callable(append_dynamic_render_geoms):
            append_dynamic_render_geoms(
                scene=self._scene,
                render_model=self.model,
                render_data=self.data,
            )

    def _draw_dynamic_gl_renderers(self) -> None:
        for renderer in getattr(self, "_dynamic_gl_renderers", ()):
            renderer.draw(
                render_data=self.data,
                scene=self._scene,
                width=self._scene_width,
                height=self.height,
                framebuffer=int(self._context.offFBO),
            )

    def close(self) -> None:
        for renderer in getattr(self, "_dynamic_gl_renderers", ()):
            close = getattr(renderer, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - cleanup must continue across renderers.
                    logger.debug(
                        "Dynamic GL renderer cleanup failed: %s",
                        exc,
                        exc_info=True,
                    )
        self._dynamic_gl_renderers = []
        resource = getattr(self, "_cuda_gl_resource", None)
        if resource is not None:
            try:
                cudart.cudaGraphicsUnregisterResource(resource)
            except Exception as exc:  # noqa: BLE001 - CUDA resource cleanup is best-effort.
                logger.debug(
                    "CUDA GL resource unregister failed: %s",
                    exc,
                    exc_info=True,
                )
            self._cuda_gl_resource = None
        context = getattr(self, "_context", None)
        if context is not None:
            context.free()
            self._context = None
        gl = getattr(self, "_gl", None)
        if gl is not None:
            del self._gl

    def __del__(self) -> None:
        try:
            self.close()
        except Exception as exc:  # noqa: BLE001 - destructors must not raise during GC.
            logger.debug("MuJoCo GL playback cleanup failed: %s", exc, exc_info=True)

    def _configure_camera_state(self, camera: Any, camera_state: Any) -> None:
        ema_active = self._cam_lookat_ema_alpha is not None
        if camera is None:
            camera_state.trackbodyid = int(getattr(self._env_raw, "torso_body_idx", 1))
            camera_state.distance = 3.0
            camera_state.elevation = -20.0
            camera_state.azimuth = 135.0
            if ema_active:
                camera_state.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                camera_state.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            return
        if isinstance(camera, mujoco.MjvCamera):
            camera_state.fixedcamid = int(camera.fixedcamid)
            trackbodyid = int(getattr(camera, "trackbodyid", -1))
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
            camera_state.trackbodyid = int(trackbodyid)
            camera_state.distance = float(camera.distance)
            camera_state.elevation = float(camera.elevation)
            camera_state.azimuth = float(camera.azimuth)
            if hasattr(camera_state, "orthographic") and hasattr(
                camera, "orthographic"
            ):
                camera_state.orthographic = camera.orthographic
            if hasattr(camera_state, "lookat") and hasattr(camera, "lookat"):
                camera_state.lookat[:] = camera.lookat
            is_tracking = camera.type in (
                mujoco.mjtCamera.mjCAMERA_TRACKING,
                int(mujoco.mjtCamera.mjCAMERA_TRACKING),
            )
            if ema_active and is_tracking:
                camera_state.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                camera_state.type = camera.type
            return
        if isinstance(camera, str):
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera
            )
            if camera_id < 0:
                raise ValueError(f'Camera "{camera}" not found in render model.')
        else:
            camera_id = int(camera)
        camera_state.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera_state.fixedcamid = camera_id

    def _configure_camera(self) -> None:
        cameras = self._cameras if hasattr(self, "_cameras") else (self._camera,)
        camera_states = (
            self._camera_states
            if hasattr(self, "_camera_states")
            else (self._camera_state,)
        )
        for camera, camera_state in zip(cameras, camera_states, strict=True):
            self._configure_camera_state(camera, camera_state)

    def _update_smooth_lookat(self, camera_index: int = 0) -> None:
        """Apply EMA smoothing to the camera lookat position."""
        if self._cam_lookat_ema_alpha is None:
            return
        camera_states = (
            self._camera_states
            if hasattr(self, "_camera_states")
            else (self._camera_state,)
        )
        camera_state = camera_states[camera_index]
        if int(camera_state.trackbodyid) < 0:
            return
        body_pos = self.data.xpos[camera_state.trackbodyid].copy()
        smooth_lookats = getattr(self, "_smooth_lookats", None)
        if smooth_lookats is None:
            smooth_lookats = [self._smooth_lookat]
            self._smooth_lookats = smooth_lookats
        smooth_lookat = smooth_lookats[camera_index]
        if smooth_lookat is None:
            smooth_lookat = body_pos
        else:
            alpha = self._cam_lookat_ema_alpha
            smooth_lookat += alpha * (body_pos - smooth_lookat)
        smooth_lookats[camera_index] = smooth_lookat
        if camera_index == 0:
            self._smooth_lookat = smooth_lookat
        camera_state.lookat[:] = smooth_lookat

    def reset_smooth_lookat(self) -> None:
        """Reset the EMA state (call on episode boundaries)."""
        self._smooth_lookat = None
        if hasattr(self, "_smooth_lookats"):
            self._smooth_lookats = [None] * len(self._smooth_lookats)

    def _render_camera_views(self) -> None:
        camera_states = (
            self._camera_states
            if hasattr(self, "_camera_states")
            else (self._camera_state,)
        )
        scene_rects = (
            self._scene_rects if hasattr(self, "_scene_rects") else (self._scene_rect,)
        )
        for camera_index, (camera_state, scene_rect) in enumerate(
            zip(camera_states, scene_rects, strict=True)
        ):
            self._update_smooth_lookat(camera_index)
            if getattr(self, "_camera_tracks", ()):
                update_body_tracking_camera(
                    camera_state, self.data, *self._camera_tracks[camera_index]
                )
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self._option,
                self._perturb,
                camera_state,
                mujoco.mjtCatBit.mjCAT_ALL,
                self._scene,
            )
            self._append_dynamic_render_geoms()
            mujoco.mjr_render(scene_rect, self._scene, self._context)
            self._draw_dynamic_gl_renderers()

    def _ensure_cuda_buffers(self) -> None:
        if self._cuda_rgba_frame is not None and self._cuda_argb_frame is not None:
            return
        self._cuda_rgba_frame = torch.empty(
            (getattr(self, "_output_height", self.height), self._output_width, 4),
            device="cuda",
            dtype=torch.uint8,
        )
        self._cuda_argb_frame = torch.empty_like(self._cuda_rgba_frame)

    def _copy_current_frame_to_cuda(self) -> torch.Tensor:
        self.enable_cuda_interop()
        self._ensure_cuda_buffers()
        assert self._cuda_rgba_frame is not None
        assert self._cuda_argb_frame is not None
        with _mujoco_gl_nvtx_range("mujoco_gl_resolve", enabled=self._nvtx_enabled):
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, int(self._context.offFBO))
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, int(self._context.offFBO_r))
            GL.glBlitFramebuffer(
                0,
                0,
                int(self._output_width),
                int(getattr(self, "_output_height", self.height)),
                0,
                0,
                int(self._output_width),
                int(getattr(self, "_output_height", self.height)),
                GL.GL_COLOR_BUFFER_BIT,
                GL.GL_NEAREST,
            )
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(self._context.offFBO))
            GL.glFinish()
        with _mujoco_gl_nvtx_range("mujoco_gl_cuda_map", enabled=self._nvtx_enabled):
            (err,) = cudart.cudaGraphicsMapResources(1, self._cuda_gl_resource, 0)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to map GL framebuffer into CUDA: {err}")
        try:
            with _mujoco_gl_nvtx_range(
                "mujoco_gl_cuda_copy", enabled=self._nvtx_enabled
            ):
                err, mapped_array = cudart.cudaGraphicsSubResourceGetMappedArray(
                    self._cuda_gl_resource,
                    0,
                    0,
                )
                if err != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(
                        f"Failed to get mapped CUDA array from GL framebuffer: {err}"
                    )
                (err,) = cudart.cudaMemcpy2DFromArray(
                    self._cuda_rgba_frame.data_ptr(),
                    int(
                        self._cuda_rgba_frame.stride(0)
                        * self._cuda_rgba_frame.element_size()
                    ),
                    mapped_array,
                    0,
                    0,
                    int(self._output_width * 4),
                    int(getattr(self, "_output_height", self.height)),
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                )
                if err != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(
                        f"Failed to copy GL framebuffer into CUDA tensor: {err}"
                    )
        finally:
            with _mujoco_gl_nvtx_range(
                "mujoco_gl_cuda_unmap", enabled=self._nvtx_enabled
            ):
                (err,) = cudart.cudaGraphicsUnmapResources(1, self._cuda_gl_resource, 0)
                if err != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(
                        f"Failed to unmap GL framebuffer from CUDA: {err}"
                    )

        # PyNvVideoCodec's GPU "ARGB" path expects the CUDA tensor in BGRA byte order.
        # The mapped MuJoCo framebuffer copy arrives as RGBA, so swap R/B here.
        with _mujoco_gl_nvtx_range(
            "mujoco_gl_channel_pack", enabled=self._nvtx_enabled
        ):
            flipped = self._cuda_rgba_frame.flip(0)
            self._cuda_argb_frame[..., 0].copy_(flipped[..., 2])  # B
            self._cuda_argb_frame[..., 1].copy_(flipped[..., 1])  # G
            self._cuda_argb_frame[..., 2].copy_(flipped[..., 0])  # R
            self._cuda_argb_frame[..., 3].copy_(flipped[..., 3])  # A
        return self._cuda_argb_frame

    def render_substep_batch(
        self,
        batch: MujocoGLRenderBatch,
        *,
        overlay_state: dict[str, Any] | None = None,
        episode_length: int = 0,
        step_reward: float = 0.0,
    ) -> np.ndarray:
        qpos_batch = np.asarray(batch.qpos)
        qvel_batch = np.asarray(batch.qvel)
        if qpos_batch.ndim != 2 or qvel_batch.ndim != 2:
            raise ValueError("qpos/qvel batches must be rank-2 arrays.")
        if qpos_batch.shape[0] != qvel_batch.shape[0]:
            raise ValueError("qpos/qvel batch sizes must match.")
        frame_count = int(qpos_batch.shape[0])
        output_height = int(getattr(self, "_output_height", self.height))
        frames = np.empty(
            (frame_count, output_height, self._output_width, 3),
            dtype=np.uint8,
        )
        scratch = np.empty((output_height, self._output_width, 3), dtype=np.uint8)
        self._configure_camera()
        for idx in range(frame_count):
            self.data.qpos[: self.model.nq] = qpos_batch[idx, : self.model.nq]
            self.data.qvel[: self.model.nv] = qvel_batch[idx, : self.model.nv]
            if batch.mocap_pos is not None:
                self.data.mocap_pos[:] = np.asarray(batch.mocap_pos)[idx]
            if batch.mocap_quat is not None:
                self.data.mocap_quat[:] = np.asarray(batch.mocap_quat)[idx]
            update_render_data = getattr(self._env_raw, "_update_render_data", None)
            if callable(update_render_data):
                update_render_data()
            upload_dynamic_render_meshes = getattr(
                self._env_raw,
                "_upload_dynamic_render_meshes",
                None,
            )
            if callable(upload_dynamic_render_meshes):
                upload_dynamic_render_meshes(mjr_context=self._context)
            mujoco.mj_forward(self.model, self.data)
            mujoco.mjr_rectangle(
                self._full_rect,
                13.0 / 255.0,
                17.0 / 255.0,
                23.0 / 255.0,
                1.0,
            )
            self._render_camera_views()
            if self._overlay_renderer is not None and overlay_state is not None:
                self._overlay_renderer.draw(
                    context=self._context,
                    full_height=output_height,
                    scene_width=self._scene_width,
                    overlay_state=overlay_state,
                    episode_length=episode_length,
                    step_reward=step_reward,
                )
            mujoco.mjr_readPixels(scratch, None, self._full_rect, self._context)
            frames[idx] = scratch[::-1]
        return frames

    def render_substep_batch_cuda(
        self,
        batch: MujocoGLRenderBatch,
        *,
        overlay_state: dict[str, Any] | None = None,
        episode_length: int = 0,
        step_reward: float = 0.0,
    ) -> list[torch.Tensor]:
        qpos_batch = np.asarray(batch.qpos)
        qvel_batch = np.asarray(batch.qvel)
        if qpos_batch.ndim != 2 or qvel_batch.ndim != 2:
            raise ValueError("qpos/qvel batches must be rank-2 arrays.")
        if qpos_batch.shape[0] != qvel_batch.shape[0]:
            raise ValueError("qpos/qvel batch sizes must match.")
        frame_count = int(qpos_batch.shape[0])
        frames: list[torch.Tensor] = []
        self._configure_camera()
        for idx in range(frame_count):
            with _mujoco_gl_nvtx_range(
                "mujoco_gl_state_copy", enabled=self._nvtx_enabled
            ):
                self.data.qpos[: self.model.nq] = qpos_batch[idx, : self.model.nq]
                self.data.qvel[: self.model.nv] = qvel_batch[idx, : self.model.nv]
                if batch.mocap_pos is not None:
                    self.data.mocap_pos[:] = np.asarray(batch.mocap_pos)[idx]
                if batch.mocap_quat is not None:
                    self.data.mocap_quat[:] = np.asarray(batch.mocap_quat)[idx]
            update_render_data = getattr(self._env_raw, "_update_render_data", None)
            if callable(update_render_data):
                with _mujoco_gl_nvtx_range(
                    "mujoco_gl_update_render_data", enabled=self._nvtx_enabled
                ):
                    update_render_data()
            upload_dynamic_render_meshes = getattr(
                self._env_raw,
                "_upload_dynamic_render_meshes",
                None,
            )
            if callable(upload_dynamic_render_meshes):
                with _mujoco_gl_nvtx_range(
                    "mujoco_gl_upload_dynamic_meshes", enabled=self._nvtx_enabled
                ):
                    upload_dynamic_render_meshes(mjr_context=self._context)
            with _mujoco_gl_nvtx_range("mujoco_gl_forward", enabled=self._nvtx_enabled):
                mujoco.mj_forward(self.model, self.data)
            with _mujoco_gl_nvtx_range("mujoco_gl_clear", enabled=self._nvtx_enabled):
                mujoco.mjr_rectangle(
                    self._full_rect,
                    13.0 / 255.0,
                    17.0 / 255.0,
                    23.0 / 255.0,
                    1.0,
                )
            with _mujoco_gl_nvtx_range(
                "mujoco_gl_render_views", enabled=self._nvtx_enabled
            ):
                self._render_camera_views()
            if self._overlay_renderer is not None and overlay_state is not None:
                with _mujoco_gl_nvtx_range(
                    "mujoco_gl_overlay", enabled=self._nvtx_enabled
                ):
                    self._overlay_renderer.draw(
                        context=self._context,
                        full_height=int(getattr(self, "_output_height", self.height)),
                        scene_width=self._scene_width,
                        overlay_state=overlay_state,
                        episode_length=episode_length,
                        step_reward=step_reward,
                    )
            with _mujoco_gl_nvtx_range(
                "mujoco_gl_frame_copy", enabled=self._nvtx_enabled
            ):
                frames.append(self._copy_current_frame_to_cuda().clone())
        return frames
