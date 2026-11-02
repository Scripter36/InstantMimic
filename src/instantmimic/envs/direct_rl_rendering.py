from __future__ import annotations

from logging import getLogger
from typing import Any

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from instantmimic.envs.render_schedule import resolve_substep_render_schedule
from instantmimic.train.full_video_recorder import (
    select_train_full_video_frame_batch,
)

logger = getLogger(__name__)


@wp.kernel(enable_backward=False)
def _copy_array2d_prefix_to_array3d_kernel(
    src: wp.array2d(dtype=float),
    dst_row_count: int,
    dst: wp.array3d(dtype=float),
    dst_plane: int,
):
    row_idx, col_idx = wp.tid()
    if row_idx < dst_row_count:
        dst[dst_plane, row_idx, col_idx] = src[row_idx, col_idx]


@wp.kernel(enable_backward=False)
def _copy_array2d_prefix_to_array4d_kernel(
    src: wp.array2d(dtype=float),
    dst_row_count: int,
    dst: wp.array4d(dtype=float),
    dst_step: int,
    dst_substep: int,
):
    row_idx, col_idx = wp.tid()
    if row_idx < dst_row_count:
        dst[dst_step, dst_substep, row_idx, col_idx] = src[row_idx, col_idx]


@wp.kernel(enable_backward=False)
def _copy_vec3_array2d_prefix_to_array4d_kernel(
    src: wp.array2d(dtype=wp.vec3),
    dst_row_count: int,
    dst: wp.array4d(dtype=wp.vec3),
    dst_step: int,
    dst_substep: int,
):
    row_idx, col_idx = wp.tid()
    if row_idx < dst_row_count:
        dst[dst_step, dst_substep, row_idx, col_idx] = src[row_idx, col_idx]


@wp.kernel(enable_backward=False)
def _copy_quat_array2d_prefix_to_array4d_kernel(
    src: wp.array2d(dtype=wp.quat),
    dst_row_count: int,
    dst: wp.array4d(dtype=wp.quat),
    dst_step: int,
    dst_substep: int,
):
    row_idx, col_idx = wp.tid()
    if row_idx < dst_row_count:
        dst[dst_step, dst_substep, row_idx, col_idx] = src[row_idx, col_idx]


def _substep_render_frame_indices(
    *,
    decimation: int,
    render_interval: int,
    phase_before_step: int,
) -> tuple[int, ...]:
    return resolve_substep_render_schedule(
        decimation=decimation,
        render_interval=render_interval,
        phase_before_step=phase_before_step,
    ).frame_indices


def _resolve_render_tracking_body_name(model: mujoco.MjModel) -> str | None:
    if int(getattr(model, "nbody", 0)) > 1:
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1)
        if body_name:
            return str(body_name)
    for candidate in ("Torso", "torso", "Pelvis", "pelvis"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate) >= 0:
            return candidate
    if int(getattr(model, "nbody", 0)) <= 1:
        return None
    return str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1))


class DirectRLRenderRuntime:
    """Rendering and playback render-state capture for DirectRLEnv."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self.collect_substeps = False
        self.capture_env_idx = 0
        self.substep_phase = 0
        self.phase_before_last_step = 0
        self.substep_qpos_buf: torch.Tensor | None = None
        self.substep_qvel_buf: torch.Tensor | None = None
        self.substep_qpos_wp = None
        self.substep_qvel_wp = None
        self.train_full_video_enabled = False
        self.train_full_video_current_step = 0
        self.train_full_video_step_count = 0
        self.train_full_video_substep_count = 1
        self.train_full_video_env_count = 1
        self.train_full_video_qpos_buf: torch.Tensor | None = None
        self.train_full_video_qvel_buf: torch.Tensor | None = None
        self.train_full_video_activation_buf: torch.Tensor | None = None
        self.train_full_video_step_activation: torch.Tensor | None = None
        self.train_full_video_step_activation_staged = False
        self.train_full_video_activation_sample_count = 0
        self.train_full_video_qpos_wp = None
        self.train_full_video_qvel_wp = None
        self.train_full_video_mocap_pos_buf: torch.Tensor | None = None
        self.train_full_video_mocap_quat_buf: torch.Tensor | None = None
        self.train_full_video_mocap_pos_wp = None
        self.train_full_video_mocap_quat_wp = None
        self.tracking_body_name: str | None = None

    def configure_spec_visuals(self, spec: mujoco.MjSpec) -> None:
        env = self._env
        if len(spec.cameras) == 0:
            camera = spec.worldbody.add_camera(name="__instantmimic_playback_camera")
            camera.fovy = spec.visual.global_.fovy
        if env.render_mode != "rgb_array":
            return
        spec.visual.global_.offwidth = env.cfg.viewer.resolution[0]
        spec.visual.global_.offheight = env.cfg.viewer.resolution[1]

    def initialize_for_model(self) -> None:
        env = self._env
        self.tracking_body_name = _resolve_render_tracking_body_name(env.model)
        env._render_tracking_body_name = self.tracking_body_name
        if env.render_mode != "rgb_array":
            return

        env.camera = mujoco.MjvCamera()
        env.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        env.camera.trackbodyid = max(
            0,
            int(
                mujoco.mj_name2id(
                    env.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.tracking_body_name,
                )
            )
            if self.tracking_body_name is not None
            else 1,
        )
        env.camera.distance = 3.0
        env.camera.elevation = -20
        env.camera.azimuth = 135.0

        render_result = env._build_render_model()
        if render_result is not None:
            env.render_model, env.render_data = render_result
        else:
            env.render_model = env.model
            env.render_data = env.data

        env.renderer = mujoco.Renderer(
            env.render_model,
            height=env.cfg.viewer.resolution[1],
            width=env.cfg.viewer.resolution[0],
        )

    def render(self, recompute: bool = False) -> np.ndarray | list[np.ndarray] | None:
        del recompute
        env = self._env
        if env.render_mode == "human" or env.render_mode is None:
            return None
        if env.render_mode != "rgb_array":
            raise NotImplementedError(
                f"Render mode {env.render_mode!r} is not supported. Please use: {env.metadata['render_modes']}."
            )
        if self.should_collect_substeps():
            frames = self.render_substep_frames()
            if len(frames) > 1:
                return frames
            if len(frames) == 1:
                return frames[0]
        return self.render_cpu()

    def set_collect_render_substeps(self, enabled: bool) -> bool:
        env = self._env
        enabled = bool(enabled)
        if enabled and int(env.cfg.sim.render_interval) >= int(env.cfg.decimation):
            logger.info(
                "Substep render capture disabled because render_interval (%d) >= decimation (%d).",
                int(env.cfg.sim.render_interval),
                int(env.cfg.decimation),
            )
            enabled = False
        if enabled:
            self.ensure_substep_buffers()
        self.collect_substeps = enabled
        env.metadata["render_fps"] = self.get_render_output_fps()
        return self.collect_substeps

    def get_render_output_fps(self) -> float:
        env = self._env
        if self.should_collect_substeps():
            render_dt = env.physics_dt * max(1, int(env.cfg.sim.render_interval))
            return 1.0 / max(render_dt, 1.0e-6)
        return 1.0 / max(env.step_dt, 1.0e-6)

    def ensure_substep_buffers(self) -> None:
        if self.substep_qpos_buf is not None and self.substep_qvel_buf is not None:
            return
        env = self._env
        qpos_view = wp.to_torch(env.mjw_data.qpos)
        qvel_view = wp.to_torch(env.mjw_data.qvel)
        env_count = int(qpos_view.shape[0])
        self.substep_qpos_buf = torch.empty(
            (int(env.cfg.decimation), env_count, int(env.model.nq)),
            device=qpos_view.device,
            dtype=qpos_view.dtype,
        )
        self.substep_qvel_buf = torch.empty(
            (int(env.cfg.decimation), env_count, int(env.model.nv)),
            device=qvel_view.device,
            dtype=qvel_view.dtype,
        )
        self.substep_qpos_wp = wp.from_torch(self.substep_qpos_buf)
        self.substep_qvel_wp = wp.from_torch(self.substep_qvel_buf)

    def should_collect_substeps(self) -> bool:
        env = self._env
        return (
            bool(self.collect_substeps)
            and int(env.cfg.decimation) > 1
            and int(env.cfg.sim.render_interval) < int(env.cfg.decimation)
            and self.substep_qpos_buf is not None
            and self.substep_qvel_buf is not None
        )

    def prepare_substep_render_capture(self) -> None:
        if not self.should_collect_substeps():
            return
        self.phase_before_last_step = self.substep_phase

    def finalize_substep_render_capture(self) -> None:
        if not self.should_collect_substeps():
            return
        env = self._env
        render_interval = max(1, int(env.cfg.sim.render_interval))
        self.substep_phase = (
            self.substep_phase + int(env.cfg.decimation)
        ) % render_interval

    def capture_substep_state(self, substep_idx: int) -> None:
        if (
            not self.should_collect_substeps()
            and not self.should_collect_train_full_video_substeps()
        ):
            return
        env = self._env
        qpos_view = wp.to_torch(env.mjw_data.qpos)
        qvel_view = wp.to_torch(env.mjw_data.qvel)
        qpos_env_count = int(qpos_view.shape[0])
        qvel_env_count = int(qvel_view.shape[0])
        if self.should_collect_substeps():
            assert self.substep_qpos_wp is not None
            assert self.substep_qvel_wp is not None
            wp.launch(
                _copy_array2d_prefix_to_array3d_kernel,
                dim=(qpos_env_count, int(env.model.nq)),
                inputs=[
                    env.mjw_data.qpos,
                    qpos_env_count,
                    self.substep_qpos_wp,
                    int(substep_idx),
                ],
            )
            wp.launch(
                _copy_array2d_prefix_to_array3d_kernel,
                dim=(qvel_env_count, int(env.model.nv)),
                inputs=[
                    env.mjw_data.qvel,
                    qvel_env_count,
                    self.substep_qvel_wp,
                    int(substep_idx),
                ],
            )
        if self.should_collect_train_full_video_substeps():
            self._capture_train_full_video_state(
                step_idx=int(self.train_full_video_current_step),
                substep_idx=int(substep_idx),
                qpos_env_count=qpos_env_count,
                qvel_env_count=qvel_env_count,
                activation=None,
            )

    def configure_train_full_video_capture(
        self,
        *,
        rollout_steps: int,
        env_count: int,
    ) -> None:
        env = self._env
        qpos_view = wp.to_torch(env.mjw_data.qpos)
        qvel_view = wp.to_torch(env.mjw_data.qvel)
        mocap_pos = getattr(env.mjw_data, "mocap_pos", None)
        mocap_quat = getattr(env.mjw_data, "mocap_quat", None)
        mocap_pos_view = None if mocap_pos is None else wp.to_torch(mocap_pos)
        mocap_quat_view = None if mocap_quat is None else wp.to_torch(mocap_quat)
        rollout_steps = max(1, int(rollout_steps))
        requested_env_count = max(1, int(env_count))
        available_env_count = int(qpos_view.shape[0])
        capture_env_count = max(1, min(requested_env_count, available_env_count))
        decimation = max(1, int(env.cfg.decimation))
        render_interval = max(1, int(env.cfg.sim.render_interval))
        substep_count = decimation if render_interval < decimation else 1
        self.train_full_video_enabled = True
        self.train_full_video_current_step = 0
        self.train_full_video_step_count = 0
        self.train_full_video_substep_count = int(substep_count)
        self.train_full_video_env_count = int(capture_env_count)
        self.train_full_video_qpos_buf = torch.empty(
            (
                rollout_steps,
                int(substep_count),
                int(capture_env_count),
                int(env.model.nq),
            ),
            device=qpos_view.device,
            dtype=qpos_view.dtype,
        )
        self.train_full_video_qvel_buf = torch.empty(
            (
                rollout_steps,
                int(substep_count),
                int(capture_env_count),
                int(env.model.nv),
            ),
            device=qvel_view.device,
            dtype=qvel_view.dtype,
        )
        self.train_full_video_mocap_pos_buf = None
        self.train_full_video_mocap_quat_buf = None
        self.train_full_video_mocap_pos_wp = None
        self.train_full_video_mocap_quat_wp = None
        if (
            mocap_pos_view is not None
            and mocap_quat_view is not None
            and int(mocap_pos_view.shape[1]) > 0
        ):
            self.train_full_video_mocap_pos_buf = torch.empty(
                (
                    rollout_steps,
                    int(substep_count),
                    int(capture_env_count),
                    int(mocap_pos_view.shape[1]),
                    3,
                ),
                device=mocap_pos_view.device,
                dtype=mocap_pos_view.dtype,
            )
            self.train_full_video_mocap_quat_buf = torch.empty(
                (
                    rollout_steps,
                    int(substep_count),
                    int(capture_env_count),
                    int(mocap_quat_view.shape[1]),
                    4,
                ),
                device=mocap_quat_view.device,
                dtype=mocap_quat_view.dtype,
            )
            self.train_full_video_mocap_pos_wp = wp.from_torch(
                self.train_full_video_mocap_pos_buf,
                dtype=wp.vec3,
            )
            self.train_full_video_mocap_quat_wp = wp.from_torch(
                self.train_full_video_mocap_quat_buf,
                dtype=wp.quat,
            )
        activation = self._resolve_train_full_video_activation_tensor()
        activation_width = self._resolve_train_full_video_activation_width(activation)
        self.train_full_video_activation_sample_count = 0
        self.train_full_video_step_activation_staged = False
        if activation_width > 0:
            self.train_full_video_activation_buf = torch.zeros(
                (
                    rollout_steps,
                    int(substep_count),
                    int(capture_env_count),
                    activation_width,
                ),
                device=qpos_view.device if activation is None else activation.device,
                dtype=torch.float32,
            )
            self.train_full_video_step_activation = torch.zeros(
                (int(capture_env_count), activation_width),
                device=qpos_view.device if activation is None else activation.device,
                dtype=torch.float32,
            )
        else:
            self.train_full_video_activation_buf = None
            self.train_full_video_step_activation = None
        self.train_full_video_qpos_wp = wp.from_torch(self.train_full_video_qpos_buf)
        self.train_full_video_qvel_wp = wp.from_torch(self.train_full_video_qvel_buf)

    def _resolve_train_full_video_activation_width(
        self,
        activation: torch.Tensor | None,
    ) -> int:
        if activation is not None:
            return int(activation.reshape(int(activation.shape[0]), -1).shape[1])
        return max(0, int(self._env._control_pipeline.full_video_activation_width()))

    def _resolve_train_full_video_activation_tensor(self) -> torch.Tensor | None:
        activation = self._env._control_pipeline.get_full_video_activation()
        if activation is None or not torch.is_tensor(activation):
            return None
        if activation.ndim < 2:
            return None
        activation = activation.detach().reshape(int(activation.shape[0]), -1)
        if activation.numel() <= 0:
            return None
        return activation.to(dtype=torch.float32)

    def should_collect_train_full_video_substeps(self) -> bool:
        return (
            bool(self.train_full_video_enabled)
            and int(self.train_full_video_substep_count) > 1
            and self.train_full_video_qpos_buf is not None
            and self.train_full_video_qvel_buf is not None
        )

    def prepare_train_full_video_step_capture(
        self,
        step_idx: int,
        *,
        activation: torch.Tensor | None = None,
    ) -> None:
        if not self.train_full_video_enabled:
            return
        self.train_full_video_current_step = int(step_idx)
        self.train_full_video_step_activation_staged = False
        if activation is None or self.train_full_video_step_activation is None:
            return
        activation = activation.reshape(int(activation.shape[0]), -1)
        target = self.train_full_video_step_activation
        if int(activation.shape[0]) < int(target.shape[0]) or int(
            activation.shape[1]
        ) != int(target.shape[1]):
            raise ValueError(
                "Train full-video activation shape is incompatible with capture "
                f"staging {tuple(target.shape)}: got {tuple(activation.shape)}."
            )
        target.copy_(activation[: int(target.shape[0])])
        self.train_full_video_step_activation_staged = True

    def capture_train_full_video_step_state(
        self,
        step_idx: int,
        *,
        activation: torch.Tensor | None = None,
    ) -> None:
        if not self.train_full_video_enabled:
            return
        if self.should_collect_train_full_video_substeps():
            return
        env = self._env
        qpos_view = wp.to_torch(env.mjw_data.qpos)
        qvel_view = wp.to_torch(env.mjw_data.qvel)
        self._capture_train_full_video_state(
            step_idx=int(step_idx),
            substep_idx=0,
            qpos_env_count=int(qpos_view.shape[0]),
            qvel_env_count=int(qvel_view.shape[0]),
            activation=activation,
        )

    def _capture_train_full_video_state(
        self,
        *,
        step_idx: int,
        substep_idx: int,
        qpos_env_count: int,
        qvel_env_count: int,
        activation: torch.Tensor | None,
    ) -> None:
        if (
            self.train_full_video_qpos_wp is None
            or self.train_full_video_qvel_wp is None
        ):
            return
        if (
            self.train_full_video_qpos_buf is None
            or self.train_full_video_qvel_buf is None
        ):
            return
        env = self._env
        step_idx = int(step_idx)
        substep_idx = int(substep_idx)
        if step_idx < 0 or step_idx >= int(self.train_full_video_qpos_buf.shape[0]):
            return
        if substep_idx < 0 or substep_idx >= int(
            self.train_full_video_qpos_buf.shape[1]
        ):
            return
        qpos_env_count = min(int(qpos_env_count), int(self.train_full_video_env_count))
        qvel_env_count = min(int(qvel_env_count), int(self.train_full_video_env_count))
        wp.launch(
            _copy_array2d_prefix_to_array4d_kernel,
            dim=(qpos_env_count, int(env.model.nq)),
            inputs=[
                env.mjw_data.qpos,
                qpos_env_count,
                self.train_full_video_qpos_wp,
                step_idx,
                substep_idx,
            ],
        )
        wp.launch(
            _copy_array2d_prefix_to_array4d_kernel,
            dim=(qvel_env_count, int(env.model.nv)),
            inputs=[
                env.mjw_data.qvel,
                qvel_env_count,
                self.train_full_video_qvel_wp,
                step_idx,
                substep_idx,
            ],
        )
        mocap_pos = getattr(env.mjw_data, "mocap_pos", None)
        mocap_quat = getattr(env.mjw_data, "mocap_quat", None)
        if (
            mocap_pos is not None
            and mocap_quat is not None
            and self.train_full_video_mocap_pos_wp is not None
            and self.train_full_video_mocap_quat_wp is not None
            and self.train_full_video_mocap_pos_buf is not None
            and self.train_full_video_mocap_quat_buf is not None
        ):
            mocap_env_count = min(
                int(qpos_env_count),
                int(self.train_full_video_env_count),
                int(self.train_full_video_mocap_pos_buf.shape[2]),
            )
            mocap_count = int(self.train_full_video_mocap_pos_buf.shape[3])
            if mocap_env_count > 0 and mocap_count > 0:
                wp.launch(
                    _copy_vec3_array2d_prefix_to_array4d_kernel,
                    dim=(mocap_env_count, mocap_count),
                    inputs=[
                        mocap_pos,
                        mocap_env_count,
                        self.train_full_video_mocap_pos_wp,
                        step_idx,
                        substep_idx,
                    ],
                )
                wp.launch(
                    _copy_quat_array2d_prefix_to_array4d_kernel,
                    dim=(mocap_env_count, mocap_count),
                    inputs=[
                        mocap_quat,
                        mocap_env_count,
                        self.train_full_video_mocap_quat_wp,
                        step_idx,
                        substep_idx,
                    ],
                )
        if activation is None:
            activation = (
                self.train_full_video_step_activation
                if self.should_collect_train_full_video_substeps()
                and self.train_full_video_step_activation_staged
                else self._resolve_train_full_video_activation_tensor()
            )
        if activation is not None and self.train_full_video_activation_buf is not None:
            activation = activation.reshape(int(activation.shape[0]), -1)
            activation_count = min(
                int(activation.shape[0]),
                int(self.train_full_video_env_count),
                int(self.train_full_video_activation_buf.shape[2]),
            )
            activation_width = min(
                int(activation.shape[1]),
                int(self.train_full_video_activation_buf.shape[3]),
            )
            if activation_count > 0 and activation_width > 0:
                target = self.train_full_video_activation_buf[
                    step_idx,
                    substep_idx,
                    :activation_count,
                    :activation_width,
                ]
                target.copy_(activation[:activation_count, :activation_width])
                self.train_full_video_activation_sample_count += 1
        self.train_full_video_step_count = max(
            int(self.train_full_video_step_count),
            step_idx + 1,
        )

    def get_train_full_video_render_tile_state_batch(
        self,
    ) -> object:
        env = self._env
        return select_train_full_video_frame_batch(
            self,
            decimation=int(env.cfg.decimation),
            render_interval=int(env.cfg.sim.render_interval),
        )

    def render_substep_frames(self) -> list[np.ndarray]:
        if not self.should_collect_substeps():
            return []
        assert self.substep_qpos_buf is not None
        assert self.substep_qvel_buf is not None
        env = self._env
        frame_indices = _substep_render_frame_indices(
            decimation=int(env.cfg.decimation),
            render_interval=int(env.cfg.sim.render_interval),
            phase_before_step=int(self.phase_before_last_step),
        )
        return [
            self.render_rgb_array_from_state(
                self.substep_qpos_buf[substep_idx, int(self.capture_env_idx)],
                self.substep_qvel_buf[substep_idx, int(self.capture_env_idx)],
            )
            for substep_idx in frame_indices
        ]

    def get_playback_render_state_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        env = self._env
        if self.should_collect_substeps():
            assert self.substep_qpos_buf is not None
            assert self.substep_qvel_buf is not None
            frame_indices = _substep_render_frame_indices(
                decimation=int(env.cfg.decimation),
                render_interval=int(env.cfg.sim.render_interval),
                phase_before_step=int(self.phase_before_last_step),
            )
            return (
                self.substep_qpos_buf[list(frame_indices), int(self.capture_env_idx)],
                self.substep_qvel_buf[list(frame_indices), int(self.capture_env_idx)],
            )

        qpos_view = wp.to_torch(env.mjw_data.qpos)
        qvel_view = wp.to_torch(env.mjw_data.qvel)
        env_idx = int(self.capture_env_idx)
        return (
            qpos_view[env_idx : env_idx + 1],
            qvel_view[env_idx : env_idx + 1],
        )

    def stage_playback_render_state_batch_cpu(self) -> tuple[np.ndarray, np.ndarray]:
        batch_qpos, batch_qvel = self.get_playback_render_state_batch()
        return (
            batch_qpos.detach().cpu().numpy(),
            batch_qvel.detach().cpu().numpy(),
        )

    def stage_playback_render_tile_state_batch_cpu(
        self,
        env_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_qpos, batch_qvel = self.get_playback_render_tile_state_batch(env_count)
        return (
            batch_qpos.detach().cpu().numpy(),
            batch_qvel.detach().cpu().numpy(),
        )

    def get_playback_render_tile_state_batch(
        self,
        env_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.should_collect_substeps():
            assert self.substep_qpos_buf is not None
            assert self.substep_qvel_buf is not None
            frame_indices = _substep_render_frame_indices(
                decimation=int(self._env.cfg.decimation),
                render_interval=int(self._env.cfg.sim.render_interval),
                phase_before_step=int(self.phase_before_last_step),
            )
            count = max(1, min(int(env_count), int(self.substep_qpos_buf.shape[1])))
            return (
                self.substep_qpos_buf[list(frame_indices), :count],
                self.substep_qvel_buf[list(frame_indices), :count],
            )

        env = self._env
        qpos_view = wp.to_torch(env.mjw_data.qpos)
        qvel_view = wp.to_torch(env.mjw_data.qvel)
        count = max(1, min(int(env_count), int(qpos_view.shape[0])))
        return (
            qpos_view[:count][None, :, :],
            qvel_view[:count][None, :, :],
        )

    def get_playback_render_mocap_batch(
        self,
        qpos_batch: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        env = self._env
        render_data = getattr(env, "render_data", None)
        sim_data = getattr(env, "data", None)
        if (
            render_data is not None
            and sim_data is not None
            and render_data is not sim_data
        ):
            return None

        mjw_data = getattr(env, "mjw_data", None)
        mocap_pos = getattr(mjw_data, "mocap_pos", None)
        mocap_quat = getattr(mjw_data, "mocap_quat", None)
        if mocap_pos is None or mocap_quat is None:
            return None

        mocap_pos_view = wp.to_torch(mocap_pos)
        mocap_quat_view = wp.to_torch(mocap_quat)
        if int(mocap_pos_view.shape[1]) <= 0:
            return None

        batch_shape = tuple(int(dim) for dim in qpos_batch.shape)
        if len(batch_shape) == 2:
            frame_count = int(batch_shape[0])
            env_idx = int(self.capture_env_idx)
            pos = mocap_pos_view[env_idx : env_idx + 1].expand(frame_count, -1, -1)
            quat = mocap_quat_view[env_idx : env_idx + 1].expand(frame_count, -1, -1)
            return pos, quat
        if len(batch_shape) == 3:
            frame_count = int(batch_shape[0])
            env_count = min(int(batch_shape[1]), int(mocap_pos_view.shape[0]))
            pos = (
                mocap_pos_view[:env_count].unsqueeze(0).expand(frame_count, -1, -1, -1)
            )
            quat = (
                mocap_quat_view[:env_count]
                .unsqueeze(0)
                .expand(
                    frame_count,
                    -1,
                    -1,
                    -1,
                )
            )
            return pos, quat
        raise ValueError("qpos_batch must be rank-2 or rank-3 for playback rendering.")

    def stage_playback_render_mocap_batch_cpu(
        self,
        qpos_batch: torch.Tensor | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        mocap_batch = self.get_playback_render_mocap_batch(qpos_batch)
        if mocap_batch is None:
            return None
        mocap_pos, mocap_quat = mocap_batch
        return (
            mocap_pos.detach().cpu().numpy(),
            mocap_quat.detach().cpu().numpy(),
        )

    def render_rgb_array_from_state(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
    ) -> np.ndarray:
        return self.render_cpu(qpos=qpos, qvel=qvel)

    def render_cpu(
        self,
        qpos: torch.Tensor | None = None,
        qvel: torch.Tensor | None = None,
    ) -> np.ndarray:
        env = self._env
        if qpos is None or qvel is None:
            mjw.get_data_into(env.data, env.model, env.mjw_data)
            qpos_np = env.data.qpos
            qvel_np = env.data.qvel
        else:
            qpos_np = qpos.detach().cpu().numpy()
            qvel_np = qvel.detach().cpu().numpy()
        if env.render_data is not env.data:
            env.render_data.qpos[:] = qpos_np[: env.render_model.nq]
            env.render_data.qvel[:] = qvel_np[: env.render_model.nv]
            env._update_render_data()
            upload_dynamic_render_meshes = getattr(
                env,
                "_upload_dynamic_render_meshes",
                None,
            )
            if callable(upload_dynamic_render_meshes):
                upload_dynamic_render_meshes(
                    mjr_context=getattr(env.renderer, "_mjr_context", None)
                )
            mujoco.mj_forward(env.render_model, env.render_data)
            env.renderer.update_scene(env.render_data, camera=env.camera)
        else:
            env.data.qpos[:] = qpos_np[: env.model.nq]
            env.data.qvel[:] = qvel_np[: env.model.nv]
            mujoco.mj_forward(env.model, env.data)
            env.renderer.update_scene(env.data, camera=env.camera)
        rgb_data = env.renderer.render()
        expected_shape = (
            env.cfg.viewer.resolution[1],
            env.cfg.viewer.resolution[0],
            3,
        )
        if rgb_data.shape != expected_shape:
            raise RuntimeError(
                "Renderer returned RGB data with shape "
                f"{rgb_data.shape}; expected {expected_shape}."
            )
        if rgb_data.dtype != np.uint8:
            raise RuntimeError(
                "Renderer returned RGB data with dtype "
                f"{rgb_data.dtype}; expected uint8."
            )
        if rgb_data.size == 0:
            return np.zeros(
                (env.cfg.viewer.resolution[1], env.cfg.viewer.resolution[0], 3),
                dtype=np.uint8,
            )
        return rgb_data[:, :, :3]

    def close(self) -> None:
        renderer = getattr(self._env, "renderer", None)
        close = getattr(renderer, "close", None)
        if callable(close):
            close()
