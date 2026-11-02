from __future__ import annotations

import logging

import numpy as np
import torch
import warp as wp
from pymotion.rotations import quat_torch

from instantmimic.envs.command import render_runtime
from instantmimic.envs.command.latent_command_runtime import (
    LatentCommandRuntime,
)
from instantmimic.envs.command.render_assets import load_wavefront_vertices_np
from instantmimic.envs.command.rewards import (
    VELOCITY_REWARD_MODE_FORWARD_SPEED_HEADING,
    resolve_velocity_heading_reward_mode,
)
from instantmimic.envs.control_pipeline import make_direct_control_runtime_spec
from instantmimic.envs.controls.fast_muscle.render_builder import (
    strip_render_muscle_artifacts,
)
from instantmimic.envs.eval_metrics import EvalEpisodeSpec, EvalMetricSpec
from instantmimic.envs.snu.tasks.snu_command import low_level_runtime
from instantmimic.envs.snu.tasks.snu_command.checkpoint_runtime import (
    apply_low_level_checkpoint_runtime_config,
    cfg_nested_get,
)
from instantmimic.envs.snu.tasks.snu_command.snu_command_env_cfg import SNUCommandEnvCfg
from instantmimic.envs.snu.tasks.snu_imitation import model_metadata
from instantmimic.envs.snu.tasks.snu_imitation.control_resolver import (
    uses_fast_muscle_control,
)
from instantmimic.envs.snu.tasks.snu_imitation.motion_settings import MotionSettings
from instantmimic.envs.snu.tasks.snu_imitation.snu_imitation_env import SNUImitationEnv
from instantmimic.envs.tracking.metrics import get_heading_frame
from instantmimic.utils.project_paths import resolve_project_path

_LOGGER = logging.getLogger(__name__)


class SNUCommandEnv(SNUImitationEnv):
    _COMMAND_REWARD_COLORS = {
        "alive_reward": (241, 196, 15),
        "upright_reward": (46, 204, 113),
        "height_reward": (230, 126, 34),
        "speed_reward": (52, 152, 219),
        "heading_reward": (155, 89, 182),
        "progress_reward": (52, 152, 219),
        "facing_reward": (155, 89, 182),
        "goal_bonus": (46, 204, 113),
        "stall_penalty": (231, 76, 60),
        "total_reward": (236, 240, 241),
    }

    def _resolve_motion_settings(self, cfg: SNUCommandEnvCfg) -> MotionSettings:
        source = cfg_nested_get(cfg, "motion", "source", default=None)
        if not any(
            str(item).strip()
            for item in ([source] if isinstance(source, str) else source or [])
        ):
            source = cfg_nested_get(cfg, "reset", "motion", "source", default=None)
        if not any(
            str(item).strip()
            for item in ([source] if isinstance(source, str) else source or [])
        ):
            raise ValueError(
                "command env requires env.motion.source or env.reset.motion.source"
            )
        return MotionSettings.from_env_cfg(cfg, source=source)

    def __init__(
        self, cfg: SNUCommandEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        if not getattr(cfg, "low_level_checkpoint", None):
            raise ValueError(
                "env.low_level_checkpoint must be set for command environments."
            )
        apply_low_level_checkpoint_runtime_config(cfg)
        render_runtime.initialize_render_state(self, cfg)
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        reward_settings = getattr(self, "reward_settings", None)
        if reward_settings is not None and reward_settings.action_smoothness_enabled:
            raise ValueError(
                "env.reward.action_smoothness is not supported by the SNU command task, which uses its own reward"
            )
        self._command_mode = str(self.cfg.command_mode).strip().lower()
        if self._command_mode not in {"velocity_heading", "point_goal"}:
            raise ValueError(
                f"command_mode must be 'velocity_heading' or 'point_goal', got {self.cfg.command_mode!r}."
            )
        self._qpos_view = wp.to_torch(self.mjw_data.qpos)
        self._qvel_view = wp.to_torch(self.mjw_data.qvel)
        self._xpos_view = wp.to_torch(self.mjw_data.xpos)
        self._xquat_view = wp.to_torch(self.mjw_data.xquat)
        self._cvel_view = wp.to_torch(self.mjw_data.cvel)
        low_level_runtime.initialize_low_level_policy(self)
        self._target_speed = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self._target_heading = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self._goal_pos_xy = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32
        )
        self._prev_goal_distance = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self._command_reset_mask_float_buf = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self._sampled_target_speed_buf = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self._sampled_target_heading_buf = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self._sampled_goal_distance_buf = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self._sampled_goal_angle_buf = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float32
        )
        self._sampled_goal_pos_xy_buf = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32
        )
        self._target_speed_wp = wp.from_torch(self._target_speed)
        self._target_heading_wp = wp.from_torch(self._target_heading)
        self._goal_pos_xy_wp = wp.from_torch(self._goal_pos_xy)
        self._prev_goal_distance_wp = wp.from_torch(self._prev_goal_distance)
        self._init_velocity_command_resample_buffers()
        self._sample_commands(
            torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        )
        low_level_state = self.latent_low_level_state
        self.actions = low_level_state.environment_action_buffer
        self._init_command_reward_debug_buffers()

    def _get_latent_command_runtime(self) -> LatentCommandRuntime:
        runtime = self.__dict__.get("_latent_command_runtime")
        if runtime is None:
            runtime = LatentCommandRuntime(self)
            self._latent_command_runtime = runtime
        return runtime

    def _command_reward_term_names(self) -> tuple[str, ...]:
        return self._get_latent_command_runtime().command_reward_term_names()

    def _init_command_reward_debug_buffers(self) -> None:
        self._get_latent_command_runtime().init_command_reward_debug_buffers()

    def _resolved_command_num_envs(self) -> int:
        return self._get_latent_command_runtime().resolved_command_num_envs()

    def _velocity_command_resample_step_bounds(self) -> tuple[int, int]:
        return (
            self._get_latent_command_runtime().velocity_command_resample_step_bounds()
        )

    def _init_velocity_command_resample_buffers(self) -> None:
        self._get_latent_command_runtime().init_velocity_command_resample_buffers()

    def _velocity_command_resample_enabled(self) -> bool:
        return self._get_latent_command_runtime().velocity_command_resample_enabled()

    def _velocity_heading_reward_mode(self) -> str:
        return resolve_velocity_heading_reward_mode(
            getattr(
                self.cfg,
                "velocity_heading_reward_mode",
                VELOCITY_REWARD_MODE_FORWARD_SPEED_HEADING,
            )
        )

    def _velocity_command_arrow_enabled(self) -> bool:
        return (
            str(getattr(self.cfg, "command_mode", "")).strip().lower()
            == "velocity_heading"
            and getattr(self.cfg, "velocity_command_arrow_tail_mesh_path", None)
            is not None
            and (
                getattr(self.cfg, "velocity_command_arrow_body_mesh_path", None)
                is not None
            )
            and (
                getattr(self.cfg, "velocity_command_arrow_head_mesh_path", None)
                is not None
            )
        )

    def _resolved_velocity_command_arrow_part_mesh_path(self, part: str) -> str | None:
        raw = getattr(self.cfg, f"velocity_command_arrow_{part}_mesh_path", None)
        if raw is None:
            return None
        return str(resolve_project_path(str(raw)))

    def _ensure_velocity_command_arrow_visual_defaults(self) -> None:
        if not self._velocity_command_arrow_enabled():
            return
        state = self.command_render_state
        scale = np.asarray(
            getattr(self.cfg, "velocity_command_arrow_scale", [1.0, 1.0, 1.0]),
            dtype=np.float64,
        ).reshape(3)
        have_cached_render_defaults = all(
            part_state.local_position is not None
            and part_state.local_quaternion is not None
            for part_state in state.arrow_parts.values()
        )
        if not have_cached_render_defaults and getattr(self, "model", None) is not None:
            try:
                self._build_render_model()
            except (OSError, ValueError, RuntimeError) as exc:
                _LOGGER.warning(
                    "Could not build velocity-arrow render defaults; using fallback bounds: %s",
                    exc,
                    exc_info=True,
                )
            state = self.command_render_state
        for part in ("tail", "body", "head"):
            part_state = state.arrow_parts[part]
            if part_state.local_position is None:
                part_state.local_position = np.zeros(3, dtype=np.float64)
            if part_state.local_quaternion is None:
                part_state.local_quaternion = np.array(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
                )
            if part_state.bounds_initialized:
                continue
            mesh_path = self._resolved_velocity_command_arrow_part_mesh_path(part)
            if mesh_path is None:
                raise ValueError(
                    f"velocity_command_arrow_{part}_mesh_path must be set."
                )
            resolved_mesh_path = resolve_project_path(mesh_path)
            if not resolved_mesh_path.is_file():
                raise ValueError(
                    f"velocity_command_arrow_{part}_mesh_path does not exist: {mesh_path}"
                )
            if resolved_mesh_path.suffix.lower() != ".obj":
                raise ValueError(
                    "velocity_command_arrow mesh paths must point to OBJ meshes for Blender export special visuals."
                )
            vertices = load_wavefront_vertices_np(str(resolved_mesh_path))
            vertices = vertices * scale.reshape(1, 3)
            part_state.bound_min_x = float(vertices[:, 0].min())
            part_state.bound_max_x = float(vertices[:, 0].max())
            part_state.bounds_initialized = True

    def _point_goal_marker_enabled(self) -> bool:
        return (
            str(getattr(self.cfg, "command_mode", "")).strip().lower() == "point_goal"
            and getattr(self.cfg, "point_goal_marker_mesh_path", None) is not None
        )

    def _resolved_point_goal_marker_mesh_path(self) -> str | None:
        raw = getattr(self.cfg, "point_goal_marker_mesh_path", None)
        if raw is None:
            return None
        return str(resolve_project_path(str(raw)))

    def _resolved_point_goal_marker_texture_path(self) -> str | None:
        raw = getattr(self.cfg, "point_goal_marker_texture_path", None)
        if raw is None:
            return None
        return str(resolve_project_path(str(raw)))

    def velocity_command_arrow_is_unit_length(self) -> bool:
        """The SNU velocity command always carries a magnitude."""
        return False

    def resolve_render_source_mjcf_path(self) -> str:
        return model_metadata.resolve_source_mjcf_path(self)

    def prepare_render_root(self, root) -> None:
        if uses_fast_muscle_control(self.cfg):
            strip_render_muscle_artifacts(root)

    def append_render_worldbody_extras(self, worldbody) -> None:
        """Nothing to add: the SNU command task has no extra render bodies."""

    def render_extra_visuals_enabled(self) -> bool:
        return False

    def render_extra_visual_template_name(self) -> str:
        raise NotImplementedError("SNU command rendering has no extra visuals.")

    def render_extra_visual_count(self) -> int:
        return 0

    def render_extra_visual_state(
        self, env_idx: int
    ) -> dict[str, dict[str, np.ndarray]]:
        return {}

    def get_blender_reference_enabled(self) -> bool:
        return render_runtime.get_blender_reference_enabled(self)

    def get_blender_special_visual_specs(self) -> list[dict[str, object]]:
        return render_runtime.get_blender_special_visual_specs(self)

    def get_blender_special_visual_state(self) -> dict[str, dict[str, np.ndarray]]:
        return render_runtime.get_blender_special_visual_state(self)

    def get_blender_special_reduced_state(self) -> dict[str, np.ndarray]:
        return render_runtime.get_blender_special_reduced_state(self)

    def _ensure_command_reward_debug_buffers(self) -> None:
        self._get_latent_command_runtime().ensure_command_reward_debug_buffers()

    def _maybe_resample_velocity_commands(self) -> None:
        self._get_latent_command_runtime().maybe_resample_velocity_commands()

    def _build_render_model(self):
        return render_runtime.build_render_model(self)

    def _update_render_data(self):
        render_runtime.update_render_data(self)

    def _upload_dynamic_render_meshes(self, *, mjr_context) -> None:
        render_runtime.upload_dynamic_render_meshes(self, mjr_context=mjr_context)

    def _current_qpos(self) -> torch.Tensor:
        return self._qpos_view

    def _current_qvel(self) -> torch.Tensor:
        return self._qvel_view

    def _root_pos_quat(self) -> tuple[torch.Tensor, torch.Tensor]:
        qpos = self._current_qpos()
        return (qpos[:, :3], qpos[:, 3:7])

    def _planar_velocity(self) -> torch.Tensor:
        qvel = self._current_qvel()
        return qvel[:, :2]

    def _heading_xy(self) -> torch.Tensor:
        root_pos, root_quat = self._root_pos_quat()
        _, root_yaw_quat = get_heading_frame(root_pos, root_quat)
        forward = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=root_yaw_quat.dtype
        )
        forward[:, 0] = 1.0
        heading = quat_torch.mul_vec(root_yaw_quat, forward)
        return heading[:, :2]

    def _up_reward(self) -> torch.Tensor:
        _, root_quat = self._root_pos_quat()
        up = torch.zeros((self.num_envs, 3), device=self.device, dtype=root_quat.dtype)
        up[:, 2] = 1.0
        root_up = quat_torch.mul_vec(root_quat, up)
        return torch.clamp(root_up[:, 2:3], min=0.0)

    def _torso_height_reward(self) -> torch.Tensor:
        xpos_view = getattr(self, "_xpos_view", None)
        torso_body_idx = getattr(self, "torso_body_idx", None)
        if xpos_view is None or torso_body_idx is None:
            return torch.zeros_like(self._target_speed)
        torso_height = xpos_view[:, int(torso_body_idx), 2:3]
        return torch.clamp(torso_height, min=0.0, max=1.0)

    def refresh_low_level_obs(self) -> None:
        low_level_runtime.refresh_low_level_obs(self)

    def get_control_runtime_spec(self):
        return make_direct_control_runtime_spec(
            action_dim=int(self.cfg.action_space), action_mode="command"
        )

    def supports_zero_residual_playback(self) -> bool:
        return True

    def get_command_diagnostic_means(self) -> dict[str, float]:
        return self._get_latent_command_runtime().get_command_diagnostic_means()

    def get_goal_reached_mask(self) -> torch.Tensor:
        return self._get_latent_command_runtime().get_goal_reached_mask()

    def get_eval_metric_specs(self) -> dict[str, EvalMetricSpec]:
        return self._get_latent_command_runtime().get_eval_metric_specs()

    def get_eval_episode_spec(self) -> EvalEpisodeSpec:
        return self._get_latent_command_runtime().get_eval_episode_spec()

    def get_video_overlay_plots(self, env_idx: int = 0) -> dict[str, object]:
        return self._get_latent_command_runtime().get_video_overlay_plots(env_idx)

    def _target_direction(self) -> torch.Tensor:
        return self._get_latent_command_runtime().target_direction()

    def _goal_local_delta_and_distance(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_latent_command_runtime().goal_local_delta_and_distance()

    def _sample_commands(self, reset_mask: torch.Tensor) -> None:
        self._get_latent_command_runtime().sample_commands(reset_mask)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != int(self.cfg.action_space):
            raise ValueError(
                f"Expected latent action dimension {int(self.cfg.action_space)}, got {actions.shape[-1]}."
            )
        low_level_runtime.capture_previous_latent_action(self)
        low_level_runtime.apply_latent_action(self, actions)
        self.ref_frame_idx += 1

    def _get_observations(self, out_obs: torch.Tensor | None = None) -> dict:
        return self._get_latent_command_runtime().get_observations(out_obs=out_obs)

    def _get_rewards(self) -> torch.Tensor:
        return self._get_latent_command_runtime().get_rewards()

    def get_persistent_command_reward_buffer(self) -> torch.Tensor:
        state = self.reward_runtime_state
        if state is None:
            raise RuntimeError("SNU reward runtime state is not initialized.")
        return state.reward

    def _reset_idx(self, reset_mask: torch.Tensor):
        super()._reset_idx(reset_mask)
        self._get_latent_command_runtime().reset_after_base_reset(reset_mask)
