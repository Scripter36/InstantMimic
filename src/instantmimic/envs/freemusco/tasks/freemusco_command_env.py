from __future__ import annotations

import logging

import numpy as np
import torch
import warp as wp
from pymotion.rotations import quat_torch

from instantmimic.envs.command import render_runtime
from instantmimic.envs.command.command_runtime import (
    CommandRuntime,
    CommandState,
)
from instantmimic.envs.command.render_assets import (
    load_wavefront_vertices_np,
)
from instantmimic.envs.controls.fast_muscle.render_builder import (
    strip_render_actuators,
)
from instantmimic.envs.eval_metrics import EvalEpisodeSpec, EvalMetricSpec
from instantmimic.envs.freemusco.tasks.freemusco_imitation_env import (
    FreemuscoImitationEnv,
)
from instantmimic.envs.tracking.metrics import get_heading_frame
from instantmimic.utils.project_paths import resolve_project_path

from .freemusco_command_env_cfg import FreemuscoCommandEnvCfg

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def compute_command_dones_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    episode_length_buf: wp.array(dtype=wp.int64),
    total_reward: wp.array(dtype=float),
    early_termination_timer: wp.array(dtype=wp.int32),
    enable_early_termination: int,
    root_body_id: int,
    termination_height: float,
    max_episode_length: int,
    early_termination_reward_threshold: float,
    early_termination_time_threshold: int,
    reset_terminated: wp.array(dtype=wp.bool),
    reset_time_outs: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()
    reset_time_outs[env_idx] = int(episode_length_buf[env_idx]) >= max_episode_length

    died = xpos[env_idx, root_body_id][2] < termination_height
    if enable_early_termination == 1:
        if total_reward[env_idx] < early_termination_reward_threshold:
            early_termination_timer[env_idx] = early_termination_timer[env_idx] + 1
        else:
            early_termination_timer[env_idx] = 0
        if early_termination_timer[env_idx] >= early_termination_time_threshold:
            died = True
    else:
        early_termination_timer[env_idx] = 0

    reset_terminated[env_idx] = died


class FreemuscoCommandEnv(FreemuscoImitationEnv):
    command_render_state: render_runtime.CommandRenderState

    cfg: FreemuscoCommandEnvCfg
    command_state: CommandState
    _command_runtime: CommandRuntime

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

    def __init__(
        self,
        cfg: FreemuscoCommandEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        if str(getattr(cfg, "species", "")).strip().lower() != "ostrich":
            raise ValueError(
                "FreemuscoCommandEnv currently requires env.species='ostrich'."
            )
        # Before super().__init__, the same as the SNU command family: the
        # blender exporter can ask for arrow state on an env that never built a
        # render model, and that path reads command_render_state.
        render_runtime.initialize_render_state(self, cfg)
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._command_mode = str(self.cfg.command_mode).strip().lower()
        if self._command_mode not in {"velocity_heading", "point_goal"}:
            raise ValueError(
                "command_mode must be 'velocity_heading' or 'point_goal', "
                f"got {self.cfg.command_mode!r}."
            )
        self._base_obs_main_dim = int(self._obs_main_dim)
        self._base_obs_suffix_dim = int(self._obs_suffix_dim)
        self._base_observation_space = int(self._obs_buf.shape[-1])
        self._base_deployable_obs_spec = dict(
            FreemuscoImitationEnv.get_deployable_obs_spec(self)
        )
        self._deployable_obs_indices = torch.tensor(
            self._base_deployable_obs_spec["policy_obs_indices"],
            device=self.device,
            dtype=torch.long,
        )
        self._deployable_obs_dim = int(self._deployable_obs_indices.numel())
        self._command_base_obs_buf: torch.Tensor | None = None
        self.xquat_view = wp.to_torch(self.mjw_data.xquat)
        self.torso_body_idx = int(self.root_body_id)
        self._initialize_command_policy()

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
        self._init_command_reward_debug_buffers()
        self.early_termination_timer = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        self.total_reward = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _get_command_runtime(self) -> CommandRuntime:
        runtime = self.__dict__.get("_command_runtime")
        if runtime is None:
            runtime = CommandRuntime(self)
            self._command_runtime = runtime
        return runtime

    def _command_reward_term_names(self) -> tuple[str, ...]:
        return self._get_command_runtime().command_reward_term_names()

    def _velocity_heading_direction_only_enabled(self) -> bool:
        return bool(getattr(self.cfg, "velocity_heading_direction_only", False))

    def _resolved_command_dim(self) -> int:
        return self._get_command_runtime().resolved_command_dim()

    def _resolved_command_num_envs(self) -> int:
        return self._get_command_runtime().resolved_command_num_envs()

    def _init_command_reward_debug_buffers(self) -> None:
        self._get_command_runtime().init_command_reward_debug_buffers()

    def _ensure_command_reward_debug_buffers(self) -> None:
        self._get_command_runtime().ensure_command_reward_debug_buffers()

    def _velocity_command_resample_step_bounds(self) -> tuple[int, int]:
        return self._get_command_runtime().velocity_command_resample_step_bounds()

    def _init_velocity_command_resample_buffers(self) -> None:
        self._get_command_runtime().init_velocity_command_resample_buffers()

    def _velocity_command_resample_enabled(self) -> bool:
        return self._get_command_runtime().velocity_command_resample_enabled()

    def _maybe_resample_velocity_commands(self) -> None:
        self._get_command_runtime().maybe_resample_velocity_commands()

    def _initialize_command_policy(self) -> None:
        self._command_dim = self._resolved_command_dim()
        self._obs_main_dim = int(self._deployable_obs_dim + self._command_dim)
        self.fast_muscle.configure_observation_layout(self._obs_main_dim)
        total_obs_dim = int(self._obs_main_dim + self._base_obs_suffix_dim)
        self._obs_buf = torch.zeros(
            (self.num_envs, total_obs_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.obs_buf = {"policy": self._obs_buf}
        self._command_obs_buf = torch.zeros(
            (self.num_envs, self._command_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self._command_base_obs_buf = torch.zeros(
            (self.num_envs, self._base_observation_space),
            device=self.device,
            dtype=torch.float32,
        )
        self.finalize_derived_spaces(
            observation_spec=total_obs_dim,
            action_spec=self.cfg.action_space,
            observations=self.obs_buf,
        )
        self.fast_muscle.bind_action_buffer(self.actions)

    def _current_qpos(self) -> torch.Tensor:
        return self.qpos_view

    def _current_qvel(self) -> torch.Tensor:
        return self.qvel_view

    def _root_pos_quat(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.xpos_view[:, self.root_body_id, :],
            self.xquat_view[:, self.root_body_id, :],
        )

    def _planar_velocity(self) -> torch.Tensor:
        qvel = self._current_qvel()
        return qvel[:, :2]

    def _heading_xy(self) -> torch.Tensor:
        root_pos, root_quat = self._root_pos_quat()
        _, root_yaw_quat = get_heading_frame(root_pos, root_quat)
        forward = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=root_yaw_quat.dtype,
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
        root_pos, _ = self._root_pos_quat()
        return torch.clamp(root_pos[:, 2:3], min=0.0, max=1.0)

    def _target_direction(self) -> torch.Tensor:
        return self._get_command_runtime().target_direction()

    def _goal_local_delta_and_distance(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_command_runtime().goal_local_delta_and_distance()

    def _sample_commands(self, reset_mask: torch.Tensor) -> None:
        self._get_command_runtime().sample_commands(reset_mask)

    def _fill_command_base_obs(self, out_obs: torch.Tensor) -> None:
        FreemuscoImitationEnv._get_observations(self, out_obs=out_obs)

    def _get_observations(self, out_obs: torch.Tensor | None = None) -> dict:
        return self._get_command_runtime().get_observations(out_obs=out_obs)

    def get_deployable_obs_spec(self) -> dict[str, object]:
        obs_spec = dict(self._base_deployable_obs_spec)
        command_indices = list(
            range(
                int(self._deployable_obs_dim),
                int(self._deployable_obs_dim + self._command_dim),
            )
        )
        obs_spec["policy_obs_indices"] = (
            list(range(int(self._deployable_obs_dim))) + command_indices
        )
        obs_spec["policy_obs_dim"] = int(self._deployable_obs_dim + self._command_dim)
        obs_spec["obs_main_dim"] = int(self._obs_main_dim)
        return obs_spec

    def _get_rewards(self) -> torch.Tensor:
        return self._get_command_runtime().get_rewards()

    def get_persistent_command_reward_buffer(self) -> torch.Tensor:
        return self._reward_buf

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        wp.launch(
            compute_command_dones_kernel,
            dim=(self.num_envs,),
            inputs=[
                self.mjw_data.xpos,
                self.episode_length_buf,
                self.total_reward,
                self.early_termination_timer,
                int(bool(self.cfg.get("enable_early_termination", True))),
                self.root_body_id,
                float(self.cfg.get("termination_height", 0.6)),
                self.max_episode_length,
                float(self.cfg.get("early_termination_reward_threshold", 0.05)),
                int(self.cfg.get("early_termination_time_threshold", 4)),
            ],
            outputs=[
                self.reset_terminated,
                self.reset_time_outs,
            ],
        )
        return self.reset_terminated, self.reset_time_outs

    def _reset_idx(self, reset_mask: torch.Tensor):
        super()._reset_idx(reset_mask)
        self._get_command_runtime().reset_after_base_reset(reset_mask)

    def _velocity_command_arrow_enabled(self) -> bool:
        return (
            str(getattr(self.cfg, "command_mode", "")).strip().lower()
            == "velocity_heading"
            and getattr(self.cfg, "velocity_command_arrow_tail_mesh_path", None)
            is not None
            and getattr(self.cfg, "velocity_command_arrow_body_mesh_path", None)
            is not None
            and getattr(self.cfg, "velocity_command_arrow_head_mesh_path", None)
            is not None
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
            except Exception as exc:  # noqa: BLE001 - render defaults are best-effort; fallback bounds remain valid.
                logger.warning(
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
                    f"velocity_command_arrow_{part}_mesh_path does not exist: "
                    f"{mesh_path}"
                )
            if resolved_mesh_path.suffix.lower() != ".obj":
                raise ValueError(
                    "velocity_command_arrow mesh paths must point to OBJ meshes "
                    "for Blender export special visuals."
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

    def get_blender_special_visual_specs(self) -> list[dict[str, object]]:
        return render_runtime.get_blender_special_visual_specs(self)

    def get_blender_special_visual_state(self) -> dict[str, dict[str, np.ndarray]]:
        return render_runtime.get_blender_special_visual_state(self)

    def get_blender_special_reduced_state(self) -> dict[str, np.ndarray]:
        return render_runtime.get_blender_special_reduced_state(self)

    # --- render hooks the shared command render runtime calls -------------

    def velocity_command_arrow_is_unit_length(self) -> bool:
        return self._velocity_heading_direction_only_enabled()

    def resolve_render_source_mjcf_path(self) -> str:
        return str(self._source_mjcf_path)

    def prepare_render_root(self, root) -> None:
        """Always strip actuators.

        Unlike the SNU family this task has no fast-muscle variant to branch
        on: its render model never drives actuators.
        """

        strip_render_actuators(root)

    def append_render_worldbody_extras(self, worldbody) -> None:
        """Nothing to add. FreeMusco has no projectile pool."""

    def render_extra_visuals_enabled(self) -> bool:
        return False

    def render_extra_visual_template_name(self) -> str:
        raise NotImplementedError("FreeMusco command rendering has no extra visuals.")

    def render_extra_visual_count(self) -> int:
        return 0

    def render_extra_visual_state(
        self, env_idx: int
    ) -> dict[str, dict[str, np.ndarray]]:
        return {}

    def _build_render_model(self):
        return render_runtime.build_render_model(self)

    def _update_render_data(self):
        render_runtime.update_render_data(self)

    def _upload_dynamic_render_meshes(self, *, mjr_context) -> None:
        render_runtime.upload_dynamic_render_meshes(self, mjr_context=mjr_context)

    def get_command_diagnostic_means(self) -> dict[str, float]:
        return self._get_command_runtime().get_command_diagnostic_means()

    def get_goal_reached_mask(self) -> torch.Tensor:
        return self._get_command_runtime().get_goal_reached_mask()

    def get_eval_metric_specs(self) -> dict[str, EvalMetricSpec]:
        return self._get_command_runtime().get_eval_metric_specs()

    def get_eval_episode_spec(self) -> EvalEpisodeSpec:
        return self._get_command_runtime().get_eval_episode_spec()

    def get_video_overlay_plots(self, env_idx: int = 0) -> dict[str, object]:
        return self._get_command_runtime().get_video_overlay_plots(env_idx)
