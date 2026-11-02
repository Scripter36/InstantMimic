from __future__ import annotations

import logging
from typing import Any

import torch
import warp as wp

from instantmimic.envs.controls.fast_muscle.config_models import ActionSettings
from instantmimic.envs.direct_rl_env import DirectRLEnv
from instantmimic.envs.eval_metrics import EvalEpisodeSpec, EvalResultMetricSpec
from instantmimic.envs.snu.tasks.snu_imitation import (
    action_runtime,
    adaptive_sampling_runtime,
    initial_state_runtime,
    joint_velocity_guard,
    model_metadata,
    motion_reference,
    nan_debug,
    observation_runtime,
    render_runtime,
    reset_runtime,
    reward_runtime,
    termination_runtime,
)
from instantmimic.envs.snu.tasks.snu_imitation.adaptive_sampling import (
    ADAPTIVE_SAMPLING_METHOD_MOTION_PLR,
)
from instantmimic.utils.startup_profiler import StartupProfiler, startup_nvtx_range

from .motion_settings import MotionSettings, NanDebugSettings
from .snu_imitation_env_cfg import SNUImitationEnvCfg

__all__ = [
    "SNUImitationEnv",
]


class SNUImitationEnv(DirectRLEnv):
    # Rewards, dones, observations and reset hooks read body and joint state
    # only, never contacts or constraint rows.
    boundary_reads_constraints = False
    cfg: SNUImitationEnvCfg
    _observation_layout: Any
    _obs_main_dim: int
    _obs_buf: torch.Tensor
    reward_runtime_state: reward_runtime.RewardRuntimeState | None
    adaptive_sampling_state: (
        adaptive_sampling_runtime.AdaptiveSamplingRuntimeState | None
    )
    initial_state_runtime_state: initial_state_runtime.InitialStateRuntimeState | None
    reset_runtime_state: reset_runtime.ResetRuntimeState | None
    observation_runtime_state: observation_runtime.ObservationRuntimeState | None

    def get_latent_prior_obs_spec(self) -> dict[str, object]:
        from instantmimic.envs.tracking.observation_spec import (
            build_deployable_obs_spec,
        )

        return build_deployable_obs_spec(
            observation_layout=self._observation_layout, obs_main_dim=self._obs_main_dim
        )

    def configure_observation_runtime_buffers(
        self,
        *,
        layout: Any,
        main_dim: int,
        buffer: torch.Tensor,
    ) -> None:
        """Install task-selected observation storage owned by this environment."""
        self._observation_layout = layout
        self._obs_main_dim = int(main_dim)
        self._obs_buf = buffer

    def configure_reward_runtime_buffers(
        self,
        *,
        reward: torch.Tensor,
        reward_terms: torch.Tensor,
        reward_term_names: tuple[str, ...],
    ) -> None:
        """Install task-selected reward storage owned by this environment."""
        state = self.reward_runtime_state
        if state is None:
            raise RuntimeError("Reward runtime state is not initialized.")
        state.reward = reward
        state.reward_terms = reward_terms
        state.reward_term_names = reward_term_names
        self.reward_buf = reward
        self.total_reward = reward

    def finalize_task_control_pipeline(self) -> None:
        """Finalize task-selected observations through the SNU runtime owner."""
        observation_runtime.finalize_task_control_pipeline(self)

    def record_raw_reward_nonfinite(self) -> None:
        """Record non-finite task rewards through the SNU diagnostics owner."""
        nan_debug.nan_debug_record_raw_reward_nonfinite(self)

    def _resolve_motion_settings(self, cfg: SNUImitationEnvCfg) -> MotionSettings:
        return MotionSettings.from_env_cfg(cfg)

    def _compute_motion_ctrl(self) -> torch.Tensor:
        return action_runtime.compute_motion_ctrl(self)

    def _resolve_body_id(self, candidates: list[str]) -> int:
        return model_metadata.resolve_body_id(self, candidates)

    def _load_motion_reference_data(
        self,
        *,
        shared_motion_reference: Any | None = None,
        clone_shared_motion_reference: bool = False,
    ) -> None:
        if shared_motion_reference is None and not clone_shared_motion_reference:
            motion_reference.load_motion_reference_data(self)
            return
        motion_reference.load_motion_reference_data(
            self,
            shared_motion_reference=shared_motion_reference,
            clone_shared_motion_reference=clone_shared_motion_reference,
        )

    def _load_single_motion_reference_data(
        self,
        motion_file: str,
        *,
        target_dt: float | None = None,
    ) -> dict[str, torch.Tensor]:
        return motion_reference.load_single_motion_reference_data(
            self,
            motion_file,
            target_dt=target_dt,
        )

    def _can_parallelize_dones_and_rewards(self) -> bool:
        return termination_runtime.can_parallelize_dones_and_rewards(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return termination_runtime.get_dones(self)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return observation_runtime.get_observations(self)

    def _get_rewards(self) -> torch.Tensor:
        return reward_runtime.get_rewards(self)

    def _post_dones_and_rewards(
        self, nonfinite_state_mask: torch.Tensor | None
    ) -> None:
        if termination_runtime.finalize_reward_termination(self):
            self._combine_done_outputs(nonfinite_state_mask)
        reward_runtime.accumulate_reward_metrics(self)

    def _prepare_substep_rewards(self) -> None:
        reward_runtime.prepare_substep_rewards(self)

    def _accumulate_substep_reward(self, substep_idx: int) -> None:
        reward_runtime.accumulate_substep_reward(self, substep_idx)

    def _post_physics_substep(self, substep_idx: int) -> None:
        if self._joint_velocity_guard_enabled:
            joint_velocity_guard.apply_substep(self)

    def get_training_diagnostic_means(self) -> dict[str, float]:
        return reward_runtime.get_training_diagnostic_means(self)

    def configure_musclenn_rollout_storage(self, rollout_steps: int) -> None:
        controller = getattr(self, "fast_muscle", None)
        if controller is not None:
            controller.configure_musclenn_rollout_storage(rollout_steps)

    def prepare_musclenn_rollout_sampling(self) -> None:
        controller = getattr(self, "fast_muscle", None)
        if controller is not None:
            controller.prepare_musclenn_rollout_sampling()

    def set_musclenn_rollout_step(self, step: int) -> None:
        controller = getattr(self, "fast_muscle", None)
        if controller is not None:
            controller.set_musclenn_rollout_step(step)

    def mark_musclenn_rollout_collected(self) -> None:
        controller = getattr(self, "fast_muscle", None)
        if controller is not None:
            controller.mark_musclenn_rollout_collected()

    def optimize_muscle_nn(self) -> dict[str, float] | None:
        controller = getattr(self, "fast_muscle", None)
        if controller is None:
            return None
        return controller.optimize_muscle_nn()

    def get_reference_tracking_metrics(self, **kwargs) -> dict[str, torch.Tensor]:
        return reward_runtime.get_reference_tracking_metrics(self, **kwargs)

    def get_eval_result_metric_specs(self) -> dict[str, EvalResultMetricSpec]:
        return {
            "tracking_valid": EvalResultMetricSpec(
                label="Tracking valid",
                result_key="valid",
                finite_required=False,
                statistic="flag",
            ),
            "ape_mean": EvalResultMetricSpec(
                label="APE mean (m)", direction="minimize"
            ),
            "ape_last": EvalResultMetricSpec(
                label="APE last (m)", direction="minimize", statistic="last"
            ),
            "global_mpjpe_mean": EvalResultMetricSpec(
                label="global MPJPE mean (m)", direction="minimize"
            ),
            "global_mpjpe_last": EvalResultMetricSpec(
                label="global MPJPE last (m)",
                direction="minimize",
                statistic="last",
            ),
            "ee_mean": EvalResultMetricSpec(label="EE mean (m)", direction="minimize"),
            "ee_last": EvalResultMetricSpec(
                label="EE last (m)", direction="minimize", statistic="last"
            ),
            "vel_mean": EvalResultMetricSpec(
                label="Velocity mean (m/s)", direction="minimize"
            ),
            "vel_last": EvalResultMetricSpec(
                label="Velocity last (m/s)", direction="minimize", statistic="last"
            ),
            "accel_mean": EvalResultMetricSpec(
                label="Acceleration mean (m/s^2)", direction="minimize"
            ),
            "accel_last": EvalResultMetricSpec(
                label="Acceleration last (m/s^2)",
                direction="minimize",
                statistic="last",
            ),
        }

    def get_eval_episode_spec(self) -> EvalEpisodeSpec:
        # Motion-end termination ends an episode at its clip's last frame; a
        # start at frame 0 of the longest clip runs the most steps.  Without it
        # the reference loops and episodes have no natural end.
        natural_horizon_steps = (
            max(self.motion_clip_frame_counts) - 1
            if self.termination_settings.motion_end
            else None
        )
        return EvalEpisodeSpec(
            validity="full_horizon",
            success="full_horizon",
            natural_completion=self.motion_end_time_outs,
            natural_horizon_steps=natural_horizon_steps,
        )

    def _build_render_model(self):
        return render_runtime.build_render_model(self)

    def _update_render_data(self) -> None:
        render_runtime.update_render_data(self)

    def _append_dynamic_render_geoms(self, *, scene, render_model, render_data) -> None:
        render_runtime.append_dynamic_render_geoms(
            self,
            scene=scene,
            render_model=render_model,
            render_data=render_data,
        )

    def _dynamic_render_maxgeom(self) -> int:
        return render_runtime.dynamic_render_maxgeom(self)

    def dynamic_gl_render_sources(self) -> list[object]:
        return render_runtime.dynamic_gl_render_sources(self)

    def _reset_idx(self, reset_mask: torch.Tensor) -> None:
        reset_runtime.reset_idx(self, reset_mask)

    def get_initial_state_spec(self):
        return initial_state_runtime.get_spec(self)

    def bind_initial_state_parameters(
        self,
        *,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        version: torch.Tensor,
        training: bool,
    ) -> None:
        initial_state_runtime.bind_parameters(
            self,
            mean=mean,
            log_std=log_std,
            version=version,
            training=training,
        )

    def initial_state_sample_metadata(self):
        return initial_state_runtime.sample_metadata(self)

    def clear_initial_state_pending(self) -> None:
        initial_state_runtime.clear_pending(self)

    def set_analysis_reset_overrides(
        self,
        *,
        clip_indices: torch.Tensor,
        start_frame_indices: torch.Tensor,
        enabled: torch.Tensor | None = None,
    ) -> None:
        reset_runtime.set_analysis_reset_overrides(
            self,
            clip_indices=clip_indices,
            start_frame_indices=start_frame_indices,
            enabled=enabled,
        )

    def clear_analysis_reset_overrides(self) -> None:
        reset_runtime.clear_analysis_reset_overrides(self)

    def set_exp_dist(self, distribution) -> None:
        adaptive_sampling_runtime.set_exp_dist(self, distribution)

    def set_adaptive_sampling_benchmark_prior(self, distribution) -> None:
        adaptive_sampling_runtime.set_adaptive_sampling_benchmark_prior(
            self, distribution
        )

    def clear_adaptive_sampling_benchmark_prior(self) -> None:
        adaptive_sampling_runtime.clear_adaptive_sampling_benchmark_prior(self)

    def update_adaptive_sampling_distribution_after_rollout(
        self,
        *,
        wp_stream: Any | None = None,
        torch_stream: torch.cuda.Stream | None = None,
    ) -> None:
        adaptive_sampling_runtime.update_adaptive_sampling_distribution_after_rollout(
            self,
            wp_stream=wp_stream,
            torch_stream=torch_stream,
        )

    def adaptive_sampling_requires_policy_scores(self) -> bool:
        return bool(
            self.adaptive_sampling_state.sampling_enabled
            and self.adaptive_sampling_state.sampling_method
            == ADAPTIVE_SAMPLING_METHOD_MOTION_PLR
        )

    def adaptive_sampling_episode_start_unit_indices(self) -> torch.Tensor:
        return self.adaptive_sampling_state.episode_start_unit_idx

    def update_motion_plr_distribution_after_policy_update(
        self,
        *,
        level_ids: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        wp_stream: Any | None = None,
        torch_stream: torch.cuda.Stream | None = None,
    ) -> None:
        adaptive_sampling_runtime.update_motion_plr_distribution_after_policy_update(
            self,
            level_ids=level_ids,
            returns=returns,
            values=values,
            wp_stream=wp_stream,
            torch_stream=torch_stream,
        )

    def get_adaptive_sampling_state(self) -> dict[str, object] | None:
        return adaptive_sampling_runtime.get_adaptive_sampling_state(self)

    def get_checkpoint_state(self) -> dict[str, object]:
        return adaptive_sampling_runtime.get_checkpoint_state(self)

    def load_checkpoint_state(self, state: dict[str, object] | None) -> bool:
        return adaptive_sampling_runtime.load_checkpoint_state(self, state)

    def load_policy_init_state(self, state: dict[str, object] | None) -> bool:
        loader = getattr(self._control_pipeline, "load_policy_init_state", None)
        return bool(loader(state)) if callable(loader) else False

    def __init__(
        self,
        cfg: SNUImitationEnvCfg,
        render_mode: str | None = None,
        shared_motion_reference: Any | None = None,
        clone_shared_motion_reference: bool = False,
        **kwargs,
    ):
        self.reward_runtime_state = None
        self.adaptive_sampling_state = None
        self.initial_state_runtime_state = None
        self.reset_runtime_state = None
        self.observation_runtime_state = None
        self.action_settings = ActionSettings.from_env_cfg(cfg)
        policy_output = getattr(self.action_settings, "policy_output", None)
        self._use_reference_action = bool(
            getattr(self.action_settings, "use_reference", False)
            or (
                self.action_settings.type == "mass"
                and getattr(policy_output, "type", None) == "pose"
                and getattr(policy_output, "reference", None) == "motion"
            )
        )
        self.motion_settings = self._resolve_motion_settings(cfg)
        self.nan_debug_settings = NanDebugSettings.from_env_cfg(cfg)
        if getattr(cfg, "control", None) is None:
            from instantmimic.envs.snu.tasks.snu_imitation.control_resolver import (
                make_default_control_cfg,
            )

            control_cfg = make_default_control_cfg(cfg)
            self._control_pipeline = control_cfg.build_control_pipeline()
        startup_profiler = StartupProfiler(
            range_factory=lambda name: startup_nvtx_range(f"snu_env:{name}")
        )
        with startup_profiler.phase("direct_rl_env_init"):
            super().__init__(cfg, render_mode, **kwargs)
        with startup_profiler.phase("done_reason_tracking"):
            self._configure_done_reason_tracking(("Torso height", "Early termination"))
        with startup_profiler.phase("body_indices"):
            model_metadata.init_body_indices(self)
        with startup_profiler.phase("joint_range"):
            if self.model.nu > 0:
                self.joint_range = wp.to_torch(self.mjw_model.actuator_ctrlrange)[
                    0
                ].clone()
            else:
                self.joint_range = torch.zeros(
                    (0, 2), device=self.device, dtype=torch.float32
                )
        with startup_profiler.phase("joint_maps"):
            model_metadata.build_joint_maps(self)
        with startup_profiler.phase("episode_state"):
            reset_runtime.initialize_episode_state(self)
        with startup_profiler.phase("task_settings"):
            self.reward_settings = reward_runtime.resolve_reward_settings(
                self.cfg,
                motion_settings=self.motion_settings,
            )
            self.reset_settings = reset_runtime.resolve_reset_settings(self.cfg)
            self.termination_settings = (
                termination_runtime.resolve_termination_settings(self.cfg)
            )
            self.observation_settings = (
                observation_runtime.resolve_observation_settings(
                    self.cfg,
                    joint_pose_reward_mode=self.reward_settings.joint_pose_mode,
                    joint_velocity_reward_mode=self.reward_settings.joint_velocity_mode,
                )
            )
        with startup_profiler.phase("motion_reference_data"):
            if shared_motion_reference is None and not clone_shared_motion_reference:
                self._load_motion_reference_data()
            else:
                self._load_motion_reference_data(
                    shared_motion_reference=shared_motion_reference,
                    clone_shared_motion_reference=clone_shared_motion_reference,
                )
        with startup_profiler.phase("termination_state"):
            termination_runtime.initialize_termination_state(self)
        with startup_profiler.phase("observation_state"):
            observation_runtime.initialize_observation_state(self)
        with startup_profiler.phase("finalize_task_control_pipeline"):
            self.finalize_task_control_pipeline()
        with startup_profiler.phase("reward_state"):
            reward_runtime.initialize_reward_state(self)
        with startup_profiler.phase("reset_behavior"):
            reset_runtime.initialize_reset_behavior(self)
        with startup_profiler.phase("adaptive_sampling"):
            adaptive_sampling_runtime.initialize(self)
            # Both halves of the reset-sampling settings are resolved by now.
            reset_runtime.initialize_reset_sampling_settings(self)
        with startup_profiler.phase("initial_state"):
            initial_state_runtime.initialize(self)
        with startup_profiler.phase("kernel_handles"):
            observation_runtime.initialize_observation_kernels(self)
            reward_runtime.initialize_reward_kernel(self)
        with startup_profiler.phase("nan_debug"):
            nan_debug.initialize_nan_debug(self)
        with startup_profiler.phase("joint_velocity_guard"):
            joint_velocity_guard.initialize(self)
        startup_profiler.log_summary(
            logger=logging.getLogger(__name__),
            prefix="SNU_ENV_STARTUP_PROFILE",
        )

    def seed(self, seed: int = -1) -> int:
        resolved_seed = super().seed(seed)
        if self.reset_runtime_state is not None:
            self.reset_runtime_state.seed.fill_(int(resolved_seed))
        return resolved_seed

    def get_video_overlay_plots(self, env_idx: int = 0) -> dict[str, Any]:
        return reward_runtime.get_video_overlay_plots(self, env_idx=env_idx)
