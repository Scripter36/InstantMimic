from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
import warp as wp

from instantmimic.envs.controls.fast_muscle.bindings import (
    FastMuscleExecutionBindings,
    FastMuscleMotionReferenceBindings,
)
from instantmimic.envs.controls.fast_muscle.config_models import (
    set_runtime_action_space,
)
from instantmimic.envs.controls.fast_muscle.controller import (
    FastMuscleController,
)
from instantmimic.envs.controls.fast_muscle.env_bridge import (
    FastMuscleEnvMixin,
)
from instantmimic.envs.controls.fast_muscle.settings import FastMuscleControlSettings
from instantmimic.envs.direct_rl_env import DirectRLEnv
from instantmimic.envs.eval_metrics import EvalEpisodeSpec
from instantmimic.envs.freemusco.tasks.freemusco_imitation import (
    runtime as _imitation_runtime,
)
from instantmimic.envs.tracking.reward_kernels import (
    launch_compute_rewards,
)
from instantmimic.motion import (
    MujocoReferencePayload,
    materialize_mujoco_reference_payload,
)
from instantmimic.muscle.parsing.mjcf import FastMuscleBootstrap
from instantmimic.utils.project_paths import resolve_project_path

from .imitation_tracking_reward import (
    build_joint_reward_metadata,
    build_reward_body_indices,
    build_reward_dof_scales,
    build_tracking_reward_kernel,
)

_LOGGER = logging.getLogger(__name__)
mat_to_quat_np = _imitation_runtime.mat_to_quat_np
apply_loop_transform_positions_np = _imitation_runtime.apply_loop_transform_positions_np
apply_loop_transform_quats_np = _imitation_runtime.apply_loop_transform_quats_np
clamp_or_wrap_frame_idx = _imitation_runtime.clamp_or_wrap_frame_idx
frame_cycle_count = _imitation_runtime.frame_cycle_count


def resolve_imitation_runtime(cfg):
    species = str(getattr(cfg, "species", "")).strip().lower()
    if species not in {"chimanoid", "ostrich"}:
        raise ValueError(
            "FreemuscoImitationEnv requires env.species to be one of "
            f"'ostrich' or 'chimanoid', got {species!r}."
        )
    # The generic runtime is shared by all FreeMusco species.  Species-specific
    # names and layouts are carried by cfg.imitation_profile on each env.
    return _imitation_runtime


class FreemuscoImitationEnv(FastMuscleEnvMixin, DirectRLEnv):
    _freemusco_species_name = "freemusco"

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        self.cfg = cfg
        self._fast_muscle_settings = FastMuscleControlSettings.from_cfg(cfg)
        set_runtime_action_space(cfg, self._fast_muscle_settings.action.space)
        self._imitation_runtime = resolve_imitation_runtime(cfg)
        self._freemusco_species_name = str(getattr(self.cfg, "species", "freemusco"))
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._imitation_runtime.initialize(
            self,
            cfg,
            render_mode=render_mode,
            **kwargs,
        )

    def _build_sim_spec(self, mjcf_path: str) -> mujoco.MjSpec:
        self._fast_muscle_bootstrap = FastMuscleBootstrap.from_mjcf(
            mjcf_path,
            rest_keyframe_name=self.cfg.get("reset_keyframe_name", None),
        )
        self._source_mjcf_path = str(self._fast_muscle_bootstrap.source_mjcf_path)
        self._parsed_mjcf_muscle_data = self._fast_muscle_bootstrap.parsed_mjcf_data
        return self._fast_muscle_bootstrap.sim_spec

    @staticmethod
    def _resolve_motion_data_dir(path: str) -> Path:
        return resolve_project_path(path)

    def _init_fast_muscle_runtime(self) -> None:
        self._nq = int(self.model.nq)
        self._nv = int(self.model.nv)
        self.fast_muscle = FastMuscleController.attach_to_host(
            self,
            self.cfg,
            self._fast_muscle_bootstrap,
            settings=self._fast_muscle_settings,
        )
        self._register_post_physics_substep_hook(self.fast_muscle.post_physics_substep)
        self.cfg.action_space = int(self.fast_muscle.policy_action_dim)
        # The base provisional builder may have allocated the authored action
        # width before the controller was attached. The fast-muscle controller
        # owns execution storage, so replace that provisional buffer here before
        # binding execution; the derived-space finalizer never touches it.
        self.actions = torch.zeros(
            (self.num_envs, int(self.fast_muscle.policy_action_dim)),
            device=self.device,
            dtype=torch.float32,
        )
        self._include_muscle_observation_blocks = bool(
            self.cfg.get("include_muscle_observation_blocks", True)
        )

    def _init_motion_runtime_flags(self) -> None:
        self._motion_loop = bool(self.cfg.motion_loop)
        self._deterministic_reset = bool(self.cfg.get("deterministic_reset", False))
        self._random_init = bool(self.cfg.random_init)
        if self._deterministic_reset and self._random_init:
            _LOGGER.info(
                "deterministic_reset: overriding random_init=True -> False "
                "for deterministic playback/eval."
            )
            self._random_init = False
        self._imitation_reward_threshold = float(self.cfg.imitation_reward_threshold)

    def _validate_motion_shapes(
        self,
        *,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        xipos: torch.Tensor,
        ximat: torch.Tensor,
        body_count: int,
    ) -> None:
        expected_qpos = (int(qpos.shape[0]), self._nq)
        expected_qvel = (int(qvel.shape[0]), self._nv)
        if tuple(qpos.shape) != expected_qpos:
            raise ValueError(
                f"Mocap qpos shape mismatch: expected {expected_qpos}, got {tuple(qpos.shape)}."
            )
        if tuple(qvel.shape) != expected_qvel:
            raise ValueError(
                f"Mocap qvel shape mismatch: expected {expected_qvel}, got {tuple(qvel.shape)}."
            )
        if tuple(xipos.shape[1:]) != (body_count, 3):
            raise ValueError(
                "Mocap xipos body layout mismatch: expected "
                f"({body_count}, 3), got {tuple(xipos.shape[1:])}."
            )
        if tuple(ximat.shape[1:]) != (body_count, 3, 3):
            raise ValueError(
                "Mocap ximat body layout mismatch: expected "
                f"({body_count}, 3, 3), got {tuple(ximat.shape[1:])}."
            )

    def _init_mujoco_tensor_views(self) -> None:
        self.qpos_view = wp.to_torch(self.mjw_data.qpos)
        self.qvel_view = wp.to_torch(self.mjw_data.qvel)
        self.xpos_view = wp.to_torch(self.mjw_data.xpos)
        self.geom_xpos_view = wp.to_torch(self.mjw_data.geom_xpos)
        self.xipos_view = wp.to_torch(self.mjw_data.xipos)

    def _compute_reference_render_body_states(
        self,
    ) -> MujocoReferencePayload:
        return materialize_mujoco_reference_payload(
            model=self.model,
            qpos=self._motion_qpos_t,
            qvel=self._motion_qvel_t,
            device=torch.device("cpu"),
            root_body_idx=self.root_body_id,
            include_cvel=True,
            include_subtree_com=False,
        )

    def _init_tracking_reward_runtime(
        self,
        *,
        ee_body_ids: tuple[int, ...],
        talus_body_names: tuple[str, ...] = (),
        toe_body_names: tuple[str, ...],
        loop_root_quat: np.ndarray,
        loop_root_translation: np.ndarray,
        motion_frame_count: int,
    ) -> None:
        self.ee_body_idx = torch.tensor(
            list(ee_body_ids),
            device=self.device,
            dtype=torch.long,
        )
        self.reward_body_idx = build_reward_body_indices(
            model=self.model,
            motion_body_count=self._motion_body_count,
            device=self.device,
        )
        self._joint_reward_meta = build_joint_reward_metadata(
            model=self.model,
            device=self.device,
            talus_body_names=talus_body_names,
            toe_body_names=toe_body_names,
        )
        self.reference_body_state = (
            self._motion_reference_payload.reference_body_state.to(
                device=self.device, dtype=torch.float32
            ).contiguous()
        )
        if self._motion_reference_payload.ref_com is None:
            raise RuntimeError("MuJoCo reference payload is missing root COM.")
        self.ref_com = self._motion_reference_payload.ref_com.to(
            device=self.device, dtype=torch.float32
        ).contiguous()
        self.dof_q_scale, self.dof_qd_scale = build_reward_dof_scales(
            joint_meta=self._joint_reward_meta,
            cfg=self.cfg,
            model=self.model,
            root_body_idx=self.root_body_id,
        )
        self.hinge_qpos_idx_i32 = self._joint_reward_meta.hinge_qpos_indices.to(
            torch.int32
        )
        self.hinge_dof_idx_i32 = self._joint_reward_meta.hinge_dof_indices.to(
            torch.int32
        )
        if self._joint_reward_meta.n_ball_joints > 0:
            self.ball_qpos_adr_i32 = self._joint_reward_meta.ball_qpos_gather[::4].to(
                torch.int32
            )
            self.ball_dof_adr_i32 = self._joint_reward_meta.ball_dof_scatter[::3].to(
                torch.int32
            )
        else:
            self.ball_qpos_adr_i32 = torch.zeros(
                0, device=self.device, dtype=torch.int32
            )
            self.ball_dof_adr_i32 = torch.zeros(
                0, device=self.device, dtype=torch.int32
            )
        self._compute_rewards_kernel = build_tracking_reward_kernel(
            cfg=self.cfg,
            model=self.model,
            device=self.device,
            root_body_idx=self.root_body_id,
            ee_body_idx=self.ee_body_idx,
            reward_body_idx=self.reward_body_idx,
            joint_meta=self._joint_reward_meta,
            loop_root_quat=loop_root_quat,
            loop_root_translation=loop_root_translation,
            motion_frame_count=motion_frame_count,
        )
        self.body_state_frame_sigma_multiplier = torch.ones(
            (motion_frame_count, 4), device=self.device, dtype=torch.float32
        )
        # The shared reward kernel's action-smoothness input; this family does
        # not enable the term, so the kernel never reads it.
        self._action_smoothness_error = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _init_observation_and_reward_buffers(
        self,
        *,
        nbody_obs: int,
        phase_dim: int,
    ) -> None:
        mc = int(self.fast_muscle.muscle_count)
        nbody = int(self.model.nbody)
        self._nbody_obs = int(nbody_obs)
        self._zero_muscle_buf = torch.zeros(
            (self.num_envs, max(mc, 1)),
            dtype=torch.float32,
            device=self.device,
        )

        proprio_dim = 1 + 2 + self._nq + self._nv
        muscle_dim = 4 * mc if self._include_muscle_observation_blocks else 0
        body_state_dim = nbody * 15
        future_ref_dim = self._nbody_obs * 24

        self._body_state_ofs = proprio_dim + muscle_dim
        self._future_ref_ofs = self._body_state_ofs + body_state_dim
        self._phase_ofs = self._future_ref_ofs + future_ref_dim

        self._obs_dim = self._phase_ofs + int(phase_dim)
        self._obs_main_dim = int(self._obs_dim)
        self._observation_layout = SimpleNamespace(
            root_target_dim=0,
            joint_pos_ofs=0,
            future_obs_ofs=self._future_ref_ofs,
            future_obs_dim=future_ref_dim,
            phase_ofs=self._phase_ofs,
            phase_dim=int(phase_dim),
        )
        self._obs_suffix_dim = (
            self.fast_muscle.prev_action_obs_dim
            + self.fast_muscle.applied_torque_obs_dim
            + self.fast_muscle.muscle_geometry_observation_dim
        )
        self.fast_muscle.configure_observation_layout(self._obs_dim)
        total_obs_dim = int(self._obs_dim + self._obs_suffix_dim)
        self._obs_buf = torch.zeros(
            (self.num_envs, total_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.obs_buf = {"policy": self._obs_buf}
        self._reward_buf = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._reward_r_terms_buf = torch.zeros(
            self.num_envs, 5, dtype=torch.float32, device=self.device
        )
        self._termination_reward_r_terms_buf = torch.zeros_like(
            self._reward_r_terms_buf
        )
        self._reward_r_term_names = (
            "pose_reward",
            "velocity_reward",
            "com_reward",
            "ee_pos_reward",
            "ee_rot_reward",
        )

    def _finalize_fast_muscle_runtime(self) -> None:
        self.finalize_derived_spaces(
            observation_spec=int(self._obs_buf.shape[-1]),
            action_spec=self.cfg.action_space,
            observations=self.obs_buf,
        )
        self.fast_muscle.bind_execution(FastMuscleExecutionBindings.from_host(self))
        self.fast_muscle.bind_motion_reference(
            FastMuscleMotionReferenceBindings(
                motion_qpos=self._motion_qpos_t,
                motion_ctrl=None,
                reference_body_state=self.reference_body_state,
                ref_frame_idx=self._motion_frame_idx,
                motion_frame_count=int(self._motion_qpos_t.shape[0]),
                multi_motion_enabled=self._multi_motion_enabled,
                motion_clip_idx=self.motion_clip_idx,
                motion_clip_start=self.motion_clip_start,
                motion_clip_length=self.motion_clip_length,
            )
        )
        self.fast_muscle.finalize_runtime_state()

    def _apply_action(self) -> None:
        self.fast_muscle.apply_action()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._imitation_runtime.pre_physics_step(self, actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._imitation_runtime.get_dones(self)

    def _reset_idx(self, reset_mask: torch.Tensor):
        super()._reset_idx(reset_mask)
        self._imitation_runtime.reset_idx(self, reset_mask)

    def _build_render_model(self):
        return self._imitation_runtime.build_render_model(self)

    def _update_render_data(self):
        self._imitation_runtime.update_render_data(self)

    def _compute_rewards_into(
        self,
        reward_terms: torch.Tensor,
        rewards: torch.Tensor,
    ) -> None:
        launch_compute_rewards(
            self._compute_rewards_kernel,
            dim=(self.num_envs,),
            inputs=[
                self.mjw_data.qpos,
                self.mjw_data.qvel,
                self.mjw_data.xpos,
                self.mjw_data.xquat,
                self.mjw_data.cvel,
                self.mjw_data.subtree_com,
                self._motion_qpos_t,
                self._motion_qvel_t,
                self.reference_body_state,
                self.ref_com,
                self.motion_clip_idx,
                self._motion_frame_idx,
                self.motion_clip_start,
                self.motion_clip_length,
                self.motion_clip_loop_root_quat,
                self.motion_clip_loop_root_pos,
                self.hinge_qpos_idx_i32,
                self.hinge_dof_idx_i32,
                self.ball_qpos_adr_i32,
                self.ball_dof_adr_i32,
                self.dof_q_scale,
                self.dof_qd_scale,
                self.body_state_frame_sigma_multiplier,
                self._action_smoothness_error,
            ],
            outputs=[
                reward_terms,
                rewards,
            ],
        )

    def _get_rewards(self) -> torch.Tensor:
        self._compute_rewards_into(self._reward_r_terms_buf, self.reward_buf)
        self._reward_buf.copy_(self.reward_buf)
        return self.reward_buf

    def _compute_termination_rewards(self) -> None:
        self._compute_rewards_into(
            self._termination_reward_r_terms_buf,
            self._reward_buf,
        )

    def _resolve_observation_output(
        self,
        out_obs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if out_obs is not None:
            return out_obs, {"policy": out_obs}
        return self.obs_buf["policy"], self.obs_buf

    def _get_observations(self, out_obs: torch.Tensor | None = None) -> dict:
        target, observations = self._resolve_observation_output(out_obs)

        if self._include_muscle_observation_blocks:
            muscle_length, muscle_velocity, muscle_force, muscle_activation = (
                self.fast_muscle.update_muscle_observables()
            )
        else:
            muscle_activation = self._zero_muscle_buf
            muscle_force = self._zero_muscle_buf
            muscle_length = self._zero_muscle_buf
            muscle_velocity = self._zero_muscle_buf

        self._imitation_runtime.launch_obs_proprio_kernel(
            self,
            target,
            muscle_activation=muscle_activation,
            muscle_force=muscle_force,
            muscle_length=muscle_length,
            muscle_velocity=muscle_velocity,
        )
        self._imitation_runtime.launch_obs_future_ref_kernel(self, target)

        offset = self._obs_dim
        if self.fast_muscle.prev_action_obs_dim > 0:
            end = offset + self.fast_muscle.prev_action_obs_dim
            target[:, offset:end].copy_(
                self.fast_muscle.get_prev_action_observation_tensor()
            )
            offset = end
        if self.fast_muscle.applied_torque_obs_dim > 0:
            self.fast_muscle.update_applied_torque_observation()
            offset = self.fast_muscle.write_applied_torque_block(target, offset)
        if self.fast_muscle.muscle_geometry_observation_dim > 0:
            self.fast_muscle.write_muscle_geometry_observation(target, offset)
        return observations

    def get_deployable_obs_spec(self) -> dict[str, object]:
        get_spec = getattr(self._imitation_runtime, "get_deployable_obs_spec", None)
        if get_spec is None:
            raise NotImplementedError(
                f"{self._freemusco_species_name} imitation does not expose a "
                "deployable observation spec."
            )
        return get_spec(self)

    def get_eval_episode_spec(self) -> EvalEpisodeSpec:
        return EvalEpisodeSpec(validity="full_horizon", success="full_horizon")

    def get_video_overlay_plots(self, env_idx: int = 0) -> dict[str, object]:
        if env_idx < 0 or env_idx >= self.num_envs:
            return {}
        reward_terms = {
            name: float(self._reward_r_terms_buf[env_idx, idx].detach().cpu().item())
            for idx, name in enumerate(self._reward_r_term_names)
        }
        return {
            "title": "Imitation Reward Overlay",
            "plots": [
                {
                    "title": "tracking reward terms + total",
                    "series": {
                        **reward_terms,
                        "total_reward": float(
                            self._reward_buf[env_idx].detach().cpu().item()
                        ),
                    },
                    "colors": {
                        "pose_reward": (46, 204, 113),
                        "velocity_reward": (241, 196, 15),
                        "com_reward": (155, 89, 182),
                        "ee_pos_reward": (230, 126, 34),
                        "ee_rot_reward": (52, 152, 219),
                        "total_reward": (236, 240, 241),
                    },
                },
            ],
        }

    def configure_musclenn_rollout_storage(self, rollout_steps: int) -> None:
        self.fast_muscle.configure_musclenn_rollout_storage(rollout_steps)

    def prepare_musclenn_rollout_sampling(self) -> None:
        self.fast_muscle.prepare_musclenn_rollout_sampling()

    def set_musclenn_rollout_step(self, step: int) -> None:
        self.fast_muscle.set_musclenn_rollout_step(step)

    def mark_musclenn_rollout_collected(self) -> None:
        # A captured rollout replays without the Python that marks new
        # samples, so the trainer commits them through this hook.
        self.fast_muscle.mark_musclenn_rollout_collected()

    def optimize_muscle_nn(self) -> dict[str, float] | None:
        return self.fast_muscle.optimize_muscle_nn()

    def get_checkpoint_state(self) -> dict[str, object]:
        state = super().get_checkpoint_state()
        state.update(self.fast_muscle.get_checkpoint_state())
        return state

    def load_checkpoint_state(self, state: dict[str, object] | None) -> bool:
        restored = bool(super().load_checkpoint_state(state))
        return bool(self.fast_muscle.load_checkpoint_state(state)) or restored

    def load_policy_init_state(self, state: dict[str, object] | None) -> bool:
        return bool(self.fast_muscle.load_policy_init_state(state))
