from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mujoco
import torch

from instantmimic.envs.control_pipeline import (
    ActionScaleSnapshot,
    ControlRuntimeSpec,
    NoControlPipeline,
)
from instantmimic.muscle.parsing.mjcf import FastMuscleBootstrap
from instantmimic.utils.mjw_kernels.private import DisableBit

from .bindings import (
    FastMuscleExecutionBindings,
    FastMuscleJointTopologyBindings,
    FastMuscleMotionReferenceBindings,
)
from .config_models import (
    MassActionSettings,
    MassPosePolicyOutputSettings,
    set_runtime_action_space,
    set_runtime_observation_space,
)
from .controller import FastMuscleController
from .settings import FastMuscleControlSettings

_JOINT_TOPOLOGY_NAMES = (
    "hinge_dof_indices",
    "hinge_body_indices",
    "hinge_parent_body_indices",
    "hinge_axis_local",
    "ball_dof_scatter",
)


def _bind_joint_topology_if_present(
    controller: FastMuscleController,
    host: Any,
) -> None:
    values = {name: getattr(host, name, None) for name in _JOINT_TOPOLOGY_NAMES}
    present = {name for name, value in values.items() if value is not None}
    if not present:
        return
    invalid = [
        name for name, value in values.items() if not isinstance(value, torch.Tensor)
    ]
    if invalid:
        raise TypeError(
            "FastMuscle host joint topology is partial or invalid; expected tensors "
            f"for all fields, invalid={invalid}."
        )
    controller.bind_joint_topology(FastMuscleJointTopologyBindings.from_host(host))


def make_fast_muscle_runtime_spec(
    controller: FastMuscleController,
) -> ControlRuntimeSpec:
    policy_action_scale_snapshot: ActionScaleSnapshot | None = (
        controller.get_policy_action_scale_snapshot()
    )
    policy_transform = controller.get_policy_transform_semantics()
    return ControlRuntimeSpec(
        action_mode=controller.action_mode,
        policy_action_dim=int(controller.policy_action_dim),
        policy_action_scale_snapshot=policy_action_scale_snapshot,
        squash_actions=policy_transform.squash_actions,
        clamp_mean=policy_transform.clamp_mean,
        clamp_mean_min=policy_transform.clamp_mean_min,
        clamp_mean_max=policy_transform.clamp_mean_max,
    )


@dataclass(frozen=True, slots=True)
class FastMuscleBootstrapSettings:
    geometry_backend: str = "waypoint"
    muscle_metadata_path: str | None = None
    position_actuator_mjcf_path: str | None = None

    @classmethod
    def from_settings(
        cls,
        settings: FastMuscleControlSettings,
    ) -> FastMuscleBootstrapSettings:
        action = settings.action
        position_actuator_mjcf_path: str | None = None
        if isinstance(action, MassActionSettings) and isinstance(
            action.policy_output,
            MassPosePolicyOutputSettings,
        ):
            position_actuator_mjcf_path = action.policy_output.stable_pd.mjcf_path
        geometry = settings.muscle.geometry
        return cls(
            geometry_backend=geometry.backend,
            muscle_metadata_path=settings.muscle.metadata_path,
            position_actuator_mjcf_path=position_actuator_mjcf_path,
        )


@dataclass(frozen=True, slots=True)
class FastMuscleControlCfg:
    settings: FastMuscleControlSettings | None = None
    bootstrap: FastMuscleBootstrapSettings | None = None
    controller: FastMuscleController | None = None
    render_adapter: object | None = None

    def __post_init__(self) -> None:
        if self.settings is None:
            return
        expected = FastMuscleBootstrapSettings.from_settings(self.settings)
        if self.bootstrap is None:
            object.__setattr__(self, "bootstrap", expected)
        elif self.bootstrap != expected:
            raise ValueError(
                "FastMuscle bootstrap settings must be derived from the supplied "
                "control settings."
            )

    def build_control_pipeline(self) -> FastMuscleControl:
        return FastMuscleControl(self)


class FastMuscleControl(NoControlPipeline):
    def __init__(
        self,
        cfg: FastMuscleControlCfg | None = None,
        *,
        controller: Any | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else FastMuscleControlCfg()
        self.env: Any | None = None
        self.controller: Any | None = (
            controller if controller is not None else self.cfg.controller
        )
        self.bootstrap: FastMuscleBootstrap | None = None

    @property
    def is_enabled(self) -> bool:
        return True

    def build_sim_spec(
        self,
        mjcf_path: str,
        default_builder: Callable[[str], mujoco.MjSpec],
    ) -> mujoco.MjSpec:
        if self.controller is not None:
            return default_builder(mjcf_path)
        settings = self.cfg.settings
        if settings is None:
            raise RuntimeError("FastMuscleControl requires parsed action settings.")
        action = settings.action
        bootstrap_settings = self.cfg.bootstrap
        if bootstrap_settings is None:
            raise RuntimeError("FastMuscleControl requires bootstrap settings.")
        uses_mass_pose = isinstance(action, MassActionSettings) and isinstance(
            action.policy_output,
            MassPosePolicyOutputSettings,
        )
        self.bootstrap = FastMuscleBootstrap.from_mjcf(
            mjcf_path,
            position_actuator_mjcf_path=bootstrap_settings.position_actuator_mjcf_path,
            geometry_backend=bootstrap_settings.geometry_backend,
            muscle_metadata_path=bootstrap_settings.muscle_metadata_path,
        )
        if uses_mass_pose:
            self.bootstrap.sim_spec.option.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        return self.bootstrap.sim_spec

    def resolve_source_mjcf_path(self, default_path: str) -> str:
        if self.bootstrap is not None:
            return str(self.bootstrap.source_mjcf_path)
        if self.controller is not None:
            source_path = self.controller.source_mjcf_path
            if source_path:
                return str(source_path)
        return default_path

    def finalize_after_model_compile(self, env: Any) -> None:
        self.env = env
        if self.controller is None:
            if self.bootstrap is None:
                raise RuntimeError(
                    "FastMuscleControl requires build_sim_spec() before finalize."
                )
            self.controller = FastMuscleController.attach_to_host(
                env,
                env.cfg,
                self.bootstrap,
                settings=self.cfg.settings,
            )
        env.fast_muscle = self.controller
        if self.controller.requires_native_actuation_disabled:
            env.mjw_model.opt.disableflags = int(
                env.mjw_model.opt.disableflags | int(DisableBit.ACTUATION)
            )

    def finalize_task_state(self, env: Any) -> None:
        self.env = env
        controller = self._require_controller()
        if not controller.runtime_state_finalized:
            dof_roles = getattr(env, "dof_roles", None)
            if dof_roles is not None:
                controller.bind_dof_roles(dof_roles)
            _bind_joint_topology_if_present(controller, env)
            controller.bind_execution(FastMuscleExecutionBindings.from_host(env))
            ref_frame_idx = getattr(env, "ref_frame_idx", None)
            motion = (
                FastMuscleMotionReferenceBindings(
                    motion_qpos=getattr(env, "motion_qpos", None),
                    motion_ctrl=getattr(env, "motion_ctrl", None),
                    reference_body_state=getattr(env, "reference_body_state", None),
                    ref_frame_idx=ref_frame_idx,
                    motion_frame_count=int(env.motion_frame_count),
                    multi_motion_enabled=bool(env._multi_motion_enabled),
                    motion_clip_idx=getattr(env, "motion_clip_idx", None),
                    motion_clip_start=getattr(env, "motion_clip_start", None),
                    motion_clip_length=getattr(env, "motion_clip_length", None),
                )
                if isinstance(ref_frame_idx, torch.Tensor)
                else None
            )
            controller.bind_motion_reference(motion)
            controller.finalize_runtime_state()
        if getattr(env, "render_mode", None) == "rgb_array":
            from . import render_utils

            render_utils.apply_character_geom_render_visibility(env)
        controller.attach_render_assets(
            render_mode=getattr(env, "render_mode", None),
            render_model=getattr(env, "render_model", env.model),
        )

    def configure_spaces(self, env: Any) -> None:
        controller = self._require_controller()
        set_runtime_action_space(env.cfg, controller.policy_action_dim)
        observation_layout = getattr(env, "_observation_layout", None)
        base_obs_dim = (
            int(observation_layout.obs_dim)
            if observation_layout is not None
            else int(getattr(env, "_obs_main_dim", 0))
        )
        obs_main_dim = base_obs_dim + int(controller.applied_torque_obs_dim)
        controller.configure_observation_layout(obs_main_dim)
        env._obs_main_dim = obs_main_dim
        set_runtime_observation_space(
            env.cfg,
            obs_main_dim + int(controller.muscle_geometry_observation_dim),
        )

    def get_prev_action_observation_dim(self) -> int:
        return int(self._require_controller().prev_action_obs_dim)

    def runtime_spec(self) -> ControlRuntimeSpec:
        return make_fast_muscle_runtime_spec(self._require_controller())

    def build_render_model(self, env: Any):
        if self.bootstrap is None:
            return None
        adapter = self.cfg.render_adapter
        builder = getattr(adapter, "build_render_model", None)
        if not callable(builder):
            return None
        return builder(self, env)

    def update_render_data(self) -> None:
        self._require_controller().update_render_tendon_rgba()

    def append_dynamic_render_geoms(self, *, scene: Any, render_data: Any) -> None:
        append_scene_geoms = getattr(
            self._require_controller(),
            "append_render_scene_geoms",
            None,
        )
        if callable(append_scene_geoms):
            append_scene_geoms(scene=scene, render_data=render_data)

    def dynamic_gl_render_sources(self) -> list[object]:
        source_getter = getattr(
            self._require_controller(),
            "dynamic_gl_render_sources",
            None,
        )
        if not callable(source_getter):
            return []
        return list(source_getter())

    def dynamic_render_maxgeom(self) -> int:
        capacity_hint = getattr(
            self._require_controller(),
            "render_scene_geom_capacity_hint",
            None,
        )
        if not callable(capacity_hint):
            return 0
        return int(capacity_hint())

    def get_prev_action_observation_tensor(self) -> torch.Tensor:
        return self._require_controller().get_prev_action_observation_tensor()

    def write_observation_extension(self, target: torch.Tensor, offset: int) -> None:
        controller = self._require_controller()
        if target.shape[-1] < offset:
            return
        if controller.applied_torque_obs_dim > 0:
            observation_layout = getattr(self.env, "_observation_layout", None)
            torque_offset = (
                int(observation_layout.obs_dim)
                if observation_layout is not None
                else int(getattr(self.env, "_task_obs_dim", offset))
            )
            controller.update_applied_torque_observation()
            controller.write_applied_torque_block(target[:, :offset], torque_offset)
        controller.write_muscle_geometry_observation(target, offset)

    def get_checkpoint_state(self) -> dict[str, object]:
        return self._require_controller().get_checkpoint_state()

    def load_checkpoint_state(self, state: dict[str, object] | None) -> bool:
        return bool(self._require_controller().load_checkpoint_state(state))

    def load_policy_init_state(self, state: dict[str, object] | None) -> bool:
        return bool(self._require_controller().load_policy_init_state(state))

    def copy_runtime_state_from(self, source: object) -> bool:
        if not isinstance(source, FastMuscleControl):
            raise TypeError(
                "FastMuscle runtime state source is not a control pipeline."
            )
        return bool(
            self._require_controller().copy_runtime_state_from(
                source._require_controller()
            )
        )

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        self._require_controller().pre_physics_step(actions)
        if self.env is not None and hasattr(self.env, "ref_frame_idx"):
            self.env.ref_frame_idx += 1

    def apply_action(self) -> None:
        self._require_controller().apply_action()

    def post_physics_substep(self, substep_idx: int) -> None:
        self._require_controller().post_physics_substep(substep_idx)

    def post_physics_substep_reads_geometry(self) -> bool:
        return self._require_controller().post_physics_substep_reads_geometry()

    def post_physics_substep_writes_state(self) -> bool:
        return self._require_controller().post_physics_substep_writes_state()

    def full_video_activation_width(self) -> int:
        return int(self._require_controller().muscle_count)

    def get_full_video_activation(self) -> torch.Tensor | None:
        return self._require_controller().get_observation_muscle_activation()

    def reset_idx(self, reset_mask: torch.Tensor) -> None:
        self._require_controller().reset_idx(reset_mask)

    def _require_controller(self) -> Any:
        if self.controller is None:
            raise RuntimeError("FastMuscleControl has not been finalized.")
        return self.controller
