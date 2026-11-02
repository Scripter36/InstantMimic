from __future__ import annotations

import torch
import warp as wp

from instantmimic.muscle.muscle_qfrc_projector import (
    compute_body_cvel_kernel,
    eval_muscle_force_magnitude_kernel,
    eval_muscle_force_magnitude_velocity_kernel,
    extract_applied_torque_kernel,
)
from instantmimic.muscle.waypoint_wrap import path_velocity_kernel

from .tendon_activation_visualizer import TendonActivationVisualizer


@wp.kernel(enable_backward=False)
def _extract_applied_torque_indexed_kernel(
    qfrc_smooth: wp.array2d(dtype=wp.float32),
    qfrc_bias: wp.array2d(dtype=wp.float32),
    torque_scale: wp.array(dtype=wp.float32),
    torque_idx: wp.array(dtype=wp.int32),
    root_dof_count: int,
    out_torque: wp.array2d(dtype=wp.float32),
):
    """``extract_applied_torque_kernel`` over the character torque indices."""
    env_idx, slot = wp.tid()
    dof_idx = torque_idx[slot]
    full_idx = dof_idx + root_dof_count
    raw = qfrc_smooth[env_idx, full_idx] - qfrc_bias[env_idx, full_idx]
    scale = torque_scale[dof_idx]
    if scale > 1.0e-8:
        out_torque[env_idx, slot] = raw / scale
    else:
        out_torque[env_idx, slot] = raw


def _controller_type():
    from .controller import FastMuscleController

    return FastMuscleController


class FastMuscleObservationMixin:
    @staticmethod
    def _as_activation_tensor(value: object) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            return value
        return None

    def _resolve_current_muscle_activation_tensor(
        self,
        *,
        kind: str,
    ) -> torch.Tensor | None:
        if self._uses_mass_coordinator:
            return self._as_activation_tensor(self._musclenn_last_activation)
        if self._policy_outputs_activation:
            return self._as_activation_tensor(self.actions)
        return None

    def get_current_muscle_activation(
        self,
        *,
        kind: str = "state",
        clamp: bool = False,
    ) -> torch.Tensor | None:
        """Return the current muscle activation signal through one public path."""

        kind_key = str(kind).strip().lower()
        if kind_key not in {"state", "command", "applied"}:
            raise ValueError(
                f"kind must be one of {{'state', 'command', 'applied'}}, got {kind!r}."
            )
        activation = self._resolve_current_muscle_activation_tensor(kind=kind_key)
        if activation is None or not bool(clamp):
            return activation
        return activation.clamp(
            self._ACTIVATION_LOWER_BOUND,
            self._ACTIVATION_UPPER_BOUND,
        )

    def attach_render_assets(self, *, render_mode: str | None, render_model) -> None:
        if self._activation_vis_enabled and render_mode == "rgb_array":
            self._setup_activation_visualization(render_model)

    def _build_tendon_visualizer(self) -> TendonActivationVisualizer:
        line_kwargs = {
            "waypoint_body": self.muscle_waypoint_body_i32.detach().cpu().numpy(),
            "waypoints_local": self.muscle_waypoints_local.detach().cpu().numpy(),
            "waypoint_start": self.muscle_waypoint_start_i32.detach().cpu().numpy(),
        }
        if self._waypoint_wrapping.count:
            from instantmimic.rendering.muscle_wrap_geometry import (
                WrappedMuscleRenderGeometry,
            )

            geometry = WrappedMuscleRenderGeometry(
                self._waypoint_wrap_spec,
                line_kwargs["waypoint_body"],
                line_kwargs["waypoints_local"],
                line_kwargs["waypoint_start"],
            )
            line_kwargs.update(
                waypoint_body=geometry.expanded_body,
                waypoints_local=geometry.expanded_local,
                waypoint_start=geometry.expanded_starts,
                waypoint_world_resolver=geometry,
            )
        return TendonActivationVisualizer(
            muscle_names=self.muscle_names,
            muscle_count=self.muscle_count,
            activation_resolver=self._resolve_activation_observation_tensor,
            activation_clamp_enabled_resolver=lambda: float(
                self._activation_clamp_enabled
            ),
            activation_lower_bound=self._ACTIVATION_LOWER_BOUND,
            activation_upper_bound=self._ACTIVATION_UPPER_BOUND,
            color_low=self._activation_color_low,
            color_high=self._activation_color_high,
            color_under=self._activation_color_under,
            color_over=self._activation_color_over,
            alpha=self._activation_color_alpha,
            line_width=self._activation_line_width,
            color_interpolation=self._activation_color_interpolation,
            composite_mode=self._activation_line_composite_mode,
            **line_kwargs,
        )

    def _ensure_tendon_visualizer(self) -> TendonActivationVisualizer:
        visualizer = self.__dict__.get("_tendon_visualizer")
        if visualizer is None:
            visualizer = self._build_tendon_visualizer()
            object.__setattr__(self, "_tendon_visualizer", visualizer)
        return visualizer

    def dynamic_gl_render_sources(self) -> list[object]:
        visualizer = self.__dict__.get("_tendon_visualizer")
        if visualizer is None:
            return []
        return list(visualizer.gl_render_sources())

    def _resolve_activation_observation_tensor(self) -> torch.Tensor | None:
        activation = self.get_current_muscle_activation(kind="state")
        if activation is None:
            return None
        if self._activation_clamp_enabled != 0.0:
            self._activation_apply_buffer.copy_(activation)
            self._activation_apply_buffer.clamp_(
                self._ACTIVATION_LOWER_BOUND,
                self._ACTIVATION_UPPER_BOUND,
            )
            activation = self._activation_apply_buffer
        return activation

    def get_observation_muscle_activation(self) -> torch.Tensor | None:
        """Return the activation signal exactly as exposed to observations."""

        return self._resolve_activation_observation_tensor()

    def update_muscle_observables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Muscle length, velocity, force and activation of the current state.

        Velocity follows the instantaneous motion of the body-fixed
        waypoint paths.
        """
        activation = self._resolve_activation_observation_tensor()
        include_active = 1.0 if activation is not None else 0.0
        include_passive = self._muscle_include_passive
        if activation is None:
            activation = self._zero_muscle_actions
        self._transform_muscle_waypoints()
        if self._waypoint_wrapping.count:
            self._evaluate_wrapped_muscle_force(
                activation,
                clamp=self._activation_clamp_enabled,
                include_active=include_active,
                include_passive=include_passive,
            )
            self._ensure_env_muscle_qfrc_projector_state()
            self._launch_waypoint_body_cvel()
            wp.launch(
                path_velocity_kernel,
                dim=(self.num_envs, self.muscle_count),
                inputs=[
                    wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                    wp.from_torch(self.muscle_waypoint_start_i32),
                    wp.from_torch(self.muscle_waypoint_body_i32),
                    self.mjw_data.subtree_com,
                    wp.from_torch(self._body_rootid_i32),
                    self._waypoint_body_cvel,
                ],
                outputs=[wp.from_torch(self._muscle_obs_velocity)],
                device=str(self.device),
            )
            return (
                self.muscle_length,
                self._muscle_obs_velocity,
                self._muscle_force_magnitude,
                activation,
            )
        force_inputs = [
            wp.from_torch(self.muscle_params, dtype=wp.float32),
            wp.from_torch(activation, dtype=wp.float32),
            wp.from_torch(self.muscle_rest_length, dtype=wp.float32),
            self.settings.muscle.force_scale,
            float(self._muscle_length_norm_min),
            float(self._muscle_length_norm_max),
            float(self._activation_clamp_enabled),
            float(include_active),
            float(include_passive),
            self._muscle_active_force_model_params,
            self._muscle_passive_force_model_params,
        ]
        length_outputs = [
            wp.from_torch(self.muscle_length, dtype=wp.float32),
            wp.from_torch(self._muscle_force_magnitude, dtype=wp.float32),
        ]
        velocity = None
        if self._observes_waypoint_path_velocity:
            self._ensure_env_muscle_qfrc_projector_state()
            self._launch_waypoint_body_cvel()
            # Length, force and velocity in one pass over each path.
            wp.launch(
                eval_muscle_force_magnitude_velocity_kernel,
                dim=(self.num_envs, self.muscle_count),
                inputs=[
                    wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                    wp.from_torch(self.muscle_waypoint_start_i32, dtype=wp.int32),
                    wp.from_torch(self.muscle_waypoint_body_i32, dtype=wp.int32),
                    self.mjw_data.subtree_com,
                    wp.from_torch(self._body_rootid_i32, dtype=wp.int32),
                    self._waypoint_body_cvel,
                    *force_inputs,
                ],
                outputs=[
                    *length_outputs,
                    wp.from_torch(self._muscle_obs_velocity, dtype=wp.float32),
                ],
            )
            velocity = self._muscle_obs_velocity
        else:
            wp.launch(
                eval_muscle_force_magnitude_kernel,
                dim=(self.num_envs, self.muscle_count),
                inputs=[
                    wp.from_torch(self.transformed_waypoints, dtype=wp.vec3),
                    wp.from_torch(self.muscle_waypoint_start_i32, dtype=wp.int32),
                    *force_inputs,
                ],
                outputs=length_outputs,
            )
        activation_obs = activation
        return (
            self.muscle_length,
            velocity,
            self._muscle_force_magnitude,
            activation_obs,
        )

    def _launch_waypoint_body_cvel(self) -> None:
        """Body spatial velocities for single-body waypoint path velocities."""
        wp.launch(
            compute_body_cvel_kernel,
            dim=self._waypoint_body_cvel.shape,
            inputs=[
                self._cdof_view,
                self.mjw_data.qvel,
                wp.from_torch(self._body_cvel_dof_offsets_i32, dtype=wp.int32),
                wp.from_torch(self._body_cvel_dof_idx_i32, dtype=wp.int32),
            ],
            outputs=[self._waypoint_body_cvel],
        )

    def update_myo_leg_muscle_observables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        return self.update_muscle_observables()

    def _write_geometry_context_block(
        self,
        target: torch.Tensor,
        offset: int,
    ) -> int:
        offset = _controller_type()._copy_float_block_(
            target,
            offset,
            self._xpos_flat_view,
        )
        offset = _controller_type()._copy_float_block_(
            target,
            offset,
            self._xquat_flat_view,
        )
        offset = _controller_type()._copy_float_block_(
            target,
            offset,
            self._xipos_flat_view,
        )
        offset = _controller_type()._copy_float_block_(
            target,
            offset,
            self._subtree_com_flat_view,
        )
        offset = _controller_type()._copy_float_block_(
            target,
            offset,
            self._cdof_flat_view,
        )
        return offset

    def _write_muscle_layer_context_block(
        self,
        target: torch.Tensor,
        offset: int,
    ) -> int:
        """Write the physical geometry and optional force-velocity state."""
        offset = self._write_geometry_context_block(target, offset)
        if self.settings.muscle.force_velocity.enabled:
            velocity = self._muscle_force_velocity_qvel_context_view
            assert velocity is not None
            offset = _controller_type()._copy_float_block_(target, offset, velocity)
        return offset

    def update_applied_torque_observation(self) -> None:
        if not self._observe_applied_torque:
            return
        torque_idx = getattr(self, "_applied_torque_torque_idx_wp", None)
        if torque_idx is not None:
            wp.launch(
                _extract_applied_torque_indexed_kernel,
                dim=(self.num_envs, self._applied_torque_obs_dim),
                inputs=[
                    self._qfrc_smooth_view,
                    self._qfrc_bias_view,
                    wp.from_torch(self._torque_action_scale, dtype=wp.float32),
                    torque_idx,
                    int(self._root_dof_count),
                ],
                outputs=[self._applied_torque_buf],
            )
            return
        wp.launch(
            extract_applied_torque_kernel,
            dim=(self.num_envs, self._applied_torque_obs_dim),
            inputs=[
                self._qfrc_smooth_view,
                self._qfrc_bias_view,
                wp.from_torch(self._torque_action_scale, dtype=wp.float32),
                int(self._root_dof_count),
            ],
            outputs=[self._applied_torque_buf],
        )

    def write_applied_torque_block(self, target: torch.Tensor, offset: int) -> int:
        if not self._observe_applied_torque or self._applied_torque_buf is None:
            return offset
        return _controller_type()._copy_float_block_(
            target,
            offset,
            self._applied_torque_buf,
        )

    def _setup_activation_visualization(self, render_model) -> None:
        self._ensure_tendon_visualizer().attach(render_model)

    def update_render_tendon_rgba(self) -> None:
        if not self._activation_vis_enabled:
            return
        self._ensure_tendon_visualizer().update()

    def append_render_scene_geoms(self, *, scene, render_data) -> None:
        if not self._activation_vis_enabled:
            return
        self._ensure_tendon_visualizer().append_to_scene(
            scene=scene,
            render_data=render_data,
        )

    def render_scene_geom_capacity_hint(self) -> int:
        if not self._activation_vis_enabled:
            return 0
        return int(self._ensure_tendon_visualizer().scene_geom_capacity_hint())
