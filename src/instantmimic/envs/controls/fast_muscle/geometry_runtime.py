from __future__ import annotations

import mujoco
import numpy as np
import torch
import warp as wp

from instantmimic.muscle.parsing.mjcf import ParsedMjcfMuscleData
from instantmimic.muscle.waypoint_wrap import (
    WaypointWrapping,
    WaypointWrapState,
    make_wrap_spec,
)

from .settings import AutoScale, ScalarScale, VectorScale


class FastMuscleGeometryMixin:
    def _load_muscle_data(self) -> None:
        parsed_mjcf_data: ParsedMjcfMuscleData | None = getattr(
            self,
            "_parsed_mjcf_muscle_data",
            None,
        )
        if parsed_mjcf_data is None:
            raise ValueError("FastMuscleController requires parsed MJCF muscle data.")

        if not isinstance(parsed_mjcf_data, ParsedMjcfMuscleData):
            raise TypeError("FastMuscleController requires parsed MJCF muscle data.")
        packed = parsed_mjcf_data.packed

        def resolve_body_ids(body_names, *, context: str) -> np.ndarray:
            cache: dict[str, int] = {}
            body_ids: list[int] = []
            for body_name in body_names:
                body_name_str = str(body_name)
                body_id = cache.get(body_name_str)
                if body_id is None:
                    body_id = mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        body_name_str,
                    )
                    if body_id < 0:
                        raise ValueError(
                            f"Body '{body_name_str}' referenced by {context} not found in stripped model."
                        )
                    cache[body_name_str] = body_id
                body_ids.append(body_id)
            return np.asarray(body_ids, dtype=np.int32)

        if packed is not None:
            muscle_names = list(packed.muscle_names)
            if len(muscle_names) == 0:
                raise ValueError(
                    f"No muscles parsed from source MJCF: {getattr(self, '_source_mjcf_path', '')}"
                )
            muscle_params = np.asarray(packed.muscle_params, dtype=np.float32).reshape(
                -1,
                5,
            )
            local_waypoints = np.asarray(
                packed.waypoint_positions, dtype=np.float32
            ).reshape(-1, 3)
            waypoint_body = resolve_body_ids(
                packed.waypoint_body_names,
                context="source MJCF muscle waypoint",
            )
            waypoint_start = np.asarray(packed.waypoint_start, dtype=np.int32)
            attachment_start = np.asarray(packed.attachment_start, dtype=np.int32)
            attachment_body = resolve_body_ids(
                packed.attachment_body_names,
                context="LBS muscle attachment",
            )
            attachment_local = np.asarray(
                packed.attachment_positions, dtype=np.float32
            ).reshape(-1, 3)
            attachment_weight = np.asarray(packed.attachment_weights, dtype=np.float32)
            uses_lbs_waypoint_attachments = bool(packed.uses_lbs_waypoint_attachments)
        else:
            muscles = parsed_mjcf_data.muscles
            if len(muscles) == 0:
                raise ValueError(
                    f"No muscles parsed from source MJCF: {getattr(self, '_source_mjcf_path', '')}"
                )

            muscle_params_list = []
            local_waypoints_list = []
            waypoint_body_list = []
            waypoint_start_list = [0]
            attachment_start_list = [0]
            attachment_body_list = []
            attachment_local_list = []
            attachment_weight_list = []
            uses_lbs_waypoint_attachments = False
            muscle_names = []

            for muscle in muscles:
                muscle_names.append(muscle.name)
                muscle_params_list.append(
                    [muscle.f0, muscle.lm, muscle.lt, muscle.pen_angle, muscle.lmax]
                )
                for waypoint in muscle.waypoints:
                    body_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, waypoint.body
                    )
                    if body_id < 0:
                        raise ValueError(
                            f"Body '{waypoint.body}' referenced in source MJCF not found in stripped model."
                        )
                    local_waypoints_list.append(waypoint.position)
                    waypoint_body_list.append(body_id)

                    attachments = list(getattr(waypoint, "attachments", []) or [])
                    if not attachments:
                        attachments = [waypoint]
                    if len(attachments) != 1:
                        uses_lbs_waypoint_attachments = True
                    for attachment in attachments:
                        attachment_body_name = str(attachment.body)
                        attachment_body_id = mujoco.mj_name2id(
                            self.model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            attachment_body_name,
                        )
                        if attachment_body_id < 0:
                            raise ValueError(
                                f"Body '{attachment_body_name}' referenced by an LBS muscle attachment not found in stripped model."
                            )
                        attachment_position = np.asarray(
                            attachment.position, dtype=np.float32
                        ).reshape(3)
                        attachment_w = float(getattr(attachment, "weight", 1.0))
                        if (
                            attachment_body_id != body_id
                            or abs(attachment_w - 1.0) > 1.0e-6
                            or not np.allclose(
                                attachment_position,
                                np.asarray(waypoint.position, dtype=np.float32).reshape(
                                    3
                                ),
                                atol=1.0e-6,
                                rtol=1.0e-6,
                            )
                        ):
                            uses_lbs_waypoint_attachments = True
                        attachment_body_list.append(attachment_body_id)
                        attachment_local_list.append(attachment_position)
                        attachment_weight_list.append(attachment_w)
                    attachment_start_list.append(len(attachment_body_list))
                waypoint_start_list.append(len(local_waypoints_list))

            muscle_params = np.asarray(muscle_params_list, dtype=np.float32)
            local_waypoints = np.asarray(
                local_waypoints_list, dtype=np.float32
            ).reshape(-1, 3)
            waypoint_body = np.asarray(waypoint_body_list, dtype=np.int32)
            waypoint_start = np.asarray(waypoint_start_list, dtype=np.int32)
            attachment_start = np.asarray(attachment_start_list, dtype=np.int32)
            attachment_body = np.asarray(attachment_body_list, dtype=np.int32)
            attachment_local = np.asarray(
                attachment_local_list, dtype=np.float32
            ).reshape(-1, 3)
            attachment_weight = np.asarray(attachment_weight_list, dtype=np.float32)

        self.muscle_count = len(muscle_names)
        self.muscle_names = muscle_names
        self.muscle_params = torch.tensor(
            muscle_params,
            device=self.device,
            dtype=torch.float32,
        )
        self.muscle_waypoint_body_i32 = torch.tensor(
            waypoint_body, device=self.device, dtype=torch.int32
        )
        self.muscle_waypoint_start_i32 = torch.tensor(
            waypoint_start, device=self.device, dtype=torch.int32
        )
        self.muscle_waypoints_local = torch.tensor(
            local_waypoints,
            device=self.device,
            dtype=torch.float32,
        )
        if uses_lbs_waypoint_attachments:
            raise ValueError(
                "Production MASS requires body-fixed waypoint attachments."
            )
        self._observes_waypoint_path_velocity = True
        self._waypoint_body_cvel = wp.empty(
            (self.num_envs, self.model.nbody),
            dtype=wp.spatial_vector,
            device=str(self.device),
        )
        self.muscle_waypoint_attachment_start_i32 = torch.tensor(
            attachment_start, device=self.device, dtype=torch.int32
        )
        self.muscle_waypoint_attachment_body_i32 = torch.tensor(
            attachment_body, device=self.device, dtype=torch.int32
        )
        self.muscle_waypoint_attachment_local = torch.tensor(
            attachment_local,
            device=self.device,
            dtype=torch.float32,
        )
        self.muscle_waypoint_attachment_weight = torch.tensor(
            attachment_weight,
            device=self.device,
            dtype=torch.float32,
        )
        self._uses_lbs_waypoint_attachments = bool(uses_lbs_waypoint_attachments)
        self.muscle_rest_length = torch.tensor(
            np.asarray(parsed_mjcf_data.rest_lengths, dtype=np.float32),
            device=self.device,
            dtype=torch.float32,
        ).clamp_min(1.0e-6)
        parsed_mujoco_l0 = parsed_mjcf_data.mujoco_l0
        if parsed_mujoco_l0 is None:
            force_velocity_length_scale = (
                self.muscle_rest_length * self.muscle_params[:, 1]
            )
        else:
            native_l0 = np.asarray(parsed_mujoco_l0, dtype=np.float32).reshape(-1)
            if native_l0.shape != (self.muscle_count,):
                raise ValueError(
                    "Parsed MuJoCo L0 must contain one value per muscle, "
                    f"got {native_l0.shape} for {self.muscle_count} muscles."
                )
            if not np.all(np.isfinite(native_l0)) or np.any(native_l0 <= 0.0):
                raise ValueError("Parsed MuJoCo L0 values must be positive and finite.")
            force_velocity_length_scale = torch.tensor(
                native_l0,
                device=self.device,
                dtype=torch.float32,
            )
        self.muscle_force_velocity_length_scale = force_velocity_length_scale

        n_waypoints = self.muscle_waypoints_local.shape[0]
        self.transformed_waypoints = torch.zeros(
            (self.num_envs, n_waypoints, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self.muscle_length = torch.zeros(
            (self.num_envs, self.muscle_count),
            device=self.device,
            dtype=torch.float32,
        )
        self._muscle_force_magnitude = torch.zeros(
            (self.num_envs, self.muscle_count),
            device=self.device,
            dtype=torch.float32,
        )
        self._zero_muscle_actions = torch.zeros_like(self.muscle_length)
        self._waypoint_wrap_spec = make_wrap_spec(
            parsed_mjcf_data, self.model, self.device
        )
        self._waypoint_wrapping = WaypointWrapping(
            self._waypoint_wrap_spec, n_waypoints, self.device
        )
        self._waypoint_wrap_state = WaypointWrapState(
            self._waypoint_wrapping.count, self.num_envs, self.device, self.muscle_count
        )
        self._muscle_all_segment_muscle_idx = torch.zeros(
            (0,), device=self.device, dtype=torch.long
        )
        self._muscle_all_segment_start_idx = torch.zeros(
            (0,), device=self.device, dtype=torch.long
        )
        self._muscle_apply_segment_muscle_idx = torch.zeros(
            (0,), device=self.device, dtype=torch.long
        )
        self._muscle_apply_segment_start_idx = torch.zeros(
            (0,), device=self.device, dtype=torch.long
        )
        self._muscle_apply_segment_end_idx = torch.zeros(
            (0,), device=self.device, dtype=torch.long
        )
        self._muscle_apply_segment_body_start_idx = torch.zeros(
            (0,), device=self.device, dtype=torch.long
        )
        self._muscle_apply_segment_body_end_idx = torch.zeros(
            (0,), device=self.device, dtype=torch.long
        )

    def _expand_character_torque_values(
        self, values: list[float], *, fill: float
    ) -> list[float]:
        """Scatter a per-character-DOF vector into the torque space.

        Authored torque vectors may list the character joint DOFs only (the
        torque space without passive DOFs); passive entries take ``fill``.
        """
        passive = tuple(getattr(self, "_passive_torque_indices", ()))
        if not passive:
            return list(values)
        torque_dim = int(self.model.nv - self._root_dof_count)
        if len(values) != torque_dim - len(passive):
            return list(values)
        passive_set = set(passive)
        source = iter(values)
        return [
            float(fill) if i in passive_set else float(next(source))
            for i in range(torque_dim)
        ]

    def _derive_torque_action_scale(self) -> torch.Tensor:
        torque_dim = int(self.model.nv - self._root_dof_count)
        authored_scale = self.settings.effective.action_scale
        if isinstance(authored_scale, (ScalarScale, VectorScale)):
            self._torque_action_scale_auto_from_rest_bounds = False
            torque_scale_cfg = (
                [authored_scale.value]
                if isinstance(authored_scale, ScalarScale)
                else list(authored_scale.values)
            )
            if len(torque_scale_cfg) == 1:
                torque_scale_cfg *= torque_dim
            torque_scale_cfg = self._expand_character_torque_values(
                torque_scale_cfg, fill=0.0
            )
            if len(torque_scale_cfg) != torque_dim:
                raise ValueError(
                    "Explicit torque_action_scale length mismatch: "
                    f"got {len(torque_scale_cfg)}, expected {torque_dim}. "
                    "Use action.scale=auto to request automatic/unit scaling."
                )
            scale = torch.tensor(
                torque_scale_cfg, device=self.device, dtype=torch.float32
            )
        else:
            assert authored_scale is None or isinstance(authored_scale, AutoScale)
            self._torque_action_scale_auto_from_rest_bounds = True
            scale = torch.ones((torque_dim,), device=self.device, dtype=torch.float32)

        # Passive DOFs (env.dof_roles) are unactuated: a zero scale masks them
        # out of the policy action, the applied torques and the residuals.
        passive = list(getattr(self, "_passive_torque_indices", ()))
        if passive:
            scale[passive] = 0.0

        return scale
