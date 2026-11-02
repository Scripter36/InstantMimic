from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from instantmimic.envs.tracking.metrics import quat_diff_to_axis_angle


@dataclass(frozen=True)
class JointStateProjection:
    """Project model state into a stable actuator-like joint coordinate order."""

    ball_output_indices: torch.Tensor
    ball_qpos_indices: torch.Tensor
    ball_dof_indices: torch.Tensor
    ball_gears: torch.Tensor
    ball_reference_quat: torch.Tensor
    hinge_output_indices: torch.Tensor
    hinge_qpos_indices: torch.Tensor
    hinge_dof_indices: torch.Tensor
    hinge_gears: torch.Tensor
    coordinate_dof_indices: torch.Tensor
    coordinate_dof_signs: torch.Tensor
    lower_limits: torch.Tensor
    upper_limits: torch.Tensor

    @property
    def coordinate_count(self) -> int:
        return int(self.lower_limits.numel())

    @classmethod
    def from_models(
        cls,
        *,
        model: mujoco.MjModel,
        coordinate_model: mujoco.MjModel,
        device: torch.device,
    ) -> JointStateProjection:
        if int(model.nq) != int(coordinate_model.nq) or int(model.nv) != int(
            coordinate_model.nv
        ):
            raise ValueError(
                "Simulation and coordinate models must have matching nq/nv; got "
                f"({model.nq}, {model.nv}) and "
                f"({coordinate_model.nq}, {coordinate_model.nv})."
            )
        ball_output_indices: list[int] = []
        ball_qpos_indices: list[list[int]] = []
        ball_dof_indices: list[list[int]] = []
        ball_gears: list[list[float]] = []
        ball_reference_quat: list[list[float]] = []
        hinge_output_indices: list[int] = []
        hinge_qpos_indices: list[int] = []
        hinge_dof_indices: list[int] = []
        hinge_gears: list[float] = []
        coordinate_dof_indices: list[int] = []
        coordinate_dof_signs: list[float] = []

        for actuator_id in range(int(coordinate_model.nu)):
            if int(coordinate_model.actuator_trntype[actuator_id]) != int(
                mujoco.mjtTrn.mjTRN_JOINT
            ):
                raise ValueError(
                    f"Coordinate actuator {actuator_id} must use joint transmission."
                )
            coordinate_joint_id = int(coordinate_model.actuator_trnid[actuator_id, 0])
            joint_name = mujoco.mj_id2name(
                coordinate_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                coordinate_joint_id,
            )
            if not joint_name:
                raise ValueError(
                    f"Coordinate actuator {actuator_id} targets an unnamed joint."
                )
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(
                    f"Coordinate joint {joint_name!r} is missing from the simulation model."
                )

            joint_type = int(model.jnt_type[joint_id])
            coordinate_joint_type = int(coordinate_model.jnt_type[coordinate_joint_id])
            if joint_type != coordinate_joint_type:
                raise ValueError(
                    f"Coordinate joint {joint_name!r} changes type between models."
                )
            qpos_adr = int(model.jnt_qposadr[joint_id])
            dof_adr = int(model.jnt_dofadr[joint_id])
            coordinate_qpos_adr = int(coordinate_model.jnt_qposadr[coordinate_joint_id])
            gear = coordinate_model.actuator_gear[actuator_id, :3]
            if joint_type == mujoco.mjtJoint.mjJNT_BALL:
                active_axes = np.flatnonzero(np.abs(gear) > 1.0e-8)
                if active_axes.size != 1 or not np.isclose(
                    abs(float(gear[active_axes[0]])), 1.0
                ):
                    raise ValueError(
                        f"Coordinate actuator {actuator_id} for {joint_name!r} "
                        "must select one unit ball-joint axis."
                    )
                gear_axis = int(active_axes[0])
                model_qpos0 = model.qpos0[qpos_adr : qpos_adr + 4]
                coordinate_qpos0 = coordinate_model.qpos0[
                    coordinate_qpos_adr : coordinate_qpos_adr + 4
                ]
                if not np.allclose(model_qpos0, coordinate_qpos0):
                    raise ValueError(
                        f"Coordinate joint {joint_name!r} has incompatible qpos0."
                    )
                ball_output_indices.append(actuator_id)
                ball_qpos_indices.append([qpos_adr + offset for offset in range(4)])
                ball_dof_indices.append([dof_adr + offset for offset in range(3)])
                ball_gears.append([float(value) for value in gear])
                ball_reference_quat.append([1.0, 0.0, 0.0, 0.0])
                coordinate_dof_indices.append(dof_adr - 6 + gear_axis)
                coordinate_dof_signs.append(float(np.sign(gear[gear_axis])))
            elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                if not np.isclose(abs(float(gear[0])), 1.0) or not np.allclose(
                    gear[1:], 0.0
                ):
                    raise ValueError(
                        f"Coordinate actuator {actuator_id} for {joint_name!r} "
                        "must use a unit hinge-joint gear."
                    )
                if not np.isclose(
                    model.qpos0[qpos_adr],
                    coordinate_model.qpos0[coordinate_qpos_adr],
                ):
                    raise ValueError(
                        f"Coordinate joint {joint_name!r} has incompatible qpos0."
                    )
                hinge_output_indices.append(actuator_id)
                hinge_qpos_indices.append(qpos_adr)
                hinge_dof_indices.append(dof_adr)
                hinge_gears.append(float(gear[0]))
                coordinate_dof_indices.append(dof_adr - 6)
                coordinate_dof_signs.append(float(np.sign(gear[0])))
            else:
                raise ValueError(
                    f"Unsupported coordinate joint type {joint_type} for {joint_name!r}."
                )

        if sorted(coordinate_dof_indices) != list(range(int(model.nv) - 6)):
            raise ValueError(
                "Coordinate actuators must cover every non-root model DOF exactly once."
            )

        return cls(
            ball_output_indices=torch.tensor(
                ball_output_indices, device=device, dtype=torch.long
            ),
            ball_qpos_indices=torch.tensor(
                ball_qpos_indices, device=device, dtype=torch.long
            ),
            ball_dof_indices=torch.tensor(
                ball_dof_indices, device=device, dtype=torch.long
            ),
            ball_gears=torch.tensor(ball_gears, device=device, dtype=torch.float32),
            ball_reference_quat=torch.tensor(
                ball_reference_quat,
                device=device,
                dtype=torch.float32,
            ),
            hinge_output_indices=torch.tensor(
                hinge_output_indices, device=device, dtype=torch.long
            ),
            hinge_qpos_indices=torch.tensor(
                hinge_qpos_indices, device=device, dtype=torch.long
            ),
            hinge_dof_indices=torch.tensor(
                hinge_dof_indices, device=device, dtype=torch.long
            ),
            hinge_gears=torch.tensor(hinge_gears, device=device, dtype=torch.float32),
            coordinate_dof_indices=torch.tensor(
                coordinate_dof_indices,
                device=device,
                dtype=torch.long,
            ),
            coordinate_dof_signs=torch.tensor(
                coordinate_dof_signs,
                device=device,
                dtype=torch.float32,
            ),
            lower_limits=torch.tensor(
                coordinate_model.actuator_ctrlrange[:, 0],
                device=device,
                dtype=torch.float32,
            ),
            upper_limits=torch.tensor(
                coordinate_model.actuator_ctrlrange[:, 1],
                device=device,
                dtype=torch.float32,
            ),
        )

    def project_policy_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if int(actions.shape[-1]) != self.coordinate_count:
            raise ValueError(
                f"Expected {self.coordinate_count} model-DOF actions, got "
                f"{int(actions.shape[-1])}."
            )
        return actions[:, self.coordinate_dof_indices] * self.coordinate_dof_signs

    def project(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(qpos.shape[0])
        joint_pos = torch.empty(
            (batch_size, self.coordinate_count),
            device=qpos.device,
            dtype=qpos.dtype,
        )
        joint_vel = torch.empty_like(joint_pos)
        self.project_into(qpos, qvel, out_pos=joint_pos, out_vel=joint_vel)
        return joint_pos, joint_vel

    def project_into(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
        *,
        out_pos: torch.Tensor,
        out_vel: torch.Tensor,
    ) -> None:
        expected_shape = (int(qpos.shape[0]), self.coordinate_count)
        if out_pos.shape != expected_shape or out_vel.shape != expected_shape:
            raise ValueError(
                f"Joint projection outputs must have shape {expected_shape}; got "
                f"{tuple(out_pos.shape)} and {tuple(out_vel.shape)}."
            )
        out_pos.zero_()
        out_vel.zero_()

        if self.ball_output_indices.numel() > 0:
            ball_quat = qpos[:, self.ball_qpos_indices]
            reference = self.ball_reference_quat.unsqueeze(0).expand_as(ball_quat)
            ball_axis_angle = quat_diff_to_axis_angle(reference, ball_quat)
            out_pos[:, self.ball_output_indices] = torch.sum(
                ball_axis_angle * self.ball_gears.unsqueeze(0), dim=-1
            )
            ball_velocity = qvel[:, self.ball_dof_indices]
            out_vel[:, self.ball_output_indices] = torch.sum(
                ball_velocity * self.ball_gears.unsqueeze(0), dim=-1
            )

        if self.hinge_output_indices.numel() > 0:
            out_pos[:, self.hinge_output_indices] = qpos[
                :, self.hinge_qpos_indices
            ] * self.hinge_gears.unsqueeze(0)
            out_vel[:, self.hinge_output_indices] = qvel[
                :, self.hinge_dof_indices
            ] * self.hinge_gears.unsqueeze(0)
