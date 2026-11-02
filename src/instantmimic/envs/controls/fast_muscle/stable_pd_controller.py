from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import warp as wp

from instantmimic.motion.warp_utils import clip_global_frame
from instantmimic.utils.mjw_step_optimized import factor_solve_i_no_l_store

from .bindings import FastMuscleMotionReferenceBindings, FastMuscleSimulationBindings
from .config_models import StablePDSettings


@wp.kernel(enable_backward=False)
def _clear_ctrl_and_qfrc(
    ctrl: wp.array2d(dtype=wp.float32),
    qfrc_applied: wp.array2d(dtype=wp.float32),
):
    env_idx, value_idx = wp.tid()
    if value_idx < ctrl.shape[1]:
        ctrl[env_idx, value_idx] = 0.0
    if value_idx < qfrc_applied.shape[1]:
        qfrc_applied[env_idx, value_idx] = 0.0


@wp.kernel(enable_backward=False)
def _write_motion_residual_target(
    motion_ctrl: wp.array2d(dtype=wp.float32),
    ref_frame_idx: wp.array(dtype=wp.int32),
    motion_frame_count: int,
    actions: wp.array2d(dtype=wp.float32),
    target_ctrl: wp.array2d(dtype=wp.float32),
):
    env_idx, actuator_idx = wp.tid()
    frame_idx = ref_frame_idx[env_idx] % motion_frame_count
    target_ctrl[env_idx, actuator_idx] = (
        motion_ctrl[frame_idx, actuator_idx] + actions[env_idx, actuator_idx]
    )


@wp.kernel(enable_backward=False)
def _write_multi_motion_residual_target(
    motion_ctrl: wp.array2d(dtype=wp.float32),
    motion_clip_idx: wp.array(dtype=wp.int32),
    ref_frame_idx: wp.array(dtype=wp.int32),
    clip_start: wp.array(dtype=wp.int32),
    clip_length: wp.array(dtype=wp.int32),
    actions: wp.array2d(dtype=wp.float32),
    target_ctrl: wp.array2d(dtype=wp.float32),
):
    env_idx, actuator_idx = wp.tid()
    clip_idx = motion_clip_idx[env_idx]
    frame_idx = clip_global_frame(
        ref_frame_idx[env_idx],
        clip_idx,
        clip_start,
        clip_length,
    )
    target_ctrl[env_idx, actuator_idx] = (
        motion_ctrl[frame_idx, actuator_idx] + actions[env_idx, actuator_idx]
    )


@wp.kernel(enable_backward=False)
def _prepare_spd_qm_dense(
    target_ctrl: wp.array2d(dtype=wp.float32),
    actuator_length: wp.array2d(dtype=wp.float32),
    actuator_velocity: wp.array2d(dtype=wp.float32),
    kp: wp.array(dtype=wp.float32),
    kv: wp.array(dtype=wp.float32),
    dof_to_actuator: wp.array(dtype=wp.int32),
    qM: wp.array3d(dtype=wp.float32),
    qfrc_bias: wp.array2d(dtype=wp.float32),
    qfrc_constraint: wp.array2d(dtype=wp.float32),
    dt: float,
    regularization: float,
    p_diff: wp.array2d(dtype=wp.float32),
    v_diff: wp.array2d(dtype=wp.float32),
    kv_diag: wp.array2d(dtype=wp.float32),
    rhs: wp.array2d(dtype=wp.float32),
):
    env_idx, dof_idx = wp.tid()
    p = float(0.0)
    v = float(0.0)
    k = float(0.0)

    actuator_idx = dof_to_actuator[dof_idx]
    if actuator_idx >= 0:
        pos_err = (
            target_ctrl[env_idx, actuator_idx] - actuator_length[env_idx, actuator_idx]
        )
        vel_err = -actuator_velocity[env_idx, actuator_idx]
        p = pos_err * kp[actuator_idx]
        v = vel_err * kv[actuator_idx]
        k = kv[actuator_idx]
        qM[env_idx, dof_idx, dof_idx] += dt * k + regularization

    p_diff[env_idx, dof_idx] = p
    v_diff[env_idx, dof_idx] = v
    kv_diag[env_idx, dof_idx] = k
    rhs[env_idx, dof_idx] = (
        -qfrc_bias[env_idx, dof_idx] + p + v + qfrc_constraint[env_idx, dof_idx]
    )


@wp.kernel(enable_backward=False)
def _restore_dense_qm_write_spd_qfrc(
    p_diff: wp.array2d(dtype=wp.float32),
    v_diff: wp.array2d(dtype=wp.float32),
    kv_diag: wp.array2d(dtype=wp.float32),
    ddq: wp.array2d(dtype=wp.float32),
    dt: float,
    regularization: float,
    torque_scale: float,
    torque_limit: float,
    root_dof_count: int,
    qM: wp.array3d(dtype=wp.float32),
    output_qfrc: wp.array2d(dtype=wp.float32),
):
    env_idx, dof_idx = wp.tid()
    if dof_idx < root_dof_count:
        output_qfrc[env_idx, dof_idx] = 0.0
    else:
        torque = (
            p_diff[env_idx, dof_idx]
            + v_diff[env_idx, dof_idx]
            - dt * kv_diag[env_idx, dof_idx] * ddq[env_idx, dof_idx]
        ) * torque_scale
        if torque_limit > 0.0:
            torque = wp.clamp(torque, -torque_limit, torque_limit)
        output_qfrc[env_idx, dof_idx] = torque
        qM[env_idx, dof_idx, dof_idx] -= dt * kv_diag[env_idx, dof_idx] + regularization


@wp.kernel(enable_backward=False)
def _prepare_spd_m_packed(
    target_ctrl: wp.array2d(dtype=wp.float32),
    actuator_length: wp.array2d(dtype=wp.float32),
    actuator_velocity: wp.array2d(dtype=wp.float32),
    kp: wp.array(dtype=wp.float32),
    kv: wp.array(dtype=wp.float32),
    dof_to_actuator: wp.array(dtype=wp.int32),
    m_rownnz: wp.array(dtype=wp.int32),
    m_rowadr: wp.array(dtype=wp.int32),
    mass_matrix: wp.array2d(dtype=wp.float32),
    qfrc_bias: wp.array2d(dtype=wp.float32),
    qfrc_constraint: wp.array2d(dtype=wp.float32),
    dt: float,
    regularization: float,
    p_diff: wp.array2d(dtype=wp.float32),
    v_diff: wp.array2d(dtype=wp.float32),
    kv_diag: wp.array2d(dtype=wp.float32),
    rhs: wp.array2d(dtype=wp.float32),
):
    env_idx, dof_idx = wp.tid()
    p = float(0.0)
    v = float(0.0)
    k = float(0.0)

    actuator_idx = dof_to_actuator[dof_idx]
    if actuator_idx >= 0:
        pos_err = (
            target_ctrl[env_idx, actuator_idx] - actuator_length[env_idx, actuator_idx]
        )
        vel_err = -actuator_velocity[env_idx, actuator_idx]
        p = pos_err * kp[actuator_idx]
        v = vel_err * kv[actuator_idx]
        k = kv[actuator_idx]
        diagonal_idx = m_rowadr[dof_idx] + m_rownnz[dof_idx] - 1
        mass_matrix[env_idx, diagonal_idx] += dt * k + regularization

    p_diff[env_idx, dof_idx] = p
    v_diff[env_idx, dof_idx] = v
    kv_diag[env_idx, dof_idx] = k
    rhs[env_idx, dof_idx] = (
        -qfrc_bias[env_idx, dof_idx] + p + v + qfrc_constraint[env_idx, dof_idx]
    )


@wp.kernel(enable_backward=False)
def _restore_packed_m_write_spd_qfrc(
    p_diff: wp.array2d(dtype=wp.float32),
    v_diff: wp.array2d(dtype=wp.float32),
    kv_diag: wp.array2d(dtype=wp.float32),
    ddq: wp.array2d(dtype=wp.float32),
    dt: float,
    regularization: float,
    torque_scale: float,
    torque_limit: float,
    root_dof_count: int,
    m_rownnz: wp.array(dtype=wp.int32),
    m_rowadr: wp.array(dtype=wp.int32),
    mass_matrix: wp.array2d(dtype=wp.float32),
    output_qfrc: wp.array2d(dtype=wp.float32),
):
    env_idx, dof_idx = wp.tid()
    if dof_idx < root_dof_count:
        output_qfrc[env_idx, dof_idx] = 0.0
    else:
        torque = (
            p_diff[env_idx, dof_idx]
            + v_diff[env_idx, dof_idx]
            - dt * kv_diag[env_idx, dof_idx] * ddq[env_idx, dof_idx]
        ) * torque_scale
        if torque_limit > 0.0:
            torque = wp.clamp(torque, -torque_limit, torque_limit)
        output_qfrc[env_idx, dof_idx] = torque
        diagonal_idx = m_rowadr[dof_idx] + m_rownnz[dof_idx] - 1
        mass_matrix[env_idx, diagonal_idx] -= (
            dt * kv_diag[env_idx, dof_idx] + regularization
        )


def stable_pd_dof_to_actuator(
    model: object,
    root_dof_count: int,
    actuated_dofs: Sequence[int] | None,
) -> list[int]:
    """Map each coordinate to its position actuator, or -1 without one.

    ``actuated_dofs`` lists the non-root coordinates that must carry exactly
    one actuator each (``None``: every non-root coordinate); no other
    coordinate may carry one.
    """
    nv = int(model.nv)
    expected = (
        list(range(root_dof_count, nv))
        if actuated_dofs is None
        else sorted(root_dof_count + int(dof) for dof in actuated_dofs)
    )
    if int(model.nu) != len(expected):
        raise ValueError(
            "MASS pose output requires one position actuator per actuated "
            f"coordinate; got nu={model.nu} for {len(expected)} coordinates."
        )
    dof_to_actuator = [-1] * nv
    for actuator_idx in range(int(model.nu)):
        joint_id = int(model.actuator_trnid[actuator_idx, 0])
        gear = model.actuator_gear[actuator_idx]
        local_dof = max(range(6), key=lambda idx: abs(float(gear[idx])))
        dof_idx = int(model.jnt_dofadr[joint_id]) + local_dof
        if 0 <= dof_idx < nv:
            dof_to_actuator[dof_idx] = actuator_idx
    actuated = [dof for dof, actuator in enumerate(dof_to_actuator) if actuator >= 0]
    if actuated != expected:
        raise ValueError(
            "MASS pose output position actuators must drive exactly the "
            f"coordinates {expected}; they drive {actuated}."
        )
    return dof_to_actuator


class StablePDController:
    """MASS-compatible Stable-PD torque calculator used before MuscleNN."""

    def __init__(
        self,
        *,
        simulation: FastMuscleSimulationBindings,
        motion: FastMuscleMotionReferenceBindings | None,
        root_dof_count: int,
        forward_control_dynamics: Callable[[], None],
        settings: StablePDSettings,
        rest_ctrl: torch.Tensor | None = None,
        actuated_dofs: Sequence[int] | None = None,
    ) -> None:
        """Stable PD around a reference motion or a fixed rest target.

        Exactly one of ``motion`` and ``rest_ctrl`` (one target per actuator)
        is given. ``actuated_dofs`` lists the non-root coordinates that carry
        a position actuator, one each; ``None`` means every non-root
        coordinate. Coordinates without one receive no Stable-PD torque.
        """
        if (motion is None) == (rest_ctrl is None):
            raise ValueError(
                "Stable PD needs exactly one target source: a reference motion "
                "or a rest target."
            )
        self.settings = settings
        self.motion = motion
        self.forward_control_dynamics = forward_control_dynamics
        self.model = simulation.model
        self.mjw_model = simulation.mjw_model
        self.mjw_data = simulation.mjw_data
        self.num_envs = simulation.num_envs
        self.device = simulation.device
        self.physics_dt = simulation.physics_dt
        self.root_dof_count = int(root_dof_count)

        if not 0 <= self.root_dof_count < int(self.model.nv):
            raise ValueError(
                "MASS pose output requires a valid root DOF count; "
                f"got root_dof_count={self.root_dof_count}, nv={self.model.nv}."
            )

        self.target_ctrl = torch.zeros(
            (self.num_envs, int(self.model.nu)),
            device=self.device,
            dtype=torch.float32,
        )
        self.target_ctrl_wp = wp.from_torch(self.target_ctrl, dtype=wp.float32)

        actuator_gainprm = wp.to_torch(self.mjw_model.actuator_gainprm)
        actuator_biasprm = wp.to_torch(self.mjw_model.actuator_biasprm)
        if actuator_gainprm.ndim == 3:
            actuator_gain0 = actuator_gainprm[0, :, 0]
            actuator_bias2 = actuator_biasprm[0, :, 2]
        else:
            actuator_gain0 = actuator_gainprm[:, 0]
            actuator_bias2 = actuator_biasprm[:, 2]
        self.kp = (
            actuator_gain0.clone().to(device=self.device, dtype=torch.float32)
            * settings.kp_scale
        )
        self.kv = (
            torch.clamp(-actuator_bias2, min=0.0)
            .clone()
            .to(device=self.device, dtype=torch.float32)
            * settings.kv_scale
        )
        self.kp_wp = wp.from_torch(self.kp, dtype=wp.float32)
        self.kv_wp = wp.from_torch(self.kv, dtype=wp.float32)

        dof_to_actuator = stable_pd_dof_to_actuator(
            self.model, self.root_dof_count, actuated_dofs
        )
        self.dof_to_actuator = torch.tensor(
            dof_to_actuator, device=self.device, dtype=torch.int32
        )
        self.rest_ctrl = (
            None
            if rest_ctrl is None
            else rest_ctrl.to(device=self.device, dtype=torch.float32).view(1, -1)
        )
        if self.rest_ctrl is not None and int(self.rest_ctrl.shape[1]) != int(
            self.model.nu
        ):
            raise ValueError(
                f"Stable PD rest target has {self.rest_ctrl.shape[1]} entries for "
                f"{self.model.nu} actuators."
            )
        self.dof_to_actuator_wp = wp.from_torch(self.dof_to_actuator, dtype=wp.int32)

        mass_matrix_wp = getattr(self.mjw_data, "M", None)
        if mass_matrix_wp is None:
            mass_matrix_wp = self.mjw_data.qM
        self.mass_matrix_wp = mass_matrix_wp
        mass_matrix = wp.to_torch(mass_matrix_wp)
        self.mass_matrix_is_packed = mass_matrix.ndim == 2
        has_dense_shape = bool(
            mass_matrix.ndim == 3
            and mass_matrix.shape[1] == mass_matrix.shape[2]
            and mass_matrix.shape[1] >= int(self.model.nv)
        )
        has_packed_shape = bool(
            self.mass_matrix_is_packed
            and mass_matrix.shape[0] == self.num_envs
            and mass_matrix.shape[1] > 0
        )
        if not (has_dense_shape or has_packed_shape):
            raise ValueError(
                "MASS StablePDController requires a dense MuJoCo Warp mass matrix; "
                f"got {tuple(mass_matrix.shape)}."
            )

        self.factor = torch.zeros_like(wp.to_torch(self.mjw_data.qLD))
        self.diag_inv = torch.zeros_like(wp.to_torch(self.mjw_data.qLDiagInv))
        self.rhs = torch.zeros_like(self.diag_inv)
        self.ddq = torch.zeros_like(self.diag_inv)
        self.p_diff = torch.zeros_like(self.diag_inv)
        self.v_diff = torch.zeros_like(self.diag_inv)
        self.kv_diag = torch.zeros_like(self.diag_inv)

        self.factor_wp = wp.from_torch(self.factor, dtype=wp.float32)
        self.diag_inv_wp = wp.from_torch(self.diag_inv, dtype=wp.float32)
        self.rhs_wp = wp.from_torch(self.rhs, dtype=wp.float32)
        self.ddq_wp = wp.from_torch(self.ddq, dtype=wp.float32)
        self.p_diff_wp = wp.from_torch(self.p_diff, dtype=wp.float32)
        self.v_diff_wp = wp.from_torch(self.v_diff, dtype=wp.float32)
        self.kv_diag_wp = wp.from_torch(self.kv_diag, dtype=wp.float32)

    def _write_target_ctrl(self, actions: torch.Tensor) -> None:
        if self.rest_ctrl is not None:
            torch.add(actions, self.rest_ctrl, out=self.target_ctrl)
            return
        assert self.motion is not None
        motion_ctrl = self.motion.motion_ctrl
        if motion_ctrl is None:
            raise RuntimeError("MASS pose output requires motion control targets.")
        if self.motion.multi_motion_enabled:
            assert self.motion.motion_clip_idx is not None
            assert self.motion.motion_clip_start is not None
            assert self.motion.motion_clip_length is not None
            wp.launch(
                _write_multi_motion_residual_target,
                dim=(self.num_envs, int(self.model.nu)),
                inputs=[
                    motion_ctrl,
                    self.motion.motion_clip_idx,
                    self.motion.ref_frame_idx,
                    self.motion.motion_clip_start,
                    self.motion.motion_clip_length,
                    actions,
                ],
                outputs=[self.target_ctrl_wp],
            )
        else:
            wp.launch(
                _write_motion_residual_target,
                dim=(self.num_envs, int(self.model.nu)),
                inputs=[
                    motion_ctrl,
                    self.motion.ref_frame_idx,
                    self.motion.motion_frame_count,
                    actions,
                ],
                outputs=[self.target_ctrl_wp],
            )

    def _run_mjw_forward(self) -> None:
        self.forward_control_dynamics()

    def compute_torque(
        self,
        actions: torch.Tensor,
        output_qfrc: torch.Tensor,
    ) -> torch.Tensor:
        output_qfrc_wp = wp.from_torch(output_qfrc, dtype=wp.float32)
        wp.launch(
            _clear_ctrl_and_qfrc,
            dim=(self.num_envs, max(int(self.model.nu), int(self.model.nv))),
            inputs=[self.mjw_data.ctrl, self.mjw_data.qfrc_applied],
        )
        # Required, not redundant with the environment's state prefix. Stable PD
        # reads `qfrc_constraint` below, and the policy boundary leaves the
        # acceleration stage one substep behind on purpose, so the prefix alone
        # would hand us a stale constraint force. This forward is what makes it
        # current. Removing it silently changes the torque, so keep it even when
        # the surrounding state looks fresh. See
        # docs/reference/mujoco-warp.md, "Stateful control state-prefix
        # schedule".
        self._run_mjw_forward()
        self._write_target_ctrl(actions)
        regularization = self.settings.regularization
        prepare_outputs = [
            self.p_diff_wp,
            self.v_diff_wp,
            self.kv_diag_wp,
            self.rhs_wp,
        ]
        common_prepare_inputs = [
            self.target_ctrl_wp,
            self.mjw_data.actuator_length,
            self.mjw_data.actuator_velocity,
            self.kp_wp,
            self.kv_wp,
            self.dof_to_actuator_wp,
        ]
        if self.mass_matrix_is_packed:
            wp.launch(
                _prepare_spd_m_packed,
                dim=(self.num_envs, int(self.model.nv)),
                inputs=[
                    *common_prepare_inputs,
                    self.mjw_model.M_rownnz,
                    self.mjw_model.M_rowadr,
                    self.mass_matrix_wp,
                    self.mjw_data.qfrc_bias,
                    # Current only because compute_torque ran a forward first.
                    self.mjw_data.qfrc_constraint,
                    self.physics_dt,
                    regularization,
                ],
                outputs=prepare_outputs,
            )
        else:
            wp.launch(
                _prepare_spd_qm_dense,
                dim=(self.num_envs, int(self.model.nv)),
                inputs=[
                    *common_prepare_inputs,
                    self.mass_matrix_wp,
                    self.mjw_data.qfrc_bias,
                    # Current only because compute_torque ran a forward first.
                    self.mjw_data.qfrc_constraint,
                    self.physics_dt,
                    regularization,
                ],
                outputs=prepare_outputs,
            )

        factor_solve_i_no_l_store(
            self.mjw_model,
            self.mjw_data,
            self.mass_matrix_wp,
            self.factor_wp,
            self.diag_inv_wp,
            self.ddq_wp,
            self.rhs_wp,
        )
        common_restore_inputs = [
            self.p_diff_wp,
            self.v_diff_wp,
            self.kv_diag_wp,
            self.ddq_wp,
            self.physics_dt,
            regularization,
            self.settings.torque_scale,
            self.settings.torque_limit,
            self.root_dof_count,
        ]
        if self.mass_matrix_is_packed:
            wp.launch(
                _restore_packed_m_write_spd_qfrc,
                dim=(self.num_envs, int(self.model.nv)),
                inputs=[
                    *common_restore_inputs,
                    self.mjw_model.M_rownnz,
                    self.mjw_model.M_rowadr,
                    self.mass_matrix_wp,
                ],
                outputs=[output_qfrc_wp],
            )
        else:
            wp.launch(
                _restore_dense_qm_write_spd_qfrc,
                dim=(self.num_envs, int(self.model.nv)),
                inputs=[*common_restore_inputs, self.mass_matrix_wp],
                outputs=[output_qfrc_wp],
            )
        return output_qfrc


__all__ = ("StablePDController", "stable_pd_dof_to_actuator")
