import logging
import os
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
import torch
from pymotion.io.bvh import BVH
from pymotion.rotations import quat, quat_torch

from instantmimic.motion import materialize_mujoco_reference_payload
from instantmimic.motion import quaternion as torch_utils
from instantmimic.motion.bvh_mapping import (
    BvhMapValue,
    bvh_map_body_names,
    bvh_map_targets,
)
from instantmimic.motion.cache import (
    acquire_motion_cache_write_lock,
    build_motion_cache_metadata,
    load_motion_cache,
    motion_cache_file_path,
    save_motion_cache,
)
from instantmimic.motion.quaternion import quat_conj_np as _quat_conj_np
from instantmimic.motion.quaternion import quat_mul_np as _quat_mul_np
from instantmimic.motion.quaternion import quat_normalize_np as _quat_normalize_np
from instantmimic.motion.quaternion import quat_rotate_vec_np as _quat_rotate_vec_np
from instantmimic.utils.project_paths import resolve_cache_path, resolve_project_path

_LOGGER = logging.getLogger(__name__)
_H5_QVEL_CACHE_CONTRACT_VERSION = 1
_MOTION_SAMPLING_CACHE_CONTRACT_VERSION = 1
# 1: every source rests unrecorded passive coordinates at qpos0; earlier
# BVH and body-space MimicKit payloads left them at zero.
_PASSIVE_REST_CACHE_CONTRACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class MotionLoadOptions:
    """How to turn a motion file into the tensors an environment replays.

    Seven functions took these as eleven separate keyword arguments and passed
    them straight down. The bundle is what the motion cache is keyed on, so it
    is also the thing that has to stay together: a loader that gained an option
    the key did not know about would serve one clip's tensors for another's
    request.

    `root_auto_calibration` reaches only the HDF5 loader,
    `mimickit_joint_groups` only the MimicKit loaders and `rest_bodies` only
    the BVH loader; each is refused rather than ignored when the chosen format
    cannot honour it.
    """

    motion_scale: float
    motion_offset: list[float]
    bvh_map: dict[str, BvhMapValue]
    target_dt: float
    mimickit_joint_groups: list[dict[str, object]] | None = None
    loop_boundary_vel_from_first_frame: bool = True
    loop_boundary_vel_extrapolate: bool = False
    smooth_cycle: bool = False
    smooth_cycle_blend_frames: int = 30
    smooth_cycle_mode: str = "tail_warp"
    root_auto_calibration: bool = False
    # Model qpos and DOF addresses of the character (root and joints;
    # instantmimic.envs.dof_roles); the remaining coordinates are passive. Every
    # source leaves the passive coordinates it does not record at qpos0 with
    # zero velocity (`_rest_unrecorded_passive_coordinates`), so a model-space
    # HDF5 recording may also hold only these coordinates. None: the model has
    # no passive coordinates and HDF5 recordings must span the whole model.
    character_coordinates: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    # Bodies the reference tracks with their own joints held at qpos0 and zero
    # velocity for the whole clip (`_rest_body_joints`). BVH only.
    rest_bodies: tuple[str, ...] = ()

    def cache_metadata(
        self, *, motion_file: Path, model: mujoco.MjModel
    ) -> dict[str, object]:
        """The cache key for a clip loaded with these options.

        The shared builder identifies the source, model, and authored options.
        Loader contract versions prevent reuse of payloads sampled or measured
        under older semantics.
        """

        metadata = build_motion_cache_metadata(
            motion_file=motion_file,
            motion_scale=self.motion_scale,
            motion_offset=self.motion_offset,
            bvh_map=self.bvh_map,
            target_dt=self.target_dt,
            model=model,
            mimickit_joint_groups=self.mimickit_joint_groups,
            loop_boundary_vel_from_first_frame=(
                self.loop_boundary_vel_from_first_frame
            ),
            loop_boundary_vel_extrapolate=self.loop_boundary_vel_extrapolate,
            smooth_cycle=self.smooth_cycle,
            smooth_cycle_blend_frames=self.smooth_cycle_blend_frames,
            smooth_cycle_mode=self.smooth_cycle_mode,
            root_auto_calibration=self.root_auto_calibration,
        )
        metadata["motion_sampling_contract_version"] = (
            _MOTION_SAMPLING_CACHE_CONTRACT_VERSION
        )
        if self.character_coordinates is not None:
            metadata["passive_rest_contract_version"] = (
                _PASSIVE_REST_CACHE_CONTRACT_VERSION
            )
            metadata["character_coordinates"] = [
                [int(v) for v in self.character_coordinates[0]],
                [int(v) for v in self.character_coordinates[1]],
            ]
        if self.rest_bodies:
            # Only when set, so clips cached before the option keep their key.
            metadata["rest_bodies"] = sorted(self.rest_bodies)
        if motion_file.suffix.lower() in {".h5", ".hdf5"}:
            # HDF5 payloads may now carry measured qvel. Keep old cached
            # finite-difference payloads from satisfying this loader contract.
            metadata["h5_qvel_contract_version"] = _H5_QVEL_CACHE_CONTRACT_VERSION
        return metadata


_SMOOTH_CYCLE_MODES = ("tail_warp", "c1_bridge")

Y_UP_TO_Z_UP = quat.to_matrix(
    quat.mul(
        quat.from_angle_axis(np.array([np.pi / 2]), np.array([0, 0, 1])),
        quat.from_angle_axis(np.array([np.pi / 2]), np.array([1, 0, 0])),
    )
).astype(np.float32)


def _root_qpos_addr(model: mujoco.MjModel) -> int:
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[joint_id])
    return -1


def _root_body_id(model: mujoco.MjModel) -> int:
    if model.njnt <= 0:
        raise ValueError("Motion model has no joints from which to infer a root body.")
    first_joint_id = min(
        range(model.njnt), key=lambda joint_id: int(model.jnt_qposadr[joint_id])
    )
    return int(model.jnt_bodyid[first_joint_id])


def _normalize_smooth_cycle_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _SMOOTH_CYCLE_MODES:
        raise ValueError(
            f"Unsupported smooth_cycle_mode: {mode!r}. "
            f"Expected one of: {', '.join(_SMOOTH_CYCLE_MODES)}."
        )
    return normalized


def _normalize_root_bake_offset(root_bake_offset: list[float] | None) -> list[float]:
    if root_bake_offset is None:
        return [0.0, 0.0, 0.0]
    if len(root_bake_offset) != 3:
        raise ValueError(
            f"root_bake_offset must have 3 values, got {root_bake_offset!r}"
        )
    return [float(v) for v in root_bake_offset]


def _geom_half_extent_z(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> float:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return float(size[0])

    xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    if geom_type in {
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }:
        return float(abs(xmat[2, 2]) * size[1] + size[0])
    if geom_type in {
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    }:
        return float(np.sum(np.abs(xmat[2, :]) * size[:3]))
    return float(model.geom_rbound[geom_id])


def _lowest_character_geom_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    candidates: list[float] = []
    fallback_candidates: list[float] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) == 0:
            continue
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue

        low_z = float(data.geom_xpos[geom_id, 2]) - _geom_half_extent_z(
            model, data, geom_id
        )
        fallback_candidates.append(low_z)
        if (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
        ):
            candidates.append(low_z)

    if candidates:
        return min(candidates)
    if fallback_candidates:
        return min(fallback_candidates)
    return float(np.min(data.xpos[1:, 2])) if model.nbody > 1 else 0.0


def _compute_root_height_calibration_z_offset(
    *,
    qpos_calibration: torch.Tensor,
    model: mujoco.MjModel,
    root_qpos_adr: int,
) -> float:
    data = mujoco.MjData(model)
    data.qpos[:] = (
        qpos_calibration.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    lowest_z = _lowest_character_geom_z(model, data)
    z_offset = -lowest_z
    root_z = float(qpos_calibration[root_qpos_adr + 2].item())
    _LOGGER.info(
        "Auto height calibration: calibration_root_z=%.6f lowest_geom_z=%.6f "
        "z_bake=%.6f",
        root_z,
        lowest_z,
        z_offset,
    )
    return z_offset


def _effective_root_bake_offset(
    *,
    qpos_calibration: torch.Tensor,
    model: mujoco.MjModel,
    root_qpos_adr: int,
    root_bake_offset: list[float] | None,
    root_auto_calibration: bool,
) -> list[float]:
    offset = _normalize_root_bake_offset(root_bake_offset)
    if not root_auto_calibration:
        return offset

    calibration_qpos = qpos_calibration.clone()
    calibration_qpos[root_qpos_adr + 0] += offset[0]
    calibration_qpos[root_qpos_adr + 1] += offset[1]
    offset[2] = _compute_root_height_calibration_z_offset(
        qpos_calibration=calibration_qpos,
        model=model,
        root_qpos_adr=root_qpos_adr,
    )
    return offset


SMPL_MIMICKIT_DOF_BODY_ORDER = (
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
)


def _cardinal_axis_label_and_sign(axis: list[float]) -> tuple[str, float] | None:
    axis_abs = [abs(v) for v in axis]
    axis_idx = max(range(3), key=axis_abs.__getitem__)
    if axis_abs[axis_idx] < 1.0 - 1.0e-4:
        return None
    if any(abs(axis[i]) > 1.0e-4 for i in range(3) if i != axis_idx):
        return None
    return "xyz"[axis_idx], 1.0 if axis[axis_idx] >= 0.0 else -1.0


def build_smpl_mimickit_motion_joint_groups(
    model: mujoco.MjModel,
) -> list[dict[str, object]]:
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    ball_type = int(mujoco.mjtJoint.mjJNT_BALL)
    joint_groups: list[dict[str, object]] = []

    for source_dof_adr, body_name in enumerate(SMPL_MIMICKIT_DOF_BODY_ORDER):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(
                f"Missing SMPL body in model for motion mapping: {body_name}"
            )

        joint_count = int(model.body_jntnum[body_id])
        joint_adr = int(model.body_jntadr[body_id])
        source_offset = source_dof_adr * 3

        if joint_count == 1:
            joint_id = joint_adr
            joint_type = int(model.jnt_type[joint_id])
            qpos_adr = int(model.jnt_qposadr[joint_id])

            if joint_type == ball_type:
                dof_adr = int(model.jnt_dofadr[joint_id])
                joint_groups.append(
                    {
                        "body_name": body_name,
                        "kind": 3,
                        "qpos_adrs": [qpos_adr],
                        "dof_adrs": [dof_adr, dof_adr + 1, dof_adr + 2],
                        "dof_adr_local": source_offset,
                        "dof_dim": 3,
                    }
                )
                continue

            if joint_type == hinge_type:
                dof_adr = int(model.jnt_dofadr[joint_id])
                limited = int(model.jnt_limited[joint_id]) == 1
                raw_lower = float(model.jnt_range[joint_id, 0]) if limited else None
                raw_upper = float(model.jnt_range[joint_id, 1]) if limited else None
                joint_groups.append(
                    {
                        "body_name": body_name,
                        "kind": 4,
                        "qpos_adrs": [qpos_adr],
                        "dof_adrs": [dof_adr],
                        "dof_adr_local": source_offset,
                        "dof_dim": 3,
                        "axis": model.jnt_axis[joint_id].tolist(),
                        "raw_lower": raw_lower,
                        "raw_upper": raw_upper,
                    }
                )
                continue

        if joint_count == 3:
            joint_ids = [joint_adr + i for i in range(3)]
            if all(int(model.jnt_type[jid]) == hinge_type for jid in joint_ids):
                order_labels: list[str] = []
                axis_signs: list[float] = []
                raw_lower: list[float] = []
                raw_upper: list[float] = []
                qpos_adrs: list[int] = []
                dof_adrs: list[int] = []
                axis_ok = True
                for joint_id in joint_ids:
                    axis_info = _cardinal_axis_label_and_sign(
                        model.jnt_axis[joint_id].tolist()
                    )
                    if axis_info is None:
                        axis_ok = False
                        break
                    axis_label, axis_sign = axis_info
                    order_labels.append(axis_label)
                    axis_signs.append(axis_sign)
                    qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
                    dof_adrs.append(int(model.jnt_dofadr[joint_id]))
                    if int(model.jnt_limited[joint_id]) == 1:
                        raw_lower.append(float(model.jnt_range[joint_id, 0]))
                        raw_upper.append(float(model.jnt_range[joint_id, 1]))
                    else:
                        raw_lower.append(float("-inf"))
                        raw_upper.append(float("inf"))
                if axis_ok and len(set(order_labels)) == 3:
                    joint_groups.append(
                        {
                            "body_name": body_name,
                            "kind": 2,
                            "qpos_adrs": qpos_adrs,
                            "dof_adrs": dof_adrs,
                            "dof_adr_local": source_offset,
                            "dof_dim": 3,
                            "order_labels": order_labels,
                            "axis_signs": axis_signs,
                            "raw_lower": raw_lower,
                            "raw_upper": raw_upper,
                        }
                    )
                    continue

        raise ValueError(
            "Unsupported joint layout for MimicKit SMPL motion mapping: "
            f"{body_name} has {joint_count} joint(s)."
        )

    return joint_groups


def _infer_mimickit_joint_groups_from_model(
    model: mujoco.MjModel,
) -> list[dict[str, object]]:
    """Infer a minimal MimicKit qpos mapping from the MuJoCo model layout.

    This is used when a ball-joint model is loaded before the env-specific
    MimicKit joint groups are constructed. It only needs enough information to
    map MimicKit DOFs into MuJoCo qpos.
    """
    joint_groups: list[dict[str, object]] = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
            continue

        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr_local = int(model.jnt_dofadr[joint_id]) - 6
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
            joint_groups.append(
                {
                    "body_name": body_name,
                    "kind": 3,
                    "qpos_adrs": [qpos_adr],
                    "dof_adr_local": dof_adr_local,
                    "dof_dim": 3,
                }
            )
        elif joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            joint_groups.append(
                {
                    "body_name": body_name,
                    "kind": 1,
                    "qpos_adrs": [qpos_adr],
                    "dof_adr_local": dof_adr_local,
                    "dof_dim": 1,
                }
            )

    return joint_groups


def _select_continuous_tait_bryan_sequence(
    euler_wrapped: torch.Tensor,
    raw_lower: np.ndarray | None = None,
    raw_upper: np.ndarray | None = None,
) -> torch.Tensor:
    """Pick a per-frame nearest equivalent Euler branch (fast greedy)."""
    if euler_wrapped.ndim != 2 or euler_wrapped.shape[1] != 3:
        raise ValueError(
            f"Expected euler_wrapped shape [F,3], got {tuple(euler_wrapped.shape)}"
        )

    e_np = euler_wrapped.detach().cpu().numpy().astype(np.float64, copy=True)
    frame_count = int(e_np.shape[0])
    if frame_count == 0:
        return euler_wrapped

    two_pi = 2.0 * np.pi

    # Primary branch in [-pi, pi].
    p = (e_np + np.pi) % two_pi - np.pi
    # Secondary equivalent branch for intrinsic Tait-Bryan:
    # [a, b, c] <-> [a+pi, pi-b, c+pi]
    a = np.empty_like(p)
    a[:, 0] = p[:, 0] + np.pi
    a[:, 1] = np.pi - p[:, 1]
    a[:, 2] = p[:, 2] + np.pi
    a = (a + np.pi) % two_pi - np.pi

    if raw_lower is None:
        raw_lower = np.full(3, -np.inf, dtype=np.float64)
    if raw_upper is None:
        raw_upper = np.full(3, np.inf, dtype=np.float64)

    out = np.zeros_like(p)

    def _limit_penalty(v: np.ndarray) -> float:
        below = np.maximum(raw_lower - v, 0.0)
        above = np.maximum(v - raw_upper, 0.0)
        return float(np.sum((below + above) ** 2))

    # frame 0: choose lower-penalty representation around zero.
    cands0 = [p[0], a[0]]
    best0 = cands0[0]
    best0_cost = np.inf
    for c0 in cands0:
        c = c0 + two_pi * np.round((-c0) / two_pi)
        cost = 50.0 * _limit_penalty(c) + 0.01 * float(np.dot(c, c))
        if cost < best0_cost:
            best0_cost = cost
            best0 = c
    out[0] = best0

    # Greedy nearest-to-previous selection:
    # for each branch, add 2pi*k (k from nearest integer per-axis) to be closest to prev.
    for fi in range(1, frame_count):
        prev = out[fi - 1]
        best = p[fi]
        best_cost = np.inf
        for base in (p[fi], a[fi]):
            cand = base + two_pi * np.round((prev - base) / two_pi)
            c_cont = float(np.sum((cand - prev) ** 2))
            c_lim = _limit_penalty(cand)
            c_mag = float(np.dot(cand, cand))
            cost = c_cont + 20.0 * c_lim + 0.001 * c_mag
            if cost < best_cost:
                best_cost = cost
                best = cand
        out[fi] = best

    return torch.as_tensor(out, device=euler_wrapped.device, dtype=euler_wrapped.dtype)


def _select_continuous_angle_sequence(
    angle_wrapped: torch.Tensor,
    raw_lower: float | None = None,
    raw_upper: float | None = None,
) -> torch.Tensor:
    """Pick a per-frame nearest equivalent 1-DOF angle branch."""
    if angle_wrapped.ndim != 1:
        raise ValueError(
            f"Expected angle_wrapped shape [F], got {tuple(angle_wrapped.shape)}"
        )

    a_np = angle_wrapped.detach().cpu().numpy().astype(np.float64, copy=True)
    frame_count = int(a_np.shape[0])
    if frame_count == 0:
        return angle_wrapped

    two_pi = 2.0 * np.pi
    lower = -np.inf if raw_lower is None else float(raw_lower)
    upper = np.inf if raw_upper is None else float(raw_upper)
    out = np.zeros_like(a_np)

    def _limit_penalty(v: float) -> float:
        below = max(lower - v, 0.0)
        above = max(v - upper, 0.0)
        return float((below + above) ** 2)

    def _candidate_set(base: float, center: float | None = None) -> list[float]:
        if center is None:
            k0 = int(np.round((-base) / two_pi))
        else:
            k0 = int(np.round((center - base) / two_pi))
        return [base + two_pi * float(k0 + dk) for dk in (-1, 0, 1)]

    best0 = a_np[0]
    best0_cost = np.inf
    for cand in _candidate_set(float(a_np[0])):
        cost = 50.0 * _limit_penalty(cand) + 0.01 * float(cand * cand)
        if cost < best0_cost:
            best0_cost = cost
            best0 = cand
    out[0] = best0

    for fi in range(1, frame_count):
        prev = float(out[fi - 1])
        best = float(a_np[fi])
        best_cost = np.inf
        for cand in _candidate_set(float(a_np[fi]), prev):
            c_cont = float((cand - prev) ** 2)
            c_lim = _limit_penalty(cand)
            c_mag = float(cand * cand)
            cost = c_cont + 20.0 * c_lim + 0.001 * c_mag
            if cost < best_cost:
                best_cost = cost
                best = cand
        out[fi] = best

    return torch.as_tensor(out, device=angle_wrapped.device, dtype=angle_wrapped.dtype)


def _apply_mimickit_joint_groups_to_qpos(
    qpos: torch.Tensor,
    dof_pos: torch.Tensor,
    mimickit_joint_groups: list[dict[str, object]],
) -> np.ndarray:
    """Write the grouped MimicKit DOFs into ``qpos``; return the written columns."""
    frame_count = int(qpos.shape[0])
    written = np.zeros(int(qpos.shape[1]), dtype=bool)

    for group in mimickit_joint_groups:
        kind = int(group.get("kind", -1))
        dof_adr = int(group.get("dof_adr_local", -1))
        dof_dim = int(group.get("dof_dim", 0))
        qpos_adrs = list(group.get("qpos_adrs", []))

        if dof_adr < 0 or dof_dim <= 0:
            continue

        if kind == 1:
            qpos[:, int(qpos_adrs[0])] = dof_pos[:, dof_adr]
            written[int(qpos_adrs[0])] = True
            continue

        if kind == 3:
            qadr = int(qpos_adrs[0])
            qpos[:, qadr : qadr + 4] = torch_utils.scaled_axis_angle_to_quat(
                dof_pos[:, dof_adr : dof_adr + 3]
            )
            written[qadr : qadr + 4] = True
            continue

        if kind == 4:
            qadr = int(qpos_adrs[0])
            local_rotation = torch_utils.scaled_axis_angle_to_quat(
                dof_pos[:, dof_adr : dof_adr + 3]
            )
            hinge_angle = _extract_hinge_angle_from_local_rotation(
                local_rotation,
                np.asarray(group["axis"], dtype=np.float64),
            )
            raw_lower = group.get("raw_lower", None)
            raw_upper = group.get("raw_upper", None)
            qpos[:, qadr] = _select_continuous_angle_sequence(
                hinge_angle,
                None if raw_lower is None else float(raw_lower),
                None if raw_upper is None else float(raw_upper),
            )
            written[qadr] = True
            continue

        if kind != 2:
            continue

        expmap = dof_pos[:, dof_adr : dof_adr + 3]
        quat = torch_utils.scaled_axis_angle_to_quat(expmap)

        order_labels = np.asarray(group["order_labels"], dtype="<U1")
        order_np = np.broadcast_to(order_labels, (frame_count, 3))
        hinge_euler = quat_torch.to_euler(quat, order_np)
        hinge_euler = (hinge_euler + np.pi) % (2.0 * np.pi) - np.pi

        raw_lower = np.full(3, -np.inf, dtype=np.float64)
        raw_upper = np.full(3, np.inf, dtype=np.float64)
        axis_signs = np.asarray(group["axis_signs"], dtype=np.float64)
        group_lower = np.asarray(group.get("raw_lower", raw_lower), dtype=np.float64)
        group_upper = np.asarray(group.get("raw_upper", raw_upper), dtype=np.float64)
        for k in range(3):
            if not np.isfinite(group_lower[k]) or not np.isfinite(group_upper[k]):
                continue
            if axis_signs[k] < 0.0:
                raw_lower[k] = -group_upper[k]
                raw_upper[k] = -group_lower[k]
            else:
                raw_lower[k] = group_lower[k]
                raw_upper[k] = group_upper[k]

        hinge_euler = _select_continuous_tait_bryan_sequence(
            hinge_euler, raw_lower, raw_upper
        )

        for k, qadr in enumerate(qpos_adrs):
            val = hinge_euler[:, k]
            if axis_signs[k] < 0.0:
                val = -val
            qpos[:, int(qadr)] = val
            written[int(qadr)] = True

    return written


def _decompose_hinge_chain(
    local_rotation: torch.Tensor,
    axes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    iterations: int = 30,
) -> torch.Tensor:
    """Least-squares angles of a body's hinge sequence for a target rotation.

    MuJoCo composes a body's joints in order with axes in the body frame, so the
    body rotation is ``Rot(a_0, q_0) Rot(a_1, q_1) ...``. Each frame minimises the
    geodesic angle to ``local_rotation`` by Gauss-Newton with an active set on
    the joint ranges, starting from the per-axis swing-twist angles. A rotation
    the chain cannot represent leaves the least-squares residual.
    """
    q_target = local_rotation.to(torch.float64)
    q_target = q_target / torch.linalg.norm(q_target, dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )
    axes_t = torch.as_tensor(axes, dtype=torch.float64, device=q_target.device)
    axes_t = axes_t / torch.linalg.norm(axes_t, dim=-1, keepdim=True)
    lo = torch.as_tensor(lower, dtype=torch.float64, device=q_target.device)
    hi = torch.as_tensor(upper, dtype=torch.float64, device=q_target.device)
    angles = torch.stack(
        [_extract_hinge_angle_from_local_rotation(q_target, a) for a in axes], dim=1
    )
    angles = torch.minimum(torch.maximum(angles, lo), hi)
    count = int(axes_t.shape[0])
    eye = torch.eye(count, dtype=torch.float64, device=q_target.device)
    for _ in range(iterations):
        half = 0.5 * angles
        factors = torch.cat(
            (torch.cos(half)[..., None], torch.sin(half)[..., None] * axes_t), dim=-1
        )
        # suffix[k] = Rot(a_k+1) ... Rot(a_K-1): maps axis k into the body frame.
        jac = []
        suffix = torch.zeros_like(factors[:, 0])
        suffix[:, 0] = 1.0
        for k in reversed(range(count)):
            jac.append(
                torch_utils.quat_rotate_inverse_unit(
                    suffix, axes_t[k].expand_as(suffix[:, 1:])
                )
            )
            suffix = torch_utils.quat_mul(factors[:, k], suffix)
        error = torch_utils.quat_to_scaled_axis_angle(
            torch_utils.quat_mul(torch_utils.quat_conj(q_target), suffix)
        )
        # d log(R_t^T R exp(w)) / dw = Jr^-1(error): exact first order at a
        # non-zero residual, so the iteration converges to the true minimum.
        theta = torch.linalg.norm(error, dim=-1)[..., None, None]
        skew = torch.zeros(
            error.shape[:-1] + (3, 3), dtype=torch.float64, device=error.device
        )
        skew[..., 0, 1], skew[..., 0, 2] = -error[..., 2], error[..., 1]
        skew[..., 1, 0], skew[..., 1, 2] = error[..., 2], -error[..., 0]
        skew[..., 2, 0], skew[..., 2, 1] = -error[..., 1], error[..., 0]
        safe = theta.clamp_min(1.0e-6)
        coeff = torch.where(
            theta < 1.0e-6,
            torch.full_like(theta, 1.0 / 12.0),
            1.0 / safe**2 - (1.0 + torch.cos(safe)) / (2.0 * safe * torch.sin(safe)),
        )
        jr_inv = (
            torch.eye(3, dtype=torch.float64, device=error.device)
            + 0.5 * skew
            + coeff * (skew @ skew)
        )
        jac = jr_inv @ torch.stack(jac[::-1], dim=-1)  # (F, 3, K)
        gradient = jac.transpose(1, 2) @ error[..., None]
        # Active set: a joint at a limit whose descent direction leaves the
        # range is held there, and the others are solved for without it.
        held = ((angles <= lo) & (gradient[..., 0] > 0.0)) | (
            (angles >= hi) & (gradient[..., 0] < 0.0)
        )
        free = (~held).to(torch.float64)
        jac_free = jac * free[:, None, :]
        normal = jac_free.transpose(1, 2) @ jac_free + 1.0e-6 * eye
        step = torch.linalg.solve(
            normal, -(jac_free.transpose(1, 2) @ error[..., None])
        )
        angles = torch.minimum(torch.maximum(angles + step[..., 0] * free, lo), hi)
    return angles.to(local_rotation.dtype)


def _extract_hinge_angle_from_local_rotation(
    local_rotation: torch.Tensor, joint_axis: np.ndarray
) -> torch.Tensor:
    """Extract signed hinge angle using a swing-twist projection onto joint axis."""
    if local_rotation.ndim != 2 or local_rotation.shape[1] != 4:
        raise ValueError(
            f"Expected local_rotation shape [F,4], got {tuple(local_rotation.shape)}"
        )

    axis_np = np.asarray(joint_axis, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis_np))
    if axis_norm < 1.0e-8:
        return torch.zeros(
            (local_rotation.shape[0],),
            device=local_rotation.device,
            dtype=local_rotation.dtype,
        )
    axis_np = axis_np / axis_norm

    axis = torch.as_tensor(
        axis_np, device=local_rotation.device, dtype=local_rotation.dtype
    )
    q = local_rotation / torch.linalg.norm(
        local_rotation, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    qv = q[:, 1:4]
    proj_mag = torch.sum(qv * axis[None, :], dim=-1, keepdim=True)
    twist_vec = proj_mag * axis[None, :]
    twist = torch.cat((q[:, 0:1], twist_vec), dim=-1)
    twist = twist / torch.linalg.norm(twist, dim=-1, keepdim=True).clamp_min(1.0e-8)

    twist_w = twist[:, 0]
    twist_s = torch.sum(twist[:, 1:4] * axis[None, :], dim=-1)
    angle = 2.0 * torch.atan2(twist_s, twist_w)
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def _heading_yaw_quat_from_quat_np(q: np.ndarray) -> np.ndarray:
    forward = _quat_rotate_vec_np(_quat_normalize_np(q), np.array([1.0, 0.0, 0.0]))
    yaw = float(np.arctan2(forward[1], forward[0]))
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def _compute_loop_root_transform_np(
    qpos_np: np.ndarray, root_qpos_adr: int, vertical_axis: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """One-cycle root transform (heading-only rotation + translation)."""
    if root_qpos_adr < 0 or qpos_np.shape[0] == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), np.zeros(
            3, dtype=np.float64
        )

    frame_count = int(qpos_np.shape[0])
    last_i = frame_count - 1
    prev_i = max(0, frame_count - 2)

    p_first = qpos_np[0, root_qpos_adr : root_qpos_adr + 3]
    p_last = qpos_np[last_i, root_qpos_adr : root_qpos_adr + 3]
    p_prev = qpos_np[prev_i, root_qpos_adr : root_qpos_adr + 3]

    q_first = _quat_normalize_np(
        qpos_np[0, root_qpos_adr + 3 : root_qpos_adr + 7].astype(np.float64, copy=True)
    )
    q_last = _quat_normalize_np(
        qpos_np[last_i, root_qpos_adr + 3 : root_qpos_adr + 7].astype(
            np.float64, copy=True
        )
    )
    q_prev = _quat_normalize_np(
        qpos_np[prev_i, root_qpos_adr + 3 : root_qpos_adr + 7].astype(
            np.float64, copy=True
        )
    )

    q_delta = _quat_mul_np(q_last, _quat_conj_np(q_prev))
    q_ext = _quat_mul_np(q_delta, q_last)
    q_rel = _quat_mul_np(q_ext, _quat_conj_np(q_first))
    loop_root_quat = _heading_yaw_quat_from_quat_np(q_rel)
    loop_root_quat = _quat_normalize_np(loop_root_quat)

    p_ext = p_last + (p_last - p_prev)
    loop_root_pos = p_ext - _quat_rotate_vec_np(loop_root_quat, p_first)
    loop_root_pos = loop_root_pos.astype(np.float64, copy=True)
    loop_root_pos[vertical_axis] = 0.0

    return loop_root_quat, loop_root_pos


def _compute_loop_root_transform_from_body_pose_np(
    root_pos_np: np.ndarray,
    root_quat_np: np.ndarray,
    vertical_axis: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    if root_pos_np.shape[0] == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), np.zeros(
            3, dtype=np.float64
        )

    frame_count = int(root_pos_np.shape[0])
    last_i = frame_count - 1
    prev_i = max(0, frame_count - 2)
    p_first = np.asarray(root_pos_np[0], dtype=np.float64)
    p_last = np.asarray(root_pos_np[last_i], dtype=np.float64)
    p_prev = np.asarray(root_pos_np[prev_i], dtype=np.float64)
    q_first = _quat_normalize_np(np.asarray(root_quat_np[0], dtype=np.float64))
    q_last = _quat_normalize_np(np.asarray(root_quat_np[last_i], dtype=np.float64))
    q_prev = _quat_normalize_np(np.asarray(root_quat_np[prev_i], dtype=np.float64))

    q_delta = _quat_mul_np(q_last, _quat_conj_np(q_prev))
    q_ext = _quat_mul_np(q_delta, q_last)
    q_rel = _quat_mul_np(q_ext, _quat_conj_np(q_first))
    loop_root_quat = _quat_normalize_np(_heading_yaw_quat_from_quat_np(q_rel))
    p_ext = p_last + (p_last - p_prev)
    loop_root_pos = p_ext - _quat_rotate_vec_np(loop_root_quat, p_first)
    loop_root_pos = loop_root_pos.astype(np.float64, copy=True)
    loop_root_pos[vertical_axis] = 0.0
    return loop_root_quat, loop_root_pos


def _materialize_root_body_pose_np(
    model: mujoco.MjModel,
    qpos: torch.Tensor,
    root_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    qpos_np = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    root_pos = np.empty((qpos_np.shape[0], 3), dtype=np.float64)
    root_quat = np.empty((qpos_np.shape[0], 4), dtype=np.float64)
    data = mujoco.MjData(model)
    for frame_idx, qpos_row in enumerate(qpos_np):
        data.qpos[:] = qpos_row
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        root_pos[frame_idx] = data.xpos[root_body_id]
        root_quat[frame_idx] = data.xquat[root_body_id]
    return root_pos, root_quat


def _apply_root_transform_to_qpos_row_np(
    qpos_row: np.ndarray,
    root_qpos_adr: int,
    loop_root_quat: np.ndarray,
    loop_root_pos: np.ndarray,
) -> np.ndarray:
    out = qpos_row.copy()
    if root_qpos_adr < 0:
        return out
    p = out[root_qpos_adr : root_qpos_adr + 3]
    q = _quat_normalize_np(out[root_qpos_adr + 3 : root_qpos_adr + 7])
    out[root_qpos_adr : root_qpos_adr + 3] = _quat_rotate_vec_np(loop_root_quat, p) + (
        loop_root_pos
    )
    out[root_qpos_adr + 3 : root_qpos_adr + 7] = _quat_normalize_np(
        _quat_mul_np(loop_root_quat, q)
    )
    return out


def _build_loop_next_qpos(
    qpos_np: np.ndarray,
    root_qpos_adr: int,
    loop_root_quat: np.ndarray,
    loop_root_pos: np.ndarray,
) -> np.ndarray:
    """Build cycle-next pose from first frame using one-cycle root transform."""
    next_qpos = qpos_np[0].copy()
    if root_qpos_adr >= 0:
        next_qpos = _apply_root_transform_to_qpos_row_np(
            next_qpos, root_qpos_adr, loop_root_quat, loop_root_pos
        )
    return next_qpos


def _smooth_cycle_qpos(
    qpos: torch.Tensor,
    model: mujoco.MjModel,
    root_qpos_adr: int,
    blend_frames: int,
    loop_root_quat: np.ndarray,
    loop_root_pos: np.ndarray,
) -> torch.Tensor:
    """Warp the tail frames toward the loop seam target."""
    frame_count = int(qpos.shape[0])
    if frame_count < 2:
        return qpos

    bn = max(0, min(int(blend_frames), frame_count))
    if bn == 0:
        return qpos

    qpos_np = qpos.detach().cpu().numpy().astype(np.float64, copy=True)
    next_qpos = _build_loop_next_qpos(
        qpos_np=qpos_np,
        root_qpos_adr=root_qpos_adr,
        loop_root_quat=loop_root_quat,
        loop_root_pos=loop_root_pos,
    )
    seam_delta = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, seam_delta, 1.0, qpos_np[-1], next_qpos)

    for i in range(bn):
        fi = frame_count - bn + i
        t = float(i) / float(max(bn - 1, 1))
        s = 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))
        if s <= 0.0:
            continue

        q_frame = qpos_np[fi].copy()
        mujoco.mj_integratePos(model, q_frame, seam_delta * s, 1.0)
        qpos_np[fi] = q_frame

    return torch.as_tensor(qpos_np, device=qpos.device, dtype=qpos.dtype)


def _smooth_cycle_qpos_c1_bridge(
    qpos: torch.Tensor,
    model: mujoco.MjModel,
    root_qpos_adr: int,
    blend_frames: int,
    loop_root_quat: np.ndarray,
    loop_root_pos: np.ndarray,
) -> torch.Tensor:
    """Preserve C0 tail smoothing while nudging the seam derivative to C1."""
    frame_count = int(qpos.shape[0])
    if frame_count < 3:
        return _smooth_cycle_qpos(
            qpos=qpos,
            model=model,
            root_qpos_adr=root_qpos_adr,
            blend_frames=blend_frames,
            loop_root_quat=loop_root_quat,
            loop_root_pos=loop_root_pos,
        )

    bn = max(0, min(int(blend_frames), frame_count))
    if bn < 3:
        return _smooth_cycle_qpos(
            qpos=qpos,
            model=model,
            root_qpos_adr=root_qpos_adr,
            blend_frames=blend_frames,
            loop_root_quat=loop_root_quat,
            loop_root_pos=loop_root_pos,
        )

    qpos = _smooth_cycle_qpos(
        qpos=qpos,
        model=model,
        root_qpos_adr=root_qpos_adr,
        blend_frames=blend_frames,
        loop_root_quat=loop_root_quat,
        loop_root_pos=loop_root_pos,
    )
    qpos_np = qpos.detach().cpu().numpy().astype(np.float64, copy=True)
    start_i = frame_count - bn
    interval_count = bn - 1

    q_end = qpos_np[-1].copy()
    q_end_next = qpos_np[1].copy()
    if root_qpos_adr >= 0:
        q_end_next = _apply_root_transform_to_qpos_row_np(
            q_end_next,
            root_qpos_adr=root_qpos_adr,
            loop_root_quat=loop_root_quat,
            loop_root_pos=loop_root_pos,
        )

    current_end_step = np.zeros(model.nv, dtype=np.float64)
    end_step = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, current_end_step, 1.0, qpos_np[-2], qpos_np[-1])
    mujoco.mj_differentiatePos(model, end_step, 1.0, q_end, q_end_next)
    step_correction = end_step - current_end_step

    for i in range(1, bn - 1):
        fi = start_i + i
        u = float(i) / float(max(interval_count - 1, 1))
        u2 = u * u
        ease = 3.0 * u2 - 2.0 * u2 * u
        remaining_steps = np.sqrt(float(interval_count - i))
        tangent = -step_correction * remaining_steps * ease
        q_frame = qpos_np[fi].copy()
        mujoco.mj_integratePos(model, q_frame, tangent, 1.0)
        qpos_np[fi] = q_frame

    return torch.as_tensor(qpos_np, device=qpos.device, dtype=qpos.dtype)


def _recompute_qvel_from_qpos(
    qpos: torch.Tensor,
    model: mujoco.MjModel,
    control_dt: float,
    root_qpos_adr: int,
    loop_root_quat: np.ndarray,
    loop_root_pos: np.ndarray,
    loop_boundary_vel_from_first_frame: bool,
    loop_boundary_vel_extrapolate: bool,
) -> torch.Tensor:
    """Recompute generalized velocities from qpos using MuJoCo's canonical diff."""
    frame_count = int(qpos.shape[0])
    qvel_np = np.zeros((frame_count, model.nv), dtype=np.float64)
    if frame_count < 2:
        return torch.as_tensor(qvel_np, device=qpos.device, dtype=qpos.dtype)

    qpos_np = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    tmp = np.zeros(model.nv, dtype=np.float64)

    for i in range(frame_count - 1):
        mujoco.mj_differentiatePos(model, tmp, control_dt, qpos_np[i], qpos_np[i + 1])
        qvel_np[i] = tmp

    if loop_boundary_vel_from_first_frame:
        next_qpos = _build_loop_next_qpos(
            qpos_np=qpos_np,
            root_qpos_adr=root_qpos_adr,
            loop_root_quat=loop_root_quat,
            loop_root_pos=loop_root_pos,
        )
        mujoco.mj_differentiatePos(model, tmp, control_dt, qpos_np[-1], next_qpos)
        qvel_np[-1] = tmp
    elif loop_boundary_vel_extrapolate:
        qvel_np[-1] = qvel_np[-2]
    # Else: keep the last-frame velocity as zeros (boundary invalid by design).

    return torch.as_tensor(qvel_np, device=qpos.device, dtype=qpos.dtype)


def _motion_sample_times(
    frame_count: int, source_dt: float, target_dt: float
) -> np.ndarray:
    """Use the authored target step, holding the endpoint past the source duration."""
    target_frame_count = _resampled_frame_count(
        frame_count=frame_count, source_dt=source_dt, target_dt=target_dt
    )
    sample_dt = source_dt if np.isclose(source_dt, target_dt) else target_dt
    duration = float(source_dt) * float(max(0, frame_count - 1))
    return np.minimum(
        np.arange(target_frame_count, dtype=np.float64) * float(sample_dt), duration
    )


def _resample_motion_frames(
    local_positions: torch.Tensor,
    local_rotations: torch.Tensor,
    source_dt: float,
    target_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample motion frames to the target dt using lerp (pos) + slerp (quat)."""
    if np.isclose(source_dt, target_dt):
        return local_positions, local_rotations

    frame_count = int(local_positions.shape[0])
    if frame_count < 2:
        return local_positions, local_rotations

    sample_times_np = _motion_sample_times(frame_count, source_dt, target_dt)
    sample_times = torch.as_tensor(
        sample_times_np, device=local_positions.device, dtype=local_positions.dtype
    )

    src_idx_float = sample_times / float(source_dt)
    idx0 = torch.floor(src_idx_float).to(torch.long)
    idx0 = torch.clamp(idx0, 0, frame_count - 2)
    idx1 = idx0 + 1
    alpha = (src_idx_float - idx0.to(src_idx_float.dtype)).clamp(0.0, 1.0)

    pos0 = local_positions[idx0]
    pos1 = local_positions[idx1]
    interp_positions = torch.lerp(pos0, pos1, alpha[:, None, None])

    rot0 = local_rotations[idx0]
    rot1 = local_rotations[idx1]
    interp_rotations = quat_torch.slerp(rot0, rot1, alpha[:, None, None], shortest=True)
    interp_rotations = interp_rotations / torch.linalg.norm(
        interp_rotations, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)

    # Keep exact boundary samples to avoid tiny numeric drift at loop seams.
    interp_positions[0] = local_positions[0]
    interp_rotations[0] = local_rotations[0]
    if np.isclose(sample_times_np[-1], source_dt * (frame_count - 1)):
        interp_positions[-1] = local_positions[-1]
        interp_rotations[-1] = local_rotations[-1]

    return interp_positions, interp_rotations


def _resample_mimickit_frames(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    dof_pos: torch.Tensor,
    source_dt: float,
    target_dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample MimicKit frame tensors to target dt."""
    if np.isclose(source_dt, target_dt):
        return root_pos, root_quat, dof_pos

    frame_count = int(root_pos.shape[0])
    if frame_count < 2:
        return root_pos, root_quat, dof_pos

    sample_times_np = _motion_sample_times(frame_count, source_dt, target_dt)
    sample_times = torch.as_tensor(
        sample_times_np,
        device=root_pos.device,
        dtype=root_pos.dtype,
    )

    src_idx_float = sample_times / float(source_dt)
    idx0 = torch.floor(src_idx_float).to(torch.long)
    idx0 = torch.clamp(idx0, 0, frame_count - 2)
    idx1 = idx0 + 1
    alpha = (src_idx_float - idx0.to(src_idx_float.dtype)).clamp(0.0, 1.0)

    pos0 = root_pos[idx0]
    pos1 = root_pos[idx1]
    out_root_pos = torch.lerp(pos0, pos1, alpha[:, None])

    quat0 = root_quat[idx0]
    quat1 = root_quat[idx1]
    out_root_quat = quat_torch.slerp(quat0, quat1, alpha[:, None], shortest=True)
    out_root_quat = out_root_quat / torch.linalg.norm(
        out_root_quat, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)

    dof0 = dof_pos[idx0]
    dof1 = dof_pos[idx1]
    out_dof_pos = torch.lerp(dof0, dof1, alpha[:, None])

    out_root_pos[0] = root_pos[0]
    out_root_quat[0] = root_quat[0]
    out_dof_pos[0] = dof_pos[0]
    if np.isclose(sample_times_np[-1], source_dt * (frame_count - 1)):
        out_root_pos[-1] = root_pos[-1]
        out_root_quat[-1] = root_quat[-1]
        out_dof_pos[-1] = dof_pos[-1]

    return out_root_pos, out_root_quat, out_dof_pos


def _resample_qpos_frames(
    qpos: torch.Tensor,
    model: mujoco.MjModel,
    source_dt: float,
    target_dt: float,
) -> torch.Tensor:
    """Resample MuJoCo qpos frames on a uniform target time grid."""
    if np.isclose(source_dt, target_dt):
        return qpos

    frame_count = int(qpos.shape[0])
    if frame_count < 2:
        return qpos

    sample_times = _motion_sample_times(frame_count, source_dt, target_dt)
    target_frame_count = int(sample_times.shape[0])

    src_idx_float = sample_times / float(source_dt)
    idx0 = np.floor(src_idx_float).astype(np.int64)
    idx0 = np.clip(idx0, 0, frame_count - 2)
    idx1 = idx0 + 1
    alpha = np.clip(src_idx_float - idx0.astype(np.float64), 0.0, 1.0)

    qpos_np = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    out_qpos = np.zeros((target_frame_count, model.nq), dtype=np.float64)
    tangent = np.zeros(model.nv, dtype=np.float64)

    for sample_idx in range(target_frame_count):
        base_idx = int(idx0[sample_idx])
        next_idx = int(idx1[sample_idx])
        blend = float(alpha[sample_idx])

        if blend <= 0.0:
            out_qpos[sample_idx] = qpos_np[base_idx]
            continue
        if blend >= 1.0:
            out_qpos[sample_idx] = qpos_np[next_idx]
            continue

        mujoco.mj_differentiatePos(
            model,
            tangent,
            1.0,
            qpos_np[base_idx],
            qpos_np[next_idx],
        )
        q_interp = qpos_np[base_idx].copy()
        mujoco.mj_integratePos(model, q_interp, tangent, blend)
        out_qpos[sample_idx] = q_interp

    out_qpos[0] = qpos_np[0]
    if np.isclose(sample_times[-1], source_dt * (frame_count - 1)):
        out_qpos[-1] = qpos_np[-1]

    return torch.as_tensor(out_qpos, device=qpos.device, dtype=qpos.dtype)


def _resample_h5_qvel_frames(
    qvel: np.ndarray,
    *,
    source_dt: float,
    target_dt: float,
    device: torch.device,
) -> torch.Tensor:
    """Linearly resample HDF5 generalized velocities on the qpos time grid."""
    if np.isclose(source_dt, target_dt):
        return torch.tensor(qvel, dtype=torch.float32, device=device)

    frame_count = int(qvel.shape[0])
    if frame_count < 2:
        return torch.tensor(qvel, dtype=torch.float32, device=device)

    sample_times = _motion_sample_times(frame_count, source_dt, target_dt)
    source_times = np.arange(frame_count, dtype=np.float64) * float(source_dt)
    resampled = np.stack(
        [
            np.interp(sample_times, source_times, qvel[:, channel])
            for channel in range(qvel.shape[1])
        ],
        axis=1,
    )
    return torch.tensor(resampled, dtype=torch.float32, device=device)


def _resampled_frame_count(
    *,
    frame_count: int,
    source_dt: float,
    target_dt: float,
) -> int:
    if np.isclose(source_dt, target_dt) or frame_count < 2:
        return int(frame_count)

    duration = float(source_dt) * float(int(frame_count) - 1)
    return max(2, int(round(duration / float(target_dt))) + 1)


def _require_tracked_mimickit_frames(
    in_dict: dict[str, object], *, motion_path: object
) -> None:
    """Reject analysis clips whose optional ``tracked`` mask marks gaps."""

    if "tracked" not in in_dict:
        return
    tracked = np.asarray(in_dict["tracked"])
    if tracked.dtype != np.bool_ or tracked.ndim != 1:
        raise ValueError(
            f"MimicKit tracked mask must be a 1D bool array in {motion_path}, "
            f"got dtype={tracked.dtype} shape={tracked.shape}"
        )
    if not tracked.all():
        missing = np.flatnonzero(~tracked)
        raise ValueError(
            f"MimicKit motion {motion_path} contains untracked frames "
            f"beginning at {missing[:8].tolist()}; it is an analysis artifact, "
            "not a runtime reference."
        )


def _estimate_mimickit_payload_frame_count(
    in_dict: dict[str, object],
    *,
    motion_path: Path,
    target_dt: float,
    smooth_cycle: bool,
) -> int:
    frames = np.asarray(in_dict.get("frames", []))
    if frames.ndim != 2 or frames.shape[0] < 2:
        raise ValueError(
            f"Invalid MimicKit motion frame shape: {frames.shape} for {motion_path}"
        )
    _require_tracked_mimickit_frames(in_dict, motion_path=motion_path)
    if not np.isfinite(frames).all():
        raise ValueError("MimicKit frames contain non-finite values.")
    source_fps = float(in_dict.get("fps", 0.0))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"Invalid MimicKit fps: {source_fps} in {motion_path}")
    frame_count = int(frames.shape[0])
    loop_mode = int(in_dict.get("loop_mode", 1))
    if loop_mode == 1 and smooth_cycle and frame_count > 2:
        frame_count -= 1
    return _resampled_frame_count(
        frame_count=frame_count,
        source_dt=1.0 / source_fps,
        target_dt=float(target_dt),
    )


def estimate_loaded_motion_frame_count(
    motion_file: str,
    *,
    target_dt: float,
    smooth_cycle: bool = False,
) -> int:
    """Estimate the loaded frame count for a motion file.

    MimicKit ``.pkl`` remains an external dataset format and is deserialized
    with Python pickle; callers must provide a trusted, validated dataset path.
    The ``.npz`` path uses ``allow_pickle=False``.
    """
    motion_path = resolve_project_path(motion_file)
    motion_ext = motion_path.suffix.lower()

    if motion_ext == ".pkl":
        with motion_path.open("rb") as filestream:
            in_dict = pickle.load(filestream)
        if not isinstance(in_dict, dict):
            raise ValueError(f"Unsupported MimicKit motion format in {motion_path}")
        return _estimate_mimickit_payload_frame_count(
            in_dict,
            motion_path=motion_path,
            target_dt=float(target_dt),
            smooth_cycle=smooth_cycle,
        )

    if motion_ext == ".npz":
        with np.load(motion_path, allow_pickle=False) as payload:
            in_dict = {key: payload[key] for key in payload.files}
        return _estimate_mimickit_payload_frame_count(
            in_dict,
            motion_path=motion_path,
            target_dt=float(target_dt),
            smooth_cycle=smooth_cycle,
        )

    if motion_ext in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required for HDF5 motion loading. Install with: uv add h5py"
            ) from exc
        with h5py.File(motion_path, "r") as handle:
            qpos = handle["qpos"]
            fps = float(handle.attrs["fps"])
            frame_count = int(qpos.shape[0])
        if frame_count < 1:
            raise ValueError(f"HDF5 qpos is empty in {motion_path}")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"Invalid HDF5 fps: {fps} in {motion_path}")
        loaded_frame_count = _resampled_frame_count(
            frame_count=frame_count,
            source_dt=1.0 / fps,
            target_dt=float(target_dt),
        )
        if smooth_cycle and loaded_frame_count > 2:
            loaded_frame_count -= 1
        return loaded_frame_count

    if motion_ext == ".bvh":
        bvh = BVH()
        bvh.load(str(motion_path))
        local_positions, *_ = bvh.get_data()
        frame_count = int(local_positions.shape[0])
        if smooth_cycle and frame_count > 2:
            frame_count -= 1
        return _resampled_frame_count(
            frame_count=frame_count,
            source_dt=cast(float, bvh.data["frame_time"]),
            target_dt=float(target_dt),
        )

    raise ValueError(
        f"Unsupported motion format for frame-count estimate: {motion_path}"
    )


def _canonicalize_loop_qpos_source(
    qpos: torch.Tensor,
    model: mujoco.MjModel,
    root_qpos_adr: int,
    smooth_cycle: bool,
    smooth_cycle_blend_frames: int,
    smooth_cycle_mode: str = "tail_warp",
) -> torch.Tensor:
    """Build a source-rate canonical loop clip before target-rate projection."""
    if not smooth_cycle or int(qpos.shape[0]) <= 1:
        return qpos

    mode = _normalize_smooth_cycle_mode(smooth_cycle_mode)
    qpos_np_for_loop = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    loop_root_quat_np, loop_root_pos_np = _compute_loop_root_transform_np(
        qpos_np_for_loop, root_qpos_adr
    )
    if mode == "c1_bridge":
        qpos = _smooth_cycle_qpos_c1_bridge(
            qpos=qpos,
            model=model,
            root_qpos_adr=root_qpos_adr,
            blend_frames=int(smooth_cycle_blend_frames),
            loop_root_quat=loop_root_quat_np,
            loop_root_pos=loop_root_pos_np,
        )
    else:
        qpos = _smooth_cycle_qpos(
            qpos=qpos,
            model=model,
            root_qpos_adr=root_qpos_adr,
            blend_frames=int(smooth_cycle_blend_frames),
            loop_root_quat=loop_root_quat_np,
            loop_root_pos=loop_root_pos_np,
        )

    if int(qpos.shape[0]) > 2:
        qpos = qpos[:-1].contiguous()

    return qpos


def _recompute_mimickit_qvel_from_qpos(
    qpos: torch.Tensor,
    control_dt: float,
    model: mujoco.MjModel,
    root_qpos_adr: int,
    loop_root_quat: np.ndarray,
    loop_root_pos: np.ndarray,
    loop_boundary_vel_from_first_frame: bool,
    loop_boundary_vel_extrapolate: bool,
) -> torch.Tensor:
    """Compute generalized velocities for MimicKit qpos with optional cyclic seam handling."""
    return _recompute_qvel_from_qpos(
        qpos=qpos,
        model=model,
        control_dt=control_dt,
        root_qpos_adr=root_qpos_adr,
        loop_root_quat=loop_root_quat,
        loop_root_pos=loop_root_pos,
        loop_boundary_vel_from_first_frame=loop_boundary_vel_from_first_frame,
        loop_boundary_vel_extrapolate=loop_boundary_vel_extrapolate,
    )


def load_mimickit_pkl_file(
    motion_file: str,
    *,
    device: torch.device,
    model: mujoco.MjModel,
    options: MotionLoadOptions,
) -> dict[str, torch.Tensor | int]:
    """Load a trusted external MimicKit pickle motion dataset.

    The ``.pkl`` format is retained for dataset compatibility and is not a
    safe serialization boundary: Python pickle reconstruction can execute
    code supplied by the file. Only load files from a trusted, validated
    dataset source.
    """
    with open(motion_file, "rb") as filestream:
        in_dict = pickle.load(filestream)
    return load_mimickit_motion_payload(
        in_dict,
        motion_file=motion_file,
        device=device,
        model=model,
        options=options,
    )


def load_mimickit_npz_file(
    motion_file: str,
    *,
    device: torch.device,
    model: mujoco.MjModel,
    options: MotionLoadOptions,
) -> dict[str, torch.Tensor | int]:
    with np.load(motion_file, allow_pickle=False) as payload:
        in_dict = {key: payload[key] for key in payload.files}
    return load_mimickit_motion_payload(
        in_dict,
        motion_file=motion_file,
        device=device,
        model=model,
        options=options,
    )


def load_mimickit_motion_payload(
    in_dict: dict[str, object],
    motion_file: str,
    *,
    device: torch.device,
    model: mujoco.MjModel,
    options: MotionLoadOptions,
) -> dict[str, torch.Tensor | int]:
    motion_scale = options.motion_scale
    motion_offset = options.motion_offset
    target_dt = options.target_dt
    mimickit_joint_groups = options.mimickit_joint_groups
    loop_boundary_vel_from_first_frame = options.loop_boundary_vel_from_first_frame
    loop_boundary_vel_extrapolate = options.loop_boundary_vel_extrapolate
    smooth_cycle = options.smooth_cycle
    smooth_cycle_blend_frames = options.smooth_cycle_blend_frames
    smooth_cycle_mode = options.smooth_cycle_mode

    if not isinstance(in_dict, dict):
        raise ValueError(f"Unsupported MimicKit motion format in {motion_file}")

    frames = np.asarray(in_dict.get("frames", []), dtype=np.float32)
    if frames.ndim != 2 or frames.shape[0] < 2 or frames.shape[1] < 7:
        raise ValueError(
            f"Invalid MimicKit motion frame shape: {frames.shape} for {motion_file}"
        )

    _require_tracked_mimickit_frames(in_dict, motion_path=motion_file)
    if not np.isfinite(frames).all():
        raise ValueError("MimicKit frames contain non-finite values.")
    source_fps = float(in_dict.get("fps", 0.0))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"Invalid MimicKit fps: {source_fps} in {motion_file}")
    source_dt = 1.0 / source_fps
    control_dt = float(target_dt)

    loop_mode = int(in_dict.get("loop_mode", 1))

    frames_t = torch.as_tensor(frames, device=device, dtype=torch.float32)
    root_pos = frames_t[:, 0:3] * float(motion_scale)
    root_pos = (
        root_pos
        + torch.as_tensor(motion_offset, device=device, dtype=torch.float32)[None, :]
    )
    root_rot_exp_map = frames_t[:, 3:6]
    root_quat = torch_utils.scaled_axis_angle_to_quat(root_rot_exp_map)
    dof_pos = frames_t[:, 6:]

    if mimickit_joint_groups is None and model.nq != 7 + dof_pos.shape[1]:
        mimickit_joint_groups = _infer_mimickit_joint_groups_from_model(model)

    if mimickit_joint_groups is None:
        if dof_pos.shape[1] != model.nv - 6:
            raise ValueError(
                "MimicKit dof dimension mismatch: "
                f"motion has {dof_pos.shape[1]}, model expects {model.nv - 6}."
            )
        if model.nq != 7 + dof_pos.shape[1]:
            raise ValueError(
                "MimicKit model nq mismatch: "
                f"nq={model.nq}, expected {7 + dof_pos.shape[1]}."
            )
    else:
        mapped_dof_dim = 0
        max_qpos_end = 7
        for group in mimickit_joint_groups:
            dof_adr = int(group.get("dof_adr_local", -1))
            dof_dim = int(group.get("dof_dim", 0))
            qpos_adrs = [int(v) for v in group.get("qpos_adrs", [])]
            kind = int(group.get("kind", -1))

            if dof_adr < 0 or dof_dim <= 0:
                continue

            if dof_adr + dof_dim > dof_pos.shape[1]:
                raise ValueError(
                    "MimicKit joint-group dof mapping exceeds motion dimension: "
                    f"dof_adr={dof_adr}, dof_dim={dof_dim}, motion_dim={dof_pos.shape[1]}."
                )

            mapped_dof_dim += dof_dim

            if kind == 3:
                if len(qpos_adrs) != 1:
                    raise ValueError(
                        "Ball-joint MimicKit mapping must provide one qpos address."
                    )
                qpos_end = qpos_adrs[0] + 4
            elif kind == 4:
                if len(qpos_adrs) != 1:
                    raise ValueError(
                        "Projected hinge MimicKit mapping must provide one qpos address."
                    )
                if dof_dim != 3:
                    raise ValueError(
                        "Projected hinge MimicKit mapping expects 3 source DoFs."
                    )
                if "axis" not in group:
                    raise ValueError(
                        "Projected hinge MimicKit mapping requires an 'axis' entry."
                    )
                qpos_end = qpos_adrs[0] + 1
            else:
                if len(qpos_adrs) != dof_dim:
                    raise ValueError(
                        "MimicKit joint-group qpos mapping mismatch: "
                        f"kind={kind}, qpos_adrs={qpos_adrs}, dof_dim={dof_dim}."
                    )
                qpos_end = max(qpos_adrs) + 1

            max_qpos_end = max(max_qpos_end, qpos_end)

        if mapped_dof_dim != dof_pos.shape[1]:
            raise ValueError(
                "MimicKit joint-group dof mapping mismatch: "
                f"mapped={mapped_dof_dim}, motion_dim={dof_pos.shape[1]}."
            )
        if max_qpos_end > model.nq:
            raise ValueError(
                "MimicKit joint-group qpos mapping exceeds model nq: "
                f"required={max_qpos_end}, nq={model.nq}."
            )

    frame_count = int(root_pos.shape[0])

    qpos = torch.zeros((frame_count, model.nq), device=device, dtype=torch.float32)
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_quat
    if mimickit_joint_groups is None:
        qpos[:, 7:] = dof_pos
        recorded_qpos = np.ones(int(model.nq), dtype=bool)
    else:
        recorded_qpos = _apply_mimickit_joint_groups_to_qpos(
            qpos, dof_pos, mimickit_joint_groups
        )
        recorded_qpos[0:7] = True
    _rest_unrecorded_passive_coordinates(
        qpos, recorded_qpos, model=model, options=options
    )

    root_qpos_adr = -1
    for jnt_id in range(model.njnt):
        if model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
            root_qpos_adr = int(model.jnt_qposadr[jnt_id])
            break

    qpos_np_for_loop = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    loop_root_quat_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    loop_root_pos_np = np.zeros(3, dtype=np.float64)
    if loop_mode == 1:
        qpos = _canonicalize_loop_qpos_source(
            qpos=qpos,
            model=model,
            root_qpos_adr=root_qpos_adr,
            smooth_cycle=smooth_cycle,
            smooth_cycle_blend_frames=smooth_cycle_blend_frames,
            smooth_cycle_mode=smooth_cycle_mode,
        )

    qpos = _resample_qpos_frames(
        qpos=qpos,
        model=model,
        source_dt=source_dt,
        target_dt=control_dt,
    )
    frame_count = int(qpos.shape[0])
    qpos_np_for_loop = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    if loop_mode == 1:
        loop_root_quat_np, loop_root_pos_np = _compute_loop_root_transform_np(
            qpos_np_for_loop, root_qpos_adr
        )

    boundary_from_first = False
    boundary_extrapolate = True
    if loop_mode == 1:
        boundary_from_first = loop_boundary_vel_from_first_frame
        boundary_extrapolate = loop_boundary_vel_extrapolate
        if smooth_cycle:
            boundary_from_first = True
            boundary_extrapolate = False

    qvel = _recompute_mimickit_qvel_from_qpos(
        qpos=qpos,
        control_dt=control_dt,
        model=model,
        root_qpos_adr=root_qpos_adr,
        loop_root_quat=loop_root_quat_np,
        loop_root_pos=loop_root_pos_np,
        loop_boundary_vel_from_first_frame=boundary_from_first,
        loop_boundary_vel_extrapolate=boundary_extrapolate,
    )

    qpos_mask = torch.ones((model.nq), device=device, dtype=torch.bool)
    qvel_mask = torch.ones((model.nv), device=device, dtype=torch.bool)
    body_mask = torch.ones((model.nbody), device=device, dtype=torch.bool)
    reference_payload = materialize_mujoco_reference_payload(
        model=model,
        qpos=qpos,
        qvel=qvel,
        device=device,
    )
    xpos = reference_payload.xpos
    xquat = reference_payload.xquat
    if reference_payload.subtree_com is None:
        raise RuntimeError("SNU motion reference payload is missing subtree COM.")
    subtree_com = reference_payload.subtree_com
    reference_body_state = reference_payload.reference_body_state

    return {
        "qpos": qpos,
        "qvel": qvel,
        "xpos": xpos,
        "xquat": xquat,
        "subtree_com": subtree_com,
        "reference_body_state": reference_body_state,
        "qpos_mask": qpos_mask,
        "qvel_mask": qvel_mask,
        "body_mask": body_mask,
        "frame_count": frame_count,
        "loop_root_quat": torch.tensor(
            loop_root_quat_np, device=device, dtype=torch.float32
        ),
        "loop_root_pos": torch.tensor(
            loop_root_pos_np, device=device, dtype=torch.float32
        ),
    }


def _discard_unreadable_motion_cache(
    cache_file: Path, exc: BaseException, *, after_lock: bool
) -> None:
    """Drop a cache file that would not load, so the next attempt rebuilds it.

    The same recovery ran before and after the write lock; only the message
    differed, and both hand-rolled the missing-file case that `unlink` already
    has a flag for.
    """

    _LOGGER.debug(
        "Could not load motion cache %s%s; deleting and rebuilding: %s",
        cache_file,
        " after lock" if after_lock else "",
        exc,
        exc_info=True,
    )
    cache_file.unlink(missing_ok=True)


def _resample_gait_nearest(
    src: np.ndarray, sample_indices: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Resample a per-frame signal whose values must not be averaged.

    A gait phase index and a foot-contact flag are both categorical, so the
    nearest source frame is the only correct answer. They had a resampler each,
    written the same way.
    """

    nearest = np.rint(sample_indices).astype(np.int64)
    return torch.tensor(src[nearest], dtype=torch.float32, device=device)


def _resample_gait_continuous(
    src: np.ndarray,
    sample_indices: np.ndarray,
    device: torch.device,
    *,
    expected_ndim: int,
) -> torch.Tensor:
    """Linearly resample a per-frame signal, one channel or many.

    `expected_ndim` is what the caller's HDF5 key promises: the two separate
    functions this replaced would each fail on the other's rank, and a
    mis-shaped clip should still be rejected rather than quietly reinterpreted.
    """

    if src.ndim != expected_ndim:
        raise ValueError(
            f"expected a {expected_ndim}-D per-frame signal, got shape {src.shape}"
        )
    src_x = np.arange(src.shape[0], dtype=np.float64)
    if src.ndim == 1:
        resampled = np.interp(sample_indices, src_x, src)
    else:
        resampled = np.stack(
            [
                np.interp(sample_indices, src_x, src[:, channel])
                for channel in range(src.shape[1])
            ],
            axis=1,
        )
    return torch.tensor(resampled, dtype=torch.float32, device=device)


def _rest_unrecorded_passive_coordinates(
    qpos: np.ndarray | torch.Tensor,
    recorded_qpos: np.ndarray,
    *,
    model: mujoco.MjModel,
    options: MotionLoadOptions,
) -> None:
    """Put the passive coordinates a source did not record at ``qpos0``, in place.

    Passive coordinates (instantmimic.envs.dof_roles) are the model coordinates
    outside ``options.character_coordinates``. A character-only HDF5
    recording, a BVH body map and body-space MimicKit joint groups write only
    character coordinates, and the zero a source leaves behind is not a pose: a
    passive free body's quaternion would be (0, 0, 0, 0). Every source calls
    this on the qpos it read, before smoothing, resampling and velocity
    estimation, so the passive coordinates stay at ``qpos0`` and their velocity
    is zero. ``recorded_qpos`` (bool, ``nq``) marks the columns the source
    wrote; those keep their values, as in a whole-model recording.
    """
    character = options.character_coordinates
    if character is None:
        return
    recorded = np.asarray(recorded_qpos, dtype=bool)
    if recorded.shape != (int(model.nq),):
        raise ValueError(
            f"recorded_qpos has shape {recorded.shape}; expected ({int(model.nq)},)."
        )
    rest = ~recorded
    rest[np.asarray(character[0], dtype=np.int64)] = False
    columns = np.flatnonzero(rest)
    if columns.size == 0:
        return
    values = np.asarray(model.qpos0, dtype=np.float64)[columns]
    if isinstance(qpos, torch.Tensor):
        qpos[:, torch.as_tensor(columns, device=qpos.device)] = torch.as_tensor(
            values, dtype=qpos.dtype, device=qpos.device
        )
    else:
        qpos[:, columns] = values.astype(qpos.dtype)


def _expand_character_recording(
    qpos_np: np.ndarray,
    qvel_np: np.ndarray | None,
    *,
    model: mujoco.MjModel,
    options: MotionLoadOptions,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Expand a character-coordinate recording to the full model.

    A model with passive coordinates (flex vertices, free props, declared
    passive joints) can replay a recording made on the character alone: its
    qpos columns are the character's qpos addresses in model order, the passive
    ones take qpos0 and zero velocity like every other source's
    (`_rest_unrecorded_passive_coordinates`). Full-width recordings pass
    unchanged.
    """
    character = options.character_coordinates
    if character is None or qpos_np.ndim != 2:
        return qpos_np, qvel_np
    qpos_adr, dof_adr = character
    if int(qpos_np.shape[1]) == int(model.nq) or int(qpos_np.shape[1]) != len(qpos_adr):
        return qpos_np, qvel_np
    character_qpos = np.asarray(qpos_adr, dtype=np.int64)
    full_qpos = np.zeros((qpos_np.shape[0], int(model.nq)), dtype=qpos_np.dtype)
    full_qpos[:, character_qpos] = qpos_np
    recorded_qpos = np.zeros(int(model.nq), dtype=bool)
    recorded_qpos[character_qpos] = True
    _rest_unrecorded_passive_coordinates(
        full_qpos, recorded_qpos, model=model, options=options
    )
    full_qvel = None
    if qvel_np is not None:
        if qvel_np.ndim != 2 or int(qvel_np.shape[1]) != len(dof_adr):
            raise ValueError(
                f"HDF5 qvel shape {qvel_np.shape} does not match the character "
                f"recording's {len(dof_adr)} DOFs."
            )
        full_qvel = np.zeros((qvel_np.shape[0], int(model.nv)), dtype=qvel_np.dtype)
        full_qvel[:, np.asarray(dof_adr, dtype=np.int64)] = qvel_np
    return full_qpos, full_qvel


def load_h5_file(
    motion_file: str,
    *,
    device: torch.device,
    model: mujoco.MjModel,
    options: MotionLoadOptions,
) -> dict[str, torch.Tensor | int | float]:
    """Load a MuJoCo-space HDF5 motion file."""
    motion_scale = options.motion_scale
    motion_offset = options.motion_offset
    bvh_map = options.bvh_map
    target_dt = options.target_dt
    loop_boundary_vel_from_first_frame = options.loop_boundary_vel_from_first_frame
    loop_boundary_vel_extrapolate = options.loop_boundary_vel_extrapolate
    smooth_cycle = options.smooth_cycle
    smooth_cycle_blend_frames = options.smooth_cycle_blend_frames
    smooth_cycle_mode = options.smooth_cycle_mode
    root_auto_calibration = options.root_auto_calibration

    try:
        import h5py

    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 motion loading. Install with: uv add h5py"
        ) from exc

    with h5py.File(motion_file, "r") as handle:
        qpos_np = np.asarray(handle["qpos"][:], dtype=np.float32)
        qvel_np = (
            np.asarray(handle["qvel"][:], dtype=np.float32)
            if "qvel" in handle
            else None
        )
        fps = float(handle.attrs["fps"])
        gait_phase_np = (
            np.asarray(handle["gait_phase"][:], dtype=np.float32)
            if "gait_phase" in handle
            else None
        )
        motion_phase_np = (
            np.asarray(handle["motion_phase"][:], dtype=np.float32)
            if "motion_phase" in handle
            else None
        )
        foot_contact_np = (
            np.asarray(handle["foot_contact"][:], dtype=np.float32)
            if "foot_contact" in handle
            else None
        )
        foot_r_target_np = (
            np.asarray(handle["foot_r_target"][:], dtype=np.float32)
            if "foot_r_target" in handle
            else None
        )
        foot_l_target_np = (
            np.asarray(handle["foot_l_target"][:], dtype=np.float32)
            if "foot_l_target" in handle
            else None
        )
        ref_stride = float(handle.attrs["stride"]) if "stride" in handle.attrs else None
        ref_cadence = (
            float(handle.attrs["cadence"]) if "cadence" in handle.attrs else None
        )
        ref_forward_axis = (
            int(handle.attrs["forward_axis"])
            if "forward_axis" in handle.attrs
            else None
        )
        height_calibration_frame = (
            int(handle.attrs["height_calibration_frame"])
            if "height_calibration_frame" in handle.attrs
            else 0
        )
        qpos_complete = bool(handle.attrs.get("qpos_complete", False))

    qpos_np, qvel_np = _expand_character_recording(
        qpos_np, qvel_np, model=model, options=options
    )
    if (
        qpos_np.ndim != 2
        or qpos_np.shape[0] < 1
        or int(qpos_np.shape[1]) != int(model.nq)
    ):
        raise ValueError(
            f"HDF5 qpos shape {qpos_np.shape} does not match model.nq={model.nq}. "
            "Wrong skeleton or motion file?"
        )
    if not np.isfinite(qpos_np).all():
        raise ValueError("HDF5 qpos contains non-finite values.")
    if qvel_np is not None:
        expected_qvel_shape = (int(qpos_np.shape[0]), int(model.nv))
        if qvel_np.ndim != 2 or tuple(qvel_np.shape) != expected_qvel_shape:
            raise ValueError(
                f"HDF5 qvel shape {qvel_np.shape} does not match "
                f"expected {expected_qvel_shape}."
            )
        if not np.isfinite(qvel_np).all():
            raise ValueError("HDF5 qvel contains non-finite values.")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid HDF5 fps: {fps} in {motion_file}")

    source_dt = 1.0 / float(fps)
    control_dt = float(target_dt)
    qpos = torch.tensor(qpos_np, dtype=torch.float32, device=device)
    root_qpos_adr = _root_qpos_addr(model)
    has_free_root = root_qpos_adr >= 0

    if not has_free_root:
        requested_offset = _normalize_root_bake_offset(motion_offset)
        if (
            not np.isclose(float(motion_scale), 1.0)
            or any(not np.isclose(value, 0.0) for value in requested_offset)
            or root_auto_calibration
        ):
            raise ValueError(
                "motion scale, offset, and auto-calibration are free-root-only "
                "HDF5 options; convert scalar-root qpos in model coordinates first."
            )
        if smooth_cycle:
            raise ValueError(
                "smooth_cycle is a free-root-only HDF5 option for now; disable it "
                "for scalar-root model-space motion."
            )
        if loop_boundary_vel_from_first_frame:
            raise ValueError(
                "loop_boundary_vel_from_first_frame is a free-root-only HDF5 "
                "option; use loop_boundary_vel_extrapolate for scalar-root motion."
            )

    scale_f = float(motion_scale)
    qpos_modified = False
    if has_free_root and scale_f != 1.0:
        qpos[:, root_qpos_adr : root_qpos_adr + 3] *= scale_f
        qpos_modified = True

    height_calibration_frame = int(
        np.clip(height_calibration_frame, 0, int(qpos.shape[0]) - 1)
    )
    effective_root_bake_offset = (
        _effective_root_bake_offset(
            qpos_calibration=qpos[height_calibration_frame],
            model=model,
            root_qpos_adr=root_qpos_adr,
            root_bake_offset=motion_offset,
            root_auto_calibration=root_auto_calibration,
        )
        if has_free_root
        else [0.0, 0.0, 0.0]
    )
    offset_t = torch.tensor(
        effective_root_bake_offset, dtype=torch.float32, device=device
    )
    if has_free_root:
        qpos[:, root_qpos_adr : root_qpos_adr + 3] += offset_t.unsqueeze(0)

    # A complete qpos/qvel recording is already in model coordinates. Preserve
    # its physical joint state, including values beyond a soft hinge range;
    # qpos-only recordings retain the historical clamp behavior.
    preserve_complete_qpos = qvel_np is not None and qpos_complete
    if not preserve_complete_qpos:
        for joint_id in range(model.njnt):
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                continue
            if not bool(model.jnt_limited[joint_id]):
                continue
            qpos_adr = int(model.jnt_qposadr[joint_id])
            lo = float(model.jnt_range[joint_id, 0])
            hi = float(model.jnt_range[joint_id, 1])
            qpos_before_clamp = qpos[:, qpos_adr].clone()
            qpos[:, qpos_adr] = torch.clamp(qpos[:, qpos_adr], lo, hi)
            if not torch.equal(qpos[:, qpos_adr], qpos_before_clamp):
                qpos_modified = True

    resampled_qpos = not np.isclose(source_dt, control_dt)
    if not np.isclose(source_dt, control_dt):
        qpos = _resample_qpos_frames(qpos, model, source_dt, control_dt)

    if smooth_cycle:
        qpos_modified = True
        qpos = _canonicalize_loop_qpos_source(
            qpos=qpos,
            model=model,
            root_qpos_adr=root_qpos_adr,
            smooth_cycle=True,
            smooth_cycle_blend_frames=smooth_cycle_blend_frames,
            smooth_cycle_mode=smooth_cycle_mode,
        )

    frame_count = int(qpos.shape[0])
    if has_free_root:
        qpos_np64 = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
        loop_root_quat_np, loop_root_pos_np = _compute_loop_root_transform_np(
            qpos_np64, root_qpos_adr
        )
    else:
        root_pos_np, root_quat_np = _materialize_root_body_pose_np(
            model,
            qpos,
            _root_body_id(model),
        )
        loop_root_quat_np, loop_root_pos_np = (
            _compute_loop_root_transform_from_body_pose_np(
                root_pos_np,
                root_quat_np,
            )
        )
    if qvel_np is not None and not qpos_modified:
        if resampled_qpos:
            qvel = _resample_h5_qvel_frames(
                qvel_np,
                source_dt=source_dt,
                target_dt=control_dt,
                device=device,
            )
        else:
            qvel = torch.tensor(qvel_np, dtype=torch.float32, device=device)
    else:
        if qvel_np is not None and qpos_modified:
            _LOGGER.warning(
                "Ignoring measured HDF5 qvel after qpos transformation; "
                "recomputing qvel from transformed qpos for %s.",
                motion_file,
            )
        qvel = _recompute_qvel_from_qpos(
            qpos=qpos,
            model=model,
            control_dt=control_dt,
            root_qpos_adr=root_qpos_adr,
            loop_root_quat=loop_root_quat_np,
            loop_root_pos=loop_root_pos_np,
            loop_boundary_vel_from_first_frame=loop_boundary_vel_from_first_frame,
            loop_boundary_vel_extrapolate=loop_boundary_vel_extrapolate,
        )

    reference_payload = materialize_mujoco_reference_payload(
        model=model,
        qpos=qpos,
        qvel=qvel,
        device=device,
    )
    xpos = reference_payload.xpos
    xquat = reference_payload.xquat
    if reference_payload.subtree_com is None:
        raise RuntimeError("SNU motion reference payload is missing subtree COM.")
    subtree_com = reference_payload.subtree_com
    reference_body_state = reference_payload.reference_body_state

    mapped_body_names: set[str] | None = (
        None if qpos_complete else (bvh_map_body_names(bvh_map) if bvh_map else None)
    )
    qpos_mask = torch.zeros(model.nq, device=device, dtype=torch.bool)
    qvel_mask = torch.zeros(model.nv, device=device, dtype=torch.bool)
    body_mask = torch.zeros(model.nbody, device=device, dtype=torch.bool)

    for joint_id in range(model.njnt):
        jnt_type = int(model.jnt_type[joint_id])
        qpos_adr = int(model.jnt_qposadr[joint_id])
        dof_adr = int(model.jnt_dofadr[joint_id])
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if mapped_body_names is not None and body_name not in mapped_body_names:
            continue

        body_mask[body_id] = True
        if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
            qpos_mask[qpos_adr : qpos_adr + 7] = True
            qvel_mask[dof_adr : dof_adr + 6] = True
        elif jnt_type == int(mujoco.mjtJoint.mjJNT_BALL):
            qpos_mask[qpos_adr : qpos_adr + 4] = True
            qvel_mask[dof_adr : dof_adr + 3] = True
        elif jnt_type in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            qpos_mask[qpos_adr] = True
            qvel_mask[dof_adr] = True

    if mapped_body_names is None:
        body_mask[:] = True
    body_mask[0] = False

    result: dict[str, torch.Tensor | int | float] = {
        "qpos": qpos,
        "qvel": qvel,
        "xpos": xpos,
        "xquat": xquat,
        "subtree_com": subtree_com,
        "reference_body_state": reference_body_state,
        "qpos_mask": qpos_mask,
        "qvel_mask": qvel_mask,
        "body_mask": body_mask,
        "frame_count": frame_count,
        "root_bake_offset": torch.tensor(
            effective_root_bake_offset, device=device, dtype=torch.float32
        ),
        "loop_root_quat": torch.tensor(
            loop_root_quat_np, device=device, dtype=torch.float32
        ),
        "loop_root_pos": torch.tensor(
            loop_root_pos_np, device=device, dtype=torch.float32
        ),
    }

    src_frames = int(qpos_np.shape[0])
    gait_sample_indices = (
        _motion_sample_times(src_frames, source_dt, control_dt) / source_dt
    )
    if gait_phase_np is not None:
        result["gait_phase"] = _resample_gait_nearest(
            gait_phase_np, gait_sample_indices, device
        )
    if motion_phase_np is not None:
        result["motion_phase"] = _resample_gait_continuous(
            motion_phase_np, gait_sample_indices, device, expected_ndim=1
        )
    if foot_contact_np is not None:
        result["foot_contact"] = _resample_gait_nearest(
            foot_contact_np, gait_sample_indices, device
        )
    if foot_r_target_np is not None:
        result["foot_r_target"] = _resample_gait_continuous(
            foot_r_target_np, gait_sample_indices, device, expected_ndim=2
        )
    if foot_l_target_np is not None:
        result["foot_l_target"] = _resample_gait_continuous(
            foot_l_target_np, gait_sample_indices, device, expected_ndim=2
        )
    if ref_stride is not None:
        result["ref_stride"] = ref_stride
    if ref_cadence is not None:
        result["ref_cadence"] = ref_cadence
    if ref_forward_axis is not None:
        result["ref_forward_axis"] = ref_forward_axis

    return result


# Generalized coordinates per articulated joint type: (qpos width, qvel width).
_JOINT_WIDTHS = {
    int(mujoco.mjtJoint.mjJNT_BALL): (4, 3),
    int(mujoco.mjtJoint.mjJNT_SLIDE): (1, 1),
    int(mujoco.mjtJoint.mjJNT_HINGE): (1, 1),
}


def _rest_body_joints(
    model: mujoco.MjModel,
    *,
    bvh_map: dict[str, BvhMapValue],
    rest_bodies: Sequence[str],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Resolve ``rest_bodies`` to body ids and the qpos and DOF addresses of
    their own joints.

    A rest body is tracked like a mapped body, but its joints hold the model's
    ``qpos0`` with zero velocity instead of taking a BVH rotation. A body that
    a bvh_map entry also drives, a body the model lacks, a body without joints
    (nothing to hold) and the free root (a pose, not a rest angle) are refused.
    """

    driven = {
        target: key
        for key, value in bvh_map.items()
        for target in bvh_map_targets(value)
    }
    body_ids: list[int] = []
    qpos_columns: list[int] = []
    dof_columns: list[int] = []
    for name in rest_bodies:
        if name in driven:
            raise ValueError(
                f"rest_bodies body {name!r} is also driven by bvh_map entry "
                f"{driven[name]!r}; a body either follows the BVH or rests."
            )
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"rest_bodies body {name!r} not found in MuJoCo model.")
        if body_id in body_ids:
            raise ValueError(f"rest_bodies lists body {name!r} more than once.")
        jnt_adr = int(model.body_jntadr[body_id])
        jnt_num = int(model.body_jntnum[body_id])
        if jnt_num == 0:
            raise ValueError(
                f"rest_bodies body {name!r} has no joints to hold at rest."
            )
        for jnt_id in range(jnt_adr, jnt_adr + jnt_num):
            jnt_type = int(model.jnt_type[jnt_id])
            if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(
                    f"rest_bodies body {name!r} has a free joint; only articulated "
                    "joints can be held at rest."
                )
            nq, nv = _JOINT_WIDTHS[jnt_type]
            qpos_adr = int(model.jnt_qposadr[jnt_id])
            dof_adr = int(model.jnt_dofadr[jnt_id])
            qpos_columns.extend(range(qpos_adr, qpos_adr + nq))
            dof_columns.extend(range(dof_adr, dof_adr + nv))
        body_ids.append(body_id)
    return (
        body_ids,
        np.asarray(qpos_columns, dtype=np.int64),
        np.asarray(dof_columns, dtype=np.int64),
    )


def _hold_rest_body_joints(
    qpos: torch.Tensor,
    qvel: torch.Tensor | None,
    *,
    model: mujoco.MjModel,
    qpos_columns: np.ndarray,
    dof_columns: np.ndarray,
) -> None:
    """Set the rest-body coordinates to ``qpos0`` and their velocity to zero, in place."""

    if qpos_columns.size == 0:
        return
    qpos[:, torch.as_tensor(qpos_columns, device=qpos.device)] = torch.as_tensor(
        np.asarray(model.qpos0, dtype=np.float64)[qpos_columns],
        dtype=qpos.dtype,
        device=qpos.device,
    )
    if qvel is not None:
        qvel[:, torch.as_tensor(dof_columns, device=qvel.device)] = 0.0


def _load_motion_file_uncached(
    motion_file: str,
    *,
    device: torch.device,
    model: mujoco.MjModel,
    options: MotionLoadOptions,
):
    """Load a motion clip. Despite the name, this is the entry point for all five formats.

    Dispatches on the file extension to the `.pkl`, `.npz`, `.h5`/`.hdf5` and
    `.bvh` loaders. The name is kept because it is a public contract; renaming it
    would mean adding an alias, which this repository does not do without a
    stated need.

    Three arguments are format-specific and are refused rather than ignored
    when the chosen format does not implement them: `root_auto_calibration`
    reaches only the HDF5 loader, `mimickit_joint_groups` only the MimicKit
    `.pkl` and `.npz` loaders, and `rest_bodies` only the BVH loader.

    The `.pkl` branch intentionally retains the external MimicKit pickle
    format. Treat that input as trusted dataset data; it is separate from the
    restricted loader used for machine-local MJCF caches.
    """

    motion_ext = os.path.splitext(motion_file)[1].lower()
    if motion_ext == ".pkl":
        return load_mimickit_pkl_file(
            motion_file=motion_file, device=device, model=model, options=options
        )
    if motion_ext == ".npz":
        return load_mimickit_npz_file(
            motion_file=motion_file, device=device, model=model, options=options
        )
    if motion_ext in {".h5", ".hdf5"}:
        return load_h5_file(
            motion_file=motion_file, device=device, model=model, options=options
        )

    motion_scale = options.motion_scale
    motion_offset = options.motion_offset
    bvh_map = options.bvh_map
    target_dt = options.target_dt
    loop_boundary_vel_from_first_frame = options.loop_boundary_vel_from_first_frame
    loop_boundary_vel_extrapolate = options.loop_boundary_vel_extrapolate
    smooth_cycle = options.smooth_cycle
    smooth_cycle_blend_frames = options.smooth_cycle_blend_frames
    smooth_cycle_mode = options.smooth_cycle_mode
    rest_body_ids, rest_qpos_columns, rest_dof_columns = _rest_body_joints(
        model, bvh_map=bvh_map, rest_bodies=options.rest_bodies
    )

    # load bvh file
    bvh = BVH()
    bvh.load(motion_file)
    bvh.set_scale(motion_scale)
    (
        local_rotations,
        local_positions,
        parents,
        offsets,
        end_sites,
        end_site_parents,
    ) = bvh.get_data()

    local_positions = torch.tensor(local_positions, device=device, dtype=torch.float32)
    local_rotations_mat = quat_torch.to_matrix(
        torch.tensor(local_rotations, device=device, dtype=torch.float32)
    )

    # convert y up to z up
    transformation_matrix = torch.tensor(
        Y_UP_TO_Z_UP, device=device, dtype=torch.float32
    )
    local_positions = torch.einsum(
        "ij,...j->...i", transformation_matrix, local_positions
    )
    local_rotations_mat = torch.einsum(
        "ij,...jk,lk->...il",
        transformation_matrix,
        local_rotations_mat,
        transformation_matrix,
    )

    # apply motion offset
    motion_offset = torch.tensor(motion_offset, device=device, dtype=torch.float32)
    local_positions[:, 0, :] += motion_offset[None, :]
    local_rotations = quat_torch.from_matrix(local_rotations_mat)

    motion_dt = cast(float, bvh.data["frame_time"])

    frame_count = local_positions.shape[0]
    control_dt = motion_dt

    bvh_joint_names = cast(np.ndarray[str], bvh.data["names"])

    total_nv = model.nv
    total_nq = model.nq
    body_count = model.nbody

    # Init qpos/qvel
    qpos = torch.zeros((frame_count, total_nq), device=device, dtype=torch.float32)
    qvel = torch.zeros((frame_count, total_nv), device=device, dtype=torch.float32)

    # Masks
    qpos_mask = torch.zeros((total_nq), device=device, dtype=torch.bool)
    qvel_mask = torch.zeros((total_nv), device=device, dtype=torch.bool)
    body_mask = torch.zeros((body_count), device=device, dtype=torch.bool)

    # Store root joint address for loop calc if found
    root_qpos_adr = -1

    # Iterate over BVH joints and map to MuJoCo joints
    # A bvh_map entry may list several bodies; each gets the joint's rotation.
    mapped_targets = [
        (i, target)
        for i, name in enumerate(bvh_joint_names)
        if name in bvh_map
        for target in bvh_map_targets(bvh_map[name])
    ]
    for i, target_body_name in mapped_targets:
        # Find body ID in MuJoCo
        target_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, target_body_name
        )
        if target_body_id < 0:
            raise ValueError(f"Body {target_body_name} not found in MuJoCo model")

        # Map body mask
        body_mask[target_body_id] = True

        jnt_num = model.body_jntnum[target_body_id]
        jnt_adr = model.body_jntadr[target_body_id]

        if jnt_num == 0:
            continue

        # For 3-DOF hinge stacks (e.g., hip/ankle/shoulder), decompose using
        # the exact intrinsic axis order declared in the MJCF body.
        hinge_angle_by_joint_id: dict[int, torch.Tensor] = {}
        hinge_joint_ids: list[int] = []
        hinge_axis_labels: list[str] = []
        hinge_axis_signs: list[float] = []
        hinge_axes_are_cardinal = True

        for j in range(jnt_num):
            jnt_id = jnt_adr + j
            if model.jnt_type[jnt_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            hinge_joint_ids.append(jnt_id)
            axis = np.asarray(model.jnt_axis[jnt_id], dtype=np.float64)
            axis_idx = int(np.argmax(np.abs(axis)))
            axis_mag = float(np.abs(axis[axis_idx]))
            orth = np.delete(axis, axis_idx)
            if axis_mag < 1.0 - 1.0e-4 or not np.allclose(orth, 0.0, atol=1.0e-4):
                hinge_axes_are_cardinal = False
                continue
            hinge_axis_labels.append("xyz"[axis_idx])
            hinge_axis_signs.append(1.0 if axis[axis_idx] >= 0.0 else -1.0)

        if (
            len(hinge_joint_ids) == 3
            and hinge_axes_are_cardinal
            and len(set(hinge_axis_labels)) == 3
        ):
            axis_order = np.asarray(hinge_axis_labels, dtype="<U1")
            order = np.broadcast_to(axis_order, (frame_count, 3))
            hinge_euler = quat_torch.to_euler(local_rotations[:, i], order)
            hinge_euler = (hinge_euler + np.pi) % (2.0 * np.pi) - np.pi
            raw_lower = np.full(3, -np.inf, dtype=np.float64)
            raw_upper = np.full(3, np.inf, dtype=np.float64)
            for k, hinge_jnt_id in enumerate(hinge_joint_ids):
                if model.jnt_limited[hinge_jnt_id]:
                    lo = float(model.jnt_range[hinge_jnt_id, 0])
                    hi = float(model.jnt_range[hinge_jnt_id, 1])
                    if hinge_axis_signs[k] < 0.0:
                        raw_lower[k] = -hi
                        raw_upper[k] = -lo
                    else:
                        raw_lower[k] = lo
                        raw_upper[k] = hi
            hinge_euler = _select_continuous_tait_bryan_sequence(
                hinge_euler, raw_lower, raw_upper
            )
            for k, hinge_jnt_id in enumerate(hinge_joint_ids):
                val = hinge_euler[:, k]
                if hinge_axis_signs[k] < 0.0:
                    val = -val
                hinge_angle_by_joint_id[hinge_jnt_id] = val
        elif len(hinge_joint_ids) >= 2 and len(hinge_joint_ids) == jnt_num:
            # Any other hinge sequence (e.g. a talocrural + oblique subtalar
            # pair): least-squares decomposition in MuJoCo's joint order.
            # Projecting the rotation onto each axis separately ignores
            # that the joints compose.
            lower = np.full(len(hinge_joint_ids), -np.inf)
            upper = np.full(len(hinge_joint_ids), np.inf)
            for k, hinge_jnt_id in enumerate(hinge_joint_ids):
                if model.jnt_limited[hinge_jnt_id]:
                    lower[k], upper[k] = model.jnt_range[hinge_jnt_id]
            chain = _decompose_hinge_chain(
                local_rotations[:, i],
                np.asarray(model.jnt_axis[hinge_joint_ids], dtype=np.float64),
                lower,
                upper,
            )
            for k, hinge_jnt_id in enumerate(hinge_joint_ids):
                hinge_angle_by_joint_id[hinge_jnt_id] = (
                    _select_continuous_angle_sequence(
                        chain[:, k],
                        raw_lower=None if np.isinf(lower[k]) else float(lower[k]),
                        raw_upper=None if np.isinf(upper[k]) else float(upper[k]),
                    )
                )

        fallback_hinge_angle_by_joint_id: dict[int, torch.Tensor] = {}

        for j in range(jnt_num):
            jnt_id = jnt_adr + j
            jnt_type = model.jnt_type[jnt_id]
            qpos_adr = model.jnt_qposadr[jnt_id]
            dof_adr = model.jnt_dofadr[jnt_id]

            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                assert qpos_adr == 0, "Root joint is expected to be at index 0"
                # Root Joint Handling (Free Joint)
                # Use index i (which should correspond to root in BVH if mapped correctly)
                # Usually root is index 0 in BVH, but we rely on bvh_map.

                # Position (3) + Quaternion (4)
                qpos[:, qpos_adr : qpos_adr + 3] = local_positions[
                    :, i
                ]  # Usually i=0 for root
                qpos[:, qpos_adr + 3 : qpos_adr + 7] = local_rotations[:, i]

                # Velocity
                # Linear velocity (3) - Global
                qvel[:-1, dof_adr : dof_adr + 3] = (
                    qpos[1:, qpos_adr : qpos_adr + 3]
                    - qpos[:-1, qpos_adr : qpos_adr + 3]
                ) / control_dt

                # Angular velocity (3) - Local (Body Frame)
                q_prev = qpos[:-1, qpos_adr + 3 : qpos_adr + 7]
                q_next = qpos[1:, qpos_adr + 3 : qpos_adr + 7]
                q_rel = torch_utils.quat_mul(torch_utils.quat_conj(q_prev), q_next)
                # Ensure continuity? Quat diff usually small.
                q_rel = torch.where(q_rel[:, 0:1] < 0.0, -q_rel, q_rel)
                omega_local = torch_utils.quat_to_scaled_axis_angle(q_rel) / control_dt

                qvel[:-1, dof_adr + 3 : dof_adr + 6] = omega_local

                # Mark masks (7 for pos, 6 for vel)
                qpos_mask[qpos_adr : qpos_adr + 7] = True
                qvel_mask[dof_adr : dof_adr + 6] = True

                # Store for loop offset calculation
                # Assuming there's only one root/free joint of interest for locomotion
                if root_qpos_adr == -1:
                    root_qpos_adr = qpos_adr

            elif jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                if jnt_id in hinge_angle_by_joint_id:
                    qpos[:, qpos_adr] = hinge_angle_by_joint_id[jnt_id]
                else:
                    if jnt_id not in fallback_hinge_angle_by_joint_id:
                        joint_axis = np.asarray(
                            model.jnt_axis[jnt_id], dtype=np.float64
                        )
                        val = _extract_hinge_angle_from_local_rotation(
                            local_rotations[:, i], joint_axis
                        )
                        if model.jnt_limited[jnt_id]:
                            val = _select_continuous_angle_sequence(
                                val,
                                raw_lower=float(model.jnt_range[jnt_id, 0]),
                                raw_upper=float(model.jnt_range[jnt_id, 1]),
                            )
                        else:
                            val = _select_continuous_angle_sequence(val)
                        fallback_hinge_angle_by_joint_id[jnt_id] = val
                    qpos[:, qpos_adr] = fallback_hinge_angle_by_joint_id[jnt_id]

                # Limits
                lower_limit = model.jnt_range[jnt_id, 0]
                upper_limit = model.jnt_range[jnt_id, 1]
                is_limited = model.jnt_limited[jnt_id]

                if is_limited:
                    qpos[:, qpos_adr] = torch.clamp(
                        qpos[:, qpos_adr], lower_limit, upper_limit
                    )

                # Velocity
                qvel[:-1, dof_adr] = (
                    qpos[1:, qpos_adr] - qpos[:-1, qpos_adr]
                ) / control_dt

                qpos_mask[qpos_adr] = True
                qvel_mask[dof_adr] = True

            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                qpos[:, qpos_adr : qpos_adr + 4] = local_rotations[:, i]

                # Velocity (Angular velocity in 3D)
                q_prev = local_rotations[:-1, i]
                q_next = local_rotations[1:, i]

                q_rel = torch_utils.quat_mul(torch_utils.quat_conj(q_prev), q_next)

                # Convert to angular velocity (scaled axis-angle / dt)
                omega_local = torch_utils.quat_to_scaled_axis_angle(q_rel) / control_dt
                qvel[:-1, dof_adr : dof_adr + 3] = omega_local

                qpos_mask[qpos_adr : qpos_adr + 4] = True
                qvel_mask[dof_adr : dof_adr + 3] = True

            elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                # Slide joint (1 DOF translation)
                # Not strictly implemented for BVH usually, but can look at position
                pass

    # Rest bodies are tracked and their joints count as recorded, at qpos0.
    for body_id in rest_body_ids:
        body_mask[body_id] = True
    qpos_mask[torch.as_tensor(rest_qpos_columns, device=device)] = True
    qvel_mask[torch.as_tensor(rest_dof_columns, device=device)] = True
    _hold_rest_body_joints(
        qpos,
        None,
        model=model,
        qpos_columns=rest_qpos_columns,
        dof_columns=rest_dof_columns,
    )

    _rest_unrecorded_passive_coordinates(
        qpos, qpos_mask.cpu().numpy(), model=model, options=options
    )

    qpos = _canonicalize_loop_qpos_source(
        qpos=qpos,
        model=model,
        root_qpos_adr=root_qpos_adr,
        smooth_cycle=smooth_cycle,
        smooth_cycle_blend_frames=smooth_cycle_blend_frames,
        smooth_cycle_mode=smooth_cycle_mode,
    )

    qpos = _resample_qpos_frames(
        qpos=qpos,
        model=model,
        source_dt=motion_dt,
        target_dt=target_dt,
    )
    frame_count = int(qpos.shape[0])
    control_dt = float(target_dt)
    qpos_np_for_loop = qpos.detach().cpu().numpy().astype(np.float64, copy=False)
    loop_root_quat_np, loop_root_pos_np = _compute_loop_root_transform_np(
        qpos_np_for_loop, root_qpos_adr
    )

    # Canonical generalized velocity from qpos (MuJoCo diff, cyclic boundary).
    boundary_from_first = loop_boundary_vel_from_first_frame
    boundary_extrapolate = loop_boundary_vel_extrapolate
    if smooth_cycle:
        # In smooth_cycle mode, always use cyclic boundary diff from first-next pose.
        boundary_from_first = True
        boundary_extrapolate = False

    qvel = _recompute_qvel_from_qpos(
        qpos=qpos,
        model=model,
        control_dt=control_dt,
        root_qpos_adr=root_qpos_adr,
        loop_root_quat=loop_root_quat_np,
        loop_root_pos=loop_root_pos_np,
        loop_boundary_vel_from_first_frame=boundary_from_first,
        loop_boundary_vel_extrapolate=boundary_extrapolate,
    )
    # Smoothing and resampling interpolate through qpos; pin the rest values
    # exactly so the reference holds them on every frame.
    _hold_rest_body_joints(
        qpos,
        qvel,
        model=model,
        qpos_columns=rest_qpos_columns,
        dof_columns=rest_dof_columns,
    )

    reference_payload = materialize_mujoco_reference_payload(
        model=model,
        qpos=qpos,
        qvel=qvel,
        device=device,
    )
    xpos = reference_payload.xpos
    xquat = reference_payload.xquat
    if reference_payload.subtree_com is None:
        raise RuntimeError("SNU motion reference payload is missing subtree COM.")
    subtree_com = reference_payload.subtree_com
    reference_body_state = reference_payload.reference_body_state

    return {
        "qpos": qpos,
        "qvel": qvel,
        "xpos": xpos,
        "xquat": xquat,
        "subtree_com": subtree_com,
        "reference_body_state": reference_body_state,
        "qpos_mask": qpos_mask,
        "qvel_mask": qvel_mask,
        "body_mask": body_mask,
        "frame_count": frame_count,
        "loop_root_quat": torch.tensor(
            loop_root_quat_np, device=device, dtype=torch.float32
        ),
        "loop_root_pos": torch.tensor(
            loop_root_pos_np, device=device, dtype=torch.float32
        ),
    }


# Which loader accepts which of the two format-specific options. Requesting one
# a format does not implement used to be silently dropped on the way to the
# loader, so a config could ask for root auto-calibration on a `.bvh` clip and
# get an uncalibrated motion with no indication.
_ROOT_AUTO_CALIBRATION_FORMATS = frozenset({".h5", ".hdf5"})
_MIMICKIT_JOINT_GROUP_FORMATS = frozenset({".pkl", ".npz"})
_REST_BODY_FORMATS = frozenset({".bvh"})


def _reject_options_the_format_ignores(
    motion_ext: str,
    *,
    mimickit_joint_groups: list[dict[str, object]] | None,
    root_auto_calibration: bool,
    rest_bodies: Sequence[str] = (),
) -> None:
    if root_auto_calibration and motion_ext not in _ROOT_AUTO_CALIBRATION_FORMATS:
        raise NotImplementedError(
            f"root_auto_calibration is not implemented for {motion_ext} motions; "
            f"it is only honoured for {sorted(_ROOT_AUTO_CALIBRATION_FORMATS)}."
        )
    if mimickit_joint_groups and motion_ext not in _MIMICKIT_JOINT_GROUP_FORMATS:
        raise NotImplementedError(
            f"mimickit_joint_groups is not implemented for {motion_ext} motions; "
            f"it is only honoured for {sorted(_MIMICKIT_JOINT_GROUP_FORMATS)}."
        )
    if rest_bodies and motion_ext not in _REST_BODY_FORMATS:
        raise NotImplementedError(
            f"rest_bodies is not implemented for {motion_ext} motions; "
            f"it is only honoured for {sorted(_REST_BODY_FORMATS)}."
        )


def load_bvh_file(
    motion_file: str,
    motion_scale: float,
    motion_offset: list[float],
    bvh_map: dict[str, BvhMapValue],
    device: torch.device,
    model: mujoco.MjModel,
    target_dt: float,
    mimickit_joint_groups: list[dict[str, object]] | None = None,
    loop_boundary_vel_from_first_frame: bool = True,
    loop_boundary_vel_extrapolate: bool = False,
    smooth_cycle: bool = False,
    smooth_cycle_blend_frames: int = 30,
    smooth_cycle_mode: str = "tail_warp",
    root_auto_calibration: bool = False,
    cache_dir: str | None = None,
    cache_mode: str = "off",
    character_coordinates: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    rest_bodies: Sequence[str] = (),
):
    # Before the cache lookup, not inside the uncached loader: a cache entry
    # written before this check existed is still eligible, so validating later
    # let `cache_mode=auto` return the uncalibrated motion the check refuses
    # while deleting the cache made the same request raise.
    _reject_options_the_format_ignores(
        os.path.splitext(motion_file)[1].lower(),
        mimickit_joint_groups=mimickit_joint_groups,
        root_auto_calibration=root_auto_calibration,
        rest_bodies=rest_bodies,
    )
    if rest_bodies:
        # Like the format check: a request the loader would refuse must not be
        # answered from the cache.
        _rest_body_joints(model, bvh_map=bvh_map, rest_bodies=rest_bodies)
    normalized_cache_mode = str(cache_mode).strip().lower()
    if normalized_cache_mode not in {"off", "auto", "refresh"}:
        raise ValueError(
            f"Unsupported motion cache mode: {cache_mode!r}. "
            "Expected one of: off, auto, refresh."
        )

    options = MotionLoadOptions(
        motion_scale=motion_scale,
        motion_offset=motion_offset,
        bvh_map=bvh_map,
        target_dt=target_dt,
        mimickit_joint_groups=mimickit_joint_groups,
        loop_boundary_vel_from_first_frame=loop_boundary_vel_from_first_frame,
        loop_boundary_vel_extrapolate=loop_boundary_vel_extrapolate,
        smooth_cycle=smooth_cycle,
        smooth_cycle_blend_frames=smooth_cycle_blend_frames,
        smooth_cycle_mode=smooth_cycle_mode,
        root_auto_calibration=root_auto_calibration,
        character_coordinates=character_coordinates,
        rest_bodies=tuple(str(name) for name in rest_bodies),
    )
    motion_path = resolve_project_path(motion_file)
    cache_root = resolve_cache_path(cache_dir or "motion_cache")
    cache_metadata = options.cache_metadata(motion_file=motion_path, model=model)
    cache_file = motion_cache_file_path(cache_dir=cache_root, metadata=cache_metadata)

    if normalized_cache_mode == "auto" and cache_file.exists():
        try:
            return load_motion_cache(cache_file, device=device)
        except Exception as exc:  # noqa: BLE001 - corrupt/stale cache should be rebuilt.
            _discard_unreadable_motion_cache(cache_file, exc, after_lock=False)

    def _build_uncached_payload() -> dict[str, torch.Tensor | int]:
        return _load_motion_file_uncached(
            motion_file=str(motion_path),
            device=device,
            model=model,
            options=options,
        )

    if normalized_cache_mode == "off":
        return _build_uncached_payload()

    with acquire_motion_cache_write_lock(cache_file) as owns_lock:
        if normalized_cache_mode == "auto" and cache_file.exists():
            try:
                return load_motion_cache(cache_file, device=device)
            except Exception as exc:  # noqa: BLE001 - corrupt/stale cache should be rebuilt.
                _discard_unreadable_motion_cache(cache_file, exc, after_lock=True)

        payload = _build_uncached_payload()
        if owns_lock:
            save_motion_cache(
                cache_file,
                metadata=cache_metadata,
                payload=payload,
            )
        return payload
