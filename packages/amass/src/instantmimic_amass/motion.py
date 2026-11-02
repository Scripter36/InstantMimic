from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

LOOP_MODE_TO_INT = {"clamp": 0, "wrap": 1}


LOOP_MODE_ENCODING = "mimickit_clamp0_wrap1"


def _quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    values = np.asarray(quat_wxyz, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape [F,4], got {values.shape}")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("Cannot convert zero-length quaternion to rotation vector.")
    values = values / norms
    return Rotation.from_quat(values[:, [1, 2, 3, 0]]).as_rotvec()


quat_wxyz_to_rotvec = _quat_wxyz_to_rotvec


def _free_root_layout(model: mujoco.MjModel) -> tuple[int, int]:
    free_joints = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(free_joints) != 1:
        raise ValueError(
            "MimicKit conversion requires exactly one free root joint; "
            f"found {len(free_joints)}."
        )
    joint_id = free_joints[0]
    qpos_adr = int(model.jnt_qposadr[joint_id])
    dof_adr = int(model.jnt_dofadr[joint_id])
    if qpos_adr != 0 or dof_adr != 0:
        raise ValueError(
            "MimicKit conversion requires the free root joint at qpos/dof address 0; "
            f"got qpos={qpos_adr}, dof={dof_adr}."
        )
    if model.nq < 7 or model.nv < 6:
        raise ValueError(
            f"Invalid free-root model layout: nq={model.nq}, nv={model.nv}."
        )
    return qpos_adr, dof_adr


def qpos_to_mimickit_frames(qpos: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Convert MuJoCo qpos rows to MimicKit ``[root, model DOFs]`` frames.

    The output stores root translation, root exponential-map rotation, then
    non-root joint coordinates in model DOF order.  Ball joints use local
    quaternion logarithms (scaled axis-angle), matching the repository motion
    loader.  A free root at the beginning of the model is required explicitly;
    silently assuming that layout would corrupt models with a different root.
    """

    root_qadr, root_dofadr = _free_root_layout(model)
    del root_dofadr
    values = np.asarray(qpos, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != model.nq:
        raise ValueError(f"Expected qpos shape [F,{model.nq}], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("qpos must contain only finite values.")

    # Keep the float64 precision produced by the retargeting solver.  The
    # runtime loader may downcast to float32 on the selected device, but the
    # artifact itself must retain enough precision for CPU audits and exact
    # floor-margin round trips.
    frames = np.zeros((values.shape[0], model.nv), dtype=np.float64)
    frames[:, 0:3] = values[:, root_qadr : root_qadr + 3]
    frames[:, 3:6] = _quat_wxyz_to_rotvec(values[:, root_qadr + 3 : root_qadr + 7])

    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    ball_type = int(mujoco.mjtJoint.mjJNT_BALL)
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == free_type:
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        dof_offset = int(model.jnt_dofadr[joint_id]) - 6
        if dof_offset < 0:
            raise ValueError(
                "Non-root joint appears before the six free-root DOFs: "
                f"joint_id={joint_id}, dof_adr={model.jnt_dofadr[joint_id]}"
            )
        frame_adr = 6 + dof_offset
        if joint_type == ball_type:
            frames[:, frame_adr : frame_adr + 3] = _quat_wxyz_to_rotvec(
                values[:, qadr : qadr + 4]
            )
        elif joint_type in (hinge_type, slide_type):
            frames[:, frame_adr] = values[:, qadr]
        else:
            raise ValueError(
                f"Unsupported joint type {joint_type} for joint {joint_id}"
            )
    return frames


def mimickit_frames_to_qpos(frames: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Convert MimicKit ``[root, model DOFs]`` frames to MuJoCo qpos rows.

    MimicKit stores the free-root translation and exponential-map rotation in
    the first six columns, followed by non-root coordinates in MuJoCo DOF
    order. Ball-joint coordinates are scaled axis-angle vectors and are
    reconstructed as local wxyz quaternions. The decoder deliberately starts
    from zero qpos, matching the legacy offline loader's model-independent
    frame interpretation.
    """

    root_qadr, root_dofadr = _free_root_layout(model)
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != model.nv:
        raise ValueError(f"Expected frames shape [F,{model.nv}], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("frames must contain only finite values.")

    qpos = np.zeros((values.shape[0], model.nq), dtype=np.float64)
    qpos[:, root_qadr : root_qadr + 3] = values[:, 0:3]
    root_xyzw = Rotation.from_rotvec(values[:, 3:6]).as_quat()
    qpos[:, root_qadr + 3 : root_qadr + 7] = root_xyzw[:, [3, 0, 1, 2]]

    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    ball_type = int(mujoco.mjtJoint.mjJNT_BALL)
    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == free_type:
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        dof_offset = int(model.jnt_dofadr[joint_id]) - root_dofadr
        if dof_offset < 6:
            raise ValueError(
                "Non-root joint appears before the six free-root DOFs: "
                f"joint_id={joint_id}, dof_adr={model.jnt_dofadr[joint_id]}"
            )
        frame_adr = dof_offset
        if joint_type == ball_type:
            quat_xyzw = Rotation.from_rotvec(
                values[:, frame_adr : frame_adr + 3]
            ).as_quat()
            qpos[:, qadr : qadr + 4] = quat_xyzw[:, [3, 0, 1, 2]]
        elif joint_type in (hinge_type, slide_type):
            qpos[:, qadr] = values[:, frame_adr]
        else:
            raise ValueError(
                f"Unsupported joint type {joint_type} for joint {joint_id}"
            )
    return qpos


def build_mimickit_payload(
    frames: np.ndarray,
    fps: float,
    *,
    loop_mode: str,
    motion_joint_layout: str = "model",
    robot_xml: Path | None = None,
    source_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the portable MimicKit dictionary used by NPZ and pickle writers."""

    if loop_mode not in LOOP_MODE_TO_INT:
        raise ValueError(f"Unsupported loop mode {loop_mode!r}")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    frames = np.asarray(frames)
    if frames.dtype != np.float64:
        # Preserve the established float32 payload contract for callers that
        # already provide float32 frames, while retaining float64 frames from
        # the native qpos conversion above.
        frames = frames.astype(np.float32, copy=False)
    if frames.ndim != 2 or not np.isfinite(frames).all():
        raise ValueError(f"frames must be a finite 2D array, got {frames.shape}")
    payload: dict[str, Any] = {
        "fps": fps,
        "frames": frames,
        "loop_mode": LOOP_MODE_TO_INT[loop_mode],
        "loop_mode_encoding": LOOP_MODE_ENCODING,
        "format": "mimickit_motion",
        "frame_layout": "root_pos_expmap_model_dof",
        "mimickit_motion_joint_layout": str(motion_joint_layout),
    }
    if robot_xml is not None:
        payload["mjcf_path"] = str(robot_xml)
    if source_path is not None:
        payload["source_motion_path"] = str(source_path)
    if metadata:
        payload.update(metadata)
    return payload


def save_mimickit_motion(output_path, frames, fps, **kwargs):
    payload = build_mimickit_payload(frames, fps, **kwargs)
    for key, value in payload.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f"NPZ metadata {key!r} contains an object value")
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("AMASS motions must use the portable .npz format")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(output_path)


def load_mimickit_motion(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}
