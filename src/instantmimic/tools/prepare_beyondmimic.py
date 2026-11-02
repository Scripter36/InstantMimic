"""Prepare the pinned mjlab G1 model and public BeyondMimic side kick.

Run this script with the virtual environment of the pinned mjlab checkout. It
exports the fully edited mjlab tracking scene (not merely its source XML),
verifies that the exported scene recompiles to the same model signature, safely
expands PBHC's public 23-DoF side kick to the documented 29-DoF G1 layout, and
builds the exact array contract consumed by mjlab's tracking task.

Existing output is never overwritten. Repeating the command is accepted only
when the newly staged tree is byte-identical to the existing tree.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib.numpy_pickle
import mujoco
import numpy as np

MJLAB_COMMIT = "15ebce8840ee2205f4c62d0c20df65dd1794cab4"
PUBLIC_SIDE_KICK_SHA256 = (
    "3152b3b53537e426bba57c5c3e04bd5dff74b00a38ee3a33a9c8dc1de7228821"
)
PUBLIC_SIDE_KICK_FRAMES = 179
INPUT_FPS = 30
OUTPUT_FPS = 50

SOURCE_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

G1_JOINT_NAMES: tuple[str, ...] = (
    *SOURCE_JOINT_NAMES[:19],
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    *SOURCE_JOINT_NAMES[19:],
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_ALLOWED_JOBLIB_GLOBALS: dict[tuple[str, str], Any] = {
    (
        "joblib.numpy_pickle",
        "NumpyArrayWrapper",
    ): joblib.numpy_pickle.NumpyArrayWrapper,
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
}


class _RestrictedNumpyUnpickler(joblib.numpy_pickle.NumpyUnpickler):
    def find_class(self, module: str, name: str) -> Any:
        try:
            return _ALLOWED_JOBLIB_GLOBALS[(module, name)]
        except KeyError as exc:
            raise ValueError(
                f"Rejected pickle global {module}.{name}; "
                f"allowed={tuple(sorted(_ALLOWED_JOBLIB_GLOBALS))}"
            ) from exc


def restricted_joblib_load(path: Path) -> Any:
    """Load only NumPy-only joblib payloads through an exact global allowlist."""
    with path.open("rb") as handle:
        return _RestrictedNumpyUnpickler(
            str(path),
            handle,
            ensure_native_byte_order=True,
            mmap_mode=None,
        ).load()


def _validate_numpy_only(value: Any, path: str = "root") -> None:
    if isinstance(value, np.ndarray | str | bool | int | float | np.generic):
        return
    if value is None:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} has non-string key {key!r}")
            _validate_numpy_only(child, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains rejected type {type(value).__name__}")


def _array(value: Any, *, field: str, width: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(
            f"{field} must have shape (frames, {width}); got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{field} must contain finite numeric values")
    return array


def expand_public_side_kick(motion: Mapping[str, Any]) -> np.ndarray:
    """Expand PBHC's 23 coordinates to xyz + xyzw + 29 G1 coordinates."""
    if int(motion.get("fps", -1)) != INPUT_FPS:
        raise ValueError(f"Public side kick must be 30 Hz; got {motion.get('fps')!r}")
    root = _array(motion.get("root_trans_offset"), field="root_trans_offset", width=3)
    quat = _array(motion.get("root_rot"), field="root_rot", width=4)
    dof = _array(motion.get("dof"), field="dof (expected 23 coordinates)", width=23)
    if not (len(root) == len(quat) == len(dof)):
        raise ValueError("root_trans_offset, root_rot, and dof frame counts differ")

    quat_norm = np.linalg.norm(quat.astype(np.float64), axis=1)
    if not np.allclose(quat_norm, 1.0, atol=2.0e-7, rtol=0.0):
        raise ValueError("root_rot is not normalized within atol=2e-7")

    source = dof.astype(np.float32, copy=False)
    wrist_zeros = np.zeros((len(source), 3), dtype=np.float32)
    target = np.concatenate(
        (source[:, :19], wrist_zeros, source[:, 19:], wrist_zeros),
        axis=1,
    )
    output = np.concatenate(
        (
            root.astype(np.float32, copy=False),
            quat.astype(np.float32, copy=False),
            target,
        ),
        axis=1,
    )
    if output.shape != (len(root), 36):
        raise AssertionError(f"Unexpected expanded motion shape {output.shape}")
    return np.ascontiguousarray(output)


def write_float32_csv(path: Path, values: np.ndarray) -> None:
    """Write a deterministic CSV that round-trips every float32 exactly."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("CSV values must be a finite 2D float32 array")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, array, delimiter=",", fmt="%.9g")
    restored = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if not np.array_equal(restored, array):
        raise AssertionError("CSV serialization did not round-trip float32 exactly")


def deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an uncompressed NPZ with fixed member metadata and stable bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name, value in arrays.items():
            if not name or "/" in name or name.endswith(".npy"):
                raise ValueError(f"Invalid NPZ member name {name!r}")
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asanyarray(value),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, index: int) -> str:
    value = mujoco.mj_id2name(model, object_type, index)
    return "" if value is None else value


def _floats(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _dof_slice(model: mujoco.MjModel, joint_id: int) -> slice:
    start = int(model.jnt_dofadr[joint_id])
    stop = (
        int(model.jnt_dofadr[joint_id + 1]) if joint_id + 1 < model.njnt else model.nv
    )
    return slice(start, stop)


def model_signature(model: mujoco.MjModel) -> dict[str, Any]:
    """Return the compiled fields used to verify portable model export."""
    options = {
        "timestep": float(model.opt.timestep),
        "integrator": int(model.opt.integrator),
        "cone": int(model.opt.cone),
        "jacobian": int(model.opt.jacobian),
        "solver": int(model.opt.solver),
        "iterations": int(model.opt.iterations),
        "tolerance": float(model.opt.tolerance),
        "ls_iterations": int(model.opt.ls_iterations),
        "ls_tolerance": float(model.opt.ls_tolerance),
        "ccd_iterations": int(model.opt.ccd_iterations),
        "ccd_tolerance": float(model.opt.ccd_tolerance),
        "impratio": float(model.opt.impratio),
        "gravity": _floats(model.opt.gravity),
        "wind": _floats(model.opt.wind),
        "magnetic": _floats(model.opt.magnetic),
        "density": float(model.opt.density),
        "viscosity": float(model.opt.viscosity),
        "disableflags": int(model.opt.disableflags),
        "enableflags": int(model.opt.enableflags),
    }
    bodies = [
        {
            "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, i),
            "parent_id": int(model.body_parentid[i]),
            "mocap_id": int(model.body_mocapid[i]),
            "pos": _floats(model.body_pos[i]),
            "quat": _floats(model.body_quat[i]),
            "mass": float(model.body_mass[i]),
            "inertia": _floats(model.body_inertia[i]),
            "inertial_pos": _floats(model.body_ipos[i]),
            "inertial_quat": _floats(model.body_iquat[i]),
        }
        for i in range(model.nbody)
    ]
    joints = []
    for i in range(model.njnt):
        dofs = _dof_slice(model, i)
        joints.append(
            {
                "name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                "type": int(model.jnt_type[i]),
                "body_id": int(model.jnt_bodyid[i]),
                "qpos_address": int(model.jnt_qposadr[i]),
                "dof_address": int(model.jnt_dofadr[i]),
                "group": int(model.jnt_group[i]),
                "limited": int(model.jnt_limited[i]),
                "actuator_force_limited": int(model.jnt_actfrclimited[i]),
                "pos": _floats(model.jnt_pos[i]),
                "axis": _floats(model.jnt_axis[i]),
                "range": _floats(model.jnt_range[i]),
                "actuator_force_range": _floats(model.jnt_actfrcrange[i]),
                "margin": float(model.jnt_margin[i]),
                "solref": _floats(model.jnt_solref[i]),
                "solimp": _floats(model.jnt_solimp[i]),
                "armature": _floats(model.dof_armature[dofs]),
                "damping": _floats(model.dof_damping[dofs]),
                "frictionloss": _floats(model.dof_frictionloss[dofs]),
            }
        )
    geoms = [
        {
            "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, i),
            "body_id": int(model.geom_bodyid[i]),
            "type": int(model.geom_type[i]),
            "data_id": int(model.geom_dataid[i]),
            "group": int(model.geom_group[i]),
            "contype": int(model.geom_contype[i]),
            "conaffinity": int(model.geom_conaffinity[i]),
            "condim": int(model.geom_condim[i]),
            "priority": int(model.geom_priority[i]),
            "size": _floats(model.geom_size[i]),
            "pos": _floats(model.geom_pos[i]),
            "quat": _floats(model.geom_quat[i]),
            "friction": _floats(model.geom_friction[i]),
            "margin": float(model.geom_margin[i]),
            "gap": float(model.geom_gap[i]),
            "solmix": float(model.geom_solmix[i]),
            "solref": _floats(model.geom_solref[i]),
            "solimp": _floats(model.geom_solimp[i]),
        }
        for i in range(model.ngeom)
    ]
    actuators = [
        {
            "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            "transmission_type": int(model.actuator_trntype[i]),
            "transmission_id": np.asarray(
                model.actuator_trnid[i], dtype=np.int64
            ).tolist(),
            "dynamics_type": int(model.actuator_dyntype[i]),
            "gain_type": int(model.actuator_gaintype[i]),
            "bias_type": int(model.actuator_biastype[i]),
            "control_limited": int(model.actuator_ctrllimited[i]),
            "force_limited": int(model.actuator_forcelimited[i]),
            "ctrlrange": _floats(model.actuator_ctrlrange[i]),
            "forcerange": _floats(model.actuator_forcerange[i]),
            "gear": _floats(model.actuator_gear[i]),
            "gainprm": _floats(model.actuator_gainprm[i]),
            "biasprm": _floats(model.actuator_biasprm[i]),
            "dynprm": _floats(model.actuator_dynprm[i]),
        }
        for i in range(model.nu)
    ]
    return {
        "dimensions": {
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "nbody": model.nbody,
            "njnt": model.njnt,
            "ngeom": model.ngeom,
            "nmesh": model.nmesh,
        },
        "options": options,
        "bodies": bodies,
        "joints": joints,
        "geoms": geoms,
        "actuators": actuators,
        "meshes": [
            _name(model, mujoco.mjtObj.mjOBJ_MESH, i) for i in range(model.nmesh)
        ],
    }


def compare_model_signatures(
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    rtol: float,
) -> list[str]:
    """Return path-qualified exact/discrete or tolerance-bounded mismatches."""
    mismatches: list[str] = []

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict):
            if not isinstance(right, dict):
                mismatches.append(f"{path}: expected dict, got {type(right).__name__}")
                return
            if set(left) != set(right):
                mismatches.append(
                    f"{path}: keys differ: {sorted(left)!r} != {sorted(right)!r}"
                )
                return
            for key in left:
                compare(left[key], right[key], f"{path}.{key}" if path else key)
            return
        if isinstance(left, list):
            if not isinstance(right, list) or len(left) != len(right):
                mismatches.append(
                    f"{path}: list shape differs: {len(left)} != "
                    f"{len(right) if isinstance(right, list) else 'non-list'}"
                )
                return
            for index, (left_item, right_item) in enumerate(
                zip(left, right, strict=True)
            ):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left, float):
            if not isinstance(right, int | float) or not np.isclose(
                left,
                right,
                atol=atol,
                rtol=rtol,
            ):
                mismatches.append(f"{path}: {left!r} != {right!r}")
            return
        if left != right:
            mismatches.append(f"{path}: {left!r} != {right!r}")

    compare(reference, candidate, "")
    return mismatches


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _git_commit(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("Preparation requires a clean source checkout")
    return result.stdout.strip()


def _one_motion(payload: Any) -> tuple[str, Mapping[str, Any]]:
    _validate_numpy_only(payload)
    if not isinstance(payload, dict) or len(payload) != 1:
        raise ValueError("Expected exactly one top-level motion")
    name, motion = next(iter(payload.items()))
    if not isinstance(name, str) or not isinstance(motion, dict):
        raise TypeError("Expected one string-keyed motion dictionary")
    return name, motion


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _format_numbers(values: Any) -> str:
    return " ".join(f"{float(value):.17g}" for value in np.ravel(values))


def _write_precise_runtime_xml(
    approximate_xml: Path,
    output_xml: Path,
    model: mujoco.MjModel,
) -> None:
    """Replace lossy MjSpec-exported fields with compiled 17-digit values."""
    tree = ET.parse(approximate_xml)
    root = tree.getroot()

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        insert_at = next(
            (
                index
                for index, element in enumerate(root)
                if element.tag in {"visual", "statistic", "default", "asset"}
            ),
            0,
        )
        root.insert(insert_at, option)
    integrators = {
        int(mujoco.mjtIntegrator.mjINT_EULER): "Euler",
        int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST): "implicitfast",
    }
    cones = {
        int(mujoco.mjtCone.mjCONE_PYRAMIDAL): "pyramidal",
        int(mujoco.mjtCone.mjCONE_ELLIPTIC): "elliptic",
    }
    jacobians = {
        int(mujoco.mjtJacobian.mjJAC_AUTO): "auto",
        int(mujoco.mjtJacobian.mjJAC_DENSE): "dense",
        int(mujoco.mjtJacobian.mjJAC_SPARSE): "sparse",
    }
    solvers = {
        int(mujoco.mjtSolver.mjSOL_PGS): "PGS",
        int(mujoco.mjtSolver.mjSOL_CG): "CG",
        int(mujoco.mjtSolver.mjSOL_NEWTON): "Newton",
    }
    option.attrib.update(
        {
            "timestep": f"{float(model.opt.timestep):.17g}",
            "integrator": integrators[int(model.opt.integrator)],
            "cone": cones[int(model.opt.cone)],
            "jacobian": jacobians[int(model.opt.jacobian)],
            "solver": solvers[int(model.opt.solver)],
            "iterations": str(int(model.opt.iterations)),
            "tolerance": f"{float(model.opt.tolerance):.17g}",
            "ls_iterations": str(int(model.opt.ls_iterations)),
            "ls_tolerance": f"{float(model.opt.ls_tolerance):.17g}",
            "ccd_iterations": str(int(model.opt.ccd_iterations)),
            "ccd_tolerance": f"{float(model.opt.ccd_tolerance):.17g}",
            "impratio": f"{float(model.opt.impratio):.17g}",
            "gravity": _format_numbers(model.opt.gravity),
            "wind": _format_numbers(model.opt.wind),
            "magnetic": _format_numbers(model.opt.magnetic),
            "density": f"{float(model.opt.density):.17g}",
            "viscosity": f"{float(model.opt.viscosity):.17g}",
        }
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Exported scene XML has no worldbody")
    body_elements = list(worldbody.iter("body"))
    if len(body_elements) != model.nbody - 1:
        raise ValueError(
            f"Body count mismatch: XML={len(body_elements)}, model={model.nbody - 1}"
        )
    for body_id, element in enumerate(body_elements, start=1):
        element.set("pos", _format_numbers(model.body_pos[body_id]))
        element.set("quat", _format_numbers(model.body_quat[body_id]))
        inertial = element.find("inertial")
        if inertial is not None:
            inertial.set("pos", _format_numbers(model.body_ipos[body_id]))
            inertial.set("quat", _format_numbers(model.body_iquat[body_id]))
            inertial.set("mass", f"{float(model.body_mass[body_id]):.17g}")
            inertial.set(
                "diaginertia",
                _format_numbers(model.body_inertia[body_id]),
            )

    joint_elements = list(worldbody.iter("joint"))
    if len(joint_elements) != model.njnt:
        raise ValueError(
            f"Joint count mismatch: XML={len(joint_elements)}, model={model.njnt}"
        )
    for joint_id, element in enumerate(joint_elements):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        dofs = _dof_slice(model, joint_id)
        element.set("pos", _format_numbers(model.jnt_pos[joint_id]))
        element.set("axis", _format_numbers(model.jnt_axis[joint_id]))
        element.set("range", _format_numbers(model.jnt_range[joint_id]))
        element.set("margin", f"{float(model.jnt_margin[joint_id]):.17g}")
        armature = np.asarray(model.dof_armature[dofs])
        damping = np.asarray(model.dof_damping[dofs])
        frictionloss = np.asarray(model.dof_frictionloss[dofs])
        if not (armature.size == damping.size == frictionloss.size == 1):
            raise ValueError(f"Expected scalar hinge DoF fields for joint {joint_id}")
        element.set("armature", f"{float(armature[0]):.17g}")
        element.set("damping", f"{float(damping[0]):.17g}")
        element.set("frictionloss", f"{float(frictionloss[0]):.17g}")
        if bool(model.jnt_actfrclimited[joint_id]):
            element.set(
                "actfrcrange",
                _format_numbers(model.jnt_actfrcrange[joint_id]),
            )

    geom_elements = list(worldbody.iter("geom"))
    if len(geom_elements) != model.ngeom:
        raise ValueError(
            f"Geom count mismatch: XML={len(geom_elements)}, model={model.ngeom}"
        )
    for geom_id, element in enumerate(geom_elements):
        is_mesh = int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
        if not is_mesh:
            element.set("size", _format_numbers(model.geom_size[geom_id]))
            element.set("pos", _format_numbers(model.geom_pos[geom_id]))
            element.set("quat", _format_numbers(model.geom_quat[geom_id]))
        element.set("friction", _format_numbers(model.geom_friction[geom_id]))
        element.set("margin", f"{float(model.geom_margin[geom_id]):.17g}")
        element.set("gap", f"{float(model.geom_gap[geom_id]):.17g}")
        element.set("solmix", f"{float(model.geom_solmix[geom_id]):.17g}")
        element.set("solref", _format_numbers(model.geom_solref[geom_id]))
        element.set("solimp", _format_numbers(model.geom_solimp[geom_id]))
        element.set("group", str(int(model.geom_group[geom_id])))
        element.set("contype", str(int(model.geom_contype[geom_id])))
        element.set("conaffinity", str(int(model.geom_conaffinity[geom_id])))
        element.set("condim", str(int(model.geom_condim[geom_id])))
        element.set("priority", str(int(model.geom_priority[geom_id])))

    actuator_parent = root.find("actuator")
    actuator_elements = [] if actuator_parent is None else list(actuator_parent)
    if len(actuator_elements) != model.nu:
        raise ValueError(
            f"Actuator count mismatch: XML={len(actuator_elements)}, model={model.nu}"
        )
    for actuator_id, element in enumerate(actuator_elements):
        element.set(
            "ctrlrange",
            _format_numbers(model.actuator_ctrlrange[actuator_id]),
        )
        element.set(
            "forcerange",
            _format_numbers(model.actuator_forcerange[actuator_id]),
        )
        element.set("gear", _format_numbers(model.actuator_gear[actuator_id]))
        element.set("gainprm", _format_numbers(model.actuator_gainprm[actuator_id]))
        element.set("biasprm", _format_numbers(model.actuator_biasprm[actuator_id]))
        element.set("dynprm", _format_numbers(model.actuator_dynprm[actuator_id]))

    key_parent = root.find("keyframe")
    key_elements = [] if key_parent is None else list(key_parent)
    if len(key_elements) != model.nkey:
        raise ValueError(
            f"Keyframe count mismatch: XML={len(key_elements)}, model={model.nkey}"
        )
    for key_id, element in enumerate(key_elements):
        element.set("qpos", _format_numbers(model.key_qpos[key_id]))
        element.set("ctrl", _format_numbers(model.key_ctrl[key_id]))

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="unicode", xml_declaration=False)
    with output_xml.open("a", encoding="utf-8") as handle:
        handle.write("\n")


def _generate_motion_npz(scene: Any, csv_path: Path, output_path: Path) -> None:
    import torch
    from mjlab.scripts.csv_to_npz import MotionLoader
    from mjlab.sim.sim import Simulation, SimulationCfg

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / OUTPUT_FPS
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device="cpu")
    scene.initialize(sim.mj_model, sim.model, sim.data)
    scene.reset()

    motion = MotionLoader(
        motion_file=str(csv_path),
        input_fps=INPUT_FPS,
        output_fps=OUTPUT_FPS,
        device=sim.device,
    )
    robot = scene["robot"]
    joint_ids = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]
    logged: dict[str, list[np.ndarray]] = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    for _ in range(motion.output_frames):
        (
            (
                root_pos,
                root_quat,
                root_lin_vel,
                root_ang_vel,
                joint_pos_input,
                joint_vel_input,
            ),
            _,
        ) = motion.get_next_state()
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0:3] = root_pos
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = root_quat
        root_state[:, 7:10] = root_lin_vel
        root_state[:, 10:] = root_ang_vel
        robot.write_root_state_to_sim(root_state)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_ids] = joint_pos_input
        joint_vel[:, joint_ids] = joint_vel_input
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        logged["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        logged["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        logged["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
        logged["body_quat_w"].append(
            robot.data.body_link_quat_w[0].cpu().numpy().copy()
        )
        logged["body_lin_vel_w"].append(
            robot.data.body_link_lin_vel_w[0].cpu().numpy().copy()
        )
        logged["body_ang_vel_w"].append(
            robot.data.body_link_ang_vel_w[0].cpu().numpy().copy()
        )
        torch.testing.assert_close(
            robot.data.body_link_lin_vel_w[0, 0], root_lin_vel[0]
        )
        torch.testing.assert_close(
            robot.data.body_link_ang_vel_w[0, 0], root_ang_vel[0]
        )

    arrays: dict[str, np.ndarray] = {"fps": np.array([OUTPUT_FPS], dtype=np.int64)}
    arrays.update({name: np.stack(values) for name, values in logged.items()})
    deterministic_npz(output_path, arrays)


def _prepare_staged(
    staged: Path,
    *,
    mjlab_source: Path,
    side_kick_pkl: Path,
) -> None:
    actual_commit = _git_commit(mjlab_source)
    if actual_commit != MJLAB_COMMIT:
        raise RuntimeError(
            f"mjlab checkout must be {MJLAB_COMMIT}; got {actual_commit}"
        )
    actual_side_kick_hash = _sha256(side_kick_pkl)
    if actual_side_kick_hash != PUBLIC_SIDE_KICK_SHA256:
        raise RuntimeError(
            "Public side-kick hash mismatch: "
            f"{actual_side_kick_hash} != {PUBLIC_SIDE_KICK_SHA256}"
        )

    import mjlab
    from mjlab.scene import Scene
    from mjlab.tasks.tracking.config.g1.env_cfgs import (
        unitree_g1_flat_tracking_env_cfg,
    )

    imported_mjlab = Path(mjlab.__file__).resolve()
    expected_package = (mjlab_source / "src/mjlab").resolve()
    if not imported_mjlab.is_relative_to(expected_package):
        raise RuntimeError(
            f"Imported mjlab from {imported_mjlab}, expected {expected_package}. "
            "Run with the pinned checkout's virtual environment."
        )
    tracking_cfg = unitree_g1_flat_tracking_env_cfg()
    scene = Scene(tracking_cfg.scene, device="cpu")
    reference_model = scene.compile()
    tracking_cfg.sim.mujoco.apply(reference_model)
    reference_signature = model_signature(reference_model)

    mjcf_dir = staged / "mjcf"
    scene.write(mjcf_dir)
    source_assets = mjlab_source / "src/mjlab/asset_zoo/robots/unitree_g1/xmls/assets"
    shutil.copytree(source_assets, mjcf_dir / "assets")
    shutil.copy2(mjlab_source / "LICENSE", staged / "MJLAB_LICENSE")

    readable_xml_path = mjcf_dir / "scene.xml"
    _write_precise_runtime_xml(
        readable_xml_path,
        readable_xml_path,
        reference_model,
    )
    portable_model = mujoco.MjModel.from_xml_path(str(readable_xml_path))
    portable_signature = model_signature(portable_model)
    mismatches = compare_model_signatures(
        reference_signature,
        portable_signature,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    if mismatches:
        raise RuntimeError(
            "Precise runtime XML differs from the pinned mjlab scene:\n"
            + "\n".join(mismatches[:50])
        )

    binary_model_path = mjcf_dir / "scene-reference-mujoco-3.10.0.mjb"
    mujoco.mj_saveModel(reference_model, str(binary_model_path))
    exported_model = mujoco.MjModel.from_binary_path(str(binary_model_path))
    exported_signature = model_signature(exported_model)
    binary_mismatches = compare_model_signatures(
        reference_signature,
        exported_signature,
        atol=0.0,
        rtol=0.0,
    )
    if binary_mismatches:
        raise RuntimeError(
            "Binary model differs from the pinned mjlab scene:\n"
            + "\n".join(binary_mismatches[:50])
        )
    _write_json(staged / "model-signature.json", portable_signature)

    motion_name, motion = _one_motion(restricted_joblib_load(side_kick_pkl))
    public_csv = expand_public_side_kick(motion)
    if len(public_csv) != PUBLIC_SIDE_KICK_FRAMES:
        raise RuntimeError(
            f"Public side kick must have {PUBLIC_SIDE_KICK_FRAMES} frames; "
            f"got {len(public_csv)}"
        )
    csv_path = staged / "motions/source/side_kick.csv"
    write_float32_csv(csv_path, public_csv)
    npz_path = staged / "motions/converted/side_kick_motion.npz"
    _generate_motion_npz(scene, csv_path, npz_path)

    with np.load(npz_path) as motion_npz:
        motion_arrays = {
            name: {
                "shape": list(motion_npz[name].shape),
                "dtype": str(motion_npz[name].dtype),
                "content_sha256": hashlib.sha256(
                    np.ascontiguousarray(motion_npz[name]).tobytes()
                ).hexdigest(),
            }
            for name in motion_npz.files
        }
    if (
        _git_commit(mjlab_source) != actual_commit
        or _sha256(side_kick_pkl) != actual_side_kick_hash
    ):
        raise RuntimeError("Preparation inputs changed during generation")
    manifest = {
        "mjlab": {
            "url": "https://github.com/mujocolab/mjlab.git",
            "commit": actual_commit,
            "license": "Apache-2.0",
            "source_g1_xml_sha256": _sha256(
                mjlab_source / "src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
            ),
        },
        "public_side_kick": {
            "url": "https://github.com/TeleHuman/PBHC.git",
            "path": "example/motion_data/Side_kick.pkl",
            "sha256": actual_side_kick_hash,
            "motion_key": motion_name,
            "license": "CC-BY-NC-4.0",
            "input_fps": INPUT_FPS,
            "input_frames": len(public_csv),
            "mapping": {
                "target_0_18": "source_0_18",
                "target_19_21": "zero left wrist roll/pitch/yaw",
                "target_22_25": "source_19_22",
                "target_26_28": "zero right wrist roll/pitch/yaw",
            },
            "root_translation": "preserved",
            "root_quaternion": "preserved xyzw; validated, not modified",
        },
        "model": {
            "scene": "mjcf/scene.xml",
            "reference_binary_mujoco_3_10_0": (
                "mjcf/scene-reference-mujoco-3.10.0.mjb"
            ),
            "signature": "model-signature.json",
            "portable_xml_comparison_atol": 1.0e-12,
            "portable_xml_comparison_rtol": 1.0e-12,
            "reference_vs_portable_xml_mismatch_count": 0,
            "reference_vs_binary_mismatch_count": 0,
            "warp_nconmax": tracking_cfg.sim.nconmax,
            "warp_njmax": tracking_cfg.sim.njmax,
        },
        "motion": {
            "source_csv": "motions/source/side_kick.csv",
            "source_csv_sha256": _sha256(csv_path),
            "converted_npz": "motions/converted/side_kick_motion.npz",
            "converted_npz_sha256": _sha256(npz_path),
            "output_fps": OUTPUT_FPS,
            "arrays": motion_arrays,
        },
        "reference_artifact_limitation": (
            "No byte- or array-equality claim is made against the inaccessible "
            "external motion registry artifact."
        ),
    }
    _write_json(staged / "manifest.json", manifest)


def prepare(
    *,
    mjlab_source: Path,
    side_kick_pkl: Path,
    output_dir: Path,
) -> str:
    """Prepare output atomically, or verify that existing output is identical."""
    mjlab_source = mjlab_source.resolve()
    side_kick_pkl = side_kick_pkl.resolve()
    output_dir = output_dir.resolve()
    if not mjlab_source.is_dir():
        raise FileNotFoundError(f"mjlab source does not exist: {mjlab_source}")
    if not side_kick_pkl.is_file():
        raise FileNotFoundError(f"side-kick source does not exist: {side_kick_pkl}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".prepare-beyondmimic-",
        dir=output_dir.parent,
    ) as temporary:
        staged = Path(temporary) / "prepared"
        staged.mkdir()
        _prepare_staged(
            staged,
            mjlab_source=mjlab_source,
            side_kick_pkl=side_kick_pkl,
        )
        if output_dir.exists():
            existing_hashes = _tree_hashes(output_dir)
            staged_hashes = _tree_hashes(staged)
            if existing_hashes != staged_hashes:
                differing = sorted(set(existing_hashes) | set(staged_hashes))
                details = [
                    f"{path}: existing={existing_hashes.get(path)} "
                    f"staged={staged_hashes.get(path)}"
                    for path in differing
                    if existing_hashes.get(path) != staged_hashes.get(path)
                ]
                raise FileExistsError(
                    f"Existing output differs: {output_dir}\n" + "\n".join(details[:50])
                )
            return "unchanged"
        staged.replace(output_dir)
    return "created"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjlab-source", required=True, type=Path)
    parser.add_argument("--side-kick-pkl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    status = prepare(
        mjlab_source=args.mjlab_source,
        side_kick_pkl=args.side_kick_pkl,
        output_dir=args.output_dir,
    )
    print(f"{status}: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
