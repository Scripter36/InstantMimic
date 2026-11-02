from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Collection, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from instantmimic.motion.bvh_mapping import BvhMapValue, normalize_bvh_map

MOTION_CACHE_VERSION = 2
_MOTION_CACHE_WRITE_LOCK_SUFFIX = ".lock"
_MOTION_CACHE_LOCK_POLL_INTERVAL_SEC = 0.1
_MOTION_CACHE_LOCK_STALE_SEC = 3600.0
_MOTION_CACHE_LOCK_WAIT_TIMEOUT_SEC = 300.0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _hash_jsonable(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_body_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        for body_id in range(model.nbody)
    ]


def _model_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        or f"joint_{joint_id}"
        for joint_id in range(model.njnt)
    ]


def model_signature(
    model: mujoco.MjModel, *, exclude_geoms: Collection[str] = ()
) -> str:
    """Identify the compiled inputs to motion mapping, FK, COM and grounding.

    Recompute from the small reference-producing arrays because MjModel is
    mutable. Visual mesh/texture buffers are deliberately excluded: grounding
    uses the compiled geom bounds, not the original mesh vertices. This avoids
    serializing large visual assets once for every clip in a dataset.
    ``exclude_geoms`` names geoms to leave out, so a model that keeps
    physics-neutral visual geoms signs like the model without them.
    """
    excluded = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(name))
        for name in exclude_geoms
    }
    if -1 in excluded:
        raise ValueError("model_signature exclude_geoms names a missing geom")
    geom_keep = np.ones(int(model.ngeom), dtype=bool)
    geom_keep[sorted(excluded)] = False
    payload = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "body_names": _model_body_names(model),
        "joint_names": _model_joint_names(model),
        "body_jntadr": np.asarray(model.body_jntadr, dtype=np.int32),
        "body_jntnum": np.asarray(model.body_jntnum, dtype=np.int32),
        "jnt_type": np.asarray(model.jnt_type, dtype=np.int32),
        "jnt_bodyid": np.asarray(model.jnt_bodyid, dtype=np.int32),
        "jnt_qposadr": np.asarray(model.jnt_qposadr, dtype=np.int32),
        "jnt_dofadr": np.asarray(model.jnt_dofadr, dtype=np.int32),
        "jnt_limited": np.asarray(model.jnt_limited, dtype=np.int32),
        "jnt_axis": np.asarray(model.jnt_axis),
        "jnt_range": np.asarray(model.jnt_range),
    }
    for name in (
        "qpos0",
        "body_parentid",
        "body_rootid",
        "body_weldid",
        "body_mocapid",
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "body_mass",
        "body_subtreemass",
        "body_inertia",
        "body_dofadr",
        "body_dofnum",
        "body_simple",
        "body_sameframe",
        "dof_bodyid",
        "dof_jntid",
        "dof_parentid",
        "jnt_pos",
        "geom_type",
        "geom_bodyid",
        "geom_sameframe",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_rbound",
        "geom_contype",
        "geom_conaffinity",
    ):
        value = np.asarray(getattr(model, name))
        payload[name] = value[geom_keep] if name.startswith("geom_") else value
    return _hash_jsonable(payload)


def build_motion_cache_metadata(
    *,
    motion_file: Path,
    motion_scale: float,
    motion_offset: list[float],
    bvh_map: dict[str, BvhMapValue],
    target_dt: float,
    model: mujoco.MjModel,
    mimickit_joint_groups: list[dict[str, object]] | None,
    loop_boundary_vel_from_first_frame: bool,
    loop_boundary_vel_extrapolate: bool,
    smooth_cycle: bool,
    smooth_cycle_blend_frames: int,
    smooth_cycle_mode: str = "tail_warp",
    root_auto_calibration: bool = False,
) -> dict[str, Any]:
    stat = motion_file.stat()
    return {
        "cache_version": MOTION_CACHE_VERSION,
        "motion_file": {
            "path": str(motion_file),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "motion_scale": float(motion_scale),
        "motion_offset": [float(v) for v in motion_offset],
        "bvh_map": dict(sorted(normalize_bvh_map(bvh_map).items())),
        "target_dt": float(target_dt),
        "model_signature": model_signature(model),
        # Normalized so an empty list and None cannot key two cache entries for
        # the same clip. A format that ignores this option can only ever carry
        # its default now, since `load_bvh_file` refuses the combination, so no
        # ignored option splits the cache.
        "mimickit_joint_groups": mimickit_joint_groups or None,
        "loop_boundary_vel_from_first_frame": bool(loop_boundary_vel_from_first_frame),
        "loop_boundary_vel_extrapolate": bool(loop_boundary_vel_extrapolate),
        "smooth_cycle": bool(smooth_cycle),
        "smooth_cycle_blend_frames": int(smooth_cycle_blend_frames),
        "smooth_cycle_mode": str(smooth_cycle_mode),
        "root_auto_calibration": bool(root_auto_calibration),
    }


def motion_cache_file_path(*, cache_dir: Path, metadata: dict[str, Any]) -> Path:
    model_sig = str(metadata["model_signature"])
    motion_info = metadata["motion_file"]
    motion_path = Path(str(motion_info["path"]))
    clip_sig = _hash_jsonable(metadata)
    clip_stem = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in motion_path.stem
    )
    return (
        cache_dir
        / f"v{MOTION_CACHE_VERSION}"
        / model_sig
        / f"{clip_stem}-{clip_sig[:16]}.pt"
    )


def payload_to_cpu(payload: dict[str, Any]) -> dict[str, Any]:
    cpu_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            cpu_payload[key] = value.detach().to(device="cpu").contiguous()
        else:
            cpu_payload[key] = value
    return cpu_payload


def payload_to_device(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    device_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            device_payload[key] = value.to(device=device)
        else:
            device_payload[key] = value
    return device_payload


def load_motion_cache(cache_file: Path, *, device: torch.device) -> dict[str, Any]:
    cache_blob = torch.load(cache_file, map_location="cpu", weights_only=False)
    if not isinstance(cache_blob, dict):
        raise ValueError(f"Invalid motion cache payload in {cache_file}")
    if int(cache_blob.get("cache_version", -1)) != MOTION_CACHE_VERSION:
        raise ValueError(f"Unsupported motion cache version in {cache_file}")
    payload = cache_blob.get("payload", None)
    if not isinstance(payload, dict):
        raise ValueError(f"Motion cache missing payload in {cache_file}")
    return payload_to_device(payload, device)


def save_motion_cache(
    cache_file: Path,
    *,
    metadata: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_blob = {
        "cache_version": MOTION_CACHE_VERSION,
        "metadata": metadata,
        "payload": payload_to_cpu(payload),
    }
    fd, tmp_path_str = tempfile.mkstemp(
        dir=cache_file.parent,
        prefix=f"{cache_file.stem}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        torch.save(cache_blob, tmp_path)
        os.replace(tmp_path, cache_file)
    finally:
        with suppress(FileNotFoundError):
            tmp_path.unlink()


@contextmanager
def acquire_motion_cache_write_lock(cache_file: Path) -> Iterator[bool]:
    lock_file = cache_file.with_suffix(
        cache_file.suffix + _MOTION_CACHE_WRITE_LOCK_SUFFIX
    )
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + _MOTION_CACHE_LOCK_WAIT_TIMEOUT_SEC
    owns_lock = False
    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            owns_lock = True
            break
        except FileExistsError:
            try:
                age_sec = time.time() - lock_file.stat().st_mtime
            except FileNotFoundError:
                continue
            if age_sec > _MOTION_CACHE_LOCK_STALE_SEC:
                with suppress(FileNotFoundError):
                    lock_file.unlink()
                continue
            if cache_file.exists() or time.monotonic() >= deadline:
                break
            time.sleep(_MOTION_CACHE_LOCK_POLL_INTERVAL_SEC)

    try:
        yield owns_lock
    finally:
        if owns_lock:
            with suppress(FileNotFoundError):
                lock_file.unlink()
