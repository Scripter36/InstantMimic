#!/usr/bin/env python3
"""Apply a PHC AMASS filter report to FK-retargeted humanoid motion files."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .motion import (
    mimickit_frames_to_qpos,
    qpos_to_mimickit_frames,
)

DEFAULT_HEIGHT_OFFSET = -0.02


@dataclass(frozen=True)
class PostFilterResult:
    status: str
    source_path: str
    output_path: str
    frames: int = 0
    fps: float = 0.0
    message: str = ""
    phc_action: str = ""
    phc_issue: str = ""
    phc_end_frame: int = -1


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _load_report(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_report(path: Path, rows: Iterable[PostFilterResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def _target_indices(
    frame_count: int, source_fps: float, target_fps: float
) -> np.ndarray:
    if target_fps <= 0.0 or target_fps >= source_fps:
        return np.arange(frame_count, dtype=np.int64)
    duration = (frame_count - 1) / source_fps
    times = np.arange(0.0, duration + 0.5 / target_fps, 1.0 / target_fps)
    indices = np.rint(times * source_fps).astype(np.int64)
    return np.unique(np.clip(indices, 0, frame_count - 1))


def _filtered_frame_count(
    *,
    source_frames: int,
    source_fps: float,
    retarget_fps: float,
    phc_end_frame: int,
) -> int:
    if phc_end_frame < 0:
        return -1
    indices = _target_indices(
        int(source_frames),
        float(source_fps),
        float(retarget_fps),
    )
    return int(np.count_nonzero(indices < int(phc_end_frame)))


def _payload_frame_count(payload: dict[str, Any]) -> tuple[int, float]:
    frames = np.asarray(payload.get("frames", []))
    fps = float(payload.get("fps", 0.0))
    if frames.ndim != 2 or frames.shape[0] < 2 or fps <= 0.0:
        raise ValueError(
            f"invalid MimicKit pkl payload: frames={frames.shape} fps={fps}"
        )
    return int(frames.shape[0]), fps


def _load_motion_payload(path):
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _save_motion_payload(path, payload):
    if "frames_sha256" in payload:
        payload["frames_sha256"] = hashlib.sha256(
            np.asarray(payload["frames"]).tobytes()
        ).hexdigest()
    for key, value in payload.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError(f"NPZ metadata {key!r} contains an object value")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()


def _payload_scalar_str(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0].item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip()


def _resolve_robot_xml_for_payload(
    payload: dict[str, Any],
    *,
    robot_xml: Path | None,
) -> Path | None:
    if robot_xml is not None:
        return _resolve(robot_xml)
    raw_mjcf_path = payload.get("mjcf_path")
    if raw_mjcf_path is None:
        return None
    mjcf_text = _payload_scalar_str(raw_mjcf_path)
    if not mjcf_text:
        return None
    return _resolve(Path(mjcf_text))


def _height_renormalize_truncated_payload(
    payload: dict[str, Any],
    *,
    robot_xml: Path | None,
    height_offset: float,
) -> bool:
    robot_xml_path = _resolve_robot_xml_for_payload(payload, robot_xml=robot_xml)
    if robot_xml_path is None:
        return False

    import mujoco

    from .fk import (
        _normalize_motion_height,
    )

    model = mujoco.MjModel.from_xml_path(str(robot_xml_path))
    qpos = mimickit_frames_to_qpos(np.asarray(payload["frames"]), model)
    qpos = _normalize_motion_height(qpos, model, height_offset=float(height_offset))
    payload["frames"] = qpos_to_mimickit_frames(qpos, model).astype(
        np.float32,
        copy=False,
    )
    payload["height_renormalized_after_phc"] = np.asarray(True)
    payload["height_renormalize_height_offset"] = np.asarray(
        float(height_offset), dtype=np.float32
    )
    payload["height_renormalize_mjcf_path"] = str(robot_xml_path)
    return True


def _copy_or_link(
    *,
    source_path: Path,
    output_path: Path,
    link_mode: str,
    overwrite: bool,
) -> PostFilterResult:
    if output_path.exists() or output_path.is_symlink():
        if not overwrite:
            payload = _load_motion_payload(source_path)
            frames, fps = _payload_frame_count(payload)
            return PostFilterResult(
                status="skipped",
                source_path=str(source_path),
                output_path=str(output_path),
                frames=frames,
                fps=fps,
                phc_action="keep",
            )
        _remove_existing(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        output_path.symlink_to(
            os.path.relpath(source_path.resolve(), output_path.parent.resolve())
        )
        status = "linked"
    elif link_mode == "hardlink":
        os.link(source_path, output_path)
        status = "linked"
    elif link_mode == "copy":
        shutil.copy2(source_path, output_path)
        status = "copied"
    else:
        raise ValueError(f"unsupported link mode: {link_mode}")

    payload = _load_motion_payload(source_path)
    frames, fps = _payload_frame_count(payload)
    return PostFilterResult(
        status=status,
        source_path=str(source_path),
        output_path=str(output_path),
        frames=frames,
        fps=fps,
        phc_action="keep",
    )


def _truncate_motion(
    *,
    source_path: Path,
    output_path: Path,
    report_row: dict[str, Any],
    overwrite: bool,
    min_output_frames: int,
    height_renormalize_truncated: bool,
    height_offset: float,
    robot_xml: Path | None,
) -> PostFilterResult:
    payload = _load_motion_payload(source_path)

    frame_count, fps = _payload_frame_count(payload)
    keep_count = _filtered_frame_count(
        source_frames=int(report_row.get("source_frames", 0)),
        source_fps=float(report_row.get("source_fps", 0.0)),
        retarget_fps=fps,
        phc_end_frame=int(report_row.get("phc_end_frame", -1)),
    )
    keep_count = min(max(0, keep_count), frame_count)
    if keep_count < int(min_output_frames):
        if (output_path.exists() or output_path.is_symlink()) and overwrite:
            _remove_existing(output_path)
        return PostFilterResult(
            status="filtered",
            source_path=str(source_path),
            output_path=str(output_path),
            message=f"PHC truncation leaves {keep_count} retargeted frames",
            phc_action="truncate",
            phc_issue=str(report_row.get("phc_issue", "")),
            phc_end_frame=int(report_row.get("phc_end_frame", -1)),
        )

    if output_path.exists() or output_path.is_symlink():
        if not overwrite:
            return PostFilterResult(
                status="skipped",
                source_path=str(source_path),
                output_path=str(output_path),
                frames=keep_count,
                fps=fps,
                phc_action="truncate",
                phc_issue=str(report_row.get("phc_issue", "")),
                phc_end_frame=int(report_row.get("phc_end_frame", -1)),
            )
        _remove_existing(output_path)

    output_payload = dict(payload)
    for key, value in list(output_payload.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (frame_count,):
            output_payload[key] = value[:keep_count].copy()
    output_payload["frames"] = np.asarray(payload["frames"])[:keep_count].astype(
        np.float32,
        copy=True,
    )
    height_renormalized = False
    if height_renormalize_truncated:
        height_renormalized = _height_renormalize_truncated_payload(
            output_payload,
            robot_xml=robot_xml,
            height_offset=float(height_offset),
        )
    output_payload.update(
        {
            "phc_action": "truncate",
            "phc_issue": str(report_row.get("phc_issue", "")),
            "phc_end_frame": int(report_row.get("phc_end_frame", -1)),
            "phc_source_frames": int(report_row.get("source_frames", 0)),
            "phc_source_fps": float(report_row.get("source_fps", 0.0)),
            "source_retargeted_motion_path": str(source_path),
            "phc_truncate_height_renormalized": np.asarray(height_renormalized),
        }
    )

    _save_motion_payload(output_path, output_payload)

    return PostFilterResult(
        status="truncated",
        source_path=str(source_path),
        output_path=str(output_path),
        frames=keep_count,
        fps=fps,
        phc_action="truncate",
        phc_issue=str(report_row.get("phc_issue", "")),
        phc_end_frame=int(report_row.get("phc_end_frame", -1)),
    )


def _relative_output_path(row: dict[str, Any], reference_root: Path) -> Path:
    relative_path = row.get("relative_path")
    if isinstance(relative_path, str) and relative_path.strip():
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Report paths must stay inside the motion root")
        return relative
    output_path = Path(str(row.get("output_path", ""))).expanduser().resolve()
    return output_path.relative_to(reference_root)


def _apply_one(
    row: dict[str, Any],
    *,
    source_root: Path,
    reference_root: Path,
    output_root: Path,
    link_mode: str,
    overwrite: bool,
    min_output_frames: int,
    height_renormalize_truncated: bool,
    height_offset: float,
    robot_xml: Path | None,
) -> PostFilterResult:
    try:
        relative = _relative_output_path(row, reference_root)
    except Exception as exc:  # noqa: BLE001 - CSV row boundary reports path resolution failure.
        return PostFilterResult(
            status="failed",
            source_path="",
            output_path="",
            message=f"could not resolve row output_path relative to reference root: {exc}",
            phc_action=str(row.get("phc_action", "")),
            phc_issue=str(row.get("phc_issue", "")),
            phc_end_frame=int(row.get("phc_end_frame", -1)),
        )

    source_path = source_root / relative
    output_path = output_root / relative
    action = str(row.get("phc_action", "") or "")
    issue = str(row.get("phc_issue", "") or "")

    if row.get("status") not in {"converted", "skipped"} or action == "skip":
        if (output_path.exists() or output_path.is_symlink()) and overwrite:
            _remove_existing(output_path)
        return PostFilterResult(
            status="filtered",
            source_path=str(source_path),
            output_path=str(output_path),
            phc_action=action,
            phc_issue=issue,
            phc_end_frame=int(row.get("phc_end_frame", -1)),
        )

    if not source_path.is_file():
        return PostFilterResult(
            status="missing",
            source_path=str(source_path),
            output_path=str(output_path),
            message="source GMR pkl does not exist",
            phc_action=action,
            phc_issue=issue,
            phc_end_frame=int(row.get("phc_end_frame", -1)),
        )

    try:
        if action == "truncate":
            return _truncate_motion(
                source_path=source_path,
                output_path=output_path,
                report_row=row,
                overwrite=overwrite,
                min_output_frames=min_output_frames,
                height_renormalize_truncated=height_renormalize_truncated,
                height_offset=height_offset,
                robot_xml=robot_xml,
            )
        result = _copy_or_link(
            source_path=source_path,
            output_path=output_path,
            link_mode=link_mode,
            overwrite=overwrite,
        )
        return PostFilterResult(
            **{
                **asdict(result),
                "phc_action": action or "keep",
                "phc_issue": issue,
                "phc_end_frame": int(row.get("phc_end_frame", -1)),
            }
        )
    except Exception as exc:  # noqa: BLE001 - batch item boundary returns a failed result row.
        return PostFilterResult(
            status="failed",
            source_path=str(source_path),
            output_path=str(output_path),
            message=str(exc),
            phc_action=action,
            phc_issue=issue,
            phc_end_frame=int(row.get("phc_end_frame", -1)),
        )


def apply_phc_filter_report(
    *,
    phc_report: Path,
    source_root: Path,
    reference_root: Path,
    output_root: Path,
    link_mode: str = "symlink",
    overwrite: bool = False,
    min_output_frames: int = 2,
    height_renormalize_truncated: bool = True,
    height_offset: float = DEFAULT_HEIGHT_OFFSET,
    robot_xml: Path | None = None,
) -> list[PostFilterResult]:
    phc_report = _resolve(phc_report)
    source_root = _resolve(source_root)
    reference_root = _resolve(reference_root)
    output_root = _resolve(output_root)

    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("Filter output must be outside the source motion directory")
    rows = _load_report(phc_report)
    sources = {}
    for row in rows:
        relative = _relative_output_path(row, reference_root=reference_root)
        source = source_root / relative
        if source.is_file():
            sources[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    identity = {
        "report_sha256": hashlib.sha256(phc_report.read_bytes()).hexdigest(),
        "source_root": str(source_root),
        "reference_root": str(reference_root),
        "sources": sources,
        "link_mode": link_mode,
        "min_output_frames": min_output_frames,
        "height_renormalize_truncated": bool(height_renormalize_truncated),
        "height_offset": float(height_offset),
        "robot_xml": str(robot_xml) if robot_xml is not None else None,
    }
    if robot_xml is not None:
        from .model_identity import model_file_sha256

        identity["model_sha256"] = model_file_sha256(robot_xml)
    from .model_identity import model_file_sha256

    implicit_models = {}
    if robot_xml is None and height_renormalize_truncated:
        for row in rows:
            if row.get("phc_action") != "truncate":
                continue
            source = source_root / _relative_output_path(row, reference_root)
            if source.is_file():
                selected = _resolve_robot_xml_for_payload(
                    _load_motion_payload(source), robot_xml=None
                )
                if selected is not None:
                    implicit_models[str(selected)] = model_file_sha256(selected)
    identity["implicit_models"] = implicit_models
    manifest = output_root / ".amass-filter.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError(
                "Filter inputs or settings changed; choose a fresh output directory"
            )
    elif output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            "Filter output is not managed; choose an empty output directory"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)
    results = [
        _apply_one(
            row,
            source_root=source_root,
            reference_root=reference_root,
            output_root=output_root,
            link_mode=link_mode,
            overwrite=True,
            min_output_frames=min_output_frames,
            height_renormalize_truncated=bool(height_renormalize_truncated),
            height_offset=float(height_offset),
            robot_xml=robot_xml,
        )
        for row in rows
    ]
    if not any(result.status == "failed" for result in results):
        output_root.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return results
