from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_PHC_TRUNCATABLE_ISSUES = frozenset({"airborne", "sitting"})


@dataclass(frozen=True)
class PHCFilterDecision:
    action: str
    issue: str
    end_frame: int = -1
    message: str = ""


def discover_input_files(input_root: Path) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob("*.npz")
        if path.is_file()
        and (path.name.endswith("_poses.npz") or path.name.endswith("_stageii.npz"))
    )


def filter_input_files(
    files: list[Path],
    input_root: Path,
    include_patterns: list[str] | None,
) -> list[Path]:
    if not include_patterns:
        return files
    selected: list[Path] = []
    for path in files:
        relative = path.relative_to(input_root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in include_patterns):
            selected.append(path)
    return selected


def build_phc_occlusion_key(
    input_path: Path,
    input_root: Path,
    *,
    prefix: str = "0",
) -> str:
    relative = input_path.relative_to(input_root).with_suffix("")
    return f"{prefix}-{'_'.join(relative.parts)}"


def _load_phc_occlusion_npz(path: Path) -> dict[str, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        keys = np.asarray(data["keys"], dtype=str)
        issues = np.asarray(data["issues"], dtype=str)
        idx_offsets = np.asarray(data["idx_offsets"], dtype=np.int64)
        idx_values = np.asarray(data["idx_values"], dtype=np.int64)

    if len(keys) != len(issues) or len(idx_offsets) != len(keys) + 1:
        raise ValueError(
            "invalid PHC occlusion npz shapes: "
            f"keys={keys.shape}, issues={issues.shape}, offsets={idx_offsets.shape}"
        )

    metadata: dict[str, dict[str, Any]] = {}
    for index, key in enumerate(keys):
        start = int(idx_offsets[index])
        end = int(idx_offsets[index + 1])
        metadata[str(key)] = {
            "issue": str(issues[index]),
            "idxes": [int(value) for value in idx_values[start:end]],
        }
    return metadata


def save_phc_occlusion_metadata_npz(metadata: Mapping[str, Any], path: Path) -> None:
    keys: list[str] = []
    issues: list[str] = []
    idx_offsets = [0]
    idx_values: list[int] = []
    for key in sorted(metadata):
        entry = metadata[key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"PHC metadata entry is not a mapping: {key}")
        keys.append(str(key))
        issues.append(_phc_issue(entry))
        idx_values.extend(_phc_idxes(entry))
        idx_offsets.append(len(idx_values))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        keys=np.asarray(keys, dtype=str),
        issues=np.asarray(issues, dtype=str),
        idx_offsets=np.asarray(idx_offsets, dtype=np.int64),
        idx_values=np.asarray(idx_values, dtype=np.int64),
    )


def _phc_issue(entry: Mapping[str, Any]) -> str:
    issue = entry.get("issue", "unknown")
    if hasattr(issue, "item"):
        issue = issue.item()
    if isinstance(issue, bytes):
        issue = issue.decode("utf-8", errors="replace")
    return str(issue).strip().lower() or "unknown"


def _phc_idxes(entry: Mapping[str, Any]) -> list[int]:
    idxes = entry.get("idxes", [])
    if idxes is None:
        return []
    if hasattr(idxes, "tolist"):
        idxes = idxes.tolist()
    if isinstance(idxes, (int, np.integer)):
        return [int(idxes)]
    if isinstance(idxes, float):
        return [int(idxes)]
    return sorted(int(index) for index in idxes if int(index) >= 0)


def decide_phc_filter_action(
    *,
    key: str,
    entry: Mapping[str, Any] | None,
    source_frames: int,
    source_fps: float = 30.0,
    min_frames: int = 10,
    missing_action: str = "keep",
) -> PHCFilterDecision:
    if missing_action not in {"keep", "skip"}:
        raise ValueError(f"invalid PHC missing action: {missing_action}")

    if entry is None:
        if missing_action == "skip":
            return PHCFilterDecision(
                action="skip",
                issue="missing",
                message=f"missing PHC occlusion metadata for {key}",
            )
        return PHCFilterDecision(action="keep", issue="missing")

    issue = _phc_issue(entry)
    if issue == "normal":
        return PHCFilterDecision(action="keep", issue=issue)

    if issue in _PHC_TRUNCATABLE_ISSUES:
        idxes = _phc_idxes(entry)
        if not idxes:
            return PHCFilterDecision(
                action="skip",
                issue=issue,
                message=f"PHC issue {issue} has no truncation idxes",
            )
        frame_scale = max(1, int(round(source_fps / 30.0)))
        end_frame = min(idxes[0] * frame_scale, source_frames)
        if end_frame >= source_frames:
            return PHCFilterDecision(
                action="keep",
                issue=issue,
                message=(
                    f"PHC first bad frame {idxes[0]} at 30Hz is outside source clip"
                ),
            )
        if end_frame < min_frames:
            return PHCFilterDecision(
                action="skip",
                issue=issue,
                end_frame=end_frame,
                message=(
                    f"PHC truncation leaves {end_frame} frames, below "
                    f"min_frames={min_frames}"
                ),
            )
        return PHCFilterDecision(action="truncate", issue=issue, end_frame=end_frame)

    return PHCFilterDecision(
        action="skip",
        issue=issue,
        message=f"PHC issue is not usable for flat-floor training: {issue}",
    )


def _read_input_metadata(input_path: Path) -> tuple[int, float]:
    with np.load(input_path, allow_pickle=False) as data:
        if "poses" not in data or "trans" not in data:
            raise ValueError("missing required AMASS keys: poses/trans")
        poses = data["poses"]
        trans = data["trans"]
        if poses.ndim != 2 or trans.ndim != 2 or trans.shape[1] != 3:
            raise ValueError(
                f"invalid AMASS array shapes: poses={poses.shape}, trans={trans.shape}"
            )
        if poses.shape[0] != trans.shape[0]:
            raise ValueError(
                f"frame count mismatch: poses={poses.shape[0]}, trans={trans.shape[0]}"
            )
        fps_value = data.get("mocap_framerate", data.get("fps", 0.0))
        if hasattr(fps_value, "item"):
            fps_value = fps_value.item()
        return int(poses.shape[0]), float(fps_value)


def load_phc_occlusion_metadata(path):
    if Path(path).suffix != ".npz":
        raise ValueError("Convert trusted PHC metadata to NPZ before preparing motions")
    return _load_phc_occlusion_npz(Path(path))
