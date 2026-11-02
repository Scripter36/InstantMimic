from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .phc import (
    _read_input_metadata,
    build_phc_occlusion_key,
    decide_phc_filter_action,
    discover_input_files,
    filter_input_files,
    load_phc_occlusion_metadata,
)

PHC_AMASS_SPLITS: dict[str, frozenset[str]] = {
    "train": frozenset(
        {
            "CMU",
            "MPI_Limits",
            "TotalCapture",
            "KIT",
            "EKUT",
            "TCD_handMocap",
            "BMLhandball",
            "DanceDB",
            "ACCAD",
            "BMLmovi",
            "BioMotionLab_NTroje",
            "Eyes_Japan_Dataset",
            "DFaust_67",
        }
    ),
    "val": frozenset({"HumanEva", "MPI_HDM05", "SFU", "MPI_mosh"}),
    "test": frozenset({"Transitions_mocap", "SSM_synced"}),
}


PHC_SPLIT_CHOICES = ("all", "train", "val", "vald", "test", "train_val_test")


@dataclass(frozen=True)
class AMASSPHCReportRow:
    status: str
    input_path: str
    output_path: str
    relative_path: str
    frames: int = 0
    fps: float = 0.0
    message: str = ""
    source_frames: int = 0
    source_fps: float = 0.0
    phc_key: str = ""
    phc_issue: str = ""
    phc_action: str = ""
    phc_end_frame: int = -1
    phc_split: str = ""


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _write_jsonl(path: Path, rows: Iterable[AMASSPHCReportRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def _phc_split_for_path(input_path: Path, input_root: Path) -> str:
    parts = input_path.relative_to(input_root).parts
    if not parts:
        return "outside"
    dataset = parts[0]
    for split, datasets in PHC_AMASS_SPLITS.items():
        if dataset in datasets:
            return split
    return "outside"


def _phc_split_datasets(phc_split: str) -> frozenset[str] | None:
    if phc_split == "all":
        return None
    if phc_split == "vald":
        phc_split = "val"
    if phc_split == "train_val_test":
        return frozenset().union(*PHC_AMASS_SPLITS.values())
    try:
        return PHC_AMASS_SPLITS[phc_split]
    except KeyError as exc:
        raise ValueError(f"invalid PHC AMASS split: {phc_split!r}") from exc


def _filter_phc_split(
    files: list[Path],
    input_root: Path,
    phc_split: str,
) -> list[Path]:
    datasets = _phc_split_datasets(phc_split)
    if datasets is None:
        return files
    selected: list[Path] = []
    for path in files:
        parts = path.relative_to(input_root).parts
        if parts and parts[0] in datasets:
            selected.append(path)
    return selected


def _build_one_row(
    input_path: Path,
    *,
    input_root: Path,
    reference_root: Path,
    phc_metadata: Mapping[str, Any],
    phc_key_prefix: str,
    phc_min_frames: int,
    phc_missing_action: str,
) -> AMASSPHCReportRow:
    relative_path = input_path.relative_to(input_root).with_suffix(".npz")
    phc_key = build_phc_occlusion_key(input_path, input_root, prefix=phc_key_prefix)
    source_frames, source_fps = _read_input_metadata(input_path)
    phc_split = _phc_split_for_path(input_path, input_root)
    if source_frames < 2:
        return AMASSPHCReportRow(
            status="filtered",
            input_path=str(input_path),
            output_path=str(reference_root / relative_path),
            relative_path=relative_path.as_posix(),
            frames=source_frames,
            fps=source_fps,
            message=f"clip has fewer than 2 frames: {source_frames}",
            source_frames=source_frames,
            source_fps=source_fps,
            phc_key=phc_key,
            phc_issue="invalid",
            phc_action="skip",
            phc_split=phc_split,
        )
    if source_fps <= 0.0:
        return AMASSPHCReportRow(
            status="filtered",
            input_path=str(input_path),
            output_path=str(reference_root / relative_path),
            relative_path=relative_path.as_posix(),
            frames=source_frames,
            fps=source_fps,
            message=f"invalid source fps: {source_fps}",
            source_frames=source_frames,
            source_fps=source_fps,
            phc_key=phc_key,
            phc_issue="invalid",
            phc_action="skip",
            phc_split=phc_split,
        )
    decision = decide_phc_filter_action(
        key=phc_key,
        entry=phc_metadata.get(phc_key),
        source_frames=source_frames,
        source_fps=source_fps,
        min_frames=phc_min_frames,
        missing_action=phc_missing_action,
    )
    status = "filtered" if decision.action == "skip" else "converted"
    return AMASSPHCReportRow(
        status=status,
        input_path=str(input_path),
        output_path=str(reference_root / relative_path),
        relative_path=relative_path.as_posix(),
        frames=source_frames,
        fps=source_fps,
        message=decision.message,
        source_frames=source_frames,
        source_fps=source_fps,
        phc_key=phc_key,
        phc_issue=decision.issue,
        phc_action=decision.action,
        phc_end_frame=decision.end_frame,
        phc_split=phc_split,
    )


def build_amass_phc_filter_report(
    *,
    input_root: Path,
    reference_root: Path,
    phc_occlusion_metadata: Path,
    include_patterns: list[str] | None = None,
    limit: int | None = None,
    phc_key_prefix: str = "0",
    phc_min_frames: int = 10,
    phc_missing_action: str = "keep",
    phc_split: str = "all",
) -> list[AMASSPHCReportRow]:
    input_root = _resolve(input_root)
    reference_root = _resolve(reference_root)
    phc_metadata = load_phc_occlusion_metadata(_resolve(phc_occlusion_metadata))
    files = filter_input_files(
        discover_input_files(input_root),
        input_root,
        include_patterns or [],
    )
    files = _filter_phc_split(files, input_root, phc_split)
    if limit is not None:
        files = files[: max(0, int(limit))]
    return [
        _build_one_row(
            path,
            input_root=input_root,
            reference_root=reference_root,
            phc_metadata=phc_metadata,
            phc_key_prefix=phc_key_prefix,
            phc_min_frames=phc_min_frames,
            phc_missing_action=phc_missing_action,
        )
        for path in files
    ]
