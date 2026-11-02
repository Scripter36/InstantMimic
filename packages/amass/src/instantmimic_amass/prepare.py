"""Prepare licensed local AMASS files as grounded, PHC-filtered humanoid motions."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .filter import apply_phc_filter_report
from .fk import (
    _load_amass_npz_metadata,
    _normalize_motion_height,
    _target_indices,
    load_amass_fk_qpos,
)
from .model_identity import compiled_model_sha256
from .motion import load_mimickit_motion, qpos_to_mimickit_frames, save_mimickit_motion
from .phc import load_phc_occlusion_metadata
from .report import PHC_AMASS_SPLITS, build_amass_phc_filter_report


@lru_cache(maxsize=4)
def _model(path):
    model = mujoco.MjModel.from_xml_path(path)
    return model, compiled_model_sha256(model)


def _convert(task):
    source, destination, model_path, fps, height_offset = task
    model, model_hash = _model(str(model_path))
    identity = json.dumps(
        {
            "version": 3,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "model_sha256": model_hash,
            "fps": fps,
            "height_offset": height_offset,
            "loop_mode": "clamp",
            "conversion": "reduced_skeleton_fk",
        },
        sort_keys=True,
    )
    source_frames, source_fps = _load_amass_npz_metadata(source)
    expected_frames = len(_target_indices(source_frames, source_fps, fps))
    expected_fps = min(fps, source_fps)
    if destination.exists():
        try:
            payload = load_mimickit_motion(destination)
            frames = payload["frames"]
            if (
                str(payload.get("conversion_identity", "")) == identity
                and frames.ndim == 2
                and frames.shape[0] == expected_frames
                and float(payload["fps"]) == expected_fps
                and int(payload["loop_mode"]) == 0
                and str(payload["loop_mode_encoding"]) == "mimickit_clamp0_wrap1"
                and str(payload["frame_layout"]) == "root_pos_expmap_model_dof"
                and str(payload["frames_sha256"])
                == hashlib.sha256(frames.tobytes()).hexdigest()
                and frames.shape[1] == model.nv
                and np.isfinite(frames).all()
            ):
                return
        except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
            pass
    qpos, actual_fps = load_amass_fk_qpos(source, robot_xml=model_path, target_fps=fps)
    qpos = _normalize_motion_height(qpos, model, height_offset=height_offset)
    frames = qpos_to_mimickit_frames(qpos, model)
    save_mimickit_motion(
        destination,
        frames,
        actual_fps,
        loop_mode="clamp",
        robot_xml=model_path,
        source_path=source,
        metadata={
            "conversion_identity": identity,
            "retargeting": "reduced_skeleton_fk",
            "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
        },
    )


def prepare_dataset(
    *,
    raw_root: Path,
    model: Path,
    metadata: Path,
    output: Path,
    split="train",
    fps=30.0,
    height_offset=-0.02,
    workers=1,
    limit=None,
):
    _model.cache_clear()  # Model snapshots live for one preparation job.
    raw_root, model, metadata, output = [
        Path(p).resolve() for p in (raw_root, model, metadata, output)
    ]
    if not raw_root.is_dir():
        raise ValueError(
            "Raw root must contain the original AMASS collection directories"
        )
    if (
        not np.isfinite(fps)
        or fps <= 0
        or not np.isfinite(height_offset)
        or workers < 1
    ):
        raise ValueError(
            "FPS and worker count must be positive; height offset must be finite"
        )
    scratch = output.with_name(output.name + ".fk")
    report_path = output.with_name(output.name + ".phc.jsonl")
    rows = build_amass_phc_filter_report(
        input_root=raw_root,
        reference_root=scratch,
        phc_occlusion_metadata=metadata,
        phc_split=split,
        limit=limit,
    )
    if not rows:
        raise ValueError(f"No AMASS motions found for split {split!r}")
    rows = [
        replace(
            row,
            status="filtered",
            phc_action="skip",
            message="Fewer than two frames remain at the target FPS",
        )
        if (
            row.phc_action != "skip"
            and len(_target_indices(row.source_frames, row.source_fps, fps)) < 2
        )
        else row
        for row in rows
    ]
    phc_metadata = load_phc_occlusion_metadata(metadata)
    coverage = defaultdict(Counter)
    for row in rows:
        collection = Path(row.relative_path).parts[0]
        coverage[collection][
            "annotated" if row.phc_key in phc_metadata else "unannotated"
        ] += 1
    matching_collection = any(
        key.startswith("0-" + collection + "_")
        for collection in coverage
        for key in phc_metadata
    )
    if not matching_collection:
        raise ValueError(
            "No PHC metadata keys match this dataset; check collection names and metadata"
        )
    tasks = [
        (Path(row.input_path), scratch / row.relative_path, model, fps, height_offset)
        for row in rows
        if row.phc_action != "skip"
    ]
    if workers == 1:
        for task in tasks:
            _convert(task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_convert, tasks))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows)
    )
    results = apply_phc_filter_report(
        phc_report=report_path,
        source_root=scratch,
        reference_root=scratch,
        output_root=output,
        link_mode="copy",
        overwrite=True,
        robot_xml=model,
        height_offset=height_offset,
    )
    failures = [asdict(row) for row in results if row.status == "failed"]
    if failures:
        raise RuntimeError(f"AMASS filtering failed: {failures}")
    frames = sum(row.frames for row in results if row.status != "filtered")
    duration = sum(
        (row.frames - 1) / row.fps
        for row in results
        if row.status != "filtered" and row.frames > 1 and row.fps > 0
    )
    model_obj, model_hash = _model(str(model))
    del model_obj
    summary = {
        "split": split,
        "collections": {k: dict(v) for k, v in coverage.items()},
        "clips_scanned": len(rows),
        "clips_written": sum(row.status != "filtered" for row in results),
        "frames": frames,
        "duration_seconds": duration,
        "target_fps": fps,
        "height_offset": height_offset,
        "model_sha256": model_hash,
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "phc_actions": dict(Counter(row.phc_action for row in rows)),
        "missing_annotations": "No listed occlusion; retained according to PHC policy",
    }
    output.with_name(output.name + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--metadata", required=True, type=Path, help="Converted PHC occlusion NPZ"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=tuple(PHC_AMASS_SPLITS), default="train")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--height-offset", type=float, default=-0.02)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = vars(parser.parse_args(argv))
    print(json.dumps(prepare_dataset(**args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
