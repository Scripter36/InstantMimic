from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from instantmimic_amass.phc import save_phc_occlusion_metadata_npz
from instantmimic_amass.prepare import prepare_dataset


@pytest.fixture
def dataset(tmp_path):
    model = tmp_path / "robot.xml"
    model.write_text(
        '<mujoco><worldbody><geom type="plane" size="0 0 .1"/>'
        '<body name="pelvis"><freejoint/><geom type="sphere" size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    raw = tmp_path / "raw"
    for collection, name in [
        ("KIT", "keep"),
        ("KIT", "trim"),
        ("KIT", "stairs"),
        ("HumanEva", "validation"),
    ]:
        source = raw / collection / "10" / (name + "_poses.npz")
        source.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            source,
            poses=np.zeros((120, 156)),
            trans=np.zeros((120, 3)),
            mocap_framerate=120.0,
        )
    metadata = tmp_path / "metadata.npz"
    save_phc_occlusion_metadata_npz(
        {
            "0-KIT_10_keep_poses": {"issue": "normal"},
            "0-KIT_10_trim_poses": {"issue": "airborne", "idxes": [15]},
            "0-KIT_10_stairs_poses": {"issue": "stairs"},
            "0-HumanEva_10_validation_poses": {"issue": "normal"},
        },
        metadata,
    )
    return dict(
        raw_root=raw, model=model, metadata=metadata, output=tmp_path / "filtered"
    )


def test_original_data_to_filtered_npz_roundtrip_and_resume(dataset):
    summary = prepare_dataset(**dataset)
    assert summary["clips_scanned"] == 3
    assert summary["clips_written"] == 2
    assert summary["frames"] == 46
    output = dataset["output"]
    files = sorted(output.rglob("*.npz"))
    assert len(files) == 2
    with np.load(files[0], allow_pickle=False) as payload:
        for key in payload.files:
            assert not payload[key].dtype.hasobject
        assert payload["loop_mode"] == 0
    files[0].write_bytes(b"broken")
    again = prepare_dataset(**dataset)
    assert again == summary
    with np.load(files[0], allow_pickle=False) as payload:
        assert payload["frames"].shape[1] == 6
    with pytest.raises(ValueError, match="changed"):
        prepare_dataset(**dataset, height_offset=0.1)


def test_prepare_cli_works_outside_checkout(dataset, tmp_path):
    args = [sys.executable, "-m", "instantmimic_amass.prepare"]
    for key, value in dataset.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    result = subprocess.run(args, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["clips_written"] == 2


def test_metadata_key_mismatch_does_not_silently_disable_filter(dataset):
    save_phc_occlusion_metadata_npz(
        {"wrong-key": {"issue": "normal"}}, dataset["metadata"]
    )
    with pytest.raises(ValueError, match="No PHC metadata keys"):
        prepare_dataset(**dataset)
    assert not dataset["output"].exists()


def test_invalid_clip_cannot_fake_metadata_coverage(dataset):
    source = dataset["raw_root"] / "KIT/10/invalid_poses.npz"
    np.savez(
        source, poses=np.zeros((1, 156)), trans=np.zeros((1, 3)), mocap_framerate=120.0
    )
    save_phc_occlusion_metadata_npz(
        {"wrong-key": {"issue": "normal"}}, dataset["metadata"]
    )
    with pytest.raises(ValueError, match="No PHC metadata keys"):
        prepare_dataset(**dataset)


def test_unannotated_subset_with_valid_collection_metadata_is_supported(dataset):
    save_phc_occlusion_metadata_npz(
        {"0-KIT_another_problem_clip": {"issue": "stairs"}}, dataset["metadata"]
    )
    summary = prepare_dataset(**dataset)
    assert summary["collections"]["KIT"] == {"unannotated": 3}


def test_model_snapshot_is_scoped_to_each_prepare_job(dataset):
    first = prepare_dataset(**dataset)
    dataset["model"].write_text(
        dataset["model"].read_text().replace('size=".1"', 'size=".2"')
    )
    second = prepare_dataset(
        **{**dataset, "output": dataset["output"].with_name("changed_model")}
    )
    assert first["model_sha256"] != second["model_sha256"]
    with (
        np.load(dataset["output"] / "KIT/10/keep_poses.npz") as a,
        np.load(
            dataset["output"].with_name("changed_model") / "KIT/10/keep_poses.npz"
        ) as b,
    ):
        np.testing.assert_allclose(b["frames"][:, 2] - a["frames"][:, 2], 0.1)


@pytest.mark.parametrize(
    "field, value", [("fps", 300.0), ("loop_mode", 1), ("frame_layout", "wrong")]
)
def test_resume_rebuilds_cache_with_invalid_motion_contract(dataset, field, value):
    prepare_dataset(**dataset)
    cache = (
        dataset["output"].with_name(dataset["output"].name + ".fk")
        / "KIT/10/keep_poses.npz"
    )
    with np.load(cache, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload[field] = value
    np.savez(cache, **payload)
    prepare_dataset(**dataset)
    with np.load(
        dataset["output"] / "KIT/10/keep_poses.npz", allow_pickle=False
    ) as data:
        assert float(data["fps"]) == 30.0
        assert int(data["loop_mode"]) == 0
        assert str(data["frame_layout"]) == "root_pos_expmap_model_dof"


def test_too_short_downsampled_clip_is_filtered_without_aborting_dataset(dataset):
    source = dataset["raw_root"] / "KIT/10/short_poses.npz"
    np.savez(
        source, poses=np.zeros((2, 156)), trans=np.zeros((2, 3)), mocap_framerate=120.0
    )
    summary = prepare_dataset(**dataset)
    assert summary["clips_scanned"] == 4
    assert summary["clips_written"] == 2
    report = dataset["output"].with_name(dataset["output"].name + ".phc.jsonl")
    short = next(
        json.loads(line)
        for line in report.read_text().splitlines()
        if "short_poses" in line
    )
    assert short["status"] == "filtered"
    assert "two frames" in short["message"]
