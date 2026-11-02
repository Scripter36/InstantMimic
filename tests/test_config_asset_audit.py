"""Missing runtime inputs require explicit declarations and explicit deferral."""

import json

import pytest

from instantmimic.tools import validate_configs as audit


def test_runtime_asset_deferral_is_explicit_and_exact(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = "data/robot/model.xml"
    declarations = {path: "prepare the robot"}
    with pytest.raises(ValueError, match="missing.*prepare"):
        audit._checkout_asset(path, declarations=declarations)
    deferred = set()
    audit._checkout_asset(path, declarations=declarations, deferred=deferred)
    assert deferred == {path}
    with pytest.raises(ValueError, match="missing"):
        audit._checkout_asset(
            "data/robot/typo.xml", declarations=declarations, deferred=deferred
        )
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text("<mujoco/>")
    deferred.clear()
    audit._checkout_asset(path, declarations=declarations, deferred=deferred)
    assert not deferred


@pytest.mark.parametrize(
    "path",
    [
        "../elsewhere/model.xml",
        "/absolute/model.xml",
        "data/../model.xml",
        "data\\model.xml",
    ],
)
def test_runtime_declarations_reject_nonportable_paths(tmp_path, monkeypatch, path):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/runtime-assets.json").write_text(
        json.dumps({"schema": 1, "files": {path: "prepare"}})
    )
    with pytest.raises(ValueError, match="checkout-relative"):
        audit._runtime_asset_declarations()


def test_declaration_never_allows_a_symlink_to_escape_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "data").symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(audit, "ROOT", root)
    with pytest.raises(ValueError, match="outside"):
        audit._checkout_asset(
            "data/model.xml", declarations={"data/model.xml": "prepare"}, deferred=set()
        )


def test_motion_file_is_checked_separately_from_model(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    model = tmp_path / "data/robot/model.xml"
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco/>")
    cfg = OmegaConf.create(
        {
            "env": {
                "sim": {"mjcf_path": "data/robot/model.xml"},
                "motion": {"file": "data/robot/clip.npz"},
            }
        }
    )
    with pytest.raises(ValueError, match="clip.npz"):
        audit._validate_assets(cfg, declarations={}, deferred=set())


def _write_unavailable_task(root):
    config = root / "config/robot/walk.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "schema_version: 1\nenv:\n  action: {type: musubi}\n"
        "  sim: {mjcf_path: data/robot/model.xml}\n"
    )
    return config


def test_unavailable_method_passes_composition_and_asset_validation(
    tmp_path, monkeypatch
):
    _write_unavailable_task(tmp_path)
    model = tmp_path / "data/robot/model.xml"
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco/>")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    report = audit.audit()
    assert report["status"] == "passed"
    assert not report["failures"]
    assert report["configurations"] == ["robot/walk"]


def test_unavailable_method_does_not_hide_a_missing_asset(tmp_path, monkeypatch):
    _write_unavailable_task(tmp_path)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    report = audit.audit()
    assert report["status"] == "failed"
    assert "model.xml" in report["failures"][0]["message"]


def test_unrelated_abstract_error_is_not_an_availability_exemption(
    tmp_path, monkeypatch
):
    _write_unavailable_task(tmp_path)
    model = tmp_path / "data/robot/model.xml"
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco/>")
    monkeypatch.setattr(audit, "ROOT", tmp_path)

    def unexpected(_cfg):
        raise NotImplementedError("unrelated abstract method")

    monkeypatch.setattr(audit, "load_semantic_config", unexpected)
    report = audit.audit()
    assert report["status"] == "failed"
    assert report["failures"][0]["error_type"] == "NotImplementedError"


def test_pattern_asset_deferral_requires_the_exact_authored_pattern(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    pattern = "data/robot/clips/*.npz"
    deferred = set()
    audit._checkout_asset(
        pattern,
        pattern=True,
        declarations={pattern: "prepare clips"},
        deferred=deferred,
    )
    assert deferred == {pattern}
    with pytest.raises(ValueError, match="missing"):
        audit._checkout_asset(
            "data/robot/other/*.npz",
            pattern=True,
            declarations={pattern: "prepare clips"},
            deferred=deferred,
        )


def test_motion_library_requires_every_named_array(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    monkeypatch.setattr(audit, "ROOT", tmp_path)
    cfg = OmegaConf.create(
        {"env": {"motion_data_dir": "data/bird/mocap", "motion_clips": ["walk"]}}
    )
    for array in ("qpos", "qvel", "xipos"):
        path = tmp_path / f"data/bird/mocap/{array}/walk.npy"
        path.parent.mkdir(parents=True)
        path.touch()
    with pytest.raises(ValueError, match="ximat/walk.npy"):
        audit._validate_assets(cfg, declarations={}, deferred=set())


def test_recursive_motion_pattern_finds_nested_collection_subject(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    motion = tmp_path / "data/motions/KIT/10/clip.npz"
    motion.parent.mkdir(parents=True)
    motion.write_bytes(b"fixture")
    audit._checkout_asset("data/motions/**/*.npz", pattern=True)
    with pytest.raises(ValueError, match="missing"):
        audit._checkout_asset("data/motions/**/*.pkl", pattern=True)
