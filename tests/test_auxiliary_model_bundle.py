"""Auxiliary physics models survive snapshotting and exact canonical renames."""

import shutil
from pathlib import Path

import mujoco
import pytest
from omegaconf import OmegaConf

from instantmimic.play import resolver
from instantmimic.train.repro_bundle import _save_training_repro_bundle

OLD_PD = "data/snu_humanoid/mjcf/snu_humanoid_ball_feet_pd.xml"
NEW_PD = "data/snu_humanoid/mjcf/humanoid_pd.xml"


def test_auxiliary_snapshot_resolves_after_original_model_directory_is_deleted(
    tmp_path,
):
    source = tmp_path / "models"
    source.mkdir()
    (source / "body.xml").write_text(
        '<mujocoinclude><worldbody><body><joint name="hinge"/>'
        '<geom type="sphere" size=".1"/></body></worldbody></mujocoinclude>'
    )
    (source / "simulation.xml").write_text(
        '<mujoco><include file="body.xml"/></mujoco>'
    )
    (source / "controller.xml").write_text(
        '<mujoco><include file="body.xml"/><actuator>'
        '<position joint="hinge" kp="4"/></actuator></mujoco>'
    )
    cfg = OmegaConf.create(
        {
            "env": {
                "sim": {"mjcf_path": str(source / "simulation.xml")},
                "action": {
                    "policy_output": {
                        "stable_pd": {"mjcf_path": str(source / "controller.xml")}
                    }
                },
            }
        }
    )
    run = tmp_path / "run"
    _save_training_repro_bundle(cfg, run, resolved_seed=7)
    snapshot_path = resolver.bundle_config_path(run)
    snapshot = OmegaConf.load(snapshot_path)
    shutil.rmtree(source)
    resolved = resolver.resolve_artifact_paths(
        snapshot, run, bundle_cfg_path=snapshot_path
    )
    simulation = mujoco.MjModel.from_xml_path(resolved.env.sim.mjcf_path)
    controller = mujoco.MjModel.from_xml_path(
        resolved.env.action.policy_output.stable_pd.mjcf_path
    )
    assert simulation.nv == controller.nv == 1
    assert simulation.nu == 0
    assert controller.nu == 1
    assert controller.actuator_gainprm[0, 0] == 4
    assert Path(resolved.env.action.policy_output.stable_pd.mjcf_path).is_relative_to(
        run
    )


def test_exact_retired_auxiliary_name_migrates_only_when_original_is_missing(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(
        resolver, "resolve_project_relative_path", lambda p: str(tmp_path / p)
    )
    canonical = tmp_path / NEW_PD
    canonical.parent.mkdir(parents=True)
    canonical.write_text("<mujoco/>")
    actual = resolver._resolve_auxiliary_model_for_play(OLD_PD, tmp_path / "run")
    assert actual == str(canonical)
    assert "no snapshot" in caplog.text
    original = tmp_path / OLD_PD
    original.write_text("<mujoco/>")
    assert resolver._resolve_auxiliary_model_for_play(OLD_PD, tmp_path / "run") == str(
        original
    )


def test_auxiliary_bundle_snapshot_wins_over_existing_checkout_model(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        resolver, "resolve_project_relative_path", lambda p: str(tmp_path / p)
    )
    checkout = tmp_path / OLD_PD
    checkout.parent.mkdir(parents=True)
    checkout.write_text("<mujoco/>")
    run = tmp_path / "run"
    bundled = run / "run_bundle" / "mjcf_tree" / OLD_PD
    bundled.parent.mkdir(parents=True)
    bundled.write_text("<mujoco/>")
    assert resolver._resolve_auxiliary_model_for_play(OLD_PD, run) == str(bundled)


@pytest.mark.parametrize(
    "name",
    [
        OLD_PD,
        "model.xml",
        "other/humanoid_pd.xml",
        "data/snu_humanoid/mjcf/variant_pd.xml",
    ],
)
def test_auxiliary_missing_input_is_an_error_without_existing_exact_replacement(
    tmp_path, monkeypatch, name
):
    monkeypatch.setattr(
        resolver, "resolve_project_relative_path", lambda p: str(tmp_path / p)
    )
    with pytest.raises(FileNotFoundError, match="auxiliary model is missing"):
        resolver._resolve_auxiliary_model_for_play(name, tmp_path / "run")
