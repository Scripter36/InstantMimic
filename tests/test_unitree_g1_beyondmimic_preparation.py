from __future__ import annotations

import subprocess
from pathlib import Path

import joblib
import mujoco
import numpy as np
import pytest

from instantmimic.tools.prepare_beyondmimic import (
    G1_JOINT_NAMES,
    _git_commit,
    compare_model_signatures,
    deterministic_npz,
    expand_public_side_kick,
    model_signature,
    restricted_joblib_load,
    write_float32_csv,
)


def _minimal_model(*, damping: float = 0.2) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        f"""
        <mujoco model="signature_fixture">
          <option timestep="0.005" iterations="10" ls_iterations="20"/>
          <compiler angle="radian" autolimits="true"/>
          <worldbody>
            <body name="pelvis" pos="0 0 0.8">
              <freejoint name="floating_base_joint"/>
              <geom name="pelvis_collision" type="sphere" size="0.1"
                    mass="2" friction="0.6 0.01 0.001"/>
              <body name="child" pos="0 0 -0.2">
                <inertial pos="0 0 -0.1" mass="1"
                          diaginertia="0.01 0.02 0.03"/>
                <joint name="hinge" axis="0 1 0" range="-1 2"
                       armature="0.01" damping="{damping}" frictionloss="0.03"/>
                <geom name="child_collision" type="capsule"
                      size="0.04 0.1" mass="1" condim="1"/>
              </body>
            </body>
          </worldbody>
          <actuator>
            <position name="hinge" joint="hinge" kp="12" kv="3"
                      ctrlrange="-2 2" forcerange="-8 8"/>
          </actuator>
        </mujoco>
        """
    )


def _public_motion_fixture(frames: int = 3) -> dict[str, object]:
    root = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3) / 10
    quat = np.tile(np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32), (frames, 1))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    dof = np.arange(frames * 23, dtype=np.float32).reshape(frames, 23) / 100
    return {
        "fps": 30,
        "root_trans_offset": root,
        "root_rot": quat,
        "dof": dof,
    }


def test_expand_side_kick_inserts_wrists_without_transforming_root() -> None:
    motion = _public_motion_fixture()

    result = expand_public_side_kick(motion)

    assert result.shape == (3, 36)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[:, :3], motion["root_trans_offset"])
    np.testing.assert_array_equal(result[:, 3:7], motion["root_rot"])
    np.testing.assert_array_equal(result[:, 7:26], motion["dof"][:, :19])
    np.testing.assert_array_equal(result[:, 26:29], 0.0)
    np.testing.assert_array_equal(result[:, 29:33], motion["dof"][:, 19:])
    np.testing.assert_array_equal(result[:, 33:36], 0.0)
    assert len(G1_JOINT_NAMES) == 29


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("fps", 50, "30 Hz"),
        ("root_rot", np.zeros((3, 3), dtype=np.float32), "root_rot"),
        ("dof", np.zeros((3, 29), dtype=np.float32), "23"),
    ],
)
def test_expand_side_kick_rejects_noncanonical_schema(
    field: str,
    value: object,
    match: str,
) -> None:
    motion = _public_motion_fixture()
    motion[field] = value

    with pytest.raises(ValueError, match=match):
        expand_public_side_kick(motion)


def test_float32_csv_is_byte_stable_and_exactly_roundtrippable(
    tmp_path: Path,
) -> None:
    array = expand_public_side_kick(_public_motion_fixture())
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    write_float32_csv(first, array)
    write_float32_csv(second, array)

    assert first.read_bytes() == second.read_bytes()
    np.testing.assert_array_equal(
        np.loadtxt(first, delimiter=",", dtype=np.float32),
        array,
    )


def test_deterministic_npz_is_byte_stable(tmp_path: Path) -> None:
    arrays = {
        "fps": np.array([50], dtype=np.int64),
        "joint_pos": np.arange(12, dtype=np.float32).reshape(3, 4),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    deterministic_npz(first, arrays)
    deterministic_npz(second, arrays)

    assert first.read_bytes() == second.read_bytes()
    with np.load(first) as restored:
        assert tuple(restored.files) == tuple(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(restored[name], expected)


class _ForbiddenGlobal:
    def __reduce__(self):
        return str, ("must not be reconstructed",)


def test_restricted_joblib_loader_rejects_unlisted_globals(tmp_path: Path) -> None:
    payload = tmp_path / "forbidden.pkl"
    joblib.dump(_ForbiddenGlobal(), payload, compress=0)

    with pytest.raises(ValueError, match=r"Rejected pickle global .*str"):
        restricted_joblib_load(payload)


def test_model_signature_covers_required_compiled_fields() -> None:
    signature = model_signature(_minimal_model())

    assert signature["options"]["timestep"] == pytest.approx(0.005)
    assert signature["options"]["iterations"] == 10
    assert signature["options"]["ls_iterations"] == 20
    assert signature["bodies"][1]["name"] == "pelvis"
    assert signature["bodies"][1]["mass"] == pytest.approx(2.0)
    assert signature["joints"][1]["name"] == "hinge"
    assert signature["joints"][1]["axis"] == pytest.approx([0.0, 1.0, 0.0])
    assert signature["joints"][1]["range"] == pytest.approx([-1.0, 2.0])
    assert signature["joints"][1]["armature"] == pytest.approx([0.01])
    assert signature["joints"][1]["damping"] == pytest.approx([0.2])
    assert signature["joints"][1]["frictionloss"] == pytest.approx([0.03])
    assert signature["geoms"][0]["name"] == "pelvis_collision"
    assert signature["geoms"][0]["friction"] == pytest.approx([0.6, 0.01, 0.001])
    assert signature["actuators"][0]["name"] == "hinge"
    assert signature["actuators"][0]["ctrlrange"] == pytest.approx([-2.0, 2.0])
    assert signature["actuators"][0]["forcerange"] == pytest.approx([-8.0, 8.0])


def test_model_signature_comparison_identifies_numeric_mismatch() -> None:
    reference = model_signature(_minimal_model(damping=0.2))
    candidate = model_signature(_minimal_model(damping=0.25))

    mismatches = compare_model_signatures(
        reference,
        candidate,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    assert any("joints[1].damping[0]" in mismatch for mismatch in mismatches)


def test_model_signature_comparison_ignores_mapping_insertion_order() -> None:
    reference = {"dimensions": {"nq": 36, "nv": 35}}
    candidate = {"dimensions": {"nv": 35, "nq": 36}}

    assert (
        compare_model_signatures(
            reference,
            candidate,
            atol=0.0,
            rtol=0.0,
        )
        == []
    )


@pytest.mark.parametrize("change", ["tracked", "untracked"])
def test_preparation_requires_clean_git_sources(tmp_path, change):
    def git(*args):
        return (
            subprocess.check_output(
                ["git", "-C", str(tmp_path), *args], stderr=subprocess.PIPE
            )
            .decode()
            .strip()
        )

    git("init", "-q")
    source = tmp_path / "model.xml"
    source.write_text("<mujoco/>")
    git("add", "model.xml")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    expected = git("rev-parse", "HEAD")
    assert _git_commit(tmp_path) == expected
    if change == "tracked":
        source.write_text('<mujoco model="changed"/>')
    else:
        (tmp_path / "extra.py").write_text("value = 1")
    with pytest.raises(RuntimeError, match="clean"):
        _git_commit(tmp_path)
