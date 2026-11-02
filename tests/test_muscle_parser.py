"""Body-fixed muscle path boundaries of the public MJCF input port."""

from __future__ import annotations

from dataclasses import asdict

import mujoco
import numpy as np
import pytest

from instantmimic.muscle.parsing import mjcf


def _source(tmp_path, geometry=None):
    shape = "0.1 0.2" if geometry == "cylinder" else "0.1"
    obstacle = (
        f'<geom name="obstacle" type="{geometry}" size="{shape}" pos="0 0 0"/>'
        if geometry
        else ""
    )
    segment = '<geom geom="obstacle"/>' if geometry else ""
    path = tmp_path / "muscles.xml"
    path.write_text(f"""<mujoco><size nuser_actuator="4"/>
<worldbody><body name="segment"><joint name="hinge" type="hinge"/>
<geom type="sphere" size="0.05" pos="0 0 0.3" mass="1"/>
{obstacle}<site name="start" pos="-0.4 0 0"/>
<site name="end" pos="0.4 0 0"/></body></worldbody>
<tendon><spatial name="path"><site site="start"/>{segment}<site site="end"/></spatial></tendon>
<actuator><muscle name="fiber" tendon="path" force="10" lengthrange="0.1 1" user="1 0 0 1.5"/></actuator>
</mujoco>""")
    return path


@pytest.fixture(autouse=True)
def disable_parser_disk_cache(monkeypatch):
    monkeypatch.setattr(mjcf, "mjcf_cache_dir", lambda: None)


@pytest.mark.parametrize("geometry", ["sphere", "cylinder"])
def test_raw_compiled_geometry_is_rejected_before_metadata_override(
    tmp_path, monkeypatch, geometry
):
    source = _source(tmp_path, geometry)
    parsed = mjcf.ParsedMjcfMuscleData(
        muscles=[], rest_lengths=np.empty(0, dtype=np.float32)
    )
    metadata_called = False

    def metadata(*args, **kwargs):
        nonlocal metadata_called
        metadata_called = True
        return parsed

    monkeypatch.setattr(mjcf, "parse_muscles_from_metadata_xml", metadata)
    with pytest.raises(ValueError, match="authored tendon topology"):
        mjcf.FastMuscleBootstrap.from_mjcf(
            str(source), muscle_metadata_path="metadata.xml"
        )
    assert not metadata_called


def test_site_polyline_preserves_transmission_and_physical_parameters(tmp_path):
    source = _source(tmp_path)
    parsed = mjcf.parse_muscles_from_mjcf(str(source))
    assert len(parsed.muscles) == 1
    muscle = parsed.muscles[0]
    assert (
        muscle.name,
        muscle.f0,
        muscle.lm,
        muscle.lt,
        muscle.pen_angle,
        muscle.lmax,
    ) == ("fiber", 10, 1, 0, 0, 1.5)
    np.testing.assert_array_equal(
        parsed.rest_lengths, np.asarray([0.8], dtype=np.float32)
    )
    assert [(point.body, len(point.attachments)) for point in muscle.waypoints] == [
        ("segment", 1),
        ("segment", 1),
    ]


def test_supplied_compiled_model_is_reused_without_compiling_again(
    tmp_path, monkeypatch
):
    source = _source(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(source))

    def unexpected_compile(*args, **kwargs):
        raise AssertionError(
            "parser attempted to compile an existing source model again"
        )

    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", unexpected_compile)
    parsed = mjcf.parse_muscles_from_mjcf(str(source), compiled_model=model)
    assert len(parsed.muscles) == 1
    np.testing.assert_array_equal(
        parsed.rest_lengths, np.asarray([0.8], dtype=np.float32)
    )


def test_bootstrap_compiles_the_source_once_for_metadata_and_reuses_it(
    tmp_path, monkeypatch
):
    source = _source(tmp_path)
    original = mujoco.MjModel.from_xml_path
    calls = 0

    def compile_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def metadata(_path, *, source_mjcf_path, compiled_model):
        return mjcf.parse_muscles_from_mjcf(
            source_mjcf_path, compiled_model=compiled_model
        )

    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", compile_once)
    monkeypatch.setattr(mjcf, "parse_muscles_from_metadata_xml", metadata)
    bootstrap = mjcf.FastMuscleBootstrap.from_mjcf(
        str(source), muscle_metadata_path="metadata.xml"
    )
    assert calls == 1
    assert len(bootstrap.parsed_mjcf_data.muscles) == 1
    assert bootstrap.geometry_backend == "waypoint"
    assert asdict(bootstrap.parsed_mjcf_data.muscles[0])["name"] == "fiber"


def test_compiled_site_path_preserves_rest_keyframe_and_parameters(tmp_path):
    source = _source(tmp_path)
    source.write_text(
        source.read_text().replace(
            "</mujoco>", '<keyframe><key name="rest" qpos="0"/></keyframe></mujoco>'
        )
    )
    parsed = mjcf.parse_muscles_from_mjcf(str(source), rest_keyframe_name="rest")
    assert len(parsed.muscles) == 1
    np.testing.assert_array_equal(
        parsed.rest_lengths, np.asarray([0.8], dtype=np.float32)
    )
    muscle = parsed.muscles[0]
    assert (muscle.f0, muscle.lm, muscle.lt, muscle.pen_angle, muscle.lmax) == (
        10,
        1,
        0,
        0,
        1.5,
    )


def test_supplied_model_geometry_is_checked_even_with_site_only_xml(tmp_path):
    source = _source(tmp_path, "sphere")
    model = mujoco.MjModel.from_xml_path(str(source))
    _source(tmp_path)
    with pytest.raises(ValueError, match="topology does not match"):
        mjcf.parse_muscles_from_mjcf(str(source), compiled_model=model)


def test_source_geometry_cannot_be_hidden_by_an_unrelated_site_only_model(tmp_path):
    source = _source(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(source))
    _source(tmp_path, "sphere")
    with pytest.raises(ValueError, match="topology does not match"):
        mjcf.parse_muscles_from_mjcf(str(source), compiled_model=model)


def test_metadata_cache_cannot_hide_unsupported_source_geometry(tmp_path, monkeypatch):
    source = _source(tmp_path, "sphere")

    def unexpected_cache_lookup(*args, **kwargs):
        raise AssertionError("metadata cache was consulted before source validation")

    monkeypatch.setattr(mjcf, "load_pickle_cache", unexpected_cache_lookup)
    with pytest.raises(ValueError, match="authored tendon topology"):
        mjcf.parse_muscles_from_metadata_xml(
            "missing.xml", source_mjcf_path=str(source)
        )


@pytest.mark.parametrize("use_keyframe", [False, True])
def test_authored_muscle_order_is_validated_on_both_parser_paths(
    tmp_path, use_keyframe
):
    source = _source(tmp_path)
    source.write_text(
        source.read_text().replace(
            "</mujoco>",
            '<custom><text name="instantmimic_muscle_names" data=\'["other"]\'/></custom>'
            '<keyframe><key name="rest" qpos="0"/></keyframe></mujoco>',
        )
    )
    with pytest.raises(ValueError, match="muscle order"):
        mjcf.parse_muscles_from_mjcf(
            str(source), rest_keyframe_name="rest" if use_keyframe else None
        )


def _metadata(tmp_path, *, waypoint_extra="", path_tag=None):
    path = tmp_path / "metadata.xml"
    waypoints = (
        f'<Waypoint body="segment" p="-0.4 0 0" {waypoint_extra}/>'
        '<Waypoint body="segment" p="0.4 0 0"/>'
    )
    if path_tag:
        waypoints = f"<{path_tag}>{waypoints}</{path_tag}>"
    path.write_text(
        '<Muscles><Unit name="fiber" f0="10" lm="1" lt="0" '
        f'pen_angle="0" lmax="1.5">{waypoints}</Unit></Muscles>'
    )
    return path


def test_sidecar_keeps_one_body_fixed_attachment_per_point(tmp_path):
    source = _source(tmp_path)
    metadata = _metadata(tmp_path)
    parsed = mjcf.parse_muscles_from_metadata_xml(
        str(metadata), source_mjcf_path=str(source)
    )
    np.testing.assert_array_equal(
        parsed.rest_lengths, np.asarray([0.8], dtype=np.float32)
    )
    packed = mjcf._pack_muscle_data(parsed)
    np.testing.assert_array_equal(packed.attachment_start, [0, 1, 2])
    np.testing.assert_array_equal(
        packed.attachment_positions, packed.waypoint_positions
    )
    np.testing.assert_array_equal(packed.attachment_weights, [1, 1])


@pytest.mark.parametrize(
    "metadata_kwargs",
    [
        {"waypoint_extra": 'attachment="segment other"'},
        {"path_tag": "AlternatePath"},
    ],
)
def test_sidecar_rejects_extensions_outside_fixed_polyline_contract(
    tmp_path, metadata_kwargs
):
    source = _source(tmp_path)
    metadata = _metadata(tmp_path, **metadata_kwargs)
    with pytest.raises(ValueError, match="fixed"):
        mjcf.parse_muscles_from_metadata_xml(
            str(metadata), source_mjcf_path=str(source)
        )


@pytest.mark.parametrize("geometry", [None, "sphere"])
@pytest.mark.parametrize("gear", [0.5, 2.0])
def test_scaled_transmissions_are_rejected_before_geometry_or_metadata_parse(
    tmp_path, geometry, gear
):
    source = _source(tmp_path, geometry)
    source.write_text(
        source.read_text().replace(
            '<muscle name="fiber"', f'<muscle gear="{gear}" name="fiber"'
        )
    )
    with pytest.raises(ValueError, match="unit tendon gear"):
        mjcf.parse_muscles_from_mjcf(str(source))
    with pytest.raises(ValueError, match="unit tendon gear"):
        mjcf.FastMuscleBootstrap.from_mjcf(
            str(source), muscle_metadata_path="missing.xml"
        )


def test_unwrapped_geometry_does_not_require_an_initialized_warp_runtime():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from instantmimic.muscle.waypoint_wrap import WaypointWrapping; assert WaypointWrapping(None, 2, 'cpu').count == 0",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
