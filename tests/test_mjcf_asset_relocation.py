"""Included compiler declarations retain main-file asset paths after expansion."""

from pathlib import Path

import mujoco
import numpy as np

from instantmimic.muscle.parsing.mjcf import build_stripped_mjcf_spec


def test_stripped_model_resolves_compiler_paths_after_relocation(tmp_path: Path):
    for name in ("original", "relocated"):
        root = tmp_path / name
        (root / "assets").mkdir(parents=True)
        (root / "parts").mkdir()
        (root / "assets/tetra.obj").write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
        )
        (root / "parts/body.xml").write_text(
            '<mujocoinclude><compiler angle="radian" inertiafromgeom="auto"/>'
            '<asset><mesh name="tetra" file="tetra.obj"/></asset>'
            '<worldbody><body><joint type="hinge"/>'
            '<geom type="mesh" mesh="tetra" density="1"/></body></worldbody>'
            "</mujocoinclude>"
        )
        (root / "model.xml").write_text(
            '<mujoco><include file="parts/body.xml"/>'
            '<compiler meshdir="assets"/></mujoco>'
        )
    original = tmp_path / "original/model.xml"
    native = mujoco.MjModel.from_xml_path(str(original))
    # Remove the original tree: a relocated model must stand on its own.
    import shutil

    shutil.rmtree(original.parent)
    actual = build_stripped_mjcf_spec(str(tmp_path / "relocated/model.xml")).compile()
    np.testing.assert_array_equal(actual.body_mass, native.body_mass)
    np.testing.assert_array_equal(actual.body_inertia, native.body_inertia)
    assert actual.nmesh == native.nmesh == 1
