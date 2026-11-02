"""Fixed-radius transmissions agree with independently evaluated MuJoCo paths."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from instantmimic.muscle.parsing.mjcf import parse_muscles_from_mjcf


@pytest.fixture
def ostrich_model():
    path = (
        Path(__file__).resolve().parents[1]
        / "data/freemusco/animal/ostrich/ostrich.xml"
    )
    return path, mujoco.MjModel.from_xml_path(str(path))


def test_parser_preserves_wrapped_transmission_topology_and_normalization(
    ostrich_model,
):
    path, model = ostrich_model
    parsed = parse_muscles_from_mjcf(str(path), compiled_model=model)
    expected = np.count_nonzero(model.wrap_type == int(mujoco.mjtWrap.mjWRAP_CYLINDER))
    assert parsed.wraps is not None
    assert parsed.wraps.radius.size == expected
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for i, muscle in enumerate(parsed.muscles):
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, muscle.name)
        np.testing.assert_allclose(
            parsed.rest_lengths[i], data.actuator_length[actuator], rtol=2e-6, atol=1e-7
        )
    assert (
        sum(len(m.waypoints) for m in parsed.muscles)
        == np.count_nonzero(model.wrap_type == int(mujoco.mjtWrap.mjWRAP_SITE))
        + 2 * expected
    )


def test_wrapped_path_lengths_follow_changing_pose(ostrich_model):
    from instantmimic.muscle.muscle_qfrc_projector import transform_waypoints_kernel
    from instantmimic.muscle.waypoint_wrap import (
        WaypointWrapping,
        WaypointWrapState,
        make_wrap_spec,
        path_lengths_kernel,
    )

    path, model = ostrich_model
    parsed = parse_muscles_from_mjcf(str(path), compiled_model=model)
    bodies = torch.tensor(
        [model.body(p.body).id for m in parsed.muscles for p in m.waypoints],
        dtype=torch.int32,
    )
    local = torch.tensor(
        np.stack([p.position for m in parsed.muscles for p in m.waypoints]),
        dtype=torch.float32,
    )
    starts = torch.tensor(
        np.cumsum([0] + [len(m.waypoints) for m in parsed.muscles]), dtype=torch.int32
    )
    wrapping = WaypointWrapping(make_wrap_spec(parsed, model, "cpu"), len(local), "cpu")
    state = WaypointWrapState(wrapping.count, 1, "cpu")
    points = wp.zeros((1, len(local)), dtype=wp.vec3, device="cpu")
    lengths = wp.zeros((1, len(parsed.muscles)), dtype=wp.float32, device="cpu")
    data = mujoco.MjData(model)
    rng = np.random.default_rng(42)
    for amplitude in (0.0, 0.025, 0.075):
        data.qpos[:] = model.qpos0
        velocity = rng.normal(size=model.nv) * amplitude
        velocity[:6] = 0
        mujoco.mj_integratePos(model, data.qpos, velocity, 1.0)
        mujoco.mj_forward(model, data)
        xpos = wp.array(data.xpos[None].astype(np.float32), dtype=wp.vec3, device="cpu")
        xquat = wp.array(
            data.xquat[None].astype(np.float32), dtype=wp.quat, device="cpu"
        )
        wp.launch(
            transform_waypoints_kernel,
            dim=(1, len(local)),
            inputs=[
                xpos,
                xquat,
                wp.from_torch(bodies),
                wp.from_torch(local, dtype=wp.vec3),
            ],
            outputs=[points],
            device="cpu",
        )
        wrapping.transform(xpos, xquat, points, state)
        wp.launch(
            path_lengths_kernel,
            dim=(1, len(parsed.muscles)),
            inputs=[points, wp.from_torch(starts), wrapping.point_arc, state.arc],
            outputs=[lengths],
            device="cpu",
        )
        expected = [
            data.actuator_length[model.actuator(m.name).id] for m in parsed.muscles
        ]
        np.testing.assert_allclose(lengths.numpy()[0], expected, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
@pytest.mark.parametrize("side", [False, True])
def test_small_wrapped_paths_and_rendered_arcs_match_native_geometry(
    kind, side, tmp_path
):
    from instantmimic.muscle.waypoint_wrap import make_wrap_spec
    from instantmimic.rendering.muscle_wrap_geometry import WrappedMuscleRenderGeometry

    side_attribute = 'sidesite="side"' if side else ""
    model = mujoco.MjModel.from_xml_string(f'''<mujoco>
      <worldbody>
        <geom name="wrap" type="{kind}" size=".3 .7" quat=".98 .1 .1 .1"/>
        <site name="origin" pos="-1 .2 0"/><site name="side" pos="0 .5 0"/>
        <body name="moving" pos="1 0 0">
          <joint type="slide" axis="0 1 0"/>
          <geom type="sphere" size=".01" mass="1"/><site name="insertion" pos="0 .2 .1"/>
        </body>
      </worldbody>
      <tendon><spatial name="path"><site site="origin"/>
        <geom geom="wrap" {side_attribute}/><site site="insertion"/>
      </spatial></tendon>
      <actuator><muscle name="muscle" tendon="path" force="100" lengthrange=".5 3"/></actuator>
    </mujoco>''')
    # Parser accepts an explicit compiled model independently of source-file/cache parsing.
    source = tmp_path / "small.xml"
    mujoco.mj_saveLastXML(str(source), model)
    parsed = parse_muscles_from_mjcf(str(source), compiled_model=model)
    points = [point for muscle in parsed.muscles for point in muscle.waypoints]
    body = np.array([model.body(point.body).id for point in points], dtype=np.int32)
    local = np.array([point.position for point in points])
    geometry = WrappedMuscleRenderGeometry(
        make_wrap_spec(parsed, model, "cpu"),
        body,
        local,
        [0, len(points)],
        arc_segments=32,
    )
    data = mujoco.MjData(model)
    for position in (-0.2, 0.25, 2.0):
        data.qpos[0] = position
        mujoco.mj_forward(model, data)
        actual = geometry(data)[0].numpy().copy()
        assert np.isfinite(actual).all()
        length = np.linalg.norm(np.diff(actual, axis=0), axis=1).sum()
        np.testing.assert_allclose(length, data.ten_length[0], atol=4e-4, rtol=2e-4)
        # Same renderer geometry from buffered Warp body poses, without consulting live qpos.
        from types import SimpleNamespace

        buffered = SimpleNamespace(
            xpos=wp.array(data.xpos[None].astype("f"), dtype=wp.vec3, device="cpu"),
            xquat=wp.array(data.xquat[None].astype("f"), dtype=wp.quat, device="cpu"),
        )
        np.testing.assert_allclose(geometry(buffered)[0].numpy(), actual, atol=2e-6)
        # Envelope-theorem point reactions agree with native virtual work, including no-wrap.
        original = actual[geometry.point_index[:-1]]
        jacobian = np.zeros(model.nv)
        for i in range(len(points) - 1):
            direction = original[i + 1] - original[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-7:
                continue
            ja = np.zeros((3, model.nv))
            jb = ja.copy()
            mujoco.mj_jac(model, data, ja, None, original[i].astype("d"), int(body[i]))
            mujoco.mj_jac(
                model, data, jb, None, original[i + 1].astype("d"), int(body[i + 1])
            )
            jacobian += direction / norm @ (jb - ja)
        native = np.zeros((model.ntendon, model.nv))
        mujoco.mju_sparse2dense(
            native,
            data.ten_J,
            model.ten_J_rownnz,
            model.ten_J_rowadr,
            model.ten_J_colind,
        )
        np.testing.assert_allclose(jacobian, native[0], atol=2e-6)
