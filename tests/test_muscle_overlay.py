from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from instantmimic.envs.controls.fast_muscle.tendon_activation_visualizer import (
    MuscleLineSceneVisualizer,
)
from instantmimic.play.viser_viewer import _resolve_fast_muscle_source_points
from instantmimic.rendering.muscle_overlay import (
    _muscle_endpoint_world,
    _WarpMuscleLineOverlay,
)


@wp.kernel(enable_backward=False)
def _endpoints(
    body: wp.array[int],
    local: wp.array[wp.vec3],
    world_points: wp.array2d[wp.vec3],
    starts: wp.array[int],
    attachment_body: wp.array[int],
    local_weight: wp.array[wp.vec4],
    use_lbs: bool,
    xpos: wp.array2d[wp.vec3],
    xquat: wp.array2d[wp.quat],
    output: wp.array2d[wp.vec3],
):
    world, endpoint = wp.tid()
    output[world, endpoint] = _muscle_endpoint_world(
        world,
        endpoint,
        use_lbs,
        body,
        local,
        world_points,
        False,
        starts,
        attachment_body,
        local_weight,
        xpos,
        xquat,
    )


def _source(lbs):
    body = np.array([0, 1, 1], dtype=np.int32)
    local = np.array([[-0.6, 0, 0], [0, 0, 0], [0.6, 0, 0]], dtype=np.float32)
    kwargs = {}
    if lbs:
        kwargs = dict(
            waypoint_attachment_start=np.arange(4, dtype=np.int32),
            waypoint_attachment_body=body,
            waypoint_attachment_local=local,
            waypoint_attachment_weight=np.ones(3, dtype=np.float32),
        )
    return MuscleLineSceneVisualizer(
        muscle_count=1,
        activation_resolver=lambda: torch.tensor([[0.2], [0.8]]),
        activation_clamp_enabled_resolver=lambda: 0.0,
        activation_lower_bound=0.0,
        activation_upper_bound=1.0,
        color_low=np.array([0.0, 0.0, 1.0, 1.0]),
        color_high=np.array([1.0, 0.0, 0.0, 1.0]),
        color_under=np.array([0.0, 1.0, 1.0, 1.0]),
        color_over=np.array([1.0, 1.0, 0.0, 1.0]),
        alpha=1.0,
        waypoint_body=body,
        waypoints_local=local,
        waypoint_start=np.array([0, 3], dtype=np.int32),
        **kwargs,
    ).build_gl_render_source()


def _data():
    xpos = np.array(
        [[[0, 0, -3], [0.2, 0, -3]], [[0, 0.5, -3], [0.2, 0.5, -3]]], dtype=np.float32
    )
    xquat = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (2, 2, 1))
    # A quarter turn on the second body exercises nontrivial local coordinates.
    xquat[:, 1] = [2**-0.5, 0, 0, 2**-0.5]
    return SimpleNamespace(
        xpos=wp.array(xpos, dtype=wp.vec3, device="cpu"),
        xquat=wp.array(xquat, dtype=wp.quat, device="cpu"),
        cam_xpos=wp.zeros((2, 1), dtype=wp.vec3, device="cpu"),
        cam_xmat=wp.array(
            np.tile(np.eye(3, dtype=np.float32), (2, 1, 1, 1)),
            dtype=wp.mat33,
            device="cpu",
        ),
    )


def _rc():
    return SimpleNamespace(
        rgb_data=wp.zeros((2, 32 * 32), dtype=wp.uint32, device="cpu"),
        rgb_adr=wp.array([0], dtype=int, device="cpu"),
        depth_data=wp.full((2, 32 * 32), value=10.0, dtype=float, device="cpu"),
        depth_adr=wp.array([0], dtype=int, device="cpu"),
    )


@pytest.mark.parametrize("lbs", [False, True])
def test_body_local_and_weight_one_lbs_match_cpu_points_and_draw(lbs):
    source, data = _source(lbs), _data()
    overlay = _WarpMuscleLineOverlay(source)
    actual = wp.empty((2, source.vertex_count), dtype=wp.vec3, device="cpu")
    wp.launch(
        _endpoints,
        dim=(2, source.vertex_count),
        inputs=[
            overlay.endpoint_body,
            overlay.endpoint_local,
            overlay._endpoint_world,
            overlay.endpoint_attachment_start,
            overlay.endpoint_attachment_body,
            overlay.endpoint_attachment_local_weight,
            overlay.use_lbs,
            data.xpos,
            data.xquat,
            actual,
        ],
        device="cpu",
    )
    xpos, xquat = data.xpos.numpy(), data.xquat.numpy()
    for world in range(2):
        expected = _resolve_fast_muscle_source_points(
            source, SimpleNamespace(xpos=xpos[world], xquat=xquat[world])
        )
        np.testing.assert_allclose(actual.numpy()[world], expected, atol=2e-7)
        # Independent expected transformation for the final rotated endpoint.
        np.testing.assert_allclose(
            expected[-1], [0.2, world * 0.5 + 0.6, -3.0], atol=2e-7
        )

    batch_rc, single_rc = _rc(), _rc()
    values = torch.tensor([[0.2], [0.8]])
    overlay.prepare_batch(data=data, world_count=2, activation_batch=values)
    signature = overlay.prepared_graph_signature
    overlay.draw_prepared_batch(
        data=data,
        rc=batch_rc,
        camera_id=0,
        width=32,
        height=32,
        fovy=45.0,
        world_count=2,
    )
    for world in range(2):
        overlay.draw(
            data=data,
            rc=single_rc,
            camera_id=0,
            width=32,
            height=32,
            fovy=45.0,
            env_idx=world,
            world_id=world,
            activation=values[world],
        )
    np.testing.assert_array_equal(single_rc.rgb_data.numpy(), batch_rc.rgb_data.numpy())
    assert all(np.count_nonzero(row) for row in batch_rc.rgb_data.numpy())
    assert not np.array_equal(
        batch_rc.rgb_data.numpy()[0], batch_rc.rgb_data.numpy()[1]
    )
    overlay.prepare_batch(data=data, world_count=2, activation_batch=values.flip(0))
    # Only the first single staging adds an independent persistent buffer.
    assert signature[:2] == overlay.prepared_graph_signature[:2]
    stable = overlay.prepared_graph_signature
    overlay.prepare_batch(data=data, world_count=2, activation_batch=values)
    assert stable == overlay.prepared_graph_signature
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("path", [None, "", "   "])
def test_lbs_requires_an_explicit_mesh_bundle(path):
    import mujoco

    from instantmimic.rendering.muscle_overlay import _WarpLBSMeshOverlay

    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(
        ValueError, match="requires an explicit video_warp_lbs_mesh_bundle path"
    ):
        _WarpLBSMeshOverlay(model=model, bundle_path=path, mode="muscle")


@pytest.mark.parametrize("mapped", [False, True])
def test_lbs_loads_explicit_external_bundle_and_optional_mapping(tmp_path, mapped):
    import json

    import mujoco

    from instantmimic.rendering.muscle_overlay import _WarpLBSMeshOverlay

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="Pelvis"><geom size=".1"/></body></worldbody></mujoco>'
    )
    bundle = {
        "boneNames": ["Pelvis"],
        "bindPositions": {"Pelvis": [0, 0, 0]},
        "meshes": [
            {
                "name": "muscle_a",
                "type": "muscle",
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                "faces": [[0, 1, 2]],
                "normals": [[0, 0, 1]] * 3,
                "boneIndices": [[0, 0, 0, 0]] * 3,
                "boneWeights": [[1, 0, 0, 0]] * 3,
            }
        ],
    }
    bundle_path = tmp_path / "external_bundle.json"
    bundle_path.write_text(json.dumps(bundle))
    mapping_path = None
    if mapped:
        mapping_path = tmp_path / "external_mapping.json"
        mapping_path.write_text(
            json.dumps(
                {
                    "muscleNames": ["muscle_a"],
                    "channelToMuscleIndex": [0, 0],
                }
            )
        )
    overlay = _WarpLBSMeshOverlay(
        model=model,
        bundle_path=bundle_path,
        mode="muscle",
        activation_mapping_path=mapping_path,
    )
    assert overlay.vertex_count == 3
    assert overlay.face_count == 1
    np.testing.assert_allclose(overlay.vertex_local_weight_np[1, 0], [1, 0, 0, 1])
    if mapped:
        assert overlay.activation_mapping.activation_channel_count == 2
        np.testing.assert_array_equal(
            overlay.activation_mapping.mesh_channel_indices, [0, 1]
        )
    else:
        assert overlay.activation_mapping is None
        assert (
            overlay._stage_activation_tensor(None, torch.ones(1, 7), world_count=1) == 0
        )


def test_default_playback_does_not_require_optional_lbs_assets():
    import mujoco

    from instantmimic.rendering.mujoco_warp_playback import MujocoWarpPlaybackRenderer

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="cam"/><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    with wp.ScopedDevice("cpu"):
        renderer = MujocoWarpPlaybackRenderer(
            env_raw=SimpleNamespace(),
            model=model,
            width=8,
            height=8,
            camera="cam",
            use_textures=False,
        )
    assert renderer._lbs_mesh_overlay is None
    assert not torch.cuda.is_initialized()


def test_dynamic_world_endpoints_match_body_local_overlay_and_keep_stable_buffers():
    from dataclasses import replace

    base, data = _source(False), _data()
    xpos, xquat = data.xpos.numpy(), data.xquat.numpy()
    world = torch.tensor(
        np.stack(
            [
                _resolve_fast_muscle_source_points(
                    base, SimpleNamespace(xpos=xpos[i], xquat=xquat[i])
                )
                for i in range(2)
            ]
        ),
        dtype=torch.float32,
    )
    dynamic = replace(base, endpoint_world_resolver=lambda data: world)
    reference_overlay, overlay = (
        _WarpMuscleLineOverlay(base),
        _WarpMuscleLineOverlay(dynamic),
    )
    reference, actual = _rc(), _rc()
    for line, rc in ((reference_overlay, reference), (overlay, actual)):
        line.prepare_batch(data=data, world_count=2)
        line.draw_prepared_batch(
            data=data, rc=rc, camera_id=0, width=32, height=32, fovy=45.0, world_count=2
        )
    np.testing.assert_array_equal(actual.rgb_data.numpy(), reference.rgb_data.numpy())
    assert np.count_nonzero(actual.rgb_data.numpy()) > 0
    signature = overlay.prepared_graph_signature
    world[:, :, 1].add_(0.2)
    overlay.prepare_batch(data=data, world_count=2)
    assert signature == overlay.prepared_graph_signature
    assert overlay._uses_world
