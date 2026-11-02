"""Selected playback compilation and generic body-tracking camera contracts."""

from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
import torch

from instantmimic.config.validation import ConfigSchemaError, _validate_play
from instantmimic.play.console import format_play_startup_summary
from instantmimic.play.render import resolve_playback_camera_view_layout
from instantmimic.play.stepper import _maybe_build_compiled_playback_policy
from instantmimic.rendering.mujoco_gl_playback import (
    resolve_camera_body_tracking,
    update_body_tracking_camera,
)
from instantmimic.rl.eager_fallback import EagerFallbackPolicy


@pytest.mark.parametrize("execution", [False, True])
def test_only_selected_policy_is_compiled_and_warmed(execution):
    calls, options = [], []

    class Agent:
        def policy_mean_into(self, obs, actions):
            calls.append("mean")
            actions.fill_(2)

        def policy_execution_mean_into(self, obs, actions):
            calls.append("execution_mean")
            actions.fill_(3)

    def compile_policy(fn, **kwargs):
        options.append(kwargs)
        return fn

    actions = torch.zeros((1, 2))
    with patch("torch.compile", side_effect=compile_policy):
        compiled = _maybe_build_compiled_playback_policy(
            agent=Agent(),
            device=SimpleNamespace(type="cuda"),
            obs_policy=torch.zeros((1, 3)),
            agent_actions=actions,
            play_torch_stream=None,
            stochastic_play=False,
            eager_fallback=EagerFallbackPolicy(allowed=False),
            use_execution_mean=execution,
        )
    key = "execution_mean" if execution else "mean"
    other = "mean" if execution else "execution_mean"
    assert options == [{"mode": "max-autotune-no-cudagraphs", "dynamic": False}]
    assert calls == [key]
    assert compiled[other] is None
    result = torch.empty_like(actions)
    compiled[key](torch.ones((1, 3)), result)
    torch.testing.assert_close(result, torch.full_like(result, 3 if execution else 2))
    torch.testing.assert_close(actions, torch.zeros_like(actions))


def test_camera_tracking_follows_centroid_without_changing_physics_or_pose():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="a" pos="1 -0.2 0.1"><freejoint/>'
        '<geom size=".05"/></body><body name="b" pos="1 0.2 0.1"><freejoint/>'
        '<geom size=".05"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    ids, shift = resolve_camera_body_tracking(model, 1, [("a", "b")], [(0, 0, 0.02)])[0]
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 0.9, 123, -25
    for delta in (0, 3):
        data.qpos[0], data.qpos[7] = 1 + delta, 1.4 + delta
        mujoco.mj_kinematics(model, data)
        before = data.qpos.copy()
        update_body_tracking_camera(camera, data, ids, shift)
        np.testing.assert_allclose(camera.lookat, [1.2 + delta, 0, 0.12])
        np.testing.assert_array_equal(data.qpos, before)
        assert (camera.distance, camera.azimuth, camera.elevation) == (0.9, 123, -25)
    with pytest.raises(ValueError, match="unknown"):
        resolve_camera_body_tracking(model, 1, [("missing",)])
    with pytest.raises(ValueError, match="camera count"):
        resolve_camera_body_tracking(model, 2, [("a",)])


@pytest.mark.parametrize(
    "view",
    [
        {"track_bodies": []},
        {"track_bodies": [""]},
        {"lookat_offset": [0, 0, 0]},
        {"track_bodies": ["a"], "lookat_offset": [0, 1]},
        {"track_bodies": ["a"], "lookat_offset": [0, 1, float("nan")]},
    ],
)
def test_invalid_camera_settings_are_rejected(view):
    with pytest.raises(ConfigSchemaError):
        _validate_play(
            {
                "num_envs": 1,
                "steps": 2,
                "action": {"mode": "mean"},
                "video": {"render_backend": "mujoco_gl"},
                "camera": {"views": {"cameras": [view]}},
            }
        )


def test_camera_layout_preserves_per_view_tracking():
    base = mujoco.MjvCamera()
    cfg = {
        "playback_camera_views": {
            "cameras": [{"track_bodies": ["a", "b"], "lookat_offset": [0, 0, 0.2]}, {}]
        }
    }
    _validate_play(
        {
            "num_envs": 1,
            "steps": 2,
            "action": {"mode": "mean"},
            "video": {"render_backend": "mujoco_gl"},
            "camera": {"views": cfg["playback_camera_views"]},
        }
    )
    layout = resolve_playback_camera_view_layout(
        base, cfg, default_width=64, default_height=32
    )
    assert layout.track_bodies == (("a", "b"), ())
    assert layout.lookat_offsets == ((0, 0, 0.2), (0, 0, 0))


def test_startup_summary_includes_policy_compile_phase():
    record = SimpleNamespace(name="policy_compile_warmup", seconds=1.25)
    assert "compile 1.25s" in format_play_startup_summary([record])
