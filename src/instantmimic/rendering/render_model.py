from __future__ import annotations

import copy

import mujoco


def clone_playback_render_model(
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Clone a camera-bearing model and its current render state."""
    if int(model.ncam) > 0:
        render_model = copy.copy(model)
        render_data = mujoco.MjData(render_model)
        _copy_render_state(data, render_data)
        return render_model, render_data

    raise ValueError(
        "MuJoCo-Warp playback requires a real camera in the compiled render model. "
        "Build the render-only model with ensure_playback_camera_element() before "
        "compilation; MuJoCo 3.16 cannot derive a safe editable MjSpec from an "
        "arbitrary compiled model."
    )


def _copy_render_state(source: mujoco.MjData | None, target: mujoco.MjData) -> None:
    if source is None:
        return
    for name in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "ctrl", "userdata"):
        source_value = getattr(source, name)
        target_value = getattr(target, name)
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"Playback render data field {name!r} changed shape: "
                f"source={source_value.shape}, render={target_value.shape}."
            )
        target_value[:] = source_value
    target.time = source.time
    mujoco.mj_forward(target.model, target)
