# Video rendering

`play` can record a video of the policy. On a Linux machine without a display,
render with MuJoCo's EGL backend:

```bash
MUJOCO_GL=egl uv run play --config-name humanoid/imitation/walk_pd \
  +play.checkpoint='logs/YOUR_RUN/checkpoints/best_agent.pt' \
  play.video.enabled=true +play.video.render_backend=mujoco_gl \
  play.video.length=300 play.steps=300
```

This needs the NVIDIA EGL libraries that match your driver. `play.steps` and
`play.video.length` set how long the clip is.

## Camera views and tracking

`play.camera.views` arranges one or more cameras in the video. A camera with
`track_bodies` follows the average position of those bodies, shifted by
`lookat_offset`, at a fixed distance, azimuth and elevation. For the included
humanoid, a camera following the pelvis looks like this:

```bash
MUJOCO_GL=egl uv run play --config-name humanoid/imitation/walk_pd \
  +play.checkpoint='logs/YOUR_RUN/checkpoints/best_agent.pt' \
  play.video.enabled=true +play.video.render_backend=mujoco_gl \
  play.video.length=32 play.steps=8 \
  '+play.camera.views={resolution:[256,256],cameras:[{track_bodies:[pelvis],lookat_offset:[0,0,0.2],distance:3.0,azimuth_deg:135,elevation_deg:-20}]}'
```

Use either `play.camera.views` or the single-camera `play.camera` settings, not
both. `lookat_offset` only applies together with `track_bodies`.

## Encoding

By default (`play.video.encoder_backend=auto`), videos are encoded on the GPU
with NVENC and fall back to imageio when the frame size is outside what NVENC
supports. Selecting `pynvcodec` explicitly reports an unsupported size as an
error instead.
