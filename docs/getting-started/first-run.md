# First training run

From the repository root, train the included humanoid walk:

```bash
uv run train --config-name humanoid/imitation/walk_pd
```

The config selects the model, reference motion, number of parallel
environments and PPO settings. To see every resolved value without training:

```bash
uv run train --config-name humanoid/imitation/walk_pd --cfg job --resolve
```

To check that everything runs before committing to a full run, stop after one
iteration:

```bash
uv run train --config-name humanoid/imitation/walk_pd trainer.max_iterations=1
```

The first run compiles kernels before training starts; later runs start faster
because the compiled kernels are cached.

## Replay the policy

Each run gets a directory under `logs/ppo/`, named after the config and the
start time, with its resolved config, logs and checkpoints. Replay the best
checkpoint with:

```bash
uv run play --config-name humanoid/imitation/walk_pd \
  +play.checkpoint='logs/ppo/humanoid/imitation/walk_pd/YOUR_RUN/checkpoints/best_agent.pt' \
  play.video.enabled=false play.steps=300
```

Playback restores the policy together with the config it was trained with. To
record a video, see [video rendering](../guides/rendering.md); to resume
training or evaluate further, see
[training and playback](../guides/training-and-playback.md).
