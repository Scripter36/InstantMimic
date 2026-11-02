# Configuration

Configs are Hydra files under `config/`. A task config pulls in shared files
through its `defaults` list, so a single YAML file rarely shows every value. To
see the full resolved config, and to check every config in the repository:

```bash
uv run train --config-name humanoid/imitation/walk_pd --cfg job --resolve
uv run python -m instantmimic.tools.validate_configs --allow-unprepared-assets --json
```

Configs are validated when a run starts, so unknown keys and incompatible
settings are reported before anything is built.

| Section | Contents |
| --- | --- |
| `env` | Model, number of parallel environments, reference motion, actions, observations, rewards, resets and termination |
| `models` | Actor and critic networks, action distribution and observation normalization |
| `agent` | PPO rollouts, discounting, losses and optimizer |
| `trainer` | Iteration count, initialization, checkpoints and benchmark evaluation |
| `runtime` | Compilation, CUDA graphs and PPO precision |
| `experiment` | Run directory and logger |
| `play` | Checkpoint, playback length, cameras and video |

## Overrides

Change a value with its full dotted path, for example
`trainer.max_iterations=10`. Prefix a key the config does not define with `+`,
as in `+runtime.precision=float16`. Quote lists and mappings for the shell.
Keys a config omits take their defaults, which `--cfg job --resolve` shows.

A checkpoint is tied to its model and its observation and action layout, so
change those only when starting a new run.

## Compilation and CUDA graphs

`runtime.compile` turns on `torch.compile`, and `runtime.compile_mode` selects
its mode. `act_cuda_graph`, `update_cuda_graph` and `shuffle_cuda_graph` capture
action sampling, the PPO update and minibatch shuffling as CUDA graphs. Playback
has its own `play.use_cuda_graph` setting.

### Reduced precision

`runtime.precision` runs the policy network in `float32` (the default),
`bfloat16` or `float16`; the master weights, gradients, probabilities and
simulator data stay in FP32. With `float16`, the loss is scaled dynamically:
`runtime.loss_scale` sets the starting scale (default 128), steps with
non-finite gradients are skipped and halve the scale, and checkpoints save the
scaler state. To try it on a config that does not set it:

```bash
uv run train --config-name humanoid/imitation/walk_pd \
  +runtime.precision=float16 +runtime.loss_scale=128
```

`runtime.mixed_precision` is an older, deprecated setting that applies autocast
to the PPO update only; `runtime.precision` takes precedence.

## Data paths

Paths are relative to the repository root. Humanoid models are under
`data/humanoid/mjcf/` and their reference motions under
`data/humanoid/motions/mimickit/`. The SNU model is
`data/snu_humanoid/mjcf/humanoid.xml`, with its includes and the pose-control
model next to it; they refer to each other and to their meshes by relative
path, so move them together.
