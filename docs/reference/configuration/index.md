# Configuration

Every run is described by one Hydra config under `config/`. A task config is
usually short: it lists shared files in its `defaults` and overrides only what
differs. For example, `config/humanoid/imitation/walk_pd.yaml` builds on the
shared humanoid settings in `base.yaml` and sets its own motion (abridged):

```yaml
# @package _global_
defaults:
  - base
  - _self_

env:
  motion:
    source: data/humanoid/motions/mimickit/humanoid_walk_sanitized.pkl
experiment:
  directory: humanoid/imitation/walk_pd
```

Because values come from several files, print the resolved config to see what
a run will actually use:

```bash
uv run train --config-name humanoid/imitation/walk_pd --cfg job --resolve
```

## Sections

| Section | Contents | Page |
| --- | --- | --- |
| `env` | Simulation, motion, observations, rewards, resets and termination | [Environment](environment.md) |
| `env.action`, `env.muscle` | Action types, muscle model and MASS | [Actions and muscles](actions-and-muscles.md) |
| `models` | Actor and critic networks and the action distribution | [Models](models.md) |
| `agent` | PPO rollouts, losses and optimizer | [PPO](ppo.md) |
| `trainer` | Iteration count, benchmark evaluation, stopping and resuming | [Training and playback](training-and-playback.md) |
| `experiment` | Run directory, logger and checkpoints | [Training and playback](training-and-playback.md) |
| `runtime` | Compilation, CUDA graphs and precision | [Training and playback](training-and-playback.md) |
| `play` | Playback, viewer, camera and video | [Training and playback](training-and-playback.md) |

The top level also holds `schema_version` (always `1`), `seed` (`-1` picks a
random seed, which is then saved with the run) and `task`, the Gymnasium
environment to create, such as `SNU-Imitation-v0`.

## Overrides

Change any value from the command line with its full dotted path, and prefix a
key with `+` to add one that the config does not define:

```bash
uv run train --config-name humanoid/imitation/walk_pd \
  trainer.max_iterations=300 +experiment.group=walk-sweep
```

Quote lists and mappings for the shell, for example
`'env.spawn_offset=[0,0,0.05]'`.

## Units

- Training lengths and intervals (`trainer.max_iterations`,
  `trainer.benchmark.interval`, checkpoint intervals) count **iterations**. One
  PPO iteration collects `agent.rollouts` control steps from every environment
  and then updates the policy, so `num_envs: 2048` with `rollouts: 6` collects
  12,288 transitions per iteration.
- Episode, benchmark and playback lengths count **control steps** unless the
  key ends in `_s` or `_seconds`.
- Physical values are in SI units (metres, seconds, kilograms, radians); keys in
  degrees end in `_deg`.
- A range is written as a two-element list, `[min, max]`, with both ends
  included.

## Validation

When a run starts, the config is checked before anything large is allocated.
Unknown keys, values out of range, incompatible combinations (for example a
compile mode that conflicts with the CUDA graph settings) and PPO batches that
do not divide evenly into minibatches are reported as errors. The `env` section
is checked by the environment itself when it is created.

To check every config in the repository without starting a run:

```bash
uv run python -m instantmimic.tools.validate_configs --allow-unprepared-assets --json
```

## Saved configs

Each run saves its resolved config and model files in `run_bundle/` next to its
checkpoints. `play` rebuilds the environment, model and agent from that bundle,
so a checkpoint keeps playing even if the configs in `config/` change later.
Overrides under `play.*` change playback only; overriding other sections at
playback time changes the restored config and prints a warning.

Training from a checkpoint with `trainer.initialization` uses the config given
on the command line, not the bundle. To resume a run, pass the same task config
it was started with.
