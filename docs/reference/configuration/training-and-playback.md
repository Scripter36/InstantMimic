# Training and playback

This page covers `trainer` (the training loop), `experiment` (where results go),
`runtime` (how the GPU work is executed) and `play` (the `play` command).
The humanoid walk, `humanoid/imitation/walk_pd`, sets (abridged):

```yaml
trainer:
  max_iterations: 150
  benchmark:
    interval: 5
    steps: 300
    num_envs: 32
    objective:
      metric: reward_mean
      direction: maximize
experiment:
  directory: humanoid/imitation/walk_pd
  logger:
    type: ${oc.env:INSTANTMIMIC_LOGGER,tensorboard}
  write_interval: 1
  episode_metrics_interval: 1
  checkpoint:
    periodic:
      interval: 50
    best:
      source: auto
runtime:
  compile: true
  compile_update_fullgraph: true
  act_cuda_graph: true
  update_cuda_graph: true
  shuffle_cuda_graph: true
  precision: float32
```

## Benchmark evaluation

`trainer.benchmark` evaluates the policy during training in a separate set of
environments, using the policy's mean action.

| Key | Meaning |
| --- | --- |
| `interval` | Evaluate every this many iterations (or use `interval_seconds` of active training time instead) |
| `steps`, `num_envs` | Length of each evaluation in control steps, and number of environments |
| `deterministic_reset` | Start every evaluation from the same states |
| `termination.type` | `inherit` the training termination, `disabled`, or `phc_tracking` with a `distance` |
| `objective` | The metric that defines the best policy, and whether to `maximize` or `minimize` it |
| `thresholds` | Pass criteria such as `success_rate`, `ape_mean` or `consecutive_passes`, used to stop training early |
| `execution` | `inline` pauses training to evaluate, and `overlap` evaluates while training continues |
| `save_history` | Append every result to a history file in the run directory |

Benchmark time is not counted as active training time.

For multi-clip datasets, `trainer.clip_failure_benchmark` instead scans each
clip for where the policy fails. A run uses one of the two.

## Stopping, resuming and videos

| Key | Meaning |
| --- | --- |
| `trainer.max_iterations` | Number of training iterations |
| `trainer.stopping` | Stop early: `type: benchmark` when the benchmark thresholds pass, `reward` or `episode_length` when the training average reaches `threshold`, or `time_budget` after `timeout_seconds` of active training (with `warmup_iterations: 0`) |
| `trainer.initialization` | Start from a checkpoint: `type: resume` continues a run, `type: policy_init` starts a new run from a policy's weights. `checkpoint` is its path |
| `trainer.recording.periodic` | Record a video every `interval` iterations, `length` steps long |
| `trainer.post_training_playback` | Run playback, optionally with `video`, when training ends |

## Run directory, logging and checkpoints

| Key | Meaning |
| --- | --- |
| `experiment.directory` | Runs are written to `logs/ppo/<directory>/<start time>_ppo_custom_<name>/` |
| `experiment.name` | Optional run name, by default the config name |
| `experiment.group` | Group name passed to W&B and MLflow |
| `experiment.logger` | `type: tensorboard`, `wandb` or `mlflow`, with backend settings (see [experiment logging](../../getting-started/installation.md#experiment-logging)) |
| `experiment.write_interval` | Log training metrics every this many iterations |
| `experiment.episode_metrics_interval` | Log episode statistics every this many iterations |
| `experiment.checkpoint.periodic.interval` | Save a full checkpoint every this many iterations |
| `experiment.checkpoint.best` | Keep `best_agent.pt` up to date: `source: auto` uses the benchmark objective when there is a benchmark and the training reward otherwise. `interval` sets how often, in iterations, a new best is written to disk |
| `experiment.checkpoint.disabled` | Save no checkpoints |

## Runtime

| Key | Meaning |
| --- | --- |
| `compile`, `compile_mode` | Use `torch.compile` for the policy and the update, and its mode (default `max-autotune-no-cudagraphs`) |
| `compile_update_fullgraph` | Compile the whole PPO update as one graph. Needs `compile` and `update_cuda_graph` |
| `act_cuda_graph`, `update_cuda_graph`, `shuffle_cuda_graph` | Capture action sampling, the PPO update and minibatch shuffling as CUDA graphs. `shuffle_cuda_graph` needs `update_cuda_graph` and an `agent.shuffle_mode` other than `off` |
| `precision`, `loss_scale` | Policy network precision (see below) |
| `fail_on_nonfinite_update` | Stop if an update produces non-finite values |

`compile_mode: reduce-overhead` and `max-autotune` let PyTorch capture its own
CUDA graphs, so they cannot be combined with `act_cuda_graph` or
`update_cuda_graph`. The GPU is chosen by the process environment, for example
`CUDA_VISIBLE_DEVICES=1 uv run train ...`.

### Reduced precision

`runtime.precision` runs the policy network in `float32` (the default),
`bfloat16` or `float16`, during both rollouts and updates. The master weights,
gradients, probabilities and everything exchanged with the simulator stay in
FP32. With `float16`, the loss is scaled dynamically on the GPU:
`runtime.loss_scale` sets the starting scale (default 128, at most $2^{24}$), a
step with non-finite gradients is skipped and halves the scale, and after 2,000
finite steps the scale doubles again. Checkpoints save the scaler state, and
CUDA graphs and compilation work at every precision. The included configs use
`float32`. To train one in half precision:

```bash
uv run train --config-name humanoid/imitation/walk_pd \
  runtime.precision=float16 +runtime.loss_scale=128
```

## Playback

`play` restores a run from its checkpoint and saved config. Its settings only
affect playback:

| Key | Meaning |
| --- | --- |
| `play.checkpoint` | The checkpoint to load (add it with `+play.checkpoint=...`) |
| `play.steps` | Number of control steps. Without it and without video, playback runs interactively |
| `play.num_envs` | Number of environments to play |
| `play.time_limit_policy` | `extend` lengthens the episode limit to cover `play.steps`, and `respect` keeps the training limit |
| `play.done_policy` | When an episode ends: `reset_continue` or `stop` |
| `play.action.mode` | `mean` (the default for evaluation) or `sample` |
| `play.tracking` | Playback-only motion choice, start frame, deterministic reset and termination |
| `play.viewer` | Interactive viewer: `backend: mujoco` or `viser` |
| `play.camera` | Camera distance, azimuth and elevation, or several `views` (see [Video rendering](../../guides/rendering.md)) |
| `play.video` | `enabled`, `length`, `render_backend` (`auto`, `mujoco_gl` or `warp`), `encoder_backend` and `tile_envs` |
| `play.use_cuda_graph` | Capture the playback step as a CUDA graph |
