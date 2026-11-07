# BeyondMimic tracking

This task reproduces the motion-tracking stage of
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) for the
29-DoF Unitree G1, following the
[mjlab tracking implementation](https://github.com/mujocolab/mjlab/tree/15ebce8840ee2205f4c62d0c20df65dd1794cab4/src/mjlab/tasks/tracking)
at a fixed revision. The later diffusion and deployment stages of BeyondMimic
are not included.

The policy runs at 50 Hz on a 200 Hz simulation. The actor sees 160
observations and the critic 286, including privileged ones. Rewards,
termination, actions relative to the default pose, model randomization and
adaptive motion sampling all follow mjlab's task.

## Prepare the G1 model and motion

The G1 model and the side-kick motion come from other repositories, so you
prepare them once with a conversion script. Clone both at the revisions used
here:

```bash
mkdir -p external
git clone https://github.com/mujocolab/mjlab external/mjlab
git -C external/mjlab checkout 15ebce8840ee2205f4c62d0c20df65dd1794cab4
git clone --filter=blob:none https://github.com/TeleHuman/PBHC external/PBHC
git -C external/PBHC checkout ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0
```

Then run the conversion inside mjlab's own environment. It needs a GPU, and the
first run installs mjlab's dependencies:

```bash
beyondmimic_checkout="$PWD"
uv run --project external/mjlab --frozen --no-dev --extra cu128 --with joblib \
  python "$beyondmimic_checkout/src/instantmimic/tools/prepare_beyondmimic.py" \
  --mjlab-source "$beyondmimic_checkout/external/mjlab" \
  --side-kick-pkl "$beyondmimic_checkout/external/PBHC/example/motion_data/Side_kick.pkl" \
  --output-dir "$beyondmimic_checkout/data/unitree_g1/beyondmimic"
```

The script checks the mjlab revision and the motion file's SHA-256, then writes
the scene (`mjcf/scene.xml`) and the 50 Hz motion
(`motions/converted/side_kick_motion.npz`) to the output directory. It never
overwrites different existing output. The side kick is from PBHC under
CC BY-NC 4.0; see `data/unitree_g1/PROVENANCE.md` for the sources and their
terms. The prepared files are ignored by Git.

## Train and replay

```bash
uv run train --config-name unitree_g1/beyondmimic/side_kick
uv run play --config-name unitree_g1/beyondmimic/side_kick \
  +play.checkpoint='logs/ppo/unitree_g1/beyondmimic/side_kick/YOUR_RUN/checkpoints/best_agent.pt' \
  play.video.enabled=false play.steps=300
```

Training runs 6,000 iterations with 4,096 environments by default. Other configs
in the same directory:

| Config | Purpose |
| --- | --- |
| `side_kick` | The default: fused Warp task kernels, CUDA graphs and compiled PPO updates |
| `side_kick_hpo` | Opt-in training settings selected by the 60 s HPO search |
| `side_kick_reference` | The same task without fused kernels, for comparison |
| `side_kick_native_ppo` | InstantMimic's PPO settings instead of mjlab's |
| `side_kick_native_ppo_muon` | As above, with the hybrid Muon optimizer |
| `side_kick_eval` | Deterministic resets and commands for playback |

Resuming restores the policy, optimizer and observation normalization, but the
episodes and adaptive motion-sampling statistics start over, so the resumed run
does not follow the exact same trajectory.

## Use the HPO configuration

The opt-in `side_kick_hpo` preset inherits `side_kick` and applies the training
settings selected by the
[60-second HPO search](hpo.md#beyondmimic-side-kick): hybrid Muon/AdamW,
16-step rollouts, three learning epochs, a 640/320/160 critic and the tuned PPO
and learning-rate values. The task, actor network, runtime and training length
are unchanged:

```bash
uv run train --config-name unitree_g1/beyondmimic/side_kick_hpo \
  experiment.name=side_kick_hpo
```

## Evaluate tracking

The evaluator measures tracking with mjlab's metrics, running one episode per
world with the policy's mean actions:

```bash
uv run python -m instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.evaluate \
  --checkpoint logs/ppo/unitree_g1/beyondmimic/side_kick/YOUR_RUN/checkpoints/best_agent.pt \
  --seed 42 --output tracking-evaluation.json
```

It runs 1,024 episodes by default; `--episodes 128` gives a quicker estimate.

## How this port was made

A coding agent ported this task under a contract that we wrote and reviewed
at every stage: reproduce the public Stage-1 tracking task, prove parity
against the official BeyondMimic code and the pinned mjlab implementation,
keep that parity on the optimized CUDA-graph path, and pass a fixed tracking
benchmark. The port has two stages:

1. A reference port that keeps mjlab's task structure, used to check parity
   term by term (`side_kick_reference`).
2. The production path, which fuses the task's observation, reward,
   termination and reset computations into Warp kernels and runs rollout and
   PPO updates as CUDA graphs (`side_kick`). On the same work it runs a training
   iteration 4.9× faster than the reference port.

Matching the learning setup took PPO options that InstantMimic did not have:
privileged critic observations, RSL-RL's empirical observation normalization,
its initialization order, a learned standard deviation trained directly rather
than as its logarithm, and learning-rate adaptation after every minibatch. They
are options that default off for other tasks. The direct standard deviation
matters: with InstantMimic's log-space parameterization instead, the same task
learned measurably slower over 500 iterations.

Three seeds trained for 6,000 iterations with the `side_kick` settings pass
every quality gate of the 1,024-episode evaluation and track about as well as
mjlab. Averaged over iterations 5,000, 5,500 and 6,000 and then over seeds:

| Metric | InstantMimic | mjlab (pinned revision) |
| --- | ---: | ---: |
| Success rate | 0.976 | 0.985 |
| MPKPE | 0.107 m | 0.106 m |
| Root-relative MPKPE | 0.030 m | 0.030 m |
| End-effector position error | 0.047 m | 0.046 m |
| End-effector orientation error | 0.170 rad | 0.167 rad |

The lowest success of any evaluated checkpoint is 0.958.

Training throughput on one RTX 5090, with 4,096 environments, 24-step rollouts
and the same PPO workload (5 epochs, 4 minibatches), counting both rollout
collection and PPO updates, in thousands of environment steps per second
(kFPS). The first row was measured on the current code; the other two on the
development branch before release.

| Comparison | Baseline (kFPS) | InstantMimic (kFPS) | Ratio |
| --- | ---: | ---: | ---: |
| [mjlab 1.6.0](https://github.com/mujocolab/mjlab), one seed, iterations 100–199 | 245 | 541 | 2.21× |
| [mjlab](https://github.com/mujocolab/mjlab/tree/15ebce8840ee2205f4c62d0c20df65dd1794cab4/src/mjlab/tasks/tracking) at the parity revision, three seeds | 247 | 470 | 1.90× |
| [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) on Isaac Lab 2.1.0, iterations 100–199 | 118 | 470 | 3.99× |

The first row is the closest match: both sides use MuJoCo Warp 3.16, InstantMimic
runs with its optimized simulator kernels and parallel kernel launch, and both
totals cover rollout collection and the update, each from its own framework's
timers, though PyTorch, Warp and TF32 settings differ. In the second row the
two sides differ in MuJoCo Warp revision and in throughput definition (mjlab's
final reported value, run under our measurement harness with profiling markers
enabled, against InstantMimic's active training time). The original BeyondMimic repository has since retired its Isaac Lab
implementation in favor of mjlab; its gap is in rollout collection, and its PPO
update takes about as long as the other two.

How the port was reviewed shaped the skill. The skill was written after this
port and revised with what later reviews caught: a public version of the port
that duplicated simulator work at half speed while still matching mjlab's
throughput, a normalizer that the fused rollout never updated, tolerance
checks set without measuring repeat noise, and random draws in the fused
kernels that all reused one value per environment, which made reset, push and
observation noise perfectly correlated and the task easier than mjlab's. Those
became the
[source-semantics and discriminating-evidence rules](assistant-workflows.md#port-an-environment-from-another-framework)
of `migrate-external-env`.

## Check configs before preparing the assets

`validate_configs` normally requires every asset a config refers to. With
`--allow-unprepared-assets`, the G1 files declared in `data/runtime-assets.json`
are reported as `requires-preparation` instead, so the configs can be checked
before you run the conversion:

```bash
uv run python -m instantmimic.tools.validate_configs --allow-unprepared-assets --json
```
