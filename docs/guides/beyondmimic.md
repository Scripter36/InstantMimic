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
| `side_kick_reference` | The same task without fused kernels, for comparison |
| `side_kick_native_ppo` | InstantMimic's PPO settings instead of mjlab's |
| `side_kick_native_ppo_muon` | As above, with the hybrid Muon optimizer |
| `side_kick_eval` | Deterministic resets and commands for playback |

Resuming restores the policy, optimizer and observation normalization, but the
episodes and adaptive motion-sampling statistics start over, so the resumed run
does not follow the exact same trajectory.

## Evaluate tracking

The evaluator measures tracking with mjlab's metrics, running one episode per
world with the policy's mean actions:

```bash
uv run python -m instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.evaluate \
  --checkpoint logs/ppo/unitree_g1/beyondmimic/side_kick/YOUR_RUN/checkpoints/best_agent.pt \
  --seed 42 --output tracking-evaluation.json
```

It runs 1,024 episodes by default; `--episodes 128` gives a quicker estimate.

## Check configs before preparing the assets

`validate_configs` normally requires every asset a config refers to. With
`--allow-unprepared-assets`, the G1 files declared in `data/runtime-assets.json`
are reported as `requires-preparation` instead, so the configs can be checked
before you run the conversion:

```bash
uv run python -m instantmimic.tools.validate_configs --allow-unprepared-assets --json
```
