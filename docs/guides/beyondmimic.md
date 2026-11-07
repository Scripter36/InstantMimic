# BeyondMimic tracking

This task is a port of the motion-tracking stage of
[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) for the
29-DoF Unitree G1, taken from the
[mjlab tracking task](https://github.com/mujocolab/mjlab/tree/15ebce8840ee2205f4c62d0c20df65dd1794cab4/src/mjlab/tasks/tracking)
at a pinned revision. BeyondMimic's diffusion and deployment stages are not
included.

The policy runs at 50 Hz on a 200 Hz simulation. The actor sees 160
observations and the critic 286, including privileged ones. Rewards,
termination, actions, model randomization and adaptive motion sampling follow
mjlab.

## Prepare the G1 model and motion

BeyondMimic's code now runs on mjlab, which holds the G1 model. BeyondMimic
does not publish its side-kick motion. It credits KungfuBot, whose code,
[PBHC](https://github.com/TeleHuman/PBHC), ships `Side_kick.pkl`. Clone
both at the pinned revisions:

```bash
mkdir -p external
git clone https://github.com/mujocolab/mjlab external/mjlab
git -C external/mjlab checkout 15ebce8840ee2205f4c62d0c20df65dd1794cab4
git clone --filter=blob:none https://github.com/TeleHuman/PBHC external/PBHC
git -C external/PBHC checkout ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0
```

Then convert them inside mjlab's environment. This needs a GPU, and the first
run installs mjlab's dependencies:

```bash
beyondmimic_checkout="$PWD"
uv run --project external/mjlab --frozen --no-dev --extra cu128 --with joblib \
  python "$beyondmimic_checkout/src/instantmimic/tools/prepare_beyondmimic.py" \
  --mjlab-source "$beyondmimic_checkout/external/mjlab" \
  --side-kick-pkl "$beyondmimic_checkout/external/PBHC/example/motion_data/Side_kick.pkl" \
  --output-dir "$beyondmimic_checkout/data/unitree_g1/beyondmimic"
```

The script verifies the mjlab revision and the motion file's SHA-256, then
writes `mjcf/scene.xml` and `motions/converted/side_kick_motion.npz`. It refuses to
overwrite existing output that differs. The side kick is licensed CC BY-NC 4.0;
see `data/unitree_g1/PROVENANCE.md`.

To validate the configs before converting, pass `--allow-unprepared-assets`:

```bash
uv run python -m instantmimic.tools.validate_configs --allow-unprepared-assets --json
```

## Train and replay

```bash
uv run train --config-name unitree_g1/beyondmimic/side_kick
uv run play --config-name unitree_g1/beyondmimic/side_kick \
  +play.checkpoint='logs/ppo/unitree_g1/beyondmimic/side_kick/YOUR_RUN/checkpoints/best_agent.pt' \
  play.video.enabled=false play.steps=300
```

Training runs 6,000 iterations with 4,096 environments.

| Config | Purpose |
| --- | --- |
| `side_kick` | Default, with fused Warp kernels and CUDA graphs |
| `side_kick_hpo` | Training settings found by [HPO](#tune-the-training-settings-with-hpo) |
| `side_kick_reference` | Unfused port that keeps mjlab's structure |

A resumed run restores the policy, optimizer and observation normalizer but
restarts episodes and motion-sampling statistics.

## Evaluate tracking

The evaluator runs 1,024 parallel episodes with the policy's mean actions and
reports mjlab's tracking metrics:

```bash
uv run python -m instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking.evaluate \
  --checkpoint logs/ppo/unitree_g1/beyondmimic/side_kick/YOUR_RUN/checkpoints/best_agent.pt \
  --seed 42 --output tracking-evaluation.json
```

## How this port was made

A coding agent ported the task under a contract we wrote, and we reviewed its
work. It had to match BeyondMimic and the pinned mjlab task, keep that match on the
CUDA-graph path, and pass a fixed tracking benchmark.

The agent first built `side_kick_reference`, which keeps mjlab's structure for
term-by-term checks. `side_kick` then fuses the task into Warp kernels and runs
a training iteration 4.9× faster than the reference. To match mjlab's PPO, the agent added options
such as privileged critic observations and RSL-RL's observation normalization.

Tracking over three seeds, averaged over iterations 5,000 to 6,000:

| Metric | InstantMimic | mjlab |
| --- | ---: | ---: |
| Success rate | 0.968 | 0.985 |
| MPKPE | 0.110 m | 0.106 m |
| Root-relative MPKPE | 0.031 m | 0.030 m |
| End-effector position error | 0.048 m | 0.046 m |
| End-effector orientation error | 0.172 rad | 0.167 rad |

Training throughput on an RTX 5090, in thousands of environment steps per
second:

| Compared with | Baseline | InstantMimic | Speedup |
| --- | ---: | ---: | ---: |
| [BeyondMimic on mjlab 1.6.0](https://github.com/mujocolab/mjlab/tree/v1.6.0/src/mjlab/tasks/tracking) | 245 | 541 | 2.2× |
| [BeyondMimic on Isaac Lab 2.1.0](https://github.com/HybridRobotics/whole_body_tracking/tree/cd65172032893724b445448818c34165846d847d) | 118 | 541 | 4.6× |

All runs use 4,096 environments, 24-step rollouts, 5 epochs and 4 minibatches,
timed over iterations 100 to 199.

Reviews of this port shaped the
[`migrate-external-env`](assistant-workflows.md#port-an-environment-from-another-framework)
skill. They caught a normalizer that the fused rollout never updated, and
kernels that reused one random draw for reset, push and observation noise.

## Tune the training settings with HPO

After the port passed its parity checks, we searched its training settings with
[`run_hpo`](hpo.md#beyondmimic-side-kick). The search changes only the optimizer,
PPO and network settings. Each trial trains for 60 seconds of active time on
three seeds, and the search keeps the trials that balance final reward against
reward over the run. The selected trial became the `side_kick_hpo` config: hybrid
Muon/AdamW, 16-step rollouts, 3 epochs, a 640/320/160 critic and retuned PPO
values.

```bash
uv run train --config-name unitree_g1/beyondmimic/side_kick_hpo
```

A 16-step rollout collects two thirds of the environment steps of a 24-step
one, so each iteration is also cheaper. Together with the port's higher
throughput, the tuned settings reach mjlab's final tracking error 4.8× sooner than mjlab:

<figure class="im-figure">
<svg viewBox="0 0 680 330" role="img" aria-label="MPKPE against active training time for mjlab and two InstantMimic configurations">
<defs><clipPath id="im-bm-clip-docs"><rect x="64" y="12" width="600" height="258"/></clipPath></defs>
<line class="plot-grid" x1="64" x2="664" y1="229.3" y2="229.3"/>
<text class="plot-tick" x="54" y="233.3" text-anchor="end">100</text>
<line class="plot-grid" x1="64" x2="664" y1="174.9" y2="174.9"/>
<text class="plot-tick" x="54" y="178.9" text-anchor="end">120</text>
<line class="plot-grid" x1="64" x2="664" y1="120.6" y2="120.6"/>
<text class="plot-tick" x="54" y="124.6" text-anchor="end">140</text>
<line class="plot-grid" x1="64" x2="664" y1="66.3" y2="66.3"/>
<text class="plot-tick" x="54" y="70.3" text-anchor="end">160</text>
<line class="plot-grid" x1="64" x2="664" y1="12.0" y2="12.0"/>
<text class="plot-tick" x="54" y="16.0" text-anchor="end">180</text>
<line class="plot-axis" x1="64.0" x2="64.0" y1="270" y2="275"/>
<text class="plot-tick" x="64.0" y="292" text-anchor="middle">0</text>
<line class="plot-axis" x1="206.9" x2="206.9" y1="270" y2="275"/>
<text class="plot-tick" x="206.9" y="292" text-anchor="middle">10</text>
<line class="plot-axis" x1="349.7" x2="349.7" y1="270" y2="275"/>
<text class="plot-tick" x="349.7" y="292" text-anchor="middle">20</text>
<line class="plot-axis" x1="492.6" x2="492.6" y1="270" y2="275"/>
<text class="plot-tick" x="492.6" y="292" text-anchor="middle">30</text>
<line class="plot-axis" x1="635.4" x2="635.4" y1="270" y2="275"/>
<text class="plot-tick" x="635.4" y="292" text-anchor="middle">40</text>
<path class="plot-axis" fill="none" d="M64,12 V270 H664"/>
<text class="plot-label" x="364.0" y="320" text-anchor="middle">Active training time on an RTX 5090 (min)</text>
<text class="plot-label" transform="translate(18 141.0) rotate(-90)" text-anchor="middle">MPKPE (mm)</text>
<line class="plot-threshold" x1="64" x2="664" y1="209.7" y2="209.7"/>
<g clip-path="url(#im-bm-clip-docs)" fill="none" stroke-width="2.5" stroke-linejoin="round">
<polyline class="im-s-mjlab" points="112.5,-967.4 160.8,-99.4 209.0,16.3 257.5,76.9 305.3,127.8 353.0,146.8 400.7,180.7 448.3,181.1 496.0,193.3 543.6,202.7 591.2,207.4 638.7,218.9"/>
<polyline class="im-s-baseline" points="85.6,-678.5 107.3,-241.7 128.8,2.4 150.3,83.9 171.8,123.3 193.3,142.6 214.7,167.7 236.0,176.6 257.4,196.5 278.8,206.4 300.2,205.0 321.7,214.1"/>
<polyline class="im-s-hpo" points="78.5,-292.0 92.8,63.4 107.1,148.6 121.5,184.8 135.8,197.3 150.2,191.8 164.6,190.8 178.9,215.5 193.3,213.9 207.8,217.4 222.1,235.4 236.5,224.7"/>
</g>
<line class="im-s-mjlab" stroke-width="1.5" stroke-dasharray="2 3" x1="600.5" x2="600.5" y1="209.7" y2="270"/>
<circle class="im-s-mjlab im-fill" cx="600.5" cy="209.7" r="4.5" stroke-width="1.5" style="stroke:var(--im-surface,#fff)"/>
<line class="im-s-baseline" stroke-width="1.5" stroke-dasharray="2 3" x1="311.2" x2="311.2" y1="209.7" y2="270"/>
<circle class="im-s-baseline im-fill" cx="311.2" cy="209.7" r="4.5" stroke-width="1.5" style="stroke:var(--im-surface,#fff)"/>
<line class="im-s-hpo" stroke-width="1.5" stroke-dasharray="2 3" x1="175.5" x2="175.5" y1="209.7" y2="270"/>
<circle class="im-s-hpo im-fill" cx="175.5" cy="209.7" r="4.5" stroke-width="1.5" style="stroke:var(--im-surface,#fff)"/>
<path class="im-figure__span" d="M178.5,239.7 V243.7 H308.2 V239.7"/>
<text class="im-figure__gain" x="243.3" y="259.7" text-anchor="middle"><tspan>2.2×</tspan> HPO</text>
<path class="im-figure__span" d="M314.2,239.7 V243.7 H597.5 V239.7"/>
<text class="im-figure__gain" x="455.8" y="259.7" text-anchor="middle"><tspan>2.2×</tspan> throughput</text>
</svg>
<div class="im-figure__legend"><span><i class="im-s-mjlab" aria-hidden="true"></i><span>mjlab with RSL-RL PPO</span><strong>37 min 33 s</strong></span><span><i class="im-s-baseline" aria-hidden="true"></i><span>InstantMimic with mjlab's PPO settings (<code>side_kick</code>)</span><strong>17 min 18 s</strong></span><span><i class="im-s-hpo" aria-hidden="true"></i><span>InstantMimic with HPO-tuned settings (<code>side_kick_hpo</code>)</span><strong>7 min 48 s</strong></span><span><i class="im-figure__threshold" aria-hidden="true"></i><span>mjlab final MPKPE, 107 mm</span></span></div>
<figcaption>Mean of seeds 42 and 43. Each time is when a run first reaches mjlab's mean MPKPE over iterations 5,000–6,000.</figcaption>
</figure>

The search scored each trial on 60 seconds of training, so we validated the
selected settings on full training runs:

- **Full training.** We trained three seeds of each config for 6,000
  iterations and evaluated 1,024 episodes at iterations 5,000, 5,500 and 6,000.
  Success rose from 0.968 to 0.984 and MPKPE fell from 110 mm to 100 mm, on
  every seed.
- **Robustness.** With the training pushes enabled during evaluation, MPKPE
  fell from 137 mm to 129 mm. From random start frames, it fell from 106 mm to
  102 mm. The action rate was 2–4 % higher.
- **Sim-to-sim transfer.** Evaluated without retraining in BeyondMimic's Isaac
  Lab implementation, success was 0.959 for `side_kick_hpo` and 0.917 for
  `side_kick`.

These results cover one motion clip trained for 6,000 iterations. Upstream
BeyondMimic trains for 30,000.
