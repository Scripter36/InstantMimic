# Motion learning demonstrations

The [project page](https://scripter36.github.io/projects/instantmimic/#motion-demo)
replays the measured active training time of each policy before playing it.
The README preview shows the first four seconds of the same videos.

| Motion | Config | Seed | Training iterations | Active training time |
| --- | --- | ---: | ---: | ---: |
| Walk | [walk_pd.yaml](https://github.com/Scripter36/InstantMimic/blob/HEAD/config/humanoid/imitation/walk_pd.yaml) | 42 | 70 | 1.136 s |
| Run | [run_pd.yaml](https://github.com/Scripter36/InstantMimic/blob/HEAD/config/humanoid/imitation/run_pd.yaml) | 43 | 55 | 0.878 s |
| Spinkick | [spinkick_pd.yaml](https://github.com/Scripter36/InstantMimic/blob/HEAD/config/humanoid/imitation/spinkick_pd.yaml) | 43 | 75 | 1.238 s |
| Backflip | [backflip_pd.yaml](https://github.com/Scripter36/InstantMimic/blob/HEAD/config/humanoid/imitation/backflip_pd.yaml) | 43 | 120 | 1.933 s |
| Cartwheel | [cartwheel_pd.yaml](https://github.com/Scripter36/InstantMimic/blob/HEAD/config/humanoid/imitation/cartwheel_pd.yaml) | 43 | 85 | 1.363 s |

These are individual runs on an NVIDIA GeForce RTX 5090, not multi-seed medians.
Run, spinkick, backflip and cartwheel show the first checkpoint that passed the
evaluation below. Walk first passed at iteration 65 (1.056 s); the video uses
iteration 70. The times were measured on a pre-release revision of this code;
the released code can produce different training times and learning curves.

## Measurement

Training uses 2,048 environments, six-step rollouts, and the graph, compile, and
precision settings authored in the configs. The shared settings are in
[base.yaml](https://github.com/Scripter36/InstantMimic/blob/HEAD/config/humanoid/imitation/base.yaml).
Active training time includes rollout collection, PPO updates, and normal
training overhead. It excludes initialization, compile setup, benchmark
evaluation, checkpoint saving, and recording.

Every five training iterations, the policy is evaluated in 32 deterministic
worlds for 300 control steps (10 simulated seconds). Success requires every
world to complete that horizon without early termination and with valid
tracking metrics. The detection time has the resolution of this evaluation
interval; it is not a measurement of the earliest possible successful
intermediate policy.

## Videos

Each selected policy completes a separate 10-second playback without early
termination. Simulation states from that playback are rendered with the
Filament studio appearance at 1440 by 960 pixels and 60 frames per second.
The translucent green figure is the reference motion, shifted 1.2 m behind the
policy for comparison. The cartwheel uses the hand-contact model
`humanoid_pd_hands.xml` selected by its config. Cameras follow horizontal motion
with a smoothed-velocity lookahead and a fixed target height.

The carousel plays five seconds per motion after replaying the training time;
the full 10-second videos are linked on the project page. No training runs in
the visitor's browser.
