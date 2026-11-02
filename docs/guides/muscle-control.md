# Muscle simulation and control

This guide covers the muscle-actuated characters and the experiments of the
MUSUBI paper: MUSUBI itself, the MASS baseline and the direct-control baselines.

## Models and motions

| Family | Models | Motions |
| --- | --- | --- |
| SNU humanoid | Full-body musculoskeletal humanoid, with hand and self-collision variants | Walk, run and backflip clips captured by our group |
| FreeMusco | Ostrich and Chimanoid, with variants that split each muscle into two or four actuators | Reference motion arrays from FreeMusco |

## MUSUBI

The MUSUBI task configs, models and motions are included. To see how a task is
set up, print its resolved config:

```bash
uv run train --config-name snu_humanoid/imitation/walk_musubi --cfg job --resolve
uv run train --config-name freemusco/run/ostrich/musubi --cfg job --resolve
uv run train --config-name freemusco/experiments/split_geometry/run/ostrich/musubi_split4 --cfg job --resolve
```

## MASS baseline

MASS ([Lee et al. 2019, *Scalable Muscle-actuated Human Simulation and Control*](https://doi.org/10.1145/3306346.3322972))
splits control between a policy and a muscle coordinator network, MuscleNN. The
policy produces a joint-torque target; MuscleNN converts it into muscle
activations, and the resulting muscle forces drive the body. PPO trains the
policy. MuscleNN is trained in the same loop to make the torque produced by its
activations match the target, with a regularization term on the activations.

| Configs | Policy output | MuscleNN target |
| --- | --- | --- |
| `snu_humanoid/imitation/mass/{walk,run,backflip}_mass` | Residual added to the reference pose | StablePD torque toward that pose, with gains from `humanoid_pd.xml` |
| `snu_humanoid/imitation/mass/{walk,run,backflip}_mass_torque` | Joint torque | The policy's torque |
| `freemusco/imitation/ostrich/single/mass` | Joint torque | The policy's torque |

```bash
uv run train --config-name snu_humanoid/imitation/mass/walk_mass
uv run train --config-name snu_humanoid/imitation/mass/walk_mass_torque
```

The SNU configs use `data/snu_humanoid/mjcf/humanoid.xml` with its body and
muscle includes and the bundled `walk.bvh`, `run.bvh` and `backflip.bvh`
motions. MuscleNN's network size, optimizer, batch size, epochs and loss weights
are set under `env.action.muscle_nn`.

### Monitor and replay

Training logs `Loss / MuscleNN target loss` and
`Loss / MuscleNN regularization loss` alongside the policy reward and
tracking metrics. Checkpoints store MuscleNN with the policy, so playback and
resume restore both:

```bash
uv run play --config-name snu_humanoid/imitation/mass/walk_mass \
  +play.checkpoint='logs/YOUR_RUN/checkpoints/best_agent.pt' \
  play.video.enabled=false play.steps=300
```

See [simulation details](#simulation-details) for how the muscle forces are
applied.

## Direct-control baselines

The paper's direct-control baselines are ordinary configs:

| Baseline | SNU humanoid | FreeMusco |
| --- | --- | --- |
| Activation output (One-Level) | `snu_humanoid/imitation/{walk,run,backflip}_activation` | `freemusco/run/{ostrich,chimanoid}/activation`, `freemusco/imitation/{ostrich,chimanoid}/single/activation`, split-geometry `activation_split{2,4}` |
| Unconstrained torque | `snu_humanoid/imitation/torque/{walk,run,backflip}_unconstrained_torque` | |
| Muscle-bounded torque | `snu_humanoid/imitation/torque/{walk,run,backflip}_clipped_torque` | |
| MASS torque control | `snu_humanoid/imitation/mass/{walk,run,backflip}_mass_torque` | `freemusco/imitation/ostrich/single/mass` |

```bash
uv run train --config-name snu_humanoid/imitation/walk_activation
uv run train --config-name freemusco/run/ostrich/activation
uv run train --config-name snu_humanoid/imitation/torque/walk_clipped_torque
```

- **Activation output** (`env.action.type=activation`): the policy outputs one
  excitation per muscle, clamped to $[0, 1]$, and the muscles apply the
  resulting active and passive forces. `agent.activation_regularization` adds
  `weight * mean((mu - target)^2)` on the policy mean `mu` to the PPO loss
  (logged as `Loss / Policy auxiliary loss`); the activation presets use weight
  0.1 and target 0.
- **Unconstrained torque**: the policy outputs joint torques directly, without
  muscle limits or passive muscle forces. The muscles are used only to scale the
  action range at startup.
- **Muscle-bounded torque**: joint torques are clipped to the range the muscles
  can produce in the current pose, and passive muscle forces are added.

These configs run on the released code. The paper's results came from an
earlier version whose rewards, termination and network details differ in
places, so expect the same setup rather than identical numbers.

## Simulation details

Each muscle follows a path of waypoints attached to bodies, optionally wrapping
around fixed spheres and cylinders (the Ostrich model uses these). Forces,
muscle observations and rendered tendons all use the path at the current pose.
FreeMusco configs can also use MuJoCo's native actuators for comparison.

MASS pose control turns off MuJoCo's own actuators and applies its StablePD
torque as an external force; joint damping is still integrated implicitly (see
[differences from stock MuJoCo Warp](../reference/simulation.md#differences-from-stock-mujoco-warp)).
With `muscle.force_velocity.implicit=true`, the muscles' force-velocity damping
is integrated implicitly as well, within the sparsity of the mass matrix; the
number of terms that do not fit is printed at startup.

## LAFAN configs

The SNU configs under `snu_humanoid/lafan/` use LAFAN1 clips retargeted to the
SNU humanoid. LAFAN1 is not redistributed here, so these configs need locally
prepared files; `data/runtime-assets.json` lists the expected paths.
