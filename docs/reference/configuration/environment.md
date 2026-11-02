# Environment

The `env` section belongs to the environment selected by `task`. This page
describes the motion-imitation environment (`SNU-Imitation-v0`) used by the
humanoid, SNU and AMASS imitation tasks. Tasks that select another
environment define their own `env` keys; print their resolved config to see
them. Actions and muscles are covered in
[Actions and muscles](actions-and-muscles.md).

The humanoid walk, for example, resolves to (abridged):

```yaml
env:
  sim:
    mjcf_path: data/humanoid/mjcf/humanoid_pd.xml
    hz: 120
    integrator: implicitfast
  scene:
    num_envs: 2048
  decimation: 4
  episode_length_s: 15.0
  motion:
    source: data/humanoid/motions/mimickit/humanoid_walk_sanitized.pkl
  observations:
    include_root_target: false
    joint_pose: 6d
    future_mode: phc_v6
    future_frames: 1
  reset:
    random_start: true
    motion_noise_std: 0.02
  reward:
    tracking:
      type: body_state
      frame: world
      weights: {position: 100.0, rotation: 5.0, linear_velocity: 0.1, angular_velocity: 0.1}
    combination:
      type: power_sum
      q: 2.0
  termination:
    min_torso_height: 0.8
    early:
      type: reward
      reward_threshold: 0.15
      consecutive_steps: 4
```

## Simulation

| Key | Meaning |
| --- | --- |
| `sim.mjcf_path` | The MuJoCo model |
| `sim.hz` | Physics rate in Hz |
| `decimation` | Physics steps per policy step; the policy runs at `hz / decimation` |
| `sim.integrator` | `implicitfast`, `euler` or `rk4` |
| `sim.njmax`, `sim.nconmax` | Constraint and contact capacity per environment |
| `sim.iterations`, `sim.ls_iterations` | Solver and line-search iterations |
| `scene.num_envs` | Number of parallel environments |
| `episode_length_s` | Maximum episode length in seconds |

`use_cuda_graph`, `use_parallelized_kernel_launch` and `use_optimized_kernel`
control how the simulation step is executed; see
[Simulation](../simulation.md#configuration).

## Motion

`motion.source` is a path, a glob pattern or a list of either, relative to the
repository root. MimicKit `.pkl` and `.npz`, BVH and HDF5 motion files are
supported. With several clips, each episode starts from one of them.

| Key | Meaning |
| --- | --- |
| `motion.scale`, `motion.offset` | Scale and translate the motion before use |
| `motion.bvh_map` | Maps BVH joint names to model body names |
| `motion.loop` | How a clip continues past its end. With `smooth: true`, the end is blended into the start over `blend_frames` frames, using `mode: tail_warp` (the default) or `c1_bridge` |
| `spawn_offset` | Offset added to the character's starting position |

## Observations

| Key | Meaning |
| --- | --- |
| `observations.joint_pose` | Joint rotation encoding: `qpos`, `6d` or `multi_hinge_local_quat` |
| `observations.joint_velocity` | Joint velocity encoding: `qvel` (the default), `hinge_local_angvel` or `multi_hinge_local_angvel` |
| `observations.include_root_target` | Add the reference root position and rotation (default `true`) |
| `observations.future_mode` | How upcoming reference frames are encoded; see below |
| `observations.future_frames` | How many upcoming reference frames the policy sees |
| `observations.future_frame_step` | Spacing between those frames, in motion frames |
| `observations.previous_action` | `action` adds the previous action to the observation; `null` leaves it out |

The default `future_mode`, `ee_future_root`, gives the position, rotation and
velocities of each end effector in the upcoming reference frames, relative to
the reference root of that frame. `phc_v6`, used by the humanoid tasks, follows
the observation of [PHC](https://arxiv.org/abs/2305.06456) (Luo et al. 2023,
*Perpetual Humanoid Control*): for each body it gives the difference from the
reference in position, rotation, linear and angular velocity, plus the
reference position and rotation, all in the character's heading frame. `none`
leaves the upcoming frames out.

## Resets

| Key | Meaning |
| --- | --- |
| `reset.random_start` | Start each episode from a random frame instead of the first |
| `reset.motion_noise_std` | Noise added to the reference pose at reset |
| `reset.deterministic` | Use the same start every time |
| `reset.motion_index`, `reset.start_frame` | Start from a specific clip and frame |

`adaptive_sampling` starts more episodes from the parts of the motion where the
policy does poorly. With `enabled: true` and the default
`method: reward_softmax`, the motion is split into chunks of
`chunk_size` frames, each chunk keeps a running average (`ema_alpha`) of the
episode `metric` (`reward` or `length`) started from it, and chunks are sampled
with a softmax at `temperature`. `random_sample_prob` keeps some uniform
sampling. Adaptive sampling only applies with `reset.random_start: true` and
`reset.deterministic: false`.

## Reward

The tracking reward compares the character with the reference at every step.

`reward.tracking.type` selects what is compared:

- `body_state`: the position, rotation, linear and angular velocity of each
  body, in the `frame` given (`world` for the included tasks). `weights` sets
  the weight of each of the four errors, and `body_weights` can weight bodies
  individually.
- `joint`: joint angles and velocities.

`reward.combination.type` turns the weighted errors into one reward:

| Type | Reward |
| --- | --- |
| `product` | Product of the per-term rewards `exp(-weight * error)` |
| `weighted_sum` | Sum of the per-term rewards, weighted by `combination.weights` |
| `raw_sum` | `reward.alive` minus the weighted errors |
| `power_sum` | One reward from the total weighted error $E$: $\exp(-\beta E)$ for `q: 1`, and $(1 + (q - 1)\beta E)^{-1/(q-1)}$ otherwise. `beta` is $\beta$; `epsilon` is the smallest reward |

In `product` and `weighted_sum`, `reward.alive` (default 0) sets a floor on
each per-term reward, which becomes `alive + (1 - alive) * exp(-weight * error)`.
`reward.combination.scale` multiplies the result. Logged per-term rewards are
unscaled.

`reward.action_smoothness` adds a term `exp(-weight * mean(delta^2))` that
penalizes changes in the command from one control step to the next. It works
with `torque` actions, where the command is the joint torque divided by
`torque_gain`, and with `mass` pose output, where it is the PD target. `source`
selects `torque`, `pd_target` or `auto` (the one that fits the action type).

## Termination

An episode ends when the time limit is reached, when the torso falls below
`termination.min_torso_height`, or, with `termination.nonfinite: true`, when the
simulation produces non-finite values. By default a clip loops when the
episode outlasts it; with `termination.motion_end: true`, reaching the clip's
last frame ends the episode instead.

`termination.early` optionally ends episodes where the character has lost
track of the reference:

| `type` | Ends the episode when |
| --- | --- |
| `reward` | The reward stays below `reward_threshold` for `consecutive_steps` steps |
| `phc_tracking` | The mean body position error exceeds `distance` (in metres) |
| `deviation_bounds` | The centre of mass (`com_distance`), chosen bodies (`body_distance`) or joints (`joint_angle`) move too far from the reference |

## Observation noise

`observation_noise` adds Gaussian noise to the policy's observations during
training. `std` sets its size and `units` its scale: `normalized` (relative to
the observation normalizer, the default) or `raw`. Benchmarks and playback see
clean observations.

## Debugging non-finite states

With `nan_debug.enabled: true`, the environment keeps the last `buffer_size`
steps of every environment and, when one produces a non-finite state, renders up
to `max_env_videos` videos of what led to it in `output_directory`. Training
then stops, unless `raise_on_nan` is set to `false`.
