# Actions and muscles

`env.action` defines what the policy outputs and how it moves the character.
`env.muscle` configures the muscle model used by the muscle-actuated tasks.

## Action types

| `env.action.type` | The policy outputs | Used by |
| --- | --- | --- |
| `ctrl` | Controls for the model's MuJoCo actuators, such as PD targets | Humanoid PD imitation |
| `musubi` | Not included in this release (see [MUSUBI](../../guides/muscle-control.md#musubi)) | Muscle-actuated tasks |
| `mujoco_activation` | Activations for the model's MuJoCo muscle actuators | No included config |
| `activation` | One excitation per muscle, clamped to $[0, 1]$ | Activation-output baselines |
| `torque` | Joint torques | Direct-torque baselines |
| `mass` | A pose or torque target that MuscleNN turns into muscle activations | [MASS](../../guides/muscle-control.md#mass-baseline) |

`env.action.space` is the action size, which depends on the model. The
MUSUBI configs can be inspected but not trained in this release.

An action type comes with matching observation, model and agent settings, so
switch between them by choosing another task config or by adding a preset to a
config's `defaults` rather than overriding `env.action.type` alone:

```yaml
defaults:
  - base
  - presets/activation_mode@_global_
  - _self_
```

The SNU presets are in `config/snu_humanoid/imitation/presets/`.

### `ctrl`

`scale` multiplies the action. With `use_reference: true`, the action is added
to the reference motion's actuator targets, so the policy learns a correction
to the reference instead of the full command.

### `torque`

`scale` maps the policy output to torques. A number scales every joint
equally, a list gives one scale per joint, and `auto` uses the torque range the muscles can produce in the rest pose.
`bounds.env_clip: true` clips the torque to what the muscles can produce in the
current pose (the muscle-bounded baseline), and `false` leaves it unconstrained.

### `mass`

`policy_output` sets what the policy controls:

| Key | Meaning |
| --- | --- |
| `policy_output.type` | `pose` or `torque` |
| `policy_output.scale` | For `pose`, the residual scale in radians. For `torque`, as for `torque` actions above |
| `policy_output.reference` | For `pose`, the pose the residual is added to: `motion` or `rest` |
| `policy_output.stable_pd` | For `pose`, the StablePD controller: `mjcf_path` (the model whose actuator gains it uses), `kp_scale`, `kv_scale` |

`muscle_nn` configures the MuscleNN coordinator:

| Key | Meaning |
| --- | --- |
| `hidden_dims` | Hidden layer sizes |
| `output_activation` | Output function: `tanh_relu` for pose and `sine` for torque by default. A checkpoint only loads with the setting it was trained with |
| `input_scale`, `target_scale` | Scales applied to the inputs and the torque target |
| `optimizer`, `learning_rate`, `batch_size`, `epochs`, `grad_clip` | How MuscleNN is trained |
| `target_loss_weight`, `regularization_weight` | Weights of the torque-matching loss and the activation regularization |
| `buffer_capacity` | Number of recent samples MuscleNN trains on |
| `substep_sampling_mode` | Train on `all` physics substeps or a `random` one per step |
| `frozen` | Keep MuscleNN fixed, for example when starting from a trained checkpoint |

## Muscle model

| Key | Meaning |
| --- | --- |
| `force_scale` | Multiplier on all muscle forces |
| `length_range` | Range of normalized muscle length used for the force-length curves and observations |
| `active_force` | Active force-length curve: `type: gaussian` (with `gamma`, `lmin`, `lmax`) or `type: mujoco` |
| `passive_force` | Passive force-length curve: `type: exponential` (with `k_pe`, `e_mo`, `lmax`, `fpmax`) or `type: mujoco`. `env: true` applies it in the simulation |
| `force_velocity` | Force-velocity relation: `enabled`, `vmax`, `fvmax`, and `implicit` to integrate its damping implicitly |
| `metadata_path` | Optional file with extra muscle metadata |

Muscles act instantly, without activation dynamics, and follow paths of
waypoints attached to bodies, optionally wrapping around spheres and cylinders.
See [Muscle simulation and control](../../guides/muscle-control.md#simulation-details).
To print a task's full muscle settings:

```bash
uv run train --config-name snu_humanoid/imitation/walk_musubi --cfg job --resolve
```
