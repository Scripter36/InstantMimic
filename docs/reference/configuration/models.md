# Models

`models` describes the policy (actor) and value (critic) networks. The humanoid
imitation tasks use:

```yaml
models:
  shared: true
  actor:
    architecture: mlp
    layers: [384, 384, 192]
    activation: silu
    normalization: none
    distribution:
      std: parameter
      initial_log_std: -2.0
      log_std_bounds: [-20.0, 2.0]
      bound_mode: clamp
      action_transform: none
```

## Networks

| Key | Meaning |
| --- | --- |
| `architecture` | `mlp`, or `residual_mlp` with skip connections scaled by `residual_scale` |
| `layers` | Hidden layer sizes |
| `activation` | `elu`, `relu`, `tanh`, `silu`, `mish` or `leaky_relu` |
| `normalization` | `none` or `layer_norm` after each hidden layer |
| `output_initialization_std` | Optional standard deviation for initializing the actor's output layer |

With `shared: true`, the value function is a separate output head on the
actor's network. With `shared: false`, define `models.critic` with its own
`architecture` and `layers`. Its `activation` and `normalization` must match the
actor's.

## Action distribution

The actor outputs the mean of a diagonal Gaussian. `distribution` sets how its
standard deviation is produced:

| Key | Meaning |
| --- | --- |
| `std` | `parameter`: one learned value per action dimension. `fixed`: a constant. `network`: predicted by the network from the observation |
| `parameterization` | Whether a learned `parameter` std is stored as its logarithm (`log`, the default) or directly (`scalar`) |
| `initial_log_std` | Starting log standard deviation |
| `log_std_bounds` | Allowed range of the log standard deviation |
| `bound_mode` | How the range is enforced: `clamp` (hard limit) or `tanh` (smooth) |
| `action_transform` | `none`, or `tanh` to squash sampled actions into $[-1, 1]$ |
| `bound_loss` | For `std: network` with `bound_mode: clamp`: a penalty of `weight` on predicted log standard deviations that come within `margin` of `log_std_bounds` |

The environment scales actions to physical units (see `env.action.scale`), so
`initial_log_std` is in the policy's own units. If you change the action scale,
adjust `initial_log_std` to keep the same exploration.
