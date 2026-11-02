# PPO

`agent` configures PPO. The humanoid imitation tasks use:

```yaml
agent:
  class: PPO
  rollouts: 6
  learning_epochs: 5
  mini_batches: 4
  shuffle_mode: per_update
  discount_factor: 0.95
  gae_lambda: 0.95
  time_limit_bootstrap: true
  optimizer:
    type: muon
    momentum: 0.95
    nesterov: true
    adjust_lr_fn: match_rms_adamw
    ns_steps: 5
    eps: 2.0e-5
    weight_decay: 3.0e-5
    adamw:
      betas: [0.9, 0.999]
      eps: 2.0e-5
      weight_decay: 3.0e-5
  learning_rate:
    initial: 1.0e-3
    bounds: [1.0e-6, 1.0e-2]
    scheduler:
      type: kl_adaptive
      threshold: 0.1
      factor: 1.5
  state_preprocessor: RunningStandardScaler
  value_preprocessor: RunningStandardScaler
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2
  clip_predicted_values: true
  value_loss_clip_mode: clamp
  entropy_loss_scale: 1.0e-3
  value_loss_scale: 4.5
```

## Collection and updates

| Key | Meaning |
| --- | --- |
| `rollouts` | Control steps collected from every environment per iteration |
| `learning_epochs` | Passes over the collected data per iteration |
| `mini_batches` | Minibatches per pass; the number of transitions must divide evenly |
| `shuffle_mode` | Shuffle the data once per iteration (`per_update`), every pass (`per_epoch`), or not at all (`off`) |
| `discount_factor`, `gae_lambda` | Discount and GAE λ, per control step |
| `time_limit_bootstrap` | When an episode ends only because of the time limit, bootstrap its return from the value estimate |

## Optimizer and learning rate

`optimizer.type` is `adam`, `adamw`, `sgd` or `muon`, each with the usual
settings (`betas`, `eps`, `weight_decay`, `momentum`, `nesterov`, …).

Muon updates the hidden weight matrices; the remaining parameters (input and
output layers, normalization layers and biases) use AdamW, configured under
`optimizer.adamw`. Muon's own `eps` and `weight_decay` apply to the matrices
only. `sgd` cannot be captured in a CUDA graph, so it needs
`runtime.update_cuda_graph: false`.

`learning_rate.initial` sets the starting rate. Without a `scheduler`, the rate
stays constant. With `scheduler.type: kl_adaptive`, the rate is divided by
`factor` when the KL divergence between the old and new policy exceeds twice
`threshold`, and multiplied by `factor` when it falls below half of it, within
`learning_rate.bounds`. `scheduler.update_mode` sets how often the rate is
adjusted: `per_epoch` (the default), `per_update` or `per_minibatch`.

## Losses

| Key | Meaning |
| --- | --- |
| `ratio_clip` | PPO clipping range of the probability ratio |
| `value_clip`, `clip_predicted_values` | Clip the new value prediction to within `value_clip` of the old one |
| `value_loss_clip_mode` | How the clipped and unclipped value losses are combined: `clamp` or `max` |
| `value_loss_scale` | Weight of the value loss |
| `entropy_loss_scale` | Weight of the entropy bonus |
| `grad_norm_clip` | Gradient norm limit; `separate_grad_norm_clip: true` clips actor and critic separately when they do not share a network |
| `kl_estimator` | How the KL for the scheduler is computed: `sampled` or `analytic` |

## Normalization

`state_preprocessor` and `value_preprocessor` normalize observations and value
targets with running statistics: `RunningStandardScaler`,
`RslRlEmpiricalNormalization` or `none`. `reward_preprocessor` can scale
rewards with `RunningReturnScaler`.

## Optional blocks

- `activation_regularization` adds a penalty on the policy mean for the muscle
  activation baselines; see
  [Muscle simulation and control](../../guides/muscle-control.md#direct-control-baselines).
- `latent_prior` turns the policy into the latent (VAE) policy used by the
  [AMASS tasks](../../data/amass-retargeting.md).
