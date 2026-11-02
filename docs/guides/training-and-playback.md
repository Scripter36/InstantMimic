# Training and playback

`train` builds the environment, policy and trainer from a Hydra config and runs
PPO. `play` loads a checkpoint and runs its policy. Run both from the repository
root, because configs refer to models and motions by relative path.

## Configure a run

Print the resolved config, then override any value on the command line:

```bash
uv run train --config-name humanoid/imitation/walk_pd --cfg job --resolve
uv run train --config-name humanoid/imitation/walk_pd trainer.max_iterations=300
```

Overrides use full dotted paths such as `trainer.max_iterations`. Prefix a key
with `+` to add one the config does not define. The
[configuration reference](../reference/configuration.md) describes the main
settings.

Each run writes to `logs/ppo/<config directory>/<start time>_ppo_custom_<config>/`:

| Path | Contents |
| --- | --- |
| `run_bundle/` | The resolved config and the files needed to rebuild the policy |
| `checkpoints/agent_iter_N.pt` | Periodic full training checkpoints |
| `checkpoints/best_agent.pt` | The checkpoint with the best benchmark result so far |
| `train.log`, TensorBoard events | Training logs and metrics |

To log to Weights & Biases or MLflow instead of TensorBoard, see
[experiment logging](../getting-started/installation.md#experiment-logging).

## Resume or start from a policy

Resuming restores the policy, optimizer, observation normalization, random
state and the environment's own learning state, and continues the run:

```bash
uv run train --config-name humanoid/imitation/walk_pd \
  ++trainer.initialization.type=resume \
  ++trainer.initialization.checkpoint='logs/ppo/humanoid/imitation/walk_pd/YOUR_RUN/checkpoints/agent_iter_N.pt'
```

To start a new run from a trained policy's weights instead, use
`trainer.initialization.type=policy_init`; the optimizer and counters start
fresh. For MASS, resuming also restores the MuscleNN coordinator (see
[muscle control](muscle-control.md#mass-baseline)).

## Evaluation

```bash
uv run play --config-name humanoid/imitation/walk_pd \
  +play.checkpoint='logs/ppo/humanoid/imitation/walk_pd/YOUR_RUN/checkpoints/best_agent.pt' \
  play.video.enabled=false play.steps=300
```

Playback reports the episode length and tracking metrics as well as the reward.
To compare policies, evaluate them with the same model, motion, episode length
and action settings.
