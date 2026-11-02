---
name: train
description: Use when training a policy in this repository, constructing train commands, or choosing train overrides.
---

# Train

Use the package entrypoint from the project root:

```bash
uv run train --config-name <config> <hydra overrides...>
```

Current PD and MASS examples:

```bash
uv run train --config-name humanoid/imitation/walk_pd
uv run train --config-name snu_humanoid/imitation/mass/walk_mass
```

Before launch, follow the config's complete defaults chain and runtime consumer.
Use `--cfg job --resolve` to inspect effective settings without starting training;
keep private logger destinations out of shared output.

- Preserve the logger selected by normal config, dotenv and explicit overrides.
  TensorBoard is a valid full-run logger; do not force W&B or MLflow.
- For a wiring smoke, use `$smoke-run`: TensorBoard and a short iteration bound,
  preserving authored environment count, graph, compile and precision settings.
- Read [Experiment Protocol](../../references/experiment-protocol.md) for full
  experiments and [Optional Local Workflows](../../references/optional-local-workflows.md)
  before GPU work. Reuse existing authority; request only missing authority.
- Record seed, resolved task, runtime modes, logger, run directory and budget.
  Keep training success separate from checkpoint evaluation and video validity.
- Use `$config-variant-check` for config edits. For a reportable seeded
  comparison, follow the shared experiment protocol and the user's seed count;
  a speed probe or smoke does not become a three-seed training study.
