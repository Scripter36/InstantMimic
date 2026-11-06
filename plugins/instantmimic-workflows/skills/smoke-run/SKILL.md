---
name: smoke-run
description: Use when validating wiring, Hydra composition, train/play startup, or a small sanity check before a larger experiment.
---

# Smoke Run

Smoke runs prove wiring only. Keep them short and local.

## Rules

- Read [Optional Local Workflows](../../references/optional-local-workflows.md) and follow the active repository or host GPU-coordination policy before launching GPU smoke commands.
- Use tensorboard, not W&B or MLflow: `experiment.logger.type=tensorboard`.
- Do not create a local MLflow store as a smoke-test convenience or as fallback for
  a configured remote tracking server.
- Keep CUDA graph and compile flags unchanged.
- Keep the config's authored `env.scene.num_envs`. A different environment count
  changes compiled shapes, can trigger separate Torch/Triton compilation, and can
  expose unsupported Triton shapes. Bound smoke duration with iterations instead.
- Inspect process exit, actual iteration completion and saved checkpoint when
  applicable. A launched process alone is not a completed smoke.
- Retain logs referenced by verification/review until that work is complete;
  delete disposable smoke output only when no longer needed.

## Train Smoke Shape

```bash
uv run train --config-name <config> experiment.logger.type=tensorboard trainer.max_iterations=1 seed=<seed>
```

Add minimal overrides for the specific wiring under test. Do not turn off:

- `env.use_cuda_graph`
- `runtime.act_cuda_graph`
- `runtime.update_cuda_graph`
- `runtime.compile`

## Play Smoke Shape

```bash
uv run play ++play.checkpoint=<ckpt_path> play.video.enabled=false play.steps=<small>
```

Use video only if the change touches rendering/video behavior.
