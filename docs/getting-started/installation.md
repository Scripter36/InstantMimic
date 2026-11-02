# Installation

InstantMimic needs Python 3.12 and an NVIDIA GPU whose driver supports CUDA 12.8.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone
the repository and install the locked dependencies:

```bash
git clone https://github.com/Scripter36/InstantMimic.git
cd InstantMimic
uv sync --locked
```

Run commands from the repository root. Configs and models are loaded from
`config/` and `data/` by relative path.

## Check the setup

These commands compose a config and validate every config without using the GPU:

```bash
uv run train --config-name humanoid/imitation/walk_pd --cfg job --resolve
uv run python -m instantmimic.tools.validate_configs --allow-unprepared-assets --json
```

With `--allow-unprepared-assets`, inputs that are prepared separately, such as the
[BeyondMimic](../guides/beyondmimic.md) assets, are listed as requiring
preparation instead of failing the check.

Then run the [first training example](first-run.md). The first run compiles
kernels before its first iteration.

## Experiment logging

The included configs log to TensorBoard in each run's directory under `logs/`:

```bash
uv run tensorboard --logdir logs
```

To log to Weights & Biases or MLflow instead, set `INSTANTMIMIC_LOGGER` to
`wandb` or `mlflow`, or override `experiment.logger.type` for a single run.
`train` and `play` read `.env` and `.env.local` in the repository root, so the
variables can live there:

```bash
# .env
INSTANTMIMIC_LOGGER=wandb
WANDB_API_KEY=your-api-key
```

For W&B, sign in with `uv run wandb login` or set `WANDB_API_KEY`. Runs go to
the `InstantMimic` project unless you set another project or entity:

```bash
uv run train --config-name humanoid/imitation/walk_pd \
  +experiment.logger.project=my-project +experiment.logger.entity=my-team
```

For MLflow, set `MLFLOW_TRACKING_URI` to your tracking server or pass
`+experiment.logger.tracking_uri=...`; otherwise MLflow uses its default local
store. Runs go to the `InstantMimic` experiment unless
`+experiment.logger.experiment_name` is set.

## Headless video

On Linux, install the NVIDIA EGL libraries that match the driver and set
`MUJOCO_GL=egl` for processes that render. Training without video needs no
graphics context. See [video rendering](../guides/rendering.md).

## Development and documentation

```bash
uv sync --locked --group dev
uv sync --locked --group docs
uv run --group docs mkdocs serve
```

`mkdocs serve` previews the documentation locally; `mkdocs build --strict`
builds the site.
