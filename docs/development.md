# Development

Install the development dependencies and run the checks that CI runs, plus the
CPU tests:

```bash
uv sync --locked --group dev
uv run ruff check src tests plugins
uv run ruff format --check src tests plugins
uv run pytest -m "not cuda and not integration"
uv run python -m instantmimic.tools.validate_configs --allow-unprepared-assets --json
```

Tests that need a GPU are marked `@pytest.mark.cuda`; run them with
`uv run pytest -m cuda` on a machine with an NVIDIA GPU. Tests build small
synthetic configs and models rather than loading the configs in `config/`,
which `validate_configs` checks separately.

## Source layout

| Directory | Contents |
| --- | --- |
| `src/instantmimic/config/` | Typed settings, composition and validation |
| `src/instantmimic/train/` | Training loop, checkpoints, metrics and benchmark evaluation |
| `src/instantmimic/play/` | Playback and video |
| `src/instantmimic/envs/` | Environments, including the MuJoCo Warp base environment |
| `src/instantmimic/rl/ppo/` | PPO rollouts, losses and updates |
| `src/instantmimic/rl/models/` | Actor and critic networks |
| `src/instantmimic/envs/controls/fast_muscle/` | Muscle control, including the MASS coordinator |
| `src/instantmimic/muscle/` | Muscle mechanics, geometry and model parsing |
| `src/instantmimic/hpo/` | Hyperparameter search controller |
| `src/instantmimic/rendering/` | Playback rendering and video encoding |

## Documentation

```bash
uv run --group docs mkdocs serve
uv run --group docs mkdocs build --strict
```

Pages are listed in `mkdocs.yml`; add new pages there and link between pages
with relative paths.
