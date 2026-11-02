# Contributing

Thanks for your interest in InstantMimic. Set up the development environment
from a clone of the repository (Python 3.12):

```bash
uv sync --locked --group dev
```

The lock file pins MuJoCo, MuJoCo Warp and Warp together; please keep them
pinned when changing simulation or muscle code.

Before opening a pull request, run:

```bash
uv run ruff check src tests plugins
uv run ruff format --check src tests plugins
uv run pytest -m "not cuda and not integration"
uv run python -m instantmimic.tools.validate_configs --allow-unprepared-assets --json
```

CI runs the Ruff checks, the config check, the package build and the
documentation build. If your change touches GPU code, please also run the
relevant tests marked `@pytest.mark.cuda` on a GPU, and mention in the pull
request what you ran.

A few conventions:

- Add focused CPU tests for logic or numerical changes. Tests build small
  synthetic configs instead of loading the configs in `config/`.
- For documentation changes, run `uv run --group docs mkdocs build --strict`.
- When you modify third-party code or add data, keep the upstream copyright and
  license notices, and record the source and license of new data next to it.

See [Development](docs/development.md) for the source layout and
[Assistant workflows](docs/guides/assistant-workflows.md) for the optional
agent plugin.
