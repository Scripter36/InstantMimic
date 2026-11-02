from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from instantmimic.config.config import load_semantic_config
from instantmimic.rendering.mujoco_gl import apply_mujoco_gl
from instantmimic.utils.experiment_logging import load_experiment_dotenv
from instantmimic.utils.hydra_cli import (
    add_project_config_dir,
    run_hydra_without_outputs,
)
from instantmimic.utils.project_paths import framework_config_dir, project_root

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = str(_REPO_ROOT / "config")
# _CONFIG_DIR feeds @hydra.main, which is evaluated at import time and cannot
# raise usefully. main() calls framework_config_dir() to fail with an
# actionable message before Hydra ever looks at the path.
_HYDRA_CONTROL_ARGS = {
    "-h",
    "--help",
    "--hydra-help",
    "--cfg",
    "--info",
    "-m",
    "--multirun",
}


@hydra.main(config_path=_CONFIG_DIR, config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    semantic = load_semantic_config(cfg)
    from instantmimic.play.runtime import run_playback

    run_playback(semantic)


def cli() -> None:
    # Playback always renders, so it claims a backend here rather than at import
    # time. `force_headless=True` reproduces what the old
    # `rendering/mujoco_gl_bootstrap` module did as an import side effect of
    # `instantmimic.play`; an explicit MUJOCO_GL in the environment still wins.
    apply_mujoco_gl(force_headless=True)
    framework_config_dir()  # fail with an actionable message outside a checkout
    root = project_root()
    load_experiment_dotenv(root, _REPO_ROOT)
    sys.argv[1:] = add_project_config_dir(
        sys.argv[1:],
        project_root=root,
        framework_config_dir=_CONFIG_DIR,
    )
    argv = sys.argv[1:]
    if not any(arg in _HYDRA_CONTROL_ARGS for arg in argv):
        from instantmimic.play.runtime import maybe_run_checkpoint_first_cli

        if maybe_run_checkpoint_first_cli(argv):
            return
    run_hydra_without_outputs(main)


if __name__ == "__main__":
    cli()
