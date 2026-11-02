from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from instantmimic.config.config import load_semantic_config
from instantmimic.rendering.mujoco_gl import apply_mujoco_gl
from instantmimic.utils.experiment_logging import (
    load_experiment_dotenv,
)
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


@hydra.main(config_path=_CONFIG_DIR, config_name="config.yaml", version_base="1.3")
def _hydra_main(cfg: DictConfig) -> None:
    semantic = load_semantic_config(cfg)
    from instantmimic.train.runtime import run_training

    run_training(semantic)


def main() -> None:
    # Before anything imports mujoco. MuJoCo reads MUJOCO_GL when it is
    # imported, not when a renderer is built, and `instantmimic.train.runtime`
    # -- imported further down this function -- pulls it in. Setting the
    # variable after that import has no effect, which is why the previous
    # module-scope call sat immediately above `import mujoco` in
    # `envs/direct_rl_env.py`.
    apply_mujoco_gl(force_headless=False)
    framework_config_dir()  # fail with an actionable message outside a checkout
    root = project_root()
    load_experiment_dotenv(root, _REPO_ROOT)
    sys.argv[1:] = add_project_config_dir(
        sys.argv[1:],
        project_root=root,
        framework_config_dir=_CONFIG_DIR,
    )
    run_hydra_without_outputs(_hydra_main)


if __name__ == "__main__":
    main()
