from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from instantmimic.envs.snu.tasks.snu_command.checkpoint_runtime import (
    apply_low_level_checkpoint_runtime_config,
)
from instantmimic.envs.snu.tasks.snu_command.low_level_runtime import (
    initialize_low_level_policy,
)


def test_reference_relative_actions_are_rejected_when_autoload_is_off():
    env = SimpleNamespace(
        cfg=OmegaConf.create({"action": {"type": "ctrl", "use_reference": True}})
    )
    with pytest.raises(ValueError, match="use_reference=false"):
        initialize_low_level_policy(env)


def test_checkpoint_restores_control_type_and_rate_without_simulator_flags(tmp_path):
    run = tmp_path / "run"
    config = run / "run_bundle/config/train_config_resolved.yaml"
    config.parent.mkdir(parents=True)
    model = run / "run_bundle/mjcf_tree/data/model.xml"
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco/>")
    checkpoint = run / "checkpoints/nested/agent.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    OmegaConf.save(
        OmegaConf.create(
            {
                "env": {
                    "sim": {
                        "mjcf_path": "run_bundle/mjcf_tree/data/model.xml",
                        "hz": 240,
                    },
                    "decimation": 8,
                    "action": {"type": "ctrl", "scale": 0.25, "use_reference": False},
                    "motion": {
                        "source": ["data/humanoid/motions/amass/phc_train/**/*.npz"]
                    },
                    "observations": {"future_frames": 2},
                }
            }
        ),
        config,
    )
    cfg = OmegaConf.create(
        {
            "low_level_checkpoint": str(checkpoint),
            "low_level_config_autoload": True,
            "sim": {"mjcf_path": "unused.xml", "hz": 120},
            "decimation": 4,
            "action": {"type": "other", "scale": 1},
            "reset": {"motion": {"source": []}},
        }
    )
    apply_low_level_checkpoint_runtime_config(cfg)
    assert cfg.action.type == "ctrl"
    assert cfg.action.scale == 0.25
    assert cfg.sim.hz == 240
    assert cfg.decimation == 8
    assert cfg.sim.mjcf_path == str(model)
    assert sorted(cfg.sim) == ["hz", "mjcf_path"]


def test_command_environment_registration_imports_runtime_dependencies():
    from instantmimic.envs.snu.tasks.snu_command.snu_command_env import SNUCommandEnv

    assert SNUCommandEnv.__name__ == "SNUCommandEnv"
