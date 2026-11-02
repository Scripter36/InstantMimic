"""Evaluator launch, planner commands, and search run-directory behavior.

Cycle execution is covered by test_hpo_cycle.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from instantmimic.hpo import (
    agent_runtime,
    planner,
)
from instantmimic.hpo.agent_runtime import (
    _build_codex_command,
    _build_opencode_command,
    resolve_agent_spec,
)
from instantmimic.hpo.candidate_executor import (
    build_candidate_runtime_env,
)
from instantmimic.hpo.evaluator_launch import (
    build_hq_command,
    build_slurm_command,
    resolve_evaluator_launch_config,
    write_slurm_script,
)
from instantmimic.hpo.run_context import (
    experiment_directory_for_run,
    prepare_hpo_runtime_task,
    split_hpo_runtime_cfg,
)


def _write_test_train_config(config_file: Path) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "seed": 0,
                "env": {
                    "scene": {"num_envs": 1},
                    "observations": {"previous_action": "action"},
                },
                "agent": {"rollouts": 1},
                "trainer": {"max_iterations": 1},
            }
        ),
        config_file,
        resolve=False,
    )
    config_file.write_text(
        "# @package _global_\n" + config_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _write_test_semantic_config(config_file: Path) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "schema_version": 1,
                "seed": 7,
                "task": "synthetic-hpo",
                "env": {},
                "models": {
                    "shared": True,
                    "actor": {
                        "architecture": "mlp",
                        "layers": [8],
                        "activation": "silu",
                        "normalization": "none",
                        "distribution": {
                            "std": "parameter",
                            "initial_log_std": -2.0,
                            "log_std_bounds": [-5.0, 2.0],
                            "bound_mode": "clamp",
                            "action_transform": "none",
                        },
                    },
                },
                "agent": {
                    "class": "PPO",
                    "rollouts": 1,
                    "learning_epochs": 1,
                    "mini_batches": 1,
                    "discount_factor": 0.99,
                    "gae_lambda": 0.95,
                    "optimizer": {"type": "adam"},
                    "learning_rate": {"initial": 0.001},
                },
                "trainer": {"max_iterations": 1},
                "experiment": {
                    "directory": "synthetic/hpo",
                    "logger": {"type": "tensorboard"},
                },
                "runtime": {},
                "play": {
                    "num_envs": 1,
                    "steps": 1,
                    "action": {"mode": "mean"},
                },
            }
        ),
        config_file,
        resolve=False,
    )


def test_the_planner_uses_the_canonical_agent_runtime() -> None:
    """There is one agent runtime, and the planner is its only caller."""

    assert planner.AgentSpec is agent_runtime.AgentSpec
    assert planner.resolve_agent_spec is agent_runtime.resolve_agent_spec
    assert planner.run_agent is agent_runtime.run_agent
    assert planner.agent_label is agent_runtime.agent_label


STRATEGY_RESPONSE = "\n".join(
    [
        "# Current Strategy",
        "## Best Baseline",
        "none",
        "## What Seems Promising",
        "none",
        "## What To Avoid",
        "none",
        "## Next Experiment",
        "none",
        "## Wildcards",
        "none",
        "",
    ]
)


def test_search_runtime_env_overrides_inherited_compiler_caches_when_keyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited = {
        "WARP_CACHE_DIR": "/shared/warp",
        "WP_CACHE_DIR": "/shared/wp",
        "WARP_CACHE_PATH": "/shared/warp-path",
        "TORCHINDUCTOR_CACHE_DIR": "/shared/torchinductor",
        "TRITON_CACHE_DIR": "/shared/triton",
        "CUDA_CACHE_PATH": "/shared/cuda",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("INSTANTMIMIC_LOGGER", "wandb")

    env = build_candidate_runtime_env(cache_key="line search")

    assert env["WARP_CACHE_DIR"] == env["WP_CACHE_DIR"]
    assert env["WARP_CACHE_DIR"] == env["WARP_CACHE_PATH"]
    for name, inherited_value in inherited.items():
        assert env[name] != inherited_value
        assert "instantmimic-hpo-cache" in env[name]
    assert env["INSTANTMIMIC_LOGGER"] == "wandb"


def test_hpo_runtime_uses_unique_timestamped_state_dirs(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "config" / "hpo" / "demo.yaml"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(
        "\n".join(
            [
                "name: demo-task",
                "batch:",
                "  motions:",
                "    backflip_pd:",
                "      config_path: config/humanoid/imitation",
                "      config_name: backflip_pd",
                "  seeds: [1]",
                "  fixed_train_overrides: [trainer.max_iterations=12000]",
                "objective:",
                "  mode: scalar",
                "  strategy: wall_clock",
                "",
            ]
        ),
        encoding="utf-8",
    )
    task_cfg = OmegaConf.load(task_path)
    run_root = tmp_path / "logs/hpo/demo_task"
    existing = run_root / "2026-07-04_12-00-00"
    existing.mkdir(parents=True)

    first = prepare_hpo_runtime_task(
        root=tmp_path,
        task_path=task_path,
        task_cfg=task_cfg,
        timestamp_name="2026-07-04_12-00-00",
    )
    second = prepare_hpo_runtime_task(
        root=tmp_path,
        task_path=task_path,
        task_cfg=task_cfg,
        timestamp_name="2026-07-04_12-00-00",
    )

    assert first.run_dir == existing.with_name("2026-07-04_12-00-00_0001")
    assert second.run_dir == existing.with_name("2026-07-04_12-00-00_0002")
    assert first.state_dir == first.run_dir / "state"
    assert first.task_cfg.output_root == str(
        first.run_dir.relative_to(tmp_path) / "evaluator"
    )
    assert "output_dir" not in first.task_cfg
    assert not (tmp_path / "workspaces").exists()
    assert (
        experiment_directory_for_run(first.task_cfg, task_path, first.run_dir)
        == "hpo/demo_task/2026-07-04_12-00-00_0001"
    )


def test_hpo_runtime_resume_reuses_legacy_workspace_run_dir(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "config" / "hpo" / "demo.yaml"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(
        "\n".join(
            [
                "name: demo-task",
                "config_name: backflip_pd",
                "max_iterations: 12000",
                "seed: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    task_cfg = OmegaConf.load(task_path)
    run_dir = tmp_path / "workspaces" / "hpo-demo_task" / "runs" / "2026-07-04_12-00-00"
    run_dir.mkdir(parents=True)

    context = prepare_hpo_runtime_task(
        root=tmp_path,
        task_path=task_path,
        task_cfg=task_cfg,
        resume=run_dir,
        timestamp_name="2026-07-04_12-00-00",
    )

    assert context.resumed is True
    assert context.run_dir == run_dir
    assert context.state_dir == run_dir / "state"
    assert not run_dir.with_name("2026-07-04_12-00-00_0001").exists()


def test_hydra_hpo_runtime_overrides_compose_without_append(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "config"
    runtime_config = config_root / "hpo" / "runtime_test.yaml"
    runtime_config.parent.mkdir(parents=True)
    runtime_config.write_text(
        "\n".join(
            [
                "# @package _global_",
                "",
                "runtime:",
                "  cycles: 3",
                "  dry_run: false",
                "  evaluator:",
                "    launcher: slurm",
                "planning:",
                "  platform: anthropic",
                "  model: test-model",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with initialize_config_dir(config_dir=str(config_root), version_base="1.3"):
        cfg = compose(
            config_name="hpo/runtime_test",
            overrides=[
                "runtime.cycles=1",
                "runtime.dry_run=true",
                "runtime.evaluator.launcher=local",
                "planning.platform=openai",
                "planning.model=null",
            ],
        )

    assert cfg.runtime.cycles == 1
    assert cfg.runtime.dry_run is True
    assert cfg.runtime.evaluator.launcher == "local"
    assert cfg.planning.platform == "openai"
    assert cfg.planning.model is None


def test_split_hpo_runtime_cfg_removes_runner_fields_from_task() -> None:
    cfg = OmegaConf.create(
        {
            "name": "demo-task",
            "planning": {"platform": "openai", "model": None},
            "runtime": {
                "cycles": 1,
                "dry_run": True,
                "evaluator": {"launcher": "local"},
            },
        }
    )

    task_cfg, runtime_cfg = split_hpo_runtime_cfg(cfg)

    assert "runtime" not in task_cfg
    assert task_cfg.planning.platform == "openai"
    assert runtime_cfg.cycles == 1
    assert runtime_cfg.dry_run is True
    assert runtime_cfg.evaluator.launcher == "local"


def test_resolve_agent_spec_rejects_legacy_provider_field() -> None:
    task_cfg = OmegaConf.create(
        {
            "planner": {
                "platform": "openai",
                "provider": "claude",
                "model": "claude-sonnet-4-5",
            },
        }
    )

    with pytest.raises(ValueError, match="provider.*platform"):
        resolve_agent_spec(
            task_cfg,
            "planner",
            default_model=None,
            default_reasoning_effort="high",
        )


def test_seed_train_worker_runs_request_with_redirected_log(
    tmp_path: Path,
) -> None:
    import logging

    from omegaconf import OmegaConf

    from instantmimic.hpo import seed_train_worker

    leaked_log = tmp_path / "leaked_train.log"
    trial_log = tmp_path / "trial.log"
    root_logger = logging.getLogger()
    original_handlers = tuple(root_logger.handlers)
    original_level = root_logger.level
    for handler in original_handlers:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)

    def fake_compose_train_cfg(
        *,
        config_path: str | None,
        config_name: str,
        overrides: list[str],
    ):
        assert config_path == "config/humanoid/imitation"
        assert config_name == "backflip_pd"
        assert "seed=11" in overrides
        return OmegaConf.create({"seed": 11, "env": {"seed": 11}})

    def fake_run_training(cfg) -> None:
        assert int(cfg.seed) == 11
        logging.basicConfig(level=logging.INFO)
        print("training stdout")
        os.write(1, b"native stdout\n")
        os.write(2, b"native stderr\n")
        logging.info("TIME_TO_REWARD_THRESHOLD_SECONDS=1.250000")
        logging.warning("training warning")
        logging.getLogger().addHandler(logging.FileHandler(leaked_log))

    request = seed_train_worker.SeedTrainRequest(
        config_path="config/humanoid/imitation",
        config_name="backflip_pd",
        overrides=["seed=11"],
        log_file=str(trial_log),
        frames_per_run=100,
    )

    try:
        response = seed_train_worker._run_seed_request(
            request,
            compose_train_cfg_fn=fake_compose_train_cfg,
            run_training_fn=fake_run_training,
        )

        assert response.returncode == 0
        assert response.wall_seconds is not None
        assert response.throughput_fps is not None
        log_text = trial_log.read_text(encoding="utf-8")
        assert "training stdout" in log_text
        assert "native stdout" in log_text
        assert "native stderr" in log_text
        assert "TIME_TO_REWARD_THRESHOLD_SECONDS=1.250000" in log_text
        assert "training warning" in log_text
        root_handlers = logging.getLogger().handlers
        assert not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == leaked_log
            for handler in root_handlers
        )
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)


def test_seed_train_worker_default_runner_loads_semantic_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omegaconf import OmegaConf

    from instantmimic.config.schema import SemanticConfig
    from instantmimic.hpo import seed_train_worker

    cfg = OmegaConf.create({"schema_version": 1, "sentinel": "composed"})
    semantic = SemanticConfig.from_mapping(
        {
            "schema_version": 1,
            "seed": 7,
            "task": "Synthetic-v0",
            "env": {},
            "models": {},
            "agent": {},
            "trainer": {},
            "experiment": {},
            "runtime": {},
            "play": {},
        }
    )
    seen: dict[str, object] = {}

    def fake_load_semantic_config(value):
        seen["source"] = value
        return semantic

    def fake_run_training(value) -> None:
        seen["runtime"] = value

    monkeypatch.setattr(
        seed_train_worker,
        "load_semantic_config",
        fake_load_semantic_config,
        raising=False,
    )
    monkeypatch.setattr("instantmimic.train.runtime.run_training", fake_run_training)

    seed_train_worker._default_run_training(cfg)

    assert seen["source"] is cfg
    assert seen["runtime"] is semantic


def test_codex_command_uses_configured_sandbox() -> None:
    command = _build_codex_command(
        root=Path.cwd(),
        message_path=Path("/tmp/codex-message.md"),
        platform="openai",
        model=None,
        reasoning_effort="high",
        sandbox="read-only",
        command_prefix=("codex",),
        codex_config_overrides=(),
    )

    assert command[command.index("-s") + 1] == "read-only"


def test_codex_command_reads_prompt_from_stdin() -> None:
    """The prompt is not an argument at all -- a long one would not fit in argv."""

    command = _build_codex_command(
        root=Path.cwd(),
        message_path=Path("/tmp/codex-message.md"),
        platform="openai",
        model=None,
        reasoning_effort="high",
        sandbox="read-only",
        command_prefix=("codex",),
        codex_config_overrides=(),
    )

    assert command[-1] == "-"


def test_run_agent_passes_codex_prompt_on_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = "Return OK." * 100_000
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(agent_runtime.subprocess, "run", fake_run)

    returncode = agent_runtime.run_agent(
        root=tmp_path,
        prompt=prompt,
        message_path=tmp_path / "codex-message.md",
        agent=agent_runtime.AgentSpec(
            backend="codex",
            platform="openai",
            model=None,
            reasoning_effort="high",
            sandbox="read-only",
            command=("codex",),
        ),
        timeout_seconds=30.0,
        dry_run=False,
    )

    assert returncode == 0
    assert captured["command"][-1] == "-"
    assert prompt not in captured["command"]
    assert captured["kwargs"]["input"] == prompt
    assert captured["kwargs"]["text"] is True


def test_resolve_agent_spec_supports_opencode_planner() -> None:
    task_cfg = OmegaConf.create(
        {
            "planner": {
                "platform": "opencode",
                "model": "zai-coding-plan/glm-5.2",
                "agent": "hpo",
                "variant": "high",
                "pure": True,
            }
        }
    )

    agent = resolve_agent_spec(
        task_cfg,
        "planner",
        default_model=None,
        default_reasoning_effort="high",
    )

    assert agent.backend == "opencode"
    assert agent.platform == "opencode"
    assert agent.command == ("opencode", "run")
    assert agent.model == "zai-coding-plan/glm-5.2"
    assert agent.opencode_agent == "hpo"
    assert agent.opencode_variant == "high"
    assert agent.opencode_pure is True


def test_resolve_agent_spec_skips_unset_codex_config_defaults() -> None:
    task_cfg = OmegaConf.create(
        {
            "planner": {
                "platform": "codex",
                "codex_config": {
                    "model_context_window": 200000,
                    "model_auto_compact_token_limit": None,
                },
            },
        }
    )

    agent = resolve_agent_spec(
        task_cfg,
        "planner",
        default_model=None,
        default_reasoning_effort="high",
    )

    assert agent.codex_config_overrides == (("model_context_window", "200000"),)


def test_opencode_command_uses_headless_hpo_options() -> None:
    prompt_file = Path("/tmp/opencode-prompt.md")
    command = _build_opencode_command(
        root=Path("/repo"),
        prompt_file=prompt_file,
        model="zai-coding-plan/glm-5.2",
        command_prefix=("opencode", "run"),
        agent_name="hpo",
        variant="high",
        output_format="default",
        use_pure=True,
        skip_permissions=True,
    )

    assert command == [
        "opencode",
        "run",
        "--dir",
        "/repo",
        "--format",
        "default",
        "--model",
        "zai-coding-plan/glm-5.2",
        "--agent",
        "hpo",
        "--variant",
        "high",
        "--pure",
        "--dangerously-skip-permissions",
        "Read the attached prompt file and follow its instructions exactly.",
        "--file",
        str(prompt_file),
    ]


def test_run_agent_passes_opencode_prompt_as_attached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = "Return OK." * 100_000
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(agent_runtime.subprocess, "run", fake_run)

    returncode = agent_runtime.run_agent(
        root=tmp_path,
        prompt=prompt,
        message_path=tmp_path / "opencode-message.md",
        agent=agent_runtime.AgentSpec(
            backend="opencode",
            platform="opencode",
            model="zai-coding-plan/glm-5.2",
            reasoning_effort="high",
            sandbox="read-only",
            command=("opencode", "run"),
        ),
        timeout_seconds=30.0,
        dry_run=False,
    )

    command = captured["command"]
    prompt_file = tmp_path / "opencode-message.prompt.md"
    assert returncode == 0
    assert prompt not in command
    assert command[command.index("--file") + 1] == str(prompt_file)
    assert command.index(
        "Read the attached prompt file and follow its instructions exactly."
    ) < command.index("--file")
    assert prompt_file.read_text(encoding="utf-8") == prompt
    assert captured["kwargs"].get("input") is None
    assert (tmp_path / "opencode-message.md").read_text(encoding="utf-8") == "done"


def test_hpo_slurm_launcher_command_requests_parallel_gpus(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "eval.sbatch.sh"
    log_path = tmp_path / "eval.log"

    command = build_slurm_command(
        script_path=script_path,
        slurm_log_path=log_path,
        job_name="hpo_demo_c001",
        max_parallel=3,
        sbatch_args=["--partition=gpu"],
    )

    assert command[:2] == ["sbatch", "--parsable"]
    assert "--wait" not in command
    assert "--parsable" in command
    assert "--gpus-per-node=3" in command
    assert "--partition=gpu" in command
    assert command[-1] == str(script_path)


def test_hpo_slurm_launcher_can_submit_through_ssh(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "eval.sbatch.sh"
    log_path = tmp_path / "eval.log"

    command = build_slurm_command(
        script_path=script_path,
        slurm_log_path=log_path,
        job_name="hpo_demo_c001",
        max_parallel=1,
        sbatch_args=["--partition=all"],
        slurm_host="slurm.example.test",
    )

    assert command[:2] == ["ssh", "slurm.example.test"]
    assert "sbatch --parsable" in command[2]
    assert "--wait" not in command[2]
    assert "--gpus-per-node=1" in command[2]
    assert "--partition=all" in command[2]
    assert str(script_path) in command[2]


def test_hpo_slurm_script_uses_compile_cache_slot_pool(
    tmp_path: Path,
) -> None:
    script_path = tmp_path / "eval.sbatch.sh"

    write_slurm_script(
        root=Path("/repo"),
        script_path=script_path,
        command=[
            "uv",
            "run",
            "python",
            "src/instantmimic/hpo/evaluate.py",
            "task.yaml",
        ],
        max_parallel=1,
    )

    script_text = script_path.read_text(encoding="utf-8")
    assert 'export PATH="$HOME/.local/bin:$PATH"' in script_text
    assert 'compile_cache_slots="2"' in script_text
    assert "compile_cache_parent" in script_text
    assert "flock -n" in script_text
    assert (
        'compile_cache_root="$compile_cache_parent/slot_${compile_cache_slot}"'
        in script_text
    )
    assert "TORCHINDUCTOR_CACHE_DIR" in script_text
    assert "TRITON_CACHE_DIR" in script_text
    assert "WARP_CACHE_DIR" in script_text
    assert "WARP_CACHE_PATH" in script_text
    assert "CUDA_CACHE_PATH" in script_text
    assert "trap release_compile_cache_slot EXIT" in script_text
    assert 'rm -rf "$compile_cache_root"' not in script_text


def test_hpo_hq_launcher_command_includes_parallel_resource() -> None:
    command = build_hq_command(
        command=[
            "uv",
            "run",
            "python",
            "src/instantmimic/hpo/evaluate.py",
            "task.yaml",
        ],
        job_name="hpo_demo_c001",
        max_parallel=2,
        submit_args=["--time-limit=1h"],
    )

    assert command[:3] == ["hq", "--output-mode=quiet", "submit"]
    assert "--wait" not in command
    assert "--name=hpo_demo_c001" in command
    assert "--resource" in command
    assert "gpus/nvidia=2" in command
    assert "--time-limit=1h" in command
    assert command[-5:] == [
        "uv",
        "run",
        "python",
        "src/instantmimic/hpo/evaluate.py",
        "task.yaml",
    ]


def test_hpo_hq_launcher_forwards_compile_cache_env() -> None:
    command = build_hq_command(
        command=[
            "uv",
            "run",
            "python",
            "src/instantmimic/hpo/evaluate.py",
            "task.yaml",
        ],
        job_name="hpo_demo_c001",
        max_parallel=1,
        submit_args=[],
        env={
            "UV_CACHE_DIR": "/tmp/uv-cache-demo",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/cache-demo/torchinductor",
            "TRITON_CACHE_DIR": "/tmp/cache-demo/triton",
            "WARP_CACHE_DIR": "/tmp/cache-demo/warp",
            "WARP_CACHE_PATH": "/tmp/cache-demo/warp",
            "CUDA_CACHE_PATH": "/tmp/cache-demo/cuda",
            "UNRELATED_ENV": "ignored",
        },
    )

    separator_index = command.index("--")
    forwarded_env = set(command[:separator_index])

    assert "--env=UV_CACHE_DIR=/tmp/uv-cache-demo" in forwarded_env
    assert (
        "--env=TORCHINDUCTOR_CACHE_DIR=/tmp/cache-demo/torchinductor" in forwarded_env
    )
    assert "--env=TRITON_CACHE_DIR=/tmp/cache-demo/triton" in forwarded_env
    assert "--env=WARP_CACHE_DIR=/tmp/cache-demo/warp" in forwarded_env
    assert "--env=WARP_CACHE_PATH=/tmp/cache-demo/warp" in forwarded_env
    assert "--env=CUDA_CACHE_PATH=/tmp/cache-demo/cuda" in forwarded_env
    assert not any("UNRELATED_ENV" in part for part in command)


def test_hpo_hq_launcher_defaults_compile_cache_env() -> None:
    command = build_hq_command(
        command=[
            "uv",
            "run",
            "python",
            "src/instantmimic/hpo/evaluate.py",
            "task.yaml",
        ],
        job_name="hpo_demo_c001",
        max_parallel=1,
        submit_args=[],
        env={"USER": "demo.user"},
    )

    separator_index = command.index("--")
    forwarded_env = set(command[:separator_index])
    cache_root = "/tmp/instantmimic-hq-cache-demo.user/hpo/hpo_demo"

    assert f"--env=TORCHINDUCTOR_CACHE_DIR={cache_root}/torchinductor" in forwarded_env
    assert f"--env=TRITON_CACHE_DIR={cache_root}/triton" in forwarded_env
    assert f"--env=WARP_CACHE_DIR={cache_root}/warp" in forwarded_env
    assert f"--env=WARP_CACHE_PATH={cache_root}/warp" in forwarded_env
    assert f"--env=CUDA_CACHE_PATH={cache_root}/cuda" in forwarded_env
    assert not any("/tmp/warp_cache" in part for part in command)


def test_hpo_hq_launcher_defaults_cache_under_repo_root() -> None:
    command = build_hq_command(
        command=[
            "uv",
            "run",
            "python",
            "src/instantmimic/hpo/evaluate.py",
            "task.yaml",
        ],
        job_name="hpo_demo_c001",
        max_parallel=1,
        submit_args=[],
        env={"USER": "demo.user"},
        cache_parent="/cache/hq",
    )

    separator_index = command.index("--")
    forwarded_env = set(command[:separator_index])
    cache_root = "/cache/hq/hpo_demo"

    assert f"--env=TORCHINDUCTOR_CACHE_DIR={cache_root}/torchinductor" in forwarded_env
    assert f"--env=TRITON_CACHE_DIR={cache_root}/triton" in forwarded_env
    assert f"--env=WARP_CACHE_DIR={cache_root}/warp" in forwarded_env
    assert f"--env=WARP_CACHE_PATH={cache_root}/warp" in forwarded_env
    assert f"--env=CUDA_CACHE_PATH={cache_root}/cuda" in forwarded_env
    assert not any("/tmp/instantmimic-hq-cache" in part for part in command)


def test_hpo_hq_launcher_reuses_cache_across_cycles() -> None:
    def compile_cache_root_for(job_name: str) -> str:
        command = build_hq_command(
            command=[
                "uv",
                "run",
                "python",
                "src/instantmimic/hpo/evaluate.py",
                "task.yaml",
            ],
            job_name=job_name,
            max_parallel=1,
            submit_args=[],
            cache_parent="/cache/hq",
        )
        return next(
            str(Path(part.removeprefix("--env=TORCHINDUCTOR_CACHE_DIR=")).parent)
            for part in command[: command.index("--")]
            if part.startswith("--env=TORCHINDUCTOR_CACHE_DIR=")
        )

    cycle_1_root = compile_cache_root_for("hpo_demo_c001")
    cycle_2_root = compile_cache_root_for("hpo_demo_c002")

    assert cycle_1_root == "/cache/hq/hpo_demo"
    assert cycle_2_root == cycle_1_root


def test_resolve_evaluator_launch_config_prefers_cli_over_task_cfg() -> None:
    task_cfg = OmegaConf.create(
        {
            "evaluator": {
                "launcher": "slurm",
                "max_parallel": 2,
                "slurm_host": "slurm.example.test",
                "slurm_sbatch_args": ["--partition=cfg"],
                "hq_submit_args": ["--time-limit=30m"],
            }
        }
    )

    launch_cfg = resolve_evaluator_launch_config(
        task_cfg,
        launcher="hq",
        max_parallel=4,
        slurm_sbatch_args=["--partition=cli"],
        hq_submit_args=["--time-limit=1h"],
    )

    assert launch_cfg.launcher == "hq"
    assert launch_cfg.max_parallel == 4
    assert launch_cfg.slurm_host == "slurm.example.test"
    assert launch_cfg.slurm_sbatch_args == ("--partition=cli",)
    assert launch_cfg.hq_submit_args == ("--time-limit=1h",)


def test_opencode_defaults_reasoning_effort_to_variant() -> None:
    task_cfg = OmegaConf.create(
        {
            "planner": {
                "platform": "opencode",
                "model": "vendor/model",
            },
        }
    )

    agent = agent_runtime.resolve_agent_spec(
        task_cfg,
        "planner",
        default_model=None,
        default_reasoning_effort="high",
    )

    assert agent.backend == "opencode"
    assert agent.opencode_variant == "high"


def test_opencode_explicit_variant_overrides_reasoning_effort() -> None:
    task_cfg = OmegaConf.create(
        {
            "planner": {
                "platform": "opencode",
                "model": "vendor/model",
                "reasoning_effort": "high",
                "variant": "max",
            },
        }
    )

    agent = agent_runtime.resolve_agent_spec(
        task_cfg,
        "planner",
        default_model=None,
        default_reasoning_effort="minimal",
    )

    assert agent.opencode_variant == "max"
