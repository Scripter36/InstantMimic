"""Evaluate an InstantMimic checkpoint with the pinned mjlab protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from omegaconf import open_dict

from instantmimic.bootstrap import prepare_environment_runtime
from instantmimic.config.artifacts import load_artifact_semantic_record
from instantmimic.config.loader import load_semantic_config
from instantmimic.play.config import (
    bundle_config_path,
    lower_playback_execution_config,
    resolve_artifact_paths,
)
from instantmimic.play.policy import policy_mean_into
from instantmimic.play.setup import build_playback_agent_runtime
from instantmimic.rl.runtime import copy_agent_observation_into, set_seed

from .benchmark_protocol import (
    EVALUATION_EPISODES,
    EVALUATION_HORIZON_SECONDS,
    EVALUATION_SAMPLING_MODE,
    METRIC_ORDER,
    QUALITY_THRESHOLDS,
    meets_quality_threshold,
)
from .metrics import BeyondMimicBenchmarkAccumulator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_evaluation_runtime(
    checkpoint_path: Path,
    *,
    seed: int,
    episodes: int = EVALUATION_EPISODES,
) -> tuple[Any, Any]:
    artifact = load_artifact_semantic_record(checkpoint_path)
    values = artifact.config.to_dict()
    values["seed"] = int(seed)
    env = values["env"]
    env["seed"] = int(seed)
    env["scene"]["num_envs"] = episodes
    env["episode_length_s"] = EVALUATION_HORIZON_SECONDS
    env["motion"]["sampling_mode"] = EVALUATION_SAMPLING_MODE
    env["observations"]["corruption"] = True
    env["randomization"]["push_enabled"] = False
    values["play"] = {
        "checkpoint": checkpoint_path.as_posix(),
        "num_envs": episodes,
        "steps": None,
        "use_cuda_graph": False,
        "action": {"mode": "mean"},
        "video": {"enabled": False},
    }
    semantic = load_semantic_config(values)
    log_dir = checkpoint_path.parent.parent.resolve()
    execution = lower_playback_execution_config(
        semantic,
        seed=int(seed),
        log_dir=log_dir,
    )
    execution = resolve_artifact_paths(
        execution,
        log_dir,
        bundle_cfg_path=bundle_config_path(log_dir),
    )
    with open_dict(execution):
        execution.env.scene.num_envs = episodes
        execution.checkpoint = checkpoint_path.as_posix()
    return semantic, execution


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    seed: int,
    episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    """Evaluate one episode per world; custom counts are labeled explicitly."""
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_checkpoint}")

    semantic, cfg = _build_evaluation_runtime(
        resolved_checkpoint,
        seed=int(seed),
        episodes=episodes,
    )
    set_seed(int(seed))
    prepare_environment_runtime()

    env = gym.make(cfg.task, cfg=cfg.env)
    try:
        setup = build_playback_agent_runtime(
            cfg=semantic,
            env=env,
            checkpoint_path=resolved_checkpoint,
            environment_cfg=cfg.env,
            restore_env_state=False,
        )
        env_raw = setup.env_raw
        required = (
            "snapshot_benchmark_reference",
            "get_benchmark_robot_state",
            "benchmark_end_effector_indices",
        )
        missing = [name for name in required if not hasattr(env_raw, name)]
        if missing:
            raise TypeError(
                f"Task {cfg.task!r} does not implement benchmark API: {missing}"
            )

        observations, _ = env.reset(seed=int(seed))
        agent_observations = torch.empty(
            episodes,
            setup.agent.obs_dim,
            device=setup.device,
            dtype=observations["policy"].dtype,
        )
        copy_agent_observation_into(
            agent=setup.agent,
            observations=observations,
            destination=agent_observations,
        )
        actions = torch.empty(
            episodes,
            setup.agent_action_dim,
            device=setup.device,
            dtype=agent_observations.dtype,
        )
        accumulator = BeyondMimicBenchmarkAccumulator(
            num_envs=episodes,
            end_effector_indices=env_raw.benchmark_end_effector_indices,
            device=setup.device,
        )

        episode_returns = torch.zeros(episodes, device=setup.device)
        steps = 0
        max_steps = int(env_raw.max_episode_length)
        while not bool(torch.all(accumulator.done).item()):
            if steps >= max_steps:
                unfinished = int(torch.count_nonzero(~accumulator.done).item())
                raise RuntimeError(
                    f"{unfinished} episodes did not finish within {max_steps} steps"
                )
            reference = env_raw.snapshot_benchmark_reference()
            policy_mean_into(setup.agent, agent_observations, actions)
            active = ~accumulator.done
            observations, rewards, terminated, truncated, _ = env.step(actions)
            episode_returns.add_(rewards.reshape(-1) * active)
            robot = env_raw.get_benchmark_robot_state()
            accumulator.update(
                reference=reference,
                robot_body_position_world=robot.body_position_world,
                robot_body_quaternion_world=robot.body_quaternion_world,
                robot_joint_velocity=robot.joint_velocity,
            )
            accumulator.mark_dones(
                terminated=terminated,
                truncated=truncated,
            )
            copy_agent_observation_into(
                agent=setup.agent,
                observations=observations,
                destination=agent_observations,
            )
            steps += 1

        metrics = accumulator.finalize()
        return {
            "protocol": {
                "name": (
                    "mjlab-nightly-side-kick"
                    if episodes == EVALUATION_EPISODES
                    else "side-kick-custom-episode-count"
                ),
                "evaluation_episodes": episodes,
                "horizon_seconds": EVALUATION_HORIZON_SECONDS,
                "sampling_mode": EVALUATION_SAMPLING_MODE,
                "observation_corruption": True,
                "pushes_enabled": False,
                "policy_action": "mean",
                "metric_order": list(METRIC_ORDER),
                "quality_thresholds": dict(QUALITY_THRESHOLDS),
            },
            "checkpoint": {
                "path": resolved_checkpoint.as_posix(),
                "sha256": _sha256(resolved_checkpoint),
            },
            "seed": int(seed),
            "steps": steps,
            "metrics": metrics,
            "episode_reward_mean": float(episode_returns.mean().item()),
            "meets_quality_threshold": meets_quality_threshold(metrics),
        }
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--episodes", type=int, default=EVALUATION_EPISODES)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    result = evaluate_checkpoint(
        args.checkpoint, seed=args.seed, episodes=args.episodes
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
