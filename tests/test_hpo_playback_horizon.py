"""Disabling evaluator video still runs bounded, headless playback."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from instantmimic.hpo import evaluate
from instantmimic.hpo.candidate_executor import hydra_overrides_from_command


@pytest.mark.parametrize(
    "overrides, expected_steps",
    [([], 321), (["play.steps=17"], 17), (["++play.steps=23"], 23)],
)
def test_nonvideo_playback_keeps_a_bounded_horizon(tmp_path, overrides, expected_steps):
    config_file = tmp_path / "synthetic.yaml"
    config_file.write_text(
        "play:\n  steps: null\n  video:\n    enabled: true\n  output: {}\n"
    )
    config = evaluate._resolve_config_spec(tmp_path, "synthetic")
    job = evaluate.Job(
        "candidate", "motion", config, 1, "group", "run", tmp_path / "train.log", ()
    )
    job.best_checkpoint = Path("/synthetic/best_agent.pt")
    command = evaluate._build_play_command(
        job,
        play_video=False,
        play_video_length=321,
        play_overrides=overrides,
        disable_default_play_overrides=True,
    )
    with initialize_config_dir(config_dir=str(tmp_path), version_base=None):
        cfg = compose(
            config_name="synthetic", overrides=hydra_overrides_from_command(command)
        )

    assert cfg.play.video.enabled is False
    assert cfg.play.steps == expected_steps


@pytest.mark.parametrize("steps", ["0", "-1", "null"])
def test_nonvideo_playback_refuses_an_interactive_horizon(tmp_path, steps):
    config = evaluate.ConfigSpec(
        "synthetic",
        tmp_path,
        tmp_path / "synthetic.yaml",
        "synthetic",
        "",
        "",
        "synthetic",
        "motion",
    )
    job = evaluate.Job(
        "candidate", "motion", config, 1, "group", "run", tmp_path / "train.log", ()
    )
    job.best_checkpoint = Path("/synthetic/best_agent.pt")
    with pytest.raises(ValueError, match="positive play.steps"):
        evaluate._build_play_command(
            job,
            play_video=False,
            play_video_length=321,
            play_overrides=[f"play.steps={steps}"],
            disable_default_play_overrides=True,
        )


@pytest.mark.parametrize("overrides, steps", [([], 321), (["play.steps=17"], 17)])
def test_nonvideo_evidence_uses_the_executed_horizon(
    tmp_path, monkeypatch, overrides, steps
):
    (tmp_path / "synthetic.yaml").write_text("play: {}\n")
    config = evaluate._resolve_config_spec(tmp_path, "synthetic")
    job = evaluate.Job(
        "candidate", "motion", config, 1, "group", "run", tmp_path / "train.log", ()
    )
    job.best_checkpoint = Path("/synthetic/best_agent.pt")

    def play(_command, *, stdout, **_kwargs):
        stdout.write(
            f"Episode 1 finished: length={steps} total_reward=2\n"
            f"PLAYBACK_RESULT episode_length={steps} episode_reward=2\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(evaluate.subprocess, "run", play)
    evaluate._run_play(
        job,
        play_video=False,
        play_video_length=321,
        play_overrides=overrides,
        disable_default_play_overrides=True,
        play_steps=None,
        dry_run=False,
    )

    assert job.play_returncode == 0
    assert job.episode_length == steps
    assert job.fall_count == 0
    assert job.truncated_count == 1
