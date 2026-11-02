"""An isolated trial timeout obeys the configured retry policy."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from instantmimic.hpo import evaluate


@pytest.mark.parametrize(
    "retry_triggers, failures, expected_attempts, expected_returncode",
    [(["any_failure"], 1, 2, 0), (["any_failure"], 3, 3, 124), (["cublas"], 1, 1, 124)],
)
def test_isolated_timeout_obeys_retry_policy(
    tmp_path,
    monkeypatch,
    retry_triggers,
    failures,
    expected_attempts,
    expected_returncode,
):
    config_path = tmp_path / "synthetic.yaml"
    config_path.write_text("defaults: [_self_]\n")
    task = OmegaConf.create(
        {
            "name": "retry-test",
            "batch": {
                "seeds": [1],
                "seed_batch": "off",
                "max_parallel": 1,
                "eval_source": "train_benchmark",
                "retries_per_seed": 2,
                "retry_triggers": retry_triggers,
                "trial_timeout_seconds": 1,
            },
        }
    )
    experiment = evaluate.ExperimentSpec(
        "candidate",
        "",
        {"motion": evaluate.MotionRunSpec("motion", tmp_path, "synthetic", ())},
    )
    attempts = []

    def launch(job, _overrides, *, dry_run):
        assert not dry_run
        attempts.append(job.train_retry_count)
        job.train_log.write_text("training\n")
        returncode = None if len(attempts) <= failures else 0
        job.process = SimpleNamespace(poll=lambda: returncode)

    def finalize(job, returncode):
        job.train_returncode = returncode

    monkeypatch.setattr(evaluate, "_launch_job", launch)
    monkeypatch.setattr(evaluate, "_finalize_train_job", finalize)
    monkeypatch.setattr(evaluate, "_terminate_running_jobs", lambda _jobs: None)
    monkeypatch.setattr(evaluate, "_trial_timed_out", lambda _job, _policy: True)
    monkeypatch.setattr(
        evaluate, "_populate_train_benchmark_metrics", lambda *a, **k: None
    )
    monkeypatch.setattr(evaluate.time, "sleep", lambda _seconds: None)

    jobs = evaluate._run_jobs_for_experiment(
        task, experiment, tmp_path / "evaluation", dry_run=False
    )

    assert len(attempts) == expected_attempts
    assert len(jobs) == 1
    assert jobs[0].train_retry_count == expected_attempts - 1
    assert jobs[0].train_returncode == expected_returncode
