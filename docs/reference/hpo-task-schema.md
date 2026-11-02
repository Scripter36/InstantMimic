# HPO task schema

An HPO task file describes what a search may change, how candidates are
evaluated and compared, and which planner proposes them. `validate_task` in
`src/instantmimic/hpo/schema.py` checks the file and rejects unknown keys;
objective-specific checks are in `src/instantmimic/hpo/objective/`.

To see the included task fully resolved:

```bash
uv run run_hpo --config-name hpo/backflip_pd --cfg job --resolve
```

`validate_configs` checks the task files in the repository as well:

```bash
uv run python -m instantmimic.tools.validate_configs --allow-unprepared-assets --json
```

## Task layout

| Block | Meaning |
| --- | --- |
| `name`, `description` | Task identity and planner context. |
| `parameter_space` | Required nonempty list of exact training override paths. |
| `forbidden_prefixes` | Additional forbidden roots or dotted prefixes. |
| `batch` | Motions/configs, seeds, fixed overrides and trial/evaluation settings. |
| `objective` | Comparator mode, strategy and its metric/budget policy. |
| `planner` | Model CLI, instructions, permissions, timeouts and proposal policy. |
| `runtime` | Final cycle index, resume directory, dry run and evaluator launcher. |
| `hpo_tracking` | Configured logging backend and artifact policy. |

The controller sets `state_dir` and `output_root` itself, so leave them out of
the task. Shared defaults are under `config/hpo/`.

## Search boundary

`parameter_space` lists the exact dotted Hydra paths the planner may change,
and `forbidden_prefixes` lists paths it may never touch: `env` covers `env` and
every `env.*` key. For a search over training settings only, forbid the
environment, evaluation, seeds and budgets so every candidate is measured the
same way. The controller validates each candidate's full config before
training it.

## Objectives

| Mode | Strategies |
| --- | --- |
| `scalar` | `wall_clock`, `time_to_reward_threshold`, `time_to_tracking_benchmark`, `frames_to_tracking_benchmark`, `tracking_benchmark_auc` |
| `pareto` | `pareto`, `learning_curve_pareto`, `axes_pareto` |

A scalar search keeps the best candidate for one objective. A Pareto search
keeps the whole trade-off front under `state/objective/` and also recommends one
representative from it.

For a learning-curve Pareto objective, specify one budget axis:
`active_time_budget_seconds` or `frame_budget`. Define `final_quality.key`,
`goal`, seed `aggregate`, checkpoint `statistic` (`terminal` or `best_per_seed`),
its noise `epsilon`, and `anytime_quality.epsilon`. Each complete candidate
produces a final-quality/AUC point from evaluations of every configured seed.
Use `terminal` if you will use the last checkpoint and `best_per_seed` if you
will pick the best one.

Active-time tasks use `terminal_time_tolerance_seconds`; frame tasks use
`terminal_frame_tolerance`. `terminal_budget_policy` is `strict` or
`hold_last_valid`, which extends the last valid evaluation to the end of the
budget.

Learning-curve representative selection supports `balanced`, `final_quality`,
`anytime_quality` and `final_quality_constrained`; the last requires
`anytime_quality_floor`. Generic `pareto` and `axes_pareto` have their own axis
and selection settings, checked by their validators in
`src/instantmimic/hpo/objective/`.

## Batch and execution

`batch.motions` maps each logical motion to its checkout-relative `config_path`
and `config_name`. `batch.seeds` fixes the seed set. Other batch fields include
`seed_batch`, `max_parallel`, `retries_per_seed`, `retry_triggers`,
`trial_timeout_seconds`, `stop_on_fail`, `compile_cache_scope`,
`fixed_train_overrides`, `eval_source`, `train_benchmark_recent_window`,
`playback`, `min_experiments_per_cycle`, `max_experiments_per_cycle` and
`lock_motion_configs`. `batch.fixed_train_overrides` holds the training,
stopping and evaluation settings shared by every candidate.

`batch.max_parallel` limits concurrent trials inside one evaluator, and
`runtime.evaluator.max_parallel` limits concurrent evaluators.
`runtime.evaluator.launcher` accepts `local`, `slurm` or `hq`;
backend-specific arguments and setup are described in the
[HPO guide](../guides/hpo.md#evaluator-launchers).

`planner.platform`, `model`, `reasoning_effort`, `sandbox` and `command` select
the external CLI; see
[planners, permissions and costs](../guides/hpo.md#planners-permissions-and-costs).

## State and resume

The canonical run directory is `logs/hpo/<task>/<run>/`.
`state/status.yaml` records the state format version, and resume refuses a
run written in an incompatible format.

```bash
uv run run_hpo --config-name hpo/backflip_pd \
  runtime.resume='logs/hpo/TASK/RUN' runtime.cycles=10
```

`runtime.cycles` is the last cycle to run: resuming after cycle 7 with 10 runs
cycles 8 to 10. Resume with the same task, objective, parameter space, seeds
and evaluator. An interrupted search keeps the state of its completed cycles.
