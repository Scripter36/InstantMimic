# Phase 0–1: Freeze the Search Contract

Asserts A1–A5. Exit artifact: a self-contained HPO task YAML whose
`parameter_space`, `objective`, and `batch` blocks are frozen.

## Discover Before Asking

Inspect the migration report, resolved training config, runtime consumers,
learning curves, profiler evidence, checkpoints, and existing HPO
implementation. Determine what is already known before presenting one compact
approval card.

Separate target correctness from cross-simulator learning parity. A documented
source/target quality gap may become baseline context rather than a blocker when
semantics, lifecycle, native execution, and stable target learning are valid.
Stale evidence or a semantic defect remains blocking. Treat a new explicit
resource request as authority to propose a new backend, not as evidence that old
hardware timings remain comparable.

Collect or infer all decisions that could otherwise interrupt execution:

- baseline revision/snapshot and evidence paths;
- fixed evaluator metrics and protocol;
- allowed training files and parameter paths;
- excluded files, semantics, and parameter paths;
- seeds, GPU equivalence rule, target/backend, and concurrency;
- per-trial active-time budget, cycle count, retries and any explicitly requested
  overall limits;
- logging, artifact retention, remote mutation, cost, and agent permissions.

Preserve the requested run length and choose a stated finite cycle count only
when one was not supplied. Do not add an unrequested overall time, GPU-hour or
plateau cutoff. Authentication and material spending authority cannot be inferred.

Audit whether the repository can execute the contract: it needs a fixed-metric
evaluator at active-time checkpoints, configured-seed aggregation, Pareto archiving,
immutable source bootstrap, resume state, and a usable local/SSH/Slurm command
path. Remote checkout, installation, and path mapping are agent bootstrap work,
not application features. Put missing deterministic evaluator or Pareto support
and its CPU tests in the same approval card; do not discover that gap after GPU
work starts.

## Define the Inline Parameter Space

Inspect the fully composed train config and relevant runtime consumers. Select
the intentionally scoped learning parameters and write them directly in the
HPO task YAML as a list of exact dotted Hydra paths. Do not generate a
second manifest, expanded classification packet, fingerprint approval, or
runtime reviewer workflow.

Candidate categories may include topology, normalization, initialization,
optimizer, scheduler, precision, batching, rollout/update cadence, and other
algorithm-specific settings discovered in the implementation. This is discovery
guidance, not a mandatory list. Do not whitelist a path without tracing it.

A2 lists the frozen domains. Express them in top-level `forbidden_prefixes`,
so proposal validation rejects them mechanically:

```yaml
forbidden_prefixes: [schema_version, seed, task, env, trainer, experiment, play, runtime]
```

Write bare root segments, not dotted prefixes: matching is
`path == prefix or path.startswith(f"{prefix}.")`, so `env` covers `env.scene.num_envs`
while `env.` matches nothing. Derive the exact list from A2 and the task's own
config (`config/hpo/search_space/pd_training.yaml` is a tracked example); tasks
that tune a reward or objective surface add `objective` and `reward`. `parameter_space` is
the allowlist and this is the denylist — a path must clear both.

The agent preparing the task must verify each listed path against the composed
config or a known optional runtime consumer. The runtime only treats this list as
the exact proposal allowlist. It must not implement the agent's discovery or
review procedure.

Define the search unit as one semantic axis (see `SKILL.md` § Vocabulary).
Give a coupled bundle one descriptive identity and 2–4 complete labeled variants,
such as optimizer family plus its learning-rate rule and bounds. A bundle never
expands the authorization boundary (A4): check every member path against the
task's exact `parameter_space` list and frozen-domain rules, and compose and
validate each complete variant before GPU execution. Do not bundle independent
ideas merely to search faster.

Track explored, deferred, and closed regions by exact parameter paths. Put any
task-specific strategy in a short `planner.instructions` block in the same YAML;
do not create a task-specific program file.

## Freeze the Objective

Use the fixed evaluator's quality value; reward may be measured only if its
definition and scale are frozen. Never tune reward terms or substitute a new
quality proxy during search.

Use the candidate-level `learning_curve_pareto` comparator for a
training-only search.

`objective.mode` selects the comparator. `mode: pareto` accepts
`pareto`, `learning_curve_pareto`, and `axes_pareto`; a scalar strategy under it
is a validation error. Read `docs/reference/hpo-task-schema.md`
from the target checkout for accepted keys. Resolve that path from the checkout
root, not this skill's installed cache directory.

```yaml
batch:
  eval_source: train_benchmark
  seeds: [1, 2, 3]
  lock_motion_configs: true
  min_experiments_per_cycle: 1
  max_experiments_per_cycle: 4
  fixed_train_overrides:
    - ++trainer.stopping={type:time_budget,warmup_iterations:0,timeout_seconds:ACTIVE_TIME_BUDGET}
parameter_space:
  - agent.learning_rate.initial
  - agent.learning_rate.scheduler.threshold
  - agent.ratio_clip
forbidden_prefixes: [schema_version, seed, task, env, trainer, experiment, play, runtime]
objective:
  mode: pareto
  strategy: learning_curve_pareto
  active_time_budget_seconds: ACTIVE_TIME_BUDGET
  terminal_time_tolerance_seconds: ONE_BASELINE_ITERATION_OR_LESS
  final_quality: {key: fixed_quality, aggregate: median, statistic: best_per_seed, epsilon: 0.0}
  anytime_quality: {epsilon: 0.0}
  selection: final_quality_constrained
  anytime_quality_floor: BASELINE_MINUS_EPSILON
  guards: []
```

Replace `fixed_quality` with an immutable metric emitted on every
`BENCHMARK_EVAL`. Derive the floor and both epsilons from baseline calibration;
zero is only valid when measurement resolution and repeats justify it.
Match the quality statistic to the checkpoint that will actually be deployed or
played. When each seed emits its own evaluator-selected best checkpoint, use
`best_per_seed` and aggregate those seedwise maxima; configure checkpoint
selection to use the same immutable evaluator metric. Keep `terminal` only for
workflows that intentionally deploy the budget-terminal policy. Record terminal
quality and collapse separately as stability diagnostics, never as a silent
replacement for selected-checkpoint quality.
`selection` is a compatibility recommendation, never the authoritative result.
Freeze `trainer.benchmark.interval_seconds`; choose an active-time budget that is
an integer multiple of it. Set terminal tolerance from measured baseline
iteration granularity, and reject a candidate whose final evaluation falls
outside that budget window. Keep all stopping and evaluator paths outside the
proposal override whitelist.

Each candidate first yields a fixed active-time learning curve. Aggregate the
same configured seed set at every matched evaluation target, then reduce the complete
curve to one candidate-level Pareto point:

```text
(selected-checkpoint fixed_eval_quality, normalized AUC over active time)
```

Intermediate evaluations are raw evidence, promotion signals, and AUC samples;
they are not independent Pareto candidates. Default quality aggregation to mean
when the evaluator protocol does not define it; for multi-seed robustness,
prefer median when establishing a new protocol and freeze the choice.
Require every configured seed to finish. Record median quality, quality spread, and
`max/median` time as stability fields; do not drop a failed or slow seed. Queue,
remote bootstrap, compilation, and evaluation time are operational metrics.

Freeze an epsilon/noise margin from baseline repeats so negligible fluctuations
do not change dominance decisions.

## Approval Card

When calibration is missing or uses different hardware, define a baseline rule
for comparison horizon, evaluation cadence and noise margin. Preserve a per-trial
active-time budget already supplied by the user. Normally collect 12–20 useful
evaluation points without excessive overhead. Calibration does not authorize
shortening the requested search or adding an overall time limit.

Reuse existing authorization; this card records the contract and asks only for
missing material authority. It is not another approval step when scope, GPU and
budget are already authorized.

Show only: frozen benchmark identity, whitelist summary, evaluator and quality
aggregate, Pareto axes, seeds, baseline/calibration rule, cycle count and per-trial
budget, user-specified overall limits, execution target and concurrency, missing
adapter work, expected remote
writes, artifact policy, permissions, and stop rules. Approval transitions the
run to unattended execution.

Use this compact shape:

```yaml
benchmark: {snapshot, evaluator, fixed_semantics}
scope: {allowed, frozen}
objective: {quality_aggregate, time_aggregate, guards, epsilon_rule}
budget: {seeds, calibration_rule, cycles, trial_active_seconds, retries}
resource: {transport, executor, target, gpu_class, concurrency}
bootstrap: {remote_root, agent_actions, writes, retention}
permissions: {remote_mutation, gpu_use, cost, subprocess_agents}
stop: {user_limits, execution_failures}
```
