---
name: tune-hyperparameters
description: Derive and run bounded, training-only Pareto HPO for an InstantMimic environment, especially after migration. Use when selecting a safe parameter whitelist and blacklist, calibrating active-training-time and evaluation budgets, comparing learning quality against speed, or running the search locally or through SSH and Slurm.
---

# Tune Hyperparameters

Run this as one approved autonomous state machine. Treat the environment and
evaluator as the benchmark; tune only the learning system.

## Vocabulary

These terms have exact meanings here. Do not substitute a colloquial reading.

| Term | Meaning |
| --- | --- |
| candidate | One complete proposed configuration. It yields exactly one Pareto point, not one point per evaluation. |
| semantic axis | One idea being varied. The default axis is one scalar or list path; a coupled bundle is one axis with 2–4 complete labeled variants. |
| coupled bundle | A multi-path axis used only when changing one member alone produces an invalid, uncalibrated, or misleading candidate. Every member path must already be in `parameter_space`. |
| active time | Training seconds excluding benchmark, checkpoint, queue, bootstrap, and compile time. Budgets and AUC are measured in active time only. |
| final quality | The frozen quality statistic of the checkpoint that would actually be deployed or played, aggregated over the frozen seed set. |
| anytime quality | Normalized AUC of the quality curve over the active-time budget. |
| frontier | The epsilon-Pareto archive over `(final quality, anytime quality)`. It is the authoritative result. |
| anchor | The frontier point currently being extended. A temporary view, never a result. |
| frontier role | Which anchor is selected next: `balanced`, `final_quality`, or `anytime_quality`. |
| execution fingerprint | Source snapshot + dependency lock + Python/runtime + simulator/CUDA stack + GPU model + GPU sharing (one evaluator per GPU) + evaluator contract. Bootstrap happens once per fingerprint. |
| transport / executor | Independent fields. `transport: local \| ssh` is how the agent reaches the host running the outer loop; `executor: direct \| hq \| slurm` is who allocates the GPU. They map to `runtime.evaluator.launcher` (`local`, `hq`, `slurm`), which both scalar and Pareto tasks accept through one shared launcher layer. |
| invalid candidate | Any candidate whose seed crashed, diverged, timed out, changed evaluated work, or produced incomplete evidence. It never enters the frontier, regardless of its numbers. |

## Invariants

Assert these at every phase boundary. Later phases reference them by id.

- **A1 Orientation.** For new or unfamiliar work, read `README.md`, applicable
  repository instructions and task documentation before the composed config and
  runtime consumer. For a known task, read its canonical run state and any
  owning workspace README first. Obey applicable nested instructions and preserve unrelated changes.
- **A2 Benchmark is frozen.** Environment dynamics, assets, observations,
  actions, reward and reward scale, termination, reset and cross-episode state,
  evaluator settings, seeds, graph and compile modes, logger/checkpoint cadence,
  and search-budget controls are never search axes. Each evaluator runs alone
  on its GPU: co-locating evaluators (for example several clients under NVIDIA
  MPS) makes active training time depend on the co-tenants, so it invalidates
  the speed objective and active-time budgets.
- **A3 Correctness precedes search.** Require fresh target-environment
  correctness and stable-learning evidence first. HPO must not tune
  around a semantic defect. Exact source-simulator learning parity is not
  required unless a frozen migration contract made it a hard gate.
- **A4 Allowlist is exact.** `parameter_space` is a list of exact dotted Hydra
  paths, verified against the composed config before GPU work. The runtime
  treats it as the exact proposal allowlist and implements no discovery or
  review of its own.
- **A5 Authority.** Reuse the user's existing authorization. Ask once only for
  missing material scope or authority (remote setup, spending, model subprocesses,
  publication). Within authorized scope, execute and record routine decisions
  without another confirmation gate.
- **A6 Requested budgets.** Preserve the user's cycle count and per-trial active
  time. If cycles are unspecified, choose and state a finite count. Do not infer
  an overall wall-clock, GPU-hour or no-improvement cutoff from a per-trial budget;
  apply those overall limits only when explicitly requested. Freeze retry rules
  and preserve exact resume semantics.
- **A7 Quiet execution.** One contract summary, batched phase updates, occasional
  long-run heartbeats, one terminal report.

## Run State

The search is resumable because the runtime persists it. Read state from disk,
never from conversation memory.

```text
<run_dir>/state/
  status.yaml                    # schema_version 2; current cycle; resume pointer
  strategy.md                    # planner strategy and axis registry
  progress.md                    # per-cycle narrative
  results.tsv                    # one row per candidate, fixed header
  cycles/NNN/                    # per-cycle proposal and evidence
  objective/
    pareto_frontier.md           # AUTHORITATIVE result for a frontier task
    pareto_frontier.json         # machine-readable frontier
  best_result.md                 # the comparator's own rendering of the best
```

`status.yaml` records `last_status` (lifecycle: `completed`, `failed:<phase>`, or
`dry_run`) and `last_decision` (the comparator's verdict) as separate fields. A
scalar `discard` is an ordinary result, not a failure.

Resume refuses state from an incompatible schema version; start a fresh run
instead of editing `status.yaml`.

Resume by pointing the runtime at that directory and raising the terminal cycle:

```bash
uv run run_hpo --config-name hpo/<task> \
  runtime.resume=<run_dir> runtime.cycles=<final_cycle_index>
```

`runtime.cycles` is the final cycle index, not a delta: resuming a run at cycle
40 with `runtime.cycles=60` executes cycles 41–60.

## Resume Fast Path

When continuing a run, start from its run directory, `state/status.yaml` and exact
resume command. Read an owning workspace README only when one exists; the
controller's canonical state lives under `logs/hpo/`. Mechanically compare the persisted
run contract, including task YAML, run directory, inline parameter space,
objective, seeds, budget, evaluator, backend/runtime path, logger state, and
source/config fingerprint. If they match, launch the requested additional cycles.

Do not redesign the parameter space, change seed count, create a new run, or
modify runtime code before attempting the supported exact resume. If the task
YAML no longer matches the persisted run, stop before GPU work and report the
mismatch. For urgent unattended continuation, establish tmux or the requested
supervisor and verify the active cycle before documentation or broader validation.

## Phase Loading

Read only the reference for the active phase group, then persist its exit
artifact before advancing.

| State | Load | Exit artifact | Asserts |
|---|---|---|---|
| `intake` → `approved` | [Contract](references/phase-0-1-contract.md) | self-contained task YAML and frozen search contract | A1, A2, A3, A4, A5 |
| `approved` → `ready` | [Bootstrap](references/phase-2-bootstrap.md) | verified execution path and resumable full-search launch | A4, A6 |
| `ready` → `searched` | [Search](references/phase-3-search.md) | validated Pareto archive under `state/objective/` | A2, A6, A7 |
| `searched` → `reported` | [Review](references/phase-4-review.md) | recommendation and resumable handoff | A2, A6 |

Do not advance past an unresolved validity gate. Continue independent work, then
record the gate as `missing`, `invalid`, or `blocked` without weakening it.

## Resource Rule

Load only execution guidance needed by the already selected backend. Do
not load remote-resource procedures for a local continuation, or local procedures
for a frozen remote run. Model new execution choices as independent `transport`
and `executor`; do not reopen that choice on an exact resume. Repository policy overrides this skill's
generic defaults.

## Terminal Rule

Before reporting completion, reinspect the final state and require fresh evidence
for the frozen scope, resource fingerprint, evaluated work, per-seed validity,
objective extraction, and Pareto calculation. Otherwise report the bounded work
as `incomplete`; never promote a numerically attractive invalid candidate.
