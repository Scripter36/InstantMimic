---
name: optimize-runtime
description: Iteratively optimize a specified workload by choosing suitable metrics and diagnostic tools, making causal code changes, remeasuring and validating outputs until no actionable axes remain. Use for sustained CPU, GPU, renderer, training, I/O or memory optimization; use a profiling skill for a single capture and tune-hyperparameters for learning-parameter search.
---

# Optimize Runtime

Own the complete loop: **measure/diagnose → explain → change → remeasure →
validate → reassess**. Continue while evidence supports another actionable axis.
One speedup, a successful capture or a target number is not completion. kFPS,
Nsight, CUDA and learning checks are workload-specific choices, not prerequisites.

## Choose the objective and tools

Read applicable repository instructions, the user's goal and actual runtime
consumers. Reuse existing authorization. Record a compact measurement contract:

- representative input/output and work volume, such as scenes, image resolution,
  data size, requests, frames or simulation rollouts;
- source snapshot (commit plus dirty diff when needed), relevant hardware/software,
  execution settings, cache/load state and output destination;
- primary metric with units, direction, aggregation and measurement boundary;
  constraints on output quality, latency tails, memory or other tradeoffs;
- repeat/warmup plan, effect resolution derived from variation/user needs,
  allowed numerical/visual differences and required behavioral regressions;
- allowed files/layers, compatibility/maintainability requirements, inherited
  budgets and material authority.

Infer a reasonable objective from the task and evidence, state it and ask only
when a material tradeoff remains unresolved. Choose tools to answer the current
causal question. Read [Measurement and Tools](references/measurement-and-tools.md)
for metric selection, CPU/native/Python, graphics, CUDA, memory and I/O routes.
Use a small representative probe before an expensive capture. Missing one tool
is not a blocker when another supplies the needed evidence. Do not install or
run an irrelevant GPU profiler for CPU-only work.

Preserve the requested outputs, quality and executed work. Lowering resolution,
samples, fidelity, precision or workload size is a tradeoff requiring explicit
scope, not an implementation speedup. Preserve required runtime modes and
behavior; apply graph, simulator or learning constraints only to paths that use
them. A simulator-internals restriction still permits common orchestration and
official-API reuse within that restriction.

Bound each measurement, diagnosis and regression run. Honor user/parent limits;
do not invent an overall cycle, time or no-improvement cutoff in place of the
exhaustion request. Ask only for missing material authority. Limits or unavailable
necessary evidence yield a resumable **incomplete** result, not exhaustion.

Keep baselines comparable and observe relevant interference: CPU load, storage,
GPU occupancy, thermal/clock changes or background jobs. Exclude invalid samples
with documented reasons, while retaining real tail/jank behavior;
do not terminate others. Read [Optional Local Workflows](../../references/optional-local-workflows.md)
before GPU work. For training experiments also use
[Experiment Protocol](../../references/experiment-protocol.md); its logger/seed
and playback conventions do not impose training work on a renderer or CPU task.

## Persistent evidence

Use one ignored task workspace following repository policy. Keep canonical
outputs in their normal location (training runs in `logs/` when applicable),
reusable caches outside the task directory, and record:

- `README.md`: contract, chosen tools/reasons, accepted result, next action, resume;
- `baseline.json`: immutable starting config, commands and raw metric samples;
- `axes.json`: discovered axes, status, evidence, hypotheses and closure reason;
- `cycles/NNN/`: source diff, config/tool fingerprints, raw diagnostic/measurement/
  validation evidence, decision and accepted revision for each attempt.

When invoked by a parent workflow, inherit its frozen scope/budgets and reuse
its workspace. Route baseline/axes/cycles to the supplied artifact prefix; keep
the parent's contract/state authoritative. Return status and raw evidence paths
without advancing parent phases, changing its gates or replacing independent
review. Do not create a second approval gate or state machine.

These are agent artifacts, not an application API or new controller. Use relative
artifact references and omit private endpoints. On resume verify code/config/
hardware fingerprints before reusing evidence. Rebaseline changed workloads
explicitly while retaining earlier results.

## Repeat the optimization cycle

1. **Measure and locate.** Establish a baseline under the objective's actual
   conditions, including cold startup or tail load when those are the goal.
   Use suitable sampling, traces, counters, allocation data or phase timings to
   attribute the cost. Profiling is optional when simpler measurements establish
   ownership. Read [Axis Review](references/axis-review.md); inspect the whole
   relevant path, not a fixed list of CUDA kernels or training phases.
2. **Explain one hypothesis.** Trace avoidable cost to its owner and dependencies.
   Estimate attainable benefit in the target metric relative to noise and code
   complexity. Eliminate unnecessary work before optimizing its implementation.
   Use a deeper tool only when it answers an unresolved question; record its
   selection or justified omission. Re-select tools when the bottleneck changes.
3. **Make a narrow change.** Protect the affected behavioral contract first and
   make one causal change (a tightly coupled implementation may be one change).
   Preserve readable code, resource/output ownership and supported fallbacks.
   Fix shared layers when they own the cost and check other consumers. Avoid a
   second runtime or input-specific branch merely to favor the benchmark.
   Preserve other worktree edits.
4. **Remeasure under normal conditions.** Compare the candidate with the actual
   incumbent and retain initial-baseline progress. Match inputs, outputs, load,
   modes and timing boundaries. Use repeated/interleaved trials as needed for
   drift/noise. Benchmark with expensive diagnostic instrumentation disabled;
   an already calibrated production counter may remain part of normal operation.
   Capture the target metric and guardrails, not only a locally faster phase.
5. **Validate and decide.** Follow [Validation](references/validation.md) for this
   workload. Adopt a useful measured gain only when affected output quality,
   correctness, lifecycle, compatibility and resource checks pass at justified
   complexity. Reject guardrail violations and revert only this attempt's owned
   changes; honor explicitly permitted metric tradeoffs.
   Mark noisy/interfered/incomplete evidence inconclusive; resolve it before
   claiming a gain. A failed idea does not close an axis.
6. **Reassess the accepted path.** Obtain fresh relevant whole-path attribution
   using appropriate tools or measurements. Reopen changed assumptions, discover
   new axes and revisit prior closures. Rejected candidates leave the incumbent
   in place; test other supported hypotheses. Return to step 1 while actionable
   axes remain; repeated heavy Nsight captures are not mandatory.

For training, apply existing nsys/ncu/smoke/evaluation skills when their specific
capabilities are needed. Runtime implementation optimization and convergence
search are different tasks; use `$tune-hyperparameters` only when requested.

## Exhaustion and handoff

Read [closure rules](references/axis-review.md#closure-and-reopening) before ending.
Completion requires final-path attribution and normal-condition baseline-to-final
measurements, evidence-backed review of every discovered in-scope axis, passing
affected output/regression/guardrail checks and a maintainable intended diff.
No actionable or unexamined axis may remain. Required representative workloads
and applicable startup/resource checks must support the retained gains.

High utilization, one no-gain trial, fixed attempts or reaching a desired metric
does not exhaust a workload. Promising unmeasured axes or unavailable necessary
evidence stay open/blocked; do not relabel them out of scope. Follow explicit stop
requests and limits with an incomplete result and exact continuation.

Report **exhausted within the recorded scope and measurement resolution** only
when those conditions hold, not global optimality. Otherwise report **incomplete**
and the blocker/limit. Include chosen metric/tools, accepted/rejected changes,
measured gains and tradeoffs, quality checks, remaining bounds and artifact paths.
Commit only authorized changes; optimization does not imply push/publication.
