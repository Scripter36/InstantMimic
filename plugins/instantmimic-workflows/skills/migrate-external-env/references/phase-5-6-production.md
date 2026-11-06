# Phase 5–6: Delegated Optimization and Migration Repair

Asserts I3, I6. Phase 5 delegates execution to `$optimize-runtime`; this reference
owns the migration-specific inputs, acceptance requirements and handoff. The
measurement/change/remeasure/validation loop and axis-closure procedure live only
in [Optimize Runtime](../../optimize-runtime/SKILL.md).

## Phase 5 — Invoke Optimize Runtime

Enter after the Phase 4 vertical slice, semantic/lifecycle checks and applicable
minimum learning-viability check pass. Missing correctness evidence or a known
semantic/learning defect goes to Phase 6; do not optimize around it.

Pass the child skill:

- unchanged `contract.json`, source/target identity and resolved production config;
- the frozen performance objective, quality/numerical tolerances, useful-effect
  threshold or predeclared calibration rule, representative work and baselines;
- allowed layers/simulator extensions, graph/compile/precision and other fixed
  semantics, existing learning/repair budgets, remaining attempts and authority;
- fresh oracle/lifecycle/learning evidence and final-code dependency information;
- the existing migration workspace, `evidence/optimization/` artifact prefix and
  the migration-specific acceptance requirements below.

Read and run the child skill, not a parallel local optimization loop. Reuse the
parent workspace/authority; do not create a second contract, approval gate or
migration state machine. The parent `README.md`, `contract.json`, `state.json`
and `ledger.jsonl` remain authoritative. Store child baseline, axes and cycles
under the supplied prefix. Track every new learning-setting attempt against the
parent cap; a valid equivalence bridge may preserve independent learning evidence
instead of launching another complete training run.

### Migration-specific acceptance requirements

These constrain the child loop; they do not prescribe a second implementation
procedure or a universal kFPS ratio.

- **Production ownership.** On device-backed paths, attribute actual graph nodes/
  kernels to learning, simulator, task/reset and bookkeeping. Task-owned hot work
  must use native device execution; for MuJoCo Warp, fused Warp task kernels.
  Framework exceptions require the frozen measured materiality allowance. Actor,
  critic and optimizer math may remain in their learning framework.
- **Simulator stages.** Establish expected/observed integration, prefix, full
  forward, collision/constraint and solve counts at steady and reset/wrap/push
  boundaries. Explain each refresh by its consumers and actual eligibility/
  fallback path. Helper names or disabled specialization flags do not prove
  narrow or equivalent execution. CPU counts alone do not establish GPU behavior.
- **Lifecycle and sensors.** Preserve reset observations and reset-to-next-step,
  partial/automatic resets and external state edits. Capture solve-stage force/
  acceleration history before a prefix clears those slots. Required current
  dynamics and conservative fallbacks must remain valid. Shared-contact task
  checks do not establish independent force-history/trajectory equivalence;
  retain the applicable learned-policy evaluation and declared tolerances.
- **Capacity safety.** Contact/constraint peaks and overflow checks must cover
  every physics substep, representative/trained policies, falls/terminations and
  automatic resets, with justified headroom. Smoke/random actions alone cannot
  justify smaller buffers.
- **Source comparison.** Use the selected maintained source and target default,
  matched work/modes where possible and disclosed backend differences. Supply
  normal-condition source/target measurements and explain material gaps by actual
  phases/calls. A required unavailable baseline remains missing; do not substitute
  historical numbers or attribute all gaps to absent specialized kernels.
  Matching the source's speed is not enough (I8): also measure the target's own
  established path for the same model and work, and explain any gap to it. A
  fallback to fuller dynamics needs a named consumer that requires it. Run the
  source baseline stock, without added instrumentation, and use one throughput
  definition for both sides.
- **Startup reliability.** Where compile/graphs are used, verify isolated cold
  startup and actual authored capture/replay. Warm-cache success after a failed
  cold launch is not repair; prewarm/capture ordering must pass on the final code.
  If success depends on interrupted/partial caches, reproduce that state or add
  a deterministic prewarm regression rather than relying on a warm retry.

### Return to the migration controller

Return child `exhausted within scope` or `incomplete`, accepted code/config
identity, axis registry and raw diagnostic/measurement/regression paths. Record
these in the existing migration state/ledger and `phase_exits`; no schema change
or new production runtime API is needed.

Advance `implemented` → `optimized` only when the child reports scoped exhaustion
and every frozen Phase 5 requirement has fresh passing evidence on that code.
A child status is not a mechanical migration verdict. Limits, blocked/open axes
or a failed migration requirement keep the migration incomplete; record the
missing/failed gates and continue independent work without relabeling the state.

For Phase 7, forward raw source/config/diff, final attribution, sample/call-count,
validation and factual execution-manifest artifacts. Keep the child axis registry,
hypotheses, adoption decisions and self-assessment as implementation working
artifacts; they are not reviewer inputs or proof of passing a gate.

## Phase 6 — Repair Migration Correctness or Learning

Optimization candidates and inconclusive timing stay inside the child loop.
Use this phase for source/target semantic or lifecycle mismatches, learning
failure, or migration-specific acceptance failures. Make a source-faithful fix,
refresh affected oracles/evidence and return to Phase 5 when its entry conditions
hold. Do not edit a frozen contract in place; invalid source evidence requires a
replacement contract/run with earlier artifacts retained.

Track non-learning semantic/lifecycle repair hypotheses against the separate
per-blocker allowance frozen in Phase 1. Exhausting it leaves that migration gate
blocked and the result incomplete; preserve evidence and continue independent
work. This allowance does not cap ordinary child optimization cycles, count a
no-gain candidate as a correctness repair, or establish axis exhaustion.

For learning/reward failures, retain the existing finite-budget rules:

- finish only the frozen per-setting budget; never wait for a reward threshold;
- one frozen config and seed set is one learning-setting attempt;
- baseline plus at most two causally justified corrective settings by default,
  within the actual frozen cap; no blind tuning or seed replacement;
- when semantics and viability pass, report non-blocking source-return differences
  as backend statistical context, not an excuse to retrain indefinitely;
- exhausted attempts or an unmet blocking requirement means incomplete, with
  preserved evidence and continuation. It does not mean optimization exhaustion.

After a correctness repair, the child's dependency/closure checks must revisit
affected measurements, validation and axes before the migration can be optimized.
The Phase 7 raw-evidence review remains mandatory and independent.
