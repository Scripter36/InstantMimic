# Phase 2–4: Discover, Test, and Port

Asserts I3, I5. Exit artifacts: `ledger.jsonl`, executable oracles, and the
vertical slice.

## Phase 2 — Discover the Source Contract

Trace composed source configuration into actual reset, step, train, play, and
evaluation consumers. Inventory the complete reachable behavior surface instead
of treating a predefined checklist as exhaustive.

For every discovered subsystem or behavior, append one `ledger.jsonl` line with
`phase`, `kind`, `subject`, `evidence`, and exactly one `disposition` from
`migrate`, `equivalent_replacement`, `intentional_omission`, or `unresolved`.
An `unresolved` disposition blocks any gate that depends on it (I5). Include ordering, timing, stochastic and persistent state,
cross-episode lifecycle, learning/evaluation behavior, and every specialized
feature owned by that environment.

The ledger should naturally discover items such as spaces and mappings, reward,
observation, termination, reset, model topology, normalization, initialization,
optimizer/scheduler/precision cadence, timeout bootstrap, and evaluation protocol
when they exist. These examples are probes, not a closed schema.

Trace target base classes, controllers and simulator adapters as well as source
task code. Inventory state readers/writers by stage: action preparation,
pre-solve state, integration, post-solve sensors, endpoint observation/reward,
automatic/manual/partial reset, motion wrap and external pushes. Identify full
forward, prefix and integration work at each boundary. A stock/native simulator
call can still repeat collision or solving unnecessarily.

Trace checkpoint restoration into the evaluation consumer. With an explicit
InstantMimic checkpoint, saved config plus CLI dotlist overrides defines playback;
an alternate config name alone does not establish a different evaluation protocol.

## Phase 3 — Build Oracles

State each observable contract before implementation, then build the smallest
focused regression that protects it. Prefer source-faithful pure functions,
executable source fixtures, and deterministic golden cases over source-text
assertions. Protect every discovered contract, including assets/topology,
lifecycle, learning/evaluation, and specialized behavior.

Demonstrate that critical tests fail under plausible ordering, scale, timing, and
reset mistakes. A control must fail by assertion; a control that crashes proves
nothing and leaves the gate missing. Run learning oracles (normalization, losses,
schedules) through the production captured or fused path as well as the eager
one. If the source cannot be observed, record missing evidence rather than
inventing a pass.

## Phase 4 — Implement the Vertical Slice

Choose the closest current InstantMimic family by runtime contract, not framework
name. Implement the smallest complete slice:

- canonical assets with provenance and explicit ordering;
- config model and Hydra composition;
- environment behavior and registration;
- train, play, checkpoint, and evaluation integration;
- focused contract tests.

Keep paths project-relative. Preserve a readable semantic oracle, but avoid a
second full runtime when pure functions or fixtures provide equivalent review and
differential value. Do not add source-framework abstractions or compatibility
aliases without a target-runtime need.

Before handing off to Phase 5, require the focused semantic/lifecycle checks and
applicable minimum finite-budget learning-viability evidence, reusing fresh
existing evidence when valid. Charge new baseline learning to the frozen attempt
budget; do not add a separate unbudgeted full training run. A known defect goes
to Phase 6 instead of being hidden by optimization.
