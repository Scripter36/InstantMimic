# Phase 0–1: Preflight and Contract

Asserts I1, I2, I6. Exit artifact: `contract.json` in the run workspace, in the
exact shape given in `SKILL.md` § Run State.

## Phase 0 — Silent Read-only Preflight

Before asking the user anything:

1. Locate the source, target family, configs, assets, checkpoints, entrypoints,
   executable source oracle, and applicable repository instructions.
2. Follow the named project's official README, task docs and retirement or
   migration notices. Identify the currently maintained implementation before
   installing a simulator stack or selecting a comparison baseline. If the user
   requires an older revision, retain it and document that choice. Record the
   canonical URL, selected task, discovery date and exact commit/lock identity
   (plus dirty diff hash when needed) as raw source evidence. Freeze that chosen
   revision as `source_identity`; do not silently switch sources later.
3. Inspect worktree state, writable scope, dependencies, credentials, execution
   policy, accelerator availability at the intended execution boundary, expected
   runtime, and whether fresh review can enforce both context isolation and a
   read-only execution boundary.
4. Classify each needed capability as `ready`, `needs-approval`, `unavailable`,
   or `unnecessary`. Resource discovery does not grant authority.
5. Follow [Optional Local Workflows](../../../references/optional-local-workflows.md)
   for shared accelerators and environment-specific execution boundaries.
6. Create the run workspace and write an initial `state.json` with
   `"state": "preflight"`.

## Phase 1 — One Approval Card

Infer everything repository evidence can answer. Show one compact card with:

- source and requested target scope;
- consequential assumptions and backend limitations;
- blocking semantic, lifecycle, learning/evaluation, optimization, cold-start,
  profiling and independent-review gates. Include maintained-source selection,
  simulator orchestration and matched-performance evidence; freeze any numeric
  speed requirement only when requested or justified by baseline measurement;
- exact local/remote execution boundary, expected time, writes, and runs;
- per-setting iteration or wall-time budget, evaluation cadence and seeds, total
  learning-setting attempts, and whether achieved source return is blocking;
- additional authority needed, including GPU, network, subprocess model calls,
  credentials, commit, push, or publication;
- fallback for every unavailable capability.

Use this card to record the contract. Existing user authorization can already
cover the work; ask only for missing material authority or an unresolved scope
choice. Do not request confirmation again for authorized GPU runs, edits or local
commits. Publication, spending or a new backend still require applicable authority.

## Writing The Contract

On approval, write `contract.json` and advance `state.json` to `approved`.

Each gate becomes one `required_gates` entry. `id` is a stable slug used by
every later artifact and by the reviewer's output. `claim` must be falsifiable:
state what would have to be observed for the gate to fail. `acceptance` is one
of the four acceptance classes in `SKILL.md` § Vocabulary. Per I8, each claim
also names the path it is measured on, its known-bad control, where any
tolerance comes from, and its comparator. Freeze a `tolerant` bound only after
measuring same-path repeat noise on the production device.

Include a blocking execution-scope/budget gate whose exact claim records the
chosen seed set, per-setting stopping budget, evaluation cadence, maximum setting
attempts, source/target execution boundary, permitted simulator extensions and
material authority. Use literal values resolved from the user and preflight,
not a link to mutable chat or a later manifest. This keeps the agreed limits
inside immutable `contract.json` without inventing extra schema keys. The raw
learning manifest must account for every attempt against those frozen values.

Also freeze a finite per-blocker non-learning correctness-repair allowance in
that gate: three distinct causally supported hypotheses by default, unless the
user/contract specifies otherwise. This bounds Phase 6 semantic/lifecycle repair,
not ordinary optimization cycles or no-gain hypotheses. Record the literal chosen
limit before mutation; exhausting it means a blocked migration, not axis closure.

When simulator backends differ, freeze reward formula and evaluation protocol as
`exact` gates. Keep the source simulator's achieved return as non-blocking raw
comparison evidence unless the user explicitly requires a target. In that case,
freeze a blocking `statistical` gate and its tolerance. Every `required_gates`
entry is blocking to the mechanical verdict; acceptance class does not make it
optional. Keep learning viability blocking: finite-budget training must remain
numerically healthy and improve over a frozen untrained or random baseline under
the agreed evaluation protocol. That check alone is not discriminating (I8): a
port with a disabled normalizer or a slower precision still passes it. Add
learning-state invariants on the production path, such as normalizer sample
counts, optimizer steps and scheduler updates matching their formulas. When the
source and target share a simulator backend, also freeze a learning-speed
non-inferiority gate whose tolerance comes from the source's own seed spread.

Freeze what establishes simulator-orchestration and performance completion:
state dependency/call-count evidence, actual runtime-path markers, warm matched
throughput, and numerical/lifecycle regression. Record both the source default
and intended target default, simulator extension restrictions, fallback paths,
backend versions and which differences prevent isolated causal attribution.
Do not infer that specialized-kernel flags explain the whole source/target gap.
Include the delegated Phase 5 objective (units/direction/aggregation), quality and
resource guardrails, useful-effect resolution or fixed calibration rule, and
in-scope axis-closure requirement in the frozen gate claims. `$optimize-runtime`
inherits these values; it cannot reopen gates, extend learning attempts or add
an unrequested global cycle/time cap. Keep the existing strict contract schema.

Validate the file before proceeding; the packet builder rejects any other shape:

```bash
uv run python plugins/instantmimic-workflows/scripts/migration_review.py packet \
  --contract workspaces/<topic>/<initiative>/<migration-slug>/contract.json --repo-root . \
  --artifact probe=contract=workspaces/<topic>/<initiative>/<migration-slug>/contract.json \
  --output /dev/null
```

Exit 0 means the contract validates. A wrong key set fails loudly, for example
`error: contract keys must be [...]`. Every `--artifact` path must be a regular
file, so the probe passes the contract itself.

A schema error here is cheap; the same error discovered in Phase 7 invalidates a
completed run's handoff.

## Budget Assertions

Freeze these with the contract; they bound Phase 6 and are checked by I6.

- Default to one baseline setting plus at most two causally justified corrective
  settings.
- One setting may contain the frozen seed set; changing seeds alone does not
  create a new hypothesis.
- Never define `while reward < target` behavior.
- Never extend iterations, wall time, seeds, or attempts after seeing a result.

## After Approval

Later uncertainty is an implementation problem: choose the most source-faithful
in-scope option, append it to `ledger.jsonl`, and continue. If new material
authority is required, do not cross that boundary; record the dependent gate in
`state.json` `blocked_gates` and continue independent work.

Do not demand exact PhysX/MuJoCo-Warp trajectories. Do not relax a tolerance
after a failure. If `source_identity` changes, the contract is invalid: refresh
the affected evidence rather than editing the frozen file in place.
