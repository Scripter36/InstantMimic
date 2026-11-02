---
name: migrate-external-env
description: Migrate or audit an IsaacLab, Isaac Lab, mjlab, or other external simulation and reinforcement-learning environment in InstantMimic. Covers source discovery, behavioral contracts, assets, implementation, learning/evaluation integration, production optimization, evidence, and independent review.
---

# Migrate External Env

Run this as one approved autonomous state machine. Discover current source and
target behavior; never rely on a static framework API map.

A `$skill-name` reference below names another skill in this plugin. Codex
invokes it in that form; Claude Code invokes the same skill as
`/instantmimic-workflows:skill-name`.

## Vocabulary

These terms have exact meanings here. Do not substitute a colloquial reading.

| Term | Meaning |
| --- | --- |
| gate | One named acceptance requirement with an id, a claim, and an acceptance class. Gates are frozen in Phase 1 and never added, removed, or reworded later. |
| acceptance class | How a gate may be satisfied: `exact` (backend-independent formula, ordering, shape, scale, cadence, or state), `tolerant` (backend-sensitive, predeclared tolerance), `statistical` (rollout, distribution, success, or aggregate), `unsupported` (explicit omission plus its consequence). |
| gate result | `pass`, `fail`, or `missing`. `missing` means the evidence cannot establish the claim; it is never a pass. |
| frozen run contract | The Phase 1 approval output, persisted as `contract.json`. After approval it is read-only for the rest of the run. |
| learning-setting attempt | One frozen configuration plus its frozen seed set, run to its frozen budget. Changing only seeds does not create a new attempt. |
| materiality exception | A permission, declared in Phase 1 with a measured threshold, to keep non-native framework-tensor work in a hot path. Absent a Phase 1 declaration, there is no exception. |
| causal freshness | Evidence is stale only when a change can affect the gate it supports. Judge by the dependency map in Phase 7, never by file timestamp. |
| terminal status | Exactly one of `production-ready` or `incomplete`. |
| migration-correct | A functional subset is faithful. This is a description, never a terminal status. |

## Invariants

Assert these at every phase boundary. Later phases reference them by id.

- **I1 Orientation.** Read `README.md`, applicable repository instructions and
  task documentation. Establish the maintained source and authoritative runtime
  consumers before freezing the contract. Preserve unrelated changes.
- **I2 Authority.** Reuse existing user authorization. Ask once only for missing
  material scope or authority; routine in-scope implementation decisions do not
  require another confirmation. Record consequential decisions for review.
- **I3 No weakening.** Never weaken a frozen gate, disable graph/compile paths,
  or reduce work to manufacture parity or speed.
- **I4 Communication.** Give concise updates on findings and consequential
  changes, with periodic progress during long runs. Keep raw command/retry detail
  in run state rather than narrating it all.
- **I5 Missing is not pass.** A missing required capability or artifact is
  `missing`.
- **I6 Finite budgets.** Never use reward attainment as a loop condition. Freeze
  learning budgets and retry counts before execution; stop when exhausted.

## Run State

The run is resumable only if its state is on disk. Create one workspace at
Phase 0 and write every exit artifact into it. Use a repository workspace tool
when one is configured; otherwise create an ignored directory directly under
`workspaces/<topic>/<initiative>/<migration-slug>/`. Preserve that path in state
and continuation commands. Chat is not state.

```text
workspaces/<topic>/<initiative>/<migration-slug>/
  README.md       # human resume point: question, status, exact resume command
  state.json      # machine progress pointer, rewritten at each phase exit
  contract.json   # Phase 1 exit; read-only afterwards
  ledger.jsonl    # append-only source dispositions, issues, decisions
  evidence/       # raw artifacts referenced by the packet
  packet.json     # Phase 7 exit, built by migration_review.py
  review.json     # reviewer output, saved byte-for-byte unchanged
  verdict.json    # migration_review.py verdict output; authoritative
```

`README.md` stays the primary human resume point, per the repository's task
workspace contract. `state.json` is the machine pointer the phases advance:

```json
{
  "schema_version": 1,
  "state": "preflight | approved | implemented | optimized | reported",
  "workspace": "workspaces/<topic>/<initiative>/<migration-slug>",
  "phase_exits": {"<state>": ["<relative path>", "..."]},
  "blocked_gates": [{"id": "...", "reason": "..."}]
}
```

`contract.json` is not free-form: `migration_review.py` validates it and rejects
any other key set. Write exactly this shape, and expand `required_gates` to the
gates frozen in Phase 1.

```json
{
  "schema_version": 1,
  "original_user_request": "<the user's request, verbatim>",
  "source_identity": {"kind": "git_commit", "value": "<sha or lock identity>"},
  "required_gates": [
    {"id": "reward_formula", "claim": "<falsifiable statement>", "acceptance": "exact"}
  ]
}
```

`required_gates` contains blocking acceptance requirements only. Non-blocking
source return or speed context belongs in raw comparison artifacts and the
report, not a gate the verdict would mistakenly treat as blocking.
`gate.acceptance` must be one of the four acceptance classes above.

Artifact ids and kinds are free-form strings to the packet builder, so this skill
fixes the convention. Use the gate id as the artifact id prefix so
`review.json` `evidence` entries resolve unambiguously — the verdict step
validates them against packet artifact ids. Use these kinds: `source` (raw source
evidence), `diff`, `config` (resolved configuration), `test`, `differential`,
`coldstart`, `profile`, `benchmark`, `learning_manifest`. For example
`--artifact reward_formula.differential=differential=workspaces/<topic>/<initiative>/<slug>/evidence/reward_diff.txt`.

`ledger.jsonl` carries one JSON object per line with at least `{"phase", "kind", "subject",
"disposition", "evidence"}`; it is the private record that I4 keeps out of chat.

On resume, read `README.md`, then `state.json`, verify `contract.json` still
validates, and re-enter at the recorded state. Do not restart an earlier phase whose exit
artifacts are present and still causally fresh.

## Phase Loading

Do not preload all instructions. On entering a phase group, read its reference
completely, execute it, write its exit artifact into the workspace, update
`state.json`, then advance.

| State | Load | Exit artifact | Asserts |
|---|---|---|---|
| `preflight` → `approved` | [Phase 0–1](references/phase-0-1-contract.md) | `contract.json` | I1, I2, I6 |
| `approved` → `implemented` | [Phase 2–4](references/phase-2-4-port.md) | `ledger.jsonl`, oracles, vertical slice | I3, I5 |
| `implemented` → `optimized` | [Phase 5–6](references/phase-5-6-production.md), invoking `$optimize-runtime` | delegated optimization evidence under `evidence/optimization/` | I3, I6 |
| `optimized` → `reported` | [Phase 7–8](references/phase-7-8-review.md) | `packet.json`, `review.json`, `verdict.json` | I4, I5 |

Phase 5 delegates its iterative optimization procedure to
[Optimize Runtime](../optimize-runtime/SKILL.md). Migration retains the frozen
contract/budgets, phase state and final independent verdict; child completion
alone cannot mark the migration production-ready.

Do not advance when the current reference says a prerequisite is unresolved;
continue independent work and append the dependent gate to `state.json`
`blocked_gates`.

## Review Isolation

Independent review is a separate role, not another phase prompt in this
conversation. Build a strict raw packet from the repository root with
`uv run python plugins/instantmimic-workflows/scripts/migration_review.py`,
then start a fresh read-only reviewer with no conversation history and
`$review-external-env-migration`.

Prefer the host's native subagent only when it provides both a fresh context and
an enforceable read-only boundary. Otherwise, a new non-interactive CLI process
is an allowed fallback only when subprocess model execution, credentials, cost,
and permissions were included in the frozen run contract. Never use resume or
fork: both can carry implementation context.

Give the reviewer only the packet path, packet digest, repository root, and
review-skill name. Do not transmit summaries, diagnoses, suspected bugs, prior
verdicts, or desired classifications. If fresh review cannot be created, mark
the independent-review gate `missing`; a cold self-review may inform the report
but cannot produce `production-ready`.

## Terminal Rule

Reinspect the final diff, worktree, verdict, and causal evidence before reporting.
Report `production-ready` only when `verdict.json` says `accept` and every frozen
gate has fresh passing evidence after the last relevant change. Otherwise report
`incomplete`, even if a functional subset is migration-correct.
