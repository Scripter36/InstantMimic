# Phase 4: Audit and Report

Asserts A2, A6. Exit artifact: the recommendation set and a resumable handoff.

## Mechanical Audit

Recompute the result from raw evidence and require:

- unchanged environment, reward, evaluator, seed set, and protected hashes;
- unchanged task YAML and inline parameter space;
- identical GPU class and compatible software fingerprint;
- complete configured-seed evidence at matched evaluation points;
- correct active-time boundaries and separately reported queue/bootstrap/eval;
- no accepted crash, timeout, missing seed, or workload mismatch;
- candidate-level final/AUC reduction and epsilon-Pareto dominance recomputed
  independently of the search ledger;
- the recommended configuration materialized without temporary search scaffolds.

Any violation invalidates the affected candidate; do not repair the report by
silently omitting it.

## Select Recommendations

The frontier is authoritative; `best_experiment` and `best_result.md` are
compatibility projections of the frozen recommendation policy. Never use the word
`best` to mean the whole Pareto result, and do not declare one universal winner.
Report, when present:

- the final-quality champion;
- the anytime-quality champion;
- the balanced point;
- the final-quality champion subject to the frozen anytime-quality floor.

For each, include concrete config, source/environment fingerprint, per-seed
results, median and spread, active time, GPU-seconds, checkpoint/evidence paths,
and playback or fixed-evaluation artifact when applicable.

## Handoff

Persist the frontier, invalid candidates with reasons, issue ledger, remaining
budget, remote cleanup state, and exact resume command. Report setup and transfer
overheads separately so future searches can decide whether infrastructure work
is worthwhile without contaminating the learning objective.
