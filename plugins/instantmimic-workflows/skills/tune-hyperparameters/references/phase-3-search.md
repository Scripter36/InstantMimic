# Phase 3: Run the Bounded Search

Asserts A2, A6, A7. Exit artifact: the validated epsilon-Pareto archive under `<run_dir>/state/objective/`.

Launch the complete approved cycle budget in one unattended invocation. Observe
up to the first five cycles while the same process continues. This observation
window is neither a launch gate nor a stopping condition. Intervene only when
evidence shows an actual execution or contract failure.

## Candidate Gate

Before every launch, resolve the complete configuration and mechanically reject
the candidate if it changes a frozen path, protected file hash, evaluator field,
seed set, runtime-path marker, hardware class, or evaluated work. Validate types,
ranges, coupled constraints, and effective batch/sample counts.

Propose only training-system changes justified by current evidence. Specialized
algorithm knobs may be explored only after the task YAML is deliberately updated
to include their exact paths; discovering them does not authorize environment or
evaluator changes. Candidate validation must use the inline `parameter_space`
list as its exact allowlist.

## Evaluation Loop

For each candidate:

1. Run the frozen configured seed set under the same GPU class and concurrency policy.
2. Capture fixed evaluation quality at the frozen active-time cadence.
3. Mark the candidate invalid if any seed crashes, diverges, times out, changes
   work, or produces incomplete evidence.
4. Aggregate time by median and quality by the frozen aggregate; retain quality
   spread, median quality, and `max/median` timing.
5. Compute the frozen selected-checkpoint quality statistic and normalized
   active-time AUC, yielding one Pareto point for the complete candidate rather
   than one point per evaluation. Keep terminal quality as a stability field.
6. Update an epsilon-Pareto archive; never collapse the result to a weighted sum
   unless the frozen contract explicitly requests one.
7. Persist hypothesis, diff, fingerprints, raw evidence paths, verdict, and next
   direction before proposing another candidate.

Use queue time and GPU-seconds to schedule work, not to change candidate quality.
Parallel seeds reduce turnaround but do not alter the frozen active-time budget.

## Anchor Discipline

Select an anchor and change exactly one semantic axis. Prefer a single
whitelisted path. When one hypothesis requires coupled calibration, use
one named bundle whose 2–4 labeled variants each set the complete profile; reject
partial variants and unrelated member paths. Rotate between `balanced`,
`final_quality`, and `anytime_quality` frontier roles. A candidate that does not
beat its parent may still enter the archive when globally non-dominated. Persist
the point snapshot, anchor role, semantic identity, member paths, variant,
direction, and explored status; do not repeat an exhausted neighborhood.

Keep every untested allowed axis visible in a structured registry. Defer an axis
only with a path-specific reason. Natural-language memory may summarize this
registry but cannot close, reopen, rename, or merge axes.

## Recovery

### Assertion set R — retry classification

Before any retry, classify the failure. Retry only R1; never retry R2 unchanged.

- **R1** transient infrastructure: node loss, preemption, transport error,
  scheduler timeout, transient filesystem error. Bounded retries allowed.
- **R2** deterministic: config rejection, correctness failure, NaN, divergence,
  out-of-memory. A retry cannot change the outcome.

For R2, make a conservative in-scope repair when possible, restore
the last valid state otherwise, and append the issue and resolution to the run
ledger without asking the user mid-run.

Credential expiry, lost authority, exhausted cost, or unavailable artifacts are
terminal blockers. Preserve state and stop cleanly; do not broaden permissions.

## Stop

Complete the requested cycle count unless the user stops the run or an actual
execution/validity failure prevents progress. Apply overall time, GPU-hour or
no-improvement limits only when explicitly requested; a per-trial budget does not
imply them. Quality threshold attainment does not replace the cycle count.
Preserve the frontier and resumable state on interruption.
