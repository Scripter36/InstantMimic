---
name: experiment-report
description: Use when summarizing experiment results, comparing seeds, preparing tables, or reporting playback rewards and videos.
---

# Experiment Report

Use the metric and protocol requested by the task. For learned-policy quality,
evaluate the selected checkpoint; training reward is not evaluation evidence.
For speed, report a warm, unprofiled executed workload with rollout/update and
active-time definitions; playback is not a substitute for a training benchmark.

Before comparing or reporting results, read [Experiment Protocol](../../references/experiment-protocol.md)
and apply its benchmark-validity gate. Confirm the actual baseline or documented
equivalence, resolved configuration/runtime markers, and executed work before
interpreting performance differences. If the gate fails, mark the comparison
invalid, disclose the confounder, retract derived conclusions, and do not silently
replace the run. For shared-code performance changes, verify the required
regression coverage before reporting measurements. Keep the detailed policy in the
reference rather than duplicating it here.

Read [Optional Local Workflows](../../references/optional-local-workflows.md) and
follow its artifact-routing policy when the report needs tables, figures,
temporary parsing scripts, or context beyond the final response. Store the curated
report in the configured task-artifact location and reference canonical runs in
`logs/` when that layout is active.

Never present profiler time as ordinary throughput. Record interference checks,
hardware/software versions, graph/compile/precision modes, warmup/window,
executed frames and whether compilation, evaluation and checkpoint work are
included. Mark single-seed speed probes as such; do not imply convergence parity.
Separate implementation throughput from isolated kernel effects. Use actual
baseline code or demonstrate operational equivalence; flag differences in source
revision, backend, precision and work instead of attributing all gaps to kernels.

## Learned-policy Quality Fields

For checkpoint/quality reports, per seed:

- seed
- selected logger, logical group, and backend run identity/name
- run directory
- best checkpoint
- playback reward
- playback episode length
- video path, if produced

For a 3-seed setting:

- mean reward across the 3 best-checkpoint playback runs
- any failed/missing playback separately, not silently dropped

## Parsing Helper

```bash
uv run python plugins/instantmimic-workflows/scripts/parse_playback_results.py <log files...> --markdown
```

The helper parses `PLAYBACK_RESULT ...` lines and nearby `Saved video to:` lines.

## Output Shape

Use a compact table, then one short interpretation paragraph. For quality reports, name missing checkpoint/evaluation evidence and distinguish an optional unrequested video from a failed video request. For speed-only reports, use the timing fields above; do not launch playback just to fill this quality table.
