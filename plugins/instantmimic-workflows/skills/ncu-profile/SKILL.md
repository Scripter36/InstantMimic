---
name: ncu-profile
description: Use when inspecting a specific CUDA kernel's occupancy, memory throughput, stalls, or source-level metrics with Nsight Compute.
---

# NCU Profile

Use ncu after nsys identifies a kernel or narrow range. NCU is slower and noisier than nsys.

## Before Capture

- Read [Optional Local Workflows](../../references/optional-local-workflows.md) and
  follow the active repository or host GPU-coordination policy before launching
  local captures.
- Confirm the kernel belongs to the intended simulator/task/learning phase and
  that its invocation count is necessary; repeated expensive solves can be an
  orchestration defect even when each kernel is efficient.
- Pick one kernel name/filter or one tiny launch window.
- Keep the workload short.
- Preserve graph/compile flags unless the task explicitly studies non-graph behavior.
- Write reports under an `ncu/` directory in the owning task workspace.

## Basic Capture

```bash
ncu --target-processes all --set basic --kernel-name 'regex:<expression>' --launch-skip <n> --launch-count <m> --export <out/report> --force-overwrite uv run <command>
```

## Deeper Sections

Use only when basic capture points to a specific bottleneck:

```bash
ncu --target-processes all --section MemoryWorkloadAnalysis --kernel-name 'regex:<expression>' --launch-count <m> --export <out/report> --force-overwrite uv run <command>
ncu --target-processes all --section SchedulerStats --section WarpStateStats --kernel-name 'regex:<expression>' --launch-count <m> --export <out/report> --force-overwrite uv run <command>
```

## Report

State kernel filter, launch skip/count, command, report path, and the top bottleneck. Compare against non-profiler wall-clock only when available.
Follow the artifact-routing policy in the optional local workflows reference when
the investigation has multiple reports, plots, or temporary parsers; keep raw NCU
captures under `workspaces/<topic>/<initiative>/<task>/ncu/` and keep curated context and conclusions in
the same owning task workspace.
