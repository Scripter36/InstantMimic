---
name: nsys-profile
description: Use when profiling end-to-end training or playback timeline, CUDA API overhead, NVTX ranges, or CPU/GPU launch behavior with Nsight Systems.
---

# Nsys Profile

Use nsys first for timeline and wall-clock attribution.

## Capture

Keep captures short. Preserve graph/compile flags.
Read [Optional Local Workflows](../../references/optional-local-workflows.md) and
follow the active repository or host GPU-coordination policy before launching
local captures.

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --output <out/profile> uv run train --config-name <config> trainer.max_iterations=<small>
```

To capture a PyTorch NVTX range such as `training_loop`, set
`NSYS_NVTX_PROFILER_REGISTER_ONLY=0` on the `nsys` process. Without it, ranges
emitted through PyTorch may not register as capture triggers even when they are
visible to NVTX tracing.

Run a minimal probe before a formal capture:

```bash
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --capture-range=nvtx \
  --nvtx-capture=training_loop \
  --output <out/probe> \
  uv run train --config-name <config> trainer.max_iterations=<small>
```

If the probe captures the range, use the same environment variable and NVTX
trigger options for the formal capture. Prefer this boundary over adding CUDA
Profiler API calls to application code; use the CUDA Profiler API only as a
fallback when the NVTX-trigger probe still fails.

Record the resolved runtime and actual executed phase; successful capture does
not prove that the intended graph nodes were traced. For simulator investigations,
count and attribute integration, position/velocity prefix, full forward and solver
work per control action, including reset/wrap/push boundaries. Compare counts to
the actual decimation and state consumers before treating native kernels as
necessary work. Use an unprofiled run for final throughput.

Detailed NVTX adds synchronization to expose phase wall time. Keep its results
separate from production timing. For detailed rollout NVTX where supported:

```bash
uv run train --config-name <config> runtime.nsys_nvtx_mode=detailed
```

For graph-node ownership/count analysis, add `--cuda-graph-trace=node` to the
capture command (check the installed `nsys profile --help`). The usual default
traces graphs as a whole, so it cannot establish internal solver/kernel counts.
Node tracing adds substantial overhead; use it for attribution, not speed rows.

## Summaries

```bash
nsys stats --report nvtx_sum <profile.nsys-rep>
nsys stats --report osrt_sum <profile.nsys-rep>
nsys stats --report cuda_api_sum <profile.nsys-rep>
nsys stats --report cuda_gpu_kern_sum <profile.nsys-rep>
```

Keep raw `.nsys-rep` captures under `workspaces/<topic>/<initiative>/<task>/nsys/`. When the study
needs a durable summary, follow the artifact-routing policy in the optional local
workflows reference and place tables, plots, commands, and conclusions in its
configured task-artifact location.
Do not retain a generated SQLite export beside its source `.nsys-rep`.

## Interpret

- `nvtx_sum`: repo phase/range timing
- `osrt_sum`: CPU/syscall overhead
- `cuda_api_sum`: launch/copy/sync API overhead
- `cuda_gpu_kern_sum`: kernel time candidates for ncu

Do not use profiler timings as final benchmark rows unless the task is profiling-specific.
