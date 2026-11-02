# Optional Local Workflows

These workflows are self-contained. For local GPU jobs, execute
directly on an accessible device after observing physical occupancy. Observation
is not a reservation; do not terminate other workloads. For comparisons, record intended
load/contention and exclude timing only for documented violations of those
conditions. Keep legitimate tails, GC pauses and representative background load
in the result. Use Slurm/HQ when selected and available; do not bypass scheduler
allocation on a managed execution path.

A coordinator does not grant GPU access. Inspect the active execution policy
rather than assuming that every agent execution path has the same device
access. If the current execution path does not expose the required GPU/backend, do not run
GPU-backed train, play, render, benchmark, profile or compile validation through
that path. Do not pass explicit GPU IDs to bypass missing discovery. Run GPU
work only from an authorized path that exposes the device. CPU-only rendering,
profiling, benchmarks and checks do not require GPU access; select their tools
and execution boundary independently.

Route raw captures and generated run output according to the active repository
artifact policy. Keep application outputs in their normal project locations
(training runs in `logs/`). Keep raw profiler captures,
curated tables, figures, and summaries in the owning `workspaces/<topic>/<initiative>/<task>/`
directory. Keep reusable caches outside the repository.

Follow the repository's artifact policy and workspace tool when present. If
neither exists, use an ignored task directory and keep runtime independent of
it. Do not require optional host integrations.
