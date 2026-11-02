# Match the objective to the execution path

Tool names below are options, not installation requirements or a fixed pipeline.
Check what the current platform, backend and permissions actually support. Select
the least disruptive tool that can distinguish the live hypotheses; use another
when it supplies equivalent evidence. A missing tool blocks completion only when
necessary evidence cannot otherwise be established.

## Objective and comparison boundary

Choose the metric that captures the user's outcome, with units/direction and
explicit constraints. State tradeoffs rather than silently choosing for the user.
Tail percentiles need enough events and representative load/window to support the
claimed quantile; do not label the maximum of a few samples as p99.

| Goal | Possible primary metric | Work/constraints to hold or report |
| --- | --- | --- |
| Interactive rendering | CPU/GPU frame time, displayed frame latency or p95/p99 frame time | Scene, camera/motion, resolution, lighting/shadows, visual quality, vsync/present mode, frames in flight and capture path. |
| Offline rendering/video | Total export time or finished frames/second | Same frames/scene, warm/cold assets, render quality, readback and encode scope; a fast draw alone does not prove fast video output. |
| CPU batch/data work | Completion latency or items/bytes per second | Dataset/output, algorithm semantics, thread count, cache/storage state and peak memory. |
| Memory reduction | Peak/retained host RAM, GPU memory or allocation rate | Ownership/lifetime, outputs, workload coverage and latency/throughput guardrails. Time attribution alone cannot establish memory gains. |
| Request processing | Tail latency at stated load or sustainable throughput | Concurrency, queueing, error rate and resource limits; do not substitute average latency. |
| Training/simulation | Active-time throughput, step latency or time to fixed quality | Authored environment/horizon, algorithm/work, task/evaluation semantics, precision and graph/compile path. |

Warm steady state is useful only when it matches the goal. Startup, shader or
code compilation, first frame, cache misses and cold data access may be the goal
rather than exclusions. State whether loading, presentation, readback, encoding,
checkpoint/evaluation or I/O is inside the boundary. For a resource objective,
predeclare acceptable time/quality changes; do not automatically reject a memory
win solely because it is not a speed win.

For throughput, a removable critical-path time fraction `f` implies ideal gain
`f / (1 - f)`. This is not a general closure formula for memory, cost or tail
latency. Use metric-specific bounds and controlled observations for those goals.

## Tool selection

| Path/question | Suitable evidence/tools | Limits and interpretation |
| --- | --- | --- |
| Python CPU hotspots | Existing timers, `cProfile`, sampling such as `py-spy`, native-stack sampling where needed | Distinguish Python dispatch from native-library time and I/O waits. Deterministic profiling can distort call-heavy work. |
| Native CPU/thread contention | Benchmark harness, system sampling/counters such as `perf`, thread/wait traces and lock instrumentation | Identify compute, scheduling, locks, cache misses and memory bandwidth; a low CPU busy percentage alone is not a diagnosis. |
| Vulkan or other graphics | Renderer pass counters, GPU timestamp queries, CPU submission/present timings, graphics frame capture, supported graphics/system profiler | Separate scene updates, culling/draw submission, shadows, raster/shaders, fences/present, readback and encode. Nsight Systems can help when its graphics tracing supports the backend. A CUDA kernel profiler does not profile Vulkan graphics shaders. |
| CUDA timeline/launches | `$nsys-profile`, phase markers and graph-node tracing where needed | Trace the actual CUDA path, ownership and counts. Heavy tracing changes timing; use normal-condition measurements for adoption. |
| Necessary hot CUDA kernel | `$ncu-profile` or supported hardware-counter evidence | Use only when kernel memory/occupancy/instruction evidence is needed; do not route CPU or Vulkan shader work to NCU. |
| Allocation/retained memory | Allocation counters, Python `tracemalloc`, native heap profiling, allocator/GPU memory telemetry as appropriate | `tracemalloc` does not cover arbitrary native/GPU allocations. Distinguish live, reserved/cached, resident and peak memory at defined lifecycle points. |
| File/I/O pipeline | Phase/byte counters, I/O statistics, syscall/wait traces or equivalent | Separate storage waits, decoding/serialization, copies and compute; cache hits and missing work can mimic a speedup. |

Examples are backend-dependent; a graphics capture can explain passes/resources
without supplying representative production timing. Likewise command submission
or an asynchronous API's return time is not completed device work. Establish
completion with appropriate timestamps/fences or the application's real boundary,
without inserting diagnostic synchronization into the timed production path.

Use existing renderer markers/counters when sufficient. Record enabled debug
modes, validation layers, readback counters, tracing overhead and presentation
settings; diagnostic instrumentation can change CPU/GPU overlap. For pure CPU
work, neither GPU access checks nor Nsight installation is required.
