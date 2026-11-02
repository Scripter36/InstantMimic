# Discover and close optimization axes

Use source ownership plus current measurements of the relevant whole path. These
are discovery probes, not an exhaustive checklist or mandatory rewrite. Inspect
workload-specific costs. Tool choice follows each question: a CPU allocation
problem, Vulkan graphics pass and CUDA kernel need different evidence.

Group related small costs; many tiny operations can form a material batching or
dispatch axis. High utilization does not prove that work is necessary. For time/
throughput, attribute the actual critical path, calibrated against normal-condition
measurements. Summed overlapping GPU durations or a perturbed trace are not an
elapsed-time fraction. Ideal throughput gain for removable fraction `f` is
`f / (1 - f)`; use separate resource/tail-latency bounds for other objectives.

| Axis | Evidence to inspect | Candidate direction and guard |
| --- | --- | --- |
| CPU algorithms/data layout | Work counts, sampled stacks, native/Python boundary, cache/bandwidth counters | Remove redundant work, improve locality/vectorization; preserve algorithm/output semantics. |
| Dispatch/batching | Tiny calls, Python callbacks, per-item/term/draw loops | Cache invariants or batch submissions without dropping required work or callbacks. |
| Parallelism/contention | Thread/queue/lock waits, task imbalance, oversubscription and CPU/GPU overlap | Improve scheduling/ownership; preserve dependencies and avoid a favorable but unrepresentative concurrency level. |
| Memory/lifetime | Peak/live/reserved allocations, copies, churn, retained objects and device residency | Reuse or release resources and reduce copies; check leaks, lifetime, latency and resource tradeoffs. |
| Graphics/render pipeline | CPU scene/culling/submission, per-pass GPU time, geometry/shadow/shader work | Eliminate redundant updates/draws or improve necessary passes; preserve images, scene coverage and quality. |
| Synchronization/presentation | Fences, barriers, scalar/readback stalls, frames in flight, vsync and queues | Remove dependency-free barriers; preserve ordering, pacing, latency and completion semantics. |
| I/O/output | Bytes, syscalls, decode/serialize/upload/readback/encode and storage waits | Cache/batch/overlap safely; preserve actual output/cadence and realistic cache state. |
| Code/shader/graph startup | Compile, load, first-use prewarm, capture/replay and fallback | Repair reuse or startup safely; measure cold or warm conditions according to the objective. |
| Simulator orchestration, when present | Integrations, prefixes, full forwards, collision/constraint and solver counts | Trace decimation and state/sensor consumers; preserve reset/wrap/push and necessary dynamics. |
| CUDA/device work, when present | Kernel ownership, conversion/allocation chains, necessary hot-kernel counters | Fuse/reuse or improve memory/instruction behavior at required precision; a native kernel can still be an unnecessary call. |
| Policy/learning, when present | Inference/update/normalization/shuffle and metric/checkpoint phases | Optimize implementation at unchanged math/work; learning-parameter search is a separate task. |

For a simulator, inspect helper bodies and eligibility/fallbacks: a kinematics or
prefix name does not prove narrow work. Compare expected/observed calls in steady
steps and lifecycle boundaries, then connect CPU traces to the actual device path
when applicable. Separate common orchestration from optional specialization when
attributing gaps. Apply the same ownership discipline to a renderer's draw helper,
a CPU wrapper or an I/O pipeline.

## Axis registry

Each entry needs id/owner, observed metric contribution or bound, raw evidence,
hypothesis, expected effect, next experiment and status:

- `open`: supported opportunity or unresolved measurement/ownership question;
- `testing`: one hypothesis is being evaluated;
- `improved`: accepted change; residual potential still needs reassessment;
- `closed`: no useful in-scope action remains under current evidence;
- `blocked`: useful work needs unavailable necessary evidence/resources/authority;
- `out_of_scope`: the recorded user/parent contract explicitly excludes it.

Keep attempts/outcomes under the axis. Failed implementations do not reject all
other solutions. Inconclusive/noisy trials give no negative effect bound. An
efficient kernel does not close redundant call scheduling. Code complexity is a
real cost: name concrete duplication, fragility or maintenance risk when declining
a metric improvement, rather than using an unexplained complexity label.

## Closure and reopening

Close an axis only with a final-path reason, such as a metric-specific residual
bound below declared useful resolution, controlled measurements/source analysis
ruling out the remaining ideas, necessary work with no supported safe improvement,
or concrete correctness/quality/maintainability constraints rejecting supported
in-scope changes. A GPU time share cannot close a CPU memory-retention axis.

Explicit exclusions stay `out_of_scope`. Promising untestable hypotheses stay
`blocked` and prevent exhaustion of the requested scope. Missing Nsight is not a
blocker for a CPU task or graphics path with adequate alternative evidence.

After adoption revisit every closure assumption. Reopen axes affected by a moved
bottleneck, critical path, memory/lifetime, pass/call count, layout, input mix,
startup or validation. Add newly discovered axes. Final scoped exhaustion needs
no open/testing/improved/blocked in-scope entries and fresh final-path review;
never delete an inconvenient axis to make the registry appear complete.
