# Validate an optimization before adoption

Map the diff to affected contracts. Preserve causally independent evidence rather
than rerunning everything or trusting all old results. Declare tolerances and a
fixed quality protocol before observing the candidate; changing them after a
failure is a different experiment, not validation of the original one.

| Changed path | Evidence proportional to the change |
| --- | --- |
| CPU algorithm, data layout or batching | Independent output/ordering/edge-case equivalence on representative data, numeric tolerances, thread safety and resource bounds. |
| Graphics/renderer | Matched scenes/cameras/resolution/lighting and representative frames; image/numeric/visual quality checks suited to deterministic or stochastic output, feature coverage, resource lifetime and backend errors. Preserve shadow/detail/culling behavior, not merely a clean log. |
| Streams/threads/queues/synchronization | Producer/consumer ordering, resource ownership/lifetime, completion semantics, pacing/tail latency and supported fallback/concurrency paths. |
| Allocation/cache/I/O | Correct outputs/bytes, cold/warm cache semantics, peak/live/retained resources, no leak or overflow, and performance guardrails. |
| Startup/compiler/shader/graph | Relevant cold-start/first-use and steady-state paths; compilation/prewarm/capture and fallback behavior where present, using isolated caches when needed. |
| Simulation task/base/controller | Behavioral/state differentials, sensor history, partial/automatic reset, reset-to-next-step and external state edits; integration/solve counts and capacity headroom. |
| Learning implementation/numerical order | Applicable forward/gradient/update checks, checkpoint/resume, fixed policy evaluation and finite learning regression when lower-level equivalence cannot establish behavior. |
| Output/logging/checkpoint/video | Required records/cadence, valid complete outputs, decode/encoder metadata and sampled actual content when touched. |

Only applicable rows impose work. A CPU parser does not require CUDA graph or
policy evaluation; a graphics change needs visual/graphics validation rather than
an unrelated training reward. For a renderer shared with training/playback,
verify integration only where the changed boundary actually affects those users.

Keep references independent enough to catch plausible mistakes. Two paths sharing
the changed formula may agree incorrectly. For stochastic rendering or contact
simulation, declare comparison protocols and acceptable distributions/quality;
do not promise bitwise identity across independent device execution. Inspect
representative difficult cases and dynamic sequences, not one favorable image.

For simulation/learning, a fresh equivalence bridge may preserve independent past
learning evidence. Refresh downstream evidence when semantics, stochastic state,
learning math or evaluation changed. With explicit checkpoint playback, saved
config plus CLI dotlist overrides defines evaluation; an alternate config name
alone does not establish another protocol. Preserve applicable noise, reset,
pushes, sampling, horizon and selected-checkpoint quality criteria.

## Measurement acceptance

Keep raw samples and initial/incumbent/candidate fingerprints. Match inputs,
outputs, load, concurrency, cache/warmup and actual executed work. For graphics
include presentation/frame-pacing and render/readback/encode boundaries; for
simulation include representative policy/state/reset/contact mix. Observe relevant
background CPU/storage/GPU activity, thermal/clock changes and path changes.
Exclude only documented violations of the comparison conditions, and stabilize
the baseline before adoption. Legitimate slow frames, tail requests and GC pauses
are workload evidence, not outliers to discard for a better result.

Use enough repeated normal-condition trials to resolve the target metric. Report
its appropriate aggregation/distribution, not only the best sample. Do not inflate
thresholds after seeing a candidate to manufacture closure. Heavy profiling,
debug validation, diagnostic readbacks and inserted synchronization are attribution
methods, not ordinary performance results; disable them or quantify their impact.

A shorter CPU submit call does not establish a completed faster GPU frame. Use
the objective's real completion boundary without adding artificial synchronization
to the production path. A memory objective needs defined resource accounting and
lifecycle points, including reserved versus live allocation; a time-only win does
not satisfy it. Treat memory/time/quality regressions using the declared guardrails
rather than assuming every objective is maximum FPS.

Check affected representative workloads and consumers for shared-layer changes.
Re-run only causally affected checks after edits; obtain final normal-condition
measurements and relevant attribution on exactly the accepted code. Remove
benchmark-only counters, favorable fixed state or disabled checks from production.

If a candidate fails, retain evidence and restore only its owned changes. Never
reset a shared worktree or erase another contributor's edits. Use isolated source
variants when comparison or reversal would be ambiguous. Inconclusive candidates
cannot become accepted implementations through renaming.
