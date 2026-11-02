# Phase 2: Bootstrap Execution and Baseline

Asserts A4, A6. Exit artifact: a verified execution path plus a launched, resumable full search.

## Resolve the Backend

Represent the target with two independent fields:

```yaml
transport: local | ssh
executor: direct | hq | slurm
```

These are planning fields, not extra task-schema keys. `direct` maps to the
`local` evaluator launcher on the execution host; `hq` and `slurm` map to their
same-named launchers. SSH is an optional control transport, not a GPU allocator.
Supported routes include local/SSH direct and scheduler execution where the
configured launcher supports it. `runtime.evaluator.launcher` selects the
launcher (`src/instantmimic/hpo/evaluator_launch.py`); `objective.mode` only
selects how results are scored and ranked.

With remote Slurm, SSH only bootstraps the workspace, submits and monitors jobs
on the login node, and retrieves compact artifacts; Slurm owns GPU allocation.
Use direct SSH on an accessible workstation after checking physical GPU occupancy.
Telemetry does not reserve the device; the caller owns local scheduling.

`hq` (HyperQueue) and `slurm` take `runtime.evaluator.max_parallel`,
`runtime.evaluator.slurm_host`, `runtime.evaluator.slurm_sbatch_args`, and
`runtime.evaluator.hq_submit_args`. Both submit one job per evaluator phase and
wait, so the planning loop never holds an allocation. Keep the outer loop outside
GPU allocations; only the evaluator phase uses the selected direct or scheduler
execution path.

Run non-interactive connectivity, scheduler, quota, GPU, storage, runtime, and
artifact-path probes. Never infer access from a hostname alone. If a scheduler is
present, do not bypass it with a direct GPU process.

Inspect the selected launcher instead of assuming it implements this contract.
If it only transfers a job script or assumes an identical absolute checkout,
the agent must bootstrap the checkout and installation itself, then render
remote commands and task paths against the verified remote root. Do not add
remote clone, checkout, install, or path discovery to the training runtime.

Choose a control topology that the available agent can actually operate and the
runtime accepts. The agent
performs the required `git`, environment installation, path mapping, upload,
submission, monitoring, and compact artifact retrieval steps directly. Persist
the resolved commands and remote paths so resume does not depend on chat memory.

## Bootstrap Once Per Fingerprint

Create an immutable execution fingerprint from at least:

```text
source snapshot + dependency lock + Python/runtime + simulator/CUDA stack
+ GPU model + evaluator contract
```

For a clean tree, fetch and check out the exact commit. For uncommitted work,
create a content-addressed snapshot such as an ephemeral Git ref or bundle; never
label the moving working tree as a revision. Detect and reuse a matching remote
checkout and installed environment. Otherwise clone/fetch, install from the
frozen lock, and record the resulting fingerprint.

Keep datasets, assets, caches, checkpoints, and raw logs on remote persistent or
shared storage. Send source deltas through Git and send only manifests/configs
per candidate. Retrieve metrics, summaries, and selected artifacts instead of
rsyncing complete output trees.

On Slurm, perform any required GPU-dependent startup validation in an allocation
rather than on the login node. Use isolated concurrent compile caches unless the
runtime proves safe sharing. Persist reusable dependency and source caches;
clean only run-scoped scratch state according to the approved retention policy.

Implement missing fixed-evaluator or Pareto aggregation before baseline GPU
work. Compose the task, verify every inline `parameter_space` path, and protect
exact allowlist rejection, metric extraction, matched-point aggregation, epsilon
dominance, archive resume, and representative selection with deterministic CPU
tests. Parameter discovery remains an agent procedure; tooling changes are
infrastructure, never additional search axes.

## Calibrate the Baseline

Reuse migration runs only when revision, resolved config, runtime path, hardware,
work, and evaluator protocol match. Otherwise run a fresh baseline with the
contract's configured seed set.

Measure startup/compile, active rollout/update time, evaluation time, and
end-to-end wall time separately. Derive a finite active-time horizon from the
observed learning curve and requested per-trial budget. Choose a time-scaled evaluation
interval that normally yields 12–20 useful points while keeping evaluation
overhead small; freeze it after calibration.

Verify authority, resource access, supervision, and resumable state, then launch
the complete bounded search. Run a required baseline as its first candidate; do
not split it into a separate invocation. Standalone dry runs and GPU smoke runs
are optional and should be used only for a new execution path whose failure
cannot be diagnosed safely from the bounded, resumable search itself. Record
unavailable credentials or infrastructure as a pre-execution blocker rather
than repeatedly prompting later.
