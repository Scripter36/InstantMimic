# Simulation and execution

After `DirectRLEnv` construction, every physics step and dynamics refresh runs
through the repository's allocation-free MuJoCo Warp orchestration in
`src/instantmimic/utils/mjw_step_optimized.py`. Upstream `mujoco_warp.step()`
and `mujoco_warp.forward()` allocate temporary Warp arrays, which prevents the
nested conditional CUDA graph used by benchmark evaluation. The orchestration
reuses the upstream kernels where it can; the specialized kernels in
`src/instantmimic/utils/mjw_kernels/` replace selected stages when the model is
eligible. Environment orchestration is in `envs/direct_rl_env.py` and
`envs/substep_integrators.py`.

## Supported versions

The runtime is qualified against `mujoco==3.16.0`, `mujoco-warp==3.16.0` and
`warp-lang==1.15.0`, as pinned in `pyproject.toml` and `uv.lock`. The adapters
follow the 3.16 model and data layout, including `njmax_nnz`, inertia-tile
metadata, the block-plus-sparse-tail `qLD` layout, sleep/island state, flex
collision, sensor history and RK4 activation state. To upgrade any of them,
follow the [upgrade checklist](#upgrade-checklist). Private MuJoCo Warp imports go through one boundary,
`utils/mjw_kernels/private.py`.

## Runtime contract

`DirectRLEnv` creates one `OptimizedFwdPositionScratch` after model and data
construction. It owns persistent buffers for collision, transmission, tendon
bias, constraints, sensors, solvers and integrators, including flex collision,
passive and fluid forces, sensor history, sleep and island state, Euler damping,
RK4 stages and implicit derivative workspaces. These buffers exist even when the
specialized kernels are disabled.

After construction, environment code must use:

- `_run_mjw_step()` for a complete physics step;
- `_run_mjw_forward()` for a dynamics refresh without integration.

Reset hooks, StablePD torque generation and task-specific refreshes use the same
helpers. Leaf environments must not allocate their own physics scratch or call
`mujoco_warp.step()`/`forward()` directly.

### State prefix and action suffix

Each physics substep is split into two allocation-free stages:

1. the position-and-velocity state prefix: kinematics, tendon and transmission
   geometry, collision and constraint geometry, COM velocity, passive and bias
   forces, and position/velocity sensors;
2. the action-dependent suffix: control, actuation, acceleration, constraint
   solve, acceleration sensors and integration.

After every substep, including the last one at a policy boundary, the base
environment runs the task's post-substep hook and then refreshes the next state
prefix, so any task-level `qpos`/`qvel` clamp is reflected in derived state.
Control projection, rewards and observations then read current `xpos`, `xquat`,
`cvel`, tendon geometry and the other derived buffers without a second forward.
Resets also refresh the prefix. The policy boundary does not run a second
acceleration or constraint solve: `qacc` and constraint-force buffers are
refreshed by the next action suffix.

One exception avoids building collision and constraint state twice. In an
auto-resetting step, the reset prefix rebuilds contacts and constraint rows for
every world, so the last substep's endpoint refresh may skip collision and
constraint construction. Rewards, dones, terminal observations and the last
endpoint callbacks then see the previous substep's contacts and rows. The base
environment takes this exception only when the concrete environment class sets
`boundary_reads_constraints = False` in its own body, no post-physics hook is
registered, the control pipeline's endpoint callback reads no geometry, and the
model has no sleep and no sensors. The declaration is not inherited.
`SNUImitationEnv` declares it.

With `env.use_parallelized_kernel_launch=true`, independent branches of the
prefix run on side CUDA streams. COM velocity starts once its inputs are ready
when constraint construction does not read `cvel` or `cdof_dot`, and collision
runs alongside the inertia branch on models without sleep or a contact-filter
callback. Sensor callbacks and sleep keep sequential ordering.

### Integrators

`env.sim.integrator` selects `implicitfast`, `euler` or `rk4`; each runs one
step of the compiled MuJoCo integrator per substep through the allocation-free
path. Euler uses a preallocated adapter when damping needs temporary matrices.
RK4 keeps persistent stage buffers and runs the repository forward pipeline for
every stage; intermediate callbacks see stage times $t_0$, $t_0 + h/2$,
$t_0 + h/2$ and $t_0 + h$, and the clock is restored before the accepted
advance. Implicit-fast keeps upstream-shaped allocation-free adapters.

## Configuration

The three execution switches are independent:

| Option | Meaning | Allocation-free orchestration |
| --- | --- | --- |
| `env.use_cuda_graph` | Capture and replay the environment step | Always used |
| `env.use_parallelized_kernel_launch` | Run independent MuJoCo Warp branches on side CUDA streams | Always used, with or without side streams |
| `env.use_optimized_kernel` | Enable the eligible specialized kernels below | Always used, with or without specialized kernels |

When `env.use_cuda_graph` is off, the repo-owned MuJoCo Warp step suffix is
captured as its own CUDA graph and replayed; the graph is cached per step
variant. The suffix runs directly while an outer graph is already recording the
torch or Warp stream, because CUDA forbids a nested capture on a capturing
stream, and on non-CUDA devices, where a capture could not pair the Warp and
torch work. It also runs directly when the configured model/integrator fails
the allocation-free capture checks (constrained RK4 with graph conditionals,
sleep/SolverCompact combinations without scratch), logging that fallback once.
Playback, benchmark evaluation, clip-failure analysis and training rollouts all
own such outer graphs and keep working through the same rule. The retired
`env.compile_mjw_step` key is rejected rather than ignored; delete it from a
configuration that still sets it.
`env.sim.warn_overflow` controls per-world GPU diagnostic printing and defaults
to `false`; solver overflow bits are still recorded.

### Specialized kernels

Each stage checks its own structural eligibility; enabling
`env.use_optimized_kernel` does not force every model through every kernel, and
no model-name list is used. `MJW_KERNEL_PATH` and `MJW_SPARSE_HESSIAN` startup
records report the actual per-model selection.

| Stage | Eligible structure | Otherwise |
| --- | --- | --- |
| Bottom-up COM, composite inertia and bias-force tree reductions | Model body-tree topology | Depth-by-depth reduction |
| Tree L'DL factor+solve for solves that keep no factor | Plans whose critical path is short, or short relative to the dense cost | Dense tile factorization |
| Indexed dense inertia factor+solve | At least 32 DOFs in one dense inertia tile with no sparse tail | Compatible factor path |
| Persistent incremental Newton | Dense pyramidal Newton with 1–79 DOFs (`nv_pad` ≤ 80), non-`DISCRETE` integrator, noncompact context, positive iteration budget | Persistent-scratch upstream solver |
| Sparse Newton Hessian factor | Static coupling pattern with fill at most 0.6 of the dense triangle and a shared factor within 46 KB (see below) | Dense blocked Hessian factor |
| Actuator-centric implicit derivatives | Supported transmission and derivative layout | Preallocated compatible derivative path |

Models that can never emit a constraint row (no equality, limits, friction loss,
flex, candidate contact pair, collision sensor or custom row) skip collision,
`make_constraint` and the solver, which sets `qacc = qacc_smooth` as MuJoCo
does for `nefc == 0`. On the fused Newton path, worlds without rows in a given
step take the same result before Hessian assembly.

The Newton loop keeps the upstream setup, convergence and overflow tracking and
final force recovery. Parallel reductions can still round differently in
float32, so numerical equivalence with upstream is tolerance-based.

#### Dense Jacobian resolution for the fused Newton path

MuJoCo Warp resolves an `auto` Jacobian to sparse above 32 DOFs
(`mujoco_warp._src.io.is_sparse`), so `put_model`'s raised dense cap alone leaves
mid-size models such as the 35-DOF G1 on the sparse path. `DirectRLEnv` closes
that gap by setting the compiled model's `opt.jacobian` to dense before upload
(`instantmimic.utils.mjw_model.promote_dense_jacobian_for_fused_newton`) and
logging one `JACOBIAN_DENSE_AUTO_RESOLVED` INFO line. The rule applies only when
all of these hold:

- `env.use_optimized_kernel` resolved on for this device;
- `env.sim.is_sparse` is unset, so the config authored no Jacobian choice;
- the compiled model still has `mjJAC_AUTO`, and MuJoCo Warp would resolve it to
  sparse;
- the fused Newton path's model-side gates hold: `nv <= DENSE_NV_MAX` (pads to
  `nv_pad <= 80`), Newton solver, pyramidal cone, non-`DISCRETE` integrator, a
  positive iteration budget, and no sleeping model. Sleep selects the compact
  solver context, which the fused iteration rejects.

This is a structural rule; no model or task is named. An authored
`env.sim.is_sparse` (and an explicit XML `jacobian`) always wins, and with
`env.use_optimized_kernel` off nothing changes: upstream's sparse resolution
above 32 DOFs stands. `tests/test_mjw_dense_jacobian_resolution.py` holds the
resolution matrix.

#### Newton occupancy

The persistent Newton kernel launches one CTA per world, so resident CTAs per SM
are limited by registers. It requests a CUDA `__launch_bounds__` minimum of

$$
\mathtt{min\_blocks} = \min\left(12, \max\left(1,
  \left\lfloor\frac{\mathtt{nworld}}{\mathtt{device.sm\_count}}\right\rfloor
\right)\right)
$$

CTAs per SM (`newton_fusion.newton_min_blocks_per_sm()`). The clamp of 12 is a
measured constant: 16 is within half a percent and 24 spills registers.

#### Sparse Newton Hessian factor

`mjw_kernels/sparse_hessian.py` factors $H = M + J^\top D J$ over a static
pattern: the union of $M$'s tree ancestry, the filtered collision pairs,
equality constraints and any body groups declared by a custom contact hook.
The symbolic factorization and index tables are generated per model. The
numeric $L^\top D L$ runs over elimination-tree levels inside one thread block
per world with a fixed summation order; the backward solve pushes each solved
unknown to its descendants level by level. An active row whose support falls
outside the pattern is dropped from the factor and counted in
`sparse_hessian.violation_counter`; `DirectRLEnv` checks the counter at the
training metric cadence and raises an error. Models outside the fused Newton
(sparse Jacobians or more than 79 DOFs) use the upstream iteration with the
factor replaced, from `NONFUSED_MIN_NV` DOFs up. `MIN_NV`, `MAX_DENSITY` and
`NONFUSED_MIN_NV` are measured thresholds.

### Model upload

`utils/mjw_model.put_model` is `mujoco_warp.put_model` with the dense-Jacobian
cap raised from 60 to 79 DOFs (`nv_pad` 80) and a wider dense Hessian block at
`nv_pad` 80. It edits the single `nv_max = 60` assignment of the pinned upstream
source and fails if that line is missing; every other model goes to the upstream
function. Environments and the playback renderer upload through it. The module
also owns `promote_dense_jacobian_for_fused_newton`, which resolves an `auto`
Jacobian to dense before upload when the fused Newton path applies (see Dense
Jacobian resolution above).

## Differences from stock MuJoCo Warp

- The implicit velocity derivative `M - h*qDeriv` gates its actuator term on
  ACTUATION and its damping term on DAMPER, as MuJoCo does. Stock 3.16 drops
  both when either flag is set, which would drop joint damping when MASS pose
  control disables native actuation.
- Flex contact deduplication breaks exact depth ties by element, vertex, flex
  and geom ids instead of by atomic slot order, so the surviving contact is
  deterministic. Row order in `d.contact` still follows the atomic counter.
- Benchmark evaluation captures a fixed number of native solver iterations with
  per-world convergence guards inside its outer conditional graph, because a
  conditional child graph cannot be nested there. Training and playback capture
  keep the conditional solver.

Results therefore match upstream within floating-point tolerance rather than
bitwise.

## Extension points

Tasks and controllers may register construction-time additions before the first
step or capture; registration is then frozen so graph topology stays fixed:

- `_register_mjw_implicit_qh_contributor()` adds a named implicit-fast
  effective-inertia term. MASS force-velocity damping uses it when
  `muscle.force_velocity.implicit=true`.
- `_install_mjw_passive_force_hook()` adds generalized forces after the native
  passive stage.
- `_install_mjw_contact_hook()` runs after the native narrowphase; declaring its
  `body_groups` keeps the sparse Hessian factor eligible.
- `_install_mjw_pre_advance_modifier()` replaces the acceleration used for the
  velocity update, and optionally the velocity used for position drift, just
  before integration. Only one modifier may be installed.

Each callback must be allocation-free.

## Muscle control

See [muscle simulation and control](../guides/muscle-control.md#simulation-details)
for muscle force application, MASS coordination and controller buffers.

## Upgrade checklist

For every MuJoCo Warp upgrade:

1. Diff every upstream function wrapped by the local allocation adapters and the
   exports in `utils/mjw_kernels/private.py`.
2. Re-audit reachable forward, solver, sensor, collision and integration paths
   for temporary Warp arrays, and update adapter expectations and scratch shapes
   together.
3. Compare stock and repository results with side streams on and off and with
   specialized kernels on and off.
4. Capture and replay the full environment step, and confirm benchmark
   evaluation captures its nested conditional graph without an eager fallback.
5. Re-audit `mjw_model.put_model` and confirm the dense Newton path still
   matches CPU MuJoCo above 60 DOFs (`tests/test_mjw_model.py`). Also re-check
   `mujoco_warp._src.io.is_sparse`: its 32-DOF auto threshold decides when the
   dense Jacobian rule fires.

## Measuring changes

When measuring a runtime change, keep the model, motion, integrator,
environment count, rollout length, compilation/capture settings and execution
switches fixed, and separate cold cache/setup cost from active training.
Focused checks:

```bash
uv run pytest -q tests/test_mjw_data_contracts.py tests/test_blocked_dense_factor_solve.py \
  tests/test_mjw_newton_fusion.py tests/test_mjw_tree_fusion.py \
  tests/test_mjw_dense_jacobian_resolution.py \
  tests/test_direct_rl_env_state_prefix.py tests/test_mjw_step_parallel_kinematics.py
```

Run the `cuda`-marked tests when a GPU is available, one process at a time.
