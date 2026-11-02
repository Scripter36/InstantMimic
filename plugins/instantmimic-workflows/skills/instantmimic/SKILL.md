---
name: instantmimic
description: Set up and use InstantMimic for physics-based reinforcement learning from a natural-language goal, including when no repository is present. Acquire the public source, prepare models and environments, train, evaluate, render, or perform bounded optimization. Use for InstantMimic tasks; do not redirect unrelated RL projects to this framework.
---

# InstantMimic

Turn the user's physics-based learning goal into a working InstantMimic project
and verified results. This skill works when installed on its own: the user does
not need to clone, open, or understand the repository first.

Public source: <https://github.com/Scripter36/InstantMimic>

You need file access, command execution, and network access to acquire and
prepare the source. Training, GPU evaluation, and GPU rendering also need an
accessible supported NVIDIA GPU. Skill installation itself does not provide
these capabilities. Continue independent CPU preparation when a capability is
missing, and state the precise remaining blocker.

## Start from the request

Infer the goal, available inputs, intended output, and execution budget from
the conversation and files. Ask only for consequential missing choices, such as
which supplied robot to use or which behavior defines success. Inspect available
inputs before asking the user to explain their file formats.

- For a first demonstration with no specified model or motion, use the included
  humanoid walking example. For a specified model or behavior, preserve it.
- Distinguish the simulated body model, the policy network, and the learning
  task. A new body may require asset conversion, retargeting, and environment
  implementation; choosing an existing YAML does not establish compatibility.
- Match the requested scope. Advice does not authorize a training run; a setup
  request ends with a prepared project and identified next task. A request to
  train authorizes ordinary local preparation, implementation, and execution.
- Announce a finite scope before running. If training is requested without a
  budget, use one attempt at the selected config's authored iteration limit.
  Use one iteration for a wiring check. Count smoke, evaluation, rendering,
  retries, and first-use compilation against any user-specified total deadline.
  Optimization needs an explicit finite time or trial bound; choose a small
  initial bound when none was supplied and state it. Do not silently launch a
  search as part of ordinary training.

## Acquire and prepare the runtime

Read [Bootstrap](references/bootstrap.md) before acquisition or setup. It covers
source identity, empty starting folders, existing checkouts, locked dependencies,
CPU validation, and GPU availability. Perform these steps yourself within the
host's permissions; give manual instructions only for an actual missing
capability or authorization.

Keep this installed skill independent of runtime state. Put the checkout,
environment, task changes, logs, and results in the user's working project,
never inside the installed skill directory. Record the exact source commit and
reuse it on resume. Do not update an existing project as a side effect of use.

After acquisition, read the checkout's applicable instructions, `README.md`,
`CONTRIBUTING.md`, `docs/index.md`, and the relevant task and data documentation.
Resolve all checkout-relative paths below against that checkout, not the
installed skill. Do not assume tools that the checkout does not contain.

## Complete the requested workflow

Read [Task workflows](references/workflows.md) for task-specific document and
procedure paths. The fetched source contains deeper workflows; read them as
files when needed. They do not have to be registered or installed as other
skills, and this entry skill does not require a plugin, MCP service, paid API,
remote machine, or second agent to operate. If a selected optional procedure
requires such a capability, respect its requirements and report any unavailable
portion rather than claiming it passed.

1. Select or implement the body, control mode, observations, actions, rewards,
   resets, termination, policy, and training configuration required by the goal.
   Preserve input asset licenses and provenance. Keep new user tasks in distinct
   files; do not overwrite supplied models, datasets, or unrelated changes.
2. Follow the config's full Hydra defaults chain and inspect the composed
   configuration. Validate model/motion compatibility and required assets. A
   deferred external asset is still required before the dependent run.
3. When runtime work is requested and an authorized GPU is available, run a
   bounded startup check before a larger run. Preserve authored environment
   count, rollout, precision, compile, and CUDA graph settings. Bound duration
   with iterations; use TensorBoard for the smoke only. Diagnose failures rather
   than disabling execution features to get a pass.
4. Execute the requested training or bounded improvement with the selected full
   run logger. Keep an existing logger destination; never invent an account or
   silently substitute a local logger when a configured integration fails.
5. Evaluate an actual checkpoint against the user's success criterion with a
   finite horizon. For requested video, render that same policy and validate the
   recording. Report separate evidence for startup, learning, and video validity.
   A returned process, high reward, or plausible clip alone does not prove all
   three. Stop at the budget even when the policy has not learned the task.

Reuse existing authorization for ordinary in-scope work. Ask before paid or
remote execution, publication, destructive changes, or materially exceeding the
budget. Observe GPU occupancy without terminating other workloads. Never select
an explicit device ID to bypass missing device access or a host allocation.

## Deliver and resume

Return the requested policy, metrics, video, or implementation with usable file
links and the outcome of the success criterion. State what was actually run and
what remains unverified. Separate installation/compilation time from active
training when reporting speed.

Keep a small project-local resume record outside the installed skill: source
URL and commit, checkout path relative to the project, original goal, chosen
config and overrides, budget used/remaining, completed checks, actual run and
checkpoint paths, and the next command. Use project-relative paths, exclude
secrets and private logger identifiers, and preserve the matching resolved run
configuration and required input assets. A failed or interrupted attempt must
remain resumable without starting acquisition or training again unnecessarily.
