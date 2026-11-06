# Assistant workflows

The repository ships an optional `instantmimic-workflows` plugin for Codex and
Claude Code. Its skills guide an agent through training, evaluation, profiling,
optimization and porting environments.

## Start without a checkout

If you would rather let an agent set everything up, install only the
`instantmimic` skill:

```text
Repository: https://github.com/Scripter36/InstantMimic
Skill path: plugins/instantmimic-workflows/skills/instantmimic
```

Use your agent's skill installer. In Codex, for example:

```text
$skill-installer install the instantmimic skill from
https://github.com/Scripter36/InstantMimic/tree/main/plugins/instantmimic-workflows/skills/instantmimic
```

Other agents can load the same directory; keep its `references/` and `agents/`
folders next to `SKILL.md`.

Then describe what you want from your own project directory, for example:

> Use InstantMimic to train the included humanoid walking task and record an
> evaluation video. Prepare the source and dependencies yourself. Use the local
> GPU and keep the entire attempt within 10 minutes.

The agent clones InstantMimic into your project, records the commit it used,
installs the locked dependencies and carries out the task, reading the other
workflow files from the clone as it needs them. It needs shell and network
access and, for training, an NVIDIA GPU; if something is missing, such as a GPU
or licensed data, it tells you.

## Included skills

| Skill | Purpose |
| --- | --- |
| `instantmimic` | Start from a goal without a checkout: set up InstantMimic, then train, evaluate, render or optimize. |
| `train` | Build training commands for a task. |
| `smoke-run` | Check that a task starts and trains for a few iterations. |
| `play-best` | Pick a checkpoint, evaluate it and record a video. |
| `config-variant-check` | Add or change Hydra configs and check that they compose. |
| `experiment-report` | Summarize results and compare runs fairly. |
| `nsys-profile` | Profile a whole training or playback run with Nsight Systems. |
| `ncu-profile` | Inspect one CUDA kernel with Nsight Compute. |
| `optimize-runtime` | Speed up a workload by repeatedly measuring, changing and validating it. |
| `tune-hyperparameters` | Prepare and run an HPO search over training settings. |
| `migrate-external-env` | Port an environment from another framework, such as Isaac Lab or mjlab. |
| `review-external-env-migration` | Review a finished port independently. |

## Install for Codex

From the checkout root:

```bash
codex plugin marketplace add "$PWD/plugins"
codex plugin add instantmimic-workflows@instantmimic-local
```

Start a new session afterwards so the skills are loaded; `codex plugin list`
shows what is installed.

## Install for Claude Code

From the checkout root:

```bash
claude plugin marketplace add --scope local "$PWD/plugins"
claude plugin install --scope local instantmimic-workflows@instantmimic-local
```

`claude plugin list` shows what is installed. Claude Code caches plugins, so
after pulling updates, refresh it with:

```bash
claude plugin update instantmimic-workflows@instantmimic-local --scope local
```

Start a new session to use the updated skills.

## Use a workflow

In Codex, invoke a skill as `$optimize-runtime`; in Claude Code, as
`/instantmimic-workflows:optimize-runtime`. Describe the task, the workload,
any constraints and the result you want. For a single profile, `nsys-profile`
or `ncu-profile` can be used on their own. HPO searches call an external
planner; see [planners, permissions and costs](hpo.md#planners-permissions-and-costs).

## Port an environment from another framework

`migrate-external-env` ports a task from Isaac Lab, mjlab or another simulation
and RL framework. Point it at the source repository, name the task, and state
the budget. For example:

> Port the tracking task from mjlab at commit 15ebce8 into InstantMimic. Match
> its rewards, observations, terminations and randomization, then optimize it
> for training throughput. Use the local GPU.

The agent works through four stages and records each one in a workspace
directory, so a long port can be resumed:

1. **Contract.** It reads the source and freezes a list of gates: the formulas,
   orderings, scales and cadences that must match exactly, the values that may
   differ within a stated tolerance, the learning results that must hold, and
   anything it will leave out. You approve this contract once. The target is
   the source's own semantics, including its learning setup: policy
   distribution, observation normalization, initialization, optimizer and
   learning-rate schedule. Where InstantMimic lacks an equivalent, the agent
   adds it as an option rather than using the nearest native setting.
   InstantMimic's own learning settings come afterwards, as separate opt-in
   configs measured against the parity config.
2. **Port.** It records every source behavior it finds in a ledger, writes
   tests that pin each one to the source (and checks that they fail on
   plausible ordering, scale, timing and reset mistakes), then implements the
   task against them.
3. **Production.** It moves the task onto the fused GPU path through
   `optimize-runtime`, keeping the parity checks passing after each change.
4. **Review.** It packs the raw evidence and hands it to
   `review-external-env-migration` in a fresh context, with no summary of what
   it believes. The reviewer counts a check as passed only if a broken port
   would have failed it: the check ran on the production path, a known-bad
   variant fails it, and its comparator is the strongest one available. The
   port is reported production-ready only when that review accepts every gate.

The skill was written from the [BeyondMimic G1 port](beyondmimic.md#how-this-port-was-made)
and revised with what that port's reviews caught.

## Where results go

Training and HPO runs write to `logs/` as usual. For analyses that produce
several files, the skills use a `workspaces/` directory, which Git ignores. The
plugin's own files are under `plugins/instantmimic-workflows/`.
