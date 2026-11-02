# Hyperparameter optimization

Because a policy trains in seconds, InstantMimic can afford to search over
training settings with an LLM agent as the planner. Each cycle, the agent reads
the results so far, proposes new settings, and `run_hpo` validates them,
trains and evaluates the candidates, and records the outcome under `logs/hpo/`.

The prompts are built by `build_planning_prompt` and `build_prepare_prompt` in
`src/instantmimic/hpo/planner.py`, and the agent CLIs are launched from
`src/instantmimic/hpo/agent_runtime.py`. Each cycle directory keeps the exact
prompts that were sent (`planning.prompt.md`, `proposal.prompt.md`) next to the
agent's replies.

## Try the included search

`hpo/backflip_pd` searches PPO settings for the humanoid backflip. Print its
resolved task, then preview one cycle without training or calling the planner:

```bash
uv run run_hpo --config-name hpo/backflip_pd --cfg job --resolve
uv run run_hpo --config-name hpo/backflip_pd runtime.dry_run=true runtime.cycles=1
```

To run the search for real, install and sign in to the planner CLI (Codex by
default; see [planners](#planners-permissions-and-costs)) and drop
`runtime.dry_run=true`:

```bash
uv run run_hpo --config-name hpo/backflip_pd runtime.cycles=10
```

To continue a search later, point `runtime.resume` at its directory.
`runtime.cycles` is the last cycle to run, so resuming a run that finished
cycle 7 with `runtime.cycles=10` runs cycles 8 to 10:

```bash
uv run run_hpo --config-name hpo/backflip_pd \
  runtime.resume='logs/hpo/TASK/RUN' runtime.cycles=10
```

A scalar search keeps the best candidate for its objective; a Pareto search
keeps the trade-off front between its objectives. The
[task schema](../reference/hpo-task-schema.md) describes how to write your own
search task, and the optional [assistant workflow](assistant-workflows.md) can
help prepare one.

## Paper search result

`humanoid/imitation/variants/backflip_pd_optimized_cycle65` is the backflip
setting the paper's search selected at cycle 65, applied on top of the search's
starting config `backflip_pd_baseline`:

| Setting | Paper (median of 3 seeds) | This release (median of 10 seeds, RTX 5090) |
| --- | ---: | ---: |
| Starting config | 18.77 s | 19.64 s (17.82–24.42 s) |
| Cycle 65 | 2.17 s | 2.13 s (2.00–2.38 s) |

Times are active training time until 90% of benchmark episodes succeed, as
evaluated by `hpo/backflip_pd`. They vary with the GPU and from seed to seed,
so compare settings on the same machine over several seeds.

## Evaluator launchers

Candidates can be trained on the current machine or through a job scheduler,
selected with `runtime.evaluator.launcher`:

| Launcher | Where candidates run |
| --- | --- |
| `local` | On the current machine |
| `slurm` | As Slurm jobs; set `slurm_host` and `slurm_sbatch_args` |
| `hq` | As HyperQueue jobs; set `hq_submit_args` |

With a scheduler, the execution hosts need their own copy of the checkout, its
dependencies and data. `runtime.evaluator.max_parallel` limits how many
evaluators run at once.

## Planners, permissions and costs

Choose the planner with `planner.platform` and `planner.model`, and optionally
`planner.command`. Each real cycle makes planner calls, which a hosted provider
may bill, and trains the candidates on your GPUs.

The planner runs as an agent CLI with these permissions:

| Platform | How it runs |
| --- | --- |
| `openai` | Codex with `-s <planner.sandbox> -a never` (no approval prompts). The default sandbox is `workspace-write`; the backflip task uses `read-only`. |
| `ollama` | `ollama launch codex` with the same arguments. Requires a model; `planner.auto_start_ollama=true` starts a configured local server. |
| `claude` | Claude Code with `--permission-mode bypassPermissions`; `planner.sandbox` does not apply. |
| `opencode` | OpenCode with `--dangerously-skip-permissions` while `planner.skip_permissions=true` (the default); set it to `false` to use OpenCode's own permission settings. Requires a model. |

Since the planner can read the checkout, run it with credentials and permissions
you are comfortable giving it.
