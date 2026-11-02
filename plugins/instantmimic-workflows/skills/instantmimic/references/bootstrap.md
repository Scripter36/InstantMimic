# Bootstrap from the installed skill

## Choose a project and source

Work in the user's selected project directory. If none is specified, use the
current working directory and choose an unoccupied child named `InstantMimic`
for the runtime checkout. If the host starts inside the installed skill, move
to a separate writable working project first. Do not clone over an existing
directory or put generated state into a skill/plugin installation cache.

Use the public HTTPS source by default:

```text
https://github.com/Scripter36/InstantMimic.git
```

A user-selected local checkout, mirror, or revision takes precedence. Verify its
identity and capabilities; record a source override without presenting it as a
successful fetch from the public URL. Never substitute another checkout
automatically.

For inspection-only or explicit no-install requests, do not install tools or
dependencies. Source acquisition and static inspection can proceed with the
tools already available; the CPU-validation section below explains the limit.

Check `git --version` and `uv --version`. Follow the host's supported user-local
installation method if a missing tool can be installed within current authority.
Do not require administrator access or change drivers/system CUDA as a routine
setup step. `uv` can provide the Python version required by `pyproject.toml`.

For a new checkout, set the destination to the actual unoccupied path and run:

```bash
im_checkout='./InstantMimic'
git clone -- https://github.com/Scripter36/InstantMimic.git "$im_checkout"
git -C "$im_checkout" rev-parse HEAD
```

If the user supplied a release, tag, or commit, resolve and check out that
revision in this new clone before dependency installation. Otherwise use the
remote's default branch tip at acquisition time, record its full commit, and
keep that identity for the project. Do not hardcode a local commit
or assume a branch named `master` or `main`. A later resume uses the recorded
checkout and commit; no automatic pull or dependency upgrade is needed.

For an existing checkout, inspect its origin, commit, status, and project resume
record before reuse. Preserve dirty work. Do not reset, switch revisions, clean,
or replace its lock file automatically. If it conflicts with a requested version,
use a separate unoccupied checkout. Do not expose credentials embedded in remote
URLs or local configuration in shared output.

## Check that runnable source was delivered

Before installing anything, verify that the selected revision includes at least
`pyproject.toml`, `uv.lock`, `src/instantmimic/`, `config/`, `data/`, and
`docs/getting-started/installation.md`. Read the actual source metadata and
installation guide. If any of these is missing, report the revision as
incomplete and stop dependent setup. An unavailable URL is not a reason to request GitHub
credentials for an intended public download or to fetch a different project.

If a future layout changes, discover the documented replacements from that
revision before continuing. Use only commands documented in the acquired
revision.

The source checkout is the runtime distribution. Keep its `config/`, `data/`,
and package directories intact; a wheel-only install does not supply them.
Run InstantMimic commands from this checkout's root. Keep user task additions in
namespaced config/data/source files within this project-owned checkout.

## Prepare dependencies and validate on CPU

CPU-only execution can still download and install dependencies: plain `uv run`
may synchronize the environment automatically. For an explicit no-install task,
skip `uv sync` and plain `uv run`. If this checkout already has a verified
compatible environment, use `uv run --no-sync` for the commands below. Otherwise
inspect source, configuration defaults, and declared asset paths statically;
report composition/runtime checks as not run. Do not borrow an unrelated
environment or describe static inspection as successful Hydra composition.

Read `docs/getting-started/installation.md` and execute its locked setup from
the checkout root when installation is in scope. In the public source this is:

```bash
uv sync --locked
```

Honor the actual platform and Python constraints of the acquired revision. Do
not independently upgrade MuJoCo, MuJoCo Warp, Warp, or Torch to resolve a setup
failure: inspect the lock and platform support first. Keep reusable tool/compiler
caches outside the checkout. Do not install development or documentation groups
unless the task needs them.

Compose the chosen config without constructing a GPU environment. For the
included walking demonstration:

```bash
uv run train --config-name humanoid/imitation/walk_pd --cfg job --resolve
```

Inspect the result for task, assets, observations/actions, policy, environment
count, horizons, graph/compile mode, logger, and training limit. Keep credentials
and private destinations out of saved or shared output.

For a checkout-wide CPU asset/config audit when relevant:

```bash
uv run python -m instantmimic.tools.validate_configs --allow-unprepared-assets --json
```

The flag permits declared preparation requirements; it does not satisfy them.
Verify the selected task's actual required inputs before execution. Read
`data/README.md`, the family's provenance files, and the task-specific preparation
guide. Obtain user-supplied or licensed inputs through their authorized source;
do not replace a requested task silently with an included example.

## Verify the execution device

Only probe CUDA when GPU execution is in scope and the host permits it. Observe
physical occupancy with the host's available NVIDIA tooling and verify that the
same execution path and locked Python environment expose a usable CUDA device.
Resource discovery, a driver report, and an allocated device are different facts.
Follow the host's allocation policy and do not terminate or displace other work.

If the GPU is occupied, hidden, unsupported, or the user requested code-only
work, finish relevant CPU preparation and report what blocks runtime validation.
Do not route around that limitation through SSH, a cloud provider, a scheduler,
or explicit GPU IDs. Remote/paid execution requires the user's direction.

For a requested startup check on an accessible GPU, follow
`docs/getting-started/first-run.md`. The documented example is:

```bash
uv run train --config-name humanoid/imitation/walk_pd \
  experiment.logger.type=tensorboard trainer.max_iterations=1
```

Use the chosen task instead for user-specific work. Preserve authored execution
settings. Record actual iteration completion and the reported output paths.
Compilation can dominate this check; do not describe one iteration as successful
learning or a measured training speed.
