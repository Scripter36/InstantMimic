# Task workflows after acquisition

All paths in this reference are relative to the acquired InstantMimic checkout.
Read the relevant public guide first, then the indicated detailed procedure only
when needed. Open procedure files directly; named sibling skills do not need to
be installed. Resolve a procedure's relative links from its own directory.
If a path is absent in the chosen revision, discover its documented equivalent
or report the missing capability.

| User goal | Public guides | Detailed procedure files |
| --- | --- | --- |
| Train or resume an included task | `docs/getting-started/first-run.md`, `docs/guides/training-and-playback.md`, `docs/reference/configuration/` | `plugins/instantmimic-workflows/skills/train/SKILL.md` |
| Change a config or policy network | `docs/reference/configuration/`, `docs/development.md`; inspect the selected config and runtime consumers | `plugins/instantmimic-workflows/skills/config-variant-check/SKILL.md` |
| Evaluate or record a checkpoint | `docs/guides/training-and-playback.md`, `docs/guides/rendering.md` | `plugins/instantmimic-workflows/skills/play-best/SKILL.md` |
| Use muscle control | `docs/guides/muscle-control.md`, relevant `data/` provenance | Training and evaluation procedures above |
| Use Unitree G1 tracking | `docs/guides/beyondmimic.md` | Training and evaluation procedures above; prepare the declared external inputs first |
| Prepare AMASS or latent policies | `docs/data/amass-retargeting.md` | Follow its stage-specific config, input, and checkpoint requirements |
| Bring a new body or learning task | `docs/development.md`, `docs/reference/configuration/`, `docs/reference/simulation.md`; inspect `src/instantmimic/envs/` and the closest compatible task | For an existing external environment port: `plugins/instantmimic-workflows/skills/migrate-external-env/SKILL.md` |
| Improve learning settings within a budget | `docs/guides/hpo.md`, `docs/reference/hpo-task-schema.md` | `plugins/instantmimic-workflows/skills/tune-hyperparameters/SKILL.md` |
| Improve runtime speed or memory | `docs/development.md`, `docs/reference/simulation.md`, `docs/reference/configuration/` | `plugins/instantmimic-workflows/skills/optimize-runtime/SKILL.md`; load profiling procedures only if needed |

## New models and tasks

Inspect input formats, units, body/joint names, actuator/control semantics,
joint limits, initial state, motion correspondence, and licensing. Convert or
retarget assets when needed, preserving originals. Do not treat an arbitrary
URDF, MJCF, or motion file as directly interchangeable with an included body.

Discover the current environment registration, step/reset buffers, simulation
orchestration, observation/action layouts, reward and termination consumers, and
policy construction from the acquired source. Implement only the needed task
extension with deterministic CPU checks where practical, then run a scoped
runtime and learning validation within budget. Use the repository's maintained
simulation path and preserve its allocation, graph, and compile contracts.

For a new task without a source environment, define observable success from the
user's requested behavior and implement the necessary environment. For a port,
the migration procedure adds source equivalence and independent review; if its
required reviewer or other capabilities are unavailable, report the missing
validation and do not label the port production-ready.

## Finite runs and meaningful results

Resolve requested time budgets against the actual controls in the acquired
runtime. Use its bounded-run mechanism or a host process timeout that captures
logs, signals cleanly, and accounts for child processes. Do not assume active
training seconds include setup, evaluation, or compilation. Keep finite
iteration limits as well, and report an interrupted run honestly.

For ordinary training, use the selected config and full-run logger. Do not add
an optimization loop. For optimization, set a finite trial/time budget and
evaluation criterion before starting. Automated planners may require separate
model credentials and incur costs: inspect `docs/guides/hpo.md` before
launching them. A local bounded comparison can be done without a planner when
appropriate; do not claim the automated controller ran if it did not.

Select checkpoints from the actual run's outputs and metadata. Preserve the
matching resolved config, body, motion, and policy architecture. Evaluate for a
finite horizon matched to the requested behavior; report termination and work
actually executed alongside reward. Preserve comparisons' environment count,
rollout length, reset behavior, horizons, precision, graph, and compile settings.
Separate cold setup costs from normal-condition throughput.

If video is requested, follow the acquired rendering guide, including Linux EGL
requirements where applicable. Bound both simulation steps and video length.
Verify that the video decodes, contains the expected frames and dimensions, and
uses the evaluated policy. If rendering is blocked, deliver the policy and
evaluation with that limitation instead of inventing a video result.
