# Included models and motion data

This distribution includes the models and motions its configurations use.
Configuration paths are relative to the repository root.

| Family | Included models | Included motion |
| --- | --- | --- |
| [Humanoid](humanoid/PROVENANCE.md) | Force-limited PD humanoid | Five MimicKit clips: walk, run, backflip, cartwheel and spinkick |
| [SNU humanoid](snu_humanoid/PROVENANCE.md) | Base and StablePD models, hand/full-body variants and required includes/meshes | This group's `walk.bvh`, `run.bvh` and `backflip.bvh` |
| [FreeMusco](freemusco/PROVENANCE.md) | Ostrich and Chimanoid, including 2×/4× muscle-count variants | Tracked reference arrays and local loop packs |
| [Ant](ant/PROVENANCE.md) | Quadruped from Newton's `nv_ant.xml`, as used by Isaac Lab's Ant task | None; the task rewards forward progress |

The corresponding provenance files record the source and license of each
included file. MimicKit's license travels with its clips.

`render/command_goal.obj` is the point-goal marker used by the downstream goal
tasks. It is project-authored under Apache-2.0 and ships as geometry only; the
task configurations colour it with `point_goal_marker_rgba`.

AMASS source data and AMASS-derived motion datasets are not bundled. Runtime
logs, trained checkpoints and generated caches are also not part of the data
bundle. Training creates its output under `logs/`.

## Locally prepared G1 tracking inputs

BeyondMimic uses separately prepared model and motion files described in
[Unitree G1 provenance](unitree_g1/PROVENANCE.md). They are not included in this
data bundle. Their exact paths and preparation instructions are declared in
`runtime-assets.json`; `validate_configs --allow-unprepared-assets` checks
their configurations before preparation.
