# InstantMimic

InstantMimic is an AI-native framework that learns physics-based character
skills on the GPU in seconds. Its training loop keeps simulation, rollout
collection and PPO updates on the GPU, which makes motion imitation fast enough
for a coding agent to run the experiments itself. The repository ships the
agent skills for that work: setting up, training, evaluating, profiling, tuning
hyperparameters and porting environments from other frameworks, as was done for
[BeyondMimic](guides/beyondmimic.md#how-this-port-was-made). See
[assistant workflows](guides/assistant-workflows.md) to install them.

## Get started

1. [Install](getting-started/installation.md) the repository and its locked
   dependencies.
2. Train your [first policy](getting-started/first-run.md), a humanoid walk,
   and replay it.

## Guides

- [Training and playback](guides/training-and-playback.md): run directories,
  evaluation, checkpoints and resuming.
- [Writing an environment](guides/writing-environments.md): the methods a task
  implements, the order of a step and the state each method can rely on.
- [Muscle simulation and control](guides/muscle-control.md): MUSUBI, its MASS
  and direct-control baselines, and the SNU and FreeMusco models.
- [AMASS motions and latent policies](data/amass-retargeting.md): preparing
  AMASS, training a latent policy and point-goal navigation.
- [BeyondMimic](guides/beyondmimic.md): Unitree G1 motion tracking.
- [Hyperparameter optimization](guides/hpo.md): LLM-planned searches over
  training settings.
- [Video rendering](guides/rendering.md): headless capture and camera tracking.

## Reference

- [Motion demonstrations](reference/motion-demo.md): configs and timings behind
  the project-page videos.
- [Configuration](reference/configuration/index.md): every config section,
  from the environment and actions to PPO, training and playback.
- [Simulation](reference/simulation.md): the MuJoCo Warp runtime and its kernels.
- [HPO task schema](reference/hpo-task-schema.md): search task files.

The included models and motions keep their own licenses, listed in
`data/README.md` and each data family's `PROVENANCE.md`.
