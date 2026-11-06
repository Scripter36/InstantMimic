# Experiment Protocol

Follow applicable repository instructions. A valid benchmark establishes actual
baseline identity, matched resolved config and runtime markers, executed work,
and regression coverage for shared-code changes. Disclose invalid comparisons
and retract derived conclusions. A disabled optimization flag is not proof that
the rest of two execution paths match.

- A reportable seeded comparison defaults to exactly 3 seeds unless the user or
  frozen task contract specifies another count. Exploratory tuning and
  HPO use their explicit configured seed count; do not encode this
  reporting default as a runtime/schema restriction.
- One setting's reportable seeds use the same backend-independent logical group:
  `experiment.group`.
- Full training preserves normal logger selection from config, dotenv and explicit
  overrides. TensorBoard, W&B and MLflow are valid when supported by the config;
  do not replace the selected backend for convenience.
- Smoke runs use `experiment.logger.type=tensorboard`, never W&B or MLflow.
- Treat the MLflow tracking destination as user-owned configuration. Do not inject,
  replace, unset, or reinterpret `MLFLOW_TRACKING_URI` unless the user explicitly
  requests a destination change.
- A request to run a full experiment authorizes normal writes to its existing
  configured logger; do not add a separate destination approval gate.
- Resolve configuration after normal dotenv loading, but treat endpoint, hostname,
  project, and path values as secret-adjacent. Never print or persist them. Report
  only backend plus `configured`, `implicit_local`, `conflict`, or `unavailable`.
- Surface source conflicts without revealing values. Never inject local MLflow as
  fallback for an unavailable configured integration.
- Training uses `job_type=train`: W&B passes it through `wandb.init`, while MLflow
  records it as the `job_type` run tag. Both receive `experiment.group` through
  their native grouping field or tag.
- Separate evaluation runs use the same `experiment.group` as their training
  runs with a distinct `job_type`, such as `eval`.
- Keep run names unique if overriding names; group is shared.
- Do not disable CUDA graph or compile flags as a workaround:
  - `env.use_cuda_graph`
  - `runtime.act_cuda_graph`
  - `runtime.update_cuda_graph`
  - `runtime.compile`
- Post-training evaluation uses `play` from the best checkpoint.
- Per seed, record checkpoint path, reward, episode length, and video path if produced.
- Record the selected logger plus its run identity and logical group.
- For 3-seed summary, report mean reward across best-checkpoint playback runs.
