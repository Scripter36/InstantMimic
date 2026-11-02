---
name: config-variant-check
description: Use when adding, editing, or reviewing Hydra config variants in this repository.
---

# Config Variant Check

Config variants should compose and smoke, not hard-code brittle assertions.

## Rules

- Do not point configs at absolute personal paths or sibling repos.
- Match config family to data family, e.g. `config/snu_humanoid` uses `data/snu_humanoid`.
- Prefer updating stale paths to canonical repo-relative paths.

## Validation Preference

1. Hydra composition check for the changed config.
2. Short tensorboard smoke if runtime wiring matters.
3. Focused tests only when protecting public behavior or policy.

Read applicable repository instructions when present. Repository `config/`
files are not test inputs or fixtures: compose changed real configs as a checkout
check, and create minimal temporary configs for behavior tests. Trace the runtime
consumer as well as the composed leaf value before claiming a setting is active.
