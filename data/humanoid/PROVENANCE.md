# Humanoid model and motion provenance

## Model

`mjcf/humanoid_pd.xml` and the hand-contact variant
`mjcf/humanoid_pd_hands.xml` are authored in this repository.
The default uses foot contact; the hand-contact variant also enables hand contact.
Their topology follows the long-standing MuJoCo / DeepMind Control humanoid
convention; the joint and force-limited PD actuation are repository variants.
These models are distributed under Apache-2.0, as recorded by the repository
[LICENSE](../../LICENSE). They have no external mesh or MJCF include dependencies.

## Motion clips

The following clips come from the official MimicKit data bundle linked by
https://github.com/xbpeng/MimicKit:

- `motions/mimickit/humanoid_walk_sanitized.pkl`
- `motions/mimickit/humanoid_run.pkl`
- `motions/mimickit/humanoid_backflip.pkl`
- `motions/mimickit/humanoid_cartwheel.pkl`
- `motions/mimickit/humanoid_spinkick.pkl`

The recorded source bundle has SHA-256
`a8b98e69cb63d4d6cf41dbd64024a0a6e2367398efbae6bac320d884eee93cba`.
MimicKit and that bundle are provided under Apache-2.0. The full source license
is preserved at [motions/mimickit/LICENSE](motions/mimickit/LICENSE).

The walk file is a locally modified derivative of upstream `humanoid_walk.pkl`.
The other four clips are byte-identical to the corresponding files in the
recorded bundle. No AMASS motion or derivative is included.

## AMASS imitation model

`mjcf/humanoid_amass.xml` is a project-authored, joint-driven humanoid model under Apache-2.0. It uses the same 28-actuator reduced skeleton as the other humanoid tasks, with the contact geometry, joint stiffness, damping and force limits used for the AMASS experiment. It has no external mesh or texture dependencies.
