# SNU humanoid base model and motion provenance

## Source and included files

The SNU musculoskeletal model was produced by this research group. This
distribution includes its base model and the corresponding StablePD actuation:

- `mjcf/humanoid.xml`
- `mjcf/humanoid_pd.xml`
- `mjcf/include/body.xml`
- `mjcf/include/muscles.xml`
- the 23 OBJ meshes referenced by the shared body include, under `mjcf/assets/`

The muscle model reads both includes. The StablePD model reads the shared body
include. Neither include has further MJCF dependencies. The tendon include carries the
model's muscle parameters and muscle-name ordering.

The included motion files are this group's own captures:

- `motions/single/walk.bvh`
- `motions/single/run.bvh`
- `motions/single/backflip.bvh`

## License and redistribution

The models, their meshes and includes, and this group's three capture clips
listed above are distributed under Apache-2.0.

## Additional model variants

This distribution also includes `snu_humanoid_ball_feet_hand_muscle.xml`,
`snu_humanoid_ball_full_muscle.xml`,
`snu_humanoid_ball_full_leg_self_collision_muscle.xml`, with their required
body/asset and tendon includes. They derive from the same group-authored base model under the
same license. They do not add external motion datasets or separately sourced
muscle parameter sets.
