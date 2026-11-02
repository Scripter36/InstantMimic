# AMASS motions and latent policies

The AMASS experiment trains a joint-driven humanoid to imitate a large
collection of motions with a latent (VAE) policy. During imitation training, the
VAE posterior also observes the reference motion, while its prior and decoder
see only the observations that are available without one. Downstream command
tasks then freeze this policy and learn to steer it through its latent space.

This page walks through obtaining the data, converting it into reference
motions, training the latent policy and using it for point-goal navigation.

## Obtain the original data

AMASS is distributed by its authors under its own
[license](https://amass.is.tue.mpg.de/license.html), so this repository does not
include it. Create an account at [AMASS](https://amass.is.tue.mpg.de/), then
download and extract the SMPL/SMPL-H body-data archives of the collections you
want. Keep the original collection and directory names:

```text
data/amass/raw/
  ACCAD/<subject>/<clip>_poses.npz
  KIT/<subject>/<clip>_poses.npz
  CMU/<subject>/<clip>_poses.npz
  ...
```

The data can also live outside the checkout; the conversion command takes its
location as `--raw-root`. You do not need SMPL body model weights or a separate
retargeting tool.

## Prepare the PHC metadata

Clip filtering follows [PHC](https://github.com/ZhengyiLuo/PHC) and uses its
`amass_copycat_occlusion_v3.pkl` metadata. Download only that file: it is the
`amass_occlusion_v3` entry in
[PHC's download script](https://github.com/ZhengyiLuo/PHC/blob/96e92bb83cd91a4ad162e155ea3c1e50d5001dc7/download_data.sh).
Save it under `data/amass/metadata/` and convert it once:

```bash
uv run --project packages/amass --extra metadata python -m instantmimic_amass.metadata \
  data/amass/metadata/amass_copycat_occlusion_v3.pkl \
  data/amass/metadata/amass_copycat_occlusion_v3.npz
```

Before opening the pickle, the converter checks that the file matches the
SHA-256 of the copy we used. If PHC publishes a different version, pass its
checksum with `--expected-sha256`. Later steps read only the converted NPZ.

## Convert and filter

```bash
uv run --project packages/amass python -m instantmimic_amass.prepare \
  --raw-root data/amass/raw \
  --model data/humanoid/mjcf/humanoid_amass.xml \
  --metadata data/amass/metadata/amass_copycat_occlusion_v3.npz \
  --output data/humanoid/motions/amass/phc_train \
  --split train --workers 8
```

The conversion runs on the CPU. It maps SMPL joint rotations to the humanoid's
reduced skeleton, downsamples to 30 Hz, sets the height so the lowest point of
the character's geometry is 0.02 m below the floor, and applies PHC's split and
occlusion rules. The output is a directory of MimicKit-format NPZ files with
`frames`, `fps` and `loop_mode`; this humanoid uses 34 coordinates per frame.

The train split includes ACCAD, BioMotionLab_NTroje, BMLhandball, BMLmovi, CMU,
DanceDB, DFaust_67, EKUT, Eyes_Japan_Dataset, KIT, MPI_Limits, TCD_handMocap and
TotalCapture. Like PHC, it keeps clips that have no occlusion annotation. The
tool prints how many clips of each collection were annotated, and the
`.summary.json` file next to the output records clip counts, total duration,
filtering decisions and the hashes of the model and metadata. The total duration
depends on which archives you downloaded.

The `.fk` directory next to the output caches intermediate results, so an
interrupted conversion can resume. If you change the filtering inputs, write to
a new output directory.

Converted motions are specific to `humanoid_amass.xml`. To use another model,
convert again with that model.

## Train the latent policy

```bash
uv run train --config-name humanoid/amass/imitation/phc_train_pd_vae
```

PPO trains the prior, posterior and decoder together
(`src/instantmimic/rl/ppo/latent_prior.py`); there is no separate VAE training
step. The prior is only as general as the motion collection it was trained on.

## Point-goal navigation

Pass the trained checkpoint to the point-goal task:

```bash
uv run train --config-name humanoid/amass/command/goal \
  env.low_level_checkpoint=/absolute/path/to/run/checkpoints/best_agent.pt
```

The task restores the frozen policy's model, observation layout, control rate
and action semantics from the `run_bundle/` directory saved next to the
checkpoint, so keep the two together. The frozen policy can be a flat MLP
actor/critic with `RunningStandardScaler` observations and absolute (not
reference-relative) actions, as in the included AMASS config.

Episodes start from walking and running clips of ACCAD. If your dataset does not
include them, change `env.low_level_reset_motion_subsets` to patterns that match
clips you have.

`data/amass/` and `data/humanoid/motions/amass/` are ignored by Git, so the raw
and converted motions stay out of commits.
