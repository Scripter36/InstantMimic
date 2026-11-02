# AMASS preparation

CPU tools for converting a locally obtained AMASS SMPL/SMPL-H dataset to humanoid reference motions. The package does not download or redistribute motion data or SMPL body models. The reduced-skeleton FK path needs no SMPL model weights or IK solver.

From the repository root:

```bash
uv run --project packages/amass python -m instantmimic_amass.prepare --help
```

See [the AMASS guide](../../docs/data/amass-retargeting.md) for the data layout, PHC metadata provenance, preprocessing and training commands.
