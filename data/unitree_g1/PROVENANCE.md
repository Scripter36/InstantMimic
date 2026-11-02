# G1 BeyondMimic runtime inputs

The runtime model and side-kick trajectory are prepared locally, not bundled.
Follow `docs/guides/beyondmimic.md` and keep the downloaded sources' notices.

| Input | Identity | Terms |
| --- | --- | --- |
| mjlab tracking scene and conversion | `15ebce8840ee2205f4c62d0c20df65dd1794cab4` | mjlab source: Apache-2.0; model assets retain their upstream notices |
| PBHC side kick | repository `ffac5cded61fe78b39b051f09ac2ed0f6a1ccea0`, `example/motion_data/Side_kick.pkl` | CC BY-NC 4.0 |

The source PKL SHA-256 is
`3152b3b53537e426bba57c5c3e04bd5dff74b00a38ee3a33a9c8dc1de7228821`.
The preparation tool verifies it before restricted NumPy-only loading, expands
the source 23-joint layout to 29 joints, and regenerates body-state tensors using
the pinned source converter.

`beyondmimic/` contains the local model, meshes, motion and preparation manifest. That entire directory is ignored and excluded from package
builds. The repository's source license does not replace upstream data terms.
