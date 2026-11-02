# FreeMusco model and motion provenance

## Source

The included `animal/ostrich/` and `animal/chimanoid/` families originate from
[FreeMusco](https://github.com/palkan21/FreeMusco), Minkwan Kim and Yoonsang Lee,
SIGGRAPH Asia 2025. The layout follows upstream `Data/Muscle/Animal/`.
The ostrich model originates further upstream in
[OstrichRL](https://github.com/vittorione94/ostrichrl).

## License

The published BSD-3-Clause and MIT notices are retained in [LICENSE](LICENSE)
and [LICENSE.ostrichrl](LICENSE.ostrichrl), with [NOTICE.md](NOTICE.md).

## Redistribution

This source distribution includes the models, referenced meshes, actuator and
bone includes, and tracked reference-motion arrays. No trained checkpoints are
included. No additional license grant is asserted here.

## Local variants

The `_split2` and `_split4` models and their actuator/tendon/bone includes are
this project's muscle-count variants of the upstream models. The split variants
retain the same mechanical coordinates and total muscle force capacity while
dividing each original muscle into two or four actuator channels. The imitation
motion arrays and loop packs are retained with their existing filenames; pack
metadata is included where present.
