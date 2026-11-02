"""BVH joint to MuJoCo body mapping (`env.motion.bvh_map`).

Each entry maps a BVH joint to one body name or to a list of body names. A
list gives every listed body the joint's local rotation, for a model that
splits one BVH segment into several bodies (a single BVH toe joint driving
both toe bodies of a two-toe foot).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

BvhMapValue = str | Sequence[str]


def bvh_map_targets(value: BvhMapValue) -> tuple[str, ...]:
    """The MuJoCo bodies one bvh_map entry drives."""

    if isinstance(value, str):
        return (value,)
    targets = tuple(str(name) for name in value)
    if not targets:
        raise ValueError("A bvh_map entry must name at least one body.")
    return targets


def bvh_map_body_names(bvh_map: Mapping[str, BvhMapValue]) -> set[str]:
    """Every MuJoCo body a bvh_map drives."""

    return {name for value in bvh_map.values() for name in bvh_map_targets(value)}


def normalize_bvh_map(bvh_map: Mapping[object, object]) -> dict[str, str | list[str]]:
    """A plain bvh_map: string keys, a body name or a list of body names per entry."""

    out: dict[str, str | list[str]] = {}
    for key, value in bvh_map.items():
        targets = bvh_map_targets(value if isinstance(value, str) else list(value))
        out[str(key)] = targets[0] if isinstance(value, str) else list(targets)
    return out
