from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from instantmimic.envs.config_access import get_value

CHARACTER_RENDER_MODES = ("mesh", "collision", "both")

_CHARACTER_VISUAL_GEOM_GROUPS = (2,)
_CHARACTER_COLLISION_GEOM_GROUPS = (0, 3, 4)


def normalize_character_render_mode(value: object) -> str:
    mode = str(value).strip().lower()
    if mode not in CHARACTER_RENDER_MODES:
        raise ValueError(
            "character_render_mode must be one of: "
            + ", ".join(CHARACTER_RENDER_MODES)
            + f"; got {value!r}."
        )
    return mode


def resolve_character_render_mode(cfg: Any, *, default: str = "mesh") -> str:
    raw_mode = get_value(cfg, "character_render_mode", default)
    return normalize_character_render_mode(raw_mode)


def character_geom_group_visibility(mode: object) -> dict[int, bool]:
    normalized = normalize_character_render_mode(mode)
    show_mesh = normalized in {"mesh", "both"}
    show_collision = normalized in {"collision", "both"}
    visibility = dict.fromkeys(_CHARACTER_COLLISION_GEOM_GROUPS, show_collision)
    visibility.update(dict.fromkeys(_CHARACTER_VISUAL_GEOM_GROUPS, show_mesh))
    return visibility


def _body_name(model: Any, body_id: int) -> str:
    try:
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(body_id),
        )
    except (TypeError, ValueError):
        return ""
    return str(name) if name else ""


def _root_body_name_candidates(env_raw: Any | None) -> tuple[str, ...]:
    if env_raw is None:
        return ()
    cfg = getattr(env_raw, "cfg", None)
    candidates: list[str] = []
    for key in ("root_body_names", "root_body_name"):
        value = get_value(cfg, key, None) if cfg is not None else None
        if isinstance(value, str):
            candidates.append(value)
        elif value is not None:
            try:
                candidates.extend(str(item) for item in value)
            except TypeError:
                continue
    profile = getattr(env_raw, "_profile", None)
    profile_name = getattr(profile, "root_body_name", None)
    if profile_name is not None:
        candidates.append(str(profile_name))
    return tuple(name for name in candidates if name)


def _explicit_root_body_ids(model: Any, env_raw: Any | None) -> tuple[int, ...]:
    if env_raw is None:
        return ()
    nbody = int(getattr(model, "nbody", 0))
    root_ids: list[int] = []
    for attr in ("root_body_idx", "root_body_id"):
        value = getattr(env_raw, attr, None)
        if value is None:
            continue
        try:
            body_id = int(value)
        except (TypeError, ValueError):
            continue
        if 0 < body_id < nbody:
            root_ids.append(body_id)
    for name in _root_body_name_candidates(env_raw):
        try:
            body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        except (TypeError, ValueError):
            continue
        if 0 < body_id < nbody:
            root_ids.append(body_id)
    return tuple(dict.fromkeys(root_ids))


def character_body_mask(model: Any, *, env_raw: Any | None = None) -> np.ndarray:
    """Return bodies belonging to the controlled character and reference ghost."""
    nbody = int(getattr(model, "nbody", 0))
    body_parentid = getattr(model, "body_parentid", None)
    if nbody <= 0 or body_parentid is None:
        return np.zeros(max(0, nbody), dtype=bool)

    parent_ids = np.asarray(body_parentid, dtype=np.int32).reshape(-1)
    if parent_ids.size < nbody:
        return np.zeros(nbody, dtype=bool)

    root_ids = list(_explicit_root_body_ids(model, env_raw))
    if not root_ids:
        root_ids = [
            body_id
            for body_id in range(1, nbody)
            if int(parent_ids[body_id]) == 0
            and not _body_name(model, body_id).startswith("ghost_")
        ][:1]

    mask = np.zeros(nbody, dtype=bool)
    for body_id in root_ids:
        mask[int(body_id)] = True
    for body_id in range(1, nbody):
        parent_id = int(parent_ids[body_id])
        if 0 <= parent_id < nbody and mask[parent_id]:
            mask[body_id] = True
    for body_id in range(1, nbody):
        if _body_name(model, body_id).startswith("ghost_"):
            mask[body_id] = True
    return mask


def character_geom_semantic_masks(
    model: Any,
    *,
    env_raw: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return visual-character, collision-character, and scene geom masks."""
    geom_groups = np.asarray(model.geom_group)
    geom_body_ids = np.asarray(model.geom_bodyid)
    geom_types = np.asarray(model.geom_type)
    body_mask = character_body_mask(model, env_raw=env_raw)
    if body_mask.size:
        character_mask = body_mask[geom_body_ids]
    else:
        # Lightweight test doubles and older callers may not expose body hierarchy.
        character_mask = geom_body_ids != 0
    visual_mask = character_mask & (
        (geom_groups == _CHARACTER_VISUAL_GEOM_GROUPS[0])
        | (geom_types == int(mujoco.mjtGeom.mjGEOM_MESH))
    )
    collision_mask = character_mask & ~visual_mask
    return visual_mask, collision_mask, ~character_mask


def resolve_character_render_mode_for_model(
    model: Any,
    mode: object,
    *,
    env_raw: Any | None = None,
) -> str:
    """Keep primitive-only characters visible under the default mesh mode."""
    normalized = normalize_character_render_mode(mode)
    if normalized != "mesh":
        return normalized
    visual_mask, collision_mask, _ = character_geom_semantic_masks(
        model,
        env_raw=env_raw,
    )
    if not bool(np.any(visual_mask)) and bool(np.any(collision_mask)):
        return "collision"
    return normalized


def classify_character_geom_semantic(
    model: Any,
    geom_ids: list[int] | np.ndarray,
    *,
    body_id: int,
    env_raw: Any | None = None,
) -> str:
    """Classify a MuJoCo geom batch as character visual/collision or scene."""
    geom_ids_np = np.asarray(geom_ids, dtype=np.int32).reshape(-1)
    visual_mask, collision_mask, scene_mask = character_geom_semantic_masks(
        model,
        env_raw=env_raw,
    )
    if geom_ids_np.size == 0:
        body_mask = character_body_mask(model, env_raw=env_raw)
        if body_mask.size and 0 <= int(body_id) < body_mask.size:
            return "visual" if bool(body_mask[int(body_id)]) else "scene"
        return "visual" if int(body_id) != 0 else "scene"
    if bool(np.all(scene_mask[geom_ids_np])):
        return "scene"
    if bool(np.all(collision_mask[geom_ids_np])):
        return "collision"
    if bool(np.any(visual_mask[geom_ids_np])):
        return "visual"
    return "scene"


def hidden_character_geom_mask(
    model: Any,
    mode: object,
    *,
    env_raw: Any | None = None,
) -> np.ndarray:
    normalized = normalize_character_render_mode(mode)
    visual_mask, collision_mask, _ = character_geom_semantic_masks(
        model,
        env_raw=env_raw,
    )
    if normalized == "mesh":
        return collision_mask
    if normalized == "collision":
        return visual_mask
    return np.zeros(int(model.ngeom), dtype=bool)


def normalize_character_geom_groups(
    model: Any,
    *,
    env_raw: Any | None = None,
) -> None:
    geom_groups = np.asarray(model.geom_group)
    visual_mask, collision_mask, scene_mask = character_geom_semantic_masks(
        model,
        env_raw=env_raw,
    )
    geom_groups[scene_mask] = 1
    geom_groups[visual_mask] = _CHARACTER_VISUAL_GEOM_GROUPS[0]
    geom_groups[collision_mask] = 3


def apply_character_render_mode_to_model(
    model: Any,
    mode: object,
    *,
    env_raw: Any | None = None,
) -> bool:
    resolved_mode = resolve_character_render_mode_for_model(
        model,
        mode,
        env_raw=env_raw,
    )
    hidden_mask = hidden_character_geom_mask(model, resolved_mode, env_raw=env_raw)
    if not bool(np.any(hidden_mask)):
        return False
    model.geom_matid[hidden_mask] = -1
    model.geom_rgba[hidden_mask, 3] = 0.0
    return True
