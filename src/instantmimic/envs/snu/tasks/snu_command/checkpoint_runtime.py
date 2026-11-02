from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path

from omegaconf import OmegaConf

from instantmimic.envs.config_access import get_value as cfg_get
from instantmimic.utils.project_paths import (
    resolve_project_path,
)

_LOW_LEVEL_ACTION_KEYS = ("type", "scale", "use_reference")
_LOW_LEVEL_MOTION_KEYS = (
    "scale",
    "offset",
    "bvh_map",
    "rest_bodies",
    "joint_layout",
    "root_auto_calibration",
    "cache_mode",
    "cache_directory",
    "loop",
)


def cfg_set(cfg, key: str, value) -> None:
    if isinstance(cfg, MutableMapping):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def cfg_to_container(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def resolve_low_level_checkpoint_config_path(checkpoint_path: str | Path) -> Path:
    resolved = Path(checkpoint_path).expanduser().resolve()
    for parent in resolved.parents:
        candidate = parent / "run_bundle" / "config" / "train_config_resolved.yaml"
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Could not locate run_bundle/config/train_config_resolved.yaml for "
        f"low-level checkpoint {resolved}."
    )


def resolve_checkpoint_config_path_value(raw_path: str, bundle_cfg_path: Path) -> str:
    raw = Path(str(raw_path)).expanduser()
    run_dir = bundle_cfg_path.parent.parent.parent
    candidates: list[Path] = [resolve_project_path(str(raw_path)).resolve()]

    if not raw.is_absolute():
        candidates.append((run_dir / raw).resolve())
        candidates.append((bundle_cfg_path.parent / raw).resolve())

    parts = list(raw.parts)
    if "run_bundle" in parts:
        run_bundle_idx = parts.index("run_bundle")
        candidates.append((run_dir / Path(*parts[run_bundle_idx:])).resolve())
    if "mjcf_tree" in parts:
        mjcf_tree_idx = parts.index("mjcf_tree")
        candidates.append(
            (run_dir / "run_bundle" / Path(*parts[mjcf_tree_idx:])).resolve()
        )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def iter_motion_pattern_strings(raw_value) -> list[str]:
    if raw_value is None:
        return []
    value = cfg_to_container(raw_value)
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def cfg_nested_get(cfg, *keys: str, default=None):
    current = cfg
    for key in keys:
        current = cfg_get(current, key, None)
        if current is None:
            return default
    return current


def cfg_nested_set(cfg, *keys: str, value) -> None:
    current = cfg
    for key in keys[:-1]:
        child = cfg_get(current, key, None)
        if child is None:
            child = {}
            cfg_set(current, key, child)
        current = child
    cfg_set(current, keys[-1], value)


def infer_amass_motion_root(low_level_env_cfg) -> str | None:
    motion_candidates = iter_motion_pattern_strings(
        cfg_nested_get(low_level_env_cfg, "motion", "source", default=None)
    )
    marker = "data/humanoid/motions/amass/"
    for raw_text in motion_candidates:
        text = str(raw_text)
        marker_idx = text.find(marker)
        if marker_idx < 0:
            continue
        rest = text[marker_idx + len(marker) :]
        root_name = rest.split("/", 1)[0].strip()
        if root_name:
            return text[:marker_idx] + marker + root_name
    return None


def infer_reset_motion_sources_from_low_level_cfg(
    low_level_env_cfg,
    command_cfg,
) -> list[str]:
    motion_root = infer_amass_motion_root(low_level_env_cfg)
    if motion_root is None:
        return []
    subsets = iter_motion_pattern_strings(
        cfg_get(
            command_cfg,
            "low_level_reset_motion_subsets",
            ["ACCAD/*Walking_c3d/*.npz", "ACCAD/*Running_c3d/*.npz"],
        )
    )
    return [f"{motion_root}/{subset.lstrip('/')}" for subset in subsets]


def apply_low_level_checkpoint_runtime_config(cfg) -> None:
    checkpoint = str(cfg_get(cfg, "low_level_checkpoint", "") or "").strip()
    if not checkpoint or not bool(cfg_get(cfg, "low_level_config_autoload", False)):
        return

    bundle_cfg_path = resolve_low_level_checkpoint_config_path(checkpoint)
    low_level_cfg = OmegaConf.load(bundle_cfg_path)
    low_env_cfg = cfg_get(low_level_cfg, "env", {})
    low_sim_cfg = cfg_get(low_env_cfg, "sim", {})

    low_mjcf_path = cfg_get(low_sim_cfg, "mjcf_path", None)
    if low_mjcf_path:
        sim_cfg = cfg_get(cfg, "sim", None)
        if sim_cfg is not None:
            cfg_set(
                sim_cfg,
                "mjcf_path",
                resolve_checkpoint_config_path_value(
                    str(low_mjcf_path),
                    bundle_cfg_path,
                ),
            )

    low_action_cfg = cfg_get(low_env_cfg, "action", {})
    for key in _LOW_LEVEL_ACTION_KEYS:
        value = cfg_get(low_action_cfg, key, None)
        if value is not None:
            cfg_nested_set(cfg, "action", key, value=cfg_to_container(value))

    if bool(cfg_get(low_action_cfg, "use_reference", False)):
        raise ValueError(
            "Frozen downstream control requires action.use_reference=false"
        )
    hz = cfg_get(low_sim_cfg, "hz", None)
    if hz is not None:
        cfg_nested_set(cfg, "sim", "hz", value=hz)
    decimation = cfg_get(low_env_cfg, "decimation", None)
    if decimation is not None:
        cfg_set(cfg, "decimation", decimation)

    low_motion_cfg = cfg_get(low_env_cfg, "motion", {})
    for key in _LOW_LEVEL_MOTION_KEYS:
        value = cfg_get(low_motion_cfg, key, None)
        if value is not None:
            cfg_nested_set(cfg, "motion", key, value=cfg_to_container(value))

    for key in ("observations", "spawn_offset"):
        value = cfg_get(low_env_cfg, key, None)
        if value is not None:
            cfg_set(cfg, key, cfg_to_container(value))

    reset_motion_sources = iter_motion_pattern_strings(
        cfg_nested_get(cfg, "reset", "motion", "source", default=None)
    )
    if not reset_motion_sources:
        inferred = infer_reset_motion_sources_from_low_level_cfg(low_env_cfg, cfg)
        if not inferred:
            inferred = iter_motion_pattern_strings(
                cfg_nested_get(low_env_cfg, "motion", "source", default=None)
            )
        if inferred:
            cfg_nested_set(cfg, "reset", "motion", "source", value=inferred)
