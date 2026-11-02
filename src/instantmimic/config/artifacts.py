from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from instantmimic.config.loader import load_semantic_config
from instantmimic.config.schema import SemanticConfig
from instantmimic.rl.checkpoint_io import load_checkpoint_payload

# Format 2 keeps the run's config only in its run bundle. Format 1 also embedded
# it in every checkpoint; that copy is read only when no run bundle exists.
CHECKPOINT_FORMAT_VERSION = 2
_EMBEDDED_CONFIG_FORMAT_VERSION = 1
CHECKPOINT_METADATA_KEY = "_instantmimic"
RUN_BUNDLE_DIRNAME = "run_bundle"
RUN_BUNDLE_CONFIG = Path("config") / "train_config_resolved.yaml"
RUN_BUNDLE_MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class ArtifactSemanticConfig:
    config: SemanticConfig
    resolved_seed: int | None
    bundle_config_path: Path | None = None


def build_checkpoint_metadata() -> dict[str, object]:
    """Build the versioned, non-tensor metadata stored in every checkpoint.

    The config and resolved seed live in the run bundle, next to the MJCF
    snapshot they describe, so a run has one editable copy of its settings.
    """

    return {"checkpoint_format_version": CHECKPOINT_FORMAT_VERSION}


def run_bundle_root(log_dir: str | Path) -> Path:
    return Path(log_dir) / RUN_BUNDLE_DIRNAME


def _valid_seed(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _checkpoint_metadata(
    checkpoint_path: Path,
) -> Mapping[str, object] | None:
    payload = load_checkpoint_payload(checkpoint_path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Checkpoint payload must be a mapping: {checkpoint_path}")
    raw_metadata = payload.get(CHECKPOINT_METADATA_KEY)
    if raw_metadata is None:
        return None
    if not isinstance(raw_metadata, Mapping):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has invalid {CHECKPOINT_METADATA_KEY} metadata."
        )
    format_version = raw_metadata.get("checkpoint_format_version")
    if format_version not in (
        _EMBEDDED_CONFIG_FORMAT_VERSION,
        CHECKPOINT_FORMAT_VERSION,
    ):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} uses unsupported checkpoint format "
            f"version {format_version!r}; expected {CHECKPOINT_FORMAT_VERSION}."
        )
    return raw_metadata


def _embedded_semantic_config(
    metadata: Mapping[str, object],
    *,
    checkpoint_path: Path,
) -> ArtifactSemanticConfig:
    """Read the config a format-1 checkpoint embedded (no run bundle)."""

    resolved_seed = _valid_seed(metadata.get("resolved_seed"))
    if resolved_seed is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} metadata has invalid resolved_seed."
        )
    semantic_values = metadata.get("semantic_config")
    if not isinstance(semantic_values, Mapping):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} metadata is missing semantic_config."
        )
    semantic = load_semantic_config(semantic_values)
    if metadata.get("config_schema_version") != semantic.schema_version:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} config schema metadata disagrees with "
            "semantic_config.schema_version."
        )
    return ArtifactSemanticConfig(config=semantic, resolved_seed=resolved_seed)


def _adjacent_bundle_config_path(checkpoint_path: Path) -> Path | None:
    for parent in checkpoint_path.parents:
        candidate = run_bundle_root(parent) / RUN_BUNDLE_CONFIG
        if candidate.is_file():
            return candidate
    return None


def _manifest_resolved_seed(bundle_root: Path) -> int | None:
    try:
        manifest = json.loads(
            (bundle_root / RUN_BUNDLE_MANIFEST).read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Run bundle manifest is not JSON: {bundle_root}") from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"Run bundle manifest must be a mapping: {bundle_root}")
    if "resolved_seed" not in manifest:
        return None  # pre-format-2 run bundle
    resolved_seed = _valid_seed(manifest["resolved_seed"])
    if resolved_seed is None:
        raise RuntimeError(
            f"Run bundle manifest has invalid resolved_seed: {bundle_root}"
        )
    return resolved_seed


def load_artifact_semantic_config(checkpoint_path: str | Path) -> SemanticConfig:
    """Load the config of the run that produced ``checkpoint_path``."""

    return load_artifact_semantic_record(checkpoint_path).config


def load_artifact_semantic_record(
    checkpoint_path: str | Path,
) -> ArtifactSemanticConfig:
    """Load a checkpoint's run config and actual training seed.

    The adjacent run bundle is authoritative. Only a format-1 checkpoint without
    a run bundle falls back to the config it embedded.
    """

    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_checkpoint}")
    bundle_path = _adjacent_bundle_config_path(resolved_checkpoint)
    if bundle_path is None:
        metadata = _checkpoint_metadata(resolved_checkpoint)
        if (
            metadata is not None
            and metadata["checkpoint_format_version"] == _EMBEDDED_CONFIG_FORMAT_VERSION
        ):
            return _embedded_semantic_config(
                metadata, checkpoint_path=resolved_checkpoint
            )
        raise RuntimeError(
            f"Checkpoint {resolved_checkpoint} has no adjacent run bundle "
            f"({RUN_BUNDLE_DIRNAME}/{RUN_BUNDLE_CONFIG.as_posix()} in the run "
            "directory); its config lives there."
        )

    bundle = OmegaConf.load(bundle_path)
    semantic = load_semantic_config(bundle)
    resolved_seed = _manifest_resolved_seed(bundle_path.parent.parent)
    if resolved_seed is None:
        # Runs before format 2 recorded the seed in each checkpoint, and runs
        # before that in the bundled env config.
        metadata = _checkpoint_metadata(resolved_checkpoint)
        if metadata is not None:
            resolved_seed = _valid_seed(metadata.get("resolved_seed"))
    if resolved_seed is None:
        env = bundle.get("env", {})
        resolved_seed = _valid_seed(env.get("seed") if hasattr(env, "get") else None)
    if resolved_seed is None:
        resolved_seed = _valid_seed(semantic.seed)
    return ArtifactSemanticConfig(
        config=semantic, resolved_seed=resolved_seed, bundle_config_path=bundle_path
    )


def clone_checkpoint_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Detach immutable metadata from caller-owned mappings before checkpoint writes."""

    return copy.deepcopy(dict(metadata))


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CHECKPOINT_METADATA_KEY",
    "RUN_BUNDLE_CONFIG",
    "RUN_BUNDLE_DIRNAME",
    "RUN_BUNDLE_MANIFEST",
    "ArtifactSemanticConfig",
    "build_checkpoint_metadata",
    "clone_checkpoint_metadata",
    "load_artifact_semantic_config",
    "load_artifact_semantic_record",
    "run_bundle_root",
]
