from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from instantmimic.config.schema import SemanticConfig
from instantmimic.utils.project_paths import (
    resolve_project_path,
    resolve_project_path_pattern,
)

_LOGGER = logging.getLogger(__name__)


def latest_best_checkpoint(log_root_path: Path) -> Path:
    if not log_root_path.is_dir():
        raise FileNotFoundError(
            f"Experiment log directory does not exist: {log_root_path}"
        )
    for experiment_dir in sorted(
        log_root_path.iterdir(), key=lambda path: path.name, reverse=True
    ):
        best_path = experiment_dir / "checkpoints" / "best_agent.pt"
        if experiment_dir.is_dir() and best_path.is_file():
            return best_path
    raise FileNotFoundError(
        f"No best_agent.pt checkpoint found under experiment runs in {log_root_path}. "
        "Set play.checkpoint to an existing checkpoint file."
    )


def resolve_checkpoint_path(
    checkpoint: str | None,
    log_root_path: Path,
) -> Path:
    if checkpoint:
        raw = checkpoint.strip()
        if raw:
            candidate = resolve_project_path(raw)
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(
                f"play.checkpoint must be an existing checkpoint file: {candidate}"
            )

    return latest_best_checkpoint(log_root_path)


def normalize_task_override(override: str) -> str | None:
    text = override.strip()
    if not text or "=" not in text:
        return None
    if text.startswith("~"):
        _LOGGER.warning("Skipping unsupported deletion override: %s", text)
        return None
    if text.startswith("++"):
        return text[2:]
    if text.startswith("+"):
        return text[1:]
    return text


def task_dotlist_overrides() -> list[str]:
    if not HydraConfig.initialized():
        return []
    dotlist: list[str] = []
    for item in HydraConfig.get().overrides.task:
        normalized = normalize_task_override(item)
        if normalized is not None:
            dotlist.append(normalized)
    return dotlist


def bundle_config_path(log_dir: Path) -> Path:
    return log_dir / "run_bundle" / "config" / "train_config_resolved.yaml"


def resolve_project_relative_path(path: str) -> str:
    return str(resolve_project_path(path))


def resolve_mjcf_path_for_play(
    path: str,
    log_dir: Path,
    *,
    bundle_cfg_path: Path | None = None,
) -> str:
    """Resolve a playback MJCF path, preferring the run bundle's snapshot.

    The snapshot is the model the run trained on; the authored project path is
    used only when the bundle does not hold the file, since the project copy
    may have been regenerated since.
    """
    raw = Path(path).expanduser()
    candidates: list[Path] = []

    if not raw.is_absolute():
        candidates.append((log_dir / raw).resolve())
        candidates.append((log_dir / "run_bundle" / "mjcf_tree" / raw).resolve())
        if bundle_cfg_path is not None:
            candidates.append((bundle_cfg_path.parent / raw).resolve())

    parts = list(raw.parts)
    if "run_bundle" in parts:
        run_bundle_idx = parts.index("run_bundle")
        candidates.append((log_dir / Path(*parts[run_bundle_idx:])).resolve())
    if "mjcf_tree" in parts:
        mjcf_tree_idx = parts.index("mjcf_tree")
        candidates.append(
            (log_dir / "run_bundle" / Path(*parts[mjcf_tree_idx:])).resolve()
        )
    project_candidate = Path(resolve_project_relative_path(path)).resolve()
    candidates.append(project_candidate)

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return str(candidate)
    return str(project_candidate)


def resolve_bundled_input_path_for_play(
    path: str,
    log_dir: Path,
    *,
    bundle_cfg_path: Path | None = None,
) -> str:
    """Prefer an immutable repro-bundle input over the authored source path."""

    raw = Path(path).expanduser()
    bundle_root = (log_dir / "run_bundle").resolve()
    input_root = (bundle_root / "inputs").resolve()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(input_root / raw.name)
    elif "run_bundle" not in raw.parts:
        candidate = (input_root / raw).resolve()
        if candidate.is_relative_to(input_root):
            candidates.append(candidate)
    if bundle_cfg_path is not None and not raw.is_absolute():
        config_bundle_root = bundle_cfg_path.parent.parent.resolve()
        candidate = (config_bundle_root / "inputs" / raw).resolve()
        if candidate.is_relative_to(config_bundle_root / "inputs"):
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return resolve_mjcf_path_for_play(
        path,
        log_dir,
        bundle_cfg_path=bundle_cfg_path,
    )


def resolve_motion_files_for_play(
    path: str,
    log_dir: Path,
    *,
    bundle_cfg_path: Path | None = None,
) -> list[str]:
    extra_roots: list[Path] = [log_dir]
    if bundle_cfg_path is not None:
        extra_roots.append(bundle_cfg_path.parent)
    matches = resolve_project_path_pattern(path, extra_roots=extra_roots)
    if matches:
        return [str(match) for match in matches]
    raise FileNotFoundError(f"motion.source pattern matched no files: {path}")


# Exact checkout-relative names retired by the canonical model rename. This
# compatibility is only for artifacts without the auxiliary model snapshot.
_RENAMED_AUXILIARY_MODELS = {
    "data/snu_humanoid/mjcf/snu_humanoid_ball_feet_pd.xml": "data/snu_humanoid/mjcf/humanoid_pd.xml",
}


def _resolve_auxiliary_model_for_play(
    path: str, log_dir: Path, *, bundle_cfg_path: Path | None = None
) -> str:
    resolved = resolve_mjcf_path_for_play(
        path, log_dir=log_dir, bundle_cfg_path=bundle_cfg_path
    )
    if Path(resolved).is_file():
        return resolved
    replacement = _RENAMED_AUXILIARY_MODELS.get(path)
    if replacement is not None:
        canonical = Path(resolve_project_relative_path(replacement))
        if canonical.is_file():
            _LOGGER.warning(
                "Artifact auxiliary model has no snapshot; using the renamed "
                "checkout model %s for %s",
                replacement,
                path,
            )
            return str(canonical)
    raise FileNotFoundError(f"Artifact auxiliary model is missing: {path}")


def resolve_artifact_paths(
    cfg: DictConfig | SemanticConfig,
    log_dir: Path,
    *,
    bundle_cfg_path: Path | None = None,
) -> Any:
    """Resolve portable artifact paths on a detached runtime config."""

    merged_cfg = OmegaConf.create(
        cfg.to_dict() if hasattr(cfg, "to_dict") else OmegaConf.to_container(cfg)
    )
    with open_dict(merged_cfg):
        merged_cfg.env.sim.mjcf_path = resolve_mjcf_path_for_play(
            str(merged_cfg.env.sim.mjcf_path),
            log_dir=log_dir,
            bundle_cfg_path=bundle_cfg_path,
        )
        auxiliary_key = "env.action.policy_output.stable_pd.mjcf_path"
        auxiliary = OmegaConf.select(merged_cfg, auxiliary_key)
        if auxiliary is not None:
            OmegaConf.update(
                merged_cfg,
                auxiliary_key,
                _resolve_auxiliary_model_for_play(
                    str(auxiliary), log_dir, bundle_cfg_path=bundle_cfg_path
                ),
            )
        motion = merged_cfg.env.get("motion", None)
        if motion is not None and "source" in motion:
            source = motion.source
            patterns = [source] if isinstance(source, str) else list(source or [])
            resolved_sources: list[str] = []
            seen_sources: set[str] = set()
            for path in patterns:
                pattern = str(path or "").strip()
                if not pattern:
                    continue
                for resolved_path in resolve_motion_files_for_play(
                    pattern,
                    log_dir=log_dir,
                    bundle_cfg_path=bundle_cfg_path,
                ):
                    if resolved_path in seen_sources:
                        continue
                    seen_sources.add(resolved_path)
                    resolved_sources.append(resolved_path)
            motion.source = (
                resolved_sources[0]
                if isinstance(source, str) and len(resolved_sources) == 1
                else resolved_sources
            )
        reset = merged_cfg.env.get("reset", None)
        if reset is not None and reset.get("motion_source", None) is not None:
            reset.motion_source = resolve_bundled_input_path_for_play(
                str(reset.motion_source),
                log_dir=log_dir,
                bundle_cfg_path=bundle_cfg_path,
            )
        if "muscle_file" in merged_cfg.env:
            del merged_cfg.env["muscle_file"]

    return merged_cfg


def normalize_cli_override_arg(arg: str) -> str | None:
    text = arg.strip()
    if not text or "=" not in text or text.startswith("-"):
        return None
    if text.startswith("~"):
        raise ValueError(
            f"Deletion overrides are not supported in checkpoint-first playback: {arg}"
        )
    if text.startswith("++"):
        return text[2:]
    if text.startswith("+"):
        return text[1:]
    return text


def cli_dotlist_overrides(argv: list[str]) -> list[str]:
    dotlist: list[str] = []
    for arg in argv:
        normalized = normalize_cli_override_arg(arg)
        if normalized is not None:
            dotlist.append(normalized)
    return dotlist


def validate_playback_dotlist_overrides(overrides: list[str]) -> None:
    """Warn when playback explicitly changes the training artifact config."""

    artifact_paths = [
        override.split("=", 1)[0].strip()
        for override in overrides
        if not override.split("=", 1)[0].strip().startswith("play.")
    ]
    if artifact_paths:
        _LOGGER.warning(
            "Playback overrides modify the training artifact config and may change "
            "reproducibility: %s",
            ", ".join(artifact_paths),
        )


def cli_override_value(argv: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for override in cli_dotlist_overrides(argv):
        if override.startswith(prefix):
            return override[len(prefix) :]
    return None
