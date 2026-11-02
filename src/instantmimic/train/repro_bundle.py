from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from instantmimic.config.artifacts import (
    RUN_BUNDLE_CONFIG,
    RUN_BUNDLE_MANIFEST,
    run_bundle_root,
)
from instantmimic.config.schema import SemanticConfig
from instantmimic.utils.project_paths import (
    candidate_project_roots,
    resolve_project_path,
    user_cache_root,
)

_LOGGER = logging.getLogger(__name__)

_LINUX_FICLONE = 0x40049409
_REPRO_BUNDLE_CACHE_SCHEMA = 2
_REPRO_BUNDLE_CACHE_DIR = user_cache_root("repro_bundle")


def _resolve_source_path(path: str | Path) -> Path:
    return resolve_project_path(path)


def _source_relative_path(path: Path) -> Path:
    resolved = path.resolve()
    for root in candidate_project_roots():
        if resolved.is_relative_to(root):
            return resolved.relative_to(root)
    return Path(resolved.name)


def _file_signature(path: Path) -> tuple[str, int, int]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def _repro_bundle_cache_dir() -> Path | None:
    cache_dir = _REPRO_BUNDLE_CACHE_DIR
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return cache_dir


def _repro_bundle_cache_key(source_mjcf: Path) -> str:
    payload = {
        "schema": _REPRO_BUNDLE_CACHE_SCHEMA,
        "root_signature": _file_signature(source_mjcf),
    }
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _dependency_cache_payload_is_valid(
    payload: Any,
    *,
    source_mjcf: Path,
) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("schema") != _REPRO_BUNDLE_CACHE_SCHEMA:
        return False
    if tuple(payload.get("root_signature", ())) != _file_signature(source_mjcf):
        return False

    for raw_signature in payload.get("xml_signatures", ()):
        try:
            path_str, size, mtime_ns = raw_signature
            if _file_signature(Path(path_str)) != (path_str, int(size), int(mtime_ns)):
                return False
        except (OSError, TypeError, ValueError):
            return False
    return True


def _load_dependency_cache(source_mjcf: Path) -> tuple[list[Path], list[Path]] | None:
    cache_dir = _repro_bundle_cache_dir()
    if cache_dir is None:
        return None
    cache_path = cache_dir / f"dependencies-{_repro_bundle_cache_key(source_mjcf)}.json"
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not _dependency_cache_payload_is_valid(payload, source_mjcf=source_mjcf):
        return None

    try:
        xml_files = [Path(path) for path in payload["xml_files"]]
        asset_files = [Path(path) for path in payload["asset_files"]]
    except (KeyError, TypeError):
        return None
    return xml_files, asset_files


def _store_dependency_cache(
    source_mjcf: Path,
    *,
    xml_files: list[Path],
    asset_files: list[Path],
) -> None:
    cache_dir = _repro_bundle_cache_dir()
    if cache_dir is None:
        return
    payload = {
        "schema": _REPRO_BUNDLE_CACHE_SCHEMA,
        "root_signature": _file_signature(source_mjcf),
        "xml_signatures": [_file_signature(path) for path in xml_files],
        "xml_files": [str(path.expanduser().resolve()) for path in xml_files],
        "asset_files": [str(path.expanduser().resolve()) for path in asset_files],
    }
    key = _repro_bundle_cache_key(source_mjcf)
    cache_path = cache_dir / f"dependencies-{key}.json"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=cache_dir,
            prefix=f".dependencies-{key}-",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle)
        os.replace(tmp_path, cache_path)
    except OSError:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                tmp_path.unlink()


def _collect_mjcf_bundle_dependencies(
    source_mjcf: Path,
) -> tuple[list[Path], list[Path]]:
    cached = _load_dependency_cache(source_mjcf)
    if cached is not None:
        return cached

    source_xml_files = _collect_mjcf_xml_dependencies(source_mjcf)
    source_asset_files = _collect_mjcf_asset_dependencies(source_xml_files)
    _store_dependency_cache(
        source_mjcf,
        xml_files=source_xml_files,
        asset_files=source_asset_files,
    )
    return source_xml_files, source_asset_files


def _collect_mjcf_xml_dependencies(root_xml: Path) -> list[Path]:
    root_xml = root_xml.resolve()
    source_root = root_xml.parent
    queue = [root_xml]
    visited: set[Path] = set()
    ordered: list[Path] = []

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        ordered.append(current)

        try:
            tree = ET.parse(current)
        except Exception:
            _LOGGER.exception("Failed parsing MJCF XML dependency: %s", current)
            continue

        xml_root = tree.getroot()
        for include in xml_root.findall(".//include"):
            include_file = include.get("file")
            if not include_file:
                continue
            include_path = (current.parent / include_file).resolve()
            if not include_path.exists():
                include_path = (source_root / include_file).resolve()
            if include_path.exists():
                queue.append(include_path)
            else:
                _LOGGER.warning(
                    "MJCF include path not found while bundling: %s (from %s)",
                    include_file,
                    current,
                )

    return ordered


def _collect_mjcf_asset_dependencies(xml_files: list[Path]) -> list[Path]:
    ordered: list[Path] = []
    seen: set[Path] = set()

    # MuJoCo resolves meshdir/texturedir relative to the main XML file,
    # not included files.  The first entry in xml_files is the main XML.
    main_dir = xml_files[0].parent if xml_files else Path()

    # Pre-collect meshdir/texturedir from *all* XML files (compiler directives
    # in includes like shared.xml apply globally, just as MuJoCo does).
    global_mesh_base = main_dir
    global_texture_base = main_dir
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except (OSError, ET.ParseError):
            continue
        compiler = tree.getroot().find("compiler")
        if compiler is not None:
            meshdir = compiler.get("meshdir")
            texturedir = compiler.get("texturedir")
            if meshdir:
                global_mesh_base = (main_dir / meshdir).resolve()
            if texturedir:
                global_texture_base = (main_dir / texturedir).resolve()

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except Exception:
            _LOGGER.exception("Failed parsing MJCF asset dependencies: %s", xml_path)
            continue

        xml_root = tree.getroot()
        mesh_base = global_mesh_base
        texture_base = global_texture_base
        asset_base = main_dir

        for element in xml_root.findall(".//*[@file]"):
            file_attr = element.get("file")
            if not file_attr:
                continue

            base_dir = asset_base
            if element.tag == "mesh":
                base_dir = mesh_base
            elif element.tag == "texture":
                base_dir = texture_base

            asset_path = (base_dir / file_attr).resolve()
            if not asset_path.exists():
                _LOGGER.warning(
                    "MJCF asset path not found while bundling: %s (from %s)",
                    file_attr,
                    xml_path,
                )
                continue
            if asset_path in seen:
                continue
            seen.add(asset_path)
            ordered.append(asset_path)

    return ordered


def _copy_mjcf_xml_with_shared_meshdir(
    source_xml: Path,
    target_xml: Path,
    *,
    main_xml_dir: Path | None = None,
    mjcf_root: Path | None = None,
) -> Path | None:
    target_xml.parent.mkdir(parents=True, exist_ok=True)
    if not _xml_may_define_compiler_paths(source_xml):
        _copy_file_fast(source_xml, target_xml)
        return None

    tree = ET.parse(source_xml)
    xml_root = tree.getroot()

    # MuJoCo resolves meshdir relative to the main XML, not includes.
    resolve_base = main_xml_dir if main_xml_dir is not None else source_xml.parent

    shared_meshdir: Path | None = None
    compiler = xml_root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir:
            meshdir_path = (resolve_base / meshdir).resolve()
            if meshdir_path.exists():
                target_meshdir = (
                    mjcf_root / _source_relative_path(meshdir_path)
                    if mjcf_root is not None
                    else meshdir_path
                )
                rel_meshdir = os.path.relpath(target_meshdir, start=target_xml.parent)
                compiler.set("meshdir", rel_meshdir)
                shared_meshdir = meshdir_path

    tree.write(target_xml, encoding="utf-8", xml_declaration=True)
    return shared_meshdir


def _xml_may_define_compiler_paths(source_xml: Path) -> bool:
    try:
        with source_xml.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if b"<compiler" in chunk:
                    return True
    except OSError:
        return True
    return False


def _copy_file_fast(source: str | Path, target: str | Path) -> None:
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            import fcntl

            with source.open("rb") as source_handle, target.open("wb") as target_handle:
                fcntl.ioctl(
                    target_handle.fileno(),
                    _LINUX_FICLONE,
                    source_handle.fileno(),
                )
            shutil.copystat(source, target, follow_symlinks=True)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _save_training_repro_bundle(cfg: Any, log_dir: Path, *, resolved_seed: int) -> None:
    """Save the run's config, seed and MJCF XMLs (sharing mesh assets).

    The bundle is the run's single config record: playback and checkpoint
    consumers read it, so a failure here must stop training.
    """
    bundle_root = run_bundle_root(log_dir)
    config_dir = (bundle_root / RUN_BUNDLE_CONFIG).parent
    mjcf_root = bundle_root / "mjcf_tree"

    source_mjcf = _resolve_source_path(str(cfg.env.sim.mjcf_path))
    if not source_mjcf.exists():
        raise FileNotFoundError(f"Source MJCF path does not exist: {source_mjcf}")
    config_dir.mkdir(parents=True, exist_ok=True)
    mjcf_root.mkdir(parents=True, exist_ok=True)

    snapshot_values = (
        cfg.to_dict()
        if isinstance(cfg, SemanticConfig)
        else OmegaConf.to_container(cfg, resolve=True)
    )
    snapshot_cfg = OmegaConf.create(snapshot_values)
    model_sources = {"env.sim.mjcf_path": source_mjcf}
    auxiliary_key = "env.action.policy_output.stable_pd.mjcf_path"
    auxiliary_path = OmegaConf.select(snapshot_cfg, auxiliary_key)
    if auxiliary_path is not None:
        model_sources[auxiliary_key] = _resolve_source_path(str(auxiliary_path))
    bundled_xml_files: list[str] = []
    shared_meshdirs: set[str] = set()
    bundled_models: dict[str, Path] = {}
    for role, model_source in model_sources.items():
        source_xml_files, source_asset_files = _collect_mjcf_bundle_dependencies(
            model_source
        )
        for source_xml in source_xml_files:
            target_xml = mjcf_root / _source_relative_path(source_xml)
            shared_meshdir = _copy_mjcf_xml_with_shared_meshdir(
                source_xml,
                target_xml,
                main_xml_dir=model_source.parent,
                mjcf_root=mjcf_root,
            )
            relative_xml = str(_source_relative_path(target_xml))
            if relative_xml not in bundled_xml_files:
                bundled_xml_files.append(relative_xml)
            if shared_meshdir is not None:
                shared_meshdirs.add(str(_source_relative_path(shared_meshdir)))
            if source_xml == model_source:
                bundled_models[role] = target_xml
        for source_asset in source_asset_files:
            target_asset = mjcf_root / _source_relative_path(source_asset)
            _copy_file_fast(source_asset, target_asset)
    with open_dict(snapshot_cfg):
        for role, path in bundled_models.items():
            OmegaConf.update(
                snapshot_cfg, role, str(path.resolve().relative_to(log_dir.resolve()))
            )
    bundled_mjcf = bundled_models.get("env.sim.mjcf_path")

    config_path = bundle_root / RUN_BUNDLE_CONFIG
    OmegaConf.save(snapshot_cfg, config_path, resolve=True)

    task_overrides: list[str] = []
    if HydraConfig.initialized():
        task_overrides = list(HydraConfig.get().overrides.task)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "resolved_seed": int(resolved_seed),
        "config_path": str(_source_relative_path(config_path)),
        "source_mjcf_path": str(_source_relative_path(source_mjcf)),
        "bundled_mjcf_path": (
            str(_source_relative_path(bundled_mjcf))
            if bundled_mjcf is not None
            else None
        ),
        "bundled_model_paths": {
            role: str(path.resolve().relative_to(log_dir.resolve()))
            for role, path in bundled_models.items()
        },
        "bundled_xml_files": bundled_xml_files,
        "shared_meshdirs": sorted(shared_meshdirs),
        "task_overrides": task_overrides,
    }
    (bundle_root / RUN_BUNDLE_MANIFEST).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
