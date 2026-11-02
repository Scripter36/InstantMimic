"""Utilities for reading muscle definitions directly from MJCF."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import os
import pickle
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from instantmimic.utils.project_paths import resolve_cache_path
from instantmimic.utils.startup_profiler import startup_nvtx_range, startup_nvtx_wrapped

from .source_xml import (
    MuscleParser,
    MuscleUnitInfo,
    WaypointAttachmentInfo,
    WaypointInfo,
)

# The cache key includes this version, so entries written with older packed
# dataclass schemas miss and are regenerated instead of being deserialized.
_CACHE_SCHEMA_VERSION = 20

_CACHE_ALLOWED_GLOBALS = frozenset(
    {
        ("collections", "OrderedDict"),
        ("instantmimic.muscle.parsing.mjcf", "PackedMjcfMuscleData"),
        ("instantmimic.muscle.parsing.mjcf", "ParsedWaypointWraps"),
        ("numpy", "dtype"),
        ("numpy._core.numeric", "_frombuffer"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch.storage", "_load_from_bytes"),
    }
)


def _safe_torch_load_from_bytes(serialized: bytes) -> Any:
    """Load a tensor cache payload without enabling arbitrary pickle globals."""
    import torch

    try:
        return torch.load(
            io.BytesIO(serialized),
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, pickle.PickleError, RuntimeError, TypeError, ValueError) as exc:
        raise pickle.UnpicklingError("invalid tensor cache payload") from exc


class _CacheUnpickler(pickle.Unpickler):
    """Unpickler for the small set of values produced by local cache writers."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _CACHE_ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"disallowed cache global: {module}.{name}")
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _safe_torch_load_from_bytes
        return super().find_class(module, name)


_DEFAULT_CACHE_DIR = Path("mjcf_muscle_cache")
_MUJOCO_DEFAULT_MUSCLE_RANGE = (0.75, 1.05)


@dataclass(frozen=True)
class ParsedWaypointWraps:
    """Compiled fixed-radius wraps; tangent slots follow each start point."""

    body_names: list[str]
    side_body_names: list[str | None]
    kind: np.ndarray
    center: np.ndarray
    quaternion: np.ndarray
    radius: np.ndarray
    side_local: np.ndarray
    start_point: np.ndarray


@dataclass
class PackedMjcfMuscleData:
    """Array-oriented muscle metadata used for fast startup/cache restore."""

    muscle_names: list[str]
    muscle_params: np.ndarray
    waypoint_start: np.ndarray
    waypoint_body_names: list[str]
    waypoint_positions: np.ndarray
    attachment_start: np.ndarray
    attachment_body_names: list[str]
    attachment_positions: np.ndarray
    attachment_weights: np.ndarray
    rest_lengths: np.ndarray
    uses_lbs_waypoint_attachments: bool
    mujoco_l0: np.ndarray | None = None
    wraps: ParsedWaypointWraps | None = None


class _LazyPackedMuscleSequence(Sequence[MuscleUnitInfo]):
    def __init__(self, packed: PackedMjcfMuscleData) -> None:
        self._packed = packed
        self._muscles: list[MuscleUnitInfo] | None = None

    def _materialize(self) -> list[MuscleUnitInfo]:
        muscles = self._muscles
        if muscles is None:
            muscles = _unpack_muscle_data(self._packed)
            self._muscles = muscles
        return muscles

    def __len__(self) -> int:
        return len(self._packed.muscle_names)

    def __iter__(self):
        return iter(self._materialize())

    def __getitem__(self, index):
        return self._materialize()[index]


@dataclass
class ParsedMjcfMuscleData:
    """Muscle definitions parsed from MJCF plus qpos0 tendon lengths."""

    muscles: Sequence[MuscleUnitInfo]
    rest_lengths: np.ndarray
    packed: PackedMjcfMuscleData | None = None
    mujoco_l0: np.ndarray | None = None
    wraps: ParsedWaypointWraps | None = None


def _pack_muscle_data(data: ParsedMjcfMuscleData) -> PackedMjcfMuscleData:
    if data.packed is not None:
        return data.packed

    muscle_names: list[str] = []
    muscle_params: list[list[float]] = []
    waypoint_start: list[int] = [0]
    waypoint_body_names: list[str] = []
    waypoint_positions: list[np.ndarray] = []
    attachment_start: list[int] = [0]
    attachment_body_names: list[str] = []
    attachment_positions: list[np.ndarray] = []
    attachment_weights: list[float] = []
    uses_lbs_waypoint_attachments = False

    for muscle in data.muscles:
        muscle_names.append(str(muscle.name))
        muscle_params.append(
            [
                float(muscle.f0),
                float(muscle.lm),
                float(muscle.lt),
                float(muscle.pen_angle),
                float(muscle.lmax),
            ]
        )
        for waypoint in muscle.waypoints:
            waypoint_body = str(waypoint.body)
            waypoint_pos = np.asarray(waypoint.position, dtype=np.float32).reshape(3)
            waypoint_body_names.append(waypoint_body)
            waypoint_positions.append(waypoint_pos)

            attachments = list(getattr(waypoint, "attachments", []) or [])
            if not attachments:
                attachments = [waypoint]
            if len(attachments) != 1:
                uses_lbs_waypoint_attachments = True
            for attachment in attachments:
                attachment_body = str(attachment.body)
                attachment_position = np.asarray(
                    attachment.position, dtype=np.float32
                ).reshape(3)
                attachment_weight = float(getattr(attachment, "weight", 1.0))
                if (
                    attachment_body != waypoint_body
                    or abs(attachment_weight - 1.0) > 1.0e-6
                    or not np.allclose(
                        attachment_position,
                        waypoint_pos,
                        atol=1.0e-6,
                        rtol=1.0e-6,
                    )
                ):
                    uses_lbs_waypoint_attachments = True
                attachment_body_names.append(attachment_body)
                attachment_positions.append(attachment_position)
                attachment_weights.append(attachment_weight)
            attachment_start.append(len(attachment_body_names))
        waypoint_start.append(len(waypoint_body_names))

    packed = PackedMjcfMuscleData(
        muscle_names=muscle_names,
        muscle_params=np.asarray(muscle_params, dtype=np.float32).reshape(-1, 5),
        waypoint_start=np.asarray(waypoint_start, dtype=np.int32),
        waypoint_body_names=waypoint_body_names,
        waypoint_positions=np.asarray(waypoint_positions, dtype=np.float32).reshape(
            -1, 3
        ),
        attachment_start=np.asarray(attachment_start, dtype=np.int32),
        attachment_body_names=attachment_body_names,
        attachment_positions=np.asarray(attachment_positions, dtype=np.float32).reshape(
            -1, 3
        ),
        attachment_weights=np.asarray(attachment_weights, dtype=np.float32),
        rest_lengths=np.asarray(data.rest_lengths, dtype=np.float32),
        uses_lbs_waypoint_attachments=bool(uses_lbs_waypoint_attachments),
        wraps=data.wraps,
        mujoco_l0=(
            np.asarray(data.mujoco_l0, dtype=np.float32)
            if data.mujoco_l0 is not None
            else None
        ),
    )
    data.packed = packed
    return packed


def _unpack_muscle_data(packed: PackedMjcfMuscleData) -> list[MuscleUnitInfo]:
    muscles: list[MuscleUnitInfo] = []
    params = np.asarray(packed.muscle_params, dtype=np.float32).reshape(-1, 5)
    waypoint_start = np.asarray(packed.waypoint_start, dtype=np.int32).reshape(-1)
    attachment_start = np.asarray(packed.attachment_start, dtype=np.int32).reshape(-1)
    waypoint_positions = np.asarray(
        packed.waypoint_positions, dtype=np.float32
    ).reshape(-1, 3)
    attachment_positions = np.asarray(
        packed.attachment_positions, dtype=np.float32
    ).reshape(-1, 3)
    attachment_weights = np.asarray(
        packed.attachment_weights, dtype=np.float32
    ).reshape(-1)

    for muscle_idx, muscle_name in enumerate(packed.muscle_names):
        w_begin = int(waypoint_start[muscle_idx])
        w_end = int(waypoint_start[muscle_idx + 1])
        waypoints: list[WaypointInfo] = []
        for waypoint_idx in range(w_begin, w_end):
            a_begin = int(attachment_start[waypoint_idx])
            a_end = int(attachment_start[waypoint_idx + 1])
            attachments = [
                WaypointAttachmentInfo(
                    body=packed.attachment_body_names[attachment_idx],
                    position=attachment_positions[attachment_idx].astype(
                        np.float64,
                        copy=True,
                    ),
                    weight=float(attachment_weights[attachment_idx]),
                )
                for attachment_idx in range(a_begin, a_end)
            ]
            waypoints.append(
                WaypointInfo(
                    body=packed.waypoint_body_names[waypoint_idx],
                    position=waypoint_positions[waypoint_idx].astype(
                        np.float64,
                        copy=True,
                    ),
                    attachments=attachments,
                )
            )
        f0, lm, lt, pen_angle, lmax = params[muscle_idx].tolist()
        muscles.append(
            MuscleUnitInfo(
                name=muscle_name,
                f0=float(f0),
                lm=float(lm),
                lt=float(lt),
                pen_angle=float(pen_angle),
                lmax=float(lmax),
                waypoints=waypoints,
            )
        )
    return muscles


@dataclass
class FastMuscleBootstrap:
    """Bundle the source MJCF, parsed muscle payload, and stripped sim spec."""

    source_mjcf_path: str
    parsed_mjcf_data: ParsedMjcfMuscleData
    sim_spec: mujoco.MjSpec
    geometry_backend: str

    @classmethod
    @startup_nvtx_wrapped("mjcf_muscle:fast_muscle_bootstrap_from_mjcf")
    def from_mjcf(
        cls,
        source_mjcf_path: str,
        *,
        position_actuator_mjcf_path: str | None = None,
        geometry_backend: str = "waypoint",
        rest_keyframe_name: str | None = None,
        muscle_metadata_path: str | None = None,
    ) -> FastMuscleBootstrap:
        requested_backend = str(geometry_backend).strip().lower()
        if requested_backend != "waypoint":
            raise ValueError("geometry_backend must be 'waypoint'.")
        with startup_nvtx_range("mjcf_muscle:validate_source_model"):
            compiled_model = _validated_waypoint_model(source_mjcf_path)
        if muscle_metadata_path:
            _require_metadata_site_paths(compiled_model)
        with startup_nvtx_range("mjcf_muscle:parse_muscle_payload"):
            parsed_mjcf_data = (
                parse_muscles_from_metadata_xml(
                    muscle_metadata_path,
                    source_mjcf_path=source_mjcf_path,
                    compiled_model=compiled_model,
                )
                if muscle_metadata_path
                else parse_muscles_from_mjcf(
                    source_mjcf_path,
                    rest_keyframe_name=rest_keyframe_name,
                    compiled_model=compiled_model,
                )
            )
        with startup_nvtx_range("mjcf_muscle:build_stripped_sim_spec"):
            sim_spec = build_stripped_mjcf_spec(
                source_mjcf_path,
                position_actuator_mjcf_path=position_actuator_mjcf_path,
            )
        return cls(
            source_mjcf_path=str(Path(source_mjcf_path).expanduser().resolve()),
            parsed_mjcf_data=parsed_mjcf_data,
            sim_spec=sim_spec,
            geometry_backend="waypoint",
        )


def mjcf_cache_dir() -> Path | None:
    cache_dir = resolve_cache_path(_DEFAULT_CACHE_DIR)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return cache_dir


def file_signature(path: Path) -> tuple[str, int, int]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def _xml_dependency_signatures_are_current(
    signatures: tuple[tuple[str, int, int], ...],
) -> bool:
    for path_str, size, mtime_ns in signatures:
        try:
            if file_signature(Path(path_str)) != (
                str(path_str),
                int(size),
                int(mtime_ns),
            ):
                return False
        except (OSError, TypeError, ValueError):
            return False
    return True


def _collect_xml_dependency_signatures_uncached(
    xml_path: Path,
    *,
    _seen: set[Path] | None = None,
    _source_root: Path | None = None,
) -> tuple[tuple[str, int, int], ...]:
    resolved = xml_path.expanduser().resolve()
    source_root = resolved.parent if _source_root is None else _source_root
    seen = set() if _seen is None else _seen
    if resolved in seen:
        return ()
    seen.add(resolved)

    signatures = [file_signature(resolved)]
    root = ET.parse(resolved).getroot()
    for include in root.iter("include"):
        file_attr = include.get("file")
        if not file_attr:
            continue
        include_path = _resolve_include_path(
            file_attr,
            include_dir=resolved.parent,
            source_root=source_root,
        )
        signatures.extend(
            _collect_xml_dependency_signatures_uncached(
                include_path,
                _seen=seen,
                _source_root=source_root,
            )
        )
    return tuple(signatures)


def collect_xml_dependency_signatures(
    xml_path: Path,
) -> tuple[tuple[str, int, int], ...]:
    resolved = xml_path.expanduser().resolve()
    key = cache_key("xml_dependency_signatures", {"root": file_signature(resolved)})
    cached = load_pickle_cache("xml_dependency_signatures", key)
    if isinstance(cached, tuple) and _xml_dependency_signatures_are_current(cached):
        return cached

    signatures = _collect_xml_dependency_signatures_uncached(resolved)
    store_pickle_cache("xml_dependency_signatures", key, signatures)
    return signatures


def cache_key(kind: str, payload: dict[str, Any]) -> str:
    serialized = repr(
        {
            "schema": _CACHE_SCHEMA_VERSION,
            "kind": kind,
            "payload": payload,
        }
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_pickle_cache(kind: str, key: str) -> Any | None:
    """Load a schema-versioned machine-local cache entry.

    Cache entries use a restricted pickle loader and must still be confined to
    the machine-local cache directory. External or downloaded artifacts must
    not be copied into this directory.
    """
    cache_dir = mjcf_cache_dir()
    if cache_dir is None:
        return None
    cache_path = cache_dir / f"{kind}-{key}.pkl"
    try:
        with cache_path.open("rb") as handle:
            payload = _CacheUnpickler(handle).load()
    except (
        OSError,
        EOFError,
        pickle.PickleError,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
        RuntimeError,
    ):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _CACHE_SCHEMA_VERSION
        or payload.get("kind") != kind
    ):
        return None
    return payload.get("value")


def store_pickle_cache(kind: str, key: str, value: Any) -> None:
    cache_dir = mjcf_cache_dir()
    if cache_dir is None:
        return
    cache_path = cache_dir / f"{kind}-{key}.pkl"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=cache_dir,
            prefix=f".{kind}-{key}-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            pickle.dump(
                {
                    "schema": _CACHE_SCHEMA_VERSION,
                    "kind": kind,
                    "value": value,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(tmp_path, cache_path)
    except OSError:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)


def _metadata_parse_cache_key(
    muscle_metadata_path: str,
    *,
    source_mjcf_path: str,
) -> str:
    metadata_path = Path(muscle_metadata_path).expanduser().resolve()
    source_path = Path(source_mjcf_path).expanduser().resolve()
    payload = {
        "metadata": file_signature(metadata_path),
        "source_xml": collect_xml_dependency_signatures(source_path),
    }
    return cache_key("metadata_parse", payload)


def _stripped_mjcf_cache_key(
    source_mjcf_path: str,
    *,
    preserve_position_actuators: bool,
    position_actuator_mjcf_path: str | None,
) -> str:
    source_path = Path(source_mjcf_path).expanduser().resolve()
    payload = {
        "source_xml": collect_xml_dependency_signatures(source_path),
        "preserve_position_actuators": bool(preserve_position_actuators),
        "position_actuator_xml": (
            collect_xml_dependency_signatures(
                Path(position_actuator_mjcf_path).expanduser().resolve()
            )
            if position_actuator_mjcf_path
            else None
        ),
    }
    return cache_key("stripped_mjcf", payload)


def _resolve_include_path(
    file_attr: str,
    *,
    include_dir: Path,
    source_root: Path,
) -> Path:
    raw_path = Path(file_attr).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    include_relative = (include_dir / raw_path).resolve()
    if include_relative.exists():
        return include_relative
    root_relative = (source_root / raw_path).resolve()
    if root_relative.exists():
        return root_relative
    return include_relative


def _expand_includes_in_place(
    element: ET.Element,
    base_dir: Path,
    *,
    source_root: Path,
) -> None:
    for child in list(element):
        if child.tag == "include":
            file_attr = child.get("file")
            if not file_attr:
                raise ValueError("Encountered <include> without a file attribute.")
            include_path = _resolve_include_path(
                file_attr,
                include_dir=base_dir,
                source_root=source_root,
            )
            include_root = ET.parse(include_path).getroot()
            _absolutize_relative_paths_in_place(include_root, include_path.parent)
            _expand_includes_in_place(
                include_root,
                include_path.parent,
                source_root=source_root,
            )

            insert_idx = list(element).index(child)
            element.remove(child)
            for include_child in list(include_root):
                element.insert(insert_idx, copy.deepcopy(include_child))
                insert_idx += 1
            continue

        if isinstance(child.tag, str):
            _expand_includes_in_place(child, base_dir, source_root=source_root)


def load_expanded_mjcf_root(source_mjcf_path: str) -> tuple[ET.Element, Path]:
    source_path = Path(source_mjcf_path).expanduser().resolve()
    root = ET.parse(source_path).getroot()
    _absolutize_relative_paths_in_place(root, source_path.parent)
    _expand_includes_in_place(
        root,
        source_path.parent,
        source_root=source_path.parent,
    )
    # Expanded includes can add compiler declarations before the main one.
    # Preserve document-order attribute precedence in a single declaration so
    # subsequent asset relocation sees every effective directory setting.
    compilers = root.findall("compiler")
    if len(compilers) > 1:
        effective = {}
        for compiler in compilers:
            effective.update(compiler.attrib)
            root.remove(compiler)
        root.insert(0, ET.Element("compiler", effective))
    return root, source_path


def _resolve_compiler_dir(base_dir: Path, value: str) -> Path:
    raw_path = Path(value).expanduser()
    resolved = (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (base_dir / raw_path).resolve()
    )
    if resolved.exists():
        return resolved

    local_sibling = (base_dir / raw_path.name).resolve()
    if local_sibling.exists():
        return local_sibling
    return resolved


def absolutize_compiler_dirs(root: ET.Element, source_path: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        return

    meshdir = compiler.get("meshdir")
    texturedir = compiler.get("texturedir")
    if meshdir:
        compiler.set("meshdir", str(_resolve_compiler_dir(source_path.parent, meshdir)))
    if texturedir:
        compiler.set(
            "texturedir",
            str(_resolve_compiler_dir(source_path.parent, texturedir)),
        )


def _absolutize_relative_paths_in_place(root: ET.Element, base_dir: Path) -> None:
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        texturedir = compiler.get("texturedir")
        if meshdir:
            compiler.set("meshdir", str(_resolve_compiler_dir(base_dir, meshdir)))
        if texturedir:
            compiler.set("texturedir", str(_resolve_compiler_dir(base_dir, texturedir)))


def _absolutize_asset_files_without_compiler_dirs(
    root: ET.Element,
    source_path: Path,
) -> None:
    compiler = root.find("compiler")
    has_meshdir = bool(compiler is not None and compiler.get("meshdir"))
    has_texturedir = bool(compiler is not None and compiler.get("texturedir"))

    for asset_elem in root.findall(".//asset/*"):
        file_attr = asset_elem.get("file")
        if not file_attr:
            continue
        if asset_elem.tag == "mesh" and has_meshdir:
            continue
        if asset_elem.tag == "texture" and has_texturedir:
            continue
        file_path = Path(file_attr)
        if file_path.is_absolute():
            continue
        asset_elem.set("file", str((source_path.parent / file_path).resolve()))


def absolutize_mjcf_asset_paths(root: ET.Element, source_path: Path) -> None:
    """Preserve source-relative MJCF assets before compiling an XML string."""
    absolutize_compiler_dirs(root, source_path)
    _absolutize_asset_files_without_compiler_dirs(root, source_path)


def remove_all_actuator_elements(element: ET.Element) -> None:
    for child in list(element):
        if child.tag == "actuator":
            element.remove(child)
            continue
        if isinstance(child.tag, str):
            remove_all_actuator_elements(child)


def preserve_only_position_actuator_elements(root: ET.Element) -> None:
    for actuator in root.findall(".//actuator"):
        for child in list(actuator):
            if child.tag != "position":
                actuator.remove(child)


def graft_position_actuators(
    target_root: ET.Element,
    actuator_source_root: ET.Element,
) -> None:
    """Add position actuators without replacing the target model's dynamics."""
    position_elements = [
        copy.deepcopy(element)
        for actuator in actuator_source_root.findall(".//actuator")
        for element in actuator
        if element.tag == "position"
    ]
    if not position_elements:
        raise ValueError("Position actuator MJCF contains no <position> actuators.")

    remove_all_actuator_elements(target_root)

    referenced_classes = {
        class_name
        for element in position_elements
        if (class_name := element.get("class"))
    }
    if referenced_classes:
        target_default = target_root.find("default")
        if target_default is None:
            target_default = ET.Element("default")
            target_root.insert(0, target_default)
        existing_classes = {
            class_name
            for default in target_root.findall(".//default")
            if (class_name := default.get("class"))
        }
        source_defaults = {
            str(default.get("class")): default
            for default in actuator_source_root.findall(".//default")
            if default.get("class")
        }
        for class_name in sorted(referenced_classes):
            if class_name in existing_classes:
                continue
            source_default = source_defaults.get(class_name)
            if source_default is None:
                raise ValueError(
                    f"Position actuator default class {class_name!r} is missing."
                )
            target_default.append(copy.deepcopy(source_default))

    actuator_block = ET.SubElement(target_root, "actuator")
    actuator_block.extend(position_elements)


def remove_all_tendon_elements(element: ET.Element) -> None:
    for child in list(element):
        if child.tag == "tendon":
            element.remove(child)
            continue
        if isinstance(child.tag, str):
            remove_all_tendon_elements(child)


def collect_referenced_site_names(root: ET.Element) -> set[str]:
    named_sites = {
        str(name)
        for site_elem in root.findall(".//site")
        if (name := site_elem.get("name"))
    }
    if not named_sites:
        return set()

    referenced: set[str] = set()
    for elem in root.iter():
        for attr_name, attr_value in elem.attrib.items():
            if elem.tag == "site" and attr_name == "name":
                continue
            if attr_value in named_sites:
                referenced.add(str(attr_value))
    return referenced


def remove_unreferenced_named_sites(
    element: ET.Element,
    *,
    referenced_site_names: set[str],
) -> None:
    for child in list(element):
        if child.tag == "site":
            site_name = child.get("name")
            if site_name and site_name not in referenced_site_names:
                element.remove(child)
                continue
        if isinstance(child.tag, str):
            remove_unreferenced_named_sites(
                child,
                referenced_site_names=referenced_site_names,
            )


def _collect_spatial_waypoint_site_names(root: ET.Element) -> dict[str, list[str]]:
    spatial_by_name: dict[str, list[str]] = {}
    for tendon_elem in root.findall(".//tendon"):
        for spatial in tendon_elem.findall("spatial"):
            tendon_name = spatial.get("name")
            if not tendon_name:
                continue

            ordered_sites: list[str] = []
            for child in list(spatial):
                if not isinstance(child.tag, str):
                    continue
                if child.tag == "site":
                    site_name = child.get("site")
                    if not site_name:
                        raise ValueError(
                            f"Spatial tendon '{tendon_name}' contains a <site> without a site attribute."
                        )
                    ordered_sites.append(str(site_name))
                    continue
                raise ValueError(
                    "Fast-muscle paths require body-fixed site polylines; "
                    f"tendon '{tendon_name}' uses unsupported child tag '{child.tag}'."
                )

            spatial_by_name[str(tendon_name)] = ordered_sites
    return spatial_by_name


def _derive_f0(
    model: mujoco.MjModel, actuator_id: int, actuator_elem: ET.Element
) -> float:
    gain_force = float(model.actuator_gainprm[actuator_id, 2])
    if abs(gain_force) > 1.0e-8:
        return gain_force

    force_attr = actuator_elem.get("force")
    if force_attr:
        return float(force_attr)

    actuator_name = actuator_elem.get("name", f"actuator[{actuator_id}]")
    raise ValueError(f"Unable to derive f0 for actuator '{actuator_name}'.")


def _parse_vec_attr(
    elem: ET.Element, name: str, *, default: tuple[float, ...]
) -> np.ndarray:
    value = elem.get(name)
    if value is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in value.split()], dtype=np.float64)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    qvec = np.asarray([x, y, z], dtype=np.float64)
    return vec + 2.0 * np.cross(qvec, np.cross(qvec, vec) + w * vec)


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


@dataclass
class _XmlSiteInfo:
    body: str
    local_pos: np.ndarray
    world_pos: np.ndarray


@dataclass
class _XmlBodyInfo:
    world_pos: np.ndarray
    world_quat: np.ndarray
    world_rot: np.ndarray


@dataclass
class _XmlKinematicInfo:
    sites: dict[str, _XmlSiteInfo]
    bodies: dict[str, _XmlBodyInfo]


@startup_nvtx_wrapped("mjcf_muscle:collect_xml_kinematic_infos")
def _collect_xml_kinematic_infos(
    root: ET.Element,
) -> _XmlKinematicInfo | None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return _XmlKinematicInfo(sites={}, bodies={})

    site_infos: dict[str, _XmlSiteInfo] = {}
    body_infos: dict[str, _XmlBodyInfo] = {}

    def visit_body(
        body: ET.Element,
        parent_pos: np.ndarray,
        parent_quat: np.ndarray,
    ) -> bool:
        body_name = body.get("name")
        if not body_name:
            return False
        if any(
            attr in body.attrib for attr in ("euler", "axisangle", "xyaxes", "zaxis")
        ):
            return False

        local_pos = _parse_vec_attr(body, "pos", default=(0.0, 0.0, 0.0))
        local_quat = _parse_vec_attr(body, "quat", default=(1.0, 0.0, 0.0, 0.0))
        if local_pos.shape != (3,) or local_quat.shape != (4,):
            return False
        local_quat = local_quat / max(float(np.linalg.norm(local_quat)), 1.0e-12)

        world_pos = parent_pos + _quat_rotate_wxyz(parent_quat, local_pos)
        world_quat = _quat_mul_wxyz(parent_quat, local_quat)
        world_quat = world_quat / max(float(np.linalg.norm(world_quat)), 1.0e-12)
        world_rot = _quat_to_matrix_wxyz(world_quat)
        body_name_str = str(body_name)
        body_infos[body_name_str] = _XmlBodyInfo(
            world_pos=world_pos,
            world_quat=world_quat,
            world_rot=world_rot,
        )

        for site in body.findall("site"):
            site_name = site.get("name")
            if not site_name:
                continue
            if any(
                attr in site.attrib
                for attr in ("euler", "axisangle", "xyaxes", "zaxis")
            ):
                return False
            site_pos = _parse_vec_attr(site, "pos", default=(0.0, 0.0, 0.0))
            if site_pos.shape != (3,):
                return False
            site_infos[str(site_name)] = _XmlSiteInfo(
                body=body_name_str,
                local_pos=site_pos,
                world_pos=world_pos + world_rot @ site_pos,
            )

        for child in body.findall("body"):
            if not visit_body(child, world_pos, world_quat):
                return False
        return True

    identity_pos = np.zeros(3, dtype=np.float64)
    identity_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for body in worldbody.findall("body"):
        if not visit_body(body, identity_pos, identity_quat):
            return None
    return _XmlKinematicInfo(
        sites=site_infos,
        bodies=body_infos,
    )


def _single_body_attachment(site_info: _XmlSiteInfo) -> list[WaypointAttachmentInfo]:
    return [
        WaypointAttachmentInfo(
            body=site_info.body,
            position=site_info.local_pos.copy(),
            weight=1.0,
        )
    ]


def _world_point_to_body_local(
    kinematic: _XmlKinematicInfo,
    body_name: str,
    world_pos: np.ndarray,
) -> np.ndarray | None:
    body_info = kinematic.bodies.get(body_name)
    if body_info is None:
        return None
    return body_info.world_rot.T @ (world_pos - body_info.world_pos)


def _adapt_metadata_muscles_to_mjcf_frame(
    muscles: list[MuscleUnitInfo],
    kinematic: _XmlKinematicInfo,
) -> ParsedMjcfMuscleData:
    adapted_muscles: list[MuscleUnitInfo] = []
    rest_lengths: list[float] = []

    for muscle in muscles:
        adapted_waypoints: list[WaypointInfo] = []
        global_points: list[np.ndarray] = []
        for waypoint in muscle.waypoints:
            world_pos = np.asarray(waypoint.position, dtype=np.float64).reshape(3)
            primary_local = _world_point_to_body_local(
                kinematic,
                str(waypoint.body),
                world_pos,
            )
            if primary_local is None:
                raise ValueError(
                    f"Waypoint body '{waypoint.body}' from muscle metadata was "
                    "not found in the source MJCF."
                )

            attachments = [
                WaypointAttachmentInfo(
                    body=str(waypoint.body),
                    position=primary_local.copy(),
                    weight=1.0,
                )
            ]
            body_info = kinematic.bodies[str(waypoint.body)]
            global_points.append(
                body_info.world_pos + body_info.world_rot @ primary_local
            )
            adapted_waypoints.append(
                WaypointInfo(
                    body=str(waypoint.body),
                    position=primary_local,
                    attachments=attachments,
                )
            )

        if len(global_points) < 2:
            continue
        rest_length = float(
            sum(
                np.linalg.norm(global_points[idx + 1] - global_points[idx])
                for idx in range(len(global_points) - 1)
            )
        )
        if not np.isfinite(rest_length) or rest_length <= 0.0:
            raise ValueError(
                f"Muscle '{muscle.name}' from metadata has invalid rest length "
                f"{rest_length!r}."
            )
        adapted_muscles.append(
            MuscleUnitInfo(
                name=muscle.name,
                f0=muscle.f0,
                lm=muscle.lm,
                lt=muscle.lt,
                pen_angle=muscle.pen_angle,
                lmax=muscle.lmax,
                waypoints=adapted_waypoints,
            )
        )
        rest_lengths.append(rest_length)

    return ParsedMjcfMuscleData(
        muscles=adapted_muscles,
        rest_lengths=np.asarray(rest_lengths, dtype=np.float32),
    )


@startup_nvtx_wrapped("mjcf_muscle:parse_muscles_from_metadata_xml")
def parse_muscles_from_metadata_xml(
    muscle_metadata_path: str,
    *,
    source_mjcf_path: str,
    compiled_model: mujoco.MjModel | None = None,
) -> ParsedMjcfMuscleData:
    """Map fixed sidecar muscle points into the source MJCF body frames."""
    source_model = _validated_waypoint_model(
        source_mjcf_path, compiled_model=compiled_model
    )
    _require_metadata_site_paths(source_model)
    with startup_nvtx_range("mjcf_muscle:metadata_load_source_root"):
        root, _ = load_expanded_mjcf_root(source_mjcf_path)
        _collect_spatial_waypoint_site_names(root)
    with startup_nvtx_range("mjcf_muscle:metadata_cache_lookup"):
        cache_key = _metadata_parse_cache_key(
            muscle_metadata_path,
            source_mjcf_path=source_mjcf_path,
        )
        cached = load_pickle_cache("metadata_parse", cache_key)
    if isinstance(cached, PackedMjcfMuscleData):
        return ParsedMjcfMuscleData(
            muscles=_LazyPackedMuscleSequence(cached),
            rest_lengths=cached.rest_lengths,
            packed=cached,
            wraps=cached.wraps,
            mujoco_l0=cached.mujoco_l0,
        )
    if isinstance(cached, ParsedMjcfMuscleData):
        return cached

    with startup_nvtx_range("mjcf_muscle:metadata_collect_kinematic"):
        kinematic = _collect_xml_kinematic_infos(root)
    if kinematic is None:
        raise ValueError(
            "muscle_metadata_path requires source MJCF body transforms to be "
            "parsable without compiling the full model."
        )

    metadata_path = Path(muscle_metadata_path).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = metadata_path.resolve()
    with startup_nvtx_range("mjcf_muscle:metadata_parse_xml"):
        muscles = MuscleParser(str(metadata_path)).parse()
    if not muscles:
        raise ValueError(f"No muscles were parsed from metadata XML: {metadata_path}")
    with startup_nvtx_range("mjcf_muscle:metadata_adapt_to_mjcf_frame"):
        parsed = _adapt_metadata_muscles_to_mjcf_frame(muscles, kinematic)
    packed = _pack_muscle_data(parsed)
    with startup_nvtx_range("mjcf_muscle:metadata_cache_write"):
        store_pickle_cache("metadata_parse", cache_key, packed)
    return parsed


def _derive_f0_from_xml(actuator_elem: ET.Element) -> float | None:
    force_attr = actuator_elem.get("force")
    if force_attr:
        return float(force_attr)

    gainprm_attr = actuator_elem.get("gainprm")
    if gainprm_attr:
        gainprm = [float(x) for x in gainprm_attr.split()]
        if len(gainprm) >= 3 and abs(gainprm[2]) > 1.0e-8:
            return float(gainprm[2])
    return None


def _derive_normalized_lengths_from_xml(
    actuator_elem: ET.Element,
) -> tuple[float, float, float, float] | None:
    user_attr = actuator_elem.get("user")
    if not user_attr:
        return None
    user = [float(x) for x in user_attr.split()]
    if len(user) < 4:
        return None
    return (float(user[0]), float(user[1]), float(user[2]), float(user[3]))


def _collect_muscle_default_attrs(root: ET.Element) -> dict[str | None, dict[str, str]]:
    """Resolve inherited MJCF ``<default><muscle ...>`` attributes by class."""
    resolved: dict[str | None, dict[str, str]] = {None: {}}

    def visit(default_elem: ET.Element, parent_attrs: dict[str, str]) -> None:
        attrs = dict(parent_attrs)
        muscle_default = default_elem.find("muscle")
        if muscle_default is not None:
            attrs.update(muscle_default.attrib)
        class_name = default_elem.get("class")
        key = str(class_name) if class_name else None
        resolved[key] = attrs
        for child in default_elem.findall("default"):
            visit(child, attrs)

    for default_elem in root.findall("./default"):
        visit(default_elem, resolved[None])
    return resolved


def _derive_mujoco_l0_from_xml(
    actuator_elem: ET.Element,
    muscle_default_attrs: dict[str | None, dict[str, str]],
) -> float | None:
    """Derive MuJoCo's muscle length scale without compiling the source model."""
    if actuator_elem.tag != "muscle":
        return None
    class_name = actuator_elem.get("class")
    attrs = dict(muscle_default_attrs.get(str(class_name) if class_name else None, {}))
    attrs.update(actuator_elem.attrib)
    lengthrange_attr = attrs.get("lengthrange")
    if not lengthrange_attr:
        return None
    lengthrange = [float(value) for value in lengthrange_attr.split()]
    range_values = [
        float(value)
        for value in attrs.get(
            "range",
            f"{_MUJOCO_DEFAULT_MUSCLE_RANGE[0]} {_MUJOCO_DEFAULT_MUSCLE_RANGE[1]}",
        ).split()
    ]
    if len(lengthrange) != 2 or len(range_values) != 2:
        return None
    physical_width = lengthrange[1] - lengthrange[0]
    normalized_width = range_values[1] - range_values[0]
    if physical_width <= 0.0 or normalized_width <= 0.0:
        return None
    l0 = physical_width / normalized_width
    return float(l0) if np.isfinite(l0) and l0 > 0.0 else None


def _validate_muscle_name_order(
    root: ET.Element,
    *,
    muscle_names: Sequence[str],
) -> None:
    """Check that authored names agree with the parsed transmission order."""

    muscle_names = tuple(str(name) for name in muscle_names)
    order_element = root.find("./custom/text[@name='instantmimic_muscle_names']")
    if order_element is not None:
        raw_order = str(order_element.get("data") or "").strip()
        try:
            authored_order = json.loads(raw_order)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "instantmimic_muscle_names must contain a JSON string list."
            ) from exc
        if not isinstance(authored_order, list) or not all(
            isinstance(name, str) and name for name in authored_order
        ):
            raise ValueError(
                "instantmimic_muscle_names must contain a JSON string list."
            )
        if tuple(authored_order) != muscle_names:
            raise ValueError(
                "Authored muscle order does not match the MJCF muscle actuator order."
            )


def _parse_muscles_from_mjcf_xml_fast(
    root: ET.Element,
    *,
    spatial_waypoints: dict[str, list[str]],
) -> ParsedMjcfMuscleData | None:
    kinematic = _collect_xml_kinematic_infos(root)
    if kinematic is None:
        return None
    site_infos = kinematic.sites

    muscles: list[MuscleUnitInfo] = []
    rest_lengths: list[float] = []
    mujoco_l0: list[float] = []
    mujoco_l0_available = True
    spatial_tendon_names = set(spatial_waypoints.keys())
    muscle_default_attrs = _collect_muscle_default_attrs(root)

    for actuator_elem in root.findall(".//actuator/*"):
        if not isinstance(actuator_elem.tag, str):
            continue
        if actuator_elem.tag not in {"muscle", "general"}:
            if actuator_elem.get("tendon"):
                return None
            continue

        name = actuator_elem.get("name")
        tendon_name = actuator_elem.get("tendon")
        if not name or not tendon_name or tendon_name not in spatial_tendon_names:
            return None

        f0 = _derive_f0_from_xml(actuator_elem)
        normalized_lengths = _derive_normalized_lengths_from_xml(actuator_elem)
        native_l0 = _derive_mujoco_l0_from_xml(
            actuator_elem,
            muscle_default_attrs,
        )
        if f0 is None or normalized_lengths is None:
            return None
        if native_l0 is None:
            mujoco_l0_available = False

        site_names = spatial_waypoints[tendon_name]
        waypoints: list[WaypointInfo] = []
        global_points: list[np.ndarray] = []
        for site_name in site_names:
            site_info = site_infos.get(site_name)
            if site_info is None:
                return None
            attachments = _single_body_attachment(site_info)
            waypoints.append(
                WaypointInfo(
                    body=site_info.body,
                    position=site_info.local_pos.copy(),
                    attachments=attachments,
                )
            )
            global_points.append(site_info.world_pos)

        if len(global_points) < 2:
            continue
        rest_length = float(
            sum(
                np.linalg.norm(global_points[idx + 1] - global_points[idx])
                for idx in range(len(global_points) - 1)
            )
        )
        if not np.isfinite(rest_length) or rest_length <= 0.0:
            return None

        lm, lt, pen_angle, lmax = normalized_lengths
        muscles.append(
            MuscleUnitInfo(
                name=str(name),
                f0=float(f0),
                lm=lm,
                lt=lt,
                pen_angle=pen_angle,
                lmax=lmax,
                waypoints=waypoints,
            )
        )
        rest_lengths.append(rest_length)
        if native_l0 is not None:
            mujoco_l0.append(native_l0)

    if not muscles:
        return None
    _validate_muscle_name_order(root, muscle_names=[muscle.name for muscle in muscles])
    return ParsedMjcfMuscleData(
        muscles=muscles,
        rest_lengths=np.asarray(rest_lengths, dtype=np.float32),
        mujoco_l0=(
            np.asarray(mujoco_l0, dtype=np.float32) if mujoco_l0_available else None
        ),
    )


def _derive_normalized_lengths(
    model: mujoco.MjModel,
    actuator_id: int,
    actuator_elem: ET.Element,
    rest_length: float,
) -> tuple[float, float, float, float]:
    if (
        int(getattr(model, "nuser_actuator", 0)) >= 4
        and hasattr(model, "actuator_user")
        and model.actuator_user.shape[1] >= 4
    ):
        user = model.actuator_user[actuator_id, :4]
        return (
            float(user[0]),
            float(user[1]),
            float(user[2]),
            float(user[3]),
        )

    actuator_name = actuator_elem.get("name", f"actuator[{actuator_id}]")
    tendon_id = int(model.actuator_trnid[actuator_id, 0])
    if tendon_id < 0:
        raise ValueError(
            f"Actuator '{actuator_name}' does not reference a tendon, so muscle lengths cannot be derived."
        )

    if rest_length <= 0.0 or not np.isfinite(rest_length):
        raise ValueError(
            f"Actuator '{actuator_name}' has invalid rest length {rest_length!r}."
        )

    lengthrange = np.asarray(model.actuator_lengthrange[actuator_id], dtype=np.float64)
    gainprm = np.asarray(model.actuator_gainprm[actuator_id], dtype=np.float64)
    normalized_range_lo = float(gainprm[0])
    normalized_range_hi = float(gainprm[1])
    normalized_range_width = normalized_range_hi - normalized_range_lo
    if abs(normalized_range_width) <= 1.0e-8:
        raise ValueError(
            f"Actuator '{actuator_name}' has degenerate gainprm range {normalized_range_lo}, {normalized_range_hi}."
        )

    optimal_fiber_length = float(
        (float(lengthrange[1]) - float(lengthrange[0])) / normalized_range_width
    )
    # MuJoCo muscle normalization uses the actuator gain-range coordinates:
    #   L = range_lo + (length - lengthrange_lo) / L0
    # Rearranging gives LT = lengthrange_lo - range_lo * L0.
    # This matches the transmission-side normalization even when tendon_lengthspring
    # stores a different spring reference for wrapped tendons.
    tendon_slack_length = float(
        float(lengthrange[0]) - normalized_range_lo * optimal_fiber_length
    )
    return (
        float(optimal_fiber_length / rest_length),
        float(tendon_slack_length / rest_length),
        0.0,
        float(lengthrange[1] / rest_length),
    )


def _derive_mujoco_l0_from_model(
    model: mujoco.MjModel,
    actuator_id: int,
) -> float:
    lengthrange = np.asarray(model.actuator_lengthrange[actuator_id], dtype=np.float64)
    gainprm = np.asarray(model.actuator_gainprm[actuator_id], dtype=np.float64)
    physical_width = float(lengthrange[1] - lengthrange[0])
    normalized_width = float(gainprm[1] - gainprm[0])
    if physical_width <= 0.0 or normalized_width <= 0.0:
        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        raise ValueError(
            f"Actuator '{actuator_name or actuator_id}' has an invalid MuJoCo "
            "muscle normalization range."
        )
    return physical_width / normalized_width


def _require_metadata_site_paths(model):
    if np.any(model.wrap_type != int(mujoco.mjtWrap.mjWRAP_SITE)):
        raise ValueError(
            "Wrapped sources require their authored tendon topology; metadata override is unsupported."
        )


def _validate_spatial_topology(root, model):
    """A reused compiled model must describe the same authored transmissions."""
    for tendon in root.findall(".//tendon/spatial"):
        name = tendon.get("name")
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name or "")
        authored = [
            (child.tag, child.get(child.tag), child.get("sidesite"))
            for child in tendon
            if isinstance(child.tag, str)
        ]
        compiled = []
        if index >= 0:
            begin = int(model.tendon_adr[index])
            for i in range(begin, begin + int(model.tendon_num[index])):
                site = int(model.wrap_type[i]) == int(mujoco.mjtWrap.mjWRAP_SITE)
                tag = "site" if site else "geom"
                obj = mujoco.mjtObj.mjOBJ_SITE if site else mujoco.mjtObj.mjOBJ_GEOM
                obj_name = mujoco.mj_id2name(model, obj, int(model.wrap_objid[i]))
                side = int(model.wrap_prm[i]) if not site else -1
                side_name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, side)
                    if side >= 0
                    else None
                )
                compiled.append((tag, obj_name, side_name))
        if authored != compiled:
            raise ValueError(
                f"Compiled tendon '{name}' topology does not match the source XML."
            )


def _validated_waypoint_model(
    source_mjcf_path: str,
    *,
    compiled_model: mujoco.MjModel | None = None,
) -> mujoco.MjModel:
    """Validate the original transmission before metadata or cache overrides."""
    model = compiled_model
    if model is None:
        model = mujoco.MjModel.from_xml_path(
            str(Path(source_mjcf_path).expanduser().resolve())
        )
    tendon_actuators = model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_TENDON)
    unit_gear = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    if not np.allclose(
        model.actuator_gear[tendon_actuators], unit_gear, rtol=0.0, atol=1.0e-7
    ):
        raise ValueError(
            "Fast-muscle paths require unit tendon gear; scaled transmissions are unsupported."
        )
    supported = [
        int(mujoco.mjtWrap.mjWRAP_SITE),
        int(mujoco.mjtWrap.mjWRAP_SPHERE),
        int(mujoco.mjtWrap.mjWRAP_CYLINDER),
    ]
    if not np.isin(np.asarray(model.wrap_type), supported).all():
        raise ValueError(
            "Fast-muscle paths require sites with sphere/cylinder wraps; "
            "pulley and fixed-joint tendon geometry is unsupported."
        )
    return model


def parse_muscles_from_mjcf(
    source_mjcf_path: str,
    *,
    rest_keyframe_name: str | None = None,
    compiled_model: mujoco.MjModel | None = None,
) -> ParsedMjcfMuscleData:
    root, source_path = load_expanded_mjcf_root(source_mjcf_path)
    model = _validated_waypoint_model(str(source_path), compiled_model=compiled_model)
    _validate_spatial_topology(root, model)
    has_wraps = bool(np.any(model.wrap_type != int(mujoco.mjtWrap.mjWRAP_SITE)))
    spatial_waypoints = (
        {str(t.get("name")): [] for t in root.findall(".//tendon/spatial")}
        if has_wraps
        else _collect_spatial_waypoint_site_names(root)
    )
    spatial_tendon_names = set(spatial_waypoints.keys())

    if rest_keyframe_name is None and not has_wraps:
        parsed = _parse_muscles_from_mjcf_xml_fast(
            root,
            spatial_waypoints=spatial_waypoints,
        )
        if parsed is not None:
            return parsed

    data = mujoco.MjData(model)
    if rest_keyframe_name:
        key_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            str(rest_keyframe_name),
        )
        if key_id < 0:
            raise ValueError(
                f"Keyframe '{rest_keyframe_name}' was not found in {source_path}."
            )
        data.qpos[:] = model.key_qpos[key_id]
    else:
        data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    muscles: list[MuscleUnitInfo] = []
    wrap_rows = []
    point_offset = 0
    rest_lengths: list[float] = []
    mujoco_l0: list[float] = []

    for actuator_elem in root.findall(".//actuator/*"):
        if not isinstance(actuator_elem.tag, str):
            continue
        if actuator_elem.tag not in {"muscle", "general"}:
            if actuator_elem.get("tendon"):
                name = actuator_elem.get("name", actuator_elem.tag)
                raise ValueError(
                    f"Unsupported actuator tag '{actuator_elem.tag}' for tendon actuator '{name}'."
                )
            continue

        name = actuator_elem.get("name")
        tendon_name = actuator_elem.get("tendon")
        if not name or not tendon_name:
            raise ValueError(
                f"Muscle actuator is missing name or tendon attribute in {source_path}."
            )

        if tendon_name not in spatial_tendon_names:
            raise ValueError(
                f"Actuator '{name}' references tendon '{tendon_name}' but no spatial tendon definition was found."
            )

        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(name))
        if actuator_id < 0:
            raise ValueError(f"Actuator '{name}' was not found in compiled model.")

        rest_length = float(data.actuator_length[actuator_id])
        lm, lt, pen_angle, lmax = _derive_normalized_lengths(
            model=model,
            actuator_id=actuator_id,
            actuator_elem=actuator_elem,
            rest_length=rest_length,
        )

        waypoints: list[WaypointInfo] = []
        tendon_id = int(model.actuator_trnid[actuator_id, 0])
        begin = int(model.tendon_adr[tendon_id])
        end = begin + int(model.tendon_num[tendon_id])
        for wrap_id in range(begin, end):
            kind = int(model.wrap_type[wrap_id])
            if kind in (
                int(mujoco.mjtWrap.mjWRAP_CYLINDER),
                int(mujoco.mjtWrap.mjWRAP_SPHERE),
            ):
                if (
                    wrap_id == begin
                    or wrap_id + 1 == end
                    or int(model.wrap_type[wrap_id - 1])
                    != int(mujoco.mjtWrap.mjWRAP_SITE)
                    or int(model.wrap_type[wrap_id + 1])
                    != int(mujoco.mjtWrap.mjWRAP_SITE)
                ):
                    raise ValueError(
                        f"Tendon {tendon_name!r} requires site-geom-site wrapping."
                    )
                geom = int(model.wrap_objid[wrap_id])
                body_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])
                )
                if not body_name:
                    raise ValueError("Wrap geometry requires a named body.")
                side = int(model.wrap_prm[wrap_id])
                side_name = (
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[side])
                    )
                    if side >= 0
                    else None
                )
                if side >= 0 and not side_name:
                    raise ValueError("Wrap sidesite requires a named body.")
                wrap_rows.append(
                    (
                        str(body_name),
                        side_name,
                        kind,
                        model.geom_pos[geom].copy(),
                        model.geom_quat[geom].copy(),
                        float(model.geom_size[geom, 0]),
                        model.site_pos[side].copy() if side >= 0 else np.zeros(3),
                        point_offset + len(waypoints) - 1,
                    )
                )
                for _ in range(2):
                    local_pos = model.geom_pos[geom].copy()
                    waypoints.append(
                        WaypointInfo(
                            body=str(body_name),
                            position=local_pos,
                            attachments=[
                                WaypointAttachmentInfo(
                                    body=str(body_name),
                                    position=local_pos.copy(),
                                    weight=1.0,
                                )
                            ],
                        )
                    )
                continue
            if kind != int(mujoco.mjtWrap.mjWRAP_SITE):
                raise ValueError(
                    f"Tendon '{tendon_name}' uses unsupported pulley or fixed tendon geometry."
                )
            site_id = int(model.wrap_objid[wrap_id])
            body_id = int(model.site_bodyid[site_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not body_name:
                raise ValueError(
                    f"Site {site_id} on tendon '{tendon_name}' has no associated body name."
                )
            local_pos = np.asarray(model.site_pos[site_id], dtype=np.float64).copy()
            waypoints.append(
                WaypointInfo(
                    body=str(body_name),
                    position=local_pos,
                    attachments=[
                        WaypointAttachmentInfo(
                            body=str(body_name),
                            position=local_pos.copy(),
                            weight=1.0,
                        )
                    ],
                )
            )

        muscles.append(
            MuscleUnitInfo(
                name=str(name),
                f0=_derive_f0(model, actuator_id, actuator_elem),
                lm=lm,
                lt=lt,
                pen_angle=pen_angle,
                lmax=lmax,
                waypoints=waypoints,
            )
        )
        point_offset += len(waypoints)
        rest_lengths.append(rest_length)
        mujoco_l0.append(_derive_mujoco_l0_from_model(model, actuator_id))

    _validate_muscle_name_order(root, muscle_names=[muscle.name for muscle in muscles])
    wraps = (
        ParsedWaypointWraps(
            body_names=[r[0] for r in wrap_rows],
            side_body_names=[r[1] for r in wrap_rows],
            kind=np.asarray([r[2] for r in wrap_rows], dtype=np.int32),
            center=np.asarray([r[3] for r in wrap_rows], dtype=np.float32),
            quaternion=np.asarray([r[4] for r in wrap_rows], dtype=np.float32),
            radius=np.asarray([r[5] for r in wrap_rows], dtype=np.float32),
            side_local=np.asarray([r[6] for r in wrap_rows], dtype=np.float32),
            start_point=np.asarray([r[7] for r in wrap_rows], dtype=np.int32),
        )
        if wrap_rows
        else None
    )
    return ParsedMjcfMuscleData(
        wraps=wraps,
        muscles=muscles,
        rest_lengths=np.asarray(rest_lengths, dtype=np.float32),
        mujoco_l0=np.asarray(mujoco_l0, dtype=np.float32),
    )


@startup_nvtx_wrapped("mjcf_muscle:build_stripped_mjcf_spec")
def build_stripped_mjcf_spec(
    source_mjcf_path: str,
    *,
    preserve_position_actuators: bool = False,
    position_actuator_mjcf_path: str | None = None,
) -> mujoco.MjSpec:
    with startup_nvtx_range("mjcf_muscle:stripped_mjcf_cache_lookup"):
        cache_key = _stripped_mjcf_cache_key(
            source_mjcf_path,
            preserve_position_actuators=preserve_position_actuators,
            position_actuator_mjcf_path=position_actuator_mjcf_path,
        )
        cached_xml = load_pickle_cache("stripped_mjcf", cache_key)
    if isinstance(cached_xml, str):
        return mujoco.MjSpec.from_string(cached_xml)

    xml_text = _build_stripped_mjcf_xml_uncached(
        source_mjcf_path,
        preserve_position_actuators=preserve_position_actuators,
        position_actuator_mjcf_path=position_actuator_mjcf_path,
    )
    with startup_nvtx_range("mjcf_muscle:stripped_mjcf_cache_write"):
        store_pickle_cache("stripped_mjcf", cache_key, xml_text)
    return mujoco.MjSpec.from_string(xml_text)


def _build_stripped_mjcf_xml_uncached(
    source_mjcf_path: str,
    *,
    preserve_position_actuators: bool = False,
    position_actuator_mjcf_path: str | None = None,
) -> str:
    root, source_path = load_expanded_mjcf_root(source_mjcf_path)
    if position_actuator_mjcf_path is not None:
        actuator_root, _ = load_expanded_mjcf_root(position_actuator_mjcf_path)
        graft_position_actuators(root, actuator_root)
    elif preserve_position_actuators:
        preserve_only_position_actuator_elements(root)
    else:
        remove_all_actuator_elements(root)
    remove_all_tendon_elements(root)
    remove_unreferenced_named_sites(
        root,
        referenced_site_names=collect_referenced_site_names(root),
    )
    absolutize_compiler_dirs(root, source_path)
    _absolutize_asset_files_without_compiler_dirs(root, source_path)
    return ET.tostring(root, encoding="unicode")
