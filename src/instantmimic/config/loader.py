from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NoReturn

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pydantic import ValidationError

from instantmimic.config import coerce
from instantmimic.config.schema import SemanticConfig, ThawedValue
from instantmimic.config.validation import ConfigSchemaError, validate_semantic_mapping

_NULLABLE_INHERITED_BLOCK_PATHS = (
    ("agent", "learning_rate", "scheduler"),
    ("trainer", "benchmark"),
)


def _fail(path: str, phrase: str, cause: BaseException | None = None) -> NoReturn:
    coerce.raise_translated(ConfigSchemaError(path, phrase), cause)


_COERCE = coerce.Coercions(_fail)


def _normalize_plain(value: object, path: str) -> ThawedValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        mapping = _COERCE.mapping(value, path, string_keys=True)
        return {
            key: _normalize_plain(item, f"{path}.{key}" if path else key)
            for key, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _normalize_plain(item, f"{path}.{index}")
            for index, item in enumerate(value)
        ]
    raise ConfigSchemaError(
        path or "config", f"unsupported value type {type(value).__name__}"
    )


def _plain_mapping(source: object) -> dict[str, ThawedValue]:
    if isinstance(source, DictConfig):
        try:
            resolved = OmegaConf.to_container(
                source, resolve=True, throw_on_missing=True
            )
        except OmegaConfBaseException as exc:
            raise ConfigSchemaError(
                "config", f"could not resolve OmegaConf input: {exc}"
            ) from exc
    elif isinstance(source, Mapping):
        resolved = source
    else:
        raise ConfigSchemaError("config", "expected a DictConfig or mapping")
    if not isinstance(resolved, Mapping):
        raise ConfigSchemaError("config", "resolved root must be a mapping")
    normalized = _normalize_plain(resolved, "")
    if not isinstance(normalized, dict):
        raise ConfigSchemaError("config", "resolved root must be a mapping")
    for path in _NULLABLE_INHERITED_BLOCK_PATHS:
        parent: dict[str, ThawedValue] = normalized
        for key in path[:-1]:
            child = parent.get(key)
            if not isinstance(child, dict):
                break
            parent = child
        else:
            if parent.get(path[-1], object()) is None:
                del parent[path[-1]]
    return normalized


def load_semantic_config(
    cfg: DictConfig | Mapping[str, object],
) -> SemanticConfig:
    """Resolve and freeze one composed v1 document."""

    values = _plain_mapping(cfg)
    if "schema_version" not in values:
        raise ConfigSchemaError(
            "schema_version",
            "missing; this loader accepts only schema version 1",
        )
    version = values["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        raise ConfigSchemaError("schema_version", "must be the integer 1")
    if version != 1:
        raise ConfigSchemaError(
            "schema_version",
            f"unsupported version {version}; this loader accepts version 1",
        )
    validate_semantic_mapping(values)
    try:
        return SemanticConfig.from_mapping(values)
    except ValidationError as exc:
        error = exc.errors(include_url=False)[0]
        # Union branches add class/tag names to loc. Keep only the path through
        # authored data, including the final missing or unknown field.
        current: object = values
        parts: list[str] = []
        location = error["loc"]
        for index, part in enumerate(location):
            if isinstance(current, Mapping):
                if part not in current and index < len(location) - 1:
                    continue
                parts.append(str(part))
                current = current.get(part)
            elif isinstance(current, list) and isinstance(part, int):
                parts.append(str(part))
                current = current[part] if part < len(current) else None
            else:
                parts.append(str(part))
                current = None
        raise ConfigSchemaError(".".join(parts) or "config", error["msg"]) from exc
