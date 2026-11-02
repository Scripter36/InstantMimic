from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from instantmimic.envs.config_access import get_block
from instantmimic.utils.project_paths import resolve_project_path_pattern


class MotionLoopSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    smooth: bool = False
    blend_frames: int = Field(default=30, ge=0)
    mode: Literal["tail_warp", "c1_bridge"] = "tail_warp"
    velocity_from_first_frame: bool = True
    velocity_extrapolate: bool = False


class MotionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: tuple[str, ...]
    scale: float = Field(gt=0.0)
    offset: tuple[float, float, float]
    bvh_map: dict[str, str | tuple[str, ...]]
    tracking_body_names: tuple[str, ...] = ()
    rest_bodies: tuple[str, ...] = ()
    joint_layout: Literal["model", "smpl"] = "model"
    root_auto_calibration: bool = False
    cache_mode: Literal["off", "auto", "refresh"] = "off"
    cache_directory: str = "motion_cache"
    loop: MotionLoopSettings = MotionLoopSettings()

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            raw_sources = (value,)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            raw_sources = tuple(value)
        else:
            raise ValueError("motion.source must be a path string or sequence")
        sources = tuple(str(item).strip() for item in raw_sources)
        if not sources or any(not item for item in sources):
            raise ValueError("motion.source must contain at least one non-empty path")
        return sources

    @field_validator("offset", mode="before")
    @classmethod
    def _normalize_offset(cls, value: object) -> tuple[float, float, float]:
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise ValueError("motion.offset must contain exactly 3 values")
        values = tuple(float(item) for item in value)
        if len(values) != 3 or not all(math.isfinite(item) for item in values):
            raise ValueError("motion.offset must contain exactly 3 finite values")
        return values  # type: ignore[return-value]

    @field_validator("cache_directory")
    @classmethod
    def _validate_cache_directory(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("motion.cache_directory must be non-empty")
        return value

    @field_validator("tracking_body_names", "rest_bodies", mode="before")
    @classmethod
    def _normalize_body_names(
        cls, value: object, info: ValidationInfo
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(
            value, (str, bytes, bytearray)
        ):
            raise ValueError(f"motion.{info.field_name} must be a sequence")
        names = tuple(str(item).strip() for item in value)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(
                f"motion.{info.field_name} must contain unique non-empty names"
            )
        return names

    @classmethod
    def from_env_cfg(
        cls,
        cfg: Any,
        *,
        source: object | None = None,
    ) -> MotionSettings:
        block = get_block(cfg, "motion")
        if isinstance(block, Mapping):
            values = dict(block.items())
        elif hasattr(block, "__dict__"):
            values = vars(block).copy()
        else:
            values = dict(block)
        if source is not None:
            values["source"] = source
        return cls.model_validate(values)

    def resolve_sources(self) -> tuple[str, ...]:
        resolved: list[str] = []
        seen: set[str] = set()
        for pattern in self.source:
            matches = resolve_project_path_pattern(pattern)
            if not matches:
                raise FileNotFoundError(
                    f"motion.source pattern matched no files: {pattern}"
                )
            for match in matches:
                resolved_path = str(match)
                if resolved_path in seen:
                    continue
                seen.add(resolved_path)
                resolved.append(resolved_path)
        if not resolved:
            raise FileNotFoundError("motion.source resolved to no files")
        return tuple(resolved)


class NanDebugSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    buffer_size: int = Field(default=120, ge=2)
    max_env_videos: int = Field(default=4, gt=0)
    output_directory: str = "nan_debug"
    raise_on_nan: bool = True

    @field_validator("output_directory")
    @classmethod
    def _validate_output_directory(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("nan_debug.output_directory must be non-empty")
        return value

    @classmethod
    def from_env_cfg(cls, cfg: Any) -> NanDebugSettings:
        block = get_block(cfg, "nan_debug", required=False) or {}
        if not isinstance(block, Mapping):
            block = dict(block)
        return cls.model_validate(block)


__all__ = ("MotionLoopSettings", "MotionSettings", "NanDebugSettings")
