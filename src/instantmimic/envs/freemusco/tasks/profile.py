from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from instantmimic.envs.config_access import get_value as cfg_get


@dataclass(frozen=True)
class FreemuscoRunProfile:
    """Resolved model/layout policy for the generic FreeMusco Run runtime.

    The profile is intentionally data-only.  Runtime code consumes the resolved
    names, indices, and policy switches without selecting an implementation by
    model identity.
    """

    species: str
    root_body_name: str
    head_body_name: str
    torso_body_name: str | None
    foot_body_names: tuple[str, str]
    pelvis_geom_name: str
    root_tilt_joint_names: tuple[str, ...]
    root_yaw_joint_name: str
    forward_joint_name: str | None
    forward_velocity_source: str
    forward_velocity_sensor_name: str
    qpos_observation_skip: str
    torso_tilt_source: str
    termination_torso_tilt_mode: str
    supports_dynamic_target_forward_velocity: bool

    @classmethod
    def from_cfg(cls, cfg: Any) -> FreemuscoRunProfile:
        species = str(cfg_get(cfg, "species", "")).strip()
        if not species:
            raise ValueError("FreemuscoRunEnv requires cfg.species.")

        foot_body_names = tuple(
            str(name) for name in cfg_get(cfg, "foot_body_names", ())
        )
        if len(foot_body_names) != 2:
            raise ValueError("foot_body_names must contain exactly two body names.")

        root_tilt_joint_names = tuple(
            str(name) for name in cfg_get(cfg, "root_tilt_joint_names", ())
        )
        forward_joint_name = cfg_get(cfg, "forward_joint_name", None)
        torso_body_name = cfg_get(cfg, "torso_body_name", None)
        qpos_observation_skip = (
            str(
                cfg_get(
                    cfg,
                    "qpos_observation_skip",
                    "forward_joint" if forward_joint_name else "first",
                )
            )
            .strip()
            .lower()
        )
        torso_tilt_source = (
            str(
                cfg_get(
                    cfg,
                    "torso_tilt_source",
                    "torso_upright" if torso_body_name else "root_tilt_xy",
                )
            )
            .strip()
            .lower()
        )

        profile = cls(
            species=species,
            root_body_name=str(cfg_get(cfg, "root_body_name", "")).strip(),
            head_body_name=str(cfg_get(cfg, "head_body_name", "")).strip(),
            torso_body_name=(
                None if torso_body_name is None else str(torso_body_name).strip()
            ),
            foot_body_names=(foot_body_names[0], foot_body_names[1]),
            pelvis_geom_name=str(cfg_get(cfg, "pelvis_geom_name", "pelvis")).strip(),
            root_tilt_joint_names=root_tilt_joint_names,
            root_yaw_joint_name=str(cfg_get(cfg, "root_yaw_joint_name", "")).strip(),
            forward_joint_name=(
                None if forward_joint_name is None else str(forward_joint_name).strip()
            ),
            forward_velocity_source=str(
                cfg_get(cfg, "forward_velocity_source", "qvel_root_x")
            )
            .strip()
            .lower(),
            forward_velocity_sensor_name=str(
                cfg_get(cfg, "forward_velocity_sensor_name", "torso_subtreelinvel")
            ).strip(),
            qpos_observation_skip=qpos_observation_skip,
            torso_tilt_source=torso_tilt_source,
            termination_torso_tilt_mode=str(
                cfg_get(cfg, "termination_torso_tilt_mode", "xy_norm")
            )
            .strip()
            .lower(),
            supports_dynamic_target_forward_velocity=bool(
                cfg_get(cfg, "supports_dynamic_target_forward_velocity", False)
            ),
        )
        profile.validate_config()
        return profile

    def validate_config(self) -> None:
        required_names = {
            "root_body_name": self.root_body_name,
            "head_body_name": self.head_body_name,
            "pelvis_geom_name": self.pelvis_geom_name,
            "root_yaw_joint_name": self.root_yaw_joint_name,
        }
        missing = [name for name, value in required_names.items() if not value]
        if missing:
            raise ValueError(f"FreeMusco Run profile has empty names: {missing}")
        if any(not name for name in self.foot_body_names):
            raise ValueError("foot_body_names must not contain empty names.")
        if self.torso_body_name == "":
            raise ValueError("torso_body_name must be omitted or non-empty.")
        if self.qpos_observation_skip not in {"first", "forward_joint"}:
            raise ValueError(
                "qpos_observation_skip must be 'first' or 'forward_joint', got "
                f"{self.qpos_observation_skip!r}."
            )
        if (
            self.qpos_observation_skip == "forward_joint"
            and not self.forward_joint_name
        ):
            raise ValueError(
                "qpos_observation_skip='forward_joint' requires forward_joint_name."
            )
        if self.forward_velocity_source not in {
            "qvel_root_x",
            "sensor_subtreelinvel_x",
        }:
            raise ValueError(
                "forward_velocity_source must be 'qvel_root_x' or "
                f"'sensor_subtreelinvel_x', got {self.forward_velocity_source!r}."
            )
        if self.torso_tilt_source not in {"torso_upright", "root_tilt_xy"}:
            raise ValueError(
                "torso_tilt_source must be 'torso_upright' or 'root_tilt_xy', got "
                f"{self.torso_tilt_source!r}."
            )
        if self.torso_tilt_source == "torso_upright" and not self.torso_body_name:
            raise ValueError(
                "torso_tilt_source='torso_upright' requires torso_body_name."
            )
        if (
            self.torso_tilt_source == "root_tilt_xy"
            and len(self.root_tilt_joint_names) != 2
        ):
            raise ValueError(
                "torso_tilt_source='root_tilt_xy' requires two root_tilt_joint_names."
            )
        if self.termination_torso_tilt_mode not in {"xy_norm", "root_rot_y_abs"}:
            raise ValueError(
                "termination_torso_tilt_mode must be 'xy_norm' or "
                f"'root_rot_y_abs', got {self.termination_torso_tilt_mode!r}."
            )


@dataclass(frozen=True)
class FreemuscoRunLayout:
    """Dimensions and resolved indices consumed by generic kernels."""

    nq: int
    nv: int
    na: int
    nu: int
    nbody: int
    qpos_observation_skip_idx: int
    root_tilt_qpos_idx_0: int
    root_tilt_qpos_idx_1: int

    @property
    def native_state_observation_dim(self) -> int:
        return 4 + (self.nq - 1) + self.nv + self.na + (3 * self.nu) + 1

    def validate(self) -> None:
        dimensions = {
            "nq": self.nq,
            "nv": self.nv,
            "na": self.na,
            "nu": self.nu,
            "nbody": self.nbody,
        }
        invalid = {
            name: value
            for name, value in dimensions.items()
            if value < (1 if name in {"nq", "nv", "nbody"} else 0)
        }
        if invalid:
            raise ValueError(
                f"FreeMusco Run model dimensions must be positive: {invalid}"
            )
        if not 0 <= self.qpos_observation_skip_idx < self.nq:
            raise ValueError(
                "Resolved qpos observation skip index is outside qpos layout: "
                f"{self.qpos_observation_skip_idx} for nq={self.nq}."
            )
        for name, value in (
            ("root_tilt_qpos_idx_0", self.root_tilt_qpos_idx_0),
            ("root_tilt_qpos_idx_1", self.root_tilt_qpos_idx_1),
        ):
            if value >= self.nq:
                raise ValueError(
                    f"Resolved {name}={value} is outside qpos layout nq={self.nq}."
                )
