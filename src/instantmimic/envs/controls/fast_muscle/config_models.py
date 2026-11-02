from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Annotated, Any, Literal, Self

from omegaconf import OmegaConf, open_dict
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from instantmimic.envs.config_access import get_block, get_value

type ActionType = Literal["ctrl", "activation", "torque", "mass", "mujoco_activation"]
type ScaleValue = Literal["auto"] | float | tuple[float, ...]


class _SettingsModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


def _numeric_tuple(value: object, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a numeric sequence")
    values = tuple(float(item) for item in value)
    if not values or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{field_name} must contain finite values")
    return values


def _torque_scale_tuple(value: object, *, field_name: str) -> tuple[float, ...]:
    values = _numeric_tuple(value, field_name=field_name)
    if any(item < 0.0 for item in values):
        raise ValueError(f"{field_name} values must be nonnegative")
    # Zero entries deliberately mask torque coordinates out of the action space.
    return values


def _ordered_pair(
    value: object,
    *,
    field_name: str,
    strictly_positive: bool = False,
) -> tuple[float, float]:
    values = _numeric_tuple(value, field_name=field_name)
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError(f"{field_name} must be an ascending [min, max] pair")
    if strictly_positive and values[0] <= 0.0:
        raise ValueError(f"{field_name} values must be positive")
    return values[0], values[1]


class TorqueBoundsSettings(_SettingsModel):
    env_clip: bool = False
    chunk_size: int = Field(default=8, gt=0)


class MuscleNNOptimizerSettings(_SettingsModel):
    type: Literal["adam", "adamw"] = "adam"


class MuscleNNSettings(_SettingsModel):
    hidden_dims: tuple[int, ...] = (1024, 512, 512)
    # Unset output functions take the pose or torque coordinator default.
    output_activation: (
        Literal["clamp", "sine", "sigmoid", "tanh_relu", "identity"] | None
    ) = None
    target_includes_passive_force: bool | None = None
    input_scale: float = Field(default=1.0, gt=0.0)
    # ``running`` standardizes every column of the scaled input
    # ``x / input_scale`` with its cumulative mean and deviation over the
    # collected MuscleNN samples, clamped to
    # ``[-input_normalization_clip, input_normalization_clip]`` and with the
    # deviation floored at ``input_normalization_min_std``.
    input_normalization: Literal["none", "running"] = "none"
    input_normalization_clip: float = Field(default=5.0, gt=0.0)
    input_normalization_min_std: float = Field(default=1.0e-3, gt=0.0)
    target_scale: float = Field(default=1.0, gt=0.0)
    target_loss_weight: float = Field(default=1.0, gt=0.0)
    substep_sampling_mode: Literal["all", "random"] = "all"
    reuse_activation_across_substeps: bool = False
    # Freeze the coordinator when adapting a policy from its checkpoint.
    frozen: bool = False
    optimizer: MuscleNNOptimizerSettings = MuscleNNOptimizerSettings()
    learning_rate: float = Field(default=1.0e-4, gt=0.0)
    batch_size: int = Field(default=128, gt=0)
    epochs: int = Field(default=3, gt=0)
    regularization_weight: float = Field(default=0.01, ge=0.0)
    bound_loss_weight: float = Field(default=0.0, ge=0.0)
    bound_tolerance: tuple[float, float] = (0.0, 0.0)
    buffer_capacity: int = Field(default=32768, gt=0)
    grad_clip: float = Field(default=0.5, gt=0.0)

    @field_validator("hidden_dims")
    @classmethod
    def _validate_hidden_dims(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(dim <= 0 for dim in value):
            raise ValueError("action.muscle_nn.hidden_dims must be positive")
        return value

    @field_validator("bound_tolerance", mode="before")
    @classmethod
    def _validate_bound_tolerance(cls, value: object) -> tuple[float, float]:
        return _ordered_pair(value, field_name="action.muscle_nn.bound_tolerance")

    @model_validator(mode="after")
    def _validate_input_normalization(self) -> Self:
        if self.input_normalization == "running":
            return self
        fields = type(self).model_fields
        authored = [
            name
            for name in ("input_normalization_clip", "input_normalization_min_std")
            if getattr(self, name) != fields[name].default
        ]
        if authored:
            raise ValueError(
                f"action.muscle_nn.{', '.join(authored)} apply only with "
                "action.muscle_nn.input_normalization='running'"
            )
        return self


class _ActionSettings(_SettingsModel):
    type: ActionType
    space: int = Field(gt=0)


class _ScaledActionSettings(_ActionSettings):
    scale: ScaleValue = 1.0

    @field_validator("scale", mode="before")
    @classmethod
    def _normalize_scale(cls, value: object) -> ScaleValue:
        if isinstance(value, str):
            if value != "auto":
                raise ValueError("action.scale string value must be 'auto'")
            return value
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return _torque_scale_tuple(value, field_name="action.scale")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("action.scale must be finite and positive")
        return numeric


class CtrlActionSettings(_ActionSettings):
    type: Literal["ctrl"]
    scale: float = Field(default=1.0, gt=0.0)
    use_reference: bool = False


class MujocoActivationActionSettings(_ActionSettings):
    type: Literal["mujoco_activation"]


class ActivationActionSettings(_ActionSettings):
    """The policy outputs one excitation per muscle; the environment clamps it to [0, 1]."""

    type: Literal["activation"]


class TorqueActionSettings(_ScaledActionSettings):
    type: Literal["torque"]
    bounds: TorqueBoundsSettings = TorqueBoundsSettings()


class StablePDSettings(_SettingsModel):
    mjcf_path: str = Field(min_length=1)
    kp_scale: float = Field(default=1.0, ge=0.0)
    kv_scale: float = Field(default=1.0, ge=0.0)
    torque_scale: float = Field(default=1.0, gt=0.0)
    torque_limit: float = Field(default=0.0, ge=0.0)
    regularization: float = Field(default=1.0e-6, gt=0.0)


class MassTorquePolicyOutputSettings(_SettingsModel):
    type: Literal["torque"]
    scale: ScaleValue = "auto"

    @field_validator("scale", mode="before")
    @classmethod
    def _normalize_scale(cls, value: object) -> ScaleValue:
        if isinstance(value, str):
            if value != "auto":
                raise ValueError(
                    "action.policy_output.scale string value must be 'auto'"
                )
            return value
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return _torque_scale_tuple(value, field_name="action.policy_output.scale")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("action.policy_output.scale must be finite and positive")
        return numeric


class MassPosePolicyOutputSettings(_SettingsModel):
    type: Literal["pose"]
    scale: float = Field(default=0.1, gt=0.0)
    # The Stable-PD target is this pose plus the scaled action: the reference
    # motion's actuator targets, or the actuator lengths at the model's qpos0
    # for tasks without a reference motion.
    reference: Literal["motion", "rest"] = "motion"
    stable_pd: StablePDSettings


type MassPolicyOutputSettings = Annotated[
    MassTorquePolicyOutputSettings | MassPosePolicyOutputSettings,
    Field(discriminator="type"),
]


class MassActionSettings(_ActionSettings):
    type: Literal["mass"]
    policy_output: MassPolicyOutputSettings
    bounds: TorqueBoundsSettings = TorqueBoundsSettings()
    muscle_nn: MuscleNNSettings = MuscleNNSettings()


type ActionConfig = Annotated[
    CtrlActionSettings
    | ActivationActionSettings
    | TorqueActionSettings
    | MassActionSettings
    | MujocoActivationActionSettings,
    Field(discriminator="type"),
]

_ACTION_ADAPTER = TypeAdapter(ActionConfig)


class ActionSettings:
    """Strict parser facade for the environment-owned action union."""

    @classmethod
    def from_env_cfg(cls, cfg: Any) -> ActionConfig:
        action = get_block(cfg, "action")
        if get_value(action, "type", None) == "musubi":
            from instantmimic.rl.models.musubi_actor_critic import raise_unavailable

            raise_unavailable()
        if not isinstance(action, Mapping):
            try:
                action = vars(action)
            except TypeError:
                action = dict(action)
        try:
            return _ACTION_ADAPTER.validate_python(action)
        except ValidationError as exc:
            raise ValueError(f"invalid env.action: {exc}") from exc


def _set_runtime_space(cfg: Any, name: str, value: int) -> None:
    resolved_value = int(value)
    if OmegaConf.is_config(cfg):
        with open_dict(cfg):
            cfg[name] = resolved_value
    elif isinstance(cfg, MutableMapping):
        cfg[name] = resolved_value
    else:
        setattr(cfg, name, resolved_value)


def set_runtime_action_space(cfg: Any, action_space: int) -> None:
    """Set DirectRLEnv's derived action-space field on its mutable cfg copy."""
    _set_runtime_space(cfg, "action_space", action_space)


def set_runtime_observation_space(cfg: Any, observation_space: int) -> None:
    """Set DirectRLEnv's derived observation-space field on its mutable cfg copy."""
    _set_runtime_space(cfg, "observation_space", observation_space)


class PassiveForceSettings(_SettingsModel):
    env: bool = True
    type: Literal["exponential", "mujoco"] = "exponential"
    k_pe: float = Field(default=5.5, gt=0.0)
    e_mo: float = Field(default=0.3, gt=0.0)
    lmax: float = Field(default=1.6, gt=1.0)
    fpmax: float = Field(default=1.3, ge=0.0)


class ActiveForceSettings(_SettingsModel):
    type: Literal["gaussian", "mujoco"] = "gaussian"
    gamma: float = Field(default=0.45, gt=0.0)
    lmin: float = Field(default=0.5, gt=0.0, lt=1.0)
    lmax: float = Field(default=1.6, gt=1.0)


class ForceVelocitySettings(_SettingsModel):
    enabled: bool = False
    vmax: float = Field(default=1.5, gt=0.0)
    fvmax: float = Field(default=1.5, gt=1.0)
    implicit: bool = True


class MuscleDynamicsSettings(_SettingsModel):
    model: Literal["instantaneous"] = "instantaneous"


class MuscleGeometrySettings(_SettingsModel):
    backend: Literal["waypoint"] = "waypoint"
    waypoint_attachment: Literal["single_body"] = "single_body"


class NativeTileSettings(_SettingsModel):
    # The packed tile event holds the tile muscle in five bits and the local
    # start point in nine: at most 32 muscles and 288 points per tile.
    muscle_cap: int = Field(default=32, gt=0, le=32)
    segment_cap: int = Field(default=256, gt=0, le=287)
    event_cap: int = Field(default=512, gt=0)
    # Bodies one tile reduces; the kernel accumulates per-body wrenches in
    # shared memory, so a larger cap only packs more muscles per tile.
    body_cap: int = Field(default=32, gt=0, le=32)

    @model_validator(mode="after")
    def _fit_packed_tile_points(self) -> Self:
        if self.muscle_cap + self.segment_cap > 288:
            raise ValueError(
                "muscle.lbs.body_wrench.native_tile: muscle_cap + segment_cap "
                f"must be at most 288 tile points, got {self.muscle_cap} + "
                f"{self.segment_cap}"
            )
        return self


class BodyWrenchSettings(_SettingsModel):
    # ``native_tile`` is the hand-written CUDA tile kernel and the only fast
    # path for body-fixed waypoint forces; ``pairchain`` is the Warp
    # chunked LBS path. Unset, it is ``native_tile`` wherever the kernel can
    # run (see ``MuscleSettings.body_wrench_forward_mode``).
    forward_mode: Literal["pairchain", "native_tile"] | None = None
    native_tile: NativeTileSettings = NativeTileSettings()


class MuscleLBSSettings(_SettingsModel):
    attachment_layout: Literal["body_wrench", "waypoint_force"] = "body_wrench"
    forward_reduction: Literal["partial", "serial"] = "partial"
    forward_chunk_size: int = Field(default=2048, gt=0)
    forward_tile_block_dim: int = Field(default=128, gt=0)
    body_event_chunk_size: int = Field(default=8192, gt=0)
    muscle_length_by_muscle_min_batch: int = Field(default=512, gt=0)
    body_wrench: BodyWrenchSettings = BodyWrenchSettings()
    sparse_forward: bool = True
    sparse_direct: bool = False
    sparse_direct_backward: bool = True

    @model_validator(mode="after")
    def _validate_tile_size(self) -> MuscleLBSSettings:
        if self.forward_tile_block_dim > self.forward_chunk_size:
            raise ValueError(
                "muscle.lbs.forward_tile_block_dim must not exceed forward_chunk_size"
            )
        return self


class MuscleSettings(_SettingsModel):
    @property
    def body_wrench_forward_mode(self) -> str:
        """The authored body-wrench forward mode, else ``native_tile`` when
        its kernel can run with a body-wrench layout."""
        authored = self.lbs.body_wrench.forward_mode
        if authored is not None:
            return authored
        native_runs = self.lbs.attachment_layout == "body_wrench"
        return "native_tile" if native_runs else "pairchain"

    metadata_path: str | None = None
    force_scale: float = Field(default=1.0, gt=0.0)
    length_range: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Positive normalized muscle-length bounds used by physical "
            "force-length evaluation and muscle observations."
        ),
    )
    active_force: ActiveForceSettings = ActiveForceSettings()
    passive_force: PassiveForceSettings = PassiveForceSettings()
    force_velocity: ForceVelocitySettings = ForceVelocitySettings()
    dynamics: MuscleDynamicsSettings = MuscleDynamicsSettings()
    geometry: MuscleGeometrySettings = MuscleGeometrySettings()
    lbs: MuscleLBSSettings = MuscleLBSSettings()

    @field_validator("length_range", mode="before")
    @classmethod
    def _validate_length_range(cls, value: object) -> tuple[float, float] | None:
        if value is None:
            return None
        lower, upper = _ordered_pair(
            value,
            field_name="muscle.length_range",
            strictly_positive=True,
        )
        if lower >= upper:
            raise ValueError("muscle.length_range must satisfy min < max")
        return lower, upper

    @classmethod
    def from_env_cfg(cls, cfg: Any) -> MuscleSettings:
        muscle = get_block(cfg, "muscle", required=False) or {}
        if not isinstance(muscle, Mapping):
            muscle = dict(muscle)
        return cls.model_validate(muscle)


class ActivationVisualizationSettings(_SettingsModel):
    enabled: bool = False
    color_low: tuple[float, ...] = (0.20, 0.25, 0.90)
    color_high: tuple[float, ...] = (1.00, 0.15, 0.10)
    alpha: float = Field(default=0.95, ge=0.0, le=1.0)
    interpolation: Literal["rgb", "hsv"] = "rgb"
    line_width: float = Field(default=3.0, gt=0.0)
    composite_mode: str = "alpha"

    @field_validator("color_low", "color_high", mode="before")
    @classmethod
    def _validate_color(cls, value: object) -> tuple[float, ...]:
        values = _numeric_tuple(value, field_name="activation visualization color")
        if len(values) not in {3, 4} or any(
            item < 0.0 or item > 1.0 for item in values
        ):
            raise ValueError("activation visualization colors must be RGB or RGBA")
        return values

    @classmethod
    def from_env_cfg(cls, cfg: Any) -> ActivationVisualizationSettings:
        block = get_block(cfg, "activation_visualization", required=False) or {}
        if not isinstance(block, Mapping):
            block = dict(block)
        return cls.model_validate(block)


class FastMuscleObservationSettings(_SettingsModel):
    previous_action: Literal["action"] | None = "action"
    applied_torque: bool = False
    muscle_geometry: bool = Field(default=True, strict=True)

    @classmethod
    def from_env_cfg(cls, cfg: Any) -> FastMuscleObservationSettings:
        observations = get_block(cfg, "observations", required=False) or {}
        values = {
            "previous_action": get_value(observations, "previous_action", "action"),
            "applied_torque": get_value(observations, "applied_torque", False),
            "muscle_geometry": get_value(observations, "muscle_geometry", True),
        }
        return cls.model_validate(values)


__all__ = (
    "ActionConfig",
    "ActionSettings",
    "ActivationVisualizationSettings",
    "FastMuscleObservationSettings",
    "MuscleSettings",
    "set_runtime_action_space",
    "set_runtime_observation_space",
)
