from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import GenericAlias, MappingProxyType
from typing import NoReturn, get_args, get_origin

from pydantic import ValidationError

from instantmimic.config import agent_config as agent_settings
from instantmimic.config import coerce, schema_fields
from instantmimic.config import experiment_config as experiment_settings
from instantmimic.config import play_config as play_settings
from instantmimic.config import trainer_config as trainer_settings
from instantmimic.config.agent_config import PpoAgentConfig
from instantmimic.config.model_config import (
    MlpNetworkConfig,
    ModelsConfig,
    ResidualNetworkConfig,
)
from instantmimic.config.policy_config import BoundLossConfig, DistributionConfig
from instantmimic.config.runtime_config import RuntimeConfig
from instantmimic.config.schema import (
    BoundsField,
    EnumField,
    IntegerField,
    NullableField,
    NumberField,
    OptionalField,
    ScalarValue,
    SequenceField,
    StringMapField,
    SymmetricBoundsField,
    TaggedUnionField,
    ThawedValue,
    UntaggedUnionField,
)
from instantmimic.config.settings import field_names


class ConfigSchemaError(ValueError):
    """A schema error whose message starts with the failing dotted path."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.detail = message
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True, slots=True)
class _TypeSpec:
    expected: type[object]


@dataclass(frozen=True, slots=True)
class _SequenceSpec:
    item: object
    min_length: int
    max_length: int | None


@dataclass(frozen=True, slots=True)
class _ObjectSpec:
    fields: Mapping[str, object]
    optional: frozenset[str]


@dataclass(frozen=True, slots=True)
class _EnumSpec:
    values: tuple[ScalarValue, ...]


@dataclass(frozen=True, slots=True)
class _TaggedSpec:
    discriminator: str
    variants: Mapping[str, _ObjectSpec]


@dataclass(frozen=True, slots=True)
class _BoundsSpec:
    pass


@dataclass(frozen=True, slots=True)
class _SymmetricBoundsSpec:
    pass


@dataclass(frozen=True, slots=True)
class _IntegerSpec:
    minimum: int | None
    maximum: int | None


@dataclass(frozen=True, slots=True)
class _NumberSpec:
    minimum: float | None
    maximum: float | None
    exclusive_minimum: bool
    exclusive_maximum: bool


@dataclass(frozen=True, slots=True)
class _UntaggedSpec:
    variants: tuple[_ObjectSpec, ...]


@dataclass(frozen=True, slots=True)
class _NullableSpec:
    value: object


@dataclass(frozen=True, slots=True)
class _StringMapSpec:
    value: object


_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "seed",
        "task",
        "env",
        "models",
        "agent",
        "trainer",
        "experiment",
        "runtime",
        "play",
    }
)
_LEGACY_PATHS = frozenset(
    {
        "memory",
        "checkpoint",
        "video",
        "env.scene",
        "env.seed",
        "env.log_dir",
        "env.viewer",
        "env.motion_file",
        "env.motion_files",
        "models.policy",
        "models.value",
        "agent.lambda",
        "agent.gradient_steps",
        "agent.optimizer_kwargs",
        "agent.fixed_sigma",
        "agent.fixed_log_std",
        "agent.min_log_std",
        "agent.max_log_std",
        "agent.learning_rate_scheduler",
        "agent.actor_learning_rate",
        "agent.critic_learning_rate",
        "agent.alpha_learning_rate",
        "agent.experiment",
        "trainer.class",
        "trainer.timesteps",
        "trainer.environment_info",
    }
)
_LEGACY_PREFIXES = (
    "video_",
    "play_",
    "trainer.eval_",
    "trainer.clip_failure_benchmark_",
)


def _schema_type_name(definition: object) -> str:
    if isinstance(definition, type):
        return definition.__name__
    return type(definition).__name__


def _compile_object_schema(fields: Mapping[str, object], *, path: str) -> _ObjectSpec:
    compiled: dict[str, object] = {}
    optional: set[str] = set()
    for name, definition in fields.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"{path} field names must be non-empty strings")
        if isinstance(definition, OptionalField):
            optional.add(name)
            definition = definition.schema
        compiled[name] = _compile_schema(definition, path=f"{path}.{name}")
    return _ObjectSpec(MappingProxyType(compiled), frozenset(optional))


def _compile_schema(definition: object, *, path: str) -> object:
    if definition in (str, int, float, bool):
        return _TypeSpec(definition)
    if isinstance(definition, Mapping):
        return _compile_object_schema(definition, path=path)
    if isinstance(definition, NullableField):
        return _NullableSpec(_compile_schema(definition.schema, path=path))
    if isinstance(definition, StringMapField):
        return _StringMapSpec(
            _compile_schema(definition.value_schema, path=f"{path}.*")
        )
    if isinstance(definition, SequenceField):
        if definition.min_length < 0:
            raise TypeError(f"{path} min_length must be nonnegative")
        if (
            definition.max_length is not None
            and definition.max_length < definition.min_length
        ):
            raise TypeError(f"{path} max_length must not be below min_length")
        return _SequenceSpec(
            _compile_schema(definition.item_schema, path=f"{path}[]"),
            definition.min_length,
            definition.max_length,
        )
    if isinstance(definition, EnumField):
        if not definition.values:
            raise TypeError(f"{path} enum must contain at least one value")
        return _EnumSpec(definition.values)
    if isinstance(definition, BoundsField):
        return _BoundsSpec()
    if isinstance(definition, SymmetricBoundsField):
        return _SymmetricBoundsSpec()
    if isinstance(definition, IntegerField):
        return _IntegerSpec(definition.minimum, definition.maximum)
    if isinstance(definition, NumberField):
        minimum = float(definition.minimum) if definition.minimum is not None else None
        maximum = float(definition.maximum) if definition.maximum is not None else None
        return _NumberSpec(
            minimum,
            maximum,
            definition.exclusive_minimum,
            definition.exclusive_maximum,
        )
    if isinstance(definition, TaggedUnionField):
        variants = {
            tag: _compile_object_schema(fields, path=f"{path}<{tag}>")
            for tag, fields in definition.variants.items()
        }
        if not variants:
            raise TypeError(f"{path} tagged union must contain at least one variant")
        return _TaggedSpec(definition.discriminator, MappingProxyType(variants))
    if isinstance(definition, UntaggedUnionField):
        variants = tuple(
            _compile_object_schema(fields, path=f"{path}<variant-{index}>")
            for index, fields in enumerate(definition.variants)
        )
        return _UntaggedSpec(variants)
    if isinstance(definition, GenericAlias):
        origin = get_origin(definition)
        arguments = get_args(definition)
        if origin is list and len(arguments) == 1:
            return _SequenceSpec(
                _compile_schema(arguments[0], path=f"{path}[]"), 0, None
            )
    raise TypeError(
        f"{path} uses unsupported schema {_schema_type_name(definition)}; "
        "register a closed typed schema"
    )


def _error(path: str, message: str, cause: BaseException | None = None) -> NoReturn:
    coerce.raise_translated(ConfigSchemaError(path, message), cause)


_COERCE = coerce.Coercions(_error)


def _is_legacy_path(path: str) -> bool:
    return path.startswith(_LEGACY_PREFIXES) or any(
        path == legacy or path.startswith(f"{legacy}.") for legacy in _LEGACY_PATHS
    )


def _mapping(value: object, path: str) -> Mapping[str, object]:
    return _COERCE.mapping(value, path, string_keys=True)


def _closed(
    value: object,
    *,
    path: str,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    mapping = _mapping(value, path)
    unknown, missing = coerce.closed_keys(mapping, allowed, required)
    for key in unknown:
        full_path = f"{path}.{key}" if path else key
        if _is_legacy_path(full_path):
            _error(
                full_path,
                "legacy key is not valid in schema version 1",
            )
        label = "unknown top-level key" if not path else "unknown key"
        _error(full_path, label)
    for key in missing:
        full_path = f"{path}.{key}" if path else key
        _error(full_path, "missing required key")
    return mapping


def _string(value: object, path: str, *, nonempty: bool = False) -> str:
    return _COERCE.string(value, path, nonempty=nonempty)


def _bool(value: object, path: str) -> bool:
    return _COERCE.boolean(value, path)


def _int(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    number = _COERCE.integer(value, path)
    if minimum is not None and number < minimum:
        _error(path, f"must be at least {minimum}")
    if maximum is not None and number > maximum:
        _error(path, f"must be at most {maximum}")
    return number


def _finite_number(value: object, path: str) -> float:
    return float(_COERCE.finite(value, path))


def _number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
    strict_maximum: bool = False,
) -> float:
    numeric = _finite_number(value, path)
    if minimum is not None:
        invalid = numeric <= minimum if strict_minimum else numeric < minimum
        if invalid:
            operator = "greater than" if strict_minimum else "at least"
            _error(path, f"must be {operator} {minimum}")
    if maximum is not None:
        invalid = numeric >= maximum if strict_maximum else numeric > maximum
        if invalid:
            operator = "less than" if strict_maximum else "at most"
            _error(path, f"must be {operator} {maximum}")
    return numeric


def _enum(value: object, path: str, choices: set[str] | frozenset[str]) -> str:
    text = _string(value, path)
    if text not in choices:
        _error(path, f"must be one of {sorted(choices)!r}")
    return text


def _jsonl_filename(value: object, path: str) -> str:
    filename = _string(value, path, nonempty=True)
    if (
        not filename.endswith(".jsonl")
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        _error(path, "must be a local .jsonl filename without directory components")
    return filename


def _sequence(value: object, path: str) -> Sequence[object]:
    return _COERCE.sequence(value, path)


def _bounds(value: object, path: str) -> tuple[float, float]:
    sequence = _sequence(value, path)
    if len(sequence) != 2:
        _error(path, "must contain exactly two finite endpoints")
    endpoints: list[float] = []
    for endpoint in sequence:
        endpoints.append(_finite_number(endpoint, path))
    lower, upper = endpoints
    if lower > upper:
        _error(path, "lower endpoint must not exceed upper endpoint")
    return lower, upper


def _positive_sequence_of_ints(value: object, path: str) -> None:
    sequence = _sequence(value, path)
    if not sequence:
        _error(path, "must not be empty")
    for index, item in enumerate(sequence):
        _int(item, f"{path}.{index}", minimum=1)


def _validate_compiled(value: object, spec: object, path: str) -> None:
    if isinstance(spec, _TypeSpec):
        if spec.expected is float:
            _number(value, path)
        elif spec.expected is int:
            _int(value, path)
        elif spec.expected is bool:
            _bool(value, path)
        else:
            _string(value, path)
        return
    if isinstance(spec, _EnumSpec):
        if not any(
            type(value) is type(candidate) and value == candidate
            for candidate in spec.values
        ):
            _error(path, f"must be one of {sorted(spec.values, key=repr)!r}")
        return
    if isinstance(spec, _BoundsSpec):
        _bounds(value, path)
        return
    if isinstance(spec, _SymmetricBoundsSpec):
        lower, upper = _bounds(value, path)
        if lower != -upper:
            _error(path, "must be symmetric bounds shaped exactly as [-v, v]")
        return
    if isinstance(spec, _IntegerSpec):
        _int(value, path, minimum=spec.minimum, maximum=spec.maximum)
        return
    if isinstance(spec, _NumberSpec):
        numeric = _number(value, path)
        if spec.minimum is not None:
            invalid = (
                numeric <= spec.minimum
                if spec.exclusive_minimum
                else numeric < spec.minimum
            )
            if invalid:
                operator = "greater than" if spec.exclusive_minimum else "at least"
                _error(path, f"must be {operator} {spec.minimum}")
        if spec.maximum is not None:
            invalid = (
                numeric >= spec.maximum
                if spec.exclusive_maximum
                else numeric > spec.maximum
            )
            if invalid:
                operator = "less than" if spec.exclusive_maximum else "at most"
                _error(path, f"must be {operator} {spec.maximum}")
        return
    if isinstance(spec, _SequenceSpec):
        sequence = _sequence(value, path)
        if len(sequence) < spec.min_length:
            _error(path, f"must contain at least {spec.min_length} items")
        if spec.max_length is not None and len(sequence) > spec.max_length:
            _error(path, f"must contain at most {spec.max_length} items")
        for index, item in enumerate(sequence):
            _validate_compiled(item, spec.item, f"{path}.{index}")
        return
    if isinstance(spec, _ObjectSpec):
        mapping = _closed(
            value,
            path=path,
            allowed=set(spec.fields),
            required=set(spec.fields) - set(spec.optional),
        )
        for key, child_spec in spec.fields.items():
            if key in mapping:
                _validate_compiled(mapping[key], child_spec, f"{path}.{key}")
        return
    if isinstance(spec, _TaggedSpec):
        mapping = _mapping(value, path)
        if spec.discriminator not in mapping:
            _error(f"{path}.{spec.discriminator}", "missing required discriminator")
        tag = _string(mapping[spec.discriminator], f"{path}.{spec.discriminator}")
        variant = spec.variants.get(tag)
        if variant is None:
            _error(
                f"{path}.{spec.discriminator}",
                f"unsupported discriminator {tag!r}",
            )
        fields = dict(variant.fields)
        fields[spec.discriminator] = _TypeSpec(str)
        _validate_compiled(
            mapping,
            _ObjectSpec(MappingProxyType(fields), variant.optional),
            path,
        )
        return
    if isinstance(spec, _UntaggedSpec):
        matches = 0
        for variant in spec.variants:
            try:
                _validate_compiled(value, variant, path)
            except ConfigSchemaError:
                continue
            matches += 1
        if matches != 1:
            _error(
                path, f"matched {matches} untagged union variants; expected exactly one"
            )
        return
    if isinstance(spec, _NullableSpec):
        if value is not None:
            _validate_compiled(value, spec.value, path)
        return
    if isinstance(spec, _StringMapSpec):
        mapping = _mapping(value, path)
        for key, item in mapping.items():
            _validate_compiled(item, spec.value, f"{path}.{key}")
        return
    raise AssertionError(f"unhandled compiled schema {type(spec).__name__}")


_DISTRIBUTION_KEYS = frozenset(DistributionConfig.model_fields)
_MLP_NETWORK_KEYS = field_names(MlpNetworkConfig)
_RESIDUAL_NETWORK_KEYS = field_names(ResidualNetworkConfig)
_ACTIVATIONS = {"elu", "relu", "tanh", "silu", "mish", "leaky_relu"}
_NORMALIZATIONS = {"none", "layer_norm"}


def _validate_distribution(value: object) -> None:
    distribution = _closed(
        value,
        path="models.actor.distribution",
        allowed=_DISTRIBUTION_KEYS,
        required=_DISTRIBUTION_KEYS - {"parameterization", "bound_loss"},
    )
    _enum(
        distribution["std"],
        "models.actor.distribution.std",
        {"parameter", "fixed", "network"},
    )
    parameterization = _enum(
        distribution.get("parameterization", "log"),
        "models.actor.distribution.parameterization",
        {"log", "scalar"},
    )
    if parameterization == "scalar" and distribution["std"] == "network":
        _error(
            "models.actor.distribution.parameterization",
            "scalar parameterization requires state-independent std",
        )
    bounds = _bounds(
        distribution["log_std_bounds"], "models.actor.distribution.log_std_bounds"
    )
    initial = _number(
        distribution["initial_log_std"], "models.actor.distribution.initial_log_std"
    )
    if not bounds[0] <= initial <= bounds[1]:
        _error(
            "models.actor.distribution.initial_log_std",
            "must lie within models.actor.distribution.log_std_bounds",
        )
    bound_mode = _enum(
        distribution["bound_mode"],
        "models.actor.distribution.bound_mode",
        {"clamp", "tanh"},
    )
    if parameterization == "scalar" and bound_mode != "clamp":
        _error(
            "models.actor.distribution.bound_mode",
            "scalar parameterization requires clamp",
        )
    if "bound_loss" in distribution:
        bound_loss = _closed(
            distribution["bound_loss"],
            path="models.actor.distribution.bound_loss",
            allowed=field_names(BoundLossConfig),
            required=field_names(BoundLossConfig),
        )
        _number(
            bound_loss["weight"],
            "models.actor.distribution.bound_loss.weight",
            minimum=0.0,
            strict_minimum=True,
        )
        margin = _number(
            bound_loss["margin"],
            "models.actor.distribution.bound_loss.margin",
            minimum=0.0,
        )
        if distribution["std"] != "network":
            _error(
                "models.actor.distribution.std",
                "must be 'network' when bound_loss is configured",
            )
        if bound_mode != "clamp":
            _error(
                "models.actor.distribution.bound_mode",
                "must be 'clamp' when bound_loss is configured",
            )
        if 2.0 * margin >= bounds[1] - bounds[0]:
            _error(
                "models.actor.distribution.bound_loss.margin",
                "must leave a non-empty interval inside log_std_bounds",
            )
    _enum(
        distribution["action_transform"],
        "models.actor.distribution.action_transform",
        {"none", "tanh"},
    )


def _validate_network(
    value: object, path: str, *, actor: bool, allow_output_initialization: bool = False
) -> Mapping[str, object]:
    raw = _mapping(value, path)
    if "architecture" not in raw:
        _error(f"{path}.architecture", "missing required key")
    architecture = _enum(
        raw["architecture"], f"{path}.architecture", {"mlp", "residual_mlp"}
    )
    allowed = set(
        _MLP_NETWORK_KEYS if architecture == "mlp" else _RESIDUAL_NETWORK_KEYS
    )
    required = set(allowed)
    if actor:
        allowed.add("distribution")
        required.add("distribution")
        if allow_output_initialization:
            allowed.add("output_initialization_std")
    mapping = _closed(raw, path=path, allowed=allowed, required=required)
    _positive_sequence_of_ints(mapping["layers"], f"{path}.layers")
    _enum(mapping["activation"], f"{path}.activation", _ACTIVATIONS)
    _enum(mapping["normalization"], f"{path}.normalization", _NORMALIZATIONS)
    if architecture == "residual_mlp":
        _number(
            mapping["residual_scale"],
            f"{path}.residual_scale",
            minimum=0,
            strict_minimum=True,
        )
    if actor:
        if "output_initialization_std" in mapping:
            _number(
                mapping["output_initialization_std"],
                f"{path}.output_initialization_std",
                minimum=0,
            )
        _validate_distribution(mapping["distribution"])
    return mapping


def _validate_models(value: object) -> Mapping[str, object]:
    models = _closed(
        value,
        path="models",
        allowed=field_names(ModelsConfig),
        required={"shared", "actor"},
    )
    _bool(models["shared"], "models.shared")
    _validate_network(
        models["actor"], "models.actor", actor=True, allow_output_initialization=True
    )
    if "critic" in models:
        _validate_network(models["critic"], "models.critic", actor=False)
    return models


def _validate_optimizer(value: object) -> None:
    optimizer = _mapping(value, "agent.optimizer")
    if "type" not in optimizer:
        _error("agent.optimizer.type", "missing required discriminator")
    optimizer_type = _enum(
        optimizer["type"], "agent.optimizer.type", schema_fields.OPTIMIZER_TYPES
    )
    fields = schema_fields.OPTIMIZER_FIELDS[optimizer_type]
    optimizer = _closed(
        optimizer, path="agent.optimizer", allowed=fields, required={"type"}
    )
    for key in ("weight_decay", "momentum", "eps"):
        if key in optimizer:
            _number(optimizer[key], f"agent.optimizer.{key}", minimum=0)
    if "betas" in optimizer:
        betas = _sequence(optimizer["betas"], "agent.optimizer.betas")
        if len(betas) != 2:
            _error("agent.optimizer.betas", "must contain exactly two values")
        for index, beta in enumerate(betas):
            _number(
                beta,
                f"agent.optimizer.betas.{index}",
                minimum=0,
                maximum=1,
                strict_maximum=True,
            )
    for key in ("amsgrad", "nesterov"):
        if key in optimizer:
            _bool(optimizer[key], f"agent.optimizer.{key}")
    if "adjust_lr_fn" in optimizer:
        _enum(
            optimizer["adjust_lr_fn"],
            "agent.optimizer.adjust_lr_fn",
            schema_fields.ADJUST_LR_FNS,
        )
    if "ns_steps" in optimizer:
        _int(optimizer["ns_steps"], "agent.optimizer.ns_steps", minimum=1)
    if "adamw" in optimizer:
        adamw = _closed(
            optimizer["adamw"],
            path="agent.optimizer.adamw",
            allowed=schema_fields.MUON_COMPANION_FIELDS,
        )
        for key in ("eps", "weight_decay"):
            if key in adamw:
                _number(adamw[key], f"agent.optimizer.adamw.{key}", minimum=0)
        if "betas" in adamw:
            betas = _sequence(adamw["betas"], "agent.optimizer.adamw.betas")
            if len(betas) != 2:
                _error("agent.optimizer.adamw.betas", "must contain exactly two values")
            for index, beta in enumerate(betas):
                _number(
                    beta,
                    f"agent.optimizer.adamw.betas.{index}",
                    minimum=0,
                    maximum=1,
                    strict_maximum=True,
                )
        if "amsgrad" in adamw:
            _bool(adamw["amsgrad"], "agent.optimizer.adamw.amsgrad")


def _validate_learning_rate(value: object) -> None:
    learning_rate = _closed(
        value,
        path="agent.learning_rate",
        allowed=field_names(agent_settings.PpoLearningRateConfig),
        required={"initial"},
    )
    initial = _number(
        learning_rate["initial"],
        "agent.learning_rate.initial",
        minimum=0,
        strict_minimum=True,
    )
    if "bounds" in learning_rate:
        bounds = _bounds(learning_rate["bounds"], "agent.learning_rate.bounds")
        if not bounds[0] <= initial <= bounds[1]:
            _error(
                "agent.learning_rate.initial",
                "must lie within agent.learning_rate.bounds",
            )
    if "scheduler" in learning_rate:
        scheduler = _mapping(
            learning_rate["scheduler"], "agent.learning_rate.scheduler"
        )
        if "type" not in scheduler:
            _error(
                "agent.learning_rate.scheduler.type",
                "missing required discriminator",
            )
        scheduler_type = _enum(
            scheduler["type"], "agent.learning_rate.scheduler.type", {"kl_adaptive"}
        )
        assert scheduler_type == "kl_adaptive"
        scheduler = _closed(
            scheduler,
            path="agent.learning_rate.scheduler",
            allowed=field_names(agent_settings.KlSchedulerConfig),
            required={"type", "threshold", "factor"},
        )
        _number(
            scheduler["threshold"],
            "agent.learning_rate.scheduler.threshold",
            minimum=0,
            strict_minimum=True,
        )
        _number(
            scheduler["factor"],
            "agent.learning_rate.scheduler.factor",
            minimum=1,
            strict_minimum=True,
        )
        if "update_mode" in scheduler:
            _enum(
                scheduler["update_mode"],
                "agent.learning_rate.scheduler.update_mode",
                {"per_update", "per_epoch", "per_minibatch"},
            )
    return


def _validate_agent(value: object) -> tuple[Mapping[str, object], str]:
    agent = _mapping(value, "agent")
    if "class" not in agent:
        _error("agent.class", "missing required discriminator")
    agent_class = _enum(agent["class"], "agent.class", {"PPO"})
    allowed = field_names(PpoAgentConfig)
    required = {
        "class",
        "rollouts",
        "discount_factor",
        "optimizer",
        "learning_rate",
        "learning_epochs",
        "mini_batches",
        "gae_lambda",
    }
    agent = _closed(agent, path="agent", allowed=allowed, required=required)
    _int(agent["rollouts"], "agent.rollouts", minimum=1)
    _number(agent["discount_factor"], "agent.discount_factor", minimum=0, maximum=1)
    _validate_optimizer(agent["optimizer"])
    _validate_learning_rate(agent["learning_rate"])
    if "time_limit_bootstrap" in agent:
        _bool(agent["time_limit_bootstrap"], "agent.time_limit_bootstrap")
    for key in ("state_preprocessor", "value_preprocessor"):
        if key in agent:
            _enum(
                agent[key],
                f"agent.{key}",
                {
                    "RunningStandardScaler",
                    "RslRlEmpiricalNormalization",
                    "none",
                },
            )
    if "reward_preprocessor" in agent:
        reward_preprocessor = agent["reward_preprocessor"]
        if isinstance(reward_preprocessor, Mapping):
            preprocessor = _closed(
                reward_preprocessor,
                path="agent.reward_preprocessor",
                allowed=field_names(agent_settings.ReturnScalerConfig),
                required={"type"},
            )
            _enum(
                preprocessor["type"],
                "agent.reward_preprocessor.type",
                {"RunningReturnScaler"},
            )
            for key in ("g_max", "eps"):
                if key in preprocessor:
                    _number(
                        preprocessor[key],
                        f"agent.reward_preprocessor.{key}",
                        minimum=0,
                        strict_minimum=True,
                    )
        else:
            _enum(
                reward_preprocessor,
                "agent.reward_preprocessor",
                {"RunningReturnScaler", "none"},
            )
    for key in (
        "grad_norm_clip",
        "ratio_clip",
        "value_clip",
        "entropy_loss_scale",
        "value_loss_scale",
    ):
        if key in agent:
            _number(agent[key], f"agent.{key}", minimum=0)
    if "activation_regularization" in agent:
        block = _closed(
            agent["activation_regularization"],
            path="agent.activation_regularization",
            allowed=field_names(agent_settings.ActivationRegularizationConfig),
            required={"weight", "target"},
        )
        _number(block["weight"], "agent.activation_regularization.weight", minimum=0)
        _number(block["target"], "agent.activation_regularization.target")
        if "latent_prior" in agent:
            _error(
                "agent.activation_regularization",
                "is not supported together with agent.latent_prior",
            )
    if "latent_prior" in agent:
        block = _closed(
            agent["latent_prior"],
            path="agent.latent_prior",
            allowed=field_names(agent_settings.LatentPriorConfig),
            required={"enabled"},
        )
        _bool(block["enabled"], "agent.latent_prior.enabled")
        if "latent_dim" in block:
            _int(block["latent_dim"], "agent.latent_prior.latent_dim", minimum=1)
        for key in ("kl_weight", "posterior_sample_alpha", "ar1_weight"):
            if key in block:
                _number(block[key], f"agent.latent_prior.{key}", minimum=0)
        if "ar1_phi" in block:
            _number(
                block["ar1_phi"],
                "agent.latent_prior.ar1_phi",
                minimum=-1,
                maximum=1,
            )
        for key in ("posterior_hidden_dims", "prior_hidden_dims"):
            if key in block:
                _positive_sequence_of_ints(block[key], f"agent.latent_prior.{key}")
        if "eval_mode" in block:
            _enum(
                block["eval_mode"],
                "agent.latent_prior.eval_mode",
                {"prior", "posterior"},
            )
        for key in ("use_full_obs_critic", "critic_use_detached_posterior_mu"):
            if key in block:
                _bool(block[key], f"agent.latent_prior.{key}")
        min_logvar = (
            _number(block["min_logvar"], "agent.latent_prior.min_logvar")
            if "min_logvar" in block
            else -20.0
        )
        max_logvar = (
            _number(block["max_logvar"], "agent.latent_prior.max_logvar")
            if "max_logvar" in block
            else 10.0
        )
        if min_logvar > max_logvar:
            _error(
                "agent.latent_prior.max_logvar",
                "must not be below agent.latent_prior.min_logvar",
            )
    if "initial_state" in agent:
        block = _closed(
            agent["initial_state"],
            path="agent.initial_state",
            allowed=field_names(agent_settings.InitialStateConfig),
            required=set(),
        )
        if "learning_rate" in block:
            _number(
                block["learning_rate"],
                "agent.initial_state.learning_rate",
                minimum=0,
                strict_minimum=True,
            )
        if "initial_log_std" in block:
            _number(block["initial_log_std"], "agent.initial_state.initial_log_std")
        if "log_std_bounds" in block:
            bounds = _bounds(
                block["log_std_bounds"], "agent.initial_state.log_std_bounds"
            )
            if bounds[0] >= bounds[1]:
                _error(
                    "agent.initial_state.log_std_bounds",
                    "must be strictly increasing",
                )
            if (
                "initial_log_std" in block
                and not bounds[0] <= block["initial_log_std"] <= bounds[1]
            ):
                _error(
                    "agent.initial_state.initial_log_std",
                    "must lie within agent.initial_state.log_std_bounds",
                )
        for key in ("ratio_clip", "reference_kl_weight"):
            if key in block:
                _number(block[key], f"agent.initial_state.{key}", minimum=0)
        if "max_grad_norm" in block:
            _number(
                block["max_grad_norm"],
                "agent.initial_state.max_grad_norm",
                minimum=0,
                strict_minimum=True,
            )
        if "baseline_ema" in block:
            _number(
                block["baseline_ema"],
                "agent.initial_state.baseline_ema",
                minimum=0,
                maximum=1,
                strict_maximum=True,
            )
        if "max_sample_age" in block:
            _int(
                block["max_sample_age"],
                "agent.initial_state.max_sample_age",
                minimum=0,
            )
    _int(agent["learning_epochs"], "agent.learning_epochs", minimum=1)
    _int(agent["mini_batches"], "agent.mini_batches", minimum=1)
    _number(agent["gae_lambda"], "agent.gae_lambda", minimum=0, maximum=1)
    if "shuffle_mode" in agent:
        _enum(
            agent["shuffle_mode"],
            "agent.shuffle_mode",
            {"per_update", "per_epoch", "off"},
        )
    if "parameter_initialization" in agent:
        _enum(
            agent["parameter_initialization"],
            "agent.parameter_initialization",
            {"default", "rsl_rl"},
        )
    for key in ("clip_predicted_values", "separate_grad_norm_clip"):
        if key in agent:
            _bool(agent[key], f"agent.{key}")
    if "value_loss_clip_mode" in agent:
        _enum(
            agent["value_loss_clip_mode"],
            "agent.value_loss_clip_mode",
            {"clamp", "max"},
        )
    if "kl_estimator" in agent:
        _enum(agent["kl_estimator"], "agent.kl_estimator", {"sampled", "analytic"})
    return agent, agent_class


_LEGACY_EVAL_METRICS = {
    "score",
    "success_rate",
    "reward_mean",
    "episode_length_mean",
}
_EVAL_THRESHOLD_KEYS = field_names(trainer_settings.EvaluationThresholdsConfig)
_EVAL_OBJECTIVE_THRESHOLD_KEYS = _EVAL_THRESHOLD_KEYS - {"consecutive_passes"}
_EVAL_LOGGING_STATISTICS = frozenset({"mean", "rate", "last", "count", "flag", "step"})


def _validate_benchmark(value: object) -> Mapping[str, object]:
    benchmark = _closed(
        value,
        path="trainer.benchmark",
        allowed=field_names(trainer_settings.BenchmarkConfig),
        required={"steps", "num_envs"},
    )
    has_interval = "interval" in benchmark
    has_seconds = "interval_seconds" in benchmark
    if has_interval == has_seconds:
        _error(
            "trainer.benchmark",
            "exactly one of interval or interval_seconds is required",
        )
    if has_interval:
        _int(benchmark["interval"], "trainer.benchmark.interval", minimum=1)
    else:
        _number(
            benchmark["interval_seconds"],
            "trainer.benchmark.interval_seconds",
            minimum=0,
            strict_minimum=True,
        )
    _int(benchmark["steps"], "trainer.benchmark.steps", minimum=1)
    _int(benchmark["num_envs"], "trainer.benchmark.num_envs", minimum=1)
    if "env_overrides" in benchmark:
        _mapping(
            benchmark["env_overrides"],
            "trainer.benchmark.env_overrides",
        )
    if "metric_frame" in benchmark:
        _enum(
            benchmark["metric_frame"],
            "trainer.benchmark.metric_frame",
            {"world", "translation", "heading", "root"},
        )
    if "best_metric" in benchmark:
        _enum(
            benchmark["best_metric"],
            "trainer.benchmark.best_metric",
            _LEGACY_EVAL_METRICS,
        )
    if "best_metric_tie_breakers" in benchmark:
        metrics = _sequence(
            benchmark["best_metric_tie_breakers"],
            "trainer.benchmark.best_metric_tie_breakers",
        )
        for index, metric in enumerate(metrics):
            _enum(
                metric,
                f"trainer.benchmark.best_metric_tie_breakers.{index}",
                _LEGACY_EVAL_METRICS,
            )
    if "metrics" in benchmark:
        metrics = _sequence(benchmark["metrics"], "trainer.benchmark.metrics")
        seen_metrics: set[str] = set()
        for index, metric in enumerate(metrics):
            name = _string(
                metric,
                f"trainer.benchmark.metrics.{index}",
                nonempty=True,
            )
            if name in seen_metrics:
                _error(
                    f"trainer.benchmark.metrics.{index}",
                    f"duplicate metric {name!r}",
                )
            seen_metrics.add(name)
    if "logging" in benchmark:
        logging_cfg = _closed(
            benchmark["logging"],
            path="trainer.benchmark.logging",
            allowed=field_names(trainer_settings.EvaluationLoggingConfig),
        )
        if "statistics" in logging_cfg:
            statistics = _sequence(
                logging_cfg["statistics"],
                "trainer.benchmark.logging.statistics",
            )
            seen_statistics: set[str] = set()
            for index, statistic in enumerate(statistics):
                name = _enum(
                    statistic,
                    f"trainer.benchmark.logging.statistics.{index}",
                    _EVAL_LOGGING_STATISTICS,
                )
                if name in seen_statistics:
                    _error(
                        f"trainer.benchmark.logging.statistics.{index}",
                        f"duplicate statistic {name!r}",
                    )
                seen_statistics.add(name)
        if "exclude" in logging_cfg:
            excluded = _sequence(
                logging_cfg["exclude"],
                "trainer.benchmark.logging.exclude",
            )
            seen_excluded: set[str] = set()
            for index, metric in enumerate(excluded):
                name = _string(
                    metric,
                    f"trainer.benchmark.logging.exclude.{index}",
                    nonempty=True,
                )
                if name in seen_excluded:
                    _error(
                        f"trainer.benchmark.logging.exclude.{index}",
                        f"duplicate metric {name!r}",
                    )
                seen_excluded.add(name)
    if "motion_groups" in benchmark:
        groups = _sequence(
            benchmark["motion_groups"], "trainer.benchmark.motion_groups"
        )
        if not groups:
            _error("trainer.benchmark.motion_groups", "must not be empty")
        seen_names: set[str] = set()
        seen_clip_indices: set[int] = set()
        for index, raw_group in enumerate(groups):
            path = f"trainer.benchmark.motion_groups.{index}"
            group = _closed(
                raw_group,
                path=path,
                allowed=field_names(trainer_settings.MotionGroupConfig),
                required={"name", "clip_index"},
            )
            name = _string(group["name"], f"{path}.name", nonempty=True)
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                _error(
                    f"{path}.name",
                    "must start with a letter and contain only letters, digits, and underscores",
                )
            clip_index = _int(group["clip_index"], f"{path}.clip_index", minimum=0)
            if name in seen_names:
                _error(f"{path}.name", f"duplicate motion group name {name!r}")
            if clip_index in seen_clip_indices:
                _error(
                    f"{path}.clip_index",
                    f"duplicate motion group clip_index {clip_index}",
                )
            seen_names.add(name)
            seen_clip_indices.add(clip_index)
    if "objective" in benchmark:
        if "best_metric" in benchmark:
            _error(
                "trainer.benchmark",
                "objective and best_metric are mutually exclusive",
            )
        objective = _closed(
            benchmark["objective"],
            path="trainer.benchmark.objective",
            allowed=field_names(trainer_settings.EvaluationObjectiveConfig),
            required={"metric", "direction"},
        )
        _string(
            objective["metric"],
            "trainer.benchmark.objective.metric",
            nonempty=True,
        )
        _enum(
            objective["direction"],
            "trainer.benchmark.objective.direction",
            {"maximize", "minimize"},
        )
    if "execution" in benchmark:
        _enum(
            benchmark["execution"],
            "trainer.benchmark.execution",
            {"inline", "overlap"},
        )
    for key in (
        "include_root_in_ape",
        "deterministic_reset",
        "require_full_horizon",
        "use_best_candidate",
        "save_history",
        "cuda_graph",
    ):
        if key in benchmark:
            _bool(benchmark[key], f"trainer.benchmark.{key}")
    if (
        benchmark.get("execution", "inline") == "overlap"
        and benchmark.get("cuda_graph", True) is False
    ):
        _error(
            "trainer.benchmark.cuda_graph",
            "must be true when execution is overlap",
        )
    if "episode_semantics" in benchmark:
        _enum(
            benchmark["episode_semantics"],
            "trainer.benchmark.episode_semantics",
            {"full_horizon", "terminal_success"},
        )
    if "termination" in benchmark:
        termination = _mapping(
            benchmark["termination"], "trainer.benchmark.termination"
        )
        if "type" not in termination:
            _error(
                "trainer.benchmark.termination.type", "missing required discriminator"
            )
        termination_type = _enum(
            termination["type"],
            "trainer.benchmark.termination.type",
            {"inherit", "disabled", "phc_tracking"},
        )
        allowed = (
            field_names(trainer_settings.EvaluationTerminationConfig)
            if termination_type == "phc_tracking"
            else field_names(trainer_settings.EvaluationTerminationConfig)
            - {"distance"}
        )
        required = allowed
        termination = _closed(
            termination,
            path="trainer.benchmark.termination",
            allowed=allowed,
            required=required,
        )
        if "distance" in termination:
            _number(
                termination["distance"],
                "trainer.benchmark.termination.distance",
                minimum=0,
                strict_minimum=True,
            )
    if "thresholds" in benchmark:
        thresholds = _closed(
            benchmark["thresholds"],
            path="trainer.benchmark.thresholds",
            allowed=_EVAL_THRESHOLD_KEYS,
        )
        for key in _EVAL_THRESHOLD_KEYS - {"success_rate", "consecutive_passes"}:
            if key in thresholds:
                _number(
                    thresholds[key], f"trainer.benchmark.thresholds.{key}", minimum=0
                )
        if "success_rate" in thresholds:
            _number(
                thresholds["success_rate"],
                "trainer.benchmark.thresholds.success_rate",
                minimum=0,
                maximum=1,
            )
        if "consecutive_passes" in thresholds:
            _int(
                thresholds["consecutive_passes"],
                "trainer.benchmark.thresholds.consecutive_passes",
                minimum=1,
            )
    if "history_filename" in benchmark:
        _jsonl_filename(
            benchmark["history_filename"], "trainer.benchmark.history_filename"
        )
    return benchmark


def _validate_clip_failure_benchmark(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        for removed, message in _REMOVED_CLIP_FAILURE_BENCHMARK_KEYS.items():
            if removed in value:
                _error(f"trainer.clip_failure_benchmark.{removed}", message)
    benchmark = _closed(
        value,
        path="trainer.clip_failure_benchmark",
        allowed=field_names(trainer_settings.ClipFailureBenchmarkConfig),
        required={"interval", "mode", "window_steps", "batch_size"},
    )
    for key in ("interval", "window_steps", "batch_size"):
        _int(benchmark[key], f"trainer.clip_failure_benchmark.{key}", minimum=1)
    if "anchor_stride" in benchmark:
        _int(
            benchmark["anchor_stride"],
            "trainer.clip_failure_benchmark.anchor_stride",
            minimum=1,
        )
    _enum(
        benchmark["mode"],
        "trainer.clip_failure_benchmark.mode",
        {"anchor", "full_motion"},
    )
    if "execution" in benchmark:
        _enum(
            benchmark["execution"],
            "trainer.clip_failure_benchmark.execution",
            {"auto", "inline", "subprocess", "shared_motion"},
        )
    for key in (
        "preserve_training_graphs",
        "clone_shared_motion_tensors",
        "save_history",
    ):
        if key in benchmark:
            _bool(benchmark[key], f"trainer.clip_failure_benchmark.{key}")
    if "clip_indices" in benchmark and benchmark["clip_indices"] is not None:
        clips = _sequence(
            benchmark["clip_indices"],
            "trainer.clip_failure_benchmark.clip_indices",
        )
        for index, clip in enumerate(clips):
            _int(
                clip,
                f"trainer.clip_failure_benchmark.clip_indices.{index}",
                minimum=0,
            )
    if "history_filename" in benchmark:
        _jsonl_filename(
            benchmark["history_filename"],
            "trainer.clip_failure_benchmark.history_filename",
        )
    if "timeout_seconds" in benchmark:
        _number(
            benchmark["timeout_seconds"],
            "trainer.clip_failure_benchmark.timeout_seconds",
            minimum=0,
            strict_minimum=True,
        )
    return benchmark


def _validate_trainer(value: object) -> Mapping[str, object]:
    trainer = _closed(
        value,
        path="trainer",
        allowed=field_names(trainer_settings.TrainerConfig),
        required={"max_iterations"},
    )
    _int(trainer["max_iterations"], "trainer.max_iterations", minimum=1)
    if "training_stage" in trainer:
        _enum(
            trainer["training_stage"],
            "trainer.training_stage",
            {"full", "musclenn_only"},
        )
    if "benchmark" in trainer:
        _validate_benchmark(trainer["benchmark"])
    if "clip_failure_benchmark" in trainer:
        _validate_clip_failure_benchmark(trainer["clip_failure_benchmark"])
    if "benchmark" in trainer and "clip_failure_benchmark" in trainer:
        _error(
            "trainer",
            "configure only one of benchmark and clip_failure_benchmark",
        )
    if "initialization" in trainer:
        initialization = _mapping(trainer["initialization"], "trainer.initialization")
        if "type" not in initialization:
            _error("trainer.initialization.type", "missing required discriminator")
        _enum(
            initialization["type"],
            "trainer.initialization.type",
            {"resume", "policy_init"},
        )
        initialization = _closed(
            initialization,
            path="trainer.initialization",
            allowed=field_names(trainer_settings.InitializationConfig),
            required={"type", "checkpoint"},
        )
        _string(
            initialization["checkpoint"],
            "trainer.initialization.checkpoint",
            nonempty=True,
        )
    if "stopping" in trainer:
        stopping = _mapping(trainer["stopping"], "trainer.stopping")
        if "type" not in stopping:
            _error("trainer.stopping.type", "missing required discriminator")
        stopping_type = _enum(
            stopping["type"],
            "trainer.stopping.type",
            {"reward", "episode_length", "benchmark", "time_budget"},
        )
        allowed = field_names(trainer_settings.StoppingConfig)
        if stopping_type not in {"reward", "episode_length"}:
            allowed -= {"threshold"}
        required = {"type"}
        if stopping_type in {"reward", "episode_length"}:
            required.add("threshold")
        if stopping_type == "time_budget":
            required.update({"warmup_iterations", "timeout_seconds"})
        stopping = _closed(
            stopping, path="trainer.stopping", allowed=allowed, required=required
        )
        if "threshold" in stopping:
            _number(stopping["threshold"], "trainer.stopping.threshold")
        if "warmup_iterations" in stopping:
            _int(
                stopping["warmup_iterations"],
                "trainer.stopping.warmup_iterations",
                minimum=0,
            )
        if "timeout_seconds" in stopping:
            _number(
                stopping["timeout_seconds"],
                "trainer.stopping.timeout_seconds",
                minimum=0,
                strict_minimum=True,
            )
        if "fail_on_miss" in stopping:
            _bool(stopping["fail_on_miss"], "trainer.stopping.fail_on_miss")
        if stopping_type == "time_budget":
            if int(stopping.get("warmup_iterations", 1)) != 0:
                _error(
                    "trainer.stopping",
                    "time_budget stopping requires warmup_iterations=0",
                )
            if bool(stopping.get("fail_on_miss", False)):
                _error(
                    "trainer.stopping",
                    "time_budget stopping cannot set fail_on_miss=true",
                )
        if stopping_type == "benchmark":
            benchmark = trainer.get("benchmark")
            if not isinstance(benchmark, Mapping):
                _error(
                    "trainer.stopping.type",
                    "benchmark stopping requires trainer.benchmark",
                )
            thresholds = benchmark.get("thresholds")
            if not isinstance(thresholds, Mapping) or not thresholds:
                _error(
                    "trainer.benchmark.thresholds",
                    "benchmark stopping requires nonempty thresholds",
                )
            if not any(key in thresholds for key in _EVAL_OBJECTIVE_THRESHOLD_KEYS):
                _error(
                    "trainer.benchmark.thresholds",
                    "benchmark stopping requires at least one objective metric threshold",
                )
    if "recording" in trainer:
        recording = _closed(
            trainer["recording"],
            path="trainer.recording",
            allowed=field_names(trainer_settings.RecordingConfig),
        )
        if "periodic" in recording:
            periodic = _closed(
                recording["periodic"],
                path="trainer.recording.periodic",
                allowed=field_names(trainer_settings.PeriodicRecordingConfig),
                required={"interval", "length"},
            )
            _int(
                periodic["interval"],
                "trainer.recording.periodic.interval",
                minimum=1,
            )
            _int(
                periodic["length"],
                "trainer.recording.periodic.length",
                minimum=1,
            )
        if "full_run" in recording:
            full_run = _closed(
                recording["full_run"],
                path="trainer.recording.full_run",
                allowed=field_names(trainer_settings.FullRunRecordingConfig),
            )
            if "encoder_backend" in full_run:
                _enum(
                    full_run["encoder_backend"],
                    "trainer.recording.full_run.encoder_backend",
                    {"auto", "imageio", "pynvcodec"},
                )
            if "codec" in full_run:
                _enum(
                    full_run["codec"],
                    "trainer.recording.full_run.codec",
                    {"h264"},
                )
            for key in ("tile_envs", "tile_cols"):
                if key in full_run:
                    _int(
                        full_run[key],
                        f"trainer.recording.full_run.{key}",
                        minimum=1,
                    )
            if "tile_cols" in full_run and "tile_envs" not in full_run:
                _error(
                    "trainer.recording.full_run.tile_cols",
                    "requires trainer.recording.full_run.tile_envs",
                )
            if (
                "tile_cols" in full_run
                and full_run["tile_cols"] > full_run["tile_envs"]
            ):
                _error(
                    "trainer.recording.full_run.tile_cols",
                    "must not exceed trainer.recording.full_run.tile_envs",
                )
            if "camera" in full_run:
                camera = _closed(
                    full_run["camera"],
                    path="trainer.recording.full_run.camera",
                    allowed=field_names(play_settings.CameraConfig),
                )
                for key in ("azimuth_deg", "elevation_deg"):
                    if key in camera:
                        _number(
                            camera[key],
                            f"trainer.recording.full_run.camera.{key}",
                        )
                if "distance" in camera:
                    _number(
                        camera["distance"],
                        "trainer.recording.full_run.camera.distance",
                        minimum=0,
                        strict_minimum=True,
                    )
            if "benchmark" in full_run:
                benchmark = _closed(
                    full_run["benchmark"],
                    path="trainer.recording.full_run.benchmark",
                    allowed=field_names(play_settings.RecordingBenchmarkConfig),
                )
                if "warmup_frames" in benchmark:
                    _int(
                        benchmark["warmup_frames"],
                        "trainer.recording.full_run.benchmark.warmup_frames",
                        minimum=0,
                    )
            if "warp" in full_run:
                warp = _closed(
                    full_run["warp"],
                    path="trainer.recording.full_run.warp",
                    allowed=field_names(play_settings.TrainingWarpVideoConfig),
                )
                if "use_textures" in warp:
                    _bool(
                        warp["use_textures"],
                        "trainer.recording.full_run.warp.use_textures",
                    )
                if "floor_shader_mode" in warp:
                    floor_shader_mode = _string(
                        warp["floor_shader_mode"],
                        "trainer.recording.full_run.warp.floor_shader_mode",
                    )
                    if floor_shader_mode == "procedural_grid":
                        _error(
                            "trainer.recording.full_run.warp.floor_shader_mode",
                            "'procedural_grid' is no longer supported; use 'texture'",
                        )
                    if floor_shader_mode != "texture":
                        _error(
                            "trainer.recording.full_run.warp.floor_shader_mode",
                            "must be 'texture'",
                        )
                if "render_block_dim" in warp:
                    _int(
                        warp["render_block_dim"],
                        "trainer.recording.full_run.warp.render_block_dim",
                        minimum=1,
                    )
                if "lbs_mesh_mode" in warp:
                    _enum(
                        warp["lbs_mesh_mode"],
                        "trainer.recording.full_run.warp.lbs_mesh_mode",
                        {"off", "skin", "muscle"},
                    )
                for key in ("lbs_mesh_bundle", "lbs_activation_mapping"):
                    if key in warp and warp[key] is not None:
                        _string(
                            warp[key],
                            f"trainer.recording.full_run.warp.{key}",
                            nonempty=True,
                        )
                if "lbs_mesh_shading_mode" in warp:
                    _enum(
                        warp["lbs_mesh_shading_mode"],
                        "trainer.recording.full_run.warp.lbs_mesh_shading_mode",
                        {"flat", "smooth"},
                    )
                if "activation_log_stats" in warp:
                    _bool(
                        warp["activation_log_stats"],
                        "trainer.recording.full_run.warp.activation_log_stats",
                    )
    if "post_training_playback" in trainer:
        playback = _closed(
            trainer["post_training_playback"],
            path="trainer.post_training_playback",
            allowed=field_names(trainer_settings.PostTrainingPlaybackConfig),
        )
        if "video" in playback:
            _bool(playback["video"], "trainer.post_training_playback.video")
        if "video_length" in playback:
            _int(
                playback["video_length"],
                "trainer.post_training_playback.video_length",
                minimum=1,
            )
        if "use_cuda_graph" in playback:
            _bool(
                playback["use_cuda_graph"],
                "trainer.post_training_playback.use_cuda_graph",
            )
        if "subprocess_timeout_seconds" in playback:
            _number(
                playback["subprocess_timeout_seconds"],
                "trainer.post_training_playback.subprocess_timeout_seconds",
                minimum=0,
                strict_minimum=True,
            )
    return trainer


def _validate_logger(value: object) -> None:
    logger = _mapping(value, "experiment.logger")
    if "type" not in logger:
        _error("experiment.logger.type", "missing required discriminator")
    logger_type = _enum(
        logger["type"], "experiment.logger.type", schema_fields.LOGGER_TYPES
    )
    fields = schema_fields.LOGGER_FIELDS[logger_type]
    logger = _closed(
        logger, path="experiment.logger", allowed=fields, required={"type"}
    )
    for key in fields - {"type", "tags"}:
        if key in logger:
            _string(logger[key], f"experiment.logger.{key}", nonempty=True)
    if "tags" in logger:
        tags = _mapping(logger["tags"], "experiment.logger.tags")
        for key, item in tags.items():
            if key == schema_fields.BACKEND_GROUP_FIELD:
                _error(
                    f"experiment.logger.tags.{schema_fields.BACKEND_GROUP_FIELD}",
                    "use backend-independent experiment.group",
                )
            _string(key, "experiment.logger.tags", nonempty=True)
            _string(item, f"experiment.logger.tags.{key}")


def _validate_experiment(value: object) -> None:
    experiment = _closed(
        value,
        path="experiment",
        allowed=field_names(experiment_settings.ExperimentConfig),
        required={"directory", "logger"},
    )
    _string(experiment["directory"], "experiment.directory", nonempty=True)
    if "name" in experiment:
        _string(experiment["name"], "experiment.name")
    if "group" in experiment:
        _string(experiment["group"], "experiment.group", nonempty=True)
    _validate_logger(experiment["logger"])
    for key in (
        "write_interval",
        "episode_metrics_interval",
        "writer_max_queue",
        "writer_flush_secs",
    ):
        if key in experiment:
            _int(experiment[key], f"experiment.{key}", minimum=1)
    if "checkpoint" in experiment:
        checkpoint = _closed(
            experiment["checkpoint"],
            path="experiment.checkpoint",
            allowed=field_names(experiment_settings.CheckpointConfig),
        )
        if "disabled" in checkpoint:
            _bool(checkpoint["disabled"], "experiment.checkpoint.disabled")
        if "replay" in checkpoint:
            _bool(checkpoint["replay"], "experiment.checkpoint.replay")
        if "periodic" in checkpoint:
            periodic = _closed(
                checkpoint["periodic"],
                path="experiment.checkpoint.periodic",
                allowed=field_names(experiment_settings.PeriodicCheckpointConfig),
                required={"interval"},
            )
            _int(
                periodic["interval"],
                "experiment.checkpoint.periodic.interval",
                minimum=1,
            )
        if "best" in checkpoint:
            best = _mapping(checkpoint["best"], "experiment.checkpoint.best")
            if "source" not in best:
                _error(
                    "experiment.checkpoint.best.source",
                    "missing required discriminator",
                )
            source = _enum(
                best["source"],
                "experiment.checkpoint.best.source",
                {"auto", "training", "benchmark"},
            )
            allowed = (
                field_names(experiment_settings.BestCheckpointConfig) - {"metric"}
                if source == "auto"
                else field_names(experiment_settings.BestCheckpointConfig)
            )
            required = allowed
            best = _closed(
                best,
                path="experiment.checkpoint.best",
                allowed=allowed,
                required=required,
            )
            _int(
                best["interval"],
                "experiment.checkpoint.best.interval",
                minimum=1,
            )
            if source == "training":
                _enum(
                    best["metric"],
                    "experiment.checkpoint.best.metric",
                    {"episode_reward_mean"},
                )
            elif source == "benchmark":
                _string(
                    best["metric"],
                    "experiment.checkpoint.best.metric",
                    nonempty=True,
                )


_RUNTIME_RULE_MESSAGES = {
    schema_fields.RULE_TORCH_GRAPH_CONFLICT: (
        "must not enable Torch-managed CUDA graphs when external CUDA graphs are enabled; use 'default' or 'max-autotune-no-cudagraphs'"
    ),
}

_REMOVED_RUNTIME_KEYS = {
    "compile_update_fullgraph": (
        "was removed; full update compilation is implied by runtime.compile "
        "together with runtime.update_cuda_graph"
    ),
    "shuffle_cuda_graph": (
        "was removed; PPO captures its minibatch shuffle inside the update "
        "graph whenever runtime.update_cuda_graph=true"
    ),
    "compile_act_into": (
        "was removed; runtime.compile compiles the rollout act_into path "
        "whenever compilation is enabled"
    ),
}


_REMOVED_ENV_KEYS = {
    "compile_mjw_step": (
        "was removed; the repo-owned MuJoCo Warp step suffix is always "
        "captured as a CUDA graph whenever env.use_cuda_graph is off"
    ),
}


_REMOVED_CLIP_FAILURE_BENCHMARK_KEYS = {
    "cuda_graph": (
        "was removed; the clip-failure benchmark inherits env.use_cuda_graph "
        "from the run"
    ),
    "compile": (
        "was removed; the clip-failure benchmark inherits runtime.compile from the run"
    ),
}


def _validate_runtime(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        for removed, message in _REMOVED_RUNTIME_KEYS.items():
            if removed in value:
                _error(f"runtime.{removed}", message)
    runtime = _closed(value, path="runtime", allowed=schema_fields.RUNTIME_FIELDS)
    try:
        RuntimeConfig.from_mapping(runtime)
    except ValidationError as exc:
        error = exc.errors(include_url=False)[0]
        path = ".".join(("runtime", *(str(part) for part in error["loc"])))
        raise ConfigSchemaError(path, error["msg"]) from exc
    for path, rule in schema_fields.runtime_rule_violations(runtime):
        _error(path, _RUNTIME_RULE_MESSAGES[rule])
    return runtime


def _validate_play_action(value: object) -> None:
    action = _closed(
        value,
        path="play.action",
        allowed=field_names(play_settings.PlayActionConfig),
        required={"mode"},
    )
    _enum(action["mode"], "play.action.mode", {"mean", "sample"})


def _validate_play_tracking(value: object) -> None:
    tracking = _closed(
        value,
        path="play.tracking",
        allowed=field_names(play_settings.PlayTrackingConfig),
        required={"mode"},
    )
    _enum(tracking["mode"], "play.tracking.mode", {"reference", "free_run"})
    if "metric_frame" in tracking:
        _enum(tracking["metric_frame"], "play.tracking.metric_frame", {"root", "world"})
    for key in (
        "include_root_in_ape",
        "require_full_horizon",
        "deterministic_reset",
    ):
        if key in tracking:
            _bool(tracking[key], f"play.tracking.{key}")
    for key in ("motion_index", "start_frame"):
        if key in tracking and tracking[key] is not None:
            _int(tracking[key], f"play.tracking.{key}", minimum=0)
    if "termination" in tracking:
        _enum(
            tracking["termination"],
            "play.tracking.termination",
            {"inherit", "disabled"},
        )


def _validate_play_video_warp(value: object) -> None:
    warp = _closed(
        value,
        path="play.video.warp",
        allowed=field_names(play_settings.WarpVideoConfig),
    )
    if "use_textures" in warp:
        _bool(warp["use_textures"], "play.video.warp.use_textures")
    if "floor_shader_mode" in warp:
        floor_shader_mode = _string(
            warp["floor_shader_mode"],
            "play.video.warp.floor_shader_mode",
        )
        if floor_shader_mode == "procedural_grid":
            _error(
                "play.video.warp.floor_shader_mode",
                "'procedural_grid' is no longer supported; use 'texture'",
            )
        if floor_shader_mode != "texture":
            _error("play.video.warp.floor_shader_mode", "must be 'texture'")
    if "render_block_dim" in warp:
        _int(warp["render_block_dim"], "play.video.warp.render_block_dim", minimum=1)
    if "lbs_mesh_mode" in warp:
        _enum(
            warp["lbs_mesh_mode"],
            "play.video.warp.lbs_mesh_mode",
            {"off", "skin", "muscle"},
        )
    if "lbs_mesh_bundle" in warp and warp["lbs_mesh_bundle"] is not None:
        _string(
            warp["lbs_mesh_bundle"], "play.video.warp.lbs_mesh_bundle", nonempty=True
        )
    if "lbs_mesh_shading_mode" in warp:
        _enum(
            warp["lbs_mesh_shading_mode"],
            "play.video.warp.lbs_mesh_shading_mode",
            {"flat", "smooth"},
        )
    if "lbs_activation_mapping" in warp and warp["lbs_activation_mapping"] is not None:
        _string(
            warp["lbs_activation_mapping"],
            "play.video.warp.lbs_activation_mapping",
            nonempty=True,
        )


def _validate_play(value: object) -> None:
    play = _closed(
        value,
        path="play",
        allowed=field_names(play_settings.PlayConfig),
        required={"num_envs", "steps", "action"},
    )
    num_envs = _int(play["num_envs"], "play.num_envs", minimum=1)
    if play["steps"] is not None:
        _int(play["steps"], "play.steps", minimum=1)
    if "checkpoint" in play:
        _string(play["checkpoint"], "play.checkpoint", nonempty=True)
    if "seed" in play and play["seed"] is not None:
        _int(play["seed"], "play.seed", minimum=-1)
    if "time_limit_policy" in play:
        _enum(
            play["time_limit_policy"], "play.time_limit_policy", {"extend", "respect"}
        )
    if "done_policy" in play:
        _enum(play["done_policy"], "play.done_policy", {"reset_continue", "stop"})
    if "use_cuda_graph" in play:
        _bool(play["use_cuda_graph"], "play.use_cuda_graph")
    if "allow_eager_fallback" in play:
        _bool(play["allow_eager_fallback"], "play.allow_eager_fallback")
    _validate_play_action(play["action"])
    if "tracking" in play:
        _validate_play_tracking(play["tracking"])
    if "viewer" in play:
        viewer = _closed(
            play["viewer"],
            path="play.viewer",
            allowed=field_names(play_settings.PlayViewerConfig),
        )
        backend = (
            _enum(viewer["backend"], "play.viewer.backend", {"mujoco", "viser"})
            if "backend" in viewer
            else "mujoco"
        )
        if "resolution" in viewer:
            resolution = _sequence(viewer["resolution"], "play.viewer.resolution")
            if len(resolution) != 2:
                _error("play.viewer.resolution", "must contain width and height")
            for index, item in enumerate(resolution):
                _int(item, f"play.viewer.resolution.{index}", minimum=1)
        if "port" in viewer:
            _int(viewer["port"], "play.viewer.port", minimum=1, maximum=65535)
            if backend != "viser":
                _error(
                    "play.viewer.port",
                    "is valid only when play.viewer.backend is 'viser'",
                )
        if "reset_on_done" in viewer:
            _bool(viewer["reset_on_done"], "play.viewer.reset_on_done")
    if "video" in play:
        video = _closed(
            play["video"],
            path="play.video",
            allowed=field_names(play_settings.PlayVideoConfig),
        )
        if "enabled" in video:
            _bool(video["enabled"], "play.video.enabled")
        for key in ("tile_envs", "tile_cols", "overlay_panel_width"):
            if key in video:
                _int(video[key], f"play.video.{key}", minimum=1)
        if "length" in video:
            length = video["length"]
            if isinstance(length, str):
                _enum(length, "play.video.length", {"episode"})
            else:
                _int(length, "play.video.length", minimum=1)
        if "render_backend" in video:
            _enum(
                video["render_backend"],
                "play.video.render_backend",
                {"auto", "warp", "mujoco_gl"},
            )
        if "encoder_backend" in video:
            _enum(
                video["encoder_backend"],
                "play.video.encoder_backend",
                {"auto", "imageio", "pynvcodec"},
            )
        if "codec" in video:
            _enum(video["codec"], "play.video.codec", {"h264"})
        if "overlay_rewards" in video:
            _bool(video["overlay_rewards"], "play.video.overlay_rewards")
        if "overlay_history_seconds" in video:
            _number(
                video["overlay_history_seconds"],
                "play.video.overlay_history_seconds",
                minimum=0,
            )
        if "remux_timeout_seconds" in video:
            _number(
                video["remux_timeout_seconds"],
                "play.video.remux_timeout_seconds",
                minimum=0,
                strict_minimum=True,
            )
        if "warp" in video:
            _validate_play_video_warp(video["warp"])
            if video.get("render_backend", "warp") not in {"auto", "warp"}:
                _error(
                    "play.video.warp",
                    "is valid only with play.video.render_backend 'auto' or 'warp'",
                )
        if "tile_envs" in video and video["tile_envs"] != num_envs:
            _error("play.video.tile_envs", "must equal play.num_envs when provided")
        if "tile_cols" in video and "tile_envs" not in video:
            _error("play.video.tile_cols", "requires play.video.tile_envs")
        if "tile_cols" in video and video["tile_cols"] > video["tile_envs"]:
            _error("play.video.tile_cols", "must not exceed play.video.tile_envs")
    if "camera" in play:
        camera = _closed(
            play["camera"],
            path="play.camera",
            allowed=field_names(play_settings.PlayCameraConfig),
        )
        for key in ("azimuth_deg", "elevation_deg"):
            if key in camera:
                _number(camera[key], f"play.camera.{key}")
        if "distance" in camera:
            _number(
                camera["distance"],
                "play.camera.distance",
                minimum=0,
                strict_minimum=True,
            )
        if "lookat_ema_alpha" in camera:
            _number(
                camera["lookat_ema_alpha"],
                "play.camera.lookat_ema_alpha",
                minimum=0,
                maximum=1,
            )
        if "views" in camera:
            scalar_camera_fields = {
                "azimuth_deg",
                "elevation_deg",
                "distance",
            } & set(camera)
            if scalar_camera_fields:
                _error(
                    "play.camera.views",
                    "is mutually exclusive with scalar camera fields",
                )
            views = _closed(
                camera["views"],
                path="play.camera.views",
                allowed=field_names(play_settings.CameraViewsConfig),
                required={"cameras"},
            )
            if "layout" in views:
                _enum(
                    views["layout"],
                    "play.camera.views.layout",
                    {"horizontal", "vertical"},
                )
            if "resolution" in views:
                resolution = _sequence(
                    views["resolution"], "play.camera.views.resolution"
                )
                if len(resolution) != 2:
                    _error(
                        "play.camera.views.resolution",
                        "must contain width and height",
                    )
                for index, item in enumerate(resolution):
                    _int(item, f"play.camera.views.resolution.{index}", minimum=1)
            cameras = _sequence(views["cameras"], "play.camera.views.cameras")
            if not cameras:
                _error("play.camera.views.cameras", "must not be empty")
            for index, item in enumerate(cameras):
                path = f"play.camera.views.cameras.{index}"
                view = _closed(
                    item,
                    path=path,
                    allowed=field_names(play_settings.CameraViewConfig),
                )
                if "lookat" in view:
                    lookat = _sequence(view["lookat"], f"{path}.lookat")
                    if len(lookat) != 3:
                        _error(f"{path}.lookat", "must contain exactly three values")
                    for axis, value in enumerate(lookat):
                        _number(value, f"{path}.lookat.{axis}")
                if "track_bodies" in view:
                    bodies = _sequence(view["track_bodies"], f"{path}.track_bodies")
                    if not bodies:
                        _error(f"{path}.track_bodies", "must not be empty")
                    for body_index, name in enumerate(bodies):
                        _string(
                            name, f"{path}.track_bodies.{body_index}", nonempty=True
                        )
                if "lookat_offset" in view:
                    offset = _sequence(view["lookat_offset"], f"{path}.lookat_offset")
                    if len(offset) != 3:
                        _error(
                            f"{path}.lookat_offset", "must contain exactly three values"
                        )
                    for axis, value in enumerate(offset):
                        _number(value, f"{path}.lookat_offset.{axis}")
                    if not view.get("track_bodies"):
                        _error(f"{path}.lookat_offset", "requires track_bodies")
                for key in ("azimuth_deg", "elevation_deg"):
                    if key in view:
                        _number(view[key], f"{path}.{key}")
                if "distance" in view:
                    _number(
                        view["distance"],
                        f"{path}.distance",
                        minimum=0,
                        strict_minimum=True,
                    )
            if num_envs != 1:
                _error("play.camera.views", "requires play.num_envs=1")
            video = play.get("video", {})
            if (
                not isinstance(video, Mapping)
                or video.get("render_backend") != "mujoco_gl"
            ):
                _error(
                    "play.camera.views",
                    "requires play.video.render_backend='mujoco_gl'",
                )
            if bool(video.get("overlay_rewards", False)):
                _error("play.camera.views", "is incompatible with reward overlays")
    if "benchmark" in play:
        benchmark = _closed(
            play["benchmark"],
            path="play.benchmark",
            allowed=field_names(play_settings.PlayBenchmarkConfig),
        )
        if "warmup_frames" in benchmark:
            _int(benchmark["warmup_frames"], "play.benchmark.warmup_frames", minimum=0)
        if "stage_sync" in benchmark:
            _bool(benchmark["stage_sync"], "play.benchmark.stage_sync")
    if "output" in play:
        output = _closed(
            play["output"],
            path="play.output",
            allowed=field_names(play_settings.PlayOutputConfig),
        )
        if "directory" in output and output["directory"] is not None:
            _string(output["directory"], "play.output.directory", nonempty=True)
        if "emit_result_payload" in output:
            _bool(output["emit_result_payload"], "play.output.emit_result_payload")


def validate_semantic_mapping(values: Mapping[str, ThawedValue]) -> None:
    """Validate one resolved production schema-version-1 mapping."""
    from instantmimic.rl.models.musubi_actor_critic import reject_unavailable_selectors

    reject_unavailable_selectors(values)
    root = _closed(values, path="", allowed=_ROOT_KEYS, required=_ROOT_KEYS)
    _int(root["schema_version"], "schema_version")
    _int(root["seed"], "seed", minimum=-1)
    _string(root["task"], "task", nonempty=True)
    env = _mapping(root["env"], "env")
    for removed, message in _REMOVED_ENV_KEYS.items():
        if removed in env:
            _error(f"env.{removed}", message)
    observations = env.get("observations")
    if isinstance(observations, Mapping) and "muscle_geometry" in observations:
        _bool(observations["muscle_geometry"], "env.observations.muscle_geometry")
    agent, _ = _validate_agent(root["agent"])
    scene = env.get("scene")
    raw_num_envs = (
        scene.get("num_envs") if isinstance(scene, Mapping) else env.get("num_envs")
    )
    if raw_num_envs is not None:
        num_envs_path = (
            "env.scene.num_envs" if isinstance(scene, Mapping) else "env.num_envs"
        )
        num_envs = _int(raw_num_envs, num_envs_path, minimum=1)
        transition_count = int(agent["rollouts"]) * num_envs
        mini_batches = int(agent["mini_batches"])
        if transition_count % mini_batches != 0:
            _error(
                "agent.mini_batches",
                "must divide agent.rollouts * environment count exactly "
                f"({transition_count} transitions)",
            )

    models = _validate_models(root["models"])
    if models["shared"] and "critic" in models:
        _error("models.critic", "must be omitted when models.shared is true for PPO")
    if not models["shared"] and "critic" not in models:
        _error("models.critic", "is required when models.shared is false for PPO")
    _validate_trainer(root["trainer"])
    trainer = _mapping(root["trainer"], "trainer")
    clip_failure_benchmark = trainer.get("clip_failure_benchmark")
    if (
        isinstance(clip_failure_benchmark, Mapping)
        and clip_failure_benchmark.get("execution") == "inline"
        and env.get("use_cuda_graph") is True
    ):
        _error(
            "trainer.clip_failure_benchmark.execution",
            "inline execution conflicts with env.use_cuda_graph=true; use auto, "
            "subprocess, or shared_motion",
        )
    _validate_experiment(root["experiment"])
    experiment = _mapping(root["experiment"], "experiment")
    checkpoint = experiment.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        best = checkpoint.get("best")
        if isinstance(best, Mapping) and best.get("source") == "benchmark":
            trainer = _mapping(root["trainer"], "trainer")
            benchmark = trainer.get("benchmark")
            clip_failure_benchmark = trainer.get("clip_failure_benchmark")
            if not isinstance(benchmark, Mapping) and not isinstance(
                clip_failure_benchmark, Mapping
            ):
                _error(
                    "experiment.checkpoint.best.source",
                    "benchmark source requires trainer.benchmark or "
                    "trainer.clip_failure_benchmark",
                )
            metric = best.get("metric")
            if (
                isinstance(clip_failure_benchmark, Mapping)
                and metric != "fixed_horizon_reward_mean"
            ):
                _error(
                    "experiment.checkpoint.best.metric",
                    "clip_failure benchmark requires fixed_horizon_reward_mean",
                )
            if isinstance(benchmark, Mapping) and metric not in _LEGACY_EVAL_METRICS:
                objective = benchmark.get("objective")
                if not isinstance(objective, Mapping) or metric != objective.get(
                    "metric"
                ):
                    _error(
                        "experiment.checkpoint.best.metric",
                        "environment metric requires a matching "
                        "trainer.benchmark.objective with explicit direction",
                    )
    _validate_runtime(root["runtime"])
    _validate_play(root["play"])
