"""Pure schema-v1 lowering for RL model, optimizer, and agent construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, NoReturn

from instantmimic.config import agent_config as agent_settings
from instantmimic.config import coerce, schema_fields
from instantmimic.config.agent_config import PpoAgentConfig
from instantmimic.config.model_config import (
    ACTOR_ADAPTER,
    CRITIC_ADAPTER,
    MlpNetworkConfig,
    ModelsConfig,
    ResidualNetworkConfig,
)
from instantmimic.config.policy_config import DistributionConfig, parse_optimizer
from instantmimic.config.runtime_config import RuntimeConfig
from instantmimic.config.settings import field_names
from instantmimic.config.validation import (
    _validate_distribution,
    _validate_network,
    _validate_optimizer,
)

type PlainScalar = str | int | float | bool | None
type PlainValue = PlainScalar | tuple["PlainValue", ...] | Mapping[str, "PlainValue"]


_MODEL_KEYS = field_names(ModelsConfig)
_MLP_ROLE_KEYS = field_names(MlpNetworkConfig)
_RESIDUAL_ROLE_KEYS = field_names(ResidualNetworkConfig)
_PPO_AGENT_KEYS = field_names(PpoAgentConfig)
_MUON_COMPANION = {
    "betas": (0.9, 0.999),
    "eps": 2.0e-5,
    "weight_decay": 3.0e-5,
    "amsgrad": False,
}


def _freeze_plain(value: object, *, path: str) -> PlainValue:
    if isinstance(value, Mapping):
        frozen: dict[str, PlainValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_plain(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_plain(item, path=f"{path}.{index}")
            for index, item in enumerate(value)
        )
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(
        f"{path} must contain plain Python values, got {type(value).__name__}"
    )


def _frozen_mapping(
    values: Mapping[str, object], *, path: str
) -> Mapping[str, PlainValue]:
    frozen = _freeze_plain(values, path=path)
    assert isinstance(frozen, Mapping)
    return frozen


# This module words the shared failures as "<path> must be ..." and raises
# TypeError for all of them, where the config surfaces raise their own error
# with the phrase as written.
_MESSAGES = {
    coerce.BOOLEAN: "must be a boolean",
    coerce.INTEGER: "must be an integer",
    coerce.NUMBER: "must be numeric",
    coerce.FINITE: "must be finite",
    coerce.STRING: "must be a string",
    coerce.EMPTY: "must not be empty",
    coerce.MAPPING: "must be a mapping",
    coerce.MAPPING_KEYS: "must have string keys",
    coerce.SEQUENCE: "must be a sequence",
}


def _fail(path: str, phrase: str, cause: BaseException | None = None) -> NoReturn:
    coerce.raise_translated(TypeError(f"{path} {_MESSAGES[phrase]}"), cause)


_COERCE = coerce.Coercions(_fail)


def _mapping(value: object, path: str) -> Mapping[str, object]:
    return _COERCE.mapping(value, path)


def _root(config: object, name: str) -> Mapping[str, object]:
    if isinstance(config, Mapping):
        if name not in config:
            raise ValueError(f"missing required {name} root")
        return _mapping(config[name], name)
    try:
        value = getattr(config, name)
    except AttributeError as exc:
        raise ValueError(f"missing required {name} root") from exc
    return _mapping(value, name)


def _closed(
    value: object,
    *,
    path: str,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    mapping = _mapping(value, path)
    unknown, missing = coerce.closed_keys(mapping, allowed, required)
    if unknown:
        raise ValueError(f"unknown {path} fields: {sorted(unknown)!r}")
    if missing:
        raise ValueError(f"{path} is missing required fields: {list(missing)!r}")
    return mapping


def _bool(value: object, path: str) -> bool:
    return _COERCE.boolean(value, path)


def _int(value: object, path: str) -> int:
    return _COERCE.integer(value, path)


def _float(value: object, path: str) -> float:
    return float(_COERCE.finite(value, path))


def _string(value: object, path: str) -> str:
    return _COERCE.string(value, path)


def _sequence(value: object, path: str) -> Sequence[object]:
    return _COERCE.sequence(value, path)


def _pair(value: object, path: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{path} must contain exactly two numbers")
    return _float(value[0], f"{path}.0"), _float(value[1], f"{path}.1")


def _int_tuple(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path} must be a sequence")
    return tuple(_int(item, f"{path}.{index}") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class DistributionBuildSpec:
    """Immutable diagonal-Gaussian distribution semantics and constructor flags."""

    std: Literal["parameter", "fixed", "network"]
    parameterization: Literal["log", "scalar"]
    fixed_sigma: bool
    fixed_log_std: bool
    initial_log_std: float
    log_std_bounds: tuple[float, float]
    bound_mode: Literal["clamp", "tanh"]
    action_transform: Literal["none", "tanh"]
    constructor_kwargs: Mapping[str, PlainValue]


@dataclass(frozen=True, slots=True)
class ModelRoleBuildSpec:
    """One actor or critic role with detached constructor values."""

    role: Literal["actor", "critic"]
    architecture: Literal["mlp", "residual_mlp"]
    constructor_kwargs: Mapping[str, PlainValue]
    distribution: DistributionBuildSpec | None = None


@dataclass(frozen=True, slots=True)
class OptimizerBuildSpec:
    """Optimizer selection without parameters, tensors, or parameter grouping."""

    type: Literal["adam", "adamw", "muon"]
    kwargs: Mapping[str, PlainValue]
    capturable: bool
    companion_adamw_kwargs: Mapping[str, PlainValue] | None = None


@dataclass(frozen=True, slots=True)
class PPOAgentBuildSpec:
    """Pure PPO construction plan."""

    agent_class: Literal["PPO"]
    shared: bool
    actor: ModelRoleBuildSpec
    critic: ModelRoleBuildSpec | None
    optimizer: OptimizerBuildSpec
    scheduler: Mapping[str, PlainValue] | None
    initial_state: Mapping[str, PlainValue] | None
    time_limit_bootstrap: bool | None
    constructor_kwargs: Mapping[str, PlainValue]
    runtime: RuntimeConfig
    latent_prior: Mapping[str, PlainValue] | None = None


type AgentBuildSpec = PPOAgentBuildSpec


def _lower_distribution(value: object) -> DistributionBuildSpec:
    raw = _mapping(value, "models.actor.distribution")
    _validate_distribution(raw)
    distribution = DistributionConfig.from_mapping(raw)
    std = distribution.std
    parameterization = distribution.parameterization
    bound_mode = distribution.bound_mode
    action_transform = distribution.action_transform
    initial = float(distribution.initial_log_std)
    bounds = tuple(float(item) for item in distribution.log_std_bounds)
    std_flags = {
        "parameter": (True, False),
        "fixed": (True, True),
        "network": (False, False),
    }
    fixed_sigma, fixed_log_std = std_flags[std]
    kwargs = {
        "initial_log_std": initial,
        "min_log_std": bounds[0],
        "max_log_std": bounds[1],
        "fixed_sigma": fixed_sigma,
        "fixed_log_std": fixed_log_std,
        "log_std_bound_mode": bound_mode,
        "squash_actions": action_transform == "tanh",
    }
    if distribution.bound_loss is not None:
        kwargs["log_std_bound_loss_weight"] = float(distribution.bound_loss.weight)
        kwargs["log_std_bound_loss_margin"] = float(distribution.bound_loss.margin)
    if parameterization != "log":
        kwargs["std_parameterization"] = parameterization
    return DistributionBuildSpec(
        std=std,
        parameterization=parameterization,
        fixed_sigma=fixed_sigma,
        fixed_log_std=fixed_log_std,
        initial_log_std=initial,
        log_std_bounds=bounds,
        bound_mode=bound_mode,
        action_transform=action_transform,
        constructor_kwargs=_frozen_mapping(kwargs, path="distribution kwargs"),
    )


def _lower_model_role(
    value: object,
    *,
    role: Literal["actor", "critic"],
    expected_architecture: Literal["mlp", "residual_mlp"] | None,
    allow_output_initialization: bool = False,
) -> ModelRoleBuildSpec:
    raw = _mapping(value, f"models.{role}")
    architecture = _string(raw.get("architecture"), f"models.{role}.architecture")
    if architecture not in {"mlp", "residual_mlp"}:
        raise ValueError(f"unsupported models.{role}.architecture: {architecture!r}")
    if expected_architecture is not None and architecture != expected_architecture:
        raise ValueError(
            f"models.{role}.architecture must be {expected_architecture!r}, got {architecture!r}"
        )
    role_keys = _MLP_ROLE_KEYS if architecture == "mlp" else _RESIDUAL_ROLE_KEYS
    actor_keys = {"distribution"}
    if allow_output_initialization:
        actor_keys.add("output_initialization_std")
    allowed = role_keys | (actor_keys if role == "actor" else frozenset())
    required = role_keys | ({"distribution"} if role == "actor" else frozenset())
    mapping = _closed(raw, path=f"models.{role}", allowed=allowed, required=required)
    _validate_network(
        mapping,
        f"models.{role}",
        actor=role == "actor",
        allow_output_initialization=allow_output_initialization,
    )
    role_parser = ACTOR_ADAPTER if role == "actor" else CRITIC_ADAPTER
    typed = role_parser.validate_python(dict(mapping))
    kwargs: dict[str, object] = {
        "architecture": typed.architecture,
        "layers": typed.layers,
        "activation": typed.activation,
        "normalization": typed.normalization,
    }
    if architecture == "residual_mlp":
        kwargs["residual_scale"] = float(typed.residual_scale)
    distribution = (
        _lower_distribution(mapping["distribution"]) if role == "actor" else None
    )
    if role == "actor" and "output_initialization_std" in mapping:
        kwargs["output_initialization_std"] = _float(
            mapping["output_initialization_std"],
            "models.actor.output_initialization_std",
        )
    return ModelRoleBuildSpec(
        role=role,
        architecture=architecture,
        constructor_kwargs=_frozen_mapping(kwargs, path=f"{role} kwargs"),
        distribution=distribution,
    )


_RUNTIME_RULE_MESSAGES = {
    schema_fields.RULE_TORCH_GRAPH_CONFLICT: (
        "{path} enables Torch-managed CUDA graphs while external runtime CUDA "
        "graphs are enabled"
    ),
}


def _validate_runtime(runtime: Mapping[str, object]) -> RuntimeConfig:
    runtime = _closed(runtime, path="runtime", allowed=schema_fields.RUNTIME_FIELDS)
    runtime = RuntimeConfig.from_mapping(runtime)
    for path, rule in schema_fields.runtime_rule_violations(runtime):
        raise ValueError(_RUNTIME_RULE_MESSAGES[rule].format(path=path))
    return runtime


def _lower_optimizer(
    value: object,
    *,
    capturable: bool,
) -> OptimizerBuildSpec:
    raw = _mapping(value, "agent.optimizer")
    optimizer_type = _string(raw.get("type"), "agent.optimizer.type")
    if optimizer_type not in schema_fields.OPTIMIZER_TYPES:
        raise ValueError(f"unsupported agent.optimizer.type: {optimizer_type!r}")
    _validate_optimizer(raw)
    optimizer = parse_optimizer(raw)
    kwargs = {
        key: item for key, item in optimizer.items() if key not in {"type", "adamw"}
    }
    companion: Mapping[str, PlainValue] | None = None
    if optimizer_type == "muon":
        companion_values: dict[str, object] = dict(_MUON_COMPANION)
        if "adamw" in optimizer:
            companion_values.update(optimizer["adamw"])
        kwargs.update(
            {
                "compile_ns": True,
                "batched_updates": True,
                "adamw_betas": companion_values["betas"],
                "adamw_eps": companion_values["eps"],
                "adamw_weight_decay": companion_values["weight_decay"],
                "adamw_amsgrad": companion_values["amsgrad"],
            }
        )
        companion = _frozen_mapping(
            {**companion_values, "capturable": capturable},
            path="Muon companion AdamW kwargs",
        )
    return OptimizerBuildSpec(
        type=optimizer.type,
        kwargs=_frozen_mapping(kwargs, path="optimizer kwargs"),
        capturable=capturable,
        companion_adamw_kwargs=companion,
    )


def _same_shared_role_settings(
    actor: ModelRoleBuildSpec,
    critic: ModelRoleBuildSpec,
) -> None:
    for key in ("activation", "normalization"):
        if actor.constructor_kwargs[key] != critic.constructor_kwargs[key]:
            raise ValueError(
                f"actor and critic {key} must match while constructors share one {key} argument"
            )


def _preprocessor(value: object, path: str) -> str | None:
    name = _string(value, path)
    return None if name == "none" else name


def _lower_reward_preprocessor(
    value: object,
) -> tuple[str | None, Mapping[str, PlainValue] | None]:
    if not isinstance(value, Mapping):
        return _preprocessor(value, "agent.reward_preprocessor"), None
    block = _closed(
        value,
        path="agent.reward_preprocessor",
        allowed=field_names(agent_settings.ReturnScalerConfig),
        required=frozenset({"type"}),
    )
    name = _string(block["type"], "agent.reward_preprocessor.type")
    kwargs = {
        key: _float(block[key], f"agent.reward_preprocessor.{key}")
        for key in ("g_max", "eps")
        if key in block
    }
    return name, _frozen_mapping(kwargs, path="reward preprocessor kwargs")


def _lower_ppo(
    models: Mapping[str, object],
    agent: Mapping[str, object],
    runtime: RuntimeConfig,
) -> PPOAgentBuildSpec:
    agent = _closed(
        agent,
        path="agent",
        allowed=_PPO_AGENT_KEYS,
        required=frozenset(
            {
                "class",
                "rollouts",
                "learning_epochs",
                "mini_batches",
                "discount_factor",
                "gae_lambda",
                "optimizer",
                "learning_rate",
            }
        ),
    )
    shared = _bool(models["shared"], "models.shared")
    if shared and "critic" in models:
        raise ValueError("models.critic must be omitted when PPO models.shared=true")
    if not shared and "critic" not in models:
        raise ValueError("models.critic is required when PPO models.shared=false")
    actor = _lower_model_role(
        models["actor"],
        role="actor",
        expected_architecture=None,
        allow_output_initialization=True,
    )
    critic = (
        _lower_model_role(models["critic"], role="critic", expected_architecture=None)
        if not shared
        else None
    )
    if critic is not None:
        _same_shared_role_settings(actor, critic)
    update_graph = runtime.get("update_cuda_graph") is True
    optimizer = _lower_optimizer(agent["optimizer"], capturable=update_graph)
    learning_rate = _closed(
        agent["learning_rate"],
        path="agent.learning_rate",
        allowed=field_names(agent_settings.PpoLearningRateConfig),
        required=frozenset({"initial"}),
    )
    initial_lr = _float(learning_rate["initial"], "agent.learning_rate.initial")
    bounds = (
        _pair(learning_rate["bounds"], "agent.learning_rate.bounds")
        if "bounds" in learning_rate
        else (initial_lr, initial_lr)
    )
    scheduler: Mapping[str, PlainValue] | None = None
    kwargs: dict[str, object] = {
        "rollouts": _int(agent["rollouts"], "agent.rollouts"),
        "learning_epochs": _int(agent["learning_epochs"], "agent.learning_epochs"),
        "mini_batches": _int(agent["mini_batches"], "agent.mini_batches"),
        "discount_factor": _float(agent["discount_factor"], "agent.discount_factor"),
        "lam": _float(agent["gae_lambda"], "agent.gae_lambda"),
        "learning_rate": initial_lr,
        "min_learning_rate": bounds[0],
        "max_learning_rate": bounds[1],
        "kl_threshold": 0.0,
        "lr_factor": 1.0,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "entropy_loss_scale": 1.0e-3,
        "value_loss_scale": 4.5,
        "layers": actor.constructor_kwargs["layers"],
        "activation": actor.constructor_kwargs["activation"],
        "optimizer_name": optimizer.type,
        "optimizer_kwargs": optimizer.kwargs,
        **dict(actor.distribution.constructor_kwargs if actor.distribution else {}),
    }
    if "scheduler" in learning_rate:
        scheduler_raw = _closed(
            learning_rate["scheduler"],
            path="agent.learning_rate.scheduler",
            allowed=field_names(agent_settings.KlSchedulerConfig),
            required=frozenset({"type", "threshold", "factor"}),
        )
        scheduler_type = _string(
            scheduler_raw["type"], "agent.learning_rate.scheduler.type"
        )
        if scheduler_type != "kl_adaptive":
            raise ValueError(f"unsupported learning-rate scheduler: {scheduler_type!r}")
        threshold = _float(
            scheduler_raw["threshold"], "agent.learning_rate.scheduler.threshold"
        )
        factor = _float(scheduler_raw["factor"], "agent.learning_rate.scheduler.factor")
        update_mode = _string(
            scheduler_raw.get("update_mode", "per_epoch"),
            "agent.learning_rate.scheduler.update_mode",
        )
        if update_mode not in {"per_update", "per_epoch", "per_minibatch"}:
            raise ValueError(
                "unsupported agent.learning_rate.scheduler.update_mode: "
                f"{update_mode!r}"
            )
        scheduler = _frozen_mapping(
            {
                "type": scheduler_type,
                "threshold": threshold,
                "factor": factor,
                "bounds": bounds,
                "update_mode": update_mode,
            },
            path="scheduler",
        )
        kwargs["kl_threshold"] = threshold
        kwargs["lr_factor"] = factor
        kwargs["kl_scheduler_update_mode"] = update_mode
    optional_simple = {
        "shuffle_mode": _string,
        "grad_norm_clip": _float,
        "separate_grad_norm_clip": _bool,
        "ratio_clip": _float,
        "value_clip": _float,
        "clip_predicted_values": _bool,
        "value_loss_clip_mode": _string,
        "entropy_loss_scale": _float,
        "value_loss_scale": _float,
        "kl_estimator": _string,
    }
    for key, converter in optional_simple.items():
        if key in agent:
            kwargs[key] = converter(agent[key], f"agent.{key}")
    for key in ("state_preprocessor", "value_preprocessor"):
        if key in agent:
            kwargs[key] = _preprocessor(agent[key], f"agent.{key}")
    if "reward_preprocessor" in agent:
        name, preprocessor_kwargs = _lower_reward_preprocessor(
            agent["reward_preprocessor"]
        )
        kwargs["reward_preprocessor"] = name
        if preprocessor_kwargs:
            kwargs["reward_preprocessor_kwargs"] = preprocessor_kwargs
    network_kwargs: dict[str, object] = {}
    if "output_initialization_std" in actor.constructor_kwargs:
        kwargs["actor_output_initialization_std"] = actor.constructor_kwargs[
            "output_initialization_std"
        ]
    if not shared:
        assert critic is not None
        network_kwargs.update(
            {
                "separate": True,
                "actor_layers": actor.constructor_kwargs["layers"],
                "critic_layers": critic.constructor_kwargs["layers"],
            }
        )
    if actor.constructor_kwargs["normalization"] == "layer_norm":
        network_kwargs["use_layer_norm"] = True
    if shared and actor.architecture != "mlp":
        network_kwargs["network_type"] = actor.architecture
    elif not shared:
        assert critic is not None
        if actor.architecture != "mlp":
            network_kwargs["actor_network_type"] = actor.architecture
        if critic.architecture != "mlp":
            network_kwargs["critic_network_type"] = critic.architecture
    residual_scales = {
        float(role.constructor_kwargs["residual_scale"])
        for role in (actor, critic)
        if role is not None and role.architecture == "residual_mlp"
    }
    if len(residual_scales) > 1:
        raise ValueError("actor and critic residual_scale must match")
    if residual_scales:
        network_kwargs["residual_scale"] = residual_scales.pop()
    if "parameter_initialization" in agent:
        parameter_initialization = _string(
            agent["parameter_initialization"],
            "agent.parameter_initialization",
        )
        if parameter_initialization not in {"default", "rsl_rl"}:
            raise ValueError(
                "unsupported agent.parameter_initialization: "
                f"{parameter_initialization!r}"
            )
        if parameter_initialization == "rsl_rl" and shared:
            raise ValueError(
                "agent.parameter_initialization='rsl_rl' requires models.shared=false"
            )
        if parameter_initialization == "rsl_rl":
            assert critic is not None
            if (
                actor.architecture != "mlp"
                or critic.architecture != "mlp"
                or actor.constructor_kwargs["normalization"] != "none"
                or critic.constructor_kwargs["normalization"] != "none"
            ):
                raise ValueError(
                    "agent.parameter_initialization='rsl_rl' requires plain MLP "
                    "actor/critic models with normalization=none"
                )
        network_kwargs["parameter_initialization"] = parameter_initialization
    if "activation_regularization" in agent:
        block = _mapping(
            agent["activation_regularization"], "agent.activation_regularization"
        )
        network_kwargs["activation_regularization_weight"] = _float(
            block["weight"], "agent.activation_regularization.weight"
        )
        network_kwargs["activation_regularization_target"] = _float(
            block["target"], "agent.activation_regularization.target"
        )
    if network_kwargs:
        kwargs["network_kwargs"] = network_kwargs
    time_limit_bootstrap = (
        _bool(agent["time_limit_bootstrap"], "agent.time_limit_bootstrap")
        if "time_limit_bootstrap" in agent
        else None
    )
    return PPOAgentBuildSpec(
        agent_class="PPO",
        shared=shared,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        scheduler=scheduler,
        latent_prior=(
            _frozen_mapping(
                _mapping(agent["latent_prior"], "agent.latent_prior"),
                path="agent.latent_prior",
            )
            if "latent_prior" in agent
            else None
        ),
        initial_state=(
            _frozen_mapping(
                _mapping(agent["initial_state"], "agent.initial_state"),
                path="agent.initial_state",
            )
            if "initial_state" in agent
            else None
        ),
        time_limit_bootstrap=time_limit_bootstrap,
        constructor_kwargs=_frozen_mapping(kwargs, path="PPO constructor kwargs"),
        runtime=runtime,
    )


def lower_agent_build_spec(config: object) -> AgentBuildSpec:
    """Lower a closed production PPO configuration without runtime allocation."""
    from instantmimic.rl.models.musubi_actor_critic import reject_unavailable_selectors

    reject_unavailable_selectors(config)
    models = _closed(
        _root(config, "models"),
        path="models",
        allowed=_MODEL_KEYS,
        required=frozenset({"shared", "actor"}),
    )
    agent = _root(config, "agent")
    if "class" not in agent:
        raise ValueError("agent.class is required; PPO is not implicit")
    if _string(agent["class"], "agent.class") != "PPO":
        raise ValueError("unsupported agent.class; expected PPO")
    return _lower_ppo(models, agent, _validate_runtime(_root(config, "runtime")))


__all__ = [
    "AgentBuildSpec",
    "DistributionBuildSpec",
    "ModelRoleBuildSpec",
    "OptimizerBuildSpec",
    "PPOAgentBuildSpec",
    "lower_agent_build_spec",
]
