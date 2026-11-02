"""Closed production PPO settings and optional reset learning."""

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, TypeAdapter

from instantmimic.config.policy_config import Number, OptimizerConfig
from instantmimic.config.settings import Settings


class KlSchedulerConfig(Settings):
    type: Literal["kl_adaptive"]
    threshold: Number
    factor: Number
    update_mode: Literal["per_update", "per_epoch", "per_minibatch"] | None = None


class PpoLearningRateConfig(Settings):
    initial: Number
    bounds: tuple[Number, Number] | None = None
    scheduler: KlSchedulerConfig | None = None


class ReturnScalerConfig(Settings):
    type: Literal["RunningReturnScaler"]
    g_max: Number | None = None
    eps: Number | None = None


class LatentPriorConfig(Settings):
    enabled: bool
    latent_dim: int | None = None
    kl_weight: Number | None = None
    posterior_sample_alpha: Number | None = None
    posterior_hidden_dims: tuple[int, ...] | None = None
    prior_hidden_dims: tuple[int, ...] | None = None
    eval_mode: Literal["prior", "posterior"] | None = None
    use_full_obs_critic: bool | None = None
    critic_use_detached_posterior_mu: bool | None = None
    ar1_weight: Number | None = None
    ar1_phi: Number | None = None
    min_logvar: Number | None = None
    max_logvar: Number | None = None


class ActivationRegularizationConfig(Settings):
    """Squared penalty on the policy mean around ``target``, for activation outputs."""

    weight: Number
    target: Number


class InitialStateConfig(Settings):
    learning_rate: Number | None = None
    initial_log_std: Number | None = None
    log_std_bounds: tuple[Number, Number] | None = None
    ratio_clip: Number | None = None
    reference_kl_weight: Number | None = None
    max_grad_norm: Number | None = None
    baseline_ema: Number | None = None
    max_sample_age: int | None = None


type Preprocessor = Literal[
    "RunningStandardScaler", "RslRlEmpiricalNormalization", "none"
]
type RewardPreprocessor = Literal["RunningReturnScaler", "none"] | ReturnScalerConfig


class CommonAgentConfig(Settings):
    """Sparse common settings accepted by the existing direct semantic API.

    Loading a runnable config still requires an algorithm and its required fields
    through validate_semantic_mapping. This class has no algorithm discriminator.
    """

    rollouts: int | None = None
    discount_factor: Number | None = None
    optimizer: OptimizerConfig | None = None
    time_limit_bootstrap: bool | None = None
    grad_norm_clip: Number | None = None
    state_preprocessor: Preprocessor | None = None
    reward_preprocessor: RewardPreprocessor | None = None


class PpoAgentConfig(CommonAgentConfig):
    algorithm: Literal["PPO"] = Field(alias="class")
    learning_rate: PpoLearningRateConfig | None = None
    learning_epochs: int | None = None
    mini_batches: int | None = None
    gae_lambda: Number | None = None
    shuffle_mode: Literal["per_update", "per_epoch", "off"] | None = None
    parameter_initialization: Literal["default", "rsl_rl"] | None = None
    value_preprocessor: Preprocessor | None = None
    separate_grad_norm_clip: bool | None = None
    ratio_clip: Number | None = None
    value_clip: Number | None = None
    clip_predicted_values: bool | None = None
    value_loss_clip_mode: Literal["clamp", "max"] | None = None
    entropy_loss_scale: Number | None = None
    value_loss_scale: Number | None = None
    kl_estimator: Literal["sampled", "analytic"] | None = None
    latent_prior: LatentPriorConfig | None = None
    initial_state: InitialStateConfig | None = None
    activation_regularization: ActivationRegularizationConfig | None = None


type AgentConfig = PpoAgentConfig | CommonAgentConfig
AGENT_MODELS = {"PPO": PpoAgentConfig}
_AGENT_ADAPTER = TypeAdapter(PpoAgentConfig)


def parse_agent(value: Mapping[str, object]) -> AgentConfig:
    if isinstance(value, Settings):
        value = value.to_dict()
    if "class" not in value:
        return CommonAgentConfig.model_validate(dict(value))
    return _AGENT_ADAPTER.validate_python(dict(value))
