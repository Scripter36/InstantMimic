"""Production actor and critic architectures."""

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from instantmimic.config.policy_config import DistributionConfig, Number
from instantmimic.config.settings import Settings


class MlpNetworkConfig(Settings):
    architecture: Literal["mlp"]
    layers: tuple[int, ...]
    activation: Literal["elu", "relu", "tanh", "silu", "mish", "leaky_relu"]
    normalization: Literal["none", "layer_norm"]


class ResidualNetworkConfig(MlpNetworkConfig):
    architecture: Literal["residual_mlp"]
    residual_scale: Number


class ActorFields(Settings):
    distribution: DistributionConfig | None = None
    output_initialization_std: Number | None = None


class MlpActorConfig(MlpNetworkConfig, ActorFields):
    pass


class ResidualActorConfig(ResidualNetworkConfig, ActorFields):
    pass


class MlpCriticConfig(MlpNetworkConfig):
    pass


class ResidualCriticConfig(ResidualNetworkConfig):
    pass


type ActorConfig = Annotated[
    MlpActorConfig | ResidualActorConfig, Field(discriminator="architecture")
]
type CriticConfig = Annotated[
    MlpCriticConfig | ResidualCriticConfig, Field(discriminator="architecture")
]
ACTOR_ADAPTER = TypeAdapter(ActorConfig)
CRITIC_ADAPTER = TypeAdapter(CriticConfig)
NETWORK_MODELS = {"mlp": MlpNetworkConfig, "residual_mlp": ResidualNetworkConfig}


class ModelsConfig(Settings):
    shared: bool | None = None
    actor: ActorConfig | None = None
    critic: CriticConfig | None = None
