"""Logger variants and experiment/checkpoint authoring settings."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import AfterValidator, BeforeValidator, Field

from instantmimic.config.settings import Settings
from instantmimic.config.values import freeze_value


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("tags must be a mapping")
    return dict(value)


type StringMapping = Annotated[
    Mapping[str, str], BeforeValidator(_string_mapping), AfterValidator(freeze_value)
]


class TensorboardLoggerConfig(Settings):
    type: Literal["tensorboard"]


class WandbLoggerConfig(Settings):
    type: Literal["wandb"]
    project: str | None = None
    job_type: str | None = None
    entity: str | None = None


class MlflowLoggerConfig(Settings):
    type: Literal["mlflow"]
    experiment_name: str | None = None
    tags: StringMapping | None = None
    tracking_uri: str | None = None


type LoggerConfig = Annotated[
    TensorboardLoggerConfig | WandbLoggerConfig | MlflowLoggerConfig,
    Field(discriminator="type"),
]
LOGGER_MODELS = {
    "tensorboard": TensorboardLoggerConfig,
    "wandb": WandbLoggerConfig,
    "mlflow": MlflowLoggerConfig,
}


class PeriodicCheckpointConfig(Settings):
    interval: int | None = None


class BestCheckpointConfig(Settings):
    source: Literal["auto", "training", "benchmark"] = "auto"
    metric: str | None = None
    interval: int | None = None


class CheckpointConfig(Settings):
    disabled: bool = False
    replay: bool = False
    periodic: PeriodicCheckpointConfig | None = None
    best: BestCheckpointConfig | None = None


class ExperimentConfig(Settings):
    directory: str | None = None
    name: str | None = None
    group: str | None = None
    logger: LoggerConfig | None = None
    write_interval: int = 0
    episode_metrics_interval: int = 0
    checkpoint: CheckpointConfig | None = None
    writer_max_queue: int | None = None
    writer_flush_secs: int | None = None
