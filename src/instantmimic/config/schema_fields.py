"""The field sets and cross-field rules of schema v1, in one place.

Several modules read the same blocks and had each written the field names out
again. Keeping the shared sets here prevents validation and build-spec lowering
from disagreeing about which fields are accepted.

The sets live here and the rules are computed here. Wording stays with the
caller -- these functions name the rule that failed, never the sentence -- so
each surface keeps the errors it has always raised.
"""

from __future__ import annotations

from collections.abc import Mapping

# --- runtime ----------------------------------------------------------------
from typing import get_args

from instantmimic.config.experiment_config import LOGGER_MODELS
from instantmimic.config.policy_config import OPTIMIZER_MODELS, AdamWCompanionConfig
from instantmimic.config.runtime_config import (
    DEFAULT_COMPILE_MODE,
    CompileMode,
    NvtxMode,
    RuntimeConfig,
)
from instantmimic.config.settings import field_names

RUNTIME_FIELDS = frozenset(RuntimeConfig.model_fields)
RUNTIME_BOOL_FIELDS = frozenset(
    name
    for name, field in RuntimeConfig.model_fields.items()
    if field.annotation is bool
)
RUNTIME_ENUM_FIELDS = RUNTIME_FIELDS - RUNTIME_BOOL_FIELDS
SUPPORTED_COMPILE_MODES = frozenset(get_args(CompileMode))
TORCH_GRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})
"""Modes in which Torch manages CUDA graphs itself."""

NSYS_NVTX_MODES = frozenset(get_args(NvtxMode))

RULE_TORCH_GRAPH_CONFLICT = "torch_graph_conflict"


def runtime_rule_violations(
    runtime: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    """Every cross-field rule the `runtime` block breaks, as `(path, rule)`.

    `rule` is one of the `RULE_*` constants above. The caller turns it into its
    own message and reports in the order returned.

    An absent `compile_mode` means `DEFAULT_COMPILE_MODE`, which is what the
    agent factories fall back to. This function does not re-check membership;
    an unsupported mode is the caller's own error, raised before this runs.
    """

    violations: list[tuple[str, str]] = []
    compile_enabled = runtime.get("compile") is True
    external_graphs = (
        runtime.get("act_cuda_graph") is True
        or runtime.get("update_cuda_graph") is True
    )
    compile_mode = runtime.get("compile_mode", DEFAULT_COMPILE_MODE)
    if (
        compile_enabled
        and external_graphs
        and compile_mode in TORCH_GRAPH_COMPILE_MODES
    ):
        violations.append(("runtime.compile_mode", RULE_TORCH_GRAPH_CONFLICT))
    return tuple(violations)


# --- agent.optimizer --------------------------------------------------------

MUON_COMPANION_FIELDS = frozenset(AdamWCompanionConfig.model_fields)
OPTIMIZER_FIELDS = {
    name: frozenset(model.model_fields) for name, model in OPTIMIZER_MODELS.items()
}
OPTIMIZER_TYPES = frozenset(OPTIMIZER_FIELDS)
ADJUST_LR_FNS = frozenset({"match_rms_adamw"})


# --- experiment.logger ------------------------------------------------------

LOGGER_FIELDS = {name: field_names(model) for name, model in LOGGER_MODELS.items()}
LOGGER_TYPES = frozenset(LOGGER_FIELDS)

BACKEND_GROUP_FIELD = "group"
"""Backend-specific field rejected in favor of `experiment.group`."""
