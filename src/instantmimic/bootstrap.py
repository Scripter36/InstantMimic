"""Process setup the training and playback entrypoints both perform.

`train/runtime.py` and `play/runtime.py` opened with the same two lines and then,
further down, made the same two calls before `gym.make`. Keeping one copy is
what stops the two runs from starting in different states -- which is how
`MUJOCO_GL` came to be decided in two places before P2-09.
"""

from __future__ import annotations

import logging
import warnings

from instantmimic.utils.network_precision import configure_ieee_float32_precision


def _install_benign_torch_warning_filters() -> None:
    """Hide known Torch implementation warnings after explicit setup."""
    warnings.filterwarnings(
        "ignore",
        message=r"TypedStorage is deprecated\..*",
        category=UserWarning,
        module=r"torch\._inductor\.select_algorithm",
    )
    dynamo_torch_logger = logging.getLogger("torch._dynamo.variables.torch")
    if not any(
        getattr(filter_, "_instantmimic_profiler_warning_filter", False)
        for filter_ in dynamo_torch_logger.filters
    ):

        class _ProfilerWarningFilter(logging.Filter):
            _instantmimic_profiler_warning_filter = True

            def filter(self, record: logging.LogRecord) -> bool:
                message = record.getMessage()
                return not (
                    message.startswith("Profiler function ")
                    and message.endswith(" will be ignored")
                )

        dynamo_torch_logger.addFilter(_ProfilerWarningFilter())


def configure_process() -> None:
    """Set the process-global state a run needs before it builds anything.

    `basicConfig` is a no-op once a handler exists, so a caller that installed
    its own logging first keeps it.
    """

    logging.basicConfig(level=logging.INFO)
    configure_ieee_float32_precision()
    _install_benign_torch_warning_filters()


def prepare_environment_runtime() -> None:
    """Bring up Warp and register the built-in tasks, in that order.

    Imported here rather than at module scope so that importing this module
    costs nothing: `tests/test_entrypoint_modules.py` requires the CLI import
    path to leave Warp uninitialized.
    """

    from instantmimic.envs.registration import (
        register_builtin_env_tasks as _register_env_tasks,
    )
    from instantmimic.utils.warp_config import initialize_warp_runtime

    initialize_warp_runtime()
    _register_env_tasks()
