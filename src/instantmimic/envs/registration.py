"""Registration entrypoint for built-in Gymnasium environments."""

from __future__ import annotations

import importlib
import importlib.util
import logging

logger = logging.getLogger(__name__)

BUILTIN_TASK_MODULES = (
    "instantmimic.envs.freemusco.tasks",
    "instantmimic.envs.snu.tasks.snu_run",
    "instantmimic.envs.snu.tasks.snu_imitation",
    "instantmimic.envs.unitree_g1.tasks.beyondmimic_tracking",
)


def _is_absent(module_name: str) -> bool:
    """Whether a task module is not part of this distribution at all.

    A family that is absent is skipped; one that is present but fails to import
    is a defect and must raise, because swallowing it turns a missing dependency
    into an unexplained "task id not registered" much later.

    `find_spec` imports the parent package to search it, so a parent whose own
    imports fail raises `ModuleNotFoundError` here too. Only the name that
    actually went missing tells the two apart: if it is this module or one of its
    ancestors, the family is genuinely absent; anything else is that family
    failing to load and is re-raised.
    """

    try:
        return importlib.util.find_spec(module_name) is None
    except ModuleNotFoundError as exc:
        ancestors = {
            ".".join(module_name.split(".")[: index + 1])
            for index in range(module_name.count(".") + 1)
        }
        if exc.name in ancestors:
            return True
        raise


def register_builtin_env_tasks() -> None:
    """Import all built-in task packages through one canonical registry."""

    for module_name in BUILTIN_TASK_MODULES:
        if _is_absent(module_name):
            logger.debug(
                "Task family not present in this distribution: %s", module_name
            )
            continue
        importlib.import_module(module_name)
