from __future__ import annotations

from collections.abc import Sequence
from math import pi

import torch

SOURCE_JOINT_ORDER = (
    "front_left_leg",
    "front_right_leg",
    "left_back_leg",
    "right_back_leg",
    "front_left_foot",
    "front_right_foot",
    "left_back_foot",
    "right_back_foot",
)

# Physical traversal order in nv_ant.xml. This must remain independent of the
# source articulation's runtime data/action order above.
MJCF_MODEL_JOINT_ORDER = (
    "hip_1",
    "ankle_1",
    "hip_2",
    "ankle_2",
    "hip_3",
    "ankle_3",
    "hip_4",
    "ankle_4",
)

SOURCE_TO_MJCF_JOINT = {
    "front_left_leg": "hip_1",
    "front_right_leg": "hip_2",
    "left_back_leg": "hip_3",
    "right_back_leg": "hip_4",
    "front_left_foot": "ankle_1",
    "front_right_foot": "ankle_2",
    "left_back_foot": "ankle_3",
    "right_back_foot": "ankle_4",
}

SOURCE_MJCF_JOINT_ORDER = tuple(
    SOURCE_TO_MJCF_JOINT[source_name] for source_name in SOURCE_JOINT_ORDER
)

# nv_ant.xml intentionally keeps its original actuator order. The source task's
# action order is mapped explicitly at startup rather than inheriting XML order.
ACTUATOR_JOINT_ORDER = (
    "hip_4",
    "ankle_4",
    "hip_1",
    "ankle_1",
    "hip_2",
    "ankle_2",
    "hip_3",
    "ankle_3",
)

DEFAULT_ROOT_POSITION = (0.0, 0.0, 0.5)
DEFAULT_ROOT_QUATERNION = (1.0, 0.0, 0.0, 0.0)
DEFAULT_JOINT_POSITIONS = (
    0.0,
    0.0,
    0.0,
    0.0,
    pi / 4,
    -pi / 4,
    -pi / 4,
    pi / 4,
)


def action_to_actuator_indices(
    actuator_joint_order: Sequence[str],
) -> tuple[int, ...]:
    """Return the actuator slot for every source-ordered action."""

    actuator_indices = {name: index for index, name in enumerate(actuator_joint_order)}
    try:
        return tuple(
            actuator_indices[SOURCE_TO_MJCF_JOINT[source_name]]
            for source_name in SOURCE_JOINT_ORDER
        )
    except KeyError as exc:
        raise ValueError(
            "Ant actuators must cover all source-ordered joints; "
            f"missing {exc.args[0]!r}."
        ) from exc


def map_actions_to_actuator_ctrl(
    actions: torch.Tensor,
    source_to_actuator_indices: torch.Tensor,
    *,
    action_scale: float,
) -> torch.Tensor:
    """Map source-ordered normalized actions to XML actuator control slots."""

    ctrl = torch.zeros_like(actions)
    ctrl.index_copy_(1, source_to_actuator_indices, actions * action_scale)
    return ctrl


__all__ = [
    "ACTUATOR_JOINT_ORDER",
    "DEFAULT_JOINT_POSITIONS",
    "DEFAULT_ROOT_POSITION",
    "DEFAULT_ROOT_QUATERNION",
    "MJCF_MODEL_JOINT_ORDER",
    "SOURCE_JOINT_ORDER",
    "SOURCE_MJCF_JOINT_ORDER",
    "SOURCE_TO_MJCF_JOINT",
    "action_to_actuator_indices",
    "map_actions_to_actuator_ctrl",
]
