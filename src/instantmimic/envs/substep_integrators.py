"""How each physics substep of a DirectRLEnv advances, selected by sim.integrator.

``implicitfast``, ``euler`` and ``rk4`` each run one MuJoCo step of the compiled
integrator per substep; implicitfast qH contributors apply to ``implicitfast``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mujoco
import torch

SUBSTEP_IMPLICITFAST = "implicitfast"
SUBSTEP_EULER = "euler"
SUBSTEP_RK4 = "rk4"


class SubstepIntegrator:
    """The env's own substep: one native MuJoCo step and its endpoint refresh.

    ``DirectRLEnv._run_physics_substep`` calls, in order:
    ``advance``, the task's ``_post_physics_substep``,
    ``refresh_endpoint_state_prefix``, the control endpoint callbacks,
    ``finish_substep``, the task's ``_post_state_prefix_substep``.
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def before_pre_physics_step(self) -> None:
        """Called before the env's own ``_pre_physics_step`` for every step."""

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called after the env's own ``_pre_physics_step`` for every step."""
        del actions

    def advance(self, substep_idx: int) -> None:
        self.env._advance_physics_substep(substep_idx)

    def refresh_endpoint_state_prefix(self, *, constraints: bool = True) -> None:
        """``constraints=False`` permits skipping collision and constraint
        construction; the env's reset prefix rebuilds them."""
        self.env._run_mjw_state_prefix(constraints=constraints)

    def finish_substep(self, substep_idx: int) -> None:
        del substep_idx

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class IntegratorSpec:
    """A ``sim.integrator`` value: the compiled MuJoCo integrator and its kind."""

    native: mujoco.mjtIntegrator
    substep: str


BASE_INTEGRATORS: Mapping[str, IntegratorSpec] = {
    "implicitfast": IntegratorSpec(
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST, SUBSTEP_IMPLICITFAST
    ),
    "euler": IntegratorSpec(mujoco.mjtIntegrator.mjINT_EULER, SUBSTEP_EULER),
    "rk4": IntegratorSpec(mujoco.mjtIntegrator.mjINT_RK4, SUBSTEP_RK4),
}


def resolve_integrator_spec(name: Any) -> IntegratorSpec:
    key = str(name).strip().lower()
    try:
        return BASE_INTEGRATORS[key]
    except KeyError as exc:
        raise ValueError(
            f"Invalid integrator: {name!r}; expected one of {sorted(BASE_INTEGRATORS)}"
        ) from exc
