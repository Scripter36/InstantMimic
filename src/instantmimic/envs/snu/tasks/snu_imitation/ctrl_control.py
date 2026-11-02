from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from instantmimic.envs.control_pipeline import ControlRuntimeSpec, NoControlPipeline
from instantmimic.envs.snu.tasks.snu_imitation import action_runtime, nan_debug

_LOGGER = logging.getLogger(__name__)


@dataclass
class SNUCtrlControlCfg:
    action_dim: int
    action_scale: float = 1.0
    use_ref_action: bool = False
    action_mode: str = "ctrl"
    clamp_actions: bool = True

    def build_control_pipeline(self) -> SNUCtrlControl:
        return SNUCtrlControl(self)


class SNUCtrlControl(NoControlPipeline):
    def __init__(self, cfg: SNUCtrlControlCfg) -> None:
        self.cfg = cfg
        self.env: Any | None = None

    @property
    def is_enabled(self) -> bool:
        return True

    def finalize_after_model_compile(self, env: Any) -> None:
        self.env = env
        compiled_action_dim = int(env.model.nu)
        if compiled_action_dim <= 0:
            raise ValueError(
                "SNU ctrl control requires at least one compiled actuator; "
                f"got model.nu={compiled_action_dim}."
            )
        configured_action_dim = int(self.cfg.action_dim)
        if configured_action_dim != compiled_action_dim:
            _LOGGER.warning(
                "Overriding configured SNU ctrl action_dim=%d with compiled "
                "model.nu=%d so action storage matches the ctrl kernel.",
                configured_action_dim,
                compiled_action_dim,
            )
            self.cfg.action_dim = compiled_action_dim
        expected_shape = (int(env.num_envs), compiled_action_dim)
        actions = getattr(env, "actions", None)
        if actions is None or tuple(actions.shape) != expected_shape:
            env.actions = torch.zeros(
                expected_shape,
                device=env.device,
                dtype=torch.float32,
            )

    def configure_spaces(self, env: Any) -> None:
        env.cfg.action_space = int(self.cfg.action_dim)

    def runtime_spec(self) -> ControlRuntimeSpec:
        return ControlRuntimeSpec(
            action_mode=str(self.cfg.action_mode),
            policy_action_dim=int(self.cfg.action_dim),
        )

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        assert self.env is not None
        actions = nan_debug.sanitize_action_input(self.env, actions)
        scale = float(self.cfg.action_scale)
        if self.cfg.clamp_actions:
            torch.clamp(actions, -1.0 / scale, 1.0 / scale, out=self.env.actions)
        else:
            self.env.actions.copy_(actions)
        self.env.actions.mul_(scale)
        if hasattr(self.env, "ref_frame_idx"):
            self.env.ref_frame_idx += 1

    def apply_action(self) -> None:
        assert self.env is not None
        action_runtime.apply_ctrl_action(
            self.env,
            use_ref_action=bool(self.cfg.use_ref_action),
        )

    def reset_idx(self, reset_mask: torch.Tensor) -> None:
        assert self.env is not None
        self.env.actions.masked_fill_(reset_mask.unsqueeze(-1), 0.0)
