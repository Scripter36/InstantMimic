"""Bridge environments that attach their fast-muscle controller directly."""

from typing import Any

import torch

from .control import make_fast_muscle_runtime_spec


class FastMuscleEnvMixin:
    fast_muscle: Any

    def seed(self, seed: int = -1) -> int:
        resolved_seed = super().seed(seed)
        # Warp kernels consume signed int32 seeds; preserve all 32 seed bits.
        self._freemusco_initial_seed = (int(resolved_seed) + 2**31) % 2**32 - 2**31
        reset_seed = self.__dict__.get("_freemusco_reset_seed")
        if reset_seed is not None:
            reset_seed.fill_(self._freemusco_initial_seed)
        return resolved_seed

    def _reset_seed_buffer(self) -> torch.Tensor:
        seed = self.__dict__.get("_freemusco_reset_seed")
        if seed is None:
            seed = torch.full(
                (1,),
                self.__dict__.get("_freemusco_initial_seed", 0),
                dtype=torch.int32,
                device=self.device,
            )
            self._freemusco_reset_seed = seed
        return seed

    def _sample_reset_seed(self) -> torch.Tensor:
        """Advance a persistent kernel seed, including during graph replay.

        Constructor/reset seeding restarts this sequence. Ordinary resets keep
        advancing it independently of the physics counter. Device arithmetic
        keeps replay independent of host RNG or Python counter updates.
        """
        return self._reset_seed_buffer().add_(1)

    def get_checkpoint_state(self) -> dict[str, object]:
        state = super().get_checkpoint_state()
        reset_seed = self.__dict__.get("_freemusco_reset_seed")
        if reset_seed is not None:
            state["freemusco_reset"] = {
                "schema_version": 1,
                "seed": reset_seed.detach().clone(),
            }
        return state

    def load_checkpoint_state(self, state: dict[str, object] | None) -> bool:
        reset_state = state.get("freemusco_reset") if isinstance(state, dict) else None
        source = None
        if isinstance(state, dict) and "freemusco_reset" in state:
            if not isinstance(reset_state, dict):
                raise ValueError(
                    "Checkpoint field 'freemusco_reset' must be a mapping."
                )
            if (
                type(reset_state.get("schema_version")) is not int
                or reset_state["schema_version"] != 1
            ):
                raise ValueError(
                    "Incompatible FreeMusco reset checkpoint schema_version."
                )
            source = reset_state.get("seed")
            if (
                not isinstance(source, torch.Tensor)
                or source.shape != (1,)
                or source.dtype != torch.int32
            ):
                raise ValueError(
                    "FreeMusco reset checkpoint seed must be an int32 tensor of shape (1,)."
                )
            source = source.detach().to(device=self.device).clone()
        restored = bool(super().load_checkpoint_state(state))
        if source is not None:
            self._reset_seed_buffer().copy_(source)
            return True
        return restored

    def get_control_runtime_spec(self):
        fast_muscle = getattr(self, "fast_muscle", None)
        if fast_muscle is None:
            return super().get_control_runtime_spec()
        return make_fast_muscle_runtime_spec(fast_muscle)

    def copy_runtime_state_from(self, source_env: Any) -> bool:
        """Copy the attached controller's eval-relevant state from a peer env.

        The controller is attached to the host directly rather than through
        the base control pipeline, whose copy is a no-op. Without this the
        training benchmark environment kept its initial MuscleNN.
        """
        fast_muscle = getattr(self, "fast_muscle", None)
        if fast_muscle is None:
            return bool(super().copy_runtime_state_from(source_env))
        source_raw = getattr(source_env, "unwrapped", source_env)
        source_fast_muscle = getattr(source_raw, "fast_muscle", None)
        if source_fast_muscle is None:
            raise TypeError(
                "FastMuscle runtime state source has no attached fast_muscle "
                "controller."
            )
        return bool(fast_muscle.copy_runtime_state_from(source_fast_muscle))
