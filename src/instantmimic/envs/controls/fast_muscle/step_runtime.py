from __future__ import annotations

import copy
import logging
import os

import torch
import warp as wp

from instantmimic.envs.controls.fast_muscle.buffer_ops import copy_2d_
from instantmimic.muscle.muscle_qfrc_projector import (
    write_generalized_force_actions_indexed_kernel,
    write_generalized_force_actions_kernel,
)
from instantmimic.utils.network_precision import network_matmul_precision

from .config_models import MassPosePolicyOutputSettings

logger = logging.getLogger(__name__)


def _controller_type():
    from .controller import FastMuscleController

    return FastMuscleController


# Torque-coordinate inputs expand when passive joints are added. Sparse
# moment-arm entries retain their original muscle-coordinate indexing.
_TORQUE_INDEXED_MUSCLENN_INPUT_BLOCKS = frozenset({"target"})
_MUSCLENN_NORMALIZER_FILL = {"mean": 0.0, "variance": 1.0, "shift": 0.0, "inv_std": 1.0}


def _widen_character_mass_payload(
    payload: dict[str, object],
    *,
    blocks: tuple[tuple[str, int, int], ...],
    character: tuple[int, ...],
    torque_dim: int,
) -> dict[str, object]:
    """Carry a MASS payload saved over the character DOFs onto passive DOFs.

    A model that adds passive joint DOFs (env.dof_roles) widens
    the torque-indexed MuscleNN input blocks by one column per passive DOF and
    the torque scale by one structural zero. The saved first layer reads the
    new columns with zero weight, so the network computes exactly what it
    computed on the saved model; muscles span no passive DOF, so the other
    blocks keep their columns. A payload of any other width is returned
    unchanged for the identity and shape checks to refuse.
    """
    widened = dict(payload)
    identity = payload.get("runtime_identity")
    if isinstance(identity, dict):
        scale = identity.get("torque_action_scale")
        if isinstance(scale, list) and len(scale) == len(character) < torque_dim:
            full = [0.0] * torque_dim
            for index, value in zip(character, scale, strict=True):
                full[index] = value
            widened["runtime_identity"] = {**identity, "torque_action_scale": full}
    model = payload.get("model")
    if not isinstance(model, dict):
        return widened
    kept: list[int] = []
    for name, start, end in blocks:
        if name in _TORQUE_INDEXED_MUSCLENN_INPUT_BLOCKS and end - start == torque_dim:
            kept.extend(start + index for index in character)
        else:
            kept.extend(range(start, end))
    width = blocks[-1][2]
    weight = model.get("net.0.weight")
    if (
        not torch.is_tensor(weight)
        or len(kept) == width
        or int(weight.shape[1]) != len(kept)
    ):
        return widened
    columns = torch.tensor(kept, dtype=torch.long, device=weight.device)
    wide_model = dict(model)
    wide_weight = weight.new_zeros((weight.shape[0], width))
    wide_weight[:, columns] = weight
    wide_model["net.0.weight"] = wide_weight
    for name, fill in _MUSCLENN_NORMALIZER_FILL.items():
        buffer = model.get(f"input_normalizer.{name}")
        if torch.is_tensor(buffer):
            wide_buffer = buffer.new_full((width,), fill)
            wide_buffer[columns] = buffer
            wide_model[f"input_normalizer.{name}"] = wide_buffer
    widened["model"] = wide_model
    return widened


class FastMuscleStepMixin:
    def pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != self._policy_action_dim:
            raise ValueError(
                f"Expected action dimension {self._policy_action_dim}, got {actions.shape[-1]}."
            )
        if self._policy_outputs_activation:
            torch.clamp(actions, 0.0, 1.0, out=self.actions)
        elif self._uses_mass_pose_coordinator:
            policy_output = self._mass_action_settings().policy_output
            assert isinstance(policy_output, MassPosePolicyOutputSettings)
            self.actions.copy_(actions)
            self.actions.mul_(policy_output.scale)
        else:
            copy_2d_(self.actions, actions)
        self._update_prev_action_observation()
        if self._uses_mass_coordinator:
            self._musclenn_substep_counter = 0

    def _launch_write_torque_kernel(
        self,
        actions: torch.Tensor,
        apply_scale: float = 1.0,
    ) -> None:
        """Write torque actions to qfrc_applied, dispatching to the indexed
        kernel variant when some DOFs are masked (torque_action_scale == 0)."""
        if self._has_masked_torque_dofs:
            wp.launch(
                write_generalized_force_actions_indexed_kernel,
                dim=(self.num_envs, self._active_torque_count),
                inputs=[
                    wp.from_torch(actions, dtype=wp.float32),
                    self._torque_action_scale_wp,
                    self._active_torque_indices_wp,
                    apply_scale,
                    int(self._root_dof_count),
                ],
                outputs=[self._qfrc_applied_view],
            )
        else:
            wp.launch(
                write_generalized_force_actions_kernel,
                dim=(self.num_envs, self._policy_action_dim),
                inputs=[
                    wp.from_torch(actions, dtype=wp.float32),
                    self._torque_action_scale_wp,
                    apply_scale,
                    int(self._root_dof_count),
                ],
                outputs=[self._qfrc_applied_view],
            )

    def _prepare_mass_torque_command(self) -> torch.Tensor:
        """Prepare the direct-torque MASS command before the shared MuscleNN suffix.

        The command spans every non-root coordinate and is zero where the
        physical torque scale masks the policy action.
        """
        if not self._uses_torque_feasible_bounds():
            self._clear_applied_force_buffers()
            return self._mass_policy_torque_command()
        # Clip a scratch copy: clipping the policy action in place would
        # intersect the bounds of every substep in the control interval.
        command = self._copy_into_mass_torque_command(
            self._mass_policy_torque_command()
        )
        self._prepare_torque_passive_muscle_state()
        self._clip_torque_actions_to_feasible_bounds(command)
        if self._has_masked_torque_dofs:
            command.mul_(self._musclenn_torque_active_mask)
        return command

    def _prepare_mass_pose_command(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare total Stable PD torque and its passive muscle contribution."""
        spd_torque = self._compute_mass_pose_torque_actions()
        return spd_torque, self._compute_mass_pose_passive_torque_actions()

    def _apply_mass_command_via_musclenn(
        self,
        torque_actions: torch.Tensor,
        pose_passive_torque_actions: torch.Tensor | None,
    ) -> None:
        """Shared torque-target -> MuscleNN -> muscle-force MASS execution path."""
        raw_context = self._build_musclenn_raw_context()
        contribution = self._musclenn_torque_helper.compute_activation_torque_jacobian(
            raw_context
        )
        if self._uses_mass_pose_coordinator:
            if pose_passive_torque_actions is None:
                raise RuntimeError("MASS pose command requires passive torque state.")
            torque_actions = self._prepare_mass_pose_targets(
                torque_actions, pose_passive_torque_actions, contribution
            )
        musclenn_input = self._write_musclenn_step_input(contribution, torque_actions)
        self._sanitize_musclenn_input_inplace(musclenn_input)
        training_target = musclenn_input[:, self._musclenn_sparse_entry_dim :]
        with torch.no_grad(), network_matmul_precision():
            activation = self._musclenn_model(
                self._prepare_musclenn_model_input(musclenn_input)
            ).to(dtype=torch.float32)
        self._musclenn_last_activation.copy_(activation)
        self._record_musclenn_rollout_batch(
            musclenn_input.detach(),
            training_target.detach(),
            raw_context.detach(),
        )
        self._apply_activation_to_qfrc(
            activation,
            include_active=1.0,
            include_passive=self._muscle_include_passive,
        )
        self._musclenn_substep_counter += 1

    def apply_action(self) -> None:
        if self._action_mode == "torque":
            if (
                self._uses_torque_feasible_bounds()
                or self._uses_torque_passive_muscle_forces()
            ):
                self._prepare_torque_passive_muscle_state()
                if self._uses_torque_feasible_bounds():
                    self._clip_torque_actions_to_feasible_bounds()
            else:
                self._clear_applied_force_buffers()
            self._launch_write_torque_kernel(self.actions)
            return
        if self._policy_outputs_activation:
            self._apply_activation_to_qfrc(
                self.actions,
                include_active=1.0,
                include_passive=self._muscle_include_passive,
            )
            return
        if not self._uses_mass_coordinator:
            raise ValueError(
                f"Unsupported physical control mode: {self._action_mode!r}."
            )
        assert self._musclenn_model is not None
        assert self._musclenn_torque_helper is not None
        if (
            self._reuse_musclenn_activation_across_substeps_enabled()
            and int(self._musclenn_substep_counter) > 0
        ):
            self._apply_activation_to_qfrc(
                self._musclenn_last_activation,
                include_active=1.0,
                include_passive=self._muscle_include_passive,
            )
            self._musclenn_substep_counter += 1
            return
        if self._uses_mass_pose_coordinator:
            torque_actions, passive_torque = self._prepare_mass_pose_command()
        else:
            torque_actions, passive_torque = self._prepare_mass_torque_command(), None
        self._apply_mass_command_via_musclenn(torque_actions, passive_torque)

    def reset_idx(self, reset_mask: torch.Tensor) -> None:
        for tensor in (
            self.actions,
            self._prev_action_obs,
            self.muscle_length,
            self._muscle_force_magnitude,
        ):
            _controller_type()._masked_zero_2d_(tensor, reset_mask)
        if self._uses_mass_coordinator:
            _controller_type()._masked_zero_2d_(
                self._musclenn_last_activation, reset_mask
            )
        if self._observe_applied_torque and self._applied_torque_buf is not None:
            _controller_type()._masked_zero_2d_(self._applied_torque_buf, reset_mask)
        _controller_type()._masked_zero_2d_(self._xfrc_applied_flat_view, reset_mask)
        _controller_type()._masked_zero_2d_(self._qfrc_applied_view, reset_mask)

    # The contract keys a MASS checkpoint must carry, in comparison order.
    # Both `get_checkpoint_state` and `load_checkpoint_state` build their tuple
    # from this, so the two cannot drift.
    _MASS_CONTRACT_IDENTITY_KEYS = (
        "output_activation",
        "command_semantics",
        "activation_decoder_semantics",
        "physical_dynamics_model",
    )

    def _mass_contract_identity_values(self) -> dict[str, object]:
        """Physical and neural semantics required by a MASS checkpoint."""
        muscle_nn = self.settings.effective.mass_muscle_nn
        assert muscle_nn is not None
        return {
            "output_activation": str(muscle_nn.output_activation),
            "command_semantics": "absolute_torque_target",
            "activation_decoder_semantics": "direct_network_activation",
            "physical_dynamics_model": str(self.settings.muscle.dynamics.model),
        }

    def _mass_checkpoint_runtime_identity(self) -> dict[str, object]:
        identity = {
            "torque_action_scale": self._torque_action_scale.detach().cpu().tolist(),
            "muscle_count": int(self.muscle_count),
            "muscle_names": list(self.muscle_names),
        }
        normalization = _controller_type()._musclenn_input_normalization_identity(self)
        if normalization is not None:
            identity["musclenn_input_normalization"] = normalization
        return identity

    def _musclenn_input_normalization_identity(self) -> dict[str, object] | None:
        """What the MuscleNN input normalization is, or ``None`` without one.

        The statistics themselves are network buffers and travel with its
        ``state_dict``; this records the function they parameterize.
        """
        muscle_nn = getattr(self.settings.action, "muscle_nn", None)
        if muscle_nn is None or muscle_nn.input_normalization == "none":
            return None
        return {
            "mode": str(muscle_nn.input_normalization),
            "statistics": "cumulative_per_column_population_moments",
            "clip": float(muscle_nn.input_normalization_clip),
            "min_std": float(muscle_nn.input_normalization_min_std),
        }

    def get_checkpoint_state(self) -> dict[str, object]:
        if (
            not self._uses_mass_coordinator
            or self._musclenn_model is None
            or self._musclenn_optimizer is None
        ):
            return {}
        mass_state = {
            "model": self._musclenn_model.state_dict(),
            "optimizer": self._musclenn_optimizer.state_dict(),
            **_controller_type()._mass_contract_identity_values(self),
            "runtime_identity": self._mass_checkpoint_runtime_identity(),
        }
        generator = getattr(self, "_musclenn_shuffle_generator", None)
        if generator is not None:
            mass_state["shuffle_generator"] = generator.get_state()
        return {"mass": mass_state}

    def copy_runtime_state_from(self, source: object) -> bool:
        """Copy the evaluation network into existing model storage."""
        if self._uses_mass_coordinator != bool(
            getattr(source, "_uses_mass_coordinator", False)
        ):
            raise ValueError("FastMuscle runtime-state modes do not match.")
        if not self._uses_mass_coordinator:
            return False
        source_model = getattr(source, "_musclenn_model", None)
        if self._musclenn_model is None or source_model is None:
            raise RuntimeError("FastMuscle runtime-state model is unavailable.")
        controller = _controller_type()
        if controller._mass_contract_identity_values(
            self
        ) != controller._mass_contract_identity_values(source):
            raise ValueError("FastMuscle runtime-state MASS contracts do not match.")
        if controller._musclenn_input_normalization_identity(
            self
        ) != controller._musclenn_input_normalization_identity(source):
            raise ValueError(
                "FastMuscle runtime-state MuscleNN input normalizations do not match."
            )
        destination, saved = (
            self._musclenn_model.state_dict(),
            source_model.state_dict(),
        )
        if destination.keys() != saved.keys():
            raise ValueError("FastMuscle runtime-state model layouts do not match.")
        for name, tensor in destination.items():
            if tensor.shape != saved[name].shape:
                raise ValueError(
                    f"FastMuscle runtime-state tensor {name!r} shape mismatch."
                )
        for name, tensor in destination.items():
            tensor.copy_(saved[name])
        return True

    def load_checkpoint_state(
        self,
        state: dict[str, object] | None,
        *,
        strict_identity: bool | None = None,
    ) -> bool:
        """Restore controller state, verifying the MASS contract identity.

        `strict_identity` governs missing contract keys and missing runtime
        identity for contract-v1 checkpoints. Default True refuses them; False
        warns and loads.
        Any identity value that is present but disagrees, including an explicit
        ``None``, always raises whichever way this is set -- that is a real
        contract difference, not a legacy gap.

        Real artifacts need the escape hatch: checkpoints under `logs/` from
        August 2026 carry `env_state.mass` with `model`, `optimizer` and
        `shuffle_generator` and none of the six identity keys. Set
        `INSTANTMIMIC_ALLOW_LEGACY_MASS_CHECKPOINT=1` to load one without
        threading the argument through every caller.
        """

        if strict_identity is None:
            strict_identity = os.environ.get(
                "INSTANTMIMIC_ALLOW_LEGACY_MASS_CHECKPOINT", ""
            ).strip().lower() not in {"1", "true", "yes"}
        restored = False
        if not isinstance(state, dict):
            return restored
        if (
            self._uses_mass_coordinator
            and self._musclenn_model is not None
            and self._musclenn_optimizer is not None
        ):
            payload = state.get("mass")
            if isinstance(payload, dict):
                # Use the declared controller contract for duck-typed callers.
                controller = _controller_type()
                expected_values = controller._mass_contract_identity_values(self)
                # Absent keys are reported, not defaulted. Filling them in with
                # the historical defaults meant a checkpoint that never recorded
                # `output_activation` was compared as if it had said
                # "tanh_relu", and passed whenever the current runtime happened
                # to agree.
                identity_keys = controller._MASS_CONTRACT_IDENTITY_KEYS
                missing = [key for key in identity_keys if key not in payload]
                if missing and strict_identity:
                    raise ValueError(
                        "MASS checkpoint is missing contract identity keys "
                        f"{sorted(missing)}; it predates the identity block and "
                        "cannot be verified against this runtime. Set "
                        "INSTANTMIMIC_ALLOW_LEGACY_MASS_CHECKPOINT=1 to load it "
                        "unverified."
                    )
                if missing:
                    logger.warning(
                        "Loading a MASS checkpoint without contract identity "
                        "keys %s. It cannot be verified against this runtime, "
                        "and a contract difference will not be detected.",
                        sorted(missing),
                    )
                differing = {
                    key: (payload[key], expected_values[key])
                    for key in identity_keys
                    if key in payload
                    and type(expected_values[key])(payload[key]) != expected_values[key]
                }
                if differing:
                    raise ValueError(
                        "MASS checkpoint contract mismatch: "
                        + ", ".join(
                            f"{key} saved={saved!r} expected={expected!r}"
                            for key, (saved, expected) in sorted(differing.items())
                        )
                    )
                expected_runtime_identity = self._mass_checkpoint_runtime_identity()
                runtime_identity_missing = "runtime_identity" not in payload
                if runtime_identity_missing and strict_identity:
                    raise ValueError(
                        "MASS checkpoint is missing runtime identity; it cannot "
                        "be verified against this runtime. Set "
                        "INSTANTMIMIC_ALLOW_LEGACY_MASS_CHECKPOINT=1 to load a "
                        "contract-v1 checkpoint unverified."
                    )
                if runtime_identity_missing:
                    logger.warning(
                        "Loading a MASS contract-v1 checkpoint without runtime "
                        "identity. Its torque scale and muscle layout cannot be "
                        "verified against this runtime."
                    )
                saved_runtime_identity = payload.get("runtime_identity")
                if (
                    not runtime_identity_missing
                    and saved_runtime_identity != expected_runtime_identity
                ):
                    raise ValueError(
                        "MASS checkpoint runtime identity mismatch: "
                        f"saved={saved_runtime_identity}, "
                        f"expected={expected_runtime_identity}"
                    )
                model_state = payload.get("model")
                optimizer_state = payload.get("optimizer")
                shuffle_generator_state = payload.get("shuffle_generator")
            else:
                model_state = None
                optimizer_state = None
                shuffle_generator_state = None
        else:
            model_state = None
            optimizer_state = None
            shuffle_generator_state = None

        mutation_requested = any(
            value is not None for value in (model_state, optimizer_state)
        ) or torch.is_tensor(shuffle_generator_state)
        if not mutation_requested:
            return restored

        model_snapshot = (
            copy.deepcopy(self._musclenn_model.state_dict())
            if model_state is not None
            else None
        )
        optimizer_snapshot = (
            copy.deepcopy(self._musclenn_optimizer.state_dict())
            if optimizer_state is not None
            else None
        )
        model_training = (
            bool(self._musclenn_model.training) if model_state is not None else None
        )
        shuffle_generator = getattr(self, "_musclenn_shuffle_generator", None)
        shuffle_generator_snapshot = (
            shuffle_generator.get_state().clone()
            if shuffle_generator is not None
            and torch.is_tensor(shuffle_generator_state)
            else None
        )
        pending_generator_state_existed = hasattr(
            self, "_musclenn_shuffle_generator_state"
        )
        pending_generator_state_snapshot = getattr(
            self, "_musclenn_shuffle_generator_state", None
        )
        if torch.is_tensor(pending_generator_state_snapshot):
            pending_generator_state_snapshot = (
                pending_generator_state_snapshot.detach().clone()
            )
        graph_attribute_names = (
            "_musclenn_update_graph",
            "_musclenn_shuffle_graph",
            "_musclenn_graph_signature",
            "_musclenn_compiled_minibatch_ready",
            "_run_musclenn_graph_minibatch_fn",
            "_musclenn_update_graph_disabled_after_restore",
            "_musclenn_use_update_graph",
        )
        graph_attribute_snapshot = {
            name: (hasattr(self, name), getattr(self, name, None))
            for name in graph_attribute_names
        }

        optimizer_load_attempted = False
        try:
            mass_restored = False
            if model_state is not None:
                self._musclenn_model.load_state_dict(model_state)
                mass_restored = True
            if optimizer_state is not None:
                optimizer_load_attempted = True
                self._musclenn_optimizer.load_state_dict(optimizer_state)
                mass_restored = True
            if torch.is_tensor(shuffle_generator_state):
                if shuffle_generator is None:
                    self._musclenn_shuffle_generator_state = (
                        shuffle_generator_state.detach().clone()
                    )
                else:
                    shuffle_generator.set_state(shuffle_generator_state)
                mass_restored = True
            if mass_restored:
                invalidate_graphs = getattr(
                    self, "_invalidate_musclenn_update_graphs", None
                )
                if callable(invalidate_graphs):
                    had_captured_graph = bool(
                        getattr(self, "_musclenn_update_graph", None) is not None
                    )
                    invalidate_graphs()
                    if had_captured_graph:
                        # Recapturing after Warp/PPO graphs have claimed allocator
                        # pools is unsafe. A live runtime restore therefore falls
                        # back to eager MuscleNN updates until the next process start.
                        self._musclenn_update_graph_disabled_after_restore = True
                        self._musclenn_use_update_graph = False
                self._musclenn_model.eval()
                restored = True
        except Exception:
            if model_snapshot is not None:
                self._musclenn_model.load_state_dict(model_snapshot)
                self._musclenn_model.train(bool(model_training))
            if optimizer_snapshot is not None:
                self._musclenn_optimizer.load_state_dict(optimizer_snapshot)
            if shuffle_generator_snapshot is not None:
                shuffle_generator.set_state(shuffle_generator_snapshot)
            if pending_generator_state_existed:
                self._musclenn_shuffle_generator_state = (
                    pending_generator_state_snapshot
                )
            elif hasattr(self, "_musclenn_shuffle_generator_state"):
                del self._musclenn_shuffle_generator_state
            for name, (existed, value) in graph_attribute_snapshot.items():
                if existed:
                    setattr(self, name, value)
                elif hasattr(self, name):
                    delattr(self, name)
            if optimizer_load_attempted:
                invalidate_graphs = getattr(
                    self, "_invalidate_musclenn_update_graphs", None
                )
                if callable(invalidate_graphs):
                    invalidate_graphs()
                else:
                    self._musclenn_update_graph = None
                    self._musclenn_shuffle_graph = None
                    self._musclenn_graph_signature = None
                    self._musclenn_compiled_minibatch_ready = False
                self._musclenn_update_graph_disabled_after_restore = True
                self._musclenn_use_update_graph = False
            raise
        return restored

    def load_policy_init_state(self, state: dict[str, object] | None) -> bool:
        """Restore the MuscleNN a policy_init start continues from.

        A MASS policy commands MuscleNN, so the network is part of the
        policy's inference path and loads with it. Its optimizer and shuffle
        state start fresh, as the agent's optimizers do. A payload saved
        on a model without this one's passive DOFs is widened onto them.
        """
        if not isinstance(state, dict) or not self._uses_mass_coordinator:
            return False
        payload = state.get("mass")
        if not isinstance(payload, dict):
            return False
        payload = {
            key: value
            for key, value in payload.items()
            if key not in ("optimizer", "shuffle_generator")
        }
        torque_dim = int(self.model.nv - self._root_dof_count)
        passive = set(getattr(self, "_passive_torque_indices", ()))
        if passive:
            payload = _widen_character_mass_payload(
                payload,
                blocks=self._musclenn_input_block_layout(
                    int(self._musclenn_step_input.shape[-1])
                ),
                character=tuple(i for i in range(torque_dim) if i not in passive),
                torque_dim=torque_dim,
            )
        return bool(self.load_checkpoint_state({"mass": payload}))
