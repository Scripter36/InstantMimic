from __future__ import annotations

import copy
import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from instantmimic.muscle.muscle_dynamics import MuscleDynamicsLayer
from instantmimic.muscle.parsing.mjcf import (
    cache_key,
    collect_xml_dependency_signatures,
    file_signature,
    load_pickle_cache,
    store_pickle_cache,
)
from instantmimic.muscle.runtime_specs import MuscleLayerRuntimeSpec
from instantmimic.rl.optimizers import build_optimizer
from instantmimic.utils.startup_profiler import startup_nvtx_wrapped
from instantmimic.utils.torch_warp_capture import warp_stream_from_torch

from .bindings import FastMuscleSimulationBindings
from .config_models import (
    MassActionSettings,
    MassPosePolicyOutputSettings,
    MuscleNNSettings,
)
from .muscle_nn import (
    RAW_ZERO_IS_REST_OUTPUT_ACTIVATIONS,
    MassMuscleNN,
    MuscleNNInputNormalizer,
)
from .stable_pd_controller import StablePDController

logger = logging.getLogger(__name__)

_AUTO_TORQUE_SCALE_CACHE_KIND = "auto_torque_scale"


# Accumulated per minibatch: total, target, regularization and bound losses,
# pre-clip gradient norm and the clipped-step indicator.
_MUSCLENN_METRIC_COUNT = 6


def _resolve_musclenn_optimizer_launch_policy(
    *,
    device_type: str,
    optimizer_name: str,
) -> tuple[bool, dict[str, bool] | None]:
    graph_capable = device_type == "cuda" and optimizer_name.strip().lower() in {
        "adam",
        "adamw",
    }
    # The multi-tensor (foreach) kernels update every parameter tensor in a
    # few launches and are capturable, so the captured update records them
    # instead of a kernel chain per parameter. Compiling fused MuscleNN Adam
    # leaves a new AdamW guard to be built inside the later PPO capture in
    # Torch 2.11: ID_MATCH formats a CUDA exp_avg tensor with repr(), whose
    # masked_select/nonzero synchronizes to the CPU and invalidates capture.
    # Keep foreach until the compiled optimizer path is fully prewarmed;
    # the fused CUDA kernel itself can be captured.
    optimizer_kwargs = {"foreach": True} if graph_capable else None
    return graph_capable, optimizer_kwargs


def _controller_type():
    from .controller import FastMuscleController

    return FastMuscleController


class FastMuscleTrainingMixin:
    def _mass_action_settings(self) -> MassActionSettings:
        action = self.settings.action
        assert isinstance(action, MassActionSettings)
        return action

    def _mass_muscle_nn(self) -> MuscleNNSettings:
        """MuscleNN settings with the muscle-model defaults filled in."""
        muscle_nn = self.settings.effective.mass_muscle_nn
        assert muscle_nn is not None
        return muscle_nn

    def _musclenn_collects_rollout_samples(self) -> bool:
        """Whether this environment records MuscleNN training samples.

        Only the training environment configures rollout storage (before its
        first step and any graph capture); benchmark, playback and any other
        environment never does. The storage shape is fixed before capture and
        does not change between graph replays.
        """
        return bool(
            int(getattr(self, "_musclenn_rollout_steps", 0)) > 0
            and self._musclenn_rollout_input_buffer.numel() > 0
        )

    @startup_nvtx_wrapped("fast_muscle:initialize_mass_state")
    def _initialize_mass_state(self) -> None:
        self._musclenn_model: MassMuscleNN | None = None
        self._musclenn_optimizer: torch.optim.Optimizer | None = None
        self._musclenn_torque_helper: MuscleDynamicsLayer | None = None
        self._musclenn_sparse_entry_dim = 0
        self._musclenn_substep_counter = 0
        self._musclenn_last_loss = 0.0
        self._musclenn_last_reg_loss = 0.0
        self._musclenn_last_target_loss = 0.0
        self._musclenn_compile_full_update = False
        self._musclenn_compile_mode = "max-autotune-no-cudagraphs"
        self._musclenn_use_update_graph = False
        self._musclenn_use_shuffle_graph = False
        self._musclenn_update_torch_stream: torch.cuda.Stream | None = None
        self._musclenn_update_graph: Any | None = None
        self._musclenn_shuffle_graph: torch.cuda.CUDAGraph | None = None
        self._musclenn_shuffle_generator: torch.Generator | None = None
        self._musclenn_shuffle_generator_state: torch.Tensor | None = None
        self._musclenn_update_runtime_signature: tuple[object, ...] | None = None
        self._musclenn_graph_signature: tuple[object, ...] | None = None
        self._musclenn_graph_inputs: torch.Tensor | None = None
        self._musclenn_graph_targets: torch.Tensor | None = None
        self._musclenn_graph_contexts: torch.Tensor | None = None
        self._musclenn_graph_valid_mask: torch.Tensor | None = None
        self._musclenn_graph_shuffled_inputs: torch.Tensor | None = None
        self._musclenn_graph_shuffled_targets: torch.Tensor | None = None
        self._musclenn_graph_shuffled_contexts: torch.Tensor | None = None
        self._musclenn_graph_shuffled_valid_mask: torch.Tensor | None = None
        self._musclenn_graph_permutations: torch.Tensor | None = None
        self._musclenn_graph_metrics: torch.Tensor | None = None
        self._musclenn_graph_start_event: torch.cuda.Event | None = None
        self._musclenn_graph_end_event: torch.cuda.Event | None = None
        self._musclenn_update_graph_disabled_after_restore = False
        self._musclenn_compiled_minibatch_ready = False
        self._run_musclenn_graph_minibatch_fn = self._run_musclenn_graph_minibatch
        self._musclenn_update_pending = False
        self._musclenn_rollout_step = 0
        self._musclenn_rollout_steps = 0
        self._musclenn_rollout_dirty = False
        self._musclenn_rollout_random_substep_indices = torch.zeros(
            (0, 0), device=self.device, dtype=torch.int64
        )
        self._musclenn_rollout_random_substep_mask = torch.zeros(
            (0,), device=self.device, dtype=torch.bool
        )
        self._musclenn_rollout_random_substep_mask_f = torch.zeros(
            (0, 1), device=self.device, dtype=torch.float32
        )
        self._musclenn_last_activation = torch.zeros(
            (self.num_envs, self.muscle_count),
            device=self.device,
            dtype=torch.float32,
        )
        self._musclenn_input_buffer = torch.zeros(
            (0, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_target_buffer = torch.zeros(
            (0, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_context_buffer = torch.zeros(
            (0, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_buffer_ptr = 0
        self._musclenn_buffer_filled = 0
        self._musclenn_rollout_input_buffer = torch.zeros(
            (0, 0, 0, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_rollout_target_buffer = torch.zeros(
            (0, 0, 0, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_rollout_context_buffer = torch.zeros(
            (0, 0, 0, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_entry_torque_idx = torch.zeros(
            (0,), device=self.device, dtype=torch.long
        )
        self._musclenn_entry_muscle_idx = torch.zeros(
            (0,), device=self.device, dtype=torch.long
        )
        self._musclenn_entry_flat_idx = torch.zeros(
            (0,), device=self.device, dtype=torch.long
        )
        self._musclenn_step_input = torch.zeros(
            (0, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_sparse_scale = torch.ones(
            (1, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_input_blocks: tuple[tuple[str, int, int], ...] = ()
        self._musclenn_input_block_weights = torch.zeros(
            (0, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_input_column_moments = torch.zeros(
            (4, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_input_block_stats = torch.zeros(
            (2, 0), device=self.device, dtype=torch.float32
        )
        self._musclenn_torque_active_mask: torch.Tensor | None = None
        self._musclenn_torque_scale_inv: torch.Tensor | None = None
        self._mass_active_torque_indices: torch.Tensor | None = None
        self._mass_torque_command: torch.Tensor | None = None
        self._stable_pd_controller: StablePDController | None = None
        self._mass_pose_spd_qfrc: torch.Tensor | None = None
        self._mass_pose_torque_actions: torch.Tensor | None = None
        self._mass_pose_passive_torque_actions: torch.Tensor | None = None
        self._mass_pose_model_target_actions: torch.Tensor | None = None
        self._mass_pose_total_target_actions: torch.Tensor | None = None
        self._mass_pose_active_lower: torch.Tensor | None = None
        self._mass_pose_active_upper: torch.Tensor | None = None
        self._mass_pose_target_qfrc: torch.Tensor | None = None
        self._mass_pose_passive_qfrc: torch.Tensor | None = None
        self._mass_pose_clip_lower_qfrc: torch.Tensor | None = None
        self._mass_pose_clip_upper_qfrc: torch.Tensor | None = None
        self._musclenn_raw_context = torch.zeros(
            (self.num_envs, int(self.helper_obs_context_dim)),
            device=self.device,
            dtype=torch.float32,
        )
        if not self._uses_mass_coordinator:
            return
        muscle_nn = self._mass_muscle_nn()

        if self._uses_mass_pose_coordinator:
            self._initialize_mass_pose_state()

        helper_obs_context_dim = int(self.helper_obs_context_dim)
        include_passive_target = muscle_nn.target_includes_passive_force
        helper_spec = self._build_muscle_helper_runtime_spec(
            obs_context_dim=helper_obs_context_dim,
            # Active-torque supervision matches the policy command directly;
            # total-torque supervision additionally includes passive muscle force.
            include_passive_force_in_output=include_passive_target,
            # MuscleNN reads and fits every non-root coordinate. A zero torque
            # scale only removes a coordinate from the policy action, whose
            # command is then zero there.
            select_active_torque_dofs=False,
        )
        self._musclenn_torque_helper = MuscleDynamicsLayer(helper_spec).to(self.device)
        self._musclenn_torque_helper.prepare_waypoint_wrap_workspace(
            batch_size=self.num_envs
        )
        self._musclenn_torque_helper.eval()
        (
            self._musclenn_entry_torque_idx,
            self._musclenn_entry_muscle_idx,
        ) = self._build_musclenn_sparse_entry_indices(
            self._musclenn_torque_helper.torque_active_torque_idx,
            self._musclenn_torque_helper.torque_active_muscle_idx,
            active_torque_count=int(self.model.nv - self._root_dof_count),
            muscle_count=int(self.muscle_count),
            device=self.device,
        )
        self._musclenn_sparse_entry_dim = int(self._musclenn_entry_torque_idx.numel())
        input_dim = self._musclenn_sparse_entry_dim + int(
            self.model.nv - self._root_dof_count
        )
        self._musclenn_entry_flat_idx = (
            self._musclenn_entry_torque_idx * int(self.muscle_count)
            + self._musclenn_entry_muscle_idx
        )
        self._musclenn_step_input = torch.zeros(
            (self.num_envs, input_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self._musclenn_sparse_scale = torch.ones(
            (1, int(self._musclenn_entry_torque_idx.numel())),
            device=self.device,
            dtype=torch.float32,
        )
        torque_dim = int(self.model.nv - self._root_dof_count)
        self._initialize_mass_command_state(torque_dim)
        self._initialize_musclenn_input_statistics(input_dim)
        self._musclenn_model = MassMuscleNN(
            input_dim=input_dim,
            muscle_count=self.muscle_count,
            hidden_dims=muscle_nn.hidden_dims,
            output_activation=muscle_nn.output_activation,
            input_normalizer=self._build_musclenn_input_normalizer(input_dim),
        ).to(self.device)
        self._musclenn_input_scale_inv = 1.0 / muscle_nn.input_scale
        self._refresh_musclenn_sparse_scale()
        self._musclenn_model.eval()
        musclenn_optimizer_name = muscle_nn.optimizer.type
        graph_capable_musclenn, optimizer_kwargs = (
            _resolve_musclenn_optimizer_launch_policy(
                device_type=self.device.type,
                optimizer_name=musclenn_optimizer_name,
            )
        )
        self._musclenn_optimizer = build_optimizer(
            params=list(self._musclenn_model.named_parameters()),
            optimizer_name=musclenn_optimizer_name,
            learning_rate=muscle_nn.learning_rate,
            optimizer_kwargs=optimizer_kwargs,
            capturable=bool(graph_capable_musclenn),
            device=self.device,
        )
        if graph_capable_musclenn:
            self._initialize_musclenn_optimizer_state_for_graph()
        capacity = muscle_nn.buffer_capacity
        self._musclenn_input_buffer = torch.zeros(
            (capacity, input_dim), device=self.device, dtype=torch.float32
        )
        self._musclenn_target_buffer = torch.zeros(
            (capacity, torque_dim),
            device=self.device,
            dtype=torch.float32,
        )
        if (
            self._musclenn_torque_helper is not None
            and self._musclenn_torque_helper.include_passive_force_in_output
            and helper_obs_context_dim > 0
        ):
            self._musclenn_context_buffer = torch.zeros(
                (capacity, helper_obs_context_dim),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self._musclenn_context_buffer = torch.zeros(
                (0, 0), device=self.device, dtype=torch.float32
            )
        self._musclenn_buffer_ptr = 0
        self._musclenn_buffer_filled = 0

    def _build_musclenn_input_normalizer(
        self, input_dim: int
    ) -> MuscleNNInputNormalizer | None:
        muscle_nn = self._mass_muscle_nn()
        if muscle_nn.input_normalization == "none":
            return None
        return MuscleNNInputNormalizer(
            input_dim,
            clip=muscle_nn.input_normalization_clip,
            min_std=muscle_nn.input_normalization_min_std,
        )

    def _musclenn_input_normalizer(self) -> MuscleNNInputNormalizer | None:
        model = getattr(self, "_musclenn_model", None)
        return getattr(model, "input_normalizer", None)

    def _musclenn_input_block_layout(
        self, input_dim: int
    ) -> tuple[tuple[str, int, int], ...]:
        """Named ranges of the sparse force map and commanded joint torques."""
        entry_dim = int(self._musclenn_sparse_entry_dim)
        return (("J", 0, entry_dim), ("target", entry_dim, input_dim))

    def _initialize_musclenn_input_statistics(self, input_dim: int) -> None:
        """Persistent scratch for the per-block input diagnostics."""
        blocks = tuple(
            block
            for block in self._musclenn_input_block_layout(input_dim)
            if block[2] > block[1]
        )
        weights = torch.zeros(
            (len(blocks), input_dim), device=self.device, dtype=torch.float32
        )
        for row, (_, start, end) in enumerate(blocks):
            weights[row, start:end] = 1.0 / float(end - start)
        self._musclenn_input_blocks = blocks
        self._musclenn_input_block_weights = weights
        # Rows: per-column mean, population variance, mean |x| and deviation.
        self._musclenn_input_column_moments = torch.zeros(
            (4, input_dim), device=self.device, dtype=torch.float32
        )
        # Rows: per-block mean |x| and mean per-column deviation.
        self._musclenn_input_block_stats = torch.zeros(
            (2, len(blocks)), device=self.device, dtype=torch.float32
        )

    @torch.no_grad()
    def _observe_musclenn_rollout_inputs(self, inputs: torch.Tensor) -> None:
        """Input moments of one newly collected rollout batch.

        Runs once per collected rollout, when it is committed to the replay
        buffer at the start of a MuscleNN update and before the update's
        (captured or eager) epochs: it refreshes the per-block diagnostics
        and, with ``input_normalization: running``, merges the batch into the
        normalizer, so this update and the next rollout read the same
        statistics. Only an environment that optimizes MuscleNN commits
        rollouts; benchmark and playback environments receive the statistics
        with the network state.
        """
        moments = getattr(self, "_musclenn_input_column_moments", None)
        if (
            moments is None
            or moments.numel() == 0
            or inputs.numel() == 0
            or int(inputs.shape[-1]) != int(moments.shape[1])
        ):
            return
        count = int(inputs.shape[0])
        column_mean, column_variance, column_mean_abs, column_std = moments.unbind(0)
        torch.mean(inputs, dim=0, out=column_mean)
        torch.var(inputs, dim=0, correction=0, out=column_variance)
        torch.linalg.vector_norm(inputs, ord=1, dim=0, out=column_mean_abs)
        column_mean_abs.div_(float(count))
        torch.sqrt(column_variance, out=column_std)
        block_stats = self._musclenn_input_block_stats
        weights = self._musclenn_input_block_weights
        torch.mv(weights, column_mean_abs, out=block_stats[0])
        torch.mv(weights, column_std, out=block_stats[1])
        normalizer = self._musclenn_input_normalizer()
        if normalizer is not None:
            # The network reads x / input_scale (`_prepare_musclenn_model_input`).
            input_scale_inv = float(getattr(self, "_musclenn_input_scale_inv", 1.0))
            normalizer.update_from_moments(
                column_mean * input_scale_inv,
                column_variance * (input_scale_inv * input_scale_inv),
                count,
            )

    def _musclenn_input_block_metrics(self) -> dict[str, float]:
        """Mean |x| and mean per-column deviation of each raw input block."""
        blocks = getattr(self, "_musclenn_input_blocks", ())
        if not blocks:
            return {}
        mean_abs_values, std_values = self._musclenn_input_block_stats.tolist()
        metrics: dict[str, float] = {}
        for (name, _, _), mean_abs, std in zip(
            blocks, mean_abs_values, std_values, strict=True
        ):
            metrics[f"input_{name}_mean_abs"] = float(mean_abs)
            metrics[f"input_{name}_std"] = float(std)
        return metrics

    def _initialize_mass_command_state(self, torque_dim: int) -> None:
        """Full-width command scratch and the MuscleNN torque normalization.

        The policy commands only the coordinates with a nonzero torque scale;
        MuscleNN and its loss span every non-root coordinate,
        with a zero command and a zero inverse scale on masked ones.
        """
        active_mask = torch.ones(torque_dim, device=self.device, dtype=torch.float32)
        if self._has_masked_torque_dofs:
            self._mass_active_torque_indices = self._active_torque_indices.to(
                device=self.device, dtype=torch.long
            )
            active_mask.zero_()
            active_mask.index_fill_(0, self._mass_active_torque_indices, 1.0)
        self._musclenn_torque_active_mask = active_mask
        self._musclenn_torque_scale_inv = torch.zeros_like(active_mask)
        self._mass_torque_command = torch.zeros(
            (self.num_envs, torque_dim), device=self.device, dtype=torch.float32
        )

    def _mass_policy_torque_command(self) -> torch.Tensor:
        """The policy torque action in full non-root width."""
        if not self._has_masked_torque_dofs:
            return self.actions
        command = self._mass_torque_command
        assert command is not None and self._mass_active_torque_indices is not None
        command.zero_()
        command.index_copy_(1, self._mass_active_torque_indices, self.actions)
        return command

    def _copy_into_mass_torque_command(self, command: torch.Tensor) -> torch.Tensor:
        scratch = self._mass_torque_command
        assert scratch is not None
        if command is not scratch:
            scratch.copy_(command)
        return scratch

    def _initialize_mass_pose_state(self) -> None:
        """Allocate the Stable-PD adapter state only for pose-output MASS."""
        if not self._uses_mass_pose_coordinator:
            return
        action = self._mass_action_settings()
        policy_output = action.policy_output
        assert isinstance(policy_output, MassPosePolicyOutputSettings)
        motion = None
        rest_ctrl = None
        if policy_output.reference == "motion":
            motion = self._require_motion_reference("MASS pose output")
            if motion.motion_ctrl is None:
                raise RuntimeError("MASS pose output requires motion control targets.")
        else:
            rest_ctrl = self._mass_pose_rest_ctrl()
        # Passive coordinates carry no Stable-PD actuator.
        if self.passive_torque_indices:
            actuated_dofs = self.character_torque_indices
        else:
            actuated_dofs = None
        simulation = FastMuscleSimulationBindings(
            model=self.model,
            mjw_model=self.mjw_model,
            mjw_data=self.mjw_data,
            device=self.device,
            num_envs=self.num_envs,
            physics_dt=self.physics_dt,
            step_dt=self.step_dt,
        )
        self._stable_pd_controller = StablePDController(
            simulation=simulation,
            motion=motion,
            root_dof_count=self._root_dof_count,
            forward_control_dynamics=(
                self._require_execution_bindings().refresh_control_dynamics
            ),
            settings=policy_output.stable_pd,
            rest_ctrl=rest_ctrl,
            actuated_dofs=actuated_dofs,
        )
        self._mass_pose_spd_qfrc = torch.zeros(
            (self.num_envs, int(self.model.nv)),
            device=self.device,
            dtype=torch.float32,
        )
        self._mass_pose_torque_actions = torch.zeros(
            (self.num_envs, int(self.model.nv - self._root_dof_count)),
            device=self.device,
            dtype=torch.float32,
        )
        self._mass_pose_passive_torque_actions = torch.zeros_like(
            self._mass_pose_torque_actions
        )
        self._mass_pose_model_target_actions = torch.zeros_like(
            self._mass_pose_torque_actions
        )
        self._mass_pose_total_target_actions = torch.zeros_like(
            self._mass_pose_torque_actions
        )
        self._mass_pose_active_lower = torch.zeros_like(self._mass_pose_torque_actions)
        self._mass_pose_active_upper = torch.zeros_like(self._mass_pose_torque_actions)
        self._mass_pose_target_qfrc = torch.zeros_like(self._mass_pose_spd_qfrc)
        self._mass_pose_passive_qfrc = torch.zeros_like(self._mass_pose_spd_qfrc)
        self._mass_pose_clip_lower_qfrc = torch.zeros_like(self._mass_pose_spd_qfrc)
        self._mass_pose_clip_upper_qfrc = torch.zeros_like(self._mass_pose_spd_qfrc)

    def _mass_pose_rest_ctrl(self) -> torch.Tensor:
        """Position-actuator lengths at the model's qpos0, the rest pose target."""
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        return torch.tensor(
            data.actuator_length, device=self.device, dtype=torch.float32
        )

    @staticmethod
    def _build_unique_musclenn_sparse_entry_indices(
        torque_idx: torch.Tensor,
        muscle_idx: torch.Tensor,
        *,
        active_torque_count: int,
        muscle_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torque_idx_long = torque_idx.detach().to(device=device, dtype=torch.long)
        muscle_idx_long = muscle_idx.detach().to(device=device, dtype=torch.long)
        if torque_idx_long.numel() != muscle_idx_long.numel():
            raise ValueError(
                "MuscleNN sparse torque and muscle index arrays must have the same length."
            )
        if torque_idx_long.numel() == 0:
            empty = torch.zeros((0,), device=device, dtype=torch.long)
            return empty, empty

        active_torque_count = int(active_torque_count)
        muscle_count = int(muscle_count)
        if active_torque_count <= 0 or muscle_count <= 0:
            raise ValueError(
                "MuscleNN sparse entry construction requires positive torque and muscle counts."
            )
        if bool((torque_idx_long < 0).any()) or bool(
            (torque_idx_long >= active_torque_count).any()
        ):
            raise ValueError("MuscleNN sparse torque index is out of range.")
        if bool((muscle_idx_long < 0).any()) or bool(
            (muscle_idx_long >= muscle_count).any()
        ):
            raise ValueError("MuscleNN sparse muscle index is out of range.")

        keys = muscle_idx_long * active_torque_count + torque_idx_long
        unique_keys = torch.unique(keys, sorted=True)
        unique_muscle_idx = torch.div(
            unique_keys,
            active_torque_count,
            rounding_mode="floor",
        )
        unique_torque_idx = unique_keys.remainder(active_torque_count)
        return unique_torque_idx, unique_muscle_idx

    def _build_musclenn_sparse_entry_indices(
        self,
        torque_idx: torch.Tensor,
        muscle_idx: torch.Tensor,
        *,
        active_torque_count: int,
        muscle_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._uses_mass_pose_coordinator:
            return self._build_unique_musclenn_sparse_entry_indices(
                torque_idx,
                muscle_idx,
                active_torque_count=active_torque_count,
                muscle_count=muscle_count,
                device=device,
            )
        return (
            torque_idx.detach().clone().to(device=device, dtype=torch.long),
            muscle_idx.detach().clone().to(device=device, dtype=torch.long),
        )

    def _prepare_musclenn_model_input(self, value: torch.Tensor) -> torch.Tensor:
        return value * float(getattr(self, "_musclenn_input_scale_inv", 1.0))

    def _refresh_musclenn_sparse_scale(self) -> None:
        """Refresh the MuscleNN torque normalization after a scale change.

        The MuscleNN inverse scale is the policy inverse scale with masked
        coordinates zeroed instead of the reciprocal of a clamped zero.
        """
        torque_action_scale_inv = getattr(self, "_torque_action_scale_inv", None)
        command_scale_inv = getattr(self, "_musclenn_torque_scale_inv", None)
        if (
            not bool(getattr(self, "_uses_mass_coordinator", False))
            or torque_action_scale_inv is None
            or command_scale_inv is None
        ):
            return
        torch.mul(
            torque_action_scale_inv,
            self._musclenn_torque_active_mask,
            out=command_scale_inv,
        )
        entry_torque_idx = getattr(self, "_musclenn_entry_torque_idx", None)
        if (
            bool(getattr(self, "_uses_mass_pose_coordinator", False))
            or entry_torque_idx is None
            or int(entry_torque_idx.numel()) <= 0
        ):
            return
        torch.index_select(
            command_scale_inv,
            0,
            entry_torque_idx,
            out=self._musclenn_sparse_scale.view(-1),
        )

    def _write_musclenn_step_input(
        self,
        contribution: torch.Tensor,
        torque_actions: torch.Tensor,
    ) -> torch.Tensor:
        """The instantaneous decoder's input: sparse ``J^T A`` entries, target."""
        target = self._musclenn_step_input
        entry_dim = int(self._musclenn_sparse_entry_dim)
        if entry_dim > 0:
            sparse_target = target[:, :entry_dim]
            torch.index_select(
                contribution.view(contribution.shape[0], -1),
                1,
                self._musclenn_entry_flat_idx,
                out=sparse_target,
            )
            sparse_target.mul_(self._musclenn_sparse_scale)
        target[:, entry_dim:].copy_(torque_actions)
        return target

    @staticmethod
    def _sanitize_musclenn_input_inplace(value: torch.Tensor) -> torch.Tensor:
        """Keep terminated non-finite trajectories out of the shared MuscleNN."""
        return torch.nan_to_num(
            value,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            out=value,
        )

    def _compute_mass_pose_torque_actions(self) -> torch.Tensor:
        spd_qfrc = self._mass_pose_spd_qfrc
        torque_actions = self._mass_pose_torque_actions
        if (
            self._stable_pd_controller is None
            or spd_qfrc is None
            or torque_actions is None
        ):
            raise RuntimeError(
                "MASS pose output requires initialized Stable-PD adapter state."
            )
        self._stable_pd_controller.compute_torque(
            self.actions,
            spd_qfrc,
        )
        torque_actions.copy_(spd_qfrc[:, self._root_dof_count :])
        return torque_actions

    def _write_mass_pose_qfrc(
        self,
        torque: torch.Tensor,
        target_qfrc: torch.Tensor,
    ) -> None:
        target_qfrc.zero_()
        target_qfrc[:, self._root_dof_count :].copy_(torque)

    def _compute_mass_pose_passive_torque_actions(self) -> torch.Tensor:
        passive_actions = self._mass_pose_passive_torque_actions
        passive_qfrc = self._mass_pose_passive_qfrc
        if passive_actions is None or passive_qfrc is None:
            raise RuntimeError("MASS pose passive-force adapter state is unavailable.")
        passive_actions.zero_()
        passive_qfrc.zero_()
        if self._muscle_include_passive == 0.0:
            return passive_actions

        self._prepare_torque_passive_muscle_state()
        passive_qfrc.copy_(self._qfrc_applied_view)
        passive_actions.copy_(passive_qfrc[:, self._root_dof_count :])
        return passive_actions

    def _prepare_mass_pose_targets(
        self,
        spd_torque_actions: torch.Tensor,
        passive_torque_actions: torch.Tensor,
        contribution: torch.Tensor,
    ) -> torch.Tensor:
        active_target = self._mass_pose_model_target_actions
        total_target = self._mass_pose_total_target_actions
        active_lower = self._mass_pose_active_lower
        active_upper = self._mass_pose_active_upper
        target_qfrc = self._mass_pose_target_qfrc
        clip_lower_qfrc = self._mass_pose_clip_lower_qfrc
        clip_upper_qfrc = self._mass_pose_clip_upper_qfrc
        if any(
            value is None
            for value in (
                active_target,
                total_target,
                active_lower,
                active_upper,
                target_qfrc,
                clip_lower_qfrc,
                clip_upper_qfrc,
            )
        ):
            raise RuntimeError("MASS pose target adapter state is unavailable.")
        assert active_target is not None
        assert total_target is not None
        assert active_lower is not None and active_upper is not None
        assert target_qfrc is not None
        assert clip_lower_qfrc is not None and clip_upper_qfrc is not None
        torch.sub(spd_torque_actions, passive_torque_actions, out=active_target)

        torch.sum(torch.clamp(contribution, max=0.0), dim=-1, out=active_lower)
        torch.sum(torch.clamp(contribution, min=0.0), dim=-1, out=active_upper)
        if self._uses_torque_feasible_bounds():
            torch.maximum(active_target, active_lower, out=active_target)
            torch.minimum(active_target, active_upper, out=active_target)

        torch.add(active_target, passive_torque_actions, out=total_target)
        torch.add(active_lower, passive_torque_actions, out=active_lower)
        torch.add(active_upper, passive_torque_actions, out=active_upper)
        self._write_mass_pose_qfrc(total_target, target_qfrc)
        self._write_mass_pose_qfrc(active_lower, clip_lower_qfrc)
        self._write_mass_pose_qfrc(active_upper, clip_upper_qfrc)
        if self._musclenn_torque_helper.include_passive_force_in_output:
            return total_target
        return active_target

    def configure_musclenn_rollout_storage(self, rollout_steps: int) -> None:
        if not self._uses_mass_coordinator:
            return
        rollout_steps = max(0, int(rollout_steps))
        self._musclenn_rollout_step = 0
        self._musclenn_rollout_steps = rollout_steps
        self._musclenn_rollout_dirty = False
        input_dim = int(self._musclenn_input_buffer.shape[-1])
        target_dim = int(self._musclenn_target_buffer.shape[-1])
        decimation = self.control_interval_substeps
        sampling_mode = self._resolve_musclenn_substep_sampling_mode()
        rollout_substeps = (
            1
            if self._reuse_musclenn_activation_across_substeps_enabled()
            else (decimation if sampling_mode == "all" else 1)
        )
        self._musclenn_rollout_input_buffer = torch.zeros(
            (rollout_steps, rollout_substeps, self.num_envs, input_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self._musclenn_rollout_target_buffer = torch.zeros(
            (rollout_steps, rollout_substeps, self.num_envs, target_dim),
            device=self.device,
            dtype=torch.float32,
        )
        helper_obs_context_dim = int(self.helper_obs_context_dim)
        if (
            self._musclenn_torque_helper is not None
            and self._musclenn_torque_helper.include_passive_force_in_output
            and helper_obs_context_dim > 0
        ):
            self._musclenn_rollout_context_buffer = torch.zeros(
                (
                    rollout_steps,
                    rollout_substeps,
                    self.num_envs,
                    helper_obs_context_dim,
                ),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self._musclenn_rollout_context_buffer = torch.zeros(
                (0, 0, 0, 0), device=self.device, dtype=torch.float32
            )
        if (
            sampling_mode == "random"
            and not self._reuse_musclenn_activation_across_substeps_enabled()
        ):
            self._musclenn_rollout_random_substep_indices = torch.zeros(
                (rollout_steps, self.num_envs),
                device=self.device,
                dtype=torch.int64,
            )
            self._musclenn_rollout_random_substep_mask = torch.zeros(
                (self.num_envs,),
                device=self.device,
                dtype=torch.bool,
            )
            self._musclenn_rollout_random_substep_mask_f = torch.zeros(
                (self.num_envs, 1),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self._musclenn_rollout_random_substep_indices = torch.zeros(
                (0, 0), device=self.device, dtype=torch.int64
            )
            self._musclenn_rollout_random_substep_mask = torch.zeros(
                (0,), device=self.device, dtype=torch.bool
            )
            self._musclenn_rollout_random_substep_mask_f = torch.zeros(
                (0, 1), device=self.device, dtype=torch.float32
            )

    def set_musclenn_rollout_step(self, step: int) -> None:
        if not self._uses_mass_coordinator:
            return
        self._musclenn_rollout_step = max(0, int(step))

    def mark_musclenn_rollout_collected(self) -> None:
        if (
            not self._uses_mass_coordinator
            or self._musclenn_rollout_steps <= 0
            or self._musclenn_rollout_input_buffer.numel() == 0
        ):
            return
        self._musclenn_rollout_dirty = True

    def prepare_musclenn_rollout_sampling(self) -> None:
        if not self._uses_mass_coordinator:
            return
        # The rollout reads the network the begun update is still writing.
        self._require_no_pending_musclenn_update()
        if self._reuse_musclenn_activation_across_substeps_enabled():
            return
        if self._resolve_musclenn_substep_sampling_mode() != "random":
            return
        if self._musclenn_rollout_random_substep_indices.numel() == 0:
            return
        decimation = self.control_interval_substeps
        self._musclenn_rollout_random_substep_indices.random_(0, decimation)

    def _refresh_rest_pose_context_for_auto_torque_scale(self) -> None:
        mjw_model = getattr(self, "mjw_model", None)
        mjw_data = getattr(self, "mjw_data", None)
        if mjw_model is None or mjw_data is None:
            return
        # This hook runs while DirectRLEnv is still constructing its control
        # pipeline, before the persistent forward scratch is allocated.  It is
        # the one production bootstrap call allowed to use upstream forward;
        # every post-construction runtime refresh goes through _run_mjw_forward.
        mjw.forward(mjw_model, mjw_data)

    @torch.no_grad()
    @startup_nvtx_wrapped("fast_muscle:initialize_auto_torque_action_scale")
    def _initialize_auto_torque_action_scale_from_rest_bounds(self) -> bool:
        if not bool(getattr(self, "_torque_action_scale_auto_from_rest_bounds", False)):
            return False
        action_mode = str(getattr(self, "_action_mode", ""))
        mass_torque_output = bool(
            action_mode == "mass"
            and not bool(getattr(self, "_uses_mass_pose_coordinator", False))
        )
        if action_mode != "torque" and not mass_torque_output:
            return False
        release_transient_helper = action_mode == "torque"

        def finish(initialized: bool) -> bool:
            if release_transient_helper:
                self._muscle_dynamics_helper = None
                self._muscle_dynamics_helper_spec = None
            return initialized

        if mass_torque_output:
            # MASS already owns the exact dynamics layer used to fit MuscleNN.
            # Reuse it for rest-pose normalization instead of allocating the
            # transient helper required by direct-torque control.
            helper = getattr(self, "_musclenn_torque_helper", None)
            helper_spec = None
        else:
            helper = getattr(self, "_muscle_dynamics_helper", None)
            helper_spec = getattr(self, "_muscle_dynamics_helper_spec", None)
        helper_context_dim = int(self.helper_obs_context_dim)
        if helper is None and helper_spec is None or int(helper_context_dim) <= 0:
            return finish(False)
        raw_context_builder = getattr(self, "_build_musclenn_raw_context", None)
        if not callable(raw_context_builder):
            return finish(False)

        active_idx = _controller_type()._torque_action_scale_active_indices(self)
        cache_key_builder = getattr(
            self,
            "_auto_torque_action_scale_cache_key",
            None,
        )
        cache_key = (
            cache_key_builder(helper=helper, helper_spec=helper_spec)
            if callable(cache_key_builder)
            else None
        )
        cached_target_active = (
            load_pickle_cache(_AUTO_TORQUE_SCALE_CACHE_KIND, cache_key)
            if cache_key
            else None
        )
        cached_target_is_valid = bool(
            torch.is_tensor(cached_target_active)
            and int(cached_target_active.numel()) == int(active_idx.numel())
            and bool(torch.all(torch.isfinite(cached_target_active)).item())
            and bool(torch.all(cached_target_active > 0.0).item())
        )
        if cached_target_is_valid:
            target_active = cached_target_active.to(
                dtype=self._torque_action_scale.dtype,
            ).reshape(-1)
        else:
            if not mass_torque_output:
                ensure_helper = getattr(self, "_ensure_muscle_dynamics_helper", None)
                helper = ensure_helper() if callable(ensure_helper) else helper
            if helper is None:
                return finish(False)
            refresh_rest_pose_context = getattr(
                self,
                "_refresh_rest_pose_context_for_auto_torque_scale",
                None,
            )
            if callable(refresh_rest_pose_context):
                refresh_rest_pose_context()
            raw_context = raw_context_builder()
            if not isinstance(raw_context, torch.Tensor) or raw_context.numel() == 0:
                return finish(False)
            lower, upper = helper.compute_activation_torque_bounds(raw_context[:1])
            if not bool((torch.isfinite(lower) & torch.isfinite(upper)).all().item()):
                return finish(False)
            half_range = 0.5 * (upper - lower)
            if half_range.ndim != 2 or int(half_range.shape[0]) == 0:
                return finish(False)
            target_active = half_range[0].to(
                device=self._torque_action_scale.device,
                dtype=self._torque_action_scale.dtype,
            )
            if cache_key and int(target_active.numel()) == int(active_idx.numel()):
                store_pickle_cache(
                    _AUTO_TORQUE_SCALE_CACHE_KIND,
                    cache_key,
                    target_active.detach().cpu(),
                )
        if int(target_active.numel()) != int(active_idx.numel()):
            logger.warning(
                "Rest-pose torque scale init skipped: active_idx=%d target=%d.",
                int(active_idx.numel()),
                int(target_active.numel()),
            )
            return finish(False)

        target_active = torch.clamp(target_active, min=1.0e-3)
        if target_active.device.type == "cpu":
            target_scale = (
                self._torque_action_scale.detach().cpu().to(dtype=target_active.dtype)
            )
            target_scale.index_copy_(0, active_idx.detach().cpu(), target_active)
        else:
            target_scale = self._torque_action_scale.detach().clone()
            target_scale.index_copy_(
                0,
                active_idx.to(device=target_scale.device),
                target_active,
            )
        _controller_type()._apply_torque_action_scale_(self, target_scale, agent=None)
        target_active_stats = target_active.detach()
        logger.info(
            "Initialized torque_action_scale from rest-pose muscle bounds "
            "(mean=%.6g, min=%.6g, max=%.6g).",
            float(target_active_stats.mean().item()),
            float(target_active_stats.min().item()),
            float(target_active_stats.max().item()),
        )
        return finish(True)

    def _auto_torque_action_scale_cache_key(
        self,
        *,
        helper: MuscleDynamicsLayer | None,
        helper_spec: MuscleLayerRuntimeSpec | None = None,
    ) -> str | None:
        source_path = getattr(self, "_source_mjcf_path", "")
        if not source_path:
            return None
        try:
            source_signature = collect_xml_dependency_signatures(Path(source_path))
        except (OSError, RuntimeError, ET.ParseError):
            return None
        metadata_signature = None
        metadata_path = self.settings.muscle.metadata_path
        if metadata_path:
            try:
                metadata_signature = file_signature(Path(metadata_path))
            except (OSError, RuntimeError):
                return None
        if helper is not None:
            include_passive = helper.include_passive_force_in_output
        elif helper_spec is not None:
            include_passive = helper_spec.mechanics.include_passive_force_in_output
        else:
            return None
        return cache_key(
            _AUTO_TORQUE_SCALE_CACHE_KIND,
            {
                "format_version": 2,
                "source": source_signature,
                "metadata": metadata_signature,
                "muscle_settings": self.settings.muscle.model_dump(mode="json"),
                "length_range": list(self.settings.effective.length_range),
                "action_mode": self._action_mode,
                "include_passive_force_in_output": bool(include_passive),
                "active_torque_indices": self._torque_action_scale_active_indices()
                .cpu()
                .tolist(),
                "minimum_scale": 1.0e-3,
            },
        )

    @torch.no_grad()
    def sync_policy_action_scales(self, agent: Any) -> bool:
        if agent is None:
            return False
        snapshot = _controller_type().get_policy_action_scale_snapshot(self)
        if snapshot is None:
            return False
        set_action_scales = getattr(agent, "set_action_scales", None)
        if not callable(set_action_scales):
            return False
        set_action_scales(
            mean_scale=snapshot.mean_scale,
            std_scale=snapshot.std_scale,
        )
        return True

    def _torque_action_scale_active_indices(self) -> torch.Tensor:
        if getattr(self, "_has_masked_torque_dofs", False):
            return self._active_torque_indices.long()
        return torch.arange(
            int(self._torque_action_scale.numel()),
            device=self._torque_action_scale.device,
            dtype=torch.long,
        )

    @torch.no_grad()
    def _apply_torque_action_scale_(
        self,
        new_scale: torch.Tensor,
        *,
        agent: Any | None = None,
    ) -> None:
        target_source = new_scale.detach()
        target = target_source.to(
            device=self._torque_action_scale.device,
            dtype=self._torque_action_scale.dtype,
        )
        if target.shape != self._torque_action_scale.shape:
            raise ValueError(
                "Unexpected torque scale shape: "
                f"expected {tuple(self._torque_action_scale.shape)}, got {tuple(target.shape)}."
            )
        valid = torch.isfinite(target) & (target > 0.0)
        target = torch.where(valid, target, self._torque_action_scale)
        self._torque_action_scale.copy_(target)
        self._torque_action_scale_inv.copy_(
            torch.reciprocal(torch.clamp(self._torque_action_scale, min=1.0e-6))
        )
        _controller_type()._refresh_musclenn_sparse_scale(self)
        _controller_type().sync_policy_action_scales(self, agent)

    def _build_musclenn_raw_context(self) -> torch.Tensor:
        """Pack current geometry and velocity for the environment's muscle model."""
        target = self._musclenn_raw_context
        end = self._write_muscle_layer_context_block(target, 0)
        expected_dim = int(self.helper_obs_context_dim)
        if end != expected_dim:
            raise RuntimeError(
                "Muscle-layer raw context width mismatch: "
                f"expected {expected_dim}, wrote {int(end)}."
            )
        return target

    def _record_musclenn_rollout_batch(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        raw_context: torch.Tensor | None = None,
    ) -> None:
        """Record a substep's input, physical torque target and optional geometry."""
        if (
            inputs.numel() == 0
            or self._musclenn_rollout_steps <= 0
            or self._musclenn_rollout_input_buffer.numel() == 0
        ):
            return
        step = int(self._musclenn_rollout_step)
        substep = int(self._musclenn_substep_counter)
        if (
            step >= self._musclenn_rollout_steps
            or substep >= self.control_interval_substeps
        ):
            return
        self._musclenn_rollout_dirty = True
        if self._reuse_musclenn_activation_across_substeps_enabled():
            if substep > 0:
                return
            self._musclenn_rollout_input_buffer[step, 0].copy_(inputs)
            self._musclenn_rollout_target_buffer[step, 0].copy_(target)
            if (
                raw_context is not None
                and self._musclenn_rollout_context_buffer.numel() != 0
            ):
                self._musclenn_rollout_context_buffer[step, 0].copy_(raw_context)
            return
        if self._resolve_musclenn_substep_sampling_mode() == "random":
            selected_substeps = self._musclenn_rollout_random_substep_indices[step]
            random_mask = getattr(self, "_musclenn_rollout_random_substep_mask", None)
            if random_mask is None or random_mask.shape != (inputs.shape[0],):
                random_mask = torch.zeros(
                    (inputs.shape[0],),
                    device=inputs.device,
                    dtype=torch.bool,
                )
                self._musclenn_rollout_random_substep_mask = random_mask
            random_mask_f = getattr(
                self, "_musclenn_rollout_random_substep_mask_f", None
            )
            if random_mask_f is None or random_mask_f.shape != (inputs.shape[0], 1):
                random_mask_f = torch.zeros(
                    (inputs.shape[0], 1),
                    device=inputs.device,
                    dtype=inputs.dtype,
                )
                self._musclenn_rollout_random_substep_mask_f = random_mask_f
            torch.eq(
                selected_substeps,
                substep,
                out=random_mask,
            )
            random_mask_f.copy_(random_mask.view(-1, 1))
            self._musclenn_rollout_input_buffer[step, 0].lerp_(
                inputs,
                random_mask_f,
            )
            self._musclenn_rollout_target_buffer[step, 0].lerp_(
                target,
                random_mask_f,
            )
            if (
                raw_context is not None
                and self._musclenn_rollout_context_buffer.numel() != 0
            ):
                self._musclenn_rollout_context_buffer[step, 0].lerp_(
                    raw_context,
                    random_mask_f,
                )
            return
        self._musclenn_rollout_input_buffer[step, substep].copy_(inputs)
        self._musclenn_rollout_target_buffer[step, substep].copy_(target)
        if (
            raw_context is not None
            and self._musclenn_rollout_context_buffer.numel() != 0
        ):
            self._musclenn_rollout_context_buffer[step, substep].copy_(raw_context)

    def _get_current_musclenn_rollout_view(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        if self._musclenn_rollout_input_buffer.numel() == 0:
            return None
        inputs_all = self._musclenn_rollout_input_buffer.view(
            -1, self._musclenn_rollout_input_buffer.shape[-1]
        )
        target_all = self._musclenn_rollout_target_buffer.view(
            -1, self._musclenn_rollout_target_buffer.shape[-1]
        )
        if inputs_all.shape[0] == 0:
            return None
        context_all: torch.Tensor | None = None
        if self._musclenn_rollout_context_buffer.numel() != 0:
            context_all = self._musclenn_rollout_context_buffer.view(
                -1, self._musclenn_rollout_context_buffer.shape[-1]
            )
        return inputs_all, target_all, context_all

    def _append_musclenn_ring_buffer(
        self,
        inputs_all: torch.Tensor,
        target_all: torch.Tensor,
        context_all: torch.Tensor | None,
    ) -> None:
        input_buffer = getattr(self, "_musclenn_input_buffer", None)
        target_buffer = getattr(self, "_musclenn_target_buffer", None)
        context_buffer = getattr(self, "_musclenn_context_buffer", None)
        if input_buffer is None or target_buffer is None:
            return
        capacity = int(input_buffer.shape[0])
        if capacity <= 0 or inputs_all.shape[0] == 0:
            return
        if inputs_all.shape[0] >= capacity:
            input_buffer.copy_(inputs_all[-capacity:])
            target_buffer.copy_(target_all[-capacity:])
            if (
                context_all is not None
                and context_buffer is not None
                and context_buffer.numel() != 0
            ):
                context_buffer.copy_(context_all[-capacity:])
            self._musclenn_buffer_ptr = 0
            self._musclenn_buffer_filled = capacity
            return
        ptr = int(getattr(self, "_musclenn_buffer_ptr", 0))
        batch = int(inputs_all.shape[0])
        end = ptr + batch
        if end <= capacity:
            input_buffer[ptr:end].copy_(inputs_all)
            target_buffer[ptr:end].copy_(target_all)
            if (
                context_all is not None
                and context_buffer is not None
                and context_buffer.numel() != 0
            ):
                context_buffer[ptr:end].copy_(context_all)
        else:
            first = capacity - ptr
            second = end - capacity
            input_buffer[ptr:].copy_(inputs_all[:first])
            input_buffer[:second].copy_(inputs_all[first:])
            target_buffer[ptr:].copy_(target_all[:first])
            target_buffer[:second].copy_(target_all[first:])
            if (
                context_all is not None
                and context_buffer is not None
                and context_buffer.numel() != 0
            ):
                context_buffer[ptr:].copy_(context_all[:first])
                context_buffer[:second].copy_(context_all[first:])
        self._musclenn_buffer_ptr = end % capacity
        filled = int(getattr(self, "_musclenn_buffer_filled", 0))
        self._musclenn_buffer_filled = min(capacity, filled + batch)

    def _commit_musclenn_rollout_to_ring_buffer(self) -> None:
        filled = int(getattr(self, "_musclenn_buffer_filled", 0))
        if not bool(getattr(self, "_musclenn_rollout_dirty", False)) and filled > 0:
            return
        rollout_view = self._get_current_musclenn_rollout_view()
        if rollout_view is None:
            self._musclenn_rollout_dirty = False
            return
        inputs_all, target_all, context_all = rollout_view
        self._observe_musclenn_rollout_inputs(inputs_all)
        self._append_musclenn_ring_buffer(inputs_all, target_all, context_all)
        self._musclenn_rollout_dirty = False

    def _get_musclenn_buffer_view(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        self._commit_musclenn_rollout_to_ring_buffer()
        input_buffer = getattr(self, "_musclenn_input_buffer", None)
        target_buffer = getattr(self, "_musclenn_target_buffer", None)
        context_buffer = getattr(self, "_musclenn_context_buffer", None)
        filled = int(getattr(self, "_musclenn_buffer_filled", 0))
        if input_buffer is None or target_buffer is None or input_buffer.numel() == 0:
            return self._get_current_musclenn_rollout_view()
        if filled <= 0:
            return self._get_current_musclenn_rollout_view()
        capacity = int(input_buffer.shape[0])
        if filled < capacity:
            inputs_all = input_buffer[:filled]
            target_all = target_buffer[:filled]
            context_all: torch.Tensor | None = None
            if context_buffer is not None and context_buffer.numel() != 0:
                context_all = context_buffer[:filled]
            return inputs_all, target_all, context_all
        ptr = int(getattr(self, "_musclenn_buffer_ptr", 0))
        if ptr == 0:
            inputs_all = input_buffer
            target_all = target_buffer
            context_all = (
                context_buffer
                if context_buffer is not None and context_buffer.numel() != 0
                else None
            )
            return inputs_all, target_all, context_all
        inputs_all = torch.cat(
            [
                input_buffer[ptr:],
                input_buffer[:ptr],
            ],
            dim=0,
        )
        target_all = torch.cat(
            [
                target_buffer[ptr:],
                target_buffer[:ptr],
            ],
            dim=0,
        )
        context_all = None
        if context_buffer is not None and context_buffer.numel() != 0:
            context_all = torch.cat(
                [
                    context_buffer[ptr:],
                    context_buffer[:ptr],
                ],
                dim=0,
            )
        return inputs_all, target_all, context_all

    @staticmethod
    def _filter_finite_musclenn_samples(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Exclude terminal/nonfinite rows from nonlinear auxiliary training."""
        if context is None:
            return inputs, targets, None
        finite = torch.isfinite(inputs).all(dim=-1)
        finite.logical_and_(torch.isfinite(targets).all(dim=-1))
        finite.logical_and_(torch.isfinite(context).all(dim=-1))
        return inputs[finite], targets[finite], context[finite]

    @staticmethod
    def _compute_musclenn_bound_penalty(
        activation_pre_clamp: torch.Tensor,
        *,
        lower_tolerance: float = 0.0,
        upper_tolerance: float = 0.0,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        lower_violation = torch.clamp(
            -(activation_pre_clamp + float(lower_tolerance)),
            min=0.0,
        )
        upper_violation = torch.clamp(
            activation_pre_clamp - (1.0 + float(upper_tolerance)),
            min=0.0,
        )
        penalty = lower_violation.square() + upper_violation.square()
        if sample_weight is None:
            return penalty.mean()
        denominator = sample_weight.sum().clamp_min(1.0) * int(penalty.shape[1])
        return (penalty * sample_weight.view(-1, 1)).sum() / denominator

    def _clip_musclenn_grad_inplace(
        self, limit: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clip the MuscleNN gradient; return its pre-clip norm and whether
        clipping changed it (both 0-d tensors, read without a sync)."""
        assert self._musclenn_model is not None
        grads = [
            param.grad
            for param in self._musclenn_model.parameters()
            if param.grad is not None
        ]
        if not grads:
            zero = torch.zeros((), device=self.device, dtype=torch.float32)
            return zero, zero.clone()
        if self._uses_mass_pose_coordinator:
            # The pose coordinator clips by the largest absolute gradient entry.
            bound = float(limit)
            grad_norm = torch.nn.utils.get_total_norm(grads)
            clipped = torch.stack([grad.abs().amax() for grad in grads]).amax() > bound
            for grad in grads:
                grad.clamp_(min=-bound, max=bound)
            return grad_norm, clipped.to(torch.float32)

        # Torque-output MASS historically used global norm clipping. Applying the
        # pose controller's elementwise clamp here leaves the aggregate gradient
        # effectively unbounded for its substantially larger MuscleNN.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._musclenn_model.parameters(),
            float(limit),
        )
        return grad_norm, self._musclenn_grad_norm_was_clipped(grad_norm, limit)

    @staticmethod
    def _musclenn_grad_norm_was_clipped(
        grad_norm: torch.Tensor, limit: float
    ) -> torch.Tensor:
        # clip_grad_norm_ scales by min(1, limit / (norm + 1e-6)).
        return ((grad_norm + 1.0e-6) > float(limit)).to(torch.float32)

    def _initialize_musclenn_optimizer_state_for_graph(self) -> None:
        if self.device.type != "cuda" or self._musclenn_optimizer is None:
            return
        assert self._musclenn_model is not None
        saved_params = [
            param.detach().clone() for param in self._musclenn_model.parameters()
        ]
        with torch.no_grad():
            self._musclenn_optimizer.zero_grad(set_to_none=True)
            for param in self._musclenn_model.parameters():
                param.grad = torch.zeros_like(param)
            self._musclenn_optimizer.step()
            self._musclenn_optimizer.zero_grad(set_to_none=False)
            for group in self._musclenn_optimizer.param_groups:
                for param in group["params"]:
                    state = self._musclenn_optimizer.state.get(param)
                    if not state:
                        continue
                    step = state.get("step")
                    if torch.is_tensor(step):
                        step.zero_()
                    else:
                        state["step"] = 0
                    exp_avg = state.get("exp_avg")
                    if torch.is_tensor(exp_avg):
                        exp_avg.zero_()
                    exp_avg_sq = state.get("exp_avg_sq")
                    if torch.is_tensor(exp_avg_sq):
                        exp_avg_sq.zero_()
            for param, saved_param in zip(
                self._musclenn_model.parameters(), saved_params, strict=True
            ):
                param.copy_(saved_param)
            self._musclenn_optimizer.zero_grad(set_to_none=False)

    def _invalidate_musclenn_update_graphs(self) -> None:
        stream = getattr(self, "_musclenn_update_torch_stream", None)
        if stream is not None:
            stream.synchronize()
        self._musclenn_update_graph = None
        self._musclenn_shuffle_graph = None
        self._musclenn_graph_signature = None
        self._musclenn_compiled_minibatch_ready = False
        self._run_musclenn_graph_minibatch_fn = self._run_musclenn_graph_minibatch

    def configure_musclenn_update_runtime(
        self,
        *,
        full_update_compile: bool,
        compile_mode: str,
        use_update_graph: bool,
        use_shuffle_graph: bool,
    ) -> None:
        """Mirror the PPO update runtime for the auxiliary MuscleNN optimizer.

        full_update_compile is the caller's derived condition (runtime
        compile together with the update graph); this runtime additionally
        requires its own capture support below.
        """
        optimizer = getattr(self, "_musclenn_optimizer", None)
        optimizer_capturable = bool(
            optimizer is not None
            and optimizer.param_groups
            and all(
                bool(group.get("capturable", False)) for group in optimizer.param_groups
            )
        )
        graph_enabled = bool(
            use_update_graph
            and self.device.type == "cuda"
            and self._uses_mass_coordinator
            and optimizer_capturable
            and self._musclenn_graph_runtime_is_supported()
            and not bool(
                getattr(self, "_musclenn_update_graph_disabled_after_restore", False)
            )
        )
        runtime_signature = (
            bool(full_update_compile and graph_enabled),
            str(compile_mode),
            graph_enabled,
            bool(use_shuffle_graph and graph_enabled),
        )
        if runtime_signature == getattr(
            self, "_musclenn_update_runtime_signature", None
        ):
            return

        if use_update_graph and self._uses_mass_coordinator and not graph_enabled:
            logger.info(
                "MuscleNN update CUDA graph unavailable; using eager updates "
                "(reason=%s).",
                "unsupported_optimizer_or_buffer_contract",
            )

        self._invalidate_musclenn_update_graphs()
        self._musclenn_compile_full_update = bool(full_update_compile and graph_enabled)
        self._musclenn_compile_mode = str(compile_mode)
        self._musclenn_use_update_graph = graph_enabled
        self._musclenn_use_shuffle_graph = bool(use_shuffle_graph and graph_enabled)
        # Capture on a dedicated stream and prewarm before the repo's Warp-backed
        # PPO/rollout graphs so this graph owns a stable Torch graph memory pool.
        self._musclenn_update_torch_stream = (
            torch.cuda.Stream(device=self.device) if graph_enabled else None
        )
        self._musclenn_shuffle_generator = (
            torch.Generator(device=self.device)
            if self._musclenn_use_shuffle_graph
            else None
        )
        if self._musclenn_shuffle_generator is not None:
            pending_state = getattr(self, "_musclenn_shuffle_generator_state", None)
            if pending_state is not None:
                self._musclenn_shuffle_generator.set_state(pending_state)
                self._musclenn_shuffle_generator_state = None
            else:
                shuffle_seed = int(
                    torch.empty((), dtype=torch.int64).random_(0, 2**31 - 1).item()
                )
                self._musclenn_shuffle_generator.manual_seed(shuffle_seed)
        self._musclenn_update_runtime_signature = runtime_signature

    def _musclenn_warp_stream_scope(self):
        """Launch the helper's Warp kernels on the current Torch stream.

        The MuscleNN loss runs the helper's Warp kernels between Torch ops;
        without this they would use Warp's own stream, racing the Torch
        stream eagerly and escaping a Torch CUDA-graph capture.
        """
        if self.device.type != "cuda":
            return nullcontext()
        return wp.ScopedStream(
            warp_stream_from_torch(torch.cuda.current_stream(self.device)),
            sync_enter=False,
            sync_exit=False,
        )

    def _musclenn_graph_runtime_is_supported(self) -> bool:
        input_buffer = getattr(self, "_musclenn_input_buffer", None)
        target_buffer = getattr(self, "_musclenn_target_buffer", None)
        torque_helper = getattr(self, "_musclenn_torque_helper", None)
        if input_buffer is None or target_buffer is None or torque_helper is None:
            return False
        capacity = int(input_buffer.shape[0])
        batch_size = min(self._mass_action_settings().muscle_nn.batch_size, capacity)
        return bool(
            capacity > 0
            and batch_size > 0
            and capacity % batch_size == 0
            and self._mass_action_settings().muscle_nn.epochs > 0
            and (
                not bool(torque_helper.include_passive_force_in_output)
                # Total-force supervision uses the physics helper eagerly;
                # the captured sparse loss fits the active-force map.
            )
        )

    def _musclenn_graph_buffer_is_eligible(self) -> bool:
        if not bool(getattr(self, "_musclenn_use_update_graph", False)):
            return False
        input_buffer = getattr(self, "_musclenn_input_buffer", None)
        if input_buffer is None or not self._musclenn_graph_runtime_is_supported():
            return False
        capacity = int(input_buffer.shape[0])
        return bool(int(getattr(self, "_musclenn_buffer_filled", 0)) == capacity)

    def _musclenn_graph_workspace_signature(self) -> tuple[object, ...]:
        input_buffer = self._musclenn_input_buffer
        target_buffer = self._musclenn_target_buffer
        capacity = int(input_buffer.shape[0])
        return (
            capacity,
            int(input_buffer.shape[1]),
            int(target_buffer.shape[1]),
            (
                int(self._musclenn_context_buffer.shape[1])
                if self._musclenn_context_buffer.numel() > 0
                else 0
            ),
            min(self._mass_action_settings().muscle_nn.batch_size, capacity),
            self._mass_action_settings().muscle_nn.epochs,
            int(self._musclenn_sparse_entry_dim),
            id(self._musclenn_model),
            id(self._musclenn_optimizer),
            self._mass_action_settings().muscle_nn.grad_clip,
            self._mass_action_settings().muscle_nn.regularization_weight,
            self._mass_action_settings().muscle_nn.bound_loss_weight,
            self._mass_action_settings().muscle_nn.bound_tolerance[0],
            self._mass_action_settings().muscle_nn.bound_tolerance[1],
            self._mass_action_settings().muscle_nn.target_scale,
            self._mass_action_settings().muscle_nn.target_loss_weight,
            bool(self._uses_mass_pose_coordinator),
            bool(self._has_masked_torque_dofs),
        )

    def _ensure_musclenn_graph_workspace(self) -> None:
        signature = self._musclenn_graph_workspace_signature()
        if signature == getattr(self, "_musclenn_graph_signature", None):
            return
        stream = self._musclenn_update_torch_stream
        assert stream is not None
        stream.synchronize()
        capacity = int(self._musclenn_input_buffer.shape[0])
        epochs = self._mass_action_settings().muscle_nn.epochs
        self._musclenn_graph_inputs = torch.empty_like(self._musclenn_input_buffer)
        self._musclenn_graph_targets = torch.empty_like(self._musclenn_target_buffer)
        self._musclenn_graph_contexts = torch.empty_like(self._musclenn_context_buffer)
        self._musclenn_graph_valid_mask = torch.ones(
            capacity, device=self.device, dtype=torch.float32
        )
        self._musclenn_graph_shuffled_inputs = torch.empty_like(
            self._musclenn_input_buffer
        )
        self._musclenn_graph_shuffled_targets = torch.empty_like(
            self._musclenn_target_buffer
        )
        self._musclenn_graph_shuffled_contexts = torch.empty_like(
            self._musclenn_context_buffer
        )
        self._musclenn_graph_shuffled_valid_mask = torch.ones_like(
            self._musclenn_graph_valid_mask
        )
        self._musclenn_graph_permutations = torch.empty(
            (epochs, capacity), device=self.device, dtype=torch.int32
        )
        self._musclenn_graph_metrics = torch.zeros(
            _MUSCLENN_METRIC_COUNT, device=self.device, dtype=torch.float32
        )
        self._musclenn_graph_start_event = torch.cuda.Event(enable_timing=True)
        self._musclenn_graph_end_event = torch.cuda.Event(enable_timing=True)
        self._musclenn_update_graph = None
        self._musclenn_shuffle_graph = None
        self._musclenn_compiled_minibatch_ready = False
        self._run_musclenn_graph_minibatch_fn = self._run_musclenn_graph_minibatch
        self._musclenn_graph_signature = signature

    def _stage_musclenn_graph_inputs(self) -> None:
        graph_inputs = self._musclenn_graph_inputs
        graph_targets = self._musclenn_graph_targets
        graph_contexts = self._musclenn_graph_contexts
        graph_valid_mask = self._musclenn_graph_valid_mask
        assert graph_inputs is not None and graph_targets is not None
        assert graph_contexts is not None and graph_valid_mask is not None
        capacity = int(self._musclenn_input_buffer.shape[0])
        ptr = int(getattr(self, "_musclenn_buffer_ptr", 0))
        if ptr == 0:
            graph_inputs.copy_(self._musclenn_input_buffer)
            graph_targets.copy_(self._musclenn_target_buffer)
            if graph_contexts.numel() > 0:
                graph_contexts.copy_(self._musclenn_context_buffer)
        else:
            tail = capacity - ptr
            graph_inputs[:tail].copy_(self._musclenn_input_buffer[ptr:])
            graph_inputs[tail:].copy_(self._musclenn_input_buffer[:ptr])
            graph_targets[:tail].copy_(self._musclenn_target_buffer[ptr:])
            graph_targets[tail:].copy_(self._musclenn_target_buffer[:ptr])
            if graph_contexts.numel() > 0:
                graph_contexts[:tail].copy_(self._musclenn_context_buffer[ptr:])
                graph_contexts[tail:].copy_(self._musclenn_context_buffer[:ptr])

        valid = torch.isfinite(graph_inputs).all(dim=-1)
        valid.logical_and_(torch.isfinite(graph_targets).all(dim=-1))
        if graph_contexts.numel() > 0:
            valid.logical_and_(torch.isfinite(graph_contexts).all(dim=-1))
            torch.nan_to_num(
                graph_contexts,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
                out=graph_contexts,
            )
        graph_valid_mask.copy_(valid)
        torch.nan_to_num(
            graph_inputs, nan=0.0, posinf=0.0, neginf=0.0, out=graph_inputs
        )
        torch.nan_to_num(
            graph_targets, nan=0.0, posinf=0.0, neginf=0.0, out=graph_targets
        )

    def _generate_musclenn_graph_permutations(
        self, generator: torch.Generator | None = None
    ) -> None:
        assert self._musclenn_graph_permutations is not None
        capacity = int(self._musclenn_graph_permutations.shape[1])
        for permutation in self._musclenn_graph_permutations:
            torch.randperm(capacity, generator=generator, out=permutation)

    def _capture_musclenn_shuffle_graph(self) -> None:
        generator = self._musclenn_shuffle_generator
        stream = self._musclenn_update_torch_stream
        assert generator is not None and stream is not None
        saved_generator_state = generator.get_state()
        permutations = self._musclenn_graph_permutations
        assert permutations is not None
        saved_permutations = permutations.clone()
        try:
            with torch.cuda.stream(stream):
                with torch.cuda.nvtx.range("warmup_musclenn_shuffle_graph"):
                    for _ in range(3):
                        self._generate_musclenn_graph_permutations(generator)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            graph.register_generator_state(generator)
            with torch.cuda.nvtx.range("capture_musclenn_shuffle_graph"):
                with torch.cuda.graph(graph, stream=stream):
                    self._generate_musclenn_graph_permutations(generator)
            stream.synchronize()
            generator.set_state(saved_generator_state)
        except Exception as error:
            original_error = error
            for component, operation in (
                ("synchronizing the shuffle stream", stream.synchronize),
                (
                    "restoring shuffle RNG",
                    lambda: generator.set_state(saved_generator_state),
                ),
                (
                    "restoring shuffle workspace",
                    lambda: permutations.copy_(saved_permutations),
                ),
            ):
                try:
                    operation()
                except Exception as cleanup_error:  # noqa: BLE001 - Preserve the original shuffle capture failure during cleanup.
                    original_error.add_note(
                        f"MuscleNN shuffle cleanup failed while {component}: {cleanup_error!r}. Restoration is best-effort."
                    )
            raise
        self._musclenn_shuffle_graph = graph

    def _refresh_musclenn_graph_permutations(self) -> None:
        if self._musclenn_use_shuffle_graph:
            if self._musclenn_shuffle_graph is None:
                self._capture_musclenn_shuffle_graph()
            assert self._musclenn_shuffle_graph is not None
            with torch.cuda.nvtx.range("replay_musclenn_shuffle_graph"):
                self._musclenn_shuffle_graph.replay()
            return
        self._generate_musclenn_graph_permutations()

    def _run_musclenn_minibatch(
        self,
        batch_input: torch.Tensor,
        batch_target: torch.Tensor,
        batch_context: torch.Tensor | None,
        *,
        graph_safe: bool,
        sample_weight: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """One MuscleNN optimizer step on a minibatch.

        Returns the weighted total loss, the unweighted target, regularization,
        and bound terms, the pre-clip gradient norm and whether
        clipping changed the gradient (``1.0``) or not (``0.0``).
        """
        entry_dim = int(self._musclenn_sparse_entry_dim)
        entry_torque_idx = self._musclenn_entry_torque_idx
        entry_muscle_idx = self._musclenn_entry_muscle_idx
        torque_target = batch_target

        batch_j = batch_input[:, :entry_dim]
        prepared_input = self._prepare_musclenn_model_input(batch_input)
        forward_with_preact = getattr(self._musclenn_model, "forward_with_preact", None)
        if callable(forward_with_preact):
            model_output, activation_raw = forward_with_preact(prepared_input)
        else:
            model_output = self._musclenn_model(prepared_input)
            activation_raw = model_output
        activation = model_output
        activation_for_bound_loss = activation
        if self._mass_muscle_nn().output_activation == "clamp":
            # An explicit clamp still needs the raw value for recovery losses.
            activation_for_bound_loss = activation_raw

        if self._musclenn_torque_helper.include_passive_force_in_output:
            if batch_context is None:
                raise RuntimeError(
                    "Passive-enabled MASS optimization requires raw context."
                )
            tau_pred = self._musclenn_torque_helper.torque_from_activation(
                activation,
                batch_context,
            )
            if not self._uses_mass_pose_coordinator:
                tau_pred = tau_pred * self._musclenn_torque_scale_inv.view(1, -1)
        else:
            tau_pred = torch.zeros_like(torque_target)
            if entry_dim > 0:
                muscle_activation = activation.index_select(1, entry_muscle_idx)
                tau_pred = tau_pred.scatter_add(
                    1,
                    entry_torque_idx.view(1, -1).expand(batch_input.shape[0], -1),
                    batch_j * muscle_activation,
                )

        muscle_nn = self._mass_action_settings().muscle_nn
        target_scale = muscle_nn.target_scale
        target_weight = muscle_nn.target_loss_weight
        reg_weight = muscle_nn.regularization_weight
        bound_weight = muscle_nn.bound_loss_weight
        target_error = (tau_pred - torque_target) / target_scale
        if sample_weight is None:
            target_loss = target_error.square().mean()
        else:
            target_loss = (target_error.square() * sample_weight.view(-1, 1)).sum() / (
                sample_weight.sum().clamp_min(1.0) * int(target_error.shape[1])
            )
        if (
            self._uses_mass_pose_coordinator
            and self._mass_muscle_nn().output_activation
            in RAW_ZERO_IS_REST_OUTPUT_ACTIVATIONS
        ):
            if sample_weight is None:
                reg_loss = 0.5 * (
                    activation.square().mean() + activation_raw.square().mean()
                )
            else:
                denominator = sample_weight.sum().clamp_min(1.0) * int(
                    activation.shape[1]
                )
                reg_loss = (
                    0.5
                    * (
                        (activation.square() * sample_weight.view(-1, 1)).sum()
                        + (activation_raw.square() * sample_weight.view(-1, 1)).sum()
                    )
                    / denominator
                )
        else:
            if sample_weight is None:
                reg_loss = activation.square().mean()
            else:
                reg_loss = (activation.square() * sample_weight.view(-1, 1)).sum() / (
                    sample_weight.sum().clamp_min(1.0) * int(activation.shape[1])
                )
        bound_loss = activation.new_zeros(())
        if bound_weight > 0.0:
            bound_loss = self._compute_musclenn_bound_penalty(
                activation_for_bound_loss,
                lower_tolerance=muscle_nn.bound_tolerance[0],
                upper_tolerance=muscle_nn.bound_tolerance[1],
                sample_weight=sample_weight,
            )
        loss = (
            target_weight * target_loss
            + reg_weight * reg_loss
            + bound_weight * bound_loss
        )
        self._musclenn_optimizer.zero_grad(
            set_to_none=bool(
                not graph_safe
                and not self._uses_mass_pose_coordinator
                and not bool(getattr(self, "_musclenn_use_update_graph", False))
            )
        )
        loss.backward()
        grad_clip = muscle_nn.grad_clip
        if graph_safe and not self._uses_mass_pose_coordinator:
            # foreach=None takes the multi-tensor kernels wherever they exist
            # (CUDA): one norm and one scale launch instead of one per tensor.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._musclenn_model.parameters(),
                grad_clip,
                foreach=None,
            )
            grad_clipped = self._musclenn_grad_norm_was_clipped(grad_norm, grad_clip)
        else:
            grad_norm, grad_clipped = self._clip_musclenn_grad_inplace(grad_clip)
        self._musclenn_optimizer.step()
        return (
            loss,
            target_loss,
            reg_loss,
            bound_loss,
            grad_norm,
            grad_clipped,
        )

    def _run_musclenn_graph_minibatch(
        self,
        batch_input: torch.Tensor,
        batch_target: torch.Tensor,
        batch_context: torch.Tensor,
        batch_valid_mask: torch.Tensor,
        metrics: torch.Tensor,
    ) -> None:
        step_metrics = self._run_musclenn_minibatch(
            batch_input,
            batch_target,
            batch_context if batch_context.numel() > 0 else None,
            graph_safe=True,
            sample_weight=batch_valid_mask,
        )
        for slot, value in enumerate(step_metrics):
            metrics[slot].add_(value.detach())

    def _restore_musclenn_optimizer_state_inplace(
        self,
        saved_state: dict[torch.Tensor, dict[str, object]],
    ) -> None:
        assert self._musclenn_optimizer is not None
        for param, state in self._musclenn_optimizer.state.items():
            saved_param_state = saved_state.get(param)
            if saved_param_state is None:
                # Lazy Adam allocations can already be referenced by a captured
                # graph. Keep their storage, initialized as before the first step.
                for value in state.values():
                    if torch.is_tensor(value):
                        value.zero_()
                continue
            for key in tuple(state.keys()):
                if key not in saved_param_state:
                    del state[key]
            for key, saved_value in saved_param_state.items():
                current_value = state.get(key)
                if torch.is_tensor(saved_value):
                    if (
                        torch.is_tensor(current_value)
                        and current_value.shape == saved_value.shape
                        and current_value.dtype == saved_value.dtype
                        and current_value.device == saved_value.device
                    ):
                        current_value.copy_(saved_value)
                    else:
                        state[key] = saved_value.detach().clone()
                else:
                    state[key] = copy.deepcopy(saved_value)

    @contextmanager
    def _musclenn_setup_transaction(self) -> Iterator[None]:
        """Keep successful graph storage stable and undo failed setup updates."""
        model = self._musclenn_model
        optimizer = self._musclenn_optimizer
        stream = self._musclenn_update_torch_stream
        assert model is not None and optimizer is not None and stream is not None
        parameters = tuple(model.parameters())
        values = [
            (value, value.detach().clone()) for value in (*parameters, *model.buffers())
        ]
        gradients = [
            (
                parameter.grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for parameter in parameters
        ]
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        stable_state = {
            parameter: optimizer_before["state"][index]
            for group, saved_group in zip(
                optimizer.param_groups, optimizer_before["param_groups"], strict=True
            )
            for parameter, index in zip(
                group["params"], saved_group["params"], strict=True
            )
            if index in optimizer_before["state"]
        }

        def restore_model() -> None:
            with torch.no_grad():
                for value, saved in values:
                    value.copy_(saved)

        def restore_gradients() -> None:
            with torch.no_grad():
                for parameter, (gradient, saved) in zip(
                    parameters, gradients, strict=True
                ):
                    parameter.grad = gradient
                    if gradient is not None:
                        gradient.copy_(saved)

        try:
            yield
            stream.synchronize()
            restore_model()
            self._restore_musclenn_optimizer_state_inplace(stable_state)
            optimizer.zero_grad(set_to_none=False)
        except Exception as error:
            original_error = error

            def cleanup(operation: Callable[[], None], component: str) -> None:
                try:
                    operation()
                except Exception as cleanup_error:  # noqa: BLE001 - Preserve the original setup error during best-effort cleanup.
                    original_error.add_note(
                        f"MuscleNN setup cleanup failed while {component}: {cleanup_error!r}. Restoration is best-effort."
                    )

            # A host exception can leave writes queued on the setup stream.
            # Finish them before copying the original live state back.
            cleanup(stream.synchronize, "synchronizing the setup stream")
            cleanup(restore_model, "restoring model parameters and buffers")
            cleanup(
                lambda: optimizer.load_state_dict(optimizer_before),
                "restoring optimizer state",
            )
            cleanup(restore_gradients, "restoring existing gradients")
            raise

    def _compile_and_warm_musclenn_graph_minibatch(self) -> None:
        if self._musclenn_compiled_minibatch_ready:
            return
        if self._musclenn_compile_full_update:
            self._run_musclenn_graph_minibatch_fn = torch.compile(
                self._run_musclenn_graph_minibatch,
                mode=self._musclenn_compile_mode,
                dynamic=False,
            )

        stream = self._musclenn_update_torch_stream
        contexts = self._musclenn_graph_contexts
        valid_mask = self._musclenn_graph_valid_mask
        shuffled_inputs = self._musclenn_graph_shuffled_inputs
        shuffled_targets = self._musclenn_graph_shuffled_targets
        shuffled_contexts = self._musclenn_graph_shuffled_contexts
        shuffled_valid_mask = self._musclenn_graph_shuffled_valid_mask
        permutations = self._musclenn_graph_permutations
        metrics = self._musclenn_graph_metrics
        assert stream is not None
        assert contexts is not None and valid_mask is not None
        assert shuffled_inputs is not None and shuffled_targets is not None
        assert shuffled_contexts is not None and shuffled_valid_mask is not None
        assert permutations is not None and metrics is not None
        batch_size = min(
            self._mass_action_settings().muscle_nn.batch_size,
            int(self._musclenn_graph_inputs.shape[0]),
        )
        torch.index_select(
            self._musclenn_graph_inputs,
            0,
            permutations[0],
            out=shuffled_inputs,
        )
        torch.index_select(
            self._musclenn_graph_targets,
            0,
            permutations[0],
            out=shuffled_targets,
        )
        if shuffled_contexts.numel() > 0:
            torch.index_select(
                contexts,
                0,
                permutations[0],
                out=shuffled_contexts,
            )
        torch.index_select(
            valid_mask,
            0,
            permutations[0],
            out=shuffled_valid_mask,
        )
        metrics.zero_()
        with self._musclenn_setup_transaction():
            self._run_musclenn_graph_minibatch_fn(
                shuffled_inputs[:batch_size],
                shuffled_targets[:batch_size],
                shuffled_contexts[:batch_size],
                shuffled_valid_mask[:batch_size],
                metrics,
            )
        metrics.zero_()
        stream.synchronize()
        self._musclenn_compiled_minibatch_ready = True

    def _run_musclenn_graph_epoch_loops(self) -> None:
        inputs = self._musclenn_graph_inputs
        targets = self._musclenn_graph_targets
        shuffled_inputs = self._musclenn_graph_shuffled_inputs
        shuffled_targets = self._musclenn_graph_shuffled_targets
        contexts = self._musclenn_graph_contexts
        valid_mask = self._musclenn_graph_valid_mask
        shuffled_contexts = self._musclenn_graph_shuffled_contexts
        shuffled_valid_mask = self._musclenn_graph_shuffled_valid_mask
        permutations = self._musclenn_graph_permutations
        metrics = self._musclenn_graph_metrics
        assert inputs is not None and targets is not None
        assert shuffled_inputs is not None and shuffled_targets is not None
        assert contexts is not None and valid_mask is not None
        assert shuffled_contexts is not None and shuffled_valid_mask is not None
        assert permutations is not None and metrics is not None
        capacity = int(inputs.shape[0])
        muscle_nn = self._mass_action_settings().muscle_nn
        batch_size = min(muscle_nn.batch_size, capacity)
        for epoch in range(muscle_nn.epochs):
            torch.index_select(
                inputs,
                0,
                permutations[epoch],
                out=shuffled_inputs,
            )
            torch.index_select(
                targets,
                0,
                permutations[epoch],
                out=shuffled_targets,
            )
            if contexts.numel() > 0:
                torch.index_select(
                    contexts,
                    0,
                    permutations[epoch],
                    out=shuffled_contexts,
                )
            torch.index_select(
                valid_mask,
                0,
                permutations[epoch],
                out=shuffled_valid_mask,
            )
            for start in range(0, capacity, batch_size):
                self._run_musclenn_graph_minibatch_fn(
                    shuffled_inputs[start : start + batch_size],
                    shuffled_targets[start : start + batch_size],
                    shuffled_contexts[start : start + batch_size],
                    shuffled_valid_mask[start : start + batch_size],
                    metrics,
                )

    def _capture_musclenn_update_graph(self) -> None:
        stream = self._musclenn_update_torch_stream
        assert stream is not None
        self._compile_and_warm_musclenn_graph_minibatch()
        graph = torch.cuda.CUDAGraph()
        # This graph fits sparse active-force descriptors using only Torch.
        # Physical total-force supervision runs through the eager helper path.
        with self._musclenn_setup_transaction():
            with torch.cuda.nvtx.range("capture_musclenn_update_graph"):
                with torch.cuda.graph(graph, stream=stream):
                    self._run_musclenn_graph_epoch_loops()
        assert self._musclenn_graph_metrics is not None
        self._musclenn_graph_metrics.zero_()
        stream.synchronize()
        self._musclenn_update_graph = graph
        logger.info(
            "Captured full MuscleNN update graph (mode=%s, samples=%d, epochs=%d).",
            "compiled" if self._musclenn_compile_full_update else "eager",
            int(self._musclenn_input_buffer.shape[0]),
            self._mass_action_settings().muscle_nn.epochs,
        )

    def prewarm_musclenn_update_graph(self) -> bool:
        """Reserve and capture the steady-state graph before other runtime graphs."""
        if (
            not bool(getattr(self, "_musclenn_use_update_graph", False))
            or not self._musclenn_graph_runtime_is_supported()
        ):
            return False
        self._ensure_musclenn_graph_workspace()
        stream = self._musclenn_update_torch_stream
        assert stream is not None
        current_stream = torch.cuda.current_stream(self.device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            self._stage_musclenn_graph_inputs()
            if (
                self._musclenn_use_shuffle_graph
                and self._musclenn_shuffle_graph is None
            ):
                self._capture_musclenn_shuffle_graph()
            assert self._musclenn_graph_permutations is not None
            capacity = int(self._musclenn_graph_permutations.shape[1])
            bootstrap_permutation = torch.arange(
                capacity,
                device=self.device,
                dtype=self._musclenn_graph_permutations.dtype,
            )
            self._musclenn_graph_permutations.copy_(bootstrap_permutation)
            if self._musclenn_update_graph is None:
                self._capture_musclenn_update_graph()
        stream.synchronize()
        self._musclenn_model.eval()
        return True

    def _launch_musclenn_update_graph(self) -> bool:
        """Commit the rollout and replay the captured update without waiting.

        Runs on the MuscleNN update stream, after the producer stream's work:
        the rollout commit, staging, and
        the update graph. Only the staged-row validity check reads back to
        the host. Returns ``False``, with the rollout committed, when the
        eager path must run instead.
        """
        self._ensure_musclenn_graph_workspace()
        stream = self._musclenn_update_torch_stream
        metrics = self._musclenn_graph_metrics
        start_event = self._musclenn_graph_start_event
        end_event = self._musclenn_graph_end_event
        assert stream is not None and metrics is not None
        assert start_event is not None and end_event is not None
        current_stream = torch.cuda.current_stream(self.device)
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            self._commit_musclenn_rollout_to_ring_buffer()
            eligible = self._musclenn_graph_buffer_is_eligible()
            if eligible:
                self._stage_musclenn_graph_inputs()
        if not eligible:
            current_stream.wait_stream(stream)
            return False
        stream.synchronize()
        assert self._musclenn_graph_valid_mask is not None
        if not bool(self._musclenn_graph_valid_mask.all().item()):
            # A fixed CUDA graph cannot omit an optimizer step. Replaying an
            # all-invalid minibatch would still advance Adam momentum, so use
            # the eager path, which compacts invalid rows exactly.
            return False
        with torch.cuda.stream(stream):
            self._refresh_musclenn_graph_permutations()
            if self._musclenn_update_graph is None:
                raise RuntimeError(
                    "MuscleNN update graph was not prewarmed before training."
                )
            metrics.zero_()
            with torch.cuda.nvtx.range("replay_musclenn_update_graph"):
                self._musclenn_update_graph.replay()
            end_event.record(stream)
        return True

    def _finish_musclenn_update_graph(self) -> dict[str, float]:
        """Wait for a launched update graph and read back its metrics."""
        stream = self._musclenn_update_torch_stream
        metrics = self._musclenn_graph_metrics
        start_event = self._musclenn_graph_start_event
        end_event = self._musclenn_graph_end_event
        assert stream is not None and metrics is not None
        assert start_event is not None and end_event is not None
        end_event.synchronize()
        # The next rollout reads the updated network and statistics.
        torch.cuda.current_stream(self.device).wait_stream(stream)
        metric_values = metrics.tolist()
        gpu_time_ns = float(start_event.elapsed_time(end_event) * 1.0e6)
        muscle_nn = self._mass_action_settings().muscle_nn
        total_batches = muscle_nn.epochs * (
            int(self._musclenn_input_buffer.shape[0])
            // min(
                muscle_nn.batch_size,
                int(self._musclenn_input_buffer.shape[0]),
            )
        )
        self._musclenn_model.eval()
        return {
            **self._musclenn_update_metrics(metric_values, total_batches),
            "buffer_size": float(self._musclenn_input_buffer.shape[0]),
            "gpu_time_ns": gpu_time_ns,
            "cuda_graph": 1.0,
        }

    def _musclenn_update_metrics(
        self, metric_totals: list[float], total_batches: int
    ) -> dict[str, float]:
        """Per-minibatch means of the accumulated update metrics.

        ``loss`` is the weighted total; the loss terms are unweighted.
        """
        inv_batches = 1.0 / float(total_batches)
        (
            avg_loss,
            avg_target_loss,
            avg_reg_loss,
            avg_bound_loss,
            avg_grad_norm,
            grad_clip_fraction,
        ) = (float(value) * inv_batches for value in metric_totals)
        self._musclenn_last_loss = avg_loss
        self._musclenn_last_reg_loss = avg_reg_loss
        self._musclenn_last_target_loss = avg_target_loss
        return {
            "loss": avg_loss,
            "target_loss": avg_target_loss,
            "reg_loss": avg_reg_loss,
            "bound_loss": avg_bound_loss,
            "grad_norm": avg_grad_norm,
            "grad_clip_fraction": grad_clip_fraction,
            **self._musclenn_input_block_metrics(),
        }

    def _musclenn_update_is_configured(self) -> bool:
        return bool(
            self._uses_mass_coordinator
            and self._musclenn_model is not None
            and self._musclenn_optimizer is not None
            and self._musclenn_torque_helper is not None
            and not self._mass_muscle_nn().frozen
        )

    def _validate_musclenn_update_settings(self) -> None:
        muscle_nn = self._mass_action_settings().muscle_nn
        if muscle_nn.target_scale <= 0.0:
            raise ValueError("musclenn_target_scale must be positive.")
        if muscle_nn.target_loss_weight <= 0.0:
            raise ValueError("musclenn_target_loss_weight must be positive.")

    def _require_no_pending_musclenn_update(self) -> None:
        if getattr(self, "_musclenn_update_pending", False):
            raise RuntimeError(
                "A MuscleNN update was begun but never finished: call "
                "optimize_muscle_nn() before the next rollout"
            )

    def begin_muscle_nn_update(self) -> bool:
        """Start the captured MuscleNN update without waiting for it.

        The update shares no data with the policy update, so the trainer
        begins it first and runs the policy update while it executes on its
        own stream. ``optimize_muscle_nn`` then waits for it and returns its
        metrics; it must be called before the next rollout. Returns whether
        an update is now in flight (only the captured update runs
        asynchronously; otherwise ``optimize_muscle_nn`` runs it).
        """
        self._require_no_pending_musclenn_update()
        if not self._musclenn_update_is_configured() or not bool(
            getattr(self, "_musclenn_use_update_graph", False)
        ):
            return False
        self._validate_musclenn_update_settings()
        self._musclenn_model.train()
        self._musclenn_update_pending = self._launch_musclenn_update_graph()
        return self._musclenn_update_pending

    def optimize_muscle_nn(self) -> dict[str, float] | None:
        """Run, or finish a begun, MuscleNN update and return its metrics."""
        if getattr(self, "_musclenn_update_pending", False):
            self._musclenn_update_pending = False
            return self._finish_musclenn_update_graph()
        if not self._musclenn_update_is_configured():
            return None
        self._validate_musclenn_update_settings()
        muscle_nn = self._mass_action_settings().muscle_nn
        self._musclenn_model.train()
        if (
            bool(getattr(self, "_musclenn_use_update_graph", False))
            and self._launch_musclenn_update_graph()
        ):
            return self._finish_musclenn_update_graph()

        buffer_view = self._get_musclenn_buffer_view()
        if buffer_view is None:
            return None
        inputs_all, target_all, context_all = buffer_view
        inputs_all, target_all, context_all = self._filter_finite_musclenn_samples(
            inputs_all,
            target_all,
            context_all,
        )
        if inputs_all.shape[0] == 0:
            return None

        batch_size = min(muscle_nn.batch_size, int(inputs_all.shape[0]))
        epochs = muscle_nn.epochs
        metric_totals = torch.zeros(
            _MUSCLENN_METRIC_COUNT, device=self.device, dtype=torch.float32
        )
        total_batches = 0

        eager_update_stream = (
            self._musclenn_update_torch_stream
            if bool(getattr(self, "_musclenn_use_update_graph", False))
            else None
        )
        if eager_update_stream is not None:
            eager_update_stream.wait_stream(torch.cuda.current_stream(self.device))
        eager_stream_context = (
            torch.cuda.stream(eager_update_stream)
            if eager_update_stream is not None
            else nullcontext()
        )
        timing_events = (
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            if self.device.type == "cuda"
            else None
        )
        with eager_stream_context, self._musclenn_warp_stream_scope():
            if timing_events is not None:
                timing_events[0].record()
            for _ in range(epochs):
                perm = torch.randperm(inputs_all.shape[0], device=self.device)
                for start in range(0, int(inputs_all.shape[0]), batch_size):
                    batch_idx = perm[start : start + batch_size]
                    batch_input = inputs_all.index_select(0, batch_idx)
                    batch_target = target_all.index_select(0, batch_idx)
                    batch_context = (
                        None
                        if context_all is None
                        else context_all.index_select(0, batch_idx)
                    )
                    step_metrics = self._run_musclenn_minibatch(
                        batch_input,
                        batch_target,
                        batch_context,
                        graph_safe=False,
                    )
                    for slot, value in enumerate(step_metrics):
                        metric_totals[slot].add_(value.detach())
                    total_batches += 1
            if timing_events is not None:
                timing_events[1].record()
        if eager_update_stream is not None:
            eager_update_stream.synchronize()
        self._musclenn_model.eval()
        if total_batches <= 0:
            return None
        timing: dict[str, float] = {}
        if timing_events is not None:
            timing_events[1].synchronize()
            timing["gpu_time_ns"] = float(
                timing_events[0].elapsed_time(timing_events[1]) * 1.0e6
            )
        return {
            **self._musclenn_update_metrics(metric_totals.tolist(), total_batches),
            "buffer_size": float(inputs_all.shape[0]),
            **timing,
        }
