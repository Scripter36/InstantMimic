from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass

import mujoco_warp as mjw
import torch
import warp as wp

from instantmimic.envs.config_access import get_block, get_value
from instantmimic.envs.direct_rl_env import DirectRLEnv
from instantmimic.envs.snu.tasks.snu_imitation import (
    adaptive_sampling_runtime,
    initial_state_runtime,
)
from instantmimic.envs.snu.tasks.snu_imitation.adaptive_sampling import (
    RESET_SETTING_ADAPTIVE_CHUNK_SIZE,
    RESET_SETTING_ADAPTIVE_ENABLED,
    RESET_SETTING_ADAPTIVE_UNIT_COUNT,
    RESET_SETTING_COUNT,
    RESET_SETTING_FIXED_MOTION_IDX,
    RESET_SETTING_PRE_FAILURE_WINDOW_STEPS,
    RESET_SETTING_RANDOM_INIT,
    RESET_SETTING_START_FRAME_IDX,
    RESET_SETTING_START_HORIZON_STEPS,
    RESET_SETTING_TOTAL_MOTION_FRAME_COUNT,
    adaptive_accumulate_reset_metrics_kernel,
    reset_from_motion_multi_select_kernel,
    reset_from_motion_select_kernel,
    reset_from_motion_state_kernel,
)
from instantmimic.utils.warp_kernels import copy_1d_int64_kernel

_LOGGER = logging.getLogger(__name__)
__all__ = (
    "ResetRuntimeState",
    "ResetSettings",
    "resolve_reset_settings",
    "initialize_episode_state",
    "initialize_reset_behavior",
    "initialize_reset_sampling_settings",
    "set_analysis_reset_overrides",
    "clear_analysis_reset_overrides",
    "reset_idx",
)


@dataclass(frozen=True)
class ResetSettings:
    deterministic_reset: bool
    random_init_enabled: bool
    motion_rand_std: float
    fixed_motion_idx: int | None
    start_frame_idx: int | None


@dataclass(frozen=True)
class ResetRuntimeState:
    """Persistent buffers and constants used by reset orchestration."""

    motion_frame_idx: torch.Tensor
    motion_max_width: int
    seed: torch.Tensor
    seed_wp: wp.array
    analysis_override_enabled: torch.Tensor
    analysis_override_clip_idx: torch.Tensor
    analysis_override_start_frame_idx: torch.Tensor
    sampling_settings: torch.Tensor
    sampling_settings_wp: wp.array
    spawn_offset_vec3: wp.vec3


@wp.kernel(enable_backward=False)
def reset_passive_coordinates_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    passive_qpos_adr: wp.array(dtype=wp.int32),
    passive_qpos0: wp.array(dtype=float),
    passive_dof_adr: wp.array(dtype=wp.int32),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
):
    """Put a reset world's passive coordinates at qpos0, at rest.

    A passive joint inside the character (instantmimic.envs.dof_roles) is
    relative to its parent body, so qpos0 is its rest pose on the reset
    character. A free passive body's qpos0 is its compile-time world pose, not
    a pose relative to the character: a task with such bodies places them in a
    hook after this reset.
    """
    env_idx, elem_idx = wp.tid()
    if not reset_mask[env_idx]:
        return
    if elem_idx < passive_qpos_adr.shape[0]:
        qpos[env_idx, passive_qpos_adr[elem_idx]] = passive_qpos0[elem_idx]
    if elem_idx < passive_dof_adr.shape[0]:
        qvel[env_idx, passive_dof_adr[elem_idx]] = 0.0


def reset_passive_coordinates(env, reset_mask: torch.Tensor) -> None:
    """Rest the passive coordinates of the reset worlds (no-op without any)."""
    passive_qpos_adr = getattr(env, "passive_qpos_adr_i32", None)
    count = 0 if passive_qpos_adr is None else int(passive_qpos_adr.numel())
    if count <= 0:
        return
    wp.launch(
        reset_passive_coordinates_kernel,
        dim=(env.num_envs, count),
        inputs=[
            reset_mask,
            env.passive_qpos_adr_i32,
            env.passive_qpos0,
            env.passive_dof_adr_i32,
        ],
        outputs=[env.mjw_data.qpos, env.mjw_data.qvel],
    )


def resolve_reset_settings(cfg) -> ResetSettings:
    reset = get_block(cfg, "reset")
    deterministic_reset = bool(get_value(reset, "deterministic", False))
    random_init_enabled = bool(get_value(reset, "random_start", True))
    motion_rand_std = float(get_value(reset, "motion_noise_std", 0.0))
    if deterministic_reset:
        random_init_enabled = False
        motion_rand_std = 0.0
    fixed_motion_idx_cfg = get_value(reset, "motion_index", None)
    start_frame_idx_cfg = get_value(reset, "start_frame", None)
    return ResetSettings(
        deterministic_reset=deterministic_reset,
        random_init_enabled=random_init_enabled,
        motion_rand_std=motion_rand_std,
        fixed_motion_idx=(
            None if fixed_motion_idx_cfg is None else int(fixed_motion_idx_cfg)
        ),
        start_frame_idx=(
            None if start_frame_idx_cfg is None else int(start_frame_idx_cfg)
        ),
    )


def _startup_reset_phase(env, name: str):
    profiler = getattr(env, "_startup_reset_profiler", None)
    if profiler is None:
        return nullcontext()
    return profiler.phase(name)


def initialize_episode_state(env) -> None:
    env.ref_frame_idx = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    env.motion_clip_idx = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.int32
    )
    env.bootstrap_time_outs = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.bool
    )
    env.motion_end_time_outs = torch.zeros_like(env.bootstrap_time_outs)
    motion_frame_idx = torch.full(
        (env.num_envs,), -1, device=env.device, dtype=torch.int32
    )
    motion_max_width = max(int(env.model.nq), int(env.model.nv), int(env.model.nu))
    seed = torch.tensor(
        [0 if env.cfg.seed is None else int(env.cfg.seed)],
        device=env.device,
        dtype=torch.int32,
    )
    seed_wp = wp.from_torch(seed, dtype=wp.int32)
    analysis_override_enabled = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.bool
    )
    analysis_override_clip_idx = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.int32
    )
    analysis_override_start_frame_idx = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.int32
    )
    # The packed settings are allocated with the other reset buffers because
    # their shape is fixed at construction; adaptive initialization fills the
    # contents later without replacing this tensor or its Warp view.
    sampling_settings = torch.zeros(
        RESET_SETTING_COUNT, device=env.device, dtype=torch.int32
    )
    sampling_settings_wp = wp.from_torch(sampling_settings, dtype=wp.int32)
    env.early_termination_timer = torch.zeros(
        env.num_envs, device=env.device, dtype=torch.int32
    )
    env.motion_offset = torch.tensor(
        env.motion_settings.offset, device=env.device, dtype=torch.float32
    )
    env.spawn_offset = torch.tensor(
        env.cfg.spawn_offset, device=env.device, dtype=torch.float32
    )
    env.reset_runtime_state = ResetRuntimeState(
        motion_frame_idx=motion_frame_idx,
        motion_max_width=motion_max_width,
        seed=seed,
        seed_wp=seed_wp,
        analysis_override_enabled=analysis_override_enabled,
        analysis_override_clip_idx=analysis_override_clip_idx,
        analysis_override_start_frame_idx=analysis_override_start_frame_idx,
        sampling_settings=sampling_settings,
        sampling_settings_wp=sampling_settings_wp,
        spawn_offset_vec3=wp.vec3(
            env.spawn_offset[0], env.spawn_offset[1], env.spawn_offset[2]
        ),
    )


def initialize_reset_sampling_settings(env) -> None:
    """Stage the nine run-constant reset-sampling settings on the device.

    `resolve_reset_settings` fixes the `reset.*` half at construction and
    `adaptive_sampling_runtime.initialize` fixes the adaptive half, so the
    selection kernel reads one array instead of taking nine launch scalars —
    and nine Python `int()` conversions — on every reset. Call this after
    adaptive sampling is initialized.
    """
    reset_settings = env.reset_settings
    adaptive = env.adaptive_sampling_state
    values = [0] * RESET_SETTING_COUNT
    values[RESET_SETTING_RANDOM_INIT] = int(reset_settings.random_init_enabled)
    values[RESET_SETTING_START_FRAME_IDX] = (
        -1
        if reset_settings.start_frame_idx is None
        else int(reset_settings.start_frame_idx)
    )
    values[RESET_SETTING_ADAPTIVE_ENABLED] = int(adaptive.sampling_enabled)
    values[RESET_SETTING_ADAPTIVE_CHUNK_SIZE] = int(adaptive.chunk_size)
    values[RESET_SETTING_ADAPTIVE_UNIT_COUNT] = int(adaptive.unit_count)
    values[RESET_SETTING_PRE_FAILURE_WINDOW_STEPS] = int(
        adaptive.sonic_pre_failure_window_steps
    )
    values[RESET_SETTING_TOTAL_MOTION_FRAME_COUNT] = int(env.motion_frame_count)
    values[RESET_SETTING_START_HORIZON_STEPS] = int(env._reset_start_horizon_steps)
    values[RESET_SETTING_FIXED_MOTION_IDX] = (
        -1
        if reset_settings.fixed_motion_idx is None
        else int(reset_settings.fixed_motion_idx)
    )

    env.reset_runtime_state.sampling_settings.copy_(
        torch.as_tensor(values, device=env.device, dtype=torch.int32)
    )


def set_analysis_reset_overrides(
    env,
    *,
    clip_indices: torch.Tensor,
    start_frame_indices: torch.Tensor,
    enabled: torch.Tensor | None = None,
) -> None:
    clip_indices_t = torch.as_tensor(
        clip_indices, device=env.device, dtype=torch.int32
    ).reshape(-1)
    start_frame_indices_t = torch.as_tensor(
        start_frame_indices, device=env.device, dtype=torch.int32
    ).reshape(-1)
    enabled_t = (
        torch.as_tensor(enabled, device=env.device, dtype=torch.bool).reshape(-1)
        if enabled is not None
        else torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    )

    if clip_indices_t.shape[0] != env.num_envs:
        raise ValueError(
            f"clip_indices must have length {env.num_envs}, got {clip_indices_t.shape[0]}."
        )
    if start_frame_indices_t.shape[0] != env.num_envs:
        raise ValueError(
            "start_frame_indices must have length "
            f"{env.num_envs}, got {start_frame_indices_t.shape[0]}."
        )
    if enabled_t.shape[0] != env.num_envs:
        raise ValueError(
            f"enabled must have length {env.num_envs}, got {enabled_t.shape[0]}."
        )

    if bool(enabled_t.any().item()):
        active_clip_idx = clip_indices_t[enabled_t]
        active_start_idx = start_frame_indices_t[enabled_t]
        if int(active_clip_idx.min().item()) < 0:
            raise ValueError("analysis reset clip indices must be >= 0.")
        if int(active_start_idx.min().item()) < 0:
            raise ValueError("analysis reset start frame indices must be >= 0.")
        if env._multi_motion_enabled:
            if int(active_clip_idx.max().item()) >= int(env.motion_clip_count):
                raise ValueError(
                    "analysis reset clip index is out of range for loaded motion clips."
                )
            active_lengths = env.motion_clip_length.to(torch.long).index_select(
                0, active_clip_idx.to(torch.long)
            )
            if bool((active_start_idx.to(torch.long) >= active_lengths).any().item()):
                raise ValueError(
                    "analysis reset start frame index is out of range for its clip."
                )
        else:
            if bool((active_clip_idx != 0).any().item()):
                raise ValueError(
                    "analysis reset clip index must be 0 when multi-motion is disabled."
                )
            if bool(
                (active_start_idx.to(torch.long) >= int(env.motion_frame_count))
                .any()
                .item()
            ):
                raise ValueError(
                    "analysis reset start frame index is out of range for the motion."
                )

    env.reset_runtime_state.analysis_override_enabled.copy_(enabled_t)
    env.reset_runtime_state.analysis_override_clip_idx.copy_(clip_indices_t)
    env.reset_runtime_state.analysis_override_start_frame_idx.copy_(
        start_frame_indices_t
    )


def clear_analysis_reset_overrides(env) -> None:
    env.reset_runtime_state.analysis_override_enabled.zero_()
    env.reset_runtime_state.analysis_override_clip_idx.zero_()
    env.reset_runtime_state.analysis_override_start_frame_idx.zero_()


def _initialize_reset_motion(env) -> None:
    """Keep an optional frame-aligned reset reference separate from tracking."""
    env.reset_motion_qpos = env.motion_qpos
    env.reset_motion_qvel = env.motion_qvel
    source = get_value(get_block(env.cfg, "reset"), "motion_source", None)
    if source is None:
        return
    if not isinstance(source, str) or not source.strip():
        raise ValueError(
            "reset.motion_source must be a nonempty single motion filename."
        )
    initial_state = get_block(env.cfg, "initial_state", required=False)
    if env._multi_motion_enabled or bool(get_value(initial_state, "enabled", False)):
        raise ValueError(
            "reset.motion_source requires a single tracking clip and initial_state.enabled=false."
        )
    motion = env._load_single_motion_reference_data(source)
    if int(motion["frame_count"]) != int(env.motion_frame_count):
        raise ValueError(
            "reset.motion_source must match the tracking reference frame count."
        )
    for key, reference in (("qpos", env.motion_qpos), ("qvel", env.motion_qvel)):
        value = motion[key]
        if (
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or value.device != reference.device
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                f"reset.motion_source has incompatible or nonfinite {key}."
            )
        if not torch.equal(motion[f"{key}_mask"], getattr(env, f"motion_{key}_mask")):
            raise ValueError(f"reset.motion_source must match the tracking {key} mask.")
    env.reset_motion_qpos = motion["qpos"]
    env.reset_motion_qvel = motion["qvel"]
    _LOGGER.info(
        "Using a separate reset motion reference (%d frames).", env.motion_frame_count
    )


def initialize_reset_behavior(env) -> None:
    _initialize_reset_motion(env)
    settings = env.reset_settings
    env.reset_settings = settings
    reset = get_block(env.cfg, "reset")
    raw_random_init_enabled = bool(get_value(reset, "random_start", True))
    raw_motion_rand_std = float(get_value(reset, "motion_noise_std", 0.0))
    if settings.fixed_motion_idx is not None:
        if settings.fixed_motion_idx < 0:
            raise ValueError("reset.motion_index must be >= 0 when provided.")
        if env._multi_motion_enabled:
            if settings.fixed_motion_idx >= int(env.motion_clip_count):
                raise ValueError(
                    f"reset.motion_index={settings.fixed_motion_idx} is out of range for "
                    f"{env.motion_clip_count} motion clips."
                )
        elif settings.fixed_motion_idx != 0:
            raise ValueError(
                "reset.motion_index is only valid for multi-motion configs or equal to 0."
            )
    if settings.start_frame_idx is not None:
        if settings.start_frame_idx < 0:
            raise ValueError("reset.start_frame must be >= 0 when provided.")
        if env._multi_motion_enabled:
            if settings.fixed_motion_idx is None:
                raise ValueError(
                    "reset.start_frame requires reset.motion_index when using multiple motion clips."
                )
            clip_length = int(env.motion_clip_length[settings.fixed_motion_idx].item())
            if settings.start_frame_idx >= clip_length:
                raise ValueError(
                    f"reset.start_frame={settings.start_frame_idx} is out of range for "
                    f"clip {settings.fixed_motion_idx} with length {clip_length}."
                )
        elif settings.start_frame_idx >= int(env.motion_frame_count):
            raise ValueError(
                f"reset.start_frame={settings.start_frame_idx} is out of range for "
                f"motion length {env.motion_frame_count}."
            )
    if not settings.deterministic_reset:
        return

    if raw_random_init_enabled:
        _LOGGER.info(
            "reset.deterministic: overriding reset.random_start=true -> false "
            "for deterministic playback/eval."
        )
    if raw_motion_rand_std != 0.0:
        _LOGGER.info(
            "reset.deterministic: overriding reset.motion_noise_std=%.6f -> 0.0 "
            "for deterministic playback/eval.",
            raw_motion_rand_std,
        )


def reset_idx(env, reset_mask: torch.Tensor) -> None:
    with _startup_reset_phase(env, "snu_reset_idx_body"):
        _reset_idx_body(env, reset_mask=reset_mask)
    control_pipeline = getattr(env, "_control_pipeline", None)
    if control_pipeline is not None and control_pipeline.is_enabled:
        with _startup_reset_phase(env, "snu_control_reset_idx"):
            control_pipeline.reset_idx(reset_mask)


def _reset_idx_body(env, *, reset_mask: torch.Tensor) -> None:
    reset_settings = env.reset_settings
    if env.adaptive_sampling_state.sampling_uses_episode_metrics:
        with _startup_reset_phase(env, "snu_adaptive_pre_reset_state"):
            wp.launch(
                copy_1d_int64_kernel,
                dim=(env.num_envs,),
                inputs=[
                    env.episode_length_buf,
                    env.adaptive_sampling_state.episode_length_before_reset,
                ],
            )
            adaptive_sampling_runtime.capture_terminal_state_before_reset(env)

    with _startup_reset_phase(env, "direct_episode_length_reset"):
        DirectRLEnv._reset_idx(env, reset_mask)
    with _startup_reset_phase(env, "mjw_reset_data"):
        mjw.reset_data(env.mjw_model, env.mjw_data, wp.from_torch(reset_mask))

    if env.adaptive_sampling_state.sampling_uses_episode_metrics:
        with _startup_reset_phase(env, "snu_adaptive_accumulate_reset_metrics"):
            wp.launch(
                adaptive_accumulate_reset_metrics_kernel,
                dim=(env.num_envs,),
                inputs=[
                    reset_mask,
                    env.reset_terminated,
                    env.reset_time_outs,
                    env.adaptive_sampling_state.episode_length_before_reset,
                    int(env.adaptive_sampling_state.metric_type),
                    int(env.max_episode_length),
                    env.adaptive_sampling_state.episode_reward_sum,
                    int(env.adaptive_sampling_state.blame_mode),
                    int(env.adaptive_sampling_state.terminal_window_chunk_count),
                    env.adaptive_sampling_state.episode_start_unit_idx,
                    env.adaptive_sampling_state.episode_terminal_unit_idx,
                    env.adaptive_sampling_state.episode_terminal_clip_unit_start,
                    env.adaptive_sampling_state.episode_terminal_clip_unit_count,
                    env.adaptive_sampling_state.metric_sum,
                    env.adaptive_sampling_state.metric_count,
                    env.adaptive_sampling_state.metric_event_count,
                ],
            )

    if env._multi_motion_enabled:
        with _startup_reset_phase(env, "reset_from_motion_multi_select"):
            wp.launch(
                reset_from_motion_multi_select_kernel,
                dim=(env.num_envs,),
                inputs=[
                    reset_mask,
                    env.reset_runtime_state.seed_wp,
                    env.common_step_counter,
                    env.reset_runtime_state.sampling_settings_wp,
                    float(env.adaptive_sampling_state.random_sample_prob),
                    env.adaptive_sampling_state.cdf,
                    env.reset_runtime_state.analysis_override_enabled,
                    env.reset_runtime_state.analysis_override_clip_idx,
                    env.reset_runtime_state.analysis_override_start_frame_idx,
                    env.motion_clip_start,
                    env.motion_clip_length,
                    env.motion_frame_to_clip_idx,
                    env.motion_clip_unit_start,
                    env.motion_clip_unit_count,
                ],
                outputs=[
                    env.motion_clip_idx,
                    env.ref_frame_idx,
                    env.adaptive_sampling_state.episode_start_unit_idx,
                    env.early_termination_timer,
                    env.reset_runtime_state.motion_frame_idx,
                ],
            )
    else:
        with _startup_reset_phase(env, "reset_from_motion_select"):
            wp.launch(
                reset_from_motion_select_kernel,
                dim=(env.num_envs,),
                inputs=[
                    reset_mask,
                    env.reset_runtime_state.seed_wp,
                    env.common_step_counter,
                    int(reset_settings.random_init_enabled),
                    int(
                        -1
                        if reset_settings.start_frame_idx is None
                        else reset_settings.start_frame_idx
                    ),
                    int(env.adaptive_sampling_state.sampling_enabled),
                    float(env.adaptive_sampling_state.random_sample_prob),
                    int(env.adaptive_sampling_state.chunk_size),
                    int(env.adaptive_sampling_state.unit_count),
                    env.adaptive_sampling_state.cdf,
                    int(env.adaptive_sampling_state.sonic_pre_failure_window_steps),
                    env.motion_frame_count,
                    int(env._reset_start_horizon_steps),
                    env.reset_runtime_state.analysis_override_enabled,
                    env.reset_runtime_state.analysis_override_start_frame_idx,
                ],
                outputs=[
                    env.ref_frame_idx,
                    env.adaptive_sampling_state.episode_start_unit_idx,
                    env.early_termination_timer,
                    env.reset_runtime_state.motion_frame_idx,
                ],
            )

    with _startup_reset_phase(env, "reset_from_motion_state"):
        wp.launch(
            reset_from_motion_state_kernel,
            dim=(env.num_envs, int(env.reset_runtime_state.motion_max_width)),
            inputs=[
                reset_mask,
                env.reset_runtime_state.seed_wp,
                env.common_step_counter,
                env.reset_runtime_state.motion_frame_idx,
                getattr(env, "reset_motion_qpos", env.motion_qpos),
                getattr(env, "reset_motion_qvel", env.motion_qvel),
                float(reset_settings.motion_rand_std),
                env.reset_runtime_state.spawn_offset_vec3,
                env.model.nq,
                env.model.nv,
                env.model.nu,
                int(getattr(env, "root_qpos_count", 7)),
                int(getattr(env, "root_dof_count", 6)),
            ],
            outputs=[env.actions, env.mjw_data.qpos, env.mjw_data.qvel],
        )
    with _startup_reset_phase(env, "learned_initial_state"):
        initial_state_runtime.apply_after_reference_reset(env, reset_mask)
    with _startup_reset_phase(env, "passive_coordinates"):
        reset_passive_coordinates(env, reset_mask)
