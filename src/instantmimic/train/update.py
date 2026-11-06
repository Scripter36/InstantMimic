from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter_ns
from typing import Any

import torch

from instantmimic.rl.runtime_setup import runtime_setup_time_ns


def _build_musclenn_only_loss_stub(agent: Any) -> dict[str, float]:
    optimizer = getattr(agent, "optimizer", None)
    learning_rate = 0.0
    if optimizer is not None:
        param_groups = getattr(optimizer, "param_groups", None) or []
        if param_groups:
            group_lr = param_groups[0].get("lr", 0.0)
            learning_rate = (
                float(group_lr.detach().item())
                if torch.is_tensor(group_lr)
                else float(group_lr)
            )
    std = 0.0
    policy_std_mean = getattr(agent, "policy_std_mean", None)
    if callable(policy_std_mean):
        std_value = policy_std_mean()
        std = (
            float(std_value.detach().item())
            if torch.is_tensor(std_value)
            else float(std_value.item() if hasattr(std_value, "item") else std_value)
        )
    return {
        "policy": 0.0,
        "value": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "clip_fraction": 0.0,
        "lr": learning_rate,
        "std": std,
    }


def _configure_musclenn_update_runtime(
    *,
    agent: Any,
    env_raw_for_agent: Any,
) -> Any | None:
    fast_muscle_controller = getattr(env_raw_for_agent, "fast_muscle", None)
    configure = getattr(
        fast_muscle_controller,
        "configure_musclenn_update_runtime",
        None,
    )
    if not callable(configure):
        return fast_muscle_controller
    configure(
        update_fullgraph_compile=bool(
            getattr(agent, "_compile_update_fullgraph", False)
        ),
        compile_mode=str(getattr(agent, "_compile_mode", "max-autotune-no-cudagraphs")),
        use_update_graph=bool(getattr(agent, "_use_update_graph", False)),
        use_shuffle_graph=bool(getattr(agent, "_use_shuffle_graph", False)),
    )
    return fast_muscle_controller


def _maybe_update_adaptive_sampling_after_rollout(
    env_raw_for_agent: Any,
    *,
    wp_stream: Any | None = None,
    torch_stream: torch.cuda.Stream | None = None,
    device: torch.device | None = None,
) -> None:
    update_adaptive_sampling_distribution = getattr(
        env_raw_for_agent,
        "update_adaptive_sampling_distribution_after_rollout",
        None,
    )
    if callable(update_adaptive_sampling_distribution):
        nvtx_context = (
            torch.cuda.nvtx.range("adaptive_sampling_update_after_rollout")
            if torch.cuda.is_available()
            else nullcontext()
        )
        with nvtx_context:
            update_adaptive_sampling_distribution(
                wp_stream=wp_stream,
                torch_stream=torch_stream,
            )
        if torch_stream is not None and device is not None and device.type == "cuda":
            torch.cuda.current_stream(device).wait_stream(torch_stream)


def _maybe_update_motion_plr_after_policy_update(
    env_raw_for_agent: Any,
    *,
    agent: Any,
    level_ids: torch.Tensor | None,
    wp_stream: Any | None = None,
    torch_stream: torch.cuda.Stream | None = None,
    device: torch.device | None = None,
) -> None:
    requires_policy_scores = getattr(
        env_raw_for_agent,
        "adaptive_sampling_requires_policy_scores",
        None,
    )
    if not callable(requires_policy_scores) or not bool(requires_policy_scores()):
        return
    if level_ids is None:
        raise RuntimeError("motion PLR rollout level IDs were not collected")
    buffer = getattr(agent, "buffer", None)
    # PPO has computed GAE by this point. Under PopArt, returns and values share
    # the same positive scale, so abs(returns - values) preserves PLR score ranks.
    returns = getattr(buffer, "returns", None)
    values = getattr(buffer, "values", None)
    if not torch.is_tensor(returns) or not torch.is_tensor(values):
        raise RuntimeError("motion PLR requires a PPO buffer with returns and values")
    update = getattr(
        env_raw_for_agent,
        "update_motion_plr_distribution_after_policy_update",
        None,
    )
    if not callable(update):
        raise RuntimeError("motion PLR environment update hook is unavailable")
    update(
        level_ids=level_ids,
        returns=returns,
        values=values,
        wp_stream=wp_stream,
        torch_stream=torch_stream,
    )
    if torch_stream is not None and device is not None and device.type == "cuda":
        torch.cuda.current_stream(device).wait_stream(torch_stream)


def _run_training_update_stage(
    *,
    training_stage: str,
    checkpoint_manager: Any,
    agent: Any,
    env_raw_for_agent: Any,
    rollout_obs: torch.Tensor | Any,
    update_count: int,
    checkpoint_controller: Any | None = None,
) -> tuple[dict[str, float], int, int]:
    if checkpoint_controller is None:
        checkpoint_manager.wait_for_update_boundary()
    else:
        checkpoint_controller.wait_for_update_boundary()
    fast_muscle_controller = _configure_musclenn_update_runtime(
        agent=agent,
        env_raw_for_agent=env_raw_for_agent,
    )
    optimize_muscle_nn = getattr(env_raw_for_agent, "optimize_muscle_nn", None)
    begin_muscle_nn_update = getattr(
        fast_muscle_controller, "begin_muscle_nn_update", None
    )
    if callable(optimize_muscle_nn) and callable(begin_muscle_nn_update):
        # The MuscleNN update shares no data with the policy update: launch
        # it on its own stream first so the two run concurrently, then wait
        # for it below, before the next rollout.
        launch_nvtx_context = (
            torch.cuda.nvtx.range("musclenn_update_launch")
            if torch.cuda.is_available()
            else nullcontext()
        )
        with launch_nvtx_context:
            begin_muscle_nn_update()

    update_setup_start_ns = runtime_setup_time_ns(agent)
    update_t0_ns = perf_counter_ns()
    if training_stage == "musclenn_only":
        last_losses: dict[str, float] = _build_musclenn_only_loss_stub(agent)
    else:
        last_losses = dict(agent.update())
        update_initial_state = getattr(agent, "update_initial_state", None)
        if callable(update_initial_state):
            initial_state_metrics = update_initial_state()
            if initial_state_metrics:
                last_losses.update(initial_state_metrics)
    # Policy update time only: the MuscleNN update reports its own time.
    policy_update_elapsed_ns = max(
        0,
        perf_counter_ns()
        - update_t0_ns
        - (runtime_setup_time_ns(agent) - update_setup_start_ns),
    )

    if callable(optimize_muscle_nn):
        musclenn_nvtx_context = (
            torch.cuda.nvtx.range("musclenn_update_host")
            if torch.cuda.is_available()
            else nullcontext()
        )
        with musclenn_nvtx_context:
            musclenn_stats = optimize_muscle_nn()
        if musclenn_stats:
            last_losses = dict(last_losses)
            last_losses["musclenn_loss"] = float(musclenn_stats.get("loss", 0.0))
            last_losses["musclenn_target_loss"] = float(
                musclenn_stats.get("target_loss", 0.0)
            )
            last_losses["musclenn_reg_loss"] = float(
                musclenn_stats.get("reg_loss", 0.0)
            )
            last_losses["musclenn_bound_loss"] = float(
                musclenn_stats.get("bound_loss", 0.0)
            )
            last_losses["musclenn_buffer_size"] = float(
                musclenn_stats.get("buffer_size", 0.0)
            )
            # Pre-clip gradient norm, clipped-step fraction and the per-block
            # raw input statistics (`input_<block>_mean_abs` / `_std`).
            for key, value in musclenn_stats.items():
                if key in ("grad_norm", "grad_clip_fraction") or key.startswith(
                    "input_"
                ):
                    last_losses[f"musclenn_{key}"] = float(value)
            last_losses["musclenn_gpu_time_ns"] = float(
                musclenn_stats.get("gpu_time_ns", 0.0)
            )
            last_losses["musclenn_cuda_graph"] = float(
                musclenn_stats.get("cuda_graph", 0.0)
            )

    update_gpu_time_ns = last_losses.get("update_gpu_time_ns")
    update_elapsed_ns = (
        max(0, int(update_gpu_time_ns))
        if update_gpu_time_ns is not None
        else policy_update_elapsed_ns
    )
    return last_losses, update_count + 1, update_elapsed_ns


def _run_training_update_cycle(
    *,
    training_stage: str,
    checkpoint_manager: Any,
    agent: Any,
    env_raw_for_agent: Any,
    rollout_obs: torch.Tensor | Any,
    update_count: int,
    checkpoint_controller: Any | None = None,
) -> tuple[dict[str, float], int, int]:
    return _run_training_update_stage(
        training_stage=training_stage,
        checkpoint_manager=checkpoint_manager,
        agent=agent,
        env_raw_for_agent=env_raw_for_agent,
        rollout_obs=rollout_obs,
        update_count=update_count,
        checkpoint_controller=checkpoint_controller,
    )
