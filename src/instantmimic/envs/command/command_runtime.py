from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from pymotion.rotations import quat_torch

from instantmimic.envs.command.command_sampling import (
    COMMAND_MODE_POINT_GOAL,
    COMMAND_MODE_VELOCITY_HEADING,
    sample_commands_kernel,
    sample_point_goals_with_world_radius_limit_torch,
    update_velocity_command_resample_mask_kernel,
)
from instantmimic.envs.command.rewards import (
    VELOCITY_REWARD_MODE_FORWARD_SPEED_HEADING,
    compute_point_goal_reward_terms,
    compute_velocity_heading_reward_terms,
    resolve_velocity_heading_reward_mode,
)
from instantmimic.envs.eval_metrics import EvalEpisodeSpec, EvalMetricSpec
from instantmimic.envs.tracking.metrics import get_heading_frame

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class CommandState:
    """Persistent buffers and scheduler state owned by the command runtime."""

    eval_metrics_enabled: bool = False
    reward_term_names: tuple[str, ...] = ()
    reward_terms: torch.Tensor | None = None
    goal_reached: torch.Tensor | None = None
    velocity_error: torch.Tensor | None = None
    heading_alignment: torch.Tensor | None = None
    velocity_resample_min_steps: int = 0
    velocity_resample_max_steps: int = 0
    velocity_resample_steps_remaining: torch.Tensor | None = None
    velocity_resample_mask: torch.Tensor | None = None
    velocity_resample_steps_remaining_wp: wp.array | None = None


class CommandRuntime:
    """Command-task logic for velocity and goal tracking.

    The env supplies morphology/control hooks such as root pose, planar velocity,
    and base observations.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        state = env.__dict__.get("command_state")
        if state is None:
            state = CommandState()
            env.command_state = state
        elif not isinstance(state, CommandState):
            raise TypeError("env.command_state must be CommandState")
        self.state = state

    def _direction_only_enabled(self) -> bool:
        env = self.env
        resolver: Callable[[], bool] | None = getattr(
            env, "_velocity_heading_direction_only_enabled", None
        )
        if resolver is None:
            return False
        return bool(resolver())

    def command_reward_term_names(self) -> tuple[str, ...]:
        if self.env._command_mode == "velocity_heading":
            if self._direction_only_enabled():
                return (
                    "alive_reward",
                    "upright_reward",
                    "height_reward",
                    "heading_reward",
                )
            return (
                "alive_reward",
                "upright_reward",
                "height_reward",
                "speed_reward",
                "heading_reward",
            )
        return (
            "alive_reward",
            "upright_reward",
            "height_reward",
            "progress_reward",
            "facing_reward",
            "goal_bonus",
            "stall_penalty",
        )

    def resolved_command_dim(self) -> int:
        env = self.env
        if env._command_mode == "velocity_heading":
            return 2 if self._direction_only_enabled() else 4
        return 4

    def resolved_command_num_envs(self) -> int:
        env = self.env
        try:
            return int(env.num_envs)
        except (AttributeError, TypeError, ValueError):
            goal_pos_xy = getattr(env, "_goal_pos_xy", None)
            if goal_pos_xy is not None:
                return int(goal_pos_xy.shape[0])
            target_speed = getattr(env, "_target_speed", None)
            if target_speed is not None:
                return int(target_speed.shape[0])
            cfg = getattr(env, "cfg", None)
            scene = getattr(cfg, "scene", None)
            return int(getattr(scene, "num_envs", 0))

    def init_command_reward_debug_buffers(self) -> None:
        env = self.env
        names = self.command_reward_term_names()
        self.state.eval_metrics_enabled = False
        self.state.reward_term_names = names
        self.state.reward_terms = torch.zeros(
            (self.resolved_command_num_envs(), len(names)),
            device=env.device,
            dtype=torch.float32,
        )

    def ensure_command_reward_debug_buffers(self) -> None:
        env = self.env
        names = self.command_reward_term_names()
        num_envs = self.resolved_command_num_envs()
        device = env._target_speed.device
        buf = self.state.reward_terms
        existing_names = self.state.reward_term_names
        if (
            buf is None
            or existing_names != names
            or tuple(buf.shape) != (num_envs, len(names))
            or buf.device != device
        ):
            self.state.reward_term_names = names
            self.state.reward_terms = torch.zeros(
                (num_envs, len(names)),
                device=device,
                dtype=torch.float32,
            )

    def configure_eval_metrics(self) -> None:
        env = self.env
        num_envs = self.resolved_command_num_envs()
        device = env._target_speed.device
        if self.state.eval_metrics_enabled:
            for name, value in (
                ("goal_reached", self.state.goal_reached),
                ("velocity_error", self.state.velocity_error),
                ("heading_alignment", self.state.heading_alignment),
            ):
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != (num_envs,)
                    or value.device != device
                ):
                    raise RuntimeError(
                        f"persistent evaluation metric buffer {name} was replaced"
                    )
            return
        self.state.goal_reached = torch.zeros(
            num_envs,
            device=device,
            dtype=torch.bool,
        )
        self.state.velocity_error = torch.zeros(
            num_envs,
            device=device,
            dtype=torch.float32,
        )
        self.state.heading_alignment = torch.zeros(
            num_envs,
            device=device,
            dtype=torch.float32,
        )
        self.state.eval_metrics_enabled = True

    def velocity_command_resample_step_bounds(self) -> tuple[int, int]:
        env = self.env
        min_s = max(
            0.0,
            float(getattr(env.cfg, "velocity_command_resample_interval_min_s", 0.0)),
        )
        max_s = max(
            0.0,
            float(getattr(env.cfg, "velocity_command_resample_interval_max_s", 0.0)),
        )
        if max_s <= 0.0:
            return 0, 0
        if max_s < min_s:
            max_s = min_s
        step_dt = max(float(getattr(env, "step_dt", 0.0)), 1.0e-6)
        min_steps = max(1, int(round(min_s / step_dt)))
        max_steps = max(min_steps, int(round(max_s / step_dt)))
        return min_steps, max_steps

    def init_velocity_command_resample_buffers(self) -> None:
        env = self.env
        min_steps, max_steps = self.velocity_command_resample_step_bounds()
        self.state.velocity_resample_min_steps = int(min_steps)
        self.state.velocity_resample_max_steps = int(max_steps)
        num_envs = self.resolved_command_num_envs()
        self.state.velocity_resample_steps_remaining = torch.zeros(
            num_envs,
            device=env.device,
            dtype=torch.int32,
        )
        self.state.velocity_resample_mask = torch.zeros(
            num_envs,
            device=env.device,
            dtype=torch.bool,
        )
        self.state.velocity_resample_steps_remaining_wp = wp.from_torch(
            self.state.velocity_resample_steps_remaining
        )

    def velocity_command_resample_enabled(self) -> bool:
        env = self.env
        return (
            env._command_mode == "velocity_heading"
            and self.state.velocity_resample_max_steps > 0
        )

    def maybe_resample_velocity_commands(self) -> None:
        env = self.env
        if not self.velocity_command_resample_enabled():
            return
        if self.state.velocity_resample_steps_remaining.is_cuda:
            wp.launch(
                update_velocity_command_resample_mask_kernel,
                dim=(self.resolved_command_num_envs(),),
                inputs=[self.state.velocity_resample_steps_remaining_wp],
                outputs=[self.state.velocity_resample_mask],
            )
            env._sample_commands(self.state.velocity_resample_mask)
            return

        self.state.velocity_resample_steps_remaining.sub_(1)
        self.state.velocity_resample_mask.copy_(
            self.state.velocity_resample_steps_remaining <= 0
        )
        if bool(self.state.velocity_resample_mask.any().item()):
            env._sample_commands(self.state.velocity_resample_mask)

    def target_direction(self) -> torch.Tensor:
        env = self.env
        override = getattr(env, "__dict__", {}).get("_target_direction")
        if callable(override):
            return override()
        target_dir = torch.cat(
            (torch.cos(env._target_heading), torch.sin(env._target_heading)),
            dim=-1,
        )
        return torch.nn.functional.normalize(target_dir, dim=-1)

    def goal_local_delta_and_distance(self) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env
        root_pos, root_quat = env._root_pos_quat()
        root_pos_xz, root_yaw_quat = get_heading_frame(root_pos, root_quat)
        goal_delta_xy = env._goal_pos_xy - root_pos_xz[:, :2]
        goal_delta = torch.zeros(
            (env.num_envs, 3), device=env.device, dtype=root_pos.dtype
        )
        goal_delta[:, :2] = goal_delta_xy
        goal_local = quat_torch.mul_vec(quat_torch.inverse(root_yaw_quat), goal_delta)
        goal_distance = torch.linalg.vector_norm(goal_delta_xy, dim=-1, keepdim=True)
        return goal_local[:, :2], goal_distance

    def sample_commands(self, reset_mask: torch.Tensor) -> None:
        env = self.env
        fixed_target_speed = getattr(env.cfg, "fixed_target_speed", None)
        fixed_target_heading = getattr(env.cfg, "fixed_target_heading", None)
        target_speed_min = float(getattr(env.cfg, "target_speed_min", 0.0))
        target_speed_max = float(getattr(env.cfg, "target_speed_max", 0.0))
        direction_only = (
            env._command_mode == "velocity_heading" and self._direction_only_enabled()
        )
        fixed_target_speed_enabled = int(
            direction_only or fixed_target_speed is not None
        )
        fixed_target_heading_enabled = int(fixed_target_heading is not None)
        velocity_resample_min_steps = self.state.velocity_resample_min_steps
        velocity_resample_max_steps = self.state.velocity_resample_max_steps
        point_goal_world_radius_max = getattr(
            env.cfg, "point_goal_world_radius_max", None
        )
        point_goal_world_radius_enabled = int(point_goal_world_radius_max is not None)
        point_goal_world_radius_value = (
            0.0
            if point_goal_world_radius_max is None
            else float(point_goal_world_radius_max)
        )
        if reset_mask.is_cuda:
            command_mode = (
                COMMAND_MODE_POINT_GOAL
                if env._command_mode == "point_goal"
                else COMMAND_MODE_VELOCITY_HEADING
            )
            wp.launch(
                sample_commands_kernel,
                dim=(env.num_envs,),
                inputs=[
                    reset_mask,
                    env.common_step_counter,
                    command_mode,
                    0.0 if direction_only else target_speed_min,
                    0.0 if direction_only else target_speed_max,
                    fixed_target_speed_enabled,
                    (
                        0.0
                        if direction_only
                        else (
                            0.0
                            if fixed_target_speed is None
                            else float(fixed_target_speed)
                        )
                    ),
                    fixed_target_heading_enabled,
                    0.0
                    if fixed_target_heading is None
                    else float(fixed_target_heading),
                    velocity_resample_min_steps,
                    velocity_resample_max_steps,
                    float(env.cfg.point_goal_distance_min),
                    float(env.cfg.point_goal_distance_max),
                    point_goal_world_radius_enabled,
                    point_goal_world_radius_value,
                    env.mjw_data.qpos,
                ],
                outputs=[
                    env._target_speed_wp,
                    env._target_heading_wp,
                    env._goal_pos_xy_wp,
                    env._prev_goal_distance_wp,
                    self.state.velocity_resample_steps_remaining_wp,
                ],
            )
            return

        env._command_reset_mask_float_buf.copy_(
            reset_mask.reshape(-1, 1).to(dtype=torch.float32)
        )
        if direction_only:
            env._sampled_target_speed_buf.zero_()
        else:
            env._sampled_target_speed_buf.uniform_(target_speed_min, target_speed_max)
        env._sampled_target_heading_buf.uniform_(-math.pi, math.pi)
        if direction_only:
            env._sampled_target_speed_buf.zero_()
        elif fixed_target_speed is not None:
            env._sampled_target_speed_buf.fill_(float(fixed_target_speed))
        if fixed_target_heading is not None:
            env._sampled_target_heading_buf.fill_(float(fixed_target_heading))
        env._target_speed.lerp_(
            env._sampled_target_speed_buf,
            env._command_reset_mask_float_buf,
        )
        env._target_heading.lerp_(
            env._sampled_target_heading_buf,
            env._command_reset_mask_float_buf,
        )
        velocity_steps_remaining = self.state.velocity_resample_steps_remaining
        if (
            env._command_mode == "velocity_heading"
            and velocity_steps_remaining is not None
            and velocity_resample_max_steps > 0
        ):
            sampled_steps = torch.randint(
                velocity_resample_min_steps,
                velocity_resample_max_steps + 1,
                (velocity_steps_remaining.shape[0],),
                device=velocity_steps_remaining.device,
                dtype=torch.int32,
            )
            velocity_steps_remaining.copy_(
                torch.where(reset_mask, sampled_steps, velocity_steps_remaining)
            )

        if env._command_mode != "point_goal":
            return
        root_pos, _ = env._root_pos_quat()
        if point_goal_world_radius_max is None:
            env._sampled_goal_distance_buf.uniform_(
                float(env.cfg.point_goal_distance_min),
                float(env.cfg.point_goal_distance_max),
            )
            env._sampled_goal_angle_buf.uniform_(-math.pi, math.pi)
            torch.cos(
                env._sampled_goal_angle_buf,
                out=env._sampled_goal_pos_xy_buf[:, 0:1],
            )
            torch.sin(
                env._sampled_goal_angle_buf,
                out=env._sampled_goal_pos_xy_buf[:, 1:2],
            )
            env._sampled_goal_pos_xy_buf.mul_(env._sampled_goal_distance_buf)
            env._sampled_goal_pos_xy_buf.add_(root_pos[:, :2])
        else:
            env._sampled_goal_distance_buf.uniform_(0.0, 1.0)
            env._sampled_goal_angle_buf.uniform_(0.0, 1.0)
            sample_point_goals_with_world_radius_limit_torch(
                root_pos_xy=root_pos[:, :2],
                distance_u=env._sampled_goal_distance_buf,
                angle_u=env._sampled_goal_angle_buf,
                distance_min=float(env.cfg.point_goal_distance_min),
                distance_max=float(env.cfg.point_goal_distance_max),
                world_radius_max=float(point_goal_world_radius_max),
                out_goal_pos_xy=env._sampled_goal_pos_xy_buf,
                out_goal_distance=env._sampled_goal_distance_buf,
            )
        env._goal_pos_xy.lerp_(
            env._sampled_goal_pos_xy_buf,
            env._command_reset_mask_float_buf.expand(-1, 2),
        )
        env._prev_goal_distance.lerp_(
            env._sampled_goal_distance_buf,
            env._command_reset_mask_float_buf,
        )

    def get_observations(
        self,
        out_obs: torch.Tensor | None = None,
    ) -> Mapping[str, torch.Tensor]:
        env = self.env
        observations = getattr(env, "obs_buf", None)
        if out_obs is not None:
            target = out_obs
            result = {"policy": target}
        elif isinstance(observations, Mapping) and "policy" in observations:
            target = observations["policy"]
            result = observations
        else:
            target = env._obs_buf
            result = {"policy": target}
        deployable_observation_indices = env._deployable_obs_indices
        command_dim = env._command_dim
        command_observation_buffer = env._command_obs_buf
        deployable_dim = int(deployable_observation_indices.numel())
        deployable_target = target[:, :deployable_dim]
        if env._command_base_obs_buf is None:
            raise RuntimeError("Command observation buffer is not initialized.")
        base_target = env._command_base_obs_buf
        env._fill_command_base_obs(base_target)
        torch.index_select(
            base_target[:, : env._base_obs_main_dim],
            1,
            deployable_observation_indices,
            out=deployable_target,
        )
        command_ofs = deployable_dim

        if env._command_mode == "velocity_heading":
            if self._direction_only_enabled():
                command_observation_buffer[:, 0:1].copy_(torch.cos(env._target_heading))
                command_observation_buffer[:, 1:2].copy_(torch.sin(env._target_heading))
            else:
                target_dir = self.target_direction()
                forward_speed = torch.sum(
                    env._planar_velocity() * target_dir,
                    dim=-1,
                    keepdim=True,
                )
                command_observation_buffer[:, 0:1].copy_(env._target_speed)
                command_observation_buffer[:, 1:2].copy_(torch.cos(env._target_heading))
                command_observation_buffer[:, 2:3].copy_(torch.sin(env._target_heading))
                command_observation_buffer[:, 3:4].copy_(
                    env._target_speed - forward_speed
                )
        else:
            goal_local_delta, goal_distance = self.goal_local_delta_and_distance()
            command_observation_buffer[:, 0:2].copy_(goal_local_delta)
            command_observation_buffer[:, 2:3].copy_(goal_distance)
            command_observation_buffer[:, 3:4].copy_(env._target_speed)

        target[:, command_ofs : command_ofs + command_dim].copy_(
            command_observation_buffer
        )
        suffix_ofs = command_ofs + command_dim
        if env._base_obs_suffix_dim > 0:
            target[:, suffix_ofs:].copy_(
                env._command_base_obs_buf[:, env._base_obs_main_dim :]
            )
        return result

    def _velocity_heading_reward_mode(self) -> str:
        env = self.env
        resolver: Callable[[], str] | None = getattr(
            env, "_velocity_heading_reward_mode", None
        )
        if resolver is not None:
            return resolver()
        return resolve_velocity_heading_reward_mode(
            getattr(
                env.cfg,
                "velocity_heading_reward_mode",
                VELOCITY_REWARD_MODE_FORWARD_SPEED_HEADING,
            )
        )

    def get_rewards(self) -> torch.Tensor:
        env = self.env
        self.ensure_command_reward_debug_buffers()
        eval_metrics_enabled = self.state.eval_metrics_enabled
        if eval_metrics_enabled:
            self.state.goal_reached.zero_()
        alive_reward = torch.full_like(
            env._target_speed,
            float(env.cfg.command_alive_reward),
        )
        upright_reward = float(env.cfg.command_upright_weight) * env._up_reward()
        height_reward = (
            float(getattr(env.cfg, "command_torso_height_weight", 0.0))
            * env._torso_height_reward()
        )
        if env._command_mode == "velocity_heading":
            target_dir = self.target_direction()
            planar_velocity = env._planar_velocity()
            heading_vec = env._heading_xy()
            heading_alignment = torch.sum(heading_vec * target_dir, dim=-1)
            if eval_metrics_enabled:
                self.state.heading_alignment.copy_(heading_alignment)
            if self._direction_only_enabled():
                heading_reward = float(
                    env.cfg.velocity_heading_heading_weight
                ) * torch.clamp(
                    heading_alignment.unsqueeze(-1),
                    min=0.0,
                )
                reward = alive_reward + upright_reward + height_reward + heading_reward
                self.state.reward_terms[:, 0:1].copy_(alive_reward)
                self.state.reward_terms[:, 1:2].copy_(upright_reward)
                self.state.reward_terms[:, 2:3].copy_(height_reward)
                self.state.reward_terms[:, 3:4].copy_(heading_reward)
            else:
                if eval_metrics_enabled:
                    target_velocity = env._target_speed * target_dir
                    velocity_error = torch.linalg.vector_norm(
                        planar_velocity - target_velocity,
                        dim=-1,
                    )
                    self.state.velocity_error.copy_(velocity_error)
                speed_reward, heading_reward = compute_velocity_heading_reward_terms(
                    planar_velocity=planar_velocity,
                    heading_vec=heading_vec,
                    target_dir=target_dir,
                    target_speed=env._target_speed,
                    speed_sigma=float(env.cfg.velocity_heading_speed_sigma),
                    heading_weight=float(env.cfg.velocity_heading_heading_weight),
                    reward_mode=self._velocity_heading_reward_mode(),
                )
                reward = (
                    alive_reward
                    + upright_reward
                    + height_reward
                    + speed_reward
                    + heading_reward
                )
                self.state.reward_terms[:, 0:1].copy_(alive_reward)
                self.state.reward_terms[:, 1:2].copy_(upright_reward)
                self.state.reward_terms[:, 2:3].copy_(height_reward)
                self.state.reward_terms[:, 3:4].copy_(speed_reward)
                self.state.reward_terms[:, 4:5].copy_(heading_reward)
            self.maybe_resample_velocity_commands()
        else:
            root_pos, _ = env._root_pos_quat()
            progress_reward, facing_reward, goal_bonus, stall_penalty = (
                compute_point_goal_reward_terms(
                    root_pos_xy=root_pos[:, :2],
                    heading_vec=env._heading_xy(),
                    goal_pos_xy=env._goal_pos_xy,
                    prev_goal_distance=env._prev_goal_distance,
                    goal_radius=float(env.cfg.point_goal_radius),
                    progress_weight=float(env.cfg.point_goal_progress_weight),
                    facing_weight=float(env.cfg.point_goal_facing_weight),
                    goal_bonus_value=float(env.cfg.point_goal_bonus),
                    stall_penalty_weight=float(env.cfg.point_goal_stall_penalty_weight),
                    min_progress_for_stall=float(
                        env.cfg.point_goal_min_progress_for_stall
                    ),
                )
            )
            goal_distance = torch.linalg.vector_norm(
                env._goal_pos_xy - root_pos[:, :2],
                dim=-1,
                keepdim=True,
            )
            env._prev_goal_distance.copy_(goal_distance)
            reward = (
                alive_reward
                + upright_reward
                + height_reward
                + progress_reward
                + facing_reward
                + goal_bonus
                + stall_penalty
            )
            self.state.reward_terms[:, 0:1].copy_(alive_reward)
            self.state.reward_terms[:, 1:2].copy_(upright_reward)
            self.state.reward_terms[:, 2:3].copy_(height_reward)
            self.state.reward_terms[:, 3:4].copy_(progress_reward)
            self.state.reward_terms[:, 4:5].copy_(facing_reward)
            self.state.reward_terms[:, 5:6].copy_(goal_bonus)
            self.state.reward_terms[:, 6:7].copy_(stall_penalty)
            goal_reached_mask = (
                goal_distance <= float(env.cfg.point_goal_radius)
            ).reshape(-1)
            if eval_metrics_enabled:
                self.state.goal_reached.copy_(goal_reached_mask)
            if goal_reached_mask.is_cuda or bool(goal_reached_mask.any().item()):
                env._sample_commands(goal_reached_mask)
        reward = reward.reshape(-1)
        reward_out = env.reward_buf
        if reward_out is not reward:
            reward_out.copy_(reward)

        # Done logic needs a persistent previous-step reward because caller-bound
        # StepIO reward slots rotate and reset handling clears this state in-place.
        done_reward = env.get_persistent_command_reward_buffer()
        if done_reward is not reward_out:
            done_reward.copy_(reward_out)
        env.total_reward = done_reward
        return reward_out

    def get_goal_reached_mask(self) -> torch.Tensor:
        if not self.state.eval_metrics_enabled:
            # Playback reads goal reaches without a benchmark. Enabling the metric
            # buffers changes what the step writes, so drop any captured step.
            self.configure_eval_metrics()
            self.env.invalidate_step_graph_cache()
        assert self.state.goal_reached is not None
        return self.state.goal_reached

    def get_eval_metric_specs(self) -> dict[str, EvalMetricSpec]:
        env = self.env
        self.configure_eval_metrics()
        if env._command_mode == "point_goal":
            return {
                "goal_reaches_per_env": EvalMetricSpec(
                    value=self.state.goal_reached,
                    reduction="event_count",
                    label="Goal reaches per env",
                )
            }
        metrics = {
            "heading_alignment_mean": EvalMetricSpec(
                value=self.state.heading_alignment,
                reduction="step_mean",
                label="Heading alignment mean",
            )
        }
        if not self._direction_only_enabled():
            metrics = {
                "velocity_error_mean": EvalMetricSpec(
                    value=self.state.velocity_error,
                    reduction="step_mean",
                    label="Velocity error mean (m/s)",
                ),
                **metrics,
            }
        return metrics

    def get_eval_episode_spec(self) -> EvalEpisodeSpec:
        return EvalEpisodeSpec(validity="always", success=None)

    def reset_after_base_reset(self, reset_mask: torch.Tensor) -> None:
        env = self.env
        early_timer = getattr(env, "early_termination_timer", None)
        if early_timer is not None:
            early_timer.masked_fill_(reset_mask, 0)
        total_reward = getattr(env, "total_reward", None)
        if total_reward is not None and total_reward.shape[0] == reset_mask.shape[0]:
            total_reward.masked_fill_(reset_mask, 0.0)
        env._sample_commands(reset_mask)

    def get_command_diagnostic_means(self) -> dict[str, float]:
        env = self.env
        metrics: dict[str, float] = {}
        if env._command_mode == "velocity_heading":
            target_dir = self.target_direction()
            forward_speed = torch.sum(
                env._planar_velocity() * target_dir, dim=-1, keepdim=True
            )
            heading_align = torch.clamp(
                torch.sum(env._heading_xy() * target_dir, dim=-1, keepdim=True), min=0.0
            )
            metrics["forward_speed_mean"] = float(forward_speed.mean().item())
            metrics["heading_align_mean"] = float(heading_align.mean().item())
        return metrics

    def get_video_overlay_plots(self, env_idx: int = 0) -> dict[str, object]:
        env = self.env
        num_envs = self.resolved_command_num_envs()
        if env_idx < 0 or env_idx >= num_envs:
            return {}
        self.ensure_command_reward_debug_buffers()
        series = {
            name: float(self.state.reward_terms[env_idx, idx].detach().cpu().item())
            for idx, name in enumerate(self.state.reward_term_names)
        }
        series["total_reward"] = float(env.total_reward[env_idx].detach().cpu().item())
        colors = {
            name: env._COMMAND_REWARD_COLORS[name]
            for name in series
            if name in env._COMMAND_REWARD_COLORS
        }
        return {
            "title": "Command Reward Overlay",
            "plots": [
                {
                    "title": "command reward terms + total",
                    "series": series,
                    "colors": colors,
                },
            ],
        }
