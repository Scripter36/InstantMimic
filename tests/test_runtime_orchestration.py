from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from instantmimic.play import runtime as playback_runtime
from instantmimic.play.stepper import (
    _configure_playback_env_cuda_graph_mode,
    _policy_sample_graph_safe_into,
    _resolve_playback_graph_mode,
)
from instantmimic.rl.models.actor_critic import ActorCritic
from instantmimic.train.metrics import build_training_iteration_scalars
from instantmimic.train.rollout import RolloutRuntime
from instantmimic.train.update import _run_training_update_cycle


def test_playback_sampling_uses_live_gaussian_noise_and_fixed_output_storage():
    torch.manual_seed(9)
    network = ActorCritic(3, 2, layers=(8,), initial_log_std=-1.0)
    observations = torch.randn(4, 3)
    noise = torch.randn(4, 2)
    scaled = torch.empty_like(observations)
    calls = []

    def fill_noise():
        calls.append("noise")
        return noise

    agent = SimpleNamespace(
        _policy_scaled_obs=scaled,
        _obs_scale_eval_into_fn=lambda obs, output: output.copy_(obs * 0.3),
        _fill_policy_noise_=fill_noise,
        _act_into_fn=network.act_into,
    )
    outputs = (torch.empty(4, 2), torch.empty(4, 1), torch.empty(4, 1))
    pointers = tuple(tensor.data_ptr() for tensor in outputs)
    _policy_sample_graph_safe_into(agent, observations, *outputs)
    expected = tuple(torch.empty_like(tensor) for tensor in outputs)
    with torch.no_grad():
        network.act_into(observations * 0.3, *expected, noise)
    for actual, wanted in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert tuple(tensor.data_ptr() for tensor in outputs) == pointers
    assert calls == ["noise"]


def test_rollout_binds_the_original_action_slot_without_copying():
    runtime = object.__new__(RolloutRuntime)
    actions = torch.randn(5, 3, 2)
    runtime.buf = SimpleNamespace(actions=actions)
    for step in (0, 2, 4):
        output = runtime._resolve_rollout_action_binding(step)
        assert output.data_ptr() == actions[step].data_ptr()
        assert (
            output.untyped_storage().data_ptr() == actions.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(output, actions[step], rtol=0, atol=0)


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("video", [False, True])
def test_graph_mode_selection_keeps_normal_mean_and_sample_support(sample, video):
    env = SimpleNamespace(
        cfg=SimpleNamespace(use_cuda_graph=True), step_with_io=lambda *a: None
    )
    _configure_playback_env_cuda_graph_mode(env, take_video=video)
    assert env.cfg.use_cuda_graph is True
    mode = _resolve_playback_graph_mode(
        requested=True,
        device=torch.device("cuda"),
        env_raw=env,
        take_video=video,
        play_length=4,
        stochastic_play=sample,
    )
    assert mode == ("video_device_step" if video else "headless")


def test_renderer_activation_comes_from_the_environment_owner(monkeypatch):
    activation = torch.tensor([[0.1, 0.6]])
    env = SimpleNamespace(get_full_video_activation=lambda: activation)
    monkeypatch.setattr(
        playback_runtime,
        "MujocoWarpPlaybackRenderer",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    renderer = playback_runtime._build_playback_renderer(
        video_render_backend="warp",
        renderer_kwargs={},
        video_tile_layout=SimpleNamespace(env_count=1, cols=1),
        cfg=DictConfig({}),
        mujoco_gl_overlay_renderer=None,
        character_render_mode="collision",
        activation=playback_runtime._environment_activation(env),
    )
    assert renderer.lbs_activation_tensor is activation
    assert playback_runtime._environment_activation(object()) is None


class _PolicyOwner:
    def __init__(self, events):
        self.network = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)
        self.events = events

    def policy_std_mean(self):
        return torch.tensor(0.5)

    def update(self):
        self.events.append("policy_update")
        loss = self.network(torch.ones(4, 3)).square().mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {
            "policy": float(loss.detach()),
            "value": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "lr": 0.01,
            "std": 0.5,
        }


class _MuscleOwner:
    def __init__(self, events):
        self.network = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)
        self.events = events
        self.fast_muscle = SimpleNamespace(
            begin_muscle_nn_update=lambda: events.append("muscle_begin")
        )

    def optimize_muscle_nn(self):
        self.events.append("muscle_wait")
        loss = self.network(torch.ones(4, 3)).square().mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "target_loss": float(loss.detach()),
            "reg_loss": 0.1,
            "bound_loss": 0.2,
            "buffer_size": 4,
        }


@pytest.mark.parametrize("stage", ["musclenn_only", "full"])
def test_mass_stage_ownership_and_plain_metrics(stage):
    torch.manual_seed(31)
    events = []
    policy, muscle = _PolicyOwner(events), _MuscleOwner(events)
    before_policy = copy.deepcopy(policy.network.state_dict())
    before_muscle = copy.deepcopy(muscle.network.state_dict())
    checkpoint = SimpleNamespace(
        wait_for_update_boundary=lambda: events.append("checkpoint_wait")
    )
    losses, count, elapsed = _run_training_update_cycle(
        training_stage=stage,
        checkpoint_manager=checkpoint,
        agent=policy,
        env_raw_for_agent=muscle,
        rollout_obs=torch.zeros(2, 4, 3),
        update_count=7,
    )
    assert count == 8 and elapsed >= 0
    assert events == [
        "checkpoint_wait",
        "muscle_begin",
        *(["policy_update"] if stage == "full" else []),
        "muscle_wait",
    ]
    assert any(
        not torch.equal(value, muscle.network.state_dict()[name])
        for name, value in before_muscle.items()
    )
    changed_policy = any(
        not torch.equal(value, policy.network.state_dict()[name])
        for name, value in before_policy.items()
    )
    assert changed_policy == (stage == "full")
    step, frame = build_training_iteration_scalars(
        last_losses=losses,
        agent=policy,
        env_raw=muscle,
        frames=32,
        update_count=count,
        last_mean_reward=0.8,
        last_mean_episode_timesteps=4,
        timing_rollouts=1,
        physics_time_ns_acc=1000,
        update_time_ns_acc=elapsed,
        active_train_seconds=0.1,
        checkpoint_time_seconds=0.0,
    )
    assert step["Loss / MuscleNN target loss"] == losses["musclenn_target_loss"]
    assert step["Loss / MuscleNN bound loss"] == 0.2
    assert frame["Loss / MuscleNN bound loss (frames)"] == 0.2
    assert step["Info / Updates"] == 8


def test_benchmark_step_io_uses_the_policy_action_storage():
    from instantmimic.train.benchmark_runner import _TrackingBenchmarkWorkspace

    workspace = _TrackingBenchmarkWorkspace(
        device=torch.device("cpu"),
        total_envs=3,
        obs_dim=4,
        policy_act_dim=2,
        env_act_dim=2,
    )
    assert workspace.env_actions is workspace.policy_actions
    assert workspace.step_io.inputs.actions is workspace.policy_actions
    assert workspace.matches(total_envs=3, obs_dim=4, policy_act_dim=2, env_act_dim=2)
    with pytest.raises(ValueError, match="action widths"):
        _TrackingBenchmarkWorkspace(
            device=torch.device("cpu"),
            total_envs=3,
            obs_dim=4,
            policy_act_dim=2,
            env_act_dim=3,
        )


@pytest.mark.parametrize("obs_dim", [3, 5])
def test_policy_initialization_restores_gaussian_policy_and_scaler_prefix(
    tmp_path, obs_dim
):
    from instantmimic.rl.normalization import RunningStandardScaler
    from instantmimic.train.checkpointing import _load_agent_checkpoint

    torch.manual_seed(71)
    source = ActorCritic(3, 2, layers=(4,))
    target = ActorCritic(obs_dim, 2, layers=(4,))
    source_scaler = RunningStandardScaler(3)
    source_scaler.update_stats(torch.arange(24, dtype=torch.float32).reshape(8, 3))
    target_scaler = RunningStandardScaler(obs_dim)
    value_scaler = RunningStandardScaler(1)
    optimizer = torch.optim.Adam(target.parameters(), lr=0.0123)
    initial = copy.deepcopy(target.state_dict())
    scaler_tail = target_scaler.running_variance[3:].clone()
    initial_value = copy.deepcopy(value_scaler.state_dict())
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {"network": source.state_dict(), "obs_scaler": source_scaler.state_dict()},
        checkpoint,
    )
    agent = SimpleNamespace(
        network=target,
        obs_scaler=target_scaler,
        value_scaler=value_scaler,
        optimizer=optimizer,
    )
    _load_agent_checkpoint(agent, checkpoint, checkpoint_load_mode="policy_init")
    current = target.state_dict()
    for name, expected in source.state_dict().items():
        if name.startswith("value_head."):
            torch.testing.assert_close(current[name], initial[name], rtol=0, atol=0)
        elif name in {"trunk.0.weight", "actor_trunk.0.weight"} and obs_dim > 3:
            torch.testing.assert_close(current[name][:, :3], expected, rtol=0, atol=0)
            torch.testing.assert_close(
                current[name][:, 3:], initial[name][:, 3:], rtol=0, atol=0
            )
        else:
            torch.testing.assert_close(current[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        target_scaler.running_mean[:3], source_scaler.running_mean, rtol=0, atol=0
    )
    torch.testing.assert_close(
        target_scaler.running_variance[:3],
        source_scaler.running_variance,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        target_scaler.running_variance[3:], scaler_tail, rtol=0, atol=0
    )
    for name, expected in initial_value.items():
        torch.testing.assert_close(
            value_scaler.state_dict()[name], expected, rtol=0, atol=0
        )
    assert optimizer.state == {}
    assert optimizer.param_groups[0]["lr"] == 0.0123


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("fixed_sigma", [False, True])
@pytest.mark.parametrize("mean_mode", ["none", "clamp", "tanh"])
def test_benchmark_preserves_authored_policy_precision(
    bf16, separate, fixed_sigma, mean_mode
):
    from instantmimic.rl.ppo.agent import PPOAgent
    from instantmimic.train.benchmark_runner import _BenchmarkPolicyRunner

    torch.manual_seed(117)
    agent = PPOAgent(
        5,
        2,
        3,
        "cpu",
        rollouts=2,
        learning_epochs=1,
        mini_batches=1,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=3e-4,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.0,
        value_loss_scale=1.0,
        kl_threshold=0.02,
        layers=(8, 4),
        activation="silu",
        fixed_sigma=fixed_sigma,
        squash_actions=mean_mode == "tanh",
        clamp_mean=mean_mode == "clamp",
        clamp_mean_min=-0.4,
        clamp_mean_max=0.7,
        network_kwargs={"separate": separate, "policy_mlp_bf16": bf16},
        inference_only=True,
    )
    agent.obs_scaler.update_stats(torch.randn(32, 5))
    agent.network.action_mean_scale.copy_(torch.tensor([0.7, 1.3]))
    runner = _BenchmarkPolicyRunner(
        agent=agent,
        device=torch.device("cpu"),
        obs_dim=5,
        batch_size=3,
        compile_enabled=False,
        compile_mode="default",
        use_cuda_graph=False,
    )
    observations = torch.randn(3, 5)
    expected, actual = torch.empty(3, 2), torch.empty(3, 2)
    agent.policy_mean_into(observations, expected)
    assert agent.network.policy_mlp_bf16 is bf16
    if bf16:
        assert not runner.supports_preallocated_policy_mean
        assert not runner.uses_preallocated_policy_path
        with pytest.raises(RuntimeError, match="equivalent native policy snapshot"):
            runner.policy_mean_into(observations, actual, require_preallocated=True)
        runner.policy_mean_into(observations, actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert runner.supports_preallocated_policy_mean
        runner.policy_mean_into(observations, actual, require_preallocated=True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert agent.network.policy_mlp_bf16 is bf16


@pytest.mark.parametrize("squash", [False, True])
@pytest.mark.parametrize("clamp", [False, True])
def test_native_output_kernel_matches_gaussian_mean_bounds(squash, clamp):
    import warp as wp

    from instantmimic.train.benchmark_runner import _tracking_eval_policy_output_kernel

    torch.manual_seed(63)
    features, weight, bias, scale = (
        torch.randn(3, 4),
        torch.randn(2, 4),
        torch.randn(2),
        torch.tensor([0.7, 1.3]),
    )
    actual = torch.empty(3, 2)
    wp.launch(
        _tracking_eval_policy_output_kernel,
        dim=(3, 2),
        inputs=[
            wp.from_torch(features),
            wp.from_torch(weight),
            wp.from_torch(bias),
            wp.from_torch(scale),
            1,
            int(squash),
            int(clamp),
            -0.4,
            0.7,
            wp.from_torch(actual),
        ],
        device="cpu",
    )
    expected = (features @ weight.t() + bias) * scale
    if squash:
        expected.tanh_()
    elif clamp:
        expected.clamp_(-0.4, 0.7)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("normalizer", ["standard", "rsl", "disabled"])
def test_complete_policy_snapshot_is_frozen_refreshable_and_capture_eligible(
    bf16, normalizer
):
    from instantmimic.rl.normalization import (
        RslRlEmpiricalNormalization,
        RunningStandardScaler,
    )
    from instantmimic.rl.ppo.agent import PPOAgent
    from instantmimic.train.benchmark_runner import (
        RolloutBenchmarkRunner,
        _BenchmarkPolicyRunner,
    )

    torch.manual_seed(119)
    agent = PPOAgent.__new__(PPOAgent)
    agent.device = torch.device("cpu")
    agent.obs_dim, agent.obs_main_dim, agent.act_dim, agent.num_envs = 5, 5, 2, 3
    agent.asymmetric_observations = False
    agent.network = ActorCritic(5, 2, layers=(8, 4), policy_mlp_bf16=bf16)
    agent.obs_scaler = (
        RslRlEmpiricalNormalization(5)
        if normalizer == "rsl"
        else RunningStandardScaler(5, enabled=normalizer != "disabled")
    )
    agent._policy_scaled_obs = torch.empty(3, 5)
    agent._obs_scale_eval_into_fn = agent._obs_scale_eval_into
    runner = _BenchmarkPolicyRunner(
        agent=agent,
        device=torch.device("cpu"),
        obs_dim=5,
        batch_size=3,
        compile_enabled=False,
        compile_mode="default",
        use_cuda_graph=False,
    )
    snapshot = runner._complete_policy_snapshot
    assert snapshot is not None
    pointers = {
        f"network.{name}": value.data_ptr()
        for name, value in snapshot.network.state_dict().items()
    } | {
        f"scaler.{name}": value.data_ptr()
        for name, value in snapshot.scaler.state_dict().items()
    }
    observations = torch.randn(3, 5)
    before, actual, updated = torch.empty(3, 2), torch.empty(3, 2), torch.empty(3, 2)
    agent.policy_mean_into(observations, before)
    with torch.no_grad():
        agent.network.policy_head.bias.add_(0.4)
        agent.obs_scaler.update_stats(torch.randn(32, 5) * 1.4 + 0.5)
    runner.policy_mean_into(observations, actual)
    torch.testing.assert_close(actual, before, rtol=0, atol=0)
    agent.policy_mean_into(observations, updated)
    assert not torch.equal(updated, before)
    runner.refresh_preallocated_policy_state()
    runner.policy_mean_into(observations, actual)
    torch.testing.assert_close(actual, updated, rtol=0, atol=0)
    refreshed = {
        f"network.{name}": value.data_ptr()
        for name, value in snapshot.network.state_dict().items()
    } | {
        f"scaler.{name}": value.data_ptr()
        for name, value in snapshot.scaler.state_dict().items()
    }
    assert pointers == refreshed
    assert snapshot.scaled_obs.data_ptr() != agent._policy_scaled_obs.data_ptr()

    # Capability selection uses device metadata only; this test neither starts
    # CUDA nor substitutes for executing the nested graph on a GPU.
    runner._device = torch.device("cuda")
    benchmark = RolloutBenchmarkRunner.__new__(RolloutBenchmarkRunner)
    benchmark._use_cuda_graph = True
    benchmark._device = torch.device("cuda")
    benchmark._torch_stream, benchmark._wp_stream = object(), object()
    benchmark._policy_runner = runner
    env = SimpleNamespace(
        reset_into=lambda *args: None,
        step_with_io=lambda *args: None,
    )
    assert runner.can_capture_complete_policy_mean
    assert runner.supports_preallocated_policy_mean is (
        not bf16 and normalizer != "rsl"
    )
    assert benchmark._captured_eval_supported(env, env.step_with_io)
    assert benchmark._use_cuda_graph is True
    assert not torch.cuda.is_initialized()


def test_custom_policy_cannot_claim_frozen_overlap_capture():
    from instantmimic.rl.normalization import RunningStandardScaler
    from instantmimic.rl.ppo.agent import PPOAgent
    from instantmimic.train.benchmark_runner import _BenchmarkPolicyRunner

    class CustomActor(ActorCritic):
        def policy_mean_into(self, observations, actions):
            super().policy_mean_into(observations, actions)
            actions.add_(0.125)

    agent = PPOAgent.__new__(PPOAgent)
    agent.device = torch.device("cpu")
    agent.obs_dim, agent.obs_main_dim, agent.act_dim, agent.num_envs = 5, 5, 2, 3
    agent.asymmetric_observations = False
    agent.network = CustomActor(5, 2, layers=(8, 4))
    agent.obs_scaler = RunningStandardScaler(5)
    agent._policy_scaled_obs = torch.empty(3, 5)
    agent._obs_scale_eval_into_fn = agent._obs_scale_eval_into
    runner = _BenchmarkPolicyRunner(
        agent=agent,
        device=agent.device,
        obs_dim=5,
        batch_size=3,
        compile_enabled=False,
        compile_mode="default",
        use_cuda_graph=False,
    )
    observations, expected, actual = (
        torch.randn(3, 5),
        torch.empty(3, 2),
        torch.empty(3, 2),
    )
    agent.policy_mean_into(observations, expected)
    runner.policy_mean_into(observations, actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not runner.supports_preallocated_policy_mean
    assert runner._complete_policy_snapshot is None
    runner._device = torch.device("cuda")
    assert not runner.can_capture_complete_policy_mean
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize(
    "normalizer", ["RunningStandardScaler", "RslRlEmpiricalNormalization", None]
)
@pytest.mark.parametrize("bound_step", [False, True])
def test_eager_rollout_benchmark_refreshes_policy_and_normalizer_each_run(
    bf16, normalizer, bound_step, monkeypatch, tmp_path
):
    from instantmimic.rl.ppo.agent import PPOAgent
    from instantmimic.train import benchmark_runner as benchmark

    torch.manual_seed(213)
    agent = PPOAgent(
        5,
        2,
        3,
        "cpu",
        rollouts=2,
        learning_epochs=1,
        mini_batches=1,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=3e-4,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.0,
        value_loss_scale=1.0,
        kl_threshold=0.02,
        layers=(8, 4),
        state_preprocessor=normalizer,
        network_kwargs={"policy_mlp_bf16": bf16},
        inference_only=True,
    )
    observations = torch.randn(3, 5)
    actions_seen = []

    class Environment:
        @property
        def unwrapped(self):
            return self

        def reset(self):
            return {"policy": observations.clone()}, {}

        def step(self, actions):
            actions_seen.append(actions.clone())
            return (
                {"policy": observations.clone()},
                actions.sum(dim=-1),
                torch.zeros(3, dtype=torch.bool),
                torch.zeros(3, dtype=torch.bool),
                {},
            )

    class BoundEnvironment(Environment):
        def step_with_io(self, step_io):
            obs, rewards, terminated, truncated, _ = self.step(step_io.inputs.actions)
            step_io.outputs.observations["policy"].copy_(obs["policy"])
            step_io.outputs.rewards.copy_(rewards)
            step_io.outputs.terminated.copy_(terminated)
            step_io.outputs.truncated.copy_(truncated)

    env = BoundEnvironment() if bound_step else Environment()
    monkeypatch.setattr(
        benchmark, "_build_tracking_benchmark_env", lambda *a, **kw: env
    )
    runner = benchmark.RolloutBenchmarkRunner(
        cfg=DictConfig({}),
        agent=agent,
        source_env=None,
        log_dir=tmp_path,
        device=agent.device,
        act_dim=2,
        num_envs=3,
        play_steps=2,
        use_cuda_graph=False,
        compile_enabled=False,
    )
    previous = None
    for run_index in range(2):
        if run_index:
            with torch.no_grad():
                agent.network.policy_head.bias.add_(0.7)
                agent.network.policy_head.weight.mul_(1.4)
            agent.obs_scaler.update_stats(torch.randn(31, 5) * 1.7 + 0.4)
        expected = torch.empty(3, 2)
        agent.policy_mean_into(observations, expected)
        actions_seen.clear()
        result = runner.run()
        assert len(actions_seen) == 2
        for actual in actions_seen:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert result["steps"] == 2
        assert result["reward_mean"] == pytest.approx(
            float(expected.sum(dim=-1).mean() * 2), abs=1e-6
        )
        if previous is not None:
            assert not torch.equal(previous, expected)
        previous = expected.clone()

    # Eager warmup also executes the current snapshot before a later run.
    with torch.no_grad():
        agent.network.policy_head.bias.add_(0.5)
    agent.obs_scaler.update_stats(torch.randn(23, 5) + 0.7)
    agent.policy_mean_into(observations, expected)
    actions_seen.clear()
    runner.prewarm_graphs()
    assert len(actions_seen) == 2
    for actual in actions_seen:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_captured_eval_graph_runs_conditional_horizon_through_capture_window(
    monkeypatch,
):
    from contextlib import nullcontext
    from unittest.mock import MagicMock, Mock

    from instantmimic.train import benchmark_runner as benchmark

    runner = benchmark.RolloutBenchmarkRunner.__new__(benchmark.RolloutBenchmarkRunner)
    runner._device = torch.device("cuda")
    runner._torch_stream = Mock()
    runner._wp_stream = Mock()
    runner._play_steps = 3
    runner._captured_eval_graph = None
    runner._captured_eval_step_graph = None
    runner._policy_runner = Mock(supports_preallocated_policy_mean=True)
    runner._captured_eval_supported = Mock(return_value=True)
    workspace = Mock()
    runner._warmup_captured_eval_step_body = Mock(return_value=workspace)
    runner._prepare_captured_eval_temporal_state = Mock()
    runner._captured_eval_reset_body = Mock()
    runner._captured_eval_step_body = Mock()
    env_raw = Mock()
    events: list[str] = []
    env_raw.prepare_step_capture_window.side_effect = lambda: events.append("prepare")
    runner._torch_stream.synchronize.side_effect = lambda: events.append("sync")
    env_raw.finalize_step_capture_window.side_effect = lambda: events.append("finalize")
    step_graph = object()
    step_capture = MagicMock()
    step_capture.__enter__.return_value = SimpleNamespace(graph=step_graph)
    full_graph = object()
    full_capture = MagicMock()
    full_capture.__enter__.return_value = SimpleNamespace(graph=full_graph)
    capture_launch = Mock()
    capture_while = Mock()
    monkeypatch.setattr(benchmark.torch.cuda, "current_stream", lambda *_: Mock())
    monkeypatch.setattr(benchmark.torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(benchmark.torch.cuda.nvtx, "range", lambda _: nullcontext())
    monkeypatch.setattr(
        benchmark.wp, "ScopedStream", lambda *_args, **_kwargs: nullcontext()
    )
    captures = iter([step_capture, full_capture])
    monkeypatch.setattr(benchmark, "capture_warp_graph", lambda **_: next(captures))
    monkeypatch.setattr(benchmark.wp, "capture_launch", capture_launch)
    monkeypatch.setattr(benchmark.wp, "capture_while", capture_while)

    assert runner._ensure_captured_eval_graph(
        env_raw=env_raw,
        workspace=workspace,
        step_with_io=Mock(),
    )
    assert runner._captured_eval_graph is full_graph
    assert runner._captured_eval_step_graph is step_graph
    runner._captured_eval_reset_body.assert_called_once_with(
        env_raw=env_raw, workspace=workspace
    )
    capture_while.assert_called_once_with(
        workspace.eval_condition_wp, while_body=step_graph
    )
    capture_launch.assert_called_once_with(full_graph, stream=runner._wp_stream)
    assert events == ["prepare", "sync", "finalize"]
