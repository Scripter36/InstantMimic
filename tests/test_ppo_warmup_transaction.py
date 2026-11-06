"""Production warmup must be observationally neutral on success and failure.

CPU tests run the actual PPO/optimizer work. Stream facades cover cleanup
ordering only; they do not stand in for execution on a CUDA device.
"""

from __future__ import annotations

import copy

import pytest
import torch

from instantmimic.rl.ppo.agent import PPOAgent


def _agent(optimizer_name="adamw", *, initialized=False):
    torch.manual_seed(81)
    agent = PPOAgent(
        obs_dim=4,
        act_dim=2,
        num_envs=2,
        device="cpu",
        rollouts=4,
        learning_epochs=2,
        mini_batches=2,
        shuffle_mode="per_epoch",
        discount_factor=0.99,
        lam=0.95,
        learning_rate=3e-4,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.01,
        value_loss_scale=1.0,
        kl_threshold=0.01,
        kl_estimator="analytic",
        fixed_sigma=False,
        layers=(8, 8),
        optimizer_name=optimizer_name,
        optimizer_kwargs={"compile_ns": False} if optimizer_name == "muon" else {},
        state_preprocessor="RunningStandardScaler",
        value_preprocessor="RunningStandardScaler",
        reward_preprocessor="RunningReturnScaler",
    )
    # Give every workspace a known finite pre-call value, including outputs
    # which are normally allocated with empty() before their first write.
    with torch.no_grad():
        for owner in (agent, agent.buffer):
            for value in vars(owner).values():
                if isinstance(value, torch.Tensor):
                    value.fill_(0.125 if value.is_floating_point() else 0)
    if initialized:
        _rollout(agent)
        agent.update()
    _rollout(agent)
    agent.network.register_buffer("warmup_forward_count", torch.tensor(0.0))

    def track_forward(module, args):
        del module, args
        agent.network.warmup_forward_count.add_(1)

    agent.network.trunk.register_forward_pre_hook(track_forward)
    # This generator and permutation storage normally exist on graph paths.
    agent._shuffle_generator = torch.Generator().manual_seed(29)
    for index, parameter in enumerate(agent.network.parameters()):
        parameter.grad = None if index % 2 else torch.full_like(parameter, 0.037)
    return agent


def _rollout(agent):
    with torch.no_grad():
        agent.buffer.obs.copy_(torch.randn_like(agent.buffer.obs) * 0.2)
        agent.buffer.rewards.copy_(
            torch.linspace(0.1, 1.0, agent.rollouts * agent.num_envs).reshape(
                agent.rollouts, agent.num_envs, 1
            )
        )
        agent.buffer.dones.zero_()
        for step in range(agent.rollouts):
            agent.act_into(step)
        agent._refresh_update_rewards()
        agent._refresh_epoch_permutations()


def _tensors(agent):
    values = {}
    for label, owner in (("agent", agent), ("rollout", agent.buffer)):
        for name, value in vars(owner).items():
            if isinstance(value, torch.Tensor):
                values[label + "." + name] = value
    values.update(
        {"network." + name: value for name, value in agent.network.state_dict().items()}
    )
    values.update(
        {"obs." + name: value for name, value in agent.obs_scaler.state_dict().items()}
    )
    values.update(
        {
            "value." + name: value
            for name, value in agent.value_scaler.state_dict().items()
        }
    )
    if agent.reward_preprocessor is not None:
        values.update(
            {
                "reward." + name: value
                for name, value in agent.reward_preprocessor.live_state_dict().items()
            }
        )
    for index, value in enumerate(agent._epoch_permutations or []):
        values[f"permutation.{index}"] = value
    return values


def _snapshot(agent):
    return {
        "tensors": {
            name: value.detach().clone() for name, value in _tensors(agent).items()
        },
        "optimizer": copy.deepcopy(agent.optimizer.state_dict()),
        "gradients": [
            (p.grad, None if p.grad is None else p.grad.clone())
            for p in agent.network.parameters()
        ],
        "rng": torch.get_rng_state().clone(),
        "shuffle": agent._shuffle_generator.get_state().clone(),
    }


def _equal(first, second, path="state"):
    assert type(first) is type(second), path
    if isinstance(first, torch.Tensor):
        assert first.shape == second.shape and first.dtype == second.dtype, path
        assert torch.equal(
            first.detach().reshape(-1).view(torch.uint8),
            second.detach().reshape(-1).view(torch.uint8),
        ), path
    elif isinstance(first, dict):
        assert first.keys() == second.keys(), path
        for key in first:
            _equal(first[key], second[key], path + "." + str(key))
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second), path
        for index, (a, b) in enumerate(zip(first, second, strict=True)):
            _equal(a, b, f"{path}.{index}")
    else:
        assert first == second, path


def _unchanged(agent, before):
    for name, value in _tensors(agent).items():
        _equal(value, before["tensors"][name], name)
    _equal(agent.optimizer.state_dict(), before["optimizer"], "optimizer")
    _equal(torch.get_rng_state(), before["rng"], "rng")
    _equal(agent._shuffle_generator.get_state(), before["shuffle"], "shuffle")
    for parameter, (gradient, value) in zip(
        agent.network.parameters(), before["gradients"], strict=True
    ):
        assert parameter.grad is gradient
        if gradient is not None:
            _equal(gradient, value, "gradient")


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
@pytest.mark.parametrize("initialized", [False, True])
def test_failed_actual_warmup_restores_the_entire_live_update_boundary(
    monkeypatch, optimizer_name, initialized
):
    agent = _agent(optimizer_name, initialized=initialized)
    before = _snapshot(agent)
    run = agent._run_update_epoch_loops
    failure = RuntimeError("injected host failure after actual PPO update")

    def fail(**kwargs):
        run(**kwargs)
        torch.rand(2)
        torch.rand(2, generator=agent._shuffle_generator)
        raise failure

    monkeypatch.setattr(agent, "_run_update_epoch_loops", fail)
    cuda_initialized = torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match="injected host failure") as caught:
        agent._warmup_compiled_update_minibatch()
    assert caught.value is failure
    _unchanged(agent, before)
    assert torch.cuda.is_initialized() == cuda_initialized


def _optimizer_pointers(agent):
    names = {parameter: name for name, parameter in agent.network.named_parameters()}
    return {
        (names[parameter], key): value.data_ptr()
        for parameter, state in agent.optimizer.state.items()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("max_minibatches", [1, None])
def test_successful_actual_warmup_preserves_storage_and_the_next_update(
    optimizer_name, initialized, max_minibatches
):
    agent = _agent(optimizer_name, initialized=initialized)
    for group in agent.optimizer.param_groups:
        group["lr"] = torch.tensor(float(group["lr"]))
    # Opaque wrappers model the storage ownership required by graph callers.
    wrapper = object()
    agent._optimizer_lr_wp = [wrapper]
    before = _snapshot(agent)
    references = _optimizer_pointers(agent)
    workspace_pointers = {
        name: value.data_ptr() for name, value in _tensors(agent).items()
    }
    lr_pointers = [group["lr"].data_ptr() for group in agent.optimizer.param_groups]
    # Construct the control through the real optimizer builder. Deepcopying a
    # composite optimizer is not its supported reconstruction contract.
    reference = _agent(optimizer_name, initialized=initialized)
    for group in reference.optimizer.param_groups:
        group["lr"] = torch.tensor(float(group["lr"]))
    _equal(
        agent.network.state_dict(), reference.network.state_dict(), "control network"
    )
    _equal(
        agent.optimizer.state_dict(),
        reference.optimizer.state_dict(),
        "control optimizer",
    )
    agent._warmup_compiled_update_minibatch(max_minibatches=max_minibatches)
    _unchanged(agent, before)
    assert agent._optimizer_lr_wp == [wrapper]
    assert _optimizer_pointers(agent) == references
    assert {
        name: value.data_ptr() for name, value in _tensors(agent).items()
    } == workspace_pointers
    assert [
        group["lr"].data_ptr() for group in agent.optimizer.param_groups
    ] == lr_pointers
    # Remove the test-only stateful hooks before comparing normal training.
    agent.network.trunk._forward_pre_hooks.clear()
    reference.network.trunk._forward_pre_hooks.clear()
    result = agent.update()
    torch.set_rng_state(before["rng"])
    expected = reference.update()
    _equal(result, expected, "metrics")
    _equal(agent.network.state_dict(), reference.network.state_dict(), "network")
    _equal(agent.optimizer.state_dict(), reference.optimizer.state_dict(), "optimizer")


class _DeferredStream:
    def __init__(self, *, fail_sync=False):
        self.pending = []
        self.fail_sync = fail_sync
        self.waits = 0
        self.syncs = 0

    def wait_stream(self, producer):
        del producer
        self.waits += 1

    def synchronize(self):
        self.syncs += 1
        if self.fail_sync:
            raise RuntimeError("injected synchronization failure")
        while self.pending:
            self.pending.pop(0)()


def _stream_facade(agent, monkeypatch, *, fail_sync=False):
    from contextlib import nullcontext

    stream = _DeferredStream(fail_sync=fail_sync)
    agent._use_update_graph = True
    agent._update_torch_stream = stream
    agent._update_scaled_obs = torch.zeros(
        agent.rollouts * agent.num_envs, agent.obs_dim
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    return stream


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
@pytest.mark.parametrize("initialized", [False, True])
def test_failed_actual_warmup_drains_deferred_writes_before_rollback(
    monkeypatch, optimizer_name, initialized
):
    agent = _agent(optimizer_name, initialized=initialized)
    stream = _stream_facade(agent, monkeypatch)
    before = _snapshot(agent)
    run = agent._run_update_epoch_loops
    failure = RuntimeError("host failure after enqueueing device writes")

    def fail(**kwargs):
        run(**kwargs)

        def deferred():
            with torch.no_grad():
                next(agent.network.parameters()).add_(100)
                agent.network.warmup_forward_count.add_(100)
                agent.buffer.log_probs.add_(100)
                agent._update_metrics.add_(100)

        stream.pending.append(deferred)
        raise failure

    monkeypatch.setattr(agent, "_run_update_epoch_loops", fail)
    cuda_initialized = torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match="host failure") as caught:
        agent._warmup_compiled_update_minibatch()
    assert caught.value is failure
    assert stream.waits == 1 and stream.syncs >= 1
    stream.synchronize()
    _unchanged(agent, before)
    assert torch.cuda.is_initialized() == cuda_initialized


@pytest.mark.parametrize("cleanup_failure", ["synchronize", "optimizer"])
def test_failed_cleanup_preserves_the_original_error_and_continues_rollback(
    monkeypatch, cleanup_failure
):
    agent = _agent("muon", initialized=True)
    if cleanup_failure == "synchronize":
        _stream_facade(agent, monkeypatch, fail_sync=True)
    before = _snapshot(agent)
    run = agent._run_update_epoch_loops
    failure = RuntimeError("original update failure")

    def fail(**kwargs):
        run(**kwargs)
        raise failure

    monkeypatch.setattr(agent, "_run_update_epoch_loops", fail)
    if cleanup_failure == "optimizer":

        def fail_optimizer(saved):
            del saved
            raise RuntimeError("injected optimizer rollback failure")

        monkeypatch.setattr(agent, "_restore_optimizer_state_inplace", fail_optimizer)
    with pytest.raises(RuntimeError, match="original update failure") as caught:
        agent._warmup_compiled_update_minibatch()
    assert caught.value is failure
    assert any("best-effort" in note for note in failure.__notes__)
    for name, value in _tensors(agent).items():
        _equal(value, before["tensors"][name], name)
    _equal(torch.get_rng_state(), before["rng"], "rng")
    if cleanup_failure == "synchronize":
        _equal(agent.optimizer.state_dict(), before["optimizer"], "optimizer")
    for parameter, (gradient, value) in zip(
        agent.network.parameters(), before["gradients"], strict=True
    ):
        assert parameter.grad is gradient
        if gradient is not None:
            _equal(gradient, value, "gradient")


def _stage(agent, name):
    if name == "observations":
        agent._update_scaled_obs = torch.full(
            (agent.rollouts * agent.num_envs, agent.obs_dim), 0.125
        )
        return lambda: agent._warmup_compiled_obs_scalers(
            warmup_update=True
        ), "_obs_scale_update_into_fn"
    if name == "views":
        return agent._warmup_compiled_prepare_update_views, "_prepare_update_views_fn"
    # Enter the setup phase after wrapper compilation. The callbacks retain
    # the actual eager PPO work so this unit isolates the transaction boundary.
    agent._compile_enabled = True
    agent._compile_full_update = True
    agent._compiled_update_wrappers_ready = True
    agent._compiled_model_functions_warmed = True
    # CPU agents do not build update streams, but the capture prewarm is gated
    # on the captured update path rather than on fullgraph compilation.
    agent._update_scaled_obs = torch.zeros(
        agent.rollouts * agent.num_envs, agent.obs_dim
    )
    previous_use_update_graph = agent._use_update_graph

    def invoke_prewarm():
        agent._use_update_graph = True
        try:
            return agent._prewarm_compiled_update_capture_path()
        finally:
            agent._use_update_graph = previous_use_update_graph

    return invoke_prewarm, "_run_update_epoch_loops"


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
@pytest.mark.parametrize("stage", ["observations", "views", "prewarm"])
def test_setup_stage_failure_preserves_the_live_boundary(
    monkeypatch, optimizer_name, stage
):
    agent = _agent(optimizer_name, initialized=True)
    invoke, operation_name = _stage(agent, stage)
    before = _snapshot(agent)
    operation = getattr(agent, operation_name)
    failure = RuntimeError("injected setup-stage failure")

    def fail(*args, **kwargs):
        operation(*args, **kwargs)
        raise failure

    monkeypatch.setattr(agent, operation_name, fail)
    cuda_initialized = torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match="setup-stage failure") as caught:
        invoke()
    assert caught.value is failure
    _unchanged(agent, before)
    if stage == "prewarm":
        assert not agent._compiled_update_capture_prewarmed
    assert torch.cuda.is_initialized() == cuda_initialized


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
@pytest.mark.parametrize("stage", ["observations", "views", "prewarm"])
def test_setup_stage_success_preserves_next_update_and_shuffle_budget(
    optimizer_name, stage
):
    agent = _agent(optimizer_name, initialized=True)
    invoke, _ = _stage(agent, stage)
    before = _snapshot(agent)
    entry_rng = before["rng"].clone()
    pointers = {name: value.data_ptr() for name, value in _tensors(agent).items()}
    reference = _agent(optimizer_name, initialized=True)
    _stage(reference, stage)
    torch.set_rng_state(entry_rng)
    invoke()
    _unchanged(agent, before)
    if stage == "prewarm":
        # Prewarming must not consume the authored shuffle draw: the first
        # real update starts from the permutations and RNG it entered with.
        # Once prepared, a second call must not consume another shuffle either.
        prepared = _snapshot(agent)
        invoke()
        _unchanged(agent, prepared)
    assert {
        name: value.data_ptr() for name, value in _tensors(agent).items()
    } == pointers
    agent.network.trunk._forward_pre_hooks.clear()
    reference.network.trunk._forward_pre_hooks.clear()
    result = agent.update()
    torch.set_rng_state(before["rng"])
    expected = reference.update()
    _equal(result, expected, "stage metrics")
    _equal(agent.network.state_dict(), reference.network.state_dict(), "stage network")
    _equal(
        agent.optimizer.state_dict(),
        reference.optimizer.state_dict(),
        "stage optimizer",
    )


@pytest.mark.parametrize("stage", ["observations", "views"])
def test_small_setup_failure_drains_deferred_work_before_restoration(
    monkeypatch, stage
):
    agent = _agent(initialized=True)
    invoke, operation_name = _stage(agent, stage)
    before = _snapshot(agent)
    operation = getattr(agent, operation_name)
    failure = RuntimeError("host failure with pending setup work")
    pending = []
    synchronizations = []

    def synchronize(stream=None):
        del stream
        synchronizations.append(True)
        while pending:
            pending.pop(0)()

    def fail(*args, **kwargs):
        operation(*args, **kwargs)

        def delayed():
            with torch.no_grad():
                if stage == "observations":
                    agent.obs_scaler.running_mean.add_(100)
                    agent._update_scaled_obs.add_(100)
                else:
                    agent.network.value_head.weight.add_(100)
                    agent.buffer.values.add_(100)
                    agent._update_last_values.add_(100)

        pending.append(delayed)
        raise failure

    monkeypatch.setattr(agent, "_synchronize_warmup_work", synchronize)
    monkeypatch.setattr(agent, operation_name, fail)
    cuda_initialized = torch.cuda.is_initialized()
    with pytest.raises(RuntimeError, match="pending setup work") as caught:
        invoke()
    assert caught.value is failure and synchronizations
    synchronize()
    _unchanged(agent, before)
    assert torch.cuda.is_initialized() == cuda_initialized


@pytest.mark.parametrize("stage", ["observations", "views"])
def test_small_setup_cleanup_sync_failure_preserves_the_primary_error(
    monkeypatch, stage
):
    agent = _agent(initialized=True)
    invoke, operation_name = _stage(agent, stage)
    before = _snapshot(agent)
    operation = getattr(agent, operation_name)
    failure = RuntimeError("primary setup failure")

    def fail(*args, **kwargs):
        operation(*args, **kwargs)
        raise failure

    def bad_sync(stream=None):
        del stream
        raise RuntimeError("injected setup synchronization failure")

    monkeypatch.setattr(agent, "_synchronize_warmup_work", bad_sync)
    monkeypatch.setattr(agent, operation_name, fail)
    with pytest.raises(RuntimeError, match="primary setup failure") as caught:
        invoke()
    assert caught.value is failure
    assert any("best-effort" in note for note in failure.__notes__)
    _unchanged(agent, before)


def test_prewarm_refresh_failure_restores_permutations_and_random_state(monkeypatch):
    agent = _agent("muon", initialized=True)
    invoke, _ = _stage(agent, "prewarm")
    before = _snapshot(agent)
    refresh = agent._refresh_epoch_permutations
    failure = RuntimeError("permutation refresh failed after writing")

    def fail():
        refresh()
        torch.rand(3, generator=agent._shuffle_generator)
        raise failure

    monkeypatch.setattr(agent, "_refresh_epoch_permutations", fail)
    with pytest.raises(RuntimeError, match="permutation refresh failed") as caught:
        invoke()
    assert caught.value is failure
    _unchanged(agent, before)
    assert not agent._compiled_update_capture_prewarmed
