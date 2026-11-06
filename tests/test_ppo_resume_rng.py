"""Full PPO resume preserves random continuation without affecting policy loads."""

from __future__ import annotations

import copy

import pytest
import torch

from instantmimic.rl.ppo.agent import PPOAgent
from instantmimic.train.checkpointing import _load_agent_checkpoint


def _agent(
    *, shuffle="per_epoch", inference_only=False, device="cpu", layers=(4,), **kwargs
):
    return PPOAgent(
        obs_dim=4,
        act_dim=2,
        num_envs=2,
        device=device,
        rollouts=4,
        learning_epochs=2,
        mini_batches=2,
        discount_factor=0.99,
        lam=0.95,
        learning_rate=3e-4,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        clip_predicted_values=True,
        entropy_loss_scale=0.0,
        value_loss_scale=1.0,
        kl_threshold=0.0,
        layers=layers,
        state_preprocessor=None,
        value_preprocessor=None,
        shuffle_mode=shuffle,
        inference_only=inference_only,
        **kwargs,
    )


def _random_continuation(agent):
    noise = agent._fill_policy_noise_().clone()
    agent._refresh_epoch_permutations()
    return noise, [p.clone() for p in agent._epoch_permutations or []]


@pytest.mark.parametrize("shuffle", ["off", "per_update", "per_epoch"])
def test_full_resume_restores_eager_policy_and_minibatch_randomness(tmp_path, shuffle):
    torch.manual_seed(42)
    source = _agent(shuffle=shuffle)
    for _ in range(5):
        _random_continuation(source)
    checkpoint = tmp_path / "agent.pt"
    torch.save(source.state_dict(), checkpoint)
    expected_noise, expected_permutations = _random_continuation(source)
    torch.manual_seed(9)
    resumed = _agent(shuffle=shuffle)
    resumed.load(checkpoint)
    noise, permutations = _random_continuation(resumed)
    torch.testing.assert_close(noise, expected_noise, rtol=0, atol=0)
    assert len(permutations) == len(expected_permutations)
    for actual, expected in zip(permutations, expected_permutations, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["inference", "policy_init", "legacy_resume"])
def test_policy_loads_and_legacy_resume_keep_the_callers_random_state(tmp_path, mode):
    source = _agent()
    payload = copy.deepcopy(source.state_dict())
    if mode == "legacy_resume":
        payload.pop("torch_rng_state", None)
        payload.pop("torch_cuda_rng_state", None)
    checkpoint = tmp_path / "agent.pt"
    torch.save(payload, checkpoint)
    target = _agent(inference_only=mode == "inference")
    torch.manual_seed(219)
    before = torch.get_rng_state().clone()
    if mode == "policy_init":
        _load_agent_checkpoint(target, checkpoint, checkpoint_load_mode="policy_init")
    else:
        target.load(checkpoint)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_invalid_resume_rng_rolls_back_parameters_and_random_state(tmp_path):
    source = _agent()
    payload = copy.deepcopy(source.state_dict())
    payload["torch_rng_state"] = torch.zeros(1, dtype=torch.uint8)
    checkpoint = tmp_path / "invalid_rng.pt"
    torch.save(payload, checkpoint)
    target = _agent()
    before_parameters = copy.deepcopy(target.network.state_dict())
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="RNG state"):
        target.load(checkpoint)
    for name, parameter in target.network.state_dict().items():
        torch.testing.assert_close(parameter, before_parameters[name], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)


def test_rng_snapshot_restores_only_the_agents_cuda_device(monkeypatch):
    from types import SimpleNamespace

    cuda_state = torch.Generator().manual_seed(81).get_state()
    calls = []
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: calls.append(("get", device)) or cuda_state.clone(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device: calls.append(("set", device, state.clone())),
    )
    cpu = SimpleNamespace(device=torch.device("cpu"))
    assert set(PPOAgent._checkpoint_rng_state(cpu)) == {"torch_rng_state"}
    assert calls == []
    device = torch.device("cuda:2")
    snapshot = PPOAgent._checkpoint_rng_state(SimpleNamespace(device=device))
    assert set(snapshot) == {"torch_rng_state", "torch_cuda_rng_state"}
    PPOAgent._restore_checkpoint_rng_state(SimpleNamespace(device=device), snapshot)
    assert calls[0] == ("get", device)
    assert calls[1][:2] == ("set", device)
    torch.testing.assert_close(calls[1][2], cuda_state, rtol=0, atol=0)


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("optimizer", ["adam", "adamw"])
@pytest.mark.parametrize(
    "source_backend", [(False, False), (True, False), (False, True)]
)
def test_full_resume_restores_active_graph_optimizer_contract(
    tmp_path, monkeypatch, precision, populated, optimizer, source_backend
):
    source_foreach, source_fused = source_backend
    source = _agent(
        precision=precision,
        optimizer_name=optimizer,
        optimizer_kwargs={"foreach": source_foreach, "fused": source_fused},
    )
    if populated:
        sum(p.square().sum() for p in source.network.parameters()).backward()
        source.optimizer.step()
    payload = copy.deepcopy(source.state_dict())
    # Legacy full payloads must also receive the active runtime's capture contract.
    payload.pop("torch_rng_state", None)
    payload.pop("torch_cuda_rng_state", None)
    checkpoint = tmp_path / "eager.pt"
    torch.save(payload, checkpoint)
    target = _agent(
        precision=precision,
        optimizer_name=optimizer,
        optimizer_kwargs={"foreach": False, "fused": False},
    )
    # Model the constructor's graph metadata with CPU tensors. Only the Warp
    # wrapper boundary is mocked; optimizer loading/materialization is real.
    target._use_update_graph = True
    target._optimizer_graph_state_initialized = True
    for group in target.optimizer.param_groups:
        group["capturable"] = True
        group["lr"] = torch.as_tensor(group["lr"])
    monkeypatch.setattr(
        "instantmimic.rl.ppo.agent.wp.from_torch", lambda *a, **k: object()
    )
    target.load(checkpoint)
    for group in target.optimizer.param_groups:
        assert group["capturable"]
        assert group["foreach"] is False
        assert group["fused"] == source_fused
        assert isinstance(group["lr"], torch.Tensor)
        assert group["lr"].dtype == torch.float32
        for parameter in group["params"]:
            state = target.optimizer.state[parameter]
            assert state["step"].device == parameter.device
            assert state["step"].item() == (1 if populated else 0)
    assert all(wrapper is not None for wrapper in target._optimizer_lr_wp)
    for old_parameter, new_parameter in zip(
        source.network.parameters(), target.network.parameters(), strict=True
    ):
        state = target.optimizer.state[new_parameter]
        for name in ("exp_avg", "exp_avg_sq"):
            expected = (
                source.optimizer.state[old_parameter][name]
                if populated
                else torch.zeros_like(new_parameter)
            )
            torch.testing.assert_close(state[name], expected, rtol=0, atol=0)


def test_capture_preparation_normalizes_the_hybrid_adam_backend_without_resetting_state():
    from instantmimic.rl.optimizers import HybridMuonAdamW, prepare_optimizer_state

    matrix = torch.nn.Parameter(torch.ones(2, 2))
    vector = torch.nn.Parameter(torch.ones(2))
    optimizer = HybridMuonAdamW(
        muon_params=[matrix],
        adamw_params=[vector],
        muon_kwargs={"lr": 0.01, "compile_ns": False},
        adamw_kwargs={"lr": 0.01, "foreach": True},
    )
    matrix.grad = torch.full_like(matrix, 0.1)
    vector.grad = torch.full_like(vector, 0.2)
    optimizer.step()
    before = {
        parameter: {name: value.clone() for name, value in state.items()}
        for parameter, state in optimizer.state.items()
    }
    prepare_optimizer_state(optimizer, capturable=True)
    group = optimizer.adamw_optimizer.param_groups[0]
    assert group["capturable"] is True
    assert group["foreach"] is False
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            torch.testing.assert_close(value, before[parameter][name], rtol=0, atol=0)


def test_capture_preparation_rejects_differentiable_optimizer_semantics():
    from instantmimic.rl.optimizers import prepare_optimizer_state

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter], differentiable=True)
    with pytest.raises(ValueError, match="does not support differentiable optimizers"):
        prepare_optimizer_state(optimizer, capturable=True)
    assert optimizer.param_groups[0]["differentiable"] is True
    assert optimizer.state == {}


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
@pytest.mark.parametrize("populated", [False, True])
def test_cpu_checkpoint_resumes_capture_and_two_gpu_replays(
    tmp_path, precision, optimizer, populated
):
    import math

    source_kwargs = (
        {"compile_ns": False, "adamw_foreach": True}
        if optimizer == "muon"
        else {"foreach": True}
    )
    source = _agent(
        precision=precision,
        layers=(8, 8),
        optimizer_name=optimizer,
        optimizer_kwargs=source_kwargs,
    )
    if populated:
        sum(p.square().sum() for p in source.network.parameters()).backward()
        source.optimizer.step()
    payload = copy.deepcopy(source.state_dict())
    target = _agent(
        device="cuda",
        precision=precision,
        layers=(8, 8),
        optimizer_name=optimizer,
        optimizer_kwargs=(
            {"compile_ns": False, "adamw_foreach": False}
            if optimizer == "muon"
            else {"foreach": False}
        ),
        use_update_graph=True,
        use_shuffle_graph=True,
        compile=True,
    )
    # CPU and CUDA use different policy/shuffle RNG backends. Initialize those
    # new CUDA streams explicitly; this test isolates optimizer continuation.
    target_payload = target.state_dict()
    for name in ("policy_noise_seed", "shuffle_generator_state"):
        payload[name] = target_payload[name].cpu()
    checkpoint = tmp_path / "cpu_eager.pt"
    torch.save(payload, checkpoint)
    target.load(checkpoint)

    for group in target.optimizer.param_groups:
        assert isinstance(group["lr"], torch.Tensor)
        assert group["lr"].device.type == "cuda"
        if "capturable" in group:
            assert group["capturable"] is True
            assert group["foreach"] is False
        for parameter in group["params"]:
            state = target.optimizer.state[parameter]
            assert all(value.device == parameter.device for value in state.values())
            if "step" in state:
                assert state["step"].item() == (1 if populated else 0)
    if populated:
        for old_parameter, parameter in zip(
            source.network.parameters(), target.network.parameters(), strict=True
        ):
            for name, value in source.optimizer.state[old_parameter].items():
                torch.testing.assert_close(
                    target.optimizer.state[parameter][name].cpu(), value, rtol=0, atol=0
                )

    for update_index in range(3):
        with torch.no_grad():
            target.buffer.obs.copy_(torch.randn_like(target.buffer.obs) * 0.1)
            target.buffer.rewards.copy_(
                torch.linspace(
                    0.1, 1.0, target.rollouts * target.num_envs, device="cuda"
                ).reshape(target.rollouts, target.num_envs, 1)
            )
            target.buffer.dones.zero_()
            for step in range(target.rollouts):
                target.act_into(step)
        metrics = target.update()
        assert target._has_captured_update_graphs()
        assert all(math.isfinite(float(value)) for value in metrics.values())
        expected_step = (1 if populated else 0) + (
            update_index + 1
        ) * target.learning_epochs * target.mini_batches
        for state in target.optimizer.state.values():
            if "step" in state:
                assert state["step"].item() == expected_step
            assert all(torch.isfinite(value).all().item() for value in state.values())
        assert all(torch.isfinite(p).all().item() for p in target.network.parameters())
