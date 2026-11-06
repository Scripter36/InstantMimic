"""Same-policy precision and graph-compatible FP16 optimizer transactions."""

import copy
import logging

import pytest
import torch

from instantmimic.config.runtime_config import RuntimeConfig
from instantmimic.rl.ppo.agent import PPOAgent
from instantmimic.rl.ppo.loss_scaler import CapturableLossScaler


def _agent(**kwargs):
    options = dict(
        obs_dim=5,
        act_dim=3,
        num_envs=2,
        device="cpu",
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
        entropy_loss_scale=0.001,
        value_loss_scale=1.0,
        kl_threshold=0.0,
        layers=(16,),
        state_preprocessor=None,
        value_preprocessor=None,
    )
    options.update(kwargs)
    return PPOAgent(**options)


def _rollout(agent):
    agent.buffer.obs.copy_(torch.randn_like(agent.buffer.obs))
    for step in range(agent.rollouts):
        agent.act_into(step)
    agent.buffer.rewards.copy_(torch.randn_like(agent.buffer.rewards))
    agent.buffer.dones.zero_()


@pytest.mark.parametrize(
    "precision,dtype",
    [
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ],
)
@pytest.mark.parametrize("fixed_sigma", [True, False])
def test_collection_and_update_share_mlp_precision(precision, dtype, fixed_sigma):
    torch.manual_seed(17)
    agent = _agent(precision=precision, fixed_sigma=fixed_sigma)
    model = agent.network
    seen = []
    handle = model.policy_head.register_forward_hook(
        lambda module, args, output: seen.append(output.dtype)
    )
    _rollout(agent)
    obs = agent.buffer.obs[:-1].reshape(-1, 5)
    actions = agent.buffer.actions.reshape(-1, 3)
    # A conflicting ambient context must not change the configured policy.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logp, values, entropy = model.evaluate(obs, actions)
    handle.remove()
    assert seen and all(actual == dtype for actual in seen)
    torch.testing.assert_close(
        logp, agent.buffer.log_probs.reshape(-1, 1), atol=2e-5, rtol=0
    )
    assert logp.dtype == values.dtype == entropy.dtype == torch.float32
    metrics = agent.update()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert metrics["clip_fraction"] == 0.0
    assert (agent.loss_scaler is not None) == (precision == "float16")


@pytest.mark.parametrize("optimizer", ["adam", "adamw", "muon"])
@pytest.mark.parametrize("bad_gradient", [float("inf"), float("nan")])
def test_fp16_overflow_preserves_parameters_and_all_optimizer_state(
    optimizer, bad_gradient
):
    torch.manual_seed(7)
    agent = _agent(precision="float16", optimizer_name=optimizer)
    _rollout(agent)
    agent.update()
    saved_params = [p.detach().clone() for p in agent.network.parameters()]
    saved_state = {
        p: {key: value.clone() for key, value in state.items()}
        for p, state in agent.optimizer.state.items()
    }
    scale = agent.loss_scaler.scale.item()
    parameter = next(p for p in agent.network.parameters() if p.requires_grad)
    hook = parameter.register_hook(lambda grad: torch.full_like(grad, bad_gradient))
    _rollout(agent)
    agent.update()
    hook.remove()
    for p, saved in zip(agent.network.parameters(), saved_params, strict=True):
        torch.testing.assert_close(p, saved, atol=0, rtol=0)
    for p, state in saved_state.items():
        for key, saved in state.items():
            torch.testing.assert_close(
                agent.optimizer.state[p][key], saved, atol=0, rtol=0
            )
    assert agent.loss_scaler.scale.item() == scale / 2
    assert agent.loss_scaler.skipped_steps.item() == 1
    _rollout(agent)
    agent.update()
    assert any(
        not torch.equal(p, saved)
        for p, saved in zip(agent.network.parameters(), saved_params, strict=True)
    )


def test_loss_scaling_recovers_small_fp16_gradient_and_grows():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = CapturableLossScaler(device="cpu", initial_scale=128.0, growth_interval=2)
    scaler.prepare_optimizer(optimizer)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.float16):
            loss = model(torch.ones(1, 1)).float().sum() * 1e-8
        scaler.scale_loss(loss).backward()
        scaler.unscale_(optimizer)
        assert model.weight.grad.item() > 0
        scaler.step(optimizer)
    assert scaler.scale.item() == 256.0


def test_scaler_matches_unscaled_adamw_and_clipping():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = torch.optim.AdamW([parameter], lr=0.03, weight_decay=0.1, amsgrad=True)
    ref_optimizer = torch.optim.AdamW(
        [reference], lr=0.03, weight_decay=0.1, amsgrad=True, fused=True
    )
    scaler = CapturableLossScaler(device="cpu", initial_scale=128.0)
    scaler.prepare_optimizer(optimizer)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        scaler.scale_loss(parameter.square().sum()).backward()
        reference.square().sum().backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([parameter], 0.1)
        torch.nn.utils.clip_grad_norm_([reference], 0.1)
        scaler.step(optimizer)
        ref_optimizer.step()
        torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
        for key, value in optimizer.state[parameter].items():
            torch.testing.assert_close(
                value, ref_optimizer.state[reference][key], atol=0, rtol=0
            )


def test_precision_and_scaler_checkpoint_roundtrip_and_legacy_load(tmp_path):
    agent = _agent(precision="float16")
    _rollout(agent)
    agent.update()
    agent.loss_scaler.scale.fill_(32.0)
    path = tmp_path / "agent.pt"
    torch.save(copy.deepcopy(agent.state_dict()), path)
    restored = _agent(precision="float16")
    restored.load(path)
    assert restored.loss_scaler.scale.item() == 32.0
    legacy = copy.deepcopy(agent.state_dict())
    del legacy["loss_scaler"]
    torch.save(legacy, path)
    restored.load(path)
    assert restored.loss_scaler.scale.item() == 128.0


def test_update_warmup_restores_scaler_state():
    agent = _agent(precision="float16")
    _rollout(agent)
    before = copy.deepcopy(agent.loss_scaler.state_dict())
    agent._warmup_compiled_update_minibatch()
    for key, value in before.items():
        torch.testing.assert_close(
            agent.loss_scaler.state_dict()[key], value, atol=0, rtol=0
        )


@pytest.mark.parametrize("precision", ["float32", "bfloat16", "float16"])
def test_compiled_fp16_scaler_and_policy_update(precision):
    # AOTAutograd exercises the compiled update without requiring CUDA or
    # compiling a hardware-specific kernel in this CPU regression test.
    from unittest.mock import patch

    compile_fn = torch.compile

    def compile_cpu(fn, **kwargs):
        kwargs.pop("mode", None)
        return compile_fn(fn, backend="aot_eager", **kwargs)

    with patch("torch.compile", compile_cpu):
        agent = _agent(
            precision=precision,
            compile=True,
            compile_act_into=True,
        )
        for _ in range(2):
            _rollout(agent)
            result = agent.update()
        assert all(torch.isfinite(p).all() for p in agent.network.parameters())
        assert result["clip_fraction"] == 0.0


@pytest.mark.parametrize("bad", ["fp16", "float64", "bad"])
def test_precision_validation_rejects_unknown_dtype(bad):
    from pydantic import ValidationError

    from instantmimic.config.runtime_config import RuntimeConfig

    with pytest.raises(ValidationError):
        RuntimeConfig.from_mapping({"precision": bad})


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_loss_scale_validation(bad):
    from pydantic import ValidationError

    from instantmimic.config.runtime_config import RuntimeConfig

    with pytest.raises(ValidationError):
        RuntimeConfig.from_mapping({"loss_scale": bad})


def test_legacy_mixed_precision_keeps_rollout_unchanged():
    agent = _agent(mixed_precision=True)
    assert agent.precision is None
    assert agent.network.policy_mlp_dtype is None
    inference = _agent(precision="float16", inference_only=True)
    assert inference.network.policy_mlp_dtype == torch.float16
    assert inference.loss_scaler is None


def test_invalid_scaler_checkpoint_rolls_back_the_whole_load(tmp_path):
    agent = _agent(precision="float16")
    _rollout(agent)
    agent.update()
    before = copy.deepcopy(agent.state_dict())
    invalid = copy.deepcopy(before)
    for value in invalid["network"].values():
        if value.is_floating_point():
            value.add_(1.0)
    invalid["loss_scaler"]["scale"].fill_(float("nan"))
    path = tmp_path / "invalid.pt"
    torch.save(invalid, path)
    with pytest.raises(ValueError, match="loss_scaler"):
        agent.load(path)
    after = agent.state_dict()
    for key in before["network"]:
        torch.testing.assert_close(
            before["network"][key], after["network"][key], atol=0, rtol=0
        )
    for key in before["loss_scaler"]:
        torch.testing.assert_close(
            before["loss_scaler"][key], after["loss_scaler"][key], atol=0, rtol=0
        )


@pytest.mark.cuda
@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize("optimizer", ["adam", "adamw", "muon"])
def test_fp16_cuda_graph_replay_skips_overflow(detailed, optimizer):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    agent = _agent(
        device="cuda",
        precision="float16",
        optimizer_name=optimizer,
        compile=True,
        use_update_graph=True,
        detailed_nvtx=detailed,
    )
    _rollout(agent)
    agent.update()
    assert agent._has_captured_update_graphs()
    # Graph replay reads the scale tensor. Force scaled-backward overflow
    # without changing the captured graph, then compare all trainable state.
    parameters = [p for p in agent.network.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in parameters]
    states = {
        p: {k: v.clone() for k, v in state.items()}
        for p, state in agent.optimizer.state.items()
    }
    skipped = int(agent.loss_scaler.skipped_steps.item())
    agent.loss_scaler.scale.fill_(2.0**100)
    _rollout(agent)
    agent.update()
    assert agent.loss_scaler.skipped_steps.item() == skipped + 1
    for p, saved in zip(parameters, before, strict=True):
        torch.testing.assert_close(p, saved, atol=0, rtol=0)
    for p, state in states.items():
        for key, saved in state.items():
            torch.testing.assert_close(
                agent.optimizer.state[p][key], saved, atol=0, rtol=0
            )


@pytest.mark.parametrize("optimizer", ["adam", "adamw", "muon"])
def test_compiled_scaler_reads_changed_overflow_flag(optimizer):
    agent = _agent(precision="float16", optimizer_name=optimizer)
    scaler = agent.loss_scaler
    unscale = torch.compile(scaler.unscale_, backend="aot_eager", fullgraph=True)
    step = torch.compile(scaler.step, backend="aot_eager")
    params = [p for p in agent.network.parameters() if p.requires_grad]
    for overflow in (False, True, False):
        agent.optimizer.zero_grad(set_to_none=True)
        for p in params:
            p.grad = torch.full_like(p, 1.0 if not overflow else float("inf"))
        before = [p.detach().clone() for p in params]
        states = {
            p: {k: v.clone() for k, v in state.items()}
            for p, state in agent.optimizer.state.items()
        }
        unscale(agent.optimizer)
        agent._clip_grad_norm_inplace()
        step(agent.optimizer)
        if overflow:
            for p, saved in zip(params, before, strict=True):
                torch.testing.assert_close(p, saved, atol=0, rtol=0)
            for p, state in states.items():
                for key, saved in state.items():
                    torch.testing.assert_close(
                        agent.optimizer.state[p][key], saved, atol=0, rtol=0
                    )
        else:
            assert any(
                not torch.equal(p, saved)
                for p, saved in zip(params, before, strict=True)
            )
    assert scaler.skipped_steps.item() == 1


@pytest.mark.cuda
def test_fp16_cuda_graph_can_resume_a_cpu_optimizer_checkpoint(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = _agent(precision="float16")
    _rollout(source)
    source.update()
    path = tmp_path / "cpu_checkpoint.pt"
    payload = copy.deepcopy(source.state_dict())
    target = _agent(
        device="cuda",
        precision="float16",
        use_update_graph=True,
        compile=True,
    )
    # CPU uses Torch noise; CUDA uses Warp noise. The full-resume contract
    # requires an explicit initial Warp seed when crossing that RNG backend.
    payload["policy_noise_seed"] = target.state_dict()["policy_noise_seed"].cpu()
    torch.save(payload, path)
    target.load(path)
    for group in target.optimizer.param_groups:
        assert group["capturable"]
        assert group["lr"].device.type == "cuda"
        for parameter in group["params"]:
            assert target.optimizer.state[parameter]["step"].device.type == "cuda"
    _rollout(target)
    target.update()
    assert target._has_captured_update_graphs()


@pytest.mark.parametrize("optimizer", ["adam", "adamw", "muon"])
def test_fp16_optimizer_step_does_not_clone_parameter_or_optimizer_state(optimizer):
    from torch.utils._python_dispatch import TorchDispatchMode

    agent = _agent(precision="float16", optimizer_name=optimizer)
    for parameter in agent.network.parameters():
        parameter.grad = torch.ones_like(parameter)
    live = {
        tensor.data_ptr()
        for group in agent.optimizer.param_groups
        for parameter in group["params"]
        for tensor in (parameter, *agent.optimizer.state[parameter].values())
    }

    class ForbidStateSnapshot(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.clone.default:
                assert args[0].data_ptr() not in live, (
                    "Optimizer took a live-state backup"
                )
            return func(*args, **(kwargs or {}))

    with ForbidStateSnapshot():
        agent.loss_scaler.step(agent.optimizer)


@pytest.mark.parametrize("optimizer_cls", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("amsgrad,maximize", [(False, False), (True, True)])
def test_native_overflow_skip_matches_finite_adam_and_preserves_state(
    optimizer_cls, amsgrad, maximize
):
    from instantmimic.rl.optimizers import optimizer_step_if_finite

    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [-1.0, -0.5]]))
    reference = torch.nn.Parameter(parameter.detach().clone())
    options = dict(
        lr=0.01, weight_decay=0.1, amsgrad=amsgrad, maximize=maximize, fused=True
    )
    optimizer = optimizer_cls([parameter], **options)
    control = optimizer_cls([reference], **options)
    scaler = CapturableLossScaler(device="cpu")
    scaler.prepare_optimizer(optimizer)
    for skip in (False, True, False):
        parameter.grad = torch.full_like(parameter, 0.3)
        reference.grad = torch.full_like(reference, 0.3)
        optimizer_step_if_finite(optimizer, torch.tensor(float(skip)))
        if not skip:
            control.step()
        torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
        for key, value in optimizer.state[parameter].items():
            torch.testing.assert_close(
                value, control.state[reference][key], atol=0, rtol=0
            )


@pytest.mark.parametrize(
    "batched,nesterov", [(False, False), (False, True), (True, False), (True, True)]
)
def test_muon_masked_step_matches_original_and_skips_decay_and_momentum(
    batched, nesterov
):
    from instantmimic.rl.optimizers import CapturableMuon

    torch.manual_seed(4)
    parameters = [torch.nn.Parameter(torch.randn(8, 8)) for _ in range(2)]
    reference = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    options = dict(
        lr=0.01,
        weight_decay=0.1,
        momentum=0.9,
        nesterov=nesterov,
        batched_updates=batched,
        compile_ns=False,
    )
    optimizer = CapturableMuon(parameters, **options)
    control = CapturableMuon(reference, **options)
    for skip in (False, True, False):
        for p, q in zip(parameters, reference, strict=True):
            grad = torch.randn_like(p)
            p.grad, q.grad = grad, grad.clone()
        optimizer.step(found_inf=torch.tensor(float(skip)))
        if not skip:
            control.step()
        for p, q in zip(parameters, reference, strict=True):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
            torch.testing.assert_close(
                optimizer.state[p]["momentum_buffer"],
                control.state[q]["momentum_buffer"],
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("enabled", [False, True])
def test_deprecated_runtime_key_warns_and_preserves_authored_value(enabled):
    with pytest.warns(FutureWarning, match="mixed_precision.*deprecated"):
        runtime = RuntimeConfig.from_mapping({"mixed_precision": enabled})
    assert runtime.mixed_precision is enabled
    assert runtime.ppo_precision is None
    assert dict(runtime) == {"mixed_precision": enabled}


def test_default_runtime_does_not_opt_in_to_shared_precision(recwarn):
    runtime = RuntimeConfig.from_mapping({})
    assert runtime.ppo_precision is None
    assert not recwarn


@pytest.mark.parametrize("precision", ["float32", "bfloat16", "float16"])
def test_explicit_runtime_precision_overrides_deprecated_key(precision):
    with pytest.warns(FutureWarning, match="mixed_precision.*deprecated"):
        runtime = RuntimeConfig.from_mapping(
            {"mixed_precision": True, "precision": precision}
        )
    assert runtime.ppo_precision == precision


@pytest.mark.parametrize("inference_only", [False, True])
def test_legacy_cpu_and_playback_keep_fp32_policy(inference_only, caplog):
    with caplog.at_level(logging.WARNING):
        agent = _agent(mixed_precision=True, inference_only=inference_only)
    assert agent.precision is None
    assert agent.network.policy_mlp_dtype is None
    assert agent.loss_scaler is None
    assert not agent._use_mixed_precision
    assert "deprecated" in caplog.text
    obs = torch.randn(2, 5)
    seen = []
    handle = agent.network.trunk.register_forward_hook(
        lambda m, a, out: seen.append(out.dtype)
    )
    with torch.no_grad():
        agent.network.policy_mean(obs)
    handle.remove()
    assert seen == [torch.float32]


def test_legacy_update_context_does_not_force_rollout_precision(monkeypatch):
    # Exercise the legacy CUDA context on CPU without touching a CUDA device.
    # The old policy did ordinary FP32 collection and ambient autocast evaluation.
    agent = _agent(mixed_precision=True)
    assert agent.precision is None
    agent._use_mixed_precision = True
    original_autocast = torch.amp.autocast
    requested = []

    def cpu_context(*, device_type, dtype):
        requested.append((device_type, dtype))
        return original_autocast(device_type="cpu", dtype=dtype)

    monkeypatch.setattr(torch.amp, "autocast", cpu_context)
    seen = []
    handle = agent.network.trunk.register_forward_hook(
        lambda m, a, out: seen.append(out.dtype)
    )
    _rollout(agent)
    assert seen == [torch.float32] * agent.rollouts
    seen.clear()
    with agent._update_autocast_ctx():
        agent.network.evaluate(torch.randn(2, 5), torch.randn(2, 3))
    handle.remove()
    assert requested == [("cuda", torch.bfloat16)]
    assert seen == [torch.bfloat16]


def test_legacy_fp16_api_does_not_silently_enable_scaling(caplog):
    with caplog.at_level(logging.WARNING):
        agent = _agent(mixed_precision=True, amp_dtype="float16")
    assert agent.precision is None
    assert agent.loss_scaler is None
    assert "loss scaling" in caplog.text


def test_explicit_float32_takes_the_plain_path_outside_autocast():
    network = _agent(precision="float32").network
    # Same compiled graph as a network without a precision setting.
    assert network._policy_mlp_dtype_in_effect("cpu") is None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert network._policy_mlp_dtype_in_effect("cpu") == torch.float32


@pytest.mark.parametrize("precision", ["float32", "bfloat16", "float16"])
def test_explicit_precision_ignores_legacy_update_autocast(precision):
    agent = _agent(precision=precision, mixed_precision=True)
    # A stale legacy flag must not enable an outer update-only autocast.
    agent._use_mixed_precision = True
    with agent._update_autocast_ctx():
        value = torch.nn.Linear(5, 3)(torch.randn(2, 5))
    assert value.dtype == torch.float32
    assert agent.network.policy_mlp_dtype == getattr(torch, precision)
