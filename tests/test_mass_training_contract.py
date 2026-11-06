"""Observable contracts of the environment-owned MASS optimizer and storage."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from instantmimic.envs.controls.fast_muscle.muscle_nn import (
    MassMuscleNN,
    MuscleNNInputNormalizer,
)
from instantmimic.envs.controls.fast_muscle.settings import FastMuscleControlSettings
from instantmimic.envs.controls.fast_muscle.training_runtime import (
    FastMuscleTrainingMixin,
)


class Runtime(FastMuscleTrainingMixin):
    pass


def runtime(*, pose=False, passive=False, output=None):
    obj = Runtime()
    policy = (
        {
            "type": "pose",
            "scale": 0.04,
            "reference": "rest",
            "stable_pd": {"mjcf_path": "data/synthetic.xml"},
        }
        if pose
        else {"type": "torque", "scale": 1.0}
    )
    obj.settings = FastMuscleControlSettings.from_cfg(
        {
            "action": {
                "type": "mass",
                "space": 2,
                "policy_output": policy,
                "muscle_nn": {
                    "target_includes_passive_force": passive,
                    "hidden_dims": [4],
                    "output_activation": output or ("tanh_relu" if pose else "sine"),
                    "input_scale": 2.0,
                    "target_scale": 1.7,
                    "target_loss_weight": 1.3,
                    "regularization_weight": 0.2,
                    "bound_loss_weight": 0.1,
                    "grad_clip": 0.4,
                },
            }
        }
    )
    obj.device = torch.device("cpu")
    obj._uses_mass_coordinator = True
    obj._uses_mass_pose_coordinator = pose
    obj._musclenn_use_update_graph = False
    obj._musclenn_input_scale_inv = 0.5
    obj._musclenn_sparse_entry_dim = 4
    obj._musclenn_entry_torque_idx = torch.tensor([0, 0, 1, 1])
    obj._musclenn_entry_muscle_idx = torch.tensor([0, 1, 1, 2])
    obj._musclenn_torque_scale_inv = torch.tensor([0.5, 1.5])
    matrix = torch.tensor([[0.7, -0.4, 0.0], [0.0, 0.3, 0.9]])
    obj._musclenn_torque_helper = SimpleNamespace(
        include_passive_force_in_output=passive,
        torque_from_activation=lambda activations, context: activations @ matrix.T
        + context,
    )
    obj._musclenn_model = MassMuscleNN(
        6, 3, (4,), output_activation=obj._mass_muscle_nn().output_activation
    )
    obj._musclenn_optimizer = torch.optim.Adam(
        obj._musclenn_model.parameters(), lr=0.003
    )
    return obj, matrix


@pytest.mark.parametrize("pose", [False, True])
@pytest.mark.parametrize("passive", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("output", [None, "identity"])
def test_optimizer_matches_dense_physical_torque_objective(
    pose, passive, weighted, output
):
    torch.manual_seed(42)
    obj, matrix = runtime(pose=pose, passive=passive, output=output)
    model = copy.deepcopy(obj._musclenn_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    target = torch.tensor([[0.2, -0.1], [-0.3, 0.4], [0.8, -0.2]])
    inputs = torch.cat((torch.tensor([[0.7, -0.4, 0.3, 0.9]]).expand(3, -1), target), 1)
    context = torch.tensor([[0.1, -0.2], [0.4, -0.3], [-0.1, 0.2]]) if passive else None
    weights = torch.tensor([[1.0], [0.0], [1.0]]) if weighted else torch.ones(3, 1)
    activation, raw = model.forward_with_preact(inputs * 0.5)
    predicted = activation @ matrix.T
    if passive:
        predicted = predicted + context
        if not pose:
            predicted = predicted * torch.tensor([0.5, 1.5])
    denominator = weights.sum()
    target_loss = (((predicted - target) / 1.7).square() * weights).sum() / (
        denominator * 2
    )
    regularizer = activation.square()
    if pose:
        regularizer = 0.5 * (regularizer + raw.square())
    reg_loss = (regularizer * weights).sum() / (denominator * 3)
    bound_loss = (
        (torch.relu(-activation).square() + torch.relu(activation - 1).square())
        * weights
    ).sum() / (denominator * 3)
    loss = 1.3 * target_loss + 0.2 * reg_loss + 0.1 * bound_loss
    loss.backward()
    gradients = [p.grad for p in model.parameters()]
    norm = torch.nn.utils.get_total_norm(gradients)
    if pose:
        clipped = torch.stack([g.abs().amax() for g in gradients]).amax() > 0.4
        for grad in gradients:
            grad.clamp_(-0.4, 0.4)
    else:
        clipped = norm + 1e-6 > 0.4
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)
    optimizer.step()
    actual = obj._run_musclenn_minibatch(
        inputs,
        target,
        context,
        graph_safe=False,
        sample_weight=weights if weighted else None,
    )
    for value, expected in zip(
        actual,
        (loss, target_loss, reg_loss, bound_loss, norm, clipped.float()),
        strict=True,
    ):
        torch.testing.assert_close(value, expected, rtol=2e-5, atol=2e-6)
    for value, expected in zip(
        obj._musclenn_model.parameters(), model.parameters(), strict=True
    ):
        torch.testing.assert_close(value, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("with_context", [False, True])
def test_ring_buffer_retains_chronological_rows_after_wrap_and_large_append(
    with_context,
):
    obj = Runtime()
    obj._musclenn_input_buffer = torch.zeros(5, 2)
    obj._musclenn_target_buffer = torch.zeros(5, 1)
    obj._musclenn_context_buffer = (
        torch.zeros(5, 3) if with_context else torch.empty(0, 0)
    )
    obj._musclenn_buffer_ptr = obj._musclenn_buffer_filled = 0
    obj._musclenn_rollout_dirty = False
    history = torch.empty(0, 2)
    for start, count in [(0, 3), (3, 4), (7, 8)]:
        values = (
            torch.arange(start, start + count).float().view(-1, 1).expand(-1, 2).clone()
        )
        targets = values[:, :1] * 3
        context = values[:, :1].expand(-1, 3) * 2 if with_context else None
        obj._append_musclenn_ring_buffer(values, targets, context)
        history = torch.cat((history, values))[-5:]
        actual, target, raw = obj._get_musclenn_buffer_view()
        torch.testing.assert_close(actual, history)
        torch.testing.assert_close(target, history[:, :1] * 3)
        if with_context:
            torch.testing.assert_close(raw, history[:, :1].expand(-1, 3) * 2)
        else:
            assert raw is None


def test_input_statistics_merge_scaled_batches_once_per_rollout():
    obj, _ = runtime()
    obj._musclenn_model.input_normalizer = MuscleNNInputNormalizer(
        6, clip=5, min_std=0.001
    )
    obj._initialize_musclenn_input_statistics(6)
    first = torch.arange(18).float().reshape(3, 6)
    second = torch.arange(12).float().reshape(2, 6) * 3 - 7
    obj._observe_musclenn_rollout_inputs(first)
    obj._observe_musclenn_rollout_inputs(second)
    normalizer = obj._musclenn_input_normalizer()
    expected = torch.cat((first, second)).double() * 0.5
    assert normalizer.count == 5
    torch.testing.assert_close(normalizer.mean, expected.mean(0), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        normalizer.variance, expected.var(0, unbiased=False), rtol=1e-6, atol=1e-6
    )


class SetupStream:
    """CPU façade for delayed stream writes; it does not validate CUDA kernels."""

    def __init__(self, error=None):
        self.pending = []
        self.error = error

    def synchronize(self):
        while self.pending:
            self.pending.pop(0)()
        if self.error is not None:
            error, self.error = self.error, None
            raise error


def _setup_capture_runtime(monkeypatch, populated, failure):
    from contextlib import nullcontext

    obj, _ = runtime()
    obj._musclenn_model.register_buffer("setup_count", torch.zeros(()))
    inputs = torch.ones(4, 6)
    if populated:
        obj._musclenn_model(inputs).square().mean().backward()
        obj._musclenn_optimizer.step()
    obj._musclenn_optimizer.zero_grad(set_to_none=False)
    for index, parameter in enumerate(obj._musclenn_model.parameters()):
        parameter.grad = torch.full_like(parameter, 0.2) if index % 2 else None
    error = RuntimeError("injected setup failure")
    stream = SetupStream(error if failure == "synchronize" else None)
    obj._musclenn_update_torch_stream = stream
    obj._musclenn_compile_full_update = False
    obj._musclenn_compiled_minibatch_ready = False
    obj._musclenn_graph_inputs = inputs
    obj._musclenn_input_buffer = inputs
    obj._musclenn_graph_targets = torch.zeros(4, 2)
    obj._musclenn_graph_contexts = torch.empty(0, 0)
    obj._musclenn_graph_valid_mask = torch.ones(4, 1)
    obj._musclenn_graph_shuffled_inputs = torch.empty_like(inputs)
    obj._musclenn_graph_shuffled_targets = torch.empty(4, 2)
    obj._musclenn_graph_shuffled_contexts = torch.empty(0, 0)
    obj._musclenn_graph_shuffled_valid_mask = torch.empty(4, 1)
    obj._musclenn_graph_permutations = torch.arange(4).view(1, -1)
    obj._musclenn_graph_metrics = torch.zeros(6)
    obj._musclenn_update_graph = None

    def step(*args):
        del args
        obj._musclenn_optimizer.zero_grad(set_to_none=False)
        obj._musclenn_model(inputs).square().mean().backward()
        obj._musclenn_optimizer.step()
        obj._musclenn_model.setup_count.add_(1)
        if failure == "host":

            def delayed_write():
                with torch.no_grad():
                    next(obj._musclenn_model.parameters()).add_(10)

            stream.pending.append(delayed_write)
            raise error

    obj._run_musclenn_graph_minibatch_fn = step
    obj._run_musclenn_graph_epoch_loops = step
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args: nullcontext())
    return obj, stream, error


def _assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("method", ["warmup", "capture"])
@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("failure", ["host", "synchronize"])
def test_setup_failure_restores_state_after_draining_pending_work(
    monkeypatch, method, populated, failure
):
    initialized = torch.cuda.is_initialized()
    obj, stream, error = _setup_capture_runtime(monkeypatch, populated, failure)
    before_model = copy.deepcopy(obj._musclenn_model.state_dict())
    before_optimizer = copy.deepcopy(obj._musclenn_optimizer.state_dict())
    gradients = [
        (parameter.grad, None if parameter.grad is None else parameter.grad.clone())
        for parameter in obj._musclenn_model.parameters()
    ]
    if method == "capture":
        obj._compile_and_warm_musclenn_graph_minibatch = lambda: None
        invoke = obj._capture_musclenn_update_graph
    else:
        invoke = obj._compile_and_warm_musclenn_graph_minibatch
    with pytest.raises(RuntimeError, match="injected setup failure") as raised:
        invoke()
    assert raised.value is error
    stream.synchronize()
    _assert_state_equal(obj._musclenn_model.state_dict(), before_model)
    _assert_state_equal(obj._musclenn_optimizer.state_dict(), before_optimizer)
    for parameter, (gradient, value) in zip(
        obj._musclenn_model.parameters(), gradients, strict=True
    ):
        assert parameter.grad is gradient
        if value is not None:
            torch.testing.assert_close(gradient, value, rtol=0, atol=0)
    assert obj._musclenn_update_graph is None
    assert not obj._musclenn_compiled_minibatch_ready
    assert torch.cuda.is_initialized() == initialized


@pytest.mark.parametrize("method", ["warmup", "capture"])
@pytest.mark.parametrize("cleanup_failure", ["synchronize", "optimizer_restore"])
def test_setup_cleanup_preserves_original_exception_and_continues_restoration(
    monkeypatch, method, cleanup_failure
):
    obj, stream, error = _setup_capture_runtime(monkeypatch, True, "host")
    before = copy.deepcopy(obj._musclenn_model.state_dict())
    before_optimizer = copy.deepcopy(obj._musclenn_optimizer.state_dict())
    gradients = [parameter.grad for parameter in obj._musclenn_model.parameters()]
    cleanup_error = RuntimeError("injected cleanup failure")
    if cleanup_failure == "synchronize":
        stream.error = cleanup_error
    else:

        def fail_restore(state):
            del state
            raise cleanup_error

        monkeypatch.setattr(obj._musclenn_optimizer, "load_state_dict", fail_restore)
    if method == "capture":
        obj._compile_and_warm_musclenn_graph_minibatch = lambda: None
        invoke = obj._capture_musclenn_update_graph
    else:
        invoke = obj._compile_and_warm_musclenn_graph_minibatch
    with pytest.raises(RuntimeError, match="injected setup failure") as raised:
        invoke()
    assert raised.value is error
    assert any(
        "injected cleanup failure" in note and "best-effort" in note
        for note in error.__notes__
    )
    _assert_state_equal(obj._musclenn_model.state_dict(), before)
    if cleanup_failure == "synchronize":
        _assert_state_equal(obj._musclenn_optimizer.state_dict(), before_optimizer)
    for parameter, gradient in zip(
        obj._musclenn_model.parameters(), gradients, strict=True
    ):
        assert parameter.grad is gradient


@pytest.mark.parametrize("method", ["warmup", "capture"])
@pytest.mark.parametrize("populated", [False, True])
def test_successful_setup_preserves_storage_and_the_next_optimizer_step(
    monkeypatch, method, populated
):
    obj, _, _ = _setup_capture_runtime(monkeypatch, populated, None)
    model_before = copy.deepcopy(obj._musclenn_model.state_dict())
    reference = copy.deepcopy(obj._musclenn_model)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.003)
    reference_optimizer.load_state_dict(
        copy.deepcopy(obj._musclenn_optimizer.state_dict())
    )
    captured_pointers = []

    def pointers():
        return (
            [tensor.data_ptr() for tensor in obj._musclenn_model.parameters()]
            + [
                parameter.grad.data_ptr()
                for parameter in obj._musclenn_model.parameters()
            ]
            + [
                value.data_ptr()
                for state in obj._musclenn_optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            ]
        )

    original_step = obj._run_musclenn_graph_minibatch_fn

    def recording_step(*args):
        original_step(*args)
        captured_pointers[:] = pointers()

    obj._run_musclenn_graph_minibatch_fn = recording_step
    obj._run_musclenn_graph_epoch_loops = recording_step
    if method == "capture":
        obj._compile_and_warm_musclenn_graph_minibatch = lambda: None
        obj._capture_musclenn_update_graph()
        assert obj._musclenn_update_graph is not None
    else:
        obj._compile_and_warm_musclenn_graph_minibatch()
        assert obj._musclenn_compiled_minibatch_ready
    _assert_state_equal(obj._musclenn_model.state_dict(), model_before)
    assert pointers() == captured_pointers
    for model, optimizer in [
        (obj._musclenn_model, obj._musclenn_optimizer),
        (reference, reference_optimizer),
    ]:
        optimizer.zero_grad(set_to_none=False)
        model(torch.ones(4, 6)).square().mean().backward()
        optimizer.step()
    _assert_state_equal(obj._musclenn_model.state_dict(), reference.state_dict())
    _assert_state_equal(
        obj._musclenn_optimizer.state_dict(), reference_optimizer.state_dict()
    )
    assert pointers() == captured_pointers


@pytest.mark.parametrize("failure_call", [2, 4])
def test_shuffle_capture_failure_restores_rng_and_workspace_after_delayed_writes(
    monkeypatch, failure_call
):
    from contextlib import nullcontext

    obj = Runtime()
    obj._musclenn_graph_permutations = torch.zeros(2, 5, dtype=torch.long)
    obj._musclenn_shuffle_generator = torch.Generator().manual_seed(17)
    stream = SetupStream()
    obj._musclenn_update_torch_stream = stream
    obj._musclenn_shuffle_graph = None
    before = obj._musclenn_shuffle_generator.get_state()
    workspace = obj._musclenn_graph_permutations.clone()
    error = RuntimeError("shuffle setup failed")
    generate = obj._generate_musclenn_graph_permutations
    calls = 0

    def failing_generate(generator):
        nonlocal calls
        calls += 1
        generate(generator)
        if calls == failure_call:
            stream.pending.append(lambda: generate(generator))
            raise error

    obj._generate_musclenn_graph_permutations = failing_generate
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "CUDAGraph",
        lambda: SimpleNamespace(register_generator_state=lambda generator: None),
    )
    with pytest.raises(RuntimeError, match="shuffle setup failed") as raised:
        obj._capture_musclenn_shuffle_graph()
    assert raised.value is error
    stream.synchronize()
    assert torch.equal(obj._musclenn_shuffle_generator.get_state(), before)
    assert torch.equal(obj._musclenn_graph_permutations, workspace)
    assert obj._musclenn_shuffle_graph is None


def test_shuffle_capture_preserves_rng_and_first_replay_consumption(monkeypatch):
    from contextlib import nullcontext

    obj = Runtime()
    obj._musclenn_graph_permutations = torch.zeros(2, 5, dtype=torch.long)
    obj._musclenn_shuffle_generator = torch.Generator().manual_seed(19)
    obj._musclenn_update_torch_stream = SetupStream()
    obj._musclenn_shuffle_graph = None
    obj._musclenn_use_shuffle_graph = True
    generator = obj._musclenn_shuffle_generator
    before = generator.get_state()
    pointer = obj._musclenn_graph_permutations.data_ptr()
    graph = SimpleNamespace(
        register_generator_state=lambda value: None,
        replay=lambda: obj._generate_musclenn_graph_permutations(generator),
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    obj._capture_musclenn_shuffle_graph()
    assert obj._musclenn_shuffle_graph is graph
    assert torch.equal(generator.get_state(), before)
    assert obj._musclenn_graph_permutations.data_ptr() == pointer
    reference = torch.Generator()
    reference.set_state(before)
    expected = torch.stack([torch.randperm(5, generator=reference) for _ in range(2)])
    obj._refresh_musclenn_graph_permutations()
    assert torch.equal(obj._musclenn_graph_permutations, expected)
    assert torch.equal(generator.get_state(), reference.get_state())
    assert obj._musclenn_graph_permutations.data_ptr() == pointer
