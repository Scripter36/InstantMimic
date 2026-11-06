"""Strict fullgraph compilation and gradient parity of the Warp muscle ops.

Compiled RL losses may only reach Warp through the opaque dispatcher operators,
so these tests compile a real muscle loss with fullgraph=True and compare the
operator gradients against the uncompiled eager Function they replace.
"""

from __future__ import annotations

import copy
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from muscle_warp_fixtures import (
    make_context,
    make_force_velocity_layer,
    make_muscle_layer,
    with_warp_forward,
)

from instantmimic.muscle import muscle_dynamics as md
from instantmimic.muscle.warp_custom_ops import (
    muscle_torque_bounds,
    muscle_torque_cached_lbs_forward,
    muscle_torque_forward,
    warp_op_key,
    warp_stream_for_handle,
)
from instantmimic.utils import torch_warp_capture
from instantmimic.utils.torch_warp_capture import warp_stream_from_torch


def _eager_loss(layer, activation, context) -> torch.Tensor:
    torque = md._WarpMuscleDynamicsFromActivation.apply(
        activation.sin(), context, layer, False
    )
    return torque.square().sum()


def _op_loss(layer, activation, context) -> torch.Tensor:
    torque = muscle_torque_forward(activation.sin(), context, warp_op_key(layer), False)
    return torque.square().sum()


@pytest.mark.parametrize(
    ("uses_lbs", "native_tile"),
    # The CUDA native-tile path is covered by the marked tests below; this
    # public layer only dispatches Warp bodies on CUDA.
    [(False, False), (True, False)],
)
def test_gradient_matches_the_eager_function(uses_lbs: bool, native_tile: bool) -> None:
    layer = make_muscle_layer(uses_lbs=uses_lbs, native_tile=native_tile)
    context = make_context(layer)
    activation = torch.full((4, 3), 0.4)

    eager_input = activation.clone().requires_grad_()
    eager_loss = _eager_loss(layer, eager_input, context)
    eager_grad = torch.autograd.grad(eager_loss, eager_input)[0]

    op_input = activation.clone().requires_grad_()
    op_loss = _op_loss(layer, op_input, context)
    op_grad = torch.autograd.grad(op_loss, op_input)[0]

    torch.testing.assert_close(op_loss, eager_loss, rtol=0, atol=0)
    torch.testing.assert_close(op_grad, eager_grad, rtol=0, atol=0)
    assert torch.count_nonzero(op_grad) > 0


def test_cached_lbs_force_gradient_matches_the_eager_function() -> None:
    layer = make_muscle_layer(uses_lbs=True)
    context = make_context(layer)
    activation = torch.full((4, 3), 0.5)
    _, _, unit_force = layer._warp_muscle_dynamics_from_activation_forward(
        activation,
        context,
        include_passive_force=False,
    )
    passive = torch.zeros_like(unit_force)

    eager_input = activation.clone().requires_grad_()
    eager_loss = (
        md._WarpMuscleDynamicsFromActivationCachedLBSForce.apply(
            eager_input.sin(), context, unit_force, passive, layer, False
        )
        .square()
        .sum()
    )
    eager_grad = torch.autograd.grad(eager_loss, eager_input)[0]

    op_input = activation.clone().requires_grad_()
    op_loss = (
        muscle_torque_cached_lbs_forward(
            op_input.sin(),
            context,
            unit_force,
            passive,
            warp_op_key(layer),
            False,
        )
        .square()
        .sum()
    )
    op_grad = torch.autograd.grad(op_loss, op_input)[0]

    torch.testing.assert_close(op_loss, eager_loss, rtol=0, atol=0)
    torch.testing.assert_close(op_grad, eager_grad, rtol=0, atol=0)


def test_strict_fullgraph_compile_carries_a_muscle_loss() -> None:
    layer = make_muscle_layer()
    context = make_context(layer)
    activation = torch.full((4, 3), 0.4)

    reference = activation.clone().requires_grad_()
    reference_loss = _eager_loss(layer, reference, context)
    reference_grad = torch.autograd.grad(reference_loss, reference)[0]

    def compiled_loss(value: torch.Tensor) -> torch.Tensor:
        return _op_loss(layer, value, context)

    explanation = torch._dynamo.explain(compiled_loss)(activation.clone())
    assert explanation.graph_count == 1, explanation
    assert explanation.graph_break_count == 0, explanation

    compiled = torch.compile(compiled_loss, backend="inductor", fullgraph=True)
    compiled_input = activation.clone().requires_grad_()
    loss = compiled(compiled_input)
    grad = torch.autograd.grad(loss, compiled_input)[0]

    torch.testing.assert_close(loss, reference_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grad, reference_grad, rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(grad) > 0


def test_production_layer_entry_point_compiles_fullgraph() -> None:
    """The layer call itself must be traceable, not just the raw operator."""
    layer = with_warp_forward(make_muscle_layer())
    context = make_context(layer)
    activation = torch.full((4, 3), 0.4)

    def loss(value: torch.Tensor) -> torch.Tensor:
        return (
            layer.torque_from_activation(
                value.sin(), context, include_passive_force=False
            )
            .square()
            .sum()
        )

    reference_input = activation.clone().requires_grad_()
    reference_loss = _eager_loss(layer, reference_input, context)
    (reference_grad,) = torch.autograd.grad(reference_loss, reference_input)

    compiled = torch.compile(loss, backend="inductor", fullgraph=True)
    compiled_input = activation.clone().requires_grad_()
    compiled_loss = compiled(compiled_input)
    (compiled_grad,) = torch.autograd.grad(compiled_loss, compiled_input)

    torch.testing.assert_close(compiled_loss, reference_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(compiled_grad, reference_grad, rtol=1e-6, atol=1e-6)


def test_operators_are_generic_across_layer_instances() -> None:
    layers = [
        make_muscle_layer(muscle_count=1, torque_dim=1),
        make_muscle_layer(muscle_count=3, torque_dim=2),
    ]
    assert len({warp_op_key(layer) for layer in layers}) == 2

    for layer in layers:
        context = make_context(layer)
        activation = torch.full((4, layer.muscle_count), 0.25)

        eager_input = activation.clone().requires_grad_()
        eager_loss = _eager_loss(layer, eager_input, context)
        eager_grad = torch.autograd.grad(eager_loss, eager_input)[0]

        compiled = torch.compile(
            lambda value, ctx=context, lyr=layer: _op_loss(lyr, value, ctx),
            backend="aot_eager",
            fullgraph=True,
        )
        op_input = activation.clone().requires_grad_()
        op_loss = compiled(op_input)
        op_grad = torch.autograd.grad(op_loss, op_input)[0]

        assert op_grad.shape == (4, layer.muscle_count)
        torch.testing.assert_close(op_grad, eager_grad, rtol=1e-6, atol=1e-6)


def test_torque_bounds_return_owned_copies() -> None:
    layer = with_warp_forward(make_muscle_layer())
    context = make_context(layer)
    key = warp_op_key(layer)

    first_lower, first_upper = muscle_torque_bounds(context, key, True)
    expected_lower = first_lower.detach().clone()
    expected_upper = first_upper.detach().clone()

    changed = context.clone()
    changed[:, -4] = 2.0
    second_lower, second_upper = muscle_torque_bounds(changed, key, True)

    torch.testing.assert_close(first_lower, expected_lower, rtol=0, atol=0)
    torch.testing.assert_close(first_upper, expected_upper, rtol=0, atol=0)
    assert not torch.equal(first_lower, second_lower) or not torch.equal(
        first_upper, second_upper
    )


def test_backward_survives_losing_the_layers_last_reference() -> None:
    layer = make_muscle_layer()
    key = warp_op_key(layer)
    context = make_context(layer)
    activation = torch.full((4, 3), 0.4)
    grad_input = activation.clone().requires_grad_()
    loss = muscle_torque_forward(grad_input.sin(), context, key, False).square().sum()

    del layer
    gc.collect()

    (grad,) = torch.autograd.grad(loss, grad_input)
    assert torch.count_nonzero(grad) > 0


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy], ids=["copy", "deepcopy"])
def test_copied_layer_first_call_compiles_fullgraph(copier) -> None:
    """A copy registers while copying, not inside its first compiled call."""
    layer = make_muscle_layer()
    original_key = warp_op_key(layer)
    clone = copier(layer)
    # Reading the carried attribute must be all the copy does; allocating a
    # handle inside the compiled call below would break fullgraph tracing.
    clone_key = clone._warp_op_key
    assert clone_key != original_key

    context = make_context(clone)
    activation = torch.full((4, 3), 0.4)
    with_warp_forward(clone)

    del layer
    gc.collect()

    compiled = torch.compile(
        lambda value: clone.torque_from_activation(
            value.sin(), context, include_passive_force=False
        )
        .square()
        .sum(),
        backend="inductor",
        fullgraph=True,
    )
    compiled_input = activation.clone().requires_grad_()
    compiled_loss = compiled(compiled_input)
    (compiled_grad,) = torch.autograd.grad(compiled_loss, compiled_input)

    reference_input = activation.clone().requires_grad_()
    reference_loss = (
        md._WarpMuscleDynamicsFromActivation.apply(
            reference_input.sin(), context, clone, False
        )
        .square()
        .sum()
    )
    (reference_grad,) = torch.autograd.grad(reference_loss, reference_input)

    torch.testing.assert_close(compiled_loss, reference_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(compiled_grad, reference_grad, rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(compiled_grad) > 0


def _fresh_stream_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_warp_capture, "_WARP_STREAMS", {})


def test_stream_lookup_reuses_one_wrapper_across_torch_stream_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """torch returns a new stream object per call for the same CUDA handle."""
    _fresh_stream_cache(monkeypatch)
    created: list[SimpleNamespace] = []

    def create(handle: int) -> SimpleNamespace:
        wrapper = SimpleNamespace(cuda_stream=handle)
        created.append(wrapper)
        return wrapper

    first = warp_stream_for_handle(7, SimpleNamespace(cuda_stream=1), create)
    # A second call sees a different current stream and a different torch
    # object, but the same handle: it must keep the cached wrapper.
    second = warp_stream_for_handle(7, SimpleNamespace(cuda_stream=2), create)

    assert first is second
    assert len(created) == 1


def test_stream_lookup_never_duplicates_a_stream_warp_already_owns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_stream_cache(monkeypatch)
    owned = SimpleNamespace(cuda_stream=11)

    def create(handle: int) -> SimpleNamespace:
        raise AssertionError(f"must reuse the owning stream for handle {handle}")

    assert warp_stream_for_handle(11, owned, create) is owned


def test_stream_lookup_keeps_an_owner_seen_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_stream_cache(monkeypatch)
    owner = SimpleNamespace(cuda_stream=13)
    created: list[SimpleNamespace] = []

    def create(handle: int) -> SimpleNamespace:
        wrapper = SimpleNamespace(cuda_stream=handle)
        created.append(wrapper)
        return wrapper

    assert warp_stream_for_handle(13, owner, create) is owner
    # Warp's current stream moved elsewhere, but the handle is still owned.
    assert warp_stream_for_handle(13, SimpleNamespace(cuda_stream=2), create) is owner
    assert not created


class _FakeWarpStream:
    def __init__(self, handle: int) -> None:
        self.cuda_stream = handle


class _FakeWarp:
    """Just enough of warp to exercise the shared stream registry."""

    def __init__(self) -> None:
        self.current: _FakeWarpStream | None = None
        self.created: list[_FakeWarpStream] = []

    def get_device(self, name: str) -> str:
        return name

    def get_stream(self, device: str) -> _FakeWarpStream | None:
        return self.current

    def Stream(self, device: str, *, cuda_stream: int) -> _FakeWarpStream:
        stream = _FakeWarpStream(cuda_stream)
        self.created.append(stream)
        return stream


def test_stream_lookup_returns_the_retained_wrapper_over_a_later_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_stream_cache(monkeypatch)
    created: list[SimpleNamespace] = []

    def create(handle: int) -> SimpleNamespace:
        wrapper = SimpleNamespace(cuda_stream=handle)
        created.append(wrapper)
        return wrapper

    cached = warp_stream_for_handle(11, None, create)
    # A later duplicate wrapper for the same handle must not be handed back:
    # the caller would drop the wrapper the registry does not retain, and
    # collecting it unregisters the handle under the cached stream.
    duplicate = SimpleNamespace(cuda_stream=11)
    assert warp_stream_for_handle(11, duplicate, create) is cached
    assert torch_warp_capture._WARP_STREAMS[11] is cached
    del duplicate
    gc.collect()
    assert warp_stream_for_handle(11, None, create) is cached
    assert len(created) == 1


def test_warp_stream_from_torch_shares_one_wrapper_per_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_stream_cache(monkeypatch)
    fake_warp = _FakeWarp()
    monkeypatch.setattr(torch_warp_capture, "wp", fake_warp)
    first = warp_stream_from_torch(SimpleNamespace(cuda_stream=29, device="cuda:0"))
    assert fake_warp.created == [first]

    # Warp's current stream is now a duplicate of the same handle; the shared
    # helper keeps returning the wrapper it retains instead of wrapping again.
    fake_warp.current = _FakeWarpStream(29)
    second = warp_stream_from_torch(SimpleNamespace(cuda_stream=29, device="cuda:0"))
    assert second is first
    assert fake_warp.created == [first]
    assert torch_warp_capture._WARP_STREAMS[29] is first


def test_collecting_a_duplicate_wrapper_leaves_the_shared_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_stream_cache(monkeypatch)
    fake_warp = _FakeWarp()
    monkeypatch.setattr(torch_warp_capture, "wp", fake_warp)
    torch_stream = SimpleNamespace(cuda_stream=37, device="cuda:0")
    cached = warp_stream_from_torch(torch_stream)

    # An external caller wraps the same handle itself and then drops it, as
    # MuscleNN used to do with wp.stream_from_torch.
    duplicate = _FakeWarpStream(37)
    duplicate_ref = weakref.ref(duplicate)
    del duplicate
    gc.collect()
    assert duplicate_ref() is None

    assert torch_warp_capture._WARP_STREAMS[37] is cached
    assert warp_stream_from_torch(torch_stream) is cached
    assert fake_warp.created == [cached]


def test_force_velocity_backward_rebuilds_the_saved_unit_force() -> None:
    """The op backward carries no saved force, so it must rebuild the same one.

    The forward caches a force-velocity-scaled unit force from its precomputed
    force kernel; the op backward recomputes it from geometry. The eager
    Function keeps the forward's tensor, so matching it proves the rebuild is
    equivalent, including the native force-velocity adjoint's input.
    """
    layer = with_warp_forward(make_force_velocity_layer())
    context = make_context(layer, velocity=0.35)
    activation = torch.full((4, 3), 0.4)

    eager_input = activation.clone().requires_grad_()
    eager_loss = _eager_loss(layer, eager_input, context)
    (eager_grad,) = torch.autograd.grad(eager_loss, eager_input)
    assert torch.count_nonzero(eager_grad) > 0

    compiled = torch.compile(
        lambda value: layer.torque_from_activation(
            value.sin(), context, include_passive_force=False
        )
        .square()
        .sum(),
        backend="inductor",
        fullgraph=True,
    )
    op_input = activation.clone().requires_grad_()
    op_loss = compiled(op_input)
    (op_grad,) = torch.autograd.grad(op_loss, op_input)

    torch.testing.assert_close(op_loss, eager_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(op_grad, eager_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_force_velocity_gradient_and_capture_replay_match_the_eager_function() -> (
    None
):
    """Native-tile force-velocity layers keep the native adjoint.

    The op boundary saves no unit force, so the backward rebuilds the
    force-velocity-scaled magnitude the forward cached. Comparing against the
    eager Function (which caches it) catches a fallback to generic geometry.
    """
    import warp as wp

    from instantmimic.utils.torch_warp_capture import capture_warp_graph

    device = torch.device("cuda")
    layer = make_force_velocity_layer(device, native_tile=True)
    assert layer._uses_native_body_wrench_projection
    assert layer.muscle_force_velocity_enabled
    context = make_context(layer, velocity=0.35, device=device)
    activation = torch.full((4, 3), 0.4, device=device)

    def eager_loss(value: torch.Tensor) -> torch.Tensor:
        return (
            md._WarpMuscleDynamicsFromActivation.apply(
                value.sin(), context, layer, False
            )
            .square()
            .sum()
        )

    compiled_loss = torch.compile(
        lambda value: layer.torque_from_activation(
            value.sin(), context, include_passive_force=False
        )
        .square()
        .sum(),
        backend="inductor",
        fullgraph=True,
    )

    warp_device = wp.get_device(str(device))
    owned = wp.Stream(warp_device)
    torch_stream = wp.stream_to_torch(owned)
    with (
        torch.cuda.stream(torch_stream),
        wp.ScopedStream(owned, sync_enter=False, sync_exit=False),
    ):
        for _ in range(3):
            warmup_input = activation.clone().requires_grad_()
            torch.autograd.grad(compiled_loss(warmup_input), warmup_input)
        torch.cuda.synchronize(device)

        op_input = activation.clone().requires_grad_()
        op_loss = compiled_loss(op_input)
        (op_grad,) = torch.autograd.grad(op_loss, op_input)
        reference_input = activation.clone().requires_grad_()
        reference_loss = eager_loss(reference_input)
        (reference_grad,) = torch.autograd.grad(reference_loss, reference_input)
        torch.testing.assert_close(op_loss, reference_loss, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(op_grad, reference_grad, rtol=1e-5, atol=1e-6)
        assert torch.count_nonzero(op_grad) > 0

        # Inputs live outside the captured region: a clone recorded in the
        # graph would overwrite the replay-time values with the original.
        captured_input = activation.clone().requires_grad_()
        with capture_warp_graph(stream=owned) as capture:
            captured_loss = compiled_loss(captured_input)
            (captured_grad,) = torch.autograd.grad(captured_loss, captured_input)

        for value in (0.4, 0.2, 0.7):
            with torch.no_grad():
                captured_input.fill_(value)
            torch.cuda.synchronize(device)
            wp.capture_launch(capture.graph, stream=owned)
            wp.synchronize_stream(owned)

            replay_reference = activation.clone()
            replay_reference.fill_(value)
            replay_reference.requires_grad_()
            replay_loss = eager_loss(replay_reference)
            (replay_grad,) = torch.autograd.grad(replay_loss, replay_reference)
            torch.testing.assert_close(captured_loss, replay_loss, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(captured_grad, replay_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_capture_replay_on_a_warp_owned_stream_keeps_the_handle_registered() -> None:
    import warp as wp

    from instantmimic.utils.torch_warp_capture import capture_warp_graph

    device = torch.device("cuda")
    layer = make_muscle_layer(device)
    context = make_context(layer, device=device)
    activation = torch.full((4, 3), 0.4, device=device)

    compiled = torch.compile(
        lambda value: _op_loss(layer, value, context),
        backend="inductor",
        fullgraph=True,
    )
    warp_device = wp.get_device(str(device))
    owned = wp.Stream(warp_device)
    torch_stream = wp.stream_to_torch(owned)
    handle = int(torch_stream.cuda_stream)

    with (
        torch.cuda.stream(torch_stream),
        wp.ScopedStream(owned, sync_enter=False, sync_exit=False),
    ):
        warmup_input = activation.clone().requires_grad_()
        for _ in range(3):
            warmup_loss = compiled(warmup_input)
            torch.autograd.grad(warmup_loss, warmup_input)
        torch.cuda.synchronize(device)

        # The operator must adopt the stream this agent owns rather than
        # register a second wrapper for the same CUDA handle.
        current = wp.get_stream(warp_device)

        def _never_create(_handle: int) -> object:
            raise AssertionError("a duplicate Warp stream wrapper was created")

        assert warp_stream_for_handle(handle, current, _never_create) is current

        # Inputs live outside the captured region: a clone recorded in the
        # graph would overwrite the replay-time values with the original.
        captured_input = activation.clone().requires_grad_()
        second_input = activation.clone().requires_grad_()
        with capture_warp_graph(stream=owned) as capture:
            captured_loss = compiled(captured_input)
            (captured_grad,) = torch.autograd.grad(captured_loss, captured_input)
        # A second capture on the same stream proves the handle stayed
        # registered after the operators ran (the unknown-stream regression).
        with capture_warp_graph(stream=owned):
            second_loss = compiled(second_input)
            torch.autograd.grad(second_loss, second_input)

        for value in (0.4, 0.2, 0.7):
            with torch.no_grad():
                captured_input.fill_(value)
            torch.cuda.synchronize(device)
            wp.capture_launch(capture.graph, stream=owned)
            wp.synchronize_stream(owned)

            reference = activation.clone()
            reference.fill_(value)
            reference.requires_grad_()
            reference_loss = _eager_loss(layer, reference, context)
            (reference_grad,) = torch.autograd.grad(reference_loss, reference)
            torch.testing.assert_close(
                captured_loss, reference_loss, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                captured_grad, reference_grad, rtol=1e-5, atol=1e-6
            )
