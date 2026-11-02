"""Wrapped MASS mechanics against MuJoCo virtual work, including captured execution."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
import warp as wp
from freemusco_runtime_fixtures import build_freemusco_imitation_cfg

from instantmimic.envs.freemusco.tasks.freemusco_imitation_env import (
    FreemuscoImitationEnv,
)
from instantmimic.muscle.muscle_dynamics import MuscleDynamicsLayer


@contextmanager
def wrapped_physics(force_velocity=False, graph=False):
    cfg = build_freemusco_imitation_cfg(species="ostrich", mass=True)
    cfg.muscle = {"force_velocity": {"enabled": force_velocity}}
    cfg.use_cuda_graph = graph
    cfg.include_muscle_observation_blocks = True
    env = FreemuscoImitationEnv(cfg)
    try:
        env.reset()
        model = mujoco.MjModel.from_xml_path(
            str(Path(__file__).resolve().parents[1] / cfg.sim.mjcf_path)
        )
        yield env, model, env.fast_muscle._musclenn_torque_helper
    finally:
        env.close()


def context_and_oracle(model, layer, names, amplitude, force_velocity):
    data = mujoco.MjData(model)
    rng = np.random.default_rng(41)
    displacement = rng.normal(size=model.nv) * amplitude
    displacement[:6] = 0
    mujoco.mj_integratePos(model, data.qpos, displacement, 1.0)
    data.qvel[:] = rng.normal(size=model.nv) * 0.1
    data.qvel[:6] = 0
    mujoco.mj_forward(model, data)
    values = [
        getattr(data, n).ravel()
        for n in ("xpos", "xquat", "xipos", "subtree_com", "cdof")
    ]
    if force_velocity:
        values.append(data.qvel)
    context = torch.tensor(
        np.concatenate(values), dtype=torch.float32, device=layer.muscle_params.device
    )[None]
    ids = [model.actuator(n).trnid[0] for n in names]
    jacobian = np.zeros((model.ntendon, model.nv))
    mujoco.mju_sparse2dense(
        jacobian, data.ten_J, model.ten_J_rownnz, model.ten_J_rowadr, model.ten_J_colind
    )
    moment = -jacobian[ids, 6:].T
    lengths = data.ten_length[ids]
    params = layer.muscle_params.detach().cpu().numpy().astype(np.float64)
    rest = layer.muscle_rest_length.detach().cpu().numpy()
    normalized = np.clip(
        (lengths / rest - params[:, 2]) / params[:, 1],
        layer.muscle_length_norm_min,
        layer.muscle_length_norm_max,
    )
    assert layer.muscle_active_force_type == "gaussian"
    unit = np.exp(-((normalized - 1.0) ** 2) / layer.muscle_active_force_gamma)
    gain = params[:, 0] * np.cos(params[:, 3]) * layer.muscle_force_scale
    assert layer.muscle_passive_force_type == "exponential"
    passive = (
        np.where(
            normalized > 1.0,
            np.expm1(
                np.clip(
                    layer.muscle_passive_force_k_pe
                    * (normalized - 1.0)
                    / layer.muscle_passive_force_e_mo,
                    -80,
                    80,
                )
            )
            / np.expm1(layer.muscle_passive_force_k_pe),
            0.0,
        )
        * gain
    )
    if force_velocity:
        velocity = jacobian[ids] @ data.qvel
        scale = (
            layer.muscle_force_velocity_length_scale.detach().cpu().numpy()
            * layer.muscle_force_velocity_vmax
        )
        v = velocity / np.maximum(scale, 1.0e-6)
        maximum = layer.muscle_force_velocity_fvmax
        width = maximum - 1.0
        multiplier = np.where(
            v <= -1.0,
            0.0,
            np.where(
                v <= 0.0,
                (v + 1.0) ** 2,
                np.where(v <= width, maximum - (width - v) ** 2 / width, maximum),
            ),
        )
        unit *= multiplier
    J = torch.tensor(
        moment * (unit * gain)[None], dtype=torch.float32, device=context.device
    )[None]
    passive_torque = torch.tensor(
        moment @ passive, dtype=torch.float32, device=context.device
    )[None]
    return context, J, passive_torque, data


def check_physics(env, model, layer, fv):
    activation = torch.linspace(0.2, 0.8, layer.muscle_count, device=env.device)[
        None
    ].requires_grad_()
    for amplitude in (0.0, 0.075, 0.3):
        context, expected_J, expected_passive, data = context_and_oracle(
            model, layer, env.fast_muscle.muscle_names, amplitude, fv
        )
        actual_J = layer.compute_activation_torque_jacobian(context).clone()
        torch.testing.assert_close(actual_J, expected_J, atol=0.006, rtol=2.0e-4)
        lower, upper = layer.compute_activation_torque_bounds(context)
        torch.testing.assert_close(
            lower, expected_J.clamp(max=0.0).sum(2), atol=0.012, rtol=2.0e-4
        )
        torch.testing.assert_close(
            upper, expected_J.clamp(min=0.0).sum(2), atol=0.012, rtol=2.0e-4
        )
        expected = (expected_J @ activation.detach()[..., None]).squeeze(
            -1
        ) + expected_passive
        torque = layer.torque_from_activation(
            activation, context, include_passive_force=True
        )
        torch.testing.assert_close(torque, expected, atol=0.012, rtol=2.0e-4)
        controller = env.fast_muscle
        for name in ("xpos", "xquat", "xipos", "subtree_com", "cdof", "qvel"):
            target = wp.to_torch(getattr(controller.mjw_data, name))
            target.copy_(
                torch.tensor(
                    getattr(data, name), device=target.device, dtype=target.dtype
                )[None]
            )
        controller._apply_activation_to_qfrc(
            activation.detach(), include_active=1.0, include_passive=1.0
        )
        applied = wp.to_torch(controller.mjw_data.qfrc_applied)[:, 6:]
        torch.testing.assert_close(applied, expected, atol=0.012, rtol=2.0e-4)
        lengths, velocities, _, _ = controller.update_muscle_observables()
        ids = [model.actuator(n).trnid[0] for n in controller.muscle_names]
        torch.testing.assert_close(
            lengths,
            torch.tensor(data.ten_length[ids], device=env.device, dtype=torch.float32)[
                None
            ],
            atol=3e-6,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            velocities,
            torch.tensor(
                data.ten_velocity[ids], device=env.device, dtype=torch.float32
            )[None],
            atol=3e-6,
            rtol=2e-5,
        )
        visualizer = controller._ensure_tendon_visualizer()._line_visualizer
        source = visualizer.build_gl_render_source()
        endpoints = source.endpoint_world_resolver(controller.mjw_data)
        native_endpoints = source.endpoint_world_resolver(data).clone().to(env.device)
        torch.testing.assert_close(endpoints, native_endpoints, atol=3e-6, rtol=2e-5)
        # Another geometry evaluation must not corrupt a pending activation backward.
        other, *_ = context_and_oracle(
            model, layer, env.fast_muscle.muscle_names, amplitude + 0.04, fv
        )
        layer.compute_activation_torque_jacobian(other)
        cotangent = torch.linspace(-0.7, 0.9, layer.torque_dim, device=env.device)[None]
        (gradient,) = torch.autograd.grad((torque * cotangent).sum(), activation)
        expected_gradient = (expected_J.transpose(1, 2) @ cotangent[..., None]).squeeze(
            -1
        )
        torch.testing.assert_close(gradient, expected_gradient, atol=0.012, rtol=2.0e-4)
        # Caller-owned strided caches are used in the actual rollout/coordinator path.
        unit_storage = torch.empty((1, layer.muscle_count * 2), device=env.device)
        passive_storage = torch.empty_like(unit_storage)
        unit, passive = unit_storage[:, ::2], passive_storage[:, ::2]
        returned = layer.precompute_lbs_force_magnitude_cache(
            context,
            include_passive_force=True,
            unit_force_magnitude=unit,
            passive_force_magnitude=passive,
        )
        assert returned[0].data_ptr() == unit.data_ptr()
        cached = layer.torque_from_activation(
            activation,
            context,
            include_passive_force=True,
            lbs_unit_force_magnitude=unit,
            lbs_passive_force_magnitude=passive,
        )
        torch.testing.assert_close(cached, expected, atol=0.012, rtol=2.0e-4)
        (cached_grad,) = torch.autograd.grad((cached * cotangent).sum(), activation)
        torch.testing.assert_close(
            cached_grad, expected_gradient, atol=0.012, rtol=2.0e-4
        )
        # Controller observables use native tendon velocity, not the tangent chord length.
        if amplitude == 0.0:
            env.reset()
            assert torch.isfinite(
                env.step(torch.zeros_like(env.actions))[0]["policy"]
            ).all()


@pytest.mark.parametrize("force_velocity", [False, True])
def test_wrapped_mass_virtual_work_torque_bounds_and_activation_gradient(
    force_velocity,
):
    with wrapped_physics(force_velocity) as (env, model, layer):
        check_physics(env, model, layer, force_velocity)


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.parametrize("force_velocity", [False, True])
def test_wrapped_mass_cuda_graph_cache_gradient_and_compile(force_velocity):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wp.init()
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        with wrapped_physics(force_velocity, graph=True) as (env, model, layer):
            # First captured step must work before any helper geometry is warmed.
            env.step(torch.zeros_like(env.actions))
            check_physics(env, model, layer, force_velocity)
            assert env._step_graph is not None
            context, _, _, _ = context_and_oracle(
                model, layer, env.fast_muscle.muscle_names, 0.0, force_velocity
            )
            activation = torch.full(
                (1, layer.muscle_count), 0.4, device=env.device, requires_grad=True
            )

            def evaluate(a, c):
                return layer.torque_from_activation(a, c, include_passive_force=True)

            for _ in range(3):
                evaluate(activation, context).sum().backward()
                activation.grad.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                torque = evaluate(activation, context)
                torque.sum().backward()
            for amplitude in (0.03, 0.15):
                changed, J, passive, _ = context_and_oracle(
                    model,
                    layer,
                    env.fast_muscle.muscle_names,
                    amplitude,
                    force_velocity,
                )
                context.copy_(changed)
                activation.grad.zero_()
                graph.replay()
                torch.testing.assert_close(
                    torque,
                    (J @ activation.detach()[..., None]).squeeze(-1) + passive,
                    atol=0.012,
                    rtol=2.0e-4,
                )
                torch.testing.assert_close(
                    activation.grad, J.sum(1), atol=0.012, rtol=2.0e-4
                )
            source = env.fast_muscle._ensure_tendon_visualizer()._line_visualizer.build_gl_render_source()
            source.endpoint_world_resolver(env.fast_muscle.mjw_data)
            render_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(render_graph, stream=stream):
                rendered = source.endpoint_world_resolver(env.fast_muscle.mjw_data)
            before = rendered.clone()
            delta = torch.tensor([0.03, -0.01, 0.04], device=env.device)
            wp.to_torch(env.fast_muscle.mjw_data.xpos).add_(delta)
            render_graph.replay()
            torch.testing.assert_close(rendered, before + delta, atol=3e-6, rtol=2e-5)
            layer.to("cpu")
            assert layer._waypoint_wrapping.inputs[0].device.is_cpu
            cpu_context = context.cpu()
            layer.forward_backend = "reference"
            cpu_value = evaluate(activation.detach().cpu(), cpu_context).clone()
            layer.to(env.device)
            layer.forward_backend = "auto"
            assert layer._waypoint_wrapping.inputs[0].device.is_cuda
            torch.testing.assert_close(
                evaluate(activation, context),
                cpu_value.to(env.device),
                atol=0.012,
                rtol=2.0e-4,
            )
            compiled = torch.compile(evaluate)
            torch.testing.assert_close(
                compiled(activation, context),
                evaluate(activation, context),
                atol=0.006,
                rtol=2.0e-4,
            )
    current.wait_stream(stream)


@pytest.mark.cuda
@pytest.mark.integration
def test_unwrapped_native_force_and_adjoint_remain_equivalent():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wp.init()
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        cfg = build_freemusco_imitation_cfg(species="chimanoid", mass=True)
        cfg.use_cuda_graph = True
        env = FreemuscoImitationEnv(cfg)
        try:
            env.reset()
            env.step(torch.zeros_like(env.actions))
            layer = env.fast_muscle._musclenn_torque_helper
            assert not layer._waypoint_wrapping.count
            assert layer._uses_native_body_wrench_projection
            model = mujoco.MjModel.from_xml_path(
                str(Path(__file__).resolve().parents[1] / cfg.sim.mjcf_path)
            )
            context, jacobian, passive, _ = context_and_oracle(
                model, layer, env.fast_muscle.muscle_names, 0.075, False
            )
            activation = torch.full(
                (1, layer.muscle_count), 0.4, device=env.device, requires_grad=True
            )
            # NumPy float64 virtual work remains independent of GPU matmul modes.
            expected = torch.tensor(
                (
                    jacobian.detach().cpu().numpy().astype(np.float64)
                    @ activation.detach().cpu().numpy()[..., None]
                ).squeeze(-1)
                + passive.cpu().numpy(),
                device=env.device,
                dtype=torch.float32,
            )
            expected_gradient = jacobian.sum(1)
            actual = layer.torque_from_activation(
                activation, context, include_passive_force=True
            )
            (actual_gradient,) = torch.autograd.grad(actual.sum(), activation)
            torch.testing.assert_close(actual, expected, atol=0.012, rtol=2e-4)
            torch.testing.assert_close(
                actual_gradient, expected_gradient, atol=0.012, rtol=2e-4
            )
        finally:
            env.close()
    current.wait_stream(stream)


@pytest.mark.cuda
@pytest.mark.integration
def test_native_prepare_reserves_a_new_batch_before_capture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wp.init()
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        with wrapped_physics() as (env, model, layer):
            batch = 3
            assert layer.prepare_native_body_wrench_full_force_workspace(
                batch_size=batch, device=env.device
            )
            data = env.fast_muscle.mjw_data
            fields = {
                name: wp.to_torch(getattr(data, name))
                .expand(batch, -1, -1)
                .contiguous()
                for name in ("xpos", "xquat", "xipos", "subtree_com", "cdof")
            }
            activation = torch.full((batch, layer.muscle_count), 0.4, device=env.device)
            output = torch.empty((batch, layer.torque_dim), device=env.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                layer.launch_native_body_wrench_full_force_into(
                    activation=activation,
                    body_pos=fields["xpos"],
                    body_quat=fields["xquat"],
                    xipos=fields["xipos"],
                    subtree_com=fields["subtree_com"],
                    cdof=fields["cdof"],
                    output=output,
                    include_passive_force=True,
                )
            graph.replay()
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output[1:], output[:1].expand_as(output[1:]))
    current.wait_stream(stream)


@pytest.mark.cuda
@pytest.mark.integration
def test_wrapped_native_matches_sparse_for_translated_short_segments():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wp.init()
    current = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        with wrapped_physics() as (env, model, layer):
            # Environment grids put short tendon segments far from the origin.
            # Compare the two float32 implementations at identical world poses;
            # independent MuJoCo virtual-work checks cover the mechanics above.
            context, _, _, _ = context_and_oracle(
                model, layer, env.fast_muscle.muscle_names, 0.075, False
            )
            batch = 64
            context = context.expand(batch, -1).clone()
            translation = torch.zeros((batch, 1, 3), device=env.device)
            translation[:, 0, 0] = torch.linspace(0.0, 100.0, batch, device=env.device)
            offset = 0
            for name, width in (
                ("xpos", 3),
                ("xquat", 4),
                ("xipos", 3),
                ("subtree_com", 3),
            ):
                size = layer.body_count * width
                if name != "xquat":
                    context[:, offset : offset + size].view(batch, -1, width).add_(
                        translation
                    )
                offset += size
            activation = torch.linspace(0.2, 0.8, layer.muscle_count, device=env.device)
            activation = activation.expand(batch, -1).contiguous().requires_grad_()
            assert layer._uses_native_body_wrench_projection
            native = layer.torque_from_activation(
                activation, context, include_passive_force=True
            )
            native_gradient = torch.autograd.grad(native.sum(), activation)[0]
            native = native.detach().clone()
            unit, passive = layer.precompute_lbs_force_magnitude_cache(
                context, include_passive_force=True
            )
            cached = (
                layer.torque_from_activation(
                    activation,
                    context,
                    include_passive_force=True,
                    lbs_unit_force_magnitude=unit,
                    lbs_passive_force_magnitude=passive,
                )
                .detach()
                .clone()
            )
            spec = env.fast_muscle._build_muscle_helper_runtime_spec(
                obs_context_dim=env.fast_muscle.helper_obs_context_dim,
                select_active_torque_dofs=False,
            )
            spec = replace(
                spec,
                projection=replace(
                    spec.projection,
                    lbs_body_wrench_forward_mode="pairchain",
                    static_metadata_cache_key=None,
                ),
            )
            sparse_layer = MuscleDynamicsLayer(spec).to(env.device)
            assert not sparse_layer._uses_native_body_wrench_projection
            sparse = sparse_layer.torque_from_activation(
                activation, context, include_passive_force=True
            )
            sparse_gradient = torch.autograd.grad(sparse.sum(), activation)[0]
            torch.testing.assert_close(native, sparse, atol=0.012, rtol=2e-4)
            torch.testing.assert_close(cached, sparse, atol=0.012, rtol=2e-4)
            torch.testing.assert_close(
                native_gradient, sparse_gradient, atol=0.012, rtol=2e-4
            )
    current.wait_stream(stream)
