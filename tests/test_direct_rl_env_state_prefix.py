from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from mujoco_warp._src.types import IntegratorType

import instantmimic.envs.direct_rl_env as direct_rl_env
from instantmimic.envs.direct_rl_env import DirectRLEnv


def _make_direct_rl_env_stub() -> DirectRLEnv:
    env = DirectRLEnv.__new__(DirectRLEnv)
    env._initialize_runtime_state()
    env._integrator_spec = DirectRLEnv._resolve_integrator_spec("implicitfast")
    return env


def test_substep_reward_mean_is_supported_by_base_state_prefix_lifecycle():
    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(reward=SimpleNamespace(substep_mean=True))
    assert env._resolve_substep_rewards_enabled() is True


def test_use_parallelized_kernel_launch_requires_cuda():
    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(use_parallelized_kernel_launch=True)
    env.device = torch.device("cpu")

    with pytest.raises(ValueError, match="use_parallelized_kernel_launch"):
        env._resolve_use_parallelized_kernel_launch()


def test_use_optimized_kernel_does_not_require_parallelized_launch():
    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(use_optimized_kernel=True)
    env._use_parallelized_kernel_launch = False
    env.device = torch.device("cuda")

    assert env._resolve_use_optimized_kernel() is True


def test_run_mjw_step_uses_custom_path_without_parallel_streams(monkeypatch):
    custom_calls = []
    stock_calls = []

    def fake_step_optimized(
        model, data, streams, scratch=None, *, use_optimized_kernel=None
    ):
        custom_calls.append((model, data, streams, scratch, use_optimized_kernel))

    monkeypatch.setattr(
        direct_rl_env, "step_optimized_fwd_position", fake_step_optimized
    )
    monkeypatch.setattr(
        direct_rl_env.mjw,
        "step",
        lambda model, data: stock_calls.append((model, data)),
    )
    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(use_cuda_graph=False)
    env._compile_mjw_step = False
    env._use_parallelized_kernel_launch = False
    env._use_optimized_kernel = False
    env._optimized_fwd_position_streams = ()
    env._optimized_fwd_position_scratch = object()
    env.mjw_model = object()
    env.mjw_data = object()

    env._run_mjw_step()

    assert custom_calls == [
        (
            env.mjw_model,
            env.mjw_data,
            (),
            env._optimized_fwd_position_scratch,
            False,
        )
    ]
    assert stock_calls == []


@pytest.mark.parametrize(
    ("step_kwargs", "step_method", "graph_attrs"),
    [
        (
            {},
            "_run_mjw_step_optimized_fwd_position",
            ("_mjw_step_graph", "_mjw_step_parallelized_graph"),
        ),
        (
            {"skip_kinematics": True},
            "_run_mjw_step_skip_optimized_fwd_position",
            ("_mjw_step_skip_kinematics_graph", "_mjw_step_skip_parallelized_graph"),
        ),
        (
            {"skip_state_prefix": True},
            "_run_mjw_step_after_state_prefix_optimized_fwd_position",
            (
                "_mjw_step_after_state_prefix_graph",
                "_mjw_step_after_state_prefix_parallelized_graph",
            ),
        ),
    ],
)
@pytest.mark.parametrize("parallelized", [False, True])
def test_compile_mjw_step_launches_each_captured_variant_once_per_call(
    monkeypatch, step_kwargs, step_method, graph_attrs, parallelized
):
    graph_attr = graph_attrs[parallelized]
    events = []
    graph = object()

    @contextmanager
    def fake_capture(*_args):
        events.append("capture_begin")
        try:
            yield SimpleNamespace(graph=graph)
        finally:
            events.append("capture_end")

    def fake_launch(*args):
        events.append(("launch", args[0]))

    monkeypatch.setattr(
        direct_rl_env, "_stream_aligned_cuda_graph_capture", fake_capture
    )
    monkeypatch.setattr(direct_rl_env, "_launch_stream_aligned_cuda_graph", fake_launch)
    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(use_cuda_graph=False)
    env.device = torch.device("cpu")
    env._compile_mjw_step = True
    env._use_parallelized_kernel_launch = parallelized
    env._env_graph_torch_stream = "torch-stream"
    env._env_graph_wp_stream = "wp-stream"
    setattr(env, graph_attr, None)
    env.mjw_model = object()
    env.mjw_data = object()
    setattr(
        env,
        step_method,
        lambda _model, _data: events.append("step"),
    )

    env._run_mjw_step(**step_kwargs)
    env._run_mjw_step(**step_kwargs)

    assert events == [
        "capture_begin",
        "step",
        "capture_end",
        ("launch", graph),
        ("launch", graph),
    ]
    assert getattr(env, graph_attr) is graph


@pytest.mark.parametrize(
    ("step_kwargs", "step_method", "graph_attrs"),
    [
        (
            {},
            "_run_mjw_step_optimized_fwd_position",
            ("_mjw_step_graph", "_mjw_step_parallelized_graph"),
        ),
        (
            {"skip_kinematics": True},
            "_run_mjw_step_skip_optimized_fwd_position",
            ("_mjw_step_skip_kinematics_graph", "_mjw_step_skip_parallelized_graph"),
        ),
        (
            {"skip_state_prefix": True},
            "_run_mjw_step_after_state_prefix_optimized_fwd_position",
            (
                "_mjw_step_after_state_prefix_graph",
                "_mjw_step_after_state_prefix_parallelized_graph",
            ),
        ),
    ],
)
@pytest.mark.parametrize("parallelized", [False, True])
def test_compile_mjw_step_capture_failure_does_not_launch(
    monkeypatch, step_kwargs, step_method, graph_attrs, parallelized
):
    graph_attr = graph_attrs[parallelized]
    events = []

    @contextmanager
    def fake_capture(*_args):
        events.append("capture_begin")
        try:
            yield SimpleNamespace(graph=object())
        finally:
            events.append("capture_end")

    def fake_launch(*_args):
        events.append("launch")

    monkeypatch.setattr(
        direct_rl_env, "_stream_aligned_cuda_graph_capture", fake_capture
    )
    monkeypatch.setattr(direct_rl_env, "_launch_stream_aligned_cuda_graph", fake_launch)
    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(use_cuda_graph=False)
    env.device = torch.device("cpu")
    env._compile_mjw_step = True
    env._use_parallelized_kernel_launch = parallelized
    env._env_graph_torch_stream = "torch-stream"
    env._env_graph_wp_stream = "wp-stream"
    setattr(env, graph_attr, None)
    env.mjw_model = object()
    env.mjw_data = object()

    def failing_step(_model, _data):
        events.append("step")
        raise RuntimeError("capture failed")

    setattr(env, step_method, failing_step)

    with pytest.raises(RuntimeError, match="capture failed"):
        env._run_mjw_step(**step_kwargs)

    assert events == ["capture_begin", "step", "capture_end"]
    assert getattr(env, graph_attr) is None


def test_run_mjw_forward_uses_custom_allocation_free_path(monkeypatch):
    calls = []

    def fake_forward_optimized(
        model, data, streams, scratch=None, *, use_optimized_kernel=None
    ):
        calls.append((model, data, streams, scratch, use_optimized_kernel))

    monkeypatch.setattr(
        direct_rl_env,
        "forward_optimized_fwd_position",
        fake_forward_optimized,
    )
    env = _make_direct_rl_env_stub()
    env._use_parallelized_kernel_launch = False
    env._use_optimized_kernel = False
    env._optimized_fwd_position_streams = ()
    env._optimized_fwd_position_scratch = object()
    env.mjw_model = object()
    env.mjw_data = object()

    env._run_mjw_forward()

    assert calls == [
        (
            env.mjw_model,
            env.mjw_data,
            (),
            env._optimized_fwd_position_scratch,
            False,
        )
    ]


def test_run_mjw_forward_after_state_prefix_uses_custom_suffix(monkeypatch):
    calls = []

    def fake_forward_after_state_prefix(
        model, data, streams, scratch=None, *, use_optimized_kernel=None
    ):
        calls.append((model, data, streams, scratch, use_optimized_kernel))

    monkeypatch.setattr(
        direct_rl_env,
        "forward_after_state_prefix_optimized_fwd_position",
        fake_forward_after_state_prefix,
    )
    env = _make_direct_rl_env_stub()
    env._use_parallelized_kernel_launch = False
    env._use_optimized_kernel = False
    env._optimized_fwd_position_streams = ()
    env._optimized_fwd_position_scratch = object()
    env.mjw_model = object()
    env.mjw_data = object()

    env._run_mjw_forward_after_state_prefix()

    assert calls == [
        (
            env.mjw_model,
            env.mjw_data,
            (),
            env._optimized_fwd_position_scratch,
            False,
        )
    ]


def test_control_forward_selects_suffix_only_during_active_pipeline():
    env = _make_direct_rl_env_stub()
    calls = []
    env._run_mjw_forward = lambda: calls.append("full")
    env._run_mjw_forward_after_state_prefix = lambda: calls.append("suffix")

    env._step_pipeline_active = False
    env._run_mjw_control_forward()
    env._step_pipeline_active = True
    env._run_mjw_control_forward()

    assert calls == ["full", "suffix"]


def test_install_implicit_velocity_derivative_hook_uses_persistent_scratch():
    env = _make_direct_rl_env_stub()
    hook = lambda _qh: None
    scratch = SimpleNamespace(
        implicit_qderiv=object(),
        implicit_qacc=object(),
        implicit_qld=object(),
        implicit_ql_diag_inv=object(),
        implicit_deriv_call=lambda: None,
        implicit_qh_contributors=(),
        implicit_qh_registration_frozen=False,
    )
    env._optimized_fwd_position_scratch = scratch
    env.mjw_model = SimpleNamespace(
        is_sparse=True,
        opt=SimpleNamespace(integrator=IntegratorType.IMPLICITFAST),
    )

    env._install_mjw_implicit_velocity_derivative_hook(hook)

    assert [item.name for item in scratch.implicit_qh_contributors] == [
        "fast_muscle.force_velocity"
    ]
    assert scratch.implicit_qh_contributors[0].apply is hook


def test_register_implicit_qh_contributors_preserves_order_and_rejects_duplicate():
    env = _make_direct_rl_env_stub()
    scratch = SimpleNamespace(
        implicit_qderiv=object(),
        implicit_qacc=object(),
        implicit_qld=object(),
        implicit_ql_diag_inv=object(),
        implicit_deriv_call=lambda: None,
        implicit_qh_contributors=(),
        implicit_qh_registration_frozen=False,
    )
    env._optimized_fwd_position_scratch = scratch
    env.mjw_model = SimpleNamespace(
        is_sparse=False,
        opt=SimpleNamespace(integrator=IntegratorType.IMPLICITFAST),
    )
    first = lambda _qh: None
    second = lambda _qh: None

    env._register_mjw_implicit_qh_contributor("first", first)
    env._register_mjw_implicit_qh_contributor("second", second)

    assert [item.name for item in scratch.implicit_qh_contributors] == [
        "first",
        "second",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        env._register_mjw_implicit_qh_contributor("first", first)


def test_register_implicit_qh_contributor_rejects_late_registration():
    env = _make_direct_rl_env_stub()
    env._optimized_fwd_position_scratch = SimpleNamespace(
        implicit_qderiv=object(),
        implicit_qacc=object(),
        implicit_qld=object(),
        implicit_ql_diag_inv=object(),
        implicit_deriv_call=lambda: None,
        implicit_qh_contributors=(),
        implicit_qh_registration_frozen=True,
    )
    env.mjw_model = SimpleNamespace(
        is_sparse=False,
        opt=SimpleNamespace(integrator=IntegratorType.IMPLICITFAST),
    )

    with pytest.raises(RuntimeError, match="before stepping"):
        env._register_mjw_implicit_qh_contributor("late", lambda _qh: None)


def test_install_pre_advance_modifier_uses_single_persistent_slot():
    env = _make_direct_rl_env_stub()
    env._optimized_fwd_position_scratch = SimpleNamespace(
        implicit_qh_registration_frozen=False,
        pre_advance_modifier=None,
    )
    env._step_graph = None
    env._step_graph_cache = {}
    callback = lambda _model, _data, qacc, qvel: (qacc, qvel)
    implicit_factor = object()

    env._install_mjw_pre_advance_modifier(
        "contact",
        callback,
        implicit_factor=implicit_factor,
    )

    installed = env._optimized_fwd_position_scratch.pre_advance_modifier
    assert installed.name == "contact"
    assert installed.apply is callback
    assert installed.implicit_factor is implicit_factor
    with pytest.raises(ValueError, match="only one"):
        env._install_mjw_pre_advance_modifier("second", callback)


def test_install_pre_advance_modifier_rejects_late_registration():
    env = _make_direct_rl_env_stub()
    env._optimized_fwd_position_scratch = SimpleNamespace(
        implicit_qh_registration_frozen=True,
        pre_advance_modifier=None,
    )
    env._step_graph = None
    env._step_graph_cache = {}

    with pytest.raises(RuntimeError, match="before stepping"):
        env._install_mjw_pre_advance_modifier(
            "late",
            lambda _model, _data, qacc, qvel: (qacc, qvel),
        )


def test_retained_cholesky_factor_rejects_nonsymmetric_free_body_correction():
    env = _make_direct_rl_env_stub()
    env.mjw_model = SimpleNamespace(body_freeadr=SimpleNamespace(size=1))
    env._optimized_fwd_position_scratch = SimpleNamespace(
        implicit_qh_registration_frozen=False, pre_advance_modifier=None
    )
    env._step_graph = None
    env._step_graph_cache = {}
    callback = lambda _model, _data, qacc, qvel: (qacc, qvel)
    with pytest.raises(ValueError, match="nonsymmetric"):
        env._install_mjw_pre_advance_modifier(
            "contact", callback, implicit_factor=object()
        )
    env._install_mjw_pre_advance_modifier("observe", callback)


def test_passive_force_hook_runs_after_native_passive_once():
    env = _make_direct_rl_env_stub()
    calls = []
    env._optimized_fwd_position_scratch = SimpleNamespace(
        implicit_qh_registration_frozen=False,
        passive_call=lambda m, d: calls.append(("native", m, d)),
    )
    env._step_graph = None
    env._step_graph_cache = {}

    env._install_mjw_passive_force_hook(
        "arch", lambda m, d: calls.append(("hook", m, d))
    )
    env._optimized_fwd_position_scratch.passive_call("m", "d")

    assert calls == [("native", "m", "d"), ("hook", "m", "d")]
    with pytest.raises(ValueError, match="only one"):
        env._install_mjw_passive_force_hook("second", lambda m, d: None)


def test_passive_force_hook_rejects_sleeping_models():
    env = _make_direct_rl_env_stub()
    env._optimized_fwd_position_scratch = SimpleNamespace(
        implicit_qh_registration_frozen=False,
        passive_call=lambda m, d: None,
        advance_call=lambda m, d, qacc: None,
    )
    env._step_graph = None
    env._step_graph_cache = {}

    with pytest.raises(RuntimeError, match="sleeping"):
        env._install_mjw_passive_force_hook("arch", lambda m, d: None)


def test_passive_force_hook_rejects_late_registration():
    env = _make_direct_rl_env_stub()
    env._optimized_fwd_position_scratch = SimpleNamespace(
        implicit_qh_registration_frozen=True,
        passive_call=lambda m, d: None,
    )
    env._step_graph = None
    env._step_graph_cache = {}

    with pytest.raises(RuntimeError, match="before stepping"):
        env._install_mjw_passive_force_hook("late", lambda m, d: None)


def test_contact_hook_runs_after_native_narrowphase_once():
    env = _make_direct_rl_env_stub()
    calls = []
    env._optimized_fwd_position_scratch = SimpleNamespace(
        implicit_qh_registration_frozen=False,
        collision_context=object(),
        constraint_free=False,
        collision_narrowphase_call=lambda m, d, ctx: calls.append(
            ("native", m, d, ctx)
        ),
    )
    env._step_graph = None
    env._step_graph_cache = {}

    env._install_mjw_contact_hook("skin", lambda m, d: calls.append(("hook", m, d)))
    env._optimized_fwd_position_scratch.collision_narrowphase_call("m", "d", "ctx")

    assert calls == [("native", "m", "d", "ctx"), ("hook", "m", "d")]
    with pytest.raises(ValueError, match="only one"):
        env._install_mjw_contact_hook("second", lambda m, d: None)


@pytest.mark.parametrize(
    ("scratch_kwargs", "message"),
    [
        ({"constraint_free": True}, "constraint rows"),
        ({"advance_call": lambda m, d, qacc: None}, "sleeping"),
        ({"implicit_qh_registration_frozen": True}, "before stepping"),
    ],
)
def test_contact_hook_rejects_unsupported_scratch(scratch_kwargs, message):
    env = _make_direct_rl_env_stub()
    scratch = {
        "implicit_qh_registration_frozen": False,
        "collision_context": object(),
        "constraint_free": False,
        "collision_narrowphase_call": lambda m, d, ctx: None,
    }
    scratch.update(scratch_kwargs)
    env._optimized_fwd_position_scratch = SimpleNamespace(**scratch)
    env._step_graph = None
    env._step_graph_cache = {}

    with pytest.raises(RuntimeError, match=message):
        env._install_mjw_contact_hook("skin", lambda m, d: None)


def _sparse_hessian_hook_stub(monkeypatch):
    """A stub env with a solver context, optimized kernels and a recording plan."""
    from instantmimic.utils.mjw_kernels import newton_fusion, sparse_hessian

    calls = []
    plan = SimpleNamespace(
        nv=8, nslots=20, nterms=30, levels=4, density=0.5, path="nonfused"
    )

    def plan_for_model(m, ctx, groups, block, **kwargs):
        calls.append(groups)
        return (None, "stub") if groups is None else (plan, "")

    monkeypatch.setattr(sparse_hessian, "plan_for_model", plan_for_model)
    monkeypatch.setattr(newton_fusion, "newton_block_dim", lambda m: 64)
    env = _make_direct_rl_env_stub()
    env._use_optimized_kernel = True
    env.model = SimpleNamespace(nbody=4)
    env.mjw_model = object()
    env._optimized_fwd_position_scratch = SimpleNamespace(
        implicit_qh_registration_frozen=False,
        collision_context=object(),
        constraint_free=False,
        collision_narrowphase_call=lambda m, d, ctx: None,
        solver_context=SimpleNamespace(sparse_hessian_plan="stale"),
    )
    env._step_graph = None
    env._step_graph_cache = {}
    return env, calls, plan


def test_contact_hook_body_groups_select_the_sparse_hessian(monkeypatch):
    env, calls, plan = _sparse_hessian_hook_stub(monkeypatch)
    ctx = env._optimized_fwd_position_scratch.solver_context

    env._install_mjw_contact_hook("skin", lambda m, d: None, body_groups=[(1, 2), [3]])

    assert calls == [[[1, 2], [3]]] and ctx.sparse_hessian_plan is plan


def test_contact_hook_without_body_groups_keeps_the_dense_factor(monkeypatch):
    env, calls, plan = _sparse_hessian_hook_stub(monkeypatch)
    ctx = env._optimized_fwd_position_scratch.solver_context

    env._install_mjw_contact_hook("skin", lambda m, d: None)

    assert calls == [None] and ctx.sparse_hessian_plan is None


def test_contact_hook_rejects_invalid_body_ids_before_installing(monkeypatch):
    env, calls, plan = _sparse_hessian_hook_stub(monkeypatch)
    scratch = env._optimized_fwd_position_scratch
    native = scratch.collision_narrowphase_call

    with pytest.raises(ValueError, match=r"invalid body ids: \[-1, 4\]"):
        env._install_mjw_contact_hook(
            "skin", lambda m, d: None, body_groups=[[1, 4], [-1, 2]]
        )

    assert scratch.collision_narrowphase_call is native and calls == []
    env._install_mjw_contact_hook("skin", lambda m, d: None, body_groups=[[1, 3]])
    assert calls == [[[1, 3]]] and scratch.solver_context.sparse_hessian_plan is plan


def test_sparse_hessian_violations_fail_the_metric_check(monkeypatch):
    """The fused path zeroes the violation counter on install; a nonzero count at the
    metric check raises and names the hook."""
    from instantmimic.utils.mjw_kernels import sparse_hessian

    env, calls, plan = _sparse_hessian_hook_stub(monkeypatch)
    plan.path = "fused"
    slots = direct_rl_env.wp.full(3, 5.0, dtype=float, device="cpu")
    monkeypatch.setattr(sparse_hessian, "violation_counter", lambda h, p: slots)
    env._optimized_fwd_position_scratch.solver_context.hfactor = object()

    env._install_mjw_contact_hook("skin", lambda m, d: None, body_groups=[[1, 3]])

    counter = env._sparse_hessian_violations
    assert counter is not None and not slots.numpy().any()
    assert env.consume_diagnostic_stats() == {}
    counter[1] = 2.0
    with pytest.raises(
        RuntimeError, match=r"2 constraint-row couplings in 1 world\(s\).*'skin'"
    ):
        env.consume_diagnostic_stats()
    assert not slots.numpy().any()


def test_nonfused_sparse_hessian_has_no_violation_check(monkeypatch):
    env, calls, plan = _sparse_hessian_hook_stub(monkeypatch)

    env._install_mjw_contact_hook("skin", lambda m, d: None, body_groups=[[1, 3]])

    ctx = env._optimized_fwd_position_scratch.solver_context
    assert ctx.sparse_hessian_plan is plan and env._sparse_hessian_violations is None
    assert env.consume_diagnostic_stats() == {}


def test_install_implicit_velocity_derivative_hook_rejects_missing_scratch():
    env = _make_direct_rl_env_stub()
    env._optimized_fwd_position_scratch = None
    env.mjw_model = SimpleNamespace(
        is_sparse=False,
        opt=SimpleNamespace(integrator=IntegratorType.IMPLICITFAST),
    )

    with pytest.raises(RuntimeError, match="derivative scratch"):
        env._install_mjw_implicit_velocity_derivative_hook(object())


def test_install_implicit_velocity_derivative_hook_rejects_full_implicit():
    env = _make_direct_rl_env_stub()
    env._optimized_fwd_position_scratch = SimpleNamespace(
        implicit_qderiv=object(),
        implicit_deriv_call=lambda: None,
    )
    env.mjw_model = SimpleNamespace(
        is_sparse=False,
        opt=SimpleNamespace(integrator=IntegratorType.IMPLICIT),
    )

    with pytest.raises(RuntimeError, match="implicit-fast integrator"):
        env._install_mjw_implicit_velocity_derivative_hook(object())


def test_run_mjw_state_prefix_uses_custom_allocation_free_path(monkeypatch):
    calls = []

    def fake_state_prefix(
        model, data, streams, scratch=None, *, use_optimized_kernel=None, constraints
    ):
        calls.append((model, data, streams, scratch, use_optimized_kernel, constraints))

    monkeypatch.setattr(
        direct_rl_env,
        "forward_state_prefix_optimized_fwd_position",
        fake_state_prefix,
    )
    env = _make_direct_rl_env_stub()
    env._use_parallelized_kernel_launch = False
    env._use_optimized_kernel = False
    env._optimized_fwd_position_streams = ()
    env._optimized_fwd_position_scratch = object()
    env.mjw_model = object()
    env.mjw_data = object()

    env._run_mjw_state_prefix()
    env._run_mjw_state_prefix(constraints=False)

    assert calls == [
        (
            env.mjw_model,
            env.mjw_data,
            (),
            env._optimized_fwd_position_scratch,
            False,
            constraints,
        )
        for constraints in (True, False)
    ]


def test_allocation_free_capture_support_does_not_require_parallel_streams():
    env = _make_direct_rl_env_stub()
    env._use_parallelized_kernel_launch = False
    env._optimized_fwd_position_scratch = SimpleNamespace(
        solver_context=object(),
        solver_nsolving=object(),
    )
    env.mjw_model = SimpleNamespace(nv=1)
    env.mjw_data = SimpleNamespace(njmax=1)

    assert env.supports_allocation_free_step_capture() is True


def test_allocation_free_capture_support_rejects_incomplete_implicit_scratch():
    env = _make_direct_rl_env_stub()
    env._use_optimized_kernel = False
    env._optimized_fwd_position_scratch = SimpleNamespace(
        solver_context=object(),
        solver_nsolving=object(),
        implicit_call=None,
    )
    env.mjw_model = SimpleNamespace(
        nv=1,
        opt=SimpleNamespace(
            integrator=IntegratorType.IMPLICITFAST,
            disableflags=0,
        ),
    )
    env.mjw_data = SimpleNamespace(njmax=1)

    assert env.supports_allocation_free_step_capture() is False

    env._optimized_fwd_position_scratch.implicit_call = lambda *_args: None

    assert env.supports_allocation_free_step_capture() is True


def test_optimized_capture_support_accepts_preallocated_implicit_fallback():
    env = _make_direct_rl_env_stub()
    env._use_optimized_kernel = True
    env._optimized_fwd_position_scratch = SimpleNamespace(
        solver_context=object(),
        solver_nsolving=object(),
        implicit_actuator_vel=None,
        implicit_qderiv_pair_mask=None,
        implicit_qderiv=object(),
        implicit_qacc=object(),
        implicit_max_moment_rownnz=0,
        implicit_qh_contributors=(),
        implicit_call=lambda *_args: None,
    )
    env.mjw_model = SimpleNamespace(
        nv=1,
        nu=0,
        is_sparse=False,
        opt=SimpleNamespace(
            integrator=IntegratorType.IMPLICITFAST,
            disableflags=0,
        ),
    )
    env.mjw_data = SimpleNamespace(njmax=1)

    assert env.supports_allocation_free_step_capture() is True


def test_run_mjw_step_passes_parallel_scratch_without_optimized_kernels(monkeypatch):
    calls = []

    def fake_step_optimized(
        model, data, streams, scratch=None, *, use_optimized_kernel=None
    ):
        calls.append((model, data, streams, scratch, use_optimized_kernel))

    monkeypatch.setattr(
        direct_rl_env, "step_optimized_fwd_position", fake_step_optimized
    )
    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(use_cuda_graph=False)
    env._compile_mjw_step = False
    env._use_parallelized_kernel_launch = True
    env._use_optimized_kernel = False
    env._optimized_fwd_position_streams = ("stream-a", "stream-b")
    env._optimized_fwd_position_scratch = object()
    env.mjw_model = object()
    env.mjw_data = object()

    env._run_mjw_step()

    assert calls == [
        (
            env.mjw_model,
            env.mjw_data,
            env._optimized_fwd_position_streams,
            env._optimized_fwd_position_scratch,
            False,
        )
    ]


def test_run_mjw_step_uses_optimized_kernel_scratch_when_enabled(monkeypatch):
    calls = []

    def fake_step_optimized(
        model, data, streams, scratch=None, *, use_optimized_kernel=None
    ):
        calls.append((model, data, streams, scratch, use_optimized_kernel))

    monkeypatch.setattr(
        direct_rl_env, "step_optimized_fwd_position", fake_step_optimized
    )
    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(use_cuda_graph=False)
    env._compile_mjw_step = False
    env._use_parallelized_kernel_launch = True
    env._use_optimized_kernel = True
    env._optimized_fwd_position_streams = ("stream-a", "stream-b")
    env._optimized_fwd_position_scratch = object()
    env.mjw_model = object()
    env.mjw_data = object()

    env._run_mjw_step()

    assert calls == [
        (
            env.mjw_model,
            env.mjw_data,
            env._optimized_fwd_position_streams,
            env._optimized_fwd_position_scratch,
            True,
        )
    ]


def test_run_mjw_step_uses_skip_optimized_when_both_flags_are_enabled(monkeypatch):
    calls = []

    def fake_step_skip_optimized(
        model, data, streams, scratch=None, *, use_optimized_kernel=None
    ):
        calls.append((model, data, streams, scratch, use_optimized_kernel))

    monkeypatch.setattr(
        direct_rl_env,
        "step_skip_kinematics_optimized_fwd_position",
        fake_step_skip_optimized,
    )
    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(use_cuda_graph=False)
    env._compile_mjw_step = False
    env._use_parallelized_kernel_launch = True
    env._use_optimized_kernel = False
    env._optimized_fwd_position_streams = ("stream-a", "stream-b")
    env._optimized_fwd_position_scratch = object()
    env.mjw_model = object()
    env.mjw_data = object()

    env._run_mjw_step(skip_kinematics=True)

    assert calls == [
        (
            env.mjw_model,
            env.mjw_data,
            env._optimized_fwd_position_streams,
            env._optimized_fwd_position_scratch,
            False,
        )
    ]


def test_run_mjw_step_uses_state_prefix_suffix_when_prefix_is_fresh(monkeypatch):
    calls = []

    def fake_step_after_state_prefix(
        model, data, streams, scratch=None, *, use_optimized_kernel=None
    ):
        calls.append((model, data, streams, scratch, use_optimized_kernel))

    monkeypatch.setattr(
        direct_rl_env,
        "step_after_state_prefix_optimized_fwd_position",
        fake_step_after_state_prefix,
    )
    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(use_cuda_graph=False)
    env._compile_mjw_step = False
    env._use_parallelized_kernel_launch = True
    env._use_optimized_kernel = True
    env._optimized_fwd_position_streams = ("stream-a", "stream-b")
    env._optimized_fwd_position_scratch = object()
    env.mjw_model = object()
    env.mjw_data = object()

    env._run_mjw_step(skip_state_prefix=True)

    assert calls == [
        (
            env.mjw_model,
            env.mjw_data,
            env._optimized_fwd_position_streams,
            env._optimized_fwd_position_scratch,
            True,
        )
    ]


def test_run_mjw_kinematics_uses_parallel_tail(monkeypatch):
    calls = []

    def fake_kinematics(model, data, streams):
        calls.append((model, data, streams))

    monkeypatch.setattr(direct_rl_env, "kinematics_parallel_tail", fake_kinematics)
    env = _make_direct_rl_env_stub()
    env._use_parallelized_kernel_launch = True
    env._optimized_fwd_position_streams = (
        "stream-a",
        "stream-b",
        "stream-c",
        "stream-d",
        "stream-e",
    )
    env.mjw_model = object()
    env.mjw_data = object()

    env._run_mjw_kinematics()

    assert calls == [
        (env.mjw_model, env.mjw_data, env._optimized_fwd_position_streams[:4])
    ]


def test_step_body_refreshes_complete_state_prefix_before_hooks_and_rewards(
    monkeypatch,
):
    events = []

    def fake_launch(*args, **kwargs):
        events.append("counter")

    monkeypatch.setattr(direct_rl_env.wp, "launch", fake_launch)

    env = _make_direct_rl_env_stub()
    env.cfg = SimpleNamespace(decimation=2, scene=SimpleNamespace(num_envs=1))
    env.episode_length_buf = torch.zeros(1, dtype=torch.int32)
    env.common_step_counter = torch.zeros(1, dtype=torch.int32)
    env.reset_buf = torch.zeros(1, dtype=torch.bool)
    env._substep_rewards_enabled = True
    env._prepare_substep_rewards = lambda: events.append("prepare")
    env._apply_action = lambda: events.append("apply")
    env._run_mjw_step = lambda *, skip_state_prefix=False: events.append(
        ("step", skip_state_prefix)
    )
    env._run_mjw_state_prefix = lambda *, constraints=True: events.append(
        "state_prefix" if constraints else "state_prefix_deferred"
    )
    env._post_physics_substep = lambda substep_idx: events.append(("post", substep_idx))
    env._control_pipeline = SimpleNamespace(
        post_physics_substep=lambda substep_idx: events.append(
            ("pipeline_post", substep_idx)
        )
    )
    env._post_state_prefix_substep = lambda substep_idx: events.append(
        ("post_prefix", substep_idx)
    )
    env._accumulate_substep_reward = lambda substep_idx: events.append(
        ("reward", substep_idx)
    )
    env._compute_dones_and_rewards = lambda: events.append("done_reward") or None
    env._accumulate_done_reason_stats = lambda nonfinite_state_mask: events.append(
        "done_stats"
    )
    env._reset_idx = lambda reset_mask: events.append("reset")
    observations = {"policy": torch.zeros(1, 1)}
    env.obs_buf = observations
    env._get_observations = lambda: events.append("obs") or observations
    env._render_runtime = SimpleNamespace(
        capture_substep_state=lambda substep_idx: events.append(
            ("capture", substep_idx)
        ),
        close=lambda: None,
    )

    DirectRLEnv._step_body(env)

    assert env.obs_buf is observations

    assert events[:15] == [
        "prepare",
        "apply",
        ("step", True),
        ("post", 0),
        "state_prefix",
        ("pipeline_post", 0),
        ("post_prefix", 0),
        ("reward", 0),
        ("capture", 0),
        "apply",
        ("step", True),
        ("post", 1),
        "state_prefix",
        ("pipeline_post", 1),
        ("post_prefix", 1),
    ]
    assert ("reward", 1) in events
    assert events.index(("reward", 1)) < events.index(("capture", 1))


class _DeclaredEnv(DirectRLEnv):
    boundary_reads_constraints = False


class _UndeclaredSubclassEnv(_DeclaredEnv):
    pass


def _boundary_schedule(
    env_cls, monkeypatch, *, reset_done=True, reads_geometry=False, hooks=False
):
    """Endpoint refreshes of one two-substep step and its reset prefix."""
    monkeypatch.setattr(direct_rl_env.wp, "launch", lambda *a, **k: None)
    monkeypatch.setattr(
        direct_rl_env, "state_prefix_can_defer_constraints", lambda m: True
    )
    events = []
    env = env_cls.__new__(env_cls)
    env._initialize_runtime_state()
    env._integrator_spec = DirectRLEnv._resolve_integrator_spec("implicitfast")
    env.cfg = SimpleNamespace(decimation=2, scene=SimpleNamespace(num_envs=1))
    env.mjw_model = object()
    env.episode_length_buf = torch.zeros(1, dtype=torch.int32)
    env.common_step_counter = torch.zeros(1, dtype=torch.int32)
    env.reset_buf = torch.zeros(1, dtype=torch.bool)
    env._substep_rewards_enabled = False
    if hooks:
        env._post_physics_substep_hooks.append(lambda substep_idx: None)
    env._apply_action = lambda: None
    env._run_mjw_step = lambda *, skip_state_prefix=False: events.append("step")
    env._run_mjw_state_prefix = lambda *, constraints=True: events.append(
        "prefix" if constraints else "deferred"
    )
    env._control_pipeline = SimpleNamespace(
        post_physics_substep=lambda substep_idx: None,
        post_physics_substep_reads_geometry=lambda: reads_geometry,
    )
    env._compute_dones_and_rewards = lambda: events.append("dones") or None
    env._accumulate_done_reason_stats = lambda mask: None
    env._reset_idx = lambda reset_mask: events.append("reset")
    observations = {"policy": torch.zeros(1, 1)}
    env.obs_buf = observations
    env._get_observations = lambda: observations
    env._render_runtime = SimpleNamespace(
        capture_substep_state=lambda substep_idx: None, close=lambda: None
    )
    DirectRLEnv._step_body(env, reset_done=reset_done)
    return events


def test_auto_reset_step_defers_last_endpoint_constraints_to_reset_prefix(
    monkeypatch,
):
    assert _boundary_schedule(_DeclaredEnv, monkeypatch) == [
        "step",
        "prefix",
        "step",
        "deferred",
        "dones",
        "reset",
        "prefix",
    ]


@pytest.mark.parametrize(
    ("env_cls", "options"),
    [
        (DirectRLEnv, {}),
        # The declaration is not inherited: a subclass may add contact reads.
        (_UndeclaredSubclassEnv, {}),
        # No reset prefix follows.
        (_DeclaredEnv, {"reset_done": False}),
        (_DeclaredEnv, {"reads_geometry": True}),
        (_DeclaredEnv, {"hooks": True}),
    ],
)
def test_boundary_endpoint_keeps_constraints_unless_declared_and_reset(
    monkeypatch, env_cls, options
):
    events = _boundary_schedule(env_cls, monkeypatch, **options)
    assert "deferred" not in events
    assert events[:4] == ["step", "prefix", "step", "prefix"]


def test_boundary_declaration_is_not_inherited():
    assert DirectRLEnv.boundary_reads_constraints
    assert not _DeclaredEnv.boundary_reads_constraints
    assert _UndeclaredSubclassEnv.boundary_reads_constraints


def test_constraint_deferral_needs_a_model_without_sleep_or_sensors():
    from mujoco_warp._src.types import EnableBit

    from instantmimic.utils.mjw_step_optimized import (
        state_prefix_can_defer_constraints,
    )

    def model(enableflags=0, nsensor=0):
        return SimpleNamespace(
            opt=SimpleNamespace(enableflags=enableflags, disableflags=0),
            nsensor=nsensor,
        )

    assert state_prefix_can_defer_constraints(model())
    assert not state_prefix_can_defer_constraints(model(enableflags=EnableBit.SLEEP))
    assert not state_prefix_can_defer_constraints(model(nsensor=1))


def test_run_mjw_com_state_uses_parallel_com_pos_vel(monkeypatch):
    calls = []

    def fake_com_state(model, data, streams):
        calls.append((model, data, streams))

    monkeypatch.setattr(direct_rl_env, "com_pos_vel_parallel", fake_com_state)
    env = _make_direct_rl_env_stub()
    env._use_parallelized_kernel_launch = True
    env._optimized_fwd_position_streams = (
        "stream-a",
        "stream-b",
        "stream-c",
        "stream-d",
        "stream-e",
    )
    env.mjw_model = object()
    env.mjw_data = object()

    env._run_mjw_com_state()

    assert calls == [(env.mjw_model, env.mjw_data, env._optimized_fwd_position_streams)]
