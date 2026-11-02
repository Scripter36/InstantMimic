from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp
from mujoco_warp._src import derivative

from instantmimic.envs.direct_rl_env import (
    _launch_stream_aligned_cuda_graph,
    _stream_aligned_cuda_graph_capture,
    make_stream_aligned_cuda_graph_streams,
)
from instantmimic.utils.mjw_step_optimized import (
    BroadphaseType,
    OptimizedFwdPositionScratch,
    com_pos_vel_parallel,
    deriv_smooth_vel_dense_actuator_centric,
    forward_optimized_fwd_position,
    forward_state_prefix_optimized_fwd_position,
    fwd_position_optimized,
    fwd_velocity_optimized,
    kinematics_parallel_tail,
    make_optimized_fwd_position_scratch,
    make_optimized_fwd_position_streams,
    step_after_state_prefix_optimized_fwd_position,
    step_optimized_fwd_position,
    step_skip_kinematics_optimized_fwd_position,
    transmission_with_scratch,
)


@wp.kernel(enable_backward=False)
def _clear_capture_condition(condition: wp.array(dtype=wp.int32)):
    condition[0] = 0


def _make_data_pair(
    nworld: int = 16,
    *,
    enable_sleep: bool = False,
    integrator: int | None = None,
    solver_type: int | None = None,
):
    mj_model = mujoco.MjModel.from_xml_path("data/humanoid/mjcf/humanoid_pd.xml")
    if enable_sleep:
        mj_model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    if integrator is not None:
        mj_model.opt.integrator = integrator
    if solver_type is not None:
        mj_model.opt.solver = solver_type
    model = mjw.put_model(mj_model)
    stock_data = mjw.put_data(
        mj_model,
        mujoco.MjData(mj_model),
        nworld=nworld,
        njmax=64,
        nconmax=8,
    )
    parallel_data = mjw.put_data(
        mj_model,
        mujoco.MjData(mj_model),
        nworld=nworld,
        njmax=64,
        nconmax=8,
    )
    return mj_model, model, stock_data, parallel_data


def _make_two_tree_sleep_data_pair(nworld: int = 2):
    mj_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option jacobian="dense"/>
          <worldbody>
            <body name="left" pos="-0.5 0 0">
              <joint type="hinge" limited="true" range="-10 10"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.4"
                    contype="0" conaffinity="0"/>
            </body>
            <body name="right" pos="0.5 0 0">
              <joint type="hinge" limited="true" range="-10 10"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.4"
                    contype="0" conaffinity="0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    mj_model.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    host = mujoco.MjData(mj_model)
    host.qpos[:] = [0.3, -0.2]
    mujoco.mj_forward(mj_model, host)
    model = mjw.put_model(mj_model)
    stock_data = mjw.put_data(mj_model, host, nworld=nworld, nconmax=0, njmax=8)
    custom_data = mjw.put_data(mj_model, host, nworld=nworld, nconmax=0, njmax=8)
    return model, stock_data, custom_data


def _actuator_edge_mjcf() -> str:
    return """
    <mujoco>
      <option timestep="0.01" integrator="implicitfast"/>
      <worldbody>
        <body name="link" pos="0 0 1">
          <joint name="hinge" type="hinge" actuatorgravcomp="true"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.5"/>
        </body>
      </worldbody>
      <actuator>
        <general name="limited" joint="hinge"
          dyntype="filterexact" dynprm="0.05"
          gaintype="affine" gainprm="1 0 0.2"
          biastype="affine" biasprm="0 0 -0.1"
          actearly="true" actlimited="true" actrange="-0.2 0.2"
          forcelimited="true" forcerange="-0.05 0.05"/>
      </actuator>
    </mujoco>
    """


def _delayed_actuator_mjcf() -> str:
    return """
    <mujoco>
      <option timestep="0.01" integrator="implicitfast"/>
      <worldbody>
        <body pos="0 0 1">
          <joint name="hinge" type="hinge"/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.5"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="delayed" joint="hinge" nsample="4" delay="0.02"/>
      </actuator>
    </mujoco>
    """


def _seed_state(mj_model, stock_data, parallel_data, nworld: int) -> None:
    torch.manual_seed(1234)
    qpos = (
        torch.as_tensor(mj_model.qpos0, device="cuda", dtype=torch.float32)
        .expand(nworld, -1)
        .clone()
    )
    qvel = torch.randn((nworld, mj_model.nv), device="cuda") * 0.01
    ctrl = torch.randn((nworld, mj_model.nu), device="cuda") * 0.05
    for data in (stock_data, parallel_data):
        wp.to_torch(data.qpos).copy_(qpos)
        wp.to_torch(data.qvel).copy_(qvel)
        wp.to_torch(data.ctrl).copy_(ctrl)


@pytest.mark.parametrize("inputs", ["none", "pos vel ff"])
@pytest.mark.parametrize("sensors_disabled", [False, True])
def test_forward_separates_actuator_count_from_control_storage(
    inputs, sensors_disabled
):
    host_model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco><option jacobian="dense"/>
          <worldbody><body><joint name="hinge"/>
            <geom size=".1" mass="1"/>
          </body></worldbody>
          <actuator><dcmotor joint="hinge" input="{inputs}"
            motorconst="1" resistance="1"/></actuator>
        </mujoco>
        """
    )
    if sensors_disabled:
        host_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    host = mujoco.MjData(host_model)
    host.qpos[:] = 0.3
    host.qvel[:] = 0.2
    host.ctrl[:] = np.arange(host_model.nu) * 0.1
    m = mjw.put_model(host_model)
    m.opt.run_rne_postconstraint = True
    stock = mjw.put_data(host_model, host, nworld=2, nconmax=4, njmax=8)
    custom = mjw.put_data(host_model, host, nworld=2, nconmax=4, njmax=8)
    scratch = make_optimized_fwd_position_scratch(
        m, custom, "cpu", include_optimized_kernel_scratch=False
    )

    mjw.forward(m, stock)
    forward_optimized_fwd_position(m, custom, (), scratch, use_optimized_kernel=False)
    wp.synchronize()

    for name in ("actuator_force", "qfrc_actuator", "qacc", "cfrc_ext", "cfrc_int"):
        np.testing.assert_allclose(
            getattr(custom, name).numpy(), getattr(stock, name).numpy(), atol=1e-6
        )
    if inputs == "none":
        assert np.any(custom.actuator_force.numpy() != 0.0)


def test_contact_sensor_stages_reuse_scratch_and_match_stock(monkeypatch):
    host_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><option jacobian="dense" iterations="4"/>
          <worldbody>
            <geom type="plane" size="1 1 .1"/>
            <body pos="0 0 .09"><freejoint/><geom size=".1" mass="1"/></body>
          </worldbody>
          <sensor>
            <contact data="found dist pos normal tangent" reduce="mindist"/>
            <contact data="found force torque"/>
          </sensor>
        </mujoco>
        """
    )
    m = mjw.put_model(host_model)
    assert m.sensor_contact_pos_adr.size == m.sensor_contact_acc_adr.size == 1
    host = mujoco.MjData(host_model)
    stock = mjw.put_data(host_model, host, nworld=2, nconmax=4, njmax=32)
    custom = mjw.put_data(host_model, host, nworld=2, nconmax=4, njmax=32)
    scratch = make_optimized_fwd_position_scratch(
        m, custom, "cpu", include_optimized_kernel_scratch=False
    )
    mjw.forward(m, stock)
    forward_optimized_fwd_position(m, custom, (), scratch, use_optimized_kernel=False)
    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "full", "clone"):
            patch.setattr(
                wp, name, lambda *a, **kw: pytest.fail("sensor forward allocated")
            )
        forward_optimized_fwd_position(
            m, custom, (), scratch, use_optimized_kernel=False
        )
    wp.synchronize()
    assert np.any(stock.sensordata.numpy() != 0.0)
    np.testing.assert_allclose(
        custom.sensordata.numpy(), stock.sensordata.numpy(), atol=1e-5
    )


def test_optimized_fwd_position_starts_crb_after_com_pos_before_constraint(
    monkeypatch,
):
    calls = []

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(name)

        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.smooth, "com_pos", mark("com_pos"))
    monkeypatch.setattr(optimized, "com_pos_parallel_tail", mark("com_pos"))
    monkeypatch.setattr(optimized.smooth, "camlight", mark("camlight"))
    monkeypatch.setattr(optimized.smooth, "flex", mark("flex"))
    monkeypatch.setattr(optimized.smooth, "tendon", mark("tendon"))
    monkeypatch.setattr(optimized.smooth, "crb", mark("crb"))
    monkeypatch.setattr(optimized.smooth, "tendon_armature", mark("tendon_armature"))
    monkeypatch.setattr(optimized.smooth, "transmission", mark("transmission"))
    monkeypatch.setattr(
        optimized.constraint, "make_constraint", mark("make_constraint")
    )
    monkeypatch.setattr(
        optimized.island,
        "island",
        lambda *_args: pytest.fail("sleep-off forward must not discover islands"),
    )

    model = SimpleNamespace(
        opt=SimpleNamespace(
            run_collision_detection=False,
            enableflags=0,
            disableflags=0,
        ),
        nacttrnbody=0,
        ntendon=0,
        neq=0,
        ntree=2,
    )
    fwd_position_optimized(
        model,
        object(),
        streams=None,
        factorize=False,
        skip_kinematics=True,
    )

    assert calls.index("transmission") < calls.index("com_pos")
    assert calls.index("com_pos") < calls.index("crb")
    assert calls.index("crb") < calls.index("make_constraint")
    assert calls.index("tendon") < calls.index("tendon_armature")


def test_optimized_fwd_position_uses_preallocated_tendon_call(monkeypatch):
    calls = []

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(name)

        return _inner

    def fail_stock_tendon(*_args, **_kwargs):
        raise AssertionError("tendon geometry must use persistent scratch")

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "com_pos_parallel_tail", mark("com_pos"))
    monkeypatch.setattr(optimized.smooth, "camlight", mark("camlight"))
    monkeypatch.setattr(optimized.smooth, "flex", mark("flex"))
    monkeypatch.setattr(optimized.smooth, "tendon", fail_stock_tendon)
    monkeypatch.setattr(optimized.smooth, "crb", mark("crb"))
    monkeypatch.setattr(optimized.smooth, "tendon_armature", mark("armature"))
    monkeypatch.setattr(optimized, "transmission_with_scratch", mark("transmission"))

    model = SimpleNamespace(
        opt=SimpleNamespace(run_collision_detection=False),
        nacttrnbody=0,
    )
    scratch = SimpleNamespace(
        tendon_call=mark("preallocated_tendon"),
        constraint_call=mark("constraint"),
    )

    fwd_position_optimized(
        model,
        object(),
        streams=None,
        scratch=scratch,
        factorize=False,
        skip_kinematics=True,
    )

    assert "preallocated_tendon" in calls


@pytest.mark.parametrize("constraints", [True, False])
def test_optimized_fwd_position_enqueues_collision_before_constraint(
    monkeypatch, constraints
):
    calls = []

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append(("main", getattr(kernel, "__name__", str(kernel))))

    def fake_launch_in_stream(stream, kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append((stream, getattr(kernel, "__name__", str(kernel))))

    def fake_call_in_stream(stream, fn, *args):
        fn(*args)

    def fake_call_in_stream_after(stream, _wait_streams, fn, *args):
        fn(*args)

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(("main", name))

        _inner.__name__ = name
        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)
    monkeypatch.setattr(optimized, "_launch_in_stream", fake_launch_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream", fake_call_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream_after", fake_call_in_stream_after)
    monkeypatch.setattr(optimized, "_wait_for_streams", lambda _streams: None)
    monkeypatch.setattr(optimized.smooth, "camlight", mark("camlight"))
    monkeypatch.setattr(optimized.smooth, "flex", mark("flex"))
    monkeypatch.setattr(optimized.smooth, "tendon", mark("tendon"))
    monkeypatch.setattr(optimized.smooth, "crb", mark("crb"))
    monkeypatch.setattr(optimized.smooth, "tendon_armature", mark("tendon_armature"))
    monkeypatch.setattr(optimized.smooth, "transmission", mark("transmission"))
    monkeypatch.setattr(optimized.collision_driver, "collision", mark("collision"))
    monkeypatch.setattr(
        optimized.constraint, "make_constraint", mark("make_constraint")
    )

    model = SimpleNamespace(
        opt=SimpleNamespace(run_collision_detection=True),
        all_actuator_trntypes_are_joint=True,
        body_tree=(),
        nbody=1,
        njnt=1,
        nacttrnbody=0,
        ntendon=0,
        nsite=0,
        body_mass=object(),
        body_parentid=object(),
        body_subtreemass=object(),
        body_rootid=object(),
        body_inertia=object(),
        jnt_type=object(),
        jnt_dofadr=object(),
        jnt_bodyid=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        xipos=object(),
        subtree_com=object(),
        ximat=object(),
        xmat=object(),
        xanchor=object(),
        xaxis=object(),
        cinert=object(),
        cdof=object(),
    )

    fwd_position_optimized(
        model,
        data,
        streams=tuple(f"stream-{i}" for i in range(12)),
        factorize=False,
        skip_kinematics=True,
        constraints=constraints,
    )

    assert ("stream-7", "_cinert") in calls
    assert ("stream-8", "_cdof") in calls
    assert ("main", "transmission") in calls
    if not constraints:
        # Deferred: neither stage runs; contacts and rows stay as they were.
        assert ("main", "collision") not in calls
        assert ("main", "make_constraint") not in calls
        return
    assert calls.index(("main", "crb")) < calls.index(("main", "collision"))
    assert calls.index(("main", "collision")) < calls.index(("main", "make_constraint"))


def test_deferred_constraints_reject_sleeping_models():
    import instantmimic.utils.mjw_step_optimized as optimized

    model = SimpleNamespace(opt=SimpleNamespace(enableflags=optimized.EnableBit.SLEEP))
    with pytest.raises(ValueError, match="sleeping models"):
        fwd_position_optimized(model, SimpleNamespace(), None, constraints=False)


@pytest.mark.parametrize("contactfilter_enabled", [False, True])
def test_optimized_fwd_position_runs_crb_branch_on_dedicated_stream(
    monkeypatch,
    contactfilter_enabled,
):
    calls = []

    def fake_call_in_stream(stream, fn, *args):
        calls.append((stream, getattr(fn, "__name__", str(fn))))

    def fake_call_in_stream_after(stream, wait_streams, fn, *args):
        calls.append((stream, tuple(wait_streams), getattr(fn, "__name__", str(fn))))
        fn(*args)

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(("main", name))

        _inner.__name__ = name
        return _inner

    def fake_start_velocity(*_args, **_kwargs):
        calls.append(("pipeline_velocity",))
        return optimized.PendingComVelocity(started=True, stream="stream-14")

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "_call_in_stream", fake_call_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream_after", fake_call_in_stream_after)
    monkeypatch.setattr(optimized, "com_pos_parallel_tail", mark("com_pos"))
    monkeypatch.setattr(
        optimized,
        "_wait_for_streams",
        lambda streams: calls.append(("wait", tuple(streams))),
    )
    monkeypatch.setattr(optimized, "_start_pipelined_com_velocity", fake_start_velocity)
    monkeypatch.setattr(optimized.smooth, "camlight", mark("camlight"))
    monkeypatch.setattr(optimized.smooth, "flex", mark("flex"))
    monkeypatch.setattr(optimized.smooth, "tendon", mark("tendon"))
    monkeypatch.setattr(optimized.smooth, "crb", mark("crb"))
    monkeypatch.setattr(optimized.smooth, "tendon_armature", mark("tendon_armature"))
    monkeypatch.setattr(optimized.smooth, "factor_m", mark("factor_m"))
    monkeypatch.setattr(optimized.smooth, "transmission", mark("transmission"))
    monkeypatch.setattr(optimized.collision_driver, "collision", mark("collision"))
    monkeypatch.setattr(
        optimized.constraint, "make_constraint", mark("make_constraint")
    )

    model = SimpleNamespace(
        opt=SimpleNamespace(run_collision_detection=True),
        callback=SimpleNamespace(
            contactfilter=(lambda *_args: None) if contactfilter_enabled else None,
            sensor=None,
        ),
        all_actuator_trntypes_are_joint=True,
        nacttrnbody=0,
    )
    streams = tuple(f"stream-{i}" for i in range(15))

    fwd_position_optimized(
        model,
        object(),
        streams=streams,
        factorize=True,
        skip_kinematics=True,
        pipeline_velocity=True,
    )

    crb_stream = "stream-13"
    constraint_stream = "stream-4"
    assert (crb_stream, "crb") in calls
    assert (crb_stream, "tendon_armature") in calls
    assert (crb_stream, "factor_m") in calls
    collision_call = (
        (constraint_stream, (crb_stream,), "_run_collision_sequence")
        if contactfilter_enabled
        else (constraint_stream, "_run_collision_sequence")
    )
    constraint_call = (constraint_stream, (crb_stream,), "make_constraint")
    assert collision_call in calls
    assert constraint_call in calls
    assert calls.index((crb_stream, "crb")) < calls.index(constraint_call)
    if contactfilter_enabled:
        assert calls.index(("wait", (constraint_stream,))) < calls.index(
            ("pipeline_velocity",)
        )
    else:
        assert calls.index(("pipeline_velocity",)) < calls.index((crb_stream, "crb"))


def test_position_velocity_pipeline_rejects_sensor_callback():
    import instantmimic.utils.mjw_step_optimized as optimized

    streams = tuple(f"stream-{i}" for i in range(15))
    model = SimpleNamespace(
        opt=SimpleNamespace(disableflags=0, enableflags=0),
        callback=SimpleNamespace(sensor=lambda *_args: None),
    )

    assert not optimized._can_pipeline_position_velocity(model, streams)


def test_constraint_velocity_dependency_is_limited_to_enabled_connect_and_weld():
    import instantmimic.utils.mjw_step_optimized as optimized

    model = SimpleNamespace(
        opt=SimpleNamespace(disableflags=0),
        callback=SimpleNamespace(contactfilter=None),
        eq_connect_adr=SimpleNamespace(size=0),
        eq_wld_adr=SimpleNamespace(size=0),
    )
    assert not optimized._constraint_reads_com_velocity(model)

    model.eq_connect_adr.size = 1
    assert optimized._constraint_reads_com_velocity(model)

    model.opt.disableflags = optimized.DisableBit.EQUALITY
    assert not optimized._constraint_reads_com_velocity(model)

    model.callback.contactfilter = lambda *_args: None
    assert optimized._constraint_reads_com_velocity(model)


def test_optimized_fwd_position_preserves_v312_sleep_collision_island_schedule(
    monkeypatch,
):
    calls = []

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(name)

        _inner.__name__ = name
        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "com_pos_parallel_tail", mark("com_pos"))
    monkeypatch.setattr(optimized.smooth, "camlight", mark("camlight"))
    monkeypatch.setattr(optimized.smooth, "flex", mark("flex"))
    monkeypatch.setattr(optimized.smooth, "tendon", mark("tendon"))
    monkeypatch.setattr(optimized.smooth, "crb", mark("crb"))
    monkeypatch.setattr(optimized.smooth, "tendon_armature", mark("tendon_armature"))
    monkeypatch.setattr(optimized.smooth, "factor_m", mark("factor_m"))
    monkeypatch.setattr(optimized.smooth, "transmission", mark("transmission"))
    monkeypatch.setattr(optimized.collision_driver, "collision", mark("collision"))
    monkeypatch.setattr(optimized.wp, "clone", lambda _value: "awake-snapshot")
    monkeypatch.setattr(
        optimized.constraint,
        "make_constraint",
        mark("make_constraint"),
    )
    monkeypatch.setattr(
        optimized,
        "sleep",
        SimpleNamespace(
            wake_tendon=mark("wake_tendon"),
            update_sleep_trees=mark("update_sleep_trees"),
            wake_collision=mark("wake_collision"),
            update_sleep=mark("update_sleep"),
            wake_equality=mark("wake_equality"),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        optimized,
        "island",
        SimpleNamespace(island=mark("island")),
        raising=False,
    )

    model = SimpleNamespace(
        opt=SimpleNamespace(
            run_collision_detection=True,
            disableflags=0,
            enableflags=optimized.EnableBit.SLEEP,
        ),
        all_actuator_trntypes_are_joint=False,
        nacttrnbody=1,
        ntendon=1,
        neq=1,
        ntree=2,
    )

    fwd_position_optimized(
        model,
        SimpleNamespace(body_awake="awake"),
        streams=None,
        factorize=True,
        skip_kinematics=True,
    )

    collision_indices = [i for i, call in enumerate(calls) if call == "collision"]
    update_sleep_indices = [i for i, call in enumerate(calls) if call == "update_sleep"]

    assert len(collision_indices) == 2
    assert len(update_sleep_indices) == 2
    assert calls.index("tendon") < calls.index("wake_tendon")
    assert calls.index("wake_tendon") < calls.index("update_sleep_trees")
    assert calls.index("update_sleep_trees") < calls.index("crb")
    assert calls.index("factor_m") < collision_indices[0]
    assert collision_indices[0] < calls.index("wake_collision")
    assert calls.index("wake_collision") < update_sleep_indices[0]
    assert update_sleep_indices[0] < collision_indices[1]
    assert collision_indices[1] < calls.index("make_constraint")
    assert calls.index("make_constraint") < calls.index("wake_equality")
    assert calls.index("wake_equality") < update_sleep_indices[1]
    assert update_sleep_indices[1] < calls.index("island")
    assert calls.index("island") < calls.index("transmission")


def test_fwd_position_skip_kinematics_preserves_v312_sleep_island_schedule(
    monkeypatch,
):
    calls = []

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(name)

        _inner.__name__ = name
        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.smooth, "com_pos", mark("com_pos"))
    monkeypatch.setattr(optimized.smooth, "camlight", mark("camlight"))
    monkeypatch.setattr(optimized.smooth, "flex", mark("flex"))
    monkeypatch.setattr(optimized.smooth, "tendon", mark("tendon"))
    monkeypatch.setattr(optimized.smooth, "crb", mark("crb"))
    monkeypatch.setattr(optimized.smooth, "tendon_armature", mark("tendon_armature"))
    monkeypatch.setattr(optimized.smooth, "factor_m", mark("factor_m"))
    monkeypatch.setattr(optimized.smooth, "transmission", mark("transmission"))
    monkeypatch.setattr(optimized.collision_driver, "collision", mark("collision"))
    monkeypatch.setattr(optimized.wp, "clone", lambda _value: "awake-snapshot")
    monkeypatch.setattr(
        optimized.constraint,
        "make_constraint",
        mark("make_constraint"),
    )
    monkeypatch.setattr(
        optimized,
        "sleep",
        SimpleNamespace(
            wake_tendon=mark("wake_tendon"),
            update_sleep_trees=mark("update_sleep_trees"),
            wake_collision=mark("wake_collision"),
            update_sleep=mark("update_sleep"),
            wake_equality=mark("wake_equality"),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        optimized,
        "island",
        SimpleNamespace(island=mark("island")),
        raising=False,
    )

    model = SimpleNamespace(
        opt=SimpleNamespace(
            run_collision_detection=True,
            disableflags=0,
            enableflags=optimized.EnableBit.SLEEP,
        ),
        ntendon=1,
        neq=1,
        ntree=2,
    )

    optimized.fwd_position_skip_kinematics(
        model,
        SimpleNamespace(body_awake="awake"),
        factorize=True,
    )

    collision_indices = [i for i, call in enumerate(calls) if call == "collision"]
    update_sleep_indices = [i for i, call in enumerate(calls) if call == "update_sleep"]

    assert len(collision_indices) == 2
    assert len(update_sleep_indices) == 2
    assert calls.index("tendon") < calls.index("wake_tendon")
    assert calls.index("update_sleep_trees") < calls.index("crb")
    assert calls.index("factor_m") < collision_indices[0]
    assert collision_indices[0] < calls.index("wake_collision")
    assert update_sleep_indices[0] < collision_indices[1]
    assert collision_indices[1] < calls.index("make_constraint")
    assert calls.index("wake_equality") < update_sleep_indices[1]
    assert update_sleep_indices[1] < calls.index("island")
    assert calls.index("island") < calls.index("transmission")


@pytest.mark.parametrize(
    "forward_name",
    [
        "forward_optimized_fwd_position",
        "forward_skip_kinematics_optimized_fwd_position",
        "forward_skip_kinematics",
    ],
)
def test_custom_forward_preserves_v312_sleep_wake_before_position(
    monkeypatch,
    forward_name,
):
    calls = []

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(name)

        _inner.__name__ = name
        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(
        optimized,
        "sleep",
        SimpleNamespace(wake=mark("wake"), update_sleep=mark("update_sleep")),
        raising=False,
    )
    monkeypatch.setattr(
        optimized,
        "fwd_position_optimized",
        mark("fwd_position_optimized"),
    )
    monkeypatch.setattr(
        optimized,
        "fwd_position_skip_kinematics",
        mark("fwd_position_skip_kinematics"),
    )
    monkeypatch.setattr(
        optimized,
        "_forward_after_position",
        mark("forward_after_position"),
    )

    model = SimpleNamespace(
        opt=SimpleNamespace(
            disableflags=0,
            enableflags=optimized.EnableBit.SLEEP,
        )
    )

    if forward_name == "forward_skip_kinematics":
        getattr(optimized, forward_name)(model, object())
    else:
        getattr(optimized, forward_name)(model, object(), streams=None)

    position_index = min(
        i
        for i, call in enumerate(calls)
        if call in {"fwd_position_optimized", "fwd_position_skip_kinematics"}
    )
    assert calls[:position_index] == ["wake", "update_sleep"]


def test_custom_integrate_delegates_full_implicit_to_stock_implicit(monkeypatch):
    calls = []

    def fake_implicit(*_args, **_kwargs):
        calls.append("implicit")

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.mjw_forward, "implicit", fake_implicit)

    model = SimpleNamespace(
        opt=SimpleNamespace(integrator=optimized.IntegratorType.IMPLICIT)
    )

    optimized._integrate(model, object())

    assert calls == ["implicit"]


def test_dense_implicit_derivative_invokes_external_velocity_hook(monkeypatch):
    calls = []

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(
        optimized.derivative,
        "_qderiv_actuator_passive",
        object(),
    )
    monkeypatch.setattr(optimized.wp, "launch", lambda *_args, **_kwargs: None)

    scratch = optimized.OptimizedFwdPositionScratch(
        collision_context=object(),
        transmission_moment_nnz=object(),
        implicit_qderiv_pair_mask=None,
        implicit_qderiv=object(),
        implicit_qacc=object(),
        implicit_qh_contributors=(
            optimized.ImplicitQHContributor("test", lambda out: calls.append(out)),
        ),
    )
    model = SimpleNamespace(
        has_fluid=False,
        is_sparse=False,
        nactuator=0,
        nu=0,
        nv=1,
        ntendon=0,
        opt=SimpleNamespace(disableflags=0, timestep=0.01),
        M_fullm_i=SimpleNamespace(size=1),
        M_fullm_j=SimpleNamespace(size=1),
        dof_damping=object(),
        dof_dampingpoly=object(),
        M_elemid=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qvel=object(),
        M=object(),
    )
    out = SimpleNamespace(zero_=lambda: None)

    optimized.deriv_smooth_vel_dense_actuator_centric(model, data, out, scratch)

    assert calls == [out]


def test_implicit_qh_contributors_are_additive_ordered_and_exactly_once():
    import instantmimic.utils.mjw_step_optimized as optimized

    calls = []
    qh = {"value": 10.0}

    def add_two(target):
        calls.append("first")
        target["value"] += 2.0

    def triple(target):
        calls.append("second")
        target["value"] *= 3.0

    scratch = SimpleNamespace(
        implicit_qh_contributors=(
            optimized.ImplicitQHContributor("first", add_two),
            optimized.ImplicitQHContributor("second", triple),
        )
    )

    optimized._apply_implicit_qh_contributors(scratch, qh)

    assert calls == ["first", "second"]
    assert qh["value"] == 36.0


@pytest.mark.parametrize("use_optimized_kernel", [False, True])
def test_implicit_contributor_scratch_is_capture_safe_without_native_derivatives(
    use_optimized_kernel,
):
    import instantmimic.utils.mjw_step_optimized as optimized

    model = SimpleNamespace(
        is_sparse=True,
        nv=2,
        nactuator=0,
        nu=0,
        opt=SimpleNamespace(
            integrator=optimized.IntegratorType.IMPLICITFAST,
            disableflags=(
                optimized.DisableBit.ACTUATION
                | optimized.DisableBit.SPRING
                | optimized.DisableBit.DAMPER
            ),
            graph_conditional=False,
        ),
    )
    scratch = SimpleNamespace(
        implicit_qh_contributors=(
            optimized.ImplicitQHContributor("contact", lambda _qh: None),
        ),
        implicit_deriv_call=lambda *_args: None,
        implicit_qderiv=object(),
        implicit_qld=object(),
        implicit_ql_diag_inv=object(),
        implicit_qacc=object(),
    )

    assert optimized.supports_allocation_free_implicit_capture(
        model,
        scratch,
        use_optimized_kernel=use_optimized_kernel,
    )


def test_fwd_position_starts_com_pos_before_waiting_for_geom_site_tails(
    monkeypatch,
):
    calls = []

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append(("launch", getattr(kernel, "__name__", str(kernel))))

    def fake_launch_in_stream(stream, kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append(
            ("launch_stream", stream, getattr(kernel, "__name__", str(kernel)))
        )

    def fake_launch_in_stream_after(
        stream,
        wait_streams,
        kernel,
        *,
        dim,
        inputs,
        outputs,
        **_kwargs,
    ):
        calls.append(
            (
                "launch_stream_after",
                stream,
                tuple(wait_streams),
                getattr(kernel, "__name__", str(kernel)),
            )
        )

    def fake_call_in_stream(stream, fn, *args):
        calls.append(("call_stream", stream, getattr(fn, "__name__", str(fn))))
        fn(*args)

    def fake_call_in_stream_after(stream, wait_streams, fn, *args):
        calls.append(
            (
                "call_stream_after",
                stream,
                tuple(wait_streams),
                getattr(fn, "__name__", str(fn)),
            )
        )
        fn(*args)

    def fake_wait_for_streams(streams):
        calls.append(("wait_for_streams", tuple(streams)))

    def fake_wait_stream(stream):
        calls.append(("wait_stream", stream))

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(("call", name))

        _inner.__name__ = name
        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)
    monkeypatch.setattr(optimized.wp, "wait_stream", fake_wait_stream)
    monkeypatch.setattr(optimized, "_launch_in_stream", fake_launch_in_stream)
    monkeypatch.setattr(
        optimized,
        "_launch_in_stream_after",
        fake_launch_in_stream_after,
        raising=False,
    )
    monkeypatch.setattr(optimized, "_call_in_stream", fake_call_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream_after", fake_call_in_stream_after)
    monkeypatch.setattr(optimized, "_wait_for_streams", fake_wait_for_streams)
    monkeypatch.setattr(optimized.collision_driver, "collision", mark("collision"))
    monkeypatch.setattr(optimized.smooth, "camlight", mark("camlight"))
    monkeypatch.setattr(optimized.smooth, "flex", mark("flex"))
    monkeypatch.setattr(optimized.smooth, "tendon", mark("tendon"))
    monkeypatch.setattr(optimized.smooth, "crb", mark("crb"))
    monkeypatch.setattr(optimized.smooth, "tendon_armature", mark("tendon_armature"))
    monkeypatch.setattr(optimized.smooth, "transmission", mark("transmission"))
    monkeypatch.setattr(
        optimized.constraint, "make_constraint", mark("make_constraint")
    )

    model = SimpleNamespace(
        opt=SimpleNamespace(run_collision_detection=True),
        all_actuator_trntypes_are_joint=True,
        nacttrnbody=0,
        ntendon=0,
        nbody=1,
        nbranch=1,
        ngeom=1,
        nsite=1,
        njnt=1,
        body_tree=(),
        qpos0=object(),
        body_parentid=object(),
        body_mocapid=object(),
        body_jntnum=object(),
        body_jntadr=object(),
        body_pos=object(),
        body_quat=object(),
        body_ipos=object(),
        body_iquat=object(),
        body_rootid=object(),
        body_weldid=object(),
        body_mass=object(),
        body_subtreemass=object(),
        body_inertia=object(),
        jnt_type=object(),
        jnt_qposadr=object(),
        jnt_dofadr=object(),
        jnt_pos=object(),
        jnt_axis=object(),
        jnt_bodyid=object(),
        body_branches=object(),
        body_branch_start=object(),
        geom_bodyid=object(),
        geom_pos=object(),
        geom_quat=object(),
        site_bodyid=object(),
        site_pos=object(),
        site_quat=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qpos=object(),
        mocap_pos=object(),
        mocap_quat=object(),
        xpos=object(),
        xquat=object(),
        xanchor=object(),
        xaxis=object(),
        xmat=object(),
        xipos=object(),
        ximat=object(),
        geom_xpos=object(),
        geom_xmat=object(),
        site_xpos=object(),
        site_xmat=object(),
        subtree_com=object(),
        cinert=object(),
        cdof=object(),
    )
    streams = tuple(f"stream-{i}" for i in range(14))

    fwd_position_optimized(
        model,
        data,
        streams=streams,
        factorize=False,
        skip_kinematics=False,
    )

    subtree_idx = calls.index(("launch", "_subtree_com_init"))
    geom_site_waits = [
        idx
        for idx, call in enumerate(calls)
        if call[0] == "wait_for_streams"
        and ("stream-2" in call[1] or "stream-3" in call[1])
    ]
    assert geom_site_waits
    assert subtree_idx < geom_site_waits[0]
    assert ("call_stream", "stream-4", "_run_collision_sequence") in calls
    assert (
        "call_stream_after",
        "stream-4",
        ("stream-13",),
        "make_constraint",
    ) in calls


def test_optimized_fwd_position_uses_preallocated_scratch(monkeypatch):
    calls = []

    def fail_stock_collision(*_args):
        msg = "optimized scratch path should not allocate stock collision context"
        raise AssertionError(msg)

    def fail_stock_transmission(*_args):
        msg = "optimized scratch path should not allocate transmission scratch"
        raise AssertionError(msg)

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(name)

        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    scratch = SimpleNamespace(collision_context=object())
    monkeypatch.setattr(optimized.collision_driver, "collision", fail_stock_collision)
    monkeypatch.setattr(optimized.smooth, "transmission", fail_stock_transmission)
    monkeypatch.setattr(
        optimized,
        "collision_with_context",
        mark("collision_with_context"),
        raising=False,
    )
    monkeypatch.setattr(
        optimized,
        "transmission_with_scratch",
        mark("transmission_with_scratch"),
        raising=False,
    )
    monkeypatch.setattr(optimized, "com_pos_parallel_tail", mark("com_pos"))
    monkeypatch.setattr(optimized.smooth, "camlight", mark("camlight"))
    monkeypatch.setattr(optimized.smooth, "flex", mark("flex"))
    monkeypatch.setattr(optimized.smooth, "tendon", mark("tendon"))
    monkeypatch.setattr(optimized.smooth, "crb", mark("crb"))
    monkeypatch.setattr(optimized.smooth, "tendon_armature", mark("tendon_armature"))
    monkeypatch.setattr(
        optimized.constraint, "make_constraint", mark("make_constraint")
    )
    monkeypatch.setattr(optimized, "_wait_for_streams", lambda _streams: None)

    model = SimpleNamespace(
        opt=SimpleNamespace(run_collision_detection=True),
        all_actuator_trntypes_are_joint=True,
        nacttrnbody=0,
    )

    fwd_position_optimized(
        model,
        object(),
        streams=None,
        scratch=scratch,
        factorize=False,
        skip_kinematics=True,
    )

    assert "collision_with_context" in calls
    assert "transmission_with_scratch" in calls


def test_transmission_with_scratch_matches_312_signature(monkeypatch):
    captured = {}

    class ScratchArray:
        def zero_(self):
            captured["zeroed"] = True

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        captured["kernel"] = kernel
        captured["inputs"] = inputs
        captured["outputs"] = outputs
        captured["dim"] = dim

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)

    kernel = SimpleNamespace(
        adj=SimpleNamespace(
            args=[
                SimpleNamespace(label="body_isdofancestor"),
            ]
        )
    )
    monkeypatch.setattr(optimized.smooth, "_transmission", kernel)

    model = SimpleNamespace(
        nv=1,
        nactuator=1,
        nu=1,
        nacttrnbody=0,
        body_parentid=object(),
        body_rootid=object(),
        body_weldid=object(),
        body_dofnum=object(),
        body_dofadr=object(),
        jnt_type=object(),
        jnt_qposadr=object(),
        jnt_dofadr=object(),
        dof_bodyid=object(),
        dof_parentid=object(),
        site_bodyid=object(),
        site_quat=object(),
        ten_J_rownnz=object(),
        ten_J_rowadr=object(),
        ten_J_colind=object(),
        actuator_trntype=object(),
        actuator_trnid=object(),
        actuator_gear=object(),
        actuator_cranklength=object(),
        body_isdofancestor=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qpos=object(),
        xquat=object(),
        site_xpos=object(),
        site_xmat=object(),
        subtree_com=object(),
        cdof=object(),
        ten_J=object(),
        ten_length=object(),
        actuator_length=object(),
        moment_rownnz=object(),
        moment_rowadr=object(),
        moment_colind=object(),
        actuator_moment=object(),
    )
    scratch = SimpleNamespace(transmission_moment_nnz=ScratchArray())

    transmission_with_scratch(model, data, scratch)

    assert captured["zeroed"] is True
    assert captured["kernel"] is kernel
    crank_idx = captured["inputs"].index(model.actuator_cranklength)
    assert captured["inputs"][crank_idx + 1] is model.actuator_gear
    assert captured["inputs"][crank_idx + 2] is model.body_isdofancestor


def test_optimized_fwd_velocity_overlaps_velocity_and_force_branches(monkeypatch):
    calls = []

    def fake_launch_in_stream(stream, kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append((stream, getattr(kernel, "__name__", str(kernel))))

    def fake_call_in_stream(stream, fn, *args):
        calls.append((stream, getattr(fn, "__name__", str(fn))))

    def fake_call_in_stream_after(stream, _wait_streams, fn, *args):
        calls.append((stream, getattr(fn, "__name__", str(fn))))

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(("main", name))

        _inner.__name__ = name
        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "_launch_in_stream", fake_launch_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream", fake_call_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream_after", fake_call_in_stream_after)
    monkeypatch.setattr(optimized, "_wait_for_streams", lambda _streams: None)
    monkeypatch.setattr(optimized.smooth, "com_vel", mark("com_vel"))
    monkeypatch.setattr(optimized.smooth, "rne", mark("rne"))
    monkeypatch.setattr(optimized.smooth, "tendon_bias", mark("tendon_bias"))

    model = SimpleNamespace(
        nactuator=28,
        nu=28,
        nv=28,
        ntendon=0,
        block_dim=SimpleNamespace(actuator_velocity=32),
        actuator_moment=object(),
        ten_J_rownnz=object(),
        ten_J_rowadr=object(),
        ten_J_colind=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qvel=object(),
        moment_rownnz=object(),
        moment_rowadr=object(),
        moment_colind=object(),
        actuator_moment=object(),
        actuator_velocity=object(),
        ten_J=object(),
        ten_velocity=object(),
        qfrc_bias=object(),
    )
    streams = tuple(f"stream-{i}" for i in range(12))

    fwd_velocity_optimized(model, data, streams)

    assert ("stream-9", "_actuator_velocity") in calls
    assert ("stream-10", "_tendon_velocity") in calls
    assert ("stream-11", "passive") in calls
    assert calls.index(("stream-11", "passive")) < calls.index(("main", "rne"))
    assert ("main", "tendon_bias") not in calls


def test_optimized_fwd_velocity_starts_passive_before_com_vel_when_safe(monkeypatch):
    calls = []

    def fake_launch_in_stream(stream, kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append((stream, getattr(kernel, "__name__", str(kernel))))

    def fake_call_in_stream_after(stream, wait_streams, fn, *args):
        calls.append(
            (
                stream,
                tuple(wait_streams),
                getattr(fn, "__name__", str(fn)),
            )
        )

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(("main", name))

        _inner.__name__ = name
        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "_launch_in_stream", fake_launch_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream_after", fake_call_in_stream_after)
    monkeypatch.setattr(optimized, "_wait_for_streams", lambda _streams: None)
    monkeypatch.setattr(optimized.smooth, "com_vel", mark("com_vel"))
    monkeypatch.setattr(optimized.smooth, "rne", mark("rne"))
    monkeypatch.setattr(optimized.smooth, "tendon_bias", mark("tendon_bias"))

    model = SimpleNamespace(
        nactuator=28,
        nu=28,
        nv=28,
        ntendon=0,
        nJten=0,
        has_fluid=False,
        callback=SimpleNamespace(passive=None),
        block_dim=SimpleNamespace(actuator_velocity=32),
        ten_J_rownnz=object(),
        ten_J_rowadr=object(),
        ten_J_colind=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qvel=object(),
        moment_rownnz=object(),
        moment_rowadr=object(),
        moment_colind=object(),
        actuator_moment=object(),
        actuator_velocity=object(),
        ten_J=object(),
        ten_velocity=object(),
        qfrc_bias=object(),
    )
    streams = tuple(f"stream-{i}" for i in range(13))

    fwd_velocity_optimized(model, data, streams)

    passive_call = ("stream-11", (), "passive")
    assert passive_call in calls
    assert calls.index(passive_call) < calls.index(("main", "com_vel"))


def test_optimized_fwd_velocity_keeps_fluid_passive_after_com_vel(monkeypatch):
    calls = []

    def fake_launch_in_stream(stream, kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append((stream, getattr(kernel, "__name__", str(kernel))))

    def fake_call_in_stream(stream, fn, *args):
        calls.append((stream, getattr(fn, "__name__", str(fn))))

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(("main", name))

        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "_launch_in_stream", fake_launch_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream", fake_call_in_stream)
    monkeypatch.setattr(optimized, "_wait_for_streams", lambda _streams: None)
    monkeypatch.setattr(optimized.smooth, "com_vel", mark("com_vel"))
    monkeypatch.setattr(optimized.smooth, "rne", mark("rne"))
    monkeypatch.setattr(optimized.smooth, "tendon_bias", mark("tendon_bias"))

    model = SimpleNamespace(
        nactuator=28,
        nu=28,
        nv=28,
        ntendon=0,
        nJten=0,
        has_fluid=True,
        callback=SimpleNamespace(passive=None),
        block_dim=SimpleNamespace(actuator_velocity=32),
        ten_J_rownnz=object(),
        ten_J_rowadr=object(),
        ten_J_colind=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qvel=object(),
        moment_rownnz=object(),
        moment_rowadr=object(),
        moment_colind=object(),
        actuator_moment=object(),
        actuator_velocity=object(),
        ten_J=object(),
        ten_velocity=object(),
        qfrc_bias=object(),
    )
    streams = tuple(f"stream-{i}" for i in range(13))

    fwd_velocity_optimized(model, data, streams)

    assert calls.index(("main", "com_vel")) < calls.index(("stream-11", "passive"))


def test_optimized_fwd_velocity_prepares_tendon_bias_parallel_to_rne(monkeypatch):
    calls = []

    class ScratchArray:
        def __init__(self, name):
            self.name = name

        def zero_(self):
            calls.append(("zero", self.name))

    def fake_launch_in_stream(stream, kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append((stream, getattr(kernel, "__name__", str(kernel))))

    def fake_call_in_stream_after(stream, wait_streams, fn, *args):
        calls.append(
            (
                stream,
                tuple(wait_streams),
                getattr(fn, "__name__", str(fn)),
            )
        )

    def fake_call_in_stream(stream, fn, *args):
        calls.append((stream, getattr(fn, "__name__", str(fn))))
        fn(*args)

    def fake_wait_stream(stream):
        calls.append(("wait", stream))

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(("main", name))

        _inner.__name__ = name
        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "_launch_in_stream", fake_launch_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream_after", fake_call_in_stream_after)
    monkeypatch.setattr(optimized, "_call_in_stream", fake_call_in_stream)
    monkeypatch.setattr(optimized, "_wait_for_streams", lambda _streams: None)
    monkeypatch.setattr(optimized.wp, "wait_stream", fake_wait_stream)
    monkeypatch.setattr(optimized.wp, "launch", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimized.smooth, "com_vel", mark("com_vel"))
    monkeypatch.setattr(optimized.smooth, "rne", mark("rne"))
    monkeypatch.setattr(optimized.smooth, "tendon_bias", mark("tendon_bias"))
    monkeypatch.setattr(
        optimized,
        "tendon_bias_prepare_with_scratch",
        mark("tendon_bias_prepare_with_scratch"),
        raising=False,
    )
    monkeypatch.setattr(
        optimized,
        "tendon_bias_apply_with_scratch",
        mark("tendon_bias_apply"),
        raising=False,
    )

    model = SimpleNamespace(
        nactuator=28,
        nu=28,
        ntendon=2,
        nJten=4,
        max_ten_J_rownnz=3,
        has_fluid=False,
        callback=SimpleNamespace(passive=None),
        block_dim=SimpleNamespace(actuator_velocity=32),
        ten_J_rownnz=object(),
        ten_J_rowadr=object(),
        ten_J_colind=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qvel=object(),
        moment_rownnz=object(),
        moment_rowadr=object(),
        moment_colind=object(),
        actuator_moment=object(),
        actuator_velocity=object(),
        ten_J=object(),
        ten_velocity=object(),
        qfrc_bias=object(),
    )
    scratch = OptimizedFwdPositionScratch(
        collision_context=object(),
        transmission_moment_nnz=object(),
        tendon_bias_Jdot=ScratchArray("ten_Jdot"),
        tendon_bias_coef=ScratchArray("ten_bias_coef"),
    )
    streams = tuple(f"stream-{i}" for i in range(13))

    fwd_velocity_optimized(model, data, streams, scratch=scratch)

    assert ("stream-12", "tendon_bias_prepare_with_scratch") in calls
    assert calls.index(("stream-12", "tendon_bias_prepare_with_scratch")) < calls.index(
        ("main", "rne")
    )
    assert calls.index(("main", "rne")) < calls.index(("wait", "stream-12"))
    assert calls.index(("wait", "stream-12")) < calls.index(
        ("main", "tendon_bias_apply")
    )
    assert ("main", "tendon_bias") not in calls


def test_optimized_fwd_velocity_can_start_actuation_on_actuator_stream(monkeypatch):
    calls = []

    def fake_launch_in_stream(stream, kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append((stream, getattr(kernel, "__name__", str(kernel))))

    def fake_call_in_stream_after(stream, wait_streams, fn, *args):
        calls.append((stream, tuple(wait_streams), getattr(fn, "__name__", str(fn))))

    def fake_start_actuation(stream, model, data, *, apply_limits=True):
        calls.append((stream, "start_actuation", apply_limits))

    def fake_wait_for_streams(streams):
        calls.append(("wait_for_streams", tuple(streams)))

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(("main", name))

        _inner.__name__ = name
        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "_launch_in_stream", fake_launch_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream_after", fake_call_in_stream_after)
    monkeypatch.setattr(optimized, "_wait_for_streams", fake_wait_for_streams)
    monkeypatch.setattr(
        optimized,
        "_start_fwd_actuation_no_tendon",
        fake_start_actuation,
    )
    monkeypatch.setattr(optimized.smooth, "com_vel", mark("com_vel"))
    monkeypatch.setattr(optimized.smooth, "rne", mark("rne"))
    monkeypatch.setattr(optimized.smooth, "tendon_bias", mark("tendon_bias"))

    model = SimpleNamespace(
        nactuator=28,
        nu=28,
        nv=28,
        ntendon=0,
        nJten=0,
        has_fluid=False,
        callback=SimpleNamespace(passive=None),
        block_dim=SimpleNamespace(actuator_velocity=32),
        ten_J_rownnz=object(),
        ten_J_rowadr=object(),
        ten_J_colind=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qvel=object(),
        moment_rownnz=object(),
        moment_rowadr=object(),
        moment_colind=object(),
        actuator_moment=object(),
        actuator_velocity=object(),
        ten_J=object(),
        ten_velocity=object(),
        qfrc_bias=object(),
    )
    streams = tuple(f"stream-{i}" for i in range(13))

    result = optimized.fwd_velocity_optimized(
        model,
        data,
        streams,
        start_actuation=True,
    )

    assert ("stream-9", "_actuator_velocity") in calls
    assert ("stream-9", "start_actuation", True) in calls
    assert calls.index(("stream-9", "start_actuation", True)) < calls.index(
        ("main", "com_vel")
    )
    wait_calls = [call for call in calls if call[0] == "wait_for_streams"]
    assert all("stream-9" not in call[1] for call in wait_calls)
    assert result.actuation_started is True
    assert result.actuation_stream == "stream-9"
    assert result.actuator_limits_pending is False


@pytest.mark.parametrize("callback_name", ["passive", "sensor"])
def test_early_actuation_waits_for_callbacks(callback_name):
    import instantmimic.utils.mjw_step_optimized as optimized

    callback = SimpleNamespace(
        control=None,
        act_dyn=None,
        act_gain=None,
        act_bias=None,
        passive=None,
        sensor=None,
    )
    setattr(callback, callback_name, object())
    model = SimpleNamespace(
        nactuator=1,
        nu=1,
        ntendon=0,
        nsensor=0,
        nhistory=0,
        opt=SimpleNamespace(disableflags=0),
        callback=callback,
    )

    assert not optimized._can_start_actuation_early(model)


def test_optimized_fwd_velocity_can_defer_early_actuator_limits(monkeypatch):
    calls = []

    def fake_launch_in_stream(stream, kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append((stream, getattr(kernel, "__name__", str(kernel))))

    def fake_call_in_stream_after(stream, wait_streams, fn, *args):
        calls.append((stream, tuple(wait_streams), getattr(fn, "__name__", str(fn))))

    def fake_start_actuation(stream, model, data, *, apply_limits=True):
        calls.append((stream, "start_actuation", apply_limits))

    def fake_wait_for_streams(streams):
        calls.append(("wait_for_streams", tuple(streams)))

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(("main", name))

        _inner.__name__ = name
        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "_launch_in_stream", fake_launch_in_stream)
    monkeypatch.setattr(optimized, "_call_in_stream_after", fake_call_in_stream_after)
    monkeypatch.setattr(optimized, "_wait_for_streams", fake_wait_for_streams)
    monkeypatch.setattr(
        optimized,
        "_start_fwd_actuation_no_tendon",
        fake_start_actuation,
    )
    monkeypatch.setattr(optimized.smooth, "com_vel", mark("com_vel"))
    monkeypatch.setattr(optimized.smooth, "rne", mark("rne"))
    monkeypatch.setattr(optimized.smooth, "tendon_bias", mark("tendon_bias"))

    model = SimpleNamespace(
        nactuator=28,
        nu=28,
        nv=28,
        ntendon=0,
        nJten=0,
        has_fluid=False,
        callback=SimpleNamespace(passive=None),
        block_dim=SimpleNamespace(actuator_velocity=32),
        ten_J_rownnz=object(),
        ten_J_rowadr=object(),
        ten_J_colind=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qvel=object(),
        moment_rownnz=object(),
        moment_rowadr=object(),
        moment_colind=object(),
        actuator_moment=object(),
        actuator_velocity=object(),
        ten_J=object(),
        ten_velocity=object(),
        qfrc_bias=object(),
    )
    streams = tuple(f"stream-{i}" for i in range(13))

    result = optimized.fwd_velocity_optimized(
        model,
        data,
        streams,
        start_actuation=True,
        defer_actuator_limits=True,
    )

    assert ("stream-9", "start_actuation", False) in calls
    assert result.actuator_limits_pending is True


def test_forward_after_position_waits_for_early_actuation_before_optimized_acceleration(
    monkeypatch,
):
    calls = []

    class Zeroable:
        def zero_(self):
            calls.append(("zero", "sensordata"))

    def fake_fwd_velocity(
        model,
        data,
        streams,
        *,
        scratch=None,
        start_actuation=False,
        defer_actuator_limits=False,
        **_kwargs,
    ):
        calls.append(("fwd_velocity", start_actuation, defer_actuator_limits))
        return SimpleNamespace(
            actuation_started=True,
            actuation_stream="act-stream",
            actuator_limits_pending=True,
        )

    def fail_stock_fwd_actuation(*_args):
        msg = "early actuation path should skip stock fwd_actuation"
        raise AssertionError(msg)

    def fake_wait_stream(stream):
        calls.append(("wait", stream))

    def fake_fwd_acceleration(*_args, **_kwargs):
        calls.append(("fwd_acceleration",))

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append((name,))

        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "fwd_velocity_optimized", fake_fwd_velocity)
    monkeypatch.setattr(
        optimized.mjw_forward, "fwd_actuation", fail_stock_fwd_actuation
    )
    monkeypatch.setattr(optimized, "fwd_acceleration_optimized", fake_fwd_acceleration)
    monkeypatch.setattr(optimized.wp, "wait_stream", fake_wait_stream)
    monkeypatch.setattr(optimized.sensor, "sensor_pos", mark("sensor_pos"))
    monkeypatch.setattr(optimized.sensor, "sensor_vel", mark("sensor_vel"))
    monkeypatch.setattr(optimized.sensor, "sensor_acc", mark("sensor_acc"))
    monkeypatch.setattr(optimized.solver, "solve", mark("solve"))

    model = SimpleNamespace(
        nv=1,
        opt=SimpleNamespace(enableflags=0, disableflags=0),
        sensor_e_potential=0,
        sensor_e_kinetic=0,
        callback=SimpleNamespace(control=None),
        nactuator=28,
        nu=28,
        ntendon=0,
        nsensor=0,
    )
    data = SimpleNamespace(njmax=1, sensordata=Zeroable(), energy=Zeroable())

    optimized._forward_after_position(
        model,
        data,
        streams=tuple(f"stream-{i}" for i in range(13)),
        scratch=object(),
        use_optimized_kernel=True,
    )

    assert ("fwd_velocity", True, True) in calls
    assert calls.index(("wait", "act-stream")) < calls.index(("fwd_acceleration",))


def test_moved_state_prefix_splits_velocity_from_action_dependent_dynamics(
    monkeypatch,
):
    calls = []

    class Zeroable:
        def zero_(self):
            calls.append(("zero",))

    def fake_fwd_velocity(
        model,
        data,
        streams,
        *,
        scratch=None,
        start_actuation=False,
        defer_actuator_limits=False,
        **_kwargs,
    ):
        calls.append(("velocity", start_actuation, defer_actuator_limits))
        return optimized.OptimizedFwdVelocityResult()

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append((name,))

        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "fwd_velocity_optimized", fake_fwd_velocity)
    monkeypatch.setattr(optimized.sensor, "sensor_pos", mark("sensor_pos"))
    monkeypatch.setattr(optimized.sensor, "sensor_vel", mark("sensor_vel"))
    monkeypatch.setattr(optimized.sensor, "sensor_acc", mark("sensor_acc"))
    monkeypatch.setattr(optimized.mjw_forward, "fwd_actuation", mark("actuation"))
    monkeypatch.setattr(optimized, "fwd_acceleration_optimized", mark("acceleration"))
    monkeypatch.setattr(optimized.solver, "solve", mark("solve"))

    model = SimpleNamespace(
        nv=1,
        opt=SimpleNamespace(enableflags=0, disableflags=0),
        sensor_e_potential=0,
        sensor_e_kinetic=0,
        callback=SimpleNamespace(control=None),
        nactuator=0,
        nu=0,
        ntendon=0,
        nsensor=0,
    )
    data = SimpleNamespace(njmax=1, sensordata=Zeroable(), energy=Zeroable())
    streams = tuple(f"stream-{i}" for i in range(13))

    optimized._forward_state_prefix_after_position(
        model,
        data,
        streams,
        scratch=object(),
        use_optimized_kernel=True,
        start_actuation=False,
    )

    assert ("velocity", False, False) in calls
    assert ("sensor_pos",) in calls
    assert ("sensor_vel",) in calls
    assert ("actuation",) not in calls
    assert ("acceleration",) not in calls
    assert ("solve",) not in calls

    calls.clear()
    optimized._forward_after_state_prefix(
        model,
        data,
        streams,
        scratch=object(),
        use_optimized_kernel=True,
    )

    assert ("velocity", False, False) not in calls
    assert ("sensor_pos",) not in calls
    assert ("sensor_vel",) not in calls
    assert ("actuation",) in calls
    assert ("acceleration",) in calls
    assert ("solve",) in calls
    assert ("sensor_acc",) in calls


def test_action_suffix_uses_preallocated_actuation_call(monkeypatch):
    calls = []

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append((name,))

        return _inner

    def fail_stock_actuation(*_args, **_kwargs):
        raise AssertionError("tendon actuation must use persistent scratch")

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.mjw_forward, "fwd_actuation", fail_stock_actuation)
    monkeypatch.setattr(optimized.mjw_forward, "fwd_acceleration", mark("acceleration"))
    monkeypatch.setattr(optimized.solver, "solve", mark("solve"))
    monkeypatch.setattr(optimized.sensor, "sensor_acc", mark("sensor_acc"))

    model = SimpleNamespace(
        nv=1,
        opt=SimpleNamespace(disableflags=0, integrator=optimized.IntegratorType.EULER),
        callback=SimpleNamespace(control=None),
    )
    data = SimpleNamespace(
        njmax=1,
    )
    scratch = SimpleNamespace(
        actuation_call=mark("preallocated_actuation"),
        solver_context=None,
        solver_nsolving=None,
        sensor_acc_call=None,
    )

    optimized._forward_after_state_prefix(
        model,
        data,
        streams=(),
        scratch=scratch,
        use_optimized_kernel=False,
    )

    assert ("preallocated_actuation",) in calls
    assert ("acceleration",) in calls
    assert ("solve",) in calls
    assert ("sensor_acc",) in calls


def test_forward_after_position_uses_early_actuation_with_stock_acceleration(
    monkeypatch,
):
    calls = []

    class Zeroable:
        def zero_(self):
            calls.append(("zero", "sensordata"))

    def fake_fwd_velocity(
        model,
        data,
        streams,
        *,
        scratch=None,
        start_actuation=False,
        defer_actuator_limits=False,
        **_kwargs,
    ):
        calls.append(("fwd_velocity", start_actuation, defer_actuator_limits))
        return SimpleNamespace(
            actuation_started=True,
            actuation_stream="act-stream",
            actuator_limits_pending=False,
        )

    def fail_stock_fwd_actuation(*_args):
        msg = "early actuation path should skip stock fwd_actuation"
        raise AssertionError(msg)

    def fake_wait_stream(stream):
        calls.append(("wait", stream))

    def fake_stock_fwd_acceleration(*_args, **kwargs):
        calls.append(("stock_fwd_acceleration", kwargs.get("factorize")))

    def fail_optimized_fwd_acceleration(*_args, **_kwargs):
        msg = "use_optimized_kernel=false should use stock fwd_acceleration"
        raise AssertionError(msg)

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append((name,))

        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "fwd_velocity_optimized", fake_fwd_velocity)
    monkeypatch.setattr(
        optimized.mjw_forward, "fwd_actuation", fail_stock_fwd_actuation
    )
    monkeypatch.setattr(
        optimized.mjw_forward,
        "fwd_acceleration",
        fake_stock_fwd_acceleration,
    )
    monkeypatch.setattr(
        optimized,
        "fwd_acceleration_optimized",
        fail_optimized_fwd_acceleration,
    )
    monkeypatch.setattr(optimized.wp, "wait_stream", fake_wait_stream)
    monkeypatch.setattr(optimized.sensor, "sensor_pos", mark("sensor_pos"))
    monkeypatch.setattr(optimized.sensor, "sensor_vel", mark("sensor_vel"))
    monkeypatch.setattr(optimized.sensor, "sensor_acc", mark("sensor_acc"))
    monkeypatch.setattr(optimized.solver, "solve", mark("solve"))

    model = SimpleNamespace(
        nv=1,
        opt=SimpleNamespace(enableflags=0, disableflags=0),
        sensor_e_potential=0,
        sensor_e_kinetic=0,
        callback=SimpleNamespace(control=None),
        nactuator=28,
        nu=28,
        ntendon=0,
        nsensor=0,
    )
    data = SimpleNamespace(njmax=1, sensordata=Zeroable(), energy=Zeroable())

    optimized._forward_after_position(
        model,
        data,
        streams=tuple(f"stream-{i}" for i in range(13)),
        scratch=object(),
        use_optimized_kernel=False,
    )

    assert ("fwd_velocity", True, False) in calls
    assert calls.index(("wait", "act-stream")) < calls.index(
        ("stock_fwd_acceleration", True)
    )


def test_optimized_fwd_acceleration_dense_factorize_uses_no_store_solve(
    monkeypatch,
):
    calls = []

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append(("launch", getattr(kernel, "__name__", str(kernel))))

    def fake_xfrc_accumulate(model, data, qfrc):
        calls.append(("xfrc_accumulate", model, data, qfrc))

    def fake_no_store(model, data, matrix, x, y):
        calls.append(("no_store_factor_solve", model, data, matrix, x, y))

    def fail_stock_factor_solve(*_args, **_kwargs):
        msg = "optimized fwd_acceleration should not store qLD in dense factorize path"
        raise AssertionError(msg)

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)
    monkeypatch.setattr(optimized.mjw_forward, "xfrc_accumulate", fake_xfrc_accumulate)
    monkeypatch.setattr(
        optimized,
        "_factor_solve_i_dense_no_l_store",
        fake_no_store,
    )
    monkeypatch.setattr(optimized.smooth, "factor_solve_i", fail_stock_factor_solve)

    model = SimpleNamespace(
        is_sparse=False,
        nv=2,
        body_treeid=object(),
        dof_bodyid=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qfrc_applied=object(),
        tree_awake=object(),
        qfrc_bias=object(),
        qfrc_passive=object(),
        qfrc_actuator=object(),
        qfrc_smooth=object(),
        qM=object(),
        qacc_smooth=object(),
    )

    optimized.fwd_acceleration_optimized(model, data, factorize=True)

    assert any(call[0] == "launch" for call in calls)
    assert any(call[0] == "xfrc_accumulate" for call in calls)
    assert any(call[0] == "no_store_factor_solve" for call in calls)


def test_optimized_fwd_acceleration_accepts_m_structure_storage(monkeypatch):
    calls = []

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append(("launch", getattr(kernel, "__name__", str(kernel)), inputs))

    def fake_xfrc_accumulate(model, data, qfrc):
        calls.append(("xfrc_accumulate", model, data, qfrc))

    def fake_no_store(model, data, matrix, x, y):
        calls.append(("no_store_factor_solve", matrix, x, y))

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)
    monkeypatch.setattr(optimized.mjw_forward, "xfrc_accumulate", fake_xfrc_accumulate)
    monkeypatch.setattr(
        optimized,
        "_factor_solve_i_dense_no_l_store",
        fake_no_store,
    )

    model = SimpleNamespace(
        is_sparse=True,
        nv=2,
        body_treeid=object(),
        dof_bodyid=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qfrc_applied=object(),
        tree_awake=object(),
        qfrc_bias=object(),
        qfrc_passive=object(),
        qfrc_actuator=object(),
        qfrc_smooth=object(),
        M="M-structure",
        qacc_smooth=object(),
    )

    optimized.fwd_acceleration_optimized(model, data, factorize=True)

    assert (
        "no_store_factor_solve",
        "M-structure",
        data.qacc_smooth,
        data.qfrc_smooth,
    ) in calls


def test_dense_no_store_factor_solve_accepts_m_tiles(monkeypatch):
    calls = []

    block_dofs = SimpleNamespace(size=5)
    block_elems = SimpleNamespace(size=4)

    class Tile:
        size = 2
        adr = block_dofs
        elemid = block_elems

    def fake_launch_tiled(kernel, *, dim, inputs, outputs, block_dim):
        calls.append((kernel, dim, inputs, outputs, block_dim))

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch_tiled", fake_launch_tiled)

    model = SimpleNamespace(
        M_tiles=(Tile(),),
        qLD_block_total=4,
        block_dim=SimpleNamespace(
            cholesky_factorize_solve=64,
            small_cholesky=32,
        ),
    )
    data = SimpleNamespace(
        nworld=3,
        qLD=SimpleNamespace(shape=(3, 4)),
        qLDiagInv="diag",
    )

    optimized._factor_solve_i_dense_no_l_store(
        model,
        data,
        "M-structure",
        "qacc",
        "rhs",
    )

    assert calls
    _, dim, inputs, outputs, block_dim = calls[0]
    assert dim == (3, block_dofs.size)
    assert inputs == ["M-structure", block_elems, block_dofs, "rhs"]
    assert outputs == ["qacc"]
    assert block_dim == 32


def test_factor_solve_no_store_preserves_compact_diag_output(monkeypatch):
    calls = []

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        calls.append((kernel, dim, inputs, outputs))

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)

    compact_kernel = object()
    monkeypatch.setattr(
        optimized.smooth,
        "_small_cholesky_factorize_solve_block",
        lambda size: compact_kernel if size == 2 else None,
    )

    tile = SimpleNamespace(
        size=2,
        adr=SimpleNamespace(size=2),
        elemid=SimpleNamespace(size=0),
    )

    model = SimpleNamespace(
        M_tiles=(tile,),
        qLD_block_total=0,
        M_rowadr="row-adr",
        qLD_block_adr="block-adr",
        block_dim=SimpleNamespace(small_cholesky=16),
    )
    data = SimpleNamespace(
        nworld=3,
        qLD=SimpleNamespace(shape=(3, 0)),
        qLDiagInv="internal-diag",
    )

    optimized.factor_solve_i_no_l_store(
        model,
        data,
        "M-structure",
        "unused-L",
        "caller-diag",
        "solution",
        "rhs",
    )

    assert calls == [
        (
            compact_kernel,
            (3, 2),
            ["row-adr", "block-adr", "M-structure", tile.adr, "rhs"],
            ["caller-diag", "solution", data.qLD],
        )
    ]


def test_make_optimized_scratch_accepts_m_fullm_indices(monkeypatch):
    launched = []

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        launched.append((dim, inputs, outputs))

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)
    monkeypatch.setattr(optimized.wp, "empty", lambda *args, **_kwargs: object())
    monkeypatch.setattr(optimized.wp, "zeros", lambda *args, **_kwargs: object())
    monkeypatch.setattr(optimized.wp, "ones", lambda *args, **_kwargs: object())

    fullm_i = SimpleNamespace(shape=(7,))
    fullm_j = object()
    model = SimpleNamespace(
        opt=SimpleNamespace(
            disableflags=0, enableflags=0, run_rne_postconstraint=False
        ),
        sensor_contact_pos_adr=SimpleNamespace(size=0),
        sensor_contact_acc_adr=SimpleNamespace(size=0),
        sensor_rne_postconstraint=False,
        nsensortaxel=0,
        nacttrnbody=0,
        ntendon=0,
        is_sparse=True,
        nactuator=1,
        nu=1,
        nv=2,
        M_fullm_i=fullm_i,
        M_fullm_j=fullm_j,
        M_elemid=object(),
        actuator_trntype=(),
    )
    data = SimpleNamespace(
        nworld=1,
        naconmax=4,
        qpos=SimpleNamespace(device=None),
        M=SimpleNamespace(shape=(1, 3)),
    )

    monkeypatch.setattr(
        optimized.tree_fusion, "create_tree_reduction_scratch", lambda m: None
    )
    scratch = optimized.make_optimized_fwd_position_scratch(
        model,
        data,
        include_optimized_kernel_scratch=True,
    )

    assert scratch.implicit_qderiv is not None
    assert launched[0][0] == 7
    assert launched[0][1] == [fullm_i, fullm_j]


def test_allocation_free_scratch_keeps_runtime_buffers_without_optimized_kernels(
    monkeypatch,
):
    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "empty", lambda *args, **_kwargs: object())
    monkeypatch.setattr(optimized.wp, "zeros", lambda *args, **_kwargs: object())
    monkeypatch.setattr(optimized.wp, "ones", lambda *args, **_kwargs: object())

    model = SimpleNamespace(
        opt=SimpleNamespace(
            disableflags=0, enableflags=0, run_rne_postconstraint=False
        ),
        sensor_contact_pos_adr=SimpleNamespace(size=0),
        sensor_contact_acc_adr=SimpleNamespace(size=0),
        sensor_rne_postconstraint=False,
        nsensortaxel=0,
        nacttrnbody=0,
        ntendon=2,
        nJten=3,
        nbody=5,
        sensor_subtree_vel=True,
        is_sparse=True,
        nactuator=0,
        nu=0,
        nv=0,
        actuator_trntype=(),
    )
    data = SimpleNamespace(
        nworld=1,
        naconmax=4,
        njmax=0,
        qpos=SimpleNamespace(device=None),
        M=SimpleNamespace(shape=(1, 0)),
    )

    monkeypatch.setattr(
        optimized.tree_fusion, "create_tree_reduction_scratch", lambda m: None
    )
    scratch = optimized.make_optimized_fwd_position_scratch(
        model,
        data,
        include_optimized_kernel_scratch=False,
    )

    assert scratch.tendon_bias_Jdot is not None
    assert scratch.tendon_bias_coef is not None
    assert scratch.sensor_subtree_bodyvel is not None
    assert scratch.sensor_vel_call is not None


@pytest.mark.parametrize(
    ("awake_prev", "expect_nacon_reset"),
    [(None, True), ("snapshot", False)],
)
def test_collision_with_context_resets_only_full_pass_nacon(
    monkeypatch, awake_prev, expect_nacon_reset
):
    calls = []

    class Counter:
        def __init__(self, name):
            self.name = name

        def zero_(self):
            calls.append(("zero", self.name))

    def fake_broadphase(model, data, ctx, awake_prev=None):
        calls.append(("broadphase", awake_prev))

    def fake_narrowphase(model, data, ctx):
        calls.append(("narrowphase", ctx))

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.collision_driver, "sap_broadphase", fake_broadphase)
    monkeypatch.setattr(optimized.collision_driver, "_narrowphase", fake_narrowphase)

    model = SimpleNamespace(
        opt=SimpleNamespace(
            disableflags=0,
            enableflags=0,
            broadphase=optimized.BroadphaseType.SAP_TILE,
        ),
        callback=SimpleNamespace(contactfilter=None),
        nflex=0,
    )
    data = SimpleNamespace(
        naconmax=4,
        nacon=Counter("nacon"),
        ncollision=Counter("ncollision"),
    )

    optimized.collision_with_context(model, data, "ctx", awake_prev=awake_prev)

    assert ("zero", "ncollision") in calls
    assert (("zero", "nacon") in calls) is expect_nacon_reset
    assert ("broadphase", awake_prev) in calls


def test_collision_with_context_uses_preallocated_narrowphase_call(monkeypatch):
    calls = []

    class Counter:
        def zero_(self):
            return None

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(
        optimized.collision_driver,
        "nxn_broadphase",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        optimized.collision_driver,
        "_narrowphase",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("stock narrowphase must not run")
        ),
    )
    model = SimpleNamespace(
        opt=SimpleNamespace(
            disableflags=0,
            enableflags=0,
            broadphase=optimized.BroadphaseType.NXN,
        ),
        callback=SimpleNamespace(contactfilter=None),
        nflex=0,
    )
    data = SimpleNamespace(naconmax=4, nacon=Counter(), ncollision=Counter())

    optimized.collision_with_context(
        model,
        data,
        "ctx",
        narrowphase_call=lambda *_args: calls.append("preallocated"),
    )

    assert calls == ["preallocated"]


def test_sleep_collision_reuses_persistent_awake_snapshot(monkeypatch):
    calls = []
    awake_snapshot = object()
    scratch = SimpleNamespace(collision_awake_prev=awake_snapshot)
    data = SimpleNamespace(body_awake=object())

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(
        optimized,
        "_run_collision_once",
        lambda m, d, scratch, awake_prev=None: calls.append(("collision", awake_prev)),
    )
    monkeypatch.setattr(
        optimized,
        "sleep",
        SimpleNamespace(
            wake_collision=lambda m, d: calls.append(("wake", None)),
            update_sleep=lambda m, d: calls.append(("sleep", None)),
        ),
    )
    monkeypatch.setattr(
        optimized.wp,
        "copy",
        lambda dst, src: calls.append(("copy", dst, src)),
    )

    optimized._run_collision_sequence(object(), data, scratch, sleep_enabled=True)

    assert calls == [
        ("collision", None),
        ("wake", None),
        ("copy", awake_snapshot, data.body_awake),
        ("sleep", None),
        ("collision", awake_snapshot),
    ]


@pytest.mark.parametrize(
    "broadphase",
    [BroadphaseType.SAP_TILE, BroadphaseType.SAP_SEGMENTED],
)
def test_sap_broadphase_reuses_preallocated_arrays(monkeypatch, tmp_path, broadphase):
    import instantmimic.utils.mjw_step_optimized as optimized

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    mj_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom type="plane" size="1 1 0.1"/>
            <body pos="0 0 0.2">
              <joint type="free"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    host_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, host_data)
    model = mjw.put_model(mj_model)
    model.opt.broadphase = broadphase
    data = mjw.put_data(mj_model, host_data, nworld=1, nconmax=4, njmax=16)
    scratch = optimized.make_optimized_fwd_position_scratch(model, data)
    assert scratch.collision_broadphase_call is not None

    monkeypatch.setattr(
        optimized.wp,
        "empty",
        lambda *_args, **_kwargs: pytest.fail(
            "SAP broadphase allocated after scratch construction"
        ),
    )

    optimized._run_collision_once(model, data, scratch)
    wp.synchronize()


@pytest.mark.parametrize(
    "variant", ["plain", "self", "self-sap", "two-flex", "two-flex-apart"]
)
def test_flex_collision_reuses_preallocated_arrays(monkeypatch, tmp_path, variant):
    import instantmimic.utils.mjw_step_optimized as optimized

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    selfcollide = (
        "sap" if variant == "self-sap" else "auto" if variant == "self" else "none"
    )
    filt = ' contype="1" conaffinity="2"' if variant == "two-flex-apart" else ""
    self_contact = f'<contact selfcollide="{selfcollide}"{filt}/>'
    count = "8 8 1" if variant == "self-sap" else "3 3 1"
    second_flex = (
        f"""
        <flexcomp name="cloth2" type="grid" count="3 3 1"
                  spacing="0.1 0.1 0.1" dim="2" mass="0.1"
                  radius="0.01" pos="0.5 0 0.02">
          <contact selfcollide="none"{filt}/>
        </flexcomp>
        """
        if variant.startswith("two-flex")
        else ""
    )
    mj_model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <geom type="plane" size="1 1 0.1"/>
            <geom type="sphere" size="0.04" pos="0.05 0.05 0.02"/>
            <flexcomp name="cloth" type="grid" count="{count}"
                      spacing="0.1 0.1 0.1" dim="2" mass="0.1"
                      radius="0.01" pos="0 0 0.02">
              {self_contact}
            </flexcomp>
            {second_flex}
          </worldbody>
        </mujoco>
        """
    )
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(
            mj_model,
            mujoco.MjData(mj_model),
            nworld=1,
            nconmax=512,
            njmax=1024,
        )
        mjw.forward(model, data)
        scratch = make_optimized_fwd_position_scratch(model, data, "cpu")
        assert model.nflex == (2 if variant.startswith("two-flex") else 1)
        assert scratch.flex_collision_call is not None
        assert model.has_flex_selfcollide is (variant in ("self", "self-sap"))
        for allocation_name in ("empty", "zeros", "full"):
            monkeypatch.setattr(
                optimized.wp,
                allocation_name,
                lambda *_args, **_kwargs: pytest.fail(
                    "flex collision allocated after scratch construction"
                ),
            )
        optimized._run_collision_once(model, data, scratch)
        wp.synchronize()


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["self-sap", "two-flex"])
def test_flex_collision_scratch_can_be_captured_and_replayed(variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    import instantmimic.utils.mjw_step_optimized as optimized

    selfcollide = "sap" if variant == "self-sap" else "none"
    count = "8 8 1" if variant == "self-sap" else "3 3 1"
    second_flex = (
        """
        <flexcomp name="cloth2" type="grid" count="3 3 1"
                  spacing="0.1 0.1 0.1" dim="2" mass="0.1"
                  radius="0.01" pos="0.5 0 0.02">
          <contact selfcollide="none"/>
        </flexcomp>
        """
        if variant == "two-flex"
        else ""
    )
    mj_model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <geom type="plane" size="1 1 0.1"/>
            <geom type="sphere" size="0.04" pos="0.05 0.05 0.02"/>
            <flexcomp name="cloth" type="grid" count="{count}"
                      spacing="0.1 0.1 0.1" dim="2" mass="0.1"
                      radius="0.01" pos="0 0 0.02">
              <contact selfcollide="{selfcollide}"/>
            </flexcomp>
            {second_flex}
          </worldbody>
        </mujoco>
        """
    )
    model = mjw.put_model(mj_model)
    data = mjw.put_data(
        mj_model,
        mujoco.MjData(mj_model),
        nworld=1,
        nconmax=512,
        njmax=1024,
    )
    scratch = make_optimized_fwd_position_scratch(model, data, "cuda:0")

    optimized._run_collision_once(model, data, scratch)
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        optimized._run_collision_once(model, data, scratch)
    wp.capture_launch(capture.graph)
    wp.synchronize()

    assert int(wp.to_torch(data.nacon).cpu().item()) > 0


def _flex_model_xml(variant: str) -> str:
    """Flex scenes for the collision tests: ``self-sap`` (self-collision with the
    SAP pass and its FPS filter), ``two-flex`` (two flexes that can collide) and
    ``two-flex-apart`` (two flexes whose contype/conaffinity rule out a pair, so
    the flex-flex pass is skipped)."""
    selfcollide = "sap" if variant == "self-sap" else "none"
    count = "8 8 1" if variant == "self-sap" else "3 3 1"
    # contype 1 / conaffinity 2 on both: neither flex can touch the other, while
    # both still touch the plane and the sphere (contype 1 / conaffinity 1).
    filt = ' contype="1" conaffinity="2"' if variant == "two-flex-apart" else ""
    second = (
        f"""
        <flexcomp name="cloth2" type="grid" count="3 3 1"
                  spacing="0.1 0.1 0.1" dim="2" mass="0.1"
                  radius="0.01" pos="0.05 0 0.02">
          <contact selfcollide="none"{filt}/>
        </flexcomp>
        """
        if variant.startswith("two-flex")
        else ""
    )
    return f"""
        <mujoco>
          <worldbody>
            <geom type="plane" size="1 1 0.1"/>
            <geom type="sphere" size="0.04" pos="0.05 0.05 0.02"/>
            <flexcomp name="cloth" type="grid" count="{count}"
                      spacing="0.1 0.1 0.1" dim="2" mass="0.1"
                      radius="0.01" pos="0 0 0.02">
              <contact selfcollide="{selfcollide}"{filt}/>
            </flexcomp>
            {second}
          </worldbody>
        </mujoco>
        """


def _contact_set(data) -> np.ndarray:
    n = int(data.nacon.numpy()[0])
    rows = np.concatenate(
        [
            data.contact.dist.numpy()[:n, None],
            data.contact.pos.numpy()[:n],
            data.contact.frame.numpy()[:n, 0],
        ],
        axis=1,
    )
    return rows[np.lexsort(rows.T[::-1])]


@pytest.mark.parametrize("variant", ["two-flex", "two-flex-apart"])
def test_flex_collision_matches_upstream_collision(monkeypatch, tmp_path, variant):
    """The scratch flex call (preallocated workspaces, the flex-flex pass skipped
    when no pair can collide) finds the contacts upstream collision finds.

    Upstream keeps the lower candidate slot of two equally deep near duplicates,
    and the slots come from an atomic counter; the scratch call breaks that tie by
    candidate identity instead. ``two-flex`` has such ties, so the reference is
    upstream collision with only that filter swapped in."""
    from mujoco_warp._src import collision_driver, collision_flex

    import instantmimic.utils.mjw_step_optimized as optimized

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    mj_model = mujoco.MjModel.from_xml_string(_flex_model_xml(variant))
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
        out = []
        for use_scratch in (False, True):
            data = mjw.put_data(
                mj_model, mujoco.MjData(mj_model), nworld=1, nconmax=512, njmax=1024
            )
            mjw.forward(model, data)
            if use_scratch:
                scratch = make_optimized_fwd_position_scratch(model, data, "cpu")
                optimized._run_collision_once(model, data, scratch)
            else:
                with monkeypatch.context() as patch:
                    if variant == "two-flex":
                        patch.setattr(
                            collision_flex,
                            "_filter_and_write_contacts",
                            optimized._FlexContactFilterCall(),
                        )
                    collision_driver.collision(model, data)
            wp.synchronize()
            out.append(_contact_set(data))
    assert len(out[0]) > 0
    np.testing.assert_allclose(out[1], out[0], rtol=0, atol=1e-6)


def test_flex_contact_filter_does_not_depend_on_candidate_slot_order(tmp_path):
    """Narrowphase threads take candidate slots from an atomic counter, so the
    slot order changes between runs (and between eager launches and a graph
    replay). The filter keeps the same contacts in every order, including when
    two near duplicates are equally deep (the candidates from two flex-flex
    element pairs on either side of a shared vertex)."""
    import itertools

    from mujoco_warp._src import collision_flex

    import instantmimic.utils.mjw_step_optimized as optimized

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    mj_model = mujoco.MjModel.from_xml_string(_flex_model_xml("two-flex"))
    tilted = np.array([0.00157, 0.15307, 0.98821])
    # dist, pos, normal, element pair of a flex-flex candidate (flexes 0 and 1)
    candidates = [
        (-0.0199, (-1e-5, 0.0, 0.02), tilted, (0, 2)),
        (-0.0199, (1e-5, 0.0, 0.02), tilted * [-1, -1, 1], (7, 1)),
        (-0.0100, (0.1, 0.0, 0.02), (0.0, 0.0, 1.0), (1, 1)),
        (-0.0150, (0.1002, 0.0, 0.02), (0.0, 0.0, -1.0), (9, 9)),
        (-0.0200, (0.05, 0.05, 0.02), (0.0, 0.0, 1.0), (3, 3)),
    ]
    filter_call = optimized._FlexContactFilterCall()
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(
            mj_model, mujoco.MjData(mj_model), nworld=1, nconmax=512, njmax=1024
        )
        ws = collision_flex._allocate_flex_workspace(model, data)
        out = []
        for order in itertools.permutations(range(len(candidates))):
            rows = [candidates[k] for k in order]
            n = len(rows)
            for name, values in (
                ("dist", [r[0] for r in rows]),
                ("pos", [r[1] for r in rows]),
                ("nrm", [r[2] for r in rows]),
                ("elem", [r[3] for r in rows]),
                ("geom", [(-1, -1)] * n),
                ("flex", [(0, 1)] * n),
                ("vert", [(-1, -1)] * n),
                ("worldid", [0] * n),
            ):
                array = getattr(ws, name)
                host = array.numpy()
                host[:n] = values
                array.assign(host)
            ws.ncand.fill_(n)
            data.nacon.zero_()
            filter_call(model, data, ws, enable_fps=True)
            out.append(_contact_set(data))
    # The deeper of two near duplicates survives; of the equally deep pair, the
    # candidate with the lower element ids.
    np.testing.assert_allclose(
        out[0][:, :2], [[-0.0200, 0.05], [-0.0199, -1e-5], [-0.0150, 0.1002]], atol=1e-7
    )
    np.testing.assert_allclose(out[0][1, 4:], tilted, atol=1e-4)
    for contacts in out[1:]:
        np.testing.assert_array_equal(contacts, out[0])


@pytest.mark.parametrize("unroll", [False, True])
def test_flex_collision_turns_the_conditional_fps_loop_off_only_when_unrolled(unroll):
    """In the nested benchmark capture (unroll_solver_iterations) the flex call
    runs with graph_conditional off, so upstream takes its fixed-count FPS loop;
    otherwise it keeps the conditional loop. The option is restored after."""
    import instantmimic.utils.mjw_step_optimized as optimized

    seen = []
    model = SimpleNamespace(opt=SimpleNamespace(graph_conditional=True))
    scratch = SimpleNamespace(
        flex_collision_call=lambda m, d, ctx: seen.append(m.opt.graph_conditional),
        unroll_solver_iterations=unroll,
    )
    with wp.ScopedDevice("cpu"):
        optimized._flex_collision_for_capture(scratch)(model, None, None)
    assert seen == [not unroll]
    assert model.opt.graph_conditional is True


@pytest.mark.cuda
def test_flex_fps_runs_inside_a_side_stream_forked_into_a_capture():
    """MuJoCo Warp's flex FPS filter loops with wp.capture_while, which Warp can
    build only on the registered capture stream. On a side stream forked into the
    capture (the parallel step's collision stream) the call takes upstream's
    fixed-count loop; the replayed graph finds the eager contacts (a scene with
    flex-flex contacts, so the FPS filter has candidates)."""
    import instantmimic.utils.mjw_step_optimized as optimized

    mj_model = mujoco.MjModel.from_xml_string(_flex_model_xml("two-flex"))
    model = mjw.put_model(mj_model)

    def fresh():
        data = mjw.put_data(
            mj_model, mujoco.MjData(mj_model), nworld=1, nconmax=512, njmax=1024
        )
        mjw.forward(model, data)
        return data

    eager = fresh()
    scratch = make_optimized_fwd_position_scratch(model, eager, "cuda:0")
    optimized._run_collision_once(model, eager, scratch)
    wp.synchronize()

    data = fresh()
    scratch = make_optimized_fwd_position_scratch(model, data, "cuda:0")
    optimized._run_collision_once(model, data, scratch)  # compile outside capture
    wp.synchronize()
    side = wp.Stream("cuda:0")
    with wp.ScopedCapture() as capture:
        main = wp.get_stream()
        side.wait_stream(main)
        with wp.ScopedStream(side):
            optimized._run_collision_once(model, data, scratch)
        main.wait_stream(side)
    assert model.opt.graph_conditional  # restored after the unrolled call
    for array in (data.nacon, data.contact.dist, data.contact.pos, data.contact.frame):
        array.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize()

    assert len(_contact_set(eager)) > 4
    np.testing.assert_allclose(
        _contact_set(data), _contact_set(eager), rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("passive_disabled", [False, True])
def test_fluid_passive_reuses_preallocated_array(
    monkeypatch,
    tmp_path,
    passive_disabled,
):
    import instantmimic.utils.mjw_step_optimized as optimized

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    mj_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option density="1" viscosity="0.1"/>
          <worldbody>
            <body pos="0 0 1">
              <joint type="free"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    if passive_disabled:
        mj_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SPRING)
        mj_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_DAMPER)
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=1)
        mjw.forward(model, data)
        scratch = make_optimized_fwd_position_scratch(model, data, "cpu")
        assert model.has_fluid and scratch.passive_call is not None
        monkeypatch.setattr(
            optimized.wp,
            "empty",
            lambda *_args, **_kwargs: pytest.fail(
                "fluid passive allocated after scratch construction"
            ),
        )
        scratch.passive_call(model, data)
        wp.synchronize()


@pytest.mark.parametrize("gravcomp", [0.0, 0.5])
def test_passive_skips_gravity_compensation_only_without_gravcomp_bodies(
    tmp_path, gravcomp
):
    from mujoco_warp._src import passive

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    mj_model = mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <body pos="0 0 1" gravcomp="{gravcomp}">
              <joint type="hinge" axis="0 1 0" damping="0.1"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0.3 0 0"/>
              <body pos="0.3 0 0">
                <joint type="hinge" axis="0 1 0" stiffness="2"/>
                <geom type="sphere" size="0.05"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = [0.3, -0.2]
    mj_data.qvel[:] = [0.5, 1.0]
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(mj_model, mj_data, nworld=2)
        reference = mjw.put_data(mj_model, mj_data, nworld=2)
        mjw.forward(model, data)
        mjw.forward(model, reference)
        scratch = make_optimized_fwd_position_scratch(model, data, "cpu")
        skipped = passive._gravity_force in scratch.passive_call._skipped_launches
        assert skipped == (gravcomp == 0.0)
        scratch.passive_call(model, data)
        passive.passive(model, reference)
        wp.synchronize()
        for name in ("qfrc_passive", "qfrc_gravcomp"):
            np.testing.assert_array_equal(
                getattr(data, name).numpy(), getattr(reference, name).numpy()
            )
        assert np.any(reference.qfrc_gravcomp.numpy()) == (gravcomp != 0.0)


def _constraint_free_mjcf(variant: str = "") -> str:
    """Two damped hinges over a floor; only the floor collides by default."""
    limit = 'limited="true" range="-1 1"' if variant == "limit" else ""
    friction = 'frictionloss="0.1"' if variant == "frictionloss" else ""
    collide = "" if variant == "contact" else 'contype="0" conaffinity="0"'
    equality = (
        '<equality><joint joint1="j0" joint2="j1"/></equality>'
        if variant == "equality"
        else ""
    )
    return f"""
    <mujoco>
      <option integrator="implicitfast"/>
      <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body pos="0 0 1">
          <joint name="j0" type="hinge" axis="0 1 0" damping="0.5" {limit} {friction}/>
          <geom type="capsule" size="0.05" fromto="0 0 0 0.4 0 0" {collide}/>
          <body pos="0.4 0 0">
            <joint name="j1" type="hinge" axis="0 1 0" damping="0.5"/>
            <geom type="sphere" size="0.05" contype="0" conaffinity="0"/>
          </body>
        </body>
      </worldbody>
      {equality}
    </mujoco>
    """


@pytest.mark.parametrize(
    ("variant", "ball_limits", "expected"),
    [
        ("", (), True),
        ("limit", (), False),
        ("frictionloss", (), False),
        ("contact", (), False),
        ("equality", (), False),
        ("", ("ball",), False),
    ],
)
def test_constraint_rows_are_impossible_only_without_any_row_source(
    variant, ball_limits, expected
):
    import instantmimic.utils.mjw_step_optimized as optimized

    mj_model = mujoco.MjModel.from_xml_string(_constraint_free_mjcf(variant))
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
    assert optimized._constraint_rows_impossible(model, ball_limits) is expected


def test_constraint_free_step_matches_upstream(tmp_path):
    from instantmimic.utils.mjw_step_optimized import step_optimized_fwd_position

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    mj_model = mujoco.MjModel.from_xml_string(_constraint_free_mjcf())
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = [0.4, -0.3]
    mj_data.qvel[:] = [1.0, -2.0]
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(mj_model, mj_data, nworld=2)
        reference = mjw.put_data(mj_model, mj_data, nworld=2)
        scratch = make_optimized_fwd_position_scratch(model, data, "cpu")
        assert scratch.constraint_free
        for _ in range(5):
            step_optimized_fwd_position(model, data, (), scratch)
            mjw.step(model, reference)
        wp.synchronize()
        for name in ("qpos", "qvel", "qacc"):
            np.testing.assert_allclose(
                getattr(data, name).numpy(),
                getattr(reference, name).numpy(),
                rtol=1e-5,
                atol=1e-6,
            )
        assert not np.any(data.qfrc_constraint.numpy())
        assert not np.any(data.nefc.numpy())


@pytest.mark.cuda
def test_constraint_free_step_captures_with_side_streams():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    mj_model = mujoco.MjModel.from_xml_string(_constraint_free_mjcf())
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = [0.4, -0.3]
    mj_data.qvel[:] = [1.0, -2.0]
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(mj_model, mj_data, nworld=4)
        reference = mjw.put_data(mj_model, mj_data, nworld=4)
        scratch = make_optimized_fwd_position_scratch(model, data, "cuda:0")
        assert scratch.constraint_free
        # The skipped collision/constraint stream must not join the capture.
        streams = make_optimized_fwd_position_streams("cuda:0")
        step_optimized_fwd_position(model, data, streams, scratch)
        mjw.step(model, reference)
        with wp.ScopedCapture() as capture:
            step_optimized_fwd_position(model, data, streams, scratch)
        for _ in range(3):
            wp.capture_launch(capture.graph)
            mjw.step(model, reference)
        wp.synchronize()
        for name in ("qpos", "qvel"):
            np.testing.assert_allclose(
                getattr(data, name).numpy(),
                getattr(reference, name).numpy(),
                rtol=1e-5,
                atol=1e-6,
            )


def test_sensor_acc_equality_reuses_preallocated_counters(monkeypatch, tmp_path):
    import instantmimic.utils.mjw_step_optimized as optimized

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    mj_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="a" pos="0 0 1">
              <joint type="free"/>
              <geom type="sphere" size="0.1"/>
              <site name="force_site"/>
            </body>
            <body name="b" pos="0 0 2">
              <joint type="free"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <equality>
            <connect body1="a" body2="b" anchor="0 0 1.5"/>
          </equality>
          <sensor>
            <force site="force_site"/>
          </sensor>
        </mujoco>
        """
    )
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(
            mj_model,
            mujoco.MjData(mj_model),
            nworld=1,
            nconmax=4,
            njmax=16,
        )
        mjw.forward(model, data)
        scratch = make_optimized_fwd_position_scratch(model, data, "cpu")
        assert model.sensor_rne_postconstraint and scratch.sensor_acc_call is not None
        monkeypatch.setattr(
            optimized.wp,
            "zeros",
            lambda *_args, **_kwargs: pytest.fail(
                "sensor RNE allocated after scratch construction"
            ),
        )
        scratch.sensor_acc_call(model, data)
        wp.synchronize()


def test_sensor_disable_preallocated_calls_follow_early_return():
    mj_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body><joint name="j"/><geom size="0.1"/></body>
          </worldbody>
          <sensor><jointpos joint="j"/></sensor>
        </mujoco>
        """
    )
    mj_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=1)
        scratch = make_optimized_fwd_position_scratch(model, data, "cpu")
        scratch.sensor_pos_call(model, data)
        scratch.sensor_vel_call(model, data)
        scratch.sensor_acc_call(model, data)


def test_sensor_history_reuses_preallocated_staging(monkeypatch, tmp_path):
    import instantmimic.utils.mjw_step_optimized as optimized

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    mj_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.01"/>
          <worldbody>
            <body><joint name="j"/><geom size="0.1"/></body>
          </worldbody>
          <sensor>
            <jointpos joint="j" nsample="4" delay="0.02"/>
          </sensor>
        </mujoco>
        """
    )
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=1)
        scratch = make_optimized_fwd_position_scratch(model, data, "cpu")
        assert model.nhistory > 0
        monkeypatch.setattr(
            optimized.wp,
            "empty_like",
            lambda *_args, **_kwargs: pytest.fail(
                "sensor history allocated after scratch construction"
            ),
        )
        scratch.sensor_pos_call(model, data)
        wp.synchronize()


def test_energy_velocity_reuses_preallocated_array(monkeypatch, tmp_path):
    import instantmimic.utils.mjw_step_optimized as optimized

    wp.config.kernel_cache_dir = str(tmp_path / "warp-cache")
    mj_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option><flag energy="enable"/></option>
          <worldbody>
            <body><joint name="j"/><geom size="0.1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mj_model)
        data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=1)
        wp.to_torch(data.qvel).fill_(0.3)
        mjw.forward(model, data)
        expected = wp.to_torch(data.energy).clone()
        scratch = make_optimized_fwd_position_scratch(model, data, "cpu")
        assert scratch.energy_vel_call is not None
        monkeypatch.setattr(
            optimized.wp,
            "zeros",
            lambda *_args, **_kwargs: pytest.fail(
                "kinetic energy allocated after scratch construction"
            ),
        )
        data.energy.zero_()
        scratch.energy_vel_call(model, data)
        wp.synchronize()
        torch.testing.assert_close(wp.to_torch(data.energy), expected)


def test_optimized_fwd_acceleration_fuses_pending_actuator_limits(monkeypatch):
    launched = []

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        launched.append(getattr(kernel, "__name__", str(kernel)))

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)
    monkeypatch.setattr(
        optimized.mjw_forward,
        "xfrc_accumulate",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(optimized.smooth, "solve_m", lambda *_args, **_kwargs: None)

    model = SimpleNamespace(
        is_sparse=False,
        nv=2,
        ngravcomp=0,
        opt=SimpleNamespace(disableflags=0),
        jnt_actfrclimited=object(),
        jnt_actgravcomp=object(),
        jnt_actfrcrange=object(),
        dof_jntid=object(),
    )
    data = SimpleNamespace(
        nworld=1,
        qfrc_applied=object(),
        qfrc_bias=object(),
        qfrc_passive=object(),
        qfrc_actuator=object(),
        qfrc_gravcomp=object(),
        qfrc_smooth=object(),
        qacc_smooth=object(),
    )

    optimized.fwd_acceleration_optimized(
        model,
        data,
        factorize=False,
        actuator_limits_pending=True,
    )

    assert "_qfrc_actuator_gravcomp_limits_and_qfrc_smooth" in launched
    assert "_qfrc_smooth" not in launched


def test_forward_after_position_precomputes_optimized_implicit_deriv_before_solver(
    monkeypatch,
):
    calls = []

    class Zeroable:
        def zero_(self):
            calls.append(("zero", "buffer"))

    def fake_fwd_velocity(
        model,
        data,
        streams,
        *,
        scratch=None,
        start_actuation=False,
        defer_actuator_limits=False,
        **_kwargs,
    ):
        calls.append(("fwd_velocity", start_actuation, defer_actuator_limits))
        return SimpleNamespace(
            actuation_started=False,
            actuation_stream=None,
            actuator_limits_pending=False,
        )

    def fake_deriv(model, data, qderiv, scratch):
        calls.append(("deriv", qderiv, scratch))

    fake_deriv.__name__ = "deriv_smooth_vel_dense_actuator_centric"

    def fake_call_in_stream(stream, fn, *args):
        calls.append(("call_in_stream", stream, getattr(fn, "__name__", str(fn))))
        fn(*args)

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append((name,))

        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "fwd_velocity_optimized", fake_fwd_velocity)
    monkeypatch.setattr(optimized, "fwd_acceleration_optimized", mark("acceleration"))
    monkeypatch.setattr(
        optimized,
        "deriv_smooth_vel_dense_actuator_centric",
        fake_deriv,
    )
    monkeypatch.setattr(optimized, "_call_in_stream", fake_call_in_stream)
    monkeypatch.setattr(optimized.mjw_forward, "fwd_actuation", mark("actuation"))
    monkeypatch.setattr(optimized.sensor, "sensor_pos", mark("sensor_pos"))
    monkeypatch.setattr(optimized.sensor, "sensor_vel", mark("sensor_vel"))
    monkeypatch.setattr(optimized.sensor, "sensor_acc", mark("sensor_acc"))
    monkeypatch.setattr(optimized.solver, "solve", mark("solve"))

    model = SimpleNamespace(
        has_fluid=False,
        opt=SimpleNamespace(
            enableflags=0,
            disableflags=0,
            integrator=optimized.IntegratorType.IMPLICITFAST,
        ),
        sensor_e_potential=0,
        sensor_e_kinetic=0,
        callback=SimpleNamespace(control=None),
        is_sparse=False,
        nactuator=1,
        nu=1,
        nv=2,
        ntendon=0,
        nsensor=0,
    )
    data = SimpleNamespace(njmax=1, sensordata=Zeroable(), energy=Zeroable())
    scratch = SimpleNamespace(
        implicit_actuator_vel=object(),
        implicit_qderiv_pair_mask=object(),
        implicit_max_moment_rownnz=1,
        implicit_qderiv=object(),
        implicit_qacc=object(),
    )
    streams = tuple(f"stream-{i}" for i in range(14))

    pending = optimized._forward_after_position(
        model,
        data,
        streams=streams,
        scratch=scratch,
        use_optimized_kernel=True,
    )

    assert calls.index(
        ("call_in_stream", "stream-9", "deriv_smooth_vel_dense_actuator_centric")
    ) < calls.index(("solve",))
    assert pending.started is True
    assert pending.qderiv is scratch.implicit_qderiv
    assert pending.qacc is scratch.implicit_qacc


def test_solve_with_scratch_reuses_monolithic_context_and_condition(monkeypatch):
    import instantmimic.utils.mjw_step_optimized as optimized

    calls = []
    context = SimpleNamespace(
        done=object(),
        grad=object(),
        Mgrad=object(),
        search=object(),
        search_dot=object(),
        prev_grad=object(),
        prev_Mgrad=object(),
        search_unchanged=SimpleNamespace(zero_=lambda: None),
        grad_scale=object(),
    )

    class Condition:
        def fill_(self, value):
            calls.append(("fill", value))

    condition = Condition()
    model = SimpleNamespace(
        stat=SimpleNamespace(meaninertia=object()),
        nv=2,
        ntree=1,
        is_sparse=False,
        opt=SimpleNamespace(
            disableflags=0,
            enableflags=0,
            solver=optimized.solver.types.SolverType.CG,
            tolerance=object(),
            iterations=7,
            graph_conditional=True,
            integrator=optimized.IntegratorType.IMPLICITFAST,
        ),
        block_dim=SimpleNamespace(solve_init_search_cg=32),
    )
    data = SimpleNamespace(
        njmax=4,
        nworld=3,
        nefc=object(),
        qacc=object(),
        qacc_warmstart=object(),
        qacc_smooth=object(),
        qfrc_smooth=object(),
        qfrc_constraint=object(),
        solver_niter=object(),
        efc=SimpleNamespace(Ma=object()),
    )
    scratch = SimpleNamespace(
        solver_context=context,
        solver_nsolving=condition,
        solver_uses_compact=False,
    )

    monkeypatch.setattr(
        optimized.solver,
        "_solve_init_dof",
        lambda warmstart, sparse: ("init-dof", warmstart, sparse),
    )
    monkeypatch.setattr(
        optimized.solver,
        "init_context",
        lambda m, d, ctx, grad=True, compact=False: calls.append(
            ("init", ctx, grad, compact)
        ),
    )
    monkeypatch.setattr(
        optimized.solver,
        "_update_gradient",
        lambda m, d, ctx, compact=False: calls.append(("gradient", ctx)),
    )
    monkeypatch.setattr(
        optimized.wp,
        "launch",
        lambda kernel, **kwargs: calls.append(("launch", kernel, kwargs)),
    )
    monkeypatch.setattr(
        optimized.wp,
        "launch_tiled",
        lambda *args, **kwargs: calls.append(("launch_tiled", kwargs.get("outputs"))),
    )
    monkeypatch.setattr(
        optimized.wp,
        "capture_while",
        lambda cond, **kwargs: calls.append(("capture_while", cond, kwargs)),
    )
    monkeypatch.setattr(optimized.solver, "_use_incremental", lambda _m: False)

    optimized.solve_with_scratch(model, data, scratch)
    optimized.solve_with_scratch(model, data, scratch)

    assert calls.count(("init", context, False, False)) == 2
    assert calls.count(("gradient", context)) == 2
    assert calls.count(("fill", 3)) == 2
    conditional_calls = [call for call in calls if call[0] == "capture_while"]
    assert len(conditional_calls) == 2
    assert all(call[1] is condition for call in conditional_calls)
    assert all(call[2]["ctx"] is context for call in conditional_calls)
    init_launches = [
        call for call in calls if call[:2] == ("launch", ("init-dof", True, False))
    ]
    assert len(init_launches) == 2


def test_fwd_acceleration_sleep_rebuilds_active_dofs_and_uses_compact_solve(
    monkeypatch,
):
    import instantmimic.utils.mjw_step_optimized as optimized

    calls = []
    monkeypatch.setattr(
        optimized,
        "_launch_qfrc_smooth",
        lambda m, d: calls.append("qfrc"),
    )
    monkeypatch.setattr(
        optimized.mjw_forward,
        "xfrc_accumulate",
        lambda m, d, qfrc: calls.append("xfrc"),
    )
    monkeypatch.setattr(
        optimized.island,
        "update_active_dofs",
        lambda m, d: calls.append("active"),
    )
    monkeypatch.setattr(
        optimized.solver,
        "smooth_solve_compact",
        lambda m, d: calls.append("compact"),
    )
    monkeypatch.setattr(
        optimized,
        "_factor_solve_i_dense",
        lambda *_args, **_kwargs: pytest.fail("sleep path used full factor solve"),
    )

    model = SimpleNamespace(
        nv=3,
        opt=SimpleNamespace(
            enableflags=optimized.EnableBit.SLEEP,
            disableflags=0,
        ),
    )
    data = SimpleNamespace(qfrc_smooth=object())

    optimized.fwd_acceleration_optimized(model, data, factorize=True)

    assert calls == ["qfrc", "xfrc", "active", "compact"]


def test_solver_compaction_matches_312_sleep_flag_contract():
    import instantmimic.utils.mjw_step_optimized as optimized

    model = SimpleNamespace(
        opt=SimpleNamespace(
            enableflags=optimized.EnableBit.SLEEP,
            disableflags=optimized.DisableBit.ISLAND,
        )
    )

    assert not optimized._sleep_enabled(model)
    assert optimized._solver_compact_enabled(model)


def test_sleep_discovery_includes_single_tree_models(monkeypatch):
    calls = []

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append(name)

        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.smooth, "com_pos", mark("com_pos"))
    monkeypatch.setattr(optimized.smooth, "camlight", mark("camlight"))
    monkeypatch.setattr(optimized.smooth, "flex", mark("flex"))
    monkeypatch.setattr(optimized.smooth, "tendon", mark("tendon"))
    monkeypatch.setattr(optimized.smooth, "crb", mark("crb"))
    monkeypatch.setattr(optimized.smooth, "tendon_armature", mark("armature"))
    monkeypatch.setattr(optimized.smooth, "transmission", mark("transmission"))
    monkeypatch.setattr(optimized.constraint, "make_constraint", mark("constraint"))
    monkeypatch.setattr(
        optimized,
        "sleep",
        SimpleNamespace(
            update_sleep=mark("update_sleep"),
            wake_equality=mark("wake_equality"),
        ),
    )
    monkeypatch.setattr(
        optimized,
        "island",
        SimpleNamespace(island=mark("island")),
    )
    model = SimpleNamespace(
        opt=SimpleNamespace(
            run_collision_detection=False,
            disableflags=0,
            enableflags=optimized.EnableBit.SLEEP,
        ),
        ntendon=0,
        neq=0,
        ntree=1,
    )

    optimized.fwd_position_skip_kinematics(model, object(), factorize=False)

    assert calls.count("island") == 1
    assert calls.index("update_sleep") < calls.index("island")
    assert calls.index("island") < calls.index("transmission")


def test_preallocated_upstream_call_reuses_bound_buffers(monkeypatch):
    import instantmimic.utils.mjw_step_optimized as optimized

    events = []

    class Buffer:
        def fill_(self, value):
            events.append(("fill", self, value))

    empty_buffer = Buffer()
    full_buffer = Buffer()

    def upstream_body():
        first = wp.empty((3,), dtype=float)
        second = wp.full((4,), 7, dtype=int)
        return first, second

    bound_call = optimized._PreallocatedWarpCall(
        upstream_body,
        (("empty", empty_buffer), ("full", full_buffer)),
    )
    monkeypatch.setattr(
        optimized.wp,
        "empty",
        lambda *_args, **_kwargs: pytest.fail("live allocation was not intercepted"),
    )
    monkeypatch.setattr(
        optimized.wp,
        "full",
        lambda *_args, **_kwargs: pytest.fail("live allocation was not intercepted"),
    )

    first_result = bound_call()
    second_result = bound_call()

    assert first_result == (empty_buffer, full_buffer)
    assert second_result == first_result
    assert events == [
        ("fill", full_buffer, 7),
        ("fill", full_buffer, 7),
    ]


def test_preallocated_upstream_call_reuses_clone_buffer(monkeypatch):
    import instantmimic.utils.mjw_step_optimized as optimized

    source = object()
    clone_buffer = object()
    copies = []

    def upstream_body():
        return wp.clone(source)

    bound_call = optimized._PreallocatedWarpCall(
        upstream_body,
        (("clone", clone_buffer),),
    )
    monkeypatch.setattr(
        optimized.wp,
        "clone",
        lambda *_args, **_kwargs: pytest.fail("live allocation was not intercepted"),
    )
    monkeypatch.setattr(
        optimized.wp,
        "copy",
        lambda destination, value: copies.append((destination, value)),
    )

    assert bound_call() is clone_buffer
    assert bound_call() is clone_buffer
    assert copies == [(clone_buffer, source), (clone_buffer, source)]


def test_preallocated_upstream_call_reuses_ones_buffer(monkeypatch):
    import instantmimic.utils.mjw_step_optimized as optimized

    ones_buffer = SimpleNamespace(fill_=lambda value: fills.append(value))
    fills = []

    def upstream_body():
        return wp.ones((2, 3), dtype=int)

    bound_call = optimized._PreallocatedWarpCall(
        upstream_body,
        (("ones", ones_buffer),),
    )
    monkeypatch.setattr(
        optimized.wp,
        "ones",
        lambda *_args, **_kwargs: pytest.fail("live allocation was not intercepted"),
    )

    assert bound_call() is ones_buffer
    assert bound_call() is ones_buffer
    assert fills == [1, 1]


@pytest.mark.parametrize(
    ("request_shape", "request_dtype", "message"),
    [((2, 4), int, "shape changed"), ((2, 3), float, "dtype changed")],
)
def test_preallocated_upstream_call_rejects_changed_requests(
    request_shape, request_dtype, message
):
    import instantmimic.utils.mjw_step_optimized as optimized

    with wp.ScopedDevice("cpu"):
        buffer = wp.zeros((2, 3), dtype=int)
        same = optimized._PreallocatedWarpCall(
            lambda: wp.zeros((2, 3), dtype=int), (("zeros", buffer),)
        )
        assert same() is buffer
        changed = optimized._PreallocatedWarpCall(
            lambda: wp.zeros(request_shape, dtype=request_dtype), (("zeros", buffer),)
        )
        with pytest.raises(RuntimeError, match=message):
            changed()


def test_persistent_step_scratch_rejects_the_discrete_integrator():
    import instantmimic.utils.mjw_step_optimized as optimized

    mjm = mujoco.MjModel.from_xml_string(
        '<mujoco><option integrator="discrete"/><worldbody><body><freejoint/>'
        '<geom size="0.1"/></body></worldbody></mujoco>'
    )
    with wp.ScopedDevice("cpu"):
        model = mjw.put_model(mjm)
        data = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=1)
        with pytest.raises(ValueError, match="DISCRETE"):
            optimized.make_optimized_fwd_position_scratch(model, data, "cpu")


def test_rk4_capture_requires_persistent_scratch_and_no_graph_conditionals():
    import instantmimic.utils.mjw_step_optimized as optimized

    model = SimpleNamespace(
        opt=SimpleNamespace(
            integrator=optimized.IntegratorType.RK4,
            enableflags=0,
            graph_conditional=False,
        )
    )
    assert not optimized.supports_allocation_free_implicit_capture(
        model,
        SimpleNamespace(),
        use_optimized_kernel=True,
    )
    scratch = SimpleNamespace(rk4=object())
    assert optimized.supports_allocation_free_implicit_capture(
        model,
        scratch,
        use_optimized_kernel=True,
    )
    model.opt.graph_conditional = True
    assert not optimized.supports_allocation_free_implicit_capture(
        model,
        scratch,
        use_optimized_kernel=True,
    )


def test_sleep_graph_conditional_is_not_reported_as_nested_capture_safe():
    import instantmimic.utils.mjw_step_optimized as optimized

    model = SimpleNamespace(
        opt=SimpleNamespace(
            integrator=optimized.IntegratorType.IMPLICITFAST,
            enableflags=optimized.EnableBit.SLEEP,
            disableflags=0,
            graph_conditional=True,
        )
    )
    assert not optimized.supports_allocation_free_implicit_capture(
        model,
        SimpleNamespace(),
        use_optimized_kernel=True,
    )


def test_euler_integration_uses_preallocated_call(monkeypatch):
    import instantmimic.utils.mjw_step_optimized as optimized

    calls = []
    model = SimpleNamespace(
        opt=SimpleNamespace(integrator=optimized.IntegratorType.EULER)
    )
    data = object()
    scratch = SimpleNamespace(
        euler_call=lambda m, d: calls.append(("preallocated", m, d))
    )
    monkeypatch.setattr(
        optimized.mjw_forward,
        "euler",
        lambda *_args: pytest.fail("stock Euler must not allocate during capture"),
    )

    optimized._integrate(model, data, scratch=scratch)

    assert calls == [("preallocated", model, data)]


def test_forward_after_position_does_not_fork_implicit_deriv_before_conditional_solver(
    monkeypatch,
):
    calls = []

    class Zeroable:
        def zero_(self):
            calls.append(("zero", "buffer"))

    def fake_fwd_velocity(
        model,
        data,
        streams,
        *,
        scratch=None,
        start_actuation=False,
        defer_actuator_limits=False,
        **_kwargs,
    ):
        calls.append(("fwd_velocity", start_actuation, defer_actuator_limits))
        return SimpleNamespace(
            actuation_started=False,
            actuation_stream=None,
            actuator_limits_pending=False,
        )

    def fail_deriv(*_args, **_kwargs):
        msg = "conditional solver path must not fork implicit deriv before solve"
        raise AssertionError(msg)

    fail_deriv.__name__ = "deriv_smooth_vel_dense_actuator_centric"

    def fail_call_in_stream(*_args, **_kwargs):
        msg = "conditional solver path must not start side-stream deriv work"
        raise AssertionError(msg)

    def mark(name):
        def _inner(*_args, **_kwargs):
            calls.append((name,))

        return _inner

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized, "fwd_velocity_optimized", fake_fwd_velocity)
    monkeypatch.setattr(optimized, "fwd_acceleration_optimized", mark("acceleration"))
    monkeypatch.setattr(
        optimized,
        "deriv_smooth_vel_dense_actuator_centric",
        fail_deriv,
    )
    monkeypatch.setattr(optimized, "_call_in_stream", fail_call_in_stream)
    monkeypatch.setattr(optimized.mjw_forward, "fwd_actuation", mark("actuation"))
    monkeypatch.setattr(optimized.sensor, "sensor_pos", mark("sensor_pos"))
    monkeypatch.setattr(optimized.sensor, "sensor_vel", mark("sensor_vel"))
    monkeypatch.setattr(optimized.sensor, "sensor_acc", mark("sensor_acc"))
    monkeypatch.setattr(optimized.solver, "solve", mark("solve"))

    model = SimpleNamespace(
        has_fluid=False,
        opt=SimpleNamespace(
            enableflags=0,
            disableflags=0,
            integrator=optimized.IntegratorType.IMPLICITFAST,
            graph_conditional=True,
        ),
        sensor_e_potential=0,
        sensor_e_kinetic=0,
        callback=SimpleNamespace(control=None),
        is_sparse=False,
        nactuator=1,
        nu=1,
        nv=2,
        ntendon=0,
        nsensor=0,
    )
    data = SimpleNamespace(njmax=1, sensordata=Zeroable(), energy=Zeroable())
    scratch = SimpleNamespace(
        implicit_actuator_vel=object(),
        implicit_qderiv_pair_mask=object(),
        implicit_max_moment_rownnz=1,
        implicit_qderiv=object(),
        implicit_qacc=object(),
    )
    streams = tuple(f"stream-{i}" for i in range(14))

    pending = optimized._forward_after_position(
        model,
        data,
        streams=streams,
        scratch=scratch,
        use_optimized_kernel=True,
    )

    assert ("solve",) in calls
    assert pending.started is False


def test_integrate_uses_precomputed_optimized_implicit_deriv_after_waiting(
    monkeypatch,
):
    calls = []

    def fake_wait_stream(stream):
        calls.append(("wait", stream))

    def fake_no_store(model, data, matrix, x, y):
        calls.append(("factor_solve", matrix, x, y))

    def fake_advance(model, data, qacc):
        calls.append(("advance", qacc))

    def fail_recompute(*_args, **_kwargs):
        msg = "precomputed implicit derivative should not be recomputed"
        raise AssertionError(msg)

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "wait_stream", fake_wait_stream)
    monkeypatch.setattr(
        optimized,
        "_factor_solve_i_dense_no_l_store",
        fake_no_store,
    )
    monkeypatch.setattr(optimized.mjw_forward, "_advance", fake_advance)
    monkeypatch.setattr(optimized, "implicit_dense_deriv_optimized", fail_recompute)

    model = SimpleNamespace(
        opt=SimpleNamespace(integrator=optimized.IntegratorType.IMPLICITFAST),
    )
    data = SimpleNamespace(efc=SimpleNamespace(Ma="ma"))
    pending = optimized.OptimizedImplicitDerivResult(
        started=True,
        stream="deriv-stream",
        qderiv="qderiv",
        qacc="qacc",
    )

    optimized._integrate(
        model,
        data,
        use_optimized_kernel=True,
        pending_implicit_deriv=pending,
    )

    assert calls == [
        ("wait", "deriv-stream"),
        ("factor_solve", "qderiv", "qacc", "ma"),
        ("advance", "qacc"),
    ]


def test_integrate_uses_optimized_implicit_derivative_when_scratch_available(
    monkeypatch,
):
    calls = []

    def fake_implicit(m, d, scratch):
        calls.append(("optimized_implicit", m, d, scratch))

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(
        optimized,
        "implicit_dense_deriv_optimized",
        fake_implicit,
        raising=False,
    )

    model = SimpleNamespace(
        opt=SimpleNamespace(integrator=optimized.IntegratorType.IMPLICITFAST),
        is_sparse=False,
        nactuator=1,
        nu=1,
    )
    data = object()
    scratch = SimpleNamespace(
        implicit_actuator_vel=object(),
        implicit_qderiv_pair_mask=object(),
        implicit_max_moment_rownnz=1,
    )

    optimized._integrate(model, data, scratch=scratch)

    assert calls == [("optimized_implicit", model, data, scratch)]


def test_integrate_applies_external_velocity_hook_with_sparse_constraints(
    monkeypatch,
):
    calls = []

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(
        optimized,
        "deriv_smooth_vel_dense_actuator_centric",
        lambda m, d, out, scratch: calls.extend([("deriv", out), ("hook", out)]),
    )
    monkeypatch.setattr(
        optimized,
        "factor_solve_i_no_l_store",
        lambda m, d, qderiv, qld, diag, qacc, ma, tree_factor_solve=None: calls.append(
            ("factor", qderiv, qacc, ma)
        ),
    )
    monkeypatch.setattr(
        optimized.mjw_forward,
        "_advance",
        lambda m, d, qacc: calls.append(("advance", qacc)),
    )
    monkeypatch.setattr(
        optimized.mjw_forward,
        "implicit",
        lambda *_args: pytest.fail("sparse hook path must not use allocating implicit"),
    )

    model = SimpleNamespace(
        has_fluid=False,
        is_sparse=True,
        nactuator=0,
        nu=0,
        nv=2,
        opt=SimpleNamespace(
            integrator=optimized.IntegratorType.IMPLICITFAST,
            disableflags=0,
        ),
    )
    data = SimpleNamespace(efc=SimpleNamespace(Ma="ma"))
    scratch = SimpleNamespace(
        implicit_deriv_call=lambda m, d, out: calls.append(("deriv", out)),
        implicit_qh_contributors=(
            optimized.ImplicitQHContributor(
                "test", lambda out: calls.append(("hook", out))
            ),
        ),
        implicit_qld="qld",
        implicit_ql_diag_inv="diag",
        implicit_qderiv="qderiv",
        implicit_qacc="qacc",
        implicit_call=object(),
    )

    optimized._integrate(
        model,
        data,
        scratch=scratch,
        use_optimized_kernel=True,
    )

    assert calls == [
        ("deriv", "qderiv"),
        ("hook", "qderiv"),
        ("factor", "qderiv", "qacc", "ma"),
        ("advance", "qacc"),
    ]


def test_integrate_applies_dense_contributors_with_optimized_kernels_disabled(
    monkeypatch,
):
    calls = []
    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(
        optimized,
        "factor_solve_i_no_l_store",
        lambda m, d, qderiv, qld, diag, qacc, ma, tree_factor_solve=None: calls.append(
            ("factor", qderiv, qacc, ma)
        ),
    )
    monkeypatch.setattr(
        optimized,
        "_advance_with_scratch",
        lambda m, d, qacc, scratch: calls.append(("advance", qacc)),
    )
    model = SimpleNamespace(
        is_sparse=False,
        nactuator=0,
        nu=0,
        nv=2,
        opt=SimpleNamespace(
            integrator=optimized.IntegratorType.IMPLICITFAST,
            disableflags=(
                optimized.DisableBit.ACTUATION
                | optimized.DisableBit.SPRING
                | optimized.DisableBit.DAMPER
            ),
        ),
    )
    data = SimpleNamespace(efc=SimpleNamespace(Ma="ma"))
    scratch = SimpleNamespace(
        implicit_deriv_call=lambda m, d, out: calls.append(("deriv", out)),
        implicit_qh_contributors=(
            optimized.ImplicitQHContributor(
                "contact", lambda out: calls.append(("contact", out))
            ),
        ),
        implicit_qld="qld",
        implicit_ql_diag_inv="diag",
        implicit_qderiv="qderiv",
        implicit_qacc="qacc",
        implicit_call=object(),
    )

    optimized._integrate(
        model,
        data,
        scratch=scratch,
        use_optimized_kernel=False,
    )

    assert calls == [
        ("deriv", "qderiv"),
        ("contact", "qderiv"),
        ("factor", "qderiv", "qacc", "ma"),
        ("advance", "qacc"),
    ]


def test_optimized_implicit_dense_uses_no_store_factor_solve(monkeypatch):
    calls = []

    def fake_empty(shape, *, dtype):
        return ("empty", shape, dtype)

    def fake_deriv(model, data, qderiv, scratch):
        calls.append(("deriv", model, data, qderiv, scratch))

    def fake_no_store(model, data, matrix, x, y):
        calls.append(("no_store_factor_solve", model, data, matrix, x, y))

    def fail_stock_factor_solve(*_args, **_kwargs):
        msg = "optimized implicit dense path should not store qLD"
        raise AssertionError(msg)

    def fake_advance(model, data, qacc):
        calls.append(("advance", model, data, qacc))

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "empty", fake_empty)
    monkeypatch.setattr(
        optimized,
        "deriv_smooth_vel_dense_actuator_centric",
        fake_deriv,
    )
    monkeypatch.setattr(
        optimized,
        "_factor_solve_i_dense_no_l_store",
        fake_no_store,
    )
    monkeypatch.setattr(optimized.smooth, "factor_solve_i", fail_stock_factor_solve)
    monkeypatch.setattr(optimized.mjw_forward, "_advance", fake_advance)

    model = SimpleNamespace(
        has_fluid=False,
        opt=SimpleNamespace(disableflags=0),
        is_sparse=False,
        nactuator=1,
        nu=1,
        nv=2,
    )
    data = SimpleNamespace(
        nworld=1,
        qM=SimpleNamespace(shape=(1, 4, 4)),
        efc=SimpleNamespace(Ma=object()),
    )
    scratch = SimpleNamespace(
        implicit_actuator_vel=object(),
        implicit_qderiv_pair_mask=object(),
        implicit_max_moment_rownnz=1,
    )

    optimized.implicit_dense_deriv_optimized(model, data, scratch)

    assert [call[0] for call in calls] == [
        "deriv",
        "no_store_factor_solve",
        "advance",
    ]


def test_optimized_deriv_skips_tendon_damping_when_no_tendons(monkeypatch):
    launched = []

    class Out:
        shape = (1, 4, 4)

        def zero_(self):
            launched.append("zero")

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        launched.append(kernel)

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)
    monkeypatch.setattr(optimized.wp, "copy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(optimized.derivative, "_qderiv_actuator_passive_vel", object())
    monkeypatch.setattr(
        optimized.derivative,
        "_qderiv_actuator_passive",
        object(),
    )
    tendon_kernel = object()
    monkeypatch.setattr(
        optimized.derivative,
        "_qderiv_tendon_damping",
        tendon_kernel,
    )

    model = SimpleNamespace(
        has_fluid=False,
        actuator_ctrlspec=object(),
        actuator_acc0=object(),
        actuator_lengthrange=object(),
        opt=SimpleNamespace(disableflags=0, timestep=object()),
        is_sparse=False,
        nactuator=1,
        nu=1,
        nv=2,
        ntendon=0,
        qM_fullm_i=SimpleNamespace(size=1),
        qM_fullm_j=object(),
        actuator_dyntype=object(),
        actuator_gaintype=object(),
        actuator_biastype=object(),
        actuator_actadr=object(),
        actuator_actnum=object(),
        actuator_forcelimited=object(),
        actuator_actlimited=object(),
        actuator_dynprm=object(),
        actuator_gainprm=object(),
        actuator_biasprm=object(),
        actuator_actearly=object(),
        actuator_forcerange=object(),
        actuator_actrange=object(),
        dof_damping=object(),
        dof_dampingpoly=object(),
        ten_J_rownnz=object(),
        ten_J_rowadr=object(),
        ten_J_colind=object(),
        tendon_damping=object(),
        tendon_dampingpoly=object(),
    )
    data = SimpleNamespace(
        actuator_length=object(),
        actuator_velocity=object(),
        nworld=1,
        act=object(),
        ctrl=object(),
        act_dot=object(),
        actuator_force=object(),
        moment_rownnz=object(),
        moment_rowadr=object(),
        moment_colind=object(),
        actuator_moment=object(),
        qvel=object(),
        qM=object(),
        ten_J=object(),
        ten_velocity=object(),
    )
    scratch = SimpleNamespace(
        implicit_actuator_vel=object(),
        implicit_qderiv_pair_mask=object(),
        implicit_max_moment_rownnz=1,
    )

    optimized.deriv_smooth_vel_dense_actuator_centric(
        model,
        data,
        Out(),
        scratch,
    )

    assert tendon_kernel not in launched


def test_optimized_deriv_uses_stock_passive_finalize(monkeypatch):
    launched = []

    class Out:
        shape = (1, 4, 4)

        def zero_(self):
            launched.append("zero")

    def fake_launch(kernel, *, dim, inputs, outputs, **_kwargs):
        launched.append(kernel)

    import instantmimic.utils.mjw_step_optimized as optimized

    monkeypatch.setattr(optimized.wp, "launch", fake_launch)
    monkeypatch.setattr(optimized.wp, "copy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(optimized.derivative, "_qderiv_actuator_passive_vel", object())
    stock_passive_kernel = object()
    monkeypatch.setattr(
        optimized.derivative,
        "_qderiv_actuator_passive",
        stock_passive_kernel,
    )
    monkeypatch.setattr(
        optimized.derivative,
        "_qderiv_tendon_damping",
        object(),
    )

    model = SimpleNamespace(
        has_fluid=False,
        actuator_ctrlspec=object(),
        actuator_acc0=object(),
        actuator_lengthrange=object(),
        opt=SimpleNamespace(disableflags=0, timestep=object()),
        is_sparse=False,
        nactuator=1,
        nu=1,
        nv=2,
        ntendon=0,
        qM_fullm_i=SimpleNamespace(size=1),
        qM_fullm_j=object(),
        actuator_dyntype=object(),
        actuator_gaintype=object(),
        actuator_biastype=object(),
        actuator_actadr=object(),
        actuator_actnum=object(),
        actuator_forcelimited=object(),
        actuator_actlimited=object(),
        actuator_dynprm=object(),
        actuator_gainprm=object(),
        actuator_biasprm=object(),
        actuator_actearly=object(),
        actuator_forcerange=object(),
        actuator_actrange=object(),
        dof_damping=object(),
        dof_dampingpoly=object(),
        ten_J_rownnz=object(),
        ten_J_rowadr=object(),
        ten_J_colind=object(),
        tendon_damping=object(),
        tendon_dampingpoly=object(),
    )
    data = SimpleNamespace(
        actuator_length=object(),
        actuator_velocity=object(),
        nworld=1,
        act=object(),
        ctrl=object(),
        act_dot=object(),
        actuator_force=object(),
        moment_rownnz=object(),
        moment_rowadr=object(),
        moment_colind=object(),
        actuator_moment=object(),
        qvel=object(),
        qM=object(),
        ten_J=object(),
        ten_velocity=object(),
    )
    scratch = SimpleNamespace(
        implicit_actuator_vel=object(),
        implicit_qderiv_pair_mask=object(),
        implicit_max_moment_rownnz=1,
    )

    optimized.deriv_smooth_vel_dense_actuator_centric(
        model,
        data,
        Out(),
        scratch,
    )

    assert "zero" in launched
    assert stock_passive_kernel in launched
    assert "_qderiv_dense_passive_baseline" not in {
        getattr(kernel, "__name__", str(kernel)) for kernel in launched
    }


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_parallel_kinematics_tail_matches_stock_kinematics():
    nworld = 16
    mj_model, model, stock_data, parallel_data = _make_data_pair(nworld)
    _seed_state(mj_model, stock_data, parallel_data, nworld)
    streams = make_optimized_fwd_position_streams("cuda:0")

    mjw.kinematics(model, stock_data)
    kinematics_parallel_tail(model, parallel_data, streams)
    wp.synchronize()

    for name in ("xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat"):
        assert torch.allclose(
            wp.to_torch(getattr(stock_data, name)),
            wp.to_torch(getattr(parallel_data, name)),
            atol=1.0e-6,
            rtol=1.0e-6,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_parallel_com_pos_vel_matches_stock_com_pos_vel():
    nworld = 16
    mj_model, model, stock_data, parallel_data = _make_data_pair(nworld)
    _seed_state(mj_model, stock_data, parallel_data, nworld)
    streams = make_optimized_fwd_position_streams("cuda:0")

    mjw.kinematics(model, stock_data)
    mjw.com_pos(model, stock_data)
    mjw.com_vel(model, stock_data)

    mjw.kinematics(model, parallel_data)
    com_pos_vel_parallel(model, parallel_data, streams)
    wp.synchronize()

    for name in ("subtree_com", "cinert", "cdof", "cvel", "cdof_dot"):
        assert torch.allclose(
            wp.to_torch(getattr(stock_data, name)),
            wp.to_torch(getattr(parallel_data, name)),
            atol=1.0e-6,
            rtol=1.0e-6,
        )


def test_reward_launch_uses_small_block_dim_for_more_ctas(monkeypatch):
    from instantmimic.envs.snu.tasks.snu_imitation import snu_imitation_env as env_mod
    from instantmimic.envs.tracking.reward_kernels import (
        REWARD_KERNEL_BLOCK_DIM,
    )

    calls = []

    def fake_launch(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(env_mod.wp, "launch", fake_launch)

    env = env_mod.SNUImitationEnv.__new__(env_mod.SNUImitationEnv)
    env.adaptive_sampling_state = (
        env_mod.adaptive_sampling_runtime.AdaptiveSamplingRuntimeState()
    )
    env.cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=2048))
    env.reward_settings = SimpleNamespace()
    reward = torch.zeros(2048)
    reward_terms = torch.zeros((2048, 1))
    env.reward_runtime_state = env_mod.reward_runtime.RewardRuntimeState(
        reward=reward,
        reward_terms=reward_terms,
        reward_term_names=("reward",),
        substep_mean_enabled=False,
        substep_reward_accumulator=torch.zeros_like(reward),
        substep_reward_terms_accumulator=torch.zeros_like(reward_terms),
        substep_reward_sample=torch.zeros_like(reward),
        substep_reward_terms_sample=torch.zeros_like(reward_terms),
        substep_reference_frame_index=torch.zeros(2048, dtype=torch.int64),
        compute_rewards_kernel=object(),
    )
    env.mjw_data = SimpleNamespace(
        qpos=object(),
        qvel=object(),
        xpos=object(),
        xquat=object(),
        cvel=object(),
        subtree_com=object(),
    )
    env.reward_motion_qpos = object()
    env.reward_motion_qvel = object()
    env.reward_reference_body_state = object()
    env.reward_ref_com = object()
    env.motion_clip_idx = object()
    env.ref_frame_idx = object()
    env.reward_motion_clip_start = object()
    env.reward_motion_clip_length = object()
    env.reward_motion_clip_loop_root_quat = object()
    env.reward_motion_clip_loop_root_pos = object()
    env.hinge_qpos_idx_i32 = object()
    env.hinge_dof_idx_i32 = object()
    env.ball_qpos_adr_i32 = object()
    env.ball_dof_adr_i32 = object()
    env.dof_q_scale = object()
    env.dof_qd_scale = object()
    env.body_state_frame_sigma_multiplier = object()
    env.body_state_sigma_sq = object()
    env.body_state_sigma_knot_start = object()
    env.adaptive_sampling_state.sampling_enabled = False

    assert env._get_rewards() is reward

    assert calls[0][1]["block_dim"] == REWARD_KERNEL_BLOCK_DIM


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_step_optimized_fwd_position_matches_stock_step():
    nworld = 16
    mj_model, model, stock_data, parallel_data = _make_data_pair(nworld)
    _seed_state(mj_model, stock_data, parallel_data, nworld)
    streams = make_optimized_fwd_position_streams("cuda:0")

    for _ in range(4):
        mjw.step(model, stock_data)
        step_optimized_fwd_position(model, parallel_data, streams)
    wp.synchronize()

    assert torch.allclose(
        wp.to_torch(stock_data.qpos),
        wp.to_torch(parallel_data.qpos),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        wp.to_torch(stock_data.qvel),
        wp.to_torch(parallel_data.qvel),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


@pytest.mark.cuda
def test_parallel_step_preserves_connect_equality_velocity_dependency():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    mj_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="a" pos="0 0 1">
              <joint type="free"/>
              <geom type="sphere" size="0.1"/>
            </body>
            <body name="b" pos="0 0 2">
              <joint type="free"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <equality>
            <connect body1="a" body2="b" anchor="0 0 1.5"/>
          </equality>
        </mujoco>
        """
    )
    model = mjw.put_model(mj_model)
    sequential = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=4)
    parallel = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=4)
    wp.to_torch(sequential.qvel).fill_(0.02)
    wp.to_torch(parallel.qvel).fill_(0.02)
    streams = make_optimized_fwd_position_streams("cuda:0")
    sequential_scratch = make_optimized_fwd_position_scratch(
        model, sequential, "cuda:0", include_optimized_kernel_scratch=False
    )
    parallel_scratch = make_optimized_fwd_position_scratch(
        model, parallel, "cuda:0", include_optimized_kernel_scratch=False
    )

    assert model.eq_connect_adr.size > 0
    for _ in range(4):
        step_optimized_fwd_position(
            model,
            sequential,
            (),
            scratch=sequential_scratch,
            use_optimized_kernel=False,
        )
        step_optimized_fwd_position(
            model,
            parallel,
            streams,
            scratch=parallel_scratch,
            use_optimized_kernel=False,
        )
    wp.synchronize()

    for field in ("qpos", "qvel", "cvel", "cdof_dot", "qacc"):
        torch.testing.assert_close(
            wp.to_torch(getattr(parallel, field)),
            wp.to_torch(getattr(sequential, field)),
            atol=1.0e-5,
            rtol=1.0e-5,
            msg=lambda msg, field=field: f"{field} differs:\n{msg}",
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_optimized_kernel", [False, True])
def test_deferred_constraint_prefix_keeps_rows_and_matches_full_prefix(
    use_optimized_kernel,
):
    """On each step's endpoint state, a prefix without constraints leaves the
    contacts and rows of the full prefix before it untouched, and a full prefix
    after it rebuilds the same ones (the policy-boundary refresh followed by the
    reset prefix). Contacts and rows are compared per world as sorted sets:
    their order follows atomic slot allocation."""
    nworld = 16
    mj_model, model, data, other = _make_data_pair(nworld)
    _seed_state(mj_model, data, other, nworld)
    # Lower the free root so the feet touch the floor.
    wp.to_torch(data.qpos)[:, 2] -= 0.1
    streams = make_optimized_fwd_position_streams("cuda:0")
    scratch = make_optimized_fwd_position_scratch(
        model, data, "cuda:0", include_optimized_kernel_scratch=use_optimized_kernel
    )

    def prefix(constraints=True):
        forward_state_prefix_optimized_fwd_position(
            model,
            data,
            streams,
            scratch=scratch,
            use_optimized_kernel=use_optimized_kernel,
            constraints=constraints,
        )

    def snapshot():
        wp.synchronize()
        nefc = data.nefc.numpy()
        nacon = int(data.nacon.numpy()[0])
        jac = data.efc.J.numpy()
        pos = data.efc.pos.numpy()
        world = data.contact.worldid.numpy()[:nacon]
        cpos = data.contact.pos.numpy()[:nacon]
        out = {"nacon": np.array([nacon]), "nefc": nefc.copy()}
        for w in range(nworld):
            r = np.concatenate([jac[w, : nefc[w]], pos[w, : nefc[w], None]], axis=1)
            out[f"efc{w}"] = r[np.lexsort(r.T[::-1])]
            c = cpos[world == w]
            out[f"contact{w}"] = c[np.lexsort(c.T[::-1])]
        for field in ("xpos", "cvel", "cdof_dot", "actuator_length", "qfrc_passive"):
            out[field] = getattr(data, field).numpy().copy()
        return out

    def assert_same(actual, expected):
        for name, value in expected.items():
            if use_optimized_kernel:
                # Deterministic tree reductions: bitwise.
                np.testing.assert_array_equal(actual[name], value, err_msg=name)
            else:
                # The stock kernels' atomic sums vary in the last bits.
                np.testing.assert_allclose(
                    actual[name], value, rtol=1.0e-6, atol=1.0e-8, err_msg=name
                )

    prefix()
    contacts = 0
    for _ in range(4):
        step_after_state_prefix_optimized_fwd_position(
            model,
            data,
            streams,
            scratch=scratch,
            use_optimized_kernel=use_optimized_kernel,
        )
        prefix()
        full = snapshot()
        contacts += int(full["nacon"][0])
        prefix(constraints=False)
        assert_same(snapshot(), full)
        prefix()
        assert_same(snapshot(), full)
    assert contacts > 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_optimized_kernel", [False, True])
def test_moved_state_prefix_matches_stock_forward_state_boundary(
    use_optimized_kernel,
):
    nworld = 16
    mj_model, model, stock_data, split_data = _make_data_pair(nworld)
    _seed_state(mj_model, stock_data, split_data, nworld)
    streams = make_optimized_fwd_position_streams("cuda:0")
    scratch = make_optimized_fwd_position_scratch(
        model,
        split_data,
        "cuda:0",
        include_optimized_kernel_scratch=use_optimized_kernel,
    )

    state_fields = (
        "xpos",
        "xquat",
        "xmat",
        "xipos",
        "ximat",
        "geom_xpos",
        "geom_xmat",
        "subtree_com",
        "cinert",
        "cdof",
        "cvel",
        "cdof_dot",
        "ten_length",
        "ten_velocity",
        "actuator_length",
        "actuator_velocity",
    )

    forward_state_prefix_optimized_fwd_position(
        model,
        split_data,
        streams,
        scratch=scratch,
        use_optimized_kernel=use_optimized_kernel,
    )
    for _ in range(4):
        mjw.step(model, stock_data)
        step_after_state_prefix_optimized_fwd_position(
            model,
            split_data,
            streams,
            scratch=scratch,
            use_optimized_kernel=use_optimized_kernel,
        )

        torch.testing.assert_close(
            wp.to_torch(split_data.qpos),
            wp.to_torch(stock_data.qpos),
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        torch.testing.assert_close(
            wp.to_torch(split_data.qvel),
            wp.to_torch(stock_data.qvel),
            atol=1.0e-5,
            rtol=1.0e-5,
        )

        mjw.forward(model, stock_data)
        forward_state_prefix_optimized_fwd_position(
            model,
            split_data,
            streams,
            scratch=scratch,
            use_optimized_kernel=use_optimized_kernel,
        )
        wp.synchronize()

        for field in state_fields:
            torch.testing.assert_close(
                wp.to_torch(getattr(split_data, field)),
                wp.to_torch(getattr(stock_data, field)),
                atol=1.0e-5,
                rtol=1.0e-5,
                msg=lambda msg, field=field: f"{field} differs:\n{msg}",
            )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "integrator",
    [
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        mujoco.mjtIntegrator.mjINT_IMPLICIT,
    ],
)
def test_allocation_free_sequential_step_matches_stock_step(integrator):
    nworld = 16
    mj_model, model, stock_data, custom_data = _make_data_pair(
        nworld,
        integrator=integrator,
    )
    _seed_state(mj_model, stock_data, custom_data, nworld)
    scratch = make_optimized_fwd_position_scratch(
        model,
        custom_data,
        "cuda:0",
        include_optimized_kernel_scratch=False,
    )

    for _ in range(4):
        mjw.step(model, stock_data)
        step_optimized_fwd_position(
            model,
            custom_data,
            (),
            scratch=scratch,
            use_optimized_kernel=False,
        )
    wp.synchronize()

    assert torch.allclose(
        wp.to_torch(stock_data.qpos),
        wp.to_torch(custom_data.qpos),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        wp.to_torch(stock_data.qvel),
        wp.to_torch(custom_data.qvel),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "integrator",
    [
        mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        mujoco.mjtIntegrator.mjINT_IMPLICIT,
    ],
)
def test_nonoptimized_implicit_step_supports_conditional_capture(integrator):
    _, model, _, data = _make_data_pair(nworld=4, integrator=integrator)
    scratch = make_optimized_fwd_position_scratch(
        model,
        data,
        "cuda:0",
        include_optimized_kernel_scratch=False,
    )
    condition = wp.ones((1,), dtype=wp.int32, device="cuda:0")

    def step_body():
        step_optimized_fwd_position(
            model,
            data,
            (),
            scratch=scratch,
            use_optimized_kernel=False,
        )
        wp.launch(
            _clear_capture_condition,
            dim=1,
            inputs=[condition],
            device="cuda:0",
        )

    step_body()
    wp.synchronize()
    condition.fill_(1)

    with wp.ScopedCapture() as capture:
        wp.capture_while(condition, while_body=step_body)

    wp.capture_launch(capture.graph)
    wp.synchronize()


@pytest.mark.cuda
@pytest.mark.parametrize("use_optimized_kernel", [False, True])
def test_312_persistent_rk4_tracks_mujoco_like_stock_step(use_optimized_kernel):
    """RK4 amplifies float32 rounding past bitwise stock parity, so both runs are
    held to the double-precision MuJoCo step. A rowless world takes the smooth
    acceleration directly (MuJoCo's nefc == 0 result) where stock re-solves it
    densely, which rounds differently."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    nworld = 4
    mj_model, model, stock_data, custom_data = _make_data_pair(
        nworld,
        integrator=mujoco.mjtIntegrator.mjINT_RK4,
    )
    model.opt.graph_conditional = False
    _seed_state(mj_model, stock_data, custom_data, nworld)
    initial = {
        field: wp.to_torch(getattr(custom_data, field)).cpu().numpy()
        for field in ("qpos", "qvel", "ctrl")
    }
    scratch = make_optimized_fwd_position_scratch(model, custom_data, "cuda:0")
    assert scratch.rk4 is not None

    for _ in range(2):
        mjw.step(model, stock_data)
        step_optimized_fwd_position(
            model,
            custom_data,
            (),
            scratch=scratch,
            use_optimized_kernel=use_optimized_kernel,
        )
    wp.synchronize()
    nefc = stock_data.nefc.numpy()
    assert (nefc == 0).any() and (nefc > 0).any()

    reference = {"qpos": [], "qvel": []}
    for world in range(nworld):
        host = mujoco.MjData(mj_model)
        for field, value in initial.items():
            getattr(host, field)[:] = value[world]
        for _ in range(2):
            mujoco.mj_step(mj_model, host)
        for field, rows in reference.items():
            rows.append(getattr(host, field).copy())
    for field, rows in reference.items():
        expected = np.stack(rows)
        stock_error = np.abs(getattr(stock_data, field).numpy() - expected).max()
        custom_error = np.abs(getattr(custom_data, field).numpy() - expected).max()
        assert custom_error <= 1.25 * stock_error + 1.0e-5, (
            f"{field}: custom error {custom_error:.3e}, stock error {stock_error:.3e}"
        )


@pytest.mark.cuda
def test_312_persistent_rk4_activation_matches_stock_step():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    nworld = 4
    mj_model = mujoco.MjModel.from_xml_string(_actuator_edge_mjcf())
    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model = mjw.put_model(mj_model)
    model.opt.graph_conditional = False
    stock_data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=nworld)
    custom_data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=nworld)
    for data in (stock_data, custom_data):
        wp.to_torch(data.qpos).fill_(0.3)
        wp.to_torch(data.qvel).fill_(0.1)
        wp.to_torch(data.act).fill_(0.05)
        wp.to_torch(data.ctrl).fill_(0.15)
    scratch = make_optimized_fwd_position_scratch(model, custom_data, "cuda:0")
    assert scratch.rk4 is not None
    assert scratch.rk4.act_t0 is not None
    assert scratch.rk4.act_dot_rk is not None

    for _ in range(2):
        mjw.step(model, stock_data)
        step_optimized_fwd_position(
            model,
            custom_data,
            (),
            scratch=scratch,
            use_optimized_kernel=True,
        )
    wp.synchronize()

    for field in ("qpos", "qvel", "act"):
        torch.testing.assert_close(
            wp.to_torch(getattr(custom_data, field)),
            wp.to_torch(getattr(stock_data, field)),
            atol=3.0e-5,
            rtol=3.0e-5,
            msg=lambda msg, field=field: f"{field} differs:\n{msg}",
        )


@pytest.mark.cuda
def test_312_persistent_rk4_can_be_captured_and_replayed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    nworld = 4
    mj_model = mujoco.MjModel.from_xml_string(_actuator_edge_mjcf())
    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model = mjw.put_model(mj_model)
    stock_data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=nworld)
    data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=nworld)
    model.opt.graph_conditional = False
    for target in (stock_data, data):
        wp.to_torch(target.qpos).fill_(0.3)
        wp.to_torch(target.qvel).fill_(0.1)
        wp.to_torch(target.act).fill_(0.05)
        wp.to_torch(target.ctrl).fill_(0.15)
    streams = make_optimized_fwd_position_streams("cuda:0")
    scratch = make_optimized_fwd_position_scratch(model, data, "cuda:0")
    torch_stream, wp_stream = make_stream_aligned_cuda_graph_streams(
        torch.device("cuda")
    )

    step_optimized_fwd_position(
        model,
        data,
        streams,
        scratch=scratch,
        use_optimized_kernel=True,
    )
    wp.synchronize()
    with _stream_aligned_cuda_graph_capture(
        torch.device("cuda"),
        torch_stream,
        wp_stream,
    ) as capture:
        step_optimized_fwd_position(
            model,
            data,
            streams,
            scratch=scratch,
            use_optimized_kernel=True,
        )
    _launch_stream_aligned_cuda_graph(
        capture.graph,
        torch.device("cuda"),
        torch_stream,
        wp_stream,
    )
    for _ in range(2):
        mjw.step(model, stock_data)
    wp.synchronize()

    for field in ("qpos", "qvel", "act"):
        torch.testing.assert_close(
            wp.to_torch(getattr(data, field)),
            wp.to_torch(getattr(stock_data, field)),
            atol=3.0e-5,
            rtol=3.0e-5,
            msg=lambda msg, field=field: f"captured {field} differs:\n{msg}",
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("parallelized", [False, True])
def test_step_optimized_kernel_matches_stock_step(parallelized):
    nworld = 16
    mj_model, model, stock_data, parallel_data = _make_data_pair(nworld)
    _seed_state(mj_model, stock_data, parallel_data, nworld)
    streams = make_optimized_fwd_position_streams("cuda:0") if parallelized else ()
    scratch = make_optimized_fwd_position_scratch(model, parallel_data, "cuda:0")

    for _ in range(4):
        mjw.step(model, stock_data)
        step_optimized_fwd_position(
            model,
            parallel_data,
            streams,
            scratch=scratch,
            use_optimized_kernel=True,
        )
    wp.synchronize()

    assert torch.allclose(
        wp.to_torch(stock_data.qpos),
        wp.to_torch(parallel_data.qpos),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        wp.to_torch(stock_data.qvel),
        wp.to_torch(parallel_data.qvel),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("parallelized", [False, True])
def test_sleep_compact_solver_matches_stock_step(parallelized):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    nworld = 4
    mj_model, model, stock_data, custom_data = _make_data_pair(
        nworld,
        enable_sleep=True,
    )
    _seed_state(mj_model, stock_data, custom_data, nworld)
    streams = make_optimized_fwd_position_streams("cuda:0") if parallelized else ()
    scratch = make_optimized_fwd_position_scratch(model, custom_data, "cuda:0")

    assert scratch.solver_uses_compact
    assert scratch.solver_compact_model is not None
    assert scratch.solver_compact_data is not None
    compact_model = scratch.solver_compact_model
    compact_data = scratch.solver_compact_data
    compact_context = scratch.solver_context

    for _ in range(3):
        mjw.step(model, stock_data)
        step_optimized_fwd_position(
            model,
            custom_data,
            streams,
            scratch=scratch,
            use_optimized_kernel=True,
        )
    wp.synchronize()

    assert scratch.solver_compact_model is compact_model
    assert scratch.solver_compact_data is compact_data
    assert scratch.solver_context is compact_context
    torch.testing.assert_close(
        wp.to_torch(custom_data.qpos),
        wp.to_torch(stock_data.qpos),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        wp.to_torch(custom_data.qvel),
        wp.to_torch(stock_data.qvel),
        atol=2.0e-5,
        rtol=2.0e-5,
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "solver_type",
    [mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtSolver.mjSOL_CG],
)
def test_312_persistent_solver_outputs_match_stock_step(solver_type):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    nworld = 4
    mj_model, model, stock_data, custom_data = _make_data_pair(
        nworld,
        solver_type=solver_type,
    )
    _seed_state(mj_model, stock_data, custom_data, nworld)
    scratch = make_optimized_fwd_position_scratch(
        model,
        custom_data,
        "cuda:0",
        include_optimized_kernel_scratch=False,
    )

    for _ in range(3):
        mjw.step(model, stock_data)
        step_optimized_fwd_position(
            model,
            custom_data,
            (),
            scratch=scratch,
            use_optimized_kernel=False,
        )
    wp.synchronize()

    for field in ("qacc", "qacc_smooth", "qfrc_constraint"):
        torch.testing.assert_close(
            wp.to_torch(getattr(custom_data, field)),
            wp.to_torch(getattr(stock_data, field)),
            atol=2.0e-5,
            rtol=2.0e-5,
            msg=lambda msg, field=field: f"{field} differs:\n{msg}",
        )
    torch.testing.assert_close(
        wp.to_torch(custom_data.efc.force),
        wp.to_torch(stock_data.efc.force),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.equal(
        wp.to_torch(custom_data.solver_niter),
        wp.to_torch(stock_data.solver_niter),
    )


@pytest.mark.cuda
def test_312_optimized_implicitfast_cg_step_remains_finite_and_matches_stock():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    nworld = 4
    mj_model, model, stock_data, custom_data = _make_data_pair(
        nworld,
        solver_type=mujoco.mjtSolver.mjSOL_CG,
    )
    _seed_state(mj_model, stock_data, custom_data, nworld)
    scratch = make_optimized_fwd_position_scratch(model, custom_data, "cuda:0")

    for _ in range(3):
        mjw.step(model, stock_data)
        step_optimized_fwd_position(
            model,
            custom_data,
            (),
            scratch=scratch,
            use_optimized_kernel=True,
        )
    wp.synchronize()

    for field in ("qpos", "qvel", "qacc", "qacc_smooth", "qfrc_constraint"):
        custom = wp.to_torch(getattr(custom_data, field))
        stock = wp.to_torch(getattr(stock_data, field))
        assert torch.isfinite(custom).all(), f"{field} contains non-finite values"
        atol = 2.0e-5 if field in ("qpos", "qvel") else 1.0e-3
        rtol = 2.0e-5 if field in ("qpos", "qvel") else 1.0e-3
        torch.testing.assert_close(
            custom,
            stock,
            atol=atol,
            rtol=rtol,
            msg=lambda msg, field=field: f"{field} differs:\n{msg}",
        )


@pytest.mark.cuda
def test_sleep_compact_solver_matches_stock_with_inactive_trees():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from mujoco_warp._src import forward as mjw_forward
    from mujoco_warp._src import solver as mjw_solver

    model, stock_data, custom_data = _make_two_tree_sleep_data_pair()
    scratch = make_optimized_fwd_position_scratch(model, custom_data, "cuda:0")
    tree_awake = torch.tensor([[1, 0], [0, 1]], device="cuda", dtype=torch.int32)
    for data in (stock_data, custom_data):
        wp.to_torch(data.tree_awake).copy_(tree_awake)

    mjw_forward.fwd_acceleration(model, stock_data, factorize=True)
    mjw_solver.solve(model, stock_data)
    import instantmimic.utils.mjw_step_optimized as optimized

    optimized.fwd_acceleration_optimized(model, custom_data, factorize=True)
    optimized.solve_with_scratch(model, custom_data, scratch)
    wp.synchronize()

    assert torch.equal(
        wp.to_torch(custom_data.ncdof),
        torch.ones(2, device="cuda", dtype=torch.int32),
    )
    assert torch.equal(
        wp.to_torch(custom_data.dof_cdof),
        wp.to_torch(stock_data.dof_cdof),
    )
    for field in ("qacc_smooth", "qacc", "qfrc_constraint"):
        torch.testing.assert_close(
            wp.to_torch(getattr(custom_data, field)),
            wp.to_torch(getattr(stock_data, field)),
            atol=1.0e-6,
            rtol=1.0e-6,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_optimized_kernel", [False, True])
def test_step_parallel_paths_match_stock_step_with_sleep_enabled(
    use_optimized_kernel,
):
    nworld = 16
    mj_model, model, stock_data, parallel_data = _make_data_pair(
        nworld,
        enable_sleep=True,
    )
    _seed_state(mj_model, stock_data, parallel_data, nworld)
    streams = make_optimized_fwd_position_streams("cuda:0")
    scratch = (
        make_optimized_fwd_position_scratch(model, parallel_data, "cuda:0")
        if use_optimized_kernel
        else None
    )

    for _ in range(4):
        mjw.step(model, stock_data)
        step_optimized_fwd_position(
            model,
            parallel_data,
            streams,
            scratch=scratch,
            use_optimized_kernel=use_optimized_kernel,
        )
    wp.synchronize()

    assert torch.allclose(
        wp.to_torch(stock_data.qpos),
        wp.to_torch(parallel_data.qpos),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        wp.to_torch(stock_data.qvel),
        wp.to_torch(parallel_data.qvel),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_step_skip_kinematics_optimized_fwd_position_matches_stock_step():
    nworld = 16
    mj_model, model, stock_data, parallel_data = _make_data_pair(nworld)
    _seed_state(mj_model, stock_data, parallel_data, nworld)
    streams = make_optimized_fwd_position_streams("cuda:0")

    mjw.kinematics(model, parallel_data)
    mjw.com_pos(model, parallel_data)
    mjw.com_vel(model, parallel_data)

    mjw.step(model, stock_data)
    step_skip_kinematics_optimized_fwd_position(model, parallel_data, streams)
    wp.synchronize()

    assert torch.allclose(
        wp.to_torch(stock_data.qpos),
        wp.to_torch(parallel_data.qpos),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        wp.to_torch(stock_data.qvel),
        wp.to_torch(parallel_data.qvel),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_step_optimized_fwd_position_can_be_captured_and_replayed():
    nworld = 16
    mj_model, model, _, parallel_data = _make_data_pair(nworld)
    _seed_state(mj_model, parallel_data, parallel_data, nworld)
    streams = make_optimized_fwd_position_streams("cuda:0")
    torch_stream, wp_stream = make_stream_aligned_cuda_graph_streams(
        torch.device("cuda")
    )

    with _stream_aligned_cuda_graph_capture(
        torch.device("cuda"),
        torch_stream,
        wp_stream,
    ) as capture:
        step_optimized_fwd_position(model, parallel_data, streams)
    _launch_stream_aligned_cuda_graph(
        capture.graph,
        torch.device("cuda"),
        torch_stream,
        wp_stream,
    )
    wp.synchronize()

    assert torch.isfinite(wp.to_torch(parallel_data.qpos)).all()
    assert torch.isfinite(wp.to_torch(parallel_data.qvel)).all()


@pytest.mark.cuda
def test_persistent_solver_scratch_can_be_captured_and_replayed():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    nworld = 4
    mj_model, model, _, data = _make_data_pair(nworld)
    _seed_state(mj_model, data, data, nworld)
    streams = make_optimized_fwd_position_streams("cuda:0")
    scratch = make_optimized_fwd_position_scratch(model, data, "cuda:0")
    torch_stream, wp_stream = make_stream_aligned_cuda_graph_streams(
        torch.device("cuda")
    )

    step_optimized_fwd_position(
        model,
        data,
        streams,
        scratch=scratch,
        use_optimized_kernel=True,
    )
    wp.synchronize()

    with _stream_aligned_cuda_graph_capture(
        torch.device("cuda"),
        torch_stream,
        wp_stream,
    ) as capture:
        step_optimized_fwd_position(
            model,
            data,
            streams,
            scratch=scratch,
            use_optimized_kernel=True,
        )
    _launch_stream_aligned_cuda_graph(
        capture.graph,
        torch.device("cuda"),
        torch_stream,
        wp_stream,
    )
    wp.synchronize()

    assert torch.isfinite(wp.to_torch(data.qpos)).all()
    assert torch.isfinite(wp.to_torch(data.qvel)).all()


@pytest.mark.cuda
@pytest.mark.parametrize("disable_island", [False, True])
def test_sleep_step_scratch_can_be_captured_and_replayed(disable_island):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    model, stock_data, data = _make_two_tree_sleep_data_pair()
    if disable_island:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ISLAND)
    model.opt.graph_conditional = False
    streams = make_optimized_fwd_position_streams("cuda:0")
    scratch = make_optimized_fwd_position_scratch(model, data, "cuda:0")
    assert scratch.island_mapping_call is not None
    assert (scratch.island_call is None) is disable_island
    torch_stream, wp_stream = make_stream_aligned_cuda_graph_streams(
        torch.device("cuda")
    )

    step_optimized_fwd_position(
        model,
        data,
        streams,
        scratch=scratch,
        use_optimized_kernel=True,
    )
    wp.synchronize()

    with _stream_aligned_cuda_graph_capture(
        torch.device("cuda"),
        torch_stream,
        wp_stream,
    ) as capture:
        step_optimized_fwd_position(
            model,
            data,
            streams,
            scratch=scratch,
            use_optimized_kernel=True,
        )
    _launch_stream_aligned_cuda_graph(
        capture.graph,
        torch.device("cuda"),
        torch_stream,
        wp_stream,
    )
    for _ in range(2):
        mjw.step(model, stock_data)
    wp.synchronize()

    for field in ("qpos", "qvel", "tree_island", "dof_island"):
        torch.testing.assert_close(
            wp.to_torch(getattr(data, field)),
            wp.to_torch(getattr(stock_data, field)),
            atol=1.0e-6,
            rtol=1.0e-6,
            msg=lambda msg, field=field: f"captured {field} differs:\n{msg}",
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_actuator_centric_dense_deriv_smooth_vel_matches_stock():
    nworld = 16
    mj_model, model, stock_data, parallel_data = _make_data_pair(nworld)
    _seed_state(mj_model, stock_data, parallel_data, nworld)
    scratch = make_optimized_fwd_position_scratch(model, parallel_data, "cuda:0")

    mjw.forward(model, stock_data)
    mjw.forward(model, parallel_data)

    stock_matrix = stock_data.M if hasattr(stock_data, "M") else stock_data.qM
    parallel_matrix = (
        parallel_data.M if hasattr(parallel_data, "M") else parallel_data.qM
    )
    stock_qderiv = wp.empty(stock_matrix.shape, dtype=float, device="cuda:0")
    parallel_qderiv = wp.empty(parallel_matrix.shape, dtype=float, device="cuda:0")
    derivative.deriv_smooth_vel(model, stock_data, stock_qderiv)
    deriv_smooth_vel_dense_actuator_centric(
        model,
        parallel_data,
        parallel_qderiv,
        scratch,
    )
    wp.synchronize()

    assert torch.allclose(
        wp.to_torch(stock_qderiv),
        wp.to_torch(parallel_qderiv),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


@pytest.mark.cuda
def test_312_actearly_activation_and_force_clamp_match_stock():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    mj_model = mujoco.MjModel.from_xml_string(_actuator_edge_mjcf())
    model = mjw.put_model(mj_model)
    stock_data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=4)
    custom_data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=4)
    for data in (stock_data, custom_data):
        wp.to_torch(data.qpos).fill_(0.4)
        wp.to_torch(data.qvel).fill_(0.3)
        wp.to_torch(data.act).fill_(0.19)
        wp.to_torch(data.ctrl).fill_(10.0)
    scratch = make_optimized_fwd_position_scratch(model, custom_data, "cuda:0")

    mjw.forward(model, stock_data)
    from instantmimic.utils.mjw_step_optimized import forward_optimized_fwd_position

    forward_optimized_fwd_position(
        model,
        custom_data,
        (),
        scratch=scratch,
        use_optimized_kernel=True,
    )
    stock_qderiv = wp.empty(stock_data.M.shape, dtype=float, device="cuda:0")
    custom_qderiv = wp.empty(custom_data.M.shape, dtype=float, device="cuda:0")
    derivative.deriv_smooth_vel(model, stock_data, stock_qderiv)
    deriv_smooth_vel_dense_actuator_centric(
        model,
        custom_data,
        custom_qderiv,
        scratch,
    )
    wp.synchronize()

    for field in ("act_dot", "actuator_force", "qfrc_actuator", "qfrc_smooth"):
        torch.testing.assert_close(
            wp.to_torch(getattr(custom_data, field)),
            wp.to_torch(getattr(stock_data, field)),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
    torch.testing.assert_close(
        wp.to_torch(custom_qderiv),
        wp.to_torch(stock_qderiv),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        wp.to_torch(custom_data.actuator_force).abs(),
        torch.full((4, 1), 0.05, device="cuda"),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


@pytest.mark.cuda
def test_312_delayed_control_uses_preallocated_stock_actuation_path():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    mj_model = mujoco.MjModel.from_xml_string(_delayed_actuator_mjcf())
    model = mjw.put_model(mj_model)
    stock_data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=4)
    custom_data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=4)
    for data in (stock_data, custom_data):
        wp.to_torch(data.qpos).fill_(0.2)
        wp.to_torch(data.qvel).fill_(0.1)
    scratch = make_optimized_fwd_position_scratch(model, custom_data, "cuda:0")

    for ctrl in (0.2, -0.4, 0.7, -0.1):
        wp.to_torch(stock_data.ctrl).fill_(ctrl)
        wp.to_torch(custom_data.ctrl).fill_(ctrl)
        mjw.step(model, stock_data)
        step_optimized_fwd_position(
            model,
            custom_data,
            (),
            scratch=scratch,
            use_optimized_kernel=True,
        )
    wp.synchronize()

    for field in (
        "qpos",
        "qvel",
        "act_dot",
        "actuator_force",
        "qfrc_actuator",
        "history",
    ):
        torch.testing.assert_close(
            wp.to_torch(getattr(custom_data, field)),
            wp.to_torch(getattr(stock_data, field)),
            atol=1.0e-5,
            rtol=1.0e-5,
            msg=lambda msg, field=field: f"{field} differs:\n{msg}",
        )


@pytest.mark.cuda
@pytest.mark.parametrize("gravity_disabled", [False, True])
def test_312_optimized_actuator_gravcomp_matches_stock(gravity_disabled):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    mj_model = mujoco.MjModel.from_xml_string(_actuator_edge_mjcf())
    if gravity_disabled:
        mj_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    model = mjw.put_model(mj_model)
    stock_data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=4)
    custom_data = mjw.put_data(mj_model, mujoco.MjData(mj_model), nworld=4)
    for data in (stock_data, custom_data):
        wp.to_torch(data.qpos).fill_(0.7)
        wp.to_torch(data.qvel).zero_()
        wp.to_torch(data.act).zero_()
        wp.to_torch(data.ctrl).zero_()
    scratch = make_optimized_fwd_position_scratch(model, custom_data, "cuda:0")

    mjw.forward(model, stock_data)
    from instantmimic.utils.mjw_step_optimized import forward_optimized_fwd_position

    forward_optimized_fwd_position(
        model,
        custom_data,
        (),
        scratch=scratch,
        use_optimized_kernel=True,
    )
    wp.synchronize()

    for field in ("qfrc_gravcomp", "qfrc_actuator", "qfrc_smooth"):
        torch.testing.assert_close(
            wp.to_torch(getattr(custom_data, field)),
            wp.to_torch(getattr(stock_data, field)),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
    if gravity_disabled:
        torch.testing.assert_close(
            wp.to_torch(custom_data.qfrc_actuator),
            wp.to_torch(custom_data.actuator_force),
            atol=1.0e-6,
            rtol=1.0e-6,
        )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
@pytest.mark.parametrize("fluidshape", ["none", "ellipsoid"])
def test_articulated_fluid_actuator_derivative_matches_native_and_reuses_scratch(
    monkeypatch, fluidshape, device
):
    xml = f"""
    <mujoco><option integrator="implicitfast" jacobian="dense"
      density="1.2" viscosity=".02" wind=".2 .1 0"/>
      <worldbody><body><joint name="j" type="hinge"/>
        <geom type="ellipsoid" size=".1 .2 .3" mass="1" fluidshape="{fluidshape}"/>
        <body pos="0 0 .4"><joint type="ball"/>
          <geom type="box" size=".1 .15 .2" mass=".5"/>
        </body></body></worldbody>
      <actuator><velocity joint="j" kv=".3"/></actuator>
    </mujoco>"""
    if device == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    host = mujoco.MjModel.from_xml_string(xml)
    with wp.ScopedDevice(device):
        m = mjw.put_model(host)
        d = mjw.put_data(host, mujoco.MjData(host), nworld=2, njmax=8, nconmax=4)
        assert m.has_fluid and m.body_freeadr.size == 0
        d.qvel.fill_(0.4)
        mjw.forward(m, d)
        scratch = make_optimized_fwd_position_scratch(m, d, device)
        expected = wp.empty_like(d.M)
        actual = wp.empty_like(d.M)
        derivative.deriv_smooth_vel(m, d, expected)
        deriv_smooth_vel_dense_actuator_centric(m, d, actual, scratch)
        with monkeypatch.context() as patch:
            for name in ("empty", "empty_like", "zeros", "clone", "full"):
                patch.setattr(
                    wp, name, lambda *a, **kw: pytest.fail("fluid derivative allocated")
                )
            deriv_smooth_vel_dense_actuator_centric(m, d, actual, scratch)
        if device == "cuda":
            with wp.ScopedCapture() as capture:
                deriv_smooth_vel_dense_actuator_centric(m, d, actual, scratch)
            wp.capture_launch(capture.graph)
            wp.capture_launch(capture.graph)
        np.testing.assert_allclose(
            actual.numpy(), expected.numpy(), atol=1e-6, rtol=1e-6
        )


@pytest.mark.cuda
@pytest.mark.parametrize("sleep_enabled", [False, True])
@pytest.mark.parametrize(
    "solver_type", [mujoco.mjtSolver.mjSOL_NEWTON, mujoco.mjtSolver.mjSOL_CG]
)
@pytest.mark.parametrize("jacobian", ["dense", "sparse"])
@pytest.mark.parametrize("iterations", [0, 4])
def test_flattened_solver_matches_conditional_and_replays_inside_outer_loop(
    solver_type, jacobian, iterations, sleep_enabled
):
    if not wp.is_cuda_available():
        pytest.skip("CUDA required")
    host = mujoco.MjModel.from_xml_string(f"""
      <mujoco><option integrator="implicitfast" jacobian="{jacobian}" iterations="{iterations}"/>
        <worldbody><geom type="plane" size="1 1 .1"/>
          <body pos="0 0 .09"><freejoint/><geom size=".1" mass="1"/></body>
        </worldbody></mujoco>""")
    host.opt.solver = solver_type
    if sleep_enabled:
        host.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
    if sleep_enabled and solver_type == mujoco.mjtSolver.mjSOL_CG:
        with pytest.raises(ValueError, match="sleeping requires the Newton solver"):
            mjw.put_model(host)
        return
    with wp.ScopedDevice("cuda"):
        m = mjw.put_model(host)
        m.opt.graph_conditional = True
        d = mjw.put_data(host, mujoco.MjData(host), nworld=2, njmax=16, nconmax=4)
        # Mixed contact/rowless worlds exercise independent convergence guards.
        qpos = d.qpos.numpy()
        qpos[1, 2] = 1.0
        d.qpos.assign(qpos)
        mjw.forward(m, d)
        from instantmimic.utils import mjw_step_optimized as runtime

        scratch = make_optimized_fwd_position_scratch(
            m, d, "cuda", include_optimized_kernel_scratch=False
        )
        arrays = [
            d.qacc,
            d.qfrc_constraint,
            d.efc.Ma,
            d.efc.force,
            d.efc.state,
            d.solver_niter,
            d.overflow,
            scratch.solver_nsolving,
        ]
        arrays += [
            v
            for v in vars(scratch.solver_context).values()
            if isinstance(v, wp.array) and v.size
        ]
        saved = [(a, wp.clone(a)) for a in {a.ptr: a for a in arrays}.values()]

        def restore():
            for a, value in saved:
                wp.copy(a, value)

        runtime.solve_with_scratch(m, d, scratch)
        fields = [
            d.qacc,
            d.qfrc_constraint,
            d.efc.force,
            d.efc.state,
            d.solver_niter,
            d.overflow,
        ]
        expected = [a.numpy().copy() for a in fields]
        restore()
        scratch.unroll_solver_iterations = True
        condition = wp.ones(1, dtype=int)

        def body():
            runtime.solve_with_scratch(m, d, scratch)
            wp.launch(_clear_capture_condition, dim=1, inputs=[condition])

        body()
        restore()
        condition.fill_(1)
        with wp.ScopedCapture() as capture:
            wp.capture_while(condition, while_body=body)
        scratch.unroll_solver_iterations = False
        for _ in range(2):
            restore()
            condition.fill_(1)
            wp.capture_launch(capture.graph)
            for index, (actual, reference) in enumerate(
                zip(fields, expected, strict=True)
            ):
                if index < 3:
                    np.testing.assert_allclose(
                        actual.numpy(), reference, atol=1e-5, rtol=1e-5
                    )
                else:
                    np.testing.assert_array_equal(actual.numpy(), reference)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_nonempty_tactile_forward_reuses_scratch_and_graph(monkeypatch, device):
    if device == "cuda" and not wp.is_cuda_available():
        pytest.skip("CUDA required")
    host = mujoco.MjModel.from_xml_string("""
      <mujoco><option iterations="4"><flag multiccd="enable"/></option>
        <asset><mesh name="surface" builtin="sphere" params="0"/></asset>
        <worldbody>
          <body pos="0 0 1"><freejoint/><geom name="sensor" type="mesh" mesh="surface"/></body>
          <geom type="box" size=".7 .7 .3"/>
        </worldbody>
        <sensor><tactile geom="sensor" mesh="surface"/></sensor>
      </mujoco>""")
    with wp.ScopedDevice(device):
        m = mjw.put_model(host)
        stock = mjw.put_data(host, mujoco.MjData(host), nworld=2, njmax=128, nconmax=16)
        d = mjw.put_data(host, mujoco.MjData(host), nworld=2, njmax=128, nconmax=16)
        scratch = make_optimized_fwd_position_scratch(m, d, device)
        mjw.forward(m, stock)

        def forward():
            forward_optimized_fwd_position(m, d, (), scratch, use_optimized_kernel=True)

        forward()
        with monkeypatch.context() as patch:
            for name in ("empty", "empty_like", "zeros", "clone", "full"):
                patch.setattr(
                    wp, name, lambda *a, **kw: pytest.fail("tactile forward allocated")
                )
            forward()
            if device == "cuda":
                with wp.ScopedCapture() as capture:
                    forward()
                wp.capture_launch(capture.graph)
                wp.capture_launch(capture.graph)
        assert np.any(stock.sensordata.numpy() != 0)
        np.testing.assert_allclose(
            d.sensordata.numpy(), stock.sensordata.numpy(), atol=1e-5
        )
