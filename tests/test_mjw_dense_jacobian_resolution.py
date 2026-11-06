"""Dense Jacobian resolution for the fused Newton path (CPU)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

import instantmimic.envs.direct_rl_env as direct_rl_env
from instantmimic.envs.direct_rl_env import DirectRLEnv
from instantmimic.envs.direct_rl_env_cfg import (
    DirectRLEnvCfg,
    InteractiveSceneCfg,
    SimulationCfg,
    ViewerCfg,
)
from instantmimic.utils.mjw_kernels.private import io as mjw_io
from instantmimic.utils.mjw_model import (
    DENSE_NV_MAX,
    promote_dense_jacobian_for_fused_newton,
    put_model,
)

_AUTO = mujoco.mjtJacobian.mjJAC_AUTO
_DENSE = mujoco.mjtJacobian.mjJAC_DENSE
_SPARSE = mujoco.mjtJacobian.mjJAC_SPARSE


def _jacobian_name(value: str | mujoco.mjtJacobian) -> str:
    if isinstance(value, str):
        return value
    return mujoco.mjtJacobian(int(value)).name.removeprefix("mjJAC_").lower()


def _spec(
    n_free: int, n_hinge: int, *, jacobian: str | mujoco.mjtJacobian = "auto"
) -> mujoco.MjSpec:
    """Synthetic Newton model whose DOF count is 6 * (n_free + 1) + n_hinge."""

    bodies = "".join(
        f'<body pos="{0.4 * i} 0 0.2"><freejoint/>'
        '<geom type="box" size="0.05 0.04 0.03"/></body>'
        for i in range(n_free)
    )
    chain = "".join(
        '<body pos="0.1 0 0"><joint type="hinge" axis="0 1 0"/>'
        '<geom type="capsule" size="0.02 0.04"/></body>'
        for _ in range(n_hinge)
    )
    return mujoco.MjSpec.from_string(
        f"""<mujoco>
          <option jacobian="{_jacobian_name(jacobian)}" solver="Newton" cone="pyramidal"
                  integrator="Euler" iterations="100"/>
          <worldbody>
            <geom type="plane" size="5 5 .1"/>
            {bodies}
            <body pos="-1 0 0.5"><freejoint/>
              <geom type="box" size="0.05 0.04 0.03"/>{chain}
            </body>
          </worldbody>
        </mujoco>"""
    )


def _model(
    *,
    n_free: int = 4,
    n_hinge: int = 5,
    jacobian: str | mujoco.mjtJacobian = "auto",
) -> mujoco.MjModel:
    """Compile the synthetic model with the Jacobian written into the MJCF."""

    return mujoco.MjSpec.compile(_spec(n_free, n_hinge, jacobian=jacobian))


def _rollout(model: mujoco.MjModel, steps: int = 40) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


def test_auto_jacobian_is_promoted_when_the_dense_fused_newton_path_applies():
    mjm = _model()  # nv = 35, above the 32-DOF upstream sparse threshold
    assert mjm.nv == 35
    assert mjm.opt.jacobian == _AUTO and mjw_io.is_sparse(mjm)

    assert promote_dense_jacobian_for_fused_newton(
        mjm, use_optimized_kernel=True, authored_is_sparse=None
    )
    assert mjm.opt.jacobian == _DENSE
    with wp.ScopedDevice("cpu"):
        uploaded = put_model(mjm)
    assert not uploaded.is_sparse and uploaded.nv_pad == 48


def test_optimized_kernels_off_keep_the_upstream_sparse_resolution():
    mjm = _model()
    assert not promote_dense_jacobian_for_fused_newton(
        mjm, use_optimized_kernel=False, authored_is_sparse=None
    )
    assert mjm.opt.jacobian == _AUTO
    with wp.ScopedDevice("cpu"):
        assert put_model(mjm).is_sparse


def test_promotion_matches_a_model_compiled_with_a_dense_jacobian():
    """Setting `opt.jacobian` after compile is the same model as compiling dense."""

    promoted = _model()
    compiled_dense = _model(jacobian=_DENSE)
    assert promote_dense_jacobian_for_fused_newton(
        promoted, use_optimized_kernel=True, authored_is_sparse=None
    )
    assert promoted.opt.jacobian == compiled_dense.opt.jacobian == _DENSE
    promoted_state = _rollout(promoted)
    dense_state = _rollout(compiled_dense)
    np.testing.assert_array_equal(promoted_state[0], dense_state[0])
    np.testing.assert_array_equal(promoted_state[1], dense_state[1])


@pytest.mark.parametrize(("authored", "jacobian"), [(True, _SPARSE), (False, _DENSE)])
def test_authored_is_sparse_wins(authored, jacobian):
    mjm = _model(jacobian=jacobian)
    assert not promote_dense_jacobian_for_fused_newton(
        mjm, use_optimized_kernel=True, authored_is_sparse=authored
    )
    assert mjm.opt.jacobian == jacobian
    with wp.ScopedDevice("cpu"):
        assert put_model(mjm).is_sparse is authored


def test_auto_jacobian_below_the_sparse_threshold_is_already_dense():
    mjm = _model(n_free=4, n_hinge=0)  # nv = 30
    assert mjm.nv == 30 and mjm.opt.jacobian == _AUTO
    assert not mjw_io.is_sparse(mjm)
    assert not promote_dense_jacobian_for_fused_newton(
        mjm, use_optimized_kernel=True, authored_is_sparse=None
    )
    assert mjm.opt.jacobian == _AUTO
    with wp.ScopedDevice("cpu"):
        assert not put_model(mjm).is_sparse


def test_implicitfast_integrator_is_promoted():
    """The G1 side_kick config solves `implicitfast`; only DISCRETE is excluded."""

    mjm = _model()
    mjm.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert promote_dense_jacobian_for_fused_newton(
        mjm, use_optimized_kernel=True, authored_is_sparse=None
    )
    assert mjm.opt.jacobian == _DENSE


def test_models_above_the_dense_cap_keep_the_upstream_sparse_resolution():
    mjm = _model(n_free=12, n_hinge=8)
    assert mjm.nv > DENSE_NV_MAX
    assert not promote_dense_jacobian_for_fused_newton(
        mjm, use_optimized_kernel=True, authored_is_sparse=None
    )
    assert mjm.opt.jacobian == _AUTO


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda opt: setattr(opt, "solver", mujoco.mjtSolver.mjSOL_CG),
            id="cg-solver",
        ),
        pytest.param(
            lambda opt: setattr(opt, "cone", mujoco.mjtCone.mjCONE_ELLIPTIC),
            id="elliptic-cone",
        ),
        pytest.param(
            lambda opt: setattr(opt, "integrator", mujoco.mjtIntegrator.mjINT_DISCRETE),
            id="discrete-integrator",
        ),
        pytest.param(lambda opt: setattr(opt, "iterations", 0), id="no-iterations"),
        pytest.param(
            lambda opt: setattr(
                opt,
                "enableflags",
                opt.enableflags | mujoco.mjtEnableBit.mjENBL_SLEEP,
            ),
            id="sleeping-compact-solve",
        ),
    ],
)
def test_models_outside_the_fused_newton_structure_keep_the_auto_jacobian(mutate):
    mjm = _model()
    mutate(mjm.opt)
    assert not promote_dense_jacobian_for_fused_newton(
        mjm, use_optimized_kernel=True, authored_is_sparse=None
    )
    assert mjm.opt.jacobian == _AUTO


@pytest.mark.parametrize(
    ("xml_jacobian", "jacobian", "upstream_sparse"),
    [("dense", _DENSE, False), ("sparse", _SPARSE, True)],
)
def test_xml_authored_jacobian_is_never_rewritten(
    xml_jacobian, jacobian, upstream_sparse
):
    """An explicit MJCF Jacobian wins even with `env.sim.is_sparse` unset."""

    mjm = _model(jacobian=xml_jacobian)
    assert mjm.nv == 35 and mjm.opt.jacobian == jacobian
    assert mjw_io.is_sparse(mjm) is upstream_sparse
    assert not promote_dense_jacobian_for_fused_newton(
        mjm, use_optimized_kernel=True, authored_is_sparse=None
    )
    assert mjm.opt.jacobian == jacobian
    with wp.ScopedDevice("cpu"):
        assert put_model(mjm).is_sparse is upstream_sparse


@pytest.mark.parametrize(
    ("n_free", "n_hinge", "nv", "promotes", "nv_pad"),
    [
        pytest.param(4, 2, 32, False, None, id="nv-32-dense"),
        pytest.param(4, 3, 33, True, 48, id="nv-33-promotes"),
        pytest.param(9, 1, 61, True, 64, id="nv-61-above-upstream-cap"),
        pytest.param(11, 7, 79, True, 80, id="nv-79-at-dense-cap"),
        pytest.param(12, 2, 80, False, None, id="nv-80-above-dense-cap"),
    ],
)
def test_sparse_threshold_and_dense_cap_boundaries(
    n_free, n_hinge, nv, promotes, nv_pad
):
    mjm = _model(n_free=n_free, n_hinge=n_hinge)
    assert mjm.nv == nv
    assert mjw_io.is_sparse(mjm) is (nv > 32)

    assert (
        promote_dense_jacobian_for_fused_newton(
            mjm, use_optimized_kernel=True, authored_is_sparse=None
        )
        is promotes
    )
    assert mjm.opt.jacobian == (_DENSE if promotes else _AUTO)
    with wp.ScopedDevice("cpu"):
        uploaded = put_model(mjm)
    if promotes:
        assert not uploaded.is_sparse and uploaded.nv_pad == nv_pad
    else:
        assert uploaded.is_sparse is (nv > 32)


def _stub_direct_rl_env(
    *,
    model: mujoco.MjModel,
    resolved_optimized: bool,
    authored_optimized: bool = False,
    authored_is_sparse: bool | None = None,
) -> DirectRLEnv:
    env = DirectRLEnv.__new__(DirectRLEnv)
    env.model = model
    env.cfg = SimpleNamespace(
        use_optimized_kernel=authored_optimized,
        sim=SimpleNamespace(is_sparse=authored_is_sparse, warn_overflow=False),
    )
    env._use_optimized_kernel = resolved_optimized
    return env


def test_upload_resolves_before_put_model_with_the_resolved_policy(monkeypatch):
    """The upload path must pass the resolved policy, not the authored flag."""

    events: list[tuple[object, ...]] = []
    real_promote = direct_rl_env.promote_dense_jacobian_for_fused_newton
    real_put_model = direct_rl_env.put_model

    def promote(mjm, **kwargs):
        changed = real_promote(mjm, **kwargs)
        events.append(("promote", changed, dict(kwargs), int(mjm.opt.jacobian)))
        return changed

    def upload(mjm, *args, **kwargs):
        events.append(("put_model", int(mjm.opt.jacobian)))
        uploaded = real_put_model(mjm, *args, **kwargs)
        events.append(("uploaded", bool(uploaded.is_sparse)))
        return uploaded

    monkeypatch.setattr(
        direct_rl_env, "promote_dense_jacobian_for_fused_newton", promote
    )
    monkeypatch.setattr(direct_rl_env, "put_model", upload)
    env = _stub_direct_rl_env(
        model=_model(), resolved_optimized=True, authored_optimized=False
    )

    env._upload_mjw_model()

    assert [event[0] for event in events] == ["promote", "put_model", "uploaded"]
    _, changed, kwargs, jacobian = events[0]
    assert changed is True and jacobian == int(_DENSE)
    assert kwargs == {"use_optimized_kernel": True, "authored_is_sparse": None}
    assert events[1][1] == int(_DENSE)  # put_model saw the promoted model
    assert events[2][1] is False  # and uploaded it dense
    assert not env.mjw_model.opt.warn_overflow


def test_upload_keeps_the_auto_jacobian_when_the_policy_resolves_off(monkeypatch):
    """A config that authored the flag on must not promote where it resolved off."""

    uploaded_sparse: list[bool] = []
    real_put_model = direct_rl_env.put_model

    def upload(mjm, *args, **kwargs):
        uploaded = real_put_model(mjm, *args, **kwargs)
        uploaded_sparse.append(bool(uploaded.is_sparse))
        return uploaded

    monkeypatch.setattr(direct_rl_env, "put_model", upload)
    env = _stub_direct_rl_env(
        model=_model(), resolved_optimized=False, authored_optimized=True
    )

    env._upload_mjw_model()

    assert env.model.opt.jacobian == _AUTO
    assert uploaded_sparse == [True]


def test_promotion_logs_exactly_one_info_line(monkeypatch, caplog):
    env = _stub_direct_rl_env(model=_model(), resolved_optimized=True)

    with caplog.at_level(logging.INFO, logger=direct_rl_env.__name__):
        assert env._resolve_dense_jacobian_for_upload() is True

    promoted = [
        record
        for record in caplog.records
        if "JACOBIAN_DENSE_AUTO_RESOLVED" in record.getMessage()
    ]
    assert len(promoted) == 1
    assert promoted[0].levelno == logging.INFO
    assert "nv=35" in promoted[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=direct_rl_env.__name__):
        assert env._resolve_dense_jacobian_for_upload() is False
    assert not [
        record
        for record in caplog.records
        if "JACOBIAN_DENSE_AUTO_RESOLVED" in record.getMessage()
    ]


class _MinimalEnv(DirectRLEnv):
    def _get_observations(self):
        return {}

    def _get_rewards(self):
        return torch.zeros(self.num_envs)

    def _get_dones(self):
        zeros = torch.zeros(self.num_envs, dtype=torch.bool)
        return zeros, zeros.clone()


def _minimal_env_cfg(
    tmp_path: Path, *, authored_optimized: bool, is_sparse: bool | None = None
) -> DirectRLEnvCfg:
    mjcf_path = tmp_path / "dense_rule_mjcf.xml"
    mjcf_path.write_text(_spec(4, 5).to_xml())
    cfg = DirectRLEnvCfg(
        viewer=ViewerCfg(resolution=(64, 64)),
        sim=SimulationCfg(
            mjcf_path=str(mjcf_path),
            integrator="implicitfast",
            render_interval=1,
            njmax=64,
            nconmax=8,
            hz=120,
            iterations=100,
            is_sparse=is_sparse,
        ),
        scene=InteractiveSceneCfg(num_envs=1),
        decimation=1,
        episode_length_s=1.0,
        observation_space=4,
        action_space=2,
    )
    cfg.use_cuda_graph = False
    cfg.compile_mjw_step = False
    cfg.use_parallelized_kernel_launch = False
    cfg.use_optimized_kernel = authored_optimized
    return cfg


def _resolve_optimized_on(self) -> bool:
    """Stand in for a CUDA device where the specialized kernels resolve on."""

    return True


def _resolve_optimized_off(self) -> bool:
    """Stand in for a device where CUDA-only policies resolve off."""

    return False


def _record_startup_calls(monkeypatch) -> list[tuple[object, ...]]:
    events: list[tuple[object, ...]] = []
    real_promote = direct_rl_env.promote_dense_jacobian_for_fused_newton
    real_put_model = direct_rl_env.put_model
    real_mjdata = direct_rl_env.mujoco.MjData

    def promote(mjm, **kwargs):
        changed = real_promote(mjm, **kwargs)
        events.append(("promote", changed, kwargs["use_optimized_kernel"]))
        return changed

    def upload(mjm, *args, **kwargs):
        events.append(("put_model", int(mjm.opt.jacobian)))
        return real_put_model(mjm, *args, **kwargs)

    def create_data(mjm, *args, **kwargs):
        events.append(("mjdata", int(mjm.opt.jacobian)))
        return real_mjdata(mjm, *args, **kwargs)

    monkeypatch.setattr(
        direct_rl_env, "promote_dense_jacobian_for_fused_newton", promote
    )
    monkeypatch.setattr(direct_rl_env, "put_model", upload)
    monkeypatch.setattr(direct_rl_env.mujoco, "MjData", create_data)
    return events


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="CPU startup wiring: warp and torch must both resolve to the CPU",
)
def test_startup_promotes_before_upload_and_mjdata(monkeypatch, tmp_path):
    """Scene creation resolves the Jacobian before upload and `MjData`."""

    events = _record_startup_calls(monkeypatch)
    monkeypatch.setattr(
        DirectRLEnv, "_resolve_use_optimized_kernel", _resolve_optimized_on
    )
    cfg = _minimal_env_cfg(tmp_path, authored_optimized=False)

    env = _MinimalEnv(cfg, device="cpu")
    try:
        names = [event[0] for event in events]
        assert names.index("promote") < names.index("put_model")
        assert names.index("put_model") < names.index("mjdata")
        promote_event = events[names.index("promote")]
        assert promote_event[1:] == (True, True)  # changed, resolved policy on
        assert events[names.index("put_model")][1] == int(_DENSE)
        assert events[names.index("mjdata")][1] == int(_DENSE)
        assert not env.mjw_model.is_sparse
    finally:
        env.close()


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="CPU startup wiring: warp and torch must both resolve to the CPU",
)
def test_startup_keeps_upstream_sparse_when_the_policy_resolves_off(
    monkeypatch, tmp_path
):
    """An authored-on flag that resolves off leaves auto sparse."""

    events = _record_startup_calls(monkeypatch)
    monkeypatch.setattr(
        DirectRLEnv, "_resolve_use_optimized_kernel", _resolve_optimized_off
    )
    cfg = _minimal_env_cfg(tmp_path, authored_optimized=True)

    env = _MinimalEnv(cfg, device="cpu")
    try:
        names = [event[0] for event in events]
        promote_event = events[names.index("promote")]
        assert promote_event[1:] == (False, False)  # unchanged, resolved off
        assert events[names.index("put_model")][1] == int(_AUTO)
        assert env.model.opt.jacobian == _AUTO
        assert env.mjw_model.is_sparse
    finally:
        env.close()
