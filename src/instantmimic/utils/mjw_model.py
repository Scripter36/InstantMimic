"""MuJoCo Warp model upload with dense Jacobians beyond MuJoCo Warp's 60-DOF cap.

``mujoco_warp.put_model`` (3.16.0) rejects dense Jacobians for ``nv > 60``
("TODO(team): remove after _update_gradient for Newton uses tile operations for
islands"). The dense Newton path sizes its tiles by ``nv_pad`` (``nv + 1`` rounded
up to 16 for Newton), so it compiles and runs for larger models; at nv = 70
(nv_pad 80) 60 steps of a contact-rich model match CPU MuJoCo within 2e-7 in
qpos. Models without islands, the only kind this repository builds, are not
affected by the TODO. ``put_model`` here lifts the cap to DENSE_NV_MAX (nv_pad
80) and leaves every other model to the upstream function; at nv_pad 80 it also
widens the dense J^T D J Hessian block (WIDE_JTDAJ_BLOCK_DIM).

The lift edits the one ``nv_max = 60`` assignment of the pinned upstream source
and fails loudly when that line is not found exactly once (see the upgrade
checklist in ``docs/reference/simulation.md``).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable

import mujoco
import mujoco_warp as mjw

from instantmimic.utils.mjw_kernels.private import io as mjw_io

UPSTREAM_DENSE_NV_MAX = 60
DENSE_NV_MAX = 79  # Newton pads nv + 1 to 16: nv <= 79 keeps nv_pad <= 80
# Dense J^T D J Hessian block width at nv_pad 80 (upstream's 128 is tuned for
# smaller tiles): on a 68-DOF model (2048 worlds, ~157 rows) 256 threads
# took 336 us against 405 us for 128 (384: 358, 512: 348), with H identical bit for
# bit; at nv_pad 64 the two widths tie, so smaller models keep upstream's value.
WIDE_JTDAJ_BLOCK_DIM = 256


@functools.cache
def _wide_dense_put_model() -> Callable[[mujoco.MjModel], object]:
    source = inspect.getsource(mjw_io.put_model)
    line = f"nv_max = {UPSTREAM_DENSE_NV_MAX}"
    if source.count(line) != 1:
        raise RuntimeError(
            f"mujoco_warp.put_model no longer contains '{line}' exactly once; "
            "re-audit the dense DOF cap before raising it"
        )
    namespace = dict(vars(mjw_io))
    exec(
        compile(
            source.replace(line, f"nv_max = {DENSE_NV_MAX}"),
            mjw_io.__file__,
            "exec",
        ),
        namespace,
    )
    return namespace["put_model"]


def put_model(mjm: mujoco.MjModel):
    """``mujoco_warp.put_model``, with dense Jacobians allowed up to DENSE_NV_MAX."""
    if (
        mjm.nv <= UPSTREAM_DENSE_NV_MAX
        or mjm.opt.jacobian != mujoco.mjtJacobian.mjJAC_DENSE
    ):
        return mjw.put_model(mjm)
    if mjm.nv > DENSE_NV_MAX:
        raise ValueError(
            f"dense Jacobians support nv <= {DENSE_NV_MAX} here (nv = {mjm.nv})"
        )
    if mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP and not (
        mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ISLAND
    ):
        # the upstream cap guards the island path, which sleeping models use
        raise ValueError(
            f"dense Jacobians above nv = {UPSTREAM_DENSE_NV_MAX} need a model"
            " without sleeping islands"
        )
    m = _wide_dense_put_model()(mjm)
    if m.nv_pad >= 80:
        m.block_dim.update_gradient_JTDAJ_dense = WIDE_JTDAJ_BLOCK_DIM
    return m


def _dense_fused_newton_structure(mjm: mujoco.MjModel) -> bool:
    """Model-side structural gates of the dense fused Newton solver.

    Mirrors the parts of ``instantmimic.utils.mjw_kernels.newton_fusion.eligible``
    that are known from a compiled ``MjModel``: Newton with a positive iteration
    budget, pyramidal cone, a non-``DISCRETE`` integrator, and a DOF count that
    pads to at most 80 (``nv <= DENSE_NV_MAX``). Sleeping models are excluded
    because they solve through a compact context, which the fused iteration
    explicitly rejects. The remaining gates (CUDA device, line-search width,
    per-world scratch) come from the uploaded model and its data.
    """
    return (
        0 < mjm.nv <= DENSE_NV_MAX
        and mjm.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
        and mjm.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
        and mjm.opt.integrator != mujoco.mjtIntegrator.mjINT_DISCRETE
        and mjm.opt.iterations > 0
        and not (mjm.opt.enableflags & mujoco.mjtEnableBit.mjENBL_SLEEP)
    )


def promote_dense_jacobian_for_fused_newton(
    mjm: mujoco.MjModel,
    *,
    use_optimized_kernel: bool,
    authored_is_sparse: bool | None,
) -> bool:
    """Resolve an ``auto`` Jacobian to dense when the fused Newton path wants it.

    ``mujoco_warp`` resolves ``mjJAC_AUTO`` to sparse above 32 DOFs, which keeps
    the dense fused Newton solver away from mid-size models such as the 35-DOF
    G1. Raising the dense cap in ``put_model`` only helps models that already
    compile as dense. When the specialized kernels are on, the user left
    ``env.sim.is_sparse`` unset, and the fused path's structural gates hold, set
    the compiled model's Jacobian to dense before upload so that path is
    reachable. An authored ``env.sim.is_sparse`` or XML Jacobian setting always
    wins, and the rule does nothing when the specialized kernels are off.

    Returns whether the model was changed.
    """

    if not use_optimized_kernel or authored_is_sparse is not None:
        return False
    if mjm.opt.jacobian != mujoco.mjtJacobian.mjJAC_AUTO:
        return False
    # Only act where upstream would pick sparse; below that the model is already
    # dense and the fused path needs no Jacobian change.
    if not mjw_io.is_sparse(mjm):
        return False
    if not _dense_fused_newton_structure(mjm):
        return False
    mjm.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    return True
