"""Allocation-free MuJoCo Warp forward/step orchestration, derived from MuJoCo Warp 3.16.

Copyright 2025 The Newton Developers, licensed under the Apache License,
Version 2.0. Derived from mujoco_warp forward.py, smooth.py and solver.py.
Local changes fuse upstream stages that this repository always runs together and
hold their scratch in a persistent per-model cache so a captured step allocates
nothing. ``_qfrc_actuator_gravcomp_limits_and_qfrc_smooth`` merges the upstream
``_qfrc_actuator`` and ``_qfrc_smooth`` kernels into one launch; the
``_tile_cholesky_*_no_l_store`` factories drop the upstream factor store because
no later stage reads L back.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import FunctionType
from typing import Any

import numpy as np
import warp as wp

from instantmimic.utils.mjw_kernels import (
    newton_fusion,
    sparse_hessian,
    tree_fusion,
    tree_ldl,
    velocity_fusion,
)
from instantmimic.utils.mjw_kernels.indexed_factor import (
    make_indexed_cholesky_factor_solve_kernel,
)
from instantmimic.utils.mjw_kernels.kinematics_fusion import body_frames_kernel
from instantmimic.utils.mjw_kernels.private import (
    BroadphaseType,
    CollisionContext,
    Data,
    DisableBit,
    EnableBit,
    IntegratorType,
    JointType,
    Model,
    TileSet,
    TrnType,
    cache_kernel,
    collision_driver,
    collision_flex,
    constraint,
    derivative,
    event_scope,
    history,
    island,
    mjw_forward,
    passive,
    sensor,
    sleep,
    smooth,
    solver,
    support,
)
from instantmimic.utils.warp_kernels import expose_kernel_name

OptimizedFwdPositionStreams = tuple[wp.Stream, ...]


@dataclass(frozen=True)
class ImplicitQHContributor:
    """Named additive contribution to the implicit-fast effective inertia."""

    name: str
    apply: Callable[[wp.array], None]


@dataclass(frozen=True)
class PreAdvanceModifier:
    """Named modifier for acceleration and position-drift velocity before advance."""

    name: str
    apply: Callable[
        [Model, Data, wp.array, wp.array | None],
        tuple[wp.array, wp.array | None],
    ]
    implicit_factor: wp.array | None = None


KINEMATICS_TAIL_STREAMS = slice(0, 4)
COLLISION_CONSTRAINT_STREAM = 4
CAMLIGHT_STREAM = 5
TRANSMISSION_STREAM = 6
COM_POS_CINERT_STREAM = 7
COM_POS_CDOF_STREAM = 8
ACTUATOR_VELOCITY_STREAM = 9
TENDON_VELOCITY_STREAM = 10
PASSIVE_STREAM = 11
TENDON_BIAS_STREAM = 12
CRB_STREAM = 13
POSITION_VELOCITY_STREAM = 14
NUM_OPTIMIZED_FWD_POSITION_STREAMS = 15


def _skipped_passive_launches(m: Model) -> frozenset[Any]:
    """Passive kernels that cannot contribute for this model.

    Upstream launches gravity compensation over every (world, body, dof) and
    each thread exits on a zero ``body_gravcomp``; MuJoCo skips it when no body
    has gravity compensation (``ngravcomp == 0``), which leaves the zeroed
    ``qfrc_gravcomp`` unchanged.
    """
    gravcomp = getattr(m, "body_gravcomp", None)
    if gravcomp is not None and not np.any(gravcomp.numpy()):
        return frozenset({passive._gravity_force})
    return frozenset()


def _constraint_rows_impossible(m: Model, ball_joint_limits: Sequence[Any]) -> bool:
    """Whether ``make_constraint`` can never emit a row for this model.

    Rows come from equality, frictionloss, joint and tendon limits, contacts
    of candidate geom pairs (explicit pairs included), flexes and the
    repository's ball-limit rows. Without any, MuJoCo's ``nefc == 0`` result
    holds every step: no collision or constraint work, ``qacc = qacc_smooth``.
    Collision-sensor pairs, user-populated contacts and sleeping keep the
    stock path.
    """
    opt = getattr(m, "opt", None)
    if opt is None or not getattr(opt, "run_collision_detection", False):
        return False
    if ball_joint_limits or _solver_compact_enabled(m):
        return False
    if int(getattr(m, "neq", 1)) or int(getattr(m, "nflex", 1)):
        return False
    if int(getattr(m, "nsensorcollision", 1)):
        return False
    for name in (
        "jnt_limited_slide_hinge_adr",
        "jnt_limited_ball_adr",
        "tendon_limited_adr",
    ):
        if getattr(m, name).size:
            return False
    for name in ("dof_frictionloss", "tendon_frictionloss"):
        values = getattr(m, name)
        if values.size and np.any(values.numpy() > 0.0):
            return False
    pairs = m.nxn_pairid_filtered.numpy().reshape(-1, 2)
    return not np.any(pairs[:, 0] > -2)


class _PreallocatedWarpCall:
    """Bind direct Warp allocations in one upstream function to stable buffers."""

    def __init__(
        self,
        fn: Any,
        allocations: Sequence[tuple[str, wp.array]],
        *,
        global_overrides: dict[str, Any] | None = None,
        skipped_launches: frozenset[Any] = frozenset(),
    ) -> None:
        raw_fn = getattr(fn, "__wrapped__", fn)
        globals_dict = dict(raw_fn.__globals__)
        globals_dict["wp"] = self
        if global_overrides:
            globals_dict.update(global_overrides)
        self._fn = FunctionType(
            raw_fn.__code__,
            globals_dict,
            raw_fn.__name__,
            raw_fn.__defaults__,
            raw_fn.__closure__,
        )
        self._allocations = tuple(allocations)
        self._next_allocation = 0
        self._skipped_launches = skipped_launches
        self._launch_substitutions: dict[Any, Any] = {}

    @classmethod
    def from_allocator(cls, fn: Any, *args: Any) -> _PreallocatedWarpCall:
        """Bind an audited, allocation-only upstream workspace constructor.

        Record real arrays once during construction; subsequent calls replay
        the same allocation sequence including zero/full initialization. A
        constructor that starts launching kernels requires an explicit audit.
        """
        allocations: list[tuple[str, wp.array]] = []

        class Recorder:
            def __getattr__(self, name: str) -> Any:
                if name in {"empty", "empty_like", "zeros", "ones", "full", "clone"}:

                    def allocate(*a: Any, **kw: Any) -> wp.array:
                        buffer = getattr(wp, name)(*a, **kw)
                        allocations.append((name, buffer))
                        return buffer

                    return allocate
                if name in {"launch", "launch_tiled", "capture_if", "capture_while"}:
                    raise RuntimeError(
                        "workspace allocator unexpectedly launches kernels"
                    )
                return getattr(wp, name)

        raw = getattr(fn, "__wrapped__", fn)
        recording = FunctionType(
            raw.__code__,
            {**raw.__globals__, "wp": Recorder()},
            raw.__name__,
            raw.__defaults__,
            raw.__closure__,
        )
        recording(*args)
        return cls(fn, allocations)

    def add_launch_substitutions(self, substitutions: dict[Any, Any]) -> None:
        """Replace launches of specific upstream kernels.

        Each value is called as ``fn(launch, kernel, kwargs)`` in place of the
        upstream ``wp.launch``/``wp.launch_tiled`` of that kernel object.
        """
        overlap = set(substitutions) & set(self._launch_substitutions)
        if overlap:
            raise ValueError(
                "a launch substitution is already registered for this kernel"
            )
        self._launch_substitutions.update(substitutions)

    def __getattr__(self, name: str) -> Any:
        return getattr(wp, name)

    def _take(self, operation: str) -> wp.array:
        index = self._next_allocation
        if index >= len(self._allocations):
            raise RuntimeError(
                f"unexpected wp.{operation}() in preallocated upstream call"
            )
        expected_operation, buffer = self._allocations[index]
        if expected_operation != operation:
            raise RuntimeError(
                "preallocated upstream allocation order changed: "
                f"expected wp.{expected_operation}(), got wp.{operation}()"
            )
        self._next_allocation += 1
        return buffer

    def _dispatch(self, launcher: Any, kernel: Any, args: tuple, kwargs: dict) -> Any:
        if kernel in self._skipped_launches:
            return None
        substitute = self._launch_substitutions.get(kernel)
        if substitute is None:
            return launcher(kernel, *args, **kwargs)
        if args:
            raise RuntimeError(
                "substituted upstream launch passed positional arguments"
            )
        return substitute(launcher, kernel, kwargs)

    def launch(self, kernel: Any, *args: Any, **kwargs: Any) -> Any:
        return self._dispatch(wp.launch, kernel, args, kwargs)

    def launch_tiled(self, kernel: Any, *args: Any, **kwargs: Any) -> Any:
        return self._dispatch(wp.launch_tiled, kernel, args, kwargs)

    @staticmethod
    def _check(operation: str, buffer: wp.array, shape: Any, dtype: Any) -> None:
        """Reject a replayed request whose shape or dtype differs from the
        recorded buffer (an upstream change that kept the allocation order)."""
        if not isinstance(buffer, wp.array):
            return
        if shape is not None:
            dims = (shape,) if isinstance(shape, int | np.integer) else shape
            if tuple(int(n) for n in dims) != tuple(buffer.shape):
                raise RuntimeError(
                    f"preallocated wp.{operation}() shape changed: recorded"
                    f" {tuple(buffer.shape)}, requested {tuple(dims)}"
                )
        if dtype is not None:
            canonical = {float: wp.float32, int: wp.int32, bool: wp.bool}
            if canonical.get(dtype, dtype) != buffer.dtype:
                raise RuntimeError(
                    f"preallocated wp.{operation}() dtype changed: recorded"
                    f" {buffer.dtype}, requested {dtype}"
                )

    def _take_checked(self, operation: str, args: tuple, kwargs: dict) -> wp.array:
        buffer = self._take(operation)
        shape = kwargs.get("shape", args[0] if args else None)
        self._check(operation, buffer, shape, kwargs.get("dtype"))
        return buffer

    def empty(self, *args: Any, **kwargs: Any) -> wp.array:
        return self._take_checked("empty", args, kwargs)

    def empty_like(self, source: wp.array, *_args: Any, **kwargs: Any) -> wp.array:
        buffer = self._take("empty_like")
        self._check(
            "empty_like",
            buffer,
            getattr(source, "shape", None),
            kwargs.get("dtype", getattr(source, "dtype", None)),
        )
        return buffer

    def clone(self, source: wp.array, *_args: Any, **_kwargs: Any) -> wp.array:
        buffer = self._take("clone")
        self._check(
            "clone",
            buffer,
            getattr(source, "shape", None),
            getattr(source, "dtype", None),
        )
        wp.copy(buffer, source)
        return buffer

    def zeros(self, *args: Any, **kwargs: Any) -> wp.array:
        buffer = self._take_checked("zeros", args, kwargs)
        buffer.zero_()
        return buffer

    def ones(self, *args: Any, **kwargs: Any) -> wp.array:
        buffer = self._take_checked("ones", args, kwargs)
        buffer.fill_(1)
        return buffer

    def full(
        self,
        *args: Any,
        value: Any | None = None,
        **_kwargs: Any,
    ) -> wp.array:
        if value is None:
            if len(args) < 2:
                raise TypeError("wp.full() value is missing")
            value = args[1]
        buffer = self._take("full")
        self._check(
            "full",
            buffer,
            _kwargs.get("shape", args[0] if args else None),
            _kwargs.get("dtype"),
        )
        buffer.fill_(value)
        return buffer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._next_allocation = 0
        result = self._fn(*args, **kwargs)
        if self._next_allocation != len(self._allocations):
            raise RuntimeError(
                "preallocated upstream call did not consume its expected allocations"
            )
        return result


class _BallLimitConstraintCall(_PreallocatedWarpCall):
    """Insert directional rows at the native ball stage without global patches."""

    def __init__(self, m: Model, d: Data, nnz: wp.array, buffers: Any) -> None:
        super().__init__(constraint.make_constraint, (("empty", nnz),))
        from instantmimic.utils.mjw_kernels.private import types

        self._limit_kernel = constraint._limit_ball(
            m.is_sparse, m.opt.solver == types.SolverType.NEWTON, False
        )
        self._model = m
        self._data = d
        self._nnz = nnz
        self._buffers = buffers
        self._inserted = False

    def launch(self, *args: Any, **kwargs: Any) -> Any:
        kernel = args[0] if args else kwargs.pop("kernel")
        result = super().launch(kernel, *args[1:], **kwargs)
        if kernel is self._limit_kernel:
            self._buffers.launch(self._model, self._data, self._nnz)
            self._inserted = True
        return result

    def __call__(self, m: Model, d: Data) -> Any:
        self._model, self._data = m, d
        self._inserted = False
        self._buffers.overflow.zero_()
        self._buffers.singular.zero_()
        result = super().__call__(m, d)
        if not (m.opt.disableflags & (DisableBit.CONSTRAINT | DisableBit.LIMIT)):
            if not self._inserted:
                raise RuntimeError("MuJoCo Warp native ball constraint launch changed")
        return result


class _AttributeOverrides:
    """Delegate module attributes except for explicitly replaced call sites."""

    def __init__(self, target: Any, overrides: dict[str, Any]) -> None:
        self._target = target
        self._overrides = dict(overrides)

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._target, name)


class _PreallocatedSensorDelayCall:
    """Reuse one sensor-history staging array across sequential sensor stages."""

    def __init__(self, fresh_sensordata: wp.array) -> None:
        self._call = _PreallocatedWarpCall(
            history.apply_sensor_delay,
            (("empty_like", fresh_sensordata),),
        )

    def __call__(self, m: Model, d: Data, sensorid: wp.array) -> None:
        if getattr(m, "nhistory", 0) == 0 or sensorid.shape[0] == 0:
            return
        self._call(m, d, sensorid)


def _make_contact_sensor_call(m: Model, d: Data, adr: wp.array, device: Any):
    allocations = ()
    if adr.size:
        shape = (d.nworld, adr.size, m.opt.contact_sensor_maxmatch)
        allocations = (
            ("zeros", wp.empty((d.nworld, adr.size), dtype=int, device=device)),
            ("full", wp.empty(shape, dtype=int, device=device)),
            ("full", wp.empty(shape, dtype=float, device=device)),
            ("empty", wp.empty(shape, dtype=float, device=device)),
        )
    return _PreallocatedWarpCall(sensor._eval_sensor_contact, allocations)


@dataclass(frozen=True)
class BlockedDenseFactorSolveScratch:
    """Persistent indexed layout and factor for a single dense inertia block."""

    nv: int
    pad: int
    block_size: int
    block_dim: int
    U: wp.array
    indices: wp.array
    identity: wp.array
    kernel: Any


@dataclass(frozen=True)
class RK4Scratch:
    """Persistent state and accumulation buffers for repo-owned RK4."""

    qpos_t0: wp.array
    qvel_t0: wp.array
    qvel_rk: wp.array
    qacc_rk: wp.array
    act_t0: wp.array | None
    act_dot_rk: wp.array | None
    time_t0: wp.array


@dataclass
class OptimizedFwdPositionScratch:
    """Persistent scratch buffers used to keep CUDA graph capture allocation-free."""

    collision_context: CollisionContext
    transmission_moment_nnz: wp.array
    collision_narrowphase_call: Any | None = None
    collision_broadphase_call: Any | None = None
    flex_collision_call: Any | None = None
    transmission_body_ncon: wp.array | None = None
    tendon_wrap_geom_xpos: wp.array | None = None
    tendon_call: Any | None = None
    tendon_bias_Jdot: wp.array | None = None
    tendon_bias_coef: wp.array | None = None
    implicit_actuator_vel: wp.array | None = None
    implicit_qderiv_pair_mask: wp.array | None = None
    implicit_qderiv: wp.array | None = None
    implicit_qacc: wp.array | None = None
    implicit_max_moment_rownnz: int = 0
    collision_awake_prev: wp.array | None = None
    collision_incremental_broadphase_call: Any | None = None
    solver_context: Any | None = None
    solver_nsolving: wp.array | None = None
    solver_uses_compact: bool = False
    # Flatten only solver control flow inside an external conditional graph.
    unroll_solver_iterations: bool = False
    solver_compact_model: Any | None = None
    solver_compact_data: Any | None = None
    constraint_call: Any | None = None
    ball_limit_buffers: Any | None = None
    # No constraint row is possible (``_constraint_rows_impossible``): collision,
    # make_constraint and the solver are skipped.
    constraint_free: bool = False
    # Allocation-free ``support.mul_m`` (efc.Ma = M qacc for every world).
    mul_m_call: Any | None = None
    actuation_delayed_ctrl: wp.array | None = None
    actuation_tendon_force: wp.array | None = None
    actuation_call: Any | None = None
    sensor_pos_call: Any | None = None
    sensor_subtree_bodyvel: wp.array | None = None
    sensor_vel_call: Any | None = None
    sensor_acc_call: Any | None = None
    rne_postconstraint_call: Any | None = None
    energy_vel_call: Any | None = None
    passive_call: Any | None = None
    advance_call: Any | None = None
    island_call: Any | None = None
    island_mapping_call: Any | None = None
    rk4: RK4Scratch | None = None
    euler_call: Any | None = None
    implicit_call: Any | None = None
    implicit_deriv_call: Any | None = None
    implicit_free_body_call: Any | None = None
    implicit_qld: wp.array | None = None
    implicit_ql_diag_inv: wp.array | None = None
    implicit_qh_contributors: tuple[ImplicitQHContributor, ...] = ()
    implicit_qh_registration_frozen: bool = False
    pre_advance_modifier: PreAdvanceModifier | None = None
    dense_factor_solve: BlockedDenseFactorSolveScratch | None = None
    tree_reduction: tree_fusion.TreeReductionScratch | None = None
    tree_factor_solve: tree_ldl.TreeFactorSolveScratch | None = None


@dataclass(frozen=True)
class OptimizedFwdVelocityResult:
    """Async work started by the optimized velocity pass."""

    actuation_started: bool = False
    actuation_stream: wp.Stream | None = None
    actuator_limits_pending: bool = False


@dataclass(frozen=True)
class PendingComVelocity:
    """COM-velocity work launched alongside the position-stage tail."""

    started: bool = False
    stream: wp.Stream | None = None


@dataclass(frozen=True)
class KinematicsTailStreams:
    """Side streams carrying kinematic tail work after the branch kernel."""

    body_matrix: wp.Stream | None = None
    body_inertial: wp.Stream | None = None
    geom: wp.Stream | None = None
    site: wp.Stream | None = None

    def as_tuple(self) -> tuple[wp.Stream, ...]:
        return tuple(
            dict.fromkeys(
                stream
                for stream in (
                    self.body_matrix,
                    self.body_inertial,
                    self.geom,
                    self.site,
                )
                if stream is not None
            )
        )


@dataclass(frozen=True)
class OptimizedImplicitDerivResult:
    """Async implicit qDeriv work started before solver.solve."""

    started: bool = False
    stream: wp.Stream | None = None
    qderiv: wp.array | None = None
    qacc: wp.array | None = None


@wp.kernel(enable_backward=False)
def _mark_dense_qderiv_pair_mask(
    qM_fullm_i: wp.array[int],
    qM_fullm_j: wp.array[int],
    mask_out: wp.array2d[int],
):
    elemid = wp.tid()
    dofi = qM_fullm_i[elemid]
    dofj = qM_fullm_j[elemid]
    mask_out[dofi, dofj] = 1
    mask_out[dofj, dofi] = 1


@wp.kernel(enable_backward=False)
def _qderiv_actuator_passive_actuation_dense_actuator_centric(
    # Data in:
    moment_rownnz_in: wp.array2d[int],
    moment_rowadr_in: wp.array2d[int],
    moment_colind_in: wp.array2d[int],
    actuator_moment_in: wp.array2d[float],
    # In:
    vel_in: wp.array2d[float],
    pair_mask_in: wp.array2d[int],
    # Out:
    qDeriv_out: wp.array3d[float],
):
    worldid, actid, rowi, rowj = wp.tid()
    rownnz = moment_rownnz_in[worldid, actid]
    if rowi >= rownnz or rowj > rowi:
        return

    vel = vel_in[worldid, actid]
    if vel == 0.0:
        return

    rowadr = moment_rowadr_in[worldid, actid]
    sparse_i = rowadr + rowi
    sparse_j = rowadr + rowj
    dofi = moment_colind_in[worldid, sparse_i]
    dofj = moment_colind_in[worldid, sparse_j]
    if pair_mask_in[dofi, dofj] == 0:
        return

    contrib = (
        actuator_moment_in[worldid, sparse_i]
        * actuator_moment_in[worldid, sparse_j]
        * vel
    )
    if contrib == 0.0:
        return

    wp.atomic_add(qDeriv_out, worldid, dofi, dofj, contrib)
    if dofi != dofj:
        wp.atomic_add(qDeriv_out, worldid, dofj, dofi, contrib)


@wp.kernel(enable_backward=False)
def _qderiv_actuator_passive_actuation_m_structure_actuator_centric(
    # Model:
    M_elemid: wp.array2d[int],
    # Data in:
    moment_rownnz_in: wp.array2d[int],
    moment_rowadr_in: wp.array2d[int],
    moment_colind_in: wp.array2d[int],
    actuator_moment_in: wp.array2d[float],
    # In:
    vel_in: wp.array2d[float],
    # Out:
    qDeriv_out: wp.array2d[float],
):
    worldid, actid, rowi, rowj = wp.tid()
    rownnz = moment_rownnz_in[worldid, actid]
    if rowi >= rownnz or rowj > rowi:
        return

    vel = vel_in[worldid, actid]
    if vel == 0.0:
        return

    rowadr = moment_rowadr_in[worldid, actid]
    sparse_i = rowadr + rowi
    sparse_j = rowadr + rowj
    dofi = moment_colind_in[worldid, sparse_i]
    dofj = moment_colind_in[worldid, sparse_j]
    elemid = M_elemid[dofi, dofj]
    if elemid < 0:
        return

    contrib = (
        actuator_moment_in[worldid, sparse_i]
        * actuator_moment_in[worldid, sparse_j]
        * vel
    )
    if contrib == 0.0:
        return

    wp.atomic_add(qDeriv_out[worldid], elemid, contrib)


@cache_kernel
def _tile_cholesky_factorize_solve_no_l_store(tile: TileSet):
    """Dense Cholesky factorize+solve tile kernel that does not persist L."""

    # `module="unique"` is the mujoco-warp `@cache_kernel` convention: every tile
    # specialization compiles from the same source, so each needs its own module.
    @wp.kernel(module="unique", enable_backward=False)
    @expose_kernel_name
    def cholesky_factorize_solve_no_l_store(
        # In:
        M: wp.array3d[float],
        y: wp.array2d[float],
        adr: wp.array[int],
        # Out:
        x: wp.array2d[float],
    ):
        worldid, nodeid = wp.tid()
        TILE_SIZE = wp.static(tile.size)

        dofid = adr[nodeid]
        M_tile = wp.tile_load(
            M[worldid],
            shape=(TILE_SIZE, TILE_SIZE),
            offset=(dofid, dofid),
        )
        y_slice = wp.tile_load(y[worldid], shape=(TILE_SIZE,), offset=(dofid,))

        L_tile = wp.tile_cholesky(M_tile)
        x_slice = wp.tile_cholesky_solve(L_tile, y_slice)
        wp.tile_store(x[worldid], x_slice, offset=(dofid,))

    return cholesky_factorize_solve_no_l_store


@cache_kernel
def _tile_cholesky_factorize_solve_m_no_l_store(tile: TileSet):
    """MuJoCo-Warp M-structure factor+solve without persisting dense L."""
    block_size = tile.size
    block_area = block_size * block_size

    # `module="unique"` is the mujoco-warp `@cache_kernel` convention: every tile
    # specialization compiles from the same source, so each needs its own module.
    @wp.kernel(module="unique", enable_backward=False)
    @expose_kernel_name
    def cholesky_factorize_solve_m_no_l_store(
        # Data in:
        M_in: wp.array2d[float],
        # In:
        block_elemid: wp.array[int],
        block_dof: wp.array[int],
        y: wp.array2d[float],
        # Out:
        x: wp.array2d[float],
    ):
        worldid, blk = wp.tid()
        start = block_dof[blk]

        idx = wp.tile_load(
            block_elemid,
            shape=(block_area,),
            offset=(blk * block_area,),
            storage="shared",
        )
        block = wp.tile_load_indexed(
            M_in[worldid],
            idx,
            shape=(block_area,),
            storage="shared",
        )

        L = wp.tile_cholesky(
            wp.tile_reshape(block, (block_size, block_size)),
            fill_mode="upper",
        )
        rhs = wp.tile_load(y[worldid], shape=(block_size,), offset=(start,))
        sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")
        wp.tile_store(x[worldid], sol, offset=(start,))

    return cholesky_factorize_solve_m_no_l_store


def _has_kernel_args(kernel: Any) -> bool:
    return hasattr(getattr(kernel, "adj", None), "args")


def _inertia_matrix(d: Data) -> wp.array:
    if hasattr(d, "M"):
        return d.M
    return d.qM


def _fullm_indices(m: Model) -> tuple[wp.array, wp.array]:
    if hasattr(m, "M_fullm_i") and hasattr(m, "M_fullm_j"):
        return m.M_fullm_i, m.M_fullm_j
    return m.qM_fullm_i, m.qM_fullm_j


def _matrix_tiles(m: Model) -> Sequence[TileSet]:
    if hasattr(m, "M_tiles"):
        return m.M_tiles
    return m.qM_tiles


def _has_m_structure(m: Model, d: Data | None = None) -> bool:
    if hasattr(m, "M_tiles"):
        return True
    return d is not None and hasattr(d, "M")


def _array_width(array: Any) -> int:
    shape = tuple(getattr(array, "shape", ()))
    return int(shape[1]) if len(shape) > 1 else 0


def _has_sparse_m_tail(m: Model, d: Data) -> bool:
    """Return whether 3.16 stores an LDL sparse tail after dense M blocks."""
    if not hasattr(m, "qLD_block_total"):
        return bool(getattr(m, "qLD_has_sparse", False))
    return _array_width(d.qLD) > int(m.qLD_block_total)


def _factor_solve_i_dense_no_l_store(
    m: Model,
    d: Data,
    M: wp.array,
    x: wp.array,
    y: wp.array,
    D: wp.array | None = None,
) -> None:
    """Dense factor+solve for implicit integration when L is not reused."""
    if _has_m_structure(m, d):
        if _has_sparse_m_tail(m, d):
            smooth.factor_solve_i(m, d, M, d.qLD, d.qLDiagInv, x, y)
            return

        diag_output = d.qLDiagInv if D is None else D
        for tile in _matrix_tiles(m):
            elemid = getattr(tile, "elemid", None)
            if getattr(elemid, "size", 0) == 0:
                wp.launch(
                    smooth._small_cholesky_factorize_solve_block(tile.size),
                    dim=(d.nworld, tile.adr.size),
                    inputs=[m.M_rowadr, m.qLD_block_adr, M, tile.adr, y],
                    outputs=[diag_output, x, d.qLD],
                    block_dim=m.block_dim.small_cholesky,
                )
            else:
                # 128 threads hide the indexed gather's latency: a 31-dof
                # block takes 26 us instead of 35 us at 32 (4096 worlds).
                block_dim = 32 if tile.size <= 16 else _TILE_FACTOR_SOLVE_BLOCK_DIM
                wp.launch_tiled(
                    _tile_cholesky_factorize_solve_m_no_l_store(tile),
                    dim=(d.nworld, tile.adr.size),
                    inputs=[M, tile.elemid, tile.adr, y],
                    outputs=[x],
                    block_dim=block_dim,
                )
        return

    for tile in _matrix_tiles(m):
        wp.launch_tiled(
            _tile_cholesky_factorize_solve_no_l_store(tile),
            dim=(d.nworld, tile.adr.size),
            inputs=[M, y, tile.adr],
            outputs=[x],
            block_dim=m.block_dim.cholesky_factorize_solve,
        )


def factor_solve_i_no_l_store(
    m: Model,
    d: Data,
    M: wp.array,
    L: wp.array,
    D: wp.array,
    x: wp.array,
    y: wp.array,
    tree_factor_solve: tree_ldl.TreeFactorSolveScratch | None = None,
) -> None:
    """Factor and solve while skipping unused dense-factor output.

    The generated tree L'DL solve, when given, covers dense blocks and a sparse
    tail alike and writes only the solution. Otherwise sparse regions use the
    upstream factor workspace because their solve consumes it, and fully dense
    models use the fused tile solve that keeps the Cholesky factor in shared
    memory.
    """
    if tree_factor_solve is not None:
        tree_ldl.tree_factor_solve(tree_factor_solve, M, x, y)
        return
    if _has_sparse_m_tail(m, d):
        smooth.factor_solve_i(m, d, M, L, D, x, y)
        return
    _factor_solve_i_dense_no_l_store(m, d, M, x, y, D)


# Small blocked tiles avoid the long dependency chain of a full-matrix tile.
_BLOCKED_DENSE_FACTOR_BLOCK_SIZE = 16
_BLOCKED_DENSE_FACTOR_BLOCK_DIM = 64
_BLOCKED_DENSE_FACTOR_MIN_NV = 32
_TILE_FACTOR_SOLVE_BLOCK_DIM = 128


@wp.kernel(enable_backward=False)
def _copy_dense_factor_kernel(
    U_in: wp.array3d[float],
    qLD_block_adr: wp.array[int],
    size: int,
    L_out: wp.array2d[float],
):
    """Persist the blocked upper factor in MuJoCo Warp's dense-block layout."""
    worldid, slot = wp.tid()
    row = slot // size
    col = slot - row * size
    L_out[worldid, qLD_block_adr[0] + slot] = U_in[worldid, row, col]


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def create_blocked_dense_factor_solve_scratch(
    m: Model,
    d: Data,
    device: Any = None,
    *,
    minimum_nv: int = _BLOCKED_DENSE_FACTOR_MIN_NV,
) -> BlockedDenseFactorSolveScratch | None:
    """Allocate blocked factor+solve buffers when the model qualifies.

    Returns ``None`` for models whose inertia matrix has a sparse tail or is not
    one dense tile spanning every DOF, and for small models where the
    single-tile kernel is already cheap. Constraint-Jacobian sparsity is
    independent of this inertia-layout decision.
    """
    nv = getattr(m, "nv", 0)
    if not isinstance(nv, int) or nv < minimum_nv:
        return None
    if not hasattr(m, "qLD_block_total") or _has_sparse_m_tail(m, d):
        return None
    tiles = getattr(m, "M_tiles", None)
    if tiles is None or len(tiles) != 1:
        return None
    tile = tiles[0]
    elemid = getattr(tile, "elemid", None)
    if int(getattr(tile, "size", 0)) != nv or elemid is None:
        return None
    if getattr(getattr(tile, "adr", None), "size", 0) != 1:
        return None
    if tuple(getattr(elemid, "shape", ())) != (nv * nv,):
        return None
    nworld = getattr(d, "nworld", 0)
    if not isinstance(nworld, int) or nworld <= 0:
        return None
    if getattr(d, "M", None) is None:
        return None

    block_size = _BLOCKED_DENSE_FACTOR_BLOCK_SIZE
    pad = _round_up(nv, block_size)
    U = wp.zeros((nworld, pad, pad), dtype=float, device=device)
    indices = elemid.numpy().reshape(nv, nv)
    nblk = pad // block_size
    padded = np.full((pad, pad), d.M.shape[1], dtype=np.int32)
    padded[:nv, :nv] = indices
    layout = (
        padded.reshape(nblk, block_size, nblk, block_size)
        .transpose(0, 2, 1, 3)
        .copy()
        .reshape(-1)
    )
    identity = np.zeros((block_size, block_size), dtype=np.float32)
    for index in range(block_size):
        if (nblk - 1) * block_size + index >= nv:
            identity[index, index] = 1.0
    return BlockedDenseFactorSolveScratch(
        nv=nv,
        pad=pad,
        block_size=block_size,
        block_dim=_BLOCKED_DENSE_FACTOR_BLOCK_DIM,
        U=U,
        indices=wp.array(layout, dtype=int, device=device),
        identity=wp.array(identity.reshape(-1), dtype=float, device=device),
        kernel=make_indexed_cholesky_factor_solve_kernel(block_size, pad, nv),
    )


def _blocked_dense_factor_solve(
    d: Data,
    M: wp.array,
    x: wp.array,
    y: wp.array,
    scratch: BlockedDenseFactorSolveScratch,
    *,
    qLD_block_adr: wp.array | None = None,
    factor_out: wp.array | None = None,
    factor_workspace: wp.array | None = None,
) -> None:
    nworld = d.nworld
    workspace = scratch.U if factor_workspace is None else factor_workspace
    wp.launch_tiled(
        scratch.kernel,
        dim=nworld,
        block_dim=scratch.block_dim,
        inputs=[
            M,
            scratch.indices,
            scratch.identity,
            y.reshape((nworld, scratch.nv, 1)),
            scratch.pad,
            scratch.nv,
        ],
        outputs=[workspace, x.reshape((nworld, scratch.nv, 1))],
    )
    if factor_out is not None:
        if qLD_block_adr is None:
            raise RuntimeError("persisting a blocked factor requires qLD block offsets")
        wp.launch(
            _copy_dense_factor_kernel,
            dim=(nworld, scratch.nv * scratch.nv),
            inputs=[workspace, qLD_block_adr, scratch.nv],
            outputs=[factor_out],
        )


def _dense_factor_solve_scratch(
    scratch: OptimizedFwdPositionScratch | None,
) -> BlockedDenseFactorSolveScratch | None:
    if scratch is None:
        return None
    return getattr(scratch, "dense_factor_solve", None)


def _tree_factor_solve_scratch(
    scratch: OptimizedFwdPositionScratch | None,
) -> tree_ldl.TreeFactorSolveScratch | None:
    if scratch is None:
        return None
    return getattr(scratch, "tree_factor_solve", None)


def _implicit_factor_workspace(
    scratch: OptimizedFwdPositionScratch | None,
) -> wp.array | None:
    """Return the pre-advance modifier's requested implicit qH factor buffer."""
    modifier = getattr(scratch, "pre_advance_modifier", None)
    return getattr(modifier, "implicit_factor", None)


def _reject_unretained_implicit_factor(
    scratch: OptimizedFwdPositionScratch | None,
) -> None:
    if _implicit_factor_workspace(scratch) is not None:
        raise RuntimeError(
            "a pre-advance modifier requested the implicit qH factor, but this "
            "integration path does not factor qH with the blocked dense kernel"
        )


def _factor_solve_i_dense(
    m: Model,
    d: Data,
    M: wp.array,
    x: wp.array,
    y: wp.array,
    dense_factor_solve: BlockedDenseFactorSolveScratch | None = None,
    *,
    retain_factor: bool = False,
    factor_workspace: wp.array | None = None,
    tree_factor_solve: tree_ldl.TreeFactorSolveScratch | None = None,
) -> None:
    """Dense factor+solve that prefers the blocked kernel when scratch exists.

    When no factor is kept, a generated tree L'DL solve (``tree_factor_solve``)
    replaces both, including a sparse tail: it reads M's CSR directly. ``d.qLD``
    is then not written; only the CG solver reads it, and CG retains the factor.
    """
    if factor_workspace is not None and dense_factor_solve is None:
        raise RuntimeError(
            "a retained factor workspace requires the blocked dense path"
        )
    if tree_factor_solve is not None and not retain_factor and factor_workspace is None:
        tree_ldl.tree_factor_solve(tree_factor_solve, M, x, y)
        return
    if dense_factor_solve is not None:
        _blocked_dense_factor_solve(
            d,
            M,
            x,
            y,
            dense_factor_solve,
            qLD_block_adr=m.qLD_block_adr if retain_factor else None,
            factor_out=d.qLD if retain_factor else None,
            factor_workspace=factor_workspace,
        )
        return
    if retain_factor:
        smooth.factor_solve_i(m, d, M, d.qLD, d.qLDiagInv, x, y)
        return
    _factor_solve_i_dense_no_l_store(m, d, M, x, y)


@wp.kernel(enable_backward=False)
def _qfrc_actuator_gravcomp_limits_and_qfrc_smooth(
    # Model:
    jnt_actfrclimited: wp.array[bool],
    jnt_actgravcomp: wp.array[int],
    jnt_actfrcrange: wp.array2d[wp.vec2],
    dof_jntid: wp.array[int],
    # Data in:
    qfrc_gravcomp_in: wp.array2d[float],
    qfrc_applied_in: wp.array2d[float],
    qfrc_bias_in: wp.array2d[float],
    qfrc_passive_in: wp.array2d[float],
    qfrc_actuator_in: wp.array2d[float],
    # In:
    gravity_enabled: bool,
    # Data out:
    qfrc_actuator_out: wp.array2d[float],
    qfrc_smooth_out: wp.array2d[float],
):
    worldid, dofid = wp.tid()
    jntid = dof_jntid[dofid]

    qfrc = qfrc_actuator_in[worldid, dofid]
    if gravity_enabled and jnt_actgravcomp[jntid]:
        qfrc += qfrc_gravcomp_in[worldid, dofid]
    if jnt_actfrclimited[jntid]:
        frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], jntid]
        qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])

    qfrc_actuator_out[worldid, dofid] = qfrc
    qfrc_smooth_out[worldid, dofid] = (
        qfrc_passive_in[worldid, dofid]
        - qfrc_bias_in[worldid, dofid]
        + qfrc
        + qfrc_applied_in[worldid, dofid]
    )


def _can_start_actuation_early(m: Model) -> bool:
    if not getattr(m, "nactuator", 0):
        return False
    if m.opt.disableflags & DisableBit.ACTUATION:
        return False
    if getattr(m, "ntendon", 0):
        return False
    if getattr(m, "nsensor", 0):
        return False
    if getattr(m, "nhistory", 0):
        return False
    for callback_name in (
        "control",
        "act_dyn",
        "act_gain",
        "act_bias",
        "passive",
        "sensor",
    ):
        if _callback_enabled(m, callback_name):
            return False
    return True


def _start_fwd_actuation_no_tendon(
    stream: wp.Stream | None,
    m: Model,
    d: Data,
    *,
    apply_limits: bool = True,
) -> None:
    _launch_in_stream(
        stream,
        mjw_forward._actuator_force,
        dim=(d.nworld, m.nactuator),
        inputs=[
            m.na,
            m.opt.timestep,
            m.jnt_type,
            m.actuator_trntype,
            m.actuator_dyntype,
            m.actuator_gaintype,
            m.actuator_biastype,
            m.actuator_ctrladr,
            m.actuator_ctrlnum,
            m.actuator_ctrlspec,
            m.actuator_actadr,
            m.actuator_actnum,
            m.actuator_trnid,
            m.actuator_dynprm,
            m.actuator_gainprm,
            m.actuator_biasprm,
            m.actuator_actlimited,
            m.actuator_actrange,
            m.actuator_actearly,
            m.actuator_forcelimited,
            m.actuator_forcerange,
            m.actuator_ctrllimited,
            m.actuator_ctrlrange,
            m.actuator_gear,
            m.actuator_acc0,
            m.actuator_lengthrange,
            d.act,
            d.ctrl,
            d.actuator_length,
            d.actuator_velocity,
            m.opt.disableflags & DisableBit.CLAMPCTRL,
        ],
        outputs=[d.act_dot, d.actuator_force],
    )
    _call_in_stream(stream, d.qfrc_actuator.zero_)
    _launch_in_stream(
        stream,
        mjw_forward._qfrc_actuator,
        dim=(d.nworld, m.nactuator),
        inputs=[
            d.moment_rownnz,
            d.moment_rowadr,
            d.moment_colind,
            d.actuator_moment,
            d.actuator_force,
        ],
        outputs=[d.qfrc_actuator],
    )
    if apply_limits:
        _launch_in_stream(
            stream,
            mjw_forward._qfrc_actuator_gravcomp_limits,
            dim=(d.nworld, m.nv),
            inputs=[
                m.jnt_actfrclimited,
                m.jnt_actgravcomp,
                m.jnt_actfrcrange,
                m.dof_jntid,
                d.qfrc_gravcomp,
                d.qfrc_actuator,
                not (m.opt.disableflags & DisableBit.GRAVITY),
            ],
            outputs=[d.qfrc_actuator],
        )


def _sleep_enabled(m: Model) -> bool:
    sleep_bit = getattr(EnableBit, "SLEEP", 0)
    opt = getattr(m, "opt", None)
    return bool(
        opt is not None
        and getattr(opt, "enableflags", 0) & sleep_bit
        and not (getattr(opt, "disableflags", 0) & DisableBit.ISLAND)
    )


def _solver_compact_enabled(m: Model) -> bool:
    """Match 3.16 solver.solve: SLEEP alone selects constrained compaction."""
    sleep_bit = getattr(EnableBit, "SLEEP", 0)
    return bool(getattr(getattr(m, "opt", None), "enableflags", 0) & sleep_bit)


def _launch_qfrc_smooth(m: Model, d: Data) -> None:
    qfrc_smooth = mjw_forward._qfrc_smooth
    if _has_kernel_args(qfrc_smooth):
        wp.launch(
            qfrc_smooth,
            dim=(d.nworld, m.nv),
            inputs=[d.qfrc_applied, d.qfrc_bias, d.qfrc_passive, d.qfrc_actuator],
            outputs=[d.qfrc_smooth],
        )
        return

    enable_sleep = _sleep_enabled(m)
    wp.launch(
        qfrc_smooth(enable_sleep),
        dim=(d.nworld, m.nv),
        inputs=[
            m.body_treeid,
            m.dof_bodyid,
            d.qfrc_applied,
            d.tree_awake,
            d.qfrc_bias,
            d.qfrc_passive,
            d.qfrc_actuator,
        ],
        outputs=[d.qfrc_smooth],
    )


def _apply_qfrc_actuator_limits(m: Model, d: Data) -> None:
    wp.launch(
        mjw_forward._qfrc_actuator_gravcomp_limits,
        dim=(d.nworld, m.nv),
        inputs=[
            m.jnt_actfrclimited,
            m.jnt_actgravcomp,
            m.jnt_actfrcrange,
            m.dof_jntid,
            d.qfrc_gravcomp,
            d.qfrc_actuator,
            not (m.opt.disableflags & DisableBit.GRAVITY),
        ],
        outputs=[d.qfrc_actuator],
    )


def fwd_acceleration_optimized(
    m: Model,
    d: Data,
    factorize: bool = False,
    *,
    actuator_limits_pending: bool = False,
    dense_factor_solve: BlockedDenseFactorSolveScratch | None = None,
    tree_factor_solve: tree_ldl.TreeFactorSolveScratch | None = None,
) -> None:
    if actuator_limits_pending:
        if not _has_kernel_args(mjw_forward._qfrc_smooth) and _sleep_enabled(m):
            _apply_qfrc_actuator_limits(m, d)
            _launch_qfrc_smooth(m, d)
        else:
            wp.launch(
                _qfrc_actuator_gravcomp_limits_and_qfrc_smooth,
                dim=(d.nworld, m.nv),
                inputs=[
                    m.jnt_actfrclimited,
                    m.jnt_actgravcomp,
                    m.jnt_actfrcrange,
                    m.dof_jntid,
                    d.qfrc_gravcomp,
                    d.qfrc_applied,
                    d.qfrc_bias,
                    d.qfrc_passive,
                    d.qfrc_actuator,
                    not (m.opt.disableflags & DisableBit.GRAVITY),
                ],
                outputs=[d.qfrc_actuator, d.qfrc_smooth],
            )
    else:
        _launch_qfrc_smooth(m, d)
    mjw_forward.xfrc_accumulate(m, d, d.qfrc_smooth)

    if _sleep_enabled(m):
        island.update_active_dofs(m, d)
        solver.smooth_solve_compact(m, d)
        return

    if factorize:
        _factor_solve_i_dense(
            m,
            d,
            _inertia_matrix(d),
            d.qacc_smooth,
            d.qfrc_smooth,
            dense_factor_solve,
            retain_factor=(
                getattr(getattr(m, "opt", None), "solver", None)
                == solver.types.SolverType.CG
            ),
            tree_factor_solve=tree_factor_solve,
        )
    else:
        smooth.solve_m(m, d, d.qacc_smooth, d.qfrc_smooth)


def _normalize_warp_device(device: Any | None) -> str | None:
    if device is None:
        return None

    device_type = getattr(device, "type", None)
    if device_type == "cuda":
        index = getattr(device, "index", None)
        return f"cuda:{0 if index is None else index}"

    name = str(device)
    if name == "cuda":
        return "cuda:0"
    return name


def _warp_array_to_cpu_list(array: Any) -> list[Any]:
    return wp.to_torch(array).detach().cpu().tolist()


def _estimate_max_actuator_moment_rownnz(m: Model) -> int:
    if getattr(m, "nactuator", 0) <= 0:
        return 1

    try:
        trntypes = _warp_array_to_cpu_list(m.actuator_trntype)
        trnids = _warp_array_to_cpu_list(m.actuator_trnid)
        jnt_types = _warp_array_to_cpu_list(m.jnt_type)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return int(getattr(m, "nv", 1))

    max_rownnz = 1
    for actid, trntype in enumerate(trntypes):
        if trntype in (int(TrnType.JOINT), int(TrnType.JOINTINPARENT)):
            jntid = int(trnids[actid][0])
            jnt_type = int(jnt_types[jntid])
            if jnt_type == int(JointType.FREE):
                max_rownnz = max(max_rownnz, 6)
            elif jnt_type == int(JointType.BALL):
                max_rownnz = max(max_rownnz, 3)
            else:
                max_rownnz = max(max_rownnz, 1)
        elif trntype == int(TrnType.TENDON):
            max_rownnz = max(max_rownnz, int(getattr(m, "max_ten_J_rownnz", 1)))
        else:
            return int(getattr(m, "nv", max_rownnz))

    return min(int(getattr(m, "nv", max_rownnz)), max_rownnz)


def make_optimized_fwd_position_streams(
    device: Any | None = None,
) -> OptimizedFwdPositionStreams:
    """Create persistent side streams for optimized MuJoCo Warp position passes."""
    wp_device = _normalize_warp_device(device)
    return tuple(
        wp.Stream(wp_device) for _ in range(NUM_OPTIMIZED_FWD_POSITION_STREAMS)
    )


def _make_collision_narrowphase_call(
    m: Model,
    d: Data,
    wp_device: Any,
) -> Any:
    """Bind upstream convex-narrowphase temporaries to persistent buffers."""
    if not hasattr(m, "opt") or not hasattr(m, "geom_pair_type_count"):
        return None
    collision_table = collision_driver.MJ_COLLISION_TABLE
    if m.opt.disableflags & DisableBit.NATIVECCD:
        collision_table = collision_table.copy()
        collision_table[
            (collision_driver.GeomType.BOX, collision_driver.GeomType.BOX)
        ] = collision_driver.CollisionType.PRIMITIVE
    convex_pairs = [
        pair
        for pair, collision_type in collision_table.items()
        if collision_type == collision_driver.CollisionType.CONVEX
    ]
    convex_globals = getattr(
        collision_driver.convex_narrowphase,
        "__wrapped__",
        collision_driver.convex_narrowphase,
    ).__globals__
    upper_trid_index = convex_globals["upper_trid_index"]
    geom_type_count = len(collision_driver.GeomType)

    def pair_count(first: Any, second: Any) -> int:
        pair_idx = upper_trid_index(
            geom_type_count,
            int(first.value),
            int(second.value),
        )
        return int(m.geom_pair_type_count[pair_idx])

    convex_collision_count = sum(pair_count(*pair) for pair in convex_pairs)
    allocations: list[tuple[str, wp.array]] = []
    if convex_collision_count > 0:
        box = collision_driver.GeomType.BOX
        mesh = collision_driver.GeomType.MESH
        box_box_count = pair_count(box, box)
        if (box, box) not in convex_pairs:
            box_box_count = 0
        box_mesh_count = pair_count(box, mesh)
        mesh_mesh_count = pair_count(mesh, mesh)
        cylinder = collision_driver.GeomType.CYLINDER
        cylinder_count = (
            pair_count(cylinder, cylinder)
            + pair_count(cylinder, box)
            + pair_count(cylinder, mesh)
        )
        epa_iterations = (
            16 if box_box_count == convex_collision_count else int(m.opt.ccd_iterations)
        )
        use_multiccd = not bool(m.opt.disableflags & DisableBit.MULTICCD)
        max_polygon = 4 if box_box_count > 0 else 0
        max_mesh_degree = 3 if box_box_count > 0 else 0
        # upstream collision_convex.convex_narrowphase's multiccd sizing (3.16)
        if use_multiccd and cylinder_count > 0:
            max_polygon = max(int(m.npolygonmax), 16)
            max_mesh_degree = max(int(m.nmeshdegmax), 3)
        elif use_multiccd and mesh_mesh_count + box_mesh_count > 0:
            max_polygon = max(
                int(m.npolygonmax),
                4 if box_mesh_count else max_polygon,
            )
            max_mesh_degree = max(int(m.nmeshdegmax), 3)

        max_epa_faces = int(convex_globals["MJ_MAX_EPAFACES"])
        max_epa_horizon = int(convex_globals["MJ_MAX_EPAHORIZON"])
        naccdmax = int(d.naccdmax)
        allocations.extend(
            [
                (
                    "zeros",
                    wp.empty(
                        geom_type_count * (geom_type_count + 1) // 2,
                        dtype=int,
                        device=wp_device,
                    ),
                ),
                (
                    "empty",
                    wp.empty(
                        (naccdmax, 10 + 2 * epa_iterations),
                        dtype=wp.vec3,
                        device=wp_device,
                    ),
                ),
                (
                    "empty",
                    wp.empty(
                        (naccdmax, 10 + 2 * epa_iterations),
                        dtype=int,
                        device=wp_device,
                    ),
                ),
                (
                    "empty",
                    wp.empty(
                        (naccdmax, 6 + max_epa_faces * epa_iterations),
                        dtype=int,
                        device=wp_device,
                    ),
                ),
                (
                    "empty",
                    wp.empty(
                        (naccdmax, 6 + max_epa_faces * epa_iterations),
                        dtype=wp.vec3,
                        device=wp_device,
                    ),
                ),
                (
                    "empty",
                    wp.empty(
                        (naccdmax, 6 + max_epa_faces * epa_iterations),
                        dtype=float,
                        device=wp_device,
                    ),
                ),
                (
                    "empty",
                    wp.empty(
                        (naccdmax, max_epa_horizon),
                        dtype=int,
                        device=wp_device,
                    ),
                ),
            ]
        )
        allocations.extend(
            (
                "empty",
                wp.empty(shape, dtype=dtype, device=wp_device),
            )
            for shape, dtype in (
                ((naccdmax, 2 * max_polygon), wp.vec3),
                ((naccdmax, 2 * max_polygon), wp.vec3),
                ((naccdmax, max_polygon), wp.vec3),
                ((naccdmax, max_polygon), float),
                ((naccdmax, max_mesh_degree), int),
                ((naccdmax, max_mesh_degree), int),
                ((naccdmax, max_mesh_degree), wp.vec3),
                ((naccdmax, max_mesh_degree), wp.vec3),
                ((naccdmax, max_mesh_degree), wp.vec3),
                ((naccdmax, max_polygon), wp.vec3),
                ((naccdmax, max_polygon), wp.vec3),
            )
        )

    convex_narrowphase_call = _PreallocatedWarpCall(
        collision_driver.convex_narrowphase,
        allocations,
    )
    return _PreallocatedWarpCall(
        collision_driver._narrowphase,
        (),
        global_overrides={"convex_narrowphase": convex_narrowphase_call},
    )


@wp.func
def _vec2i_order(a: wp.vec2i, b: wp.vec2i) -> int:
    """-1, 0 or 1 as ``a`` is lexicographically before, equal to or after ``b``."""
    for k in range(2):
        if a[k] < b[k]:
            return -1
        if a[k] > b[k]:
            return 1
    return 0


@wp.func
def _flex_candidate_yields(
    i: int,
    j: int,
    cand_dist: wp.array[float],
    cand_geom: wp.array[wp.vec2i],
    cand_flex: wp.array[wp.vec2i],
    cand_elem: wp.array[wp.vec2i],
    cand_vert: wp.array[wp.vec2i],
) -> bool:
    """Whether flex candidate ``i`` yields to its near duplicate ``j``.

    The deeper candidate wins. An equal depth is broken by the candidates'
    element, vertex, flex and geom ids, which name the narrowphase thread that
    wrote them. The slot decides only between candidates with the same ids: one
    thread wrote those in sequence, so their slot order is fixed.
    """
    dist_i = cand_dist[i]
    dist_j = cand_dist[j]
    if dist_j != dist_i:
        return dist_j < dist_i
    order = _vec2i_order(cand_elem[j], cand_elem[i])
    if order == 0:
        order = _vec2i_order(cand_vert[j], cand_vert[i])
    if order == 0:
        order = _vec2i_order(cand_flex[j], cand_flex[i])
    if order == 0:
        order = _vec2i_order(cand_geom[j], cand_geom[i])
    if order == 0:
        return j < i
    return order < 0


@wp.kernel(enable_backward=False)
def _filter_flex_candidates_by_identity(
    # In:
    ncand: wp.array[int],
    epsilon: float,
    sort_key: wp.array[wp.int64],
    sort_val: wp.array[int],
    cand_dist: wp.array[float],
    cand_pos: wp.array[wp.vec3],
    cand_geom: wp.array[wp.vec2i],
    cand_flex: wp.array[wp.vec2i],
    cand_elem: wp.array[wp.vec2i],
    cand_vert: wp.array[wp.vec2i],
    # Out:
    cand_active_out: wp.array[int],
):
    """MuJoCo Warp 3.16 ``_filter_flex_candidates_sorted`` with the tie between
    equally deep near duplicates broken by :func:`_flex_candidate_yields`.

    Upstream breaks it by candidate slot, which the narrowphase threads take with
    an atomic counter, so which duplicate survives depended on thread scheduling.
    """
    si = wp.tid()
    ncand_limit = wp.min(ncand[0], cand_active_out.shape[0])
    if si >= ncand_limit:
        if si < cand_active_out.shape[0]:
            cand_active_out[sort_val[si]] = 0
        return

    i = sort_val[si]
    my_key = sort_key[si]
    my_group = my_key >> wp.int64(32)
    my_spatial = my_key & wp.int64(0x7FFFFFFF)
    eps_key = wp.int64(wp.ceil(epsilon * 1000000.0)) + wp.int64(1)
    pos_i = cand_pos[i]
    eps2 = epsilon * epsilon

    keep = int(1)

    j = si - 1
    while j >= 0:
        key_j = sort_key[j]
        if (key_j >> wp.int64(32)) != my_group:
            break
        spatial_j = key_j & wp.int64(0x7FFFFFFF)
        if my_spatial - spatial_j >= eps_key:
            break
        oj = sort_val[j]
        diff = pos_i - cand_pos[oj]
        if wp.dot(diff, diff) < eps2:
            if _flex_candidate_yields(
                i, oj, cand_dist, cand_geom, cand_flex, cand_elem, cand_vert
            ):
                keep = 0
                break
        j = j - 1

    if keep == 1:
        j = si + 1
        while j < ncand_limit:
            key_j = sort_key[j]
            if (key_j >> wp.int64(32)) != my_group:
                break
            spatial_j = key_j & wp.int64(0x7FFFFFFF)
            if spatial_j - my_spatial >= eps_key:
                break
            oj = sort_val[j]
            diff = pos_i - cand_pos[oj]
            if wp.dot(diff, diff) < eps2:
                if _flex_candidate_yields(
                    i, oj, cand_dist, cand_geom, cand_flex, cand_elem, cand_vert
                ):
                    keep = 0
                    break
            j = j + 1

    cand_active_out[i] = keep


class _FlexContactFilterCall:
    """MuJoCo Warp's flex candidate filter and contact write, independent of
    candidate slot order.

    Upstream drops a candidate when a deeper one lies within 1 mm in its group
    and, at equal depth, keeps the one in the lower slot. Narrowphase threads take
    slots from an atomic counter, so which of two equally deep near duplicates
    survived (with its position and normal) changed with thread scheduling, e.g.
    between eager launches and a CUDA graph replay. This call runs upstream's
    ``_filter_and_write_contacts`` with that filter launch replaced by
    :func:`_filter_flex_candidates_by_identity`; the set of contacts written is
    then a function of the candidates, not of their slots (their rows in
    ``d.contact`` still follow the atomic contact counter). It allocates nothing.
    """

    def __init__(self) -> None:
        self._workspace: Any | None = None
        self._call = _PreallocatedWarpCall(
            collision_flex._filter_and_write_contacts, ()
        )
        self._call.add_launch_substitutions(
            {collision_flex._filter_flex_candidates_sorted: self._filter_by_identity}
        )

    def __call__(self, m: Model, d: Data, ws: Any, enable_fps: bool = False) -> None:
        self._workspace = ws
        try:
            self._call(m, d, ws, enable_fps)
        finally:
            self._workspace = None

    def _filter_by_identity(self, launch: Any, _kernel: Any, kwargs: dict) -> Any:
        ws = self._workspace
        kwargs = dict(kwargs)
        if len(kwargs["inputs"]) != 6:
            raise RuntimeError("MuJoCo Warp flex candidate filter launch changed")
        kwargs["inputs"] = [*kwargs["inputs"], ws.geom, ws.flex, ws.elem, ws.vert]
        return launch(_filter_flex_candidates_by_identity, **kwargs)


def _make_flex_collision_call(
    m: Model,
    d: Data,
    wp_device: Any,
) -> Any | None:
    """Bind MuJoCo Warp 3.16 flex-collision temporaries to stable buffers.

    Both contact-writing passes (flex-geom and flex self/flex-flex) filter their
    candidates with :class:`_FlexContactFilterCall`, so the contacts do not
    depend on the order in which narrowphase threads took candidate slots.
    """
    if getattr(m, "nflex", 0) == 0 or getattr(m, "nflexelem", 0) == 0:
        return None

    with wp.ScopedDevice(wp_device):
        workspace_call = _PreallocatedWarpCall.from_allocator(
            collision_flex._allocate_flex_workspace, m, d
        )
    sap_call = _PreallocatedWarpCall(
        collision_flex._run_flex_sap_sort,
        tuple(
            ("empty", wp.empty(shape, dtype=dtype, device=wp_device))
            for shape, dtype in (
                ((d.nworld, m.nflexelem, 2), float),
                ((d.nworld, m.nflexelem), float),
                ((d.nworld, m.nflexelem, 2), int),
                ((d.nworld, m.nflexelem), int),
                ((d.nworld, m.nflexelem), int),
                (d.nworld + 1, int),
                ((d.nworld, m.nflexelem), wp.vec3),
                ((d.nworld, m.nflexelem), wp.vec3),
            )
        ),
    )
    narrow_call = _PreallocatedWarpCall(
        collision_flex._run_flex_narrowphase,
        (("empty", wp.empty(d.naconmax * 8, dtype=wp.vec3, device=wp_device)),),
    )
    filter_call = _FlexContactFilterCall()
    geom_collision = _PreallocatedWarpCall(
        collision_flex._flex_geom_collision,
        (),
        global_overrides={"_filter_and_write_contacts": filter_call},
    )
    sap_collision = _PreallocatedWarpCall(
        collision_flex._flex_sap_collision,
        (),
        global_overrides={
            "_run_flex_narrowphase": narrow_call,
            "_filter_and_write_contacts": filter_call,
        },
    )
    # Upstream runs the flex-flex SAP pass (sort, candidates, FPS filter) whenever
    # nflex > 1. Skip it, and the sort when no flex self-collides, if no two
    # flexes can collide by contype/conaffinity (the pass would find nothing).
    contype = m.flex_contype.numpy()
    conaff = m.flex_conaffinity.numpy()
    pairs = any(
        (contype[f] & conaff[h]) or (contype[h] & conaff[f])
        for f in range(int(m.nflex))
        for h in range(f + 1, int(m.nflex))
    )
    selfcollide = bool(getattr(m, "has_flex_selfcollide", False))

    def sap_collision_if_possible(m, d, ctx, ws, is_self, **kwargs):
        if not is_self and not pairs:
            return None
        return sap_collision(m, d, ctx, ws, is_self, **kwargs)

    def sap_sort_if_needed(m, d):
        if not pairs and not selfcollide:
            return None
        return sap_call(m, d)

    return _PreallocatedWarpCall(
        collision_flex.flex_collision,
        (),
        global_overrides={
            "_allocate_flex_workspace": workspace_call,
            "_flex_geom_collision": geom_collision,
            "_run_flex_sap_sort": sap_sort_if_needed,
            "_flex_sap_collision": sap_collision_if_possible,
        },
    )


def make_optimized_fwd_position_scratch(
    m: Model,
    d: Data,
    device: Any | None = None,
    *,
    include_optimized_kernel_scratch: bool = True,
    ball_joint_limits: Sequence[Any] = (),
) -> OptimizedFwdPositionScratch:
    """Allocate persistent fwd-position scratch outside CUDA graph capture."""
    if getattr(getattr(m, "opt", None), "integrator", None) == IntegratorType.DISCRETE:
        # the monolithic solve has no DISCRETE smooth-force path and _integrate none
        raise ValueError("the persistent step path does not support DISCRETE")
    wp_device = _normalize_warp_device(device)
    if wp_device is None:
        wp_device = getattr(getattr(d, "qpos", None), "device", None)

    collision_context = CollisionContext(
        collision_pair=wp.empty(d.naconmax, dtype=wp.vec2i, device=wp_device),
        collision_pairid=wp.empty(d.naconmax, dtype=wp.vec2i, device=wp_device),
        collision_worldid=wp.empty(d.naconmax, dtype=int, device=wp_device),
    )
    collision_narrowphase_call = _make_collision_narrowphase_call(
        m,
        d,
        wp_device,
    )
    flex_collision_call = _make_flex_collision_call(m, d, wp_device)
    collision_broadphase_call = None
    collision_awake_prev = None
    collision_incremental_broadphase_call = None
    opt = getattr(m, "opt", None)
    broadphase = getattr(opt, "broadphase", None)
    if broadphase in (BroadphaseType.SAP_TILE, BroadphaseType.SAP_SEGMENTED):
        ngeom = int(getattr(m, "ngeom", 0))
        sap_allocations = tuple(
            ("empty", wp.empty(shape, dtype=dtype, device=wp_device))
            for shape, dtype in (
                ((d.nworld, ngeom, 2), float),
                ((d.nworld, ngeom), float),
                ((d.nworld, ngeom, 2), int),
                ((d.nworld, ngeom), int),
                ((d.nworld, ngeom), int),
                (
                    d.nworld + 1 if broadphase == BroadphaseType.SAP_SEGMENTED else 0,
                    int,
                ),
            )
        )
        collision_broadphase_call = _PreallocatedWarpCall(
            collision_driver.sap_broadphase,
            sap_allocations,
        )
    sleep_enabled = bool(
        opt is not None
        and not (getattr(opt, "disableflags", 0) & DisableBit.ISLAND)
        and (getattr(opt, "enableflags", 0) & getattr(EnableBit, "SLEEP", 0))
    )
    if sleep_enabled and getattr(d, "body_awake", None) is not None:
        collision_awake_prev = wp.empty_like(d.body_awake)
        if broadphase == BroadphaseType.NXN and getattr(
            opt, "graph_conditional", False
        ):
            collision_incremental_broadphase_call = _PreallocatedWarpCall(
                collision_driver.nxn_broadphase,
                (("zeros", wp.zeros(1, dtype=int, device=wp_device)),),
            )
    transmission_body_ncon = None
    if getattr(m, "nacttrnbody", 0):
        transmission_body_ncon = wp.zeros(
            (d.nworld, m.nacttrnbody),
            dtype=int,
            device=wp_device,
        )
    # These buffers replace direct allocations in upstream smooth.tendon_bias().
    # They are part of the allocation-free runtime contract, not an optional
    # performance optimization.
    tendon_bias_Jdot = None
    tendon_bias_coef = None
    if getattr(m, "ntendon", 0):
        tendon_bias_Jdot = wp.zeros(
            (d.nworld, getattr(m, "nJten", 0)),
            dtype=float,
            device=wp_device,
        )
        tendon_bias_coef = wp.zeros(
            (d.nworld, m.ntendon),
            dtype=float,
            device=wp_device,
        )

    passive_allocations: list[tuple[str, wp.array]] = []
    passive_disabled = bool(
        getattr(getattr(m, "opt", None), "disableflags", 0) & DisableBit.SPRING
        and getattr(getattr(m, "opt", None), "disableflags", 0) & DisableBit.DAMPER
    )
    if not passive_disabled and getattr(m, "nflex", 0) > 0:
        passive_allocations.extend(
            (
                (
                    "zeros",
                    wp.zeros(
                        (d.nworld, m.nbody),
                        dtype=wp.spatial_vector,
                        device=wp_device,
                    ),
                ),
                (
                    "zeros",
                    wp.zeros(
                        (d.nworld, m.nbody),
                        dtype=wp.spatial_vector,
                        device=wp_device,
                    ),
                ),
            )
        )
    if (
        not passive_disabled
        and getattr(m, "nflex", 0)
        and getattr(m, "nflexintcell", 0) > 0
    ):
        passive_allocations.extend(
            (
                (
                    "empty",
                    wp.empty(
                        (d.nworld, m.nflexintcell, 27),
                        dtype=wp.vec3,
                        device=wp_device,
                    ),
                ),
                (
                    "empty",
                    wp.empty(
                        (d.nworld, m.nflexintcell, 27),
                        dtype=wp.vec3,
                        device=wp_device,
                    ),
                ),
            )
        )
    passive_globals: dict[str, Any] = {}
    if not passive_disabled and getattr(m, "has_fluid", False):
        fluid_applied = wp.empty(
            (d.nworld, m.nbody),
            dtype=wp.spatial_vector,
            device=wp_device,
        )
        passive_globals["_fluid"] = _PreallocatedWarpCall(
            passive._fluid,
            (("empty", fluid_applied),),
        )
    passive_call = _PreallocatedWarpCall(
        passive.passive,
        passive_allocations,
        global_overrides=passive_globals,
        skipped_launches=_skipped_passive_launches(m),
    )

    solver_uses_compact = _solver_compact_enabled(m)
    island_call = None
    island_mapping_call = None
    if sleep_enabled and getattr(m, "ntree", 0) > 0:
        island_call = _PreallocatedWarpCall(
            island.island,
            (("empty", wp.empty((d.nworld, m.ntree), dtype=int, device=wp_device)),),
        )
    if solver_uses_compact and getattr(m, "ntree", 1) > 1:
        # Upstream lazily resizes these views; do so before any capture.
        for name, width in (
            ("dof_islandid", m.nv),
            ("efc_islandid", d.njmax),
            ("island_idofadr", m.ntree),
        ):
            if getattr(d, name).shape[1] != width:
                setattr(
                    d, name, wp.empty((d.nworld, width), dtype=int, device=wp_device)
                )
        island_mapping_call = _PreallocatedWarpCall(
            island.compute_island_mapping,
            (("empty", wp.empty((d.nworld, d.njmax), dtype=int, device=wp_device)),),
        )

    advance_call = None
    if sleep_enabled:
        island_can_sleep = wp.ones(
            (d.nworld, m.ntree),
            dtype=int,
            device=wp_device,
        )
        sleep_call = _PreallocatedWarpCall(
            sleep.sleep,
            (("ones", island_can_sleep),),
        )
        velocity_smooth_overrides: dict[str, Any] = {}
        if getattr(m, "ntendon", 0):
            velocity_smooth_overrides["tendon_bias"] = _PreallocatedWarpCall(
                smooth.tendon_bias,
                (
                    ("zeros", tendon_bias_Jdot),
                    ("zeros", tendon_bias_coef),
                ),
            )
        sleep_fwd_velocity_call = _PreallocatedWarpCall(
            mjw_forward.fwd_velocity,
            (),
            global_overrides={
                "passive": _AttributeOverrides(
                    passive,
                    {"passive": passive_call},
                ),
                "smooth": _AttributeOverrides(smooth, velocity_smooth_overrides),
            },
        )
        advance_call = _PreallocatedWarpCall(
            mjw_forward._advance,
            (),
            global_overrides={
                "sleep": _AttributeOverrides(sleep, {"sleep": sleep_call}),
                "fwd_velocity": sleep_fwd_velocity_call,
            },
        )
    implicit_actuator_vel = None
    implicit_qderiv_pair_mask = None
    implicit_qderiv = None
    implicit_qacc = None
    implicit_max_moment_rownnz = 0
    implicit_qld = None
    implicit_ql_diag_inv = None
    rk4_scratch = None
    euler_call = None
    implicit_call = None
    implicit_deriv_call = None
    integrator = getattr(getattr(m, "opt", None), "integrator", None)
    if getattr(m, "nv", 0) > 0 and integrator == IntegratorType.RK4:
        act_t0 = None
        act_dot_rk = None
        if getattr(m, "na", 0):
            act_t0 = wp.empty(d.act.shape, dtype=d.act.dtype, device=wp_device)
            act_dot_rk = wp.zeros(
                (d.nworld, m.na),
                dtype=float,
                device=wp_device,
            )
        rk4_scratch = RK4Scratch(
            qpos_t0=wp.empty(d.qpos.shape, dtype=d.qpos.dtype, device=wp_device),
            qvel_t0=wp.empty(d.qvel.shape, dtype=d.qvel.dtype, device=wp_device),
            qvel_rk=wp.zeros(
                (d.nworld, m.nv),
                dtype=float,
                device=wp_device,
            ),
            qacc_rk=wp.zeros(
                (d.nworld, m.nv),
                dtype=float,
                device=wp_device,
            ),
            act_t0=act_t0,
            act_dot_rk=act_dot_rk,
            time_t0=wp.empty(d.time.shape, dtype=d.time.dtype, device=wp_device),
        )
    if (
        getattr(m, "nv", 0) > 0
        and integrator == IntegratorType.EULER
        and _euler_damping_needs_scratch(m)
    ):
        euler_call = _PreallocatedWarpCall(
            mjw_forward.euler,
            (
                (
                    "empty",
                    wp.empty((d.nworld, m.nv), dtype=float, device=wp_device),
                ),
                (
                    "empty",
                    wp.empty((d.nworld, m.nv), dtype=float, device=wp_device),
                ),
                (
                    "clone",
                    wp.empty(d.M.shape, dtype=d.M.dtype, device=wp_device),
                ),
                (
                    "empty_like",
                    wp.empty(d.qLD.shape, dtype=d.qLD.dtype, device=wp_device),
                ),
                (
                    "empty",
                    wp.empty((d.nworld, m.nv), dtype=float, device=wp_device),
                ),
            ),
            global_overrides={"_advance": advance_call or mjw_forward._advance},
        )
    if getattr(m, "nv", 0) > 0 and integrator in (
        IntegratorType.IMPLICITFAST,
        IntegratorType.IMPLICIT,
    ):
        implicit_qacc = wp.empty((d.nworld, m.nv), dtype=float, device=wp_device)
        implicit_allocations: list[tuple[str, wp.array]] = []
        if integrator == IntegratorType.IMPLICIT:
            implicit_qderiv = wp.empty(
                _inertia_matrix(d).shape,
                dtype=float,
                device=wp_device,
            )
            implicit_allocations.extend(
                (("empty", implicit_qderiv), ("empty", implicit_qacc))
            )
        elif integrator == IntegratorType.IMPLICITFAST:
            implicit_qderiv = wp.empty(
                (d.nworld, m.nC),
                dtype=float,
                device=wp_device,
            )
            implicit_qld = wp.empty(
                d.qLD.shape,
                dtype=d.qLD.dtype,
                device=wp_device,
            )
            implicit_ql_diag_inv = wp.empty(
                (d.nworld, m.nv),
                dtype=float,
                device=wp_device,
            )
            implicit_allocations.extend(
                (
                    ("empty", implicit_qderiv),
                    ("empty_like", implicit_qld),
                    ("empty", implicit_ql_diag_inv),
                    ("empty", implicit_qacc),
                )
            )
        if implicit_allocations:
            derivative_overrides: dict[str, Any] = {
                "deriv_smooth_vel": deriv_smooth_vel
            }
            if _deriv_smooth_vel_allocates_actuator_velocity(m):
                implicit_actuator_vel = wp.empty(
                    (d.nworld, m.nactuator),
                    dtype=float,
                    device=wp_device,
                )
                derivative_overrides["deriv_smooth_vel"] = _PreallocatedWarpCall(
                    deriv_smooth_vel,
                    (("empty", implicit_actuator_vel),),
                )
            if integrator == IntegratorType.IMPLICITFAST:
                implicit_deriv_call = derivative_overrides["deriv_smooth_vel"]
            if integrator == IntegratorType.IMPLICIT:
                rne_shape = (d.nworld, m.nbody, m.nv)
                derivative_overrides["deriv_rne_vel"] = _PreallocatedWarpCall(
                    derivative.deriv_rne_vel,
                    (
                        (
                            "zeros",
                            wp.empty(
                                rne_shape,
                                dtype=wp.spatial_vector,
                                device=wp_device,
                            ),
                        ),
                        (
                            "zeros",
                            wp.empty(
                                (d.nworld, m.nv, m.nv),
                                dtype=wp.spatial_vector,
                                device=wp_device,
                            ),
                        ),
                        (
                            "zeros",
                            wp.empty(
                                rne_shape,
                                dtype=wp.spatial_vector,
                                device=wp_device,
                            ),
                        ),
                        (
                            "zeros",
                            wp.empty(
                                rne_shape,
                                dtype=wp.spatial_vector,
                                device=wp_device,
                            ),
                        ),
                    ),
                )
            implicit_call = _PreallocatedWarpCall(
                mjw_forward.implicit,
                implicit_allocations,
                global_overrides={
                    "_advance": advance_call or mjw_forward._advance,
                    "derivative": _AttributeOverrides(
                        derivative,
                        derivative_overrides,
                    ),
                },
            )

    if (
        include_optimized_kernel_scratch
        and getattr(m, "nactuator", 0) > 0
        and getattr(m, "nv", 0) > 0
    ):
        if implicit_actuator_vel is None:
            implicit_actuator_vel = wp.empty(
                (d.nworld, m.nactuator),
                dtype=float,
                device=wp_device,
            )
        implicit_qderiv_pair_mask = wp.zeros(
            (m.nv, m.nv),
            dtype=int,
            device=wp_device,
        )
        fullm_i, fullm_j = _fullm_indices(m)
        wp.launch(
            _mark_dense_qderiv_pair_mask,
            dim=fullm_i.shape[0],
            inputs=[fullm_i, fullm_j],
            outputs=[implicit_qderiv_pair_mask],
        )
        implicit_max_moment_rownnz = _estimate_max_actuator_moment_rownnz(m)
        if implicit_qderiv is None:
            implicit_qderiv = wp.empty(
                _inertia_matrix(d).shape,
                dtype=float,
                device=wp_device,
            )
        if implicit_qacc is None:
            implicit_qacc = wp.empty(
                (d.nworld, m.nv),
                dtype=float,
                device=wp_device,
            )

    constraint_efc_nnz = wp.empty((d.nworld,), dtype=int, device=wp_device)
    constraint_call = _PreallocatedWarpCall(
        constraint.make_constraint,
        (("empty", constraint_efc_nnz),),
    )
    ball_limit_buffers = None
    if ball_joint_limits:
        from instantmimic.utils.mjw_kernels.ball_limits import make_ball_limit_buffers

        ball_limit_buffers = make_ball_limit_buffers(ball_joint_limits, m, d, wp_device)
        constraint_call = _BallLimitConstraintCall(
            m, d, constraint_efc_nnz, ball_limit_buffers
        )

    constraint_free = _constraint_rows_impossible(m, ball_joint_limits)
    mul_m_call = _PreallocatedWarpCall(
        support.mul_m,
        (("empty", wp.empty(0, dtype=bool, device=wp_device)),),
    )

    tendon_wrap_geom_xpos = None
    tendon_allocations: list[tuple[str, wp.array]] = []
    if int(getattr(m, "ntendon", 0)) > 0:
        tendon_wrap_geom_xpos = wp.empty(
            (d.nworld, int(getattr(m, "nwrap", 0))),
            dtype=wp.spatial_vector,
            device=wp_device,
        )
        tendon_allocations.append(("empty", tendon_wrap_geom_xpos))
    tendon_call = _PreallocatedWarpCall(
        smooth.tendon,
        tendon_allocations,
    )

    actuation_delayed_ctrl = None
    actuation_tendon_force = None
    actuation_allocations: list[tuple[str, wp.array]] = []
    actuation_enabled = bool(
        getattr(m, "nactuator", 0)
        and not (
            getattr(getattr(m, "opt", None), "disableflags", 0) & DisableBit.ACTUATION
        )
    )
    if actuation_enabled and int(getattr(m, "nhistory", 0)) > 0:
        actuation_delayed_ctrl = wp.empty(
            (d.nworld, m.nu),
            dtype=float,
            device=wp_device,
        )
        actuation_allocations.append(("empty", actuation_delayed_ctrl))
    if actuation_enabled and int(getattr(m, "ntendon", 0)) > 0:
        actuation_tendon_force = wp.empty(
            (d.nworld, m.ntendon),
            dtype=float,
            device=wp_device,
        )
        actuation_allocations.append(("zeros", actuation_tendon_force))
    actuation_call = _PreallocatedWarpCall(
        mjw_forward.fwd_actuation,
        actuation_allocations,
    )

    sensors_enabled = not bool(
        getattr(getattr(m, "opt", None), "disableflags", 0) & DisableBit.SENSOR
    )
    sensor_delay_call = None
    if sensors_enabled and int(getattr(m, "nhistory", 0)) > 0:
        sensor_delay_call = _PreallocatedSensorDelayCall(wp.empty_like(d.sensordata))
    sensor_history_globals = (
        {
            "history": _AttributeOverrides(
                history,
                {"apply_sensor_delay": sensor_delay_call},
            )
        }
        if sensor_delay_call is not None
        else {}
    )

    rangefinder_count = int(getattr(m, "nrangefinder", 0))
    rangefinder_dist = wp.empty(
        (d.nworld, rangefinder_count), dtype=float, device=wp_device
    )
    sensor_allocations: list[tuple[str, wp.array]] = [("empty", rangefinder_dist)]
    sensor_rangefinder_adr = getattr(m, "sensor_rangefinder_adr", None)
    if int(getattr(sensor_rangefinder_adr, "size", 0)) > 0:
        sensor_allocations.extend(
            (
                (
                    "empty",
                    wp.empty(
                        (d.nworld, rangefinder_count),
                        dtype=wp.vec3,
                        device=wp_device,
                    ),
                ),
                (
                    "empty",
                    wp.empty(
                        (d.nworld, rangefinder_count),
                        dtype=wp.vec3,
                        device=wp_device,
                    ),
                ),
                (
                    "empty",
                    wp.empty(
                        (d.nworld, rangefinder_count),
                        dtype=int,
                        device=wp_device,
                    ),
                ),
                (
                    "empty",
                    wp.empty(
                        (d.nworld, rangefinder_count),
                        dtype=wp.vec3,
                        device=wp_device,
                    ),
                ),
            )
        )
    sensor_collision = wp.empty(
        (d.nworld, int(getattr(m, "nsensorcollision", 0)), 8, 7),
        dtype=float,
        device=wp_device,
    )
    sensor_allocations.append(("full", sensor_collision))
    if not sensors_enabled:
        sensor_allocations.clear()
    sensor_globals: dict[str, Any] = dict(sensor_history_globals)
    sensor_globals["_eval_sensor_contact"] = _make_contact_sensor_call(
        m, d, m.sensor_contact_pos_adr, wp_device
    )
    energy_vel_call = None
    energy_enabled = bool(
        getattr(getattr(m, "opt", None), "enableflags", 0) & EnableBit.ENERGY
    )
    if bool(getattr(m, "sensor_e_kinetic", False)) or energy_enabled:
        energy_mv = wp.empty((d.nworld, m.nv), dtype=float, device=wp_device)
        energy_vel_call = _PreallocatedWarpCall(
            sensor.energy_vel,
            (("zeros", energy_mv),),
        )
        sensor_globals["energy_vel"] = energy_vel_call
    sensor_pos_call = _PreallocatedWarpCall(
        sensor.sensor_pos,
        sensor_allocations,
        global_overrides=sensor_globals,
    )
    sensor_subtree_bodyvel = None
    sensor_vel_globals: dict[str, Any] = dict(sensor_history_globals)
    if bool(getattr(m, "sensor_subtree_vel", False)):
        sensor_subtree_bodyvel = wp.empty(
            (d.nworld, m.nbody),
            dtype=wp.spatial_vector,
            device=wp_device,
        )
        sensor_vel_globals["smooth"] = _AttributeOverrides(
            smooth,
            {
                "subtree_vel": _PreallocatedWarpCall(
                    smooth.subtree_vel,
                    (("empty", sensor_subtree_bodyvel),),
                )
            },
        )
    tactile_allocations = ()
    if sensors_enabled and m.nsensortaxel:
        tactile_allocations = (
            (
                "zeros",
                wp.empty((d.nworld, m.ntactileweld), dtype=int, device=wp_device),
            ),
            (
                "full",
                wp.empty(
                    (d.nworld, m.ntactileweld, sensor.MJ_MAXCONPAIR),
                    dtype=int,
                    device=wp_device,
                ),
            ),
            (
                "zeros",
                wp.empty(
                    (d.nworld, m.ntactileweld, (m.ngeom + 31) // 32),
                    dtype=wp.uint32,
                    device=wp_device,
                ),
            ),
        )
    sensor_vel_call = _PreallocatedWarpCall(
        sensor.sensor_vel,
        tactile_allocations,
        global_overrides=sensor_vel_globals,
    )
    sensor_acc_globals: dict[str, Any] = dict(sensor_history_globals)
    sensor_acc_globals["_eval_sensor_contact"] = _make_contact_sensor_call(
        m, d, m.sensor_contact_acc_adr, wp_device
    )
    rne_postconstraint_call = None
    if m.opt.run_rne_postconstraint or (
        sensors_enabled and m.sensor_rne_postconstraint
    ):
        rne_allocations = []
        if m.neq:
            rne_allocations.extend(
                ("zeros", wp.empty(d.nworld, dtype=int, device=wp_device))
                for _ in range(2)
            )
        if m.ntendon:
            rne_allocations.extend(
                (
                    "zeros",
                    wp.empty((d.nworld, m.ntendon), dtype=float, device=wp_device),
                )
                for _ in range(2)
            )
            if m.nJten:
                rne_allocations.append(
                    (
                        "zeros",
                        wp.empty((d.nworld, m.nJten), dtype=float, device=wp_device),
                    )
                )
        rne_postconstraint_call = _PreallocatedWarpCall(
            smooth.rne_postconstraint, rne_allocations
        )
        sensor_acc_globals["smooth"] = _AttributeOverrides(
            smooth, {"rne_postconstraint": rne_postconstraint_call}
        )
    sensor_acc_call = _PreallocatedWarpCall(
        sensor.sensor_acc,
        (),
        global_overrides=sensor_acc_globals,
    )

    solver_context = None
    solver_nsolving = None
    solver_compact_model = None
    solver_compact_data = None
    if getattr(d, "njmax", 0) != 0 and m.nv != 0:
        if solver_uses_compact:
            nvp = d.nvmax_pad
            graph_conditional = bool(
                m.opt.graph_conditional and wp.get_device(wp_device).is_cuda
            )
            compact_opt = dataclasses.replace(
                m.opt,
                graph_conditional=graph_conditional,
                tolerance=d.ctol,
                ls_tolerance=d.cls_tol,
            )
            solver_compact_model = dataclasses.replace(
                m,
                opt=compact_opt,
                nv=nvp,
                nv_pad=nvp,
                is_sparse=False,
                dof_tri_row=d.cdof_tri_row,
                dof_tri_col=d.cdof_tri_col,
            )
            compact_efc = dataclasses.replace(d.efc, J=d.cJ, Ma=d.cMa)
            solver_compact_data = dataclasses.replace(
                d,
                M=d.cM,
                qfrc_smooth=d.cqfrc_smooth,
                qacc_smooth=d.cqacc_smooth,
                qacc_warmstart=d.cqacc_warmstart,
                qacc=d.cqacc,
                qfrc_constraint=d.cqfrc_constraint,
                efc=compact_efc,
            )
            solver_context = solver._create_solver_context(
                solver_compact_model,
                solver_compact_data,
            )
            solver_context.compact_m_full = m
            solver_context.compact_d_full = d
        else:
            solver_context = solver._create_solver_context(m, d)
            newton_fusion.attach_scratch(m, d, solver_context, wp_device)
        solver_nsolving = wp.empty((1,), dtype=int, device=wp_device)

    dense_factor_solve = create_blocked_dense_factor_solve_scratch(m, d, wp_device)

    return OptimizedFwdPositionScratch(
        collision_context=collision_context,
        transmission_moment_nnz=wp.zeros((d.nworld,), dtype=int, device=wp_device),
        collision_narrowphase_call=collision_narrowphase_call,
        collision_broadphase_call=collision_broadphase_call,
        flex_collision_call=flex_collision_call,
        transmission_body_ncon=transmission_body_ncon,
        tendon_wrap_geom_xpos=tendon_wrap_geom_xpos,
        tendon_call=tendon_call,
        tendon_bias_Jdot=tendon_bias_Jdot,
        tendon_bias_coef=tendon_bias_coef,
        implicit_actuator_vel=implicit_actuator_vel,
        implicit_qderiv_pair_mask=implicit_qderiv_pair_mask,
        implicit_qderiv=implicit_qderiv,
        implicit_qacc=implicit_qacc,
        implicit_max_moment_rownnz=implicit_max_moment_rownnz,
        collision_awake_prev=collision_awake_prev,
        collision_incremental_broadphase_call=collision_incremental_broadphase_call,
        solver_context=solver_context,
        solver_nsolving=solver_nsolving,
        solver_uses_compact=solver_uses_compact,
        solver_compact_model=solver_compact_model,
        solver_compact_data=solver_compact_data,
        constraint_call=constraint_call,
        ball_limit_buffers=ball_limit_buffers,
        constraint_free=constraint_free,
        mul_m_call=mul_m_call,
        actuation_delayed_ctrl=actuation_delayed_ctrl,
        actuation_tendon_force=actuation_tendon_force,
        actuation_call=actuation_call,
        sensor_pos_call=sensor_pos_call,
        sensor_subtree_bodyvel=sensor_subtree_bodyvel,
        sensor_vel_call=sensor_vel_call,
        sensor_acc_call=sensor_acc_call,
        rne_postconstraint_call=rne_postconstraint_call,
        energy_vel_call=energy_vel_call,
        passive_call=passive_call,
        advance_call=advance_call,
        island_call=island_call,
        island_mapping_call=island_mapping_call,
        rk4=rk4_scratch,
        euler_call=euler_call,
        implicit_call=implicit_call,
        implicit_deriv_call=implicit_deriv_call,
        implicit_free_body_call=(
            _make_implicit_free_body_call(implicit_qderiv)
            if integrator == IntegratorType.IMPLICITFAST and implicit_qderiv is not None
            else None
        ),
        implicit_qld=implicit_qld,
        implicit_ql_diag_inv=implicit_ql_diag_inv,
        dense_factor_solve=dense_factor_solve,
        tree_reduction=(
            tree_fusion.create_tree_reduction_scratch(m)
            if include_optimized_kernel_scratch
            else None
        ),
        tree_factor_solve=(
            tree_ldl.create_tree_factor_solve_scratch(m)
            if include_optimized_kernel_scratch
            else None
        ),
    )


def _solve_monolithic_with_scratch(
    m: Model,
    d: Data,
    ctx: Any,
    nsolving: wp.array,
    *,
    compact: bool = False,
    use_optimized_kernel: bool = False,
    mul_m_call: Any | None = None,
    unroll_solver_iterations: bool = False,
) -> None:
    warmstart = not (m.opt.disableflags & DisableBit.WARMSTART)
    wp.launch(
        solver._solve_init_dof(warmstart, m.is_sparse),
        dim=(d.nworld, m.nv),
        inputs=[d.nefc, d.qacc_warmstart, d.qacc_smooth],
        outputs=[d.qacc, d.qfrc_constraint],
    )

    fused = (
        use_optimized_kernel
        and mul_m_call is not None
        and newton_fusion.eligible(m, ctx, compact=compact)
    )
    sparse_plan = getattr(ctx, "sparse_hessian_plan", None)
    iteration = solver._solver_iteration
    if fused:
        newton_fusion.init_context(m, d, ctx, mul_m_call)
    else:
        solver.init_context(m, d, ctx, grad=False, compact=compact)
        if (
            use_optimized_kernel
            and sparse_plan is not None
            and sparse_plan.path == "nonfused"
        ):
            update_gradient, iteration = sparse_hessian.nonfused_solver_functions(
                sparse_plan
            )
            update_gradient(m, d, ctx, compact=compact)
        else:
            solver._update_gradient(m, d, ctx, compact=compact)
    if solver._use_incremental(m):
        ctx.search_unchanged.zero_()
    nsolving.fill_(d.nworld)
    if m.opt.solver == solver.types.SolverType.CG:
        wp.launch_tiled(
            solver._solve_init_search_cg_tiled,
            dim=d.nworld,
            inputs=[m.nv, m.opt.tolerance, m.stat.meaninertia, ctx.grad, ctx.Mgrad],
            outputs=[
                ctx.search,
                ctx.search_dot,
                ctx.prev_grad,
                ctx.prev_Mgrad,
                ctx.done,
                nsolving,
            ],
            block_dim=m.block_dim.solve_init_search_cg,
        )
    if fused:
        newton_fusion.run(m, d, ctx, nsolving)
    elif (
        m.opt.iterations != 0
        and m.opt.graph_conditional
        and not unroll_solver_iterations
    ):
        wp.capture_while(
            nsolving,
            while_body=iteration,
            m=m,
            d=d,
            ctx=ctx,
            nsolving=nsolving,
            compact=compact,
        )
    else:
        for _ in range(m.opt.iterations):
            iteration(m, d, ctx, nsolving, compact=compact)

    if fused:
        wp.launch(
            newton_fusion.qfrc_constraint_from_grad,
            dim=(d.nworld, m.nv),
            inputs=[d.nefc, d.qfrc_smooth, d.efc.Ma, ctx.grad, ctx.grad_scale],
            outputs=[d.qfrc_constraint],
        )
    elif solver._use_incremental(m):
        wp.launch(
            solver._qfrc_constraint_from_grad,
            dim=(d.nworld, m.nv),
            inputs=[d.qfrc_smooth, d.efc.Ma, ctx.grad, ctx.grad_scale],
            outputs=[d.qfrc_constraint],
        )


@event_scope
def solve_with_scratch(
    m: Model,
    d: Data,
    scratch: OptimizedFwdPositionScratch,
    *,
    use_optimized_kernel: bool = False,
) -> None:
    """Solve constraints using persistent buffers suitable for nested capture."""
    if getattr(scratch, "constraint_free", False) or d.njmax == 0 or m.nv == 0:
        # MuJoCo's nefc == 0 result; implicit integration reads efc.Ma = M qacc.
        wp.copy(d.qacc, d.qacc_smooth)
        if m.nv:
            scratch.mul_m_call(m, d, d.efc.Ma, d.qacc)
        d.qfrc_constraint.zero_()
        d.solver_niter.fill_(0)
        return

    ctx = scratch.solver_context
    nsolving = scratch.solver_nsolving
    if ctx is None or nsolving is None:
        raise RuntimeError("optimized solver scratch is not initialized")
    expected_compact = _solver_compact_enabled(m)
    if bool(scratch.solver_uses_compact) != expected_compact:
        raise RuntimeError("optimized solver scratch layout no longer matches model")
    if not expected_compact:
        _solve_monolithic_with_scratch(
            m,
            d,
            ctx,
            nsolving,
            use_optimized_kernel=use_optimized_kernel,
            mul_m_call=getattr(scratch, "mul_m_call", None),
            unroll_solver_iterations=getattr(
                scratch, "unroll_solver_iterations", False
            ),
        )
        return

    compact_m = scratch.solver_compact_model
    compact_d = scratch.solver_compact_data
    if compact_m is None or compact_d is None:
        raise RuntimeError("optimized compact solver scratch is not initialized")
    island.update_active_dofs(m, d)
    solver._compact_gather(m, d)
    _solve_monolithic_with_scratch(
        compact_m,
        compact_d,
        ctx,
        nsolving,
        compact=True,
        unroll_solver_iterations=getattr(scratch, "unroll_solver_iterations", False),
    )
    solver._compact_scatter(m, d)
    if getattr(m, "ntree", 1) > 1:
        mapping_call = getattr(scratch, "island_mapping_call", None)
        if mapping_call is None:
            island.compute_island_mapping(m, d)
        else:
            mapping_call(m, d)


def _stream_or_none(
    streams: Sequence[wp.Stream] | None,
    index: int,
) -> wp.Stream | None:
    if streams is None:
        return None
    if index >= len(streams):
        return None
    return streams[index]


def _launch_in_stream(
    stream: wp.Stream | None,
    kernel: Any,
    *,
    dim: tuple[int, int],
    inputs: list[Any],
    outputs: list[Any],
    **kwargs: Any,
) -> None:
    if stream is None:
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, **kwargs)
        return

    with wp.ScopedStream(stream, sync_enter=True, sync_exit=False):
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, **kwargs)


def _launch_in_stream_after(
    stream: wp.Stream | None,
    wait_streams: Sequence[wp.Stream | None],
    kernel: Any,
    *,
    dim: tuple[int, int],
    inputs: list[Any],
    outputs: list[Any],
    **kwargs: Any,
) -> None:
    deps = [wait_stream for wait_stream in wait_streams if wait_stream is not None]
    if stream is None:
        for wait_stream in deps:
            wp.wait_stream(wait_stream)
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, **kwargs)
        return

    with wp.ScopedStream(stream, sync_enter=True, sync_exit=False):
        for wait_stream in deps:
            wp.wait_stream(wait_stream)
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, **kwargs)


def _call_in_stream(stream: wp.Stream | None, fn: Any, *args: Any) -> None:
    if stream is None:
        fn(*args)
        return

    with wp.ScopedStream(stream, sync_enter=True, sync_exit=False):
        fn(*args)


def _call_in_stream_after(
    stream: wp.Stream | None,
    wait_streams: Sequence[wp.Stream | None],
    fn: Any,
    *args: Any,
) -> None:
    deps = [wait_stream for wait_stream in wait_streams if wait_stream is not None]
    if stream is None:
        for wait_stream in deps:
            wp.wait_stream(wait_stream)
        fn(*args)
        return

    with wp.ScopedStream(stream, sync_enter=True, sync_exit=False):
        for wait_stream in deps:
            wp.wait_stream(wait_stream)
        fn(*args)


def _wait_for_streams(streams: Sequence[wp.Stream] | None) -> None:
    if streams is None:
        return
    for stream in streams:
        wp.wait_stream(stream)


def _track_stream_once(streams: list[wp.Stream], stream: wp.Stream | None) -> None:
    if stream is not None and stream not in streams:
        streams.append(stream)


def _transmission_depends_only_on_qpos(m: Model) -> bool:
    """Return whether actuator transmission can be computed before kinematic tails.

    MuJoCo Warp does not currently expose host-side transmission type counts other than body
    transmissions. The structural fallback is conservative: without tendons/sites/body
    transmissions, supported actuator transmissions can only be joint or joint-in-parent.
    """
    explicit = getattr(m, "all_actuator_trntypes_are_joint", None)
    if explicit is not None:
        return bool(explicit)
    return (
        getattr(m, "nacttrnbody", 0) == 0
        and getattr(m, "ntendon", 0) == 0
        and getattr(m, "nsite", 0) == 0
    )


def _kernel_has_arg(kernel: Any, arg_name: str) -> bool:
    args = getattr(getattr(kernel, "adj", None), "args", ())
    return any(getattr(arg, "label", None) == arg_name for arg in args)


def _callback_enabled(m: Model, name: str) -> bool:
    return bool(getattr(getattr(m, "callback", None), name, None))


def optimized_kernel_capabilities(m, scratch, *, enabled):
    """Report actual per-model selection without changing runtime state."""
    ctx = getattr(scratch, "solver_context", None)
    return {
        "tree": bool(enabled and getattr(scratch, "tree_reduction", None) is not None),
        "indexed_factor": bool(
            enabled and getattr(scratch, "dense_factor_solve", None) is not None
        ),
        "persistent_newton": bool(
            enabled
            and ctx is not None
            and newton_fusion.eligible(m, ctx, compact=scratch.solver_uses_compact)
        ),
    }


def _resolve_use_optimized_kernel(
    scratch: OptimizedFwdPositionScratch | None,
    use_optimized_kernel: bool | None,
) -> bool:
    if use_optimized_kernel is None:
        return scratch is not None
    return bool(use_optimized_kernel)


def _select_tree_reduction(scratch, use_optimized_kernel):
    tree = getattr(scratch, "tree_reduction", None)
    # Deterministic stage evaluation is a numerical requirement. Its scratch
    # selects cooperative or per-level launches using the authored kernel flag.
    if use_optimized_kernel or (tree is not None and tree.deterministic):
        return tree
    return None


@event_scope
def collision_with_context(
    m: Model,
    d: Data,
    ctx: CollisionContext,
    awake_prev: wp.array | None = None,
    narrowphase_call: Any | None = None,
    broadphase_call: Any | None = None,
    flex_call: Any | None = None,
) -> None:
    """Collision detection using a persistent context to avoid graph alloc/free nodes."""
    if d.naconmax == 0 or m.opt.disableflags & (
        DisableBit.CONSTRAINT | DisableBit.CONTACT
    ):
        d.nacon.zero_()
        return

    incremental = awake_prev is not None
    d.ncollision.zero_()
    if not incremental:
        d.nacon.zero_()
    if m.opt.broadphase == BroadphaseType.NXN:
        broadphase = broadphase_call or collision_driver.nxn_broadphase
    else:
        broadphase = broadphase_call or collision_driver.sap_broadphase
    broadphase(m, d, ctx, awake_prev)

    if narrowphase_call is None:
        narrowphase_call = collision_driver._narrowphase
    narrowphase_call(m, d, ctx)
    if getattr(m, "nflex", 0) > 0 and not incremental:
        (flex_call or collision_flex.flex_collision)(m, d, ctx)
    if m.callback.contactfilter:
        m.callback.contactfilter(m, d)


def _position_sleep_enabled(m: Model) -> bool:
    sleep_bit = getattr(EnableBit, "SLEEP", 0)
    return bool(
        not (getattr(m.opt, "disableflags", 0) & DisableBit.ISLAND)
        and (getattr(m.opt, "enableflags", 0) & sleep_bit)
    )


def _forward_sleep_wake(m: Model, d: Data) -> None:
    if _position_sleep_enabled(m):
        sleep.wake(m, d)
        sleep.update_sleep(m, d)


def _run_collision_once(
    m: Model,
    d: Data,
    scratch: OptimizedFwdPositionScratch | None,
    awake_prev: wp.array | None = None,
) -> None:
    if scratch is None:
        collision_driver.collision(m, d, awake_prev=awake_prev)
        return

    collision_with_context(
        m,
        d,
        scratch.collision_context,
        awake_prev,
        getattr(scratch, "collision_narrowphase_call", None),
        (
            getattr(scratch, "collision_incremental_broadphase_call", None)
            or getattr(scratch, "collision_broadphase_call", None)
            if awake_prev is not None
            else getattr(scratch, "collision_broadphase_call", None)
        ),
        _flex_collision_for_capture(scratch),
    )


def _flex_collision_for_capture(scratch: OptimizedFwdPositionScratch) -> Any:
    """The flex collision call, made safe for the capture it runs in.

    MuJoCo Warp's flex FPS filter loops with ``wp.capture_while``. Warp builds the
    conditional node only on a registered capture stream with no unjoined forks, but
    the parallel step runs collision on a side stream forked into the capture, and
    the nested benchmark capture unrolls conditional loops (as the solver does). In
    those cases the call runs with ``graph_conditional`` off, which makes upstream
    use its fixed-count FPS loop.
    """
    flex_call = getattr(scratch, "flex_collision_call", None) or (
        collision_flex.flex_collision
    )

    def call(m: Model, d: Data, ctx: Any) -> None:
        device = wp.get_device()
        captures = getattr(device, "captures", None) or {}
        side_stream = (
            bool(captures) and device.is_cuda and device.stream not in captures
        )
        unroll = side_stream or bool(
            getattr(scratch, "unroll_solver_iterations", False)
        )
        conditional = m.opt.graph_conditional
        if unroll:
            m.opt.graph_conditional = False
        try:
            flex_call(m, d, ctx)
        finally:
            m.opt.graph_conditional = conditional

    return call


def _run_collision_sequence(
    m: Model,
    d: Data,
    scratch: OptimizedFwdPositionScratch | None,
    sleep_enabled: bool,
) -> None:
    if not sleep_enabled:
        _run_collision_once(m, d, scratch)
        return

    _run_collision_once(m, d, scratch)
    sleep.wake_collision(m, d)
    awake_prev = (
        getattr(scratch, "collision_awake_prev", None) if scratch is not None else None
    )
    if awake_prev is None:
        awake_prev = wp.clone(d.body_awake)
    else:
        wp.copy(awake_prev, d.body_awake)
    sleep.update_sleep(m, d)
    _run_collision_once(m, d, scratch, awake_prev)


@event_scope
def transmission_with_scratch(
    m: Model,
    d: Data,
    scratch: OptimizedFwdPositionScratch,
) -> None:
    """Transmission pass using persistent scratch to avoid graph alloc/free nodes."""
    moment_nnz = scratch.transmission_moment_nnz
    moment_nnz.zero_()

    transmission_inputs = [
        m.nv,
        m.body_parentid,
        m.body_rootid,
        m.body_weldid,
        m.body_dofnum,
        m.body_dofadr,
        m.jnt_type,
        m.jnt_qposadr,
        m.jnt_dofadr,
        m.dof_bodyid,
        m.dof_parentid,
        m.site_bodyid,
        m.site_quat,
        m.ten_J_rownnz,
        m.ten_J_rowadr,
        m.ten_J_colind,
        m.actuator_trntype,
        m.actuator_trnid,
        m.actuator_cranklength,
        m.actuator_gear,
    ]
    if _kernel_has_arg(smooth._transmission, "body_isdofancestor"):
        transmission_inputs.append(m.body_isdofancestor)
    transmission_inputs.extend(
        [
            d.qpos,
            d.xquat,
            d.site_xpos,
            d.site_xmat,
            d.subtree_com,
            d.cdof,
            d.ten_J,
            d.ten_length,
            moment_nnz,
        ]
    )

    wp.launch(
        smooth._transmission,
        dim=(d.nworld, m.nactuator),
        inputs=transmission_inputs,
        outputs=[
            d.actuator_length,
            d.moment_rownnz,
            d.moment_rowadr,
            d.moment_colind,
            d.actuator_moment,
        ],
    )

    if m.nacttrnbody:
        ncon = scratch.transmission_body_ncon
        if ncon is None:
            msg = "optimized transmission scratch is missing body transmission buffer"
            raise RuntimeError(msg)
        ncon.zero_()
        body_moment_inputs = [
            m.opt.cone,
            m.body_parentid,
            m.body_rootid,
            m.dof_bodyid,
            m.geom_bodyid,
            m.actuator_trnid,
        ]
        if _kernel_has_arg(smooth._transmission_body_moment, "body_isdofancestor"):
            body_moment_inputs.append(m.body_isdofancestor)
        body_moment_inputs.extend(
            [
                m.actuator_trntype_body_adr,
                d.subtree_com,
                d.cdof,
                d.moment_rowadr,
                d.contact.dist,
                d.contact.pos,
                d.contact.frame,
                d.contact.includemargin,
                d.contact.dim,
                d.contact.geom,
                d.contact.efc_address,
                d.contact.worldid,
                d.efc.J_rownnz,
                d.efc.J_rowadr,
                d.efc.J_colind,
                d.efc.J,
                d.nacon,
                m.is_sparse,
            ]
        )
        wp.launch(
            smooth._transmission_body_moment,
            dim=(m.nacttrnbody, d.naconmax, m.nv),
            inputs=body_moment_inputs,
            outputs=[d.actuator_moment, ncon],
        )

        wp.launch(
            smooth._transmission_body_moment_scale,
            dim=(d.nworld, m.nacttrnbody, m.nv),
            inputs=[m.actuator_trntype_body_adr, d.moment_rowadr, ncon],
            outputs=[d.actuator_moment],
        )


@event_scope
def com_pos_parallel_tail(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    *,
    kinematics_tail_streams: KinematicsTailStreams | None = None,
    tree_reduction: tree_fusion.TreeReductionScratch | None = None,
):
    """Compute COM position state and split independent inertial/DOF tails."""
    if (
        kinematics_tail_streams is not None
        and kinematics_tail_streams.body_inertial is not None
    ):
        wp.wait_stream(kinematics_tail_streams.body_inertial)

    wp.launch(
        smooth._subtree_com_init,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_mass, d.xipos],
        outputs=[d.subtree_com],
    )

    tree_fusion.subtree_com_accumulate(m, d, tree_reduction)

    wp.launch(
        smooth._subtree_div,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_subtreemass, d.subtree_com],
        outputs=[d.subtree_com],
    )
    _launch_in_stream(
        _stream_or_none(streams, COM_POS_CINERT_STREAM),
        smooth._cinert,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.body_rootid,
            m.body_mass,
            m.body_inertia,
            d.xipos,
            d.ximat,
            d.subtree_com,
        ],
        outputs=[d.cinert],
    )
    cdof_wait_streams = []
    if (
        kinematics_tail_streams is not None
        and kinematics_tail_streams.body_matrix is not None
    ):
        cdof_wait_streams.append(kinematics_tail_streams.body_matrix)
    if cdof_wait_streams:
        _launch_in_stream_after(
            _stream_or_none(streams, COM_POS_CDOF_STREAM),
            cdof_wait_streams,
            smooth._cdof,
            dim=(d.nworld, m.njnt),
            inputs=[
                m.body_rootid,
                m.jnt_type,
                m.jnt_dofadr,
                m.jnt_bodyid,
                d.xmat,
                d.xanchor,
                d.xaxis,
                d.subtree_com,
            ],
            outputs=[d.cdof],
        )
    else:
        _launch_in_stream(
            _stream_or_none(streams, COM_POS_CDOF_STREAM),
            smooth._cdof,
            dim=(d.nworld, m.njnt),
            inputs=[
                m.body_rootid,
                m.jnt_type,
                m.jnt_dofadr,
                m.jnt_bodyid,
                d.xmat,
                d.xanchor,
                d.xaxis,
                d.subtree_com,
            ],
            outputs=[d.cdof],
        )
    _wait_for_streams(
        [
            stream
            for stream in (
                _stream_or_none(streams, COM_POS_CINERT_STREAM),
                _stream_or_none(streams, COM_POS_CDOF_STREAM),
            )
            if stream is not None
        ]
    )


@event_scope
def com_pos_vel_parallel(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
) -> None:
    """Compute COM position and velocity state, overlapping cinert with com_vel."""
    if streams is None:
        smooth.com_pos(m, d)
        smooth.com_vel(m, d)
        return

    wp.launch(
        smooth._subtree_com_init,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_mass, d.xipos],
        outputs=[d.subtree_com],
    )

    for i in reversed(range(len(m.body_tree))):
        body_tree = m.body_tree[i]
        wp.launch(
            smooth._subtree_com_acc,
            dim=(d.nworld, body_tree.size),
            inputs=[m.body_parentid, d.subtree_com, body_tree],
            outputs=[d.subtree_com],
        )

    wp.launch(
        smooth._subtree_div,
        dim=(d.nworld, m.nbody),
        inputs=[m.body_subtreemass, d.subtree_com],
        outputs=[d.subtree_com],
    )

    cinert_stream = _stream_or_none(streams, COM_POS_CINERT_STREAM)
    cdof_stream = _stream_or_none(streams, COM_POS_CDOF_STREAM)
    _launch_in_stream(
        cinert_stream,
        smooth._cinert,
        dim=(d.nworld, m.nbody),
        inputs=[
            m.body_rootid,
            m.body_mass,
            m.body_inertia,
            d.xipos,
            d.ximat,
            d.subtree_com,
        ],
        outputs=[d.cinert],
    )
    _launch_in_stream(
        cdof_stream,
        smooth._cdof,
        dim=(d.nworld, m.njnt),
        inputs=[
            m.body_rootid,
            m.jnt_type,
            m.jnt_dofadr,
            m.jnt_bodyid,
            d.xmat,
            d.xanchor,
            d.xaxis,
            d.subtree_com,
        ],
        outputs=[d.cdof],
    )

    wp.launch(smooth._comvel_root, dim=(d.nworld, 6), inputs=[], outputs=[d.cvel])
    if cdof_stream is not None:
        wp.wait_stream(cdof_stream)
    wp.launch(
        smooth._comvel_branch,
        dim=(d.nworld, m.nbranch),
        inputs=[
            m.body_parentid,
            m.body_jntnum,
            m.body_jntadr,
            m.body_dofadr,
            m.jnt_type,
            m.body_branches,
            m.body_branch_start,
            d.qvel,
            d.cdof,
        ],
        outputs=[d.cvel, d.cdof_dot],
    )
    if cinert_stream is not None:
        wp.wait_stream(cinert_stream)


@event_scope
def _start_kinematics_parallel_tail(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    *,
    fuse_body_frames: bool = False,
) -> KinematicsTailStreams:
    """Start forward kinematics with independent tail kernels on streams."""
    wp.launch(
        smooth._kinematics_branch,
        dim=(d.nworld, m.nbranch),
        inputs=[
            m.qpos0,
            m.body_parentid,
            m.body_mocapid,
            m.body_jntnum,
            m.body_jntadr,
            m.body_pos,
            m.body_quat,
            m.jnt_type,
            m.jnt_qposadr,
            m.jnt_pos,
            m.jnt_axis,
            m.body_branches,
            m.body_branch_start,
            d.qpos,
            d.mocap_pos,
            d.mocap_quat,
        ],
        outputs=[d.xpos, d.xquat, d.xanchor, d.xaxis],
    )

    body_matrix_stream = _stream_or_none(streams, 0)
    body_inertial_stream = _stream_or_none(streams, 0 if fuse_body_frames else 1)
    if fuse_body_frames:
        _launch_in_stream(
            body_matrix_stream,
            body_frames_kernel,
            dim=(d.nworld, m.nbody),
            inputs=[m.body_ipos, m.body_iquat, d.xpos, d.xquat],
            outputs=[d.xmat, d.xipos, d.ximat],
        )
    else:
        _launch_in_stream(
            body_matrix_stream,
            smooth._compute_body_matrices,
            dim=(d.nworld, m.nbody),
            inputs=[d.xquat],
            outputs=[d.xmat],
        )
        _launch_in_stream(
            body_inertial_stream,
            smooth._compute_body_inertial_frames,
            dim=(d.nworld, m.nbody),
            inputs=[m.body_ipos, m.body_iquat, d.xpos, d.xquat],
            outputs=[d.xipos, d.ximat],
        )
    _launch_in_stream(
        _stream_or_none(streams, 2),
        smooth._geom_local_to_global,
        dim=(d.nworld, m.ngeom),
        inputs=[
            m.body_rootid,
            m.body_weldid,
            m.body_mocapid,
            m.geom_bodyid,
            m.geom_pos,
            m.geom_quat,
            d.xpos,
            d.xquat,
        ],
        outputs=[d.geom_xpos, d.geom_xmat],
    )
    _launch_in_stream(
        _stream_or_none(streams, 3),
        smooth._site_local_to_global,
        dim=(d.nworld, m.nsite),
        inputs=[m.site_bodyid, m.site_pos, m.site_quat, d.xpos, d.xquat],
        outputs=[d.site_xpos, d.site_xmat],
    )
    return KinematicsTailStreams(
        body_matrix=body_matrix_stream,
        body_inertial=body_inertial_stream,
        geom=_stream_or_none(streams, 2),
        site=_stream_or_none(streams, 3),
    )


@event_scope
def kinematics_parallel_tail(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
):
    """Forward kinematics with independent body/geom/site tail kernels on streams."""
    tail_streams = _start_kinematics_parallel_tail(m, d, streams)
    _wait_for_streams(tail_streams.as_tuple())


@event_scope
def fwd_position_optimized(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None = None,
    factorize: bool = True,
    skip_kinematics: bool = False,
    use_optimized_kernel: bool | None = None,
    pipeline_velocity: bool = False,
    constraints: bool = True,
) -> PendingComVelocity:
    """MuJoCo Warp position pass with independent branches on side CUDA streams.

    ``constraints=False`` skips collision and constraint construction: contacts
    and constraint rows keep their previous values until a later pass rebuilds
    them.
    """
    use_optimized_kernel = _resolve_use_optimized_kernel(scratch, use_optimized_kernel)
    tree_reduction = _select_tree_reduction(scratch, use_optimized_kernel)
    sleep_enabled = _position_sleep_enabled(m)
    if not constraints and sleep_enabled:
        # Islands and sleep updates read the constraint rows built here.
        raise ValueError("sleeping models must build constraints in every pass")
    pending_velocity = PendingComVelocity()
    pipeline_velocity = pipeline_velocity and _can_pipeline_position_velocity(
        m, streams
    )
    kinematics_tail_streams = None
    if not skip_kinematics:
        if streams is None:
            kinematics_parallel_tail(m, d, None)
        else:
            kinematics_tail_streams = _start_kinematics_parallel_tail(
                m,
                d,
                streams[:4],
                fuse_body_frames=use_optimized_kernel and bool(streams),
            )

    active_streams: list[wp.Stream] = []
    transmission_stream = _stream_or_none(streams, TRANSMISSION_STREAM)
    transmission_started = False
    if _transmission_depends_only_on_qpos(m):
        if scratch is None:
            _call_in_stream(transmission_stream, smooth.transmission, m, d)
        else:
            _call_in_stream(
                transmission_stream,
                transmission_with_scratch,
                m,
                d,
                scratch,
            )
        _track_stream_once(active_streams, transmission_stream)
        transmission_started = True

    if kinematics_tail_streams is None:
        com_pos_parallel_tail(m, d, streams, tree_reduction=tree_reduction)
    else:
        com_pos_parallel_tail(
            m,
            d,
            streams,
            kinematics_tail_streams=kinematics_tail_streams,
            tree_reduction=tree_reduction,
        )

    if pipeline_velocity and not _constraint_reads_com_velocity(m):
        pending_velocity = _start_pipelined_com_velocity(m, d, streams, tree_reduction)

    camlight_stream = _stream_or_none(streams, CAMLIGHT_STREAM)
    _call_in_stream(camlight_stream, smooth.camlight, m, d)
    if camlight_stream is not None:
        active_streams.append(camlight_stream)

    smooth.flex(m, d)
    if kinematics_tail_streams is not None:
        _wait_for_streams(kinematics_tail_streams.as_tuple())
        kinematics_tail_streams = None
    tendon_call = getattr(scratch, "tendon_call", None) if scratch is not None else None
    if tendon_call is None:
        smooth.tendon(m, d)
    else:
        tendon_call(m, d)
    if sleep_enabled and getattr(m, "ntendon", 0) > 0:
        sleep.wake_tendon(m, d)
        sleep.update_sleep_trees(m, d)

    early_transmission = getattr(m, "nacttrnbody", 0) == 0
    if early_transmission and not transmission_started:
        if scratch is None:
            _call_in_stream(transmission_stream, smooth.transmission, m, d)
        else:
            _call_in_stream(
                transmission_stream,
                transmission_with_scratch,
                m,
                d,
                scratch,
            )
        _track_stream_once(active_streams, transmission_stream)
        transmission_started = True

    crb_stream = _stream_or_none(streams, CRB_STREAM)
    _call_in_stream(crb_stream, tree_fusion.crb, m, d, tree_reduction)
    _call_in_stream(crb_stream, smooth.tendon_armature, m, d)
    if factorize:
        _call_in_stream(crb_stream, smooth.factor_m, m, d)
    _track_stream_once(active_streams, crb_stream)

    collision_stream = _stream_or_none(streams, COLLISION_CONSTRAINT_STREAM)
    constraint_stream = collision_stream
    constraint_call = (
        getattr(scratch, "constraint_call", None) if scratch is not None else None
    )
    if constraint_call is None:
        constraint_call = constraint.make_constraint
    crb_wait_streams = [crb_stream] if crb_stream is not None else []
    # A constraint-free model emits no contacts or rows; nothing to build.
    if constraints and not getattr(scratch, "constraint_free", False):
        if m.opt.run_collision_detection:
            if sleep_enabled or _callback_enabled(m, "contactfilter"):
                _call_in_stream_after(
                    collision_stream,
                    crb_wait_streams,
                    _run_collision_sequence,
                    m,
                    d,
                    scratch,
                    sleep_enabled,
                )
            else:
                _call_in_stream(
                    collision_stream,
                    _run_collision_sequence,
                    m,
                    d,
                    scratch,
                    sleep_enabled,
                )
            _call_in_stream_after(
                constraint_stream,
                crb_wait_streams,
                constraint_call,
                m,
                d,
            )
        else:
            _call_in_stream_after(
                constraint_stream,
                crb_wait_streams,
                constraint_call,
                m,
                d,
            )
        _track_stream_once(active_streams, constraint_stream)
        if constraint_stream is not None:
            _wait_for_streams([constraint_stream])
    # make_constraint intentionally runs before fwd_velocity in MuJoCo Warp and
    # reads the previous cvel/cdof_dot values.  Start com_vel only after that
    # read completes; it can still overlap the remaining independent position
    # branches (for example transmission and camera/light work).
    if pipeline_velocity and not pending_velocity.started:
        pending_velocity = _start_pipelined_com_velocity(m, d, streams, tree_reduction)
    if sleep_enabled:
        if getattr(m, "neq", 0) > 0:
            sleep.wake_equality(m, d)
        sleep.update_sleep(m, d)
    if sleep_enabled:
        island_call = getattr(scratch, "island_call", None) if scratch else None
        if island_call is None:
            island.island(m, d)
        else:
            island_call(m, d)

    _wait_for_streams(active_streams)
    if not transmission_started:
        if scratch is None:
            smooth.transmission(m, d)
        else:
            transmission_with_scratch(m, d, scratch)
    return pending_velocity


@event_scope
def fwd_position_skip_kinematics(m: Model, d: Data, factorize: bool = True):
    """Position-dependent MuJoCo Warp pass for states with fresh kinematics."""
    sleep_enabled = _position_sleep_enabled(m)
    smooth.com_pos(m, d)
    smooth.camlight(m, d)
    smooth.flex(m, d)
    smooth.tendon(m, d)
    if sleep_enabled and getattr(m, "ntendon", 0) > 0:
        sleep.wake_tendon(m, d)
        sleep.update_sleep_trees(m, d)
    smooth.crb(m, d)
    smooth.tendon_armature(m, d)
    if factorize:
        smooth.factor_m(m, d)
    if m.opt.run_collision_detection:
        _run_collision_sequence(m, d, None, sleep_enabled)
    constraint.make_constraint(m, d)
    if sleep_enabled:
        if getattr(m, "neq", 0) > 0:
            sleep.wake_equality(m, d)
        sleep.update_sleep(m, d)
    if sleep_enabled:
        island.island(m, d)
    smooth.transmission(m, d)


def _has_tendon_bias_scratch(
    m: Model,
    scratch: OptimizedFwdPositionScratch | None,
) -> bool:
    return (
        getattr(m, "ntendon", 0) > 0
        and scratch is not None
        and scratch.tendon_bias_Jdot is not None
        and scratch.tendon_bias_coef is not None
    )


def _has_implicit_dense_deriv_scratch(
    m: Model,
    scratch: OptimizedFwdPositionScratch | None,
) -> bool:
    if scratch is None:
        return False
    has_actuator_scratch = (
        getattr(m, "nactuator", 0) > 0
        and scratch.implicit_actuator_vel is not None
        and scratch.implicit_qderiv_pair_mask is not None
        and scratch.implicit_max_moment_rownnz > 0
    )
    has_contributors = bool(getattr(scratch, "implicit_qh_contributors", ()))
    return not bool(getattr(m, "is_sparse", False)) and (
        has_actuator_scratch or has_contributors
    )


def _has_preallocated_implicit_dense_deriv_scratch(
    m: Model,
    scratch: OptimizedFwdPositionScratch | None,
) -> bool:
    return (
        _has_implicit_dense_deriv_scratch(m, scratch)
        and scratch is not None
        and getattr(scratch, "implicit_qderiv", None) is not None
        and getattr(scratch, "implicit_qacc", None) is not None
    )


def _has_preallocated_implicit_contributor_scratch(
    m: Model,
    scratch: OptimizedFwdPositionScratch | None,
) -> bool:
    return bool(
        _has_preallocated_implicitfast_scratch(scratch)
        and scratch is not None
        and bool(getattr(scratch, "implicit_qh_contributors", ()))
    )


def _has_preallocated_implicitfast_scratch(
    scratch: OptimizedFwdPositionScratch | None,
) -> bool:
    return bool(
        scratch is not None
        and callable(getattr(scratch, "implicit_deriv_call", None))
        and getattr(scratch, "implicit_qderiv", None) is not None
        and getattr(scratch, "implicit_qld", None) is not None
        and getattr(scratch, "implicit_ql_diag_inv", None) is not None
        and getattr(scratch, "implicit_qacc", None) is not None
    )


def _implicitfast_native_needs_deriv(m: Model) -> bool:
    return bool(
        ~(
            m.opt.disableflags
            | ~(DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER)
        )
    )


def _implicitfast_needs_deriv(
    m: Model, scratch: OptimizedFwdPositionScratch | None = None
) -> bool:
    return _implicitfast_native_needs_deriv(m) or bool(
        scratch is not None and getattr(scratch, "implicit_qh_contributors", ())
    )


def _apply_implicit_qh_contributors(
    scratch: OptimizedFwdPositionScratch, qh: wp.array
) -> None:
    for contributor in getattr(scratch, "implicit_qh_contributors", ()):
        contributor.apply(qh)


def _prepare_implicit_qh(
    m: Model, d: Data, scratch: OptimizedFwdPositionScratch, qh: wp.array
) -> None:
    free = getattr(m, "body_freeadr", None)
    if free is not None and free.size:
        wp.launch(
            mjw_forward._implicit_free_body_reset_m,
            dim=(d.nworld, free.size),
            inputs=[m.body_dofadr, m.M_rownnz, m.M_rowadr, free, d.M],
            outputs=[qh],
        )
    _apply_implicit_qh_contributors(scratch, qh)


def _make_implicit_free_body_call(qh: wp.array) -> _PreallocatedWarpCall:
    call = _PreallocatedWarpCall(mjw_forward._launch_implicit_free_body_solve, ())
    kernel = mjw_forward._implicit_free_body_solve
    matrix_index = next(
        i for i, arg in enumerate(kernel.adj.args) if arg.label == "M_in"
    )

    def use_effective_matrix(launch, kernel, kwargs):
        kwargs = dict(kwargs)
        inputs = list(kwargs["inputs"])
        # Native M_in is only the initial 6x6 gather. Following the free-block
        # reset, qH=M+custom_delta; native damping/gyro/fluid are added once.
        inputs[matrix_index] = qh
        kwargs["inputs"] = inputs
        return launch(kernel, **kwargs)

    call.add_launch_substitutions({kernel: use_effective_matrix})
    return call


def _implicit_free_body_correction(
    m: Model, d: Data, qacc: wp.array, scratch: OptimizedFwdPositionScratch
) -> None:
    free = getattr(m, "body_freeadr", None)
    if free is None or not free.size:
        return
    if getattr(scratch, "implicit_qh_contributors", ()):
        call = getattr(scratch, "implicit_free_body_call", None)
        if call is None:
            raise RuntimeError("free-body implicit correction needs persistent qH")
        call(m, d, qacc)
    else:
        mjw_forward._launch_implicit_free_body_solve(m, d, qacc)


def _deriv_smooth_vel_allocates_actuator_velocity(m: Model) -> bool:
    return bool(
        getattr(m, "nactuator", 0) > 0
        and not (m.opt.disableflags & DisableBit.ACTUATION)
    )


def _deriv_smooth_vel_assembles(m: Model) -> bool:
    """Whether ``M - h*qDeriv`` needs the actuator or the dof damping term.

    MuJoCo adds the actuator velocity derivative unless ACTUATION is disabled
    and the passive damping derivative unless DAMPER is disabled, each on its
    own. MuJoCo Warp 3.16 gates both on ``not (ACTUATION | DAMPER)``, which
    drops joint damping whenever native actuation is off, including MASS
    pose control, which applies muscle forces outside the actuators.
    """
    flags = m.opt.disableflags
    return not (flags & DisableBit.ACTUATION and flags & DisableBit.DAMPER)


@event_scope
def deriv_smooth_vel(m: Model, d: Data, out: wp.array) -> None:
    """Upstream ``derivative.deriv_smooth_vel`` with MuJoCo's independent gates.

    Same kernels and allocation sequence as MuJoCo Warp 3.16; only the outer
    condition follows ``_deriv_smooth_vel_assembles``.
    """
    Mi = m.M_fullm_i
    Mj = m.M_fullm_j

    if _deriv_smooth_vel_assembles(m):
        out.zero_()
        if m.nactuator > 0 and not (m.opt.disableflags & DisableBit.ACTUATION):
            vel = wp.empty((d.nworld, m.nactuator), dtype=float)
            wp.launch(
                derivative._qderiv_actuator_passive_vel,
                dim=(d.nworld, m.nactuator),
                inputs=[
                    m.opt.timestep,
                    m.actuator_dyntype,
                    m.actuator_gaintype,
                    m.actuator_biastype,
                    m.actuator_ctrlspec,
                    m.actuator_actadr,
                    m.actuator_actnum,
                    m.actuator_dynprm,
                    m.actuator_gainprm,
                    m.actuator_biasprm,
                    m.actuator_actlimited,
                    m.actuator_actrange,
                    m.actuator_actearly,
                    m.actuator_forcelimited,
                    m.actuator_forcerange,
                    m.actuator_acc0,
                    m.actuator_lengthrange,
                    d.act,
                    d.ctrl,
                    d.act_dot,
                    d.actuator_length,
                    d.actuator_velocity,
                    d.actuator_force,
                ],
                outputs=[vel],
            )
            wp.launch(
                derivative._qderiv_actuator_passive_actuation_sparse,
                dim=(d.nworld, m.nactuator),
                inputs=[
                    m.M_elemid,
                    d.moment_rownnz,
                    d.moment_rowadr,
                    d.moment_colind,
                    d.actuator_moment,
                    vel,
                ],
                outputs=[out],
            )
        wp.launch(
            derivative._qderiv_actuator_passive,
            dim=(d.nworld, Mi.size),
            inputs=[
                m.opt.timestep,
                m.opt.disableflags,
                m.dof_damping,
                m.dof_dampingpoly,
                m.M_elemid,
                d.qvel,
                d.M,
                Mi,
                Mj,
                out,
            ],
            outputs=[out],
        )
    else:
        wp.copy(out, d.M)

    if not (m.opt.disableflags & DisableBit.DAMPER):
        wp.launch(
            derivative._qderiv_tendon_damping,
            dim=(d.nworld, Mi.size),
            inputs=[
                m.ntendon,
                m.opt.timestep,
                m.ten_J_rownnz,
                m.ten_J_rowadr,
                m.ten_J_colind,
                m.tendon_damping,
                m.tendon_dampingpoly,
                m.M_elemid,
                d.ten_J,
                d.ten_velocity,
                Mi,
                Mj,
            ],
            outputs=[out],
        )
    if m.has_fluid:
        wp.launch(
            derivative._qderiv_ellipsoid_fluid,
            dim=(d.nworld, m.body_fluid_ellipsoid_adr.size, Mi.size),
            inputs=[
                m.opt.timestep,
                m.opt.wind,
                m.opt.density,
                m.opt.viscosity,
                m.opt.integrator,
                m.body_parentid,
                m.body_rootid,
                m.body_geomnum,
                m.body_geomadr,
                m.dof_bodyid,
                m.geom_type,
                m.geom_size,
                m.geom_fluid,
                m.body_fluid_ellipsoid_adr,
                m.body_isdofancestor,
                m.M_elemid,
                d.xipos,
                d.geom_xpos,
                d.geom_xmat,
                d.subtree_com,
                d.cdof,
                d.cvel,
                Mi,
                Mj,
            ],
            outputs=[out],
        )
        wp.launch(
            derivative._qderiv_box_fluid,
            dim=(d.nworld, m.body_fluid_box_adr.size, Mi.size),
            inputs=[
                m.opt.timestep,
                m.opt.wind,
                m.opt.density,
                m.opt.viscosity,
                m.opt.integrator,
                m.body_parentid,
                m.body_rootid,
                m.body_mass,
                m.body_inertia,
                m.dof_bodyid,
                m.body_fluid_box_adr,
                m.body_isdofancestor,
                m.M_elemid,
                d.xipos,
                d.ximat,
                d.subtree_com,
                d.cdof,
                d.cvel,
                Mi,
                Mj,
            ],
            outputs=[out],
        )


def _rebind_globals(fn: Any, **overrides: Any) -> Any:
    raw = getattr(fn, "__wrapped__", fn)
    return FunctionType(
        raw.__code__,
        {**raw.__globals__, **overrides},
        raw.__name__,
        raw.__defaults__,
        raw.__closure__,
    )


def _implicit(m: Model, d: Data) -> None:
    """Upstream ``forward.implicit`` reaching ``deriv_smooth_vel`` above."""
    _rebind_globals(
        mjw_forward.implicit,
        derivative=_AttributeOverrides(
            derivative, {"deriv_smooth_vel": deriv_smooth_vel}
        ),
    )(m, d)


def _euler_damping_needs_scratch(m: Model) -> bool:
    return not bool(m.opt.disableflags & (DisableBit.EULERDAMP | DisableBit.DAMPER))


def supports_allocation_free_implicit_capture(
    m: Model,
    scratch: OptimizedFwdPositionScratch | None,
    *,
    use_optimized_kernel: bool,
) -> bool:
    """Return whether integration avoids live allocations during graph capture."""
    if _solver_compact_enabled(m) and bool(
        getattr(getattr(m, "opt", None), "graph_conditional", False)
    ):
        # The 3.16 incremental NXN sleep broadphase uses wp.capture_if(), which
        # synchronizes while constructing its own graph and cannot be nested in
        # the production outer capture.
        return False
    integrator = getattr(getattr(m, "opt", None), "integrator", None)
    if integrator == IntegratorType.EULER:
        if not _euler_damping_needs_scratch(m):
            return True
        return bool(scratch is not None and callable(scratch.euler_call))
    if integrator == IntegratorType.RK4:
        return bool(
            scratch is not None
            and getattr(scratch, "rk4", None) is not None
            and (
                not getattr(getattr(m, "opt", None), "graph_conditional", False)
                or (
                    getattr(scratch, "constraint_free", False)
                    and getattr(scratch.rk4, "time_t0", None) is not None
                )
            )
        )
    if integrator not in (IntegratorType.IMPLICITFAST, IntegratorType.IMPLICIT):
        return True
    if integrator == IntegratorType.IMPLICITFAST and not _implicitfast_needs_deriv(
        m, scratch
    ):
        return True
    if scratch is None:
        return False
    if integrator == IntegratorType.IMPLICIT:
        return callable(getattr(scratch, "implicit_call", None))
    if _has_preallocated_implicit_contributor_scratch(m, scratch):
        return True
    if getattr(scratch, "pre_advance_modifier", None) is not None:
        return not _implicitfast_needs_deriv(m, scratch) or bool(
            _has_preallocated_implicitfast_scratch(scratch)
        )
    if use_optimized_kernel:
        return (
            _has_preallocated_implicit_dense_deriv_scratch(m, scratch)
            or _has_implicit_factor_scratch(m, scratch)
            or callable(getattr(scratch, "implicit_call", None))
        )
    return callable(getattr(scratch, "implicit_call", None))


def _start_optimized_implicit_deriv_before_solve(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None,
    *,
    use_optimized_kernel: bool,
) -> OptimizedImplicitDerivResult:
    if not use_optimized_kernel:
        return OptimizedImplicitDerivResult()
    if getattr(m.opt, "integrator", None) != IntegratorType.IMPLICITFAST:
        return OptimizedImplicitDerivResult()
    if not _implicitfast_needs_deriv(m, scratch):
        return OptimizedImplicitDerivResult()
    if getattr(m.opt, "graph_conditional", False):
        # Warp's conditional solver pauses capture internally. CUDA graph capture
        # requires all forked side-stream work to be joined before that pause.
        # Keep the optimized derivative kernel, but run it later in _integrate().
        return OptimizedImplicitDerivResult()
    if not _has_preallocated_implicit_dense_deriv_scratch(m, scratch):
        return OptimizedImplicitDerivResult()

    qderiv = scratch.implicit_qderiv
    qacc = scratch.implicit_qacc
    stream = _stream_or_none(streams, ACTUATOR_VELOCITY_STREAM)
    _call_in_stream(
        stream,
        deriv_smooth_vel_dense_actuator_centric,
        m,
        d,
        qderiv,
        scratch,
    )
    return OptimizedImplicitDerivResult(
        started=True,
        stream=stream,
        qderiv=qderiv,
        qacc=qacc,
    )


@event_scope
def tendon_bias_prepare_with_scratch(
    m: Model,
    d: Data,
    scratch: OptimizedFwdPositionScratch,
) -> None:
    """Prepare tendon bias scratch that is independent of RNE qfrc output."""
    ten_Jdot = scratch.tendon_bias_Jdot
    ten_bias_coef = scratch.tendon_bias_coef
    if ten_Jdot is None or ten_bias_coef is None:
        msg = "optimized tendon bias scratch is missing tendon buffers"
        raise RuntimeError(msg)

    ten_Jdot.zero_()
    ten_bias_coef.zero_()

    wp.launch(
        smooth._tendon_dot,
        dim=(d.nworld, m.ntendon),
        inputs=[
            m.body_parentid,
            m.body_rootid,
            m.body_dofnum,
            m.body_dofadr,
            m.jnt_type,
            m.jnt_dofadr,
            m.dof_jntid,
            m.site_bodyid,
            m.tendon_adr,
            m.tendon_num,
            m.ten_J_rownnz,
            m.ten_J_rowadr,
            m.ten_J_colind,
            m.tendon_armature,
            m.wrap_type,
            m.wrap_objid,
            m.wrap_prm,
            d.site_xpos,
            d.subtree_com,
            d.cdof,
            d.cvel,
            d.cdof_dot,
        ],
        outputs=[ten_Jdot],
    )
    wp.launch(
        smooth._tendon_bias_coef,
        dim=(d.nworld, m.ntendon, m.max_ten_J_rownnz),
        inputs=[
            m.ten_J_rownnz,
            m.ten_J_rowadr,
            m.ten_J_colind,
            m.tendon_armature,
            d.qvel,
            ten_Jdot,
        ],
        outputs=[ten_bias_coef],
    )


@event_scope
def tendon_bias_apply_with_scratch(
    m: Model,
    d: Data,
    qfrc: wp.array,
    scratch: OptimizedFwdPositionScratch,
) -> None:
    """Apply precomputed tendon bias into qfrc after RNE has written qfrc."""
    ten_bias_coef = scratch.tendon_bias_coef
    if ten_bias_coef is None:
        msg = "optimized tendon bias scratch is missing tendon coefficient buffer"
        raise RuntimeError(msg)

    wp.launch(
        smooth._tendon_bias_qfrc,
        dim=(d.nworld, m.ntendon, m.max_ten_J_rownnz),
        inputs=[
            m.ten_J_rownnz,
            m.ten_J_rowadr,
            m.ten_J_colind,
            m.tendon_armature,
            d.ten_J,
            ten_bias_coef,
        ],
        outputs=[qfrc],
    )


@event_scope
def deriv_smooth_vel_dense_actuator_centric(
    m: Model,
    d: Data,
    out: wp.array,
    scratch: OptimizedFwdPositionScratch,
) -> None:
    """Dense implicit velocity derivative with actuator-centric actuation term."""
    if not _has_implicit_dense_deriv_scratch(m, scratch):
        deriv_smooth_vel(m, d, out)
        return

    fullm_i, fullm_j = _fullm_indices(m)
    use_m_structure = _has_m_structure(m, d)

    if _deriv_smooth_vel_assembles(m):
        out.zero_()
        if m.nactuator > 0 and not (m.opt.disableflags & DisableBit.ACTUATION):
            vel = scratch.implicit_actuator_vel
            if vel is None:
                msg = "optimized implicit derivative scratch is missing velocity buffer"
                raise RuntimeError(msg)
            wp.launch(
                derivative._qderiv_actuator_passive_vel,
                dim=(d.nworld, m.nactuator),
                inputs=[
                    m.opt.timestep,
                    m.actuator_dyntype,
                    m.actuator_gaintype,
                    m.actuator_biastype,
                    m.actuator_ctrlspec,
                    m.actuator_actadr,
                    m.actuator_actnum,
                    m.actuator_dynprm,
                    m.actuator_gainprm,
                    m.actuator_biasprm,
                    m.actuator_actlimited,
                    m.actuator_actrange,
                    m.actuator_actearly,
                    m.actuator_forcelimited,
                    m.actuator_forcerange,
                    m.actuator_acc0,
                    m.actuator_lengthrange,
                    d.act,
                    d.ctrl,
                    d.act_dot,
                    d.actuator_length,
                    d.actuator_velocity,
                    d.actuator_force,
                ],
                outputs=[vel],
            )
            if use_m_structure:
                wp.launch(
                    _qderiv_actuator_passive_actuation_m_structure_actuator_centric,
                    dim=(
                        d.nworld,
                        m.nactuator,
                        scratch.implicit_max_moment_rownnz,
                        scratch.implicit_max_moment_rownnz,
                    ),
                    inputs=[
                        m.M_elemid,
                        d.moment_rownnz,
                        d.moment_rowadr,
                        d.moment_colind,
                        d.actuator_moment,
                        vel,
                    ],
                    outputs=[out],
                )
            else:
                pair_mask = scratch.implicit_qderiv_pair_mask
                if pair_mask is None:
                    msg = (
                        "optimized implicit derivative scratch is missing "
                        "dense pair mask"
                    )
                    raise RuntimeError(msg)
                wp.launch(
                    _qderiv_actuator_passive_actuation_dense_actuator_centric,
                    dim=(
                        d.nworld,
                        m.nactuator,
                        scratch.implicit_max_moment_rownnz,
                        scratch.implicit_max_moment_rownnz,
                    ),
                    inputs=[
                        d.moment_rownnz,
                        d.moment_rowadr,
                        d.moment_colind,
                        d.actuator_moment,
                        vel,
                        pair_mask,
                    ],
                    outputs=[out],
                )
        if use_m_structure:
            wp.launch(
                derivative._qderiv_actuator_passive,
                dim=(d.nworld, fullm_i.size),
                inputs=[
                    m.opt.timestep,
                    m.opt.disableflags,
                    m.dof_damping,
                    m.dof_dampingpoly,
                    m.M_elemid,
                    d.qvel,
                    d.M,
                    fullm_i,
                    fullm_j,
                    out,
                ],
                outputs=[out],
            )
        else:
            wp.launch(
                derivative._qderiv_actuator_passive,
                dim=(d.nworld, fullm_i.size),
                inputs=[
                    m.opt.timestep,
                    m.opt.disableflags,
                    m.dof_damping,
                    m.dof_dampingpoly,
                    m.is_sparse,
                    d.qvel,
                    d.qM,
                    fullm_i,
                    fullm_j,
                    out,
                ],
                outputs=[out],
            )
    else:
        wp.copy(out, _inertia_matrix(d))

    if getattr(m, "ntendon", 0) > 0 and not (m.opt.disableflags & DisableBit.DAMPER):
        if use_m_structure:
            wp.launch(
                derivative._qderiv_tendon_damping,
                dim=(d.nworld, fullm_i.size),
                inputs=[
                    m.ntendon,
                    m.opt.timestep,
                    m.ten_J_rownnz,
                    m.ten_J_rowadr,
                    m.ten_J_colind,
                    m.tendon_damping,
                    m.tendon_dampingpoly,
                    m.M_elemid,
                    d.ten_J,
                    d.ten_velocity,
                    fullm_i,
                    fullm_j,
                ],
                outputs=[out],
            )

        else:
            wp.launch(
                derivative._qderiv_tendon_damping,
                dim=(d.nworld, fullm_i.size),
                inputs=[
                    m.ntendon,
                    m.opt.timestep,
                    m.ten_J_rownnz,
                    m.ten_J_rowadr,
                    m.ten_J_colind,
                    m.tendon_damping,
                    m.tendon_dampingpoly,
                    m.is_sparse,
                    d.ten_J,
                    d.ten_velocity,
                    fullm_i,
                    fullm_j,
                ],
                outputs=[out],
            )

    # Native fluid terms follow the actuator-centric assembly without allocations.
    if m.has_fluid:
        wp.launch(
            derivative._qderiv_ellipsoid_fluid,
            dim=(d.nworld, m.body_fluid_ellipsoid_adr.size, fullm_i.size),
            inputs=[
                m.opt.timestep,
                m.opt.wind,
                m.opt.density,
                m.opt.viscosity,
                m.opt.integrator,
                m.body_parentid,
                m.body_rootid,
                m.body_geomnum,
                m.body_geomadr,
                m.dof_bodyid,
                m.geom_type,
                m.geom_size,
                m.geom_fluid,
                m.body_fluid_ellipsoid_adr,
                m.body_isdofancestor,
                m.M_elemid,
                d.xipos,
                d.geom_xpos,
                d.geom_xmat,
                d.subtree_com,
                d.cdof,
                d.cvel,
                fullm_i,
                fullm_j,
            ],
            outputs=[out],
        )
        wp.launch(
            derivative._qderiv_box_fluid,
            dim=(d.nworld, m.body_fluid_box_adr.size, fullm_i.size),
            inputs=[
                m.opt.timestep,
                m.opt.wind,
                m.opt.density,
                m.opt.viscosity,
                m.opt.integrator,
                m.body_parentid,
                m.body_rootid,
                m.body_mass,
                m.body_inertia,
                m.dof_bodyid,
                m.body_fluid_box_adr,
                m.body_isdofancestor,
                m.M_elemid,
                d.xipos,
                d.ximat,
                d.subtree_com,
                d.cdof,
                d.cvel,
                fullm_i,
                fullm_j,
            ],
            outputs=[out],
        )

    _prepare_implicit_qh(m, d, scratch, out)


@event_scope
def fwd_velocity_optimized(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    start_actuation: bool = False,
    defer_actuator_limits: bool = False,
    use_optimized_kernel: bool | None = None,
    skip_com_vel: bool = False,
) -> OptimizedFwdVelocityResult:
    """MuJoCo Warp velocity pass with independent velocity/passive branches on streams."""
    if streams is None:
        mjw_forward.fwd_velocity(m, d)
        return OptimizedFwdVelocityResult()

    active_streams: list[wp.Stream] = []
    actuator_velocity_stream = _stream_or_none(streams, ACTUATOR_VELOCITY_STREAM)
    tendon_velocity_stream = _stream_or_none(streams, TENDON_VELOCITY_STREAM)
    passive_stream = _stream_or_none(streams, PASSIVE_STREAM)
    tendon_bias_stream = _stream_or_none(streams, TENDON_BIAS_STREAM)
    actuation_started = False

    _launch_in_stream(
        actuator_velocity_stream,
        mjw_forward._actuator_velocity,
        dim=(d.nworld, m.nactuator),
        inputs=[
            d.qvel,
            d.moment_rownnz,
            d.moment_rowadr,
            d.moment_colind,
            d.actuator_moment,
        ],
        outputs=[d.actuator_velocity],
    )
    if start_actuation:
        _start_fwd_actuation_no_tendon(
            actuator_velocity_stream,
            m,
            d,
            apply_limits=not defer_actuator_limits,
        )
        actuation_started = True
    else:
        _track_stream_once(active_streams, actuator_velocity_stream)

    _launch_in_stream(
        tendon_velocity_stream,
        mjw_forward._tendon_velocity,
        dim=(d.nworld, m.ntendon),
        inputs=[m.ten_J_rownnz, m.ten_J_rowadr, m.ten_J_colind, d.qvel, d.ten_J],
        outputs=[d.ten_velocity],
    )
    _track_stream_once(active_streams, tendon_velocity_stream)

    passive_started = False
    passive_call = (
        getattr(scratch, "passive_call", None) if scratch is not None else None
    )
    if passive_call is None:
        passive_call = passive.passive
    passive_wait_streams = [tendon_velocity_stream] if getattr(m, "ntendon", 0) else []
    passive_can_overlap = not getattr(m, "has_fluid", False) and not _callback_enabled(
        m, "passive"
    )
    if passive_can_overlap:
        _call_in_stream_after(
            passive_stream,
            passive_wait_streams,
            passive_call,
            m,
            d,
        )
        _track_stream_once(active_streams, passive_stream)
        passive_started = True

    if not skip_com_vel:
        smooth.com_vel(m, d)

    if not passive_started:
        if getattr(m, "ntendon", 0) and tendon_velocity_stream is not None:
            wp.wait_stream(tendon_velocity_stream)
        _call_in_stream(passive_stream, passive_call, m, d)
        _track_stream_once(active_streams, passive_stream)

    tendon_bias_started = _has_tendon_bias_scratch(m, scratch)
    if tendon_bias_started:
        _call_in_stream(
            tendon_bias_stream,
            tendon_bias_prepare_with_scratch,
            m,
            d,
            scratch,
        )

    tree_reduction = _select_tree_reduction(
        scratch, _resolve_use_optimized_kernel(scratch, use_optimized_kernel)
    )
    tree_fusion.rne(m, d, tree_reduction)
    if tendon_bias_started:
        if tendon_bias_stream is not None:
            wp.wait_stream(tendon_bias_stream)
        tendon_bias_apply_with_scratch(m, d, d.qfrc_bias, scratch)
    elif getattr(m, "ntendon", 0) > 0:
        smooth.tendon_bias(m, d, d.qfrc_bias)
    _wait_for_streams(active_streams)
    return OptimizedFwdVelocityResult(
        actuation_started=actuation_started,
        actuation_stream=actuator_velocity_stream,
        actuator_limits_pending=actuation_started and defer_actuator_limits,
    )


def _can_pipeline_position_velocity(
    m: Model,
    streams: Sequence[wp.Stream] | None,
) -> bool:
    """Return whether the velocity core is independent of the position tail."""
    return bool(
        streams
        and len(streams) > POSITION_VELOCITY_STREAM
        and not _position_sleep_enabled(m)
        and not _callback_enabled(m, "sensor")
    )


def _constraint_reads_com_velocity(m: Model) -> bool:
    """Return whether constraint construction consumes cvel/cdof_dot.

    MuJoCo Warp's connect and weld equality kernels consume these arrays.
    Other built-in constraint kernels do not, but a contact-filter callback may
    inspect arbitrary Data fields while collision and COM velocity overlap.
    """
    disableflags = getattr(m.opt, "disableflags", 0)
    if _callback_enabled(m, "contactfilter"):
        return True
    if disableflags & (DisableBit.CONSTRAINT | DisableBit.EQUALITY):
        return False
    return any(
        getattr(getattr(m, name, None), "size", 0) > 0
        for name in ("eq_connect_adr", "eq_wld_adr")
    )


def _start_pipelined_com_velocity(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream],
    tree_reduction=None,
) -> PendingComVelocity:
    stream = _stream_or_none(streams, POSITION_VELOCITY_STREAM)
    if stream is None:
        return PendingComVelocity()

    with wp.ScopedStream(stream, sync_enter=True, sync_exit=False):
        velocity_fusion.com_vel(m, d, tree_reduction)
    return PendingComVelocity(started=True, stream=stream)


def _forward_state_prefix_after_position(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None = None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
    start_actuation: bool,
    pending_velocity: PendingComVelocity | None = None,
) -> OptimizedFwdVelocityResult:
    """Refresh sensor-position and velocity-stage state after fwd_position."""
    use_optimized_kernel = _resolve_use_optimized_kernel(
        scratch,
        use_optimized_kernel,
    )
    energy = m.opt.enableflags & EnableBit.ENERGY

    d.sensordata.zero_()
    sensor_pos_call = (
        getattr(scratch, "sensor_pos_call", None) if scratch is not None else None
    )
    if sensor_pos_call is None:
        sensor.sensor_pos(m, d)
    else:
        sensor_pos_call(m, d)
    if energy:
        if m.sensor_e_potential == 0:
            sensor.energy_pos(m, d)
    else:
        d.energy.zero_()

    velocity_result = OptimizedFwdVelocityResult()
    if pending_velocity is not None and pending_velocity.started:
        if pending_velocity.stream is not None:
            wp.wait_stream(pending_velocity.stream)
        velocity_result = fwd_velocity_optimized(
            m,
            d,
            streams,
            scratch=scratch,
            start_actuation=start_actuation and _can_start_actuation_early(m),
            defer_actuator_limits=use_optimized_kernel and start_actuation,
            use_optimized_kernel=use_optimized_kernel,
            skip_com_vel=True,
        )
    elif streams is None:
        mjw_forward.fwd_velocity(m, d)
    else:
        velocity_result = fwd_velocity_optimized(
            m,
            d,
            streams,
            scratch=scratch,
            start_actuation=start_actuation and _can_start_actuation_early(m),
            defer_actuator_limits=use_optimized_kernel and start_actuation,
            use_optimized_kernel=use_optimized_kernel,
        )
    sensor_vel_call = (
        getattr(scratch, "sensor_vel_call", None) if scratch is not None else None
    )
    if sensor_vel_call is None:
        sensor.sensor_vel(m, d)
    else:
        sensor_vel_call(m, d)

    if energy and m.sensor_e_kinetic == 0:
        energy_vel_call = (
            getattr(scratch, "energy_vel_call", None) if scratch is not None else None
        )
        if energy_vel_call is None:
            sensor.energy_vel(m, d)
        else:
            energy_vel_call(m, d)

    return velocity_result


def _forward_action_suffix(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None,
    velocity_result: OptimizedFwdVelocityResult,
    *,
    use_optimized_kernel: bool | None,
) -> OptimizedImplicitDerivResult:
    """Evaluate action-dependent forces, acceleration, constraints, and sensors."""
    use_optimized_kernel = _resolve_use_optimized_kernel(
        scratch,
        use_optimized_kernel,
    )
    optimized_scratch = scratch if use_optimized_kernel else None

    if not (m.opt.disableflags & DisableBit.ACTUATION) and m.callback.control:
        m.callback.control(m, d)
    if not velocity_result.actuation_started:
        actuation_call = (
            getattr(scratch, "actuation_call", None) if scratch is not None else None
        )
        if actuation_call is None:
            mjw_forward.fwd_actuation(m, d)
        else:
            actuation_call(m, d)
    elif velocity_result.actuation_stream is not None:
        wp.wait_stream(velocity_result.actuation_stream)
    if streams is None or not use_optimized_kernel:
        mjw_forward.fwd_acceleration(m, d, factorize=True)
    else:
        fwd_acceleration_optimized(
            m,
            d,
            factorize=True,
            actuator_limits_pending=velocity_result.actuator_limits_pending,
            dense_factor_solve=_dense_factor_solve_scratch(optimized_scratch),
            tree_factor_solve=_tree_factor_solve_scratch(optimized_scratch),
        )

    pending_implicit_deriv = _start_optimized_implicit_deriv_before_solve(
        m,
        d,
        streams,
        optimized_scratch,
        use_optimized_kernel=use_optimized_kernel,
    )
    if scratch is not None and (
        getattr(scratch, "constraint_free", False)
        or d.njmax == 0
        or m.nv == 0
        or (
            getattr(scratch, "solver_context", None) is not None
            and getattr(scratch, "solver_nsolving", None) is not None
        )
    ):
        solve_with_scratch(m, d, scratch, use_optimized_kernel=use_optimized_kernel)
    else:
        solver.solve(m, d)
    sensor_acc_call = (
        getattr(scratch, "sensor_acc_call", None) if scratch is not None else None
    )
    if getattr(m.opt, "run_rne_postconstraint", False) or (
        not (m.opt.disableflags & DisableBit.SENSOR)
        and getattr(m, "sensor_rne_postconstraint", False)
    ):
        rne_call = getattr(scratch, "rne_postconstraint_call", None)
        (rne_call or smooth.rne_postconstraint)(m, d)
    if sensor_acc_call is None:
        sensor.sensor_acc(m, d, skip_rne_postconstraint=True)
    else:
        sensor_acc_call(m, d, skip_rne_postconstraint=True)
    return pending_implicit_deriv


def _forward_after_state_prefix(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None = None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
) -> OptimizedImplicitDerivResult:
    """Finish forward dynamics when position and velocity state is already fresh."""
    return _forward_action_suffix(
        m,
        d,
        streams,
        scratch,
        OptimizedFwdVelocityResult(),
        use_optimized_kernel=use_optimized_kernel,
    )


@event_scope
def forward_after_state_prefix_optimized_fwd_position(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None = None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
) -> OptimizedImplicitDerivResult:
    """Evaluate action-dependent dynamics from an already-fresh state prefix."""
    return _forward_after_state_prefix(
        m,
        d,
        streams,
        scratch=scratch,
        use_optimized_kernel=use_optimized_kernel,
    )


def _forward_after_position(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None = None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
    pending_velocity: PendingComVelocity | None = None,
) -> OptimizedImplicitDerivResult:
    velocity_result = _forward_state_prefix_after_position(
        m,
        d,
        streams,
        scratch=scratch,
        use_optimized_kernel=use_optimized_kernel,
        start_actuation=True,
        pending_velocity=pending_velocity,
    )
    return _forward_action_suffix(
        m,
        d,
        streams,
        scratch,
        velocity_result,
        use_optimized_kernel=use_optimized_kernel,
    )


def _has_implicit_factor_scratch(
    m: Model,
    scratch: OptimizedFwdPositionScratch | None,
) -> bool:
    """Whether implicit-fast can factor ``M - h*D`` with a repo-owned solve.

    That is the blocked dense kernel, or the generated tree L'DL solve when no
    factor is retained. The tree solve also covers inertia split into several
    trees (for example a body and free-floating objects), which has no single
    dense tile for the blocked kernel.

    Upstream ``deriv_smooth_vel`` allocates an actuator-velocity temporary when
    ``nu > 0``; actuated models keep the actuator-centric derivative path, which
    already routes its factor+solve through these kernels.
    """
    if scratch is None:
        return False
    if getattr(m, "nactuator", 0) != 0:
        return False
    factor = _dense_factor_solve_scratch(scratch) is not None or (
        _tree_factor_solve_scratch(scratch) is not None
        and _implicit_factor_workspace(scratch) is None
    )
    return (
        factor
        and getattr(scratch, "implicit_qderiv", None) is not None
        and getattr(scratch, "implicit_qacc", None) is not None
    )


def _advance_with_scratch(
    m: Model,
    d: Data,
    qacc: wp.array,
    scratch: OptimizedFwdPositionScratch | None,
    qvel: wp.array | None = None,
) -> None:
    modifier = (
        getattr(scratch, "pre_advance_modifier", None) if scratch is not None else None
    )
    if modifier is not None:
        qacc, qvel = modifier.apply(m, d, qacc, qvel)
    advance_call = (
        getattr(scratch, "advance_call", None) if scratch is not None else None
    )
    if advance_call is None:
        if qvel is None:
            mjw_forward._advance(m, d, qacc)
        else:
            mjw_forward._advance(m, d, qacc, qvel)
    else:
        if qvel is None:
            advance_call(m, d, qacc)
        else:
            advance_call(m, d, qacc, qvel)


@wp.kernel(enable_backward=False)
def _rk_stage_time(
    timestep: wp.array[float],
    time_t0: wp.array[float],
    scale: float,
    time: wp.array[float],
):
    world = wp.tid()
    time[world] = time_t0[world] + scale * timestep[world % timestep.shape[0]]


@event_scope
def rungekutta4_with_scratch(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch,
    *,
    use_optimized_kernel: bool,
) -> None:
    """Run 3.16 RK4 with persistent buffers and repo-owned nested forwards."""
    rk4 = scratch.rk4
    if rk4 is None:
        raise RuntimeError("RK4 scratch is not initialized")

    wp.copy(rk4.qpos_t0, d.qpos)
    wp.copy(rk4.qvel_t0, d.qvel)
    wp.copy(rk4.time_t0, d.time)
    rk4.qvel_rk.zero_()
    rk4.qacc_rk.zero_()
    if m.na:
        if rk4.act_t0 is None or rk4.act_dot_rk is None:
            raise RuntimeError("RK4 activation scratch is not initialized")
        wp.copy(rk4.act_t0, d.act)
        rk4.act_dot_rk.zero_()

    weights = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)
    perturb = (0.5, 0.5, 1.0)
    mjw_forward._rk_accumulate(
        m,
        d,
        weights[0],
        rk4.qvel_rk,
        rk4.qacc_rk,
        rk4.act_dot_rk,
    )
    for index in range(3):
        mjw_forward._rk_perturb_state(
            m,
            d,
            perturb[index],
            rk4.qpos_t0,
            rk4.qvel_t0,
            rk4.act_t0,
        )
        wp.launch(
            _rk_stage_time,
            dim=d.nworld,
            inputs=[m.opt.timestep, rk4.time_t0, perturb[index], d.time],
        )
        forward_optimized_fwd_position(
            m,
            d,
            streams,
            scratch=scratch,
            use_optimized_kernel=use_optimized_kernel,
        )
        mjw_forward._rk_accumulate(
            m,
            d,
            weights[index + 1],
            rk4.qvel_rk,
            rk4.qacc_rk,
            rk4.act_dot_rk,
        )

    wp.copy(d.qpos, rk4.qpos_t0)
    wp.copy(d.qvel, rk4.qvel_t0)
    wp.copy(d.time, rk4.time_t0)
    if m.na:
        wp.copy(d.act, rk4.act_t0)
        wp.copy(d.act_dot, rk4.act_dot_rk)
    _advance_with_scratch(m, d, rk4.qacc_rk, scratch, rk4.qvel_rk)


@event_scope
def implicit_factor_optimized(
    m: Model,
    d: Data,
    scratch: OptimizedFwdPositionScratch,
) -> None:
    """Implicit-fast integration using the repo-owned factor+solve.

    Mirrors upstream ``forward.implicit`` for the implicit-fast branch while
    reusing the persistent qDeriv/qacc scratch and the tree L'DL or blocked
    Cholesky solve (``_factor_solve_i_dense``).
    """
    qderiv = scratch.implicit_qderiv
    qacc = scratch.implicit_qacc
    if qderiv is None or qacc is None:
        msg = "implicit factor integration requires qDeriv/qacc scratch"
        raise RuntimeError(msg)
    deriv_call = scratch.implicit_deriv_call
    if deriv_call is None:
        deriv_call = deriv_smooth_vel
    deriv_call(m, d, qderiv)
    _prepare_implicit_qh(m, d, scratch, qderiv)
    _factor_solve_i_dense(
        m,
        d,
        qderiv,
        qacc,
        d.efc.Ma,
        _dense_factor_solve_scratch(scratch),
        factor_workspace=_implicit_factor_workspace(scratch),
        tree_factor_solve=_tree_factor_solve_scratch(scratch),
    )
    _implicit_free_body_correction(m, d, qacc, scratch)
    _advance_with_scratch(m, d, qacc, scratch)


@event_scope
def implicit_dense_deriv_optimized(
    m: Model,
    d: Data,
    scratch: OptimizedFwdPositionScratch,
) -> None:
    """Implicit-fast integrator using actuator-centric dense velocity derivative."""
    if not _has_implicit_dense_deriv_scratch(m, scratch):
        _reject_unretained_implicit_factor(scratch)
        _implicit(m, d)
        return

    if _implicitfast_needs_deriv(m, scratch):
        qDeriv = getattr(scratch, "implicit_qderiv", None)
        if qDeriv is None:
            qDeriv = wp.empty(_inertia_matrix(d).shape, dtype=float)
        deriv_smooth_vel_dense_actuator_centric(m, d, qDeriv, scratch)
        qacc = getattr(scratch, "implicit_qacc", None)
        if qacc is None:
            qacc = wp.empty((d.nworld, m.nv), dtype=float)
        _factor_solve_i_dense(
            m,
            d,
            qDeriv,
            qacc,
            d.efc.Ma,
            _dense_factor_solve_scratch(scratch),
            factor_workspace=_implicit_factor_workspace(scratch),
            tree_factor_solve=_tree_factor_solve_scratch(scratch),
        )
        _implicit_free_body_correction(m, d, qacc, scratch)
        _advance_with_scratch(m, d, qacc, scratch)
    else:
        _reject_unretained_implicit_factor(scratch)
        _advance_with_scratch(m, d, d.qacc, scratch)


@event_scope
def implicitfast_preallocated_with_scratch(
    m: Model,
    d: Data,
    scratch: OptimizedFwdPositionScratch,
) -> None:
    """Run implicit-fast through persistent scratch and modifier-aware advance."""
    if not _has_preallocated_implicitfast_scratch(scratch):
        raise RuntimeError("preallocated implicit-fast scratch is incomplete")
    _reject_unretained_implicit_factor(scratch)
    qderiv = scratch.implicit_qderiv
    qld = scratch.implicit_qld
    diag = scratch.implicit_ql_diag_inv
    qacc = scratch.implicit_qacc
    deriv_call = scratch.implicit_deriv_call
    deriv_call(m, d, qderiv)
    _prepare_implicit_qh(m, d, scratch, qderiv)
    factor_solve_i_no_l_store(
        m,
        d,
        qderiv,
        qld,
        diag,
        qacc,
        d.efc.Ma,
        tree_factor_solve=_tree_factor_solve_scratch(scratch),
    )
    _implicit_free_body_correction(m, d, qacc, scratch)
    _advance_with_scratch(m, d, qacc, scratch)


@event_scope
def forward_optimized_fwd_position(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
):
    """Run the repo-owned forward path, reusing scratch when it is supplied.

    Pass an empty stream sequence for sequential custom scheduling.  ``None`` is
    retained only for utility-level compatibility and can enter upstream helpers
    that allocate, so production DirectRLEnv callers must not pass it.
    """
    use_optimized_kernel = _resolve_use_optimized_kernel(
        scratch,
        use_optimized_kernel,
    )
    _forward_sleep_wake(m, d)
    pending_velocity = fwd_position_optimized(
        m,
        d,
        streams,
        scratch=scratch,
        factorize=False,
        use_optimized_kernel=use_optimized_kernel,
        pipeline_velocity=True,
    )
    return _forward_after_position(
        m,
        d,
        streams,
        scratch=scratch,
        use_optimized_kernel=use_optimized_kernel,
        pending_velocity=pending_velocity,
    )


@event_scope
def kinematics_com_state_optimized_fwd_position(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
) -> None:
    """Kinematics, COM position and COM velocity with the state prefix's kernels
    (fused body frames, the selected tree reduction), without transmission,
    collision, constraints or the rest of the velocity stage: for state edits
    that read body poses and velocities before one complete prefix.

    The result equals the prefix's only within float tolerance in general: the
    prefix uses the stock COM-velocity kernel when COM velocity is not pipelined,
    and a non-deterministic tree reduction accumulates with atomics. A caller
    that must leave the prefix's state bitwise unchanged copies the fields back
    instead of rerunning this pass."""
    use_optimized_kernel = _resolve_use_optimized_kernel(scratch, use_optimized_kernel)
    tree_reduction = _select_tree_reduction(scratch, use_optimized_kernel)
    tails = None
    if streams is None:
        kinematics_parallel_tail(m, d, None)
    else:
        tails = _start_kinematics_parallel_tail(
            m,
            d,
            streams[:4],
            fuse_body_frames=use_optimized_kernel and bool(streams),
        )
    com_pos_parallel_tail(
        m, d, streams, kinematics_tail_streams=tails, tree_reduction=tree_reduction
    )
    if tails is not None:
        _wait_for_streams(tails.as_tuple())
    velocity_fusion.com_vel(m, d, tree_reduction)


def state_prefix_can_defer_constraints(m: Model) -> bool:
    """Whether a state prefix may leave constraint construction to a later one
    (``forward_state_prefix_optimized_fwd_position(constraints=False)``):
    islands and sleep read the rows, and position/velocity sensors (touch,
    collision distance, limits) may read contacts or rows."""
    return not _position_sleep_enabled(m) and int(getattr(m, "nsensor", 0)) == 0


@event_scope
def forward_state_prefix_optimized_fwd_position(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
    constraints: bool = True,
) -> None:
    """Refresh all action-independent position and velocity stage fields.

    ``constraints=False`` leaves contacts and constraint rows as they were (see
    ``fwd_position_optimized``); the caller must rebuild them before a solve.
    """
    use_optimized_kernel = _resolve_use_optimized_kernel(
        scratch,
        use_optimized_kernel,
    )
    _forward_sleep_wake(m, d)
    pending_velocity = fwd_position_optimized(
        m,
        d,
        streams,
        scratch=scratch,
        factorize=False,
        use_optimized_kernel=use_optimized_kernel,
        pipeline_velocity=True,
        constraints=constraints,
    )
    _forward_state_prefix_after_position(
        m,
        d,
        streams,
        scratch=scratch,
        use_optimized_kernel=use_optimized_kernel,
        start_actuation=False,
        pending_velocity=pending_velocity,
    )


@event_scope
def forward_skip_kinematics_optimized_fwd_position(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
):
    """MuJoCo Warp forward dynamics with fresh kinematics and optimized fwd_position."""
    use_optimized_kernel = _resolve_use_optimized_kernel(
        scratch,
        use_optimized_kernel,
    )
    _forward_sleep_wake(m, d)
    pending_velocity = fwd_position_optimized(
        m,
        d,
        streams,
        scratch=scratch,
        factorize=False,
        skip_kinematics=True,
        use_optimized_kernel=use_optimized_kernel,
        pipeline_velocity=True,
    )
    return _forward_after_position(
        m,
        d,
        streams,
        scratch=scratch,
        use_optimized_kernel=use_optimized_kernel,
        pending_velocity=pending_velocity,
    )


@event_scope
def forward_skip_kinematics(m: Model, d: Data):
    """MuJoCo Warp forward dynamics when kinematics already matches qpos."""
    _forward_sleep_wake(m, d)
    fwd_position_skip_kinematics(m, d, factorize=False)
    _forward_after_position(m, d)


def _integrate(
    m: Model,
    d: Data,
    scratch: OptimizedFwdPositionScratch | None = None,
    streams: Sequence[wp.Stream] | None = None,
    *,
    use_optimized_kernel: bool | None = None,
    pending_implicit_deriv: OptimizedImplicitDerivResult | None = None,
) -> None:
    use_optimized_kernel = _resolve_use_optimized_kernel(
        scratch,
        use_optimized_kernel,
    )
    if m.opt.integrator != IntegratorType.IMPLICITFAST:
        _reject_unretained_implicit_factor(scratch)
    if m.opt.integrator == IntegratorType.EULER:
        if scratch is not None and scratch.euler_call is not None:
            scratch.euler_call(m, d)
        else:
            mjw_forward.euler(m, d)
    elif m.opt.integrator == IntegratorType.RK4:
        if scratch is not None and scratch.rk4 is not None:
            rungekutta4_with_scratch(
                m,
                d,
                streams,
                scratch,
                use_optimized_kernel=use_optimized_kernel,
            )
        else:
            mjw_forward.rungekutta4(m, d)
    elif m.opt.integrator in (IntegratorType.IMPLICITFAST, IntegratorType.IMPLICIT):
        if m.opt.integrator == IntegratorType.IMPLICIT:
            if scratch is not None and scratch.implicit_call is not None:
                scratch.implicit_call(m, d)
            else:
                _implicit(m, d)
            return
        if (
            use_optimized_kernel
            and pending_implicit_deriv is not None
            and pending_implicit_deriv.started
        ):
            if pending_implicit_deriv.stream is not None:
                wp.wait_stream(pending_implicit_deriv.stream)
            qderiv = pending_implicit_deriv.qderiv
            qacc = pending_implicit_deriv.qacc
            if qderiv is None or qacc is None:
                msg = "precomputed implicit derivative is missing qDeriv/qacc buffers"
                raise RuntimeError(msg)
            _factor_solve_i_dense(
                m,
                d,
                qderiv,
                qacc,
                d.efc.Ma,
                _dense_factor_solve_scratch(scratch),
                factor_workspace=_implicit_factor_workspace(scratch),
                tree_factor_solve=_tree_factor_solve_scratch(scratch),
            )
            _implicit_free_body_correction(m, d, qacc, scratch)
            _advance_with_scratch(m, d, qacc, scratch)
        elif use_optimized_kernel and _has_implicit_dense_deriv_scratch(m, scratch):
            implicit_dense_deriv_optimized(m, d, scratch)
        elif (
            use_optimized_kernel
            and _has_implicit_factor_scratch(m, scratch)
            and _implicitfast_needs_deriv(m, scratch)
        ):
            implicit_factor_optimized(m, d, scratch)
        elif _has_preallocated_implicit_contributor_scratch(m, scratch):
            implicitfast_preallocated_with_scratch(m, d, scratch)
        elif (
            scratch is not None
            and getattr(scratch, "pre_advance_modifier", None) is not None
        ):
            if _implicitfast_needs_deriv(m, scratch):
                implicitfast_preallocated_with_scratch(m, d, scratch)
            else:
                _reject_unretained_implicit_factor(scratch)
                _advance_with_scratch(m, d, d.qacc, scratch)
        elif scratch is not None and scratch.implicit_call is not None:
            _reject_unretained_implicit_factor(scratch)
            scratch.implicit_call(m, d)
        else:
            _reject_unretained_implicit_factor(scratch)
            _implicit(m, d)
    else:
        raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")


@event_scope
def step_optimized_fwd_position(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
):
    """Advance through the repo-owned path, reusing scratch when supplied.

    Production DirectRLEnv callers always supply scratch and pass either the
    configured side streams or an empty sequence for sequential execution.
    """
    if scratch is not None and hasattr(scratch, "implicit_qh_registration_frozen"):
        scratch.implicit_qh_registration_frozen = True
    pending_implicit_deriv = forward_optimized_fwd_position(
        m,
        d,
        streams,
        scratch=scratch,
        use_optimized_kernel=use_optimized_kernel,
    )
    _integrate(
        m,
        d,
        scratch=scratch,
        streams=streams,
        use_optimized_kernel=use_optimized_kernel,
        pending_implicit_deriv=pending_implicit_deriv,
    )


@event_scope
def step_skip_kinematics_optimized_fwd_position(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
):
    """Advance simulation with fresh kinematics and optimized fwd_position."""
    pending_implicit_deriv = forward_skip_kinematics_optimized_fwd_position(
        m,
        d,
        streams,
        scratch=scratch,
        use_optimized_kernel=use_optimized_kernel,
    )
    _integrate(
        m,
        d,
        scratch=scratch,
        streams=streams,
        use_optimized_kernel=use_optimized_kernel,
        pending_implicit_deriv=pending_implicit_deriv,
    )


@event_scope
def step_after_state_prefix_optimized_fwd_position(
    m: Model,
    d: Data,
    streams: Sequence[wp.Stream] | None,
    scratch: OptimizedFwdPositionScratch | None = None,
    *,
    use_optimized_kernel: bool | None = None,
):
    """Apply action-dependent dynamics and integrate from a fresh state prefix."""
    pending_implicit_deriv = _forward_after_state_prefix(
        m,
        d,
        streams,
        scratch=scratch,
        use_optimized_kernel=use_optimized_kernel,
    )
    _integrate(
        m,
        d,
        scratch=scratch,
        streams=streams,
        use_optimized_kernel=use_optimized_kernel,
        pending_implicit_deriv=pending_implicit_deriv,
    )


@event_scope
def step_skip_kinematics(m: Model, d: Data):
    """Advance simulation while reusing already-fresh kinematics."""
    forward_skip_kinematics(m, d)
    _integrate(m, d)
