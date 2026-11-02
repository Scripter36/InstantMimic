"""Directional ball limit rows for the pinned MuJoCo Warp constraint solver."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from instantmimic.utils.ball_joint_limits import ResolvedBallJointLimit
from instantmimic.utils.mjw_kernels.private import cache_kernel, constraint, types


@wp.func
def _mul(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    av = wp.vec3(a[1], a[2], a[3])
    bv = wp.vec3(b[1], b[2], b[3])
    v = a[0] * bv + b[0] * av + wp.cross(av, bv)
    return wp.vec4(a[0] * b[0] - wp.dot(av, bv), v[0], v[1], v[2])


@wp.func
def _conj(q: wp.vec4) -> wp.vec4:
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def _jac(q: wp.vec4, g: wp.vec4) -> wp.vec3:
    v = wp.vec3(q[1], q[2], q[3])
    gv = wp.vec3(g[1], g[2], g[3])
    return 0.5 * (-g[0] * v + q[0] * gv - wp.cross(v, gv))


@cache_kernel
def make_ball_limit_kernel(is_sparse: bool, newton: bool):
    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        nv: int,
        timestep: wp.array[float],
        disableflags: int,
        qadr: wp.array[int],
        dadr: wp.array[int],
        joint: wp.array[int],
        neutral: wp.array[wp.vec4],
        frame: wp.array[wp.vec4],
        swing: wp.array[wp.vec4],
        twist: wp.array[wp.vec2],
        minimum_h: wp.array[float],
        margin: wp.array[float],
        solref: wp.array[wp.vec2],
        solimp: wp.array[types.vec5],
        invweight: wp.array2d[float],
        qpos: wp.array2d[float],
        qvel: wp.array2d[float],
        njmax: int,
        njmax_nnz: int,
        nl: wp.array[int],
        nefc: wp.array[int],
        efc_type: wp.array2d[int],
        efc_id: wp.array2d[int],
        jtdaj_adr: wp.array2d[int],
        jtdaj_nrow: wp.array2d[int],
        jtdaj_nblock: wp.array[int],
        rownnz: wp.array2d[int],
        rowadr: wp.array2d[int],
        colind: wp.array3d[int],
        efc_J: wp.array3d[float],
        efc_pos: wp.array2d[float],
        efc_margin: wp.array2d[float],
        efc_D: wp.array2d[float],
        efc_vel: wp.array2d[float],
        efc_aref: wp.array2d[float],
        efc_frictionloss: wp.array2d[float],
        nnz: wp.array[int],
        overflow: wp.array[int],
        singular: wp.array[int],
    ):
        world, index, kind = wp.tid()
        adr = qadr[index]
        original = wp.normalize(
            wp.vec4(
                qpos[world, adr],
                qpos[world, adr + 1],
                qpos[world, adr + 2],
                qpos[world, adr + 3],
            )
        )
        f = frame[index]
        q = _mul(_mul(_conj(f), _mul(_conj(neutral[index]), original)), f)
        pivot = int(0)
        if wp.abs(q[0]) < 1.0e-15:
            for k in range(1, 4):
                if wp.abs(q[k]) > wp.abs(q[pivot]):
                    pivot = k
        if q[pivot] < 0.0:
            q = -q
        w, x, y, z = q[0], q[1], q[2], q[3]
        h = w * w + z * z
        # Do not emit zero-J rows at the antipode. The native scalar safety
        # constraint restores this off-domain state; expose diagnostic counts.
        if h < 0.25 * minimum_h[index] or wp.abs(w) < 0.5 * wp.sqrt(minimum_h[index]):
            wp.atomic_add(singular, world, 1)
            return
        sx, sy = w * x - z * y, w * y + z * x
        g = wp.vec4(0.0)
        r = float(0.0)
        if kind == 0:
            ax, ay = swing[index][0], swing[index][1]
            if sx < 0.0:
                ax = swing[index][2]
            if sy < 0.0:
                ay = swing[index][3]
            e = ax * sx * sx + ay * sy * sy
            dh = wp.vec4(2.0 * w, 0.0, 0.0, 2.0 * z)
            de = wp.vec4(
                2.0 * ax * sx * x + 2.0 * ay * sy * y,
                2.0 * ax * sx * w + 2.0 * ay * sy * z,
                -2.0 * ax * sx * z + 2.0 * ay * sy * w,
                -2.0 * ax * sx * y + 2.0 * ay * sy * x,
            )
            r = 1.0 - e / (h * h)
            g = -de / (h * h) + 2.0 * e * dh / (h * h * h)
        else:
            theta = 2.0 * wp.atan2(z, w)
            g = wp.vec4(-2.0 * z / h, 0.0, 0.0, 2.0 * w / h)
            if kind == 1:
                r = theta - twist[index][0]
            else:
                r = twist[index][1] - theta
                g = -g
        pos = r - margin[index]
        if pos >= 0.0:
            return
        J = _jac(q, g)
        fv = wp.vec3(f[1], f[2], f[3])
        J = J + 2.0 * wp.cross(fv, f[0] * J + wp.cross(fv, J))
        jnorm = wp.dot(J, J)
        if jnorm < 1.0e-16:
            wp.atomic_add(singular, world, 1)
            return
        efcid = wp.atomic_add(nefc, world, 1)
        wp.atomic_add(nl, world, 1)
        if efcid >= njmax:
            wp.atomic_add(overflow, world, 1)
            return
        dof = dadr[index]
        if wp.static(is_sparse):
            address = wp.atomic_add(nnz, world, 3)
            if address + 3 > njmax_nnz:
                wp.atomic_add(overflow, world, 1)
                return
            rownnz[world, efcid] = 3
            rowadr[world, efcid] = address
            for k in range(3):
                colind[world, 0, address + k] = dof + k
                efc_J[world, 0, address + k] = J[k]
            if wp.static(newton):
                block = wp.atomic_add(jtdaj_nblock, world, 1)
                jtdaj_adr[world, block] = efcid
                jtdaj_nrow[world, block] = 1
        else:
            for k in range(nv):
                efc_J[world, efcid, k] = 0.0
            for k in range(3):
                efc_J[world, efcid, dof + k] = J[k]
        velocity = (
            J[0] * qvel[world, dof]
            + J[1] * qvel[world, dof + 1]
            + J[2] * qvel[world, dof + 2]
        )
        # Match the native ball row's scalar inverse-weight approximation,
        # scaled by the squared norm of this non-unit Jacobian.
        weight = invweight[world % invweight.shape[0], dof] * jnorm
        constraint._efc_row(
            disableflags,
            world,
            timestep[world % timestep.shape[0]],
            efcid,
            pos,
            pos,
            weight,
            solref[index],
            solimp[index],
            margin[index],
            velocity,
            0.0,
            types.ConstraintType.LIMIT_JOINT,
            joint[index],
            False,
            0.0,
            efc_type,
            efc_id,
            efc_pos,
            efc_margin,
            efc_D,
            efc_vel,
            efc_aref,
            efc_frictionloss,
        )

    return kernel


@dataclass
class BallLimitBuffers:
    qadr: wp.array
    dadr: wp.array
    joint: wp.array
    neutral: wp.array
    frame: wp.array
    swing: wp.array
    twist: wp.array
    minimum_h: wp.array
    margin: wp.array
    solref: wp.array
    solimp: wp.array
    overflow: wp.array
    singular: wp.array
    kernel: object

    def launch(self, m, d, nnz):
        wp.launch(
            self.kernel,
            dim=(d.nworld, self.joint.size, 3),
            inputs=[
                m.nv,
                m.opt.timestep,
                m.opt.disableflags,
                self.qadr,
                self.dadr,
                self.joint,
                self.neutral,
                self.frame,
                self.swing,
                self.twist,
                self.minimum_h,
                self.margin,
                self.solref,
                self.solimp,
                m.dof_invweight0,
                d.qpos,
                d.qvel,
                d.njmax,
                d.njmax_nnz,
            ],
            outputs=[
                d.nl,
                d.nefc,
                d.efc.type,
                d.efc.id,
                d.efc.jtdaj_adr,
                d.efc.jtdaj_nrow,
                d.efc.jtdaj_nblock,
                d.efc.J_rownnz,
                d.efc.J_rowadr,
                d.efc.J_colind,
                d.efc.J,
                d.efc.pos,
                d.efc.margin,
                d.efc.D,
                d.efc.vel,
                d.efc.aref,
                d.efc.frictionloss,
                nnz,
                self.overflow,
                self.singular,
            ],
        )


def make_ball_limit_buffers(
    entries: Sequence[ResolvedBallJointLimit], m, d, device
) -> BallLimitBuffers:
    from instantmimic.utils.ball_joint_limits import _quaternion
    from instantmimic.utils.mjw_kernels.private import types

    if m.opt.integrator == types.IntegratorType.DISCRETE:
        # the row kernel builds the continuous-time impedance (is_discrete=False)
        raise ValueError(
            "directional ball constraints do not support the DISCRETE integrator"
        )
    if d.njmax < 3 * len(entries):
        raise ValueError(
            "directional ball constraints need three reserved rows per joint"
        )
    if m.is_sparse and d.njmax_nnz < 9 * len(entries):
        raise ValueError(
            "directional ball constraints need nine sparse entries per joint"
        )

    def array(values, dtype):
        return wp.array(np.asarray(values), dtype=dtype, device=device)

    return BallLimitBuffers(
        array([e.qposadr for e in entries], int),
        array([e.dofadr for e in entries], int),
        array([e.joint_id for e in entries], int),
        array([_quaternion(e.cfg.neutral, "neutral") for e in entries], wp.vec4),
        array([_quaternion(e.cfg.frame, "frame") for e in entries], wp.vec4),
        array(
            [
                [
                    1 / math.tan(v / 2) ** 2
                    for v in (*e.cfg.swing, *(e.cfg.swing_lower or e.cfg.swing))
                ]
                for e in entries
            ],
            wp.vec4,
        ),
        array([e.cfg.twist for e in entries], wp.vec2),
        array([e.minimum_h for e in entries], float),
        array([e.cfg.margin for e in entries], float),
        array([e.cfg.solref for e in entries], wp.vec2),
        array([e.cfg.solimp for e in entries], types.vec5),
        wp.zeros(d.nworld, dtype=int, device=device),
        wp.zeros(d.nworld, dtype=int, device=device),
        make_ball_limit_kernel(m.is_sparse, m.opt.solver == types.SolverType.NEWTON),
    )
