"""Sample fixed-radius transmission arcs from the renderer's body poses."""

from __future__ import annotations

from dataclasses import fields

import mujoco
import numpy as np
import warp as wp

from instantmimic.muscle.muscle_qfrc_projector import transform_waypoints_kernel
from instantmimic.muscle.runtime_specs import WaypointWrapSpec
from instantmimic.muscle.waypoint_wrap import WaypointWrapping, WaypointWrapState
from instantmimic.utils.mjw_kernels.private import mul_quat, rot_vec_quat


@wp.kernel(enable_backward=False)
def _sample_arcs(
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    points: wp.array2d(dtype=wp.vec3),
    arc: wp.array2d(dtype=wp.float32),
    body: wp.array(dtype=wp.int32),
    center: wp.array(dtype=wp.vec3),
    quaternion: wp.array(dtype=wp.quat),
    radius: wp.array(dtype=wp.float32),
    kind: wp.array(dtype=wp.int32),
    start: wp.array(dtype=wp.int32),
    point_index: wp.array(dtype=wp.int32),
    segments: int,
    out: wp.array2d(dtype=wp.vec3),
):
    env, instance, sample = wp.tid()
    i = start[instance]
    a, t0, t1 = points[env, i], points[env, i + 1], points[env, i + 2]
    fraction = float(sample) / float(segments)
    result = t0 + fraction * (t1 - t0)
    length = arc[env, instance]
    if length > 1.0e-8:
        b = body[instance]
        q = xquat[env, b]
        c = xpos[env, b] + rot_vec_quat(center[instance], q)
        axis = rot_vec_quat(wp.vec3(0.0, 0.0, 1.0), mul_quat(q, quaternion[instance]))
        if kind[instance] == wp.static(int(mujoco.mjtWrap.mjWRAP_SPHERE)):
            axis = wp.cross(t0 - c, t1 - c)
            if wp.length(axis) < 1.0e-8:
                axis = wp.cross(t0 - c, t0 - a)
        norm = wp.length(axis)
        if norm > 1.0e-8:
            axis = axis / norm
            h0, h1 = wp.dot(t0 - c, axis), wp.dot(t1 - c, axis)
            radial = t0 - c - h0 * axis
            tangent = wp.cross(axis, radial)
            sign = float(1.0)
            if wp.dot(tangent, t0 - a) < 0.0:
                sign = -1.0
            angle = (
                sign
                * wp.sqrt(wp.max(0.0, length * length - (h1 - h0) * (h1 - h0)))
                / radius[instance]
            )
            theta = fraction * angle
            result = (
                c
                + wp.cos(theta) * radial
                + wp.sin(theta) * tangent
                + (h0 + fraction * (h1 - h0)) * axis
            )
    if sample == 0:
        result = t0
    elif sample == segments:
        result = t1
    out[env, point_index[i + 1] + sample] = result


class WrappedMuscleRenderGeometry:
    """Persistent per-device geometry for CPU scenes, OpenGL and Warp overlays."""

    def __init__(self, spec, body, local, starts, arc_segments=16):
        if arc_segments < 2:
            raise ValueError("Wrapped rendering requires at least two arc segments.")
        self.spec = WaypointWrapSpec(
            **{f.name: getattr(spec, f.name).detach().cpu() for f in fields(spec)}
        )
        self.body = np.asarray(body, dtype=np.int32)
        self.local = np.asarray(local, dtype=np.float32)
        self.arc_segments = int(arc_segments)
        self.point_index = np.arange(len(self.local) + 1, dtype=np.int32)
        for start in self.spec.start_point.numpy():
            self.point_index[int(start) + 2 :] += arc_segments - 1
        # Static metadata supplies muscle/unit ownership; geometry is always resolved dynamically.
        self.expanded_body = np.repeat(self.body, np.diff(self.point_index))
        self.expanded_local = np.repeat(self.local, np.diff(self.point_index), axis=0)
        self.expanded_starts = self.point_index[np.asarray(starts, dtype=np.int32)]
        self._states = {}

    def __call__(self, data):
        native = isinstance(data.xpos, np.ndarray)
        if native:
            xpos = wp.array(
                data.xpos[None].astype(np.float32), dtype=wp.vec3, device="cpu"
            )
            xquat = wp.array(
                data.xquat[None].astype(np.float32), dtype=wp.quat, device="cpu"
            )
        else:
            xpos, xquat = data.xpos, data.xquat
        batch, device = xpos.shape[0], xpos.device
        key = (str(device), batch)
        state = self._states.get(key)
        if state is None:
            if device.is_capturing:
                raise RuntimeError("Warm wrapped render geometry before graph capture.")
            wrapping = WaypointWrapping(self.spec, len(self.local), str(device))
            state = (
                wrapping,
                WaypointWrapState(wrapping.count, batch, str(device)),
                wp.array(self.body, dtype=wp.int32, device=device),
                wp.array(self.local, dtype=wp.vec3, device=device),
                wp.array(self.point_index[:-1], dtype=wp.int32, device=device),
                wp.empty((batch, len(self.local)), dtype=wp.vec3, device=device),
                wp.empty(
                    (batch, int(self.point_index[-1])), dtype=wp.vec3, device=device
                ),
            )
            self._states[key] = state
        wrapping, arc, body, local, indices, points, out = state
        wp.launch(
            transform_waypoints_kernel,
            dim=(batch, len(self.local)),
            inputs=[xpos, xquat, body, local],
            outputs=[points],
            device=device,
        )
        wrapping.transform(xpos, xquat, points, arc)
        # Scatter original points; arc sampling fills inserted vertices.
        wp.launch(
            _scatter_points,
            dim=(batch, len(self.local)),
            inputs=[points, indices],
            outputs=[out],
            device=device,
        )
        wp.launch(
            _sample_arcs,
            dim=(batch, wrapping.count, self.arc_segments + 1),
            inputs=[
                xpos,
                xquat,
                points,
                arc.arc,
                wrapping.inputs[0],
                wrapping.inputs[3],
                wrapping.inputs[4],
                wrapping.inputs[5],
                wrapping.inputs[2],
                wrapping.inputs[7],
                indices,
                self.arc_segments,
            ],
            outputs=[out],
            device=device,
        )
        return wp.to_torch(out)


@wp.kernel(enable_backward=False)
def _scatter_points(
    points: wp.array2d(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    out: wp.array2d(dtype=wp.vec3),
):
    env, i = wp.tid()
    out[env, indices[i]] = points[env, i]
