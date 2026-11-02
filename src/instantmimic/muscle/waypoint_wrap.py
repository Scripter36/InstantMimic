"""Fixed-radius waypoint geometry using unchanged MuJoCo Warp wrapping math.

The upstream mathematical helper is Apache-2.0, Copyright 2025 The Newton
Developers. This module changes no simulator kernels or integration stages.
"""

from __future__ import annotations

import mujoco
import torch
import warp as wp

from instantmimic.muscle.muscle_qfrc_projector import (
    body_point_velocity_wp,
    force_magnitude_from_activation_model_wp,
    muscle_force_velocity_curve_wp,
    normalized_muscle_length_wp,
)
from instantmimic.muscle.runtime_specs import WaypointWrapSpec
from instantmimic.utils.mjw_kernels.private import (
    MJ_MAXVAL,
    mul_quat,
    quat_to_mat,
    rot_vec_quat,
    util_misc,
)


@wp.struct
class WaypointWrapKernelState:
    point_arc: wp.array(dtype=wp.int32)
    arc: wp.array2d(dtype=wp.float32)


@wp.kernel(enable_backward=False)
def wrap_points_kernel(
    xpos: wp.array2d(dtype=wp.vec3),
    xquat: wp.array2d(dtype=wp.quat),
    body: wp.array(dtype=wp.int32),
    side_body: wp.array(dtype=wp.int32),
    kind: wp.array(dtype=wp.int32),
    center: wp.array(dtype=wp.vec3),
    quaternion: wp.array(dtype=wp.quat),
    radius: wp.array(dtype=wp.float32),
    side_local: wp.array(dtype=wp.vec3),
    start_point: wp.array(dtype=wp.int32),
    points: wp.array2d(dtype=wp.vec3),
    arc: wp.array2d(dtype=wp.float32),
):
    env, instance = wp.tid()
    start = start_point[instance]
    wrap_body = body[instance]
    q = xquat[env, wrap_body]
    q = q / wp.sqrt(wp.dot(q, q) + 1.0e-8)
    position = xpos[env, wrap_body] + rot_vec_quat(center[instance], q)
    orientation = mul_quat(q, quaternion[instance])
    side = wp.vec3(MJ_MAXVAL)
    if side_body[instance] >= 0:
        side_index = side_body[instance]
        qs = xquat[env, side_index]
        qs = qs / wp.sqrt(wp.dot(qs, qs) + 1.0e-8)
        side = xpos[env, side_index] + rot_vec_quat(side_local[instance], qs)
    a = points[env, start]
    b = points[env, start + 3]
    length, t0, t1 = util_misc.wrap(
        a, b, position, quat_to_mat(orientation), radius[instance], kind[instance], side
    )
    if length < 0.0:
        t0 = (a + b) * 0.5
        t1 = t0
        length = 0.0
    points[env, start + 1] = t0
    points[env, start + 2] = t1
    arc[env, instance] = length


@wp.func
def wrapped_segment_length(
    env: int,
    start: int,
    points: wp.array2d(dtype=wp.vec3),
    point_arc: wp.array(dtype=wp.int32),
    arc: wp.array2d(dtype=wp.float32),
) -> float:
    index = point_arc[start]
    if index >= 0:
        return arc[env, index]
    return wp.length(points[env, start + 1] - points[env, start])


@wp.kernel(enable_backward=False)
def path_lengths_kernel(
    points: wp.array2d(dtype=wp.vec3),
    starts: wp.array(dtype=wp.int32),
    point_arc: wp.array(dtype=wp.int32),
    arc: wp.array2d(dtype=wp.float32),
    lengths: wp.array2d(dtype=wp.float32),
):
    env, muscle = wp.tid()
    length = float(0.0)
    for i in range(starts[muscle], starts[muscle + 1] - 1):
        length += wrapped_segment_length(env, i, points, point_arc, arc)
    lengths[env, muscle] = length


class WaypointWrapState:
    """Persistent geometry outputs for one batch shape."""

    def __init__(self, count: int, batch: int, device, muscle_count: int = 0):
        self.arc_tensor = torch.zeros(
            (batch, count), dtype=torch.float32, device=device
        )
        self.arc = wp.from_torch(self.arc_tensor, dtype=wp.float32)
        self.lengths_tensor = torch.empty(
            (batch, muscle_count), device=device, dtype=torch.float32
        )
        self.lengths = wp.from_torch(self.lengths_tensor, dtype=wp.float32)


class WaypointWrapping(torch.nn.Module):
    """A static transmission table; each caller owns its mutable batch state."""

    def __init__(
        self,
        spec: WaypointWrapSpec | None,
        point_count: int,
        device,
    ):
        super().__init__()
        self.count = 0 if spec is None else int(spec.radius.numel())
        self.register_buffer(
            "point_arc_tensor",
            torch.full((point_count,), -1, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.point_arc = None
        self.inputs = []
        self._field_types = []
        if spec is not None and self.count:
            wp.init()
            count = self.count
            if any(
                getattr(spec, n).shape[0] != count
                for n in (
                    "body",
                    "side_body",
                    "kind",
                    "center",
                    "quaternion",
                    "side_local",
                    "start_point",
                )
            ):
                raise ValueError("All wrap fields must have one row per instance.")
            if bool(
                (
                    (spec.kind != int(mujoco.mjtWrap.mjWRAP_SPHERE))
                    & (spec.kind != int(mujoco.mjtWrap.mjWRAP_CYLINDER))
                ).any()
            ):
                raise ValueError("Only sphere/cylinder wrapping is supported.")
            if bool(
                ((spec.start_point < 0) | (spec.start_point + 3 >= point_count)).any()
            ):
                raise ValueError("A wrap requires four consecutive points A/T1/T2/B.")
            starts = spec.start_point.sort().values
            if bool(((starts[1:] - starts[:-1]) < 3).any()):
                raise ValueError("Wrap tangent slots must not overlap.")
            if not bool(torch.isfinite(spec.radius).all()) or bool(
                (spec.radius <= 0).any()
            ):
                raise ValueError("Wrap radii must be finite and positive.")
            self.point_arc_tensor[spec.start_point.long() + 1] = torch.arange(
                count, dtype=torch.int32, device=device
            )
            for name, dtype in [
                ("body", wp.int32),
                ("side_body", wp.int32),
                ("kind", wp.int32),
                ("center", wp.vec3),
                ("quaternion", wp.quat),
                ("radius", wp.float32),
                ("side_local", wp.vec3),
                ("start_point", wp.int32),
            ]:
                tensor = (
                    getattr(spec, name)
                    .to(
                        device=device,
                        dtype=torch.int32 if dtype == wp.int32 else torch.float32,
                    )
                    .contiguous()
                )
                self.register_buffer(name + "_tensor", tensor, persistent=False)
                self._field_types.append((name, dtype))
        self._refresh_views()

    def _refresh_views(self):
        if self.count:
            self.point_arc = wp.from_torch(self.point_arc_tensor, dtype=wp.int32)
            self.inputs = [
                wp.from_torch(getattr(self, name + "_tensor"), dtype=dtype)
                for name, dtype in self._field_types
            ]

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._refresh_views()
        return result

    def kernel_state(self, state):
        result = WaypointWrapKernelState()
        if self.count:
            result.point_arc = self.point_arc
            result.arc = state.arc
        return result

    def transform(self, xpos, xquat, points, state: WaypointWrapState):
        if self.count:
            wp.launch(
                wrap_points_kernel,
                dim=(points.shape[0], self.count),
                inputs=[xpos, xquat, *self.inputs],
                outputs=[points, state.arc],
                device=points.device,
            )


def make_wrap_spec(parsed, model, device) -> WaypointWrapSpec | None:
    raw = parsed.wraps if parsed.packed is None else parsed.packed.wraps
    if raw is None:
        return None

    def bodies(names, optional=False):
        result = []
        for name in names:
            index = (
                -1
                if name is None
                else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            )
            if index < 0 and not (optional and name is None):
                raise ValueError(f"Unknown wrapping body {name!r}.")
            result.append(index)
        return torch.tensor(result, dtype=torch.int32, device=device)

    return WaypointWrapSpec(
        body=bodies(raw.body_names),
        side_body=bodies(raw.side_body_names, True),
        **{
            n: torch.as_tensor(getattr(raw, n), device=device)
            for n in (
                "kind",
                "center",
                "quaternion",
                "radius",
                "side_local",
                "start_point",
            )
        },
    )


@wp.kernel(enable_backward=False)
def force_from_lengths_kernel(
    lengths: wp.array2d(dtype=wp.float32),
    params: wp.array2d(dtype=wp.float32),
    activation: wp.array2d(dtype=wp.float32),
    rest: wp.array(dtype=wp.float32),
    scale: float,
    lower: float,
    upper: float,
    clamp: float,
    active: float,
    passive: float,
    active_model: wp.vec3,
    passive_model: wp.vec3,
    force: wp.array2d(dtype=wp.float32),
):
    env, muscle = wp.tid()
    normalized = normalized_muscle_length_wp(
        lengths[env, muscle],
        params[muscle, 1],
        params[muscle, 2],
        rest[muscle],
        lower,
        upper,
    )
    force[env, muscle] = force_magnitude_from_activation_model_wp(
        activation[env, muscle],
        clamp,
        active,
        passive,
        params[muscle, 0],
        params[muscle, 3],
        scale,
        normalized,
        active_model,
        passive_model,
    )


@wp.kernel(enable_backward=False)
def force_components_kernel(
    lengths: wp.array2d(dtype=wp.float32),
    params: wp.array2d(dtype=wp.float32),
    rest: wp.array(dtype=wp.float32),
    scale: float,
    lower: float,
    upper: float,
    include_passive: float,
    active_model: wp.vec3,
    passive_model: wp.vec3,
    unit: wp.array2d(dtype=wp.float32),
    passive: wp.array2d(dtype=wp.float32),
):
    env, muscle = wp.tid()
    normalized = normalized_muscle_length_wp(
        lengths[env, muscle],
        params[muscle, 1],
        params[muscle, 2],
        rest[muscle],
        lower,
        upper,
    )
    unit[env, muscle] = force_magnitude_from_activation_model_wp(
        1.0,
        0.0,
        1.0,
        0.0,
        params[muscle, 0],
        params[muscle, 3],
        scale,
        normalized,
        active_model,
        passive_model,
    )
    passive[env, muscle] = force_magnitude_from_activation_model_wp(
        0.0,
        0.0,
        0.0,
        include_passive,
        params[muscle, 0],
        params[muscle, 3],
        scale,
        normalized,
        active_model,
        passive_model,
    )


@wp.kernel(enable_backward=False)
def path_velocity_kernel(
    points: wp.array2d(dtype=wp.vec3),
    starts: wp.array(dtype=wp.int32),
    bodies: wp.array(dtype=wp.int32),
    subtree_com: wp.array2d(dtype=wp.vec3),
    roots: wp.array(dtype=wp.int32),
    cvel: wp.array2d(dtype=wp.spatial_vector),
    velocity: wp.array2d(dtype=wp.float32),
):
    env, muscle = wp.tid()
    total = float(0.0)
    for start in range(starts[muscle], starts[muscle + 1] - 1):
        a = points[env, start]
        b = points[env, start + 1]
        direction = b - a
        distance = wp.length(direction)
        if distance > 1.0e-8:
            va = body_point_velocity_wp(env, bodies[start], a, subtree_com, roots, cvel)
            vb = body_point_velocity_wp(
                env, bodies[start + 1], b, subtree_com, roots, cvel
            )
            total += wp.dot(direction / distance, vb - va)
    velocity[env, muscle] = total


@wp.kernel(enable_backward=False)
def force_velocity_activation_kernel(
    activation: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    moment: wp.array2d(dtype=wp.float32),
    offsets: wp.array(dtype=wp.int32),
    dofs: wp.array(dtype=wp.int32),
    length_scale: wp.array(dtype=wp.float32),
    root_dof_count: int,
    vmax: float,
    fvmax: float,
    clamp: float,
    effective: wp.array2d(dtype=wp.float32),
):
    env, muscle = wp.tid()
    velocity = float(0.0)
    for entry in range(offsets[muscle], offsets[muscle + 1]):
        velocity -= moment[env, entry] * qvel[env, dofs[entry] + root_dof_count]
    scale = wp.max(length_scale[muscle] * vmax, 1.0e-6)
    curve = muscle_force_velocity_curve_wp(velocity / scale, fvmax)
    value = activation[env, muscle]
    if clamp != 0.0:
        value = wp.clamp(value, 0.0, 1.0)
    effective[env, muscle] = value * curve[0]
