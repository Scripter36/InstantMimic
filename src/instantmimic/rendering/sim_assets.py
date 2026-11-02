"""Conservative removal of physics-neutral rendering assets."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import mujoco


@dataclass(frozen=True, slots=True)
class SimVisualStripSummary:
    geoms: int
    meshes: int
    materials: int
    textures: int
    skins: int


def physics_neutral_visual_geom_names(
    spec: mujoco.MjSpec,
    *,
    external_geom_dependencies: Collection[str] = (),
) -> frozenset[str]:
    """Names of the geoms ``strip_physics_neutral_visual_assets`` would remove.

    MuJoCo geom ``group`` is intentionally not consulted: it controls display
    grouping and can filter inertia inference, but it is not a collision group.
    A geom is removable only when it has no native collision bits, has explicit
    zero mass, is not retained by MuJoCo's own ``discardvisual`` reference
    analysis, and is not named by an out-of-engine dependency supplied by the
    environment.
    """

    protected = {str(name) for name in external_geom_dependencies}
    authored_geom_names = {geom.name for geom in spec.geoms if geom.name}
    missing_dependencies = protected - authored_geom_names
    if missing_dependencies:
        names = ", ".join(sorted(missing_dependencies))
        raise ValueError(
            "External simulation geom dependencies are missing from the MJCF: " + names
        )

    # Use MuJoCo's compiler as the authority on references from contact pairs,
    # tendons, sensors, and other MJCF elements. The probe may discard additional
    # non-colliding geoms whose inertia it can materialize; our stricter pass
    # below keeps those unless the author explicitly assigned mass=0.
    probe = spec.copy()
    # MuJoCo 3.16 rejects discarding massive visual geoms with forced inertia
    # inference. AUTO can retain inferred inertia while stripping, but must
    # still ignore authored inertials just as TRUE did. Both flags participate
    # in native inference: ipos selects body inference and explicitinertial
    # selects geom inference. Change only this reference-analysis copy.
    if (
        probe.compiler.inertiafromgeom
        == mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    ):
        for body in probe.bodies:
            body.explicitinertial = False
            body.ipos[0] = float("nan")
        probe.compiler.inertiafromgeom = (
            mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO
        )
    probe.compiler.discardvisual = True
    probe_model = probe.compile()
    retained_geom_names = {
        mujoco.mj_id2name(probe_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(probe_model.ngeom)
    }

    return frozenset(
        geom.name
        for geom in spec.geoms
        if geom.name
        and geom.name not in protected
        and geom.name not in retained_geom_names
        and int(geom.contype) == 0
        and int(geom.conaffinity) == 0
        and float(geom.mass) == 0.0
    )


def strip_physics_neutral_visual_assets(
    spec: mujoco.MjSpec,
    *,
    external_geom_dependencies: Collection[str] = (),
) -> SimVisualStripSummary:
    """Remove the assets proven irrelevant to native and declared external physics.

    See ``physics_neutral_visual_geom_names`` for the geom criterion.
    """

    removable = physics_neutral_visual_geom_names(
        spec, external_geom_dependencies=external_geom_dependencies
    )
    removable_geoms = [geom for geom in spec.geoms if geom.name in removable]
    for geom in removable_geoms:
        spec.delete(geom)

    referenced_meshes = {geom.meshname for geom in spec.geoms if geom.meshname}
    # MjSpec leaves names blank for meshes whose names MuJoCo infers from their
    # filenames. Geoms already refer to those inferred names, so deleting an
    # apparently anonymous mesh leaves a dangling reference in the next compile.
    unused_meshes = [
        mesh for mesh in spec.meshes if mesh.name and mesh.name not in referenced_meshes
    ]
    for mesh in unused_meshes:
        spec.delete(mesh)

    # Materials, textures, and skins do not participate in dynamics. Clear
    # material references before deleting assets so the resulting spec remains
    # independently compilable.
    for collection in (spec.geoms, spec.sites, spec.tendons):
        for element in collection:
            if hasattr(element, "material"):
                element.material = ""

    materials = list(spec.materials)
    textures = list(spec.textures)
    skins = list(spec.skins)
    for collection in (materials, textures, skins):
        for element in collection:
            spec.delete(element)

    return SimVisualStripSummary(
        geoms=len(removable_geoms),
        meshes=len(unused_meshes),
        materials=len(materials),
        textures=len(textures),
        skins=len(skins),
    )


__all__ = [
    "SimVisualStripSummary",
    "physics_neutral_visual_geom_names",
    "strip_physics_neutral_visual_assets",
]
