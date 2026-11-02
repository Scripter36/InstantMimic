"""Model-derived roles of MuJoCo coordinates (qpos) and DOFs (qvel).

Every compiled model is split once into three disjoint coordinate roles:

- **root**: the joints of the character's root body (a free joint, or a stack of
  slide/hinge joints on the pelvis), identified by body and joint
  type rather than by position;
- **character joints**: every other joint on a character body, i.e. a body in
  the root body's kinematic subtree that is not passive;
- **passive**: everything else. These coordinates keep their free MuJoCo
  dynamics but are not actuated, observed, tracked, guarded or rewarded, and a
  reset puts them at ``qpos0`` with zero velocity.

Passive coordinates are derived from the model structure, so most models need
no configuration:

- moving bodies outside the root body's subtree (extra kinematic trees with
  joints: projectiles or other props); static scenery bodies outside it are
  neither character nor passive;
- flexcomp vertex bodies (``dof="full"``/``"radial"`` vertices and trilinear
  nodes: bodies listed by a flex that move on their own slide joints) and their
  descendants.

Anything structure cannot decide is declared once through the environment
config block ``env.dof_roles`` (see :class:`DofRolesDeclaration`)::

    env:
      dof_roles:
        passive_bodies: [prop]  # body, its joints and its subtree
        passive_joints: [spring_joint]  # their bodies stay character

The same :class:`ModelDofRoles` instance is consumed by the observations,
rewards, terminations and velocity guards, resets, the motion reference and the
muscle controller. qpos and DOF addresses are kept separately
(``jnt_qposadr`` vs ``jnt_dofadr``): a free joint has 7 coordinates and 6 DOFs,
a ball joint 4 and 3.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from instantmimic.envs.config_access import get_value

__all__ = (
    "DOF_ROLES_CONFIG_KEY",
    "DofRolesDeclaration",
    "ModelDofRoles",
    "derive_dof_roles",
    "joint_widths",
    "resolve_dof_roles",
    "root_body_from_names",
)

DOF_ROLES_CONFIG_KEY = "dof_roles"
_DECLARATION_KEYS = frozenset({"passive_bodies", "passive_joints"})


def joint_widths(joint_type: int) -> tuple[int, int]:
    """Return ``(qpos width, dof width)`` of a MuJoCo joint type."""
    if int(joint_type) == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7, 6
    if int(joint_type) == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4, 3
    return 1, 1


def _names(value: Any, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Iterable):
        raise TypeError(f"env.dof_roles.{key} must be a list of names")
    names = tuple(str(name) for name in value)
    if len(set(names)) != len(names):
        raise ValueError(f"env.dof_roles.{key} lists a name twice: {list(names)}")
    return names


@dataclass(frozen=True, slots=True)
class DofRolesDeclaration:
    """Config-declared passive coordinates the model structure cannot decide.

    ``passive_bodies`` makes each named body, its joints and its whole subtree
    passive. ``passive_joints`` makes single joints passive while their bodies
    stay character bodies (observed and tracked as bodies).
    """

    passive_bodies: tuple[str, ...] = ()
    passive_joints: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, block: Any) -> DofRolesDeclaration:
        if block is None:
            return cls()
        if not isinstance(block, Mapping) and not callable(getattr(block, "get", None)):
            raise TypeError("env.dof_roles must be a mapping")
        keys = set(block.keys()) if hasattr(block, "keys") else set()
        unknown = keys - _DECLARATION_KEYS
        if unknown:
            raise ValueError(
                f"unknown env.dof_roles keys: {sorted(unknown)}; "
                f"expected a subset of {sorted(_DECLARATION_KEYS)}"
            )
        return cls(
            passive_bodies=_names(block.get("passive_bodies"), "passive_bodies"),
            passive_joints=_names(block.get("passive_joints"), "passive_joints"),
        )

    @classmethod
    def from_cfg(cls, cfg: Any) -> DofRolesDeclaration:
        if cfg is None:
            return cls()
        return cls.from_mapping(get_value(cfg, DOF_ROLES_CONFIG_KEY, None))


@dataclass(frozen=True, slots=True)
class ModelDofRoles:
    """Index sets of one compiled model. All addresses are model addresses.

    ``joint_*`` hold the character's non-root joints in model order; with the
    root leading (``root_leads``) their DOF addresses minus ``root_dof_count``
    are the "local" joint indices the controller and the task maps use.
    """

    nq: int
    nv: int
    nbody: int
    root_body: int
    root_joints: tuple[int, ...]
    root_qpos: tuple[int, ...]
    root_dof: tuple[int, ...]
    root_is_free: bool
    character_bodies: tuple[int, ...]
    passive_bodies: tuple[int, ...]
    joint_ids: tuple[int, ...]
    joint_qpos: tuple[int, ...]
    joint_dof: tuple[int, ...]
    passive_joint_ids: tuple[int, ...]
    passive_qpos: tuple[int, ...]
    passive_dof: tuple[int, ...]

    # -- derived sizes -------------------------------------------------------
    @property
    def root_qpos_count(self) -> int:
        return len(self.root_qpos)

    @property
    def root_dof_count(self) -> int:
        return len(self.root_dof)

    @property
    def character_nq(self) -> int:
        """Coordinates of the character (root and joints)."""
        return len(self.root_qpos) + len(self.joint_qpos)

    @property
    def character_nv(self) -> int:
        """DOFs of the character (root and joints)."""
        return len(self.root_dof) + len(self.joint_dof)

    @property
    def character_qpos(self) -> tuple[int, ...]:
        return tuple(sorted((*self.root_qpos, *self.joint_qpos)))

    @property
    def character_dof(self) -> tuple[int, ...]:
        return tuple(sorted((*self.root_dof, *self.joint_dof)))

    @property
    def has_passive(self) -> bool:
        return bool(self.passive_qpos or self.passive_dof)

    @property
    def motion_character_coordinates(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        """The motion loader's ``character_coordinates`` for this model.

        ``(character qpos, character DOFs)`` when the model has passive
        coordinates, which the loader then rests at ``qpos0``; None otherwise,
        so a model without passive coordinates keeps its motion cache key.
        """
        if not self.has_passive:
            return None
        return self.character_qpos, self.character_dof

    @property
    def root_leads(self) -> bool:
        """Whether the root coordinates are the leading qpos and DOF entries."""
        return self.root_qpos == tuple(range(len(self.root_qpos))) and (
            self.root_dof == tuple(range(len(self.root_dof)))
        )

    # -- local (non-root) index spaces ----------------------------------------
    def require_root_leads(self, owner: str) -> None:
        if not self.root_leads:
            raise ValueError(
                f"{owner} requires the root body's coordinates to lead qpos and "
                f"qvel; root qpos {list(self.root_qpos)}, root DOFs "
                f"{list(self.root_dof)}. Declare the character first in the MJCF."
            )

    @property
    def joint_dof_local(self) -> tuple[int, ...]:
        """Character joint DOFs in the non-root DOF space ``[root, nv)``."""
        root = self.root_dof_count
        return tuple(int(d) - root for d in self.joint_dof)

    @property
    def joint_qpos_local(self) -> tuple[int, ...]:
        """Character joint coordinates in the non-root space ``[root, nq)``."""
        root = self.root_qpos_count
        return tuple(int(q) - root for q in self.joint_qpos)

    @property
    def passive_dof_local(self) -> tuple[int, ...]:
        root = self.root_dof_count
        return tuple(int(d) - root for d in self.passive_dof)

    @property
    def joints_contiguous_after_root(self) -> bool:
        """Whether the joint DOFs fill ``[root, nv)`` with no passive DOF."""
        return self.root_leads and not self.passive_dof

    def passive_qpos0(self, model: mujoco.MjModel) -> tuple[float, ...]:
        return tuple(float(model.qpos0[q]) for q in self.passive_qpos)

    def summary(self) -> str:
        return (
            f"root body {self.root_body} ({self.root_qpos_count} qpos / "
            f"{self.root_dof_count} dof), {len(self.joint_ids)} character joints "
            f"({len(self.joint_qpos)} qpos / {len(self.joint_dof)} dof), "
            f"{len(self.passive_joint_ids)} passive joints ({len(self.passive_qpos)} "
            f"qpos / {len(self.passive_dof)} dof), {len(self.character_bodies)} "
            f"character bodies, {len(self.passive_bodies)} passive bodies"
        )


def _body_name(model: mujoco.MjModel, body: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or f"#{body}"


def _subtree(model: mujoco.MjModel, roots: Iterable[int]) -> set[int]:
    """Bodies in the subtrees of ``roots`` (roots included)."""
    parent = np.asarray(model.body_parentid)
    members = set(int(r) for r in roots)
    # MuJoCo orders a parent before its children.
    for body in range(1, int(model.nbody)):
        if int(parent[body]) in members:
            members.add(body)
    return members


def _flex_vertex_bodies(model: mujoco.MjModel) -> set[int]:
    """Bodies that a flex moves on their own coordinates.

    A flexcomp with movable vertices (``dof="full"``/``"radial"``) or nodes
    (``dof="trilinear"``) adds one body per movable vertex/node, with slide
    joints. A pinned vertex maps to its parent (bone) body instead; that body
    is a vertex body of the flex too, but it is the parent of the movable ones
    or carries non-slide joints, so it stays a character body.
    """
    if int(model.nflex) <= 0:
        return set()
    slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
    jnt_type = np.asarray(model.jnt_type)
    body_jntadr = np.asarray(model.body_jntadr)
    body_jntnum = np.asarray(model.body_jntnum)
    parent = np.asarray(model.body_parentid)
    result: set[int] = set()
    vert_adr = np.asarray(model.flex_vertadr)
    vert_num = np.asarray(model.flex_vertnum)
    vert_body = np.asarray(model.flex_vertbodyid)
    node_body = np.asarray(getattr(model, "flex_nodebodyid", np.zeros(0, int)))
    node_adr = np.asarray(getattr(model, "flex_nodeadr", np.zeros(0, int)))
    node_num = np.asarray(getattr(model, "flex_nodenum", np.zeros(0, int)))
    for flex in range(int(model.nflex)):
        bodies = {
            int(b) for b in vert_body[vert_adr[flex] : vert_adr[flex] + vert_num[flex]]
        }
        if node_num.size > flex and int(node_num[flex]) > 0:
            bodies |= {
                int(b)
                for b in node_body[node_adr[flex] : node_adr[flex] + node_num[flex]]
            }
        parents = {int(parent[b]) for b in bodies}
        for body in bodies:
            if body <= 0 or body in parents:
                continue
            count = int(body_jntnum[body])
            if count <= 0:
                continue
            start = int(body_jntadr[body])
            if all(int(jnt_type[j]) == slide for j in range(start, start + count)):
                result.add(body)
    return result


def _structural_root_body(model: mujoco.MjModel, excluded: set[int]) -> int:
    """The top body of the world-child tree with the most DOFs."""
    parent = np.asarray(model.body_parentid)
    dofnum = np.asarray(model.body_dofnum)
    best_body, best_dofs = -1, 0
    for top in range(1, int(model.nbody)):
        if int(parent[top]) != 0 or top in excluded:
            continue
        tree = _subtree(model, (top,)) - excluded
        dofs = int(sum(int(dofnum[b]) for b in tree))
        if dofs > best_dofs:
            best_body, best_dofs = top, dofs
    if best_body < 0:
        raise ValueError("The model has no kinematic tree with DOFs to use as root.")
    # A jointless top body (a fixed base) passes the root role to the first
    # descendant that carries joints.
    body = best_body
    while int(model.body_jntnum[body]) == 0:
        children = [
            b
            for b in range(body + 1, int(model.nbody))
            if int(parent[b]) == body and b not in excluded
        ]
        movable = [b for b in children if int(dofnum[b]) > 0]
        if len(movable) != 1:
            break
        body = movable[0]
    return int(body)


def _resolve_names(
    model: mujoco.MjModel, objtype: mujoco.mjtObj, names: Sequence[str], key: str
) -> list[int]:
    ids = []
    for name in names:
        obj = int(mujoco.mj_name2id(model, objtype, name))
        if obj < 0:
            raise ValueError(f"env.dof_roles.{key}: {name!r} is not in the model")
        ids.append(obj)
    return ids


def derive_dof_roles(
    model: mujoco.MjModel,
    *,
    root_body: int | None = None,
    declaration: DofRolesDeclaration | None = None,
) -> ModelDofRoles:
    """Derive the coordinate roles of ``model``.

    ``root_body`` selects the character root; by default it is the top body of
    the world-child tree with the most DOFs (flex vertex bodies excluded).
    """
    declaration = declaration or DofRolesDeclaration()
    flex_bodies = _flex_vertex_bodies(model)
    flex_subtree = _subtree(model, flex_bodies) if flex_bodies else set()
    if root_body is None:
        root_body = _structural_root_body(model, flex_subtree)
    root_body = int(root_body)
    if not 0 < root_body < int(model.nbody):
        raise ValueError(f"root body {root_body} is not a model body")
    if root_body in flex_subtree:
        raise ValueError("The root body cannot be a flex vertex body.")
    parent = np.asarray(model.body_parentid)
    dofnum = np.asarray(model.body_dofnum)
    ancestor = int(parent[root_body])
    while ancestor > 0:
        # A root below a moving body would leave the character's real root
        # coordinates outside the character (e.g. a jointless "pelvis" under a
        # free-jointed "root"): they would be classed passive and reset to qpos0.
        if dofnum[ancestor] > 0:
            raise ValueError(
                f"Root body {_body_name(model, root_body)!r} has a moving ancestor "
                f"{_body_name(model, ancestor)!r}; the root must be the top moving "
                "body of the character (check root_body_names)."
            )
        ancestor = int(parent[ancestor])

    declared_bodies = _resolve_names(
        model, mujoco.mjtObj.mjOBJ_BODY, declaration.passive_bodies, "passive_bodies"
    )
    if root_body in declared_bodies:
        raise ValueError("env.dof_roles.passive_bodies cannot contain the root body")
    declared_joints = set(
        _resolve_names(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            declaration.passive_joints,
            "passive_joints",
        )
    )

    root_tree = _subtree(model, (root_body,))
    # Bodies outside the character tree that move (a joint on them or an
    # ancestor) are passive; static scenery bodies are neither role.
    moving = [False] * int(model.nbody)
    for body in range(1, int(model.nbody)):
        moving[body] = bool(dofnum[body] > 0 or moving[int(parent[body])])
    passive_body_set = {
        body
        for body in range(1, int(model.nbody))
        if body not in root_tree and moving[body]
    }
    passive_body_set |= flex_subtree
    passive_body_set |= _subtree(model, declared_bodies)
    character_bodies = tuple(sorted(root_tree - passive_body_set))

    jnt_body = np.asarray(model.jnt_bodyid)
    jnt_type = np.asarray(model.jnt_type)
    root_joints = tuple(
        j for j in range(int(model.njnt)) if int(jnt_body[j]) == root_body
    )
    if declared_joints & set(root_joints):
        raise ValueError("env.dof_roles.passive_joints cannot name a root joint")

    def coordinates(joints: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        qpos: list[int] = []
        dof: list[int] = []
        for joint in joints:
            nq, nv = joint_widths(int(jnt_type[joint]))
            qa = int(model.jnt_qposadr[joint])
            da = int(model.jnt_dofadr[joint])
            qpos.extend(range(qa, qa + nq))
            dof.extend(range(da, da + nv))
        return tuple(qpos), tuple(dof)

    joint_ids: list[int] = []
    passive_joint_ids: list[int] = []
    for joint in range(int(model.njnt)):
        body = int(jnt_body[joint])
        if body == root_body:
            continue
        if body in passive_body_set or joint in declared_joints:
            passive_joint_ids.append(joint)
        else:
            joint_ids.append(joint)

    root_qpos, root_dof = coordinates(root_joints)
    joint_qpos, joint_dof = coordinates(joint_ids)
    passive_qpos, passive_dof = coordinates(passive_joint_ids)
    root_is_free = len(root_joints) == 1 and int(jnt_type[root_joints[0]]) == int(
        mujoco.mjtJoint.mjJNT_FREE
    )
    roles = ModelDofRoles(
        nq=int(model.nq),
        nv=int(model.nv),
        nbody=int(model.nbody),
        root_body=root_body,
        root_joints=root_joints,
        root_qpos=root_qpos,
        root_dof=root_dof,
        root_is_free=root_is_free,
        character_bodies=character_bodies,
        passive_bodies=tuple(sorted(passive_body_set)),
        joint_ids=tuple(joint_ids),
        joint_qpos=joint_qpos,
        joint_dof=joint_dof,
        passive_joint_ids=tuple(passive_joint_ids),
        passive_qpos=passive_qpos,
        passive_dof=passive_dof,
    )
    covered_q = len(roles.root_qpos) + len(roles.joint_qpos) + len(roles.passive_qpos)
    covered_v = len(roles.root_dof) + len(roles.joint_dof) + len(roles.passive_dof)
    if covered_q != int(model.nq) or covered_v != int(model.nv):
        raise RuntimeError("DOF roles do not partition the model coordinates")
    return roles


def root_body_from_names(
    model: mujoco.MjModel, names: Sequence[str] | str | None
) -> int | None:
    """The first of ``names`` that is a model body, or None."""
    if names is None:
        return None
    if isinstance(names, str):
        names = [names]
    for name in names:
        body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name)))
        if body >= 0:
            return body
    return None


def resolve_dof_roles(
    model: mujoco.MjModel,
    cfg: Any,
    *,
    root_body_names: Sequence[str] | None = None,
) -> ModelDofRoles:
    """Roles from a model and an environment config.

    The root body is the first of ``root_body_names`` (default: the config's
    ``root_body_names``) found in the model, else the structural root.
    """
    names = root_body_names
    if names is None and cfg is not None:
        names = get_value(cfg, "root_body_names", None)
    return derive_dof_roles(
        model,
        root_body=root_body_from_names(model, names),
        declaration=DofRolesDeclaration.from_cfg(cfg),
    )
