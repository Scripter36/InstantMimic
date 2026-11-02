from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np
import torch
import warp as wp

from instantmimic.envs.config_access import get_value
from instantmimic.envs.dof_roles import (
    DofRolesDeclaration,
    ModelDofRoles,
    derive_dof_roles,
)
from instantmimic.motion.bvh_mapping import bvh_map_body_names
from instantmimic.motion.loader import (
    SMPL_MIMICKIT_DOF_BODY_ORDER,
    build_smpl_mimickit_motion_joint_groups,
)
from instantmimic.utils.mjw_kernels.private import BroadphaseType

_LOGGER = logging.getLogger(__name__)
__all__ = (
    "resolve_body_id",
    "get_mimickit_motion_joint_layout",
    "get_motion_loader_kwargs",
    "resolve_source_mjcf_path",
    "init_body_indices",
    "loaded_model_dof_roles",
    "reward_body_mask",
    "resolve_env_dof_roles",
    "configure_manual_collision_pairs",
    "build_joint_maps",
)


_TALUS_BODY_NAMES = frozenset(
    {
        "TalusR",
        "TalusL",
        "R_Ankle",
        "L_Ankle",
        "RightFoot",
        "LeftFoot",
        "right_foot",
        "left_foot",
        "R_Talus0",
        "L_Talus0",
    }
)

_TOE_BODY_NAMES = frozenset(
    {
        "FootThumbR",
        "FootThumbL",
        "FootPinkyR",
        "FootPinkyL",
        "R_Toe",
        "L_Toe",
        "RightToe",
        "LeftToe",
    }
)


def resolve_body_id(env, candidates: list[str]) -> int:
    for name in candidates:
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return int(body_id)
    raise ValueError(f"None of body names were found in model: {candidates}")


def _body_name_candidates(self, key: str, default: list[str]) -> list[str]:
    raw = get_value(getattr(self, "cfg", None), key, default)
    if raw is None:
        raw = default
    if isinstance(raw, str):
        return [raw]
    return [str(name) for name in raw]


def _body_name_candidate_groups(
    self,
    key: str,
    default: list[list[str]],
) -> list[list[str]]:
    raw = get_value(getattr(self, "cfg", None), key, default)
    if raw is None:
        raw = default
    groups: list[list[str]] = []
    for group in raw:
        if isinstance(group, str):
            groups.append([group])
        else:
            groups.append([str(name) for name in group])
    return groups


def get_mimickit_motion_joint_layout(env) -> str:
    return str(env.motion_settings.joint_layout)


def get_motion_loader_kwargs(env) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    roles = loaded_model_dof_roles(env)
    character = None if roles is None else roles.motion_character_coordinates
    if character is not None:
        # Recordings may span only the character; passive coordinates rest.
        kwargs["character_coordinates"] = character
    layout = get_mimickit_motion_joint_layout(env)
    if layout == "model":
        return kwargs
    kwargs["mimickit_joint_groups"] = build_smpl_mimickit_motion_joint_groups(env.model)
    return kwargs


def resolve_source_mjcf_path(env) -> str:
    sim_cfg = getattr(env.cfg, "sim", None)
    if sim_cfg is None:
        default_path = str(getattr(env, "_source_mjcf_path", ""))
    else:
        default_path = env._resolve_cfg_path_in_place(sim_cfg, "mjcf_path")
    resolver = getattr(
        getattr(env, "_control_pipeline", None),
        "resolve_source_mjcf_path",
        None,
    )
    if callable(resolver):
        return str(resolver(default_path))
    return default_path


def loaded_model_dof_roles(env) -> ModelDofRoles | None:
    """The coordinate roles of ``env.model`` as it is now.

    A task may load its motion reference on a different model than it simulates
    (the projectile command task swaps in its projectile-free source model while
    loading); the simulation model's roles do not index that model, so the
    roles are derived for it with the same root body and declaration.
    """
    roles = getattr(env, "dof_roles", None)
    if roles is None:
        return None
    model = env.model
    if (int(model.nq), int(model.nv), int(model.nbody)) == (
        roles.nq,
        roles.nv,
        roles.nbody,
    ):
        return roles
    return derive_dof_roles(
        model,
        root_body=roles.root_body,
        declaration=DofRolesDeclaration.from_cfg(getattr(env, "cfg", None)),
    )


def reward_body_mask(env, motion_body_mask: torch.Tensor) -> torch.Tensor:
    """The bodies tracking rewards score: the reference's bodies, minus the
    world body (fixed while looped reference transforms advance every cycle) and
    the passive bodies (instantmimic.envs.dof_roles) of ``env.model``, which a
    complete model-space recording may include. The mask indexes ``env.model``,
    which while a reference loads may differ from the simulated model."""
    mask = motion_body_mask.clone()
    if mask.numel() > 0:
        mask[0] = False
    roles = loaded_model_dof_roles(env)
    if roles is not None and roles.passive_bodies:
        mask[list(roles.passive_bodies)] = False
    return mask


def resolve_env_dof_roles(env) -> ModelDofRoles:
    """The task's coordinate roles (``instantmimic.envs.dof_roles``), derived
    once from the compiled model, the resolved root body and ``env.dof_roles``.

    The imitation runtime addresses non-root joints as ``dof - root_dof_count``,
    so the root coordinates must lead qpos and qvel.
    """
    roles = getattr(env, "dof_roles", None)
    if roles is not None:
        return roles
    root_body_idx = getattr(env, "root_body_idx", None)
    roles = derive_dof_roles(
        env.model,
        root_body=None if root_body_idx is None else int(root_body_idx),
        declaration=DofRolesDeclaration.from_cfg(getattr(env, "cfg", None)),
    )
    roles.require_root_leads("The imitation environment")
    env.dof_roles = roles
    if roles.has_passive:
        _LOGGER.info("DOF roles: %s", roles.summary())
    return roles


def init_body_indices(env):
    env.root_body_idx = env._resolve_body_id(
        _body_name_candidates(env, "root_body_names", ["Pelvis", "pelvis"])
    )
    env.torso_body_idx = env._resolve_body_id(
        _body_name_candidates(env, "torso_body_names", ["Torso", "torso"])
    )
    ee_body_groups = _body_name_candidate_groups(
        env,
        "ee_body_name_groups",
        [
            ["TalusR", "R_Ankle", "RightFoot", "R_Foot", "right_foot"],
            ["TalusL", "L_Ankle", "LeftFoot", "L_Foot", "left_foot"],
            ["HandR", "R_Wrist", "RightHand", "R_Hand", "right_hand"],
            ["HandL", "L_Wrist", "LeftHand", "L_Hand", "left_hand"],
            ["Head", "head"],
        ],
    )
    ee_body_idx_list = [env._resolve_body_id(group) for group in ee_body_groups]
    env.ee_body_idx_tuple = tuple(int(body_idx) for body_idx in ee_body_idx_list)
    env.ee_body_idx = torch.tensor(
        ee_body_idx_list,
        dtype=torch.long,
        device=env.device,
    )
    roles = resolve_env_dof_roles(env)
    # Tracking metrics score the character bodies only. Passive bodies and
    # flex vertices have no reference motion.
    character = list(roles.character_bodies)
    env._tracking_ape_body_idx_excluding_root = torch.tensor(
        [body_idx for body_idx in character if body_idx != int(env.root_body_idx)],
        dtype=torch.long,
        device=env.device,
    )
    env._tracking_ape_body_idx_including_root = torch.tensor(
        character,
        dtype=torch.long,
        device=env.device,
    )


def configure_manual_collision_pairs(env):
    desired_pairs = (
        ("TalusR_geom", "floor"),
        ("TalusL_geom", "floor"),
    )

    geom_ids: dict[str, int] = {}
    for gname in {name for pair in desired_pairs for name in pair}:
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            raise ValueError(f"Geom '{gname}' not found for manual collision setup.")
        geom_ids[gname] = int(gid)

    mj_pair_ids: dict[tuple[int, int], int] = {}
    for pid in range(env.model.npair):
        g1 = int(env.model.pair_geom1[pid])
        g2 = int(env.model.pair_geom2[pid])
        key = (g1, g2) if g1 < g2 else (g2, g1)
        mj_pair_ids[key] = pid

    geom_pairs: list[tuple[int, int]] = []
    pair_ids: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for g1_name, g2_name in desired_pairs:
        g1 = geom_ids[g1_name]
        g2 = geom_ids[g2_name]
        key = (g1, g2) if g1 < g2 else (g2, g1)
        if key in seen:
            continue
        seen.add(key)
        geom_pairs.append(key)
        pair_ids.append((mj_pair_ids.get(key, -1), -1))

    geom_pairs_np = np.asarray(geom_pairs, dtype=np.int32)
    pair_ids_np = np.asarray(pair_ids, dtype=np.int32)

    env.mjw_model.nxn_geom_pair_filtered = wp.array(geom_pairs_np, dtype=wp.vec2i)
    env.mjw_model.nxn_pairid_filtered = wp.array(pair_ids_np, dtype=wp.vec2i)
    env.mjw_model.opt.broadphase = BroadphaseType.NXN


def build_joint_maps(env):
    if get_mimickit_motion_joint_layout(env) == "smpl":
        mapped_body_names = set(SMPL_MIMICKIT_DOF_BODY_ORDER)
    else:
        # Rest bodies' joints are part of the reference too, held at qpos0.
        mapped_body_names = bvh_map_body_names(env.motion_settings.bvh_map) | set(
            env.motion_settings.rest_bodies
        )

    roles = resolve_env_dof_roles(env)
    env.root_qpos_count = int(roles.root_qpos_count)
    env.root_dof_count = int(roles.root_dof_count)
    character_joints = frozenset(roles.joint_ids)

    ball_qpos_gather = []
    ball_dof_scatter = []
    n_ball = 0
    talus_dof_local = []
    toe_dof_local = []
    # Masks over the non-root spaces [root, nv) / [root, nq): passive
    # coordinates (instantmimic.envs.dof_roles) stay False.
    joint_dof_mask = [False] * max(0, int(env.model.nv) - env.root_dof_count)
    joint_qpos_mask = [False] * max(0, int(env.model.nq) - env.root_qpos_count)

    for jnt_id in range(env.model.njnt):
        if jnt_id not in character_joints:
            continue
        jnt_type = env.model.jnt_type[jnt_id]
        qpos_adr = env.model.jnt_qposadr[jnt_id]
        dof_adr = env.model.jnt_dofadr[jnt_id]
        body_id = env.model.jnt_bodyid[jnt_id]
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        local_dof = dof_adr - env.root_dof_count
        local_qpos = qpos_adr - env.root_qpos_count
        is_mapped = body_name in mapped_body_names

        if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
            ball_qpos_gather.extend(
                [qpos_adr, qpos_adr + 1, qpos_adr + 2, qpos_adr + 3]
            )
            ball_dof_scatter.extend([local_dof, local_dof + 1, local_dof + 2])
            n_ball += 1

            if is_mapped:
                for offset in range(3):
                    joint_dof_mask[local_dof + offset] = True
                for offset in range(4):
                    joint_qpos_mask[local_qpos + offset] = True

            if body_name in _TALUS_BODY_NAMES:
                talus_dof_local.extend([local_dof, local_dof + 1, local_dof + 2])
            if body_name in _TOE_BODY_NAMES:
                toe_dof_local.extend([local_dof, local_dof + 1, local_dof + 2])

        elif jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
            if is_mapped:
                joint_dof_mask[local_dof] = True
                joint_qpos_mask[local_qpos] = True

            if body_name in _TALUS_BODY_NAMES:
                talus_dof_local.append(local_dof)
            if body_name in _TOE_BODY_NAMES:
                toe_dof_local.append(local_dof)

    env.joint_dof_mask = torch.tensor(
        joint_dof_mask, device=env.device, dtype=torch.bool
    )
    env.joint_qpos_mask = torch.tensor(
        joint_qpos_mask, device=env.device, dtype=torch.bool
    )
    env.joint_dof_indices_tuple = tuple(
        idx for idx, enabled in enumerate(joint_dof_mask) if enabled
    )
    env.joint_qpos_indices_tuple = tuple(
        idx for idx, enabled in enumerate(joint_qpos_mask) if enabled
    )

    hinge_qpos_indices = []
    hinge_dof_indices = []
    hinge_body_indices = []
    hinge_parent_body_indices = []
    hinge_axis_local = []
    hinge_body_counts: dict[int, int] = {}

    for jnt_id in roles.joint_ids:
        jnt_type = env.model.jnt_type[jnt_id]
        body_id = int(env.model.jnt_bodyid[jnt_id])
        if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_qpos_indices.append(env.model.jnt_qposadr[jnt_id])
            hinge_dof_indices.append(env.model.jnt_dofadr[jnt_id] - env.root_dof_count)
            hinge_body_indices.append(body_id)
            hinge_parent_body_indices.append(int(env.model.body_parentid[body_id]))
            hinge_axis_local.append(
                tuple(float(v) for v in env.model.jnt_axis[jnt_id].tolist())
            )
            hinge_body_counts[body_id] = hinge_body_counts.get(body_id, 0) + 1

    env.hinge_qpos_indices = torch.tensor(
        hinge_qpos_indices, device=env.device, dtype=torch.long
    )
    env.hinge_dof_indices = torch.tensor(
        hinge_dof_indices, device=env.device, dtype=torch.long
    )
    env.hinge_body_indices = torch.tensor(
        hinge_body_indices, device=env.device, dtype=torch.long
    )
    env.hinge_parent_body_indices = torch.tensor(
        hinge_parent_body_indices, device=env.device, dtype=torch.long
    )
    env.hinge_multi_body_mask = torch.tensor(
        [hinge_body_counts.get(body_id, 0) > 1 for body_id in hinge_body_indices],
        device=env.device,
        dtype=torch.bool,
    )
    if hinge_axis_local:
        env.hinge_axis_local = torch.tensor(
            hinge_axis_local, device=env.device, dtype=torch.float32
        )
    else:
        env.hinge_axis_local = torch.zeros(
            (0, 3), device=env.device, dtype=torch.float32
        )
    env.hinge_qpos_indices_tuple = tuple(int(idx) for idx in hinge_qpos_indices)
    env.hinge_dof_indices_tuple = tuple(int(idx) for idx in hinge_dof_indices)
    env.hinge_body_indices_tuple = tuple(int(idx) for idx in hinge_body_indices)
    env.hinge_parent_body_indices_tuple = tuple(
        int(idx) for idx in hinge_parent_body_indices
    )
    env.hinge_axis_local_tuple = tuple(
        tuple(float(component) for component in axis) for axis in hinge_axis_local
    )
    env.hinge_multi_body_mask_tuple = tuple(
        bool(hinge_body_counts.get(body_id, 0) > 1) for body_id in hinge_body_indices
    )
    env.body_parent_idx_tuple = tuple(int(v) for v in env.model.body_parentid.tolist())
    env.ball_qpos_gather = torch.tensor(
        ball_qpos_gather, device=env.device, dtype=torch.long
    )
    env.ball_dof_scatter = torch.tensor(
        ball_dof_scatter, device=env.device, dtype=torch.long
    )
    env.n_ball_joints = n_ball
    env.talus_dof_indices = torch.tensor(
        talus_dof_local, device=env.device, dtype=torch.long
    )
    env.toe_dof_indices = torch.tensor(
        toe_dof_local, device=env.device, dtype=torch.long
    )
    env.talus_dof_indices_tuple = tuple(int(idx) for idx in talus_dof_local)
    env.toe_dof_indices_tuple = tuple(int(idx) for idx in toe_dof_local)

    # Character joint DOFs (observed, rewarded, guarded) as model addresses and
    # as indices of the non-root space; passive coordinates for resets.
    env.joint_dof_adr_tuple = tuple(int(d) for d in roles.joint_dof)
    env.joint_dof_local_tuple = tuple(int(d) for d in roles.joint_dof_local)
    env.joint_dof_adr_i32 = torch.tensor(
        env.joint_dof_adr_tuple, device=env.device, dtype=torch.int32
    )
    env.joint_dof_local_i32 = torch.tensor(
        env.joint_dof_local_tuple, device=env.device, dtype=torch.int32
    )
    env.passive_qpos_adr_i32 = torch.tensor(
        roles.passive_qpos, device=env.device, dtype=torch.int32
    )
    env.passive_qpos0 = torch.tensor(
        roles.passive_qpos0(env.model), device=env.device, dtype=torch.float32
    )
    env.passive_dof_adr_i32 = torch.tensor(
        roles.passive_dof, device=env.device, dtype=torch.int32
    )

    _LOGGER.info(
        "Joint maps: %d ball joints, %d hinge joints, %d talus DOFs, %d toe DOFs, "
        "nq=%d, nv=%d",
        n_ball,
        len(hinge_qpos_indices),
        len(talus_dof_local),
        len(toe_dof_local),
        env.model.nq,
        env.model.nv,
    )
