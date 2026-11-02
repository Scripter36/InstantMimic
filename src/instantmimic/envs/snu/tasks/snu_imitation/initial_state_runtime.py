"""SNU adapter for PPO-owned adaptive initial-state distributions."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import warp as wp

from instantmimic.envs.config_access import get_block, get_value
from instantmimic.motion.cache import model_signature
from instantmimic.rl.initial_state import InitialStateSpec
from instantmimic.utils.mjw_kernels.private import quat_integrate, quat_sub

_MODE_SAMPLE = 0
_MODE_MEAN = 1
_MODE_REFERENCE = 2
_SUPPORTED_MODES = {
    "sample": _MODE_SAMPLE,
    "mean": _MODE_MEAN,
    "reference": _MODE_REFERENCE,
}
_LOG_TWO_PI = 1.8378770664093453
_MOTION_FILE_DIGEST_CACHE: dict[tuple[str, int, int], bytes] = {}
_COORDINATES_POSE_RESIDUAL_TRAJECTORY = "pose_residual_trajectory"
_JOINT_FREE = 0
_JOINT_BALL = 1
_JOINT_SLIDE = 2
_JOINT_HINGE = 3


@dataclass(frozen=True, slots=True)
class InitialStateSettings:
    enabled: bool
    num_components: int
    residual_scale: tuple[float, ...]
    evaluation_mode: str


@dataclass(slots=True)
class InitialStateRuntimeState:
    settings: InitialStateSettings
    spec: InitialStateSpec | None
    samples: torch.Tensor | None = None
    component_ids: torch.Tensor | None = None
    old_log_prob: torch.Tensor | None = None
    versions: torch.Tensor | None = None
    pending_start: torch.Tensor | None = None
    mode: torch.Tensor | None = None
    collection_enabled: torch.Tensor | None = None
    residual_scale: torch.Tensor | None = None
    mean_wp: wp.array | None = None
    log_std_wp: wp.array | None = None
    version_wp: wp.array | None = None
    residual_scale_wp: wp.array | None = None
    mode_wp: wp.array | None = None
    collection_enabled_wp: wp.array | None = None
    ref_frame_idx_wp: wp.array | None = None
    samples_wp: wp.array | None = None
    component_ids_wp: wp.array | None = None
    old_log_prob_wp: wp.array | None = None
    versions_wp: wp.array | None = None
    pending_start_wp: wp.array | None = None
    next_qpos: torch.Tensor | None = None
    next_qpos_wp: wp.array | None = None


@wp.kernel(enable_backward=False)
def _sample_and_apply_pose_residual_trajectory_kernel(
    reset_mask: wp.array(dtype=wp.bool),
    reset_seed: wp.array(dtype=int),
    common_step_counter: wp.array(dtype=int),
    ref_frame_idx: wp.array(dtype=int),
    motion_frame_count: int,
    control_dt: float,
    nq: int,
    nv: int,
    njnt: int,
    residual_scale: wp.array(dtype=float),
    mean: wp.array2d(dtype=float),
    log_std: wp.array2d(dtype=float),
    version: wp.array(dtype=wp.int64),
    mode: wp.array(dtype=int),
    collection_enabled: wp.array(dtype=int),
    motion_qpos: wp.array2d(dtype=float),
    spawn_offset: wp.vec3,
    jnt_type: wp.array(dtype=int),
    jnt_qposadr: wp.array(dtype=int),
    jnt_dofadr: wp.array(dtype=int),
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    next_qpos: wp.array2d(dtype=float),
    samples: wp.array3d(dtype=float),
    component_ids: wp.array2d(dtype=wp.int64),
    old_log_prob: wp.array(dtype=float),
    versions: wp.array(dtype=wp.int64),
    pending_start: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()
    if not reset_mask[env_idx]:
        return

    frame = ref_frame_idx[env_idx]
    next_frame = frame + 1
    has_next = next_frame < motion_frame_count
    if not has_next:
        next_frame = frame
    component_ids[env_idx, 0] = wp.int64(frame)
    component_ids[env_idx, 1] = wp.int64(next_frame) if has_next else wp.int64(-1)
    versions[env_idx] = version[0]
    old_log_prob[env_idx] = 0.0
    sample_mode = mode[0]
    seed = reset_seed[0] + common_step_counter[0]

    for latent_idx in range(nv):
        mu = mean[frame, latent_idx]
        log_sigma = log_std[frame, latent_idx]
        current_value = 0.0
        current_epsilon = 0.0
        if sample_mode == _MODE_SAMPLE:
            random_state = wp.rand_init(seed, env_idx * 8192 + 4096 + latent_idx)
            current_epsilon = wp.randn(random_state)
            current_value = mu + wp.exp(log_sigma) * current_epsilon
            old_log_prob[env_idx] += -0.5 * (
                current_epsilon * current_epsilon + 2.0 * log_sigma + _LOG_TWO_PI
            )
        elif sample_mode == _MODE_MEAN:
            current_value = mu
        samples[env_idx, 0, latent_idx] = current_value

        next_value = current_value
        if has_next:
            next_mu = mean[next_frame, latent_idx]
            next_log_sigma = log_std[next_frame, latent_idx]
            next_value = 0.0
            next_epsilon = 0.0
            if sample_mode == _MODE_SAMPLE:
                next_random_state = wp.rand_init(
                    seed, env_idx * 8192 + 6144 + latent_idx
                )
                next_epsilon = wp.randn(next_random_state)
                next_value = next_mu + wp.exp(next_log_sigma) * next_epsilon
                old_log_prob[env_idx] += -0.5 * (
                    next_epsilon * next_epsilon + 2.0 * next_log_sigma + _LOG_TWO_PI
                )
            elif sample_mode == _MODE_MEAN:
                next_value = next_mu
        samples[env_idx, 1, latent_idx] = next_value

    for qpos_idx in range(nq):
        next_qpos[env_idx, qpos_idx] = motion_qpos[next_frame, qpos_idx]
    if nq >= 3:
        next_qpos[env_idx, 0] += spawn_offset[0]
        next_qpos[env_idx, 1] += spawn_offset[1]
        next_qpos[env_idx, 2] += spawn_offset[2]

    for joint_idx in range(njnt):
        joint_type = jnt_type[joint_idx]
        qpos_adr = jnt_qposadr[joint_idx]
        dof_adr = jnt_dofadr[joint_idx]
        if joint_type == _JOINT_FREE:
            for offset in range(3):
                scale = residual_scale[dof_adr + offset]
                qpos[env_idx, qpos_adr + offset] += (
                    samples[env_idx, 0, dof_adr + offset] * scale
                )
                next_qpos[env_idx, qpos_adr + offset] += (
                    samples[env_idx, 1, dof_adr + offset] * scale
                )
                qvel[env_idx, dof_adr + offset] = (
                    next_qpos[env_idx, qpos_adr + offset]
                    - qpos[env_idx, qpos_adr + offset]
                ) / control_dt

            current_quat = wp.quat(
                qpos[env_idx, qpos_adr + 3],
                qpos[env_idx, qpos_adr + 4],
                qpos[env_idx, qpos_adr + 5],
                qpos[env_idx, qpos_adr + 6],
            )
            next_quat = wp.quat(
                next_qpos[env_idx, qpos_adr + 3],
                next_qpos[env_idx, qpos_adr + 4],
                next_qpos[env_idx, qpos_adr + 5],
                next_qpos[env_idx, qpos_adr + 6],
            )
            current_delta = wp.vec3(
                samples[env_idx, 0, dof_adr + 3] * residual_scale[dof_adr + 3],
                samples[env_idx, 0, dof_adr + 4] * residual_scale[dof_adr + 4],
                samples[env_idx, 0, dof_adr + 5] * residual_scale[dof_adr + 5],
            )
            next_delta = wp.vec3(
                samples[env_idx, 1, dof_adr + 3] * residual_scale[dof_adr + 3],
                samples[env_idx, 1, dof_adr + 4] * residual_scale[dof_adr + 4],
                samples[env_idx, 1, dof_adr + 5] * residual_scale[dof_adr + 5],
            )
            current_quat = quat_integrate(current_quat, current_delta, 1.0)
            next_quat = quat_integrate(next_quat, next_delta, 1.0)
            for offset in range(4):
                qpos[env_idx, qpos_adr + 3 + offset] = current_quat[offset]
                next_qpos[env_idx, qpos_adr + 3 + offset] = next_quat[offset]
            angular_velocity = quat_sub(next_quat, current_quat) / control_dt
            for offset in range(3):
                qvel[env_idx, dof_adr + 3 + offset] = angular_velocity[offset]
        elif joint_type == _JOINT_BALL:
            current_quat = wp.quat(
                qpos[env_idx, qpos_adr],
                qpos[env_idx, qpos_adr + 1],
                qpos[env_idx, qpos_adr + 2],
                qpos[env_idx, qpos_adr + 3],
            )
            next_quat = wp.quat(
                next_qpos[env_idx, qpos_adr],
                next_qpos[env_idx, qpos_adr + 1],
                next_qpos[env_idx, qpos_adr + 2],
                next_qpos[env_idx, qpos_adr + 3],
            )
            current_delta = wp.vec3(
                samples[env_idx, 0, dof_adr] * residual_scale[dof_adr],
                samples[env_idx, 0, dof_adr + 1] * residual_scale[dof_adr + 1],
                samples[env_idx, 0, dof_adr + 2] * residual_scale[dof_adr + 2],
            )
            next_delta = wp.vec3(
                samples[env_idx, 1, dof_adr] * residual_scale[dof_adr],
                samples[env_idx, 1, dof_adr + 1] * residual_scale[dof_adr + 1],
                samples[env_idx, 1, dof_adr + 2] * residual_scale[dof_adr + 2],
            )
            current_quat = quat_integrate(current_quat, current_delta, 1.0)
            next_quat = quat_integrate(next_quat, next_delta, 1.0)
            for offset in range(4):
                qpos[env_idx, qpos_adr + offset] = current_quat[offset]
                next_qpos[env_idx, qpos_adr + offset] = next_quat[offset]
            angular_velocity = quat_sub(next_quat, current_quat) / control_dt
            for offset in range(3):
                qvel[env_idx, dof_adr + offset] = angular_velocity[offset]
        elif joint_type == _JOINT_SLIDE or joint_type == _JOINT_HINGE:
            scale = residual_scale[dof_adr]
            current_position = (
                qpos[env_idx, qpos_adr] + samples[env_idx, 0, dof_adr] * scale
            )
            next_position = (
                next_qpos[env_idx, qpos_adr] + samples[env_idx, 1, dof_adr] * scale
            )
            qpos[env_idx, qpos_adr] = current_position
            next_qpos[env_idx, qpos_adr] = next_position
            qvel[env_idx, dof_adr] = (next_position - current_position) / control_dt

    pending_start[env_idx] = sample_mode == _MODE_SAMPLE and collection_enabled[0] != 0


def _mapping_value(mapping: Any, key: str, default: float) -> float:
    return float(get_value(mapping, key, default))


def _same_torch_device(left: torch.device | str, right: torch.device | str) -> bool:
    left_device = torch.device(left)
    right_device = torch.device(right)
    if left_device.type != right_device.type:
        return False
    if left_device.type != "cuda":
        return left_device == right_device
    return (
        left_device.index is None
        or right_device.index is None
        or left_device.index == right_device.index
    )


def _motion_reference_fingerprint(env) -> str:
    digest = hashlib.sha256()
    for source in getattr(env, "_resolved_motion_sources", ()):
        path = Path(str(source))
        if path.is_file():
            stat = path.stat()
            cache_key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
            source_digest = _MOTION_FILE_DIGEST_CACHE.get(cache_key)
            if source_digest is None:
                source_hasher = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        source_hasher.update(chunk)
                source_digest = source_hasher.digest()
                _MOTION_FILE_DIGEST_CACHE[cache_key] = source_digest
            digest.update(source_digest)
        else:
            digest.update(str(source).encode())
        digest.update(b"\0")
    digest.update(str(int(env.motion_frame_count)).encode())
    digest.update(f":{float(env.step_dt):.12g}".encode())
    return digest.hexdigest()


def _reference_pose_fingerprint(env) -> str:
    """Identify the loaded poses after scaling, alignment, and resampling."""
    poses = env.motion_qpos.detach().cpu().contiguous().numpy()
    return hashlib.sha256(poses.tobytes()).hexdigest()


def _physics_neutral_geoms(env) -> frozenset[str]:
    """Visual geoms a rendering env keeps; a training env strips them."""
    return frozenset(getattr(env, "physics_neutral_geom_names", ()))


def _initial_state_fingerprint_source(env, settings: InitialStateSettings) -> str:
    source = (
        f"{_COORDINATES_POSE_RESIDUAL_TRAJECTORY}:{settings.num_components}:"
        f"{model_signature(env.model, exclude_geoms=_physics_neutral_geoms(env))}:"
        f"{getattr(env, 'root_qpos_count', 7)}:"
        f"{getattr(env, 'root_dof_count', 6)}:{settings.residual_scale}:"
        f"sample_arity=2:dt={float(env.step_dt):.12g}:"
    )
    return source + _motion_reference_fingerprint(env)


def _validate_pose_residual_trajectory_contract(env) -> None:
    loop = env.motion_settings.loop
    if loop.smooth or loop.velocity_from_first_frame or loop.velocity_extrapolate:
        raise ValueError(
            "pose_residual_trajectory currently requires a noncyclic "
            "zero-terminal velocity reference"
        )
    if float(env.reset_settings.motion_rand_std) != 0.0:
        raise ValueError(
            "pose_residual_trajectory requires reset.motion_noise_std=0 "
            "so both adjacent poses use the same learned trajectory"
        )


def resolve_settings(env) -> InitialStateSettings:
    cfg = get_block(env.cfg, "initial_state", required=False) or {}
    enabled = bool(get_value(cfg, "enabled", False))
    if not enabled:
        return InitialStateSettings(
            enabled=False,
            num_components=0,
            residual_scale=(),
            evaluation_mode="mean",
        )
    coordinates = str(get_value(cfg, "coordinates", "")).strip().lower()
    if coordinates != _COORDINATES_POSE_RESIDUAL_TRAJECTORY:
        raise ValueError("initial_state.coordinates must be 'pose_residual_trajectory'")
    num_components = int(env.motion_frame_count)
    evaluation_mode = str(get_value(cfg, "evaluation_mode", "mean")).strip().lower()
    if evaluation_mode not in {"mean", "reference", "sample"}:
        raise ValueError(
            "initial_state.evaluation_mode must be one of "
            "{'mean', 'reference', 'sample'}"
        )
    scale_cfg = get_block(cfg, "residual_scale", required=False) or {}
    nv = int(env.model.nv)
    root_dof_count = int(getattr(env, "root_dof_count", 6))
    roles = getattr(env, "dof_roles", None)
    if roles is None or not roles.root_is_free or root_dof_count != 6:
        raise ValueError("pose_residual_trajectory requires a six-DoF free root")
    if roles.has_passive:
        raise ValueError(
            "pose_residual_trajectory does not support passive DOFs "
            "(env.dof_roles); its latent spans every model DOF"
        )
    root_horizontal = _mapping_value(scale_cfg, "root_horizontal_position", 0.01)
    root_height = _mapping_value(scale_cfg, "root_height", 0.01)
    root_rotation = _mapping_value(scale_cfg, "root_rotation", 0.02)
    joint_rotation = _mapping_value(scale_cfg, "joint_rotation", 0.02)
    joint_translation = _mapping_value(scale_cfg, "joint_translation", 0.01)
    values = (
        root_horizontal,
        root_height,
        root_rotation,
        joint_rotation,
        joint_translation,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("initial_state residual scales must be finite and positive")
    scales = [0.0] * nv
    for joint_id in range(int(env.model.njnt)):
        joint_type = int(env.model.jnt_type[joint_id])
        dof_adr = int(env.model.jnt_dofadr[joint_id])
        if joint_type == _JOINT_FREE:
            if dof_adr != 0:
                raise ValueError(
                    "pose_residual_trajectory supports only the root free joint"
                )
            scales[0:3] = [root_horizontal, root_horizontal, root_height]
            scales[3:6] = [root_rotation] * 3
        elif joint_type == _JOINT_BALL:
            scales[dof_adr : dof_adr + 3] = [joint_rotation] * 3
        elif joint_type == _JOINT_SLIDE:
            scales[dof_adr] = joint_translation
        elif joint_type == _JOINT_HINGE:
            scales[dof_adr] = joint_rotation
        else:
            raise ValueError(f"unsupported MuJoCo joint type: {joint_type}")
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("pose_residual_trajectory could not map every generalized DoF")
    return InitialStateSettings(
        enabled=enabled,
        num_components=num_components,
        residual_scale=tuple(scales),
        evaluation_mode=evaluation_mode,
    )


def initialize(env) -> None:
    settings = resolve_settings(env)
    if settings.enabled and bool(getattr(env, "_multi_motion_enabled", False)):
        raise ValueError(
            "learned initial states do not yet support multiple motion clips"
        )
    if not settings.enabled:
        env.initial_state_runtime_state = InitialStateRuntimeState(
            settings=settings, spec=None
        )
        return
    _validate_pose_residual_trajectory_contract(env)
    latent_dim = int(env.model.nv)
    if len(settings.residual_scale) != latent_dim:
        raise RuntimeError("initial-state residual scale layout is inconsistent")
    fingerprint_source = _initial_state_fingerprint_source(env, settings)
    fingerprint = hashlib.sha256(fingerprint_source.encode()).hexdigest()[:16]
    spec = InitialStateSpec(
        num_components=settings.num_components,
        latent_dim=latent_dim,
        coordinates=_COORDINATES_POSE_RESIDUAL_TRAJECTORY,
        residual_scale=settings.residual_scale,
        fingerprint=fingerprint,
        sample_arity=2,
        component_layout="adjacent_frames",
        reference_pose_fingerprint=_reference_pose_fingerprint(env),
    )
    device = env.device
    num_envs = int(env.num_envs)
    residual_scale = torch.tensor(
        settings.residual_scale, device=device, dtype=torch.float32
    )
    state = InitialStateRuntimeState(
        settings=settings,
        spec=spec,
        samples=torch.zeros((num_envs, 2, latent_dim), device=device),
        component_ids=torch.zeros(
            (num_envs, 2),
            device=device,
            dtype=torch.long,
        ),
        old_log_prob=torch.zeros(num_envs, device=device),
        versions=torch.zeros(num_envs, device=device, dtype=torch.long),
        pending_start=torch.zeros(num_envs, device=device, dtype=torch.bool),
        mode=torch.tensor([_MODE_REFERENCE], device=device, dtype=torch.int32),
        collection_enabled=torch.zeros(1, device=device, dtype=torch.int32),
        residual_scale=residual_scale,
        next_qpos=torch.zeros((num_envs, int(env.model.nq)), device=device),
    )
    assert state.mode is not None
    assert state.collection_enabled is not None
    assert state.samples is not None
    assert state.component_ids is not None
    assert state.old_log_prob is not None
    assert state.versions is not None
    assert state.pending_start is not None
    state.residual_scale_wp = wp.from_torch(residual_scale, dtype=wp.float32)
    state.mode_wp = wp.from_torch(state.mode, dtype=wp.int32)
    state.collection_enabled_wp = wp.from_torch(
        state.collection_enabled, dtype=wp.int32
    )
    state.ref_frame_idx_wp = wp.from_torch(env.ref_frame_idx, dtype=wp.int32)
    state.samples_wp = wp.from_torch(state.samples, dtype=wp.float32)
    state.component_ids_wp = wp.from_torch(state.component_ids, dtype=wp.int64)
    state.old_log_prob_wp = wp.from_torch(state.old_log_prob, dtype=wp.float32)
    state.versions_wp = wp.from_torch(state.versions, dtype=wp.int64)
    state.pending_start_wp = wp.from_torch(state.pending_start, dtype=wp.bool)
    state.next_qpos_wp = wp.from_torch(state.next_qpos, dtype=wp.float32)
    env.initial_state_runtime_state = state


def get_spec(env) -> InitialStateSpec | None:
    state = getattr(env, "initial_state_runtime_state", None)
    return None if state is None else state.spec


def bind_parameters(
    env,
    *,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    version: torch.Tensor,
    training: bool,
) -> None:
    state = env.initial_state_runtime_state
    if state is None or state.spec is None:
        if mean.numel() or log_std.numel():
            raise ValueError(
                "initial-state parameters provided to a disabled environment"
            )
        return
    expected = (state.spec.num_components, state.spec.latent_dim)
    if mean.shape != expected or log_std.shape != expected:
        raise ValueError(
            "initial-state parameter layout mismatch: "
            f"expected {expected}, got mean={tuple(mean.shape)} "
            f"log_std={tuple(log_std.shape)}"
        )
    if version.shape != () or version.dtype != torch.long:
        raise ValueError("initial-state version must be a scalar int64 tensor")
    if (
        not _same_torch_device(mean.device, env.device)
        or not _same_torch_device(log_std.device, env.device)
        or not _same_torch_device(version.device, env.device)
    ):
        raise ValueError("initial-state parameters must share the environment device")
    state.mean_wp = wp.from_torch(mean, dtype=wp.float32)
    state.log_std_wp = wp.from_torch(log_std, dtype=wp.float32)
    state.version_wp = wp.from_torch(version.reshape(1), dtype=wp.int64)
    state.collection_enabled.fill_(1 if training else 0)
    mode_name = "sample" if training else state.settings.evaluation_mode
    state.mode.fill_(_SUPPORTED_MODES[mode_name])
    state.pending_start.zero_()


def apply_after_reference_reset(env, reset_mask: torch.Tensor) -> None:
    state = getattr(env, "initial_state_runtime_state", None)
    if state is None or state.spec is None:
        return
    if state.mean_wp is None or state.log_std_wp is None or state.version_wp is None:
        raise RuntimeError(
            "learned initial state is enabled but PPO parameters are not bound"
        )
    assert state.samples is not None
    assert state.component_ids is not None
    assert state.old_log_prob is not None
    assert state.versions is not None
    assert state.pending_start is not None
    assert state.mode is not None
    assert state.collection_enabled is not None
    assert state.residual_scale_wp is not None
    assert state.mode_wp is not None
    assert state.collection_enabled_wp is not None
    assert state.ref_frame_idx_wp is not None
    assert state.samples_wp is not None
    assert state.component_ids_wp is not None
    assert state.old_log_prob_wp is not None
    assert state.versions_wp is not None
    assert state.pending_start_wp is not None
    if state.next_qpos_wp is None:
        raise RuntimeError("pose-residual-trajectory scratch is unavailable")
    wp.launch(
        _sample_and_apply_pose_residual_trajectory_kernel,
        dim=(env.num_envs,),
        inputs=[
            reset_mask,
            env.reset_runtime_state.seed_wp,
            env.common_step_counter,
            state.ref_frame_idx_wp,
            int(env.motion_frame_count),
            float(env.step_dt),
            int(env.model.nq),
            int(env.model.nv),
            int(env.model.njnt),
            state.residual_scale_wp,
            state.mean_wp,
            state.log_std_wp,
            state.version_wp,
            state.mode_wp,
            state.collection_enabled_wp,
            env.motion_qpos,
            env.reset_runtime_state.spawn_offset_vec3,
            env.mjw_model.jnt_type,
            env.mjw_model.jnt_qposadr,
            env.mjw_model.jnt_dofadr,
            env.mjw_data.qpos,
            env.mjw_data.qvel,
        ],
        outputs=[
            state.next_qpos_wp,
            state.samples_wp,
            state.component_ids_wp,
            state.old_log_prob_wp,
            state.versions_wp,
            state.pending_start_wp,
        ],
    )


def sample_metadata(env) -> tuple[torch.Tensor, ...] | None:
    state = env.initial_state_runtime_state
    if state is None or state.spec is None:
        return None
    assert state.pending_start is not None
    assert state.samples is not None
    assert state.component_ids is not None
    assert state.old_log_prob is not None
    assert state.versions is not None
    return (
        state.pending_start,
        state.samples,
        state.component_ids,
        state.old_log_prob,
        state.versions,
    )


def clear_pending(env) -> None:
    state = env.initial_state_runtime_state
    if state is not None and state.pending_start is not None:
        state.pending_start.zero_()


__all__ = [
    "InitialStateRuntimeState",
    "InitialStateSettings",
    "apply_after_reference_reset",
    "bind_parameters",
    "clear_pending",
    "get_spec",
    "initialize",
    "resolve_settings",
    "sample_metadata",
]
