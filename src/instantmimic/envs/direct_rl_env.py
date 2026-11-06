"""Base environment for tasks that compute observations and rewards themselves.

Portions derived from Isaac Lab (BSD-3-Clause); see THIRD_PARTY_NOTICES.md.
The class name, lifecycle hooks and construction order follow
`isaaclab/envs/direct_rl_env.py`, and several explanatory comments, the
unseeded-environment warning and the startup banner's column alignment are
carried over from it verbatim.
"""

from __future__ import annotations

import copy
import math
from abc import abstractmethod
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import FrozenInstanceError, dataclass
from logging import getLogger
from typing import Any, ClassVar, ContextManager, Protocol

import gymnasium as gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import (
    ConfigAttributeError,
    MissingMandatoryValue,
    ReadonlyConfigError,
)

from instantmimic.envs.common import VecEnvObs, VecEnvStepReturn
from instantmimic.envs.config_access import get_value
from instantmimic.envs.control_pipeline import (
    make_direct_control_runtime_spec,
    resolve_control_pipeline,
)
from instantmimic.rendering.sim_assets import (
    physics_neutral_visual_geom_names,
    strip_physics_neutral_visual_assets,
)
from instantmimic.utils.ball_joint_limits import configure_ball_joint_limits
from instantmimic.utils.mjw_kernels.private import (
    collision_driver as mjw_collision_driver,
)
from instantmimic.utils.mjw_kernels.private import default_njmax_nnz
from instantmimic.utils.mjw_model import (
    DENSE_NV_MAX,
    promote_dense_jacobian_for_fused_newton,
    put_model,
)
from instantmimic.utils.mjw_step_optimized import (
    com_pos_vel_parallel,
    forward_after_state_prefix_optimized_fwd_position,
    forward_optimized_fwd_position,
    forward_state_prefix_optimized_fwd_position,
    kinematics_com_state_optimized_fwd_position,
    kinematics_parallel_tail,
    make_optimized_fwd_position_scratch,
    make_optimized_fwd_position_streams,
    optimized_kernel_capabilities,
    state_prefix_can_defer_constraints,
    step_after_state_prefix_optimized_fwd_position,
    step_optimized_fwd_position,
    step_skip_kinematics_optimized_fwd_position,
    supports_allocation_free_implicit_capture,
)
from instantmimic.utils.project_paths import resolve_project_path
from instantmimic.utils.spaces import sample_space, spec_to_gym_space
from instantmimic.utils.startup_profiler import StartupProfiler, startup_nvtx_range
from instantmimic.utils.timer import Timer
from instantmimic.utils.torch_warp_capture import (
    capture_warp_graph,
    warp_stream_from_torch,
)

from .direct_rl_env_cfg import DirectRLEnvCfg, SimulationCfg
from .direct_rl_rendering import DirectRLRenderRuntime
from .observation_noise import (
    OBSERVATION_NOISE_CONFIG_KEY,
    OBSERVATION_NOISE_SLOT_RESET,
    OBSERVATION_NOISE_SLOT_STEP,
    OBSERVATION_NOISE_SLOT_TERMINAL,
    ObservationNoiseRuntime,
    ObservationNoiseSettings,
    parse_observation_noise_settings,
)
from .substep_integrators import (
    IntegratorSpec,
    SubstepIntegrator,
    resolve_integrator_spec,
)

logger = getLogger(__name__)

_CFG_ABSENT = object()
_CFG_MANDATORY_MISSING = object()


class BufferContractError(RuntimeError):
    """Raised when an environment hook replaces a framework-bound buffer slot."""


@dataclass(frozen=True)
class StepInputs:
    """Inputs bound for one native environment step."""

    actions: torch.Tensor


@dataclass(frozen=True)
class StepOutputs:
    """Structured, preallocated outputs bound for one environment step."""

    observations: Mapping[str, torch.Tensor]
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


@dataclass(frozen=True)
class StepIO:
    """Framework-owned or caller-bound storage for one environment step."""

    inputs: StepInputs
    outputs: StepOutputs


class StepLifecycleComponent(Protocol):
    """Framework-owned behavior around the canonical step pipeline.

    ``before_step`` runs in registration order. After a successful bound step,
    ``after_step`` runs in reverse registration order.
    """

    def before_step(self, step_io: StepIO) -> None: ...

    def after_step(self, step_io: StepIO) -> None: ...

    def prepare_capture_window(self) -> None: ...

    def finalize_capture_window(self) -> None: ...


@contextmanager
def external_step_capture_window(env: Any, *, unroll_solver_iterations: bool = False):
    """Prepare an external graph, optionally flattening nested solver conditionals.

    The benchmark owns an outer conditional loop. CUDA disallows conditional nodes
    in its child graphs, so record bounded native solver iterations there. Native
    done-world guards retain early convergence; graph/compile settings stay intact.
    """
    prepare = getattr(env, "prepare_step_capture_window", None)
    finalize = getattr(env, "finalize_step_capture_window", None)
    if callable(prepare):
        prepare()
    scratch = getattr(env, "_optimized_fwd_position_scratch", None)
    previous = getattr(scratch, "unroll_solver_iterations", False)
    if unroll_solver_iterations and scratch is not None:
        scratch.unroll_solver_iterations = True
    try:
        try:
            yield
        except BaseException:
            if callable(finalize):
                try:
                    finalize()
                except Exception:
                    logger.exception(
                        "Step capture-window finalization failed while preserving "
                        "the primary capture/replay exception."
                    )
            raise
        else:
            if callable(finalize):
                finalize()
    finally:
        if unroll_solver_iterations and scratch is not None:
            scratch.unroll_solver_iterations = previous


def _env_graph_capture_supported(device: torch.device) -> bool:
    """Env step graphs record torch and Warp work together, which needs CUDA."""
    return torch.device(device).type == "cuda"


def _cuda_current_stream_is_capturing(device: torch.device) -> bool:
    with torch.cuda.device(device):
        return bool(torch.cuda.is_current_stream_capturing())


def _warp_stream_is_capturing(stream: Any | None) -> bool:
    return bool(stream is not None and getattr(stream, "is_capturing", False))


def _warp_device_is_capturing(device: torch.device) -> bool:
    return bool(getattr(wp.get_device(str(device)), "is_capturing", False))


def _current_stream_is_capturing(
    device: torch.device,
    wp_stream: Any | None = None,
) -> bool:
    """Whether an outer torch or Warp capture already records step work.

    Warp and torch work captured by an outer graph must run directly: CUDA
    forbids starting a nested capture on a stream that is already recording.
    A Warp-only capture need not have torch's current stream active, so both
    the environment's Warp graph stream and the current Warp device are
    consulted in addition to torch's current stream.
    """

    if torch.device(device).type != "cuda":
        return False
    # Read Warp's host-side capture flags first. Querying torch's current
    # stream calls into CUDA, and while a blocking stream captures in global
    # mode that query on the legacy stream itself fails with
    # cudaErrorStreamCaptureImplicit.
    if _warp_stream_is_capturing(wp_stream) or _warp_device_is_capturing(device):
        return True
    return _cuda_current_stream_is_capturing(device)


@contextmanager
def _stream_aligned_cuda_graph_capture(
    device: torch.device,
    torch_stream: torch.cuda.Stream | None = None,
    wp_stream: Any | None = None,
):
    if torch.device(device).type != "cuda":
        with wp.ScopedCapture() as capture:
            yield capture
        return

    if torch_stream is None or wp_stream is None:
        torch_stream, wp_stream = make_stream_aligned_cuda_graph_streams(device)
    current_stream = torch.cuda.current_stream(device)
    torch_stream.wait_stream(current_stream)
    with wp.ScopedStream(wp_stream, sync_enter=False, sync_exit=False):
        with torch.cuda.stream(torch_stream):
            with capture_warp_graph(stream=wp_stream) as capture:
                yield capture
    current_stream.wait_stream(torch_stream)


@contextmanager
def _startup_profile_phase(
    profiler: StartupProfiler | None,
    name: str,
):
    if profiler is None:
        with nullcontext():
            yield
        return
    with profiler.phase(name):
        yield


def _launch_stream_aligned_cuda_graph(
    graph: Any,
    device: torch.device,
    torch_stream: torch.cuda.Stream | None = None,
    wp_stream: Any | None = None,
) -> None:
    if torch.device(device).type != "cuda":
        wp.capture_launch(graph)
        return

    if torch_stream is None or wp_stream is None:
        torch_stream, wp_stream = make_stream_aligned_cuda_graph_streams(device)
    current_stream = torch.cuda.current_stream(device)
    torch_stream.wait_stream(current_stream)
    with wp.ScopedStream(wp_stream, sync_enter=False, sync_exit=False):
        with torch.cuda.stream(torch_stream):
            wp.capture_launch(graph, stream=wp_stream)
    current_stream.wait_stream(torch_stream)


def make_stream_aligned_cuda_graph_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, Any]:
    wp.init()
    torch_stream = torch.cuda.Stream(device=device)
    wp_stream = warp_stream_from_torch(torch_stream)
    return torch_stream, wp_stream


@wp.kernel(enable_backward=False)
def _reset_episode_length_kernel(
    episode_length_buf: wp.array(dtype=wp.int64),
    reset_mask: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()
    if reset_mask[env_idx]:
        episode_length_buf[env_idx] = wp.int64(0)


@wp.kernel(enable_backward=False)
def _increment_episode_length_kernel(
    episode_length_buf: wp.array(dtype=wp.int64),
):
    env_idx = wp.tid()
    episode_length_buf[env_idx] = episode_length_buf[env_idx] + wp.int64(1)


@wp.kernel(enable_backward=False)
def _increment_step_counter_kernel(
    step_counter: wp.array(dtype=wp.int32),
):
    step_counter[0] = step_counter[0] + wp.int32(1)


@wp.kernel(enable_backward=False)
def _logical_or_kernel(
    a: wp.array(dtype=wp.bool),
    b: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    out[i] = a[i] or b[i]


@wp.kernel(enable_backward=False)
def _logical_or_inplace_kernel(
    dst: wp.array(dtype=wp.bool),
    src: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    dst[i] = dst[i] or src[i]


@wp.kernel(enable_backward=False)
def _mark_nonfinite_state_envs_kernel(
    qpos: wp.array2d(dtype=float),
    qvel: wp.array2d(dtype=float),
    out: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()
    bad = int(0)
    for q_idx in range(qpos.shape[1]):
        if not wp.isfinite(qpos[env_idx, q_idx]):
            bad = int(1)
            break
    if bad == 0:
        for v_idx in range(qvel.shape[1]):
            if not wp.isfinite(qvel[env_idx, v_idx]):
                bad = int(1)
                break
    out[env_idx] = bad != 0


@wp.kernel(enable_backward=False)
def _sanitize_nonfinite_rewards_kernel(
    reward_buf: wp.array(dtype=float),
    state_bad_mask: wp.array(dtype=wp.bool),
    reset_terminated: wp.array(dtype=wp.bool),
    reset_buf: wp.array(dtype=wp.bool),
    combined_bad_mask: wp.array(dtype=wp.bool),
):
    env_idx = wp.tid()
    reward = reward_buf[env_idx]
    reward_bad = not wp.isfinite(reward)
    combined_bad = state_bad_mask[env_idx] or reward_bad
    combined_bad_mask[env_idx] = combined_bad
    if combined_bad:
        reset_terminated[env_idx] = True
        reset_buf[env_idx] = True
        reward_buf[env_idx] = 0.0


@wp.kernel(enable_backward=False)
def _accumulate_done_reason_stats_kernel(
    terminated: wp.array(dtype=wp.bool),
    time_outs: wp.array(dtype=wp.bool),
    nonfinite_mask: wp.array(dtype=wp.bool),
    reason_masks: wp.array2d(dtype=wp.bool),
    reason_counts: wp.array(dtype=wp.int32),
    terminated_total: wp.array(dtype=wp.int32),
    time_out_total: wp.array(dtype=wp.int32),
    nonfinite_total: wp.array(dtype=wp.int32),
):
    env_idx = wp.tid()

    if terminated[env_idx]:
        wp.atomic_add(terminated_total, 0, wp.int32(1))
        for reason_idx in range(reason_masks.shape[0]):
            if reason_masks[reason_idx, env_idx]:
                wp.atomic_add(reason_counts, reason_idx, wp.int32(1))

    if time_outs[env_idx]:
        wp.atomic_add(time_out_total, 0, wp.int32(1))

    if nonfinite_mask[env_idx]:
        wp.atomic_add(nonfinite_total, 0, wp.int32(1))


def _apply_sim_option_overrides(spec: mujoco.MjSpec, sim_cfg: SimulationCfg) -> None:
    iterations = getattr(sim_cfg, "iterations", None)
    if iterations is not None:
        spec.option.iterations = int(iterations)

    ls_iterations = getattr(sim_cfg, "ls_iterations", None)
    if ls_iterations is not None:
        spec.option.ls_iterations = int(ls_iterations)

    is_sparse = getattr(sim_cfg, "is_sparse", None)
    if is_sparse is not None:
        spec.option.jacobian = (
            mujoco.mjtJacobian.mjJAC_SPARSE
            if bool(is_sparse)
            else mujoco.mjtJacobian.mjJAC_DENSE
        )

    disable_multiccd = getattr(sim_cfg, "disable_multiccd", None)
    if disable_multiccd is not None:
        multiccd_bit = int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        if bool(disable_multiccd):
            spec.option.disableflags = int(spec.option.disableflags) | multiccd_bit
        else:
            spec.option.disableflags = int(spec.option.disableflags) & ~multiccd_bit


def _nvtx_method(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a method in an NVTX range, or return it untouched off CUDA.

    `reset_into`, `step` and `step_with_io` used bare
    `@torch.cuda.nvtx.range(...)` decorators, which push a range on every call
    with no way off and make a CPU run pay for NVTX it cannot use. Deciding once
    here keeps the range on CUDA -- these are the outer ranges a profile needs
    around graph replay -- while a CPU import gets the identity decorator, so
    the hot path gains no per-call indirection either way.
    """

    if not torch.cuda.is_available():
        return lambda fn: fn
    return torch.cuda.nvtx.range(name)


class DirectRLEnv(gym.Env):
    """Base class for tasks that compute observations and rewards themselves.

    A subclass owns `_get_observations`, `_get_rewards`, `_get_dones` and either
    `_pre_physics_step`/`_apply_action` or a control pipeline. This class owns the
    MuJoCo Warp model, the batched data, the reset bookkeeping and the step
    orchestration -- including CUDA graph capture of the whole step body when
    `use_cuda_graph` is set, which is why per-step allocation is forbidden in
    subclass hooks.

    All `num_envs` copies share one model and one origin. `reset` therefore
    reinitializes every copy at once; call it once before the first `step`, after
    which `step` resets terminated copies in place.

    It subclasses `gym.Env` rather than `gym.vector.VectorEnv` while returning
    batched tensors: the vector API's async and wait methods have no meaning for a
    single fused GPU step, and each RL library wraps batched environments its own
    way regardless.
    """

    is_vector_env: ClassVar[bool] = True
    """Marks the batched return contract for library wrappers that check it."""
    _is_closed: ClassVar[bool] = True
    _render_runtime: ClassVar[DirectRLRenderRuntime | None] = None
    parallelize_dones_and_rewards: ClassVar[bool] = False
    """Whether dones and rewards can be computed concurrently after physics."""
    supports_debug_vis: ClassVar[bool] = False
    """Whether this env implements ``_set_debug_vis_impl``."""
    boundary_reads_constraints: ClassVar[bool] = True
    """Whether the task reads contacts or constraint rows (``d.contact``,
    ``d.efc``, ``nacon``, ``nefc``) at a policy boundary: in the last substep's
    ``_post_state_prefix_substep`` or substep reward, in rewards, dones and
    terminal observations, or in the reset hooks before the reset prefix. A
    class that sets it ``False`` lets an auto-resetting step skip collision and
    constraint construction in the last substep's endpoint refresh, since the
    reset prefix rebuilds them for every world. The declaration is not
    inherited (``__init_subclass__`` resets it): a subclass keeps the full
    refresh until it declares again."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
    }
    """Gymnasium metadata.

    Each environment receives a deep copy before construction writes `render_fps`
    from `step_dt`, so nested mutable metadata such as `render_modes` is local to
    that environment.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A subclass may add contact reads its parent's audit did not cover.
        if "boundary_reads_constraints" not in cls.__dict__:
            cls.boundary_reads_constraints = True

    def _resolve_cfg_path_in_place(self, container: Any, key: str) -> str:
        raw_value = getattr(container, key)
        resolved = str(resolve_project_path(str(raw_value)))
        try:
            setattr(container, key, resolved)
        except (ConfigAttributeError, FrozenInstanceError, ReadonlyConfigError):
            return resolved
        return resolved

    def _build_base_sim_spec(self, mjcf_path: str) -> mujoco.MjSpec:
        """Build the simulation spec from the resolved source MJCF path."""
        return mujoco.MjSpec.from_file(mjcf_path)

    def _configure_task_sim_spec(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """Apply task-owned physics changes after control builds the spec.

        Control pipelines may replace the source spec while preparing their
        runtime representation. Task physics customization therefore belongs
        after that boundary rather than only in ``_build_base_sim_spec``.
        """

        return spec

    def _external_sim_geom_dependencies(self) -> set[str]:
        """Names of geoms consumed by physics code outside MuJoCo.

        Subclasses must declare non-colliding, massless geoms read by custom
        contact, reward, observation, or reset code so visual stripping keeps
        them even though the MJCF compiler cannot see those references.
        """

        return set()

    def _ball_joint_limit_mode(self) -> str:
        """How ``sim.ball_joint_limits`` are enforced: ``solver`` or ``force``.

        With ``force`` the limits leave the constraint solver and are exposed as
        ``_force_ball_joint_limits`` for the task to apply.
        """

        return str(getattr(self.cfg.sim, "ball_joint_limit_mode", "solver"))

    def _build_sim_spec(self, mjcf_path: str) -> mujoco.MjSpec:
        """Build the simulation spec, allowing optional control transformation."""
        control = self._control_pipeline
        if control is not None and control.is_enabled:
            spec = control.build_sim_spec(mjcf_path, self._build_base_sim_spec)
        else:
            spec = self._build_base_sim_spec(mjcf_path)
        spec = self._configure_task_sim_spec(spec)
        cfg = getattr(self, "cfg", None)
        preserve_visuals = bool(getattr(cfg, "preserve_sim_visual_assets", False))
        external_geom_dependencies = set(self._external_sim_geom_dependencies())
        if control is not None and control.is_enabled:
            external_geom_dependencies |= control.external_sim_geom_dependencies()
        if self.render_mode is not None or preserve_visuals:
            # Kept for rendering only; physics identities must ignore them so a
            # rendered run matches the stripped training model.
            self.physics_neutral_geom_names = physics_neutral_visual_geom_names(
                spec, external_geom_dependencies=external_geom_dependencies
            )
        else:
            self.physics_neutral_geom_names = frozenset()
            summary = strip_physics_neutral_visual_assets(
                spec,
                external_geom_dependencies=external_geom_dependencies,
            )
            if summary.geoms or summary.meshes:
                logger.info(
                    "SIM_PHYSICS_NEUTRAL_VISUAL_ASSETS_STRIPPED "
                    "geoms=%d meshes=%d materials=%d textures=%d skins=%d",
                    summary.geoms,
                    summary.meshes,
                    summary.materials,
                    summary.textures,
                    summary.skins,
                )
        return spec

    def _initialize_runtime_state(self) -> None:
        """Initialize base-owned state before construction invokes any hooks.

        SNU environments may attach a control pipeline before calling
        ``super().__init__`` because that pipeline participates in model building.
        Preserve that one documented pre-super hook while initializing every
        optional base slot explicitly.
        """

        control_pipeline = self.__dict__.get("_control_pipeline")
        lifecycle_components = self.__dict__.get("_step_lifecycle_components")
        self._control_pipeline = control_pipeline
        self._diagnostic_stats_sources: list[Callable[[], dict[str, float]]] = []
        # Fused-path sparse Hessian pattern-violation counter (torch view), checked at
        # the metric-sync cadence.
        self._sparse_hessian_violations: torch.Tensor | None = None
        self._diagnostic_stats_sources.append(self._check_sparse_hessian_violations)
        self._reset_state_finalizers: list[Callable[[torch.Tensor], None]] = []
        self._render_runtime: DirectRLRenderRuntime | None = None
        self._render_capture_env_idx_value = 0
        self._render_phase_before_last_step_value = 0
        self._is_closed = False
        self._startup_reset_profile_logged = False
        self._startup_reset_profiler: StartupProfiler | None = None
        self._reset_all_mask: torch.Tensor | None = None
        self._default_reward_buf: torch.Tensor | None = None
        self._default_terminated_buf: torch.Tensor | None = None
        self._default_truncated_buf: torch.Tensor | None = None
        self._default_step_outputs: StepOutputs | None = None
        self._step_lifecycle_components = (
            [] if lifecycle_components is None else lifecycle_components
        )
        self._step_lifecycle_registration_frozen = False
        self._post_physics_substep_hooks: list[Callable[[int], None]] = []
        # The native substep; __init__ resolves sim.integrator and replaces it
        # on first use when the spec builds one.
        self._integrator_spec: IntegratorSpec | None = None
        self._substep_integrator = SubstepIntegrator(self)
        self._step_graph = None
        self._step_graph_cache: dict[tuple[Any, ...], Any] = {}
        self._step_pipeline_active = False
        self._dones_torch_stream = None
        self._dones_wp_stream = None
        self._rewards_torch_stream = None
        self._rewards_wp_stream = None
        self._mjw_step_capture_supported: bool | None = None
        self._mjw_step_capture_unsupported_logged = False
        self._optimized_fwd_position_scratch = None
        self._nonfinite_state_mask_buf: torch.Tensor | None = None
        self._nonfinite_combined_bad_mask_buf: torch.Tensor | None = None
        self._terminal_obs_buf: dict[str, torch.Tensor] | None = None
        # Parsed `env.observation_noise`; the runtime exists only while the
        # training runtime has enabled noise on this instance.
        self._observation_noise_settings: ObservationNoiseSettings | None = None
        self._observation_noise_runtime: ObservationNoiseRuntime | None = None
        # Derived tasks may replace their observation layout while construction
        # is still layering inherited suffixes. Once reset/step/capture begins
        # this contract is sealed because framework bindings need stable
        # identities for the remainder of the episode.
        self._derived_spaces_sealed = False
        self._derived_spaces_fingerprint: tuple[Any, ...] | None = None
        self._derived_spaces_pending = False
        self._substep_rewards_enabled = False
        self._validate_step_buffers = False
        self._use_optimized_kernel = False

    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Build the MuJoCo Warp model and every batched buffer the step needs.

        Args:
            cfg: Resolved task configuration. Fields declared `MISSING` must be set.
            render_mode: One of `metadata["render_modes"]`. `None` skips the
                renderer and camera setup, though visual elements are still added
                to the MuJoCo spec so playback can attach a camera later.
            **kwargs: Accepted so a subclass can pass its own arguments through
                `super().__init__`; this class ignores them.
        """
        self.metadata = copy.deepcopy(type(self).metadata)
        self._initialize_runtime_state()

        # store inputs to class
        self.cfg = cfg
        self._integrator_spec = type(self)._resolve_integrator_spec(cfg.sim.integrator)
        self._observation_noise_settings = parse_observation_noise_settings(
            get_value(cfg, OBSERVATION_NOISE_CONFIG_KEY, None)
        )
        # store the render mode
        self.render_mode = render_mode
        # initialize internal variables
        self._physics_dt = self._resolve_physics_dt(self.cfg.sim)
        self._render_runtime = DirectRLRenderRuntime(self)

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            logger.warning(
                "Seed not set for the environment. The environment creation may not be deterministic."
            )

        # make sure torch is running on the correct device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # The specialized-kernel policy decides whether model upload resolves an
        # `auto` Jacobian to dense for the fused Newton path, so resolve it
        # before scene creation uploads the model.
        self._use_optimized_kernel = self._resolve_use_optimized_kernel()
        control_pipeline = self._control_pipeline
        if control_pipeline is None:
            control_pipeline = resolve_control_pipeline(
                getattr(self.cfg, "control", None)
            )
        self._control_pipeline = control_pipeline
        env_startup_profiler = StartupProfiler(
            range_factory=lambda name: startup_nvtx_range(f"direct_rl_env:{name}")
        )

        # print useful information
        logger.info("Base environment:")
        logger.info(f"\tEnvironment device    : {self.device}")
        logger.info(f"\tEnvironment seed      : {self.cfg.seed}")
        logger.info(f"\tPhysics step-size     : {self.physics_dt}")
        logger.info(
            f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}"
        )
        logger.info(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"render_interval ({self.cfg.sim.render_interval}) is below decimation "
                f"({self.cfg.decimation}), so a single environment step renders more "
                "than once. Set render_interval equal to decimation to render once."
            )
            logger.warning(msg)

        # generate scene
        with (
            Timer("[INFO]: Time taken for scene creation", "scene_creation"),
            startup_nvtx_range("direct_rl_env:scene_creation"),
        ):
            with env_startup_profiler.phase("resolve_mjcf_path"):
                mjcf_path = self._resolve_cfg_path_in_place(self.cfg.sim, "mjcf_path")
            with env_startup_profiler.phase("build_sim_spec"):
                spec = self._build_sim_spec(mjcf_path)
            with env_startup_profiler.phase("configure_sim_spec"):
                spec.option.timestep = self.physics_dt
                spec.option.integrator = self._integrator_spec.native
                _apply_sim_option_overrides(spec, self.cfg.sim)
                self._render_runtime.configure_spec_visuals(spec)

            with env_startup_profiler.phase("compile_mjmodel"):
                self.model: mujoco.MjModel = mujoco.MjSpec.compile(spec)
                if int(self.model.nu) != int(self.model.nactuator):
                    # env controls index actuators by control column
                    raise ValueError(
                        "multi-input actuators are unsupported: model has"
                        f" nu={int(self.model.nu)} controls for"
                        f" {int(self.model.nactuator)} actuators"
                    )
                ball_limit_mode = self._ball_joint_limit_mode()
                resolved_ball_limits = configure_ball_joint_limits(
                    self.model,
                    getattr(self.cfg.sim, "ball_joint_limits", ()),
                    enforcement=ball_limit_mode,
                )
                self._ball_joint_limits = (
                    resolved_ball_limits if ball_limit_mode == "solver" else ()
                )
                self._force_ball_joint_limits = (
                    resolved_ball_limits if ball_limit_mode == "force" else ()
                )
            with env_startup_profiler.phase("mjw_put_model"):
                self._upload_mjw_model()
            with env_startup_profiler.phase("create_mjdata"):
                self.data = mujoco.MjData(self.model)
            with env_startup_profiler.phase("mjw_put_data"):
                constraint_capacity = self.cfg.sim.njmax
                extra_data_options = {}
                if self._ball_joint_limits:
                    if constraint_capacity <= 0:
                        raise ValueError(
                            "ball_joint_limits requires a positive sim.njmax"
                        )
                    extra_rows = 3 * len(self._ball_joint_limits)
                    constraint_capacity += extra_rows
                    extra_data_options["njmax_nnz"] = (
                        default_njmax_nnz(
                            self.model, self.cfg.sim.nconmax, constraint_capacity
                        )
                        + 3 * extra_rows
                    )
                self.mjw_data = mjw.put_data(
                    self.model,
                    self.data,
                    nworld=self.num_envs,
                    njmax=constraint_capacity,
                    nconmax=self.cfg.sim.nconmax,
                    **extra_data_options,
                )
            with env_startup_profiler.phase("render_initialize_for_model"):
                self._render_runtime.initialize_for_model()

        with env_startup_profiler.phase("base_buffer_alloc"):
            # allocate dictionary to store metrics
            self.extras = {}

            # initialize data and constants
            # -- counter for simulation steps
            # -- counter for curriculum (GPU tensor for CUDA graph compatibility)
            self.common_step_counter = torch.zeros(
                1, device=self.device, dtype=torch.int32
            )
            # -- init buffers
            self.episode_length_buf = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self.reset_terminated = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
            self.reset_time_outs = torch.zeros_like(self.reset_terminated)
            self.reset_buf = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self.reward_buf = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self._default_reward_buf = self.reward_buf
            self._default_terminated_buf = self.reset_terminated
            self._default_truncated_buf = self.reset_time_outs
            self._nonfinite_state_mask_buf = torch.zeros_like(self.reset_buf)
            self._nonfinite_combined_bad_mask_buf = torch.zeros_like(self.reset_buf)
            self._track_done_reasons = bool(
                getattr(self.cfg, "track_done_reasons", False)
            )
            self._done_reason_labels: tuple[str, ...] = ()
            self._done_reason_mask_buf = torch.zeros(
                (0, self.num_envs), dtype=torch.bool, device=self.device
            )
            self._done_reason_count_buf = torch.zeros(
                0, dtype=torch.int32, device=self.device
            )
            self._done_reason_terminated_total_buf = torch.zeros(
                1, dtype=torch.int32, device=self.device
            )
            self._done_reason_time_out_total_buf = torch.zeros(
                1, dtype=torch.int32, device=self.device
            )
            self._done_reason_nonfinite_total_buf = torch.zeros(
                1, dtype=torch.int32, device=self.device
            )
            self._done_reason_empty_mask_buf = torch.zeros_like(self.reset_buf)

        with env_startup_profiler.phase("control_finalize_after_model_compile"):
            self._control_pipeline.finalize_after_model_compile(self)
        with env_startup_profiler.phase("control_configure_spaces"):
            self._control_pipeline.configure_spaces(self)

        # setup the action and observation spaces for Gym
        with env_startup_profiler.phase("configure_gym_env_spaces"):
            self._configure_gym_env_spaces()

        # set the framerate of the gym video recorder wrapper so that the playback speed of the produced
        # video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        with env_startup_profiler.phase("graph_runtime_state_init"):
            # CUDA graph capture state
            self._step_graph = None
            self._step_graph_cache: dict[tuple[Any, ...], Any] = {}
            self._step_pipeline_active = False
            self._mjw_step_graph = None
            self._mjw_step_skip_kinematics_graph = None
            self._mjw_step_after_state_prefix_graph = None
            self._mjw_step_parallelized_graph = None
            self._mjw_step_skip_parallelized_graph = None
            self._mjw_step_after_state_prefix_parallelized_graph = None
            self._env_graph_torch_stream = None
            self._env_graph_wp_stream = None
            self._dones_torch_stream = None
            self._dones_wp_stream = None
            self._rewards_torch_stream = None
            self._rewards_wp_stream = None
            self._use_parallelized_kernel_launch = (
                self._resolve_use_parallelized_kernel_launch()
            )
            self._optimized_fwd_position_streams = (
                make_optimized_fwd_position_streams(self.device)
                if self._use_parallelized_kernel_launch
                else ()
            )
            self._optimized_fwd_position_scratch = make_optimized_fwd_position_scratch(
                self.mjw_model,
                self.mjw_data,
                self.device,
                include_optimized_kernel_scratch=self._use_optimized_kernel,
                ball_joint_limits=self._ball_joint_limits,
            )
            logger.info(
                "MJW_KERNEL_PATH optimized=%s parallel=%s nv=%d warn_overflow=%s capabilities=%s",
                self._use_optimized_kernel,
                self._use_parallelized_kernel_launch,
                self.mjw_model.nv,
                self.mjw_model.opt.warn_overflow,
                optimized_kernel_capabilities(
                    self.mjw_model,
                    self._optimized_fwd_position_scratch,
                    enabled=self._use_optimized_kernel,
                ),
            )
            self._substep_rewards_enabled = self._resolve_substep_rewards_enabled()
            self._refresh_mjw_sparse_hessian()
        # print the environment information
        logger.info("Completed setting up the environment...")
        env_startup_profiler.log_summary(
            logger=logger,
            prefix="ENV_STARTUP_PROFILE",
        )

    def __del__(self):
        """Last-resort close, including subclasses that fail before ``super``."""
        self.close()

    def _get_env_graph_streams(self) -> tuple[torch.cuda.Stream | None, Any | None]:
        if torch.device(self.device).type != "cuda":
            return None, None
        if self._env_graph_torch_stream is None or self._env_graph_wp_stream is None:
            (
                self._env_graph_torch_stream,
                self._env_graph_wp_stream,
            ) = make_stream_aligned_cuda_graph_streams(self.device)
        return self._env_graph_torch_stream, self._env_graph_wp_stream

    def _get_dones_rewards_parallel_streams(
        self,
    ) -> tuple[torch.cuda.Stream, Any, torch.cuda.Stream, Any]:
        if self._dones_torch_stream is None or self._dones_wp_stream is None:
            (
                self._dones_torch_stream,
                self._dones_wp_stream,
            ) = make_stream_aligned_cuda_graph_streams(self.device)
        if self._rewards_torch_stream is None or self._rewards_wp_stream is None:
            (
                self._rewards_torch_stream,
                self._rewards_wp_stream,
            ) = make_stream_aligned_cuda_graph_streams(self.device)
        return (
            self._dones_torch_stream,
            self._dones_wp_stream,
            self._rewards_torch_stream,
            self._rewards_wp_stream,
        )

    def _can_parallelize_dones_and_rewards(self) -> bool:
        return bool(self.parallelize_dones_and_rewards)

    def _resolve_substep_rewards_enabled(self) -> bool:
        reward_cfg = getattr(self.cfg, "reward", None)
        enabled = bool(
            False if reward_cfg is None else getattr(reward_cfg, "substep_mean", False)
        )
        return enabled

    def _resolve_use_parallelized_kernel_launch(self) -> bool:
        enabled = bool(getattr(self.cfg, "use_parallelized_kernel_launch", False))
        if enabled and torch.device(self.device).type != "cuda":
            raise ValueError("env.use_parallelized_kernel_launch=true requires CUDA.")
        return enabled

    def _resolve_use_optimized_kernel(self) -> bool:
        enabled = bool(getattr(self.cfg, "use_optimized_kernel", False))
        if enabled and torch.device(self.device).type != "cuda":
            raise ValueError("env.use_optimized_kernel=true requires CUDA.")
        return enabled

    def _resolve_dense_jacobian_for_upload(self) -> bool:
        """Promote the compiled `auto` Jacobian for the fused Newton path.

        Uses the resolved device policy, so a config that authored
        `use_optimized_kernel` on cannot promote a Jacobian on a device where the
        specialized kernels are off. Returns whether the model changed.
        """

        if not promote_dense_jacobian_for_fused_newton(
            self.model,
            use_optimized_kernel=self._use_optimized_kernel,
            authored_is_sparse=getattr(self.cfg.sim, "is_sparse", None),
        ):
            return False
        logger.info(
            "JACOBIAN_DENSE_AUTO_RESOLVED nv=%d nv_max=%d: "
            "MuJoCo Warp would resolve auto to sparse; the "
            "optimized kernels use the dense fused Newton solver.",
            int(self.model.nv),
            DENSE_NV_MAX,
        )
        return True

    def _upload_mjw_model(self) -> None:
        """Resolve the Jacobian, then upload the compiled model.

        The Jacobian decision must precede `put_model`: the dense cap and the
        `is_sparse`-dependent data options are read from the uploaded model.
        """

        self._resolve_dense_jacobian_for_upload()
        self.mjw_model = put_model(self.model)
        self.mjw_model.opt.warn_overflow = bool(
            getattr(self.cfg.sim, "warn_overflow", False)
        )

    def _prepare_substep_rewards(self) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} enabled substep rewards but does not "
            "implement '_prepare_substep_rewards'."
        )

    def _accumulate_substep_reward(self, substep_idx: int) -> None:
        del substep_idx
        raise NotImplementedError(
            f"{self.__class__.__name__} enabled substep rewards but does not "
            "implement '_accumulate_substep_reward'."
        )

    @staticmethod
    def _resolve_physics_dt(sim_cfg: Any) -> float:
        raw_hz = getattr(sim_cfg, "hz", None)
        try:
            hz = float(raw_hz)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid `env.sim.hz`: expected positive number, got {raw_hz!r}."
            ) from exc
        if not math.isfinite(hz) or hz <= 0.0:
            raise ValueError(
                f"Invalid `env.sim.hz`: expected positive number, got {raw_hz!r}."
            )
        return 1.0 / hz

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """How many copies are simulated in parallel."""
        return self.cfg.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """Seconds advanced by one physics substep."""
        return self._physics_dt

    @property
    def step_dt(self) -> float:
        """Seconds advanced by one policy step, `physics_dt * decimation`."""
        return self.physics_dt * self.cfg.decimation

    @property
    def max_episode_length_s(self) -> float:
        """Configured time-out horizon in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self):
        """Time-out horizon in policy steps, rounded up."""
        return math.ceil(
            self.max_episode_length_s / (self.physics_dt * self.cfg.decimation)
        )

    @property
    def _render_capture_env_idx(self) -> int:
        runtime = self._render_runtime
        if runtime is None:
            return int(self._render_capture_env_idx_value)
        return int(runtime.capture_env_idx)

    @_render_capture_env_idx.setter
    def _render_capture_env_idx(self, value: int) -> None:
        runtime = self._render_runtime
        if runtime is None:
            self._render_capture_env_idx_value = int(value)
            return
        runtime.capture_env_idx = int(value)

    @property
    def _render_phase_before_last_step(self) -> int:
        runtime = self._render_runtime
        if runtime is None:
            return int(self._render_phase_before_last_step_value)
        return int(runtime.phase_before_last_step)

    @_render_phase_before_last_step.setter
    def _render_phase_before_last_step(self, value: int) -> None:
        runtime = self._render_runtime
        if runtime is None:
            self._render_phase_before_last_step_value = int(value)
            return
        runtime.phase_before_last_step = int(value)

    def get_video_overlay_plots(self, env_idx: int = 0) -> dict[str, Any]:
        """Return optional per-step plot data for video overlays.

        Subclasses can override this to expose debug scalars that should be rendered
        into recorded videos. The expected schema is:

        {
            "title": "Optional overlay title",
            "plots": [
                {
                    "title": "Plot title",
                    "series": {
                        "name_a": scalar_like,
                        "name_b": scalar_like,
                    },
                    "colors": {
                        "name_a": "#e74c3c" or (231, 76, 60),
                    },
                },
            ],
        }

        `scalar_like` may be a Python number, a 0-d tensor/ndarray, or any array-like
        object where the first element is the value for the selected env.
        """
        return {}

    """
    Operations.
    """

    def _make_startup_reset_profiler(
        self,
        *,
        label: str,
    ) -> StartupProfiler | None:
        if self._startup_reset_profile_logged:
            return None
        self._startup_reset_profile_logged = True
        return StartupProfiler(
            range_factory=lambda name: startup_nvtx_range(f"{label}:{name}")
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:
        """Reset every copy and return the first observation.

        Delegates to `_reset_idx` over all indices. Work done once at construction
        -- model compilation, buffer allocation, render setup -- is not redone.

        Args:
            seed: Reseeds the generators when given; `None` leaves them alone.
            options: Accepted for the Gymnasium signature and ignored.

        Returns:
            The observation groups and an extras mapping.
        """
        self._seal_derived_spaces()
        default_outputs = self._bind_default_step_outputs()
        profiler = self._make_startup_reset_profiler(label="env_reset")
        previous_profiler = self._startup_reset_profiler
        if profiler is not None:
            self._startup_reset_profiler = profiler
        try:
            # set the seed
            with _startup_profile_phase(profiler, "seed"):
                if seed is not None:
                    self.seed(seed)

            # reset state of scene
            with _startup_profile_phase(profiler, "alloc_reset_mask"):
                reset_mask = torch.ones(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            with _startup_profile_phase(profiler, "_reset_idx"):
                self._reset_idx(reset_mask)
            with _startup_profile_phase(profiler, "mjw_state_prefix"):
                self._finalize_reset_state(reset_mask)

            # return observations
            with _startup_profile_phase(profiler, "get_observations"):
                returned_observations = self._get_observations()
                if default_outputs is None:
                    observations = returned_observations
                    self.obs_buf = observations
                    self._default_step_outputs = StepOutputs(
                        observations=observations,
                        rewards=(
                            self.reward_buf
                            if self._default_reward_buf is None
                            else self._default_reward_buf
                        ),
                        terminated=(
                            self.reset_terminated
                            if self._default_terminated_buf is None
                            else self._default_terminated_buf
                        ),
                        truncated=(
                            self.reset_time_outs
                            if self._default_truncated_buf is None
                            else self._default_truncated_buf
                        ),
                    )
                else:
                    observations = self._accept_observation_hook(
                        bound=default_outputs.observations,
                        returned=returned_observations,
                        validate=False,
                    )
                    self.obs_buf = observations
                self._apply_observation_noise(
                    observations, slot=OBSERVATION_NOISE_SLOT_RESET
                )
            return observations, self.extras
        finally:
            if profiler is not None:
                self._startup_reset_profiler = previous_profiler
                profiler.log_summary(
                    logger=logger,
                    prefix="ENV_RESET_STARTUP_PROFILE",
                )

    @_nvtx_method("reset_into")
    def reset_into(
        self,
        out_obs: torch.Tensor,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[VecEnvObs, dict]:
        """Reset all envs and write policy observations into a caller buffer."""

        self._seal_derived_spaces()
        default_outputs = self._bind_default_step_outputs()
        if reset_mask is None:
            cached_mask = self._reset_all_mask
            if (
                not isinstance(cached_mask, torch.Tensor)
                or cached_mask.shape != (self.num_envs,)
                or cached_mask.device != self.device
            ):
                cached_mask = torch.ones(
                    self.num_envs,
                    dtype=torch.bool,
                    device=self.device,
                )
                self._reset_all_mask = cached_mask
            reset_mask = cached_mask

        profiler = self._make_startup_reset_profiler(label="env_reset_into")
        previous_profiler = self._startup_reset_profiler
        if profiler is not None:
            self._startup_reset_profiler = profiler
        try:
            with _startup_profile_phase(profiler, "_reset_idx"):
                self._reset_idx(reset_mask)
            with _startup_profile_phase(profiler, "mjw_state_prefix"):
                self._finalize_reset_state(reset_mask)
            with _startup_profile_phase(profiler, "get_observations"):
                returned_observations = self._get_observations()
                if default_outputs is None:
                    framework_observations = returned_observations
                    self.obs_buf = framework_observations
                    self._default_step_outputs = StepOutputs(
                        observations=framework_observations,
                        rewards=(
                            self.reward_buf
                            if self._default_reward_buf is None
                            else self._default_reward_buf
                        ),
                        terminated=(
                            self.reset_terminated
                            if self._default_terminated_buf is None
                            else self._default_terminated_buf
                        ),
                        truncated=(
                            self.reset_time_outs
                            if self._default_truncated_buf is None
                            else self._default_truncated_buf
                        ),
                    )
                else:
                    framework_observations = self._accept_observation_hook(
                        bound=default_outputs.observations,
                        returned=returned_observations,
                        validate=False,
                    )
                    self.obs_buf = framework_observations
                self._apply_observation_noise(
                    framework_observations, slot=OBSERVATION_NOISE_SLOT_RESET
                )
                out_obs.copy_(framework_observations["policy"])
                caller_observations = dict(framework_observations)
                caller_observations["policy"] = out_obs
            return caller_observations, self.extras
        finally:
            if profiler is not None:
                self._startup_reset_profiler = previous_profiler
                profiler.log_summary(
                    logger=logger,
                    prefix="ENV_RESET_STARTUP_PROFILE",
                )

    @_nvtx_method("step")
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Advance every copy by one policy step.

        One policy step covers `decimation` physics substeps of `1 / sim.hz`
        seconds each, reapplying the same action to each substep. In order: hand
        the action to `_pre_physics_step`, run the substeps calling `_apply_action`
        in each, compute rewards and dones, reset whatever finished, then compute
        observations on the post-reset state.

        From the substep loop onward the body is capturable as a CUDA graph, so
        nothing in that range may allocate or branch on device values once
        captured. `_pre_physics_step` runs ahead of capture.

        Args:
            action: Shape `(num_envs, action_dim)`.

        Returns:
            Observations, rewards, terminated flags, time-out flags and extras.
        """

        self._seal_derived_spaces()
        default_outputs = self._default_step_outputs
        if default_outputs is None:
            default_outputs = StepOutputs(
                observations=self.obs_buf,
                rewards=self.reward_buf,
                terminated=self.reset_terminated,
                truncated=self.reset_time_outs,
            )
            self._default_step_outputs = default_outputs
        self._default_step_io = StepIO(
            inputs=StepInputs(actions=action),
            outputs=default_outputs,
        )
        outputs, extras = self.step_with_io(self._default_step_io)
        return (
            outputs.observations,
            outputs.rewards,
            outputs.terminated,
            outputs.truncated,
            extras,
        )

    @_nvtx_method("step_with_io")
    def step_with_io(
        self,
        step_io: StepIO,
        *,
        use_cuda_graph: bool | None = None,
        reset_done: bool = True,
        validate_buffers: bool = False,
    ) -> tuple[StepOutputs, dict]:
        """Bind structured buffers and execute the canonical step pipeline.

        The returned outputs borrow the bound storage. Callers that need a snapshot
        must clone it explicitly. ``use_cuda_graph`` selects an execution strategy;
        it is not part of the environment-author hook contract.
        """
        self._seal_derived_spaces()
        outputs = step_io.outputs
        self._bind_step_outputs(outputs)
        self._step_pipeline_active = True
        try:
            self._run_step_lifecycle_before(step_io)
            result = self._run_bound_step(
                step_io,
                use_cuda_graph=use_cuda_graph,
                reset_done=reset_done,
                validate_buffers=validate_buffers,
            )
            self._run_step_lifecycle_after(step_io)
            return result
        finally:
            self._step_pipeline_active = False

    def _register_step_lifecycle_component(
        self,
        component: StepLifecycleComponent,
    ) -> None:
        graph_cache = self._step_graph_cache
        if (
            self._step_lifecycle_registration_frozen
            or self._step_graph is not None
            or bool(graph_cache)
        ):
            raise RuntimeError(
                "Step lifecycle components must be registered before graph capture."
            )
        self._step_lifecycle_components.append(component)

    @property
    def observation_noise_settings(self) -> ObservationNoiseSettings | None:
        """The parsed `env.observation_noise` block, or `None` when absent."""

        return self._observation_noise_settings

    @property
    def observation_noise_enabled(self) -> bool:
        """Whether policy observations currently receive training noise."""

        return self._observation_noise_runtime is not None

    def enable_observation_noise(
        self,
        *,
        feature_variance: torch.Tensor | None = None,
        epsilon: float = 0.0,
        statistics_count: torch.Tensor | None = None,
        min_count: float = 0.0,
    ) -> bool:
        """Start adding the configured noise to this instance's policy observation.

        Only the training runtime calls this, on its rollout environment, so
        every other instance built from the same config stays clean. The noise
        covers every policy feature. ``feature_variance`` is held
        by reference: ``units: normalized`` passes the agent normaliser's
        ``running_variance`` and ``epsilon`` so the draw of feature ``j`` is
        ``std * (sqrt(feature_variance[j]) + epsilon)``; ``units: raw`` passes
        neither. ``statistics_count`` (also by reference) gates the noise off
        while ``statistics_count[0] <= min_count``, i.e. until the normaliser
        has committed statistics. Captured step graphs are dropped so the next
        capture records the launch.

        Returns:
            Whether noise is now enabled; `False` when the config declares none
            or its ``std`` is zero or ``enabled`` is false.
        """

        settings = self._observation_noise_settings
        if settings is None or not settings.active:
            return False
        if settings.units == "normalized" and feature_variance is None:
            raise ValueError(
                "env.observation_noise.units=normalized needs the normaliser's "
                "feature_variance; pass it or use units: raw"
            )
        if settings.units == "raw" and feature_variance is not None:
            raise ValueError(
                "env.observation_noise.units=raw takes no feature_variance"
            )
        observation_space = getattr(self, "single_observation_space", None)
        policy_space = (
            observation_space.get("policy")
            if isinstance(observation_space, gym.spaces.Dict)
            else None
        )
        policy_shape = getattr(policy_space, "shape", None)
        if policy_shape is None or len(policy_shape) != 1:
            raise ValueError(
                "observation noise needs a one-dimensional policy observation "
                f"space, got {policy_space}"
            )
        width = int(policy_shape[0])
        self._observation_noise_runtime = ObservationNoiseRuntime(
            settings=settings,
            num_envs=self.num_envs,
            width=width,
            step_counter=self.common_step_counter,
            seed=get_value(self.cfg, "seed", None),
            feature_variance=feature_variance,
            epsilon=epsilon,
            statistics_count=statistics_count,
            min_count=min_count,
        )
        self.invalidate_step_graph_cache()
        logger.info(
            "Observation noise enabled: std=%g units=%s features=[0, %d) of %d "
            "policy features (training rollouts only)",
            settings.std,
            settings.units,
            width,
            int(policy_shape[0]),
        )
        return True

    def disable_observation_noise(self) -> None:
        """Stop adding observation noise and drop graphs that recorded it."""

        if self._observation_noise_runtime is None:
            return
        self._observation_noise_runtime = None
        self.invalidate_step_graph_cache()

    def _apply_observation_noise(
        self, observations: Mapping[str, torch.Tensor], *, slot: int
    ) -> None:
        runtime = self._observation_noise_runtime
        if runtime is None:
            return
        with self._nvtx_range("observation_noise"):
            runtime.apply(observations["policy"], slot=slot)

    def invalidate_step_graph_cache(self) -> None:
        """Drop captured environment-step graphs after a runtime regime change."""

        self._step_graph_cache.clear()
        self._step_graph = None

    def _run_step_lifecycle_before(self, step_io: StepIO) -> None:
        for component in self._step_lifecycle_components:
            component.before_step(step_io)

    def _run_step_lifecycle_after(self, step_io: StepIO) -> None:
        for component in reversed(self._step_lifecycle_components):
            component.after_step(step_io)

    def prepare_step_capture_window(self) -> None:
        """Prepare host-side lifecycle state before an external graph capture."""
        self._seal_derived_spaces()
        self._step_lifecycle_registration_frozen = True
        scratch = self._optimized_fwd_position_scratch
        if scratch is not None and hasattr(scratch, "implicit_qh_registration_frozen"):
            scratch.implicit_qh_registration_frozen = True
        for component in self._step_lifecycle_components:
            component.prepare_capture_window()

    def finalize_step_capture_window(self) -> None:
        """Finalize host-side lifecycle state after an external graph replay."""
        for component in reversed(self._step_lifecycle_components):
            component.finalize_capture_window()

    def _bind_step_outputs(self, outputs: StepOutputs) -> None:
        self.obs_buf = outputs.observations
        self.reward_buf = outputs.rewards
        self.reset_terminated = outputs.terminated
        self.reset_time_outs = outputs.truncated

    def _bind_default_step_outputs(self) -> StepOutputs | None:
        outputs = self._default_step_outputs
        if outputs is not None:
            self._bind_step_outputs(outputs)
        return outputs

    def _coerce_actions(self, actions: Any) -> torch.Tensor:
        """Convert public action input and enforce the vectorized batch contract."""

        # The Gym action space defines the accepted batch before a task/control
        # pipeline writes its execution storage during the physics step.
        action_space = getattr(self, "action_space", None)
        action_template = getattr(self, "actions", None)
        expected_shape = (
            tuple(action_space.shape)
            if isinstance(action_space, gym.spaces.Box)
            else (
                tuple(action_template.shape)
                if isinstance(action_template, torch.Tensor)
                else None
            )
        )
        target_dtype = torch.float32
        if isinstance(action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
            target_dtype = torch.int64
        elif action_space is None and isinstance(action_template, torch.Tensor):
            target_dtype = action_template.dtype
        try:
            action = torch.as_tensor(
                actions,
                device=self.device,
                dtype=target_dtype,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"actions must be array-like numeric data with shape {expected_shape}"
            ) from exc
        if expected_shape is not None and tuple(action.shape) != expected_shape:
            raise ValueError(
                "actions must have exact batched shape "
                f"{expected_shape}, got {tuple(action.shape)}"
            )
        return action

    @staticmethod
    def _space_fingerprint(space: gym.spaces.Space) -> tuple[Any, ...]:
        """Return a stable, process-local fingerprint for a Gym space."""

        if isinstance(space, gym.spaces.Box):
            return (
                "Box",
                tuple(space.shape),
                str(space.dtype),
                tuple(np.asarray(space.low).reshape(-1).tolist()),
                tuple(np.asarray(space.high).reshape(-1).tolist()),
            )
        if isinstance(space, gym.spaces.Discrete):
            return ("Discrete", int(space.n), int(space.start))
        if isinstance(space, gym.spaces.MultiDiscrete):
            return (
                "MultiDiscrete",
                tuple(int(value) for value in np.asarray(space.nvec).reshape(-1)),
            )
        if isinstance(space, gym.spaces.Dict):
            return (
                "Dict",
                tuple(
                    (key, DirectRLEnv._space_fingerprint(value))
                    for key, value in space.items()
                ),
            )
        if isinstance(space, gym.spaces.Tuple):
            return (
                "Tuple",
                tuple(DirectRLEnv._space_fingerprint(value) for value in space),
            )
        return (type(space).__name__, repr(space))

    @staticmethod
    def _tensor_fingerprint(tensor: torch.Tensor) -> tuple[Any, ...]:
        return (
            id(tensor),
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )

    @staticmethod
    def _canonical_device(device: torch.device | str) -> torch.device:
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        return resolved

    def _seal_derived_spaces(self) -> None:
        if getattr(self, "_derived_spaces_pending", False):
            missing = tuple(
                name
                for name in (
                    "single_observation_space",
                    "observation_space",
                    "obs_buf",
                )
                if not hasattr(self, name) or getattr(self, name) is None
            )
            if missing or self._derived_spaces_fingerprint is None:
                details = ", ".join(missing) or "finalizer fingerprint"
                raise RuntimeError(
                    "observation spaces must be finalized before runtime lifecycle; "
                    f"missing {details}"
                )
        self._derived_spaces_sealed = True

    def _set_runtime_cfg_value(self, name: str, value: Any) -> None:
        """Set a derived cfg field on either DictConfig or plain configs."""

        cfg = self.cfg
        if OmegaConf.is_config(cfg):
            with open_dict(cfg):
                cfg[name] = value
        elif isinstance(cfg, MutableMapping):
            cfg[name] = value  # type: ignore[index]
        else:
            setattr(cfg, name, value)

    def _get_runtime_cfg_value(self, name: str) -> Any:
        try:
            if OmegaConf.is_config(self.cfg):
                if OmegaConf.is_missing(self.cfg, name):
                    return _CFG_MANDATORY_MISSING
                if name not in self.cfg:
                    return _CFG_ABSENT
                return self.cfg[name]
            if hasattr(self.cfg, "get"):
                return self.cfg.get(name, _CFG_ABSENT)
            return getattr(self.cfg, name, _CFG_ABSENT)
        except AttributeError:
            return _CFG_ABSENT
        except MissingMandatoryValue:
            return _CFG_MANDATORY_MISSING

    def finalize_derived_spaces(
        self,
        *,
        observation_spec: Any,
        action_spec: Any,
        observations: Mapping[str, torch.Tensor],
        state_spec: Any | None = None,
    ) -> None:
        """Commit one caller-owned derived Gym/observation space contract.

        Derived environment layers pass the exact mapping that their writers use.
        The method validates every input and prepares all Gym objects before
        mutating configuration or framework pointers. It deliberately has no
        action-buffer argument: ``self.actions`` is execution storage and can be
        owned by a controller or a command runtime independently of the public
        action space.
        """

        if not isinstance(observations, Mapping):
            raise TypeError("observations must be a mapping of named tensors")
        try:
            single_policy_space = spec_to_gym_space(observation_spec)
            single_action_space = spec_to_gym_space(action_spec)
            single_state_space = (
                None if state_spec is None else spec_to_gym_space(state_spec)
            )
            single_observation_space = gym.spaces.Dict(
                {
                    "policy": single_policy_space,
                    **(
                        {}
                        if single_state_space is None
                        else {"critic": single_state_space}
                    ),
                }
            )
            batched_observation_space = gym.vector.utils.batch_space(
                single_observation_space, self.num_envs
            )
            batched_action_space = gym.vector.utils.batch_space(
                single_action_space, self.num_envs
            )
            batched_state_space = (
                None
                if single_state_space is None
                else gym.vector.utils.batch_space(single_state_space, self.num_envs)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid derived Gym space specification") from exc

        expected_keys = (
            {"policy"} if single_state_space is None else {"policy", "critic"}
        )
        if set(observations) != expected_keys:
            raise ValueError(
                "derived observations must contain exactly "
                f"{tuple(sorted(expected_keys))}, got {tuple(observations)}"
            )
        for name, tensor in observations.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"observations[{name!r}] must be a torch.Tensor")
            single_space = (
                single_policy_space if name == "policy" else single_state_space
            )
            if not isinstance(single_space, gym.spaces.Box):
                raise TypeError(
                    "derived observation tensors currently require flat Box spaces"
                )
            expected_shape = (self.num_envs, *tuple(single_space.shape))
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"observations[{name!r}] must have shape {expected_shape}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.dtype != torch.float32:
                raise TypeError(f"observations[{name!r}] must use torch.float32")
            expected_device = self._canonical_device(self.device)
            if tensor.device != expected_device:
                raise ValueError(
                    f"observations[{name!r}] must be on {expected_device}, "
                    f"got {tensor.device}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"observations[{name!r}] must be contiguous")

        fingerprint = (
            self._space_fingerprint(single_policy_space),
            self._space_fingerprint(single_action_space),
            None
            if single_state_space is None
            else self._space_fingerprint(single_state_space),
            id(observations),
            tuple(
                (name, self._tensor_fingerprint(tensor))
                for name, tensor in observations.items()
            ),
        )
        if getattr(self, "_derived_spaces_fingerprint", None) == fingerprint:
            return
        if getattr(self, "_derived_spaces_sealed", False):
            raise RuntimeError(
                "derived observation spaces cannot change after reset, step, "
                "terminal-buffer preparation, or capture preparation"
            )

        missing = object()
        old_cfg = {
            name: self._get_runtime_cfg_value(name)
            for name in ("observation_space", "action_space", "state_space")
        }
        old_spaces = {
            name: getattr(self, name, missing)
            for name in (
                "single_observation_space",
                "single_action_space",
                "observation_space",
                "action_space",
                "state_space",
                "obs_buf",
            )
        }
        old_default_outputs = getattr(self, "_default_step_outputs", None)
        old_default_io = getattr(self, "_default_step_io", missing)
        old_fingerprint = getattr(self, "_derived_spaces_fingerprint", None)
        old_pending = getattr(self, "_derived_spaces_pending", False)
        try:
            self._set_runtime_cfg_value("observation_space", observation_spec)
            self._set_runtime_cfg_value("action_space", action_spec)
            self._set_runtime_cfg_value("state_space", state_spec)
            self.single_observation_space = single_observation_space
            self.single_action_space = single_action_space
            self.observation_space = batched_observation_space
            self.action_space = batched_action_space
            self.state_space = batched_state_space
            self.obs_buf = observations
            if old_default_outputs is not None:
                updated_outputs = StepOutputs(
                    observations=observations,
                    rewards=old_default_outputs.rewards,
                    terminated=old_default_outputs.terminated,
                    truncated=old_default_outputs.truncated,
                )
                self._default_step_outputs = updated_outputs
                if old_default_io is not missing:
                    self._default_step_io = StepIO(
                        inputs=old_default_io.inputs,
                        outputs=updated_outputs,
                    )
            self._derived_spaces_fingerprint = fingerprint
            self._derived_spaces_pending = False
        except Exception:
            for name, value in old_cfg.items():
                try:
                    if value is _CFG_ABSENT:
                        if OmegaConf.is_config(self.cfg):
                            with open_dict(self.cfg):
                                self.cfg.pop(name, None)
                        elif isinstance(self.cfg, MutableMapping):
                            self.cfg.pop(name, None)  # type: ignore[attr-defined]
                        elif isinstance(self.cfg, Mapping):
                            pass
                        else:
                            if hasattr(self.cfg, name):
                                delattr(self.cfg, name)
                    elif value is _CFG_MANDATORY_MISSING:
                        if OmegaConf.is_config(self.cfg):
                            with open_dict(self.cfg):
                                self.cfg[name] = "???"
                        else:
                            self._set_runtime_cfg_value(name, value)
                    else:
                        self._set_runtime_cfg_value(name, value)
                except (ConfigAttributeError, FrozenInstanceError, ReadonlyConfigError):
                    pass
            for name, value in old_spaces.items():
                if value is missing:
                    self.__dict__.pop(name, None)
                else:
                    setattr(self, name, value)
            self._default_step_outputs = old_default_outputs
            if old_default_io is missing:
                self.__dict__.pop("_default_step_io", None)
            else:
                self._default_step_io = old_default_io
            self._derived_spaces_fingerprint = old_fingerprint
            self._derived_spaces_pending = old_pending
            raise

    @contextmanager
    def bind_observation_outputs(self, observations: Mapping[str, torch.Tensor]):
        """Temporarily bind an observation mapping for a no-argument hook call."""

        missing = object()
        previous_observations = getattr(self, "obs_buf", missing)
        self.obs_buf = observations
        try:
            yield observations
        finally:
            if previous_observations is missing:
                del self.obs_buf
            else:
                self.obs_buf = previous_observations

    def _run_bound_step(
        self,
        step_io: StepIO,
        *,
        use_cuda_graph: bool | None,
        reset_done: bool,
        validate_buffers: bool,
    ) -> tuple[StepOutputs, dict]:
        outputs = step_io.outputs
        self._validate_step_buffers = bool(validate_buffers)
        buffer_snapshot = self._snapshot_step_io(step_io) if validate_buffers else None
        if validate_buffers:
            self._validate_bound_step_io(
                step_io,
                hook="bind_step_io",
                snapshot=buffer_snapshot,
            )

        action = self._coerce_actions(step_io.inputs.actions)
        self._substep_integrator.before_pre_physics_step()
        self._pre_physics_step(action)
        self._substep_integrator.pre_physics_step(action)
        self._render_runtime.prepare_substep_render_capture()

        if use_cuda_graph is None:
            use_cuda_graph = self.cfg.use_cuda_graph
        if validate_buffers:
            # Buffer validation inspects Python-side hook results on every call.
            # A cached graph replay would bypass those hooks and could falsely pass.
            use_cuda_graph = False
        if use_cuda_graph:
            graph_torch_stream, graph_wp_stream = self._get_env_graph_streams()
            graph_key = self._step_graph_cache_key(
                step_io,
                reset_done=reset_done,
                unroll_solver_iterations=getattr(
                    self._optimized_fwd_position_scratch,
                    "unroll_solver_iterations",
                    False,
                ),
            )
            graph_cache = self._step_graph_cache
            graph = graph_cache.get(graph_key)
            if graph is None:
                self._step_lifecycle_registration_frozen = True
                with _stream_aligned_cuda_graph_capture(
                    self.device,
                    graph_torch_stream,
                    graph_wp_stream,
                ) as capture:
                    self._step_body(reset_done=reset_done)
                graph = capture.graph
                graph_cache[graph_key] = graph
                self._step_graph = graph
            # Capture records the step but does not complete it. Launch on both
            # cache miss and cache hit so every successful call produces outputs.
            _launch_stream_aligned_cuda_graph(
                graph,
                self.device,
                graph_torch_stream,
                graph_wp_stream,
            )
        else:
            self._step_body(reset_done=reset_done)

        if validate_buffers:
            self._validate_bound_step_io(
                step_io,
                hook="step_pipeline",
                snapshot=buffer_snapshot,
            )
        self._render_runtime.finalize_substep_render_capture()
        return outputs, self.extras

    @staticmethod
    def _tensor_layout(tensor: torch.Tensor) -> dict[str, Any]:
        storage = tensor.untyped_storage()
        return {
            "object_identity": id(tensor),
            "data_ptr": tensor.data_ptr(),
            "storage_ptr": storage.data_ptr(),
            "storage_offset": tensor.storage_offset(),
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
            "dtype": tensor.dtype,
            "device": tensor.device,
        }

    @classmethod
    def _step_graph_cache_key(
        cls,
        step_io: StepIO,
        *,
        reset_done: bool,
        unroll_solver_iterations: bool = False,
    ) -> tuple[Any, ...]:
        outputs = step_io.outputs

        def tensor_key(tensor: torch.Tensor) -> tuple[Any, ...]:
            layout = cls._tensor_layout(tensor)
            return tuple(
                layout[name]
                for name in (
                    "data_ptr",
                    "storage_ptr",
                    "storage_offset",
                    "shape",
                    "stride",
                    "dtype",
                    "device",
                )
            )

        observation_key = tuple(
            (name, tensor_key(tensor))
            for name, tensor in sorted(outputs.observations.items())
        )
        key = (
            observation_key,
            tensor_key(outputs.rewards),
            tensor_key(outputs.terminated),
            tensor_key(outputs.truncated),
            bool(reset_done),
        )

        return key + (("unrolled_solver",) if unroll_solver_iterations else ())

    @classmethod
    def _snapshot_step_io(cls, step_io: StepIO) -> dict[str, Any]:
        outputs = step_io.outputs
        return {
            "observations_mapping": outputs.observations,
            "observations": {
                name: (tensor, cls._tensor_layout(tensor))
                for name, tensor in outputs.observations.items()
            },
            "rewards": (outputs.rewards, cls._tensor_layout(outputs.rewards)),
            "terminated": (
                outputs.terminated,
                cls._tensor_layout(outputs.terminated),
            ),
            "truncated": (
                outputs.truncated,
                cls._tensor_layout(outputs.truncated),
            ),
        }

    @classmethod
    def _validate_tensor_slot(
        cls,
        *,
        hook: str,
        slot: str,
        expected: torch.Tensor,
        actual: Any,
        expected_layout: Mapping[str, Any] | None = None,
    ) -> None:
        if expected_layout is None:
            expected_layout = cls._tensor_layout(expected)
        if actual is expected and isinstance(actual, torch.Tensor):
            actual_layout = cls._tensor_layout(actual)
            if actual_layout == expected_layout:
                return
        else:
            actual_layout = (
                cls._tensor_layout(actual)
                if isinstance(actual, torch.Tensor)
                else {"type": type(actual).__name__}
            )
        raise BufferContractError(
            f"Buffer contract violation in {hook} for {slot}: "
            f"expected={expected_layout}, actual={actual_layout}"
        )

    @classmethod
    def _validate_observation_slots(
        cls,
        *,
        hook: str,
        expected: Mapping[str, torch.Tensor],
        actual: Any,
    ) -> None:
        if actual is not expected:
            raise BufferContractError(
                f"Buffer contract violation in {hook} for outputs.observations: "
                f"expected mapping object_identity={id(expected)}, "
                f"actual object_identity={id(actual)}"
            )
        if set(actual) != set(expected):
            raise BufferContractError(
                f"Buffer contract violation in {hook} for outputs.observations: "
                f"expected keys={tuple(expected)}, actual keys={tuple(actual)}"
            )
        for name, expected_tensor in expected.items():
            cls._validate_tensor_slot(
                hook=hook,
                slot=f"outputs.observations[{name!r}]",
                expected=expected_tensor,
                actual=actual[name],
            )

    def _validate_bound_step_io(
        self,
        step_io: StepIO,
        *,
        hook: str,
        snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        outputs = step_io.outputs
        if snapshot is not None:
            expected_mapping = snapshot["observations_mapping"]
            if self.obs_buf is not expected_mapping:
                raise BufferContractError(
                    f"Buffer contract violation in {hook} for outputs.observations: "
                    f"expected mapping object_identity={id(expected_mapping)}, "
                    f"actual object_identity={id(self.obs_buf)}"
                )
            observation_snapshot = snapshot["observations"]
            if set(self.obs_buf) != set(observation_snapshot):
                raise BufferContractError(
                    f"Buffer contract violation in {hook} for outputs.observations: "
                    f"expected keys={tuple(observation_snapshot)}, "
                    f"actual keys={tuple(self.obs_buf)}"
                )
            for name, (
                expected_tensor,
                expected_layout,
            ) in observation_snapshot.items():
                self._validate_tensor_slot(
                    hook=hook,
                    slot=f"outputs.observations[{name!r}]",
                    expected=expected_tensor,
                    actual=self.obs_buf[name],
                    expected_layout=expected_layout,
                )
            for name, attribute in (
                ("rewards", "reward_buf"),
                ("terminated", "reset_terminated"),
                ("truncated", "reset_time_outs"),
            ):
                expected_tensor, expected_layout = snapshot[name]
                self._validate_tensor_slot(
                    hook=hook,
                    slot=f"outputs.{name}",
                    expected=expected_tensor,
                    actual=getattr(self, attribute),
                    expected_layout=expected_layout,
                )
            return
        self._validate_observation_slots(
            hook=hook,
            expected=outputs.observations,
            actual=self.obs_buf,
        )
        self._validate_tensor_slot(
            hook=hook,
            slot="outputs.rewards",
            expected=outputs.rewards,
            actual=self.reward_buf,
        )
        self._validate_tensor_slot(
            hook=hook,
            slot="outputs.terminated",
            expected=outputs.terminated,
            actual=self.reset_terminated,
        )
        self._validate_tensor_slot(
            hook=hook,
            slot="outputs.truncated",
            expected=outputs.truncated,
            actual=self.reset_time_outs,
        )

    @classmethod
    def _accept_hook_tensor(
        cls,
        *,
        hook: str,
        slot: str,
        bound: torch.Tensor,
        returned: torch.Tensor,
        validate: bool,
    ) -> torch.Tensor:
        if validate:
            cls._validate_tensor_slot(
                hook=hook,
                slot=slot,
                expected=bound,
                actual=returned,
            )
        elif returned is not bound:
            bound.copy_(returned)
        return bound

    @classmethod
    def _accept_observation_hook(
        cls,
        *,
        bound: Mapping[str, torch.Tensor],
        returned: Mapping[str, torch.Tensor],
        validate: bool,
    ) -> Mapping[str, torch.Tensor]:
        if validate:
            cls._validate_observation_slots(
                hook="_get_observations",
                expected=bound,
                actual=returned,
            )
            return bound
        for name, bound_tensor in bound.items():
            if name not in returned:
                raise BufferContractError(
                    "Buffer contract violation in _get_observations for "
                    f"outputs.observations[{name!r}]: returned mapping omitted slot"
                )
            if returned[name] is not bound_tensor:
                bound_tensor.copy_(returned[name])
        return bound

    def _step_body(self, *, reset_done: bool = True):
        """Run all device operations for one environment step."""
        # Physics
        if self._substep_rewards_enabled:
            self._prepare_substep_rewards()
        last = self.cfg.decimation - 1
        defer = reset_done and self._defers_boundary_constraints()
        for substep_idx in range(self.cfg.decimation):
            with self._nvtx_range(f"physics_substep_{substep_idx}"):
                self._run_physics_substep(
                    substep_idx,
                    endpoint_constraints=not (defer and substep_idx == last),
                )
                if self._substep_rewards_enabled:
                    with self._nvtx_range("substep_reward"):
                        self._accumulate_substep_reward(substep_idx)
                self._render_runtime.capture_substep_state(substep_idx)

        # Update env counters (warp kernels for CUDA graph compatibility)
        with self._nvtx_range("update_env_counters"):
            wp.launch(
                _increment_episode_length_kernel,
                dim=(self.num_envs,),
                inputs=[self.episode_length_buf],
            )
            wp.launch(
                _increment_step_counter_kernel,
                dim=(1,),
                inputs=[self.common_step_counter],
            )

        # The substep loop above ends on a state prefix, not on a suffix, so the
        # position and velocity state read from here on is current. The
        # acceleration stage is deliberately not repeated at this boundary, so
        # `qacc`, `efc_force` and `qfrc_constraint` still describe the state the
        # last solve saw, one substep back. Nothing below reads them. A consumer
        # that needs them at this state must run its own forward first, the way
        # `StablePDController.compute_torque` does; do not assume the boundary
        # refreshed them. With `defer`, contacts and constraint rows are also
        # one substep back until the reset prefix below rebuilds them. See the
        # state-prefix schedule in docs/reference/simulation.md.
        nonfinite_state_mask = self._compute_dones_and_rewards()

        # Preserve observations from the state that produced this transition
        # before auto-reset overwrites the state, allowing timeout bootstrapping.
        # The bound observation slots are then reused for
        # the next policy input below.
        if self._terminal_obs_buf is not None:
            with self._nvtx_range("get_terminal_observations"):
                bound_outputs = StepOutputs(
                    observations=self.obs_buf,
                    rewards=self.reward_buf,
                    terminated=self.reset_terminated,
                    truncated=self.reset_time_outs,
                )
                terminal_outputs = StepOutputs(
                    observations=self._terminal_obs_buf,
                    rewards=self.reward_buf,
                    terminated=self.reset_terminated,
                    truncated=self.reset_time_outs,
                )
                self._bind_step_outputs(terminal_outputs)
                try:
                    returned_observations = self._get_observations()
                    self._accept_observation_hook(
                        bound=self._terminal_obs_buf,
                        returned=returned_observations,
                        validate=self._validate_step_buffers,
                    )
                    self._apply_observation_noise(
                        self._terminal_obs_buf, slot=OBSERVATION_NOISE_SLOT_TERMINAL
                    )
                finally:
                    self._bind_step_outputs(bound_outputs)

        # Reset
        if reset_done:
            with self._nvtx_range("reset_idx"):
                self._accumulate_done_reason_stats(nonfinite_state_mask)
                self._reset_idx(self.reset_buf)
                self._finalize_reset_state(self.reset_buf)

        # Observations
        with self._nvtx_range("get_observations"):
            bound_observations = self.obs_buf
            returned_observations = self._get_observations()
            self.obs_buf = self._accept_observation_hook(
                bound=bound_observations,
                returned=returned_observations,
                validate=self._validate_step_buffers,
            )
            self._apply_observation_noise(
                self.obs_buf, slot=OBSERVATION_NOISE_SLOT_STEP
            )

    @classmethod
    def _resolve_integrator_spec(cls, name: Any) -> IntegratorSpec:
        return resolve_integrator_spec(name)

    @property
    def substep_kind(self) -> str:
        """Substep kind force models see: implicitfast, euler or rk4."""
        return self._integrator_spec.substep

    def _endpoint_callbacks_read_geometry(self) -> bool:
        """Whether a callback after a substep reads its refreshed geometry."""
        return (
            bool(self._post_physics_substep_hooks)
            or self._control_pipeline.post_physics_substep_reads_geometry()
            or self._task_endpoint_callbacks_read_geometry()
        )

    def _endpoint_callbacks_write_state(self) -> bool:
        """Whether a callback after a substep changes state the next one reads."""
        return (
            bool(self._post_physics_substep_hooks)
            or self._control_pipeline.post_physics_substep_writes_state()
            or self._task_endpoint_callbacks_write_state()
        )

    @classmethod
    def _replaces_hook(cls, name: str, owner: type) -> bool:
        """Whether this class replaces ``owner``'s implementation of hook ``name``."""
        return getattr(cls, name) is not getattr(owner, name)

    def _task_endpoint_callbacks_read_geometry(self) -> bool:
        """Task declaration; conservatively true when the task adds the hook."""
        return self._replaces_hook("_post_state_prefix_substep", DirectRLEnv)

    def _task_endpoint_callbacks_write_state(self) -> bool:
        """Task declaration; conservatively true when the task adds either hook."""
        return self._replaces_hook(
            "_post_physics_substep", DirectRLEnv
        ) or self._replaces_hook("_post_state_prefix_substep", DirectRLEnv)

    def _run_companion_substeps(
        self, first_substep: int, count: int, *, refresh_end: bool
    ) -> None:
        """Advance ``count`` substeps for an integrator that drives this env.

        The caller keeps the start geometry current and owns the step boundary;
        ``refresh_end=False`` skips the last endpoint refresh, for an env whose
        endpoint callbacks read no geometry.
        """
        self._step_pipeline_active = True
        try:
            for offset in range(count):
                self._run_physics_substep(
                    first_substep + offset,
                    refresh_state_prefix=refresh_end or offset < count - 1,
                )
        finally:
            self._step_pipeline_active = False

    def _defers_boundary_constraints(self) -> bool:
        """Whether an auto-resetting step may leave the last endpoint's contacts
        and constraint rows to the reset prefix: the concrete class declares
        ``boundary_reads_constraints = False``, no registered or control endpoint
        callback reads geometry, and the model neither sleeps nor has sensors."""
        if self.boundary_reads_constraints:
            return False
        return (
            not self._post_physics_substep_hooks
            and not self._control_pipeline.post_physics_substep_reads_geometry()
            and state_prefix_can_defer_constraints(self.mjw_model)
        )

    def _run_physics_substep(
        self,
        substep_idx: int,
        *,
        refresh_state_prefix: bool = True,
        endpoint_constraints: bool = True,
    ) -> None:
        """Advance one physics substep through the substep integrator.

        ``refresh_state_prefix=False`` leaves the endpoint geometry stale; only
        a caller that refreshes before the next substep, for an env whose
        endpoint callbacks read no geometry, may pass it.
        ``endpoint_constraints=False`` refreshes the endpoint without collision
        and constraint construction; only a step whose reset prefix follows,
        for an env that reads neither in between, may pass it.
        """
        integrator = self._substep_integrator
        integrator.advance(substep_idx)
        with self._nvtx_range("post_physics_substep"):
            self._post_physics_substep(substep_idx)
        if refresh_state_prefix:
            with self._nvtx_range("refresh_state_prefix"):
                # Task hooks may clamp or otherwise mutate qpos/qvel.
                # Refresh before control callbacks so they consume endpoint
                # geometry derived from the final substep state.
                integrator.refresh_endpoint_state_prefix(
                    constraints=endpoint_constraints
                )
        with self._nvtx_range("post_state_prefix_substep"):
            self._run_control_post_physics_substep(substep_idx)
            integrator.finish_substep(substep_idx)
            self._post_state_prefix_substep(substep_idx)

    def prepare_terminal_observation_buffer(self) -> None:
        """Allocate persistent storage for the pre-reset observation endpoint."""
        self._seal_derived_spaces()
        observations = getattr(self, "obs_buf", None)
        if not isinstance(observations, Mapping) or not observations:
            raise RuntimeError(
                "Environment observations must be bound before allocating the "
                "terminal observation buffer."
            )
        if self._terminal_obs_buf is None:
            self._terminal_obs_buf = {
                name: torch.empty_like(observation)
                for name, observation in observations.items()
            }
            return
        if set(self._terminal_obs_buf) != set(observations):
            raise BufferContractError(
                "Terminal observation buffer keys changed after allocation."
            )
        for name, observation in observations.items():
            if self._terminal_obs_buf[name].shape != observation.shape:
                raise BufferContractError(
                    "Terminal observation buffer shape changed after allocation."
                )

    @property
    def terminal_obs_buf(self) -> Mapping[str, torch.Tensor] | None:
        """Persistent pre-reset observation endpoint, when requested by rollout."""
        return self._terminal_obs_buf

    def _compute_dones_and_rewards(self) -> torch.Tensor | None:
        if (
            torch.device(self.device).type == "cuda"
            and self._can_parallelize_dones_and_rewards()
        ):
            return self._compute_dones_and_rewards_parallel()
        return self._compute_dones_and_rewards_sequential()

    def _finalize_reset_state(self, reset_mask: torch.Tensor) -> None:
        """Refresh derived state and run task finalization after a reset.

        A task hook may consume the fresh prefix and mutate qpos/qvel (for
        example, an interval push). In that case it returns ``True`` so the
        final observations still see coherent derived state. A task that only
        needs part of the derived state for its edits (for example kinematics)
        makes them in ``_prepare_reset_state`` instead, before the one complete
        prefix.
        """
        self._prepare_reset_state(reset_mask)
        self._run_mjw_state_prefix()
        if self._post_reset_state_prefix(reset_mask):
            self._run_mjw_state_prefix()
        self._post_finalize_reset_state(reset_mask)
        for finalize in self._reset_state_finalizers:
            finalize(reset_mask)

    def _prepare_reset_state(self, reset_mask: torch.Tensor) -> None:
        """Edit the reset worlds' qpos/qvel before the complete state prefix.

        The hook refreshes whatever derived state it reads itself (for example
        ``_run_mjw_kinematics_com_state``, which runs the prefix's own kinematics
        and COM path); the caller's prefix then derives the rest once.
        """
        del reset_mask

    def _post_reset_state_prefix(self, reset_mask: torch.Tensor) -> bool:
        del reset_mask
        return False

    def _post_finalize_reset_state(self, reset_mask: torch.Tensor) -> None:
        """Run task finalization after the last reset state-prefix refresh."""

        del reset_mask

    def _post_dones_and_rewards(
        self, nonfinite_state_mask: torch.Tensor | None
    ) -> None:
        """Finalize reward-dependent termination and task accounting before reset.

        Structural dones precede rewards because a task may use them in its
        reward. Tasks that add reward-dependent failure can update dones here,
        recombine the reset mask, then account for the complete transition.
        """
        del nonfinite_state_mask

    def _compute_dones_and_rewards_sequential(self) -> torch.Tensor | None:
        nonfinite_state_mask = None

        with self._nvtx_range("get_dones"):
            bound_terminated = self.reset_terminated
            bound_truncated = self.reset_time_outs
            returned_terminated, returned_truncated = self._get_dones()
            self.reset_terminated = self._accept_hook_tensor(
                hook="_get_dones",
                slot="outputs.terminated",
                bound=bound_terminated,
                returned=returned_terminated,
                validate=self._validate_step_buffers,
            )
            self.reset_time_outs = self._accept_hook_tensor(
                hook="_get_dones",
                slot="outputs.truncated",
                bound=bound_truncated,
                returned=returned_truncated,
                validate=self._validate_step_buffers,
            )
            nonfinite_state_mask = self._get_nonfinite_termination_mask()
            self._combine_done_outputs(nonfinite_state_mask)

        with self._nvtx_range("get_rewards"):
            bound_rewards = self.reward_buf
            returned_rewards = self._get_rewards()
            self.reward_buf = self._accept_hook_tensor(
                hook="_get_rewards",
                slot="outputs.rewards",
                bound=bound_rewards,
                returned=returned_rewards,
                validate=self._validate_step_buffers,
            )
            self._post_dones_and_rewards(nonfinite_state_mask)
            return self._sanitize_rewards_for_nonfinite(nonfinite_state_mask)

    def _compute_dones_and_rewards_parallel(self) -> torch.Tensor | None:
        (
            dones_torch_stream,
            dones_wp_stream,
            rewards_torch_stream,
            rewards_wp_stream,
        ) = self._get_dones_rewards_parallel_streams()
        current_stream = torch.cuda.current_stream(self.device)
        dones_torch_stream.wait_stream(current_stream)
        rewards_torch_stream.wait_stream(current_stream)

        nonfinite_state_mask = None
        bound_terminated = self.reset_terminated
        bound_truncated = self.reset_time_outs
        bound_rewards = self.reward_buf
        with self._nvtx_range("get_dones"):
            with wp.ScopedStream(dones_wp_stream, sync_enter=False, sync_exit=False):
                with torch.cuda.stream(dones_torch_stream):
                    returned_terminated, returned_truncated = self._get_dones()
                    self.reset_terminated = self._accept_hook_tensor(
                        hook="_get_dones",
                        slot="outputs.terminated",
                        bound=bound_terminated,
                        returned=returned_terminated,
                        validate=self._validate_step_buffers,
                    )
                    self.reset_time_outs = self._accept_hook_tensor(
                        hook="_get_dones",
                        slot="outputs.truncated",
                        bound=bound_truncated,
                        returned=returned_truncated,
                        validate=self._validate_step_buffers,
                    )
                    nonfinite_state_mask = self._get_nonfinite_termination_mask()

        with self._nvtx_range("get_rewards"):
            with wp.ScopedStream(rewards_wp_stream, sync_enter=False, sync_exit=False):
                with torch.cuda.stream(rewards_torch_stream):
                    returned_rewards = self._get_rewards()
                    self.reward_buf = self._accept_hook_tensor(
                        hook="_get_rewards",
                        slot="outputs.rewards",
                        bound=bound_rewards,
                        returned=returned_rewards,
                        validate=self._validate_step_buffers,
                    )

        current_stream.wait_stream(dones_torch_stream)
        current_stream.wait_stream(rewards_torch_stream)
        self._combine_done_outputs(nonfinite_state_mask)
        self._post_dones_and_rewards(nonfinite_state_mask)
        return self._sanitize_rewards_for_nonfinite(nonfinite_state_mask)

    def _combine_done_outputs(self, nonfinite_state_mask: torch.Tensor | None) -> None:
        if nonfinite_state_mask is not None:
            wp.launch(
                _logical_or_inplace_kernel,
                dim=(self.num_envs,),
                inputs=[self.reset_terminated, nonfinite_state_mask],
            )
        wp.launch(
            _logical_or_kernel,
            dim=(self.num_envs,),
            inputs=[self.reset_terminated, self.reset_time_outs],
            outputs=[self.reset_buf],
        )

    def _run_mjw_step(
        self,
        *,
        skip_kinematics: bool = False,
        skip_state_prefix: bool = False,
    ):
        """Advance through the repo-owned allocation-free MuJoCo Warp path.

        With the environment step graph off, the captured suffix runs as its own
        CUDA graph: capture does not execute work, replay produces the step, and
        the launch overhead of the many small Warp launches is paid once. The
        step runs directly when an outer graph is already recording the current
        torch or Warp stream, when the full step body is captured, when the
        configured model/integrator does not satisfy the allocation-free capture
        invariants, or off CUDA, where capture cannot record the matching torch
        work.
        """
        scratch = self._optimized_fwd_position_scratch
        if scratch is not None and hasattr(scratch, "implicit_qh_registration_frozen"):
            scratch.implicit_qh_registration_frozen = True
        if skip_state_prefix:
            step_fn = self._run_mjw_step_after_state_prefix_optimized_fwd_position
            graph_attr = (
                "_mjw_step_after_state_prefix_parallelized_graph"
                if self._use_parallelized_kernel_launch
                else "_mjw_step_after_state_prefix_graph"
            )
        elif skip_kinematics:
            step_fn = self._run_mjw_step_skip_optimized_fwd_position
            graph_attr = (
                "_mjw_step_skip_parallelized_graph"
                if self._use_parallelized_kernel_launch
                else "_mjw_step_skip_kinematics_graph"
            )
        else:
            step_fn = self._run_mjw_step_optimized_fwd_position
            graph_attr = (
                "_mjw_step_parallelized_graph"
                if self._use_parallelized_kernel_launch
                else "_mjw_step_graph"
            )

        if (
            self.cfg.use_cuda_graph
            or not _env_graph_capture_supported(self.device)
            or _current_stream_is_capturing(
                self.device,
                getattr(self, "_env_graph_wp_stream", None),
            )
        ):
            step_fn(self.mjw_model, self.mjw_data)
            return

        if not self._can_capture_mjw_step():
            if not self._mjw_step_capture_unsupported_logged:
                logger.info(
                    "MuJoCo Warp step suffix cannot be captured for this model, "
                    "integrator, or solver topology; running the step directly."
                )
                self._mjw_step_capture_unsupported_logged = True
            step_fn(self.mjw_model, self.mjw_data)
            return

        graph = getattr(self, graph_attr)
        if graph is None:
            graph_torch_stream, graph_wp_stream = self._get_env_graph_streams()
            with _stream_aligned_cuda_graph_capture(
                self.device,
                graph_torch_stream,
                graph_wp_stream,
            ) as capture:
                step_fn(self.mjw_model, self.mjw_data)
            graph = capture.graph
            setattr(self, graph_attr, graph)
        graph_torch_stream, graph_wp_stream = self._get_env_graph_streams()
        _launch_stream_aligned_cuda_graph(
            graph,
            self.device,
            graph_torch_stream,
            graph_wp_stream,
        )

    def _can_capture_mjw_step(self) -> bool:
        """Whether the configured step suffix satisfies capture invariants."""
        if self._mjw_step_capture_supported is None:
            probe = getattr(self, "supports_allocation_free_step_capture", None)
            self._mjw_step_capture_supported = (
                bool(probe()) if callable(probe) else True
            )
        return self._mjw_step_capture_supported

    def _run_mjw_step_optimized_fwd_position(
        self,
        _model: Any,
        _data: Any,
    ) -> None:
        scratch = self._optimized_fwd_position_scratch
        step_optimized_fwd_position(
            self.mjw_model,
            self.mjw_data,
            self._optimized_fwd_position_streams,
            scratch,
            use_optimized_kernel=self._use_optimized_kernel,
        )

    def _run_mjw_forward(self) -> None:
        """Refresh dynamics through the repo-owned allocation-free forward path.

        Runtime code must call this helper instead of ``mujoco_warp.forward`` so
        captured reset and controller paths reuse the environment's persistent
        scratch buffers.  An empty stream tuple intentionally selects sequential
        custom scheduling; ``None`` is reserved for utility-level stock fallbacks.
        """
        forward_optimized_fwd_position(
            self.mjw_model,
            self.mjw_data,
            self._optimized_fwd_position_streams,
            self._optimized_fwd_position_scratch,
            use_optimized_kernel=self._use_optimized_kernel,
        )

    def _run_mjw_forward_after_state_prefix(self) -> None:
        """Refresh action-dependent dynamics while reusing current state fields."""
        forward_after_state_prefix_optimized_fwd_position(
            self.mjw_model,
            self.mjw_data,
            self._optimized_fwd_position_streams,
            self._optimized_fwd_position_scratch,
            use_optimized_kernel=self._use_optimized_kernel,
        )

    def _run_mjw_actuation(self) -> None:
        """Refresh actuator outputs using the prepared allocation-free call."""
        call = getattr(self._optimized_fwd_position_scratch, "actuation_call", None)
        if call is None:
            mjw.fwd_actuation(self.mjw_model, self.mjw_data)
        else:
            call(self.mjw_model, self.mjw_data)

    def _run_mjw_control_forward(self) -> None:
        """Refresh dynamics needed after control changes in the current step phase."""
        if self._step_pipeline_active:
            self._run_mjw_forward_after_state_prefix()
        else:
            self._run_mjw_forward()

    def _register_mjw_implicit_qh_contributor(
        self,
        name: str,
        callback: Callable[..., None],
    ) -> None:
        """Register one named additive implicit-fast effective-inertia term."""
        from instantmimic.utils.mjw_step_optimized import ImplicitQHContributor

        if not isinstance(name, str) or not name.strip():
            raise ValueError("implicit qH contributor name must be nonempty")
        if self.mjw_model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            raise RuntimeError(
                "implicit qH contributors require the implicit-fast integrator"
            )
        scratch = self._optimized_fwd_position_scratch
        required_scratch = (
            "implicit_qderiv",
            "implicit_qacc",
            "implicit_qld",
            "implicit_ql_diag_inv",
        )
        if scratch is None or any(
            getattr(scratch, field, None) is None for field in required_scratch
        ):
            raise RuntimeError(
                "implicit qH contributors require optimized implicit-fast "
                "derivative scratch"
            )
        if not callable(getattr(scratch, "implicit_deriv_call", None)):
            raise RuntimeError(
                "implicit qH contributors require an allocation-free derivative call"
            )
        if not callable(callback):
            raise TypeError("implicit qH contributor callback must be callable")
        if bool(getattr(scratch, "implicit_qh_registration_frozen", False)):
            raise RuntimeError(
                "implicit qH contributors must be registered before stepping or capture"
            )
        if self._step_graph is not None or bool(self._step_graph_cache):
            raise RuntimeError(
                "implicit qH contributors must be registered before graph capture"
            )
        contributors = tuple(getattr(scratch, "implicit_qh_contributors", ()))
        if any(item.name == name for item in contributors):
            raise ValueError(f"duplicate implicit qH contributor name: {name!r}")
        scratch.implicit_qh_contributors = (
            *contributors,
            ImplicitQHContributor(name=name, apply=callback),
        )

    def _install_mjw_implicit_velocity_derivative_hook(
        self,
        hook: Callable[..., None],
    ) -> None:
        """Compatibility adapter for the fast-muscle velocity contribution."""
        self._register_mjw_implicit_qh_contributor("fast_muscle.force_velocity", hook)

    def _install_mjw_passive_force_hook(
        self,
        name: str,
        callback: Callable[[Any, Any], None],
    ) -> None:
        """Run ``callback(m, d)`` right after the native passive-force stage.

        The callback runs on the passive stage's stream once the native passive
        forces are written and may add generalized forces to ``d.qfrc_passive``;
        it reads position-stage kinematics only. Like other step hooks it must be
        allocation-free and is registered before stepping so the CUDA graph
        topology stays fixed.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("passive force hook name must be nonempty")
        if not callable(callback):
            raise TypeError("passive force hook callback must be callable")
        scratch = self._optimized_fwd_position_scratch
        if scratch is None or scratch.passive_call is None:
            raise RuntimeError("passive force hook requires persistent step scratch")
        if getattr(scratch, "advance_call", None) is not None:
            # The sleep path binds the native passive call when the scratch is
            # built, so a later wrapper would not reach it.
            raise RuntimeError("passive force hook does not support sleeping models")
        if bool(getattr(scratch, "implicit_qh_registration_frozen", False)):
            raise RuntimeError(
                "passive force hook must be registered before stepping or capture"
            )
        if self._step_graph is not None or bool(self._step_graph_cache):
            raise RuntimeError(
                "passive force hook must be registered before graph capture"
            )
        installed = getattr(self, "_mjw_passive_force_hook_name", None)
        if installed is not None:
            raise ValueError(
                "only one passive force hook is supported; "
                f"already registered: {installed!r}"
            )
        native_passive = scratch.passive_call

        def passive_with_hook(m: Any, d: Any) -> None:
            native_passive(m, d)
            callback(m, d)

        scratch.passive_call = passive_with_hook
        self._mjw_passive_force_hook_name = name

    def _install_mjw_contact_hook(
        self,
        name: str,
        callback: Callable[[Any, Any], None],
        *,
        body_groups: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Run ``callback(m, d)`` right after the native collision narrowphase.

        The callback may append contacts to ``d.contact`` (through MuJoCo Warp's
        ``write_contact``) from position-stage kinematics; constraint assembly
        sees them like native contacts. It must be allocation-free and is
        registered before stepping so the CUDA graph topology stays fixed.

        ``body_groups`` lists body ids that one of its contacts may couple
        (every body a contact's two sides, blended or not, can touch), best derived
        from the producer's own data. With it the fused Newton can factor H on a
        sparse pattern; without it the dense factor stays.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("contact hook name must be nonempty")
        if not callable(callback):
            raise TypeError("contact hook callback must be callable")
        scratch = self._optimized_fwd_position_scratch
        if scratch is None or getattr(scratch, "collision_context", None) is None:
            raise RuntimeError("contact hook requires persistent step scratch")
        if getattr(scratch, "constraint_free", False):
            raise RuntimeError("contact hook needs a model with constraint rows")
        if getattr(scratch, "advance_call", None) is not None:
            raise RuntimeError("contact hook does not support sleeping models")
        if bool(getattr(scratch, "implicit_qh_registration_frozen", False)):
            raise RuntimeError(
                "contact hook must be registered before stepping or capture"
            )
        if self._step_graph is not None or bool(self._step_graph_cache):
            raise RuntimeError("contact hook must be registered before graph capture")
        installed = getattr(self, "_mjw_contact_hook_name", None)
        if installed is not None:
            raise ValueError(
                f"only one contact hook is supported; already registered: {installed!r}"
            )
        if body_groups is not None:
            body_groups = [[int(b) for b in group] for group in body_groups]
            bad = sorted(
                {
                    b
                    for group in body_groups
                    for b in group
                    if not 0 <= b < self.model.nbody
                }
            )
            if bad:
                raise ValueError(
                    f"contact hook body groups name invalid body ids: {bad}"
                )
        native = scratch.collision_narrowphase_call or mjw_collision_driver._narrowphase

        def narrowphase_with_hook(m: Any, d: Any, ctx: Any) -> None:
            native(m, d, ctx)
            callback(m, d)

        scratch.collision_narrowphase_call = narrowphase_with_hook
        self._mjw_contact_hook_name = name
        self._mjw_contact_body_groups = body_groups
        self._refresh_mjw_sparse_hessian()

    def _refresh_mjw_sparse_hessian(self) -> None:
        """Choose the fused Newton's Hessian factor for the current row producers.

        With optimized kernels, a model whose constraint rows couple a sparse set
        of dofs (``sparse_hessian.plan_for_model``) factors H with the static-pattern
        sparse L'DL; otherwise the dense blocked factor stays. Custom contact hooks
        must declare the body groups they couple. Construction-time only.

        On the fused path the factor's pattern-violation counter is zeroed here and
        read by ``_check_sparse_hessian_violations``.
        """
        scratch = self._optimized_fwd_position_scratch
        ctx = getattr(scratch, "solver_context", None)
        self._sparse_hessian_violations = None
        if ctx is None:
            return
        ctx.sparse_hessian_plan = None
        if not self._use_optimized_kernel:
            return
        from instantmimic.utils.mjw_kernels import newton_fusion, sparse_hessian

        groups: list[list[int]] | None = []
        if getattr(self, "_mjw_contact_hook_name", None) is not None:
            groups = getattr(self, "_mjw_contact_body_groups", None)
        m = self.mjw_model
        plan, reason = sparse_hessian.plan_for_model(
            m,
            ctx,
            groups,
            newton_fusion.newton_block_dim(m),
            compact=bool(getattr(scratch, "solver_uses_compact", False)),
            fused_runtime=getattr(scratch, "mul_m_call", None) is not None,
        )
        ctx.sparse_hessian_plan = plan
        if plan is None:
            logger.info("MJW_SPARSE_HESSIAN off: %s", reason)
        else:
            if plan.path == "fused":
                counter = wp.to_torch(
                    sparse_hessian.violation_counter(ctx.hfactor, plan)
                )
                counter.zero_()
                self._sparse_hessian_violations = counter
            logger.info(
                "MJW_SPARSE_HESSIAN on nv=%d slots=%d terms=%d levels=%d density=%.2f",
                plan.nv,
                plan.nslots,
                plan.nterms,
                plan.levels,
                plan.density,
            )

    def _check_sparse_hessian_violations(self) -> dict[str, float]:
        """Raise if constraint rows coupled dofs outside the sparse Hessian pattern.

        The fused factor drops such couplings, so the Newton direction is wrong from
        then on: a contact producer coupled bodies its declared ``body_groups`` do
        not cover. Runs at the metric-sync cadence; reports no scalars.
        """
        counter = self._sparse_hessian_violations
        if counter is None:
            return {}
        per_world = counter.cpu()
        if bool((per_world > 0).any()):
            # float32 tallies: exact up to 2**24 per world, approximate beyond
            total = float(per_world.sum().item())
            worlds = int((per_world > 0).sum().item())
            counter.zero_()
            hook = getattr(self, "_mjw_contact_hook_name", None)
            raise RuntimeError(
                f"sparse Hessian: {total:.0f} constraint-row couplings in {worlds} "
                "world(s) fell outside the declared coupling pattern since the last "
                "check, so the Newton factor dropped them; contact hook "
                f"{hook!r} must declare every body its contacts can couple "
                "(body_groups)"
            )
        return {}

    def _install_mjw_pre_advance_modifier(
        self,
        name: str,
        callback: Callable[..., tuple[Any, Any | None]],
        *,
        implicit_factor: Any | None = None,
    ) -> None:
        """Install the single allocation-free modifier run before integration.

        A modifier may replace the acceleration used to update ``qvel`` and
        independently provide the generalized velocity used for position drift.
        ``implicit_factor`` receives the implicit-fast qH factor each step.
        Registration is construction-only so CUDA graph topology stays fixed.
        """
        from instantmimic.utils.mjw_step_optimized import PreAdvanceModifier

        if not isinstance(name, str) or not name.strip():
            raise ValueError("pre-advance modifier name must be nonempty")
        if not callable(callback):
            raise TypeError("pre-advance modifier callback must be callable")
        scratch = self._optimized_fwd_position_scratch
        if scratch is None:
            raise RuntimeError("pre-advance modifier requires persistent step scratch")
        if bool(getattr(scratch, "implicit_qh_registration_frozen", False)):
            raise RuntimeError(
                "pre-advance modifier must be registered before stepping or capture"
            )
        if self._step_graph is not None or bool(self._step_graph_cache):
            raise RuntimeError(
                "pre-advance modifier must be registered before graph capture"
            )
        installed = getattr(scratch, "pre_advance_modifier", None)
        if installed is not None:
            raise ValueError(
                "only one pre-advance modifier is supported; "
                f"already registered: {installed.name!r}"
            )
        free_bodies = getattr(getattr(self, "mjw_model", None), "body_freeadr", None)
        if implicit_factor is not None and free_bodies is not None and free_bodies.size:
            raise ValueError(
                "retained Cholesky factors cannot represent the nonsymmetric "
                "implicit free-body correction"
            )
        scratch.pre_advance_modifier = PreAdvanceModifier(
            name=name,
            apply=callback,
            implicit_factor=implicit_factor,
        )

    @staticmethod
    def _nvtx_range(name: str) -> ContextManager[object]:
        """An NVTX range only where NVTX can do anything.

        `reset_into`, `step` and `step_with_io` carried
        `@torch.cuda.nvtx.range` decorators, which build the context manager at
        class-definition time and push a range on every call with no way to opt
        out. Gating on CUDA availability matches `startup_nvtx_range`, which
        this module already uses for construction phases, and leaves a CPU run
        paying nothing.
        """

        if not torch.cuda.is_available():
            return nullcontext()
        return torch.cuda.nvtx.range(name)

    def _run_mjw_state_prefix(self, *, constraints: bool = True) -> None:
        """Refresh complete action-independent position and velocity state.

        ``constraints=False`` skips collision and constraint construction (see
        ``_run_physics_substep``)."""
        forward_state_prefix_optimized_fwd_position(
            self.mjw_model,
            self.mjw_data,
            self._optimized_fwd_position_streams,
            self._optimized_fwd_position_scratch,
            use_optimized_kernel=self._use_optimized_kernel,
            constraints=constraints,
        )

    def _run_mjw_kinematics_com_state(self) -> None:
        """Kinematics, COM position and COM velocity as the state prefix derives them."""
        kinematics_com_state_optimized_fwd_position(
            self.mjw_model,
            self.mjw_data,
            self._optimized_fwd_position_streams,
            self._optimized_fwd_position_scratch,
            use_optimized_kernel=self._use_optimized_kernel,
        )

    def _run_mjw_step_after_state_prefix_optimized_fwd_position(
        self,
        _model: Any,
        _data: Any,
    ) -> None:
        scratch = self._optimized_fwd_position_scratch
        step_after_state_prefix_optimized_fwd_position(
            self.mjw_model,
            self.mjw_data,
            self._optimized_fwd_position_streams,
            scratch,
            use_optimized_kernel=self._use_optimized_kernel,
        )

    def _run_mjw_step_skip_optimized_fwd_position(
        self,
        _model: Any,
        _data: Any,
    ) -> None:
        scratch = self._optimized_fwd_position_scratch
        step_skip_kinematics_optimized_fwd_position(
            self.mjw_model,
            self.mjw_data,
            self._optimized_fwd_position_streams,
            scratch,
            use_optimized_kernel=self._use_optimized_kernel,
        )

    def _run_mjw_kinematics(self):
        """Refresh MuJoCo Warp kinematics using the configured kinematics implementation."""
        if self._use_parallelized_kernel_launch:
            kinematics_parallel_tail(
                self.mjw_model,
                self.mjw_data,
                self._optimized_fwd_position_streams[:4],
            )
        else:
            mjw.kinematics(self.mjw_model, self.mjw_data)

    def _run_mjw_com_state(self):
        """Refresh MuJoCo Warp COM position and velocity state after external qpos/qvel edits."""
        if self._use_parallelized_kernel_launch:
            com_pos_vel_parallel(
                self.mjw_model,
                self.mjw_data,
                self._optimized_fwd_position_streams,
            )
        else:
            mjw.com_pos(self.mjw_model, self.mjw_data)
            mjw.com_vel(self.mjw_model, self.mjw_data)

    def _get_nonfinite_termination_mask(self) -> torch.Tensor | None:
        """Return envs with non-finite core state, or ``None`` when disabled."""
        if not self._terminate_on_nonfinite_enabled():
            return None
        mask_buf = self._nonfinite_state_mask_buf
        if mask_buf is not None:
            wp.launch(
                _mark_nonfinite_state_envs_kernel,
                dim=(self.num_envs,),
                inputs=[self.mjw_data.qpos, self.mjw_data.qvel],
                outputs=[mask_buf],
            )
            return mask_buf
        qpos = wp.to_torch(self.mjw_data.qpos)
        qvel = wp.to_torch(self.mjw_data.qvel)
        qpos_bad = ~torch.isfinite(qpos).all(dim=-1)
        qvel_bad = ~torch.isfinite(qvel).all(dim=-1)
        return torch.logical_or(qpos_bad, qvel_bad)

    def _sanitize_rewards_for_nonfinite(
        self,
        nonfinite_state_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Terminate and zero rewards for envs with non-finite state or reward."""
        if not self._terminate_on_nonfinite_enabled():
            return nonfinite_state_mask
        combined_bad_mask = self._nonfinite_combined_bad_mask_buf
        if combined_bad_mask is not None and nonfinite_state_mask is not None:
            wp.launch(
                _sanitize_nonfinite_rewards_kernel,
                dim=(self.num_envs,),
                inputs=[
                    self.reward_buf,
                    nonfinite_state_mask,
                    self.reset_terminated,
                    self.reset_buf,
                ],
                outputs=[combined_bad_mask],
            )
            return combined_bad_mask

        reward_bad_mask = ~torch.isfinite(self.reward_buf)
        if nonfinite_state_mask is None:
            combined_bad_mask = reward_bad_mask
        else:
            combined_bad_mask = torch.logical_or(nonfinite_state_mask, reward_bad_mask)

        self.reset_terminated.logical_or_(combined_bad_mask)
        self.reset_buf.logical_or_(combined_bad_mask)
        self.reward_buf.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        self.reward_buf.masked_fill_(combined_bad_mask, 0.0)
        return combined_bad_mask

    def _terminate_on_nonfinite_enabled(self) -> bool:
        termination = getattr(self.cfg, "termination", None)
        if termination is not None:
            return bool(get_value(termination, "nonfinite", False))
        return bool(getattr(self.cfg, "terminate_on_nonfinite", False))

    def _configure_done_reason_tracking(self, labels: tuple[str, ...]) -> None:
        """Register per-env termination reason buffers for optional logging."""
        self._done_reason_labels = tuple(str(label) for label in labels)
        if not self._track_done_reasons:
            return
        self._done_reason_mask_buf = torch.zeros(
            (len(self._done_reason_labels), self.num_envs),
            dtype=torch.bool,
            device=self.device,
        )
        self._done_reason_count_buf = torch.zeros(
            len(self._done_reason_labels),
            dtype=torch.int32,
            device=self.device,
        )

    def _accumulate_done_reason_stats(
        self,
        nonfinite_mask: torch.Tensor | None,
    ) -> None:
        if not self._track_done_reasons:
            return
        if nonfinite_mask is None:
            nonfinite_mask = self._done_reason_empty_mask_buf
        wp.launch(
            _accumulate_done_reason_stats_kernel,
            dim=(self.num_envs,),
            inputs=[
                self.reset_terminated,
                self.reset_time_outs,
                nonfinite_mask,
                self._done_reason_mask_buf,
                self._done_reason_count_buf,
                self._done_reason_terminated_total_buf,
                self._done_reason_time_out_total_buf,
                self._done_reason_nonfinite_total_buf,
            ],
        )

    def register_diagnostic_stats_source(
        self,
        consume: Callable[[], dict[str, float]],
    ) -> None:
        """Register a component's diagnostic scalars for training logs.

        ``consume`` returns the scalars accumulated since its previous call and
        resets them; it runs at the training metric-sync cadence, outside CUDA
        graph capture.
        """
        self._diagnostic_stats_sources.append(consume)

    def register_reset_state_finalizer(
        self,
        finalize: Callable[[torch.Tensor], None],
    ) -> None:
        """Register a component step that runs on the final reset state.

        ``finalize(reset_mask)`` runs after the task's own reset finalization,
        on the refreshed kinematics of the reset worlds, inside the same
        (possibly captured) step; it must be allocation-free and mask-driven.
        """
        if (
            self._step_lifecycle_registration_frozen
            or self._step_graph is not None
            or bool(self._step_graph_cache)
        ):
            raise RuntimeError(
                "Reset state finalizers must be registered before graph capture."
            )
        self._reset_state_finalizers.append(finalize)

    def consume_diagnostic_stats(self) -> dict[str, float]:
        """Return and reset the scalars of every registered diagnostic source."""
        stats: dict[str, float] = {}
        for consume in self._diagnostic_stats_sources:
            stats.update(consume())
        return stats

    def consume_done_reason_stats(self) -> dict[str, float]:
        """Return and reset accumulated done-reason statistics."""
        if not bool(getattr(self.cfg, "track_done_reasons", False)):
            return {}

        terminated_total = int(self._done_reason_terminated_total_buf.cpu().item())
        time_out_total = int(self._done_reason_time_out_total_buf.cpu().item())
        nonfinite_total = int(self._done_reason_nonfinite_total_buf.cpu().item())
        reason_counts_cpu = self._done_reason_count_buf.cpu()

        stats: dict[str, float] = {
            "Done / Terminated total": float(terminated_total),
            "Done / Time out total": float(time_out_total),
            "Done / Nonfinite count": float(nonfinite_total),
            "Done / Nonfinite fraction": (
                float(nonfinite_total) / float(terminated_total)
                if terminated_total > 0
                else 0.0
            ),
        }
        for reason_idx, label in enumerate(self._done_reason_labels):
            count = int(reason_counts_cpu[reason_idx].item())
            stats[f"Done / {label} count"] = float(count)
            stats[f"Done / {label} fraction"] = (
                float(count) / float(terminated_total) if terminated_total > 0 else 0.0
            )

        self._done_reason_count_buf.zero_()
        self._done_reason_terminated_total_buf.zero_()
        self._done_reason_time_out_total_buf.zero_()
        self._done_reason_nonfinite_total_buf.zero_()
        return stats

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Seed the numpy and torch global generators.

        `wp.rand_init` is also called, but it does not seed Warp globally: it
        returns a per-thread RNG state, and the return value is discarded here.
        Warp kernels in this repository build their own state from a seed passed
        in explicitly -- `reset_envs_kernel` reads `random_seed[0]` -- so
        reproducibility of the Warp side comes from that argument, not from this
        call.

        Args:
            seed: Value applied to the numpy and torch generators.

        Returns:
            The seed that was applied, for the caller to record.
        """

        np.random.seed(seed)
        torch.manual_seed(seed)
        wp.rand_init(seed)
        return seed

    def render(self, recompute: bool = False) -> np.ndarray | list[np.ndarray] | None:
        """Render the current environment state using the configured render runtime."""
        return self._render_runtime.render(recompute=recompute)

    def set_collect_render_substeps(self, enabled: bool) -> bool:
        """Enable playback-time substep frame capture for multi-frame rendering."""
        return self._render_runtime.set_collect_render_substeps(enabled)

    def get_render_output_fps(self) -> float:
        """Return the effective fps of frames produced by ``render()``."""
        return self._render_runtime.get_render_output_fps()

    def get_playback_render_state_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return qpos/qvel states needed to render the just-finished playback step."""
        return self._render_runtime.get_playback_render_state_batch()

    def stage_playback_render_state_batch_cpu(self) -> tuple[np.ndarray, np.ndarray]:
        """Stage the playback render-state batch to CPU once for non-CUDA render backends."""
        return self._render_runtime.stage_playback_render_state_batch_cpu()

    def stage_playback_render_tile_state_batch_cpu(
        self,
        env_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stage current qpos/qvel for multiple envs as one tiled playback frame."""
        return self._render_runtime.stage_playback_render_tile_state_batch_cpu(
            env_count
        )

    def get_playback_render_tile_state_batch(
        self,
        env_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current qpos/qvel for multiple envs without staging through CPU."""
        return self._render_runtime.get_playback_render_tile_state_batch(env_count)

    def get_playback_render_mocap_batch(
        self,
        qpos_batch: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return mocap state shaped to match a playback render qpos batch."""
        return self._render_runtime.get_playback_render_mocap_batch(qpos_batch)

    def stage_playback_render_mocap_batch_cpu(
        self,
        qpos_batch: torch.Tensor | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Stage playback mocap state to CPU when the render model matches sim."""
        return self._render_runtime.stage_playback_render_mocap_batch_cpu(qpos_batch)

    def configure_train_full_video_capture(
        self,
        *,
        rollout_steps: int,
        env_count: int,
    ) -> None:
        """Configure rollout-local tiled state buffers for full training videos."""
        self._render_runtime.configure_train_full_video_capture(
            rollout_steps=rollout_steps,
            env_count=env_count,
        )

    def prepare_train_full_video_step_capture(
        self,
        step_idx: int,
        *,
        activation: torch.Tensor | None = None,
    ) -> None:
        """Select the rollout step slot used by subsequent render-state capture kernels."""
        self._render_runtime.prepare_train_full_video_step_capture(
            step_idx,
            activation=activation,
        )

    def capture_train_full_video_step_state(
        self,
        step_idx: int,
        *,
        activation: torch.Tensor | None = None,
    ) -> None:
        """Capture the final policy-step state when substep capture is not active."""
        self._render_runtime.capture_train_full_video_step_state(
            step_idx,
            activation=activation,
        )

    def get_train_full_video_render_tile_state_batch(
        self,
    ) -> object:
        """Return captured train rollout render states as a tiled frame batch."""
        return self._render_runtime.get_train_full_video_render_tile_state_batch()

    def _render_rgb_array_from_state(
        self,
        qpos: torch.Tensor,
        qvel: torch.Tensor,
    ) -> np.ndarray:
        return self._render_runtime.render_rgb_array_from_state(qpos, qvel)

    def get_checkpoint_state(self) -> dict[str, object]:
        """Return optional environment state to persist inside checkpoints."""
        common_step_counter = getattr(self, "common_step_counter", None)
        if not isinstance(common_step_counter, torch.Tensor):
            return {}
        return {
            "direct_rl_env": {
                "schema_version": 1,
                "common_step_counter": common_step_counter.detach().clone(),
            }
        }

    def _prepare_checkpoint_restore(
        self,
        state: dict[str, object] | None,
    ) -> torch.Tensor | None:
        """Validate base runtime state without mutating the live environment."""

        if not isinstance(state, dict) or "direct_rl_env" not in state:
            return None
        runtime_state = state["direct_rl_env"]
        if not isinstance(runtime_state, dict):
            raise ValueError("Checkpoint field 'direct_rl_env' must be a mapping.")
        schema_version = runtime_state.get("schema_version")
        if type(schema_version) is not int or schema_version != 1:
            raise ValueError(
                "DirectRLEnv checkpoint field 'schema_version' is incompatible: "
                f"checkpoint={schema_version!r}, current=1."
            )
        if "common_step_counter" not in runtime_state:
            raise ValueError(
                "DirectRLEnv checkpoint field 'common_step_counter' is missing."
            )
        destination = getattr(self, "common_step_counter", None)
        if not isinstance(destination, torch.Tensor):
            raise ValueError(
                "DirectRLEnv restore destination 'common_step_counter' must be a "
                "tensor."
            )
        try:
            source = torch.as_tensor(runtime_state["common_step_counter"])
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                "DirectRLEnv checkpoint field 'common_step_counter' is not a valid "
                "tensor."
            ) from exc
        if source.shape != destination.shape:
            raise ValueError(
                "DirectRLEnv checkpoint field 'common_step_counter' shape mismatch: "
                f"expected {tuple(destination.shape)}, got {tuple(source.shape)}."
            )
        if source.dtype != destination.dtype:
            raise ValueError(
                "DirectRLEnv checkpoint field 'common_step_counter' dtype mismatch: "
                f"expected {destination.dtype}, got {source.dtype}."
            )
        return source.to(device=destination.device).detach().clone()

    def _apply_checkpoint_restore(self, common_step_counter: torch.Tensor) -> None:
        self.common_step_counter.copy_(common_step_counter)

    def load_checkpoint_state(self, state: dict[str, object] | None) -> bool:
        """Restore optional environment state from a checkpoint payload."""
        common_step_counter = self._prepare_checkpoint_restore(state)
        if common_step_counter is None:
            return False
        self._apply_checkpoint_restore(common_step_counter)
        return True

    def copy_runtime_state_from(self, source_env: Any) -> bool:
        """Copy mutable in-process state from a peer env without host staging."""
        source_raw = (
            source_env.unwrapped if hasattr(source_env, "unwrapped") else source_env
        )
        source_control = getattr(source_raw, "_control_pipeline", None)
        copier = getattr(self._control_pipeline, "copy_runtime_state_from", None)
        if source_control is None or not callable(copier):
            return False
        return bool(copier(source_control))

    def supports_allocation_free_step_capture(self) -> bool:
        """Return whether the configured MuJoCo Warp step can be nested in a graph."""
        scratch = self._optimized_fwd_position_scratch
        if scratch is None:
            return False
        if not supports_allocation_free_implicit_capture(
            self.mjw_model,
            scratch,
            use_optimized_kernel=self._use_optimized_kernel,
        ):
            return False
        if self.mjw_data.njmax == 0 or self.mjw_model.nv == 0:
            return True
        return bool(
            getattr(scratch, "solver_context", None) is not None
            and getattr(scratch, "solver_nsolving", None) is not None
        )

    def close(self):
        """Release the renderer. Idempotent; a second call returns immediately."""
        if self._is_closed:
            return
        integrator = getattr(self, "_substep_integrator", None)
        if integrator is not None:
            integrator.close()
        render_runtime = self._render_runtime
        if render_runtime is not None:
            render_runtime.close()
        self._is_closed = True

    """
    Debug visualization.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Toggle optional environment debug visualization."""
        if not bool(getattr(self, "supports_debug_vis", False)):
            return False
        self._set_debug_vis_impl(bool(debug_vis))
        return True

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Resolve the configured space specs into Gymnasium spaces and batch them."""

        observation_space_spec = self._get_runtime_cfg_value("observation_space")
        action_space_spec = self._get_runtime_cfg_value("action_space")
        # Derived environments do not author an observation width. Their model
        # and option dependent layout is finalized after construction. Keep this
        # method available as the provisional/fixed builder for families that do
        # provide a width, while allowing the common constructor to proceed.
        if (
            action_space_spec is None
            or action_space_spec is _CFG_ABSENT
            or action_space_spec is _CFG_MANDATORY_MISSING
        ):
            raise ValueError("action_space must be available for Gym setup")

        if (
            observation_space_spec is None
            or observation_space_spec is _CFG_ABSENT
            or observation_space_spec is _CFG_MANDATORY_MISSING
        ):
            self._derived_spaces_pending = True
            self.single_action_space = spec_to_gym_space(action_space_spec)
            self.action_space = gym.vector.utils.batch_space(
                self.single_action_space, self.num_envs
            )
            self.actions = sample_space(
                self.single_action_space,
                self.device,
                batch_size=self.num_envs,
                fill_value=0,
            )
            return

        # set up spaces
        self._derived_spaces_pending = False
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = spec_to_gym_space(
            observation_space_spec
        )
        self.single_action_space = spec_to_gym_space(action_space_spec)

        # optional state space for asymmetric actor-critic architectures
        self.state_space = None
        if self.cfg.get("state_space", None):
            self.single_observation_space["critic"] = spec_to_gym_space(
                self.cfg.state_space
            )
            self.state_space = gym.vector.utils.batch_space(
                self.single_observation_space["critic"], self.num_envs
            )

        # batch the spaces for vectorized environments. The public reset/step API
        # returns the complete observation mapping, including an optional critic
        # observation, so the Gym space must describe that same mapping.
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )

        # instantiate actions (needed for tasks for which the observations computation is dependent on the actions)
        self.actions = sample_space(
            self.single_action_space,
            self.device,
            batch_size=self.num_envs,
            fill_value=0,
        )

    def get_control_runtime_spec(self):
        spec = self._control_pipeline.runtime_spec()
        if spec is not None:
            return spec
        return make_direct_control_runtime_spec(
            action_dim=int(self.cfg.action_space),
            action_mode=str(self.cfg.get("action_mode", "direct")),
        )

    def get_full_video_activation(self) -> torch.Tensor | None:
        """Return the activation currently applied by an environment controller."""
        return self._control_pipeline.get_full_video_activation()

    def _reset_idx(self, reset_mask: torch.Tensor):
        """Reset environments based on a boolean mask.

        Args:
            reset_mask: Boolean mask of shape (num_envs,). True = reset that env.
        """

        wp.launch(
            _reset_episode_length_kernel,
            dim=(self.num_envs,),
            inputs=[
                self.episode_length_buf,
                reset_mask,
            ],
        )

    """
    Subclass hooks.
    """

    def _build_render_model(self) -> tuple[mujoco.MjModel, mujoco.MjData] | None:
        """Build a separate model/data pair for rendering.

        Override in subclasses to add ghost/reference bodies for visualization.
        Return None to use the simulation model for rendering (default).
        """
        return None

    def _update_render_data(self):
        """Update render data before rendering (e.g., set ghost body positions)."""
        return None

    def _pre_physics_step(self, actions: torch.Tensor):
        """Turn a policy action into whatever `_apply_action` will write.

        Called once per policy step, before any physics substep. A control pipeline
        handles this when one is configured.

        Args:
            actions: Shape `(num_envs, action_dim)`.
        """
        if self._control_pipeline.is_enabled:
            self._control_pipeline.pre_physics_step(actions)
            return
        raise NotImplementedError(
            f"Please implement the '_pre_physics_step' method for {self.__class__.__name__}."
        )

    def _advance_physics_substep(self, substep_idx: int) -> None:
        del substep_idx
        with self._nvtx_range("apply_action"):
            self._apply_action()
        with self._nvtx_range("physics_step"):
            self._run_mjw_step(skip_state_prefix=True)

    def _apply_action(self):
        """Write the prepared control into MuJoCo Warp.

        Called once per physics substep, so it must not allocate. A control
        pipeline handles this when one is configured.
        """
        if self._control_pipeline.is_enabled:
            self._control_pipeline.apply_action()
            return
        raise NotImplementedError(
            f"Please implement the '_apply_action' method for {self.__class__.__name__}."
        )

    def _post_physics_substep(self, substep_idx: int) -> None:
        """Optional hook after each physics substep."""
        del substep_idx

    def _post_state_prefix_substep(self, substep_idx: int) -> None:
        """Optional hook after refreshing state for each physics substep."""
        del substep_idx

    def _register_post_physics_substep_hook(self, hook: Callable[[int], None]) -> None:
        """Run ``hook(substep_idx)`` with the control pipeline's substep endpoint.

        A stateful controller attached directly to the environment, rather than
        through the control pipeline, registers its endpoint commit here (for
        example controller-owned state).
        """
        if (
            self._step_lifecycle_registration_frozen
            or self._step_graph is not None
            or bool(self._step_graph_cache)
        ):
            raise RuntimeError(
                "Post-physics substep hooks must be registered before graph capture."
            )
        self._post_physics_substep_hooks.append(hook)

    def _run_control_post_physics_substep(self, substep_idx: int) -> None:
        """Run the stateful endpoint callbacks for one physics substep."""
        self._control_pipeline.post_physics_substep(substep_idx)
        for hook in self._post_physics_substep_hooks:
            hook(substep_idx)

    @abstractmethod
    def _get_observations(self) -> VecEnvObs:
        """Fill and return this step's observation groups.

        Must be a pure function of the simulation and task state: with
        terminal observations enabled (PPO time-limit bootstrap, off-policy
        agents) it runs twice per step, before and after the reset. Derive
        quantities such as velocities or phases from the current state rather
        than from a history this method advances.

        Returns:
            A mapping holding every bound slot. Ordinarily a fresh tensor is
            absorbed with a copy, but `step_with_io(validate_buffers=True)`
            requires the bound mapping and its tensors back unchanged and raises
            `BufferContractError` otherwise. What breaks graph capture is
            allocating inside the captured body, not the shape of what this
            returns.
        """
        raise NotImplementedError(
            f"Please implement the '_get_observations' method for {self.__class__.__name__}."
        )

    def _get_states(self) -> VecEnvObs | None:
        """Return the privileged critic observation, if this task has one.

        Nothing in this package calls this: an asymmetric critic reads the
        `"critic"` group out of the observation mapping instead. The hook remains
        for tasks that want an explicit place to build that group.
        """
        return None

    @abstractmethod
    def _get_rewards(self) -> torch.Tensor:
        """Return this step's reward.

        Returns:
            Shape `(num_envs,)`.
        """
        raise NotImplementedError(
            f"Please implement the '_get_rewards' method for {self.__class__.__name__}."
        )

    @abstractmethod
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two episode-end signals, kept separate for bootstrapping.

        Returns:
            Terminated and timed-out masks, each shape `(num_envs,)`. A time-out is
            not absorbing, so its value estimate must still be carried.
        """
        raise NotImplementedError(
            f"Please implement the '_get_dones' method for {self.__class__.__name__}."
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle this task's debug visuals.

        Build the visual bodies on the first enabling call and flip their visibility
        afterwards. Only reached when the subclass sets `supports_debug_vis`.
        """
        raise NotImplementedError(
            f"Debug visualization is not implemented for {self.__class__.__name__}."
        )
