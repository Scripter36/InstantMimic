"""Configuration dataclasses every DirectRLEnv subclass builds on."""

from dataclasses import MISSING, dataclass, field
from typing import Any

from instantmimic.envs.common import SpaceType
from instantmimic.utils.ball_joint_limits import BallJointLimitCfg


@dataclass
class SimulationCfg:
    mjcf_path: str
    integrator: str
    render_interval: int
    njmax: int
    nconmax: int
    hz: float
    iterations: int | None = None
    warn_overflow: bool = False
    """Print per-world MuJoCo Warp device warnings; overflow bits remain recorded.

    Disabled by default because capped solvers can emit millions of device printf
    messages during graph replay. Enable explicitly for short diagnostic runs.
    """

    ls_iterations: int | None = None
    is_sparse: bool | None = None
    disable_multiccd: bool | None = None
    ball_joint_limits: list[BallJointLimitCfg] = field(default_factory=list)
    ball_joint_limit_mode: str = "solver"
    """Enforce directional ball limits with ``solver`` rows or task-owned forces.

    ``force`` resolves the same anatomical configuration but disables the native
    scalar cap and leaves force application to the environment.  It never adds
    directional rows to the MuJoCo constraint solver.
    """


@dataclass
class InteractiveSceneCfg:
    num_envs: int


@dataclass
class ViewerCfg:
    resolution: tuple[int, int]


@dataclass(kw_only=True)
class DirectRLEnvCfg:
    """Settings a task must supply for `DirectRLEnv` to construct itself.

    A field declared `MISSING` has no default, so the generated keyword-only
    `__init__` requires it. Hydra composition fills these from
    `config/<family>/`, and the runtime hands the environment that composed
    mapping rather than building this dataclass. Subclasses add their own fields,
    and `SimulationCfg` carries the MuJoCo model paths.
    """

    def get(self, name: str, default: Any = None) -> Any:
        """Return an attribute value using mapping-style lookup semantics."""

        return getattr(self, name, default)

    # simulation settings
    viewer: ViewerCfg = MISSING
    """Offscreen render resolution. Required; there is no default."""

    sim: SimulationCfg = MISSING
    """MuJoCo model paths and solver settings. Required; there is no default."""

    # general settings
    seed: int | None = None
    """Seed applied once at the start of environment construction.

    `None` leaves the global generators untouched. Seeding before construction is
    what makes reset sampling and initial-state randomization reproducible.
    """

    decimation: int = MISSING
    """Physics substeps executed per policy step.

    The policy timestep is `decimation` times the simulation timestep, so with a
    120 Hz model and `decimation: 4` the policy runs at 30 Hz. Each substep reuses
    the same action.
    """

    episode_length_s: float = MISSING
    """Time-out horizon in seconds.

    `DirectRLEnv.max_episode_length` converts it to policy steps on each access,
    dividing by `step_dt` and rounding up. Reaching it sets the time-out flag
    rather than the terminated flag.
    """

    # environment settings
    scene: InteractiveSceneCfg = MISSING
    """How many environment copies to simulate in parallel.

    Every copy shares one MuJoCo Warp model and one origin; there is no per-copy
    world offset.
    """

    observation_space: SpaceType = MISSING
    """Shape of the `"policy"` observation group, one environment.

    Written as an int, a list of ints, a one-element set, a list of those, a tuple
    or a dict; `instantmimic.utils.spaces.spec_to_gym_space` documents each form.
    The base class does not check this against the tensor a task produces. Most
    tasks check it themselves during construction and raise on a mismatch, a few
    resolve the value from the model instead, and a few trust the config as
    written.
    """

    state_space: SpaceType | None = None
    """Shape of the optional `"critic"` observation group, one environment.

    Set this only for an asymmetric actor-critic, where the critic sees privileged
    state the policy does not. `None` makes the critic reuse the policy group.
    Same specification forms as `observation_space`.
    """

    action_space: SpaceType = MISSING
    """Shape of the action the policy emits, one environment.

    Same specification forms as `observation_space`. The control pipeline selected
    by `env.action.type` decides how these numbers reach the actuators.
    """

    control: Any | None = None
    """Optional control pipeline config. Direct actions are used when omitted."""

    observation_noise: dict[str, Any] | None = None
    """Training-only Gaussian noise on the policy observation.

    ``{std: float >= 0, units: normalized | raw, enabled: bool}``, parsed by
    `instantmimic.envs.observation_noise`. `None` (the default), ``std: 0`` and
    ``enabled: false`` add nothing. Only the training runtime enables it, on its
    rollout environment; see "Observation noise" in
    `docs/reference/configuration/environment.md`.
    """

    dof_roles: dict[str, Any] = field(default_factory=dict)
    """Passive coordinates the model structure cannot decide.

    ``{passive_bodies: [...], passive_joints: [...]}``, read by
    `instantmimic.envs.dof_roles`. Extra kinematic trees and flexcomp vertex
    bodies are passive without a declaration. Passive coordinates keep their free
    dynamics but are not actuated, observed, tracked or rewarded.
    """

    use_cuda_graph: bool = False
    """Whether to capture the step function as a CUDA graph for reduced launch overhead.
    Requires all step operations to use pre-allocated buffers with stable pointers."""

    preserve_sim_visual_assets: bool = False
    """Keep rendering-only assets in a non-rendering simulation model.

    Normal training conservatively removes only assets proven to have no native
    collision, mass, MJCF reference, or declared out-of-engine dependency. Geom
    visibility groups are not used as a physics criterion. The training runtime
    enables this override only when in-process full-rollout video renders from
    the simulation model; regular playback uses ``render_mode='rgb_array'`` and
    keeps visual assets.
    """

    terminate_on_nonfinite: bool = False
    """Whether to terminate envs immediately when state or reward becomes non-finite.

    When enabled, the base environment checks MuJoCo state tensors (``qpos``/``qvel``)
    after physics and sanitizes any non-finite rewards before reset. Disabled by default
    so existing environments keep their current behavior and performance characteristics.
    """

    track_done_reasons: bool = False
    """Whether to accumulate per-reason termination statistics for trainer logging.

    When enabled, environments may register named termination masks and the base
    environment will accumulate counts on the GPU. Disabled by default to avoid
    extra bookkeeping on runs that do not need this diagnostics stream.
    """

    compile_mjw_step: bool = False
    """Whether to compile the selected repo-owned MuJoCo Warp step suffix.

    This is useful when full-step CUDA graph capture is disabled via :attr:`use_cuda_graph`
    but the allocation-free physics suffix still benefits from compilation.
    """

    use_parallelized_kernel_launch: bool = False
    """Run independent custom MuJoCo Warp branches on CUDA side streams.

    This controls scheduling only.  When disabled, the same allocation-free
    repo-owned forward/step implementation runs sequentially on the active stream.
    """

    use_optimized_kernel: bool = False
    """Enable eligible tree, indexed inertia, Newton, and implicit kernels.

    Each stage falls back by model structure; body-state reward parallelism is
    shared by both settings and is not controlled by this simulation option.

    This is independent of :attr:`use_parallelized_kernel_launch`; persistent
    scratch and allocation-free runtime orchestration are always enabled.
    """

    log_dir: str | None = None
    """Run directory the environment may write diagnostics into. `None` disables it."""
