# Writing an environment

In InstantMimic, an environment is defined by a task: a subclass of
`DirectRLEnv`. The base class owns the MuJoCo Warp model and data, the order of
a step and of a reset, and CUDA graph capture. The task decides what an action
does, what the policy observes, the reward and when an episode ends.

The Ant locomotion task, which reproduces Isaac Lab's `Isaac-Ant-Direct-v0`, is a
short, complete example:

```bash
uv run train --config-name ant/isaac_ant_direct
```

Its code is in `src/instantmimic/envs/ant/tasks/`. Read the Warp-kernel path
(`use_native_task_kernels: true`). The torch path is a CPU reference.

## Files and registration

A task lives in its own package. Its config and MJCF model go in the shared
`config/` and `data/` trees:

```text
src/instantmimic/envs/my_robot/
├── __init__.py
└── tasks/
    ├── __init__.py          # registers the task id
    ├── my_task_env_cfg.py   # config dataclass
    ├── my_task_env.py       # the DirectRLEnv subclass
    └── my_task_kernels.py   # Warp kernels, if the task needs them
config/my_robot/my_task.yaml
data/my_robot/my_robot.xml
```

The `tasks/__init__.py` registers a Gymnasium id:

```python
import gymnasium as gym

gym.register(
    id="MyRobot-Task-v0",
    entry_point=f"{__name__}.my_task_env:MyTaskEnv",
    disable_env_checker=True,
)
```

Add the package to `BUILTIN_TASK_MODULES` in `src/instantmimic/envs/registration.py`,
which `train`, `play` and the other tools import. A task outside the repository
can skip this by naming its module in the id, for example
`task: my_project.envs:MyRobot-Task-v0`. Gymnasium then imports
`my_project.envs` before it looks the id up.

Use `config/ant/isaac_ant_direct.yaml` as a template for the config. Its `task`
key names the registered id, and its `env` block is passed to the constructor as
`cfg`. `train` does not instantiate the config dataclass, so the YAML must set
every field the task reads. `action_space` and `observation_space` are the
widths of the action vector and of the `policy` observation.

## Task methods

Every method works on all `num_envs` worlds at once. The base class allocates
`actions`, `reward_buf`, `reset_terminated`, `reset_time_outs`, `reset_buf` and
`episode_length_buf`, and the task allocates `obs_buf`. The example follows the
[CUDA graph rules](#cuda-graph-rules) and allocates nothing after `__init__`.

```python
import torch
import warp as wp

from instantmimic.envs import DirectRLEnv


class MyTaskEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Views of the simulator state, shape (num_envs, nq) and so on.
        self.qpos = wp.to_torch(self.mjw_data.qpos)
        self.qvel = wp.to_torch(self.mjw_data.qvel)
        self.ctrl = wp.to_torch(self.mjw_data.ctrl)
        # Every buffer the step uses is allocated here, once.
        self.default_qpos = self.qpos[0].clone()
        self.obs_buf = {
            "policy": torch.zeros(
                (self.num_envs, cfg.observation_space), device=self.device
            )
        }

    def _pre_physics_step(self, actions):
        self.actions.copy_(actions)

    def _apply_action(self):
        torch.mul(self.actions, self.cfg.action_scale, out=self.ctrl)

    def _get_dones(self):
        height = self.qpos[:, 2]
        torch.lt(height, self.cfg.min_height, out=self.reset_terminated)
        torch.ge(
            self.episode_length_buf,
            self.max_episode_length,
            out=self.reset_time_outs,
        )
        return self.reset_terminated, self.reset_time_outs

    def _get_rewards(self):
        self.reward_buf.copy_(self.qvel[:, 0])  # forward velocity
        return self.reward_buf

    def _get_observations(self):
        policy = self.obs_buf["policy"]
        torch.cat((self.qpos[:, 2:], self.qvel), dim=-1, out=policy)
        return self.obs_buf

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        mask = reset_mask.unsqueeze(-1)
        torch.where(mask, self.default_qpos, self.qpos, out=self.qpos)
        self.qvel.masked_fill_(mask, 0.0)
        self.ctrl.masked_fill_(mask, 0.0)
        self.actions.masked_fill_(mask, 0.0)
```

Every task implements these:

| Method | Job |
| --- | --- |
| `_pre_physics_step` | Store or convert the policy action |
| `_apply_action` | Write the action into the simulator: `ctrl`, or forces such as `qfrc_applied` |
| `_get_dones` | Fill `reset_terminated` and `reset_time_outs` |
| `_get_rewards` | Fill `reward_buf` |
| `_get_observations` | Fill `obs_buf` |
| `_reset_idx` | Put the selected worlds back at a start state |

These are optional:

| Method | Job |
| --- | --- |
| `_post_physics_substep` | Edit `qpos` or `qvel` after each physics step, or read contact forces |
| `_post_state_prefix_substep` | Read the refreshed state after each substep |
| `_post_dones_and_rewards` | End episodes on a condition that needs the reward |

## Step order

The simulator advances in physics substeps, and one policy step runs
`decimation` of them. Each substep has two halves. The **physics step** applies
the action and integrates the motion into a new `qpos` and `qvel`. The
**refresh** then runs MuJoCo's position and velocity stages (as in
`mj_fwdPosition` and `mj_fwdVelocity`) on that `qpos` and `qvel`, so that body
poses, contacts and sensors match them again. One call to `step()` runs the
following, top to bottom:

<div class="im-diagram">
<svg viewBox="0 0 680 600" role="img" aria-label="The order of one environment step">
<defs><marker id="im-arrow-step" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="im-arrowhead" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<line class="im-spine" x1="215" y1="25" x2="215" y2="525"/>
<rect class="im-hook" x="70" y="10" width="290" height="30" rx="6"/><text class="im-code" x="215" y="30" text-anchor="middle">_pre_physics_step(actions)</text><text class="im-note" x="376" y="30">store the action</text>
<rect class="im-hook" x="70" y="60" width="290" height="30" rx="6"/><text class="im-code" x="215" y="80" text-anchor="middle">_apply_action()</text><text class="im-note" x="376" y="80">write ctrl or forces</text>
<rect class="im-physics" x="70" y="100" width="290" height="30" rx="6"/><text x="215" y="120" text-anchor="middle">Physics step</text><text class="im-note" x="376" y="120">compute forces, integrate qpos, qvel</text>
<rect class="im-hook im-optional" x="70" y="140" width="290" height="30" rx="6"/><text class="im-code" x="215" y="160" text-anchor="middle">_post_physics_substep(i)</text><text class="im-note" x="376" y="160">new qpos, qvel with old poses, contacts</text>
<rect class="im-refresh" x="70" y="180" width="290" height="30" rx="6"/><text x="215" y="200" text-anchor="middle">Refresh</text><text class="im-note" x="376" y="200">recompute poses, contacts, sensors</text>
<rect class="im-hook im-optional" x="70" y="220" width="290" height="30" rx="6"/><text class="im-code" x="215" y="240" text-anchor="middle">_post_state_prefix_substep(i)</text><text class="im-note" x="376" y="240">read the refreshed state</text>
<path class="im-line" d="M 68 235 H 48 V 75 H 66" marker-end="url(#im-arrow-step)"/>
<text class="im-caption" transform="translate(38 155) rotate(-90)" text-anchor="middle">× decimation</text>
<rect class="im-base" x="70" y="270" width="290" height="30" rx="6"/><text x="215" y="290" text-anchor="middle">Advance episode counters</text><text class="im-note" x="376" y="290">episode_length_buf += 1</text>
<rect class="im-hook" x="70" y="310" width="290" height="30" rx="6"/><text class="im-code" x="215" y="330" text-anchor="middle">_get_dones()</text><text class="im-note" x="376" y="330">which episodes end, and why</text>
<rect class="im-hook" x="70" y="350" width="290" height="30" rx="6"/><text class="im-code" x="215" y="370" text-anchor="middle">_get_rewards()</text><text class="im-note" x="376" y="370">the reward for this step</text>
<rect class="im-hook im-optional" x="70" y="390" width="290" height="30" rx="6"/><text class="im-code" x="215" y="410" text-anchor="middle">_post_dones_and_rewards(...)</text><text class="im-note" x="376" y="410">end episodes based on the reward</text>
<rect class="im-hook im-optional" x="70" y="430" width="290" height="30" rx="6"/><text class="im-code" x="215" y="450" text-anchor="middle">_get_observations()</text><text class="im-note" x="376" y="450">terminal observation, if the agent asks</text>
<rect class="im-base" x="70" y="470" width="290" height="30" rx="6"/><text x="215" y="490" text-anchor="middle">Reset</text><text class="im-note" x="376" y="490">every step, masked (see Resets)</text>
<rect class="im-hook" x="70" y="510" width="290" height="30" rx="6"/><text class="im-code" x="215" y="530" text-anchor="middle">_get_observations()</text><text class="im-note" x="376" y="530">the next policy input</text>
<rect class="im-hook" x="70" y="566" width="22" height="14" rx="3"/><text class="im-caption" x="100" y="577">Task method</text>
<rect class="im-hook im-optional" x="190" y="566" width="22" height="14" rx="3"/><text class="im-caption" x="220" y="577">Optional or conditional</text>
<rect class="im-physics" x="370" y="566" width="14" height="14" rx="3"/><rect class="im-refresh" x="388" y="566" width="14" height="14" rx="3"/><rect class="im-base" x="406" y="566" width="14" height="14" rx="3"/><text class="im-caption" x="428" y="577">Done by DirectRLEnv</text>
</svg>
</div>

- The counters advance before `_get_dones`, so
  `episode_length_buf >= max_episode_length` ends an episode after exactly
  `max_episode_length` steps.
- `_get_dones` runs before `_get_rewards`, so the reward can read
  `reset_terminated`, for example to penalize a fall.
- By the time `_post_dones_and_rewards(nonfinite_state_mask)` runs, the dones
  are already combined into `reset_buf`. A task that ends more episodes there
  calls `self._combine_done_outputs(nonfinite_state_mask)` afterwards.
- With `agent.time_limit_bootstrap`, `_get_observations` also runs before the
  reset, so it must not advance any history of its own.

### Simulator state

The simulator's data falls into two groups, and each half of a substep updates
one of them:

- **State-derived quantities** depend only on `qpos` and `qvel`: body poses
  and velocities, contacts and constraint rows, tendon lengths, passive and bias
  forces, and position and velocity sensors. The refresh updates them.
- **Action-dependent quantities** also depend on `ctrl` and applied forces:
  `actuator_force`, `qfrc_actuator`, `qacc`, `efc_force`, `qfrc_constraint`, and
  the sensors evaluated at the acceleration stage (touch, force, torque,
  accelerometer and actuator force). The physics step computes them from the
  `qpos` and `qvel` it starts at, then integrates.

In the order above, this means:

- Every method sees state-derived quantities that match `qpos` and `qvel`,
  except `_post_physics_substep`. It runs between the physics step and the
  refresh, so it sees the new `qpos` and `qvel` with the poses and contacts of
  the substep's start.
- Every method sees action-dependent quantities from the start of the last
  physics step, one substep behind `qpos` and `qvel`. After a reset they are
  from before the reset. A method that needs them for the current state calls
  `self._run_mjw_forward()` first.
- Contact forces (`efc_force`, `mujoco_warp.contact_force`) are indexed by the
  constraint rows of the solve that produced them, and the refresh rebuilds
  those rows. Read them in `_post_physics_substep`, before the refresh.

## Resets

The reset runs at the end of every step, with `reset_buf` as the mask. When no
episode ended, the mask is all false. The base class calls, in order:

1. `_reset_idx(reset_mask)`: write the start state (`qpos`, `qvel`, `ctrl` and
   the task's own state) of the worlds where `reset_mask` is true. Call
   `super()._reset_idx(reset_mask)` first, which zeroes their episode length.
2. `_prepare_reset_state(reset_mask)`, optional: edits that need a few
   state-derived quantities, such as body positions, before the full refresh.
   The method computes what it reads itself, for example with
   `self._run_mjw_kinematics_com_state()`.
3. The refresh, for every world.
4. `_post_reset_state_prefix(reset_mask)`, optional: edits that need the fresh
   state, for example placing a target relative to a body. Return `True` to have
   the refresh run again.
5. `_post_finalize_reset_state(reset_mask)`, optional: bookkeeping after the
   last refresh.

Apply the mask with `torch.where`, `masked_fill_` or a Warp kernel, not with an
index list, whose length would change from step to step. `reset()` runs the same
sequence for every world before the first step.

!!! note "Skipping the last contact refresh"
    Since the reset refreshes every world anyway, a task class that sets
    `boundary_reads_constraints = False` lets the last substep's refresh skip
    contacts. Everything between that refresh and the reset then sees the
    previous substep's contacts, so set it only when the dones, rewards and
    terminal observation do not read contacts. Subclasses do not inherit it (see
    [Simulation](../reference/simulation.md#state-prefix-and-action-suffix)).

## CUDA graph rules

With CUDA graphs enabled, as in the default configs, the step is recorded once
and then replayed. Write every method as if its Python body ran only once, at
recording:

- **Allocate in `__init__`.** Inside a step, write into existing tensors with
  `copy_`, `out=`, in-place operations or Warp kernels. An expression such as
  `a + b` creates a new tensor. Keep it out of the step, or write it into a
  preallocated buffer with `torch.add(a, b, out=buf)`.
- **Do not read GPU values on the host.** `.item()`, `.cpu()`, `bool(tensor)`,
  `if tensor.any():`, `mask.nonzero()` and boolean indexing such as `x[mask]`
  wait for the GPU and fail during recording. Branch with masks and
  `torch.where` instead.
- **Python values are fixed when the step is recorded.** A config value, a
  Python counter or an `if self.flag:` is evaluated once. Anything that changes
  over training belongs in a GPU tensor, as `episode_length_buf` and
  `common_step_counter` do.
- **Keep tensors in place.** A recorded step holds the addresses of the tensors
  it uses, so update a tensor's contents instead of assigning a new one. Only
  the base class rebinds the [output buffers](#output-buffers).
- **Run the simulator through the base class.** Use `_run_mjw_step()` and
  `_run_mjw_forward()`, never `mujoco_warp.step()` or `mujoco_warp.forward()`,
  which allocate memory on every call (see
  [Simulation](../reference/simulation.md#runtime-contract)).
- **Define Warp kernels at module level.** A kernel defined inside a method is
  rebuilt on every call.

## Output buffers

The training loop has the environment write each step's outputs directly into
the rollout storage. Before a step runs, the base class points `obs_buf`,
`reward_buf`, `reset_terminated` and `reset_time_outs` at that step's slots, so
methods must write into whatever these attributes refer to at that moment
rather than into a reference kept from `__init__`. A task that holds Warp views
of these buffers rebuilds them in `_bind_step_outputs`, as `AntEnv` does.

`env.step_with_io(step_io, validate_buffers=True)` checks this contract: it
raises `BufferContractError` when a method returns or binds a different tensor.
It cannot see writes through a stale Warp view.

## Benchmark evaluation

`trainer.benchmark` asks the task when an evaluated episode counts and when it
succeeds, through `get_eval_episode_spec()`, which returns an `EvalEpisodeSpec`
(`instantmimic.envs.eval_metrics`). Without it, an episode counts and succeeds
when it runs to its time limit. The SNU imitation task (tracking a clip to its
end) and the command tasks (`validity="always"`) show the other cases.

## Testing

```bash
uv run train --config-name my_robot/my_task --cfg job --resolve
uv run train --config-name my_robot/my_task trainer.max_iterations=20
```

The first command prints the composed config, and the second is a short run.
Keep CUDA graphs and compilation on while testing, since a task that only works
with them off breaks the rules above. `tests/test_ant_direct.py` shows unit
tests for a task.
