"""Behavior of the environment boundary with the MuJoCo Warp runtime."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
import warp as wp

from instantmimic.envs.direct_rl_env import DirectRLEnv, external_step_capture_window
from instantmimic.envs.direct_rl_env_cfg import (
    DirectRLEnvCfg,
    InteractiveSceneCfg,
    SimulationCfg,
    ViewerCfg,
)


class _PendulumEnv(DirectRLEnv):
    def __init__(self, cfg):
        self.events = []
        super().__init__(cfg)
        self.observations = {"policy": torch.empty(self.num_envs, 2)}

    def _pre_physics_step(self, actions):
        self.actions.copy_(actions)

    def _apply_action(self):
        self.events.append("action")
        wp.to_torch(self.mjw_data.ctrl).copy_(self.actions)

    def _post_physics_substep(self, substep_idx):
        self.events.append(f"post:{substep_idx}")

    def _post_state_prefix_substep(self, substep_idx):
        self.events.append(f"endpoint:{substep_idx}")

    def _get_observations(self):
        self.observations["policy"][:, :1].copy_(wp.to_torch(self.mjw_data.qpos))
        self.observations["policy"][:, 1:].copy_(wp.to_torch(self.mjw_data.qvel))
        return self.observations

    def _get_rewards(self):
        self.reward_buf.copy_(wp.to_torch(self.mjw_data.qvel)[:, 0])
        return self.reward_buf

    def _get_dones(self):
        self.reset_terminated.zero_()
        self.reset_time_outs.zero_()
        return self.reset_terminated, self.reset_time_outs


def _config(tmp_path, integrator):
    path = tmp_path / "pendulum.xml"
    path.write_text("""<mujoco><option gravity="0 0 -9.81"/>
      <worldbody><body><joint name="hinge" type="hinge" axis="0 1 0" damping=".1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size=".05" mass="1"/>
      </body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>""")
    return DirectRLEnvCfg(
        sim=SimulationCfg(
            mjcf_path=str(path),
            integrator=integrator,
            render_interval=2,
            njmax=16,
            nconmax=4,
            hz=120,
        ),
        scene=InteractiveSceneCfg(num_envs=2),
        viewer=ViewerCfg(resolution=(8, 8)),
        decimation=2,
        episode_length_s=2,
        observation_space=2,
        action_space=1,
        seed=7,
        use_cuda_graph=False,
    )


@pytest.mark.parametrize("integrator", ["implicitfast", "euler", "rk4"])
def test_full_environment_step_matches_native_mujoco(tmp_path, integrator):
    """Real control, physics, endpoint refresh, rewards and observation boundary."""
    with wp.ScopedDevice("cpu"):
        env = _PendulumEnv(_config(tmp_path, integrator))
        try:
            env.reset()
            reference = [mujoco.MjData(env.model) for _ in range(2)]
            actions = torch.tensor([[0.8], [-0.4]])
            for _ in range(3):
                observations, rewards, terminated, truncated, _ = env.step(actions)
                for world, data in enumerate(reference):
                    data.ctrl[:] = actions[world].numpy()
                    for _ in range(2):
                        mujoco.mj_step(env.model, data)
                        mujoco.mj_forward(env.model, data)
                    np.testing.assert_allclose(
                        observations["policy"][world].numpy(),
                        [data.qpos[0], data.qvel[0]],
                        atol=2e-6,
                        rtol=2e-5,
                    )
                    assert float(rewards[world]) == pytest.approx(
                        data.qvel[0], abs=2e-6
                    )
                assert not terminated.any() and not truncated.any()
            assert (
                env.events
                == ["action", "post:0", "endpoint:0", "action", "post:1", "endpoint:1"]
                * 3
            )
            assert torch.equal(env.episode_length_buf, torch.tensor([3, 3]))
            assert not torch.cuda.is_initialized()
        finally:
            env.close()


def test_capture_window_preserves_lifecycle_and_primary_failure():
    events = []
    env = SimpleNamespace(
        prepare_step_capture_window=lambda: events.append("prepare"),
        finalize_step_capture_window=lambda: events.append("finalize"),
    )
    with external_step_capture_window(env):
        events.append("body")
    assert events == ["prepare", "body", "finalize"]
    with pytest.raises(ValueError, match="capture failed"):
        with external_step_capture_window(env):
            raise ValueError("capture failed")
    assert events[-2:] == ["prepare", "finalize"]


def test_reset_finalizer_registration_cannot_change_captured_lifecycle(tmp_path):
    with wp.ScopedDevice("cpu"):
        env = _PendulumEnv(_config(tmp_path, "implicitfast"))
        try:
            env._step_lifecycle_registration_frozen = True
            with pytest.raises(RuntimeError, match="before graph capture"):
                env.register_reset_state_finalizer(lambda mask: None)
        finally:
            env.close()
