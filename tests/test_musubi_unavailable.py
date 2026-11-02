"""Unsupported requests fail before a factory can allocate an environment/device."""

from types import SimpleNamespace

import pytest

from instantmimic.config.config import load_semantic_config
from instantmimic.rl.build_specs import lower_agent_build_spec
from instantmimic.rl.factory import resolve_agent_class_spec
from instantmimic.rl.models.musubi_actor_critic import MusubiActorCritic
from instantmimic.rl.ppo.factory import build_ppo_agent_from_spec


@pytest.mark.parametrize("owner", ["action", "architecture"])
def test_unavailable_selectors_are_rejected_before_other_configuration(owner):
    selector = "musubi"
    values = {"schema_version": 1}
    if owner == "action":
        values["env"] = {"action": {"type": selector}}
    else:
        values["models"] = {"actor": {"architecture": selector}}
    for operation in (
        load_semantic_config,
        lower_agent_build_spec,
        resolve_agent_class_spec,
    ):
        with pytest.raises(NotImplementedError, match="patent"):
            operation(values)


def test_constructor_and_direct_factory_do_not_allocate():
    class DeviceMustNotBeUsed:
        def __str__(self):
            pytest.fail("An unavailable request reached device initialization")

    with pytest.raises(NotImplementedError, match="patent"):
        MusubiActorCritic(device=DeviceMustNotBeUsed())
    with pytest.raises(NotImplementedError, match="patent"):
        build_ppo_agent_from_spec(
            None,
            obs_dim=1,
            act_dim=1,
            num_envs=1,
            device=DeviceMustNotBeUsed(),
            network_cls=MusubiActorCritic,
        )
    request = SimpleNamespace(
        env=SimpleNamespace(action=SimpleNamespace(type="musubi"))
    )
    with pytest.raises(NotImplementedError, match="patent"):
        resolve_agent_class_spec(request)


@pytest.mark.parametrize("family", ["run", "imitation", "command"])
def test_restored_environment_rejects_musubi_before_base_initialization(
    family, monkeypatch
):
    from instantmimic.envs.direct_rl_env import DirectRLEnv
    from instantmimic.envs.freemusco.tasks.freemusco_command_env import (
        FreemuscoCommandEnv,
    )
    from instantmimic.envs.freemusco.tasks.freemusco_imitation_env import (
        FreemuscoImitationEnv,
    )
    from instantmimic.envs.freemusco.tasks.freemusco_run_env import FreemuscoRunEnv

    def must_not_initialize(*_args, **_kwargs):
        pytest.fail("Deferred method reached simulator/device initialization")

    monkeypatch.setattr(DirectRLEnv, "__init__", must_not_initialize)
    classes = {
        "run": FreemuscoRunEnv,
        "imitation": FreemuscoImitationEnv,
        "command": FreemuscoCommandEnv,
    }
    cfg = SimpleNamespace(action={"type": "musubi", "space": 7}, species="ostrich")
    with pytest.raises(NotImplementedError, match="patent"):
        classes[family](cfg)
