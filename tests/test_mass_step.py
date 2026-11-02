"""Checkpoint and coordinate-layout contracts of the production MASS controller."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from instantmimic.envs.controls.fast_muscle import step_runtime
from instantmimic.envs.controls.fast_muscle.muscle_nn import MassMuscleNN
from instantmimic.envs.controls.fast_muscle.settings import FastMuscleControlSettings


class _Runtime(step_runtime.FastMuscleStepMixin):
    def __init__(self):
        self.settings = FastMuscleControlSettings.from_cfg(
            {
                "action": {
                    "type": "mass",
                    "space": 2,
                    "policy_output": {"type": "torque", "scale": "auto"},
                    "muscle_nn": {"hidden_dims": [4]},
                },
            }
        )
        self._uses_mass_coordinator = True
        self._musclenn_model = MassMuscleNN(3, 2, (4,), output_activation="sine")
        self._musclenn_optimizer = torch.optim.Adam(
            self._musclenn_model.parameters(), lr=0.001
        )
        self._musclenn_model(torch.ones(2, 3)).square().mean().backward()
        self._musclenn_optimizer.step()
        self._musclenn_model.eval()
        self._musclenn_shuffle_generator = torch.Generator().manual_seed(91)
        self._torque_action_scale = torch.ones(2)
        self.muscle_count, self.muscle_names = 2, ("first", "second")
        self.model = SimpleNamespace(nv=2)
        self._root_dof_count = 0
        self._musclenn_update_graph = self._musclenn_shuffle_graph = None

    def _mass_action_settings(self):
        return self.settings.action

    def _invalidate_musclenn_update_graphs(self):
        self._musclenn_update_graph = self._musclenn_shuffle_graph = None


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setattr(step_runtime, "_controller_type", lambda: _Runtime)
    return _Runtime()


@pytest.mark.parametrize("corruption", ["model", "optimizer", "rng", "identity"])
def test_failed_mass_restore_preserves_model_optimizer_and_shuffle_state(
    runtime, corruption
):
    saved = copy.deepcopy(runtime.get_checkpoint_state())
    malformed = copy.deepcopy(saved)
    for tensor in malformed["mass"]["model"].values():
        tensor.add_(0.25)
    if corruption == "model":
        malformed["mass"]["model"]["net.0.weight"] = torch.zeros(1)
    elif corruption == "optimizer":
        malformed["mass"]["optimizer"]["param_groups"][0]["params"] = []
    elif corruption == "rng":
        malformed["mass"]["shuffle_generator"] = torch.zeros(3, dtype=torch.uint8)
    else:
        malformed["mass"]["output_activation"] = "unsupported"
    with pytest.raises((ValueError, RuntimeError)):
        runtime.load_checkpoint_state(malformed)
    _equal(saved, runtime.get_checkpoint_state())


def test_policy_initialization_preserves_current_optimizer_and_shuffle_state(runtime):
    saved = copy.deepcopy(runtime.get_checkpoint_state())
    with torch.no_grad():
        for parameter in runtime._musclenn_model.parameters():
            parameter.add_(1)
    optimizer_before = copy.deepcopy(runtime._musclenn_optimizer.state_dict())
    rng_before = runtime._musclenn_shuffle_generator.get_state().clone()
    assert runtime.load_policy_init_state(saved)
    _equal(saved["mass"]["model"], runtime._musclenn_model.state_dict())
    _equal(optimizer_before, runtime._musclenn_optimizer.state_dict())
    _equal(rng_before, runtime._musclenn_shuffle_generator.get_state())


def test_passive_coordinate_widening_preserves_network_function():
    torch.manual_seed(37)
    old_weight, bias = (
        torch.randn(3, 4, dtype=torch.float64),
        torch.randn(3, dtype=torch.float64),
    )
    payload = {
        "runtime_identity": {"torque_action_scale": [2.0, 3.0]},
        "model": {
            "net.0.weight": old_weight,
            "net.0.bias": bias,
            "input_normalizer.mean": torch.arange(4, dtype=torch.float64),
            "input_normalizer.variance": torch.full((4,), 2.0, dtype=torch.float64),
        },
    }
    original = copy.deepcopy(payload)
    widened = step_runtime._widen_character_mass_payload(
        payload,
        blocks=(("moment_arms", 0, 2), ("target", 2, 5)),
        character=(0, 2),
        torque_dim=3,
    )
    assert widened["runtime_identity"]["torque_action_scale"] == [2.0, 0.0, 3.0]
    old_input = torch.randn(5, 4, dtype=torch.float64)
    new_input = torch.randn(5, 5, dtype=torch.float64)
    new_input[:, [0, 1, 2, 4]] = old_input
    expected = torch.nn.functional.linear(old_input, old_weight, bias)
    actual = torch.nn.functional.linear(
        new_input, widened["model"]["net.0.weight"], bias
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert widened["model"]["input_normalizer.mean"][3] == 0
    assert widened["model"]["input_normalizer.variance"][3] == 1
    _equal(payload, original)
