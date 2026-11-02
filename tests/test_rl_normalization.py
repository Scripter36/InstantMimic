import unittest

import pytest
import torch

from instantmimic.rl.normalization import (
    ObservationNormalizationContract,
    RslRlEmpiricalNormalization,
    RunningStandardScaler,
    adapt_observation_scaler_state,
    build_observation_preprocessor,
    resolve_observation_preprocessor_name,
)


class RunningStandardScalerTest(unittest.TestCase):
    def test_single_sample_update_uses_finite_population_variance(self) -> None:
        scaler = RunningStandardScaler(1, device="cpu")

        scaler(torch.tensor([[2.0]]), train=True)

        torch.testing.assert_close(
            scaler.running_mean,
            torch.tensor([1.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            scaler.running_variance,
            torch.tensor([1.5], dtype=torch.float64),
        )
        self.assertTrue(torch.isfinite(scaler.running_variance).all())


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (None, "identity"),
        ("none", "identity"),
        ("null", "identity"),
        ("Identity", "identity"),
        ("off", "identity"),
        ("disabled", "identity"),
        ("RunningStandardScaler", "running_standard"),
        ("running_standard_scaler", "running_standard"),
        ("running-standard-scaler", "running_standard"),
        ("  RUNNING_STANDARD  ", "running_standard"),
        ("RslRlEmpiricalNormalization", "rsl_rl_empirical"),
        ("rsl_rl_empirical_normalization", "rsl_rl_empirical"),
    ],
)
def test_observation_preprocessor_name_resolves_each_accepted_spelling(
    name: str | None, expected: str
) -> None:
    assert resolve_observation_preprocessor_name(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "runningstandard",
        "rsl_rl",
        "standard",
        "identtiy",
        # PPO never accepted its own canonical output as input; preserved.
        "rsl_rl_empirical",
    ],
)
def test_observation_preprocessor_name_rejects_anything_else(name: str) -> None:
    """A near miss has to raise, not quietly mean "no normalization"."""

    with pytest.raises(ValueError, match="Unsupported preprocessor"):
        resolve_observation_preprocessor_name(name)


@pytest.mark.parametrize(
    ("name", "expected_type", "expected_enabled"),
    [
        ("none", RunningStandardScaler, False),
        ("RunningStandardScaler", RunningStandardScaler, True),
        ("RslRlEmpiricalNormalization", RslRlEmpiricalNormalization, True),
    ],
)
def test_build_observation_preprocessor_maps_each_name_to_its_class(
    name: str, expected_type: type, expected_enabled: bool
) -> None:
    preprocessor = build_observation_preprocessor(
        preprocessor=name,
        preprocessor_kwargs=None,
        size=4,
        device=torch.device("cpu"),
    )

    assert type(preprocessor) is expected_type
    assert bool(preprocessor.enabled) is expected_enabled
    assert tuple(preprocessor.running_mean.shape) == (4,)


@pytest.mark.parametrize(
    ("name", "statistics_update", "policy_input_replay"),
    [
        ("none", "disabled", "recompute"),
        ("RunningStandardScaler", "rollout_batch", "recompute"),
        ("RslRlEmpiricalNormalization", "rollout_step", "snapshot"),
    ],
)
def test_observation_preprocessor_declares_coordinate_lifecycle(
    name: str,
    statistics_update: str,
    policy_input_replay: str,
) -> None:
    preprocessor = build_observation_preprocessor(
        preprocessor=name,
        preprocessor_kwargs=None,
        size=4,
        device=torch.device("cpu"),
    )

    assert preprocessor.contract == ObservationNormalizationContract(
        statistics_update=statistics_update,
        policy_input_replay=policy_input_replay,
    )


@pytest.mark.parametrize(
    ("name", "step_updates", "batch_updates"),
    [
        ("none", False, False),
        ("RunningStandardScaler", False, True),
        ("RslRlEmpiricalNormalization", True, False),
    ],
)
def test_observation_preprocessor_updates_only_at_declared_lifecycle_boundary(
    name: str,
    step_updates: bool,
    batch_updates: bool,
) -> None:
    preprocessor = build_observation_preprocessor(
        preprocessor=name,
        preprocessor_kwargs=None,
        size=4,
        device=torch.device("cpu"),
    )
    observations = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    initial_count = preprocessor.current_count.clone()

    assert preprocessor.observe_rollout_step(observations) is step_updates
    count_after_step = preprocessor.current_count.clone()
    assert bool(count_after_step > initial_count) is step_updates

    assert preprocessor.observe_rollout_batch(observations) is batch_updates
    count_after_batch = preprocessor.current_count.clone()
    assert bool(count_after_batch > count_after_step) is batch_updates


@pytest.mark.parametrize(
    ("statistics_update", "policy_input_replay", "message"),
    [
        ("per_episode", "recompute", "statistics update cadence"),
        ("disabled", "renormalize", "policy input replay mode"),
        ("rollout_step", "recompute", "requires policy input snapshots"),
    ],
)
def test_observation_normalization_contract_rejects_unsafe_combinations(
    statistics_update: str,
    policy_input_replay: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ObservationNormalizationContract(
            statistics_update=statistics_update,
            policy_input_replay=policy_input_replay,
        )


def test_both_preprocessors_normalize_the_same_way_through_out() -> None:
    """The builder hands either class to one call site, so `out=` must agree.

    Graph-captured normalization can write to a non-contiguous view such as
    `out_scaled[..., :obs_main_dim]`. Values are checked, not just finiteness:
    with unit statistics both classes must leave the observation alone.
    """

    obs = torch.full((2, 6), 3.0)
    for name in ("RunningStandardScaler", "RslRlEmpiricalNormalization"):
        preprocessor = build_observation_preprocessor(
            preprocessor=name,
            preprocessor_kwargs=None,
            size=4,
            device=torch.device("cpu"),
        )
        destination = torch.full((2, 6), -1.0)
        view = destination[..., :4]
        assert not view.is_contiguous()

        preprocessor(obs[..., :4], train=False, no_grad=True, out=view)

        # running_mean 0 and running_std/variance 1 leave the input unchanged,
        # up to each class's epsilon.
        torch.testing.assert_close(
            destination[..., :4], torch.full((2, 4), 3.0), rtol=1e-2, atol=1e-2
        )
        # The columns past the view must not be touched.
        torch.testing.assert_close(destination[..., 4:], torch.full((2, 2), -1.0))


def test_no_grad_is_honoured_by_only_one_preprocessor() -> None:
    """Pin the divergence rather than let a caller discover it.

    `RslRlEmpiricalNormalization.forward` accepts `no_grad` and discards it. The
    builder can hand a caller either class for one config name, so a call site
    that relies on `no_grad=True` to detach is relying on something the name
    does not guarantee. If this ever becomes uniform, this test should be the
    thing that fails.
    """

    obs = torch.full((2, 4), 3.0, requires_grad=True)
    grad_required = {}
    for name in ("RunningStandardScaler", "RslRlEmpiricalNormalization"):
        preprocessor = build_observation_preprocessor(
            preprocessor=name,
            preprocessor_kwargs=None,
            size=4,
            device=torch.device("cpu"),
        )

        grad_required[name] = preprocessor(obs, train=False, no_grad=True).requires_grad

    assert grad_required["RunningStandardScaler"] is False
    assert grad_required["RslRlEmpiricalNormalization"] is True


@pytest.mark.parametrize(
    "name", ["RunningStandardScaler", "RslRlEmpiricalNormalization"]
)
def test_widening_a_checkpoint_carries_every_per_feature_buffer(name: str) -> None:
    """Naming two buffers silently mis-scaled `RslRlEmpiricalNormalization`.

    `train/checkpointing.py` can widen a narrower
    checkpoint into the current scaler. They used to copy `running_mean` and
    `running_variance` only, so the `running_std` that actually divides the
    observation stayed at its freshly-initialized 1.0 while variance held the
    trained value -- widths matched, `load_state_dict` succeeded, and the policy
    saw observations roughly ten times too large with no error.
    """

    checkpoint = build_observation_preprocessor(
        preprocessor=name,
        preprocessor_kwargs=None,
        size=4,
        device=torch.device("cpu"),
    )
    checkpoint.update_stats(torch.randn(4096, 4) * 10.0)

    current = build_observation_preprocessor(
        preprocessor=name,
        preprocessor_kwargs=None,
        size=6,
        device=torch.device("cpu"),
    )

    def transfer(destination: torch.Tensor, source: torch.Tensor) -> None:
        destination[: source.shape[0]].copy_(
            source.to(device=destination.device, dtype=destination.dtype)
        )

    current.load_state_dict(
        adapt_observation_scaler_state(
            checkpoint_state=checkpoint.state_dict(),
            current_state=current.state_dict(),
            transfer=transfer,
        )
    )

    carried = current(torch.full((1, 6), 10.0))[0]

    # The carried columns divide by the checkpoint's deviation of about ten, so
    # an input of ten normalizes to about one. Before the fix these read ~9.8.
    assert torch.all(carried[:4].abs() < 3.0), carried.tolist()
    # The columns the checkpoint never covered keep unit statistics.
    assert torch.all(carried[4:].abs() > 3.0), carried.tolist()


if __name__ == "__main__":
    unittest.main()
