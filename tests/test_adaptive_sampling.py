"""Independent reset-selector RNG draws for SNU adaptive sampling (CPU)."""

from __future__ import annotations

import pytest
import torch
import warp as wp

from instantmimic.envs.snu.tasks.snu_imitation import adaptive_sampling
from instantmimic.envs.snu.tasks.snu_imitation.adaptive_sampling import (
    RESET_DRAW_FRAME_OR_UNIT,
    RESET_DRAW_PRE_FAILURE_OFFSET,
    RESET_DRAW_RANDOM_SAMPLE,
    RESET_DRAW_SLOT_COUNT,
    RESET_DRAW_WITHIN_CHUNK,
    reset_from_motion_multi_select_kernel,
    reset_from_motion_select_kernel,
)

wp.init()


def _warp_array(tensor: torch.Tensor):
    return wp.from_torch(tensor)


def _reset_sampling_settings(
    *,
    random_init: int,
    start_frame_idx: int,
    adaptive_enabled: int,
    adaptive_chunk_size: int,
    adaptive_unit_count: int,
    pre_failure_window_steps: int,
    total_motion_frame_count: int,
    start_horizon_steps: int,
    fixed_motion_idx: int,
):
    """The packed settings array `reset_from_motion_multi_select_kernel` reads.

    The environment stages this once at construction; here it is built per
    launch so each case can state its own settings by name.
    """

    values = [0] * adaptive_sampling.RESET_SETTING_COUNT
    values[adaptive_sampling.RESET_SETTING_RANDOM_INIT] = random_init
    values[adaptive_sampling.RESET_SETTING_START_FRAME_IDX] = start_frame_idx
    values[adaptive_sampling.RESET_SETTING_ADAPTIVE_ENABLED] = adaptive_enabled
    values[adaptive_sampling.RESET_SETTING_ADAPTIVE_CHUNK_SIZE] = adaptive_chunk_size
    values[adaptive_sampling.RESET_SETTING_ADAPTIVE_UNIT_COUNT] = adaptive_unit_count
    values[adaptive_sampling.RESET_SETTING_PRE_FAILURE_WINDOW_STEPS] = (
        pre_failure_window_steps
    )
    values[adaptive_sampling.RESET_SETTING_TOTAL_MOTION_FRAME_COUNT] = (
        total_motion_frame_count
    )
    values[adaptive_sampling.RESET_SETTING_START_HORIZON_STEPS] = start_horizon_steps
    values[adaptive_sampling.RESET_SETTING_FIXED_MOTION_IDX] = fixed_motion_idx
    return _warp_array(torch.tensor(values, dtype=torch.int32))


@wp.kernel(enable_backward=False)
def _reset_stream_draws_kernel(seed: int, out: wp.array2d(dtype=float)):
    env_idx = wp.tid()
    random_sample = wp.rand_init(
        seed,
        env_idx * RESET_DRAW_SLOT_COUNT + RESET_DRAW_RANDOM_SAMPLE,
    )
    frame_or_unit = wp.rand_init(
        seed,
        env_idx * RESET_DRAW_SLOT_COUNT + RESET_DRAW_FRAME_OR_UNIT,
    )
    within_chunk = wp.rand_init(
        seed,
        env_idx * RESET_DRAW_SLOT_COUNT + RESET_DRAW_WITHIN_CHUNK,
    )
    pre_failure = wp.rand_init(
        seed,
        env_idx * RESET_DRAW_SLOT_COUNT + RESET_DRAW_PRE_FAILURE_OFFSET,
    )
    out[env_idx, RESET_DRAW_RANDOM_SAMPLE] = wp.randf(random_sample)
    out[env_idx, RESET_DRAW_FRAME_OR_UNIT] = wp.randf(frame_or_unit)
    out[env_idx, RESET_DRAW_WITHIN_CHUNK] = wp.randf(within_chunk)
    out[env_idx, RESET_DRAW_PRE_FAILURE_OFFSET] = wp.randf(pre_failure)


@wp.kernel(enable_backward=False)
def _old_reset_stream_draws_kernel(seed: int, out: wp.array2d(dtype=float)):
    """The pre-fix state: all selector draws shared rand_init(seed, env)."""
    env_idx = wp.tid()
    state = wp.rand_init(seed, env_idx)
    for draw in range(3):
        out[env_idx, draw] = wp.randf(state)


def _reset_stream_draws(seed: int, num_envs: int) -> torch.Tensor:
    draws = torch.zeros(
        (num_envs, RESET_DRAW_SLOT_COUNT),
        dtype=torch.float32,
    )
    wp.launch(
        _reset_stream_draws_kernel,
        dim=(num_envs,),
        device="cpu",
        inputs=[seed],
        outputs=[_warp_array(draws)],
    )
    return draws


def _old_reset_stream_draws(seed: int, num_envs: int) -> torch.Tensor:
    draws = torch.zeros((num_envs, 3), dtype=torch.float32)
    wp.launch(
        _old_reset_stream_draws_kernel,
        dim=(num_envs,),
        device="cpu",
        inputs=[seed],
        outputs=[_warp_array(draws)],
    )
    return draws


def _first_cdf_index(value: float, cdf: torch.Tensor) -> int:
    lo = 0
    hi = int(cdf.numel()) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if value <= float(cdf[mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _int_draw(value: float, count: int) -> int:
    """Warp's ``int(randf * float(count))`` in float32."""
    return int((torch.tensor(value, dtype=torch.float32) * float(count)).item())


def _run_single_reset_select(
    *,
    num_envs: int,
    reset_seed: int,
    step_counter: int,
    adaptive_cdf: torch.Tensor,
    adaptive_random_sample_prob: float,
    adaptive_pre_failure_window_steps: int,
    motion_frame_count: int,
):
    ref_frame_idx = torch.full((num_envs,), -1, dtype=torch.int32)
    episode_start_unit_idx = torch.full((num_envs,), -1, dtype=torch.int32)
    early_termination_timer = torch.full((num_envs,), -1, dtype=torch.int32)
    reset_motion_frame_idx = torch.full((num_envs,), -1, dtype=torch.int32)
    wp.launch(
        reset_from_motion_select_kernel,
        dim=(num_envs,),
        device="cpu",
        inputs=[
            _warp_array(torch.ones(num_envs, dtype=torch.bool)),
            _warp_array(torch.tensor([reset_seed], dtype=torch.int32)),
            _warp_array(torch.tensor([step_counter], dtype=torch.int32)),
            1,
            -1,
            1,
            adaptive_random_sample_prob,
            1,
            int(adaptive_cdf.numel()),
            _warp_array(adaptive_cdf),
            adaptive_pre_failure_window_steps,
            motion_frame_count,
            1,
            _warp_array(torch.zeros(num_envs, dtype=torch.bool)),
            _warp_array(torch.zeros(num_envs, dtype=torch.int32)),
        ],
        outputs=[
            _warp_array(ref_frame_idx),
            _warp_array(episode_start_unit_idx),
            _warp_array(early_termination_timer),
            _warp_array(reset_motion_frame_idx),
        ],
    )
    return (
        ref_frame_idx,
        episode_start_unit_idx,
        early_termination_timer,
        (reset_motion_frame_idx),
    )


def _run_multi_reset_select(
    *,
    num_envs: int,
    reset_seed: int,
    step_counter: int,
    adaptive_enabled: int,
    adaptive_random_sample_prob: float,
    adaptive_cdf: torch.Tensor,
    adaptive_chunk_size: int,
    adaptive_unit_count: int,
    adaptive_pre_failure_window_steps: int,
    clip_length: int,
    total_motion_frame_count: int,
):
    clip_start = torch.tensor([0], dtype=torch.int32)
    clip_len = torch.tensor([clip_length], dtype=torch.int32)
    clip_unit_start = torch.tensor([0], dtype=torch.int32)
    clip_unit_count = torch.tensor([max(adaptive_unit_count, 1)], dtype=torch.int32)
    frame_to_clip_idx = torch.zeros(total_motion_frame_count, dtype=torch.int32)
    motion_clip_idx = torch.full((num_envs,), -1, dtype=torch.int32)
    ref_frame_idx = torch.full((num_envs,), -1, dtype=torch.int32)
    episode_start_unit_idx = torch.full((num_envs,), -1, dtype=torch.int32)
    early_termination_timer = torch.full((num_envs,), -1, dtype=torch.int32)
    reset_motion_frame_idx = torch.full((num_envs,), -1, dtype=torch.int32)
    wp.launch(
        reset_from_motion_multi_select_kernel,
        dim=(num_envs,),
        device="cpu",
        inputs=[
            _warp_array(torch.ones(num_envs, dtype=torch.bool)),
            _warp_array(torch.tensor([reset_seed], dtype=torch.int32)),
            _warp_array(torch.tensor([step_counter], dtype=torch.int32)),
            _reset_sampling_settings(
                random_init=1,
                start_frame_idx=-1,
                adaptive_enabled=adaptive_enabled,
                adaptive_chunk_size=adaptive_chunk_size,
                adaptive_unit_count=adaptive_unit_count,
                pre_failure_window_steps=adaptive_pre_failure_window_steps,
                total_motion_frame_count=total_motion_frame_count,
                start_horizon_steps=1,
                fixed_motion_idx=-1,
            ),
            adaptive_random_sample_prob,
            _warp_array(adaptive_cdf),
            _warp_array(torch.zeros(num_envs, dtype=torch.bool)),
            _warp_array(torch.zeros(num_envs, dtype=torch.int32)),
            _warp_array(torch.zeros(num_envs, dtype=torch.int32)),
            _warp_array(clip_start),
            _warp_array(clip_len),
            _warp_array(frame_to_clip_idx),
            _warp_array(clip_unit_start),
            _warp_array(clip_unit_count),
        ],
        outputs=[
            _warp_array(motion_clip_idx),
            _warp_array(ref_frame_idx),
            _warp_array(episode_start_unit_idx),
            _warp_array(early_termination_timer),
            _warp_array(reset_motion_frame_idx),
        ],
    )
    return (
        motion_clip_idx,
        ref_frame_idx,
        episode_start_unit_idx,
        early_termination_timer,
        reset_motion_frame_idx,
    )


@pytest.mark.parametrize("adaptive_random_sample_prob", [0.0, 0.25])
def test_single_motion_reset_selector_draws_are_independent(
    adaptive_random_sample_prob: float,
) -> None:
    num_envs = 64
    reset_seed = 123
    step_counter = 7
    unit_count = 17
    motion_frame_count = 17
    window = 5
    adaptive_cdf = torch.linspace(
        1.0 / float(unit_count),
        1.0,
        unit_count,
        dtype=torch.float32,
    )

    ref, unit, timer, frame = _run_single_reset_select(
        num_envs=num_envs,
        reset_seed=reset_seed,
        step_counter=step_counter,
        adaptive_cdf=adaptive_cdf,
        adaptive_random_sample_prob=adaptive_random_sample_prob,
        adaptive_pre_failure_window_steps=window,
        motion_frame_count=motion_frame_count,
    )

    stream_seed = reset_seed + step_counter
    draws = _reset_stream_draws(stream_seed, num_envs)
    old_draws = _old_reset_stream_draws(stream_seed, num_envs)
    expected = []
    old_expected = []
    for env_idx in range(num_envs):
        decision = float(draws[env_idx, RESET_DRAW_RANDOM_SAMPLE])
        use_uniform = adaptive_random_sample_prob > 0.0 and (
            decision < adaptive_random_sample_prob
        )
        selection = float(draws[env_idx, RESET_DRAW_FRAME_OR_UNIT])
        if use_uniform:
            frame_idx = _int_draw(selection, motion_frame_count)
        else:
            frame_idx = min(
                _first_cdf_index(selection, adaptive_cdf),
                motion_frame_count - 1,
            )
        offset = _int_draw(float(draws[env_idx, RESET_DRAW_PRE_FAILURE_OFFSET]), window)
        expected.append(max(0, frame_idx - offset))

        old_use_uniform = adaptive_random_sample_prob > 0.0 and (
            float(old_draws[env_idx, 0]) < adaptive_random_sample_prob
        )
        old_selection_index = 1 if adaptive_random_sample_prob > 0.0 else 0
        old_selection = float(old_draws[env_idx, old_selection_index])
        if old_use_uniform:
            old_frame_idx = _int_draw(old_selection, motion_frame_count)
        else:
            old_frame_idx = min(
                _first_cdf_index(old_selection, adaptive_cdf),
                motion_frame_count - 1,
            )
        old_offset = _int_draw(float(old_draws[env_idx, 0]), window)
        old_expected.append(max(0, old_frame_idx - old_offset))

    expected_t = torch.tensor(expected, dtype=torch.int32)
    old_expected_t = torch.tensor(old_expected, dtype=torch.int32)
    torch.testing.assert_close(ref, expected_t)
    torch.testing.assert_close(frame, expected_t)
    torch.testing.assert_close(unit, expected_t)
    torch.testing.assert_close(timer, torch.zeros_like(expected_t))
    assert int((expected_t != old_expected_t).sum()) > 0


@pytest.mark.parametrize(
    ("adaptive_enabled", "adaptive_unit_count", "adaptive_random_sample_prob"),
    [(1, 13, 0.0), (1, 13, 0.25), (1, 13, 1.0), (1, 0, 1.0)],
)
def test_multi_motion_reset_selector_draws_are_independent(
    adaptive_enabled: int,
    adaptive_unit_count: int,
    adaptive_random_sample_prob: float,
) -> None:
    num_envs = 64
    reset_seed = 321
    step_counter = 9
    chunk_size = 8
    clip_length = 101
    window = 7
    adaptive_cdf = (
        torch.linspace(
            1.0 / float(adaptive_unit_count),
            1.0,
            adaptive_unit_count,
            dtype=torch.float32,
        )
        if adaptive_unit_count > 0
        else torch.ones(1, dtype=torch.float32)
    )

    clip_idx, ref, unit, timer, frame = _run_multi_reset_select(
        num_envs=num_envs,
        reset_seed=reset_seed,
        step_counter=step_counter,
        adaptive_enabled=adaptive_enabled,
        adaptive_random_sample_prob=adaptive_random_sample_prob,
        adaptive_cdf=adaptive_cdf,
        adaptive_chunk_size=chunk_size,
        adaptive_unit_count=adaptive_unit_count,
        adaptive_pre_failure_window_steps=window,
        clip_length=clip_length,
        total_motion_frame_count=clip_length,
    )

    stream_seed = reset_seed + step_counter
    draws = _reset_stream_draws(stream_seed, num_envs)
    old_draws = _old_reset_stream_draws(stream_seed, num_envs)
    expected = []
    expected_units = []
    old_expected = []
    for env_idx in range(num_envs):
        use_uniform = adaptive_enabled == 0 or adaptive_unit_count <= 0
        decision_drawn = (
            adaptive_random_sample_prob > 0.0
            and adaptive_enabled == 1
            and adaptive_unit_count > 0
        )
        if decision_drawn:
            use_uniform = use_uniform or (
                float(draws[env_idx, RESET_DRAW_RANDOM_SAMPLE])
                < adaptive_random_sample_prob
            )
        selection = float(draws[env_idx, RESET_DRAW_FRAME_OR_UNIT])
        within = _int_draw(float(draws[env_idx, RESET_DRAW_WITHIN_CHUNK]), clip_length)
        if use_uniform and adaptive_unit_count > 0:
            global_unit = min(
                _int_draw(selection, adaptive_unit_count),
                adaptive_unit_count - 1,
            )
            start = global_unit * chunk_size
            frame_idx = start + _int_draw(
                float(draws[env_idx, RESET_DRAW_WITHIN_CHUNK]),
                min(chunk_size, clip_length - start),
            )
        elif use_uniform:
            frame_idx = within
        else:
            global_unit = _first_cdf_index(selection, adaptive_cdf)
            start = global_unit * chunk_size
            frame_idx = start + _int_draw(
                float(draws[env_idx, RESET_DRAW_WITHIN_CHUNK]),
                min(chunk_size, clip_length - start),
            )
        offset = (
            _int_draw(float(draws[env_idx, RESET_DRAW_PRE_FAILURE_OFFSET]), window)
            if adaptive_enabled == 1
            else 0
        )
        expected_frame = max(0, frame_idx - offset)
        expected.append(expected_frame)
        expected_units.append(expected_frame // chunk_size)

        old_use_uniform = adaptive_enabled == 0 or adaptive_unit_count <= 0
        old_state_index = 1 if decision_drawn else 0
        if decision_drawn:
            old_use_uniform = old_use_uniform or (
                float(old_draws[env_idx, 0]) < adaptive_random_sample_prob
            )
        old_selection = float(old_draws[env_idx, old_state_index])
        if old_use_uniform:
            old_local_draw = float(old_draws[env_idx, old_state_index + 1])
            old_offset_draw = (
                old_local_draw
                if adaptive_unit_count > 0
                else float(old_draws[env_idx, old_state_index + 2])
            )
        else:
            # sample_adaptive_unit and the clip helper each received the same
            # by-value state, so both repeated its first draw; the kernel's
            # offset draw did too.
            old_local_draw = old_selection
            old_offset_draw = old_selection
        if old_use_uniform and adaptive_unit_count > 0:
            old_global_unit = min(
                _int_draw(old_selection, adaptive_unit_count),
                adaptive_unit_count - 1,
            )
            old_start = old_global_unit * chunk_size
            old_frame_idx = old_start + _int_draw(
                old_local_draw,
                min(chunk_size, clip_length - old_start),
            )
        elif old_use_uniform:
            old_frame_idx = _int_draw(old_local_draw, clip_length)
        else:
            old_global_unit = _first_cdf_index(old_selection, adaptive_cdf)
            old_start = old_global_unit * chunk_size
            old_frame_idx = old_start + _int_draw(
                old_local_draw,
                min(chunk_size, clip_length - old_start),
            )
        old_offset = _int_draw(old_offset_draw, window) if adaptive_enabled == 1 else 0
        old_expected.append(max(0, old_frame_idx - old_offset))

    expected_t = torch.tensor(expected, dtype=torch.int32)
    expected_units_t = torch.tensor(expected_units, dtype=torch.int32)
    old_expected_t = torch.tensor(old_expected, dtype=torch.int32)
    torch.testing.assert_close(clip_idx, torch.zeros_like(expected_t))
    torch.testing.assert_close(ref, expected_t)
    torch.testing.assert_close(frame, expected_t)
    torch.testing.assert_close(unit, expected_units_t)
    torch.testing.assert_close(timer, torch.zeros_like(expected_t))
    assert int((expected_t != old_expected_t).sum()) > 0


def test_single_motion_reset_selector_draw_independence_property() -> None:
    """The frame and offset draws must not be the same uniform value.

    This test deliberately derives its expectation from the output distribution
    rather than the production slot numbering, so aliasing two slots cannot
    make it pass. With 64 equally likely frames and a 64-frame offset window,
    independent draws leave about half the frames positive; a reused frame draw
    always subtracts itself and clamps every result to zero.
    """
    num_envs = 4096
    frame_count = 64
    adaptive_cdf = torch.linspace(
        1.0 / float(frame_count),
        1.0,
        frame_count,
        dtype=torch.float32,
    )
    ref, _, _, frame = _run_single_reset_select(
        num_envs=num_envs,
        reset_seed=999,
        step_counter=11,
        adaptive_cdf=adaptive_cdf,
        adaptive_random_sample_prob=0.0,
        adaptive_pre_failure_window_steps=frame_count,
        motion_frame_count=frame_count,
    )
    positive_fraction = float((ref > 0).float().mean())
    assert 0.35 < positive_fraction < 0.65, positive_fraction
    torch.testing.assert_close(frame, ref)


def test_multi_motion_reset_selector_draw_independence_property() -> None:
    """Unit selection and within-unit selection must be independent draws.

    With eight equally likely units and eight frames per unit, independent
    draws put the unit index and the within-unit index in the same place about
    one eighth of the time. The by-value helper reuse made them identical for
    every environment. No production slot constant is used here.
    """
    num_envs = 4096
    unit_count = 8
    chunk_size = 8
    clip_length = unit_count * chunk_size
    adaptive_cdf = torch.linspace(
        1.0 / float(unit_count),
        1.0,
        unit_count,
        dtype=torch.float32,
    )
    _, ref, unit, _, _ = _run_multi_reset_select(
        num_envs=num_envs,
        reset_seed=1001,
        step_counter=3,
        adaptive_enabled=1,
        adaptive_random_sample_prob=0.0,
        adaptive_cdf=adaptive_cdf,
        adaptive_chunk_size=chunk_size,
        adaptive_unit_count=unit_count,
        adaptive_pre_failure_window_steps=0,
        clip_length=clip_length,
        total_motion_frame_count=clip_length,
    )
    within_unit = ref - unit * chunk_size
    same_fraction = float((within_unit == unit).float().mean())
    assert 0.05 < same_fraction < 0.25, same_fraction
