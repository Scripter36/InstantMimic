from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from instantmimic.envs.freemusco.tasks.freemusco_imitation.mocap_clip import (
    MocapClip,
    MocapLibrary,
    rotmat_array_to_quat_array,
)


@pytest.fixture
def mocap_data_dirs(tmp_path: Path) -> tuple[Path, Path]:
    chimanoid_dir = tmp_path / "chimanoid"
    ostrich_dir = tmp_path / "ostrich"
    for motion_data_dir in (chimanoid_dir, ostrich_dir):
        for key in ("qpos", "qvel", "xipos", "ximat"):
            (motion_data_dir / key).mkdir(parents=True)
    return chimanoid_dir, ostrich_dir


def _write_clip(
    motion_data_dir: Path,
    clip_name: str,
    *,
    frame_count: int = 2,
) -> None:
    qpos = np.arange(frame_count * 3, dtype=np.float64).reshape(frame_count, 3)
    qvel = (qpos + 100.0).copy()
    xipos = np.arange(frame_count * 2 * 3, dtype=np.float64).reshape(frame_count, 2, 3)
    ximat = np.tile(np.eye(3, dtype=np.float64), (frame_count, 2, 1, 1))
    ximat[0, 1] = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    arrays = {
        "qpos": qpos,
        "qvel": qvel,
        "xipos": xipos,
        "ximat": ximat.reshape(frame_count, 2, 9),
    }
    for key, array in arrays.items():
        np.save(motion_data_dir / key / f"{clip_name}.npy", array)


@pytest.mark.parametrize("data_dir_index", (0, 1))
def test_species_mocap_clips_share_model_independent_loader(
    mocap_data_dirs: tuple[Path, Path],
    data_dir_index: int,
) -> None:
    motion_data_dir = mocap_data_dirs[data_dir_index]
    _write_clip(motion_data_dir, "demo")

    clip = MocapClip.load(
        motion_data_dir=motion_data_dir,
        clip_name="demo",
        device=torch.device("cpu"),
    )

    assert isinstance(clip, MocapClip)
    assert clip.frame_count == 2
    assert tuple(clip.qpos.shape) == (2, 3)
    assert tuple(clip.qvel.shape) == (2, 3)
    assert tuple(clip.xipos.shape) == (2, 2, 3)
    assert tuple(clip.ximat.shape) == (2, 2, 3, 3)
    assert tuple(clip.xquat.shape) == (2, 2, 4)
    assert clip.qpos.dtype == torch.float32
    assert clip.qpos.device.type == "cpu"

    expected_quat = rotmat_array_to_quat_array(
        np.load(motion_data_dir / "ximat" / "demo.npy").reshape(2, 2, 3, 3)
    )
    np.testing.assert_allclose(clip.xquat.numpy(), expected_quat, atol=1.0e-6)


def test_rotation_conversion_matches_mujoco_quaternion_convention() -> None:
    ximat = np.asarray(
        [[[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]],
        dtype=np.float64,
    )
    expected = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(expected, ximat[0, 0].reshape(9))

    np.testing.assert_allclose(
        rotmat_array_to_quat_array(ximat)[0, 0], expected, atol=1.0e-7
    )


def test_mocap_loader_validates_common_array_contract(tmp_path: Path) -> None:
    motion_data_dir = tmp_path / "mocap"
    for key in ("qpos", "qvel", "xipos", "ximat"):
        (motion_data_dir / key).mkdir(parents=True)
    _write_clip(motion_data_dir, "bad")
    np.save(
        motion_data_dir / "qvel" / "bad.npy",
        np.zeros((3, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="inconsistent frame counts"):
        MocapClip.load(
            motion_data_dir=motion_data_dir,
            clip_name="bad",
            device=torch.device("cpu"),
        )


def test_common_library_preserves_clip_offsets_and_rebuilds_subsets(
    mocap_data_dirs: tuple[Path, Path],
) -> None:
    motion_data_dir = mocap_data_dirs[1]
    _write_clip(motion_data_dir, "first", frame_count=2)
    _write_clip(motion_data_dir, "second", frame_count=3)

    library = MocapLibrary.load_many(
        motion_data_dir=motion_data_dir,
        clip_names=("first", "second"),
        device=torch.device("cpu"),
    )

    assert isinstance(library, MocapLibrary)
    assert library.clip_names == ("first", "second")
    assert library.clip_start.tolist() == [0, 2]
    assert library.clip_length.tolist() == [2, 3]
    assert library.clip_start.dtype == torch.int32
    assert library.clip_length.dtype == torch.int32
    assert library.clip_start.is_contiguous()
    assert library.clip_length.is_contiguous()
    assert tuple(library.qpos.shape) == (5, 3)
    assert library.clip_count == 2

    subset_library = MocapLibrary.from_clips(
        clips=[
            MocapClip.load(
                motion_data_dir=motion_data_dir,
                clip_name="first",
                device=torch.device("cpu"),
            )
        ],
        device=torch.device("cpu"),
    )
    assert subset_library.clip_names == ("first",)
