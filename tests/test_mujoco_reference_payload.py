from __future__ import annotations

import mujoco
import numpy as np
import torch

from instantmimic.motion.mujoco_reference import materialize_mujoco_reference_payload


def _build_test_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco model="reference-test">
          <option gravity="0 0 -9.81"/>
          <worldbody>
              <body name="root" pos="0 0 0.2">
                <freejoint name="root_free"/>
                <geom type="sphere" size="0.1"/>
                <body name="child" pos="0.1 0.0 0.8">
                <joint name="child_hinge" type="hinge" axis="0 1 0"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.08"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def _direct_reference_arrays(
    model: mujoco.MjModel,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = int(qpos.shape[0])
    xpos = np.zeros((frame_count, model.nbody, 3), dtype=np.float32)
    xquat = np.zeros((frame_count, model.nbody, 4), dtype=np.float32)
    cvel = np.zeros((frame_count, model.nbody, 6), dtype=np.float32)
    subtree_com = np.zeros((frame_count, model.nbody, 3), dtype=np.float32)
    reference_body_state = np.zeros((frame_count, model.nbody, 13), dtype=np.float32)
    data = mujoco.MjData(model)

    for frame_idx in range(frame_count):
        data.qpos[:] = qpos[frame_idx].numpy()
        data.qvel[:] = qvel[frame_idx].numpy()
        mujoco.mj_forward(model, data)
        xpos[frame_idx] = data.xpos
        xquat[frame_idx] = data.xquat
        cvel[frame_idx] = data.cvel
        subtree_com[frame_idx] = data.subtree_com
        reference_body_state[frame_idx, :, :3] = data.xpos
        reference_body_state[frame_idx, :, 3:7] = data.xquat
        reference_body_state[frame_idx, :, 7:10] = data.cvel[:, 3:6]
        reference_body_state[frame_idx, :, 10:13] = data.cvel[:, :3]

    return xpos, xquat, cvel, subtree_com, reference_body_state


def test_materializer_preserves_mujoco_reference_abi_and_uses_one_forward_per_frame(
    monkeypatch,
) -> None:
    model = _build_test_model()
    qpos = torch.tensor(
        [
            [0.2, -0.1, 1.0, 0.98, 0.0, 0.2, 0.0, 0.25],
            [-0.3, 0.4, 1.2, 0.97, 0.1, 0.0, 0.2, -0.4],
        ],
        dtype=torch.float32,
    )
    qvel = torch.tensor(
        [
            [0.5, -0.2, 0.1, 0.0, 0.3, -0.1, 0.7],
            [-0.1, 0.6, -0.4, 0.2, 0.0, 0.4, -0.5],
        ],
        dtype=torch.float32,
    )
    expected = _direct_reference_arrays(model, qpos, qvel)
    forward_calls: list[int] = []
    original_mj_forward = mujoco.mj_forward

    def counting_mj_forward(test_model, test_data):
        forward_calls.append(1)
        return original_mj_forward(test_model, test_data)

    monkeypatch.setattr(mujoco, "mj_forward", counting_mj_forward)
    payload = materialize_mujoco_reference_payload(
        model=model,
        qpos=qpos,
        qvel=qvel,
        device=torch.device("cpu"),
        root_body_idx=2,
        include_cvel=True,
    )

    (
        expected_xpos,
        expected_xquat,
        expected_cvel,
        expected_subtree_com,
        expected_state,
    ) = expected
    np.testing.assert_allclose(payload.xpos.numpy(), expected_xpos, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        payload.xquat.numpy(), expected_xquat, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(payload.cvel.numpy(), expected_cvel, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        payload.subtree_com.numpy(), expected_subtree_com, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        payload.reference_body_state.numpy(), expected_state, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        payload.ref_com.numpy(), expected_subtree_com[:, 2], rtol=0.0, atol=0.0
    )

    assert len(forward_calls) == int(qpos.shape[0])
    for tensor in (
        payload.xpos,
        payload.xquat,
        payload.cvel,
        payload.subtree_com,
        payload.reference_body_state,
        payload.ref_com,
    ):
        assert tensor.device == torch.device("cpu")
        assert tensor.dtype == torch.float32
        assert tensor.is_contiguous()


def test_materializer_omits_standalone_cvel_by_default() -> None:
    model = _build_test_model()
    qpos = torch.tensor(
        [[0.2, -0.1, 1.0, 0.98, 0.0, 0.2, 0.0, 0.25]], dtype=torch.float32
    )
    qvel = torch.tensor([[0.5, -0.2, 0.1, 0.0, 0.3, -0.1, 0.7]], dtype=torch.float32)

    payload = materialize_mujoco_reference_payload(
        model=model,
        qpos=qpos,
        qvel=qvel,
        device=torch.device("cpu"),
    )

    assert payload.cvel is None


def test_materializer_omits_subtree_com_when_only_root_com_is_requested() -> None:
    model = _build_test_model()
    qpos = torch.tensor(
        [[0.2, -0.1, 1.0, 0.98, 0.0, 0.2, 0.0, 0.25]], dtype=torch.float32
    )
    qvel = torch.tensor([[0.5, -0.2, 0.1, 0.0, 0.3, -0.1, 0.7]], dtype=torch.float32)

    payload = materialize_mujoco_reference_payload(
        model=model,
        qpos=qpos,
        qvel=qvel,
        device=torch.device("cpu"),
        root_body_idx=2,
        include_subtree_com=False,
    )

    assert payload.subtree_com is None
    assert payload.ref_com is not None
