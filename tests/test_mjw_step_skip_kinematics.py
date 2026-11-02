import mujoco
import mujoco_warp as mjw
import pytest
import torch
import warp as wp

from instantmimic.utils.mjw_step_optimized import step_skip_kinematics


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_step_skip_kinematics_matches_stock_step_after_fresh_kinematics():
    torch.manual_seed(1234)
    mj_model = mujoco.MjModel.from_xml_path("data/humanoid/mjcf/humanoid_pd.xml")
    model = mjw.put_model(mj_model)
    stock_data = mjw.put_data(
        mj_model,
        mujoco.MjData(mj_model),
        nworld=16,
        njmax=64,
        nconmax=8,
    )
    skip_data = mjw.put_data(
        mj_model,
        mujoco.MjData(mj_model),
        nworld=16,
        njmax=64,
        nconmax=8,
    )

    qpos = (
        torch.as_tensor(mj_model.qpos0, device="cuda", dtype=torch.float32)
        .expand(16, -1)
        .clone()
    )
    qvel = torch.randn((16, mj_model.nv), device="cuda") * 0.01
    ctrl = torch.randn((16, mj_model.nu), device="cuda") * 0.05
    for data in (stock_data, skip_data):
        wp.to_torch(data.qpos).copy_(qpos)
        wp.to_torch(data.qvel).copy_(qvel)
        wp.to_torch(data.ctrl).copy_(ctrl)

    mjw.kinematics(model, skip_data)
    mjw.com_pos(model, skip_data)
    mjw.com_vel(model, skip_data)

    for substep_idx in range(4):
        mjw.step(model, stock_data)
        if substep_idx == 0:
            step_skip_kinematics(model, skip_data)
        else:
            mjw.step(model, skip_data)
    wp.synchronize()

    assert torch.allclose(
        wp.to_torch(stock_data.qpos),
        wp.to_torch(skip_data.qpos),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        wp.to_torch(stock_data.qvel),
        wp.to_torch(skip_data.qvel),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
