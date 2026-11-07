"""CUDA checks for the network-scoped TF32 policy."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from instantmimic.motion.loader import load_bvh_file
from instantmimic.utils.network_precision import (
    configure_ieee_float32_precision,
    network_matmul_precision,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _write_tiny_bvh(path, frame_count: int = 33) -> str:
    rows = []
    for k in range(frame_count):
        toe = 20.0 * np.sin(0.7 * k)
        tip = 15.0 * np.cos(0.5 * k)
        rows.append(f"0 1 {0.01 * k} {2.0 * k} 0 0 0 {toe} 0 0 {tip} 0")
    path.write_text(
        "HIERARCHY\nROOT Hips\n{\n OFFSET 0 0 0\n"
        " CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n"
        " JOINT Toe\n {\n  OFFSET 0 -0.9 0.1\n"
        "  CHANNELS 3 Zrotation Xrotation Yrotation\n"
        "  JOINT Tip\n  {\n   OFFSET 0 0 0.05\n"
        "   CHANNELS 3 Zrotation Xrotation Yrotation\n"
        "   End Site\n   {\n    OFFSET 0 0 0.02\n   }\n  }\n }\n}\n"
        f"MOTION\nFrames: {frame_count}\nFrame Time: 0.02\n" + "\n".join(rows) + "\n"
    )
    return str(path)


MODEL = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="root" pos="0 0 1"><freejoint/><geom size=".1"/>
    <body name="toe" pos=".1 .02 -.9">
      <joint name="toe" axis="0 1 0" range="-1.4 0.6"/><geom size=".02"/>
      <body name="tip" pos=".05 0 0">
        <joint name="tip" axis="0 1 0" range="-1.4 1.4"/><geom size=".01"/>
      </body>
    </body>
  </body>
</worldbody></mujoco>"""


def _load(motion_file: str, device: torch.device, model: mujoco.MjModel):
    return load_bvh_file(
        motion_file,
        1.0,
        [0.0, 0.0, 0.0],
        {"Hips": "root", "Toe": "toe", "Tip": "tip"},
        device,
        model,
        0.02,
        cache_mode="off",
    )


def test_bvh_bake_on_cuda_matches_cpu_at_ieee_float32(tmp_path) -> None:
    configure_ieee_float32_precision()
    model = mujoco.MjModel.from_xml_string(MODEL)
    motion = _write_tiny_bvh(tmp_path / "tiny.bvh")

    cpu = _load(motion, torch.device("cpu"), model)
    cuda = _load(motion, torch.device("cuda"), model)

    # TF32 truncates the 3x3 y-up/z-up einsums to a 10-bit mantissa. The IEEE
    # path must agree to ordinary float32 reduction error instead.
    torch.testing.assert_close(cuda["qpos"].cpu(), cpu["qpos"], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(cuda["qvel"].cpu(), cpu["qvel"], rtol=0.0, atol=1e-4)
    torch.testing.assert_close(
        cuda["reference_body_state"].cpu(),
        cpu["reference_body_state"],
        rtol=0.0,
        atol=1e-6,
    )


def test_compiled_network_scope_does_not_recompile_with_ieee_env_math() -> None:
    from torch._dynamo.utils import counters

    configure_ieee_float32_precision()
    torch._dynamo.reset()
    counters.clear()
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 16),
    ).cuda()
    compiled = torch.compile(model, mode="default")
    obs = torch.randn(8, 16, device="cuda")
    env_weight = torch.randn(16, 16, device="cuda")

    with network_matmul_precision():
        compiled(obs)

    for _ in range(3):
        # Environment-side float32 matmul runs IEEE outside the scope.
        _ = obs @ env_weight
        with network_matmul_precision():
            compiled(obs)

    assert counters["stats"]["unique_graphs"] == 1
    assert counters["frames"]["ok"] == 1
    assert sum(counters["graph_break"].values()) == 0
    assert torch.get_float32_matmul_precision() == "highest"
