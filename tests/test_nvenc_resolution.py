from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from instantmimic.play import render
from instantmimic.rendering import video_cuda


@pytest.fixture
def encoder(monkeypatch):
    module = SimpleNamespace(
        GetEncoderCaps=Mock(
            return_value={
                "width_min": 145,
                "width_max": 4096,
                "height_min": 49,
                "height_max": 4096,
            }
        ),
        CreateEncoder=Mock(return_value=object()),
    )
    monkeypatch.setattr(video_cuda, "_load_pynvcodec", lambda: module)
    monkeypatch.setattr(render, "_pynvcodec_is_available", lambda: True)
    monkeypatch.setattr(render, "_mujoco_gl_playback_is_available", lambda: True)
    return module


@pytest.mark.parametrize(
    "width,height", [(128, 128), (144, 128), (160, 48), (4097, 128), (160, 4097)]
)
def test_forced_encoder_rejects_out_of_capability_bounds_before_creation(
    encoder, width, height
):
    with pytest.raises(ValueError, match="NVENC h264 cannot encode"):
        video_cuda._create_pynv_encoder(
            width=width, height=height, fps=120, codec="h264"
        )
    encoder.CreateEncoder.assert_not_called()


@pytest.mark.parametrize("width,height", [(145, 49), (160, 128), (4096, 4096)])
def test_supported_bounds_retain_encoder_parameters(encoder, width, height):
    video_cuda._create_pynv_encoder(width=width, height=height, fps=120, codec="h264")
    args, kwargs = encoder.CreateEncoder.call_args
    assert args == (width, height, "ARGB", False)
    assert kwargs["fps"] == 120
    assert "frameRateNum" not in kwargs
    assert "frameRateDen" not in kwargs
    assert kwargs["bf"] == 0
    encoder.GetEncoderCaps.assert_called_with(gpuid=0, codec="h264")


def test_auto_uses_software_only_for_unsupported_final_output_size(encoder):
    options = dict(
        codec="h264",
        take_video=True,
        video_render_backend="mujoco_gl",
        output_height=128,
    )
    assert (
        render.select_video_encoder_backend(backend="auto", output_width=128, **options)
        == "imageio"
    )
    assert (
        render.select_video_encoder_backend(backend="auto", output_width=256, **options)
        == "pynvcodec"
    )
    with pytest.raises(RuntimeError, match="supported width 145"):
        render.select_video_encoder_backend(
            backend="pynvcodec", output_width=128, **options
        )


def test_software_selection_does_not_query_gpu_capabilities(encoder):
    assert (
        render.select_video_encoder_backend(
            backend="imageio",
            codec="h264",
            take_video=True,
            video_render_backend="mujoco_gl",
            output_width=128,
            output_height=128,
        )
        == "imageio"
    )
    encoder.GetEncoderCaps.assert_not_called()


def test_capability_probe_errors_are_not_silently_downgraded(encoder):
    encoder.GetEncoderCaps.side_effect = RuntimeError("device initialization failed")
    with pytest.raises(RuntimeError, match="device initialization failed"):
        render.select_video_encoder_backend(
            backend="auto",
            codec="h264",
            take_video=True,
            video_render_backend="mujoco_gl",
            output_width=256,
            output_height=128,
        )


def test_rejected_writer_does_not_leave_temporary_stream_files(encoder, tmp_path):
    with pytest.raises(ValueError, match="NVENC h264 cannot encode"):
        video_cuda.PyNvCodecStreamingWriter(
            output_path=tmp_path / "small.mp4",
            width=128,
            height=128,
            fps=120,
            codec="h264",
        )
    assert list(tmp_path.iterdir()) == []
