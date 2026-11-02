from __future__ import annotations

import math
import subprocess
import tempfile
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import torch

DEFAULT_VIDEO_REMUX_TIMEOUT_SECONDS = 1800.0


def _load_pynvcodec() -> Any:
    try:
        import PyNvVideoCodec as nvc
    except ImportError as exc:
        raise RuntimeError(
            "PyNvVideoCodec is required for the pynvcodec video backend."
        ) from exc
    return nvc


def _cuda_nvtx_range(name: str):
    if not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.nvtx.range(name)


def _build_cuda_array_interface_from_tensor(frame: torch.Tensor) -> dict[str, object]:
    if not isinstance(frame, torch.Tensor):
        raise TypeError("Expected a torch.Tensor for CUDA frame export.")
    if not frame.is_cuda:
        raise ValueError("PyNvVideoCodec GPU input requires a CUDA tensor.")
    if frame.dtype != torch.uint8:
        raise ValueError("PyNvVideoCodec ARGB input requires torch.uint8 data.")
    if frame.ndim != 3 or frame.shape[2] != 4:
        raise ValueError("Expected an HxWx4 uint8 CUDA tensor for ARGB input.")
    contiguous = frame.contiguous()
    element_size = int(contiguous.element_size())
    strides = tuple(int(stride) * element_size for stride in contiguous.stride())
    return {
        "shape": tuple(int(dim) for dim in contiguous.shape),
        "strides": strides,
        "typestr": "|u1",
        "data": (int(contiguous.data_ptr()), False),
        "version": 3,
    }


def _default_seek_friendly_gop_size(fps: float) -> int:
    try:
        fps_value = float(fps)
    except (TypeError, ValueError):
        return 30
    if not math.isfinite(fps_value) or fps_value <= 0.0:
        return 30
    return max(30, min(int(round(fps_value)), 60))


def validate_nvenc_resolution(*, width: int, height: int, codec: str = "h264") -> None:
    """Check the active GPU's encoder bounds before allocating an encoder."""
    caps = _load_pynvcodec().GetEncoderCaps(gpuid=0, codec=codec)
    bounds = (
        caps["width_min"],
        caps["width_max"],
        caps["height_min"],
        caps["height_max"],
    )
    if not (bounds[0] <= width <= bounds[1] and bounds[2] <= height <= bounds[3]):
        raise ValueError(
            f"NVENC {codec} cannot encode {width}x{height}: supported width "
            f"{bounds[0]}..{bounds[1]}, height {bounds[2]}..{bounds[3]}. "
            "Choose a supported resolution or encoder_backend=imageio."
        )


def _create_pynv_encoder(
    *,
    width: int,
    height: int,
    fps: float,
    codec: str,
) -> Any:
    nvc = _load_pynvcodec()
    codec_name = str(codec).strip().lower()
    if codec_name != "h264":
        raise RuntimeError(
            "PyNvVideoCodec backend currently supports only video_codec=h264."
        )
    validate_nvenc_resolution(width=int(width), height=int(height), codec=codec_name)
    frame_rate = max(1, int(round(float(fps))))
    seek_gop_size = _default_seek_friendly_gop_size(float(fps))
    return nvc.CreateEncoder(
        int(width),
        int(height),
        "ARGB",
        False,
        gpu_id=0,
        codec=codec_name,
        fps=frame_rate,
        # Insert regular IDRs so exported MP4s stay responsive when seeking.
        gop=seek_gop_size,
        idrperiod=seek_gop_size,
        # Raw H.264 has no packet timestamps; B-frames make copy-muxed MP4 DTS
        # fragile in strict players.
        bf=0,
    )


def _fps_to_ffmpeg_rate(fps: float) -> tuple[str, int]:
    rate = Fraction(str(float(fps))).limit_denominator(100000)
    if rate.numerator <= 0 or rate.denominator <= 0:
        raise ValueError(f"Invalid fps value for ffmpeg remux: {fps!r}")
    if rate.denominator == 1:
        rate_arg = str(rate.numerator)
    else:
        rate_arg = f"{rate.numerator}/{rate.denominator}"
    track_timescale = max(int(rate.numerator), 1)
    return rate_arg, track_timescale


def _remux_elementary_stream_to_mp4(
    *,
    bitstream_path: Path,
    output_path: Path,
    codec: str,
    fps: float = 30.0,
    timeout_seconds: float = DEFAULT_VIDEO_REMUX_TIMEOUT_SECONDS,
) -> Path:
    if str(codec).strip().lower() != "h264":
        raise RuntimeError(
            "Elementary-stream remux currently supports only video_codec=h264."
        )
    rate_arg, track_timescale = _fps_to_ffmpeg_rate(float(fps))
    tick_rate = 2 * Fraction(str(float(fps))).limit_denominator(100000)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "h264",
        "-r",
        rate_arg,
        "-i",
        str(bitstream_path),
        "-c",
        "copy",
        "-bsf:v",
        f"h264_metadata=level=auto:tick_rate={tick_rate}:fixed_frame_rate_flag=1",
        "-avoid_negative_ts",
        "make_zero",
        "-movflags",
        "+faststart",
        "-video_track_timescale",
        str(track_timescale),
        str(output_path),
    ]
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("video remux timeout_seconds must be finite and positive")
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg remux timed out after {timeout:.1f}s for {bitstream_path}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise RuntimeError(
            f"ffmpeg remux failed (exit {exc.returncode}): {detail}"
        ) from exc
    return output_path


class PyNvCodecStreamingWriter:
    def __init__(
        self,
        *,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        codec: str,
        remux_timeout_seconds: float = DEFAULT_VIDEO_REMUX_TIMEOUT_SECONDS,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._encoder = _create_pynv_encoder(
            width=int(width),
            height=int(height),
            fps=float(fps),
            codec=str(codec).strip().lower(),
        )
        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".h264",
            dir=self.output_path.parent,
        )
        temp.close()
        self._bitstream_path = Path(temp.name)
        self._bitstream_file = self._bitstream_path.open("ab")
        self._fps = float(fps)
        self._codec = str(codec).strip().lower()
        self._remux_timeout_seconds = float(remux_timeout_seconds)
        self._closed = False
        self._final_output_path: Path | None = None

    def append_frame(self, frame: object) -> None:
        if self._closed:
            raise RuntimeError("Cannot append frames after the writer is closed.")
        encode_input = frame
        cuda_frame_input = False
        if isinstance(frame, torch.Tensor):
            _build_cuda_array_interface_from_tensor(frame)
            encode_input = frame.contiguous()
            cuda_frame_input = True
        elif not hasattr(frame, "cuda") and not hasattr(
            frame, "__cuda_array_interface__"
        ):
            raise TypeError(
                "PyNvCodecStreamingWriter expects a CUDA tensor or CUDA Array Interface object."
            )
        else:
            cuda_frame_input = True
        if cuda_frame_input and torch.cuda.is_available():
            with _cuda_nvtx_range("pynvcodec_input_sync"):
                torch.cuda.synchronize()
        with _cuda_nvtx_range("pynvcodec_encode"):
            bitstream = self._encoder.Encode(encode_input)
        if bitstream:
            with _cuda_nvtx_range("pynvcodec_bitstream_write"):
                self._bitstream_file.write(bytes(bitstream))

    def close(self) -> Path:
        if self._closed:
            if self._final_output_path is None:
                raise RuntimeError("Video writer closed without a finalized output.")
            return self._final_output_path
        trailer = self._encoder.EndEncode()
        if trailer:
            self._bitstream_file.write(bytes(trailer))
        self._bitstream_file.flush()
        self._bitstream_file.close()
        self._closed = True
        final_output_path = _remux_elementary_stream_to_mp4(
            bitstream_path=self._bitstream_path,
            output_path=self.output_path,
            codec=self._codec,
            fps=self._fps,
            timeout_seconds=self._remux_timeout_seconds,
        )
        self._final_output_path = final_output_path
        self._bitstream_path.unlink(missing_ok=True)
        return final_output_path
