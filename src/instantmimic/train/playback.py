from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instantmimic.config.schema import SemanticConfig
from instantmimic.rendering.mujoco_gl import apply_mujoco_gl
from instantmimic.utils.project_paths import project_root

_LOGGER = logging.getLogger(__name__)
DEFAULT_PLAYBACK_SUBPROCESS_TIMEOUT_SECONDS = 3600.0


@dataclass(frozen=True)
class PostTrainingPlaybackRequest:
    checkpoint_path: Path
    log_dir: Path
    play_video: bool
    play_video_length: int | str
    play_use_cuda_graph: bool | None = None
    subprocess_timeout_seconds: float = DEFAULT_PLAYBACK_SUBPROCESS_TIMEOUT_SECONDS


class PostTrainingPlaybackRunner:
    """Run post-training playback without leaking play subprocess details into train."""

    def __init__(self, cfg: SemanticConfig) -> None:
        self.cfg = cfg

    def build_request(
        self,
        *,
        checkpoint_path: Path,
        log_dir: Path,
        play_video: bool | None = None,
        play_video_length: int | str | None = None,
        play_use_cuda_graph: bool | None = None,
    ) -> PostTrainingPlaybackRequest:
        playback = self.cfg.trainer.get("post_training_playback", {})
        if not hasattr(playback, "get"):
            raise TypeError("trainer.post_training_playback must be a mapping")
        play_video_value = playback.get("video", None)
        if play_video is None and play_video_value is None:
            has_display = bool(
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            )
            resolved_play_video = not has_display
        else:
            resolved_play_video = bool(
                play_video_value if play_video is None else play_video
            )
        video = self.cfg.play.get("video", {})
        play_steps = self.cfg.play.get("steps")
        if play_video_length is not None:
            default_length = play_video_length
        elif "video_length" in playback:
            default_length = playback["video_length"]
        elif hasattr(video, "get") and "length" in video:
            default_length = video["length"]
        elif play_steps is not None:
            default_length = play_steps
        else:
            default_length = "episode"
        resolved_length = _normalize_video_length(default_length)
        return PostTrainingPlaybackRequest(
            checkpoint_path=Path(checkpoint_path),
            log_dir=Path(log_dir),
            play_video=resolved_play_video,
            play_video_length=resolved_length,
            play_use_cuda_graph=(
                playback.get("use_cuda_graph", None)
                if play_use_cuda_graph is None
                else play_use_cuda_graph
            ),
            subprocess_timeout_seconds=_normalize_timeout_seconds(
                playback.get(
                    "subprocess_timeout_seconds",
                    DEFAULT_PLAYBACK_SUBPROCESS_TIMEOUT_SECONDS,
                ),
                owner="post-training playback subprocess",
            ),
        )

    def play_checkpoint_video(
        self,
        *,
        checkpoint_path: Path,
        log_dir: Path,
        play_video: bool,
        play_video_length: int | str,
        play_use_cuda_graph: bool | None = None,
        subprocess_timeout_seconds: float | None = None,
    ) -> Any:
        from instantmimic.play.runtime import play_checkpoint

        values = self.cfg.to_dict()
        play_cfg = values["play"]
        assert isinstance(play_cfg, dict)
        video_cfg = play_cfg.setdefault("video", {})
        assert isinstance(video_cfg, dict)
        play_cfg["checkpoint"] = str(checkpoint_path)
        video_cfg["enabled"] = bool(play_video)
        video_cfg["length"] = _normalize_video_length(play_video_length)
        if play_use_cuda_graph is not None:
            play_cfg["use_cuda_graph"] = bool(play_use_cuda_graph)
        playback_cfg = SemanticConfig.from_mapping(values)

        try:
            return play_checkpoint(
                playback_cfg,
                checkpoint_path=checkpoint_path,
                log_dir=log_dir,
            )
        except Exception as exc:
            if not is_headless_playback_context_error(exc):
                raise
            _LOGGER.warning(
                "In-process playback hit a headless GL context error; retrying via "
                "standalone play subprocess. Original error: %s",
                exc,
            )
            return self.run_playback_subprocess(
                checkpoint_path=checkpoint_path,
                play_video=play_video,
                play_video_length=play_video_length,
                play_use_cuda_graph=play_use_cuda_graph,
                subprocess_timeout_seconds=subprocess_timeout_seconds,
            )

    def run_playback_subprocess(
        self,
        *,
        checkpoint_path: Path,
        play_video: bool,
        play_video_length: int | str,
        play_use_cuda_graph: bool | None = None,
        subprocess_timeout_seconds: float | None = None,
    ) -> Any:
        from instantmimic.play.runtime import (
            PlaybackResult,
            parse_playback_result_stdout,
        )

        command = [
            sys.executable,
            "-m",
            "instantmimic.play",
            f"++play.checkpoint={checkpoint_path.as_posix()}",
            f"play.video.enabled={'true' if play_video else 'false'}",
            f"play.video.length={play_video_length}",
            "++play.output.emit_result_payload=true",
        ]
        video_cfg = self.cfg.play.get("video", {})
        if not hasattr(video_cfg, "get"):
            raise TypeError("play.video must be a mapping")

        def video_option(semantic_key: str) -> Any:
            return video_cfg.get(semantic_key, None)

        overlay_rewards = video_option("overlay_rewards")
        if overlay_rewards is not None:
            command.append(
                "++play.video.overlay_rewards="
                f"{'true' if bool(overlay_rewards) else 'false'}"
            )
        overlay_history_seconds = video_option("overlay_history_seconds")
        if overlay_history_seconds is not None:
            command.append(
                f"++play.video.overlay_history_seconds={float(overlay_history_seconds)}"
            )
        overlay_panel_width = video_option("overlay_panel_width")
        if overlay_panel_width is not None:
            command.append(
                f"++play.video.overlay_panel_width={int(overlay_panel_width)}"
            )
        video_render_backend = video_option("render_backend")
        if video_render_backend is not None:
            command.append(f"++play.video.render_backend={str(video_render_backend)}")
        video_tile_envs = video_option("tile_envs")
        if video_tile_envs is not None:
            tile_env_count = int(video_tile_envs)
            command.append(f"play.num_envs={tile_env_count}")
            command.append(f"++play.video.tile_envs={tile_env_count}")
        video_tile_cols = video_option("tile_cols")
        if video_tile_cols is not None:
            command.append(f"++play.video.tile_cols={int(video_tile_cols)}")
        video_encoder_backend = video_option("encoder_backend")
        if video_encoder_backend is not None:
            command.append(f"++play.video.encoder_backend={str(video_encoder_backend)}")
        video_codec = video_option("codec")
        if video_codec is not None:
            command.append(f"++play.video.codec={str(video_codec)}")
        if play_use_cuda_graph is not None:
            command.append(
                "++play.use_cuda_graph="
                f"{'true' if bool(play_use_cuda_graph) else 'false'}"
            )

        env = dict(os.environ)
        apply_mujoco_gl(force_headless=True)
        env.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
        timeout = _normalize_timeout_seconds(
            subprocess_timeout_seconds
            if subprocess_timeout_seconds is not None
            else DEFAULT_PLAYBACK_SUBPROCESS_TIMEOUT_SECONDS,
            owner="post-training playback subprocess",
        )
        try:
            completed = subprocess.run(
                command,
                cwd=str(project_root()),
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"post-training playback subprocess timed out after {timeout:.1f}s"
            ) from exc
        if completed.stdout:
            _LOGGER.info("Playback stdout:\n%s", completed.stdout.strip())
        if completed.stderr:
            _LOGGER.info("Playback stderr:\n%s", completed.stderr.strip())
        if completed.returncode != 0:
            raise RuntimeError(
                f"Playback subprocess failed (exit_code={completed.returncode})."
            )
        parsed_result = parse_playback_result_stdout(completed.stdout)
        if parsed_result is None:
            return PlaybackResult(checkpoint_path=checkpoint_path)
        return parsed_result

    def play_best_policy_after_training(
        self,
        *,
        checkpoint_path: Path,
        log_dir: Path,
    ) -> None:
        request = self.build_request(checkpoint_path=checkpoint_path, log_dir=log_dir)
        _LOGGER.info(
            "Starting post-training playback from %s (video=%s, video_length=%s).",
            request.checkpoint_path,
            request.play_video,
            request.play_video_length,
        )
        playback_result = self.play_checkpoint_video(
            checkpoint_path=request.checkpoint_path,
            log_dir=request.log_dir,
            play_video=request.play_video,
            play_video_length=request.play_video_length,
            play_use_cuda_graph=request.play_use_cuda_graph,
            subprocess_timeout_seconds=request.subprocess_timeout_seconds,
        )
        video_output_path = playback_result.video_output_path
        if video_output_path:
            _LOGGER.info(
                "Post-training playback saved video to: %s",
                video_output_path.as_posix(),
            )


def is_headless_playback_context_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "opengl platform library has not been loaded" in message
        or "valid opengl context has not been created" in message
        or "glfw library is not initialized" in message
    )


def _normalize_video_length(value: int | str) -> int | str:
    if isinstance(value, str) and value.strip().lower() == "episode":
        return "episode"
    length = int(value)
    if length < 1:
        raise ValueError(
            "playback video length must be a positive integer or 'episode'"
        )
    return length


def _normalize_timeout_seconds(value: object, *, owner: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner} timeout must be finite and positive") from exc
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError(f"{owner} timeout must be finite and positive")
    return timeout
