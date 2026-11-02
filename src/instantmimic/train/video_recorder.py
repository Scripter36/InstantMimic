from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


class TrainIterationVideoRecorder:
    def __init__(
        self,
        *,
        log_dir: str | Path,
        checkpoints_dir: str | Path,
        interval_iterations: int,
        video_length: int,
        save_checkpoint_fn: Callable[[Path], None],
        playback_video_fn: Callable[..., dict[str, str]],
        play_use_cuda_graph: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir).expanduser().resolve()
        self._checkpoints_dir = Path(checkpoints_dir).expanduser().resolve()
        self._interval_iterations = max(0, int(interval_iterations))
        self._video_length = max(1, int(video_length))
        self._save_checkpoint_fn = save_checkpoint_fn
        self._playback_video_fn = playback_video_fn
        self._play_use_cuda_graph = bool(play_use_cuda_graph)

    def should_record(self, iteration_index: int) -> bool:
        if self._interval_iterations <= 0:
            return False
        return (
            int(iteration_index) > 0
            and int(iteration_index) % self._interval_iterations == 0
        )

    def _target_video_path(
        self,
        *,
        iteration_index: int,
        step_index: int,
        source_path: Path,
    ) -> Path:
        suffix = source_path.suffix or ".mp4"
        return (
            self._log_dir
            / "videos"
            / "train"
            / f"train_iter_{int(iteration_index):06d}_step_{int(step_index):06d}{suffix}"
        )

    def record(
        self,
        *,
        iteration_index: int,
        step_index: int,
    ) -> Path | None:
        if not self.should_record(iteration_index):
            return None

        temp_checkpoint_path = (
            self._checkpoints_dir / f".train_video_iter_{int(iteration_index):06d}.pt"
        )
        temp_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_checkpoint_fn(temp_checkpoint_path)

        try:
            playback_result = self._playback_video_fn(
                checkpoint_path=temp_checkpoint_path,
                log_dir=self._log_dir,
                play_video=True,
                play_video_length=self._video_length,
                play_use_cuda_graph=self._play_use_cuda_graph,
                iteration_index=int(iteration_index),
                step_index=int(step_index),
            )
            video_output_path = playback_result.get("video_output_path", "")
            if not video_output_path:
                _LOGGER.warning(
                    "Train video recorder did not receive a video_output_path for iteration=%d.",
                    int(iteration_index),
                )
                return None

            source_path = Path(video_output_path).expanduser()
            target_path = self._target_video_path(
                iteration_index=int(iteration_index),
                step_index=int(step_index),
                source_path=source_path,
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                target_path.unlink()
            source_path.replace(target_path)
            _LOGGER.info(
                "Saved train video for iteration=%d to: %s",
                int(iteration_index),
                target_path.as_posix(),
            )
            return target_path
        finally:
            temp_checkpoint_path.unlink(missing_ok=True)

    def close(self) -> None:
        return None
