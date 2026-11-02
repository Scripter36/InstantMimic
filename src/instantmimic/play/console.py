from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PlayModeSummary:
    mode: str
    steps: int | None
    graph_mode: str
    video_encoder_backend: str | None = None
    video_render_backend: str | None = None
    overlay_enabled: bool = False


def _format_count(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def _format_float(value: float | int | str | None, *, precision: int = 2) -> str:
    if value is None:
        return "-"
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(scalar):
        return "-"
    return f"{scalar:.{precision}f}"


def _path_text(value: str | Path) -> str:
    return Path(value).expanduser().as_posix()


def format_play_header_lines(
    *,
    checkpoint_path: str | Path,
    config_name: str,
    log_dir: str | Path,
) -> list[str]:
    return [
        f"Play  {_path_text(checkpoint_path)}",
        f"Cfg   {config_name}",
        f"Log   {_path_text(log_dir)}",
    ]


def build_play_header_text_lines(
    *,
    checkpoint_path: str | Path,
    config_name: str,
    log_dir: str | Path,
) -> list[Any]:
    return [
        _build_fixed_label_text("Play", _path_text(checkpoint_path)),
        _build_fixed_label_text("Cfg", config_name),
        _build_fixed_label_text("Log", _path_text(log_dir)),
    ]


def _build_fixed_label_text(label: str, value: str) -> Any:
    from rich.text import Text

    text = Text()
    text.append(label, style="grey50")
    text.append(" " * (5 - len(label)))
    text.append(" ")
    text.append(str(value), style="white")
    return text


def format_play_mode_line(summary: PlayModeSummary) -> str:
    return "  ".join(
        f"{label}{'  ' if label == 'Mode' else ' '}{value}"
        for label, value in _play_mode_pairs(summary)
    )


def build_play_mode_text(summary: PlayModeSummary) -> Any:
    from rich.text import Text

    text = Text()

    def append_pair(label: str, value: str) -> None:
        if text:
            text.append("  ")
        text.append(label, style="grey50")
        text.append("  " if label == "Mode" else " ")
        text.append(value, style="white")

    for label, value in _play_mode_pairs(summary):
        append_pair(label, value)
    return text


def _play_mode_pairs(summary: PlayModeSummary) -> list[tuple[str, str]]:
    pairs = [("Mode", str(summary.mode))]
    if summary.steps is not None:
        pairs.append(("steps", _format_count(summary.steps)))
    pairs.append(("graph", str(summary.graph_mode)))
    if summary.video_encoder_backend:
        pairs.append(("encoder", str(summary.video_encoder_backend)))
    if summary.video_render_backend:
        pairs.append(("render", str(summary.video_render_backend)))
    if summary.overlay_enabled:
        pairs.append(("overlay", "on"))
    return pairs


def _play_startup_pairs(records: Any) -> list[tuple[str, str]]:
    phase_seconds: dict[str, float] = {}
    for record in records:
        if isinstance(record, tuple):
            name, seconds = record
        else:
            name = record.name
            seconds = record.seconds
        phase_name = str(name)
        phase_seconds[phase_name] = phase_seconds.get(phase_name, 0.0) + float(seconds)

    graph_seconds = sum(
        seconds for name, seconds in phase_seconds.items() if "graph" in name
    )
    parts = [
        ("env", phase_seconds.get("gym_make_env")),
        ("agent", phase_seconds.get("build_agent_runtime")),
        ("reset", phase_seconds.get("env_reset")),
        ("compile", phase_seconds.get("policy_compile_warmup")),
        ("graph", graph_seconds if graph_seconds > 0.0 else None),
    ]
    return [
        (label, f"{seconds:.2f}s") for label, seconds in parts if seconds is not None
    ]


def format_play_startup_summary(records: Any) -> str:
    pairs = _play_startup_pairs(records)
    body = "  ".join(f"{label} {seconds}" for label, seconds in pairs)
    return f"Startup  {body}" if body else "Startup  -"


def build_play_startup_text(records: Any) -> Any:
    from rich.text import Text

    pairs = _play_startup_pairs(records)
    text = Text()
    text.append("Startup", style="grey50")
    if not pairs:
        text.append("  -", style="white")
        return text
    text.append("  ")
    for index, (label, seconds) in enumerate(pairs):
        if index > 0:
            text.append("  ")
        text.append(label, style="grey50")
        text.append(" ")
        text.append(seconds, style="white")
    return text


def _tracking_result_text(result: Any) -> str:
    metrics = getattr(result, "metrics", {}) or {}
    if getattr(result, "tracking_valid", None) is True:
        state = "valid"
        ape = metrics.get("tracking_ape_mean")
        ee = metrics.get("tracking_ee_mean")
    elif getattr(result, "tracking_valid", None) is False:
        state = "invalid"
        ape = metrics.get("tracking_ape_mean_pre_failure")
        ee = metrics.get("tracking_ee_mean_pre_failure")
    else:
        state = "-"
        ape = None
        ee = None
    return (
        f"tracking {state}  "
        f"APE {_format_float(ape, precision=3)}  "
        f"EE {_format_float(ee, precision=3)}"
    )


def _tracking_result_parts(result: Any) -> list[tuple[str, str]]:
    metrics = getattr(result, "metrics", {}) or {}
    if getattr(result, "tracking_valid", None) is True:
        state = "valid"
        ape = metrics.get("tracking_ape_mean")
        ee = metrics.get("tracking_ee_mean")
    elif getattr(result, "tracking_valid", None) is False:
        state = "invalid"
        ape = metrics.get("tracking_ape_mean_pre_failure")
        ee = metrics.get("tracking_ee_mean_pre_failure")
    else:
        state = "-"
        ape = None
        ee = None
    return [
        ("tracking", state),
        ("APE", _format_float(ape, precision=3)),
        ("EE", _format_float(ee, precision=3)),
    ]


def _video_benchmark_result_text(metrics: dict[str, str]) -> str | None:
    render_avg = metrics.get("video_benchmark_render_total_avg_ms")
    render_median = metrics.get("video_benchmark_render_total_median_ms")
    render_sync_avg = metrics.get("video_benchmark_render_sync_avg_ms")
    encode_avg = metrics.get("video_benchmark_encode_total_avg_ms")
    if render_avg is None and render_median is None and render_sync_avg is None:
        return None
    warmup = metrics.get("video_benchmark_warmup_frames", "0")
    return (
        "Bench   "
        f"render {_format_float(render_avg)} ms avg / {_format_float(render_median)} med  "
        f"sync {_format_float(render_sync_avg)} ms avg  "
        f"encode {_format_float(encode_avg)} ms avg  "
        f"warmup {warmup}"
    )


def format_play_result_lines(
    result: Any,
    *,
    max_episode_length: int | None = None,
) -> list[str]:
    episode_length = getattr(result, "episode_length", None)
    episode_text = _format_count(episode_length)
    if max_episode_length is not None:
        episode_text = f"{episode_text} / {_format_count(max_episode_length)}"
    lines = [
        "Result  "
        f"reward {_format_float(getattr(result, 'episode_reward', None), precision=2)}  "
        f"episode {episode_text}  "
        f"{_tracking_result_text(result)}"
    ]
    video_output_path = getattr(result, "video_output_path", None)
    if video_output_path is not None:
        lines.append(f"Video   {_path_text(video_output_path)}")
    metrics = getattr(result, "metrics", {}) or {}
    benchmark_text = _video_benchmark_result_text(metrics)
    if benchmark_text is not None:
        lines.append(benchmark_text)
    benchmark_output_path = metrics.get("video_benchmark_output_path")
    if benchmark_output_path:
        lines.append(f"BenchJSON {_path_text(benchmark_output_path)}")
    return lines


def build_play_result_text_lines(
    result: Any,
    *,
    max_episode_length: int | None = None,
) -> list[Any]:
    from rich.text import Text

    episode_length = getattr(result, "episode_length", None)
    episode_text = _format_count(episode_length)
    if max_episode_length is not None:
        episode_text = f"{episode_text} / {_format_count(max_episode_length)}"

    result_text = Text()
    result_text.append("Result", style="grey50")
    result_text.append("  ")
    _append_result_pair(
        result_text,
        "reward",
        _format_float(getattr(result, "episode_reward", None), precision=2),
    )
    result_text.append("  ")
    _append_result_pair(result_text, "episode", episode_text)
    for label, value in _tracking_result_parts(result):
        result_text.append("  ")
        _append_result_pair(result_text, label, value)

    lines = [result_text]
    video_output_path = getattr(result, "video_output_path", None)
    if video_output_path is not None:
        video_text = Text()
        video_text.append("Video", style="grey50")
        video_text.append("   ")
        video_text.append(_path_text(video_output_path), style="white")
        lines.append(video_text)
    metrics = getattr(result, "metrics", {}) or {}
    benchmark_text = _video_benchmark_result_text(metrics)
    if benchmark_text is not None:
        bench_text = Text()
        bench_text.append("Bench", style="grey50")
        bench_text.append(benchmark_text[len("Bench") :], style="white")
        lines.append(bench_text)
    benchmark_output_path = metrics.get("video_benchmark_output_path")
    if benchmark_output_path:
        bench_json_text = Text()
        bench_json_text.append("BenchJSON", style="grey50")
        bench_json_text.append(" ")
        bench_json_text.append(_path_text(benchmark_output_path), style="white")
        lines.append(bench_json_text)
    return lines


def _append_result_pair(text: Any, label: str, value: str) -> None:
    text.append(label, style="grey50")
    text.append(" ")
    text.append(value, style="white")


def should_emit_playback_result_payload(
    *,
    emit_playback_result_payload: bool,
) -> bool:
    return bool(emit_playback_result_payload)


def print_play_header(
    *,
    checkpoint_path: str | Path,
    config_name: str,
    log_dir: str | Path,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    try:
        from rich.console import Console

        console = Console(stderr=True, highlighter=None)
        for line in build_play_header_text_lines(
            checkpoint_path=checkpoint_path,
            config_name=config_name,
            log_dir=log_dir,
        ):
            console.print(line, soft_wrap=True)
    except Exception:  # noqa: BLE001 - user-facing Rich fallback keeps playback output available.
        for line in format_play_header_lines(
            checkpoint_path=checkpoint_path,
            config_name=config_name,
            log_dir=log_dir,
        ):
            print(line, file=sys.stderr)
    print(file=sys.stderr)


def print_play_mode(summary: PlayModeSummary, *, enabled: bool = True) -> None:
    if not enabled:
        return
    try:
        from rich.console import Console

        Console(stderr=True, highlighter=None).print(
            build_play_mode_text(summary),
            soft_wrap=True,
        )
    except Exception:  # noqa: BLE001 - user-facing Rich fallback keeps playback output available.
        print(format_play_mode_line(summary), file=sys.stderr)


def print_play_startup_summary(records: Any, *, enabled: bool = True) -> None:
    if not enabled:
        return
    try:
        from rich.console import Console

        Console(stderr=True, highlighter=None).print(
            build_play_startup_text(records),
            soft_wrap=True,
        )
    except Exception:  # noqa: BLE001 - user-facing Rich fallback keeps playback output available.
        print(format_play_startup_summary(records), file=sys.stderr)
    print(file=sys.stderr)


def print_play_result(
    result: Any,
    *,
    max_episode_length: int | None = None,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    try:
        from rich.console import Console

        console = Console(stderr=True, highlighter=None)
        for line in build_play_result_text_lines(
            result,
            max_episode_length=max_episode_length,
        ):
            console.print(line, soft_wrap=True)
    except Exception:  # noqa: BLE001 - user-facing Rich fallback keeps playback output available.
        for line in format_play_result_lines(
            result,
            max_episode_length=max_episode_length,
        ):
            print(line, file=sys.stderr)


def install_compact_play_logging(
    *,
    log_dir: str | Path,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    play_log_path = Path(log_dir).expanduser().resolve() / "play.log"
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == play_log_path
        for handler in root.handlers
    ):
        play_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(play_log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
        )
        root.addHandler(file_handler)

    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.WARNING)
