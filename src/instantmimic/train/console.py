from __future__ import annotations

import logging
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Any

from tqdm.auto import tqdm

_LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConsoleState:
    iteration: int
    total_iterations: int
    eta: str
    elapsed: str = "-"
    fps: float | None = None
    train_reward: float | None = None
    train_episode_length: float | None = None
    max_episode_length: int | None = None
    eval_reward: float | None = None
    eval_episode_length: float | None = None
    eval_max_episode_length: int | None = None
    eval_ape: float | None = None
    best_reward: float | None = None
    best_iteration: int | None = None
    best_episode_length: float | None = None
    best_max_episode_length: int | None = None
    best_ape: float | None = None
    best_checkpoint_source: str | None = None
    best_checkpoint_metric: str | None = None
    best_checkpoint_value: float | None = None
    best_checkpoint_iteration: int | None = None


@dataclass(frozen=True)
class ConsoleEvent:
    kind: str
    message: str


def append_console_event(
    events: list[ConsoleEvent],
    kind: str,
    message: str,
) -> list[ConsoleEvent]:
    return [*events, ConsoleEvent(kind=kind, message=message)]


def _format_count(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def _format_compact_count(value: int | float | None) -> str:
    if value is None:
        return "-"
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        return "-"
    if scalar >= 1_000_000_000.0:
        return f"{scalar / 1_000_000_000.0:.2f}G"
    if scalar >= 1_000_000.0:
        return f"{scalar / 1_000_000.0:.2f}M"
    if scalar >= 1_000.0:
        return f"{scalar / 1_000.0:.1f}K"
    return _format_count(scalar)


def _format_float(value: float | int | None, *, precision: int = 1) -> str:
    if value is None:
        return "-"
    scalar = float(value)
    if not math.isfinite(scalar):
        return "-"
    return f"{scalar:.{precision}f}"


def _format_fps(value: float | int | None) -> str:
    if value is None:
        return "-"
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        return "-"
    if scalar >= 1_000_000.0:
        return f"{scalar / 1_000_000.0:.2f}Mfps"
    if scalar >= 1_000.0:
        return f"{scalar / 1_000.0:.1f}Kfps"
    return f"{scalar:.0f}fps"


def _format_duration_value(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    scalar = float(seconds)
    if not math.isfinite(scalar):
        return "-"
    return _format_duration(scalar)


def format_duration(seconds: float | int | None) -> str:
    return _format_duration_value(seconds)


def _format_episode(length: float | None, max_length: int | None) -> str:
    if length is None:
        return "-"
    value = _format_float(length, precision=0)
    if max_length is None:
        return value
    return f"{value} / {int(max_length):,}"


def _format_reward_with_iteration(
    reward: float | None,
    iteration: int | None,
) -> str:
    reward_text = _format_float(reward, precision=1)
    if reward_text == "-" or iteration is None:
        return reward_text
    return f"{reward_text} @ {_format_count(iteration)}"


def _format_ape(value: float | None) -> str:
    if value is None:
        return "-"
    return f"APE {_format_float(value, precision=3)}"


def _status_columns(state: TrainingConsoleState) -> list[str]:
    columns = ["train"]
    if state.eval_reward is not None or state.eval_episode_length is not None:
        columns.append("eval")
    columns.append("best")
    return columns


def _status_rows(state: TrainingConsoleState) -> list[tuple[str, list[str]]]:
    columns = _status_columns(state)
    rows: list[tuple[str, list[str]]] = []

    reward_values = [_format_float(state.train_reward, precision=1)]
    if "eval" in columns:
        reward_values.append(_format_float(state.eval_reward, precision=1))
    reward_values.append(
        _format_reward_with_iteration(state.best_reward, state.best_iteration)
    )
    rows.append(("reward", reward_values))

    episode_values = [
        _format_episode(state.train_episode_length, state.max_episode_length)
    ]
    if "eval" in columns:
        episode_values.append(
            _format_episode(
                state.eval_episode_length,
                state.eval_max_episode_length or state.max_episode_length,
            )
        )
    episode_values.append(
        _format_episode(
            state.best_episode_length,
            state.best_max_episode_length or state.max_episode_length,
        )
    )
    rows.append(("episode length", episode_values))

    if state.eval_ape is not None or state.best_ape is not None:
        tracking_values = ["-"]
        if "eval" in columns:
            tracking_values.append(_format_ape(state.eval_ape))
        tracking_values.append(_format_ape(state.best_ape))
        rows.append(("tracking", tracking_values))

    if state.best_checkpoint_metric is not None:
        source_values = ["-"]
        if "eval" in columns:
            source_values.append("-")
        source_values.append(state.best_checkpoint_source or "-")
        rows.append(("checkpoint source", source_values))

        metric_values = ["-"]
        if "eval" in columns:
            metric_values.append("-")
        metric_values.append(
            _format_reward_with_iteration(
                state.best_checkpoint_value,
                state.best_checkpoint_iteration,
            )
        )
        rows.append((state.best_checkpoint_metric, metric_values))

    return rows


def _progress_bar(width: int, completed: int, total: int) -> str:
    ratio = 0.0 if total <= 0 else min(1.0, max(0.0, float(completed) / float(total)))
    filled = int(round(width * ratio))
    return "█" * filled + "-" * (width - filled)


def _plain_progress_line(
    state: TrainingConsoleState,
    *,
    width: int = 100,
) -> str:
    total = max(0, int(state.total_iterations))
    iteration = min(max(0, int(state.iteration)), total) if total > 0 else 0
    percent = 0.0 if total <= 0 else 100.0 * float(iteration) / float(total)
    suffix = (
        f" {percent:>3.0f}%  "
        f"{_format_count(iteration)} / {_format_count(total)} iter  "
        f"{_format_fps(state.fps)}  elapsed {state.elapsed}  ETA {state.eta}"
    )
    prefix = "Training |"
    min_bar_width = 12
    max_bar_width = 40
    available = max(min_bar_width, int(width) - len(prefix) - len("|") - len(suffix))
    bar_width = min(max_bar_width, available)
    return f"{prefix}{_progress_bar(bar_width, iteration, total)}|{suffix}"


def _plain_scoreboard_lines(state: TrainingConsoleState) -> list[str]:
    columns = _status_columns(state)
    label_width = 21
    column_width = 15
    lines = [
        " " * label_width + "".join(f"{column:<{column_width}}" for column in columns)
    ]
    for label, values in _status_rows(state):
        lines.append(
            f"{label:<{label_width}}"
            + "".join(f"{value:<{column_width}}" for value in values)
        )
    return lines


def _plain_separator_line(width: int) -> str:
    return "─" * max(1, int(width))


def build_plain_status_lines(
    state: TrainingConsoleState,
    *,
    width: int = 100,
) -> list[str]:
    return [
        _plain_progress_line(state, width=width),
        "",
        *_plain_scoreboard_lines(state),
    ]


def build_plain_training_console_lines(
    state: TrainingConsoleState,
    *,
    events: Sequence[ConsoleEvent] = (),
    width: int = 100,
) -> list[str]:
    lines = [
        format_event_line(kind=event.kind, message=event.message) for event in events
    ]
    if lines:
        lines.append(_plain_separator_line(width))
    lines.extend(_plain_scoreboard_lines(state))
    lines.append(_plain_separator_line(width))
    lines.append(_plain_progress_line(state, width=width))
    return lines


def format_event_line(*, kind: str, message: str) -> str:
    return f"{kind:<9} {message}"


_EVENT_NUMBER_RE = re.compile(
    r"(?<![A-Za-z_0-9])"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?"
    r"(?::\d{2}){0,2}"
    r"[KMG]?"
    r"(?![A-Za-z_0-9])"
)


def build_event_text(*, kind: str, message: str) -> Any:
    from rich.text import Text

    text = Text()
    text.append(f"{kind:<9} ", style=_event_style(kind))
    cursor = 0
    for match in _EVENT_NUMBER_RE.finditer(message):
        if match.start() > cursor:
            text.append(message[cursor : match.start()], style="grey50")
        text.append(message[match.start() : match.end()], style="bright_white")
        cursor = match.end()
    if cursor < len(message):
        text.append(message[cursor:], style="grey50")
    return text


def format_eval_event_message(
    *,
    iteration: int,
    step: int,
    frames: int,
    active_train_seconds: float | None,
    reward: float | None,
    episode_length: float | None,
    max_episode_length: int | None,
    ape: float | None,
) -> str:
    return (
        f"iter {_format_count(iteration)}  "
        f"step {_format_count(step)}  "
        f"frames {_format_compact_count(frames)}  "
        f"active {_format_duration_value(active_train_seconds)}  "
        f"reward {_format_float(reward, precision=1)}  "
        f"episode {_format_episode(episode_length, max_episode_length)}  "
        f"APE {_format_float(ape, precision=3)}"
    )


def _event_style(kind: str) -> str:
    normalized = kind.strip().lower()
    if normalized == "new best":
        return "green"
    if normalized == "warning":
        return "yellow"
    if normalized == "error":
        return "red"
    if normalized == "eval":
        return "cyan"
    return "white"


def format_run_header_lines(
    *,
    config_name: str,
    experiment_name: str,
    log_dir: str | Path,
) -> list[str]:
    return [
        f"Run  {experiment_name}",
        f"Cfg  {config_name}",
        f"Log  {Path(log_dir).as_posix()}",
    ]


def format_startup_summary(records: Any) -> str:
    phase_seconds: dict[str, float] = {}
    for record in records:
        if isinstance(record, tuple):
            name, seconds = record
        else:
            name = record.name
            seconds = record.seconds
        phase_seconds[str(name)] = float(seconds)

    prewarm_seconds = sum(
        seconds
        for name, seconds in phase_seconds.items()
        if "prewarm" in name or name == "tracking_benchmark_prewarm_graphs"
    )
    parts = [
        ("env", phase_seconds.get("gym_make_env")),
        ("agent", phase_seconds.get("agent_from_cfg")),
        ("reset", phase_seconds.get("env_reset")),
        ("prewarm", prewarm_seconds if prewarm_seconds > 0.0 else None),
    ]
    body = "  ".join(
        f"{label} {seconds:.2f}s" for label, seconds in parts if seconds is not None
    )
    return f"Startup  {body}" if body else "Startup  -"


def print_startup_summary(records: Any, *, enabled: bool = True) -> None:
    if not enabled:
        return
    print(format_startup_summary(records), file=sys.stderr)
    print(file=sys.stderr)


def format_eta(
    *,
    iteration: int,
    total_iterations: int,
    start_time: float,
    now: float | None = None,
) -> str:
    now = perf_counter() if now is None else float(now)
    elapsed = max(0.0, now - float(start_time))
    if iteration <= 0 or total_iterations <= 0:
        return "-"
    remaining = max(0, int(total_iterations) - int(iteration))
    seconds = elapsed * float(remaining) / float(iteration)
    return _format_duration(seconds)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(round(seconds % 60))
    if secs == 60:
        secs = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        hours += 1
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class _TqdmTrainingConsole:
    def __init__(self, *, total_iterations: int) -> None:
        self._bar = tqdm(
            total=total_iterations,
            desc="Training",
            dynamic_ncols=True,
        )
        self._last_iteration = 0

    def __enter__(self) -> _TqdmTrainingConsole:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._bar.close()

    def update(self, state: TrainingConsoleState) -> None:
        iteration = int(state.iteration)
        delta = max(0, iteration - self._last_iteration)
        if delta:
            self._bar.update(delta)
            self._last_iteration = iteration

    def event(self, kind: str, message: str) -> None:
        self._bar.write(format_event_line(kind=kind, message=message))


class _RichTrainingConsole:
    def __init__(self, *, total_iterations: int) -> None:
        from rich.console import Console, Group
        from rich.live import Live

        self._console = Console(stderr=True, highlighter=None)
        self._group_type = Group
        self._live_type = Live
        self._width = self._console.width
        self._state = TrainingConsoleState(
            iteration=0,
            total_iterations=total_iterations,
            eta="-",
        )
        self._live = None

    def __enter__(self) -> _RichTrainingConsole:
        self._live = self._create_live()
        self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc, tb)

    def update(self, state: TrainingConsoleState) -> None:
        self._state = state
        if self._live is not None:
            self._live.update(self._render())

    def event(self, kind: str, message: str) -> None:
        self._console.print(build_event_text(kind=kind, message=message))

    def _create_live(self) -> Any:
        return self._live_type(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=True,
            redirect_stdout=True,
            redirect_stderr=True,
        )

    def _render(self) -> Any:
        return self._group_type(
            self._render_separator(),
            self._render_scoreboard(),
            self._render_separator(),
            self._render_progress_line(),
        )

    def _render_separator(self) -> Any:
        from rich.rule import Rule

        return Rule(style="dim")

    def _render_progress_line(self) -> Any:
        from rich.text import Text

        total = max(0, int(self._state.total_iterations))
        iteration = min(max(0, int(self._state.iteration)), total) if total > 0 else 0
        percent = 0.0 if total <= 0 else 100.0 * float(iteration) / float(total)
        suffix = (
            f" {percent:>3.0f}%  "
            f"{_format_count(iteration)} / {_format_count(total)} iter  "
            f"{_format_fps(self._state.fps)}  "
            f"elapsed {self._state.elapsed}  ETA {self._state.eta}"
        )
        prefix = "Training |"
        available = max(12, self._width - len(prefix) - len("|") - len(suffix))
        bar_width = min(40, available)
        bar = _progress_bar(bar_width, iteration, total)

        text = Text("Training ", style="white")
        text.append("|", style="dim")
        filled = bar.count("█")
        text.append("█" * filled, style="white")
        text.append("-" * (bar_width - filled), style="dim")
        text.append("|", style="dim")
        text.append(suffix, style="white")
        return text

    def _render_scoreboard(self) -> Any:
        from rich.table import Table

        columns = _status_columns(self._state)
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="left", style="dim", no_wrap=True)
        for column in columns:
            style = "green" if column == "best" else "white"
            table.add_column(justify="left", style=style, no_wrap=True)
        table.add_row("", *[self._text(column, "dim") for column in columns])
        for label, values in _status_rows(self._state):
            styled_values = []
            for column, value in zip(columns, values, strict=True):
                if value == "-":
                    styled_values.append(self._text("-", "dim"))
                elif column == "best":
                    styled_values.append(self._text(value, "green"))
                else:
                    styled_values.append(self._text(str(value), "white"))
            table.add_row(self._text(label, "dim"), *styled_values)
        return table

    @staticmethod
    def _text(value: str, style: str) -> Any:
        from rich.text import Text

        return Text(value, style=style)


def create_training_console(
    *,
    total_iterations: int,
    enabled: bool = True,
) -> _RichTrainingConsole | _TqdmTrainingConsole:
    if enabled:
        try:
            import rich  # noqa: F401

            return _RichTrainingConsole(total_iterations=total_iterations)
        except Exception as exc:  # noqa: BLE001 - terminal UI fallback should keep training alive.
            _LOGGER.debug(
                "Rich training console unavailable; falling back to tqdm: %s",
                exc,
                exc_info=True,
            )
    return _TqdmTrainingConsole(total_iterations=total_iterations)


def install_compact_console_logging(*, log_dir: Path, enabled: bool = True) -> None:
    if not enabled:
        return
    root = logging.getLogger()
    train_log_path = Path(log_dir) / "train.log"
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")) == train_log_path
        for handler in root.handlers
    ):
        train_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(train_log_path)
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


def compact_console_enabled(trainer_cfg: Any) -> bool:
    value = trainer_cfg.get("console", "rich")
    if value not in {"rich", "tqdm"}:
        raise ValueError("trainer.console must be one of: rich, tqdm")
    return value == "rich"


def print_run_header(
    *,
    config_name: str,
    experiment_name: str,
    log_dir: Path,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    for line in format_run_header_lines(
        config_name=config_name,
        experiment_name=experiment_name,
        log_dir=log_dir,
    ):
        print(line, file=sys.stderr)
    print(file=sys.stderr)
