#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

RESULT_RE = re.compile(r"PLAYBACK_RESULT(_JSON)?[ \t]+(.*)")
VIDEO_RE = re.compile(r"Saved video to:\s+(.*)")
KEYVAL_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]*)")


def _parse_payload(payload: str) -> dict[str, str]:
    return {key: value for key, value in KEYVAL_RE.findall(payload)}


def parse_text(text: str, source: str) -> list[dict[str, Any]]:
    events: list[tuple[int, str, str]] = [
        (match.start(), "video", match.group(1).strip())
        for match in VIDEO_RE.finditer(text)
    ]
    events.extend(
        (match.start(), "json" if match.group(1) else "result", match.group(2))
        for match in RESULT_RE.finditer(text)
    )
    events.sort(key=lambda event: event[0])

    last_video = ""
    rows: list[dict[str, Any]] = []
    previous_event_type = ""
    for _, event_type, payload in events:
        if event_type == "video":
            last_video = payload
            previous_event_type = event_type
            continue

        row = json.loads(payload) if event_type == "json" else _parse_payload(payload)
        if not isinstance(row, dict):
            raise TypeError("playback result must be a mapping")
        if "episode_reward" in row:
            row["reward"] = row["episode_reward"]
        row.setdefault("source", source)
        row.setdefault("video", row.get("video_output_path", last_video))
        if event_type == "json" and previous_event_type == "result":
            rows[-1] = row
        else:
            rows.append(row)
        previous_event_type = event_type
    return rows


def _read_sources(paths: list[str]) -> list[tuple[str, str]]:
    if not paths:
        return [("stdin", sys.stdin.read())]
    sources: list[tuple[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        sources.append(
            (path.as_posix(), path.read_text(encoding="utf-8", errors="replace"))
        )
    return sources


def _print_csv(rows: list[dict[str, Any]]) -> None:
    columns = [
        "source",
        "checkpoint",
        "checkpoint_path",
        "reward",
        "episode_length",
        "video",
    ]
    writer = csv.writer(sys.stdout)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row.get(column, "") for column in columns])


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = ["source", "checkpoint", "reward", "episode_length", "video"]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        checkpoint = row.get("checkpoint") or row.get("checkpoint_path", "")
        values = [
            row.get("source", ""),
            checkpoint,
            row.get("reward", ""),
            row.get("episode_length", ""),
            row.get("video", ""),
        ]
        print(
            "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse PLAYBACK_RESULT lines from logs."
    )
    parser.add_argument("paths", nargs="*", help="Log files. Reads stdin when omitted.")
    parser.add_argument("--markdown", action="store_true", help="Print markdown table")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for source, text in _read_sources(args.paths):
        rows.extend(parse_text(text, source))
    if args.markdown:
        _print_markdown(rows)
    else:
        _print_csv(rows)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
