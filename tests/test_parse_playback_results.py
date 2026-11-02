from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "instantmimic-workflows"
    / "scripts"
    / "parse_playback_results.py"
)


def _load_parser_module():
    spec = importlib.util.spec_from_file_location(
        "parse_playback_results_test_module",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_text_associates_each_result_with_its_nearby_video() -> None:
    parser = _load_parser_module()
    text = (
        "Saved video to: /tmp/first.mp4\n"
        "PLAYBACK_RESULT checkpoint=first reward=1\n"
        "Saved video to: /tmp/second.mp4\n"
        "PLAYBACK_RESULT checkpoint=second reward=2\n"
    )

    rows = parser.parse_text(text, "batch.log")

    assert [(row["checkpoint"], row["video"]) for row in rows] == [
        ("first", "/tmp/first.mp4"),
        ("second", "/tmp/second.mp4"),
    ]


def test_structured_result_replaces_paired_legacy_row_without_losing_paths() -> None:
    import json

    parser = _load_parser_module()
    payload = {
        "checkpoint": "/tmp/my run/best.pt",
        "video_output_path": "/tmp/my run/play.mp4",
    }
    text = (
        "PLAYBACK_RESULT checkpoint=/tmp/my run/best.pt\n"
        "PLAYBACK_RESULT_JSON " + json.dumps(payload) + "\n"
    )
    rows = parser.parse_text(text, "play.log")
    assert len(rows) == 1
    assert rows[0]["checkpoint"] == payload["checkpoint"]
    assert rows[0]["video"] == payload["video_output_path"]


def test_current_playback_api_reward_survives_csv_and_markdown(capsys) -> None:
    from instantmimic.play.runtime import PlaybackResult

    parser = _load_parser_module()
    result = PlaybackResult(
        checkpoint_path=Path("/tmp/current/best_agent.pt"),
        episode_reward=12.5,
        episode_length=200,
    )
    rows = parser.parse_text(result.to_json_line(), "current.log")
    parser._print_csv(rows)
    assert "12.500000" in capsys.readouterr().out
    parser._print_markdown(rows)
    assert "12.500000" in capsys.readouterr().out


def test_numeric_json_values_render_in_markdown_without_losing_reward(capsys) -> None:
    parser = _load_parser_module()
    rows = parser.parse_text(
        'PLAYBACK_RESULT_JSON {"checkpoint_path":"a|b.pt",'
        '"episode_reward":1.25,"episode_length":24}\n',
        "numeric.log",
    )
    parser._print_markdown(rows)
    rendered = capsys.readouterr().out
    assert "a\\|b.pt" in rendered
    assert "| 1.25 | 24 |" in rendered
