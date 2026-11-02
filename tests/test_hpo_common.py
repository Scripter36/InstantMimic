from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from instantmimic.hpo import agent_runtime, evaluate, tracking
from instantmimic.hpo import common as search_common
from instantmimic.hpo.common import (
    backend_for_platform,
    cfg_bool,
    cfg_int,
    cfg_str,
    ensure_list,
    normalize_platform,
    resolve_repo_path,
    section_cfg,
    toml_literal,
    utc_now,
)


def test_shared_config_helpers_preserve_runtime_contracts() -> None:
    config = OmegaConf.create(
        {
            "runtime": {
                "enabled": " yes ",
                "count": "7",
                "label": "  keep surrounding whitespace  ",
            }
        }
    )
    section = section_cfg(config, "runtime")

    assert section is not None
    assert cfg_bool(section, "enabled", False) is True
    assert cfg_int(section, "count", 3) == 7
    assert cfg_str(section, "label", "fallback") == "  keep surrounding whitespace  "
    assert cfg_str(section, "missing", "fallback") == "fallback"
    assert section_cfg(config, "missing") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "true"),
        (False, "false"),
        (4, "4"),
        (1.5, "1.5"),
        ("quoted", '"quoted"'),
        (["a", 2], '["a", 2]'),
    ],
)
def test_toml_literal_serializes_supported_values(value: object, expected: str) -> None:
    assert toml_literal(value, "agent.value") == expected


def test_toml_literal_rejects_unsupported_values() -> None:
    with pytest.raises(ValueError, match="agent.value.*unsupported value type"):
        toml_literal(object(), "agent.value")


def test_platform_helpers_normalize_alias_and_select_backend() -> None:
    assert normalize_platform(" CODEx ", "openai") == "openai"
    assert normalize_platform("", "openai") == "openai"
    assert backend_for_platform("openai") == "codex"
    assert backend_for_platform("claude") == "claude"
    with pytest.raises(ValueError, match="Unsupported planning platform"):
        backend_for_platform("unknown")


def test_list_and_path_helpers_are_root_independent() -> None:
    config = OmegaConf.create({"values": [1, 2]})
    assert ensure_list(config["values"], "values") == [1, 2]
    assert ensure_list(None, "values") == []
    with pytest.raises(ValueError, match="values.*must be a list"):
        ensure_list("not-a-list", "values")

    root = Path("/tmp/test-hpo-root")
    assert (
        resolve_repo_path("relative/file.yaml", root)
        == (root / "relative/file.yaml").resolve()
    )
    absolute = Path("/tmp/test-hpo-absolute.yaml")
    assert resolve_repo_path(absolute, root) == absolute.resolve()


def test_utc_now_is_iso8601_with_timezone() -> None:
    value = utc_now()
    assert value.endswith("+00:00")
    assert "T" in value


def test_existing_private_wrappers_delegate_to_shared_helpers() -> None:
    section = OmegaConf.create({"enabled": "true"})
    assert agent_runtime._cfg_bool(section, "enabled", False) == cfg_bool(
        section, "enabled", False
    )
    assert evaluate._ensure_list([1], "values") == ensure_list([1], "values")
    assert evaluate._resolve_repo_path("logs") == resolve_repo_path(
        "logs", evaluate.REPO_ROOT
    )
    assert tracking._section_cfg(OmegaConf.create({"x": {}}), "x") is not None


def test_same_name_formatters_remain_semantically_distinct() -> None:
    assert search_common.format_scalar("text") == '"text"'
    assert evaluate._format_scalar("text") == "text"
    assert search_common.format_scalar(["text", 1]) == '["text",1]'
    assert evaluate._format_scalar(["text", 1]) == "['text', 1]"


def test_one_boolean_reader_and_it_refuses_a_value_it_cannot_read() -> None:
    """There were two ``cfg_bool``s, and they disagreed.

    ``tracking``'s copy read anything outside its true-list as ``False``;
    ``common``'s fell through to ``bool(value)``, so the string ``"no"`` was
    ``False`` in one and ``True`` in the other. There is one now, and a value
    it does not recognize is a configuration error rather than a coin flip.
    """

    assert not hasattr(tracking, "_cfg_bool")
    for token, expected in (
        ("no", False),
        ("off", False),
        ("YES", True),
        (" on ", True),
    ):
        section = OmegaConf.create({"flag": token})
        assert cfg_bool(section, "flag", not expected) is expected

    with pytest.raises(ValueError, match="must be a boolean"):
        cfg_bool(OmegaConf.create({"flag": "unrecognized"}), "flag", True)


def test_tracking_string_parsing_keeps_its_strict_contract() -> None:
    """``cfg_str`` still diverges on purpose: a label is not a flag.

    A whitespace-only tracking label falls back to the default, while a
    whitespace-padded value elsewhere is passed through as authored.
    """

    section = OmegaConf.create({"label": "   "})
    assert tracking._cfg_str(section, "label", "fallback") == "fallback"
    assert cfg_str(section, "label", "fallback") == "   "
