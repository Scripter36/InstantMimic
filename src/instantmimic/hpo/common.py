"""Small, semantics-stable helpers shared by hpo runtimes."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig

_KEYVAL_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]*)")
_WANDB_RUN_DIR_RE = re.compile(
    r"wandb: Run data is saved locally in (.*?/wandb/run-[^\s]+)"
)


def section_cfg(task_cfg: DictConfig, key: str) -> DictConfig | None:
    value = task_cfg.get(key, None)
    if isinstance(value, DictConfig):
        return value
    return None


def cfg_str(section: DictConfig | None, key: str, default: str) -> str:
    if section is None:
        return default
    value = section.get(key, None)
    if value is None:
        return default
    return str(value)


def cfg_int(section: DictConfig | None, key: str, default: int) -> int:
    if section is None:
        return default
    value = section.get(key, None)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
FALSE_TOKENS = frozenset({"0", "false", "no", "off"})


def cfg_bool(section: DictConfig | None, key: str, default: bool) -> bool:
    """Read a boolean config value, strictly.

    A recognized token is what it says; anything else is a configuration error
    rather than a silent ``bool(value)`` coercion, which used to turn the
    string ``"no"`` into ``True`` for some readers and ``False`` for others.
    """

    if section is None:
        return default
    value = section.get(key, None)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in TRUE_TOKENS:
        return True
    if normalized in FALSE_TOKENS:
        return False
    raise ValueError(
        f"'{key}' must be a boolean; got {value!r}. Accepted spellings: "
        + ", ".join(sorted(TRUE_TOKENS | FALSE_TOKENS))
        + "."
    )


def normalize_platform(value: object | None, default: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    if normalized == "codex":
        return "openai"
    return normalized


def backend_for_platform(platform: str) -> str:
    if platform in {"openai", "ollama"}:
        return "codex"
    if platform in {"claude", "opencode"}:
        return platform
    raise ValueError(f"Unsupported planning platform '{platform}'.")


def toml_literal(value: object, field_name: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (ListConfig, list)):
        return (
            "["
            + ", ".join(toml_literal(item, field_name) for item in list(value))
            + "]"
        )
    raise ValueError(
        f"Task field '{field_name}' contains unsupported value type "
        f"{type(value).__name__}. Use booleans, numbers, strings, or lists of those."
    )


def ensure_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, list):
        return value
    raise ValueError(f"'{field_name}' must be a list.")


def resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


def parse_keyval_payload(
    path: Path,
    pattern: re.Pattern[str],
    keyval_pattern: re.Pattern[str] = _KEYVAL_RE,
) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        match = pattern.search(line)
        if match is None:
            continue
        payload_text = match.group(1)
        if payload_text.lstrip().startswith("{"):
            payload = json.loads(payload_text)
            if not isinstance(payload, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in payload.items()
            ):
                raise TypeError("Machine JSON result must map strings to strings.")
            return payload
        return {key: value for key, value in keyval_pattern.findall(payload_text)}
    return {}


def parse_wandb_run_dir(
    path: Path,
    pattern: re.Pattern[str] = _WANDB_RUN_DIR_RE,
) -> Path | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        match = pattern.search(line)
        if match is None:
            continue
        return Path(match.group(1)).resolve().parents[1]
    return None


def safe_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def slugify(text: str, *, empty_fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", text).strip("_")
    return slug or empty_fallback


def default_play_overrides(config_file: Path, config_root: Path) -> list[str]:
    fast_muscle_roots = (
        (config_root / "snu_humanoid" / "imitation").resolve(),
        (config_root / "snu_humanoid" / "lafan").resolve(),
    )
    if any(config_file.resolve().is_relative_to(root) for root in fast_muscle_roots):
        return [
            "++play.tracking.mode=reference",
            "++play.tracking.deterministic_reset=true",
        ]
    return []


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def format_scalar(value: Any) -> str:
    """Render one override value the way Hydra parses it back."""

    if isinstance(value, (ListConfig, list, tuple)):
        return "[" + ",".join(format_scalar(item) for item in list(value)) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def resolve_config_path_arg(config_path: str | None) -> str | None:
    """Absolutize a `--config-path` argument against the repository root."""

    if not config_path:
        return None
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((Path(__file__).resolve().parents[3] / path).resolve())
