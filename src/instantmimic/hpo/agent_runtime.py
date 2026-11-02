"""Shared planning-agent configuration and execution contract."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from instantmimic.hpo.common import (
    backend_for_platform as _common_backend_for_platform,
)
from instantmimic.hpo.common import (
    cfg_bool as _common_cfg_bool,
)
from instantmimic.hpo.common import (
    cfg_str as _common_cfg_str,
)
from instantmimic.hpo.common import (
    normalize_platform as _common_normalize_platform,
)
from instantmimic.hpo.common import (
    section_cfg as _common_section_cfg,
)
from instantmimic.hpo.common import (
    toml_literal as _common_toml_literal,
)
from instantmimic.hpo.evaluator_launch import command_tokens

_DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11435"
_OPENCODE_PROMPT_FILE_MESSAGE = (
    "Read the attached prompt file and follow its instructions exactly."
)
_SUPPORTED_AGENT_BACKENDS = {"codex", "claude", "opencode"}
_SUPPORTED_CODEX_SANDBOX_MODES = {
    "read-only",
    "workspace-write",
    "danger-full-access",
}
_SUPPORTED_OPENCODE_OUTPUT_FORMATS = {"default", "json"}
_SUPPORTED_PLANNING_PLATFORMS = {"openai", "claude", "ollama", "opencode"}


@dataclass(frozen=True)
class AgentSpec:
    """Resolved command-line agent configuration shared by cycle controllers."""

    backend: str
    platform: str
    model: str | None
    reasoning_effort: str
    sandbox: str = "workspace-write"
    command: tuple[str, ...] = ("codex",)
    codex_config_overrides: tuple[tuple[str, str], ...] = ()
    ollama_host: str | None = None
    auto_start_ollama: bool = False
    opencode_agent: str | None = None
    opencode_variant: str | None = None
    opencode_format: str = "default"
    opencode_pure: bool = False
    opencode_skip_permissions: bool = True

    @property
    def uses_ollama(self) -> bool:
        return self.platform == "ollama"

    @property
    def uses_claude(self) -> bool:
        return self.backend == "claude"

    @property
    def uses_opencode(self) -> bool:
        return self.backend == "opencode"


def agent_label(agent: AgentSpec) -> str:
    if agent.model:
        return f"{agent.platform}:{agent.model}"
    return agent.platform


def _section_cfg(task_cfg: DictConfig, key: str) -> DictConfig | None:
    return _common_section_cfg(task_cfg, key)


def _cfg_bool(section: DictConfig | None, key: str, default: bool) -> bool:
    return _common_cfg_bool(section, key, default)


def _cfg_str(section: DictConfig | None, key: str, default: str) -> str:
    return _common_cfg_str(section, key, default)


def _normalize_platform(value: object | None, default: str) -> str:
    return _common_normalize_platform(value, default)


def _backend_for_platform(platform: str) -> str:
    return _common_backend_for_platform(platform)


def _resolved_ollama_host(value: object | None) -> str:
    host = str(value or _DEFAULT_OLLAMA_HOST).strip()
    if not host:
        host = _DEFAULT_OLLAMA_HOST
    return host.rstrip("/")


def _toml_literal(value: object, field_name: str) -> str:
    return _common_toml_literal(value, field_name)


def _codex_config_overrides(
    section: DictConfig | None,
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    if section is None:
        return ()
    raw = section.get("codex_config", None)
    if raw is None:
        return ()
    if not isinstance(raw, (DictConfig, dict)):
        raise ValueError(f"Task field '{field_name}' must be a mapping when present.")

    entries: list[tuple[str, str]] = []
    for key, value in raw.items():
        if value is None:
            continue
        key_text = str(key).strip()
        if not key_text:
            raise ValueError(
                f"Task field '{field_name}' must not contain an empty key."
            )
        entries.append((key_text, _toml_literal(value, f"{field_name}.{key_text}")))
    return tuple(entries)


def resolve_agent_spec(
    task_cfg: DictConfig,
    section_name: str,
    *,
    default_model: str | None,
    default_reasoning_effort: str,
) -> AgentSpec:
    """Resolve one cycle's agent section without depending on a cycle module."""

    section = _section_cfg(task_cfg, section_name)
    platform_value = section.get("platform", None) if section is not None else None
    if section is not None and section.get("provider", None) is not None:
        raise ValueError(
            f"Task section '{section_name}' uses unsupported field 'provider'. "
            f"Use '{section_name}.platform' as the single source of truth."
        )
    if (
        platform_value is None
        and section is not None
        and section.get("local_llm", None) is not None
    ):
        platform_value = (
            "ollama" if _cfg_bool(section, "local_llm", False) else "openai"
        )
    platform = _normalize_platform(platform_value, "openai")
    if platform not in _SUPPORTED_PLANNING_PLATFORMS:
        supported = ", ".join(sorted(_SUPPORTED_PLANNING_PLATFORMS))
        raise ValueError(
            f"Task section '{section_name}' has unsupported platform '{platform}'. "
            f"Supported platforms: {supported}."
        )
    backend = _backend_for_platform(platform)

    model_value = section.get("model", None) if section is not None else None
    model = default_model if model_value is None else str(model_value).strip() or None
    if platform in {"ollama", "opencode"} and model is None:
        raise ValueError(
            f"Task section '{section_name}' must set 'model' when platform='{platform}'."
        )
    reasoning_effort = _cfg_str(section, "reasoning_effort", default_reasoning_effort)
    sandbox = _cfg_str(section, "sandbox", "workspace-write")
    if sandbox not in _SUPPORTED_CODEX_SANDBOX_MODES:
        supported = ", ".join(sorted(_SUPPORTED_CODEX_SANDBOX_MODES))
        raise ValueError(
            f"Task section '{section_name}' has unsupported sandbox '{sandbox}'. "
            f"Supported sandbox modes: {supported}."
        )

    command = command_tokens(
        section.get("command", None) if section is not None else None,
        f"{section_name}.command",
    )
    if not command:
        command = {
            "claude": ("claude",),
            "ollama": ("ollama", "launch", "codex"),
            "opencode": ("opencode", "run"),
        }.get(platform, ("codex",))

    codex_config_overrides = _codex_config_overrides(
        section, f"{section_name}.codex_config"
    )
    ollama_host = None
    auto_start_ollama = False
    if platform == "ollama":
        ollama_host = _resolved_ollama_host(
            section.get("host", None) if section is not None else None
        )
        auto_start_ollama = _cfg_bool(section, "auto_start_ollama", True)

    opencode_agent = None
    opencode_variant = None
    opencode_format = "default"
    opencode_pure = False
    opencode_skip_permissions = True
    if platform == "opencode":
        agent_value = section.get("agent", None) if section is not None else None
        if agent_value not in (None, ""):
            opencode_agent = str(agent_value)
        variant_value = section.get("variant", None) if section is not None else None
        if variant_value not in (None, ""):
            opencode_variant = str(variant_value)
        elif reasoning_effort:
            opencode_variant = str(reasoning_effort)
        opencode_format = _cfg_str(section, "output_format", "default")
        if opencode_format not in _SUPPORTED_OPENCODE_OUTPUT_FORMATS:
            supported = ", ".join(sorted(_SUPPORTED_OPENCODE_OUTPUT_FORMATS))
            raise ValueError(
                f"Task section '{section_name}' has unsupported output_format "
                f"'{opencode_format}'. Supported formats: {supported}."
            )
        opencode_pure = _cfg_bool(section, "pure", False)
        opencode_skip_permissions = _cfg_bool(section, "skip_permissions", True)

    return AgentSpec(
        backend=backend,
        platform=platform,
        model=model,
        reasoning_effort=reasoning_effort,
        sandbox=sandbox,
        command=command,
        codex_config_overrides=codex_config_overrides,
        ollama_host=ollama_host,
        auto_start_ollama=auto_start_ollama,
        opencode_agent=opencode_agent,
        opencode_variant=opencode_variant,
        opencode_format=opencode_format,
        opencode_pure=opencode_pure,
        opencode_skip_permissions=opencode_skip_permissions,
    )


def _build_codex_command(
    root: Path,
    message_path: Path,
    platform: str,
    model: str | None,
    reasoning_effort: str,
    sandbox: str,
    command_prefix: tuple[str, ...],
    codex_config_overrides: tuple[tuple[str, str], ...],
) -> list[str]:
    """Build a Codex-compatible CLI invocation.

    The prompt itself is not an argument: Codex reads it from `message_path`.
    """

    codex_args = ["-s", sandbox, "-a", "never"]
    for key, value in codex_config_overrides:
        codex_args.extend(["-c", f"{key}={value}"])
    codex_args.extend(
        [
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "exec",
            "--add-dir",
            "/tmp",
            "-C",
            str(root),
            "-o",
            str(message_path),
        ]
    )
    if platform == "openai" and model is not None:
        codex_args.extend(["-m", model])
    codex_args.append("-")

    if platform == "ollama":
        command = [*command_prefix]
        if model is not None:
            command.extend(["--model", model])
        command.append("--")
        command.extend(codex_args)
        return command
    return [*command_prefix, *codex_args]


def _build_claude_command(
    prompt: str,
    model: str | None,
    command_prefix: tuple[str, ...],
) -> list[str]:
    claude_args = ["-p", "--permission-mode", "bypassPermissions", "--add-dir", "/tmp"]
    if model is not None:
        claude_args.extend(["--model", model])
    claude_args.append(prompt)
    return [*command_prefix, *claude_args]


def _build_opencode_command(
    *,
    root: Path,
    prompt_file: Path,
    model: str | None,
    command_prefix: tuple[str, ...],
    agent_name: str | None,
    variant: str | None,
    output_format: str,
    use_pure: bool,
    skip_permissions: bool,
) -> list[str]:
    opencode_args = ["--dir", str(root), "--format", output_format]
    if model is not None:
        opencode_args.extend(["--model", model])
    if agent_name is not None:
        opencode_args.extend(["--agent", agent_name])
    if variant is not None:
        opencode_args.extend(["--variant", variant])
    if use_pure:
        opencode_args.append("--pure")
    if skip_permissions:
        opencode_args.append("--dangerously-skip-permissions")
    opencode_args.append(_OPENCODE_PROMPT_FILE_MESSAGE)
    opencode_args.extend(["--file", str(prompt_file)])
    return [*command_prefix, *opencode_args]


def run_agent(
    *,
    root: Path,
    prompt: str,
    message_path: Path,
    agent: AgentSpec,
    timeout_seconds: float,
    dry_run: bool,
    log_prefix: str = "[hpo:planner]",
) -> int:
    """Run a resolved planning agent and persist its response."""

    if agent.backend == "claude":
        command = _build_claude_command(prompt, agent.model, agent.command)
    elif agent.backend == "codex":
        command = _build_codex_command(
            root,
            message_path,
            agent.platform,
            agent.model,
            agent.reasoning_effort,
            agent.sandbox,
            agent.command,
            agent.codex_config_overrides,
        )
    elif agent.backend == "opencode":
        prompt_file = message_path.with_name(f"{message_path.stem}.prompt.md")
        prompt_file.write_text(prompt, encoding="utf-8")
        command = _build_opencode_command(
            root=root,
            prompt_file=prompt_file,
            model=agent.model,
            command_prefix=agent.command,
            agent_name=agent.opencode_agent,
            variant=agent.opencode_variant,
            output_format=agent.opencode_format,
            use_pure=agent.opencode_pure,
            skip_permissions=agent.opencode_skip_permissions,
        )
    else:
        supported = ", ".join(sorted(_SUPPORTED_AGENT_BACKENDS))
        raise ValueError(
            f"Unsupported agent backend '{agent.backend}'. Supported backends: {supported}."
        )
    print(f"{log_prefix} command={shlex.join(command)}")

    if dry_run:
        message_path.parent.mkdir(parents=True, exist_ok=True)
        message_path.write_text(
            "# Dry-run placeholder\n\n"
            "The agent command was inspected but not executed.\n",
            encoding="utf-8",
        )
        return 0

    run_env = None
    if agent.platform == "ollama" and agent.ollama_host is not None:
        run_env = dict(os.environ)
        run_env["OLLAMA_HOST"] = agent.ollama_host

    timeout = None if timeout_seconds <= 0 else timeout_seconds
    try:
        if agent.backend in {"claude", "opencode"}:
            completed = subprocess.run(
                command,
                check=False,
                timeout=timeout,
                env=run_env,
                cwd=str(root),
                capture_output=True,
                text=True,
            )
            message_path.write_text(completed.stdout or "", encoding="utf-8")
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
        else:
            completed = subprocess.run(
                command,
                check=False,
                timeout=timeout,
                env=run_env,
                input=prompt,
                text=True,
            )
    except subprocess.TimeoutExpired:
        with message_path.open("a", encoding="utf-8") as message_file:
            message_file.write(
                f"\n{log_prefix} planner timed out after {timeout_seconds:.1f}s\n"
            )
        return 124
    return int(completed.returncode)
