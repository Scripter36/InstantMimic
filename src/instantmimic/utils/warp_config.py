"""Repository Warp settings, applied once per process before ``wp.init()``."""

from __future__ import annotations

import os

import warp as wp

from instantmimic.utils.project_paths import user_cache_root

_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _optional_env_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(_FALSE_VALUES | _TRUE_VALUES))
    raise ValueError(f"{name} must be one of: {accepted}")


def configure_warp_launch_checks() -> None:
    """Require every Warp launch array to live on the launch device.

    Warp 1.14 made ``RELAXED`` the default, which skips this check so GPU
    kernels can read pageable CPU memory on HMM and ATS systems. On a discrete
    GPU with HMM that access migrates pages through the driver instead of
    failing, and ``CHECKED`` still classifies pageable memory as accessible.
    ``STRICT`` restores the pre-1.14 error. See
    ``docs/reference/mujoco-warp.md``.
    """

    wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.STRICT


def configure_warp_from_env(default_kernel_cache_dir: str | None = None) -> None:
    """Apply Warp settings from the environment.

    The kernel cache goes to ``WARP_CACHE_DIR``, then ``WP_CACHE_DIR``, then
    ``default_kernel_cache_dir``, then ``user_cache_root("warp")``. The last of
    those keeps the cache per user under ``XDG_CACHE_HOME`` rather than in a
    world-writable temporary directory, and keeps it across reboots. Launch
    array checks are always strict; see ``configure_warp_launch_checks``.
    """

    configure_warp_launch_checks()

    quiet = _optional_env_bool("WARP_QUIET")
    wp.config.quiet = True if quiet is None else quiet

    cache_dir = (
        os.environ.get("WARP_CACHE_DIR")
        or os.environ.get("WP_CACHE_DIR")
        or default_kernel_cache_dir
        or str(user_cache_root("warp"))
    )
    wp.config.kernel_cache_dir = cache_dir

    lineinfo = _optional_env_bool("WARP_LINEINFO")
    if lineinfo is not None:
        wp.config.lineinfo = lineinfo

    line_directives = _optional_env_bool("WARP_LINE_DIRECTIVES")
    if line_directives is not None:
        wp.config.line_directives = line_directives


def initialize_warp_runtime(default_kernel_cache_dir: str | None = None) -> None:
    """Apply repository Warp settings before idempotent runtime initialization."""
    configure_warp_from_env(default_kernel_cache_dir)
    wp.init()
