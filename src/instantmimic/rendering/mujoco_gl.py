"""One decision about which OpenGL backend MuJoCo should use.

`MUJOCO_GL` has to be set before MuJoCo initializes its renderer, which used to
mean setting it at import time: `envs/direct_rl_env.py` called into here as a
module side effect, and `instantmimic.play` pulled in a bootstrap module whose
only purpose was to run the same thing with `force=True`. Importing the playback
package therefore pinned `MUJOCO_GL=egl` in the process even when nothing was
going to render, which is invisible from the call site and impossible to opt out
of.

The resolution is a pure function plus one applier that an entrypoint calls.
"""

from __future__ import annotations

import os
import sys

DEFAULT_HEADLESS_BACKEND = "egl"


def _has_display() -> bool:
    """Whether a windowing system looks available.

    Both variables are checked because a Wayland session need not export
    `DISPLAY`, and an X session need not export `WAYLAND_DISPLAY`.
    """

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def resolve_mujoco_gl(
    *,
    force_headless: bool,
    default_backend: str | None = None,
) -> str | None:
    """The backend `MUJOCO_GL` should name, or None to leave it unset.

    Returns None when the caller's environment already chose a backend -- an
    explicit `MUJOCO_GL` always wins -- and when a display is available and the
    caller did not ask for headless.
    """

    if os.environ.get("MUJOCO_GL"):
        return None
    if not force_headless and _has_display():
        return None
    if default_backend is not None:
        return default_backend
    if sys.platform == "win32":
        return "glfw"
    if sys.platform == "darwin":
        return "cgl"
    return DEFAULT_HEADLESS_BACKEND


def apply_mujoco_gl(
    *,
    force_headless: bool,
    default_backend: str | None = None,
) -> str | None:
    """Set `MUJOCO_GL` from `resolve_mujoco_gl`, returning what was applied.

    Call this from an entrypoint, before anything builds a renderer. Importing a
    module must not do it: `MUJOCO_GL` is process-wide state, and a library
    import is the wrong place to claim it.
    """

    backend = resolve_mujoco_gl(
        force_headless=force_headless, default_backend=default_backend
    )
    if backend is not None:
        os.environ["MUJOCO_GL"] = backend
    return backend
