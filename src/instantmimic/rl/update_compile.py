"""Where the update compile-and-capture decision comes from.

The learner compiles its update bodies and captures them into an external
CUDA graph. The compiled region and the update graph are two halves of the same
execution mode, and the capture exists to replay the compiled launch sequence
without Python. Neither half is configured separately, so no runtime shows the
derived condition as a key. Compile strictness is not derived here: PPO
compiles its update wrappers with the default ``fullgraph=False``.
"""

from __future__ import annotations


def update_fullgraph_compile_enabled(
    *,
    compile_enabled: bool,
    update_graph_enabled: bool,
) -> bool:
    """Whether the update bodies are compiled and captured together."""
    return bool(compile_enabled) and bool(update_graph_enabled)


__all__ = ["update_fullgraph_compile_enabled"]
