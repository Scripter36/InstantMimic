"""Specialized MuJoCo Warp kernels selected by ``env.use_optimized_kernel``.

Each module exposes structural eligibility and an upstream-shaped fallback;
``mjw_step_optimized`` owns orchestration and persistent scratch.
"""
