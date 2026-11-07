"""Float32 matmul precision scopes for learning networks.

Environment, physics, observation, reward and asset-construction code always
runs IEEE float32. CUDA TF32 is enabled only while a policy, value, critic,
world model or training update is executing, through
``network_matmul_precision``.

PyTorch 2.11 exposes two ways to select float32 matmul precision. The new
per-backend ``torch.backends.cuda.matmul.fp32_precision`` setter, when given
``"tf32"``, leaves the legacy state unset and makes
``torch.get_float32_matmul_precision()`` raise a mixed-API ``RuntimeError``.
Inductor reads that getter for its autotune cache keys, so this module uses the
legacy ``torch.set_float32_matmul_precision`` setter, which updates both states
consistently. cuDNN convolution and RNN precision use the new per-operator API.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

IEEE_MATMUL_PRECISION = "highest"
NETWORK_MATMUL_PRECISION = "high"


def _explicit_cudnn_precision(precision: str, *, allow_tf32: bool) -> str:
    # torch 2.11 can leave a per-operator value at "none" while the legacy
    # process switch carries the effective setting. Make that state explicit
    # so the legacy and new getters stay consistent.
    if precision != "none":
        return precision
    return "tf32" if allow_tf32 else "ieee"


def _cudnn_state() -> tuple[str, str, bool]:
    conv = torch.backends.cudnn.conv.fp32_precision
    rnn = torch.backends.cudnn.rnn.fp32_precision
    try:
        allow_tf32 = bool(torch.backends.cudnn.allow_tf32)
    except RuntimeError:
        # A library may have left only the per-operator values set. The
        # explicit conv value still determines the effective setting.
        allow_tf32 = conv != "ieee"
    conv = _explicit_cudnn_precision(conv, allow_tf32=allow_tf32)
    rnn = _explicit_cudnn_precision(rnn, allow_tf32=allow_tf32)
    return conv, rnn, allow_tf32


def _set_cudnn_state(conv: str, rnn: str, allow_tf32: bool) -> None:
    # The legacy bool sets both operators to "none"; the explicit per-operator
    # values below then make the state readable through either API.
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.conv.fp32_precision = conv
    torch.backends.cudnn.rnn.fp32_precision = rnn


def configure_ieee_float32_precision() -> None:
    """Restore the process-wide IEEE float32 default.

    Environment and asset code must not inherit TF32 from a previous run or
    from a library that raised the process default.
    """
    torch.set_float32_matmul_precision(IEEE_MATMUL_PRECISION)
    _set_cudnn_state("ieee", "ieee", False)


@contextmanager
def network_matmul_precision() -> Iterator[None]:
    """Enable TF32 for network float32 matmuls and cuDNN convolutions.

    The previous process state is restored on every exit, including when the
    scoped body raises. Nested scopes preserve the outer network state.
    """
    previous_matmul = torch.get_float32_matmul_precision()
    previous_conv, previous_rnn, previous_allow = _cudnn_state()
    try:
        torch.set_float32_matmul_precision(NETWORK_MATMUL_PRECISION)
        _set_cudnn_state("tf32", "tf32", True)
        yield
    finally:
        try:
            torch.set_float32_matmul_precision(previous_matmul)
        finally:
            _set_cudnn_state(previous_conv, previous_rnn, previous_allow)


__all__ = [
    "IEEE_MATMUL_PRECISION",
    "NETWORK_MATMUL_PRECISION",
    "configure_ieee_float32_precision",
    "network_matmul_precision",
]
