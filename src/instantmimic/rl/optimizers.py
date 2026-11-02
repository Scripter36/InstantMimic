"""Graph-compatible optimizers shared by the learning runtimes."""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch

_MUON_DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_MUON_DEFAULT_EPS = 1.0e-7
_MUON_DEFAULT_NS_STEPS = 5
_MUON_COMPILE_MODE = "max-autotune-no-cudagraphs"
_SUPPORTED_OPTIMIZER_NAMES = ("adam", "adamw", "sgd", "muon")
_COMPILED_MUON_NS_SINGLE = None
_COMPILED_MUON_NS_BATCHED = None
_LOGGER = logging.getLogger(__name__)


def _is_muon_matrix_param(param: torch.nn.Parameter) -> bool:
    return bool(param.requires_grad and param.ndim == 2 and min(param.shape) > 1)


def _normalize_optimizer_params(
    params: Iterable[torch.nn.Parameter] | Iterable[tuple[str, torch.nn.Parameter]],
) -> list[tuple[str | None, torch.nn.Parameter]]:
    normalized: list[tuple[str | None, torch.nn.Parameter]] = []
    for item in params:
        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeError(
                    "Named optimizer parameters must be (name, parameter) pairs."
                )
            name, param = item
            if not isinstance(name, str) or not isinstance(param, torch.nn.Parameter):
                raise TypeError(
                    "Named optimizer parameters must be (str, torch.nn.Parameter) pairs."
                )
            normalized.append((name, param))
            continue
        if not isinstance(item, torch.nn.Parameter):
            raise TypeError(
                "Optimizer parameters must be torch.nn.Parameter instances or "
                "(name, parameter) pairs."
            )
        normalized.append((None, item))
    return normalized


def _trunk_module_index(name: str) -> int | None:
    parts = name.split(".")
    if not parts or parts[-1] != "weight":
        return None
    for index, part in enumerate(parts[:-1]):
        is_trunk = part == "trunk" or part.endswith("_trunk")
        if is_trunk and index + 1 < len(parts) - 1:
            module_index = parts[index + 1]
            if module_index.isdigit():
                return int(module_index)
    return None


def _is_residual_block_weight(name: str) -> bool:
    # Residual backbones (e.g. FlashSAC) name hidden matrices
    # ``...blocks.<index>.<layer>...weight``; their embedding and heads live
    # outside ``blocks``.
    parts = name.split(".")
    if not parts or parts[-1] != "weight":
        return False
    return any(
        part == "blocks" and parts[index + 1].isdigit()
        for index, part in enumerate(parts[:-2])
    )


def _is_muon_hidden_layer_param(
    name: str | None,
    param: torch.nn.Parameter,
) -> bool:
    if not _is_muon_matrix_param(param):
        return False
    if name is None:
        return True
    parts = name.split(".")
    if len(parts) >= 2 and parts[-2].endswith("head"):
        return False
    if _is_residual_block_weight(name):
        return True
    trunk_index = _trunk_module_index(name)
    return trunk_index is not None and trunk_index > 0


def resolve_optimizer_name(optimizer_name: str | None) -> str:
    key = "adam" if optimizer_name is None else str(optimizer_name).strip().lower()
    if key in _SUPPORTED_OPTIMIZER_NAMES:
        return key
    supported = ", ".join(
        name.upper() if name != "sgd" else "SGD" for name in _SUPPORTED_OPTIMIZER_NAMES
    )
    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}. Use one of: {supported}."
    )


def validate_graph_optimizer_constraints(
    optimizer_name: str,
    optimizer_kwargs: Mapping[str, Any],
    *,
    use_update_graph: bool,
) -> None:
    if not use_update_graph:
        return

    if optimizer_name not in {"adam", "adamw", "muon"}:
        raise ValueError(
            "runtime.update_cuda_graph=true requires optimizer=Adam, AdamW, or Muon. "
            "PyTorch SGD is not capturable in this path."
        )

    if optimizer_name == "muon":
        if optimizer_kwargs.get("adamw_foreach") is True:
            raise ValueError(
                "runtime.update_cuda_graph=true does not support "
                "optimizer_kwargs.adamw_foreach=true."
            )
        return

    if optimizer_kwargs.get("foreach") is True:
        raise ValueError(
            "runtime.update_cuda_graph=true does not support optimizer foreach mode."
        )


def _weight_decay_groups(
    params: list[torch.nn.Parameter],
    weight_decay: float,
    *,
    decay_vector_parameters: bool,
) -> list[torch.nn.Parameter] | list[dict[str, Any]]:
    if decay_vector_parameters:
        return params
    # Decay only matrix weights; biases, norm gains, and log-std vectors keep
    # their scale.
    decay = [param for param in params if param.ndim >= 2]
    no_decay = [param for param in params if param.ndim < 2]
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def build_optimizer(
    *,
    params: Iterable[torch.nn.Parameter] | Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_name: str,
    learning_rate: float,
    optimizer_kwargs: Mapping[str, Any] | None,
    capturable: bool,
    device: torch.device | str,
    decay_vector_parameters: bool = True,
) -> torch.optim.Optimizer | HybridMuonAdamW:
    """Build the configured optimizer.

    Muon parameter selection uses parameter names, so pass
    ``named_parameters()`` pairs. ``decay_vector_parameters=False`` exempts
    parameters with fewer than two dimensions from weight decay.
    """
    resolved_name = resolve_optimizer_name(optimizer_name)
    local_kwargs = dict(optimizer_kwargs or {})
    named_parameter_list = _normalize_optimizer_params(params)
    parameter_list = [param for _, param in named_parameter_list]

    if resolved_name in {"adam", "adamw"}:
        optimizer_lr: float | torch.Tensor = learning_rate
        if capturable:
            optimizer_lr = torch.tensor(
                learning_rate, device=device, dtype=torch.float32
            )

        kwargs: dict[str, Any] = {
            "lr": optimizer_lr,
            "betas": tuple(local_kwargs.pop("betas", (0.9, 0.999))),
            "eps": float(local_kwargs.pop("eps", 1.0e-8)),
            "weight_decay": float(
                local_kwargs.pop(
                    "weight_decay", 0.0 if resolved_name == "adam" else 0.01
                )
            ),
            "amsgrad": bool(local_kwargs.pop("amsgrad", False)),
            "capturable": capturable,
        }
        if "maximize" in local_kwargs:
            kwargs["maximize"] = bool(local_kwargs.pop("maximize"))
        if "differentiable" in local_kwargs:
            kwargs["differentiable"] = bool(local_kwargs.pop("differentiable"))
        if resolved_name == "adam" and "decoupled_weight_decay" in local_kwargs:
            kwargs["decoupled_weight_decay"] = bool(
                local_kwargs.pop("decoupled_weight_decay")
            )
        if "foreach" in local_kwargs:
            kwargs["foreach"] = local_kwargs.pop("foreach")
        if "fused" in local_kwargs:
            kwargs["fused"] = local_kwargs.pop("fused")
        if local_kwargs:
            raise ValueError(
                f"Unsupported optimizer_kwargs for {resolved_name.upper()}: "
                f"{sorted(local_kwargs.keys())}"
            )
        optimizer_cls = (
            torch.optim.AdamW if resolved_name == "adamw" else torch.optim.Adam
        )
        return optimizer_cls(
            _weight_decay_groups(
                parameter_list,
                kwargs["weight_decay"],
                decay_vector_parameters=decay_vector_parameters,
            ),
            **kwargs,
        )

    if resolved_name == "sgd":
        kwargs = {
            "lr": float(learning_rate),
            "momentum": float(local_kwargs.pop("momentum", 0.0)),
            "dampening": float(local_kwargs.pop("dampening", 0.0)),
            "weight_decay": float(local_kwargs.pop("weight_decay", 0.0)),
            "nesterov": bool(local_kwargs.pop("nesterov", False)),
        }
        if "foreach" in local_kwargs:
            kwargs["foreach"] = local_kwargs.pop("foreach")
        if "fused" in local_kwargs:
            kwargs["fused"] = local_kwargs.pop("fused")
        if local_kwargs:
            raise ValueError(
                f"Unsupported optimizer_kwargs for SGD: {sorted(local_kwargs.keys())}"
            )
        return torch.optim.SGD(
            _weight_decay_groups(
                parameter_list,
                kwargs["weight_decay"],
                decay_vector_parameters=decay_vector_parameters,
            ),
            **kwargs,
        )

    return _build_muon_optimizer(
        named_params=named_parameter_list,
        learning_rate=learning_rate,
        optimizer_kwargs=local_kwargs,
        capturable=capturable,
        device=device,
        decay_vector_parameters=decay_vector_parameters,
    )


def _build_muon_optimizer(
    *,
    named_params: list[tuple[str | None, torch.nn.Parameter]],
    learning_rate: float,
    optimizer_kwargs: dict[str, Any],
    capturable: bool,
    device: torch.device | str,
    decay_vector_parameters: bool = True,
) -> HybridMuonAdamW:
    muon_params = [
        param
        for name, param in named_params
        if _is_muon_hidden_layer_param(name, param)
    ]
    adamw_params = [
        param
        for name, param in named_params
        if param.requires_grad and not _is_muon_hidden_layer_param(name, param)
    ]
    if not muon_params and any(
        _is_muon_matrix_param(param) for _, param in named_params
    ):
        _LOGGER.warning(
            "Muon found no hidden matrix parameters; every parameter uses the "
            "AdamW companion."
        )

    optimizer_lr: float | torch.Tensor = learning_rate
    if capturable:
        optimizer_lr = torch.tensor(learning_rate, device=device, dtype=torch.float32)

    muon_kwargs: dict[str, Any] = {
        "lr": optimizer_lr,
        "weight_decay": float(optimizer_kwargs.pop("weight_decay", 0.1)),
        "momentum": float(optimizer_kwargs.pop("momentum", 0.95)),
        "nesterov": bool(optimizer_kwargs.pop("nesterov", True)),
        "ns_coefficients": tuple(
            optimizer_kwargs.pop("ns_coefficients", _MUON_DEFAULT_NS_COEFFICIENTS)
        ),
        "eps": float(optimizer_kwargs.pop("eps", _MUON_DEFAULT_EPS)),
        "ns_steps": int(optimizer_kwargs.pop("ns_steps", _MUON_DEFAULT_NS_STEPS)),
        "adjust_lr_fn": optimizer_kwargs.pop("adjust_lr_fn", None),
        "compile_ns": bool(optimizer_kwargs.pop("compile_ns", True)),
        "batched_updates": bool(optimizer_kwargs.pop("batched_updates", True)),
    }
    adamw_kwargs: dict[str, Any] = {
        "lr": optimizer_lr.clone() if torch.is_tensor(optimizer_lr) else optimizer_lr,
        "betas": tuple(optimizer_kwargs.pop("adamw_betas", (0.9, 0.999))),
        "eps": float(optimizer_kwargs.pop("adamw_eps", 1.0e-8)),
        "weight_decay": float(
            optimizer_kwargs.pop("adamw_weight_decay", muon_kwargs["weight_decay"])
        ),
        "amsgrad": bool(optimizer_kwargs.pop("adamw_amsgrad", False)),
        "capturable": capturable,
    }
    if "adamw_maximize" in optimizer_kwargs:
        adamw_kwargs["maximize"] = bool(optimizer_kwargs.pop("adamw_maximize"))
    if "adamw_differentiable" in optimizer_kwargs:
        adamw_kwargs["differentiable"] = bool(
            optimizer_kwargs.pop("adamw_differentiable")
        )
    if "adamw_foreach" in optimizer_kwargs:
        adamw_kwargs["foreach"] = optimizer_kwargs.pop("adamw_foreach")
    if "adamw_fused" in optimizer_kwargs:
        adamw_kwargs["fused"] = optimizer_kwargs.pop("adamw_fused")
    if optimizer_kwargs:
        raise ValueError(
            f"Unsupported optimizer_kwargs for MUON: {sorted(optimizer_kwargs.keys())}"
        )

    return HybridMuonAdamW(
        muon_params=muon_params,
        adamw_params=_weight_decay_groups(
            adamw_params,
            adamw_kwargs["weight_decay"],
            decay_vector_parameters=decay_vector_parameters,
        ),
        muon_kwargs=muon_kwargs,
        adamw_kwargs=adamw_kwargs,
    )


def _scalar_learning_rate(value: float | torch.Tensor) -> float | torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        return value.squeeze()
    return value


def _adjust_muon_lr(
    lr: float | torch.Tensor,
    adjust_lr_fn: str | None,
    param_shape: torch.Size,
) -> float | torch.Tensor:
    rows, cols = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        scale = math.sqrt(max(1.0, rows / cols))
    elif adjust_lr_fn == "match_rms_adamw":
        scale = 0.2 * math.sqrt(max(rows, cols))
    else:
        raise ValueError(
            "adjust_lr_fn must be one of: None, 'original', 'match_rms_adamw'."
        )
    return lr * scale


def _muon_zero_power_via_newton_schulz(
    grad: torch.Tensor,
    *,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    if grad.ndim != 2:
        raise ValueError("Muon only supports 2D parameter gradients.")
    if ns_steps >= 100:
        raise ValueError("ns_steps must be less than 100 for Muon.")
    if len(ns_coefficients) != 3:
        raise ValueError("ns_coefficients must contain exactly 3 values.")

    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.shape[0] > grad.shape[1]:
        ortho_grad = ortho_grad.T
    ortho_grad = ortho_grad / ortho_grad.norm().clamp(min=eps)
    for _ in range(ns_steps):
        gram = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.shape[0] > grad.shape[1]:
        ortho_grad = ortho_grad.T
    return ortho_grad


def _muon_zero_power_via_newton_schulz_batched(
    grads: torch.Tensor,
    *,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    if grads.ndim != 3:
        raise ValueError("Batched Muon only supports 3D gradient tensors.")
    if ns_steps >= 100:
        raise ValueError("ns_steps must be less than 100 for Muon.")
    if len(ns_coefficients) != 3:
        raise ValueError("ns_coefficients must contain exactly 3 values.")

    a, b, c = ns_coefficients
    ortho_grads = grads.bfloat16()
    transposed = False
    if grads.shape[1] > grads.shape[2]:
        ortho_grads = ortho_grads.transpose(1, 2)
        transposed = True
    norms = torch.linalg.vector_norm(ortho_grads.float(), dim=(1, 2), keepdim=True).to(
        dtype=ortho_grads.dtype
    )
    ortho_grads = ortho_grads / norms.clamp(min=eps)
    for _ in range(ns_steps):
        gram = torch.bmm(ortho_grads, ortho_grads.transpose(1, 2))
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        ortho_grads = torch.baddbmm(ortho_grads, gram_update, ortho_grads, beta=a)
    if transposed:
        ortho_grads = ortho_grads.transpose(1, 2)
    return ortho_grads


def _resolve_muon_ns_kernel(
    *,
    batched: bool,
    compile_ns: bool,
    device_type: str,
):
    if not compile_ns or device_type != "cuda" or not hasattr(torch, "compile"):
        return (
            _muon_zero_power_via_newton_schulz_batched
            if batched
            else _muon_zero_power_via_newton_schulz
        )

    global _COMPILED_MUON_NS_SINGLE, _COMPILED_MUON_NS_BATCHED
    if batched:
        if _COMPILED_MUON_NS_BATCHED is None:
            _COMPILED_MUON_NS_BATCHED = torch.compile(
                _muon_zero_power_via_newton_schulz_batched,
                mode=_MUON_COMPILE_MODE,
                dynamic=False,
            )
        return _COMPILED_MUON_NS_BATCHED

    if _COMPILED_MUON_NS_SINGLE is None:
        _COMPILED_MUON_NS_SINGLE = torch.compile(
            _muon_zero_power_via_newton_schulz,
            mode=_MUON_COMPILE_MODE,
            dynamic=False,
        )
    return _COMPILED_MUON_NS_SINGLE


class CapturableMuon(torch.optim.Optimizer):
    """Graph-safe Muon implementation for 2D parameters."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float | torch.Tensor = 1.0e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = _MUON_DEFAULT_NS_COEFFICIENTS,
        eps: float = _MUON_DEFAULT_EPS,
        ns_steps: int = _MUON_DEFAULT_NS_STEPS,
        adjust_lr_fn: str | None = None,
        compile_ns: bool = True,
        batched_updates: bool = True,
    ) -> None:
        lr = _scalar_learning_rate(lr)
        if isinstance(lr, torch.Tensor) and lr.dim() != 0:
            raise ValueError("Tensor lr must be 0-dimensional after squeezing")
        if isinstance(lr, torch.Tensor):
            if bool((lr < 0).item()):
                raise ValueError("Learning rate should be >= 0")
        elif lr < 0.0:
            raise ValueError("Learning rate should be >= 0")
        if momentum < 0.0:
            raise ValueError("momentum should be >= 0")
        if weight_decay < 0.0:
            raise ValueError("weight_decay should be >= 0")
        if adjust_lr_fn not in {None, "original", "match_rms_adamw"}:
            raise ValueError(
                "adjust_lr_fn must be one of: None, 'original', 'match_rms_adamw'."
            )

        defaults = {
            "lr": lr,
            "weight_decay": float(weight_decay),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "ns_coefficients": tuple(ns_coefficients),
            "eps": float(eps),
            "ns_steps": int(ns_steps),
            "adjust_lr_fn": adjust_lr_fn,
            "compile_ns": bool(compile_ns),
            "batched_updates": bool(batched_updates),
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters, but received "
                        f"shape {tuple(param.shape)}."
                    )

    @torch.no_grad()
    def step(self, closure=None, *, found_inf: torch.Tensor | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = _scalar_learning_rate(group["lr"])
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]
            compile_ns = bool(group["compile_ns"])
            batched_updates = bool(group["batched_updates"])
            decay = 1.0 - lr * weight_decay
            entries_by_shape: dict[
                torch.Size,
                list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]],
            ] = {}
            device_type = ""

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")
                if torch.is_complex(param):
                    raise RuntimeError("Muon does not support complex parameters.")

                state = self.state[param]
                momentum_buffer = state.get("momentum_buffer")
                if momentum_buffer is None:
                    momentum_buffer = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )
                    state["momentum_buffer"] = momentum_buffer
                if not device_type:
                    device_type = param.device.type
                entries_by_shape.setdefault(param.shape, []).append(
                    (param, grad, momentum_buffer)
                )

            if not entries_by_shape:
                continue

            single_ns_kernel = _resolve_muon_ns_kernel(
                batched=False, compile_ns=compile_ns, device_type=device_type
            )
            batched_ns_kernel = _resolve_muon_ns_kernel(
                batched=True, compile_ns=compile_ns, device_type=device_type
            )

            for shape, entries in entries_by_shape.items():
                adjusted_lr = _adjust_muon_lr(lr, adjust_lr_fn, shape)

                # Batch repeated MLP weight shapes to reduce launch overhead.
                if batched_updates and len(entries) > 1:
                    grads = torch.stack([grad for _, grad, _ in entries], dim=0)
                    momentum_buffers = torch.stack(
                        [buffer for _, _, buffer in entries], dim=0
                    )
                    momentum_buffers.lerp_(grads, 1.0 - momentum)
                    updates = (
                        grads.lerp(momentum_buffers, momentum)
                        if nesterov
                        else momentum_buffers
                    )
                    updates = batched_ns_kernel(
                        updates,
                        ns_coefficients=ns_coefficients,
                        ns_steps=ns_steps,
                        eps=eps,
                    )
                    for index, (param, _, momentum_buffer) in enumerate(entries):
                        if found_inf is None:
                            momentum_buffer.copy_(momentum_buffers[index])
                            param.mul_(decay)
                            param.add_(updates[index] * (-adjusted_lr))
                        else:
                            skip = found_inf.bool()
                            momentum_buffer.copy_(
                                torch.where(
                                    skip, momentum_buffer, momentum_buffers[index]
                                )
                            )
                            candidate = param * decay
                            candidate.add_(updates[index] * (-adjusted_lr))
                            param.copy_(torch.where(skip, param, candidate))
                    continue

                for param, grad, momentum_buffer in entries:
                    if found_inf is None:
                        momentum_buffer.lerp_(grad, 1.0 - momentum)
                        next_momentum = momentum_buffer
                    else:
                        next_momentum = momentum_buffer.lerp(grad, 1.0 - momentum)
                    update = (
                        grad.lerp(next_momentum, momentum)
                        if nesterov
                        else next_momentum
                    )
                    update = single_ns_kernel(
                        update,
                        ns_coefficients=ns_coefficients,
                        ns_steps=ns_steps,
                        eps=eps,
                    )
                    if found_inf is None:
                        param.mul_(decay)
                        param.add_(update * (-adjusted_lr))
                    else:
                        skip = found_inf.bool()
                        momentum_buffer.copy_(
                            torch.where(skip, momentum_buffer, next_momentum)
                        )
                        candidate = param * decay
                        candidate.add_(update * (-adjusted_lr))
                        param.copy_(torch.where(skip, param, candidate))

        return loss

    @torch.no_grad()
    def warmup_ns_kernels(self) -> None:
        """Compile Muon Newton-Schulz helpers without mutating parameters."""

        for group in self.param_groups:
            compile_ns = bool(group["compile_ns"])
            if not compile_ns:
                continue
            params = [
                param
                for param in group["params"]
                if param.requires_grad and param.device.type == "cuda"
            ]
            if not params:
                continue

            ns_coefficients = group["ns_coefficients"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            batched_updates = bool(group["batched_updates"])
            single_ns_kernel = _resolve_muon_ns_kernel(
                batched=False,
                compile_ns=True,
                device_type="cuda",
            )
            batched_ns_kernel = _resolve_muon_ns_kernel(
                batched=True,
                compile_ns=True,
                device_type="cuda",
            )
            params_by_shape: dict[
                torch.Size,
                list[torch.nn.Parameter],
            ] = {}
            for param in params:
                params_by_shape.setdefault(param.shape, []).append(param)

            for shape, shape_params in params_by_shape.items():
                sample = shape_params[0]
                if batched_updates and len(shape_params) > 1:
                    dummy = torch.zeros(
                        (len(shape_params), *shape),
                        device=sample.device,
                        dtype=sample.dtype,
                    )
                    batched_ns_kernel(
                        dummy,
                        ns_coefficients=ns_coefficients,
                        ns_steps=ns_steps,
                        eps=eps,
                    )
                    continue

                dummy = torch.zeros(
                    shape,
                    device=sample.device,
                    dtype=sample.dtype,
                )
                single_ns_kernel(
                    dummy,
                    ns_coefficients=ns_coefficients,
                    ns_steps=ns_steps,
                    eps=eps,
                )


class HybridMuonAdamW:
    """Use Muon on hidden 2D parameters and AdamW on everything else."""

    def __init__(
        self,
        *,
        muon_params: list[torch.nn.Parameter],
        adamw_params: list[torch.nn.Parameter] | list[dict[str, Any]],
        muon_kwargs: Mapping[str, Any],
        adamw_kwargs: Mapping[str, Any],
    ) -> None:
        self._muon_optimizer = (
            CapturableMuon(muon_params, **dict(muon_kwargs)) if muon_params else None
        )
        self._adamw_optimizer = (
            torch.optim.AdamW(adamw_params, **dict(adamw_kwargs))
            if adamw_params
            else None
        )
        if self._muon_optimizer is None and self._adamw_optimizer is None:
            raise ValueError("HybridMuonAdamW received no trainable parameters.")

        self.defaults: dict[str, Any] = {}
        self._refresh_param_groups()

    @property
    def muon_optimizer(self) -> CapturableMuon | None:
        return self._muon_optimizer

    @property
    def adamw_optimizer(self) -> torch.optim.AdamW | None:
        return self._adamw_optimizer

    @property
    def state(self):
        merged: dict[torch.nn.Parameter, dict[str, Any]] = {}
        if self._muon_optimizer is not None:
            merged.update(self._muon_optimizer.state)
        if self._adamw_optimizer is not None:
            merged.update(self._adamw_optimizer.state)
        return merged

    def _optimizers(self) -> list[torch.optim.Optimizer]:
        return [
            optimizer
            for optimizer in (self._muon_optimizer, self._adamw_optimizer)
            if optimizer is not None
        ]

    def _refresh_param_groups(self) -> None:
        self.param_groups = []
        for optimizer in self._optimizers():
            self.param_groups.extend(optimizer.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self._optimizers():
            optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None, *, found_inf: torch.Tensor | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._muon_optimizer is not None:
            self._muon_optimizer.step(found_inf=found_inf)
        if self._adamw_optimizer is not None:
            if found_inf is None:
                self._adamw_optimizer.step()
            else:
                optimizer_step_if_finite(self._adamw_optimizer, found_inf)
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": "hybrid_muon_adamw",
            "muon": (
                self._muon_optimizer.state_dict()
                if self._muon_optimizer is not None
                else None
            ),
            "adamw": (
                self._adamw_optimizer.state_dict()
                if self._adamw_optimizer is not None
                else None
            ),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        payload_type = state_dict.get("type")
        if payload_type not in {None, "hybrid_muon_adamw"}:
            raise ValueError(f"Unsupported optimizer state payload: {payload_type}")

        muon_state = state_dict.get("muon")
        adamw_state = state_dict.get("adamw")

        if self._muon_optimizer is not None and muon_state is None:
            raise ValueError("Missing Muon state for HybridMuonAdamW.")
        if self._muon_optimizer is None and muon_state is not None:
            raise ValueError("Received unexpected Muon state for HybridMuonAdamW.")
        if self._adamw_optimizer is not None and adamw_state is None:
            raise ValueError("Missing AdamW state for HybridMuonAdamW.")
        if self._adamw_optimizer is None and adamw_state is not None:
            raise ValueError("Received unexpected AdamW state for HybridMuonAdamW.")

        previous_state = copy.deepcopy(self.state_dict())
        try:
            self._load_optimizer_states(
                muon_state=muon_state,
                adamw_state=adamw_state,
            )
        except Exception:
            try:
                self._load_optimizer_states(
                    muon_state=previous_state["muon"],
                    adamw_state=previous_state["adamw"],
                )
            except Exception as rollback_exc:
                raise RuntimeError(
                    "Hybrid Muon/AdamW optimizer load failed and rollback also failed."
                ) from rollback_exc
            self._refresh_param_groups()
            raise

        self._refresh_param_groups()

    def _load_optimizer_states(
        self,
        *,
        muon_state: Mapping[str, Any] | None,
        adamw_state: Mapping[str, Any] | None,
    ) -> None:
        if self._muon_optimizer is not None:
            if muon_state is None:
                raise RuntimeError("Muon optimizer state is unavailable while loading.")
            self._muon_optimizer.load_state_dict(muon_state)
        if self._adamw_optimizer is not None:
            if adamw_state is None:
                raise RuntimeError(
                    "AdamW optimizer state is unavailable while loading."
                )
            self._adamw_optimizer.load_state_dict(adamw_state)


@torch.no_grad()
def prepare_optimizer_state(
    optimizer: torch.optim.Optimizer | HybridMuonAdamW,
    *,
    capturable: bool,
) -> None:
    """Materialize missing state and restore capture metadata without resetting it."""
    if isinstance(optimizer, HybridMuonAdamW):
        for child in (optimizer.muon_optimizer, optimizer.adamw_optimizer):
            if child is not None:
                prepare_optimizer_state(child, capturable=capturable)
        return
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW, CapturableMuon)):
        raise ValueError("Optimizer state preparation requires Adam, AdamW, or Muon")
    for group in optimizer.param_groups:
        if group.get("differentiable", False):
            raise ValueError(
                "Optimizer state preparation does not support differentiable optimizers"
            )
        for param in group["params"]:
            state = optimizer.state[param]
            if capturable:
                # Checkpoints carry their source runtime's group flags, LR type
                # and step storage. The active graph requires device tensors.
                group["lr"] = torch.as_tensor(
                    group["lr"], dtype=torch.float32, device=param.device
                )
                if not isinstance(optimizer, CapturableMuon):
                    group["capturable"] = True
                    # Graph optimizer validation excludes foreach; the source
                    # checkpoint may have selected it for eager execution.
                    group["foreach"] = False
                    if "step" in state:
                        state["step"] = state["step"].to(param.device)
            if isinstance(optimizer, CapturableMuon):
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                continue
            if "step" not in state:
                device = (
                    param.device
                    if group.get("capturable") or group.get("fused")
                    else "cpu"
                )
                state["step"] = torch.zeros((), dtype=torch.float32, device=device)
            for key in ("exp_avg", "exp_avg_sq"):
                if key not in state:
                    state[key] = torch.zeros_like(param)
            if group["amsgrad"] and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = torch.zeros_like(param)


@torch.no_grad()
def optimizer_step_if_finite(
    optimizer: torch.optim.Optimizer | HybridMuonAdamW,
    found_inf: torch.Tensor,
) -> None:
    """Skip all optimizer writes using a device flag, without rollback copies.

    Adam/AdamW use PyTorch's fused AMP-aware kernel (including its step-counter
    correction); Muon masks momentum and parameter writes. This execution path
    deliberately selects the fused Adam backend, independently of the backend
    used by an ordinary optimizer.step(). Optimizer state is prepared before
    compilation/capture by the loss scaler. No device scalar is read by Python.
    """
    if isinstance(optimizer, (HybridMuonAdamW, CapturableMuon)):
        optimizer.step(found_inf=found_inf)
        return
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise ValueError("Overflow-aware stepping requires Adam, AdamW, or Muon")
    # The PyTorch functional implementation forwards found_inf to the native
    # fused kernel and cancels counter increments on an overflow. Calling it
    # directly avoids retaining temporary AMP attributes on a live optimizer.
    from torch.optim import _functional

    for group in optimizer.param_groups:
        parameters = [p for p in group["params"] if p.grad is not None]
        if not parameters:
            continue
        if any(p.grad.is_sparse or p.is_complex() for p in parameters):
            raise ValueError("FP16 overflow-aware Adam requires dense real parameters")
        state = optimizer.state
        _functional.adam(
            parameters,
            [p.grad for p in parameters],
            [state[p]["exp_avg"] for p in parameters],
            [state[p]["exp_avg_sq"] for p in parameters],
            [state[p]["max_exp_avg_sq"] for p in parameters]
            if group["amsgrad"]
            else [],
            [state[p]["step"] for p in parameters],
            amsgrad=group["amsgrad"],
            has_complex=False,
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=group.get("maximize", False),
            capturable=group.get("capturable", False),
            differentiable=False,
            foreach=False,
            fused=True,
            grad_scale=None,
            found_inf=found_inf,
            decoupled_weight_decay=group.get(
                "decoupled_weight_decay", isinstance(optimizer, torch.optim.AdamW)
            ),
        )
