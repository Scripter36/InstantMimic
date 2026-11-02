"""Production muscle control components, loaded only when requested."""

from importlib import import_module

_EXPORTS = {
    "FastMuscleExecutionBindings": ".bindings",
    "FastMuscleJointTopologyBindings": ".bindings",
    "FastMuscleMotionReferenceBindings": ".bindings",
    "FastMuscleBootstrapSettings": ".control",
    "FastMuscleControl": ".control",
    "FastMuscleControlCfg": ".control",
    "FastMuscleController": ".controller",
    "FastMuscleMechanicsView": ".controller",
    "FastMuscleRenderGeometryView": ".controller",
    "FastMuscleControlSettings": ".settings",
    "activation_interval_bounds": ".settings",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
