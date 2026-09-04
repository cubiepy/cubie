"""Identify actual source workloads independently of candidate policy axes."""

from attrs import fields, has
from numpy import dtype as np_dtype

from cubie._serialize import canonical_digest


def semantic_config(value):
    """Retain typed semantic fields except unroll and placement decisions."""
    if has(type(value)):
        return (type(value).__module__, type(value).__qualname__, tuple(
            (field.name, semantic_config(getattr(value, field.name)))
            for field in fields(type(value))
            if field.eq and field.name != "unroll"
            and not field.name.endswith("_location")))
    if isinstance(value, (tuple, list)):
        return tuple(semantic_config(item) for item in value)
    return value


def workload_identity(system, solver):
    """Bind ODE, complete tableau and recursive solver settings to bytes."""
    root = solver.kernel.single_integrator._algo_step
    precision = np_dtype(root.compile_settings.precision).name
    if precision != "float32":
        raise ValueError("Hardware policy research requires actual FP32")
    factories = {}
    seen = set()

    def visit(factory, path):
        if id(factory) in seen:
            return
        seen.add(id(factory))
        factories[path] = canonical_digest(
            semantic_config(factory.compile_settings))
        for name, child in sorted(vars(factory).items()):
            if hasattr(child, "compile_settings"):
                visit(child, path + "." + name)

    visit(root, "step")
    return {
        "system_fn_hash": system.fn_hash,
        "system_semantic_config_sha256": canonical_digest(
            semantic_config(system.compile_settings)),
        "factory_semantic_config_sha256": factories,
        "tableau_sha256": canonical_digest(root.compile_settings.tableau),
        "precision": precision,
    }
