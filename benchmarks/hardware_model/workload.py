"""Describe the staged integrator workload before native compilation.

``describe_workload(solver)`` wires the cached device-function properties,
then reads their Python closures and source. It never requests the batch
kernel or executes a device function. Source locations join directly to
``static_descriptors.describe_source`` by source_path and definition line.

Loop copies describe requested source expansion, not predicted SASS size.
Newton and Krylov work stays symbolic and refers to warp-executed bodies;
per-lane active iteration counters are a different quantity.
"""

import ast
import hashlib
import inspect
import math
import numbers
import operator
from pathlib import Path
import textwrap

from cubie.CUDAFactory import CUDAFactory
from cubie.buffer_registry import buffer_registry
from cubie.cuda_simsafe import ALL_UNROLL_PARAMETERS


SCHEMA_VERSION = 1
UNKNOWN = object()
BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}
COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
CALL_ROLES = {
    "evaluate_f": "rhs",
    "evaluate_inv_mass_f": "inverse_mass_rhs",
    "evaluate_observables": "observables",
    "nonlinear_solver": "newton_solve",
    "linear_solver": "linear_solve",
    "linear_solver_fn": "linear_solve",
    "error_solver": "smoothed_error_linear_solve",
    "residual_function": "nonlinear_residual",
    "operator_apply": "linear_operator",
    "preconditioner": "preconditioner",
    "lu_solve": "generated_lu",
    "prepare_jacobian": "prepare_jacobian",
    "correction_norm_fn": "newton_correction_norm",
    "weighted_norm": "linear_residual_norm",
    "time_derivative_rhs": "time_derivative_rhs",
    "apply_mass": "mass_apply",
    "predict_stages": "stage_predictor",
}


def scalar(value):
    """Convert a genuine host scalar, without coercing runtime objects."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        return result if math.isfinite(result) else str(result)
    if type(value).__name__ == "bool_":
        return bool(value)
    return UNKNOWN


def constant_description(value):
    result = scalar(value)
    if result is not UNKNOWN:
        return result
    if isinstance(value, tuple):
        items = [constant_description(item) for item in value]
        if all(item is not UNKNOWN for item in items):
            return items
    if isinstance(getattr(value, "shape", None), tuple) and hasattr(
        value, "dtype"
    ):
        return dict(
            kind="closure_array",
            shape=list(value.shape),
            dtype=str(value.dtype),
        )
    return UNKNOWN


def evaluate(node, environment):
    """Resolve a restricted host-constant expression; never eval source."""
    if node is None:
        return UNKNOWN
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return environment.get(node.id, UNKNOWN)
    if isinstance(node, ast.UnaryOp):
        value = evaluate(node.operand, environment)
        if value is UNKNOWN:
            return UNKNOWN
        operations = {
            ast.Not: operator.not_,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Invert: operator.invert,
        }
        function = operations.get(type(node.op))
        return function(value) if function else UNKNOWN
    if isinstance(node, ast.BinOp):
        left = evaluate(node.left, environment)
        right = evaluate(node.right, environment)
        function = BINARY.get(type(node.op))
        if left is UNKNOWN or right is UNKNOWN or function is None:
            return UNKNOWN
        try:
            return function(left, right)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return UNKNOWN
    if isinstance(node, ast.BoolOp):
        values = [evaluate(value, environment) for value in node.values]
        if any(value is UNKNOWN for value in values):
            return UNKNOWN
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        values = [evaluate(node.left, environment)] + [
            evaluate(value, environment) for value in node.comparators
        ]
        if any(value is UNKNOWN for value in values):
            return UNKNOWN
        for left, operation, right in zip(values, node.ops, values[1:]):
            function = COMPARE.get(type(operation))
            if function is None or not function(left, right):
                return UNKNOWN if function is None else False
        return True
    if isinstance(node, ast.Subscript):
        value = evaluate(node.value, environment)
        index = evaluate(node.slice, environment)
        if value is UNKNOWN or index is UNKNOWN:
            return UNKNOWN
        if not isinstance(value, (tuple, list)) and not hasattr(
            value, "shape"
        ):
            return UNKNOWN
        try:
            return value[index]
        except (IndexError, TypeError):
            return UNKNOWN
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [evaluate(value, environment) for value in node.elts]
        return (
            tuple(values) if all(v is not UNKNOWN for v in values) else UNKNOWN
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        values = [evaluate(value, environment) for value in node.args]
        if node.keywords or any(value is UNKNOWN for value in values):
            return UNKNOWN
        if name in {"int32", "int64", "int"} and len(values) == 1:
            return int(values[0])
        if name in {"float32", "float64", "float"} and len(values) == 1:
            return float(values[0])
        if name == "len" and len(values) == 1:
            return len(values[0])
    return UNKNOWN


def python_function(function):
    candidate = getattr(function, "py_func", function)
    return candidate if inspect.isfunction(candidate) else None


def source_function(function):
    """Recover the exact function body and absolute source locations."""
    source_lines, first_line = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent("".join(source_lines)))
    ast.increment_lineno(tree, first_line - 1)
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
    )


def request_replication(flag, trip_count):
    """Describe the directive without predicting backend transformations."""
    enabled, count = flag if isinstance(flag, (tuple, list)) else (flag, None)
    if type(enabled) is not bool or (
        count is not None and (type(count) is not int or count < 1)
    ):
        raise ValueError(f"Invalid unroll directive: {flag!r}")
    if not enabled:
        return dict(
            mode="backend_choice",
            requested_body_copies=None,
            native_size_known=False,
        )
    copies = (
        trip_count
        if count is None
        else (
            min(int(count), trip_count)
            if trip_count is not None
            else int(count)
        )
    )
    return dict(
        mode="full" if count is None else "counted",
        requested_count=count,
        requested_body_copies=copies,
        native_size_known=False,
        remainder_iterations=(
            trip_count % count if trip_count is not None and count else None
        ),
    )


def loop_dependencies(node):
    """Report syntactic recurrence candidates, without claiming alias proof."""
    reads, writes, recurrence = set(), set(), set()
    array_reads, array_writes = set(), set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            (writes if isinstance(child.ctx, ast.Store) else reads).add(
                child.id
            )
        if isinstance(child, ast.Subscript) and isinstance(
            child.value, ast.Name
        ):
            target = (
                array_writes
                if isinstance(child.ctx, ast.Store)
                else array_reads
            )
            target.add(child.value.id)
        if isinstance(child, (ast.Assign, ast.AugAssign)):
            targets = (
                child.targets
                if isinstance(child, ast.Assign)
                else [child.target]
            )
            used = {
                item.id
                for item in ast.walk(child.value)
                if isinstance(item, ast.Name)
            }
            for target in targets:
                if isinstance(target, ast.Name) and (
                    target.id in used or isinstance(child, ast.AugAssign)
                ):
                    recurrence.add(target.id)
    return dict(
        scalar_read_write_candidates=sorted(reads & writes),
        scalar_self_recurrences=sorted(recurrence),
        buffer_read_write_candidates=sorted(array_reads & array_writes),
        alias_and_cross_iteration_dependence_proven=False,
    )


class WorkloadGraph:
    """Lexical call graph with bound closure constants and guarded loops."""

    def __init__(self, flags):
        self.flags = flags or {}
        self.functions = []
        self.identities = {}
        self.dispatchers = []
        self.limitations = []

    def add_function(self, dispatcher, role):
        function = python_function(dispatcher)
        if function is None:
            self.limitations.append(
                dict(role=role, reason="No Python source callable")
            )
            return None
        if id(function) in self.identities:
            identifier = self.identities[id(function)]
            record = self.functions[int(identifier[1:])]
            if role not in record["roles"]:
                record["roles"].append(role)
            return identifier
        if hasattr(dispatcher, "overloads"):
            if dispatcher.overloads:
                raise ValueError(
                    "Workload extraction requires uncompiled dispatchers"
                )
            self.dispatchers.append(dispatcher)
        identifier = f"f{len(self.functions)}"
        self.identities[id(function)] = identifier
        path = Path(inspect.getsourcefile(function)).resolve()
        try:
            node = source_function(function)
        except (OSError, TypeError, SyntaxError) as error:
            raise ValueError(
                f"Cannot inspect {function.__qualname__}: {error}"
            ) from error
        closure = inspect.getclosurevars(function)
        environment = dict(closure.globals, **closure.nonlocals)
        constants = {
            name: constant_description(value)
            for name, value in closure.nonlocals.items()
        }
        constants = {
            name: value
            for name, value in constants.items()
            if value is not UNKNOWN
        }
        record = dict(
            id=identifier,
            roles=[role],
            name=function.__name__,
            qualified_name=function.__qualname__,
            source=dict(
                source_path=str(path),
                line=node.lineno,
                end_line=node.end_lineno,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
            static_descriptor_join=dict(
                source_path=str(path), line=node.lineno
            ),
            arguments=[argument.arg for argument in node.args.args],
            closure_constants=constants,
            loops=[],
            calls=[],
            boundaries=[],
            buffer_access_spans={},
        )
        self.functions.append(record)
        self.visit(node.body, environment, record, [], [], True)
        return identifier

    def context(self, statement, record, loops, guards, active):
        return dict(
            source=dict(
                source_path=record["source"]["source_path"],
                line=statement.lineno,
                end_line=statement.end_lineno,
            ),
            enclosing_loops=list(loops),
            guards=list(guards),
            statically_reachable=active,
        )

    def visit(self, statements, environment, record, loops, guards, active):
        environment = dict(environment)
        for statement in statements:
            context = self.context(statement, record, loops, guards, active)
            if isinstance(statement, ast.If):
                condition = evaluate(statement.test, environment)
                truth = None if condition is UNKNOWN else bool(condition)
                ends = []
                for branch, body in (
                    (True, statement.body),
                    (False, statement.orelse),
                ):
                    guard = dict(
                        expression=ast.unparse(statement.test),
                        line=statement.lineno,
                        branch=branch,
                        constant_result=truth,
                    )
                    ends.append(
                        self.visit(
                            body,
                            environment,
                            record,
                            loops,
                            guards + [guard],
                            active and (truth is None or truth == branch),
                        )
                    )
                if truth is not None:
                    environment = ends[0 if truth else 1]
                else:
                    environment = {
                        key: value
                        for key, value in ends[0].items()
                        if key in ends[1] and value is ends[1][key]
                    }
                continue
            if isinstance(statement, (ast.For, ast.While)):
                iterator = (
                    statement.iter if isinstance(statement, ast.For) else None
                )
                flag_name = None
                flag_binding = None
                flag = None
                range_node = iterator
                if isinstance(iterator, ast.Call) and isinstance(
                    iterator.func, ast.Name
                ):
                    if iterator.func.id == "unroll_if":
                        range_node = iterator.args[0]
                        flag_binding = ast.unparse(iterator.args[1])
                        flag_name = flag_binding
                        if (
                            flag_binding == "_unroll"
                            and Path(record["source"]["source_path"]).name
                            == "buffer_registry.py"
                        ):
                            flag_name = "unroll_other_small"
                        flag = self.flags.get(
                            flag_name, evaluate(iterator.args[1], environment)
                        )
                count = None
                if isinstance(range_node, ast.Call) and isinstance(
                    range_node.func, ast.Name
                ):
                    if range_node.func.id == "range":
                        values = [
                            evaluate(arg, environment)
                            for arg in range_node.args
                        ]
                        if all(
                            isinstance(value, numbers.Integral)
                            for value in values
                        ):
                            count = len(
                                range(*(int(value) for value in values))
                            )
                iteration_kind = (
                    "newton"
                    if flag_name == "unroll_newton_exits"
                    else "krylov"
                    if flag_name == "unroll_krylov_exits"
                    else None
                )
                loop_id = f"{record['id']}:L{statement.lineno}"
                dynamic = (
                    dict(
                        symbol=("N" if iteration_kind == "newton" else "K")
                        + "[call,warp]",
                        lower_bound=0,
                        upper_bound=count,
                    )
                    if iteration_kind
                    else dict(fixed_trip_count=count)
                )
                entry = dict(
                    id=loop_id,
                    target=ast.unparse(statement.target)
                    if isinstance(statement, ast.For)
                    else None,
                    iterator=ast.unparse(iterator) if iterator else None,
                    condition=ast.unparse(statement.test)
                    if isinstance(statement, ast.While)
                    else None,
                    trip_count=count,
                    unroll_group=flag_name,
                    flag_binding=flag_binding,
                    flag=constant_description(flag)
                    if flag is not UNKNOWN
                    else None,
                    dynamic_iterations=dynamic,
                    requested_replication=request_replication(flag, count)
                    if flag is not None and flag is not UNKNOWN
                    else dict(
                        mode="ordinary_loop",
                        requested_body_copies=1,
                        native_size_known=False,
                    ),
                    dependency=loop_dependencies(statement),
                    **context,
                )
                record["loops"].append(entry)
                inner = dict(environment)
                if isinstance(statement, ast.For):
                    for target in ast.walk(statement.target):
                        if isinstance(target, ast.Name):
                            inner.pop(target.id, None)
                self.visit(
                    statement.body,
                    inner,
                    record,
                    loops + [loop_id],
                    guards,
                    active and count != 0,
                )
                for child in ast.walk(statement):
                    if isinstance(child, ast.Name) and isinstance(
                        child.ctx, ast.Store
                    ):
                        environment.pop(child.id, None)
                self.visit(
                    statement.orelse,
                    environment,
                    record,
                    loops,
                    guards,
                    active,
                )
                continue
            if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, (ast.Return, ast.Break, ast.Continue)):
                record["boundaries"].append(
                    dict(kind=type(statement).__name__, **context)
                )
            for child in ast.walk(statement):
                if isinstance(child, ast.Subscript) and isinstance(
                    child.value, ast.Name
                ):
                    name = child.value.id
                    span = record["buffer_access_spans"].setdefault(
                        name,
                        dict(
                            first_line=child.lineno,
                            last_line=child.lineno,
                            read_lines=[],
                            write_lines=[],
                        ),
                    )
                    span["last_line"] = max(span["last_line"], child.lineno)
                    mode = (
                        "write_lines"
                        if isinstance(child.ctx, ast.Store)
                        else "read_lines"
                    )
                    if child.lineno not in span[mode]:
                        span[mode].append(child.lineno)
                if isinstance(child, ast.Call) and isinstance(
                    child.func, ast.Name
                ):
                    binding = child.func.id
                    callback = environment.get(binding)
                    if not hasattr(callback, "py_func"):
                        continue
                    function = python_function(callback)
                    if function is None:
                        continue
                    role = CALL_ROLES.get(
                        binding,
                        "allocator"
                        if binding.startswith("alloc")
                        else "device_callback",
                    )
                    callee = (
                        self.add_function(callback, role) if active else None
                    )
                    arguments = [ast.unparse(value) for value in child.args]
                    record["calls"].append(
                        dict(
                            binding=binding,
                            role=role,
                            callee=callee,
                            arguments=arguments,
                            parameter_bindings=dict(
                                zip(
                                    inspect.signature(function).parameters,
                                    arguments,
                                )
                            ),
                            **self.context(
                                child, record, loops, guards, active
                            ),
                        )
                    )
            if isinstance(statement, ast.Assign):
                value = evaluate(statement.value, environment)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        environment.pop(target.id, None)
                        if value is not UNKNOWN:
                            environment[target.id] = value
            elif isinstance(statement, ast.AugAssign) and isinstance(
                statement.target, ast.Name
            ):
                environment.pop(statement.target.id, None)
        return environment


def registered_buffers(step):
    """Read real registry entries and storage-duration boundaries."""
    result = []
    seen = set()

    def visit(factory, owner):
        if id(factory) in seen:
            return
        seen.add(id(factory))
        group = buffer_registry._groups.get(factory)
        if group is not None:
            scope = (
                "nonlinear_solve"
                if type(factory).__name__ == "NewtonKrylov"
                else "linear_solve"
                if "Solver" in type(factory).__name__
                else "step_attempt"
            )
            for name, entry in group.entries.items():
                result.append(
                    dict(
                        owner=owner,
                        owner_type=type(factory).__name__,
                        name=name,
                        elements=int(entry.size),
                        element_bytes=entry.itemsize,
                        declared_bytes=int(entry.size) * entry.itemsize,
                        location=entry.location,
                        persistent=entry.persistent,
                        alias_target=entry.aliases,
                        storage_duration="integration_run"
                        if entry.persistent
                        else scope,
                        protected_ranges=[
                            list(pair) for pair in entry.protected
                        ],
                        storage_contract_only=True,
                    )
                )
        for name, child in sorted(vars(factory).items()):
            if isinstance(child, CUDAFactory):
                visit(child, f"{owner}.{name}")

    visit(step, "step")
    return result


def family_workload(step, closure):
    """Describe family call multiplicities from actual tableau/config data."""
    families = {
        "ERKStep": "ERK",
        "DIRKStep": "DIRK",
        "FIRKStep": "FIRK",
        "GenericRosenbrockWStep": "ROS",
    }
    family = families.get(type(step).__name__)
    if family is None:
        raise ValueError(
            f"Unsupported algorithm family: {type(step).__name__}"
        )
    config = step.compile_settings
    stages = int(config.tableau.stage_count)
    n = int(config.n)
    info = dict(
        family=family,
        n_states=n,
        stage_count=stages,
        solve_width=int(getattr(config, "solver_width", n)),
        iteration_counts_are_warp_body_counts=True,
        smoothed_error=bool(closure.get("use_smoothed_error", False)),
    )
    if family == "ERK":
        info.update(
            inner_solver=None,
            nonlinear_solves=[],
            rhs_calls=dict(
                stage_zero="I[not use_cached_rhs]", remaining_stages=stages - 1
            ),
            stage_zero_cache_enabled=bool(
                closure["first_same_as_last"] and stages > 1
            ),
        )
        return info
    linear = step.linear_solver
    linear_config = linear.compile_settings
    inner = str(linear.linear_correction_type)
    info.update(
        inner_solver=inner,
        krylov_max_iters=None
        if inner == "lu"
        else int(linear_config.max_iters),
        zero_initial_guess=bool(linear_config.zero_initial_guess),
    )
    if family == "ROS":
        info.update(
            newton_max_iters=None,
            nonlinear_solves=[],
            stage_linear_calls=stages,
            rhs_calls_per_step=stages,
            prepare_jacobian_calls_per_step=1,
            time_derivative_calls_per_step=1 + int(stages > 1),
        )
    else:
        info["newton_max_iters"] = int(step.solver.compile_settings.max_iters)
        if family == "DIRK":
            diagonal = list(config.tableau.diagonal)
            info["nonlinear_solves"] = [
                dict(
                    stage=index,
                    width=n,
                    executes="I[not use_cached_rhs]" if index == 0 else "1",
                    iterations=f"N[{index},warp]",
                    linear_work=f"sum(j=1..N[{index},warp], K[{index},j,warp])"
                    if inner != "lu"
                    else f"N[{index},warp] direct calls",
                )
                for index, value in enumerate(diagonal)
                if value != 0
            ]
            info["explicit_rhs_stages"] = [
                index for index, value in enumerate(diagonal) if value == 0
            ]
        else:
            info["nonlinear_solves"] = [
                dict(
                    coupled_stages=list(range(stages)),
                    width=stages * n,
                    executes="1",
                    iterations="N[coupled,warp]",
                    linear_work="sum(j=1..N[coupled,warp], K[coupled,j,warp])"
                    if inner != "lu"
                    else "N[coupled,warp] direct calls",
                )
            ]
        info["prepare_jacobian_calls_per_step"] = int(
            closure["use_cached_solve"]
        )
    if info["smoothed_error"]:
        error_solver = (
            step.error_solver if step.error_solver is not None else linear
        )
        error_config = error_solver.compile_settings
        info["smoothed_error_solve"] = dict(
            calls_per_step=1,
            width=int(error_config.solver_width),
            inner_solver=str(error_solver.linear_correction_type),
            krylov_max_iters=None
            if inner == "lu"
            else int(error_config.max_iters),
            iterations="one direct call" if inner == "lu" else "K[error,warp]",
            additional_rhs_calls=int(family == "FIRK"),
        )
    return info


def describe_workload(solver, unroll=None):
    """Return JSON-serializable, unfitted static workload descriptors.

    Parameters
    ----------
    solver
        Constructed Solver. The batch kernel must not be compiled.
    unroll : dict, optional
        Candidate group directives for source-replication descriptions.
        Overrides do not mutate the Solver or regenerate its code.

    Returns
    -------
    dict
        Family multiplicities, bound function/call/loop graph, real buffer
        declarations, source provenance and explicit model limitations.
    """
    kernel = solver.kernel
    cached_kernel = getattr(kernel._cache, "solver_kernel", None)
    if getattr(cached_kernel, "overloads", None):
        raise ValueError("Pass a Solver before batch-kernel compilation")
    integrator = kernel.single_integrator
    # Cached construction wires helpers but does not specialize a kernel.
    loop_dispatcher = integrator.device_function
    if loop_dispatcher.overloads:
        raise ValueError("The integration loop already has native overloads")
    step = integrator._algo_step
    if unroll:
        unsupported = set(unroll) - ALL_UNROLL_PARAMETERS
        if unsupported:
            raise ValueError(f"Unknown unroll groups: {sorted(unsupported)}")
        candidate, _, _ = step.compile_settings.unroll.update(unroll)
        unroll = {name: getattr(candidate, name) for name in unroll}
    step_dispatcher = step.step_function
    closure = inspect.getclosurevars(step_dispatcher.py_func).nonlocals
    graph = WorkloadGraph(unroll)
    root = graph.add_function(step_dispatcher, "algorithm_step")
    family = family_workload(step, closure)
    family["source"] = graph.functions[0]["source"]
    family["call_source_references"] = [
        dict(role=call["role"], **call["source"])
        for function in graph.functions
        for call in function["calls"]
        if call["statically_reachable"] and call["role"] in CALL_ROLES.values()
    ]
    for dispatcher in graph.dispatchers:
        if dispatcher.overloads:
            raise RuntimeError(
                "Descriptor extraction compiled a device function"
            )
    return dict(
        schema_version=SCHEMA_VERSION,
        kind="static_workload_descriptors",
        root=root,
        workload=family,
        functions=graph.functions,
        buffers=registered_buffers(step),
        compilation_check=dict(
            batch_kernel_requested=False,
            inspected_dispatchers=len(graph.dispatchers),
            native_overloads=0,
        ),
        source_expansion_rules=dict(
            fixed_loop_dynamic_work="trip_count, subject to enclosing guards",
            iterative_dynamic_work="symbolic iterations bounded by config",
            nested_static_work="multiply copies along call/loop nesting",
            backend_choice="unknown for False directives",
        ),
        limitations=graph.limitations
        + [
            "No SASS, register, spill or timing prediction.",
            "Replication excludes backend folding, rescheduling and tails.",
            "No cross-function alias analysis or register allocation.",
            "Buffer spans are lexical accesses, not optimized lifetimes.",
            "Guards and early returns need a control-flow execution model.",
            "Coefficient zero folding inside replicated loops is unresolved.",
            "Warp body iterations differ from active-lane iteration counters.",
            "Scope: step attempt; excludes time loop, controller and outputs.",
        ],
    )
