"""Describe actual implicit solver roles and symbolic warp-body regimes.

Source construction requests cached Python dispatchers, never native
specializations. Runtime masks are explicit scenario inputs; observed
per-lane totals are never substituted for warp instruction work.
"""

import argparse
import ast
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from cubie import Solver
from cubie._utils import package_source_hash
from cubie.buffer_registry import buffer_registry
from cubie.cache_root import get_cache_root_override, set_cache_root
from cubie.cubie_cache import toolchain_fingerprint

from benchmarks import placement_landscape as placement
from benchmarks.hardware_model.buffer_descriptors import registry_layout
from benchmarks.hardware_model.expansion import (
    CapturedGraph,
    describe_expansion,
    snapshot,
    source_receipt,
)
from benchmarks.hardware_model.workload import (
    python_function,
    source_function,
)


SCRIPT = Path(__file__).resolve()
FAMILIES = {
    "DIRKStep": "DIRK",
    "FIRKStep": "FIRK",
    "GenericRosenbrockWStep": "ROS",
}
LINEAR_TYPES = {
    "lu": "LUSolver",
    "mr": "MRLinearSolver",
    "sd": "MRLinearSolver",
    "bicgstab": "BiCGSTABSolver",
}
PUBLIC_LINEAR_TYPES = {
    "lu": "lu",
    "mr": "minimal_residual",
    "sd": "steepest_descent",
    "bicgstab": "bicgstab",
}


def digest(path):
    """Return the source-file byte hash."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Create a finite JSON artifact without replacing prior evidence."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, allow_nan=False, separators=(",", ":"))
        handle.write("\n")


def exact_int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} requires an integer >= {minimum}")
    return value


def statements(node):
    """Return source statements with stable lines and complete expressions."""
    return [
        dict(
            line=item.lineno,
            kind=type(item).__name__,
            source=ast.unparse(item),
        )
        for item in node.body
    ]


def recurrent_control(function, kind):
    """Admit the actual supported stop/counter structure from source AST."""
    node = source_function(function)
    group = (
        "unroll_newton_exits" if kind == "newton" else "unroll_krylov_exits"
    )
    loops = [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.For)
        and isinstance(item.iter, ast.Call)
        and ast.unparse(item.iter.func) == "unroll_if"
        and len(item.iter.args) == 2
        and ast.unparse(item.iter.args[1]) == group
    ]
    if len(loops) != 1:
        raise ValueError("Need one actual recurrent loop for this role")
    loop = loops[0]
    first = loop.body[0]
    condition = {
        "newton": "all_sync(mask, converged | failed)",
        "bicgstab": "all_sync(mask, finished)",
        "mr": "all_sync(mask, converged)",
        "sd": "all_sync(mask, converged)",
    }[kind]
    if (
        not isinstance(first, ast.If)
        or ast.unparse(first.test) != condition
        or len(first.body) != 1
        or not isinstance(first.body[0], ast.Break)
    ):
        raise ValueError("Unsupported recurrent exit semantics")
    counter = "iters_count" if kind == "newton" else "iter_count"
    updates = []
    for item in ast.walk(loop):
        target = item.target if isinstance(item, ast.AugAssign) else None
        if isinstance(item, ast.Assign) and len(item.targets) == 1:
            target = item.targets[0]
        if isinstance(target, ast.Name) and target.id == counter:
            updates.append(item)
    if len(updates) != 1:
        raise ValueError("Unsupported repeated counter update")
    expected = {
        "newton": "iters_count = selp(active, "
        "int32(iters_count + int32(1)), iters_count)",
        "bicgstab": "iter_count = selp(not finished, "
        "int32(iter_count + int32(1)), iter_count)",
        "mr": "iter_count += int32(1)",
        "sd": "iter_count += int32(1)",
    }[kind]
    if ast.unparse(updates[0]) != expected:
        raise ValueError("Unsupported counter admission semantics")
    body_calls = Counter(
        ast.unparse(item.func)
        for item in ast.walk(loop)
        if isinstance(item, ast.Call)
    )
    required_calls = (
        {
            "residual_function": 1,
            "linear_solver_fn": 1,
            "correction_norm_fn": 1,
        }
        if kind == "newton"
        else {
            "operator_apply": 2 if kind == "bicgstab" else 1,
            "preconditioner": 2 if kind == "bicgstab" else 1,
            "weighted_norm": 2 if kind == "bicgstab" else 1,
        }
    )
    if any(
        body_calls[name] != count for name, count in required_calls.items()
    ):
        raise ValueError("Recurrent helper call sites changed")
    return dict(
        group=group,
        loop_line=loop.lineno,
        vote=condition,
        counter_update=dict(line=updates[0].lineno, source=expected),
        counter_kind="warp_body_count"
        if kind in ("mr", "sd")
        else "active_lane_count",
        body_call_sites=dict(body_calls),
        selected_body_call_counts=required_calls,
        entry_statements=statements(
            ast.Module(
                body=node.body[: node.body.index(loop)], type_ignores=[]
            )
        ),
        body_statements=statements(loop),
        exit_statements=statements(
            ast.Module(
                body=node.body[node.body.index(loop) + 1 :], type_ignores=[]
            )
        ),
        body_execution="all lanes in the call-entry mask while any unfinished",
        exit_tests="B + I[B < cap]; BiCG zero-guess has an earlier vote",
        native_control_lowering="unresolved",
    )


def role_record(factory, role, graph, expanded, nonlinear=False):
    """Bind a role to its real callable, width, config and source controls."""
    function = python_function(factory.device_function)
    identifier = graph.identities.get(id(function))
    if identifier is None:
        raise ValueError(
            f"Actual role is absent from the callable graph: {role}"
        )
    record = expanded[identifier]
    config = factory.compile_settings
    actual_type = None if nonlinear else str(factory.linear_correction_type)
    kind = (
        "newton"
        if nonlinear
        else next(
            (
                name
                for name, value in PUBLIC_LINEAR_TYPES.items()
                if value == actual_type
            ),
            actual_type,
        )
    )
    expected_type = "NewtonKrylov" if nonlinear else LINEAR_TYPES.get(kind)
    if type(factory).__name__ != expected_type:
        raise ValueError("Unsupported actual inner solver class")
    width = int(config.solver_width)
    cap = int(factory.max_iters)
    closure = inspect.getclosurevars(function).nonlocals
    if "n_val" in closure and int(closure["n_val"]) != width:
        raise ValueError("Captured solver width differs from its settings")
    if kind != "lu":
        control = recurrent_control(function, kind)
        loops = [
            part
            for part in record["replicated_regions"]
            if part["unroll_group"] == control["group"]
        ]
        if len(loops) != 1 or loops[0]["iteration_upper_bound"] != cap:
            raise ValueError("Captured iteration cap differs from settings")
        iteration = dict(
            symbol=f"{'N' if nonlinear else 'K'}[{role},call,warp]",
            lower=1 if nonlinear else 0,
            upper=cap,
            source_region=loops[0],
            control=control,
            logged_counter=control["counter_kind"],
        )
    else:
        node = source_function(function)
        writes = [
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Assign)
            and any(
                ast.unparse(target) == "krylov_iters_out[0]"
                for target in item.targets
            )
        ]
        if len(writes) != 1 or ast.unparse(writes[0].value) != "int32(1)":
            raise ValueError("Direct solver no longer reports one call")
        iteration = dict(
            symbol=None,
            lower=1,
            upper=1,
            logged_counter="one_direct_call",
            source=dict(line=writes[0].lineno, source=ast.unparse(writes[0])),
            recurrent_loop=None,
        )
    group = buffer_registry._groups.get(factory)
    buffers = (
        []
        if group is None
        else [
            dict(
                name=name,
                elements=int(entry.size),
                dtype=np.dtype(entry.dtype).name,
                itemsize=int(entry.itemsize),
                location=entry.location,
                persistent=bool(entry.persistent),
            )
            for name, entry in group.entries.items()
        ]
    )
    return dict(
        role=role,
        function=identifier,
        solver_type=kind,
        actual_linear_correction_type=actual_type,
        factory_type=type(factory).__name__,
        width=width,
        cap=cap,
        zero_initial_guess=None
        if nonlinear
        else bool(config.zero_initial_guess),
        norm_reference=None if nonlinear else str(config.norm_reference),
        norm_factory=type(factory.norm).__name__,
        norm_atol=snapshot(factory.norm.atol),
        norm_rtol=snapshot(factory.norm.rtol),
        preconditioned=bool(closure.get("preconditioned", False)),
        use_cached_auxiliaries=bool(
            closure.get("use_cached_auxiliaries", False)
        ),
        iteration=iteration,
        buffers=buffers,
        source=record["source"],
        closure_sha256=record["closure_sha256"],
        closure_constants=record["closure_constants"],
        calls=record["calls"],
        operations=record["operations"],
    )


def describe_implicit_workload(solver, unroll=None):
    """Describe actual implicit roles without compiling or changing flags.

    Parameters
    ----------
    solver
        Constructed Solver with no native batch or helper overloads.
    unroll : dict, optional
        Candidate source-expansion directives; actual closures stay intact.

    Returns
    -------
    dict
        Actual role settings, source regions, symbolic calls and masks.
    """
    expansion = describe_expansion(solver, unroll)
    step = solver.kernel.single_integrator._algo_step
    family = FAMILIES.get(type(step).__name__)
    if family is None:
        raise ValueError("This adapter admits DIRK, FIRK and Rosenbrock-W")
    root_function = step.step_function
    graph = CapturedGraph()
    root = graph.add_function(root_function, "algorithm_step")
    expanded = {item["id"]: item for item in expansion["functions"]}
    if set(graph.callables) != set(expanded):
        raise ValueError("Actual callable graph changed during extraction")
    for identifier, actual in graph.callables.items():
        if (
            inspect.getsourcefile(actual)
            != expanded[identifier]["source"]["source_path"]
        ):
            if (
                Path(inspect.getsourcefile(actual)).resolve()
                != Path(
                    expanded[identifier]["source"]["source_path"]
                ).resolve()
            ):
                raise ValueError("Captured function IDs do not join")
    closure = inspect.getclosurevars(root_function.py_func).nonlocals
    config = step.compile_settings
    n, stages = int(config.n), int(config.tableau.stage_count)
    roles = {}
    if family != "ROS":
        roles["main_newton"] = role_record(
            step.solver,
            "main_newton",
            graph,
            expanded,
            nonlinear=True,
        )
    roles["main_linear"] = role_record(
        step.linear_solver,
        "main_linear",
        graph,
        expanded,
    )
    smoothing = bool(closure.get("use_smoothed_error", False))
    if smoothing:
        actual_error = (
            step.linear_solver if family == "ROS" else step.error_solver
        )
        roles["error_linear"] = role_record(
            actual_error,
            "error_linear",
            graph,
            expanded,
        )
        if (roles["error_linear"]["solver_type"] == "lu") != (
            roles["main_linear"]["solver_type"] == "lu"
        ):
            raise ValueError(
                "Mixed direct/iterative public wiring is not admitted"
            )
    width = stages * n if family == "FIRK" else n
    if roles["main_linear"]["width"] != width:
        raise ValueError("Main solver has an unexpected actual family width")
    if smoothing and roles["error_linear"]["width"] != n:
        raise ValueError("Error solver has an unexpected actual family width")
    if family == "DIRK":
        diagonal = [float(value) for value in config.tableau.diagonal]
        calls = [
            dict(
                id=f"stage{stage}",
                role="main_newton",
                width=n,
                stage=stage,
                admission_mask="M[not use_cached_rhs]"
                if stage == 0
                else "M[step]",
                multiplicity=1,
            )
            for stage, value in enumerate(diagonal)
            if value != 0
        ]
    elif family == "FIRK":
        calls = [
            dict(
                id="coupled",
                role="main_newton",
                width=width,
                stages=list(range(stages)),
                admission_mask="M[step]",
                multiplicity=1,
            )
        ]
    else:
        calls = [
            dict(
                id=f"stage{stage}",
                role="main_linear",
                width=n,
                stage=stage,
                admission_mask="M[step]",
                multiplicity=1,
            )
            for stage in range(stages)
        ]
    if smoothing:
        calls.append(
            dict(
                id="error",
                role="error_linear",
                width=n,
                admission_mask="M[step]",
                multiplicity=1,
            )
        )
    for dispatcher in graph.dispatchers:
        if dispatcher.overloads:
            raise ValueError("Source description created a native overload")
    return dict(
        schema=1,
        kind="actual_implicit_workload",
        family=family,
        n_states=n,
        stage_count=stages,
        root=root,
        roles=roles,
        step_calls=calls,
        smoothing_enabled=smoothing,
        actual_step_settings=dict(
            inexact_newton=bool(getattr(config, "inexact_newton", False)),
            prefactored=bool(getattr(config, "prefactored", False)),
            cached_solve=bool(
                closure.get("use_cached_solve", family == "ROS")
            ),
            tableau=type(config.tableau).__name__,
            diagonal=snapshot(getattr(config.tableau, "diagonal", ())),
            stage_source=expanded[root]["source"],
        ),
        registry=registry_layout(step),
        candidate=expansion["candidate"],
        functions=expansion["functions"],
        source_inventory=expansion["source_inventory_operations"],
        compilation_check=expansion["compilation_check"],
        provenance=dict(
            adapter=source_receipt(describe_implicit_workload),
            expansion=expansion["provenance"],
            package_source_hash=package_source_hash(),
            toolchain_fingerprint=toolchain_fingerprint(),
        ),
        assumptions=[
            "Step-call masks are scenario inputs, not inferred from means",
            "Newton invokes its linear solver even for frozen Newton lanes",
            "MR/SD log warp bodies; BiCGSTAB logs active lane iterations",
            "LU logs one direct call per entered lane",
            "Smoothed-error work uses its own actual role and width",
            "Source body replication is separate from dynamic call counts",
        ],
        limitations=[
            "Native instruction/register/service terms remain unresolved",
            "No guessed warp masks from aggregated per-run counter totals",
            "Unknown helper/control lowering stays in source residuals",
            "Mixed solver wiring needs an actual public construction",
            "Existing workload.py error iteration labels are not consumed",
        ],
    )


def mask(value, name):
    """Admit a nonnegative 32-lane bit mask."""
    exact_int(value, name)
    if value >= 1 << 32:
        raise ValueError("Mask exceeds the warp width")
    return value


def linear_regime(role, scenario):
    """Count calls and returned lane counters for an explicit mask trace."""
    entered = mask(scenario["entry_mask"], "entry mask")
    if not entered:
        raise ValueError("A recorded call needs at least one entered lane")
    kind = role["solver_type"]
    if kind == "lu":
        if scenario.get("active_masks", []):
            raise ValueError("LU has no Krylov iteration mask sequence")
        return dict(
            body_iterations=0,
            direct_calls=1,
            returned_lane_counts=[
                int(bool(entered & (1 << lane))) for lane in range(32)
            ],
        )
    sequence = scenario["active_masks"]
    if len(sequence) > role["cap"]:
        raise ValueError("Krylov scenario exceeds the captured cap")
    previous = entered
    for item in sequence:
        current = mask(item, "active mask")
        if not current or current & ~previous:
            raise ValueError("Krylov active masks must shrink until exit")
        previous = current
    terminal = mask(scenario["terminal_active_mask"], "terminal mask")
    if terminal & ~previous or (len(sequence) < role["cap"] and terminal):
        raise ValueError("Krylov termination contradicts its mask/cap")
    counts = [
        sum(bool(item & (1 << lane)) for item in sequence)
        if kind == "bicgstab"
        else len(sequence) * int(bool(entered & (1 << lane)))
        for lane in range(32)
    ]
    bodies = len(sequence)
    calls = role["iteration"]["control"]["selected_body_call_counts"]
    zero_guess = role["zero_initial_guess"]
    bicg_early_return = kind == "bicgstab" and zero_guess and bodies == 0
    top_votes = 0 if bicg_early_return else bodies + int(bodies < role["cap"])
    return dict(
        body_iterations=bodies,
        direct_calls=0,
        returned_lane_counts=counts,
        operator_calls=calls["operator_apply"] * bodies
        + int(not role["zero_initial_guess"]),
        preconditioner_calls=calls["preconditioner"]
        * bodies
        * int(role["preconditioned"]),
        norm_calls=1 + int(not zero_guess) + calls["weighted_norm"] * bodies,
        initial_votes=int(kind == "bicgstab" and zero_guess),
        loop_top_votes=top_votes,
        seed_region_executed=not bicg_early_return,
        body_control_entry_lanes=entered.bit_count(),
        returned_counter_lanes=entered.bit_count(),
        counts_are_not_active_thread_instruction_counts=True,
    )


def evaluate_regime(descriptor, scenarios, step_entry_mask=None):
    """Evaluate explicit per-call masks without predicting convergence.

    Each step-call entry names a descriptor call ID. Newton entries carry
    one active mask and nested linear scenario per entered body. Direct
    linear entries use ``linear_regime``. Missing calls are not zero work.
    """
    calls = {item["id"]: item for item in descriptor["step_calls"]}
    if set(scenarios) != set(calls):
        raise ValueError("A regime must account for every actual step call")
    unconditional = {
        mask(scenarios[name]["entry_mask"], "entry mask")
        for name, call in calls.items()
        if call["admission_mask"] == "M[step]"
    }
    if step_entry_mask is None:
        if len(unconditional) != 1:
            raise ValueError("Unconditional calls need one step-entry mask")
        step_entry_mask = next(iter(unconditional))
    step_entry_mask = mask(step_entry_mask, "step-entry mask")
    if not step_entry_mask or any(
        entered != step_entry_mask for entered in unconditional
    ):
        raise ValueError("Call entry differs from the step-entry mask")
    for name, call in calls.items():
        entered = mask(scenarios[name]["entry_mask"], "entry mask")
        if call["admission_mask"] not in (
            "M[step]", "M[not use_cached_rhs]"
        ):
            raise ValueError("Unsupported step-call admission condition")
        if entered & ~step_entry_mask:
            raise ValueError("Conditional call includes an unentered lane")
    results = {}
    totals = dict(newton_bodies=0, krylov_bodies=0, direct_calls=0)
    logged = [[0] * 32, [0] * 32]
    for name, call in calls.items():
        scenario = scenarios[name]
        entered = mask(scenario["entry_mask"], "entry mask")
        if entered == 0:
            if call["admission_mask"] != "M[not use_cached_rhs]":
                raise ValueError("An unconditional call cannot disappear")
            results[name] = dict(skipped_by_explicit_call_mask=True)
            continue
        role = descriptor["roles"][call["role"]]
        if role["solver_type"] != "newton":
            result = linear_regime(role, scenario)
            totals["krylov_bodies"] += result["body_iterations"]
            totals["direct_calls"] += result["direct_calls"]
            logged[1] = [
                a + b
                for a, b in zip(
                    logged[1], result["returned_lane_counts"], strict=True
                )
            ]
            results[name] = result
            continue
        sequence = scenario["active_masks"]
        nested = scenario["linear_calls"]
        if not 1 <= len(sequence) <= role["cap"] or len(nested) != len(
            sequence
        ):
            raise ValueError("Newton needs one linear call per admitted body")
        if sequence[0] != entered:
            raise ValueError("Newton starts every entered lane active")
        previous = entered
        iterations = []
        for active, linear in zip(sequence, nested, strict=True):
            active = mask(active, "Newton active mask")
            if (
                not active
                or active & ~previous
                or linear["entry_mask"] != entered
            ):
                raise ValueError(
                    "Newton masks/call entry violate source control"
                )
            previous = active
            result = linear_regime(descriptor["roles"]["main_linear"], linear)
            iterations.append(result)
            totals["newton_bodies"] += 1
            totals["krylov_bodies"] += result["body_iterations"]
            totals["direct_calls"] += result["direct_calls"]
            for lane, count in enumerate(result["returned_lane_counts"]):
                if active & (1 << lane):
                    logged[0][lane] += 1
                    logged[1][lane] += count
        terminal = mask(
            scenario["terminal_active_mask"], "Newton terminal mask"
        )
        if terminal & ~previous or (len(sequence) < role["cap"] and terminal):
            raise ValueError("Newton termination contradicts its mask/cap")
        results[name] = dict(
            body_iterations=len(sequence),
            linear_calls=iterations,
            loop_top_votes=len(sequence) + int(len(sequence) < role["cap"]),
            residual_calls=len(sequence),
            correction_norm_calls=len(sequence),
            body_control_entry_lanes=entered.bit_count(),
        )
    return dict(
        status="EXPLICIT_SYMBOLIC_REGIME_EVALUATED",
        step_entry_mask=step_entry_mask,
        calls=results,
        warp_body_totals=totals,
        logged_lane_counters=logged,
        limitations=[
            "Masks are supplied assumptions, not predicted convergence",
            "Different step-entry masks need separate warp instances",
            "Body totals are not dynamic native instruction counts",
        ],
    )


def main():
    """Construct one real source-only implicit family into a fresh cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system", choices=tuple(placement.SYSTEMS), required=True
    )
    parser.add_argument("--algo", required=True)
    parser.add_argument(
        "--linear-solver", choices=tuple(LINEAR_TYPES), required=True
    )
    parser.add_argument(
        "--inexact", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--prefactored", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--unroll", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cache = args.output / "codegen"
    cache.mkdir(exist_ok=False)
    previous = get_cache_root_override()
    solver = None
    try:
        set_cache_root(cache.resolve())
        system = placement.SYSTEMS[args.system]["build"]()
        kwargs = placement.solver_kwargs(args.system, args.algo)
        kwargs["linear_correction_type"] = PUBLIC_LINEAR_TYPES[
            args.linear_solver
        ]
        if args.inexact is not None:
            kwargs["inexact_newton"] = args.inexact
        if args.prefactored is not None:
            kwargs["prefactored"] = args.prefactored
        solver = Solver(system, **kwargs)
        candidate = (
            None
            if args.unroll is None
            else json.loads(args.unroll.read_text())
        )
        result = describe_implicit_workload(solver, candidate)
        result["construction"] = dict(
            system=args.system,
            algo=args.algo,
            linear_solver=args.linear_solver,
            inexact=args.inexact,
            prefactored=args.prefactored,
            cache_root=str(cache.resolve()),
            config_hash=solver.kernel.config_hash,
            fn_hash=system.fn_hash,
        )
        write_json(args.output / "descriptor.json", result)
        (args.output / "observer.py").write_bytes(SCRIPT.read_bytes())
        print(
            json.dumps(
                dict(
                    status="SOURCE_ONLY_IMPLICIT_WORKLOAD_PASS",
                    family=result["family"],
                    roles={
                        key: dict(
                            width=value["width"],
                            cap=value["cap"],
                            solver_type=value["solver_type"],
                        )
                        for key, value in result["roles"].items()
                    },
                    **result["compilation_check"],
                )
            )
        )
    finally:
        if solver is not None:
            solver.close()
        set_cache_root(previous)


if __name__ == "__main__":
    main()
