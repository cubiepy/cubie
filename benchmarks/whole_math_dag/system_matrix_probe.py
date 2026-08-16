"""Compile one system/ordering configuration, report JSON, serve solves."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

BLOCK_SIZE = 64


def _chain_equations(states, depth):
    # Independent per-state chains of alternating transcendentals.
    lines = []
    outputs = []
    for k in range(states):
        prev = f"c{k}_0"
        lines.append(f"{prev} = exp(0.03 * x{k})")
        for i in range(1, depth):
            fn = "sin" if i % 2 else "cos"
            name = f"c{k}_{i}"
            lines.append(f"{name} = {fn}({prev} + 0.7)")
            prev = name
        outputs.append(prev)
    for k in range(states):
        lines.append(f"dx{k} = -x{k} + {outputs[k]}")
    return "\n".join(lines)


def _wide_equations(states, width):
    # Many independent transcendentals, summed per state.
    lines = []
    terms = {k: [] for k in range(states)}
    for i in range(width):
        k = i % states
        fn = ("sin", "cos", "tanh")[i % 3]
        scale = 0.1 * (1 + i % 5)
        lines.append(f"t{i} = {fn}({scale:.1f} * x{k} + {0.01 * i:.2f})")
        terms[k].append(f"t{i}")
    for k in range(states):
        lines.append(f"dx{k} = -x{k} + ({' + '.join(terms[k])}) / {len(terms[k])}")
    return "\n".join(lines)


def _diamond_equations(states, layers):
    # Per-state diamond stacks with a cross-state mix every 8 layers.
    lines = []
    value = {k: f"x{k}" for k in range(states)}
    for layer in range(layers):
        for k in range(states):
            a = f"a{k}_{layer}"
            b = f"b{k}_{layer}"
            v = f"v{k}_{layer}"
            lines.append(f"{a} = sin({value[k]})")
            lines.append(f"{b} = cos({value[k]})")
            lines.append(f"{v} = {a} * {b} + 0.5 * ({a} + {b})")
            value[k] = v
        if layer % 8 == 7:
            mixed = {}
            for k in range(states):
                m = f"m{k}_{layer}"
                lines.append(
                    f"{m} = {value[k]} + 0.125 * {value[(k + 1) % states]}"
                )
                mixed[k] = m
            value = mixed
    for k in range(states):
        lines.append(f"dx{k} = -x{k} + {value[k]}")
    return "\n".join(lines)


def _shared_prefix_equations(stages):
    # One shared chain; two cascaded sums tap every stage.
    lines = ["s0 = exp(0.02 * x0)", "p0 = s0", "q0 = 2 * s0"]
    for i in range(1, stages):
        sign_p = "+" if i % 2 == 0 else "-"
        sign_q = "-" if i % 2 == 0 else "+"
        lines.append(f"s{i} = sin(s{i - 1} + 0.05)")
        lines.append(f"p{i} = p{i - 1} {sign_p} {1 + i % 3} * s{i}")
        lines.append(f"q{i} = q{i - 1} {sign_q} {1 + i % 4} * s{i}")
    lines.append(f"dx0 = -x0 + p{stages - 1} / {stages}")
    lines.append(f"dx1 = -x1 + q{stages - 1} / {stages}")
    return "\n".join(lines)


def build_solver(system_name, ordering):
    import cubie as qb

    fixed_kwargs = dict(
        algorithm="classical-rk4",
        dt=0.001,
        step_controller="fixed",
        output_types=["state"],
        time_logging_level="default",
        operation_ordering=ordering,
    )
    if system_name == "lorenz_fixed" or system_name == "lorenz_adaptive":
        system = qb.create_ODE_system(
            """
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            """,
            states={"x": 1.0, "y": 0.0, "z": 0.0},
            parameters={"rho": 21.0},
            constants={"sigma": 10.0, "beta": 8.0 / 3.0},
            name=f"Lorenz_{ordering}",
            precision=np.float32,
            operation_ordering=ordering,
        )
        if system_name == "lorenz_adaptive":
            solver = qb.Solver(
                system,
                algorithm="radau",
                preconditioner_type="jacobi",
                atol=1e-6,
                rtol=1e-6,
                dt_min=1e-12,
                dt_max=1e3,
                step_controller="pid",
                kp=6 / 5,
                kd=0.0,
                ki=0.0,
                output_types=["state"],
                time_logging_level="default",
                operation_ordering=ordering,
            )
            runs, duration = 2**20, 1.0
        else:
            solver = qb.Solver(system, save_every=1.0, **fixed_kwargs)
            runs, duration = 2**20, 1.0
        first = "x"
    else:
        if system_name == "chains_serial":
            equations, states = _chain_equations(8, 64), 8
        elif system_name == "chains_wide":
            equations, states = _wide_equations(4, 256), 4
        elif system_name == "diamonds":
            equations, states = _diamond_equations(4, 32), 4
        elif system_name == "shared_prefix":
            equations, states = _shared_prefix_equations(256), 2
        else:
            raise SystemExit(f"unknown system {system_name!r}")
        system = qb.create_ODE_system(
            equations,
            states={f"x{k}": 0.5 for k in range(states)},
            name=f"{system_name}_{ordering}",
            precision=np.float32,
            operation_ordering=ordering,
        )
        solver = qb.Solver(system, save_every=0.005, **fixed_kwargs)
        runs, duration = 2**13, 0.005
        first = "x0"
    inits, params = solver.build_grid(
        {first: np.linspace(0.1, 1.0, runs, dtype=np.float32)}, None
    )
    return solver, inits, params, duration


def solve_once(solver, inits, params, duration):
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(
            inits,
            params,
            duration=duration,
            blocksize=BLOCK_SIZE,
            nan_error_trajectories=False,
        )
    kernel_ms = sum(
        event.elapsed_time_ms()
        for event in solver.kernel._cuda_events
        if event.name.startswith("kernel_chunk")
    )
    state = np.asarray(result.state)
    status = np.asarray(result.status_codes, dtype=np.int32)
    return kernel_ms, state, status


def kernel_report(solver):
    dispatcher = solver.kernel.kernel
    overloads = list(dispatcher.overloads.values())
    if not overloads:
        overloads = list(
            {
                id(k): k
                for k in getattr(
                    dispatcher, "_launch_config_overloads", {}
                ).values()
            }.values()
        )
    kernel = overloads[0]
    if hasattr(kernel, "_ensure_kernel_attrs"):
        kernel._ensure_kernel_attrs()
    cubin = kernel.metadata.get("cubin")
    return {
        "typed_block_scheduler": kernel.metadata.get(
            "typed_block_scheduler"
        ),
        "regs_per_thread": int(kernel.regs_per_thread),
        "local_mem_per_thread": int(kernel.local_mem_per_thread),
        "cubin_sha256": (
            hashlib.sha256(cubin).hexdigest() if cubin else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--ordering", default="kahn")
    parser.add_argument("--save-state", type=Path, default=None)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    solver, inits, params, duration = build_solver(
        args.system, args.ordering
    )
    kernel_ms, state, status = solve_once(solver, inits, params, duration)
    compile_seconds = time.perf_counter() - start
    if args.save_state is not None:
        np.save(args.save_state, state)

    report = {
        "system": args.system,
        "ordering": args.ordering,
        "block_schedule": os.environ.get("CUBIE_BLOCK_SCHEDULE"),
        "compile_wall_seconds": compile_seconds,
        "warmup_kernel_ms": kernel_ms,
        "state_finite": bool(np.isfinite(state).all()),
        "status_nonzero": int(np.count_nonzero(status)),
        "kernel": kernel_report(solver),
    }
    print("@REPORT " + json.dumps(report), flush=True)
    if args.serve:
        for command in iter(input, "quit"):
            if command == "run":
                ms, _, _ = solve_once(solver, inits, params, duration)
                print("@SOLVE %.6f" % ms, flush=True)


if __name__ == "__main__":
    main()
