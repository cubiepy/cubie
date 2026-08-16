"""Compile one Fabbri radau/jacobi f32 configuration, report JSON.

One process = one configuration (--ordering, --fused; policy via
CUBIE_BLOCK_SCHEDULE). --serve answers ``run`` commands with
per-solve kernel times. Set a per-configuration CUBIE_CACHE_DIR.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np

DEFAULT_PICKLE = Path(
    os.environ.get(
        "FABBRI_PICKLE",
        r"C:\local_working_projects\cubie-744\generated"
        r"\Fabbri_Linder\cache_242b8c131464ae40.pkl",
    )
)
RUNS = 43008
BLOCK_SIZE = 64
DURATION = 0.0012


def build_solver(ordering, fused):
    import pickle

    import cubie as qb
    from cubie.odesystems.symbolic.symbolicODE import SymbolicODE

    data = pickle.loads(DEFAULT_PICKLE.read_bytes())
    system = SymbolicODE(
        equations=data["parsed_equations"],
        precision=data["precision"],
        all_indexed_bases=data["indexed_bases"],
        all_symbols=data["all_symbols"],
        fn_hash=data["fn_hash"],
        user_functions=data["user_functions"],
        name=f"Fabbri_probe_{ordering}_{int(fused)}",
        mass=data["mass"],
        operation_ordering=ordering,
    )
    solver = qb.Solver(
        system,
        algorithm="radau",
        preconditioner_type="jacobi",
        fuse_operator_preconditioner=fused,
        use_smoothed_error=False,
        newton_max_iters=5,
        krylov_max_iters=50,
        atol=1e-4,
        rtol=1e-4,
        output_types=["state"],
        time_logging_level="default",
        operation_ordering=ordering,
    )
    first = next(iter(system.initial_values.values_dict))
    inits, params = solver.build_grid(
        {
            first: np.full(
                RUNS,
                system.initial_values.values_dict[first],
                dtype=np.float32,
            )
        },
        None,
    )
    return solver, inits, params


def solve_once(solver, inits, params):
    with contextlib.redirect_stdout(io.StringIO()):
        result = solver.solve(
            inits,
            params,
            duration=DURATION,
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
    return {
        "kernel_ms": kernel_ms,
        "state_finite": bool(np.isfinite(state).all()),
        "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "status_nonzero": int(np.count_nonzero(status)),
    }


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
    if len(overloads) != 1:
        raise RuntimeError(
            f"expected 1 overload, got {len(overloads)}"
        )
    kernel = overloads[0]
    if hasattr(kernel, "_ensure_kernel_attrs"):
        kernel._ensure_kernel_attrs()
    cubin = kernel.metadata.get("cubin")
    source_path = Path(dispatcher.py_func.__code__.co_filename)
    return {
        "typed_block_scheduler": kernel.metadata.get(
            "typed_block_scheduler"
        ),
        "regs_per_thread": int(kernel.regs_per_thread),
        "local_mem_per_thread": int(kernel.local_mem_per_thread),
        "shared_mem_per_block": int(kernel.shared_mem_per_block),
        "cubin_sha256": (
            hashlib.sha256(cubin).hexdigest() if cubin else None
        ),
        "cubin_bytes": len(cubin) if cubin else None,
        "generated_source_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "generated_source_path": str(source_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordering", default="dfs")
    parser.add_argument("--fused", type=int, default=0)
    parser.add_argument("--timing-solves", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--save-state", type=Path, default=None)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    start = time.perf_counter()
    solver, inits, params = build_solver(
        args.ordering, bool(args.fused)
    )
    warmup = solve_once(solver, inits, params)
    compile_seconds = time.perf_counter() - start
    if args.save_state is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            result = solver.solve(
                inits,
                params,
                duration=DURATION,
                blocksize=BLOCK_SIZE,
                nan_error_trajectories=False,
            )
        np.save(args.save_state, np.asarray(result.state))

    report = {
        "ordering": args.ordering,
        "fused": bool(args.fused),
        "block_schedule": os.environ.get("CUBIE_BLOCK_SCHEDULE"),
        "hashseed": os.environ.get("PYTHONHASHSEED"),
        "cache_dir": os.environ.get("CUBIE_CACHE_DIR"),
        "compile_wall_seconds": compile_seconds,
        "warmup": warmup,
        "kernel": kernel_report(solver),
        "timings_kernel_ms": [],
    }
    for _ in range(args.timing_solves):
        report["timings_kernel_ms"].append(
            solve_once(solver, inits, params)["kernel_ms"]
        )

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print("@REPORT " + json.dumps(report), flush=True)
    if args.serve:
        for command in iter(input, "quit"):
            if command == "run":
                print(
                    "@SOLVE %.6f"
                    % solve_once(solver, inits, params)["kernel_ms"],
                    flush=True,
                )


if __name__ == "__main__":
    main()
