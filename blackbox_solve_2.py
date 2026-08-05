"""Profile a black-box ODE with Nsight Compute."""
import argparse
import os
import sys
import time
from pathlib import Path


def _parse_scalar(text):
    """Coerce a ``key=value`` string value to int/float/bool/str."""
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="ncu harness for the opaque blackbox2 ODE system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--system", type=int, default=2, choices=(1, 2),
        help="Which blackbox model to load (this script targets 2).",
    )
    p.add_argument("--algorithm", default="crank_nicolson",
                   help="Solver algorithm name (euler, backwards_euler, "
                        "backwards_euler_pc, crank_nicolson, dirk, firk, "
                        "erk, rosenbrock, or a tableau alias).")
    p.add_argument("--linear-correction-type", default="bicgstab",
                   choices=("bicgstab", "minimal_residual",
                            "steepest_descent"),
                   help="Matrix-free linear solver used by the implicit "
                        "step. Only applies to Newton-Krylov algorithms "
                        "(euler/erk are explicit and ignore it).")
    p.add_argument("--duration", type=float, default=0.01,
                   help="Integration duration in system time units.")
    p.add_argument("--dt", type=float, default=1e-3,
                   help="Step size (fixed) / initial step (adaptive).")
    p.add_argument("--precision", default="float64",
                   choices=("float32", "float64"),
                   help="blackbox2 needs float64 for finite dynamics.")
    p.add_argument("--nruns", type=int, default=8192,
                   help="Batch size. Does not affect PTX/SASS; keep low "
                        "enough to stay a single (unchunked) launch.")
    p.add_argument("--blocksize", type=int, default=256,
                   help="CUDA block size.")
    p.add_argument("--grid-type", default="verbatim",
                   choices=("verbatim", "combinatorial"),
                   help="Input grid construction strategy.")
    p.add_argument("--output-types", nargs="+", default=["state"],
                   help="Saved output categories (>=1). Narrow to just "
                        "'state' for the smallest footprint / single "
                        "launch. Options: state observables time "
                        "iteration_counters summaries.")
    p.add_argument("--settling-time", type=float, default=0.0,
                   help="Warm-up period before outputs are recorded.")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warmup solves before the profiled ones. Match to "
                        "ncu --launch-skip.")
    p.add_argument("--reps", type=int, default=1,
                   help="Profiled solves. Match to ncu --launch-count.")
    p.add_argument("--cache", dest="cache", action="store_true",
                   help="Enable the compiled-kernel disk cache.")
    p.add_argument("--no-cache", dest="cache", action="store_false",
                   help="Disable the cache so compile runs fresh (default).")
    p.set_defaults(cache=False)
    p.add_argument("--time-logging", default=None,
                   help="cubie time_logging_level (e.g. summary, detailed).")
    p.add_argument("--solver-kwarg", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Extra Solver kwarg (repeatable). Value parsed as "
                        "int/float/bool/str, e.g. --solver-kwarg "
                        "krylov_max_iters=100.")
    p.add_argument("--dump-asm", metavar="DIR", default=None,
                   help="Write PTX/SASS/LLVM + metadata for the kernel to "
                        "DIR for offline diffing.")
    p.add_argument("--workdir", default=None,
                   help="chdir here before importing cubie so ./generated "
                        "lands in a known, per-backend location. Defaults "
                        "to this script's directory (the repo root).")
    args = p.parse_args(argv)
    if args.duration <= 0.0:
        p.error("--duration must be positive")
    if args.dt <= 0.0:
        p.error("--dt must be positive")
    if args.nruns < 1:
        p.error("--nruns must be positive")
    if args.blocksize < 1:
        p.error("--blocksize must be positive")
    if args.warmup < 1:
        p.error("--warmup must be at least 1")
    if args.reps < 1:
        p.error("--reps must be at least 1")
    return args


def _backend_tag():
    """Return the active CUDA backend."""
    from cubie.cuda_backend import CUDA_BACKEND

    return CUDA_BACKEND


def _backend_metadata():
    import cubie
    meta = {
        "backend": _backend_tag(),
        "cubie_file": cubie.__file__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    try:
        import numba
        meta["numba"] = numba.__version__
    except Exception:
        pass
    for modname in ("numba_cuda", "numba_cuda_mlir"):
        try:
            mod = __import__(modname)
            meta[modname] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return meta


def _load_system(system_no, precision):
    """Load an extensionless black-box model as BigModel."""
    import shutil
    import tempfile
    from cubie import load_bigmodel_file

    here = Path(__file__).resolve().parent
    source = here / "blackbox" / f"blackbox{system_no}"
    if not source.is_file():
        raise SystemExit(f"blackbox model not found: {source}")
    name = f"blackboxsystem_{system_no}"
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / f"{name}.bigmodel"
        shutil.copyfile(source, staged)
        return load_bigmodel_file(str(staged), name=name, precision=precision)


def _dump_asm(dispatcher, out_dir, tag):
    """Best-effort dump of PTX/SASS/LLVM for offline diffing."""
    import json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    inspectors = (
        ("ptx", "inspect_asm"),
        ("sass", "inspect_sass"),
        ("llvm", "inspect_llvm"),
    )
    for ext, method in inspectors:
        fn = getattr(dispatcher, method, None)
        if fn is None:
            continue
        try:
            result = fn()
        except Exception as exc:  # nvdisasm missing, etc.
            print(f"DUMP  {ext}: unavailable ({exc!r})")
            continue
        if isinstance(result, dict):
            text = "\n\n".join(
                f"// signature: {sig}\n{body}"
                for sig, body in result.items()
            )
        else:
            text = str(result)
        path = out / f"integration_kernel_{tag}.{ext}"
        path.write_text(text, encoding="utf-8")
        written.append(str(path))
        print(f"DUMP  {ext} -> {path}")
    meta = _backend_metadata()
    try:
        stats = {}
        regs = dispatcher.get_regs_per_thread()
        lmem = dispatcher.get_local_mem_per_thread()
        stats["regs_per_thread"] = list(regs.values())
        stats["local_mem_per_thread"] = list(lmem.values())
        meta["kernel_stats"] = stats
    except Exception as exc:
        meta["kernel_stats_error"] = repr(exc)
    (out / f"metadata_{tag}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"DUMP  metadata -> {out / f'metadata_{tag}.json'}")
    return written


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # chdir *before* importing cubie: GENERATED_DIR is captured from cwd
    # at import time, so this pins ./generated per backend.
    workdir = args.workdir or str(Path(__file__).resolve().parent)
    os.chdir(workdir)

    import numpy as np

    tag = _backend_tag()
    precision = np.float64 if args.precision == "float64" else np.float32

    print(f"BACKEND  {tag}  (cwd={os.getcwd()})")

    t0 = time.perf_counter()
    system = _load_system(args.system, precision)
    t_load = time.perf_counter() - t0
    print(f"COMPILE  model_load: {t_load:.3f} s "
          f"(states={system.sizes.states}, "
          f"observables={system.sizes.observables})")

    from cubie.batchsolving.solver import Solver

    solver_kwargs = {
        "algorithm": args.algorithm,
        "dt": args.dt,
        "linear_correction_type": args.linear_correction_type,
        "output_types": list(args.output_types),
        "time_logging_level": args.time_logging,
        "cache": args.cache,
    }
    for item in args.solver_kwarg:
        if "=" not in item:
            raise SystemExit(f"--solver-kwarg needs KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        solver_kwargs[key.strip()] = _parse_scalar(value.strip())

    t0 = time.perf_counter()
    solver = Solver(system, **solver_kwargs)
    t_build = time.perf_counter() - t0
    print(f"COMPILE  solver_construct: {t_build:.3f} s")

    # Verbatim inputs: one column per run, defaults from the system.
    init = system.initial_values.values_array.astype(precision)
    par = system.parameters.values_array.astype(precision)
    y0 = np.repeat(init[:, None], args.nruns, axis=1)
    params = np.repeat(par[:, None], args.nruns, axis=1)

    def _solve():
        return solver.solve(
            y0, params,
            duration=args.duration,
            settling_time=args.settling_time,
            grid_type=args.grid_type,
            blocksize=args.blocksize,
        )

    # First solve triggers the CUDA JIT + ptxas.
    t0 = time.perf_counter()
    _solve()
    t_first = time.perf_counter() - t0
    print(f"COMPILE  first_solve_jit: {t_first:.3f} s "
          f"(total_in={t_load + t_build + t_first:.3f} s)")

    # Remaining warmup launches (already compiled).
    for _ in range(args.warmup - 1):
        _solve()

    dispatcher = solver.kernel.kernel
    try:
        regs = list(dispatcher.get_regs_per_thread().values())[0]
        lmem = list(dispatcher.get_local_mem_per_thread().values())[0]
        print(f"KERNEL   regs/thread: {regs}, local_mem/thread: {lmem} B")
    except Exception as exc:
        print(f"KERNEL   dispatcher stats unavailable: {exc!r}")

    bsk = solver.kernel
    try:
        pad = 4 if bsk.shared_memory_needs_padding else 0
        bytes_per_run = bsk.shared_memory_bytes + pad
        smem = int(bytes_per_run * min(args.nruns, args.blocksize))
        eff_bs, smem = bsk.limit_blocksize(
            args.blocksize, smem, bytes_per_run, args.nruns
        )
        blocks = max(1, -(-args.nruns // eff_bs))
        print(f"KERNEL   launch: {args.nruns} runs, "
              f"blocksize {args.blocksize}->{eff_bs}, {blocks} blocks, "
              f"{bytes_per_run} B shared/run, {smem} B dyn-shared/block")
    except Exception as exc:
        print(f"KERNEL   launch-shape unavailable: {exc!r}")

    # Profiled launches (ncu attaches here via --launch-skip=warmup).
    times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        _solve()
        times.append(time.perf_counter() - t0)
    print("HOT      "
          + ", ".join(f"{t * 1e3:.2f}" for t in times)
          + f" ms (min {min(times) * 1e3:.2f} ms)")

    if args.dump_asm:
        _dump_asm(dispatcher, args.dump_asm, tag)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
