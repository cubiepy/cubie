"""Capture source-contract-disabled IR without executing a solver kernel."""

import importlib.util
import json
from pathlib import Path
import traceback

import numpy as np

from cubie import Solver
from cubie.cache_root import set_cache_root
from benchmarks import placement_landscape as landscape


ROOT = Path(__file__).parent
VERIFY = ROOT.parent
RAW = VERIFY.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run():
    wrapper = load(
        VERIFY
        / "native_policy_profile_author_e5/source/native_policy_profile.py",
        "frozen_native_wrapper",
    )
    prior_author = load(
        VERIFY / "numerical_contraction_author_e2/diagnose.py",
        "contract_source_author",
    )
    original = wrapper.read(
        RAW / "numerical_contraction_native_e2/receipt.json"
    )
    prepared_path = wrapper.checked(original["prepared"])
    prepared = wrapper.load_prepared(prepared_path)
    request = wrapper.read(wrapper.checked(prepared["request"]))
    receipt = dict(
        status="STARTED",
        source=wrapper.asset(__file__),
        prior_receipt=wrapper.asset(
            RAW / "numerical_contraction_native_e2/receipt.json"
        ),
        prior_author=wrapper.asset(
            VERIFY / "numerical_contraction_author_e2/diagnose.py"
        ),
        prepared=wrapper.asset(prepared_path),
        records=[],
        scope="Native compilation and original IR retention only. No Solver.solve "
        "or explicit kernel launch. Public contract=False semantics reproduce "
        "the already measured flag intervention before a separate offline link.",
    )
    solver = None
    try:
        for source in ("0002", "0000"):
            case_id = "workload_006_source_" + source + "_b128_s102400"
            case = prepared["cases"][case_id]
            prior = next(
                row
                for row in original["records"]
                if row["case"] == case_id and row["mode"] == "contract_false"
            )
            folder = ROOT / ("source_" + source)
            folder.mkdir(exist_ok=False)
            kwargs, constants = wrapper.case_kwargs(
                request, case["workload"], case["candidate"]
            )
            prior_author.require(
                wrapper.compiler_record(kwargs) == case["constructor_kwargs"],
                "Changed frozen constructor",
            )
            set_cache_root(folder / "codegen")
            solver = Solver(
                landscape.SYSTEMS[request["system"]]["build"](), **kwargs
            )
            if constants:
                solver.update(constants)
            prior_author.require(
                "contract" in solver.update(contract=False),
                "Unrecognized public contract update",
            )
            solver.kernel.single_integrator.device_function
            actual_flags = [
                dict(
                    path=name,
                    kwargs=wrapper.compiler_record(dict(factory.jit_kwargs)),
                )
                for name, factory in prior_author.factories(solver.kernel)
            ]
            prior_author.require(
                actual_flags == prior["factory_flags"],
                "Actual factory flags differ from retained intervention",
            )
            with np.load(
                wrapper.checked(case["grid"]), allow_pickle=False
            ) as grid:
                inits = grid["initial_values"].copy()
                params = grid["parameters"].copy()
            prior_author.require(
                inits.dtype == params.dtype == np.float32,
                "Expected frozen FP32 inputs",
            )
            protocol = prepared["protocol"]
            solver.compile(
                inits,
                params,
                duration=protocol["duration"],
                t0=protocol["t0"],
                grid_type="verbatim",
            )
            ((signature, specialization),) = (
                solver.kernel.kernel.overloads.items()
            )
            library = specialization._codelibrary
            cubin = bytes(library._cubin)
            expected = wrapper.checked(prior["native"]["cubin"]).read_bytes()
            (folder / "kernel.cubin").write_bytes(cubin)
            naming = prepared["cubin_equivalence"]["naming_source"]
            wrapper.checked(naming)
            comparison = wrapper.compare_cubins(
                expected, cubin, naming["sha256"]
            )
            prior_author.require(
                comparison["admitted"],
                "Source-contract-disabled native identity failed",
            )
            ltoir = specialization.metadata.get("ltoir")
            prior_author.require(bool(ltoir), "Missing original cached LTOIR")
            (folder / "cached_ltoir.bin").write_bytes(bytes(ltoir))
            receipt["records"].append(
                dict(
                    source=source,
                    factory_flags=actual_flags,
                    signature=str(signature),
                    original_native=prior["native"]["cubin"],
                    cubin=wrapper.asset(folder / "kernel.cubin"),
                    cached_ltoir=wrapper.asset(folder / "cached_ltoir.bin"),
                    original_cubin_comparison=comparison,
                )
            )
            solver.close()
            solver = None
        receipt["status"] = (
            "CONTRACT_FALSE_ORIGINAL_IR_CAPTURE_AUTHOR_COMPLETE"
        )
    except Exception:
        receipt["status"] = "FAILED_RETAINED"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        wrapper.write(ROOT / "receipt.json", receipt)
        if solver is not None:
            solver.close()
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    run()
