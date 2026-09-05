"""Relink four saved implicit source IRs under an explicit FMA pair."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import traceback

from numba_cuda_mlir.linker import Linker

from capture import asset, checked, read, require


def run(plan_path, capture_path, output):
    stage = read(Path(__file__).parent / "stage_author_receipt.json")
    for source in stage["sources"]:
        checked(source)
    require(
        stage["capture_plan"] == asset(plan_path), "Stage source plan differs"
    )
    plan = read(plan_path)
    for binding in plan["bindings"]:
        checked(binding)
    captured = read(capture_path)
    require(
        captured["plan"] == asset(plan_path), "Source capture plan differs"
    )
    require(
        captured["status"]
        == "IMPLICIT_CONTRACT_FALSE_OWN_REFERENCES_AND_IR_COMPLETE",
        "Source references are incomplete",
    )
    require(
        [x["case"] for x in captured["records"]] == plan["cases"],
        "Capture case coverage differs",
    )
    require(
        captured["device"]["compute_capability"] == [8, 9],
        "Different target architecture",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="STARTED",
        source=asset(__file__),
        plan=asset(plan_path),
        capture=asset(capture_path),
        records=[],
        scope="Singleton saved-LTOIR link, no CUDA context or GPU launch.",
    )
    try:
        for case in captured["records"]:
            original = checked(case["cubin"])
            ir_path = checked(case["cached_ltoir"])
            for fma in (True, False):
                folder = output / (case["case"] + "_fma_" + str(fma))
                folder.mkdir()
                options = dict(
                    cc=(8, 9),
                    arch="sm_89",
                    lto=True,
                    optimization_level=3,
                    ftz=True,
                    prec_div=False,
                    prec_sqrt=False,
                    fma=fma,
                    debug=False,
                    lineinfo=False,
                    max_registers=None,
                    ptxas_options=None,
                )
                linker = Linker(**options)
                linker.add_ltoir(ir_path.read_bytes())
                require(
                    len(linker._object_codes) == 1, "Extra final-link input"
                )
                rendered = linker._get_linker_options(
                    False
                )._prepare_nvjitlink_options()
                require(
                    "-fma=" + ("1" if fma else "0") in rendered,
                    "Wrong final FMA option",
                )
                cubin = folder / "kernel.cubin"
                cubin.write_bytes(bytes(linker.complete().code))
                executable = Path(
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvdisasm.exe"
                )
                command = [str(executable), "-c", str(cubin)]
                process = subprocess.run(
                    command, capture_output=True, check=True
                )
                sass = folder / "kernel.sass"
                sass.write_bytes(process.stdout)
                ffma = len(
                    re.findall(
                        rb"/\*[0-9a-f]+\*/\s*(?:@!?P\d+\s+)?FFMA\b",
                        process.stdout,
                    )
                )
                record = dict(
                    source=case["source"],
                    case=case["case"],
                    workload=case["workload"],
                    fma=fma,
                    ir=asset(ir_path),
                    original=asset(original),
                    cubin=asset(cubin),
                    options=options,
                    rendered_options=rendered,
                    native_ffma_sites=ffma,
                    exact_original_bytes=cubin.read_bytes()
                    == original.read_bytes(),
                    info_log=linker.info_log,
                    error_log=linker.error_log,
                    disassembler=asset(executable),
                    command=command,
                    sass=asset(sass),
                )
                receipt["records"].append(record)
                require(
                    record["exact_original_bytes"] if fma else ffma == 0,
                    "Baseline identity or actual zero-FFMA gate failed",
                )
        receipt["status"] = "IMPLICIT_SINGLETON_LINK_PAIR_COMPLETE"
    except Exception:
        receipt["status"] = "FAILED_RETAINED"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(
        json.dumps(
            dict(status=receipt["status"], records=len(receipt["records"]))
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.plan, args.capture, args.output)
