"""Compile isolated FP32 math forms through the installed CUDA backend."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

from cubie.cuda_backend import CUDA_BACKEND
from cubie.cuda_simsafe import (
    compile_kernel_specialization,
    cuda,
    cupy,
    get_jit_kwargs,
)
from benchmarks.lorenz_mean_runtime import _compiled_cubin


FORMS = {
    "identity": "value",
    "exp": "math.exp(value)",
    "log": "math.log(value)",
    "log2": "math.log2(value)",
    "sqrt": "math.sqrt(value)",
    "pow_runtime": "value ** exponent[index]",
    "pow_1_3": "value ** float32(1.3)",
    "pow_2": "value ** float32(2)",
    "pow_3": "value ** float32(3)",
    "pow_minus1": "value ** float32(-1)",
    "pow_minus2": "value ** float32(-2)",
}


def digest(path):
    """Hash exact retained bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_form(name, expression, folder, nvdisasm):
    """Retain source, PTX and SASS without launching a kernel."""
    folder.mkdir()
    source = "\n".join([
        "import math",
        "from cubie.cuda_simsafe import cuda, float32",
        "",
        "def probe(values, exponent, output):",
        "    index = cuda.grid(1)",
        "    if index < output.size:",
        "        value = values[index]",
        "        output[index] = " + expression,
        "",
    ])
    path = folder / "kernel.py"
    path.write_text(source, encoding="utf-8")
    module_name = "math_form_" + name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    kernel = cuda.jit(**get_jit_kwargs())(module.probe)
    arrays = tuple(cupy.empty(32, dtype=cupy.float32) for _ in range(3))
    compile_kernel_specialization(kernel, arrays)
    (compiled,) = kernel.overloads.values()
    cubin, entry = _compiled_cubin(compiled)
    cubin_path = folder / "kernel.cubin"
    cubin_path.write_bytes(cubin)
    assembly = kernel.inspect_asm()
    if isinstance(assembly, dict):
        assembly = "\n".join(assembly.values())
    (folder / "kernel.ptx").write_text(assembly, encoding="utf-8")
    command = [str(nvdisasm), "-c", str(cubin_path)]
    process = subprocess.run(command, capture_output=True, text=True,
                             check=True)
    (folder / "kernel.sass").write_text(process.stdout, encoding="utf-8")
    (folder / "disassembly.stderr").write_text(process.stderr,
                                               encoding="utf-8")
    instructions = []
    pattern = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;")
    for line in process.stdout.splitlines():
        match = pattern.match(line)
        if match:
            text = match[2]
            tokens = text.split()
            opcode = tokens[1] if tokens[0].startswith("@") else tokens[0]
            instructions.append(dict(pc=int(match[1], 16), text=text,
                                     opcode=opcode))
    result = dict(
        name=name, expression=expression, entry=entry,
        instructions=instructions,
        opcode_inventory=dict(Counter(x["opcode"] for x in instructions)),
        registers=next(iter(kernel.get_regs_per_thread().values())),
        local_bytes=next(iter(kernel.get_local_mem_per_thread().values())),
        artifacts={str(p.name): digest(p) for p in folder.iterdir()
                   if p.is_file()},
        native_compilations=1, kernel_launches=0,
        timing_measurements=0,
        qualification="Isolated operation lowering, including kernel ABI; "
        "not a solver footprint, register or timing prediction.",
    )
    (folder / "receipt.json").write_text(json.dumps(result, indent=2),
                                          encoding="utf-8")
    return result


def main():
    """Compile the explicitly listed forms with default CuBIE JIT flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvdisasm", type=Path, default=Path(
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/"
        "bin/nvdisasm.exe"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "probe_source.py").write_bytes(Path(__file__).read_bytes())
    flags = get_jit_kwargs()
    flags = {key: sorted(value) if isinstance(value, set) else value
             for key, value in flags.items()}
    rows = []
    for name, expression in FORMS.items():
        row = compile_form(name, expression, args.output / name,
                           args.nvdisasm)
        rows.append(row)
        print(name, json.dumps(row["opcode_inventory"]), flush=True)
    (args.output / "receipt.json").write_text(json.dumps(dict(
        status="NATIVE_FORMS_RETAINED_REQUIRES_INDEPENDENT_REVIEW",
        backend=CUDA_BACKEND, jit_kwargs=flags,
        generator_sha256=digest(__file__),
        nvdisasm_sha256=digest(args.nvdisasm),
        cases=rows, native_compilations=len(rows), kernel_launches=0,
        timing_measurements=0,
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
