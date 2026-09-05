"""Verify the FP32 zero-to-infinity path used by Fabbri defaults."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from cubie.cuda_backend import CUDA_BACKEND
from cubie.cuda_simsafe import (
    compile_kernel_specialization,
    cuda,
    cupy,
    float32,
    get_jit_kwargs,
)
from benchmarks.lorenz_mean_runtime import _compiled_cubin


def probe(values, output):
    """Expose the intermediate values of the actual zero-parameter form."""
    index = cuda.grid(1)
    if index < values.size:
        value = values[index]
        logarithm = math.log2(value)
        power = value ** float32(-1.6951)
        denominator = float32(18003.4179) * power + float32(1)
        fraction = float32(3.573159) / denominator
        result = fraction + float32(0.025641)
        output[index, 0] = logarithm
        output[index, 1] = power
        output[index, 2] = denominator
        output[index, 3] = fraction
        output[index, 4] = result


def sha(path):
    """Bind an artifact by exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """Run one functional native check without measuring performance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "probe_source.py").write_bytes(Path(__file__).read_bytes())
    if CUDA_BACKEND != "mlir":
        raise ValueError("Exceptional form calibration requires MLIR")
    flags = get_jit_kwargs()
    kernel = cuda.jit(**flags)(probe)
    blocks = 2 * cupy.cuda.Device().attributes["MultiProcessorCount"]
    values = cupy.zeros(blocks * 128, dtype=cupy.float32)
    output = cupy.empty((values.size, 5), dtype=cupy.float32)
    compile_kernel_specialization(kernel, (values, output))
    compiled, = kernel.overloads.values()
    cubin, entry = _compiled_cubin(compiled)
    path = args.out / "kernel.cubin"
    path.write_bytes(cubin)
    assembly = kernel.inspect_asm()
    if isinstance(assembly, dict):
        assembly = "\n".join(assembly.values())
    (args.out / "kernel.ptx").write_text(assembly)
    disassembler = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/"
                        "CUDA/v13.3/bin/nvdisasm.exe")
    result = subprocess.run([str(disassembler), "-c", str(path)],
                            capture_output=True, text=True, check=True)
    (args.out / "kernel.sass").write_text(result.stdout)
    kernel[blocks, 128](values, output)
    cuda.synchronize()
    actual = output.get()
    expected = np.array([
        0xff800000, 0x7f800000, 0x7f800000, 0,
        np.float32(0.025641).view(np.uint32).item(),
    ], dtype=np.uint32)
    np.testing.assert_array_equal(
        actual.view(np.uint32), np.tile(expected, (values.size, 1)),
    )
    np.savez(args.out / "arrays.npz", inputs=values.get(), outputs=actual,
             expected_bits=expected)
    receipt = dict(
        status="FUNCTIONAL_PASS_PENDING_INDEPENDENT_REVIEW",
        backend=CUDA_BACKEND, entry=entry,
        jit_kwargs={name: sorted(value) if isinstance(value, set) else value
                    for name, value in flags.items()},
        block_threads=128, grid_blocks=blocks, kernel_launches=1,
        timing_measurements=0, expected_output_bits=expected.tolist(),
        exact_rows=int(values.size),
        semantics="LG2(+0)=-infinity; negative power of +0=+infinity; "
        "finite divided by +infinity=+0; final output is finite.",
        qualification="Isolated default-flag native FP32 functional path; "
        "not a full solver numerical validation or performance service.",
        artifacts={path.name: sha(path) for path in args.out.iterdir()
                   if path.is_file()},
    )
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
