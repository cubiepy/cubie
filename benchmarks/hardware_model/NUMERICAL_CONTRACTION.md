# Frozen RK23 contraction diagnostic

`numerical_contraction.py` tests one public compile-flag intervention on
the frozen RK23 full/stage-count-one pair. This is functional evidence,
excluded from prediction and timing. It does not replace the original
cross-policy numerical gate or establish global numerical accuracy.

The original constructor, FP32 grid, state-only output, duration of one,
262,144 trajectories and 128-thread blocks are retained. Original images
must pass the existing native identity contract before their outputs are
compared exactly with their own frozen state/status arrays. The second
pair uses `solver.update(contract=False)` before requesting the cached
device function or native specialization. Every discovered CUDAFactory
retains its complete JIT options except the removed `contract` flag.
Both intervention images, disassemblies and two independent solve
snapshots are retained. A hardware thread-capacity bound requires at
least two occupancy waves, with actual geometry checked after each solve.

The frozen constructor does not accept `jit_flags` as a Solver keyword.
The rejected first harness is preserved under
`verification/numerical_contraction_author_e1`; its CPU failure receipt
is in `verification/numerical_contraction_independent_e1`. The corrected
source hash is
`ffee481dd824e82828db2094065428aa77c1696b7808c5ea2696bae91c2c79f3`.
Its public update route passes independent CPU review in
`verification/numerical_contraction_independent_e2`.

The root-owned native run is `numerical_contraction_native_e2` beneath
the external hardware-unroll-placement evidence root. Independent audit
is `verification/numerical_contraction_native_independent_e1`.
All eight endpoint snapshots pass their functional and repeatability
checks. Removing the flag leaves each candidate's complete state/status
bytes equal to its corresponding original. The full/rolled comparison
still fails for 122,822 elements in 67,806 trajectories, with maximum
absolute difference 0.0000457763671875 at the unchanged 1e-6 absolute and
relative tolerances. The original bank remains failed.

This intervention does not establish elimination of native contraction.
FFMA instruction counts are 41/20 for original full/rolled images and
44/20 after the flag update. Installed `numba_cuda_mlir/fastmath.py:26-36`
emits module-level `fma=True` when contraction is requested, and omits
that option otherwise. Omission preserves downstream toolchain defaults.
Native code changes are retained; exact output equality does not prove
identical internal arithmetic. This result cannot rule out contraction
as a cause of the original cross-policy discrepancy.

The CLI requires explicit `--prepared`, `--wrapper` and fresh `--output`
paths. Use the exact e5 profile preparation and its source wrapper, with
the frozen `hardware-epoch-ff3a567f` runtime selected externally through
PYTHONPATH and the MLIR backend selected externally. The prepared loader
checks the runtime, wrapper, source, constructor and input assets before
execution. The shared environment and production defaults are untouched.
