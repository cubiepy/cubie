# Counted arithmetic preparation and admission

`arithmetic_service_probe.py` implements the approved
`ARITHMETIC_SERVICE_DESIGN.md`. Its separate
`arithmetic_service_worker.py` is invoked only by an explicit native mode.
Default preparation imports NumPy and standard-library modules; it does
not import CuBIE, CUDA, Numba or the installed compiler.

The first native artifact for each operation requires an independent
source/SASS/ABI review before any launch. A rejected native shape retains
the compiled PTX, cubin, disassembly and error. In particular, an absent
FMUL or MOV chain is a lowering/capability result, not zero service cost.
The implemented admission deliberately accepts specific proved shapes;
an unobserved compiler transformation is not granted a broad allowlist.

## Invocation

Run from the research worktree with its actual runtime environment. The
hardware manifest is a retained device query, not a manually filled
capacity table. This command prepares one case without compilation:

```powershell
& C:\local_working_projects\cubie\.venv\Scripts\python.exe `
  benchmarks\hardware_model\arithmetic_service_probe.py `
  --out C:\local_working_projects\cubie-notes\hardware_unroll_placement\arithmetic_ffma_prepare `
  --hardware-manifest C:\local_working_projects\cubie-notes\hardware_unroll_placement\icache_contrast_20260904\manifest.json `
  --operation ffma --body-operations 257
```

Use a fresh output directory on every invocation. Add `--compile-only`
for the independent native gate; add `--execute` for ordinary collection
after that gate passes. Supported operations are `fadd`, `fmul`, `ffma`,
`iadd3`, `imad`, `rcp` and `mov`; body sizes are 33 and 257. The default
requested repeat count is 32,769, then duration calibration selects a
common bounded odd N for both populations. `--waves` is at least two.

A matched profile uses the same operation, body and hardware manifest,
plus `--ordinary-dir <accepted-bank> --profile-multiplier 1|2
--profile-warps 1|32`. This mode validates every ordinary array before
executing exactly one intended `probe` kernel. It performs no calibration
or reciprocal-trace launch. Nsight filtering must select only `probe`;
allocation/fill operations are not target kernels. Profile elapsed time
is excluded from the ordinary measurements.

## Recorded contract

Preparation retains exact generator, worker, frozen proof-helper and
native-parser bytes, the generated Python and inline PTX, canonical
uint32 input bits, coefficients and hardware manifest. Readback verifies
the construction, not only the stored digests. The scalar kernel ABI is
four uint64 addresses followed by five uint32 values: output, seed,
expected, trace, repeat count, active-warp limit, trace selector,
multiplier bits and addend bits. Parameter offsets are 0, 8, 16, 24,
32, 36, 40, 44 and 48; trailing alignment padding is recorded separately.

Native admission requires the exact target count, per-GPR predecessor
chain and loop-carried tie, unchanged runtime coefficients, a dominating
count definition, controlled decrement/test/backedge, initial operand
completion guard and final result guard. It protects both words of each
starting timestamp across the body. The full-warp population branch is
bound to `threadIdx.x >> 5` and the runtime warp limit. Every reachable
exit must pass the final full-CTA barrier. Native scheduling/control
instructions remain in the complete opcode inventory. The final
disassembly review also checks actual address construction, output
stores and clock-difference arithmetic; the logical admission fixtures
are not a substitute for that first compiled-artifact review.

For reciprocal, 64 uncounted functional MUFU.RCP operations each have an
observable uint32 store. The native witness binds all 65 stored states,
direct result dependencies, contiguous byte offsets, trace-mode branch,
and the same reaching seed load as the counted path. Every actual
normal finite transition is checked against the PTX one-ulp bound using
integer-rational arithmetic. All CTAs and duplicate lane seeds must
agree. A nontrivial closed bit cycle is required for each seed, and N/2N
outputs must differ. These functional samples have no timing admission.

Ordinary calibration measures both one-warp and 32-warp populations at
the same N. Two six-sample blocks mirror population and N/2N order,
giving 48 measurement launches. Every measurement must have both event
time and its minimum per-lane chain interval at least 20 ms. Clock
conversion rederives the interval from qualified MHz snapshots and raw
cycles. These snapshots are not a continuous upper-bound clock trace.

Each compressed NPZ retains every output word and the trace, seed and
expected arrays. Timed traces and inactive output lanes must remain at
their sentinels. Active lanes must have exact output bits, unsigned
clock differences, matching entry/exit SMID, exact operation count and
success status. Every CTA has one common SMID. Raw JSONL, embedded rows,
ordering, NPZ hashes, actual values and native identities are joined by
the ordinary loader. It reparses retained SASS and rederives native
admission without CUDA imports. Historical readers are loaded from
their own retained source snapshots rather than silently using a newer
source epoch.

The same compiled specialization and CUfunc handle are checked before
and after every launch. Runtime compiler/source/component identity,
tool hashes, registers, absence of local/static shared frames, maximum
block size and queried one-block residency are retained. Native modes
run in one isolated child; bounded termination and process-exit receipts
record cleanup. No device attributes or driver policies are modified.

## Counter and service interpretation

The saved profile review must separately join the exact profiler target
snapshot, command/environment, report hash, ordinary source/runtime,
binary, native instruction PCs and output arrays. LaunchStats must show
1,024-thread blocks, the declared grid, one possible resident block and
at least two waves. SourceCounters must show 32 participating lanes for
each active target warp and the exact population/N/body target count.
All initialization, guard, branch, convergence, barrier and output work
is counted separately. Hardware/source counter residuals are preserved.

Raw per-lane chain distributions and paired N/2N differences are service
observations under their recorded population. Dense CTA envelopes are
`max(end)-min(begin)` over active lanes on the same SM; summing these
envelopes gives a per-SM-cycle denominator because the thread capacity
and final barrier exclude overlapping timed CTAs. Their ratios include
administration and scheduling. They do not prove a specific pipeline
map or peak throughput. Event work/second is a separate whole-device
quantity and includes both waves.

The instrument emits no fitted solver penalty, store-completion cost,
family weight or liveness multiplier. A catalog entry must state whether
it is a measured conditional interval, a published initiation capacity,
or an explicitly other-architecture proxy.

## Observed FFMA257 native form

The first compile-only artifact is retained under
`arithmetic_ffma257_compile_e1`, with its original failed admission and no
launch. It contains 336 native instructions, including a 4,192-byte hot
region of 257 FFMA instructions and five administration instructions:
one runtime multiplier MOV, counter decrement, predicate comparison,
terminal CALL and backedge. The multiplier reload and per-FFMA constant
addend remain part of the measured workload.

`observed_ffma257` admits only that complete instruction/operand/address
template and its exact nine-scalar ABI. The generic admission for other
operation/body cases is unchanged. This is a complete-program template
proof, not a cubin-hash allowlist: every native instruction, explicit
predicate, branch target, register word, address equation and output
store is checked. Its receipt lists all 257 recurrence edges and the
separate administration, initial/final guards and output witnesses.

The population predicate survives address arithmetic between its
comparison and branch. Initial XORs consume seed, expected result, both
runtime coefficients and N before the start clock. A seed copy protects
the value when the initial clock overwrites the load register. The
uniform counter decrements independently from the original GPR N used
to emit the unsigned 64-bit operation count. Starting clock words remain
live until their stores and the full 64-bit subtraction. Each result,
SMID, timestamp, success marker and operation-count output has its
recorded zero extension or paired-register producer.

The ending comparison uses the final FFMA result. Successful lanes
cancel B1 participation, then pass the conditional branch to the end
clock and eight output stores. Invalid lanes join B1 and write failure
status. Both paths and inactive warps converge through B0, full-mask
WARPSYNC and the final CTA barrier. The terminal CALL has no returning
path. The self-branch and NOP footer follow the unconditional EXIT and
are unreachable.

NVIDIA describes BREAK as leaving a specified convergence barrier in
its Ada instruction table. The primary control-flow study explains
predicate-selected removal from the barrier participation mask.
[Binary Utilities 13.3, Table 7](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#nvidia-ampere-gpu-and-ada-instruction-set),
[Control Flow Management in Modern GPUs, section V-E](https://arxiv.org/html/2407.02944v1#S5.SS5).
The study concerns Turing; the separately retained official 13.3
`nvdisasm -bbcfg -poff` output for this actual SM89 cubin confirms the
relevant branch edges. Its receipt is
`verification/arithmetic_ffma257_native_e1/cfg_command.json` and graph
`cfg.dot`. No barrier timing or broader scheduling policy is inferred.

A fresh compilation under the revised source must reproduce the same
native artifact and pass independent review before ordinary collection.
Source generation is unchanged. PTX inventory rejection occurs after
SASS and ELF retention so an eliminated MOV/FMUL case still leaves
diagnostic native evidence.
