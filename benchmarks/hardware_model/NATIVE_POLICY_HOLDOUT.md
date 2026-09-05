# Frozen native policy holdout

`native_policy_holdout.py` separates a CPU prediction freeze from an
explicit native validation run. The run never updates predictor inputs.
It compares whole-solver outcomes with the saved conditional attempted
algorithm-step rankings. Differences can expose a wrong conditional
service hypothesis, allocation/lowering error, or omitted caller and
iteration behavior; timing disagreement alone does not identify which.

## CPU freeze

Supply one exact `joint_policy_evaluation` request, its selected complete
per-workload result files, and one explicit baseline candidate identifier
for each result. Every candidate in each selected result enters the bank.
The baseline also has a separately constructed duplicate.

The runner checks the result's source-only markers, effective request
hash (including the evaluator's ERK/ROS/LU loop-regime adjustments),
conditional ranking status, graph hashes and semantic identities. It
compares production function files retained by the graph with the current
production tree. The manifest copies the request, results, graphs and
FP32 input grids, and binds exact current production/constructor source
bytes. Results retain their original ranking status, including a
`conditional_compute_data_default` whose fetch service was omitted.

The baseline constructor uses `placement_landscape.solver_kwargs`, the
request's solver overrides, public inner-solver name, exact eight unroll
levels and location settings. It applies request constants or the
landscape system constants before inspecting source. In particular,
Fabbri's landscape `ANS` constant is applied. Cached public device-function
construction creates source without requesting native specialization.
Workload identity, full owner-qualified placement registry, and shared
stride must match the frozen graph. Every native candidate repeats these
checks before compilation.

Example (PowerShell; explicit result and baseline lists have equal length):

```powershell
python -m benchmarks.hardware_model.native_policy_holdout freeze `
  --request C:/research/request.json `
  --results C:/research/kvaerno3_lu/result.json `
  --baselines source_0000_b128_s102400 `
  --out C:/research/holdout_prediction_freeze
```

`--protocol` accepts the complete JSON protocol record. The default uses
the landscape system's batch and duration, start time zero, two warmups,
four rounds of fifteen solves per block, and the mean of each block's
lowest five kernel times. These are the existing `ab_gate.py` measurement
design choices, not parameters fitted to solver performance. Batch or
protocol changes require another explicit prediction freeze.

## Native run

Run only after the source prediction freeze and runner review:

```powershell
python -m benchmarks.hardware_model.native_policy_holdout run `
  --frozen C:/research/holdout_prediction_freeze `
  --nvdisasm 'C:/explicit/CUDA/bin/nvdisasm.exe' `
  --out C:/research/holdout_native_bank
```

The runner verifies frozen bytes and current production-source identity,
requires the installed MLIR backend, and checks the declared Ada SM89
device capacities. It calls public `Solver.compile`, records compiler
kwargs and native registers/shared/local attributes, and saves the cubin,
disassembler command/output and SASS. These are validation observations.
The source forecast is not recomputed from them.

No launch method or block limiter is patched. The compiled function's
public `set_shared_memory_carveout` receives an integer percentage found
by enumerating 0 through 100 and applying the declared supported-size
roundup. A requested 8 KiB therefore uses an exactly representable hint;
an unrepresentable request is rejected. The hint remains a driver
preference: the achieved partition is explicitly unverified until a
separate hardware-counter check.

`placement_landscape.launch_geometry` computes occupancy with the
compiled function and actual dynamic allocation. Both compilation and
every completed solve check all current chunks. Requested block size,
one thread per trajectory, compiled static shared bytes, source shared
stride, the public minimum four dynamic shared bytes, and at least two
full occupancy waves per chunk are required. A later memory-manager
rechunk cannot silently turn an admitted measurement into a smaller
batch. The optional landscape NCU calculator's limiter field is retained
as a diagnostic; the compiled driver's occupancy supplies the wave check.

## Timing and numerical evidence

All candidates are compiled before timing a workload. Each forward round
is followed by its reverse; successive pairs rotate the first candidate.
Every short block is followed by a seeded uniform idle between 1.5 and
3.5 seconds, matching the AB gate's idle interval. This balances position
drift without claiming a many-candidate round is a simultaneous comparison.

Only `kernel_chunk*` CUDA events enter the timing statistic. Per-chunk
events, full-precision summed kernel milliseconds, wall milliseconds,
round/sample order and actual geometry are retained. Summary ratios pair
each candidate's block statistic with the baseline in the same round,
then take their median. Raw samples remain available independently of
the lowest-five statistic. The reported regret is the frozen default's
median paired ratio relative to the lowest eligible candidate ratio.

Each solve retains its final state and complete raw status words through
content-addressed NPZ files; repeated identical arrays reference the
same exact bytes. Counter output is disabled in the timing bank. Every
state must be nonempty FP32 and finite, and every entire status word must
equal `CUBIE_RESULT_CODES.SUCCESS` (zero). Production `ode_loop.py`
combines result flags, and `BatchSolverKernel.py` writes that status word;
iteration counts use separate arrays. The check does not mask away any
upper bits.

Candidates are compared with the explicit baseline's first warmup using
the requested solver's unchanged scalar `atol` and `rtol`, with matching
shapes/statuses and `equal_nan=False`. This is diagnostic cross-candidate
agreement under local solver tolerances, **not a global-accuracy proof**.
The independent baseline duplicate must instead match state/status bytes
exactly. A missing baseline cannot produce a successful comparison.

Compile, geometry, solve and numerical failures are retained; the bank
continues other candidates and returns `FAILED_OR_INCOMPLETE` with a
nonzero exit code. Ineligible or incomplete candidates cannot enter the
measured fastest list. A failed candidate's samples are not silently
deleted to improve its statistic. Source exceptions during a solve have
tracebacks; completed solves retain their arrays even when numerical or
measurement checks fail.

If the frozen protocol explicitly enables `diagnostic_counters`, a fresh
instrumented bank runs only after **all** ordinary workload timing banks.
It compiles separate state-plus-counter kernels and stores results under
`instrumented_diagnostics`, labelled as diagnostics and excluded from
prediction and ordinary timing ranking. Instrumentation can change
allocation, occupancy and runtime; these observations are not treated as
the counter-free kernel's execution trace.

## Verification scope

CPU author checks can verify exact source reconstruction, frozen hashes,
FP32 grids, supported carveout mappings, status/numerical behavior and
hand-calculated summary/order cases. They cannot verify native launch,
driver hint acceptance or numerical solver outcomes. Those checks occur
in the explicit GPU run after independent runner review. The runner does
not claim that CPU validation or a frozen prediction is a successful
native holdout.
