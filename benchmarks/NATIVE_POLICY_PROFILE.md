# Frozen native policy profile wrapper

`native_policy_profile.py` targets six counter-free Lorenz images
from `native_holdout_pilot10_e1/native_e2`: RK23, Kvaerno3/LU, and
Radau3/BiCGSTAB, each with full unrolling or stage count one and local
placement. It does not consume or update model predictions.

The three full-unroll cases pass independent native reproduction: exact
whole cubin, disassembly, resources, geometry and both saved output
arrays. The three stage-count-one cases pass construction, disassembly
and resource checks but fail whole-cubin identity before any solve.
Their process-global internal constant-array symbol names differ. The
independent ELF audit finds identical code, constant data and relocation
payloads, with symbol-table and file-layout differences. This wrapper
retains that strict failure; it does not normalize those binaries.

These results are bound in the external evidence receipts
`verification/native_profile_preflight_independent_e1/receipt.json` and
`verification/cubin_difference_three_independent_e1/receipt.json`.
The repaired
constructor passes a production `UnrollFlags` object and records its
normalized fields. All six actual constructors have independent CPU
admission under the frozen runtime. The earlier dictionary-constructor
failure remains preserved in the author and independent evidence.

Preparation is CPU-only. `--prepare --frozen <frozen-bank> --native
<native-bank> --disassembler <nvdisasm.exe> --out <new-directory>` binds
the original prediction manifest, native run identity, production and
placement-landscape sources, exact constructor recipe, input grids,
compiled cubin, stored disassembly, and reference arrays. All 62 original
samples for each candidate must have one identical array fingerprint and
the same eligible compiled geometry. The prepared manifest and source
hash are bound by `preparation_receipt.json`.

`--check --prepared <prepared.json> --out <receipt.json>` checks those
bindings under the selected runtime without compiling or launching. It
also verifies every own-candidate reference array and reproduces the
public constructor kwargs from the frozen request. The wrapper imports
only CuBIE and the production placement landscape; it does not import
the changing hardware-model helpers. The fixed eight-group mapping is
checked against each saved graph's full group, level and flag identity.

Native execution uses `--prepared <prepared.json> --case <case-id> --out
<output-root>`. It creates a unique UTC/PID/UUID directory for every
application-replay process. Public `Solver.compile`, `Solver.solve`,
cached kernel properties and dispatcher resource queries reproduce the
original runner. The installed compiler's code-library interface is
used solely to retrieve the cubin and apply the original carveout hint.
The backend/version, device capacities, compiler flags, exact cubin,
disassembly stdout, resource allocation and geometry must match before
the first solve. A mismatch writes a failure receipt and refuses the
solves; there is no fallback image or permissive comparison.

Disassembly comparison uses the same text-mode `nvdisasm -c` stdout as
the original runner's `disassembly_command.json`, and also compares it
with the original gzip read in text mode. Input gzip bytes retain their
own exact SHA256 binding. No instruction, address, symbol, whitespace or
register normalization is applied; gzip container timestamps are not
used as a freshly generated SASS identity.

There is one warmup solve followed by one capture solve, with the
original duration, initial time, grids, block size and state-only ABI.
The original bank's one-chunk geometry is required before and after both
solves, including at least two complete compiled occupancy waves. Every
entry of the saved final-state array and status array must match the
same original candidate in shape, dtype and exact C-order bytes after
each solve. Both new arrays are retained. Iteration counters must remain
disabled. The original bank saved the final state for all trajectories;
this is an exact check of that entire saved array, not an assertion
about unsaved intermediate trajectories.

The prepared capture contract is application replay,
`kernel_filter=regex:.*Lorenz_ltoon.*`, `launch_skip=1`, and
`launch_count=1`. These names occur in the actual six saved kernels.
Compilation does not launch a solver kernel, and the enforced single
chunk makes the skipped matching launch the warmup. The external
profiler owns filtering and collection; the wrapper never invokes NCU.
The production sources match the existing `epoch_ff3a567f` runtime, so
the reviewed wrapper can be mirrored to the persistent profiler's
research script directory without replacing that runtime.

All three stage-count-one candidates failed the original strict
cross-policy numerical comparison. Their preserved failure flags remain
in preparation and run receipts. Exact own-candidate reproduction makes
their native code suitable for diagnostic profiling; it does not make
those policies numerically eligible or qualify profiler timings as
prediction inputs. The carveout is the original driver preference;
achieved cache/shared partition remains a profiler observation.
