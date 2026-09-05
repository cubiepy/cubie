# Frozen native policy profile wrapper

`native_policy_profile.py` reproduces the six counter-free Lorenz images
from `native_holdout_pilot10_e1/native_e2`: RK23, Kvaerno3/LU, and
Radau3/BiCGSTAB, each with full unrolling or stage count one and local
placement. It does not consume or update model predictions.

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
CuBIE, the production placement landscape and its
separately hash-bound ELF comparator; it does not import the changing
hardware-model helpers. The fixed eight-group mapping is
checked against each saved graph's full group, level and flag identity.

Native execution uses `--prepared <prepared.json> --case <case-id> --out
<output-root>`. It creates a unique UTC/PID/UUID directory for every
application-replay process. Public `Solver.compile`, `Solver.solve`,
cached kernel properties and dispatcher resource queries reproduce the
original runner. The installed compiler's code-library interface is
used solely to retrieve the cubin and apply the original carveout hint.
The backend/version, device capacities, compiler flags, cubin identity
contract,
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
prediction inputs. The original carveout setter is preserved for
reproduction. Independent handle observations establish that this setter
configures a compatibility module, separately from the installed native
dispatcher's launch module. Its getter is not the actual launch function's
preference. Achieved cache/shared partition remains a profiler observation.

Independent native preflight receipts
`verification/native_profile_preflight_independent_e1/receipt.json` and
`verification/native_profile_preflight_independent_e2/receipt.json`
admit all six candidates under their respective binary contracts and exact
own-array checks. The naming amendment's independent adversarial checks
are in `verification/cubin_equivalence_independent_e2/receipt.json`.
The independent handle proof is in
`verification/carveout_handle_native_independent_e1/receipt.json`.

The public constructor receives an actual `UnrollFlags` object with
exact `(True, None)` or `(True, 1)` tuples, matching the original runner.
Preparation serializes this object's complete typed fields; a dictionary
under the nested `unroll` argument is not equivalent. CPU author checks
construct all six actual Solvers and bind their cached child factories
while asserting that no native specialization has occurred.


The revised cubin contract preserves raw cubin byte equality as an
explicit field. An unequal image is admitted only under the bound
installed compiler naming source: process-global decimal numbering of
local `constant_array_N` objects. The ELF comparator requires unchanged
symbol indices, local/object type, visibility, section, value and size;
only their complete string entries and the resulting string offsets may
change. Constant data, instruction code, relocation, debug, nv.info and
all other section payloads remain byte-identical. The complete ordered
section inventory and every other header field remain identical.

The comparator separately proves file placement: ordered regions retain
their identities, each next offset is determined by the exact declared
alignment, all intervening padding is zero, and no unclassified file
bytes remain. Program type/flags/addresses/alignment remain exact;
start/end boundaries must denote the same ELF regions. The admitted
file-backed segments have equal file and memory extents. Unsupported
NOBITS layouts, extra padding, new strings/symbols or other differences
are rejected. There is no SASS or register normalization.

Both original and fresh cubin hashes and any raw-byte inequality remain
in each run receipt. A successful unequal-byte image is labeled
`SECTION_BOUND_NATIVE_REPRODUCTION_PASS`, followed by the same strict
own-candidate whole-array checks. Original failed preflight artifacts
remain unchanged. Preparation binds the exact comparator source and the
installed naming-source file SHA256. Mirroring this version for the
persistent profiler requires mirroring that exact comparator beside the
new wrapper filename; the earlier wrapper used by active profile jobs
is not replaced.

The generated-name mapping includes unchanged generated symbols. Each
unequal-byte image must have unique generated constant-array names; a
rename cannot collide with an unchanged symbol. This admits the observed
monotone-counter output. Unequal images containing generated-name aliases
need a separate alias proof and are not admitted by this contract.
