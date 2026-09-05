# Controlled actual-function carveout diagnostic

This isolated runner reconstructs the reviewed rolled-stage/local RK23
case `workload_006_source_0000_b128_s102400`. It measures no timings. One
warmup and one capture launch test exact functional reproduction with a
requested preference on the actual launched function. Matching Nsight
captures must supply physical cache-partition observations separately.

Root owns all native compilation and GPU execution. The author pass is
CPU-only syntax, binding and source/API review; it cannot establish the
new diagnostic link's native image equality. A failed prelaunch image
gate is a retained diagnostic result, not grounds to broaden equivalence.

The runner imports the exact frozen profile wrapper named in the prepared
manifest. That wrapper validates source, installed compiler, constructor,
grid, original compile metadata and reviewed constant-symbol comparator.
It reconstructs the original public Solver and invokes public compile.
Its compatibility-module carveout setter remains an explicitly retained
part of original validation; it does not configure the diagnostic's
actual launch function.

The separate dispatcher receives the exact original `kernel.py_func`
object and effective source JIT kwargs, including any configured register
limit. The callback carrier is the exact retained diagnostic kernel LTOIR,
with a setup callback; it must equal the freshly generated kernel LTOIR. Explicit LTO remains enabled. Original
dispatcher identity, cubin bytes and target options are checked before
launch. No factory, dispatcher property, source file or installed backend
is replaced or patched.

`compile_kernel_specialization` calls the installed public `compile_for`
with exactly `_kernel_launch_args(run_params[0])`. Signature equality is
required. The diagnostic cubin must satisfy the existing section-bound
constant-symbol renumbering contract against the freshly reproduced
original. Exact `nvdisasm -c` text and all public resource attributes must
match. Any other linked-input effect fails before the first launch.

The public setup callback receives the actual CUlibrary loaded by the
native dispatcher. It resolves the exact compiled kernel name through
cuLibraryGetKernel/cuKernelGetFunction, verifies its module against
cuLibraryGetModule, and checks native registers/local/shared/maxthreads.
It sets and queries the requested function attribute before dispatch.
The actual-function occupancy query and a separate hardware thread-limit
ceiling both require at least two full waves. The frozen grid has 262144
runs, block shape (1,128), and four dynamic shared bytes. No requested
percentage is interpreted as an observed physical partition.

Only 0, 8, 16, 32, 64 and 100 are admitted requested percentages. Each
invocation creates a unique output directory and process-owned dispatcher;
use a fresh process per preference. There is one warmup then one capture,
allowing Nsight launch-skip one / launch-count one. The callback must run
once. NCU application replays receive distinct timestamp/PID/UUID output
directories and retain their own complete artifacts.

The direct launch reproduces the source array lifecycle while holding the
original kernel's memory-manager ownership: input/output initialise,
exact original ABI and geometry, input/output finalise, stream completion
and pending host-writeback drain. The kernel's public host state and
status buffers are copied without detaching them or creating a solver
result. `SolveResult.from_solver` uses those same output buffers; its
state property returns the stored array. The saved final state slice and
all statuses must match the own-candidate reference in dtype, shape and
every byte. Counter output must remain disabled through explicit source flags and
output selection; the inactive ABI buffer need not be None. The original bank's
failed cross-candidate agreement status is preserved; own-image replay
does not change it or establish global numerical accuracy.

Source/API backing:

- Frozen `BatchSolverKernel.py:684`–`:729`: allocation ownership, explicit
  compile and exact launch arguments.
- Frozen `BatchSolverKernel.py:861`–`:878`: initialise/launch/finalise.
- Frozen `BatchSolverKernel.py:989`–`:991`: effective register limit.
- Frozen `BatchSolverKernel.py:1632`–`:1665`: host buffers/counters.
- Frozen `solveresult.py:420` and `:493`: stored state-buffer ownership.
- Frozen `BatchOutputArrays.py:535`: pending writeback drain.
- Installed `cuda_simsafe.py` counterpart in frozen source, lines168–171:
  MLIR compile_for specialization helper.
- Installed `descriptor.py:2847`–`:2868`: callback receives actual
  CUlibrary after load.
- Installed `linker.py:73`–`:78`: explicit LTO selection is preserved with
  an added CUDA source input; native equivalence is still checked independently.
- Reviewed handle diagnostic native-independent receipt proves actual
  and compatibility function/module identities differ and their
  preference attributes are independently controlled.

Example root invocation requires PYTHONPATH containing the frozen
production `src`, frozen production root and the reviewed wrapper's
source directory, in that order. Use the original prepared e5 manifest
and its exact `native_policy_profile.py` wrapper. This source is frozen
for independent CPU review before any such invocation.

E2 retains the complete failed E1 images/log. E1 gained a separate
__cuda_sm70_votesync_all helper and changed main code; its unused noop
function did not remain as a text section. E2 uses only the identical PTX
version/target/address-size header, with no function or data declaration.
This is a proposed empty callback carrier, not a claim that E1 differences
were caused solely by function visibility. The strict binary gate remains
unchanged and must reject any remaining mixed-link native-code difference.
Backend FastMathOptions is recorded by explicit type and sorted flags;
unsupported option types fail before assignment into the receipt. Original
and diagnostic cached LTOIR and typed link-plan records are retained before
the binary comparison, so a failed gate remains inspectable.
RK23 has a 40-byte addressable local frame with zero allocator spill stores
and loads in the reproduced PTXAS report; local traffic is not called an
allocator spill here.

E3 uses a comment-only empty CUDA C++ translation unit in public CUSource,
replacing the PTX carrier. The installed lowerer passes the explicit LTO
plan to linker.add_file_guess_ext; that source marks the CUSource pending
input with lto=True. Driver._materialize_pending_cu invokes nvrtc.compile
with ltoir=True, adds the resulting LTOIR object, then links. The runner
requires the exact single supplied CUSource object in the emitted external
input list and compile_new_inputs_as_ltoir=True. Provider file hashes bind
this materialization route. This is a changed carrier hypothesis; neither
empty-source native equality nor physical partition is presumed. All
callback, launch, source, binary and own-array gates remain unchanged.

E2's header-only PTX produced exactly the same main-kernel and votesync
helper text bytes as E1. It altered debug/symbol metadata but did not
restore the original image. Both original images were byte-identical.
E2 records true LTO for both original and diagnostic source modules, with
only the diagnostic registering an external callback input. Their cached
LTOIR differ (13684/13688 bytes); E1 did not retain LTOIR, so no E1/E2
LTOIR identity claim is made. The surviving mixed-input result is a
measured compilation outcome, not attribution to noop function visibility.

E4 binds the exact saved diagnostic kernel LTOIR bytes as public LTOIR
callback input. Separate CPU-only links prove both original and diagnostic
kernel LTOIR alone reproduce the original native image under the unchanged
strict comparator. The extra empty module in E3 changes code; the kernel's
new LTOIR by itself does not.

Installed mlir_optimization.py817 passes freshly generated kernel LTOIR
first into base_linker.recreate_with_lto. Installed linker.py159-225 adds
that input then existing input objects through add_ltoir, which suppresses
an already-seen byte hash. E4 requires exact byte equality of carrier and
fresh kernel IR, one matching base LTO object and no pending CUDA inputs.
It reconstructs that exact installed final-input preparation, without
calling complete or compiling again, and requires one matching LTO object.
The recorded singleton is a source-bound reconstruction of the expired
final-linker local variable, not an invented observation of that variable.
The existing strict cubin, SASS and resources gate still decides admission.
No backend, module, dispatcher or compiler metadata is assigned or patched.

The retained kernel IR is validation-only provenance, outside predictor
inputs. If fresh compiler identity or deterministic internal naming changes
the kernel IR, compilation or the exact IR gate fails before launch.
The runner does not replace IR bytes or try alternative snapshots to admit
a mismatch. Every earlier carrier failure remains preserved.

E5 repairs the raw-buffer counter predicate. Frozen BatchSolverKernel
iteration_counters1662-1665 returns the host backing buffer. In contrast,
SolveResult.iteration_counters521-525 returns None when the active-output
flag is false. BatchOutputSizes.from_solver381-388 gives disabled logical
dimensions(0,0,0); its nonzero64-82 allocates a minimum placeholder.

The runner requires the frozen protocol's diagnostic_counters=False,
compile_flags.save_counters=False, active_outputs.iteration_counters=False
and integrator.save_counters=False, plus exact logical dimensions(0,0,0).
It records actual host/device placeholder shapes, dtypes and strides, and
requires the int32 rank-three device ABI. These checks run before the
separate dispatcher and after every launch. Activated counter collection
is rejected; a small backing buffer is never treated as evidence that
collection is disabled. Native image identity and complete state/status
byte checks remain mandatory. E4's strict image and first functional result
passed; its failed raw-buffer None predicate remains preserved as failed
checker provenance, not rewritten into a completed two-launch receipt.
