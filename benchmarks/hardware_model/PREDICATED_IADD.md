# Predicated integer instruction stream

Two inspected SM89 images execute 33 or 257 self-dependent predicated
IADD3 instructions per loop. PTX requests for conditional MOV/SEL updates
are optimized into those native arithmetic instructions. They are not
MOV latency probes. Other prepared forms and the second preparation
epoch remain compiler-form observations and are not timed in this bank.

The ordinary measurement contains 252 complete endpoint/clock arrays:
two images, three active-warp populations (1, 8, 32), three data-predicate
populations (0, 16, 32 lanes), seven N/2N pairs. The first pair in each
cell is retained and excluded from the 108 scored paired intervals.
N is 65,539. Each launch has 112 blocks of 1,024 threads: two complete
occupancy waves on 56 SMs at the observed one block per SM. Both images
use sixteen registers and zero local memory. Inactive warps retain their
endpoint sentinels and zero clocks; every active lane has an exact
integer endpoint and every active warp has a positive clock sample.

Reported intervals are exact differences of median warp clocks divided
by the additional target count. Loop administration remains included.
The 33-instruction body uses a conditional backward BRA. The 257 body
has an exit CALL.REL.NOINC and unconditional backward BRA, with a final
NOP also inside the timed interval. The two body lengths cannot be
subtracted as if their non-target streams were identical.

Data-predicate populations give similar observed intervals, but this does
not isolate intrinsic false-predicate dependency latency. Fresh native
encoding reads show fixed scheduling fields: 30/254 target instructions
have stall value four, two have value two, and one has value one. Target
read/write barrier fields are seven and the wait mask is zero. The
encoding map is the unofficial CuAssembler implementation, pinned at
96a9f72baf00f40b9b299653fcef8d3e2b4a3d49; it explicitly supports SM89.
Every decoded sixteen-byte instruction matches the actual ELF text.

The result requires preserving a common compiled schedule across runtime
mask scenarios. A false data predicate does not authorize deleting its
compiled issue gap. It does not establish a standalone IADD or MOV
latency, an all-false resource reservation rule, or a fitted mask factor.
No measured quantity from this bank changes a prediction parameter.

The exact sources, eight first-epoch prepared images, additional second-
epoch forms, full measurements and control encodings remain under
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`:

- `verification/predicated_move_author_e1/prepare.py` and `measure.py`;
- `predicated_move_native_layout_e1` and `predicated_move_native_layout_e2`;
- `predicated_iadd_measurement_e1/receipt.json`;
- `verification/predicated_iadd_independent_e1/receipt.json`;
- `verification/predicated_iadd_measurement_independent_e1/receipt.json`;
- `verification/predicated_iadd_control_e1/receipt.json`;
- `verification/predicated_iadd_control_independent_e1/receipt.json`.

The mapping is retained as primary implementation provenance, not executed
as downloaded code: [SM version/control placement](https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuSMVersion.py)
and [control field decoding](https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuControlCode.py).

The repository preparation and measurement modules mirror the measured
source bytes. Runtime admission binds a source path as well as its bytes;
a relocated execution requires its own independent admission receipt.
