The collector enables only CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL. It uses
the installed CUDA 13.3 CUpti_ActivityKernel12 declaration directly in C++;
there is no hand-written ctypes activity layout. The compiled header API
version is 130301 and CUDA header version is 13030. Runtime cuptiGetVersion
must match the compiled header. The file records sizeof and compiler-derived
offsets for the three carveout fields, with all kernel fields read through
the named header members.

The actual-header descriptions and the primary C API documentation define
isSharedMemoryCarveoutRequested as indicating a preferred-carveout update,
sharedMemoryCarveoutRequested as its percentage when that flag is set, and
sharedMemoryExecuted as the size selected by the driver. CacheConfig is not
authoritative on this architecture when the carveout flag is set. See
https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel12.html and the
bound local cupti_activity.h lines 3606–3930. The source code copies these
fields directly. The collected timestamps are retained metadata only;
they do not become ordinary timings or prediction inputs.

The request callback allocates an eight-byte-aligned 1 MiB host transport
buffer with _aligned_malloc. This is a transport allocation, not a model
cache size or fitted parameter. CUPTI ownership continues until its
completion callback relinquishes the buffer. A mutex protects outstanding
buffer ownership, JSON output and totals. An atomic callback count covers
the entire callback, including final free. Every parsed record and every
CUPTI return code is recorded. End-of-buffer MAX_LIMIT_REACHED is explicitly
normal; other errors and any dropped record cause failure. Exceptions do
not cross the callback ABI.

The E5 harness synchronizes each phase and closes its Solver before stop.
Stop disables the activity kind, issues the documented blocking forced
flush, queries remaining drops and finalizes CUPTI. The documented
cuptiFinalize preconditions are prior CUDA synchronization and activity
flush (local header 12292–12326). No lock is held across these CUPTI calls.
The collector verifies zero active callbacks and outstanding buffers before
closing its file. Failed quiescence leaves the file alive for late callbacks
and refuses success. A fresh process is required for each preference.

collect_metadata.py uses scoped os.add_dll_directory handles and explicit
absolute DLL paths. It verifies the actual loaded CUPTI and collector module
paths via GetModuleFileNameW and hashes all bound build/header/provider
files. It does not set environment variables. It starts the collector,
imports the exact E5 bytes and calls the existing run function unchanged.
Stop is attempted in finally, including when the solver or start fails.
Both original E5 receipts and activity JSONL survive failure.

The wrapper requires exactly two records for the exact native entry name,
distinct correlation IDs, the original grid/block geometry, and matching
register/static/dynamic resource quantities. It verifies the requested
carveout metadata against the actual E5 setter, then independently compares
both full state/status outputs to the retained reference. All E5 strict
native/IR/counter/occupancy checks still run. Other collected kernels remain
in the JSONL; they cannot silently become one of the two matching launches.

No NCU process or hardware-counter collector is involved. CUPTI activity
metadata is still an observation under instrumentation; comparison with the
NCU cohort can distinguish the two collection routes but does not establish
that instrumentation can never influence execution. No size is inferred
from the requested percentage.

Host compilation is the only author execution. build_receipt.json records
the explicit CL and LINK commands, all 139 included header hashes, selected
CUPTI import-library/DLL hashes, and object/DLL hashes. Compiler warnings
C4324 are preserved and originate from alignment declarations in the
unmodified CUPTI header. The collector is an x64 host DLL, with no device
code or native solver compilation. Independent CPU review precedes every
root-owned native capture.

Root invocation after the independent gate and GPU-idle confirmation:

    python collect_metadata.py --percent 0 --output <fresh raw root>

Use the same external frozen-production PYTHONPATH as E5. Repeat in separate
processes for 8, 16, 32, 64 and 100. The script has no author-side execution
mode that launches a kernel.

Decoder E2 reuses the exact reviewed E1 DLL and build manifest without
recompilation. The first retained activity run reports per-thread local
memory zero and total local reservation 33,030,144 bytes, whereas the same
actual E5 CUDA function reports 40 local bytes. Its exact cubin, SASS,
registers, launch geometry and both outputs match. The installed header and
runtime version agree, and both records occupy exactly 232 bytes. This is
a recorded cross-provider discrepancy; neither the documentation nor this
experiment establishes a zero-as-unavailable convention or a correction.

E2 preserves the original failed receipt unchanged. It retains both local
values for every record, their exact equality result and unresolved scope.
Every E5 actual-function local-memory check remains mandatory, including
40-byte identity in both phase attribute snapshots. The wrapper's complete
resource_consistency_passed flag is false on disagreement, and its status
is COMPLETE_WITH_LOCAL_METADATA_DISAGREEMENT, never a full metadata PASS.
The same comparison handles equal values without a special whitelist.
Any core ABI, kernel ownership, register/static/dynamic, requested setting,
drop/error, exact native image or output failure still refuses completion.
The shared-partition observation is retained under this explicitly qualified
activity route; the discrepancy is not repaired or used as a model input.
