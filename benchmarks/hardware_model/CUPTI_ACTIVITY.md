# Actual-function activity metadata

Six independent processes collect only CUPTI concurrent-kernel activity
metadata while the exact controlled RK23 image executes twice. There is
no Nsight Compute process or hardware-counter collection in this cohort.
The installed CUDA 13.3 Kernel12 header defines the record layout; a
host C++ collector reads named fields directly and binds header, DLL,
library and compiler identities. No hand-written ctypes record layout
is used. All twelve launches reproduce their own complete FP32 state
and SUCCESS arrays exactly.

The reported executed shared allocation is 8,192 bytes at requested
preference zero and 102,400 bytes at every nonzero preference (8, 16,
32, 64, 100). This reproduces the six matched NCU observations under a
different collection route. It does not prove the uninstrumented timing
process uses those partitions, or produce a general driver-choice rule.

Every activity record also reports zero local bytes per thread while
the exact actual launch function reports forty. Total activity local
reservation is 33,030,144 bytes. This is an unresolved cross-provider
metadata disagreement. The installed documentation does not establish
zero as an unavailable sentinel, so the collector does not reinterpret it.
All six receipts have status COMPLETE_WITH_LOCAL_METADATA_DISAGREEMENT
and false complete-resource consistency. Native image, function/module,
argument, output and other matching activity fields retain their exact
gates. A full resource-metadata PASS is not claimed.

The first process stopped before solver compilation because the external
PYTHONPATH omitted the reviewed wrapper's sibling import directory. Its
collector shut down cleanly with zero kernel records. The second process
completed two exact outputs and stopped at the original local-metadata
equality check. Both failures remain retained. The second decoder epoch
records the unequal values explicitly under a distinct result status;
it does not change the DLL, solver, native identity gate or measured data.

Independent native audit checks all twelve endpoints, 156 successful
driver calls, actual CUlibrary/function/module ownership, the record ABI,
and zero dropped records, outstanding buffers or active callbacks at
shutdown. The end-of-buffer CUPTI result is separately recognized by its
documented enum; other errors remain failures. Timestamps are retained
metadata, excluded from ordinary timing and model parameters.

Evidence is under
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`:

- `verification/cupti_carveout_author_e1`: C++ source, host build,
  original Python wrapper and exact provider/header manifest;
- `verification/cupti_carveout_author_e2`: qualified decoder and protocol;
- `cupti_carveout_native_e1` and `cupti_carveout_native_e2`: failures;
- `cupti_carveout_native_e3`: six separately collected activity receipts;
- `verification/cupti_carveout_independent_e1` and `cupti_carveout_independent_e2`:
  independent admission of both collector/decoder epochs;
- `verification/cupti_carveout_native_independent_e3/receipt.json`:
  independent saved-result audit.

The field interpretation follows the bound installed header and
[NVIDIA's Kernel12 record documentation](https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel12.html).

`cupti_sources` mirrors the measured author sources byte for byte. Their
bound inputs, companion artifacts and runtime manifests remain in the
original verification directories. The mirror is source provenance;
relocating an admitted collector requires a fresh path/provider manifest
and independent admission, rather than reusing an old receipt.
