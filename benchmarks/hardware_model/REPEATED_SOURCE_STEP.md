# Carried state for a subsequent attempted step

`repeated_source_step.schedule_preceded_step` executes exactly one
preceding source wave, then a caller-supplied common measured workload of
at least two complete resident waves. The same live data and instruction
cache objects, pending requests, resource-ready clocks and absolute time
continue across every boundary. It excludes only the predecessor's cost
from the reported measured suffix.

The ordinary executor's boundary waits for all compute reservations,
register results and source consumption, and store visibility/completion.
Thus earlier compute readiness cannot delay the next wave beyond its
absolute origin. Any outstanding cache requests and request-resource
clocks remain in the live cache objects. No cache is populated by assuming
that an oversized body fits, and no arbitrary warmup count or convergence
tolerance is used.

Each result retains the exact initial, predecessor-boundary and final
mutable cache state, including ordered residency, pending requests,
counters and clocks. Immutable cache specification fields are bound by the
scenario hash. Traffic counts reported for measurement exclude the
preceding wave. The module verifies its complete execution and every
measured wave against an uninterrupted ordinary three-or-more-wave
reference; the reference's total cache summaries must also agree.

This is a named carried-state sensitivity at a synchronized drained
compute boundary. It does not claim steady-state convergence or model
the actual solver's overlap between differently progressing warps. The
same declared source event stream repeats; actual FSAL, convergence and
caller transitions require their own source regimes. The preceding work
is one step per resident warp; candidate geometries still receive the
same number of measured warp attempts.
Data-cache scenarios use reused physical local slots. Trajectory-unique
backing describes different incoming work and is not this repeated-step
alternative.

No solver timings, measured iteration counters, fitted cache penalties,
native register labels, native compilations or GPU launches are inputs.
