# Timing scope of the preserved implicit capture

The frozen capture constructs the original Solver with its original
verbose settings. Consequently, Solver internals print incidental event
and kernel-time values in the native log. The capture does not create a
deliberate timing bank, collect timing samples in its result schema or
use those printed values in a prediction or performance comparison.

The phrases "no timings" and "no timing measurements" in the frozen
protocol and source scope describe the intended diagnostic analysis;
they must not be read as asserting that the original Solver emitted no
timing information. The original source, plan, protocol and logs remain
unchanged to preserve the actual experiment. Incidental timings are
excluded from all contraction conclusions and hardware-model inputs.

This qualification applies to
`C:/local_working_projects/cubie-notes/hardware_unroll_placement/numerical_implicit_capture_native_e1/receipt.json`
and its original execution log. It does not authorize changing verbose
settings or deleting recorded output.
