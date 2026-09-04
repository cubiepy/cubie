# Ada SM89 dispatch-capacity component

For SM89, `W/4` is an architectural lower bound in aggregate SM cycles
when `W` counts logical native warp instructions scheduled by the SM.
This includes uniform and control instructions. It is not an opcode
latency, an achievable throughput for every instruction mix, or a
prediction of kernel duration. Applying it to static source work first
requires an explicit source-to-native translation and a symbolic warp
execution count; this research has not established a complete numeric
pre-compile `W` for adaptive solvers.

## Source chain and scope

The Ada whitepaper's Figure 5, printed page 11 (PDF page index 10), shows
four partitions, each with its own instruction cache, warp scheduler
and dispatch path labeled `32 thread/clk`. The diagram establishes the
partition arrangement; it should not alone be read as a throughput
table for all execution units. [NVIDIA Ada whitepaper, Figure 5](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf#page=11).

CUDA 13.0's CC8.x architecture section specifies four warp schedulers,
with each issuing one instruction for one ready assigned warp at an
issue time. Its hardware-multithreading discussion states the peak
case of four warp instructions per SM clock for CC8.x, while explicitly
qualifying the latency-hiding example by instruction throughput.
Together these support four scheduler issue slots per SM clock, not
four completed arbitrary instructions every clock. [CUDA 13.0,
CC8.x architecture](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#compute-capability-8-x),
[hardware multithreading](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#multiprocessor-level).

NVIDIA's pipeline explanation includes the uniform datapath among
subpartition execution pipelines. The warp scheduler dispatches to
those pipelines or the memory input/output unit, which routes work to
shared units including control/branch. Therefore an instruction's
scalar or control character does not provide an extra scheduler issue
slot. This architectural inference combines that routing description
with the Ada/CC8.x issue limit. [NVIDIA developer explanation, Greg,
29 November 2024](https://forums.developer.nvidia.com/t/mapping-of-pipelines-to-functional-units/315200/2).

Predicated-off instructions remain scheduled even though inactive
threads do not produce results or access operands. Thus dropping their
warp instruction counts would omit scheduler demand. [CUDA Best
Practices, branch predication](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#branch-predication).

Nsight's exact source `inst_executed` increments once per warp visiting
the instruction and ignores predicates. Its definition is distinct
from thread or predicated-on thread counts. A logical execution also
need not equal a single hardware issue attempt: Nsight documents IMC
misses that issue but do not dispatch, then reissue. Extra attempts
cannot improve the lower bound obtained from logical work. The source
metric is collected by software patching in separate replay passes;
it is not a direct trace of hardware scheduler slots. [Nsight source
metrics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#source-metrics),
[IMC and SM units](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#units),
[collection overhead](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#overhead).

This justification covers the logical native instructions represented
by the reviewed SM89 SASS/source profiles, including `U*` operations,
branches, calls, predicates and memory instructions. It does not
reinterpret memory wavefronts, sectors, tensor operations, retired
threads, sample counts, or asynchronous engine work as warp issues.
An unsupported architecture or a differently defined metric requires
its own counting contract. Generic profiler prose describing schedulers
that can issue "one or more" instructions is not a reason to transplant
a dual-issue limit into this architecture-specific component.

## Bound and units

For each SM `s`, let `W_s` be its logical native warp instructions and
`C_s` its elapsed cycles in the SM clock domain. Even with perfectly
available instructions and execution units,

```text
W_s <= 4 * C_s
W = sum_s W_s
C_total = sum_s C_s >= W / 4
```

The denominator is four schedulers, not 128 FP32 lanes or a fitted
conversion factor. `W` is already warp-level and is not divided by 32.
The right-hand side is an exact rational number of aggregate SM cycles;
an integer cycle lower bound can take its ceiling. Uneven per-SM or
per-scheduler work, dependencies, pipe capacities, instruction fetch,
memory waiting, replay and launch tails can only increase the required
cycles. Their bounds combine by a maximum when they constrain the same
execution, not by automatically adding this term to every other cost.

Other resource bounds need their own units and scope. If per-SM work
assignment is known, a valid structure is

```text
C_s >= max(W_s / 4,
           max_k(P_s,k / throughput_k),
           max_w_assigned_to_s(critical_dependency_path_w),
           memory_service_bound_in_SM_cycles_s)
```

Here each pipeline's work `P_s,k`, its physical throughput, and each
dependency edge latency require separate justified definitions.
Instructions that can use multiple pipelines cannot be counted in
every pipeline simultaneously. A shared execution unit's capacity must
not be multiplied by four merely because four schedulers feed it.
The dependency expression is a path sum in the SM clock domain; paths
from different warps can overlap and must not simply be added.

Without an assignment of warps to SMs, keep the aggregate dispatch bound
and per-warp dependency expressions distinct. Under an explicit common
clock and `S` identical participating SMs, the corresponding elapsed
device-cycle lower bound can compare `W/(4*S)` with the longest warp
dependency path and similarly normalized capacity bounds. Taking a
maximum between aggregate SM cycles and a single warp's path cycles
without that normalization would mix different quantities. This note
introduces no pipeline throughput or dependency latency constants.

`sm__cycles_elapsed.sum` sums cycles over SM instances; each constituent
uses the SM clock domain. Summing `smsp__cycles_elapsed` would instead
sum over four subpartitions per SM and changes the denominator's unit.
Nsight's metric units and rollups must be preserved. [Nsight cycle
metrics and rollups](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide).

A wall-time bound `W/(4*S*f)` additionally assumes `S` participating SMs
and a specified common clock `f`; alternatively `f` must be a justified
upper bound for all relevant clocks. A nominal or advertised boost
frequency is not silently treated as a hard ceiling. This component
can stay in aggregate SM cycles and accept clock information only
under an explicit scenario. Fewer resident warps do not change the
hardware constant four, though they can prevent its capacity being
reached.

## Consistency with saved profiles

The following arithmetic uses the independently verified profiles in
`SOLVER_PROFILE_EVIDENCE.md` and `PLACEMENT_PROFILE_EVIDENCE.md`. Exact
source totals and hardware elapsed cycles are retained as separate
observations. Raw provenance and exact rational bounds are in
`verification/issue_capacity_arithmetic_20260905.json` beneath
`C:/local_working_projects/cubie-notes/hardware_unroll_placement`.

| Profile | Exact `W/4`, aggregate SM cycles | Observed `sm__cycles_elapsed.sum` | Ratio |
|---|---:|---:|---:|
| Lorenz R5 full | 2,022,869,076.25 | 2,983,676,640 | 0.677979 |
| Lorenz R5 Newton rolled | 2,060,470,993 | 2,666,686,292 | 0.772671 |
| Lorenz K3 full | 2,415,113,834.5 | 3,148,565,892 | 0.767052 |
| Lorenz K3 both rolled | 2,639,606,439.5 | 3,442,869,180 | 0.766688 |
| Chain32 K3 local stage-base | 1,276,954,566 | 3,516,229,714 | 0.363160 |
| Chain32 K3 shared stage-base | 1,291,109,042.25 | 4,736,610,342 | 0.272581 |

All six satisfy the inequality. This is a consistency check, not proof
of the architecture from fitted observations, and the ratios are not
coefficients for the model. The reviewed hardware/source instruction
residuals of 336/336/1647/1638/51/51 remain unresolved and are not
subtracted or turned into correction factors. Software and hardware
metrics may use separate replay passes; exact instruction-slot
utilization cannot be recovered by labeling the displayed ratios as
a directly measured issue-active counter.

The R5 issue bound rises under rolling while elapsed cycles fall. The
K3 issue bound rises alongside elapsed cycles. The stage-base placement
bound changes little compared with its observed slowdown. Thus this
physical component constrains the minimum work without replacing the
dependency, storage and instruction-delivery components.

## Proposed auditable source-work interface

The pre-compile interface should carry expressions rather than infer
iteration counts from profiling labels:

```text
architecture:
  compute_capability: [8, 9]
  schedulers_per_sm: 4
  maximum_logical_warp_issues_per_scheduler_cycle: 1
  provenance: Ada Figure 5 + CUDA 13.0 CC8.x sections

work_terms[]:
  helper_role / source_hash / function_id / source_region
  algorithm_family / inner_solver / unroll_policy
  effective_width: n or stages*n, from the actual call instance
  native_instructions_per_visit: expression or certified interval
  translation_contract: instruction types, flags, context, evidence
  warp_visits: symbolic expression with explicit control-flow scope
  iteration_bounds: values and citations from actual configuration
  unknowns: unproved lowering, control flow, or dynamic multiplicity

dispatch_bound:
  expression: certified_logical_warp_instruction_lower_bound / 4
  unit: aggregate_SM_cycles
  completeness: complete, partial-certified, or unresolved
```

`native_instructions_per_visit` must count emitted instructions, not
floating-point operations; one FMA is one issue. A source arithmetic
operation that folds, fuses, shares a subexpression or changes lowering
with context cannot automatically be counted as a guaranteed native
instruction. The conditional fragment translations in
`OPERATION_TRANSLATION_EVIDENCE.md` supply contracts to test, not blanket
counts for all generated helpers. No coefficients translating cycles
from benchmark timing are required or permitted here.

`warp_visits` must preserve reachable branch paths, predication and
loop nesting. Per-lane Newton/Krylov counter totals cannot supply it.
Newton, Krylov and adaptive-step multiplicities remain symbolic unless
the source/configuration proves them; a maximum iteration setting is
an upper bound, not an observed or compulsory number of visits.
Multiplying an upper bound on work by one quarter gives an upper bound
on the *dispatch-only work term*, not an upper bound on runtime and not
a valid lower bound on actual runtime.

Static loop expansion affects body size, control instructions and
translation opportunities. It must not multiply dynamic work again
after the corresponding symbolic loop visits already account for it.
If only a subset of unavoidable native work is certified, its disjoint
contributions can provide an explicitly partial lower bound while all
unresolved terms stay listed. Unknown work must not be silently marked
zero or replaced with measured average iterations. A numeric policy
ranking remains unsupported wherever these expressions cannot bound
the competing candidates.

## CPU component and concrete interface

`physical_capacity.py` implements this bounded arithmetic contract.
`dispatch_capacity(request)` takes JSON-compatible input with
`schema_version: 1`, `compute_capability: [8,9]`, an explicit
`execution_domain`, symbol definitions, and a `work` record. Work must
declare `kind: logical_native_warp_instructions`,
`scope: all_participating_sms`, provenance, and a qualification of
`certified_native_work`, `saved_native_profile_diagnostic`, or
`symbolic_native_definition`. This is a caller proof obligation, not
an automated proof of compiler lowering. Raw source operations or
unproved qualifications produce an unresolved result.

Exact constants are integers or `[numerator, denominator]`. Work
intervals use `lower` and `upper` endpoints; `upper: null` retains a
certified lower endpoint without claiming an upper endpoint. A
symbolic endpoint is a nonnegative polynomial, for example:

```json
{"terms": [
  {"coefficient": [1, 1], "powers": {"B_rhs": 1, "V_rhs": 1}},
  {"coefficient": [1, 1], "powers": {"B_linear": 1, "V_linear": 1}}
]}
```

Here each `B` is a still-unknown native instruction count per helper
visit, and each `V` is a still-unknown warp-visit count. Each symbol
requires its meaning, nonnegative counting domain and provenance.
The helper terms must describe disjoint native work; source common
subexpressions and aliases do not justify double counting. No values
are assigned to these symbols. Repeated monomials combine exactly;
interval ordering is conservatively proved by comparing nonnegative
coefficients. An ordering that this method cannot prove is unresolved.

The result separates `dispatch_work_interval` from its
`runtime_lower_bound`; the upper endpoint never becomes a runtime
upper bound. Aggregate SM cycles are always the primary scope.
`device_scenario` requires a positive SM count, explicit common-cycle
assumption and provenance. Optional `clock_scenario` additionally
requires `kind: explicit_common_clock`, exact positive `hertz`, and
provenance. Nothing queries or inserts a device clock automatically.

`combine_lower_bounds(bounds)` computes a maximum only when execution
domain, architecture, scope, units, symbolic definitions, qualification
and explicit scenarios match. It rejects mixing device cycles with
aggregate SM cycles, or bytes with cycles. External dependency and
pipeline bounds must already carry the matching physical scope and
their own evidence. No pipeline rate is supplied by this component.

`profile_example(path, expected_sha256)` builds a diagnostic request
from an independently reviewed saved analysis; the expected SHA must
come from its review receipt. It checks the exact source/aggregate
warp count and keeps hardware residuals separate. The CLI is:

```text
python benchmarks/hardware_model/physical_capacity.py --input INPUT.json --out FRESH_OUTPUT.json
```

Six source-hashed diagnostic examples, symbolic helper and interval
examples, and the validation receipt are under
`verification/capacity_component_validation_20260905` in the notes root.
The validation includes exact agreement with the six previously
reviewed rational bounds and rejection of incompatible domains/units,
raw source counts, unproved interval ordering and iteration means.
Its optional one-nanosecond common-cycle example checks dimensions;
it asserts no actual GPU frequency and supplies no model default.
No GPU query, CUDA import, native compilation, or launch is involved.
