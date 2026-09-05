# Joint candidate comparison on common work

`candidate_selection.py` consumes source-only policy graphs and their typed
allocation plans. A v2 request binds each graph and plan by path and SHA256.
Admission reconstructs the complete policy plan, checks all eight directives,
and compares every owner-qualified registry placement. Native labels, solver
timings and measured iteration counters are excluded from prediction.

Common workload identity comes from the actual ODE source identity, typed
system configuration, recursive solver configurations and complete tableau
coefficient bytes. Only the decision axes (unroll and placement fields) are
removed from that semantic identity. Private codegen-cache paths do not make
otherwise identical workloads different. Policies, placements and launch
geometry together define a distinct candidate; placement candidates can share
the same loop directives.

Each candidate supplies one block size and exact dynamic shared bytes from
the generated global shared stride. An explicit additional shared reservation
is permitted. Residency applies the supplied hardware register/shared quanta,
subpartition capacities, thread limits, block limits and shared carveout.
Actual per-block registers and the rounded `regsAssumedPerCTA` launch limit
remain separate quantities.

## Comparable execution

The scheduler evaluates one synchronized resident wave in SM cycles. Register
read-after-write and write-after-write dependencies, one issue per scheduler
per cycle, shared pipeline capacity, and the catalog's qualified dependencies
constrain issue. This is a conditional scheduler, not the installed compiler
or hardware warp-selection policy.

The request must declare common work:

```json
{"kind": "synchronized_full_waves", "warp_attempts_per_sm": 96}
```

The same number of warp attempts applies to every candidate. It must be a
multiple of each legal candidate's resident warp count and contain at least
two complete waves. The comparison cost is wave cycles multiplied by that
candidate's wave count. Simultaneous wave start/drain is an explicit modeling
assumption; asynchronous CTA replacement and launch tails are not inferred.

This normalization matters. A synthetic one-operation, 100-cycle dependency
fixture takes 103 cycles for 16 resident warps and 111 for 48 resident warps.
For the same 96 warp attempts the costs are 618 and 222 cycles. Comparing
103 against 111 would incorrectly reward doing less work. The fixture tests
the comparison algorithm; its latency is not a hardware model constant.

## Instruction and memory service boundary

Unknown opcode services and instruction delivery remain explicit symbols.
No winner is issued while a candidate has unresolved terms. Typed source
templates and executed instances are reported separately; a template count
is not a native instruction-byte forecast.

An instruction-delivery observation can enter a finite estimate only as
`exposed_instruction_stall_cycles`, scoped to one synchronized resident wave.
Its schedule binding must match the trace, resident warps, service catalog
and cache scenario. This term represents additional stalls after execution
overlap has been accounted for. Raw fetch service, miss counts and total
delivery latency cannot be added to arithmetic execution as if all service
were serialized. The current evidence does not supply this bound term.

The finite decision rule uses exact rational arithmetic:
`max_s(T[c,s] / min_q T[q,s])`, minimized across candidates. Its recommendation
applies only to the enumerated candidates, scenarios and common work. A
conditional symbolic result is not a completed hardware heuristic.

Run with an exclusive output path:

```text
python -m benchmarks.hardware_model.candidate_selection \
  --request request.json --out result.json
```

The source-only admission layer imports the existing CuBIE construction
modules. It does not request native specialization or launch kernels.
