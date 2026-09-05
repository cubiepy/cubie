# Explicit Runge--Kutta source policies

`erk_policy_graph.py` supplies the ERK adapter for the shared
`implicit_policy_graph.describe_policy_source` frontend. The actual
`ERKStep` owns the tableau, captured helper graph, buffer registry and all
eight unroll directives. Its workload has `inner_solver: null`, empty
solver roles and empty solver iteration scenarios. It has a distinct
explicit workload, graph, typed-body and allocated-plan kind.

The frontend requests cached Python dispatchers without specializing or
executing them. Actual system configuration, recursive factory settings,
tableau bytes and shared launch stride use the existing construction
identity. No native instruction count, timing, counter or fitted label is
an input.

## FSAL execution state

Call the frontend with an empty solver-scenario mapping and
`fsal_state={"first_step": bool, "all_lanes_accepted": bool}`. These facts
select runtime paths. The caller flags retain runtime value identities;
the source branch decisions remain instructions. The actual tableau
closure determines whether caching is enabled. A cacheable subsequent
step retains its `ActiveMask` and `AllSync` operations, qualified by the
declared full active warp and a separate ERK FSAL collective scope.

The expected RHS invocation count is the actual stage count minus one
exactly when caching is enabled, this is not the first step and every
participating lane accepted its previous step. The verifier checks the
complete captured RHS invocation count and the source FSAL collective.
Cached RHS values are caller persistent inputs; this model does not claim
to establish consistency with a prior timestep from a single step graph.

The existing selected-path FP32 replay provides exact output-bit
comparisons across policies with the same FSAL state. It is conditional
on that declared path; it does not predict convergence or acceptance.
Native FTZ equality remains a separate compiler hypothesis.

## Dynamic slices and constant returns

ERK passes a stage slice `stage_offset:stage_offset+n` into generated
helpers. The adapter reduces the typed integer expressions to affine
forms over source loop induction variables. Equal variable coefficients
prove a constant extent. Every intermediate interval must fit signed
int32 and the entire declared loop domain must stay inside the captured
array shape. A unit stride is required. The dynamic start identity and
byte-address expression survive into typed lowering; the execution
witness does not establish the extent. Graph `dynamic_slice_proofs`
retains these facts and the exact source site.

The step's constant success return remains a graph observable. Constant
observables do not form live register outputs in caller-cut liveness.

## Use and qualifications

The shared CLI accepts actual ERK aliases such as `rk23` and `dopri54`,
`--policy full,full,full,full,full,full,full,full`, and an explicit
`--fsal-state first_step`, `accepted_previous_step` or
`rejected_previous_step`. ERK rejects an inner-solver option. Placement
remains the actual solver buffer configuration. Direct Python callers
can configure `stage_accumulator_location` and `stage_rhs_location`.

Full, counted and compiler-directed loops use the same source frontend,
captured constant lookup, dynamic address, typed-body and allocation
machinery as the implicit families. The conditional instruction forecast
covers those admitted source forms. It remains a covered-body estimate,
not a complete native kernel or a temporal instruction working set.

Other explicit-step classes, including the separate Euler implementation,
require their actual adapters; the ERK route never labels them as ERK.
