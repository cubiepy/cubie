# Finite joint policy evaluation

`joint_policy_evaluation.py` constructs actual post-codegen source graphs,
allocates their typed native alternatives, and produces usable conditional
rankings. Every comparison belongs to one actual system, algorithm family,
inner solver and declared attempted-step regime. ERK has no inner-solver
axis. DIRK, FIRK and Rosenbrock each retain LU, minimal-residual and
BiCGSTAB as separate workloads.

The default system is Lorenz. A request selects the system, workloads,
source iteration counts, coherent masks through the existing source
regime constructor, and ERK FSAL entry state. Observed solver times,
iteration counters and native register labels are not inputs. The service
catalog contains hardware microbenchmark results and explicitly qualified
architectural transfers, not solver timing fits.

Full requests enumerate Newton and Krylov counts 1, 2 and 4 plus the
actual source-cap endpoint, removing irrelevant axes for explicit and
direct-solve workloads. The `source_cap` count token resolves separately
from each actual role, including different main and smoothing solver
caps. The family ranking combines complete scenario rows across these
declared regimes and identifies exact cap rows. Executed call counts are
checked against their own caps; sampled counts do not stand in for cap
execution. Cap endpoints do not enumerate every possible intermediate
count, mask or branch regime. Minimal-residual paths explicitly
declare a nonzero denominator and an active update; requests can change
these branch choices. The emitted graph retains the exact choices.

System construction applies the historical sweep's explicit constants
before code generation, including Fabbri ANS. A request can select other
constants or a numerical replay point. Replay values validate semantics;
they do not specialize runtime source values or enter service costs.

## Candidate design

The pilot crosses full unroll and counted-stage-one with all-local and
all-shared placements of actual positive-size relocatable step buffers.
The full design includes the full baseline, one-group count1/count2/count4/
False variants for every source-present group, and joint counted/False
variants. Its placement design contains local, each single shared buffer,
and all shared buffers. Policies and placements are crossed. This is a
finite experimental design, not an exhaustive search of all flag and
placement products. Explicit policy and placement lists in the request
define additional factorial or bisection cohorts without winner tables.

Targets come from the actual registry and public placement settings used
by the historical placement sweep. Requested settings must reach the
actual registry. Source workload identity and complete selected iteration
regime must agree across every admitted candidate. Every plan is freshly
allocated and verified against its own graph. The ordinary scenario
builder verifies it again before deriving actual shared stride, bank
wavefront demand and residency.

Register caps and shared forwarding are compiler hypotheses, not controls
that the heuristic is allowed to optimize secretly. The pilot uses the
largest source-observed unpressured allocation peak in its cohort. The
full design adds upper endpoints of published hardware occupancy
plateaus, derived by enumerating legal per-thread register counts and
applying allocation quanta, subpartition capacities and block limits.
The source peak caps the enumeration because larger budgets reproduce
the unpressured allocation. Shared forwarding has both retained-load and
proved within-region forwarding alternatives.

The same cap/forwarding scenario is applied to every compared candidate.
Infeasible entry inventories, invalid source constructions and illegal
launch geometries are retained explicitly. A minimax comparison only uses
scenario rows complete for the admitted candidate set; every incomplete
row and its reason remains in the result. A missing scenario is never
assigned a favorable cost. Results name the exact complete scenario
cohort and must not be interpreted as robustness to the omitted rows.

Block size and carveout are explicit candidate controls. The initial block
is 128 threads; requests can list other legal geometries. Full requests
enumerate the hardware's supported carveouts. Register occupancy includes
the actual shared allocation and published allocation quanta. Scenarios
which have byte-for-byte identical event/service/geometry inputs share
one cached schedule evaluation; their candidate identities and ties stay
visible. Each cost link names both the candidate's typed plan and the
representative plan retained in the shared schedule, plus their exact
event/scenario/geometry equivalence key. No measured ranking prunes
candidates.

## Equal work and finite hardware costs

All candidates execute exactly twice the least common multiple of their
admitted resident warp counts. Thus every estimate has the same number
of attempted warp steps and at least two complete occupancy waves.
Results retain total cycles and cycles per attempted warp step. The
finite matrix is ranked with exact rational minimax relative regret.
All equal-regret defaults are retained; the existing selector supplies a
deterministic representative.

The pilot declares complete L1-hit local load paths. Full requests use
the sector-cache model with actual local frames, shared carveout-adjusted
L1, recorded host L2 and both published L2 latency alternatives. The
physical assumptions remain explicit: fully associative sectors, equal
per-SM L2 capacity share, reused local slots, unlimited pending fills, and
observed L1 write-through behavior. Both store-visibility interpretations
are retained. Requests may also select complete fixed L1/L2/DRAM paths
without enabling the cache model.

Large common work counts do not require blindly repeating identical
cache waves. For reused physical slots only, the harness records the
entire drained L1/L2 ordered sector/dirty state at each wave boundary.
When the exact state repeats, deterministic scheduling of the same
operations has the same relative timestamps and traffic. The harness
multiplies that complete period's cycles and counter increments. Pending
fills must be empty. No address renaming, approximate state hash match,
warmup threshold or convergence tolerance is used. The hash in the
receipt identifies a state already compared by exact tuple equality.
Trajectory-unique backing uses ordinary uncompressed scheduling.
Instruction-hierarchy scenarios also use ordinary repeated waves because
data-cache state alone cannot establish instruction-cache recurrence.

## Instruction delivery and interpretation

Each allocation retains a synthetic 16-byte instruction-address
projection with dynamic-event links, accessed-PC union and complete
reserved cap span. The maximal-static-form envelope qualification
remains attached. These quantities are not treated as a cache hot set or
as native code size.

The pilot leaves instruction fetch as a candidate-specific symbolic term.
Full requests cross each candidate with an explicit perfect-delivery
baseline and the qualified instruction-hierarchy transfer fixture. This
fixture uses Turing constant-path service transfers with named cache
domain assumptions; it is not a measurement of target instruction-fill
service. The request records its exact evidence hash and author status.
Both alternatives enter the same uncertainty matrix. Additional named
`instruction_fetch_hypotheses` replace this finite set without changing
candidate actions. Exact graph, plan and event-PC identities are bound by
the ordinary scenario builder.

The result is conditional on its enumerated hardware, compiler and source
regimes. It is not a complete-kernel timing prediction or a bound on total
runtime. No fitted cache penalty, missing prefetcher depth or invented
line-fill latency forces a complete winner. Source caps omitted from a
custom regime grid and target instruction service remain residual
uncertainties even when finite alternatives have a minimax winner.

## Invocation and artifacts

Write a reviewable request first:

```powershell
python -m benchmarks.hardware_model.joint_policy_evaluation `
  --write-request --design pilot `
  --evidence-root C:/local_working_projects/cubie-notes/hardware_unroll_placement `
  --output C:/path/pilot_request.json
```

Run against the active research checkout on the Python import path:

```powershell
python -m benchmarks.hardware_model.joint_policy_evaluation `
  --request C:/path/pilot_request.json --output C:/path/pilot_results
```

The output retains request, source snapshots, every source graph,
compiler plan, instruction projection, distinct schedule, rejection and
finite ranking. The final receipt records any source files changed while
the cohort ran. Source construction and allocation failures remain
diagnosable rather than being replaced by measured labels. Independent
verification is required before these author results inform defaults.
