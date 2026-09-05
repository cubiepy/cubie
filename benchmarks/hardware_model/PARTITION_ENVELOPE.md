# Independent legal partition envelopes

`rank_partition_envelopes` compares executable source-policy actions while
treating achieved L1/shared partition as a hardware scenario. A requested
CUDA carveout hint belongs to the action record, but does not determine
the physical partition used for occupancy or cache capacity.

The actual-handle audit in
`verification/carveout_handle_native_independent_e1` establishes that the
installed compatibility setter can update a different function handle
from the function launched by the MLIR backend. The measured partition
sizes are validation evidence, never parameters fitted by this module.
Earlier conditional physical-geometry costs remain separate evidence.

The caller supplies the complete hardware-legal partition list for every
action in each common compiler/service scenario. Feasibility follows
that action's allocated shared bytes, block geometry and hardware
allocation quanta. An action requiring a larger shared partition keeps
its larger legal choices even if another action can use smaller ones.
This module refuses missing finite costs for any declared legal choice;
the caller must report unresolved services before attempting a finite
selection. Costs must cover identical attempted work across the matrix.

Partition choices are independent across actions. This deliberately
retains unknown coupling in the driver's choices for different compiled
functions. It is an uncertainty envelope, not a claim that the driver
actually chooses each worst-case assignment. No common driver rule is
used to narrow it without separate source or hardware proof.

For a common scenario s, let C[a,p] be the positive cost of action a at
legal partition p. Write L[a] = min_p C[a,p] and U[a] = max_p C[a,p].
For a chosen action a and any joint partition assignment, relative
regret is C[a,p_a] / min_b C[b,p_b]. For multiple actions this equals
max(1, C[a,p_a] / min_{b != a} C[b,p_b]). It is monotone increasing in
the chosen action's cost and decreasing in every competing action's
cost. Its exact maximum over independent legal partition choices is

    max(1, U[a] / min_{b != a} L[b]).

The maximum is attained by assigning a a maximizing partition and each
competitor a minimizing partition. The singleton-action regret is one.
The module constructs this joint witness and evaluates its exact ratio,
which also handles singleton and tied cases without a special division
by an empty competitor set. It then maximizes over the declared common
compiler/service scenarios and returns all minimax ties.

No Cartesian enumeration is required by the implementation. The author
validation compares it with exhaustive Cartesian enumeration on small
positive matrices, including an action whose only legal partition is
larger than some legal partitions of another action. Fixed-physical-partition
comparisons are separate conditional diagnostics; they cannot remove an
otherwise launchable action from this envelope.

This module performs arithmetic over admitted costs. It neither
constructs allocations nor validates source/native semantic equivalence.
Source graph, allocation, geometry, common-work and service admission
remain responsibilities of the joint evaluator and its independent
review. Driver request fields are retained by that evaluator as action
identity, separate from these achieved-capacity scenarios.
