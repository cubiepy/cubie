# Source numerical replay points

`describe_policy_source(..., numerical_replay={"kind":
"source_default_first_step", "time": 0.0, "effective_dt":
"untruncated_captured_initial_dt"})` binds an optional numerical check to
actual FP32 system defaults and caller initialization. The start time and
absence of event truncation are declared runtime assumptions. The timestep
comes from the actual caller closure. There is no fitted epsilon or altered
system parameter.

The builder reads actual step-call argument bindings, initial-state and
parameter copy statements, initial flags, counter resets, zeroing allocator
records and the exact no-op initializer source. Drivers or observables that
require evaluation need additional source interpretation and are rejected
by this point constructor. Every source live-in receives exact typed bits;
internal loop witnesses retain their distinct source-derived origin. The
point binds source input identities, actual workload and defaults, complete
policy and caller identity. It is replay metadata only: graph extraction,
dynamic operation counts, typed lowering and allocation do not consume
these values as compiler constants.

The numerical certificate executes the already-declared source path.
Newton/Krylov regimes remain separate hypotheses. A passing replay does not
prove that the chosen runtime convergence path occurs at these defaults.
Without an explicit point, the existing three deterministic input probes
remain in use.

Math-capable graphs declare IEEE FP32 infinity intermediates, with NaN
rejected and finite boundary results required. Each operation that produces
or consumes infinity is retained with source identity and exact input and
output bits. Divide-by-zero and overflow notifications are scoped to these
explicit IEEE source operations; invalid operations still raise. FP32
gradual underflow remains the source replay convention. This numerical
certificate does not assert equality with native approximate/FTZ math.

At actual Fabbri defaults, ACh is exactly zero. The unconditional source
term `ACh ** precision(-1.6951)` produces positive infinity; the subsequent
reciprocal produces zero. Rejecting every nonfinite intermediate therefore
rejects a valid finite-output IEEE path. The former finite-intermediate
failure remains preserved in external evidence. No operation is skipped
and neither a positive replacement ACh nor a guessed perturbation is used.

Cohort comparison checks exact boundary and observable bits at equal input
contracts. Per-candidate exceptional-operation identities and replay-point
hashes remain separate evidence because operation and value IDs can differ
under different unroll policies.
