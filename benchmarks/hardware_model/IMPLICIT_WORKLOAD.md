# Actual implicit-family source adapter

`implicit_workload.py` connects actual DIRK, FIRK and Rosenbrock-W
constructions to the expansion graph and to explicit iteration-mask
scenarios. It requests cached Python dispatchers and checks zero native
overloads. It changes neither production code nor the frozen ERK model.
Its outputs are source workload descriptors, not native register or
timing predictions.

`describe_implicit_workload(solver, unroll)` binds each main Newton,
main linear and enabled error linear role independently to its actual
factory class, public correction setting, captured callable, width,
iteration cap, norm, initial-guess policy, buffers and helper graph.
The JSON uses `mr`/`sd` as short method identifiers and also retains the
actual `minimal_residual`/`steepest_descent` setting. The public CLI maps
these short identifiers to those actual settings. It constructs each
case in a fresh codegen cache and restores the prior cache root.

FIRK's main solve has width `stages*n`; its error solve has width `n`.
DIRK has one Newton call for each nonzero tableau diagonal, with a
conditional first-stage call when that stage is implicit. Rosenbrock-W
has one direct or iterative linear call per stage and no Newton loop.
The actual captured cached-solve and smoothed-error choices accompany
the separate role records. Cached/prefactored and uncached functions
remain distinct source/helper identities. Unsupported mixed direct and
iterative public wiring is rejected; no child factory is replaced to
manufacture a mixed construction.

The adapter does not consume `workload.py`'s error-iteration label or
its aggregate assertion that iteration counts are warp-body counts.
The former selects a label from the main solver type, and the latter
does not describe the patched per-lane Newton/BiCG counters. The frozen
file remains unchanged. Its call graph and expansion data are reused;
the actual role and counter semantics are derived separately here.

`evaluate_regime(descriptor, scenarios, step_entry_mask=None)` takes one
explicit call-entry mask for each actual step call. Unconditional calls
must share the same step-entry mask; a conditional first-stage call may
enter a subset. An explicit step mask is required when no unconditional
call supplies it. Newton calls require a shrinking active
mask for every body and one nested linear-call scenario per body. Krylov
calls likewise take shrinking unfinished masks and a terminal mask.
Caps, subset relations and required call membership are checked. These
are supplied convergence scenarios, not inferred iterations or measured
warp histories. Body entry, conditional per-lane arithmetic and native
issued instructions remain different quantities.

The source-derived distinctions are:

- Newton executes residual, linear and correction-norm helpers for each
  entered body, including lanes already frozen by convergence/failure.
  It adds Newton and returned linear counts only for Newton-active lanes.
- MR/SD increment their returned count once per entered warp body in
  every entered lane. BiCGSTAB increments only unfinished lanes. All
  methods still enter the body while any participating lane is unfinished.
- A BiCG body contains two operator/preconditioner/norm call sites;
  an MR/SD body contains one of each. Actual source call sites are
  checked, and the preconditioner count follows its captured condition.
- A nonzero initial guess adds an entry operator and residual norm.
  BiCG's zero-guess, zero-body case exits before its seed region and
  loop-top vote. Initial votes and loop-top votes are reported separately.
- LU writes one returned linear count per call. It has no Krylov loop.

These follow the retained callable ASTs in `newton_krylov.py` (body and
counter updates), `linear_solver.py`, `bicgstab_solver.py` and
`lu_solver.py`; each role record provides exact source path/hash/lines.
The functions' unweighted operations, coefficient folds, source call
sites and replicated regions join directly by function ID. Full,
counted and False directives remain candidate metadata. Symbolic Newton
and Krylov body work is not multiplied into hot-code copies, and no
native counted-unroll behavior is inferred.

The eleven Lorenz CPU constructions in `implicit_workload_cpu_v4` cover
Kvaerno3, RadauIIA5 and Rosenbrock23 with LU/MR/BiCG, plus cached and
prefactored RadauIIA5 with LU/BiCG. All have zero native overloads.
For RadauIIA5 the iterative main role is width 9, cap 14, zero guess;
the error role is width 3, cap 5, nonzero guess. Rosenbrock23's iterative
main role also uses a nonzero guess. These are properties of these
actual constructions, not family-wide constant defaults.

The same directory also retains SDIRK2 and implicit-midpoint LU source
constructions. Their conditional first-stage calls verify subset masks
and the explicit step-mask requirement when no unconditional call exists.
`verification/implicit_workload_20260905/step_mask_repair.json` binds these
actual descriptors and rejects inconsistent Rosenbrock stage populations.

The model integration consumes each role's source region and actual
call-instance width, supplies a convergence-mask scenario, and schedules
its entry/body/exit regions with their own materialization alternatives.
The adapter supplies the work composition needed for that integration.
It does not yet lower these conditional/recurrent regions into the ERK
NativePlan instruction graph. No historical winner lookup, fitted
iteration multiplier or source-liveness-to-GPR multiplier is introduced.
