# Joint register selection alternatives

The evaluator compares three common compiler hypotheses: surviving dynamic
local addresses remain addressable; every completely source-proved eligible
extent uses a source-domain ISETP/SEL chain; or every eligible extent uses a
whole-extent ISETP/SEL chain. Each hypothesis applies to every source action.
The candidate never chooses its own cheapest compiler form. There is no
array-size cutoff, fitted register adjustment, or native timing input.

`register_selection_mixin.py` extracts the independently reviewed e5
functions and mixin with identical AST. `register_selection_source.py`
retains the reviewed complete-source proof bytes. The standalone original
wrapper remains available in `register_selection.py`. The split removes
the policy import cycle without changing source eligibility or home algebra.

The policy factory composes caller retention outside selection and shared
forwarding inside it. Selection first replaces its own step boundary
observables with current selected homes; caller retention then adds its
actual scalar/cell/descriptor values to that observable set. Every form,
placement, forwarding choice and caller materialization receives a fresh
typed build and fresh allocation. The wrapper binds the selection request,
complete eligibility inventory, helper source hashes and caller identity;
its verifier rebuilds the complete plan. Existing addressable requests keep
their original lowering. Selection metadata is serialized canonically so
saved JSON plans rebuild exactly.

The instruction footprint recognizes selection's explicit relative-byte
IMAD producer and does not add a second supplemental address instruction.
The allocated event projection includes actual selection and spill/reload
operations with the existing qualified service catalog. SEL is an explicit
compiler alternative, not a claim that the installed compiler uses SEL
rather than destructive predicated MOV or another selection network.

Per-action legal physical partition envelopes remain independent of the
common compiler hypothesis. Requested partition preference is separate
from achieved physical capacity. Constant-division range-optimal lowering
remains a conditional source form; no installed-compiler discovery is
asserted. Attempted-step costs retain caller live-through resources, while
outer caller arithmetic and descriptor rematerialization have the existing
explicit interval qualification.
