# Exact verification reuse in a sealed epoch

The joint evaluator can reuse successful policy graph and policy plan
verification receipts within one process. It requires an explicit
`verification_cache` request containing kind `sealed_epoch_exact_proofs`,
an absolute source manifest path, and the exact SHA256 of that manifest.
The default behavior without this request remains ordinary verification.

The physical cause of the redundant work is the original exact plan
verifier: every scenario binding calls `verify_policy_plan`, which
reconstructs the complete typed lowering and allocation for a wrapper
already checked in another service scenario. `make_policy_plan` also
replays the complete graph verification. The cache targets only these
pure proof calls. Constructors, allocation choices, instruction address
projection, catalog interpretation and scheduling remain uncached.
Every different placement or compiler/caller materialization still
constructs and verifies its own complete allocation.

A cache key binds the complete positional and keyword inputs, verifier
module/name and import-time source bytes, the complete source manifest,
every absolute external file reference in the inputs, and the Python
runtime identity. The plan wrapper contains architecture, compiler
alternatives, caller inventory and its materialization identities. The
graph contains the exact policy, placement, construction, source and
execution-regime identities. There are no coarse family, size, timing,
or opcode-only keys.

Input serialization uses a restricted C Pickler with protocol 5. Custom
object reducers are refused; the standard exact Fraction reduction is
explicitly supported. New serialized hashes undergo a complete primitive
type and path-reference admission walk. It preserves tuple/list,
bool/int, numeric bit patterns and dictionary entry order. It never
unpickles data or invokes user-defined reducers. Semantically equivalent inputs with
different ordering or alias sharing can miss the cache; they cannot
produce a false successful hit. File references are discovered from all
absolute path strings, independent of schema field names.
The admission walker visits each container object and each distinct
string once. It checks possible absolute-path prefixes before creating
path objects. Its path inventory is retained only under the complete
type-preserving argument hash. Every call recomputes that entire hash
before reusing an inventory; neither object identity nor an old hash of
mutable inputs substitutes for current input bytes. External file bytes
are rehashed even when the path inventory is reused. This removes the
repeated Python tree walk without weakening successful-proof identity.

Every access rehashes the manifest and every sealed file. All imported
CuBIE and benchmark Python modules must belong to that manifest. A
decorated verifier's import-time file hash must match it. The source
epoch must remain immutable from interpreter startup; processes must
start from that epoch rather than reuse previously imported code after
editing files. External source/evidence file bytes are rehashed on each
access. A changed external reference changes the proof key and forces
ordinary verification, including its existing expected-hash checks.
Missing references are likewise part of the key, never treated as a
previous successful file read.

Only a successful proof whose inputs and external files stayed unchanged
during replay is stored. Failures are not cached. Stored and returned
receipts are copied so mutation of a returned receipt cannot alter later
results. Nested proof calls share the same process-local context; nested
epochs are refused. Leaving the context rechecks the entire epoch.

`verification_cache_receipt.json` records hit/miss/failure counts and
complete proof identities separately from model results. Native plan
bytes, cost values and selection outputs do not acquire cache metadata.
`uncached_proofs` allows an ordinary replay on the exact same inputs for
independent comparison. This is implementation acceleration, with no
hardware timing parameter, register multiplier or fitted coefficient.
