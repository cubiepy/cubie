# Constant-bank indices through source floor division

A captured constant-table address is uniform when all scalar operands
are constants or coherent source induction values and all intermediate
operators are deterministic. The admitted FloorDiv form participates
in this recursive proof. It must separately pass the existing complete
source-range, positive constant-divisor and exact reciprocal/shift proof.
A nonuniform operand remains unproved even if its execution witness equals
an induction witness. No address is folded using a replay value.

This repairs the Fabbri FIRK correction norm source chain
`index // state_n`, then `stage_index * stage_count + contribution_index`.
The typed LDC, IMAD.HI/SHF and other native form choices are unchanged. The
range-optimal division form remains a compiler alternative, not a claim
that the installed backend selects it. Unproved table-index uniformity
continues to refuse the uniform constant-broadcast service.
