"""Rank exact costs with independent legal physical partition choices."""

from fractions import Fraction


def rational_cost(value):
    """Read an exact, positive cycle cost from the model's JSON format."""
    if type(value) is int or isinstance(value, Fraction):
        result = Fraction(value)
    elif (
        isinstance(value, list) and len(value) == 2
        and all(type(part) is int for part in value)
        and value[1] != 0
    ):
        result = Fraction(*value)
    else:
        raise ValueError("Cost must be exact integer, Fraction, or [n, d]")
    if result <= 0:
        raise ValueError("Cycle cost must be positive")
    return result


def encoded(value):
    """Encode an exact rational as a JSON numerator/denominator pair."""
    return [value.numerator, value.denominator]


def partition_keys(values):
    """Validate unique nonnegative partition sizes in canonical JSON keys."""
    if not values or len(set(values)) != len(values):
        raise ValueError("Legal partition inventory is empty or repeated")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Legal partition sizes must be nonnegative bytes")
    return {str(value) for value in values}


def rank_partition_envelopes(costs, legal_partitions):
    """Compute minimax regret over independent per-action partitions.

    Parameters
    ----------
    costs : dict
        Common scenario to action to canonical partition-byte string to
        positive exact cost. Each cost covers the same attempted work.
    legal_partitions : dict
        Common scenario to action to complete list of legal partition
        byte sizes, derived from hardware and that action's allocation.
        Every listed partition must have a finite cost.

    Returns
    -------
    dict
        Exact worst relative regret, ties, and attaining physical
        partition assignments. The partition choices can vary across
        actions; no common driver-rule correlation is asserted.
    """
    if not costs or set(costs) != set(legal_partitions):
        raise ValueError("Costs must cover every declared common scenario")
    actions = sorted(next(iter(legal_partitions.values())))
    if not actions:
        raise ValueError("Action inventory is empty")
    scenario_results, regrets, witnesses = {}, {}, {}
    for scenario in sorted(costs):
        legal = legal_partitions[scenario]
        rows = costs[scenario]
        if sorted(legal) != actions or sorted(rows) != actions:
            raise ValueError("Common scenarios must contain the same actions")
        values, extrema = {}, {}
        for action in actions:
            expected = partition_keys(legal[action])
            if set(rows[action]) != expected:
                raise ValueError("Finite costs do not cover legal partitions")
            values[action] = {
                key: rational_cost(value)
                for key, value in rows[action].items()
            }
            minimum = min(values[action].values())
            maximum = max(values[action].values())
            extrema[action] = dict(
                minimum_cost=encoded(minimum),
                maximum_cost=encoded(maximum),
                minimizing_partitions=sorted(
                    int(key) for key, value in values[action].items()
                    if value == minimum
                ),
                maximizing_partitions=sorted(
                    int(key) for key, value in values[action].items()
                    if value == maximum
                ),
            )
        for action in actions:
            assignment = {
                other: extrema[other]["minimizing_partitions"][0]
                for other in actions
            }
            assignment[action] = extrema[action]["maximizing_partitions"][0]
            realized = {
                other: values[other][str(partition)]
                for other, partition in assignment.items()
            }
            regret = realized[action] / min(realized.values())
            extrema[action]["worst_relative_regret"] = encoded(regret)
            extrema[action]["attaining_partition_assignment"] = assignment
            if action not in regrets or regret > regrets[action]:
                regrets[action] = regret
                witnesses[action] = dict(
                    common_scenario=scenario,
                    partitions=assignment,
                    realized_costs={
                        key: encoded(value) for key, value in realized.items()
                    },
                )
        scenario_results[scenario] = extrema
    best = min(regrets.values())
    ties = sorted(action for action in actions if regrets[action] == best)
    return dict(
        status="finite_independent_partition_envelope",
        default=ties[0], ties=ties,
        maximum_relative_regret={
            action: encoded(regrets[action]) for action in actions
        },
        attaining_witnesses=witnesses,
        scenario_envelopes=scenario_results,
        partition_coupling=(
            "Unknown driver coupling; each action independently takes any "
            "of its hardware-legal partitions within a common scenario"
        ),
        conservatism=(
            "Exact over the declared independent envelope; it may exceed "
            "regret under an unproved common driver partition rule"
        ),
        requested_hint_determines_partition=False,
        empirical_partition_fit=False,
    )
