"""Audit unroll banks without fitting parameters or inventing measurements.

Each observed launch is ranked by its minimum stored timing divided by the
minimum all-full timing in the same wave/block. Cubin aliases can inherit
that observation but never increase the number of measured repetitions.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = 2
FIXED_GROUPS = (
    "unroll_stage",
    "unroll_step_element",
    "unroll_accumulator",
    "unroll_solver_element",
    "unroll_norms",
)
SEVEN_GROUPS = FIXED_GROUPS + (
    "unroll_other_small",
    "unroll_converged_exits",
)
EIGHT_GROUPS = FIXED_GROUPS + (
    "unroll_other_small",
    "unroll_newton_exits",
    "unroll_krylov_exits",
)


def read_jsonl(path):
    """Load rows with exact line receipts and file identity."""
    path = Path(path).resolve()
    rows, errors = [], []
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeError) as error:
                errors.append(dict(line=line, error=str(error)))
                continue
            row = dict(row)
            row["_receipt"] = {"path": str(path), "line": line}
            rows.append(row)
    return rows, dict(
        path=str(path),
        sha256=digest.hexdigest(),
        rows=len(rows),
        parse_errors=errors,
    )


def config_key(row):
    return row.get("system"), row.get("algo")


def wave_key(row):
    return row.get("wave", ""), row.get("block", 0)


def policy_of(row):
    return (
        row.get("policy") or row.get("label", "").split("#")[0].split("@")[0]
    )


def counter_dict(values):
    return dict(sorted(Counter(values).items(), key=lambda item: str(item[0])))


def numerical_summary(rows):
    """Describe stored checks without assuming a numerical tolerance."""
    warm = [r for r in rows if r.get("task") == "solve" and r.get("warm")]
    failures = [r for r in warm if r.get("status_hist", {}).get("failed", 0)]
    mismatches = [r for r in warm if r.get("nan_match") is False]
    return {
        "warm_rows": len(warm),
        "rows_with_failed_runs": len(failures),
        "maximum_failed_runs": max(
            (r.get("status_hist", {}).get("failed", 0) for r in warm),
            default=0,
        ),
        "nan_mask_mismatches": len(mismatches),
        "nonzero_max_abs_difference_rows": sum(
            r.get("max_abs_diff", 0) > 0 for r in warm
        ),
        "first_failure": failures[0]["_receipt"] if failures else None,
        "first_nan_mismatch": mismatches[0]["_receipt"]
        if mismatches
        else None,
        "interpretation": "Recorded diagnostics only; no tolerance, "
        "trajectory validity or completion is inferred.",
    }


def resource_view(row):
    if row is None:
        return None
    return {
        "receipt": row["_receipt"],
        "source_hash": row.get("source_hash"),
        "compiler_identity": row.get("compiler_identity"),
        "cubin_sha": row.get("cubin_sha"),
        "regs": row.get("regs"),
        "local_bytes": row.get("local_bytes"),
        "spill_store_bytes": row.get("spill_store_bytes"),
        "spill_load_bytes": row.get("spill_load_bytes"),
        "sass_instructions": row.get("sass_counts", {}).get("instructions"),
        "sass_ldl_stl": row.get("sass_counts", {}).get("ldl_stl"),
        "occupancy": row.get("occupancy"),
    }


def compile_identity_key(row):
    """Keep identical bytes from different compile settings separate."""
    if row is None:
        return None
    return (
        row.get("cubin_sha"),
        row.get("source_hash"),
        json.dumps(row.get("compiler_identity"), sort_keys=True),
    )


def wave_protocol(rows, wave):
    """Resolve a recorded protocol, or the documented historical protocol."""
    manifests = [
        r
        for r in rows
        if r.get("task") == "cohort_manifest" and r.get("wave") == wave
    ]
    if manifests:
        identities = {r.get("manifest_sha256") for r in manifests}
        if len(identities) != 1:
            return None, "conflicting_cohort_manifests"
        manifest = manifests[-1].get("manifest", {})
        return manifest, "recorded_cohort_manifest"
    if wave in {"", "n", "f", "s", "t", "r"}:
        # Historical bank contract: placement_landscape.py ROUNDS=2,
        # REPEATS=3. These are protocol counts, not fitted thresholds.
        return {"protocol": {"rounds": 2, "repeats": 3}}, (
            "historical_unroll_harness_two_rounds_three_repeats"
        )
    return None, "missing_cohort_manifest"


def launch_eligibility(values, warm, compile_row, manifest):
    """Return explicit reasons a stored launch cannot support a ranking."""
    reasons = []
    if compile_row is None or not all(
        compile_row.get(field) for field in ("source_hash", "cubin_sha")
    ):
        reasons.append("missing_or_ambiguous_compile_identity")
    if manifest is None:
        reasons.append("missing_or_conflicting_protocol")
    protocol = manifest.get("protocol", {}) if manifest else {}
    rounds, repeats = protocol.get("rounds"), protocol.get("repeats")
    if not (
        type(rounds) is int
        and rounds > 0
        and type(repeats) is int
        and repeats > 0
    ):
        reasons.append("unknown_repeated_sample_protocol")
    else:
        expected = {(r, k) for r in range(rounds) for k in range(repeats)}
        observed = {(r.get("round"), r.get("rep")) for r in values}
        if observed != expected or len(values) != len(expected):
            reasons.append("incomplete_or_duplicate_round_repeat_samples")
    if len({r.get("key") for r in values}) != len(values):
        reasons.append("duplicate_sample_keys")
    policies = {policy_of(r) for r in values + warm}
    if len(policies) != 1 or any(
        policy_of(r) != r.get("label", "").split("#")[0].split("@")[0]
        for r in values + warm
    ):
        reasons.append("mixed_or_mislabeled_sample_policy")
    if not all(
        isinstance(r.get("kernel_ms"), (float, int))
        and math.isfinite(r["kernel_ms"])
        and r["kernel_ms"] > 0
        for r in values
    ):
        reasons.append("nonpositive_or_nonfinite_timing")
    if len(warm) != 1:
        reasons.append("missing_or_duplicate_warm_snapshot")
    for field in ("duration", "n_runs"):
        signatures = {r.get(field) for r in values + warm}
        if (
            len(signatures) != 1
            or None in signatures
            or any(
                not isinstance(value, (float, int)) or value <= 0
                for value in signatures
            )
        ):
            reasons.append(f"missing_or_mixed_{field}")
    if warm:
        snapshot = warm[0]
        status = snapshot.get("status_hist", {})
        if status.get("failed") != 0:
            reasons.append("failed_runs_or_missing_status_check")
        if snapshot.get("nan_match") is not True:
            reasons.append("nan_mask_mismatch_or_missing_check")
        geometry = snapshot.get("geometry", {})
        if geometry.get("waves", 0) < 2:
            reasons.append("fewer_than_two_or_unknown_occupancy_waves")
        for field in ("blocksize", "dynshared"):
            expected_value = geometry.get(field)
            if expected_value is None or any(
                r.get(field, expected_value) != expected_value for r in values
            ):
                reasons.append(f"missing_or_mixed_launch_{field}")
    if compile_row is not None:
        source = compile_row.get("source_hash")
        if any(r.get("source_hash", source) != source for r in values + warm):
            reasons.append("sample_compile_source_mismatch")
        compiler_digest = hashlib.sha256(
            json.dumps(
                compile_row.get("compiler_identity"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if any(
            r.get("compiler_identity_sha256", compiler_digest)
            != compiler_digest
            for r in values + warm
        ):
            reasons.append("sample_compile_toolchain_mismatch")
        if manifest and "source_hash" in manifest:
            if source != manifest["source_hash"]:
                reasons.append("cohort_compile_source_mismatch")
            identity = manifest.get("compiler_identity")
            if (
                identity is None
                or compile_row.get("compiler_identity") != identity
            ):
                reasons.append("cohort_compile_toolchain_mismatch_or_missing")
    return reasons


def observed_launches(rows, compiles, full):
    """Preserve raw launches; rank only complete compatible valid cohorts."""
    samples, warms = defaultdict(list), defaultdict(list)
    for row in rows:
        if row.get("task") != "solve":
            continue
        key = (row.get("label"),) + wave_key(row)
        (warms if row.get("warm") else samples)[key].append(row)
    local_reasons, protocols = {}, {}
    for (label, wave, block), values in samples.items():
        manifest, origin = wave_protocol(rows, wave)
        protocols[wave] = (manifest, origin)
        local_reasons[(label, wave, block)] = launch_eligibility(
            values,
            warms.get((label, wave, block), []),
            compiles.get(policy_of(values[0])),
            manifest,
        )
    observations = []
    for (label, wave, block), values in samples.items():
        finite = [
            r
            for r in values
            if isinstance(r.get("kernel_ms"), (float, int))
            and math.isfinite(r["kernel_ms"])
        ]
        fastest = (
            min(finite, key=lambda r: r["kernel_ms"]) if finite else values[0]
        )
        policy = policy_of(fastest)
        compile_row = compiles.get(policy)
        warm = warms.get((label, wave, block), [])
        reasons = list(local_reasons[(label, wave, block)])
        reference_key = (full, wave, block)
        reference_values = samples.get(reference_key, [])
        reference = min(
            (
                r["kernel_ms"]
                for r in reference_values
                if isinstance(r.get("kernel_ms"), (float, int))
                and math.isfinite(r["kernel_ms"])
            ),
            default=None,
        )
        if not reference_values or local_reasons.get(reference_key):
            reasons.append("missing_or_ineligible_same_block_reference")
        else:
            for field in ("duration", "n_runs"):
                if fastest.get(field) != reference_values[0].get(field):
                    reasons.append(f"reference_{field}_mismatch")
            reference_compile = compiles.get(full)
            if (
                compile_row is None
                or reference_compile is None
                or (
                    compile_row.get("source_hash")
                    != reference_compile.get("source_hash")
                    or compile_row.get("compiler_identity")
                    != reference_compile.get("compiler_identity")
                )
            ):
                reasons.append("reference_compile_identity_mismatch")
        manifest, protocol_origin = protocols[wave]
        if manifest and "source_hash" in manifest:
            duplicate_key = (full + "#2", wave, block)
            if duplicate_key not in samples or local_reasons.get(
                duplicate_key
            ):
                reasons.append("missing_or_ineligible_duplicate_reference")
            elif reference_values:
                duplicate_values = samples[duplicate_key]
                for field in ("duration", "n_runs"):
                    if duplicate_values[0].get(field) != reference_values[
                        0
                    ].get(field):
                        reasons.append(f"duplicate_reference_{field}_mismatch")
                duplicate_compile = compiles.get(
                    policy_of(duplicate_values[0])
                )
                if compile_identity_key(
                    duplicate_compile
                ) != compile_identity_key(compiles.get(full)):
                    reasons.append(
                        "duplicate_reference_compile_identity_mismatch"
                    )
        ratio = fastest["kernel_ms"] / reference if not reasons else None
        observations.append(
            {
                "label": label,
                "policy": policy,
                "wave": wave,
                "block": block,
                "sample_count": len(values),
                "distinct_sample_keys": len({r.get("key") for r in values}),
                "sample_receipts": [r["_receipt"] for r in values],
                "sample_timestamps": [r.get("time") for r in values],
                "minimum_sample": fastest["_receipt"],
                "kernel_ms_min": fastest.get("kernel_ms") if finite else None,
                "kernel_ms_max": max(
                    (r["kernel_ms"] for r in finite), default=None
                ),
                "reference_ms": reference,
                "ratio_to_full": ratio,
                "eligible": not reasons,
                "rejection_reasons": sorted(set(reasons)),
                "protocol_origin": protocol_origin,
                "blocksize": fastest.get("blocksize")
                or (
                    warm[-1].get("geometry", {}).get("blocksize")
                    if warm
                    else None
                ),
                "duration": fastest.get("duration"),
                "n_runs": fastest.get("n_runs"),
                "cubin_sha": compile_row.get("cubin_sha")
                if compile_row
                else None,
                "resources": resource_view(compile_row),
                "compile_identity": compile_identity_key(compile_row),
                "warm_receipts": [r["_receipt"] for r in warm],
                "numerical_checks": numerical_summary(warm),
            }
        )
    return observations


def analyse_config(
    config, rows, compile_rows, groups, include_observations=False
):
    """Audit one configuration, preserving aliases as inherited evidence."""
    compiles = {}
    identities = defaultdict(set)
    for row in sorted(compile_rows, key=lambda r: r.get("time", 0)):
        if row.get("status") == "ok":
            compiles[row["policy"]] = row
            identities[row["policy"]].add(compile_identity_key(row))
    conflicts = {
        policy: sorted(values, key=str)
        for policy, values in identities.items()
        if len(values) > 1
    }
    for policy in conflicts:
        del compiles[policy]
    full = "u" + "1" * len(groups)
    observations = observed_launches(rows, compiles, full)
    measured = [r for r in observations if r["ratio_to_full"] is not None]
    measured.sort(key=lambda r: r["ratio_to_full"])
    sha_observations = defaultdict(list)
    for row in measured:
        if row["compile_identity"]:
            sha_observations[row["compile_identity"]].append(row)
    aliases = [r for r in rows if r.get("task") == "alias"]
    holes = []
    for row in aliases:
        identity = compile_identity_key(compiles.get(row["label"]))
        if identity not in sha_observations:
            holes.append(
                {
                    "policy": row["label"],
                    "equals": row.get("equals"),
                    "cubin_sha": row.get("cubin_sha"),
                    "receipt": row["_receipt"],
                    "reason": "No matching cubin has a normalizable stored timing.",
                }
            )
    candidates = []
    for policy, compile_row in compiles.items():
        observed = sha_observations.get(compile_identity_key(compile_row), [])
        if not observed:
            continue
        best = observed[0]
        candidates.append(
            {
                "policy": policy,
                "ratio_to_full": best["ratio_to_full"],
                "observation": best,
                "compile_receipt": compile_row["_receipt"],
                "inherits_identical_cubin": policy != best["policy"],
            }
        )
    fixed_indexes = [
        groups.index(group) for group in FIXED_GROUPS if group in groups
    ]
    fixed = [
        row
        for row in candidates
        if row["policy"].startswith("u")
        and len(row["policy"]) == len(groups) + 1
        and all(row["policy"][index + 1] == "1" for index in fixed_indexes)
    ]
    best = measured[0] if measured else None
    best_fixed = (
        min(fixed, key=lambda r: r["ratio_to_full"]) if fixed else None
    )
    if best_fixed:
        best_fixed["ratio_to_unrestricted_best"] = (
            best_fixed["ratio_to_full"] / best["ratio_to_full"]
        )
    group_minima = {}
    for index, group in enumerate(groups):
        levels = {}
        for candidate in candidates:
            policy = candidate["policy"]
            if not policy.startswith("u") or len(policy) != len(groups) + 1:
                continue
            level = policy[index + 1]
            if (
                level not in levels
                or candidate["ratio_to_full"]
                < (levels[level]["ratio_to_full"])
            ):
                levels[level] = candidate
        group_minima[group] = levels
    result = {
        "system": config[0],
        "algo": config[1],
        "groups": list(groups),
        "family": next(
            (r.get("family") for r in rows if r.get("task") == "features"),
            None,
        ),
        "effective_solver": "bicgstab"
        if config[1].endswith("_bicgstab")
        else "not inferred; inspect algorithm settings",
        "feature_rows": [
            dict(r, _receipt=r["_receipt"])
            for r in rows
            if r.get("task") == "features"
        ],
        "compile_policies_ok": len(compiles),
        "ambiguous_compile_policies_excluded": conflicts,
        "distinct_compiled_cubins": len(
            {r.get("cubin_sha") for r in compiles.values()}
        ),
        "observed_launch_groups": len(observations),
        "normalizable_launch_groups": len(measured),
        "rejected_observations": [
            r for r in observations if not r["eligible"]
        ],
        "sample_count_distribution": counter_dict(
            r["sample_count"] for r in observations
        ),
        "alias_events": len(aliases),
        "unique_alias_policies": len({r["label"] for r in aliases}),
        "alias_coverage_holes": holes,
        "unmeasured_compiled_policies": [
            policy
            for policy, row in compiles.items()
            if compile_identity_key(row) not in sha_observations
        ],
        "best_observed": best,
        "best_five_full": best_fixed,
        "group_level_minima": group_minima,
        "numerical_checks": numerical_summary(rows),
        "capped_rows": [r for r in rows if r.get("task") == "capped"],
    }
    if include_observations:
        result["observations"] = observations
        result["alias_records"] = aliases
    return result


def history_audit(directory, current_rows):
    """Identify retained historical rows without pooling their timings."""
    current_ids = {(r.get("key"), r.get("time")) for r in current_rows}
    union = {(r.get("key"), r.get("time")): r for r in current_rows}
    snapshots = []
    for path in sorted(directory.glob("records.jsonl.bak-*")):
        rows, provenance = read_jsonl(path)
        timings = [
            r for r in rows if r.get("task") == "solve" and not r.get("warm")
        ]
        snapshots.append(
            {
                "provenance": provenance,
                "timing_rows": len(timings),
                "timed_configs": len({config_key(r) for r in timings}),
                "rows_not_retained_in_current": sum(
                    (r.get("key"), r.get("time")) not in current_ids
                    for r in rows
                ),
            }
        )
        for row in rows:
            union.setdefault((row.get("key"), row.get("time")), row)
    reference_warms = [
        r
        for r in union.values()
        if r.get("task") == "solve"
        and r.get("warm")
        and r.get("label") in {"u1111111", "u11111111"}
    ]
    return {
        "identity": "stored key plus timestamp; no timing pooling",
        "snapshots": snapshots,
        "union_records": len(union),
        "union_timing_rows": sum(
            r.get("task") == "solve" and not r.get("warm")
            for r in union.values()
        ),
        "reference_warm_cohorts": [
            {
                k: r.get(k)
                for k in (
                    "system",
                    "algo",
                    "wave",
                    "block",
                    "duration",
                    "n_runs",
                    "kernel_ms",
                    "time",
                    "geometry",
                    "_receipt",
                )
            }
            for r in sorted(reference_warms, key=lambda r: r.get("time", 0))
        ],
        "limitation": "Repeated keys can represent reruns with changed "
        "duration, timing protocol, geometry or source. Match "
        "those identities before comparing independent cohorts.",
    }


def analyse_bank(directory, include_history=False, include_observations=False):
    """Return alias-aware measurements, provenance and coverage diagnostics.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory containing records.jsonl and compiles.jsonl.
    include_history : bool
        Audit backup snapshots separately, without pooling their timings.
    include_observations : bool
        Include every launch's sample receipts and every alias event.

    Returns
    -------
    dict
        Auditable measured rankings and explicit missing evidence.
    """
    directory = Path(directory).resolve()
    rows, records_provenance = read_jsonl(directory / "records.jsonl")
    compiles, compiles_provenance = read_jsonl(directory / "compiles.jsonl")
    by_config, compile_config = defaultdict(list), defaultdict(list)
    for row in rows:
        by_config[config_key(row)].append(row)
    for row in compiles:
        compile_config[config_key(row)].append(row)
    policies = [r.get("policy", "") for r in compiles]
    length = max(
        (len(p) - 1 for p in policies if p.startswith("u")), default=7
    )
    if length not in (7, 8):
        raise ValueError(
            f"Unsupported policy width {length}; supply a schema."
        )
    groups = EIGHT_GROUPS if length == 8 else SEVEN_GROUPS
    configs = [
        analyse_config(
            config,
            entries,
            compile_config[config],
            groups,
            include_observations,
        )
        for config, entries in sorted(by_config.items())
        if any(r.get("task") == "solve" for r in entries)
    ]
    wave_summary = {}
    for wave in sorted(
        {r.get("wave", "") for r in rows if r.get("task") == "solve"}
    ):
        solves = [
            r
            for r in rows
            if r.get("task") == "solve" and r.get("wave", "") == wave
        ]
        warm = [r for r in solves if r.get("warm")]
        timed = [r for r in solves if not r.get("warm")]
        waves = [r.get("geometry", {}).get("waves") for r in warm]
        waves = [value for value in waves if value is not None]
        wave_summary[wave] = dict(
            configs=len({config_key(r) for r in solves}),
            warm_rows=len(warm),
            timing_rows=len(timed),
            minimum_occupancy_waves=min(waves) if waves else None,
            warm_rows_below_two_waves=sum(value < 2 for value in waves),
            timings_below_20ms=sum(r["kernel_ms"] < 20 for r in timed),
        )
    fixed_ratios = [
        r["best_five_full"]["ratio_to_unrestricted_best"]
        for r in configs
        if r["best_five_full"]
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "unroll_bank_audit",
        "provenance": {
            "records": records_provenance,
            "compiles": compiles_provenance,
            "analysis_source": str(Path(__file__).resolve()),
            "analysis_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "method": "min eligible complete-cohort timings / min eligible "
        "all-full timing in same wave and block; aliases inherit "
        "identical cubin/source/compiler observations without becoming "
        "independent replicates",
        "limitations": [
            "Cross-wave ratios share nominal reference policy but are not "
            "paired cross-wave timings; runtime drift remains observable.",
            "Compile rows are joined by policy within this bank. A repeated "
            "policy with different source/cubin identities needs cohort review.",
            "An inherited cubin timing is evidence of code identity, not a "
            "measurement of that policy in another source or launch geometry.",
            "Sample minima can select noise; the duplicate all-full rows "
            "remain visible and no significance threshold is invented.",
        ],
        "record_tasks": counter_dict(r.get("task") for r in rows),
        "duplicate_record_keys": len(rows) - len({r.get("key") for r in rows}),
        "compile_statuses": counter_dict(r.get("status") for r in compiles),
        "compile_source_hashes": counter_dict(
            r.get("source_hash") for r in compiles
        ),
        "waves": wave_summary,
        "ranking_eligible_configs": sum(
            c["best_observed"] is not None for c in configs
        ),
        "ranking_ineligible_configs": [
            {"system": c["system"], "algo": c["algo"]}
            for c in configs
            if c["best_observed"] is None
        ],
        "numerical_checks": numerical_summary(rows),
        "five_full_counts": {
            "configs": len(fixed_ratios),
            "same_measured_best": sum(value == 1 for value in fixed_ratios),
            "within_5_percent": sum(value <= 1.05 for value in fixed_ratios),
            "within_10_percent": sum(value <= 1.10 for value in fixed_ratios),
            "interpretation": "Descriptive counts, not fitted thresholds.",
        },
        "configs": configs,
    }
    if include_history:
        result["history"] = history_audit(directory, rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--observations", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyse_bank(args.bank, args.history, args.observations)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
