"""Run the same Theseus qualification protocol across local checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_check import run
from opt_reference_check import run as run_opt
from smollm_reference_check import run as run_smollm
from smollm2_reference_check import run as run_smollm2


EXPECTED_STATUS = "REFERENCE_SYMMETRY_ATLAS_AND_JSPACE_PROXY_PASS"


def summarize(report: dict[str, object]) -> dict[str, object]:
    symmetry_cases = report["symmetry_cases"]
    controls = report["non_symmetry_cases"]
    return {
        "model_id": report["checkpoint"]["model_id"],
        "revision": report["checkpoint"]["revision"],
        "checkpoint_digest": report["checkpoint"]["checkpoint_digest"],
        "weights_loaded": report["checkpoint"]["weights_loaded"],
        "probe_suite_name": report.get("probe_suite_name", "legacy_shared_v1"),
        "probe_set_digest": report["probe_set_digest"],
        "reference_pass": report["reference"]["pass"],
        "symmetry_case_count": len(symmetry_cases),
        "symmetry_cases_passed": sum(case["pass"] for case in symmetry_cases),
        "blind_mappings_recovered": sum(
            case["identity_criterion"]["blind_mapping_match"]
            for case in symmetry_cases
        ),
        "canonical_symmetry_matches": sum(
            case["identity_criterion"]["canonical_digest_match"]
            for case in symmetry_cases
        ),
        "control_case_count": len(controls),
        "controls_canonical_differences": sum(
            case["identity_criterion"]["canonical_digest_diff"]
            for case in controls
        ),
        "controls_behavior_changed": report["non_symmetry_summary"]["cases_with_behavior_change"],
        "atlas_entry_count": sum(
            len(entries) for entries in report["causal_atlas"].values()
        ),
        "jspace_status": report["jspace_proxy"]["status"],
        "jspace_raw_coordinate_change": report["jspace_proxy"]["raw_coordinate_change_observed"],
        "jspace_aligned_stability": report["jspace_proxy"]["permutation_stability"],
        "jspace_pruning_divergence": report["jspace_proxy"]["pruning_divergence_observed"],
        "status": report["status"],
    }


def run_replication(
    gpt2_dir: Path,
    distilgpt2_dir: Path,
    opt_dir: Path | None = None,
    smollm_dir: Path | None = None,
    smollm2_dir: Path | None = None,
) -> dict[str, object]:
    reports = [
        run(gpt2_dir, "gpt2"),
        run(distilgpt2_dir, "distilgpt2"),
    ]
    if opt_dir is not None:
        reports.append(run_opt(opt_dir, "opt-125m"))
    if smollm_dir is not None:
        reports.append(run_smollm(smollm_dir, "smollm-135m"))
    if smollm2_dir is not None:
        reports.append(run_smollm2(smollm2_dir, "smollm2-135m"))
    summaries = [summarize(report) for report in reports]
    same_probe_contract = len({summary["probe_set_digest"] for summary in summaries}) == 1
    both_pass = all(summary["status"] == EXPECTED_STATUS for summary in summaries)
    shared_summaries = [
        summary for summary in summaries
        if summary["model_id"] != "smollm2-135m"
    ]
    shared_probe_contract = len({
        summary["probe_set_digest"] for summary in shared_summaries
    }) == 1
    shared_digests = {summary["probe_set_digest"] for summary in shared_summaries}
    smollm2_summary = next(
        (summary for summary in summaries if summary["model_id"] == "smollm2-135m"),
        None,
    )
    fresh_holdout_contract = bool(
        smollm2_dir is not None
        and smollm2_summary is not None
        and smollm2_summary["probe_suite_name"] == "smollm2_fresh_v1"
        and smollm2_summary["probe_set_digest"] not in shared_digests
    )
    contract_valid = shared_probe_contract and (
        smollm2_dir is None or fresh_holdout_contract
    )
    probe_contract_groups: dict[str, list[str]] = {}
    for summary in summaries:
        probe_contract_groups.setdefault(summary["probe_set_digest"], []).append(
            summary["model_id"]
        )
    architecture_scope = (
        "cross_topology_fresh_holdout"
        if smollm2_dir is not None
        else "cross_topology"
        if smollm_dir is not None
        else "cross_architecture"
        if opt_dir is not None
        else "gpt2_family"
    )
    return {
        "status": (
            "CROSS_TOPOLOGY_FRESH_HOLDOUT_REPLICATION_PASS"
            if both_pass and contract_valid and smollm2_dir is not None
            else "CROSS_TOPOLOGY_REPLICATION_PASS"
            if both_pass and contract_valid and smollm_dir is not None
            else "CROSS_ARCHITECTURE_REPLICATION_PASS"
            if both_pass and contract_valid and opt_dir is not None
            else "CROSS_CHECKPOINT_REPLICATION_PASS"
            if both_pass and contract_valid
            else "INCONCLUSIVE"
        ),
        "claim_ceiling": (
            "LOCAL_CROSS_TOPOLOGY_FRESH_HOLDOUT_OBSERVATION_ONLY"
            if smollm2_dir is not None
            else "LOCAL_CROSS_TOPOLOGY_REPLICATION_OBSERVATION_ONLY"
            if smollm_dir is not None
            else "LOCAL_CROSS_ARCHITECTURE_REPLICATION_OBSERVATION_ONLY"
            if opt_dir is not None
            else "LOCAL_GPT2_FAMILY_REPLICATION_OBSERVATION_ONLY"
        ),
        "architecture_scope": architecture_scope,
        "same_probe_contract": same_probe_contract,
        "shared_probe_contract": shared_probe_contract,
        "fresh_holdout_contract": fresh_holdout_contract,
        "probe_contract_groups": probe_contract_groups,
        "checkpoint_count": len(summaries),
        "checkpoints": summaries,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt2-dir", required=True, type=Path)
    parser.add_argument("--distilgpt2-dir", required=True, type=Path)
    parser.add_argument("--opt-dir", default=None, type=Path)
    parser.add_argument("--smollm-dir", default=None, type=Path)
    parser.add_argument("--smollm2-dir", default=None, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        run_replication(
            args.gpt2_dir,
            args.distilgpt2_dir,
            args.opt_dir,
            args.smollm_dir,
            args.smollm2_dir,
        ),
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
