"""Run the same Theseus qualification protocol on two local checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_check import run


EXPECTED_STATUS = "REFERENCE_SYMMETRY_ATLAS_AND_JSPACE_PROXY_PASS"


def summarize(report: dict[str, object]) -> dict[str, object]:
    symmetry_cases = report["symmetry_cases"]
    controls = report["non_symmetry_cases"]
    return {
        "model_id": report["checkpoint"]["model_id"],
        "revision": report["checkpoint"]["revision"],
        "checkpoint_digest": report["checkpoint"]["checkpoint_digest"],
        "weights_loaded": report["checkpoint"]["weights_loaded"],
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
        "atlas_entry_count": len(report["causal_atlas"]["mlp_channels"])
        + len(report["causal_atlas"]["attention_heads"]),
        "jspace_status": report["jspace_proxy"]["status"],
        "jspace_raw_coordinate_change": report["jspace_proxy"]["raw_coordinate_change_observed"],
        "jspace_aligned_stability": report["jspace_proxy"]["permutation_stability"],
        "jspace_pruning_divergence": report["jspace_proxy"]["pruning_divergence_observed"],
        "status": report["status"],
    }


def run_replication(gpt2_dir: Path, distilgpt2_dir: Path) -> dict[str, object]:
    reports = [
        run(gpt2_dir, "gpt2"),
        run(distilgpt2_dir, "distilgpt2"),
    ]
    summaries = [summarize(report) for report in reports]
    same_probe_contract = len({summary["probe_set_digest"] for summary in summaries}) == 1
    both_pass = all(summary["status"] == EXPECTED_STATUS for summary in summaries)
    return {
        "status": "CROSS_CHECKPOINT_REPLICATION_PASS" if both_pass and same_probe_contract else "INCONCLUSIVE",
        "claim_ceiling": "LOCAL_GPT2_FAMILY_REPLICATION_OBSERVATION_ONLY",
        "same_probe_contract": same_probe_contract,
        "checkpoint_count": len(summaries),
        "checkpoints": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt2-dir", required=True, type=Path)
    parser.add_argument("--distilgpt2-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_replication(args.gpt2_dir, args.distilgpt2_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
