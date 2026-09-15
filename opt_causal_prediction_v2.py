"""Tie-aware v2 causal protocol for the OPT-125M Theseus experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opt_causal_prediction_check import (
    CAUSAL_PREDICTION_WIN_RATE_MIN,
    PREDICTION_RELATIVE_L2_MAX,
    RESCUE_RELATIVE_L2_MAX,
    run as run_strict_v1,
)


PROTOCOL_ID = "opt125m-causal-prediction-v2"
CLAIM_CEILING = "LOCAL_OPT_CAUSAL_PREDICTION_AND_RESCUE_V2_OBSERVATION_ONLY"


def _prediction_gate(summary: dict[str, object]) -> bool:
    tie_aware = summary["canonical_beats_random_tie_aware"]
    return (
        float(summary["canonical"]["mean_relative_l2"]) < PREDICTION_RELATIVE_L2_MAX
        and float(summary["canonical_beats_raw"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
        and float(tie_aware["canonical_tie_aware_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
    )


def run(model_dir: Path, model_id: str = "opt-125m") -> dict[str, object]:
    report = run_strict_v1(model_dir, model_id, include_tie_aware=True)
    prediction = report["prediction"]
    mlp_prediction = prediction["mlp"]
    attention_prediction = prediction["attention_head"]
    prediction_pass = _prediction_gate(mlp_prediction) and _prediction_gate(attention_prediction)
    patch = report["activation_patching"]
    patch_pass = (
        float(patch["result"]["aligned_canonical"]["mean_relative_l2"]) < PREDICTION_RELATIVE_L2_MAX
        and float(patch["result"]["canonical_beats_raw"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
    )
    rescue = report["activation_rescue"]
    rescue_pass = (
        float(rescue["result"]["aligned_canonical"]["mean_relative_l2"]) < RESCUE_RELATIVE_L2_MAX
        and float(rescue["result"]["canonical_beats_raw"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
    )
    protocol = dict(report["pre_registered_protocol"])
    protocol.update({
        "protocol_version": "v2",
        "random_control_statistic": "tie_half_credit",
        "random_control_scoring": "1.0 strict win, 0.5 exact tie, 0.0 strict loss",
        "strict_random_control_retained": True,
        "canonical_tie_aware_win_rate_min": CAUSAL_PREDICTION_WIN_RATE_MIN,
    })
    report["status"] = (
        "OPT_CAUSAL_PREDICTION_AND_RESCUE_V2_PASS"
        if prediction_pass and patch_pass and rescue_pass
        else "INCONCLUSIVE"
    )
    report["claim_ceiling"] = CLAIM_CEILING
    report["protocol_id"] = PROTOCOL_ID
    report["pre_registered_protocol"] = protocol
    prediction["pass"] = prediction_pass
    report["decision"] = {
        "prediction_pass": prediction_pass,
        "activation_patching_pass": patch_pass,
        "activation_rescue_pass": rescue_pass,
        "strict_attention_random_control": attention_prediction["canonical_beats_random"],
        "tie_aware_attention_random_control": attention_prediction["canonical_beats_random_tie_aware"],
        "tie_aware_control_retrofit": False,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="opt-125m")
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir, args.model_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
