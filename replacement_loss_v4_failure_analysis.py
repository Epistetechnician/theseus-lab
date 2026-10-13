"""Frozen post-hoc aggregate diagnosis for replacement-loss-v4.

State slice: theseus-lab-sandbox-v1.

This reads only digest-bound v4 arrays and writes aggregate metrics. Prompt
text, IDs, per-row values, activations, and logits are never emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v4-adversarial-failure-analysis-v1"
V4_PACKET_SHA256 = "1006d4a861aa442a18307fb9e1e6e5ea6429df57b0c9fd57e2d766bdb4822be5"
V4_PRIMARY_REPORT_SHA256 = "b176c567b8a06790e38a5639c7e03ab3d3a933febaa25e69212eff9078be54b2"
V4_REPLICA_REPORT_SHA256 = "a22e0eb28ca546e33340ad5a6c9e7cd8f86ee91aeec22d0e7365ede741cb18dd"
V4_REPLICATION_RECEIPT_SHA256 = "31eb9602a3b0a38a02b9ecc62b3a8082409cdd48a434ae594038fc05de78300b"
REPO_ROOT = Path(__file__).resolve().parent
PLAN_RELATIVE = Path("docs/replacement-loss-v4-failure-analysis.md")
SCRIPT_RELATIVE = Path("replacement_loss_v4_failure_analysis.py")
SEEDS = (20260927, 20260928, 20260929)
ARMS = ("ordinary_only", "compositional_augmented")
LOGIT_GUARD = 0.05
MAX_GUARD_VIOLATIONS_FOR_TAIL_CLASSIFICATION = 2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: dict[str, Any]) -> str:
    material = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _outside_repo(path: Path) -> None:
    resolved = path.resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError("diagnostic artifacts must remain outside the repository")


def _input_rows(v4_root: Path) -> list[dict[str, str]]:
    primary = _read_json(v4_root / "replacement-loss-v4-report.json")
    replica_root = Path(str(v4_root) + "-replicator")
    replica = _read_json(replica_root / "replacement-loss-v4-report.json")
    receipt = _read_json(replica_root / "replication-report.json")
    if primary.get("v4_packet_sha256") != V4_PACKET_SHA256:
        raise ValueError("primary v4 report is not bound to the declared packet")
    if replica.get("v4_packet_sha256") != V4_PACKET_SHA256:
        raise ValueError("replica v4 report is not bound to the declared packet")
    if receipt.get("assurance_level") != "REPRODUCED_LOCAL":
        raise ValueError("v4 clean-runtime reproduction is not recorded")
    if receipt.get("prediction_array_digests_match") is not True or receipt.get("outcome_array_digests_match") is not True:
        raise ValueError("v4 replication did not match prediction and outcome digest sets")
    if receipt.get("v4_packet_sha256") != V4_PACKET_SHA256:
        raise ValueError("v4 replication receipt is not bound to the declared packet")
    if primary.get("prediction_arrays") != replica.get("prediction_arrays"):
        raise ValueError("v4 primary and replica prediction reports differ")
    if primary.get("outcome_arrays") != replica.get("outcome_arrays"):
        raise ValueError("v4 primary and replica outcome reports differ")
    if _sha(v4_root / "replacement-loss-v4-report.json") != V4_PRIMARY_REPORT_SHA256:
        raise ValueError("v4 primary report bytes differ from the declared input")
    if _sha(replica_root / "replacement-loss-v4-report.json") != V4_REPLICA_REPORT_SHA256:
        raise ValueError("v4 replica report bytes differ from the declared input")
    if _sha(replica_root / "replication-report.json") != V4_REPLICATION_RECEIPT_SHA256:
        raise ValueError("v4 replication receipt differs from the declared input")
    rows = [
        {
            "path": "replacement-loss-v4-report.json",
            "sha256": _sha(v4_root / "replacement-loss-v4-report.json"),
        },
        {
            "path": "../theseus-lab-sandbox-v1-replacement-loss-v4-replicator/replacement-loss-v4-report.json",
            "sha256": _sha(replica_root / "replacement-loss-v4-report.json"),
        },
        {
            "path": "../theseus-lab-sandbox-v1-replacement-loss-v4-replicator/replication-report.json",
            "sha256": _sha(replica_root / "replication-report.json"),
        },
    ]
    expected = {
        "predictions": {int(item["seed"]): str(item["sha256"]) for item in primary["prediction_arrays"]},
        "outcomes": {int(item["seed"]): str(item["sha256"]) for item in primary["outcome_arrays"]},
    }
    for seed in SEEDS:
        for kind in ("predictions", "outcomes"):
            relative = f"{kind}/{f'seed-{seed}.npz'}"
            path = v4_root / relative
            digest = _sha(path)
            if digest != expected[kind][seed]:
                raise ValueError(f"v4 {kind} digest mismatch for seed {seed}")
            rows.append({"path": relative, "sha256": digest})
    return rows


def freeze(v4_root: Path, analysis_root: Path) -> dict[str, Any]:
    v4_root = v4_root.resolve()
    analysis_root = analysis_root.resolve()
    _outside_repo(analysis_root)
    if analysis_root.exists() and any(analysis_root.iterdir()):
        raise FileExistsError("failure-analysis custody must be new or empty at freeze")
    v4_manifest = _read_json(v4_root / "frozen-packet-manifest.json")
    if v4_manifest.get("packet_sha256") != V4_PACKET_SHA256:
        raise ValueError("v4 packet manifest digest mismatch")
    inputs = _input_rows(v4_root)
    packet_source = analysis_root / "frozen-packet"
    packet_rows = []
    for relative in (PLAN_RELATIVE, SCRIPT_RELATIVE):
        source = REPO_ROOT / relative
        target = packet_source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        packet_rows.append({"path": relative.as_posix(), "sha256": _sha(source), "bytes": source.stat().st_size})
    manifest: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": "FROZEN_BEFORE_ROW_LEVEL_V4_ARRAY_READ",
        "v4_packet_sha256": V4_PACKET_SHA256,
        "v4_outcome_status": "FAIL",
        "v4_reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "analysis_packet_files": packet_rows,
        "input_files": inputs,
        "metrics": {
            "normalized_activation_error": "l2(candidate-original)/max(l2(original),1e-12)",
            "normalized_next_logit_error": "l2(candidate-original)/max(l2(original),1e-12)",
            "summaries": ["median", "q90", "maximum", "count_above_0.05"],
            "spearman_pairs": ["adversarial_activation_error", "adversarial_next_logit_error"],
            "paired_arm_delta": "compositional_augmented minus ordinary_only median; fraction paired rows improved",
            "upper_tail_concentrated_if_violations_at_most": MAX_GUARD_VIOLATIONS_FOR_TAIL_CLASSIFICATION,
        },
        "scope": "three v4 seeds, 20 adversarial rows per seed, descriptive only",
        "claim_ceiling": "POSTHOC_LOCAL_GPT2_V4_FAILURE_DIAGNOSIS",
        "nonclaims": ["causality", "mechanistic understanding", "architecture generality", "independent science", "alignment", "consciousness", "production authorization"],
    }
    manifest["analysis_packet_sha256"] = _canonical_digest(manifest)
    _write_json(analysis_root / "freeze.json", manifest)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": manifest["freeze_state"],
        "analysis_packet_sha256": manifest["analysis_packet_sha256"],
        "input_file_count": len(inputs),
        "analysis_root": str(analysis_root),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    a, b = _rank(left), _rank(right)
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def _validate_frozen_inputs(v4_root: Path, analysis_root: Path) -> dict[str, Any]:
    manifest = _read_json(analysis_root / "freeze.json")
    digest = manifest.get("analysis_packet_sha256")
    if digest != _canonical_digest({key: value for key, value in manifest.items() if key != "analysis_packet_sha256"}):
        raise ValueError("failure-analysis freeze digest mismatch")
    if manifest.get("v4_packet_sha256") != V4_PACKET_SHA256:
        raise ValueError("failure-analysis freeze targets a different v4 packet")
    for row in manifest["analysis_packet_files"]:
        relative = Path(str(row["path"]))
        archived = analysis_root / "frozen-packet" / relative
        if _sha(archived) != row["sha256"]:
            raise ValueError(f"frozen analysis packet changed: {relative}")
    for row in manifest["input_files"]:
        rel = Path(str(row["path"]))
        if rel.parts[0] == "..":
            path = (v4_root / rel).resolve()
        else:
            path = v4_root / rel
        if _sha(path) != row["sha256"]:
            raise ValueError(f"frozen v4 input changed: {row['path']}")
    script_row = next(row for row in manifest["analysis_packet_files"] if row["path"] == SCRIPT_RELATIVE.as_posix())
    running_script = Path(__file__).resolve()
    if _sha(running_script) != script_row["sha256"]:
        raise ValueError("running diagnostic source does not match the frozen code")
    return manifest


def run(v4_root: Path, analysis_root: Path, output_root: Path) -> dict[str, Any]:
    v4_root = v4_root.resolve()
    analysis_root = analysis_root.resolve()
    output_root = output_root.resolve()
    _outside_repo(output_root)
    manifest = _validate_frozen_inputs(v4_root, analysis_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds: list[dict[str, Any]] = []
    for seed in SEEDS:
        with np.load(v4_root / "predictions" / f"seed-{seed}.npz", allow_pickle=False) as data:
            predictions = {name: data[name] for name in data.files}
        with np.load(v4_root / "outcomes" / f"seed-{seed}.npz", allow_pickle=False) as data:
            outcomes = {name: data[name] for name in data.files}
        if not np.array_equal(predictions["example_ids"], outcomes["example_ids"]):
            raise ValueError(f"prediction and outcome row order mismatch for seed {seed}")
        strata = predictions["strata"].astype(str)
        adv = strata == "adversarial"
        if int(np.sum(adv)) != 20:
            raise ValueError(f"expected exactly 20 adversarial rows for seed {seed}")
        original_activation = outcomes["original_module_activation"].astype(np.float64)
        original_logits = outcomes["original_next_logits"].astype(np.float64)
        denominator_activation = np.maximum(np.linalg.norm(original_activation, axis=1), 1e-12)
        denominator_logits = np.maximum(np.linalg.norm(original_logits, axis=1), 1e-12)
        per_arm: dict[str, Any] = {}
        row_metric_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm in ARMS:
            candidate_activation = predictions[f"{arm}_activation"].astype(np.float64)
            candidate_logits = predictions[f"{arm}_next_logits"].astype(np.float64)
            activation_error = np.linalg.norm(candidate_activation - original_activation, axis=1) / denominator_activation
            logit_error = np.linalg.norm(candidate_logits - original_logits, axis=1) / denominator_logits
            adv_activation = activation_error[adv]
            adv_logits = logit_error[adv]
            violations = int(np.sum(adv_logits > LOGIT_GUARD))
            per_arm[arm] = {
                "adversarial_activation_error": _summarize(adv_activation),
                "adversarial_next_logit_error": _summarize(adv_logits),
                "adversarial_logit_guard_violations": violations,
                "adversarial_activation_logit_spearman": _spearman(adv_activation, adv_logits),
                "failure_shape": "UPPER_TAIL_CONCENTRATED" if violations <= MAX_GUARD_VIOLATIONS_FOR_TAIL_CLASSIFICATION else "BROAD_GUARD_MISS",
            }
            row_metric_values[arm] = (adv_activation, adv_logits)
        treatment_activation, treatment_logits = row_metric_values["compositional_augmented"]
        control_activation, control_logits = row_metric_values["ordinary_only"]
        seeds.append({
            "seed": seed,
            "adversarial_rows": 20,
            "arms": per_arm,
            "paired_treatment_minus_control": {
                "activation_error_median_delta": float(np.median(treatment_activation) - np.median(control_activation)),
                "activation_error_fraction_rows_improved": float(np.mean(treatment_activation < control_activation)),
                "next_logit_error_median_delta": float(np.median(treatment_logits) - np.median(control_logits)),
                "next_logit_error_fraction_rows_improved": float(np.mean(treatment_logits < control_logits)),
            },
        })
    report: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "analysis_packet_sha256": manifest["analysis_packet_sha256"],
        "v4_packet_sha256": V4_PACKET_SHA256,
        "v4_outcome_status": "FAIL",
        "assurance_level": "INTERNAL_PASS",
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "reproducibility_status": "PENDING_CLEAN_RUNTIME_REPLAY",
        "scope": "post-hoc descriptive analysis of 3 GPT-2 seeds; 20 adversarial rows per seed",
        "seed_results": seeds,
        "claim_ceiling": "POSTHOC_LOCAL_GPT2_V4_FAILURE_DIAGNOSIS",
        "data_and_artifact_custody": "v4 arrays remain in external owner custody; this report contains aggregates only",
        "known_limitations": ["post-hoc", "three seeds from one checkpoint", "synthetic challenge generator", "prompt rows are not independent trials", "rank association is not causal"],
        "next_blocked_authority_gate": "new preregistered intervention is required for causal claims; external reviewer remains separately gated",
        "monetary_cost": "NOT_MEASURED",
        "energy": "NOT_MEASURED",
    }
    report["analysis_report_sha256"] = _canonical_digest(report)
    _write_json(output_root / "analysis-report.json", report)
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "analysis_packet_sha256": manifest["analysis_packet_sha256"],
        "analysis_report_sha256": report["analysis_report_sha256"],
        "runtime": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "prefix": sys.prefix,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "input_file_count": len(manifest["input_files"]),
    }
    _write_json(output_root / "execution-receipt.json", receipt)
    return {
        "protocol_id": PROTOCOL_ID,
        "analysis_packet_sha256": manifest["analysis_packet_sha256"],
        "analysis_report_sha256": report["analysis_report_sha256"],
        "seed_count": len(seeds),
        "output_root": str(output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze or execute the v4 aggregate failure diagnosis")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--v4-custody", type=Path, required=True)
    freeze_parser.add_argument("--analysis-root", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--v4-custody", type=Path, required=True)
    run_parser.add_argument("--analysis-root", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.v4_custody, args.analysis_root) if args.command == "freeze" else run(args.v4_custody, args.analysis_root, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
