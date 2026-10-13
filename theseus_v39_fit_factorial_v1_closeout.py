"""Independent numeric and artifact closeout for the first-MLP factorial.

State slices: theseus-lab-sandbox-v1, theseus-v39-first-mlp-fit-factorial-v1,
global-experiment-assurance-lifecycle-v1. Reads only digest-bound external
artifacts and emits a scalar validation receipt.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-v39-first-mlp-fit-factorial-v1")
PROTOCOL = "theseus-v39-first-mlp-fit-factorial-v1"
MEMBERS = (
    "theseus_v39_fit_factorial_v1.py",
    "theseus_v39_fit_factorial_v1_closeout.py",
    "tests/test_v39_fit_factorial_v1.py",
    "docs/v39-first-mlp-fit-factorial-v1.md",
    "replacement_loss_v39_runner.py",
    "replacement_loss_v38_runner.py",
    "replacement_loss_v37r1_runner.py",
    "replacement_loss_v36_runner.py",
    "replacement_loss_v36_core.py",
    "replacement_loss_v36_fixtures.py",
    "requirements-v36.lock.txt",
    "docs/replacement-loss-v39-freeze.json",
    "docs/replacement-loss-v39-results.json",
    "docs/replacement-loss-v39-all-token-ladder.md",
    "docs/v39-first-mlp-diagnosis-v1-freeze.json",
    "docs/v39-first-mlp-diagnosis-v1-results.json",
    "docs/v39-training-probe-replay-v1-freeze.json",
    "docs/v39-training-probe-replay-v1-results.json",
)
SEEDS = (37101, 37102, 37103)
STRATA = ("ordinary", "input_composition", "interchange_composition", "adversarial")
CONDITIONS = ("cap2_updates1024", "cap2_updates4096", "cap4_updates1024", "cap4_updates4096")
ARRAY_ROOT = ROOT
LIMITS = {"output_nrmse_max": 0.10, "top1_agreement_min": 0.95, "nll_increase_max": 0.03,
          "activation_nrmse_max": 0.15, "rescue_nrmse_max": 0.10,
          "resolved_fraction_min": 0.90, "effect_median_max": 0.20,
          "effect_p90_max": 0.50, "effect_cosine_min": 0.90}


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        meta = json.dumps([key, array.dtype.str, list(array.shape)], sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode()
        digest.update(meta)
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def nrmse(actual: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("closeout output shape/value")
    return float(np.sqrt(np.mean(np.square(a - b)) / max(float(np.mean(np.square(b))), 1e-8)))


def effect_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    a = np.asarray(actual)
    b = np.asarray(reference)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("closeout effect shape/value")
    mask = np.linalg.norm(b, axis=1) >= 1e-4
    if not mask.any():
        return {"resolved_fraction": 0.0, "median_error": None, "p90_error": None,
                "median_cosine": None, "passed": False}
    left, right = a[mask].astype(np.float64), b[mask].astype(np.float64)
    relative = np.linalg.norm(left - right, axis=1) / np.maximum(np.linalg.norm(right, axis=1), 1e-6)
    cosine = np.sum(left * right, axis=1) / np.maximum(
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-12
    )
    result = {"resolved_fraction": float(mask.mean()), "median_error": float(np.median(relative)),
              "p90_error": float(np.quantile(relative, 0.9)),
              "median_cosine": float(np.median(cosine))}
    result["passed"] = bool(result["resolved_fraction"] >= LIMITS["resolved_fraction_min"]
                            and result["median_error"] <= LIMITS["effect_median_max"]
                            and result["p90_error"] <= LIMITS["effect_p90_max"]
                            and result["median_cosine"] >= LIMITS["effect_cosine_min"])
    return result


def score(prediction: dict, oracle: dict, rescues: dict) -> dict:
    result = {}
    for stratum in STRATA:
        prefix = stratum + "/"
        pred, truth = prediction[prefix + "logits"], oracle[prefix + "logits"]
        value = {
            "output_nrmse": nrmse(pred, truth),
            "top1_agreement": float(np.mean(pred.argmax(1) == truth.argmax(1))),
            "nll_increase": float(np.mean(prediction[prefix + "nll"] - oracle[prefix + "nll"])),
            "effects": {"0": effect_metrics(prediction[prefix + "effect/0"], oracle[prefix + "effect/0"])},
            "joint_effect": effect_metrics(prediction[prefix + "joint_effect"], oracle[prefix + "joint_effect"]),
            "activation": {"0": nrmse(prediction[prefix + "activation/0"], oracle[prefix + "activation/0"])},
            "rescue": {"0": nrmse(rescues[stratum + "/0"], truth)},
            "rescue_prediction": {"0": nrmse(pred, rescues[stratum + "/0"])},
        }
        value["passed"] = bool(
            value["output_nrmse"] <= LIMITS["output_nrmse_max"]
            and value["top1_agreement"] >= LIMITS["top1_agreement_min"]
            and value["nll_increase"] <= LIMITS["nll_increase_max"]
            and value["joint_effect"]["passed"]
            and value["effects"]["0"]["passed"]
            and value["activation"]["0"] <= LIMITS["activation_nrmse_max"]
            and value["rescue"]["0"] <= LIMITS["rescue_nrmse_max"]
            and value["rescue_prediction"]["0"] <= LIMITS["rescue_nrmse_max"]
        )
        result[stratum] = value
    return result


def recursive_equal(left, right, path="root") -> list[str]:
    errors = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return [path + ":keys"]
        for key in sorted(left):
            errors.extend(recursive_equal(left[key], right[key], path + "/" + key))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path + ":length"]
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            errors.extend(recursive_equal(a, b, path + f"/{index}"))
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12):
            errors.append(path + ":number")
    elif left != right:
        errors.append(path + ":value")
    return errors


def verify_freeze() -> dict:
    value = json.loads((REPO / "docs/v39-first-mlp-fit-factorial-v1-freeze.json").read_text())
    actual = []
    for member in MEMBERS:
        path = REPO / member
        actual.append({"path": member, "bytes": path.stat().st_size, "sha256": file_sha(path)})
    canonical = json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if value["members"] != actual or hashlib.sha256(canonical).hexdigest() != value["packet_sha256"]:
        raise ValueError("frozen source members differ")
    return value


def verify_probe_formulas(result: dict) -> int:
    checks = 0
    for case, cells in result["factorial_cells"].items():
        for condition, values in cells.items():
            for field in ("training_probe", "fresh_probe"):
                sums = values[field]
                expected = math.sqrt(sums["squared_error_sum"] / max(
                    sums["squared_reference_sum"], sums["element_count"] * 1e-8
                ))
                if not math.isclose(expected, sums["nrmse"], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"probe formula differs: {case}/{condition}/{field}")
                checks += 1
    return checks


def verify_role_outputs(role: str, result: dict) -> tuple[int, int]:
    root = ROOT / role
    cases_checked = arrays_checked = 0
    for seed in SEEDS:
        case = f"gpt2_gelu-{seed}"
        oracle = load_npz(root / case / "oracle.npz")
        noop = load_npz(root / case / "noop.npz")
        if array_digest(oracle) != array_digest(noop):
            raise ValueError(f"guide no-op differs: {role}/{case}")
        for condition in CONDITIONS:
            prediction = load_npz(root / case / f"{condition}-predictions.npz")
            rescues = load_npz(root / case / f"{condition}-rescues.npz")
            calculated = score(prediction, oracle, rescues)
            recorded = result["heldout"]["seed_cells"][case][condition]["metrics"]
            errors = recursive_equal(calculated, recorded, f"{role}/{case}/{condition}")
            if errors:
                raise ValueError("heldout numeric closeout differs: " + ",".join(errors[:8]))
            arrays_checked += 3
            controls = {
                "zero": load_npz(root / case / "control-zero-predictions.npz"),
                f"random_cap{condition[3]}": load_npz(root / case / f"control-random_cap{condition[3]}-predictions.npz"),
                f"wrong_{condition}": load_npz(root / case / f"control-wrong_{condition}-predictions.npz"),
            }
            recorded_sensitivity = result["heldout"]["seed_cells"][case][condition]["control_sensitivity"]
            for stratum in STRATA:
                target = effect_metrics(prediction[stratum + "/joint_effect"],
                                        oracle[stratum + "/joint_effect"])["median_error"]
                margins = {
                    name: effect_metrics(values[stratum + "/joint_effect"],
                                         oracle[stratum + "/joint_effect"])["median_error"]
                    for name, values in controls.items()
                }
                expected = {"treatment_joint_median_error": target,
                            "control_joint_median_errors": margins,
                            "all_controls_worse_by_margin": target is not None and all(
                                value is not None and value >= target + 0.10 for value in margins.values()
                            )}
                if recursive_equal(expected, recorded_sensitivity[stratum], stratum):
                    raise ValueError("control sensitivity closeout differs")
            arrays_checked += len(controls)
            cases_checked += 1
    return cases_checked, arrays_checked


def verify_factorial(result: dict) -> int:
    per_seed = {}
    for seed in SEEDS:
        case = result["factorial_cells"][f"gpt2_gelu-{seed}"]
        e22 = case["cap2_updates1024"]["training_probe"]["nrmse"]
        e24 = case["cap2_updates4096"]["training_probe"]["nrmse"]
        e42 = case["cap4_updates1024"]["training_probe"]["nrmse"]
        e44 = case["cap4_updates4096"]["training_probe"]["nrmse"]
        opt = [(math.log(e22) - math.log(e24)), (math.log(e42) - math.log(e44))]
        cap = [(math.log(e22) - math.log(e42)), (math.log(e24) - math.log(e44))]
        per_seed[str(seed)] = (sum(opt) / 2, sum(cap) / 2, opt[0] - opt[1])
    for index, name in ((0, "optimization_budget_main_effect"), (1, "capacity_main_effect")):
        logs = [row[index] for row in per_seed.values()]
        ratio = math.exp(-sum(logs) / len(logs))
        recorded = result["factorial"][name]
        if not math.isclose(ratio, recorded["geometric_mean_error_ratio"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("factor main-effect formula differs")
        supported = ratio <= 0.80 and all(value > 0 for value in logs)
        if supported != recorded["supported"]:
            raise ValueError("factor support decision differs")
    return 2


def main() -> dict:
    freeze = verify_freeze()
    primary = json.loads((ROOT / "primary/result.json").read_text())
    replica = json.loads((ROOT / "replica/result.json").read_text())
    keys = ("protocol_id", "state_slice", "revision", "packet_sha256", "claim_ceiling", "scientific_outcome",
            "factorial", "heldout", "training_bindings", "challenge_sha256", "candidate_freeze_sha256",
            "prediction_freeze_sha256", "probe_arrays_sha256", "checkpoint_manifest_sha256", "guide_site_sha256")
    comparisons = {key: primary[key] == replica[key] for key in keys}
    if not all(comparisons.values()):
        raise ValueError("primary and replica do not reconcile")
    for filename in ("candidate-freeze.json", "challenge.json", "prediction-freeze.json"):
        if file_sha(ROOT / "primary" / filename) != file_sha(ROOT / "replica" / filename):
            raise ValueError("frozen binding differs across roles: " + filename)
    probe_checks = verify_probe_formulas(primary) + verify_probe_formulas(replica)
    factorial_checks = verify_factorial(primary) + verify_factorial(replica)
    primary_cases, primary_arrays = verify_role_outputs("primary", primary)
    replica_cases, replica_arrays = verify_role_outputs("replica", replica)
    deterministic_files = 0
    ignored = {"runtime.json", "result.json", "reconciliation.json"}
    for role in ("primary", "replica"):
        paths = {str(path.relative_to(ROOT / role)) for path in (ROOT / role).rglob("*")
                 if path.is_file() and path.name not in ignored}
        if role == "primary":
            primary_paths = paths
        elif paths != primary_paths:
            raise ValueError("role artifact inventories differ")
        deterministic_files += len(paths)
    for relative in primary_paths:
        if file_sha(ROOT / "primary" / relative) != file_sha(ROOT / "replica" / relative):
            raise ValueError("role artifact bytes differ: " + relative)
    validation = {
        "protocol_id": PROTOCOL, "packet_sha256": freeze["packet_sha256"],
        "assurance_level": "REPRODUCED_LOCAL", "reviewer_status": "NOT_SUBMITTED",
        "reproducibility_status": "SAME_OWNER_SECOND_ISOLATED_RUNTIME",
        "primary_replica_checks": comparisons,
        "probe_formula_cells_recomputed": probe_checks,
        "factorial_effect_formulas_recomputed": factorial_checks,
        "heldout_seed_strata_recomputed_per_role": primary_cases,
        "heldout_and_control_arrays_checked_per_role": primary_arrays,
        "deterministic_artifact_files_matched_per_role": deterministic_files,
        "claim_ceiling": freeze["claim_ceiling"],
        "limitations": ["same owner and physical host", "one reused GPT-2 guide", "not architecture-held-out or external review"],
        "next_blocked_authority_gate": "configured independent authority for stronger scientific acceptance; production requires separate operational authorization",
    }
    path = ROOT / "closeout" / "validation.json"
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(json.dumps(validation, sort_keys=True, indent=2) + "\n")
    return validation


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
