"""Frozen local V29 synthetic nonlinear-replacement qualification.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v29-nonlinear-synthetic-qualification-v1, and
global-experiment-assurance-lifecycle-v1. Generated arrays stay in external
owner custody. No model checkpoints or user data are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


STATE_SLICE = "theseus-lab-sandbox-v1"
CONTINUATION_SLICE = "theseus-replacement-loss-v29-nonlinear-synthetic-qualification-v1"
ASSURANCE_SLICE = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = "theseus-replacement-loss-v29-nonlinear-synthetic"
CLAIM_CEILING = "SYNTHETIC_NONLINEAR_REPLACEMENT_METHODS_QUALIFICATION_ONLY"
ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path("/Volumes/PrimaryED/ResearchArtifacts")
PRIMARY = ARTIFACTS / f"{STATE_SLICE}-replacement-loss-v29"
REPLICA = ARTIFACTS / f"{STATE_SLICE}-replacement-loss-v29-replicator"
FREEZE_LOCAL = ROOT / "docs/replacement-loss-v29-freeze.json"
PACKET_FILES = (
    "docs/replacement-loss-v29-nonlinear-synthetic.md",
    "replacement_loss_v29_runner.py",
    "requirements-v29.lock.txt",
)
FAMILIES = ("opt_relu", "gpt2_gelu", "llama_silu_gated")
SEEDS = (20261001, 20261002, 20261003)
DIM = 16
SOURCE_WIDTH = 24
STUDENT_WIDTH = 48
ADAPT_ROWS_PER_STRATUM = 1024
CHALLENGE_ROWS_PER_STRATUM = 128
STRATA = ("ordinary", "compositional", "adversarial")
TRAIN_STEPS = 600
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def _save_array(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return _sha(path)


def _load_array(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        return np.load(stream, allow_pickle=False)


def _external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("V29 experiment data must remain outside the repository")
    return resolved


def _status_digest() -> str:
    try:
        raw = subprocess.check_output(["git", "status", "--porcelain=v1", "-z"], cwd=ROOT)
    except (OSError, subprocess.CalledProcessError):
        raw = b"git-status-unavailable"
    excluded = set(PACKET_FILES) | {"docs/replacement-loss-v29-freeze.json"}
    kept = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        relative = entry[3:].decode("utf-8", "replace")
        if relative not in excluded and not relative.startswith("docs/replacement-loss-v29-"):
            kept.append(entry)
    return hashlib.sha256(b"\0".join(kept)).hexdigest()


def _packet_rows() -> list[dict[str, Any]]:
    rows = []
    for relative in PACKET_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen input missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha(path)})
    return rows


def freeze() -> dict[str, Any]:
    _external(PRIMARY)
    if PRIMARY.exists() or FREEZE_LOCAL.exists():
        raise FileExistsError("V29 freeze is one-shot; preserve any existing packet")
    if not sys.version.startswith("3.12.13") or importlib.metadata.version("numpy") != "2.4.6" or importlib.metadata.version("torch") != "2.6.0":
        raise RuntimeError("V29 freeze requires Python 3.12.13, NumPy 2.4.6, and PyTorch 2.6.0")
    rows = _packet_rows()
    manifest: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_tree_status_sha256_excluding_v29_packet": _status_digest(),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "freeze_state": "FROZEN_BEFORE_SYNTHETIC_ORACLE_OR_ADAPTATION_GENERATION",
        "packet_files": rows,
        "checkpoint_identities_selected": False,
        "oracle_generated": False,
        "challenge_generated": False,
        "budget_usd": 0,
        "resource_budget": {"max_total_wall_seconds": 1800, "max_peak_rss_bytes": 4294967296, "max_external_artifact_bytes": 1073741824},
        "runtime_contract": {"python": "3.12.13", "numpy": "2.4.6", "torch": "2.6.0", "cpu_threads": 1},
        "design": {
            "families": list(FAMILIES), "seeds": list(SEEDS),
            "adapt_rows_per_stratum": ADAPT_ROWS_PER_STRATUM,
            "challenge_rows_per_stratum": CHALLENGE_ROWS_PER_STRATUM,
            "strata": list(STRATA),
            "student": {"input": DIM, "hidden": STUDENT_WIDTH, "output": DIM, "activation": "GELU"},
            "optimizer": {"name": "AdamW", "steps": TRAIN_STEPS, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batching": "full_batch"},
            "controls": ["wrong_module", "random", "no_op", "affine_distillation"],
            "coordinate_permutation_experiment": False,
        },
        "thresholds": {
            "causal_direction_accuracy_min": 0.80, "causal_median_cosine_min": 0.90,
            "causal_median_normalized_error_max": 0.20, "activation_median_cosine_min": 0.90,
            "activation_median_normalized_error_max": 0.20, "behavior_mean_cross_entropy_delta_max": 0.05,
            "behavior_max_normalized_logit_error": 0.05, "all_nine_cases_required": True,
            "treatment_must_beat_each_control_on_median_causal_error": True,
        },
        "claim_ceiling": CLAIM_CEILING,
        "nonclaims": ["trained-transformer mechanism", "architecture-general validity", "independent science", "consciousness", "alignment", "production readiness"],
    }
    manifest["packet_sha256"] = _digest(manifest)
    PRIMARY.mkdir(parents=True, exist_ok=False)
    for row in rows:
        source = ROOT / row["path"]
        target = PRIMARY / "frozen-packet" / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _write_json(PRIMARY / "freeze.json", manifest)
    _write_json(FREEZE_LOCAL, manifest)
    return {"packet_sha256": manifest["packet_sha256"], "packet_file_count": len(rows), "freeze_state": manifest["freeze_state"]}


def verify(root: Path = PRIMARY) -> dict[str, Any]:
    _external(root)
    local = json.loads(FREEZE_LOCAL.read_text())
    external = json.loads((root / "freeze.json").read_text())
    if local != external or local.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V29 local and custody freeze manifests differ")
    if local.get("packet_sha256") != _digest({key: value for key, value in local.items() if key != "packet_sha256"}):
        raise ValueError("V29 freeze digest is invalid")
    for row in local["packet_files"]:
        relative = row["path"]
        if _sha(ROOT / relative) != row["sha256"] or _sha(root / "frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"frozen V29 source changed: {relative}")
    return local


def _verify_runtime() -> None:
    if not sys.version.startswith("3.12.13") or importlib.metadata.version("numpy") != "2.4.6" or importlib.metadata.version("torch") != "2.6.0":
        raise RuntimeError("V29 execution requires the frozen Python, NumPy, and PyTorch versions")


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


class SourceModule(nn.Module):
    def __init__(self, family: str):
        super().__init__()
        self.family = family
        self.left = nn.Linear(DIM, SOURCE_WIDTH)
        self.right = nn.Linear(DIM, SOURCE_WIDTH) if family == "llama_silu_gated" else None
        self.output = nn.Linear(SOURCE_WIDTH, DIM)
        self.activation = nn.ReLU() if family == "opt_relu" else nn.GELU()
        self.gate = nn.SiLU() if family == "llama_silu_gated" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.family == "llama_silu_gated":
            return self.output(self.gate(self.left(x)) * self.right(x))
        return self.output(self.activation(self.left(x)))


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(DIM, STUDENT_WIDTH)
        self.out_proj = nn.Linear(STUDENT_WIDTH, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(torch.nn.functional.gelu(self.in_proj(x)))


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _state_arrays(module: nn.Module, directory: Path) -> list[dict[str, str]]:
    rows = []
    for key, tensor in sorted(module.state_dict().items()):
        filename = key.replace(".", "_") + ".npy"
        digest = _save_array(directory / filename, tensor.detach().cpu().numpy())
        rows.append({"key": key, "file": filename, "sha256": digest})
    return rows


def _load_state(module: nn.Module, directory: Path, rows: list[dict[str, str]]) -> None:
    state = {}
    for row in rows:
        path = directory / row["file"]
        if _sha(path) != row["sha256"]:
            raise ValueError(f"state array digest mismatch: {row['file']}")
        state[row["key"]] = torch.from_numpy(_load_array(path)).float()
    module.load_state_dict(state, strict=True)
    module.eval()


def _rng_for(seed: int, family_index: int, seed_index: int, partition: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, family_index, seed_index, partition]))


def _sample_inputs(rng: np.random.Generator, stratum: str, count: int) -> np.ndarray:
    if stratum == "ordinary":
        return rng.normal(size=(count, DIM)).astype(np.float32)
    if stratum == "compositional":
        parts = rng.normal(size=(count, 2, DIM)).astype(np.float32)
        return ((parts[:, 0] + parts[:, 1]) / np.sqrt(2.0)).astype(np.float32)
    if stratum == "adversarial":
        base = rng.normal(size=(count, DIM)).astype(np.float32)
        sign = np.sign(rng.normal(size=base.shape)).astype(np.float32)
        return base + (0.35 / np.sqrt(DIM)) * sign
    raise ValueError(f"unknown stratum: {stratum}")


def _source(family: str, seed: int) -> SourceModule:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module = SourceModule(family).float().eval()
    return module


def _fit_student(x: np.ndarray, y: np.ndarray, seed: int) -> Student:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        module = Student().float()
    optimizer = torch.optim.AdamW(module.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((module(xt) - yt) ** 2)
        loss.backward()
        optimizer.step()
    module.eval()
    return module


def _fit_affine(x: np.ndarray, y: np.ndarray) -> Affine:
    design = np.concatenate([x.astype(np.float64), np.ones((len(x), 1))], axis=1)
    coef = np.linalg.lstsq(design, y.astype(np.float64), rcond=1e-8)[0]
    module = Affine().float().eval()
    with torch.no_grad():
        module.proj.weight.copy_(torch.from_numpy(coef[:-1].T.astype(np.float32)))
        module.proj.bias.copy_(torch.from_numpy(coef[-1].astype(np.float32)))
    return module


def _forward(module: nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        value = module(torch.from_numpy(x.astype(np.float32))).cpu().numpy()
    if not np.isfinite(value).all():
        raise ValueError("non-finite module output")
    return value


def _prepare(root: Path) -> dict[str, Any]:
    freeze_manifest = verify(root)
    prep = root / "preparation"
    if prep.exists():
        raise FileExistsError("V29 candidate preparation is immutable")
    prep.mkdir()
    case_rows = []
    for family_index, family in enumerate(FAMILIES):
        for seed_index, seed in enumerate(SEEDS):
            case_dir = prep / family / f"seed-{seed_index}"
            case_dir.mkdir(parents=True)
            target = _source(family, seed + family_index * 1000)
            wrong_source = _source(family, seed + 900_000 + family_index * 1000)
            target_rows = _state_arrays(target, case_dir / "source")
            wrong_rows = _state_arrays(wrong_source, case_dir / "wrong-source")
            x_parts, strata_parts = [], []
            for stratum_index, stratum in enumerate(STRATA):
                x_parts.append(_sample_inputs(_rng_for(seed, family_index, seed_index, stratum_index), stratum, ADAPT_ROWS_PER_STRATUM))
                strata_parts.append(np.full(ADAPT_ROWS_PER_STRATUM, stratum))
            adapt_x = np.concatenate(x_parts)
            adapt_strata = np.concatenate(strata_parts)
            with torch.no_grad():
                adapt_y = target(torch.from_numpy(adapt_x)).numpy()
                wrong_y = wrong_source(torch.from_numpy(adapt_x)).numpy()
            x_sha = _save_array(case_dir / "adaptation_x.npy", adapt_x)
            y_sha = _save_array(case_dir / "adaptation_target.npy", adapt_y)
            wrong_y_sha = _save_array(case_dir / "adaptation_wrong_target.npy", wrong_y)
            _save_array(case_dir / "adaptation_strata.npy", adapt_strata)
            treatment = _fit_student(adapt_x, adapt_y, seed + 50_000 + family_index * 100 + seed_index)
            wrong = _fit_student(adapt_x, wrong_y, seed + 60_000 + family_index * 100 + seed_index)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 70_000 + family_index * 100 + seed_index)
                random_student = Student().float().eval()
            affine = _fit_affine(adapt_x, adapt_y)
            module_rows = {}
            for name, module in (("treatment", treatment), ("wrong_module", wrong), ("random", random_student), ("affine_distillation", affine)):
                module_rows[name] = _state_arrays(module, case_dir / name)
            module_rows["no_op"] = []
            candidate_digest = _digest({name: module_rows[name] for name in sorted(module_rows)})
            case_rows.append({
                "family": family, "family_index": family_index, "seed_index": seed_index, "seed": seed,
                "source_state": target_rows, "wrong_source_state": wrong_rows,
                "adaptation_x_sha256": x_sha, "adaptation_target_sha256": y_sha,
                "adaptation_wrong_target_sha256": wrong_y_sha,
                "candidate_modules": module_rows, "candidate_state_bundle_sha256": candidate_digest,
            })
    preparation = {"protocol_id": PROTOCOL_ID, "packet_sha256": freeze_manifest["packet_sha256"], "cases": case_rows, "candidate_preparation_completed": True}
    _write_json(prep / "candidate-preparation.json", preparation)
    freeze_candidates(root)
    return {"case_count": len(case_rows), "candidate_freeze_sha256": _sha(root / "candidate-freeze.json")}


def freeze_candidates(root: Path) -> dict[str, Any]:
    prep = root / "preparation/candidate-preparation.json"
    if not prep.is_file():
        raise FileNotFoundError("candidate preparation is incomplete")
    if (root / "candidate-freeze.json").exists():
        raise FileExistsError("V29 candidate freeze is immutable")
    payload = json.loads(prep.read_text())
    if len(payload.get("cases", [])) != 9:
        raise ValueError("candidate freeze requires exactly nine candidates")
    manifest = {
        "protocol_id": PROTOCOL_ID, "packet_sha256": verify(root)["packet_sha256"],
        "candidate_preparation_sha256": _sha(prep), "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": [{"family": r["family"], "seed_index": r["seed_index"], "candidate_state_bundle_sha256": r["candidate_state_bundle_sha256"]} for r in payload["cases"]],
        "challenge_generation_started_after_candidate_freeze": True,
    }
    manifest["candidate_freeze_sha256"] = _digest(manifest)
    _write_json(root / "candidate-freeze.json", manifest)
    return manifest


def _make_challenges(root: Path) -> dict[str, Any]:
    verify(root)
    freeze_file = root / "candidate-freeze.json"
    if not freeze_file.is_file():
        raise FileNotFoundError("candidates must be frozen before challenge generation")
    if (root / "challenges").exists():
        raise FileExistsError("V29 challenge fixtures are immutable")
    preparation = json.loads((root / "preparation/candidate-preparation.json").read_text())
    cases, commitments = [], []
    for row in preparation["cases"]:
        family, family_index, seed_index = row["family"], row["family_index"], row["seed_index"]
        case_dir = root / "challenges" / family / f"seed-{seed_index}"
        case_dir.mkdir(parents=True)
        secret = os.urandom(32)
        rng = np.random.default_rng(np.random.SeedSequence(np.frombuffer(secret, dtype=np.uint32).astype(np.uint64).tolist()))
        x_parts, delta_parts, strata_parts = [], [], []
        for stratum in STRATA:
            x = _sample_inputs(rng, stratum, CHALLENGE_ROWS_PER_STRATUM)
            delta = rng.normal(size=x.shape).astype(np.float32)
            delta /= np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-12)
            delta *= 0.12 if stratum != "adversarial" else 0.24
            x_parts.append(x.astype(np.float32)); delta_parts.append(delta.astype(np.float32))
            strata_parts.append(np.full(CHALLENGE_ROWS_PER_STRATUM, stratum))
        x_all, delta_all, labels = np.concatenate(x_parts), np.concatenate(delta_parts), np.concatenate(strata_parts)
        adapt_x = _load_array(root / "preparation" / family / f"seed-{seed_index}" / "adaptation_x.npy")
        adapt_rows = {hashlib.sha256(value.tobytes()).hexdigest() for value in adapt_x}
        challenge_rows = {hashlib.sha256(value.tobytes()).hexdigest() for value in x_all}
        if adapt_rows & challenge_rows:
            raise ValueError("adaptation and challenge inputs overlap")
        cases.append({
            "family": family, "seed_index": seed_index,
            "inputs_sha256": _save_array(case_dir / "inputs.npy", x_all),
            "interventions_sha256": _save_array(case_dir / "interventions.npy", delta_all),
            "strata_sha256": _save_array(case_dir / "strata.npy", labels),
            "partition_disjoint": True,
        })
        commitments.append(hashlib.sha256(secret).hexdigest())
    manifest = {
        "protocol_id": PROTOCOL_ID, "packet_sha256": verify(root)["packet_sha256"],
        "candidate_freeze_sha256": _sha(freeze_file), "cases": cases,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "challenge_generation_started_after_candidate_freeze": True,
        "challenge_secret_entropy_commitments": commitments,
    }
    manifest["challenge_manifest_sha256"] = _digest(manifest)
    _write_json(root / "challenge-manifest.json", manifest)
    return {"case_count": len(cases), "challenge_manifest_sha256": manifest["challenge_manifest_sha256"]}


def _load_module(root: Path, row: dict[str, Any], name: str) -> nn.Module:
    directory = root / "preparation" / row["family"] / f"seed-{row['seed_index']}" / name
    module: nn.Module = Affine() if name == "affine_distillation" else Student()
    _load_state(module, directory, row["candidate_modules"][name])
    return module


def _score_effect(predicted: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    obs_norm = np.linalg.norm(observed, axis=1)
    resolvable = obs_norm > 1e-7
    pred_norm = np.linalg.norm(predicted, axis=1)
    dots = np.sum(predicted * observed, axis=1)
    cosine = dots / np.maximum(pred_norm * obs_norm, 1e-20)
    error = np.linalg.norm(predicted - observed, axis=1) / np.maximum(obs_norm, 1e-7)
    if not np.any(resolvable):
        return {"resolvable_count": 0, "direction_accuracy": None, "median_cosine": None, "median_normalized_error": None}
    return {
        "resolvable_count": int(resolvable.sum()),
        "direction_accuracy": float(np.mean(dots[resolvable] > 0)),
        "median_cosine": float(np.median(cosine[resolvable])),
        "median_normalized_error": float(np.median(error[resolvable])),
    }


def _readout(family_index: int) -> np.ndarray:
    rng = np.random.default_rng(18_990 + family_index)
    return rng.normal(size=(DIM, DIM)).astype(np.float32) / np.sqrt(DIM)


def _case_evaluate(root: Path, prep_row: dict[str, Any]) -> dict[str, Any]:
    family, seed_index = prep_row["family"], prep_row["seed_index"]
    fixture = root / "challenges" / family / f"seed-{seed_index}"
    x = _load_array(fixture / "inputs.npy").astype(np.float32)
    delta = _load_array(fixture / "interventions.npy").astype(np.float32)
    strata = _load_array(fixture / "strata.npy").astype(str)
    if not (len(x) == len(delta) == len(strata) == CHALLENGE_ROWS_PER_STRATUM * len(STRATA)):
        raise ValueError("challenge row count differs from the frozen protocol")
    challenge_manifest = json.loads((root / "challenge-manifest.json").read_text())
    challenge_row = next(item for item in challenge_manifest["cases"] if item["family"] == family and item["seed_index"] == seed_index)
    for key, filename in (("inputs_sha256", "inputs.npy"), ("interventions_sha256", "interventions.npy"), ("strata_sha256", "strata.npy")):
        if _sha(fixture / filename) != challenge_row[key]:
            raise ValueError(f"V29 challenge fixture digest mismatch: {filename}")
    if not challenge_row.get("partition_disjoint"):
        raise ValueError("V29 challenge fixture lacks an adaptation-disjointness receipt")
    readout = _readout(prep_row["family_index"])

    # Prediction phase: candidate outputs are persisted before source evaluation.
    predictions = {}
    for name in ("treatment", "wrong_module", "random", "affine_distillation"):
        module = _load_module(root, prep_row, name)
        base = _forward(module, x)
        changed = _forward(module, x + delta)
        predictions[name] = {"base": base, "effect": changed - base, "logits": base @ readout.T}
    predictions["no_op"] = {
        "base": np.zeros((len(x), DIM), np.float32),
        "effect": np.zeros((len(x), DIM), np.float32),
        "logits": np.zeros((len(x), DIM), np.float32),
    }
    pdir = root / "predictions" / family / f"seed-{seed_index}"
    pdir.mkdir(parents=True, exist_ok=False)
    prediction_hashes = {
        name: {key: _save_array(pdir / f"{name}-{key}.npy", value) for key, value in values.items()}
        for name, values in predictions.items()
    }
    receipt = {
        "protocol_id": PROTOCOL_ID, "packet_sha256": verify(root)["packet_sha256"],
        "candidate_freeze_sha256": _sha(root / "candidate-freeze.json"),
        "challenge_inputs_sha256": _sha(fixture / "inputs.npy"),
        "challenge_interventions_sha256": _sha(fixture / "interventions.npy"),
        "prediction_arrays": prediction_hashes, "source_challenge_outcomes_opened": False,
    }
    receipt["prediction_receipt_sha256"] = _digest(receipt)
    receipt_path = pdir / "prediction-receipt.json"
    _write_json(receipt_path, receipt)
    stored_receipt = json.loads(receipt_path.read_text())
    if stored_receipt.get("source_challenge_outcomes_opened") is not False or stored_receipt.get("prediction_receipt_sha256") != _digest({key: value for key, value in stored_receipt.items() if key != "prediction_receipt_sha256"}):
        raise ValueError("V29 prediction receipt failed its pre-outcome commitment check")
    for name, files in prediction_hashes.items():
        for key, expected_sha in files.items():
            if _sha(pdir / f"{name}-{key}.npy") != expected_sha:
                raise ValueError("V29 prediction array changed before outcome access")

    # Outcome phase starts only after prediction arrays and the receipt exist.
    source_dir = root / "preparation" / family / f"seed-{seed_index}" / "source"
    source = _source(family, prep_row["seed"] + prep_row["family_index"] * 1000)
    _load_state(source, source_dir, prep_row["source_state"])
    with torch.no_grad():
        target_base = source(torch.from_numpy(x)).numpy()
        target_changed = source(torch.from_numpy(x + delta)).numpy()
    target_effect = target_changed - target_base
    target_logits = target_base @ readout.T
    odir = root / "outcomes" / family / f"seed-{seed_index}"
    odir.mkdir(parents=True)
    outcome_hashes = {
        "target_base": _save_array(odir / "target-base.npy", target_base),
        "target_effect": _save_array(odir / "target-effect.npy", target_effect),
        "target_logits": _save_array(odir / "target-logits.npy", target_logits),
    }
    labels = np.argmax(target_logits, axis=1)
    target_shifted = target_logits - target_logits.max(axis=1, keepdims=True)
    target_ce = -target_shifted[np.arange(len(x)), labels] + np.log(np.exp(target_shifted).sum(axis=1))
    metrics = {}
    for name, values in predictions.items():
        per_stratum = {stratum: _score_effect(values["effect"][strata == stratum], target_effect[strata == stratum]) for stratum in STRATA}
        activation_error = np.linalg.norm(values["base"] - target_base, axis=1) / np.maximum(np.linalg.norm(target_base, axis=1), 1e-7)
        activation_cosine = np.sum(values["base"] * target_base, axis=1) / np.maximum(np.linalg.norm(values["base"], axis=1) * np.linalg.norm(target_base, axis=1), 1e-20)
        candidate_logits = values["logits"]
        candidate_shifted = candidate_logits - candidate_logits.max(axis=1, keepdims=True)
        candidate_ce = -candidate_shifted[np.arange(len(x)), labels] + np.log(np.exp(candidate_shifted).sum(axis=1))
        logit_error = np.linalg.norm(candidate_logits - target_logits, axis=1) / np.maximum(np.linalg.norm(target_logits, axis=1), 1e-7)
        per_stratum["activation_rescue"] = {
            "median_cosine": float(np.median(activation_cosine)),
            "median_normalized_error": float(np.median(activation_error)),
            "relative_mse_rescue_vs_noop": float(1.0 - np.mean((values["base"] - target_base) ** 2) / max(float(np.mean(target_base ** 2)), 1e-20)),
        }
        per_stratum["behavior"] = {
            "mean_cross_entropy_delta_nats": float(np.mean(candidate_ce - target_ce)),
            "max_normalized_logit_error": float(np.max(logit_error)),
            "top1_agreement": float(np.mean(np.argmax(candidate_logits, axis=1) == labels)),
        }
        metrics[name] = per_stratum
    treatment = metrics["treatment"]
    controls_beat = all(
        treatment[stratum]["median_normalized_error"] is not None
        and all(treatment[stratum]["median_normalized_error"] < metrics[control][stratum]["median_normalized_error"] for control in ("wrong_module", "random", "no_op", "affine_distillation"))
        for stratum in STRATA
    )
    treatment_pass = (
        controls_beat
        and all(treatment[stratum]["resolvable_count"] >= 100 and treatment[stratum]["direction_accuracy"] is not None and treatment[stratum]["direction_accuracy"] >= 0.80 and treatment[stratum]["median_cosine"] >= 0.90 and treatment[stratum]["median_normalized_error"] <= 0.20 for stratum in STRATA)
        and treatment["activation_rescue"]["median_cosine"] >= 0.90
        and treatment["activation_rescue"]["median_normalized_error"] <= 0.20
        and treatment["behavior"]["mean_cross_entropy_delta_nats"] <= 0.05
        and treatment["behavior"]["max_normalized_logit_error"] <= 0.05
    )
    return {
        "family": family, "seed_index": seed_index,
        "prediction_receipt_sha256": _sha(receipt_path),
        "prediction_arrays_sha256": _digest(prediction_hashes), "outcome_arrays_sha256": _digest(outcome_hashes),
        "treatment_pass": treatment_pass, "treatment_beats_all_controls_per_stratum": controls_beat,
        "metrics": metrics,
    }


def _runtime() -> dict[str, Any]:
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch")}
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() != "Darwin":
        rss *= 1024
    return {
        "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
        "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
        "packages": versions, "peak_rss_bytes": rss, "remote_workers": 0, "monetary_cost_usd": 0,
    }


def _evaluate(root: Path) -> dict[str, Any]:
    verify(root)
    _verify_runtime()
    if not (root / "candidate-freeze.json").is_file() or not (root / "challenge-manifest.json").is_file():
        raise FileNotFoundError("candidate freeze and challenge manifest are required")
    report_path = root / "result-report.json"
    if report_path.exists():
        raise FileExistsError("V29 evaluation is immutable")
    prep = json.loads((root / "preparation/candidate-preparation.json").read_text())
    freeze_row = json.loads((root / "candidate-freeze.json").read_text())
    freeze_digest = freeze_row.get("candidate_freeze_sha256")
    if freeze_digest != _digest({key: value for key, value in freeze_row.items() if key != "candidate_freeze_sha256"}):
        raise ValueError("candidate-freeze digest is invalid")
    if _sha(root / "preparation/candidate-preparation.json") != freeze_row["candidate_preparation_sha256"]:
        raise ValueError("candidate preparation changed after candidate freeze")
    challenge_manifest = json.loads((root / "challenge-manifest.json").read_text())
    challenge_digest = challenge_manifest.get("challenge_manifest_sha256")
    if challenge_digest != _digest({key: value for key, value in challenge_manifest.items() if key != "challenge_manifest_sha256"}):
        raise ValueError("challenge manifest digest is invalid")
    if challenge_manifest.get("packet_sha256") != verify(root)["packet_sha256"] or challenge_manifest.get("candidate_freeze_sha256") != _sha(root / "candidate-freeze.json"):
        raise ValueError("challenge manifest is not bound to the frozen packet and candidates")
    if challenge_manifest.get("challenge_generation_started_after_candidate_freeze") is not True or len(challenge_manifest.get("cases", [])) != 9:
        raise ValueError("challenge manifest is incomplete or violates candidate-freeze order")
    expected_keys = {(family, seed_index) for family in FAMILIES for seed_index in range(len(SEEDS))}
    actual_keys = {(row.get("family"), row.get("seed_index")) for row in challenge_manifest["cases"]}
    if actual_keys != expected_keys or any(row.get("partition_disjoint") is not True for row in challenge_manifest["cases"]):
        raise ValueError("challenge manifest must bind nine disjoint family/seed cases")
    frozen_cases = [{"family": row["family"], "seed_index": row["seed_index"], "candidate_state_bundle_sha256": row["candidate_state_bundle_sha256"]} for row in prep["cases"]]
    if freeze_row["cases"] != frozen_cases:
        raise ValueError("candidate freeze does not bind the reconstructed candidate set")
    metrics = [_case_evaluate(root, row) for row in prep["cases"]]
    all_pass = len(metrics) == 9 and all(row["treatment_pass"] for row in metrics)
    freeze_manifest = verify(root)
    report = {
        "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE, "continuation_slice": CONTINUATION_SLICE,
        "packet_sha256": freeze_manifest["packet_sha256"], "source_revision": freeze_manifest["source_revision"],
        "dirty_tree_status_sha256_excluding_v29_packet": freeze_manifest["dirty_tree_status_sha256_excluding_v29_packet"],
        "assurance_level": "INTERNAL_PASS" if all_pass else "INCONCLUSIVE",
        "outcome_status": "PASS" if all_pass else "FAIL",
        "reviewer_status": "NOT_SUBMITTED", "reproducibility_status": "PENDING_SECOND_RUNTIME",
        "claim_ceiling": CLAIM_CEILING, "cases": metrics, "runtime": _runtime(),
        "resource_budget": verify(root)["resource_budget"],
        "external_artifact_bytes_at_report": _directory_size(root),
        "data_and_artifact_custody": "all generated source functions, vectors, candidate arrays, predictions, and outcomes are in external V29 custody",
        "known_limitations": ["synthetic known functions only", "same-owner sequential proposer/evaluator", "no trained transformer or external authority", "CPU energy is not measured"],
        "next_blocked_authority_gate": "fresh-checkpoint packet for an independently trained replacement on a real model; external acceptance remains separately gated",
    }
    report["report_sha256"] = _digest(report)
    _write_json(report_path, report)
    return {"outcome_status": report["outcome_status"], "assurance_level": report["assurance_level"], "case_count": len(metrics), "passed_case_count": sum(row["treatment_pass"] for row in metrics), "report_sha256": report["report_sha256"]}


def run_primary() -> dict[str, Any]:
    verify(PRIMARY)
    _verify_runtime()
    if (PRIMARY / "result-report.json").exists() or (PRIMARY / "preparation").exists():
        raise FileExistsError("V29 primary execution is immutable")
    torch.set_num_threads(1)
    started = time.monotonic()
    _prepare(PRIMARY)
    _make_challenges(PRIMARY)
    result = _evaluate(PRIMARY)
    report_path = PRIMARY / "result-report.json"
    report = json.loads(report_path.read_text())
    elapsed = time.monotonic() - started
    report["total_wall_seconds"] = elapsed
    report["resource_blockers"] = []
    if elapsed > 1800:
        report["resource_blockers"].append("total_wall_budget_exceeded")
    if report["runtime"]["peak_rss_bytes"] > 4294967296:
        report["resource_blockers"].append("peak_rss_budget_exceeded")
    if _directory_size(PRIMARY) > 1073741824:
        report["resource_blockers"].append("external_artifact_storage_budget_exceeded")
    if report["resource_blockers"]:
        report["outcome_status"] = "INCONCLUSIVE"
        report["assurance_level"] = "INCONCLUSIVE"
    else:
        report["assurance_level"] = "INTERNAL_PASS"
    report["report_sha256"] = _digest({key: value for key, value in report.items() if key != "report_sha256"})
    _write_json(report_path, report)
    result["report_sha256"] = report["report_sha256"]
    if report["resource_blockers"]:
        result["outcome_status"] = "INCONCLUSIVE"
        result["assurance_level"] = "INCONCLUSIVE"
    return result


def replicate(primary: Path, replica: Path) -> dict[str, Any]:
    verify(primary)
    _verify_runtime()
    _external(replica)
    if replica.exists():
        raise FileExistsError("V29 replica custody must be new")
    shutil.copytree(primary / "frozen-packet", replica / "frozen-packet")
    shutil.copyfile(primary / "freeze.json", replica / "freeze.json")
    # Inputs and evaluator-only fixtures are copied; candidate arrays are rebuilt.
    shutil.copytree(primary / "preparation", replica / "preparation")
    shutil.copyfile(primary / "candidate-freeze.json", replica / "candidate-freeze.json")
    shutil.copytree(primary / "challenges", replica / "challenges")
    prep = json.loads((replica / "preparation/candidate-preparation.json").read_text())
    torch.set_num_threads(1)
    for row in prep["cases"]:
        directory = replica / "preparation" / row["family"] / f"seed-{row['seed_index']}"
        x = _load_array(directory / "adaptation_x.npy").astype(np.float32)
        y = _load_array(directory / "adaptation_target.npy").astype(np.float32)
        wrong_y = _load_array(directory / "adaptation_wrong_target.npy").astype(np.float32)
        if _sha(directory / "adaptation_x.npy") != row["adaptation_x_sha256"] or _sha(directory / "adaptation_target.npy") != row["adaptation_target_sha256"] or _sha(directory / "adaptation_wrong_target.npy") != row["adaptation_wrong_target_sha256"]:
            raise ValueError("replica adaptation artifact digest mismatch")
        treatment = _fit_student(x, y, row["seed"] + 50_000 + row["family_index"] * 100 + row["seed_index"])
        wrong = _fit_student(x, wrong_y, row["seed"] + 60_000 + row["family_index"] * 100 + row["seed_index"])
        random_student = Student()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(row["seed"] + 70_000 + row["family_index"] * 100 + row["seed_index"])
            random_student = Student().float().eval()
        affine = _fit_affine(x, y)
        for name, module in (("treatment", treatment), ("wrong_module", wrong), ("random", random_student), ("affine_distillation", affine)):
            actual = _state_arrays(module, directory / name)
            if actual != row["candidate_modules"][name]:
                raise ValueError(f"replica candidate state differs: {row['family']} seed {row['seed_index']} {name}")
    _evaluate(replica)
    primary_report = json.loads((primary / "result-report.json").read_text())
    replica_report = json.loads((replica / "result-report.json").read_text())
    primary_cases = primary_report["cases"]
    replica_cases = replica_report["cases"]
    matches = [
        (first["prediction_arrays_sha256"], first["outcome_arrays_sha256"])
        == (second["prediction_arrays_sha256"], second["outcome_arrays_sha256"])
        for first, second in zip(primary_cases, replica_cases)
    ]
    receipt = {
        "protocol_id": PROTOCOL_ID, "packet_sha256": verify(primary)["packet_sha256"],
        "primary_report_sha256_before_reconciliation": _sha(primary / "result-report.json"),
        "replica_report_sha256": _sha(replica / "result-report.json"),
        "candidate_reconstruction_match": True, "prediction_and_outcome_digests_match": bool(all(matches)),
        "runtime_prefixes_distinct": primary_report["runtime"]["prefix"] != replica_report["runtime"]["prefix"],
        "dependency_versions_match": primary_report["runtime"]["packages"] == replica_report["runtime"]["packages"],
        "case_digest_matches": matches,
    }
    required = ("candidate_reconstruction_match", "prediction_and_outcome_digests_match", "runtime_prefixes_distinct", "dependency_versions_match")
    receipt["reproduction_status"] = "REPRODUCED_LOCAL" if all(receipt[key] for key in required) else "INCONCLUSIVE"
    _write_json(replica / "replication-receipt.json", receipt)
    primary_report["reproducibility_status"] = receipt["reproduction_status"]
    if receipt["reproduction_status"] == "REPRODUCED_LOCAL":
        primary_report["assurance_level"] = "REPRODUCED_LOCAL"
    primary_report["replication_receipt_sha256"] = _sha(replica / "replication-receipt.json")
    primary_report["report_sha256"] = _digest({key: value for key, value in primary_report.items() if key != "report_sha256"})
    _write_json(primary / "result-report.json", primary_report)
    return {"reproduction_status": receipt["reproduction_status"], "case_count": len(matches), "all_case_digests_match": all(matches)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify", "run", "replicate"))
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze()
    elif args.command == "verify":
        result = {"packet_sha256": verify()["packet_sha256"], "verified": True}
    elif args.command == "run":
        result = run_primary()
    else:
        result = replicate(PRIMARY, REPLICA)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
