"""Frozen first-MLP optimization-budget by capacity pilot.

State slices: theseus-lab-sandbox-v1, theseus-v39-first-mlp-fit-factorial-v1,
global-experiment-assurance-lifecycle-v1. All model data and outputs remain on
PrimaryED; only code and scalar reports belong in the repository.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import replacement_loss_v39_runner as parent
import replacement_loss_v36_fixtures as fixture_rules

c, r = parent.c, parent.r
REPO = Path(__file__).resolve().parent
PROTOCOL = "theseus-v39-first-mlp-fit-factorial-v1"
CLAIM = "LOCAL_REUSED_GPT2_FIRST_MLP_FIT_FACTOR_DIAGNOSIS_ONLY"
PARENT = r.BASE
BASE = r.ARTIFACTS / "theseus-lab-sandbox-v1-v39-first-mlp-fit-factorial-v1"
FREEZE = REPO / "docs/v39-first-mlp-fit-factorial-v1-freeze.json"
RESULT_MD = REPO / "docs/v39-first-mlp-fit-factorial-v1-results.md"
RESULT_JSON = REPO / "docs/v39-first-mlp-fit-factorial-v1-results.json"
SEEDS = (37101, 37102, 37103)
CAPACITIES = (2, 4)
UPDATES = (1024, 4096)
METHOD = "jvp"
FAMILY = "gpt2_gelu"
SITE = 0
ADAPTATION_COUNT = 128
CHALLENGE_COUNT = 32
TRAINING_ENTROPY = "factorial-v1-adaptation"
POSITIONS = (1, 2, 4, 8)
SMALLEST_EFFECT_RATIO = 0.80
MAX_SECONDS = 8 * 60 * 60
MAX_RSS = 6 * 1024**3
MAX_BYTES = 24 * 1024**3
MIN_FREE_BYTES = 25 * 1024**3
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
ARCHIVE_MEMBERS = (*MEMBERS, str(FREEZE.relative_to(REPO)))
RECONCILIATION_KEYS = (
    "protocol_id", "state_slice", "revision", "packet_sha256", "claim_ceiling",
    "scientific_outcome", "factorial", "heldout", "training_bindings",
    "challenge_sha256", "candidate_freeze_sha256", "prediction_freeze_sha256",
    "probe_arrays_sha256", "checkpoint_manifest_sha256", "guide_site_sha256",
)


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    )
    return completed.stdout.strip()


def write_json(path: Path, value: dict) -> None:
    if not path.resolve().is_relative_to(BASE.resolve()):
        raise ValueError("write escaped factorial artifact root")
    r.mount_guard()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("JSON artifact already exists")
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def member_rows() -> list[dict]:
    return [
        {"path": path, "bytes": (REPO / path).stat().st_size, "sha256": sha(REPO / path)}
        for path in MEMBERS
    ]


def build_freeze() -> dict:
    rows = member_rows()
    return {
        "protocol_id": PROTOCOL,
        "state_slice": PROTOCOL,
        "state_slices": ["theseus-lab-sandbox-v1", PROTOCOL,
                         "global-experiment-assurance-lifecycle-v1"],
        "date": "2026-10-07",
        "revision": git_revision(),
        "freeze_status": "FROZEN_BEFORE_ADAPTATION_GENERATION_AND_FITTING",
        "claim_ceiling": CLAIM,
        "target": {
            "model_id": "mrm8488/spanish-gpt2",
            "revision": "fc396ce08c65ea334dcf541de7dc05cac89a29b2",
            "architecture_family": FAMILY,
            "site": SITE,
            "reuse_status": "REUSED_V39_GUID_ONE_CHECKPOINT_NOT_FRESH_OR_ARCHITECTURE_HELD_OUT",
        },
        "design": {
            "type": "paired_2x2_factorial_within_three_candidate_seeds",
            "seeds": list(SEEDS),
            "hidden_expansion_multipliers": list(CAPACITIES),
            "optimizer_updates": list(UPDATES),
            "candidate_count": len(SEEDS) * len(CAPACITIES) * len(UPDATES),
            "candidate_seed_pairing": "same seed and initialization RNG across both capacity and update cells; update trajectories share minibatch and JVP probe streams",
            "objective": "unchanged V39 first-block JVP objective and V36 core.fit implementation",
            "fixed_optimizer": {"name": "AdamW", "learning_rate": 0.001,
                                "weight_decay": 0.0001, "batch": 64,
                                "gradient_norm_clip": 1.0, "jvp_probes": 4,
                                "auxiliary_gradient_cap": "unchanged V36 core.fit"},
            "adaptation_prompts_per_seed": ADAPTATION_COUNT,
            "donor_prompts_per_seed": ADAPTATION_COUNT,
            "heldout_pairs_per_seed_per_stratum": CHALLENGE_COUNT,
            "heldout_strata": list(c.STRATA),
            "patch_coordinate_counts_by_stratum": list(POSITIONS),
        },
        "primary_metric": {
            "name": "pooled_training_direction_jvp_nrmse_on_adaptation_and_donor_tokens",
            "direction": "lower_is_better",
            "smallest_meaningful_main_effect_ratio": SMALLEST_EFFECT_RATIO,
            "decision": "each factor is supported only if its geometric-mean paired error ratio is at most 0.80 and all three seed-level paired ratios favor that factor",
            "analysis": "log-error factorial contrasts by seed; report interaction; no population-level significance claim from three seeds of one guide",
        },
        "heldout_gates": {
            "source": "unchanged V39/V36 thresholds",
            "output_nrmse_max": 0.10,
            "top1_agreement_min": 0.95,
            "nll_increase_max": 0.03,
            "activation_nrmse_max": 0.15,
            "actual_rescue_nrmse_max": 0.10,
            "rescue_prediction_nrmse_max": 0.10,
            "effect_resolved_fraction_min": 0.90,
            "effect_median_relative_error_max": 0.20,
            "effect_p90_relative_error_max": 0.50,
            "effect_median_cosine_min": 0.90,
            "control_error_margin_min": 0.10,
            "controls": ["guide no-op positive", "zero module", "wrong seed module at matched capacity and update budget", "random module at matched capacity"],
            "candidate_pass": "all gates and all three control-sensitivity comparisons pass in every stratum for that seed",
            "retention_pass": "one predeclared factorial cell passes candidate_pass for all three seeds",
        },
        "fixture_rules": {
            "generator": "V36 procedural rules, domain-separated by this protocol id",
            "adaptation_entropy_label": TRAINING_ENTROPY,
            "heldout_entropy": "OS entropy generated only after all candidate weights and traces are frozen",
            "text_and_token_disjointness": "against inherited intake exclusion hashes, all V39 adaptation and challenge fixtures, and every new partition; reject collisions without retry or regeneration",
            "token_bounds": "2-96; no truncation or data-dependent regeneration",
            "prediction_seal": "all candidates and controls are predicted and digest-sealed before the guide oracle is evaluated",
            "replica": "rebuild all candidates and match candidate bytes before reading primary heldout entropy; match prediction seal before its oracle evaluation",
        },
        "resources": {
            "execution": "local deterministic CPU, one thread, two existing isolated V36-compatible runtimes",
            "wall_seconds_per_role_max": MAX_SECONDS,
            "peak_rss_bytes_per_role_max": MAX_RSS,
            "artifact_bytes_per_role_max": MAX_BYTES,
            "free_space_reserve_bytes": MIN_FREE_BYTES,
            "paid_cap_usd": 0,
            "energy": "NOT_MEASURED",
            "full_economic_cost": "NOT_MEASURED",
        },
        "stop_conditions": [
            "parent source, model identity, runtime, or custody digest mismatch",
            "fresh input overlap, nonfinite value, or candidate reproduction mismatch",
            "resource cap or free-space reserve exceeded",
            "any execution error; retain root and do not retry under this identity",
        ],
        "nonclaims": [
            "no fresh checkpoint or architecture-held-out evidence",
            "no causal mechanism identity beyond the measured local intervention set",
            "no novelty, consciousness, human-alignment equivalence, independent acceptance, or production authorization",
        ],
        "reviewer_status": "NOT_SUBMITTED",
        "members": rows,
        "packet_sha256": c.digest(rows),
    }


def verify_parent() -> dict:
    old = (c.PROTOCOL, c.SLICE, c.CLAIM)
    try:
        c.PROTOCOL = "theseus-reused-gpt2-all-token-ladder-v39"
        c.SLICE = c.PROTOCOL
        c.CLAIM = "LOCAL_REUSED_GPT2_EARLY_ALL_TOKEN_REPLACEMENT_DEVELOPMENT_ONLY"
        parent.verify_source()
    finally:
        c.PROTOCOL, c.SLICE, c.CLAIM = old
    freeze = r.read(REPO / "docs/replacement-loss-v39-freeze.json")
    report = r.read(REPO / "docs/replacement-loss-v39-results.json")
    if report.get("scientific_outcome") != "FAIL" or report.get("assurance_level") != "REPRODUCED_LOCAL":
        raise ValueError("V39 terminal result is not the frozen reproduced failure")
    for role, field in (("primary", "primary_result_sha256"), ("replica", "replica_result_sha256")):
        if sha(PARENT / role / "result.json") != report[field]:
            raise ValueError("V39 terminal result bytes differ")
    validation = r.read(PARENT / "closeout/validation.json")
    if sha(PARENT / "closeout/validation.json") != report["validation_sha256"]:
        raise ValueError("V39 closeout bytes differ")
    if freeze["packet_sha256"] != report["packet_sha256"]:
        raise ValueError("V39 source packet differs")
    return {"freeze": freeze, "report": report, "validation": validation}


def verify_packet() -> str:
    parent_state = verify_parent()
    value = r.read(FREEZE)
    rows = member_rows()
    if value.get("protocol_id") != PROTOCOL or value.get("claim_ceiling") != CLAIM:
        raise ValueError("factorial protocol identity differs")
    if value.get("members") != rows or value.get("packet_sha256") != c.digest(rows):
        raise ValueError("factorial source packet changed after freeze")
    if value.get("target", {}).get("revision") != parent_state["report"]["checkpoint"]["revision"]:
        raise ValueError("frozen guide revision differs from V39")
    return value["packet_sha256"]


def verify_archive() -> None:
    for relative in ARCHIVE_MEMBERS:
        archived = BASE / "frozen-packet" / relative
        if sha(archived) != sha(REPO / relative):
            raise ValueError("frozen packet archive differs")


@contextlib.contextmanager
def new_protocol():
    old = (c.PROTOCOL, c.SLICE, c.CLAIM)
    c.PROTOCOL, c.SLICE, c.CLAIM = PROTOCOL, PROTOCOL, CLAIM
    try:
        yield
    finally:
        c.PROTOCOL, c.SLICE, c.CLAIM = old


class CapacityReplacement(nn.Module):
    """Independent two-layer MLP with a frozen expansion multiplier."""
    def __init__(self, width: int, family: str, seed: int, expansion: int,
                 mean: torch.Tensor | None = None, scale: torch.Tensor | None = None):
        super().__init__()
        if family not in c.FAMILIES or width < 1 or expansion not in CAPACITIES:
            raise ValueError("replacement dimensions or family")
        self.family, self.width, self.expansion = family, width, expansion
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.up = nn.Linear(width, expansion * width)
            self.down = nn.Linear(expansion * width, width)
        self.register_buffer("mean", torch.zeros(width) if mean is None else mean.detach().clone())
        self.register_buffer("scale", torch.ones(width) if scale is None else scale.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.up(x)
        hidden = torch.relu(hidden) if self.family == "opt_relu" else torch.nn.functional.gelu(hidden)
        return self.mean + self.scale * self.down(hidden)


def fit_candidate(teacher, x, donor, seed: int, expansion: int, updates: int, budget):
    """Call the frozen core objective with only the module width parameter changed."""
    previous = c.Replacement
    c.Replacement = lambda width, family, model_seed, mean=None, scale=None: CapacityReplacement(
        width, family, model_seed, expansion, mean, scale
    )
    try:
        model, trace = parent.ORIGINAL_FIT(teacher, x, donor, FAMILY, seed, METHOD, budget, updates)
    finally:
        c.Replacement = previous
    if model.up.out_features != expansion * model.width:
        raise ValueError("replacement capacity differs")
    if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
        raise ValueError("candidate has nonfinite parameters")
    return model.eval().requires_grad_(False), trace


def probe_directions(count: int, width: int, seed: int, fresh: bool = False) -> torch.Tensor:
    if count < 1 or width < 1:
        raise ValueError("probe dimensions")
    generator = torch.Generator().manual_seed(seed)
    if not fresh:
        torch.randint(width, (count,), generator=generator)
    return (torch.randint(2, (c.PROBES, count, width), generator=generator).float() * 2 - 1) / math.sqrt(width)


def probe_sums(teacher, student, x: torch.Tensor, donor: torch.Tensor,
               seed: int, fresh_seed: int, budget) -> dict:
    if x.shape != donor.shape or x.ndim != 2 or not torch.isfinite(x).all() or not torch.isfinite(donor).all():
        raise ValueError("probe input shape or values")
    count, width = x.shape
    output = {}
    for label, probe_seed, fresh in (("training", seed, False), ("fresh", fresh_seed, True)):
        directions = probe_directions(count, width, probe_seed, fresh=fresh)
        error_sum = reference_sum = 0.0
        element_count = 0
        for source in (x, donor):
            for start in range(0, count, 64):
                budget()
                end = min(start + 64, count)
                directions_batch = directions[:, start:end]
                guide = c.jvp(teacher, source[start:end], directions_batch).detach().double()
                candidate = c.jvp(student, source[start:end], directions_batch).detach().double()
                error_sum += float(torch.square(candidate - guide).sum())
                reference_sum += float(torch.square(guide).sum())
                element_count += guide.numel()
        error = math.sqrt(error_sum / max(reference_sum, element_count * 1e-8))
        output[label] = {
            "squared_error_sum": error_sum,
            "squared_reference_sum": reference_sum,
            "element_count": element_count,
            "nrmse": error,
        }
    return output


def factorial_summary(metrics: dict) -> dict:
    per_seed = {}
    for seed in SEEDS:
        cell = metrics[f"{FAMILY}-{seed}"]
        e22 = cell["cap2_updates1024"]["training_probe"]["nrmse"]
        e24 = cell["cap2_updates4096"]["training_probe"]["nrmse"]
        e42 = cell["cap4_updates1024"]["training_probe"]["nrmse"]
        e44 = cell["cap4_updates4096"]["training_probe"]["nrmse"]
        values = (e22, e24, e42, e44)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("factorial metric is not positive and finite")
        opt_logs = [math.log(e22) - math.log(e24), math.log(e42) - math.log(e44)]
        cap_logs = [math.log(e22) - math.log(e42), math.log(e24) - math.log(e44)]
        per_seed[str(seed)] = {
            "optimization_log_improvement": sum(opt_logs) / 2,
            "optimization_error_ratio": math.exp(-sum(opt_logs) / 2),
            "capacity_log_improvement": sum(cap_logs) / 2,
            "capacity_error_ratio": math.exp(-sum(cap_logs) / 2),
            "interaction_log_difference_in_differences": (opt_logs[0] - opt_logs[1]),
        }
    def summary(key: str) -> dict:
        logs = [per_seed[str(seed)][key] for seed in SEEDS]
        mean_log = sum(logs) / len(logs)
        ratio = math.exp(-mean_log)
        seed_direction = all(value > 0 for value in logs)
        return {
            "geometric_mean_error_ratio": ratio,
            "relative_error_reduction": 1 - ratio,
            "all_seed_directions_favor_factor": seed_direction,
            "supported": ratio <= SMALLEST_EFFECT_RATIO and seed_direction,
        }
    return {
        "per_seed": per_seed,
        "optimization_budget_main_effect": summary("optimization_log_improvement"),
        "capacity_main_effect": summary("capacity_log_improvement"),
        "interpretation_limit": "three paired candidate seeds on one reused guide; descriptive pilot only",
    }


def guard(start: float, root: Path) -> dict:
    r.mount_guard()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0
    free = shutil.disk_usage(r.ARTIFACTS).free
    if time.monotonic() - start > MAX_SECONDS or rss > MAX_RSS or size > MAX_BYTES:
        raise RuntimeError("factorial resource cap")
    if free < MIN_FREE_BYTES:
        raise RuntimeError("factorial free-space reserve")
    return {"wall_seconds": time.monotonic() - start, "peak_rss_bytes": rss,
            "artifact_bytes": size, "artifact_free_bytes": free,
            "paid_execution_usd": 0, "energy": "NOT_MEASURED"}


def guide_manifest() -> dict:
    report = r.read(REPO / "docs/replacement-loss-v39-results.json")
    value = r.read(PARENT / "checkpoints.json")
    if sha(PARENT / "checkpoints.json") != report["checkpoint_manifest_sha256"]:
        raise ValueError("parent guide manifest changed")
    if len(value.get("checkpoints", [])) != 1:
        raise ValueError("one frozen reused guide required")
    entry = value["checkpoints"][0]
    target = report["checkpoint"]
    if (entry["family"] != target["architecture_family"]
            or entry["model_id"] != target["model_id"]
            or entry["revision"] != target["revision"]):
        raise ValueError("guide identity differs from V39 result")
    guide_root = r.contained(PARENT, entry["local_path"])
    actual = {str(path.relative_to(guide_root)) for path in guide_root.rglob("*") if path.is_file()}
    if actual != {row["path"] for row in entry["files"]}:
        raise ValueError("guide file inventory differs")
    for row in entry["files"]:
        path = r.contained(guide_root, row["path"])
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("guide file digest differs")
    return value


def add_encoded_hashes(encoded: list[dict], seen_text: set[str], seen_tokens: set[str]) -> None:
    for row in encoded:
        seen_text.add(row["text_sha256"])
        seen_tokens.add(c.digest([int(token) for token in row["ids"]]))


def inherited_exclusions() -> tuple[set[str], set[str]]:
    intake = r.read(PARENT / "intake.json")
    exclusion = intake.get("exclusion", {})
    seen_text = set(exclusion.get("text_hashes", []))
    seen_tokens = set(exclusion.get("token_hashes", []))
    for seed in (36101, 36102, 36103):
        case = PARENT / "primary" / f"gpt2_gelu-{seed}"
        adaptation = r.read(case / "adaptation.json")
        for rows in adaptation["encoded"].values():
            add_encoded_hashes(rows, seen_text, seen_tokens)
        challenge = r.read(case / "challenge-fixtures.json")
        for key in ("encoded", "encoded_donors"):
            for rows in challenge[key].values():
                add_encoded_hashes(rows, seen_text, seen_tokens)
    return seen_text, seen_tokens


def copy_archive() -> None:
    for relative in ARCHIVE_MEMBERS:
        target = BASE / "frozen-packet" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        r.mount_guard()
        shutil.copyfile(REPO / relative, target)


def capture_training(model, tokenizer, seed: int, seen_text: set[str], seen_tokens: set[str], check):
    rows = {key: fixture_rules.records(FAMILY, seed, key, ADAPTATION_COUNT, TRAINING_ENTROPY)
            for key in ("adapt", "donor")}
    encoded = {key: fixture_rules.encode(tokenizer, rows[key], seen_text, seen_tokens)
               for key in ("adapt", "donor")}
    captures = {key: [] for key in ("adapt", "donor")}
    for key in ("adapt", "donor"):
        for row in encoded[key]:
            check()
            output = c.forward(model, FAMILY, row["ids"], [SITE], capture_all=True)
            captures[key].append(output["sites"][SITE]["x"].detach().cpu().numpy())
    x_rows, donor_rows = captures["adapt"], captures["donor"]
    lengths = np.asarray([(len(x), len(d)) for x, d in zip(x_rows, donor_rows, strict=True)], dtype=np.int64)
    paired = [d[np.arange(len(x)) * (len(d) - 1) // (len(x) - 1)] for x, d in zip(x_rows, donor_rows, strict=True)]
    x = torch.from_numpy(np.concatenate(x_rows).astype(np.float32, copy=False))
    donor = torch.from_numpy(np.concatenate(paired).astype(np.float32, copy=False))
    fixture_record = {"corpus": rows, "encoded": encoded}
    capture_record = {"x": x.numpy(), "donor": donor.numpy(),
                      "donor_source": np.concatenate(donor_rows).astype(np.float32, copy=False),
                      "lengths": lengths}
    return fixture_record, capture_record, x, donor


def save_npz(path: Path, arrays: dict) -> str:
    if path.exists():
        raise FileExistsError("artifact already exists")
    r.mount_guard()
    np.savez(path, **arrays)
    return c.array_digest(arrays)


def persist_candidate(root: Path, case: str, key: str, student, trace, metrics: dict) -> dict:
    weights = {name: tensor.detach().cpu().numpy().copy() for name, tensor in student.state_dict().items()}
    weight_path = root / case / f"{key}-weights.npz"
    trace_path = root / case / f"{key}-trace.npz"
    weights_sha = save_npz(weight_path, weights)
    trace_sha = save_npz(trace_path, {"readouts": trace})
    stable_metrics = {name: value for name, value in metrics.items() if name != "fit_seconds"}
    return {"weights_sha256": weights_sha, "trace_sha256": trace_sha,
            "expansion": student.expansion, "hidden_width": student.up.out_features,
            "parameter_count": sum(p.numel() for p in student.parameters()),
            "fit_metrics": stable_metrics}


def restore_candidate(root: Path, case: str, key: str, expansion: int, seed: int,
                      mean: torch.Tensor, scale: torch.Tensor) -> CapacityReplacement:
    path = root / case / f"{key}-weights.npz"
    with np.load(path, allow_pickle=False) as saved:
        state = {name: torch.from_numpy(saved[name].copy()) for name in saved.files}
    module = CapacityReplacement(len(state["mean"]), FAMILY, seed, expansion, mean, scale)
    module.load_state_dict(state, strict=True)
    return module.eval().requires_grad_(False)


def build_challenges(tokenizer, seen_text: set[str], seen_tokens: set[str], entropy: str):
    encoded, encoded_donors, raw, raw_donors = {}, {}, {}, {}
    for seed in SEEDS:
        case = f"{FAMILY}-{seed}"
        encoded[case], encoded_donors[case], raw[case], raw_donors[case] = {}, {}, {}, {}
        for stratum in c.STRATA:
            rows = fixture_rules.records(FAMILY, seed, stratum, CHALLENGE_COUNT, entropy)
            donors = fixture_rules.records(FAMILY, seed + 1000, stratum, CHALLENGE_COUNT, entropy)
            raw[case][stratum], raw_donors[case][stratum] = rows, donors
            encoded[case][stratum] = fixture_rules.encode(tokenizer, rows, seen_text, seen_tokens)
            encoded_donors[case][stratum] = fixture_rules.encode(tokenizer, donors, seen_text, seen_tokens)
    return {"encoded": encoded, "encoded_donors": encoded_donors,
            "rows": raw, "donor_rows": raw_donors}


def capacity_random(width: int, expansion: int, seed: int, mean: torch.Tensor,
                    scale: torch.Tensor) -> CapacityReplacement:
    return CapacityReplacement(width, FAMILY, seed, expansion, mean, scale).eval().requires_grad_(False)


def prediction_arrays(model, rows, donors, modules, entropy: str, case: str, check) -> dict:
    active = [SITE]
    with c.installed(model, FAMILY, modules):
        return r.predict(model, FAMILY, rows, donors, active, entropy, case, check)


def control_sensitivity(candidate: dict, controls: dict, oracle: dict) -> dict:
    summary = {}
    for stratum in c.STRATA:
        target = c.effect_metrics(candidate[stratum + "/joint_effect"],
                                  oracle[stratum + "/joint_effect"])["median_error"]
        values = {
            name: c.effect_metrics(arrays[stratum + "/joint_effect"],
                                  oracle[stratum + "/joint_effect"])["median_error"]
            for name, arrays in controls.items()
        }
        summary[stratum] = {
            "treatment_joint_median_error": target,
            "control_joint_median_errors": values,
            "all_controls_worse_by_margin": target is not None and all(
                value is not None and value >= target + 0.10 for value in values.values()
            ),
        }
    return summary


def run(role: str) -> dict:
    if role not in ("primary", "replica"):
        raise ValueError("role")
    packet = verify_packet()
    verify_archive() if role == "replica" else None
    runtime = r.runtime()
    if role == "primary":
        if BASE.exists():
            raise FileExistsError("factorial artifact root exists; no same-root retry")
        copy_archive()
    else:
        primary_runtime = r.read(BASE / "primary" / "runtime.json")
        if primary_runtime["prefix"] == runtime["prefix"] or primary_runtime["site_packages"] == runtime["site_packages"]:
            raise ValueError("replica must use the distinct isolated runtime")
        if {k: v for k, v in primary_runtime.items() if k not in ("prefix", "site_packages")} != {
            k: v for k, v in runtime.items() if k not in ("prefix", "site_packages")
        }:
            raise ValueError("primary/replica runtime fingerprints differ")
    root = BASE / role
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    check = lambda: guard(start, root)
    write_json(root / "runtime.json", runtime)
    with new_protocol():
        guide = guide_manifest()
        model, tokenizer = r.load_model(guide["checkpoints"][0])
        teacher = c.original_mlp(model, FAMILY, SITE).eval().requires_grad_(False)
        guide_site_sha = c.array_digest(c.state(teacher))
        seen_text, seen_tokens = inherited_exclusions()
        adaptation_bindings, captures, candidates = {}, {}, {}
        candidate_manifest = {"protocol_id": PROTOCOL, "packet_sha256": packet,
                              "guide_site_sha256": guide_site_sha, "seeds": {}}
        fit_metrics = {}
        for seed in SEEDS:
            check()
            case = f"{FAMILY}-{seed}"
            (root / case).mkdir()
            fixture_record, capture_record, x, donor = capture_training(
                model, tokenizer, seed, seen_text, seen_tokens, check
            )
            adaptation_path = root / case / "adaptation.json"
            write_json(adaptation_path, fixture_record)
            capture_path = root / case / "adaptation-captures.npz"
            capture_digest = save_npz(capture_path, capture_record)
            adaptation_bindings[case] = {
                "adaptation_file_sha256": sha(adaptation_path),
                "capture_file_sha256": sha(capture_path),
                "capture_arrays_sha256": capture_digest,
                "prompt_count": ADAPTATION_COUNT,
                "adaptation_token_vectors": int(len(x)),
                "feature_width": int(x.shape[1]),
            }
            captures[case] = (x, donor)
            candidates[case] = {}
            fit_metrics[case] = {}
            candidate_manifest["seeds"][case] = {}
            fit_seed = c.seed_for(seed, "candidate-fit-seed")
            fresh_seed = c.seed_for(seed, "fresh-probe-directions")
            for expansion in CAPACITIES:
                for updates in UPDATES:
                    check()
                    key = f"cap{expansion}_updates{updates}"
                    fit_started = time.monotonic()
                    student, trace = fit_candidate(teacher, x, donor, fit_seed,
                                                   expansion, updates, check)
                    projections = probe_sums(teacher, student, x, donor, fit_seed,
                                             fresh_seed, check)
                    fit_metrics[case][key] = {
                        "training_probe": projections["training"],
                        "fresh_probe": projections["fresh"],
                        "fit_seconds": time.monotonic() - fit_started,
                        "updates": updates,
                        "expansion": expansion,
                        "parameter_count": sum(p.numel() for p in student.parameters()),
                        "gradient_cap_fraction_below_one": float(np.mean(trace[:, 2] < 1.0))
                        if trace.ndim == 2 and trace.shape[1] >= 3 else None,
                        "gradient_cap_coefficient_median": float(np.median(trace[:, 2]))
                        if trace.ndim == 2 and trace.shape[1] >= 3 else None,
                    }
                    condition = persist_candidate(root, case, key, student, trace,
                                                  fit_metrics[case][key])
                    candidate_manifest["seeds"][case][key] = condition
                    candidates[case][key] = student
                    check()
            del captures[case]
            print(json.dumps({"event": "seed_candidates_frozen", "case": case,
                              "candidate_count": len(candidates[case])}), flush=True)
        candidate_manifest["adaptation_bindings"] = adaptation_bindings
        write_json(root / "candidate-freeze.json", candidate_manifest)
        if role == "replica":
            primary_freeze = r.read(BASE / "primary" / "candidate-freeze.json")
            if candidate_manifest != primary_freeze:
                raise ValueError("candidate bytes differ before heldout entropy access")
            primary_challenge = r.read(BASE / "primary" / "challenge.json")
            entropy = primary_challenge["entropy"]
        else:
            entropy = secrets.token_hex(32)
        challenge = build_challenges(tokenizer, seen_text, seen_tokens, entropy)
        challenge_bindings = {}
        for seed in SEEDS:
            case = f"{FAMILY}-{seed}"
            challenge_record = {"rows": challenge["rows"][case],
                                "donor_rows": challenge["donor_rows"][case],
                                "encoded": challenge["encoded"][case],
                                "encoded_donors": challenge["encoded_donors"][case]}
            path = root / case / "challenge-fixtures.json"
            write_json(path, challenge_record)
            challenge_bindings[case] = sha(path)
        challenge_record = {"entropy": entropy,
                            "candidate_freeze_sha256": sha(root / "candidate-freeze.json"),
                            "fixtures": challenge_bindings}
        write_json(root / "challenge.json", challenge_record)
        if role == "replica":
            pchallenge = r.read(BASE / "primary" / "challenge.json")
            if challenge_record != pchallenge:
                raise ValueError("replica challenge generation differs")
        prediction_bindings = {"protocol_id": PROTOCOL, "candidate_freeze_sha256": sha(root / "candidate-freeze.json"),
                               "challenge_sha256": sha(root / "challenge.json"), "seeds": {}}
        prediction_data = {}
        for seed in SEEDS:
            case = f"{FAMILY}-{seed}"
            prediction_bindings["seeds"][case] = {}
            prediction_data[case] = {}
            rows, donors = challenge["encoded"][case], challenge["encoded_donors"][case]
            for key in sorted(candidates[case]):
                modules = {SITE: candidates[case][key]}
                predictions = prediction_arrays(model, rows, donors, modules, entropy, case, check)
                path = root / case / f"{key}-predictions.npz"
                prediction_bindings["seeds"][case][key] = save_npz(path, predictions)
                prediction_data[case][key] = predictions
            # Controls are shared across update levels at each matched capacity.
            control_modules = {"zero": {SITE: c.Zero()}}
            for expansion in CAPACITIES:
                example_key = f"cap{expansion}_updates{UPDATES[0]}"
                example = candidates[case][example_key]
                rand_seed = c.seed_for(seed, "random-control", expansion)
                control_modules[f"random_cap{expansion}"] = {
                    SITE: capacity_random(example.width, expansion, rand_seed, example.mean, example.scale)
                }
            for expansion in CAPACITIES:
                for updates in UPDATES:
                    key = f"cap{expansion}_updates{updates}"
                    other_seed = SEEDS[(SEEDS.index(seed) + 1) % len(SEEDS)]
                    other_case = f"{FAMILY}-{other_seed}"
                    other_key = key
                    control_modules[f"wrong_{key}"] = {SITE: candidates[other_case][other_key]}
            for control_name, modules in control_modules.items():
                predictions = prediction_arrays(model, rows, donors, modules, entropy, case, check)
                path = root / case / f"control-{control_name}-predictions.npz"
                prediction_bindings["seeds"][case][control_name] = save_npz(path, predictions)
                prediction_data[case][control_name] = predictions
            print(json.dumps({"event": "candidate_predictions_sealed", "case": case,
                              "prediction_count": len(prediction_bindings["seeds"][case])}), flush=True)
        write_json(root / "prediction-freeze.json", prediction_bindings)
        if role == "replica":
            primary_prediction_freeze = r.read(BASE / "primary" / "prediction-freeze.json")
            if prediction_bindings != primary_prediction_freeze:
                raise ValueError("prediction seal differs before guide oracle")
        oracle_bindings, heldout = {}, {}
        for seed in SEEDS:
            case = f"{FAMILY}-{seed}"
            rows, donors = challenge["encoded"][case], challenge["encoded_donors"][case]
            # Core predict emits per-site activations only for active sites. Use the guide module
            # as an installed no-op so the oracle has the same native prediction schema.
            oracle = prediction_arrays(model, rows, donors,
                                       {SITE: c.original_mlp(model, FAMILY, SITE)},
                                       entropy, case, check)
            oracle_file = root / case / "oracle.npz"
            oracle_digest = save_npz(oracle_file, oracle)
            noop = prediction_arrays(model, rows, donors,
                                     {SITE: c.original_mlp(model, FAMILY, SITE)},
                                     entropy, case, check)
            if c.array_digest(noop) != oracle_digest:
                raise ValueError("actual guide no-op differs from oracle")
            noop_file = root / case / "noop.npz"
            noop_digest = save_npz(noop_file, noop)
            oracle_rescues = r.restore(model, FAMILY, rows, oracle, [SITE], check)
            save_npz(root / case / "oracle-rescues.npz", oracle_rescues)
            oracle_bindings[case] = {"oracle_sha256": oracle_digest, "noop_sha256": noop_digest}
            heldout[case] = {}
            for key in sorted(candidates[case]):
                prediction = prediction_data[case][key]
                candidate = candidates[case][key]
                rescues = r.restore(model, FAMILY, rows, prediction, [SITE], check)
                rescue_sha = save_npz(root / case / f"{key}-rescues.npz", rescues)
                metrics = c.score(prediction, oracle, rescues, [SITE])
                controls = {
                    "zero": prediction_data[case]["zero"],
                    f"random_cap{candidate.expansion}": prediction_data[case][f"random_cap{candidate.expansion}"],
                    f"wrong_{key}": prediction_data[case][f"wrong_{key}"],
                }
                sensitivity = control_sensitivity(prediction, controls, oracle)
                passed = (all(row["passed"] for row in metrics.values())
                          and all(row["all_controls_worse_by_margin"] for row in sensitivity.values()))
                heldout[case][key] = {"metrics": metrics, "control_sensitivity": sensitivity,
                                       "rescue_arrays_sha256": rescue_sha, "passed": passed}
            print(json.dumps({"event": "guide_oracle_scored", "case": case,
                              "candidate_passes": {key: value["passed"] for key, value in heldout[case].items()}}), flush=True)
        write_json(root / "oracle-bindings.json", oracle_bindings)
        factorial = factorial_summary(fit_metrics)
        candidate_passes = {
            key: all(heldout[f"{FAMILY}-{seed}"][key]["passed"] for seed in SEEDS)
            for key in sorted(next(iter(candidates.values())))
        }
        scientific_outcome = "PASS" if any(candidate_passes.values()) else "FAIL"
        result = {
            "protocol_id": PROTOCOL, "state_slice": PROTOCOL, "packet_sha256": packet,
            "revision": r.read(FREEZE)["revision"],
            "claim_ceiling": CLAIM, "scientific_outcome": scientific_outcome,
            "assurance_level": "INTERNAL_PASS", "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "PRIMARY_COMPLETE_REPLICA_PENDING" if role == "primary" else "REPLICA_COMPLETE_CLOSEOUT_PENDING",
            "factorial": factorial,
            "factorial_cells": fit_metrics,
            "heldout": {"candidate_passes_all_three_seeds": candidate_passes,
                        "seed_cells": heldout,
                        "qualified_factorial_cell_count": sum(candidate_passes.values()),
                        "all_thresholds_unchanged_from_v39": True},
            "training_bindings": adaptation_bindings,
            "challenge_sha256": sha(root / "challenge.json"),
            "candidate_freeze_sha256": sha(root / "candidate-freeze.json"),
            "prediction_freeze_sha256": sha(root / "prediction-freeze.json"),
            "probe_arrays_sha256": c.digest({
                case: {
                    key: {
                        kind: {name: value[name] for name in (
                            "squared_error_sum", "squared_reference_sum", "element_count"
                        )}
                        for kind, value in metrics[key].items()
                        if kind in ("training_probe", "fresh_probe")
                    }
                    for key in metrics
                }
                for case, metrics in fit_metrics.items()
            }),
            "checkpoint_manifest_sha256": sha(PARENT / "checkpoints.json"),
            "parent_validation_sha256": sha(PARENT / "closeout/validation.json"),
            "guide_site_sha256": guide_site_sha,
            "runtime": runtime,
            "resources": check(),
            "data_custody": "prompts, token IDs, guide assets, captures, candidates, predictions and oracle arrays on PrimaryED; runtimes on Backup ED",
            "new_training_steps": len(SEEDS) * len(CAPACITIES) * sum(UPDATES),
            "new_heldout_pairs": len(SEEDS) * len(c.STRATA) * CHALLENGE_COUNT,
            "paid_execution_usd": 0, "energy": "NOT_MEASURED", "full_economic_cost": "NOT_MEASURED",
            "limitations": ["one reused Spanish-trained GPT-2 guide", "one first MLP site", "three candidate seeds do not represent independent guide checkpoints", "local CPU runtimes share physical host and owner"],
            "next_blocked_authority_gate": "configured independent authority for stronger scientific acceptance; production requires separate operational authorization",
        }
        write_json(root / "result.json", result)
        if role == "replica":
            primary = r.read(BASE / "primary" / "result.json")
            if any(primary[key] != result[key] for key in RECONCILIATION_KEYS):
                raise ValueError("primary and replica results differ")
            write_json(root / "reconciliation.json", {"checks": {key: primary[key] == result[key]
                                                                  for key in RECONCILIATION_KEYS},
                                                    "primary_result_sha256": sha(BASE / "primary" / "result.json"),
                                                    "replica_result_sha256": sha(root / "result.json")})
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "run"))
    parser.add_argument("--role", choices=("primary", "replica"), default="primary")
    args = parser.parse_args()
    if args.action == "verify":
        print(json.dumps({"packet_sha256": verify_packet(), "protocol_id": PROTOCOL}))
    else:
        print(json.dumps(run(args.role), sort_keys=True))


if __name__ == "__main__":
    main()
