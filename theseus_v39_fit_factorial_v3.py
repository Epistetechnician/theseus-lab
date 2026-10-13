"""Frozen V3 first-MLP optimization-budget by capacity pilot.

V3 preserves the V1/V2 preflight stops, excludes V2 adaptation identities,
and removes the V39 capture mask before fitting.

State slices: theseus-lab-sandbox-v1, theseus-v39-first-mlp-fit-factorial-v3,
global-experiment-assurance-lifecycle-v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import theseus_v39_fit_factorial_v2 as prior_packet

engine = prior_packet.engine
REPO = Path(__file__).resolve().parent
PROTOCOL = "theseus-v39-first-mlp-fit-factorial-v3"
CLAIM = "LOCAL_REUSED_GPT2_FIRST_MLP_FIT_FACTOR_DIAGNOSIS_V3_ONLY"
BASE = engine.r.ARTIFACTS / "theseus-lab-sandbox-v1-v39-first-mlp-fit-factorial-v3"
FREEZE = REPO / "docs/v39-first-mlp-fit-factorial-v3-freeze.json"
V2_ROOT = engine.r.ARTIFACTS / "theseus-lab-sandbox-v1-v39-first-mlp-fit-factorial-v2"
V2_STOP_RECEIPT_SHA256 = "0c36192d525d9ca9aad0a14e790bbe6544a680ef15511ffa4376e676e97b95f2"
TRAINING_ENTROPY = "factorial-v3-adaptation"
MEMBERS = (
    "theseus_v39_fit_factorial_v3.py",
    "theseus_v39_fit_factorial_v3_closeout.py",
    "tests/test_v39_fit_factorial_v3.py",
    "docs/v39-first-mlp-fit-factorial-v3.md",
    "theseus_v39_fit_factorial_v2.py",
    "theseus_v39_fit_factorial_v2_closeout.py",
    "tests/test_v39_fit_factorial_v2.py",
    "docs/v39-first-mlp-fit-factorial-v2.md",
    "docs/v39-first-mlp-fit-factorial-v2-freeze.json",
    "docs/v39-first-mlp-fit-factorial-v2-results.md",
    "docs/v39-first-mlp-fit-factorial-v2-results.json",
    "theseus_v39_fit_factorial_v1.py",
    "theseus_v39_fit_factorial_v1_closeout.py",
    "tests/test_v39_fit_factorial_v1.py",
    "docs/v39-first-mlp-fit-factorial-v1.md",
    "docs/v39-first-mlp-fit-factorial-v1-freeze.json",
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

engine.PROTOCOL = PROTOCOL
engine.CLAIM = CLAIM
engine.BASE = BASE
engine.FREEZE = FREEZE
engine.TRAINING_ENTROPY = TRAINING_ENTROPY
engine.RESULT_MD = REPO / "docs/v39-first-mlp-fit-factorial-v3-results.md"
engine.RESULT_JSON = REPO / "docs/v39-first-mlp-fit-factorial-v3-results.json"
engine.MEMBERS = MEMBERS
engine.ARCHIVE_MEMBERS = (*MEMBERS, str(FREEZE.relative_to(REPO)))


def v2_failure_inputs() -> tuple[dict, dict, set[str], set[str]]:
    receipt_path = V2_ROOT / "primary/preflight-stop.json"
    if engine.sha(receipt_path) != V2_STOP_RECEIPT_SHA256:
        raise ValueError("V2 terminal preflight receipt bytes differ")
    receipt = engine.r.read(receipt_path)
    adaptation_path = V2_ROOT / "primary/gpt2_gelu-37101/adaptation.json"
    if engine.sha(adaptation_path) != receipt["adaptation_file_sha256"]:
        raise ValueError("V2 adaptation fixture bytes differ from stop receipt")
    adaptation = engine.r.read(adaptation_path)
    text_hashes = {row["text_sha256"] for rows in adaptation["encoded"].values() for row in rows}
    token_hashes = {
        engine.c.digest([int(token) for token in row["ids"]])
        for rows in adaptation["encoded"].values() for row in rows
    }
    if len(text_hashes) != receipt["excluded_text_identity_count"]:
        raise ValueError("V2 text identity count differs")
    if len(token_hashes) != receipt["excluded_token_identity_count"]:
        raise ValueError("V2 token identity count differs")
    if engine.c.digest(sorted(text_hashes)) != receipt["excluded_text_identity_set_sha256"]:
        raise ValueError("V2 text identity digest differs")
    if engine.c.digest(sorted(token_hashes)) != receipt["excluded_token_identity_set_sha256"]:
        raise ValueError("V2 token identity digest differs")
    if len(adaptation["encoded"]["adapt"]) != 128 or len(adaptation["encoded"]["donor"]) != 128:
        raise ValueError("V2 adaptation partition sizes differ")
    return receipt, adaptation, text_hashes, token_hashes


def verify_parent() -> dict:
    parent_state = prior_packet.verify_parent()
    v2_freeze = engine.r.read(REPO / "docs/v39-first-mlp-fit-factorial-v2-freeze.json")
    for member in v2_freeze["members"]:
        path = REPO / member["path"]
        if path.stat().st_size != member["bytes"] or engine.sha(path) != member["sha256"]:
            raise ValueError("V2 frozen source member differs")
    if engine.c.digest(v2_freeze["members"]) != v2_freeze["packet_sha256"]:
        raise ValueError("V2 packet digest differs")
    v2_result = engine.r.read(REPO / "docs/v39-first-mlp-fit-factorial-v2-results.json")
    if (v2_result["packet_sha256"] != v2_freeze["packet_sha256"]
            or v2_result["execution_status"] != "INCONCLUSIVE"):
        raise ValueError("V2 stop disposition differs")
    stop_receipt, _, text_hashes, token_hashes = v2_failure_inputs()
    return {**parent_state, "v2_result": v2_result, "v2_stop": stop_receipt,
            "v2_text_hashes": text_hashes, "v2_token_hashes": token_hashes}


_prior_exclusions = engine.inherited_exclusions


def inherited_exclusions() -> tuple[set[str], set[str]]:
    seen_text, seen_tokens = _prior_exclusions()
    _, _, text_hashes, token_hashes = v2_failure_inputs()
    seen_text.update(text_hashes)
    seen_tokens.update(token_hashes)
    return seen_text, seen_tokens


def unpad_capture(value) -> np.ndarray:
    """Validate and remove the V39 metadata-mask column before fitting."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 2 or array.shape != (96, 769) or array.dtype != np.float32:
        raise ValueError("V39 marked capture shape/dtype")
    mask = array[:, -1]
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("V39 marked capture mask is not binary")
    length = int(mask.sum())
    expected = np.arange(96) < length
    if length < 2 or not np.array_equal(mask, expected.astype(np.float32)):
        raise ValueError("V39 marked capture mask is not a contiguous sequence")
    if np.any(array[length:, :-1] != 0):
        raise ValueError("V39 padded features are not zero")
    features = array[:length, :-1].copy()
    if features.shape != (length, 768) or not np.isfinite(features).all():
        raise ValueError("V39 native feature width/value differs")
    return features


def capture_training(model, tokenizer, seed: int, seen_text: set[str], seen_tokens: set[str], check):
    rows = {key: engine.fixture_rules.records(
        engine.FAMILY, seed, key, engine.ADAPTATION_COUNT, engine.TRAINING_ENTROPY
    ) for key in ("adapt", "donor")}
    encoded = {key: engine.fixture_rules.encode(tokenizer, rows[key], seen_text, seen_tokens)
               for key in ("adapt", "donor")}
    captures = {key: [] for key in ("adapt", "donor")}
    for key in ("adapt", "donor"):
        for row in encoded[key]:
            check()
            output = engine.c.forward(model, engine.FAMILY, row["ids"], [engine.SITE], capture_all=True)
            captures[key].append(unpad_capture(output["sites"][engine.SITE]["x"]))
    x_rows, donor_rows = captures["adapt"], captures["donor"]
    lengths = np.asarray([(len(x), len(d)) for x, d in zip(x_rows, donor_rows, strict=True)], dtype=np.int64)
    paired = [d[np.arange(len(x)) * (len(d) - 1) // (len(x) - 1)]
              for x, d in zip(x_rows, donor_rows, strict=True)]
    x = torch.from_numpy(np.concatenate(x_rows).astype(np.float32, copy=False))
    donor = torch.from_numpy(np.concatenate(paired).astype(np.float32, copy=False))
    fixture_record = {"corpus": rows, "encoded": encoded}
    capture_record = {"x": x.numpy(), "donor": donor.numpy(),
                      "donor_source": np.concatenate(donor_rows).astype(np.float32, copy=False),
                      "lengths": lengths}
    return fixture_record, capture_record, x, donor


engine.verify_parent = verify_parent
engine.inherited_exclusions = inherited_exclusions
engine.capture_training = capture_training

_base_build_freeze = engine.build_freeze


def build_freeze() -> dict:
    value = _base_build_freeze()
    value["parent_terminal_inputs"] = [
        {
            "protocol_id": "theseus-v39-first-mlp-fit-factorial-v1",
            "freeze_sha256": engine.sha(REPO / "docs/v39-first-mlp-fit-factorial-v1-freeze.json"),
            "execution_status": "PREFLIGHT_STOP_BEFORE_RUNTIME_OR_DATA",
        },
        {
            "protocol_id": "theseus-v39-first-mlp-fit-factorial-v2",
            "freeze_packet_sha256": "f8d455412f6be813be91acf3bff3cb30cf6026bede98624e5072bd23d98078b7",
            "stop_receipt_sha256": V2_STOP_RECEIPT_SHA256,
            "excluded_text_identity_count": 256,
            "excluded_token_identity_count": 256,
            "training_updates": 0,
            "challenge_pairs": 0,
        },
    ]
    value["fixture_rules"]["prior_stop_fixture_use"] = "all V2 prompt and token identities are exclusion-only"
    return value


engine.build_freeze = build_freeze


def __getattr__(name: str):
    return getattr(engine, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "run"))
    parser.add_argument("--role", choices=("primary", "replica"), default="primary")
    args = parser.parse_args()
    if args.action == "verify":
        answer = {"packet_sha256": engine.verify_packet(), "protocol_id": PROTOCOL}
    else:
        answer = engine.run(args.role)
    print(json.dumps(answer, sort_keys=True))


if __name__ == "__main__":
    main()
