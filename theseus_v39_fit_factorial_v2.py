"""Frozen V2 wrapper for the first-MLP optimization-budget by capacity pilot.

V1 stopped before model I/O when its parent verification call targeted the
wrong adapter. This packet preserves V1 and binds the corrected helper call.

State slices: theseus-lab-sandbox-v1, theseus-v39-first-mlp-fit-factorial-v2,
global-experiment-assurance-lifecycle-v1.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import theseus_v39_fit_factorial_v1 as engine

REPO = Path(__file__).resolve().parent
PROTOCOL = "theseus-v39-first-mlp-fit-factorial-v2"
CLAIM = "LOCAL_REUSED_GPT2_FIRST_MLP_FIT_FACTOR_DIAGNOSIS_V2_ONLY"
BASE = engine.r.ARTIFACTS / "theseus-lab-sandbox-v1-v39-first-mlp-fit-factorial-v2"
FREEZE = REPO / "docs/v39-first-mlp-fit-factorial-v2-freeze.json"
MEMBERS = (
    "theseus_v39_fit_factorial_v2.py",
    "theseus_v39_fit_factorial_v2_closeout.py",
    "tests/test_v39_fit_factorial_v2.py",
    "docs/v39-first-mlp-fit-factorial-v2.md",
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
engine.RESULT_MD = REPO / "docs/v39-first-mlp-fit-factorial-v2-results.md"
engine.RESULT_JSON = REPO / "docs/v39-first-mlp-fit-factorial-v2-results.json"
engine.MEMBERS = MEMBERS
engine.ARCHIVE_MEMBERS = (*MEMBERS, str(FREEZE.relative_to(REPO)))


def verify_parent() -> dict:
    """Verify V39 via the shared V36 runner object bound by its adapter."""
    core = engine.c
    old = (core.PROTOCOL, core.SLICE, core.CLAIM)
    try:
        core.PROTOCOL = "theseus-reused-gpt2-all-token-ladder-v39"
        core.SLICE = core.PROTOCOL
        core.CLAIM = "LOCAL_REUSED_GPT2_EARLY_ALL_TOKEN_REPLACEMENT_DEVELOPMENT_ONLY"
        engine.r.verify_source()
    finally:
        core.PROTOCOL, core.SLICE, core.CLAIM = old
    freeze = engine.r.read(REPO / "docs/replacement-loss-v39-freeze.json")
    report = engine.r.read(REPO / "docs/replacement-loss-v39-results.json")
    if report.get("scientific_outcome") != "FAIL" or report.get("assurance_level") != "REPRODUCED_LOCAL":
        raise ValueError("V39 terminal result is not the frozen reproduced failure")
    for role, field in (("primary", "primary_result_sha256"), ("replica", "replica_result_sha256")):
        if engine.sha(engine.PARENT / role / "result.json") != report[field]:
            raise ValueError("V39 terminal result bytes differ")
    if engine.sha(engine.PARENT / "closeout/validation.json") != report["validation_sha256"]:
        raise ValueError("V39 closeout bytes differ")
    if freeze["packet_sha256"] != report["packet_sha256"]:
        raise ValueError("V39 source packet differs")
    return {"freeze": freeze, "report": report}


engine.verify_parent = verify_parent


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
