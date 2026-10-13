"""Synthetic adaptation and challenge fixture rules for replacement-loss-v4.

State slice: theseus-lab-sandbox-v1.

This file contains generation rules only. Materialized prompts and continuations
must stay in owner-approved external custody.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict

import replacement_loss_v3_fixtures as v3_fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v4-compositional-support"
SEEDS = (20260927, 20260928, 20260929)
ADAPTATION_PROMPTS_PER_ARM = 128
SHARED_ORDINARY_PROMPTS = 64
ARM_SPECIFIC_PROMPTS = 64
CHALLENGES_PER_STRATUM = 20
STRATA = ("ordinary", "compositional", "adversarial")


def _payload(kind: str, records: list[object]) -> dict[str, object]:
    rows = [asdict(record) for record in records]
    material = [
        hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for row in rows
    ]
    ids = [str(row["example_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("fixture must contain unique, non-empty records")
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "fixture_kind": kind,
        "count": len(rows),
        "stratum_counts": {
            name: sum(row["stratum"] == name for row in rows)
            for name in sorted({str(row["stratum"]) for row in rows})
        },
        "example_set_sha256": hashlib.sha256("\n".join(sorted(material)).encode("ascii")).hexdigest(),
        "records": rows,
    }


def generate_adaptation_fixtures(seed_index: int) -> dict[str, dict[str, object]]:
    """Return equal-size paired arms sharing 64 ordinary records.

    The baseline has 64 shared ordinary and 64 additional ordinary records.
    The treatment has the same 64 shared ordinary records and 64 additional
    compositional records. The runner later samples exactly 16 token positions
    per prompt, giving each arm 2,048 paired projection rows.
    """

    if seed_index not in (0, 1, 2):
        raise ValueError("seed_index must be 0, 1, or 2")
    seed = SEEDS[seed_index]
    shared = v3_fixtures._unique_records(
        f"v4-adaptation-{seed_index}-shared", seed, SHARED_ORDINARY_PROMPTS, "ordinary"
    )
    ordinary = v3_fixtures._unique_records(
        f"v4-adaptation-{seed_index}-ordinary", seed + 100, ARM_SPECIFIC_PROMPTS, "ordinary"
    )
    compositional = v3_fixtures._unique_records(
        f"v4-adaptation-{seed_index}-compositional", seed + 200, ARM_SPECIFIC_PROMPTS, "compositional"
    )
    baseline = shared + ordinary
    treatment = shared + compositional
    if len(baseline) != ADAPTATION_PROMPTS_PER_ARM or len(treatment) != ADAPTATION_PROMPTS_PER_ARM:
        raise AssertionError("adaptation arm prompt counts drifted")
    shared_ids = {record.example_id for record in shared}
    baseline_ids = {record.example_id for record in baseline}
    treatment_ids = {record.example_id for record in treatment}
    if baseline_ids & treatment_ids != shared_ids:
        raise AssertionError("paired arms must overlap only on the predeclared shared ordinary records")
    return {
        "ordinary_only": _payload("adaptation-ordinary-only", baseline),
        "compositional_augmented": _payload("adaptation-compositional-augmented", treatment),
    }


def generate_challenge_fixture(seed_index: int, entropy: bytes | None = None) -> dict[str, object]:
    """Generate a fresh held-out fixture from OS entropy after candidate freeze."""

    if seed_index not in (0, 1, 2):
        raise ValueError("seed_index must be 0, 1, or 2")
    challenge_entropy = secrets.token_bytes(32) if entropy is None else bytes(entropy)
    if len(challenge_entropy) < 16:
        raise ValueError("challenge entropy must contain at least 128 bits")
    records = v3_fixtures.generate_challenge_fixture(seed_index, challenge_entropy)
    if len(records) != CHALLENGES_PER_STRATUM * len(STRATA):
        raise AssertionError("challenge row count drifted")
    return _payload(f"challenge-seed-{seed_index}", records)


def validate_fresh_disjointness(
    adaptation: dict[str, dict[str, object]], challenge: dict[str, object]
) -> None:
    """Require fresh challenge content to be disjoint from both adaptation arms."""

    challenge_rows = challenge.get("records")
    if not isinstance(challenge_rows, list):
        raise ValueError("challenge fixture records are malformed")
    challenge_ids = {str(row["example_id"]) for row in challenge_rows}
    challenge_text = {str(row["input_text"]) for row in challenge_rows}
    challenge_content = {
        hashlib.sha256((str(row["input_text"]) + "\0" + str(row["continuation"])).encode("utf-8")).hexdigest()
        for row in challenge_rows
    }
    if len(challenge_ids) != len(challenge_rows):
        raise ValueError("challenge fixture contains duplicate IDs")
    for fixture in adaptation.values():
        rows = fixture.get("records")
        if not isinstance(rows, list):
            raise ValueError("adaptation fixture records are malformed")
        for row in rows:
            content = hashlib.sha256(
                (str(row["input_text"]) + "\0" + str(row["continuation"])).encode("utf-8")
            ).hexdigest()
            if str(row["example_id"]) in challenge_ids or str(row["input_text"]) in challenge_text or content in challenge_content:
                raise ValueError("challenge and adaptation fixtures overlap")

