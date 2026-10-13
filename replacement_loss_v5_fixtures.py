"""Custody-safe synthetic fixture rules for replacement-loss-v5.

State slice: theseus-lab-sandbox-v1. Generated prompt content stays in
external owner custody.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict
from typing import Any

import replacement_loss_v3_fixtures as v3_fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v5-balanced-adversarial-support"
SEEDS = (20261001, 20261002, 20261003)
DEVELOPMENT_ADAPTATION_SEEDS = (20260930, 20260931, 20260932)
DEVELOPMENT_TEST_SEEDS = (20991231, 20991232)
TEST_CHALLENGE_ENTROPY = b"v5-fixture-test-entropy-000000000000"
ADAPTATION_PROMPTS_PER_ARM = 128
SHARED_ORDINARY_PROMPTS = 64
ARM_SPECIFIC_PROMPTS = 64
CHALLENGES_PER_STRATUM = 20
STRATA = ("ordinary", "compositional", "adversarial")


def _payload(kind: str, records: list[object]) -> dict[str, Any]:
    rows = [asdict(record) if hasattr(record, "__dataclass_fields__") else dict(record) for record in records]
    digests = [
        hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
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
        "stratum_counts": {name: sum(row["stratum"] == name for row in rows) for name in sorted({str(row["stratum"]) for row in rows})},
        "example_set_sha256": hashlib.sha256("\n".join(sorted(digests)).encode("ascii")).hexdigest(),
        "records": rows,
    }


def _generate_adaptation_fixtures_for_seed(seed_index: int, seed: int) -> dict[str, dict[str, Any]]:
    shared = v3_fixtures._unique_records(f"v5-adaptation-{seed_index}-shared", seed, SHARED_ORDINARY_PROMPTS, "ordinary")
    ordinary = v3_fixtures._unique_records(f"v5-adaptation-{seed_index}-ordinary", seed + 100, ARM_SPECIFIC_PROMPTS, "ordinary")
    compositional = v3_fixtures._unique_records(f"v5-adaptation-{seed_index}-compositional", seed + 200, ARM_SPECIFIC_PROMPTS, "compositional")
    adversarial = v3_fixtures._unique_records(f"v5-adaptation-{seed_index}-adversarial", seed + 300, ARM_SPECIFIC_PROMPTS // 2, "adversarial")
    return {
        "ordinary_only": _payload("adaptation-ordinary-only", shared + ordinary),
        "compositional_augmented": _payload("adaptation-compositional-augmented", shared + compositional),
        "balanced_augmented": _payload("adaptation-balanced-augmented", shared + compositional[:32] + adversarial),
    }


def generate_adaptation_fixtures(seed_index: int) -> dict[str, dict[str, Any]]:
    """Generate equal-size arms with shared ordinary support and fixed row counts."""

    if seed_index not in (0, 1, 2):
        raise ValueError("seed_index must be 0, 1, or 2")
    return _generate_adaptation_fixtures_for_seed(seed_index, SEEDS[seed_index])


def generate_challenge_fixture(seed_index: int, entropy: bytes | None = None) -> dict[str, Any]:
    """Generate the sealed 60-row challenge fixture from post-candidate entropy."""

    if seed_index not in (0, 1, 2):
        raise ValueError("seed_index must be 0, 1, or 2")
    challenge_entropy = secrets.token_bytes(32) if entropy is None else bytes(entropy)
    if len(challenge_entropy) < 16:
        raise ValueError("challenge entropy must contain at least 128 bits")
    records = v3_fixtures.generate_challenge_fixture(seed_index, challenge_entropy)
    if len(records) != CHALLENGES_PER_STRATUM * len(STRATA):
        raise AssertionError("challenge row count drifted")
    return _payload(f"challenge-seed-{SEEDS[seed_index]}", records)


def validate_partition_disjointness(fixtures: list[dict[str, Any]]) -> None:
    """Reject repeated IDs, prompts, or exact prompt/continuation pairs."""

    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_content: set[str] = set()
    for fixture in fixtures:
        records = fixture.get("records")
        if not isinstance(records, list):
            raise ValueError("fixture records are malformed")
        for row in records:
            example_id = str(row["example_id"])
            prompt = str(row["input_text"])
            content = hashlib.sha256((prompt + "\0" + str(row["continuation"])).encode("utf-8")).hexdigest()
            if example_id in seen_ids or prompt in seen_prompts or content in seen_content:
                raise ValueError("fixture partitions overlap by ID or exact content")
            seen_ids.add(example_id)
            seen_prompts.add(prompt)
            seen_content.add(content)


def union_adaptation_fixture(adaptation: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Deduplicate intentional shared adaptation rows into one validation set."""

    unique: dict[str, dict[str, Any]] = {}
    for fixture in adaptation.values():
        for row in fixture["records"]:
            unique.setdefault(str(row["example_id"]), row)
    rows = list(unique.values())
    if not rows:
        raise ValueError("adaptation union cannot be empty")
    return {"records": rows}
