"""Custody-safe fixtures for replacement-loss-v6.

State slice: theseus-lab-sandbox-v1. Generated prompt text stays outside the
repository under owner-approved custody.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Iterable

import replacement_loss_v3_fixtures as v3


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v6-downstream-aware"
SEEDS = (20261101, 20261102, 20261103)
VISIBLE_SEED = 20261100
VISIBLE_COUNT = 32
ADAPTATION_PROMPTS = {"ordinary": 64, "compositional": 32, "adversarial": 32}
CHALLENGES_PER_STRATUM = 20
STRATA = ("ordinary", "compositional", "adversarial")


def _namespace(records: Iterable[Any], partition: str, seed: int) -> list[Any]:
    result = []
    for record in records:
        material = f"{PROTOCOL_ID}\0{partition}\0{seed}\0{record.example_id}"
        example_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        result.append(
            replace(
                record,
                example_id=example_id,
                partition=partition,
                input_text=f"V6 {partition} {seed}. {record.input_text}",
            )
        )
    return result


def generate_visible_fixture() -> list[Any]:
    """Generate fixed visible-only records for layer selection."""

    rows = v3._unique_records("v6-visible", VISIBLE_SEED, VISIBLE_COUNT, "visible")
    return _namespace(rows, "v6-visible", VISIBLE_SEED)


def generate_adaptation_fixture(seed_index: int) -> list[Any]:
    """Generate balanced, seed-specific training prompts."""

    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen v6 seed set")
    seed = SEEDS[seed_index]
    rows = []
    for offset, stratum in enumerate(STRATA):
        count = ADAPTATION_PROMPTS[stratum]
        source = v3._unique_records(
            f"v6-adaptation-{seed_index}-{stratum}",
            seed + (offset + 1) * 10_000,
            count,
            stratum,
        )
        rows.extend(_namespace(source, f"v6-adaptation-{seed_index}-{stratum}", seed))
    return rows


def generate_challenge_fixture(seed_index: int, secret_seed: bytes | None = None) -> list[Any]:
    """Generate held-out records from fresh entropy after candidate freeze."""

    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen v6 seed set")
    source = v3.generate_challenge_fixture(seed_index, secret_seed=secret_seed)
    return _namespace(source, f"v6-challenge-{seed_index}", SEEDS[seed_index])


def fixture_payload(records: Iterable[Any], *, fixture_kind: str) -> dict[str, Any]:
    payload = v3.fixture_payload(records, fixture_kind=f"v6-{fixture_kind}")
    payload["protocol_id"] = PROTOCOL_ID
    payload["prompt_rules_version"] = "v6-namespaced-synthetic-template-v1"
    return payload


def validate_partition_disjointness(*fixtures: dict[str, Any]) -> None:
    v3.validate_partition_disjointness(*fixtures)


def assert_stratum_counts(payload: dict[str, Any], expected: dict[str, int]) -> None:
    actual = payload.get("stratum_counts")
    if actual != expected:
        raise ValueError(f"fixture stratum counts differ from v6 preregistration: {actual}")
