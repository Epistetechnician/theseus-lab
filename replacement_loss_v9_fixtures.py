"""Fresh, custody-safe synthetic fixture rules for replacement-loss-v9.

State slice: theseus-lab-sandbox-v1. Prompt text remains in owner custody.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, replace
from typing import Iterable

import replacement_loss_v3_fixtures as v3


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v9-causal-ladder"
PROMPT_RULES_VERSION = "v9-namespaced-synthetic-template-v1"
VISIBLE_SEED = 202609270
SEEDS = (202609271, 202609272, 202609273)
VISIBLE_COUNT = 32
ADAPTATION_COUNTS = {"ordinary": 64, "compositional": 32, "adversarial": 32}
CHALLENGES_PER_STRATUM = 20
STRATA = ("ordinary", "compositional", "adversarial")


def _namespace(records: Iterable[v3.PromptRecord], partition: str, seed: int) -> list[v3.PromptRecord]:
    rows = []
    for row in records:
        example_id = hashlib.sha256(
            f"{PROTOCOL_ID}\0{partition}\0{seed}\0{row.example_id}".encode("utf-8")
        ).hexdigest()[:24]
        rows.append(
            replace(
                row,
                example_id=example_id,
                partition=partition,
                input_text=f"V9 {partition} seed {seed}. {row.input_text}",
            )
        )
    return rows


def generate_visible_fixture() -> list[v3.PromptRecord]:
    rows = v3._unique_records("v9-visible", VISIBLE_SEED, VISIBLE_COUNT, "visible")
    return _namespace(rows, "v9-visible", VISIBLE_SEED)


def generate_adaptation_fixture(seed_index: int) -> list[v3.PromptRecord]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen v9 seed set")
    seed = SEEDS[seed_index]
    rows = []
    for offset, stratum in enumerate(STRATA):
        partition = f"v9-adaptation-{seed_index}-{stratum}"
        records = v3._unique_records(
            partition,
            seed + (offset + 1) * 100_000,
            ADAPTATION_COUNTS[stratum],
            stratum,
        )
        rows.extend(_namespace(records, partition, seed))
    return rows


def generate_challenge_fixture(seed_index: int, secret_seed: bytes | None = None) -> list[v3.PromptRecord]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen v9 seed set")
    entropy = secrets.token_bytes(32) if secret_seed is None else bytes(secret_seed)
    if len(entropy) < 16:
        raise ValueError("challenge entropy must contain at least 128 bits")
    seed = int.from_bytes(hashlib.sha256(entropy + bytes([seed_index])).digest()[:8], "big")
    rows = []
    for stratum in STRATA:
        partition = f"v9-challenge-{seed_index}-{stratum}"
        stratum_seed = seed ^ int.from_bytes(hashlib.sha256(stratum.encode()).digest()[:8], "big")
        records = v3._unique_records(partition, stratum_seed, CHALLENGES_PER_STRATUM, stratum)
        rows.extend(_namespace(records, partition, SEEDS[seed_index]))
    return rows


def fixture_payload(records: Iterable[v3.PromptRecord], *, fixture_kind: str) -> dict[str, object]:
    material = list(records)
    rows = [asdict(record) for record in material]
    if not rows:
        raise ValueError("fixture must contain at least one record")
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("fixture example IDs must be unique")
    record_digests = [
        hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        for row in rows
    ]
    strata = sorted({str(row["stratum"]) for row in rows})
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "fixture_kind": fixture_kind,
        "prompt_rules_version": PROMPT_RULES_VERSION,
        "count": len(rows),
        "stratum_counts": {stratum: sum(row["stratum"] == stratum for row in rows) for stratum in strata},
        "example_set_sha256": hashlib.sha256("\n".join(sorted(record_digests)).encode("ascii")).hexdigest(),
        "records": rows,
    }


def validate_partition_disjointness(*fixtures: dict[str, object]) -> None:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_contents: set[str] = set()
    for fixture in fixtures:
        records = fixture.get("records")
        if not isinstance(records, list):
            raise ValueError("fixture records must be a list")
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("fixture row must be an object")
            example_id = str(row["example_id"])
            prompt = str(row["input_text"])
            content = prompt + "\0" + str(row["continuation"])
            if example_id in seen_ids or prompt in seen_prompts or content in seen_contents:
                raise ValueError("fixtures are not disjoint by ID and exact prompt content")
            seen_ids.add(example_id)
            seen_prompts.add(prompt)
            seen_contents.add(content)


def assert_stratum_counts(payload: dict[str, object], expected: dict[str, int]) -> None:
    actual = payload.get("stratum_counts")
    if actual != expected:
        raise ValueError(f"stratum counts differ from frozen v9 rules: {actual}")
