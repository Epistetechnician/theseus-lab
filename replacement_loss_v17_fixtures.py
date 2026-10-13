"""Synthetic data partitions for the V17 causal-supervision protocol.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v17-causal-supervised-whole-mlp-v1.
Prompt payloads are stored only in external owner custody.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Iterable

import replacement_loss_v3_fixtures as base


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v17-causal-supervised-whole-mlp-v1"
PROTOCOL_ID = "theseus-replacement-loss-v17-causal-supervised-whole-mlp"
PROMPT_RULES_VERSION = "v17-causal-supervision-synthetic-v1"
SEEDS = (202610401, 202610402, 202610403)
PER_STRATUM_COUNT = 32
CHALLENGES_PER_STRATUM = 20
STRATA = ("ordinary", "compositional", "adversarial")


def _namespace(records: Iterable[base.PromptRecord], partition: str, seed: int) -> list[base.PromptRecord]:
    result = []
    for row in records:
        example_id = hashlib.sha256(
            f"{PROTOCOL_ID}\0{partition}\0{seed}\0{row.example_id}".encode("utf-8")
        ).hexdigest()[:24]
        result.append(replace(
            row,
            example_id=example_id,
            partition=partition,
            input_text=f"Causal supervision replacement {partition} seed {seed}. {row.input_text}",
        ))
    return result


def generate_adaptation_fixture(seed_index: int) -> list[base.PromptRecord]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen V17 seed set")
    seed = SEEDS[seed_index]
    rows = []
    for stratum_index, stratum in enumerate(STRATA):
        partition = f"v17-adaptation-{seed_index}-{stratum}"
        source_seed = seed + (stratum_index + 1) * 1_000_000
        source = base._unique_records(partition, source_seed, PER_STRATUM_COUNT, stratum)
        rows.extend(_namespace(source, partition, seed))
    return rows


def generate_challenge_fixture(seed_index: int, secret_seed: bytes) -> list[base.PromptRecord]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen V17 seed set")
    if len(secret_seed) != 32:
        raise ValueError("V17 challenge entropy must contain exactly 256 bits")
    seed = int.from_bytes(hashlib.sha256(secret_seed + bytes([seed_index])).digest()[:8], "big")
    rows = []
    for stratum in STRATA:
        stratum_seed = seed ^ int.from_bytes(hashlib.sha256(stratum.encode()).digest()[:8], "big")
        partition = f"v17-challenge-{seed_index}-{stratum}"
        source = base._unique_records(partition, stratum_seed, CHALLENGES_PER_STRATUM, stratum)
        rows.extend(_namespace(source, partition, SEEDS[seed_index]))
    return rows


def fixture_payload(records: Iterable[base.PromptRecord], *, fixture_kind: str) -> dict[str, object]:
    rows = [asdict(row) for row in records]
    if not rows:
        raise ValueError("fixture must contain at least one row")
    ids = [row["example_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("fixture example IDs must be unique")
    record_digests = [
        hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for row in rows
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "fixture_kind": fixture_kind,
        "prompt_rules_version": PROMPT_RULES_VERSION,
        "count": len(rows),
        "stratum_counts": {name: sum(row["stratum"] == name for row in rows) for name in sorted({row["stratum"] for row in rows})},
        "example_set_sha256": hashlib.sha256("\n".join(sorted(record_digests)).encode("ascii")).hexdigest(),
        "records": rows,
    }


def validate_partition_disjointness(*payloads: dict[str, object]) -> None:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_content: set[str] = set()
    for payload in payloads:
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise ValueError("fixture records must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("fixture record must be an object")
            example_id = str(row["example_id"])
            prompt = str(row["input_text"])
            content = prompt + "\0" + str(row["continuation"])
            if example_id in seen_ids or prompt in seen_prompts or content in seen_content:
                raise ValueError("fixtures overlap by ID or exact prompt content")
            seen_ids.add(example_id)
            seen_prompts.add(prompt)
            seen_content.add(content)
