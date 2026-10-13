"""Synthetic fixtures for staged V19 causal/value qualification.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v19-causal-fit-qualification-v1.
Prompt payloads remain in external owner custody.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Iterable

import replacement_loss_v3_fixtures as base


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v19-causal-fit-qualification-v1"
DEV_PROTOCOL_ID = "theseus-replacement-loss-v19-causal-fit-development"
CONFIRM_PROTOCOL_ID = "theseus-replacement-loss-v19-causal-fit-confirmation"
STRATA = ("ordinary", "compositional", "adversarial")
DEVELOPMENT_SEEDS = (202610601, 202610602, 202610603)
CONFIRMATION_SEEDS = (202610701, 202610702, 202610703)
ADAPTATION_PER_STRATUM = 64
FIT_PER_STRATUM = 48
VALIDATION_PER_STRATUM = 16
CHALLENGES_PER_STRATUM = 20
_ACTIVE_STAGE = "development"
SEEDS = DEVELOPMENT_SEEDS


def activate_stage(stage: str) -> None:
    global _ACTIVE_STAGE, SEEDS
    _ACTIVE_STAGE = stage
    SEEDS = seeds_for(stage)


def seeds_for(stage: str) -> tuple[int, int, int]:
    if stage == "development":
        return DEVELOPMENT_SEEDS
    if stage == "confirmation":
        return CONFIRMATION_SEEDS
    raise ValueError("stage must be development or confirmation")


def protocol_for(stage: str) -> str:
    if stage == "development":
        return DEV_PROTOCOL_ID
    if stage == "confirmation":
        return CONFIRM_PROTOCOL_ID
    raise ValueError("stage must be development or confirmation")


def _namespace(records: Iterable[base.PromptRecord], partition: str, seed: int, protocol_id: str) -> list[base.PromptRecord]:
    result = []
    for row in records:
        example_id = hashlib.sha256(
            f"{protocol_id}\0{partition}\0{seed}\0{row.example_id}".encode("utf-8")
        ).hexdigest()[:24]
        result.append(replace(
            row,
            example_id=example_id,
            partition=partition,
            input_text=f"Causal value qualification {partition} seed {seed}. {row.input_text}",
        ))
    return result


def generate_adaptation_fixture(stage: str, seed_index: int) -> list[base.PromptRecord]:
    seeds = seeds_for(stage)
    if seed_index not in range(len(seeds)):
        raise ValueError("seed_index is outside the frozen stage seed set")
    protocol_id = protocol_for(stage)
    seed = seeds[seed_index]
    rows = []
    for stratum_index, stratum in enumerate(STRATA):
        partition = f"v19-{stage}-adaptation-{seed_index}-{stratum}"
        source_seed = seed + (stratum_index + 1) * 1_000_000
        source = base._unique_records(partition, source_seed, ADAPTATION_PER_STRATUM, stratum)
        rows.extend(_namespace(source, partition, seed, protocol_id))
    return rows


def generate_challenge_fixture(seed_index: int, secret_seed: bytes, *, stage: str | None = None) -> list[base.PromptRecord]:
    stage = stage or _ACTIVE_STAGE
    seeds = seeds_for(stage)
    if stage != "confirmation":
        raise ValueError("only confirmation fixtures may contain a challenge")
    if seed_index not in range(len(seeds)):
        raise ValueError("seed_index is outside the frozen confirmation seed set")
    if len(secret_seed) != 32:
        raise ValueError("confirmation challenge entropy must contain exactly 256 bits")
    protocol_id = protocol_for(stage)
    seed = int.from_bytes(hashlib.sha256(secret_seed + bytes([seed_index])).digest()[:8], "big")
    rows = []
    for stratum in STRATA:
        stratum_seed = seed ^ int.from_bytes(hashlib.sha256(stratum.encode()).digest()[:8], "big")
        partition = f"v19-confirmation-challenge-{seed_index}-{stratum}"
        source = base._unique_records(partition, stratum_seed, CHALLENGES_PER_STRATUM, stratum)
        rows.extend(_namespace(source, partition, seeds[seed_index], protocol_id))
    return rows


def fixture_payload(records: Iterable[base.PromptRecord], *, fixture_kind: str, stage: str | None = None) -> dict[str, object]:
    stage = stage or _ACTIVE_STAGE
    protocol_id = protocol_for(stage)
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
        "protocol_id": protocol_id,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "fixture_kind": fixture_kind,
        "prompt_rules_version": "v19-causal-value-qualification-synthetic-v1",
        "count": len(rows),
        "stratum_counts": {name: sum(row["stratum"] == name for row in rows) for name in sorted({row["stratum"] for row in rows})},
        "example_set_sha256": hashlib.sha256("\n".join(sorted(record_digests)).encode("ascii")).hexdigest(),
        "records": rows,
    }


_EXTRA_DISJOINT_PAYLOADS: tuple[dict[str, object], ...] = ()


def set_extra_disjoint_payloads(payloads: Iterable[dict[str, object]]) -> None:
    global _EXTRA_DISJOINT_PAYLOADS
    _EXTRA_DISJOINT_PAYLOADS = tuple(payloads)


def validate_partition_disjointness(*payloads: dict[str, object]) -> None:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_content: set[str] = set()
    for payload in (*_EXTRA_DISJOINT_PAYLOADS, *payloads):
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
