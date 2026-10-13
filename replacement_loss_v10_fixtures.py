"""Fresh synthetic fixture rules for replacement-loss-v10.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v10-opt-causal-fit-v1. Prompt text is generated and
retained only in external owner custody.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, replace
from typing import Iterable

import replacement_loss_v3_fixtures as v3


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v10-opt-causal-fit-v1"
PROTOCOL_ID = "theseus-replacement-loss-v10-opt-causal-fit"
PROMPT_RULES_VERSION = "v10-namespaced-synthetic-template-v1"
VISIBLE_SEED = 202609290
SEEDS = (202609291, 202609292, 202609293)
VISIBLE_COUNT = 32
ADAPTATION_PER_STRATUM = 32
CHALLENGES_PER_STRATUM = 20
STRATA = ("ordinary", "compositional", "adversarial")


def _namespace(records: Iterable[v3.PromptRecord], partition: str, seed: int) -> list[v3.PromptRecord]:
    result = []
    for row in records:
        material = f"{PROTOCOL_ID}\0{partition}\0{seed}\0{row.example_id}"
        example_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        result.append(replace(
            row,
            example_id=example_id,
            partition=partition,
            input_text=f"V10 {partition} seed {seed}. {row.input_text}",
        ))
    return result


def generate_visible_fixture() -> list[v3.PromptRecord]:
    records = v3._unique_records("v10-visible", VISIBLE_SEED, VISIBLE_COUNT, "visible")
    return _namespace(records, "v10-visible", VISIBLE_SEED)


def generate_adaptation_fixture(seed_index: int) -> list[v3.PromptRecord]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen v10 seed set")
    seed = SEEDS[seed_index]
    result = []
    for stratum_index, stratum in enumerate(STRATA):
        partition = f"v10-adaptation-{seed_index}-{stratum}"
        base = seed + (stratum_index + 1) * 1_000_000
        records = v3._unique_records(partition, base, ADAPTATION_PER_STRATUM, stratum)
        result.extend(_namespace(records, partition, seed))
    return result


def generate_challenge_fixture(seed_index: int, *, entropy: bytes | None = None) -> tuple[list[v3.PromptRecord], bytes]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen v10 seed set")
    material = secrets.token_bytes(32) if entropy is None else bytes(entropy)
    if len(material) < 32:
        raise ValueError("challenge entropy must contain at least 256 bits")
    seed = int.from_bytes(hashlib.sha256(material + bytes([seed_index])).digest()[:8], "big")
    result = []
    for stratum in STRATA:
        partition = f"v10-challenge-{seed_index}-{stratum}"
        stratum_seed = seed ^ int.from_bytes(hashlib.sha256(stratum.encode()).digest()[:8], "big")
        records = v3._unique_records(partition, stratum_seed, CHALLENGES_PER_STRATUM, stratum)
        result.extend(_namespace(records, partition, SEEDS[seed_index]))
    return result, material


def fixture_payload(records: Iterable[v3.PromptRecord], *, fixture_kind: str) -> dict[str, object]:
    rows = [asdict(row) for row in records]
    if not rows:
        raise ValueError("fixture must contain at least one row")
    ids = [row["example_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("fixture example IDs must be unique")
    digests = [hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest() for row in rows]
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "fixture_kind": fixture_kind,
        "prompt_rules_version": PROMPT_RULES_VERSION,
        "count": len(rows),
        "stratum_counts": {name: sum(row["stratum"] == name for row in rows) for name in sorted({row["stratum"] for row in rows})},
        "example_set_sha256": hashlib.sha256("\n".join(sorted(digests)).encode("ascii")).hexdigest(),
        "records": rows,
    }


def validate_partition_disjointness(*payloads: dict[str, object]) -> None:
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_contents: set[str] = set()
    for payload in payloads:
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError("fixture records must be a list")
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("fixture record must be an object")
            example_id = str(row["example_id"])
            prompt = str(row["input_text"])
            content = prompt + "\0" + str(row["continuation"])
            if example_id in seen_ids or prompt in seen_prompts or content in seen_contents:
                raise ValueError("fixtures overlap by ID or exact prompt content")
            seen_ids.add(example_id)
            seen_prompts.add(prompt)
            seen_contents.add(content)
