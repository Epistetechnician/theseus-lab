"""Fresh adaptation-only fixtures for the V15 capacity/optimization screen.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v15-capacity-optimization-diagnostic-v1.
Prompt payloads are written only to external owner custody during execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Iterable

import replacement_loss_v3_fixtures as base


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v15-capacity-optimization-diagnostic-v1"
PROTOCOL_ID = "theseus-replacement-loss-v15-capacity-optimization-diagnostic"
PROMPT_RULES_VERSION = "v15-adaptation-capacity-optimization-synthetic-v1"
VISIBLE_SEED = 202610200
SEEDS = (202610201, 202610202, 202610203)
VISIBLE_COUNT = 32
ADAPTATION_PER_STRATUM = 32
VALIDATION_PER_STRATUM = 8
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
            input_text=f"V15 capacity optimization diagnostic {partition} seed {seed}. {row.input_text}",
        ))
    return result


def generate_visible_fixture() -> list[base.PromptRecord]:
    rows = base._unique_records("v15-capacity-visible", VISIBLE_SEED, VISIBLE_COUNT, "visible")
    return _namespace(rows, "v15-visible", VISIBLE_SEED)


def generate_adaptation_fixture(seed_index: int) -> list[base.PromptRecord]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen V15 seed set")
    seed = SEEDS[seed_index]
    rows = []
    for stratum_index, stratum in enumerate(STRATA):
        partition = f"v15-adaptation-{seed_index}-{stratum}"
        source_seed = seed + (stratum_index + 1) * 1_000_000
        source = base._unique_records(partition, source_seed, ADAPTATION_PER_STRATUM, stratum)
        rows.extend(_namespace(source, partition, seed))
    return rows


def fixture_payload(records: Iterable[base.PromptRecord], *, fixture_kind: str) -> dict[str, object]:
    rows = [asdict(row) for row in records]
    if not rows:
        raise ValueError("fixture must contain at least one row")
    ids = [row["example_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("fixture example IDs must be unique")
    digests = [
        hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
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
