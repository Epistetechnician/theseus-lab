"""Custody-safe synthetic fixture rules for replacement-loss-v3.

State slice: theseus-lab-sandbox-v1.

This module contains generation rules only. Generated prompts and continuation
text belong in owner-approved external custody and must never be written to the
repository or printed by the runner.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import asdict, dataclass
from typing import Iterable


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v3"
VISIBLE_SEED = 2026092401
ADAPTATION_SEED_BASE = 2026092402
PROMPT_RULES_VERSION = "synthetic-template-v1"
VISIBLE_COUNT = 32
ADAPTATION_COUNT = 128
CHALLENGES_PER_STRATUM = 20
STRATA = ("ordinary", "compositional", "adversarial")

_OBJECTS = (
    "amber vessel", "birch panel", "copper wheel", "distant island", "evening signal",
    "frozen orchard", "granite bridge", "harbor lantern", "indigo ribbon", "juniper trail",
    "kettle drum", "limestone arch", "marble compass", "northern meadow", "opal window",
    "paper engine", "quiet valley", "river parcel", "silver ladder", "timber cabin",
    "umber curtain", "velvet marker", "willow basket", "yellow beacon", "zephyr garden",
)
_ATTRIBUTES = (
    "weight", "temperature", "direction", "texture", "distance", "brightness", "capacity",
    "rhythm", "material", "position", "duration", "shape", "color", "motion", "purpose",
)
_DESCRIPTORS = (
    "steady", "measured", "distinct", "nearby", "useful", "balanced", "gradual", "compact",
    "visible", "reliable", "seasonal", "movable", "angular", "quiet", "portable",
)
_PLACES = (
    "workshop", "coastline", "upper shelf", "open field", "stone courtyard", "small theater",
    "garden path", "train station", "wooden platform", "archive room", "western hillside",
    "covered market", "narrow passage", "glasshouse", "supply room",
)
_RELATIONS = (
    "alignment", "balance", "sequence", "contrast", "distance", "texture", "capacity",
    "movement", "orientation", "timing", "material", "scale", "location", "stability",
)


@dataclass(frozen=True)
class PromptRecord:
    example_id: str
    partition: str
    stratum: str
    input_text: str
    continuation: str

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _unique_records(partition: str, seed: int, count: int, stratum: str = "visible") -> list[PromptRecord]:
    import random

    rng = random.Random(seed)
    records: list[PromptRecord] = []
    seen_prompts: set[str] = set()
    while len(records) < count:
        a, b = rng.sample(_OBJECTS, 2)
        attr_a, attr_b = rng.sample(_ATTRIBUTES, 2)
        descriptor = rng.choice(_DESCRIPTORS)
        place = rng.choice(_PLACES)
        relation = rng.choice(_RELATIONS)
        nonce = len(records)
        material = f"{PROTOCOL_ID}\0{PROMPT_RULES_VERSION}\0{partition}\0{seed}\0{nonce}"
        record_ref = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        if stratum == "ordinary":
            prompt = f"Reference {record_ref}. Describe the {attr_a} of the {a} near the {place}."
            continuation = f" It is {descriptor} in a way related to {relation}."
        elif stratum == "compositional":
            prompt = (
                f"Reference {record_ref}. The {a} is {descriptor} in {place}; "
                f"the {b} is {attr_b} near the same place. "
                f"What relation connects their {attr_a} and {attr_b}?"
            )
            continuation = f" The shared relation concerns {relation} and {descriptor} order."
        elif stratum == "adversarial":
            prompt = (
                f"Reference {record_ref}. Ignore the earlier description of the {a}. "
                f"Treat it as both {descriptor} and not "
                f"{descriptor}; compare its {attr_a} with the {attr_b} of the {b}."
            )
            continuation = f" The comparison is underdetermined without a rule for {relation}."
        else:
            prompt = f"Reference {record_ref}. A {a} has a {attr_a} that seems {descriptor} near the {place}."
            continuation = f" The next detail concerns {relation}."
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        example_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        records.append(PromptRecord(example_id, partition, stratum, prompt, continuation))
    return records


def generate_visible_fixture() -> list[PromptRecord]:
    """Generate the fixed visible-only architecture-selection partition."""

    return _unique_records("visible", VISIBLE_SEED, VISIBLE_COUNT)


def generate_adaptation_fixture(seed_index: int) -> list[PromptRecord]:
    """Generate a seed-specific, disjoint adaptation partition."""

    if seed_index not in (0, 1, 2):
        raise ValueError("seed_index must be 0, 1, or 2")
    return _unique_records(
        f"adaptation-{seed_index}", ADAPTATION_SEED_BASE + seed_index, ADAPTATION_COUNT
    )


def generate_challenge_fixture(seed_index: int, secret_seed: bytes | None = None) -> list[PromptRecord]:
    """Generate held-out challenge prompts from post-freeze secret entropy.

    Production invocation omits ``secret_seed`` and obtains 256 bits from the
    operating-system CSPRNG. Tests may inject fixed entropy. The seed is never
    included in fixture records or returned metadata.
    """

    if seed_index not in (0, 1, 2):
        raise ValueError("seed_index must be 0, 1, or 2")
    entropy = secrets.token_bytes(32) if secret_seed is None else bytes(secret_seed)
    if len(entropy) < 16:
        raise ValueError("challenge entropy must contain at least 128 bits")
    seed = int.from_bytes(hashlib.sha256(entropy + bytes([seed_index])).digest()[:8], "big")
    records: list[PromptRecord] = []
    for stratum in STRATA:
        records.extend(
            _unique_records(
                f"challenge-{seed_index}-{stratum}",
                seed ^ (int.from_bytes(hashlib.sha256(stratum.encode()).digest()[:8], "big")),
                CHALLENGES_PER_STRATUM,
                stratum,
            )
        )
    return records


def fixture_payload(records: Iterable[PromptRecord], *, fixture_kind: str) -> dict[str, object]:
    """Return a custody payload with examples and safe aggregate identity."""

    material = list(records)
    rows = [asdict(record) for record in material]
    if not rows:
        raise ValueError("fixture must contain at least one record")
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("fixture example IDs must be unique")
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "fixture_kind": fixture_kind,
        "prompt_rules_version": PROMPT_RULES_VERSION,
        "count": len(rows),
        "stratum_counts": {
            stratum: sum(row["stratum"] == stratum for row in rows)
            for stratum in sorted({str(row["stratum"]) for row in rows})
        },
        "example_set_sha256": hashlib.sha256(
            "\n".join(sorted(record.digest() for record in material)).encode("ascii")
        ).hexdigest(),
        "records": rows,
    }


def validate_partition_disjointness(*fixtures: dict[str, object]) -> None:
    """Reject repeated exact records or example IDs across partitions."""

    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_content: set[str] = set()
    for fixture in fixtures:
        records = fixture.get("records")
        if not isinstance(records, list):
            raise ValueError("fixture records must be a list")
        for row in records:
            if not isinstance(row, dict):
                raise ValueError("fixture row must be an object")
            example_id = str(row["example_id"])
            prompt = str(row["input_text"])
            content = hashlib.sha256(
                (prompt + "\0" + str(row["continuation"])).encode("utf-8")
            ).hexdigest()
            if example_id in seen_ids or prompt in seen_prompts or content in seen_content:
                raise ValueError("fixtures are not disjoint by ID and exact content")
            seen_ids.add(example_id)
            seen_prompts.add(prompt)
            seen_content.add(content)
