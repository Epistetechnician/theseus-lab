"""Fixture namespace for replacement-loss-v8-scale-r6 (natural-text challenge arm).

State slice: theseus-lab-sandbox-v1. Fixture text remains in external custody.
Adaptation prompts reuse the r5 natural-text selection verbatim so N128 here is the
same prompt set as r5 N128; challenges can additionally be natural text.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import replace
from typing import Any, Iterable

import replacement_loss_v3_fixtures as v3
import replacement_loss_v8s5_fixtures as r5


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v8-scale-r6-natural-challenge"
SEEDS = r5.SEEDS
VISIBLE_SEED = 20261900
VISIBLE_COUNT = 32
ADAPTATION_PROMPTS = {"ordinary": 64, "compositional": 32, "adversarial": 32}
CHALLENGES_PER_STRATUM = 20
STRATA = ("ordinary", "compositional", "adversarial")

# Arm-controlled modes. Defaults reproduce the template-only behaviour.
ADAPTATION_MODE = "template"
CHALLENGE_MODE = "template"
NATURAL_PROMPTS = 128
# Largest natural adaptation arm in any packet; challenge prompts are drawn only from beyond it.
MAX_NATURAL_ADAPTATION = 512

WIKITEXT_ENV = r5.WIKITEXT_ENV
NATURAL_WORDS = r5.NATURAL_WORDS
NATURAL_MIN_WORDS = r5.NATURAL_MIN_WORDS
eligible_paragraphs = r5.eligible_paragraphs
select_natural_prompts = r5.select_natural_prompts
natural_records = r5.natural_records


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
                input_text=f"V8S6 {partition} {seed}. {record.input_text}",
            )
        )
    return result


def generate_visible_fixture() -> list[Any]:
    rows = v3._unique_records("v8s6-visible", VISIBLE_SEED, VISIBLE_COUNT, "visible")
    return _namespace(rows, "v8s6-visible", VISIBLE_SEED)


def _pool(paragraphs: list[str], seed_index: int) -> list[str]:
    """This seed's disjoint third of the pool, using r5's hash so r5 selections are reproduced."""

    return [
        text
        for text in paragraphs
        if int(hashlib.sha256(f"{r5.PROTOCOL_ID}\0pool\0{text}".encode("utf-8")).hexdigest(), 16) % len(SEEDS) == seed_index
    ]


def select_natural_challenge_prompts(
    paragraphs: list[str], seed_index: int, entropy: bytes, per_stratum: int = CHALLENGES_PER_STRATUM
) -> list[tuple[str, str]]:
    """Pick held-out (stratum, prompt) pairs from this seed's third, outside the adaptation ranking.

    The first MAX_NATURAL_ADAPTATION paragraphs in r5's adaptation ranking are reserved for
    adaptation (any arm); the rest are ranked by post-freeze secret entropy.
    """

    seed = SEEDS[seed_index]
    ranked_adaptation = sorted(
        _pool(paragraphs, seed_index),
        key=lambda text: hashlib.sha256(f"{r5.PROTOCOL_ID}\0{seed}\0{text}".encode("utf-8")).hexdigest(),
    )
    remaining = ranked_adaptation[MAX_NATURAL_ADAPTATION:]
    ranked = sorted(
        remaining,
        key=lambda text: hashlib.sha256(entropy + str(seed_index).encode("ascii") + text.encode("utf-8")).hexdigest(),
    )
    need = per_stratum * len(STRATA)
    if len(ranked) < need:
        raise ValueError("natural-text challenge pool is smaller than the requested prompt count")
    picks = [" ".join(text.split()[:NATURAL_WORDS]) for text in ranked[:need]]
    return [(STRATA[index % len(STRATA)], prompt) for index, prompt in enumerate(picks)]


def natural_challenge_records(
    seed_index: int, entropy: bytes, lines: Iterable[str] | None = None
) -> list[Any]:
    paragraphs = r5.eligible_paragraphs(r5._load_wikitext_lines() if lines is None else lines)
    rows = []
    for index, (stratum, prompt) in enumerate(select_natural_challenge_prompts(paragraphs, seed_index, entropy)):
        material = f"{PROTOCOL_ID}\0challenge-natural\0{SEEDS[seed_index]}\0{index}\0{prompt}"
        example_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        rows.append(v3.PromptRecord(example_id, f"v8s6-challenge-natural-{seed_index}", stratum, prompt, " Continued."))
    return rows


def generate_adaptation_fixture(seed_index: int) -> list[Any]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen v8 seed set")
    if ADAPTATION_MODE == "natural":
        # Reuse r5's records exactly (same ids and prompts as r5 N128/N256/N512).
        return r5.natural_records(seed_index, NATURAL_PROMPTS)
    seed = SEEDS[seed_index]
    rows = []
    for offset, stratum in enumerate(STRATA):
        source = v3._unique_records(
            f"v8s6-adaptation-{seed_index}-{stratum}",
            seed + (offset + 1) * 10_000,
            ADAPTATION_PROMPTS[stratum],
            stratum,
        )
        rows.extend(_namespace(source, f"v8s6-adaptation-{seed_index}-{stratum}", seed))
    return rows


def generate_challenge_fixture(seed_index: int, secret_seed: bytes | None = None) -> list[Any]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen v8 seed set")
    if CHALLENGE_MODE == "natural":
        entropy = secrets.token_bytes(32) if secret_seed is None else bytes(secret_seed)
        if len(entropy) < 16:
            raise ValueError("challenge entropy must contain at least 128 bits")
        return natural_challenge_records(seed_index, entropy)
    source = v3.generate_challenge_fixture(seed_index, secret_seed=secret_seed)
    return _namespace(source, f"v8s6-challenge-{seed_index}", SEEDS[seed_index])


def fixture_payload(records: Iterable[Any], *, fixture_kind: str) -> dict[str, Any]:
    payload = v3.fixture_payload(records, fixture_kind=f"v8s6-{fixture_kind}")
    payload["protocol_id"] = PROTOCOL_ID
    payload["prompt_rules_version"] = "v8s6-namespaced-synthetic-template-or-wikitext-v1"
    return payload


def validate_partition_disjointness(*fixtures: dict[str, Any]) -> None:
    v3.validate_partition_disjointness(*fixtures)


def assert_stratum_counts(payload: dict[str, Any], expected: dict[str, int]) -> None:
    actual = payload.get("stratum_counts")
    if actual != expected:
        raise ValueError(f"fixture stratum counts differ from v8 preregistration: {actual}")
