"""Fresh custody-safe fixture namespace for replacement-loss-v8.

State slice: theseus-lab-sandbox-v1. Fixture text remains in external custody.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Iterable

import replacement_loss_v3_fixtures as v3


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v8-scale-bandb-natural-text"
SEEDS = (20261901, 20261902, 20261903)
VISIBLE_SEED = 20261900
VISIBLE_COUNT = 32
ADAPTATION_PROMPTS = {"ordinary": 64, "compositional": 32, "adversarial": 32}
CHALLENGES_PER_STRATUM = 20
STRATA = ("ordinary", "compositional", "adversarial")

# Natural-text adaptation mode (r5). Template mode (default) is unchanged from earlier packets.
ADAPTATION_MODE = "template"
NATURAL_PROMPTS = 128
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
WIKITEXT_SHA256 = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
WIKITEXT_ENV = "THESEUS_WIKITEXT_PARQUET"
NATURAL_WORDS = 45
NATURAL_MIN_WORDS = 60


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
                input_text=f"V8SB {partition} {seed}. {record.input_text}",
            )
        )
    return result


def generate_visible_fixture() -> list[Any]:
    rows = v3._unique_records("v8sb-visible", VISIBLE_SEED, VISIBLE_COUNT, "visible")
    return _namespace(rows, "v8sb-visible", VISIBLE_SEED)


def eligible_paragraphs(lines: Iterable[str]) -> list[str]:
    """Non-heading paragraphs with at least NATURAL_MIN_WORDS words, whitespace-normalized and de-duplicated."""

    seen: set[str] = set()
    result = []
    for line in lines:
        text = " ".join(str(line).split())
        if len(text.split()) < NATURAL_MIN_WORDS or text.startswith("="):
            continue
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def select_natural_prompts(paragraphs: list[str], seed_index: int, count: int) -> list[str]:
    """Deterministically pick `count` paragraphs from this seed's disjoint third of the pool."""

    seed = SEEDS[seed_index]
    pool = [
        text
        for text in paragraphs
        if int(hashlib.sha256(f"{PROTOCOL_ID}\0pool\0{text}".encode("utf-8")).hexdigest(), 16) % len(SEEDS) == seed_index
    ]
    ranked = sorted(
        pool,
        key=lambda text: hashlib.sha256(f"{PROTOCOL_ID}\0{seed}\0{text}".encode("utf-8")).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError("natural-text pool is smaller than the requested prompt count")
    return [" ".join(text.split()[:NATURAL_WORDS]) for text in ranked[:count]]


def _load_wikitext_lines() -> list[str]:
    import os
    from pathlib import Path

    path = os.environ.get(WIKITEXT_ENV)
    if not path:
        raise RuntimeError(f"set {WIKITEXT_ENV} to the pinned wikitext-2 train parquet")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != WIKITEXT_SHA256:
        raise ValueError("wikitext parquet does not match the pinned sha256")
    import pyarrow.parquet as pq

    return [str(value) for value in pq.read_table(path).column("text").to_pylist()]


def natural_records(seed_index: int, count: int, lines: Iterable[str] | None = None) -> list[Any]:
    paragraphs = eligible_paragraphs(_load_wikitext_lines() if lines is None else lines)
    rows = []
    for index, prompt in enumerate(select_natural_prompts(paragraphs, seed_index, count)):
        material = f"{PROTOCOL_ID}\0natural\0{SEEDS[seed_index]}\0{index}\0{prompt}"
        example_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        rows.append(v3.PromptRecord(example_id, f"v8sb-natural-{seed_index}", "natural", prompt, " Continued."))
    return rows


def generate_adaptation_fixture(seed_index: int) -> list[Any]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen v8 seed set")
    if ADAPTATION_MODE == "natural":
        return natural_records(seed_index, NATURAL_PROMPTS)
    seed = SEEDS[seed_index]
    rows = []
    for offset, stratum in enumerate(STRATA):
        source = v3._unique_records(
            f"v8sb-adaptation-{seed_index}-{stratum}",
            seed + (offset + 1) * 10_000,
            ADAPTATION_PROMPTS[stratum],
            stratum,
        )
        rows.extend(_namespace(source, f"v8sb-adaptation-{seed_index}-{stratum}", seed))
    return rows


def generate_challenge_fixture(seed_index: int, secret_seed: bytes | None = None) -> list[Any]:
    if seed_index not in range(len(SEEDS)):
        raise ValueError("seed_index is outside the frozen v8 seed set")
    source = v3.generate_challenge_fixture(seed_index, secret_seed=secret_seed)
    return _namespace(source, f"v8sb-challenge-{seed_index}", SEEDS[seed_index])


def fixture_payload(records: Iterable[Any], *, fixture_kind: str) -> dict[str, Any]:
    payload = v3.fixture_payload(records, fixture_kind=f"v8sb-{fixture_kind}")
    payload["protocol_id"] = PROTOCOL_ID
    payload["prompt_rules_version"] = "v8sb-namespaced-synthetic-template-v1"
    return payload


def validate_partition_disjointness(*fixtures: dict[str, Any]) -> None:
    v3.validate_partition_disjointness(*fixtures)


def assert_stratum_counts(payload: dict[str, Any], expected: dict[str, int]) -> None:
    actual = payload.get("stratum_counts")
    if actual != expected:
        raise ValueError(f"fixture stratum counts differ from v8 preregistration: {actual}")
