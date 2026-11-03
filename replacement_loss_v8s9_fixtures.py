"""Fixture namespace for replacement-loss-v8-scale-r9 (news-domain replication).

State slice: theseus-lab-sandbox-v1. Fixture text remains in external custody.
Same selection mechanics as r5/r6, but the text pool is CNN/DailyMail news articles
(pinned revision and file sha256) instead of wikitext-2.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import replace
from typing import Any, Iterable

import replacement_loss_v3_fixtures as v3


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v8-scale-r9-news-domain"
SEEDS = (20261801, 20261802, 20261803)
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

# Pinned news corpus: abisee/cnn_dailymail, config 3.0.0, validation split (one article per row).
NEWS_REVISION = "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
NEWS_FILE = "3.0.0/validation-00000-of-00001.parquet"
NEWS_SHA256 = "65a5ccce932b08f050114ce6e3c39355d563fa24f194052bb2f27d1c6c499c91"
NEWS_COLUMN = "article"
NEWS_ENV = "THESEUS_CNNDM_PARQUET"
NATURAL_WORDS = 45
NATURAL_MIN_WORDS = 60
POOL_PROTOCOL_ID = PROTOCOL_ID


def eligible_paragraphs(lines: Iterable[str]) -> list[str]:
    """Same filter as r5: whitespace-normalized, at least NATURAL_MIN_WORDS words, not a '=' heading, de-duplicated.

    For this corpus one row (article) is one paragraph; no newline splitting is applied.
    """

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
    ranked = sorted(
        _pool(paragraphs, seed_index),
        key=lambda text: hashlib.sha256(f"{PROTOCOL_ID}\0{seed}\0{text}".encode("utf-8")).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError("natural-text pool is smaller than the requested prompt count")
    return [" ".join(text.split()[:NATURAL_WORDS]) for text in ranked[:count]]


def _load_news_lines() -> list[str]:
    import os
    from pathlib import Path

    path = os.environ.get(NEWS_ENV)
    if not path:
        raise RuntimeError(f"set {NEWS_ENV} to the pinned cnn_dailymail validation parquet")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != NEWS_SHA256:
        raise ValueError("news parquet does not match the pinned sha256")
    import pyarrow.parquet as pq

    return [str(value) for value in pq.read_table(path).column(NEWS_COLUMN).to_pylist()]


def natural_records(seed_index: int, count: int, lines: Iterable[str] | None = None) -> list[Any]:
    paragraphs = eligible_paragraphs(_load_news_lines() if lines is None else lines)
    rows = []
    for index, prompt in enumerate(select_natural_prompts(paragraphs, seed_index, count)):
        material = f"{PROTOCOL_ID}\0natural\0{SEEDS[seed_index]}\0{index}\0{prompt}"
        example_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        rows.append(v3.PromptRecord(example_id, f"v8s9-natural-{seed_index}", "natural", prompt, " Continued."))
    return rows


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
                input_text=f"V8S9 {partition} {seed}. {record.input_text}",
            )
        )
    return result


def generate_visible_fixture() -> list[Any]:
    rows = v3._unique_records("v8s9-visible", VISIBLE_SEED, VISIBLE_COUNT, "visible")
    return _namespace(rows, "v8s9-visible", VISIBLE_SEED)


def _pool(paragraphs: list[str], seed_index: int) -> list[str]:
    """This seed's disjoint third of the pool."""

    return [
        text
        for text in paragraphs
        if int(hashlib.sha256(f"{PROTOCOL_ID}\0pool\0{text}".encode("utf-8")).hexdigest(), 16) % len(SEEDS) == seed_index
    ]


# Challenge prompts already issued in this process. v3 validates every challenge payload
# (all families and seeds) as mutually disjoint, and the three families share each seed's pool.
USED_CHALLENGE_PROMPTS: set[str] = set()


def select_natural_challenge_prompts(
    paragraphs: list[str],
    seed_index: int,
    entropy: bytes,
    per_stratum: int = CHALLENGES_PER_STRATUM,
    exclude: set[str] | frozenset[str] = frozenset(),
) -> list[tuple[str, str]]:
    """Pick held-out (stratum, prompt) pairs from this seed's third, outside the adaptation ranking.

    The first MAX_NATURAL_ADAPTATION paragraphs in the adaptation ranking are reserved for
    adaptation (any arm); the rest are ranked by post-freeze secret entropy.
    """

    seed = SEEDS[seed_index]
    ranked_adaptation = sorted(
        _pool(paragraphs, seed_index),
        key=lambda text: hashlib.sha256(f"{PROTOCOL_ID}\0{seed}\0{text}".encode("utf-8")).hexdigest(),
    )
    remaining = [text for text in ranked_adaptation[MAX_NATURAL_ADAPTATION:] if " ".join(text.split()[:NATURAL_WORDS]) not in exclude]
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
    paragraphs = eligible_paragraphs(_load_news_lines() if lines is None else lines)
    rows = []
    picks = select_natural_challenge_prompts(paragraphs, seed_index, entropy, exclude=USED_CHALLENGE_PROMPTS)
    USED_CHALLENGE_PROMPTS.update(prompt for _, prompt in picks)
    for index, (stratum, prompt) in enumerate(picks):
        material = f"{PROTOCOL_ID}\0challenge-natural\0{SEEDS[seed_index]}\0{index}\0{prompt}"
        example_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        rows.append(v3.PromptRecord(example_id, f"v8s9-challenge-natural-{seed_index}", stratum, prompt, " Continued."))
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
            f"v8s9-adaptation-{seed_index}-{stratum}",
            seed + (offset + 1) * 10_000,
            ADAPTATION_PROMPTS[stratum],
            stratum,
        )
        rows.extend(_namespace(source, f"v8s9-adaptation-{seed_index}-{stratum}", seed))
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
    return _namespace(source, f"v8s9-challenge-{seed_index}", SEEDS[seed_index])


def fixture_payload(records: Iterable[Any], *, fixture_kind: str) -> dict[str, Any]:
    payload = v3.fixture_payload(records, fixture_kind=f"v8s9-{fixture_kind}")
    payload["protocol_id"] = PROTOCOL_ID
    payload["prompt_rules_version"] = "v8s9-namespaced-synthetic-template-or-news-v1"
    return payload


def validate_partition_disjointness(*fixtures: dict[str, Any]) -> None:
    v3.validate_partition_disjointness(*fixtures)


def assert_stratum_counts(payload: dict[str, Any], expected: dict[str, int]) -> None:
    actual = payload.get("stratum_counts")
    if actual != expected:
        raise ValueError(f"fixture stratum counts differ from v8 preregistration: {actual}")
