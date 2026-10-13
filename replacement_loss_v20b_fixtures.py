"""Fresh fixture namespace for the frozen V20B continuation.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v20b-low-rank-causal-residual-continuation-v1.
"""

from __future__ import annotations

import replacement_loss_v20_fixtures as base


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v20b-low-rank-causal-residual-continuation-v1"
DEV_PROTOCOL_ID = "theseus-replacement-loss-v20b-development"
CONFIRM_PROTOCOL_ID = "theseus-replacement-loss-v20b-confirmation"
PROMPT_RULES_VERSION = "v20b-low-rank-causal-residual-synthetic-v1"
STRATA = base.STRATA
DEVELOPMENT_SEEDS = (202610811, 202610812, 202610813)
CONFIRMATION_SEEDS = (202610911, 202610912, 202610913)
ADAPTATION_PER_STRATUM = base.ADAPTATION_PER_STRATUM
FIT_PER_STRATUM = base.FIT_PER_STRATUM
VALIDATION_PER_STRATUM = base.VALIDATION_PER_STRATUM
CHALLENGES_PER_STRATUM = base.CHALLENGES_PER_STRATUM
PER_STRATUM_COUNT = ADAPTATION_PER_STRATUM
SEEDS = DEVELOPMENT_SEEDS
PROTOCOL_ID = DEV_PROTOCOL_ID
_ACTIVE_STAGE = "development"

base.PROTOCOL_SLICE = PROTOCOL_SLICE
base.DEV_PROTOCOL_ID = DEV_PROTOCOL_ID
base.CONFIRM_PROTOCOL_ID = CONFIRM_PROTOCOL_ID
base.PROMPT_RULES_VERSION = PROMPT_RULES_VERSION
base.DEVELOPMENT_SEEDS = DEVELOPMENT_SEEDS
base.CONFIRMATION_SEEDS = CONFIRMATION_SEEDS


def activate_stage(stage: str) -> None:
    global _ACTIVE_STAGE, SEEDS, PROTOCOL_ID
    if stage not in {"development", "confirmation"}:
        raise ValueError("stage must be development or confirmation")
    _ACTIVE_STAGE = stage
    base.activate_stage(stage)
    SEEDS = DEVELOPMENT_SEEDS if stage == "development" else CONFIRMATION_SEEDS
    PROTOCOL_ID = DEV_PROTOCOL_ID if stage == "development" else CONFIRM_PROTOCOL_ID


def seeds_for(stage: str):
    return DEVELOPMENT_SEEDS if stage == "development" else CONFIRMATION_SEEDS if stage == "confirmation" else base.seeds_for(stage)


def protocol_for(stage: str) -> str:
    return DEV_PROTOCOL_ID if stage == "development" else CONFIRM_PROTOCOL_ID if stage == "confirmation" else base.protocol_for(stage)


def generate_adaptation_fixture(stage: str, seed_index: int):
    return base.generate_adaptation_fixture(stage, seed_index)


def generate_challenge_fixture(seed_index: int, secret_seed: bytes, *, stage: str | None = None):
    return base.generate_challenge_fixture(seed_index, secret_seed, stage=stage)


def fixture_payload(records, *, fixture_kind: str, stage: str | None = None):
    return base.fixture_payload(records, fixture_kind=fixture_kind, stage=stage)


def set_extra_disjoint_payloads(payloads):
    base.set_extra_disjoint_payloads(payloads)


def validate_partition_disjointness(*payloads):
    return base.validate_partition_disjointness(*payloads)
