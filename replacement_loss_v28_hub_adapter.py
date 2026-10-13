"""V28 public Hub adapter over V26's frozen selection and download rules.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v28-fresh-checkpoint-recovery-v1. V28 doubles only
the predeclared metadata result bound to 10,000 per existing query; model
eligibility and download validation stay byte-bound to the V26 adapter.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import replacement_loss_v26_hub_adapter as frozen_adapter


PROTOCOL_ID = "theseus-replacement-loss-v28-fresh-checkpoint-recovery"
V28_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v28")
SEARCH_LIMIT = 10_000
MAX_PARAMETERS = frozen_adapter.MAX_PARAMETERS
MAX_MODEL_BYTES = frozen_adapter.MAX_MODEL_BYTES
MAX_TOTAL_BYTES = frozen_adapter.MAX_TOTAL_BYTES
FAMILY_QUERIES = frozen_adapter.FAMILY_QUERIES
RESULT_PROPERTIES = frozen_adapter.RESULT_PROPERTIES

frozen_adapter.PROTOCOL_ID = PROTOCOL_ID
frozen_adapter.SEARCH_LIMIT = SEARCH_LIMIT
os.environ["HF_HOME"] = str(V28_ROOT / "hf-home")
os.environ["HF_HUB_CACHE"] = str(V28_ROOT / "hub-cache")
os.environ["HF_ASSETS_CACHE"] = str(V28_ROOT / "assets-cache")
os.environ["HF_XET_CACHE"] = str(V28_ROOT / "xet-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _relabel(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _relabel(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_relabel(child) for child in value]
    if isinstance(value, str):
        return value.replace("before_v26", "before_v28").replace("V26", "V28")
    return value


def screen_public_hub(
    excluded_model_ids: set[str],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return _relabel(frozen_adapter.screen_public_hub(excluded_model_ids, on_progress=on_progress))


def download_selected(
    custody_root: str | Path,
    screen: dict[str, Any],
    *,
    allow_public_hub_downloads: bool,
) -> list[dict[str, Any]]:
    return _relabel(frozen_adapter.download_selected(
        custody_root,
        screen,
        allow_public_hub_downloads=allow_public_hub_downloads,
    ))
