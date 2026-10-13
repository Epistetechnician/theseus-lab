"""V27 label adapter over the frozen V26 public-Hub selection algorithm.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v27-fresh-checkpoint-recovery-v1. Selection rules
and download validation are byte-bound through the V26 adapter; only the
protocol label and external cache root differ.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import replacement_loss_v26_hub_adapter as frozen_adapter


PROTOCOL_ID = "theseus-replacement-loss-v27-fresh-checkpoint-recovery"
V27_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v27")
SEARCH_LIMIT = frozen_adapter.SEARCH_LIMIT
MAX_PARAMETERS = frozen_adapter.MAX_PARAMETERS
MAX_MODEL_BYTES = frozen_adapter.MAX_MODEL_BYTES
MAX_TOTAL_BYTES = frozen_adapter.MAX_TOTAL_BYTES
FAMILY_QUERIES = frozen_adapter.FAMILY_QUERIES
RESULT_PROPERTIES = frozen_adapter.RESULT_PROPERTIES

frozen_adapter.PROTOCOL_ID = PROTOCOL_ID
os.environ["HF_HOME"] = str(V27_ROOT / "hf-home")
os.environ["HF_HUB_CACHE"] = str(V27_ROOT / "hub-cache")
os.environ["HF_ASSETS_CACHE"] = str(V27_ROOT / "assets-cache")
os.environ["HF_XET_CACHE"] = str(V27_ROOT / "xet-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _relabel(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _relabel(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_relabel(child) for child in value]
    if isinstance(value, str):
        return value.replace("before_v26", "before_v27").replace("V26", "V27")
    return value


def screen_public_hub(
    excluded_model_ids: set[str],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    result = frozen_adapter.screen_public_hub(excluded_model_ids, on_progress=on_progress)
    return _relabel(result)


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
