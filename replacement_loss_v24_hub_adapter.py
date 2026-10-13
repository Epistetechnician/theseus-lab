"""Client-compatible bounded public Hub screen for V24.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v24-public-checkpoint-acquisition-v1. Uses V23's
frozen exact-metadata candidate validator while issuing only arguments
supported by the pinned 0.36.2 client.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Keep metadata downloads made by the pinned V23 checker inside V24 custody.
# Cache paths are read during the huggingface_hub import below.
_V24_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v24")
os.environ["HF_HOME"] = str(_V24_ROOT / "hf-home")
os.environ["HF_HUB_CACHE"] = str(_V24_ROOT / "hub-cache")
os.environ["HF_ASSETS_CACHE"] = str(_V24_ROOT / "assets-cache")
os.environ["HF_XET_CACHE"] = str(_V24_ROOT / "xet-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import HfApi, snapshot_download

import replacement_loss_v23_hub_adapter as v23_rules


PROTOCOL_ID = "theseus-replacement-loss-v24-public-checkpoint-acquisition"
SEARCH_LIMIT = 1000
MAX_PARAMETERS = 500_000_000
MAX_MODEL_BYTES = 2_200_000_000
MAX_TOTAL_BYTES = 6_600_000_000
FAMILY_QUERIES = {
    "opt_relu": ("opt",),
    "gpt2_gelu": ("gpt2",),
    "llama_silu_gated": ("llama", "smollm", "open-llama", "tinyllama"),
}


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def screen_public_hub(
    excluded_model_ids: set[str],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Query bounded public model lists and apply exact parameter filters locally."""

    api = HfApi(endpoint="https://huggingface.co", token=False)
    family_screens: dict[str, Any] = {}
    selected_checkpoints = []
    for family, queries in FAMILY_QUERIES.items():
        by_identity: dict[tuple[str, str], Any] = {}
        query_receipts = []
        for query in queries:
            results = list(api.list_models(
                search=query,
                gated=False,
                sort="downloads",
                direction=-1,
                limit=SEARCH_LIMIT,
                cardData=True,
                fetch_config=True,
                token=False,
            ))
            projection = []
            for item in results:
                model_id = str(_get(item, "id", ""))
                revision = str(_get(item, "sha", "")).lower()
                downloads = int(_get(item, "downloads", 0) or 0)
                config = _get(item, "config", {}) or {}
                projection.append({
                    "model_id": model_id,
                    "revision": revision,
                    "downloads": downloads,
                    "model_type": str(_get(config, "model_type", "")).lower(),
                    "gated": _get(item, "gated"),
                    "private": _get(item, "private"),
                })
                if model_id and revision:
                    by_identity.setdefault((model_id, revision), item)
            query_receipts.append({
                "search": query,
                "result_count": len(results),
                "result_projection": projection,
                "result_projection_sha256": _canonical_digest(projection),
            })
            if on_progress is not None:
                on_progress({
                    "protocol_id": PROTOCOL_ID,
                    "family": family,
                    "completed_query_count": len(query_receipts),
                    "query_receipts": query_receipts,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                })
        ordered = sorted(
            by_identity.values(),
            key=lambda item: (
                -int(_get(item, "downloads", 0) or 0),
                str(_get(item, "id", "")).lower(),
                str(_get(item, "sha", "")).lower(),
            ),
        )
        screened_rows = []
        selected = None
        for index, item in enumerate(ordered):
            candidate, reason = v23_rules._model_candidate(api, family, item, excluded_model_ids)
            if candidate is not None:
                selected = {"family": family, **candidate}
                screened_rows.append({
                    "model_id": candidate["model_id"],
                    "revision": candidate["revision"],
                    "status": "SELECTED",
                    "parameter_count": candidate["parameter_count"],
                    "license": candidate["license"],
                    "estimated_download_bytes": candidate["estimated_download_bytes"],
                })
                screened_rows.extend({
                    "model_id": str(_get(lower, "id", "")),
                    "revision": str(_get(lower, "sha", "")),
                    "status": "NOT_EVALUATED_AFTER_HIGHER_RANKED_ELIGIBLE_CANDIDATE",
                } for lower in ordered[index + 1 :])
                break
            screened_rows.append({
                "model_id": str(_get(item, "id", "")),
                "revision": str(_get(item, "sha", "")),
                "status": "REJECTED",
                "reason": reason,
            })
        family_payload = {
            "queries": query_receipts,
            "unique_result_count": len(ordered),
            "screened_result_projection": screened_rows,
            "screened_result_projection_sha256": _canonical_digest(screened_rows),
            "selected_identity_count": 1 if selected else 0,
        }
        family_screens[family] = family_payload
        if selected:
            selected_checkpoints.append(selected)
    total_bytes = sum(int(row["estimated_download_bytes"]) for row in selected_checkpoints)
    ready = len(selected_checkpoints) == 3 and total_bytes <= MAX_TOTAL_BYTES
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "READY_TO_DOWNLOAD" if ready else "NO_ELIGIBLE_FRESH_TARGET",
        "assurance_scope": "bounded public metadata and safetensors-header screen; no hosted inference or weights downloaded",
        "family_screens": family_screens,
        "selected_checkpoints": selected_checkpoints if ready else [],
        "selected_download_bytes": total_bytes if ready else 0,
        "search_limit_per_query": SEARCH_LIMIT,
        "maximum_parameter_count": MAX_PARAMETERS,
        "maximum_model_bytes": MAX_MODEL_BYTES,
        "maximum_total_bytes": MAX_TOTAL_BYTES,
        "auth_mode": "token=False",
        "api_origin": "https://huggingface.co",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def download_selected(custody_root: str | Path, screen: dict[str, Any], *, allow_public_hub_downloads: bool) -> list[dict[str, Any]]:
    if allow_public_hub_downloads is not True:
        raise PermissionError("public model downloads require --allow-public-hub-downloads")
    if screen.get("status") != "READY_TO_DOWNLOAD" or len(screen.get("selected_checkpoints", [])) != 3:
        raise ValueError("V24 downloads only after all three eligible family selections are frozen")
    root = Path(custody_root).expanduser().resolve()
    total = 0
    downloaded = []
    for selected in screen["selected_checkpoints"]:
        model_id = str(selected["model_id"])
        revision = str(selected["revision"]).lower()
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("selected model revision is not a full immutable commit SHA")
        expected_bytes = int(selected["estimated_download_bytes"])
        if expected_bytes > MAX_MODEL_BYTES or total + expected_bytes > MAX_TOTAL_BYTES:
            raise ValueError("frozen V24 model-download budget exceeded")
        location = Path(snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=root / "hub-cache",
            token=False,
            local_files_only=False,
            allow_patterns=v23_rules.ALLOW_PATTERNS,
            max_workers=4,
            endpoint="https://huggingface.co",
        )).resolve()
        if root not in location.parents:
            raise ValueError("download escaped external V24 custody")
        files = []
        for path in sorted(location.rglob("*")):
            if not path.is_file():
                continue
            if not v23_rules._allowed_model_file(path.name):
                raise ValueError(f"unexpected downloaded file: {path.name}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            files.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        actual_bytes = sum(int(row["size_bytes"]) for row in files)
        if actual_bytes != expected_bytes:
            raise ValueError("downloaded file inventory differs from frozen Hub metadata")
        if actual_bytes > MAX_MODEL_BYTES:
            raise ValueError("downloaded model exceeds per-family byte cap")
        downloaded.append({
            "protocol_id": PROTOCOL_ID,
            "family": selected["family"],
            "model_id": model_id,
            "revision": revision,
            "model_type": selected["model_type"],
            "parameter_count": selected["parameter_count"],
            "license": selected["license"],
            "local_path": location.relative_to(root).as_posix(),
            "downloaded_bytes": actual_bytes,
            "downloaded_files": files,
            "artifact_tree_sha256": _canonical_digest(files),
        })
        total += actual_bytes
    if total > MAX_TOTAL_BYTES:
        raise ValueError("total V24 downloaded bytes exceed frozen cap")
    return downloaded
