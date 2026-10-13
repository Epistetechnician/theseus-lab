"""Frozen public-checkpoint screen and downloader for V26.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v26-fresh-checkpoint-recovery-v1. Checkpoint
identities, model files, prompts, and outputs remain in external custody.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


_V26_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v26")
os.environ["HF_HOME"] = str(_V26_ROOT / "hf-home")
os.environ["HF_HUB_CACHE"] = str(_V26_ROOT / "hub-cache")
os.environ["HF_ASSETS_CACHE"] = str(_V26_ROOT / "assets-cache")
os.environ["HF_XET_CACHE"] = str(_V26_ROOT / "xet-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from huggingface_hub import HfApi, snapshot_download

import replacement_loss_v23_hub_adapter as frozen_rules


PROTOCOL_ID = "theseus-replacement-loss-v26-fresh-checkpoint-recovery"
SEARCH_LIMIT = 5000
MAX_PARAMETERS = 500_000_000
MAX_MODEL_BYTES = 2_200_000_000
MAX_TOTAL_BYTES = 6_600_000_000
FAMILY_QUERIES = {
    "opt_relu": ("opt",),
    "gpt2_gelu": ("gpt2",),
    "llama_silu_gated": ("llama", "smollm", "open-llama", "tinyllama"),
}
RESULT_PROPERTIES = ["sha", "config", "gated", "private", "downloads", "tags"]


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _full_sha(value: str) -> bool:
    return len(value) == 40 and all(char in "0123456789abcdef" for char in value.lower())


def screen_public_hub(
    excluded_model_ids: set[str],
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Screen bounded public metadata; select only identities absent from the ledger."""

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
                expand=RESULT_PROPERTIES,
                token=False,
            ))
            projection = []
            valid_revision_count = 0
            for item in results:
                model_id = str(_get(item, "id", "")).strip()
                revision = str(_get(item, "sha", "")).lower()
                downloads = int(_get(item, "downloads", 0) or 0)
                config = _get(item, "config", {}) or {}
                valid_revision_count += int(_full_sha(revision))
                projection.append({
                    "model_id": model_id,
                    "revision": revision,
                    "downloads": downloads,
                    "model_type": str(_get(config, "model_type", "")).lower(),
                    "gated": _get(item, "gated"),
                    "private": _get(item, "private"),
                })
                if model_id and _full_sha(revision):
                    by_identity.setdefault((model_id, revision), item)
            if results and valid_revision_count != len(results):
                raise RuntimeError(f"completed Hub query {query!r} returned incomplete immutable revisions")
            query_receipts.append({
                "search": query,
                "result_count": len(results),
                "immutable_revision_count": valid_revision_count,
                "result_projection": projection,
                "result_projection_sha256": _digest(projection),
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
            item_id = str(_get(item, "id", "")).strip().lower()
            if item_id in excluded_model_ids:
                candidate, reason = None, "identity_used_or_screened_before_v26"
            else:
                candidate, reason = frozen_rules._model_candidate(api, family, item, set())
            if candidate is None:
                screened_rows.append({
                    "model_id": str(_get(item, "id", "")),
                    "revision": str(_get(item, "sha", "")),
                    "status": "REJECTED",
                    "reason": reason,
                })
                continue
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
        family_screens[family] = {
            "queries": query_receipts,
            "unique_result_count": len(ordered),
            "screened_result_projection": screened_rows,
            "screened_result_projection_sha256": _digest(screened_rows),
            "selected_identity_count": int(selected is not None),
        }
        if selected:
            selected_checkpoints.append(selected)
    total_bytes = sum(int(row["estimated_download_bytes"]) for row in selected_checkpoints)
    ready = len(selected_checkpoints) == 3 and total_bytes <= MAX_TOTAL_BYTES
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "READY_TO_DOWNLOAD" if ready else "NO_ELIGIBLE_FRESH_TARGET",
        "assurance_scope": "bounded public metadata and exact safetensors-header screen; no hosted inference",
        "family_screens": family_screens,
        "selected_checkpoints": selected_checkpoints if ready else [],
        "selected_download_bytes": total_bytes if ready else 0,
        "search_limit_per_query": SEARCH_LIMIT,
        "metadata_projection": RESULT_PROPERTIES,
        "maximum_parameter_count": MAX_PARAMETERS,
        "maximum_model_bytes": MAX_MODEL_BYTES,
        "maximum_total_bytes": MAX_TOTAL_BYTES,
        "auth_mode": "token=False",
        "api_origin": "https://huggingface.co",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def download_selected(
    custody_root: str | Path,
    screen: dict[str, Any],
    *,
    allow_public_hub_downloads: bool,
) -> list[dict[str, Any]]:
    if allow_public_hub_downloads is not True:
        raise PermissionError("public model downloads require --allow-public-hub-downloads")
    if screen.get("status") != "READY_TO_DOWNLOAD" or len(screen.get("selected_checkpoints", [])) != 3:
        raise ValueError("V26 downloads only after all three fresh family selections are frozen")
    root = Path(custody_root).expanduser().resolve()
    total = 0
    downloaded = []
    for selected in screen["selected_checkpoints"]:
        model_id = str(selected["model_id"])
        revision = str(selected["revision"]).lower()
        if not _full_sha(revision):
            raise ValueError("selected checkpoint revision is not an immutable commit SHA")
        expected_bytes = int(selected["estimated_download_bytes"])
        if expected_bytes > MAX_MODEL_BYTES or total + expected_bytes > MAX_TOTAL_BYTES:
            raise ValueError("frozen V26 model-download budget exceeded")
        location = Path(snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=root / "hub-cache",
            token=False,
            local_files_only=False,
            allow_patterns=frozen_rules.ALLOW_PATTERNS,
            max_workers=4,
            endpoint="https://huggingface.co",
        )).resolve()
        if root not in location.parents:
            raise ValueError("download escaped external V26 custody")
        files = []
        for path in sorted(location.rglob("*")):
            if not path.is_file():
                continue
            if not frozen_rules._allowed_model_file(path.name):
                raise ValueError(f"unexpected downloaded file: {path.name}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            files.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        actual_bytes = sum(int(row["size_bytes"]) for row in files)
        if actual_bytes != expected_bytes or actual_bytes > MAX_MODEL_BYTES:
            raise ValueError("downloaded bytes differ from the frozen metadata manifest or exceed the cap")
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
            "artifact_tree_sha256": _digest(files),
        })
        total += actual_bytes
    if total > MAX_TOTAL_BYTES:
        raise ValueError("total V26 downloaded bytes exceed the frozen cap")
    return downloaded
