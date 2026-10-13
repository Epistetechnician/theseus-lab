"""Opt-in, public-only Hub metadata screen and pinned model downloader.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v23-public-checkpoint-acquisition-v1. No inference
provider calls are made. All retrieved files are written below external V23
custody.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


PROTOCOL_ID = "theseus-replacement-loss-v23-public-checkpoint-acquisition"
SEARCH_LIMIT = 1000
MAX_PARAMETERS = 500_000_000
MAX_MODEL_BYTES = 2_200_000_000
MAX_TOTAL_BYTES = 6_600_000_000
FAMILY_QUERIES = {
    "opt_relu": ("opt", "opt"),
    "gpt2_gelu": ("gpt2", "gpt2"),
    "llama_silu_gated": ("llama", "llama"),
}
EXPECTED_MODEL_TYPES = {"opt_relu": "opt", "gpt2_gelu": "gpt2", "llama_silu_gated": "llama"}
BLOCKED_NAME_MARKERS = (
    "instruct", "instruction", "chat", "fine-tuned", "finetuned", "fine_tuned",
    "dpo", "sft", "rlhf", "reward", "distill", "quant", "gguf", "mlx",
    "adapter", "lora", "4bit", "8bit",
)
TOKENIZER_FILES = {
    "tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt",
    "merges.txt", "tokenizer_config.json",
}
ALLOW_PATTERNS = [
    "config.json", "generation_config.json", "model.safetensors",
    "model-*.safetensors", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt", "merges.txt",
]


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _license(card_data: Any, tags: list[str]) -> str | None:
    value = _get(card_data, "license")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    for tag in tags:
        if tag.lower().startswith("license:"):
            return tag.partition(":")[2].strip().lower() or None
    return None


def _safetensors_parameter_count(metadata: Any) -> int | None:
    counts = _get(metadata, "parameter_count")
    if not isinstance(counts, dict) or not counts:
        return None
    try:
        return sum(int(count) for count in counts.values())
    except (TypeError, ValueError):
        return None


def _allowed_model_file(name: str) -> bool:
    return bool(
        name in {
            "config.json", "generation_config.json", "model.safetensors",
            "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "tokenizer.model",
            "spiece.model", "vocab.json", "vocab.txt", "merges.txt",
        }
        or re.fullmatch(r"model-[0-9]+-of-[0-9]+\.safetensors", name)
    )


def _model_candidate(api: HfApi, family: str, item: Any, excluded_ids: set[str]) -> tuple[dict[str, Any] | None, str]:
    model_id = str(_get(item, "id", "")).strip()
    revision = str(_get(item, "sha", "")).strip().lower()
    tags = [str(tag) for tag in (_get(item, "tags", []) or [])]
    config = _get(item, "config", {}) or {}
    model_type = str(_get(config, "model_type", "")).lower()
    card_data = _get(item, "card_data", _get(item, "cardData"))
    license_id = _license(card_data, tags)
    downloads = int(_get(item, "downloads", 0) or 0)
    base = {
        "model_id": model_id,
        "revision": revision,
        "downloads": downloads,
        "model_type": model_type,
        "license": license_id,
        "gated": _get(item, "gated"),
        "private": _get(item, "private"),
        "tags": tags,
    }
    if not model_id or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        return None, "missing_immutable_revision"
    if model_id.lower() in excluded_ids:
        return None, "identity_used_or_screened_before_v23"
    if _get(item, "private") is not False or _get(item, "gated") is not False:
        return None, "not_public_ungated"
    if model_type != EXPECTED_MODEL_TYPES[family]:
        return None, "model_type_mismatch"
    if not license_id:
        return None, "license_metadata_missing"
    marker_text = (model_id + " " + " ".join(tags)).lower()
    if any(marker in marker_text for marker in BLOCKED_NAME_MARKERS):
        return None, "not_base_checkpoint_by_frozen_name_tag_rule"
    try:
        tensors = api.get_safetensors_metadata(model_id, revision=revision, token=False)
    except Exception as exc:  # external metadata is evidence, not an instruction
        return None, f"safetensors_metadata_unavailable:{type(exc).__name__}"
    parameter_count = _safetensors_parameter_count(tensors)
    if parameter_count is None:
        return None, "exact_safetensors_parameter_count_unavailable"
    if parameter_count > MAX_PARAMETERS:
        return None, "parameter_cap_exceeded"
    info = api.model_info(model_id, revision=revision, files_metadata=True, token=False)
    siblings = _get(info, "siblings", []) or []
    file_rows = []
    for sibling in siblings:
        name = str(_get(sibling, "rfilename", ""))
        if not _allowed_model_file(name):
            continue
        size = _get(sibling, "size")
        if size is None:
            lfs = _get(sibling, "lfs")
            size = _get(lfs, "size")
        if size is None:
            return None, "selected_file_size_unavailable"
        file_rows.append({"name": name, "size_bytes": int(size)})
    names = {row["name"] for row in file_rows}
    has_weights = any(name.endswith(".safetensors") and not name.endswith(".index.json") for name in names)
    if "config.json" not in names or not has_weights or not names.intersection(TOKENIZER_FILES):
        return None, "required_config_safetensors_or_tokenizer_missing"
    total_bytes = sum(int(row["size_bytes"]) for row in file_rows)
    if total_bytes > MAX_MODEL_BYTES:
        return None, "per_model_download_cap_exceeded"
    return {
        **base,
        "parameter_count": parameter_count,
        "file_manifest": sorted(file_rows, key=lambda row: row["name"]),
        "estimated_download_bytes": total_bytes,
        "files_manifest_sha256": _canonical_digest(sorted(file_rows, key=lambda row: row["name"])),
    }, "eligible"


def screen_public_hub(excluded_model_ids: set[str]) -> dict[str, Any]:
    """Screen public model metadata. Call only after the packet is frozen."""

    api = HfApi(endpoint="https://huggingface.co", token=False)
    family_receipts = {}
    selected = []
    for family, (query, _expected) in FAMILY_QUERIES.items():
        results = list(api.list_models(
            search=query,
            num_parameters="max:500M",
            gated=False,
            sort="downloads",
            limit=SEARCH_LIMIT,
            cardData=True,
            fetch_config=True,
            token=False,
        ))
        ordered = sorted(
            results,
            key=lambda item: (
                -int(_get(item, "downloads", 0) or 0),
                str(_get(item, "id", "")).lower(),
                str(_get(item, "sha", "")).lower(),
            ),
        )
        rows = []
        selected_candidate = None
        for index, item in enumerate(ordered):
            candidate, reason = _model_candidate(api, family, item, excluded_model_ids)
            if candidate is not None:
                rows.append({**candidate, "screen_status": "ELIGIBLE"})
                selected_candidate = candidate
                rows.extend({
                    "model_id": str(_get(remaining, "id", "")),
                    "revision": str(_get(remaining, "sha", "")),
                    "downloads": int(_get(remaining, "downloads", 0) or 0),
                    "screen_status": "NOT_EVALUATED_AFTER_HIGHER_RANKED_ELIGIBLE_CANDIDATE",
                } for remaining in ordered[index + 1 :])
                break
            else:
                rows.append({
                    "model_id": str(_get(item, "id", "")),
                    "revision": str(_get(item, "sha", "")),
                    "downloads": int(_get(item, "downloads", 0) or 0),
                    "model_type": str(_get(_get(item, "config", {}) or {}, "model_type", "")),
                    "screen_status": "REJECTED",
                    "reason": reason,
                })
        family_receipts[family] = {
            "query": query,
            "result_count": len(ordered),
            "result_projection": rows,
            "result_projection_sha256": _canonical_digest(rows),
            "first_eligible_rank": (len(rows) - (len(ordered) - index - 1)) if selected_candidate else None,
        }
        if selected_candidate is not None:
            selected.append({"family": family, **selected_candidate})
    overall = "READY_TO_DOWNLOAD" if len(selected) == 3 else "NO_ELIGIBLE_FRESH_TARGET"
    total_bytes = sum(int(row["estimated_download_bytes"]) for row in selected)
    if overall == "READY_TO_DOWNLOAD" and total_bytes > MAX_TOTAL_BYTES:
        overall = "NO_ELIGIBLE_FRESH_TARGET"
    return {
        "protocol_id": PROTOCOL_ID,
        "assurance_scope": "public metadata screen only; no inference calls and no model weights downloaded",
        "status": overall,
        "family_screens": family_receipts,
        "selected_checkpoints": selected if overall == "READY_TO_DOWNLOAD" else [],
        "selected_download_bytes": total_bytes if overall == "READY_TO_DOWNLOAD" else 0,
        "search_limit_per_family": SEARCH_LIMIT,
        "parameter_cap": MAX_PARAMETERS,
        "per_model_download_cap_bytes": MAX_MODEL_BYTES,
        "total_download_cap_bytes": MAX_TOTAL_BYTES,
        "api_identity": "https://huggingface.co; unauthenticated token=False",
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }


def download_selected(custody_root: str | Path, screen: dict[str, Any], *, allow_public_hub_downloads: bool) -> list[dict[str, Any]]:
    if allow_public_hub_downloads is not True:
        raise PermissionError("public model downloads require --allow-public-hub-downloads")
    if screen.get("status") != "READY_TO_DOWNLOAD" or len(screen.get("selected_checkpoints", [])) != 3:
        raise ValueError("V23 does not download unless all three frozen families have eligible selections")
    root = Path(custody_root).expanduser().resolve()
    api = HfApi(endpoint="https://huggingface.co", token=False)
    total = 0
    downloaded = []
    for selected in screen["selected_checkpoints"]:
        model_id = str(selected["model_id"])
        revision = str(selected["revision"])
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("selected model revision is not a full immutable SHA")
        if total + int(selected["estimated_download_bytes"]) > MAX_TOTAL_BYTES:
            raise ValueError("aggregate model download cap exceeded")
        result_path = Path(snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=root / "hub-cache",
            token=False,
            local_files_only=False,
            allow_patterns=ALLOW_PATTERNS,
            max_workers=4,
            endpoint="https://huggingface.co",
        )).resolve()
        if root not in result_path.parents:
            raise ValueError("download escaped external V23 custody")
        actual_files = []
        for path in sorted(result_path.rglob("*")):
            if not path.is_file():
                continue
            name = path.name
            if not _allowed_model_file(name):
                raise ValueError(f"unexpected downloaded model file: {name}")
            actual_files.append({"name": name, "size_bytes": path.stat().st_size, "sha256": _file_sha(path)})
        actual_total = sum(row["size_bytes"] for row in actual_files)
        if actual_total > MAX_MODEL_BYTES:
            raise ValueError("downloaded model exceeds frozen per-model cap")
        expected_total = int(selected["estimated_download_bytes"])
        if actual_total != expected_total:
            raise ValueError("downloaded file bytes differ from the frozen Hub file manifest")
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "model_id": model_id,
            "revision": revision,
            "family": selected["family"],
            "model_type": selected["model_type"],
            "parameter_count": selected["parameter_count"],
            "license": selected["license"],
            "downloaded_file_count": len(actual_files),
            "downloaded_bytes": actual_total,
            "downloaded_files": actual_files,
        }
        manifest["artifact_tree_sha256"] = _canonical_digest(actual_files)
        downloaded.append({**manifest, "local_path": result_path.relative_to(root).as_posix()})
        total += actual_total
    if total > MAX_TOTAL_BYTES:
        raise ValueError("aggregate downloaded model bytes exceed frozen cap")
    return downloaded


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
