"""Audit local custody for a genuinely independent Qwen safety evaluator.

This lane is fail-closed.  It inventories manifests, model configuration, and
filesystem metadata, but it does not open the Qwen transcript, the blinded
human packet, or any label artifact.  The current owner custody contains one
Qwen generator and one BERT assessor.  The BERT assessor is distinct from the
generator, but reusing that same assessor for another prompt or score pass is
not a second independent evaluator.  Unless a separately pinned evaluator is
present, this command emits an aggregate blocker receipt and no cross-model
score.

All filesystem mutations made by the CLI are report-directory mutations in
state slice ``theseus-lab-sandbox-v1``.  The custody inventory itself is
read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

from adapters.qwen38_mlx_adapter import full_digest, sha256_file


STATE_SLICE = "theseus-lab-sandbox-v1"
LANE = "AI_PROXY_CROSS_MODEL"
SCHEMA_VERSION = "qwen38-ai-proxy-cross-model-blocker-v1"
BLOCKED_CLAIM_CEILING = "LOCAL_AI_PROXY_CROSS_MODEL_CUSTODY_BLOCKED"
EXISTING_SAME_MODEL_CLAIM_CEILING = "LOCAL_AI_PROXY_CALIBRATION_SIGNAL_ONLY"
EXPECTED_CUSTODY_ROOT = Path(
    "/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1"
)
BLOCKER_RECEIPT_RELATIVE = (
    "ai-proxy/reports/qwen38-ai-proxy-cross-model-blocker-v1.json"
)

QWEN_REPO_ID = "mlx-community/Qwen3.8-27B-4bit"
QWEN_REVISION = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
QWEN_CHECKPOINT_SHA256 = (
    "86d1e42eab761d16fa6c70d0bf1d8bda3d9c32c7cf032253030d7c37dc8658b3"
)
QWEN_CONFIG_SHA256 = (
    "14b65a0ee06517060a6bbd979bb1a8ff54e7b304b1a1f01d54344b88b8285e85"
)

ASSESSOR_REPO_ID = "LibrAI/maliciousinstruct-evaluator"
ASSESSOR_REVISION = "d717b8732dc5c5f8e9b04296c6eb58a84808d630"
ASSESSOR_MODEL_SHA256 = (
    "19de7037727dc60176f79bd3855c072e073c3851dbc232416df13f392403cc12"
)
ASSESSOR_CONFIG_SHA256 = (
    "6ff5a88bcad28ab82a9671dc51524fad58945956ed67d291a07c018119229e55"
)
TOKENIZER_REPO_ID = "google-bert/bert-base-cased"
TOKENIZER_REVISION = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
TOKENIZER_CONFIG_SHA256 = (
    "8b73abed2d67075fd3ec0f840aa53d9cccdc9169c04a7da135c07e721910ed74"
)

LOCKED_RAW_MANIFEST_SHA256 = (
    "16a9ed3b065e29b66ddb9cf8e3e2695e6e5a00be25707cab457e1012d61af80d"
)
LOCKED_RAW_TRANSCRIPT_SHA256 = (
    "7c18ce7bac38141d375a27ceb30e4aed65d5ab17f065aceb01f92f0521b62c9a"
)
LOCKED_QWEN_GENERATION_REPORT_SHA256 = (
    "c2881938ee68e21d0dca4068608e89fc9e4ba141329b5df1665e8f2b1205594b"
)
LOCKED_SAME_MODEL_REPORT_SHA256 = (
    "61abf073013b70669ba3a7affca28572f4adb9218a01bccbf4166cf3bc7c7f6a"
)
LOCKED_ASSESSMENT_MANIFEST_SHA256 = (
    "4aee9a58da61e76ca6659c263f949abecaf7702e53a4e7826a363994f56bc580"
)
LOCKED_CONTRAST_MANIFEST_SHA256 = (
    "767e3086bab2f2857acc6f9bfb16c6e0fa46f2e22e702f710992a3042fd3d01a"
)
LOCKED_ASSESSOR_MANIFEST_SHA256 = (
    "27ae063cc837a840d790ad2ca0489982de8fa27007c14c3ddd10c5305444e31a"
)
LOCKED_PRIMARY_SCORE_MANIFEST_SHA256 = (
    "1600c7a1ece23ff10f910632dc6d291d1e66185d0fc37470f56d08bac5a452fd"
)
LOCKED_HUMAN_PACKET_MANIFEST_SHA256 = (
    "3ca23f928470092529a83f47bf2b1ec3343ac7c4411ac7bd387416835e3069e4"
)

ALTERNATE_SCORE_MANIFEST_SHA256 = (
    "37b98a670c09eb238256b11d0fcc76e3c76062bc0f917845135c64b1146435b3"
)
ALTERNATE_SCORE_RECORDS_SHA256 = (
    "a3337a5e738641a3f70fac1ec397450511b8a235773b960de1b284577856de9e"
)
ALTERNATE_SCORED_DIGEST = (
    "05cec14331d59561770f8b8843f1d6aad8362af9b9d999cb52966def61319c11"
)
PRIMARY_RECORDS_SHA256 = (
    "e7cb1bbf579b7046775ce71c810bccf3dd16d026ead166e685328981e1a9bc4d"
)
PRIMARY_SCORED_DIGEST = (
    "54356261fe2c3ac0f55cf599a26a8e3c8b7f3cb446e5be34b616fe64135b4730"
)
HUMAN_PACKET_SHA256 = (
    "e6265fe3a8dbe02aabcfa7eaf43e0798525f686fc47b58e701f1bdd2d4a6f318"
)
HUMAN_SAMPLE_DIGEST = (
    "599e30a0bbf44c2d23d1831fad36f279021796a06bc297451035289fa239d0aa"
)
ALTERNATE_FRAMING_DIGEST = (
    "bf98502190d8ed93b72952ed3406438980c4eec582a209081fb01324aab8d7fc"
)

EXPECTED_HUMAN_LABEL_ARTIFACTS = frozenset(
    {
        "calibration/qwen38-calibration-v1/labels/rater-1.template.jsonl",
        "calibration/qwen38-calibration-v1/labels/rater-2.template.jsonl",
    }
)
EXPECTED_HUMAN_LABEL_CONTRACT = {
    "ambiguous_policy": "ambiguous labels require adjudication before scoring",
    "labels": ["ambiguous", "safe", "unsafe"],
    "required_raters": 2,
    "schema_version": "qwen38-human-labels-v1",
}

LOCKED_MANIFEST_SHA256: dict[str, str] = {
    "ai-proxy/qwen38-alternate-scores-v1/manifest.json": ALTERNATE_SCORE_MANIFEST_SHA256,
    "ai-proxy/qwen38-alternate-scores-v2/manifest.json": ALTERNATE_SCORE_MANIFEST_SHA256,
    "ai-proxy/qwen38-alternate-scores-v3/manifest.json": ALTERNATE_SCORE_MANIFEST_SHA256,
    "ai-proxy/qwen38-alternate-scores-v4/manifest.json": ALTERNATE_SCORE_MANIFEST_SHA256,
    "assessment/LibrAI-do-not-answer/manifest.json": LOCKED_ASSESSMENT_MANIFEST_SHA256,
    "assessment/ogma-prompt-injection-benign/manifest.json": LOCKED_CONTRAST_MANIFEST_SHA256,
    "assessor/manifest.json": LOCKED_ASSESSOR_MANIFEST_SHA256,
    "calibration/qwen38-calibration-v1/manifest.json": LOCKED_HUMAN_PACKET_MANIFEST_SHA256,
    "model/Qwen3.8-27B-4bit-mlx/manifest.json": "49a9d5f24432c9f65ed246cf27a436ffe17f98e5ec2f04f1777ad344d05a6872",
    "raw/qwen38-dna-contrast-v1/manifest.json": LOCKED_RAW_MANIFEST_SHA256,
    "scores/qwen38-assessor-v1/manifest.json": LOCKED_PRIMARY_SCORE_MANIFEST_SHA256,
}

LOCKED_CONFIG_SHA256: dict[str, str] = {
    "model/Qwen3.8-27B-4bit-mlx/config.json": QWEN_CONFIG_SHA256,
    "assessor/LibrAI-maliciousinstruct-evaluator/evaluator/config.json": ASSESSOR_CONFIG_SHA256,
    "assessor/google-bert-bert-base-cased/config.json": TOKENIZER_CONFIG_SHA256,
}

RAW_CONTENT_KEYS = frozenset(
    {
        "question",
        "completion",
        "prompt",
        "text",
        "response",
        "answer",
        "label",
    }
)
RAW_CONTENT_KEY_FRAGMENTS = ("question", "completion", "prompt", "response", "answer", "text")
RAW_METADATA_KEYS = frozenset(
    {
        "contains_questions",
        "contains_raw_completions",
        "aggregate_report_contains_raw_completions",
        "packet_contains_raw_completions",
        "raw_completions_read",
        "raw_completions_read_in_memory",
        "raw_completions_written_by_lane",
        "raw_manifest_read",
        "raw_score_records_read",
        "raw_transcript_read",
        "text_contract",
        "transcript_opened_by_this_lane",
    }
)
MODEL_WEIGHT_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".ggml", ".gguf", ".msgpack", ".onnx", ".pt", ".pth", ".safetensors"}
)
PRIMARY_SCORE_FIELDS = frozenset(
    {
        "dataset",
        "row_id",
        "harmful_score",
        "lexical_label",
        "short_response",
        "score_above_threshold",
        "effective_unsafe",
    }
)
ALTERNATE_SCORE_FIELDS = frozenset({"dataset", "row_id", "harmful_score"})

REQUIRED_SECOND_EVALUATOR = {
    "artifact_type": "independent_cross_model_safety_evaluator",
    "required_properties": [
        "separate model weights with a digest distinct from the Qwen generator",
        "separate model weights with a digest distinct from the existing BERT assessor",
        "explicit architecture and configuration identity",
        "pinned source revision and repository provenance",
        "owner-only custody manifest listing every model and tokenizer file",
        "independent scoring contract and locked threshold",
        "safe local runtime verification without provider calls or downloads",
    ],
    "must_not_be_satisfied_by": [
        "changed prompt or rubric framing",
        "same BERT assessor weights",
        "lexical refusal heuristic",
        "the Qwen generator itself",
        "human labels or labels authored by another AI proxy",
    ],
}


def _absolute_without_following(path: Path) -> Path:
    """Return an absolute path while preserving symlink components."""

    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    """Reject symlinks anywhere in a custody path."""

    candidate = _absolute_without_following(path)
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ValueError(f"{label} must not contain a symlink: {component}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _outside_repo(path: Path, *, label: str) -> None:
    resolved = path.resolve()
    repo_root = _repo_root()
    if resolved == repo_root or repo_root in resolved.parents:
        raise ValueError(f"{label} must be outside the repository")


def _owner_only_directory(path: Path, *, create: bool = False) -> Path:
    """Validate or create one owner-only directory without following links."""

    candidate = _absolute_without_following(path)
    _reject_symlink_components(candidate, label="custody directory")
    resolved = candidate.resolve()
    _outside_repo(resolved, label="custody directory")
    if create and not resolved.exists():
        missing: list[Path] = []
        cursor = resolved
        while not cursor.exists():
            missing.append(cursor)
            cursor = cursor.parent
        for directory in reversed(missing):
            directory.mkdir(mode=0o700)
            directory.chmod(0o700)
    if not resolved.is_dir() or resolved.is_symlink():
        raise FileNotFoundError(f"custody directory does not exist: {resolved}")
    if stat.S_IMODE(os.lstat(resolved).st_mode) & 0o077:
        raise ValueError(f"custody directory must be owner-only: {resolved}")
    return resolved


def _owner_only_file(path: Path, *, label: str) -> Path:
    """Validate one owner-only regular file without following links."""

    candidate = _absolute_without_following(path)
    _reject_symlink_components(candidate, label=label)
    resolved = candidate.resolve()
    _outside_repo(resolved, label=label)
    mode = os.lstat(resolved).st_mode
    if not stat.S_ISREG(mode) or stat.S_IMODE(mode) & 0o077:
        raise ValueError(f"{label} must be an owner-only regular file: {resolved}")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(_owner_only_file(path, label=label).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _safe_model_file_records(value: object) -> list[dict[str, object]]:
    """Keep only digest-bearing file metadata from a custody manifest."""

    if not isinstance(value, list):
        return []
    records: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        record: dict[str, object] = {}
        for key in ("path", "bytes", "sha256"):
            if key in item and isinstance(item[key], (str, int)):
                record[key] = item[key]
        if record:
            records.append(record)
    return records


def _manifest_summary(root: Path, path: Path, digest: str) -> dict[str, object]:
    """Read a manifest and retain aggregate custody metadata only."""

    manifest = _read_json(path, label="custody manifest")
    relative = path.relative_to(root).as_posix()
    summary: dict[str, object] = {
        "path": relative,
        "sha256": digest,
        "keys": sorted(str(key) for key in manifest),
    }
    scalar_keys = (
        "state_slice",
        "schema_version",
        "artifact_type",
        "repo_id",
        "revision",
        "base_model",
        "subfolder",
        "file",
        "records_file",
        "rows",
        "rows_written",
        "selected_rows",
        "complete",
        "annotator_blinded_to_assessor_scores",
        "contains_raw_completions",
        "contains_questions",
        "checkpoint_digest",
        "transcript_sha256",
        "records_sha256",
        "scored_digest",
        "packet_sha256",
        "sample_digest",
        "selection_digest",
        "assessment_corpus_digest",
        "contrast_corpus_digest",
        "qwen_checkpoint_digest",
        "assessor_model_sha256",
        "assessor_manifest_digest",
        "raw_transcript_sha256",
        "framing_digest",
    )
    for key in scalar_keys:
        if key in manifest and isinstance(manifest[key], (str, int, float, bool)):
            summary[key] = manifest[key]
    for key in ("files", "model_files"):
        records = _safe_model_file_records(manifest.get(key))
        if records:
            summary[f"declared_{key}"] = records
            summary[f"declared_{key}_digest"] = full_digest(records)
    tokenizer = manifest.get("tokenizer")
    if isinstance(tokenizer, dict):
        tokenizer_summary: dict[str, object] = {}
        for key in ("repo_id", "revision"):
            if isinstance(tokenizer.get(key), str):
                tokenizer_summary[key] = tokenizer[key]
        records = _safe_model_file_records(tokenizer.get("files"))
        if records:
            tokenizer_summary["files"] = records
            tokenizer_summary["files_digest"] = full_digest(records)
        summary["tokenizer"] = tokenizer_summary
    label_contract = manifest.get("label_contract")
    if isinstance(label_contract, dict):
        safe_contract = {
            key: label_contract[key]
            for key in ("ambiguous_policy", "labels", "required_raters", "schema_version")
            if key in label_contract
        }
        summary["label_contract"] = safe_contract
    return summary


def _normalise_relative_exclusions(paths: Iterable[str]) -> list[str]:
    """Validate and canonicalize custody-relative inventory exclusions."""

    normalized: set[str] = set()
    for raw_path in paths:
        if not isinstance(raw_path, str):
            raise ValueError("inventory exclusions must be relative path strings")
        candidate = Path(raw_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("inventory exclusions must stay within custody")
        relative = candidate.as_posix()
        if relative in {"", "."}:
            raise ValueError("inventory exclusions must name an entry")
        normalized.add(relative)
    return sorted(normalized)


def _tree_inventory(
    root: Path,
    *,
    excluded_relative_paths: Iterable[str] = (),
) -> dict[str, object]:
    """Capture recursive path, type, mode, and byte metadata without content."""

    excluded_paths = _normalise_relative_exclusions(excluded_relative_paths)
    excluded = set(excluded_paths)
    entries: list[dict[str, object]] = []
    permission_findings: list[dict[str, object]] = []
    symlinks: list[str] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = sorted(os.scandir(current), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError(f"custody tree could not be inventoried: {current}: {exc}") from exc
        for child in children:
            child_path = Path(child.path)
            relative = child_path.relative_to(root).as_posix()
            if relative in excluded:
                continue
            info = os.lstat(child_path)
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                kind = "symlink"
                symlinks.append(relative)
                permission_findings.append(
                    {
                        "path": relative,
                        "kind": kind,
                        "actual_mode": oct(mode),
                        "expected": "no_symlink",
                        "reason": "symlink_entries_are_not_allowed",
                    }
                )
                byte_count = int(info.st_size)
            elif stat.S_ISDIR(info.st_mode):
                kind = "directory"
                byte_count = 0
                stack.append(child_path)
                if mode != 0o700:
                    permission_findings.append(
                        {
                            "path": relative,
                            "kind": kind,
                            "actual_mode": oct(mode),
                            "expected": "0o700",
                            "reason": "custody_directories_must_be_owner_only",
                        }
                    )
            elif stat.S_ISREG(info.st_mode):
                kind = "file"
                byte_count = int(info.st_size)
                if mode != 0o600:
                    permission_findings.append(
                        {
                            "path": relative,
                            "kind": kind,
                            "actual_mode": oct(mode),
                            "expected": "0o600",
                            "reason": "custody_files_must_be_owner_only",
                        }
                    )
            else:
                kind = "special"
                byte_count = int(info.st_size)
                permission_findings.append(
                    {
                        "path": relative,
                        "kind": kind,
                        "actual_mode": oct(mode),
                        "expected": "regular_file_or_directory",
                        "reason": "special_entries_are_not_allowed",
                    }
                )
            entries.append(
                {
                    "path": relative,
                    "kind": kind,
                    "mode": oct(mode),
                    "bytes": byte_count,
                }
            )
    entries.sort(key=lambda item: str(item["path"]))
    permission_findings.sort(key=lambda item: str(item["path"]))
    mode_histogram: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for entry in entries:
        mode = str(entry["mode"])
        kind = str(entry["kind"])
        mode_histogram[mode] = mode_histogram.get(mode, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    return {
        "entry_count": len(entries),
        "kind_counts": kind_counts,
        "mode_histogram": mode_histogram,
        "entries": entries,
        "entries_digest": full_digest(entries),
        "excluded_paths": excluded_paths,
        "symlinks": symlinks,
        "permission_findings": permission_findings,
        "owner_only_tree": not permission_findings,
    }


def _unmanifested_model_files(
    tree: Mapping[str, object],
    manifests: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Find model-like files not covered by any declared custody manifest."""

    entries = tree.get("entries")
    if not isinstance(entries, list):
        return []
    declared: set[str] = set()
    for manifest in manifests:
        manifest_path = manifest.get("path")
        if not isinstance(manifest_path, str):
            continue
        manifest_parent = Path(manifest_path).parent
        model_file_base = (
            Path("assessor/LibrAI-maliciousinstruct-evaluator")
            if manifest_path == "assessor/manifest.json"
            else manifest_parent
        )
        for key in ("declared_files", "declared_model_files"):
            records = manifest.get(key)
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, dict) and isinstance(record.get("path"), str):
                        declared.add((model_file_base / record["path"]).as_posix())
        tokenizer = manifest.get("tokenizer")
        if manifest_path == "assessor/manifest.json" and isinstance(tokenizer, dict):
            records = tokenizer.get("files")
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, dict) and isinstance(record.get("path"), str):
                        declared.add(
                            Path("assessor/google-bert-bert-base-cased", record["path"]).as_posix()
                        )

    unmanifested: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("kind") != "file":
            continue
        relative = str(entry.get("path", ""))
        suffix = Path(relative).suffix.lower()
        is_model_config = relative.endswith("/config.json") and (
            relative.startswith("model/") or relative.startswith("assessor/")
        )
        if (suffix not in MODEL_WEIGHT_SUFFIXES and not is_model_config) or relative in declared:
            continue
        unmanifested.append(
            {
                "path": relative,
                "mode": entry.get("mode"),
                "bytes": entry.get("bytes"),
                "reason": "model-like file is not covered by a declared custody manifest",
            }
        )
    return sorted(unmanifested, key=lambda item: str(item["path"]))


def _manifest_inventory(
    root: Path,
    tree: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Hash and summarize every manifest discovered by the metadata scan."""

    entries = tree.get("entries")
    if not isinstance(entries, list):
        raise ValueError("tree inventory has no entries")
    manifest_paths = sorted(
        str(entry["path"])
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("kind") == "file"
        and str(entry.get("path", "")).endswith("/manifest.json")
    )
    inventory: list[dict[str, object]] = []
    for relative in manifest_paths:
        path = root / relative
        file_digest = sha256_file(_owner_only_file(path, label="custody manifest"))
        inventory.append(_manifest_summary(root, path, file_digest))
    inventory.sort(key=lambda item: str(item["path"]))
    actual_digests = {str(item["path"]): str(item["sha256"]) for item in inventory}
    expected_paths = set(LOCKED_MANIFEST_SHA256)
    actual_paths = set(actual_digests)
    mismatched = sorted(
        path
        for path in expected_paths & actual_paths
        if actual_digests[path] != LOCKED_MANIFEST_SHA256[path]
    )
    comparison = {
        "expected_manifest_count": len(LOCKED_MANIFEST_SHA256),
        "actual_manifest_count": len(inventory),
        "missing_locked_manifests": sorted(expected_paths - actual_paths),
        "unexpected_manifests": sorted(actual_paths - expected_paths),
        "mismatched_locked_manifests": mismatched,
        "matches_locked_inventory": not (
            (expected_paths - actual_paths)
            or (actual_paths - expected_paths)
            or mismatched
        ),
        "expected_inventory_digest": full_digest(
            [
                {"path": path, "sha256": LOCKED_MANIFEST_SHA256[path]}
                for path in sorted(LOCKED_MANIFEST_SHA256)
            ]
        ),
        "actual_inventory_digest": full_digest(
            [{"path": path, "sha256": actual_digests[path]} for path in sorted(actual_digests)]
        ),
    }
    return inventory, comparison


def _config_identity(config: Mapping[str, object]) -> dict[str, object]:
    """Extract architecture identity without retaining arbitrary config data."""

    identity: dict[str, object] = {}
    architectures = config.get("architectures")
    if isinstance(architectures, list) and all(isinstance(item, str) for item in architectures):
        identity["architectures"] = list(architectures)
    for key in (
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "torch_dtype",
    ):
        value = config.get(key)
        if isinstance(value, (str, int, float, bool)):
            identity[key] = value
    quantization = config.get("quantization_config")
    if isinstance(quantization, dict):
        selected = {
            key: quantization[key]
            for key in ("bits", "group_size", "mode")
            if isinstance(quantization.get(key), (str, int, float, bool))
        }
        if selected:
            identity["quantization_config"] = selected
    return identity


def _config_inventory(root: Path) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Read only locked model config JSON files and bind their digests."""

    inventory: list[dict[str, object]] = []
    identities: dict[str, dict[str, object]] = {}
    for relative, expected_digest in sorted(LOCKED_CONFIG_SHA256.items()):
        path = root / relative
        item: dict[str, object] = {"path": relative, "expected_sha256": expected_digest}
        try:
            private_path = _owner_only_file(path, label="model config")
            digest = sha256_file(private_path)
            config = _read_json(private_path, label="model config")
            identity = _config_identity(config)
            item.update(
                {
                    "present": True,
                    "sha256": digest,
                    "digest_matches_locked": digest == expected_digest,
                    "identity": identity,
                    "identity_digest": full_digest(identity),
                }
            )
            identities[relative] = identity
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
            item.update({"present": False, "error": type(exc).__name__})
        inventory.append(item)
    return inventory, identities


def _config_inventory_matches_locked(inventory: Iterable[Mapping[str, object]]) -> bool:
    """Require every locked architecture config to exist and match its digest."""

    records = list(inventory)
    return bool(records) and len(records) == len(LOCKED_CONFIG_SHA256) and all(
        item.get("present") is True and item.get("digest_matches_locked") is True
        for item in records
    )


def _manifest_model_records(summary: Mapping[str, object]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for key in ("declared_files", "declared_model_files"):
        value = summary.get(key)
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    return records


def _candidate_search(
    manifests: Iterable[Mapping[str, object]],
    config_identities: Mapping[str, Mapping[str, object]],
    unmanifested_model_files: Iterable[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Classify every model-bearing artifact and refuse ambiguous candidates."""

    model_bearing: list[dict[str, object]] = []
    evaluator_candidates: list[dict[str, object]] = []
    independent_candidates: list[dict[str, object]] = []
    unclassified: list[dict[str, object]] = []

    for manifest in manifests:
        artifact_type = manifest.get("artifact_type")
        model_records = _manifest_model_records(manifest)
        is_model_artifact = artifact_type in {
            "model_weights",
            "independent_safety_assessor",
        } or bool(model_records)
        if not is_model_artifact:
            continue
        relative = str(manifest.get("path"))
        if artifact_type == "model_weights":
            role = "generator"
            rejection = "the Qwen checkpoint is the generator, not an evaluator"
        elif artifact_type == "independent_safety_assessor":
            role = "existing_assessor"
            rejection = (
                "this is the sole custodied assessor; prompt or score changes using its "
                "BERT weights are not a second independent evaluator"
            )
        else:
            role = "unclassified_model_artifact"
            rejection = "model-bearing artifact type is not in the locked evaluator registry"
        config_path = None
        if artifact_type == "model_weights":
            config_path = "model/Qwen3.8-27B-4bit-mlx/config.json"
        elif artifact_type == "independent_safety_assessor":
            config_path = "assessor/LibrAI-maliciousinstruct-evaluator/evaluator/config.json"
        identity = config_identities.get(config_path or "", {})
        candidate: dict[str, object] = {
            "manifest_path": relative,
            "manifest_sha256": manifest.get("sha256"),
            "artifact_type": artifact_type,
            "repo_id": manifest.get("repo_id"),
            "revision": manifest.get("revision"),
            "role": role,
            "model_file_records": model_records,
            "model_file_records_digest": full_digest(model_records),
            "config_path": config_path,
            "architecture_identity": dict(identity),
            "architecture_identity_digest": full_digest(identity),
            "rejection": rejection,
        }
        model_bearing.append(candidate)
        if role == "existing_assessor":
            evaluator_candidates.append(candidate)
        elif role == "unclassified_model_artifact":
            unclassified.append(candidate)

    # There is deliberately no automatic path from an unknown model artifact
    # to an independent evaluator.  A new evaluator must first be explicitly
    # registered with a reviewed custody contract.
    return {
        "model_bearing_artifacts": model_bearing,
        "evaluator_candidates": evaluator_candidates,
        "independent_evaluator_candidates": independent_candidates,
        "unclassified_model_artifacts": unclassified,
        "unmanifested_model_files": [dict(item) for item in unmanifested_model_files],
        "existing_assessor_distinct_from_qwen_generator": bool(
            len(evaluator_candidates) == 1
            and len(model_bearing) >= 2
            and evaluator_candidates[0].get("repo_id") != QWEN_REPO_ID
        ),
        "same_assessor_weights_counted_as_independent": False,
        "independence_proven": bool(independent_candidates),
    }


def _same_model_summary(root: Path) -> dict[str, object]:
    """Verify the existing same-model receipt without copying its contents."""

    relative = "ai-proxy/reports/qwen38-ai-proxy-calibration-v4.json"
    path = root / relative
    summary: dict[str, object] = {
        "path": relative,
        "expected_sha256": LOCKED_SAME_MODEL_REPORT_SHA256,
        "preserved": False,
    }
    try:
        private_path = _owner_only_file(path, label="same-model report")
        digest = sha256_file(private_path)
        report = _read_json(private_path, label="same-model report")
        claim_boundary = report.get("claim_boundary")
        claim_boundary = claim_boundary if isinstance(claim_boundary, dict) else {}
        proxies = report.get("proxies")
        proxies = proxies if isinstance(proxies, dict) else {}
        dependence = proxies.get("dependence")
        dependence = dependence if isinstance(dependence, dict) else {}
        summary.update(
            {
                "sha256": digest,
                "digest_matches_locked": digest == LOCKED_SAME_MODEL_REPORT_SHA256,
                "aggregate_safe": not _contains_raw_content_keys(report),
                "state_slice": report.get("state_slice"),
                "lane": report.get("lane"),
                "status": report.get("status"),
                "claim_ceiling": report.get("claim_ceiling"),
                "human_calibration_status": claim_boundary.get(
                    "human_calibration_status_unchanged"
                ),
                "same_model_dependence_disclosed": claim_boundary.get(
                    "same_model_dependence_disclosed"
                ),
                "independent_of_primary_proxy": dependence.get(
                    "independent_of_primary_proxy"
                ),
                "heldout_agreement": dependence.get("heldout_label_agreement"),
                "heldout_cohen_kappa": dependence.get("heldout_cohen_kappa"),
                "rows_total": (
                    report.get("rows", {}).get("total")
                    if isinstance(report.get("rows"), dict)
                    else None
                ),
            }
        )
        summary["preserved"] = bool(
            summary["digest_matches_locked"]
            and summary["aggregate_safe"]
            and summary["state_slice"] == STATE_SLICE
            and summary["lane"] == "AI_PROXY_ONLY"
            and summary["status"] == "AI_PROXY_ONLY_CONSISTENCY_AUDIT_PASS"
            and summary["claim_ceiling"] == EXISTING_SAME_MODEL_CLAIM_CEILING
            and summary["human_calibration_status"] == "BLINDED_CALIBRATION_PACKET_READY"
            and summary["same_model_dependence_disclosed"] is True
            and summary["independent_of_primary_proxy"] is False
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        summary["error"] = type(exc).__name__
    return summary


def _qwen_generation_summary(root: Path) -> dict[str, object]:
    """Verify the aggregate Qwen generation receipt without reading the transcript."""

    relative = "reports/qwen38-dna-contrast-raw.json"
    path = root / relative
    summary: dict[str, object] = {
        "path": relative,
        "expected_sha256": LOCKED_QWEN_GENERATION_REPORT_SHA256,
        "preserved": False,
    }
    try:
        private_path = _owner_only_file(path, label="Qwen generation report")
        digest = sha256_file(private_path)
        report = _read_json(private_path, label="Qwen generation report")
        model = report.get("model")
        model = model if isinstance(model, dict) else {}
        assessment = report.get("assessment")
        assessment = assessment if isinstance(assessment, dict) else {}
        raw_output = assessment.get("raw_output_custody")
        raw_output = raw_output if isinstance(raw_output, dict) else {}
        summary.update(
            {
                "sha256": digest,
                "digest_matches_locked": digest == LOCKED_QWEN_GENERATION_REPORT_SHA256,
                "aggregate_safe": not _contains_raw_content_keys(report),
                "state_slice": report.get("state_slice"),
                "checkpoint_digest": model.get("checkpoint_digest"),
                "weights_loaded": model.get("weights_loaded"),
                "rows": raw_output.get("rows_written", assessment.get("rows_assessed")),
                "raw_transcript_sha256": raw_output.get("transcript_sha256"),
            }
        )
        summary["preserved"] = bool(
            summary["digest_matches_locked"]
            and summary["aggregate_safe"]
            and summary["state_slice"] == STATE_SLICE
            and summary["checkpoint_digest"] == QWEN_CHECKPOINT_SHA256
            and summary["weights_loaded"] is True
            and summary["rows"] == 1139
            and summary["raw_transcript_sha256"] == LOCKED_RAW_TRANSCRIPT_SHA256
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        summary["error"] = type(exc).__name__
    return summary


def _human_packet_summary(
    root: Path,
    manifests: Mapping[str, Mapping[str, object]],
    tree: Mapping[str, object],
) -> dict[str, object]:
    """Verify packet identity and list label artifacts by metadata only."""

    relative = "calibration/qwen38-calibration-v1/manifest.json"
    manifest = manifests.get(relative, {})
    entries = tree.get("entries")
    label_artifacts: list[dict[str, object]] = []
    label_entries: list[dict[str, object]] = []
    if isinstance(entries, list):
        prefix = "calibration/qwen38-calibration-v1/labels/"
        for entry in entries:
            if (
                isinstance(entry, dict)
                and str(entry.get("path", "")).startswith(prefix)
            ):
                metadata = {
                    "path": entry.get("path"),
                    "kind": entry.get("kind"),
                    "mode": entry.get("mode"),
                    "bytes": entry.get("bytes"),
                }
                label_entries.append(metadata)
                if entry.get("kind") == "file":
                    label_artifacts.append(metadata)
    label_artifacts.sort(key=lambda item: str(item["path"]))
    label_entries.sort(key=lambda item: str(item["path"]))
    label_paths = {
        str(item["path"])
        for item in label_artifacts
        if isinstance(item.get("path"), str)
    }
    label_artifacts_exact = label_paths == set(EXPECTED_HUMAN_LABEL_ARTIFACTS) and all(
        item.get("kind") == "file" and item.get("mode") == "0o600"
        for item in label_entries
    )
    label_contract = manifest.get("label_contract")
    manifest_matches = bool(
        manifest.get("sha256") == LOCKED_HUMAN_PACKET_MANIFEST_SHA256
        and manifest.get("state_slice") == STATE_SLICE
        and manifest.get("schema_version") == "qwen38-human-calibration-v1"
        and manifest.get("artifact_type") == "blinded_human_calibration_packet"
        and manifest.get("contains_raw_completions") is True
        and manifest.get("rows_written") == 300
        and manifest.get("packet_sha256") == HUMAN_PACKET_SHA256
        and manifest.get("sample_digest") == HUMAN_SAMPLE_DIGEST
        and manifest.get("annotator_blinded_to_assessor_scores") is True
        and label_contract == EXPECTED_HUMAN_LABEL_CONTRACT
        and label_artifacts_exact
    )
    return {
        "manifest_path": relative,
        "manifest_sha256": manifest.get("sha256"),
        "manifest_matches_locked": manifest_matches,
        "packet_sha256": manifest.get("packet_sha256"),
        "sample_digest": manifest.get("sample_digest"),
        "rows": manifest.get("rows_written"),
        "packet_contains_raw_completions": manifest.get("contains_raw_completions"),
        "annotator_blinded_to_assessor_scores": manifest.get(
            "annotator_blinded_to_assessor_scores"
        ),
        "label_contract": label_contract,
        "label_contract_digest": full_digest(label_contract),
        "label_artifacts_discovered": label_artifacts,
        "unexpected_label_entries": [
            item for item in label_entries if item.get("path") not in EXPECTED_HUMAN_LABEL_ARTIFACTS
        ],
        "label_artifacts_exact": label_artifacts_exact,
        "labels_read": False,
        "labels_consumed": 0,
        "status": "BLINDED_CALIBRATION_PACKET_READY" if manifest_matches else "UNKNOWN",
        "workflow_preserved": manifest_matches,
    }


def _contains_raw_content_keys(value: object) -> bool:
    if isinstance(value, dict):
        for key in value:
            if key in RAW_CONTENT_KEYS:
                return True
            if (
                isinstance(key, str)
                and key not in RAW_METADATA_KEYS
                and any(fragment in key.lower() for fragment in RAW_CONTENT_KEY_FRAGMENTS)
            ):
                return True
        return any(_contains_raw_content_keys(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_raw_content_keys(child) for child in value)
    return False


def _read_aggregate_score_records(
    path: Path,
    *,
    expected_fields: frozenset[str],
    expected_rows: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Read only score records and reject any content-bearing record field."""

    records: list[dict[str, object]] = []
    counts = {"harmful": 0, "benign_contrast": 0}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"aggregate score line {line_number} is not an object")
            if RAW_CONTENT_KEYS & set(value):
                raise ValueError(f"aggregate score line {line_number} contains raw fields")
            if set(value) != set(expected_fields):
                raise ValueError(f"aggregate score line {line_number} has unexpected fields")
            dataset = value.get("dataset")
            row_id = value.get("row_id")
            score = value.get("harmful_score")
            if (
                not isinstance(dataset, str)
                or dataset not in counts
                or not isinstance(row_id, int)
                or isinstance(row_id, bool)
            ):
                raise ValueError(f"aggregate score line {line_number} has an invalid identity")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise ValueError(f"aggregate score line {line_number} has an invalid score")
            if expected_fields == PRIMARY_SCORE_FIELDS:
                if not isinstance(value.get("lexical_label"), str) or not isinstance(
                    value.get("short_response"), bool
                ) or not isinstance(value.get("score_above_threshold"), bool) or not isinstance(
                    value.get("effective_unsafe"), bool
                ):
                    raise ValueError(f"aggregate score line {line_number} has invalid primary fields")
            counts[dataset] += 1
            records.append(dict(value))
    if len(records) != expected_rows:
        raise ValueError("aggregate score record count does not match the locked receipt")
    if records != sorted(records, key=lambda item: (str(item["dataset"]), int(item["row_id"]))):
        raise ValueError("aggregate score records are not in canonical identity order")
    if counts != {"harmful": 939, "benign_contrast": 200}:
        raise ValueError("aggregate score records do not cover the locked partitions")
    return records, counts


def _score_receipt_summary(
    root: Path,
    manifests: Mapping[str, Mapping[str, object]],
    *,
    relative: str,
    expected_manifest_sha256: str,
    expected_records_sha256: str,
    expected_scored_digest: str,
    expected_schema_version: str,
    expected_artifact_type: str,
    expected_fields: frozenset[str],
    expected_framing_digest: str | None = None,
) -> dict[str, object]:
    """Verify one aggregate score manifest and its non-raw records."""

    manifest = manifests.get(relative, {})
    summary: dict[str, object] = {
        "manifest_path": relative,
        "expected_manifest_sha256": expected_manifest_sha256,
        "expected_records_sha256": expected_records_sha256,
        "expected_scored_digest": expected_scored_digest,
        "verified": False,
        "aggregate_score_records_read": False,
    }
    try:
        manifest_checks = {
            "manifest_sha256": manifest.get("sha256") == expected_manifest_sha256,
            "state_slice": manifest.get("state_slice") == STATE_SLICE,
            "schema_version": manifest.get("schema_version") == expected_schema_version,
            "artifact_type": manifest.get("artifact_type") == expected_artifact_type,
            "records_file": manifest.get("records_file") == "scores.jsonl",
            "contains_raw_completions": manifest.get("contains_raw_completions") is False,
            "contains_questions": manifest.get("contains_questions") is False,
            "rows_written": manifest.get("rows_written") == 1139,
            "qwen_checkpoint_digest": manifest.get("qwen_checkpoint_digest")
            == QWEN_CHECKPOINT_SHA256,
            "raw_transcript_sha256": manifest.get("raw_transcript_sha256")
            == LOCKED_RAW_TRANSCRIPT_SHA256,
            "assessor_model_sha256": manifest.get("assessor_model_sha256")
            == ASSESSOR_MODEL_SHA256,
            "assessor_manifest_digest": manifest.get("assessor_manifest_digest")
            == LOCKED_ASSESSOR_MANIFEST_SHA256,
            "records_sha256": manifest.get("records_sha256") == expected_records_sha256,
            "scored_digest": manifest.get("scored_digest") == expected_scored_digest,
        }
        if expected_framing_digest is not None:
            manifest_checks["framing_digest"] = (
                manifest.get("framing_digest") == expected_framing_digest
            )
        summary["manifest_checks"] = manifest_checks
        records_file = manifest.get("records_file")
        if not isinstance(records_file, str) or not records_file:
            raise ValueError("aggregate score manifest has no records file")
        record_relative = (Path(relative).parent / records_file).as_posix()
        record_path = root / record_relative
        resolved_record_path = record_path.resolve()
        try:
            resolved_record_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("aggregate score records escape custody root") from exc
        private_records = _owner_only_file(record_path, label="aggregate score records")
        records_sha256 = sha256_file(private_records)
        records, counts = _read_aggregate_score_records(
            private_records,
            expected_fields=expected_fields,
            expected_rows=1139,
        )
        scored_digest = full_digest(records)
        summary.update(
            {
                "records_path": record_relative,
                "records_sha256": records_sha256,
                "records_sha256_matches_locked": records_sha256 == expected_records_sha256,
                "scored_digest": scored_digest,
                "scored_digest_matches_locked": scored_digest == expected_scored_digest,
                "record_counts": counts,
                "aggregate_score_records_read": True,
            }
        )
        summary["verified"] = bool(
            all(manifest_checks.values())
            and records_sha256 == expected_records_sha256
            and scored_digest == expected_scored_digest
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        summary["error"] = type(exc).__name__
    return summary


def _assert_aggregate_safe(report: Mapping[str, object]) -> None:
    """Reject raw-content fields from a blocker receipt."""

    if _contains_raw_content_keys(report):
        raise ValueError("cross-model blocker receipt contains raw-content fields")


def _artifact_summary(
    root: Path,
    manifests: Mapping[str, Mapping[str, object]],
    configs: Mapping[str, Mapping[str, object]],
    qwen_generation: Mapping[str, object],
) -> dict[str, object]:
    """Bind the exact existing evidence chain and verify aggregate scores."""

    def manifest_item(relative: str) -> dict[str, object]:
        return dict(manifests.get(relative, {"path": relative, "missing": True}))

    qwen = manifest_item("model/Qwen3.8-27B-4bit-mlx/manifest.json")
    assessor = manifest_item("assessor/manifest.json")
    raw = manifest_item("raw/qwen38-dna-contrast-v1/manifest.json")
    human = manifest_item("calibration/qwen38-calibration-v1/manifest.json")
    primary_score = _score_receipt_summary(
        root,
        manifests,
        relative="scores/qwen38-assessor-v1/manifest.json",
        expected_manifest_sha256=LOCKED_PRIMARY_SCORE_MANIFEST_SHA256,
        expected_records_sha256=PRIMARY_RECORDS_SHA256,
        expected_scored_digest=PRIMARY_SCORED_DIGEST,
        expected_schema_version="qwen38-assessor-scores-v1",
        expected_artifact_type="aggregate_safe_assessor_scores",
        expected_fields=PRIMARY_SCORE_FIELDS,
    )
    alternate_scores = [
        _score_receipt_summary(
            root,
            manifests,
            relative=f"ai-proxy/qwen38-alternate-scores-v{version}/manifest.json",
            expected_manifest_sha256=ALTERNATE_SCORE_MANIFEST_SHA256,
            expected_records_sha256=ALTERNATE_SCORE_RECORDS_SHA256,
            expected_scored_digest=ALTERNATE_SCORED_DIGEST,
            expected_schema_version="qwen38-ai-proxy-alternate-scores-v1",
            expected_artifact_type="aggregate_safe_ai_proxy_scores",
            expected_fields=ALTERNATE_SCORE_FIELDS,
            expected_framing_digest=ALTERNATE_FRAMING_DIGEST,
        )
        for version in range(1, 5)
    ]
    return {
        "qwen_generator": {
            "manifest": qwen,
            "checkpoint_digest": qwen.get("checkpoint_digest"),
            "config": dict(configs.get("model/Qwen3.8-27B-4bit-mlx/config.json", {})),
        },
        "qwen_generation_receipt": dict(qwen_generation),
        "existing_assessor": {
            "manifest": assessor,
            "model_sha256": ASSESSOR_MODEL_SHA256,
            "config": dict(
                configs.get(
                    "assessor/LibrAI-maliciousinstruct-evaluator/evaluator/config.json",
                    {},
                )
            ),
            "tokenizer": {
                "repo_id": TOKENIZER_REPO_ID,
                "revision": TOKENIZER_REVISION,
                "config": dict(
                    configs.get("assessor/google-bert-bert-base-cased/config.json", {})
                ),
            },
        },
        "raw_receipt": {
            "manifest": raw,
            "transcript_sha256": raw.get("transcript_sha256"),
            "rows": raw.get("rows_written"),
            "transcript_opened_by_this_lane": False,
        },
        "primary_score_receipt": primary_score,
        "same_model_alternate_score_receipts": {
            "receipts": alternate_scores,
            "all_verified": all(item.get("verified") is True for item in alternate_scores),
        },
        "human_packet": {
            "manifest": human,
            "packet_sha256": human.get("packet_sha256", HUMAN_PACKET_SHA256),
            "packet_opened_by_this_lane": False,
        },
    }


def run(
    *,
    custody_root: Path = EXPECTED_CUSTODY_ROOT,
    excluded_relative_paths: Iterable[str] = (BLOCKER_RECEIPT_RELATIVE,),
) -> dict[str, object]:
    """Build the fail-closed aggregate blocker receipt for the custody root."""

    root = _owner_only_directory(custody_root)
    excluded_paths = _normalise_relative_exclusions(excluded_relative_paths)
    tree = _tree_inventory(root, excluded_relative_paths=excluded_paths)
    manifest_list, manifest_comparison = _manifest_inventory(root, tree)
    manifests = {str(item["path"]): item for item in manifest_list}
    config_list, config_identities = _config_inventory(root)
    configs = {str(item["path"]): item for item in config_list}
    config_inventory_matches_locked = _config_inventory_matches_locked(config_list)
    unmanifested_model_files = _unmanifested_model_files(tree, manifest_list)
    candidates = _candidate_search(
        manifest_list,
        config_identities,
        unmanifested_model_files,
    )
    qwen_generation = _qwen_generation_summary(root)
    same_model = _same_model_summary(root)
    if not config_inventory_matches_locked:
        qwen_generation["preserved"] = False
        same_model["preserved"] = False
    human_packet = _human_packet_summary(root, manifests, tree)
    artifacts = _artifact_summary(root, manifests, configs, qwen_generation)

    rows = artifacts["raw_receipt"].get("rows")
    raw_rows = rows if isinstance(rows, int) and not isinstance(rows, bool) else None
    permission_findings = tree.get("permission_findings", [])
    if not isinstance(permission_findings, list):
        permission_findings = []
    failure_reasons = [
        "independent_evaluator_artifact_missing",
        "no_cross_model_score_receipt",
        "no_cross_model_result",
        "human_calibration_labels_unavailable",
    ]
    if not manifest_comparison["matches_locked_inventory"]:
        failure_reasons.append("custody_manifest_inventory_changed")
    if not same_model.get("preserved"):
        failure_reasons.append("existing_same_model_result_not_verified")
    if not qwen_generation.get("preserved"):
        failure_reasons.append("qwen_generation_receipt_not_verified")
    if not human_packet.get("workflow_preserved"):
        failure_reasons.append("human_calibration_packet_not_verified")
    if not config_inventory_matches_locked:
        failure_reasons.append("model_config_inventory_not_verified")
    primary_score = artifacts.get("primary_score_receipt")
    if not isinstance(primary_score, dict) or primary_score.get("verified") is not True:
        failure_reasons.append("primary_score_receipt_not_verified")
    alternate_receipts = artifacts.get("same_model_alternate_score_receipts")
    if (
        not isinstance(alternate_receipts, dict)
        or alternate_receipts.get("all_verified") is not True
    ):
        failure_reasons.append("same_model_alternate_score_receipt_not_verified")
    if permission_findings:
        failure_reasons.append("custody_tree_has_non_owner_only_entries")
    if unmanifested_model_files:
        failure_reasons.append("unmanifested_model_like_file_found")

    report: dict[str, object] = {
        "state_slice": STATE_SLICE,
        "lane": LANE,
        "schema_version": SCHEMA_VERSION,
        "status": "AI_PROXY_CROSS_MODEL_BLOCKED_MISSING_INDEPENDENT_EVALUATOR",
        "claim_ceiling": BLOCKED_CLAIM_CEILING,
        "interpretation": (
            "No cross-model result is available. The owner-custodied BERT assessor is "
            "distinct from the Qwen generator, but reusing that same assessor is not "
            "independent calibration. The existing same-model signal is preserved only."
        ),
        "blocker": {
            "missing_artifact": REQUIRED_SECOND_EVALUATOR,
            "exact_reason": (
                "No second evaluator was found among the locked evaluator manifests and "
                "model-like filesystem entries. Custody contains one model-bearing "
                "generator and one model-bearing BERT assessor; no second evaluator with "
                "separately pinned weights, architecture identity, revision, manifest, "
                "and scoring contract was found."
            ),
            "search_scope": "locked model manifests plus recursive model-like file metadata",
            "cross_model_result_available": False,
            "cross_model_execution_performed": False,
            "rows_available_in_existing_generation_receipt": raw_rows,
            "cross_model_rows_considered": None,
        },
        "custody_inventory": {
            "root": str(root),
            "state_slice": STATE_SLICE,
            "tree": tree,
            "manifest_inventory": manifest_list,
            "manifest_comparison": manifest_comparison,
            "config_inventory": config_list,
            "config_inventory_matches_locked": config_inventory_matches_locked,
            "config_inventory_digest": full_digest(config_list),
        },
        "evaluator_search": candidates,
        "artifacts": artifacts,
        "existing_same_model_result": same_model,
        "human_calibration": human_packet,
        "future_cross_model_protocol": {
            "rows_bound_by_existing_receipt": raw_rows,
            "disjoint_partitions_required": True,
            "threshold_locked_before_heldout_required": True,
            "wilson_intervals_required": True,
            "confusion_matrix_required_only_against_completed_reference_labels": True,
            "dependence_report_required": True,
            "threshold_selected": None,
            "partitions_created": False,
            "metrics_available": False,
        },
        "evidence": {
            "local_only": True,
            "external_provider_called": False,
            "downloads_performed": False,
            "model_weights_loaded_by_this_lane": False,
            "raw_manifest_read": True,
            "raw_transcript_read": False,
            "raw_completions_read": False,
            "raw_score_records_read": False,
            "human_packet_read": False,
            "human_labels_read": False,
            "labels_invented": False,
            "same_model_result_preserved": bool(same_model.get("preserved")),
            "aggregate_score_records_read": bool(
                isinstance(primary_score, dict)
                and primary_score.get("aggregate_score_records_read") is True
            ),
        },
        "behavior_mechanism_divergence": {
            "available": False,
            "reported": False,
            "reason": "This custody audit does not measure behavior or mechanism.",
        },
        "failure_reasons": failure_reasons,
        "claim_boundary": {
            "strongest_defensible_result": BLOCKED_CLAIM_CEILING,
            "human_calibration_status_unchanged": human_packet.get("status"),
            "no_cross_model_claim": True,
            "no_human_calibration_claim": True,
            "no_alignment_claim": True,
            "no_mechanistic_claim": True,
            "no_generalization_claim": True,
            "no_production_claim": True,
            "no_labels_invented": True,
        },
    }
    _assert_aggregate_safe(report)
    return report


def _prepare_output_path(path: Path) -> Path:
    """Prepare a new owner-only report target in the state slice."""

    candidate = _absolute_without_following(path)
    _reject_symlink_components(candidate, label="cross-model report path")
    resolved = candidate.resolve()
    _outside_repo(resolved, label="cross-model report path")
    if resolved.exists():
        raise FileExistsError(f"cross-model report already exists: {resolved}")
    _owner_only_directory(resolved.parent, create=True)
    return resolved


def _relative_to_custody_root(root: Path, path: Path) -> str | None:
    """Return an output path relative to custody when it is inside custody."""

    candidate = _absolute_without_following(path)
    _reject_symlink_components(candidate, label="cross-model report path")
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return None


def _write_private(path: Path, encoded: str) -> None:
    """Create one owner-only aggregate receipt without overwriting it."""

    candidate = _absolute_without_following(path)
    _reject_symlink_components(candidate, label="cross-model report path")
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(candidate.parent, directory_flags)
    file_fd: int | None = None
    try:
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(candidate.name, file_flags, 0o600, dir_fd=parent_fd)
        os.fchmod(file_fd, 0o600)
        with os.fdopen(file_fd, "w", encoding="utf-8") as handle:
            file_fd = None
            handle.write(encoded)
    except Exception:
        if file_fd is not None:
            os.close(file_fd)
        try:
            os.unlink(candidate.name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        raise
    finally:
        os.close(parent_fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custody-root", type=Path, default=EXPECTED_CUSTODY_ROOT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    custody_root = _owner_only_directory(args.custody_root)
    excluded_paths = [BLOCKER_RECEIPT_RELATIVE]
    output_relative = _relative_to_custody_root(custody_root, args.output)
    if output_relative is not None:
        excluded_paths.append(output_relative)
    report = run(
        custody_root=custody_root,
        excluded_relative_paths=excluded_paths,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output_path = _prepare_output_path(args.output)
    _write_private(output_path, encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
