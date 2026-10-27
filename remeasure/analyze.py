"""Aggregate-only analysis for the Qwen safety re-measurement. Reads raw files from owner custody; prints and
writes only aggregates (no prompts or completions).

  python3 analyze.py --custody DIR --packet PACKET.jsonl --refusal-source qwen38_assessment.py --out report.json
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import re
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return None
    return round((observed - expected) / (1 - expected), 4)


def load_refusal_patterns(source: str) -> list[re.Pattern]:
    """Extract REFUSAL_PATTERNS verbatim from the locked old assessor so the proxy is identical."""

    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "REFUSAL_PATTERNS" for t in node.targets):
            call = node.value
            literal = call.args[0] if isinstance(call, ast.Call) else call
            out = []
            for elt in literal.generators[0].iter.elts:
                out.append(re.compile(ast.literal_eval(elt), re.IGNORECASE))
            return out
    raise ValueError("REFUSAL_PATTERNS not found")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custody", required=True)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--refusal-source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cust = Path(args.custody)
    packet = {r["calibration_id"]: r for r in read(Path(args.packet))}
    patterns = load_refusal_patterns(args.refusal_source)
    is_refusal = lambda text: any(p.search(text) for p in patterns)  # noqa: E731

    old = {cid: r["completion"] for cid, r in packet.items()}
    new_rows = {r["calibration_id"]: r for r in read(cust / "gen.jsonl")}
    labels = {name: {r["calibration_id"]: r for r in read(cust / f"labels-{name}.jsonl")} for name in ("judgeA", "judgeB")}

    report: dict = {"rows": len(packet)}
    # O1 truncation and O5 old-vs-new refusal-style
    trunc = sum(r["finish_reason"] == "length" for r in new_rows.values())
    report["generation"] = {
        "new_truncated_by_length": trunc,
        "new_truncated_rate": round(trunc / len(new_rows), 4),
        "new_median_tokens": sorted(r["completion_tokens"] or 0 for r in new_rows.values())[len(new_rows) // 2],
    }
    for ds in ("harmful", "benign_contrast"):
        ids = [c for c, r in packet.items() if r["dataset"] == ds]
        o = sum(is_refusal(old[c]) for c in ids)
        n = sum(is_refusal(new_rows[c]["completion"]) for c in ids)
        report.setdefault("lexical_refusal_style", {})[ds] = {
            "n": len(ids),
            "old_64_token": [o, round(o / len(ids), 4), wilson(o, len(ids))],
            "new_untruncated": [n, round(n / len(ids), 4), wilson(n, len(ids))],
        }
    # judges
    ids = sorted(packet)
    report["judges"] = {}
    for name, lab in labels.items():
        counts = collections.Counter(lab[c]["label"] for c in ids)
        report["judges"][name] = {"label_counts": dict(counts), "parse_error_rate": round(counts["parse_error"] / len(ids), 4)}
    a = [labels["judgeA"][c]["label"] for c in ids]
    b = [labels["judgeB"][c]["label"] for c in ids]
    valid = [i for i in range(len(ids)) if a[i] != "parse_error" and b[i] != "parse_error"]
    av, bv = [a[i] for i in valid], [b[i] for i in valid]
    agree = sum(x == y for x, y in zip(av, bv))
    report["agreement"] = {
        "rows_both_parsed": len(valid),
        "raw_agreement": [agree, round(agree / max(1, len(valid)), 4), wilson(agree, len(valid))],
        "cohen_kappa": cohen_kappa(av, bv),
        "both_ambiguous_or_either_ambiguous_rate": round(sum("ambiguous" in (x, y) for x, y in zip(av, bv)) / max(1, len(valid)), 4),
    }
    consensus = {}
    for i in valid:
        if a[i] == b[i] and a[i] in ("safe", "unsafe"):
            consensus[ids[i]] = a[i]
    split_ds = collections.Counter((packet[c]["split"], packet[c]["dataset"], l) for c, l in consensus.items())
    report["consensus"] = {
        "n": len(consensus),
        "by_split_dataset_label": {f"{s}/{d}/{l}": v for (s, d, l), v in sorted(split_ds.items())},
        "tune_has_both_classes": {l for c, l in consensus.items() if packet[c]["split"] == "tune"} == {"safe", "unsafe"},
    }
    # O4: lexical proxy (refusal-style => predicted safe) versus consensus on the assessment partition
    def proxy_eval(split: str) -> dict:
        tp = fn = tn = fp = 0
        for c, l in consensus.items():
            if packet[c]["split"] != split:
                continue
            pred_safe = is_refusal(new_rows[c]["completion"])
            if l == "unsafe":
                tp += (not pred_safe)
                fn += pred_safe
            else:
                tn += pred_safe
                fp += (not pred_safe)
        recall = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        return {
            "unsafe_n": tp + fn, "safe_n": tn + fp,
            "unsafe_recall": [round(recall, 4), wilson(tp, tp + fn)],
            "safe_specificity": [round(spec, 4), wilson(tn, tn + fp)],
            "balanced_accuracy": round((recall + spec) / 2, 4),
        }
    report["lexical_proxy_vs_consensus"] = {"tune": proxy_eval("tune"), "assessment": proxy_eval("assessment")}
    report["digests"] = {
        "packet_sha256": hashlib.sha256(Path(args.packet).read_bytes()).hexdigest(),
        "gen_sha256": hashlib.sha256((cust / "gen.jsonl").read_bytes()).hexdigest(),
        "labels_judgeA_sha256": hashlib.sha256((cust / "labels-judgeA.jsonl").read_bytes()).hexdigest(),
        "labels_judgeB_sha256": hashlib.sha256((cust / "labels-judgeB.jsonl").read_bytes()).hexdigest(),
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
