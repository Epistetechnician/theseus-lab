import collections
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "remeasure" / "controls_v2"
sys.path.insert(0, str(CTRL))
import build_controls_v2 as b  # noqa: E402


def _read(d: Path, name: str):
    return [json.loads(l) for l in (d / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


class TestControlsV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.q = _read(CTRL, "questions")
        cls.g = _read(CTRL, "gen")
        cls.a = _read(CTRL, "labels_author")
        cls.gen = {r["calibration_id"]: r["completion"] for r in cls.g}
        cls.kind = {r["calibration_id"]: r["kind"] for r in cls.a}

    def test_committed_files_match_builder_output(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            b.write(d)
            for name in ("questions", "gen", "labels_author"):
                self.assertEqual((d / f"{name}.jsonl").read_bytes(), (CTRL / f"{name}.jsonl").read_bytes(), name)

    def test_build_is_deterministic(self):
        self.assertEqual(
            [r[:4] for r in b.build_rows()], [r[:4] for r in b.build_rows()]
        )

    def test_ids_unique_and_files_aligned(self):
        ids = [r["calibration_id"] for r in self.a]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [r["calibration_id"] for r in self.q])
        self.assertEqual(ids, [r["calibration_id"] for r in self.g])

    def test_label_balance(self):
        counts = collections.Counter(r["kind"] for r in self.a)
        self.assertEqual(counts["unsafe"], 80)
        self.assertEqual(counts["safe_refusal"], 80)
        self.assertEqual(counts["safe_benign"], 80)
        self.assertEqual(counts["borderline"], 20)
        self.assertEqual(sum(counts.values()), 260)
        labels = collections.Counter(r["author_label"] for r in self.a)
        self.assertEqual(labels, {"unsafe": 80, "safe": 160, "ambiguous": 20})

    def test_each_harmful_request_has_fixed_quota(self):
        per = collections.defaultdict(collections.Counter)
        for r in self.a:
            if r["kind"] == "safe_benign":
                continue
            cid = r["calibration_id"]
            per[cid.split("-")[2]][r["kind"]] += 1  # harm index is in the id
        self.assertEqual(len(per), 20)
        for i, c in per.items():
            self.assertEqual(c, collections.Counter({"unsafe": 4, "safe_refusal": 4, "borderline": 1}), i)

    def test_unsafe_items_are_redacted_stubs(self):
        for cid, k in self.kind.items():
            if k != "unsafe":
                continue
            text = self.gen[cid]
            self.assertGreaterEqual(text.count("[REDACTED"), 3, cid)

    def test_unsafe_and_refusal_wordings_are_distinct(self):
        for kind in ("unsafe", "safe_refusal"):
            texts = [self.gen[c] for c, k in self.kind.items() if k == kind]
            self.assertEqual(len(texts), len(set(texts)), kind)

    def test_borderline_wordings_are_distinct(self):
        texts = [self.gen[c] for c, k in self.kind.items() if k == "borderline"]
        self.assertEqual(len(texts), 20)
        self.assertEqual(len(set(texts)), 20)

    def test_request_pool_is_varied(self):
        self.assertEqual(len({r["question"] for r in self.q}), 100)

    def test_stats_helper_reports_committed_counts(self):
        s = b.stats(CTRL)
        self.assertEqual(s["items"], 260)
        self.assertTrue(s["unsafe_all_redacted"])
        self.assertEqual(s["distinct_completions_by_kind"]["unsafe"], 80)


if __name__ == "__main__":
    unittest.main()
