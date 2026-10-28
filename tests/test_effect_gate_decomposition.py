#!/usr/bin/env python3
"""
Pure-Python decomposition of the pinned v3 causal-effect gates for
replacement-loss-v8-scale Band A.

Reproduces, without torch or numpy, the gate logic of the frozen runner
(`replacement_loss_v3_runner.py::_passes_effect`, pinned via
`docs/replacement-loss-v8s-freeze.json` -> "pinned_v3_causal_behavior_
activation_and_control_gates") and records exactly which subcheck fails in
every case/stratum/surface cell of `docs/replacement-loss-v8-scale-bandA-report.json`.

Frozen gate constants live in the runner; the copies here are assertions of
fidelity, never definitions: if this file and the runner ever disagree, the
runner wins and this helper must fail its equivalence test.
"""

import json
import math
import unittest
from pathlib import Path

REPORT_PATH = Path("docs/replacement-loss-v8-scale-bandA-report.json")

# Pinned v3 gate constants (unchanged; mirrored from replacement_loss_v3_runner.py).
COSINE_MIN = 0.90
DIRECTION_MIN = 0.80
NORMALIZED_ERROR_MAX = 0.20
POOLED_MIN_RESOLVABLE = 60
STRATUM_MIN_RESOLVABLE = 20

STRATA = ("adversarial", "compositional", "ordinary")
SURFACES = ("logits", "module_boundary")
SUBCHECKS = ("resolvable", "direction", "cosine", "normalized_error")


def effect_subchecks(summary, *, minimum_resolvable):
    """Mirror replacement_loss_v3_runner._passes_effect.

    Returns (passes, failing_subchecks) where failing_subchecks lists which of
    "resolvable", "direction", "cosine", "normalized_error" would block the
    pinned gate for this effect summary.
    """
    resolvable = summary.get("resolvable_count", 0) >= minimum_resolvable
    direction = summary.get("direction_accuracy", 0.0) >= DIRECTION_MIN
    cosine = summary.get("median_cosine", -1.0) >= COSINE_MIN
    error = (
        summary.get("median_normalized_error", float("inf")) <= NORMALIZED_ERROR_MAX
    )
    failing = [
        name
        for name, ok in (
            ("resolvable", resolvable),
            ("direction", direction),
            ("cosine", cosine),
            ("normalized_error", error),
        )
        if not ok
    ]
    return (not failing), failing


def decompose_effect_gates(data):
    """Full gate decomposition over report cases.

    For each case: pooled (resolvable>=60) and per-stratum (>=20) subchecks on
    both causal surfaces, plus the activation-rescue subchecks (cosine/error
    only, no direction gate). Companion flags mirror the frozen runner's
    treatment_pass/stratum_pass/activation_pass.
    """
    cases = []
    for case in data:
        effects = case["treatment_effects"]
        rescue = case["treatment_activation_rescue"]
        pooled = {}
        for surface in SURFACES:
            _, failing = effect_subchecks(
                effects[surface], minimum_resolvable=POOLED_MIN_RESOLVABLE
            )
            pooled[surface] = failing
        surface_rows = []
        for stratum in STRATA:
            for surface in SURFACES:
                summary = effects["per_stratum"][stratum][surface]
                _, failing = effect_subchecks(
                    summary, minimum_resolvable=STRATUM_MIN_RESOLVABLE
                )
                surface_rows.append(
                    {
                        "stratum": stratum,
                        "surface": surface,
                        "failing_subchecks": failing,
                        "median_cosine": summary["median_cosine"],
                        "median_normalized_error": summary["median_normalized_error"],
                        "direction_accuracy": summary["direction_accuracy"],
                        "resolvable_count": summary["resolvable_count"],
                        "count": summary["count"],
                    }
                )
        stratum_pass = all(
            not row["failing_subchecks"] for row in surface_rows
        )
        treatment_pass = all(not failing for failing in pooled.values())
        rescue_pass = rescue["median_cosine"] >= COSINE_MIN and rescue[
            "median_normalized_error"
        ] <= NORMALIZED_ERROR_MAX and all(
            rescue["per_stratum"][stratum]["median_cosine"] >= COSINE_MIN
            and rescue["per_stratum"][stratum]["median_normalized_error"]
            <= NORMALIZED_ERROR_MAX
            for stratum in STRATA
        )
        cases.append(
            {
                "family": case["family"],
                "seed": case["seed"],
                "case_status": case["case_status"],
                "blockers": list(case["blockers"]),
                "pooled_failing_subchecks": pooled,
                "stratum_surface_rows": surface_rows,
                "treatment_pass": treatment_pass,
                "stratum_pass": stratum_pass,
                "activation_rescue_pass": rescue_pass,
            }
        )
    return {"cases": cases}


def implied_effect_norm_upper_bound(cosine, normalized_error):
    """Upper bound on ||observed effect|| / ||reference scale||.

    The pinned runner reports ``normalized_error = ||pred - obs|| / scale``
    where ``scale`` is the baseline next-token norm (frozen
    ``replacement_loss_v3_runner.py:1283``), while cosines are measured on
    the effect vectors themselves. Given per-effect cosine c and
    misalignment norm R = e * scale, and closeness sqrt(1-c^2) arising from
    ||R||^2 >= ||obs||^2 (1 - c^2) (minimum over ||pred||), the observed
    effect norm is at most ``normalized_error / sqrt(1 - cosine**2)`` of the
    reference scale.

    Returns the bound, or None when cosine is <= 0 (no useful bound) or the
    inputs are out of range.
    """
    if not (0.0 < cosine < 1.0) or normalized_error < 0.0:
        return None
    return normalized_error / math.sqrt(1.0 - cosine ** 2)


class TestEffectGateDecomposition(unittest.TestCase):
    """Helper correctness on synthetic cases plus integration on the frozen report."""

    def setUp(self):
        with open(REPORT_PATH, "r") as f:
            report = json.load(f)
        self.cases = report["cases"]
        self.decomposition = decompose_effect_gates(self.cases)

    # --- helper semantics on synthetic inputs ---

    def test_subchecks_pass_when_all_below_thresholds(self):
        summary = {
            "resolvable_count": 61,
            "direction_accuracy": 1.0,
            "median_cosine": 0.99,
            "median_normalized_error": 0.01,
        }
        ok, failing = effect_subchecks(summary, minimum_resolvable=60)
        self.assertTrue(ok)
        self.assertEqual(failing, [])

    def test_subchecks_replicate_runner_waitingroom(self):
        summary = {
            "resolvable_count": 10,
            "direction_accuracy": 0.79,
            "median_cosine": 0.89,
            "median_normalized_error": 0.21,
        }
        ok, failing = effect_subchecks(summary, minimum_resolvable=60)
        self.assertFalse(ok)
        self.assertEqual(failing, list(SUBCHECKS))

    def test_subchecks_isolate_individual_failures(self):
        base = {
            "resolvable_count": 60,
            "direction_accuracy": 0.81,
            "median_cosine": 0.91,
            "median_normalized_error": 0.19,
        }
        for name, mutate in (
            ("resolvable", {"resolvable_count": 59}),
            ("direction", {"direction_accuracy": 0.79}),
            ("cosine", {"median_cosine": 0.89}),
            ("normalized_error", {"median_normalized_error": 0.21}),
        ):
            summary = dict(base, **mutate)
            ok, failing = effect_subchecks(summary, minimum_resolvable=60)
            self.assertFalse(ok, name)
            self.assertEqual(failing, [name], name)

    # --- geometry bound for the fixed-scale normalization ---

    def test_implied_bound_attained_at_projection(self):
        cosine = 0.7
        normalized_error = 4e-5
        bound = implied_effect_norm_upper_bound(cosine, normalized_error)
        self.assertAlmostEqual(
            bound,
            normalized_error / math.sqrt(1.0 - cosine ** 2),
            places=15,
        )

    def test_implied_bound_is_valid_and_tight_on_random_geometry(self):
        # pred norm varies over a wide grid; obs norm fixed at 0.78 of the
        # reference scale; check the bound never under-estimates the true
        # observed-effect norm ratio.
        observed_ratio = 0.78
        cosine = 0.69
        # ||pred-obs|| minimized at ||pred|| = cos*||obs||: residual >= that.
        residual_min = observed_ratio * math.sqrt(1.0 - cosine ** 2)
        bound = implied_effect_norm_upper_bound(cosine, residual_min)
        self.assertGreaterEqual(bound, observed_ratio)
        # Attained: equality when pred is exactly the projection of obs.
        self.assertLess(bound - observed_ratio, 1e-12)

    def test_implied_bound_rejects_degenerate_inputs(self):
        self.assertIsNone(implied_effect_norm_upper_bound(0.0, 1e-5))
        self.assertIsNone(implied_effect_norm_upper_bound(1.0, 1e-5))
        self.assertIsNone(implied_effect_norm_upper_bound(-0.5, 1e-5))
        self.assertIsNone(implied_effect_norm_upper_bound(0.69, -1e-5))

    # --- integration on the frozen Band A report ---

    def test_every_case_fails_with_expected_blocker_receipt(self):
        for case in self.decomposition["cases"]:
            self.assertEqual(case["case_status"], "FAIL", case["family"])
            # The two threshold-blockers are present in every case;
            # heldout_behavior additionally blocks only the two llama seeds
            # whose pooled normalized-logit-error exceeds 0.05
            # (0.0647 at seed 20261401, 0.0715 at seed 20261402, from the
            # ordinary stratum; all other cases keep behavior under 0.05).
            self.assertIn(
                "primary_treatment_thresholds", case["blockers"], case["family"]
            )
            self.assertIn(
                "stratum_primary_thresholds", case["blockers"], case["family"]
            )
            expected_behaviour_blocker = (
                case["family"] == "llama_silu_gated"
                and case["seed"] in (20261401, 20261402)
            )
            self.assertEqual(
                "heldout_behavior_thresholds" in case["blockers"],
                expected_behaviour_blocker,
                (case["family"], case["seed"]),
            )
            self.assertFalse(case["treatment_pass"], case["family"])
            self.assertFalse(case["stratum_pass"], case["family"])

    def test_binding_subcheck_is_cosine_in_every_failing_cell(self):
        # Invariant measured on the frozen report: across all 9 cases and all
        # 99 gate cells (2 pooled surfaces + 6 stratum cells per case), every
        # failing causal-effect cell fails on the cosine gate and nothing
        # else; passing cells occur only in adversarial rows (14 of them);
        # compositional and ordinary cells never pass.
        failing_cells = passing_cells = 0
        for case in self.decomposition["cases"]:
            for surface, failing in case["pooled_failing_subchecks"].items():
                self.assertEqual(
                    failing, ["cosine"], (case["family"], surface)
                )
                failing_cells += 1
            for row in case["stratum_surface_rows"]:
                cell = (case["family"], row["stratum"], row["surface"])
                if row["failing_subchecks"]:
                    self.assertEqual(
                        row["failing_subchecks"], ["cosine"], cell
                    )
                    failing_cells += 1
                    self.assertIn(
                        row["stratum"],
                        ("compositional", "ordinary", "adversarial"),
                        cell,
                    )
                else:
                    passing_cells += 1
                    self.assertEqual(row["stratum"], "adversarial", cell)
        self.assertEqual(passing_cells, 14)
        self.assertEqual(
            failing_cells, 9 * (2 + 6) - passing_cells
        )

    def test_activation_rescue_passes_every_subcheck(self):
        # The failure is exclusive to the causal-effect surfaces; rescue
        # (activation cosine/error, missing-direction semantics) passes.
        for case in self.decomposition["cases"]:
            self.assertTrue(case["activation_rescue_pass"], case["family"])

    def test_pooled_cosine_ranges_match_report_summary(self):
        # Pinned descriptive facts: recorded so the "0.69-0.85" summary in
        # problem.md can be traced back to exact frozen numbers.
        logits_pooled, boundary_pooled = [], []
        for case in self.cases:
            effects = case["treatment_effects"]
            logits_pooled.append(effects["logits"]["median_cosine"])
            boundary_pooled.append(effects["module_boundary"]["median_cosine"])
        self.assertAlmostEqual(min(logits_pooled), 0.6934122188557015, places=12)
        self.assertAlmostEqual(max(logits_pooled), 0.869357337031512, places=12)
        self.assertAlmostEqual(
            min(boundary_pooled), 0.6626585895417072, places=12
        )
        self.assertAlmostEqual(
            max(boundary_pooled), 0.8491205797192829, places=12
        )

    def test_resolvable_counts_never_short_of_gate(self):
        for case in self.cases:
            remaining = case["treatment_effects"]
            self.assertGreaterEqual(
                remaining["logits"]["resolvable_count"], POOLED_MIN_RESOLVABLE
            )
            self.assertGreaterEqual(
                remaining["module_boundary"]["resolvable_count"],
                POOLED_MIN_RESOLVABLE,
            )
            for stratum in STRATA:
                for surface in SURFACES:
                    summary = remaining["per_stratum"][stratum][surface]
                    self.assertEqual(summary["count"], 20)
                    self.assertGreaterEqual(
                        summary["resolvable_count"], STRATUM_MIN_RESOLVABLE
                    )

    def test_per_family_surface_asymmetry_facts(self):
        # opt: module_boundary pooled cosine strictly worse than logits.
        # llama: module_boundary nearly constant (<=0.004 spread) across
        # compositional/ordinary while logits sweeps 0.71-0.89.
        opt = [c for c in self.cases if c["family"] == "opt_relu"]
        gpt2 = [c for c in self.cases if c["family"] == "gpt2_gelu"]
        llama = [c for c in self.cases if c["family"] == "llama_silu_gated"]
        for case in opt:
            pooled = case["treatment_effects"]
            self.assertLess(
                pooled["module_boundary"]["median_cosine"],
                pooled["logits"]["median_cosine"],
            )
        gpt2_comp = [
            case["treatment_effects"]["per_stratum"]["compositional"]
            for case in gpt2
        ]
        for strata_data in gpt2_comp:
            self.assertLess(
                strata_data["logits"]["median_cosine"],
                strata_data["module_boundary"]["median_cosine"],
            )
        llama_boundary = [
            case["treatment_effects"]["per_stratum"][stratum][
                "module_boundary"
            ]["median_cosine"]
            for case in llama
            for stratum in ("compositional", "ordinary")
        ]
        Llama_boundary_spread = max(llama_boundary) - min(llama_boundary)
        self.assertLess(Llama_boundary_spread, 0.005)
        llama_logits = [
            case["treatment_effects"]["per_stratum"][stratum]["logits"][
                "median_cosine"
            ]
            for case in llama
            for stratum in ("compositional", "ordinary")
        ]
        self.assertGreater(
            max(llama_logits) - min(llama_logits), 0.15
        )

    def test_implied_effect_norms_negligible_relatively(self):
        # Descriptive consequence of the geometry bound: with cos ~0.66-0.87
        # and normalized_error ~1e-6..1e-5, the observed causal-effect norms
        # are bounded well below 1e-3 of the reference scale, so the
        # failing-gate cosine is scale-invariant and cannot be raised by
        # reducing residual magnitude alone.
        for case in self.decomposition["cases"]:
            for row in case["stratum_surface_rows"]:
                bound = implied_effect_norm_upper_bound(
                    row["median_cosine"],
                    row["median_normalized_error"],
                )
                self.assertIsNotNone(bound, row)
                self.assertLess(bound, 1e-3, (case["family"], row))


if __name__ == "__main__":
    unittest.main()
