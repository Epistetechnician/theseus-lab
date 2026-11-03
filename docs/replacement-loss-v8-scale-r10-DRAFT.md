# Replacement-loss-v8-scale-r10 (DRAFT, unfrozen): a float64 measurement gate

Status: **DRAFT preregistration. Not frozen. No runner, packet, node, or spend exists.** Nothing here changes any earlier outcome, gate, constant, or frozen file. Thresholds below were written before any r10 data exists; where they were informed by earlier measurements, that is stated, so a reader can discount them.

## 1. Motivation (from existing reports only)

- **r6** (`docs/replacement-loss-v8-scale-r6-results.md`): natural-text fit and challenge, Band A (opt-2.7b, gpt2-xl, SmolLM2-1.7B). At 256 and 512 prompts every per-case gate and the cluster bootstrap pass (9/9), but the arm outcome is recorded `FAIL` because the frozen mapping-aligned permutation diagnostic (absolute error <= 1e-4, float32) is exceeded in two SmolLM2 cases per arm.
- **r7** (`docs/replacement-loss-v8-scale-r7-results.md`): recomputing those cases in float64 gives module error about 1e-13 to 4.5e-12 (float32 reproduced the r6 values exactly), so the module-side misses are float32 rounding. A **logit-effect error floor of 3.7e-6 to 6.6e-6 remains in float64 and is unexplained**; its preregistered prediction P2 failed, label `PARTIAL`.
- **r8** (`docs/replacement-loss-v8-scale-r8-results.md`): for open_llama_7b_v2 at N512, a float64 downstream forward lifts compositional and ordinary logit-effect cosines from 0.80-0.93 to 0.996-0.999. The float32 effect vectors are only 2-4 float32 epsilons of the logit norm, so the float32 cosine was largely rounding noise. H1 (rounding floor) supported, H3 (architecture) rejected.
- **r9** (`docs/replacement-loss-v8-scale-r9-results.md`): the r6 pattern replicates on CNN/DailyMail news (9/9 per-case at 256 and 512), again with arm-level `FAIL` from the float32 diagnostic on SmolLM2 only.
- **Band B** (`docs/replacement-loss-v8-scale-bandb-results.md`, `docs/bandb-open-llama-diagnosis.md`): opt-6.7b `FAIL` at 256 (a real fit gap, module-boundary cosine 0.88) and `PASS` at 512; open_llama_7b_v2 `FAIL` at both.

**Reading to be tested:** in these experiments the float32 instrument, not the replacement, is the limiting factor for the equivalence diagnostic and for the cosine of small effects, *except* where there is a genuine coverage gap (opt-6.7b at 256), which float64 must not rescue.

## 2. What r10 is, and is not

r10 is a **new, separate measurement protocol** that re-measures *already-fitted* modules in float64. It does not refit, retrain, or re-select anything; fitted modules are the unchanged float32 fits cast exactly to float64. It **never reclassifies** an earlier arm: r6/r9/Band B outcomes stay as recorded. Its own outcome is a statement about the float64 instrument only.

## 3. Design

Read-only `diagnose`-style runs on the existing custody from the earlier runs (the pattern r7 and r8 used). Same checkpoints, seeds, prompts, interventions, and fitted modules; only forward/diagnostic precision changes.

| group | arms re-measured | purpose |
|---|---|---|
| Positive, Band A | r6 NN256, r6 NN512 (wikitext); r9 NN256, r9 NN512 (news) x 3 models x 3 seeds = 36 cases | per-case gates pass in float32; do they pass the float64 gate, including the SmolLM2 cases that missed 1e-4? |
| Positive, Band B | Band B N512: opt-6.7b, open_llama_7b_v2 (+ gpt2-xl) x 3 seeds = 9 cases | r8 covered open_llama only on custody N512; this adds opt-6.7b and the full arm |
| **Negative control** | Band B N256 opt-6.7b x 3 seeds (a real fit gap: module-boundary cosine 0.88, logit cosine 0.81) | float64 must **not** rescue a genuine coverage failure |

Note: r6 NN128 and T128 are deliberately **not** used as negative controls. Their pooled cosines (opt-2.7b 0.904, SmolLM2 0.969 at NN128) are already above 0.90 and they fail on other gates, so a float64 cosine gate would not be expected to fail them; using them would make the control unfalsifiable in the wrong direction.

Quantities recomputed in float64 for every case: (a) the mapping-aligned permutation diagnostic, module-boundary and logit-effect max absolute error; (b) per-stratum and pooled median logit-effect cosine; (c) module-boundary cosine.

**Availability risk (stated up front):** this needs the original custody directories (r6/r9 on `r5-*` nodes, Band B on `theseus-bandb-v8sb`). `r5-n512` has been reclaimed once already. If a custody directory is gone, that arm is **not regenerated silently**: regenerating changes the challenge entropy, so it is reported as `NOT_RUN: custody unavailable` and excluded, not substituted.

## 4. The float64 gate (written before data)

An arm-case passes the **float64 gate** if all of:

1. **Module side:** float64 mapping-aligned permutation error <= **1e-10** absolute.
2. **Logit-effect side:** float64 logit-effect permutation error <= **1e-4** absolute (the same numeric bound as the frozen float32 diagnostic, now applied at float64 precision; r7 measured 3.7e-6 to 6.6e-6 in float64, a 15x or larger margin).
3. **Cosine:** every per-stratum median logit-effect cosine >= **0.90** (the frozen case gate value, unchanged), computed in float64.

How the numbers were chosen, honestly: the 1e-10 module threshold is about 20x above the largest float64 module error r7 saw (4.5e-12) and 5 orders below the float32 bound, so it separates float64 rounding from a real mapping error. **It was informed by r7's data** (the same cases r10 re-measures for SmolLM2), so it is not a blind threshold; the Band A cases for opt-2.7b and gpt2-xl and all of Band B have no prior float64 module measurement, and those are the genuinely out-of-sample part. The logit-side 1e-4 reuses the frozen number deliberately so no new tolerance is invented for it. The unexplained 5e-6 floor is **not explained** by r10 and is neither claimed away nor used to relax anything; it simply sits well inside the stated bound.

## 5. Predictions and decision rules (numeric, before data)

- **P1 (Band A positives pass the float64 gate):** all non-`NOT_RUN` Band A positive cases pass conditions 1-3, including the 2-3 SmolLM2 cases per arm that missed 1e-4 in float32.
- **P2 (Band B positives):** opt-6.7b N512 and open_llama_7b_v2 N512 pass conditions 1-3 (open_llama per-stratum cosines >= 0.99, reproducing r8 within 0.005).
- **P3 (negative control fails):** Band B N256 opt-6.7b fails condition 3 (some stratum median cosine < 0.90) in all 3 seeds in float64 (float32 cosine 0.79-0.81; boundary cosine 0.88 shows a real fit gap). Float64 must not turn a coverage failure into a pass. There is a single negative-control arm (3 cases), which is a weakness: one arm cannot show the gate is discriminating in general.
- **P4 (no float64 module error above 1e-10):** across all positive cases.

**Classification of the r10 run:**
- `FLOAT32_WAS_LIMITING`: P1, P2, P3 and P4 all hold (excluding `NOT_RUN` arms, which are listed).
- `PARTIAL`: P3 holds but at least one positive case fails a condition; report which and by how much. (An example consistent with r7 would be a logit-effect error above 1e-4 on some arm.)
- `FALSIFIED`: the negative control passes the float64 gate (then float64 is not a discriminating instrument, and the "float32 was limiting" reading is not supported), **or** two or more Band A positive cases fail conditions 1 or 3 by an amount that float64 rounding cannot account for (module error above 1e-8, or a stratum cosine below 0.90).
- `INCONCLUSIVE`: fewer than 60% of planned cases are runnable (custody losses).

What would falsify the reading: a negative control passing, or positive-control failures that float64 does not remove. Either is reported as such.

## 6. Cost and resources (anchored on measured spend)

- Anchors: r8 float64 diagnose on an H100 for open_llama, about 5 minutes, **$0.36** (measured, `gman billing`); r9 two arms on cpu-8, **about $3.4**; r7 estimated $2-3 for two arms.
- Memory: float64 doubles weight memory. SmolLM2-1.7B (about 14 GB) and gpt2-xl (about 12 GB) should fit cpu-8 (32 GiB) as in r7; **opt-2.7b (about 22 GB weights) is tight on cpu-8**, so the plan puts opt-2.7b and all of Band B on an H100 (80 GB; r8 fit the 7B in float64).
- Estimate: roughly **$5-12** total (one H100 session plus a few cpu-8 hours). **Proposed hard cap: $20.** The estimate is mine and unmeasured for the new arms; the cap, not the estimate, is the commitment.

## 7. Threats to validity and claim ceiling

- Same owner, same three Band A checkpoints plus two 7B-scale checkpoints, 3 seeds, two corpora, one projection, synthetic intervention directions, no independent review.
- Re-measurement reuses fitted modules and challenge sets, so r10 cannot detect problems in the fits or the corpora themselves.
- The 1e-10 threshold is informed by r7 for SmolLM2 (see section 4).
- Even a `FLOAT32_WAS_LIMITING` result is a statement about the *instrument*: it does not turn any recorded `FAIL` into a pass, does not waive any frozen gate, and does not support a mechanistic, alignment, or production claim. Whether to adopt a float64 gate for *future* protocols would be a separate decision.

## 8. How to run once approved (not part of this draft's authority)

1. Review and edit this draft (thresholds, arm list, cap). Approve explicitly.
2. A runner reuses the `diagnose` pattern from r7/r8 (`replacement_loss_v8s7_*`, `replacement_loss_v8s8_*`), read-only on custody, float64 forward, with a new pinned-hash chain that includes r5 to r9 and Band B files (verify the file count in a test, not with `sed`).
3. Freeze the packet **before** any run; commit this preregistration (frozen form) before the runner.
4. Verify custody for each arm first; list any `NOT_RUN`.
5. Run H100 arms, then cpu-8 arms; stop every node; verify with `gman node ls`.
6. Write the results doc stating the classification, every deviation, and that all earlier outcomes are unchanged.
