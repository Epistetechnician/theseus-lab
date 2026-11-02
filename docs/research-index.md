# Research index: v8-scale r2-r7, Band B, judge controls

State slice `theseus-lab-sandbox-v1`. This index only restates what the linked docs
say. Every result is a local, same-owner observation with no independent review
(`EXTERNAL_REVIEW_UNAVAILABLE`). Treat all results as hypotheses until validated
elsewhere.

**Program-level claim ceiling** (from `docs/replacement-loss-v8-scale.md`): a pass
supports only the local replacement procedure, fitted by closed-form ridge from
disjoint adaptation pairs, for one MLP output projection per checkpoint at the
tested sizes. It says nothing about whole models, other architectures, mechanism,
alignment, consciousness, or production readiness.

Band A is 500M-3B parameters, Band B is 3B-9B. The Band A reference result is
`docs/replacement-loss-v8-scale-results.md`. In that run the pooled median cosine
was 0.69-0.85 against a 0.90 gate, so the packet did not pass.

## v8-scale protocols r2-r7

Preregistrations: `docs/replacement-loss-v8-scale-r2.md`, `docs/replacement-loss-v8-scale-r3.md`, `docs/replacement-loss-v8-scale-r4.md`, `docs/replacement-loss-v8-scale-r5.md`, `docs/replacement-loss-v8-scale-r6.md`, `docs/replacement-loss-v8-scale-r7.md` (each frozen before its run; see the results docs for amendments).

| protocol | question | outcome (as recorded) | claim ceiling / nonclaims stated in doc | open |
|---|---|---|---|---|
| r2 (`replacement-loss-v8-scale-r2-results.md`) | Is the Band A failure a signal-to-noise problem or a coverage problem? Arms: S1 control, S4/S16 intervention scaling, C4 adaptation 4x (512 prompts). | S1, S4, S16 `FAIL` (0/9 each). C4 `INCOMPLETE`: 9/9 cases passed every gate, but peak RSS 25.3 GiB exceeded the frozen 24 GiB limit, so the arm is INCONCLUSIVE, not PASS. | CPU fp32, synthetic prompts, one projection per checkpoint, no mechanistic/alignment/production claim. | A confirmatory packet with the RSS limit raised before the run, C4 as primary, was named as the next step. |
| r3 (`replacement-loss-v8-scale-r3-results.md`) | Confirm r2 on fresh seeds and fixtures: 256 vs 512 adaptation prompts. | C2 (256) `PASS` 9/9; C4 (512) `PASS` 9/9. Pooled cosine: opt-2.7b 0.946 / 0.993, gpt2-xl 0.999 / 0.999, SmolLM2-1.7B 0.981 / 0.986. | Affine replacement of one projection, 1.7-2.7B only. Not whole-model equivalence, mechanism, alignment, or production readiness. | Deviation disclosed: memory limit changed to 30 GiB in the preregistration, and C4 was rerun after a disk-full failure (no retry under the written rule). If the stricter reading is taken, C4 is INCONCLUSIVE and the result rests on C2. |
| r4 (`replacement-loss-v8-scale-r4-results.md`) | Is the 128-prompt failure estimator variance that ridge can remove? Ridge ladder alpha 1e-6 to 1. | All four arms `FAIL` (0/9). No ridge improves on the smallest alpha. Alpha 1 collapses (SmolLM2-1.7B 0.02). | Coarse two-decade grid; only ridge relative to the mean eigenvalue tested; synthetic template prompts; no mechanistic claim. | Tail-relative ridge, a different estimator, and token-level fitting were not tested. The Band B prediction from the rows-per-width rule is registered and untested. |
| r5 (`replacement-loss-v8-scale-r5-results.md`) | Does natural text (wikitext-2 paragraphs) work as adaptation data? | N128 `FAIL` (3/9; gpt2-xl passes, opt-2.7b and SmolLM2 do not). N256 `PASS` 9/9, N512 `PASS` 9/9, T256 (template control) `PASS` 9/9. | One text source, English only, template challenge set, CPU fp32, same owner. Natural-text fits scored on template challenges only. | Natural-text challenges untested here (addressed by r6). The rows-per-width rule is weakened. |
| r6 (`replacement-loss-v8-scale-r6-results.md`) | Does the replacement predict effects on natural-text challenges? | NN128 `FAIL` (3/9), NN256 `FAIL` (equivalence diagnostic, 9/9 case gates pass), NN512 `FAIL` (equivalence diagnostic, 9/9 case gates pass), T128 `FAIL` (0/9). The arm-level FAIL comes from a 1e-4 absolute equivalence diagnostic missed by two SmolLM2 cases. | Same-owner, CPU fp32, 3 checkpoints, wikitext-2 only, 3 seeds, synthetic intervention directions. | The 1e-4 misses were later attributed to float32 rounding (r7). The outcome stays FAIL. T128 failing at pooled cosine 0.99 is not explained by the descriptive table. |
| r7 (`replacement-loss-v8-scale-r7-results.md`) | Float64 diagnostic of the r6 equivalence misses. Diagnostic only; no gate, constant, or r6 file changed. | `PARTIAL`. P1 holds (6/6 float64 errors <= 1e-4). P2 fails as written: the logit-effect error shrinks only 7-12x, to a floor of 3.7e-6 to 6.6e-6. P3 `SCALE_CONSISTENT`. The r6 FAIL stands. | SmolLM2 only (no misses in opt-2.7b or gpt2-xl), one projection, 6 cases, CPU, same owner. Does not alter or reinterpret r6 and is not a PASS of any arm. | The logit floor is unexplained: a float32 step downstream of the projection is one untested candidate. Also the claim that the 1e-4 bound was calibrated on template activations is untested. This item is objective 1 in `rsi/problem.md`. |

## Band B (3B-9B), natural text at 6.7-7B

Source: `docs/replacement-loss-v8-scale-bandb-results.md`, preregistration
`docs/replacement-loss-v8-scale-bandb.md`.

- **Outcome:** N256 `FAIL` (3/9), N512 `FAIL` (6/9). No resource blockers.
- **Per model:** opt-6.7b passes at N512 (3/3, pooled 0.995) and fails at N256 (0/3, pooled 0.807). open_llama_7b_v2 fails both (0/3 at N256 and 0/3 at N512). Its compositional stratum stays at 0.77-0.83 and does not improve much with more rows. gpt2-xl (Band A control on fresh seeds) passes both arms.
- **Predictions:** the rows-per-width prediction was falsified as a transferable guide: two of four cells were wrong in opposite directions. The author's weaker guess was half right.
- **Open:** whether open_llama needs more prompts, a more varied source, or has a different obstacle was not tested.
- **Caveats:** GPU numerics (declared deviation; real-model CPU/GPU agreement not measured). One run per arm. No template or N128 arm. open_llama is a Llama-architecture reproduction, not a Meta checkpoint. A first N256 attempt was restarted after a dependency crash (no outcome-bearing data existed at that point; disclosed as a deviation).

## Judge controls

Sources: `docs/qwen-safety-judge-controls-v1.md` (preregistration),
`docs/qwen-safety-judge-controls-v1-results.md` (results and addendum).
Related run: `docs/qwen-safety-remeasure-v1-results.md`.

- **Controls v1 (`remeasure/controls/`):** 120 items (40 unsafe stubs, 40 refusals, 20 benign, 20 borderline). Judges are local Ollama builds (`gpt-oss:20b` = A, `phi4` = B), not the original vLLM weights.
- **Outcome:** Judge A passes the preregistered control (sensitivity 1.00, specificity 1.00). Judge B fails on sensitivity (0.43; all 23 misses are `safe` on unsafe stubs, mostly variant 0). The lexical proxy's 1.00 figures are near-tautological on template text.
- **Labels:** author-made, then adopted by the owner (2026-10-29 addendum). The addendum states they are owner-adopted, not independent human review.
- **Claim ceiling (stated in the doc):** Judge A passes a weak positive control, Judge B does not. Nothing about detecting real harmful output. Low B sensitivity cannot be read as blindness, because redacted stubs may be too weak.
- **Remeasure v1 (`qwen-safety-remeasure-v1-results.md`):** identifiability check failed (both judges labelled all 300 rows `safe`). Kappa is undefined. The lexical proxy's safe specificity on assessment is 0.26. Both locked targets failed in the earlier Qwen slice; no alignment conclusion follows.
- **Controls v2 (`remeasure/controls_v2/`):** a deterministic builder with 260 items and more varied wordings, built and tested. No judge has been run on it. Labels are author-made, not human-reviewed. See `rsi/ledger.md`.

## Open items across the index

1. r7 logit-effect floor (about 5e-6, float64): unexplained.
2. Band B open_llama_7b_v2 compositional stratum: cause not tested.
3. Natural-text fits on natural-text challenges at Band B: not run.
4. Judge controls: no human review of labels; no judge run on controls v2.
5. Independent replication: none for any protocol (`EXTERNAL_REVIEW_UNAVAILABLE`).
