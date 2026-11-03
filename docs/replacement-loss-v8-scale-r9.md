# Replacement-loss-v8-scale-r9: news-domain replication of the natural-text arms

State slice: `theseus-lab-sandbox-v1`. Protocol ID: `theseus-replacement-loss-v8-scale-r9-news-domain`.

Status: preregistration, committed before any run and frozen by `docs/replacement-loss-v8s9-freeze.json` before checkpoint selection or any challenge generation. All earlier packets remain immutable and are pinned in the chain (r5, r6, r7 and Band B files that exist on main; r8 has no files on main at freeze time, which a test asserts).

## Question

r5/r6 found that an affine MLP-output replacement fitted by closed-form ridge on natural paragraphs passes all nine Band A cases at 256 and 512 adaptation prompts (fails at 128), with natural held-out challenges. Every natural-text arm used one corpus, wikitext-2. Does the result survive a different text domain?

## Design (everything not listed is pinned unchanged from r6)

- **Corpus:** CNN/DailyMail news, Hugging Face dataset `abisee/cnn_dailymail`, config `3.0.0`, file `3.0.0/validation-00000-of-00001.parquet`, dataset revision `96df5e686bee6baa90b8bee7c28b81fa3fa6223d`, file sha256 `65a5ccce932b08f050114ce6e3c39355d563fa24f194052bb2f27d1c6c499c91` (34.7 MB, 13,368 articles, column `article`). Read from the path in `THESEUS_CNNDM_PARQUET`; the loader refuses any file whose sha256 differs. Domain: English news prose (CNN and Daily Mail), differing from encyclopedic wikitext in register, entities and byline/boilerplate conventions.
- **Eligibility filter (mirrors the wikitext paragraph filter; tested equal to r5's function):** one article is one paragraph (the file has no internal newlines); whitespace-normalized; at least 60 words; not starting with `=`; exact-duplicate removal of the article text and, after the second freeze, of the 45-word prompt. Prompt = first 45 words, continuation ` Continued.`. The raw article text, including wire/byline boilerplate such as `(CNN)` or `By . Daily Mail Reporter .`, is deliberately not cleaned; that is part of the domain.
- **Selection:** same mechanics as r5/r6: a hash splits the eligible pool into three disjoint thirds, one per seed (`20261801`-`20261803`); adaptation takes the top-N by a seeded hash rank; the first 512 ranks are reserved for adaptation of any arm; challenge prompts (60 per seed, 20 per stratum, labels select intervention direction type exactly as in v3) are drawn from beyond rank 512 ordered by post-freeze secret entropy, with cross-family disjointness. The hash namespace uses the r9 protocol ID, so selections differ from wikitext's by construction.
- **Arms:** `NN256` (256 news adaptation / news challenge) and `NN512` (512 / news). No 128 arm and no template control are run in r9.
- **Models/seeds/layers/gates/thresholds/bootstrap/caps:** the same three Band A checkpoints as r5/r6 (opt-2.7b, gpt2-xl, SmolLM2-1.7B), 3 seeds, 9 cases per arm, frozen v3 case, cluster-bootstrap and control gates, 30 GiB memory limit with the r3 outcome re-derivation; no frozen constant or gate is changed. Only the corpus and its pins differ from r6.
- **Reporting change (no gate change):** the report additionally states, separately, (a) `per_case_gates`: cases passing, bootstrap gates, and their conjunction; and (b) `equivalence_diagnostic_1e-4`: per-case module/logit max-abs errors against the frozen 1e-4 bound. The arm `outcome_status` is the unchanged frozen aggregate, which includes the diagnostic.

## Predictions and decision rule (fixed now)

Per-case result := a case's `case_status == PASS` under the frozen per-case gates; per-case result for an arm := all nine cases pass AND the three cluster-bootstrap gates pass (`per_case_and_bootstrap_pass`). Resource/identity problems are `INCONCLUSIVE`, not retried beyond a rerun from scratch of an interrupted node.

Predictions:
- **P1:** NN512 per-case result = pass (9/9 and bootstrap).
- **P2:** NN256 per-case result = pass (9/9 and bootstrap). Less confident than P1: r6 NN256 passed on wikitext but 128 failed, so the news threshold could sit higher.
- **P3:** pooled median cosine of predicted vs observed logit effect >= 0.99 in every model/arm.
- **P4 (diagnostic):** the 1e-4 equivalence diagnostic fails in at least one SmolLM2 case in at least one arm and in no opt-2.7b/gpt2-xl case (as in r6/r7). If so, both arm-level `outcome_status` values are `FAIL` despite P1/P2, and this is reported as diagnostic-only, not as a per-case miss. r7 attributed it to float32 rounding; r9 does not retest that.
- **P5:** peak RSS stays under 30 GiB.

Classification (applies to per-case result, independent of the diagnostic):
- **`REPLICATES`:** per-case result = pass at both NN256 and NN512 (all 9 cases at each, plus bootstrap).
- **`DOMAIN_DEPENDENT_FAILURE`:** per-case result = fail at NN512, or at both arms. Any case-gate or bootstrap-gate miss at NN512 on news, with no resource blocker, counts; this means the wikitext 9/9 at 512 does not transfer. Which gate and models fail will be reported descriptively (a failure confined to one model is stated as model-by-domain, still counted as failure of the 9/9 claim).
- **`THRESHOLD_SHIFT`:** per-case pass at NN512 but fail at NN256: the pass/fail boundary depends on domain, the 512-prompt claim survives.
- **`INCONCLUSIVE`:** any resource blocker or identity problem prevents evaluation of an arm.
- The arm-level `PASS`/`FAIL` under the frozen aggregate (including the 1e-4 diagnostic) is reported alongside, unaltered. A diagnostic-only FAIL does not change the classification above, and is not a PASS of the arm.

Nonparametric descriptive extras (no significance claim): pooled and per-stratum median cosine by model and arm, side by side with r6 NN256/NN512.

## Limits and nonclaims

CPU fp32, same owner, one projection per checkpoint, same three checkpoints as r5/r6, 3 seeds, one additional domain (news, English) — two domains in total, not "natural text in general". Wikitext and CNN/DailyMail are both web-era English prose written for general readers; code, non-English and dialogue are untouched. The pool is the validation split of one dataset; articles are truncated to their first 45 words (ledes and boilerplate over-represented). Intervention directions are synthetic perturbations of projection inputs. Hypotheses until independently validated. No independent review. Not whole-model equivalence, mechanistic understanding, alignment or production readiness.

## Budget

Two gman `cpu-8` nodes (r5-n256/r5-n512 style: one arm per node), about 1.5-2 hours each; hard cap $15 for this run. Spend is reported from `gman usage`/`gman billing` before and after.
