# Replacement-loss-v8-scale-r5: is the coverage effect specific to template prompts?

State slice: `theseus-lab-sandbox-v1`. Protocol ID: `theseus-replacement-loss-v8-scale-r5-natural-text`.

Status: preselection preregistration, frozen by `docs/replacement-loss-v8s5-freeze.json` before any run. All earlier packets remain immutable.

## Question

In r2, r3 and r4 more adaptation prompts fixed the Band A failure (256 and 512 pass, 128 fails, and a larger ridge does not help), but every adaptation prompt so far was a synthetic template. Is "coverage" a general property of how much input-space the adaptation set spans, or an artefact of fitting on the same template family that the held-out challenge uses? This packet fits on natural text instead.

## Design (everything else pinned from v3/v8/r2/r3/r4)

- Adaptation prompts: paragraphs from `Salesforce/wikitext` revision `b08601e04326c79dfdd32d625aee71d232d685c3`, file `wikitext-2-raw-v1/train-00000-of-00001.parquet` (sha256 `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`, checked at run time; text is not stored in Git). Eligible: non-heading paragraphs of at least 60 words; each prompt is the first 45 words. Each seed draws from its own disjoint third of the pool (by hash), ranked by a seed-keyed hash.
- Challenge fixtures, visible fixture, checkpoints (the same three), thresholds, controls, bootstrap, wall caps, 30 GiB memory limit and the r3 outcome re-derivation: unchanged. So natural-text fits are evaluated on the unchanged template challenge set (a distribution shift by design).
- Fresh seeds `20261801`-`20261803`, new `V8S5` namespace.

| arm | adaptation prompts |
|---|---|
| N128 | 128 natural paragraphs |
| N256 | 256 natural paragraphs |
| N512 | 512 natural paragraphs |
| T256 (concurrent control) | 256 template prompts (as r3 C2) |

## Outcomes and interpretation (preregistered)

Per arm: `PASS` only if all nine cases and aggregate gates pass with no resource blocker; misses are `FAIL`; resource or identity problems are `INCONCLUSIVE` and not retried. Descriptive: pooled and per-stratum median cosine by arm and model, and adaptation rows per input width.
- **N256 or N512 PASS (with T256 PASS):** coverage is not specific to the template family; natural text of comparable size also suffices on the unchanged challenge.
- **N arms FAIL while T256 PASSES:** ambiguous between template-specific coverage and train/challenge distribution shift; resolving it needs a natural-text challenge set, which is out of scope here and stated, not assumed.
- **T256 FAILS:** the r3 C2 pass did not replicate on fresh seeds and the coverage finding is weakened.
- N128 is expected to fail (the 128-prompt template result); a pass would be surprising and would weaken the rows-per-width account.

## Limits and nonclaims

CPU fp32, same owner, one projection per checkpoint, same checkpoints as the packets that motivated this, wikitext-2 only (one domain, one language), no independent review. A pass supports only that natural text works as adaptation data for this replacement on the template challenge; it is not whole-model equivalence, mechanistic understanding, alignment, or production readiness.

## Budget

Four gman `cpu-8` nodes, about 1.5-2 hours each, cap $12.
