# Replacement-loss-v8-scale-r6: natural-text challenge arm

State slice: `theseus-lab-sandbox-v1`. Protocol ID: `theseus-replacement-loss-v8-scale-r6-natural-challenge`.

Status: preselection preregistration, frozen by `docs/replacement-loss-v8s6-freeze.json` before any run. All earlier packets remain immutable; the pinned chain includes every r5 file (the r5 packet's 49 files plus the r5 freeze itself).

## Question

r5 showed that replacements fitted by closed-form ridge on natural wikitext paragraphs pass at 256+ prompts, but scored them only on template challenges. Two caveats remain: (1) does a natural-fit replacement still predict intervention effects when the held-out challenge prompts are natural text too, and (2) r5's N128 failure was compared with template-128 baselines from different seeds. This packet closes both.

## Design (everything else pinned from v3/v8/r2-r5)

- Adaptation: natural arms reuse r5's selection verbatim (same pinned wikitext-2 parquet, same seeds `20261801`-`20261803`, same per-seed disjoint thirds and ranking), so NN128/NN256/NN512 adaptation prompts are exactly r5's N128/N256/N512 prompts.
- Challenge, natural arms: 60 natural paragraphs per seed (first 45 words, continuation ` Continued.`), drawn from that seed's third but only from paragraphs ranked beyond the 512 reserved for adaptation (so disjoint from every adaptation arm and from r5's), ordered by post-freeze secret entropy (not retained). Stratum labels (ordinary/compositional/adversarial, 20 each, assigned by rank) select the intervention direction type exactly as in v3; the prompt text is natural in all three.
- Challenge, template control: unchanged v3 template challenge (fresh entropy).
- Intervention magnitude, thresholds, controls, bootstrap, checkpoints (same three as r5), wall caps, 30 GiB memory limit and the r3 outcome re-derivation: unchanged.

| arm | adaptation | challenge |
|---|---|---|
| NN128 | 128 natural paragraphs (r5 N128 prompts) | natural |
| NN256 | 256 natural paragraphs | natural |
| NN512 | 512 natural paragraphs | natural |
| T128 (same-seed control) | 128 template prompts (64/32/32), seeds as N128 | template |

## Outcomes and interpretation (preregistered)

Per arm: `PASS` only if all nine cases and aggregate gates pass with no resource blocker; misses are `FAIL`; resource or identity problems are `INCOMPLETE`/`INCONCLUSIVE` and not retried. Descriptive: pooled and per-stratum median cosine by arm and model, adaptation rows per input width, and comparison to r5 (same adaptation, template challenge).
- **NN256 or NN512 PASS:** natural-fit replacements also predict effects on natural-text challenges; the coverage account is not confined to a template challenge distribution. Both PASS is the strongest reading.
- **NN256 and NN512 FAIL while r5 N256/N512 passed on templates:** the r5 pass depended on the template challenge distribution; the natural-challenge case is a genuine distribution-shift failure, to be stated as such.
- **NN128 vs T128:** NN128 passing more cases than T128 on the same seeds supports "natural text is a better adaptation set per prompt"; the reverse or a tie weakens it. This is descriptive on 3 seeds x 3 models, not a significance claim.
- **T128 PASS:** would contradict the earlier 128-template failures (Band A, r2-S1) and be reported as such.

## Limits and nonclaims

CPU fp32, same owner, one projection per checkpoint, same checkpoints as r5, wikitext-2 only (one domain, one language), intervention directions are still synthetic Gaussian/one-hot perturbations of projection inputs (only the prompts are natural), no independent review. Not whole-model equivalence, mechanistic understanding, alignment, or production readiness.

## Budget

Four gman `cpu-8` nodes (the r5 nodes, whose disks already hold the venv and checkpoints), about 1.5-2 hours each, cap $12.
