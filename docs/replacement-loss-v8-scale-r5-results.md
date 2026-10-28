# Replacement-loss-v8-scale-r5 results: natural-text adaptation

State slice `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-r5-natural-text`, packet `8d4b6be42f34ce0a7bd5c685fde4d1ef064d2d78249e69be9e2fe1b84396171b` (identical in all four arms; refrozen once before any run with the pinned chain corrected, see commit history). Aggregate reports: `docs/replacement-loss-v8-scale-r5-{N128,N256,N512,T256}-report.json`.

## Arm outcomes (frozen gates, challenge set unchanged)

| arm | adaptation | outcome | cases passing | peak memory |
|---|---|---|---:|---:|
| N128 | 128 natural paragraphs | `FAIL` | 3/9 | 18.7 GiB |
| N256 | 256 natural paragraphs | `PASS` | 9/9 | 18.9 GiB |
| N512 | 512 natural paragraphs | `PASS` | 9/9 | 24.0 GiB |
| T256 (concurrent control) | 256 template prompts | `PASS` | 9/9 | 19.4 GiB |

Mean pooled median cosine of predicted vs observed logit effect, 3 fresh seeds (frozen gate 0.90); rows are adaptation token rows; width is the projection input width from the public model configs (opt-2.7b 10240, gpt2-xl 6400, SmolLM2-1.7B 8192; verify):

| model | arm | rows | rows / width | pooled | adversarial | compositional | ordinary | cases passing |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| opt-2.7b | N128 | 6,770 | 0.66 | 0.810 | 0.979 | 0.788 | 0.791 | 0/3 |
| opt-2.7b | N256 | 13,541 | 1.32 | 1.000 | 1.000 | 0.999 | 1.000 | 3/3 |
| opt-2.7b | N512 | 27,126 | 2.65 | 1.000 | 1.000 | 1.000 | 1.000 | 3/3 |
| opt-2.7b | T256 | 15,327 | 1.50 | 0.945 | 0.999 | 0.938 | 0.937 | 3/3 |
| gpt2-xl | N128 | 6,642 | 1.04 | 0.999 | 0.991 | 0.999 | 0.999 | 3/3 |
| gpt2-xl | N256 | 13,285 | 2.08 | 0.999 | 0.995 | 0.999 | 0.999 | 3/3 |
| gpt2-xl | N512 | 26,614 | 4.16 | 0.999 | 0.995 | 0.999 | 0.999 | 3/3 |
| gpt2-xl | T256 | 15,071 | 2.35 | 0.999 | 0.992 | 0.999 | 0.999 | 3/3 |
| SmolLM2-1.7B | N128 | 7,132 | 0.87 | 0.889 | 0.863 | 0.855 | 0.918 | 0/3 |
| SmolLM2-1.7B | N256 | 14,336 | 1.75 | 0.970 | 0.939 | 0.961 | 0.985 | 3/3 |
| SmolLM2-1.7B | N512 | 28,741 | 3.51 | 0.970 | 0.958 | 0.965 | 0.985 | 3/3 |
| SmolLM2-1.7B | T256 | 18,053 | 2.20 | 0.981 | 0.975 | 0.975 | 0.992 | 3/3 |

## Preregistered readings

- **N256 and N512 PASS, T256 PASS:** the coverage effect is **not specific to the template family**. Natural text of comparable size also suffices on the unchanged template challenge, despite the train/challenge distribution shift.
- **T256 replicates r3 C2 on fresh seeds** (0.945 / 0.999 / 0.981 vs 0.946 / 0.999 / 0.981).
- **N128 FAIL at arm level, as expected, but not uniformly:** gpt2-xl passes all three cases at 128 natural prompts (pooled 0.999), opt-2.7b (0.81) and SmolLM2-1.7B (0.89) do not. The preregistration said a pass would weaken the rows-per-width account; a per-model pass is exactly that kind of evidence.

## What this adds

- **Natural text is a better adaptation set per prompt than the templates.** At equal prompt counts it scores higher (opt-2.7b at 256: 1.000 vs 0.945; at 128: 0.81 vs about 0.70 in the earlier template baselines) while producing slightly fewer token rows.
- **Row count per input width does not explain the pattern by itself.** gpt2-xl with natural prompts reaches 0.999 at about 1.04 rows per input dimension, while the template baselines at 1.18 rows per dimension scored 0.85. Input diversity matters, not only quantity. The rows-per-width rule from the earlier descriptive table is therefore at best a rough guide for same-source data. **The registered Band B prediction was built on that rule and should be treated as weakened; it remains untested.**
- Together with r2-r4: more or more diverse adaptation data fixes the Band A failure; larger interventions and larger ridge do not.

## Limits and disclosures

- No concurrent template-128 arm: the 128-prompt template baseline comes from Band A and r2-S1 (different seeds), so the N128-vs-template-128 comparison is across seed sets.
- One text source and domain (wikitext-2), English only; the same three checkpoints as Band A; CPU fp32; synthetic template challenge set; one projection per checkpoint; same owner; no independent review (`EXTERNAL_REVIEW_UNAVAILABLE`).
- Natural-text fits were scored on **template** challenges. Whether replacements fitted on natural text predict effects on natural-text challenges was not tested.
- **Freeze correction:** the first r5 freeze (39 files) omitted the r3 and r4 packets from the pinned chain through a text-substitution slip. It was caught before any run, fixed, and refrozen (49 files). Both freezes are in Git history.
- N512's peak memory (24.0 GiB) sits at the old 24 GiB limit but under this lineage's 30 GiB limit; with the old limit it would have been recorded inconclusive.
- Not a mechanistic, alignment or production claim.
