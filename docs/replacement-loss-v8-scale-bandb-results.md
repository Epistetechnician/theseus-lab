# Replacement-loss-v8-scale-bandb results: natural-text adaptation at 6.7-7B

State slice `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-bandb-natural-text`, packet `6205046016f130b6cbaf98e9e13611f93ff5c7dde1a6b16b12da8dc412dc27ba` (55 files; identical in both arms). Aggregate reports: `docs/replacement-loss-v8-scale-bandb-{N256,N512}-report.json`. Preregistration: `docs/replacement-loss-v8-scale-bandb.md`.

## Arm outcomes (frozen gates, challenge set unchanged)

| arm | adaptation | outcome | cases passing | peak host RAM |
|---|---|---|---:|---:|
| N256 | 256 natural paragraphs | `FAIL` | 3/9 | 32.2 GiB |
| N512 | 512 natural paragraphs | `FAIL` | 6/9 | 32.7 GiB |

Both arms are `FAIL` at arm level because the 7B open_llama cases (and at N256 opt-6.7b) miss gates. No resource blockers.

## Per-model results (primary reading; mean over 3 fresh seeds, frozen gate 0.90)

Pooled/strata are median cosine of predicted vs observed logit effect. Width is the projection input width from public configs (verify).

| model | arm | rows | rows / width | pooled | adversarial | compositional | ordinary | cases passing |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| opt-6.7b | N256 | 13,573 | 0.83 | 0.807 | 0.982 | 0.801 | 0.792 | 0/3 |
| opt-6.7b | N512 | 27,035 | 1.65 | 0.995 | 1.000 | 0.994 | 0.994 | 3/3 |
| open_llama_7b_v2 | N256 | 15,302 | 1.39 | 0.879 | 0.979 | 0.768 | 0.868 | 0/3 |
| open_llama_7b_v2 | N512 | 30,485 | 2.77 | 0.923 | 0.982 | 0.830 | 0.921 | 0/3 |
| gpt2-xl (Band A control) | N256 | 13,317 | 2.08 | 0.994 | 0.970 | 0.994 | 0.996 | 3/3 |
| gpt2-xl (Band A control) | N512 | 26,523 | 4.14 | 0.995 | 0.975 | 0.995 | 0.996 | 3/3 |

Selected layers: opt-6.7b 30, open_llama 7, gpt2-xl 45 (all seeds). Worst full-case wall time 926 s (opt-6.7b N512); every case is far inside the 2 h cap.

## Against the preregistered predictions

- **P-rule (r4 rule):** predicted N256 FAIL for both 7B models (held), N512 PASS for open_llama (**contradicted**: pooled 0.923 but compositional stratum 0.83, 0/3 cases), N512 FAIL for opt-6.7b (**contradicted**: 3/3 pass at 1.65 rows/width, pooled 0.995). Two of four cells wrong in opposite directions, so the rows-per-width rule is falsified as a transferable guide, as the preregistration said a contradiction would mean.
- **P-weakened (author's guess):** expected N512 to pass for both 7B models and open_llama to do better than opt at N256. Half right: opt-6.7b passes at 512; open_llama does not, and at N256 opt was worse (0.81 vs 0.88), as guessed.
- **Reading:** opt-6.7b behaves like opt-2.7b in r5 (fails at 0.83 rows/width, passes at 1.65 with near-perfect cosine). The open_llama failure is stratum-specific: its pooled cosine is close to the gate and its adversarial stratum is fine, but compositional prompts stay at 0.77-0.83 and do not improve much from 256 to 512 (0.768 to 0.830). More rows from the same wikitext source help slowly there, so Band B does not show that natural text of this size suffices for every family. Whether open_llama needs more prompts, a more varied source, or has a different obstacle was not tested.
- gpt2-xl (Band A model on fresh seeds and GPU numerics) passes both arms with pooled 0.994-0.995, consistent with r5.

## Limits and disclosures

- **GPU, not CPU, numerics** (declared deviation). Toy check (`scripts/v8sb_toy_device_check.py`, CPU vs GPU with identical deterministic challenges, 9 toy cases): largest metric difference 0.017 (one direction decision in 60); mapping-aligned equivalence errors on GPU at most 6e-8. Real-model CPU/GPU agreement was not measured.
- **Peak-RSS limit 100 GiB**; actual peaks 32 GiB, so the change did not matter.
- **Refreeze:** the first freeze (`d9679430...`) had a re-entrancy bug in the device shim, found when writing the checkpoint manifest before any protocol run; fixed and refrozen (both in Git history).
- **N256 restarted once:** the first N256 attempt crashed at the Llama tokenizer (`protobuf`/`sentencepiece` missing from the pinned environment) before any challenge fixtures or predictions existed. I installed both on the node (not in `requirements.txt`, so the dependency set differs from the packet pin), wiped that arm's custody and reran from scratch, which rewrote that arm's checkpoint manifest (same models and revisions). N512 ran in the fixed environment from the start. This is a deviation from "no retries"; the failure preceded anything outcome-bearing.
- One run per arm, one host, same owner, no independent review (`EXTERNAL_REVIEW_UNAVAILABLE`). No template control or N128 arm. open_llama_7b_v2 is a Llama-architecture reproduction, not a Meta checkpoint. gpt2-xl revision is the current hub SHA at selection time. Natural-text fits scored on template challenges only. Not a mechanistic, alignment, or production claim.

## Cost

One gman `h100-1` node (about $5.04/h): created 18:23 UTC, stopped about 20:25 UTC, about 2.0 h, plus a first node of about 10 minutes that I deleted to switch images. About $11-12 by my own arithmetic from the node rate; I did not read a billing record. Under the $20 cap.
