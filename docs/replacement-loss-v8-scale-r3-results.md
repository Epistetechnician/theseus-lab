# Replacement-loss-v8-scale-r3 results (confirmatory coverage packet)

State slice `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-r3-coverage-confirm`, packet `655e7f21c7c27e04b0da58659ea62a964dff0dcd4dfc5882983dc9e76a5fdbdc` (identical in both arms). Aggregate reports: `docs/replacement-loss-v8-scale-r3-{C2,C4}-report.json`.

## Outcome: both preregistered arms `PASS` (`INTERNAL_PASS`)

| arm | adaptation prompts | outcome | cases passing | peak memory |
|---|---:|---|---:|---:|
| C2 | 256 | PASS | 9/9 | 19.6 GiB |
| C4 | 512 | PASS | 9/9 | 25.1 GiB (over the old 24 GiB limit; under this packet's preregistered 30 GiB limit) |

Mean pooled median cosine of predicted vs observed logit effect (3 fresh seeds each; frozen gate 0.90):

| model | 128 prompts (Band A / r2 S1) | C2 (256) | C4 (512) |
|---|---:|---:|---:|
| opt-2.7b | 0.70 | 0.946 | 0.993 |
| gpt2-xl | 0.85 | 0.999 | 0.999 |
| SmolLM2-1.7B | 0.82 | 0.981 | 0.986 |

Per-stratum cosines are all above 0.94 in both arms (weakest: opt-2.7b compositional/ordinary at 0.94 under C2); direction accuracy and held-out behavior gates pass in all nine cases of both arms (max next-token logit error 0.005 vs the 0.05 gate).

## What this supports

The r2 observation is replicated on fresh seeds and fixtures: the Band A failure of the independently fitted affine replacement was an adaptation-coverage problem, not a signal-to-noise problem. At 1.7-2.7B parameters, 256 adaptation prompts already pass every frozen gate and 512 passes with more margin, in all three tested families. Margin grows with prompt count (strongest for opt-2.7b: 0.70, 0.946, 0.993).

## Disclosures

- **Memory limit.** The 30 GiB limit (r2 used 24) was set in the preregistration before any run. The frozen v3 evaluator hard-codes 24 GiB, so this packet's runner re-derives the outcome from the evaluator's recorded case results with identical gate conditions and only the memory limit changed (unit-tested, including that failed cases, bootstrap and equivalence misses still FAIL). C4's 25.1 GiB peak would have been INCONCLUSIVE under the old limit; under this packet's limit it is not. C2 is unaffected (19.6 GiB).
- **A failed first C4 attempt.** C4 first ran on a GCP VM and died with `No space left on device` while writing adaptation captures (the VM disk was full of earlier custody roots and builds). Evidence: `docs/replacement-loss-v8-scale-r3-C4-vm-failed-attempt.log`. That failure came before challenge fixtures, predictions or outcomes existed in that run, so it could not have influenced results. The preregistration says resource terminations are not retried, so this is a deviation from the written stop rule: C4 was rerun unchanged (same packet, same checkpoints, same seeds) on a gman cpu-8 node with ample disk, and that rerun is the reported C4. If a stricter reading is preferred, C4 would be recorded INCONCLUSIVE and the confirmation rests on C2 alone.
- **No concurrent control.** No S1/128-prompt arm was re-run in this packet (stated in the preregistration); the comparison is against Band A and r2-S1 (0/9 cases passing in both).
- Both arms ran on gman cpu-8 nodes (C2 first, then C4, on the same node); the first C4 attempt ran on a GCP VM. Same three checkpoints as Band A/r2 (deliberate reuse, fresh seeds `20261601`-`20261603`). Overlap checks use regenerated visible/adaptation fixtures only.

## Limits and nonclaims

CPU fp32, same owner, synthetic prompts, one MLP output projection per checkpoint, no independent review (`EXTERNAL_REVIEW_UNAVAILABLE`). This supports only that the independently fitted affine replacement of that one projection predicts held-out causal effects when fitted from enough adaptation prompts, at 1.7-2.7B parameters. It is not whole-model equivalence, mechanistic understanding, alignment, or production readiness.
