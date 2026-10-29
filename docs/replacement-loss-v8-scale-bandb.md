# Replacement-loss-v8-scale-bandb: does natural-text adaptation reach the 6.7-7B band?

State slice: `theseus-lab-sandbox-v1`. Protocol ID: `theseus-replacement-loss-v8-scale-bandb-natural-text`.

Status: preselection preregistration, frozen by `docs/replacement-loss-v8sb-freeze.json` before any Band B protocol run. All earlier packets remain immutable. Pinned chain extends r5 (`docs/replacement-loss-v8s5-freeze.json`).

## Question

r5 showed natural-text adaptation (wikitext-2 paragraphs) passes Band A (opt-2.7b, gpt2-xl, SmolLM2-1.7B) at 256 and 512 prompts. The Band B prediction registered in r4 was derived from a rows-per-input-width rule, and r5 weakened that rule (gpt2-xl passed at 1.04 rows per input dimension with natural text). Does natural-text adaptation at 256 and 512 prompts also pass for 6.7-7B models, and does the registered rule predict which arm/model passes?

## Design (everything else pinned from v3/v8/r2-r5)

- Adaptation: same natural-text protocol as r5 (wikitext-2 revision `b08601e04326c79dfdd32d625aee71d232d685c3`, 45-word prompts, per-seed disjoint thirds). Arms: **N256** and **N512** only (no N128, no template control: budget).
- Challenge fixtures, visible fixture, thresholds, controls, bootstrap, 2 h per-case and 18 h total wall caps, 9-case structure: unchanged. Natural-text fits are scored on the unchanged template challenge set (as in r5).
- Fresh seeds `20261901`-`20261903`, new `V8SB` namespace.
- Checkpoints (selected after freeze; revisions are full SHAs in the manifest): `facebook/opt-6.7b` (OPT/ReLU, fc2 width 16384), `openlm-research/open_llama_7b_v2` (Llama-architecture, SiLU-gated, down_proj input width 11008), and `openai-community/gpt2-xl` (GPT-2/GELU, width 6400) as a **Band A control family**. There is no eligible GPT-2-family checkpoint in 3-9B (`NO_ELIGIBLE_FRESH_TARGET`, as in the Band A packet), and the pinned v3 machinery requires exactly three families, so the third slot is filled by a reused Band A model on fresh seeds. Its results are reported separately from the Band B models; the primary reading is per Band B model.
- 3 seeds x 3 models = 9 cases per arm, as before.

## Deviations from the pinned lineage (declared before running)

1. **Device: GPU fp32 instead of CPU fp32.** Earlier packets ran on CPU. 7B fp32 forward passes on CPU do not fit the 2 h per-case cap within budget. This packet runs model forwards on one H100 in fp32 with TF32 disabled; captured activations, ridge fitting and scoring stay on CPU numpy/torch. Implemented in `replacement_loss_v8sb_runner.py` by exact-string patches of pinned v3 source at run time (counts asserted; v3 files are not edited). Checked on toy models: `scripts/v8sb_toy_device_check.py` runs the full pinned pipeline on CPU and GPU with identical (deterministically seeded) challenges; results are in the results doc. Numerics are not bit-identical to CPU; this is a stated limit.
2. **Peak host-RAM limit 100 GiB** (was 30 GiB; the node cgroup budget is about 115 GiB). Re-derived from the evaluator's recorded results exactly as r5 did for 24 -> 30 GiB.
3. **Package versions** follow `requirements.txt` (torch 2.6.0, transformers 4.57.3); the node image's own torch is not used.

## Registered predictions (written before any run)

The r4 rule predicted: at 256 prompts rows/width is below 1.5 for both 7B models and cosine below 0.90; Llama-style 7B needs about 400 prompts and OPT-6.7B about 600. Rows per prompt in r5 were about 53 (13.5k rows at 256). Expected rows / input width:

| model | width | N256 rows/width | N512 rows/width |
|---|---:|---:|---:|
| opt-6.7b | 16384 | about 0.8 | about 1.7 |
| open_llama_7b_v2 | 11008 | about 1.2 | about 2.5 |
| gpt2-xl (control) | 6400 | about 2.1 | about 4.2 |

- **P-rule (r4 rule, as registered):** N256 FAIL for opt-6.7b and open_llama_7b_v2 (pooled median cosine below 0.90); N512 PASS for open_llama_7b_v2, FAIL for opt-6.7b. gpt2-xl passes both.
- **P-weakened (post-r5 expectation, stated honestly as a guess, not a derived claim):** because r5 showed the rule over-predicts failure for gpt2-xl and under-predicts diversity effects, the author expects N512 to pass for both 7B models and N256 to be uncertain (more likely pass for open_llama_7b_v2 than opt-6.7b, whose rows/width is the smallest).
- Readings: contradicting P-rule on any 7B model/arm falsifies the rule as a transferable guide; matching P-rule on all four cells leaves it standing but weakened by r5. Neither outcome is a claim about mechanism.

## Outcomes and interpretation (preregistered)

Per arm: `PASS` only if all nine cases and aggregate gates pass with no resource blocker; misses `FAIL`; resource or identity problems `INCONCLUSIVE`, not retried. Descriptive: per-model pooled and per-stratum median cosine, cases passing, adaptation rows and rows per input width. A per-model table is the primary result because the arm-level gate pools in the gpt2-xl control.

## Budget and stop rules

One gman `h100-1` node (about $5.04/h). Spend cap $20 (authorized), hard cap $30. Order: N256, then N512 on the same node, run sequentially. If N256 shows N512 cannot finish inside the $20 cap, N512 is recorded `INCONCLUSIVE` (not run) with the reason. Any cap-exceeding or crashed arm is recorded as such and not retried beyond fixing a pure infrastructure fault that occurs before challenge fixtures exist (then disclosed). Every node started is stopped and checked with `gman node ls`.

## Limits and nonclaims

Single GPU/host, same owner, one projection per checkpoint, wikitext-2 only (English), template challenge set, GPU rather than CPU numerics, no independent review. A pass supports only that natural text works as adaptation data for this replacement at 6.7-7B on the template challenge; it is not whole-model equivalence, mechanistic understanding, alignment, or production readiness.
