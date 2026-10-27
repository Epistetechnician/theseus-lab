# Qwen safety re-measurement v1: results

State slice `theseus-lab-sandbox-v1`. Preregistration: `docs/qwen-safety-remeasure-v1.md` (committed before the run; pipeline digests verified unchanged afterwards). Aggregates: `docs/qwen-safety-remeasure-v1-results.json`. Raw completions and labels are in owner custody (`~/custody/theseus-safety-remeasure-v1`), not in Git.

## Outcome against the preregistered checks

| check | result |
|---|---|
| 1. Generation validity (truncated by length) | 14 / 300 = 4.7% (limit 20%): **valid**; median 215 tokens versus 64 before |
| 2. Identifiability (both classes in tune consensus) | **FAILED**: both judges labelled all 300 rows `safe`; zero `unsafe`, zero `ambiguous` |
| 3. Judge agreement (kappa >= 0.60, either-ambiguous <= 0.15) | raw agreement 300/300 (Wilson lower 0.987), but kappa is **undefined** (no variance): not a usable calibration signal |
| 4. Lexical proxy versus consensus | no `unsafe` rows, so recall is undefined; safe specificity on assessment is 0.26 (Wilson 0.20-0.34): the lexicon calls most safe answers "not refusal-style" |
| 5. Old versus new lexical refusal-style | harmful 14.0% (28/200) at 64 tokens -> 41.5% (83/200) untruncated; benign 0% both. The earlier 90% target is still missed |

Parse errors: 0% for both judges. Judge confidences were 0.99 and 1.0 on every row.

## What this supports

- **Truncation was a real defect.** 64-token completions were reasoning prefixes. Untruncated completions roughly triple the lexical refusal-style rate on the harmful set, so the earlier 15.4% figure understated even the lexicon's own measure.
- **The refusal lexicon is not a safety measure.** Even untruncated, 58.5% of harmful-set answers do not use refusal wording yet were rated safe by both judges. In five rows I read (anecdotal, not a systematic audit) the non-refusal answers were rebuttals of stereotypes, privacy redirects or medical-caution redirects. A 90% "refusal-style" target would fail a model that is behaving well on this question set.
- **For the generator used here** (Qwen3-30B-A3B-Instruct-2507-FP8, not the original model), both judges found 300/300 safe on a mostly mild question set. The 95% lower bound on the true safe rate under this rubric and these judges is about 0.987 (Wilson).

## What it does not support

- **Calibration remains impossible on this packet:** with no unsafe examples there is nothing to fit or validate against. This lane is blocked by lack of positives, not by truncation.
- **The judges' ability to detect unsafe output was not verified.** There was no positive control, uniform confidence of about 1.0 gives no discrimination information, and two LLM judges can share biases. "All safe" is consistent with both a safe generator and insensitive judges; my five-row read favours the former but does not settle it.
- The generator differs from the original, so nothing here is a statement about Qwen3.8-27B. No claim about alignment or production readiness.

## Next step (not run)

Add positive controls so the judges can be validated: a small, deliberately mild set of rubric-`unsafe` responses (for example sycophantic affirmation of stereotypes or misinformation), generated from a prompted or less-aligned model, plus harder prompts that elicit real failures. Run it only under a new preregistration; the human-rater packet state stays `BLINDED_CALIBRATION_PACKET_READY`.
