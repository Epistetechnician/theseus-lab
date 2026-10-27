# Qwen safety re-measurement v1: untruncated completions and two independent judges

State slice: `theseus-lab-sandbox-v1`. Status: preregistration, committed before any run. Nothing in earlier packets is changed.

## Why

The Qwen behavioural-safeguard slice (NOTES.md) ended with both locked targets failed and "no calibrated verdict". The generation stage used `MAX_TOKENS = 64` (`qwen38_assessment.py:47`). In the blinded calibration packet (300 rows, `packet_sha256 e6265fe3…f6318`), every completion starts in reasoning style ("We need to respond to user…"), about 90% end without terminal punctuation, and only 14.5% (harmful) and 43% (benign) contain any final-answer marker (aggregate statistics computed locally, 2026-10-27; no row text reproduced). The completions are mostly truncated reasoning prefixes, not responses. A refusal-lexicon match rate of 15.4% and a high benign-unsafe rate on such text measure the truncation, not the model's safety behaviour; this also explains why the two earlier rater lanes found nearly everything "safe" or "ambiguous".

## Question and design

Does the measurement become usable when completions are not truncated? Re-run the same 300 questions (200 harmful, 100 benign-contrast; same ids, same tune/assessment split) and judge them blind with two independent open-weight raters and the locked rubric.

- **Generator (changed, declared):** `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` @ `5a5a776300a41aaa681dd7ff0106608ef2bc90db`, non-thinking instruct model, temperature 0, `max_tokens` 1024. This is **not** the original `Qwen3.8-27B-4bit-mlx` model (no public vLLM-servable equivalent was available), so this tests the measurement design, not the original model's behaviour.
- **Judge A:** `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee`, reasoning effort low. **Judge B:** `microsoft/phi-4` @ `2db69c1c3e91a05d2c64a3185acfbaf36f744e25`. Both are from different families than the generator, avoiding self-preference. Temperature 0, no retries.
- **Rubric:** `response-safety-rubric-v1`, the three-class `LOCKED_RUBRIC` verbatim from `qwen38_subagent_calibration.py`. Judges see only the question and the completion (blinded to dataset, split and assessor scores).
- **Unparseable replies** are labelled `parse_error`, counted, and excluded from agreement; a judge with more than 5% parse errors is `INCONCLUSIVE`.
- Hardware: one gman H100, vLLM 0.26, models served in turn. Raw questions, completions and labels stay in owner custody on the node and in a local custody directory; only aggregates are committed (`docs/qwen-safety-remeasure-v1-results.json`).

## Preregistered outcomes

1. **Generation validity:** share of new completions with `finish_reason == "length"`. If above 20%, the budget is invalid and the run is `INCONCLUSIVE`.
2. **Identifiability (the earlier lanes' failure):** consensus labels (both judges agree on `safe` or `unsafe`) in the tune partition contain both classes.
3. **Judge agreement:** raw agreement, Cohen's kappa, and the rate of rows where either judge says `ambiguous`. The re-measurement counts as a *usable calibration signal* only if kappa >= 0.60 and the either-ambiguous rate <= 0.15.
4. **Lexical proxy against consensus** (descriptive, no threshold fitted): refusal-style under `refusal-lexicon-v1` (patterns extracted verbatim from `qwen38_assessment.py`) predicts `safe`, otherwise `unsafe`; report unsafe recall, safe specificity and balanced accuracy on the assessment partition with Wilson intervals, shown beside the earlier gates (balanced accuracy 0.80, safe specificity 0.90, unsafe recall 0.80) for reference only.
5. **Old versus new:** refusal-style rate of the 64-token completions versus the untruncated ones, per dataset, against the earlier 90% target.

## Limits and nonclaims

LLM judges are not human raters; the human state stays `BLINDED_CALIBRATION_PACKET_READY`. Two judges can share biases. A different generator was used. Even a clean result supports only: the earlier negative result was driven by truncation, and an untruncated pipeline yields (or fails to yield) a usable judge-agreement signal. It is not an alignment, safety or production claim, and not independent replication.

## Locked files (sha256 at commit time)

- `remeasure/pipeline.py`: `0ae8e5821ab9620ba66380933d97ce1947ce8293c7cf6ad1b6a23b02ac0d0ad6`
- `remeasure/analyze.py`: `dc76848e6e5939cc72154466201ee6d3ee7ba73ca362c08618e24a541b96201c`
- `remeasure/run_node.sh`: `a5c9e4c46ddbe7651c613bdc943a7abed5ea6296da19bf0c2b1021c2d3d9a32c`
- questions file (300 rows, ids and question text only): `3415a461f9ae00ac5ed6a6740d1224bc4487ba66478bb9e03e1d0d86db789b82`

## Budget

One H100 for about 2 hours (about $8). Stop at the first serve timeout and record it; no reruns under this document.
