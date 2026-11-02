# RSI ledger

## 2026-10-31 — objective 2 (judge controls v2), iteration 1

- **Hypothesis:** the v1 judge-control set has too little wording variety (2 unsafe stubs, 2 refusal texts, 3 borderline texts) to test judges against more than a few templates. A deterministic builder with combinatorial stub/refusal/borderline wordings, balanced labels, and paraphrased requests should raise variety without any operational content.
- **Change (additive only, no existing file edited):**
  - `remeasure/controls_v2/build_controls_v2.py`: builder (seed 20260501, reuses v1 HARM/BENIGN pools) and `--stats DIR` helper.
  - `remeasure/controls_v2/{questions,gen,labels_author}.jsonl`: generated output (same schema as v1).
  - `tests/test_controls_v2.py`: 9 tests (committed files match builder, determinism, unique ids and aligned files, label balance, per-request quota 4/4/1, unsafe stubs contain >=3 `[REDACTED` markers, distinct unsafe and refusal texts, 100 distinct requests, stats helper).
- **Measurement command:** `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls` (baseline) vs `... --stats remeasure/controls_v2` (after).

| metric | v1 (before) | v2 (after) |
|---|---|---|
| items | 120 | 260 |
| unsafe / safe_refusal / safe_benign / borderline | 40 / 40 / 20 / 20 | 80 / 80 / 80 / 20 |
| distinct unsafe completions | 2 | 80 |
| distinct safe_refusal completions | 2 | 80 |
| distinct borderline completions | 3 | 12 |
| distinct requests | 40 | 100 |
| distinct completions total | 27 | 192 |
| unsafe items all REDACTED | true | true |

- **Tests:** `PYTHONPATH=. python3 -m unittest tests.test_controls_v2` -> 9/9 OK.
- **Full public suite** (`scripts/run_public_tests.sh`): ok=17, skipped_owner_env=5, failed=42. All 42 failures are `ModuleNotFoundError` for torch, transformers, safetensors, or huggingface_hub, which are not installed in this sandbox (not installing them per rules). None import the new code. Not a regression from this change; not verified against a torch-equipped env.
- **Verdict: KEEP** (variety metrics improved; balance and redaction tests pass).

### Limits (not measured here)
- No judge was run. Labels are author-made, not human-reviewed.
- Benign answers are the v1 20 answers reused across 4 request prefixes, so only 20 distinct benign completions.
- Borderline has only 12 distinct texts (cycled over 20 items).
- Unsafe stubs remain template-generated; a judge may label them safe for lack of detail (the design limit already stated in docs/qwen-safety-judge-controls-v1.md).

### Open questions / next objectives
- Objective 2 follow-up: whether the v2 set should be preregistered in a new doc before any judge run (not done; no judge run was requested).
- Objective 3 (docs/research-index.md) and objective 1 (r7 float32 floor analysis) remain open.

## 2026-10-31 — objective 3 (docs/research-index.md), iteration 2

- **Hypothesis:** the r2-r7, Band B, and judge-control results are spread across ~20 docs with no single index. A doc that restates each protocol's outcome, claim ceiling, and open items (taken only from the existing docs) makes the state of the program readable without changing any result.
- **Change (additive only):** new `docs/research-index.md`. No existing file edited, no frozen packet touched, no source or test changed.
- **Measurement command:** for each results doc `docs/replacement-loss-v8-scale-r{2..7}-results.md`, `docs/replacement-loss-v8-scale-bandb-results.md`, `docs/qwen-safety-judge-controls-v1-results.md`, `docs/qwen-safety-remeasure-v1-results.md`, `docs/replacement-loss-v8-scale-results.md`, and each preregistration `docs/replacement-loss-v8-scale-r{2..7}.md`, check the basename is cited in the index (`grep -c -F <basename> docs/research-index.md`).

| metric | before | after |
|---|---|---|
| results docs cited in index | 0 / 10 | 10 / 10 |
| preregistrations cited (r2-r7, Band B, judge controls v1) | 0 / 8 | 8 / 8 |

  The 8 preregistrations are `replacement-loss-v8-scale-r2..r7.md`, `replacement-loss-v8-scale-bandb.md`, and `qwen-safety-judge-controls-v1.md`.

- **Tests:** none applicable (documentation only). No test was run and none was weakened.
- **Verdict: KEEP** (coverage went from 0 to full; the content was checked against the source docs, and no new claim was added).

### Limits
- Restatements are from reading the docs, not re-running any analysis. Each row carries the outcome as its doc recorded it.
- The index inherits the docs' own caveats (same-owner, no independent review, CPU/GPU numerics differences, template-built controls).
- Preregistration cross-check was a basename grep, not a content check.

### Open questions / next objectives
- Objective 1 (r7 float32/logit floor analysis, `docs/r7-logit-floor-analysis.md`) remains open. It is the next candidate: code reading plus a numpy toy.
- Controls v2 has no preregistration doc yet (noted in iteration 1).

## 2026-10-31 — objective 1 (r7 logit-effect floor), iteration 3

- **Hypothesis:** the unexplained ~5e-6 float64 logit-effect floor in r7 comes from a float32 rounding step in the logit path (C1), and a numpy toy with a float32 logit step should reproduce its order of magnitude. The float32 delta cast in `diagnose_case` (line 173) is a second candidate.
- **Change (additive only):** `docs/r7-logit-floor-analysis.md` (labelled hypothesis), `scripts/r7_logit_floor_toy.py` (toy), `tests/test_r7_logit_floor_toy.py` (4 tests). No existing file, freeze, report, or source edited.
- **Measurement command:** `python3 scripts/r7_logit_floor_toy.py --seed {0,1,2}` and `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy`.

| metric (toy, seeds 0/1/2) | before (r7 observed, float64 logit effect) | after: f64 control | after: f64_logit32 (C1) | after: f64_delta32 |
|---|---|---|---|---|
| logit effect max abs error | 3.7e-6..6.6e-6 | 1.2e-11 / 1.3e-11 / 9.3e-12 | 4.8e-7 (all seeds) | 1.2e-11 / 1.3e-11 / 9.3e-12 |
| module effect max abs error | 1e-13..4e-12 | 9e-12..1e-11 | same | same |

- **Tests:** 4/4 OK (determinism, float64 control floor-free, delta cast cancels, C1 ulp-scale).
- **Verdict: KEEP** as an analysis. The float32 delta cast is ruled out by code reading and by the toy (cancels between arms). C1 gives the right family of error but the toy is ~10x below the observed magnitude, so C1 is not confirmed. No causal claim about the real model is made.

### Limits
- Toy only; no model run, no torch, no weights. The real max |logit| and logit-path dtype are not recorded in the r7 report.
- The toy's float32 module error is 0 (activations too small to reproduce r7's float32 module error); it tests the logit mechanism only.
- The six observed floors are not clean multiples of one ulp, so a single rounding step does not fit the numbers cleanly.

### Open questions / next objectives
- Owner-env check (not run here): record max |logit| and logit dtype for r7 SmolLM2-1.7B cases; assert float32 vs float64 at the logits tensor in the installed transformers version. If C1 holds, a float64 logit path should bring the floor to about 1e-12 (diagnostic re-run only).
- Candidate C2 (float32 rotary tables) is untested.
- Objective 2 follow-up (preregistration doc for controls v2) still open.

## 2026-10-31 — objective 1 follow-up (candidate C2, float32 rotary tables), iteration 4

- **Hypothesis:** the float32 rotary cos/sin tables (cast to float64) carry a rounding error that cancels between arms when both arms read the same row per position. A permuted row lookup would leave a t-vs-a residue; if it is of the r7 floor's order (3.7e-6..6.6e-6), C2 is a candidate source.
- **Change (additive, plus one line edited in my own `docs/r7-logit-floor-analysis.md`):** `scripts/r7_rotary_float32_toy.py`, `tests/test_r7_rotary_float32_toy.py` (5 tests), and the C2 line in the analysis doc now states the toy result.
- **Measurement command:** `python3 scripts/r7_rotary_float32_toy.py --seed {0,1,2}` and `PYTHONPATH=. python3 -m unittest tests.test_r7_rotary_float32_toy`.

| metric (toy, seeds 0/1/2) | same_index: shift_err t-vs-a | perm_index: shift_err t-vs-a | perm_index: logit_effect_err (f64 tables) |
|---|---|---|---|
| before (no rotary test existed) | n/a | n/a | n/a |
| after | 2.3e-14 / 1.6e-14 / 2.5e-14 | 3.4e-7 / 4.7e-7 / 3.6e-7 | 7.5 / 5.6 / 6.6 |

- **Tests:** 5/5 OK. The C1 toy tests (4) still pass.
- **Verdict: KEEP as an analysis, C2 NOT confirmed.** Same-index rounding cancels. A permuted lookup residue is ~10x below the r7 floor. The permuted-lookup logit error itself is O(1), so a pipeline that actually mismatched positions per arm would not produce a 5e-6 floor. This is a point against that configuration, under the assumption that the r7 pipeline does not do that (not verified from the r7 code in this iteration).

### Limits
- Toy only; no model, no torch, no weights. Rotary is a simplified half-split form with base 10000, not the SmolLM2 config.
- The permuted-index scenario is hypothetical; I did not confirm from the r7 runner that arms use the same positions.
- Combined, C1 (~5e-7) and C2 (~4e-7) each land about an order below the observed floor. Neither reproduces it; the source remains unexplained.

### Open questions / next objectives
- Owner-env check (not run): confirm in the r7 runner whether both arms share the same rotary position indices, and record max |logit| and logit dtype for the six r7 cases. This decides whether C2's permuted-index case is even reachable.
- Objective 2 follow-up (preregistration doc for controls v2) still open.
- Objective 1 remains open at the level of a confirmed source. Objective 3 has no open item beyond re-checking the index against new docs.

## 2026-10-31 — objective 1 follow-up (code-reading check of arm construction), iteration 5

- **Hypothesis:** the open owner-env question from iteration 4 (do both r7 arms share rotary position indices?) can be answered from source. If the permutation acts on positions, C2's permuted-lookup case is reachable; if it acts on the channel axis, C2 is unreachable.
- **Change (additive, one addendum section in my own `docs/r7-logit-floor-analysis.md`):** no code, test, freeze, or result file changed.
- **Measurement (code reading, no model, no torch):** `replacement_loss_v8s7_runner.py` lines 126-149 (`_predictions`), `replacement_loss_v3_runner.py` lines 478-484 (`index_select(-1, perm)`), lines 575-578 and 746 (`raw.weight = treatment.weight[:, perm]`, exact index copy).

| question | before | after |
|---|---|---|
| do arms share token positions / rotary rows? | unknown (open) | yes: `input_permutation` acts on the channel axis only |
| is the permuted module an exact copy? | unknown | yes: index copy, no arithmetic |
| is C2 (permuted rotary lookup) reachable in r7 path? | untested | no (ruled out by reading) |

- **Tests:** `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 18/18 OK (no change to tests).
- **Verdict: KEEP as an analysis.** It resolves one open question. It does not reduce the floor and gives no numeric improvement; the floor source is still unexplained.

### Limits
- Reading only. The installed transformers version was not checked, so a float32 op inside the HF forward is not excluded.
- Summation-order estimate (about 1e-15 relative in float64) is an order-of-magnitude argument, not a measured value.

### Open questions / next objectives
- Owner-env check (unchanged, not run): record max |logit| and logit dtype in the six r7 cases; assert float32 vs float64 at the logits tensor in the installed transformers version.
- Objective 2 follow-up (preregistration doc for controls v2) still open; no judge run was requested.
- Objective 3 has no open item beyond re-checking the index against new docs.

## 2026-10-31 — iteration 6 (objective status check; no change)

- **Hypothesis:** none new. Checked whether any objective in `rsi/problem.md` still has a pure-Python, measurable improvement.
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:** `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 18/18 OK. Working tree clean before this entry; branch `rsi/go-1029`.
- **Status by objective:**
  - Objective 1 (r7 floor): C1 and C2 tested and not confirmed; delta cast ruled out; arm-sharing question answered by reading. What remains (max |logit| and logit dtype in the six r7 cases; dtype assertions in the installed transformers forward) requires torch and model weights. Not runnable here.
  - Objective 2 (controls v2): done in iteration 1 (KEEP). Only open follow-up is a preregistration doc for v2, which is documentation with no measurable target and no judge run was requested.
  - Objective 3 (research index): done in iteration 2 (KEEP). Index re-check against new docs only.
- **Verdict: no KEEP or REVERT.** Re-measuring the same baselines would not be progress, so the loop stops here per RSI.md.

### Proposed next objectives (need an owner-env or a new request)
1. Owner env, r7 six cases: record max |logit| and logit tensor dtype; assert float32 vs float64 at the logits in the pinned transformers version. Decides C1 (~5e-7 toy) vs. the observed 3.7e-6..6.6e-6 floor.
2. Owner env: check whether any buffer (rotary tables, attention mask) is left float32 after `model.to(float64)`. This is the main unchecked candidate from iteration 5.
3. Optional doc: `docs/controls-v2-preregistration.md` stating the v2 generator, seed, and label quotas before any judge run, as an additive file.

### Open questions
- Whether the r7 floor is reproducible at all on a float64 path is untested; the toy cannot settle it.

## 2026-10-31 — objective 2 follow-up (borderline wording variety), iteration 7

- **Hypothesis:** the borderline set has 20 items but only 12 distinct completions, because the generator cycles 3 openers x 4 bodies with `i % 12`. Adding two hedged openers gives 20 distinct combinations, one per item, with no operational content.
- **Change (additive to my own generator; regenerated outputs):** `remeasure/controls_v2/build_controls_v2.py` (+2 BORDER_OPENERS), `remeasure/controls_v2/gen.jsonl` (only the 8 borderline rows `ctl2-border-12..19` changed; unsafe, refusal, and benign rows are byte-identical because the shuffle order is unchanged), `tests/test_controls_v2.py` (+1 test: 20 distinct borderline completions).
- **Measurement command:** `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls_v2`, before and after.

| metric | before | after |
|---|---|---|
| distinct borderline completions | 12 | 20 |
| distinct completions total | 192 | 200 |
| unsafe / safe_refusal / safe_benign distinct | 80 / 80 / 20 | 80 / 80 / 20 (unchanged) |

- **Tests:** `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK (was 18; new test added).
- **Verdict: KEEP.** Measured improvement on borderline variety; no assertion removed (`git diff --cached` check for removed assert/expect/matches! lines: no hits).

### Limits
- Borderline texts are still template-built and author-labelled ("ambiguous"), not human-reviewed. Distinct strings do not imply distinct judge-relevant content.
- Benign answers remain the 20 v1 answers reused across 4 prefixes.

### Open questions / next objectives
- Objective 2 follow-up (preregistration doc for controls v2) still open; no judge run requested.
- Objective 1 still needs owner-env data (see iteration 6).

## 2026-10-31 — iteration 8 (objective status check; no change)

- **Hypothesis:** none new. Re-verified the state left by iteration 7 and checked whether any `rsi/problem.md` objective still has a pure-Python, measurable target.
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:**
  - `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.
  - `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls_v2` -> items 260, distinct_completions_total 200, distinct_requests 100, borderline 20 distinct, unsafe_all_redacted true. Matches iteration 7.
  - `python3 -c "import torch"` -> ModuleNotFoundError (expected; not installed, not installing).
  - Working tree clean before this entry; branch `rsi/go-1029`.
- **Status by objective:**
  - Objective 1 (r7 floor): C1 and C2 tested by toy and not confirmed; delta cast ruled out; arm-sharing settled by reading. Remaining checks need torch, model weights, and the owner's r7 cases. Not runnable here.
  - Objective 2 (controls v2): generator, tests, and wording variety done (iterations 1 and 7). Its remaining follow-up, a preregistration doc, has no measurable target and is not an objective in `rsi/problem.md`.
  - Objective 3 (research index): done (iteration 2). No new results docs to index.
- **Verdict: no KEEP or REVERT.** Re-measuring the same baselines is not progress, so the loop stops per RSI.md.

### Open questions / next objectives
- Owner-env (unchanged from iteration 6): record max |logit| and logit dtype for the six r7 cases; check buffers left float32 after `model.to(float64)`.
- Whether the owner wants a preregistration doc for controls v2 before any judge run (no request so far).

## 2026-10-31 — iteration 9 (objective status check; no change)

- **Hypothesis:** none new. Verified the iteration 8 state and re-checked whether any `rsi/problem.md` objective has a pure-Python, measurable target left.
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:**
  - `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.
  - `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls_v2` -> items 260, distinct_completions_total 200, distinct_requests 100, borderline 20, unsafe_all_redacted true. Unchanged from iteration 8.
  - Working tree clean before this entry; branch `rsi/go-1029`.
- **Status by objective:** unchanged from iteration 8. Objective 1 needs owner-env data (torch, weights, r7 cases). Objectives 2 and 3 are done (iterations 1, 2, 7).
- **Verdict: no KEEP or REVERT.** Re-measuring the same baselines is not progress, so the loop stops per RSI.md.

### Open questions / next objectives
- Unchanged from iteration 6: owner-env checks for r7 (max |logit|, logit dtype, buffer dtypes after `model.to(float64)`) and an optional controls-v2 preregistration doc, which needs an explicit request.

## 2026-10-31 — iteration 16 (objective 1 code-reading refinement; no numeric change)

- **Hypothesis:** the buffer-dtype candidate (float32 buffers such as rotary tables surviving `model.to(float64)`) can only affect the r7 floor if it differs between the two arms. Checked whether it can.
- **Change:** ledger entry only. No code, test, freeze, or result file changed.
- **Measurement (code reading):** `replacement_loss_v8s7_runner.py` lines 126-149 and 166-189. `_predictions` runs the treatment (`t`) and mapping-aligned (`a`) arms through the same `model` object and the same `rows`/`inputs`/`d` tensors. `diagnose_case` converts `model` in place (line 179) before each arm. Only the fitted module differs between arms (`modules["treatment"]` vs `modules["mapping_aligned_permutation"]`, an exact index copy per iteration 5).
- **Tests:** `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.

| question | before | after |
|---|---|---|
| can a model-internal float32 buffer differ between t and a arms? | unknown | no: both arms share the same model state, so such a buffer enters `t_log - a_log` identically |
| is the buffer candidate still open? | open (owner-env) | only if the floor is compared against a non-shared reference; not the t-vs-a effect the runner reports |

- **Verdict: no KEEP or REVERT.** This is a reading result that narrows the owner-env check: the buffer-dtype check is informative only for absolute error against a float64 ground truth, not for the t-vs-a logit effect that r7 reports. The floor source remains unexplained.

### Open questions / next objectives
- Owner-env (unchanged): record max |logit| and logit dtype for the six r7 cases; if the goal is absolute error, also check buffer dtypes after `model.to(float64)`.
- Objectives 2 and 3 remain done. No pure-Python, measurable objective is left in `rsi/problem.md`; the loop stops here per RSI.md.

## 2026-10-31 — iteration 10 (objective status check; no change)

- **Hypothesis:** none new. Confirmed the iteration 9 state still holds and that no `rsi/problem.md` objective has a pure-Python, measurable target left.
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:**
  - `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.
  - `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls_v2` -> items 260, distinct_completions_total 200, distinct_requests 100, borderline 20, unsafe_all_redacted true. Unchanged.
  - `python3 -c "import torch"` -> ModuleNotFoundError (expected; not installed, not installing).
  - Working tree clean before this entry; branch `rsi/go-1029`.
- **Verdict: no KEEP or REVERT.** Objective 1 still needs owner-env data (r7 max |logit|, logit dtype, buffer dtypes after `model.to(float64)`); objectives 2 and 3 are done. Re-measuring the same baselines is not progress, so the loop stops per RSI.md.
- **Next:** this loop cannot make measurable progress without the owner env or a new objective in `rsi/problem.md`.

## 2026-10-31 — iteration 11 (objective status check; no change)

- **Hypothesis:** none new. Checked whether the frozen r7 reports record the quantities the open owner-env checks need (max |logit|, logit dtype).
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:**
  - `git status --short` -> clean before this entry; branch `rsi/go-1029`.
  - `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.
  - `grep` over `docs/replacement-loss-v8-scale-r7-{NN256,NN512}-report.json` for max-abs, logit-dtype, and dtype keys -> only `logit_effect_max_abs_error`, `module_effect_max_abs_error`, `max_abs_boundary_output`, and `spearman_fp32_error_vs_max_abs_boundary_output` appear. No max |logit| and no logit dtype field.
- **Verdict: no KEEP or REVERT.** The reports lack the data needed for objective 1's owner-env check, so that check needs the r7 cases re-run in an owner env with torch. This is a reading result, not a re-run.

### Open questions / next objectives
- Unchanged from iteration 6. Objective 1 needs owner-env data (max |logit| and logit dtype per r7 case, buffer dtypes after `model.to(float64)`). Objectives 2 and 3 are done.
- Suggested addition to the owner-env diagnostic: write max |logit| and logit dtype into the report JSON so future reviews need no re-run to answer this.

## 2026-10-31 — iteration 12 (objective status check; no change)

- **Hypothesis:** none new. Re-verified the iteration 11 state and checked whether any `rsi/problem.md` objective has a pure-Python, measurable target left.
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:**
  - `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.
  - `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls_v2` -> items 260, distinct_completions_total 200, distinct_requests 100, borderline 20, unsafe_all_redacted true. Unchanged.
  - Index coverage: each of `replacement-loss-v8-scale-r{2..7}-results.md`, `-bandb-results.md`, `-results.md`, `qwen-safety-judge-controls-v1-results.md`, and `qwen-safety-remeasure-v1-results.md` is cited in `docs/research-index.md` (grep -c >= 1). Unchanged.
  - Working tree clean before this entry; branch `rsi/go-1029`.
- **Verdict: no KEEP or REVERT.** Objective 1 still needs owner-env data (r7 max |logit|, logit dtype, buffer dtypes after `model.to(float64)`). Objectives 2 and 3 are done (iterations 1, 2, 7). Re-measuring the same baselines is not progress, so the loop stops per RSI.md.
- **Next:** unchanged from iteration 11. The loop needs the owner env or a new objective in `rsi/problem.md`.

## 2026-10-31 — iteration 13 (objective status check; no change)

- **Hypothesis:** none new. Re-verified the iteration 12 state and checked whether any `rsi/problem.md` objective has a pure-Python, measurable target left.
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:**
  - `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.
  - `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls_v2` -> items 260, distinct_completions_total 200, distinct_requests 100, borderline 20, unsafe_all_redacted true. Unchanged.
  - `python3 -c "import torch"` -> ModuleNotFoundError (expected; not installed, not installing).
  - `grep` over `docs/replacement-loss-v8-scale-r7-*-report.json` for `max_abs_logit`, `logit_dtype`, `"dtype"` -> no matches (exit 1). Unchanged from iteration 11.
  - Working tree clean before this entry; branch `rsi/go-1029`.
- **Verdict: no KEEP or REVERT.** Objective 1 still needs owner-env data. Objectives 2 and 3 are done (iterations 1, 2, 7). Re-measuring the same baselines is not progress, so the loop stops per RSI.md.
- **Next:** this loop has been idle since iteration 6 (no measurable pure-Python target). It needs either the owner env (r7 max |logit|, logit dtype, buffer dtypes after `model.to(float64)`) or a new objective in `rsi/problem.md`.

## 2026-10-31 — iteration 14 (objective status check; no change)

- **Hypothesis:** none new. Re-verified the iteration 13 state; no `rsi/problem.md` objective has a pure-Python, measurable target left.
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:**
  - `git status --short` clean; branch `rsi/go-1029`.
  - `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.
  - `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls_v2` -> items 260, distinct_completions_total 200, distinct_requests 100, borderline 20, unsafe_all_redacted true. Unchanged.
  - `python3 -c "import torch"` -> ModuleNotFoundError (expected; not installing).
- **Verdict: no KEEP or REVERT.** Objective 1 needs owner-env data; objectives 2 and 3 are done (iterations 1, 2, 7). Re-measuring the same baselines is not progress, so the loop stops per RSI.md.
- **Next:** unchanged from iteration 13. Needs the owner env or a new objective in `rsi/problem.md`.

## 2026-10-31 — iteration 15 (objective status check; no change)

- **Hypothesis:** none new. Checked whether any `rsi/problem.md` objective has a pure-Python, measurable target left.
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:**
  - `git status --short` clean; branch `rsi/go-1029`.
  - `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.
  - `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls_v2` -> unsafe_all_redacted true; counts unchanged from iteration 7.
- **Verdict: no KEEP or REVERT.** Objectives 2 and 3 are done (iterations 1, 2, 7). Objective 1 needs owner-env data (r7 max |logit|, logit dtype, buffer dtypes after `model.to(float64)`), which needs torch and model weights and is not runnable here. Re-measuring the same baselines is not progress, so the loop stops per RSI.md.
- **Next:** needs the owner env or a new objective in `rsi/problem.md`. Loop has been idle since iteration 6; further turns without either will produce only identical entries.

## 2026-10-31 — iteration 17 (objective status check; no change, loop stopped)

- **Hypothesis:** none new. Verified the state left by iteration 16 and re-read `RESEARCH.md` (no governance rule beyond `AGENTS.md`).
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:**
  - `git status --short` clean; branch `rsi/go-1029`, HEAD `ee9e720`.
  - `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.
  - `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls_v2` -> items 260, distinct_completions_total 200, distinct_requests 100, borderline 20, unsafe_all_redacted true. Unchanged.
  - `python3 -c "import torch"` -> ModuleNotFoundError (expected; not installing).
- **Verdict: no KEEP or REVERT.** Objectives 2 and 3 are done (iterations 1, 2, 7). Objective 1 needs owner-env data (r7 max |logit|, logit dtype, buffer dtypes after `model.to(float64)`) and torch plus weights, which are not available here. Re-measuring the same baselines is not progress.
- **Verdict on the loop:** stopped. Further turns without the owner env or a new objective in `rsi/problem.md` will only add identical entries.

## 2026-10-31 — iteration 18 (objective status check; no change, loop remains stopped)

- **Hypothesis:** none new. Checked whether any `rsi/problem.md` objective now has a measurable, pure-Python target.
- **Change:** none to code, tests, docs, or frozen files. This entry only.
- **Measurement:**
  - `git status --short` clean; branch `rsi/go-1029`, HEAD `1bf6b2d` before this entry.
  - `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 19/19 OK.
  - `python3 remeasure/controls_v2/build_controls_v2.py --stats remeasure/controls_v2` -> items 260, distinct_completions_total 200, distinct_requests 100, borderline 20, unsafe_all_redacted true. Unchanged.
  - `python3 -c "import torch"` -> ModuleNotFoundError (expected; not installing).
- **Verdict:** no KEEP or REVERT. Objectives 2 and 3 are done (iterations 1, 2, 7). Objective 1 still requires owner-env data (r7 max |logit|, logit dtype, buffer dtypes after `model.to(float64)`), which needs torch and model weights and is not available in this sandbox.
- **Next:** unchanged from iteration 17. Resume only with the owner env or a new objective in `rsi/problem.md`.

## 2026-11-01 — objective 1 follow-up (logit-magnitude sweep), iteration 19

- **Hypothesis:** iteration 3's "C1 toy error is ~10x below the r7 floor" may have been a magnitude artifact: the toy ran only at |logit| ~ 30 (ulp 1.9e-6). If the real logit magnitude is 10^2..10^3, a float32 rounding step in the logit path should reproduce the observed 3.7e-6..6.6e-6 floor.
- **Change (additive):** `scripts/r7_logit_floor_sweep.py` (sweep over target_logit, reuses the iteration-3 toy; one backwards-compatible parameter added to `scripts/r7_logit_floor_toy.py:run(seed, target_logit=25.0)`), `tests/test_r7_logit_floor_sweep.py` (5 tests), and an addendum in my own `docs/r7-logit-floor-analysis.md`. Frozen packets, reports, and results docs untouched.
- **Measurement commands:** `python3 scripts/r7_logit_floor_sweep.py`, `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_sweep tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2`, plus a read-only per-case pairing of the frozen r7 reports.

| metric | before (iteration 3, |logit|~30 only) | after (sweep, seeds 0-2) |
|---|---|---|
| toy C1 max error | 4.8e-7 | 3.0e-8 .. 6.1e-5 across |logit| 2..1919 |
| toy C1 error at |logit| 212-530 | n/e | 3.8e-6 / 7.6e-6 (same order as observed 3.7e-6..6.6e-6) |
| err/ulp(max abs logit) | "0.250, looked stable" | lattice-valued 0.001..0.5, not a stable fraction |
| floor vs max_abs_boundary_output correlation | unknown | none: boundary-30 case (5.88e-6) exceeds two boundary-1600 cases (5.19e-6, 3.72e-6) |
| f4 logit floor across 6 cases | unknown | f32 = 3.6e-5..4.8e-5 (flat within 1.4x over 70x magnitude span); f64/f32 ratio 1/8..1/11.5 |
| tests | 19/19 | 24/24 OK |

- **Verdict: KEEP as an analysis.** The toy now reproduces a same-order floor (3.8e-6..7.6e-6) at |logit| ~ 10^2..10^3, so C1 is live again, with a narrowed, falsifiable condition: it explains the r7 floor iff the real max |logit| is in roughly the 10^2..10^3 band (measured err/ulp reaches 0.5, so bands below ulp ~ 7.6e-6 at |logit|~64 cannot produce the two largest floors at this sample count). This remains hypothesis: the real max |logit| is unrecorded (frozen reports lack it, confirmed again by reading the report JSONs). The earlier order-of-magnitude dismissal in the doc was corrected to cite the sweep.
- **Integrity:** no assertion, test, freeze, or check weakened; all changes additive. `git diff --cached | grep -E '^-.*(assert|matches!|expect)'` -> no hits before commit.
- **Open questions / next:**
  - Owner env (unchanged, now with a concrete threshold): record max |logit| per r7 case. Below ~16-30, C1 is quantitatively dead; in the 10^2 range, C1 is the default explanation and a float64 logit path should push the floor to ~1e-12 (diagnostic re-run only).
  - Complete two-level account (f64 floor 5e-6 and flat f32 floor 4e-5 with ratio ~1/8) from one mechanism is not yet toy-tested.
  - Objectives 2 and 3 remain done (iterations 1, 2, 7).

## 2026-11-01 — objective 1 follow-up (two-level toy: one mechanism, both floors), iteration 20

- **Hypothesis:** the open item from iteration 19 — one mechanism producing BOTH the f64 floor (3.7e-6..6.6e-6) and the flat f32 floor (3.6e-5..4.8e-5, ratio 1/8..1/11.5) — is satisfiable by a single float32 logit path: single final f32 cast in the f64 pass (C1), full f32 arithmetic in the f32 pass, with the f32 arm difference arising only from summation order in the permuted-channel f32 matmul (no injected noise, no tuned constant).
- **Change (additive):** `scripts/r7_two_level_toy.py` (toy; default seeds 0-2, targets 32/64/128/256, `--seeds/--targets/--json`), `tests/test_r7_two_level_toy.py` (5 tests), addendum in my own `docs/r7-logit-floor-analysis.md`. No existing file, freeze, report, result doc, or source edited.
- **Measurement commands:** `python3 scripts/r7_two_level_toy.py [--seeds 0..15 --targets 128,160,192]`, `PYTHONPATH=. python3 -m unittest tests.test_r7_logit_floor_toy tests.test_r7_logit_floor_sweep tests.test_r7_rotary_float32_toy tests.test_r7_two_level_toy tests.test_controls_v2`.

| metric | before (no two-level test existed) | after (toy, seeds 0-15) |
|---|---|---|
| f64-pass err / ulp(max logit) | unknown | per-seed constant on a power-of-two lattice {0, 1/256..1/2}, invariant across magnitudes 128..256 |
| f32-pass err / max logit | unknown | 1.6e-7..3.0e-7 relative; same family as real r7 f32 module relative error (7.6e-8..7.6e-7), emerged without tuning |
| f64 control (no f32 step) | iteration 3: ~1e-11 | ~5e-11 (floor-free) |
| simultaneous in-band cells | unknown | at max logit ~160: e.g. seed 8: err_f64 3.815e-6 (in 3.7e-6..6.6e-6), err_f32 3.815e-5 (in 3.6e-5..4.8e-5), ratio 10.0 (in 8..11.5) |
| tests | 24/24 | 29/29 OK (`PYTHON=python3 scripts/run_public_tests.sh`: ok=21, skipped_owner_env=5, failed=42 — identical failure/skip counts to the pre-change baseline; all 42 are torch/transformers imports) |

- **Verdict: KEEP as an analysis.** One mechanism (float32 logit path) reproduces both observed floors at a common logit magnitude L ~ 130-200, with in-band ratios straddling the observed 1/8..1/11.5. C1 sharpens into a testable conjunction: r7 max |logit| ~ 10^2..2x10^2, err ratio ~1/8..1/12 (already satisfied by the frozen reports), and a linear err_f32-vs-|logit| law implying similar max |logit| across the six cases. Still hypothesis: real max |logit| and logit dtype are unrecorded in any frozen r7 report.
- **Integrity:** all changes additive; no assertion, test, freeze, or check touched. `git diff --cached | grep -E '^-.*(assert|matches!|expect)'` -> no hits.
- **Two toy bugs found and fixed during this iteration (both in the new file, pre-commit):** (1) the aligned arm initially added the unpermuted delta (real runner permutes it, `changed.index_select(-1, perm)`); (2) the aligned arm initially consumed a permuted residual with a canonical readout, mixing channel orders — the real construction permutes only the module input, leaving residual and readout canonical. Both were measurement-construction errors; no existing file was touched.
- **Open questions / next:**
  - Owner env, now one decisive measurement: record max |logit| and logit dtype per r7 case. This simultaneously decides C1's magnitude condition (~10^2..2x10^2), the linear err_f32 law for the observed flat band, and whether a float64 logit path collapses both floors to ~1e-12 (diagnostic re-run only).
  - Objectives 2 and 3 remain done (iterations 1, 2, 7). No other pure-Python, measurable target is left in `rsi/problem.md`.

## 2026-11-01 — objective 1 follow-up (joint six-case feasibility scan), iteration 21

- **Hypothesis:** iteration 20 demonstrated the two-level mechanism (float32 logit path) at ONE toy cell (seed 8, |logit|~160) and left open whether it jointly fits ALL SIX frozen r7 case pairs, at which logit magnitudes, and with what rejection band. Predeclared reading: a case is "matched" at (seed, L) iff |toy-case|/case <= 0.25 on BOTH the f64 and f32 floors; the seed pool (0..63) and grid (64..320 step 8) were fixed before the recorded run; an initial 32-seed run gave the same verdict direction and was superseded by the 64-seed run reported here.
- **Change (additive):** `scripts/r7_joint_fit_scan.py` (read-only over the frozen reports), `tests/test_r7_joint_fit_scan.py` (7 tests), and an addendum in my own `docs/r7-logit-floor-analysis.md`. No frozen packet, report, result doc, or existing source edited.
- **Measurement commands:** `PYTHONPATH=. python3 scripts/r7_joint_fit_scan.py` (2,112 direct toy cells, ~66 s), `PYTHONPATH=. python3 -m unittest tests.test_r7_joint_fit_scan tests.test_r7_logit_floor_toy tests.test_r7_logit_floor_sweep tests.test_r7_rotary_float32_toy tests.test_r7_two_level_toy tests.test_controls_v2` -> 36/36 OK, and `PYTHON=python3 scripts/run_public_tests.sh` -> ok=22, skipped_owner_env=5, failed=42 (identical torch-import failure profile to the recorded baseline; +1 module vs iteration 20).
- **A pre-declared extrapolation failed and is reported honestly:** extrapolating per-seed err_f64 = k(seed)*ulp_f32(L) across magnitudes FAILS across binades (seed 0: 1.9e-6 at L~100 vs 3e-8 at L~128), so no analytic law was used; every cell was measured directly.

| metric | before (iteration 20) | after (this scan, primary ±25%) |
|---|---|---|
| cases matched by some (seed, L) cell on grid 64..320 step 8 | n/a (1 cell demonstrated, band-level) | 3/6 (NN256-case2 L 128-280; NN512-case1 L 88-216; NN512-case2 L 128-288); NN256-case0/1 and NN512-case0: NO match |
| joint common window (all six) | unspecified | EMPTY at ±25%; at ±50% sensitivity: L in [88, 288] |
| iteration-20 cell at case level | "in the bands" | matches NN512-case1 (+2.6% e64, +5.3% e32); -42% vs NN256-case2 |
| err/ulp structural cap | "reaches 0.5, not 1.0" (iteration 19) | reaches 1.0; it is a sample statistic, not a cap |
| tests | 29/29 | 36/36 OK |

- **Verdict: KEEP as an analysis; iteration 20's single-mechanism joint account is NOT confirmed and is now constrained.** The mechanism reproduces the collective band ranges but not the six case pairs jointly at the predeclared ±25% slack; it fits all six only at doubled ±50% slack (common window L in [88, 288]). The three unmatched pairs' f64/f32 ratios (0.109/0.134/0.138) sit in gaps between matched ones (0.103/0.110/0.165), so the binding constraint is rounding-boundary discreteness, not a band edge.
- **Integrity:** all changes additive; no assertion/test/freeze touched; `git diff --cached | grep -E '^-.*(assert|matches!|expect)'` -> no hits.
- **Open questions / next:**
  - Owner env (unchanged, sharpened): record per-case max |logit| and logit dtype. New sharp prediction: if the real max |logit| is inside [88, 288] and the real cases' boundary-hit draws and sample counts exceed the toy's (256 samples), the per-case ratio gaps can close; if a recorded max |logit| falls below the matched-case windows (e.g. < ~128 needed by the two largest e64 floors), C1 as sole source is falsified by this scan.
  - A larger toy sample count (n) at fixed seeds is the pure-Python probe of the "more samples close the gaps" half of that prediction (could be iteration 22).
  - Objectives 2 and 3 remain done (iterations 1, 2, 7).

## 2026-11-01 — objective 1 follow-up (sample-count probe: gaps not closable by n), iteration 22

- **Hypothesis:** iteration 21's open prediction — "if the real cases' sample counts exceed the toy's 256 samples, the per-case ratio gaps can close" — tested directly by re-running the two-level toy at n in {256, 1024, 4096} (predeclared, fixed before the recorded run) on seeds 0..15 over the same grid L = 64..320 step 8, with the iteration-21 match rule imported unchanged (±25% primary, ±50% sensitivity). This is the pure-Python half of that prediction; the owner-env half stays separate.
- **Change (additive):** `scripts/r7_samplecount_probe.py` (probe; imports `frozen_cases`/`cell_matches` from `scripts/r7_joint_fit_scan.py` so the rule cannot drift), `tests/test_r7_samplecount_probe.py` (7 tests), and an addendum in my own `docs/r7-logit-floor-analysis.md`. No frozen packet, report, result doc, or existing source edited.
- **Measurement commands:** `python3 scripts/r7_samplecount_probe.py` (1,584 cells, ~4 min), `PYTHONPATH=. python3 -m unittest tests.test_r7_samplecount_probe tests.test_r7_joint_fit_scan tests.test_r7_two_level_toy tests.test_r7_logit_floor_sweep tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 43/43 OK, and `PYTHON=python3 scripts/run_public_tests.sh` -> ok=23, skipped_owner_env=5, failed=42 (identical torch-family import-failure profile to the recorded baseline; +1 module vs iteration 21).

| metric | before (iteration 21, n=256, seeds 0..63) | after (this probe) |
|---|---|---|
| cases matched at ±25% | 3/6 | 3/6 at ALL of n=256/1024/4096, same three cases (NN256-case2, NN512-case1, NN512-case2) |
| ±25% common window (all six) | EMPTY | EMPTY at all n |
| ±50% common window | L in [88, 288] | L in [88, 264] / [104, 264] / [88, 248] |
| distinct err_f64/ulp levels over cells | 15 (n=256) | 15 -> 7 -> 5 as n grows |
| float64 control max | ~5e-11 | 1.4e-10..1.7e-10 (floor-free) |
| err/ulp max | 1.0 | 1.0 at all n (sup already reached at n=256) |

- **Measured verdict on the recorded prediction: the "more samples" half is REJECTED.** A 16x sample increase neither matches any unmatched case at ±25% nor opens the joint window. More samples simply concentrate the per-cell max statistic toward its sup (occupied err/ulp levels shrink 15 -> 7 -> 5; the low levels empty out as every seed's best sample rises).
- **New structural finding (measured, 528 grid cells + 32 fresh-seed cells at n=4096): the toy's err_f64/ulp(max logit) is power-of-two-valued ({.., 2^-4, .., 1.0}, never 1.5x a level).** Consequence: matchability at a ±X% per-case tolerance is magnitude-, seed-, and sample-INDEPENDENT and depends only on the mantissa of e64. Relative distances of the frozen e64 values to the nearest power-of-two level: NN512-case1 2.60%, NN256-case2 15.26%, NN512-case2 22.69% (matchable); NN256-case0 26.56%, NN512-case0 29.09%, NN256-case1 29.70% (impossible at ±25% for ANY (seed, L, n)). This reproduces exactly the iteration-21 3/6 split from the constraint measured above alone.
- **Verdict: KEEP as an analysis; the prediction under test failed cleanly and is recorded.** C1-as-toy-formulated is now constrained differently than before: the per-case gaps are NOT closable by sampling, magnitude, or seed; they close only if the real logit path's float32 rounding sites produce a denser achievable set than pure power-of-two multiples of one ulp (e.g. float32 steps at different ulp scales summing across binades), or C1 is falsified as the sole source for the six cases jointly. The mantissa distances are borderline (22.7%..29.7% against a 25% tolerance), so the criterion's tolerance sensitivity is real: at ±30% all six would match at some (seed, L, n); at ±25% exactly the frozen 3/6.
- **Integrity:** all changes additive; no assertion, test, freeze, or check touched; `git diff --cached | grep -E '^-.*(assert|matches!|expect)'` -> no hits (run before commit).
- **Limits:** toy only; the power-of-two property was measured on the declared grid plus 32 fresh cells, not proven over all (seed, L, n); the real cases' sample count, max |logit|, and logit dtype remain unrecorded in any frozen report; the mantissa criterion assumes the toy's achievable set (single-ulp power-of-two levels) applies to the real logit path, which is exactly what the owner-env check would test.
- **Open questions / next:**
  - Owner env (now a sharper, data-only decision): record per-case max |logit| and logit dtype. Given a recorded maximum L*, C1 predicts e64 = (power of two)·ulp_f32(L*) within tolerance; the frozen e64 mantissas make this fail for NN256-case0/1 and NN512-case0 under the observed achievable set. If the real recorded e64/ulp_f32(L*) values are not power-of-two-valued, the toy's single-rounding mechanism is not the real one and the applicable lattice must be denser (multiple float32 sites).
   - Untested here: a mechanism variant where a second float32 rounding site (e.g. the hidden-state path) adds a different-ulp contribution, which would fill the {0.085..0.194} err/ulp mantissa gaps the three unmatched cases need. This is the natural pure-Python iteration 23.
   - Objectives 2 and 3 remain done (iterations 1, 2, 7).

## 2026-11-01 — objective 1 follow-up (two-site variant: quenches instead of densifying), iteration 23

- **Hypothesis:** iteration 22's proposed variant tested directly: a SECOND float32 rounding site (the hidden state, per arm, before the readout matmul) contributing at a different ulp scale should fill the {0.085..0.194} err/ulp mantissa gaps the three unmatched cases (NN256-case0/1, NN512-case0) need. The arm differences are the recorded-scale f64 module differences (toy eps ~1e-13; r7 f64 module-effect 1e-13..4e-12), so the site can only contribute via arm-dependent f32 flips.
- **Change (additive):** `scripts/r7_two_site_toy.py` (probe; imports `_build` from `r7_two_level_toy.py` and the iteration-21 frozen-case reader so arm construction and match rule cannot drift), `tests/test_r7_two_site_toy.py` (8 tests), and an addendum in my own `docs/r7-logit-floor-analysis.md`. No frozen packet, report, result doc, or existing source edited.
- **Measurement commands:** `python3 scripts/r7_two_site_toy.py` (544 cells: seeds 0..31, L = 64..320 step 16, predeclared config; ~25 s), `PYTHONPATH=. python3 -m unittest tests.test_r7_two_site_toy tests.test_r7_samplecount_probe tests.test_r7_joint_fit_scan tests.test_r7_two_level_toy tests.test_r7_logit_floor_sweep tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 51/51 OK, and `PYTHON=python3 scripts/run_public_tests.sh` -> ok=24, skipped_owner_env=5, failed=42 (identical torch-family import-failure profile to the recorded baseline; +1 module vs iteration 22).

| metric | before (iteration 22, single-site C1) | after (two-site M2) |
|---|---|---|
| nonzero err_f64 cells | 544/544 | 153/544 |
| cells where the hidden cast ERASES a nonzero C1 floor | n/a | 391/544 (72%) |
| cells nonzero where C1 was zero | n/a | 0/544 |
| hidden-site arm flips per cell | n/a | 0 in 306 cells, 1 in 170, 2 in 68 |
| mantissa of err/ulp(max logit) | power-of-two-valued | STILL power-of-two-valued (all 153 nonzero cells) |
| min rel. distance to the three unmatched e64 | 0.2656 / 0.2970 / 0.2909 | identical values |
| cases matched at ±25% (f64 side) | 3/6 | 3/6, same three cases |
| tests | 43/43 | 51/51 OK |

- **Measured verdict on the recorded variant proposal: REJECTED as gap-filler and as r7-compatible.** The hidden-state f32 cast QUENCHES the C1 floor to exactly 0 in 72% of cells (the arm h values differ ~1e-12, far below f32 ulp at hidden magnitude ~1e-7, so both arms round identically; the C1 logit-boundary flips are destroyed downstream of the cast) and creates an error where C1 had none in 0/544 cells. Where the site does flip (0-2 flips per cell), its contribution exists (errs 1.16e-10..1.5e-5) but the max statistic STILL lands only on power-of-two multiples of the logit ulp, so the achievable mantissa set does not densify and the three unmatched cases stay at exactly the iteration-22 distances. Leg for compatibility: r7's six observed f64 floors are all nonzero, so a pipeline whose f64 pass carries a hidden-scale f32 site is excluded UNLESS the real arm differences at that site exceed the recorded module-effect scale by orders of magnitude (unrecorded). This is a measurement, not an argument: both directions were coded as probe decisions before the run.
- **Integrity:** all changes additive; no assertion, test, freeze, or check touched; `git diff --cached | grep -E '^-.*(assert|matches!|expect)'` -> no hits (run before commit).
- **Limits:** toy only; the grid is the predeclared 17 magnitudes x 32 seeds, not all (seed, L); hidden flip counts were measured on this grid only; real sample count, max |logit|, logit dtype, and real arm-difference scale at hidden magnitude remain unrecorded in any frozen report.
- **Open questions / next:**
  - Owner env (now decisive for the whole cast-lattice family, not just single-site C1): record per-case max |logit| and logit dtype, and the observed r7 arm hidden-state differences. Given recorded L*, e64 being a non-power-of-two multiple of ulp_f32(L*) falsifies the cast-lattice family as a whole (recorded in `docs/r7-logit-floor-analysis.md`, iteration 23 addendum).
  - The remaining pure-Python candidate family listed in the addendum: summation-order effects in the f32 pass entering the f64 comparison — a different construction from every cast site probed so far; untested.
  - Objectives 2 and 3 remain done (iterations 1, 2, 7).

## 2026-11-01 — objective 1 follow-up (summation-order family: dense mantissas, no joint account), iteration 24

- **Hypothesis:** iteration 23's residual candidate (b) — summation-order effects in the f32 pass entering the f64 comparison — tested directly as mechanism M4: the f64-pass readout reduction accumulated in float32 (every partial sum rounds to f32, NO logit cast), aligned arm adding the same term values in permuted channel order; hidden state and module matmuls exact f64; module arm difference at the recorded r7 scale (toy `eps` ~1e-13). The aim is the iteration-22 blocker: whether a dense achievable set (vs pure power-of-two multiples of one ulp) can close the three unmatched cases (NN256-case0/1, NN512-case0) and produce the joint 1/8 ratio.
- **Change (additive):** `scripts/r7_summ_order_toy.py` (probe; imports `toy._build` and `scan.frozen_cases`/`cell_matches` unchanged so arm construction and match rule cannot drift; sequential-f32-accumulation primitive; predeclared P1/P2/P3 reading), `tests/test_r7_summ_order_toy.py` (9 tests), and this addendum in my own `docs/r7-logit-floor-analysis.md`. No frozen packet, report, result doc, or existing source edited.
- **Measurement commands:** `python3 scripts/r7_summ_order_toy.py` (primary grid: seeds 0..15, L = 8..320 step 8, 640 cells, ~92 s; L starts BELOW the prior grids because M4's errors track magnitude), `python3 scripts/r7_summ_order_toy.py --seeds 0,1,2,3,4,5,6,7 --logits 2,4,5,6,7` (labeled sensitivity below the grid edge), `PYTHONPATH=. python3 -m unittest tests.test_r7_summ_order_toy tests.test_r7_two_site_toy tests.test_r7_samplecount_probe tests.test_r7_joint_fit_scan tests.test_r7_two_level_toy tests.test_r7_logit_floor_sweep tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 60/60 OK, and `PYTHON=python3 scripts/run_public_tests.sh` -> ok=25, skipped_owner_env=5, failed=42 (identical torch-family ModuleNotFound profile to the recorded baseline; +1 module vs iteration 23).

| metric | before: cast-family ceiling (iterations 21-23) | after: M4 probe |
|---|---|---|
| err_f64/ulp mantissa | power-of-two-valued at both single- and two-site variants | DENSE: 557/640 nonzero cells >2% off the nearest power-of-two (rel dist 0.0625..0.3125) |
| unmatched cases f64-matchable at +-25% | 0/3 (rel distances 26.6/29.1/29.7%, unreachable) | 3/3 (best deviations 0.010 / 0.108 / 0.112) |
| all six f64-matchable at +-25% | 3/6 | 6/6 (best devs 0.010..0.153) |
| where f64 matches sit | L 88-288 (joint window at +-50%) | L = 8 only; sensitivity L in [2,7] (8 seeds): f64 common window EMPTY, NN256-case2 unmatched |
| common window joint (both e64 and e32, +-25%) | EMPTY (iterations 21-23) | EMPTY |
| exact-f64 readout control | ~1e-11 | max 1.6e-10 (<= 1e-9 predeclared bound) |
| err_f64 range | up to 1.5e-5 | 3.8e-6 .. 3.1e-4 |

- **Measured verdict on the recorded candidate: LIVE as a mantissa densifier, REJECTED as a joint account.** Predeclared P1 (dense mantissas), P2 (all three unmatched cases f64-matchable at +-25%), P3 (control floor-free) all hold, so M4 is the first mechanism breaking the iteration-22 power-of-two-lattice ceiling: all six e64 become f64-matchable. But the both-sides joint window stays EMPTY, and the reason is now numeric: at the f64-matching regime L~8 the unchanged f32 pass errs are 1.7e-6..2.4e-6 (15-30x below the e32 band) and the in-cell ratio err_f64/err_f32 is 2.2-2.7, inverted by ~15-30x against r7's e64/e32 ~ 1/8..1/12. No tested mechanism (single- or two-site casts, iterations 21-23; f32-accumulated readout, this) produces both floors at one common magnitude.
- **New recorded fact (frozen reports, read-only):** all six frozen e32 are EXACT integer multiples of 2^-21 = 4.768e-7 (n = 76, 84, 90, 92, 94, 100; zero remainder), while all six e64 are off adjacent-integer position on that lattice by 7-40%. Reading (labeled hypothesis): for the f32-leg comparison chains the smallest-magnitude involved term sits in the [4,8) binade or finer, and the e32 mantissas are integer counts of that spacing; for the e64 legs a term FINER than 2^-21 spacing is involved. This constrains any future mechanism reading of both legs; binade membership is unrecorded, so it does not identify the mechanism.
- **Tests:** 60/60 OK across the r7 family + controls; 9 new tests (frozen-case values unchanged, tolerance/match rule not loosened, only-same-or-tighter reading, accumulation primitive rounds partial sums, addition order changes the result, mantissa helper, control floor-free, single-magnitude guard regression, measured small-grid structure incl. dense mantissas and empty joint window).
- **Integrity:** all changes additive; no assertion, test, freeze, or check touched; `git diff --cached | grep -E '^-.*(assert|matches!|expect)'` -> no hits (run before commit).
- **Limits:** toy only; the permuted reduction order is a HYPOTHETICAL realization of the aligned arm's f32 accumulation (not verified against torch internals; if the real aligned reduction order matched the treatment arm's, this probe overstates the family); grid is the predeclared 16 seeds x 40 magnitudes, not all (seed, L); e64 matches at the grid edge — a sub-8 regime was probed as labeled sensitivity only; real sample count, max |logit|, logit dtype, and per-chain binade membership remain unrecorded.
- **Open questions / next:**
  - Owner env (decisive between the two now-measured families): record per-case max |logit| and logit dtype. |logit| ~ 8-or-below favors the summation-family regime for e64; ~10^2 favors C1; the full joint account additionally needs the measured 1/8 ratio, produced by NO tested mechanism as yet.
  - Pure-Python: construct an f32-pass variant that reaches e32 ~4e-5 at the SAME L~8 regime (the joint-empty-window measurement half); or probe whether the f64 and f32 legs in r7 can use DIFFERENT effective aggregation scales, which the 1/8 ratio would require.
  - Objectives 2 and 3 remain done (iterations 1, 2, 7).

## 2026-11-01 — objective 1 follow-up (chained f32 leg: the joint window is NOT empty), iteration 25

- **Hypothesis:** iteration 24's declared open experiment measured directly: an f32-pass variant reaching e32 ~4e-5 at the SAME L~8 regime where M4's f64 leg matches e64 — i.e., whether the two r7 legs use DIFFERENT effective aggregation scales (the option the ledger question named). Mechanism M5: f32 leg = CHAINED float32 pipeline (K=24 tanh-MLP layers, width 512, replaced module at chain layer 0 with the recorded alignment construction, arms recomputed independently), f64 leg = M4 reused unchanged by import.
- **Change (additive):** `scripts/r7_chained_f32_toy.py` (probe; imports `readout_pass` from `r7_summ_order_toy.py` and `frozen_cases`/`CASE_TOL` from `r7_joint_fit_scan.py` so arm construction and match rule cannot drift; predeclared Q1-Q5 stated in the docstring before the recorded run), `tests/test_r7_chained_f32_toy.py` (12 tests), addendum in my own `docs/r7-logit-floor-analysis.md`. No frozen packet, report, result doc, or existing source edited.
- **Measurement commands:** `python3 scripts/r7_chained_f32_toy.py` (primary grid: seeds 0..7, L in {2,4,6,8,9,10,12,14}, 64 cells, ~44 s), `PYTHONPATH=. python3 -m unittest tests.test_r7_chained_f32_toy tests.test_r7_summ_order_toy tests.test_r7_two_site_toy tests.test_r7_samplecount_probe tests.test_r7_joint_fit_scan tests.test_r7_two_level_toy tests.test_r7_logit_floor_sweep tests.test_r7_logit_floor_toy tests.test_r7_rotary_float32_toy tests.test_controls_v2` -> 72/72 OK, and `PYTHON=python3 scripts/run_public_tests.sh` -> ok=26, skipped_owner_env=5, failed=42 (identical torch-family ModuleNotFound profile to the recorded baseline; +1 module vs iteration 24).
- **Two pre-run test-authoring corrections (my own transcription errors, fixed before the recorded run was finalized):** (1) no single toy cell jointly matches all six cases; the feasible object is the established common-window convention (every case has SOME jointly-matching seed at that magnitude) — my initial per-cell claim (seeds 0/6 at L=8, five more cells) misread NN512-case1's f32-only membership at those cells and was corrected; (2) the half-integer lattice assertion used round-half-up arithmetic and failed under banker's rounding; fixed to compare the fractional part.

| metric | before (iteration 24: M4 + 64-wide single-readout f32 leg) | after (M5) |
|---|---|---|
| f32-side per-case matches (+-25%) at L~8 | 0/6 (errs 1.7e-6..2.4e-6, 15-30x below the band) | 6/6 across L 6..14 (best devs 0.000-0.042) |
| common window f64-only / f32-only / JOINT | L=8 / EMPTY / EMPTY | L 8..10 / L 8..12 / **L in {8, 9} (NON-EMPTY)** |
| err_f32 at the L~8 regime | 1.7e-6..2.4e-6 | 2.8e-5..3.7e-5 (in the 3.6e-5..4.8e-5 band at band edges per seed) |
| chained-leg relative noise rel = err/L | n/a | 3.37e-6..4.66e-6, magnitude-flat (ratio 1.38 over L 2..14) |
| M4 exact-f64 readout control | <=1e-9 bound | max 7.0e-12 (holds) |
| tests | 60/60 | 72/72 OK |

- **Measured verdict: the iteration-24 recorded verdict ("no tested mechanism produces both floors at one common magnitude; joint window EMPTY") is SUPERSEDED.** The two-leg two-scale construction (different effective aggregation scales, as the open question named) produces both floors jointly at +-25% at L ~ 8-9: e64 from the M4 family (f32-accumulated readout, no logit cast), e32 from the chained f32 pipeline whose depth gives rel noise ~4e-6 so e32 ~ 4e-5 lands at L~8. The observed 1/8..1/12 e64/e32 ratio emerges from the legs' scale separation rather than from one site's binade geometry. C1's L~10^2 regime is now DISCRIMINATED from M5's L~8-9 regime by a single owner-env record (per-case max |logit|, logit dtype).
- **New recorded structural results (measured + code-derived, in the addendum):** (a) construction-level lattice rule — an f32-pass statistic (f64 difference of two f32-valued logits, per `_measure`/`_next_logits` in `replacement_loss_v3_runner.py`) is an EXACT multiple of 2^-21 whenever all involved |logit| >= 4, any binade >= [4,8) suffices; generalizes iteration 24's binade reading; verified in-toy (exact integer counts at L >= 32, a half-integer count where the argmax element involved a [2,4) logit). (b) under C1 the same rule applies to the f64-leg argmax element, so the frozen e64s being OFF the 2^-21 lattice forces C1's argmax elements to involve sub-4 logits while the f32-pass argmax elements (per the e32 lattice fact) involved >= 4 — a within-case argmax-binade inconsistency that C1 must satisfy and M-family need not.
- **Limits:** toy only — width scaled 2048->512 for runtime (depth 24 recorded), tanh stands in for the gated attention path, residuals omitted (bounded activations), the M4 f64 leg's f32-accumulated-readout site remains a hypothetical realization of an "f32 site inside the f64 pass" (never verified in torch internals); the chained-leg rel-noise flatness was measured on 64 grid cells, not proven for all (seed, L); no model run, no torch, no weights; real per-case max |logit| and logit dtype remain unrecorded in any frozen report.
- **Integrity:** all changes additive; no assertion, test, freeze, or check touched; the two pre-run test corrections above were transcription fixes, not loosening: the measured joint-window membership assertions became STRICTER (exact per-case membership transcription). `git diff --cached | grep -E '^-.*(assert|matches!|expect)'` -> no hits (run before commit).
- **Open questions / next:**
  - Owner env (unchanged, now decisive between families): per-case max |logit| and logit dtype. Recorded ~10^2 favors C1 (with the sub-4-argmax constraint now attached); recorded ~8-9 favors M5. The ~5e-6 floor analysis is at family-resolved state: families live = {C1 (constrained), M5 (joint account at +-25%)}, with the argmax-lattice discriminator applying at a diagnostic re-run only.
  - Pure-Python: per-case 1/8..1/12 ratio check at finer magnitude resolution; width/depth sensitivity (d 1024/2048) to bound the d=512 conclusion.
  - Objectives 2 and 3 remain done (iterations 1, 2, 7).
