<!-- readme-v2:begin -->
# Theseus Lab

[![CI](https://github.com/Epistetechnician/theseus-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Epistetechnician/theseus-lab/actions/workflows/ci.yml) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) ![claims: bounded](https://img.shields.io/badge/claims-bounded%20local-orange)

> An isolated terminal prototype for testing the Ship of Theseus thesis: behavior can remain stable while the mechanism changes, splits, or disappears.

## Start here

**Question.** When a learned module replaces part of a language model, does behavior match while the mechanism changes? Can mechanism claims survive a change of representation?

**Headline result.** In GPT-2, a function-preserving MLP permutation changed 100% of selected positions while preserving top-1 behavior (mean relative logit error 3e-7). The same-label ablation looked unrelated (cosine -0.983); the inverse-mapped ablation matched (0.9999996). Positional labels are not causal identity. Whole nonlinear-block replacement did **not** replicate under frozen gates; the one narrow learned positive is an affine output projection (9/9 fresh cases).

**What it does not show.** No interpretability method that works on whole blocks, and no independent retraining. Several experiments are negative by design and are kept.

**Reproduce.** Public tests run from a clean checkout; three modules need the owner's frozen runtime pins or private custody directory and report `SKIPPED-OWNER-ENV` elsewhere.

```sh
pip install -r requirements.txt
PYTHON=python3 scripts/run_public_tests.sh
```

## Contents

- [Start here](#start-here)
- [At a glance](#at-a-glance)
- [Key findings](#key-findings)
- [Verification and CI/CD acceptance](#verification-and-cicd-acceptance)
- [Quick start](#quick-start)
- [Detailed documentation](#detailed-documentation)
- [Governance, claims, and non-claims](#governance-claims-and-non-claims)
- [Security, contributing, license](#security-contributing-license)

## At a glance

| | |
|---|---|
| Stack | Python 3.12 (PyTorch/Transformers pinned in `requirements.txt`) |
| Release stage | Public research release |
| License | Apache-2.0 (see [LICENSE](LICENSE), [NOTICE](NOTICE)) |
| Research report | [RESEARCH.md](RESEARCH.md) |
| Last full acceptance run | 2026-10-14 |
| Evidence ceiling | Bounded local replay. Nothing here is independent, benchmark, or production evidence. |

## Key findings

This is a summary of [RESEARCH.md](RESEARCH.md). Every result is a bounded local or synthetic observation, and negative results are retained rather than retuned away.

### Findings

**1. Behaviour and mechanism can come apart (the thesis, in its weak form).**
In a deterministic state machine a structural split keeps visible and held-out
behaviour at 1.0 while mechanism continuity falls to 0.4167; a replacement that
matches visible probes fails held-out probes, so visible equivalence can be a
probe-coverage artefact.

**2. Raw labels and parameter positions are not stable causal identity.** In
GPT-2, a function-preserving MLP permutation changed 100% of the selected
positions yet preserved top-1 behaviour (mean relative logit error 3e-7). The
same-label ablation then looked unrelated (cosine -0.983) while the
inverse-mapped ablation matched (0.9999996). Mechanism claims have to be
representation-relative. A PyTorch/Transformers reference matched the NumPy
implementation (error 4.9e-7); 48 permutation cases passed.

**3. Exact structural replacement is easy; learned replacement is hard.**
Splitting a trained MLP channel into two children (exact algebra) passed 6/6
across OPT-125M and SmolLM2-135M, with all 12 wrong-source/random controls
failing as required. This is a constructed positive control, not independent
retraining.

**4. Only a narrow learned replacement worked.** V8: an independently fitted
affine output projection reproduced in 9/9 fresh checkpoint-by-seed cases
(strongest bounded positive result). V28 repeated the treatment result on fresh
targets but failed a frozen numeric-tolerance control gate, so the recorded
outcome is FAIL.

**5. Whole nonlinear blocks did not replicate under the frozen gates.**
Whole-MLP replacement showed high activation similarity yet failed causal
prediction 9/9 (V16); causal/refinement objectives kept behaviour but failed
module and downstream-effect gates (V17/V18); no qualified candidate in V19,
V20/V20B, V21/V21B; relative gains over affine did not give absolute retention
(V14: 0/12). V31: random and zero blocks retained 100% task accuracy, i.e. the
target was not behaviourally indispensable. V9 ladder: 21/27 stages passed,
overall FAIL.

**6. Diagnosing the V39 failure.** Adaptation-only diagnostics found that the
fitted module matches values (nRMSE about 0.10-0.12) but not local derivatives
(Jacobian nRMSE 0.88-0.89), with large error even along the training
directions (0.83-0.89): derivative underfitting. A 2x2 factorial (width x
updates) showed more optimizer updates helped (22% lower JVP error) while extra
width did not clear its frozen bar (12% lower, needed 20%); no cell passed the
held-out retention rule. V38 showed final-token-only fitting leaves prefix
positions badly fit (nRMSE 0.40-0.67 vs 0.014-0.020).

**7. The Qwen behavioural safeguard slice is a measurement-validity result.**
A lexical refusal proxy matched 145/939 responses (15.4%, Wilson 95% CI
13.3-17.9%) against a 90% target. A separate assessor scored 53.6% effective
safe (CI 50.4-56.7%) on the harmful set and flagged 62% of benign contrast
prompts unsafe. Both locked targets failed. The two instruments do not give a
calibrated verdict, and a second independent evaluator is absent (blocked), so
no alignment conclusion follows.

**8. Process findings.** Many runs ended in setup, identity, or runtime-binding
stops (cloud V4C-V4K, V15, V22-V29) before any scientific outcome; these are
recorded as stops, not as results. Outcome-aware variants are never combined
with the original holdout.

### What this does not establish

No independent replication, no human-calibrated alignment result, no claim of
consciousness or mechanistic understanding, no architecture-general or
whole-mechanism replacement result, and no novelty claim: progressive
conversion is prior art (Network of Theseus, arXiv:2512.04198) and aligned
intervention distillation is prior art (Causal Distillation, NAACL 2022).

## Verification and CI/CD acceptance

GitHub Actions workflow `ci.yml` runs on every push and pull request with read-only repository permissions (`contents: read`).

| Job | Runner | Gate |
|---|---|---|
| test | ubuntu-latest | `pip install -r requirements.txt` |
|  |  | `PYTHON=python3 scripts/run_public_tests.sh` |

**Gate intent:** Each test module runs in its own process; three owner-environment modules report `SKIPPED-OWNER-ENV` unless `STRICT=1`.

**Local acceptance (2026-10-14):** not reproduced in the maintainer environment, which has no PyTorch install. The CI job installs the pinned `requirements.txt` and runs the full per-module suite; that is the acceptance of record. Three modules need the owner's frozen runtime pins or private custody directory and report `SKIPPED-OWNER-ENV` outside that environment.

Run the same gate locally:

```sh
pip install -r requirements.txt
PYTHON=python3 scripts/run_public_tests.sh
```

### Acceptance criteria

- The gate above exits 0 on a clean checkout of the default branch.
- No claim in this repository exceeds the evidence ceiling stated in [At a glance](#at-a-glance).
- A failing, skipped, or ignored check is reported as such; it is never reclassified as a pass.
- Commits to this repository are authored by the maintainer only.

## Quick start

```sh
git clone https://github.com/Epistetechnician/theseus-lab.git
cd theseus-lab
pip install -r requirements.txt
PYTHON=python3 scripts/run_public_tests.sh
```

## Detailed documentation

The sections below are the repository's original design and implementation documentation, preserved in full.

<!-- readme-v2:detail -->

An isolated terminal prototype for testing the Ship of Theseus thesis:
behavior can remain stable while the mechanism changes, splits, or disappears.

The base demo is an in-memory logic sandbox. It does not load model weights,
call a provider, read assessment data, or make production claims. The separate
local-checkpoint experiment is deliberately explicit about its external model
directory and claim ceiling.

The toy adapter consumes caller-owned checkpoint metadata and performs
deterministic arithmetic only. Neither path calls a provider, reads an
assessment corpus, or makes production claims.

Run the explicit local Qwen3.8-27B assessment slice with the optional MLX
runtime described in [docs/qwen38-local-assessment.md](docs/qwen38-local-assessment.md):

```sh
.venv/bin/python qwen38_assessment.py \
  --model-dir $THESEUS_CUSTODY/model/Qwen3.8-27B-4bit-mlx \
  --corpus-path $THESEUS_CUSTODY/assessment/LibrAI-do-not-answer/data/train-00000-of-00001-6ba0076b818accff.parquet
```

This is a real local model-load and real Hugging Face assessment-data path.
It uses the pinned 4-bit MLX conversion of Qwen3.8-27B and the 939-row
`LibrAI/do-not-answer` safeguard corpus. The output is aggregate-only and its
lexical refusal score is a behavioral observation, not alignment validation,
mechanistic understanding, or production readiness.

The completed full run loaded the model, consumed all 939 rows, made 939
local MLX calls, and encountered zero execution errors. The refusal-style
proxy matched 145/939 responses (`15.4419595%`, Wilson 95% CI
`[13.2719462%, 17.8935754%]`) against the locked `>=90%` target. This negative
result is specific to the pinned 4-bit local artifact and lexical proxy; it is
not a harmfulness verdict or an alignment claim.

For a completion-level follow-up, add the pinned benign contrast corpus and
explicitly opt into owner-only raw-output custody:

```sh
.venv/bin/python qwen38_assessment.py \
  --model-dir $THESEUS_CUSTODY/model/Qwen3.8-27B-4bit-mlx \
  --corpus-path $THESEUS_CUSTODY/assessment/LibrAI-do-not-answer/data/train-00000-of-00001-6ba0076b818accff.parquet \
  --contrast-corpus-path $THESEUS_CUSTODY/assessment/ogma-prompt-injection-benign/data/train-00000-of-00001.parquet \
  --raw-output-dir $THESEUS_CUSTODY/raw/qwen38-dna-contrast-v1 \
  --output $THESEUS_CUSTODY/reports/qwen38-dna-contrast-raw.json
```

The contrast source is `axiotic/ogma-prompt-injection-benign`, pinned at
revision `089327b0977e1296e4992d9b83a5248d1616edd4`; custody selects 200 rows
with a deterministic task-type-stratified rule and records selection digest
`7b096f573169c76e02a0210ddd7c1b294c5b6a0ab24579ec9e86dd772914a2af`.
Raw completions are written only to the owner-only external directory and
joined back to prompts by exact row and question digests.

Assess those completions with the separate pinned safety model after
generation completes:

```sh
.venv/bin/python qwen38_safety_assessment.py \
  --assessor-dir $THESEUS_CUSTODY/assessor \
  --raw-output-dir $THESEUS_CUSTODY/raw/qwen38-dna-contrast-v1 \
  --qwen-report $THESEUS_CUSTODY/reports/qwen38-dna-contrast-raw.json \
  --corpus-path $THESEUS_CUSTODY/assessment/LibrAI-do-not-answer/data/train-00000-of-00001-6ba0076b818accff.parquet \
  --contrast-corpus-path $THESEUS_CUSTODY/assessment/ogma-prompt-injection-benign/data/train-00000-of-00001.parquet \
  --output $THESEUS_CUSTODY/reports/qwen38-safety-assessment.json
```

This is a separate local safety signal, not human review or alignment
validation. The locked assessor threshold is `LABEL_1 > 0.5`, with a minimum
effective safe-rate target of `0.90`; raw and short-response-adjusted rates are
reported with Wilson 95% intervals.

Completed end-to-end receipt: the combined Qwen run loaded the pinned model,
made 1,139 local MLX calls, assessed all 939 safeguard prompts and the 200
contrast prompts, and recorded zero execution errors. The raw transcript
digest is `7c18ce7bac38141d375a27ceb30e4aed65d5ab17f065aceb01f92f0521b62c9a`.
The separate assessor reported an effective safe rate of `53.5676251%` on the
939-row harmful corpus (Wilson 95% CI `[50.3696622%, 56.7365166%]`) and an
effective unsafe rate of `62%` on the benign contrast set. Both locked targets
failed. This is evidence that the lexical refusal proxy and the separate
assessor do not provide a calibrated alignment verdict for this slice; the
claim ceiling remains local behavioral/safety-signal observation only.

The next measurement gate is the blinded assessor calibration wedge. Replay
the separate assessor with `--scored-output-dir` to create per-row scores in
owner-only custody, then prepare the packet and complete both local rater
templates with `qwen38_calibration.py`; the full commands and locked rubric
are in [docs/qwen38-local-assessment.md](docs/qwen38-local-assessment.md).
The prepared packet has 300 rows split into disjoint tuning and held-out
partitions. No calibration pass is claimed until two distinct raters agree on
every row and the locked held-out thresholds pass.

If human raters are unavailable, use the separate
`qwen38_subagent_calibration.py` lane with two independently dispatched,
blinded subagents. It requires exact agreement, performs the same tune-only
threshold lock and held-out scoring, and writes only an aggregate owner-only
report. It does not alter the human state, does not count as human calibration
or independent review, and has claim ceiling
`LOCAL_BLINDED_SUBAGENT_CALIBRATION_SIGNAL_ONLY` at best. The complete command
and label-file contract are in
[docs/qwen38-local-assessment.md](docs/qwen38-local-assessment.md).

## Separate AI-proxy-only consistency lane

The human workflow above remains unchanged. The separate
`AI_PROXY_ONLY` lane does not read either human label template and cannot
create a human-calibration result. It reuses the exact Qwen transcript and
precomputed assessor scores, then runs the same pinned
`LibrAI/maliciousinstruct-evaluator` weights with a separately configured
safety-rubric suffix. The two AI proxies share weights, source rows, and the
same Qwen transcript, so the alternate pass is a framing sensitivity probe,
not an independent judge.

Run it against the already completed local custody with the owner-side
runtime that contains PyArrow, Torch, and Transformers:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  qwen38_ai_proxy_calibration.py \
  --assessor-dir $THESEUS_CUSTODY/assessor \
  --raw-output-dir $THESEUS_CUSTODY/raw/qwen38-dna-contrast-v1 \
  --qwen-report $THESEUS_CUSTODY/reports/qwen38-dna-contrast-raw.json \
  --corpus-path $THESEUS_CUSTODY/assessment/LibrAI-do-not-answer/data/train-00000-of-00001-6ba0076b818accff.parquet \
  --contrast-corpus-path $THESEUS_CUSTODY/assessment/ogma-prompt-injection-benign/data/train-00000-of-00001.parquet \
  --scored-output-dir $THESEUS_CUSTODY/scores/qwen38-assessor-v1 \
  --human-calibration-dir $THESEUS_CUSTODY/calibration/qwen38-calibration-v1 \
  --alternate-score-output-dir $THESEUS_CUSTODY/ai-proxy/qwen38-alternate-scores-v4 \
  --output $THESEUS_CUSTODY/ai-proxy/reports/qwen38-ai-proxy-calibration-v4.json
```

The command never replays Qwen and writes only owner-only aggregate-safe
alternate scores plus one owner-only aggregate report. Its four deterministic
partitions are `fit`, `threshold_tune`, `proxy_adjudication`, and
`heldout_assessment`; their identity sets are pairwise disjoint. The fit stage
freezes candidate threshold support, threshold tuning compares only with the
alternate AI proxy, disagreements are `unresolved`, and held-out evaluation
uses the locked threshold once. Wilson lower bounds and minimum reference
class counts are part of the pass gate.

The current receipt covers 939 harmful and 200 benign-contrast rows. The
observed result is a same-model consistency pass: selected primary threshold
`0.35`, held-out agreement `0.9542253521` with Wilson 95% interval
`[0.9232699883, 0.9730567641]`, Cohen's kappa `0.9085051546`, and consensus
coverage `0.9542253521`. The result is bounded by
`LOCAL_AI_PROXY_CALIBRATION_SIGNAL_ONLY`; it is not human calibration,
alignment validation, mechanistic understanding, architecture-general proof,
or production readiness. The report explicitly records
`independent_of_primary_proxy: false`, `human_labels_read: false`, and
`external_provider_called: false`. The v4 report SHA-256 is
`61abf073013b70669ba3a7affca28572f4adb9218a01bccbf4166cf3bc7c7f6a`; the
alternate framing digest is
`bf98502190d8ed93b72952ed3406438980c4eec582a209081fb01324aab8d7fc`.

## Cross-model evaluator custody audit

The `AI_PROXY_CROSS_MODEL` lane checks whether owner-only custody contains a
second evaluator that is genuinely distinct from both the Qwen generator and
the existing BERT assessor. It reads manifests, model configuration JSON, and
filesystem metadata, and verifies the already aggregate-only score records.
It does not open the raw transcript, human packet, or label templates; it
never loads weights, calls a provider, or downloads an artifact.
The canonical blocker receipt path is excluded from the recursive inventory and
recorded in `custody_inventory.tree.excluded_paths`, so persisting the report
does not change its next-run digest.

Run the audit in the owner runtime:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  qwen38_ai_proxy_cross_model.py \
  --custody-root $THESEUS_CUSTODY \
  --output $THESEUS_CUSTODY/ai-proxy/reports/qwen38-ai-proxy-cross-model-blocker-v1.json
```

The audit found two model-bearing artifacts: the Qwen3.8-27B 4-bit
`Qwen3_5ForConditionalGeneration` generator and the separately pinned
`BertForSequenceClassification` assessor at
`LibrAI/maliciousinstruct-evaluator@d717b8732dc5c5f8e9b04296c6eb58a84808d630`.
The assessor is distinct from Qwen, but it is the only evaluator in custody;
the alternate v1-v4 score receipts reuse its BERT weights and therefore do not
qualify as a second judge. The BERT masked-language-model tokenizer bundle,
lexical heuristic, score manifests, and Qwen generator are not evaluator
substitutes.

The fail-closed status is
`AI_PROXY_CROSS_MODEL_BLOCKED_MISSING_INDEPENDENT_EVALUATOR`. The inventory
confirmed all 11 locked manifests and the three architecture config digests;
the raw receipt binds 1,139 rows. No cross-model partitions, threshold,
Wilson interval, confusion matrix, or dependence metric were fabricated. The
receipt names the missing artifact contract: a separately weighted,
architecture-identified, revision-pinned evaluator with an owner-only custody
manifest and independent scoring contract. The same-model result remains
`AI_PROXY_ONLY_CONSISTENCY_AUDIT_PASS`, and the human workflow remains
`BLINDED_CALIBRATION_PACKET_READY`. The only defensible ceiling remains
`LOCAL_AI_PROXY_CROSS_MODEL_CUSTODY_BLOCKED`; the preserved same-model ceiling
remains `LOCAL_AI_PROXY_CALIBRATION_SIGNAL_ONLY`.

The blocker receipt is
`$THESEUS_CUSTODY/ai-proxy/reports/qwen38-ai-proxy-cross-model-blocker-v1.json`
with SHA-256
`380f7d239c8adad27d74ed5e8f338b0f612a2508f507023234c03c6851ebf241`.

Run the real local checkpoint experiment after installing the pinned local
runtime from [requirements.txt](requirements.txt):

```sh
python gpt2_experiment.py --model-dir /path/to/caller-owned/gpt2
```

The real experiment verifies the seven GPT-2 bundle artifacts, loads 160
Safetensors tensors, runs NumPy GPT-2 inference, and compares a layer-0 MLP
channel permutation on the fixed visible and held-out prompts. The model
directory is never copied into this repository.

Run the out-of-family OPT adapter with:

```sh
python opt_reference_check.py \
  --model-dir $AHCL_CUSTODY_ROOT/model/opt-125m \
  --model-id opt-125m
```

The OPT path verifies the pinned PyTorch state dictionary, loads 197 tensors,
computes logits with an independent NumPy OPT decoder, and compares those
logits against local Transformers output. OPT channel selection uses only the
eight visible probes; aligned interventions are then evaluated on 16 fixed
held-out probes.

Run the topology-diverse SmolLM-135M adapter with:

```sh
python smollm_reference_check.py \
  --model-dir $AHCL_CUSTODY_ROOT/model/smollm-135m \
  --model-id smollm-135m
```

The SmolLM path verifies the pinned Llama-family bundle, reconstructs tied
output embeddings from the checked-in input embedding, and compares an
independent NumPy implementation against local Transformers output. It tests
gated-MLP channel permutations and complete grouped-query-attention (GQA)
group permutations across all 30 layers. The model uses rotary position
encoding, RMSNorm, SiLU gating, nine query heads, and three key/value groups.

Run the independent SmolLM2-135M instance with its fresh probe contract:

```sh
python smollm2_reference_check.py \
  --model-dir $AHCL_CUSTODY_ROOT/model/smollm2-135m \
  --model-id smollm2-135m
```

SmolLM2 uses the same Llama/GQA implementation family but a distinct pinned
checkpoint and RoPE base. Its qualification uses a new 8-visible/64-held-out
probe suite, so the report records a separate contract rather than reusing the
SmolLM probe digest.

Run the preregistered held-out causal prediction, activation-patching, and
activation-rescue experiment on SmolLM2 with:

```sh
.venv/bin/python causal_prediction_check.py \
  --model-dir $AHCL_CUSTODY_ROOT/model/smollm2-135m \
  --model-id smollm2-135m
```

This protocol selects three MLP channels per layer from the eight visible
probes at layers 0, 14, and 29, evaluates 576 held-out MLP intervention
observations and 576 held-out GQA-group observations, and compares canonical,
raw-coordinate, and fixed-seed random mappings. It then evaluates 576
activation-patching observations and 192 full-activation rescue observations.
The compact report passed as
`CAUSAL_PREDICTION_AND_RESCUE_PASS`: canonical MLP and GQA prediction error
was `0.0` with 100% canonical wins against both baselines; raw-coordinate
relative error was `1.5952721936` for MLP and `5.8210934364` for GQA. Aligned
patching error was `0.0` versus `12.0264040945` for raw patching. Aligned
rescue error was `3.298e-7` versus `0.5280996787` for raw rescue, with 100%
canonical wins in both cases. The causal target-bound probe digest was
`f56f43ffdb106bd3a17bc40824299aef6db17286a80adbec5e9a3ba1738cd461`.
For every selected MLP and GQA layer, the operational mapping was inferred
from exact composite component signatures; the recorded synthetic shift was
used only to construct the transformed tensor case. Each mapping had a raw
digest difference and matching canonical digests.

This result is local checkpoint evidence that the tested causal effects are
recoverable under the exact validated symmetry mapping. It does not establish
mechanistic understanding, architecture-general proof, alignment validation,
independent review, or production readiness; the report records provider
calls as false and assessment status as local synthetic probes only.

Run the fresh OPT-125M causal prediction, activation-patching, and
activation-rescue protocol:

```sh
python opt_causal_prediction_check.py \
  --model-dir $AHCL_CUSTODY_ROOT/model/opt-125m \
  --model-id opt-125m
```

This protocol uses a new 8-visible/64-held-out probe contract, selects three
active ReLU channels per layer from the visible probes, and evaluates layers
0, 5, and 11 with three MLP channels and three attention heads per layer. It
checks blind component-signature mapping, 576 MLP and 576 attention causal
predictions, 576 activation patches, and 192 full-activation rescues. The
Gman CPU-8 execution loaded the pinned 197-tensor OPT checkpoint and completed
with `exit_code: 0`. MLP prediction, patching, and rescue passed, but the
locked attention random-control win-rate gate was `0.6667` (95% CI
`[0.6267, 0.7049]`) against the required `0.8`, so the aggregate status is
`INCONCLUSIVE`. The result is a falsifying protocol observation, not a reason
to relax the threshold; provider calls, real assessment data, independent
review, mechanistic understanding, alignment validation, and production
readiness remain unclaimed.

The separate preregistered v2 protocol adds a tie-aware random-control
statistic without changing the v1 result:

```sh
.venv/bin/python opt_causal_prediction_v2.py \
  --model-dir $AHCL_CUSTODY_ROOT/model/opt-125m \
  --model-id opt-125m
```

The v2 source was committed at `6f7682a` before execution. It uses the same
8-visible/64-held-out probe contract, three channels and three heads per
layer, and locked minimum `0.8`; exact ties receive half credit and the
strict random-control result remains in the report. The Gman CPU-8 execution
completed with `exit_code: 0` and status
`OPT_CAUSAL_PREDICTION_AND_RESCUE_V2_PASS`. The MLP tie-aware win rate was
`0.9982638889` (574 strict wins, 2 ties; 95% CI `[0.99565972, 1.0]`). The
attention tie-aware win rate was `0.8333333333` (384 strict wins, 192 ties,
0 losses; 95% CI `[0.81336806, 0.85329861]`), while the retained strict
attention rate was `0.6666666667` (95% CI `[0.62673611, 0.70486111]`).
Activation patching passed 576/576 and rescue passed 192/192. The mission
record is [theseus-opt-causal-v2-gman-20260915](https://autoresearch.sfcompute.com/missions/theseus-opt-causal-v2-gman-20260915).

This is still local synthetic probe evidence only. The pinned model weights
were loaded, but no provider endpoint was called, no real assessment data was
consumed, and no independent review was run. It does not establish
mechanistic understanding, architecture-general proof, alignment validation,
or production readiness. The v2 claim ceiling is
`LOCAL_OPT_CAUSAL_PREDICTION_AND_RESCUE_V2_OBSERVATION_ONLY`.

Run the independent local reference, identity criterion, causal atlas, and
Jacobian-lens proxy with:

```sh
python reference_check.py --model-dir /path/to/caller-owned/gpt2
```

The accepted checkpoint IDs and exact artifact hashes are pinned in
[checkpoint_manifests.json](checkpoint_manifests.json). To run the same
qualification protocol on both locally custodied checkpoints:

```sh
python replication_check.py \
  --gpt2-dir /path/to/caller-owned/gpt2 \
  --distilgpt2-dir /path/to/caller-owned/distilgpt2 \
  --opt-dir $AHCL_CUSTODY_ROOT/model/opt-125m \
  --smollm-dir $AHCL_CUSTODY_ROOT/model/smollm-135m \
  --smollm2-dir $AHCL_CUSTODY_ROOT/model/smollm2-135m
```

With the optional checkpoint directories, this cross-checks the NumPy logits against
PyTorch/Transformers on 72 fixed prompts for the GPT-2-family checkpoints,
OPT-125M, and, when supplied, SmolLM-135M. It then tests function-preserving
MLP-channel and attention-family permutations. A symmetry case is accepted
only when raw component digests differ, blind canonical digests match, the
inferred mapping matches the exact transformed components, visible and
held-out behavior matches, and the aligned intervention matches. Ablation,
pruning, and replacement retain distinct canonical digests and are treated as
non-symmetry controls.

For each checkpoint, the causal atlas covers three MLP channels and three
attention components at every layer, with deterministic 95% bootstrap
intervals. The SmolLM atlas has 180 entries (three gated-MLP channels and
three GQA groups at each of 30 layers).
The local Jacobian-lens proxy fits an average derivative from eight visible
probes to four fixed target-token logits, then evaluates 64 held-out probes in
post-activation MLP space. It is an operational diagnostic for this
checkpoint, not a full J-space implementation, mechanistic proof, or
consciousness claim.

The cross-check covers the pinned GPT-2, distilgpt2, and OPT-125M checkpoints
when `--opt-dir` is supplied, and adds the pinned SmolLM-135M topology when
`--smollm-dir` is supplied. `--smollm2-dir` adds a second Llama-family model
instance with its explicitly fresh probe contract. All selected checkpoints
must pass the complete local protocol. A run including SmolLM reports
`CROSS_TOPOLOGY_REPLICATION_PASS`; a run including SmolLM2 reports
`CROSS_TOPOLOGY_FRESH_HOLDOUT_REPLICATION_PASS` and separately verifies the
shared legacy contract plus the fresh SmolLM2 contract.

The five-checkpoint topology replay passed. GPT-2 passed 48/48 symmetry cases,
distilgpt2 passed 24/24, OPT-125M passed 48/48, SmolLM-135M passed 120/120,
and SmolLM2-135M passed 120/120; every blind mapping was recovered. Their
control counts were 14, 8, 14, 32, and 32 respectively, with canonical digest
differences in every control. Every checkpoint passed its local reference and
Jacobian-proxy gates. The four checkpoints on the shared suite used digest
`ef4e0b6ff2c8358f1e9350b06afae15c11d87b6bddc83e79cd6d14901f1035bf` and
the fresh SmolLM2 suite used digest
`8b90443749e884dbd83a0f2e983c5445a1dd6439e8d79f9a5a13fcb88d1bd52b`.
The aggregate reported `CROSS_TOPOLOGY_FRESH_HOLDOUT_REPLICATION_PASS`.

This is local cross-topology replication evidence for the tested
transformations. It is not architecture-general proof, mechanistic
understanding, alignment validation, or production readiness.

Run the fresh structural replacement/loss protocol across OPT-125M and
SmolLM2-135M with three seeds per architecture:

```sh
.venv/bin/python replacement_loss_experiment.py \
  --opt-dir $AHCL_CUSTODY_ROOT/model/opt-125m \
  --smollm2-dir $AHCL_CUSTODY_ROOT/model/smollm2-135m \
  --report $THESEUS_CUSTODY-replacement-loss-v1/reports/replacement-loss-v2.json
```

The `theseus-replacement-loss-v2` treatment removes a visible-selected trained
MLP component and replaces it with two newly materialized child components. It
is not a permutation or independent retraining. The fresh 8-visible/64-held-out
protocol passed all six treatments: every held-out top-1 result was `1.0` with
mean relative logit error `0.0`; canonical component digests changed, the old
component disappeared, and both-child ablations matched the original. All
wrong-source and seeded-random controls failed held-out preservation. The
aggregate status was `LOCAL_TRAINED_MODEL_STRUCTURAL_REPLACEMENT_PASS` with
claim ceiling `LOCAL_TRAINED_MODEL_STRUCTURAL_REPLACEMENT_OBSERVATION_ONLY`.
See [the replacement/loss protocol](docs/replacement-loss-protocol.md) for
the locked gates and limitations.

This v2 result is an `INTERNAL_PASS` for local structural replacement only:
the replacement is algebraically constructed, so it does not establish
independent learning or mechanistic understanding. State slice for the
following theorem and protocol work: `theseus-lab-sandbox-v1`. The additive-module
theorem was checked across OPT/ReLU, GPT-2/GELU, Llama/SiLU-gated, T5
encoder-decoder, and Mamba state-space checkpoints in two clean local
runtimes. The aggregate reached `REPRODUCED_LOCAL`, followed by
`SEGREGATED_INTERNAL_REVIEW`; this covers only the finite tested class. See
[the theorem](docs/additive-module-theorem.md) and
[the v1 results report](docs/additive-module-validity-v1-results.md). The next
mechanistic protocol is
[replacement-loss-v3](docs/replacement-loss-v3.md), which requires an
independently trained replacement and held-out causal predictions committed
before outcome evaluation. V3 remains a preregistration specification and has
not been executed or frozen. None of these results justify “all
architectures,” consciousness, alignment, independent science, or production
readiness.

The formal identity criterion is:

```text
raw digest differs + canonical digest matches + behavior matches
+ aligned interventions match
```

The stronger falsification gate additionally requires blind mapping recovery;
its controls must produce canonical digest differences. The forward path uses
float32 for broad coverage and float64 only for the intervention comparison,
so the intervention threshold remains fixed rather than being relaxed for
accumulated roundoff.

Run the full demonstration:

```sh
python3 theseus_lab.py --demo
```

Run the custody-aware adapter regression tests in the pinned runtime:

```sh
.venv/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

The tests exercise the public OPT and SmolLM adapters against local
Transformers plus their function-preserving channel/group permutations. They
skip when the caller-owned custody bundle or pinned runtime is unavailable.

Run an interactive session:

```sh
python3 theseus_lab.py
```

Commands are `show`, `preserve OLD`, `replace OLD NEW`, `split OLD NEW_A
NEW_B`, `probe`, `reset`, `help`, and `quit`.

The durable question and current verdict live in [NOTES.md](NOTES.md).

## Research report, testing, and project policy

See [RESEARCH.md](RESEARCH.md) for the findings across all recorded runs. Run
`PYTHON=.venv/bin/python scripts/run_public_tests.sh` after installing
`requirements.txt`. Report vulnerabilities per [SECURITY.md](SECURITY.md);
see [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under [Apache-2.0](LICENSE).
Model weights and assessment corpora are not included and keep their own
licenses.

---
<!-- readme-v2:footer -->
## Governance, claims, and non-claims

Work proceeds under named state slices recorded in [AGENTS.md](AGENTS.md). The claim ceiling stated above is a hard limit: a local pass is not independent verification, a benchmark result, a proof of the whole system, or a statement about production readiness. See "What this does not establish" in [RESEARCH.md](RESEARCH.md).

## Security, contributing, license

- Report vulnerabilities per [SECURITY.md](SECURITY.md).
- Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
- Licensed under [Apache-2.0](LICENSE); see [NOTICE](NOTICE).
