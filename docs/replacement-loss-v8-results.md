# Replacement-loss v8 results

State slice: `theseus-lab-sandbox-v1`  
Protocol: `theseus-replacement-loss-v8-ridge-primary-causal`  
Assessment date: 2026-09-25

## Result

The frozen v8 primary run passed all nine checkpoint-by-seed cases. A separate
clean local runtime reproduced candidate reconstruction, predictions, outcome
arrays, dependencies, and source identity byte-for-byte. The assurance level is
`REPRODUCED_LOCAL`. This is same-owner reproduction, not independent science
or reviewer acceptance.

The result supports a narrow claim: a newly fitted affine MLP output projection,
estimated from disjoint adaptation activations without copying target weights,
predicted held-out module and logit intervention effects, activation rescue,
and next-token behavior for the tested projection in each selected checkpoint.
It does not prove whole-model identity or Theseus continuity across arbitrary
architectures.

## Frozen target and checkpoints

- v8 packet SHA-256: `b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2`
- Checkpoint-manifest SHA-256: `a4522ceedeb3a0007970f9780a3377a75fd0468f93996fad0468043f65745e27`
- Selected model assets are in external owner custody. All three revisions
  were selected after the packet freeze, are absent from the v3-v7 exclusion
  set, loaded locally with `trust_remote_code=False`, and passed the frozen
  family/activation and 500-million-parameter preflight.

| Frozen family | Checkpoint | Revision | Selected projection layer |
| --- | --- | --- | ---: |
| OPT/ReLU | [kanishka OPT BabyLM2](https://huggingface.co/kanishka/opt-babylm2-rewritten-clean-spacy_no-num-adj-earlystop-strictest-bpe_seed-211_1e-3/tree/baa6df5efc0d9f71306d115e64da230ce496b773) | `baa6df5efc0d9f71306d115e64da230ce496b773` | 11 |
| GPT-2/GELU | [AI Forever ruGPT3 small](https://huggingface.co/ai-forever/rugpt3small_based_on_gpt2/tree/a9307e696cd3c5b7f953ff4cb19d76a4d81821d5) | `a9307e696cd3c5b7f953ff4cb19d76a4d81821d5` | 0 |
| Llama/SiLU-gated | [AMD Llama 135M](https://huggingface.co/amd/AMD-Llama-135m/tree/9a224a07b9b0210299d8c19b9e694ac56ed3ba1d) | `9a224a07b9b0210299d8c19b9e694ac56ed3ba1d` | 11 |

The OPT card documents a 98.2M-parameter model trained from scratch. The AI
Forever card documents GPT-2-compatible pretraining on 80B Russian tokens and
continuation to a 2048-token context; its Hub card does not declare a license.
The AMD card documents a 135M-parameter pretrained Llama-compatible model and
Apache-2.0 licensing. The checkpoint cards and exact revisions are recorded in
the external target-screening manifest.

## Procedure and observations

The runner selected each MLP output projection using visible prompts only. It
fit a fresh affine replacement by closed-form ridge regression (`1e-6`) on
disjoint adaptation captures, with no target parameter copy. There were three
predeclared seeds per family, nine cases total, 60 held-out challenges per case
(20 ordinary, 20 compositional, 20 adversarial), and 8,020–9,191 adaptation
token rows per case. Candidate files were frozen before challenge generation.

| Measure across all cases and strata | Worst observed | Frozen gate |
| --- | ---: | ---: |
| Causal logit-effect direction accuracy | 1.000 | ≥ 0.80 |
| Causal logit-effect median cosine | 0.95015 | ≥ 0.90 |
| Causal logit-effect median normalized error | 0.00000810 | ≤ 0.20 |
| Module-boundary effect direction accuracy | 1.000 | ≥ 0.80 |
| Module-boundary effect median cosine | 0.95578 | ≥ 0.90 |
| Activation-rescue median cosine | 1.000 | ≥ 0.90 |
| Activation-rescue median normalized error | 0.00000940 | ≤ 0.20 |
| Next-token top-1 agreement | 1.000 | diagnostic; not a standalone threshold |
| Paired next-token normalized logit error | 0.001155 | ≤ 0.05 |
| Continuation cross-entropy increase | 0.024914 nats/token | ≤ 0.05 |

All 20 interventions in every stratum were resolvable. The frozen evaluator
reported all treatment and control gates passing. Mapping-aligned permutation
direction accuracy was 1.000 in every case; no-op was 0.000, while random,
raw-index permutation, and wrong-module controls remained below the treatment
in aggregate direction accuracy. The control summary is descriptive; the
frozen bootstrap and equivalence gates determine the outcome.

## Reproduction, custody, and resources

- Primary aggregate report, final SHA-256:
  `317b75f2c448493dd07acba78f1152221bcd6ff96b642de476d2f1d67402aec6`
- Final replication report SHA-256:
  `8339b4f57ef8e3d24048e457007a1ef0be7c232da77ed2bd558937a9c0e4530d`
- The primary report's replication pointer matches the final replication-report
  bytes. The replication report records all five checks as true:
  candidate reconstruction, prediction/outcome array digests, dependencies,
  distinct clean runtime, and source identity.
- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- Pre-results-document worktree-status digest recorded by the run:
  `2048501416c9586209d1832228a007e67705aab3ec52052f2935b30cc532f6b8`.
- Runtime: Python 3.12.13, PyTorch 2.6.0, Transformers 4.57.3, NumPy 2.4.6,
  macOS 26.6.1 arm64, CPU only. Dependency-inventory SHA-256:
  `abc7576dd285caa9381f25354aa1d02165b5c0b3677d9c3a54fd47692e9e4c97`.
- Frozen limits: 7,200 seconds per case, 64,800 seconds total, and 24 GiB
  peak RSS. The longest case was 116.73 seconds; protocol-stage wall time was
  792.36 seconds; peak RSS was 3,633,561,600 bytes; no resource blockers were
  reported. Monetary cost and energy were not measured.
- Model weights, tokenizers, prompts, captures, candidates, predictions,
  outcomes, and row-level diagnostics remain in external owner custody at
  `/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v8`.

## Claim ceiling and remaining gates

The claim ceiling is
`LOCAL_INDEPENDENT_AFFINE_PROJECTION_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION`.
The packet does not establish all-architecture validity, whole-model
mechanistic identity, independent science, consciousness, alignment, or
production readiness. The additive-module theorem remains valid for its stated
mathematical class; this experiment does not extend that theorem to arbitrary
AI systems.

Reviewer status is `EXTERNAL_REVIEW_UNAVAILABLE`; no review task was submitted.
The next authority gate is a blind external replication and signed review bound
to the exact v8 packet, result, artifact manifest, claim ceiling, and nonclaims.
That submission requires explicit authorization for the frozen target digest.
Human-calibrated alignment evaluation, any consciousness research program,
and production authorization remain separate workstreams.
