# Replacement-loss-v3 results: independent replacement and blind prediction

State slice: `theseus-lab-sandbox-v1`.

Protocol: `theseus-replacement-loss-v3`.

## Outcome

The frozen nine-case experiment outcome is **`FAIL`**: six cases passed and
three failed. All OPT/ReLU and Llama/SiLU-gated cases passed. All three
GPT-2/GELU cases failed activation-rescue thresholds; GPT-2 seed 0 also failed
held-out behavior thresholds. The thresholds and frozen v3 packet were not
changed.

| Family | Frozen public checkpoint | Cases passed | Cases failed |
| --- | --- | ---: | ---: |
| OPT/ReLU | `facebook/opt-350m@08ab08cc4b72ff5593870b5d527cf4230323703c` | 3/3 | 0/3 |
| GPT-2/GELU | `openai-community/gpt2-medium@6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | 0/3 | 3/3 |
| Llama/SiLU-gated | `HuggingFaceTB/SmolLM-360M@59f7ef243ee09a72cbc14cb054393a3e3b771d41` | 3/3 | 0/3 |

The checkpoint manifest records selection after the protocol freeze on
2026-09-24. The primary report labels its overall assurance `INCONCLUSIVE`
while retaining the measured experiment outcome `FAIL`. The subsequent
reconciliation establishes `REPRODUCED_LOCAL` for the frozen procedure and
outputs; reproduction does not turn the experiment into a pass.

## Frozen identities and reproduction

- Frozen packet SHA-256: `d9cf01d6caa2ac6145c0b4a7e5ddb9c2b105978993d77c1d58b5d0ad4dfc9b83`.
- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- Source worktree-status SHA-256: `14d30eca74ba02c778fbfcca065db2657bdca54b3e200b344e9a840b4da6a8c9`.
- Primary aggregate report SHA-256: `5a4cd64f7e0bb5a38f08cce6f99541625fcb34cf4a4ed18f6303c3e3413b9ea7`.
- Reconciliation file SHA-256: `72661884d87d2d559c0479edb924ec4e7ecba1cb64d4214c70719dc55d35081e`.
- Replication report SHA-256: `8c66320c9936aa563697856c846da159b529468d1c6b1b661bf66284fb82e803`.
- Replica evaluation report SHA-256: `cf2e8a6c3494f3920eef36bb4dd77ee95c540162673af11a63eb3f22cc45f1ce`.

The separate clean local runtime reconstructed candidates and matched source,
dependency, fixture, prediction, and outcome digests for all nine cases. It
was a same-owner second environment, not an external replication or
independent review. Reviewer status is `EXTERNAL_REVIEW_UNAVAILABLE`; no
external review task was submitted.

The primary run used Python 3.12.13, PyTorch 2.6.0, Transformers 4.57.3,
NumPy 2.4.6, Tokenizers 0.22.2, and Safetensors 0.8.0 on local CPU. Its peak
RSS was about 3.52 GiB and process CPU time was about 2,359.6 seconds.

## Recorded diagnostic limitation

The mapping-aligned permutation equivalence diagnostic exceeded its declared
`1e-4` absolute-error tolerance in four cases: GPT-2 seeds 0–2 had maximum
logit errors of `0.0001907`, `0.0001450`, and `0.0001678`; Llama seed 2 had a
module-boundary error of `0.0001240`. These values are retained in the frozen
reports. The per-case blocker lists omit this diagnostic, so those six
case-level `PASS` labels do not include the equivalence check. The frozen
runner separately includes both equivalence errors in its aggregate pass
condition; the four misses therefore also fail that aggregate gate. The
overall experiment remains `FAIL` from the three GPT-2 case failures even
apart from the equivalence misses.

The frozen protocol source also retains pre-run wording in its “Current
status” section saying v3 remains unfrozen. That wording is stale. The packet
was frozen before checkpoint selection and executed; this results document
records the erratum without changing any frozen packet member.

## Custody and claim ceiling

Prompts, model assets, captures, candidate weights, predictions, outcomes,
and row-level diagnostics remain in external owner custody. This repository
contains aggregate results and digests only.

Claim ceiling:
`LOCAL_INDEPENDENT_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION_ONLY` for the
three frozen checkpoints, synthetic prompt generator, and nine seed cases.
The result does not establish all-architecture validity, general mechanistic
understanding, independent science, consciousness, alignment, or production
readiness. The next blocked authority gate for an independent-science claim is
the configured external reviewer; its status remains
`EXTERNAL_REVIEW_UNAVAILABLE`.
