# Replacement-loss-v7 results

State slice: `theseus-lab-sandbox-v1`.

Protocol: `theseus-replacement-loss-v7-downstream-aware-shape-safe`.

## Registered result

The registered v7 primary outcome is **`FAIL`**, and the exact frozen run
reproduced in a separate local runtime: **`REPRODUCED_LOCAL`**. This is a
reproduced miss for the behavior-aware AdamW replacement procedure, not a
successful primary treatment.

The v7 packet SHA-256 is
`39737cbef36a9eb84ed23a236de3f6c5de7c4a4168715f06d11203d6d5918918`.
The post-freeze checkpoint manifest SHA-256 is
`935364235c239327aed5ed6094935a046dfaeddcd8d78e8f75f985dac1cb4367`.
The source revision is `202f459d6d89df9a038d78d58e0af825d287fcd8`; the
run-bound worktree status digest is
`a44ba41fbc745e9c5565219146c5788e22989086c1f5f3a2a37a154ae6bf658f`.

| Family | Frozen checkpoint and revision | Checkpoint tree SHA-256 |
| --- | --- | --- |
| OPT/ReLU | `babylm/opt-125m-strict-small-2023@7b2eca65955015893bafc76748a6f60f61e588cc` | `46a018922d14d1443cf65610e17559c068b201bdaeb4b0a3ef1ac6ea2d6634d9` |
| GPT-2/GELU | `erwanf/gpt2-mini@f12cc7ee54aa4d3e6366597f72ff7acb39b0ab3a` | `5abd4c8dcf9aafcb2849dbf317c410d13476d43bf6309d785c9a5f24efd6b02b` |
| Llama/SiLU-gated | `pgryko/babyllama-10m@5774ee24a25a5a552235a8f67f2635e3e76dbae4` | `7c684b83ac2149411ba45e6f940a50111a19e32c359ffa1394de41d2011e86b0` |

All nine predeclared cases completed, including the 60-prompt held-out
challenge per case. Candidate and prediction artifacts were frozen before
challenge entropy was drawn; predictions were sealed before original-model
outcomes were opened. The run had no resource-cap blockers. The primary
replacement failed causal thresholds in all nine cases. It also missed
behavior and activation-rescue thresholds for OPT and GPT-2; Llama passed
those two gates. Mapping-aligned permutation equivalence passed in all nine
cases, which validates that positive-control path but does not rescue the
learned treatment.

The incremental metric also failed. Mean improvement over the output-only
ridge comparator was `-0.2521358768291672`; its clustered 95% lower bound was
`-0.379704461598276`. The required thresholds were mean improvement at least
`0.005` and a positive lower bound.

## Predeclared comparator diagnostic

The frozen packet included an output-only ridge comparator fitted from the
disjoint adaptation captures, without copying target projection weights. The
v7 adapter assigned that comparator to the evaluator's `wrong_module` slot.
The primary outcome above remains `FAIL`; the comparator did not replace the
registered AdamW treatment or change the v7 gate.

As a post-run diagnostic of this predeclared comparator, its existing sealed
prediction and outcome arrays were evaluated with the same causal and
behavior thresholds. The activation-rescue statistic was calculated from
the stored comparator boundary outputs and original boundary activations;
that activation summary was not a registered v7 primary gate. Results were
consistent across all nine cases:

- Held-out causal effects: direction accuracy was `1.0` in every case; the
  worst median cosine was `0.999757`, and the largest median normalized logit
  effect error was below `9.6e-7`. Each of the three strata passed in every
  case.
- Held-out behavior: top-one agreement was `1.0` in every case; maximum
  normalized next-logit error was at most `4.1e-6`; continuation cross-entropy
  change stayed within `3.3e-6` nats per token of zero.
- Derived activation diagnostic: all nine cases and all three strata had
  median cosine `1.0`; the largest median normalized activation error was
  below `2.7e-6`.

The second runtime reproduced all nine candidate and adaptation-capture
digests and all nine prediction and outcome array digests, so it reproduced
the comparator outputs as well. This is a strong local result for an
independently fitted affine substitute at the selected MLP output projections
of these exact checkpoints. Because the comparator was secondary and the
activation summary was derived after the registered run, this diagnostic is
not promoted to a v7 primary pass or a general mechanistic claim. The next
confirmatory packet should preregister this ridge method as the treatment,
retain the causal, behavior, and activation gates, and select fresh
checkpoints only after that packet is frozen.

## Reproduction and custody

- V7 primary aggregate report SHA-256:
  `229f3cbf1c6610a7b60a1c8874438df116c90703cde884585b0a9334e2536c79`.
- V7 replication report SHA-256:
  `90a89eb692319c2626e19d1d876ccf5bfde1f8c946ce287275d1b764937460db`.
- The replication report records matching candidate reconstruction,
  prediction and outcome arrays, dependencies, distinct runtime identity, and
  source identity. The candidate reconstruction had nine cases and no
  blockers.
- Both runtimes used Python 3.12.13, PyTorch 2.6.0, Transformers 4.57.3,
  NumPy 2.4.6, Safetensors 0.8.0, and Tokenizers 0.22.2 on CPU. Runtime
  identities were distinct. Peak RSS was about 3.52 GB; total protocol-stage
  wall time was about 532 seconds. Energy and monetary cost were not measured.
- Model weights, prompts, captures, candidates, predictions, outcomes, and
  row-level diagnostics remain in owner-controlled external custody. This
  repository contains only protocol material and aggregate-safe results.

The v7 runner's primary report field `replication_report_sha256` contains
`05dfd9a8a69b5300c836cf4109ade823ec2485ac82e40ce204b79d6441865f17`, the
digest of its intermediate v6-compatible reconciliation report. The final v7
replication report itself has the SHA-256 listed above. This digest-linkage
limitation does not affect the underlying matching checks, but the frozen v7
packet was not edited; a future runner version should bind the final report
artifact directly.

## Assurance and claim ceiling

- Assurance: `REPRODUCED_LOCAL` for the exact v7 procedure and its `FAIL`
  outcome. The comparator diagnostic is secondary analysis of the same
  reproduced artifacts.
- Reviewer status: `EXTERNAL_REVIEW_UNAVAILABLE`; no review task was
  submitted. This same-owner second runtime is not independent science or
  `INDEPENDENT_ACCEPT`.
- Claim ceiling: local evidence about one independently fitted affine
  projection on one fresh checkpoint in each of three architecture families.
  It does not establish mechanism identity for an entire model, every
  architecture, consciousness, alignment, or production readiness.
- Next authority gate: a frozen digest-bound packet must be reviewed by the
  configured external reviewer with independent authority before any
  independent-science claim. Alignment and production retain their separate
  gates.
