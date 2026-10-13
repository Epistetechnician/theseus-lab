# Additive-module validity v1 results

State slice: `theseus-lab-sandbox-v1`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

Protocol: `theseus-additive-module-validity-v1`.

## Result

**`REPRODUCED_LOCAL` across the five predeclared local checkpoint strata,
followed by `SEGREGATED_INTERNAL_REVIEW`.** The result supports the exact
structural split theorem for additive components and the five tested local
checkpoint adapters. It does not establish validity for every architecture.

The formal claim is

```text
y = Σᵢ fᵢ(x)vᵢ
fⱼ(x)vⱼ → fⱼ(x)(αvⱼ) + fⱼ(x)((1−α)vⱼ)
```

for scalar `fⱼ(x)`, vector `vⱼ`, and `α ∈ ℝ`. This is an identity over exact
real arithmetic. The model checks use float32 and therefore report numerical
tolerances rather than exact bitwise equality.

## Frozen identity and coverage

- Packet digest: `9e366d52e658114ad3e8443e55ac8a2fb68cd48cf148de1ae3f8b74047feedcd`.
- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- Receipt-bound worktree status digest: `f17698a0a92c4741bfe3dcfe92b074b2e1f147657e30d761b7fe7bdf13c38400`.
- Checkpoint identities were selected after protocol freeze and run from
  owner-controlled custody, outside this repository.
- Primary and clean replica used Python 3.12.13, PyTorch 2.6.0, Transformers
  4.57.3, NumPy 2.4.6, CPU float32, and the same complete dependency digest.
  Their runtime identity digests differ.

| Family | Frozen local checkpoint | Result |
| --- | --- | --- |
| OPT/ReLU | `facebook/opt-1.3b@3f5c25d0bc631cb57ac65913f76e22c2dfb61d62` | Pass |
| GPT-2/GELU | `openai-community/gpt2-large@32b71b12589c2f8d625668d2335a01cac3249519` | Pass |
| Llama/SiLU-gated | `TinyLlama/TinyLlama-1.1B-Chat-v1.0@fe8a4ea1ffedaf415f4da2f062534de366a451e6` | Pass |
| Encoder-decoder | `google-t5/t5-base@a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1` | Pass |
| State-space | `state-spaces/mamba-370m-hf@b519127f5bfaaa1c27dd938dad051ec360972b23` | Pass |

Across all five cases, the local split error was `0`, full-model float32
relative error stayed below `1e-5`, and primary/replica output digests matched.
The worst cross-implementation relative error was `5.84e-6` (OPT). Top-1
agreement was `1.0` in every case; this is diagnostic only. The Mamba case used
the library reference fallback because accelerator kernels were unavailable.

## Assurance, custody, and limitations

- Assurance: `INTERNAL_PASS` for each local run; aggregate reproduction:
  `REPRODUCED_LOCAL`; separate blind source/receipt review:
  `SEGREGATED_INTERNAL_REVIEW`.
- Reviewer status: a separate internal reviewer verified the frozen packet,
  exact source and checkpoint digests, receipts, runtime distinction, and
  matching outputs. No external reviewer decision or independent authority
  was used; this is not `INDEPENDENT_ACCEPT`.
- Reproducibility: two clean environments on the same host and toolchain. This
  is a clean local reproduction, not an independent host replication.
- Data and artifact custody: model weights, prompt fixtures, and aggregate
  execution receipts remain in owner-controlled storage outside the
  repository. No provider was called and no paid execution occurred. Energy
  was unavailable, not measured as zero.
- Coverage limits: five finite architecture strata, four held-out prompts per
  checkpoint, one checkpoint per stratum, and float32 full-model comparisons.
  The theorem is exact; these finite adapter checks do not prove all
  architectures or future implementations.
- Runner limitations found in review: its checkpoint loader does not itself
  compare the recomputed artifact-tree digest to the owner manifest, and its
  comparator records unequal output digests without making them blockers. The
  supplied external wrapper independently checked artifact digests before
  runs, and the observed comparison receipts have matching output digests. The
  frozen v1 packet was not edited to address these findings.

## Claim ceiling and next gate

Claim ceiling:
`LOCAL_EXACT_STRUCTURAL_SPLIT_FOR_ADDITIVE_COMPONENTS_ACROSS_FIVE_FROZEN_LOCAL_CHECKPOINTS`.

Nonclaims: all-architecture validity, independent learning, mechanistic
understanding, consciousness, alignment or human equivalence, independent
science, and production readiness.

The separately specified [`replacement-loss-v3`](replacement-loss-v3.md)
protocol for independent replacement training and blind causal prediction was
subsequently frozen and run across its three predeclared families. Its
digest-referenced aggregate record is in
[`replacement-loss-v3-results.md`](replacement-loss-v3-results.md): the
procedure reproduced locally, while the scientific outcome was `FAIL` (6/9
cases passed). Independent science still requires a frozen digest-bound packet
and an external reviewer with independent authority. The human alignment
packet remains `BLINDED_CALIBRATION_PACKET_READY`; production still requires
its separate provider, security, reliability, rollback, monitoring, and
operational authorization gates.
