# Replacement-loss V14 results: paired affine versus GELU

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v14-paired-activation-ablation-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Verdict

- Scientific outcome: `FAIL`.
- Primary execution assurance: `INTERNAL_PASS`.
- Same-owner, second-runtime assurance: `REPRODUCED_LOCAL`.
- Reviewer status: `NOT_SUBMITTED`.
- Blind status: `NOT_BLIND_SAME_OWNER_REPRODUCTION`.

The held-out causal-effect comparison favored GELU over the matched affine
replacement: mean seed-clustered difference `+0.122452`, with lower 95% bound
`+0.052697` against the predeclared `> 0.05` gate. That comparative gate
passed. The full scientific gate did not: both candidates passed retention in
`0/12` seed/layer cells each. Activation rescue passed in `12/12` cells for
each candidate, and the zero-intervention no-op control passed in all cells.
Because retention was required for both candidates, the preregistered overall
outcome is `FAIL` despite the positive relative comparison.

This result says the GELU candidate better predicted held-out intervention
effects than the matched affine candidate under this one setup, but neither
replacement met the required retained-behavior gate. Activation rescue and
the zero control support those checks only; they do not overturn the retention
failure. The changed replacement mechanism and failed retention therefore do
not demonstrate successful behavioral continuity through replacement.

## Frozen identity and target

- Repository base revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- Protocol: `theseus-replacement-loss-v14-paired-activation-ablation`.
- Freeze digest: `e6f9852d68f69f77d30191fb1f11b1fbc50d661c8aa670973e23f676fa8580b9`.
- Packet digest: `3e6db084afbd7d9cfd2d072214d916886d06a85b9ff47471dd1452257c51485f`.
- Target: `grushaprasad/opt125m-rootp-surprisal-curriculum@8af2a2926bc772599ccb3b4e7ad7b40b35259a63`.
- Target tree digest: `6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466`.
- Target-selection digest: `020aa9b9f74d897d38d960807c4db91af5fe4fb17a1da1fd61afb4a02c7f970f`.
- Selected OPT `fc2` layers: 0 and 7, fixed from previous causal-effect failures.
- Candidate set: six synthetic seeds, two layers, two methods; 24 candidates.
- Stable candidate content identity: `e5ec0d9439bae3da3937d524a5c4d801e81163d51cc1316943387ac3a20a445e`.

The paired candidates were `Linear-Identity-Linear` and `Linear-GELU-Linear`,
each with 1,967,360 parameters. Initialization tensors, adaptation captures,
batch orders, update count, optimizer, and objective were matched. Each method
received 200 AdamW updates for every seed/layer pair. This reused one OPT/ReLU
checkpoint; it is not fresh-checkpoint or architecture-held-out evidence.

## Reproduction evidence

The second runtime reproduced the scientific-result digest
`b4d1daa20cd5e04f1c78ea153f430ebfe731bc9d1e749e4df2a2510d0b1c820e`.
All seven compared artifact groups were byte-identical:

| Artifact group | Files | Match |
|---|---:|---|
| Adaptation captures | 12 | Yes |
| Candidates | 24 | Yes |
| Challenge entropy | 6 | Yes |
| Challenges and interventions | 18 | Yes |
| Fixtures | 7 | Yes |
| Outcomes | 12 | Yes |
| Predictions | 60 | Yes |

Both runs used Python 3.12.13, PyTorch 2.6.0, Transformers 4.57.3, and the
same dependency-lock digest
`8b9e94033770b3484e02cd917f8712e7bedf4b8cbd134929157b92d337d1e938` on the
same macOS ARM64 host under the same owner, but with distinct Python runtimes.
The candidate-freeze envelope digests differ because each runtime receipt
includes its Python executable path; the candidate weights, fit receipts,
adaptation bytes, and stable content identity match exactly. The comparator
reports `REPRODUCED_LOCAL`, not a blind or independent replication.

| Run | Prepare | Predict | Score | Total | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Primary | 2,369.27 s | 697.52 s | 203.58 s | 3,270.37 s | 2,829,320,192 bytes |
| Second runtime | 2,495.67 s | 694.53 s | 193.14 s | 3,383.33 s | 3,228,237,824 bytes |

Both runs completed under the frozen four-hour and 16-GiB limits. No paid,
provider, cloud, remote-worker, or GMan execution was used. Monetary cost is
`NOT_APPLICABLE`; energy is `NOT_MEASURED`.

## Post-run failure localization

This read-only diagnostic summarizes the frozen aggregate outcome report and
the frozen scoring predicate. It does not change the protocol, thresholds,
score, or packet identity, and it does not copy prompts or row-level outputs
into the repository.

The paired comparison is relative: mean held-out effect cosine across the 36
nested seed/layer/stratum cells was `-0.0488` for affine and `+0.0737` for
GELU, a difference consistent with the reported `+0.122452`. The absolute
retention gate is much stricter. Neither candidate passed any of its 12
seed/layer retention rows. Across the 36 nested strata per candidate:

- `0/36` causal-effect cells passed all frozen direction, cosine, and error
  thresholds.
- `0/36` behavior cells passed both continuation cross-entropy and normalized
  logit-error thresholds.
- Activation reconstruction passed both frozen thresholds in `1/36` affine
  cells and `0/36` GELU cells. Mean normalized reconstruction error was
  `0.4251` for affine and `0.2958` for GELU, against a maximum of `0.20`.

This explains why the comparative GELU gate passed while overall retention
failed: GELU was better than the affine candidate on average, but both were
far below the preregistered absolute causal-effect and behavior criteria.
The aggregate data localizes the failure to absolute replacement fidelity;
it does not determine whether rank capacity, optimization, objective balance,
or fixture distribution caused that failure.

The reported activation-rescue check is specifically an oracle restoration
control. Its forward hook replaces the candidate module's output with the
teacher's captured original activation, then verifies next-token logits and
continuation loss. The zero error is expected from restoring the source
activation. This checks the activation-capture, hook, and downstream execution
path; it does not show that a learned candidate restores behavior.

## Custody, claim ceiling, and limitations

Synthetic prompts, checkpoint bytes, captures, candidate tensors, challenge
entropy, predictions, and row-level outcomes remain in external owner custody
at:

- `/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v14-paired-primary`
- `/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v14-paired-replica`

The repository contains the frozen V14 protocol and aggregate results; no
V14 prompts or row-level outcomes were copied into it. Reviewer status is
`NOT_SUBMITTED`; same-owner local reproduction is not independent scientific
acceptance.

Claim ceiling:
`LOCAL_GELU_ACTIVATION_EFFECT_ON_MATCHED_LOW_RANK_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT_AND_SIX_SYNTHETIC_SEEDS`.
V14 does not establish architecture-general validity, successful cumulative
replacement, a novel proof of the Ship of Theseus, consciousness, alignment,
human equivalence, or production readiness. V11 remains `FAIL` with its
original comparator `INCONCLUSIVE`; V13 remains `FAIL` with its prior
`REPRODUCED_LOCAL` status.

## Next gate

The next empirical step should be a new frozen diagnostic that separates
rank-capacity limits from training/objective failure, while treating teacher
activation restoration only as an oracle control. Do not tune on V14's held-
out prompts or outcomes. Any new method, target, threshold, or fixture
requires a new packet. Independent science still requires a separate blind
replicator and an authorized independent reviewer; alignment, consciousness,
and production remain separate gates.
