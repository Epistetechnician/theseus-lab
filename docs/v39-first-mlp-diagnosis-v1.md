# V39 first-MLP causal diagnosis v1

2026-10-07 (client date), revision `202f459d6d89df9a038d78d58e0af825d287fcd8`
plus frozen source. State slices `theseus-lab-sandbox-v1`,
`theseus-v39-first-mlp-causal-diagnosis-v1`,
`global-experiment-assurance-lifecycle-v1`.

## Question and boundary

V39 remains a reproduced FAIL, including 0/12 first-stage causal cells.
Diagnose whether its first replacement has a large local derivative/finite
response error, and measure how that error and factual/prefix shifts appear
in the decoder response. This is an adaptation-only diagnostic, not another
replacement training run or unseen-retention claim. No prior held-out
fixtures, predictions or oracle arrays are opened. The existing terminal
scalar parent report is read for provenance and motivates this follow-up.

Reuse exactly the V39 guide, first MLP site 0 and six frozen block_1 students
(three seeds 36101-36103, JVP/endpoint arms). No weight update, initialization
of a new scientific candidate, checkpoint selection, threshold change,
provider call, paid/cloud/gman execution, publication or review submission.
All raw assets/arrays/text stay on PrimaryED, runtimes on Backup ED.
Verify both mounted volumes, parent source/terminal digests and the reused
guide/candidate/capture bytes. Replay guide adaptation inputs exactly.

## Predeclared selection and measurements

Before measurements, freeze code, selectors, metrics, thresholds and this
document. Select eight of the 128 adaptation prompts per seed by a protocol-
bound torch.randperm, then sort indices. Selection does not inspect values or
errors. JVP and endpoint arms share prompts, donor alignment and patches.
Use three positions: first, floor((L-1)/2) prefix, final. Pair each donor token
at floor(i*(Ld-1)/(La-1)), matching V39's rule. Verify full adaptation and
donor input captures against retained native arrays.

At each of 144 points (3 seeds × 2 arms × 8 prompts × 3 positions), retain
teacher/student values and full output-by-input 768×768 Jacobians. Compare
each Jacobian with four independently computed JVPs; relative discrepancy
must be <=1e-5, otherwise INCONCLUSIVE. Measure value and Jacobian Frobenius
nRMSE with the declared 1e-8 mean-reference floor.

Measure eight finite patches per point: 1, 2, 4 or 8 coordinates; donor
interchange scale 1 or extrapolation 1.4. Protocol-bound coordinates are new
diagnostic directions on existing adaptation inputs, not a disjoint held-out
set. Retain all 1,152 local delta pairs. Report resolved fraction (reference
norm >=1e-4), median/p90 relative L2 error and median cosine. Zero/weak donor
responses remain unresolved; shared initial tokens can yield degenerate first-
position patches. Do not replace those points or count them as successful
effect predictions. Analytic zero-
delta controls must have relative error 1 on resolved references.

Support rule: `LOCAL_DERIVATIVE_MISMATCH_SUPPORTED` only if every JVP seed's
final-token cell has Jacobian nRMSE >.20, resolved finite-delta fraction >=.90,
and median finite-delta relative error >.20. Otherwise report
`MIXED_OR_LOCAL_FIT_REQUIRES_FURTHER_DIAGNOSIS`. These diagnostic thresholds
do not alter V39's frozen scientific thresholds or classify a new method.
No drop, reselect, retune or early stopping based on the observed scores.

## Decoder response accounting

At final tokens only, evaluate 384 paired decoder interventions. At this first
MLP, candidate/guide inputs precede the replacement, so replay the same native
input intervention. All later guide layers are frozen; site 2 remains original.

Let G use original module-0 prefix outputs, C use student prefix outputs,
g/s be teacher/student final outputs, and dg/ds their finite local changes.
Record effects:

- Teacher: Et = G(g+dg)-G(g).
- Actual candidate: Ea = C(s+ds)-C(s).
- Student final outputs in guide prefix context: Eg = G(s+ds)-G(s).
- Student delta anchored at guide factual output: Eh = G(g+ds)-G(g).

Account exactly for the vector error:
Ea-Et = (Ea-Eg) + (Eg-Eh) + (Eh-Et).
The terms are named prefix context, factual anchor and local delta transport,
respectively. Check the identity at absolute tolerance 1e-12 using float64
differences of native float32 decoder logits. Retain all four response vectors
and native local deltas. Report effect error, component relative RMS and error-
projection shares. Component norms do not add; shares may be negative or exceed
1. This ordered accounting is path-dependent and is not a unique causal
attribution. It does not establish that changing a loss would repair errors.

Actual no-op reruns reinstall the original module for the count-1/scale-1
intervention at every sampled prompt and arm (48 pairs). Outputs must match
the guide oracle exactly. Nine hermetic contracts precede freeze, including
full-Jacobian orientation/finite differences, cancellation-aware accounting
and a real small GPT-2 decoder with native input/output hooks.

## Execution, replication and stop rules

Run fresh processes in the existing primary/replica isolated prefixes. Record
the exact 34-package lock, installed-file/toolchain digests, device and threads.
Both prefixes and site-package paths must differ, toolchain bytes must match
one another and V39's recorded toolchain. Explicit HF_HUB_OFFLINE and
TRANSFORMERS_OFFLINE flags are set before model loading and recorded.
Source archive, selection and raw-array digests bind the packet. Replica
recomputes every measurement without consuming primary arrays or scores, then
reconciles eight fields. A separate NumPy implementation in a fresh closeout
process checks shapes, inventory, accounting identity and all scalar formulas
(relative tolerance 1e-10, absolute 1e-12); categorical decisions and counts
must match exactly. Shared owner, host,
system libraries, selection and instrumentation are limitations; this is
REPRODUCED_LOCAL, not independent authority.

Per role: 60 minutes including initialization/verification, 6 GiB peak RSS,
2 GiB arrays/artifacts, 5 GiB free-space reserve, one deterministic CPU thread,
paid cap $0. Cooperative time/RSS/disk/mount checks, not OS quotas. Energy and
total economic cost remain unmeasured. Custody/source drift, nonfinite data,
cap exhaustion, Jacobian check failure, no-op discrepancy, error or replica
mismatch is INCONCLUSIVE. No same-root retry; preserve every terminal state.

End state: actual primary, replica, persisted-array closeout and scalar report,
or an explicit custody/runtime stop after finishing allowed engineering work.
Reviewer NOT_SUBMITTED. Claim ceiling
`ADAPTATION_FIRST_MLP_LOCAL_AND_CONTEXT_ERROR_DIAGNOSIS_ONLY`.
No new retained-behavior pass, three-family confirmation, novelty, full mapping
equivalence, universal identity, consciousness, human alignment, independent
acceptance or production authorization. Stronger science and production keep
their separate authority gates. A next fitting experiment requires a fresh
packet and fresh held-out fixtures.
